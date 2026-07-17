#!/usr/bin/env python3
"""Prepare mono-first artistic-policy source rows from full-cadence clips.

Unlike the legacy authored-stereo fitter, this stage does not infer a camera
target from a right eye.  It only authenticates RGB frames and the matching
production depth-run artifacts needed by the render-feasibility selector.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import shutil
import struct
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np

SBSBENCH_DIR = Path(__file__).resolve().parent.parent / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))

import generate_artistic_depth_run as depth_run  # noqa: E402
import depth_input_color as input_color  # noqa: E402
import native_hdr_capture  # noqa: E402
from sbs_harness_contract import (  # noqa: E402
    HARNESS_SCHEMA,
    input_variant_hdr_source_kind,
    input_variant_metric_preview_encoding,
)

SOURCE_SCHEMA = 2
SOURCE_CONTRACT = "full-cadence-artistic-source-v2"
LABEL_FRAMES_SCHEMA = 1
LABEL_FRAMES_NAME = "label_frames.json"
DATASET_MANIFEST_NAME = "dataset_manifest.json"
SOURCE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
TEMPORAL_SOURCE_KINDS = {
    "mono-video",
    "authored-stereo",
    "gt-depth-flow",
    "native-hdr-video",
}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths_overlap(left: Path, right: Path):
    """Return whether two resolved roots are equal or contain one another."""
    return (
        left == right or left.is_relative_to(right) or right.is_relative_to(left)
    )


def validate_output_roots(run: Path, clips: Path, output: Path):
    """Reject destructive output placement before inspecting output contents."""
    run_root = run.resolve(strict=True)
    clips_root = clips.resolve(strict=True)
    output_root = output.resolve(strict=False)
    for source_root, label in ((run_root, "run"), (clips_root, "clips")):
        if paths_overlap(output_root, source_root):
            raise RuntimeError(
                f"output root overlaps {label} input root: "
                f"{output_root} and {source_root}"
            )


def remove_path(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def publish_staged_directory(staging: Path, destination: Path):
    """Swap a complete staged directory into place, restoring on failure."""
    backup = destination.with_name(
        f".{destination.name}.backup-{uuid.uuid4().hex}"
    )
    moved_existing = False
    try:
        if destination.exists() or destination.is_symlink():
            destination.replace(backup)
            moved_existing = True
        staging.replace(destination)
    except BaseException:
        if moved_existing and not destination.exists():
            backup.replace(destination)
        raise
    if moved_existing:
        remove_path(backup)


def run_contract(run: Path):
    depth_manifest_path = run / "depth_run_manifest.json"
    results_path = run / "results.json"
    if depth_manifest_path.is_file():
        payload = json.loads(depth_manifest_path.read_text(encoding="utf-8"))
        return {
            "kind": "depth_run_manifest",
            "sha256": sha256(depth_manifest_path),
            **{
                key: payload.get(key)
                for key in (
                    "purpose", "model", "suite_manifest_sha256",
                    "executable_sha256", "conf_sha256", "metric_sha256",
                    "policy_warp_source_sha256", "frame_count",
                    "harness_schema", "metric_preview_encoding",
                    "hdr_source_kind",
                )
            },
        }
    if results_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        return {
            "kind": "eval_results",
            "sha256": sha256(results_path),
            **{
                key: meta.get(key)
                for key in (
                    "model", "profile", "suite", "clip_set_sha1",
                    "eval_schema", "conf_sha256", "metric_sha256",
                    "policy_warp_source_sha256",
                )
            },
        }
    raise RuntimeError(f"run has no provenance manifest: {run}")


def dataset_contract(clips: Path):
    """Load one exact complete-production suite contract."""
    manifest_path = clips / DATASET_MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read dataset manifest: {manifest_path}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid dataset manifest: {manifest_path}")

    schema = payload.get("schema")
    if schema == 1 and not isinstance(schema, bool):
        production = payload.get("film_id")
        source_kind = "authored-stereo"
    elif schema == 2 and not isinstance(schema, bool):
        production = payload.get("production_id")
        source_kind = payload.get("source_kind")
        if source_kind not in TEMPORAL_SOURCE_KINDS:
            raise RuntimeError(
                f"{manifest_path}: unsupported temporal source_kind {source_kind!r}"
            )
    else:
        raise RuntimeError(f"unsupported dataset manifest: {manifest_path}")

    split = payload.get("split")
    if not isinstance(production, str) or not production.strip():
        raise RuntimeError(f"{manifest_path}: missing production identity")
    if split not in {"training", "development", "test"}:
        raise RuntimeError(f"{manifest_path}: invalid production split")
    try:
        weight = float(payload.get("global_policy_weight"))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"{manifest_path}: invalid global_policy_weight"
        ) from error
    if not math.isfinite(weight) or weight <= 0.0:
        raise RuntimeError(f"{manifest_path}: invalid global_policy_weight")

    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise RuntimeError(f"{manifest_path}: dataset has no exact sequence list")
    sequence_names = []
    for index, sequence in enumerate(sequences):
        clip = sequence.get("clip") if isinstance(sequence, dict) else None
        if (not isinstance(clip, str) or not clip or
                Path(clip).name != clip or clip in {".", ".."}):
            raise RuntimeError(
                f"{manifest_path}: sequence {index} has an unsafe clip identity"
            )
        sequence_names.append(clip)
    if len(sequence_names) != len(set(sequence_names)):
        raise RuntimeError(f"{manifest_path}: duplicate sequence identity")

    actual_sequences = {
        path.name for path in clips.iterdir()
        if path.is_dir() and (path / LABEL_FRAMES_NAME).is_file()
    }
    expected_sequences = set(sequence_names)
    if actual_sequences != expected_sequences:
        raise RuntimeError(
            "dataset manifest sequence set differs from prepared label-frame clips: "
            f"expected={sorted(expected_sequences)}, actual={sorted(actual_sequences)}"
        )

    manifest_hash = sha256(manifest_path)
    return {
        "path": str(manifest_path.resolve()),
        "sha256": manifest_hash,
        "schema": schema,
        "production_id": production,
        "split": split,
        "source_kind": source_kind,
        "global_policy_weight": weight,
        "sequence_names": sequence_names,
        "dataset": payload.get("dataset"),
        "domain": payload.get("domain", payload.get("dataset", "unknown")),
        "license": payload.get("license"),
        "source_split": payload.get("source_split"),
        "projection": payload.get("projection", "rectilinear"),
        "policy_role": payload.get("policy_role", "cinematic_training"),
    }


def read_json_object(path: Path, description: str):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid {description}: {path}")
    return payload


def file_stat_identity(path: Path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def png_dimensions(path: Path):
    """Read the exact PNG IHDR dimensions without decoding image pixels."""
    try:
        with path.open("rb") as stream:
            header = stream.read(24)
    except OSError as error:
        raise RuntimeError(f"cannot read PNG header: {path}") from error
    if (len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or
            header[12:16] != b"IHDR"):
        raise RuntimeError(f"invalid native-HDR preview PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid native-HDR preview geometry: {path}")
    return width, height


def authenticate_native_hdr_sources(clips: Path, names, depth_manifest):
    """Fully authenticate native scRGB sidecars and their preview mapping.

    The suite clip-hash manifest freezes ``frame_model_sources.json``.  That
    manifest, in turn, binds every large FP16 sidecar by SHA-256.  Source-row
    admission performs the full content verification once, then records every
    manifest, preview, and sidecar in the late-mutation stat guard.
    """
    if depth_manifest.get("clip_hash_source") != "manifest":
        raise RuntimeError(
            "native HDR source admission requires a frozen clip hash manifest"
        )
    frozen_path = clips / depth_run.clip_hashes.MANIFEST_NAME
    try:
        frozen = depth_run.clip_hashes.load_manifest(frozen_path)
    except depth_run.clip_hashes.ClipHashManifestError as error:
        raise RuntimeError(
            f"native-HDR frozen source authentication failed: {error}"
        ) from error

    result = {}
    for name in names:
        clip_root = clips / name
        try:
            authentication = native_hdr_capture.validate_clip(
                clip_root, full=True
            )
            payload, frames, manifest_path = (
                native_hdr_capture.load_manifest(clip_root)
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"{name}: native-HDR source authentication failed: {error}"
            ) from error
        if sha256(manifest_path) != authentication["manifest_sha256"]:
            raise RuntimeError(
                f"{name}: native-HDR frame manifest changed during authentication"
            )

        frozen_entry = frozen.get("clips", {}).get(name, {})
        frozen_records = frozen_entry.get("files")
        if not isinstance(frozen_records, list):
            raise RuntimeError(
                f"{name}: frozen clip identity has no native-HDR manifest record"
            )
        matching_records = [
            record for record in frozen_records
            if isinstance(record, dict) and
            record.get("path") == native_hdr_capture.MANIFEST_NAME
        ]
        if (len(matching_records) != 1 or
                matching_records[0].get("sha256") !=
                authentication["manifest_sha256"]):
            raise RuntimeError(
                f"{name}: native-HDR frame manifest is not frozen by clip identity"
            )

        previews = numeric_frame_map(clip_root, suffix=".png")
        if set(previews) != set(frames):
            raise RuntimeError(
                f"{name}: native-HDR model sources do not exactly match preview cadence"
            )
        expected_geometry = (
            int(authentication["width"]), int(authentication["height"])
        )
        previous_timestamp = None
        for frame_id, record in frames.items():
            preview_path = previews[frame_id].resolve()
            if record["preview_path"] != preview_path:
                raise RuntimeError(
                    f"{name}: native-HDR preview path differs for frame {frame_id}"
                )
            if png_dimensions(preview_path) != expected_geometry:
                raise RuntimeError(
                    f"{name}: native-HDR preview dimensions differ for frame {frame_id}"
                )
            timestamp = float(record["timestamp_seconds"])
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise RuntimeError(
                    f"{name}: native-HDR timestamps are not strictly increasing"
                )
            previous_timestamp = timestamp

        provenance = {
            key: authentication[key] for key in (
                "manifest", "manifest_sha256", "content_sha256",
                "width", "height", "frame_count", "verification",
            )
        }
        provenance.update({
            "contract": payload["contract"],
            "capture_encoding": payload["capture_encoding"],
            "preview_encoding": payload["preview_encoding"],
            "source_video": payload["source_video"],
            "conversion": payload["conversion"],
            "frozen_clip_manifest": str(frozen_path.resolve()),
            "frozen_clip_manifest_content_sha256": frozen[
                depth_run.clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
            ],
        })
        result[name] = {
            "provenance": provenance,
            "frames": frames,
            "manifest_path": manifest_path,
        }
    return result


def authenticate_source_identities(clips: Path, names, manifest):
    identities = manifest.get("source_identities")
    if not isinstance(identities, dict) or set(identities) != set(names):
        raise RuntimeError("depth-run source identity set differs from dataset clips")
    source = manifest.get("clip_hash_source")
    verification = manifest.get("clip_hash_verification")
    if source == "manifest":
        if verification not in {"stat", "full"}:
            raise RuntimeError("invalid depth-run clip-hash verification contract")
        expected_path = (clips / depth_run.clip_hashes.MANIFEST_NAME).resolve()
        value = manifest.get("clip_hash_manifest")
        if not isinstance(value, str) or Path(value).resolve() != expected_path:
            raise RuntimeError("depth-run clip-hash manifest path differs from suite")
        try:
            frozen = depth_run.clip_hashes.load_manifest(expected_path)
            current = depth_run.clip_hashes.verify_selected_clips(
                expected_path, clips, names, full=verification == "full"
            )
        except depth_run.clip_hashes.ClipHashManifestError as error:
            raise RuntimeError(
                f"depth-run source authentication failed: {error}"
            ) from error
        content_hash = frozen[
            depth_run.clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
        ]
        if manifest.get("clip_hash_manifest_content_sha256") != content_hash:
            raise RuntimeError("depth-run clip-hash semantic identity differs")
        for name in names:
            expected = identities[name]
            if expected != {
                "source_identity_method": depth_run.SOURCE_IDENTITY_MANIFEST,
                "source_identity_value": current[name],
                "clip_hash_manifest_content_sha256": content_hash,
            }:
                raise RuntimeError(f"{name}: depth-run source identity differs")
    elif source == "direct":
        if (verification != "direct-content" or
                manifest.get("clip_hash_manifest") is not None or
                manifest.get("clip_hash_manifest_content_sha256") is not None):
            raise RuntimeError("invalid direct source-identity provenance")
        for name in names:
            expected = {
                "source_identity_method": depth_run.SOURCE_IDENTITY_FINGERPRINT,
                "source_identity_value": depth_run.source_fingerprint(clips / name),
            }
            if identities[name] != expected:
                raise RuntimeError(f"{name}: depth-run source identity differs")
    else:
        raise RuntimeError("invalid depth-run source-identity method")
    return identities


def authenticate_depth_run(clips: Path, run: Path, dataset):
    """Authenticate every current source and consumed run artifact."""
    path = run / "depth_run_manifest.json"
    manifest = read_json_object(path, "depth-run manifest")
    if (manifest.get("schema") != depth_run.DEPTH_RUN_MANIFEST_SCHEMA or
            isinstance(manifest.get("schema"), bool) or
            manifest.get("purpose") != "artistic-policy depth supervision"):
        raise RuntimeError("unsupported depth-run manifest contract")
    if manifest.get("suite_manifest_sha256") != dataset["sha256"]:
        raise RuntimeError(
            "depth run does not authenticate the current dataset manifest"
        )
    suite_value = manifest.get("suite")
    if (not isinstance(suite_value, str) or
            Path(suite_value).resolve() != clips.resolve()):
        raise RuntimeError("depth-run suite path differs from the current suite")
    names = dataset["sequence_names"]
    run_rows = manifest.get("clips")
    if not isinstance(run_rows, list):
        raise RuntimeError("depth-run manifest has no exact clip list")
    run_names = [
        row.get("clip") if isinstance(row, dict) else None for row in run_rows
    ]
    if run_names != names or len(run_names) != len(set(run_names)):
        raise RuntimeError(
            "depth-run clip set/order differs from authenticated dataset sequences"
        )
    identities = authenticate_source_identities(clips, names, manifest)
    model = manifest.get("model")
    executable_hash = manifest.get("executable_sha256")
    conf_hash = manifest.get("conf_sha256")
    if not all(isinstance(value, str) and value for value in (
            model, executable_hash, conf_hash)):
        raise RuntimeError("depth-run generation identity is incomplete")
    input_variant = manifest.get("input_variant")
    input_color.validate_input_variant(input_variant)
    metric_preview_encoding = input_variant_metric_preview_encoding(
        input_variant
    )
    hdr_source_kind = input_variant_hdr_source_kind(input_variant)
    if (manifest.get("input_variant_sha256") !=
            input_color.input_variant_sha256(input_variant) or
            manifest.get("depth_input_color_contract_sha256") !=
            input_color.color_contract_sha256() or
            manifest.get("harness_schema") != HARNESS_SCHEMA or
            manifest.get("metric_preview_encoding") !=
            metric_preview_encoding or
            manifest.get("hdr_source_kind") != hdr_source_kind):
        raise RuntimeError("depth-run input color identity is stale")

    is_native_hdr = (
        input_variant["kind"] == input_color.INPUT_KIND_NATIVE_PQ
    )
    if is_native_hdr:
        if (dataset["source_kind"] != "native-hdr-video" or
                metric_preview_encoding != native_hdr_capture.PREVIEW_ENCODING or
                hdr_source_kind != input_color.INPUT_KIND_NATIVE_PQ):
            raise RuntimeError(
                "native-HDR depth run does not match a native-HDR dataset contract"
            )
        native_hdr_clips = authenticate_native_hdr_sources(
            clips, names, manifest
        )
    else:
        if dataset["source_kind"] == "native-hdr-video":
            raise RuntimeError(
                "native-HDR datasets require the native-PQ model input contract"
            )
        native_hdr_clips = {}

    total_frames = 0
    authenticated_hashes = {}
    watched_paths = {path, Path(dataset["path"])}
    if manifest["clip_hash_source"] == "manifest":
        watched_paths.add(clips / depth_run.clip_hashes.MANIFEST_NAME)
    for row in run_rows:
        name = row["clip"]
        clip_root = clips / name
        run_clip = run / name
        selection = source_output_selection(clip_root)
        if selection["mode"] != "label-frames":
            raise RuntimeError(f"{name}: source is not label-frame selected")
        identity = identities[name]
        expected_fields = {
            "frames": len(selection["output_frame_ids"]),
            "label_frames": len(selection["label_frame_ids"]),
            "output_selection_mode": selection["mode"],
            "output_label_frames_sha256": selection["label_frames_sha256"],
            "source_identity_method": identity["source_identity_method"],
            "source_identity_value": identity["source_identity_value"],
            "metric_preview_encoding": metric_preview_encoding,
        }
        stale = {
            key: (expected, row.get(key)) for key, expected in expected_fields.items()
            if row.get(key) != expected
        }
        if stale:
            raise RuntimeError(f"{name}: depth-run clip selection differs: {stale}")
        native_authentication = native_hdr_clips.get(name)
        if native_authentication is not None:
            native_provenance = native_authentication["provenance"]
            expected_run_provenance = {
                key: native_provenance[key] for key in (
                    "manifest", "manifest_sha256", "content_sha256",
                    "width", "height", "frame_count",
                )
            }
            # Generation publishes a cheap final stat verification after any
            # optional full pre-run audit.  Source admission above independently
            # performs the required full content authentication.
            expected_run_provenance["verification"] = "stat"
            if row.get("native_hdr_model_source") != expected_run_provenance:
                raise RuntimeError(
                    f"{name}: depth-run native-HDR source provenance differs"
                )
        elif row.get("native_hdr_model_source") is not None:
            raise RuntimeError(
                f"{name}: non-native depth run carries native-HDR provenance"
            )
        contract_path = run_clip / "contract.json"
        if row.get("contract_sha256") != sha256(contract_path):
            raise RuntimeError(f"{name}: depth-run contract SHA-256 differs")
        generation = read_json_object(
            run_clip / "generation_identity.json", "generation identity"
        )
        expected_generation = depth_run.generation_identity(
            identity, selection, executable_hash, conf_hash, model,
            input_variant,
        )
        if generation != expected_generation:
            raise RuntimeError(f"{name}: generation identity differs")
        artifacts = depth_run.depth_artifact_identity(run_clip)
        for key in (
                "artifact_content_contract", "artifact_content_sha256",
                "artifact_files"):
            if row.get(key) != artifacts[key]:
                raise RuntimeError(f"{name}: depth artifact identity differs")
        for record in artifacts["artifact_files"]:
            artifact_path = run_clip / record["path"]
            authenticated_hashes[str(artifact_path.resolve())] = record["sha256"]
            watched_paths.add(artifact_path)
        contract = read_json_object(contract_path, "harness contract")
        if (contract.get("model") != model or
                contract.get("policy_warp_source_sha256") !=
                manifest.get("policy_warp_source_sha256") or
                contract.get("metric_sha256") != manifest.get("metric_sha256")):
            raise RuntimeError(f"{name}: harness/run contract differs")
        if native_authentication is not None:
            expected_geometry = (
                int(native_authentication["provenance"]["width"]),
                int(native_authentication["provenance"]["height"]),
            )
            contract_geometry = (
                int(contract.get("source_width", 0)),
                int(contract.get("source_height", 0)),
            )
            if contract_geometry != expected_geometry:
                raise RuntimeError(
                    f"{name}: native-HDR model-source dimensions differ from "
                    "the depth-run source geometry"
                )
            native_records = native_authentication["frames"]
            for record in native_records.values():
                model_path = record["model_path"]
                preview_path = record["preview_path"]
                authenticated_hashes[str(model_path.resolve())] = record["sha256"]
                authenticated_hashes[str(preview_path.resolve())] = record[
                    "preview_sha256"
                ]
                watched_paths.update((model_path, preview_path))
            watched_paths.add(native_authentication["manifest_path"])
        total_frames += expected_fields["frames"]
        watched_paths.update(numeric_frame_map(
            clip_root, suffix=SOURCE_IMAGE_SUFFIXES
        ).values())
        watched_paths.add(clip_root / LABEL_FRAMES_NAME)
        metadata_path = clip_root / "meta.json"
        if metadata_path.is_file():
            watched_paths.add(metadata_path)
    if (manifest.get("clip_count") != len(names) or
            manifest.get("frame_count") != total_frames):
        raise RuntimeError("depth-run aggregate clip/frame counts differ")
    return {
        "manifest_sha256": sha256(path),
        "source_identities": identities,
        "artifact_content_sha256": {
            row["clip"]: row["artifact_content_sha256"] for row in run_rows
        },
        "input_variant": input_variant,
        "input_variant_sha256": input_color.input_variant_sha256(
            input_variant
        ),
        "metric_preview_encoding": metric_preview_encoding,
        "hdr_source_kind": hdr_source_kind,
        "depth_input_color_contract_sha256":
            input_color.color_contract_sha256(),
        "native_hdr_sources": {
            name: authentication["provenance"]
            for name, authentication in native_hdr_clips.items()
        },
        "_native_hdr_clips": native_hdr_clips,
        "_source_mode": manifest["clip_hash_source"],
        "_authenticated_hashes": authenticated_hashes,
        "_watched_files": [
            file_stat_identity(item)
            for item in sorted(watched_paths, key=lambda value: str(value))
        ],
    }


def revalidate_authenticated_inputs(authentication):
    """Cheap second check after content authentication and row construction."""
    for expected in authentication["_watched_files"]:
        path = Path(expected["path"])
        try:
            current = file_stat_identity(path)
        except OSError as error:
            raise RuntimeError(f"authenticated input disappeared: {path}") from error
        if current != expected:
            raise RuntimeError(
                f"authenticated input changed during preparation: {path}"
            )
    # Initial verification authenticated content.  This second pass checks the
    # identity and stat tuple of every source/manifest/artifact immediately
    # before publication without rereading large sparse float textures.


def numeric_frame_map(root: Path, prefix="frame_", suffix=".png"):
    suffixes = {
        value.lower() for value in (
            suffix if isinstance(suffix, (tuple, list, set)) else (suffix,)
        )
    }
    result = {}
    for path in root.iterdir() if root.is_dir() else ():
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        token = path.stem.removeprefix(prefix)
        if not path.stem.startswith(prefix) or not token.isdigit():
            continue
        frame_id = int(token)
        if frame_id in result:
            raise RuntimeError(
                f"duplicate numeric frame identity in {root}: {frame_id}"
            )
        result[frame_id] = path
    return result


def load_label_frames(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid {LABEL_FRAMES_NAME}: {path}") from error
    if (not isinstance(payload, dict) or
            set(payload) != {"schema", "frame_ids"} or
            payload.get("schema") != LABEL_FRAMES_SCHEMA or
            isinstance(payload.get("schema"), bool)):
        raise RuntimeError(f"{path}: expected label-frame schema {LABEL_FRAMES_SCHEMA}")
    frame_ids = payload.get("frame_ids")
    if (not isinstance(frame_ids, list) or not frame_ids or
            any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in frame_ids) or
            frame_ids != sorted(set(frame_ids))):
        raise RuntimeError(
            f"{path}: frame_ids must be nonempty strictly increasing unique "
            "nonnegative integers"
        )
    return frame_ids


def evidence_frame_ids(label_frame_ids, source_frame_ids):
    """Derive the exact target-plus-adjacent artifact selection.

    Sparse target frames are useful for label throughput, but each target needs
    a full-cadence neighbor for temporal evidence.  Prefer its predecessor;
    when the target starts the source sequence, use its successor.
    """
    source_ids = set(source_frame_ids)
    selected = set(label_frame_ids)
    for frame_id in label_frame_ids:
        if frame_id - 1 in source_ids:
            companion = frame_id - 1
        elif frame_id + 1 in source_ids:
            companion = frame_id + 1
        else:
            raise RuntimeError(
                f"label frame {frame_id} has no adjacent full-cadence evidence frame"
            )
        selected.add(companion)
    return sorted(selected)


def source_output_selection(clip_root: Path):
    """Reconstruct this source stage's exact label-frame selection.

    Native-HDR clips intentionally contain ``frame_model_sources.json`` beside
    ``frame_*.png``.  Enumerating only numeric image names keeps that manifest
    from being misclassified as a frame while preserving the generation
    contract's target-plus-adjacent selection.
    """
    label_path = clip_root / LABEL_FRAMES_NAME
    label_ids = load_label_frames(label_path)
    source_ids = sorted(numeric_frame_map(
        clip_root, suffix=SOURCE_IMAGE_SUFFIXES
    ))
    if len(source_ids) < 2:
        raise RuntimeError(
            f"{clip_root.name}: still images cannot provide consecutive evidence"
        )
    missing = sorted(set(label_ids) - set(source_ids))
    if missing:
        raise RuntimeError(
            f"label-frame manifest references missing RGB: {missing}"
        )
    return {
        "mode": "label-frames",
        "label_frame_ids": label_ids,
        "output_frame_ids": evidence_frame_ids(label_ids, source_ids),
        "label_frames_sha256": sha256(label_path),
    }


def validate_depth_contract(path: Path, input_variant=None):
    input_variant = input_variant or input_color.sdr_input_variant()
    input_color.validate_input_variant(input_variant)
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid harness contract: {path}") from error
    expected = {
        "schema": HARNESS_SCHEMA,
        "artifact_mode": "depth+baseline-disparity",
        "depth_step": "current-once",
        "depth_reuse_interval": 1,
        "depth_compensation": "none",
        "depth_override_frames": 0,
        "artistic_policy": False,
        "artistic_policy_consumed": False,
        "artistic_policy_authorization": "none",
        "model_onnx_sha256": "",
        "policy_metadata_sha256": "",
        "deployment_geometry_allowlist_sha256": "",
        "artistic_scale_override": 0.0,
        "color_mode": input_variant["color_mode"],
        "metric_preview_encoding": input_variant_metric_preview_encoding(
            input_variant
        ),
        "hdr_source_kind": input_variant_hdr_source_kind(input_variant),
        "hdr_input_scale": float(input_variant["scrgb_white_scale"] or 0.0),
        "sdr_white_level_raw": int(
            input_variant["windows_sdr_white_level_raw"] or 0
        ),
        "warp_disparity": (
            "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars"
        ),
        "warp_unclamped_disparity": (
            "unclamped_full_binocular_normalized_at_artistic_scale_1_"
            "output_eye_raster_zero_bars"
        ),
        "artistic_disparity_contract": (
            "clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_"
            "times_aspect_scale_times_content_scale_x)"
        ),
    }
    stale = {key: (value, contract.get(key)) for key, value in expected.items()
             if contract.get(key) != value}
    if stale:
        raise RuntimeError(f"incompatible depth-run harness contract {path}: {stale}")
    integer_fields = (
        "source_width", "source_height", "model_input_width", "model_input_height",
        "eye_width", "eye_height", "disparity_raster_width",
        "disparity_raster_height",
    )
    if any(
            not isinstance(contract.get(key), int) or
            isinstance(contract.get(key), bool) or contract[key] <= 0
            for key in integer_fields):
        raise RuntimeError(f"invalid depth-run geometry in {path}")
    if (contract["disparity_raster_width"] != contract["eye_width"] or
            contract["disparity_raster_height"] != contract["eye_height"]):
        raise RuntimeError(
            f"depth-run disparity is not the complete eye raster: {path}"
        )
    for key in ("content_scale_x", "content_scale_y", "artistic_full_clamp_abs"):
        value = contract.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value)) or float(value) <= 0.0):
            raise RuntimeError(f"invalid {key} in {path}")
    for key, length in (("policy_warp_source_sha256", 64), ("metric_sha256", 16)):
        value = contract.get(key)
        if (not isinstance(value, str) or len(value) != length or
                any(char not in "0123456789abcdef" for char in value)):
            raise RuntimeError(f"invalid {key} in {path}")
    return contract


def validate_float_texture(path: Path, expected_shape):
    header = np.fromfile(path, dtype="<u4", count=2)
    if header.size != 2:
        raise RuntimeError(f"invalid float-texture header: {path}")
    width, height = int(header[0]), int(header[1])
    values = np.fromfile(path, dtype="<f4", offset=8)
    if ((height, width) != tuple(expected_shape) or values.size != width * height or
            not np.isfinite(values).all()):
        raise RuntimeError(f"invalid float-texture payload/shape: {path}")


def prepare_clip(clip_root: Path, run_clip: Path, dataset,
                 authenticated_hashes=None, input_variant=None,
                 native_hdr_authentication=None):
    input_variant = input_variant or input_color.sdr_input_variant()
    input_color.validate_input_variant(input_variant)
    label_frames_path = clip_root / LABEL_FRAMES_NAME
    frame_ids = load_label_frames(label_frames_path)
    source_frames = numeric_frame_map(
        clip_root, suffix=SOURCE_IMAGE_SUFFIXES
    )
    is_native_hdr = (
        input_variant["kind"] == input_color.INPUT_KIND_NATIVE_PQ
    )
    if is_native_hdr != (native_hdr_authentication is not None):
        raise RuntimeError(
            f"{clip_root.name}: native-HDR source authentication is missing "
            "or unexpected"
        )
    if len(source_frames) < 2:
        raise RuntimeError(
            f"{clip_root.name}: still images cannot provide exact temporal "
            "safe-ceiling labels"
        )
    ordered_source_ids = sorted(source_frames)
    if any(right != left + 1
           for left, right in zip(ordered_source_ids, ordered_source_ids[1:])):
        raise RuntimeError(f"{clip_root.name}: source frames are not full cadence")
    missing = sorted(set(frame_ids) - set(source_frames))
    if missing:
        raise RuntimeError(
            f"{clip_root.name}: label frames are missing from RGB: {missing}"
        )
    selected_frame_ids = evidence_frame_ids(frame_ids, ordered_source_ids)

    contract_path = run_clip / "contract.json"
    contract = validate_depth_contract(contract_path, input_variant)
    if native_hdr_authentication is not None:
        native_frames = native_hdr_authentication["frames"]
        if set(native_frames) != set(source_frames):
            raise RuntimeError(
                f"{clip_root.name}: native-HDR sidecar cadence differs from "
                "source frames"
            )
        native_geometry = (
            int(native_hdr_authentication["provenance"]["width"]),
            int(native_hdr_authentication["provenance"]["height"]),
        )
        if native_geometry != (
                int(contract["source_width"]), int(contract["source_height"])):
            raise RuntimeError(
                f"{clip_root.name}: native-HDR sidecar dimensions differ from "
                "the harness source geometry"
            )
    if contract.get("output_selection_mode") != "label-frames":
        raise RuntimeError(
            f"{clip_root.name}: depth run did not use label-frame artifact selection"
        )
    if contract.get("label_frame_ids") != frame_ids:
        raise RuntimeError(
            f"{clip_root.name}: depth-run target IDs differ from label_frames.json"
        )
    if contract.get("output_selected_frame_ids") != selected_frame_ids:
        raise RuntimeError(
            f"{clip_root.name}: depth-run evidence IDs differ from the required "
            "target-plus-adjacent selection"
        )
    label_frames_hash = sha256(label_frames_path)
    if contract.get("output_label_frames_sha256") != label_frames_hash:
        raise RuntimeError(
            f"{clip_root.name}: depth-run label-frame manifest hash is stale"
        )
    depth_ids = set(numeric_frame_map(run_clip, "depth_"))
    disparity_ids = set(numeric_frame_map(run_clip, "baseline_disparity_", ".f32"))
    raw_ids = set(numeric_frame_map(
        run_clip, "baseline_unclamped_disparity_", ".f32"
    ))
    selected_ids = set(selected_frame_ids)
    if selected_ids != depth_ids or depth_ids != disparity_ids or depth_ids != raw_ids:
        raise RuntimeError(
            f"{clip_root.name}: depth artifacts do not exactly cover the authenticated "
            "target-plus-adjacent selection"
        )

    metadata_path = clip_root / "meta.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{clip_root.name}: invalid meta.json") from error
        if not isinstance(metadata, dict):
            raise RuntimeError(f"{clip_root.name}: invalid meta.json")
        metadata_production = metadata.get(
            "production_id", metadata.get("film_id")
        )
        mismatches = {}
        for key, expected, actual in (
            ("production_id", dataset["production_id"], metadata_production),
            ("split", dataset["split"], metadata.get("split")),
            ("source_kind", dataset["source_kind"], metadata.get("source_kind")),
        ):
            if actual is not None and actual != expected:
                mismatches[key] = (expected, actual)
        if metadata.get("global_policy_weight") is not None:
            try:
                metadata_weight = float(metadata["global_policy_weight"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"{clip_root.name}: invalid meta.json global_policy_weight"
                ) from error
            if (not math.isfinite(metadata_weight) or
                    abs(metadata_weight - dataset["global_policy_weight"]) > 1e-9):
                mismatches["global_policy_weight"] = (
                    dataset["global_policy_weight"], metadata_weight
                )
        if mismatches:
            raise RuntimeError(
                f"{clip_root.name}: meta.json differs from dataset manifest: "
                f"{mismatches}"
            )

    expected_shape = (
        int(contract["disparity_raster_height"]),
        int(contract["disparity_raster_width"]),
    )
    right_frames = numeric_frame_map(
        clip_root / "gt_right", suffix=SOURCE_IMAGE_SUFFIXES
    )
    rows = []
    for frame_id in frame_ids:
        suffix = f"{frame_id:05d}"
        source_path = source_frames[frame_id]
        baseline_path = run_clip / f"baseline_disparity_{suffix}.f32"
        raw_path = run_clip / f"baseline_unclamped_disparity_{suffix}.f32"
        for path in (baseline_path, raw_path):
            if not path.is_file():
                raise RuntimeError(f"{clip_root.name}: missing depth artifact {path}")
            validate_float_texture(path, expected_shape)
        row = {
            "source_schema": SOURCE_SCHEMA,
            "source_contract": SOURCE_CONTRACT,
            "selection_mode": "label-frames",
            "label_frames": str(label_frames_path.resolve()),
            "label_frames_sha256": label_frames_hash,
            "label_frame_ids": frame_ids,
            "output_selected_frame_ids": selected_frame_ids,
            "source": str(source_path.resolve()),
            "source_sha256": sha256(source_path),
            "baseline_disparity": str(baseline_path.resolve()),
            "baseline_disparity_sha256": (
                authenticated_hashes[str(baseline_path.resolve())]
                if authenticated_hashes is not None else sha256(baseline_path)
            ),
            "baseline_unclamped_disparity": str(raw_path.resolve()),
            "baseline_unclamped_disparity_sha256": (
                authenticated_hashes[str(raw_path.resolve())]
                if authenticated_hashes is not None else sha256(raw_path)
            ),
            "source_width": int(contract["source_width"]),
            "source_height": int(contract["source_height"]),
            "model_input_width": int(contract["model_input_width"]),
            "model_input_height": int(contract["model_input_height"]),
            "eye_width": int(contract["eye_width"]),
            "eye_height": int(contract["eye_height"]),
            "content_scale_x": float(contract["content_scale_x"]),
            "content_scale_y": float(contract["content_scale_y"]),
            "color_mode": contract["color_mode"],
            "metric_preview_encoding": contract["metric_preview_encoding"],
            "hdr_source_kind": contract["hdr_source_kind"],
            "input_variant": input_variant,
            "input_variant_sha256": input_color.input_variant_sha256(
                input_variant
            ),
            "disparity_raster_width": int(contract["disparity_raster_width"]),
            "disparity_raster_height": int(contract["disparity_raster_height"]),
            "artistic_full_clamp_abs": float(contract["artistic_full_clamp_abs"]),
            "policy_warp_source_sha256": contract["policy_warp_source_sha256"],
            "metric_sha256": contract["metric_sha256"],
            "harness_contract_sha256": (
                authenticated_hashes[str(contract_path.resolve())]
                if authenticated_hashes is not None else sha256(contract_path)
            ),
            "clip": clip_root.name,
            "frame": frame_id,
            "split": dataset["split"],
            "domain": dataset["domain"],
            "dataset": dataset["dataset"],
            "film_id": dataset["production_id"],
            "production_id": dataset["production_id"],
            "source_kind": dataset["source_kind"],
            "license": dataset["license"],
            "source_split": dataset["source_split"],
            "projection": dataset["projection"],
            "policy_role": dataset["policy_role"],
            "global_policy_weight": dataset["global_policy_weight"],
        }
        if native_hdr_authentication is not None:
            native_record = native_hdr_authentication["frames"][frame_id]
            if native_record["preview_path"] != source_path.resolve():
                raise RuntimeError(
                    f"{clip_root.name}: native-HDR source/preview mapping differs "
                    f"for frame {frame_id}"
                )
            if row["source_sha256"] != native_record["preview_sha256"]:
                raise RuntimeError(
                    f"{clip_root.name}: native-HDR preview hash differs for "
                    f"frame {frame_id}"
                )
            row.update({
                "model_source": str(native_record["model_path"].resolve()),
                "model_source_sha256": native_record["sha256"],
                "model_source_encoding": native_hdr_capture.CAPTURE_ENCODING,
                "native_hdr_source_provenance": {
                    **native_hdr_authentication["provenance"],
                    "frame": frame_id,
                    "timestamp_seconds": float(
                        native_record["timestamp_seconds"]
                    ),
                },
            })
        if (not math.isfinite(row["global_policy_weight"]) or
                row["global_policy_weight"] <= 0.0):
            raise RuntimeError(
                f"{clip_root.name}: meta.json has invalid global_policy_weight"
            )
        right_path = right_frames.get(frame_id)
        if right_path is not None:
            row.update({
                "right_eye": str(right_path.resolve()),
                "right_eye_sha256": sha256(right_path),
            })
        reference_path = clip_root / "gt_disparity" / f"frame_{suffix}.npz"
        if reference_path.is_file():
            row.update({
                "reference_disparity": str(reference_path.resolve()),
                "reference_disparity_sha256": sha256(reference_path),
            })
        rows.append(row)
    return rows, {
        "clip": clip_root.name,
        "source_frames": len(source_frames),
        "label_frames": len(frame_ids),
        "output_frames": len(selected_frame_ids),
        "first_label_frame": frame_ids[0],
        "last_label_frame": frame_ids[-1],
        "label_frames_sha256": label_frames_hash,
        "stereo_frames": sum("right_eye" in row for row in rows),
        "native_hdr_frames": sum("model_source" in row for row in rows),
    }


def prepare(run: Path, clips: Path, output: Path, overwrite=False):
    validate_output_roots(run, clips, output)
    if output.exists() and not output.is_dir():
        raise RuntimeError(f"output exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(f"output must be empty (or use --overwrite): {output}")
    dataset = dataset_contract(clips)
    depth_authentication = authenticate_depth_run(clips, run, dataset)
    rows = []
    clip_stats = []
    for clip_name in dataset["sequence_names"]:
        clip_root = clips / clip_name
        run_clip = run / clip_root.name
        if not run_clip.is_dir():
            raise RuntimeError(f"missing run output for clip: {clip_root.name}")
        clip_rows, stats = prepare_clip(
            clip_root, run_clip, dataset,
            depth_authentication["_authenticated_hashes"],
            depth_authentication["input_variant"],
            depth_authentication["_native_hdr_clips"].get(clip_name),
        )
        rows.extend(clip_rows)
        clip_stats.append(stats)
        print(f"[{clip_root.name}] {len(clip_rows)} source rows", flush=True)
    if not rows:
        raise RuntimeError("no full-cadence label-frame clips were admitted")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.partial-", dir=output.parent
    ))
    try:
        labels_path = staging / "labels.jsonl"
        with labels_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        contract = {
            "schema": SOURCE_SCHEMA,
            "source_contract": SOURCE_CONTRACT,
            "purpose": "render-feasibility source admission; no authored-camera target",
            "run": str(run.resolve()),
            "clips": str(clips.resolve()),
            "run_contract": run_contract(run),
            "depth_authentication": {
                key: value for key, value in depth_authentication.items()
                if not key.startswith("_")
            },
            "input_variant": depth_authentication["input_variant"],
            "metric_preview_encoding": depth_authentication[
                "metric_preview_encoding"
            ],
            "input_variant_sha256": depth_authentication[
                "input_variant_sha256"
            ],
            "depth_input_color_contract_sha256": depth_authentication[
                "depth_input_color_contract_sha256"
            ],
            "dataset_manifest": {
                key: dataset[key]
                for key in (
                    "path", "sha256", "schema", "production_id", "split",
                    "source_kind", "global_policy_weight",
                )
            },
            "label_frame_contract": {
                "file": LABEL_FRAMES_NAME,
                "schema": LABEL_FRAMES_SCHEMA,
                "selection": (
                    "sparse targets plus one adjacent evidence frame per target"
                ),
                "stills": "rejected",
            },
            "code": {
                "source_preparation": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256(Path(__file__).resolve()),
                },
            },
        }
        contract_path = staging / "source_contract.json"
        contract_path.write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )
        summary = {
            "schema": SOURCE_SCHEMA,
            "source_contract": SOURCE_CONTRACT,
            "accepted": len(rows),
            "labels_sha256": sha256(labels_path),
            "source_contract_sha256": sha256(contract_path),
            "clip_stats": clip_stats,
            "clip_counts": dict(Counter(row["clip"] for row in rows)),
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "stereo_rows": sum("right_eye" in row for row in rows),
            "native_hdr_rows": sum("model_source" in row for row in rows),
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        revalidate_authenticated_inputs(depth_authentication)
        publish_staged_directory(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = prepare(args.run, args.clips, args.output, args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
