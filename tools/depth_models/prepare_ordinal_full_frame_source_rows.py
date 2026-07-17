#!/usr/bin/env python3
"""Publish authenticated target-only sources for ordinal policy training.

This is deliberately separate from ``prepare_artistic_source_rows.py``.  The
ordinal bundle contains authenticated sparse targets. One invocation publishes
only those labeled target frames for exactly one production and input condition,
using one canonical ordinal bundle per manifest clip. Unlabeled cadence frames
are deliberately excluded from training, development, and sealed-test inputs.

The ordinal bundle authenticates the source bytes, production model geometry,
input-color contract, model depth, and complete two-geometry render lattice.
For native PQ, this publisher additionally authenticates the matching linear
scRGB FP16 model source rather than mistaking its PNG preview for model input.
Sealed-test manifests are rejected before any clip path is opened.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import cv2

import artistic_geometry_contract as geometry_contract
import build_ordinal_frame_label_bundle as ordinal_bundle
import depth_input_color as input_color
import native_hdr_capture


SOURCE_SCHEMA = 7
SOURCE_CONTRACT = "apollo-ordinal-target-source-v1"
SUMMARY_SCHEMA = 7
SUMMARY_CONTRACT = "apollo-ordinal-target-source-summary-v1"
SELECTION_MODE = "authenticated-sparse-targets-only"
ROW_ROLES = {"target"}
SOURCE_CADENCE_CONTRACT = "ordered-frame-id-and-source-sha256-v1"
SOURCE_IMAGE = re.compile(r"^frame_([0-9]+)[.](png|jpg|jpeg)$", re.IGNORECASE)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
WORKING_SPLITS = {"training", "development"}
WORKING_SOURCE_KINDS = {"mono-video", "native-hdr-video"}


def canonical_bytes(value):
    return (json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _load_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not an object: {path}")
    return value


def _safe_clip(value):
    if (not isinstance(value, str) or not value or value in {".", ".."} or
            Path(value).name != value or "/" in value or "\\" in value):
        raise RuntimeError(f"unsafe dataset clip identity: {value!r}")
    return value


def _sequence_frame_count(sequence, source_kind):
    key = "context_frames" if source_kind == "mono-video" else "frames"
    value = sequence.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError(f"dataset sequence has invalid {key}")
    return value


def _sequence_frame_rate(payload, sequence, source_kind):
    """Return the authenticated cadence represented by consecutive rows."""
    key = "source_frame_rate" if source_kind == "native-hdr-video" else "context_fps"
    value = sequence.get(key, payload.get(key))
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"dataset sequence has invalid {key}") from error
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"dataset sequence has invalid {key}")
    return value


def load_dataset_manifest(path):
    """Open one train/dev manifest and reject sealed test before clip access."""
    path = Path(path).resolve(strict=True)
    payload = _load_json(path, "dataset manifest")
    split = payload.get("split")
    if split not in WORKING_SPLITS:
        raise RuntimeError(
            "ordinal source publication is train/development only; "
            f"refusing split {split!r}"
        )
    source_kind = payload.get("source_kind")
    if payload.get("schema") != 2 or source_kind not in WORKING_SOURCE_KINDS:
        raise RuntimeError("unsupported ordinal source dataset contract")
    production_id = payload.get("production_id")
    if not isinstance(production_id, str) or not production_id:
        raise RuntimeError("dataset manifest lacks production identity")
    try:
        policy_weight = float(payload.get("global_policy_weight"))
    except (TypeError, ValueError) as error:
        raise RuntimeError("dataset policy weight is invalid") from error
    if not math.isfinite(policy_weight) or policy_weight <= 0.0:
        raise RuntimeError("dataset policy weight is invalid")
    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise RuntimeError("dataset manifest has no source sequences")
    rows = []
    seen = set()
    for sequence in sequences:
        if not isinstance(sequence, dict) or sequence.get("split") != split:
            raise RuntimeError("dataset sequence split differs")
        clip = _safe_clip(sequence.get("clip"))
        if clip in seen:
            raise RuntimeError("dataset manifest repeats a clip")
        seen.add(clip)
        rows.append({
            "clip": clip,
            "frame_count": _sequence_frame_count(sequence, source_kind),
            "source_frame_rate": _sequence_frame_rate(
                payload, sequence, source_kind
            ),
        })
    declared = (
        payload.get("context_frame_count")
        if source_kind == "mono-video" else payload.get("frame_count")
    )
    if declared != sum(row["frame_count"] for row in rows):
        raise RuntimeError("dataset aggregate frame count differs")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "root": path.parent,
        "payload": payload,
        "production_id": production_id,
        "split": split,
        "source_kind": source_kind,
        "global_policy_weight": policy_weight,
        "sequences": rows,
    }


def _source_frames(clip_root, expected_count):
    result = {}
    for path in clip_root.iterdir() if clip_root.is_dir() else ():
        if not path.is_file():
            continue
        match = SOURCE_IMAGE.fullmatch(path.name)
        if match is None:
            continue
        frame_id = int(match.group(1))
        if frame_id in result:
            raise RuntimeError(f"{clip_root.name}: duplicate source frame {frame_id}")
        result[frame_id] = path.resolve()
    expected = list(range(expected_count))
    if sorted(result) != expected:
        raise RuntimeError(
            f"{clip_root.name}: source frame cadence differs from manifest"
        )
    return result


def _source_cadence_identity(frames):
    frame_ids = list(frames)
    content_rows = [{
        "frame_id": frame_id,
        "source_sha256": sha256_file(frames[frame_id]),
    } for frame_id in frame_ids]
    return {
        "contract": SOURCE_CADENCE_CONTRACT,
        "frame_count": len(frame_ids),
        "frame_ids": frame_ids,
        "frame_ids_sha256":
            ordinal_bundle.run_eval.frame_id_sequence_sha256(frame_ids),
        "content_sha256": canonical_sha256(content_rows),
    }


def _image_dimensions(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode ordinal source image: {path}")
    return int(image.shape[1]), int(image.shape[0])


def _bundle_identity(path):
    path = Path(path).resolve(strict=True)
    records = ordinal_bundle.validate_frame_label_bundle(path)
    summary_path = path.parent / "summary.json"
    summary = ordinal_bundle.validate_frame_label_summary(
        path, summary_path, records=records
    )
    header = records[0]
    if (summary.get("label_bundle_sha256") != sha256_file(path) or
            summary.get("frame_count") != len(records) - 2 or
            summary.get("clip") != header["clip"] or
            summary.get("input_variant_sha256") !=
            header["input_variant_sha256"] or
            summary.get("source_frame_count") !=
            header["source_frame_count"] or
            summary.get("output_frame_count") !=
            header["output_frame_count"] or
            summary.get("output_label_frames_sha256") !=
            header["output_label_frames_sha256"]):
        raise RuntimeError(f"ordinal bundle summary is stale: {path}")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "summary": summary_path.resolve(),
        "summary_sha256": sha256_file(summary_path),
        "header": header,
        "frames": records[1:-1],
    }


def _model_geometry(header):
    allowlist = header["deployment_geometry_allowlist"]
    geometry_contract.validate_allowlist(allowlist)
    fields = (
        "source_width", "source_height", "model_input_width",
        "model_input_height", "depth_short_side", "depth_max_aspect",
    )
    values = {
        tuple(item[field] for field in fields)
        for item in allowlist["tuples"]
    }
    if len(values) != 1:
        raise RuntimeError(
            "ordinal bundle deployment geometries disagree on model input"
        )
    value = next(iter(values))
    return dict(zip(fields, value))


def _code_identities():
    paths = {
        "source_publisher": Path(__file__).resolve(),
        "ordinal_bundle": Path(ordinal_bundle.__file__).resolve(),
        "geometry_contract": Path(geometry_contract.__file__).resolve(),
        "input_color": Path(input_color.__file__).resolve(),
        "native_hdr_capture": Path(native_hdr_capture.__file__).resolve(),
    }
    return {
        role: {"path": str(path), "sha256": sha256_file(path)}
        for role, path in sorted(paths.items())
    }


def _select_sequences(dataset, selected_clips):
    sequences = dataset["sequences"]
    if selected_clips is None:
        return list(sequences), "full-dataset"
    selected_clips = list(selected_clips)
    if (not selected_clips or len(selected_clips) != len(set(selected_clips))):
        raise RuntimeError("ordinal source clip subset is empty or duplicated")
    selected = {_safe_clip(value) for value in selected_clips}
    available = {row["clip"] for row in sequences}
    unknown = sorted(selected - available)
    if unknown:
        raise RuntimeError(
            "ordinal source clip subset is outside the dataset manifest: " +
            ", ".join(unknown)
        )
    return (
        [row for row in sequences if row["clip"] in selected],
        "full-dataset" if selected == available else
        "smoke-subset-not-training-eligible",
    )


def build_rows(dataset_manifest, bundle_paths, selected_clips=None):
    """Build one exact condition bundle without publishing it."""
    dataset = load_dataset_manifest(dataset_manifest)
    selected_sequences, scope = _select_sequences(dataset, selected_clips)
    if not bundle_paths:
        raise RuntimeError("ordinal source publication has no frame bundles")
    bundles = [_bundle_identity(path) for path in bundle_paths]
    by_clip = {}
    for bundle in bundles:
        clip = bundle["header"]["clip"]
        if clip in by_clip:
            raise RuntimeError(f"ordinal source repeats bundle clip {clip!r}")
        by_clip[clip] = bundle
    expected_clips = [row["clip"] for row in selected_sequences]
    if set(by_clip) != set(expected_clips):
        raise RuntimeError(
            "ordinal bundles do not exactly cover the dataset sequence set"
        )

    variants = {
        json.dumps(bundle["header"]["input_variant"], sort_keys=True)
        for bundle in bundles
    }
    if len(variants) != 1:
        raise RuntimeError("ordinal source bundles mix input conditions")
    input_variant = bundles[0]["header"]["input_variant"]
    input_color.validate_input_variant(input_variant)
    variant_sha256 = input_color.input_variant_sha256(input_variant)
    if any(bundle["header"]["input_variant_sha256"] != variant_sha256
           for bundle in bundles):
        raise RuntimeError("ordinal source input-variant identity is stale")
    native = input_variant["kind"] == input_color.INPUT_KIND_NATIVE_PQ
    if native != (dataset["source_kind"] == "native-hdr-video"):
        raise RuntimeError(
            "native-PQ condition and native-HDR dataset role must match"
        )

    common_geometry = None
    rows = []
    bundle_identities = []
    native_identities = {}
    source_cadence_identities = {}
    for sequence in selected_sequences:
        clip = sequence["clip"]
        bundle = by_clip[clip]
        header = bundle["header"]
        geometry = _model_geometry(header)
        if common_geometry is None:
            common_geometry = geometry
        elif common_geometry != geometry:
            raise RuntimeError(
                "one production/condition has mixed production model geometry"
            )
        frames = _source_frames(
            dataset["root"] / clip, sequence["frame_count"]
        )
        source_cadence_identities[clip] = _source_cadence_identity(frames)
        source_frame_ids = list(frames)
        target_frames = bundle["frames"]
        target_ids = [frame["frame_id"] for frame in target_frames]
        target_by_id = {
            frame["frame_id"]: frame for frame in target_frames
        }
        runtime_scene_trace = header["runtime_scene_trace"]
        runtime_scene_by_frame = {
            frame["source_frame_id"]: frame
            for frame in runtime_scene_trace
        }
        output_frame_ids = header["output_selected_frame_ids"]
        expected_source_ordinals = {
            frame_id: ordinal
            for ordinal, frame_id in enumerate(source_frame_ids)
        }
        if (not target_ids or target_ids != sorted(set(target_ids)) or
                not set(target_ids).issubset(frames) or
                header["source_frame_ids"] != source_frame_ids or
                header["source_frame_count"] != len(source_frame_ids) or
                header["source_frame_ids_sha256"] !=
                ordinal_bundle.run_eval.frame_id_sequence_sha256(
                    source_frame_ids
                ) or
                set(runtime_scene_by_frame) != set(source_frame_ids) or
                len(runtime_scene_by_frame) != len(runtime_scene_trace) or
                header["runtime_scene_trace_sha256"] !=
                ordinal_bundle.canonical_sha256(runtime_scene_trace) or
                header["label_frame_ids"] != target_ids or
                header["frame_count"] != len(target_ids) or
                header["output_frame_count"] != len(output_frame_ids) or
                any(
                    target["ordinal"] != ordinal or
                    target["source_ordinal"] !=
                    expected_source_ordinals[target["frame_id"]]
                    for ordinal, target in enumerate(target_frames)
                )):
            raise RuntimeError(
                f"{clip}: ordinal bundle target selection differs from source cadence"
            )
        native_auth = None
        if native:
            native_auth = native_hdr_capture.validate_clip(
                dataset["root"] / clip, full=True
            )
            if set(native_auth["frames"]) != set(frames):
                raise RuntimeError(
                    f"{clip}: native-PQ model sources differ from RGB cadence"
                )
            for frame_id, source in frames.items():
                native_frame = native_auth["frames"][frame_id]
                if (native_frame["preview_path"] != source or
                        native_frame["preview_sha256"] !=
                        sha256_file(source)):
                    raise RuntimeError(
                        f"{clip}/frame-{frame_id}: native-PQ preview join differs"
                    )
            native_identities[clip] = {
                key: native_auth[key] for key in (
                    "manifest", "manifest_sha256", "content_sha256",
                    "width", "height", "frame_count", "verification",
                )
            }
        bundle_identities.append({
            "clip": clip,
            "path": str(bundle["path"]),
            "sha256": bundle["sha256"],
            "summary": str(bundle["summary"]),
            "summary_sha256": bundle["summary_sha256"],
            "frame_count": len(target_frames),
            "source_frame_count": len(source_frame_ids),
            "source_frame_ids_sha256": header["source_frame_ids_sha256"],
            "label_frame_ids": target_ids,
            "output_frame_count": len(output_frame_ids),
            "output_selected_frame_ids": output_frame_ids,
            "output_label_frames_sha256": header[
                "output_label_frames_sha256"
            ],
            "deployment_geometry_allowlist_sha256": header[
                "deployment_geometry_allowlist_sha256"
            ],
            "runtime_scene_trace_sha256": header[
                "runtime_scene_trace_sha256"
            ],
        })
        for source_ordinal, frame_id in enumerate(source_frame_ids):
            target = target_by_id.get(frame_id)
            if target is None:
                continue
            source = frames[frame_id]
            source_hash = sha256_file(source)
            width, height = _image_dimensions(source)
            runtime_scene = runtime_scene_by_frame[frame_id]
            if (width != geometry["source_width"] or
                    height != geometry["source_height"]):
                raise RuntimeError(
                    f"{clip}/frame-{frame_id}: source geometry differs"
                )
            row = {
                "source_schema": SOURCE_SCHEMA,
                "source_contract": SOURCE_CONTRACT,
                "selection_mode": SELECTION_MODE,
                "row_role": "target",
                "source": str(source),
                "source_sha256": source_hash,
                "source_width": width,
                "source_height": height,
                "model_input_width": geometry["model_input_width"],
                "model_input_height": geometry["model_input_height"],
                "depth_short_side": geometry["depth_short_side"],
                "depth_max_aspect": geometry["depth_max_aspect"],
                "input_variant": input_variant,
                "input_variant_sha256": variant_sha256,
                "depth_input_color_contract_sha256":
                    input_color.color_contract_sha256(),
                "ordinal_bundle": str(bundle["path"]),
                "ordinal_bundle_sha256": bundle["sha256"],
                "runtime_scene_id": runtime_scene["runtime_scene_id"],
                "runtime_scene_evidence": runtime_scene,
                "runtime_scene_trace_sha256": header[
                    "runtime_scene_trace_sha256"
                ],
                "clip": clip,
                "frame": frame_id,
                "source_ordinal": source_ordinal,
                "source_frame_rate": sequence["source_frame_rate"],
                "split": dataset["split"],
                "film_id": dataset["production_id"],
                "production_id": dataset["production_id"],
                "source_kind": dataset["source_kind"],
                "domain": dataset["payload"].get(
                    "domain", dataset["payload"].get("dataset", "unknown")
                ),
                "dataset": dataset["payload"].get("dataset"),
                "license": dataset["payload"].get("license"),
                "source_split": dataset["payload"].get("source_split"),
                "projection": dataset["payload"].get(
                    "projection", "rectilinear"
                ),
                "policy_role": dataset["payload"].get(
                    "policy_role", "cinematic_training"
                ),
                "global_policy_weight": dataset["global_policy_weight"],
            }
            provenance = target["model_input_provenance"]
            if (source_hash != provenance["source_artifact_sha256"] or
                    variant_sha256 != provenance["input_variant_sha256"] or
                    target["model_input_provenance_sha256"] !=
                    ordinal_bundle.canonical_sha256(provenance) or
                    target["source_ordinal"] != source_ordinal or
                    target["runtime_scene_evidence"] != runtime_scene):
                raise RuntimeError(
                    f"{clip}/frame-{frame_id}: target provenance differs"
                )
            row.update({
                "label_ordinal": target["ordinal"],
                "ordinal_frame_model_input_provenance": provenance,
                "ordinal_frame_model_input_provenance_sha256": target[
                    "model_input_provenance_sha256"
                ],
                "ordinal_model_depth_artifact_sha256": target[
                    "model_depth_artifact_sha256"
                ],
                "deployment_geometry_allowlist_sha256": header[
                    "deployment_geometry_allowlist_sha256"
                ],
            })
            if native_auth is not None:
                native_frame = native_auth["frames"][frame_id]
                if (native_frame["preview_path"] != source or
                        native_frame["preview_sha256"] != source_hash or
                        (native_auth["width"], native_auth["height"]) !=
                        (width, height)):
                    raise RuntimeError(
                        f"{clip}/frame-{frame_id}: native-PQ preview differs"
                    )
                row.update({
                    "model_source": str(native_frame["model_path"]),
                    "model_source_sha256": native_frame["sha256"],
                    "model_source_encoding":
                        native_hdr_capture.CAPTURE_ENCODING,
                    "native_hdr_source_manifest": native_auth["manifest"],
                    "native_hdr_source_manifest_sha256":
                        native_auth["manifest_sha256"],
                })
            rows.append(row)
    if len(rows) != sum(
            identity["frame_count"] for identity in bundle_identities):
        raise RuntimeError("ordinal target row cardinality differs from bundles")
    rows.sort(key=lambda row: (row["clip"], row["frame"]))
    bundle_identities.sort(key=lambda item: item["clip"])
    return rows, {
        "dataset": dataset,
        "input_variant": input_variant,
        "input_variant_sha256": variant_sha256,
        "model_geometry": common_geometry,
        "bundles": bundle_identities,
        "full_source_cadence": source_cadence_identities,
        "native_hdr_sources": native_identities,
        "selected_sequences": selected_sequences,
        "selected_clips": expected_clips,
        "scope": scope,
    }


def _remove_path(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _paths_overlap(left, right):
    left = Path(left).resolve(strict=False)
    right = Path(right).resolve(strict=False)
    return (
        left == right or left.is_relative_to(right) or right.is_relative_to(left)
    )


def _publish_staged(staging, destination):
    backup = destination.with_name(
        f".{destination.name}.backup-{uuid.uuid4().hex}"
    )
    moved = False
    try:
        if destination.exists() or destination.is_symlink():
            destination.replace(backup)
            moved = True
        staging.replace(destination)
    except BaseException:
        if moved and not destination.exists():
            backup.replace(destination)
        raise
    if moved:
        _remove_path(backup)


def publish(dataset_manifest, bundle_paths, output, overwrite=False,
            selected_clips=None):
    output = Path(output).resolve(strict=False)
    dataset_root = Path(dataset_manifest).resolve(strict=True).parent
    protected = [dataset_root]
    protected.extend(
        Path(path).resolve(strict=True).parent for path in bundle_paths
    )
    if any(_paths_overlap(output, path) for path in protected):
        raise RuntimeError("ordinal source output overlaps an authenticated input")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not overwrite:
            raise RuntimeError(
                "ordinal source output must be empty (or use --overwrite)"
            )
    rows, context = build_rows(
        dataset_manifest, bundle_paths, selected_clips=selected_clips
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.partial-", dir=output.parent
    ))
    try:
        labels_path = staging / "labels.jsonl"
        with labels_path.open("wb") as stream:
            for row in rows:
                stream.write(canonical_bytes(row))
        dataset = context["dataset"]
        source_frame_count = sum(
            sequence["frame_count"]
            for sequence in context["selected_sequences"]
        )
        output_frame_count = sum(
            identity["output_frame_count"]
            for identity in context["bundles"]
        )
        target_rows = [row for row in rows if row["row_role"] == "target"]
        context_rows = [row for row in rows if row["row_role"] == "context"]
        contract = {
            "schema": SOURCE_SCHEMA,
            "source_contract": SOURCE_CONTRACT,
            "purpose": (
                "target-only ordinal policy input with authenticated artistic-"
                "safety targets"
            ),
            "selection_mode": SELECTION_MODE,
            "sealed_test_policy":
                "test manifests are rejected before clip media access",
            "scope": context["scope"],
            "selected_clips": context["selected_clips"],
            "source_frame_count": source_frame_count,
            "row_count": len(rows),
            "context_row_count": 0,
            "target_row_count": len(target_rows),
            "label_frame_count": len(target_rows),
            "output_frame_count": output_frame_count,
            "dataset_manifest": {
                "path": str(dataset["path"]),
                "sha256": dataset["sha256"],
                "production_id": dataset["production_id"],
                "split": dataset["split"],
                "source_kind": dataset["source_kind"],
            },
            "input_variant": context["input_variant"],
            "input_variant_sha256": context["input_variant_sha256"],
            "depth_input_color_contract_sha256":
                input_color.color_contract_sha256(),
            "model_geometry": context["model_geometry"],
            "ordinal_bundles": context["bundles"],
            "full_source_cadence": context["full_source_cadence"],
            "native_hdr_sources": context["native_hdr_sources"],
            "labels_sha256": sha256_file(labels_path),
            "code": _code_identities(),
        }
        contract_path = staging / "source_contract.json"
        contract_path.write_bytes(canonical_bytes(contract))
        summary = {
            "schema": SUMMARY_SCHEMA,
            "contract": SUMMARY_CONTRACT,
            "source_contract": SOURCE_CONTRACT,
            "accepted": len(rows),
            "labels_sha256": sha256_file(labels_path),
            "source_contract_sha256": sha256_file(contract_path),
            "production_id": dataset["production_id"],
            "split": dataset["split"],
            "source_kind": dataset["source_kind"],
            "scope": context["scope"],
            "selected_clips": context["selected_clips"],
            "selection_mode": SELECTION_MODE,
            "source_frame_count": source_frame_count,
            "row_count": len(rows),
            "context_row_count": 0,
            "target_row_count": len(target_rows),
            "label_frame_count": len(target_rows),
            "output_frame_count": output_frame_count,
            "full_source_cadence_sha256": canonical_sha256(
                context["full_source_cadence"]
            ),
            "input_variant_sha256": context["input_variant_sha256"],
            "clip_counts": dict(sorted(Counter(
                row["clip"] for row in rows
            ).items())),
            "context_clip_counts": dict(sorted(Counter(
                row["clip"] for row in context_rows
            ).items())),
            "target_clip_counts": dict(sorted(Counter(
                row["clip"] for row in target_rows
            ).items())),
            "source_frame_rates": sorted({
                row["source_frame_rate"] for row in rows
            }),
            "native_hdr_rows": sum("model_source" in row for row in rows),
        }
        (staging / "summary.json").write_bytes(canonical_bytes(summary))
        _publish_staged(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    validate_full_frame_source_bundle(output / "labels.jsonl")
    return summary


def validate_full_frame_source_bundle(labels_path, verify_media=True):
    """Validate one canonical target-only source bundle.

    The legacy function name is retained temporarily because the orchestration
    and trainer import it directly; its accepted contract is target-only.
    """
    labels_path = Path(labels_path).resolve(strict=True)
    summary_path = labels_path.parent / "summary.json"
    contract_path = labels_path.parent / "source_contract.json"
    summary = _load_json(summary_path, "ordinal source summary")
    contract = _load_json(contract_path, "ordinal source contract")
    if (summary.get("schema") != SUMMARY_SCHEMA or
            summary.get("contract") != SUMMARY_CONTRACT or
            summary.get("source_contract") != SOURCE_CONTRACT or
            contract.get("schema") != SOURCE_SCHEMA or
            contract.get("source_contract") != SOURCE_CONTRACT or
            summary.get("labels_sha256") != sha256_file(labels_path) or
            summary.get("source_contract_sha256") !=
            sha256_file(contract_path) or
            contract.get("labels_sha256") != sha256_file(labels_path) or
            contract.get("depth_input_color_contract_sha256") !=
            input_color.color_contract_sha256()):
        raise RuntimeError(f"ordinal source bundle is stale: {labels_path}")
    code = contract.get("code")
    if not isinstance(code, dict) or set(code) != set(_code_identities()):
        raise RuntimeError("ordinal source code identity set differs")
    for role, identity in code.items():
        path = (
            Path(identity.get("path", ""))
            if isinstance(identity, dict) else Path()
        )
        if (not path.is_file() or not isinstance(identity, dict) or
                identity.get("sha256") != sha256_file(path)):
            raise RuntimeError(
                f"ordinal source code identity is stale for {role!r}"
            )
    raw_lines = labels_path.read_bytes().splitlines(keepends=True)
    if not raw_lines or any(not line.strip() for line in raw_lines):
        raise RuntimeError("ordinal source labels are empty or gapped")
    try:
        rows = [json.loads(line) for line in raw_lines]
    except (TypeError, ValueError) as error:
        raise RuntimeError("ordinal source labels are invalid JSONL") from error
    if any(canonical_bytes(row) != raw
           for row, raw in zip(rows, raw_lines)):
        raise RuntimeError("ordinal source labels are not canonical JSONL")
    if summary.get("accepted") != len(rows):
        raise RuntimeError("ordinal source summary row count differs")
    dataset_identity = contract.get("dataset_manifest")
    if not isinstance(dataset_identity, dict):
        raise RuntimeError("ordinal source dataset identity is missing")
    dataset = load_dataset_manifest(dataset_identity.get("path", ""))
    selected_clips = contract.get("selected_clips")
    if (not isinstance(selected_clips, list) or
            any(not isinstance(value, str) for value in selected_clips)):
        raise RuntimeError("ordinal source selected-clip contract is invalid")
    selected_sequences, scope = _select_sequences(dataset, selected_clips)
    expected_source_frame_count = sum(
        sequence["frame_count"] for sequence in selected_sequences
    )
    if (dataset_identity != {
            "path": str(dataset["path"]),
            "sha256": dataset["sha256"],
            "production_id": dataset["production_id"],
            "split": dataset["split"],
            "source_kind": dataset["source_kind"],
            } or contract.get("scope") != scope or
            summary.get("scope") != scope or
            summary.get("selected_clips") != selected_clips or
            contract.get("selection_mode") != SELECTION_MODE or
            summary.get("selection_mode") != SELECTION_MODE or
            contract.get("source_frame_count") !=
            expected_source_frame_count or
            summary.get("source_frame_count") !=
            expected_source_frame_count):
        raise RuntimeError("ordinal source dataset identity is stale")
    variant = contract.get("input_variant")
    input_color.validate_input_variant(variant)
    variant_hash = input_color.input_variant_sha256(variant)
    if (variant_hash != contract.get("input_variant_sha256") or
            variant_hash != summary.get("input_variant_sha256")):
        raise RuntimeError("ordinal source input-variant identity is stale")
    selected_sequence_by_clip = {
        sequence["clip"]: sequence for sequence in selected_sequences
    }
    expected_source_paths = {}
    expected_frame_rates = {}
    for sequence in selected_sequences:
        for frame_id, path in _source_frames(
                dataset["root"] / sequence["clip"],
                sequence["frame_count"],
                ).items():
            key = (sequence["clip"], frame_id)
            expected_source_paths[key] = path
            expected_frame_rates[key] = sequence["source_frame_rate"]
    full_source_cadence = contract.get("full_source_cadence")
    if (not isinstance(full_source_cadence, dict) or
            set(full_source_cadence) != set(selected_sequence_by_clip) or
            summary.get("full_source_cadence_sha256") !=
            canonical_sha256(full_source_cadence)):
        raise RuntimeError("ordinal full source cadence identity is missing")
    for clip, sequence in selected_sequence_by_clip.items():
        frame_ids = list(range(sequence["frame_count"]))
        identity = full_source_cadence.get(clip)
        if (not isinstance(identity, dict) or
                set(identity) != {
                    "contract", "frame_count", "frame_ids",
                    "frame_ids_sha256", "content_sha256",
                } or
                identity.get("contract") != SOURCE_CADENCE_CONTRACT or
                identity.get("frame_count") != len(frame_ids) or
                identity.get("frame_ids") != frame_ids or
                identity.get("frame_ids_sha256") !=
                ordinal_bundle.run_eval.frame_id_sequence_sha256(frame_ids) or
                not isinstance(identity.get("content_sha256"), str) or
                SHA256.fullmatch(identity["content_sha256"]) is None):
            raise RuntimeError(
                f"ordinal full source cadence identity differs: {clip}"
            )
        if verify_media:
            frames = {
                frame_id: expected_source_paths[(clip, frame_id)]
                for frame_id in frame_ids
            }
            if identity != _source_cadence_identity(frames):
                raise RuntimeError(
                    f"ordinal full source cadence content differs: {clip}"
                )
    bundle_records = contract.get("ordinal_bundles")
    if not isinstance(bundle_records, list) or not bundle_records:
        raise RuntimeError("ordinal source contract has no bundle identities")
    bundle_frames = {}
    bundle_scenes = {}
    bundle_paths = {}
    for identity in bundle_records:
        if not isinstance(identity, dict):
            raise RuntimeError("ordinal source bundle identity is invalid")
        path = Path(identity.get("path", "")).resolve(strict=True)
        records = ordinal_bundle.validate_frame_label_bundle(path)
        header = records[0]
        summary_file = Path(identity.get("summary", "")).resolve(strict=True)
        sequence = selected_sequence_by_clip.get(header["clip"])
        source_frame_ids = (
            list(range(sequence["frame_count"])) if sequence is not None else []
        )
        if (identity != {
                "clip": header["clip"],
                "path": str(path),
                "sha256": sha256_file(path),
                "summary": str(summary_file),
                "summary_sha256": sha256_file(summary_file),
                "frame_count": len(records) - 2,
                "source_frame_count": len(source_frame_ids),
                "source_frame_ids_sha256": header[
                    "source_frame_ids_sha256"
                ],
                "label_frame_ids": header["label_frame_ids"],
                "output_frame_count": header["output_frame_count"],
                "output_selected_frame_ids": header[
                    "output_selected_frame_ids"
                ],
                "output_label_frames_sha256": header[
                    "output_label_frames_sha256"
                ],
                "deployment_geometry_allowlist_sha256": header[
                    "deployment_geometry_allowlist_sha256"
                ],
                "runtime_scene_trace_sha256": header[
                    "runtime_scene_trace_sha256"
                ],
                } or header["input_variant"] != variant or
                header["input_variant_sha256"] != variant_hash or
                header["source_frame_ids"] != source_frame_ids or
                header["source_frame_count"] != len(source_frame_ids)):
            raise RuntimeError(f"ordinal source bundle identity is stale: {path}")
        if header["clip"] in bundle_paths:
            raise RuntimeError("ordinal source contract repeats a bundle clip")
        if _model_geometry(header) != contract.get("model_geometry"):
            raise RuntimeError("ordinal source production model geometry differs")
        bundle_paths[header["clip"]] = (path, identity["sha256"], header)
        for scene in header["runtime_scene_trace"]:
            key = (header["clip"], scene["source_frame_id"])
            if key in bundle_scenes:
                raise RuntimeError("ordinal runtime scene identity repeats")
            bundle_scenes[key] = scene
        for frame in records[1:-1]:
            key = (header["clip"], frame["frame_id"])
            if key in bundle_frames:
                raise RuntimeError("ordinal source bundle frame identity repeats")
            bundle_frames[key] = frame
    expected_clip_counts = {
        identity["clip"]: identity["frame_count"]
        for identity in bundle_records
    }
    expected_label_frame_count = sum(expected_clip_counts.values())
    expected_row_count = expected_label_frame_count
    expected_context_row_count = 0
    expected_output_frame_count = sum(
        identity["output_frame_count"] for identity in bundle_records
    )
    if (set(expected_clip_counts) != {
            item["clip"] for item in selected_sequences} or
            set(bundle_paths) != set(expected_clip_counts) or
            {clip: sum(key[0] == clip for key in bundle_frames)
             for clip in bundle_paths} != expected_clip_counts or
            contract.get("label_frame_count") !=
            expected_label_frame_count or
            summary.get("label_frame_count") !=
            expected_label_frame_count or
            contract.get("row_count") != expected_row_count or
            summary.get("row_count") != expected_row_count or
            contract.get("target_row_count") !=
            expected_label_frame_count or
            summary.get("target_row_count") !=
            expected_label_frame_count or
            contract.get("context_row_count") !=
            expected_context_row_count or
            summary.get("context_row_count") !=
            expected_context_row_count or
            contract.get("output_frame_count") !=
            expected_output_frame_count or
            summary.get("output_frame_count") !=
            expected_output_frame_count):
        raise RuntimeError("ordinal source bundle coverage differs from dataset")
    if set(bundle_scenes) != set(expected_source_paths):
        raise RuntimeError("ordinal runtime scene coverage differs from dataset")
    native_sources = contract.get("native_hdr_sources")
    if not isinstance(native_sources, dict):
        raise RuntimeError("ordinal native-HDR source identity is invalid")
    native = variant["kind"] == input_color.INPUT_KIND_NATIVE_PQ
    if native != (dataset["source_kind"] == "native-hdr-video"):
        raise RuntimeError("ordinal source dataset/input runtime regime differs")
    native_authentication = {}
    if native:
        if set(native_sources) != set(expected_clip_counts):
            raise RuntimeError("ordinal native-HDR clip coverage differs")
        for clip in sorted(expected_clip_counts):
            authentication = native_hdr_capture.validate_clip(
                dataset["root"] / clip, full=verify_media
            )
            expected_native_ids = {
                frame_id for source_clip, frame_id in expected_source_paths
                if source_clip == clip
            }
            if set(authentication["frames"]) != expected_native_ids:
                raise RuntimeError(
                    f"ordinal native-HDR cadence differs: {clip}"
                )
            for frame_id in sorted(expected_native_ids):
                native_frame = authentication["frames"][frame_id]
                expected_preview = expected_source_paths[(clip, frame_id)]
                if native_frame["preview_path"] != expected_preview:
                    raise RuntimeError(
                        f"ordinal native-HDR preview join differs: {clip}/{frame_id}"
                    )
            expected_identity = {
                key: authentication[key] for key in (
                    "manifest", "manifest_sha256", "content_sha256",
                    "width", "height", "frame_count",
                )
            }
            expected_identity["verification"] = "full"
            if native_sources[clip] != expected_identity:
                raise RuntimeError(
                    f"ordinal native-HDR identity differs: {clip}"
                )
            native_authentication[clip] = authentication
    elif native_sources:
        raise RuntimeError("non-native ordinal source carries HDR identities")
    expected_keys = []
    for row_number, row in enumerate(rows, 1):
        origin = f"{labels_path}:{row_number}"
        if (not isinstance(row, dict) or
                row.get("source_schema") != SOURCE_SCHEMA or
                row.get("source_contract") != SOURCE_CONTRACT or
                row.get("selection_mode") != SELECTION_MODE or
                row.get("row_role") not in ROW_ROLES or
                row.get("split") not in WORKING_SPLITS or
                row.get("input_variant") != variant or
                row.get("input_variant_sha256") != variant_hash or
                row.get("depth_input_color_contract_sha256") !=
                input_color.color_contract_sha256()):
            raise RuntimeError(f"{origin}: ordinal source row contract differs")
        for field in (
                "source", "source_sha256", "source_width", "source_height",
                "model_input_width", "model_input_height", "clip", "frame",
                "source_ordinal", "source_frame_rate", "row_role",
                "runtime_scene_id", "runtime_scene_evidence",
                "runtime_scene_trace_sha256",
                "film_id", "production_id", "source_kind", "domain",
                "global_policy_weight", "ordinal_bundle",
                "ordinal_bundle_sha256"):
            if field not in row:
                raise RuntimeError(f"{origin}: source row lacks {field}")
        if (row["film_id"] != row["production_id"] or
                row["production_id"] != summary.get("production_id") or
                row["split"] != summary.get("split") or
                row["source_kind"] != summary.get("source_kind")):
            raise RuntimeError(f"{origin}: dataset identity differs")
        for field in (
                "source_width", "source_height", "model_input_width",
                "model_input_height", "frame", "source_ordinal"):
            if (not isinstance(row[field], int) or isinstance(row[field], bool) or
                    row[field] < (0 if field in {
                        "frame", "source_ordinal"
                    } else 1)):
                raise RuntimeError(f"{origin}: invalid {field}")
        if (row["model_input_width"] % 14 or row["model_input_height"] % 14):
            raise RuntimeError(f"{origin}: model input is not patch aligned")
        source = Path(row["source"])
        row_key = (row["clip"], row["frame"])
        if source.resolve(strict=False) != expected_source_paths.get(row_key):
            raise RuntimeError(f"{origin}: source path differs from dataset")
        if (not isinstance(row["source_frame_rate"], (int, float)) or
                isinstance(row["source_frame_rate"], bool) or
                not math.isfinite(float(row["source_frame_rate"])) or
                float(row["source_frame_rate"]) <= 0.0 or
                float(row["source_frame_rate"]) !=
                float(expected_frame_rates.get(row_key, -1.0))):
            raise RuntimeError(f"{origin}: source cadence differs from dataset")
        if verify_media and (
                not source.is_file() or sha256_file(source) != row["source_sha256"] or
                _image_dimensions(source) !=
                (row["source_width"], row["source_height"])):
            raise RuntimeError(f"{origin}: source image is missing or changed")
        key = row_key
        target = bundle_frames.get(key)
        runtime_scene = bundle_scenes.get(key)
        expected_bundle = bundle_paths.get(row["clip"])
        if (expected_bundle is None or runtime_scene is None or
                Path(row["ordinal_bundle"]).resolve(strict=False) !=
                expected_bundle[0] or
                row["ordinal_bundle_sha256"] != expected_bundle[1] or
                row["runtime_scene_evidence"] != runtime_scene or
                row["runtime_scene_id"] !=
                runtime_scene["runtime_scene_id"] or
                row["runtime_scene_trace_sha256"] !=
                expected_bundle[2]["runtime_scene_trace_sha256"] or
                row["source_ordinal"] !=
                runtime_scene["source_frame_ordinal"]):
            raise RuntimeError(f"{origin}: ordinal bundle is missing or changed")
        target_fields = {
            "label_ordinal", "ordinal_frame_model_input_provenance",
            "ordinal_frame_model_input_provenance_sha256",
            "ordinal_model_depth_artifact_sha256",
            "deployment_geometry_allowlist_sha256",
        }
        if row["row_role"] == "target":
            provenance = row.get("ordinal_frame_model_input_provenance")
            if (target is None or not target_fields.issubset(row) or
                    not isinstance(provenance, dict) or
                    provenance != target["model_input_provenance"] or
                    row.get("label_ordinal") != target["ordinal"] or
                    row["source_ordinal"] != target["source_ordinal"] or
                    provenance.get("source_artifact_sha256") !=
                    row["source_sha256"] or
                    provenance.get("input_variant_sha256") != variant_hash or
                    row.get(
                        "ordinal_frame_model_input_provenance_sha256"
                    ) != target["model_input_provenance_sha256"] or
                    row.get("ordinal_model_depth_artifact_sha256") !=
                    target["model_depth_artifact_sha256"] or
                    row.get("deployment_geometry_allowlist_sha256") !=
                    expected_bundle[2][
                        "deployment_geometry_allowlist_sha256"
                    ]):
                raise RuntimeError(f"{origin}: target provenance differs")
        if native:
            model_source = Path(row.get("model_source", ""))
            expected_size = row["source_width"] * row["source_height"] * 8
            native_frame = native_authentication[row["clip"]]["frames"][
                row["frame"]
            ]
            if (row.get("model_source_encoding") !=
                    native_hdr_capture.CAPTURE_ENCODING or
                    model_source.resolve(strict=False) !=
                    native_frame["model_path"] or
                    row.get("model_source_sha256") != native_frame["sha256"] or
                    row.get("native_hdr_source_manifest") !=
                    native_authentication[row["clip"]]["manifest"] or
                    row.get("native_hdr_source_manifest_sha256") !=
                    native_authentication[row["clip"]]["manifest_sha256"] or
                    not model_source.is_file() or
                    model_source.stat().st_size != expected_size):
                raise RuntimeError(f"{origin}: native-PQ model source differs")
        elif any(field in row for field in (
                "model_source", "model_source_sha256",
                "model_source_encoding", "native_hdr_source_manifest",
                "native_hdr_source_manifest_sha256")):
            raise RuntimeError(f"{origin}: non-native row has model source")
        expected_keys.append((row["clip"], row["frame"]))
    expected_keys_sorted = sorted(expected_keys)
    if expected_keys != expected_keys_sorted or len(expected_keys) != len(
            set(expected_keys)):
        raise RuntimeError("ordinal source rows are reordered or duplicated")
    if summary.get("clip_counts") != dict(sorted(Counter(
            row["clip"] for row in rows
            ).items())):
        raise RuntimeError("ordinal source clip counts differ")
    target_rows = [row for row in rows if row["row_role"] == "target"]
    context_rows = [row for row in rows if row["row_role"] == "context"]
    if (summary.get("target_clip_counts") != dict(sorted(Counter(
            row["clip"] for row in target_rows
            ).items())) or
            summary.get("context_clip_counts") != dict(sorted(Counter(
                row["clip"] for row in context_rows
            ).items()))):
        raise RuntimeError("ordinal source role counts differ")
    if summary.get("source_frame_rates") != sorted({
            row["source_frame_rate"] for row in rows
            }):
        raise RuntimeError("ordinal source cadence summary differs")
    if (set(expected_keys) != set(bundle_frames) or
            {(row["clip"], row["frame"]) for row in target_rows} !=
            set(bundle_frames) or
            len(target_rows) != expected_label_frame_count or
            len(context_rows) != expected_context_row_count or
            summary.get("native_hdr_rows") != sum(
                "model_source" in row for row in rows
            )):
        raise RuntimeError("ordinal source aggregate coverage differs")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument(
        "--ordinal-bundle", required=True, action="append", type=Path
    )
    parser.add_argument(
        "--clip", action="append",
        help=(
            "authenticated manifest clip subset for smoke generation; "
            "repeat per selected clip"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = publish(
        args.dataset_manifest, args.ordinal_bundle, args.output,
        overwrite=args.overwrite, selected_clips=args.clip,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
