#!/usr/bin/env python3
"""Generate provenance-checked Apollo depth maps for a stereo training suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SBSBENCH_DIR = THIS_DIR.parent / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))
import build_clip_hash_manifest as clip_hashes  # noqa: E402
import depth_input_color as input_color  # noqa: E402
import native_hdr_capture  # noqa: E402
import sbs_harness_contract as harness_contract  # noqa: E402


SOURCE_IDENTITY_MANIFEST = "clip-hash-manifest"
SOURCE_IDENTITY_FINGERPRINT = "source-fingerprint-sha256"
SOURCE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEPTH_RUN_MANIFEST_SCHEMA = 7
DEPTH_ARTIFACT_CONTENT_CONTRACT = "apollo-artistic-depth-artifacts-v2"
MODEL_ASSET_IDENTITY_CONTRACT = "apollo-depth-model-assets-v1"
ARTIFACT_IDENTITY_FIELDS = (
    "artifact_content_contract",
    "artifact_content_sha256",
    "artifact_files",
)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_content_identity(path: Path):
    """Hash one file and reject replacement or mutation during the read."""
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    fields = ("st_size", "st_mtime_ns", "st_dev", "st_ino")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {"size": after.st_size, "sha256": digest}


def selected_depth_engine_recipe():
    """Read the recipe suffix used by ``models::engine_filename()``."""
    recipe_header = THIS_DIR.parents[1] / "src" / "model_manager.h"
    try:
        source = recipe_header.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"cannot read depth-engine recipe contract: {recipe_header}"
        ) from error
    recipes = re.findall(
        r'inline\s+constexpr\s+char\s+depth_engine_recipe\[\]\s*=\s*'
        r'"([A-Za-z0-9._-]+)"\s*;',
        source,
    )
    if len(recipes) != 1:
        raise RuntimeError(
            f"cannot resolve one depth-engine recipe from {recipe_header}"
        )
    return recipes[0]


def selected_depth_model_identity(executable: Path, model: str):
    """Hash the exact recipe-specific TensorRT plan selected by Apollo.

    The depth engine is an external build asset, so the executable hash alone
    cannot identify inference output.  ``model_manager.h`` is the canonical
    source of the recipe suffix used by ``engine_filename()``.  Fail closed if
    that contract or its selected plan cannot be resolved.  The source ONNX is
    optional at runtime, but bind it as additional provenance when present.
    """
    if (not isinstance(model, str) or not model or model in {".", ".."} or
            Path(model).name != model or "/" in model or "\\" in model):
        raise RuntimeError(f"unsafe depth model name: {model!r}")
    recipe = selected_depth_engine_recipe()
    assets = executable.parent / "assets"
    engine = assets / f"{model}.{recipe}.engine"
    if not engine.is_file():
        raise RuntimeError(
            f"missing selected TensorRT depth plan: {engine}"
        )
    engine_identity = stable_file_content_identity(engine)
    onnx = assets / f"{model}.onnx"
    identity = {
        "contract": MODEL_ASSET_IDENTITY_CONTRACT,
        "model": model,
        "engine_recipe": recipe,
        "engine_file": engine.name,
        "engine_size": engine_identity["size"],
        "engine_sha256": engine_identity["sha256"],
        "onnx_file": None,
        "onnx_size": None,
        "onnx_sha256": None,
    }
    if onnx.is_file():
        onnx_identity = stable_file_content_identity(onnx)
        identity.update({
            "onnx_file": onnx.name,
            "onnx_size": onnx_identity["size"],
            "onnx_sha256": onnx_identity["sha256"],
        })
    return identity


def eval_semantic_file_hash(path: Path):
    """Match run_eval.py's configuration identity exactly."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    digest = hashlib.sha256()
    digest.update(path.name.encode())
    digest.update(data)
    return digest.hexdigest()[:16]


def exact_numeric_files(root: Path, prefix: str, extensions, ignored_names=()):
    """Return numeric files for one strict prefix/extension contract."""
    allowed = {extension.lower() for extension in extensions}
    ignored = set(ignored_names)
    result = {}
    if not root.is_dir():
        return result
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)(\.[^.]+)$")
    for path in root.iterdir():
        if not path.is_file() or not path.name.startswith(prefix):
            continue
        if path.name in ignored:
            continue
        match = pattern.fullmatch(path.name)
        if match is None or match.group(2).lower() not in allowed:
            raise RuntimeError(
                f"invalid {prefix} artifact name or extension: {path}"
            )
        frame_id = int(match.group(1))
        if frame_id in result:
            raise RuntimeError(
                f"duplicate numeric {prefix} identity {frame_id}: "
                f"{result[frame_id]} and {path}"
            )
        result[frame_id] = path
    return result


def source_frame_files(source: Path):
    return exact_numeric_files(
        source, "frame_", SOURCE_IMAGE_EXTENSIONS,
        ignored_names=(native_hdr_capture.MANIFEST_NAME,),
    )


def depth_artifact_records(output: Path):
    """Return the exact content identity published for one depth clip.

    Logs and timing summaries are deliberately excluded.  Downstream source
    rows consume the generation/warp contracts plus the selected depth and
    exact/unclamped disparity artifacts, so only those files form the reusable
    supervision identity.
    """
    fixed = [output / "contract.json", output / "generation_identity.json"]
    hdr_stats = output / "hdr_output_stats.json"
    if hdr_stats.is_file():
        fixed.append(hdr_stats)
    missing = [path for path in fixed if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing depth artifact identity files: {missing}")
    groups = (
        exact_numeric_files(output, "depth_", {".png"}),
        exact_numeric_files(output, "baseline_disparity_", {".f32"}),
        exact_numeric_files(
            output, "baseline_unclamped_disparity_", {".f32"}
        ),
    )
    paths = list(fixed)
    for group in groups:
        paths.extend(group[frame_id] for frame_id in sorted(group))
    records = []
    for path in sorted(paths, key=lambda item: item.name):
        identity = stable_file_content_identity(path)
        records.append({
            "path": path.name,
            **identity,
        })
    return records


def depth_artifact_identity(output: Path):
    records = depth_artifact_records(output)
    payload = {
        "contract": DEPTH_ARTIFACT_CONTENT_CONTRACT,
        "files": records,
    }
    return {
        "artifact_content_contract": DEPTH_ARTIFACT_CONTENT_CONTRACT,
        "artifact_content_sha256": clip_hashes.canonical_json_sha256(payload),
        "artifact_files": records,
    }


def artifact_identity_matches(row, identity):
    """Match a freshly hashed clip against its previously published identity."""
    return isinstance(row, dict) and all(
        row.get(field) == identity[field] for field in ARTIFACT_IDENTITY_FIELDS
    )


def reusable_artifact_rows(manifest_path: Path, model_asset_identity,
                           input_variant=None):
    """Load authenticated cache rows only from the current model-asset contract.

    Missing, legacy, or malformed publication manifests deliberately yield no
    reusable rows.  Their clip directories may still exist, but without the
    prior published content digest they cannot be safely accepted as cache
    hits.
    """
    input_variant = input_variant or input_color.sdr_input_variant()
    input_color.validate_input_variant(input_variant)
    metric_preview_encoding = (
        harness_contract.input_variant_metric_preview_encoding(input_variant)
    )
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    if (not isinstance(payload, dict) or
            payload.get("schema") != DEPTH_RUN_MANIFEST_SCHEMA or
            isinstance(payload.get("schema"), bool) or
            payload.get("model_asset_identity") != model_asset_identity or
            payload.get("model_asset_identity_sha256") !=
            clip_hashes.canonical_json_sha256(model_asset_identity) or
            payload.get("input_variant") != input_variant or
            payload.get("input_variant_sha256") !=
            input_color.input_variant_sha256(input_variant) or
            payload.get("metric_preview_encoding") !=
            metric_preview_encoding):
        return {}
    rows = payload.get("clips")
    if not isinstance(rows, list):
        return {}
    result = {}
    for row in rows:
        name = row.get("clip") if isinstance(row, dict) else None
        if (not isinstance(name, str) or not name or name in result or
                Path(name).name != name or "/" in name or "\\" in name or
                not all(field in row for field in ARTIFACT_IDENTITY_FIELDS)):
            return {}
        result[name] = row
    return result


def paths_overlap(left: Path, right: Path):
    return (
        left == right or left.is_relative_to(right) or right.is_relative_to(left)
    )


def remove_path(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def publish_staged_directory(staging: Path, destination: Path):
    """Swap one validated clip into place, rolling back a failed rename."""
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


def write_json_atomic(path: Path, payload):
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_sequence_paths(suite: Path, output: Path, sequences):
    """Authenticate all source/destination paths before any destructive cleanup."""
    suite_root = suite.resolve(strict=True)
    output_root = output.resolve(strict=False)
    if paths_overlap(suite_root, output_root):
        raise RuntimeError(
            f"source and output paths overlap: {suite_root} and {output_root}"
        )
    names = []
    resolved = {}
    for row in sequences:
        if not isinstance(row, dict):
            raise RuntimeError("dataset sequence rows must be objects")
        name = row.get("clip")
        if (not isinstance(name, str) or not name or name in {".", ".."} or
                Path(name).name != name or "/" in name or "\\" in name):
            raise RuntimeError(f"unsafe dataset clip name: {name!r}")
        if name in resolved:
            raise RuntimeError(f"duplicate dataset clip name: {name}")
        source = suite / name
        if not source.is_dir():
            raise RuntimeError(f"missing source clip directory: {source}")
        source_resolved = source.resolve(strict=True)
        destination = output / name
        destination_resolved = destination.resolve(strict=False)
        if (source_resolved.parent != suite_root or
                destination_resolved.parent != output_root):
            raise RuntimeError(f"clip path escapes its declared root: {name}")
        if (source_resolved == destination_resolved or
                source_resolved.is_relative_to(destination_resolved) or
                destination_resolved.is_relative_to(source_resolved)):
            raise RuntimeError(f"source and output clip paths overlap: {name}")
        names.append(name)
        resolved[name] = (source, destination)
    return names, resolved


def source_fingerprint(source: Path):
    paths = source_frame_files(source)
    if not paths:
        raise RuntimeError(f"no numeric PNG/JPEG source frames: {source}")
    frames = [
        {"name": path.name, "sha256": sha256(path)}
        for path in sorted(paths.values(), key=lambda item: item.name)
    ]
    return hashlib.sha256(
        json.dumps(frames, sort_keys=True).encode("utf-8")
    ).hexdigest()


def resolve_source_identities(suite: Path, clip_names, verify_clip_hashes=False):
    """Resolve every selected clip once, preferring the suite's frozen hash manifest."""
    names = list(clip_names)
    if len(names) != len(set(names)):
        raise RuntimeError("dataset manifest contains duplicate clip names")
    manifest_path = suite / clip_hashes.MANIFEST_NAME
    if manifest_path.is_file():
        try:
            initial_manifest = clip_hashes.load_manifest(manifest_path)
            manifest_content_sha256 = initial_manifest[
                clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
            ]
            clip_sha1 = clip_hashes.verify_selected_clips(
                manifest_path, suite, names, full=verify_clip_hashes
            )
            verified_manifest = clip_hashes.load_manifest(manifest_path)
        except clip_hashes.ClipHashManifestError as error:
            raise RuntimeError(
                f"stale clip hash manifest {manifest_path}: {error}"
            ) from error
        if (verified_manifest[clip_hashes.MANIFEST_CONTENT_SHA256_FIELD] !=
                manifest_content_sha256):
            raise RuntimeError(
                f"stale clip hash manifest {manifest_path}: "
                "semantic content changed during verification"
            )
        manifest_file_sha256 = sha256(manifest_path)
        identities = {
            name: {
                "source_identity_method": SOURCE_IDENTITY_MANIFEST,
                "source_identity_value": clip_sha1[name],
                "clip_hash_manifest_content_sha256": manifest_content_sha256,
            }
            for name in names
        }
        provenance = {
            "clip_hash_source": "manifest",
            "clip_hash_verification": "full" if verify_clip_hashes else "stat",
            "clip_hash_manifest": str(manifest_path.resolve()),
            "clip_hash_manifest_content_sha256": manifest_content_sha256,
            "clip_hash_manifest_file_sha256": manifest_file_sha256,
        }
        return identities, provenance

    identities = {
        name: {
            "source_identity_method": SOURCE_IDENTITY_FINGERPRINT,
            "source_identity_value": source_fingerprint(suite / name),
        }
        for name in names
    }
    return identities, {
        "clip_hash_source": "direct",
        "clip_hash_verification": "direct-content",
        "clip_hash_manifest": None,
        "clip_hash_manifest_content_sha256": None,
        "clip_hash_manifest_file_sha256": None,
    }


def source_identity_matches(identity, expected):
    method = expected["source_identity_method"]
    if method == SOURCE_IDENTITY_MANIFEST:
        return (
            identity.get("schema") == 5 and
            identity.get("source_identity_method") == method and
            identity.get("source_identity_value") ==
            expected["source_identity_value"] and
            identity.get("clip_hash_manifest_content_sha256") ==
            expected["clip_hash_manifest_content_sha256"]
        )
    return (
        method == SOURCE_IDENTITY_FINGERPRINT and
        identity.get("schema") == 5 and
        identity.get("source_sha256") == expected["source_identity_value"]
    )


def generation_identity(source_identity, selection, executable_sha, conf_sha,
                        model, input_variant=None):
    input_variant = input_variant or input_color.sdr_input_variant()
    input_color.validate_input_variant(input_variant)
    payload = {
        "schema": 5,
        "executable_sha256": executable_sha,
        "conf_sha256": conf_sha,
        "model": model,
        "output_selection": selection,
        "input_variant": input_variant,
        "input_variant_sha256": input_color.input_variant_sha256(input_variant),
    }
    if source_identity["source_identity_method"] == SOURCE_IDENTITY_MANIFEST:
        payload.update(source_identity)
    else:
        payload["source_sha256"] = source_identity["source_identity_value"]
    return payload


def load_label_frame_ids(source: Path):
    path = source / "label_frames.json"
    if not path.is_file():
        return None, ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid label-frame manifest: {path}") from error
    if (not isinstance(payload, dict) or set(payload) != {"schema", "frame_ids"} or
            payload.get("schema") != 1 or isinstance(payload.get("schema"), bool)):
        raise RuntimeError(
            f"{path}: expected exactly schema=1 and frame_ids"
        )
    labels = payload.get("frame_ids")
    if (not isinstance(labels, list) or not labels or
            any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in labels) or
            labels != sorted(set(labels))):
        raise RuntimeError(
            f"{path}: frame_ids must be strictly increasing nonnegative integers"
        )
    return labels, sha256(path)


def gt_right_output_selection(source: Path):
    """Return an exact authenticated selection from ``gt_right/frame_*``."""
    if (source / "label_frames.json").is_file():
        raise RuntimeError(
            f"{source}: GT-right selection cannot override label_frames.json"
        )
    gt_root = source / "gt_right"
    if not gt_root.is_dir():
        raise RuntimeError(f"missing GT-right directory: {gt_root}")
    try:
        if gt_root.resolve(strict=True).parent != source.resolve(strict=True):
            raise RuntimeError(f"GT-right directory escapes its source clip: {gt_root}")
    except OSError as error:
        raise RuntimeError(f"cannot authenticate GT-right directory: {gt_root}") from error
    source_ids = set(source_frame_files(source))
    gt_paths = exact_numeric_files(gt_root, "frame_", SOURCE_IMAGE_EXTENSIONS)
    gt_ids = sorted(gt_paths)
    if not gt_ids:
        raise RuntimeError(f"GT-right selection contains no frames: {gt_root}")
    missing = sorted(set(gt_ids) - source_ids)
    if missing:
        raise RuntimeError(
            f"GT-right selection references missing source RGB: {missing}"
        )
    return {
        "mode": "gt-right",
        "label_frame_ids": [],
        "output_frame_ids": gt_ids,
        "label_frames_sha256": "",
    }


def output_selection(source: Path, output_gt_right_only=False):
    """Return authenticated target IDs and sparse adjacent evidence IDs."""
    if output_gt_right_only:
        return gt_right_output_selection(source)
    source_ids = set(source_frame_files(source))
    if not source_ids:
        raise RuntimeError(f"no numeric source frames: {source}")
    labels, manifest_sha256 = load_label_frame_ids(source)
    if labels is None:
        return {
            "mode": "interval",
            "label_frame_ids": [],
            "output_frame_ids": sorted(source_ids),
            "label_frames_sha256": "",
        }
    missing = sorted(set(labels) - source_ids)
    if missing:
        raise RuntimeError(f"label-frame manifest references missing RGB: {missing}")
    selected = set(labels)
    for frame_id in labels:
        if frame_id - 1 in source_ids:
            selected.add(frame_id - 1)
        elif frame_id + 1 in source_ids:
            selected.add(frame_id + 1)
        else:
            raise RuntimeError(
                f"label frame {frame_id} has no consecutive source-frame companion"
            )
    return {
        "mode": "label-frames",
        "label_frame_ids": labels,
        "output_frame_ids": sorted(selected),
        "label_frames_sha256": manifest_sha256,
    }


def manifest_is_authored_stereo(payload):
    """Recognize current or legacy manifests without guessing from paths."""
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if type(schema) is not int:
        return False
    if schema == 2:
        return payload.get("source_kind") == "authored-stereo"
    return (
        schema == 1 and
        payload.get("layout") in {"above-below", "side-by-side"} and
        payload.get("eye_order") in {"first-left", "first-right"}
    )


def valid_completed_clip(source: Path, output: Path, model: str,
                         executable_sha256=None, conf_sha256=None,
                         source_identity=None, output_gt_right_only=False,
                         input_variant=None):
    input_variant = input_variant or input_color.sdr_input_variant()
    input_color.validate_input_variant(input_variant)
    metric_preview_encoding = (
        harness_contract.input_variant_metric_preview_encoding(input_variant)
    )
    contract_path = output / "contract.json"
    identity_path = output / "generation_identity.json"
    if not contract_path.is_file() or not identity_path.is_file():
        return False
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    try:
        selection = output_selection(source, output_gt_right_only)
        output_ids = set(exact_numeric_files(output, "depth_", {".png"}))
        disparity_ids = set(exact_numeric_files(
            output, "baseline_disparity_", {".f32"}
        ))
        unclamped_disparity_ids = set(exact_numeric_files(
            output, "baseline_unclamped_disparity_", {".f32"}
        ))
    except RuntimeError:
        return False
    if source_identity is None:
        source_identity = {
            "source_identity_method": SOURCE_IDENTITY_FINGERPRINT,
            "source_identity_value": source_fingerprint(source),
        }
    selected_ids = set(selection["output_frame_ids"])
    current_identity = (
        identity.get("schema") == 5 and
        identity.get("output_selection") == selection and
        identity.get("input_variant") == input_variant and
        identity.get("input_variant_sha256") ==
        input_color.input_variant_sha256(input_variant)
    )
    hdr_stats_path = output / "hdr_output_stats.json"
    hdr_stats_valid = not hdr_stats_path.exists()
    hdr_source_kind = harness_contract.input_variant_hdr_source_kind(
        input_variant
    )
    if input_variant["color_mode"] == input_color.COLOR_MODE_HDR:
        try:
            hdr_stats = json.loads(hdr_stats_path.read_text(encoding="utf-8"))
            stats_scale = float(hdr_stats.get("input_scale", 0.0))
        except (OSError, TypeError, ValueError):
            hdr_stats = {}
            stats_scale = 0.0
        hdr_stats_valid = (
            hdr_stats.get("format") == "linear-scRGB-fp16" and
            hdr_stats.get("hdr_source_kind") == hdr_source_kind and
            hdr_stats.get("sdr_white_level_raw", 0) ==
            int(input_variant["windows_sdr_white_level_raw"] or 0) and
            stats_scale == float(input_variant["scrgb_white_scale"] or 0.0)
        )
    selection_contract_matches = (
        contract.get("output_selection_mode") == selection["mode"]
        and contract.get("label_frame_ids") == selection["label_frame_ids"]
        and contract.get("output_selected_frame_ids") == selection["output_frame_ids"]
        and contract.get("output_label_frames_sha256") ==
        selection["label_frames_sha256"]
    )
    return (
        contract.get("model") == model
        and contract.get("schema") == harness_contract.HARNESS_SCHEMA
        and contract.get("artifact_mode") == "depth+baseline-disparity"
        and contract.get("depth_step") == "current-once"
        and contract.get("artistic_policy") is False
        and contract.get("artistic_policy_consumed") is False
        and contract.get("artistic_policy_authorization") == "none"
        and contract.get("model_onnx_sha256") == ""
        and contract.get("policy_metadata_sha256") == ""
        and contract.get("deployment_geometry_allowlist_sha256") == ""
        and float(contract.get("artistic_scale_override", 0.0)) == 0.0
        and contract.get("color_mode") == input_variant["color_mode"]
        and contract.get("metric_preview_encoding") ==
        metric_preview_encoding
        and contract.get("hdr_source_kind") == hdr_source_kind
        and float(contract.get("hdr_input_scale", 0.0)) ==
        float(input_variant["scrgb_white_scale"] or 0.0)
        and contract.get("sdr_white_level_raw", 0) ==
        int(input_variant["windows_sdr_white_level_raw"] or 0)
        and contract.get("warp_disparity") ==
        "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars"
        and contract.get("warp_unclamped_disparity") ==
        "unclamped_full_binocular_normalized_at_artistic_scale_1_"
        "output_eye_raster_zero_bars"
        and contract.get("artistic_disparity_contract") ==
        "clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_"
        "times_aspect_scale_times_content_scale_x)"
        and int(contract.get("source_width", 0)) > 0
        and int(contract.get("source_height", 0)) > 0
        and int(contract.get("eye_width", 0)) > 0
        and int(contract.get("eye_height", 0)) > 0
        and int(contract.get("disparity_raster_width", 0)) > 0
        and int(contract.get("disparity_raster_height", 0)) > 0
        and int(contract["disparity_raster_width"]) == int(contract["eye_width"])
        and int(contract["disparity_raster_height"]) == int(contract["eye_height"])
        and isinstance(contract.get("policy_warp_source_sha256"), str)
        and len(contract["policy_warp_source_sha256"]) == 64
        and float(contract.get("artistic_full_clamp_abs", 0.0)) > 0.0
        and selected_ids == output_ids == disparity_ids == unclamped_disparity_ids
        and selection_contract_matches
        and current_identity
        and hdr_stats_valid
        and source_identity_matches(identity, source_identity)
        and (executable_sha256 is None or
             identity.get("executable_sha256") == executable_sha256)
        and (conf_sha256 is None or identity.get("conf_sha256") == conf_sha256)
    )


def generate(suite: Path, output: Path, executable: Path, conf: Path,
             model: str, timeout: int, resume: bool,
             verify_clip_hashes=False, output_gt_right_only=False,
             input_variant=None):
    input_variant = input_variant or input_color.sdr_input_variant()
    input_color.validate_input_variant(input_variant)
    metric_preview_encoding = (
        harness_contract.input_variant_metric_preview_encoding(input_variant)
    )
    suite_manifest_path = suite / "dataset_manifest.json"
    if not suite_manifest_path.is_file():
        raise RuntimeError(f"missing dataset manifest: {suite_manifest_path}")
    suite_manifest = json.loads(suite_manifest_path.read_text(encoding="utf-8"))
    if output_gt_right_only and not manifest_is_authored_stereo(suite_manifest):
        raise RuntimeError(
            "--output-gt-right-only requires an authenticated authored-stereo "
            "dataset manifest"
        )
    sequences = suite_manifest.get("sequences", [])
    if not sequences and suite_manifest.get("shots"):
        # Schema-1 movie manifests written before sequences became part of the
        # common depth-run contract can be consumed without re-extracting video.
        domain = suite_manifest.get("domain")
        if not domain:
            raise RuntimeError("movie manifest has shots but no domain")
        sequences = [
            {
                "clip": f"{domain}_shot_{int(row['shot']):04d}",
                "frames": row.get("samples"),
                "split": row.get("split"),
            }
            for row in suite_manifest["shots"]
        ]
    if not sequences:
        raise RuntimeError("dataset manifest contains no sequences or movie shots")
    clip_names, sequence_paths = validate_sequence_paths(suite, output, sequences)
    executable_sha = sha256(executable)
    conf_sha = eval_semantic_file_hash(conf)
    model_asset_identity = selected_depth_model_identity(executable, model)
    model_asset_identity_sha256 = clip_hashes.canonical_json_sha256(
        model_asset_identity
    )
    source_identities, clip_hash_provenance = resolve_source_identities(
        suite, clip_names, verify_clip_hashes
    )
    output.mkdir(parents=True, exist_ok=True)
    publication_manifest = output / "depth_run_manifest.json"
    publication_invalidated = False

    results = []
    selections = {}
    policy_hashes = set()
    metric_hashes = set()
    reusable_rows = reusable_artifact_rows(
        publication_manifest, model_asset_identity, input_variant
    ) if resume else {}
    for row in sequences:
        clip_name = row["clip"]
        source, destination = sequence_paths[clip_name]
        if input_variant["kind"] == input_color.INPUT_KIND_NATIVE_PQ:
            if clip_hash_provenance["clip_hash_source"] != "manifest":
                raise RuntimeError(
                    f"{clip_name}: native HDR requires a frozen clip hash manifest"
                )
            native_hdr_capture.validate_clip(source, full=verify_clip_hashes)
        selection = output_selection(source, output_gt_right_only)
        selections[clip_name] = selection
        source_identity = source_identities[clip_name]
        artifact_identity = None
        cache_contract_matches = resume and valid_completed_clip(
            source, destination, model, executable_sha, conf_sha,
            source_identity, output_gt_right_only, input_variant
        )
        if cache_contract_matches:
            try:
                candidate_identity = depth_artifact_identity(destination)
            except (OSError, RuntimeError, ValueError):
                candidate_identity = None
            previous_row = reusable_rows.get(clip_name)
            if (candidate_identity is not None and
                    artifact_identity_matches(previous_row, candidate_identity)):
                artifact_identity = candidate_identity
            else:
                print(
                    f"[{clip_name}] cached artifact identity differs; regenerate",
                    flush=True,
                )
        if artifact_identity is not None:
            print(f"[{clip_name}] reuse", flush=True)
            status = "reused"
        else:
            staging = Path(tempfile.mkdtemp(
                prefix=f".{clip_name}.partial-", dir=output
            ))
            command = [
                str(executable.resolve()), str(conf.resolve()), "--sbs-bench",
                "--frames", str(source.resolve()), "--out", str(staging.resolve()),
                "--model", model, "--depth-only",
                "--no-artistic-policy",
            ]
            if input_variant["kind"] == input_color.INPUT_KIND_WINDOWS_HDR:
                command += [
                    "--simulate-hdr", "--sdr-white-level-raw",
                    str(input_variant["windows_sdr_white_level_raw"]),
                ]
            elif input_variant["kind"] == input_color.INPUT_KIND_NATIVE_PQ:
                command.append("--native-hdr-scrgb")
            if selection["mode"] == "label-frames":
                command.append("--output-label-frames")
            elif selection["mode"] == "gt-right":
                command.append("--output-gt-right-only")
            print(f"[{clip_name}] depth", flush=True)
            try:
                try:
                    process = subprocess.run(
                        command, cwd=executable.parent, capture_output=True,
                        text=True, timeout=timeout
                    )
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(f"{clip_name}: harness timed out") from error
                if process.returncode != 0:
                    tail = (process.stdout + process.stderr)[-4000:]
                    raise RuntimeError(
                        f"{clip_name}: harness failed ({process.returncode})\n{tail}"
                    )
                identity = generation_identity(
                    source_identity, selection, executable_sha, conf_sha, model,
                    input_variant,
                )
                (staging / "generation_identity.json").write_text(
                    json.dumps(identity, indent=2) + "\n", encoding="utf-8"
                )
                if not valid_completed_clip(
                        source, staging, model, executable_sha, conf_sha,
                        source_identity, output_gt_right_only, input_variant):
                    raise RuntimeError(
                        f"{clip_name}: incomplete or mismatched depth output"
                    )

                # The old manifest remains valid while generation is isolated in
                # staging.  Remove it before the first visible clip replacement;
                # any later failure then leaves an explicitly unpublished run.
                if not publication_invalidated:
                    publication_manifest.unlink(missing_ok=True)
                    publication_invalidated = True
                publish_staged_directory(staging, destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
            status = "generated"
        clip_contract = json.loads(
            (destination / "contract.json").read_text(encoding="utf-8")
        )
        policy_hashes.add(clip_contract["policy_warp_source_sha256"])
        metric_hashes.add(clip_contract["metric_sha256"])
        if artifact_identity is None:
            artifact_identity = depth_artifact_identity(destination)
        result_row = {
            "clip": clip_name,
            "frames": len(exact_numeric_files(destination, "depth_", {".png"})),
            "label_frames": len(selection["label_frame_ids"]),
            "output_selection_mode": selection["mode"],
            "output_label_frames_sha256": selection["label_frames_sha256"],
            "source_identity_method": source_identity["source_identity_method"],
            "source_identity_value": source_identity["source_identity_value"],
            "status": status,
            "metric_preview_encoding": clip_contract[
                "metric_preview_encoding"
            ],
            "contract_sha256": sha256(destination / "contract.json"),
            **artifact_identity,
        }
        if input_variant["kind"] == input_color.INPUT_KIND_NATIVE_PQ:
            native = native_hdr_capture.validate_clip(source, full=False)
            result_row["native_hdr_model_source"] = {
                key: native[key] for key in (
                    "manifest", "manifest_sha256", "content_sha256",
                    "width", "height", "frame_count", "verification",
                )
            }
        results.append(result_row)

    if len(policy_hashes) != 1 or len(metric_hashes) != 1:
        raise RuntimeError("depth run mixed policy-warp or metric contracts")

    if clip_hash_provenance["clip_hash_source"] == "manifest":
        manifest_path = suite / clip_hashes.MANIFEST_NAME
        try:
            final_manifest = clip_hashes.load_manifest(manifest_path)
            final_identities = clip_hashes.verify_selected_clips(
                manifest_path, suite, clip_names, full=verify_clip_hashes
            )
            verified_final_manifest = clip_hashes.load_manifest(manifest_path)
        except clip_hashes.ClipHashManifestError as error:
            raise RuntimeError(
                f"clip hash manifest changed during generation: {error}"
            ) from error
        expected = {
            name: source_identities[name]["source_identity_value"]
            for name in clip_names
        }
        expected_content_sha256 = clip_hash_provenance[
            "clip_hash_manifest_content_sha256"
        ]
        if (final_identities != expected or
                final_manifest[clip_hashes.MANIFEST_CONTENT_SHA256_FIELD] !=
                expected_content_sha256 or
                verified_final_manifest[
                    clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
                ] != expected_content_sha256):
            raise RuntimeError("clip hash manifest identity changed during generation")
    else:
        changed = [
            name for name in clip_names
            if source_fingerprint(sequence_paths[name][0]) !=
            source_identities[name]["source_identity_value"]
        ]
        if changed:
            raise RuntimeError(
                f"source frames changed during depth generation: {changed}"
            )

    changed_selections = [
        name for name in clip_names
        if output_selection(sequence_paths[name][0], output_gt_right_only) !=
        selections[name]
    ]
    if changed_selections:
        raise RuntimeError(
            f"output selection changed during depth generation: {changed_selections}"
        )

    if selected_depth_model_identity(executable, model) != model_asset_identity:
        raise RuntimeError("depth model assets changed during generation")

    manifest = {
        "schema": DEPTH_RUN_MANIFEST_SCHEMA,
        "harness_schema": harness_contract.HARNESS_SCHEMA,
        "purpose": "artistic-policy depth supervision",
        "artifact_contract": {
            "exact": "baseline_disparity_<frame>.f32",
            "unclamped_scale_1": (
                "baseline_unclamped_disparity_<frame>.f32"
            ),
            "scaled": (
                "clamp(unclamped_scale_1 * artistic_scale, comfort_limit)"
            ),
            "clamp_and_source_geometry": "per-clip contract.json",
        },
        "suite": str(suite.resolve()),
        **clip_hash_provenance,
        "source_identities": {
            name: source_identities[name] for name in clip_names
        },
        "suite_manifest_sha256": sha256(suite_manifest_path),
        "executable": str(executable.resolve()),
        "executable_sha256": executable_sha,
        "conf": str(conf.resolve()),
        "conf_sha256": conf_sha,
        "model": model,
        "input_variant": input_variant,
        "input_variant_sha256": input_color.input_variant_sha256(input_variant),
        "metric_preview_encoding": metric_preview_encoding,
        "hdr_source_kind": harness_contract.input_variant_hdr_source_kind(
            input_variant
        ),
        "depth_input_color_contract_sha256": input_color.color_contract_sha256(),
        "output_gt_right_only": bool(output_gt_right_only),
        "model_asset_identity": model_asset_identity,
        "model_asset_identity_sha256": model_asset_identity_sha256,
        "policy_warp_source_sha256": next(iter(policy_hashes)),
        "metric_sha256": next(iter(metric_hashes)),
        "clips": results,
        "clip_count": len(results),
        "frame_count": sum(row["frames"] for row in results),
    }
    write_json_atomic(publication_manifest, manifest)
    return manifest


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--build-dir", type=Path, default=repo / "cmake-build-relwithdebinfo"
    )
    parser.add_argument("--conf", type=Path,
                        default=repo / "tools" / "sbsbench" / "bench.conf")
    parser.add_argument("--model", default="depth_anything_v2_fp16")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--verify-clip-hashes", action="store_true",
        help="fully verify every file hash in an existing clip_hash_manifest.json",
    )
    parser.add_argument(
        "--output-gt-right-only", action="store_true",
        help=(
            "for an authenticated authored-stereo suite without label_frames.json, "
            "materialize only source frame IDs present under gt_right/"
        ),
    )
    parser.add_argument("--simulate-hdr", action="store_true")
    parser.add_argument(
        "--native-hdr-scrgb", action="store_true",
        help=(
            "consume authenticated model_source/frame_*.scrgb16 sidecars; "
            "mutually exclusive with --simulate-hdr"
        ),
    )
    parser.add_argument(
        "--sdr-white-level-raw", type=int,
        choices=input_color.RAW_WHITE_ANCHORS,
        help=(
            "canonical Windows SDR-white raw value; required with --simulate-hdr"
        ),
    )
    args = parser.parse_args()
    if args.simulate_hdr != (args.sdr_white_level_raw is not None):
        parser.error(
            "--simulate-hdr and --sdr-white-level-raw must be supplied together"
        )
    if args.simulate_hdr and args.native_hdr_scrgb:
        parser.error("--simulate-hdr and --native-hdr-scrgb are mutually exclusive")
    if args.native_hdr_scrgb:
        input_variant = input_color.native_pq_input_variant()
    elif args.simulate_hdr:
        input_variant = input_color.windows_hdr_input_variant(
            args.sdr_white_level_raw
        )
    else:
        input_variant = input_color.sdr_input_variant()
    executable = args.build_dir / "sunshine.exe"
    for path, description in (
        (executable, "benchmark executable"), (args.conf, "benchmark config")
    ):
        if not path.is_file():
            raise RuntimeError(f"missing {description}: {path}")
    manifest = generate(
        args.suite, args.output, executable, args.conf, args.model,
        args.timeout, not args.no_resume, args.verify_clip_hashes,
        args.output_gt_right_only, input_variant,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
