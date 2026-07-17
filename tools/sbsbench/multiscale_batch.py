"""Authenticated transport for one offline artistic multiscale harness clip.

The C++ harness owns rendering semantics.  This module binds its common and
per-scale artifacts to the exact executable/config/metric/source identities so
``run_eval`` can score one scale without rerunning DA-V2.  It deliberately has
no live-stream entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct

import runtime_scene_evidence


SCHEMA = 3
CONTRACT = "apollo-authenticated-artistic-multiscale-batch-v3"
HARNESS_SCHEMA = 5
HARNESS_CONTRACT = "apollo-harness-artistic-multiscale-v5"
ARTIFACT_WRITER_CONTRACT = "apollo-bounded-multiscale-png-writer-v1"
MANIFEST = "multiscale_batch_manifest.json"
HARNESS_MANIFEST = "multiscale_contract.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CLIP_ID = re.compile(r"^[0-9a-f]{12}$")
METRIC_ID = re.compile(r"^[0-9a-f]{16}$")
SCALE_ARTIFACTS = (
    "sbs_*.png", "warp_mask_*.png", "warp_disparity_*.f32",
    "warp_unclamped_disparity_*.f32", "contract.json",
)
COMMON_ARTIFACTS = (
    "depth_*.png", "raw_*.f32", "ema_mask_*.png", "raw_shape.json",
    "runtime_scene_evidence.json", "sbs_perf.json", "hdr_output_stats.json",
)


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


def scale_slug(scale):
    value = float(scale)
    rounded = round(value * 100.0)
    if (not math.isfinite(value) or value < 0.5 or value > 1.5 or
            abs(value * 100.0 - rounded) > 1e-5):
        raise ValueError(f"invalid multiscale value: {scale!r}")
    return f"s{rounded:03d}"


def scale_float32_bits(scale):
    return struct.unpack("<I", struct.pack("<f", float(scale)))[0]


def _load_object(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not an object: {path}")
    return value


def _artifact_identity(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"multiscale artifact escapes its clip root: {path}") from error
    stat = path.stat()
    return {
        "path": relative,
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def _glob_artifacts(directory, patterns, root):
    directory = Path(directory)
    paths = {}
    for pattern in patterns:
        for path in directory.glob(pattern):
            if not path.is_file():
                continue
            identity = _artifact_identity(path, root)
            if identity["path"] in paths:
                raise ValueError(f"multiscale artifact repeats: {path}")
            paths[identity["path"]] = identity
    return [paths[key] for key in sorted(paths)]


def _frame_ids(directory, prefix, suffix):
    result = []
    pattern = re.compile(
        rf"^{re.escape(prefix)}([0-9]+){re.escape(suffix)}$", re.IGNORECASE
    )
    for path in Path(directory).iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            result.append(int(match.group(1)))
    return sorted(result)


def _strict_frame_ids(value, description, *, allow_empty=False):
    if (not isinstance(value, list) or
            (not value and not allow_empty) or
            any(not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in value) or
            value != sorted(set(value))):
        raise ValueError(f"multiscale {description} are invalid")
    return list(value)


def _expected_output_ids(source_ids, label_ids):
    """Return the exact target-only sparse selection."""
    if not label_ids:
        return list(source_ids)
    source_set = set(source_ids)
    missing = sorted(set(label_ids) - source_set)
    if missing:
        raise ValueError(
            f"multiscale label frames are absent from the source sequence: {missing}"
        )
    return list(label_ids)


def _validate_harness_selection(harness):
    source_ids = _strict_frame_ids(
        harness.get("source_frame_ids"), "source frame IDs"
    )
    if source_ids != list(range(source_ids[0], source_ids[-1] + 1)):
        raise ValueError("multiscale source frame IDs are not consecutive")
    label_ids = _strict_frame_ids(
        harness.get("label_frame_ids"), "label frame IDs", allow_empty=True
    )
    output_ids = _strict_frame_ids(
        harness.get("output_selected_frame_ids"), "selected output frame IDs"
    )
    if output_ids != _expected_output_ids(source_ids, label_ids):
        raise ValueError("multiscale selected output frame coverage differs")
    selection_mode = harness.get("output_selection_mode")
    label_sha256 = harness.get("output_label_frames_sha256")
    if label_ids:
        if (selection_mode != "label-frames" or
                not isinstance(label_sha256, str) or
                not SHA256.fullmatch(label_sha256)):
            raise ValueError("multiscale sparse label selection identity differs")
    elif selection_mode != "interval" or label_sha256 != "":
        raise ValueError("multiscale full-frame selection identity differs")
    return source_ids, label_ids, output_ids, selection_mode, label_sha256


def _validate_artifact_writer(harness, *, scale_count, output_count):
    """Authenticate the bounded writer's complete, drained artifact set."""
    writer = harness.get("artifact_writer")
    if not isinstance(writer, dict):
        raise ValueError("multiscale artifact-writer contract is missing")
    positive_fields = ("worker_count", "queue_capacity",
                       "maximum_inflight_job_bound")
    if any(not isinstance(writer.get(field), int) or
           isinstance(writer.get(field), bool) or writer[field] <= 0
           for field in positive_fields):
        raise ValueError("multiscale artifact-writer bounds are invalid")
    if (writer.get("contract") != ARTIFACT_WRITER_CONTRACT or
            writer.get("mode") != "bounded-async-worker-owned-buffers" or
            writer.get("d3d_readback_thread") != "harness-main" or
            writer.get("png_factory_scope") != "per-worker-com-mta" or
            writer.get("deterministic_unique_output_paths") is not True or
            writer.get("drained_before_publication") is not True):
        raise ValueError("multiscale artifact-writer contract differs")
    if (writer["maximum_inflight_job_bound"] !=
            writer["worker_count"] + writer["queue_capacity"] + 1):
        raise ValueError("multiscale artifact-writer bound evidence differs")
    expected_per_kind = scale_count * output_count
    expected_jobs = expected_per_kind * 2
    count_fields = ("submitted_jobs", "completed_jobs",
                    "sbs_png_jobs", "mask_png_jobs")
    if any(not isinstance(writer.get(field), int) or
           isinstance(writer.get(field), bool) or writer[field] < 0
           for field in count_fields):
        raise ValueError("multiscale artifact-writer counts are invalid")
    if (writer["submitted_jobs"] != expected_jobs or
            writer["completed_jobs"] != expected_jobs or
            writer["sbs_png_jobs"] != expected_per_kind or
            writer["mask_png_jobs"] != expected_per_kind):
        raise ValueError("multiscale artifact-writer completion evidence differs")
    return writer


def _validate_depth_state_cache(harness, *, source_count, output_count):
    value = harness.get("depth_state_cache")
    if not isinstance(value, dict):
        raise ValueError("multiscale depth-state cache provenance is missing")
    expected_keys = {
        "mode", "key_sha256", "manifest_sha256", "boundary",
        "selected_state_frame_count", "runtime_scene_frame_count",
    }
    if set(value) != expected_keys:
        raise ValueError("multiscale depth-state cache provenance differs")
    mode = value.get("mode")
    if (mode not in {"disabled", "cold-export", "authenticated-replay"} or
            value.get("boundary") !=
            "completed-production-depth-state-before-warp-prefilter" or
            value.get("selected_state_frame_count") != output_count or
            value.get("runtime_scene_frame_count") != source_count):
        raise ValueError("multiscale depth-state cache provenance differs")
    key = value.get("key_sha256")
    manifest = value.get("manifest_sha256")
    if mode == "disabled":
        if key != "" or manifest != "":
            raise ValueError("disabled depth-state cache has an identity")
    elif (not isinstance(key, str) or not SHA256.fullmatch(key) or
          not isinstance(manifest, str) or not SHA256.fullmatch(manifest)):
        raise ValueError("multiscale depth-state cache identity is invalid")
    expected_calls = 0 if mode == "authenticated-replay" else 1
    if harness.get("shipping_estimator_calls_per_source_frame") != expected_calls:
        raise ValueError("multiscale depth-state estimator-call provenance differs")
    return value


def publish(root, *, clip, clip_sha1, executable_sha256, conf_sha256,
            metric_sha256, scales):
    """Validate a completed harness tree and atomically publish its byte manifest."""
    root = Path(root).resolve()
    harness_path = root / HARNESS_MANIFEST
    harness = _load_object(harness_path, "multiscale harness contract")
    scales = tuple(float(value) for value in scales)
    expected_rows = [{
        "index": index,
        "scale": scale,
        "float32_bits": scale_float32_bits(scale),
        "directory": f"scales/{scale_slug(scale)}",
    } for index, scale in enumerate(scales)]
    if (harness.get("schema") != HARNESS_SCHEMA or
            harness.get("contract") != HARNESS_CONTRACT or
            harness.get("scope") != "offline-sbs-bench-only" or
            harness.get("scale_rows") != expected_rows or
            harness.get("common_directory") != "common"):
        raise ValueError("multiscale harness contract differs")
    (source_ids, label_ids, output_ids, selection_mode,
     label_frames_sha256) = _validate_harness_selection(harness)
    if (harness.get("source_frame_count") != len(source_ids) or
            harness.get("output_frame_count_per_scale") != len(output_ids)):
        raise ValueError("multiscale harness frame counts differ")
    _validate_artifact_writer(
        harness, scale_count=len(expected_rows), output_count=len(output_ids)
    )
    depth_state_cache = _validate_depth_state_cache(
        harness, source_count=len(source_ids), output_count=len(output_ids)
    )

    common = root / "common"
    if (_frame_ids(common, "depth_", ".png") != output_ids or
            _frame_ids(common, "raw_", ".f32") != output_ids or
            not (common / "raw_shape.json").is_file() or
            not (common / "runtime_scene_evidence.json").is_file()):
        raise ValueError("multiscale common artifacts are incomplete")
    ema_ids = _frame_ids(common, "ema_mask_", ".png")
    if ema_ids and ema_ids != output_ids:
        raise ValueError("multiscale EMA-mask artifacts are incomplete")
    try:
        scene_evidence = runtime_scene_evidence.load(
            common / "runtime_scene_evidence.json"
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("multiscale runtime scene evidence is invalid") from error
    if (scene_evidence.get("source_frame_ids") != source_ids or
            scene_evidence.get("completed_source_frame_ids") != source_ids or
            scene_evidence.get("depth_reuse_interval") != 1):
        raise ValueError("multiscale runtime scene evidence is not full cadence")
    common_artifacts = _glob_artifacts(common, COMMON_ARTIFACTS, root)

    scale_rows = []
    for expected in expected_rows:
        directory = root / expected["directory"]
        contract_path = directory / "contract.json"
        contract = _load_object(contract_path, "multiscale scale contract")
        if (contract.get("multiscale_batch") is not True or
                contract.get("multiscale_batch_contract") != HARNESS_CONTRACT or
                contract.get("multiscale_scale_index") != expected["index"] or
                contract.get("multiscale_scale_float32_bits") !=
                expected["float32_bits"] or
                contract.get("artistic_scale_override") != expected["scale"] or
                contract.get("output_selection_mode") != selection_mode or
                contract.get("label_frame_ids") != label_ids or
                contract.get("output_selected_frame_ids") != output_ids or
                contract.get("output_label_frames_sha256") !=
                label_frames_sha256 or
                contract.get("metric_sha256") != metric_sha256):
            raise ValueError(
                f"multiscale scale contract differs: {expected['directory']}"
            )
        for prefix, suffix in (
                ("sbs_", ".png"), ("warp_mask_", ".png"),
                ("warp_disparity_", ".f32"),
                ("warp_unclamped_disparity_", ".f32")):
            if _frame_ids(directory, prefix, suffix) != output_ids:
                raise ValueError(
                    f"multiscale scale artifacts are incomplete: {expected['directory']}"
                )
        scale_rows.append({
            **expected,
            "contract_sha256": sha256_file(contract_path),
            "artifacts": _glob_artifacts(directory, SCALE_ARTIFACTS, root),
        })

    if not isinstance(clip_sha1, str) or not CLIP_ID.fullmatch(clip_sha1):
        raise ValueError("invalid multiscale clip_sha1")
    if (not isinstance(executable_sha256, str) or
            not SHA256.fullmatch(executable_sha256)):
        raise ValueError("invalid multiscale executable_sha256")
    for name, value in (
            ("conf_sha256", conf_sha256),
            ("metric_sha256", metric_sha256)):
        if not isinstance(value, str) or not METRIC_ID.fullmatch(value):
            raise ValueError(f"invalid multiscale {name}")
    payload = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "clip": clip,
        "clip_sha1": clip_sha1,
        "executable_sha256": executable_sha256,
        "conf_sha256": conf_sha256,
        "metric_sha256": metric_sha256,
        "harness_contract_sha256": sha256_file(harness_path),
        "source_frame_ids": source_ids,
        "label_frame_ids": label_ids,
        "output_selected_frame_ids": output_ids,
        "output_selection_mode": selection_mode,
        "output_label_frames_sha256": label_frames_sha256,
        "depth_state_cache": depth_state_cache,
        "common_artifacts": common_artifacts,
        "scale_rows": scale_rows,
    }
    path = root / MANIFEST
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(payload))
    temporary.replace(path)
    return payload


def _validate_artifacts(root, identities):
    for identity in identities:
        if (not isinstance(identity, dict) or
                set(identity) != {"path", "size", "sha256"} or
                not isinstance(identity["path"], str) or
                not isinstance(identity["size"], int) or identity["size"] < 0 or
                not isinstance(identity["sha256"], str) or
                not SHA256.fullmatch(identity["sha256"])):
            raise ValueError("invalid multiscale artifact identity")
        path = (root / identity["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("multiscale artifact path escapes root") from error
        if (not path.is_file() or path.stat().st_size != identity["size"] or
                sha256_file(path) != identity["sha256"]):
            raise ValueError(f"multiscale artifact changed: {identity['path']}")


def validate(root, *, clip, clip_sha1, executable_sha256, conf_sha256,
             metric_sha256, scale):
    """Fail closed for one scale plus its shared artifacts; return resolved paths."""
    root = Path(root).resolve()
    path = root / MANIFEST
    payload = _load_object(path, "multiscale batch manifest")
    if path.read_bytes() != canonical_bytes(payload):
        raise ValueError("multiscale batch manifest is not canonical")
    if (payload.get("schema") != SCHEMA or payload.get("contract") != CONTRACT or
            payload.get("clip") != clip or payload.get("clip_sha1") != clip_sha1 or
            payload.get("executable_sha256") != executable_sha256 or
            payload.get("conf_sha256") != conf_sha256 or
            payload.get("metric_sha256") != metric_sha256):
        raise ValueError("multiscale batch identity differs")
    harness_path = root / HARNESS_MANIFEST
    if (payload.get("harness_contract_sha256") != sha256_file(harness_path)):
        raise ValueError("multiscale harness contract changed")
    harness = _load_object(harness_path, "multiscale harness contract")
    if (harness.get("schema") != HARNESS_SCHEMA or
            harness.get("contract") != HARNESS_CONTRACT):
        raise ValueError("multiscale harness contract differs")
    (source_ids, label_ids, output_ids, selection_mode,
     label_frames_sha256) = _validate_harness_selection(harness)
    if (harness.get("source_frame_count") != len(source_ids) or
            harness.get("output_frame_count_per_scale") != len(output_ids)):
        raise ValueError("multiscale harness frame counts differ")
    scale_rows = harness.get("scale_rows")
    if not isinstance(scale_rows, list) or not scale_rows:
        raise ValueError("multiscale harness scale rows differ")
    _validate_artifact_writer(
        harness, scale_count=len(scale_rows), output_count=len(output_ids)
    )
    depth_state_cache = _validate_depth_state_cache(
        harness, source_count=len(source_ids), output_count=len(output_ids)
    )
    if payload.get("depth_state_cache") != depth_state_cache:
        raise ValueError("multiscale depth-state cache provenance changed")
    if (payload.get("source_frame_ids") != source_ids or
            payload.get("label_frame_ids") != label_ids or
            payload.get("output_selected_frame_ids") != output_ids or
            payload.get("output_selection_mode") != selection_mode or
            payload.get("output_label_frames_sha256") !=
            label_frames_sha256):
        raise ValueError("multiscale manifest frame selection differs")
    expected_slug = scale_slug(scale)
    expected_bits = scale_float32_bits(scale)
    rows = [row for row in payload.get("scale_rows", ())
            if isinstance(row, dict) and row.get("directory") == f"scales/{expected_slug}"]
    if (len(rows) != 1 or rows[0].get("float32_bits") != expected_bits or
            rows[0].get("scale") != float(scale)):
        raise ValueError("multiscale batch lacks the requested exact scale")
    common_root = root / "common"
    if (_frame_ids(common_root, "depth_", ".png") != output_ids or
            _frame_ids(common_root, "raw_", ".f32") != output_ids):
        raise ValueError("multiscale common artifact frame coverage changed")
    try:
        scene_evidence = runtime_scene_evidence.load(
            common_root / "runtime_scene_evidence.json"
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("multiscale runtime scene evidence changed") from error
    if (scene_evidence.get("source_frame_ids") != source_ids or
            scene_evidence.get("completed_source_frame_ids") != source_ids or
            scene_evidence.get("depth_reuse_interval") != 1):
        raise ValueError("multiscale runtime scene evidence is not full cadence")
    _validate_artifacts(root, payload.get("common_artifacts", ()))
    _validate_artifacts(root, rows[0].get("artifacts", ()))
    scale_root = root / rows[0]["directory"]
    for prefix, suffix in (
            ("sbs_", ".png"), ("warp_mask_", ".png"),
            ("warp_disparity_", ".f32"),
            ("warp_unclamped_disparity_", ".f32")):
        if _frame_ids(scale_root, prefix, suffix) != output_ids:
            raise ValueError("multiscale scale artifact frame coverage changed")
    contract_path = scale_root / "contract.json"
    if rows[0].get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("multiscale scale contract changed")
    return {
        "manifest": payload,
        "manifest_path": path,
        "manifest_sha256": sha256_file(path),
        "common_root": common_root,
        "scale_root": scale_root,
        "scale_row": rows[0],
    }
