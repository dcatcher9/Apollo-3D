#!/usr/bin/env python3
"""Build authenticated per-frame ordinal safety labels from Apollo renders.

This is an experimental path.  It does not read or emit the shipping scalar
schema-10 labels.  One bundle consumes the complete 1.00..1.50/0.02 render
grid for exactly two deployment geometries and one input variant. DA-V2 and
runtime-scene evidence still bind the authenticated source sequence, while
expensive render evidence and safety classification contain only the explicit
label targets. Temporal metrics are excluded from this target-only experiment.

The expensive evaluator interface is isolated in
``parse_frame_gate_evidence``.  The join, safety classification, geometry
intersection, and canonical bundle writer operate on small normalized run
objects and can therefore remain stable if the sidecar transport evolves.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys


THIS_DIR = Path(__file__).resolve().parent
SBSBENCH_DIR = THIS_DIR.parent / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))

import artistic_geometry_contract as geometry_contract  # noqa: E402
import artistic_policy_ordinal_contract as ordinal_contract  # noqa: E402
import depth_input_color as input_color  # noqa: E402
import merge_ordinal_geometry_frontiers as geometry_merge  # noqa: E402
import multiscale_batch  # noqa: E402
import ordinal_result_cache  # noqa: E402
import run_eval  # noqa: E402
import run_multiscale_eval  # noqa: E402
import runtime_scene_evidence as scene_contract  # noqa: E402
import select_ordinal_render_frontiers as frontier_selector  # noqa: E402
import select_render_feasible_labels as scalar_selector  # noqa: E402


RUN_GRID_SCHEMA = 1
RUN_GRID_CONTRACT = "apollo-ordinal-frame-run-grid-v1"
BUNDLE_SCHEMA = 6
BUNDLE_CONTRACT = "apollo-ordinal-target-frame-label-bundle-v6"
SUMMARY_SCHEMA = 6
SUMMARY_CONTRACT = "apollo-ordinal-target-frame-label-summary-v6"
FRAME_SAFETY_EVIDENCE_SCHEMA = 2
FRAME_SAFETY_EVIDENCE_CONTRACT = "apollo-ordinal-target-frame-safety-v2"
CODE_IDENTITY_SCHEMA = 1
CODE_IDENTITY_CONTRACT = "apollo-ordinal-label-code-identity-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_SCENE_FILENAME = "runtime_scene_evidence.json"
THRESHOLDS_PATH = SBSBENCH_DIR / "thresholds.json"


def canonical_bytes(value):
    return (json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not an object: {path}")
    return value


def _validate_manifest_artifacts(artifacts, prefix, description):
    if not isinstance(artifacts, list):
        raise RuntimeError(f"{description} artifact identities are missing")
    paths = []
    for artifact in artifacts:
        if (not isinstance(artifact, dict) or
                set(artifact) != {"path", "size", "sha256"} or
                not isinstance(artifact.get("path"), str) or
                not artifact["path"].startswith(prefix) or
                not isinstance(artifact.get("size"), int) or
                artifact["size"] < 0 or
                not isinstance(artifact.get("sha256"), str) or
                not SHA256.fullmatch(artifact["sha256"])):
            raise RuntimeError(f"{description} artifact identity is invalid")
        paths.append(artifact["path"])
    if paths != sorted(set(paths)):
        raise RuntimeError(f"{description} artifact identities are noncanonical")


def _validate_multiscale_provenance(
        path, clip, header, results_payload, clip_record, scale,
        source_frame_ids, label_frame_ids, output_frame_ids,
        label_frames_sha256):
    """Join a scored scalar run to its retained authenticated batch files."""
    root = path.parent / "multiscale_provenance" / clip
    expected_names = {
        "contract.json", multiscale_batch.MANIFEST,
        multiscale_batch.HARNESS_MANIFEST,
        run_multiscale_eval.RENDER_IDENTITY_FILENAME,
    }
    actual_names = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*") if candidate.is_file()
    } if root.is_dir() else set()
    if actual_names != expected_names:
        raise RuntimeError(f"{path}/{clip}: multiscale provenance is incomplete")

    manifest_path = root / multiscale_batch.MANIFEST
    manifest = _load_json_object(manifest_path, "multiscale batch manifest")
    if manifest_path.read_bytes() != multiscale_batch.canonical_bytes(manifest):
        raise RuntimeError(
            f"{path}/{clip}: multiscale batch manifest is not canonical"
        )
    manifest_sha256 = sha256_file(manifest_path)
    results_meta = results_payload.get("meta")
    if not isinstance(results_meta, dict):
        raise RuntimeError(f"{path}/{clip}: results metadata is missing")
    if (header.get("precomputed_multiscale") is not True or
            header.get("multiscale_batch_manifest_sha256") !=
            manifest_sha256 or
            results_meta.get("precomputed_multiscale") is not True or
            results_meta.get("multiscale_batch_manifest_sha256") !=
            manifest_sha256):
        raise RuntimeError(
            f"{path}/{clip}: scored run and multiscale manifest differ"
        )
    executable_sha256 = manifest.get("executable_sha256")
    if (manifest.get("schema") != multiscale_batch.SCHEMA or
            manifest.get("contract") != multiscale_batch.CONTRACT or
            manifest.get("clip") != clip or
            manifest.get("clip_sha1") != clip_record.get("clip_sha1") or
            manifest.get("conf_sha256") != header.get("conf_sha256") or
            manifest.get("metric_sha256") != header.get("metric_sha256") or
            manifest.get("source_frame_ids") != source_frame_ids or
            manifest.get("label_frame_ids") != label_frame_ids or
            manifest.get("output_selected_frame_ids") != output_frame_ids or
            manifest.get("output_selection_mode") != "label-frames" or
            manifest.get("output_label_frames_sha256") !=
            label_frames_sha256 or
            not isinstance(executable_sha256, str) or
            not SHA256.fullmatch(executable_sha256)):
        raise RuntimeError(
            f"{path}/{clip}: retained multiscale manifest identity differs"
        )
    _validate_manifest_artifacts(
        manifest.get("common_artifacts"), "common/", "multiscale common"
    )

    rows = manifest.get("scale_rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{path}/{clip}: multiscale scale rows are missing")
    projected_rows = []
    requested = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}/{clip}: multiscale scale row is invalid")
        try:
            row_scale = float(row.get("scale"))
            slug = multiscale_batch.scale_slug(row_scale)
            bits = multiscale_batch.scale_float32_bits(row_scale)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"{path}/{clip}: multiscale scale row is invalid"
            ) from error
        contract_sha256 = row.get("contract_sha256")
        if (row.get("index") != index or
                row.get("directory") != f"scales/{slug}" or
                row.get("float32_bits") != bits or
                not isinstance(contract_sha256, str) or
                not SHA256.fullmatch(contract_sha256)):
            raise RuntimeError(
                f"{path}/{clip}: multiscale scale row identity differs"
            )
        _validate_manifest_artifacts(
            row.get("artifacts"), f"scales/{slug}/", f"multiscale {slug}"
        )
        projected_rows.append({
            "index": index,
            "scale": row_scale,
            "float32_bits": bits,
            "directory": f"scales/{slug}",
        })
        if (math.isclose(row_scale, scale, abs_tol=1e-9) and
                bits == multiscale_batch.scale_float32_bits(scale)):
            requested.append(row)
    if len(requested) != 1:
        raise RuntimeError(
            f"{path}/{clip}: retained manifest lacks the exact scored scale"
        )

    harness_path = root / multiscale_batch.HARNESS_MANIFEST
    harness = _load_json_object(harness_path, "multiscale harness contract")
    harness_sha256 = sha256_file(harness_path)
    depth_origin = harness.get("depth_state_cache")
    score_cache = results_meta.get("ordinal_score_cache")
    cached_origin = (
        isinstance(score_cache, dict) and
        score_cache.get("contract") ==
        ordinal_result_cache.PACKET_CONTRACT and
        score_cache.get("retained_provenance") == "cached-semantic-origin"
    )
    if cached_origin:
        expected_depth_keys = {
            "mode", "key_sha256", "manifest_sha256", "boundary",
            "selected_state_frame_count", "runtime_scene_frame_count",
        }
        key = depth_origin.get("key_sha256") \
            if isinstance(depth_origin, dict) else None
        depth_manifest = depth_origin.get("manifest_sha256") \
            if isinstance(depth_origin, dict) else None
        valid_identity = (
            (key == "" and depth_manifest == "") or
            (isinstance(key, str) and SHA256.fullmatch(key) and
             isinstance(depth_manifest, str) and
             SHA256.fullmatch(depth_manifest))
        )
        if (not isinstance(depth_origin, dict) or
                set(depth_origin) != expected_depth_keys or
                depth_origin.get("mode") !=
                ordinal_result_cache.CACHED_DEPTH_MODE or
                depth_origin.get("boundary") !=
                "completed-production-depth-state-before-warp-prefilter" or
                depth_origin.get("selected_state_frame_count") !=
                len(output_frame_ids) or
                depth_origin.get("runtime_scene_frame_count") !=
                len(source_frame_ids) or
                not valid_identity or
                harness.get("shipping_estimator_calls_per_source_frame") != 0 or
                manifest.get("depth_state_cache") != depth_origin):
            raise RuntimeError(
                f"{path}/{clip}: retained cached depth provenance differs"
            )
    else:
        try:
            validated_depth_origin = multiscale_batch._validate_depth_state_cache(
                harness, source_count=len(source_frame_ids),
                output_count=len(output_frame_ids),
            )
        except ValueError as error:
            raise RuntimeError(
                f"{path}/{clip}: retained depth provenance differs"
            ) from error
        if manifest.get("depth_state_cache") != validated_depth_origin:
            raise RuntimeError(
                f"{path}/{clip}: retained batch depth provenance differs"
            )
    if (manifest.get("harness_contract_sha256") != harness_sha256 or
            harness.get("schema") != multiscale_batch.HARNESS_SCHEMA or
            harness.get("contract") != multiscale_batch.HARNESS_CONTRACT or
            harness.get("scope") != "offline-sbs-bench-only" or
            harness.get("common_directory") != "common" or
            harness.get("source_frame_ids") != source_frame_ids or
            harness.get("label_frame_ids") != label_frame_ids or
            harness.get("output_selected_frame_ids") != output_frame_ids or
            harness.get("output_selection_mode") != "label-frames" or
            harness.get("output_label_frames_sha256") !=
            label_frames_sha256 or
            harness.get("source_frame_count") != len(source_frame_ids) or
            harness.get("output_frame_count_per_scale") !=
            len(output_frame_ids) or
            harness.get("scale_rows") != projected_rows):
        raise RuntimeError(
            f"{path}/{clip}: retained multiscale harness contract differs"
        )

    contract_path = root / "contract.json"
    contract = _load_json_object(contract_path, "multiscale scale contract")
    contract_sha256 = sha256_file(contract_path)
    row = requested[0]
    try:
        contract_scale = float(contract.get("artistic_scale_override"))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"{path}/{clip}: retained multiscale scale contract differs"
        ) from error
    if (row.get("contract_sha256") != contract_sha256 or
            clip_record.get("harness_contract_sha256") != contract_sha256 or
            contract.get("multiscale_batch") is not True or
            contract.get("multiscale_batch_contract") !=
            multiscale_batch.HARNESS_CONTRACT or
            contract.get("multiscale_scale_index") != row.get("index") or
            contract.get("multiscale_scale_float32_bits") !=
            row.get("float32_bits") or
            not math.isclose(contract_scale, scale, abs_tol=1e-9) or
            contract.get("metric_sha256") != header.get("metric_sha256") or
            contract.get("output_selection_mode") != "label-frames" or
            contract.get("label_frame_ids") != label_frame_ids or
            contract.get("output_selected_frame_ids") != output_frame_ids or
            contract.get("output_label_frames_sha256") !=
            label_frames_sha256 or
            contract.get("depth_step") != "current-once" or
            contract.get("depth_reuse_interval") != 1 or
            contract.get("artistic_policy") is not False):
        raise RuntimeError(
            f"{path}/{clip}: retained multiscale scale contract differs"
        )
    if (not isinstance(depth_origin, dict) or
            contract.get("depth_state_cache_mode") !=
            depth_origin.get("mode") or
            contract.get("depth_state_cache_key_sha256") !=
            depth_origin.get("key_sha256") or
            contract.get("depth_state_manifest_sha256") !=
            depth_origin.get("manifest_sha256")):
        raise RuntimeError(
            f"{path}/{clip}: retained scale depth provenance differs"
        )
    receipt_path = root / run_multiscale_eval.RENDER_IDENTITY_FILENAME
    receipt = _load_json_object(receipt_path, "multiscale render identity")
    if (receipt_path.read_bytes() != multiscale_batch.canonical_bytes(receipt) or
            receipt.get("schema") !=
            run_multiscale_eval.RENDER_IDENTITY_SCHEMA or
            receipt.get("contract") !=
            run_multiscale_eval.RENDER_IDENTITY_CONTRACT or
            receipt.get("batch_manifest_sha256") != manifest_sha256 or
            not isinstance(receipt.get("render_identity_sha256"), str) or
            not SHA256.fullmatch(receipt["render_identity_sha256"])):
        raise RuntimeError(
            f"{path}/{clip}: retained render identity differs"
        )
    return {
        "executable_sha256": executable_sha256,
        "multiscale_batch_manifest_sha256": manifest_sha256,
        "multiscale_harness_contract_sha256": harness_sha256,
        "multiscale_scale_contract_sha256": contract_sha256,
    }


def current_code_identity():
    """Return portable byte identities for every direct label-policy dependency."""
    files = {
        "builder": ("tools/depth_models/build_ordinal_frame_label_bundle.py",
                    Path(__file__).resolve()),
        "ordinal_contract": (
            "tools/depth_models/artistic_policy_ordinal_contract.py",
            Path(ordinal_contract.__file__).resolve(),
        ),
        "ordinal_result_cache": (
            "tools/depth_models/ordinal_result_cache.py",
            Path(ordinal_result_cache.__file__).resolve(),
        ),
        "ordinal_selector": (
            "tools/depth_models/select_ordinal_render_frontiers.py",
            Path(frontier_selector.__file__).resolve(),
        ),
        "geometry_merge": (
            "tools/depth_models/merge_ordinal_geometry_frontiers.py",
            Path(geometry_merge.__file__).resolve(),
        ),
        "geometry_contract": (
            "tools/depth_models/artistic_geometry_contract.py",
            Path(geometry_contract.__file__).resolve(),
        ),
        "scalar_selector": (
            "tools/depth_models/select_render_feasible_labels.py",
            Path(scalar_selector.__file__).resolve(),
        ),
        "runtime_scene_adapter": (
            "tools/sbsbench/runtime_scene_evidence.py",
            Path(scene_contract.__file__).resolve(),
        ),
        "frame_gate_adapter": (
            "tools/sbsbench/run_eval.py", Path(run_eval.__file__).resolve(),
        ),
        "multiscale_transport": (
            "tools/sbsbench/multiscale_batch.py",
            Path(multiscale_batch.__file__).resolve(),
        ),
        "metric_implementation": (
            "tools/sbsbench/sbsbench.py", SBSBENCH_DIR / "sbsbench.py",
        ),
        "metric_thresholds": (
            "tools/sbsbench/thresholds.json", THRESHOLDS_PATH,
        ),
    }
    return {
        "schema": CODE_IDENTITY_SCHEMA,
        "contract": CODE_IDENTITY_CONTRACT,
        "files": {
            key: {"logical_path": logical_path, "sha256": sha256_file(path)}
            for key, (logical_path, path) in sorted(files.items())
        },
    }


def _normalize_diagnostic_violations(value, origin):
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(f"{origin}: diagnostic violations are not a list")
    allowed_kinds = {
        "hard_min", "hard_max", "trigger_min", "trigger_max",
        "ordinal_hard_min", "ordinal_hard_max",
    }
    normalized = []
    for item in value:
        if (not isinstance(item, dict) or
                set(item) != {"metric", "kind", "bound", "value"} or
                not isinstance(item.get("metric"), str) or
                not item["metric"] or item.get("kind") not in allowed_kinds):
            raise RuntimeError(f"{origin}: diagnostic violation is invalid")
        bound = item.get("bound")
        measured = item.get("value")
        if any(
                not isinstance(number, (int, float)) or
                isinstance(number, bool) or not math.isfinite(float(number))
                for number in (bound, measured)):
            raise RuntimeError(f"{origin}: diagnostic violation is non-finite")
        normalized.append({
            "metric": item["metric"], "kind": item["kind"],
            "bound": float(bound), "value": float(measured),
        })
    normalized.sort(key=lambda item: (
        item["metric"], item["kind"], item["bound"], item["value"]
    ))
    if len({canonical_sha256(item) for item in normalized}) != len(normalized):
        raise RuntimeError(f"{origin}: diagnostic violations repeat")
    return normalized


def _finite_scale(value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or
            not math.isfinite(float(value))):
        raise RuntimeError("ordinal run scale is not finite")
    index = ordinal_contract.scale_index(value)
    return ordinal_contract.SCALES[index]


def _flatten_frame_metrics(frame, primary_metrics, hard_metrics, origin):
    groups = frame.get("metrics")
    if not isinstance(groups, dict):
        raise RuntimeError(f"{origin}: frame metric groups are missing")
    result = {}
    for role, expected_names in (
            ("primary", primary_metrics), ("hard", hard_metrics)):
        values = groups.get(role)
        if not isinstance(values, dict) or set(values) != set(expected_names):
            raise RuntimeError(
                f"{origin}: {role} frame metrics differ from the sidecar header"
            )
        overlap = set(result) & set(values)
        if overlap:
            raise RuntimeError(f"{origin}: metric roles overlap: {sorted(overlap)}")
        for name, value in values.items():
            if value is not None and (
                    not isinstance(value, (int, float)) or
                    isinstance(value, bool) or not math.isfinite(float(value))):
                raise RuntimeError(f"{origin}: metric {name!r} is not finite/null")
            result[name] = None if value is None else float(value)
    return result


def _select_clip_records(records, clip):
    groups = {}
    current = None
    for record in records[1:-1]:
        kind = record.get("record")
        if kind == "clip":
            if record.get("clip") in groups:
                raise RuntimeError("frame-gate sidecar repeats a clip identity")
            current = {"clip": record, "frames": []}
            groups[record.get("clip")] = current
        elif kind == "frame" and current is not None:
            current["frames"].append(record)
        elif kind == "clip_end":
            current = None
    if clip not in groups:
        raise RuntimeError(f"frame-gate sidecar has no clip {clip!r}")
    return groups[clip]


def load_runtime_scene_evidence(path, expected_frame_ids):
    """Validate the exact SubjectState scene sequence for a full-cadence run."""
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"runtime scene evidence is missing: {path}")
    try:
        value = scene_contract.load(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"runtime scene evidence is invalid: {path}") from error
    expected_frame_ids = list(expected_frame_ids)
    if (value.get("depth_reuse_interval") != 1 or
            value.get("source_frame_ids") != expected_frame_ids or
            value.get("completed_source_frame_ids") != expected_frame_ids or
            value.get("completed_depth_frame_count") != len(expected_frame_ids)):
        raise RuntimeError(f"runtime scene evidence contract differs: {path}")
    return value["frames"], sha256_file(path), value


def _validate_selected_runtime_scene_row(value, frame_id, source_ordinal):
    """Validate one embedded row while its full-cadence payload stays hash-bound."""
    if (not isinstance(value, dict) or set(value) != scene_contract.FRAME_KEYS or
            value.get("source_frame_id") != frame_id or
            value.get("source_frame_ordinal") != source_ordinal):
        raise RuntimeError("ordinal frame runtime scene evidence is invalid")
    for field in ("runtime_scene_id",):
        field_value = value.get(field)
        if (not isinstance(field_value, int) or isinstance(field_value, bool) or
                field_value < 0):
            raise RuntimeError("ordinal frame runtime scene evidence is invalid")
    if any(not isinstance(value.get(field), bool) for field in (
            "subject_initialized", "hard_cut", "scene_start")):
        raise RuntimeError("ordinal frame runtime scene evidence is invalid")
    age = value.get("scene_age")
    if (not isinstance(age, (int, float)) or isinstance(age, bool) or
            not math.isfinite(float(age)) or float(age) < 0.0 or
            float(age) > 65535.0 or abs(float(age) - round(float(age))) > 1e-5 or
            (value["hard_cut"] and not value["scene_start"]) or
            (value["scene_start"] and float(age) != 0.0) or
            (not value["subject_initialized"] and float(age) != 0.0)):
        raise RuntimeError("ordinal frame runtime scene evidence is invalid")
    return value


def _validate_runtime_scene_trace(value, source_frame_ids,
                                  completion_sequence_contract):
    """Validate the complete embedded shipping SubjectState sequence."""
    source_frame_ids = list(source_frame_ids)
    payload = {
        "schema": 1,
        "contract": scene_contract.CONTRACT,
        "evidence_source":
            "SubjectState[0].y after completed depth postprocess",
        "cut_rule": "prior_scene_age_gte_7_and_current_scene_age_eq_0",
        "cadence": "completed-depth-frames-only",
        "completion_sequence_contract": completion_sequence_contract,
        "depth_reuse_interval": 1,
        "source_frame_ids": source_frame_ids,
        "completed_source_frame_ids": source_frame_ids,
        "completed_depth_frame_count": len(source_frame_ids),
        "frames": value,
    }
    try:
        scene_contract.validate(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "ordinal full-cadence runtime scene trace is invalid"
        ) from error
    return value


def parse_frame_gate_evidence(path, clip):
    """Adapt one canonical evaluator sidecar to the normalized run contract.

    The adjacent ``results.json`` is checked against the authenticated digest
    in the sidecar header.  Artifact byte identities and the exact deployment
    geometry remain bound by the sidecar payload digest.
    """
    try:
        run_eval.validate_path_component(clip, "clip name")
    except ValueError as error:
        raise RuntimeError(f"invalid frame-gate clip identity: {clip!r}") from error
    path = Path(path).resolve()
    records = run_eval.validate_frame_gate_evidence(path)
    header = records[0]
    group = _select_clip_records(records, clip)
    clip_record = group["clip"]
    frames = group["frames"]
    if (header.get("contract") !=
            run_eval.SELECTED_FRAME_GATE_EVIDENCE_CONTRACT or
            clip_record.get("output_selection_contract") !=
            run_eval.SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT):
        raise RuntimeError(
            f"{path}/{clip}: ordinal labels require selected-frame evidence"
        )

    source_frame_ids = clip_record.get("full_source_frame_ids")
    label_frame_ids = clip_record.get("label_frame_ids")
    output_frame_ids = clip_record.get("output_selected_frame_ids")
    label_frames_sha256 = clip_record.get("output_label_frames_sha256")
    selection = {
        "mode": "label-frames",
        "label_frame_ids": label_frame_ids,
        "output_frame_ids": output_frame_ids,
        "label_frames_sha256": label_frames_sha256,
    }
    try:
        (source_frame_ids, label_frame_ids,
         output_frame_ids) = run_eval.validate_selected_frame_gate_coverage(
             source_frame_ids, selection
         )
    except ValueError as error:
        raise RuntimeError(
            f"{path}/{clip}: selected-frame identity is invalid"
        ) from error
    if ([frame.get("frame_id") for frame in frames] != output_frame_ids or
            clip_record.get("full_source_frame_count") != len(source_frame_ids) or
            clip_record.get("full_source_frame_ids_sha256") !=
            run_eval.frame_id_sequence_sha256(source_frame_ids) or
            clip_record.get("frame_count") != len(output_frame_ids)):
        raise RuntimeError(
            f"{path}/{clip}: selected/full source frame counts differ"
        )

    results_path = path.parent / "results.json"
    if not results_path.is_file():
        raise RuntimeError(f"frame-gate run lacks adjacent results.json: {path}")
    results_sha256 = sha256_file(results_path)
    if header.get("results_sha256") != results_sha256:
        raise RuntimeError(f"frame-gate results identity is stale: {path}")
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))

    geometry = geometry_contract.canonical_geometry_tuple(
        clip_record.get("geometry")
    )
    geometry_sha256 = canonical_sha256(geometry)
    if (clip_record.get("geometry_contract") !=
            geometry_contract.GEOMETRY_CONTRACT or
            clip_record.get("geometry_sha256") != geometry_sha256):
        raise RuntimeError(f"{path}/{clip}: geometry identity is stale")
    color = clip_record.get("color")
    variant = scalar_selector.input_variant_from_harness(
        color, f"{path}/{clip}/color"
    )
    variant_sha256 = input_color.input_variant_sha256(variant)
    if geometry["color_mode"] != variant["color_mode"]:
        raise RuntimeError(
            f"{path}/{clip}: geometry and input variant color modes differ"
        )

    pipeline = clip_record.get("pipeline")
    if not isinstance(pipeline, dict):
        raise RuntimeError(f"{path}/{clip}: pipeline provenance is missing")
    scale = _finite_scale(pipeline.get("artistic_scale_override"))
    if (pipeline.get("artistic_policy") is not False or
            pipeline.get("depth_step") != "current-once"):
        raise RuntimeError(
            f"{path}/{clip}: ordinal evidence is not a current-depth, "
            "learned-policy-disabled render"
        )
    pipeline_without_scale = dict(pipeline)
    del pipeline_without_scale["artistic_scale_override"]
    primary_metrics = header.get("primary_metrics")
    hard_metrics = header.get("hard_metrics")
    if (not isinstance(primary_metrics, list) or
            primary_metrics != sorted(set(primary_metrics)) or
            not isinstance(hard_metrics, list) or
            hard_metrics != sorted(set(hard_metrics))):
        raise RuntimeError(f"{path}: frame-gate metric declarations are invalid")

    scene_frames, scene_evidence_sha256, scene_payload = (
        load_runtime_scene_evidence(
            path.parent / clip / RUNTIME_SCENE_FILENAME, source_frame_ids
        )
    )
    if (clip_record.get("runtime_scene_contract") != scene_contract.CONTRACT or
            clip_record.get("runtime_scene_evidence_sha256") !=
            scene_evidence_sha256 or
            clip_record.get("runtime_scene_count") != 1 + max(
                item["runtime_scene_id"] for item in scene_frames
            ) or clip_record.get("completion_sequence_contract") !=
            scene_payload["completion_sequence_contract"]):
        raise RuntimeError(f"{path}/{clip}: runtime scene identity is stale")

    scene_by_id = {
        scene_frame["source_frame_id"]: scene_frame
        for scene_frame in scene_frames
    }
    frame_by_id = {frame.get("frame_id"): frame for frame in frames}
    if (len(scene_by_id) != len(scene_frames) or
            len(frame_by_id) != len(frames)):
        raise RuntimeError(f"{path}/{clip}: frame identities repeat")
    validated_output_frames = {}
    for frame in frames:
        frame_id = frame.get("frame_id")
        scene_frame = scene_by_id.get(frame_id)
        origin = f"{path}/{clip}/frame-{frame.get('frame_id')}"
        if (scene_frame is None or
                frame.get("ordinal") != scene_frame.get(
                    "source_frame_ordinal")):
            raise RuntimeError(f"{origin}: source-frame ordinal is inconsistent")
        if frame.get("runtime_scene") != scene_frame:
            raise RuntimeError(f"{origin}: runtime scene join is inconsistent")
        artifacts = frame.get("artifact_sha256")
        if (not isinstance(artifacts, dict) or
                not {"source", "depth"}.issubset(artifacts) or
                any(not isinstance(value, str) or not SHA256.fullmatch(value)
                    for value in artifacts.values())):
            raise RuntimeError(f"{origin}: artifact identities are invalid")
        validated_output_frames[frame_id] = {
            "frame": frame,
            "scene": scene_frame,
            "artifacts": dict(sorted(artifacts.items())),
            "metrics": _flatten_frame_metrics(
                frame, primary_metrics, hard_metrics, origin
            ),
            "diagnostic_violations": _normalize_diagnostic_violations(
                frame.get("violations"), origin
            ),
        }

    normalized_frames = []
    for label_ordinal, frame_id in enumerate(label_frame_ids):
        validated = validated_output_frames[frame_id]
        scene_frame = validated["scene"]
        normalized_frames.append({
            "frame_id": frame_id,
            "ordinal": label_ordinal,
            "source_ordinal": scene_frame["source_frame_ordinal"],
            "runtime_scene_id": scene_frame["runtime_scene_id"],
            "runtime_scene_evidence": scene_frame,
            "artifact_sha256": validated["artifacts"],
            "metrics": validated["metrics"],
            "diagnostic_violations": validated["diagnostic_violations"],
        })

    multiscale_identity = _validate_multiscale_provenance(
        path, clip, header, results_payload, clip_record, scale,
        source_frame_ids, label_frame_ids, output_frame_ids,
        label_frames_sha256,
    )

    common_identity = {
        "eval_schema": header.get("eval_schema"),
        "harness_schema": header.get("harness_schema"),
        "metric_sha256": header.get("metric_sha256"),
        "thresholds_sha256": header.get("thresholds_sha256"),
        "conf_sha256": header.get("conf_sha256"),
        "clip_hash_manifest_sha256": header.get(
            "clip_hash_manifest_sha256"
        ),
        "clip_set_sha1": header.get("clip_set_sha1"),
        "suite": header.get("suite"),
        "hdr_source_kind": header.get("hdr_source_kind"),
        "primary_metrics": primary_metrics,
        "hard_metrics": hard_metrics,
        "clip": clip,
        "clip_sha1": clip_record.get("clip_sha1"),
        "expected_flat": clip_record.get("expected_flat"),
        "output_selection_contract": clip_record.get(
            "output_selection_contract"
        ),
        "source_frame_count": len(source_frame_ids),
        "source_frame_ids": source_frame_ids,
        "source_frame_ids_sha256": clip_record.get(
            "full_source_frame_ids_sha256"
        ),
        "label_frame_count": len(label_frame_ids),
        "label_frame_ids": label_frame_ids,
        "output_frame_count": len(output_frame_ids),
        "output_selected_frame_ids": output_frame_ids,
        "output_label_frames_sha256": label_frames_sha256,
        "runtime_scene_count": clip_record.get("runtime_scene_count"),
        "completion_sequence_contract": clip_record.get(
            "completion_sequence_contract"
        ),
        "executable_sha256": multiscale_identity["executable_sha256"],
        "color": color,
        "pipeline_without_scale": pipeline_without_scale,
    }
    run_identity = {
        "frame_gate_evidence_sha256": sha256_file(path),
        "results_sha256": results_sha256,
        "harness_contract_sha256": clip_record.get(
            "harness_contract_sha256"
        ),
        "runtime_scene_evidence_sha256": scene_evidence_sha256,
        "run_name": header.get("run_name"),
        "geometry_sha256": geometry_sha256,
        "scale": scale,
        "multiscale_batch_manifest_sha256": multiscale_identity[
            "multiscale_batch_manifest_sha256"
        ],
        "multiscale_harness_contract_sha256": multiscale_identity[
            "multiscale_harness_contract_sha256"
        ],
        "multiscale_scale_contract_sha256": multiscale_identity[
            "multiscale_scale_contract_sha256"
        ],
    }
    return {
        "scale": scale,
        "clip": clip,
        "geometry": geometry,
        "geometry_sha256": geometry_sha256,
        "input_variant": variant,
        "input_variant_sha256": variant_sha256,
        "common_identity": common_identity,
        "run_identity": run_identity,
        "frames": normalized_frames,
        "_runtime_scene_trace": scene_frames,
        # Private adapter evidence. It is revalidated against identity below
        # and never serialized into the label bundle.
        "_results_payload": results_payload,
    }


def _validate_normalized_runs(runs, expected_input_variant_sha256):
    if not isinstance(runs, (list, tuple)) or not runs:
        raise RuntimeError("ordinal frame run grid is empty")
    if (not isinstance(expected_input_variant_sha256, str) or
            not SHA256.fullmatch(expected_input_variant_sha256)):
        raise RuntimeError("run grid has invalid expected input-variant identity")

    variants = {run.get("input_variant_sha256") for run in runs}
    if variants != {expected_input_variant_sha256}:
        raise RuntimeError("ordinal runs mix or mismatch input variants")
    for run in runs:
        try:
            input_color.validate_input_variant(run.get("input_variant"))
        except (RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError("ordinal run has invalid input variant") from error
        if (input_color.input_variant_sha256(run["input_variant"]) !=
                expected_input_variant_sha256):
            raise RuntimeError("ordinal run input-variant identity is stale")
    variant_values = {
        canonical_sha256(run.get("input_variant")) for run in runs
    }
    if len(variant_values) != 1:
        raise RuntimeError("ordinal runs disagree on the input-variant contract")
    common = {canonical_sha256(run.get("common_identity")) for run in runs}
    if len(common) != 1:
        raise RuntimeError("ordinal scale runs differ outside scale/geometry")
    common_identity = runs[0].get("common_identity")
    if (not isinstance(common_identity, dict) or
            not isinstance(common_identity.get("executable_sha256"), str) or
            not SHA256.fullmatch(common_identity["executable_sha256"])):
        raise RuntimeError("ordinal runs have invalid executable provenance")
    if (common_identity.get("output_selection_contract") !=
            run_eval.SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT):
        raise RuntimeError("ordinal runs are not selected-frame publications")
    source_frame_ids = common_identity.get("source_frame_ids")
    label_frame_ids = common_identity.get("label_frame_ids")
    output_frame_ids = common_identity.get("output_selected_frame_ids")
    label_frames_sha256 = common_identity.get("output_label_frames_sha256")
    try:
        source_frame_ids, label_frame_ids, output_frame_ids = (
            run_eval.validate_selected_frame_gate_coverage(
                source_frame_ids,
                {
                    "mode": "label-frames",
                    "label_frame_ids": label_frame_ids,
                    "output_frame_ids": output_frame_ids,
                    "label_frames_sha256": label_frames_sha256,
                },
            )
        )
    except ValueError as error:
        raise RuntimeError(
            "ordinal runs have invalid selected-frame provenance"
        ) from error
    if (common_identity.get("source_frame_count") != len(source_frame_ids) or
            common_identity.get("source_frame_ids_sha256") !=
            run_eval.frame_id_sequence_sha256(source_frame_ids) or
            common_identity.get("label_frame_count") != len(label_frame_ids) or
            common_identity.get("output_frame_count") != len(output_frame_ids)):
        raise RuntimeError("ordinal selected-frame counts/identity differ")

    grouped = {}
    for run in runs:
        scale = _finite_scale(run.get("scale"))
        geometry = geometry_contract.canonical_geometry_tuple(
            run.get("geometry")
        )
        geometry_sha256 = canonical_sha256(geometry)
        if run.get("geometry_sha256") != geometry_sha256:
            raise RuntimeError("ordinal run has stale geometry identity")
        identity = run.get("run_identity")
        required_identity = {
            "frame_gate_evidence_sha256", "results_sha256",
            "harness_contract_sha256", "runtime_scene_evidence_sha256",
            "run_name", "geometry_sha256", "scale",
            "multiscale_batch_manifest_sha256",
            "multiscale_harness_contract_sha256",
            "multiscale_scale_contract_sha256",
        }
        if (not isinstance(identity, dict) or set(identity) != required_identity or
                identity.get("geometry_sha256") != geometry_sha256 or
                _finite_scale(identity.get("scale")) != scale or
                any(not isinstance(identity.get(field), str) or
                    not SHA256.fullmatch(identity[field])
                    for field in (
                        "frame_gate_evidence_sha256", "results_sha256",
                        "harness_contract_sha256",
                        "runtime_scene_evidence_sha256",
                        "multiscale_batch_manifest_sha256",
                        "multiscale_harness_contract_sha256",
                        "multiscale_scale_contract_sha256",
                    )) or
                not isinstance(identity.get("run_name"), str) or
                not identity["run_name"]):
            raise RuntimeError("ordinal run authentication identity is invalid")
        key = (geometry_sha256, scale)
        if key in grouped:
            raise RuntimeError("ordinal frame run grid repeats geometry/scale")
        grouped[key] = run
    geometries = sorted({geometry_sha for geometry_sha, _scale in grouped})
    if len(geometries) != 2:
        raise RuntimeError("ordinal frame labels require exactly two geometries")
    if len({
            run["run_identity"]["runtime_scene_evidence_sha256"]
            for run in runs
            }) != 1:
        raise RuntimeError(
            "ordinal runs disagree on full-cadence runtime scene evidence"
        )
    traces = {
        canonical_sha256(run.get("_runtime_scene_trace")) for run in runs
    }
    if len(traces) != 1:
        raise RuntimeError(
            "ordinal runs disagree on full-cadence runtime scene trace"
        )
    _validate_runtime_scene_trace(
        runs[0].get("_runtime_scene_trace"), source_frame_ids,
        common_identity.get("completion_sequence_contract"),
    )
    for geometry_sha256 in geometries:
        actual = sorted(
            scale for candidate_geometry, scale in grouped
            if candidate_geometry == geometry_sha256
        )
        if actual != list(ordinal_contract.SCALES):
            raise RuntimeError(
                "each ordinal geometry requires every 1.00..1.50/0.02 scale"
            )
        geometry_runs = [
            grouped[(geometry_sha256, scale)]
            for scale in ordinal_contract.SCALES
        ]
        if (len({
                run["run_identity"]["multiscale_batch_manifest_sha256"]
                for run in geometry_runs
                }) != 1 or
                len({
                    run["run_identity"][
                        "multiscale_harness_contract_sha256"
                    ] for run in geometry_runs
                }) != 1 or
                any(
                    run["run_identity"]["harness_contract_sha256"] !=
                    run["run_identity"][
                        "multiscale_scale_contract_sha256"
                    ] for run in geometry_runs
                )):
            raise RuntimeError(
                "ordinal geometry scale runs do not share one batch provenance"
            )

    identity = grouped[(geometries[0], ordinal_contract.SCALES[0])]
    expected_ids = [frame.get("frame_id") for frame in identity["frames"]]
    if expected_ids != label_frame_ids:
        raise RuntimeError(
            "ordinal run frames differ from authenticated label targets"
        )
    expected_ordinals = list(range(len(expected_ids)))
    source_ordinals = {
        frame_id: ordinal for ordinal, frame_id in enumerate(source_frame_ids)
    }
    expected_source_ordinals = [source_ordinals[value] for value in expected_ids]
    for run in runs:
        frame_ids = [frame.get("frame_id") for frame in run.get("frames", [])]
        ordinals = [frame.get("ordinal") for frame in run.get("frames", [])]
        observed_source_ordinals = [
            frame.get("source_ordinal") for frame in run.get("frames", [])
        ]
        if (frame_ids != expected_ids or ordinals != expected_ordinals or
                observed_source_ordinals != expected_source_ordinals):
            raise RuntimeError(
                "ordinal scale runs do not contain the same ordered targets"
            )
        for frame, source_ordinal in zip(run["frames"], expected_source_ordinals):
            try:
                _validate_selected_runtime_scene_row(
                    frame.get("runtime_scene_evidence"), frame["frame_id"],
                    source_ordinal,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    "ordinal target runtime scene differs from full source cadence"
                ) from error

    # When runs came through the evaluator-sidecar adapter, reuse the mature
    # scalar context validator to prove that only artistic scale differs
    # within each geometry. This includes current-frame depth, policy-off,
    # warp, model, preprocessing, and exact clip/raster contracts.
    with_results = ["_results_payload" in run for run in runs]
    if any(with_results):
        if not all(with_results):
            raise RuntimeError("ordinal run grid mixes authenticated and pure runs")
        baselines = []
        for geometry_sha256 in geometries:
            control = grouped[(
                geometry_sha256, ordinal_contract.SCALES[0]
            )]["_results_payload"]
            baselines.append(scalar_selector.policy_baseline_from_meta(
                control.get("meta", {})
            ))
            for scale in ordinal_contract.SCALES:
                candidate = grouped[(
                    geometry_sha256, scale
                )]["_results_payload"]
                scalar_selector.validate_context(
                    control, candidate, scale,
                    f"ordinal/{geometry_sha256}/{scale:.2f}",
                )
        if len({canonical_sha256(value) for value in baselines}) != 1:
            raise RuntimeError(
                "ordinal geometries use different policy baseline contracts"
            )
    return grouped, geometries, expected_ids


def _frame_input_provenance(frame_runs, input_variant_sha256, model):
    source_ids = {
        frame["artifact_sha256"].get("source") for frame in frame_runs
    }
    if len(source_ids) != 1:
        raise RuntimeError("scale/geometry runs disagree on source-frame bytes")
    depth_ids = {
        frame["artifact_sha256"].get("depth") for frame in frame_runs
    }
    if len(depth_ids) != 1:
        raise RuntimeError(
            "scale/geometry runs disagree on model depth for one source frame"
        )
    model_input_values = {
        frame["artifact_sha256"].get("model_input") for frame in frame_runs
    }
    if len(model_input_values) != 1:
        raise RuntimeError(
            "scale/geometry runs disagree on model-input artifact identity"
        )
    source_sha256 = next(iter(source_ids))
    depth_sha256 = next(iter(depth_ids))
    model_input_sha256 = next(iter(model_input_values))
    if (not isinstance(source_sha256, str) or
            not SHA256.fullmatch(source_sha256) or
            not isinstance(depth_sha256, str) or
            not SHA256.fullmatch(depth_sha256) or
            (model_input_sha256 is not None and (
                not isinstance(model_input_sha256, str) or
                not SHA256.fullmatch(model_input_sha256)))):
        raise RuntimeError("ordinal run artifact authentication is invalid")
    provenance = {
        "source_artifact_sha256": source_sha256,
        "input_variant_sha256": input_variant_sha256,
        "depth_model": model,
        # Older sidecars do not publish the post-preprocess tensor. Null is
        # explicit: the source/variant/model tuple is semantic provenance, not
        # a false claim that preprocessed tensor bytes were captured.
        "model_input_artifact_sha256": model_input_sha256,
    }
    return provenance, canonical_sha256(provenance), depth_sha256


def _runtime_scene_evidence(frame_runs):
    evidence = [frame.get("runtime_scene_evidence") for frame in frame_runs]
    present_evidence = [value for value in evidence if value is not None]
    if present_evidence:
        if (len(present_evidence) != len(evidence) or
                len({canonical_sha256(value) for value in present_evidence}) != 1):
            raise RuntimeError(
                "runtime scene evidence differs or is missing across scale runs"
            )
        scene_id = present_evidence[0]["runtime_scene_id"]
        if any(frame.get("runtime_scene_id") != scene_id for frame in frame_runs):
            raise RuntimeError(
                "runtime scene identity differs from its evidence"
            )
        return present_evidence[0]
    values = [frame.get("runtime_scene_id") for frame in frame_runs]
    present = [value for value in values if value is not None]
    if present and (len(present) != len(values) or len(set(present)) != 1):
        raise RuntimeError(
            "runtime scene identity differs or is missing across scale runs"
        )
    if present:
        raise RuntimeError(
            "runtime scene identity lacks authenticated per-frame evidence"
        )
    raise RuntimeError(
        "ordinal frame lacks authenticated runtime scene evidence"
    )


def _frame_safety_evidence(grouped, geometry_ids, ordinal, metric_specs,
                           scene_evidence):
    """Classify whether this frame has enough measured evidence for a target."""
    required = []
    for metric, spec in metric_specs.items():
        required_primary = (
            spec.get("role") == "primary" and
            spec.get("required_evidence") is True
        )
        if (spec.get("role") == "hard" or required_primary or
                "ordinal_hard_min" in spec or
                "ordinal_hard_max" in spec):
            required.append((metric, spec))
    missing = []
    for metric, spec in sorted(required):
        metric_missing = []
        for geometry_id in geometry_ids:
            missing_scales = [
                scale for scale in ordinal_contract.SCALES
                if grouped[(geometry_id, scale)]["frames"][ordinal][
                    "metrics"
                ].get(metric) is None
            ]
            if not missing_scales:
                continue
            entry = {
                "metric": metric,
                "geometry_sha256": geometry_id,
                "scales": missing_scales,
            }
            metric_missing.append(entry)
        missing.extend(metric_missing)
    status = "unproven" if missing else "proven"
    return {
        "schema": FRAME_SAFETY_EVIDENCE_SCHEMA,
        "contract": FRAME_SAFETY_EVIDENCE_CONTRACT,
        "status": status,
        "scene_boundary_exemption": False,
        "missing_required_evidence": missing,
        "exempt_missing_temporal_evidence": [],
    }


def _diagnostic_absolute_violations(grouped, geometry_ids, ordinal):
    result = []
    for geometry_id in geometry_ids:
        for scale in ordinal_contract.SCALES:
            frame = grouped[(geometry_id, scale)]["frames"][ordinal]
            violations = _normalize_diagnostic_violations(
                frame.get("diagnostic_violations", []),
                f"frame {frame.get('frame_id')}/{geometry_id}/{scale:.2f}",
            )
            if violations:
                result.append({
                    "geometry_sha256": geometry_id,
                    "scale": scale,
                    "violations": violations,
                })
    return result


def build_frame_label_bundle(runs, metric_specs,
                             expected_input_variant_sha256):
    """Join a two-geometry target-only render grid and classify every target."""
    if not isinstance(metric_specs, dict) or not metric_specs:
        raise RuntimeError("ordinal frame label metric contract is empty")
    metric_specs = {
        name: spec for name, spec in metric_specs.items()
        if spec.get("temporal_evidence") is not True
    }
    if not metric_specs:
        raise RuntimeError("ordinal target-only metric contract is empty")
    grouped, geometry_ids, frame_ids = _validate_normalized_runs(
        runs, expected_input_variant_sha256
    )
    first_run = runs[0]
    common = first_run["common_identity"]
    if (common.get("metric_sha256") != run_eval.metric_contract_sha() or
            common.get("thresholds_sha256") !=
            sha256_file(THRESHOLDS_PATH)):
        raise RuntimeError(
            "ordinal render evidence does not match current metric thresholds"
        )
    declared_primary = common.get("primary_metrics")
    declared_hard = common.get("hard_metrics")
    expected_primary = sorted(
        name for name, spec in metric_specs.items()
        if spec.get("role") == "primary"
    )
    expected_hard = sorted(
        name for name, spec in metric_specs.items()
        if spec.get("role") == "hard"
    )
    if (declared_primary is not None and declared_primary != expected_primary or
            declared_hard is not None and declared_hard != expected_hard):
        raise RuntimeError(
            "ordinal metric specifications differ from render-run evidence"
        )
    model = common["pipeline_without_scale"].get("model")
    if not isinstance(model, str) or not model:
        raise RuntimeError("ordinal runs have no depth-model provenance")
    geometries = [
        grouped[(geometry_id, ordinal_contract.SCALES[0])]["geometry"]
        for geometry_id in geometry_ids
    ]
    allowlist = geometry_contract.build_allowlist(geometries)
    allowlist_sha256 = geometry_contract.allowlist_sha256(allowlist)

    run_identities = [
        grouped[(geometry_id, scale)]["run_identity"]
        for geometry_id in geometry_ids
        for scale in ordinal_contract.SCALES
    ]
    if len({canonical_sha256(item) for item in run_identities}) != len(
            run_identities):
        raise RuntimeError("ordinal scale-run authentication identities repeat")

    frame_records = []
    for ordinal, frame_id in enumerate(frame_ids):
        all_frame_runs = [
            grouped[(geometry_id, scale)]["frames"][ordinal]
            for geometry_id in geometry_ids
            for scale in ordinal_contract.SCALES
        ]
        provenance, provenance_sha256, depth_sha256 = (
            _frame_input_provenance(
                all_frame_runs, expected_input_variant_sha256, model
            )
        )
        scene_evidence = _runtime_scene_evidence(all_frame_runs)
        scene_id = (
            scene_evidence["runtime_scene_id"]
            if scene_evidence is not None else None
        )
        source_ordinal = all_frame_runs[0]["source_ordinal"]
        if any(frame.get("source_ordinal") != source_ordinal
               for frame in all_frame_runs):
            raise RuntimeError(
                "scale/geometry runs disagree on source-frame ordinal"
            )
        safety_evidence = _frame_safety_evidence(
            grouped, geometry_ids, ordinal, metric_specs, scene_evidence
        )
        diagnostic_violations = _diagnostic_absolute_violations(
            grouped, geometry_ids, ordinal
        )
        geometry_frontiers = []
        evidence_refs = {}
        for geometry_id in geometry_ids:
            control = grouped[
                (geometry_id, ordinal_contract.SCALES[0])
            ]["frames"][ordinal]["metrics"]
            candidates = {
                scale: grouped[(geometry_id, scale)]["frames"][ordinal][
                    "metrics"
                ]
                for scale in ordinal_contract.SCALES
            }
            if safety_evidence["status"] == "proven":
                frontier = frontier_selector.select_clip_frontier(
                    control, candidates, metric_specs,
                    {
                        "expected_flat": common.get("expected_flat") is True,
                        "temporal_boundary_exemption": safety_evidence[
                            "scene_boundary_exemption"
                        ],
                    },
                )
                geometry = grouped[
                    (geometry_id, ordinal_contract.SCALES[0])
                ]["geometry"]
                geometry_frontiers.append(
                    geometry_merge.build_geometry_frontier(
                        provenance["source_artifact_sha256"],
                        expected_input_variant_sha256, geometry, frontier,
                    )
                )
            evidence_refs[geometry_id] = [
                grouped[(geometry_id, scale)]["run_identity"][
                    "frame_gate_evidence_sha256"
                ] for scale in ordinal_contract.SCALES
            ]
        intersection = (
            geometry_merge.intersect_geometry_frontiers(geometry_frontiers)
            if safety_evidence["status"] == "proven" else None
        )
        frame_records.append({
            "record": "frame_label",
            "clip": common["clip"],
            "frame_id": frame_id,
            "ordinal": ordinal,
            "source_ordinal": source_ordinal,
            "runtime_scene_id": scene_id,
            "runtime_scene_evidence": scene_evidence,
            "model_input_provenance": provenance,
            "model_input_provenance_sha256": provenance_sha256,
            "model_depth_artifact_sha256": depth_sha256,
            "scale_run_evidence_sha256_by_geometry": evidence_refs,
            "frame_safety_evidence": safety_evidence,
            "diagnostic_absolute_violations": diagnostic_violations,
            "geometry_intersection": intersection,
        })

    runtime_scene_trace = _validate_runtime_scene_trace(
        first_run.get("_runtime_scene_trace"), common["source_frame_ids"],
        common.get("completion_sequence_contract"),
    )
    runtime_scene_trace_sha256 = canonical_sha256(runtime_scene_trace)
    runtime_scene_by_frame = {
        row["source_frame_id"]: row for row in runtime_scene_trace
    }
    if any(
            frame["runtime_scene_evidence"] != runtime_scene_by_frame.get(
                frame["frame_id"]
            ) for frame in frame_records):
        raise RuntimeError(
            "ordinal selected runtime scenes differ from the full trace"
        )

    code_identity = current_code_identity()
    header = {
        "record": "header",
        "schema": BUNDLE_SCHEMA,
        "contract": BUNDLE_CONTRACT,
        "clip": common["clip"],
        "clip_sha1": common["clip_sha1"],
        "input_variant": first_run["input_variant"],
        "input_variant_sha256": expected_input_variant_sha256,
        "scale_thresholds": list(ordinal_contract.SCALES),
        "metric_specs_sha256": canonical_sha256(metric_specs),
        "metric_contract_sha256": run_eval.metric_contract_sha(),
        "thresholds_sha256": sha256_file(THRESHOLDS_PATH),
        "code_identity": code_identity,
        "code_identity_sha256": canonical_sha256(code_identity),
        "common_run_identity": common,
        "deployment_geometry_allowlist": allowlist,
        "deployment_geometry_allowlist_sha256": allowlist_sha256,
        "scale_run_identities": run_identities,
        "frame_count": len(frame_ids),
        "first_frame_id": frame_ids[0],
        "last_frame_id": frame_ids[-1],
        "source_frame_count": common["source_frame_count"],
        "source_frame_ids": common["source_frame_ids"],
        "source_frame_ids_sha256": common["source_frame_ids_sha256"],
        "runtime_scene_trace_contract": scene_contract.CONTRACT,
        "runtime_scene_trace": runtime_scene_trace,
        "runtime_scene_trace_sha256": runtime_scene_trace_sha256,
        "label_frame_ids": common["label_frame_ids"],
        "output_frame_count": common["output_frame_count"],
        "output_selected_frame_ids": common[
            "output_selected_frame_ids"
        ],
        "output_label_frames_sha256": common[
            "output_label_frames_sha256"
        ],
    }
    return header, frame_records


def _summary_from_records(header, frames, bundle_sha256, payload_sha256):
    bounds = Counter()
    failures = Counter()
    unproven = Counter()
    scenes = set()
    for frame in frames:
        target = frame["geometry_intersection"]
        safety = frame["frame_safety_evidence"]
        if safety["status"] == "unproven":
            unproven.update({
                item["metric"] for item in safety["missing_required_evidence"]
            })
            if frame["runtime_scene_id"] is not None:
                scenes.add(frame["runtime_scene_id"])
            continue
        key = (
            "right-censored" if target["right_censored"] else
            "left-censored" if target["left_censored"] else
            f"safe-{target['highest_proven_safe_scale']:.2f}-"
            f"unsafe-{target['first_proven_unsafe_scale']:.2f}"
        )
        bounds[key] += 1
        for geometry_failure in target["first_unsafe_failures"]:
            failures.update(geometry_failure["failure_causes"])
        if frame["runtime_scene_id"] is not None:
            scenes.add(frame["runtime_scene_id"])
    return {
        "schema": SUMMARY_SCHEMA,
        "contract": SUMMARY_CONTRACT,
        "label_bundle_sha256": bundle_sha256,
        "payload_sha256": payload_sha256,
        "clip": header["clip"],
        "input_variant_sha256": header["input_variant_sha256"],
        "deployment_geometry_allowlist_sha256": header[
            "deployment_geometry_allowlist_sha256"
        ],
        "frame_count": len(frames),
        "source_frame_count": header["source_frame_count"],
        "output_frame_count": header["output_frame_count"],
        "output_label_frames_sha256": header[
            "output_label_frames_sha256"
        ],
        "proven_frame_count": sum(
            frame["frame_safety_evidence"]["status"] == "proven"
            for frame in frames
        ),
        "unproven_frame_count": sum(
            frame["frame_safety_evidence"]["status"] == "unproven"
            for frame in frames
        ),
        "runtime_scene_count": header["common_run_identity"].get(
            "runtime_scene_count"
        ),
        "labeled_runtime_scene_count": len(scenes) if scenes else None,
        "frontier_bounds": dict(sorted(bounds.items())),
        "first_unsafe_failure_causes": dict(sorted(failures.items())),
        "unproven_required_metrics": dict(sorted(unproven.items())),
    }


def write_frame_label_bundle(path, summary_path, header, frames):
    """Atomically publish canonical JSONL labels and their compact summary."""
    path = Path(path).resolve()
    summary_path = Path(summary_path).resolve()
    payload = [header, *frames]
    payload_bytes = [canonical_bytes(record) for record in payload]
    payload_sha256 = hashlib.sha256(b"".join(payload_bytes)).hexdigest()
    trailer = {
        "record": "trailer",
        "payload_record_count": len(payload),
        "frame_count": len(frames),
        "payload_sha256": payload_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        for encoded in payload_bytes:
            stream.write(encoded)
        stream.write(canonical_bytes(trailer))
    temporary.replace(path)
    bundle_sha256 = sha256_file(path)
    summary = _summary_from_records(
        header, frames, bundle_sha256, payload_sha256
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = summary_path.with_name(summary_path.name + ".tmp")
    summary_temporary.write_bytes(canonical_bytes(summary))
    summary_temporary.replace(summary_path)
    validate_frame_label_summary(path, summary_path)
    return summary


def _validate_frame_safety_record(value, geometry_ids):
    fields = {
        "schema", "contract", "status", "scene_boundary_exemption",
        "missing_required_evidence", "exempt_missing_temporal_evidence",
    }
    if (not isinstance(value, dict) or set(value) != fields or
            value.get("schema") != FRAME_SAFETY_EVIDENCE_SCHEMA or
            value.get("contract") != FRAME_SAFETY_EVIDENCE_CONTRACT or
            value.get("status") not in {"proven", "unproven"} or
            not isinstance(value.get("scene_boundary_exemption"), bool)):
        raise RuntimeError("ordinal frame safety evidence is invalid")

    def validate_entries(entries, description):
        if not isinstance(entries, list):
            raise RuntimeError(f"ordinal frame {description} is not a list")
        normalized_order = []
        for item in entries:
            if (not isinstance(item, dict) or
                    set(item) != {"metric", "geometry_sha256", "scales"} or
                    not isinstance(item.get("metric"), str) or
                    not item["metric"] or
                    item.get("geometry_sha256") not in geometry_ids or
                    not isinstance(item.get("scales"), list) or
                    not item["scales"]):
                raise RuntimeError(f"ordinal frame {description} entry is invalid")
            scales = []
            for scale in item["scales"]:
                scales.append(_finite_scale(scale))
            if scales != sorted(set(scales)):
                raise RuntimeError(
                    f"ordinal frame {description} scale list is noncanonical"
                )
            normalized_order.append((item["metric"], item["geometry_sha256"]))
        if normalized_order != sorted(set(normalized_order)):
            raise RuntimeError(f"ordinal frame {description} order is noncanonical")
        return {
            (item["metric"], item["geometry_sha256"])
            for item in entries
        }

    missing = validate_entries(
        value["missing_required_evidence"], "missing-required evidence"
    )
    exempt = validate_entries(
        value["exempt_missing_temporal_evidence"],
        "exempt temporal evidence",
    )
    if missing & exempt:
        raise RuntimeError("ordinal frame missing/exempt evidence overlaps")
    if ((value["status"] == "unproven") != bool(missing) or
            value["scene_boundary_exemption"] != bool(exempt)):
        raise RuntimeError("ordinal frame safety status is inconsistent")
    return value


def _validate_diagnostic_absolute_violations(value, geometry_ids):
    if not isinstance(value, list):
        raise RuntimeError("ordinal frame diagnostic evidence is not a list")
    order = []
    for item in value:
        if (not isinstance(item, dict) or
                set(item) != {"geometry_sha256", "scale", "violations"} or
                item.get("geometry_sha256") not in geometry_ids):
            raise RuntimeError("ordinal frame diagnostic evidence is invalid")
        scale = _finite_scale(item.get("scale"))
        normalized = _normalize_diagnostic_violations(
            item.get("violations"), "ordinal frame diagnostic evidence"
        )
        if not normalized or normalized != item["violations"]:
            raise RuntimeError("ordinal frame diagnostic violations are noncanonical")
        order.append((item["geometry_sha256"], scale))
    if order != sorted(set(order)):
        raise RuntimeError("ordinal frame diagnostic evidence order is noncanonical")
    return value


def validate_frame_label_bundle(path):
    raw_lines = Path(path).read_bytes().splitlines(keepends=True)
    if len(raw_lines) < 3 or any(not line.strip() for line in raw_lines):
        raise RuntimeError("ordinal frame label bundle is empty/incomplete")
    try:
        records = [json.loads(line) for line in raw_lines]
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid ordinal label JSONL: {error}") from error
    if any(canonical_bytes(record) != raw
           for record, raw in zip(records, raw_lines)):
        raise RuntimeError("ordinal frame label bundle is not canonical JSONL")
    header, trailer = records[0], records[-1]
    if (header.get("record") != "header" or
            header.get("schema") != BUNDLE_SCHEMA or
            header.get("contract") != BUNDLE_CONTRACT):
        raise RuntimeError("ordinal frame label bundle header is stale")
    header_fields = {
        "record", "schema", "contract", "clip", "clip_sha1",
        "input_variant", "input_variant_sha256", "scale_thresholds",
        "metric_specs_sha256", "metric_contract_sha256",
        "thresholds_sha256", "code_identity", "code_identity_sha256",
        "common_run_identity",
        "deployment_geometry_allowlist",
        "deployment_geometry_allowlist_sha256", "scale_run_identities",
        "frame_count", "first_frame_id", "last_frame_id",
        "source_frame_count", "source_frame_ids",
        "source_frame_ids_sha256", "runtime_scene_trace_contract",
        "runtime_scene_trace", "runtime_scene_trace_sha256",
        "label_frame_ids",
        "output_frame_count", "output_selected_frame_ids",
        "output_label_frames_sha256",
    }
    if set(header) != header_fields:
        raise RuntimeError("ordinal frame label bundle header fields differ")
    expected_code_identity = current_code_identity()
    if (header.get("code_identity") != expected_code_identity or
            header.get("code_identity_sha256") !=
            canonical_sha256(expected_code_identity)):
        raise RuntimeError("ordinal frame label code identity is stale")
    if (header.get("metric_contract_sha256") !=
            run_eval.metric_contract_sha() or
            header.get("thresholds_sha256") !=
            sha256_file(THRESHOLDS_PATH)):
        raise RuntimeError("ordinal frame label metric contract is stale")
    if (trailer.get("record") != "trailer" or set(trailer) != {
            "record", "payload_record_count", "frame_count",
            "payload_sha256"}):
        raise RuntimeError("ordinal frame label bundle trailer is missing")
    payload = b"".join(raw_lines[:-1])
    frames = records[1:-1]
    if (trailer.get("payload_record_count") != len(records) - 1 or
            trailer.get("frame_count") != len(frames) or
            trailer.get("payload_sha256") != hashlib.sha256(payload).hexdigest()):
        raise RuntimeError("ordinal frame label bundle digest/count is invalid")
    source_ids = header.get("source_frame_ids")
    expected_ids = header.get("label_frame_ids")
    output_ids = header.get("output_selected_frame_ids")
    try:
        source_ids, expected_ids, output_ids = (
            run_eval.validate_selected_frame_gate_coverage(
                source_ids,
                {
                    "mode": "label-frames",
                    "label_frame_ids": expected_ids,
                    "output_frame_ids": output_ids,
                    "label_frames_sha256": header.get(
                        "output_label_frames_sha256"
                    ),
                },
            )
        )
    except ValueError as error:
        raise RuntimeError(
            "ordinal frame label selection identity is invalid"
        ) from error
    actual_ids = [frame.get("frame_id") for frame in frames]
    if (actual_ids != expected_ids or
            header.get("frame_count") != len(frames) or
            header.get("source_frame_count") != len(source_ids) or
            header.get("source_frame_ids_sha256") !=
            run_eval.frame_id_sequence_sha256(source_ids) or
            header.get("output_frame_count") != len(output_ids) or
            header.get("first_frame_id") != expected_ids[0] or
            header.get("last_frame_id") != expected_ids[-1]):
        raise RuntimeError("ordinal frame label bundle coverage is incomplete")
    runtime_scene_trace = _validate_runtime_scene_trace(
        header.get("runtime_scene_trace"), source_ids,
        header.get("common_run_identity", {}).get(
            "completion_sequence_contract"
        ),
    )
    if (header.get("runtime_scene_trace_contract") !=
            scene_contract.CONTRACT or
            header.get("runtime_scene_trace_sha256") !=
            canonical_sha256(runtime_scene_trace)):
        raise RuntimeError(
            "ordinal full-cadence runtime scene trace identity is stale"
        )
    runtime_scene_by_frame = {
        row["source_frame_id"]: row for row in runtime_scene_trace
    }
    try:
        input_color.validate_input_variant(header.get("input_variant"))
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError("ordinal frame label input variant is invalid") from error
    if (header.get("input_variant_sha256") !=
            input_color.input_variant_sha256(header["input_variant"])):
        raise RuntimeError("ordinal frame label input-variant identity is stale")
    if (header.get("scale_thresholds") != list(ordinal_contract.SCALES) or
            not isinstance(header.get("metric_specs_sha256"), str) or
            not SHA256.fullmatch(header["metric_specs_sha256"])):
        raise RuntimeError("ordinal frame label threshold contract is stale")
    geometry_contract.validate_allowlist(header.get(
        "deployment_geometry_allowlist"
    ))
    if (header.get("deployment_geometry_allowlist_sha256") !=
            geometry_contract.allowlist_sha256(
                header["deployment_geometry_allowlist"]
            )):
        raise RuntimeError("ordinal frame label bundle geometry identity is stale")
    geometry_ids = {
        canonical_sha256(value)
        for value in header["deployment_geometry_allowlist"]["tuples"]
    }
    if len(geometry_ids) != 2:
        raise RuntimeError("ordinal frame label bundle lacks two geometries")
    common_identity = header.get("common_run_identity")
    if (not isinstance(common_identity, dict) or
            not isinstance(common_identity.get("executable_sha256"), str) or
            not SHA256.fullmatch(common_identity["executable_sha256"])):
        raise RuntimeError(
            "ordinal frame label executable provenance is invalid"
        )
    common_selection = {
        "source_frame_count": len(source_ids),
        "source_frame_ids": source_ids,
        "source_frame_ids_sha256": header["source_frame_ids_sha256"],
        "label_frame_count": len(expected_ids),
        "label_frame_ids": expected_ids,
        "output_frame_count": len(output_ids),
        "output_selected_frame_ids": output_ids,
        "output_label_frames_sha256": header[
            "output_label_frames_sha256"
        ],
        "output_selection_contract":
            run_eval.SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT,
    }
    if any(common_identity.get(key) != value
           for key, value in common_selection.items()):
        raise RuntimeError(
            "ordinal frame label common selected-frame identity differs"
        )
    run_identities = header.get("scale_run_identities")
    if not isinstance(run_identities, list) or len(run_identities) != (
            2 * ordinal_contract.FRONTIER_SIZE):
        raise RuntimeError("ordinal frame label scale-run identities are incomplete")
    evidence_by_key = {}
    run_identity_fields = {
        "frame_gate_evidence_sha256", "results_sha256",
        "harness_contract_sha256", "runtime_scene_evidence_sha256",
        "run_name", "geometry_sha256", "scale",
        "multiscale_batch_manifest_sha256",
        "multiscale_harness_contract_sha256",
        "multiscale_scale_contract_sha256",
    }
    for identity in run_identities:
        if not isinstance(identity, dict) or set(identity) != run_identity_fields:
            raise RuntimeError("ordinal frame label scale-run identity is invalid")
        try:
            scale = _finite_scale(identity.get("scale"))
        except RuntimeError as error:
            raise RuntimeError(
                "ordinal frame label scale-run identity is invalid"
            ) from error
        geometry_id = identity.get("geometry_sha256")
        evidence_sha256 = identity.get("frame_gate_evidence_sha256")
        key = (geometry_id, scale)
        if (geometry_id not in geometry_ids or key in evidence_by_key or
                not isinstance(evidence_sha256, str) or
                not SHA256.fullmatch(evidence_sha256) or
                any(not isinstance(identity.get(field), str) or
                    not SHA256.fullmatch(identity[field])
                    for field in (
                        "results_sha256", "harness_contract_sha256",
                        "runtime_scene_evidence_sha256",
                        "multiscale_batch_manifest_sha256",
                        "multiscale_harness_contract_sha256",
                        "multiscale_scale_contract_sha256",
                    )) or not isinstance(identity.get("run_name"), str) or
                not identity["run_name"]):
            raise RuntimeError("ordinal frame label scale-run lattice is invalid")
        evidence_by_key[key] = evidence_sha256
    expected_run_keys = {
        (geometry_id, scale) for geometry_id in geometry_ids
        for scale in ordinal_contract.SCALES
    }
    if set(evidence_by_key) != expected_run_keys:
        raise RuntimeError("ordinal frame label scale-run lattice is incomplete")
    if len({
            identity["runtime_scene_evidence_sha256"]
            for identity in run_identities
            }) != 1:
        raise RuntimeError(
            "ordinal frame label full-cadence runtime scene identity differs"
        )
    for geometry_id in geometry_ids:
        identities = [
            identity for identity in run_identities
            if identity["geometry_sha256"] == geometry_id
        ]
        if (len({
                identity["multiscale_batch_manifest_sha256"]
                for identity in identities
                }) != 1 or
                len({
                    identity["multiscale_harness_contract_sha256"]
                    for identity in identities
                }) != 1 or
                any(
                    identity["harness_contract_sha256"] !=
                    identity["multiscale_scale_contract_sha256"]
                    for identity in identities
                )):
            raise RuntimeError(
                "ordinal frame label multiscale provenance lattice is invalid"
            )
    frame_fields = {
        "record", "clip", "frame_id", "ordinal", "source_ordinal",
        "runtime_scene_id",
        "runtime_scene_evidence",
        "model_input_provenance", "model_input_provenance_sha256",
        "model_depth_artifact_sha256",
        "scale_run_evidence_sha256_by_geometry", "frame_safety_evidence",
        "diagnostic_absolute_violations", "geometry_intersection",
    }
    source_ordinals = {
        frame_id: ordinal for ordinal, frame_id in enumerate(source_ids)
    }
    for ordinal, frame in enumerate(frames):
        if (set(frame) != frame_fields or
                frame.get("record") != "frame_label" or
                frame.get("ordinal") != ordinal or
                frame.get("source_ordinal") != source_ordinals[
                    frame.get("frame_id")
                ] or
                frame.get("clip") != header.get("clip")):
            raise RuntimeError("ordinal frame label record identity is invalid")
        scene_id = frame.get("runtime_scene_id")
        if scene_id is not None and (
                not isinstance(scene_id, int) or isinstance(scene_id, bool) or
                scene_id < 0):
            raise RuntimeError("ordinal frame runtime scene identity is invalid")
        scene_evidence = _validate_selected_runtime_scene_row(
            frame.get("runtime_scene_evidence"), frame["frame_id"],
            frame["source_ordinal"],
        )
        if scene_evidence.get("runtime_scene_id") != scene_id:
            raise RuntimeError("ordinal frame runtime scene evidence is invalid")
        if scene_evidence != runtime_scene_by_frame.get(frame["frame_id"]):
            raise RuntimeError(
                "ordinal selected runtime scene differs from the full trace"
            )
        provenance = frame.get("model_input_provenance")
        provenance_fields = {
            "source_artifact_sha256", "input_variant_sha256", "depth_model",
            "model_input_artifact_sha256",
        }
        if (not isinstance(provenance, dict) or
                set(provenance) != provenance_fields or
                provenance.get("input_variant_sha256") !=
                header["input_variant_sha256"] or
                not isinstance(provenance.get("source_artifact_sha256"), str) or
                not SHA256.fullmatch(provenance["source_artifact_sha256"]) or
                not isinstance(provenance.get("depth_model"), str) or
                not provenance["depth_model"] or
                (provenance.get("model_input_artifact_sha256") is not None and (
                    not isinstance(
                        provenance["model_input_artifact_sha256"], str
                    ) or not SHA256.fullmatch(
                        provenance["model_input_artifact_sha256"]
                    ))) or frame.get("model_input_provenance_sha256") !=
                canonical_sha256(provenance)):
            raise RuntimeError("ordinal frame model-input provenance is stale")
        depth_sha256 = frame.get("model_depth_artifact_sha256")
        if (not isinstance(depth_sha256, str) or
                not SHA256.fullmatch(depth_sha256)):
            raise RuntimeError("ordinal frame depth identity is invalid")
        evidence = frame.get("scale_run_evidence_sha256_by_geometry")
        if not isinstance(evidence, dict) or set(evidence) != geometry_ids:
            raise RuntimeError("ordinal frame evidence references are incomplete")
        for geometry_id, values in evidence.items():
            expected = [
                evidence_by_key[(geometry_id, scale)]
                for scale in ordinal_contract.SCALES
            ]
            if values != expected:
                raise RuntimeError("ordinal frame evidence references are stale")
        safety = _validate_frame_safety_record(
            frame.get("frame_safety_evidence"), geometry_ids
        )
        if (safety["scene_boundary_exemption"] and
                (not isinstance(scene_evidence, dict) or
                 not (scene_evidence.get("scene_start") is True or
                      scene_evidence.get("hard_cut") is True))):
            raise RuntimeError(
                "ordinal temporal exemption is not on a runtime scene boundary"
            )
        _validate_diagnostic_absolute_violations(
            frame.get("diagnostic_absolute_violations"), geometry_ids
        )
        if safety["status"] == "unproven":
            if frame.get("geometry_intersection") is not None:
                raise RuntimeError("unproven ordinal frame carries a safety target")
            continue
        if frame.get("geometry_intersection") is None:
            raise RuntimeError("proven ordinal frame lacks a safety target")
        geometry_merge.validate_geometry_intersection(
            frame["geometry_intersection"]
        )
        intersection = frame["geometry_intersection"]
        if (intersection.get("source_sha256") !=
                provenance["source_artifact_sha256"] or
                intersection.get("input_variant_sha256") !=
                header["input_variant_sha256"] or {
                    item["deployment_geometry_sha256"]
                    for item in intersection["geometry_frontiers"]
                } != geometry_ids):
            raise RuntimeError(
                "ordinal frame intersection provenance is inconsistent"
            )
    return records


def validate_frame_label_summary(bundle_path, summary_path, records=None):
    """Validate a canonical summary as an exact projection of its bundle."""
    bundle_path = Path(bundle_path)
    summary_path = Path(summary_path)
    authenticated_records = validate_frame_label_bundle(bundle_path)
    if records is not None and records != authenticated_records:
        raise RuntimeError(
            "ordinal frame label summary records differ from its bundle"
        )
    summary = _load_json_object(summary_path, "ordinal frame label summary")
    try:
        summary_bytes = summary_path.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"cannot read ordinal frame label summary: {summary_path}"
        ) from error
    if summary_bytes != canonical_bytes(summary):
        raise RuntimeError("ordinal frame label summary is not canonical JSON")
    header = authenticated_records[0]
    frames = authenticated_records[1:-1]
    trailer = authenticated_records[-1]
    expected = _summary_from_records(
        header,
        frames,
        sha256_file(bundle_path),
        trailer["payload_sha256"],
    )
    if summary != expected:
        raise RuntimeError(
            "ordinal frame label summary differs from its authenticated bundle"
        )
    return summary


def load_run_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "contract", "clip", "input_variant_sha256", "runs",
    }
    if (not isinstance(value, dict) or set(value) != required or
            value.get("schema") != RUN_GRID_SCHEMA or
            value.get("contract") != RUN_GRID_CONTRACT or
            not isinstance(value.get("clip"), str) or not value["clip"] or
            not isinstance(value.get("runs"), list) or not value["runs"]):
        raise RuntimeError("ordinal frame run manifest is invalid")
    try:
        run_eval.validate_path_component(value["clip"], "clip name")
    except ValueError as error:
        raise RuntimeError("ordinal frame run manifest clip is invalid") from error
    run_paths = []
    for item in value["runs"]:
        if not isinstance(item, str) or not item:
            raise RuntimeError("ordinal frame run manifest path is invalid")
        candidate = Path(item)
        run_paths.append(
            candidate.resolve() if candidate.is_absolute()
            else (path.parent / candidate).resolve()
        )
    return value, run_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    manifest, run_paths = load_run_manifest(args.run_manifest)
    thresholds_path = Path(args.thresholds).resolve()
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    metric_specs = run_eval.target_only_gate_thresholds(thresholds)["metrics"]
    runs = [
        parse_frame_gate_evidence(path, manifest["clip"])
        for path in run_paths
    ]
    threshold_hashes = {
        run["common_identity"].get("thresholds_sha256") for run in runs
    }
    if threshold_hashes != {sha256_file(thresholds_path)}:
        raise RuntimeError("thresholds file differs from authenticated render runs")
    header, frames = build_frame_label_bundle(
        runs, metric_specs, manifest["input_variant_sha256"]
    )
    summary = write_frame_label_bundle(
        args.output, args.summary, header, frames
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
