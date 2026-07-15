#!/usr/bin/env python3
"""Create the fail-closed manifest required to deploy an artistic policy.

Export is only a candidate build. Promotion binds that exact ONNX to every later approval:
sealed labels, depth neutrality, current core/extended render gates, and a named headset review.
Paths remain audit hints only; every consumed artifact is authenticated by its file bytes.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
from pathlib import Path

import cv2

from artistic_geometry_contract import (
    MAX_HEIGHT,
    MAX_WIDTH,
    allowlist_sha256,
    aspect_aligned_dims,
    canonical_geometry_tuple,
    tuple_key,
    validate_allowlist,
    validate_geometry_tuple,
)


THIS_DIR = Path(__file__).resolve().parent
SBSBENCH_DIR = THIS_DIR.parent / "sbsbench"
REPO_ROOT = THIS_DIR.parents[1]
DEPLOYMENT_CONTRACT = "apollo-artistic-policy-deployment-v1"
NEUTRALITY_CONTRACT = "apollo-dav2-srgb-native-capped-v1"
SEALED_APPROVAL_CONTRACT = "sealed-test-artistic-policy-v2"
POLICY_CONTRACT = "safe-frontier-multistyle-apollo-v1"
POLICY_FEATURE_CONTRACT = "multiscale-dino-depth-dpt-stats-v1"
POLICY_WARP_CONTRACT = "apollo-safe-frontier-v1"
HARNESS_SCHEMA = 24
EVAL_SCHEMA = 29
MAX_NORMALIZED_MEAN_DRIFT = 1.0 / 1024.0
MAX_NORMALIZED_P99_DRIFT = 2.0 / 1024.0
SEALED_EVALUATION_SCHEMA = 11
MAX_UNSAFE_CEILING_OVERSHOOT_SCALE = 0.05
MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE = 0.01
EXACT_DISPARITY_CONTRACT = (
    "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars"
)
UNCLAMPED_DISPARITY_CONTRACT = (
    "unclamped_full_binocular_normalized_at_artistic_scale_1_"
    "output_eye_raster_zero_bars"
)
ARTISTIC_DISPARITY_CONTRACT = (
    "clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_"
    "times_aspect_scale_times_content_scale_x)"
)
BASELINE_RESULT_FIELDS = (
    "profile",
    "pop_strength",
    "adaptive_pop",
    "adaptive_pop_max",
    "ema",
    "ema_edge_change",
    "ema_edge_gradient",
    "ema_edge_strength",
    "minmax_ema",
    "subject_lock",
    "subject_recenter",
    "subject_stretch",
    "depth_short_side",
    "depth_max_aspect",
    "zero_plane",
    "depth_step",
    "depth_compensation",
    "literal_bestv2",
)
POLICY_BASELINE_KEYS = set(BASELINE_RESULT_FIELDS) | {
    "depth_model",
    "harness_schema",
    "eval_schema",
    "warp_contract",
    "policy_warp_source_sha256",
    "metric_sha256",
}
POLICY_OUTPUT_SEMANTICS = {
    "artistic_global_0": "safe_scale_ceiling",
    "artistic_global_1": "safe_ceiling_confidence",
    "confidence_semantics": "hard actionable probability",
    "action_threshold": 0.5,
    "preset_rules": {
        "safe_cap": (
            "safe_ceiling_confidence >= 0.5 ? "
            "clamp(safe_scale_ceiling, 1.0, 1.5) : 1.0"
        ),
        "clean": "1.0",
        "balanced": "1.0 + 0.5 * (safe_cap - 1.0)",
        "immersive": "safe_cap",
    },
}
POLICY_RUNTIME = {
    "confidence_semantics": "hard actionable probability",
    "action_threshold": 0.5,
    "inactive_ceiling": 1.0,
    "ceiling_bounds": [1.0, 1.5],
    "preset_rules": {
        "safe_cap": "confidence >= 0.5 ? clamp(ceiling, 1.0, 1.5) : 1.0",
        "clean": "1.0",
        "balanced": "1.0 + 0.5 * (safe_cap - 1.0)",
        "immersive": "safe_cap",
    },
}


def _normalized_text(path):
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace(
        "\r", "\n"
    )


def _cmake_contract_files(variable):
    """Read the source list used to compile Apollo's contract identities.

    The CMake list is the single source of truth. Parsing it here avoids a second
    hand-maintained file list that could approve a policy against a different
    implementation boundary than the runtime binary.
    """
    contract_cmake = REPO_ROOT / "cmake" / "prep" / "artistic_warp_contract.cmake"
    contents = _normalized_text(contract_cmake)
    match = re.search(
        rf"set\({re.escape(variable)}\s+(.*?)\)", contents, re.DOTALL
    )
    if match is None:
        raise RuntimeError(f"cannot find {variable} in {contract_cmake}")
    relative_paths = re.findall(
        r'"\$\{CMAKE_SOURCE_DIR\}/([^"\r\n]+)"', match.group(1)
    )
    if not relative_paths:
        raise RuntimeError(f"{variable} has no repository files")
    return tuple(relative_paths)


def current_contract_identities():
    """Reproduce the exact CMake identities compiled into Apollo."""
    warp_files = _cmake_contract_files("APOLLO_ARTISTIC_WARP_CONTRACT_FILES")
    warp_payload = "".join(
        f"{relative}\n{_normalized_text(REPO_ROOT / relative)}\n"
        for relative in warp_files
    )
    metric_files = _cmake_contract_files("APOLLO_ARTISTIC_METRIC_CONTRACT_FILES")
    metric_payload = "".join(
        f"{Path(relative).name}{_normalized_text(REPO_ROOT / relative)}"
        for relative in metric_files
    )
    return {
        "policy_warp_source_sha256": hashlib.sha256(
            warp_payload.encode("utf-8")
        ).hexdigest(),
        "metric_sha256": hashlib.sha256(
            metric_payload.encode("utf-8")
        ).hexdigest()[:16],
    }


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hash(value, length=64):
    return (
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None
    )


def _require_hash(value, origin, length=64):
    if not _is_hash(value, length):
        raise RuntimeError(f"{origin} is not a lowercase SHA-256 identity")
    return value


def _read_hashed_json(path: Path, origin: str):
    try:
        data = path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {origin}: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{origin} is not a JSON object")
    return payload, hashlib.sha256(data).hexdigest()


def _require_nonempty_string(value, origin):
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{origin} must be a non-empty string")
    return value.strip()


def _require_render_timestamp(value, origin):
    value = _require_nonempty_string(value, origin)
    if re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?"
            r"(?:Z|[+-]\d{2}:\d{2})?", value) is None:
        raise RuntimeError(f"{origin} is not an ISO-8601 render timestamp")
    try:
        datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"{origin} is not a valid render timestamp") from error
    return value


def _same(left, right, origin):
    if left != right:
        raise RuntimeError(f"{origin} differs: {left!r} != {right!r}")


def _validate_unsafe_ceiling_overshoot(payload, decision):
    evidence = payload.get("unsafe_ceiling_overshoot")
    if not isinstance(evidence, dict):
        raise RuntimeError("sealed-test evaluation lacks unsafe-ceiling evidence")
    numeric = {}
    for key in (
            "maximum_scale", "maximum_limit_scale",
            "film_balanced_mean_scale", "film_balanced_mean_limit_scale",
            "film_balanced_overshoot_rate_pct"):
        value = evidence.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value))):
            raise RuntimeError(
                f"sealed-test evaluation has invalid unsafe-ceiling {key}"
            )
        numeric[key] = float(value)
    if (numeric["maximum_scale"] < 0.0 or
            numeric["film_balanced_mean_scale"] < 0.0 or
            not 0.0 <= numeric["film_balanced_overshoot_rate_pct"] <= 100.0):
        raise RuntimeError("sealed-test evaluation has invalid unsafe-ceiling evidence")
    if (abs(numeric["maximum_limit_scale"] -
            MAX_UNSAFE_CEILING_OVERSHOOT_SCALE) > 1e-12 or
            abs(numeric["film_balanced_mean_limit_scale"] -
                MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE) > 1e-12):
        raise RuntimeError("sealed-test evaluation uses different unsafe-ceiling limits")
    if (evidence.get("maximum_pass") is not True or
            evidence.get("film_balanced_mean_pass") is not True or
            numeric["maximum_scale"] > MAX_UNSAFE_CEILING_OVERSHOOT_SCALE + 1e-9 or
            numeric["film_balanced_mean_scale"] >
            MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE + 1e-9):
        raise RuntimeError("sealed-test evaluation failed unsafe-ceiling guards")
    guards = decision.get("guards")
    if (not isinstance(guards, dict) or
            decision.get("unsafe_overshoot_guard_required") is not True or
            guards.get("unsafe_ceiling_maximum") is not True or
            guards.get("unsafe_ceiling_film_balanced_mean") is not True):
        raise RuntimeError("sealed-test decision lacks unsafe-ceiling guards")
    if decision.get("unsafe_ceiling_overshoot") != evidence:
        raise RuntimeError("sealed-test decision has inconsistent unsafe-ceiling evidence")
    return evidence


def _validate_policy_sidecar_contract(metadata):
    _same(metadata.get("policy_contract"), POLICY_CONTRACT, "policy contract")
    _same(
        metadata.get("policy_feature_contract"),
        POLICY_FEATURE_CONTRACT,
        "policy feature contract",
    )
    _same(
        metadata.get("input"),
        {
            "name": "pixel_values",
            "dtype": "float32",
            "shape": [1, 3, "H", "W"],
        },
        "policy input contract",
    )
    _same(
        metadata.get("outputs"),
        {
            "predicted_depth": {
                "dtype": "float32",
                "shape": [1, "H", "W"],
            },
            "artistic_global": {
                "dtype": "float32",
                "shape": [1, 2],
                "channels": [
                    "safe_scale_ceiling",
                    "safe_ceiling_confidence",
                ],
            },
        },
        "policy outputs contract",
    )
    _same(
        metadata.get("output_semantics"),
        POLICY_OUTPUT_SEMANTICS,
        "policy output semantics",
    )
    _same(metadata.get("bounds"), {"scale_delta_max": 0.5}, "policy bounds")
    _same(metadata.get("runtime"), POLICY_RUNTIME, "policy runtime")


def canonical_suite_clips(suite):
    directory = SBSBENCH_DIR / (
        "baselines" if suite == "core" else "baselines_extended"
    )
    clips = {path.stem for path in directory.glob("*.json")}
    if not clips:
        raise RuntimeError(f"canonical {suite} suite has no committed baselines")
    return clips


def canonical_core_first_frame_hashes():
    identities = {}
    for clip in sorted(canonical_suite_clips("core")):
        candidates = sorted(
            path for path in (SBSBENCH_DIR / "clips" / clip).glob("frame_*.*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not candidates:
            raise RuntimeError(f"canonical core clip {clip} has no source frames")
        identities[clip] = sha256(candidates[0])
    return identities


def validate_export_chain(onnx, metadata_path, checkpoint, evaluation):
    onnx = Path(onnx).resolve()
    metadata_path = Path(metadata_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    evaluation = Path(evaluation).resolve()
    onnx_hash = sha256(onnx)
    checkpoint_hash = sha256(checkpoint)
    metadata, metadata_hash = _read_hashed_json(
        metadata_path, "artistic-policy metadata"
    )
    evaluation_payload, evaluation_hash = _read_hashed_json(
        evaluation, "sealed-test evaluation"
    )

    if metadata.get("schema") != 4:
        raise RuntimeError("artistic-policy metadata schema is not 4")
    _validate_policy_sidecar_contract(metadata)
    deployed_model = _require_nonempty_string(
        metadata.get("deployed_model"), "deployed model"
    )
    _same(onnx.stem, deployed_model, "candidate ONNX stem")
    _same(metadata.get("onnx_sha256"), onnx_hash, "candidate ONNX hash")
    _require_hash(metadata.get("depth_weights_sha256"), "depth weights")
    metric_hash = _require_hash(
        metadata.get("metric_sha256"), "metric contract", length=16
    )
    baseline = metadata.get("policy_baseline")
    if not isinstance(baseline, dict):
        raise RuntimeError("artistic-policy metadata lacks a policy baseline")
    if set(baseline) != POLICY_BASELINE_KEYS:
        raise RuntimeError(
            "artistic-policy policy_baseline has missing or unknown fields"
        )
    current_contracts = current_contract_identities()
    _same(
        metric_hash,
        current_contracts["metric_sha256"],
        "current compiled metric contract",
    )
    _same(baseline.get("metric_sha256"), metric_hash, "baseline metric contract")
    warp_hash = _require_hash(
        baseline.get("policy_warp_source_sha256"), "warp source contract"
    )
    _same(
        warp_hash,
        current_contracts["policy_warp_source_sha256"],
        "current compiled warp contract",
    )
    _same(baseline.get("harness_schema"), HARNESS_SCHEMA, "harness schema")
    _same(baseline.get("eval_schema"), EVAL_SCHEMA, "evaluation schema")
    _same(
        baseline.get("warp_contract"), POLICY_WARP_CONTRACT, "warp contract"
    )
    base_depth_model = _require_nonempty_string(
        metadata.get("base_depth_model"), "base depth model"
    )
    _same(
        baseline.get("depth_model"), base_depth_model,
        "baseline base-depth model",
    )
    geometry_allowlist = metadata.get("deployment_geometry_allowlist")
    validate_allowlist(geometry_allowlist)
    geometry_hash = allowlist_sha256(geometry_allowlist)
    _same(
        metadata.get("deployment_geometry_allowlist_sha256"), geometry_hash,
        "metadata deployment geometry identity",
    )

    approval = metadata.get("approval_contract")
    if not isinstance(approval, dict):
        raise RuntimeError("artistic-policy metadata lacks sealed-test approval")
    required_approval = {
        "contract": SEALED_APPROVAL_CONTRACT,
        "evaluation_schema": SEALED_EVALUATION_SCHEMA,
        "split": "test",
        "decision_accepted": True,
        "evaluation_sha256": evaluation_hash,
        "checkpoint_sha256": checkpoint_hash,
        "metric_sha256": metric_hash,
    }
    for key, expected in required_approval.items():
        _same(approval.get(key), expected, f"sealed approval {key}")
    _same(
        metadata.get("evaluation_sha256"), evaluation_hash,
        "metadata evaluation hash",
    )
    for key, length in (
            ("active_split_sha256", 64),
            ("label_fitter_identity_sha256", 64),
            ("test_labels_sha256", 64),
            ("deployment_geometry_allowlist_sha256", 64)):
        _require_hash(approval.get(key), f"sealed approval {key}", length)
    _same(
        approval["deployment_geometry_allowlist_sha256"], geometry_hash,
        "sealed approval deployment geometry identity",
    )
    productions = approval.get("sealed_test_productions")
    if (
        not isinstance(productions, list)
        or not productions
        or any(not isinstance(value, str) or not value for value in productions)
        or productions != sorted(set(productions))
    ):
        raise RuntimeError("sealed approval production identities are invalid")

    if evaluation_payload.get("schema") != SEALED_EVALUATION_SCHEMA:
        raise RuntimeError(
            f"sealed-test evaluation schema is not {SEALED_EVALUATION_SCHEMA}"
        )
    decision = evaluation_payload.get("decision")
    if not isinstance(decision, dict) or decision.get("accepted") is not True:
        raise RuntimeError("sealed-test evaluation is not accepted")
    unsafe_ceiling_overshoot = _validate_unsafe_ceiling_overshoot(
        evaluation_payload, decision
    )
    _same(
        approval.get("unsafe_ceiling_overshoot"),
        unsafe_ceiling_overshoot,
        "sealed approval unsafe-ceiling evidence",
    )
    evaluation_identities = {
        "split": "test",
        "checkpoint_sha256": checkpoint_hash,
        "active_split_sha256": approval["active_split_sha256"],
        "metric_sha256": metric_hash,
        "label_fitter_identity_sha256": approval[
            "label_fitter_identity_sha256"
        ],
        "test_labels_sha256": approval["test_labels_sha256"],
        "deployment_geometry_allowlist_sha256": geometry_hash,
        "deployment_geometry_allowlist": geometry_allowlist,
        "val_films": productions,
    }
    for key, expected in evaluation_identities.items():
        _same(evaluation_payload.get(key), expected, f"evaluation {key}")

    return {
        "deployed_model": deployed_model,
        "base_depth_model": base_depth_model,
        "onnx_sha256": onnx_hash,
        "metadata_sha256": metadata_hash,
        "checkpoint_sha256": checkpoint_hash,
        "evaluation_sha256": evaluation_hash,
        "metric_sha256": metric_hash,
        "policy_warp_source_sha256": warp_hash,
        "active_split_sha256": approval["active_split_sha256"],
        "label_fitter_identity_sha256": approval[
            "label_fitter_identity_sha256"
        ],
        "test_labels_sha256": approval["test_labels_sha256"],
        "deployment_geometry_allowlist": geometry_allowlist,
        "deployment_geometry_allowlist_sha256": geometry_hash,
        "sealed_test_productions": productions,
        "policy_baseline": baseline,
    }


def validate_neutrality_report(
        report_path, onnx, reference, model, expected_image_hashes):
    report_path = Path(report_path).resolve()
    onnx = Path(onnx).resolve()
    reference = Path(reference).resolve()
    report, report_hash = _read_hashed_json(report_path, "neutrality report")
    if report.get("schema") != 4:
        raise RuntimeError("neutrality report schema is not 4")
    _same(
        report.get("preprocessing_contract"), NEUTRALITY_CONTRACT,
        "neutrality preprocessing contract",
    )
    if report.get("passed") is not True:
        raise RuntimeError("depth-neutrality report did not pass")
    preprocessing = report.get("preprocessing")
    expected_preprocessing = {
        "depth_short_side": model["policy_baseline"]["depth_short_side"],
        "depth_max_aspect": model["policy_baseline"]["depth_max_aspect"],
        "max_width": MAX_WIDTH,
        "max_height": MAX_HEIGHT,
        "resize_interpolation": "opencv-inter-linear",
        "color_conversion": "opencv-bgr8-to-rgb-srgb",
    }
    _same(preprocessing, expected_preprocessing, "neutrality preprocessing")
    candidate = report.get("candidate")
    reference_entry = report.get("reference")
    if not isinstance(candidate, dict) or not isinstance(reference_entry, dict):
        raise RuntimeError("neutrality model identities are missing")
    candidate_hash = sha256(onnx)
    reference_hash = sha256(reference)
    if candidate_hash == reference_hash:
        raise RuntimeError("neutrality reference and candidate are identical")
    _same(candidate.get("sha256"), candidate_hash, "neutrality candidate hash")
    _same(reference_entry.get("sha256"), reference_hash,
          "neutrality reference hash")
    _same(candidate_hash, model["onnx_sha256"], "promoted candidate hash")
    _same(onnx.stem, model["deployed_model"], "promoted candidate model")
    _same(reference.stem, model["base_depth_model"], "neutrality reference model")

    limits = report.get("limits")
    if not isinstance(limits, dict):
        raise RuntimeError("neutrality report lacks fixed limits")
    mean_limit = limits.get("production_normalized_mean_abs")
    p99_limit = limits.get("production_normalized_p99_abs")
    if (
        not isinstance(mean_limit, (int, float))
        or isinstance(mean_limit, bool)
        or not math.isfinite(mean_limit)
        or mean_limit <= 0.0
        or mean_limit > MAX_NORMALIZED_MEAN_DRIFT
        or not isinstance(p99_limit, (int, float))
        or isinstance(p99_limit, bool)
        or not math.isfinite(p99_limit)
        or p99_limit <= 0.0
        or p99_limit > MAX_NORMALIZED_P99_DRIFT
    ):
        raise RuntimeError("neutrality report relaxed the production drift limits")

    rows = report.get("images")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("neutrality report has no image evidence")
    seen_hashes = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("passed") is not True:
            raise RuntimeError(f"neutrality image {index} did not pass")
        image_path = Path(_require_nonempty_string(
            row.get("image"), f"neutrality image {index} path"
        ))
        image_hash = _require_hash(
            row.get("image_sha256"), f"neutrality image {index} hash"
        )
        _same(sha256(image_path), image_hash, f"neutrality image {index} bytes")
        seen_hashes.add(image_hash)
        source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"neutrality image {index} cannot be decoded")
        source_height, source_width = source.shape[:2]
        _same(row.get("source_width"), source_width,
              f"neutrality image {index} source width")
        _same(row.get("source_height"), source_height,
              f"neutrality image {index} source height")
        model_width, model_height = aspect_aligned_dims(
            source_width,
            source_height,
            depth_short_side=expected_preprocessing["depth_short_side"],
            depth_max_aspect=expected_preprocessing["depth_max_aspect"],
            max_width=min(MAX_WIDTH, source_width),
            max_height=min(MAX_HEIGHT, source_height),
        )
        shape = row.get("input_shape")
        if (
            not isinstance(shape, list)
            or shape != [1, 3, model_height, model_width]
        ):
            raise RuntimeError(f"neutrality image {index} shape is invalid")
        normalized = row.get("production_normalized")
        if not isinstance(normalized, dict):
            raise RuntimeError(f"neutrality image {index} lacks normalized drift")
        mean_value = normalized.get("mean_abs")
        p99_value = normalized.get("p99_abs")
        if (
            not isinstance(mean_value, (int, float))
            or not math.isfinite(mean_value)
            or mean_value > mean_limit
            or not isinstance(p99_value, (int, float))
            or not math.isfinite(p99_value)
            or p99_value > p99_limit
        ):
            raise RuntimeError(f"neutrality image {index} exceeds fixed drift limits")
    missing = {
        clip: digest for clip, digest in expected_image_hashes.items()
        if digest not in seen_hashes
    }
    if missing:
        raise RuntimeError(
            "neutrality report lacks canonical core first frames: "
            + ", ".join(sorted(missing))
        )
    return {
        "report_sha256": report_hash,
        "reference_model": model["base_depth_model"],
        "reference_onnx_sha256": reference_hash,
        "candidate_onnx_sha256": candidate_hash,
        "preprocessing_contract": NEUTRALITY_CONTRACT,
        "limits": {
            "production_normalized_mean_abs": float(mean_limit),
            "production_normalized_p99_abs": float(p99_limit),
        },
        "canonical_core_first_frames": dict(sorted(expected_image_hashes.items())),
        "evidence_image_count": len(rows),
    }


def _validate_gate_baseline(meta, baseline, origin):
    for field in BASELINE_RESULT_FIELDS:
        _same(meta.get(field), baseline.get(field), f"{origin} {field}")


def _render_geometry(meta, origin):
    source_width = meta.get("source_width")
    source_height = meta.get("source_height")
    if (
        not isinstance(source_width, int)
        or isinstance(source_width, bool)
        or source_width <= 0
        or not isinstance(source_height, int)
        or isinstance(source_height, bool)
        or source_height <= 0
    ):
        raise RuntimeError(f"{origin} has invalid source geometry")
    value = {
        "source_width": source_width,
        "source_height": source_height,
        "model_input_width": meta.get("model_input_width"),
        "model_input_height": meta.get("model_input_height"),
        "depth_short_side": meta.get("depth_short_side"),
        "depth_max_aspect": meta.get("depth_max_aspect"),
        "eye_width": meta.get("eye_width"),
        "eye_height": meta.get("eye_height"),
        "content_scale_x": meta.get("content_scale_x"),
        "content_scale_y": meta.get("content_scale_y"),
        "disparity_raster_width": meta.get("disparity_raster_width"),
        "disparity_raster_height": meta.get("disparity_raster_height"),
        "color_mode": meta.get("color_mode"),
    }
    try:
        validate_geometry_tuple(value)
    except RuntimeError as error:
        raise RuntimeError(f"{origin} has invalid deployment geometry: {error}") from error
    return canonical_geometry_tuple(value)


def validate_render_gate(
        results_path, suite, style, model, expected_clips=None):
    results_path = Path(results_path).resolve()
    payload, results_hash = _read_hashed_json(
        results_path, f"{suite} render gate"
    )
    if payload.get("verdict") != "pass":
        raise RuntimeError(f"{suite} render gate did not pass")
    if payload.get("regressions") != [] or payload.get("hard_failures") != []:
        raise RuntimeError(f"{suite} render gate contains failures")
    meta = payload.get("meta")
    clips = payload.get("clips")
    if not isinstance(meta, dict) or not isinstance(clips, dict):
        raise RuntimeError(f"{suite} render gate is incomplete")
    if "artifact_metric_sha256" in meta:
        raise RuntimeError(f"{suite} render gate was rescored, not freshly rendered")
    expected_clips = set(expected_clips or canonical_suite_clips(suite))
    if set(clips) != expected_clips:
        raise RuntimeError(f"{suite} render gate does not contain the full suite")
    clip_set = meta.get("clip_set_sha1")
    if (
        not isinstance(clip_set, dict)
        or set(clip_set) != expected_clips
        or any(re.fullmatch(r"[0-9a-f]{12}", value or "") is None
               for value in clip_set.values())
    ):
        raise RuntimeError(f"{suite} render gate clip identities are invalid")

    required = {
        "run_kind": "policy_candidate_gate",
        "suite": suite,
        "eval_schema": model["policy_baseline"]["eval_schema"],
        "model": model["deployed_model"],
        "metric_sha256": model["metric_sha256"],
        "policy_warp_source_sha256": model["policy_warp_source_sha256"],
        "model_onnx_sha256": model["onnx_sha256"],
        "policy_metadata_sha256": model["metadata_sha256"],
        "deployment_geometry_allowlist_sha256": model[
            "deployment_geometry_allowlist_sha256"
        ],
        "artistic_policy": True,
        "artistic_policy_consumed": True,
        "artistic_policy_authorization": "candidate-evaluation",
        "artistic_style": style,
        "artistic_scale_override": 0,
        "depth_step": "current-once",
        "depth_reuse_interval": 1,
        "depth_compensation": "none",
        "output_interval": 1,
        "output_gt_right_only": False,
        "literal_bestv2": False,
        "gpu_contention": False,
    }
    for key, expected in required.items():
        _same(meta.get(key), expected, f"{suite} render meta {key}")
    baseline_identities = meta.get("baseline_identities")
    if (
        not isinstance(baseline_identities, dict)
        or set(baseline_identities) != expected_clips
        or any(re.fullmatch(r"[0-9a-f]{64}", value or "") is None
               for value in baseline_identities.values())
    ):
        raise RuntimeError(f"{suite} render gate baseline identities are invalid")
    timestamp = _require_render_timestamp(
        meta.get("timestamp"), f"{suite} render timestamp"
    )
    _validate_gate_baseline(meta, model["policy_baseline"], f"{suite} render")

    clip_required = {
        key: value for key, value in required.items()
        if key not in {"run_kind", "suite", "eval_schema", "gpu_contention"}
    }
    clip_required["harness_schema"] = model["policy_baseline"]["harness_schema"]
    clip_required.update({
        "artifact_mode": "full",
        "warp_disparity": EXACT_DISPARITY_CONTRACT,
        "warp_unclamped_disparity": UNCLAMPED_DISPARITY_CONTRACT,
        "artistic_disparity_contract": ARTISTIC_DISPARITY_CONTRACT,
    })
    allowed_geometries = {
        tuple_key(value)
        for value in model["deployment_geometry_allowlist"]["tuples"]
    }
    observed_geometries = {}
    for clip, entry in clips.items():
        clip_meta = entry.get("meta") if isinstance(entry, dict) else None
        if not isinstance(clip_meta, dict):
            raise RuntimeError(f"{suite}/{clip} has no harness metadata")
        if "artifact_metric_sha256" in clip_meta:
            raise RuntimeError(f"{suite}/{clip} was rescored, not freshly rendered")
        for key, expected in clip_required.items():
            _same(clip_meta.get(key), expected, f"{suite}/{clip} {key}")
        _same(clip_meta.get("clip_sha1"), clip_set[clip],
              f"{suite}/{clip} source identity")
        _validate_gate_baseline(
            clip_meta, model["policy_baseline"], f"{suite}/{clip}"
        )
        geometry = _render_geometry(clip_meta, f"{suite}/{clip}")
        geometry_identity = tuple_key(geometry)
        if geometry_identity not in allowed_geometries:
            raise RuntimeError(
                f"{suite}/{clip} used a deployment geometry outside the allow-list"
            )
        observed_geometries[geometry_identity] = geometry
    return {
        "results_sha256": results_hash,
        "suite": suite,
        "artistic_style": style,
        "verdict": "pass",
        "eval_schema": required["eval_schema"],
        "harness_schema": clip_required["harness_schema"],
        "metric_sha256": model["metric_sha256"],
        "policy_warp_source_sha256": model["policy_warp_source_sha256"],
        "model_onnx_sha256": model["onnx_sha256"],
        "policy_metadata_sha256": model["metadata_sha256"],
        "deployment_geometry_allowlist_sha256": model[
            "deployment_geometry_allowlist_sha256"
        ],
        "artistic_policy_consumed": True,
        "artistic_policy_authorization": "candidate-evaluation",
        "timestamp": timestamp,
        "clip_set_sha1": dict(sorted(clip_set.items())),
        "baseline_identities": dict(sorted(baseline_identities.items())),
        "observed_deployment_geometries": [
            observed_geometries[key] for key in sorted(observed_geometries)
        ],
    }


def build_manifest(
        onnx, metadata, checkpoint, evaluation, reference_depth_onnx,
        neutrality_report, core_results, extended_results,
        balanced_core_results, balanced_extended_results,
        headset_review=None, stage="production",
        expected_core_clips=None, expected_extended_clips=None,
        expected_neutrality_images=None):
    if stage not in {"headset-review", "production"}:
        raise RuntimeError("deployment stage must be headset-review or production")
    if stage == "production":
        if (not isinstance(headset_review, dict) or
                headset_review.get("approved") is not True):
            raise RuntimeError("an explicit approved headset review is required")
        reviewer = _require_nonempty_string(
            headset_review.get("reviewer"), "headset reviewer"
        )
        device = _require_nonempty_string(
            headset_review.get("device"), "headset device"
        )
        resolution = _require_nonempty_string(
            headset_review.get("resolution"), "headset resolution"
        )
        color_mode = _require_nonempty_string(
            headset_review.get("color_mode"), "headset color mode"
        )
        notes = _require_nonempty_string(
            headset_review.get("notes"), "headset review notes"
        )
        refresh_hz = headset_review.get("refresh_hz")
        if (
            not isinstance(refresh_hz, (int, float))
            or isinstance(refresh_hz, bool)
            or not math.isfinite(refresh_hz)
            or refresh_hz <= 0.0
        ):
            raise RuntimeError("headset refresh rate must be positive")
    elif headset_review is not None:
        raise RuntimeError("staged headset-review manifests must not claim a review")

    model = validate_export_chain(onnx, metadata, checkpoint, evaluation)
    geometry_tuples = model["deployment_geometry_allowlist"]["tuples"]
    if stage == "production":
        geometry_index = headset_review.get("geometry_index")
        if (
            not isinstance(geometry_index, int)
            or isinstance(geometry_index, bool)
            or not 0 <= geometry_index < len(geometry_tuples)
        ):
            raise RuntimeError(
                "headset review must select an exact deployment geometry index"
            )
        reviewed_geometry = geometry_tuples[geometry_index]
        expected_resolution = (
            f"{2 * reviewed_geometry['eye_width']}x"
            f"{reviewed_geometry['eye_height']}"
        )
        _same(resolution, expected_resolution, "headset SBS resolution")
        _same(color_mode, reviewed_geometry["color_mode"], "headset color mode")
    expected_neutrality_images = (
        expected_neutrality_images or canonical_core_first_frame_hashes()
    )
    neutrality = validate_neutrality_report(
        neutrality_report, onnx, reference_depth_onnx, model,
        expected_neutrality_images,
    )
    core = validate_render_gate(
        core_results, "core", "immersive", model, expected_core_clips
    )
    extended = validate_render_gate(
        extended_results, "extended", "immersive", model, expected_extended_clips
    )
    balanced_core = validate_render_gate(
        balanced_core_results, "core", "balanced", model, expected_core_clips
    )
    balanced_extended = validate_render_gate(
        balanced_extended_results, "extended", "balanced", model,
        expected_extended_clips,
    )
    allowed_geometries = {
        tuple_key(value): value
        for value in model["deployment_geometry_allowlist"]["tuples"]
    }
    observed_geometries = {
        tuple_key(value): value
        for gate in (core, extended, balanced_core, balanced_extended)
        for value in gate["observed_deployment_geometries"]
    }
    missing_geometries = set(allowed_geometries) - set(observed_geometries)
    if missing_geometries:
        raise RuntimeError(
            "fresh core/extended render gates do not cover every deployment "
            f"geometry ({len(missing_geometries)} missing)"
        )
    if stage == "production":
        reviewed_geometry_identity = tuple_key(reviewed_geometry)
        if reviewed_geometry_identity not in observed_geometries:
            raise RuntimeError(
                "headset-reviewed deployment geometry was not covered by a fresh "
                "render gate"
            )
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    manifest = {
        "schema": 1,
        "contract": DEPLOYMENT_CONTRACT,
        "stage": stage,
        "approved": stage == "production",
        "created_at": created_at,
        "model": {
            key: model[key] for key in (
                "deployed_model",
                "base_depth_model",
                "onnx_sha256",
                "metadata_sha256",
                "checkpoint_sha256",
                "evaluation_sha256",
                "metric_sha256",
                "policy_warp_source_sha256",
                "active_split_sha256",
                "label_fitter_identity_sha256",
                "test_labels_sha256",
                "deployment_geometry_allowlist",
                "deployment_geometry_allowlist_sha256",
                "sealed_test_productions",
            )
        },
        "neutrality": neutrality,
        "render_gates": {
            "core": core,
            "extended": extended,
            "balanced_core": balanced_core,
            "balanced_extended": balanced_extended,
        },
        "deployment_geometry_coverage": [
            observed_geometries[key] for key in sorted(observed_geometries)
        ],
    }
    if stage == "production":
        manifest["headset_review"] = {
            "approved": True,
            "reviewer": reviewer,
            "device": device,
            "resolution": resolution,
            "refresh_hz": float(refresh_hz),
            "color_mode": color_mode,
            "deployment_geometry_index": geometry_index,
            "deployment_geometry": reviewed_geometry,
            "deployment_geometry_allowlist_sha256": model[
                "deployment_geometry_allowlist_sha256"
            ],
            "style": "immersive",
            "notes": notes,
            "reviewed_at": created_at,
        }
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--reference-depth-onnx", required=True, type=Path)
    parser.add_argument("--neutrality-report", required=True, type=Path)
    parser.add_argument("--core-results", required=True, type=Path)
    parser.add_argument("--extended-results", required=True, type=Path)
    parser.add_argument("--balanced-core-results", required=True, type=Path)
    parser.add_argument("--balanced-extended-results", required=True, type=Path)
    parser.add_argument(
        "--stage-headset-review", action="store_true",
        help="write a gate-approved, non-production manifest for explicit live review",
    )
    parser.add_argument("--approve-headset-review", action="store_true")
    parser.add_argument("--headset-reviewer")
    parser.add_argument("--headset-device")
    parser.add_argument("--headset-resolution")
    parser.add_argument("--headset-refresh-hz", type=float)
    parser.add_argument("--headset-color-mode")
    parser.add_argument("--headset-geometry-index", type=int)
    parser.add_argument("--headset-notes")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.stage_headset_review and args.approve_headset_review:
        parser.error("--stage-headset-review cannot also approve a completed review")
    stage = "headset-review" if args.stage_headset_review else "production"
    headset_review = None if stage == "headset-review" else {
        "approved": args.approve_headset_review,
        "reviewer": args.headset_reviewer,
        "device": args.headset_device,
        "resolution": args.headset_resolution,
        "refresh_hz": args.headset_refresh_hz,
        "color_mode": args.headset_color_mode,
        "geometry_index": args.headset_geometry_index,
        "notes": args.headset_notes,
    }
    manifest = build_manifest(
        args.onnx,
        args.metadata,
        args.checkpoint,
        args.evaluation,
        args.reference_depth_onnx,
        args.neutrality_report,
        args.core_results,
        args.extended_results,
        args.balanced_core_results,
        args.balanced_extended_results,
        headset_review,
        stage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
