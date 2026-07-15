#!/usr/bin/env python3
"""Build schema-8 multi-style labels from exact Apollo scale-grid renders.

The default ``immersive`` target is the highest-pop point on the *connected* safe frontier
above identity. Authored stereo is retained as a diagnostic, never confused with the product's
objective. This is a deterministic render-feasibility selector, not a neural teacher.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
SBSBENCH_DIR = THIS_DIR.parent / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))
import sbsbench  # noqa: E402

from artistic_policy_contract import ART_SCALE_DELTA_MAX  # noqa: E402

PROTECTED_PRIMARY_AXES = {"warp", "stability"}
EXACT_POP_METRIC = "exact_pop_spread_pct"
STYLE_NAMES = ("immersive", "balanced", "clean")
DEFAULT_STYLE = "immersive"
POLICY_CONTRACT = "safe-frontier-multistyle-apollo-v1"
POLICY_WARP_CONTRACT = "apollo-safe-frontier-v1"
MAX_CANDIDATE_SCALE_STEP = 0.10
WARP_DISPARITY_CONTRACT = (
    "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars"
)
WARP_UNCLAMPED_DISPARITY_CONTRACT = (
    "unclamped_full_binocular_normalized_at_artistic_scale_1_"
    "output_eye_raster_zero_bars"
)
ARTISTIC_DISPARITY_CONTRACT = (
    "clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_"
    "times_aspect_scale_times_content_scale_x)"
)
WARP_MASK_CONTRACT = {"red": "forward_disocclusion_before_fill"}
CLIP_POLICY_CONTRACT_FIELDS = (
    "harness_schema", "model", "profile", "metric_sha256",
    "policy_warp_source_sha256",
    "source_width", "source_height", "model_input_width", "model_input_height",
    "eye_width", "eye_height", "color_mode",
    "content_scale_x", "content_scale_y",
    "disparity_raster_width", "disparity_raster_height",
    "artistic_full_clamp_abs",
    "depth_step", "depth_reuse_interval", "depth_compensation",
    "depth_override_frames",
    "ema", "ema_edge_change", "ema_edge_gradient", "ema_edge_strength",
    "minmax_ema", "subject_lock", "subject_recenter", "subject_stretch",
    "depth_short_side", "depth_max_aspect", "pop_strength",
    "adaptive_pop", "adaptive_pop_max", "zero_plane", "artistic_style",
    "artistic_policy", "artistic_policy_consumed", "artistic_policy_authorization",
    "model_onnx_sha256", "policy_metadata_sha256",
    "deployment_geometry_allowlist_sha256", "output_interval", "output_gt_right_only",
    "literal_bestv2", "cuda_graph", "artifact_mode", "warp_mask",
    "warp_disparity", "warp_unclamped_disparity",
    "artistic_disparity_contract",
)
CLIP_TOP_META_FIELDS = (
    "model", "profile", "metric_sha256", "policy_warp_source_sha256",
    "depth_step", "depth_reuse_interval", "depth_compensation",
    "ema", "ema_edge_change", "ema_edge_gradient", "ema_edge_strength",
    "minmax_ema", "subject_lock", "subject_recenter", "subject_stretch",
    "depth_short_side", "depth_max_aspect", "pop_strength",
    "adaptive_pop", "adaptive_pop_max", "zero_plane", "artistic_style",
    "artistic_policy", "artistic_policy_consumed", "artistic_policy_authorization",
    "model_onnx_sha256", "policy_metadata_sha256",
    "deployment_geometry_allowlist_sha256", "artistic_scale_override", "output_interval",
    "output_gt_right_only", "literal_bestv2", "cuda_graph",
)
POLICY_BASELINE_FIELDS = (
    "profile",
    "model",
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
    "metric_sha256",
    "policy_warp_source_sha256",
)


def semantic_file_hash(paths):
    """Match run_eval.py's path-name plus normalized-source metric identity."""
    digest = hashlib.sha256()
    for path in map(Path, paths):
        digest.update(path.name.encode())
        data = path.read_bytes()
        if path.suffix.lower() in {".py", ".json", ".conf", ".md", ".hlsl"}:
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(data)
    return digest.hexdigest()[:16]


def metric_contract_hash(thresholds_path):
    return semantic_file_hash((
        SBSBENCH_DIR / "sbsbench.py",
        thresholds_path,
        SBSBENCH_DIR / "run_eval.py",
    ))


def validate_metric_contract(meta, thresholds_path):
    expected = metric_contract_hash(Path(thresholds_path).resolve())
    actual = meta.get("metric_sha256")
    if actual != expected:
        raise RuntimeError(
            "threshold/metric implementation differs from the render grid: "
            f"{expected} != {actual}"
        )
    return expected


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_candidate(value):
    try:
        scale_text, path_text = value.split("=", 1)
        scale = float(scale_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate must be SCALE=results.json") from exc
    if not 1.0 - ART_SCALE_DELTA_MAX <= scale <= 1.0 + ART_SCALE_DELTA_MAX:
        raise argparse.ArgumentTypeError("candidate scale is outside the model contract")
    path = Path(path_text).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"candidate results do not exist: {path}")
    return scale, path


def grid_context_args(extra_args):
    """Remove only the intentional scale treatment from the resolved harness arguments."""
    normalized = []
    index = 0
    extra_args = list(extra_args or [])
    while index < len(extra_args):
        token = extra_args[index]
        if token == "--artistic-scale-override":
            if index + 1 >= len(extra_args):
                raise RuntimeError("artistic scale override has no value")
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def validate_candidate_scale_grid(scales):
    """Require an identity-anchored dense grid before treating samples as an interval."""
    scales = sorted(float(scale) for scale in scales)
    lower_bound = 1.0 - ART_SCALE_DELTA_MAX
    upper_bound = 1.0 + ART_SCALE_DELTA_MAX
    if (not scales or any(not np.isfinite(scale) for scale in scales) or
            min(scales) < lower_bound - 1e-8 or
            max(scales) > upper_bound + 1e-8):
        raise RuntimeError("candidate grid contains a scale outside the model contract")
    if 1.0 not in scales:
        raise RuntimeError("candidate grid must include exact identity scale 1.0")
    if not np.isclose(max(scales), upper_bound, rtol=0.0, atol=1e-8):
        raise RuntimeError(
            "candidate grid must cover the full upward model contract through "
            f"scale {upper_bound:.6g}"
        )
    gaps = [right - left for left, right in zip(scales, scales[1:])]
    sparse = [gap for gap in gaps if gap > MAX_CANDIDATE_SCALE_STEP + 1e-8]
    if sparse:
        raise RuntimeError(
            "candidate grid is too sparse to prove a connected safe interval: "
            f"max step {max(sparse):.6g} exceeds {MAX_CANDIDATE_SCALE_STEP:.6g}"
        )
    return scales


def validate_uncompensated_run(payload, origin):
    """Reject temporal/external depth treatments from spatial policy supervision."""
    meta = payload.get("meta", {})
    if meta.get("depth_compensation") != "none":
        raise RuntimeError(
            f"{origin}: policy labels require depth_compensation=none"
        )
    for clip, entry in payload.get("clips", {}).items():
        clip_meta = entry.get("meta", {})
        if clip_meta.get("depth_compensation") != "none":
            raise RuntimeError(
                f"{origin}/{clip}: policy labels require depth_compensation=none"
            )
        override_frames = clip_meta.get("depth_override_frames")
        if (not isinstance(override_frames, int) or
                isinstance(override_frames, bool) or override_frames != 0):
            raise RuntimeError(
                f"{origin}/{clip}: policy labels require depth_override_frames=0"
            )


def validate_clip_contract_context(control, candidate, scale, clip, origin):
    """Compare the exact per-clip harness contracts behind two aggregates."""
    control_meta = control["clips"][clip].get("meta", {})
    candidate_meta = candidate["clips"][clip].get("meta", {})
    for label, meta in (("control", control_meta), (origin, candidate_meta)):
        missing = [field for field in CLIP_POLICY_CONTRACT_FIELDS
                   if field not in meta]
        if missing:
            raise RuntimeError(
                f"{label}/{clip}: clip harness contract lacks: " +
                ", ".join(missing)
            )
        validate_source_raster_contract(meta, meta)
        expected = {
            "harness_schema": 24,
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
            "output_gt_right_only": True,
            "literal_bestv2": False,
            "artifact_mode": "full",
            "warp_mask": WARP_MASK_CONTRACT,
            "warp_disparity": WARP_DISPARITY_CONTRACT,
            "warp_unclamped_disparity": WARP_UNCLAMPED_DISPARITY_CONTRACT,
            "artistic_disparity_contract": ARTISTIC_DISPARITY_CONTRACT,
        }
        stale = {field: (value, meta.get(field))
                 for field, value in expected.items()
                 if meta.get(field) != value}
        if stale:
            raise RuntimeError(
                f"{label}/{clip}: invalid clip harness contract: {stale}"
            )
        run_meta = control.get("meta", {}) if label == "control" else \
            candidate.get("meta", {})
        top_mismatch = {
            field: (run_meta.get(field), meta.get(field))
            for field in CLIP_TOP_META_FIELDS
            if field not in run_meta or run_meta.get(field) != meta.get(field)
        }
        if top_mismatch:
            raise RuntimeError(
                f"{label}/{clip}: clip harness differs from run metadata: "
                f"{top_mismatch}"
            )
    mismatches = {
        field: (control_meta[field], candidate_meta[field])
        for field in CLIP_POLICY_CONTRACT_FIELDS
        if control_meta[field] != candidate_meta[field]
    }
    if mismatches:
        raise RuntimeError(
            f"{origin}/{clip}: clip policy/raster contract differs: {mismatches}"
        )
    control_scale = control_meta.get("artistic_scale_override")
    candidate_scale = candidate_meta.get("artistic_scale_override")
    if control_scale != 1.0 or candidate_scale != float(scale):
        raise RuntimeError(
            f"{origin}/{clip}: clip artistic scale provenance differs: "
            f"control={control_scale}, candidate={candidate_scale}, expected={scale}"
        )
    if float(scale) == 1.0:
        if ("aggregate" not in control["clips"][clip] or
                "aggregate" not in candidate["clips"][clip] or
                control["clips"][clip]["aggregate"] !=
                candidate["clips"][clip]["aggregate"]):
            raise RuntimeError(
                f"{origin}/{clip}: scale-1 candidate aggregate differs from control"
            )


def validate_context(control, candidate, scale, origin):
    validate_uncompensated_run(control, "control")
    validate_uncompensated_run(candidate, origin)
    control_meta = control.get("meta", {})
    candidate_meta = candidate.get("meta", {})
    if control_meta.get("artistic_scale_override") != 1.0:
        raise RuntimeError(
            "control: policy identity requires artistic_scale_override=1.0"
        )
    same = ("clip_set_sha1", "eval_schema", "metric_sha256", "model", "profile",
            "conf_sha256", "depth_step", "output_interval", "output_gt_right_only")
    mismatch = {key: (control_meta.get(key), candidate_meta.get(key))
                for key in same if control_meta.get(key) != candidate_meta.get(key)}
    if mismatch:
        raise RuntimeError(f"{origin}: incompatible render context: {mismatch}")
    semantic_mismatch = {
        key: (control_meta.get(key), candidate_meta.get(key))
        for key in POLICY_BASELINE_FIELDS
        if control_meta.get(key) != candidate_meta.get(key)
    }
    control_args = grid_context_args(control_meta.get("extra_args"))
    candidate_args = grid_context_args(candidate_meta.get("extra_args"))
    if control_args != candidate_args:
        semantic_mismatch["extra_args_without_scale"] = (
            control_args, candidate_args
        )
    if semantic_mismatch:
        raise RuntimeError(
            f"{origin}: policy/geometry context differs: {semantic_mismatch}"
        )
    if not candidate_meta.get("output_gt_right_only"):
        raise RuntimeError(f"{origin}: render did not select exact authored frame identities")
    if control_meta.get("artistic_policy") is not False or \
            candidate_meta.get("artistic_policy") is not False:
        raise RuntimeError(
            f"{origin}: safe-frontier renders must disable the learned artistic policy"
        )
    if (control_meta.get("artistic_policy_consumed") is not False or
            candidate_meta.get("artistic_policy_consumed") is not False or
            control_meta.get("model_onnx_sha256") or
            candidate_meta.get("model_onnx_sha256") or
            control_meta.get("policy_metadata_sha256") or
            candidate_meta.get("policy_metadata_sha256")):
        raise RuntimeError(
            f"{origin}: safe-frontier renders unexpectedly consumed a learned policy"
        )
    if control_meta.get("literal_bestv2") or candidate_meta.get("literal_bestv2"):
        raise RuntimeError(
            f"{origin}: safe-frontier renders must use the production Apollo warp"
        )
    if (control_meta.get("depth_step") != "current-once" or
            candidate_meta.get("depth_step") != "current-once"):
        raise RuntimeError(
            f"{origin}: safe-frontier renders require current-frame depth"
        )
    actual = float(candidate_meta.get("artistic_scale_override", 0.0))
    if abs(actual - scale) > 1e-6:
        raise RuntimeError(f"{origin}: scale provenance mismatch: {actual} != {scale}")
    if set(control.get("clips", {})) != set(candidate.get("clips", {})):
        raise RuntimeError(f"{origin}: candidate clips differ from control")
    for clip in sorted(control["clips"]):
        validate_clip_contract_context(control, candidate, scale, clip, origin)


def policy_baseline_from_meta(meta):
    """Freeze every production input that changes the learned safe frontier.

    The policy does not observe these settings, so applying it under a different warp/depth
    configuration would make its artifact and stability guarantees meaningless. Keep this
    semantic dictionary in labels, checkpoints, and exported model metadata; runtime compares
    it field-for-field instead of relying on a path-dependent sunshine.conf hash.
    """
    missing = [field for field in POLICY_BASELINE_FIELDS if field not in meta]
    if missing:
        raise RuntimeError(
            "control results lack policy-affecting configuration: " +
            ", ".join(missing)
        )
    baseline = {field: meta[field] for field in POLICY_BASELINE_FIELDS}
    if baseline["depth_compensation"] != "none":
        raise RuntimeError("control results must use uncompensated current-frame depth")
    if not re.fullmatch(r"[0-9a-f]{64}", str(baseline["policy_warp_source_sha256"])):
        raise RuntimeError("control results have an invalid policy warp source hash")
    baseline["depth_model"] = baseline.pop("model")
    baseline.update({
        "harness_schema": 24,
        "eval_schema": 29,
        "warp_contract": POLICY_WARP_CONTRACT,
    })
    if int(meta.get("eval_schema", -1)) != baseline["eval_schema"]:
        raise RuntimeError(
            f"unsupported evaluator schema for policy labels: {meta.get('eval_schema')}"
        )
    return baseline


def metric_applicable(metric, spec, control, clip_meta):
    """Return whether this clip is contractually able to provide the metric.

    Missing required evidence fails closed. A GT-derived metric is not required for a clip
    that explicitly has no such ground truth; treating not-applicable as a regression would
    make authored movie shots impossible to label.
    """
    if spec.get("role") == "hard":
        return True
    if spec.get("role") != "primary":
        return True
    if metric.startswith("depth_gt_") and clip_meta.get("required_gt_depth"):
        return True
    if metric.startswith("flow_") and clip_meta.get("required_gt_flow"):
        return True
    # Primary metrics that could not be measured in the identity control (for example no
    # static support or no silhouette band) are inapplicable, not evidence of candidate harm.
    # Once control establishes evidence, however, every candidate must retain it.
    value = control.get(metric)
    return value is not None and np.isfinite(value)


def project_protected_worst_metrics(entry, metric_specs, origin):
    """Replace protected clip averages with run_eval's directional worst frame."""
    aggregate = entry.get("aggregate")
    if not isinstance(aggregate, dict):
        raise RuntimeError(f"{origin}: clip entry has no aggregate metrics")
    projected = dict(aggregate)
    worst_frames = entry.get("worst_frame", {})
    for metric, spec in metric_specs.items():
        if (spec.get("role") != "primary" or
                spec.get("axis") not in PROTECTED_PRIMARY_AXES or
                metric not in aggregate or aggregate[metric] is None):
            continue
        evidence = worst_frames.get(metric) if isinstance(worst_frames, dict) else None
        value = evidence.get("worst_value") if isinstance(evidence, dict) else None
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not np.isfinite(value)):
            raise RuntimeError(
                f"{origin}: protected metric {metric} lacks finite worst-frame evidence"
            )
        projected[metric] = float(value)
    return projected


def feasibility_violations(control, candidate, metric_specs, clip_meta=None):
    """Return fail-closed hard, warp, and stability constraint violations."""
    clip_meta = clip_meta or {}
    violations = []
    for metric, spec in metric_specs.items():
        role = spec.get("role")
        protected = role == "hard" or (
            role == "primary" and spec.get("axis") in PROTECTED_PRIMARY_AXES
        )
        if not protected:
            continue
        if not metric_applicable(metric, spec, control, clip_meta):
            continue
        value = candidate.get(metric)
        if value is None or not np.isfinite(value):
            violations.append(metric + ":missing")
            continue
        if role == "hard":
            if sbsbench.metric_gate_failed(value, value, spec):
                violations.append(metric + ":hard")
            continue
        baseline = control.get(metric)
        if baseline is None or not np.isfinite(baseline):
            violations.append(metric + ":missing-control")
        elif sbsbench.metric_gate_failed(baseline, value, spec):
            violations.append(metric + ":regression")
    return violations


def constraint_margins(control, candidate, metric_specs, clip_meta=None):
    """Return normalized distance from every protected failure boundary.

    A margin of zero is exactly on the permitted boundary; one has one full tolerance (or
    hard-bound magnitude) of headroom. Missing evidence is represented by a negative margin.
    """
    clip_meta = clip_meta or {}
    margins = {}
    for metric, spec in metric_specs.items():
        role = spec.get("role")
        protected = role == "hard" or (
            role == "primary" and spec.get("axis") in PROTECTED_PRIMARY_AXES
        )
        if not protected:
            continue
        if not metric_applicable(metric, spec, control, clip_meta):
            continue
        value = candidate.get(metric)
        if value is None or not np.isfinite(value):
            margins[metric] = -1.0
            continue
        if role == "hard":
            bounds = []
            if "hard_min" in spec:
                denominator = max(abs(float(spec["hard_min"])),
                                  float(spec.get("abs_floor", 0.0)), 1e-6)
                bounds.append((float(value) - float(spec["hard_min"])) / denominator)
            if "hard_max" in spec:
                denominator = max(abs(float(spec["hard_max"])),
                                  float(spec.get("abs_floor", 0.0)), 1e-6)
                bounds.append((float(spec["hard_max"]) - float(value)) / denominator)
            margins[metric] = min(bounds) if bounds else 1.0
            continue
        baseline = control.get(metric)
        if baseline is None or not np.isfinite(baseline):
            margins[metric] = -1.0
            continue
        tolerance = max(float(spec.get("abs_floor", 0.0)),
                        abs(float(baseline)) * float(spec.get("rel_tol", 0.0)), 1e-6)
        if spec.get("better") == "higher":
            boundary = float(baseline) - tolerance
            margins[metric] = (float(value) - boundary) / tolerance
        else:
            boundary = float(baseline) + tolerance
            margins[metric] = (boundary - float(value)) / tolerance
    return margins


def artifact_burden(control, candidate, metric_specs, clip_meta=None):
    """Signed warp/stability degradation in tolerance units; lower is cleaner."""
    clip_meta = clip_meta or {}
    terms = []
    for metric, spec in metric_specs.items():
        if (spec.get("role") != "primary" or
                spec.get("axis") not in PROTECTED_PRIMARY_AXES):
            continue
        if not metric_applicable(metric, spec, control, clip_meta):
            continue
        baseline, value = control.get(metric), candidate.get(metric)
        if baseline is None or value is None:
            continue
        tolerance = max(float(spec.get("abs_floor", 0.0)),
                        abs(float(baseline)) * float(spec.get("rel_tol", 0.0)), 1e-6)
        degradation = (float(baseline) - float(value) if spec.get("better") == "higher"
                       else float(value) - float(baseline))
        terms.append(degradation / tolerance)
    return float(np.mean(terms)) if terms else 0.0


def safety_margin_reliability(scale, evidence, connected_scales):
    """Graded evidence reliability derived only from safety margin and grid support."""
    if abs(float(scale) - 1.0) < 1e-6:
        return 0.0
    margins = evidence[scale]["constraint_margins"]
    minimum_margin = min(margins.values()) if margins else 1.0
    margin_strength = float(np.clip(minimum_margin, 0.0, 1.0))
    direction_count = sum(
        1 for candidate_scale in connected_scales
        if (candidate_scale - 1.0) * (scale - 1.0) > 0.0
    )
    support_strength = min(1.0, direction_count / 2.0)
    return 0.5 + 0.5 * margin_strength * support_strength


def select_clip(control_agg, candidates, metric_specs, clip_meta=None):
    """Select explicit style targets on the identity-connected safe component."""
    clip_meta = clip_meta or {}
    validate_candidate_scale_grid(candidates)
    identity = candidates.get(1.0)
    if identity is None:
        raise RuntimeError("scale grid has no identity candidate")
    control_violations = feasibility_violations(
        control_agg, identity, metric_specs, clip_meta
    )
    if control_violations:
        raise RuntimeError("identity render violates its own feasibility contract: " +
                           ", ".join(control_violations))
    identity_pop = identity.get(EXACT_POP_METRIC)
    if identity_pop is None or not np.isfinite(identity_pop):
        raise RuntimeError(f"identity render has no finite {EXACT_POP_METRIC}")

    evidence = {}
    for scale, aggregate in sorted(candidates.items()):
        violations = feasibility_violations(
            control_agg, aggregate, metric_specs, clip_meta
        )
        exact_pop = aggregate.get(EXACT_POP_METRIC)
        if exact_pop is None or not np.isfinite(exact_pop):
            violations.append(EXACT_POP_METRIC + ":missing")
        evidence[scale] = {
            "metrics": aggregate,
            "violations": violations,
            "constraint_margins": constraint_margins(
                control_agg, aggregate, metric_specs, clip_meta
            ),
            "artifact_burden": artifact_burden(
                control_agg, aggregate, metric_specs, clip_meta
            ),
            "individually_feasible": not violations,
            "connected": False,
        }

    scales = sorted(candidates)
    identity_index = scales.index(1.0)
    connected = {1.0}
    evidence[1.0]["connected"] = True
    # A safe endpoint is meaningful only when every sampled intermediate point is safe.
    # Stop permanently at the first failure in each direction; do not jump over a hole.
    for indices in (range(identity_index + 1, len(scales)),
                    range(identity_index - 1, -1, -1)):
        for index in indices:
            scale = scales[index]
            if evidence[scale]["violations"]:
                break
            connected.add(scale)
            evidence[scale]["connected"] = True

    connected_scales = sorted(connected)
    upward = [scale for scale in connected_scales if scale >= 1.0]
    safe_scale_ceiling = float(max(upward))
    # Style is a runtime choice, not something the model should infer. The model learns only
    # the scene's safe ceiling; deterministic presets are derived from it. Keep the offline
    # balanced label on an actually rendered connected point instead of inventing an endpoint.
    balanced_request = 1.0 + 0.5 * (safe_scale_ceiling - 1.0)
    balanced = min(upward, key=lambda scale: (abs(scale - balanced_request), scale))
    style_targets = {
        "clean": 1.0,
        "balanced": float(balanced),
        "immersive": safe_scale_ceiling,
    }

    authored_pool = [
        scale for scale in connected_scales
        if candidates[scale].get("stereo_gt_psnr") is not None and
        np.isfinite(candidates[scale]["stereo_gt_psnr"])
    ]
    authored = (max(authored_pool,
                    key=lambda scale: (float(candidates[scale]["stereo_gt_psnr"]),
                                       -abs(scale - 1.0)))
                if authored_pool else 1.0)
    margin_reliability = safety_margin_reliability(
        safe_scale_ceiling, evidence, connected_scales
    )
    ceiling_confidence = 0.0 if abs(safe_scale_ceiling - 1.0) < 1e-6 else 1.0
    return {
        "safe_scale_ceiling": safe_scale_ceiling,
        "ceiling_confidence": ceiling_confidence,
        "safety_margin_reliability": margin_reliability,
        "style_targets": style_targets,
        "authored_fit_scale": float(authored),
        "authored_fit_psnr": (
            float(candidates[authored]["stereo_gt_psnr"])
            if authored_pool else None
        ),
        "connected_safe_scales": connected_scales,
        "safe_scale_min": float(min(connected_scales)),
        "safe_scale_max": float(max(connected_scales)),
        "identity_exact_pop_spread_pct": float(identity_pop),
        "candidate_grid": evidence,
    }


def load_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if row.get("clip") is None:
                    raise RuntimeError(f"{path}:{line_number}: row has no clip")
                rows.append(row)
    if not rows:
        raise RuntimeError(f"source label bundle is empty: {path}")
    return rows


def source_label_contract(source_labels, control_meta):
    """Verify and retain the schema-7 stereo/source-depth provenance."""
    source_labels = Path(source_labels).resolve()
    fitter_path = source_labels.parent / "label_fitter_contract.json"
    summary_path = source_labels.parent / "summary.json"
    if not fitter_path.is_file() or not summary_path.is_file():
        raise RuntimeError(
            f"source label bundle is incomplete beside {source_labels}"
        )
    fitter = load_json(fitter_path)
    summary = load_json(summary_path)
    if int(fitter.get("schema", 0)) != 7 or int(summary.get("schema", 0)) != 7:
        raise RuntimeError("source labels do not use the required schema-7 fitter")
    if summary.get("labels_sha256") != sha256(source_labels):
        raise RuntimeError("source label summary does not match labels.jsonl")
    if summary.get("label_fitter_contract_sha256") != sha256(fitter_path):
        raise RuntimeError("source label fitter contract hash is stale")
    run = fitter.get("run_contract", {})
    if run.get("model") != control_meta.get("model"):
        raise RuntimeError("source label depth model differs from the render grid")
    if run.get("conf_sha256") != control_meta.get("conf_sha256"):
        raise RuntimeError("source label configuration differs from the render grid")
    return {
        "labels": {"path": str(source_labels), "sha256": sha256(source_labels)},
        "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "fitter_contract": {
            "path": str(fitter_path), "sha256": sha256(fitter_path)
        },
        "run_contract": run,
    }


def validate_source_raster_contract(row, contract):
    """Validate the source row against authoritative output-eye raster geometry."""
    integer_fields = (
        "source_width", "source_height", "model_input_width", "model_input_height",
        "eye_width", "eye_height",
        "disparity_raster_width", "disparity_raster_height",
    )
    for field in integer_fields:
        expected = contract.get(field)
        actual = row.get(field)
        if (not isinstance(expected, int) or isinstance(expected, bool) or
                expected <= 0):
            raise RuntimeError(f"source harness has invalid {field}")
        if (not isinstance(actual, int) or isinstance(actual, bool) or
                actual != expected):
            raise RuntimeError(
                f"source row {field} differs from its harness contract"
            )
    for field in ("content_scale_x", "content_scale_y"):
        try:
            expected = float(contract[field])
            actual = float(row[field])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"source row/harness has invalid {field}") from error
        if (not np.isfinite(expected) or not 0.0 < expected <= 1.0 or
                not np.isfinite(actual) or
                not np.isclose(actual, expected, rtol=0.0, atol=1e-8)):
            raise RuntimeError(
                f"source row {field} differs from its harness contract"
            )
    if (contract.get("color_mode") != "sdr-srgb-8bit" or
            row.get("color_mode") != contract.get("color_mode")):
        raise RuntimeError("source row/harness has incompatible color mode")
    if (contract["disparity_raster_width"] != contract["eye_width"] or
            contract["disparity_raster_height"] != contract["eye_height"]):
        raise RuntimeError(
            "source harness disparity raster is not the full output-eye raster"
        )
    try:
        row_clamp = float(row["artistic_full_clamp_abs"])
        contract_clamp = float(contract["artistic_full_clamp_abs"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("source row/harness has invalid artistic clamp") from error
    if (not np.isfinite(contract_clamp) or contract_clamp <= 0.0 or
            not np.isclose(row_clamp, contract_clamp, rtol=0.0, atol=1e-8)):
        raise RuntimeError(
            "source row artistic_full_clamp_abs differs from its harness contract"
        )
    return contract["disparity_raster_height"], contract["disparity_raster_width"]


def validate_loaded_disparity_rasters(row, contract, clamped, unclamped):
    """Check shapes decoded from each .f32 header against the harness eye raster."""
    expected_shape = validate_source_raster_contract(row, contract)
    if clamped.ndim != 2 or tuple(clamped.shape) != expected_shape:
        raise RuntimeError(
            f"clamped disparity texture header differs from contract: "
            f"{clamped.shape} != {expected_shape}"
        )
    if unclamped.ndim != 2 or tuple(unclamped.shape) != expected_shape:
        raise RuntimeError(
            f"unclamped disparity texture header differs from contract: "
            f"{unclamped.shape} != {expected_shape}"
        )
    if not np.isfinite(clamped).all() or not np.isfinite(unclamped).all():
        raise RuntimeError("disparity raster contains non-finite values")
    return expected_shape


def content_raster_values(raster, contract):
    """Select centered content pixels exactly as ContentToSourceUV does in HLSL."""
    height, width = raster.shape
    scale_x = np.float32(contract["content_scale_x"])
    scale_y = np.float32(contract["content_scale_y"])
    one, half = np.float32(1.0), np.float32(0.5)
    x = np.float32(
        (np.arange(width, dtype=np.float32) + half) / np.float32(width)
    )
    y = np.float32(
        (np.arange(height, dtype=np.float32) + half) / np.float32(height)
    )
    lo_x = np.float32(half * np.float32(one - scale_x))
    lo_y = np.float32(half * np.float32(one - scale_y))
    hi_x = np.float32(lo_x + scale_x)
    hi_y = np.float32(lo_y + scale_y)
    valid_x = (x >= lo_x) & (x <= hi_x)
    valid_y = (y >= lo_y) & (y <= hi_y)
    content = raster[valid_y][:, valid_x]
    if content.size == 0:
        raise RuntimeError("output-eye disparity raster has no content-valid pixels")
    return content


def validate_source_harness(row, policy_baseline):
    """Prove that the exact raw field uses the baseline later checked at runtime."""
    baseline_path = Path(row["baseline_disparity"]).resolve()
    contract_path = baseline_path.parent / "contract.json"
    if not contract_path.is_file():
        raise RuntimeError(f"source baseline has no harness contract: {contract_path}")
    if row.get("harness_contract_sha256") != sha256(contract_path):
        raise RuntimeError(f"source harness contract hash mismatch: {contract_path}")
    contract = load_json(contract_path)
    if int(row.get("label_schema", 0)) != 7 or \
            row.get("policy_contract") != "stereo-fit-source-v2":
        raise RuntimeError("source row does not use the required schema-7 contract")
    if (int(contract.get("schema", 0)) != policy_baseline["harness_schema"] or
            contract.get("artistic_policy") is not False or
            contract.get("artistic_policy_consumed") is not False or
            contract.get("artistic_policy_authorization") != "none" or
            contract.get("model_onnx_sha256") or
            contract.get("policy_metadata_sha256") or
            contract.get("deployment_geometry_allowlist_sha256") or
            float(contract.get("artistic_scale_override", 0.0)) != 0.0 or
            contract.get("depth_compensation") != "none" or
            contract.get("depth_override_frames") != 0):
        raise RuntimeError(f"source harness is not an unbiased baseline: {contract_path}")
    if (contract.get("warp_disparity") !=
            "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars" or
            contract.get("warp_unclamped_disparity") !=
            "unclamped_full_binocular_normalized_at_artistic_scale_1_output_eye_raster_zero_bars" or
            contract.get("artistic_disparity_contract") !=
            ARTISTIC_DISPARITY_CONTRACT):
        raise RuntimeError(f"source harness disparity contract is stale: {contract_path}")
    mapping = {
        "profile": "profile",
        "depth_model": "model",
        "pop_strength": "pop_strength",
        "adaptive_pop": "adaptive_pop",
        "adaptive_pop_max": "adaptive_pop_max",
        "ema": "ema",
        "ema_edge_change": "ema_edge_change",
        "ema_edge_gradient": "ema_edge_gradient",
        "ema_edge_strength": "ema_edge_strength",
        "minmax_ema": "minmax_ema",
        "subject_lock": "subject_lock",
        "subject_recenter": "subject_recenter",
        "subject_stretch": "subject_stretch",
        "depth_short_side": "depth_short_side",
        "depth_max_aspect": "depth_max_aspect",
        "zero_plane": "zero_plane",
        "depth_step": "depth_step",
        "literal_bestv2": "literal_bestv2",
        "policy_warp_source_sha256": "policy_warp_source_sha256",
        "metric_sha256": "metric_sha256",
    }
    mismatches = {}
    for baseline_key, contract_key in mapping.items():
        expected = policy_baseline[baseline_key]
        actual = contract.get(contract_key)
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                equal = np.isclose(float(expected), float(actual), rtol=0.0, atol=1e-8)
            except (TypeError, ValueError):
                equal = False
        else:
            equal = expected == actual
        if not equal:
            mismatches[baseline_key] = (expected, actual)
    if mismatches:
        raise RuntimeError(
            f"source harness differs from policy baseline: {mismatches}"
        )
    validate_source_raster_contract(row, contract)
    return contract


def raw_disparity_path(source):
    """Resolve the schema-8 unclamped disparity artifact, failing closed."""
    explicit = source.get("baseline_unclamped_disparity")
    if explicit:
        path = Path(explicit).resolve()
    else:
        baseline = Path(source["baseline_disparity"]).resolve()
        prefix = "baseline_disparity_"
        if not baseline.name.startswith(prefix):
            raise RuntimeError(
                f"cannot derive unclamped disparity path from {baseline}"
            )
        path = baseline.with_name(
            "baseline_unclamped_disparity_" + baseline.name[len(prefix):]
        )
    if not path.is_file():
        raise RuntimeError(f"missing exact unclamped baseline disparity: {path}")
    expected = source.get("baseline_unclamped_disparity_sha256")
    actual = sha256(path)
    if expected is not None and expected != actual:
        raise RuntimeError(f"unclamped disparity hash mismatch: {path}")
    return path, actual


def clamp_aware_summary(raw, scale, clamp_abs, perceived_scale=1.0):
    """Apply the shipping policy clamp to exact scale-1 full-binocular disparity."""
    clamp_abs = float(clamp_abs)
    perceived_scale = float(perceived_scale)
    if not np.isfinite(clamp_abs) or clamp_abs <= 0.0:
        raise RuntimeError("invalid artistic full-disparity clamp")
    if not np.isfinite(perceived_scale) or perceived_scale <= 0.0:
        raise RuntimeError("invalid perceived disparity scale")
    scaled = raw * float(scale)
    final = np.clip(scaled, -clamp_abs, clamp_abs)
    return {
        "scale": float(scale),
        "hlsl_full_clamp_abs": clamp_abs,
        "comfort_clamp_abs_pct": clamp_abs * perceived_scale * 100.0,
        "mean_abs_disparity_pct": float(
            np.mean(np.abs(final)) * perceived_scale * 100.0
        ),
        "p95_abs_disparity_pct": float(
            np.percentile(np.abs(final), 95) * perceived_scale * 100.0
        ),
        "exact_pop_spread_pct": float(
            (np.percentile(final, 95) - np.percentile(final, 5)) *
            perceived_scale * 100.0
        ),
        "clamped_pixel_pct": float(np.mean(np.abs(scaled) > clamp_abs) * 100.0),
    }


def write_bundle(source_labels, control_path, candidate_specs, output,
                 thresholds_path, overwrite=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(f"output must be empty (or use --overwrite): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    rows = load_rows(source_labels)
    control = load_json(control_path)
    thresholds_path = Path(thresholds_path).resolve()
    validate_metric_contract(control.get("meta", {}), thresholds_path)
    policy_baseline = policy_baseline_from_meta(control.get("meta", {}))
    source_contract = source_label_contract(
        source_labels, control.get("meta", {})
    )
    candidate_runs = {}
    candidate_sources = []
    for scale, path in candidate_specs:
        if scale in candidate_runs:
            raise RuntimeError(f"duplicate candidate scale: {scale}")
        payload = load_json(path)
        validate_context(control, payload, scale, path)
        candidate_runs[scale] = payload
        candidate_sources.append({"scale": scale, "path": str(path), "sha256": sha256(path)})
    candidate_scales = validate_candidate_scale_grid(candidate_runs)

    metric_specs = load_json(thresholds_path)["metrics"]
    source_clips = {row["clip"] for row in rows}
    missing = source_clips - set(control.get("clips", {}))
    if missing:
        raise RuntimeError(f"render grid is missing source-label clips: {sorted(missing)}")

    decisions = {}
    for clip in sorted(source_clips):
        control_agg = project_protected_worst_metrics(
            control["clips"][clip], metric_specs, f"control/{clip}"
        )
        candidate_aggs = {
            scale: project_protected_worst_metrics(
                payload["clips"][clip], metric_specs,
                f"candidate-{scale:g}/{clip}",
            )
            for scale, payload in candidate_runs.items()
        }
        identity_agg = candidate_aggs.get(1.0)
        worst_identity_mismatch = {
            metric: (control_agg.get(metric), identity_agg.get(metric))
            for metric, spec in metric_specs.items()
            if (spec.get("role") == "primary" and
                spec.get("axis") in PROTECTED_PRIMARY_AXES and
                control_agg.get(metric) != identity_agg.get(metric))
        }
        if worst_identity_mismatch:
            raise RuntimeError(
                f"{clip}: scale-1 worst-frame evidence differs from control: "
                f"{worst_identity_mismatch}"
            )
        decisions[clip] = select_clip(
            control_agg,
            candidate_aggs,
            metric_specs,
            control["clips"][clip].get("meta", {}),
        )

    evidence_payload = {
        "schema": 8,
        "policy_contract": POLICY_CONTRACT,
        "policy_baseline": policy_baseline,
        "clips": decisions,
    }
    evidence_path = output / "render_grid_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_rows = []
    for source in rows:
        harness_contract = validate_source_harness(source, policy_baseline)
        decision = decisions[source["clip"]]
        safe_scale_ceiling = decision["safe_scale_ceiling"]
        ceiling_confidence = decision["ceiling_confidence"]
        margin_reliability = decision["safety_margin_reliability"]
        disparity = sbsbench.load_float_texture(source["baseline_disparity"])
        if disparity.size == 0 or not np.isfinite(disparity).all():
            raise RuntimeError(
                f"invalid exact baseline disparity: {source['baseline_disparity']}"
            )
        raw_path, raw_sha256 = raw_disparity_path(source)
        raw_disparity = sbsbench.load_float_texture(raw_path)
        validate_loaded_disparity_rasters(
            source, harness_contract, disparity, raw_disparity
        )
        content_disparity = content_raster_values(disparity, harness_contract)
        content_raw_disparity = content_raster_values(
            raw_disparity, harness_contract
        )
        disparity_mean_abs_pct = float(
            np.mean(np.abs(content_disparity)) * 100.0
        )
        disparity_p95_pct = float(
            np.percentile(np.abs(content_disparity), 95) * 100.0
        )
        try:
            source_width = int(source["source_width"])
            source_height = int(source["source_height"])
            clamp_abs = float(source["artistic_full_clamp_abs"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "source label lacks exact source-aspect/clamp provenance"
            ) from error
        if source_width <= 0 or source_height <= 0:
            raise RuntimeError("source label has invalid source geometry")
        # The artifact is already output-eye normalized. Match perceived_disparity_pct's
        # reference-aspect conversion without applying content_scale_x a second time.
        perceived_scale = (
            (float(source["eye_width"]) / float(source["eye_height"])) /
            (5120.0 / 2160.0)
        )
        style_render_targets = {
            name: clamp_aware_summary(
                content_raw_disparity, target_scale, clamp_abs, perceived_scale
            )
            for name, target_scale in decision["style_targets"].items()
        }
        ceiling_render_target = clamp_aware_summary(
            content_raw_disparity, safe_scale_ceiling, clamp_abs, perceived_scale
        )
        row = dict(source)
        row.update({
            "label_schema": 8,
            "policy_contract": POLICY_CONTRACT,
            "stereo_fit_multiplier": source.get(
                "stereo_fit_multiplier", source["baseline_multiplier"]
            ),
            "stereo_fit_confidence": source.get(
                "stereo_fit_confidence", source["confidence"]
            ),
            "style_targets": decision["style_targets"],
            "style_render_targets": style_render_targets,
            "safe_scale_ceiling": safe_scale_ceiling,
            "ceiling_confidence": ceiling_confidence,
            "safety_margin_reliability": margin_reliability,
            # Compatibility aliases for the current global trainer. Both are explicitly the
            # learned safe cap, never a user-selected style target.
            "baseline_multiplier": safe_scale_ceiling,
            "confidence": ceiling_confidence,
            "render_evidence_confidence": margin_reliability,
            "safe_scale_min": decision["safe_scale_min"],
            "safe_scale_max": decision["safe_scale_max"],
            "authored_fit_scale": decision["authored_fit_scale"],
            "authored_fit_psnr": decision["authored_fit_psnr"],
            "render_grid_key": source["clip"],
            "safe_ceiling_render_target": ceiling_render_target,
            "safe_ceiling_exact_pop_spread_pct": ceiling_render_target[
                "exact_pop_spread_pct"
            ],
            "baseline_disparity_mean_abs_pct": disparity_mean_abs_pct,
            "baseline_disparity_p95_pct": disparity_p95_pct,
            "baseline_unclamped_disparity": str(raw_path),
            "baseline_unclamped_disparity_sha256": raw_sha256,
            "baseline_unclamped_disparity_mean_abs_pct": float(
                np.mean(np.abs(content_raw_disparity)) * 100.0
            ),
        })
        output_rows.append(row)

    labels_path = output / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    code = {
        "label_fitter": THIS_DIR / "artistic_stereo_label_fitter.py",
        "policy_contract": THIS_DIR / "artistic_policy_contract.py",
        "label_preparation": Path(__file__).resolve(),
        "image_loader": SBSBENCH_DIR / "sbsbench.py",
        "evaluator_runner": SBSBENCH_DIR / "run_eval.py",
    }
    contract = {
        "schema": 8,
        "label_fitter": "exact-apollo-connected-safe-frontier",
        "policy_contract": POLICY_CONTRACT,
        "policy_baseline": policy_baseline,
        "label_fitter_config": {
            "candidate_scales": candidate_scales,
            "max_candidate_scale_step": MAX_CANDIDATE_SCALE_STEP,
            "objective": "connected-safe-frontier-multistyle",
            "default_style": DEFAULT_STYLE,
            "styles": {
                "immersive": "safe_scale_ceiling",
                "balanced": (
                    "nearest connected sampled scale to 1 + 0.5 * "
                    "(safe_scale_ceiling - 1)"
                ),
                "clean": "identity scale 1.0",
            },
            "learned_target": "safe_scale_ceiling",
            "authored_fit": "diagnostic only; never a learned product target",
            "protected_primary_axes": sorted(PROTECTED_PRIMARY_AXES),
            "protected_metric_reduction": (
                "run_eval directional worst_frame.worst_value; clip averages are not "
                "used for warp/stability feasibility"
            ),
            "exact_pop_metric": EXACT_POP_METRIC,
            "connected_frontier": "stop at first failed sampled scale in each direction",
            "confidence_semantics": (
                "hard actionable target: identity=0, nonidentity safe ceiling=1"
            ),
            "reliability_semantics": (
                "safety_margin_reliability is 0 for identity and 0.5..1 for a "
                "nonidentity ceiling from protected margins and connected-grid support; "
                "never derived from authored PSNR"
            ),
        },
        "rendered_disparity_supervision": {
            "artifact": "baseline_unclamped_disparity_*.f32",
            "artifact_semantics": (
                "exact unclamped full-binocular normalized disparity at artistic scale 1"
            ),
            "final_equation": (
                "clamp(raw * scale, -artistic_full_clamp_abs, "
                "+artistic_full_clamp_abs)"
            ),
            "clamp_provenance": (
                "source-label harness contract computed from source-color dimensions"
            ),
        },
        "code": {role: {"path": str(path), "sha256": sha256(path)}
                 for role, path in code.items()},
        "model_limits": {"scale_delta_max": ART_SCALE_DELTA_MAX},
        "source_labels": {"path": str(Path(source_labels).resolve()),
                          "sha256": sha256(source_labels)},
        "source_label_contract": source_contract,
        "control": {"path": str(Path(control_path).resolve()),
                    "sha256": sha256(control_path)},
        "candidates": candidate_sources,
        "render_grid_evidence": {
            "path": str(evidence_path), "sha256": sha256(evidence_path)
        },
        "thresholds": {"path": str(Path(thresholds_path).resolve()),
                       "sha256": sha256(thresholds_path)},
    }
    contract_path = output / "label_fitter_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    scale_counts = Counter(row["baseline_multiplier"] for row in output_rows)
    summary = {
        "schema": 8,
        "accepted": len(output_rows),
        "rejected": 0,
        "labels_sha256": sha256(labels_path),
        "label_fitter_contract_sha256": sha256(contract_path),
        "clip_counts": dict(Counter(row["clip"] for row in output_rows)),
        "domain_counts": dict(Counter(row.get("domain") for row in output_rows)),
        "policy_role_counts": dict(Counter(row.get("policy_role") for row in output_rows)),
        "selected_scale_counts": {str(key): value for key, value in sorted(scale_counts.items())},
        "default_style": DEFAULT_STYLE,
        "style_scale_counts": {
            name: {
                str(key): value for key, value in sorted(Counter(
                    decision["style_targets"][name] for decision in decisions.values()
                ).items())
            }
            for name in STYLE_NAMES
        },
        "render_grid_evidence_sha256": sha256(evidence_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-labels", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=parse_candidate,
                        help="repeat SCALE=results.json; grid must include 1.0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--thresholds", type=Path,
                        default=SBSBENCH_DIR / "thresholds.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = write_bundle(
        args.source_labels, args.control, args.candidate, args.output,
        args.thresholds, args.overwrite
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "labels": summary["accepted"],
        "selected_scale_counts": summary["selected_scale_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
