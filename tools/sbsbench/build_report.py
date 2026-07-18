#!/usr/bin/env python3
# flake8: noqa: E501 -- embedded HTML/CSS and user-facing metric prose are intentionally contiguous.
"""Assemble the SBS A/B report directly from two run_eval.py runs (control + treatment):
control-vs-treatment bar charts (one pair per clip), the gate's verdict, and one
section per triggered issue with control/treatment crops at each issue's WORST frame.

Usage: generate_report.py <control_run_dir> <treat_run_dir> <out.html>
       (run dirs = <build-dir>/sbs_eval/<label>/ containing results.json + <clip>/sbs_*.png)
"""
import base64
import functools
import html
import io
import json
import os
import sys

# Pin numeric kernels before NumPy initializes. Frame-level workers own parallelism.
for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np  # noqa: E402  (thread limits must precede numeric-runtime import)
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

ctrl_dir, treat_dir, out_html = sys.argv[1], sys.argv[2], sys.argv[3]
allow_config_diff = "--allow-config-diff" in sys.argv[4:]
allow_model_diff = "--allow-model-diff" in sys.argv[4:]
allow_depth_step_diff = "--allow-depth-step-diff" in sys.argv[4:]
allow_executable_diff = "--allow-executable-diff" in sys.argv[4:]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbsbench  # noqa: E402
import sbs_interocular_phase_chroma  # noqa: E402
import sbs_interocular_photometric_rivalry  # noqa: E402
import sbs_stereo_window_metrics  # noqa: E402
import sbs_warp_shear_metrics  # noqa: E402
import run_eval  # noqa: E402  (evaluation-contract and clip identity helpers)
import offline_oracle_report  # noqa: E402  (optional diagnostic appendix only)

# Report verification performs the same authoritative pixel remeasurement as the evaluator. A
# direct top-level invocation cannot safely spawn Windows workers; the documented guarded
# generate_report.py launcher preselects the faster process backend before importing this module.
if __name__ == "__main__":
    os.environ[sbsbench.SEQUENCE_SPATIAL_BACKEND_ENV] = "thread"
else:
    os.environ.setdefault(sbsbench.SEQUENCE_SPATIAL_BACKEND_ENV, "thread")
sbsbench.enable_reusable_spatial_executor()

CTRL = json.load(open(os.path.join(ctrl_dir, "results.json")))
TREAT = json.load(open(os.path.join(treat_dir, "results.json")))
THRESHOLD_CFG = json.load(open(os.path.join(SCRIPT_DIR, "thresholds.json")))
THR = THRESHOLD_CFG["metrics"]
CURRENT_METRIC_SHA = run_eval.metric_contract_sha()
CURRENT_LABEL_SHA = run_eval.label_contract_sha()
CURRENT_METRIC_RUNTIME = run_eval.metric_runtime_provenance()
REPORT_SHA = run_eval.sha256_files([
    os.path.abspath(__file__), os.path.abspath(offline_oracle_report.__file__)])


def _validate_metric_runtime(run, side, expected):
    """Require report inputs to use the numeric runtime active for this report."""
    observed = run.get("meta", {}).get("metric_runtime")
    if observed != expected:
        raise SystemExit(
            f"refusing {side} metrics produced by a different numeric runtime: "
            f"recorded={observed!r}, current={expected!r}")


_validate_metric_runtime(CTRL, "control", CURRENT_METRIC_RUNTIME)
_validate_metric_runtime(TREAT, "treatment", CURRENT_METRIC_RUNTIME)

# An A/B report may compare different code, profile/treatment arguments, or (only when explicitly
# requested) depth models. Its evidence remains invalid if the source set or metric contract changed.
_SAME_CONTEXT = ["clip_set_sha1", "mode", "eval_schema", "depth_step", "suite", "run_kind",
                 "metric_sha256", "label_contract_sha256", "metric_runtime", "executable_sha256",
                 "runtime_shader_sha256"]
if not allow_model_diff:
    _SAME_CONTEXT.extend(["model", "engine_sha256", "onnx_sha256"])
if not allow_config_diff:
    _SAME_CONTEXT.append("conf_sha256")
if allow_depth_step_diff:
    _SAME_CONTEXT.remove("depth_step")
if allow_executable_diff:
    _SAME_CONTEXT.remove("executable_sha256")
    _SAME_CONTEXT.remove("runtime_shader_sha256")
_mismatched_context = {k: (CTRL.get("meta", {}).get(k), TREAT.get("meta", {}).get(k))
                       for k in _SAME_CONTEXT
                       if CTRL.get("meta", {}).get(k) != TREAT.get("meta", {}).get(k)}
if _mismatched_context:
    raise SystemExit(f"refusing incompatible A/B report: {_mismatched_context}")
if CTRL.get("meta", {}).get("eval_schema") != run_eval.EVAL_SCHEMA:
    raise SystemExit(
        f"refusing stale evaluator schema {CTRL.get('meta', {}).get('eval_schema')!r}; "
        f"rerun with current schema {run_eval.EVAL_SCHEMA}")
if CTRL.get("meta", {}).get("metric_sha256") != CURRENT_METRIC_SHA:
    raise SystemExit("refusing stale evaluation artifacts: rescore or rerun both inputs with the current eval contract")
if CTRL.get("meta", {}).get("label_contract_sha256") != CURRENT_LABEL_SHA:
    raise SystemExit(
        "refusing stale label eligibility/provenance contract: rescore or rerun both inputs")

CLIPS_ROOT = CTRL.get("meta", {}).get("clips_root") or os.path.join(SCRIPT_DIR, "clips")


@functools.lru_cache(maxsize=None)
def source_files(clip):
    return sbsbench.indexed_files(
        os.path.join(CLIPS_ROOT, clip, "frame_*.*"), "frame_")


@functools.lru_cache(maxsize=None)
def gt_depth_files(clip):
    return sbsbench.indexed_files(
        os.path.join(CLIPS_ROOT, clip, "gt_depth", "frame_*.*"), "frame_")


@functools.lru_cache(maxsize=None)
def gt_flow_files(clip):
    return sbsbench.indexed_files(
        os.path.join(CLIPS_ROOT, clip, "gt_flow", "frame_*.npz"), "frame_")


@functools.lru_cache(maxsize=None)
def run_files(run, clip, prefix):
    return sbsbench.indexed_files(
        os.path.join(run, clip, prefix + "*.*"), prefix)


def source_path(clip, frame_id):
    return source_files(clip).get(frame_id)


def gt_depth_path(clip, frame_id):
    return gt_depth_files(clip).get(frame_id)


def artifact_path(run, clip, prefix, frame_id):
    return run_files(run, clip, prefix).get(frame_id)


def _clip_name(clip):
    # Prefer the name run_eval captured into results.json; else the repo clip's meta.json; else id.
    nm = CTRL["clips"].get(clip, {}).get("meta", {}).get("name")
    if nm:
        return nm
    mp = os.path.join(CLIPS_ROOT, clip, "meta.json")
    if os.path.exists(mp):
        try:
            return json.load(open(mp)).get("name", clip)
        except Exception:
            pass
    return clip


_NAME_CACHE = {}


def name(clip):
    """Scene display name for a clip id (from its meta.json; falls back to the id)."""
    if clip not in _NAME_CACHE:
        _NAME_CACHE[clip] = _clip_name(clip)
    return _NAME_CACHE[clip]


CLIPS = sorted(CTRL["clips"])
if set(CLIPS) != set(TREAT["clips"]):
    raise SystemExit("refusing A/B report with different clip sets")


def _validate_authoritative_results(
        run, run_dir, side, clips_root, remeasurement_session=None):
    """Remeasure report inputs; JSON aggregate caches never authenticate themselves."""
    try:
        return run_eval.verify_results_against_artifacts(
            run, run_dir, clips_root, THRESHOLD_CFG,
            remeasurement_session=remeasurement_session)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"refusing {side} results that fail authoritative remeasurement: {exc}") from exc


_validated_run_dirs = {}
_remeasurement_session = run_eval.new_remeasurement_session()
for _run, _run_dir, _side in (
        (CTRL, ctrl_dir, "control"), (TREAT, treat_dir, "treatment")):
    _validation_key = os.path.normcase(os.path.abspath(_run_dir))
    if _validation_key not in _validated_run_dirs:
        _validated_run_dirs[_validation_key] = _validate_authoritative_results(
            _run, _run_dir, _side, CLIPS_ROOT,
            remeasurement_session=_remeasurement_session)
# Compact decision/report vector. Numeric support fields remain in results.json but do not become
# separate report axes. Metrics omitted here are intentionally not presented as SBS quality.
# metric, header, worse-is-higher, always-show, notable-threshold
COLS = [
    ("exact_visible_pop_spread_pct", "visible_pop", False, True, 0),
    ("exact_binocular_support_pct", "binocular_support", False, True, 0),
    ("exact_positive_disparity_pct", "disp_positive", True, True, 0),
    ("exact_negative_disparity_pct", "disp_negative", True, True, 0),
    ("exact_over_3pct_area_pct", "over_limit_area", True, False, 0.01),
    ("exact_polarity_ok", "depth_polarity", False, True, 0),
    ("exact_local_polarity_component_pct", "local_polarity", True, False, 0.01),
    ("exact_mapping_stretch_pct", "mapping_stretch", True, True, 0),
    ("exact_mapping_fold_pct", "mapping_fold", True, False, 0.01),
    ("exact_symmetry_residual_p95_pct", "camera_symmetry", True, False, 0.01),
    ("warp_cross_row_shear_severity_pct", "row_shear", True, False, 0.01),
    ("experimental_stereo_window_crossed_burden_pct", "window_conflict", True, False, 0.001),
    ("interocular_phase_orientation_burden_pct", "phase_conflict", True, False, 0.01),
    ("interocular_exposure_rivalry_burden_pct", "exposure_rivalry", True, False, 0.01),
    ("interocular_color_gain_rivalry_burden_pct", "color_rivalry", True, False, 0.01),

    ("source_coverage_pct", "render_coverage", False, True, 0),
    ("image_integrity_pct", "render_integrity", False, True, 0),
    ("source_coverage_worst_patch_bad_pct", "localized_missing", True, True, 0),
    ("image_integrity_worst_patch_bad_pct", "localized_texture_damage", True, True, 0),
    ("vmisalign_p99_pct", "vertical_mismatch", True, True, 0),

    ("depth_gt_affine_nrmse_pct", "gt_depth_affine_nrmse", True, True, 0),
    ("depth_gt_edge_f1", "gt_edge_f1", False, True, 0),
    ("depth_gt_polarity_ok", "gt_polarity", False, True, 0),
    ("static_jitter_p95", "static_jitter", True, True, 0),
    ("flow_temporal_p95", "flow_temporal", True, True, 0),
    ("depth_gt_lag_f1_p95", "gt_depth_lag", True, True, 0),
]

# The scalar artifact score was removed: correlated metrics must not cancel one another. Sorting
# uses only the explicit decision role, with the compact declaration order breaking ties.


def impact(k):
    return {"hard": 3.0, "primary": 2.0, "diagnostic": 1.0}.get(
        THR.get(k, {}).get("role"), 0.0)


COLS = sorted(COLS, key=lambda c: -impact(c[0]))
SHORT = {k: h for k, h, *_ in COLS}
EXACT_STEREO_VISUAL_METRICS = {
    "exact_visible_pop_spread_pct",
}


def durl(im, w=None, jpg=False, q=82):
    if w and im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    if jpg:
        im.convert("RGB").save(b, "JPEG", quality=q)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    im.convert("RGB").save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def load_depth(p):
    return sbsbench.load_depth(p)


def frame_path(run, clip, i):
    return artifact_path(run, clip, "sbs_", i)


def mid_frame(run, clip):
    frame_ids = sorted(run_files(run, clip, "sbs_"))
    return frame_ids[len(frame_ids) // 2] if frame_ids else 0


def normalize_sbs_images(images):
    """Resize packed SBS images to one common per-eye raster for visual A/B evidence.

    Evaluation runs may intentionally use different output resolutions.  Comparing their raw
    arrays would either fail or turn ordinary resampling into a spatial offset.  Keep the SBS
    seam exact and compare at the smallest available raster so neither run is upscaled.
    """
    eye_w = min(image.width // 2 for image in images)
    height = min(image.height for image in images)
    normalized = []
    for image in images:
        if image.width % 2:
            raise ValueError(f"packed SBS width must be even, got {image.width}")
        source_eye_width = image.width // 2
        eyes = [image.crop((offset, 0, offset + source_eye_width, image.height))
                for offset in (0, source_eye_width)]
        eyes = [eye if eye.size == (eye_w, height) else
                eye.resize((eye_w, height), Image.LANCZOS) for eye in eyes]
        packed = Image.new(image.mode, (eye_w * 2, height))
        packed.paste(eyes[0], (0, 0))
        packed.paste(eyes[1], (eye_w, 0))
        normalized.append(packed)
    return normalized


def crop_at_silhouette(clip, idx):
    """Control/treatment left-eye crops at the strongest depth silhouette of frame idx (falls
    back to center if the depth is flat). Returns (ctrl_durl, treat_durl) or None."""
    cp, tp = frame_path(ctrl_dir, clip, idx), frame_path(treat_dir, clip, idx)
    dp = artifact_path(ctrl_dir, clip, "depth_", idx)
    if not (cp and tp and dp):
        return None
    depth = load_depth(dp)
    sbs_c, sbs_t = normalize_sbs_images([Image.open(cp), Image.open(tp)])
    ew, eh = sbs_c.width // 2, sbs_c.height
    gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    dh, dw = depth.shape
    band = gx[int(dh * 0.15):int(dh * 0.85)]
    colscore = band.sum(0)
    lo, hi = int(dw * 0.1), int(dw * 0.9)
    cx_d = int(np.argmax(colscore[lo:hi]) + lo) if colscore[lo:hi].max() > 0.1 else dw // 2
    rowscore = gx[:, max(0, cx_d - 2):cx_d + 3].sum(1)
    cy_d = int(np.argmax(rowscore)) if rowscore.max() > 0 else dh // 2
    cx, cy = int(cx_d / dw * ew), int(cy_d / dh * eh)
    cw, ch = min(480, ew), min(360, eh)
    x0 = max(0, min(ew - cw, cx - cw // 2))
    y0 = max(0, min(eh - ch, cy - ch // 2))
    out = []
    for img in (sbs_c, sbs_t):
        # Crop as a modest-width JPEG; the page CSS scales it up. Keep the report light (many
        # crops embed as data URIs) so the artifact viewer loads reliably.
        out.append(durl(img.crop((x0, y0, x0 + cw, y0 + ch)), w=380, jpg=True, q=78))
    return out


EVIDENCE_METADATA = {}


def _expanded_map(values, size=5):
    """Visual-only dilation; localization always uses the undilated detector map."""
    values = np.asarray(values, dtype=np.uint8)
    return np.asarray(Image.fromarray(values, mode="L").filter(ImageFilter.MaxFilter(size)))


def _artifact_analysis_rgb(value_map, support_map):
    """Black/cyan detector support with red/yellow metric-specific severity."""
    values = np.asarray(value_map, dtype=np.float32)
    support = np.asarray(support_map, dtype=bool)
    if values.shape != support.shape or values.ndim != 2:
        raise ValueError("artifact value/support maps must be matching HxW arrays")
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    scale = float(np.max(values[positive])) if positive.any() else 1.0
    normalized = np.zeros(values.shape, dtype=np.uint8)
    normalized[finite] = np.rint(
        np.clip(values[finite] / max(scale, 1e-12), 0.0, 1.0) * 255.0).astype(np.uint8)
    support_visible = _expanded_map(support.astype(np.uint8) * 255) > 0
    severity_visible = _expanded_map(normalized)
    heat = np.zeros((*values.shape, 3), dtype=np.uint8)
    heat[support_visible] = (0, 105, 135)
    heat[..., 0] = np.maximum(heat[..., 0], severity_visible)
    heat[..., 1] = np.maximum(
        heat[..., 1], (severity_visible.astype(np.float32) * 0.72).astype(np.uint8))
    return heat


def _support_analysis_rgb(support):
    """Separate analysis mask: cyan means evaluated support; black means excluded."""
    support = np.asarray(support, dtype=bool)
    if support.ndim != 2:
        raise ValueError("analysis support must be HxW")
    image = np.zeros((*support.shape, 3), dtype=np.uint8)
    image[support] = (0, 210, 235)
    return image


def _label_analysis_image(image, lines):
    """Label non-source visualizations so overlays cannot be mistaken for source artifacts."""
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    labels = [str(line) for line in lines if line]
    if not labels:
        return image
    line_height = 13
    box_height = 6 + line_height * len(labels)
    draw.rectangle((0, 0, image.width, min(image.height, box_height)), fill=(5, 12, 16))
    for index, label in enumerate(labels):
        draw.text((5, 3 + index * line_height), label, fill=(235, 250, 252))
    return image


def _crop_array_at_normalized(array, center, fractions):
    """Crop equal normalized source regions from potentially different output rasters."""
    array = np.asarray(array)
    height, width = array.shape[:2]
    nx, ny = center
    fraction_w, fraction_h = fractions
    crop_width = max(1, min(width, int(round(width * fraction_w))))
    crop_height = max(1, min(height, int(round(height * fraction_h))))
    cx = int(round(nx * width - 0.5))
    cy = int(round(ny * height - 0.5))
    x0 = max(0, min(width - crop_width, cx - crop_width // 2))
    y0 = max(0, min(height - crop_height, cy - crop_height // 2))
    return array[y0:y0 + crop_height, x0:x0 + crop_width]


def _shear_evidence_for_run(run, clip, idx):
    """Rerun the exact-map row-shear detector independently for both eyes."""
    exact = _exact_source_evidence_for_run(run, clip, idx)
    source = source_path(clip, idx)
    mapping_path = artifact_path(run, clip, "warp_map_", idx)
    shape_path = os.path.join(run, clip, "warp_map_shape.json")
    if not exact or not source or not mapping_path or not os.path.exists(shape_path):
        return None
    shape = json.load(open(shape_path, encoding="utf-8"))
    mapping = sbsbench.load_warp_mapping(mapping_path, shape)
    source_rgb = sbsbench.load_rgb(source)
    measured = []
    for eye_map in sbsbench.split_eyes(mapping):
        metrics, maps = sbs_warp_shear_metrics.measure_cross_row_shear(
            eye_map, shape, source=source_rgb, return_maps=True)
        measured.append({"metrics": metrics, "maps": maps})
    return {"output": exact["output"], "reference": exact["expected"],
            "measured": measured}


def cross_row_shear_evidence(clip, idx, metric="warp_cross_row_shear_severity_pct"):
    """Metric-authenticated crop at the strongest unsupported horizontal row tear."""
    runs = [_shear_evidence_for_run(run, clip, idx) for run in (ctrl_dir, treat_dir)]
    if not all(runs):
        return None
    ranked = []
    for run_index, data in enumerate(runs):
        for eye_index, measured in enumerate(data["measured"]):
            value = measured["metrics"].get(metric)
            if value is None:
                continue
            row_shear = np.asarray(
                measured["maps"]["row_shear_ref_px_per_row"], dtype=np.float32)
            bad = np.asarray(measured["maps"]["bad"], dtype=bool)
            localized = np.where(bad & np.isfinite(row_shear), row_shear, -np.inf)
            strongest = float(np.max(localized)) if np.isfinite(localized).any() else -np.inf
            ranked.append((float(value), strongest, -run_index, -eye_index,
                           run_index, eye_index, measured))
    if not ranked:
        return None
    _, _, _, _, selected_run, eye_index, selected = max(ranked)
    row_shear = np.asarray(selected["maps"]["row_shear_ref_px_per_row"], dtype=np.float32)
    support = np.asarray(selected["maps"]["support"], dtype=bool)
    bad = np.asarray(selected["maps"]["bad"], dtype=bool)
    localized = np.where(bad & np.isfinite(row_shear), row_shear, -np.inf)
    if np.isfinite(localized).any():
        cy, cx = np.unravel_index(int(np.argmax(localized)), localized.shape)
    else:
        locations = np.argwhere(support)
        if not locations.size:
            return None
        center = (np.asarray(support.shape, dtype=np.float32) - 1.0) * 0.5
        cy, cx = locations[np.argmin(np.sum((locations - center) ** 2, axis=1))]
    height, width = row_shear.shape
    normalized_center = ((float(cx) + 0.5) / width, (float(cy) + 0.5) / height)
    crop_fractions = (min(1.0, 480.0 / width), min(1.0, 360.0 / height))
    reference = _crop_array_at_normalized(
        runs[selected_run]["reference"][eye_index], normalized_center, crop_fractions)
    control = _crop_array_at_normalized(
        runs[0]["output"][eye_index], normalized_center, crop_fractions)
    treatment = _crop_array_at_normalized(
        runs[1]["output"][eye_index], normalized_center, crop_fractions)
    contribution = np.where(bad, row_shear, 0.0)
    analysis = _crop_array_at_normalized(
        _artifact_analysis_rgb(contribution, support), normalized_center, crop_fractions)
    support_count = int(selected["metrics"].get(
        "warp_cross_row_shear_support_count", 0))
    selected_tag = CTRL_TAG if selected_run == 0 else TREAT_TAG
    eye_name = "left" if eye_index == 0 else "right"
    analysis_image = _label_analysis_image(Image.fromarray(analysis), [
        "row-shear detector (not source content)",
        f"cyan=support, red/yellow=unsupported tear; n={support_count}",
    ])
    EVIDENCE_METADATA[(clip, idx, metric)] = {
        "eye": eye_name, "selected_run": selected_tag, "support": support_count,
    }
    return (
        durl(Image.fromarray(np.clip(reference * 255.0, 0, 255).astype(np.uint8)),
             w=380, jpg=True, q=88),
        durl(Image.fromarray(np.clip(control * 255.0, 0, 255).astype(np.uint8)),
             w=380, jpg=True, q=88),
        durl(Image.fromarray(np.clip(treatment * 255.0, 0, 255).astype(np.uint8)),
             w=380, jpg=True, q=88),
        durl(analysis_image, w=380),
    )


def _exact_source_evidence_for_run(run, clip, idx):
    """Return per-eye output, exact mapped source, residual, and mapping-topology maps."""
    output_path = frame_path(run, clip, idx)
    mapping_path = artifact_path(run, clip, "warp_map_", idx)
    source = source_path(clip, idx)
    shape_path = os.path.join(run, clip, "warp_map_shape.json")
    if not output_path or not mapping_path or not source or not os.path.exists(shape_path):
        return None
    shape = json.load(open(shape_path, encoding="utf-8"))
    mapping = sbsbench.load_warp_mapping(mapping_path, shape)
    output = sbsbench.load_rgb(output_path)
    if output.shape[:2] != mapping.shape:
        raise ValueError(f"report warp map {mapping.shape} != output {output.shape[:2]}")
    source_rgb = sbsbench.load_rgb(source)
    hdr_scale = None
    hdr_path = os.path.join(run, clip, "hdr_output_stats.json")
    if os.path.exists(hdr_path):
        hdr_scale = float(json.load(open(hdr_path, encoding="utf-8"))["input_scale"])
    eye_outputs = sbsbench.split_eyes(output)
    eye_maps = sbsbench.split_eyes(mapping)
    height, eye_width = eye_outputs[0].shape[:2]
    scale_x, scale_y = float(shape["content_scale_x"]), float(shape["content_scale_y"])
    output_v = (np.arange(height, dtype=np.float32) + 0.5) / height
    lo_y = 0.5 * (1.0 - scale_y)
    source_v = np.broadcast_to(
        np.clip((output_v - lo_y) / scale_y, 0.0, 1.0)[:, None], (height, eye_width))
    if hdr_scale is None:
        sample_source = source_rgb
    else:
        sample_source = (sbsbench._srgb_to_linear(source_rgb) * hdr_scale).astype(
            np.float16).astype(np.float32)
    expected, residuals, topology = [], [], []
    baseline_step = float(shape["source_width"]) / max(scale_x * eye_width, 1.0)
    for eye, sampled_u in zip(eye_outputs, eye_maps):
        # The sidecar preserves raw Reproject U for geometry. Displayed color and topology both
        # consume main_ps's saturated coordinate; raw U remains available to comfort metrics.
        live_sampled_u = np.clip(sampled_u, 0.0, 1.0)
        mapped = sbsbench._sample_rgb_uv(sample_source, live_sampled_u, source_v)
        if hdr_scale is not None:
            mapped = sbsbench._hdr_preview_rgb(mapped)
        expected.append(mapped)
        residuals.append(np.abs(sbsbench.rgb_luma(eye) - sbsbench.rgb_luma(mapped)))
        source_step = np.diff(live_sampled_u * float(shape["source_width"]), axis=1,
                              prepend=live_sampled_u[:, :1] * float(shape["source_width"]))
        topology.append(np.clip(1.0 - np.abs(source_step) / max(baseline_step, 1e-6), 0.0, 1.0))
    return {"output": eye_outputs, "expected": expected, "residual": residuals,
            "topology": topology}


def _detector_inputs_for_run(run, clip, idx):
    """Load the exact artifacts used by the retained perceptual detectors."""
    output_path = frame_path(run, clip, idx)
    mapping_path = artifact_path(run, clip, "warp_map_", idx)
    mask_path = artifact_path(run, clip, "warp_mask_", idx)
    depth_path = artifact_path(run, clip, "depth_", idx)
    input_path = source_path(clip, idx)
    shape_path = os.path.join(run, clip, "warp_map_shape.json")
    if not all((output_path, mapping_path, mask_path, depth_path, input_path)):
        return None
    if not os.path.exists(shape_path):
        return None
    shape = json.load(open(shape_path, encoding="utf-8"))
    mapping = sbsbench.load_warp_mapping(mapping_path, shape)
    output = sbsbench.load_rgb(output_path)
    source = sbsbench.load_rgb(input_path)
    depth = sbsbench.load_depth(depth_path)
    mask = np.asarray(Image.open(mask_path).convert("RGB"), np.float32) / 255.0
    if output.shape[:2] != mapping.shape or mask.shape[:2] != mapping.shape:
        raise ValueError("detector output/map/mask geometry mismatch")
    metric_source = source
    sample_transform = None
    hdr_path = os.path.join(run, clip, "hdr_output_stats.json")
    if os.path.exists(hdr_path):
        hdr_scale = float(json.load(open(hdr_path, encoding="utf-8"))["input_scale"])
        metric_source = (
            sbsbench._srgb_to_linear(source) * hdr_scale
        ).astype(np.float16).astype(np.float32)
        sample_transform = sbsbench._hdr_preview_rgb
    return {
        "shape": shape,
        "mapping": mapping,
        "mask": mask,
        "depth": depth,
        "source": metric_source,
        "source_preview": source,
        "sample_transform": sample_transform,
        "eyes": sbsbench.split_eyes(output),
    }


def _stereo_pair_arrays(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("registered eye evidence must share geometry")
    separator = np.zeros((left.shape[0], 4, 3), dtype=np.float32)
    if left.ndim == 2:
        left = np.repeat(left[..., None], 3, axis=2)
        right = np.repeat(right[..., None], 3, axis=2)
    return np.concatenate((left[..., :3], separator, right[..., :3]), axis=1)


def _rgb_data_url(array, width=380):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    # Registered evidence intentionally carries NaN outside exact mutual support. Display those
    # pixels as black; never rely on NumPy's platform-dependent float-to-uint8 invalid cast.
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    return durl(Image.fromarray(
        np.clip(array[..., :3] * 255.0, 0, 255).astype(np.uint8)),
        w=width, jpg=True, q=88)


def interocular_conflict_evidence(clip, idx, metric):
    """Source-registered stereo pairs and the phase/orientation conflict map."""
    inputs = [_detector_inputs_for_run(run, clip, idx)
              for run in (ctrl_dir, treat_dir)]
    if not all(inputs):
        return None
    measured = []
    for data in inputs:
        left_map, right_map = sbsbench.split_eyes(data["mapping"])
        metrics, maps = sbs_interocular_phase_chroma.measure_interocular_phase_chroma(
            data["source"], data["eyes"][0], data["eyes"][1], left_map, right_map,
            data["shape"], warp_mask=data["mask"],
            source_sample_transform=data["sample_transform"],
            return_maps=True)
        measured.append((metrics, maps))
    ranked = [(float(item[0][metric]), run_index)
              for run_index, item in enumerate(measured)
              if item[0].get(metric) is not None]
    if not ranked:
        return None
    _, selected_run = max(ranked)
    maps = measured[selected_run][1]
    support = np.asarray(maps["phase_orientation_support"], dtype=bool)
    values = np.asarray(maps["phase_orientation_conflict_pct"], dtype=np.float32)
    localized = np.where(support & np.isfinite(values), values, -np.inf)
    if not np.isfinite(localized).any():
        return None
    cy, cx = np.unravel_index(int(np.argmax(localized)), localized.shape)
    center = ((float(cx) + 0.5) / values.shape[1],
              (float(cy) + 0.5) / values.shape[0])
    fractions = (min(1.0, 480.0 / values.shape[1]),
                 min(1.0, 360.0 / values.shape[0]))
    reference = _crop_array_at_normalized(
        maps["registered_source_rgb"], center, fractions)
    stereo_pairs = []
    for _, run_maps in measured:
        left = _crop_array_at_normalized(
            run_maps["registered_left_rgb"], center, fractions)
        right = _crop_array_at_normalized(
            run_maps["registered_right_rgb"], center, fractions)
        stereo_pairs.append(_stereo_pair_arrays(left, right))
    analysis = _crop_array_at_normalized(
        _artifact_analysis_rgb(np.where(support, values, 0.0), support),
        center, fractions)
    support_count = int(measured[selected_run][0].get(
        "interocular_phase_orientation_support_count", 0))
    EVIDENCE_METADATA[(clip, idx, metric)] = {
        "selected_run": CTRL_TAG if selected_run == 0 else TREAT_TAG,
        "support": support_count,
        "detector": "phase/orientation",
        "stereo_pairs": True,
    }
    analysis_image = _label_analysis_image(Image.fromarray(analysis), [
        "registered conflict map (not source content)",
        f"cyan=support, red/yellow=conflict; n={support_count}",
    ])
    return (_rgb_data_url(reference), _rgb_data_url(stereo_pairs[0], 520),
            _rgb_data_url(stereo_pairs[1], 520), durl(analysis_image, w=380))


def interocular_photometric_evidence(clip, idx, metric):
    """Exact-source-registered exposure or colour-gain rivalry evidence."""
    inputs = [_detector_inputs_for_run(run, clip, idx)
              for run in (ctrl_dir, treat_dir)]
    if not all(inputs):
        return None
    measured = []
    for data in inputs:
        left_map, right_map = sbsbench.split_eyes(data["mapping"])
        metrics, maps = (
            sbs_interocular_photometric_rivalry
            .measure_interocular_photometric_rivalry(
                data["source"], data["eyes"][0], data["eyes"][1],
                left_map, right_map, data["shape"], warp_mask=data["mask"],
                source_sample_transform=data["sample_transform"], return_maps=True))
        measured.append((metrics, maps))
    ranked = [(float(item[0][metric]), run_index)
              for run_index, item in enumerate(measured)
              if item[0].get(metric) is not None]
    if not ranked:
        return None
    _, selected_run = max(ranked)
    maps = measured[selected_run][1]
    exposure = metric.startswith("interocular_exposure_")
    value_key = "exposure_rivalry_pct" if exposure else "color_gain_rivalry_pct"
    support = np.asarray(maps["photometric_support"], dtype=bool)
    values = np.asarray(maps[value_key], dtype=np.float32)
    localized = np.where(support & np.isfinite(values), values, -np.inf)
    if not np.isfinite(localized).any():
        return None
    cy, cx = np.unravel_index(int(np.argmax(localized)), localized.shape)
    center = ((float(cx) + 0.5) / values.shape[1],
              (float(cy) + 0.5) / values.shape[0])
    fractions = (min(1.0, 480.0 / values.shape[1]),
                 min(1.0, 360.0 / values.shape[0]))
    reference = _crop_array_at_normalized(
        maps["registered_source_rgb"], center, fractions)
    stereo_pairs = []
    for _, run_maps in measured:
        left = _crop_array_at_normalized(
            run_maps["registered_left_rgb"], center, fractions)
        right = _crop_array_at_normalized(
            run_maps["registered_right_rgb"], center, fractions)
        stereo_pairs.append(_stereo_pair_arrays(left, right))
    analysis = _crop_array_at_normalized(
        _artifact_analysis_rgb(np.where(support, values, 0.0), support),
        center, fractions)
    prefix = ("interocular_exposure_rivalry" if exposure else
              "interocular_color_gain_rivalry")
    support_count = int(measured[selected_run][0].get(
        f"{prefix}_support_count", 0))
    EVIDENCE_METADATA[(clip, idx, metric)] = {
        "selected_run": CTRL_TAG if selected_run == 0 else TREAT_TAG,
        "support": support_count,
        "detector": "exposure rivalry" if exposure else "colour-gain rivalry",
        "stereo_pairs": True,
    }
    analysis_image = _label_analysis_image(Image.fromarray(analysis), [
        "registered rivalry map (not source content)",
        f"cyan=support, red/yellow=rivalry; n={support_count}",
    ])
    return (_rgb_data_url(reference), _rgb_data_url(stereo_pairs[0], 520),
            _rgb_data_url(stereo_pairs[1], 520), durl(analysis_image, w=380))


def stereo_window_evidence(clip, idx, metric):
    """Border-localized source, stereo pairs, and crossed window-risk contribution."""
    inputs = [_detector_inputs_for_run(run, clip, idx)
              for run in (ctrl_dir, treat_dir)]
    if not all(inputs):
        return None
    measured = []
    for data in inputs:
        shape = data["shape"]
        depth = sbsbench.resize_depth(
            data["depth"], int(shape["source_width"]), int(shape["source_height"]))
        coverage = data["mask"][..., 0] < 0.5
        metrics, maps = sbs_stereo_window_metrics.measure_stereo_window_violation(
            data["mapping"], shape, data["source"], depth=depth,
            coverage_mask=coverage, source_sample_transform=data["sample_transform"],
            return_maps=True)
        measured.append((metrics, maps))
    ranked = [(float(item[0][metric]), run_index)
              for run_index, item in enumerate(measured)
              if item[0].get(metric) is not None]
    if not ranked:
        return None
    _, selected_run = max(ranked)
    maps = measured[selected_run][1]
    support = np.asarray(maps["support"], dtype=bool)
    risk = np.asarray(maps["crossed_risk"], dtype=bool)
    contribution = np.asarray(maps["crossed_contribution"], dtype=np.float32)
    localized = np.where(risk & np.isfinite(contribution), contribution, -np.inf)
    if np.isfinite(localized).any():
        cy, cx = np.unravel_index(int(np.argmax(localized)), localized.shape)
    else:
        locations = np.argwhere(support)
        if not locations.size:
            return None
        cy, cx = locations[len(locations) // 2]
    left_cut = bool(np.asarray(maps["crossed_left_cut"], dtype=bool)[cy, cx])
    right_cut = bool(np.asarray(maps["crossed_right_cut"], dtype=bool)[cy, cx])
    use_left = left_cut or (not right_cut and cx < contribution.shape[1] // 2)
    x_center = 0.18 if use_left else 0.82
    y_center = (float(cy) + 0.5) / contribution.shape[0]
    fractions = (0.36, min(1.0, 360.0 / contribution.shape[0]))
    source_crop = _crop_array_at_normalized(
        maps["source_luma"], (x_center, y_center), fractions)
    stereo_pairs = []
    for data in inputs:
        left = _crop_array_at_normalized(
            data["eyes"][0], (x_center, y_center), fractions)
        right = _crop_array_at_normalized(
            data["eyes"][1], (x_center, y_center), fractions)
        stereo_pairs.append(_stereo_pair_arrays(left, right))
    analysis = _crop_array_at_normalized(
        _artifact_analysis_rgb(np.where(risk, contribution, 0.0), support),
        (x_center, y_center), fractions)
    support_count = int(measured[selected_run][0].get(
        "experimental_stereo_window_support_count", 0))
    EVIDENCE_METADATA[(clip, idx, metric)] = {
        "selected_run": CTRL_TAG if selected_run == 0 else TREAT_TAG,
        "support": support_count,
        "border": "left" if use_left else "right",
        "stereo_pairs": True,
    }
    analysis_image = _label_analysis_image(Image.fromarray(analysis), [
        "stereo-window detector (not source content)",
        f"cyan=support, red/yellow=crossed cut; n={support_count}",
    ])
    return (_rgb_data_url(source_crop), _rgb_data_url(stereo_pairs[0], 520),
            _rgb_data_url(stereo_pairs[1], 520), durl(analysis_image, w=380))


def source_residual_evidence(clip, idx, metric=None):
    """Exact mapped-source or source-coordinate topology evidence for one metric."""
    runs = [_exact_source_evidence_for_run(run, clip, idx) for run in (ctrl_dir, treat_dir)]
    if not all(runs):
        return None
    eye_w = min(data["output"][0].shape[1] for data in runs)
    height = min(data["output"][0].shape[0] for data in runs)

    def resized(array):
        if array.shape[:2] == (height, eye_w):
            return array
        if array.ndim == 3:
            image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), "RGB")
            return np.asarray(image.resize((eye_w, height), Image.BILINEAR), np.float32) / 255.0
        return sbsbench.resize_to(array, eye_w, height)

    kind = "topology" if metric in {
        "exact_mapping_stretch_pct", "exact_mapping_fold_pct"
    } else "residual"
    maps = [[resized(data[kind][eye]) for eye in range(2)] for data in runs]
    eye_idx = max(range(2), key=lambda eye: float(np.percentile(
        np.abs(maps[1][eye] - maps[0][eye]), 95)))
    delta = maps[1][eye_idx] - maps[0][eye_idx]
    score = sbsbench._box3(np.abs(delta))
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    cw, ch = min(480, eye_w), min(360, height)
    x0 = max(0, min(eye_w - cw, int(cx) - cw // 2))
    y0 = max(0, min(height - ch, int(cy) - ch // 2))
    crop = (x0, y0, x0 + cw, y0 + ch)
    expected = resized(runs[1]["expected"][eye_idx])
    control = resized(runs[0]["output"][eye_idx])
    treatment = resized(runs[1]["output"][eye_idx])
    d = delta[y0:y0 + ch, x0:x0 + cw] * 255.0
    heat = np.zeros((*d.shape, 3), np.uint8)
    heat[..., 0] = np.clip(d * 12.0, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip(-d * 12.0, 0, 255).astype(np.uint8)
    return (
        durl(Image.fromarray((expected * 255.0).astype(np.uint8)).crop(crop), w=380, jpg=True),
        durl(Image.fromarray((control * 255.0).astype(np.uint8)).crop(crop), w=380, jpg=True),
        durl(Image.fromarray((treatment * 255.0).astype(np.uint8)).crop(crop), w=380, jpg=True),
        durl(Image.fromarray(heat), w=380, jpg=True, q=88),
    )


def static_jitter_evidence(clip, idx):
    """Unmodified source, per-run temporal delta, and a separate static-support mask."""
    prev_idx = sbsbench.predecessor_frame_id(source_files(clip), idx)
    if prev_idx is None:
        return None
    paths = [frame_path(run, clip, i) for run in (ctrl_dir, treat_dir)
             for i in (prev_idx, idx)]
    src_now = source_path(clip, idx)
    src_prev = source_path(clip, prev_idx)
    if not all(paths) or not src_now or not src_prev:
        return None
    images = normalize_sbs_images([Image.open(p).convert("RGB") for p in paths])
    ew, eh = images[0].width // 2, images[0].height
    stable = sbsbench.static_region_mask(
        sbsbench.load_gray(src_now), sbsbench.load_gray(src_prev), ew, eh)
    deltas = []
    for before, now in ((images[0], images[1]), (images[2], images[3])):
        bg = np.asarray(before.convert("L"), np.float32) / 255.0
        ng = np.asarray(now.convert("L"), np.float32) / 255.0
        be, ne = sbsbench.split_eyes(bg), sbsbench.split_eyes(ng)
        deltas.append([np.abs(ne[i] - be[i]) * stable for i in range(2)])
    eye_idx = max(range(2), key=lambda i: float(np.percentile(
        np.abs(deltas[1][i] - deltas[0][i])[stable], 95)) if stable.any() else 0.0)
    signed = (deltas[1][eye_idx] - deltas[0][eye_idx]) * 255.0
    score = sbsbench._box3(np.abs(signed))
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    cw, ch = min(480, ew), min(360, eh)
    x0 = max(0, min(ew - cw, int(cx) - cw // 2))
    y0 = max(0, min(eh - ch, int(cy) - ch // 2))
    source = Image.open(src_now).convert("RGB").resize((ew, eh), Image.BILINEAR)
    ctrl_heat = np.zeros((eh, ew, 3), np.uint8)
    treat_heat = np.zeros((eh, ew, 3), np.uint8)
    ctrl_heat[..., 0] = np.clip(deltas[0][eye_idx] * 255.0 * 8.0, 0, 255).astype(np.uint8)
    treat_heat[..., 0] = np.clip(deltas[1][eye_idx] * 255.0 * 8.0, 0, 255).astype(np.uint8)
    support_count = int(np.count_nonzero(stable))
    support_pct = 100.0 * support_count / max(stable.size, 1)
    support_image = _support_analysis_rgb(stable)
    crop = (x0, y0, x0 + cw, y0 + ch)
    support_crop = _label_analysis_image(Image.fromarray(support_image).crop(crop), [
        "analysis mask (not source content)",
        f"cyan=static support; n={support_count} ({support_pct:.1f}%)",
    ])
    EVIDENCE_METADATA[(clip, idx, "static_jitter_p95")] = {
        "support": support_count, "support_pct": support_pct,
    }
    return (
        durl(source.crop(crop), w=380, jpg=True, q=88),
        durl(Image.fromarray(ctrl_heat).crop(crop), w=380, jpg=True, q=88),
        durl(Image.fromarray(treat_heat).crop(crop), w=380, jpg=True, q=88),
        durl(support_crop, w=380),
    )


def ground_truth_depth_evidence(clip, idx):
    """Ground truth, aligned control/treatment depth, and signed error-delta map."""
    gp = gt_depth_path(clip, idx)
    cp = artifact_path(ctrl_dir, clip, "depth_", idx)
    tp = artifact_path(treat_dir, clip, "depth_", idx)
    if not gp or not cp or not tp:
        return None
    gt, control, treatment = load_depth(gp), load_depth(cp), load_depth(tp)
    kind = CTRL["clips"].get(clip, {}).get("meta", {}).get("gt_depth_kind", "disparity")
    if kind in ("metric", "depth"):
        gt, valid = sbsbench.resize_metric_depth(gt, control.shape[1], control.shape[0])
        target = np.zeros_like(gt)
        target[valid] = 1.0 / gt[valid]
    else:
        gt = sbsbench.resize_depth(gt, control.shape[1], control.shape[0])
        valid = np.isfinite(gt)
        valid &= gt >= 0.0
        target = gt
    if treatment.shape != control.shape:
        treatment = sbsbench.resize_depth(treatment, control.shape[1], control.shape[0])

    ca = sbsbench.align_relative_depth(control, target, valid)[0]
    ta = sbsbench.align_relative_depth(treatment, target, valid)[0]
    lo, hi = np.percentile(target[valid], (1, 99))
    if hi - lo < 1e-4:
        lo, hi = 0.0, 1.0

    def gray(a): return np.clip((a - lo) / (hi - lo), 0, 1)
    signed = np.zeros_like(target)
    signed[valid] = ((np.abs(ta[valid] - target[valid]) - np.abs(ca[valid] - target[valid]))
                     * 255.0 / max(hi - lo, 1e-4))
    heat = np.zeros((*gt.shape, 3), np.uint8)
    heat[..., 0] = np.clip(signed * 5.0, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip(-signed * 5.0, 0, 255).astype(np.uint8)
    shown_gt = np.where(valid, target, lo)
    images = [Image.fromarray((gray(a) * 255).astype(np.uint8)) for a in (shown_gt, ca, ta)]
    images.append(Image.fromarray(heat))
    return tuple(durl(im, w=380, jpg=True, q=88) for im in images)


def ground_truth_lag_evidence(clip, idx):
    """Previous/current GT beside aligned control/treatment depth for lag validation."""
    previous_id = sbsbench.predecessor_frame_id(gt_depth_files(clip), idx)
    if previous_id is None:
        return None
    previous_path, current_path = gt_depth_path(clip, previous_id), gt_depth_path(clip, idx)
    control_path = artifact_path(ctrl_dir, clip, "depth_", idx)
    treatment_path = artifact_path(treat_dir, clip, "depth_", idx)
    if not previous_path or not current_path or not control_path or not treatment_path:
        return None
    control, treatment = load_depth(control_path), load_depth(treatment_path)
    kind = CTRL["clips"].get(clip, {}).get("meta", {}).get("gt_depth_kind", "disparity")

    def target_depth(path):
        ground_truth = load_depth(path)
        if kind in ("metric", "depth"):
            ground_truth, valid = sbsbench.resize_metric_depth(
                ground_truth, control.shape[1], control.shape[0])
            target = np.zeros_like(ground_truth)
            target[valid] = 1.0 / ground_truth[valid]
            return target, valid
        target = sbsbench.resize_depth(ground_truth, control.shape[1], control.shape[0])
        valid = np.isfinite(target) & (target >= 0.0)
        return target, valid

    previous, previous_valid = target_depth(previous_path)
    current, current_valid = target_depth(current_path)
    if treatment.shape != control.shape:
        treatment = sbsbench.resize_depth(treatment, control.shape[1], control.shape[0])
    control_aligned = sbsbench.align_relative_depth(control, current, current_valid)[0]
    treatment_aligned = sbsbench.align_relative_depth(treatment, current, current_valid)[0]
    combined = np.concatenate((previous[previous_valid], current[current_valid]))
    lo, hi = np.percentile(combined, (1, 99))
    if hi - lo < 1e-4:
        lo, hi = 0.0, 1.0

    def image_for(values, valid):
        shown = np.where(valid, values, lo)
        gray = np.clip((shown - lo) / (hi - lo), 0, 1)
        return Image.fromarray((gray * 255).astype(np.uint8))

    images = (image_for(previous, previous_valid), image_for(current, current_valid),
              image_for(control_aligned, current_valid),
              image_for(treatment_aligned, current_valid))
    return tuple(durl(image, w=380, jpg=True, q=88) for image in images)


def flow_temporal_evidence(clip, idx):
    """Source-flow-compensated temporal residual for both runs and signed treatment delta."""
    previous_id = sbsbench.predecessor_frame_id(source_files(clip), idx)
    if previous_id is None:
        return None
    src_now = source_path(clip, idx)
    src_prev = source_path(clip, previous_id)
    paths = [frame_path(ctrl_dir, clip, previous_id), frame_path(ctrl_dir, clip, idx),
             frame_path(treat_dir, clip, previous_id), frame_path(treat_dir, clip, idx)]
    if not src_now or not src_prev or not all(paths):
        return None
    images = [sbsbench.load_gray(p) for p in paths]
    eyes = [sbsbench.split_eyes(a) for a in images]
    eh, ew = eyes[0][0].shape
    scale = min(1.0, 256.0 / ew)
    vw, vh = max(32, round(ew * scale)), max(24, round(eh * scale))
    now_src = sbsbench.load_gray(src_now)
    prev_src = sbsbench.load_gray(src_prev)
    gt_flow = gt_flow_files(clip).get(idx)
    if not gt_flow:
        return None
    with np.load(gt_flow, allow_pickle=False) as flow_data:
        reference_flow = np.asarray(flow_data["flow"], dtype=np.float32)
        reference_valid = (np.asarray(flow_data["valid"], dtype=bool)
                           if "valid" in flow_data else None)
    u, v, flow_valid = sbsbench.resize_forward_flow_to_current(
        reference_flow, reference_valid, vw, vh)
    now_small = sbsbench.resize_to(now_src, vw, vh)
    prev_small = sbsbench.resize_to(prev_src, vw, vh)
    src_warp, valid = sbsbench.warp_previous_with_flow(prev_small, u, v)
    reliable = flow_valid & valid & (np.abs(now_small - src_warp) <= 10.0 / 255.0)
    reliable &= ~sbsbench.hdilate(~reliable, 1)
    deltas = []
    for run_eyes in (eyes[:2], eyes[2:]):
        per_eye = []
        for eye_idx in range(2):
            before = sbsbench.resize_to(run_eyes[0][eye_idx], vw, vh)
            now = sbsbench.resize_to(run_eyes[1][eye_idx], vw, vh)
            warped, ok = sbsbench.warp_previous_with_flow(before, u, v)
            per_eye.append(np.abs(now - warped) * reliable * ok)
        deltas.append(per_eye)
    eye_idx = max(range(2), key=lambda i: float(np.percentile(
        np.abs(deltas[1][i] - deltas[0][i])[reliable], 95)) if reliable.any() else 0.0)
    signed = (deltas[1][eye_idx] - deltas[0][eye_idx]) * 255.0
    score = sbsbench._box3(np.abs(signed))
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    cw, ch = min(220, vw), min(170, vh)
    x0, y0 = max(0, min(vw - cw, int(cx) - cw // 2)), max(0, min(vh - ch, int(cy) - ch // 2))
    source_rgb = np.repeat((now_small[..., None] * 255.0).astype(np.uint8), 3, axis=2)
    source_rgb[~reliable] = (source_rgb[~reliable] * 0.18).astype(np.uint8)
    ctrl_heat = np.zeros((vh, vw, 3), np.uint8)
    treat_heat = np.zeros((vh, vw, 3), np.uint8)
    ctrl_heat[..., 0] = np.clip(deltas[0][eye_idx] * 255.0 * 8.0, 0, 255).astype(np.uint8)
    treat_heat[..., 0] = np.clip(deltas[1][eye_idx] * 255.0 * 8.0, 0, 255).astype(np.uint8)
    signed_heat = np.zeros((vh, vw, 3), np.uint8)
    signed_heat[..., 0] = np.clip(signed * 8.0, 0, 255).astype(np.uint8)
    signed_heat[..., 2] = np.clip(-signed * 8.0, 0, 255).astype(np.uint8)
    crop = (x0, y0, x0 + cw, y0 + ch)
    return tuple(durl(Image.fromarray(a).crop(crop), w=380, jpg=True, q=88)
                 for a in (source_rgb, ctrl_heat, treat_heat, signed_heat))


def visual_evidence_images(clip, idx, metric=None):
    """Matched control/treatment crops plus an amplified RGB difference heatmap.

    The crop is selected from the shared control depth, so both modes show exactly the same
    source region.  The heatmap is deliberately labelled as amplified: it is evidence of where
    the renderers differ, while the adjacent metric supplies the direction of the change.
    """
    if metric in ("source_coverage_pct", "image_integrity_pct",
                  "exact_mapping_stretch_pct", "exact_mapping_fold_pct",
                  ):
        return source_residual_evidence(clip, idx, metric)
    if metric == "warp_cross_row_shear_severity_pct":
        return cross_row_shear_evidence(clip, idx, metric)
    if metric == "interocular_phase_orientation_burden_pct":
        return interocular_conflict_evidence(clip, idx, metric)
    if metric in {
            "interocular_exposure_rivalry_burden_pct",
            "interocular_color_gain_rivalry_burden_pct"}:
        return interocular_photometric_evidence(clip, idx, metric)
    if metric == "experimental_stereo_window_crossed_burden_pct":
        return stereo_window_evidence(clip, idx, metric)
    if metric == "static_jitter_p95":
        return static_jitter_evidence(clip, idx)
    if metric == "flow_temporal_p95":
        return flow_temporal_evidence(clip, idx)
    if metric in ("depth_gt_affine_nrmse_pct", "depth_gt_edge_f1",
                  "depth_gt_polarity_ok"):
        return ground_truth_depth_evidence(clip, idx)
    if metric == "depth_gt_lag_f1_p95":
        return ground_truth_lag_evidence(clip, idx)
    if metric in EXACT_STEREO_VISUAL_METRICS:
        cp, tp = frame_path(ctrl_dir, clip, idx), frame_path(treat_dir, clip, idx)
        dp = artifact_path(ctrl_dir, clip, "depth_", idx)
        if not (cp and tp and dp):
            return None
        depth = load_depth(dp)
        ctrl, treat = normalize_sbs_images(
            [Image.open(cp).convert("RGB"), Image.open(tp).convert("RGB")])
        ew, eh = ctrl.width // 2, ctrl.height
        gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
        dh, dw = depth.shape
        band = gx[int(dh * 0.15):int(dh * 0.85)]
        colscore = band.sum(0)
        lo, hi = int(dw * 0.1), int(dw * 0.9)
        cx_d = int(np.argmax(colscore[lo:hi]) + lo) if colscore[lo:hi].max() > 0.1 else dw // 2
        rowscore = gx[:, max(0, cx_d - 2):cx_d + 3].sum(1)
        cy_d = int(np.argmax(rowscore)) if rowscore.max() > 0 else dh // 2
        cx, cy = int(cx_d / dw * ew), int(cy_d / dh * eh)
        cw, ch = min(300, ew), min(300, eh)
        x0 = max(0, min(ew - cw, cx - cw // 2))
        y0 = max(0, min(eh - ch, cy - ch // 2))

        def stereo_pair(image):
            left = image.crop((x0, y0, x0 + cw, y0 + ch))
            right = image.crop((ew + x0, y0, ew + x0 + cw, y0 + ch))
            pair = Image.new("RGB", (cw * 2 + 4, ch), (18, 24, 29))
            pair.paste(left, (0, 0))
            pair.paste(right, (cw + 4, 0))
            return pair

        ctrl_pair, treat_pair = stereo_pair(ctrl), stereo_pair(treat)
        a, b = np.asarray(ctrl_pair, np.float32), np.asarray(treat_pair, np.float32)
        delta = np.mean(np.abs(b - a), axis=2)
        v = np.clip(delta * 5.0, 0, 255)
        heat = np.zeros((*v.shape, 3), np.uint8)
        heat[..., 0] = v.astype(np.uint8)
        heat[..., 1] = np.clip((v - 64) * 1.7, 0, 255).astype(np.uint8)
        return (durl(ctrl_pair, w=520, jpg=True, q=86), durl(treat_pair, w=520, jpg=True, q=86),
                durl(Image.fromarray(heat), w=520, jpg=True, q=82))
    pair = crop_at_silhouette(clip, idx)
    if not pair:
        return None
    cp, tp = frame_path(ctrl_dir, clip, idx), frame_path(treat_dir, clip, idx)
    dp = artifact_path(ctrl_dir, clip, "depth_", idx)
    depth = load_depth(dp)
    ctrl, treat = normalize_sbs_images(
        [Image.open(cp).convert("RGB"), Image.open(tp).convert("RGB")])
    ew, eh = ctrl.width // 2, ctrl.height
    gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    dh, dw = depth.shape
    band = gx[int(dh * 0.15):int(dh * 0.85)]
    score = band.sum(0)
    lo, hi = int(dw * 0.1), int(dw * 0.9)
    cx_d = int(np.argmax(score[lo:hi]) + lo) if score[lo:hi].max() > 0.1 else dw // 2
    rows = gx[:, max(0, cx_d - 2):cx_d + 3].sum(1)
    cy_d = int(np.argmax(rows)) if rows.max() > 0 else dh // 2
    cx, cy = int(cx_d / dw * ew), int(cy_d / dh * eh)
    cw, ch = min(480, ew), min(360, eh)
    x0 = max(0, min(ew - cw, cx - cw // 2))
    y0 = max(0, min(eh - ch, cy - ch // 2))
    a = np.asarray(ctrl.crop((x0, y0, x0 + cw, y0 + ch)), np.float32)
    b = np.asarray(treat.crop((x0, y0, x0 + cw, y0 + ch)), np.float32)
    delta = np.mean(np.abs(b - a), axis=2)
    # Black means unchanged. Red/yellow/white means progressively larger RGB disagreement.
    v = np.clip(delta * 5.0, 0, 255)
    heat = np.zeros((*v.shape, 3), np.uint8)
    heat[..., 0] = v.astype(np.uint8)
    heat[..., 1] = np.clip((v - 64) * 1.7, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip((v - 160) * 2.7, 0, 255).astype(np.uint8)
    return pair[0], pair[1], durl(Image.fromarray(heat), w=380, jpg=True, q=82)


def run_label(run, run_dir, default):
    """Human name for a run. The run identity must survive identical harness arguments."""
    stored = run.get("meta", {}).get("run_name")
    if stored:
        return stored
    dirname = os.path.basename(os.path.normpath(run_dir))
    if dirname:
        return dirname
    ex = run["meta"].get("extra_args") or []
    if ex:
        return " ".join(ex).replace("--", "")
    mode = run["meta"].get("mode", "")
    models = sorted({e["meta"].get("model", "") for e in run["clips"].values()})
    if mode:
        return f"{mode}" + (f" ({models[0]})" if len(models) == 1 and models[0] else "")
    return default


CTRL_MODE = CTRL["meta"].get("mode")
TREAT_MODE = TREAT["meta"].get("mode")
IS_MODE_CMP = bool(CTRL_MODE and TREAT_MODE and CTRL_MODE != TREAT_MODE)
CTRL_PROFILES = {e.get("meta", {}).get("profile") for e in CTRL["clips"].values()
                 if e.get("meta", {}).get("profile")}
TREAT_PROFILES = {e.get("meta", {}).get("profile") for e in TREAT["clips"].values()
                  if e.get("meta", {}).get("profile")}
IS_PROFILE_CMP = bool(CTRL_PROFILES and TREAT_PROFILES and CTRL_PROFILES != TREAT_PROFILES)
IS_COMPARISON_ONLY = (TREAT.get("meta", {}).get("run_kind") == "comparison-only" or
                      TREAT.get("verdict") == "comparison_only")
IS_TRADEOFF_CMP = IS_MODE_CMP or IS_PROFILE_CMP
CTRL_NAME = run_label(CTRL, ctrl_dir, "control")
TREAT_NAME = run_label(TREAT, treat_dir, "treatment")
# Short tags for inline value labels and image captions (arrow is always CTRL -> TREAT).
CTRL_TAG = (CTRL_MODE if IS_MODE_CMP else
            next(iter(CTRL_PROFILES)) if IS_PROFILE_CMP and len(CTRL_PROFILES) == 1 else "control")
TREAT_TAG = (TREAT_MODE if IS_MODE_CMP else
             next(iter(TREAT_PROFILES)) if IS_PROFILE_CMP and len(TREAT_PROFILES) == 1 else
             "treatment")


def treatment_name():
    return TREAT_NAME


ctrl_agg = {c: CTRL["clips"][c]["aggregate"] for c in CLIPS}
treat_agg = {c: TREAT["clips"][c]["aggregate"] for c in CLIPS}


def _metric_state_value(run, clip, key):
    """Return the authenticated evidence state and optional finite aggregate value.

    ``unsupported`` is a legitimate abstention, while ``missing`` is a fail-closed evaluator
    defect. Presentation code must retain that distinction instead of turning both into ``n/a``.
    """
    observed = run["clips"][clip].get("aggregate", {})
    spec = THR.get(key)
    state = "applicable"
    if spec is not None:
        metadata = run["clips"][clip].get("meta", {})
        state = sbsbench.metric_evidence_state(key, spec, observed, metadata)
        if state != "applicable":
            return state, None
    if not sbsbench.metric_value_valid(observed, key):
        return "missing", None
    return state, float(observed[key])


def _metric_value(run, clip, key):
    """Finite aggregate value only when independent evidence is applicable."""
    return _metric_state_value(run, clip, key)[1]


def expected_flat(run, clip):
    value = run["clips"].get(clip, {}).get("meta", {}).get("expected_flat")
    if value is not None:
        return bool(value)
    mp = os.path.join(CLIPS_ROOT, clip, "meta.json")
    try:
        return bool(json.load(open(mp)).get("expected_flat"))
    except Exception:
        return False


# Expected-flat clips remain visible as false-stereo diagnostics but cannot raise or lower the
# general-content feature verdict. They exercise a different objective from ordinary scenes.
DECISION_CLIPS = [c for c in CLIPS if not expected_flat(CTRL, c)] or CLIPS
DECISION_SCOPE = ("extended_screening" if TREAT.get("meta", {}).get("suite") == "extended"
                  else "core_screening")
SOURCE_ARTIFACT_CLIPS = [c for c in CLIPS
                         if CTRL["clips"].get(c, {}).get("meta", {}).get("source_artifacts")]


# Compute the verdict once from the aggregate dictionaries and reuse the same object for both the
# HTML conclusion and a machine-readable sidecar. This prevents downstream automation (or a human
# ad-hoc script) from accidentally passing the per-clip wrapper instead of its `aggregate` member.
AB_METRIC_DECISION = sbsbench.evaluate_ab_decision(
    ctrl_agg, treat_agg, DECISION_CLIPS, THR, hard_clip_ids=CLIPS,
    clip_meta={clip: TREAT["clips"][clip].get("meta", {}) for clip in CLIPS})
AB_DECISION = sbsbench.gate_ab_decision(AB_METRIC_DECISION, CTRL, TREAT)
colmax = {
    key: max((abs(value) for clip in CLIPS for value in
              (_metric_value(CTRL, clip, key), _metric_value(TREAT, clip, key))
              if value is not None), default=0.0)
    for key, *_ in COLS
}
ACTIVE = [col for col in COLS if col[3] or colmax[col[0]] > col[4]]
CLEAN = [col for col in COLS if col not in ACTIVE and col[2]]


# The dashboard visualizes the exact gate inputs instead of projecting unrelated metrics onto a
# shared radar scale. Primary/style dumbbells normalize only by configured noise; hard bullets
# retain raw engineering bounds; the heatmap exposes the per-clip events behind the summary.
PRIMARY_STYLE_AXES = (
    ("exact_visible_pop_spread_pct", "Visible stereo relief", "style objective", "%"),
    ("depth_gt_affine_nrmse_pct", "GT depth accuracy", "primary · depth", "%"),
    ("depth_gt_edge_f1", "GT depth boundaries", "primary · depth edge", "%"),
    ("static_jitter_p95", "Static stability", "primary · stability", " luma"),
    ("depth_gt_lag_f1_p95", "GT depth timing", "primary · depth stability", " F1"),
)

HARD_DISPLAY = (
    ("exact_binocular_support_pct", "Exact binocular support", "%"),
    ("exact_positive_disparity_pct", "Positive disparity tail", "%"),
    ("exact_negative_disparity_pct", "Negative disparity tail", "%"),
    ("exact_polarity_ok", "Depth/disparity polarity", "%"),
    ("exact_symmetry_residual_p95_pct", "Camera symmetry residual", "%"),
    ("vmisalign_p99_pct", "Vertical alignment P99", "%"),
    ("source_coverage_pct", "Source coverage", "%"),
    ("image_integrity_pct", "Image integrity", "%"),
    ("source_coverage_worst_patch_bad_pct", "Worst localized missing patch", "%"),
    ("image_integrity_worst_patch_bad_pct", "Worst localized texture damage", "%"),
    ("depth_gt_polarity_ok", "Authenticated GT depth polarity", "%"),
)

HARD_SUPPORT_KEYS = {
    "exact_binocular_support_pct": "exact_binocular_support_count",
    "exact_positive_disparity_pct": "exact_binocular_support_count",
    "exact_negative_disparity_pct": "exact_binocular_support_count",
    "exact_polarity_ok": "exact_polarity_support_pct",
    "exact_symmetry_residual_p95_pct": "exact_binocular_support_count",
    "vmisalign_p99_pct": "vmisalign_support_pct",
    "source_coverage_pct": "source_fidelity_support_pct",
    "image_integrity_pct": "image_integrity_support",
    "source_coverage_worst_patch_bad_pct": "source_fidelity_support_pct",
    "image_integrity_worst_patch_bad_pct": "image_integrity_support",
}

SUPPORTING_HEATMAP_AXES = (
    ("exact_mapping_stretch_pct", "stretch"),
    ("exact_mapping_fold_pct", "fold"),
    ("exact_local_polarity_component_pct", "local polarity"),
    ("warp_cross_row_shear_severity_pct", "row shear"),
    ("exact_over_3pct_area_pct", "over-limit"),
    ("experimental_stereo_window_crossed_burden_pct", "window"),
    ("interocular_phase_orientation_burden_pct", "phase"),
    ("interocular_exposure_rivalry_burden_pct", "exposure"),
    ("interocular_color_gain_rivalry_burden_pct", "colour"),
    ("flow_temporal_p95", "flow temporal"),
)


def _paired_aggregate(key, clips=DECISION_CLIPS):
    """Return matched raw aggregate pairs and keep unsupported evidence out of the mean."""
    pairs = [(_metric_value(CTRL, clip, key), _metric_value(TREAT, clip, key))
             for clip in clips]
    return [(control, treatment) for control, treatment in pairs
            if control is not None and treatment is not None]


def _metric_missing_clips(key, clips=DECISION_CLIPS):
    """Clips where either side is missing required evidence, not legitimately unsupported."""
    return [clip for clip in clips
            if "missing" in (_metric_state_value(CTRL, clip, key)[0],
                             _metric_state_value(TREAT, clip, key)[0])]


def _metric_reportable(key, clips=CLIPS):
    """Keep axes with a matched sample or a fail-closed missing-evidence event."""
    return bool(_paired_aggregate(key, clips) or _metric_missing_clips(key, clips))


def _paired_mean_aggregate(key, clips=DECISION_CLIPS):
    """Return control/treatment means over identical clips with authenticated evidence."""
    pairs = _paired_aggregate(key, clips)
    if not pairs:
        return None, None
    return (float(np.mean([pair[0] for pair in pairs])),
            float(np.mean([pair[1] for pair in pairs])))


def _mean_perf(run, key):
    values = [run["clips"][clip].get("perf_ms", {}).get(key) for clip in CLIPS]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _metric_noise(key, control):
    spec = THR[key]
    return max(float(spec.get("abs_floor", 0.0)),
               abs(float(control)) * float(spec.get("rel_tol", 0.0)), 1e-9)


def _display_value(value, unit=""):
    return "n/a" if value is None else f"{value:.2f}{unit}"


def _dumbbell_rows():
    rows = []
    for key, label, role, unit in PRIMARY_STYLE_AXES:
        pairs = _paired_aggregate(key)
        missing = _metric_missing_clips(key)
        if missing:
            rows.append(f'<div class="dumbbell-row is-bad"><div class="dumbbell-copy">'
                        f'<b>{html.escape(label)}</b><small>{html.escape(role)} · '
                        f'missing required evidence on {len(missing)} clip(s); partial means are '
                        f'not shown</small></div><div class="dumbbell-na">missing</div></div>')
            continue
        if not pairs:
            rows.append(f'<div class="dumbbell-row is-unsupported"><div class="dumbbell-copy">'
                        f'<b>{html.escape(label)}</b><small>{html.escape(role)} · no authenticated '
                        f'matched evidence</small></div><div class="dumbbell-na">unsupported</div></div>')
            continue
        control = float(np.mean([pair[0] for pair in pairs]))
        treatment = float(np.mean([pair[1] for pair in pairs]))
        spec = THR[key]
        noise = _metric_noise(key, control)
        signed_delta = treatment - control
        improvement_delta = signed_delta if spec.get("better") == "higher" else -signed_delta
        effect = improvement_delta / noise
        treatment_pos = 50.0 + 15.0 * max(-3.0, min(3.0, effect))
        line_left = min(50.0, treatment_pos)
        line_width = max(abs(treatment_pos - 50.0), 0.35)
        movement = sbsbench.metric_delta_class(control, treatment, spec)
        movement_class = {"improved": "is-good", "regressed": "is-bad"}.get(
            movement, "is-noise")
        direction = "higher is better" if spec.get("better") == "higher" else "lower is better"
        effect_text = ("within noise" if movement == "noise" else
                       f'{abs(effect):.1f}× noise toward '
                       f'{"treatment" if movement == "improved" else "control"}')
        tooltip = html.escape(
            f'{label}: {CTRL_TAG} {control:.4f} to {TREAT_TAG} {treatment:.4f}; '
            f'noise tolerance {noise:.4f}; {direction}', quote=True)
        rows.append(
            f'<div class="dumbbell-row {movement_class}" title="{tooltip}">'
            f'<div class="dumbbell-copy"><b>{html.escape(label)}</b>'
            f'<small>{html.escape(role)} · {direction} · n={len(pairs)}/{len(DECISION_CLIPS)}'
            f'</small></div>'
            f'<div class="dumbbell-plot"><div class="dumbbell-scale">'
            f'<span class="noise-zone"></span><i class="dumbbell-line" '
            f'style="left:{line_left:.1f}%;width:{line_width:.1f}%"></i>'
            f'<i class="dumbbell-dot dot-control" style="left:50%"></i>'
            f'<i class="dumbbell-dot dot-treatment" style="left:{treatment_pos:.1f}%"></i>'
            f'</div><div class="dumbbell-direction"><span>control-favoured</span>'
            f'<span>± noise</span><span>treatment-favoured</span></div></div>'
            f'<div class="dumbbell-values"><code>{_display_value(control, unit)}</code>'
            f'<span>→</span><code>{_display_value(treatment, unit)}</code>'
            f'<small class="{movement_class}">{effect_text} · ±{noise:.2f}{unit}</small></div></div>')
    return "".join(rows)


def _runtime_strip():
    rows = []
    for key, label in (("depth_infer", "depth"), ("warp_infer", "warp"),
                       ("sbs_composite_cpu", "CPU composite")):
        control, treatment = _mean_perf(CTRL, key), _mean_perf(TREAT, key)
        if control is None and treatment is None:
            continue
        rows.append(f'<span><b>{html.escape(label)}</b> '
                    f'<code>{_display_value(control, " ms")}</code> → '
                    f'<code>{_display_value(treatment, " ms")}</code></span>')
    return (f'<div class="runtime-strip"><small>Runtime context · not a quality vote</small>'
            f'{"".join(rows)}</div>' if rows else "")


def _hard_worst(run, key, spec):
    states = {clip: _metric_state_value(run, clip, key) for clip in CLIPS}
    missing = [clip for clip, (state, _value) in states.items() if state == "missing"]
    if missing:
        return None, missing[0], "missing"
    applicable = {clip: {key: value} for clip, (state, value) in states.items()
                  if state == "applicable" and value is not None}
    value, clip = sbsbench.worst_hard_metric(applicable, key, spec, CLIPS)
    return value, clip, ("applicable" if value is not None else "unsupported")


def _hard_support(run, clip, key):
    support_key = HARD_SUPPORT_KEYS.get(key)
    if not clip or not support_key:
        return None
    value = run["clips"].get(clip, {}).get("aggregate", {}).get(support_key)
    if not sbsbench.metric_value_valid({support_key: value}, support_key):
        return None
    value = float(value)
    percent_support = support_key.endswith("_pct") or support_key == "image_integrity_support"
    return (f'{value:.1f}%' if percent_support else
            f'n={int(round(value)):,}')


def _hard_state(value, spec, evidence_state):
    if evidence_state == "missing":
        return "hard-missing"
    if evidence_state == "unsupported":
        return "hard-unsupported"
    failed = (("hard_min" in spec and value < spec["hard_min"])
              or ("hard_max" in spec and value > spec["hard_max"]))
    return "hard-fail" if failed else "hard-pass"


def _hard_bullet_rows():
    rows = []
    configured_hard = {key for key, spec in THR.items() if spec.get("role") == "hard"}
    displayed = [entry for entry in HARD_DISPLAY if entry[0] in configured_hard]
    displayed.extend((key, SHORT.get(key, key), "")
                     for key in sorted(configured_hard - {entry[0] for entry in displayed}))
    for key, label, unit in displayed:
        spec = THR[key]
        control, control_clip, control_state = _hard_worst(CTRL, key, spec)
        treatment, treatment_clip, treatment_state = _hard_worst(TREAT, key, spec)
        values = [value for value in (control, treatment) if value is not None]
        if "hard_max" in spec:
            bound_value = float(spec["hard_max"])
            scale_max = max([bound_value * 1.25, 1e-6] + [value * 1.05 for value in values])
            bound_pos = 100.0 * bound_value / scale_max
            track_class = "limit-max"
            bound = f'≤ {bound_value:.2f}{unit}'
        else:
            bound_value = float(spec["hard_min"])
            scale_max = max([100.0, bound_value, 1e-6] + values)
            bound_pos = 100.0 * bound_value / scale_max
            track_class = "limit-min"
            bound = f'≥ {bound_value:.2f}{unit}'

        def position(value):
            return 0.0 if value is None else max(0.0, min(100.0, 100.0 * value / scale_max))

        markers = ""
        if control is not None:
            markers += f'<i class="hard-dot dot-control" style="left:{position(control):.1f}%"></i>'
        if treatment is not None:
            markers += f'<i class="hard-dot dot-treatment" style="left:{position(treatment):.1f}%"></i>'
        control_support = _hard_support(CTRL, control_clip, key)
        treatment_support = _hard_support(TREAT, treatment_clip, key)
        support = ""
        if control_support or treatment_support:
            support = (f' · support {html.escape(CTRL_TAG)} {control_support or "n/a"}; '
                       f'{html.escape(TREAT_TAG)} {treatment_support or "n/a"}')
        def location(side_state, clip):
            if side_state == "missing":
                return f'missing evidence at {html.escape(name(clip))}'
            if side_state == "unsupported":
                return "unsupported"
            return html.escape(name(clip))

        locations = (f'worst clip {html.escape(CTRL_TAG)} '
                     f'{location(control_state, control_clip)}; {html.escape(TREAT_TAG)} '
                     f'{location(treatment_state, treatment_clip)}')
        rows.append(
            f'<div class="hard-bullet-row"><div class="hard-bullet-copy">'
            f'<b>{html.escape(label)}</b><small>{bound} · {locations}{support}</small></div>'
            f'<div class="hard-bullet-plot"><div class="hard-track {track_class}" '
            f'style="--bound:{bound_pos:.1f}%"><i class="hard-bound"></i>{markers}</div></div>'
            f'<div class="hard-bullet-values">'
            f'<code class="{_hard_state(control, spec, control_state)}">'
            f'{"missing" if control_state == "missing" else _display_value(control, unit)}</code>'
            f'<span>→</span>'
            f'<code class="{_hard_state(treatment, spec, treatment_state)}">'
            f'{"missing" if treatment_state == "missing" else _display_value(treatment, unit)}</code>'
            f'</div></div>')
    return "".join(rows), len(displayed)


def _heatmap_cell(clip, key):
    control_state, control = _metric_state_value(CTRL, clip, key)
    treatment_state, treatment = _metric_state_value(TREAT, clip, key)
    label = SHORT.get(key, key)
    if "missing" in (control_state, treatment_state):
        missing_sides = "/".join(
            tag for tag, state in ((CTRL_TAG, control_state), (TREAT_TAG, treatment_state))
            if state == "missing")
        title = html.escape(
            f'{name(clip)} · {label}: required evidence missing for {missing_sides}',
            quote=True)
        return f'<td class="heat-missing" title="{title}">missing</td>'
    if control is None or treatment is None:
        title = html.escape(f'{name(clip)} · {label}: unsupported authenticated evidence', quote=True)
        return f'<td class="heat-unsupported" title="{title}">n/a</td>'
    spec = THR[key]
    noise = _metric_noise(key, control)
    signed_delta = treatment - control
    improvement_delta = signed_delta if spec.get("better") == "higher" else -signed_delta
    effect = improvement_delta / noise
    movement = sbsbench.metric_delta_class(control, treatment, spec)
    if movement == "noise":
        cell_class, cell_text = "heat-noise", f'{abs(effect):.1f}×'
    else:
        strength = ("heat-strong" if abs(effect) >= 4.0 else
                    "heat-mid" if abs(effect) >= 2.0 else "heat-soft")
        cell_class = f'{"heat-good" if movement == "improved" else "heat-bad"} {strength}'
        cell_text = f'{"+" if movement == "improved" else "−"}{abs(effect):.1f}×'
    title = html.escape(
        f'{name(clip)} · {label}: {CTRL_TAG} {control:.4f} to {TREAT_TAG} {treatment:.4f}; '
        f'{movement}; {abs(effect):.2f} times configured noise', quote=True)
    return f'<td class="{cell_class}" title="{title}">{cell_text}</td>'


def _heatmap():
    primary = tuple((key, label) for key, label, _, _ in PRIMARY_STYLE_AXES
                    if _metric_reportable(key, CLIPS))
    supporting = tuple((key, label) for key, label in SUPPORTING_HEATMAP_AXES
                       if _metric_reportable(key, CLIPS))
    axes = primary + supporting
    if not axes:
        return '<p class="sub">No matched authenticated metric evidence is available.</p>'
    headers = "".join(f'<th title="{html.escape(label, quote=True)}">{html.escape(label)}</th>'
                      for _, label in axes)
    group_headers = ""
    if primary:
        group_headers += f'<th colspan="{len(primary)}">primary / style</th>'
    if supporting:
        group_headers += f'<th colspan="{len(supporting)}">supporting diagnostics</th>'
    rows = []
    for clip in CLIPS:
        cells = "".join(_heatmap_cell(clip, key) for key, _ in axes)
        rows.append(f'<tr><th class="heat-clip" title="{html.escape(clip, quote=True)}">'
                    f'{html.escape(name(clip))}</th>'
                    f'{cells}</tr>')
    return (f'<div class="heatmap-wrap"><table class="heatmap"><thead>'
            f'<tr class="heat-groups"><th class="heat-clip" rowspan="2">clip</th>'
            f'{group_headers}</tr>'
            f'<tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            f'<div class="heat-legend"><span class="heat-good heat-soft">+</span> treatment-favoured '
            f'<span class="heat-bad heat-soft">−</span> control-favoured '
            f'<span class="heat-noise">≈</span> within noise '
            f'<span class="heat-unsupported">n/a</span> unsupported '
            f'<span class="heat-missing">missing</span> fail-closed evidence · '
            f'number = multiples of noise</div>')


def grouped_quality_section():
    hard_rows, hard_count = _hard_bullet_rows()
    axes_card = (f'<article class="decision-card decision-axes"><div class="decision-head">'
                 f'<h3>Primary and style movement</h3><span>matched means · tolerance-normalized</span>'
                 f'</div><p>Control is fixed at the centre. Treatment moves right when favoured and '
                 f'left when control is favoured; the shaded centre is the configured noise band.</p>'
                 f'<div class="dumbbell-list">{_dumbbell_rows()}</div>{_runtime_strip()}</article>')
    hard_card = (f'<article class="decision-card decision-hard"><div class="decision-head">'
                 f'<h3>Hard gate · all {hard_count} constraints</h3><span>worst raw clip value</span>'
                 f'</div><p>Each dot is the safety-worst per-clip aggregate. Every row must pass '
                 f'independently; the vertical tick is the engineering limit.</p>'
                 f'<div class="hard-bullet-list">{hard_rows}</div></article>')
    heat_card = (f'<article class="decision-card decision-heat"><div class="decision-head">'
                 f'<h3>Where the movement occurs</h3><span>clip × metric event map</span></div>'
                 f'<p>Colour follows each metric\'s configured direction and tolerance. Unsupported '
                 f'cells are hatched; missing required evidence is red and fail-closed.</p>'
                 f'{_heatmap()}</article>')
    return (f'<section><h2>Metrics by group</h2><p class="sub">These three views mirror the '
            f'evaluator contract: coequal primary/style movement, fail-closed hard limits, and '
            f'per-clip evidence. They summarize but never replace the canonical per-clip gate.</p>'
            f'<div class="decision-grid">{axes_card}{hard_card}{heat_card}</div></section>')


def scorecard_charts():
    """Grouped horizontal bars retain every table value while making A/B movement scannable."""
    charts = []
    for metric, label, worse, _, _ in ACTIVE:
        values = [(c, _metric_value(CTRL, c, metric),
                   _metric_value(TREAT, c, metric)) for c in CLIPS]
        numeric = [abs(v) for _, a, b in values for v in (a, b) if v is not None]
        scale = max(numeric, default=1.0) or 1.0
        rows = []
        for c, a, b in values:
            if a is None or b is None:
                control_state = _metric_state_value(CTRL, c, metric)[0]
                treatment_state = _metric_state_value(TREAT, c, metric)[0]
                missing = "missing" in (control_state, treatment_state)
                state_label = "missing required evidence" if missing else "not applicable"
                state_class = "bar-missing" if missing else "bar-na"
                delta_class = "bar-bad" if missing else "bar-flat"
                delta_label = "missing" if missing else "n/a"
                rows.append(f'<div class="bar-row"><div class="bar-scene" title="{c}">{name(c)}</div>'
                            f'<div class="bar-pair"><span class="{state_class}">'
                            f'{state_label}</span></div>'
                            f'<span class="bar-delta {delta_class}">{delta_label}</span></div>')
                continue
            aw = max(0.8, abs(a) / scale * 100.0) if a else 0.0
            bw = max(0.8, abs(b) / scale * 100.0) if b else 0.0
            delta = b - a
            spec = THR[metric]
            movement = sbsbench.metric_delta_class(a, b, spec)
            is_vote = (spec.get("role") == "primary"
                       and spec.get("scope") != "conformance")
            move_cls = ("bar-flat" if not is_vote or movement == "noise" else
                        "bar-good" if movement == "improved" else "bar-bad")
            pct = delta / a * 100.0 if a else (100.0 if b else 0.0)
            delta_text = (f'{spec.get("role", "diagnostic")} {pct:+.0f}%'
                          if not is_vote else "within noise" if movement == "noise" else
                          f'{"better" if movement == "improved" else "worse"} {abs(pct):.0f}%')
            rows.append(
                f'<div class="bar-row"><div class="bar-scene" title="{c}">{name(c)}</div>'
                f'<div class="bar-pair"><div class="bar-line"><span class="bar-tag">{CTRL_TAG}</span>'
                f'<span class="bar-track"><i class="bar-fill bar-control" style="width:{aw:.1f}%"></i></span>'
                f'<b>{a:.2f}</b></div><div class="bar-line"><span class="bar-tag">{TREAT_TAG}</span>'
                f'<span class="bar-track"><i class="bar-fill {move_cls}" style="width:{bw:.1f}%"></i></span>'
                f'<b>{b:.2f}</b></div></div><span class="bar-delta {move_cls}">{delta_text}</span></div>')
        spec = THR[metric]
        role_note = ("diagnostic only · " if spec.get("role") == "diagnostic" else
                     "hard bound · " if spec.get("role") == "hard" else "")
        direction = role_note + ("lower is better" if worse else "higher is better")
        charts.append(f'<article class="metric-chart"><div class="chart-head">'
                      f'<h3>{mtip(metric, label)}</h3><span>{direction}</span></div>{"".join(rows)}</article>')
    return '<div class="chart-grid">' + "".join(charts) + '</div>'


# metric -> (short header, what it measures, direction). Only the ones that appear render.
METRIC_DEFS = [
    ("exact_visible_pop_spread_pct",
         "visible_pop",
         "Source-structure-weighted visible relief from the exact production warp map. Unlike raw requested disparity, this estimates stereo volume carried by image structure a viewer can actually fuse; its support is reported separately.",
         "higher = more visible stereo relief"),
    ("exact_binocular_support_pct",
         "binocular_support",
         "Common rendered area backed by source samples uniquely visible in both exact production eye maps, limited by the smaller per-eye Jacobian. This independent floor prevents a collapsed or mostly non-overlapping render from hiding unsafe disparity tails.",
         "must remain above the evidence floor"),
    ("exact_positive_disparity_pct",
         "disp_positive",
         "Output-area-weighted p99.9 positive tail of actual x_right - x_left after independently inverting both exact eye maps onto common, uniquely visible source samples.",
         "must stay below comfort limit"),
    ("exact_negative_disparity_pct",
         "disp_negative",
         "Magnitude of the output-area-weighted p0.1 negative tail of actual x_right - x_left on the same mutually valid binocular support.",
         "must stay below comfort limit"),
    ("exact_polarity_ok",
         "depth_polarity",
         "Hard high-near depth-to-disparity polarity contract from exact shader coordinates. Flat depth reports unsupported evidence instead of inventing a fit.",
         "must remain 100% when supported"),
    ("exact_local_polarity_component_pct",
         "local_polarity_component",
         "Largest connected output region participating in a supported local depth/disparity-order reversal, as a percentage of content. This separates isolated noise from a salient contiguous inversion.",
         "lower = smaller coherent reversal"),
    ("exact_over_3pct_area_pct",
         "comfort_over_3_area",
         "Exact content area beyond the current 3% heuristic comfort boundary. Area complements the p0.1/p99.9 hard tail by revealing how much of the image carries the violation.",
         "lower = less over-limit burden"),
    ("source_coverage_pct",
         "coverage",
         "Worst-eye pixels matching the exact source sample selected by the production shader. This is a renderer-conformance check with no candidate-chosen correspondence search; it does not judge whether the selected geometry looks perceptually correct.",
         "renderer conformance; must remain above limit"),
    ("image_integrity_pct",
         "integrity",
         "Worst-eye retention of exact mapped-source texture, with lower and upper gradient bounds. It detects renderer-side blur or ringing relative to the requested sample, not geometry-induced perceptual artifacts already present in that sample.",
         "renderer conformance; must remain above limit"),
    ("source_coverage_worst_patch_bad_pct",
         "localized_missing",
         "Worst resolution-scaled local patch fraction whose rendered samples no longer match the exact production-selected source. Unlike the global coverage percentage, this preserves a small coherent missing sword, limb, face, or subtitle region.",
         "lower = less localized missing output"),
    ("image_integrity_worst_patch_bad_pct",
         "localized_texture_damage",
         "Worst supported local patch fraction failing exact mapped-source gradient-energy and orientation checks. It complements the global integrity percentage so a small coherent blur/ringing defect is not diluted by the rest of the frame.",
         "lower = less localized blur/ringing"),
    ("exact_mapping_stretch_pct",
         "mapping_stretch",
         "Pixels where the exact source-coordinate Jacobian advances below 35% of the undistorted rate. This detects repeated/rubber-banded columns independently of image texture.",
         "lower = less warp stretch"),
    ("static_jitter_p95",
         "static_jitter",
         "Worst-eye signed-source-conditioned temporal residual over regions whose source neighborhood stayed static after allowing for horizontal disparity. Camera/object motion is excluded.",
         "lower = steadier static content"),
    ("flow_temporal_p95",
         "flow_temporal",
         "Worst-eye temporal residual after flow compensation, subtracting the registered signed mono-source change before taking magnitude so legitimate motion/interpolation error is not learned as a stereo defect.",
         "lower = steadier moving content"),
    ("depth_gt_affine_nrmse_pct",
         "gt_depth_affine_nrmse",
         "Prediction error against committed ground-truth inverse depth after robust positive-polarity IRLS scale/shift alignment; sparse GT evidence fails closed.",
         "lower = more accurate depth"),
    ("depth_gt_edge_f1",
         "gt_edge_f1",
         "Strict depth-boundary F1 against committed ground truth with one-pixel positional tolerance.",
         "higher = better boundary placement"),
    ("depth_gt_lag_f1_p95",
         "gt_depth_lag",
         "P95 amount by which predicted depth boundaries match the previous GT frame better than the current frame. Positive values directly indicate held/stale depth on moving geometry.",
         "lower = less one-frame depth lag"),
    ("vmisalign_p99_pct",
         "vmis",
         "Texture-weighted P99 vertical L↔R offset as a percentage of eye height. The upper tail catches a localized epipolar fault that a P95 summary can dilute.",
         "must be ≈ 0"),
    ("exact_mapping_fold_pct", "mapping_fold",
         "Exact source-coordinate steps that reverse direction, indicating a horizontal warp fold.",
         "lower = fewer folds"),
    ("exact_symmetry_residual_p95_pct", "camera_symmetry",
         "P95 common-camera translation residual after independently inverting both exact eye maps onto the same source samples. It is a hard renderer-conformance constraint, not stereo disparity or a perceptual-quality vote.",
         "must remain at or below the symmetric-camera limit"),
    ("warp_cross_row_shear_severity_pct", "row_shear",
         "Unsupported change in horizontal warp displacement between adjacent rows, normalized through image coordinates. Source-authored horizontal boundaries, aspect-fit bars, clamps, folds and collapsed topology are excluded before the strongest horizontal tear runs are summarized.",
         "lower = less row-wise tearing"),
    ("experimental_stereo_window_crossed_burden_pct", "window_conflict",
         "Crossed disparity that is physically cut by a lateral image boundary, weighted independently by border proximity, source contrast, spatial frequency, orientation, and disparity. Central pop is excluded.",
         "lower = less perceptible stereo-window conflict; experimental"),
    ("interocular_phase_orientation_burden_pct", "phase_conflict",
         "Localized coherent phase/orientation disagreement after registering both final eyes to unique common source coordinates. Equal-detail support prevents unilateral blur from being counted twice; component-weighted burden remains sensitive below a five-percent footprint.",
         "lower = less binocular structural conflict; experimental"),
    ("interocular_exposure_rivalry_burden_pct", "exposure_rivalry",
         "Coherent source-relative log-luminance disagreement between the registered eyes. A shared binocular transfer cancels, while a unilateral global or localized exposure change remains visible.",
         "lower = less binocular exposure rivalry; experimental"),
    ("interocular_color_gain_rivalry_burden_pct", "color_rivalry",
         "Coherent source-relative linear-light opponent-colour disagreement between the registered eyes. It catches unilateral white-balance, RGB-gain and hue errors while shared binocular colour transforms cancel.",
         "lower = less binocular colour rivalry; experimental"),
    ("depth_gt_polarity_ok", "gt_polarity",
         "Explicit prediction-to-GT sign check; a negative fit is a catastrophic near/far inversion.",
         "must remain 100%"),
]
_ROLE_ORDER = {"hard": 0, "primary": 1, "diagnostic": 2, "reported": 3}


def metric_group(key):
    spec = THR.get(key, {})
    role = spec.get("role", "reported")
    axis = spec.get("axis", "summary" if key == "score" else "stereo")
    return role, axis


METRIC_DEFS = sorted(METRIC_DEFS,
                     key=lambda m: (_ROLE_ORDER[metric_group(m[0])[0]], metric_group(m[0])[1],
                                    -impact(m[0])))


DEF_BY_KEY = {k: (what, d) for k, h, what, d in METRIC_DEFS}


def tip_text(metric):
    d = DEF_BY_KEY.get(metric)
    return f"{d[0]} ({d[1]})".replace('"', "'") if d else ""


def mtip(metric, label):
    """Metric label wrapped with a native-title tooltip (reliable inside the scroll container)."""
    t = tip_text(metric)
    return f'<span class="mtip" title="{t}">{label}</span>' if t else label


def metrics_section():
    present = ({k for aggs in (ctrl_agg, treat_agg) for agg in aggs.values() for k in agg}
               | {i["metric"] for i in CTRL["issues"]})
    rows = "".join(
        f'<tr><td class="mgroup"><span>{THR[k].get("scope", "reported")}</span>'
        f'<small>{metric_group(k)[0]} &middot; {metric_group(k)[1]}</small></td>'
        f'<td class="mname">{h}</td><td class="mwhat">{what}</td><td class="mdir">{d}</td></tr>'
        for k, h, what, d in METRIC_DEFS if k in present and k in THR)
    return (f'<details class="fold metric-defs"><summary>Metric definitions and decision roles</summary>'
            f'<div class="fold-body"><p class="sub"><b>Perceptual</b> metrics are possible '
            f'model labels but remain experimental until qualified. <b>Conformance</b> metrics '
            f'catch exact renderer contract failures and are never perceptual labels. '
            f'<b>GT-only</b> and <b>temporal-only</b> metrics apply only when their authenticated '
            f'evidence exists. Hard constraints can reject; primary metrics can vote; diagnostics '
            f'only support a conclusion.</p>'
            f'<div class="tablewrap"><table class="mtab"><thead><tr><th>scope / role / axis</th><th>metric</th>'
            f'<th>what it measures</th><th>direction</th></tr></thead><tbody>{rows}</tbody></table>'
            f'</div></div></details>')


def conclusion_section():
    """Auto-derived verdict using per-clip metric gates; means summarize but never decide."""
    wins, costs = [], []
    for k, h, worse, _, _ in COLS:
        # Headline prose is reserved for independent primary axes. Hard constraints have their
        # own fail-closed card, while diagnostics stay in the grouped/exception sections and
        # cannot turn a treatment into a winner by sheer metric count.
        if THR.get(k, {}).get("role") != "primary":
            continue
        a, b = _paired_mean_aggregate(k)
        if a is None or b is None:
            continue
        if a < 1e-6 and b < 1e-6:
            continue
        pct = (b - a) / a * 100 if a else 100.0
        movement = sbsbench.metric_delta_class(a, b, THR[k])
        if movement == "noise":
            continue
        # In a coequal comparison neither direction is "better/worse" globally (it's a tradeoff);
        # split by which run each metric favors instead.
        favors_treat = movement == "improved"
        txt = f"{mtip(k, '<b>' + h + '</b>')} {CTRL_TAG} {a:.2f} → {TREAT_TAG} {b:.2f} ({pct:+.0f}%)"
        (wins if favors_treat else costs).append(txt)
    li = ""
    decision = AB_DECISION
    if IS_TRADEOFF_CMP:
        if wins:
            li += f'<li class="c-win">{TREAT_NAME} is better on: {" · ".join(wins)}</li>'
        if costs:
            li += f'<li class="c-cost">{CTRL_NAME} is better on: {" · ".join(costs)}</li>'
    else:
        if wins:
            li += f'<li class="c-win">Primary mean movements favor treatment: {" · ".join(wins)}</li>'
        if costs:
            li += f'<li class="c-cost">Primary mean movements favor control: {" · ".join(costs)}</li>'
    axis_parts = []
    for axis, movement in sorted(decision["axes"].items()):
        axis_parts.append(f'<b>{axis}</b>: {len(movement["improved"])} win(s), '
                          f'{len(movement["regressed"])} cost(s)')
    if axis_parts:
        li += f'<li class="c-score">Primary axes: {" · ".join(axis_parts)}</li>'
    state = decision["verdict"]
    if state == "reject_run_gate":
        blockers = sum(len(run_gate["blockers"]) for run_gate in
                       decision["canonical_gate"].values() if isinstance(run_gate, dict)
                       and "blockers" in run_gate)
        verdict = (f'<b>Reject automated screen:</b> the canonical evaluator gate has {blockers} '
                   f'blocking result(s). The metric-only A/B verdict was '
                   f'<code>{html.escape(decision["ab_verdict"])}</code>.')
    elif state == "reject_evidence":
        verdict = (f'<b>Reject evidence:</b> {len(decision["missing_evidence"])} required '
                   f'hard/primary comparison value(s) are missing.')
    elif state == "reject_hard":
        verdict = (f'<b>Reject treatment:</b> {len(decision["hard_failures"])} configured '
                   f'disparity/integrity engineering bound(s) fail.')
    elif state == "reject_primary":
        verdict = (f'<b>Reject treatment:</b> {decision["regressed"]} primary-axis cost(s) '
                   f'with no compensating primary-axis win.')
    elif state == "tradeoff":
        verdict = ('<b>Primary-quality tradeoff:</b> coequal axes move in different or mixed '
                   f'directions. Per-clip event counts are evidence, not weights. Use the '
                   f'per-axis vector plus matched visual/headset evidence.')
    elif state == "screen_candidate":
        verdict = (f'<b>Experimental automated-screen candidate:</b> '
                   f'{decision["improved"]} primary-axis win(s), no primary-axis costs and no '
                   f'configured engineering-bound failure. Headset/perceptual qualification is '
                   f'still required.')
    else:
        verdict = ("<b>No automated regression detected:</b> configured engineering bounds pass, "
                   "but all enabled primary proxies remain within noise. This is not perceptual "
                   "or headset validation; diagnostic metrics cannot vote.")
    head = (f"{CTRL_NAME} → {TREAT_NAME}" if IS_TRADEOFF_CMP else f"Treatment: <b>{treatment_name()}</b>")
    scope_note = ("Extended-suite automated screening; held-out renderer failures and headset "
                  "validation remain required."
                  if DECISION_SCOPE == "extended_screening" else
                  "Core-suite automated screening; confirm on the public extended suite, held-out "
                  "renderer failures, and headset evidence.")
    qualified_labels = len(TREAT.get("meta", {}).get("training_labels", {})
                           .get("qualified_metrics", []))
    return (f'<section><h2>Conclusion</h2>'
            f'<p class="sub" style="margin-bottom:12px">{head} — decision over '
            f'{len(DECISION_CLIPS)} non-flat clip(s); expected-flat diagnostics remain below. '
            f'<b>{scope_note}</b> Qualified training labels: <b>{qualified_labels}</b>.</p>'
            f'<ul class="concl">{li}<li>{verdict}</li></ul>{gate_strip()}</section>')


def gate_strip():
    canonical = AB_DECISION["canonical_gate"]
    if not canonical["passed"]:
        items = []
        for side in ("control", "treatment"):
            for blocker in canonical[side]["blockers"]:
                field = blocker.get("field") or blocker.get("kind")
                items.append(f'<li><code>{side}.{html.escape(str(field))}</code></li>')
        return (f'<div class="gate gate-fail"><b>Gate: CANONICAL RUN REJECTED</b>'
                f'<ul>{"".join(items)}</ul></div>')
    hard = TREAT.get("hard_failures", [])
    if hard:
        items = "".join(
            f'<li><code>{name(r["clip"])}.{r["metric"]}</code> = {r["value"]}</li>' for r in hard)
        return (f'<div class="gate gate-fail"><b>Gate: {len(hard)} HARD COMFORT/INTEGRITY '
                f'FAILURE(S)</b><ul>{items}</ul></div>')
    evidence = TREAT.get("evidence_failures", [])
    if evidence:
        items = "".join(
            f'<li><code>{name(r["clip"])}.{r["metric"]}</code> missing/invalid evidence</li>'
            for r in evidence)
        return (f'<div class="gate gate-fail"><b>Gate: {len(evidence)} EVIDENCE '
                f'FAILURE(S)</b><ul>{items}</ul></div>')
    if IS_COMPARISON_ONLY:
        return ('<div class="gate gate-info"><b>Gate: COMPARISON ONLY</b> — committed baselines '
                'were not consulted; conclusions come from this matched control/treatment pair.</div>')
    regs = TREAT.get("regressions", [])
    noun = ("difference(s) vs " + CTRL_NAME if IS_TRADEOFF_CMP else "regression(s)")
    if not regs:
        return ('<div class="gate gate-pass"><b>Gate: PASS</b> — no '
                + noun + ' past threshold (run_eval exit 0).</div>')
    arrow = "→"
    items = "".join(f'<li><code>{name(r["clip"])}.{r["metric"]}</code> {r["baseline"]} {arrow} {r["value"]}'
                    + (f' <span class="wf">worst frame {r["frame"]}</span>' if "frame" in r else "")
                    + "</li>" for r in regs)
    cls = "gate-info" if IS_TRADEOFF_CMP else "gate-fail"
    label = (f"{len(regs)} {noun}" if IS_TRADEOFF_CMP else
             f"{len(regs)} REGRESSION(S) — run_eval exit 1")
    return f'<div class="gate {cls}"><b>Gate: {label}</b><ul>{items}</ul></div>'


def _evidence_card(item, kind, axis=None):
    """Render one metric-specific matched visual card."""
    _, delta, metric, c, a, b = item
    better = THR.get(metric, {}).get("better", "lower")
    treatment_worse = (delta > 0) if better == "lower" else (delta < 0)
    source = TREAT if treatment_worse else CTRL
    wf = source["clips"][c].get("worst_frame", {}).get(metric, {})
    frame = wf.get("frame", mid_frame(ctrl_dir, c))
    imgs = visual_evidence_images(c, frame, metric)
    if not imgs:
        return ""
    pct = delta / a * 100 if a else 100.0
    cls = ("evidence-cost" if kind == "regression" else "evidence-win" if kind == "improvement"
           else "evidence-noise")
    badge = kind.replace("_", " ")
    is_gt = metric in ("depth_gt_affine_nrmse_pct",
                       "depth_gt_edge_f1",
                       "depth_gt_polarity_ok")
    is_gt_lag = metric == "depth_gt_lag_f1_p95"
    is_exact_source = metric in {
        "source_coverage_pct", "image_integrity_pct",
        "source_coverage_worst_patch_bad_pct",
        "image_integrity_worst_patch_bad_pct",
    }
    is_shear = metric == "warp_cross_row_shear_severity_pct"
    is_window = metric.startswith("experimental_stereo_window_crossed_")
    is_interocular_conflict = metric in {
        "interocular_phase_orientation_burden_pct",
        "interocular_exposure_rivalry_burden_pct",
        "interocular_color_gain_rivalry_burden_pct",
    }
    metadata = EVIDENCE_METADATA.get((c, frame, metric), {})
    source_label = ("source &middot; unmodified" if metric == "static_jitter_p95"
                    else "source · bright = reliable optical flow" if metric == "flow_temporal_p95"
                    else "ground-truth depth" if is_gt else "source")
    if is_exact_source:
        source_label = "exact shader-mapped source · conformance reference"
    if is_shear:
        source_label = ("exact mapped source &middot; artifact-free "
                        f'{metadata.get("eye", "selected")} eye reference')
    if is_window:
        source_label = (f'source at {metadata.get("border", "selected")} stereo-window border')
    if is_interocular_conflict:
        source_label = "registered mono source"
    ctrl_label = (f"{CTRL_TAG} · temporal change" if metric == "static_jitter_p95" else
                  f"{CTRL_TAG} · flow residual" if metric == "flow_temporal_p95" else
                  f"{CTRL_TAG} · aligned depth" if is_gt else
                  f"{CTRL_TAG} · left | right" if metric in EXACT_STEREO_VISUAL_METRICS
                  else CTRL_TAG)
    treat_label = (f"{TREAT_TAG} · temporal change" if metric == "static_jitter_p95" else
                   f"{TREAT_TAG} · flow residual" if metric == "flow_temporal_p95" else
                   f"{TREAT_TAG} · aligned depth" if is_gt else
                   f"{TREAT_TAG} · left | right" if metric in EXACT_STEREO_VISUAL_METRICS
                   else TREAT_TAG)
    if is_shear:
        eye_name = metadata.get("eye", "selected")
        ctrl_label = f"{CTRL_TAG} &middot; {eye_name} eye"
        treat_label = f"{TREAT_TAG} &middot; {eye_name} eye"
    if is_window:
        ctrl_label = f"{CTRL_TAG} &middot; border crop left | right"
        treat_label = f"{TREAT_TAG} &middot; border crop left | right"
    if is_interocular_conflict:
        ctrl_label = f"{CTRL_TAG} &middot; registered left | right"
        treat_label = f"{TREAT_TAG} &middot; registered left | right"
    analysis_label = "delta: red worse / blue better"
    if is_shear:
        analysis_label = ("row-shear detector &middot; cyan support &middot; "
                          f'selected {metadata.get("selected_run", "run")}')
    elif is_window:
        analysis_label = ("crossed stereo-window map &middot; "
                          f'selected {metadata.get("selected_run", "run")}')
    elif is_interocular_conflict:
        analysis_label = (f'{metadata.get("detector", "binocular")} conflict map &middot; '
                          f'selected {metadata.get("selected_run", "run")}')
    elif metric == "static_jitter_p95":
        analysis_label = "analysis mask (not source content) &middot; cyan = static support"
    if is_gt_lag and len(imgs) == 4:
        panels = (f'<div class="quad"><figure><span class="tag">previous ground-truth depth</span>'
                  f'<img src="{imgs[0]}"></figure><figure><span class="tag">current ground-truth depth</span>'
                  f'<img src="{imgs[1]}"></figure><figure><span class="tag">{CTRL_TAG} &middot; aligned depth</span>'
                  f'<img src="{imgs[2]}"></figure><figure><span class="tag t-treat">{TREAT_TAG} &middot; aligned depth</span>'
                  f'<img src="{imgs[3]}"></figure></div>')
    else:
        panels = (f'<div class="quad"><figure><span class="tag">{source_label}</span><img src="{imgs[0]}"></figure>'
              f'<figure><span class="tag">{ctrl_label}</span><img src="{imgs[1]}"></figure>'
              f'<figure><span class="tag t-treat">{treat_label}</span><img src="{imgs[2]}"></figure>'
              f'<figure><span class="tag t-diff">{analysis_label}</span>'
              f'<img src="{imgs[3]}"></figure></div>' if len(imgs) == 4 else
              f'<div class="triplet"><figure><span class="tag">{ctrl_label}</span>'
              f'<img src="{imgs[0]}"></figure><figure><span class="tag t-treat">{treat_label}</span>'
              f'<img src="{imgs[1]}"></figure><figure><span class="tag t-diff">abs diff &times;5</span>'
              f'<img src="{imgs[2]}"></figure></div>')
    axis_label = f'<span class="axis-label">{axis}</span>' if axis else ""
    return (f'<article class="evidence-card {cls}"><div class="ic-head">{axis_label}'
            f'<span class="clipname">{name(c)}</span><span class="pill">{badge}</span>'
            f'<span class="metricval">{mtip(metric, SHORT.get(metric, metric))}: '
            f'<b>{a:.2f}</b> &rarr; {b:.2f} ({pct:+.0f}%) &middot; frame {frame}</span></div>'
            f'{panels}</article>')


def _strongest_change(metric, clips=DECISION_CLIPS):
    floor = THR.get(metric, {}).get("abs_floor", 0.0)
    candidates = []
    for c in clips:
        a, b = _metric_value(CTRL, c, metric), _metric_value(TREAT, c, metric)
        if a is None or b is None:
            continue
        delta = b - a
        denom = max(floor, abs(a) * 0.05, 1e-6)
        candidates.append((abs(delta) / denom, delta, metric, c, a, b))
    return max(candidates, default=None)


def _change_kind(item):
    _, delta, metric, _, a, _ = item
    spec = THR[metric]
    significant = abs(delta) > max(spec.get("abs_floor", 0.0), abs(a) * spec.get("rel_tol", 0.0))
    if not significant:
        return "within_noise"
    improved = delta > 0 if spec.get("better") == "higher" else delta < 0
    return "improvement" if improved else "regression"


def visual_evidence_section():
    """Show one representative example for every quality and reference axis."""
    axes = (("Stereo volume", ("exact_visible_pop_spread_pct",)),
            ("Warp geometry", ("exact_mapping_stretch_pct", "exact_mapping_fold_pct",
                               "warp_cross_row_shear_severity_pct")),
            ("Perceptual artifacts", (
                "experimental_stereo_window_crossed_burden_pct",
                "interocular_phase_orientation_burden_pct",
                "interocular_exposure_rivalry_burden_pct",
                "interocular_color_gain_rivalry_burden_pct")),
            ("Renderer conformance", (
                "source_coverage_pct", "image_integrity_pct",
                "source_coverage_worst_patch_bad_pct",
                "image_integrity_worst_patch_bad_pct")),
            ("Temporal stability", ("static_jitter_p95", "flow_temporal_p95",
                                     "depth_gt_lag_f1_p95")),
            ("Ground-truth depth", ("depth_gt_affine_nrmse_pct", "depth_gt_edge_f1")))
    cards = []
    for axis, metrics in axes:
        item = max((item for metric in metrics if (item := _strongest_change(metric))),
                   default=None)
        if item:
            cards.append(_evidence_card(item, _change_kind(item), axis))
    cards = "".join(cards)
    if not cards:
        return ""
    return (f'<section><h2>Quality-axis visual evidence</h2>'
            f'<p class="sub">One strongest matched example for each decision/reference axis. Exact '
            f'mapped-source heatmaps verify renderer conformance, not perceptual quality; stereo '
            f'uses visible relief and shows both eyes, while reference depth shows aligned '
            f'predictions against ground truth. A within-noise '
            f'badge means the '
            f'example is illustrative, not a decision event.</p>{cards}</section>')


def source_artifact_section():
    """Show inspected original frames whose baked artifacts can confound warp metrics."""
    cards = []
    for clip in CLIPS:
        clip_meta = CTRL["clips"].get(clip, {}).get("meta", {})
        note = clip_meta.get("source_artifacts")
        if not note:
            continue
        frame = mid_frame(ctrl_dir, clip)
        path = source_path(clip, frame)
        if not path:
            continue
        image_url = durl(Image.open(path).convert("RGB"), w=420, jpg=True, q=84)
        kind = html.escape(clip_meta.get("content_type", "source"))
        cards.append(
            f'<article class="source-card"><img src="{image_url}"><div>'
            f'<div class="ic-head"><span class="clipname">{html.escape(name(clip))}</span>'
            f'<span class="pill p-info">{kind}</span><span class="metricval">original frame {frame}</span></div>'
            f'<p>{html.escape(note)}</p></div></article>')
    if not cards:
        return ""
    return (f'<section><h2>Original-source artifact audit</h2>'
            f'<p class="sub">These effects are already present before depth estimation or stereo '
            f'warping. Exact mapped-source coverage/integrity tests verify whether the renderer reproduced its '
            f'requested source sample; they are renderer-conformance diagnostics, not perceptual '
            f'artifact labels. Their cancellation of a baked highlight, bloom edge, rain splash or '
            f'generative inconsistency does not prove that the displaced stereo result looks clean. '
            f'Perceptual conclusions still require the original and rendered images together.</p>'
            f'{"".join(cards)}</section>')


def diagnostic_evidence_section():
    """Only surface unusually large diagnostic moves after the primary evidence."""
    metrics = ("exact_mapping_fold_pct", "warp_cross_row_shear_severity_pct")
    candidates = []
    for metric in metrics:
        item = _strongest_change(metric)
        if not item:
            continue
        _, delta, _, _, a, _ = item
        floor = THR[metric].get("abs_floor", 0.0)
        if abs(delta) >= max(floor * 2.0, abs(a) * 0.5):
            candidates.append(item)
    cards = "".join(_evidence_card(item, _change_kind(item), "Diagnostic exception")
                    for item in sorted(candidates, reverse=True)[:3])
    if not cards:
        return ""
    return (f'<section><h2>Large diagnostic changes</h2>'
            f'<p class="sub">Shown only when a supporting metric changes by at least 50% and twice '
            f'its absolute noise floor. These examples explain a large secondary movement but do '
            f'not vote against the primary axes.</p>{cards}</section>')


def clean_footer():
    if not CLEAN:
        return ""
    items = ", ".join(f"{h} {colmax[k]:.2f}" for k, h, *_ in CLEAN)
    return (f'<p class="foot"><b>Clean this run (max ≈ 0, collapsed):</b> {items}. Still measured '
            f'every run — any one auto-appears as a column when it crosses threshold.</p>')


meta = CTRL["meta"]

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Apollo SBS evaluation report</title><style>
:root{--bg:#f5f6f7;--panel:#fff;--ink:#12181d;--muted:#5c6a74;--line:#dbe1e6;--accent:#0e8f9c;
  --accent-soft:#d7eef0;--good:#1f9d63;--warn:#c98a1a;--crit:#c4483a;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Consolas,monospace;--sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme:dark){:root{--bg:#0c1013;--panel:#141a1f;--ink:#e8edf0;--muted:#8b9aa4;
  --line:#252e35;--accent:#38c0cd;--accent-soft:#123037;--good:#3fca86;--warn:#e0a94a;--crit:#e56a5c;}}
:root[data-theme="light"]{--bg:#f5f6f7;--panel:#fff;--ink:#12181d;--muted:#5c6a74;--line:#dbe1e6;--accent:#0e8f9c;--accent-soft:#d7eef0;--good:#1f9d63;--warn:#c98a1a;--crit:#c4483a;}
:root[data-theme="dark"]{--bg:#0c1013;--panel:#141a1f;--ink:#e8edf0;--muted:#8b9aa4;--line:#252e35;--accent:#38c0cd;--accent-soft:#123037;--good:#3fca86;--warn:#e0a94a;--crit:#e56a5c;}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;background:var(--bg)}
.wrap{max-width:1060px;margin:0 auto;padding:56px 24px 96px;color:var(--ink);font-family:var(--sans);line-height:1.6;background:var(--bg);-webkit-font-smoothing:antialiased}
h1,h2{text-wrap:balance;line-height:1.15;margin:0}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:14px}
h1{font-size:36px;font-weight:680;letter-spacing:-.02em}
.lede{color:var(--muted);font-size:16.5px;max-width:68ch;margin-top:14px}
.meta{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:20px;display:flex;gap:18px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:16px}
section{margin-top:52px}
h2{font-size:15px;font-family:var(--mono);letter-spacing:.03em;text-transform:uppercase;color:var(--ink);padding-bottom:12px;border-bottom:1px solid var(--line);margin-bottom:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.sub{color:var(--muted);font-size:14px;margin:0 0 20px;max-width:72ch}
.foot{margin-top:14px;color:var(--muted);font-size:13px}.foot b{color:var(--ink)}
.fold{margin-top:26px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
.fold>summary{cursor:pointer;list-style:none;padding:14px 17px;font-family:var(--mono);font-size:12px;font-weight:650;color:var(--ink);display:flex;align-items:center;gap:10px}.fold>summary::-webkit-details-marker{display:none}.fold>summary:before{content:"+";color:var(--accent);font-size:16px;line-height:1}.fold[open]>summary:before{content:"−"}.fold[open]>summary{border-bottom:1px solid var(--line)}
.fold-body{padding:18px}.fold-body>.sub:last-child{margin-bottom:0}.metric-defs{margin-top:28px}
.gate{border-radius:10px;padding:14px 18px;font-size:14px;margin-top:26px;border:1px solid var(--line)}
.gate-pass{background:color-mix(in srgb,var(--good) 9%,transparent);border-color:color-mix(in srgb,var(--good) 40%,var(--line))}
.gate-fail{background:color-mix(in srgb,var(--crit) 8%,transparent);border-color:color-mix(in srgb,var(--crit) 40%,var(--line))}
.gate-info{background:color-mix(in srgb,var(--accent) 8%,transparent);border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}
.gate ul{margin:8px 0 0;padding-left:20px}.gate li{margin:2px 0}
.gate .wf{font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.concl{margin:0 0 18px;padding-left:20px;font-size:14.5px}
.concl li{margin:7px 0;max-width:78ch}
.concl b{color:var(--ink)}
.c-win{color:var(--good)}.c-win b{color:var(--good)}
.c-cost{color:var(--crit)}.c-cost b{color:var(--crit)}
.c-score{font-size:15.5px;color:var(--ink)}.c-score b{color:var(--accent);font-size:16px}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:right;padding:11px 13px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:middle}
th:first-child,td:first-child{text-align:left;min-width:225px}
thead th{font-family:var(--mono);font-size:11px;letter-spacing:.02em;text-transform:uppercase;color:var(--muted);font-weight:600;background:var(--panel)}
thead th[title]:not([title=""]),.mtip{cursor:help;text-decoration:underline dotted;text-underline-offset:3px;text-decoration-color:color-mix(in srgb,var(--muted) 60%,transparent)}
tbody tr:last-child td{border-bottom:none}
td{font-family:var(--mono);font-variant-numeric:tabular-nums}
.mtab td,.mtab th{text-align:left;white-space:normal}
.mtab .mname{font-family:var(--mono);font-size:12.5px;color:var(--accent);font-weight:600;white-space:nowrap;vertical-align:top}
.mtab .mwhat{font-family:var(--sans);font-size:13.5px;color:var(--ink);max-width:60ch}
.mtab .mdir{font-family:var(--mono);font-size:11.5px;color:var(--muted);white-space:nowrap;vertical-align:top}
.mtab .mgroup{font-family:var(--mono);white-space:nowrap;vertical-align:top}.mgroup span{display:block;font-size:11.5px;color:var(--ink);font-weight:650;text-transform:uppercase}.mgroup small{display:block;font-size:10.5px;color:var(--muted)}
.decision-grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:16px;align-items:start}
.decision-card{border:1px solid var(--line);border-radius:11px;background:var(--panel);padding:15px;min-width:0}
.decision-card>p{font-size:12px;color:var(--muted);margin:5px 0 14px}
.decision-axes{grid-column:span 5}.decision-hard{grid-column:span 7}.decision-heat{grid-column:1/-1}
.decision-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;flex-wrap:wrap}
.decision-head h3{font-family:var(--mono);font-size:13px;color:var(--ink);margin:0}.decision-head>span{font-family:var(--mono);font-size:10px;color:var(--muted)}
.dumbbell-list,.hard-bullet-list{display:grid;gap:2px}
.dumbbell-row{display:grid;grid-template-columns:minmax(135px,.9fr) minmax(150px,1.1fr);gap:6px 10px;align-items:center;padding:8px 0;border-top:1px solid color-mix(in srgb,var(--line) 65%,transparent)}
.dumbbell-copy b,.hard-bullet-copy b{display:block;font-family:var(--mono);font-size:10.5px;color:var(--ink)}
.dumbbell-copy small,.hard-bullet-copy small{display:block;font-family:var(--mono);font-size:8.5px;line-height:1.35;color:var(--muted)}
.dumbbell-scale{height:15px;position:relative;border-radius:8px;background:linear-gradient(to right,color-mix(in srgb,var(--crit) 9%,transparent),color-mix(in srgb,var(--muted) 7%,transparent) 35%,color-mix(in srgb,var(--muted) 7%,transparent) 65%,color-mix(in srgb,var(--good) 9%,transparent))}
.noise-zone{position:absolute;left:35%;width:30%;top:0;bottom:0;border-left:1px dashed var(--muted);border-right:1px dashed var(--muted);opacity:.55}
.dumbbell-line{position:absolute;top:6px;height:3px;background:var(--muted);border-radius:3px}
.dumbbell-dot,.hard-dot{position:absolute;width:7px;height:7px;border-radius:50%;transform:translateX(-50%);box-shadow:0 0 0 1px var(--panel)}
.dumbbell-dot{top:4px}.dot-control{background:var(--accent)}.dot-treatment{background:var(--warn)}
.dumbbell-row.is-good .dot-treatment{background:var(--good)}.dumbbell-row.is-bad .dot-treatment{background:var(--crit)}.dumbbell-row.is-noise .dot-treatment{background:var(--muted)}
.dumbbell-direction{display:flex;justify-content:space-between;font-family:var(--mono);font-size:7.5px;color:var(--muted);margin-top:2px}.dumbbell-direction span:nth-child(2){text-align:center}
.dumbbell-values{grid-column:1/-1;display:flex;justify-content:flex-end;align-items:center;gap:5px;font-family:var(--mono);font-size:9px}.dumbbell-values code{font-size:9px;padding:0 4px}.dumbbell-values small{margin-left:auto}.dumbbell-values .is-good{color:var(--good)}.dumbbell-values .is-bad{color:var(--crit)}.dumbbell-values .is-noise{color:var(--muted)}
.dumbbell-row.is-unsupported{grid-template-columns:1fr auto}.dumbbell-na{font-family:var(--mono);font-size:9px;color:var(--muted);padding:3px 7px;border:1px dashed var(--line);border-radius:5px}.dumbbell-row.is-bad .dumbbell-na{color:var(--crit);border-color:color-mix(in srgb,var(--crit) 45%,var(--line))}
.runtime-strip{display:flex;gap:7px 12px;flex-wrap:wrap;border-top:1px solid var(--line);margin-top:10px;padding-top:10px;font-family:var(--mono);font-size:9px;color:var(--muted)}.runtime-strip>small{flex-basis:100%}.runtime-strip code{font-size:8.5px;padding:0 3px}.runtime-strip b{color:var(--ink)}
.hard-bullet-row{display:grid;grid-template-columns:minmax(185px,1.15fr) minmax(130px,.85fr) auto;gap:8px;align-items:center;padding:6px 0;border-top:1px solid color-mix(in srgb,var(--line) 65%,transparent)}
.hard-track{height:16px;position:relative;border-radius:8px}.hard-track.limit-max{background:linear-gradient(to right,color-mix(in srgb,var(--good) 10%,transparent) 0 var(--bound),color-mix(in srgb,var(--crit) 13%,transparent) var(--bound) 100%)}.hard-track.limit-min{background:linear-gradient(to right,color-mix(in srgb,var(--crit) 13%,transparent) 0 var(--bound),color-mix(in srgb,var(--good) 10%,transparent) var(--bound) 100%)}
.hard-bound{position:absolute;left:var(--bound);top:-2px;bottom:-2px;width:2px;background:var(--ink);opacity:.55}.hard-dot.dot-control{top:2px}.hard-dot.dot-treatment{bottom:1px}
.hard-bullet-values{display:grid;grid-template-columns:60px 10px 60px;gap:3px;align-items:center;text-align:right;font-family:var(--mono);font-size:9px}.hard-bullet-values code{font-size:8.5px;padding:0 3px}.hard-pass{color:var(--good)}.hard-fail,.hard-missing{color:var(--crit)}.hard-unsupported{color:var(--muted)}
.heatmap-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:9px}.heatmap{min-width:870px;table-layout:fixed;font-size:9px}.heatmap th,.heatmap td{padding:5px 4px;text-align:center;white-space:normal;border-right:1px solid color-mix(in srgb,var(--line) 55%,transparent)}.heatmap .heat-clip{min-width:128px;width:128px;text-align:left;position:sticky;left:0;z-index:1;background:var(--panel)}.heatmap thead th{font-size:8px;line-height:1.2}.heat-groups th{font-size:8.5px;letter-spacing:.04em}.heatmap tbody th{font-family:var(--mono);font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.heatmap td{font-family:var(--mono);font-variant-numeric:tabular-nums}.heat-good{color:var(--good)}.heat-bad,.heat-missing{color:var(--crit)}.heat-missing{background:color-mix(in srgb,var(--crit) 18%,transparent);font-weight:700}.heat-good.heat-soft{background:color-mix(in srgb,var(--good) 12%,transparent)}.heat-good.heat-mid{background:color-mix(in srgb,var(--good) 22%,transparent)}.heat-good.heat-strong{background:color-mix(in srgb,var(--good) 34%,transparent)}.heat-bad.heat-soft{background:color-mix(in srgb,var(--crit) 12%,transparent)}.heat-bad.heat-mid{background:color-mix(in srgb,var(--crit) 22%,transparent)}.heat-bad.heat-strong{background:color-mix(in srgb,var(--crit) 34%,transparent)}.heat-noise{color:var(--muted);background:color-mix(in srgb,var(--muted) 7%,transparent)}.heat-unsupported{color:var(--muted);background:repeating-linear-gradient(135deg,transparent 0 4px,color-mix(in srgb,var(--muted) 12%,transparent) 4px 7px)}
.heat-legend{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:8px;font-family:var(--mono);font-size:8.5px;color:var(--muted)}.heat-legend span{display:inline-block;min-width:24px;padding:1px 4px;text-align:center;border:1px solid var(--line);border-radius:4px}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.metric-chart{border:1px solid var(--line);border-radius:11px;background:var(--panel);padding:14px 14px 10px;min-width:0}
.chart-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:9px}
.chart-head h3{font-family:var(--mono);font-size:13px;color:var(--ink);margin:0}.chart-head>span{font-family:var(--mono);font-size:10px;color:var(--muted)}
.bar-row{display:grid;grid-template-columns:112px minmax(130px,1fr) 72px;align-items:center;gap:8px;padding:6px 0;border-top:1px solid color-mix(in srgb,var(--line) 65%,transparent)}
.bar-scene{font-family:var(--mono);font-size:10.5px;color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-pair{display:grid;gap:3px}.bar-line{display:grid;grid-template-columns:34px minmax(50px,1fr) 42px;align-items:center;gap:5px}
.bar-tag{font-family:var(--mono);font-size:9px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-line b{font-family:var(--mono);font-size:9.5px;font-weight:500;text-align:right;font-variant-numeric:tabular-nums}
.bar-track{height:5px;background:color-mix(in srgb,var(--muted) 10%,transparent);border-radius:5px;overflow:hidden}
.bar-fill{display:block;height:100%;min-width:0;border-radius:5px}.bar-control{background:var(--accent)}
.bar-good{background:var(--good)}.bar-bad{background:var(--crit)}.bar-flat{background:var(--muted)}
.bar-delta{font-family:var(--mono);font-size:9.5px;text-align:right;background:none}.bar-delta.bar-good{color:var(--good)}.bar-delta.bar-bad{color:var(--crit)}.bar-delta.bar-flat{color:var(--muted)}
.bar-na,.bar-missing{font-family:var(--mono);font-size:9.5px;align-self:center}.bar-na{color:var(--muted)}.bar-missing{color:var(--crit);font-weight:700}
.clipname{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--ink)}
.pill{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:20px;font-weight:600;white-space:nowrap}
.p-warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.p-crit{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.p-info{color:var(--accent);background:var(--accent-soft)}
.issue-clip{margin-top:20px}
.ic-head{display:flex;align-items:baseline;gap:14px;margin-bottom:10px;flex-wrap:wrap}
.metricval{font-family:var(--mono);font-size:13px;color:var(--muted)}.metricval b{color:var(--ink)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.pair.single{grid-template-columns:1fr;max-width:540px}
.pair figure{margin:0;position:relative}
.pair img{width:100%;border-radius:9px;border:1px solid var(--line);display:block}
.evidence-card{margin-top:22px;padding:14px;border:1px solid var(--line);border-radius:11px;background:var(--panel)}
.evidence-card .pill{color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent)}
.evidence-card.evidence-cost .pill{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.evidence-card.evidence-noise .pill{color:var(--muted);background:color-mix(in srgb,var(--muted) 14%,transparent)}
.source-card{display:grid;grid-template-columns:minmax(260px,420px) 1fr;gap:18px;align-items:center;margin-top:16px;padding:12px;border:1px solid var(--line);border-radius:11px;background:var(--panel)}
.source-card img{width:100%;display:block;border-radius:8px;border:1px solid var(--line)}.source-card p{font-size:13.5px;color:var(--muted);margin:6px 0 0}.source-card .ic-head{margin:0}
.axis-label{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--accent);padding-right:4px}
.triplet{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.quad{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.triplet figure,.quad figure{margin:0;position:relative}.triplet img,.quad img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
.tag.t-diff{color:var(--accent)}
.tag{position:absolute;top:8px;left:8px;font-family:var(--mono);font-size:11px;font-weight:600;padding:2px 8px;border-radius:5px;background:color-mix(in srgb,var(--bg) 82%,transparent);border:1px solid var(--line);color:var(--ink)}
.tag.t-treat{color:var(--warn)}
pre{font-family:var(--mono);font-size:12px;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px;overflow-x:auto;color:var(--ink);line-height:1.7}
code{font-family:var(--mono);font-size:12px;background:var(--panel);border:1px solid var(--line);padding:1px 6px;border-radius:5px}
@media (max-width:900px){.decision-axes,.decision-hard,.decision-heat{grid-column:1/-1}.chart-grid{grid-template-columns:1fr}}
@media (max-width:640px){.wrap{padding:38px 14px 72px}.source-card{grid-template-columns:1fr}.pair,.triplet{grid-template-columns:1fr}.quad{grid-template-columns:1fr 1fr}.dumbbell-row,.hard-bullet-row{grid-template-columns:1fr}.dumbbell-values,.hard-bullet-values{grid-column:1/-1;justify-self:stretch}.hard-bullet-values{grid-template-columns:1fr 12px 1fr}.hard-bullet-values code:first-child{text-align:right}.hard-bullet-values code:last-child{text-align:left}.bar-row{grid-template-columns:88px minmax(110px,1fr) 62px}h1{font-size:28px}}
@media (max-width:380px){.quad{grid-template-columns:1fr}.bar-row{grid-template-columns:72px minmax(100px,1fr) 54px}}
</style></head><body>

<div class="wrap">
  <div class="eyebrow">Apollo SBS 3D &middot; run_eval A/B report</div>
  <h1>__H1__</h1>
  <p class="lede">Generated from two <code>run_eval.py</code> runs over the committed clip set —
  the real pipeline and gated metrics. __LEDE__</p>
  <div class="meta"><span>__DATE__</span><span>control __CTRL_SHA__</span>
  <span>treatment __TREAT_SHA__</span>
  <span>__NCLIPS__ clips</span><span>__MODELS__</span><span>report __REPORT_SHA__</span></div>

  __CONCLUSION__

  __METRICS__

  __GROUP_DASHBOARD__

  __SOURCE_ARTIFACTS__

  __VISUAL_EVIDENCE__

  __DIAGNOSTIC_EVIDENCE__

  __LEARNED_ORACLES__

  <section>
    <h2>Reproduce</h2>
    <pre>python tools/sbsbench/run_eval.py --label ctrl                              # control (gates vs baselines)
python tools/sbsbench/run_eval.py --label treat --extra __TREAT_ARGS__     # treatment
python tools/sbsbench/generate_report.py &lt;build&gt;/sbs_eval/ctrl &lt;build&gt;/sbs_eval/treat report.html</pre>
    <p style="color:var(--muted);font-size:13px;margin-top:12px">Metrics: <code>tools/sbsbench/sbsbench.py</code>
    &middot; gate: <code>thresholds.json</code> &middot; plan: <code>docs/sbs-benchmark-plan.md</code></p>
  </section>

  <details class="fold bar-fold">
    <summary>Per-clip bar comparison — __CTRL_NAME__ → __TREAT_NAME__</summary>
    <div class="fold-body"><p class="sub">Matched <b>__CTRL_TAG__</b> and <b>__TREAT_TAG__</b> bars
    for every clip. Bars share a scale within each metric; exact values remain printed beside them.
    Green is better, red is worse, and grey is within noise.</p>
    __CHARTS__
    __FOOTER__</div>
  </details>
</div></body></html>
"""

models = ", ".join(sorted({m for r in (CTRL, TREAT)
                           for m in {e["meta"].get("model", "?") for e in r["clips"].values()}}))
if IS_TRADEOFF_CMP:
    h1 = f"{CTRL_NAME} vs. {TREAT_NAME}"
    comparison_kind = "modes" if IS_MODE_CMP else "profiles"
    displayed_verdict = AB_DECISION["verdict"].replace("_", " ")
    lede = (f"Comparing two pipeline {comparison_kind} on identical clips: <b>{CTRL_NAME}</b> against "
            f"<b>{TREAT_NAME}</b>. The canonical report verdict is "
            f"<b>{html.escape(displayed_verdict)}</b>; read the per-axis split, gate, and matched "
            f"per-clip evidence below.")
else:
    h1 = "Control vs. treatment, by issue"
    lede = (f"Matched comparison-only run: <b>{CTRL_NAME}</b> against <b>{TREAT_NAME}</b>; "
            "committed baselines were not consulted." if IS_COMPARISON_ONLY else
            f"Treatment under test: <b>{TREAT_NAME}</b>, gated against the committed baselines.")
ctrl_exe_sha = CTRL["meta"].get("executable_sha256", "?")
treat_exe_sha = TREAT["meta"].get("executable_sha256", "?")
ctrl_shader_sha = CTRL["meta"].get("runtime_shader_sha256", "?")
treat_shader_sha = TREAT["meta"].get("runtime_shader_sha256", "?")
ctrl_engine_sha = CTRL["meta"].get("engine_sha256", "?")
treat_engine_sha = TREAT["meta"].get("engine_sha256", "?")
ctrl_sha = (CTRL["meta"].get("git_sha", "?") +
            ("+dirty" if CTRL["meta"].get("git_dirty") else "") +
            f" · exe {ctrl_exe_sha[:12]} · shaders {ctrl_shader_sha[:12]}"
            f" · engine {ctrl_engine_sha[:12]}")
treat_sha = (TREAT["meta"].get("git_sha", "?") +
             ("+dirty" if TREAT["meta"].get("git_dirty") else "") +
             f" · exe {treat_exe_sha[:12]} · shaders {treat_shader_sha[:12]}"
             f" · engine {treat_engine_sha[:12]}")
HTML = (HTML.replace("__H1__", h1).replace("__LEDE__", lede)
        .replace("__CTRL_NAME__", CTRL_NAME).replace("__TREAT_NAME__", TREAT_NAME)
        .replace("__DATE__", meta["timestamp"][:10]).replace("__CTRL_SHA__", ctrl_sha)
        .replace("__TREAT_SHA__", treat_sha)
        .replace("__NCLIPS__", str(len(CLIPS)))
        .replace("__REPORT_SHA__", REPORT_SHA)
        .replace("__MODELS__", models).replace("__CONCLUSION__", conclusion_section())
        .replace("__SOURCE_ARTIFACTS__", source_artifact_section())
        .replace("__VISUAL_EVIDENCE__", visual_evidence_section())
        .replace("__DIAGNOSTIC_EVIDENCE__", diagnostic_evidence_section())
        .replace("__LEARNED_ORACLES__",
                 offline_oracle_report.build_section(treat_dir, CLIPS, name))
        .replace("__CTRL_TAG__", CTRL_TAG).replace("__TREAT_TAG__", TREAT_TAG)
        .replace("__GROUP_DASHBOARD__", grouped_quality_section())
        .replace("__CHARTS__", scorecard_charts())
        .replace("__METRICS__", metrics_section())
        .replace("__FOOTER__", clean_footer())
        .replace("__TREAT_ARGS__", " ".join(TREAT["meta"].get("extra_args") or ["profile defaults"])))
os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
with open(out_html, "w", encoding="utf-8") as f:
    f.write(HTML)
decision_path = os.path.join(os.path.dirname(os.path.abspath(out_html)), "decision.json")
with open(decision_path, "w", encoding="utf-8") as f:
    json.dump({
        "schema": 3,
        "control": CTRL_NAME,
        "treatment": TREAT_NAME,
        "eval_schema": TREAT.get("meta", {}).get("eval_schema"),
        "metric_sha256": TREAT.get("meta", {}).get("metric_sha256"),
        "control_executable_sha256": CTRL.get("meta", {}).get("executable_sha256"),
        "treatment_executable_sha256": TREAT.get("meta", {}).get("executable_sha256"),
        "control_runtime_shader_sha256": CTRL.get("meta", {}).get("runtime_shader_sha256"),
        "treatment_runtime_shader_sha256": TREAT.get("meta", {}).get("runtime_shader_sha256"),
        "control_engine_sha256": CTRL.get("meta", {}).get("engine_sha256"),
        "treatment_engine_sha256": TREAT.get("meta", {}).get("engine_sha256"),
        "control_onnx_sha256": CTRL.get("meta", {}).get("onnx_sha256"),
        "treatment_onnx_sha256": TREAT.get("meta", {}).get("onnx_sha256"),
        "report_sha256": REPORT_SHA,
        "clips": CLIPS,
        "decision_clips": DECISION_CLIPS,
        "decision_scope": DECISION_SCOPE,
        "source_artifact_clips": SOURCE_ARTIFACT_CLIPS,
        **AB_DECISION,
    }, f, indent=2, sort_keys=True)
print("wrote", out_html, f"({len(HTML) // 1024} KB)")
print("decision", decision_path, AB_DECISION["verdict"])
