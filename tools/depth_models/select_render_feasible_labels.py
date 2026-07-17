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
import sbs_harness_contract as sbs_contract  # noqa: E402

from artistic_policy_contract import ART_SCALE_DELTA_MAX  # noqa: E402
import depth_input_color as input_color  # noqa: E402

PROTECTED_PRIMARY_AXES = {"warp", "stability"}
EXACT_POP_METRIC = "exact_pop_spread_pct"
STYLE_NAMES = ("immersive", "balanced", "clean")
DEFAULT_STYLE = "immersive"
POLICY_CONTRACT = "safe-frontier-multistyle-apollo-v1"
POLICY_WARP_CONTRACT = "apollo-safe-frontier-v1"
GENERIC_SOURCE_SCHEMA = 2
GENERIC_SOURCE_CONTRACT = "full-cadence-artistic-source-v2"
MAX_CANDIDATE_SCALE_STEP = 0.10
EXPECTED_HARNESS_SCHEMA = sbs_contract.HARNESS_SCHEMA
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
    "eye_width", "eye_height", "color_mode", "hdr_source_kind",
    "metric_preview_encoding",
    "hdr_input_scale", "sdr_white_level_raw",
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
HARNESS_INPUT_FIELDS = (
    "color_mode", "hdr_source_kind", "metric_preview_encoding", "hdr_input_scale",
    "sdr_white_level_raw",
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


def _input_variant_harness_context(input_variant):
    input_color.validate_input_variant(input_variant)
    return {
        "color_mode": input_variant["color_mode"],
        "hdr_source_kind": sbs_contract.input_variant_hdr_source_kind(
            input_variant
        ),
        "metric_preview_encoding": (
            sbs_contract.input_variant_metric_preview_encoding(input_variant)
        ),
        "hdr_input_scale": float(input_variant["scrgb_white_scale"] or 0.0),
        "sdr_white_level_raw": int(
            input_variant["windows_sdr_white_level_raw"] or 0
        ),
    }


def authenticated_input_variant(payload, origin, allow_legacy_sdr=False):
    """Validate the canonical variant object and its semantic SHA-256."""
    variant = payload.get("input_variant") if isinstance(payload, dict) else None
    variant_hash = (
        payload.get("input_variant_sha256") if isinstance(payload, dict) else None
    )
    if variant is None and variant_hash is None and allow_legacy_sdr:
        variant = input_color.sdr_input_variant()
        variant_hash = input_color.input_variant_sha256(variant)
    elif variant is None or variant_hash is None:
        raise RuntimeError(f"{origin}: missing authenticated input variant")
    try:
        input_color.validate_input_variant(variant)
        expected_hash = input_color.input_variant_sha256(variant)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"{origin}: invalid input variant") from error
    if variant_hash != expected_hash:
        raise RuntimeError(f"{origin}: input variant hash is stale")
    return variant, expected_hash


def input_variant_from_harness(meta, origin):
    """Authenticate the compact color fields emitted by the C++ harness."""
    if not isinstance(meta, dict):
        raise RuntimeError(f"{origin}: harness color provenance is missing")
    missing = [field for field in HARNESS_INPUT_FIELDS if field not in meta]
    if missing:
        raise RuntimeError(
            f"{origin}: harness color provenance lacks: " + ", ".join(missing)
        )
    color_mode = meta.get("color_mode")
    hdr_source_kind = meta.get("hdr_source_kind")
    white_raw = meta.get("sdr_white_level_raw")
    if not isinstance(white_raw, int) or isinstance(white_raw, bool):
        raise RuntimeError(f"{origin}: invalid HDR SDR-white provenance")
    if (color_mode == input_color.COLOR_MODE_SDR and
            hdr_source_kind == sbs_contract.HDR_SOURCE_SDR):
        variant = input_color.sdr_input_variant()
    elif (color_mode == input_color.COLOR_MODE_HDR and
          hdr_source_kind == sbs_contract.HDR_SOURCE_SIMULATED):
        try:
            variant = input_color.windows_hdr_input_variant(
                white_raw
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"{origin}: invalid HDR SDR-white provenance"
            ) from error
    elif (color_mode == input_color.COLOR_MODE_HDR and
          hdr_source_kind == sbs_contract.HDR_SOURCE_NATIVE_PQ):
        if white_raw != 0:
            raise RuntimeError(
                f"{origin}: native PQ input has simulated SDR-white provenance"
            )
        variant = input_color.native_pq_input_variant()
    else:
        raise RuntimeError(
            f"{origin}: unsupported harness color/source provenance"
        )
    expected = _input_variant_harness_context(variant)
    scale = meta.get("hdr_input_scale")
    if (not isinstance(scale, (int, float)) or isinstance(scale, bool) or
            not np.isfinite(float(scale))):
        raise RuntimeError(f"{origin}: invalid HDR input scale provenance")
    actual = {
        "color_mode": color_mode,
        "hdr_source_kind": hdr_source_kind,
        "metric_preview_encoding": meta.get("metric_preview_encoding"),
        "hdr_input_scale": float(scale),
        "sdr_white_level_raw": white_raw,
    }
    mismatch = {}
    for field in HARNESS_INPUT_FIELDS:
        if field == "hdr_input_scale":
            equal = np.isclose(
                expected[field], actual[field], rtol=0.0, atol=1e-9
            )
        else:
            equal = expected[field] == actual[field]
        if not equal:
            mismatch[field] = (expected[field], actual[field])
    if mismatch:
        raise RuntimeError(
            f"{origin}: harness input variant provenance differs: {mismatch}"
        )
    return variant


def validate_run_input_variant(payload, origin, expected_variant=None):
    """Require one exact input variant across every clip in a render run."""
    clips = payload.get("clips", {}) if isinstance(payload, dict) else {}
    if not isinstance(clips, dict) or not clips:
        raise RuntimeError(f"{origin}: render run has no clip color provenance")
    variants = {}
    for clip, entry in clips.items():
        meta = entry.get("meta", {}) if isinstance(entry, dict) else {}
        variant = input_variant_from_harness(meta, f"{origin}/{clip}")
        variants[input_color.input_variant_sha256(variant)] = variant
    if len(variants) != 1:
        raise RuntimeError(f"{origin}: render clips use different input variants")
    variant_hash, variant = next(iter(variants.items()))
    if expected_variant is not None:
        input_color.validate_input_variant(expected_variant)
        expected_hash = input_color.input_variant_sha256(expected_variant)
        if variant_hash != expected_hash:
            raise RuntimeError(
                f"{origin}: render input variant differs from source labels"
            )
    return variant


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


def validate_target_evidence_ids(label_frame_ids, selected_frame_ids, origin):
    """Validate sparse targets plus one available adjacent evidence frame."""
    selected = set(selected_frame_ids)
    required = set(label_frame_ids)
    for frame_id in label_frame_ids:
        if frame_id - 1 in selected:
            required.add(frame_id - 1)
        elif frame_id + 1 in selected:
            required.add(frame_id + 1)
        else:
            raise RuntimeError(
                f"{origin}: label frame {frame_id} has no adjacent evidence frame"
            )
    if selected != required:
        raise RuntimeError(
            f"{origin}: label selection contains unauthenticated evidence frames"
        )


def output_selection_contract(meta, origin):
    """Normalize the frame-selection provenance of one render-grid clip.

    Explicit modern contracts carry the exact emitted frame identities.  The
    legacy gt-right form remains readable so existing authored-stereo bundles
    can still be regenerated, but interval/all-frame renders are never valid
    temporal safe-ceiling supervision.
    """
    mode = meta.get("output_selection_mode")
    if mode is None:
        if meta.get("output_gt_right_only") is True:
            return {
                "mode": "gt-right",
                "label_frame_ids": None,
                "selected_frame_ids": None,
                "label_frames_sha256": "",
                "legacy": True,
            }
        raise RuntimeError(
            f"{origin}: policy labels require gt-right or label-frames output selection"
        )
    if mode not in {"gt-right", "label-frames"}:
        raise RuntimeError(f"{origin}: invalid policy output selection mode {mode!r}")
    if meta.get("output_interval") != 1:
        raise RuntimeError(f"{origin}: selected policy frames require output_interval=1")
    expected_gt_right = mode == "gt-right"
    if meta.get("output_gt_right_only") is not expected_gt_right:
        raise RuntimeError(
            f"{origin}: {mode} selection disagrees with output_gt_right_only"
        )
    selected_frame_ids = meta.get("output_selected_frame_ids")
    if (not isinstance(selected_frame_ids, list) or not selected_frame_ids or
            any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in selected_frame_ids) or
            selected_frame_ids != sorted(set(selected_frame_ids))):
        raise RuntimeError(
            f"{origin}: explicit output selection lacks exact increasing frame identities"
        )
    label_hash = meta.get("output_label_frames_sha256")
    if mode == "label-frames":
        if (not isinstance(label_hash, str) or len(label_hash) != 64 or
                any(char not in "0123456789abcdef" for char in label_hash)):
            raise RuntimeError(
                f"{origin}: label-frames selection lacks its raw-file SHA-256"
            )
        label_frame_ids = meta.get("label_frame_ids")
        if (not isinstance(label_frame_ids, list) or not label_frame_ids or
                any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in label_frame_ids) or
                label_frame_ids != sorted(set(label_frame_ids))):
            raise RuntimeError(
                f"{origin}: label selection lacks exact increasing target identities"
            )
        validate_target_evidence_ids(
            label_frame_ids, selected_frame_ids, origin
        )
    elif label_hash != "":
        raise RuntimeError(f"{origin}: gt-right selection has a label-frame hash")
    else:
        label_frame_ids = meta.get("label_frame_ids", [])
        if label_frame_ids not in (None, []):
            raise RuntimeError(f"{origin}: gt-right selection has label target IDs")
    return {
        "mode": mode,
        "label_frame_ids": tuple(label_frame_ids or ()),
        "selected_frame_ids": tuple(selected_frame_ids),
        "label_frames_sha256": label_hash,
        "legacy": False,
    }


def validate_source_render_selection(source, render_meta, origin):
    selection = output_selection_contract(render_meta, origin)
    frame = int(source["frame"])
    if source.get("source_contract") == GENERIC_SOURCE_CONTRACT:
        if selection["mode"] != "label-frames" or selection["legacy"]:
            raise RuntimeError(
                f"{origin}: generic mono/stereo source rows require label-frames rendering"
            )
        try:
            declared_ids = tuple(int(value) for value in source["label_frame_ids"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{origin}: source row has invalid label frame IDs") from error
        try:
            declared_selected_ids = tuple(
                int(value) for value in source["output_selected_frame_ids"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{origin}: source row has invalid evidence frame IDs") from error
        if (declared_ids != selection["label_frame_ids"] or
                declared_selected_ids != selection["selected_frame_ids"] or
                source.get("label_frames_sha256") !=
                selection["label_frames_sha256"]):
            raise RuntimeError(
                f"{origin}: render selection differs from the authenticated source targets"
            )
    elif selection["mode"] != "gt-right":
        raise RuntimeError(
            f"{origin}: legacy authored-stereo rows require gt-right selection"
        )
    if selection["mode"] == "label-frames":
        if frame not in selection["label_frame_ids"]:
            raise RuntimeError(f"{origin}: source frame {frame} was not a label target")
    elif (selection["selected_frame_ids"] is not None and
          frame not in selection["selected_frame_ids"]):
        raise RuntimeError(f"{origin}: source frame {frame} was not rendered")
    return selection


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
        input_variant_from_harness(meta, f"{label}/{clip}")
        validate_source_raster_contract(meta, meta)
        expected = {
            "harness_schema": EXPECTED_HARNESS_SCHEMA,
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
        if ("output_selection_mode" in run_meta or
                "output_selection_mode" in meta):
            if run_meta.get("output_selection_mode") != meta.get(
                    "output_selection_mode"):
                raise RuntimeError(
                    f"{label}/{clip}: clip output selection mode differs from run metadata"
                )
        output_selection_contract(meta, f"{label}/{clip}")
    mismatches = {
        field: (control_meta[field], candidate_meta[field])
        for field in CLIP_POLICY_CONTRACT_FIELDS
        if control_meta[field] != candidate_meta[field]
    }
    if mismatches:
        raise RuntimeError(
            f"{origin}/{clip}: clip policy/raster contract differs: {mismatches}"
        )
    control_selection = output_selection_contract(
        control_meta, f"control/{clip}"
    )
    candidate_selection = output_selection_contract(
        candidate_meta, f"{origin}/{clip}"
    )
    if control_selection != candidate_selection:
        raise RuntimeError(
            f"{origin}/{clip}: output frame selection differs from control"
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
    control_variant = validate_run_input_variant(control, "control")
    candidate_variant = validate_run_input_variant(candidate, origin)
    if (input_color.input_variant_sha256(control_variant) !=
            input_color.input_variant_sha256(candidate_variant)):
        raise RuntimeError(f"{origin}: render input variant differs from control")
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
    if "output_selection_mode" in control_meta or \
            "output_selection_mode" in candidate_meta:
        if control_meta.get("output_selection_mode") != candidate_meta.get(
                "output_selection_mode"):
            raise RuntimeError(f"{origin}: run output selection modes differ")
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
        "harness_schema": EXPECTED_HARNESS_SCHEMA,
        "eval_schema": 30,
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
    identity_violations = feasibility_violations(
        control_agg, identity, metric_specs, clip_meta
    )
    identity_pop = identity.get(EXACT_POP_METRIC)
    if identity_pop is None or not np.isfinite(identity_pop):
        raise RuntimeError(f"identity render has no finite {EXACT_POP_METRIC}")

    # A measured hard-bound failure at scale 1 is useful negative supervision:
    # the optional controller must abstain because no multiplier can make the
    # already-bad baseline an authenticated safe action.  Missing evidence or a
    # relative-regression mismatch is not a negative label; those remain broken
    # evaluations and fail closed.
    if identity_violations and not all(
            violation.endswith(":hard") for violation in identity_violations):
        raise RuntimeError(
            "identity render has incomplete or inconsistent feasibility evidence: " +
            ", ".join(identity_violations)
        )

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

    if identity_violations:
        return {
            "safe_scale_ceiling": 1.0,
            "ceiling_confidence": 0.0,
            "safety_margin_reliability": 0.0,
            "style_targets": {
                "clean": 1.0, "balanced": 1.0, "immersive": 1.0,
            },
            "authored_fit_scale": 1.0,
            "authored_fit_psnr": None,
            "connected_safe_scales": [],
            "safe_scale_min": 1.0,
            "safe_scale_max": 1.0,
            "identity_feasible": False,
            "identity_violations": identity_violations,
            "selection_reason": "identity-hard-failure-nonactionable",
            "identity_exact_pop_spread_pct": float(identity_pop),
            "candidate_grid": evidence,
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
        "identity_feasible": True,
        "identity_violations": [],
        "selection_reason": "identity-connected-safe-frontier",
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


def validate_source_input_variant(row, expected_variant, origin):
    """Bind a source row to its bundle's authenticated input identity."""
    legacy = (
        int(row.get("label_schema", 0)) == 7 and
        row.get("policy_contract") == "stereo-fit-source-v2"
    )
    variant, variant_hash = authenticated_input_variant(
        row, origin, allow_legacy_sdr=legacy
    )
    expected_hash = input_color.input_variant_sha256(expected_variant)
    if variant_hash != expected_hash:
        raise RuntimeError(f"{origin}: input variant differs from source bundle")
    if row.get("color_mode") != variant["color_mode"]:
        raise RuntimeError(f"{origin}: color mode differs from input variant")
    expected_source_kind = sbs_contract.input_variant_hdr_source_kind(variant)
    if row.get("hdr_source_kind") != expected_source_kind:
        raise RuntimeError(
            f"{origin}: HDR source kind differs from input variant"
        )
    sbs_contract.validate_metric_preview_encoding(
        row["color_mode"], row.get("metric_preview_encoding"), origin,
        expected_source_kind,
    )
    return variant


def source_label_contract(source_labels, control_meta):
    """Verify legacy stereo-fit or generic mono/stereo source provenance."""
    source_labels = Path(source_labels).resolve()
    fitter_path = source_labels.parent / "label_fitter_contract.json"
    generic_path = source_labels.parent / "source_contract.json"
    summary_path = source_labels.parent / "summary.json"
    if not summary_path.is_file() or (fitter_path.is_file() == generic_path.is_file()):
        raise RuntimeError(
            f"source bundle must have exactly one legacy/generic contract beside {source_labels}"
        )
    summary = load_json(summary_path)
    if summary.get("labels_sha256") != sha256(source_labels):
        raise RuntimeError("source label summary does not match labels.jsonl")
    if generic_path.is_file():
        contract = load_json(generic_path)
        if (contract.get("schema") != GENERIC_SOURCE_SCHEMA or
                summary.get("schema") != GENERIC_SOURCE_SCHEMA or
                contract.get("source_contract") != GENERIC_SOURCE_CONTRACT or
                summary.get("source_contract") != GENERIC_SOURCE_CONTRACT):
            raise RuntimeError("generic source rows use an unsupported contract")
        if summary.get("source_contract_sha256") != sha256(generic_path):
            raise RuntimeError("generic source contract hash is stale")
        run = contract.get("run_contract", {})
        contract_key = "source_contract"
        contract_path = generic_path
        allow_legacy_sdr = False
    else:
        contract = load_json(fitter_path)
        if int(contract.get("schema", 0)) != 7 or int(summary.get("schema", 0)) != 7:
            raise RuntimeError("source labels do not use the required schema-7 fitter")
        if summary.get("label_fitter_contract_sha256") != sha256(fitter_path):
            raise RuntimeError("source label fitter contract hash is stale")
        run = contract.get("run_contract", {})
        contract_key = "fitter_contract"
        contract_path = fitter_path
        allow_legacy_sdr = True
    if run.get("model") != control_meta.get("model"):
        raise RuntimeError("source label depth model differs from the render grid")
    if run.get("conf_sha256") != control_meta.get("conf_sha256"):
        raise RuntimeError("source label configuration differs from the render grid")
    input_variant, input_variant_hash = authenticated_input_variant(
        contract,
        "source label contract",
        allow_legacy_sdr=allow_legacy_sdr,
    )
    expected_color_hash = input_color.color_contract_sha256()
    color_hash = contract.get("depth_input_color_contract_sha256")
    if color_hash is None and allow_legacy_sdr:
        color_hash = expected_color_hash
    if color_hash != expected_color_hash:
        raise RuntimeError("source label input color contract hash is stale")
    return {
        "labels": {"path": str(source_labels), "sha256": sha256(source_labels)},
        "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        contract_key: {
            "path": str(contract_path), "sha256": sha256(contract_path)
        },
        "kind": "generic-source" if generic_path.is_file() else "legacy-stereo-fit",
        "run_contract": run,
        "input_variant": input_variant,
        "input_variant_sha256": input_variant_hash,
        "depth_input_color_contract_sha256": color_hash,
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
    input_variant_from_harness(contract, "source harness")
    if row.get("color_mode") != contract.get("color_mode"):
        raise RuntimeError("source row/harness has incompatible color mode")
    if row.get("hdr_source_kind") != contract.get("hdr_source_kind"):
        raise RuntimeError(
            "source row/harness has incompatible HDR source kind"
        )
    if (row.get("metric_preview_encoding") !=
            contract.get("metric_preview_encoding")):
        raise RuntimeError(
            "source row/harness has incompatible metric preview encoding"
        )
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


def validate_source_harness(row, policy_baseline, expected_variant=None):
    """Prove that the exact raw field uses the baseline later checked at runtime."""
    baseline_path = Path(row["baseline_disparity"]).resolve()
    contract_path = baseline_path.parent / "contract.json"
    if not contract_path.is_file():
        raise RuntimeError(f"source baseline has no harness contract: {contract_path}")
    if row.get("harness_contract_sha256") != sha256(contract_path):
        raise RuntimeError(f"source harness contract hash mismatch: {contract_path}")
    contract = load_json(contract_path)
    legacy = (int(row.get("label_schema", 0)) == 7 and
              row.get("policy_contract") == "stereo-fit-source-v2")
    generic = (row.get("source_schema") == GENERIC_SOURCE_SCHEMA and
               row.get("source_contract") == GENERIC_SOURCE_CONTRACT)
    if legacy == generic:
        raise RuntimeError("source row has an ambiguous or unsupported source contract")
    expected_variant = expected_variant or input_color.sdr_input_variant()
    try:
        input_color.validate_input_variant(expected_variant)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError("policy baseline has an invalid input variant") from error
    validate_source_input_variant(row, expected_variant, "source row")
    harness_variant = input_variant_from_harness(contract, str(contract_path))
    if (input_color.input_variant_sha256(harness_variant) !=
            input_color.input_variant_sha256(expected_variant)):
        raise RuntimeError("source harness input variant differs from policy baseline")
    if generic:
        label_frames_path = Path(row.get("label_frames", "")).resolve()
        frame_ids = row.get("label_frame_ids")
        selected_frame_ids = row.get("output_selected_frame_ids")
        invalid = (not label_frames_path.is_file() or
                   sha256(label_frames_path) != row.get("label_frames_sha256") or
                   not isinstance(frame_ids, list) or not frame_ids or
                   frame_ids != sorted(set(frame_ids)) or
                   any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                       for value in frame_ids) or
                   not isinstance(selected_frame_ids, list) or
                   selected_frame_ids != sorted(set(selected_frame_ids)) or
                   any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                       for value in selected_frame_ids) or
                   int(row.get("frame", -1)) not in frame_ids)
        if invalid:
            raise RuntimeError("generic source row has invalid temporal label-frame provenance")
        try:
            validate_target_evidence_ids(
                frame_ids, selected_frame_ids, "generic source row"
            )
        except RuntimeError as error:
            raise RuntimeError(
                "generic source row has invalid temporal label-frame provenance"
            ) from error
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


def identity_grid_artifacts(source, results_path, embedded_contract):
    """Load scale-1 disparity artifacts from the render grid geometry.

    The schema-7 source bundle is generated by a depth-only run whose output-eye
    geometry can differ from the render grid.  Schema 8 must describe the grid
    geometry, not that source-depth artifact, otherwise two output geometries
    silently collapse into identical training examples.
    """
    results_path = Path(results_path).resolve()
    clip = str(source["clip"])
    try:
        frame = int(source["frame"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("source label has an invalid frame identity") from error
    clip_root = results_path.parent / clip
    contract_path = clip_root / "contract.json"
    if not contract_path.is_file():
        raise RuntimeError(f"identity render grid has no harness contract: {contract_path}")
    contract = load_json(contract_path)
    _, source_variant_hash = authenticated_input_variant(
        source,
        "source row",
        allow_legacy_sdr=(
            int(source.get("label_schema", 0)) == 7 and
            source.get("policy_contract") == "stereo-fit-source-v2"
        ),
    )
    contract_variant = input_variant_from_harness(contract, str(contract_path))
    if (source_variant_hash !=
            input_color.input_variant_sha256(contract_variant)):
        raise RuntimeError(
            "identity render-grid input variant differs from source labels"
        )
    normalized_contract = dict(contract)
    normalized_contract["harness_schema"] = normalized_contract.pop("schema", None)
    mismatches = {
        field: (embedded_contract.get(field), normalized_contract.get(field))
        for field in CLIP_POLICY_CONTRACT_FIELDS
        if embedded_contract.get(field) != normalized_contract.get(field)
    }
    if mismatches:
        raise RuntimeError(
            f"identity render-grid contract differs from results.json: "
            f"{contract_path}: {mismatches}"
        )
    if output_selection_contract(
            embedded_contract, f"results/{clip}") != output_selection_contract(
                normalized_contract, str(contract_path)):
        raise RuntimeError(
            f"identity render-grid output selection differs from results.json: {contract_path}"
        )
    for field in ("source_width", "source_height", "model_input_width",
                  "model_input_height"):
        if int(source.get(field, 0)) != int(contract.get(field, -1)):
            raise RuntimeError(
                f"identity render-grid {field} differs from source labels"
            )
    if source.get("color_mode") != contract.get("color_mode"):
        raise RuntimeError("identity render-grid color mode differs from source labels")
    if source.get("hdr_source_kind") != contract.get("hdr_source_kind"):
        raise RuntimeError(
            "identity render-grid HDR source kind differs from source labels"
        )
    if (source.get("metric_preview_encoding") !=
            contract.get("metric_preview_encoding")):
        raise RuntimeError(
            "identity render-grid metric preview encoding differs from source labels"
        )
    if (int(contract.get("eye_width", 0)) <= 0 or
            int(contract.get("eye_height", 0)) <= 0 or
            int(contract.get("disparity_raster_width", 0)) !=
            int(contract.get("eye_width", -1)) or
            int(contract.get("disparity_raster_height", 0)) !=
            int(contract.get("eye_height", -1))):
        raise RuntimeError("identity render-grid disparity raster is invalid")
    suffix = f"{frame:05d}.f32"
    clamped_path = clip_root / f"warp_disparity_{suffix}"
    unclamped_path = clip_root / f"warp_unclamped_disparity_{suffix}"
    for path, description in (
        (clamped_path, "clamped"), (unclamped_path, "unclamped")
    ):
        if not path.is_file():
            raise RuntimeError(
                f"identity render grid is missing {description} disparity: {path}"
            )
    clamped = sbsbench.load_float_texture(clamped_path)
    unclamped = sbsbench.load_float_texture(unclamped_path)
    expected_shape = (
        int(contract["disparity_raster_height"]),
        int(contract["disparity_raster_width"]),
    )
    for raster, description in ((clamped, "clamped"),
                                (unclamped, "unclamped")):
        if (raster.ndim != 2 or tuple(raster.shape) != expected_shape or
                not np.isfinite(raster).all()):
            raise RuntimeError(
                f"identity render-grid {description} disparity is invalid: "
                f"{raster.shape} != {expected_shape}"
            )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "clamped_path": clamped_path,
        "clamped_sha256": sha256(clamped_path),
        "clamped": clamped,
        "unclamped_path": unclamped_path,
        "unclamped_sha256": sha256(unclamped_path),
        "unclamped": unclamped,
    }


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
    source_contract = source_label_contract(
        source_labels, control.get("meta", {})
    )
    input_variant = source_contract["input_variant"]
    input_variant_hash = source_contract["input_variant_sha256"]
    for index, row in enumerate(rows, 1):
        validate_source_input_variant(
            row, input_variant, f"{source_labels}:{index}"
        )
    validate_run_input_variant(control, "control", input_variant)
    policy_baseline = policy_baseline_from_meta(control.get("meta", {}))
    candidate_runs = {}
    candidate_paths = {}
    candidate_sources = []
    for scale, path in candidate_specs:
        if scale in candidate_runs:
            raise RuntimeError(f"duplicate candidate scale: {scale}")
        payload = load_json(path)
        validate_context(control, payload, scale, path)
        candidate_runs[scale] = payload
        candidate_paths[scale] = Path(path).resolve()
        candidate_sources.append({
            "scale": scale,
            "path": str(path),
            "sha256": sha256(path),
            "input_variant_sha256": input_variant_hash,
            **_input_variant_harness_context(input_variant),
        })
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
        "input_variant": input_variant,
        "input_variant_sha256": input_variant_hash,
        "depth_input_color_contract_sha256":
            input_color.color_contract_sha256(),
        **_input_variant_harness_context(input_variant),
        "clips": decisions,
    }
    evidence_path = output / "render_grid_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_rows = []
    for source in rows:
        validate_source_harness(source, policy_baseline, input_variant)
        decision = decisions[source["clip"]]
        safe_scale_ceiling = decision["safe_scale_ceiling"]
        ceiling_confidence = decision["ceiling_confidence"]
        margin_reliability = decision["safety_margin_reliability"]
        identity_contract = candidate_runs[1.0]["clips"][source["clip"]]["meta"]
        validate_source_render_selection(
            source, identity_contract, f"identity/{source['clip']}"
        )
        grid = identity_grid_artifacts(
            source, candidate_paths[1.0], identity_contract
        )
        disparity = grid["clamped"]
        raw_disparity = grid["unclamped"]
        harness_contract = grid["contract"]
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
            source_width = int(harness_contract["source_width"])
            source_height = int(harness_contract["source_height"])
            clamp_abs = float(harness_contract["artistic_full_clamp_abs"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "source label lacks exact source-aspect/clamp provenance"
            ) from error
        if source_width <= 0 or source_height <= 0:
            raise RuntimeError("source label has invalid source geometry")
        # The artifact is already output-eye normalized. Match perceived_disparity_pct's
        # reference-aspect conversion without applying content_scale_x a second time.
        perceived_scale = (
            (float(harness_contract["eye_width"]) /
             float(harness_contract["eye_height"])) /
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
        if source.get("source_contract") == GENERIC_SOURCE_CONTRACT:
            stereo_fit_multiplier = None
            stereo_fit_confidence = None
        else:
            stereo_fit_multiplier = source.get(
                "stereo_fit_multiplier", source["baseline_multiplier"]
            )
            stereo_fit_confidence = source.get(
                "stereo_fit_confidence", source["confidence"]
            )
        row.update({
            "label_schema": 8,
            "policy_contract": POLICY_CONTRACT,
            "stereo_fit_multiplier": stereo_fit_multiplier,
            "stereo_fit_confidence": stereo_fit_confidence,
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
            "identity_feasible": decision["identity_feasible"],
            "identity_violations": decision["identity_violations"],
            "selection_reason": decision["selection_reason"],
            "authored_fit_scale": decision["authored_fit_scale"],
            "authored_fit_psnr": decision["authored_fit_psnr"],
            "render_grid_key": source["clip"],
            "safe_ceiling_render_target": ceiling_render_target,
            "safe_ceiling_exact_pop_spread_pct": ceiling_render_target[
                "exact_pop_spread_pct"
            ],
            "baseline_disparity_mean_abs_pct": disparity_mean_abs_pct,
            "baseline_disparity_p95_pct": disparity_p95_pct,
            "source_depth_baseline_disparity": source["baseline_disparity"],
            "source_depth_baseline_disparity_sha256": source.get(
                "baseline_disparity_sha256"
            ),
            "source_depth_harness_contract_sha256": source.get(
                "harness_contract_sha256"
            ),
            "baseline_disparity": str(grid["clamped_path"]),
            "baseline_disparity_sha256": grid["clamped_sha256"],
            "baseline_unclamped_disparity": str(grid["unclamped_path"]),
            "baseline_unclamped_disparity_sha256": grid["unclamped_sha256"],
            "harness_contract_sha256": sha256(grid["contract_path"]),
            "source_width": int(harness_contract["source_width"]),
            "source_height": int(harness_contract["source_height"]),
            "model_input_width": int(harness_contract["model_input_width"]),
            "model_input_height": int(harness_contract["model_input_height"]),
            "eye_width": int(harness_contract["eye_width"]),
            "eye_height": int(harness_contract["eye_height"]),
            "content_scale_x": float(harness_contract["content_scale_x"]),
            "content_scale_y": float(harness_contract["content_scale_y"]),
            "disparity_raster_width": int(
                harness_contract["disparity_raster_width"]
            ),
            "disparity_raster_height": int(
                harness_contract["disparity_raster_height"]
            ),
            "color_mode": harness_contract["color_mode"],
            "hdr_source_kind": harness_contract["hdr_source_kind"],
            "metric_preview_encoding": harness_contract[
                "metric_preview_encoding"
            ],
            "hdr_input_scale": float(harness_contract["hdr_input_scale"]),
            "sdr_white_level_raw": int(
                harness_contract["sdr_white_level_raw"]
            ),
            "input_variant": input_variant,
            "input_variant_sha256": input_variant_hash,
            "depth_input_color_contract_sha256":
                input_color.color_contract_sha256(),
            "artistic_full_clamp_abs": clamp_abs,
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
        "depth_input_color": THIS_DIR / "depth_input_color.py",
        "depth_input_color_contract":
            THIS_DIR / "depth_input_color_contract.json",
        "label_preparation": Path(__file__).resolve(),
        "image_loader": SBSBENCH_DIR / "sbsbench.py",
        "evaluator_runner": SBSBENCH_DIR / "run_eval.py",
    }
    contract = {
        "schema": 8,
        "label_fitter": "exact-apollo-connected-safe-frontier",
        "policy_contract": POLICY_CONTRACT,
        "policy_baseline": policy_baseline,
        "input_variant": input_variant,
        "input_variant_sha256": input_variant_hash,
        "depth_input_color_contract_sha256":
            input_color.color_contract_sha256(),
        **_input_variant_harness_context(input_variant),
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
            "identity_hard_failure": (
                "a fully measured hard-bound failure at scale 1 is retained "
                "as a confidence-zero no-op negative; every sampled candidate "
                "remains disconnected and identity is not declared feasible"
            ),
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
        "identity_feasible_clips": sum(
            1 for decision in decisions.values()
            if decision["identity_feasible"]
        ),
        "identity_infeasible_clips": sum(
            1 for decision in decisions.values()
            if not decision["identity_feasible"]
        ),
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
        "input_variant_sha256": input_variant_hash,
        **_input_variant_harness_context(input_variant),
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
