#!/usr/bin/env python3
"""Strict readers for the diagnostic depth-coordinate-v2 JSON dump files.

The JSON serialization schemas are deliberately independent of the GPU coordinate-contract
schema.  Each document also binds the exact generated contract schema/tag and complete physical
field layout, so a consumer cannot silently reinterpret a same-sized state buffer.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Dict

try:
    from . import depth_coordinate_v2_contract as coordinate_contract
    from . import generate_depth_coordinate_v2_contract as generator
except ImportError:  # Direct script/module loading from tools/sbsbench.
    import depth_coordinate_v2_contract as coordinate_contract  # type: ignore
    import generate_depth_coordinate_v2_contract as generator  # type: ignore


DUMP_MANIFEST_SCHEMA = 13
DEPTH_INPUT_REGION_SCHEMA = 1
WINDOW_VIDEO_BORDER_SCHEMA = 2
WINDOW_VIDEO_OBSERVER_STATUSES = frozenset({
    "starting", "ok", "no-foreground", "unsupported", "unavailable",
    "accessibility", "warming", "incomplete", "changed", "no-video",
    "ambiguous", "helper-missing", "launch-failed", "helper-exited",
    "protocol-error", "stale", "stopped",
})
WINDOW_VIDEO_MAPPING_STATUSES = frozenset({
    "ok", "invalid-video-rect", "invalid-capture-rect", "unsupported-rotation",
    "extent-mismatch", "foreground-mismatch", "outside-capture",
})
SHADOW_STATE_DUMP_SCHEMA = 16
SHADOW_FRAME_STATS_DUMP_SCHEMA = 2
LIVE_RENDERER_SOURCE_CLOSURE_SHA256 = (
    "115ddcf1cf8058064516421d9ea2c0d34631af999184ef62effe5ad9cd28e79e"
)
DIAGNOSTIC_SOURCE_CLOSURE_SHA256 = (
    "7977b2e9adaf33e24b091af7acf2377f1c52300246f27c76a2d39f4189046fe8"
)

_CONTRACT = coordinate_contract.load_contract()
_CONTRACT_TAG = generator.contract_tag(_CONTRACT)
_STATE_FIELDS = tuple(_CONTRACT["shadow_state"]["fields"])
_FRAME_STATS_FIELDS = tuple(_CONTRACT["frame_stats"]["fields"])
_DEFAULTS = coordinate_contract.CALIBRATED_DEFAULTS
_RESERVED_CALIBRATION_REVISION = 0xFFFFFFFF
_MAXIMUM_VIDEO_REGION_TRIM_FRACTION = 0.02
_ROI_EXTERIOR_ZERO_TOLERANCE_OUTPUT_EYE_PX = 0.005
_PRODUCTION_DEPTH_SHORT_SIDE = 432
_PRODUCTION_DEPTH_MAX_ASPECT = 4.0
_PRODUCTION_CALIBRATION = coordinate_contract.MODEL_CALIBRATIONS[0]
_AUTHENTICATED_TENSOR_SHAPES = frozenset(
    _PRODUCTION_CALIBRATION.calibrated_input_shapes)
_MAXIMUM_SOURCE_LONG_SIDE = 5120
_MAXIMUM_SOURCE_PIXELS = 5120 * 2160


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]

_STATE_ROOT_KEYS = {
    "schema", "coordinate_contract", "source", "capture", "rendered_output_selected",
    "wire_contract", "units", "constants", "fields", "named_values", "decoded",
    "adaptation_semantics",
}
_STATE_CONSTANT_KEYS = {
    "raw_coordinate_scale", "collapse_abs_epsilon", "far_tau", "near_log_tau",
    "gain_per_pop",
    "reference_pop_strength", "reference_gain_at_reference_pop", "requested_gain",
    "requested_pop_strength", "direct_container_limit", "max_horizontal_slope",
    "max_vertical_shear", "convergence_curve_default",
    "vertical_majorant_share",
}
_DECODED_KEYS = {
    "frame_valid", "camera_valid", "calibration_revision", "confirmed_cut_count", "contract_tag",
    "requested_gain", "requested_pop_strength", "latched_scale",
    "convergence_curve", "container_scale", "effective_gain",
    "camera_center_integrity_bits", "renderer_authorization_bits",
}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _uint32(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{label} must be a uint32")
    return value


def _uint64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{label} must be a uint64")
    return value


def _int32(value: Any, label: str) -> int:
    if (isinstance(value, bool) or not isinstance(value, int) or
            not -0x80000000 <= value <= 0x7FFFFFFF):
        raise ValueError(f"{label} must be an int32")
    return value


def _canonical_hwnd(value: Any, label: str) -> tuple[str, int]:
    if (not isinstance(value, str) or not value.startswith("0x") or len(value) < 3 or
            value[2] == "0" or any(character not in "0123456789ABCDEF" for character in value[2:])):
        raise ValueError(f"{label} must be canonical uppercase hexadecimal")
    parsed = int(value, 16)
    if parsed <= 0 or parsed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{label} is outside uintptr range")
    return value, parsed


def _pixel_extent(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, dict) or set(value) != {"width", "height"}:
        raise ValueError(f"{label} must be a width/height object")
    width = _uint32(value.get("width"), f"{label} width")
    height = _uint32(value.get("height"), f"{label} height")
    if width == 0 or height == 0:
        raise ValueError(f"{label} must be non-empty")
    return width, height


def _pixel_rect(value: Any, label: str) -> tuple[int, int, int, int]:
    if not isinstance(value, dict) or set(value) != {"left", "top", "right", "bottom"}:
        raise ValueError(f"{label} must be a half-open rectangle object")
    left = _uint32(value.get("left"), f"{label} left")
    top = _uint32(value.get("top"), f"{label} top")
    right = _uint32(value.get("right"), f"{label} right")
    bottom = _uint32(value.get("bottom"), f"{label} bottom")
    if right <= left or bottom <= top:
        raise ValueError(f"{label} must be non-empty")
    return left, top, right, bottom


def _fit_host_sbs_v2_depth_tensor_shape(width: int, height: int) -> tuple[int, int]:
    """Small dependency-free mirror of the production patch-aligned shape fitter."""

    patch = _PRODUCTION_CALIBRATION.preprocess.patch_multiple
    maximum = _PRODUCTION_CALIBRATION.preprocess.maximum_dimension

    def round_to_patch(value: float) -> int:
        # All inputs are positive. floor(x + .5) therefore mirrors C++ std::round.
        quotient = _float32(_float32(value) / _float32(float(patch)))
        return max(patch, int(math.floor(quotient + 0.5)) * patch)

    if width <= 0 or height <= 0:
        return 0, 0
    aspect = _float32(_float32(float(width)) / _float32(float(height)))
    max_aspect = _float32(_PRODUCTION_DEPTH_MAX_ASPECT)
    if aspect >= 1.0:
        fitted_aspect = min(aspect, max_aspect)
    else:
        inverse_aspect = _float32(_float32(1.0) / aspect)
        fitted_aspect = _float32(_float32(1.0) / min(inverse_aspect, max_aspect))
    bounded_short = min(max(_PRODUCTION_DEPTH_SHORT_SIDE, patch), maximum)
    max_width = max(patch, (min(width, maximum) // patch) * patch)
    max_height = max(patch, (min(height, maximum) // patch) * patch)
    requested_short = round_to_patch(float(bounded_short))
    if fitted_aspect >= 1.0:
        for candidate_height in range(min(requested_short, max_height), patch - 1, -patch):
            candidate_width = round_to_patch(
                _float32(_float32(float(candidate_height)) * fitted_aspect))
            if candidate_width <= max_width:
                return candidate_width, candidate_height
    else:
        for candidate_width in range(min(requested_short, max_width), patch - 1, -patch):
            candidate_height = round_to_patch(
                _float32(_float32(float(candidate_width)) / fitted_aspect))
            if candidate_height <= max_height:
                return candidate_width, candidate_height
    return patch, patch


def _plan_host_sbs_v2_video_region(
        semantic: tuple[int, int, int, int], source_width: int, source_height: int,
        tensor_width: int, tensor_height: int
        ) -> tuple[tuple[int, int, int, int], float] | None:
    """Exact integer/float32 mirror of the production inward-only ROI planner."""

    left, top, right, bottom = semantic
    width = right - left
    height = bottom - top
    if (width <= 0 or height <= 0 or right > source_width or bottom > source_height or
            (tensor_width, tensor_height) not in _AUTHENTICATED_TENSOR_SHAPES or
            semantic == (0, 0, source_width, source_height)):
        return None

    def source_supported(candidate_width: int, candidate_height: int) -> bool:
        return (max(candidate_width, candidate_height) <= _MAXIMUM_SOURCE_LONG_SIDE and
                candidate_width * candidate_height <= _MAXIMUM_SOURCE_PIXELS and
                _fit_host_sbs_v2_depth_tensor_shape(candidate_width, candidate_height) in
                    _AUTHENTICATED_TENSOR_SHAPES)

    exact_shape = _fit_host_sbs_v2_depth_tensor_shape(width, height)
    if (exact_shape == (tensor_width, tensor_height) and
            width * tensor_height == height * tensor_width and
            source_supported(width, height)):
        return semantic, 0.0

    fitted_width = width
    fitted_height = height
    if width * tensor_height > height * tensor_width:
        fitted_width = height * tensor_width // tensor_height
    elif width * tensor_height < height * tensor_width:
        fitted_height = width * tensor_height // tensor_width
    if (fitted_width > width or fitted_height > height or
            fitted_width < tensor_width or fitted_height < tensor_height):
        return None
    trimmed_fraction = _float32(
        1.0 - (fitted_width * fitted_height) / (width * height))
    if trimmed_fraction > _float32(_MAXIMUM_VIDEO_REGION_TRIM_FRACTION):
        return None
    remove_x = width - fitted_width
    remove_y = height - fitted_height
    inference = (
        left + remove_x // 2,
        top + remove_y // 2,
        left + remove_x // 2 + fitted_width,
        top + remove_y // 2 + fitted_height,
    )
    if (_fit_host_sbs_v2_depth_tensor_shape(fitted_width, fitted_height) !=
            (tensor_width, tensor_height) or
            not source_supported(fitted_width, fitted_height)):
        return None
    return inference, trimmed_fraction


def _roi_renderer_constants(
        inference: tuple[int, int, int, int], source_width: int, source_height: int,
        tensor_width: int, tensor_height: int) -> tuple[float, float]:
    """Mirror the production constant-buffer/shader float32 instruction order."""

    left, top, right, bottom = inference
    width = _float32(float(source_width))
    height = _float32(float(source_height))
    roi_left = _float32(_float32(float(left)) / width)
    roi_top = _float32(_float32(float(top)) / height)
    roi_right = _float32(_float32(float(right)) / width)
    roi_bottom = _float32(_float32(float(bottom)) / height)
    roi_width = _float32(roi_right - roi_left)
    roi_height = _float32(roi_bottom - roi_top)
    source_height_in_source_u = _float32(height / max(width, _float32(1.0)))
    roi_pixel_aspect = _float32(
        _float32(roi_width * width) /
        max(_float32(roi_height * height), _float32(1.0e-6)))
    vertical_slope = _float32(
        _float32(_float32(_DEFAULTS.max_vertical_shear) * roi_pixel_aspect) *
        _float32(_float32(float(tensor_height)) /
                 max(_float32(float(tensor_width)), _float32(1.0))))
    vertical_budget = _float32(vertical_slope * source_height_in_source_u)
    return roi_width, vertical_budget


def validate_depth_input_region_document(
        document: Any, *, matched_frame_id: int | None = None,
        source_width: int | None = None, source_height: int | None = None,
        tensor_width: int | None = None, tensor_height: int | None = None) -> Dict[str, Any]:
    """Validate the authoritative analysis domain and its full-source renderer placement."""

    if not isinstance(document, dict) or set(document) != {
            "schema", "capture", "role", "matched_frame_id", "mode",
            "coordinate_space", "analysis", "renderer", "authorization"}:
        raise ValueError("depth_input_region.json has an unknown layout")
    if document.get("schema") != DEPTH_INPUT_REGION_SCHEMA:
        raise ValueError("depth_input_region.json has an unknown serialization schema")
    if document.get("capture") != (
            "same matched source/color/model/depth/render frame as the parent Dump 3D package"):
        raise ValueError("depth_input_region.json has unknown capture semantics")
    if document.get("role") != (
            "authoritative analysis-domain placement and live-render embedding contract"):
        raise ValueError("depth_input_region.json claims unknown authority")

    frame_id = _uint64(document.get("matched_frame_id"), "depth input matched frame id")
    if frame_id == 0 or (matched_frame_id is not None and frame_id != matched_frame_id):
        raise ValueError("depth_input_region.json does not match the parent frame")
    mode = document.get("mode")
    if mode not in {"full-source", "video-region"}:
        raise ValueError("depth_input_region.json has an unknown analysis mode")

    coordinate = document.get("coordinate_space")
    if not isinstance(coordinate, dict) or set(coordinate) != {
            "name", "rect_semantics", "source_extent_px", "semantic_rect_px",
            "inference_rect_px"}:
        raise ValueError("depth_input_region.json has an unknown coordinate layout")
    if (coordinate.get("name") != "matched-source-pixels" or
            coordinate.get("rect_semantics") != "half-open [left, top, right, bottom)"):
        raise ValueError("depth_input_region.json has unknown rectangle semantics")
    width, height = _pixel_extent(
        coordinate.get("source_extent_px"), "depth input source extent")
    if ((source_width is not None and width != source_width) or
            (source_height is not None and height != source_height)):
        raise ValueError("depth_input_region.json source extent does not match the dump")
    inference = _pixel_rect(
        coordinate.get("inference_rect_px"), "depth input inference rectangle")
    if inference[2] > width or inference[3] > height:
        raise ValueError("depth_input_region.json inference rectangle is outside the source")
    semantic_value = coordinate.get("semantic_rect_px")
    semantic = None if semantic_value is None else _pixel_rect(
        semantic_value, "depth input semantic rectangle")
    if semantic is not None and (semantic[2] > width or semantic[3] > height):
        raise ValueError("depth_input_region.json semantic rectangle is outside the source")

    authorization_value = document.get("authorization")
    authorization = None
    if authorization_value is not None:
        if not isinstance(authorization_value, dict) or set(authorization_value) != {
                "observer_generation", "hwnd", "process_id", "document_id", "video_id"}:
            raise ValueError("depth_input_region.json has an unknown authorization layout")
        hwnd, hwnd_value = _canonical_hwnd(
            authorization_value.get("hwnd"), "depth input HWND")
        authorization = {
            "observer_generation": _uint64(
                authorization_value.get("observer_generation"), "observer generation"),
            "hwnd": hwnd_value,
            "hwnd_text": hwnd,
            "process_id": _uint32(authorization_value.get("process_id"), "process id"),
            "document_id": _int32(authorization_value.get("document_id"), "document id"),
            "video_id": _int32(authorization_value.get("video_id"), "video id"),
        }
        if any(value == 0 for key, value in authorization.items() if key != "hwnd_text"):
            raise ValueError("depth_input_region.json has incomplete ROI authorization")

    analysis = document.get("analysis")
    if not isinstance(analysis, dict) or set(analysis) != {
            "analysis_generation", "tensor_extent_px", "trimmed_area_fraction",
            "crop_method", "scene_analysis_domain", "input_domain_reset"}:
        raise ValueError("depth_input_region.json has an unknown analysis layout")
    generation = _uint64(analysis.get("analysis_generation"), "analysis generation")
    tensor_w, tensor_h = _pixel_extent(
        analysis.get("tensor_extent_px"), "depth input tensor extent")
    if ((tensor_width is not None and tensor_w != tensor_width) or
            (tensor_height is not None and tensor_h != tensor_height)):
        raise ValueError("depth_input_region.json tensor extent does not match dumped geometry")
    trimmed_fraction = _finite_number(
        analysis.get("trimmed_area_fraction"), "trimmed area fraction")
    if not isinstance(analysis.get("input_domain_reset"), bool):
        raise ValueError("depth_input_region.json input_domain_reset must be boolean")

    renderer = document.get("renderer")
    if not isinstance(renderer, dict) or set(renderer) != {
            "final_parallax_units", "full_source_parallax_scale",
            "inside_inference_rect", "outside", "source_sampling", "inverse_iterations"}:
        raise ValueError("depth_input_region.json has an unknown renderer layout")
    scale = _finite_number(
        renderer.get("full_source_parallax_scale"), "full-source parallax scale")
    if (renderer.get("inside_inference_rect") != "no taper" or
            renderer.get("source_sampling") !=
                "full matched source; never clamp to inference rectangle" or
            renderer.get("inverse_iterations") != 11):
        raise ValueError("depth_input_region.json has unknown renderer semantics")

    inference_width = inference[2] - inference[0]
    inference_height = inference[3] - inference[1]
    if mode == "full-source":
        if (semantic is not None or authorization is not None or
                inference != (0, 0, width, height) or generation != 0 or
                trimmed_fraction != 0.0 or analysis.get("crop_method") != "full-source" or
                analysis.get("scene_analysis_domain") != "full-source" or
                renderer.get("final_parallax_units") != "full-source-u" or
                scale != 1.0 or renderer.get("outside") is not None):
            raise ValueError("depth_input_region.json has inconsistent full-source semantics")
    else:
        if (semantic is None or authorization is None or generation == 0 or
                inference == (0, 0, width, height)):
            raise ValueError("depth_input_region.json has incomplete video-region authority")
        expected_plan = _plan_host_sbs_v2_video_region(
            semantic, width, height, tensor_w, tensor_h)
        if expected_plan is None or expected_plan[0] != inference:
            raise ValueError(
                "depth_input_region.json is not the deterministic authenticated inward fit")
        expected_trim = expected_plan[1]
        expected_scale, expected_vertical_slope = _roi_renderer_constants(
            inference, width, height, tensor_w, tensor_h)
        if (trimmed_fraction < 0.0 or
                not math.isclose(trimmed_fraction, expected_trim, rel_tol=0.0, abs_tol=1.0e-7) or
                analysis.get("crop_method") != "same-format D3D11 CopySubresourceRegion" or
                analysis.get("scene_analysis_domain") != "inference-rectangle-only" or
                renderer.get("final_parallax_units") != "roi-local-source-u" or
                scale != expected_scale):
            raise ValueError("depth_input_region.json has inconsistent video-region analysis")
        outside = renderer.get("outside")
        if not isinstance(outside, dict) or set(outside) != {
                "construction", "horizontal_slope_source_u_per_source_u",
                "vertical_slope_source_u_per_source_v", "beyond_collar"}:
            raise ValueError("depth_input_region.json has an unknown exterior collar layout")
        horizontal_slope = _finite_number(
            outside.get("horizontal_slope_source_u_per_source_u"),
            "horizontal collar slope")
        vertical_slope = _finite_number(
            outside.get("vertical_slope_source_u_per_source_v"),
            "vertical collar slope")
        if (outside.get("construction") != "signed soft-threshold collar" or
                outside.get("beyond_collar") != "exact zero parallax" or
                horizontal_slope != _float32(_DEFAULTS.max_horizontal_slope) or
                vertical_slope != expected_vertical_slope):
            raise ValueError("depth_input_region.json has inconsistent exterior collar semantics")

    return {
        "matched_frame_id": frame_id,
        "mode": mode,
        "source_width": width,
        "source_height": height,
        "semantic_rect": semantic,
        "inference_rect": inference,
        "inference_width": inference_width,
        "inference_height": inference_height,
        "tensor_width": tensor_w,
        "tensor_height": tensor_h,
        "analysis_generation": generation,
        "authorization": authorization,
        "trimmed_area_fraction": trimmed_fraction,
        "input_domain_reset": analysis["input_domain_reset"],
        "full_source_parallax_scale": scale,
        "horizontal_slope_source_u_per_source_u": (
            None if mode == "full-source" else horizontal_slope),
        "vertical_slope_source_u_per_source_v": (
            None if mode == "full-source" else vertical_slope),
    }


def validate_window_video_border_document(
        document: Any, *, matched_frame_id: int | None = None,
        source_width: int | None = None, source_height: int | None = None) -> Dict[str, Any]:
    """Validate optional, diagnostic-only matched-source video-border evidence."""

    root_keys = {
        "schema", "capture", "role", "matched_frame_id", "coordinate_space",
        "identity", "freshness",
    }
    if not isinstance(document, dict) or set(document) != root_keys:
        raise ValueError("window_video_border.json has an unknown layout")
    if document.get("schema") != WINDOW_VIDEO_BORDER_SCHEMA:
        raise ValueError("window_video_border.json has an unknown serialization schema")
    if document.get("capture") != (
            "same matched source/color/depth/render frame as the parent Dump 3D package"):
        raise ValueError("window_video_border.json has unknown capture semantics")
    if document.get("role") != (
            "diagnostic-only window-video border evidence; no geometry or renderer authority"):
        raise ValueError("window_video_border.json claims unknown authority")

    frame_id = _uint64(document.get("matched_frame_id"), "matched frame id")
    if frame_id == 0 or (matched_frame_id is not None and frame_id != matched_frame_id):
        raise ValueError("window_video_border.json does not match the parent frame")

    coordinate = document.get("coordinate_space")
    if not isinstance(coordinate, dict) or set(coordinate) != {
            "name", "rect_semantics", "source_extent_px", "capture_rect_px"}:
        raise ValueError("window_video_border.json has an unknown coordinate layout")
    if (coordinate.get("name") != "matched-source-pixels" or
            coordinate.get("rect_semantics") != "half-open [left, top, right, bottom)"):
        raise ValueError("window_video_border.json has unknown rectangle semantics")
    extent = coordinate.get("source_extent_px")
    rect = coordinate.get("capture_rect_px")
    if (not isinstance(extent, dict) or set(extent) != {"width", "height"} or
            not isinstance(rect, dict) or set(rect) != {"left", "top", "right", "bottom"}):
        raise ValueError("window_video_border.json has malformed capture geometry")
    width = _uint32(extent.get("width"), "source width")
    height = _uint32(extent.get("height"), "source height")
    if (width == 0 or height == 0 or
            (source_width is not None and width != source_width) or
            (source_height is not None and height != source_height)):
        raise ValueError("window_video_border.json source extent does not match the dump")
    left = _int32(rect.get("left"), "rectangle left")
    top = _int32(rect.get("top"), "rectangle top")
    right = _int32(rect.get("right"), "rectangle right")
    bottom = _int32(rect.get("bottom"), "rectangle bottom")
    if left < 0 or top < 0 or right <= left or bottom <= top or right > width or bottom > height:
        raise ValueError("window_video_border.json rectangle is empty or out of bounds")

    identity = document.get("identity")
    if not isinstance(identity, dict) or set(identity) != {
            "hwnd", "process_id", "document_id", "video_id", "generation"}:
        raise ValueError("window_video_border.json has an unknown identity layout")
    hwnd, hwnd_value = _canonical_hwnd(identity.get("hwnd"), "window-video HWND")
    process_id = _uint32(identity.get("process_id"), "process id")
    document_id = _int32(identity.get("document_id"), "document id")
    video_id = _int32(identity.get("video_id"), "video id")
    generation = _uint64(identity.get("generation"), "generation")
    if hwnd_value <= 0 or process_id == 0 or document_id == 0 or video_id == 0 or generation == 0:
        raise ValueError("window_video_border.json has an incomplete identity")

    freshness = document.get("freshness")
    if not isinstance(freshness, dict) or set(freshness) != {
            "latest_heartbeat_age_ms_at_capture", "maximum_heartbeat_age_ms",
            "geometry_continuity_ms_at_capture", "source_content_age_ms_at_capture",
            "fresh", "causal_geometry"}:
        raise ValueError("window_video_border.json has an unknown freshness layout")
    heartbeat_age_ms = _uint32(
        freshness.get("latest_heartbeat_age_ms_at_capture"), "latest heartbeat age")
    maximum_heartbeat_age_ms = _uint32(
        freshness.get("maximum_heartbeat_age_ms"), "maximum heartbeat age")
    geometry_continuity_ms = _uint64(
        freshness.get("geometry_continuity_ms_at_capture"), "geometry continuity")
    source_content_age_ms = _uint64(
        freshness.get("source_content_age_ms_at_capture"), "source content age")
    if (freshness.get("fresh") is not True or maximum_heartbeat_age_ms == 0 or
            heartbeat_age_ms > maximum_heartbeat_age_ms):
        raise ValueError("window_video_border.json is stale")
    if (freshness.get("causal_geometry") is not True or
            geometry_continuity_ms < source_content_age_ms):
        raise ValueError("window_video_border.json geometry postdates the source content")
    return {
        "matched_frame_id": frame_id,
        "source_width": width,
        "source_height": height,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "hwnd": hwnd_value,
        "hwnd_text": hwnd,
        "process_id": process_id,
        "document_id": document_id,
        "video_id": video_id,
        "generation": generation,
        "latest_heartbeat_age_ms_at_capture": heartbeat_age_ms,
        "maximum_heartbeat_age_ms": maximum_heartbeat_age_ms,
        "geometry_continuity_ms_at_capture": geometry_continuity_ms,
        "source_content_age_ms_at_capture": source_content_age_ms,
    }


def camera_center_integrity_bits(
        center: float, inverse_scale: float, convergence_curve: float, revision: int) -> int:
    """Mirror the authenticated SM5 uint32 checksum over the latched camera center."""

    words = (
        struct.unpack("<I", struct.pack("<f", center))[0],
        struct.unpack("<I", struct.pack("<f", inverse_scale))[0],
        struct.unpack("<I", struct.pack("<f", convergence_curve))[0],
        revision,
    )
    checksum = 0
    for word in words:
        checksum = ((checksum ^ word) * 16777619) & 0xFFFFFFFF
    return checksum


def _calibration_revision(value: Any, label: str) -> int:
    revision = _uint32(value, label)
    if revision == _RESERVED_CALIBRATION_REVISION:
        raise ValueError(f"{label} uses the reserved calibration revision")
    return revision


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return math.isclose(float(left), float(right), rel_tol=1.0e-6, abs_tol=1.0e-8)


def _require_coordinate_binding(value: Any, count_key: str, count: int) -> None:
    shader = coordinate_contract.SHADER_IMPLEMENTATION
    expected = {
        "schema": coordinate_contract.CONTRACT_SCHEMA,
        "tag": _CONTRACT_TAG,
        "source_closure_schema": shader.source_closure_schema,
        "source_compile_flags": shader.source_compile_flags,
        "source_macro_count": shader.source_macro_count,
        "source_closure_sha256": shader.source_closure_sha256,
        count_key: count,
    }
    if value != expected:
        raise ValueError("dump has an unknown depth-coordinate-v2 contract binding")


def validate_shadow_state_document(document: Any) -> Dict[str, Any]:
    """Validate one parsed ``shadow_state.json`` and return its typed named values."""

    if not isinstance(document, dict) or set(document) != _STATE_ROOT_KEYS:
        raise ValueError("shadow_state.json has missing or unknown root fields")
    if document.get("schema") != SHADOW_STATE_DUMP_SCHEMA:
        raise ValueError("shadow_state.json has an unknown serialization schema")
    _require_coordinate_binding(
        document.get("coordinate_contract"), "state_word_count", len(_STATE_FIELDS))
    rendered = document.get("rendered_output_selected")
    expected_wire = (
        "authenticated live Host-SBS renderer input; not a client wire contract"
        if rendered is True else
        "experimental diagnostic shadow; not selected by the renderer or client"
    )
    if (document.get("source") != _CONTRACT["shadow_state"]["source"] or
            document.get("capture") != _CONTRACT["shadow_state"]["capture"] or
            not isinstance(rendered, bool) or
            document.get("wire_contract") != expected_wire):
        raise ValueError("shadow_state.json has unknown capture semantics")
    if document.get("units") != {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": "one-eye source-U per curve unit",
            "parallax": "signed one-eye source-U"}:
        raise ValueError("shadow_state.json has unknown units")
    if document.get("adaptation_semantics") != {
            "coordinate":
                "immediate-first-usable-center-latched-until-cut-fixed-authenticated-scale-retained-across-unusable",
            "convergence_curve":
                "arithmetic-mean-center-is-zero-plane",
            "requested_gain": "immutable-cfg-pop-strength",
            "container_scale":
                "abi-retained-identity-pointwise-soft-container-is-map-local",
            "near_curve":
                "fixed-contract-logarithmic-tau-independent-of-content-occupancy",
            "spatial_conditioner":
                "fixed-75pct-vertical-majorant-share-then-horizontal-majorant"}:
        raise ValueError("shadow_state.json has unknown adaptation semantics")

    constants = document.get("constants")
    if not isinstance(constants, dict) or set(constants) != _STATE_CONSTANT_KEYS:
        raise ValueError("shadow_state.json has an invalid constants object")
    for key in _STATE_CONSTANT_KEYS:
        _finite_number(constants[key], f"shadow_state.json constants.{key}")
    expected_defaults = {
        "collapse_abs_epsilon": _DEFAULTS.collapse_abs_epsilon,
        "far_tau": _DEFAULTS.far_tau,
        "near_log_tau": _DEFAULTS.near_log_tau,
        "gain_per_pop": _DEFAULTS.gain_per_pop,
        "reference_pop_strength": _DEFAULTS.reference_pop_strength,
        "reference_gain_at_reference_pop":
            _DEFAULTS.gain_per_pop * _DEFAULTS.reference_pop_strength,
        "direct_container_limit": _DEFAULTS.direct_container_limit,
        "max_horizontal_slope": _DEFAULTS.max_horizontal_slope,
        "max_vertical_shear": _DEFAULTS.max_vertical_shear,
        "vertical_majorant_share": _DEFAULTS.vertical_majorant_share,
        "convergence_curve_default": _DEFAULTS.convergence_curve_default,
    }
    if any(not _same_number(constants.get(key), value)
           for key, value in expected_defaults.items()):
        raise ValueError("shadow_state.json constants disagree with the generated contract")
    if (_finite_number(constants["raw_coordinate_scale"], "raw_coordinate_scale") <= 0.0 or
            _finite_number(constants["requested_gain"], "requested_gain") < 0.0 or
            _finite_number(constants["requested_pop_strength"],
                           "requested_pop_strength") < 0.0):
        raise ValueError("shadow_state.json has invalid runtime constants")
    if not _same_number(
            constants["requested_gain"],
            float(constants["requested_pop_strength"]) * _DEFAULTS.gain_per_pop):
        raise ValueError("shadow_state.json requested gain disagrees with requested pop")

    fields = document.get("fields")
    named = document.get("named_values")
    expected_names = [field["name"] for field in _STATE_FIELDS]
    if (not isinstance(fields, list) or len(fields) != len(_STATE_FIELDS) or
            not isinstance(named, dict) or set(named) != set(expected_names)):
        raise ValueError("shadow_state.json does not contain the exact state layout")
    typed: Dict[str, Any] = {}
    for index, (serialized, descriptor) in enumerate(zip(fields, _STATE_FIELDS)):
        expected_type = ("float32" if descriptor["gpu_encoding"] == "float" else
                         "uint32-bitcast")
        if (not isinstance(serialized, dict) or
                set(serialized) != {"index", "name", "type", "value"} or
                serialized.get("index") != index or
                serialized.get("name") != descriptor["name"] or
                serialized.get("type") != expected_type):
            raise ValueError(f"shadow_state.json field {index} has unknown semantics")
        value = serialized.get("value")
        if descriptor["gpu_encoding"] == "float":
            value = _finite_number(value, f"shadow_state.json field {index}")
            named_matches = _same_number(named.get(descriptor["name"]), value)
        else:
            value = _uint32(value, f"shadow_state.json field {index}")
            named_matches = (
                _uint32(
                    named.get(descriptor["name"]),
                    f"shadow_state.json named_values.{descriptor['name']}",
                ) == value
            )
        if not named_matches:
            raise ValueError(f"shadow_state.json field {index} disagrees with named_values")
        typed[descriptor["name"]] = value
    _calibration_revision(
        typed["calibration_revision"],
        "shadow_state.json state calibration_revision",
    )
    if typed["contract_tag_bits"] != _CONTRACT_TAG:
        raise ValueError("shadow_state.json state words have the wrong contract tag")
    expected_camera_integrity = camera_center_integrity_bits(
        float(typed["center"]),
        float(typed["inverse_scale"]),
        float(typed["convergence_curve"]),
        int(typed["calibration_revision"]),
    )
    if typed["camera_center_integrity_bits"] != expected_camera_integrity:
        raise ValueError("shadow_state.json camera center integrity checksum disagrees")
    expected_authorization = _CONTRACT_TAG if typed["frame_valid"] > 0.5 else 0
    if typed["renderer_authorization_bits"] != expected_authorization:
        raise ValueError("shadow_state.json renderer authorization disagrees with frame validity")
    if any(typed[name] != 0 for name in (
            "mapping_state_reserved_1", "mapping_state_reserved_2")):
        raise ValueError("shadow_state.json has nonzero reserved mapping state")

    decoded = document.get("decoded")
    if not isinstance(decoded, dict) or set(decoded) != _DECODED_KEYS:
        raise ValueError("shadow_state.json has an invalid decoded object")
    if not isinstance(decoded.get("frame_valid"), bool) or not isinstance(
            decoded.get("camera_valid"), bool):
        raise ValueError("shadow_state.json decoded validity fields must be boolean")
    if (typed["frame_valid"] not in (0.0, 1.0) or
            decoded["frame_valid"] != (typed["frame_valid"] > 0.5)):
        raise ValueError("shadow_state.json has an invalid or inconsistent frame_valid flag")
    _calibration_revision(
        decoded.get("calibration_revision"),
        "shadow_state.json decoded.calibration_revision",
    )
    for key in ("confirmed_cut_count", "contract_tag", "camera_center_integrity_bits",
                "renderer_authorization_bits"):
        _uint32(decoded.get(key), f"shadow_state.json decoded.{key}")
    for key in ("requested_gain", "requested_pop_strength", "latched_scale",
                "convergence_curve", "container_scale", "effective_gain"):
        _finite_number(decoded.get(key), f"shadow_state.json decoded.{key}")
    expected_decoded = {
        "calibration_revision": typed["calibration_revision"],
        "confirmed_cut_count": typed["confirmed_cut_count"],
        "contract_tag": _CONTRACT_TAG,
        "requested_gain": constants["requested_gain"],
        "requested_pop_strength": constants["requested_pop_strength"],
        "camera_valid": (
            typed["inverse_scale"] > 0.0 and typed["calibration_revision"] > 0),
        "latched_scale": (1.0 / typed["inverse_scale"]
                          if decoded["camera_valid"] else 0.0),
        "convergence_curve": typed["convergence_curve"],
        "container_scale": typed["container_scale"],
        "effective_gain": (constants["requested_gain"]
                           if decoded["frame_valid"] else 0.0),
        "camera_center_integrity_bits": expected_camera_integrity,
        "renderer_authorization_bits": expected_authorization,
    }
    if any(
            (decoded.get(key) != value if key in {
                "camera_valid", "calibration_revision", "confirmed_cut_count", "contract_tag",
                "camera_center_integrity_bits", "renderer_authorization_bits",
            } else not _same_number(decoded.get(key), value))
            for key, value in expected_decoded.items()):
        raise ValueError("shadow_state.json decoded values disagree with the state words")
    convergence_valid = _same_number(
        typed["convergence_curve"], _DEFAULTS.convergence_curve_default)
    if (typed["container_scale"] != 1.0 or
            not convergence_valid or
            decoded["effective_gain"] < 0.0 or
            decoded["effective_gain"] > float(decoded["requested_gain"]) + 1.0e-7 or
            (decoded["camera_valid"] and not _same_number(
                decoded["latched_scale"], constants["raw_coordinate_scale"])) or
            (decoded["frame_valid"] and not decoded["camera_valid"]) or
            (not decoded["camera_valid"] and
              (typed["center"] != 0.0 or typed["inverse_scale"] != 0.0 or
              not _same_number(
                  typed["convergence_curve"], _DEFAULTS.convergence_curve_default)))):
        raise ValueError("shadow_state.json decoded safety state is out of range")
    return typed


def validate_shadow_frame_stats_document(document: Any) -> Dict[str, float]:
    """Validate one parsed ``shadow_frame_stats.json`` under its independent schema."""

    expected_root = {"schema", "coordinate_contract", "source", "named_values"}
    if not isinstance(document, dict) or set(document) != expected_root:
        raise ValueError("shadow_frame_stats.json has missing or unknown root fields")
    if document.get("schema") != SHADOW_FRAME_STATS_DUMP_SCHEMA:
        raise ValueError("shadow_frame_stats.json has an unknown serialization schema")
    _require_coordinate_binding(
        document.get("coordinate_contract"),
        "frame_stats_word_count",
        len(_FRAME_STATS_FIELDS),
    )
    if document.get("source") != _CONTRACT["frame_stats"]["source"]:
        raise ValueError("shadow_frame_stats.json has an unknown source")
    named = document.get("named_values")
    names = [field["name"] for field in _FRAME_STATS_FIELDS]
    if not isinstance(named, dict) or set(named) != set(names):
        raise ValueError("shadow_frame_stats.json does not contain the exact stats layout")
    values = {name: _finite_number(named[name], f"shadow_frame_stats.json {name}")
              for name in names}
    if values["valid"] not in (0.0, 1.0):
        raise ValueError("shadow_frame_stats.json valid must be zero or one")
    valid_count = values["valid_count"]
    texel_count = values["texel_count"]
    if (valid_count < 0.0 or texel_count <= 0.0 or valid_count > texel_count or
            valid_count != math.floor(valid_count) or
            texel_count != math.floor(texel_count) or
            values["population_std"] < 0.0):
        raise ValueError("shadow_frame_stats.json has invalid counts or spread")
    if values["valid"] > 0.5:
        if (valid_count != texel_count or values["maximum"] < values["minimum"]):
            raise ValueError("shadow_frame_stats.json valid statistics are inconsistent")
    elif any(values[key] != 0.0 for key in
             ("mean", "population_std", "minimum", "maximum")):
        raise ValueError("shadow_frame_stats.json invalid statistics must be canonical zero")
    return values


def validate_v2_dump_manifest_document(document: Any) -> Dict[str, Any]:
    """Validate the schema-13 V2 geometry fragment of ``dump_manifest.json``.

    The full package contains legacy/color/model metadata owned by other contracts. This reader
    deliberately validates only the candidate -> full-resolution ownership refinement -> vertical
    envelopes/share -> row-majorant chain, its analysis domain, and every authenticated
    intermediate.
    """

    if not isinstance(document, dict) or document.get("schema") != DUMP_MANIFEST_SCHEMA:
        raise ValueError("dump_manifest.json has an unknown serialization schema")
    renderer = document.get("renderer")
    shadow = document.get("parallax_v2_shadow")
    artifacts = document.get("artifacts")
    dimensions = document.get("dimensions")
    if not all(isinstance(value, dict) for value in
               (renderer, shadow, artifacts, dimensions)):
        raise ValueError("dump_manifest.json is missing its V2 geometry objects")

    input_summary = document.get("depth_input_region")
    input_descriptor = artifacts.get("depth_input_region.json")
    input_preview_descriptor = artifacts.get("depth_input_source.png")
    if (not isinstance(input_summary, dict) or set(input_summary) != {
            "available", "artifact", "mode", "geometry_authority", "renderer_authority"} or
            input_summary.get("available") is not True or
            input_summary.get("artifact") != "depth_input_region.json" or
            input_summary.get("mode") not in {"full-source", "video-region"} or
            input_summary.get("geometry_authority") is not True or
            input_summary.get("renderer_authority") is not True or
            not isinstance(input_descriptor, dict) or set(input_descriptor) != {
                "available", "required", "stage", "description", "sha256"} or
            input_descriptor.get("available") is not True or
            input_descriptor.get("required") is not True or
            input_descriptor.get("stage") != "depth analysis input region" or
            not isinstance(input_descriptor.get("description"), str) or
            not input_descriptor["description"] or
            not _is_sha256_hex(input_descriptor.get("sha256")) or
            not isinstance(input_preview_descriptor, dict) or
            set(input_preview_descriptor) != {"available", "required", "stage", "description"} or
            input_preview_descriptor.get("available") is not True or
            input_preview_descriptor.get("required") is not False or
            input_preview_descriptor.get("stage") != "model-depth input source preview" or
            input_preview_descriptor.get("description") !=
                "Spatially exact full-source or cropped color input submitted to the calibrated preprocess; transfer-aware PNG is diagnostic only and never numeric model authority."):
        raise ValueError("dump_manifest.json has an invalid depth-input-region contract")
    input_mode = input_summary["mode"]

    active = shadow.get("active")
    selected = renderer.get("parallax_v2_render_selected")
    if not isinstance(active, bool) or not isinstance(selected, bool):
        raise ValueError("dump_manifest.json has invalid V2 availability flags")
    shadow_selected = shadow.get("rendered_output_selected")
    if not isinstance(shadow_selected, bool) or shadow_selected != selected:
        raise ValueError("dump_manifest.json V2 selection flags disagree")
    if selected and not active:
        raise ValueError("dump_manifest.json selects V2 without an active producer")
    requested = renderer.get("parallax_v2_render_requested")
    mapping_matches = renderer.get("mapping_artifacts_match_selected_renderer")
    if (not isinstance(requested, bool) or
            not isinstance(mapping_matches, bool) or
            (selected and not requested) or
            (input_mode == "video-region" and not mapping_matches)):
        raise ValueError("dump_manifest.json has invalid V2 renderer request attribution")

    expected_position = (
        "shadow_final_parallax + depth_input_region embedding"
        if selected and input_mode == "video-region" else
        "shadow_final_parallax" if selected else None
    )
    expected_authority = (
        ("authenticated crop-local parallax-v2 conditioned field plus depth-input-region embedding"
         if input_mode == "video-region" else
         "authenticated-parallax-v2-orientation-selective-conditioned-field")
        if selected else None
    )
    expected_inverse = (
        "11-step contractive fixed point; no forward-warp owner/visibility splat and no synthetic fill"
        if selected else None
    )
    expected_coordinate_role = (
        "shadow_coordinate is diagnostic only; it has no renderer authority"
        if selected else None
    )
    expected_live_shader_source = ({
        "source_closure_schema": generator.SOURCE_CLOSURE_SCHEMA,
        "source_compile_flags": generator.SHADER_COMPILE_FLAGS,
        "source_macro_count": 0,
        "source_closure_sha256": LIVE_RENDERER_SOURCE_CLOSURE_SHA256,
        "source_file": "sbs_reprojection_v2_live_ps.hlsl",
        "entrypoint": "main_ps",
        "target": "ps_5_0",
        "diagnostic_source_closure_sha256": DIAGNOSTIC_SOURCE_CLOSURE_SHA256,
        "mapping_source_file": "sbs_reprojection_v2_diagnostics_ps.hlsl",
        "mapping_entrypoint": "mapping_ps",
        "mask_source_file": "sbs_reprojection_v2_diagnostics_ps.hlsl",
        "mask_entrypoint": "mask_ps",
    } if selected else None)
    expected_vertical_role = (
        "least column-wise upper envelope v+ >= ownership-refined candidate with adjacent-row source-U change <= "
        "max_vertical_shear/target_width; diagnostic evidence only"
        if selected else None
    )
    expected_ownership_role = (
        (("conservative full-resolution crop-local source-contour foreground ownership applied "
          "to candidate before the vertical conditioner; may only raise uniquely owned far-side "
          "boundary texels") if input_mode == "video-region" else
         ("conservative full-resolution source-contour foreground ownership applied to candidate "
          "before the vertical conditioner; may only raise uniquely owned far-side boundary texels"))
        if selected else None
    )
    expected_conditioned_role = (
        "fixed 75/25 share of column upper/lower envelopes; may raise or lower candidate and "
        "feeds the row majorant"
        if selected else None
    )
    expected_final_role = (
        (("least row-wise crop-local q >= shadow_vertical_conditioned with horizontal slope <= "
          "max_horizontal_slope and vertical shear <= max_vertical_shear; q plus "
          "depth_input_region embedding is live position authority")
         if input_mode == "video-region" else
         ("least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
          "max_horizontal_slope and vertical shear <= max_vertical_shear; q may raise or lower "
          "candidate and is the live position authority"))
        if selected else None
    )
    expected_collar_defocus = ({
        "enabled": False,
        "role": ("disabled after live hand-boundary halo regression; live color uses one "
                 "linear sample at the inverse-warped coordinate"),
        "kernel": "none",
        "hdr": "native source sample; no clamp, tone map, or gamma conversion",
    } if selected else None)
    if (renderer.get("authority") != expected_authority or
            renderer.get("parallax_v2_inverse") != expected_inverse or
            renderer.get("live_shader_source") != expected_live_shader_source or
            renderer.get("parallax_v2_coordinate_role") != expected_coordinate_role or
            renderer.get("parallax_v2_position_field") != expected_position or
            renderer.get("parallax_v2_ownership_refined_role") != expected_ownership_role or
            renderer.get("parallax_v2_vertical_majorant_role") != expected_vertical_role or
            renderer.get("parallax_v2_vertical_conditioned_role") !=
                expected_conditioned_role or
            renderer.get("parallax_v2_conditioner_role") != expected_final_role or
            renderer.get("collar_defocus") != expected_collar_defocus):
        raise ValueError("dump_manifest.json has unknown V2 conditioner attribution")

    expected_artifacts = {
        "shadow_candidate_parallax.f32": (
            "parallax-v2 pre-limiter candidate displacement", True),
        "shadow_ownership_refined_parallax.f32": (
            "parallax-v2 full-resolution contour ownership refinement", True),
        "shadow_ownership_refined_parallax_shape.json":
            ("parallax-v2 full-resolution contour ownership refinement contract", False),
        "shadow_ownership_refined_parallax.png":
            ("parallax-v2 full-resolution contour ownership refinement preview", False),
        "shadow_ownership_refined_parallax_heat.png":
            ("parallax-v2 full-resolution contour ownership refinement preview", False),
        "shadow_vertical_majorant.f32": (
            "parallax-v2 vertical shear-limiter intermediate", False),
        "shadow_vertical_majorant_shape.json":
            ("parallax-v2 vertical shear-limiter intermediate contract", False),
        "shadow_vertical_majorant.png":
            ("parallax-v2 vertical shear-limiter intermediate preview", False),
        "shadow_vertical_majorant_heat.png":
            ("parallax-v2 vertical shear-limiter intermediate preview", False),
        "shadow_vertical_conditioned.f32":
            ("parallax-v2 orientation-selective vertical conditioner", False),
        "shadow_vertical_conditioned_shape.json":
            ("parallax-v2 orientation-selective vertical conditioner contract", False),
        "shadow_vertical_conditioned.png":
            ("parallax-v2 orientation-selective vertical conditioner preview", False),
        "shadow_vertical_conditioned_heat.png":
            ("parallax-v2 orientation-selective vertical conditioner preview", False),
        "shadow_final_parallax.f32": (
            "parallax-v2 final conditioned displacement field", True),
    }
    for name, (stage, required) in expected_artifacts.items():
        descriptor = artifacts.get(name)
        # Exact geometry fields carry a mandatory SHA-256 of the written bytes; a manifest
        # that describes a geometry field without binding its content is invalid.
        hashed = name.endswith(".f32")
        expected_keys = {"available", "required", "stage", "description"}
        if hashed:
            expected_keys = expected_keys | {"sha256"}
        if (not isinstance(descriptor, dict) or
                set(descriptor) != expected_keys or
                descriptor.get("available") is not active or
                descriptor.get("required") is not required or
                descriptor.get("stage") != stage or
                not isinstance(descriptor.get("description"), str) or
                not descriptor["description"]):
            raise ValueError(
                "dump_manifest.json has an invalid V2 geometry artifact contract")
        if hashed and not _is_sha256_hex(descriptor.get("sha256")):
            raise ValueError(
                "dump_manifest.json geometry artifact lacks a valid content sha256")
    dimension_names = (
        "shadow_candidate_parallax", "shadow_ownership_refined_parallax",
        "shadow_vertical_majorant",
        "shadow_vertical_conditioned", "shadow_final_parallax")
    geometry_dimensions = [dimensions.get(name) for name in dimension_names]
    if active:
        if any(not isinstance(value, dict) for value in geometry_dimensions):
            raise ValueError("dump_manifest.json omits an active V2 geometry dimension")
        expected_dimensions = geometry_dimensions[0]
        if (set(expected_dimensions) != {"width", "height", "format", "format_value"} or
                type(expected_dimensions["width"]) is not int or
                type(expected_dimensions["height"]) is not int or
                expected_dimensions["width"] <= 0 or expected_dimensions["height"] <= 0 or
                expected_dimensions["format"] != "DXGI_FORMAT_R32_FLOAT" or
                expected_dimensions["format_value"] != 41 or
                any(value != expected_dimensions for value in geometry_dimensions[1:])):
            raise ValueError("dump_manifest.json V2 geometry dimensions disagree")

        geometry_width = expected_dimensions["width"]
        geometry_height = expected_dimensions["height"]
        model_dimensions = dimensions.get("model_input")
        raw_dimensions = dimensions.get("raw_depth")
        warp_dimensions = dimensions.get("warp_depth")
        source_dimensions = dimensions.get("source")
        analysis_source_dimensions = dimensions.get("analysis_source")
        if (not isinstance(model_dimensions, dict) or
                set(model_dimensions) != {"width", "height", "channels", "layout", "dtype"} or
                model_dimensions.get("width") != geometry_width or
                model_dimensions.get("height") != geometry_height or
                model_dimensions.get("channels") != 3 or
                model_dimensions.get("layout") != "NCHW" or
                model_dimensions.get("dtype") != "float32-le" or
                not isinstance(raw_dimensions, dict) or
                set(raw_dimensions) != {"width", "height", "format"} or
                raw_dimensions.get("width") != geometry_width or
                raw_dimensions.get("height") != geometry_height or
                raw_dimensions.get("format") != "float32-le structured buffer" or
                not isinstance(warp_dimensions, dict) or
                set(warp_dimensions) != {"width", "height", "format", "format_value"} or
                warp_dimensions.get("width") != geometry_width or
                warp_dimensions.get("height") != geometry_height or
                warp_dimensions.get("format") != "DXGI_FORMAT_R32_FLOAT" or
                warp_dimensions.get("format_value") != 41):
            raise ValueError("dump_manifest.json crop-local tensor dimensions disagree")
        expected_texture_keys = {"width", "height", "format", "format_value"}
        if (not isinstance(source_dimensions, dict) or
                set(source_dimensions) != expected_texture_keys or
                not isinstance(analysis_source_dimensions, dict) or
                set(analysis_source_dimensions) != expected_texture_keys or
                type(source_dimensions.get("width")) is not int or
                type(source_dimensions.get("height")) is not int or
                source_dimensions["width"] <= 0 or source_dimensions["height"] <= 0 or
                type(analysis_source_dimensions.get("width")) is not int or
                type(analysis_source_dimensions.get("height")) is not int or
                analysis_source_dimensions["width"] <= 0 or
                analysis_source_dimensions["height"] <= 0 or
                (source_dimensions["format"], source_dimensions["format_value"]) not in {
                    ("DXGI_FORMAT_B8G8R8A8_UNORM", 87),
                    ("DXGI_FORMAT_R16G16B16A16_FLOAT", 10)} or
                analysis_source_dimensions["format"] != source_dimensions["format"] or
                analysis_source_dimensions["format_value"] !=
                    source_dimensions["format_value"]):
            raise ValueError("dump_manifest.json has invalid analysis-source dimensions")
    elif any(value is not None for value in geometry_dimensions):
        raise ValueError("dump_manifest.json exposes inactive V2 geometry dimensions")

    if input_mode == "video-region":
        warp_map_descriptor = artifacts.get("warp_map.f32")
        warp_map_shape_descriptor = artifacts.get("warp_map_shape.json")
        if (not isinstance(warp_map_descriptor, dict) or
                set(warp_map_descriptor) != {
                    "available", "required", "stage", "description", "sha256"} or
                warp_map_descriptor.get("available") is not True or
                warp_map_descriptor.get("required") is not True or
                warp_map_descriptor.get("stage") != "exact full-source inverse-warp mapping" or
                not isinstance(warp_map_descriptor.get("description"), str) or
                not warp_map_descriptor["description"] or
                not _is_sha256_hex(warp_map_descriptor.get("sha256")) or
                not isinstance(warp_map_shape_descriptor, dict) or
                set(warp_map_shape_descriptor) != {
                    "available", "required", "stage", "description"} or
                warp_map_shape_descriptor.get("available") is not True or
                warp_map_shape_descriptor.get("required") is not True or
                warp_map_shape_descriptor.get("stage") != "inverse-warp mapping contract" or
                not isinstance(warp_map_shape_descriptor.get("description"), str) or
                not warp_map_shape_descriptor["description"] or
                not isinstance(dimensions.get("warp_map"), dict)):
            raise ValueError("dump_manifest.json lacks authoritative ROI warp-map evidence")

    border_summary = document.get("window_video_border")
    border_descriptor = artifacts.get("window_video_border.json")
    border_available = False
    if (not isinstance(border_summary, dict) or set(border_summary) != {
            "available", "artifact", "observer_status", "mapping_status",
            "geometry_authority", "renderer_authority"} or
            not isinstance(border_descriptor, dict)):
        raise ValueError("dump_manifest.json has an invalid window-video border contract")
    border_available = border_summary.get("available")
    border_required = input_mode == "video-region"
    expected_border_descriptor_keys = {"available", "required", "stage", "description"}
    if border_required:
        expected_border_descriptor_keys.add("sha256")
    if (set(border_descriptor) != expected_border_descriptor_keys or
            not isinstance(border_available, bool) or
            border_descriptor.get("available") is not border_available or
            border_descriptor.get("required") is not border_required or
            border_descriptor.get("stage") != "matched-frame window-video border" or
            not isinstance(border_descriptor.get("description"), str) or
            not border_descriptor["description"] or
            border_summary.get("artifact") != (
                "window_video_border.json" if border_available else None) or
            border_summary.get("observer_status") not in WINDOW_VIDEO_OBSERVER_STATUSES or
            border_summary.get("mapping_status") not in WINDOW_VIDEO_MAPPING_STATUSES or
            (border_available and (
                border_summary.get("observer_status") != "ok" or
                border_summary.get("mapping_status") != "ok")) or
            (border_required and (
                not border_available or
                not _is_sha256_hex(border_descriptor.get("sha256")))) or
            border_summary.get("geometry_authority") is not False or
            border_summary.get("renderer_authority") is not False):
        raise ValueError("dump_manifest.json has an inconsistent window-video border contract")
    return {
        "active": active,
        "rendered_output_selected": selected,
        "mapping_artifacts_match_selected_renderer": mapping_matches,
        "position_field": expected_position,
        "ownership_refined_available": active,
        "vertical_majorant_available": active,
        "vertical_conditioned_available": active,
        "depth_input_region_available": True,
        "depth_input_mode": input_mode,
        "position_authority": (
            ["shadow_final_parallax", "depth_input_region"]
            if input_mode == "video-region" else ["shadow_final_parallax"]),
        "window_video_border_available": border_available,
    }


def _is_sha256_hex(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


_GEOMETRY_CHAIN_FIELDS = (
    "shadow_candidate_parallax",
    "shadow_ownership_refined_parallax",
    "shadow_vertical_majorant",
    "shadow_vertical_conditioned",
    "shadow_final_parallax",
)


def _verify_roi_exterior_zero_warp_map(
        dump_dir: Any, manifest: Dict[str, Any], input_region: Dict[str, Any],
        final_parallax: Any) -> Dict[str, Any]:
    """Prove exact-zero exterior samples from the selected full-source inverse map."""

    import hashlib
    import json
    import os

    import numpy as np

    artifacts = manifest["artifacts"]
    descriptor = artifacts["warp_map.f32"]
    map_path = os.path.join(os.fspath(dump_dir), "warp_map.f32")
    try:
        with open(map_path, "rb") as handle:
            payload = handle.read()
    except OSError as error:
        raise ValueError("authoritative ROI warp_map.f32 is missing") from error
    digest = hashlib.sha256(payload).hexdigest()
    if digest != descriptor["sha256"]:
        raise ValueError("warp_map.f32 content hash mismatch")

    try:
        with open(os.path.join(os.fspath(dump_dir), "warp_map_shape.json"),
                  encoding="utf-8") as handle:
            shape = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("authoritative ROI warp_map_shape.json is missing or malformed") from error

    dimensions = manifest["dimensions"]
    packed = dimensions.get("packed_sbs")
    eye = dimensions.get("eye")
    source = dimensions.get("source")
    warp_dimensions = dimensions.get("warp_map")
    if not all(isinstance(value, dict) for value in (packed, eye, source, warp_dimensions)):
        raise ValueError("ROI warp-map evidence lacks full-source/output dimensions")
    map_width = _uint32(warp_dimensions.get("width"), "warp-map width")
    map_height = _uint32(warp_dimensions.get("height"), "warp-map height")
    eye_width = _uint32(eye.get("width"), "eye width")
    eye_height = _uint32(eye.get("height"), "eye height")
    source_width = input_region["source_width"]
    source_height = input_region["source_height"]
    expected_shape_keys = {
        "schema", "width", "height", "eye_width", "eye_height", "source_width",
        "source_height", "content_scale_x", "content_scale_y", "dtype", "layout",
        "channels", "validity", "live_sample_source_u_normalized",
        "derived_inverse_displacement_output_eye_px",
        "derived_signed_binocular_disparity_px", "displacement_preview",
    }
    expected_validity = {
        "content": "derive from content_scale_x/content_scale_y and packed output coordinate",
        "inverse": ("11-step contractive fixed-point solution of crop-local q embedded by "
                    "depth_input_region.json scale and outside-only zero-plane collar"),
        "mask": ("warp_mask.png red marks finite-source boundary extrapolation; V2 has no "
                 "internal owner or synthetic-fill path"),
    }
    preview = shape.get("displacement_preview")
    if (map_width == 0 or map_height == 0 or eye_width == 0 or eye_height == 0 or
            packed.get("width") != map_width or packed.get("height") != map_height or
            map_width != 2 * eye_width or map_height != eye_height or
            source.get("width") != source_width or source.get("height") != source_height or
            warp_dimensions.get("format") != "DXGI_FORMAT_R32_FLOAT" or
            warp_dimensions.get("format_value") != 41 or
            set(shape) != expected_shape_keys or
            shape.get("schema") != 2 or shape.get("width") != map_width or
            shape.get("height") != map_height or shape.get("eye_width") != eye_width or
            shape.get("eye_height") != eye_height or
            shape.get("source_width") != source_width or
            shape.get("source_height") != source_height or
            shape.get("dtype") != "float32-le" or shape.get("layout") != "row-major" or
            shape.get("channels") != ["raw_reproject_source_u_normalized"] or
            shape.get("validity") != expected_validity or
            shape.get("live_sample_source_u_normalized") !=
                "clamp(raw_reproject_source_u_normalized, 0, 1)" or
            shape.get("derived_inverse_displacement_output_eye_px") !=
                "(raw_reproject_source_u_normalized - aspect_fitted_unwarped_source_u) * content_scale_x * eye_width" or
            shape.get("derived_signed_binocular_disparity_px") !=
                "invert both eye maps at common source-U samples; x_right - x_left" or
            not isinstance(preview, dict) or set(preview) != {
                "file", "range_px", "normalization", "negative", "zero", "positive",
                "bars", "nonfinite"} or
            preview.get("file") != "warp_displacement_heat.png" or
            preview.get("normalization") !=
                "symmetric finite-content p98 absolute displacement" or
            preview.get("negative") != "blue" or preview.get("zero") != "green" or
            preview.get("positive") != "red" or preview.get("bars") != "black" or
            preview.get("nonfinite") != "magenta" or
            not isinstance(preview.get("range_px"), list) or
            len(preview["range_px"]) != 2 or
            not math.isclose(
                _finite_number(preview["range_px"][0], "displacement preview lower"),
                -_finite_number(preview["range_px"][1], "displacement preview upper"),
                rel_tol=0.0, abs_tol=1.0e-7) or preview["range_px"][1] <= 0.0):
        raise ValueError("ROI warp-map dimensions/shape contract disagree")

    content_scale_x = _finite_number(shape.get("content_scale_x"), "content scale x")
    content_scale_y = _finite_number(shape.get("content_scale_y"), "content scale y")
    content_fit = dimensions.get("content_fit")
    source_aspect = _float32(
        _float32(float(source_width)) / _float32(float(source_height)))
    eye_aspect = _float32(
        _float32(float(eye_width)) / _float32(float(eye_height)))
    expected_scale_x = (_float32(source_aspect / eye_aspect)
                        if eye_aspect > source_aspect else _float32(1.0))
    expected_scale_y = (_float32(eye_aspect / source_aspect)
                        if eye_aspect < source_aspect else _float32(1.0))
    if (not 0.0 < content_scale_x <= 1.0 or not 0.0 < content_scale_y <= 1.0 or
            content_scale_x != expected_scale_x or content_scale_y != expected_scale_y or
            not isinstance(content_fit, dict) or
            _finite_number(content_fit.get("scale_x"), "manifest content scale x") !=
                expected_scale_x or
            _finite_number(content_fit.get("scale_y"), "manifest content scale y") !=
                expected_scale_y):
        raise ValueError("ROI warp-map content-fit contract disagrees")

    values = np.frombuffer(payload, dtype="<f4")
    if values.size != map_width * map_height:
        raise ValueError("warp_map.f32 size disagrees with full-source packed output")
    if not np.isfinite(values).all():
        raise ValueError("warp_map.f32 contains non-finite values")
    values = values.reshape(map_height, map_width)

    packed_u = ((np.arange(map_width, dtype=np.float32) + np.float32(0.5)) /
                np.float32(map_width)).astype(np.float32)
    right_eye = packed_u > np.float32(0.5)
    eye_u = np.where(
        right_eye,
        (packed_u - np.float32(0.5)) * np.float32(2.0),
        packed_u * np.float32(2.0),
    ).astype(np.float32)
    scale_x = np.float32(content_scale_x)
    scale_y = np.float32(content_scale_y)
    lo_x = np.float32(0.5) * (np.float32(1.0) - scale_x)
    hi_x = lo_x + scale_x
    valid_x = (eye_u >= lo_x) & (eye_u <= hi_x)
    unwarped_u = np.clip((eye_u - lo_x) / scale_x, 0.0, 1.0).astype(np.float32)

    left, top, right, bottom = input_region["inference_rect"]
    roi_left = np.float32(left) / np.float32(source_width)
    roi_top = np.float32(top) / np.float32(source_height)
    roi_right = np.float32(right) / np.float32(source_width)
    roi_bottom = np.float32(bottom) / np.float32(source_height)
    outside_dx = np.maximum(
        np.maximum(roi_left - unwarped_u, np.float32(0.0)),
        np.maximum(unwarped_u - roi_right, np.float32(0.0)),
    ).astype(np.float32)
    maximum_embedded_magnitude = np.float32(
        np.float32(input_region["full_source_parallax_scale"]) *
        np.float32(np.max(np.abs(final_parallax))))
    horizontal_slope = np.float32(
        input_region["horizontal_slope_source_u_per_source_u"])
    vertical_slope = np.float32(
        input_region["vertical_slope_source_u_per_source_v"])
    # Keep the proof set strictly beyond the largest possible collar support. This margin is
    # wider than a few float32 ULPs but far below a visible output-pixel displacement.
    support_margin = np.float32(2.0e-6)

    sample_count = 0
    exterior_content_sample_count = 0
    maximum_error_px = 0.0
    lo_y = np.float32(0.5) * (np.float32(1.0) - scale_y)
    hi_y = lo_y + scale_y
    for row in range(map_height):
        eye_v = np.float32((row + 0.5) / map_height)
        if eye_v < lo_y or eye_v > hi_y:
            continue
        source_v = np.float32(np.clip((eye_v - lo_y) / scale_y, 0.0, 1.0))
        outside_dy = np.maximum(
            np.maximum(roi_top - source_v, np.float32(0.0)),
            np.maximum(source_v - roi_bottom, np.float32(0.0)),
        ).astype(np.float32)
        outside = (outside_dx > 0.0) | (outside_dy > 0.0)
        exterior_content_sample_count += int(np.count_nonzero(valid_x & outside))
        budget = horizontal_slope * outside_dx + vertical_slope * outside_dy
        proof_mask = valid_x & outside & (
            budget >= maximum_embedded_magnitude + support_margin)
        count = int(np.count_nonzero(proof_mask))
        if count == 0:
            continue
        error_px = np.abs(
            (values[row, proof_mask] - unwarped_u[proof_mask]) *
            np.float32(content_scale_x * eye_width))
        maximum_error_px = max(maximum_error_px, float(np.max(error_px)))
        sample_count += count

    if sample_count == 0:
        horizontal_collar_px = float(
            maximum_embedded_magnitude / horizontal_slope * source_width)
        vertical_collar_px = float(
            maximum_embedded_magnitude / vertical_slope * source_height)
        return {
            "applicable": False,
            "has_exterior_zero_plane": False,
            "sample_count": 0,
            "beyond_collar_sample_count": 0,
            "beyond_collar_fraction": 0.0,
            "exterior_content_sample_count": exterior_content_sample_count,
            "max_abs_output_eye_px": None,
            "max_abs_identity_error_output_eye_px": None,
            "max_embedded_parallax_source_u": float(maximum_embedded_magnitude),
            "max_horizontal_collar_source_px": horizontal_collar_px,
            "max_vertical_collar_source_px": vertical_collar_px,
            "tolerance_output_eye_px": _ROI_EXTERIOR_ZERO_TOLERANCE_OUTPUT_EYE_PX,
        }
    if maximum_error_px > _ROI_EXTERIOR_ZERO_TOLERANCE_OUTPUT_EYE_PX:
        raise ValueError(
            "ROI warp map is nonzero beyond conservative collar support "
            f"(max {maximum_error_px:.9g} output-eye px)")
    horizontal_collar_px = float(
        maximum_embedded_magnitude / horizontal_slope * source_width)
    vertical_collar_px = float(
        maximum_embedded_magnitude / vertical_slope * source_height)
    return {
        "applicable": True,
        "has_exterior_zero_plane": True,
        "sample_count": sample_count,
        "beyond_collar_sample_count": sample_count,
        "beyond_collar_fraction": sample_count / exterior_content_sample_count,
        "exterior_content_sample_count": exterior_content_sample_count,
        "max_abs_output_eye_px": maximum_error_px,
        "max_abs_identity_error_output_eye_px": maximum_error_px,
        "max_embedded_parallax_source_u": float(maximum_embedded_magnitude),
        "max_horizontal_collar_source_px": horizontal_collar_px,
        "max_vertical_collar_source_px": vertical_collar_px,
        "tolerance_output_eye_px": _ROI_EXTERIOR_ZERO_TOLERANCE_OUTPUT_EYE_PX,
    }


def verify_v2_dump_geometry(dump_dir: Any) -> Dict[str, Any]:
    """Verify a dump directory's geometry evidence against its own manifest.

    Checks, fail-closed:
      1. every chain field's ``.f32`` bytes hash to the manifest descriptor's ``sha256``;
      2. every field matches the manifest geometry dimensions and is entirely finite;
      3. the conditioning chain is internally consistent in exact float32:
         ``vertical_majorant``/``vertical_conditioned``/``final`` are bitwise equal to the
         recurrences recomputed from ``ownership_refined``; and the ownership refinement never
         lowers the candidate.

    Returns a summary dict on success; raises ``ValueError`` on the first violation.
    """

    import hashlib
    import json
    import os

    import numpy as np

    manifest_path = os.path.join(os.fspath(dump_dir), "dump_manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    fragment = validate_v2_dump_manifest_document(manifest)
    if not fragment["active"]:
        raise ValueError("dump has no active V2 geometry to verify")

    dimensions = manifest["dimensions"]["shadow_final_parallax"]
    height, width = int(dimensions["height"]), int(dimensions["width"])
    artifacts = manifest["artifacts"]

    source_dimensions = manifest["dimensions"].get("source")
    analysis_source_dimensions = manifest["dimensions"].get("analysis_source")
    if not isinstance(source_dimensions, dict) or not isinstance(analysis_source_dimensions, dict):
        raise ValueError("dump manifest lacks source/analysis-source dimensions")
    source_width = _uint32(source_dimensions.get("width"), "dump source width")
    source_height = _uint32(source_dimensions.get("height"), "dump source height")
    frame_id = _uint64(manifest.get("matched_frame_id"), "dump manifest matched frame id")
    if source_width == 0 or source_height == 0 or frame_id == 0:
        raise ValueError("dump_manifest.json has invalid matched source geometry")

    input_region_path = os.path.join(os.fspath(dump_dir), "depth_input_region.json")
    try:
        with open(input_region_path, "rb") as handle:
            input_region_payload = handle.read()
        input_region_document = json.loads(input_region_payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("depth_input_region.json is missing or malformed") from error
    input_region_digest = hashlib.sha256(input_region_payload).hexdigest()
    if input_region_digest != artifacts["depth_input_region.json"]["sha256"]:
        raise ValueError("depth_input_region.json content hash mismatch")
    input_region = validate_depth_input_region_document(
        input_region_document,
        matched_frame_id=frame_id,
        source_width=source_width,
        source_height=source_height,
        tensor_width=width,
        tensor_height=height,
    )
    if input_region["mode"] != fragment["depth_input_mode"]:
        raise ValueError("depth-input-region mode disagrees with the manifest")
    if (analysis_source_dimensions.get("width") != input_region["inference_width"] or
            analysis_source_dimensions.get("height") != input_region["inference_height"]):
        raise ValueError("analysis-source dimensions disagree with depth_input_region.json")

    border = None
    if fragment["window_video_border_available"]:
        border_path = os.path.join(os.fspath(dump_dir), "window_video_border.json")
        try:
            with open(border_path, "rb") as handle:
                border_payload = handle.read()
            border_document = json.loads(border_payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "advertised window_video_border.json is missing or malformed") from error
        border_descriptor = artifacts["window_video_border.json"]
        if "sha256" in border_descriptor:
            border_digest = hashlib.sha256(border_payload).hexdigest()
            if border_digest != border_descriptor["sha256"]:
                raise ValueError("window_video_border.json content hash mismatch")
        border = validate_window_video_border_document(
            border_document,
            matched_frame_id=frame_id,
            source_width=source_width,
            source_height=source_height,
        )
    if input_region["mode"] == "video-region":
        if border is None:
            raise ValueError("video-region dump has no authenticated semantic border")
        if input_region["semantic_rect"] != (
                border["left"], border["top"], border["right"], border["bottom"]):
            raise ValueError("depth input semantic rectangle disagrees with window-video border")
        authorization = input_region["authorization"]
        if authorization != {
                "observer_generation": border["generation"],
                "hwnd": border["hwnd"],
                "hwnd_text": border["hwnd_text"],
                "process_id": border["process_id"],
                "document_id": border["document_id"],
                "video_id": border["video_id"]}:
            raise ValueError("depth input authorization disagrees with window-video border")

    fields: Dict[str, Any] = {}
    for name in _GEOMETRY_CHAIN_FIELDS:
        path = os.path.join(os.fspath(dump_dir), name + ".f32")
        with open(path, "rb") as handle:
            payload = handle.read()
        digest = hashlib.sha256(payload).hexdigest()
        expected = artifacts[name + ".f32"]["sha256"]
        if digest != expected:
            raise ValueError(f"{name}.f32 content hash mismatch: {digest} != {expected}")
        values = np.frombuffer(payload, dtype="<f4")
        if values.size != width * height:
            raise ValueError(f"{name}.f32 has {values.size} texels; expected {width * height}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name}.f32 contains non-finite values")
        fields[name] = values.reshape(height, width)

    candidate = fields["shadow_candidate_parallax"]
    ownership = fields["shadow_ownership_refined_parallax"]
    if np.any(ownership < candidate):
        raise ValueError("ownership refinement lowered the candidate field")

    # Exact float32 replicas of the production recurrences (validated bitwise against the
    # GPU intermediates; see docs/host-sbs.md#cliff-conditioning).
    vertical_step = np.float32(_DEFAULTS.max_vertical_shear / width)
    horizontal_step = np.float32(_DEFAULTS.max_horizontal_slope / width)
    share = np.float32(_DEFAULTS.vertical_majorant_share)
    share_complement = np.float32(1.0 - float(share))

    majorant = ownership.astype(np.float32).copy()
    for row in range(1, height):
        majorant[row] = np.maximum(majorant[row], majorant[row - 1] - vertical_step)
    for row in range(height - 2, -1, -1):
        majorant[row] = np.maximum(majorant[row], majorant[row + 1] - vertical_step)
    minorant = ownership.astype(np.float32).copy()
    for row in range(1, height):
        minorant[row] = np.minimum(ownership[row], minorant[row - 1] + vertical_step)
    for row in range(height - 2, -1, -1):
        minorant[row] = np.minimum(minorant[row], minorant[row + 1] + vertical_step)
    conditioned = (share * majorant + share_complement * minorant).astype(np.float32)
    final = conditioned.copy()
    for col in range(1, width):
        final[:, col] = np.maximum(final[:, col], final[:, col - 1] - horizontal_step)
    for col in range(width - 2, -1, -1):
        final[:, col] = np.maximum(final[:, col], final[:, col + 1] - horizontal_step)

    for name, recomputed in (
        ("shadow_vertical_majorant", majorant),
        ("shadow_vertical_conditioned", conditioned),
        ("shadow_final_parallax", final),
    ):
        if not np.array_equal(fields[name], recomputed):
            mismatch = float(np.max(np.abs(fields[name] - recomputed)))
            raise ValueError(
                f"{name}.f32 is not the exact recurrence of the dumped ownership field "
                f"(max abs diff {mismatch})")

    warp_depth_path = os.path.join(os.fspath(dump_dir), "warp_depth.f32")
    try:
        with open(warp_depth_path, "rb") as handle:
            warp_depth_payload = handle.read()
    except OSError as error:
        raise ValueError("warp_depth.f32 is missing") from error
    warp_descriptor = artifacts.get("warp_depth.f32")
    if (not isinstance(warp_descriptor, dict) or
            hashlib.sha256(warp_depth_payload).hexdigest() != warp_descriptor.get("sha256")):
        raise ValueError("warp_depth.f32 content hash mismatch")
    warp_depth = np.frombuffer(warp_depth_payload, dtype="<f4")
    if (warp_depth.size != width * height or not np.isfinite(warp_depth).all()):
        raise ValueError("warp_depth.f32 has invalid crop-local geometry")
    if not np.array_equal(warp_depth.reshape(height, width), fields["shadow_final_parallax"]):
        raise ValueError("warp_depth.f32 is not the exact final crop-local position field")

    exterior_zero = None
    if input_region["mode"] == "video-region":
        exterior_zero = _verify_roi_exterior_zero_warp_map(
            dump_dir,
            manifest,
            input_region,
            fields["shadow_final_parallax"],
        )

    return {
        "width": width,
        "height": height,
        "texels": width * height,
        "chain_fields_verified": list(_GEOMETRY_CHAIN_FIELDS),
        "depth_input_region_verified": True,
        "depth_input_region": input_region,
        "roi_exterior_zero_evidence": exterior_zero,
        "window_video_border_verified": border is not None,
        "window_video_border": border,
    }


__all__ = [
    "DEPTH_INPUT_REGION_SCHEMA",
    "DIAGNOSTIC_SOURCE_CLOSURE_SHA256",
    "DUMP_MANIFEST_SCHEMA",
    "LIVE_RENDERER_SOURCE_CLOSURE_SHA256",
    "SHADOW_FRAME_STATS_DUMP_SCHEMA",
    "SHADOW_STATE_DUMP_SCHEMA",
    "WINDOW_VIDEO_BORDER_SCHEMA",
    "camera_center_integrity_bits",
    "validate_depth_input_region_document",
    "validate_window_video_border_document",
    "validate_v2_dump_manifest_document",
    "validate_shadow_frame_stats_document",
    "validate_shadow_state_document",
    "verify_v2_dump_geometry",
]
