#!/usr/bin/env python3
"""Strict readers for the diagnostic depth-coordinate-v2 JSON dump files.

The JSON serialization schemas are deliberately independent of the GPU coordinate-contract
schema.  Each document also binds the exact generated contract schema/tag and complete physical
field layout, so a consumer cannot silently reinterpret a same-sized state buffer.
"""

from __future__ import annotations

import math
import re
import struct
from typing import Any, Dict, Tuple

try:
    from . import depth_coordinate_v2_contract as coordinate_contract
    from . import generate_depth_coordinate_v2_contract as generator
    from . import host_sbs_shader_manifest as shader_manifest
    from . import prod_zipdepth_convex2x as convex2x_contract
except ImportError:  # Direct script/module loading from tools/sbsbench.
    import depth_coordinate_v2_contract as coordinate_contract  # type: ignore
    import generate_depth_coordinate_v2_contract as generator  # type: ignore
    import host_sbs_shader_manifest as shader_manifest  # type: ignore
    import prod_zipdepth_convex2x as convex2x_contract  # type: ignore


DUMP_MANIFEST_SCHEMA = 49
GPU_TRACE_RING_SCHEMA = 5
GPU_TRACE_CONTRACT_SCHEMA = 6
GPU_TRACE_DECODED_SCHEMA = 7
GPU_TRACE_RING_TAG = 0x48525447
GPU_TRACE_RECORD_TAG = 0x31525447
GPU_TRACE_CAPACITY = 300
GPU_TRACE_HEADER_WORD_COUNT = 16
GPU_TRACE_RECORD_WORD_COUNT = 136
GPU_TRACE_TRANSACTION_WORD_COUNT = 64
GPU_TRACE_PLANE_WORD_COUNT = 40
GPU_TRACE_CONDITION_WORD_COUNT = 6
GPU_TRACE_RING_WORD_COUNT = (
    GPU_TRACE_HEADER_WORD_COUNT + GPU_TRACE_CAPACITY * GPU_TRACE_RECORD_WORD_COUNT)
GPU_TRACE_RING_BYTE_COUNT = GPU_TRACE_RING_WORD_COUNT * 4
GPU_TRACE_SOURCE_CLOSURE_SHA256 = shader_manifest.GPU_TRACE_GROUP.source_closure_sha256
_GPU_TRACE_SPEC = shader_manifest.GPU_TRACE_GROUP.specs[0]
GPU_TRACE_SHADER_SPEC = (
    _GPU_TRACE_SPEC.source_file,
    _GPU_TRACE_SPEC.source_entrypoint,
    _GPU_TRACE_SPEC.source_target,
)
GPU_TRACE_DECISION_COOKIE = 0xD1EC15A5
GPU_TRACE_TOKEN_LOW_COOKIE = 0xA3756C91
GPU_TRACE_TOKEN_HIGH_COOKIE = 0x5C8A936E
GPU_TRACE_WORK_FLAGS_COOKIE = 0x6F435257
GPU_TRACE_PROPOSAL_MAGIC = 0x504F5250
GPU_TRACE_REQUEST_MAGIC = 0x54535152
GPU_TRACE_RECEIPT_MAGIC = 0x47524243
GPU_TRACE_OPTIONAL_RECEIPT_MAGIC = 0x52434F4F
GPU_TRACE_WORK_NONE = 0
GPU_TRACE_WORK_OPTIONAL_OCR = 1 << 0
GPU_TRACE_WORK_SUBTITLE_OBSERVATION = 1 << 1
GPU_TRACE_KNOWN_FLAGS = 0x3F
GPU_TRACE_FLAG_INPUT_DOMAIN_RESET = 1 << 0
GPU_TRACE_FLAG_DUMP_FORCED_AT_ENQUEUE = 1 << 1
GPU_TRACE_FLAG_OCR_RECORD_SUBMITTED = 1 << 2
GPU_TRACE_FLAG_SUBTITLE_SUPPRESSED = 1 << 3
GPU_TRACE_FLAG_CONDITION_EXECUTED = 1 << 4
GPU_TRACE_FLAG_SUBTITLE_BRANCH_GATED = 1 << 5
GPU_TRACE_HEADER_OFFSETS = {
    "schema": 0, "tag": 1, "capacity": 2, "record_words": 3,
    "next_sequence_low": 4, "next_sequence_high": 5,
    "next_slot": 6, "committed_count": 7, "reserved_begin": 8,
    "end": 16,
}
GPU_TRACE_RECORD_OFFSETS = {
    "schema": 0, "commit_tag": 1, "sequence_low": 2, "sequence_high": 3,
    "frame_low": 4, "frame_high": 5,
    "analysis_generation_low": 6, "analysis_generation_high": 7,
    "domain_tag_low": 8, "domain_tag_high": 9,
    "transaction_token_low": 10, "transaction_token_high": 11,
    "submission_class": 12, "depth_disposition": 13, "expected_work": 14,
    "subtitle_disposition": 15, "flags": 16, "host_subtitle_outcome": 17,
    "source_width": 18, "source_height": 19,
    "field_width": 20, "field_height": 21, "transaction_words": 22,
    "reserved0": 23, "transaction_begin": 24, "subtitle_locator_begin": 88,
    "subtitle_condition_begin": 128,
    "observation_timestamp_low": 134, "observation_timestamp_high": 135,
    "reserved_begin": 136, "end": 136,
}
SUBTITLE_OCR_RECORD_SCHEMA = coordinate_contract.SUBTITLE_OCR.record_schema
SUBTITLE_OCR_RECORD_TAG = coordinate_contract.SUBTITLE_OCR.record_tag
SUBTITLE_OCR_RECORD_WORD_COUNT = coordinate_contract.SUBTITLE_OCR.record_word_count
SUBTITLE_OCR_HEADER_WORD_COUNT = coordinate_contract.SUBTITLE_OCR.record_header_word_count
SUBTITLE_OCR_RAW_BOX_CAPACITY = coordinate_contract.SUBTITLE_OCR.raw_box_capacity
SUBTITLE_OCR_FINAL_BOX_CAPACITY = coordinate_contract.SUBTITLE_OCR.final_box_capacity
SUBTITLE_OCR_BOX_WORD_COUNT = coordinate_contract.SUBTITLE_OCR.box_word_count
SUBTITLE_OCR_BOX_FLAG_RIBBON = coordinate_contract.SUBTITLE_OCR.box_flag_ribbon
SUBTITLE_OCR_BOX_KNOWN_FLAGS = coordinate_contract.SUBTITLE_OCR.box_known_flags
SUBTITLE_OCR_RAW_BOX_WORD_OFFSET = coordinate_contract.SUBTITLE_OCR.raw_box_offset
SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET = coordinate_contract.SUBTITLE_OCR.final_box_offset
SUBTITLE_OCR_OUTPUT_WIDTH = coordinate_contract.SUBTITLE_OCR.output_shape[3]
SUBTITLE_PLANE_STATE_SCHEMA = coordinate_contract.SUBTITLE_OCR.plane_schema
SUBTITLE_PLANE_STATE_TAG = coordinate_contract.SUBTITLE_OCR.plane_tag
SUBTITLE_PLANE_STATE_WORD_COUNT = coordinate_contract.SUBTITLE_OCR.plane_word_count
SUBTITLE_PLANE_COVER_CAPACITY = coordinate_contract.SUBTITLE_OCR.plane_cover_capacity
SUBTITLE_PLANE_CURRENT_VALID_FLAG = coordinate_contract.SUBTITLE_OCR.plane_current_valid_flag
SUBTITLE_PLANE_ADAPTIVE_OFFSET = coordinate_contract.SUBTITLE_OCR.plane_adaptive_offset
DEPTH_INPUT_REGION_SCHEMA = 4
WINDOW_REGION_SCHEMA = 1
WINDOW_REGION_AUTHORITY_KINDS = frozenset({"chromium-video", "foreground-client"})
SHADOW_STATE_DUMP_SCHEMA = 18
SHADOW_FRAME_STATS_DUMP_SCHEMA = 3
LIVE_RENDERER_SOURCE_CLOSURE_SHA256 = (
    shader_manifest.PARALLAX_V2_LIVE_RENDERER_GROUP.source_closure_sha256)
DIAGNOSTIC_SOURCE_CLOSURE_SHA256 = (
    shader_manifest.PARALLAX_V2_LIVE_DIAGNOSTIC_GROUP.source_closure_sha256)
_CONTRACT = coordinate_contract.load_contract()
_CONTRACT_TAG = generator.contract_tag(_CONTRACT)
_STATE_FIELDS = tuple(_CONTRACT["shadow_state"]["fields"])
_FRAME_STATS_FIELDS = tuple(_CONTRACT["frame_stats"]["fields"])
_DEFAULTS = coordinate_contract.CALIBRATED_DEFAULTS
_RESERVED_CALIBRATION_REVISION = 0xFFFFFFFF
_ROI_EXTERIOR_ZERO_TOLERANCE_OUTPUT_EYE_PX = 0.005
_RAW_COORDINATE_SCALE_AUTHENTICATION_TOLERANCE = 2.0e-6
_REQUESTED_GAIN_AUTHENTICATION_TOLERANCE = 1.0e-7
_PRODUCTION_DEPTH_SHORT_SIDE = 432
_PRODUCTION_DEPTH_MAX_ASPECT = 4.0
_PRODUCTION_CALIBRATION = coordinate_contract.MODEL_CALIBRATIONS[0]
_AUTHENTICATED_DAV2_TENSOR_SHAPES = frozenset(
    _PRODUCTION_CALIBRATION.calibrated_input_shapes)
_AUTHENTICATED_SINGLE_HIGH_TENSOR_SHAPES = frozenset(
    (shape.width, shape.height)
    for shape in convex2x_contract.supported_high_shapes())
_COMPOSITE_RUNTIME_PROVENANCE_KEYS = {
    "schema", "runtime", "model", "onnx_sha256", "embedded_dav2_onnx_sha256",
    "zipdepth_checkpoint_sha256", "guidance_preprocess_source_closure_sha256",
    "engine_recipe", "engine_artifact", "active_engine_manifest",
}
_MAXIMUM_SOURCE_LONG_SIDE = 5120
_MAXIMUM_SOURCE_PIXELS = 5120 * 2160

_SUBTITLE_MODE_NONE = "none"
_SUBTITLE_MODE_CONTOURS = "subtitle-contours-v1"
_SUBTITLE_OCR_CONTRACT_SCHEMA = coordinate_contract.SUBTITLE_OCR.schema
_SUBTITLE_OCR_MODEL_NAME = coordinate_contract.SUBTITLE_OCR.logical_model
_SUBTITLE_OCR_ASSET_PATH = coordinate_contract.SUBTITLE_OCR.asset_path
_SUBTITLE_OCR_ARTIFACT_ONNX_SHA256 = (
    coordinate_contract.SUBTITLE_OCR.artifact_onnx_sha256)
_SUBTITLE_OCR_SOURCE_URL = coordinate_contract.SUBTITLE_OCR.source_url
_SUBTITLE_OCR_SOURCE_ONNX_SHA256 = coordinate_contract.SUBTITLE_OCR.source_onnx_sha256
_SUBTITLE_OCR_CONVERSION_TOOL = coordinate_contract.SUBTITLE_OCR.conversion_tool
_SUBTITLE_OCR_CONVERSION_VERSION = coordinate_contract.SUBTITLE_OCR.conversion_version
_SUBTITLE_OCR_CONVERSION_RECIPE = coordinate_contract.SUBTITLE_OCR.conversion_recipe
_SUBTITLE_OCR_CONVERSION_CALIBRATION_PROFILE = (
    coordinate_contract.SUBTITLE_OCR.conversion_calibration_profile)
_SUBTITLE_OCR_ENGINE_RECIPE = coordinate_contract.SUBTITLE_OCR.engine_recipe
_SUBTITLE_OCR_SHADER_SPECS = (
    ("host_sbs_ocr_preprocess_cs.hlsl", "main", "cs_5_0"),
    ("host_sbs_ocr_boxes_cs.hlsl", "cells_main", "cs_5_0"),
    ("host_sbs_ocr_boxes_cs.hlsl", "resolve_main", "cs_5_0"),
)
_SUBTITLE_PLANE_SHADER_SPECS = tuple(
    spec for spec in coordinate_contract.SHADER_SOURCE_SPECS
    if spec[0] in {
        "host_sbs_subtitle_locator_cs.hlsl", "host_sbs_subtitle_contour_cs.hlsl",
        "host_sbs_subtitle_contour_distance_cs.hlsl",
    } or (spec[0] in {
        "depth_coordinate_v2_vertical_limit_cs.hlsl", "depth_coordinate_v2_limit_cs.hlsl"
    } and spec[1] == "subtitle_main")
)
_SUBTITLE_CONTOUR_ARTIFACT_FILES = {
    "contour_support": "contour_support.u32",
    "contour_field_pin": "contour_field_pin.u32",
    "contour_distance": "contour_distance.f32",
    "contour_identity": "contour_identity.u32",
    "contour_history": "contour_history.rgba32f",
    "contour_source": "contour_source.rgba16f",
    "ocr_probability": "ocr_probability.f32",
    "gaussian_weight": "gaussian_weight.f32",
    "gaussian_blend": "gaussian_blend.f32",
    "gaussian_vertical_conditioned": "gaussian_vertical_conditioned.f32",
    "gaussian_control": "gaussian_control.u32",
}
_SUBTITLE_CONTOUR_ARTIFACT_STAGES = {
    "contour_support": "native current subtitle contour support",
    "contour_field_pin": "projected subtitle contour field pins",
    "contour_distance": "native-source squared Euclidean subtitle distance",
    "contour_identity": "SCT5 subtitle temporal-mask identity",
    "contour_history": "native subtitle contour history",
    "contour_source": "joined subtitle source display crop",
    "ocr_probability": "joined OCR detector probability",
    "gaussian_weight": "literal max-union Gaussian subtitle weight",
    "gaussian_blend": "literal Gaussian UI-Base proposal",
    "gaussian_vertical_conditioned": "post-Gaussian shared vertical75-25 field",
    "gaussian_control": "actual Gaussian smoothing-enabled control",
}


def _float32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("value is not representable as finite float32") from error
    if not math.isfinite(result):
        raise ValueError("value is not representable as finite float32")
    return result


_STATE_ROOT_KEYS = {
    "schema", "coordinate_contract", "source", "capture", "rendered_output_selected",
    "wire_contract", "units", "constants", "fields", "named_values", "decoded",
    "adaptation_semantics",
}
_STATE_CONSTANT_KEYS = {
    "raw_coordinate_scale", "collapse_abs_epsilon",
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
    "joint_plane_mode",
}
_DECODED_FLOAT_KEYS = {
    "requested_gain", "requested_pop_strength", "latched_scale",
    "convergence_curve", "container_scale", "effective_gain",
}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _finite_float32(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    try:
        return _float32(result)
    except ValueError as error:
        raise ValueError(f"{label} must be a finite float32") from error


def _display_budget_for_mode(requested_gain: Any, mode: int | None) -> float:
    if type(mode) is not int or mode != 3:
        raise ValueError("invalid joint plane mode")
    return _float32(_DEFAULTS.direct_container_limit)


def display_budget_for_mode(requested_gain: float, mode: int) -> float:
    """Return the immutable one-eye representation bound, independent of strength."""
    return _display_budget_for_mode(requested_gain, mode)


def adaptation_semantics_for_mode(mode: int) -> Dict[str, str]:
    if type(mode) is not int or mode != 3:
        raise ValueError("invalid adaptive policy identity; expected 3")
    return {
        "coordinate": "bounded-source-time-linear-raw-mean-and-robust-amplitude-reference",
        "convergence_curve": "continuously-tracked-mean-is-zero-plane-no-cut-latch",
        "requested_gain": "immutable-cfg-pop-strength-independent-of-display-limit",
        "container_scale": "abi-retained-identity-mode3-hard-representation-cap",
        "near_curve": "linear-coordinate-no-depth-curve-before-hard-representation-cap",
        "spatial_conditioner": "fixed-75pct-vertical-majorant-share-then-horizontal-majorant",
    }


def _same_serialized_number(left: Any, right: Any) -> bool:
    """Compare redundant JSON numbers exactly, including the sign of zero."""

    if left != right:
        return False
    if left == 0:
        return math.copysign(1.0, float(left)) == math.copysign(1.0, float(right))
    return True


def shadow_state_constant_float32(document: Any, name: str) -> float:
    """Return one serialized shadow-state constant with native float32 semantics."""

    constants = document.get("constants") if isinstance(document, dict) else None
    if (name not in _STATE_CONSTANT_KEYS or not isinstance(constants, dict) or
            name not in constants):
        raise ValueError(f"shadow_state.json has no known constant {name!r}")
    return _finite_float32(
        constants[name], f"shadow_state.json constants.{name}")


def shadow_frame_valid_from_statistics(
        state_document: Any, statistics: Dict[str, float]) -> bool:
    """Mirror native moments, mandatory quantiles and finite robust-amplitude admission."""
    collapse = shadow_state_constant_float32(state_document, "collapse_abs_epsilon")
    raw_scale = shadow_state_constant_float32(state_document, "raw_coordinate_scale")
    try:
        minimum, maximum, mean, std = (statistics[name] for name in (
            "minimum", "maximum", "mean", "population_std"))
        low, high, bin_width = (statistics[name] for name in (
            "percentile_low", "percentile_high", "percentile_bin_width"))
        half_span = _float32(_float32(high - low) * _float32(.5))
        reference = _float32(max(half_span, raw_scale))
        inverse = _float32(1.0 / reference) if reference > 0.0 else 0.0
        return bool(
            statistics["valid"] == 1.0 and statistics["percentile_valid"] == 1.0 and
            all(math.isfinite(value) for value in (
                minimum, maximum, mean, std, low, high, bin_width,
                half_span, raw_scale, reference, inverse)) and
            maximum > minimum and minimum <= mean <= maximum and std > collapse and
            minimum <= low <= high <= maximum and bin_width >= 0.0 and
            raw_scale > 0.0 and reference > 0.0 and inverse > 0.0)
    except (KeyError, OverflowError, ValueError, ZeroDivisionError):
        return False


def _uint32(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{label} must be a uint32")
    return value


def _uint64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{label} must be a uint64")
    return value


def _uint32_words(payload: Any, word_count: int, label: str) -> tuple[int, ...]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError(f"{label} must be a little-endian uint32 byte record")
    raw = bytes(payload)
    expected_size = word_count * 4
    if len(raw) != expected_size:
        raise ValueError(f"{label} must contain exactly {word_count} uint32 words")
    return struct.unpack(f"<{word_count}I", raw)


def _uint64_words(low: int, high: int) -> int:
    return low | (high << 32)


def _float32_bits(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _subtitle_tensor_content(
        field_width: int, field_height: int,
        tensor_content: tuple[int, int, int, int] | None
        ) -> tuple[int, int, int, int]:
    """Validate the real-source half-open rectangle inside one DAV2 tensor."""

    content = ((0, 0, field_width, field_height)
               if tensor_content is None else tensor_content)
    if (not isinstance(content, tuple) or len(content) != 4 or
            any(isinstance(value, bool) or not isinstance(value, int) for value in content)):
        raise ValueError("subtitle tensor content must be an integer half-open rectangle")
    left, top, right, bottom = content
    if (left < 0 or top < 0 or left >= right or top >= bottom or
            right > field_width or bottom > field_height):
        raise ValueError("subtitle tensor content is outside the DAV2 field")
    return content


def _subtitle_project_row_ceil(
        source_width: int, source_height: int,
        tensor_content: tuple[int, int, int, int], detector_y: int) -> int | None:
    """Project a detector edge into the real tensor content using production integers."""

    if (source_width <= 0 or source_height <= 0 or detector_y < 0 or
            detector_y > coordinate_contract.SUBTITLE_OCR.output_shape[2]):
        return None
    content_height = tensor_content[3] - tensor_content[1]
    crop_height = min(
        source_height,
        (source_width * coordinate_contract.SUBTITLE_OCR.source_crop_aspect_height +
         coordinate_contract.SUBTITLE_OCR.source_crop_aspect_width - 1) //
        coordinate_contract.SUBTITLE_OCR.source_crop_aspect_width,
    )
    crop_top = source_height - crop_height
    denominator = coordinate_contract.SUBTITLE_OCR.output_shape[2] * source_height
    numerator = (
        (crop_top * coordinate_contract.SUBTITLE_OCR.output_shape[2] +
         detector_y * crop_height) * content_height
    )
    projected = numerator // denominator + int(numerator % denominator != 0)
    return tensor_content[1] + min(projected, content_height)


def _subtitle_dynamic_roi(
        source_width: int, source_height: int,
        tensor_content: tuple[int, int, int, int]) -> tuple[int, int] | None:
    top = _subtitle_project_row_ceil(
        source_width, source_height, tensor_content,
        coordinate_contract.SUBTITLE_OCR.ocr_safe_row_top)
    bottom = _subtitle_project_row_ceil(
        source_width, source_height, tensor_content,
        coordinate_contract.SUBTITLE_OCR.ocr_safe_row_bottom)
    return None if top is None or bottom is None or top >= bottom else (top, bottom)


def _subtitle_ribbon_min_bottom(
        source_width: int, source_height: int,
        tensor_content: tuple[int, int, int, int]) -> int | None:
    return _subtitle_project_row_ceil(
        source_width, source_height, tensor_content,
        coordinate_contract.SUBTITLE_OCR.ocr_safe_row_bottom -
        coordinate_contract.SUBTITLE_OCR.ribbon_bottom_tolerance_pixels)


def _decode_subtitle_ocr_boxes(
        words: tuple[int, ...], *, offset: int, count: int, capacity: int,
        field_width: int, field_height: int, roi_top: int, roi_bottom: int,
        tensor_content: tuple[int, int, int, int], ribbon_min_bottom: int,
        final_boxes: bool, label: str) -> list[Dict[str, Any]]:
    decoded: list[Dict[str, Any]] = []
    for slot in range(capacity):
        start = offset + slot * SUBTITLE_OCR_BOX_WORD_COUNT
        values = words[start:start + SUBTITLE_OCR_BOX_WORD_COUNT]
        if slot >= count:
            if any(values):
                raise ValueError(f"{label} unused box slots must be canonical zero")
            continue
        left, top, right, bottom, score_bits, box_flags, island_count, gap_count = values
        score = _float32_bits(score_bits)
        content_left, content_top, content_right, content_bottom = tensor_content
        if (left >= right or top >= bottom or left < content_left or
                top < content_top or right > content_right or
                bottom > content_bottom or top < roi_top):
            raise ValueError(
                f"{label} boxes must be nonempty half-open rectangles inside the field")
        minimum_score = coordinate_contract.SUBTITLE_OCR.detector_min_mean_score
        if not math.isfinite(score) or not minimum_score <= score <= 1.0:
            raise ValueError(
                f"{label} box scores must be finite float32 values in "
                f"[{minimum_score},1.0]")
        if box_flags & ~SUBTITLE_OCR_BOX_KNOWN_FLAGS:
            raise ValueError(f"{label} boxes have unknown kind flags")
        if (island_count == 0 or island_count > SUBTITLE_OCR_OUTPUT_WIDTH or
                gap_count >= island_count):
            raise ValueError(f"{label} boxes have invalid topology counts")
        ribbon = bool(box_flags & SUBTITLE_OCR_BOX_FLAG_RIBBON)
        if ribbon:
            if (island_count < coordinate_contract.SUBTITLE_OCR.ribbon_min_structural_gaps + 1 or
                    gap_count < coordinate_contract.SUBTITLE_OCR.ribbon_min_structural_gaps):
                raise ValueError(f"{label} ribbon topology is inconsistent")
            if final_boxes:
                if (left != content_left or right != content_right or
                        bottom != content_bottom or
                        top < roi_top):
                    raise ValueError(f"{label} ribbon cover is not the canonical bottom strip")
            elif (bottom < ribbon_min_bottom or bottom > roi_bottom or
                  (right - left) * coordinate_contract.SUBTITLE_OCR.ribbon_min_width_denominator <
                  (content_right - content_left) *
                  coordinate_contract.SUBTITLE_OCR.ribbon_min_width_numerator):
                raise ValueError(
                    f"{label} ribbon core is outside its projected bottom tolerance or not wide")
        elif bottom > roi_bottom:
            raise ValueError(f"{label} ordinary boxes must remain inside the OCR ROI")
        decoded.append({
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "score": score,
            "score_bits": score_bits,
            "box_flags": box_flags,
            "kind": "ribbon" if ribbon else "text",
            "island_count": island_count,
            "structural_gap_count": gap_count,
        })
    return decoded


def validate_subtitle_ocr_record(
        payload: Any, *, matched_frame_id: int, analysis_generation: int,
        source_width: int, source_height: int, field_width: int, field_height: int,
        roi_top: int, roi_bottom: int,
        tensor_content: tuple[int, int, int, int] | None = None) -> Dict[str, Any]:
    """Validate the retained OCR8 pairs; preselection line membership is not in this ABI.

    Center eligibility uses a group's complete tight-core union before capacity selection.
    Individual retained fragments may be off center, and omitted group members cannot be
    reconstructed from these words. The authenticated producer contract owns that rule.
    """

    expected_frame = _uint64(matched_frame_id, "OCR8 matched frame id")
    expected_generation = _uint64(analysis_generation, "OCR8 analysis generation")
    expected_source_width = _uint32(source_width, "OCR8 source width")
    expected_source_height = _uint32(source_height, "OCR8 source height")
    expected_field_width = _uint32(field_width, "OCR8 field width")
    expected_field_height = _uint32(field_height, "OCR8 field height")
    expected_roi_top = _uint32(roi_top, "OCR8 ROI top")
    expected_roi_bottom = _uint32(roi_bottom, "OCR8 ROI bottom")
    expected_content = _subtitle_tensor_content(
        expected_field_width, expected_field_height, tensor_content)
    if (expected_source_width == 0 or expected_source_height == 0 or
            expected_field_width == 0 or expected_field_height == 0 or
            not coordinate_contract.subtitle_ocr_field_is_calibrated(
                expected_field_width, expected_field_height) or
            expected_roi_top >= expected_roi_bottom or
            expected_roi_bottom > expected_field_height):
        raise ValueError("OCR8 expected source, field, or ROI geometry is invalid")
    dynamic_roi = _subtitle_dynamic_roi(
        expected_source_width, expected_source_height, expected_content)
    if dynamic_roi != (expected_roi_top, expected_roi_bottom):
        raise ValueError(
            "OCR8 expected field/ROI geometry does not match the calibrated dynamic policy")
    ribbon_min_bottom = _subtitle_ribbon_min_bottom(
        expected_source_width, expected_source_height, expected_content)
    if ribbon_min_bottom is None or not expected_roi_top < ribbon_min_bottom <= expected_roi_bottom:
        raise ValueError("OCR8 projected ribbon bottom tolerance is invalid")

    words = _uint32_words(payload, SUBTITLE_OCR_RECORD_WORD_COUNT, "OCR8 record")
    if words[0] != SUBTITLE_OCR_RECORD_SCHEMA or words[1] != SUBTITLE_OCR_RECORD_TAG:
        raise ValueError("OCR8 record has an unknown schema or tag")
    flags = words[2]
    raw_count = words[3]
    final_count = words[4]
    if flags not in (0, 1):
        raise ValueError("OCR8 flags must be exactly zero (abstain) or one (authoritative)")
    if (raw_count > SUBTITLE_OCR_RAW_BOX_CAPACITY or
            final_count > SUBTITLE_OCR_FINAL_BOX_CAPACITY):
        raise ValueError("OCR8 record exceeds its fixed box capacity")
    if final_count != raw_count:
        raise ValueError("OCR8 raw and final box counts must match exactly")
    if flags == 0 and (raw_count != 0 or final_count != 0):
        raise ValueError("an abstaining OCR8 record cannot publish boxes")
    if words[15] != 0:
        raise ValueError("OCR8 reserved header word must be zero")

    frame_id = _uint64_words(words[5], words[6])
    generation = _uint64_words(words[7], words[8])
    identity = {
        "matched_frame_id": frame_id,
        "analysis_generation": generation,
        "source_width": words[9],
        "source_height": words[10],
        "field_width": words[11],
        "field_height": words[12],
        "roi_top": words[13],
        "roi_bottom": words[14],
    }
    expected_identity = {
        "matched_frame_id": expected_frame,
        "analysis_generation": expected_generation,
        "source_width": expected_source_width,
        "source_height": expected_source_height,
        "field_width": expected_field_width,
        "field_height": expected_field_height,
        "roi_top": expected_roi_top,
        "roi_bottom": expected_roi_bottom,
    }
    if identity != expected_identity:
        raise ValueError("OCR8 record identity disagrees with the matched dump frame")

    raw_boxes = _decode_subtitle_ocr_boxes(
        words, offset=SUBTITLE_OCR_RAW_BOX_WORD_OFFSET, count=raw_count,
        capacity=SUBTITLE_OCR_RAW_BOX_CAPACITY, field_width=expected_field_width,
        field_height=expected_field_height, roi_top=expected_roi_top,
        roi_bottom=expected_roi_bottom, tensor_content=expected_content,
        ribbon_min_bottom=ribbon_min_bottom,
        final_boxes=False, label="OCR8 raw")
    final_boxes = _decode_subtitle_ocr_boxes(
        words, offset=SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET, count=final_count,
        capacity=SUBTITLE_OCR_FINAL_BOX_CAPACITY, field_width=expected_field_width,
        field_height=expected_field_height, roi_top=expected_roi_top,
        roi_bottom=expected_roi_bottom, tensor_content=expected_content,
        ribbon_min_bottom=ribbon_min_bottom,
        final_boxes=True, label="OCR8 final")
    for index, (core, cover) in enumerate(zip(raw_boxes, final_boxes)):
        metadata_keys = (
            "score_bits", "box_flags", "island_count", "structural_gap_count")
        if any(core[key] != cover[key] for key in metadata_keys):
            raise ValueError(f"OCR8 pair {index} metadata does not match")
        if (cover["left"] > core["left"] or cover["top"] > core["top"] or
                cover["right"] < core["right"] or cover["bottom"] < core["bottom"]):
            raise ValueError(f"OCR8 pair {index} cover does not contain its core")
    return {
        "schema": words[0],
        "tag": words[1],
        "authoritative": flags == 1,
        "flags": flags,
        "raw_count": raw_count,
        "final_count": final_count,
        **identity,
        "raw_boxes": raw_boxes,
        "final_boxes": final_boxes,
        "tensor_content_rect": expected_content,
        "ordinary_cover_horizontal_guard": {
            "cells": coordinate_contract.SUBTITLE_OCR.ordinary_cover_horizontal_guard_cells,
            "policy": coordinate_contract.SUBTITLE_OCR.ordinary_cover_horizontal_guard_policy,
        },
        "line_centering": {
            "group_policy": coordinate_contract.SUBTITLE_OCR.line_group_policy,
            "minimum_vertical_span_iou_ratio": [
                coordinate_contract.SUBTITLE_OCR.line_group_min_vertical_iou_numerator,
                coordinate_contract.SUBTITLE_OCR.line_group_min_vertical_iou_denominator],
            "maximum_horizontal_gap_height_ratio": [
                coordinate_contract.SUBTITLE_OCR.line_group_max_horizontal_gap_height_numerator,
                coordinate_contract.SUBTITLE_OCR.line_group_max_horizontal_gap_height_denominator],
            "horizontal_gap_metric": coordinate_contract.SUBTITLE_OCR.line_group_horizontal_gap_metric,
            "center_reference": coordinate_contract.SUBTITLE_OCR.line_center_reference,
            "reference_params": coordinate_contract.SUBTITLE_OCR.line_center_reference_params,
            "center_tolerance_ratio": [
                coordinate_contract.SUBTITLE_OCR.line_center_tolerance_numerator,
                coordinate_contract.SUBTITLE_OCR.line_center_tolerance_denominator],
            "search_bottom_height_ratio": [
                coordinate_contract.SUBTITLE_OCR.line_search_bottom_height_numerator,
                coordinate_contract.SUBTITLE_OCR.line_search_bottom_height_denominator],
            "search_region": coordinate_contract.SUBTITLE_OCR.line_search_region,
            "preselection_geometry_verified": False,
            "verification_limit": "retained-pairs-do-not-encode-complete-preselection-line-membership-or-reference",
        },
    }


def _validate_adaptive_ui_tail(words: tuple[int, ...], source_width: int,
                               content: tuple[int, int, int, int], *,
                               joint_plane_mode: int | None = 3,
                               requested_gain: float | None = None) -> None:
    display_budget = _display_budget_for_mode(requested_gain, joint_plane_mode)
    tail = words[SUBTITLE_PLANE_ADAPTIVE_OFFSET:SUBTITLE_PLANE_STATE_WORD_COUNT]
    if tail[0] == 0:
        if any(tail):
            raise ValueError("UIP4 inactive adaptive UI tail must be zero")
        if words[coordinate_contract.SUBTITLE_OCR.plane_current_slot_mask_word] != 0:
            raise ValueError("UIP4 current cover requires an active adaptive UI tail")
        return
    if not 0 < source_width <= _MAXIMUM_SOURCE_LONG_SIDE:
        raise ValueError("UIP4 mode3 source width exceeds its supported histogram range")
    maximum_level = (source_width + 99) // 100
    if (tail[0] != 1 or tail[1] & ~15 or (tail[1] & 3) == 3 or
            any(tail[i] > maximum_level for i in (2, 10, 11))):
        raise ValueError("UIP4 adaptive UI schema, flags, or index is invalid")
    applied, limit = _float32_bits(tail[3]), _float32_bits(tail[12])
    expected_limit = _float32(display_budget)
    # The representation constant is copied directly, so it requires its exact bits.
    expected_limit_bits = struct.unpack("<I", struct.pack("<f", expected_limit))[0]
    area = (content[2] - content[0]) * (content[3] - content[1])
    last = _uint64_words(tail[4], tail[5])
    approach = _uint64_words(tail[6], tail[7])
    retreat = _uint64_words(tail[8], tail[9])
    if (not math.isfinite(applied) or not math.isfinite(limit) or
            tail[12] != expected_limit_bits or
            not 0.0 <= applied <= limit or
            tail[13] > area or tail[14] > tail[13] or tail[15] > tail[13] or
            (bool(tail[1] & 8) != (last != 0)) or
            (bool(tail[1] & 1) != (approach != 0)) or
            (bool(tail[1] & 2) != (retreat != 0)) or
            approach > last or retreat > last):
        raise ValueError("UIP4 adaptive UI range, clock, or count is invalid")
    capped_conflict = tail[14] * 5 > tail[13] or tail[14] * 50 > area
    if bool(tail[1] & 4) != capped_conflict:
        raise ValueError("UIP4 adaptive UI capped-conflict flag does not match its counts")


def validate_subtitle_plane_snapshot(
        payload: Any, *, matched_frame_id: int, analysis_generation: int,
        source_width: int, source_height: int, field_width: int, field_height: int,
        tensor_content: tuple[int, int, int, int] | None = None,
        expected_scene_epoch: int | None = None,
        expected_observation_timestamp_us: int | None = None,
        joint_plane_mode: int = 3, requested_gain: float | None = None) -> Dict[str, Any]:
    """Authenticate UIP4 layout/domain/controller only; it contains no cover geometry.

    A trace snapshot cannot establish actual OCR8 covers. Call validate_subtitle_plane_state
    with the independently validated current OCR8 record to authenticate those covers.
    """
    policy = coordinate_contract.SUBTITLE_OCR
    words = _uint32_words(payload, SUBTITLE_PLANE_STATE_WORD_COUNT, "UIP4 plane state")
    expected_frame = _uint64(matched_frame_id, "UIP4 matched frame")
    expected_generation = _uint64(analysis_generation, "UIP4 analysis generation")
    expected_extents = tuple(_uint32(value, "UIP4 extent") for value in
                             (source_width, source_height, field_width, field_height))
    if (not expected_frame or any(value == 0 for value in expected_extents) or
            words[0] != SUBTITLE_PLANE_STATE_SCHEMA or words[1] != SUBTITLE_PLANE_STATE_TAG):
        raise ValueError("UIP4 state has invalid schema/tag/expected domain")
    flags = words[policy.plane_flags_word]
    mask = words[policy.plane_current_slot_mask_word]
    # Legacy API names remain compatible: this is a current-core probability cue mask,
    # not proof that any qualifying same-pixel RGB reference or count was reset.
    resets = words[policy.plane_history_reset_mask_word]
    warmup = words[policy.plane_warmup_word]
    warmup_known_bits = policy.plane_warmup_remaining_mask | policy.plane_warmup_cue_bit
    capacity_mask = (1 << SUBTITLE_PLANE_COVER_CAPACITY) - 1
    if (warmup & ~warmup_known_bits or (warmup & policy.plane_warmup_remaining_mask) > policy.plane_warmup_observations or flags & ~SUBTITLE_PLANE_CURRENT_VALID_FLAG or mask & ~capacity_mask or
            mask & (mask + 1) or (flags == 0 and mask != 0) or resets & ~mask or
            any(words[policy.plane_reserved_offset:policy.plane_adaptive_offset])):
        raise ValueError("UIP4 flags/slot mask/reserved words are not canonical")
    frame = _uint64_words(*words[policy.plane_frame_offset:policy.plane_frame_offset + 2])
    generation = _uint64_words(*words[policy.plane_analysis_generation_offset:policy.plane_analysis_generation_offset + 2])
    timestamp = _uint64_words(*words[policy.plane_observation_timestamp_offset:policy.plane_observation_timestamp_offset + 2])
    content = _subtitle_tensor_content(field_width, field_height,
                                      tuple(words[policy.plane_content_offset:policy.plane_content_offset + 4]))
    if tensor_content is not None and content != _subtitle_tensor_content(field_width, field_height, tensor_content):
        raise ValueError("UIP4 content differs from the exact analysis domain")
    roi = _subtitle_dynamic_roi(source_width, source_height, content)
    if (frame != expected_frame or generation != expected_generation or timestamp == 0 or
            tuple(words[policy.plane_source_extent_offset:policy.plane_source_extent_offset + 2]) != expected_extents[:2] or
            tuple(words[policy.plane_field_extent_offset:policy.plane_field_extent_offset + 2]) != expected_extents[2:] or
            roi is None or tuple(words[policy.plane_roi_offset:policy.plane_roi_offset + 2]) != roi):
        raise ValueError("UIP4 frame/time/source/field/ROI identity differs from current publication")
    epoch = words[policy.plane_cut_epoch_word]
    if expected_scene_epoch is not None and epoch != _uint32(expected_scene_epoch, "UIP4 cut epoch"):
        raise ValueError("UIP4 cut epoch differs from authenticated geometry")
    if (expected_observation_timestamp_us is not None and
            timestamp != _uint64(expected_observation_timestamp_us, "UIP4 expected observation time")):
        raise ValueError("UIP4 observation timestamp differs from completed source")
    _validate_adaptive_ui_tail(words, source_width, content,
                              joint_plane_mode=joint_plane_mode, requested_gain=requested_gain)
    tail = words[policy.plane_adaptive_offset:]
    if warmup and (flags != SUBTITLE_PLANE_CURRENT_VALID_FLAG or mask == 0 or tail[0] != 1 or tail[15] != 0):
        raise ValueError("UIP4 warmup requires current nonempty authority and a valid probe")
    if resets and tail[15]:
        raise ValueError("UIP4 invalid Base probe cannot publish history resets")
    return {
        "schema": words[0], "tag": words[1], "flags": flags,
        "current_valid": flags == SUBTITLE_PLANE_CURRENT_VALID_FLAG,
        "current_slot_mask": mask, "current_count": mask.bit_count(),
        "history_reset_mask": resets,
        "warmup_packed": warmup, "warmup_remaining": warmup & policy.plane_warmup_remaining_mask,
        "warmup_previous_cue": bool(warmup & policy.plane_warmup_cue_bit),
        "matched_frame_id": frame, "analysis_generation": generation,
        "observation_timestamp_us": timestamp, "scene_epoch": epoch,
        "source_width": source_width, "source_height": source_height,
        "field_width": field_width, "field_height": field_height,
        "roi_top": roi[0], "roi_bottom": roi[1], "tensor_content_rect": content,
        "target": _float32_bits(tail[3]) if tail[0] else None,
        "target_bits": tail[3], "joint_plane_mode": joint_plane_mode,
        "display_budget": _display_budget_for_mode(requested_gain, joint_plane_mode),
        "adaptive": {
            "initialized": tail[0] == 1, "flags": tail[1], "current_level": tail[2],
            "accepted_observation_timestamp_us": _uint64_words(tail[4], tail[5]),
            "approach_since_us": _uint64_words(tail[6], tail[7]),
            "retreat_since_us": _uint64_words(tail[8], tail[9]),
            "approach_level": tail[10], "retreat_level": tail[11],
            "display_budget": _float32_bits(tail[12]),
            "covered_count": tail[13], "conflict_count": tail[14], "invalid_count": tail[15],
        },
        "cover_geometry_verified": False,
    }


def validate_subtitle_plane_state(payload: Any, *, ocr_record: Dict[str, Any], **context: Any) -> Dict[str, Any]:
    """Join UIP4 current slots to every paired cover in the exact validated current OCR8 record."""
    state = validate_subtitle_plane_snapshot(payload, **context)
    if not isinstance(ocr_record, dict) or type(ocr_record.get("authoritative")) is not bool:
        raise ValueError("UIP4 requires the independently validated current OCR8 record")
    for key in ("matched_frame_id", "analysis_generation", "source_width", "source_height",
                "field_width", "field_height", "roi_top", "roi_bottom", "tensor_content_rect"):
        if ocr_record.get(key) != state[key]:
            raise ValueError("UIP4 and OCR8 frame/domain identity differs")
    count = _uint32(ocr_record.get("final_count"), "OCR8 current cover count")
    boxes = ocr_record.get("final_boxes")
    if not isinstance(boxes, list) or len(boxes) != count or count > SUBTITLE_PLANE_COVER_CAPACITY:
        raise ValueError("UIP4 requires all exact current OCR8 covers")
    if state["current_valid"]:
        if not ocr_record["authoritative"] or state["current_slot_mask"] != (1 << count) - 1:
            raise ValueError("UIP4 current slots differ from authoritative current OCR8")
        current = [dict(box) for box in boxes]
    else:
        current = []
    kinds = sum(1 << index for index, box in enumerate(current) if box["kind"] == "ribbon")
    return {**state, "current_rectangles": current, "current_ribbon_mask": kinds,
            "cover_geometry_verified": True}


def validate_subtitle_condition_words(payload: Any, *, plane_state: Dict[str, Any],
                                     geometry_ready: bool, require_ocr_kinds: bool = True) -> tuple[int, ...]:
    """Authenticate SCP4 without granting geometry from a plane-only trace snapshot."""
    words = _uint32_words(payload, coordinate_contract.SUBTITLE_OCR.condition_param_word_count, "SCP4 condition")
    if words[:2] != (coordinate_contract.SUBTITLE_OCR.condition_param_schema,
                     coordinate_contract.SUBTITLE_OCR.condition_param_tag):
        raise ValueError("SCP4 condition has an invalid schema/tag")
    count = plane_state["current_count"]
    active = bool(geometry_ready and plane_state["current_valid"] and count and
                  plane_state["adaptive"]["invalid_count"] == 0)
    if not active:
        if any(words[2:]):
            raise ValueError("SCP4 inactive payload must be canonical positive zero")
    elif (words[2] != count or words[3] >> count or words[4] != 2 or
          words[5] != plane_state["target_bits"]):
        raise ValueError("SCP4 active payload differs from UIP4 slots/applied plane")
    elif require_ocr_kinds and (not plane_state.get("cover_geometry_verified") or
                               words[3] != plane_state.get("current_ribbon_mask")):
        raise ValueError("SCP4 ribbon kinds require actual current OCR8 covers")
    return words


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


def fit_host_sbs_v2_depth_tensor_shape(width: int, height: int) -> tuple[int, int]:
    """Return the exact production DAV2 coarse shape derived from a source extent."""

    patch = _PRODUCTION_CALIBRATION.preprocess.patch_multiple
    maximum = _PRODUCTION_CALIBRATION.preprocess.maximum_dimension

    def round_to_patch(value: float) -> int:
        # All inputs are positive. floor(x + .5) therefore mirrors C++ std::round.
        quotient = _float32(_float32(value) / _float32(float(patch)))
        return max(patch, int(math.floor(quotient + 0.5)) * patch)

    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
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


def expected_capture_grid_for_source(
        source_width: int, source_height: int, *, scale: int) -> tuple[int, int] | None:
    """Return the exact calibrated coarse or single-high grid for a source extent.

    ``scale=1`` selects the legacy DAV2 grid and ``scale=2`` selects the fused public grid.
    Sources whose fitted DAV2 shape is not one of the calibrated point profiles have no
    production grid and return ``None``.
    """

    if type(scale) is not int or scale not in (1, 2):
        raise ValueError("capture-grid scale must be exactly 1 or 2")
    if (type(source_width) is not int or type(source_height) is not int or
            source_width <= 0 or source_height <= 0 or
            max(source_width, source_height) > _MAXIMUM_SOURCE_LONG_SIDE or
            source_width * source_height > _MAXIMUM_SOURCE_PIXELS):
        return None
    coarse = fit_host_sbs_v2_depth_tensor_shape(source_width, source_height)
    if coarse not in _AUTHENTICATED_DAV2_TENSOR_SHAPES:
        return None
    if scale == 1:
        return coarse
    high = (2 * coarse[0], 2 * coarse[1])
    return high if high in _AUTHENTICATED_SINGLE_HIGH_TENSOR_SHAPES else None


# Retain the private spelling for existing internal callers while keeping one implementation.
_fit_host_sbs_v2_depth_tensor_shape = fit_host_sbs_v2_depth_tensor_shape


def _capture_grid_relation(
        width: int, height: int) -> tuple[str, int, int, int] | None:
    """Return ``(kind, scale, DAV2 width, DAV2 height)`` for an exact live grid."""

    if (width, height) in _AUTHENTICATED_DAV2_TENSOR_SHAPES:
        return "legacy-dav2", 1, width, height
    if ((width, height) in _AUTHENTICATED_SINGLE_HIGH_TENSOR_SHAPES and
            width % 2 == 0 and height % 2 == 0 and
            (width // 2, height // 2) in _AUTHENTICATED_DAV2_TENSOR_SHAPES):
        return "single-high-convex2x", 2, width // 2, height // 2
    return None


def capture_grid_relation_for_source(
        source_width: int, source_height: int,
        width: int, height: int) -> tuple[str, int, int, int] | None:
    """Classify a grid only when it is the exact profile derived from its source extent."""

    relation = _capture_grid_relation(width, height)
    if relation is None:
        return None
    expected = expected_capture_grid_for_source(
        source_width, source_height, scale=relation[1])
    return relation if expected == (width, height) else None


def composite_engine_filename_matches(engine_name, model, recipe, onnx_sha256):
    """Strict current cache tokens or historical explicit device/ONNX names; never paths."""
    if not isinstance(engine_name, str):
        return False
    prefix = re.escape(f"{model}.{recipe}.")
    current = prefix + r"cache-[0-9a-f]{64}\.engine"
    legacy = (prefix + r"trt[0-9]+_[0-9]+_[0-9]+_[0-9]+-sm[0-9]+-gpu[0-9a-f]{1,16}" +
              re.escape(f"-onnx{onnx_sha256}.engine"))
    return bool(re.fullmatch(current, engine_name) or re.fullmatch(legacy, engine_name))


def validate_composite_runtime_provenance(
        value: Any, *, expected_dav2_onnx_sha256: str | None = None,
        expected_preprocess_source_closure_sha256: str | None = None) -> Dict[str, Any]:
    """Authenticate schema-2 fused runtime evidence against the frozen production contract."""

    if not isinstance(value, dict) or set(value) != _COMPOSITE_RUNTIME_PROVENANCE_KEYS:
        raise ValueError(
            "dump_manifest.json lacks exact schema-2 fused composite runtime provenance")
    contract = convex2x_contract.load_contract()
    sources = contract["sources"]
    fused = sources["fused_onnx"]
    dav2 = sources["dav2"]
    zipdepth = sources["zipdepth"]
    tensorrt = contract["tensorrt"]
    model = str(fused["logical_model"])
    onnx_sha256 = str(fused["sha256"])
    dav2_sha256 = str(dav2["onnx_sha256"])
    zipdepth_sha256 = str(zipdepth["checkpoint_sha256"])
    engine_recipe = str(tensorrt["engine_recipe"])
    engine_artifact = value.get("engine_artifact")
    active_manifest = f"{model}.active-engine.json"
    if (value.get("schema") != int(contract["schema"]) or
            value.get("runtime") != "dav2_zipdepth_convex2x_composite" or
            value.get("model") != model or
            value.get("onnx_sha256") != onnx_sha256 or
            value.get("embedded_dav2_onnx_sha256") != dav2_sha256 or
            value.get("zipdepth_checkpoint_sha256") != zipdepth_sha256 or
            value.get("guidance_preprocess_source_closure_sha256") !=
            _PRODUCTION_CALIBRATION.preprocess.source_closure_sha256 or
            value.get("engine_recipe") != engine_recipe or
            not composite_engine_filename_matches(engine_artifact, model, engine_recipe, onnx_sha256) or
            value.get("active_engine_manifest") != active_manifest):
        raise ValueError(
            "dump_manifest.json has unauthenticated schema-2 fused composite runtime provenance")
    if (expected_dav2_onnx_sha256 is not None and
            dav2_sha256 != expected_dav2_onnx_sha256):
        raise ValueError(
            "dump_manifest.json fused composite does not embed the captured DAV2 model")
    if (expected_preprocess_source_closure_sha256 is not None and
            value["guidance_preprocess_source_closure_sha256"] !=
            expected_preprocess_source_closure_sha256):
        raise ValueError(
            "dump_manifest.json fused composite does not bind the captured preprocess closure")
    return dict(value)


def _plan_host_sbs_v2_window_region(
        semantic: tuple[int, int, int, int], source_width: int, source_height: int,
        tensor_width: int, tensor_height: int
        ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], float] | None:
    """Mirror the exact-source, centered integer contain-fit used by production."""

    left, top, right, bottom = semantic
    width = right - left
    height = bottom - top
    grid = capture_grid_relation_for_source(
        source_width, source_height, tensor_width, tensor_height)
    if (width <= 0 or height <= 0 or right > source_width or bottom > source_height or
            grid is None or semantic == (0, 0, source_width, source_height)):
        return None

    # Production chooses the ROI contain fit in the calibrated embedded-DAV2 grid. The fused
    # single-high path then realizes that exact rectangle at 2x; planning independently in the
    # high grid would silently authorize rounding variants that the producer cannot emit.
    _, scale, plan_width, plan_height = grid

    content_left = 0
    content_top = 0
    content_right = plan_width
    content_bottom = plan_height
    if width * plan_height > height * plan_width:
        content_height = max(1, plan_width * height // width)
        content_top = (plan_height - content_height) // 2
        content_bottom = content_top + content_height
    elif width * plan_height < height * plan_width:
        content_width = max(1, plan_height * width // height)
        content_left = (plan_width - content_width) // 2
        content_right = content_left + content_width
    coarse_content = (content_left, content_top, content_right, content_bottom)
    content_area = (content_right - content_left) * (content_bottom - content_top)
    padded_fraction = _float32(
        1.0 - content_area / (plan_width * plan_height))
    content = tuple(scale * value for value in coarse_content)
    return semantic, content, padded_fraction


def _roi_renderer_constants(
        inference: tuple[int, int, int, int], source_width: int, source_height: int,
        tensor_content: tuple[int, int, int, int]) -> tuple[float, float]:
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
        _float32(_float32(float(tensor_content[3] - tensor_content[1])) /
                 max(_float32(float(tensor_content[2] - tensor_content[0])),
                     _float32(1.0))))
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
    if mode not in {"full-source", "window-region"}:
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
                "authority_kind", "observer_generation", "hwnd", "process_id",
                "document_id", "video_id"}:
            raise ValueError("depth_input_region.json has an unknown authorization layout")
        authority_kind = authorization_value.get("authority_kind")
        if authority_kind not in WINDOW_REGION_AUTHORITY_KINDS:
            raise ValueError("depth_input_region.json has an unknown authority kind")
        hwnd, hwnd_value = _canonical_hwnd(
            authorization_value.get("hwnd"), "depth input HWND")
        authorization = {
            "authority_kind": authority_kind,
            "observer_generation": _uint64(
                authorization_value.get("observer_generation"), "observer generation"),
            "hwnd": hwnd_value,
            "hwnd_text": hwnd,
            "process_id": _uint32(authorization_value.get("process_id"), "process id"),
            "document_id": _int32(authorization_value.get("document_id"), "document id"),
            "video_id": _int32(authorization_value.get("video_id"), "video id"),
        }
        if (authorization["observer_generation"] == 0 or authorization["hwnd"] == 0 or
                authorization["process_id"] == 0):
            raise ValueError("depth_input_region.json has incomplete ROI authorization")
        has_dom_identity = authorization["document_id"] != 0 and authorization["video_id"] != 0
        if authority_kind == "chromium-video" and not has_dom_identity:
            raise ValueError("depth_input_region.json has incomplete Chromium ROI authorization")
        if authority_kind == "foreground-client" and (
                authorization["document_id"] != 0 or authorization["video_id"] != 0):
            raise ValueError("depth_input_region.json foreground authority carries DOM identity")

    analysis = document.get("analysis")
    if not isinstance(analysis, dict) or set(analysis) != {
            "analysis_generation", "tensor_extent_px", "tensor_content_rect_px",
            "padded_area_fraction", "fit_method", "crop_method",
            "scene_analysis_domain", "input_domain_reset"}:
        raise ValueError("depth_input_region.json has an unknown analysis layout")
    generation = _uint64(analysis.get("analysis_generation"), "analysis generation")
    tensor_w, tensor_h = _pixel_extent(
        analysis.get("tensor_extent_px"), "depth input tensor extent")
    if ((tensor_width is not None and tensor_w != tensor_width) or
            (tensor_height is not None and tensor_h != tensor_height)):
        raise ValueError("depth_input_region.json tensor extent does not match dumped geometry")
    if capture_grid_relation_for_source(width, height, tensor_w, tensor_h) is None:
        raise ValueError(
            "depth_input_region.json tensor extent does not match the exact source-derived "
            "calibrated DAV2 or single-high convex-2x grid")
    tensor_content = _pixel_rect(
        analysis.get("tensor_content_rect_px"), "depth input tensor content rectangle")
    if tensor_content[2] > tensor_w or tensor_content[3] > tensor_h:
        raise ValueError("depth input tensor content rectangle is outside its tensor")
    padded_fraction = _finite_number(
        analysis.get("padded_area_fraction"), "padded area fraction")
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
                tensor_content != (0, 0, tensor_w, tensor_h) or
                padded_fraction != 0.0 or analysis.get("fit_method") != "full-tensor" or
                analysis.get("crop_method") != "full-source" or
                analysis.get("scene_analysis_domain") != "full-source" or
                renderer.get("final_parallax_units") != "full-source-u" or
                scale != 1.0 or renderer.get("outside") is not None):
            raise ValueError("depth_input_region.json has inconsistent full-source semantics")
    else:
        if (semantic is None or authorization is None or generation == 0 or
                inference == (0, 0, width, height)):
            raise ValueError("depth_input_region.json has incomplete window-region authority")
        expected_plan = _plan_host_sbs_v2_window_region(
            semantic, width, height, tensor_w, tensor_h)
        if (expected_plan is None or expected_plan[0] != inference or
                expected_plan[1] != tensor_content):
            raise ValueError(
                "depth_input_region.json is not the deterministic authenticated integer contain fit")
        expected_padding = expected_plan[2]
        expected_scale, expected_vertical_slope = _roi_renderer_constants(
            inference, width, height, tensor_content)
        if (padded_fraction < 0.0 or padded_fraction >= 1.0 or
                padded_fraction != expected_padding or
                analysis.get("fit_method") !=
                "centered-integer-contain-with-edge-replicated-excluded-padding" or
                analysis.get("crop_method") != "direct retained-source rectangle sampling" or
                analysis.get("scene_analysis_domain") != "inference-rectangle-only" or
                renderer.get("final_parallax_units") != "roi-local-source-u" or
                scale != expected_scale):
            raise ValueError("depth_input_region.json has inconsistent window-region analysis")
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
        "tensor_content_rect": tensor_content,
        "padded_area_fraction": padded_fraction,
        "input_domain_reset": analysis["input_domain_reset"],
        "full_source_parallax_scale": scale,
        "horizontal_slope_source_u_per_source_u": (
            None if mode == "full-source" else horizontal_slope),
        "vertical_slope_source_u_per_source_v": (
            None if mode == "full-source" else vertical_slope),
    }


def validate_window_region_document(
        document: Any, *, matched_frame_id: int | None = None,
        source_width: int | None = None, source_height: int | None = None) -> Dict[str, Any]:
    """Validate optional matched-source window-region provenance."""

    root_keys = {
        "schema", "capture", "role", "authority_kind", "matched_frame_id",
        "coordinate_space", "identity", "freshness",
    }
    if not isinstance(document, dict) or set(document) != root_keys:
        raise ValueError("window_region.json has an unknown layout")
    if document.get("schema") != WINDOW_REGION_SCHEMA:
        raise ValueError("window_region.json has an unknown serialization schema")
    if document.get("capture") != (
            "same matched source/color/depth/render frame as the parent Dump 3D package"):
        raise ValueError("window_region.json has unknown capture semantics")
    if document.get("role") != (
            "matched-window region provenance; no independent geometry or renderer authority"):
        raise ValueError("window_region.json claims unknown authority")
    authority_kind = document.get("authority_kind")
    if authority_kind not in WINDOW_REGION_AUTHORITY_KINDS:
        raise ValueError("window_region.json has an unknown authority kind")

    frame_id = _uint64(document.get("matched_frame_id"), "matched frame id")
    if frame_id == 0 or (matched_frame_id is not None and frame_id != matched_frame_id):
        raise ValueError("window_region.json does not match the parent frame")

    coordinate = document.get("coordinate_space")
    if not isinstance(coordinate, dict) or set(coordinate) != {
            "name", "rect_semantics", "source_extent_px", "capture_rect_px"}:
        raise ValueError("window_region.json has an unknown coordinate layout")
    if (coordinate.get("name") != "matched-source-pixels" or
            coordinate.get("rect_semantics") != "half-open [left, top, right, bottom)"):
        raise ValueError("window_region.json has unknown rectangle semantics")
    extent = coordinate.get("source_extent_px")
    rect = coordinate.get("capture_rect_px")
    if (not isinstance(extent, dict) or set(extent) != {"width", "height"} or
            not isinstance(rect, dict) or set(rect) != {"left", "top", "right", "bottom"}):
        raise ValueError("window_region.json has malformed capture geometry")
    width = _uint32(extent.get("width"), "source width")
    height = _uint32(extent.get("height"), "source height")
    if (width == 0 or height == 0 or
            (source_width is not None and width != source_width) or
            (source_height is not None and height != source_height)):
        raise ValueError("window_region.json source extent does not match the dump")
    left = _int32(rect.get("left"), "rectangle left")
    top = _int32(rect.get("top"), "rectangle top")
    right = _int32(rect.get("right"), "rectangle right")
    bottom = _int32(rect.get("bottom"), "rectangle bottom")
    if left < 0 or top < 0 or right <= left or bottom <= top or right > width or bottom > height:
        raise ValueError("window_region.json rectangle is empty or out of bounds")

    identity = document.get("identity")
    if not isinstance(identity, dict) or set(identity) != {
            "hwnd", "process_id", "document_id", "video_id", "generation"}:
        raise ValueError("window_region.json has an unknown identity layout")
    hwnd, hwnd_value = _canonical_hwnd(identity.get("hwnd"), "window-region HWND")
    process_id = _uint32(identity.get("process_id"), "process id")
    document_id = _int32(identity.get("document_id"), "document id")
    video_id = _int32(identity.get("video_id"), "video id")
    generation = _uint64(identity.get("generation"), "generation")
    if hwnd_value <= 0 or process_id == 0 or generation == 0:
        raise ValueError("window_region.json has an incomplete identity")
    if authority_kind == "chromium-video" and (document_id == 0 or video_id == 0):
        raise ValueError("window_region.json has incomplete Chromium identity")
    if authority_kind == "foreground-client" and (document_id != 0 or video_id != 0):
        raise ValueError("window_region.json foreground authority carries DOM identity")

    freshness = document.get("freshness")
    if not isinstance(freshness, dict) or set(freshness) != {
            "latest_observation_age_ms_at_capture", "maximum_observation_age_ms",
            "geometry_continuity_ms_at_capture", "source_content_age_ms_at_capture",
            "fresh", "causal_geometry"}:
        raise ValueError("window_region.json has an unknown freshness layout")
    observation_age_ms = _uint32(
        freshness.get("latest_observation_age_ms_at_capture"), "latest observation age")
    maximum_observation_age_ms = _uint32(
        freshness.get("maximum_observation_age_ms"), "maximum observation age")
    geometry_continuity_ms = _uint64(
        freshness.get("geometry_continuity_ms_at_capture"), "geometry continuity")
    source_content_age_ms = _uint64(
        freshness.get("source_content_age_ms_at_capture"), "source content age")
    if (freshness.get("fresh") is not True or maximum_observation_age_ms == 0 or
            observation_age_ms > maximum_observation_age_ms):
        raise ValueError("window_region.json is stale")
    if (freshness.get("causal_geometry") is not True or
            geometry_continuity_ms < source_content_age_ms):
        raise ValueError("window_region.json geometry postdates the source content")
    return {
        "authority_kind": authority_kind,
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
        "latest_observation_age_ms_at_capture": observation_age_ms,
        "maximum_observation_age_ms": maximum_observation_age_ms,
        "geometry_continuity_ms_at_capture": geometry_continuity_ms,
        "source_content_age_ms_at_capture": source_content_age_ms,
    }


def camera_center_integrity_bits(
        center: float, inverse_scale: float, convergence_curve: float, revision: int,
        joint_plane_mode: int = 3) -> int:
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
    checksum = ((checksum ^ joint_plane_mode) * 16777619) & 0xFFFFFFFF
    return checksum


def camera_center_integrity_for_state_values(values: Dict[str, Any]) -> int:
    """Seal the complete generated tuple; mode 3 includes all sixteen controller-tail words."""
    checksum = camera_center_integrity_bits(
        values["center"], values["inverse_scale"], values["convergence_curve"],
        values["calibration_revision"], values["joint_plane_mode_bits"])
    for descriptor in _STATE_FIELDS[12:]:
        value = values[descriptor["name"]]
        bits = (struct.unpack("<I", struct.pack("<f", value))[0]
                if descriptor["gpu_encoding"] == "float" else int(value))
        checksum = ((checksum ^ bits) * 16777619) & 0xFFFFFFFF
    return checksum


def _validate_adaptive_controller_tail(values: Dict[str, Any]) -> None:
    mode = values["joint_plane_mode_bits"]

    def clock(name: str) -> int:
        return _uint64_words(values[name + "_low"], values[name + "_high"])

    last = clock("gain_last_observation")
    first = clock("gain_seed_first")
    seed_last = clock("gain_seed_last")
    count = values["gain_seed_count"]
    armed = values["gain_clock_armed"]
    cap = values["gain_display_limit"]
    samples = 1
    limit = _DEFAULTS.direct_container_limit
    if (armed > 1 or count > samples or values["gain_reserved0"] != 0 or
            values["gain_reserved1"] != 0 or (armed != 0 and last == 0) or
            not 0.0 <= cap <= _float32(limit)):
        raise ValueError("adaptive controller tail has invalid clock, count, reserved words or display budget")
    target_inverse = values["gain_target_inverse_scale"]
    target_nearest = values["gain_target_nearest"]
    no_target = (values["gain_target_zero"] == 0.0 and
                 target_inverse == 0.0 and target_nearest == 0.0)
    try:
        reciprocal_product = _float32(target_inverse * target_nearest)
    except ValueError:
        reciprocal_product = math.inf
    target_valid = (target_inverse > 0.0 and target_nearest > 0.0 and
                    abs(reciprocal_product - 1.0) <= 2.0e-6)
    if (not no_target and not target_valid) or (armed != 0 and not target_valid):
        raise ValueError("adaptive controller tail has an invalid nearest reference or reciprocal")
    if count == 0:
        if (armed != 0 or first != 0 or seed_last != 0 or values["gain_seed_mean_nearest"] != 0.0 or
                values["gain_seed_mean_zero"] != 0.0):
            raise ValueError("adaptive controller tail has nonzero empty seed state")
    elif (first == 0 or seed_last < first or last < seed_last or
          values["gain_seed_mean_nearest"] <= 0.0 or cap <= 0.0 or
          seed_last != first):
        raise ValueError("adaptive controller tail has invalid source-time seed state")


def validate_parallax_state_words(
        words: Any, *, raw_coordinate_scale: float,
        expected_joint_plane_mode: int | None = 3,
        requested_gain: float | None = None) -> Dict[str, Any]:
    """Authenticate the generated 28-word GPU tuple, including the adaptive controller tail.

    ``words`` may be the exact little-endian byte record or a uint32 sequence. Source clocks
    are microseconds; held tuples intentionally need not equal a later request timestamp.
    ``requested_gain`` is retained for caller compatibility; the mode 3 representation
    budget is independent of stream strength.
    """
    if isinstance(words, (bytes, bytearray, memoryview)):
        raw = _uint32_words(words, len(_STATE_FIELDS), "ParallaxState")
    elif isinstance(words, (list, tuple)) and len(words) == len(_STATE_FIELDS):
        raw = tuple(_uint32(value, f"ParallaxState word {index}")
                    for index, value in enumerate(words))
    else:
        raise ValueError("ParallaxState must contain the exact generated uint32 layout")
    scale = _finite_float32(raw_coordinate_scale, "ParallaxState raw coordinate scale")
    if scale <= 0.0:
        raise ValueError("ParallaxState raw coordinate scale must be positive")
    if (expected_joint_plane_mode is not None and
            (isinstance(expected_joint_plane_mode, bool) or
             not isinstance(expected_joint_plane_mode, int) or
             expected_joint_plane_mode != 3)):
        raise ValueError("ParallaxState expected joint plane mode must be adaptive policy identity 3")
    values = {
        descriptor["name"]: (
            _finite_float32(_float32_bits(word), f"ParallaxState {descriptor['name']}")
            if descriptor["gpu_encoding"] == "float" else word)
        for descriptor, word in zip(_STATE_FIELDS, raw)
    }
    mode = values["joint_plane_mode_bits"]
    if (mode != 3 or
            (expected_joint_plane_mode is not None and mode != expected_joint_plane_mode) or
            values["mapping_state_reserved_2"] != 0):
        raise ValueError("ParallaxState has invalid mode or reserved mapping state")
    if values["contract_tag_bits"] != _CONTRACT_TAG:
        raise ValueError("ParallaxState has the wrong contract tag")
    if values["camera_center_integrity_bits"] != camera_center_integrity_for_state_values(values):
        raise ValueError("ParallaxState camera center integrity checksum disagrees")
    _validate_adaptive_controller_tail(values)
    valid = values["frame_valid"]
    if (valid not in (0.0, 1.0) or values["container_scale"] != 1.0 or
            values["convergence_curve"] != _float32(_DEFAULTS.convergence_curve_default) or
            values["renderer_authorization_bits"] != (_CONTRACT_TAG if valid == 1.0 else 0)):
        raise ValueError("ParallaxState violates frame/camera validity semantics")
    revision = _calibration_revision(values["calibration_revision"], "ParallaxState revision")
    inverse = values["inverse_scale"]
    camera_initialized = inverse > 0.0 and revision > 0
    camera_empty = values["center"] == 0.0 and inverse == 0.0
    samples = 1
    if ((valid == 1.0 and values["gain_clock_armed"] != 1) or
            (camera_initialized and values["gain_seed_count"] != samples) or
            (camera_empty and values["gain_seed_count"] >= samples) or
            (valid == 1.0 and not camera_initialized) or
            (valid == 0.0 and not (camera_initialized or camera_empty))):
        raise ValueError("ParallaxState violates frame/camera validity semantics")
    if mode == 3 and (
            (camera_initialized and inverse > _float32(1.0 / scale) + 2.0e-6) or
            (values["gain_target_nearest"] != 0.0 and values["gain_target_nearest"] < scale) or
            (values["gain_seed_mean_nearest"] != 0.0 and values["gain_seed_mean_nearest"] < scale)):
        raise ValueError("ParallaxState Host amplitude reference is below the model prior")
    budget = _display_budget_for_mode(requested_gain, mode)
    stored = values["gain_display_limit"]
    if (stored != 0.0 or valid == 1.0) and stored != budget:
        raise ValueError("ParallaxState adaptive representation budget disagrees with the contract")
    return values


def _calibration_revision(value: Any, label: str) -> int:
    revision = _uint32(value, label)
    if revision == _RESERVED_CALIBRATION_REVISION:
        raise ValueError(f"{label} uses the reserved calibration revision")
    return revision


def _within_absolute_tolerance(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return abs(float(left) - float(right)) <= tolerance


def _require_coordinate_binding(value: Any, count_key: str, count: int) -> int:
    shader = coordinate_contract.SHADER_IMPLEMENTATION
    current = {
        "schema": coordinate_contract.CONTRACT_SCHEMA,
        "tag": _CONTRACT_TAG,
        "source_closure_schema": shader.source_closure_schema,
        "source_compile_flags": shader.source_compile_flags,
        "source_macro_count": shader.source_macro_count,
        "source_closure_sha256": shader.source_closure_sha256,
        count_key: count,
    }
    if value != current:
        raise ValueError("dump has an unknown depth-coordinate-v2 contract binding")
    return _CONTRACT_TAG


def validate_shadow_state_document(document: Any) -> Dict[str, Any]:
    """Validate one parsed ``shadow_state.json`` and return its typed named values."""

    if not isinstance(document, dict) or set(document) != _STATE_ROOT_KEYS:
        raise ValueError("shadow_state.json has missing or unknown root fields")
    if document.get("schema") != SHADOW_STATE_DUMP_SCHEMA:
        raise ValueError("shadow_state.json has an unknown serialization schema")
    coordinate_tag = _require_coordinate_binding(
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
    mode_view = document.get("decoded")
    decoded_mode = mode_view.get("joint_plane_mode") if isinstance(mode_view, dict) else None
    if (isinstance(decoded_mode, bool) or not isinstance(decoded_mode, int) or
            decoded_mode != 3):
        raise ValueError("shadow_state.json has an invalid joint plane mode")
    gain_unit = "coordinate unit"
    accepted_units = (
        {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": f"one-eye source-U per {gain_unit}",
            "parallax": "signed one-eye source-U",
        },
        {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": f"one-eye full-source-U per {gain_unit}",
            "parallax": "signed one-eye full-source-U",
        },
        {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": f"one-eye ROI-local source-U per {gain_unit}",
            "parallax": (
                "signed one-eye ROI-local source-U; full-source renderer authority "
                "additionally requires depth_input_region embedding"),
        },
    )
    if document.get("units") not in accepted_units:
        raise ValueError("shadow_state.json has unknown units")
    if document.get("adaptation_semantics") != adaptation_semantics_for_mode(decoded_mode):
        raise ValueError("shadow_state.json has unknown adaptation semantics")

    constants = document.get("constants")
    if not isinstance(constants, dict) or set(constants) != _STATE_CONSTANT_KEYS:
        raise ValueError("shadow_state.json has an invalid constants object")
    native_constants = {
        key: shadow_state_constant_float32(document, key)
        for key in _STATE_CONSTANT_KEYS
    }
    expected_defaults = {
        "collapse_abs_epsilon": _DEFAULTS.collapse_abs_epsilon,
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
    if any(_float32(constants.get(key)) != _float32(value)
           for key, value in expected_defaults.items()):
        raise ValueError("shadow_state.json constants disagree with the generated contract")
    if (native_constants["raw_coordinate_scale"] <= 0.0 or
            native_constants["requested_gain"] <= 0.0 or
            native_constants["requested_pop_strength"] <= 0.0):
        raise ValueError("shadow_state.json has invalid runtime constants")
    native_requested_gain = _float32(
        native_constants["requested_pop_strength"] *
        _float32(_DEFAULTS.gain_per_pop)
    )
    if not _within_absolute_tolerance(
            native_constants["requested_gain"],
            native_requested_gain,
            _REQUESTED_GAIN_AUTHENTICATION_TOLERANCE):
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
            serialized_value = _finite_number(
                value, f"shadow_state.json field {index}")
            value = _finite_float32(
                serialized_value, f"shadow_state.json field {index}")
            serialized_named_value = _finite_number(
                named.get(descriptor["name"]),
                f"shadow_state.json named_values.{descriptor['name']}",
            )
            _finite_float32(
                serialized_named_value,
                f"shadow_state.json named_values.{descriptor['name']}",
            )
            # These two JSON fields are redundant serializations of one native word.
            # Compare their serialized values exactly before using the float32 word.
            named_matches = _same_serialized_number(
                serialized_named_value, serialized_value)
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
    if typed["contract_tag_bits"] != coordinate_tag:
        raise ValueError("shadow_state.json state words have the wrong contract tag")
    expected_camera_integrity = camera_center_integrity_for_state_values(typed)
    if typed["camera_center_integrity_bits"] != expected_camera_integrity:
        raise ValueError("shadow_state.json camera center integrity checksum disagrees")
    expected_authorization = coordinate_tag if typed["frame_valid"] > 0.5 else 0
    if typed["renderer_authorization_bits"] != expected_authorization:
        raise ValueError("shadow_state.json renderer authorization disagrees with frame validity")
    if (typed["joint_plane_mode_bits"] != 3 or
            typed["joint_plane_mode_bits"] != decoded_mode or
            typed["mapping_state_reserved_2"] != 0):
        raise ValueError("shadow_state.json has nonzero reserved mapping state")
    _validate_adaptive_controller_tail(typed)
    if decoded_mode == 3 and (
            (typed["inverse_scale"] > 0.0 and
             typed["inverse_scale"] > _float32(1.0 / native_constants["raw_coordinate_scale"]) + 2.0e-6) or
            (typed["gain_target_nearest"] != 0.0 and
             typed["gain_target_nearest"] < native_constants["raw_coordinate_scale"]) or
            (typed["gain_seed_mean_nearest"] != 0.0 and
             typed["gain_seed_mean_nearest"] < native_constants["raw_coordinate_scale"])):
        raise ValueError("shadow_state.json Host amplitude reference is below the model prior")
    budget = _display_budget_for_mode(native_constants["requested_gain"], decoded_mode)
    stored_budget = typed["gain_display_limit"]
    if ((stored_budget != 0.0 or typed["frame_valid"] == 1.0) and
            stored_budget != budget):
        raise ValueError("shadow_state.json adaptive representation budget disagrees with stream strength")

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
    native_decoded = {
        key: _finite_float32(
            decoded.get(key), f"shadow_state.json decoded.{key}")
        for key in ("requested_gain", "requested_pop_strength", "latched_scale",
                    "convergence_curve", "container_scale", "effective_gain")
    }
    camera_valid = (
        typed["inverse_scale"] > 0.0 and typed["calibration_revision"] > 0)
    native_latched_scale = (
        _float32(1.0 / _float32(typed["inverse_scale"]))
        if camera_valid else 0.0)
    expected_decoded = {
        "calibration_revision": typed["calibration_revision"],
        "confirmed_cut_count": typed["confirmed_cut_count"],
        "contract_tag": coordinate_tag,
        "requested_gain": constants["requested_gain"],
        "requested_pop_strength": constants["requested_pop_strength"],
        "camera_valid": camera_valid,
        "latched_scale": native_latched_scale,
        "convergence_curve": typed["convergence_curve"],
        "container_scale": typed["container_scale"],
        "effective_gain": constants["requested_gain"] if decoded["frame_valid"] else 0.0,
        "camera_center_integrity_bits": expected_camera_integrity,
        "renderer_authorization_bits": expected_authorization,
        "joint_plane_mode": typed["joint_plane_mode_bits"],
    }
    # These are redundant decoded views of authenticated words/constants, not
    # independent measurements.  Require their canonical values exactly so a
    # pair of permissive comparisons cannot bridge an invalid state word to a
    # plausible decoded value.
    if any(
        (not _same_serialized_number(decoded.get(key), value)
         if key in _DECODED_FLOAT_KEYS else decoded.get(key) != value)
        for key, value in expected_decoded.items()
    ):
        raise ValueError("shadow_state.json decoded values disagree with the state words")
    convergence_valid = (
        typed["convergence_curve"] == _DEFAULTS.convergence_curve_default)
    if (typed["container_scale"] != 1.0 or
            not convergence_valid or
            native_decoded["effective_gain"] < 0.0 or
            native_decoded["effective_gain"] >
            native_decoded["requested_gain"] + 1.0e-7 or
            (decoded["frame_valid"] and not decoded["camera_valid"]) or
            ((
                (camera_valid and typed["gain_seed_count"] != 1) or
                (not camera_valid and typed["gain_seed_count"] >= 1) or
                (decoded["frame_valid"] and typed["gain_clock_armed"] != 1))) or
            (not decoded["camera_valid"] and (
                typed["center"] != 0.0 or typed["inverse_scale"] != 0.0 or
                typed["convergence_curve"] !=
                _DEFAULTS.convergence_curve_default))):
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
    values = {name: _finite_float32(named[name], f"shadow_frame_stats.json {name}")
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
    if values["percentile_valid"] not in (0.0, 1.0):
        raise ValueError("shadow_frame_stats.json percentile validity must be zero or one")
    if values["percentile_valid"] == 1.0:
        if (values["valid"] != 1.0 or values["percentile_bin_width"] < 0.0 or
                not values["minimum"] <= values["percentile_low"] <=
                values["percentile_high"] <= values["maximum"]):
            raise ValueError("shadow_frame_stats.json quantile statistics are inconsistent")
    elif any(values[key] != 0.0 for key in
             ("percentile_low", "percentile_high", "percentile_bin_width")):
        raise ValueError("shadow_frame_stats.json unavailable quantiles must be canonical zero")
    return values


def _validate_shadow_manifest_summary(value: Any) -> Dict[str, Any]:
    expected_keys = _DECODED_KEYS | {
        "raw_coordinate_scale", "rendered_output_selected",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("dump_manifest.json has an invalid V2 state summary")
    if (value.get("frame_valid") is not True or
            value.get("camera_valid") is not True or
            value.get("rendered_output_selected") is not True):
        raise ValueError("dump_manifest.json has an invalid V2 state summary")
    _calibration_revision(
        value.get("calibration_revision"),
        "dump_manifest.json V2 state calibration_revision")
    for key in ("confirmed_cut_count", "contract_tag", "camera_center_integrity_bits",
                "renderer_authorization_bits", "joint_plane_mode"):
        _uint32(value.get(key), f"dump_manifest.json V2 state {key}")
    native_value = {
        key: _finite_float32(
            value.get(key), f"dump_manifest.json V2 state {key}")
        for key in ("requested_gain", "requested_pop_strength", "latched_scale",
                    "convergence_curve", "container_scale", "effective_gain",
                    "raw_coordinate_scale")
    }
    native_requested_gain = _float32(
        native_value["requested_pop_strength"] *
        _float32(_DEFAULTS.gain_per_pop)
    )
    if (value["contract_tag"] != _CONTRACT_TAG or
            value["renderer_authorization_bits"] != _CONTRACT_TAG or
            value["calibration_revision"] == 0 or
            native_value["requested_gain"] <= 0.0 or
            native_value["requested_pop_strength"] <= 0.0 or
            native_value["effective_gain"] <= 0.0 or
            native_value["raw_coordinate_scale"] <= 0.0 or
            not _within_absolute_tolerance(
                native_value["requested_gain"], native_requested_gain,
                _REQUESTED_GAIN_AUTHENTICATION_TOLERANCE) or
            not _within_absolute_tolerance(
                native_value["effective_gain"],
                native_value["requested_gain"],
                _REQUESTED_GAIN_AUTHENTICATION_TOLERANCE) or
            value["joint_plane_mode"] != 3 or
            native_value["latched_scale"] <= 0.0 or
            native_value["convergence_curve"] !=
            _float32(_DEFAULTS.convergence_curve_default) or
            native_value["container_scale"] != 1.0):
        raise ValueError("dump_manifest.json has an invalid V2 state summary")
    return value


def _required_hashed_artifact(
        artifacts: Dict[str, Any], name: str, label: str, *,
        byte_count: int | None = None) -> Dict[str, Any]:
    descriptor = artifacts.get(name)
    keys = {"available", "required", "stage", "description", "sha256"}
    if byte_count is not None:
        keys.add("byte_count")
    if (not isinstance(descriptor, dict) or set(descriptor) != keys or
            descriptor.get("available") is not True or
            descriptor.get("required") is not True or
            not isinstance(descriptor.get("stage"), str) or not descriptor["stage"] or
            not isinstance(descriptor.get("description"), str) or
            not descriptor["description"] or
            not _is_sha256_hex(descriptor.get("sha256"))):
        raise ValueError(f"dump_manifest.json has an invalid {label} artifact descriptor")
    if byte_count is not None and (type(descriptor.get("byte_count")) is not int or
                                  descriptor["byte_count"] != byte_count):
        raise ValueError(f"dump_manifest.json has an invalid {label} byte_count")
    return descriptor


def _subtitle_contour_artifact_layouts(
        source_width: int, source_height: int, field_width: int, field_height: int
        ) -> Dict[str, tuple[Dict[str, Any], int]]:
    """Return the eleven exact native tuple shapes, formats and byte lengths."""
    policy = coordinate_contract.EXPECTED_SUBTITLE_OCR["contour_policy"]
    if any(type(value) is not int or not 0 < value <= 0xFFFFFFFF
           for value in (source_width, source_height)) or any(
               type(value) is not int or not 0 < value <= _MAXIMUM_SOURCE_LONG_SIDE
               for value in (field_width, field_height)):
        raise ValueError("subtitle contour source/field extent is unsupported")
    crop_height = min(source_height, (source_width + 5) // 6)
    eligible = max(source_width, source_height) <= policy["source_max_extent"]
    native_width, native_height = ((source_width, crop_height) if eligible else (1, 1))
    native_count = native_width * native_height
    field_count = field_width * field_height

    def texture(width: int, height: int, fmt: str, value: int) -> Dict[str, Any]:
        return {"width": width, "height": height, "format": "DXGI_FORMAT_" + fmt,
                "format_value": value}

    return {
        "contour_support": (texture(native_width, native_height, "R32_UINT", 42), native_count * 4),
        "contour_field_pin": (texture(field_width, field_height, "R32_UINT", 42), field_count * 4),
        "contour_distance": (texture(field_width, field_height, "R32_FLOAT", 41), field_count * 4),
        "contour_identity": ({"word_count": policy["history_word_count"],
                              "format": "uint32-le structured buffer"}, policy["history_word_count"] * 4),
        "contour_history": (texture(native_width, native_height, "R32G32B32A32_FLOAT", 2), native_count * 16),
        "contour_source": (texture(native_width, native_height, "R16G16B16A16_FLOAT", 10), native_count * 8),
        "ocr_probability": ({"width": coordinate_contract.SUBTITLE_OCR.output_shape[3],
                             "height": coordinate_contract.SUBTITLE_OCR.output_shape[2],
                             "format": "float32-le structured buffer"},
                            math.prod(coordinate_contract.SUBTITLE_OCR.output_shape) * 4),
        "gaussian_weight": (texture(field_width, field_height, "R32_FLOAT", 41), field_count * 4),
        "gaussian_blend": (texture(field_width, field_height, "R32_FLOAT", 41), field_count * 4),
        "gaussian_vertical_conditioned": (texture(field_width, field_height, "R32_FLOAT", 41), field_count * 4),
        "gaussian_control": ({"word_count": 1, "format": "uint32-le structured buffer"}, 4),
    }


def _subtitle_shader_contract(
        specs: tuple[tuple[str, str, str], ...]) -> Dict[str, Any]:
    shader = coordinate_contract.SHADER_IMPLEMENTATION
    return {
        "depth_coordinate_v2_schema": coordinate_contract.CONTRACT_SCHEMA,
        "depth_coordinate_v2_tag": _CONTRACT_TAG,
        "source_closure_schema": shader.source_closure_schema,
        "source_compile_flags": shader.source_compile_flags,
        "source_macro_count": shader.source_macro_count,
        "source_closure_sha256": shader.source_closure_sha256,
        "source_specs": [
            {
                "source_file": source_file,
                "entrypoint": entrypoint,
                "target": target,
            }
            for source_file, entrypoint, target in specs
        ],
    }


def _subtitle_ocr_producer_contract() -> Dict[str, Any]:
    return {
        "contract_schema": _SUBTITLE_OCR_CONTRACT_SCHEMA,
        "record_schema": SUBTITLE_OCR_RECORD_SCHEMA,
        "record_tag": SUBTITLE_OCR_RECORD_TAG,
        "record_word_count": SUBTITLE_OCR_RECORD_WORD_COUNT,
        "raw_box_capacity": SUBTITLE_OCR_RAW_BOX_CAPACITY,
        "final_box_capacity": SUBTITLE_OCR_FINAL_BOX_CAPACITY,
        "model": {
            "name": _SUBTITLE_OCR_MODEL_NAME,
            "asset_path": _SUBTITLE_OCR_ASSET_PATH,
            "artifact_onnx_sha256": _SUBTITLE_OCR_ARTIFACT_ONNX_SHA256,
            "source_url": _SUBTITLE_OCR_SOURCE_URL,
            "source_onnx_sha256": _SUBTITLE_OCR_SOURCE_ONNX_SHA256,
            "conversion_tool": _SUBTITLE_OCR_CONVERSION_TOOL,
            "conversion_version": _SUBTITLE_OCR_CONVERSION_VERSION,
            "conversion_recipe": _SUBTITLE_OCR_CONVERSION_RECIPE,
            "conversion_calibration_profile": (
                _SUBTITLE_OCR_CONVERSION_CALIBRATION_PROFILE),
            "engine_recipe": _SUBTITLE_OCR_ENGINE_RECIPE,
            "preprocess_profile": coordinate_contract.SUBTITLE_OCR.preprocess_profile,
            "source_crop": coordinate_contract.SUBTITLE_OCR.source_crop,
            "input": {
                "name": coordinate_contract.SUBTITLE_OCR.input_name,
                "dtype": coordinate_contract.SUBTITLE_OCR.input_dtype,
                "layout": coordinate_contract.SUBTITLE_OCR.input_layout,
                "shape": list(coordinate_contract.SUBTITLE_OCR.input_shape),
                "channels": list(coordinate_contract.SUBTITLE_OCR.input_channels),
                "imagenet_mean": list(coordinate_contract.SUBTITLE_OCR.imagenet_mean),
                "imagenet_std": list(coordinate_contract.SUBTITLE_OCR.imagenet_std),
            },
            "output": {
                "name": coordinate_contract.SUBTITLE_OCR.output_name,
                "dtype": coordinate_contract.SUBTITLE_OCR.output_dtype,
                "layout": coordinate_contract.SUBTITLE_OCR.output_layout,
                "shape": list(coordinate_contract.SUBTITLE_OCR.output_shape),
            },
        },
        "shader_contract": _subtitle_shader_contract(_SUBTITLE_OCR_SHADER_SPECS),
    }


def _subtitle_plane_resolver_contract() -> Dict[str, Any]:
    return {
        "state_schema": SUBTITLE_PLANE_STATE_SCHEMA,
        "state_tag": SUBTITLE_PLANE_STATE_TAG,
        "state_word_count": SUBTITLE_PLANE_STATE_WORD_COUNT,
        "cover_capacity": SUBTITLE_PLANE_COVER_CAPACITY,
        "contour_policy": dict(coordinate_contract.EXPECTED_SUBTITLE_OCR["contour_policy"]),
        "adaptive_ui": {
            "policy_id": coordinate_contract.CALIBRATED_DEFAULTS.adaptive_policy_id,
            "state_offset": SUBTITLE_PLANE_ADAPTIVE_OFFSET, "state_word_count": 16,
            "clock": "source-observation-microseconds",
            "geometry_authority": coordinate_contract.SUBTITLE_OCR.geometry_authority,
            "probe": "union-of-current-authorized-covers-in-content",
            "clearance_per_eye_source_pixels": 2,
            "enter_covered_fraction": _float32(0.20),
            "enter_content_fraction": _float32(0.02),
            "release_covered_fraction": _float32(0.15),
            "release_content_fraction": _float32(0.015),
            "approach_delay_ms": 100, "retreat_delay_ms": 1500,
            "approach_source_u_per_second": _float32(0.03),
            "retreat_source_u_per_second": _float32(0.005),
            "geometric_pin": "full-on-current-contour-authority",
            "host_model_policy": {
                "mode": 3,
                "maximum_plane_source_u": _float32(_DEFAULTS.direct_container_limit),
                "candidate_step_per_eye_source_pixels": 4,
                "last_candidate": "exact-representation-cap",
                "maximum_candidate_index": "ceil-analysis-source-width-over-100",
                "core_range_per_eye_source_pixels": 0,
                "retreat_requires_applied_target": True,
                "capped_conflict": "derived-from-top-candidate-counts",
            },
        },
        "current_cover_policy": {
            "selection": "all-valid-current-ocr8-core-cover-pairs",
            "capacity": SUBTITLE_PLANE_COVER_CAPACITY,
            "geometry_source": "exact-current-ocr8-record-final-payload",
            "history_authority": "per-pixel-four-genuine-rgb-confirmations",
            "box_identity_resets_history": False,
            "held_reuse": "byte-identical-latest-published-ocr8-and-uip2-tuple",
        },
        "target_policy": {
            "units": "one-eye-analysis-source-U",
            "selection": "authenticated-current-cover-conflict-histogram",
            "candidate_step_per_eye_source_pixels": 4,
            "clearance_per_eye_source_pixels": 2,
            "approach_dwell_us": 100000,
            "retreat_dwell_us": 1500000,
            "approach_rate_source_u_per_second": _float32(0.03),
            "retreat_rate_source_u_per_second": _float32(0.005),
            "rate_tick_cap_us": 250000,
            "current_cover_pin": "confirmed-native-temporal-support-literal-gaussian-weight",
            "collar": "untruncated-max-union-gaussian-before-shared-slope-conditioning",
            "representation_limit": "direct-parallax-container",
        },
        "shader_contract": _subtitle_shader_contract(_SUBTITLE_PLANE_SHADER_SPECS),
    }


def _validate_subtitle_conditioning_manifest(
        document: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    if "overlay_conditioning" in document:
        raise ValueError("retired overlay-conditioning authority is not supported")
    _required_hashed_artifact(
        artifacts, "subtitle_conditioning.json", "subtitle conditioning metadata")
    subtitle = document.get("subtitle_conditioning")
    expected_keys = {"mode", "request", "producer", "resolver", "artifacts"}
    if not isinstance(subtitle, dict) or set(subtitle) != expected_keys:
        raise ValueError("dump_manifest.json has an invalid subtitle-conditioning contract")

    mode = subtitle.get("mode")
    if mode == _SUBTITLE_MODE_NONE:
        if subtitle != {
                "mode": _SUBTITLE_MODE_NONE,
                "request": None,
                "producer": None,
                "resolver": None,
                "artifacts": {}}:
            raise ValueError("inactive subtitle conditioning is not canonical")
        return {
            "mode": _SUBTITLE_MODE_NONE,
            "live": False,
            "request": None,
            "artifact_files": {},
            "subtitle_evidence_complete": False,
        }
    if mode == _SUBTITLE_MODE_CONTOURS:
        expected_files = {
            "ocr_record": "subtitle_ocr_record.u32",
            "locator_state": "subtitle_locator_state.u32",
            "base_field": "shadow_base_final_parallax.f32",
            "conditioned_field": "shadow_final_parallax.f32",
            **_SUBTITLE_CONTOUR_ARTIFACT_FILES,
        }
        if subtitle.get("request") is not True:
            raise ValueError("active UIP4 subtitle conditioning must bind an enabled request")
        if subtitle.get("producer") != _subtitle_ocr_producer_contract():
            raise ValueError("active UIP4 subtitle conditioning has unknown OCR8 provenance")
        if subtitle.get("resolver") != _subtitle_plane_resolver_contract():
            raise ValueError("active UIP4 subtitle conditioning has unknown resolver provenance")
        if subtitle.get("artifacts") != expected_files:
            raise ValueError("active UIP4 subtitle conditioning has unknown artifact roles")
        required = {
            "subtitle_ocr_record.u32": "OCR8 subtitle boxes for atomic target",
            "subtitle_locator_state.u32": "compact UIP4 current subtitle plane state",
            "shadow_base_final_parallax.f32": (
                "ordinary post-limiter V2 field before UIP4 conditioning"),
        }
        for name, stage in required.items():
            descriptor = _required_hashed_artifact(
                artifacts, name, f"active subtitle {name}")
            if descriptor.get("stage") != stage:
                raise ValueError(
                    f"dump_manifest.json misattributes active subtitle artifact {name}")
        dimensions = document.get("dimensions", {})
        source_shape = dimensions.get("analysis_source", {})
        field_shape = dimensions.get("shadow_final_parallax", {})
        layouts = _subtitle_contour_artifact_layouts(
            source_shape.get("width"), source_shape.get("height"),
            field_shape.get("width"), field_shape.get("height"))
        for role, (layout, byte_count) in layouts.items():
            name = _SUBTITLE_CONTOUR_ARTIFACT_FILES[role]
            descriptor = _required_hashed_artifact(
                artifacts, name, f"active subtitle {name}", byte_count=byte_count)
            if descriptor["stage"] != _SUBTITLE_CONTOUR_ARTIFACT_STAGES[role]:
                raise ValueError(f"dump_manifest.json misattributes contour artifact {name}")
            if dimensions.get(role) != layout:
                raise ValueError(f"dump_manifest.json contour tuple dimensions disagree for {role}")
        return {
            "mode": _SUBTITLE_MODE_CONTOURS,
            "live": True,
            "request": True,
            "artifact_files": expected_files,
            "subtitle_evidence_complete": True,
        }
    raise ValueError("unsupported subtitle-conditioning authority")


_GPU_TRACE_ARTIFACTS = {
    "gpu_trace_ring.u32": "diagnostic GPU accepted-root completion history",
    "gpu_trace.json": "decoded diagnostic GPU history",
    "gpu_trace_contract.json": "diagnostic GPU trace wire contract",
}


def _validate_gpu_trace_manifest(
        document: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    summary = document.get("gpu_trace")
    if not isinstance(summary, dict) or not isinstance(summary.get("available"), bool):
        raise ValueError("dump_manifest.json has an invalid GPU trace summary")
    available = summary["available"]
    for name, stage in _GPU_TRACE_ARTIFACTS.items():
        descriptor = artifacts.get(name)
        expected_keys = {"available", "required", "stage", "description"}
        if available:
            expected_keys.add("sha256")
        if (not isinstance(descriptor, dict) or set(descriptor) != expected_keys or
                descriptor.get("available") is not available or
                descriptor.get("required") is not False or
                descriptor.get("stage") != stage or
                not isinstance(descriptor.get("description"), str) or
                not descriptor["description"] or
                (available and not _is_sha256_hex(descriptor.get("sha256")))):
            raise ValueError(
                f"dump_manifest.json has an invalid optional GPU trace artifact {name}")
    if not available:
        expected = {
            "available": False,
            "required": False,
            "rendering_authority": False,
            "raw_artifact": None,
            "decoded_artifact": None,
            "contract_artifact": None,
        }
        if summary != expected:
            raise ValueError("unavailable GPU trace summary is not canonical")
        return {"available": False}

    expected_keys = {
        "available", "required", "rendering_authority", "raw_artifact",
        "decoded_artifact", "contract_artifact", "record_count",
        "oldest_sequence", "next_sequence", "matched_sequence",
        "source_closure_sha256",
    }
    if (set(summary) != expected_keys or summary.get("required") is not False or
            summary.get("rendering_authority") is not False or
            summary.get("raw_artifact") != "gpu_trace_ring.u32" or
            summary.get("decoded_artifact") != "gpu_trace.json" or
            summary.get("contract_artifact") != "gpu_trace_contract.json" or
            summary.get("source_closure_sha256") !=
            GPU_TRACE_SOURCE_CLOSURE_SHA256):
        raise ValueError("available GPU trace summary is not canonical")
    count = _uint32(summary.get("record_count"), "GPU trace record count")
    oldest = _uint64(summary.get("oldest_sequence"), "GPU trace oldest sequence")
    next_sequence = _uint64(summary.get("next_sequence"), "GPU trace next sequence")
    matched = _uint64(summary.get("matched_sequence"), "GPU trace matched sequence")
    if (count == 0 or count > GPU_TRACE_CAPACITY or oldest == 0 or
            next_sequence != oldest + count or not oldest <= matched < next_sequence):
        raise ValueError("available GPU trace summary has impossible chronology")
    return {
        "available": True,
        "record_count": count,
        "oldest_sequence": oldest,
        "next_sequence": next_sequence,
        "matched_sequence": matched,
    }


def _validate_final_parallax_manifest(document: Dict[str, Any]) -> Dict[str, Any]:
    final = document.get("final_parallax")
    contract = coordinate_contract.FINAL_PARALLAX
    expected = {
        "contract_schema": contract.schema,
        "artifact": "shadow_final_parallax.f32",
        "warp_input_artifact": "shadow_final_parallax.f32",
        "authority": contract.authority,
        "publication_policy": contract.publication_policy,
        "reuse_policy": contract.reuse_policy,
        "invalid_policy": contract.invalid_policy,
        "current_rgb_policy": contract.current_rgb_policy,
        "warp_relation": "same authenticated artifact",
    }
    if final != expected:
        raise ValueError("dump_manifest.json has an invalid final-parallax contract")
    return expected


def _validate_warp_map_manifest(
        document: Dict[str, Any], artifacts: Dict[str, Any],
        dimensions: Dict[str, Any], input_mode: str,
        joint_plane_mode: int = 3) -> Dict[str, Any]:
    """Validate the manifest-resident replacement for ``warp_map_shape.json``."""
    if type(joint_plane_mode) is not int or joint_plane_mode != 3:
        raise ValueError("dump_manifest.json has an invalid joint plane mode")
    contract = document.get("warp_map_contract")
    map_descriptor = artifacts.get("warp_map.f32")
    mask_descriptor = artifacts.get("warp_mask.png")
    required = input_mode == "window-region"
    expected_schema = 2
    if not isinstance(contract, dict) or contract.get("schema") != expected_schema:
        raise ValueError("dump_manifest.json has an invalid warp-map contract")
    available = contract.get("available")
    if not isinstance(available, bool):
        raise ValueError("dump_manifest.json has an invalid warp-map availability flag")
    if not available:
        if (contract != {"available": False, "schema": 2, "artifact": None} or
                required or dimensions.get("warp_map") is not None or
                dimensions.get("warp_mask") is not None):
            raise ValueError("dump_manifest.json lacks required inverse-map evidence")
        for descriptor, stage in (
                (map_descriptor, "exact inverse-warp mapping"),
                (mask_descriptor, "V2 boundary-extrapolation mask")):
            if (not isinstance(descriptor, dict) or set(descriptor) != {
                    "available", "required", "stage", "description"} or
                    descriptor.get("available") is not False or
                    descriptor.get("required") is not False or
                    descriptor.get("stage") != stage or
                    not isinstance(descriptor.get("description"), str) or
                    not descriptor["description"]):
                raise ValueError("unavailable inverse-map evidence is not canonical")
        return {"available": False}

    expected_keys = {
        "available", "schema", "artifact", "width", "height", "eye_width",
        "eye_height", "source_width", "source_height", "content_scale_x",
        "content_scale_y", "dtype", "layout", "channels", "validity",
        "live_sample_source_u_normalized",
        "derived_inverse_displacement_output_eye_px",
        "derived_signed_binocular_disparity_px",
    }
    expected_validity = {
        "content": "derive from content_scale_x/content_scale_y and packed output coordinate",
        "inverse": (
            "11-step contractive fixed-point solution of crop-local q embedded by "
            "depth_input_region.json scale and outside-only zero-plane collar"
            if required else
            "11-step contractive fixed-point solution of the signed final-parallax field"),
        "mask": ("warp_mask.png red marks finite-source boundary extrapolation; V2 has no "
                 "internal owner or synthetic-fill path"),
    }
    expected_channel = "raw_reproject_source_u_normalized"
    expected_sampling = "clamp(raw_reproject_source_u_normalized, 0, 1)"
    expected_displacement = (
        "(raw_reproject_source_u_normalized - aspect_fitted_unwarped_source_u) * "
        "content_scale_x * eye_width")
    expected_disparity = (
        "invert both eye maps at common source-U samples; x_right - x_left")
    if (set(contract) != expected_keys or contract.get("artifact") != "warp_map.f32" or
            contract.get("dtype") != "float32-le" or
            contract.get("layout") != "row-major" or
            contract.get("channels") != [expected_channel] or
            contract.get("validity") != expected_validity or
            contract.get("live_sample_source_u_normalized") !=
            expected_sampling or
            contract.get("derived_inverse_displacement_output_eye_px") !=
            expected_displacement or
            contract.get("derived_signed_binocular_disparity_px") !=
            expected_disparity):
        raise ValueError("dump_manifest.json has unknown warp-map semantics")

    map_dimensions = dimensions.get("warp_map")
    mask_dimensions = dimensions.get("warp_mask")
    packed = dimensions.get("packed_sbs")
    eye = dimensions.get("eye")
    source = dimensions.get("source")
    if not all(isinstance(value, dict) for value in (
            map_dimensions, mask_dimensions, packed, eye, source)):
        raise ValueError("inverse-map evidence lacks output/source dimensions")
    width = _uint32(contract.get("width"), "warp-map width")
    height = _uint32(contract.get("height"), "warp-map height")
    eye_width = _uint32(contract.get("eye_width"), "warp-map eye width")
    eye_height = _uint32(contract.get("eye_height"), "warp-map eye height")
    source_width = _uint32(contract.get("source_width"), "warp-map source width")
    source_height = _uint32(contract.get("source_height"), "warp-map source height")
    scale_x = _finite_number(contract.get("content_scale_x"), "warp-map content scale x")
    scale_y = _finite_number(contract.get("content_scale_y"), "warp-map content scale y")
    content_fit = dimensions.get("content_fit")
    if (width == 0 or height == 0 or eye_width == 0 or eye_height == 0 or
            width != 2 * eye_width or height != eye_height or
            map_dimensions != {
                "width": width, "height": height,
                "format": "DXGI_FORMAT_R32_FLOAT", "format_value": 41} or
            mask_dimensions != {
                "width": width, "height": height,
                "format": "DXGI_FORMAT_B8G8R8A8_UNORM", "format_value": 87} or
            packed.get("width") != width or packed.get("height") != height or
            eye != {"width": eye_width, "height": eye_height} or
            source.get("width") != source_width or source.get("height") != source_height or
            not 0.0 < scale_x <= 1.0 or not 0.0 < scale_y <= 1.0 or
            not isinstance(content_fit, dict) or
            _finite_number(content_fit.get("scale_x"), "manifest content scale x") != scale_x or
            _finite_number(content_fit.get("scale_y"), "manifest content scale y") != scale_y):
        raise ValueError("inverse-map manifest geometry disagrees")

    for descriptor, stage in (
            (map_descriptor, "exact full-source inverse-warp mapping" if required else
             "exact inverse-warp mapping"),
            (mask_descriptor, "V2 boundary-extrapolation mask")):
        if (not isinstance(descriptor, dict) or set(descriptor) != {
                "available", "required", "stage", "description", "sha256"} or
                descriptor.get("available") is not True or
                descriptor.get("required") is not required or
                descriptor.get("stage") != stage or
                not isinstance(descriptor.get("description"), str) or
                not descriptor["description"] or
                not _is_sha256_hex(descriptor.get("sha256"))):
            raise ValueError("dump_manifest.json lacks authenticated inverse-map evidence")
    return {
        "available": True,
        "width": width,
        "height": height,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
    }


def validate_v2_dump_manifest_document(document: Any) -> Dict[str, Any]:
    """Validate a supported V2 geometry fragment of ``dump_manifest.json``.

    The full package contains color/model metadata owned by other contracts. This reader
    deliberately validates only the candidate -> vertical envelopes/share -> row-majorant chain,
    its analysis domain, and every authenticated
    intermediate.
    """

    if (not isinstance(document, dict) or
            document.get("schema") != DUMP_MANIFEST_SCHEMA):
        raise ValueError("dump_manifest.json has an unknown serialization schema")
    if (document.get("capture_status") != "complete" or
            document.get("published_atomically") is not True):
        raise ValueError("dump_manifest.json is not a complete atomic publication")
    timestamp = _uint64(document.get("matched_observation_timestamp_us"),
                        "dump matched observation timestamp")
    if timestamp == 0:
        raise ValueError("dump matched observation timestamp must be nonzero")
    schema = document["schema"]
    renderer = document.get("renderer")
    shadow = document.get("parallax_v2_shadow")
    artifacts = document.get("artifacts")
    dimensions = document.get("dimensions")
    if not all(isinstance(value, dict) for value in
               (renderer, shadow, artifacts, dimensions)):
        raise ValueError("dump_manifest.json is missing its V2 geometry objects")

    preview_contract = document.get("float_previews")
    if preview_contract != {
            "packaged": False,
            "generator": "tools/sbsbench/generate_dump_previews.py",
            "normalization": "finite p2-p98 computed on demand"}:
        raise ValueError("dump_manifest.json has an invalid on-demand preview contract")
    packaged_png_allowlist = {
        "source.png", "depth_input_source.png", "sbs.png", "warp_mask.png"}
    for name in artifacts:
        if ((name.endswith(".png") and name not in packaged_png_allowlist) or
                (name.endswith("_shape.json") and name != "model_input_shape.json") or
                name.startswith("warp_depth")):
            raise ValueError(
                f"dump_manifest.json advertises retired schema-38 artifact {name}")

    shader = coordinate_contract.SHADER_IMPLEMENTATION
    expected_shadow_shader = {
        "source_closure_schema": shader.source_closure_schema,
        "source_compile_flags": shader.source_compile_flags,
        "source_macro_count": shader.source_macro_count,
        "source_closure_sha256": shader.source_closure_sha256,
    }
    if (set(shadow) != {
            "requested", "active", "rendered_output_selected", "shader_source", "state"} or
            shadow.get("requested") is not False or
            shadow.get("active") is not True or
            shadow.get("rendered_output_selected") is not True or
            shadow.get("shader_source") != expected_shadow_shader):
        raise ValueError("dump_manifest.json has an invalid V2 shadow attribution")
    shadow_state_summary = _validate_shadow_manifest_summary(shadow.get("state"))

    state_descriptor = _required_hashed_artifact(
        artifacts, "shadow_state.json", "V2 shadow-state")
    stats_descriptor = _required_hashed_artifact(
        artifacts, "shadow_frame_stats.json", "V2 frame-statistics")
    if (state_descriptor.get("stage") !=
            "parallax-v2 shot calibration and attenuation state" or
            stats_descriptor.get("stage") != "parallax-v2 current-frame moments"):
        raise ValueError("dump_manifest.json has invalid V2 state artifact attribution")

    input_summary = document.get("depth_input_region")
    input_descriptor = artifacts.get("depth_input_region.json")
    input_preview_descriptor = artifacts.get("depth_input_source.png")
    expected_input_preview_description = (
        "Spatially exact full-source preview or post-completion reconstruction of the logical "
        "ROI source. Live ROI preprocessing samples the retained full matched texture directly; "
        "this transfer-aware PNG is diagnostic only, while model_input.f32 is numeric authority."
    )
    if (not isinstance(input_summary, dict) or set(input_summary) != {
            "available", "artifact", "mode", "geometry_authority", "renderer_authority"} or
            input_summary.get("available") is not True or
            input_summary.get("artifact") != "depth_input_region.json" or
            input_summary.get("mode") not in {"full-source", "window-region"} or
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
            expected_input_preview_description):
        raise ValueError("dump_manifest.json has an invalid depth-input-region contract")
    input_mode = input_summary["mode"]
    subtitle_conditioning = _validate_subtitle_conditioning_manifest(
        document, artifacts)
    subtitle_live = subtitle_conditioning["live"]
    gpu_trace = _validate_gpu_trace_manifest(document, artifacts)
    final_parallax = _validate_final_parallax_manifest(document)
    warp_map = _validate_warp_map_manifest(
        document, artifacts, dimensions, input_mode,
        shadow_state_summary["joint_plane_mode"])

    active = shadow.get("active")
    selected = renderer.get("parallax_v2_render_selected")
    if not isinstance(active, bool) or not isinstance(selected, bool):
        raise ValueError("dump_manifest.json has invalid V2 availability flags")
    shadow_selected = shadow.get("rendered_output_selected")
    if not isinstance(shadow_selected, bool) or shadow_selected != selected:
        raise ValueError("dump_manifest.json V2 selection flags disagree")
    if selected and not active:
        raise ValueError("dump_manifest.json selects V2 without an active producer")
    if subtitle_live and (not active or not selected):
        raise ValueError("active UIP4 subtitle conditioning requires selected V2 geometry")
    requested = renderer.get("parallax_v2_render_requested")
    mapping_matches = renderer.get("mapping_artifacts_match_selected_renderer")
    if (not isinstance(requested, bool) or
            not isinstance(mapping_matches, bool) or
            (selected and not requested) or
            mapping_matches != warp_map["available"]):
        raise ValueError("dump_manifest.json has invalid V2 renderer request attribution")

    expected_position = (
        "shadow_final_parallax + depth_input_region embedding"
        if selected and input_mode == "window-region" else
        "shadow_final_parallax" if selected else None
    )
    expected_authority = (
        ("authenticated crop-local atomic final field plus depth-input-region embedding"
         if input_mode == "window-region" else
         "authenticated-parallax-v2-atomic-final-field")
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
    expected_live_renderer_source_closure = LIVE_RENDERER_SOURCE_CLOSURE_SHA256
    expected_diagnostic_source_closure = DIAGNOSTIC_SOURCE_CLOSURE_SHA256
    expected_live_shader_source = ({
        "source_closure_schema": generator.SOURCE_CLOSURE_SCHEMA,
        "source_compile_flags": generator.SHADER_COMPILE_FLAGS,
        "source_macro_count": 0,
        "source_closure_sha256": expected_live_renderer_source_closure,
        "source_file": "sbs_reprojection_v2_live_ps.hlsl",
        "entrypoint": "main_ps",
        "target": "ps_5_0",
        "diagnostic_source_closure_sha256": expected_diagnostic_source_closure,
        "mapping_source_file": "sbs_reprojection_v2_diagnostics_ps.hlsl",
        "mapping_entrypoint": "mapping_ps",
        "mask_source_file": "sbs_reprojection_v2_diagnostics_ps.hlsl",
        "mask_entrypoint": "mask_ps",
    } if selected else None)
    expected_vertical_role = (
        "least column-wise upper envelope v+ >= candidate with adjacent-row source-U change <= "
        "max_vertical_shear/content_width; diagnostic evidence only"
        if selected else None
    )
    expected_conditioned_role = (
        "fixed 75/25 share of column upper/lower envelopes; may raise or lower candidate and "
        "feeds the row majorant"
        if selected else None
    )
    expected_final_role = (
        (("least row-wise crop-local q >= shadow_vertical_conditioned with horizontal slope <= "
          "max_horizontal_slope and vertical shear <= max_vertical_shear produces "
          "shadow_base_final_parallax; UIP4 publishes shadow_final_parallax atomically as "
          "direct live authority with depth_input_region embedding")
         if subtitle_live and input_mode == "window-region" else
         ("least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
          "max_horizontal_slope and vertical shear <= max_vertical_shear produces "
          "shadow_base_final_parallax; UIP4 publishes shadow_final_parallax atomically as "
          "direct live authority")
         if subtitle_live else
         ("least row-wise crop-local q >= shadow_vertical_conditioned with horizontal slope <= "
          "max_horizontal_slope and vertical shear <= max_vertical_shear publishes "
          "shadow_final_parallax atomically as direct live authority with "
          "depth_input_region embedding")
         if input_mode == "window-region" else
         ("least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
          "max_horizontal_slope and vertical shear <= max_vertical_shear publishes "
          "shadow_final_parallax atomically as direct live authority"))
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
            renderer.get("parallax_v2_vertical_majorant_role") != expected_vertical_role or
            renderer.get("parallax_v2_vertical_conditioned_role") !=
            expected_conditioned_role or
            renderer.get("parallax_v2_conditioner_role") != expected_final_role or
            renderer.get("collar_defocus") != expected_collar_defocus):
        raise ValueError("dump_manifest.json has unknown V2 conditioner attribution")

    expected_artifacts = {
        "shadow_coordinate.f32": (
            "parallax-v2 canonical coordinate diagnostic", True),
        "shadow_candidate_parallax.f32": (
            "parallax-v2 pre-limiter candidate displacement", True),
        "shadow_vertical_majorant.f32": (
            "parallax-v2 vertical shear-limiter intermediate", False),
        "shadow_vertical_conditioned.f32":
            ("parallax-v2 orientation-selective vertical conditioner", False),
        "shadow_final_parallax.f32": (
            "parallax-v2 atomic final displacement field", True),
    }
    if subtitle_live:
        expected_artifacts["shadow_base_final_parallax.f32"] = (
            "ordinary post-limiter V2 field before UIP4 conditioning", True)
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
        "shadow_coordinate", "shadow_candidate_parallax",
        "shadow_vertical_majorant",
        "shadow_vertical_conditioned",
        *(('shadow_base_final_parallax',) if subtitle_live else ()),
        "shadow_final_parallax")
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
        capture_grid = _capture_grid_relation(geometry_width, geometry_height)
        if capture_grid is None:
            raise ValueError(
                "active V2 geometry is neither calibrated DAV2 nor an exact single-high grid")
        composite_runtime = document.get("composite_runtime_provenance")
        if capture_grid[0] == "single-high-convex2x":
            raw_proof = document.get(coordinate_contract.CAPTURE_PROVENANCE_KEY)
            expected_dav2 = (
                raw_proof.get("onnx_sha256") if isinstance(raw_proof, dict) else None)
            expected_preprocess = (
                raw_proof.get("preprocess_source_closure_sha256")
                if isinstance(raw_proof, dict) else None)
            validate_composite_runtime_provenance(
                composite_runtime,
                expected_dav2_onnx_sha256=expected_dav2,
                expected_preprocess_source_closure_sha256=expected_preprocess)
        elif composite_runtime is not None:
            raise ValueError(
                "legacy DAV2 dump must not claim fused composite runtime provenance")
        if subtitle_live and not coordinate_contract.subtitle_ocr_field_is_calibrated(
                geometry_width, geometry_height):
            raise ValueError(
                "active UIP4 subtitle conditioning requires a calibrated live field")
        model_dimensions = dimensions.get("model_input")
        raw_dimensions = dimensions.get("raw_depth")
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
                raw_dimensions.get("format") != "float32-le structured buffer"):
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
                    ("DXGI_FORMAT_B8G8R8X8_UNORM", 88),
                    ("DXGI_FORMAT_R8G8B8A8_UNORM", 28),
                    ("DXGI_FORMAT_R16G16B16A16_FLOAT", 10)} or
                analysis_source_dimensions["format"] != source_dimensions["format"] or
                analysis_source_dimensions["format_value"] !=
                source_dimensions["format_value"]):
            raise ValueError("dump_manifest.json has invalid analysis-source dimensions")
        for label, extent in (
                ("source", source_dimensions),
                ("analysis-source", analysis_source_dimensions)):
            extent_width = extent["width"]
            extent_height = extent["height"]
            if (max(extent_width, extent_height) > _MAXIMUM_SOURCE_LONG_SIDE or
                    extent_width * extent_height > _MAXIMUM_SOURCE_PIXELS):
                raise ValueError(
                    f"dump_manifest.json {label} dimensions exceed supported Host SBS V2 bounds")
        source_capture_grid = capture_grid_relation_for_source(
            source_dimensions["width"], source_dimensions["height"],
            geometry_width, geometry_height)
        if source_capture_grid is None:
            raise ValueError(
                "active V2 geometry does not match the exact source-derived production profile")
        capture_grid = source_capture_grid
    elif any(value is not None for value in geometry_dimensions):
        raise ValueError("dump_manifest.json exposes inactive V2 geometry dimensions")
    if not subtitle_live and (
            "shadow_base_final_parallax.f32" in artifacts or
            "shadow_base_final_parallax" in dimensions):
        raise ValueError("inactive subtitle conditioning exposes a UIP4 base field")

    region_summary = document.get("window_region")
    region_descriptor = artifacts.get("window_region.json")
    region_available = False
    if (not isinstance(region_summary, dict) or set(region_summary) != {
            "available", "artifact", "observer_status", "mapping_status",
            "geometry_authority", "renderer_authority"} or
            not isinstance(region_descriptor, dict)):
        raise ValueError("dump_manifest.json has an invalid window-region contract")
    region_available = region_summary.get("available")
    region_observer_status = region_summary.get("observer_status")
    region_mapping_status = region_summary.get("mapping_status")
    region_required = input_mode == "window-region"
    expected_region_descriptor_keys = {"available", "required", "stage", "description"}
    if region_required:
        expected_region_descriptor_keys.add("sha256")
    if (set(region_descriptor) != expected_region_descriptor_keys or
            not isinstance(region_available, bool) or
            region_descriptor.get("available") is not region_available or
            region_descriptor.get("required") is not region_required or
            region_descriptor.get("stage") != "matched-frame window region provenance" or
            not isinstance(region_descriptor.get("description"), str) or
            not region_descriptor["description"] or
            region_summary.get("artifact") != (
                "window_region.json" if region_available else None) or
            not isinstance(region_observer_status, str) or not region_observer_status or
            not isinstance(region_mapping_status, str) or not region_mapping_status or
            (region_required and (
                not region_available or region_observer_status != "ok" or
                region_mapping_status != "ok" or
                not _is_sha256_hex(region_descriptor.get("sha256")))) or
            region_summary.get("geometry_authority") is not False or
            region_summary.get("renderer_authority") is not False):
        raise ValueError("dump_manifest.json has an inconsistent window-region contract")
    if region_observer_status == "ok-fullscreen":
        if (not region_available or region_mapping_status != "ok" or
                input_mode != "full-source"):
            raise ValueError(
                "ok-fullscreen Chromium evidence requires available mapped provenance, "
                "and full-source depth input")
    return {
        "manifest_schema": schema,
        "matched_observation_timestamp_us": timestamp,
        "active": active,
        "rendered_output_selected": selected,
        "mapping_artifacts_match_selected_renderer": mapping_matches,
        "position_field": expected_position,
        "vertical_majorant_available": active,
        "vertical_conditioned_available": active,
        "capture_grid_kind": capture_grid[0],
        "capture_grid_scale": capture_grid[1],
        "embedded_dav2_width": capture_grid[2],
        "embedded_dav2_height": capture_grid[3],
        "composite_runtime_authenticated": composite_runtime is not None,
        "depth_input_region_available": True,
        "depth_input_mode": input_mode,
        "position_authority": (
            ["shadow_final_parallax", "depth_input_region"]
            if input_mode == "window-region" else ["shadow_final_parallax"]),
        "window_region_available": region_available,
        "window_region_observer_status": region_observer_status,
        "window_region_mapping_status": region_mapping_status,
        "subtitle_conditioning": subtitle_conditioning,
        "gpu_trace": gpu_trace,
        "final_parallax": final_parallax,
        "warp_map": warp_map,
        "shadow_state_summary": shadow_state_summary,
    }


def _is_sha256_hex(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


_GEOMETRY_CHAIN_FIELDS = (
    "shadow_coordinate",
    "shadow_candidate_parallax",
    "shadow_vertical_majorant",
    "shadow_vertical_conditioned",
    "shadow_final_parallax",
)


def _read_hashed_artifact(
        dump_dir: Any, artifacts: Dict[str, Any], name: str) -> bytes:
    import hashlib
    import os

    path = os.path.join(os.fspath(dump_dir), name)
    try:
        with open(path, "rb") as handle:
            payload = handle.read()
    except OSError as error:
        raise ValueError(f"{name} is missing") from error
    descriptor = artifacts.get(name)
    if (not isinstance(descriptor, dict) or
            hashlib.sha256(payload).hexdigest() != descriptor.get("sha256")):
        raise ValueError(f"{name} content hash mismatch")
    if "byte_count" in descriptor and len(payload) != descriptor["byte_count"]:
        raise ValueError(f"{name} exact byte_count mismatch")
    return payload


def _gpu_trace_contract_expected_sections() -> Dict[str, Any]:
    return {
        "decoded_trace_schema": GPU_TRACE_DECODED_SCHEMA,
        "ring": {
            "schema": GPU_TRACE_RING_SCHEMA,
            "tag": GPU_TRACE_RING_TAG,
            "capacity": GPU_TRACE_CAPACITY,
            "header_words": GPU_TRACE_HEADER_WORD_COUNT,
            "record_words": GPU_TRACE_RECORD_WORD_COUNT,
            "record_bytes": GPU_TRACE_RECORD_WORD_COUNT * 4,
            "ring_words": GPU_TRACE_RING_WORD_COUNT,
            "ring_bytes": GPU_TRACE_RING_BYTE_COUNT,
            "record_array_word_offset": GPU_TRACE_HEADER_WORD_COUNT,
            "record_array_byte_offset": GPU_TRACE_HEADER_WORD_COUNT * 4,
            "record_slot_word_stride": GPU_TRACE_RECORD_WORD_COUNT,
            "record_slot_byte_stride": GPU_TRACE_RECORD_WORD_COUNT * 4,
            "commit_protocol": (
                "header tag invalidated before slot overwrite; payload then record tag; "
                "cursor then header tag last"),
        },
        "header_word_offsets": GPU_TRACE_HEADER_OFFSETS,
        "record_word_offsets": GPU_TRACE_RECORD_OFFSETS,
        "record_sections": {
            "transaction": {
                "word_offset": 24, "word_count": 64,
                "validity": (
                    "immutable postprocessed transaction snapshot; branch remains invalid "
                    "unless RQST/CBRG/cookies/token/class all authenticate"),
            },
            "subtitle_locator": {
                "word_offset": 88, "word_count": 40,
                "validity": (
                    "post-finalization UIP4 for an authenticated subtitle observation; "
                    "held_with_depth preserves the byte-identical immediately prior tuple "
                    "and frame identity; "
                    "subtitle_suppressed bytes are frozen/unused"),
            },
            "subtitle_condition": {
                "word_offset": 128, "word_count": 6,
                "validity": (
                    "post-finalization SCP4 for an authenticated subtitle observation; active payload "
                    "matches current UIP4 slots and applied target, inactive payload is zero with fresh "
                    "schema/tag; held_with_depth preserves the byte-identical immediately prior params; "
                    "suppressed words are unused; actual OCR8 covers are verified separately"),
            },
            "observation_timestamp": {
                "word_offset": 134, "word_count": 2,
                "validity": (
                    "nonzero monotonic source-observation microseconds; zero is invalid"),
            },
            "reserved": {
                "word_offset": 136, "word_count": 0, "validity": "must be zero"},
        },
        "transaction_word_offsets": {
            "receipt": {"word_offset": 0, "word_count": 8},
            "request": {"word_offset": 8, "word_count": 8},
            "indirect_dispatch_groups": {
                "word_offset": 16, "word_count": 48, "group_word_stride": 4},
        },
        "trace_constant_word_offsets": {
            "frame_low": 0, "frame_high": 1,
            "analysis_generation_low": 2, "analysis_generation_high": 3,
            "domain_tag_low": 4, "domain_tag_high": 5,
            "transaction_token_low": 6, "transaction_token_high": 7,
            "expected_work": 8, "submission_class": 9, "flags": 10,
            "host_subtitle_outcome": 11, "source_width": 12,
            "source_height": 13, "field_width": 14, "field_height": 15,
            "observation_timestamp_low": 16, "observation_timestamp_high": 17,
            "padding0": 18, "padding1": 19,
        },
        "tags": {"header": GPU_TRACE_RING_TAG, "record": GPU_TRACE_RECORD_TAG},
        "receipt_abi": {
            "decision_cookie": GPU_TRACE_DECISION_COOKIE,
            "token_low_cookie": GPU_TRACE_TOKEN_LOW_COOKIE,
            "token_high_cookie": GPU_TRACE_TOKEN_HIGH_COOKIE,
            "work_flags_cookie": GPU_TRACE_WORK_FLAGS_COOKIE,
            "proposal_magic": GPU_TRACE_PROPOSAL_MAGIC,
            "request_magic": GPU_TRACE_REQUEST_MAGIC,
            "receipt_magic": GPU_TRACE_RECEIPT_MAGIC,
            "optional_ocr_receipt_magic": GPU_TRACE_OPTIONAL_RECEIPT_MAGIC,
            "branch": {"reuse": 0, "infer": 1},
            "work": {
                "none": GPU_TRACE_WORK_NONE,
                "optional_ocr": GPU_TRACE_WORK_OPTIONAL_OCR,
                "subtitle_observation": GPU_TRACE_WORK_SUBTITLE_OBSERVATION,
            },
        },
        "enums": {
            "submission_class": {"invalid": 0, "force_infer": 1, "gpu_undecided": 2},
            "depth_disposition": {"invalid": 0, "reuse": 1, "infer": 2},
            "subtitle_disposition": {
                "suppressed": 0, "optional_ocr": 1, "abstention": 2,
                "held_with_depth": 5, "invalid": 6,
            },
            "host_subtitle_outcome": {
                "suppressed": 0, "ordinary_record": 1,
            },
            "flags": {
                "input_domain_reset": GPU_TRACE_FLAG_INPUT_DOMAIN_RESET,
                "dump_forced_at_enqueue": GPU_TRACE_FLAG_DUMP_FORCED_AT_ENQUEUE,
                "ocr_record_submitted": GPU_TRACE_FLAG_OCR_RECORD_SUBMITTED,
                "subtitle_suppressed": GPU_TRACE_FLAG_SUBTITLE_SUPPRESSED,
                "condition_executed": GPU_TRACE_FLAG_CONDITION_EXECUTED,
                "subtitle_branch_gated": GPU_TRACE_FLAG_SUBTITLE_BRANCH_GATED,
                "known_mask": GPU_TRACE_KNOWN_FLAGS,
            },
        },
        "subtitle_locator": {
            "schema": SUBTITLE_PLANE_STATE_SCHEMA,
            "tag": SUBTITLE_PLANE_STATE_TAG,
            "word_count": GPU_TRACE_PLANE_WORD_COUNT,
        },
        "subtitle_condition": {
            "schema": coordinate_contract.SUBTITLE_OCR.condition_param_schema,
            "tag": coordinate_contract.SUBTITLE_OCR.condition_param_tag,
            "word_count": GPU_TRACE_CONDITION_WORD_COUNT,
        },
        "shader": {
            "source_file": GPU_TRACE_SHADER_SPEC[0],
            "entrypoint": GPU_TRACE_SHADER_SPEC[1],
            "profile": GPU_TRACE_SHADER_SPEC[2],
            "source_closure_schema": generator.SOURCE_CLOSURE_SCHEMA,
            "source_compile_flags": generator.SHADER_COMPILE_FLAGS,
            "source_macro_count": 0,
            "source_closure_sha256": GPU_TRACE_SOURCE_CLOSURE_SHA256,
        },
    }


def validate_gpu_trace_contract_document(document: Any) -> Dict[str, Any]:
    """Validate the exact diagnostic trace wire/schema/provenance document."""

    expected_keys = {
        "schema", "role", "byte_order", "decoded_trace_schema", "ring", "header_word_offsets",
        "record_word_offsets", "record_sections", "transaction_word_offsets",
        "trace_constant_word_offsets", "tags", "receipt_abi", "enums",
        "subtitle_locator", "subtitle_condition", "shader",
    }
    if (not isinstance(document, dict) or set(document) != expected_keys or
            document.get("schema") != GPU_TRACE_CONTRACT_SCHEMA or
            document.get("role") !=
            "diagnostic-only accepted-root completion history; never rendering authority" or
            document.get("byte_order") != "little-endian"):
        raise ValueError("gpu_trace_contract.json has an unknown layout")
    for name, expected in _gpu_trace_contract_expected_sections().items():
        if document.get(name) != expected:
            raise ValueError(f"gpu_trace_contract.json has stale {name}")
    return {
        "schema": GPU_TRACE_CONTRACT_SCHEMA,
        "capacity": GPU_TRACE_CAPACITY,
        "record_words": GPU_TRACE_RECORD_WORD_COUNT,
        "source_closure_sha256": GPU_TRACE_SOURCE_CLOSURE_SHA256,
    }


def _gpu_trace_authenticate_receipt(
        transaction: tuple[int, ...], token: int, expected_work: int,
        submission_class: int) -> tuple[int, bool, bool]:
    valid_work = expected_work in {
        GPU_TRACE_WORK_NONE, GPU_TRACE_WORK_OPTIONAL_OCR,
        GPU_TRACE_WORK_SUBTITLE_OBSERVATION}
    token_low, token_high = token & 0xFFFFFFFF, token >> 32
    request_valid = (
        token != 0 and valid_work and transaction[8] == token_low and
        transaction[9] == token_high and
        transaction[10] == (token_low ^ GPU_TRACE_TOKEN_LOW_COOKIE) and
        transaction[11] == (token_high ^ GPU_TRACE_TOKEN_HIGH_COOKIE) and
        transaction[12] == GPU_TRACE_REQUEST_MAGIC and
        transaction[13] == expected_work and
        transaction[14] == (0 if expected_work == 0 else
                            expected_work ^ GPU_TRACE_WORK_FLAGS_COOKIE) and
        transaction[15] == 0)
    decision = transaction[0]
    marker = transaction[7]
    receipt_valid = (
        request_valid and submission_class in {1, 2} and decision in {0, 1} and
        not (submission_class == 1 and decision == 0) and
        transaction[1] == (decision ^ GPU_TRACE_DECISION_COOKIE ^ marker) and
        transaction[2] == token_low and transaction[3] == token_high and
        transaction[4] == (token_low ^ GPU_TRACE_TOKEN_LOW_COOKIE) and
        transaction[5] == (token_high ^ GPU_TRACE_TOKEN_HIGH_COOKIE) and
        transaction[6] == GPU_TRACE_RECEIPT_MAGIC and
        (marker == 0 or
         (marker == GPU_TRACE_OPTIONAL_RECEIPT_MAGIC and
          expected_work == GPU_TRACE_WORK_OPTIONAL_OCR and decision == 1)))
    optional = receipt_valid and marker == GPU_TRACE_OPTIONAL_RECEIPT_MAGIC
    depth = 0 if not receipt_valid else (1 if decision == 0 else 2)
    return depth, receipt_valid, optional


def _gpu_trace_subtitle_disposition(
        expected_work: int, host_outcome: int, depth: int,
        receipt_valid: bool, optional_ocr: bool, flags: int) -> int:
    if flags & ~GPU_TRACE_KNOWN_FLAGS:
        return 6
    published = bool(flags & GPU_TRACE_FLAG_OCR_RECORD_SUBMITTED)
    suppressed = bool(flags & GPU_TRACE_FLAG_SUBTITLE_SUPPRESSED)
    conditioned = bool(flags & GPU_TRACE_FLAG_CONDITION_EXECUTED)
    branch_gated = bool(flags & GPU_TRACE_FLAG_SUBTITLE_BRANCH_GATED)
    if branch_gated:
        if (not receipt_valid or expected_work == GPU_TRACE_WORK_NONE or
                suppressed or published or conditioned):
            return 6
        if (expected_work not in {GPU_TRACE_WORK_OPTIONAL_OCR,
                                  GPU_TRACE_WORK_SUBTITLE_OBSERVATION} or
                host_outcome != 1):
            return 6
        if depth == 1:
            return 5 if not optional_ocr else 6
        if depth != 2:
            return 6
        return 1 if optional_ocr else 2
    # Authenticated reuse can only retain the branch-gated tuple. Keep the existing
    # force-root invalid-receipt diagnostic abstention; it grants no depth authority.
    if expected_work == GPU_TRACE_WORK_NONE:
        return 0 if host_outcome == 0 and suppressed and not published and not conditioned else 6
    if receipt_valid and depth == 1:
        return 6
    if expected_work == GPU_TRACE_WORK_OPTIONAL_OCR:
        if host_outcome != 1 or suppressed or not published or not conditioned:
            return 6
        return 1 if optional_ocr else 2
    if expected_work == GPU_TRACE_WORK_SUBTITLE_OBSERVATION:
        return 2 if host_outcome == 1 and not suppressed and published and conditioned else 6
    return 6


def _gpu_trace_domain_tag(
        input_region: Dict[str, Any], color_mode: Any,
        field_width: int, field_height: int) -> int:
    color = {"srgb": 0, "linear_sdr": 1, "scrgb_hdr": 2}.get(color_mode)
    if color is None:
        raise ValueError("GPU trace requires a known dump color_mode")
    if input_region["mode"] == "full-source":
        authority = 0
    else:
        authority = {
            "chromium-video": 1,
            "foreground-client": 2,
        }.get(input_region["authorization"]["authority_kind"])
        if authority is None:
            raise ValueError("GPU trace has an unknown analysis authority")
    content = input_region["tensor_content_rect"]
    values = (
        input_region["source_width"], input_region["source_height"],
        input_region["inference_width"], input_region["inference_height"],
        *content, input_region["analysis_generation"],
        1 if input_region["mode"] == "window-region" else 0,
        authority, color, field_width, field_height,
    )
    result = 1469598103934665603
    for value in values:
        for byte in range(8):
            result ^= (value >> (byte * 8)) & 0xFF
            result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return result or 1


_GPU_TRACE_CLASS_NAMES = {0: "invalid", 1: "force-infer", 2: "gpu-undecided"}
_GPU_TRACE_DEPTH_NAMES = {0: "invalid", 1: "reuse", 2: "infer"}
_GPU_TRACE_SUBTITLE_NAMES = {
    0: "suppressed", 1: "optional-ocr", 2: "abstention",
    5: "held-with-depth", 6: "invalid",
}
_GPU_TRACE_OUTCOME_NAMES = {
    0: "suppressed", 1: "ordinary-record",
}
_GPU_TRACE_WORK_NAMES = {
    0: "none", 1: "optional-ocr", 2: "subtitle-observation",
}


def _gpu_trace_json_float(word: int) -> float | None:
    value = _float32_bits(word)
    return value if math.isfinite(value) else None


def validate_gpu_trace_ring(
        payload: Any, *, matched_frame_id: int,
        analysis_generation: int, source_width: int, source_height: int,
        field_width: int, field_height: int, expected_domain_tag: int,
        input_domain_reset: bool) -> Dict[str, Any]:
    """Validate, authenticate, and chronologically decode one raw GPU trace ring."""

    words = _uint32_words(payload, GPU_TRACE_RING_WORD_COUNT, "GPU trace ring")
    if (words[0] != GPU_TRACE_RING_SCHEMA or words[1] != GPU_TRACE_RING_TAG or
            words[2] != GPU_TRACE_CAPACITY or words[3] != GPU_TRACE_RECORD_WORD_COUNT or
            any(words[index] != 0 for index in range(8, GPU_TRACE_HEADER_WORD_COUNT))):
        raise ValueError("GPU trace ring has an invalid committed header")
    next_sequence = _uint64_words(words[4], words[5])
    next_slot, count = words[6], words[7]
    if (next_sequence == 0 or next_slot >= GPU_TRACE_CAPACITY or
            count == 0 or count > GPU_TRACE_CAPACITY or next_sequence <= count):
        raise ValueError("GPU trace ring has impossible cursor state")
    oldest_sequence = next_sequence - count
    oldest_slot = (next_slot + GPU_TRACE_CAPACITY - count) % GPU_TRACE_CAPACITY
    records = []
    matched_sequence = 0
    previous_observation_timestamp_us = 0
    for ordinal in range(count):
        slot = (oldest_slot + ordinal) % GPU_TRACE_CAPACITY
        base = GPU_TRACE_HEADER_WORD_COUNT + slot * GPU_TRACE_RECORD_WORD_COUNT
        record = words[base:base + GPU_TRACE_RECORD_WORD_COUNT]
        sequence = _uint64_words(record[2], record[3])
        if (record[0] != GPU_TRACE_RING_SCHEMA or record[1] != GPU_TRACE_RECORD_TAG or
                sequence != oldest_sequence + ordinal or record[22] != 64 or
                record[23] != 0 or any(
                    record[index] != 0 for index in range(
                        GPU_TRACE_RECORD_OFFSETS["reserved_begin"],
                        GPU_TRACE_RECORD_WORD_COUNT))):
            raise ValueError("GPU trace ring has an invalid committed record")
        frame_id = _uint64_words(record[4], record[5])
        generation = _uint64_words(record[6], record[7])
        domain_tag = _uint64_words(record[8], record[9])
        token = _uint64_words(record[10], record[11])
        observation_timestamp_us = _uint64_words(record[134], record[135])
        if (frame_id == 0 or domain_tag == 0 or token == 0 or
                record[18] == 0 or record[19] == 0 or
                record[20] == 0 or record[21] == 0):
            raise ValueError("GPU trace record has empty identity or dimensions")
        if (observation_timestamp_us == 0 or
                (previous_observation_timestamp_us != 0 and
                 observation_timestamp_us < previous_observation_timestamp_us)):
            raise ValueError("GPU trace record has an invalid observation timestamp")
        previous_observation_timestamp_us = observation_timestamp_us
        transaction = tuple(record[24:88])
        depth, receipt_valid, optional = _gpu_trace_authenticate_receipt(
            transaction, token, record[14], record[12])
        if record[13] != depth:
            raise ValueError("GPU trace depth disposition disagrees with raw receipt")
        subtitle = _gpu_trace_subtitle_disposition(
            record[14], record[17], depth, receipt_valid, optional, record[16])
        if record[15] != subtitle:
            raise ValueError("GPU trace subtitle disposition disagrees with raw proof")
        plane = tuple(record[88:128])
        condition = tuple(record[128:134])
        suppressed = bool(record[16] & GPU_TRACE_FLAG_SUBTITLE_SUPPRESSED)
        subtitle_held = subtitle == 5
        if subtitle_held and ordinal != 0:
            previous_slot = (oldest_slot + ordinal - 1) % GPU_TRACE_CAPACITY
            previous_base = (GPU_TRACE_HEADER_WORD_COUNT +
                             previous_slot * GPU_TRACE_RECORD_WORD_COUNT)
            if (plane != tuple(words[previous_base + 88:previous_base + 128]) or
                    condition != tuple(words[previous_base + 128:previous_base + 134])):
                raise ValueError(
                    "GPU trace held subtitle tuple differs from the immediately prior record")
        if not suppressed:
            plane_frame_id = _uint64_words(plane[4], plane[5])
            if (plane_frame_id == 0 or
                    (plane_frame_id >= frame_id if subtitle_held else plane_frame_id != frame_id)):
                raise ValueError("GPU trace has an invalid finalized UIP4 frame")
            decoded_plane = validate_subtitle_plane_snapshot(
                struct.pack("<40I", *plane), matched_frame_id=plane_frame_id,
                analysis_generation=generation, source_width=record[18], source_height=record[19],
                field_width=record[20], field_height=record[21])
            if subtitle == 2 and (decoded_plane["current_valid"] or decoded_plane["current_slot_mask"]):
                raise ValueError("GPU trace abstention cannot grant current UIP4 slots")
            if decoded_plane["observation_timestamp_us"] > observation_timestamp_us:
                raise ValueError("GPU trace UIP4 source time cannot be newer than the transaction")
            if not subtitle_held and decoded_plane["observation_timestamp_us"] != observation_timestamp_us:
                raise ValueError("GPU trace UIP4 source time differs from current publication")
            validate_subtitle_condition_words(
                struct.pack("<6I", *condition), plane_state=decoded_plane,
                geometry_ready=decoded_plane["current_valid"], require_ocr_kinds=False)
        else:
            decoded_plane = None

        matched = frame_id == matched_frame_id
        if matched:
            if (generation != analysis_generation or domain_tag != expected_domain_tag or
                    record[18] != source_width or record[19] != source_height or
                    record[20] != field_width or record[21] != field_height or
                    bool(record[16] & GPU_TRACE_FLAG_INPUT_DOMAIN_RESET) !=
                    input_domain_reset):
                raise ValueError("GPU trace matched record disagrees with the dump domain")
            matched_sequence = sequence

        condition_valid = (
            not suppressed and
            condition[0] == coordinate_contract.SUBTITLE_OCR.condition_param_schema and
            condition[1] == coordinate_contract.SUBTITLE_OCR.condition_param_tag)
        condition_target = _gpu_trace_json_float(condition[5])
        host_condition_executed = bool(
            record[16] & GPU_TRACE_FLAG_CONDITION_EXECUTED)
        branch_gated = bool(
            record[16] & GPU_TRACE_FLAG_SUBTITLE_BRANCH_GATED)
        condition_executed = bool(
            host_condition_executed or
            (branch_gated and subtitle in {1, 2}))
        condition_published = condition_executed or subtitle_held
        condition_active = bool(
            condition_published and condition_valid and condition[2] != 0 and
            condition[4] == 2 and condition_target is not None and
            0.0 <= condition_target <=
            coordinate_contract.CALIBRATED_DEFAULTS.direct_container_limit)
        condition_summary = ({
            "active": False,
            "unused": True,
            "reason": "subtitle-suppressed transaction freezes UIP4 and skips conditioning",
        } if suppressed else {
            "active": condition_active,
            "valid": condition_valid,
            "executed": condition_executed,
            "held_with_depth": subtitle_held,
            "unused": False,
            "schema": condition[0],
            "tag": condition[1],
            "current_count": condition[2],
            "current_kinds": condition[3],
            "active_level": condition[4],
            "target": condition_target,
            "target_bits": condition[5],
        })
        records.append({
            "sequence": sequence,
            "ring_slot": slot,
            "frame_id": frame_id,
            "analysis_generation": generation,
            "domain_tag": domain_tag,
            "transaction_token": token,
            "observation_timestamp_us": observation_timestamp_us,
            "submission_class": {
                "value": record[12], "name": _GPU_TRACE_CLASS_NAMES.get(record[12], "invalid")},
            "depth_disposition": {
                "value": record[13], "name": _GPU_TRACE_DEPTH_NAMES.get(record[13], "invalid")},
            "expected_work": {
                "value": record[14], "name": _GPU_TRACE_WORK_NAMES.get(record[14], "invalid")},
            "subtitle_disposition": {
                "value": record[15],
                "name": _GPU_TRACE_SUBTITLE_NAMES.get(record[15], "invalid")},
            "host_subtitle_outcome": {
                "value": record[17],
                "name": _GPU_TRACE_OUTCOME_NAMES.get(record[17], "invalid")},
            "flags": {
                "value": record[16],
                "input_domain_reset": bool(record[16] & GPU_TRACE_FLAG_INPUT_DOMAIN_RESET),
                "dump_forced_at_enqueue": bool(
                    record[16] & GPU_TRACE_FLAG_DUMP_FORCED_AT_ENQUEUE),
                "ocr_record_submitted": bool(
                    record[16] & GPU_TRACE_FLAG_OCR_RECORD_SUBMITTED),
                "subtitle_suppressed": suppressed,
                "condition_executed": condition_executed,
                "condition_executed_host_proven": host_condition_executed,
                "subtitle_branch_gated": branch_gated,
            },
            "analysis_source": {"width": record[18], "height": record[19]},
            "field": {"width": record[20], "height": record[21]},
            "matched_dump_frame": matched,
            "subtitle_plane": {
                "schema": plane[0], "tag": plane[1],
                "flags": {"value": plane[2], "current_valid": plane[2] == SUBTITLE_PLANE_CURRENT_VALID_FLAG},
                "current_slot_mask": plane[3], "current_count": plane[3].bit_count(),
                "frame_id": _uint64_words(plane[4], plane[5]),
                "analysis_generation": _uint64_words(plane[6], plane[7]),
                "observation_timestamp_us": _uint64_words(plane[8], plane[9]),
                "cut_epoch": plane[20], "target": _gpu_trace_json_float(plane[27]),
                "target_bits": plane[27],
                "adaptive": {
                    "initialized": plane[24] == 1, "flags": plane[25], "current_level": plane[26],
                    "accepted_observation_timestamp_us": _uint64_words(plane[28], plane[29]),
                    "approach_since_us": _uint64_words(plane[30], plane[31]),
                    "retreat_since_us": _uint64_words(plane[32], plane[33]),
                    "approach_level": plane[34], "retreat_level": plane[35],
                    "display_budget": _gpu_trace_json_float(plane[36]),
                    "covered_count": plane[37], "conflict_count": plane[38], "invalid_count": plane[39],
                },
            },
            "subtitle_condition": condition_summary,
            "transaction_words": list(transaction),
            "subtitle_plane_words": list(plane),
            "subtitle_condition_words": list(condition),
        })
    if matched_sequence == 0:
        raise ValueError("GPU trace ring does not contain the dump's matched frame")
    decoded = {
        "schema": GPU_TRACE_DECODED_SCHEMA,
        "role": "decoded diagnostic history; no rendering authority",
        "ring": {
            "schema": GPU_TRACE_RING_SCHEMA, "tag": GPU_TRACE_RING_TAG,
            "capacity": GPU_TRACE_CAPACITY, "record_words": GPU_TRACE_RECORD_WORD_COUNT,
            "next_sequence": next_sequence, "next_slot": next_slot,
            "committed_count": count, "oldest_sequence": oldest_sequence,
        },
        "capture_match": {
            "matched_frame_id": matched_frame_id,
            "analysis_generation": analysis_generation,
            "source_width": source_width, "source_height": source_height,
            "field_width": field_width, "field_height": field_height,
            "sequence": matched_sequence,
        },
        "records": records,
    }
    return {
        "record_count": count,
        "oldest_sequence": oldest_sequence,
        "next_sequence": next_sequence,
        "matched_sequence": matched_sequence,
        "decoded": decoded,
    }


def _verify_gpu_trace_artifacts(
        dump_dir: Any, manifest: Dict[str, Any], *,
        input_region: Dict[str, Any], field_width: int, field_height: int) -> Dict[str, Any]:
    import json

    summary = manifest["gpu_trace"]
    if summary["available"] is not True:
        return {"available": False}
    artifacts = manifest["artifacts"]
    raw = _read_hashed_artifact(dump_dir, artifacts, "gpu_trace_ring.u32")
    decoded_payload = _read_hashed_artifact(dump_dir, artifacts, "gpu_trace.json")
    contract_payload = _read_hashed_artifact(
        dump_dir, artifacts, "gpu_trace_contract.json")
    try:
        decoded_document = json.loads(decoded_payload.decode("utf-8"))
        contract_document = json.loads(contract_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("GPU trace JSON artifact is malformed") from error
    validate_gpu_trace_contract_document(contract_document)
    expected_domain = _gpu_trace_domain_tag(
        input_region, manifest.get("color_mode"), field_width, field_height)
    ring = validate_gpu_trace_ring(
        raw,
        matched_frame_id=_uint64(
            manifest.get("matched_frame_id"), "dump manifest matched frame id"),
        analysis_generation=input_region["analysis_generation"],
        source_width=input_region["inference_width"],
        source_height=input_region["inference_height"],
        field_width=field_width,
        field_height=field_height,
        expected_domain_tag=expected_domain,
        input_domain_reset=input_region["input_domain_reset"],
    )
    if decoded_document != ring["decoded"]:
        raise ValueError("gpu_trace.json disagrees with the authenticated raw ring")
    for key in ("record_count", "oldest_sequence", "next_sequence", "matched_sequence"):
        if summary.get(key) != ring[key]:
            raise ValueError("dump manifest GPU trace summary disagrees with the raw ring")
    matched_record = next(
        record for record in ring["decoded"]["records"]
        if record["matched_dump_frame"])
    if matched_record["observation_timestamp_us"] != manifest["matched_observation_timestamp_us"]:
        raise ValueError("GPU trace matched source observation timestamp disagrees with dump")
    return {
        "available": True,
        "record_count": ring["record_count"],
        "oldest_sequence": ring["oldest_sequence"],
        "next_sequence": ring["next_sequence"],
        "matched_sequence": ring["matched_sequence"],
        "source_closure_sha256": GPU_TRACE_SOURCE_CLOSURE_SHA256,
        "matched_record": matched_record,
        "records": ring["decoded"]["records"],
    }


def _subtitle_contour_cover_mask(
        rectangles: Any, source_width: int, source_height: int,
        crop_top: int, crop_height: int, content: tuple[int, int, int, int]) -> Any:
    import numpy as np

    left, top, right, bottom = content
    x = 2 * left * source_width + (2 * np.arange(source_width, dtype=np.int64) + 1) * (right - left)
    y = 2 * top * source_height + (2 * (np.arange(crop_height, dtype=np.int64) + crop_top) + 1) * (bottom - top)
    covered = np.zeros((crop_height, source_width), dtype=bool)
    for rectangle in rectangles:
        covered |= ((x[None, :] >= 2 * rectangle["left"] * source_width) &
                    (x[None, :] < 2 * rectangle["right"] * source_width) &
                    (y[:, None] >= 2 * rectangle["top"] * source_height) &
                    (y[:, None] < 2 * rectangle["bottom"] * source_height))
    return covered


def _subtitle_contour_project_support(
        support: Any, source_height: int, crop_top: int,
        field_width: int, field_height: int, content: tuple[int, int, int, int]) -> Any:
    """Independent integer summed-area oracle for the complete bilinear-node footprint."""
    import numpy as np

    native = np.asarray(support, dtype=bool)
    crop_height, source_width = native.shape
    left, top, right, bottom = content
    x = np.arange(right - left, dtype=np.int64)
    y = np.arange(bottom - top, dtype=np.int64)
    begin_x = np.maximum(0, ((2 * x - 1) * source_width) // (2 * (right - left)))
    end_x = np.minimum(source_width, -(-(2 * x + 3) * source_width // (2 * (right - left))))
    begin_y = np.maximum(crop_top, ((2 * y - 1) * source_height) // (2 * (bottom - top))) - crop_top
    end_y = np.minimum(crop_top + crop_height, -(-(2 * y + 3) * source_height // (2 * (bottom - top)))) - crop_top
    begin_y = np.clip(begin_y, 0, crop_height)
    end_y = np.clip(end_y, 0, crop_height)
    prefix = np.pad(native.astype(np.int64), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    sums = (prefix[end_y[:, None], end_x[None, :]] - prefix[begin_y[:, None], end_x[None, :]] -
            prefix[end_y[:, None], begin_x[None, :]] + prefix[begin_y[:, None], begin_x[None, :]])
    result = np.zeros((field_height, field_width), dtype=bool)
    result[top:bottom, left:right] = sums > 0
    return result


def _subtitle_contour_squared_source_distance(
        support: Any, source_width: int, source_height: int, crop_top: int,
        field_width: int, field_height: int, content: tuple[int, int, int, int]) -> Any:
    """Independent float64 nearest-five-center minimum; no classifier or stereo rendering.

    Horizontal nearest-center queries and a float64 parabola envelope avoid the native FP32
    prefix/hull implementation. The declared numerical envelope is checked separately.
    """
    import numpy as np

    mask = np.asarray(support, dtype=bool)
    left, top, right, bottom = content
    result = np.full((field_height, field_width), np.inf, dtype=np.float64)
    if not mask.any():
        return result
    offset = coordinate_contract.EXPECTED_SUBTITLE_OCR["contour_policy"]["gaussian_center_offset_source_pixels"]
    expanded = np.zeros((mask.shape[0] + 2 * offset, source_width + 2 * offset), dtype=bool)
    for dx, dy in ((0, 0), (-offset, 0), (offset, 0), (0, -offset), (0, offset)):
        expanded[offset + dy:offset + dy + mask.shape[0], offset + dx:offset + dx + source_width] |= mask
    x = (np.arange(right - left, dtype=np.float64) + .5) * source_width / (right - left) - .5 + offset
    y = (np.arange(bottom - top, dtype=np.float64) + .5) * source_height / (bottom - top) - .5 - crop_top + offset
    costs = np.full((expanded.shape[0], right - left), np.inf, dtype=np.float64)
    for row in range(expanded.shape[0]):
        centers = np.flatnonzero(expanded[row])
        if not centers.size:
            continue
        position = np.searchsorted(centers, x)
        before = centers[np.maximum(position - 1, 0)]
        after = centers[np.minimum(position, len(centers) - 1)]
        costs[row] = np.minimum((x - before) ** 2, (x - after) ** 2)
    rows = np.flatnonzero(np.isfinite(costs[:, 0]))
    for column in range(right - left):
        hull, starts = [], []
        for row in rows:
            row = int(row)
            start = -math.inf
            while hull:
                previous = hull[-1]
                start = .5 * (row + previous) + (costs[row, column] - costs[previous, column]) / (2 * (row - previous))
                if start > starts[-1]:
                    break
                hull.pop(); starts.pop()
            hull.append(row); starts.append(-math.inf if len(hull) == 1 else start)
        chosen = np.maximum(np.searchsorted(starts, y, side="left") - 1, 0)
        selected = np.asarray(hull, dtype=np.int64)[chosen]
        result[top:bottom, left + column] = costs[selected, column] + (y - selected) ** 2
    return result


def _subtitle_gaussian_q_error_bound(reference: Any, source_width: int, source_height: int) -> Any:
    """Explicit IEEE FP32 coordinate/square/add bound shared with native exhaustive tests.

    Native x uses four and y six epsilon-scaled source-coordinate operations. This is a
    representational admission envelope, never a quality tolerance or exact-word claim.
    """
    import numpy as np

    q = np.asarray(reference, dtype=np.float64)
    epsilon = float(np.finfo(np.float32).eps)
    delta = math.hypot(4 * epsilon * source_width, 6 * epsilon * source_height)
    return 2 * np.sqrt(q) * delta + delta * delta + 4 * epsilon * np.maximum(q, 1)


def _subtitle_gaussian_weight_from_q(q: Any, source_width: int, source_height: int) -> Any:
    import numpy as np

    policy = coordinate_contract.EXPECTED_SUBTITLE_OCR["contour_policy"]
    sigma = np.divide(np.float32(min(source_width, source_height)),
                      np.float32(policy["gaussian_sigma_short_side_divisor"]), dtype=np.float32)
    denominator = np.multiply(np.float32(2), np.multiply(sigma, sigma, dtype=np.float32), dtype=np.float32)
    values = np.asarray(q, dtype=np.float32)
    with np.errstate(under="ignore"):
        result = np.exp(-values.astype(np.float64) / float(denominator)).astype(np.float32)
    result[np.isposinf(values)] = np.float32(0)
    result[values == 0] = np.float32(1)
    return result


def _subtitle_current_cover_gaussian_q(field_width: int, field_height: int, source_width: int,
                                       source_height: int, content: tuple[int, int, int, int],
                                       rectangles: list[Dict[str, Any]]) -> Any:
    """Presentation distance to current rectangles; this never changes native support/q."""
    import numpy as np

    cx = np.clip(np.arange(field_width), content[0], content[2] - 1)
    cy = np.clip(np.arange(field_height), content[1], content[3] - 1)
    scale_x = np.float32(source_width) / np.float32(content[2] - content[0])
    scale_y = np.float32(source_height) / np.float32(content[3] - content[1])
    x = cx.astype(np.float32) + np.float32(.5)
    y = cy.astype(np.float32) + np.float32(.5)
    q = np.full((field_height, field_width), np.inf, dtype=np.float32)
    for rectangle in rectangles:
        left, right = np.float32(rectangle["left"]), np.float32(rectangle["right"])
        top, bottom = np.float32(rectangle["top"]), np.float32(rectangle["bottom"])
        dx = np.maximum(np.maximum(left - x, np.float32(0)), x - right) * scale_x
        dy = np.maximum(np.maximum(top - y, np.float32(0)), y - bottom) * scale_y
        q = np.minimum(q, dy[:, None] * dy[:, None] + dx[None, :] * dx[None, :])
    return q


def validate_subtitle_contour_tuple(
        payloads: Any, *, evidence_frame_id: int, analysis_generation: int,
        source_width: int, source_height: int, field_width: int, field_height: int,
        tensor_content: tuple[int, int, int, int], roi_top: int, roi_bottom: int,
        scene_epoch: int, observation_timestamp_us: int, geometry_ready: bool,
        ocr_authoritative: bool, current_rectangles: Any,
        plane_state: Any) -> Dict[str, Any]:
    """Authenticate the eleven joined contour/Gaussian artifacts, without reimplementing classification.

    The context is supplied by the strict OCR8/UIP4, geometry and accepted-root trace readers.
    Source belongs to the retained inference owner. It is deliberately not compared with current
    presentation RGB. Probability is opaque detector evidence, never a recovered glyph alpha.
    """
    import numpy as np

    if not isinstance(payloads, dict) or set(payloads) != set(_SUBTITLE_CONTOUR_ARTIFACT_FILES):
        raise ValueError("subtitle contour tuple must contain all eleven exact artifact roles")
    layouts = _subtitle_contour_artifact_layouts(source_width, source_height, field_width, field_height)
    for role, payload in payloads.items():
        if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) != layouts[role][1]:
            raise ValueError(f"{role} has an invalid exact byte length/type")
    content = _subtitle_tensor_content(field_width, field_height, tensor_content)
    _uint64(evidence_frame_id, "contour inference-owner frame")
    _uint64(analysis_generation, "contour analysis generation")
    _uint64(observation_timestamp_us, "contour inference-owner timestamp")
    _uint32(scene_epoch, "contour scene epoch")
    if (evidence_frame_id == 0 or observation_timestamp_us == 0 or
            type(geometry_ready) is not bool or type(ocr_authoritative) is not bool or
            not isinstance(current_rectangles, list)):
        raise ValueError("subtitle contour tuple has invalid owner/gate identity")
    if (type(roi_top) is not int or type(roi_bottom) is not int or
            not content[1] <= roi_top < roi_bottom <= content[3]):
        raise ValueError("subtitle contour tuple has invalid authenticated OCR ROI")
    if len(current_rectangles) > SUBTITLE_PLANE_COVER_CAPACITY:
        raise ValueError("subtitle contour tuple has too many current covers")
    for rectangle in current_rectangles:
        if (not isinstance(rectangle, dict) or
                any(type(rectangle.get(key)) is not int
                    for key in ("left", "top", "right", "bottom")) or
                not content[0] <= rectangle["left"] < rectangle["right"] <= content[2] or
                not content[1] <= rectangle["top"] < rectangle["bottom"] <= content[3]):
            raise ValueError("subtitle contour tuple has invalid authenticated current covers")
    if not isinstance(plane_state, dict) or plane_state.get("cover_geometry_verified") is not True:
        raise ValueError("subtitle contour tuple requires authenticated UIP4/current OCR8 state")
    current_count = _uint32(plane_state.get("current_count"), "contour UIP4 current count")
    flags = _uint32(plane_state.get("flags"), "contour UIP4 flags")
    plane_content = plane_state.get("tensor_content_rect")
    if (current_count > SUBTITLE_PLANE_COVER_CAPACITY or current_count != len(current_rectangles) or
            flags not in (0, SUBTITLE_PLANE_CURRENT_VALID_FLAG) or
            plane_state.get("current_valid") != (flags == SUBTITLE_PLANE_CURRENT_VALID_FLAG) or
            plane_state.get("current_slot_mask") != (1 << current_count) - 1 or
            (flags == 0 and current_count != 0) or
            plane_state.get("current_rectangles") != current_rectangles or
            plane_state.get("matched_frame_id") != evidence_frame_id or
            plane_state.get("analysis_generation") != analysis_generation or
            plane_state.get("observation_timestamp_us") != observation_timestamp_us or
            plane_state.get("scene_epoch") != scene_epoch or
            plane_state.get("source_width") != source_width or plane_state.get("source_height") != source_height or
            plane_state.get("field_width") != field_width or plane_state.get("field_height") != field_height or
            not isinstance(plane_content, (tuple, list)) or tuple(plane_content) != content):
        raise ValueError("subtitle contour tuple disagrees with authenticated UIP4/current OCR8 context")
    current_authority = bool(plane_state["current_valid"] and current_count and
                             plane_state["adaptive"]["invalid_count"] == 0)
    policy = coordinate_contract.EXPECTED_SUBTITLE_OCR["contour_policy"]
    eligible = max(source_width, source_height) <= policy["source_max_extent"]
    crop_height = min(source_height, (source_width + 5) // 6)
    crop_top = source_height - crop_height
    native_shape = ((crop_height, source_width) if eligible else (1, 1))
    support = np.frombuffer(payloads["contour_support"], dtype="<u4").reshape(native_shape)
    pin = np.frombuffer(payloads["contour_field_pin"], dtype="<u4").reshape(field_height, field_width)
    distance = np.frombuffer(payloads["contour_distance"], dtype="<f4").reshape(field_height, field_width)
    identity = _uint32_words(payloads["contour_identity"], policy["history_word_count"], "SCT5 temporal-mask identity")
    source = np.frombuffer(payloads["contour_source"], dtype="<f2").reshape(*native_shape, 4)
    history = np.frombuffer(payloads["contour_history"], dtype="<f4").reshape(*native_shape, 4)
    probability = np.frombuffer(payloads["ocr_probability"], dtype="<f4")
    authority = bool(eligible and geometry_ready and ocr_authoritative and current_authority)
    snapshot_gate = bool(eligible and geometry_ready and ocr_authoritative)
    expected_identity = [
        policy["history_schema"], policy["history_tag"], int(authority), scene_epoch if snapshot_gate else 0,
        evidence_frame_id & 0xFFFFFFFF, evidence_frame_id >> 32,
        analysis_generation & 0xFFFFFFFF, analysis_generation >> 32,
        source_width, source_height, crop_top, crop_height,
        field_width, field_height, roi_top, roi_bottom, *content,
        observation_timestamp_us & 0xFFFFFFFF, observation_timestamp_us >> 32,
        policy["recipe"], *([0] * (policy["history_word_count"] - 23)),
    ]
    expected_identity[policy["history_snapshot_valid_word"]] = int(snapshot_gate)
    if identity != tuple(expected_identity):
        raise ValueError("SCT5 temporal-mask identity disagrees with exact owner/clock/generation/cut/source/ROI/content/recipe/snapshot/current authority")
    admitted = support != 0
    if np.any(support > 1) or np.any(pin > 1):
        raise ValueError("subtitle temporal support/pin has nonbinary support or unknown enum values")
    if (np.any(np.isnan(distance)) or np.any(distance < 0) or np.any(np.isneginf(distance)) or
            np.any((distance == 0) & (distance.view(np.uint32) != 0))):
        raise ValueError("Gaussian squared source distance has invalid nonnegative float/positive-zero evidence")
    if not snapshot_gate:
        if np.any(source.view(np.uint16)) or np.any(probability.view(np.uint32)):
            raise ValueError("globally invalid OCR/geometry/source must publish canonical cleared source/probability snapshots")
    else:
        if not np.isfinite(source).all() or np.any((source[..., 3] != 0) & (source[..., 3] != 1)):
            raise ValueError("contour source snapshot has invalid finite RGBA16F/alpha evidence")
        if np.any(source[source[..., 3] == 0].view(np.uint16)):
            raise ValueError("alpha-zero contour source pixels must be canonical zero")
    if not authority and (admitted.any() or pin.any() or np.any(distance.view(np.uint32) != policy["distance_empty_bits"])):
        raise ValueError("invalid contour pin authority must publish canonical empty support/pins/distance")
    color_valid = ((source[..., 3] == 1) & np.isfinite(source[..., :3]).all(axis=2) &
                   (source[..., :3] >= 0).all(axis=2) & (source[..., :3] <= 1).all(axis=2))
    if authority:
        covers = _subtitle_contour_cover_mask(current_rectangles, source_width, source_height, crop_top, crop_height, content)
        if np.any(admitted & (~covers | ~color_valid)):
            raise ValueError("native temporal support escapes current covers or valid current source pixels")
    # Authenticate the saved state, not an unobserved temporal chain or a replicated classifier.
    # Valid empty OCR still retains current RGB at count0. Counts0/1 use current RGB exactly;
    # established streaks carry a fixed anchor bounded against this genuine current observation.
    valid_bit = policy["history_feature_valid_bit"]
    mask_bit = policy["history_feature_mask_bit"]
    confirmation_count = policy["history_confirmation_observations"]
    count_shift = policy["history_feature_count_shift"]
    legal_codes = np.array([
        valid_bit | (count << count_shift) | (mask_bit if count == confirmation_count else 0)
        for count in range(confirmation_count + 1)
    ], dtype=np.float32)
    code = history[..., 3]
    valid_history = color_valid & snapshot_gate
    if (np.any(history[~valid_history].view(np.uint32)) or
            np.any(~np.isin(code[valid_history], legal_codes))):
        raise ValueError("temporal history has noncanonical invalid pixels or unknown confirmation codes")
    # Cast only validated small exact codes, avoiding NaN/Inf integer coercion.
    counts = np.zeros(native_shape, dtype=np.uint32)
    counts[valid_history] = code[valid_history].astype(np.uint32) >> count_shift
    if np.any(admitted != (valid_history & (counts == confirmation_count))):
        raise ValueError("temporal history confirmed count and binary support disagree")
    if not authority and np.any(counts):
        raise ValueError("temporal history without current pin authority must retain count0 only")
    if authority and np.any((counts != 0) & ~covers):
        raise ValueError("temporal history confirmation streak escapes current covers")
    current_rgb = source[..., :3].astype(np.float32)
    current_anchor = valid_history & (counts <= 1)
    if not np.array_equal(history[..., :3][current_anchor].view(np.uint32),
                          current_rgb[current_anchor].view(np.uint32)):
        raise ValueError("temporal history count0/1 anchor is not bit-exact current source RGB")
    established = valid_history & (counts >= 2)
    anchor = history[..., :3][established]
    if (not np.isfinite(anchor).all() or np.any((anchor < 0) | (anchor > 1)) or
            np.any(np.max(np.abs(anchor - current_rgb[established]), axis=1) >
                   np.float32(policy["history_rgb_distance"]))):
        raise ValueError("temporal history established anchor is invalid or differs too far from current RGB")
    anchor_mismatch = valid_history & np.any(
        history[..., :3].view(np.uint32) != current_rgb.view(np.uint32), axis=2)
    expected_pin = _subtitle_contour_project_support(admitted, source_height, crop_top, field_width, field_height, content)
    if not np.array_equal(pin, expected_pin.astype(np.uint32)):
        raise ValueError("contour field pins differ from the exact integer bilinear footprint")
    reference = _subtitle_contour_squared_source_distance(admitted, source_width, source_height, crop_top,
        field_width, field_height, content)
    finite_reference = np.isfinite(reference)
    if (np.any(distance[~finite_reference].view(np.uint32) != policy["distance_empty_bits"]) or
            not np.isfinite(distance[finite_reference]).all()):
        raise ValueError("Gaussian squared source distance empty/padding or finite-content state differs from admitted centers")
    error = np.abs(distance[finite_reference].astype(np.float64) - reference[finite_reference])
    bound = _subtitle_gaussian_q_error_bound(reference[finite_reference], source_width, source_height)
    if np.any(error > bound):
        raise ValueError("Gaussian squared source distance differs from independent nearest-five-center minimum beyond its IEEE coordinate bound")
    weight = np.frombuffer(payloads["gaussian_weight"], dtype="<f4").reshape(field_height, field_width)
    proposal = np.frombuffer(payloads["gaussian_blend"], dtype="<f4").reshape(field_height, field_width)
    vertical = np.frombuffer(payloads["gaussian_vertical_conditioned"], dtype="<f4").reshape(field_height, field_width)
    control = _uint32_words(payloads["gaussian_control"], 1, "Gaussian smoothing control")[0]
    warmup = _uint32(plane_state.get("warmup_packed"), "UIP4 packed warmup")
    warmup_policy = coordinate_contract.SUBTITLE_OCR
    if warmup & ~(warmup_policy.plane_warmup_remaining_mask | warmup_policy.plane_warmup_cue_bit) or (warmup & warmup_policy.plane_warmup_remaining_mask) > warmup_policy.plane_warmup_observations or (warmup and not authority):
        raise ValueError("UIP4 packed warmup lacks current nonempty valid authority")
    warmup_remaining = warmup & warmup_policy.plane_warmup_remaining_mask
    presentation_q = distance
    if warmup_remaining:
        presentation_q = np.minimum(distance, _subtitle_current_cover_gaussian_q(
            field_width, field_height, source_width, source_height, content, current_rectangles))
    expected_control = int(authority and np.isfinite(presentation_q[content[1], content[0]]) and presentation_q[content[1], content[0]] >= 0)
    if control not in (0, 1) or control != expected_control:
        raise ValueError("Gaussian smoothing control disagrees with current authority and first-content distance")
    if (not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)) or
            not np.isfinite(proposal).all() or not np.isfinite(vertical).all() or
            np.any(np.abs(proposal) > np.float32(_DEFAULTS.direct_container_limit)) or
            np.any(np.abs(vertical) > np.float32(_DEFAULTS.direct_container_limit))):
        raise ValueError("Gaussian weight/proposal/vertical contains invalid finite unit-range/container evidence")
    cx = np.clip(np.arange(field_width), content[0], content[2] - 1)
    cy = np.clip(np.arange(field_height), content[1], content[3] - 1)
    joined_q = presentation_q[np.ix_(cy, cx)]
    expected_weight = (_subtitle_gaussian_weight_from_q(joined_q, source_width, source_height)
                       if control else np.zeros_like(weight))
    if (np.any(np.abs(weight.astype(np.float64) - expected_weight.astype(np.float64)) > policy["gaussian_weight_absolute_error_limit"]) or
            np.any(weight[(joined_q == 0) & bool(control)].view(np.uint32) != np.float32(1).view(np.uint32)) or
            (not control and np.any(weight.view(np.uint32)))):
        raise ValueError("Gaussian weight differs from literal untruncated exp or exact disabled/center endpoints")
    return {
        "tuple_verified": True, "authority_valid": authority, "snapshot_gate_valid": snapshot_gate,
        "native_source_eligible": eligible, "native_width": native_shape[1], "native_height": native_shape[0],
        "inference_owner_frame_id": evidence_frame_id, "inference_owner_observation_timestamp_us": observation_timestamp_us,
        "support_pixels": int(admitted.sum()), "field_pin_nodes": int(pin.sum()),
        "weak_support_pixels": 0,
        "support_semantics": "binary-current-temporal-RGB-stability-not-semantic-text-membership",
        "history_rgb_valid_pixels": int(valid_history.sum()),
        "history_confirmation_count_histogram": {
            str(count): int(np.count_nonzero(valid_history & (counts == count)))
            for count in range(confirmation_count + 1)},
        "history_anchor_current_bit_mismatch_pixels": int(anchor_mismatch.sum()),
        "history_anchor_current_numeric_mismatch_pixels": int(np.count_nonzero(
            valid_history & np.any(history[..., :3] != current_rgb, axis=2))),
        "history_confirmation_chain_verified": False,
        "warmup_packed": warmup, "warmup_remaining": warmup_remaining,
        "gaussian_presentation_policy": "current-cover-max-union-until-four-genuine-cue-free-observations-otherwise-confirmed-pixels",
        "source_alpha_zero_pixels": int(np.count_nonzero(source[..., 3] == 0)),
        "probability_nonfinite_words": int(np.count_nonzero(~np.isfinite(probability))),
        "probability_out_of_range_finite_words": int(np.count_nonzero(np.isfinite(probability) & ((probability < 0) | (probability > 1)))),
        "probability_semantics": "opaque-joined-detector-evidence-not-glyph-alpha",
        "source_pixel_relation": "sealed-inference-owner-snapshot-no-current-presentation-RGB-equality-claim",
        "integer_bilinear_projection_verified": True, "squared_source_distance_verified": True,
        "squared_source_distance_admission": "float64-independent-nearest-five-center-IEEE-coordinate-error-envelope",
        "squared_source_distance_max_absolute_error": float(error.max()) if error.size else 0.0,
        "squared_source_distance_max_error_to_bound_ratio": float(np.max(error / bound)) if error.size else 0.0,
        "gaussian_smoothing_enabled": bool(control), "gaussian_weight_verified": True,
        "gaussian_weight_max_absolute_error": float(np.max(np.abs(weight.astype(np.float64) - expected_weight))),
    }


def _verify_subtitle_conditioning_artifacts(
        dump_dir: Any, manifest: Dict[str, Any], *, matched_frame_id: int,
        analysis_generation: int, source_width: int, source_height: int,
        field_width: int, field_height: int,
        tensor_content: tuple[int, int, int, int],
        confirmed_cut_count: int, gpu_trace: Dict[str, Any], shadow_state: Dict[str, Any],
        joint_plane_mode: int = 3, requested_gain: float | None = None) -> Dict[str, Any]:
    import json

    metadata_payload = _read_hashed_artifact(dump_dir, manifest["artifacts"], "subtitle_conditioning.json")
    try:
        metadata = json.loads(metadata_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("subtitle_conditioning.json is malformed") from error
    if metadata != manifest.get("subtitle_conditioning"):
        raise ValueError("subtitle_conditioning.json disagrees with the manifest")
    if metadata.get("mode") != _SUBTITLE_MODE_CONTOURS:
        return {"mode": _SUBTITLE_MODE_NONE, "subtitle_evidence_verified": False}
    content = _subtitle_tensor_content(field_width, field_height, tensor_content)
    dynamic_roi = _subtitle_dynamic_roi(source_width, source_height, content)
    if dynamic_roi is None:
        raise ValueError("active UIP4 subtitle conditioning has unsupported source/field geometry")
    roi_top, roi_bottom = dynamic_roi
    ocr_payload = _read_hashed_artifact(dump_dir, manifest["artifacts"], "subtitle_ocr_record.u32")
    plane_payload = _read_hashed_artifact(dump_dir, manifest["artifacts"], "subtitle_locator_state.u32")
    plane_words = _uint32_words(plane_payload, SUBTITLE_PLANE_STATE_WORD_COUNT, "UIP4 plane state")
    evidence_frame_id = matched_frame_id
    publication_timestamp = manifest["matched_observation_timestamp_us"]
    matched_trace = gpu_trace.get("matched_record")
    held_with_depth = bool(matched_trace and
                           matched_trace["subtitle_disposition"]["name"] == "held-with-depth")
    if held_with_depth:
        evidence_frame_id = matched_trace["subtitle_plane"]["frame_id"]
        if list(plane_words) != matched_trace["subtitle_plane_words"]:
            raise ValueError("held UIP4 artifact differs from the authenticated trace tuple")
        publication_timestamp = _uint64_words(plane_words[8], plane_words[9])
        if publication_timestamp == 0 or publication_timestamp > manifest["matched_observation_timestamp_us"]:
            raise ValueError("held UIP4 source time must be positive and no newer than current source")
        origin = next((row for row in gpu_trace["records"] if
                       row["frame_id"] == evidence_frame_id and
                       row["subtitle_disposition"]["name"] in {"optional-ocr", "abstention"}), None)
        if origin is not None and (origin["subtitle_plane_words"] != list(plane_words) or
                                   origin["observation_timestamp_us"] != publication_timestamp):
            raise ValueError("held UIP4 tuple differs from its original source-time publication")
    ocr = validate_subtitle_ocr_record(
        ocr_payload, matched_frame_id=evidence_frame_id, analysis_generation=analysis_generation,
        source_width=source_width, source_height=source_height,
        field_width=field_width, field_height=field_height,
        roi_top=roi_top, roi_bottom=roi_bottom, tensor_content=content)
    plane = validate_subtitle_plane_state(
        plane_payload, ocr_record=ocr, matched_frame_id=evidence_frame_id,
        analysis_generation=analysis_generation, source_width=source_width, source_height=source_height,
        field_width=field_width, field_height=field_height, tensor_content=content,
        expected_scene_epoch=confirmed_cut_count,
        expected_observation_timestamp_us=publication_timestamp,
        joint_plane_mode=joint_plane_mode, requested_gain=requested_gain)
    state_timestamp = _uint64_words(shadow_state["gain_last_observation_low"],
                                   shadow_state["gain_last_observation_high"])
    geometry_ready = bool(
        shadow_state["frame_valid"] == 1.0 and
        shadow_state["renderer_authorization_bits"] == _CONTRACT_TAG and
        shadow_state["joint_plane_mode_bits"] == 3 and
        state_timestamp == publication_timestamp and publication_timestamp != 0 and
        shadow_state["confirmed_cut_count"] == plane["scene_epoch"])
    if plane["current_valid"] != bool(geometry_ready and ocr["authoritative"]):
        raise ValueError("UIP4 current-valid verdict differs from current OCR8/geometry authority")
    tail = plane_words[SUBTITLE_PLANE_ADAPTIVE_OFFSET:]
    if not geometry_ready and tail[0] and (tail[1] & 11 or any(tail[index] for index in range(4, 12))):
        raise ValueError("unavailable geometry must disarm adaptive UI clock and dwell")
    if (not ocr["authoritative"] or tail[15] != 0) and (tail[1] & 3 or any(tail[index] for index in range(6, 12))):
        raise ValueError("invalid OCR/probe must disarm adaptive UI dwell")
    active = bool(plane["current_valid"] and plane["current_count"] and tail[15] == 0)
    if matched_trace is not None:
        condition = validate_subtitle_condition_words(
            struct.pack("<6I", *matched_trace["subtitle_condition_words"]),
            plane_state=plane, geometry_ready=geometry_ready)
        active = condition[4] == coordinate_contract.SUBTITLE_OCR.condition_param_fully_active_value
    contour = validate_subtitle_contour_tuple(
        {role: _read_hashed_artifact(dump_dir, manifest["artifacts"], name)
         for role, name in _SUBTITLE_CONTOUR_ARTIFACT_FILES.items()},
        evidence_frame_id=evidence_frame_id, analysis_generation=analysis_generation,
        source_width=source_width, source_height=source_height,
        field_width=field_width, field_height=field_height, tensor_content=content,
        roi_top=roi_top, roi_bottom=roi_bottom, scene_epoch=plane["scene_epoch"],
        observation_timestamp_us=publication_timestamp, geometry_ready=geometry_ready,
        ocr_authoritative=ocr["authoritative"], current_rectangles=plane["current_rectangles"],
        plane_state=plane)
    return {
        "mode": _SUBTITLE_MODE_CONTOURS, "subtitle_evidence_verified": True,
        "geometry_ready": geometry_ready, "geometry_publication_timestamp_us": publication_timestamp,
        "evidence_frame_id": evidence_frame_id, "held_with_depth": held_with_depth,
        "ocr_authoritative": ocr["authoritative"], "ocr_raw_count": ocr["raw_count"],
        "ocr_ordinary_cover_horizontal_guard": ocr["ordinary_cover_horizontal_guard"],
        "ocr_line_centering": ocr["line_centering"],
        "ocr_final_count": ocr["final_count"], "source_width": source_width, "source_height": source_height,
        "field_width": field_width, "field_height": field_height, "tensor_content_rect": content,
        "roi_top": roi_top, "roi_bottom": roi_bottom,
        "current_valid": plane["current_valid"], "current_slot_mask": plane["current_slot_mask"],
        "current_count": plane["current_count"], "current_rectangles": plane["current_rectangles"],
        "current_ribbon_mask": plane["current_ribbon_mask"],
        "target": plane["target"], "target_bits": plane["target_bits"], "active_level": 2 if active else 0,
        "joint_plane_mode": joint_plane_mode, "display_budget": plane["display_budget"],
        "adaptive": plane["adaptive"], "warmup_packed": plane["warmup_packed"],
        "warmup_remaining": plane["warmup_remaining"], "contour": contour,
    }


def _subtitle_gaussian_joined_base(base_field: Any, subtitle: Dict[str, Any]) -> tuple[Any, tuple[int, int, int, int]]:
    import numpy as np

    base = np.asarray(base_field, dtype=np.float32)
    if base.ndim != 2 or not np.isfinite(base).all() or np.any(np.abs(base) > np.float32(_DEFAULTS.direct_container_limit)):
        raise ValueError("Gaussian Base must be a finite two-dimensional sealed display field")
    height, width = base.shape
    content = _subtitle_tensor_content(width, height, subtitle.get("tensor_content_rect"))
    cx = np.clip(np.arange(width), content[0], content[2] - 1)
    cy = np.clip(np.arange(height), content[1], content[3] - 1)
    return base[np.ix_(cy, cx)], content


def _subtitle_gaussian_target(subtitle: Dict[str, Any]) -> Any:
    import numpy as np

    target32 = np.float32(subtitle.get("target"))
    if (subtitle.get("joint_plane_mode") != 3 or subtitle.get("active_level") != 2 or
            not np.isfinite(target32) or
            not np.float32(0) <= target32 <= np.float32(_DEFAULTS.direct_container_limit)):
        raise ValueError("Gaussian proposal lacks authenticated fully active current plane")
    return target32


def _subtitle_gaussian_literal_blend(base: Any, weight: Any, target: Any) -> Any:
    import numpy as np

    base = np.asarray(base, dtype=np.float32)
    m = np.asarray(weight, dtype=np.float32)
    if m.shape != base.shape or not np.isfinite(m).all() or np.any((m < 0) | (m > 1)):
        raise ValueError("Gaussian weight must be finite unit-range and match Base")
    one_minus = np.subtract(np.float32(1), m, dtype=np.float32)
    base_term = np.multiply(one_minus, base, dtype=np.float32)
    ui_term = np.multiply(m, np.float32(target), dtype=np.float32)
    proposal = np.add(base_term, ui_term, dtype=np.float32)
    proposal[m == 0] = base[m == 0]
    proposal[m == 1] = np.float32(target)
    return proposal


def _subtitle_gaussian_blend_interval(base: Any, weight: Any, target: Any) -> tuple[Any, Any]:
    """SM5 representational enclosure for separate subtract/multiply/multiply/add.

    Each exact double operation on binary FP32 operands is enclosed by its adjacent FP32
    values, including signed-zero flush only for genuinely subnormal operation results.
    This never admits FMA or an alternate interpolation formula; endpoints are exact.
    """
    import numpy as np

    def floor32(value):
        exact = np.asarray(value, dtype=np.float64)
        rounded = exact.astype(np.float32)
        enclosed = np.where(rounded.astype(np.float64) > exact,
                            np.nextafter(rounded, np.float32(-np.inf)), rounded).astype(np.float32)
        return np.where((exact > 0) & (exact < np.finfo(np.float32).tiny),
                        np.float32(0), enclosed).astype(np.float32)

    def ceil32(value):
        exact = np.asarray(value, dtype=np.float64)
        rounded = exact.astype(np.float32)
        enclosed = np.where(rounded.astype(np.float64) < exact,
                            np.nextafter(rounded, np.float32(np.inf)), rounded).astype(np.float32)
        return np.where((exact < 0) & (exact > -np.finfo(np.float32).tiny),
                        np.float32(-0.0), enclosed).astype(np.float32)

    b = np.asarray(base, dtype=np.float32).astype(np.float64)
    m = np.asarray(weight, dtype=np.float32).astype(np.float64)
    om_lo, om_hi = floor32(1.0 - m).astype(np.float64), ceil32(1.0 - m).astype(np.float64)
    bt_lo = floor32(np.minimum(om_lo * b, om_hi * b)).astype(np.float64)
    bt_hi = ceil32(np.maximum(om_lo * b, om_hi * b)).astype(np.float64)
    ui_lo = floor32(m * float(np.float32(target))).astype(np.float64)
    ui_hi = ceil32(m * float(np.float32(target))).astype(np.float64)
    return floor32(bt_lo + ui_lo), ceil32(bt_hi + ui_hi)


def validate_subtitle_gaussian_fields(
        payloads: Dict[str, bytes], base_field: Any, subtitle: Dict[str, Any], *,
        final_field: Any = None) -> Dict[str, Any]:
    """Authenticate actual saved M/proposal/C75 and optional renderer Final; no stereo renderer.

    Call validate_subtitle_contour_tuple first for independent q/M/current-authority admission.
    The limiter is replayed from the saved literal proposal, never from a fabricated ideal M.
    """
    import numpy as np

    base, content = _subtitle_gaussian_joined_base(base_field, subtitle)
    if set(payloads) != set(_SUBTITLE_CONTOUR_ARTIFACT_FILES):
        raise ValueError("Gaussian field proof requires all eleven joined artifact roles")
    shape = base.shape
    leaves = {}
    for role in ("gaussian_weight", "gaussian_blend", "gaussian_vertical_conditioned"):
        payload = payloads[role]
        if len(payload) != base.size * 4:
            raise ValueError("Gaussian field leaf has wrong exact byte length")
        leaves[role] = np.frombuffer(payload, dtype="<f4").reshape(shape)
        if not np.isfinite(leaves[role]).all():
            raise ValueError("Gaussian field leaf contains nonfinite data")
    control = _uint32_words(payloads["gaussian_control"], 1, "Gaussian smoothing control")[0]
    if control not in (0, 1) or bool(control) != bool(subtitle["contour"]["gaussian_smoothing_enabled"]):
        raise ValueError("Gaussian field control does not bind the authenticated tuple verdict")
    m, proposal, vertical = (leaves[role] for role in ("gaussian_weight", "gaussian_blend", "gaussian_vertical_conditioned"))
    if np.any((m < 0) | (m > 1)):
        raise ValueError("Gaussian weight is not finite unit-range")
    if not control:
        if (np.any(m.view(np.uint32)) or not np.array_equal(proposal.view(np.uint32), base.view(np.uint32)) or
                not np.array_equal(vertical.view(np.uint32), base.view(np.uint32))):
            raise ValueError("inactive Gaussian M/proposal/C75 must publish exact Base bits without reconditioning")
        replayed_final = base
    else:
        target = _subtitle_gaussian_target(subtitle)
        if (np.any(proposal[m == 0].view(np.uint32) != base[m == 0].view(np.uint32)) or
                np.any(proposal[m == 1].view(np.uint32) != target.view(np.uint32))):
            raise ValueError("Gaussian literal blend zero/one endpoints are not bit-exact Base/target")
        low, high = _subtitle_gaussian_blend_interval(base, m, target)
        interior = (m > 0) & (m < 1)
        impossible_negative_zero = interior & (proposal == 0) & np.signbit(proposal) & ~np.signbit(base) & ~np.signbit(target)
        if (np.any(impossible_negative_zero) or
                np.any((proposal[interior] < low[interior]) | (proposal[interior] > high[interior]))):
            raise ValueError("Gaussian proposal differs from separate literal precise arithmetic beyond SM5 representational enclosure")
        _, replayed_vertical, replayed_final = _replay_v2_limiter_fields(proposal, content[2] - content[0])
        if not np.array_equal(vertical.view(np.uint32), replayed_vertical.view(np.uint32)):
            raise ValueError("Gaussian vertical-conditioned field is not exact shared C75/serial-Q30 replay of saved proposal")
    if final_field is not None:
        final = np.asarray(final_field, dtype=np.float32)
        if final.shape != shape or not np.array_equal(final.view(np.uint32), replayed_final.view(np.uint32)):
            raise ValueError("Gaussian Final is not exact shared horizontal Q30 replay or inactive Base copy")
    return {"gaussian_blend_verified": True, "gaussian_vertical_conditioned_verified": True,
            "gaussian_final_verified": final_field is not None, "gaussian_smoothing_enabled": bool(control),
            "blend_arithmetic_admission": "separate-fp32-operation-adjacent-representable-enclosure-bit-exact-endpoints",
            "post_conditioning": "shared-vertical75-25-horizontal-Q30-no-exact-repin"}


def _replay_subtitle_contour_conditioner(base_field: Any, subtitle: Dict[str, Any], distance_field: Any) -> Any:
    """Mathematical literal Gaussian proposal helper for authored fixtures, not actual exp words.

    Actual native validation consumes saved M/blend and validate_subtitle_gaussian_fields.
    """
    import numpy as np

    base, content = _subtitle_gaussian_joined_base(base_field, subtitle)
    q = np.asarray(distance_field, dtype=np.float32)
    if q.shape != base.shape:
        raise ValueError("Gaussian Base/q field shapes differ")
    if not subtitle["contour"]["authority_valid"] or not np.isfinite(q[content[1], content[0]]) or q[content[1], content[0]] < 0:
        return base.copy()
    x = np.clip(np.arange(base.shape[1]), content[0], content[2] - 1)
    y = np.clip(np.arange(base.shape[0]), content[1], content[3] - 1)
    weight = _subtitle_gaussian_weight_from_q(q[np.ix_(y, x)], subtitle["source_width"], subtitle["source_height"])
    return _subtitle_gaussian_literal_blend(base, weight, _subtitle_gaussian_target(subtitle))


def _verify_roi_exterior_zero_warp_map(
        dump_dir: Any, manifest: Dict[str, Any], input_region: Dict[str, Any],
        final_parallax: Any) -> Dict[str, Any]:
    """Prove exact-zero exterior samples from the selected full-source inverse map."""

    import hashlib
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

    dimensions = manifest["dimensions"]
    packed = dimensions.get("packed_sbs")
    eye = dimensions.get("eye")
    source = dimensions.get("source")
    warp_dimensions = dimensions.get("warp_map")
    shape = manifest.get("warp_map_contract")
    if not all(isinstance(value, dict) for value in (packed, eye, source, warp_dimensions)):
        raise ValueError("ROI warp-map evidence lacks full-source/output dimensions")
    map_width = _uint32(warp_dimensions.get("width"), "warp-map width")
    map_height = _uint32(warp_dimensions.get("height"), "warp-map height")
    eye_width = _uint32(eye.get("width"), "eye width")
    eye_height = _uint32(eye.get("height"), "eye height")
    source_width = input_region["source_width"]
    source_height = input_region["source_height"]
    if (not isinstance(shape, dict) or
            map_width == 0 or map_height == 0 or eye_width == 0 or eye_height == 0 or
            packed.get("width") != map_width or packed.get("height") != map_height or
            map_width != 2 * eye_width or map_height != eye_height or
            source.get("width") != source_width or source.get("height") != source_height or
            warp_dimensions.get("format") != "DXGI_FORMAT_R32_FLOAT" or
            warp_dimensions.get("format_value") != 41 or
            shape.get("schema") != 2 or shape.get("width") != map_width or
            shape.get("height") != map_height or shape.get("eye_width") != eye_width or
            shape.get("eye_height") != eye_height or
            shape.get("source_width") != source_width or
            shape.get("source_height") != source_height or
            shape.get("artifact") != "warp_map.f32"):
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

    mask_path = os.path.join(os.fspath(dump_dir), "warp_mask.png")
    try:
        with open(mask_path, "rb") as handle:
            mask_payload = handle.read()
    except OSError as error:
        raise ValueError("authoritative ROI warp_mask.png is missing") from error
    mask_descriptor = artifacts["warp_mask.png"]
    if hashlib.sha256(mask_payload).hexdigest() != mask_descriptor["sha256"]:
        raise ValueError("warp_mask.png content hash mismatch")

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
    # Continuous ROI projection has no ownership or synthetic-fill mask channels.
    import io
    from PIL import Image

    with Image.open(io.BytesIO(mask_payload)) as image:
        mask = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if mask.shape != (map_height, map_width, 3):
        raise ValueError("ROI boundary mask dimensions disagree")
    if np.any(mask[:, :, 1:] != 0) or np.any(~np.isin(mask[:, :, 0], (0, 255))):
        raise ValueError("continuous ROI boundary mask has noncanonical channels")

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


def _replay_v2_limiter_fields(candidate: Any, content_width: int) -> Tuple[Any, Any, Any]:
    """Replay the contract's serial/Q30 limiter branches exactly in NumPy."""

    import numpy as np

    values = np.asarray(candidate, dtype=np.float32)
    if values.ndim != 2 or values.size == 0 or content_width <= 0:
        raise ValueError("limiter replay requires a non-empty 2-D field and positive content width")
    height, width = values.shape
    vertical_step = np.float32(_DEFAULTS.max_vertical_shear / content_width)
    horizontal_step = np.float32(_DEFAULTS.max_horizontal_slope / content_width)
    share = np.float32(_DEFAULTS.vertical_majorant_share)
    share_complement = np.float32(1.0 - float(share))

    if height <= 32:
        majorant = values.copy()
        for row in range(1, height):
            majorant[row] = np.maximum(majorant[row], majorant[row - 1] - vertical_step)
        for row in range(height - 2, -1, -1):
            majorant[row] = np.maximum(majorant[row], majorant[row + 1] - vertical_step)
        minorant = values.copy()
        for row in range(1, height):
            minorant[row] = np.minimum(values[row], minorant[row - 1] + vertical_step)
        for row in range(height - 2, -1, -1):
            minorant[row] = np.minimum(minorant[row], minorant[row + 1] + vertical_step)
    else:
        q30_scale = np.float32(1073741824.0)
        limit = np.float32(_DEFAULTS.direct_container_limit)
        finite = np.where(np.isfinite(values), values, np.float32(0.0)).astype(np.float32)
        bounded = np.clip(finite, -limit, limit).astype(np.float32)
        upper_q30 = np.ceil(
            np.multiply(bounded, q30_scale, dtype=np.float32)
        ).astype(np.int64)
        lower_q30 = np.floor(
            np.multiply(bounded, q30_scale, dtype=np.float32)
        ).astype(np.int64)
        limit_q30 = int(np.ceil(np.float32(limit * q30_scale)))
        step_q30 = max(1, min(0x80000000 // content_width, 2 * limit_q30))

        majorant_q30 = upper_q30.copy()
        for row in range(1, height):
            majorant_q30[row] = np.maximum(
                upper_q30[row], majorant_q30[row - 1] - step_q30)
        for row in range(height - 2, -1, -1):
            majorant_q30[row] = np.maximum(
                majorant_q30[row], majorant_q30[row + 1] - step_q30)
        minorant_q30 = lower_q30.copy()
        for row in range(1, height):
            minorant_q30[row] = np.minimum(
                lower_q30[row], minorant_q30[row - 1] + step_q30)
        for row in range(height - 2, -1, -1):
            minorant_q30[row] = np.minimum(
                minorant_q30[row], minorant_q30[row + 1] + step_q30)
        majorant = np.divide(
            majorant_q30.astype(np.float32), q30_scale, dtype=np.float32)
        minorant = np.divide(
            minorant_q30.astype(np.float32), q30_scale, dtype=np.float32)

    conditioned = np.add(
        np.multiply(share, majorant, dtype=np.float32),
        np.multiply(share_complement, minorant, dtype=np.float32),
        dtype=np.float32,
    )
    conditioned = np.minimum(np.maximum(conditioned, minorant), majorant).astype(np.float32)

    if width <= 32:
        final = conditioned.copy()
        for column in range(1, width):
            final[:, column] = np.maximum(
                final[:, column], final[:, column - 1] - horizontal_step)
        for column in range(width - 2, -1, -1):
            final[:, column] = np.maximum(
                final[:, column], final[:, column + 1] - horizontal_step)
    else:
        q30_scale = np.float32(1073741824.0)
        limit = np.float32(_DEFAULTS.direct_container_limit)
        bounded = np.clip(conditioned, -limit, limit).astype(np.float32)
        candidate_q30 = np.ceil(
            np.multiply(bounded, q30_scale, dtype=np.float32)
        ).astype(np.int64)
        limit_q30 = int(np.ceil(np.float32(limit * q30_scale)))
        step_q30 = max(1, min(0x20000000 // content_width, 2 * limit_q30))
        final_q30 = candidate_q30.copy()
        for column in range(1, width):
            final_q30[:, column] = np.maximum(
                candidate_q30[:, column], final_q30[:, column - 1] - step_q30)
        for column in range(width - 2, -1, -1):
            final_q30[:, column] = np.maximum(
                final_q30[:, column], final_q30[:, column + 1] - step_q30)
        final = np.divide(
            final_q30.astype(np.float32), q30_scale, dtype=np.float32)

    return majorant, conditioned, final


def verify_v2_dump_geometry(dump_dir: Any) -> Dict[str, Any]:
    """Verify a dump directory's geometry evidence against its own manifest.

    Checks, fail-closed:
      1. every chain field's ``.f32`` bytes hash to the manifest descriptor's ``sha256``;
      2. every field matches the manifest geometry dimensions and is entirely finite;
      3. the conditioning chain is internally consistent with the contract's serial/Q30 branches:
         ``vertical_majorant``/``vertical_conditioned``/ordinary Base are bitwise equal to the
         recurrences recomputed from ``candidate``; the native Gaussian proposal and shared limiter chain
         exactly reproduce the atomic final field; and the atomic final artifact is the sole
         warp-input authority.

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

    def read_hashed_json_artifact(name: str) -> tuple[bytes, Dict[str, Any]]:
        path = os.path.join(os.fspath(dump_dir), name)
        try:
            with open(path, "rb") as handle:
                payload = handle.read()
            document = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{name} is missing or malformed") from error
        if not isinstance(document, dict):
            raise ValueError(f"{name} must contain a JSON object")
        descriptor = artifacts[name]
        if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
            raise ValueError(f"{name} content hash mismatch")
        return payload, document

    _, shadow_state_document = read_hashed_json_artifact("shadow_state.json")
    shadow_state = validate_shadow_state_document(shadow_state_document)
    _, shadow_stats_document = read_hashed_json_artifact("shadow_frame_stats.json")
    shadow_stats = validate_shadow_frame_stats_document(shadow_stats_document)
    if shadow_state_document.get("rendered_output_selected") is not True:
        raise ValueError("shadow_state.json is not selected renderer state")
    expected_frame_valid = shadow_frame_valid_from_statistics(
        shadow_state_document, shadow_stats)
    if (shadow_state["frame_valid"] > 0.5) != expected_frame_valid:
        raise ValueError("V2 shadow state disagrees with its exact-frame statistics")
    expected_shadow_summary = dict(shadow_state_document["decoded"])
    expected_shadow_summary["raw_coordinate_scale"] = (
        shadow_state_document["constants"]["raw_coordinate_scale"])
    expected_shadow_summary["rendered_output_selected"] = True
    summary = fragment["shadow_state_summary"]
    if any(
        (not _same_serialized_number(summary.get(key), value)
         if key in _DECODED_FLOAT_KEYS or key == "raw_coordinate_scale"
         else summary.get(key) != value)
        for key, value in expected_shadow_summary.items()
    ):
        raise ValueError("dump manifest V2 state summary disagrees with shadow_state.json")

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
    content_left, content_top, content_right, content_bottom = (
        input_region["tensor_content_rect"])
    expected_stats_texel_count = float(
        (content_right - content_left) * (content_bottom - content_top))
    if shadow_stats["texel_count"] != expected_stats_texel_count:
        raise ValueError("V2 shadow statistics disagree with the exact analysis content")
    gpu_trace = _verify_gpu_trace_artifacts(
        dump_dir,
        manifest,
        input_region=input_region,
        field_width=width,
        field_height=height,
    )

    window_region = None
    if fragment["window_region_available"]:
        region_path = os.path.join(os.fspath(dump_dir), "window_region.json")
        try:
            with open(region_path, "rb") as handle:
                region_payload = handle.read()
            region_document = json.loads(region_payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "advertised window_region.json is missing or malformed") from error
        region_descriptor = artifacts["window_region.json"]
        if "sha256" in region_descriptor:
            region_digest = hashlib.sha256(region_payload).hexdigest()
            if region_digest != region_descriptor["sha256"]:
                raise ValueError("window_region.json content hash mismatch")
        window_region = validate_window_region_document(
            region_document,
            matched_frame_id=frame_id,
            source_width=source_width,
            source_height=source_height,
        )
    if input_region["mode"] == "window-region":
        if window_region is None:
            raise ValueError("window-region dump has no authenticated provenance")
        if input_region["semantic_rect"] != (
                window_region["left"], window_region["top"],
                window_region["right"], window_region["bottom"]):
            raise ValueError("depth input semantic rectangle disagrees with window provenance")
        authorization = input_region["authorization"]
        if authorization != {
                "authority_kind": window_region["authority_kind"],
                "observer_generation": window_region["generation"],
                "hwnd": window_region["hwnd"],
                "hwnd_text": window_region["hwnd_text"],
                "process_id": window_region["process_id"],
                "document_id": window_region["document_id"],
                "video_id": window_region["video_id"]}:
            raise ValueError("depth input authorization disagrees with window provenance")

    if fragment["window_region_observer_status"] == "ok-fullscreen":
        if window_region is None or window_region["authority_kind"] != "chromium-video":
            raise ValueError("ok-fullscreen dump has no matched Chromium provenance")
        if (window_region["left"], window_region["top"],
                window_region["right"], window_region["bottom"]) != (
                0, 0, source_width, source_height):
            raise ValueError(
                "ok-fullscreen window region is not the exact full source")
        if input_region["mode"] != "full-source":
            raise ValueError(
                "ok-fullscreen dump is not bound to the full-source analysis domain")

    subtitle_summary = _verify_subtitle_conditioning_artifacts(
        dump_dir,
        manifest,
        matched_frame_id=frame_id,
        analysis_generation=input_region["analysis_generation"],
        source_width=input_region["inference_width"],
        source_height=input_region["inference_height"],
        field_width=width,
        field_height=height,
        tensor_content=input_region["tensor_content_rect"],
        confirmed_cut_count=shadow_state["confirmed_cut_count"],
        gpu_trace=gpu_trace,
        joint_plane_mode=shadow_state["joint_plane_mode_bits"],
        requested_gain=shadow_state_constant_float32(shadow_state_document, "requested_gain"),
        shadow_state=shadow_state,
    )
    subtitle_live = subtitle_summary["mode"] == _SUBTITLE_MODE_CONTOURS
    geometry_chain_fields = (
        "shadow_coordinate",
        "shadow_candidate_parallax",
        "shadow_vertical_majorant",
        "shadow_vertical_conditioned",
        *(('shadow_base_final_parallax',) if subtitle_live else ()),
        "shadow_final_parallax",
    )
    fields: Dict[str, Any] = {}
    for name in geometry_chain_fields:
        payload = _read_hashed_artifact(
            dump_dir, artifacts, name + ".f32")
        values = np.frombuffer(payload, dtype="<f4")
        if values.size != width * height:
            raise ValueError(f"{name}.f32 has {values.size} texels; expected {width * height}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name}.f32 contains non-finite values")
        fields[name] = values.reshape(height, width)
    candidate = fields["shadow_candidate_parallax"]
    if shadow_state["frame_valid"] == 0.0 and (
            np.any(fields["shadow_coordinate"] != np.float32(0.0)) or
            np.any(candidate != np.float32(0.0))):
        raise ValueError("unavailable geometry must publish exact zero coordinate and candidate")

    # Exact replicas of the production serial/Q30 branches (validated bitwise against the GPU
    # intermediates; see docs/host-sbs.md#cliff-conditioning).
    content_left, content_top, content_right, content_bottom = (
        input_region["tensor_content_rect"])
    content_width = content_right - content_left
    majorant, conditioned, final = _replay_v2_limiter_fields(candidate, content_width)

    recurrence_fields = [
        ("shadow_vertical_majorant", majorant),
        ("shadow_vertical_conditioned", conditioned),
    ]
    recurrence_fields.append((
        "shadow_base_final_parallax" if subtitle_live else
        "shadow_final_parallax",
        final,
    ))
    for name, recomputed in recurrence_fields:
        if not np.array_equal(fields[name], recomputed):
            mismatch = float(np.max(np.abs(fields[name] - recomputed)))
            raise ValueError(
                f"{name}.f32 is not the exact recurrence/serial-Q30 limiter replay of the dumped candidate field "
                f"(max abs diff {mismatch})")

    if subtitle_live:
        gaussian = validate_subtitle_gaussian_fields(
            {role: _read_hashed_artifact(dump_dir, artifacts, name)
             for role, name in _SUBTITLE_CONTOUR_ARTIFACT_FILES.items()},
            fields["shadow_base_final_parallax"], subtitle_summary,
            final_field=fields["shadow_final_parallax"])
        subtitle_summary["gaussian"] = gaussian

    exterior_zero = None
    if input_region["mode"] == "window-region":
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
        "chain_fields_verified": list(geometry_chain_fields),
        "depth_input_region_verified": True,
        "depth_input_region": input_region,
        "roi_exterior_zero_evidence": exterior_zero,
        "window_region_verified": window_region is not None,
        "window_region": window_region,
        "subtitle_conditioning": subtitle_summary,
        "gpu_trace": gpu_trace,
        "final_parallax_verified": True,
    }


def generate_float_artifact_previews(
        dump_dir: Any, artifact_name: str, output_dir: Any) -> Dict[str, Any]:
    """Generate diagnostic PNGs from one authenticated schema-46 ``.f32`` artifact.

    Preview files are deliberately written outside the atomic package contract. Scalar tensors
    receive finite-p2/p98 grayscale and jet views; ``model_input.f32`` receives an RGB view by
    reversing its authenticated ImageNet normalization without applying another sRGB transfer.
    """

    import hashlib
    import json
    import os
    from pathlib import Path

    import numpy as np
    from PIL import Image

    if (not isinstance(artifact_name, str) or not artifact_name.endswith(".f32") or
            Path(artifact_name).name != artifact_name):
        raise ValueError("preview artifact must be one package-local .f32 filename")
    root = Path(os.fspath(dump_dir))
    destination = Path(os.fspath(output_dir))
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if (resolved_destination == resolved_root or
            resolved_root in resolved_destination.parents):
        raise ValueError("preview output directory must be outside the atomic dump package")
    try:
        manifest = json.loads((root / "dump_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("dump_manifest.json is missing or malformed") from error
    validate_v2_dump_manifest_document(manifest)
    # A preview is presented as derived from an authenticated package, so validate the entire
    # required geometry/state transaction before creating any output, not just the requested file.
    verify_v2_dump_geometry(root)
    descriptor = manifest["artifacts"].get(artifact_name)
    if (not isinstance(descriptor, dict) or descriptor.get("available") is not True or
            not _is_sha256_hex(descriptor.get("sha256"))):
        raise ValueError("requested float artifact is not authenticated by this dump")
    try:
        payload = (root / artifact_name).read_bytes()
    except OSError as error:
        raise ValueError(f"{artifact_name} is missing") from error
    if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"{artifact_name} content hash mismatch")

    stem = artifact_name[:-4]
    extent = manifest["dimensions"].get(stem)
    if not isinstance(extent, dict):
        raise ValueError(f"dump manifest has no dimensions for {artifact_name}")
    width = _uint32(extent.get("width"), f"{artifact_name} width")
    height = _uint32(extent.get("height"), f"{artifact_name} height")
    if width == 0 or height == 0:
        raise ValueError(f"{artifact_name} has empty dimensions")
    values = np.frombuffer(payload, dtype="<f4")

    if artifact_name == "model_input.f32":
        if (extent.get("channels") != 3 or extent.get("layout") != "NCHW" or
                values.size != width * height * 3):
            raise ValueError("model_input.f32 dimensions/layout disagree")
        shape_descriptor = _required_hashed_artifact(
            manifest["artifacts"], "model_input_shape.json", "model-input shape")
        if shape_descriptor.get("stage") != "model-input contract":
            raise ValueError(
                "dump_manifest.json has an invalid model-input shape artifact descriptor")
        try:
            shape_payload = (root / "model_input_shape.json").read_bytes()
            if hashlib.sha256(shape_payload).hexdigest() != shape_descriptor["sha256"]:
                raise ValueError("model_input_shape.json content hash mismatch")
            shape = json.loads(shape_payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("model_input_shape.json is missing or malformed") from error
        means = np.asarray(shape.get("imagenet_mean"), dtype=np.float32)
        stds = np.asarray(shape.get("imagenet_std"), dtype=np.float32)
        if (shape.get("width") != width or shape.get("height") != height or
                shape.get("dtype") != "float32-le" or shape.get("layout") != "NCHW" or
                shape.get("channels") != ["R", "G", "B"] or
                means.shape != (3,) or stds.shape != (3,) or
                not np.isfinite(means).all() or not np.isfinite(stds).all() or
                not np.all(stds > 0.0)):
            raise ValueError("model_input_shape.json has invalid ImageNet normalization")
        planes = values.reshape(3, height, width)
        rgb = np.moveaxis(planes * stds[:, None, None] + means[:, None, None], 0, 2)
        nonfinite = ~np.isfinite(rgb).all(axis=2)
        encoded = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        encoded[nonfinite] = np.asarray([255, 0, 255], dtype=np.uint8)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "model_input.png"
        Image.fromarray(encoded, mode="RGB").save(path)
        return {
            "artifact": artifact_name,
            "width": width,
            "height": height,
            "files": [str(path)],
            "normalization": "channel * imagenet_std + imagenet_mean; clamped to [0,1]",
        }

    if values.size != width * height:
        raise ValueError(f"{artifact_name} byte size disagrees with manifest dimensions")
    field = values.reshape(height, width)
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        raise ValueError(f"{artifact_name} has no finite values")
    low, high = np.percentile(finite, (2.0, 98.0)).astype(np.float32)
    if not float(high) > float(low):
        low = np.float32(np.min(finite))
        high = np.float32(np.max(finite))
    if float(high) > float(low):
        normalized = np.clip((field - low) / (high - low), 0.0, 1.0)
    else:
        normalized = np.full(field.shape, np.float32(0.5), dtype=np.float32)
    invalid = ~np.isfinite(field)
    preview_values = np.where(invalid, np.float32(0.0), normalized)
    gray_values = np.rint(preview_values * 255.0).astype(np.uint8)
    gray = np.repeat(gray_values[:, :, None], 3, axis=2)
    jet = np.empty_like(gray)
    jet[:, :, 0] = np.rint(
        np.clip(1.5 - np.abs(4.0 * preview_values - 3.0), 0.0, 1.0) * 255.0)
    jet[:, :, 1] = np.rint(
        np.clip(1.5 - np.abs(4.0 * preview_values - 2.0), 0.0, 1.0) * 255.0)
    jet[:, :, 2] = np.rint(
        np.clip(1.5 - np.abs(4.0 * preview_values - 1.0), 0.0, 1.0) * 255.0)
    gray[invalid] = jet[invalid] = np.asarray([255, 0, 255], dtype=np.uint8)
    destination.mkdir(parents=True, exist_ok=True)
    gray_path = destination / f"{stem}.png"
    heat_path = destination / f"{stem}_heat.png"
    Image.fromarray(gray, mode="RGB").save(gray_path)
    Image.fromarray(jet, mode="RGB").save(heat_path)
    return {
        "artifact": artifact_name,
        "width": width,
        "height": height,
        "files": [str(gray_path), str(heat_path)],
        "normalization": "finite p2-p98",
        "preview_low": float(low),
        "preview_high": float(high),
    }


__all__ = [
    "DEPTH_INPUT_REGION_SCHEMA",
    "DIAGNOSTIC_SOURCE_CLOSURE_SHA256",
    "DUMP_MANIFEST_SCHEMA",
    "GPU_TRACE_CAPACITY",
    "GPU_TRACE_CONDITION_WORD_COUNT",
    "GPU_TRACE_CONTRACT_SCHEMA",
    "GPU_TRACE_DECODED_SCHEMA",
    "GPU_TRACE_HEADER_OFFSETS",
    "GPU_TRACE_HEADER_WORD_COUNT",
    "GPU_TRACE_RECORD_OFFSETS",
    "GPU_TRACE_RECORD_TAG",
    "GPU_TRACE_RECORD_WORD_COUNT",
    "GPU_TRACE_RING_BYTE_COUNT",
    "GPU_TRACE_RING_SCHEMA",
    "GPU_TRACE_RING_TAG",
    "GPU_TRACE_RING_WORD_COUNT",
    "GPU_TRACE_SOURCE_CLOSURE_SHA256",
    "GPU_TRACE_TRANSACTION_WORD_COUNT",
    "LIVE_RENDERER_SOURCE_CLOSURE_SHA256",
    "SHADOW_FRAME_STATS_DUMP_SCHEMA",
    "SHADOW_STATE_DUMP_SCHEMA",
    "SUBTITLE_PLANE_STATE_SCHEMA",
    "SUBTITLE_PLANE_STATE_TAG",
    "SUBTITLE_OCR_BOX_WORD_COUNT",
    "SUBTITLE_OCR_FINAL_BOX_CAPACITY",
    "SUBTITLE_OCR_FINAL_BOX_WORD_OFFSET",
    "SUBTITLE_OCR_HEADER_WORD_COUNT",
    "SUBTITLE_OCR_RAW_BOX_CAPACITY",
    "SUBTITLE_OCR_RAW_BOX_WORD_OFFSET",
    "SUBTITLE_OCR_RECORD_SCHEMA",
    "SUBTITLE_OCR_RECORD_TAG",
    "SUBTITLE_OCR_RECORD_WORD_COUNT",
    "WINDOW_REGION_AUTHORITY_KINDS",
    "WINDOW_REGION_SCHEMA",
    "camera_center_integrity_bits",
    "camera_center_integrity_for_state_values",
    "adaptation_semantics_for_mode",
    "display_budget_for_mode",
    "generate_float_artifact_previews",
    "shadow_frame_valid_from_statistics",
    "shadow_state_constant_float32",
    "fit_host_sbs_v2_depth_tensor_shape",
    "expected_capture_grid_for_source",
    "capture_grid_relation_for_source",
    "validate_depth_input_region_document",
    "validate_composite_runtime_provenance",
    "validate_gpu_trace_contract_document",
    "validate_gpu_trace_ring",
    "validate_subtitle_contour_tuple",
    "validate_subtitle_gaussian_fields",
    "SUBTITLE_PLANE_STATE_SCHEMA",
    "SUBTITLE_PLANE_STATE_TAG",
    "SUBTITLE_PLANE_STATE_WORD_COUNT",
    "SUBTITLE_PLANE_COVER_CAPACITY",
    "SUBTITLE_PLANE_CURRENT_VALID_FLAG",
    "SUBTITLE_PLANE_ADAPTIVE_OFFSET",
    "validate_subtitle_plane_snapshot",
    "validate_subtitle_plane_state",
    "validate_subtitle_condition_words",
    "validate_subtitle_ocr_record",
    "validate_window_region_document",
    "validate_v2_dump_manifest_document",
    "validate_shadow_frame_stats_document",
    "validate_shadow_state_document",
    "validate_parallax_state_words",
    "verify_v2_dump_geometry",
]
