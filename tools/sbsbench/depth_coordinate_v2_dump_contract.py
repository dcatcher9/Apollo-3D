#!/usr/bin/env python3
"""Strict readers for the diagnostic depth-coordinate-v2 JSON dump files.

The JSON serialization schemas are deliberately independent of the GPU coordinate-contract
schema.  Each document also binds the exact generated contract schema/tag and complete physical
field layout, so a consumer cannot silently reinterpret a same-sized state buffer.
"""

from __future__ import annotations

import math
import struct
from itertools import product
from typing import Any, Dict, Tuple

try:
    from . import depth_coordinate_v2_contract as coordinate_contract
    from . import generate_depth_coordinate_v2_contract as generator
except ImportError:  # Direct script/module loading from tools/sbsbench.
    import depth_coordinate_v2_contract as coordinate_contract  # type: ignore
    import generate_depth_coordinate_v2_contract as generator  # type: ignore


DUMP_MANIFEST_SCHEMA = 32
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
SUBTITLE_LOCATOR_STATE_SCHEMA = coordinate_contract.SUBTITLE_OCR.locator_schema
SUBTITLE_LOCATOR_STATE_TAG = coordinate_contract.SUBTITLE_OCR.locator_tag
SUBTITLE_LOCATOR_STATE_WORD_COUNT = coordinate_contract.SUBTITLE_OCR.locator_word_count
SUBTITLE_LOCATOR_HEADER_WORD_COUNT = (
    coordinate_contract.SUBTITLE_OCR.locator_header_word_count)
SUBTITLE_LOCATOR_RECT_CAPACITY = (
    coordinate_contract.SUBTITLE_OCR.locator_rectangle_capacity)
SUBTITLE_LOCATOR_OWNER_WORD_OFFSET = coordinate_contract.SUBTITLE_OCR.locator_owner_offset
SUBTITLE_LOCATOR_PENDING_WORD_OFFSET = coordinate_contract.SUBTITLE_OCR.locator_pending_offset
SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET = coordinate_contract.SUBTITLE_OCR.locator_current_offset
SUBTITLE_LOCATOR_KIND_WORD = coordinate_contract.SUBTITLE_OCR.locator_kind_word
SUBTITLE_LOCATOR_OWNER_KIND_SHIFT = (
    coordinate_contract.SUBTITLE_OCR.locator_owner_kind_shift)
SUBTITLE_LOCATOR_PENDING_KIND_SHIFT = (
    coordinate_contract.SUBTITLE_OCR.locator_pending_kind_shift)
SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT = (
    coordinate_contract.SUBTITLE_OCR.locator_current_kind_shift)
SUBTITLE_LOCATOR_KIND_MASK = coordinate_contract.SUBTITLE_OCR.locator_kind_mask
SUBTITLE_LOCATOR_FLAG_OWNER = 1 << 0
SUBTITLE_LOCATOR_FLAG_PENDING = 1 << 1
SUBTITLE_LOCATOR_FLAG_TARGET_VALID = 1 << 2
SUBTITLE_LOCATOR_FLAG_TARGET_RESET = 1 << 3
SUBTITLE_LOCATOR_KNOWN_FLAGS = (
    SUBTITLE_LOCATOR_FLAG_OWNER |
    SUBTITLE_LOCATOR_FLAG_PENDING |
    SUBTITLE_LOCATOR_FLAG_TARGET_VALID |
    SUBTITLE_LOCATOR_FLAG_TARGET_RESET
)
SUBTITLE_LOCATOR_EVENT_NONE = 0
SUBTITLE_LOCATOR_EVENT_BIRTH = 1
SUBTITLE_LOCATOR_EVENT_DEATH = 2
SUBTITLE_LOCATOR_EVENT_HANDOFF = 3
DEPTH_INPUT_REGION_SCHEMA = 3
WINDOW_REGION_SCHEMA = 1
WINDOW_REGION_AUTHORITY_KINDS = frozenset({"chromium-video", "foreground-client"})
SHADOW_STATE_DUMP_SCHEMA = 16
SHADOW_FRAME_STATS_DUMP_SCHEMA = 2
LIVE_RENDERER_SOURCE_CLOSURE_SHA256 = (
    "41cd0bf450afa1cd3585b1945e006a003d11bae02c88c04e5d5b32d1a69e0f42"
)
DIAGNOSTIC_SOURCE_CLOSURE_SHA256 = (
    "0f83d7a1a7f30c2ba9eee01f0fd6c4c7780478b46d4a4e435788e3a4d91e54c1"
)
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
_AUTHENTICATED_TENSOR_SHAPES = frozenset(
    _PRODUCTION_CALIBRATION.calibrated_input_shapes)
_MAXIMUM_SOURCE_LONG_SIDE = 5120
_MAXIMUM_SOURCE_PIXELS = 5120 * 2160

_SUBTITLE_MODE_NONE = "none"
_SUBTITLE_MODE_SLR12 = "subtitle-slr12"
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
_SUBTITLE_LOCATOR_SHADER_SPECS = (
    ("host_sbs_subtitle_locator_cs.hlsl", "resolve_main", "cs_5_0"),
    ("host_sbs_subtitle_locator_cs.hlsl", "condition_prepare_main", "cs_5_0"),
    ("host_sbs_subtitle_locator_cs.hlsl", "condition_main", "cs_5_0"),
)


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
    """Apply the native frame-resolve validity predicate to validated dump evidence."""

    collapse_epsilon = shadow_state_constant_float32(
        state_document, "collapse_abs_epsilon")
    return bool(
        statistics["valid"] > 0.5 and
        statistics["population_std"] > collapse_epsilon)


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
    """Validate and decode the sole current OCR8 208-word lower-text record."""

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
    }


def _decode_subtitle_locator_rectangles(
        words: tuple[int, ...], *, offset: int, count: int,
        field_width: int, field_height: int, roi_top: int, roi_bottom: int,
        tensor_content: tuple[int, int, int, int], ribbon_min_bottom: int,
        ribbon_mask: int, current_cover: bool,
        label: str) -> list[Dict[str, Any]]:
    decoded: list[Dict[str, Any]] = []
    for slot in range(SUBTITLE_LOCATOR_RECT_CAPACITY):
        start = offset + slot * 4
        left, top, right, bottom = words[start:start + 4]
        if slot >= count:
            if left != 0 or top != 0 or right != 0 or bottom != 0:
                raise ValueError(f"{label} unused rectangle slots must be canonical zero")
            continue
        ribbon = bool(ribbon_mask & (1 << slot))
        content_left, content_top, content_right, content_bottom = tensor_content
        if (left >= right or top >= bottom or left < content_left or
                top < content_top or right > content_right or
                bottom > content_bottom or top < roi_top):
            raise ValueError(
                f"{label} rectangles must be nonempty half-open field coordinates")
        if current_cover and ribbon:
            if (left != content_left or right != content_right or
                    bottom != content_bottom):
                raise ValueError(f"{label} ribbon cover is not the canonical bottom strip")
        elif ribbon and bottom < ribbon_min_bottom:
            raise ValueError(f"{label} ribbon core is outside its projected bottom tolerance")
        elif bottom > roi_bottom:
            raise ValueError(f"{label} rectangles must remain inside the OCR ROI")
        decoded.append({
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "kind": "ribbon" if ribbon else "text",
            "ribbon": ribbon,
        })
    return decoded


def _subtitle_rectangle_aggregate(
        rectangles: list[Dict[str, int]]) -> tuple[tuple[int, int, int, int], int]:
    if not rectangles:
        return (0, 0, 0, 0), 0
    bbox = (
        min(rectangle["left"] for rectangle in rectangles),
        min(rectangle["top"] for rectangle in rectangles),
        max(rectangle["right"] for rectangle in rectangles),
        max(rectangle["bottom"] for rectangle in rectangles),
    )
    area = sum(
        (rectangle["right"] - rectangle["left"]) *
        (rectangle["bottom"] - rectangle["top"])
        for rectangle in rectangles)
    return bbox, area


def _subtitle_coherent_lines(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Mirror the frozen SLR12 vertically-adjacent ordinary-line predicate."""

    width_a = a["right"] - a["left"]
    width_b = b["right"] - b["left"]
    height_a = a["bottom"] - a["top"]
    height_b = b["bottom"] - b["top"]
    overlap = max(0, min(a["right"], b["right"]) - max(a["left"], b["left"]))
    center_x_delta_twice = abs(
        (a["left"] + a["right"]) - (b["left"] + b["right"]))
    left_delta = abs(a["left"] - b["left"])
    right_delta = abs(a["right"] - b["right"])
    center_y_delta_twice = abs(
        (a["top"] + a["bottom"]) - (b["top"] + b["bottom"]))
    gap = 0
    if a["bottom"] <= b["top"]:
        gap = b["top"] - a["bottom"]
    elif b["bottom"] <= a["top"]:
        gap = a["top"] - b["bottom"]
    shorter_height = min(height_a, height_b)
    taller_height = max(height_a, height_b)
    return (
        overlap * 2 >= min(width_a, width_b) and
        (center_x_delta_twice <= taller_height or
         left_delta <= taller_height or right_delta <= taller_height) and
        taller_height <= 2 * shorter_height and
        center_y_delta_twice >= shorter_height and
        gap * 2 <= taller_height
    )


def _subtitle_same_baseline_segments(
        a: Dict[str, Any], b: Dict[str, Any], field_width: int) -> bool:
    """Mirror the frozen SLR12 horizontally-disjoint same-baseline predicate."""

    if not (a["right"] <= b["left"] or b["right"] <= a["left"]):
        return False
    height_a = a["bottom"] - a["top"]
    height_b = b["bottom"] - b["top"]
    shorter_height = min(height_a, height_b)
    taller_height = max(height_a, height_b)
    vertical_overlap = max(
        0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]))
    center_y_delta_twice = abs(
        (a["top"] + a["bottom"]) - (b["top"] + b["bottom"]))
    horizontal_gap = (
        b["left"] - a["right"] if a["right"] <= b["left"]
        else a["left"] - b["right"])
    combined_span = max(a["right"], b["right"]) - min(a["left"], b["left"])
    policy = coordinate_contract.SUBTITLE_OCR
    return (
        vertical_overlap * 4 >= 3 * shorter_height and
        taller_height <= 2 * shorter_height and
        center_y_delta_twice <= shorter_height and
        horizontal_gap <= 8 * taller_height and
        combined_span * policy.locator_max_width_denominator <=
        field_width * policy.locator_max_width_numerator
    )


def _subtitle_qualified_ocr_core(
        rectangle: Dict[str, Any], field_width: int) -> bool:
    """Mirror SLR12's generic core geometry gate after OCR8 validation."""

    width = rectangle["right"] - rectangle["left"]
    height = rectangle["bottom"] - rectangle["top"]
    policy = coordinate_contract.SUBTITLE_OCR
    if (width < policy.locator_min_width_cells or
            height < policy.locator_min_height_cells or
            width * policy.locator_min_aspect_denominator <
            policy.locator_min_aspect_numerator * height):
        return False
    if rectangle["kind"] == "ribbon":
        return True
    return (width * policy.locator_max_width_denominator <=
            field_width * policy.locator_max_width_numerator)


def _subtitle_selected_ocr_indices(
        raw_boxes: list[Dict[str, Any]], field_width: int) -> list[int]:
    """Replay frozen SLR12 component closure/selection on same-frame OCR8 cores."""

    qualified = [
        index for index, rectangle in enumerate(raw_boxes)
        if _subtitle_qualified_ocr_core(rectangle, field_width)
    ]
    ordinary = [index for index in qualified if raw_boxes[index]["kind"] == "text"]
    ribbon = [index for index in qualified if raw_boxes[index]["kind"] == "ribbon"]

    unseen = set(ordinary)
    components: list[list[int]] = []
    while unseen:
        root = min(unseen)
        reached = {root}
        frontier = [root]
        unseen.remove(root)
        while frontier:
            left = frontier.pop()
            for right in tuple(unseen):
                if (_subtitle_coherent_lines(raw_boxes[left], raw_boxes[right]) or
                        _subtitle_same_baseline_segments(
                            raw_boxes[left], raw_boxes[right], field_width)):
                    reached.add(right)
                    frontier.append(right)
                    unseen.remove(right)
        components.append(sorted(reached))

    best: list[int] = []
    best_key: tuple[int, int, int, int] | None = None
    for component in components:
        if len(component) > SUBTITLE_LOCATOR_RECT_CAPACITY:
            continue
        rectangles = [raw_boxes[index] for index in component]
        bbox, area = _subtitle_rectangle_aggregate(rectangles)
        policy = coordinate_contract.SUBTITLE_OCR
        if ((bbox[2] - bbox[0]) * policy.locator_max_width_denominator >
                field_width * policy.locator_max_width_numerator):
            continue
        # Selection maximizes summed core area. Ties prefer lower bottom, then lower top, then
        # smaller left exactly as BuildCurrentStack's deterministic comparison does.
        key = (area, bbox[3], bbox[1], -bbox[0])
        if best_key is None or key > best_key:
            best = component
            best_key = key

    selected = sorted(
        best + ribbon,
        key=lambda index: (
            raw_boxes[index]["top"], raw_boxes[index]["left"], index))
    if len(selected) > SUBTITLE_LOCATOR_RECT_CAPACITY:
        return []
    return selected


def _subtitle_rectangles_are_top_left_ordered(
        rectangles: list[Dict[str, Any]]) -> bool:
    keys = [
        (rectangle["top"], rectangle["left"])
        for rectangle in rectangles
    ]
    return keys == sorted(keys)


def validate_subtitle_locator_state(
        payload: Any, *, matched_frame_id: int, analysis_generation: int,
        source_width: int, source_height: int,
        field_width: int, field_height: int,
        tensor_content: tuple[int, int, int, int] | None = None,
        expected_scene_epoch: int | None = None) -> Dict[str, Any]:
    """Validate and decode the sole current compact SLR12 80-word state."""

    expected_frame = _uint64(matched_frame_id, "SLR12 matched frame id")
    expected_generation = _uint64(analysis_generation, "SLR12 analysis generation")
    expected_source_width = _uint32(source_width, "SLR12 source width")
    expected_source_height = _uint32(source_height, "SLR12 source height")
    expected_field_width = _uint32(field_width, "SLR12 field width")
    expected_field_height = _uint32(field_height, "SLR12 field height")
    authenticated_scene_epoch = (
        None if expected_scene_epoch is None else
        _uint32(expected_scene_epoch, "SLR12 expected scene epoch")
    )
    expected_content = _subtitle_tensor_content(
        expected_field_width, expected_field_height, tensor_content)
    if not coordinate_contract.subtitle_ocr_field_is_calibrated(
            expected_field_width, expected_field_height):
        raise ValueError("SLR12 expected field geometry is invalid")
    dynamic_roi = _subtitle_dynamic_roi(
        expected_source_width, expected_source_height, expected_content)
    if dynamic_roi is None:
        raise ValueError("SLR12 expected field geometry is invalid")
    expected_roi_top, expected_roi_bottom = dynamic_roi
    ribbon_min_bottom = _subtitle_ribbon_min_bottom(
        expected_source_width, expected_source_height, expected_content)
    if ribbon_min_bottom is None or not expected_roi_top < ribbon_min_bottom <= expected_roi_bottom:
        raise ValueError("SLR12 projected ribbon bottom tolerance is invalid")

    words = _uint32_words(payload, SUBTITLE_LOCATOR_STATE_WORD_COUNT, "SLR12 state")
    if words[0] != SUBTITLE_LOCATOR_STATE_SCHEMA or words[1] != SUBTITLE_LOCATOR_STATE_TAG:
        raise ValueError("SLR12 state has an unknown schema or tag")
    flags = words[2]
    if flags & ~SUBTITLE_LOCATOR_KNOWN_FLAGS:
        raise ValueError("SLR12 state has unknown flags")
    owner_count = words[4]
    pending_count = words[12]
    current_count = words[20]
    if any(count > SUBTITLE_LOCATOR_RECT_CAPACITY for count in (
            owner_count, pending_count, current_count)):
        raise ValueError("SLR12 state exceeds its fixed rectangle capacity")
    fade = words[24]
    if fade > 2:
        raise ValueError("SLR12 fade step must be zero, one, or two")
    if words[21] not in {
            SUBTITLE_LOCATOR_EVENT_NONE,
            SUBTITLE_LOCATOR_EVENT_BIRTH,
            SUBTITLE_LOCATOR_EVENT_DEATH,
            SUBTITLE_LOCATOR_EVENT_HANDOFF}:
        raise ValueError("SLR12 state has an unknown last event")
    packed_kinds = words[SUBTITLE_LOCATOR_KIND_WORD]
    known_kind_bits = (
        (SUBTITLE_LOCATOR_KIND_MASK << SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) |
        (SUBTITLE_LOCATOR_KIND_MASK << SUBTITLE_LOCATOR_PENDING_KIND_SHIFT) |
        (SUBTITLE_LOCATOR_KIND_MASK << SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT)
    )
    if packed_kinds & ~known_kind_bits:
        raise ValueError("SLR12 state has unknown packed-kind bits")

    def kind_mask(shift: int, count: int, label: str) -> int:
        value = (packed_kinds >> shift) & SUBTITLE_LOCATOR_KIND_MASK
        used = (1 << count) - 1 if count else 0
        if value & ~used:
            raise ValueError(f"SLR12 {label} kind mask exceeds its rectangle count")
        return value

    owner_kinds = kind_mask(SUBTITLE_LOCATOR_OWNER_KIND_SHIFT, owner_count, "owner")
    pending_kinds = kind_mask(SUBTITLE_LOCATOR_PENDING_KIND_SHIFT, pending_count, "pending")
    current_kinds = kind_mask(SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT, current_count, "current")

    generation = _uint64_words(words[10], words[11])
    frame_id = _uint64_words(words[22], words[23])
    if (generation != expected_generation or frame_id != expected_frame or
            words[27] != expected_field_width or words[28] != expected_field_height):
        raise ValueError("SLR12 state identity disagrees with the matched dump frame")
    if authenticated_scene_epoch is not None and words[26] != authenticated_scene_epoch:
        raise ValueError(
            "SLR12 state scene epoch disagrees with authenticated cut generation")

    owner = _decode_subtitle_locator_rectangles(
        words, offset=SUBTITLE_LOCATOR_OWNER_WORD_OFFSET, count=owner_count,
        field_width=expected_field_width, field_height=expected_field_height,
        roi_top=expected_roi_top, roi_bottom=expected_roi_bottom,
        tensor_content=expected_content, ribbon_min_bottom=ribbon_min_bottom,
        ribbon_mask=owner_kinds, current_cover=False, label="SLR12 owner")
    pending = _decode_subtitle_locator_rectangles(
        words, offset=SUBTITLE_LOCATOR_PENDING_WORD_OFFSET, count=pending_count,
        field_width=expected_field_width, field_height=expected_field_height,
        roi_top=expected_roi_top, roi_bottom=expected_roi_bottom,
        tensor_content=expected_content, ribbon_min_bottom=ribbon_min_bottom,
        ribbon_mask=pending_kinds, current_cover=False, label="SLR12 pending")
    current = _decode_subtitle_locator_rectangles(
        words, offset=SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET, count=current_count,
        field_width=expected_field_width, field_height=expected_field_height,
        roi_top=expected_roi_top, roi_bottom=expected_roi_bottom,
        tensor_content=expected_content, ribbon_min_bottom=ribbon_min_bottom,
        ribbon_mask=current_kinds, current_cover=True,
        label="SLR12 current-authority")
    # Owner and pending store cores and therefore expose the producer's canonical core order.
    # Current stores paired covers; their expansion can change cover-top order, so its canonical
    # order is checked later against the selected OCR core order rather than cover coordinates.
    for label, rectangles in (("owner", owner), ("pending", pending)):
        if not _subtitle_rectangles_are_top_left_ordered(rectangles):
            raise ValueError(f"SLR12 {label} rectangles are not in canonical top/left order")
    owner_bbox, owner_area = _subtitle_rectangle_aggregate(owner)
    pending_bbox, pending_area = _subtitle_rectangle_aggregate(pending)
    if tuple(words[5:9]) != owner_bbox or words[9] != owner_area:
        raise ValueError("SLR12 owner bbox or area disagrees with its rectangles")
    if tuple(words[13:17]) != pending_bbox or words[17] != pending_area:
        raise ValueError("SLR12 pending bbox or area disagrees with its rectangles")

    target = _float32_bits(words[18])
    target_valid = bool(flags & SUBTITLE_LOCATOR_FLAG_TARGET_VALID)
    owner_flag = bool(flags & SUBTITLE_LOCATOR_FLAG_OWNER)
    pending_flag = bool(flags & SUBTITLE_LOCATOR_FLAG_PENDING)
    target_reset = bool(flags & SUBTITLE_LOCATOR_FLAG_TARGET_RESET)
    if owner_flag != (owner_count != 0) or pending_flag != (pending_count != 0):
        raise ValueError("SLR12 owner/pending flags disagree with their rectangle counts")
    if owner_flag != (words[3] != 0):
        raise ValueError("SLR12 owner generation disagrees with owner authority")
    if current_count > owner_count:
        raise ValueError("SLR12 current rectangle count cannot exceed its owner count")
    if current_count != 0 and not (owner_flag and target_valid and fade in (1, 2)):
        raise ValueError("SLR12 current rectangles require owner and valid target authority")
    if target_valid:
        if (not owner_flag or target_reset or words[19] != words[3] or
                not math.isfinite(target) or
                abs(target) > _DEFAULTS.direct_container_limit or fade not in (1, 2)):
            raise ValueError("SLR12 valid target identity or representation is inconsistent")
    elif target_reset:
        if (not owner_flag or words[18] != 0 or words[19] != 0 or
                current_count != 0 or fade != 0):
            raise ValueError("SLR12 target reset must clear target and current authority")

    counter = words[25]
    grace_bounds = {
        "left": words[29] & 0xFFFF,
        "right": words[29] >> 16,
        "top": words[30] & 0xFFFF,
        "bottom": words[30] >> 16,
    }
    packed_grace_zero = words[29] == 0 and words[30] == 0
    if owner_flag:
        if (counter > coordinate_contract.SUBTITLE_OCR.locator_target_max_unreliable_holds or
                not packed_grace_zero):
            raise ValueError("SLR12 unreliable-target hold exceeds its authenticated limit")
        # Current authority is required to increment this counter, but an observation without
        # current OCR authority preserves an existing valid target/counter without aging or
        # conditioning. The serialized frame therefore may have current_count == 0 here.
        if counter != 0 and not (
                target_valid and fade in (1, 2) and
                words[21] == SUBTITLE_LOCATOR_EVENT_NONE):
            raise ValueError("SLR12 unreliable-target hold requires live target authority")
        if not target_valid and not target_reset and (
                words[18] != 0 or words[19] != 0 or current_count != 0 or fade != 0):
            raise ValueError("SLR12 owner without target authority must clear target words")
        if target_reset and counter != 0:
            raise ValueError("SLR12 target reset must clear unreliable-target hold")
    elif counter == 0:
        if (words[18] != 0 or words[19] != 0 or not packed_grace_zero or
                current_count != 0 or target_valid or target_reset or fade != 0):
            raise ValueError("SLR12 absent owner/grace state must be canonical zero")
    else:
        if counter > coordinate_contract.SUBTITLE_OCR.locator_death_grace_observations:
            raise ValueError("SLR12 death-grace exceeds the authenticated observation limit")
        if (target_valid or target_reset or words[19] != 0 or
                not math.isfinite(target) or
                abs(target) > _DEFAULTS.direct_container_limit or
                current_count != 0 or fade != 0 or
                grace_bounds["left"] >= grace_bounds["right"] or
                grace_bounds["top"] >= grace_bounds["bottom"] or
                grace_bounds["left"] < expected_content[0] or
                grace_bounds["right"] > expected_content[2] or
                grace_bounds["top"] < expected_roi_top or
                grace_bounds["bottom"] > expected_roi_bottom):
            raise ValueError("SLR12 death-grace target or packed bounds are inconsistent")

    return {
        "schema": words[0],
        "tag": words[1],
        "flags": flags,
        "owner_generation": words[3],
        "owner_count": owner_count,
        "owner_bbox": owner_bbox,
        "owner_area": owner_area,
        "analysis_generation": generation,
        "pending_count": pending_count,
        "pending_bbox": pending_bbox,
        "pending_area": pending_area,
        "target": target if target_valid else None,
        "cached_target": target if not owner_flag and counter != 0 else None,
        "target_bits": words[18],
        "target_generation": words[19],
        "target_reset": target_reset,
        "current_count": current_count,
        "last_event": words[21],
        "matched_frame_id": frame_id,
        "fade": fade,
        "target_grace": counter if not owner_flag else 0,
        "unreliable_target_holds": counter if owner_flag else 0,
        "scene_epoch": words[26],
        "field_width": words[27],
        "field_height": words[28],
        "tensor_content_rect": expected_content,
        "packed_grace_x": words[29],
        "packed_grace_y": words[30],
        "packed_kinds": packed_kinds,
        "owner_ribbon_mask": owner_kinds,
        "pending_ribbon_mask": pending_kinds,
        "current_ribbon_mask": current_kinds,
        "grace_bounds": grace_bounds if not owner_flag and counter != 0 else None,
        "owner_rectangles": owner,
        "pending_rectangles": pending,
        "current_rectangles": current,
    }


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


def _plan_host_sbs_v2_window_region(
        semantic: tuple[int, int, int, int], source_width: int, source_height: int,
        tensor_width: int, tensor_height: int
        ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], float] | None:
    """Mirror the exact-source, centered integer contain-fit used by production."""

    left, top, right, bottom = semantic
    width = right - left
    height = bottom - top
    if (width <= 0 or height <= 0 or right > source_width or bottom > source_height or
            (tensor_width, tensor_height) not in _AUTHENTICATED_TENSOR_SHAPES or
            semantic == (0, 0, source_width, source_height)):
        return None

    content_left = 0
    content_top = 0
    content_right = tensor_width
    content_bottom = tensor_height
    if width * tensor_height > height * tensor_width:
        content_height = max(1, tensor_width * height // width)
        content_top = (tensor_height - content_height) // 2
        content_bottom = content_top + content_height
    elif width * tensor_height < height * tensor_width:
        content_width = max(1, tensor_height * width // height)
        content_left = (tensor_width - content_width) // 2
        content_right = content_left + content_width
    content = (content_left, content_top, content_right, content_bottom)
    content_area = (content_right - content_left) * (content_bottom - content_top)
    padded_fraction = _float32(
        1.0 - content_area / (tensor_width * tensor_height))
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
                analysis.get("crop_method") != "same-format D3D11 CopySubresourceRegion" or
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
    accepted_units = (
        {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": "one-eye source-U per curve unit",
            "parallax": "signed one-eye source-U",
        },
        {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": "one-eye full-source-U per curve unit",
            "parallax": "signed one-eye full-source-U",
        },
        {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": "one-eye ROI-local source-U per curve unit",
            "parallax": (
                "signed one-eye ROI-local source-U; full-source renderer authority "
                "additionally requires depth_input_region embedding"),
        },
    )
    if document.get("units") not in accepted_units:
        raise ValueError("shadow_state.json has unknown units")
    if document.get("adaptation_semantics") != {
            "coordinate": (
                "immediate-first-usable-center-latched-until-cut-fixed-"
                "authenticated-scale-retained-across-unusable"
            ),
            "convergence_curve": "arithmetic-mean-center-is-zero-plane",
            "requested_gain": "immutable-cfg-pop-strength",
            "container_scale": (
                "abi-retained-identity-pointwise-soft-container-is-map-local"
            ),
            "near_curve": (
                "fixed-contract-logarithmic-tau-independent-of-content-occupancy"
            ),
            "spatial_conditioner": (
                "fixed-75pct-vertical-majorant-share-then-horizontal-majorant"
            )}:
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
    expected_camera_integrity = camera_center_integrity_bits(
        float(typed["center"]),
        float(typed["inverse_scale"]),
        float(typed["convergence_curve"]),
        int(typed["calibration_revision"]),
    )
    if typed["camera_center_integrity_bits"] != expected_camera_integrity:
        raise ValueError("shadow_state.json camera center integrity checksum disagrees")
    expected_authorization = coordinate_tag if typed["frame_valid"] > 0.5 else 0
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
        "effective_gain": (constants["requested_gain"]
                           if decoded["frame_valid"] else 0.0),
        "camera_center_integrity_bits": expected_camera_integrity,
        "renderer_authorization_bits": expected_authorization,
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
            (camera_valid and not _within_absolute_tolerance(
                native_latched_scale, native_constants["raw_coordinate_scale"],
                _RAW_COORDINATE_SCALE_AUTHENTICATION_TOLERANCE)) or
            (decoded["frame_valid"] and not decoded["camera_valid"]) or
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
                "renderer_authorization_bits"):
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
                native_value["effective_gain"], native_value["requested_gain"],
                _REQUESTED_GAIN_AUTHENTICATION_TOLERANCE) or
            not _within_absolute_tolerance(
                native_value["latched_scale"], native_value["raw_coordinate_scale"],
                _RAW_COORDINATE_SCALE_AUTHENTICATION_TOLERANCE) or
            native_value["convergence_curve"] !=
            _float32(_DEFAULTS.convergence_curve_default) or
            native_value["container_scale"] != 1.0):
        raise ValueError("dump_manifest.json has an invalid V2 state summary")
    return value


def _required_hashed_artifact(
        artifacts: Dict[str, Any], name: str, label: str) -> Dict[str, Any]:
    descriptor = artifacts.get(name)
    if (not isinstance(descriptor, dict) or set(descriptor) != {
            "available", "required", "stage", "description", "sha256"} or
            descriptor.get("available") is not True or
            descriptor.get("required") is not True or
            not isinstance(descriptor.get("stage"), str) or not descriptor["stage"] or
            not isinstance(descriptor.get("description"), str) or
            not descriptor["description"] or
            not _is_sha256_hex(descriptor.get("sha256"))):
        raise ValueError(f"dump_manifest.json has an invalid {label} artifact descriptor")
    return descriptor


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


def _subtitle_locator_resolver_contract() -> Dict[str, Any]:
    return {
        "state_schema": SUBTITLE_LOCATOR_STATE_SCHEMA,
        "state_tag": SUBTITLE_LOCATOR_STATE_TAG,
        "state_word_count": SUBTITLE_LOCATOR_STATE_WORD_COUNT,
        "rectangle_capacity": SUBTITLE_LOCATOR_RECT_CAPACITY,
        "target_policy": {
            "units": "binocular-source-pixels",
            "placement": {
                "primary": "aggregate-owner-median-member-center",
                "fallback_on_primary_failure": True,
                "fallback_span": (
                    "ordinary-core-horizontal-bounds-else-owner-core-horizontal-bounds"),
                "fallback_top": "ordinary-core-top-else-owner-core-top",
                "fallback_step_denominator": (
                    coordinate_contract.SUBTITLE_OCR.
                    locator_target_horizontal_step_denominator),
                "fallback_max_radius_steps": (
                    coordinate_contract.SUBTITLE_OCR.
                    locator_target_horizontal_fallback_max_radius_steps),
                "fallback_order_within_radius": ["negative", "positive"],
                "fallback_radius_policy": "first-reliable-radius",
                "fallback_requires_unclamped_sample_strip": True,
                "fallback_minimum_coherent_rows": 2,
                "fallback_row_median_delta_max": (
                    coordinate_contract.SUBTITLE_OCR.
                    locator_target_max_row_median_delta_binocular_source_pixels),
                "fallback_probe_target": "mean-medians",
                "fallback_pair_target_delta_max": (
                    coordinate_contract.SUBTITLE_OCR.
                    locator_target_max_row_median_delta_binocular_source_pixels),
                "fallback_pair_conflict": "unreliable-stop-search",
                "fallback_within_radius_policy": "maximum-mean-within-delta",
                "ribbon_places_fallback_with_ordinary": False,
            },
            "selection": {
                "applies_to": "primary",
                "samples_per_row": 16,
                "median_indices": [7, 8],
                "iqr_lower_indices": [3, 4],
                "iqr_upper_indices": [11, 12],
                "row_validity": "independent-finite-direct-container",
                "minimum_coherent_rows": 1,
                "single_valid_row": "median",
                "both_valid_within_delta": "mean-medians",
                "both_valid_beyond_delta": "maximum-median",
            },
            "evidence": {
                "row_iqr_max": (
                    coordinate_contract.SUBTITLE_OCR.
                    locator_target_max_row_iqr_binocular_source_pixels),
                "row_median_delta_max": (
                    coordinate_contract.SUBTITLE_OCR.
                    locator_target_max_row_median_delta_binocular_source_pixels),
            },
            "deadband": (
                coordinate_contract.SUBTITLE_OCR.locator_target_deadband_binocular_source_pixels),
            "ema_alpha": coordinate_contract.SUBTITLE_OCR.locator_target_ema_alpha,
            "maximum_slew": (
                coordinate_contract.SUBTITLE_OCR.locator_target_max_slew_binocular_source_pixels),
            "maximum_residual": (
                coordinate_contract.SUBTITLE_OCR.
                locator_target_max_residual_binocular_source_pixels),
            "unreliable_hold": {
                "owner_state_word": 25,
                "maximum_distinct_observations": (
                    coordinate_contract.SUBTITLE_OCR.locator_target_max_unreliable_holds),
                "increment_requires": (
                    "continuing-same-scene-owner-current-authority-valid-target"),
                "preserve_without_current_authority": True,
                "duplicate_observation_ages": False,
                "hard_cut_allowed": False,
            },
            "representation_limit": "direct-parallax-container",
        },
        "shader_contract": _subtitle_shader_contract(_SUBTITLE_LOCATOR_SHADER_SPECS),
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
    if mode == _SUBTITLE_MODE_SLR12:
        expected_files = {
            "ocr_record": "subtitle_ocr_record.u32",
            "locator_state": "subtitle_locator_state.u32",
            "base_field": "shadow_base_final_parallax.f32",
            "conditioned_field": "shadow_final_parallax.f32",
        }
        if subtitle.get("request") is not True:
            raise ValueError("active SLR12 subtitle conditioning must bind an enabled request")
        if subtitle.get("producer") != _subtitle_ocr_producer_contract():
            raise ValueError("active SLR12 subtitle conditioning has unknown OCR8 provenance")
        if subtitle.get("resolver") != _subtitle_locator_resolver_contract():
            raise ValueError("active SLR12 subtitle conditioning has unknown resolver provenance")
        if subtitle.get("artifacts") != expected_files:
            raise ValueError("active SLR12 subtitle conditioning has unknown artifact roles")
        required = {
            "subtitle_ocr_record.u32": "same-frame OCR8 subtitle boxes",
            "subtitle_locator_state.u32": "compact SLR12 subtitle authority state",
            "shadow_base_final_parallax.f32": (
                "ordinary post-limiter V2 field before SLR12 conditioning"),
        }
        for name, stage in required.items():
            descriptor = _required_hashed_artifact(
                artifacts, name, f"active subtitle {name}")
            if descriptor.get("stage") != stage:
                raise ValueError(
                    f"dump_manifest.json misattributes active subtitle artifact {name}")
        return {
            "mode": _SUBTITLE_MODE_SLR12,
            "live": True,
            "request": True,
            "artifact_files": expected_files,
            "subtitle_evidence_complete": True,
        }
    raise ValueError("unsupported subtitle-conditioning authority")


def validate_v2_dump_manifest_document(document: Any) -> Dict[str, Any]:
    """Validate a supported V2 geometry fragment of ``dump_manifest.json``.

    The full package contains color/model metadata owned by other contracts. This reader
    deliberately validates only the candidate -> full-resolution ownership refinement -> vertical
    envelopes/share -> row-majorant chain, its analysis domain, and every authenticated
    intermediate.
    """

    if (not isinstance(document, dict) or
            document.get("schema") != DUMP_MANIFEST_SCHEMA):
        raise ValueError("dump_manifest.json has an unknown serialization schema")
    if (document.get("capture_status") != "complete" or
            document.get("published_atomically") is not True):
        raise ValueError("dump_manifest.json is not a complete atomic publication")
    schema = document["schema"]
    renderer = document.get("renderer")
    shadow = document.get("parallax_v2_shadow")
    artifacts = document.get("artifacts")
    dimensions = document.get("dimensions")
    if not all(isinstance(value, dict) for value in
               (renderer, shadow, artifacts, dimensions)):
        raise ValueError("dump_manifest.json is missing its V2 geometry objects")

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
        "Spatially exact full-source or cropped color input submitted to the calibrated "
        "preprocess; transfer-aware PNG is diagnostic only and never numeric model authority."
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
        raise ValueError("active SLR12 subtitle conditioning requires selected V2 geometry")
    requested = renderer.get("parallax_v2_render_requested")
    mapping_matches = renderer.get("mapping_artifacts_match_selected_renderer")
    if (not isinstance(requested, bool) or
            not isinstance(mapping_matches, bool) or
            (selected and not requested) or
            (input_mode == "window-region" and not mapping_matches)):
        raise ValueError("dump_manifest.json has invalid V2 renderer request attribution")

    expected_position = (
        "shadow_final_parallax + depth_input_region embedding"
        if selected and input_mode == "window-region" else
        "shadow_final_parallax" if selected else None
    )
    expected_authority = (
        ("authenticated crop-local parallax-v2 conditioned field plus depth-input-region embedding"
         if input_mode == "window-region" else
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
        "least column-wise upper envelope v+ >= ownership-refined candidate with adjacent-row source-U change <= "
        "max_vertical_shear/content_width; diagnostic evidence only"
        if selected else None
    )
    expected_ownership_role = (
        (("conservative full-resolution crop-local source-contour foreground ownership applied "
          "to candidate before the vertical conditioner; may only raise uniquely owned far-side "
          "boundary texels") if input_mode == "window-region" else
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
          "max_horizontal_slope and vertical shear <= max_vertical_shear produces "
          "shadow_base_final_parallax; SLR12 applies the analytic anisotropic rectangle "
          "budget/fade from same-frame current authority and publishes shadow_final_parallax, "
          "which plus depth_input_region embedding is live position authority")
         if subtitle_live and input_mode == "window-region" else
         ("least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
          "max_horizontal_slope and vertical shear <= max_vertical_shear produces "
          "shadow_base_final_parallax; SLR12 applies the analytic anisotropic rectangle "
          "budget/fade from same-frame current authority and publishes "
          "shadow_final_parallax as live position authority")
         if subtitle_live else
         ("least row-wise crop-local q >= shadow_vertical_conditioned with horizontal slope <= "
          "max_horizontal_slope and vertical shear <= max_vertical_shear; q plus "
          "depth_input_region embedding is live position authority")
         if input_mode == "window-region" else
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
    if subtitle_live:
        expected_artifacts["shadow_base_final_parallax.f32"] = (
            "ordinary post-limiter V2 field before SLR12 conditioning", True)
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
        if subtitle_live and (
                geometry_width, geometry_height) not in _AUTHENTICATED_TENSOR_SHAPES:
            raise ValueError(
                "active SLR12 subtitle conditioning requires a calibrated DAV2 field")
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
    elif any(value is not None for value in geometry_dimensions):
        raise ValueError("dump_manifest.json exposes inactive V2 geometry dimensions")
    if not subtitle_live and (
            "shadow_base_final_parallax.f32" in artifacts or
            "shadow_base_final_parallax" in dimensions):
        raise ValueError("inactive subtitle conditioning exposes an SLR12 base field")

    if input_mode == "window-region":
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
            if input_mode == "window-region" else ["shadow_final_parallax"]),
        "window_region_available": region_available,
        "window_region_observer_status": region_observer_status,
        "window_region_mapping_status": region_mapping_status,
        "subtitle_conditioning": subtitle_conditioning,
        "shadow_state_summary": shadow_state_summary,
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
    return payload


def _verify_subtitle_conditioning_artifacts(
        dump_dir: Any, manifest: Dict[str, Any], *, matched_frame_id: int,
        analysis_generation: int, source_width: int, source_height: int,
        field_width: int, field_height: int,
        tensor_content: tuple[int, int, int, int],
        confirmed_cut_count: int) -> Dict[str, Any]:
    import json

    metadata_payload = _read_hashed_artifact(
        dump_dir, manifest["artifacts"], "subtitle_conditioning.json")
    try:
        metadata = json.loads(metadata_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("subtitle_conditioning.json is malformed") from error
    if metadata != manifest.get("subtitle_conditioning"):
        raise ValueError("subtitle_conditioning.json disagrees with the manifest")
    mode = metadata.get("mode")
    if mode == _SUBTITLE_MODE_SLR12:
        content = _subtitle_tensor_content(field_width, field_height, tensor_content)
        dynamic_roi = _subtitle_dynamic_roi(source_width, source_height, content)
        if dynamic_roi is None:
            raise ValueError(
                "active SLR12 subtitle conditioning has unsupported source/field geometry")
        roi_top, roi_bottom = dynamic_roi
        ocr_payload = _read_hashed_artifact(
            dump_dir, manifest["artifacts"], "subtitle_ocr_record.u32")
        locator_payload = _read_hashed_artifact(
            dump_dir, manifest["artifacts"], "subtitle_locator_state.u32")
        ocr = validate_subtitle_ocr_record(
            ocr_payload,
            matched_frame_id=matched_frame_id,
            analysis_generation=analysis_generation,
            source_width=source_width,
            source_height=source_height,
            field_width=field_width,
            field_height=field_height,
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            tensor_content=content,
        )
        locator = validate_subtitle_locator_state(
            locator_payload,
            matched_frame_id=matched_frame_id,
            analysis_generation=analysis_generation,
            source_width=source_width,
            source_height=source_height,
            field_width=field_width,
            field_height=field_height,
            tensor_content=content,
            expected_scene_epoch=confirmed_cut_count,
        )
        if not ocr["authoritative"] and locator["current_count"] != 0:
            raise ValueError("an abstaining OCR8 record cannot authorize SLR12 current rectangles")
        selected_indices = _subtitle_selected_ocr_indices(
            ocr["raw_boxes"], content[2] - content[0])
        selected_final_rectangles = [
            (ocr["final_boxes"][index]["left"],
             ocr["final_boxes"][index]["top"],
             ocr["final_boxes"][index]["right"],
             ocr["final_boxes"][index]["bottom"],
             ocr["final_boxes"][index]["kind"])
            for index in selected_indices
        ]
        final_rectangles = {
            (rectangle["left"], rectangle["top"], rectangle["right"], rectangle["bottom"],
             rectangle["kind"])
            for rectangle in ocr["final_boxes"]
        }
        if any(
                (rectangle["left"], rectangle["top"],
                 rectangle["right"], rectangle["bottom"],
                 rectangle["kind"]) not in final_rectangles
                for rectangle in locator["current_rectangles"]):
            raise ValueError(
                "SLR12 current rectangles/kinds are not exact covers from same-frame OCR8")
        current_tuples = [
            (rectangle["left"], rectangle["top"],
             rectangle["right"], rectangle["bottom"], rectangle["kind"])
            for rectangle in locator["current_rectangles"]
        ]
        selected_position = 0
        selection_order_valid = True
        for rectangle in current_tuples:
            while (selected_position < len(selected_final_rectangles) and
                   selected_final_rectangles[selected_position] != rectangle):
                selected_position += 1
            if selected_position < len(selected_final_rectangles):
                selected_position += 1
            else:
                selection_order_valid = False
                break
        if not selection_order_valid:
            raise ValueError(
                "SLR12 current rectangles are outside the frozen same-frame OCR8 selection")
        return {
            "mode": _SUBTITLE_MODE_SLR12,
            "subtitle_evidence_verified": True,
            "ocr_authoritative": ocr["authoritative"],
            "ocr_raw_count": ocr["raw_count"],
            "ocr_final_count": ocr["final_count"],
            "source_width": ocr["source_width"],
            "source_height": ocr["source_height"],
            "field_width": ocr["field_width"],
            "field_height": ocr["field_height"],
            "tensor_content_rect": content,
            "roi_top": ocr["roi_top"],
            "roi_bottom": ocr["roi_bottom"],
            "owner_count": locator["owner_count"],
            "pending_count": locator["pending_count"],
            "current_count": locator["current_count"],
            "last_event": locator["last_event"],
            "cached_target": locator["cached_target"],
            "target_grace": locator["target_grace"],
            "unreliable_target_holds": locator["unreliable_target_holds"],
            "grace_bounds": locator["grace_bounds"],
            "current_rectangles": locator["current_rectangles"],
            "target": locator["target"],
            "fade": locator["fade"],
        }
    return {
        "mode": _SUBTITLE_MODE_NONE,
        "subtitle_evidence_verified": False,
    }


def _sm5_power_of_two_division_candidates(
        numerator: float, denominator: int) -> tuple[Any, ...]:
    """Return every bit-exact SM5 result for this bounded positive division.

    SLR12's three divisions all have an exactly representable power-of-two numerator and an
    exactly representable positive integer denominator.  The D3D11.3 functional specification,
    section 3.1.3.1, permits ``x / y`` to execute as ``x * (1 / y)`` with a reciprocal accurate
    to one ULP.  Scaling that reciprocal by these power-of-two numerators is exact in SLR12's
    bounded normal range.  A nonrepresentable quotient therefore admits exactly its two
    bracketing float32 values; an exactly representable quotient admits itself and its two
    immediate neighbors.  The choice is device-dependent: NVIDIA hardware uses the lower
    bracketing value for 0.5 / 1101 while WARP and NumPy use the upper one.

    Keeping this as a finite set of exact bit patterns is important.  It is not an epsilon: a
    captured field must still equal one globally consistent SM5 recurrence bit for bit.
    """

    import numpy as np

    denominator32 = np.float32(denominator)
    numerator32 = np.float32(numerator)
    if (denominator <= 0 or denominator > _MAXIMUM_SOURCE_LONG_SIDE or
            int(denominator32) != denominator or
            not np.isfinite(numerator32) or numerator32 <= np.float32(0.0)):
        raise ValueError(
            "SM5 replay division requires bounded exact positive float32 operands")
    numerator_integer, numerator_denominator = float(numerator32).as_integer_ratio()
    if ((numerator_integer & (numerator_integer - 1)) != 0 or
            (numerator_denominator & (numerator_denominator - 1)) != 0):
        raise ValueError("SM5 replay division numerator must be an exact power of two")

    nearest = np.divide(numerator32, denominator32, dtype=np.float32)
    nearest_integer, nearest_denominator = float(nearest).as_integer_ratio()
    # Compare the rounded float with numerator / denominator using integers only.  Within the
    # production bound (<= 5120), a non-power-of-two denominator is far from a float32 binade
    # boundary, so the one-ULP reciprocal allowance contains exactly the two floats bracketing
    # the rational.  A power-of-two denominator makes the rational exact, hence the explicit
    # exact-plus-neighbors case above.  Multiplication by SLR12's power-of-two numerator only
    # changes the exponent, so it neither introduces rounding nor changes this candidate count.
    comparison = (
        nearest_integer * numerator_denominator * denominator -
        numerator_integer * nearest_denominator
    )
    if comparison == 0:
        return (
            nearest,
            np.nextafter(nearest, np.float32(-np.inf), dtype=np.float32),
            np.nextafter(nearest, np.float32(np.inf), dtype=np.float32),
        )
    direction = np.float32(-np.inf if comparison > 0 else np.inf)
    adjacent = np.nextafter(nearest, direction, dtype=np.float32)
    return (nearest, adjacent)


def _replay_slr12_conditioner(
        base_field: Any, subtitle: Dict[str, Any], *,
        division_values: tuple[Any, Any, Any] | None = None) -> Any:
    """Replay the frozen SLR12 analytic rectangle conditioner in float32 order."""

    import numpy as np

    base = np.asarray(base_field, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError("SLR12 Base field must be a two-dimensional float32 array")
    height, width = base.shape
    content = _subtitle_tensor_content(
        width, height, subtitle.get("tensor_content_rect"))
    content_width = content[2] - content[0]
    # The shader evaluates all synthetic padding at its nearest real content cell and loads Base
    # there, including the inactive/empty-current path.
    x_cells = np.clip(np.arange(width, dtype=np.int64), content[0], content[2] - 1)
    y_cells = np.clip(np.arange(height, dtype=np.int64), content[1], content[3] - 1)
    base_for_conditioning = base[np.ix_(y_cells, x_cells)]
    rectangles = subtitle["current_rectangles"]
    if not rectangles:
        return base_for_conditioning.copy()
    source_width = _uint32(subtitle.get("source_width"), "SLR12 analysis source width")
    if source_width == 0:
        raise ValueError("SLR12 analysis source width must be positive")
    target = subtitle.get("target")
    fade = subtitle.get("fade")
    if target is None or fade not in (1, 2):
        raise ValueError("SLR12 current geometry lacks valid target/fade authority")
    target32 = np.float32(target)
    if (not np.isfinite(target32) or
            abs(target32) > np.float32(_DEFAULTS.direct_container_limit)):
        raise ValueError("SLR12 target exceeds its authenticated representation limit")

    if ((width, height) not in _AUTHENTICATED_TENSOR_SHAPES or
            width != subtitle.get("field_width") or
            height != subtitle.get("field_height")):
        raise ValueError("SLR12 conditioner field does not match its calibrated authority")
    x = x_cells[None, :]
    y = y_cells[:, None]
    if division_values is None:
        horizontal_step = np.float32(
            np.float32(_DEFAULTS.max_horizontal_slope) / np.float32(content_width))
        vertical_step = np.float32(
            np.float32(_DEFAULTS.max_vertical_shear) / np.float32(content_width))
        core_range = np.float32(np.float32(0.5) / np.float32(source_width))
    else:
        horizontal_step, vertical_step, core_range = (
            np.float32(value) for value in division_values)
    best_distance = np.full(base.shape, np.float32(np.inf), dtype=np.float32)
    for rectangle in rectangles:
        left = rectangle["left"]
        top = rectangle["top"]
        right = rectangle["right"]
        bottom = rectangle["bottom"]
        if rectangle.get("ribbon") is True:
            # A bottom ribbon is a constant plane whose only exterior boundary is its
            # corrected top edge.  Its authenticated cover is full-width/to-bottom, but
            # keeping the analytic replay explicitly top-only prevents future geometry
            # refactors from silently reintroducing side or bottom collars.
            dx = np.zeros_like(x, dtype=np.float32)
            dy = np.maximum(top - y, 0).astype(np.float32)
        else:
            dx = np.where(x < left, left - x,
                          np.where(x >= right, x - (right - 1), 0)).astype(np.float32)
            dy = np.where(y < top, top - y,
                          np.where(y >= bottom, y - (bottom - 1), 0)).astype(np.float32)
        horizontal_distance = np.multiply(dx, horizontal_step, dtype=np.float32)
        vertical_distance = np.multiply(dy, vertical_step, dtype=np.float32)
        distance = np.add(horizontal_distance, vertical_distance, dtype=np.float32)
        best_distance = np.minimum(best_distance, distance)

    budget = np.add(core_range, best_distance, dtype=np.float32)
    delta = np.subtract(base_for_conditioning, target32, dtype=np.float32)
    safe = np.less_equal(np.abs(delta), budget)
    base_in_range = np.less_equal(
        np.abs(base_for_conditioning), np.float32(_DEFAULTS.direct_container_limit))
    signed_budget = np.where(delta < np.float32(0.0), -budget, budget).astype(np.float32)
    full = np.add(target32, signed_budget, dtype=np.float32)
    if fade == 1:
        half_delta = np.multiply(
            np.float32(0.5), np.subtract(full, base_for_conditioning, dtype=np.float32),
            dtype=np.float32)
        conditioned = np.add(base_for_conditioning, half_delta, dtype=np.float32)
    else:
        conditioned = full
    return np.where(
        base_in_range & ~safe, conditioned, base_for_conditioning).astype(np.float32)


def _replay_slr12_conditioner_sm5_candidates(
        base_field: Any, subtitle: Dict[str, Any]):
    """Yield the finite, globally consistent bit-exact SM5 SLR12 recurrences."""

    import numpy as np

    base = np.asarray(base_field, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError("SLR12 Base field must be a two-dimensional float32 array")
    source_width = _uint32(subtitle.get("source_width"), "SLR12 analysis source width")
    width = base.shape[1]
    height = base.shape[0]
    content = _subtitle_tensor_content(
        width, height, subtitle.get("tensor_content_rect"))
    content_width = content[2] - content[0]
    choices = (
        _sm5_power_of_two_division_candidates(
            _DEFAULTS.max_horizontal_slope, content_width),
        _sm5_power_of_two_division_candidates(
            _DEFAULTS.max_vertical_shear, content_width),
        _sm5_power_of_two_division_candidates(0.5, source_width),
    )
    seen = set()
    for division_values in product(*choices):
        bit_key = tuple(
            int(np.asarray(value, dtype=np.float32).view(np.uint32))
            for value in division_values)
        if bit_key in seen:
            continue
        seen.add(bit_key)
        yield _replay_slr12_conditioner(
            base, subtitle, division_values=division_values)


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
            "(raw_reproject_source_u_normalized - aspect_fitted_unwarped_source_u) * "
            "content_scale_x * eye_width" or
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


def _replay_v2_limiter_fields(ownership: Any, content_width: int) -> Tuple[Any, Any, Any]:
    """Replay the contract's serial/Q30 limiter branches exactly in NumPy."""

    import numpy as np

    values = np.asarray(ownership, dtype=np.float32)
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
         recurrences recomputed from ``ownership_refined``; SLR12's analytic rectangle budget and
         fade exactly reproduce the selected final field; and ownership refinement never lowers
         the candidate.

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
    )
    subtitle_live = subtitle_summary["mode"] == _SUBTITLE_MODE_SLR12
    geometry_chain_fields = (
        "shadow_candidate_parallax",
        "shadow_ownership_refined_parallax",
        "shadow_vertical_majorant",
        "shadow_vertical_conditioned",
        *(('shadow_base_final_parallax',) if subtitle_live else ()),
        "shadow_final_parallax",
    )
    fields: Dict[str, Any] = {}
    for name in geometry_chain_fields:
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

    # Exact replicas of the production serial/Q30 branches (validated bitwise against the GPU
    # intermediates; see docs/host-sbs.md#cliff-conditioning).
    content_left, content_top, content_right, content_bottom = (
        input_region["tensor_content_rect"])
    content_width = content_right - content_left
    majorant, conditioned, final = _replay_v2_limiter_fields(ownership, content_width)

    recurrence_fields = [
        ("shadow_vertical_majorant", majorant),
        ("shadow_vertical_conditioned", conditioned),
    ]
    recurrence_fields.append((
        "shadow_base_final_parallax" if subtitle_live else "shadow_final_parallax",
        final,
    ))
    for name, recomputed in recurrence_fields:
        if not np.array_equal(fields[name], recomputed):
            mismatch = float(np.max(np.abs(fields[name] - recomputed)))
            raise ValueError(
                f"{name}.f32 is not the exact recurrence/serial-Q30 limiter replay of the dumped ownership field "
                f"(max abs diff {mismatch})")

    if subtitle_live:
        replay_matched = False
        minimum_mismatch = math.inf
        for replayed_subtitle in _replay_slr12_conditioner_sm5_candidates(
                fields["shadow_base_final_parallax"], subtitle_summary):
            if np.array_equal(fields["shadow_final_parallax"], replayed_subtitle):
                replay_matched = True
                break
            minimum_mismatch = min(
                minimum_mismatch,
                float(np.max(np.abs(
                    fields["shadow_final_parallax"] - replayed_subtitle))))
        if not replay_matched:
            if subtitle_summary["current_count"] == 0:
                raise ValueError(
                    "SLR12 has no current geometry but the final field is not the exact "
                    "content-clamped Base extension")
            raise ValueError(
                "shadow_final_parallax.f32 is not the exact SLR12 rectangle-conditioning "
                f"recurrence (minimum max abs diff {minimum_mismatch})")

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
        "final_recurrence_verified": True,
    }


__all__ = [
    "DEPTH_INPUT_REGION_SCHEMA",
    "DIAGNOSTIC_SOURCE_CLOSURE_SHA256",
    "DUMP_MANIFEST_SCHEMA",
    "LIVE_RENDERER_SOURCE_CLOSURE_SHA256",
    "SHADOW_FRAME_STATS_DUMP_SCHEMA",
    "SHADOW_STATE_DUMP_SCHEMA",
    "SUBTITLE_LOCATOR_CURRENT_WORD_OFFSET",
    "SUBTITLE_LOCATOR_EVENT_BIRTH",
    "SUBTITLE_LOCATOR_EVENT_DEATH",
    "SUBTITLE_LOCATOR_EVENT_HANDOFF",
    "SUBTITLE_LOCATOR_EVENT_NONE",
    "SUBTITLE_LOCATOR_FLAG_OWNER",
    "SUBTITLE_LOCATOR_FLAG_PENDING",
    "SUBTITLE_LOCATOR_FLAG_TARGET_RESET",
    "SUBTITLE_LOCATOR_FLAG_TARGET_VALID",
    "SUBTITLE_LOCATOR_HEADER_WORD_COUNT",
    "SUBTITLE_LOCATOR_OWNER_WORD_OFFSET",
    "SUBTITLE_LOCATOR_PENDING_WORD_OFFSET",
    "SUBTITLE_LOCATOR_RECT_CAPACITY",
    "SUBTITLE_LOCATOR_STATE_SCHEMA",
    "SUBTITLE_LOCATOR_STATE_TAG",
    "SUBTITLE_LOCATOR_STATE_WORD_COUNT",
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
    "shadow_frame_valid_from_statistics",
    "shadow_state_constant_float32",
    "validate_depth_input_region_document",
    "validate_subtitle_locator_state",
    "validate_subtitle_ocr_record",
    "validate_window_region_document",
    "validate_v2_dump_manifest_document",
    "validate_shadow_frame_stats_document",
    "validate_shadow_state_document",
    "verify_v2_dump_geometry",
]
