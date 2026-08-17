#!/usr/bin/env python3
"""Read-only Python view of the generated depth-coordinate-v2 calibration contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Optional, Sequence


CONTRACT_PATH = (
    Path(__file__).resolve().parent / "contracts" / "depth-coordinate-v2-v1.json"
)
CALIBRATED_DEFAULT_NAMES = (
    "collapse_abs_epsilon",
    "far_tau",
    "near_log_tau",
    "gain_per_pop",
    "reference_pop_strength",
    "direct_container_limit",
    "max_horizontal_slope",
    "max_vertical_shear",
    "vertical_majorant_share",
    "convergence_curve_default",
)


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


MODEL_CALIBRATION_KEYS = {
    "calibration_id", "depth_model", "depth_model_url", "onnx_sha256",
    "raw_coordinate_scale", "calibrated_input_shapes", "preprocess",
}
PREPROCESS_KEYS = {
    "profile", "source_closure_schema", "source_file", "source_entrypoint",
    "source_target", "source_compile_flags", "source_macro_count",
    "source_closure_sha256", "model_input_schema", "dtype", "layout", "channels",
    "patch_multiple", "maximum_dimension", "imagenet_mean", "imagenet_std", "stage",
}
SHADER_IMPLEMENTATION_KEYS = {
    "source_closure_schema", "source_compile_flags", "source_macro_count", "source_specs",
    "source_closure_sha256",
}
SHADER_SOURCE_SPECS = (
    ("rgb_to_nchw_cs.hlsl", "main", "cs_5_0"),
    ("rgb_to_nchw_cs.hlsl", "content_main", "cs_5_0"),
    ("rgb_to_nchw_cs.hlsl", "pad_main", "cs_5_0"),
    ("buffer_to_tex_cs.hlsl", "main", "cs_5_0"),
    ("buffer_to_tex_cs.hlsl", "pad_main", "cs_5_0"),
    ("depth_minmax_ema_cs.hlsl", "main", "cs_5_0"),
    ("depth_hist_cs.hlsl", "main", "cs_5_0"),
    ("depth_scene_cut_evidence_cs.hlsl", "main", "cs_5_0"),
    ("depth_scene_cut_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_valid_history_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_moments_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_frame_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_state_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_map_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_ownership_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_vertical_limit_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_limit_cs.hlsl", "main", "cs_5_0"),
    ("host_sbs_ocr_preprocess_cs.hlsl", "main", "cs_5_0"),
    ("host_sbs_ocr_boxes_cs.hlsl", "cells_main", "cs_5_0"),
    ("host_sbs_ocr_boxes_cs.hlsl", "resolve_main", "cs_5_0"),
    ("host_sbs_subtitle_locator_cs.hlsl", "resolve_main", "cs_5_0"),
    ("host_sbs_subtitle_locator_cs.hlsl", "condition_prepare_main", "cs_5_0"),
    ("host_sbs_subtitle_locator_cs.hlsl", "condition_main", "cs_5_0"),
)
EXPECTED_FINAL_PARALLAX = {
    "schema": 2,
    "authority": "complete-atomic-subtitle-conditioned-r32f-live-render-authority",
    "publication_policy": (
        "authenticated-infer-or-cpu-known-publication-or-authenticated-cadence-due-"
        "subtitle-publication-direct-render"),
    "reuse_policy": (
        "ordinary-reuse-holds-complete-depth-ocr-slr-final-tuple-byte-for-byte;"
        "authenticated-cadence-due-reuse-holds-depth-and-publishes-current-ocr-or-"
        "abstention-slr-final-tuple-against-retained-base"),
    "invalid_policy": "fail-closed-flat",
    "current_rgb_policy": "always-current-never-retained",
}
EXPECTED_SUBTITLE_OCR = {
    "schema": 13,
    "logical_model": "ppocrv6_tiny_det_modelopt_fp16",
    "asset_path": "models/ppocrv6_tiny_det_modelopt045_mixed_fp16_fp32io.onnx",
    "artifact_onnx_sha256": (
        "169a233ba0ff7cac27f8ec7dccb6a406e614b25b21fe6a5638c423bf2118bb44"),
    "source_url": (
        "https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/"
        "2ba1506c0380b8f0b03dd142459aac66d4421f6c/inference.onnx?download=true"),
    "source_onnx_sha256": (
        "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8"),
    "conversion_tool": "nvidia-modelopt",
    "conversion_version": "0.45.0",
    "conversion_recipe": "nvidia-modelopt-autocast-fp16-keep-io-fp32-v1",
    "conversion_calibration_profile": "apollo-live8-bottom960x160-v1",
    "engine_recipe": (
        "trt-strong-modelopt045-fp16-iofp32-tf32-fixed960x160-level5-v2"),
    "preprocess_profile": "apollo-ppocrv6-bottom-6x1-bgr-imagenet-v1",
    "source_crop": "bottom-6:1",
    "input_tensor": {
        "name": "x", "dtype": "float32", "layout": "NCHW",
        "shape": [1, 3, 160, 960], "channels": ["B", "G", "R"],
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    },
    "output_tensor": {
        "name": "fetch_name_0", "dtype": "float32", "layout": "NCHW",
        "shape": [1, 1, 160, 960],
    },
    "field_policy": {
        "detector_active_probability_threshold": 0.2,
        "detector_min_mean_score": 0.4,
        "locator_max_width_numerator": 9,
        "locator_max_width_denominator": 10,
        "locator_min_width_cells": 48,
        "locator_min_height_cells": 6,
        "locator_min_aspect_numerator": 2,
        "locator_min_aspect_denominator": 1,
        "locator_provisional_min_vertical_overlap_numerator": 3,
        "locator_provisional_min_vertical_overlap_denominator": 4,
        "locator_provisional_max_height_ratio": 2,
        "locator_provisional_max_center_y_delta_shorter_height": 1,
        "locator_corner_edge_divisor": 32,
        "locator_corner_bottom_rows": 16,
        "locator_match_iou_threshold": 0.6,
        "locator_death_grace_observations": 6,
        "locator_target_horizontal_fallback_max_radius_steps": 2,
        "locator_target_horizontal_step_denominator": 16,
        "locator_target_max_row_iqr_binocular_source_pixels": 8.0,
        "locator_target_max_row_median_delta_binocular_source_pixels": 4.0,
        "locator_target_max_residual_binocular_source_pixels": 8.0,
        "locator_target_max_unreliable_holds": 2,
        "locator_target_deadband_binocular_source_pixels": 1.0,
        "locator_target_ema_alpha": 0.125,
        "locator_target_max_slew_binocular_source_pixels": 0.25,
        "ocr_safe_row_top": 24,
        "ocr_safe_row_bottom": 155,
        "source_crop_aspect_width": 6,
        "source_crop_aspect_height": 1,
        "text_join_gap_cells": 4,
        "ribbon_join_gap_cells": 12,
        "ribbon_structural_gap_min_cells": 3,
        "ribbon_min_structural_gaps": 3,
        "ribbon_min_width_numerator": 1,
        "ribbon_min_width_denominator": 2,
        "ribbon_bottom_tolerance_pixels": 2,
        "ribbon_bottom_tolerance_projection":
            "exact-ceil-detector-edge-through-bottom-crop-v1",
        "ribbon_cover_pad_limit": 8,
    },
    "ocr_record": {
        "schema": 3, "tag": 0x3852434F, "word_count": 208,
        "header_word_count": 16, "box_word_count": 8,
        "box_flag_ribbon": 1, "box_known_flags": 1,
        "raw_box_offset": 16, "raw_box_capacity": 16,
        "final_box_offset": 144, "final_box_capacity": 8,
    },
    "locator_state": {
        "schema": 13, "tag": 0x33314C53, "word_count": 80,
        "header_word_count": 32, "rectangle_capacity": 4,
        "owner_offset": 32, "pending_offset": 48, "current_offset": 64,
        "kind_word": 31,
        "owner_kind_shift": 0, "pending_kind_shift": 4,
        "current_kind_shift": 8, "kind_mask": 15,
        "provisional_current_flag": 1 << 4,
        "provisional_target_word": 29, "provisional_fade_word": 30,
    },
    "condition_params": {
        "schema": 3, "tag": 0x33504353, "word_count": 6,
    },
}


@dataclass(frozen=True)
class CalibratedDefaults:
    collapse_abs_epsilon: float
    far_tau: float
    near_log_tau: float
    gain_per_pop: float
    reference_pop_strength: float
    direct_container_limit: float
    max_horizontal_slope: float
    max_vertical_shear: float
    vertical_majorant_share: float
    convergence_curve_default: float


@dataclass(frozen=True)
class ModelInputContract:
    profile: str
    source_closure_schema: int
    source_file: str
    source_entrypoint: str
    source_target: str
    source_compile_flags: int
    source_macro_count: int
    source_closure_sha256: str
    model_input_schema: int
    dtype: str
    layout: str
    channels: tuple[str, str, str]
    patch_multiple: int
    maximum_dimension: int
    imagenet_mean: tuple[float, float, float]
    imagenet_std: tuple[float, float, float]
    stage: str


@dataclass(frozen=True)
class ModelCalibration:
    calibration_id: str
    depth_model: str
    depth_model_url: str
    onnx_sha256: str
    raw_coordinate_scale: float
    calibrated_input_shapes: tuple[tuple[int, int], ...]
    preprocess: ModelInputContract


@dataclass(frozen=True)
class ShaderImplementation:
    source_closure_schema: int
    source_compile_flags: int
    source_macro_count: int
    source_specs: tuple[tuple[str, str, str], ...]
    source_closure_sha256: str


@dataclass(frozen=True)
class FinalParallaxContract:
    schema: int
    authority: str
    publication_policy: str
    reuse_policy: str
    invalid_policy: str
    current_rgb_policy: str


@dataclass(frozen=True)
class SubtitleOcrContract:
    schema: int
    logical_model: str
    asset_path: str
    artifact_onnx_sha256: str
    source_url: str
    source_onnx_sha256: str
    conversion_tool: str
    conversion_version: str
    conversion_recipe: str
    conversion_calibration_profile: str
    engine_recipe: str
    preprocess_profile: str
    source_crop: str
    input_name: str
    input_dtype: str
    input_layout: str
    input_shape: tuple[int, int, int, int]
    input_channels: tuple[str, str, str]
    imagenet_mean: tuple[float, float, float]
    imagenet_std: tuple[float, float, float]
    output_name: str
    output_dtype: str
    output_layout: str
    output_shape: tuple[int, int, int, int]
    detector_active_probability_threshold: float
    detector_min_mean_score: float
    locator_max_width_numerator: int
    locator_max_width_denominator: int
    locator_min_width_cells: int
    locator_min_height_cells: int
    locator_min_aspect_numerator: int
    locator_min_aspect_denominator: int
    locator_provisional_min_vertical_overlap_numerator: int
    locator_provisional_min_vertical_overlap_denominator: int
    locator_provisional_max_height_ratio: int
    locator_provisional_max_center_y_delta_shorter_height: int
    locator_corner_edge_divisor: int
    locator_corner_bottom_rows: int
    locator_match_iou_threshold: float
    locator_death_grace_observations: int
    locator_target_horizontal_fallback_max_radius_steps: int
    locator_target_horizontal_step_denominator: int
    locator_target_max_row_iqr_binocular_source_pixels: float
    locator_target_max_row_median_delta_binocular_source_pixels: float
    locator_target_max_residual_binocular_source_pixels: float
    locator_target_max_unreliable_holds: int
    locator_target_deadband_binocular_source_pixels: float
    locator_target_ema_alpha: float
    locator_target_max_slew_binocular_source_pixels: float
    ocr_safe_row_top: int
    ocr_safe_row_bottom: int
    source_crop_aspect_width: int
    source_crop_aspect_height: int
    text_join_gap_cells: int
    ribbon_join_gap_cells: int
    ribbon_structural_gap_min_cells: int
    ribbon_min_structural_gaps: int
    ribbon_min_width_numerator: int
    ribbon_min_width_denominator: int
    ribbon_bottom_tolerance_pixels: int
    ribbon_bottom_tolerance_projection: str
    ribbon_cover_pad_limit: int
    record_schema: int
    record_tag: int
    record_word_count: int
    record_header_word_count: int
    box_word_count: int
    box_flag_ribbon: int
    box_known_flags: int
    raw_box_offset: int
    raw_box_capacity: int
    final_box_offset: int
    final_box_capacity: int
    locator_schema: int
    locator_tag: int
    locator_word_count: int
    locator_header_word_count: int
    locator_rectangle_capacity: int
    locator_owner_offset: int
    locator_pending_offset: int
    locator_current_offset: int
    locator_kind_word: int
    locator_owner_kind_shift: int
    locator_pending_kind_shift: int
    locator_current_kind_shift: int
    locator_kind_mask: int
    locator_provisional_current_flag: int
    locator_provisional_target_word: int
    locator_provisional_fade_word: int
    condition_param_schema: int
    condition_param_tag: int
    condition_param_word_count: int


def _subtitle_ocr_contract(contract: dict[str, Any]) -> SubtitleOcrContract:
    value = contract.get("subtitle_ocr")
    if value != EXPECTED_SUBTITLE_OCR:
        raise ValueError(
            "depth-coordinate-v2 subtitle_ocr identity or ABI does not match the current "
            "authenticated PP-OCRv6/OCR8/SLR13 contract")
    input_tensor = value["input_tensor"]
    output_tensor = value["output_tensor"]
    field_policy = value["field_policy"]
    record = value["ocr_record"]
    locator = value["locator_state"]
    condition = value["condition_params"]
    return SubtitleOcrContract(
        schema=value["schema"],
        logical_model=value["logical_model"],
        asset_path=value["asset_path"],
        artifact_onnx_sha256=value["artifact_onnx_sha256"],
        source_url=value["source_url"],
        source_onnx_sha256=value["source_onnx_sha256"],
        conversion_tool=value["conversion_tool"],
        conversion_version=value["conversion_version"],
        conversion_recipe=value["conversion_recipe"],
        conversion_calibration_profile=value["conversion_calibration_profile"],
        engine_recipe=value["engine_recipe"],
        preprocess_profile=value["preprocess_profile"],
        source_crop=value["source_crop"],
        input_name=input_tensor["name"],
        input_dtype=input_tensor["dtype"],
        input_layout=input_tensor["layout"],
        input_shape=tuple(input_tensor["shape"]),
        input_channels=tuple(input_tensor["channels"]),
        imagenet_mean=tuple(float(item) for item in input_tensor["imagenet_mean"]),
        imagenet_std=tuple(float(item) for item in input_tensor["imagenet_std"]),
        output_name=output_tensor["name"],
        output_dtype=output_tensor["dtype"],
        output_layout=output_tensor["layout"],
        output_shape=tuple(output_tensor["shape"]),
        detector_active_probability_threshold=float(
            field_policy["detector_active_probability_threshold"]),
        detector_min_mean_score=float(field_policy["detector_min_mean_score"]),
        locator_max_width_numerator=field_policy["locator_max_width_numerator"],
        locator_max_width_denominator=field_policy["locator_max_width_denominator"],
        locator_min_width_cells=field_policy["locator_min_width_cells"],
        locator_min_height_cells=field_policy["locator_min_height_cells"],
        locator_min_aspect_numerator=field_policy["locator_min_aspect_numerator"],
        locator_min_aspect_denominator=field_policy["locator_min_aspect_denominator"],
        locator_provisional_min_vertical_overlap_numerator=field_policy[
            "locator_provisional_min_vertical_overlap_numerator"],
        locator_provisional_min_vertical_overlap_denominator=field_policy[
            "locator_provisional_min_vertical_overlap_denominator"],
        locator_provisional_max_height_ratio=field_policy[
            "locator_provisional_max_height_ratio"],
        locator_provisional_max_center_y_delta_shorter_height=field_policy[
            "locator_provisional_max_center_y_delta_shorter_height"],
        locator_corner_edge_divisor=field_policy["locator_corner_edge_divisor"],
        locator_corner_bottom_rows=field_policy["locator_corner_bottom_rows"],
        locator_match_iou_threshold=float(field_policy["locator_match_iou_threshold"]),
        locator_death_grace_observations=field_policy[
            "locator_death_grace_observations"],
        locator_target_horizontal_fallback_max_radius_steps=field_policy[
            "locator_target_horizontal_fallback_max_radius_steps"],
        locator_target_horizontal_step_denominator=field_policy[
            "locator_target_horizontal_step_denominator"],
        locator_target_max_row_iqr_binocular_source_pixels=float(
            field_policy["locator_target_max_row_iqr_binocular_source_pixels"]),
        locator_target_max_row_median_delta_binocular_source_pixels=float(
            field_policy["locator_target_max_row_median_delta_binocular_source_pixels"]),
        locator_target_max_residual_binocular_source_pixels=float(
            field_policy["locator_target_max_residual_binocular_source_pixels"]),
        locator_target_max_unreliable_holds=field_policy[
            "locator_target_max_unreliable_holds"],
        locator_target_deadband_binocular_source_pixels=float(
            field_policy["locator_target_deadband_binocular_source_pixels"]),
        locator_target_ema_alpha=float(field_policy["locator_target_ema_alpha"]),
        locator_target_max_slew_binocular_source_pixels=float(
            field_policy["locator_target_max_slew_binocular_source_pixels"]),
        ocr_safe_row_top=field_policy["ocr_safe_row_top"],
        ocr_safe_row_bottom=field_policy["ocr_safe_row_bottom"],
        source_crop_aspect_width=field_policy["source_crop_aspect_width"],
        source_crop_aspect_height=field_policy["source_crop_aspect_height"],
        text_join_gap_cells=field_policy["text_join_gap_cells"],
        ribbon_join_gap_cells=field_policy["ribbon_join_gap_cells"],
        ribbon_structural_gap_min_cells=field_policy["ribbon_structural_gap_min_cells"],
        ribbon_min_structural_gaps=field_policy["ribbon_min_structural_gaps"],
        ribbon_min_width_numerator=field_policy["ribbon_min_width_numerator"],
        ribbon_min_width_denominator=field_policy["ribbon_min_width_denominator"],
        ribbon_bottom_tolerance_pixels=field_policy["ribbon_bottom_tolerance_pixels"],
        ribbon_bottom_tolerance_projection=(
            field_policy["ribbon_bottom_tolerance_projection"]),
        ribbon_cover_pad_limit=field_policy["ribbon_cover_pad_limit"],
        record_schema=record["schema"],
        record_tag=record["tag"],
        record_word_count=record["word_count"],
        record_header_word_count=record["header_word_count"],
        box_word_count=record["box_word_count"],
        box_flag_ribbon=record["box_flag_ribbon"],
        box_known_flags=record["box_known_flags"],
        raw_box_offset=record["raw_box_offset"],
        raw_box_capacity=record["raw_box_capacity"],
        final_box_offset=record["final_box_offset"],
        final_box_capacity=record["final_box_capacity"],
        locator_schema=locator["schema"],
        locator_tag=locator["tag"],
        locator_word_count=locator["word_count"],
        locator_header_word_count=locator["header_word_count"],
        locator_rectangle_capacity=locator["rectangle_capacity"],
        locator_owner_offset=locator["owner_offset"],
        locator_pending_offset=locator["pending_offset"],
        locator_current_offset=locator["current_offset"],
        locator_kind_word=locator["kind_word"],
        locator_owner_kind_shift=locator["owner_kind_shift"],
        locator_pending_kind_shift=locator["pending_kind_shift"],
        locator_current_kind_shift=locator["current_kind_shift"],
        locator_kind_mask=locator["kind_mask"],
        locator_provisional_current_flag=locator["provisional_current_flag"],
        locator_provisional_target_word=locator["provisional_target_word"],
        locator_provisional_fade_word=locator["provisional_fade_word"],
        condition_param_schema=condition["schema"],
        condition_param_tag=condition["tag"],
        condition_param_word_count=condition["word_count"],
    )


def _shader_implementation(contract: dict[str, Any]) -> ShaderImplementation:
    implementation = contract.get("shader_implementation")
    if (not isinstance(implementation, dict) or
            set(implementation) != SHADER_IMPLEMENTATION_KEYS):
        raise ValueError("depth-coordinate-v2 shader_implementation schema mismatch")
    if (implementation.get("source_closure_schema") != 2 or
            implementation.get("source_compile_flags") != 0x00008800 or
            implementation.get("source_macro_count") != 0):
        raise ValueError("depth-coordinate-v2 shader implementation policy is unsupported")
    entries = implementation.get("source_specs")
    if not isinstance(entries, list):
        raise ValueError("depth-coordinate-v2 shader source specs must be an ordered list")
    parsed: list[tuple[str, str, str]] = []
    for entry in entries:
        if (not isinstance(entry, dict) or
                set(entry) != {"source_file", "source_entrypoint", "source_target"}):
            raise ValueError("depth-coordinate-v2 shader source spec schema mismatch")
        values = tuple(entry[name] for name in (
            "source_file", "source_entrypoint", "source_target"))
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("depth-coordinate-v2 shader source spec values must be strings")
        parsed.append(values)  # type: ignore[arg-type]
    if tuple(parsed) != SHADER_SOURCE_SPECS:
        raise ValueError("depth-coordinate-v2 shader source set is unsupported")
    digest = implementation.get("source_closure_sha256")
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError("depth-coordinate-v2 shader source digest must be lowercase SHA-256")
    return ShaderImplementation(
        source_closure_schema=implementation["source_closure_schema"],
        source_compile_flags=implementation["source_compile_flags"],
        source_macro_count=implementation["source_macro_count"],
        source_specs=tuple(parsed),
        source_closure_sha256=digest,
    )


def _final_parallax_contract(contract: dict[str, Any]) -> FinalParallaxContract:
    value = contract.get("final_parallax")
    if value != EXPECTED_FINAL_PARALLAX:
        raise ValueError(
            "depth-coordinate-v2 final_parallax contract is unsupported")
    return FinalParallaxContract(
        schema=value["schema"],
        authority=value["authority"],
        publication_policy=value["publication_policy"],
        reuse_policy=value["reuse_policy"],
        invalid_policy=value["invalid_policy"],
        current_rgb_policy=value["current_rgb_policy"],
    )


def _model_calibrations(contract: dict[str, Any]) -> tuple[ModelCalibration, ...]:
    entries = contract.get("model_calibrations")
    if not isinstance(entries, list) or not entries:
        raise ValueError("depth-coordinate-v2 model_calibrations must be non-empty")
    result: list[ModelCalibration] = []
    calibration_ids: set[str] = set()
    model_names: set[str] = set()
    model_hashes: set[str] = set()
    for index, entry in enumerate(entries):
        prefix = f"depth-coordinate-v2 model_calibrations[{index}]"
        if not isinstance(entry, dict) or set(entry) != MODEL_CALIBRATION_KEYS:
            raise ValueError(f"{prefix} schema mismatch")
        preprocess = entry.get("preprocess")
        if not isinstance(preprocess, dict) or set(preprocess) != PREPROCESS_KEYS:
            raise ValueError(f"{prefix}.preprocess schema mismatch")
        string_fields = (
            "calibration_id", "depth_model", "depth_model_url", "onnx_sha256")
        if any(not isinstance(entry.get(name), str) or not entry[name]
               for name in string_fields):
            raise ValueError(f"{prefix} identity strings must be non-empty")
        if (len(entry["onnx_sha256"]) != 64 or
                any(character not in "0123456789abcdef" for character in entry["onnx_sha256"])):
            raise ValueError(f"{prefix}.onnx_sha256 must be lowercase SHA-256")
        scale = entry.get("raw_coordinate_scale")
        if (not isinstance(scale, (int, float)) or isinstance(scale, bool) or
                not math.isfinite(float(scale)) or float(scale) <= 0.0):
            raise ValueError(f"{prefix}.raw_coordinate_scale must be finite and positive")
        for name in ("profile", "dtype", "layout", "stage"):
            if not isinstance(preprocess.get(name), str) or not preprocess[name]:
                raise ValueError(f"{prefix}.preprocess.{name} must be a non-empty string")
        source_closure_sha256 = preprocess.get("source_closure_sha256")
        if (not isinstance(source_closure_sha256, str) or len(source_closure_sha256) != 64 or
                any(character not in "0123456789abcdef"
                    for character in source_closure_sha256)):
            raise ValueError(
                f"{prefix}.preprocess.source_closure_sha256 must be lowercase SHA-256")
        if (preprocess.get("source_closure_schema") != 2 or
                preprocess.get("source_file") != "rgb_to_nchw_cs.hlsl" or
                preprocess.get("source_entrypoint") != "main" or
                preprocess.get("source_target") != "cs_5_0" or
                preprocess.get("source_compile_flags") != 0x00008800 or
                preprocess.get("source_macro_count") != 0):
            raise ValueError(f"{prefix}.preprocess shader source identity is unsupported")
        if preprocess.get("model_input_schema") != 1:
            raise ValueError(f"{prefix}.preprocess.model_input_schema must be one")
        if preprocess.get("dtype") != "float32-le" or preprocess.get("layout") != "NCHW":
            raise ValueError(f"{prefix}.preprocess tensor semantics are unsupported")
        channels = preprocess.get("channels")
        if channels != ["R", "G", "B"]:
            raise ValueError(f"{prefix}.preprocess.channels must be RGB")
        patch = preprocess.get("patch_multiple")
        maximum = preprocess.get("maximum_dimension")
        if (not isinstance(patch, int) or isinstance(patch, bool) or patch <= 0 or
                not isinstance(maximum, int) or isinstance(maximum, bool) or
                maximum < patch or maximum % patch):
            raise ValueError(f"{prefix}.preprocess spatial contract is invalid")
        shape_entries = entry.get("calibrated_input_shapes")
        if not isinstance(shape_entries, list) or not shape_entries:
            raise ValueError(f"{prefix}.calibrated_input_shapes must be non-empty")
        calibrated_shapes: list[tuple[int, int]] = []
        for shape in shape_entries:
            if (not isinstance(shape, dict) or set(shape) != {"width", "height"} or
                    not isinstance(shape.get("width"), int) or
                    isinstance(shape.get("width"), bool) or shape["width"] <= 0 or
                    not isinstance(shape.get("height"), int) or
                    isinstance(shape.get("height"), bool) or shape["height"] <= 0 or
                    shape["width"] > maximum or shape["height"] > maximum or
                    shape["width"] % patch or shape["height"] % patch):
                raise ValueError(f"{prefix}.calibrated_input_shapes is invalid")
            calibrated_shapes.append((shape["width"], shape["height"]))
        if len(set(calibrated_shapes)) != len(calibrated_shapes):
            raise ValueError(f"{prefix}.calibrated_input_shapes contains duplicates")
        vectors: dict[str, tuple[float, float, float]] = {}
        for name in ("imagenet_mean", "imagenet_std"):
            values = preprocess.get(name)
            if (not isinstance(values, list) or len(values) != 3 or
                    any(not isinstance(value, (int, float)) or isinstance(value, bool) or
                        not math.isfinite(float(value)) for value in values) or
                    (name == "imagenet_std" and any(float(value) <= 0.0 for value in values))):
                raise ValueError(f"{prefix}.preprocess.{name} is invalid")
            vectors[name] = tuple(float(value) for value in values)  # type: ignore[assignment]
        if (entry["calibration_id"] in calibration_ids or
                entry["depth_model"] in model_names or
                entry["onnx_sha256"] in model_hashes):
            raise ValueError(
                "depth-coordinate-v2 calibration ids, model names, and hashes must be unique")
        calibration_ids.add(entry["calibration_id"])
        model_names.add(entry["depth_model"])
        model_hashes.add(entry["onnx_sha256"])
        result.append(ModelCalibration(
            calibration_id=entry["calibration_id"],
            depth_model=entry["depth_model"],
            depth_model_url=entry["depth_model_url"],
            onnx_sha256=entry["onnx_sha256"],
            raw_coordinate_scale=float(scale),
            calibrated_input_shapes=tuple(calibrated_shapes),
            preprocess=ModelInputContract(
                profile=preprocess["profile"],
                source_closure_schema=preprocess["source_closure_schema"],
                source_file=preprocess["source_file"],
                source_entrypoint=preprocess["source_entrypoint"],
                source_target=preprocess["source_target"],
                source_compile_flags=preprocess["source_compile_flags"],
                source_macro_count=preprocess["source_macro_count"],
                source_closure_sha256=source_closure_sha256,
                model_input_schema=preprocess["model_input_schema"],
                dtype=preprocess["dtype"],
                layout=preprocess["layout"],
                channels=tuple(channels),
                patch_multiple=patch,
                maximum_dimension=maximum,
                imagenet_mean=vectors["imagenet_mean"],
                imagenet_std=vectors["imagenet_std"],
                stage=preprocess["stage"],
            ),
        ))
    return tuple(result)


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the manifest and fail closed if its calibrated-default surface is malformed."""

    with path.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    provenance = contract.get("capture_provenance") if isinstance(contract, dict) else None
    if (not isinstance(provenance, dict) or
            set(provenance) != {"schema", "manifest_key", "binding"} or
            not isinstance(provenance.get("schema"), int) or
            isinstance(provenance.get("schema"), bool) or provenance["schema"] < 1 or
            any(not isinstance(provenance.get(name), str) or not provenance[name]
                for name in ("manifest_key", "binding"))):
        raise ValueError("depth-coordinate-v2 capture_provenance schema mismatch")
    defaults = contract.get("calibrated_defaults") if isinstance(contract, dict) else None
    if not isinstance(defaults, dict) or set(defaults) != set(CALIBRATED_DEFAULT_NAMES):
        raise ValueError("depth-coordinate-v2 calibrated_defaults schema mismatch")
    for name in CALIBRATED_DEFAULT_NAMES:
        value = defaults[name]
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value))):
            raise ValueError(f"depth-coordinate-v2 default {name} must be finite")
        if name == "convergence_curve_default":
            if float(value) != 0.0:
                raise ValueError("convergence_curve_default must be exactly zero")
        elif float(value) <= 0.0:
            raise ValueError(f"depth-coordinate-v2 default {name} must be positive")
    if defaults["max_horizontal_slope"] >= 1.0:
        raise ValueError("max_horizontal_slope must be below one")
    majorant_share = _float32(defaults["vertical_majorant_share"])
    minorant_share = _float32(_float32(1.0) - majorant_share)
    if majorant_share <= 0.0 or minorant_share <= 0.0:
        raise ValueError(
            "vertical_majorant_share and its complement must remain positive in float32")
    _shader_implementation(contract)
    _final_parallax_contract(contract)
    _subtitle_ocr_contract(contract)
    _model_calibrations(contract)
    return contract


_CONTRACT = load_contract()
CONTRACT_CANONICAL_SHA256 = hashlib.sha256(json.dumps(
    _CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("ascii")).hexdigest()
CALIBRATED_DEFAULTS = CalibratedDefaults(
    **{name: float(_CONTRACT["calibrated_defaults"][name])
       for name in CALIBRATED_DEFAULT_NAMES}
)
CONTRACT_SCHEMA = int(_CONTRACT["schema"])
CAPTURE_PROVENANCE_SCHEMA = int(_CONTRACT["capture_provenance"]["schema"])
CAPTURE_PROVENANCE_KEY = str(_CONTRACT["capture_provenance"]["manifest_key"])
CAPTURE_PROVENANCE_BINDING = str(_CONTRACT["capture_provenance"]["binding"])
SHADER_IMPLEMENTATION = _shader_implementation(_CONTRACT)
FINAL_PARALLAX = _final_parallax_contract(_CONTRACT)
SUBTITLE_OCR = _subtitle_ocr_contract(_CONTRACT)
MODEL_CALIBRATIONS = _model_calibrations(_CONTRACT)


def subtitle_ocr_field_is_calibrated(field_width: int, field_height: int) -> bool:
    """Return whether a subtitle field is one of the DAV2 model's calibrated shapes."""

    return any(
        (field_width, field_height) in calibration.calibrated_input_shapes
        for calibration in MODEL_CALIBRATIONS
    )


def subtitle_target_is_representable_source_u(target: float) -> bool:
    """Return whether an SLR13 target fits the authenticated signed parallax container."""

    value = _float32(target)
    return math.isfinite(value) and abs(value) <= _float32(
        CALIBRATED_DEFAULTS.direct_container_limit)


def _subtitle_row_observation(
        samples: tuple[float, ...], binocular_scale: float) -> tuple[bool, bool, float]:
    """Return ``(valid, coherent, median)`` in SLR13's SM5 float32 order."""

    converted: list[float] = []
    for sample in samples:
        try:
            value = _float32(sample)
        except (OverflowError, TypeError, ValueError, struct.error):
            return False, False, 0.0
        if not subtitle_target_is_representable_source_u(value):
            return False, False, 0.0
        converted.append(value)
    converted.sort()

    median = _float32(_float32(converted[7] + converted[8]) * _float32(0.5))
    # Match the explicit precise HLSL quantiles: q1=.5*(v3+v4), q3=.5*(v11+v12).
    q1 = _float32(_float32(0.5) * _float32(converted[3] + converted[4]))
    q3 = _float32(_float32(0.5) * _float32(converted[11] + converted[12]))
    iqr = _float32(q3 - q1)
    iqr_pixels = _float32(iqr * binocular_scale)
    coherent = (
        math.isfinite(iqr_pixels) and
        iqr_pixels <= _float32(
            SUBTITLE_OCR.locator_target_max_row_iqr_binocular_source_pixels)
    )
    return True, coherent, median


def select_subtitle_local_plane_source_u(
        first_row: tuple[float, ...] | list[float],
        second_row: tuple[float, ...] | list[float],
        source_width: int) -> Optional[float]:
    """Select SLR13's reliable local supporting plane from two independent rows.

    Each row contains exactly 16 R32_FLOAT Base samples. A row is valid only when all of its
    samples are finite and fit the signed direct-parallax container, and is coherent when its
    Tukey IQR is at most the generated binocular-pixel threshold. Two valid primary rows use their
    robust medians without an IQR gate: close medians are averaged and separated medians select the
    larger source-U value (the nearer supporting plane). With only one valid row, that row must
    still be coherent before its median can stand in for the missing independent observation.

    All state-affecting arithmetic and comparisons are rounded to float32 in the same explicit
    order as the shader. The thresholds use only addition, subtraction, multiplication, and the
    exact power-of-two factor 0.5, so unlike reciprocal-based coordinate bounds there is no
    alternate SM5 division candidate to admit at a boundary.
    """

    if (isinstance(source_width, bool) or not isinstance(source_width, int) or
            source_width <= 0):
        raise ValueError("subtitle target source width must be a positive integer")
    if len(first_row) != 16 or len(second_row) != 16:
        raise ValueError("subtitle target evidence must contain exactly 16 samples per row")
    binocular_scale = _float32(_float32(2.0) * _float32(source_width))
    if not math.isfinite(binocular_scale) or binocular_scale <= 0.0:
        raise ValueError("subtitle target binocular scale must be finite and positive")

    first_valid, first_coherent, first_median = _subtitle_row_observation(
        tuple(first_row), binocular_scale)
    second_valid, second_coherent, second_median = _subtitle_row_observation(
        tuple(second_row), binocular_scale)
    if first_valid and second_valid:
        median_delta = _float32(second_median - first_median)
        median_delta_pixels = _float32(abs(median_delta) * binocular_scale)
        if median_delta_pixels <= _float32(
                SUBTITLE_OCR.locator_target_max_row_median_delta_binocular_source_pixels):
            return _float32(
                _float32(first_median + second_median) * _float32(0.5))
        return max(first_median, second_median)
    if first_coherent:
        return first_median
    if second_coherent:
        return second_median
    return None


def select_subtitle_local_plane_fallback_probe_source_u(
        first_row: Sequence[float],
        second_row: Sequence[float],
        source_width: int) -> Optional[float]:
    """Select one strict fallback probe from two mutually consistent coherent rows."""

    if (isinstance(source_width, bool) or not isinstance(source_width, int) or
            source_width <= 0):
        raise ValueError("subtitle target source width must be a positive integer")
    if len(first_row) != 16 or len(second_row) != 16:
        raise ValueError("subtitle target evidence must contain exactly 16 samples per row")
    binocular_scale = _float32(_float32(2.0) * _float32(source_width))
    if not math.isfinite(binocular_scale) or binocular_scale <= 0.0:
        raise ValueError("subtitle target binocular scale must be finite and positive")
    first_valid, first_coherent, first_median = _subtitle_row_observation(
        tuple(first_row), binocular_scale)
    second_valid, second_coherent, second_median = _subtitle_row_observation(
        tuple(second_row), binocular_scale)
    median_delta_pixels = _float32(
        abs(_float32(second_median - first_median)) * binocular_scale)
    if (not first_valid or not second_valid or not first_coherent or not second_coherent or
            median_delta_pixels > _float32(
                SUBTITLE_OCR.locator_target_max_row_median_delta_binocular_source_pixels)):
        return None
    return _float32(_float32(first_median + second_median) * _float32(0.5))


def subtitle_target_fallback_strip_is_unclamped(
        center: float, content_left: int, content_right: int) -> bool:
    """Return whether a fallback's rounded 16-sample strip lies in half-open content bounds."""

    if (isinstance(content_left, bool) or not isinstance(content_left, int) or
            isinstance(content_right, bool) or not isinstance(content_right, int) or
            content_left < 0 or content_left >= content_right or content_right > 0xffff):
        raise ValueError("subtitle target content bounds must be canonical field coordinates")
    try:
        center_f32 = _float32(center)
    except (OverflowError, TypeError, ValueError, struct.error):
        return False
    if not math.isfinite(center_f32):
        return False
    first_x = _float32(center_f32 - _float32(30.0))
    last_x = _float32(first_x + _float32(4.0 * 15.0))
    rounded_first = math.floor(_float32(first_x + _float32(0.5)))
    rounded_last = math.floor(_float32(last_x + _float32(0.5)))
    return rounded_first >= content_left and rounded_last < content_right


def subtitle_target_probe_centers(
        cores: Sequence[tuple[int, int, int, int]],
        kinds: Sequence[int]) -> tuple[float, ...]:
    """Return SLR13's aggregate primary and bounded near-center fallback positions.

    ``kinds`` uses the compact locator convention: zero is ordinary text and one is a ribbon.
    The primary remains the median of all owner-member centers. The fallback step is one sixteenth
    of the ordinary-core horizontal span when any ordinary member exists, otherwise one sixteenth
    of the whole owner span. Positions are returned in shader order: primary, ``-step``, ``+step``,
    ``-2*step``, ``+2*step``.
    """

    if (len(cores) == 0 or len(cores) > SUBTITLE_OCR.locator_rectangle_capacity or
            len(kinds) != len(cores)):
        raise ValueError("subtitle target placement requires one canonical owner stack")
    normalized: list[tuple[int, int, int, int]] = []
    normalized_kinds: list[int] = []
    for rectangle, kind in zip(cores, kinds):
        if (len(rectangle) != 4 or
                any(isinstance(value, bool) or not isinstance(value, int)
                    for value in rectangle)):
            raise ValueError("subtitle target cores must contain four integer coordinates")
        left, top, right, bottom = rectangle
        if (left < 0 or top < 0 or left >= right or top >= bottom or
                right > 0xffff or bottom > 0xffff):
            raise ValueError("subtitle target core is outside the authenticated field domain")
        if isinstance(kind, bool) or not isinstance(kind, int) or kind not in (0, 1):
            raise ValueError("subtitle target kind must be ordinary text or ribbon")
        normalized.append((left, top, right, bottom))
        normalized_kinds.append(kind)

    member_centers = sorted(
        _float32(_float32(left + right - 1) * _float32(0.5))
        for left, _top, right, _bottom in normalized
    )
    member_count = len(member_centers)
    if member_count & 1:
        primary = member_centers[member_count // 2]
    else:
        primary = _float32(
            _float32(member_centers[member_count // 2 - 1] +
                     member_centers[member_count // 2]) * _float32(0.5))

    placement = [
        rectangle for rectangle, kind in zip(normalized, normalized_kinds) if kind == 0
    ]
    if not placement:
        placement = normalized
    span = max(rectangle[2] for rectangle in placement) - min(
        rectangle[0] for rectangle in placement)
    step = _float32(
        _float32(span) /
        _float32(SUBTITLE_OCR.locator_target_horizontal_step_denominator))
    centers = [primary]
    fallback_count = 2 * SUBTITLE_OCR.locator_target_horizontal_fallback_max_radius_steps
    for fallback_index in range(fallback_count):
        radius = fallback_index // 2 + 1
        offset = _float32(step * _float32(radius))
        centers.append(_float32(
            primary - offset if (fallback_index & 1) == 0 else primary + offset))
    return tuple(centers)


def select_subtitle_local_plane_with_fallback_source_u(
        primary_first_row: Sequence[float],
        primary_second_row: Sequence[float],
        fallback_probes: Sequence[tuple[Sequence[float], Sequence[float]]],
        fallback_centers: Sequence[float],
        content_bounds: tuple[int, int],
        source_width: int) -> Optional[float]:
    """Select the primary or the first mutually consistent strict fallback radius.

    A reliable primary short-circuits exactly. Fallback is considered only after primary failure;
    every candidate needs an interior, unclamped strip and two coherent rows whose medians differ
    by at most four pixels. A sole valid candidate wins its radius. Two valid candidates must also
    differ by at most four pixels, then the larger-U target wins. A conflicting radius fails the
    complete observation without searching farther. ``fallback_probes`` and ``fallback_centers``
    exclude the primary and follow ``-radius, +radius`` order.
    """

    primary = select_subtitle_local_plane_source_u(
        list(primary_first_row), list(primary_second_row), source_width)
    if primary is not None:
        return primary
    expected_fallbacks = (
        2 * SUBTITLE_OCR.locator_target_horizontal_fallback_max_radius_steps)
    if len(fallback_probes) != expected_fallbacks or len(fallback_centers) != expected_fallbacks:
        raise ValueError(
            f"subtitle target fallback requires exactly {expected_fallbacks} probes")
    if len(content_bounds) != 2:
        raise ValueError("subtitle target fallback requires half-open content bounds")
    content_left, content_right = content_bounds
    binocular_scale = _float32(_float32(2.0) * _float32(source_width))
    for pair_base in range(0, expected_fallbacks, 2):
        pair_targets: list[Optional[float]] = []
        for probe, center in zip(
                fallback_probes[pair_base:pair_base + 2],
                fallback_centers[pair_base:pair_base + 2]):
            if len(probe) != 2:
                raise ValueError("subtitle target fallback probe must contain two rows")
            candidate = None
            if subtitle_target_fallback_strip_is_unclamped(
                    center, content_left, content_right):
                candidate = select_subtitle_local_plane_fallback_probe_source_u(
                    probe[0], probe[1], source_width)
            pair_targets.append(candidate)
        negative, positive = pair_targets
        if negative is not None and positive is not None:
            pair_delta_pixels = _float32(
                abs(_float32(positive - negative)) * binocular_scale)
            if pair_delta_pixels > _float32(
                    SUBTITLE_OCR.locator_target_max_row_median_delta_binocular_source_pixels):
                return None
            return max(negative, positive)
        if negative is not None:
            return negative
        if positive is not None:
            return positive
    return None


def subtitle_ocr_project_row_ceil(
        source_width: int, source_height: int,
        field_width: int, field_height: int,
        detector_y: int) -> Optional[int]:
    """Project one detector row edge by the authenticated exact rational ceil rule."""

    if (source_width <= 0 or source_height <= 0 or
            detector_y < 0 or detector_y > SUBTITLE_OCR.output_shape[2] or
            not subtitle_ocr_field_is_calibrated(field_width, field_height)):
        return None
    crop_height = min(
        source_height,
        (source_width * SUBTITLE_OCR.source_crop_aspect_height +
         SUBTITLE_OCR.source_crop_aspect_width - 1) //
        SUBTITLE_OCR.source_crop_aspect_width,
    )
    crop_top = source_height - crop_height
    denominator = SUBTITLE_OCR.output_shape[2] * source_height
    numerator = (
        (crop_top * SUBTITLE_OCR.output_shape[2] + detector_y * crop_height) *
        field_height
    )
    return min(
        numerator // denominator + int(numerator % denominator != 0),
        field_height,
    )


def subtitle_ocr_ribbon_min_bottom(
        source_width: int, source_height: int,
        field_width: int, field_height: int) -> Optional[int]:
    """Return the inclusive field-row lower bound for a ribbon core's bottom edge."""

    return subtitle_ocr_project_row_ceil(
        source_width, source_height, field_width, field_height,
        SUBTITLE_OCR.ocr_safe_row_bottom -
        SUBTITLE_OCR.ribbon_bottom_tolerance_pixels,
    )


def subtitle_ocr_dynamic_roi(
        source_width: int,
        source_height: int,
        field_width: int,
        field_height: int) -> Optional[tuple[int, int]]:
    """Project the authenticated safe OCR-row interval into one calibrated DAV2 field."""

    top = subtitle_ocr_project_row_ceil(
        source_width, source_height, field_width, field_height,
        SUBTITLE_OCR.ocr_safe_row_top)
    bottom = subtitle_ocr_project_row_ceil(
        source_width, source_height, field_width, field_height,
        SUBTITLE_OCR.ocr_safe_row_bottom)
    if top is None or bottom is None or top >= bottom:
        return None
    return top, bottom


def find_model_calibration(
        depth_model: str, depth_model_url: str, onnx_sha256: str) -> Optional[ModelCalibration]:
    """Return the one calibration bound to an exact model name, URL, and ONNX digest."""

    matches = [
        calibration for calibration in MODEL_CALIBRATIONS
        if calibration.depth_model == depth_model and
        calibration.depth_model_url == depth_model_url and
        calibration.onnx_sha256 == onnx_sha256
    ]
    if len(matches) > 1:
        raise ValueError("depth-coordinate-v2 model calibration identity is ambiguous")
    return matches[0] if matches else None


__all__ = [
    "CALIBRATED_DEFAULTS",
    "CALIBRATED_DEFAULT_NAMES",
    "CONTRACT_PATH",
    "CONTRACT_CANONICAL_SHA256",
    "CONTRACT_SCHEMA",
    "EXPECTED_SUBTITLE_OCR",
    "CAPTURE_PROVENANCE_BINDING",
    "CAPTURE_PROVENANCE_KEY",
    "CAPTURE_PROVENANCE_SCHEMA",
    "CalibratedDefaults",
    "EXPECTED_FINAL_PARALLAX",
    "FINAL_PARALLAX",
    "FinalParallaxContract",
    "MODEL_CALIBRATIONS",
    "MODEL_CALIBRATION_KEYS",
    "ModelCalibration",
    "ModelInputContract",
    "PREPROCESS_KEYS",
    "SHADER_IMPLEMENTATION",
    "SHADER_IMPLEMENTATION_KEYS",
    "SHADER_SOURCE_SPECS",
    "ShaderImplementation",
    "SUBTITLE_OCR",
    "SubtitleOcrContract",
    "find_model_calibration",
    "load_contract",
    "subtitle_ocr_dynamic_roi",
    "subtitle_ocr_field_is_calibrated",
    "subtitle_ocr_project_row_ceil",
    "subtitle_ocr_ribbon_min_bottom",
    "select_subtitle_local_plane_fallback_probe_source_u",
    "select_subtitle_local_plane_source_u",
    "select_subtitle_local_plane_with_fallback_source_u",
    "subtitle_target_fallback_strip_is_unclamped",
    "subtitle_target_probe_centers",
    "subtitle_target_is_representable_source_u",
]
