#!/usr/bin/env python3
"""Read-only Python view of the generated depth-coordinate-v2 calibration contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Optional


CONTRACT_PATH = (
    Path(__file__).resolve().parent / "contracts" / "depth-coordinate-v2-v1.json"
)
CALIBRATED_DEFAULT_NAMES = (
    "collapse_abs_epsilon",
    "adaptive_policy_id",
    "gain_per_pop",
    "reference_pop_strength",
    "direct_container_limit",
    "max_horizontal_slope",
    "max_vertical_shear",
    "vertical_majorant_share",
    "convergence_curve_default",
    "adaptive_time_constant_seconds",
    "adaptive_log_scale_rate",
    "adaptive_zero_budget_per_second",
    "adaptive_max_tick_gap_ms",
    "host_percentile_bin_count",
    "host_percentile_low",
    "host_percentile_high",
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
    ("rgb_to_nchw_near_identical_cs.hlsl", "fused_main", "cs_5_0"),
    ("buffer_to_tex_cs.hlsl", "main", "cs_5_0"),
    ("buffer_to_tex_cs.hlsl", "pad_main", "cs_5_0"),
    ("depth_minmax_ema_cs.hlsl", "main", "cs_5_0"),
    ("depth_scene_cut_evidence_cs.hlsl", "main", "cs_5_0"),
    ("depth_scene_cut_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_moments_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_frame_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_histogram_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_quantiles_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_state_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_map_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_vertical_limit_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_limit_cs.hlsl", "main", "cs_5_0"),
    ("host_sbs_ocr_preprocess_cs.hlsl", "main", "cs_5_0"),
    ("host_sbs_ocr_boxes_cs.hlsl", "cells_main", "cs_5_0"),
    ("host_sbs_ocr_boxes_cs.hlsl", "resolve_main", "cs_5_0"),
    ("host_sbs_subtitle_locator_cs.hlsl", "resolve_main", "cs_5_0"),
    ("host_sbs_subtitle_locator_cs.hlsl", "condition_main", "cs_5_0"),
    ("host_sbs_subtitle_contour_cs.hlsl", "temporal_main", "cs_5_0"),
    ("host_sbs_subtitle_contour_cs.hlsl", "project_main", "cs_5_0"),
    ("host_sbs_subtitle_contour_cs.hlsl", "history_main", "cs_5_0"),
    ("host_sbs_subtitle_contour_distance_cs.hlsl", "horizontal_main", "cs_5_0"),
    ("host_sbs_subtitle_contour_distance_cs.hlsl", "vertical_main", "cs_5_0"),
    ("host_sbs_subtitle_contour_cs.hlsl", "snapshot_main", "cs_5_0"),
    ("depth_coordinate_v2_vertical_limit_cs.hlsl", "subtitle_main", "cs_5_0"),
    ("depth_coordinate_v2_limit_cs.hlsl", "subtitle_main", "cs_5_0"),
)
EXPECTED_FINAL_PARALLAX = {'schema': 3,
 'authority': 'complete-atomic-subtitle-conditioned-r32f-live-render-authority',
 'publication_policy': 'authenticated-infer-or-cpu-known-publication-direct-render',
 'reuse_policy': 'reuse-holds-complete-depth-ocr-uip4-final-tuple-byte-for-byte',
 'invalid_policy': 'fail-closed-flat',
 'current_rgb_policy': 'always-current-never-retained'}
EXPECTED_SUBTITLE_OCR = {'schema': 27,
 'logical_model': 'ppocrv6_tiny_det_modelopt_fp16',
 'asset_path': 'models/ppocrv6_tiny_det_modelopt045_mixed_fp16_fp32io.onnx',
 'artifact_onnx_sha256': '169a233ba0ff7cac27f8ec7dccb6a406e614b25b21fe6a5638c423bf2118bb44',
 'source_url': 'https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/2ba1506c0380b8f0b03dd142459aac66d4421f6c/inference.onnx?download=true',
 'source_onnx_sha256': '193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8',
 'conversion_tool': 'nvidia-modelopt',
 'conversion_version': '0.45.0',
 'conversion_recipe': 'nvidia-modelopt-autocast-fp16-keep-io-fp32-v1',
 'conversion_calibration_profile': 'apollo-live8-bottom960x160-v1',
 'engine_recipe': 'trt-strong-modelopt045-fp16-iofp32-tf32-fixed960x160-level5-v2',
 'preprocess_profile': 'apollo-ppocrv6-bottom-6x1-bgr-imagenet-v1',
 'source_crop': 'bottom-6:1',
 'input_tensor': {'name': 'x',
                  'dtype': 'float32',
                  'layout': 'NCHW',
                  'shape': [1, 3, 160, 960],
                  'channels': ['B', 'G', 'R'],
                  'imagenet_mean': [0.485, 0.456, 0.406],
                  'imagenet_std': [0.229, 0.224, 0.225]},
 'output_tensor': {'name': 'fetch_name_0',
                   'dtype': 'float32',
                   'layout': 'NCHW',
                   'shape': [1, 1, 160, 960]},
 'field_policy': {'live_field_shapes': [{'width': 1540, 'height': 868},
                                        {'width': 2044, 'height': 868},
                                        {'width': 2072, 'height': 868},
                                        {'width': 868, 'height': 1540},
                                        {'width': 868, 'height': 2044},
                                        {'width': 868, 'height': 2072},
                                        {'width': 1148, 'height': 868},
                                        {'width': 1232, 'height': 868},
                                        {'width': 1260, 'height': 868},
                                        {'width': 1316, 'height': 868},
                                        {'width': 1400, 'height': 868},
                                        {'width': 1736, 'height': 868},
                                        {'width': 1876, 'height': 868},
                                        {'width': 1932, 'height': 868},
                                        {'width': 1960, 'height': 868},
                                        {'width': 868, 'height': 1148},
                                        {'width': 868, 'height': 1232},
                                        {'width': 868, 'height': 1260},
                                        {'width': 868, 'height': 1316},
                                        {'width': 868, 'height': 1400},
                                        {'width': 868, 'height': 1736},
                                        {'width': 868, 'height': 1876},
                                        {'width': 868, 'height': 1932},
                                        {'width': 868, 'height': 1960}],
                  'detector_active_probability_threshold': 0.2,
                  'detector_min_mean_score': 0.4,
                  'ocr_safe_row_top': 24,
                  'ocr_safe_row_bottom': 155,
                  'source_crop_aspect_width': 6,
                  'source_crop_aspect_height': 1,
                  'text_join_gap_cells': 4,
                  'ordinary_cover_horizontal_guard_cells': 1,
                  'ordinary_cover_horizontal_guard_policy': 'ordinary-only-add-guard-cells-times-producer-ocr-cell-width-to-horizontal-db-unclip-cover-clamped-detector-bounds-before-map-current-canonical-cover-all-consumers-no-second-mask-vertical-and-ribbon-unchanged-v1',
                  'line_group_min_vertical_iou_numerator': 1,
                  'line_group_min_vertical_iou_denominator': 2,
                  'line_group_max_horizontal_gap_height_numerator': 4,
                  'line_group_max_horizontal_gap_height_denominator': 1,
                  'line_group_horizontal_gap_metric': 'native-source-horizontal-gap-to-current-union-edge-relative-to-max-candidate-and-fixed-anchor-height-exact-cross-multiplied-v1',
                  'line_center_tolerance_numerator': 1,
                  'line_center_tolerance_denominator': 10,
                  'line_search_bottom_height_numerator': 1,
                  'line_search_bottom_height_denominator': 4,
                  'line_search_region': 'lower-quarter-of-center-reference-intersect-existing-safe-detector-crop-v1',
                  'line_group_policy': 'same-detector-active-row-band-first-qualifying-fixed-tight-core-anchor-span-iou-and-bounded-native-horizontal-union-gap-before-capacity-union-eligibility-only-original-ordinary-and-ribbon-pairs-v2',
                  'line_center_reference': 'video-roi-otherwise-full-screen-horizontal-center-and-width-in-analysis-source-pixels-v1',
                  'line_center_reference_params': 'ocr-resolve-prefix16-then-signed-twice-center-unsigned-width-signed-search-top-zero-reserved-word-80bytes-v1',
                  'ribbon_join_gap_cells': 12,
                  'ribbon_structural_gap_min_cells': 3,
                  'ribbon_min_structural_gaps': 3,
                  'ribbon_min_width_numerator': 1,
                  'ribbon_min_width_denominator': 2,
                  'ribbon_bottom_tolerance_pixels': 2,
                  'ribbon_bottom_tolerance_projection': 'exact-ceil-detector-edge-through-bottom-crop-v1',
                  'ribbon_cover_pad_limit': 8},
 'candidate_selection': {'policy': 'bounded-largest-core-area-v1',
                         'capacity': 'min-raw-and-final-box-capacity',
                         'ranking': 'tight-core-field-area-descending',
                         'tie_break': 'earlier-raster-discovery',
                         'publication_order': 'raster-discovery-order',
                         'overflow': 'retain-best-paired-candidates'},
 'ocr_record': {'schema': 3,
                'tag': 944915279,
                'word_count': 208,
                'header_word_count': 16,
                'box_word_count': 8,
                'box_flag_ribbon': 1,
                'box_known_flags': 1,
                'raw_box_offset': 16,
                'raw_box_capacity': 16,
                'final_box_offset': 144,
                'final_box_capacity': 8},
 'condition_params': {'schema': 4,
                      'word_count': 6,
                      'current_count_word': 2,
                      'current_kind_mask_word': 3,
                      'active_word': 4,
                      'fully_active_value': 2,
                      'target_word': 5,
                      'tag': 877675347},
 'geometry_authority': 'same-observation-sealed-parallax-state-required-for-ui-probe-and-time-advance-v1',
 'contour_policy': {'schema': 5,
                    'recipe': 23,
                    'source': 'same-joined-ocr-source-bottom-crop-display-srgb-finite-rgba16f-v1',
                    'source_max_extent': 5120,
                    'source_extent_policy': 'native-exact-crop-for-supported-source-otherwise-canonical-1x1-empty-support-history-source-and-zero-probability-v1',
                    'history_jitter_radius_source_pixels': 2,
                    'probability_validity_policy': 'all-interpolation-contributors-finite-in-unit-range-even-without-fresh-threshold-v1',
                    'diagnostic_snapshot_policy': 'same-joined-valid-ocr-geometry-source-clock-copy-before-next-admission-clear-invalid-whole-reuse-hold-v1',
                    'native_support': 'r32u-binary-current-temporally-stable-ui-support-v1',
                    'authority': 'current-ocr8-uip4-scp4-cut-source-clock-identity-only-v2',
                    'projection': 'native-visible-support-to-all-bilinear-field-nodes-integer-footprint-v1',
                    'history': 'current-cover-fixed-rgb-anchor-four-genuine-confirmations-same-pixel-first-current-core-probability-cue-blocks-neighbor-fallback-v4',
                    'history_schema': 5,
                    'history_tag': 894714707,
                    'history_word_count': 32,
                    'conditioning': 'post-gaussian-shared-vertical75-25-and-horizontal-q30-no-presentation-support-or-invalid-current-authority-exact-base-v2',
                    'reuse': 'whole-depth-ocr-uip4-native-mask-field-pin-q-gaussian-weight-blend-vertical-control-history-source-probability-final-tuple-held-v1',
                    'uncertain_support': 'raw-support-abstains-presentation-current-cover-fallback-until-four-genuine-cue-free-observations-v3',
                    'history_snapshot_valid_word': 23,
                    'history_feature_valid_bit': 2,
                    'history_feature_mask_bit': 1,
                    'history_rgb_distance': 0.08,
                    'mask_evidence_policy': 'current-ocr8-cover-color-agnostic-four-genuine-fixed-anchor-matches-same-pixel-first-probability-cue-blocks-neighbor-fallback-in-current-raw-core-v8',
                    'history_confirmation_observations': 4,
                    'history_feature_count_shift': 2,
                    'history_anchor_policy': 'same-pixel-qualified-fixed-anchor-first-even-with-current-core-probability-cue-otherwise-unflagged-min-rgb-distance-then-squared-spatial-distance-then-scan-order-v3',
                    'distance_policy': 'source-pixel-five-center-euclidean-squared-r32f-positive-infinity-v1',
                    'distance_empty_bits': 2139095040,
                    'gaussian_center_offset_source_pixels': 2,
                    'gaussian_sigma_short_side_divisor': 60.0,
                    'gaussian_weight_policy': 'fp32-exp-negative-min-native-q-current-cover-q-until-four-genuine-cue-free-observations-max-union-v3',
                    'gaussian_weight_absolute_error_limit': 2e-06,
                    'gaussian_blend_policy': 'precise-ui-times-M-plus-base-times-one-minus-M-bit-exact-endpoints-v1',
                    'history_reserved_offset': 24,
                    'history_reserved_word_count': 8,
                    'probability_change_ratio': 0.125,
                    'probability_change_policy': 'current-core-native-center-soft-probability-unit-mass-spatial-tv-strict-ratio-cue-both-zero-no-cue-one-zero-cue-v3'},
 'plane_state': {'schema': 4,
                 'word_count': 40,
                 'header_word_count': 24,
                 'cover_capacity': 8,
                 'flags_word': 2,
                 'current_valid_flag': 1,
                 'current_slot_mask_word': 3,
                 'frame_offset': 4,
                 'analysis_generation_offset': 6,
                 'observation_timestamp_offset': 8,
                 'source_extent_offset': 10,
                 'field_extent_offset': 12,
                 'roi_offset': 14,
                 'content_offset': 16,
                 'cut_epoch_word': 20,
                 'reserved_offset': 23,
                 'reserved_word_count': 1,
                 'adaptive_offset': 24,
                 'adaptive_word_count': 16,
                 'tag': 877676885,
                 'history_reset_mask_word': 21,
                 'warmup_word': 22,
                 'warmup_observations': 4,
                 'warmup_cue_bit': 8}}


@dataclass(frozen=True)
class CalibratedDefaults:
    collapse_abs_epsilon: float
    adaptive_policy_id: int
    gain_per_pop: float
    reference_pop_strength: float
    direct_container_limit: float
    max_horizontal_slope: float
    max_vertical_shear: float
    vertical_majorant_share: float
    convergence_curve_default: float
    adaptive_time_constant_seconds: float
    adaptive_log_scale_rate: float
    adaptive_zero_budget_per_second: float
    adaptive_max_tick_gap_ms: float
    host_percentile_bin_count: float
    host_percentile_low: float
    host_percentile_high: float


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
    geometry_authority: str
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
    ocr_safe_row_top: int
    ocr_safe_row_bottom: int
    source_crop_aspect_width: int
    source_crop_aspect_height: int
    text_join_gap_cells: int
    ordinary_cover_horizontal_guard_cells: int
    ordinary_cover_horizontal_guard_policy: str
    line_group_min_vertical_iou_numerator: int
    line_group_min_vertical_iou_denominator: int
    line_group_max_horizontal_gap_height_numerator: int
    line_group_max_horizontal_gap_height_denominator: int
    line_group_horizontal_gap_metric: str
    line_center_tolerance_numerator: int
    line_center_tolerance_denominator: int
    line_search_bottom_height_numerator: int
    line_search_bottom_height_denominator: int
    line_search_region: str
    line_group_policy: str
    line_center_reference: str
    line_center_reference_params: str
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
    plane_schema: int
    plane_word_count: int
    plane_header_word_count: int
    plane_cover_capacity: int
    plane_flags_word: int
    plane_current_valid_flag: int
    plane_current_slot_mask_word: int
    plane_frame_offset: int
    plane_analysis_generation_offset: int
    plane_observation_timestamp_offset: int
    plane_source_extent_offset: int
    plane_field_extent_offset: int
    plane_roi_offset: int
    plane_content_offset: int
    plane_cut_epoch_word: int
    # Legacy reset-mask name: word21 is the current-core probability cue mask.
    plane_history_reset_mask_word: int
    plane_warmup_word: int
    plane_warmup_observations: int
    plane_warmup_cue_bit: int
    plane_warmup_remaining_mask: int
    plane_reserved_offset: int
    plane_reserved_word_count: int
    plane_adaptive_offset: int
    plane_adaptive_word_count: int
    plane_tag: int
    condition_param_schema: int
    condition_param_tag: int
    condition_param_word_count: int
    condition_param_current_count_word: int
    condition_param_current_kind_mask_word: int
    condition_param_active_word: int
    condition_param_fully_active_value: int
    condition_param_target_word: int


def _subtitle_ocr_contract(contract: dict[str, Any]) -> SubtitleOcrContract:
    value = contract.get("subtitle_ocr")
    if value != EXPECTED_SUBTITLE_OCR:
        raise ValueError(
            "depth-coordinate-v2 subtitle_ocr identity or ABI does not match the current "
            "authenticated PP-OCRv6/OCR8/UIP4 contract")
    for name in ("ordinary_cover_horizontal_guard_cells",
                 "line_search_bottom_height_numerator", "line_search_bottom_height_denominator",
                 "line_group_min_vertical_iou_numerator", "line_group_min_vertical_iou_denominator",
                 "line_group_max_horizontal_gap_height_numerator", "line_group_max_horizontal_gap_height_denominator",
                 "line_center_tolerance_numerator", "line_center_tolerance_denominator"):
        if type(value["field_policy"][name]) is not int:
            raise ValueError("subtitle_ocr detector guard and line policy values must be exact integers")
    for name in ("plane_state", "condition_params"):
        if any(type(word) is not int for word in value[name].values()):
            raise ValueError(f"subtitle_ocr.{name} layout words must be exact integers")
    for name in ("schema", "recipe", "history_schema", "history_tag", "history_word_count",
                 "history_reserved_offset", "history_reserved_word_count"):
        if type(value["contour_policy"][name]) is not int:
            raise ValueError("subtitle_ocr.contour_policy layout words must be exact integers")
    input_tensor = value["input_tensor"]
    output_tensor = value["output_tensor"]
    field_policy = value["field_policy"]
    record = value["ocr_record"]
    plane = value["plane_state"]
    condition = value["condition_params"]
    return SubtitleOcrContract(
        geometry_authority=value["geometry_authority"],
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
        ocr_safe_row_top=field_policy["ocr_safe_row_top"],
        ocr_safe_row_bottom=field_policy["ocr_safe_row_bottom"],
        source_crop_aspect_width=field_policy["source_crop_aspect_width"],
        source_crop_aspect_height=field_policy["source_crop_aspect_height"],
        text_join_gap_cells=field_policy["text_join_gap_cells"],
        ordinary_cover_horizontal_guard_cells=field_policy["ordinary_cover_horizontal_guard_cells"],
        ordinary_cover_horizontal_guard_policy=field_policy["ordinary_cover_horizontal_guard_policy"],
        line_group_min_vertical_iou_numerator=field_policy["line_group_min_vertical_iou_numerator"],
        line_group_min_vertical_iou_denominator=field_policy["line_group_min_vertical_iou_denominator"],
        line_group_max_horizontal_gap_height_numerator=field_policy["line_group_max_horizontal_gap_height_numerator"],
        line_group_max_horizontal_gap_height_denominator=field_policy["line_group_max_horizontal_gap_height_denominator"],
        line_group_horizontal_gap_metric=field_policy["line_group_horizontal_gap_metric"],
        line_center_tolerance_numerator=field_policy["line_center_tolerance_numerator"],
        line_center_tolerance_denominator=field_policy["line_center_tolerance_denominator"],
        line_search_bottom_height_numerator=field_policy["line_search_bottom_height_numerator"],
        line_search_bottom_height_denominator=field_policy["line_search_bottom_height_denominator"],
        line_search_region=field_policy["line_search_region"],
        line_group_policy=field_policy["line_group_policy"],
        line_center_reference=field_policy["line_center_reference"],
        line_center_reference_params=field_policy["line_center_reference_params"],
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
        plane_schema=plane["schema"],
        plane_word_count=plane["word_count"],
        plane_header_word_count=plane["header_word_count"],
        plane_cover_capacity=plane["cover_capacity"],
        plane_flags_word=plane["flags_word"],
        plane_current_valid_flag=plane["current_valid_flag"],
        plane_current_slot_mask_word=plane["current_slot_mask_word"],
        plane_frame_offset=plane["frame_offset"],
        plane_analysis_generation_offset=plane["analysis_generation_offset"],
        plane_observation_timestamp_offset=plane["observation_timestamp_offset"],
        plane_source_extent_offset=plane["source_extent_offset"],
        plane_field_extent_offset=plane["field_extent_offset"],
        plane_roi_offset=plane["roi_offset"],
        plane_content_offset=plane["content_offset"],
        plane_cut_epoch_word=plane["cut_epoch_word"],
        plane_history_reset_mask_word=plane["history_reset_mask_word"],
        plane_warmup_word=plane["warmup_word"],
        plane_warmup_observations=plane["warmup_observations"],
        plane_warmup_cue_bit=plane["warmup_cue_bit"],
        plane_warmup_remaining_mask=plane["warmup_cue_bit"] - 1,
        plane_reserved_offset=plane["reserved_offset"],
        plane_reserved_word_count=plane["reserved_word_count"],
        plane_adaptive_offset=plane["adaptive_offset"],
        plane_adaptive_word_count=plane["adaptive_word_count"],
        plane_tag=plane["tag"],
        condition_param_schema=condition["schema"],
        condition_param_tag=condition["tag"],
        condition_param_word_count=condition["word_count"],
        condition_param_current_count_word=condition["current_count_word"],
        condition_param_current_kind_mask_word=condition["current_kind_mask_word"],
        condition_param_active_word=condition["active_word"],
        condition_param_fully_active_value=condition["fully_active_value"],
        condition_param_target_word=condition["target_word"],
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
                preprocess.get("source_file") != "rgb_to_nchw_near_identical_cs.hlsl" or
                preprocess.get("source_entrypoint") != "fused_main" or
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
        if name == "adaptive_policy_id":
            if type(value) is not int or value != 3:
                raise ValueError("adaptive_policy_id must be exactly integer three")
        elif name == "convergence_curve_default":
            if float(value) != 0.0:
                raise ValueError("convergence_curve_default must be exactly zero")
        elif float(value) <= 0.0:
            raise ValueError(f"depth-coordinate-v2 default {name} must be positive")
    if defaults["max_horizontal_slope"] >= 1.0:
        raise ValueError("max_horizontal_slope must be below one")
    contour_policy = contract["subtitle_ocr"]["contour_policy"]
    offset = contour_policy["gaussian_center_offset_source_pixels"]
    if type(offset) is not int or not 0 <= offset <= 2:
        raise ValueError("subtitle Gaussian center offset must be a bounded source-pixel integer")
    for name in ("gaussian_sigma_short_side_divisor", "gaussian_weight_absolute_error_limit"):
        value = contour_policy[name]
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise ValueError("subtitle Gaussian parameters must be positive finite floats")
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
    **{name: (_CONTRACT["calibrated_defaults"][name] if name == "adaptive_policy_id"
              else float(_CONTRACT["calibrated_defaults"][name]))
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


def subtitle_ocr_field_cell_scale(field_width: int, field_height: int) -> int:
    """Return 1 for coarse fields, 2 for convex-2x live fields, or 0 if invalid."""

    coarse = any(
        (field_width, field_height) in calibration.calibrated_input_shapes
        for calibration in MODEL_CALIBRATIONS
    )
    if coarse:
        return 1
    live = any(
        shape.get("width") == field_width and shape.get("height") == field_height
        for shape in _CONTRACT["subtitle_ocr"]["field_policy"]["live_field_shapes"]
    )
    return 2 if live else 0


def subtitle_ocr_field_is_calibrated(field_width: int, field_height: int) -> bool:
    """Return whether OCR projection may target a coarse or Stage-2 live field."""

    return subtitle_ocr_field_cell_scale(field_width, field_height) != 0


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
]
