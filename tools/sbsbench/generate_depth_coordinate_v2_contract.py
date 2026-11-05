#!/usr/bin/env python3
"""Generate native and HLSL declarations for the complete depth-coordinate-v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

try:
    from . import generate_host_sbs_shader_manifest as shader_manifest_generator
except ImportError:  # Direct script/module loading from tools/sbsbench.
    import generate_host_sbs_shader_manifest as shader_manifest_generator  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT / "tools" / "sbsbench" / "contracts" /
    "depth-coordinate-v2-v1.json"
)
CPP_TARGET = ROOT / "src" / "generated" / "depth_coordinate_v2_contract.h"
HLSL_TARGET = (
    ROOT / "src_assets" / "windows" / "assets" / "shaders" / "directx" /
    "include" / "depth_coordinate_v2_contract.generated.hlsl"
)
HLSL_OCR_ASSERT_TARGET = (
    ROOT / "src_assets" / "windows" / "assets" / "shaders" / "directx" /
    "include" / "depth_coordinate_v2_ocr_assert.generated.hlsl"
)
# Current-outline snapshots compile with their own include directories. They consume the same
# authenticated contract as the root producer and must change before closure pins are refreshed.
SHADER_ROOT = HLSL_TARGET.parent.parent
HLSL_TARGETS = (
    HLSL_TARGET,
    *(SHADER_ROOT / "current_outline" / variant / "include" / HLSL_TARGET.name
      for variant in ("colour", "enclosure")),
)
HLSL_OCR_ASSERT_TARGETS = (
    HLSL_OCR_ASSERT_TARGET,
    *(SHADER_ROOT / "current_outline" / variant / "include" / HLSL_OCR_ASSERT_TARGET.name
      for variant in ("colour", "enclosure")),
)
# CBC owns the physical-black model predicates. Each independently authenticated shader group
# must keep includes inside its own root, so F2 consumes a byte-identical generated mirror rather
# than a second editable algorithm or an include escaping to the CBC group. Materialize the
# mirror before refreshing any closure, and reject drift during ordinary contract checks.
PHYSICAL_BLACK_MODEL_SOURCE = SHADER_ROOT / "current_outline" / "enclosure" / "physical_black_model_shared.hlsl"
PHYSICAL_BLACK_MODEL_MIRRORS = (
    SHADER_ROOT / "current_outline" / "outline" / PHYSICAL_BLACK_MODEL_SOURCE.name,
    SHADER_ROOT / "current_outline" / "hole" / PHYSICAL_BLACK_MODEL_SOURCE.name,
)
CURRENT_STROKE_MATH_SOURCE = SHADER_ROOT / "current_outline" / "inside" / "current_stroke_math_shared.hlsl"
CURRENT_STROKE_MATH_TARGETS = tuple(
    SHADER_ROOT / "current_outline" / variant / CURRENT_STROKE_MATH_SOURCE.name
    for variant in ("outline", "hole")
)
CBC_UNIT_STATE_SOURCE = SHADER_ROOT / "current_outline" / "outline" / "cbc_unit_state_shared.hlsl"
CBC_UNIT_STATE_MIRRORS = (
    SHADER_ROOT / "current_outline" / "hole" / CBC_UNIT_STATE_SOURCE.name,
)
CURRENT_OUTLINE_RUNTIME_HEADER = ROOT / "src" / "host_sbs_current_outline_runtime_programs.h"
CURRENT_BACKGROUND_RUNTIME_HEADER = ROOT / "src" / "host_sbs_selected_glyph_push_pull_runtime.h"
CURRENT_SHARED_UI_WARP_RUNTIME_HEADER = ROOT / "src" / "host_sbs_shared_ui_warp_runtime.h"
CURRENT_OUTLINE_RUNTIME_ROOTS = {
    name: ("outline" if name == "stability" else "hole" if name == "coverage" else name)
    for name in ("source5", "colour", "components", "enclosure", "roi", "outline", "stability",
                 "inside", "selector", "hole", "coverage")
}
# Lines of LIMITER_SERIAL_MAX_LINES cells or fewer keep the serial float recurrence. Longer
# lines are scanned in LIMITER_LINE_CHUNKS Q30 chunks, LIMITER_GROUP_LINES adjacent lines per
# group, so neighboring threads read and write neighboring texels.
LIMITER_SERIAL_MAX_LINES = 32
LIMITER_GROUP_LINES = 8
LIMITER_LINE_CHUNKS = 8
LIMITER_Q_FRACTION_BITS = 30
LIMITER_Q_SCALE = 1 << LIMITER_Q_FRACTION_BITS

EXPECTED_TOP_LEVEL_KEYS = {
    "schema", "vector_width", "calibrated_defaults", "constant_buffer", "frame_stats",
    "shadow_state", "model_calibrations", "capture_provenance", "shader_implementation",
    "subtitle_ocr", "final_parallax", "selected_map",
}


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


EXPECTED_FIELD_KEYS = {"index", "name", "type", "gpu_encoding", "initial"}
EXPECTED_LAYOUT_FIELD_KEYS = {"index", "name", "type"}
EXPECTED_DEFAULT_NAMES = (
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
EXPECTED_DEFAULT_KEYS = set(EXPECTED_DEFAULT_NAMES)
EXPECTED_MODEL_CALIBRATION_KEYS = {
    "calibration_id", "depth_model", "depth_model_url", "onnx_sha256",
    "raw_coordinate_scale", "calibrated_input_shapes", "preprocess",
}
EXPECTED_PREPROCESS_KEYS = {
    "profile", "source_closure_schema", "source_file", "source_entrypoint",
    "source_target", "source_compile_flags", "source_macro_count",
    "source_closure_sha256", "model_input_schema", "dtype", "layout", "channels",
    "patch_multiple", "maximum_dimension", "imagenet_mean", "imagenet_std", "stage",
}
EXPECTED_SHADER_IMPLEMENTATION_KEYS = {
    "source_closure_schema", "source_compile_flags", "source_macro_count", "source_specs",
    "source_closure_sha256",
}
EXPECTED_SHADER_SPEC_KEYS = {
    "source_file", "source_entrypoint", "source_target",
}
CANONICAL_SELECTED_MAP = {'schema': 1,
 'authority': 'base-inverse-plus-analytic-ui-glyph-selector-v1',
 'base_inverse': 'at-most-11-step-contractive-fixed-point-on-ordinary-base-with-existing-roi-embedding',
 'ui_source_u': 'destination-source-u-plus-eye-sign-times-scp-target-times-analysis-width-over-full-source-width',
 'ui_source_v': 'unchanged-source-v',
 'selector': 'ui-if-either-base-or-ui-bilinear-color-footprint-has-any-surviving-native-glyph-contributor-otherwise-base',
 'contributor_policy': 'saturate-u-then-clamp-finite-image-indices-positive-bilinear-weights-only',
 'current_rgb_revocation': 'same-native-texel-display-srgb-fp16-roundtrip-linf-at-most-contour-history-rgb-distance-no-new-support',
 'combined_map': 'potentially-discontinuous-selected-map-no-monotonicity-or-final-field-inverse-claim',
 'sample_policy': 'one-native-bilinear-color-sample-at-selected-uv',
 'invalid_subtitle_policy': 'preserve-authorized-base',
 'empty_policy': 'joined-new-empty-matte-zero-scp-inactive-immediate-base-no-added-frame-delay'}
CANONICAL_FINAL_PARALLAX = {'schema': 4,
 'authority': 'authenticated-subtitle-conditioned-r32f-diagnostic-only',
 'publication_policy': 'atomic-joined-publication-retains-independent-final-diagnostic',
 'reuse_policy': 'reuse-holds-complete-base-state-matte-sct-scp-source-and-diagnostics-tuple',
 'invalid_policy': 'fail-closed-flat',
 'current_rgb_policy': 'bounded-matched-color-reuse-current-draw-can-only-revoke-matte'}
CANONICAL_SUBTITLE_OCR = {'schema': 38,
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
 'output_tensor': {'name': 'fetch_name_0', 'dtype': 'float32', 'layout': 'NCHW', 'shape': [1, 1, 160, 960]},
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
                  'ordinary_cover_horizontal_guard_cells': 4,
                  'ordinary_cover_horizontal_guard_policy': 'ordinary-only-four-producer-cell-horizontal-guard-and-ceil-half-tight-detector-height-vertical-guard-add-db-unclip-clamp-before-map-current-cover-search-no-rectangle-protection-ribbon-unchanged-v2',
                  'line_group_min_vertical_iou_numerator': 1,
                  'line_group_min_vertical_iou_denominator': 2,
                  'line_group_max_horizontal_gap_height_numerator': 4,
                  'line_group_max_horizontal_gap_height_denominator': 1,
                  'line_group_horizontal_gap_metric': 'native-source-horizontal-gap-to-current-union-edge-relative-to-max-candidate-and-fixed-anchor-height-exact-cross-multiplied-v1',
                  'line_center_tolerance_numerator': 1,
                  'line_center_tolerance_denominator': 50,
                  'line_search_bottom_height_numerator': 1,
                  'line_search_bottom_height_denominator': 4,
                  'line_search_region': 'lower-quarter-of-center-reference-intersect-existing-safe-detector-crop-v1',
                  'line_group_policy': 'local-column-island-row-components-true-piece-bounds-fixed-span-ordinary-aggregation-then-centered-native-line-union-before-classified-capacity-v3',
                  'line_center_reference': 'chromium-video-or-foreground-client-roi-otherwise-full-screen-horizontal-center-and-width-in-analysis-source-pixels-v2',
                  'line_center_reference_params': 'ocr-resolve-prefix20-then-source-clock-low-high-domain-reset-reserved-96bytes-v2',
                  'ribbon_join_gap_cells': 12,
                  'ribbon_structural_gap_min_cells': 3,
                  'ribbon_min_structural_gaps': 3,
                  'ribbon_min_width_numerator': 1,
                  'ribbon_min_width_denominator': 2,
                  'ribbon_bottom_tolerance_pixels': 2,
                  'ribbon_bottom_tolerance_projection': 'exact-ceil-detector-edge-through-bottom-crop-v1',
                  'ribbon_cover_pad_limit': 8,
                  'ordinary_cover_vertical_guard_height_numerator': 1,
                  'ordinary_cover_vertical_guard_height_denominator': 2},
 'candidate_selection': {'policy': 'bounded-largest-core-area-v1',
                         'capacity': 'min-raw-and-final-box-capacity',
                         'ranking': 'tight-core-field-area-descending',
                         'tie_break': 'earlier-raster-discovery',
                         'publication_order': 'raster-discovery-order',
                         'overflow': 'retain-best-paired-candidates'},
 'ocr_record': {'schema': 4,
                'tag': 944915279,
                'word_count': 256,
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
 'geometry_authority': 'native-glyph-matte-selects-analytic-ui-otherwise-ordinary-base-inverse-v1',
 'ui_plane_policy': 'current-clearance-goal-first-visible-immediate-symmetric-source-time-exponential-native-slew-empty-reset-v3',
 'contour_policy': {'schema': 5,
                    'recipe': 27,
                    'source': 'same-joined-ocr-source-bottom-crop-display-srgb-finite-rgba16f-v1',
                    'source_max_extent': 5120,
                    'source_extent_policy': 'native-exact-crop-for-supported-source-otherwise-canonical-1x1-empty-support-history-source-and-zero-probability-v1',
                    'history_jitter_radius_source_pixels': 2,
                    'spatial_edge_radius_source_pixels': 2,
                    'spatial_edge_rgb_distance': 0.08,
                    'probability_validity_policy': 'all-four-interpolation-contributors-finite-in-unit-range-current-native-center-bilinear-probability-v2',
                    'probability_sampling_arithmetic': 'precise-fp32-divide-center-multiply-subtract-floor-fraction-separated-lerp-delta-multiply-add-no-contraction-no-support-epsilon-v1',
                    'diagnostic_snapshot_policy': 'same-joined-valid-ocr-geometry-source-clock-copy-before-next-admission-clear-invalid-whole-reuse-hold-v1',
                    'native_support': 'r32u-binary-current-classified-centered-bounded-search-valid-current-peak-ink-color-short-stroke-or-local-dark-outline-positive-diagnostic-count-v6',
                    'authority': 'current-ocr8-uip4-scp4-cut-source-clock-and-current-signed-reference-geometry-no-history-vote-authority-v3',
                    'projection': 'native-visible-support-to-all-bilinear-field-nodes-integer-footprint-v1',
                    'history': 'diagnostic-current-cover-fixed-rgb-anchor-four-genuine-observations-same-pixel-first-cue-blocks-neighbor-fallback-no-four-count-render-gate-v5',
                    'history_schema': 5,
                    'history_tag': 894714707,
                    'history_word_count': 32,
                    'conditioning': 'horizontal-only-post-minimum-slope-budget-shared-q30-guard-vertical-proposal-bit-copy-unchanged-rows-and-empty-or-invalid-current-exact-base-v4',
                    'reuse': 'whole-depth-ocr-uip4-native-mask-field-pin-q-gaussian-weight-blend-vertical-control-history-source-probability-final-tuple-held-v1',
                    'uncertain_support': 'current-rgb-glyph-admits-withdraws-immediately-no-coarse-cover-pin-no-four-count-render-wait-uip4-word22-zero-v6',
                    'history_snapshot_valid_word': 23,
                    'history_feature_valid_bit': 2,
                    'history_feature_mask_bit': 1,
                    'history_rgb_distance': 0.08,
                    'mask_evidence_policy': 'current-core-strong-probability-edge-anchors-64x8-brightest-actual-center-rgb-representatives-hue-dedup-strict-color-stroke-or-eight-explicit-local-dark-outline-neighbors-no-probability-only-bypass-v13',
                    'history_confirmation_observations': 4,
                    'history_feature_count_shift': 2,
                    'history_anchor_policy': 'same-pixel-qualified-fixed-anchor-first-even-with-current-core-probability-cue-otherwise-unflagged-min-rgb-distance-then-squared-spatial-distance-then-scan-order-v3',
                    'distance_policy': 'native-five-center-rectangular-field-footprint-plus-completion-exact-rational-y-slab-horizontal-q30-parallax-pixel-budget-positive-infinity-off-slab-v3',
                    'distance_empty_bits': 2139095040,
                    'gaussian_center_offset_source_pixels': 2,
                    'gaussian_sigma_short_side_divisor': 120.0,
                    'gaussian_weight_policy': 'fp32-equivalent-bounded-minimum-ui-depth-influence-from-slope-budget-zero-budget-ui-gap-or-outside-interval-base-v8',
                    'gaussian_weight_absolute_error_limit': 2e-06,
                    'gaussian_blend_policy': 'precise-nearest-ui-plus-minus-budget-closed-base-ui-roundoff-clamp-bit-exact-endpoints-v3',
                    'history_reserved_offset': 24,
                    'history_reserved_word_count': 8,
                    'probability_change_ratio': 0.125,
                    'probability_change_policy': 'current-core-native-center-soft-probability-unit-mass-spatial-tv-strict-ratio-cue-both-zero-no-cue-one-zero-cue-v3',
                    'ordinary_search_pad_height_numerator': 1,
                    'ordinary_search_pad_height_denominator': 2,
                    'ordinary_search_policy': 'ordinary-only-current-fixed-core-half-iou-line-envelope-corrective-x-side-wide-only-beyond-one-detector-cell-signed-current-reference-otherwise-outward-floor-24-detector-pixel-inset-nonclamped-side-top-ceil-half-core-height-bottom-full-cover-ribbon-full-cover-unchanged-v3',
                    'diagnostic_search_policy': 'fixed-rgb-diagnostic-counts-retain-current-coarse-cover-scope-no-four-count-render-wait-v2',
                    'distance_completion_source_pixels': 3,
                    'distance_axis_policy': 'horizontal-q30-limiter-step-exact-source-content-geometry-no-vertical-cost-or-cross-row-transition-v2',
                    'gaussian_sigma_policy': 'legacy-serialized-divisor-unused-by-minimum-slope-budget-recipe-v1',
                    'palette_policy': 'per-current-core-64x8-exact-center-grid-valid-strong-probability-native-edge-q16-64-bins-count-and-max-peak-earliest-grid-index-actual-anchor-representative-sort-peak-count-bin-hue-prune-cap-eight-v2',
                    'palette_match_policy': 'exact-integer-q16-representative-linf-symmetric-three-twentieths-only-no-common-brightening-cone-v2',
                    'outline_policy': 'current-dark-low-chroma-native-edge-plus-eight-explicit-fixed-current-palette-neighbor-calls-including-diagonals-radius-six-no-dilation-v2',
                    'long_line_policy': 'palette-foreground-native-edge-normal-tangent-color-coherent-run-no-longer-than-three-core-native-heights-v1',
                    'palette_grid_columns': 64,
                    'palette_grid_rows': 8,
                    'palette_channel_max': 65535,
                    'palette_bin_count': 64,
                    'palette_max_modes': 8,
                    'palette_brightness_numerator': 1,
                    'palette_brightness_denominator': 2,
                    'palette_match_rgb_numerator': 3,
                    'palette_match_rgb_denominator': 20,
                    'outline_value_numerator': 2,
                    'outline_value_denominator': 5,
                    'outline_chroma_numerator': 3,
                    'outline_chroma_denominator': 20,
                    'outline_probe_radius_source_pixels': 6,
                    'long_line_core_height_numerator': 3,
                    'long_line_core_height_denominator': 1,
                    'palette_representative_index_bits': 10,
                    'palette_representative_index_mask': 1023,
                    'palette_hue_channel_max': 4095,
                    'palette_merge_hue_numerator': 3,
                    'palette_merge_hue_denominator': 20,
                    'ordinary_search_inset_detector_pixels': 24,
                    'ordinary_search_center_bias_detector_pixels': 8,
                    'palette_representative_policy': 'max-key-q16-max-channel-shift10-or-1023-minus-grid-index-reload-actual-valid-center-anchor-no-neighbor-endpoint-no-mean-color-v1',
                    'palette_merge_policy': 'greedy-dimmer-representative-q12-normalized-rgb-nearest-integer-max-channel-distance-three-twentieths-from-retained-brighter-mode-v1',
                    'ordinary_search_reference_policy': 'current-canonical-scl1-clock-frame-generation-source-crop-field-content-bounded-signed-reference-only-zero48-source-center-fallback-invalid-nonzero-fail-closed-never-layout-votes-v1',
                    'ordinary_search_inset_preserves_core': True,
                    'distance_row_policy': 'inclusive-exact-integer-rational-field-node-center-plus-minus-source-height-over-content-height-plus-completion-native-row-membership-v1',
                    'conditioning_control_policy': 'current-authority-and-any-finite-nonnegative-first-content-column-budget-row-local-infinity-publishes-original-base-zero-weight-v1'},
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
                 'adaptive_known_flags': 28,
                 'adaptive_retired_approach_flag': 1,
                 'tag': 877676885,
                 'history_reset_mask_word': 21,
                 'warmup_word': 22,
                 'warmup_observations': 4,
                 'warmup_cue_bit': 8,
                 'adaptive_retired_retreat_flag': 2,
                 'retreat_time_constant_ms': 100,
                 'adaptive_presented_flag': 16,
                 'smoothing_time_constant_ms': 100,
                 'slew_source_pixels_per_second': 30,
                 'max_tick_gap_ms': 50},
 'classifier': {'policy': 'current-local-lines-envelope-orientation-bracketed-stack-and-tight-two-percent-group-center-span-before-protection-v6',
                'local_candidate_capacity': 1980,
                'local_candidate_word_count': 24,
                'capacity_proof': 'at-most60-disjoint-active-column-islands-times33-valid-three-row-pieces-in131-safe-rows-v1',
                'scratch_policy': 'fixed-per-estimator-uint4-uav-unretained-candidates-no-readback-v1',
                'history_offset': 208,
                'history_word_count': 48,
                'history_tag': 827081555,
                'history_schema': 1,
                'history_known_flags': 1,
                'x_bins': 8,
                'min_populated_bins': 6,
                'envelope_low_quantile_numerator': 1,
                'envelope_high_quantile_numerator': 9,
                'envelope_quantile_denominator': 10,
                'horizontal_tan': 0.03492076949174773,
                'scene_tan': 0.17632698070846498,
                'orientation_policy': 'native-source-least-squares-of-nearest-rank-active-mask-y10-y90-in-eight-horizontal-bins-v1',
                'candidate_assembly_policy': 'per-band-raster-order-global-current-pool-deferred-stack-assistance-and-publication-v1',
                'stack_policy': 'current-individually-centered-ambiguous-single-flat-envelope-bounded-by-scene-angle-bracketed-by-independent-horizontal-centered-height-compatible-overlapping-anchors-never-teaches-v1',
                'stack_max_gap_height_numerator': 1,
                'stack_max_gap_height_denominator': 1,
                'prior_bin_count': 16,
                'prior_decay': 1,
                'prior_increment': 4,
                'prior_cap': 255,
                'prior_assist': 8,
                'learning_gap_us': 100000,
                'min_height_ratio': 0.5,
                'max_height_ratio': 2.0,
                'learning_policy': 'independent-current-horizontal-entry-after-authoritative-blank-not-per-frame-assisted-candidates-never-teach-v1',
                'release_policy': 'current-classified-covers-only-empty-current-clears-mask-immediately-layout-never-authorizes-pixels-v1',
                'reset_policy': 'domain-context-source-field-content-reference-or-invalid-clock-clears-prior-layout-survives-scene-cuts-v1',
                'unknown_policy': 'current-compatible-line-or-qualified-position-height-prior-required-no-fixed-frame-wait-v1',
                'prior_geometry_policy': 'ambiguous-prior-requires-both-envelopes-within-scene-angle-and-either-one-horizontal-or-coherent-same-direction-moderate-slopes-unknown-separate-v1',
                'prior_only_group_center_policy': 'current-half-open-group-must-contain-signed-reference-center-without-borrowing-prior-location-as-current-line-evidence-v1',
                'ordinary_group_center_policy': 'current-ordinary-tight-group-must-contain-signed-reference-center-half-open-before-horizontal-current-line-or-prior-admission-ribbons-exempt-v1'},
 'glyph_matte_policy': {'schema': 1,
                        'source': 'same-joined-native-current-support-and-display-rgba16f-v1',
                        'steps': 3,
                        'rgb_distance': 0.12,
                        'brightness_split': 0.4,
                        'collar_radius_source_pixels': 3,
                        'collar_radius_squared': 10,
                        'body_policy': 'three-synchronous-eight-neighbor-valid-rgba-rgb-geodesic-steps-same-peak-brightness-class-v1',
                        'collar_policy': '37-native-offsets-dx-dy-within-three-and-squared-distance-at-most-ten-v1',
                        'empty_policy': 'same-current-authority-zero-without-delay',
                        'history_policy': 'no-new-history-or-observation-wait',
                        'field_policy': 'no-field-pitch-or-distance-budget-dilation',
                        'format': 'R32_UINT-literal-zero-or-one'}}

EXPECTED_CONSTANT_FIELD_NAMES = (
    "raw_coordinate_scale",
    "collapse_abs_epsilon",
    "coordinate_reserved0",
    "coordinate_reserved1",
    "requested_gain",
    "max_horizontal_slope",
    "direct_container_limit",
    "convergence_curve_default",
    "joint_plane_mode",
    "joint_plane_reserved0",
    "joint_plane_reserved1",
    "joint_plane_reserved2",
    "joint_observation_timestamp_low",
    "joint_observation_timestamp_high",
    "joint_observation_reserved0",
    "joint_observation_reserved1",
)
EXPECTED_FRAME_STAT_FIELD_NAMES = (
    "mean",
    "population_std",
    "minimum",
    "maximum",
    "valid_count",
    "texel_count",
    "valid",
    "reserved",
    "percentile_low", "percentile_high", "percentile_valid", "percentile_bin_width",
)
EXPECTED_SHADOW_STATE_FIELD_NAMES = (
    "center",
    "inverse_scale",
    "convergence_curve",
    "container_scale",
    "calibration_revision",
    "frame_valid",
    "confirmed_cut_count",
    "contract_tag_bits",
    "camera_center_integrity_bits",
    "renderer_authorization_bits",
    "joint_plane_mode_bits",
    "mapping_state_reserved_2",
    "gain_last_observation_low", "gain_last_observation_high",
    "gain_clock_armed", "gain_seed_count",
    "gain_target_zero", "gain_target_inverse_scale", "gain_target_nearest",
    "gain_display_limit",
    "gain_seed_first_low", "gain_seed_first_high",
    "gain_seed_last_low", "gain_seed_last_high",
    "gain_seed_mean_nearest", "gain_seed_mean_zero", "gain_reserved0", "gain_reserved1",
)
CONTRACT_TAG_SENTINEL = "contract_tag"
SHADER_MANIFEST = shader_manifest_generator.load_manifest()
PREPROCESS_SHADER_ROOT = shader_manifest_generator.SHADER_ROOT
PREPROCESS_SHADER_SPECS = shader_manifest_generator.group_specs(
    SHADER_MANIFEST, "preprocess")
PARALLAX_V2_SHADER_SPECS = shader_manifest_generator.group_specs(
    SHADER_MANIFEST, "parallax_v2_producer")
SHADER_COMPILE_FLAGS = SHADER_MANIFEST["shader_compile_flags"]
SOURCE_CLOSURE_SCHEMA = SHADER_MANIFEST["source_closure_schema"]
ALGORITHM_TAG_SHADER_DIGEST_SENTINEL = (
    "shader-source-closure-sha256-authenticated-separately-v1")

def _identifier(value: str) -> str:
    identifier = re.sub(r"[^a-zA-Z0-9_]", "_", value)
    if not identifier or identifier[0].isdigit():
        raise ValueError(f"cannot turn {value!r} into an identifier")
    return identifier


def _upper_identifier(value: str) -> str:
    return _identifier(value).upper()


shader_source_closure_sha256 = shader_manifest_generator.shader_source_closure_sha256


def validate_renderer_source_closure_pins(
        shader_root: Path = PREPROCESS_SHADER_ROOT) -> dict[str, str]:
    """Validate and expose the renderer projection from the named closure manifest."""

    shader_manifest_generator.validate_group_pins(SHADER_MANIFEST, shader_root)
    groups = {
        group["name"]: group["source_closure_sha256"]
        for group in SHADER_MANIFEST["groups"]
    }
    return {
        "parallax_v2_live_renderer_source_closure_sha256":
            groups["parallax_v2_live_renderer"],
        "parallax_v2_p010_y_source_closure_sha256":
            groups["parallax_v2_p010_y"],
        "parallax_v2_diagnostic_source_closure_sha256":
            groups["parallax_v2_live_diagnostic"],
    }


def canonical_bytes(contract: dict[str, Any]) -> bytes:
    """Return the complete normalized manifest used for its full provenance digest."""

    return json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def contract_digest(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(contract)).hexdigest()


def contract_tag_semantic_bytes(contract: dict[str, Any]) -> bytes:
    """Canonical tag input without the independently authenticated shader-body digest.

    The generated HLSL is itself part of the shader source closure, so hashing the closure digest
    into the GPU tag would be self-referential.  Replace only that digest with a fixed domain
    sentinel.  All shader specs/compiler policy and every non-implementation contract semantic
    remain in the tag; the complete manifest digest plus shader closure digest authenticate the
    final source bodies without a circular fixed-point search.
    """

    tag_contract = dict(contract)
    implementation = dict(contract["shader_implementation"])
    implementation["source_closure_sha256"] = ALGORITHM_TAG_SHADER_DIGEST_SENTINEL
    tag_contract["shader_implementation"] = implementation
    return canonical_bytes(tag_contract)


def contract_tag_semantic_digest(contract: dict[str, Any]) -> str:
    return hashlib.sha256(contract_tag_semantic_bytes(contract)).hexdigest()


def tag_is_finite_normal(tag: int) -> bool:
    """Whether ``tag`` survives an exact uint-as-float-as-uint GPU round trip."""

    exponent = (tag >> 23) & 0xFF
    return 0 < exponent < 0xFF


def contract_tag(contract: dict[str, Any]) -> int:
    # One GPU word is available and transported through a float4 via asfloat/asuint. Reject
    # zero/subnormal and NaN/Inf encodings because drivers may flush or canonicalize those bits.
    digest = hashlib.sha256(contract_tag_semantic_bytes(contract)).digest()
    tag = int.from_bytes(digest[:4], "big")
    nonce = 0
    while not tag_is_finite_normal(tag):
        nonce += 1
        candidate = hashlib.sha256(
            digest + b"\0depth-coordinate-v2-tag\0" + nonce.to_bytes(4, "big")
        ).digest()
        tag = int.from_bytes(candidate[:4], "big")
    return tag


def validate_contract(
        contract: Any, *, verify_shader_source_closure: bool = True,
        verify_preprocess_source_closure: bool = True) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise ValueError("manifest root must be an object")
    if set(contract) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError(
            "manifest keys must be exactly " + ", ".join(sorted(EXPECTED_TOP_LEVEL_KEYS)))
    schema = contract.get("schema")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema < 1:
        raise ValueError("manifest schema must be a positive integer")
    if contract.get("vector_width") != 4:
        raise ValueError("depth-coordinate-v2 state vector width must be exactly four")

    provenance = contract.get("capture_provenance")
    if (not isinstance(provenance, dict) or
            set(provenance) != {"schema", "manifest_key", "binding"} or
            not isinstance(provenance.get("schema"), int) or
            isinstance(provenance.get("schema"), bool) or provenance["schema"] < 1 or
            any(not isinstance(provenance.get(name), str) or not provenance[name]
                for name in ("manifest_key", "binding"))):
        raise ValueError(
            "capture_provenance must contain a positive schema, manifest_key, and binding")

    if contract.get("subtitle_ocr") != CANONICAL_SUBTITLE_OCR:
        raise ValueError(
            "subtitle_ocr must exactly match the authenticated PP-OCRv6/OCR8/UIP4 contract")
    for name in ("ordinary_cover_horizontal_guard_cells",
                 "ordinary_cover_vertical_guard_height_numerator",
                 "ordinary_cover_vertical_guard_height_denominator",
                 "line_search_bottom_height_numerator", "line_search_bottom_height_denominator",
                 "line_group_min_vertical_iou_numerator", "line_group_min_vertical_iou_denominator",
                 "line_group_max_horizontal_gap_height_numerator", "line_group_max_horizontal_gap_height_denominator",
                 "line_center_tolerance_numerator", "line_center_tolerance_denominator"):
        if type(contract["subtitle_ocr"]["field_policy"][name]) is not int:
            raise ValueError("subtitle_ocr detector guard and line policy values must be exact integers")
    for name in ("plane_state", "condition_params"):
        if any(type(value) is not int for value in contract["subtitle_ocr"][name].values()):
            raise ValueError(f"subtitle_ocr.{name} layout words must be exact integers")
    for name, expected in CANONICAL_SUBTITLE_OCR["contour_policy"].items():
        if type(expected) is bool:
            if type(contract["subtitle_ocr"]["contour_policy"][name]) is not bool:
                raise ValueError("subtitle_ocr.contour_policy switches must be exact booleans")
            continue
        if type(expected) is not int:
            continue
        if type(contract["subtitle_ocr"]["contour_policy"][name]) is not int:
            raise ValueError("subtitle_ocr.contour_policy layout words must be exact integers")
    if contract.get("selected_map") != CANONICAL_SELECTED_MAP:
        raise ValueError("selected_map must exactly match the authenticated analytic glyph selector")
    if contract.get("final_parallax") != CANONICAL_FINAL_PARALLAX:
        raise ValueError(
            "final_parallax must exactly match the authenticated direct-render contract")

    shader_implementation = contract.get("shader_implementation")
    if (not isinstance(shader_implementation, dict) or
            set(shader_implementation) != EXPECTED_SHADER_IMPLEMENTATION_KEYS):
        raise ValueError(
            "shader_implementation keys must be exactly " +
            ", ".join(sorted(EXPECTED_SHADER_IMPLEMENTATION_KEYS)))
    if (shader_implementation.get("source_closure_schema") != SOURCE_CLOSURE_SCHEMA or
            shader_implementation.get("source_compile_flags") != SHADER_COMPILE_FLAGS or
            shader_implementation.get("source_macro_count") != 0):
        raise ValueError("shader_implementation compiler/source identity is unsupported")
    source_specs = shader_implementation.get("source_specs")
    if not isinstance(source_specs, list):
        raise ValueError("shader_implementation.source_specs must be an ordered list")
    parsed_specs: list[tuple[str, str, str]] = []
    for index, spec in enumerate(source_specs):
        if not isinstance(spec, dict) or set(spec) != EXPECTED_SHADER_SPEC_KEYS:
            raise ValueError(
                f"shader_implementation.source_specs[{index}] has an unknown schema")
        values = tuple(spec[name] for name in (
            "source_file", "source_entrypoint", "source_target"))
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(
                f"shader_implementation.source_specs[{index}] values must be non-empty strings")
        parsed_specs.append(values)  # type: ignore[arg-type]
    if tuple(parsed_specs) != PARALLAX_V2_SHADER_SPECS:
        raise ValueError("shader_implementation.source_specs is not the selected V2 shader set")
    shader_digest = shader_implementation.get("source_closure_sha256")
    if (not isinstance(shader_digest, str) or
            re.fullmatch(r"[0-9a-f]{64}", shader_digest) is None):
        raise ValueError("shader_implementation.source_closure_sha256 must be lowercase SHA-256")
    producer_closure_pin = shader_manifest_generator.closure_group(
        SHADER_MANIFEST, "parallax_v2_producer")["source_closure_sha256"]
    if verify_shader_source_closure and shader_digest != producer_closure_pin:
        raise ValueError(
            "shader_implementation.source_closure_sha256 is not the generated producer view")

    calibrated_defaults = contract.get("calibrated_defaults")
    if not isinstance(calibrated_defaults, dict) or set(calibrated_defaults) != EXPECTED_DEFAULT_KEYS:
        raise ValueError(
            "calibrated_defaults keys must be exactly " +
            ", ".join(sorted(EXPECTED_DEFAULT_KEYS)))
    for name, value in calibrated_defaults.items():
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value))):
            raise ValueError(f"calibrated default {name} must be finite")
        if name == "adaptive_policy_id":
            if type(value) is not int or value != 3:
                raise ValueError("adaptive_policy_id must be exactly integer 3")
        elif name == "convergence_curve_default":
            if float(value) != 0.0:
                raise ValueError("convergence_curve_default must be exactly zero")
        elif float(value) <= 0.0:
            raise ValueError(f"calibrated default {name} must be positive")
    if calibrated_defaults["max_horizontal_slope"] >= 1.0:
        raise ValueError("max_horizontal_slope must be below one")
    contour_policy = contract["subtitle_ocr"]["contour_policy"]
    offset = contour_policy["gaussian_center_offset_source_pixels"]
    if type(offset) is not int or not 0 <= offset <= 2:
        raise ValueError("subtitle Gaussian center offset must be a bounded source-pixel integer")
    completion = contour_policy['distance_completion_source_pixels']
    if type(completion) is not int or not 0 <= completion <= 3:
        raise ValueError("subtitle distance completion must be a bounded source-pixel integer")
    for name in ("gaussian_sigma_short_side_divisor", "gaussian_weight_absolute_error_limit"):
        value = contour_policy[name]
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise ValueError("subtitle Gaussian parameters must be positive finite floats")
    if (calibrated_defaults["host_percentile_bin_count"] != 256 or
            not 0.0 < calibrated_defaults["host_percentile_low"] <
            calibrated_defaults["host_percentile_high"] < 1.0):
        raise ValueError("Host percentile histogram bins or ranks are invalid")
    if calibrated_defaults["adaptive_max_tick_gap_ms"] != 250:
        raise ValueError("adaptive source-clock gap limit is invalid")
    for name in ("max_horizontal_slope", "max_vertical_shear"):
        scaled = float(calibrated_defaults[name]) * LIMITER_Q_SCALE
        if not scaled.is_integer() or scaled <= 0.0 or scaled > 0xFFFFFFFF:
            raise ValueError(f"{name} must have an exact unsigned Q30 numerator")
    limiter_q_limit = math.ceil(
        _float32(calibrated_defaults["direct_container_limit"]) * LIMITER_Q_SCALE)
    if limiter_q_limit <= 0 or 4 * limiter_q_limit > 0x7FFFFFFF:
        raise ValueError("direct_container_limit is unsafe for signed Q30 limiter arithmetic")
    majorant_share = _float32(calibrated_defaults["vertical_majorant_share"])
    minorant_share = _float32(_float32(1.0) - majorant_share)
    if majorant_share <= 0.0 or minorant_share <= 0.0:
        raise ValueError(
            "vertical_majorant_share and its complement must remain positive in float32")

    calibrations = contract.get("model_calibrations")
    if not isinstance(calibrations, list) or not calibrations:
        raise ValueError("model_calibrations must be a non-empty list")
    seen_ids: set[str] = set()
    seen_models: set[str] = set()
    seen_hashes: set[str] = set()
    for position, calibration in enumerate(calibrations):
        prefix = f"model_calibrations[{position}]"
        if not isinstance(calibration, dict) or set(calibration) != EXPECTED_MODEL_CALIBRATION_KEYS:
            raise ValueError(
                f"{prefix} keys must be exactly " +
                ", ".join(sorted(EXPECTED_MODEL_CALIBRATION_KEYS)))
        calibration_id = calibration.get("calibration_id")
        model = calibration.get("depth_model")
        url = calibration.get("depth_model_url")
        digest = calibration.get("onnx_sha256")
        if not isinstance(calibration_id, str) or not calibration_id:
            raise ValueError(f"{prefix}.calibration_id must be a non-empty string")
        if (not isinstance(model, str) or not model or _identifier(model) != model):
            raise ValueError(f"{prefix}.depth_model must be an identifier")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"{prefix}.depth_model_url must be an HTTPS URL")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{prefix}.onnx_sha256 must be lowercase SHA-256")
        scale = calibration.get("raw_coordinate_scale")
        if (not isinstance(scale, (int, float)) or isinstance(scale, bool) or
                not math.isfinite(float(scale)) or float(scale) <= 0.0):
            raise ValueError(f"{prefix}.raw_coordinate_scale must be finite and positive")
        if calibration_id in seen_ids or model in seen_models or digest in seen_hashes:
            raise ValueError("model calibration ids, model names, and ONNX hashes must be unique")
        seen_ids.add(calibration_id)
        seen_models.add(model)
        seen_hashes.add(digest)

        preprocess = calibration.get("preprocess")
        if not isinstance(preprocess, dict) or set(preprocess) != EXPECTED_PREPROCESS_KEYS:
            raise ValueError(
                f"{prefix}.preprocess keys must be exactly " +
                ", ".join(sorted(EXPECTED_PREPROCESS_KEYS)))
        for name in ("profile", "stage"):
            if not isinstance(preprocess.get(name), str) or not preprocess[name]:
                raise ValueError(f"{prefix}.preprocess.{name} must be a non-empty string")
        source_closure_sha256 = preprocess.get("source_closure_sha256")
        if (not isinstance(source_closure_sha256, str) or
                re.fullmatch(r"[0-9a-f]{64}", source_closure_sha256) is None):
            raise ValueError(
                f"{prefix}.preprocess.source_closure_sha256 must be lowercase SHA-256")
        expected_source_identity = {
            "source_closure_schema": SOURCE_CLOSURE_SCHEMA,
            "source_file": PREPROCESS_SHADER_SPECS[0][0],
            "source_entrypoint": PREPROCESS_SHADER_SPECS[0][1],
            "source_target": PREPROCESS_SHADER_SPECS[0][2],
            "source_compile_flags": SHADER_COMPILE_FLAGS,
            "source_macro_count": 0,
        }
        if any(preprocess.get(name) != value
               for name, value in expected_source_identity.items()):
            raise ValueError(
                f"{prefix}.preprocess shader source identity is unsupported")
        preprocess_closure_pin = shader_manifest_generator.closure_group(
            SHADER_MANIFEST, "preprocess")["source_closure_sha256"]
        if (verify_preprocess_source_closure and
                source_closure_sha256 != preprocess_closure_pin):
            raise ValueError(
                f"{prefix}.preprocess.source_closure_sha256 is not the generated preprocess view")
        if (preprocess.get("model_input_schema") != 1 or
                preprocess.get("dtype") != "float32-le" or
                preprocess.get("layout") != "NCHW" or
                preprocess.get("channels") != ["R", "G", "B"]):
            raise ValueError(f"{prefix}.preprocess has unsupported model-input semantics")
        patch = preprocess.get("patch_multiple")
        maximum = preprocess.get("maximum_dimension")
        if (not isinstance(patch, int) or isinstance(patch, bool) or patch <= 0 or
                not isinstance(maximum, int) or isinstance(maximum, bool) or
                maximum < patch or maximum % patch):
            raise ValueError(
                f"{prefix}.preprocess dimensions need a positive patch multiple and aligned max")
        shapes = calibration.get("calibrated_input_shapes")
        if not isinstance(shapes, list) or not shapes:
            raise ValueError(f"{prefix}.calibrated_input_shapes must be non-empty")
        seen_shapes: set[tuple[int, int]] = set()
        for shape_index, shape in enumerate(shapes):
            if (not isinstance(shape, dict) or set(shape) != {"width", "height"} or
                    not isinstance(shape.get("width"), int) or
                    isinstance(shape.get("width"), bool) or shape["width"] <= 0 or
                    not isinstance(shape.get("height"), int) or
                    isinstance(shape.get("height"), bool) or shape["height"] <= 0 or
                    shape["width"] > maximum or shape["height"] > maximum or
                    shape["width"] % patch or shape["height"] % patch):
                raise ValueError(
                    f"{prefix}.calibrated_input_shapes[{shape_index}] violates preprocessing")
            dimensions = (shape["width"], shape["height"])
            if dimensions in seen_shapes:
                raise ValueError(f"{prefix}.calibrated_input_shapes contains a duplicate")
            seen_shapes.add(dimensions)
        for name in ("imagenet_mean", "imagenet_std"):
            values = preprocess.get(name)
            if (not isinstance(values, list) or len(values) != 3 or
                    any(not isinstance(value, (int, float)) or isinstance(value, bool) or
                        not math.isfinite(float(value)) for value in values) or
                    (name == "imagenet_std" and any(float(value) <= 0.0 for value in values))):
                raise ValueError(f"{prefix}.preprocess.{name} must contain three finite values")

    def validate_float_layout(
            name: str, fields: Any, expected_names: tuple[str, ...]) -> None:
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"{name} fields must be a non-empty list")
        if len(fields) % contract["vector_width"]:
            raise ValueError(f"{name} field count must contain complete float4 vectors")
        if any(not isinstance(field, dict) or set(field) != EXPECTED_LAYOUT_FIELD_KEYS
               for field in fields):
            raise ValueError(
                f"every {name} field must contain exactly " +
                ", ".join(sorted(EXPECTED_LAYOUT_FIELD_KEYS)))
        if [field["index"] for field in fields] != list(range(len(fields))):
            raise ValueError(f"{name} fields must be ordered by contiguous index")
        names = [field["name"] for field in fields]
        if any(not isinstance(value, str) or not value or _identifier(value) != value
               for value in names):
            raise ValueError(f"{name} field names must be C/HLSL identifiers")
        if len(set(names)) != len(names):
            raise ValueError(f"{name} field names must be unique")
        if tuple(names) != expected_names:
            raise ValueError(
                f"{name} physical field order must be exactly " +
                ", ".join(expected_names))
        expected_types = (["float32"] * 8 + ["uint32"] * 8
                          if name == "constant_buffer" else ["float32"] * len(fields))
        if [field["type"] for field in fields] != expected_types:
            raise ValueError(f"{name} has an invalid ordered field type layout")

    constant_buffer = contract.get("constant_buffer")
    if (not isinstance(constant_buffer, dict) or
            set(constant_buffer) != {"name", "register", "fields"}):
        raise ValueError("constant_buffer must contain exactly name, register, and fields")
    if (_identifier(str(constant_buffer.get("name", ""))) != constant_buffer.get("name") or
            not re.fullmatch(r"b[0-9]+", str(constant_buffer.get("register", "")))):
        raise ValueError("constant_buffer requires an identifier name and bN register")
    validate_float_layout(
        "constant_buffer", constant_buffer.get("fields"), EXPECTED_CONSTANT_FIELD_NAMES)

    frame_stats = contract.get("frame_stats")
    if (not isinstance(frame_stats, dict) or set(frame_stats) != {"source", "fields"} or
            not isinstance(frame_stats.get("source"), str) or not frame_stats["source"]):
        raise ValueError("frame_stats must contain a non-empty source and fields")
    validate_float_layout(
        "frame_stats", frame_stats.get("fields"), EXPECTED_FRAME_STAT_FIELD_NAMES)

    shadow_state = contract.get("shadow_state")
    if (not isinstance(shadow_state, dict) or
            set(shadow_state) != {"source", "capture", "fields"}):
        raise ValueError("shadow_state must contain exactly source, capture, and fields")
    for key in ("source", "capture"):
        if not isinstance(shadow_state.get(key), str) or not shadow_state[key]:
            raise ValueError(f"shadow_state {key} must be a non-empty string")
    fields = shadow_state.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("manifest fields must be a non-empty list")
    if len(fields) % contract["vector_width"]:
        raise ValueError("manifest field count must contain complete float4 vectors")
    if any(not isinstance(field, dict) or set(field) != EXPECTED_FIELD_KEYS
           for field in fields):
        raise ValueError(
            "every field must contain exactly " + ", ".join(sorted(EXPECTED_FIELD_KEYS)))
    indices = [field["index"] for field in fields]
    if indices != list(range(len(fields))):
        raise ValueError("manifest fields must be ordered by contiguous index")
    names = [field["name"] for field in fields]
    if any(not isinstance(name, str) or not name or _identifier(name) != name
           for name in names):
        raise ValueError("field names must be non-empty C/HLSL identifiers")
    if len(set(names)) != len(names):
        raise ValueError("field names must be unique")
    if tuple(names) != EXPECTED_SHADOW_STATE_FIELD_NAMES:
        raise ValueError(
            "shadow_state physical field order must be exactly " +
            ", ".join(EXPECTED_SHADOW_STATE_FIELD_NAMES))

    tag_fields = []
    for field in fields:
        field_type = field["type"]
        encoding = field["gpu_encoding"]
        initial = field["initial"]
        if field_type == "float32":
            if encoding != "float":
                raise ValueError(f"float32 field {field['name']} must use float encoding")
            if (not isinstance(initial, (int, float)) or isinstance(initial, bool) or
                    not math.isfinite(float(initial))):
                raise ValueError(f"float32 field {field['name']} needs a finite initial value")
        elif field_type == "uint32":
            if encoding != "uint_bits":
                raise ValueError(f"uint32 field {field['name']} must use uint_bits encoding")
            if initial == CONTRACT_TAG_SENTINEL:
                tag_fields.append(field)
            elif (not isinstance(initial, int) or isinstance(initial, bool) or
                  not 0 <= initial <= 0xFFFFFFFF):
                raise ValueError(f"uint32 field {field['name']} needs a uint32 initial value")
        else:
            raise ValueError(f"unsupported field type for {field['name']}: {field_type!r}")
    if (len(tag_fields) != 1 or tag_fields[0]["name"] != "contract_tag_bits" or
            tag_fields[0]["index"] !=
            EXPECTED_SHADOW_STATE_FIELD_NAMES.index("contract_tag_bits")):
        raise ValueError(
            "contract_tag_bits must be the sole uint32 contract tag sentinel at its "
            "authenticated physical index")
    contract_tag(contract)
    return contract


def load_contract(
        path: Path = MANIFEST, *, verify_shader_source_closure: bool = True,
        verify_preprocess_source_closure: bool = True) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return validate_contract(
            json.load(stream),
            verify_shader_source_closure=verify_shader_source_closure,
            verify_preprocess_source_closure=verify_preprocess_source_closure,
        )


def _float_literal(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric)}.0f"
    return f"{numeric:.9g}f"


def _initial_word_cpp(field: dict[str, Any], tag: int) -> str:
    initial = field["initial"]
    if initial == CONTRACT_TAG_SENTINEL:
        return "contract_tag"
    if field["type"] == "uint32":
        return f"{int(initial)}u"
    return f"std::bit_cast<std::uint32_t>({_float_literal(initial)})"


def _classifier_declarations(policy: dict[str, Any], language: str) -> list[str]:
    """Project the complete classifier policy; numeric values have one canonical owner."""
    result: list[str] = []
    if language == "cpp":
        result.append("  inline constexpr std::string_view subtitle_ocr_classifier_policy_json = " +
                      json.dumps(json.dumps(policy, separators=(",", ":"))) + ";")
        result.append(f"  inline constexpr std::uint32_t subtitle_ocr_current_prefix_word_count = {policy['history_offset']}u;")
    else:
        result.append(f"#define V2_OCR_CURRENT_PREFIX_WORD_COUNT {policy['history_offset']}u")
    for key, value in policy.items():
        if isinstance(value, str):
            if language == "cpp":
                result.append(f"  inline constexpr std::string_view subtitle_ocr_classifier_{key} = {json.dumps(value)};")
            continue
        name = key if key.startswith("local_candidate_") else "classifier_" + key
        if type(value) is int:
            literal = f"{value}u"
            native_type = "std::uint32_t"
        elif type(value) is float:
            literal = _float_literal(value)
            native_type = "float"
        else:
            raise ValueError(f"unsupported classifier parameter: {key}")
        if language == "cpp":
            result.append(f"  inline constexpr {native_type} subtitle_ocr_{name} = {literal};")
        else:
            result.append(f"#define V2_OCR_{name.upper()} {literal}")
    return result


def render_cpp(contract: dict[str, Any]) -> str:
    fields = contract["shadow_state"]["fields"]
    defaults = contract["calibrated_defaults"]
    constants = contract["constant_buffer"]["fields"]
    frame_stats = contract["frame_stats"]["fields"]
    tag = contract_tag(contract)
    digest = contract_digest(contract)
    tag_semantic_digest = contract_tag_semantic_digest(contract)
    calibrations = contract["model_calibrations"]
    subtitle_ocr = contract["subtitle_ocr"]
    final_parallax = contract["final_parallax"]
    ocr_input = subtitle_ocr["input_tensor"]
    ocr_output = subtitle_ocr["output_tensor"]
    field_policy = subtitle_ocr["field_policy"]
    ocr_record = subtitle_ocr["ocr_record"]
    plane_state = subtitle_ocr["plane_state"]
    condition_params = subtitle_ocr["condition_params"]
    calibrated_shapes = [
        (calibration["calibration_id"], shape["width"], shape["height"])
        for calibration in calibrations
        for shape in calibration["calibrated_input_shapes"]
    ]
    live_field_shapes = [
        (shape["width"], shape["height"])
        for shape in field_policy["live_field_shapes"]
    ]
    lines = [
        "// Generated by tools/sbsbench/generate_depth_coordinate_v2_contract.py.",
        "// Edit tools/sbsbench/contracts/depth-coordinate-v2-v1.json instead.",
        "#pragma once",
        "",
        "#include \"host_sbs_shader_manifest.h\"",
        "",
        "#include <array>",
        "#include <bit>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <string_view>",
        "#include <type_traits>",
        "",
        "namespace models::depth_coordinate_v2 {",
        f"  inline constexpr std::uint32_t contract_schema = {contract['schema']}u;",
        f"  inline constexpr std::uint32_t contract_tag = 0x{tag:08X}u;",
        f'  inline constexpr std::string_view contract_canonical_sha256 = "{digest}";',
        f'  inline constexpr std::string_view contract_tag_semantic_sha256 = "{tag_semantic_digest}";',
        f'  inline constexpr std::string_view shadow_state_source = "{contract["shadow_state"]["source"]}";',
        f'  inline constexpr std::string_view shadow_state_capture = "{contract["shadow_state"]["capture"]}";',
        f'  inline constexpr std::string_view frame_stats_source = "{contract["frame_stats"]["source"]}";',
        f'  inline constexpr std::string_view constant_buffer_name = "{contract["constant_buffer"]["name"]}";',
        f'  inline constexpr std::string_view constant_buffer_register = "{contract["constant_buffer"]["register"]}";',
        f"  inline constexpr std::uint32_t capture_provenance_schema = "
        f"{contract['capture_provenance']['schema']}u;",
        f"  inline constexpr std::string_view capture_provenance_manifest_key = "
        f"{json.dumps(contract['capture_provenance']['manifest_key'])};",
        f"  inline constexpr std::string_view capture_provenance_binding = "
        f"{json.dumps(contract['capture_provenance']['binding'])};",
        f"  inline constexpr std::uint32_t final_parallax_contract_schema = "
        f"{final_parallax['schema']}u;",
        f"  inline constexpr std::string_view final_parallax_authority = "
        f"{json.dumps(final_parallax['authority'])};",
        f"  inline constexpr std::string_view final_parallax_publication_policy = "
        f"{json.dumps(final_parallax['publication_policy'])};",
        f"  inline constexpr std::string_view final_parallax_reuse_policy = "
        f"{json.dumps(final_parallax['reuse_policy'])};",
        f"  inline constexpr std::string_view final_parallax_invalid_policy = "
        f"{json.dumps(final_parallax['invalid_policy'])};",
        f"  inline constexpr std::string_view final_parallax_current_rgb_policy = "
        f"{json.dumps(final_parallax['current_rgb_policy'])};",
        f"  inline constexpr std::string_view selected_map_contract_json = {json.dumps(json.dumps(contract['selected_map'], separators=(',', ':')))};",
        f"  inline constexpr std::string_view subtitle_glyph_matte_policy_json = {json.dumps(json.dumps(subtitle_ocr['glyph_matte_policy'], separators=(',', ':')))};",
        f"  inline constexpr std::uint32_t subtitle_glyph_matte_schema = {subtitle_ocr['glyph_matte_policy']['schema']}u;",
        f"  inline constexpr std::uint32_t subtitle_glyph_matte_steps = {subtitle_ocr['glyph_matte_policy']['steps']}u;",
        f"  inline constexpr float subtitle_glyph_matte_rgb_distance = {_float_literal(subtitle_ocr['glyph_matte_policy']['rgb_distance'])};",
        f"  inline constexpr float subtitle_glyph_matte_brightness_split = {_float_literal(subtitle_ocr['glyph_matte_policy']['brightness_split'])};",
        f"  inline constexpr std::uint32_t subtitle_glyph_matte_collar_radius_source_pixels = {subtitle_ocr['glyph_matte_policy']['collar_radius_source_pixels']}u;",
        f"  inline constexpr std::uint32_t subtitle_glyph_matte_collar_radius_squared = {subtitle_ocr['glyph_matte_policy']['collar_radius_squared']}u;",
        f"  inline constexpr std::string_view subtitle_geometry_authority = "
        f"{json.dumps(subtitle_ocr['geometry_authority'])};",
        f"  inline constexpr std::string_view subtitle_ui_plane_policy = {json.dumps(subtitle_ocr['ui_plane_policy'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_contract_schema = "
        f"{subtitle_ocr['schema']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_schema = {subtitle_ocr['contour_policy']['schema']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_recipe = {subtitle_ocr['contour_policy']['recipe']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_history_tag = {subtitle_ocr['contour_policy']['history_tag']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_history_word_count = {subtitle_ocr['contour_policy']['history_word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_history_snapshot_valid_word = {subtitle_ocr['contour_policy']['history_snapshot_valid_word']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_history_reserved_offset = {subtitle_ocr['contour_policy']['history_reserved_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_history_reserved_word_count = {subtitle_ocr['contour_policy']['history_reserved_word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_history_feature_valid_bit = {subtitle_ocr['contour_policy']['history_feature_valid_bit']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_history_feature_mask_bit = {subtitle_ocr['contour_policy']['history_feature_mask_bit']}u;",
        f"  inline constexpr float subtitle_contour_history_rgb_distance = {_float_literal(subtitle_ocr['contour_policy']['history_rgb_distance'])};",
        f"  inline constexpr float subtitle_contour_probability_change_ratio = {_float_literal(subtitle_ocr['contour_policy']['probability_change_ratio'])};",
        f"  inline constexpr std::uint32_t subtitle_contour_history_confirmation_observations = {subtitle_ocr['contour_policy']['history_confirmation_observations']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_history_feature_count_shift = {subtitle_ocr['contour_policy']['history_feature_count_shift']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_source_max_extent = {subtitle_ocr['contour_policy']['source_max_extent']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_history_jitter_radius_source_pixels = {subtitle_ocr['contour_policy']['history_jitter_radius_source_pixels']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_ordinary_search_pad_height_numerator = {subtitle_ocr['contour_policy']['ordinary_search_pad_height_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_ordinary_search_pad_height_denominator = {subtitle_ocr['contour_policy']['ordinary_search_pad_height_denominator']}u;",
        f"  inline constexpr std::string_view subtitle_contour_ordinary_search_policy = {json.dumps(subtitle_ocr['contour_policy']['ordinary_search_policy'])};",
        f"  inline constexpr std::string_view subtitle_contour_diagnostic_search_policy = {json.dumps(subtitle_ocr['contour_policy']['diagnostic_search_policy'])};",
        f"  inline constexpr std::uint32_t subtitle_contour_spatial_edge_radius_source_pixels = {subtitle_ocr['contour_policy']['spatial_edge_radius_source_pixels']}u;",
        f"  inline constexpr float subtitle_contour_spatial_edge_rgb_distance = {_float_literal(subtitle_ocr['contour_policy']['spatial_edge_rgb_distance'])};",
        f"  inline constexpr std::uint32_t subtitle_contour_distance_empty_bits = {subtitle_ocr['contour_policy']['distance_empty_bits']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_gaussian_center_offset_source_pixels = {subtitle_ocr['contour_policy']['gaussian_center_offset_source_pixels']}u;",
        f"  inline constexpr float subtitle_contour_gaussian_sigma_short_side_divisor = {_float_literal(subtitle_ocr['contour_policy']['gaussian_sigma_short_side_divisor'])};",
        f"  inline constexpr std::uint32_t subtitle_contour_distance_completion_source_pixels = {subtitle_ocr['contour_policy']['distance_completion_source_pixels']}u;",
        f"  inline constexpr float subtitle_contour_gaussian_weight_absolute_error_limit = {_float_literal(subtitle_ocr['contour_policy']['gaussian_weight_absolute_error_limit'])};",
        f"  inline constexpr std::string_view subtitle_contour_policy_json = "
        f"{json.dumps(json.dumps(subtitle_ocr['contour_policy'], separators=(',', ':')))};",
        f"  inline constexpr std::string_view subtitle_ocr_model_name = "
        f"{json.dumps(subtitle_ocr['logical_model'])};",
        f"  inline constexpr std::string_view subtitle_ocr_asset_path = "
        f"{json.dumps(subtitle_ocr['asset_path'])};",
        f"  inline constexpr std::string_view subtitle_ocr_artifact_onnx_sha256 = "
        f"{json.dumps(subtitle_ocr['artifact_onnx_sha256'])};",
        f"  inline constexpr std::string_view subtitle_ocr_source_url = "
        f"{json.dumps(subtitle_ocr['source_url'])};",
        f"  inline constexpr std::string_view subtitle_ocr_source_onnx_sha256 = "
        f"{json.dumps(subtitle_ocr['source_onnx_sha256'])};",
        f"  inline constexpr std::string_view subtitle_ocr_conversion_tool = "
        f"{json.dumps(subtitle_ocr['conversion_tool'])};",
        f"  inline constexpr std::string_view subtitle_ocr_conversion_version = "
        f"{json.dumps(subtitle_ocr['conversion_version'])};",
        f"  inline constexpr std::string_view subtitle_ocr_conversion_recipe = "
        f"{json.dumps(subtitle_ocr['conversion_recipe'])};",
        f"  inline constexpr std::string_view subtitle_ocr_conversion_calibration_profile = "
        f"{json.dumps(subtitle_ocr['conversion_calibration_profile'])};",
        f"  inline constexpr std::string_view subtitle_ocr_engine_recipe = "
        f"{json.dumps(subtitle_ocr['engine_recipe'])};",
        f"  inline constexpr std::string_view subtitle_ocr_preprocess_profile = "
        f"{json.dumps(subtitle_ocr['preprocess_profile'])};",
        f"  inline constexpr std::string_view subtitle_ocr_source_crop = "
        f"{json.dumps(subtitle_ocr['source_crop'])};",
        f"  inline constexpr std::string_view subtitle_ocr_input_name = "
        f"{json.dumps(ocr_input['name'])};",
        f"  inline constexpr std::string_view subtitle_ocr_input_dtype = "
        f"{json.dumps(ocr_input['dtype'])};",
        f"  inline constexpr std::string_view subtitle_ocr_input_layout = "
        f"{json.dumps(ocr_input['layout'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_input_n = {ocr_input['shape'][0]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_input_c = {ocr_input['shape'][1]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_input_height = {ocr_input['shape'][2]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_input_width = {ocr_input['shape'][3]}u;",
        "  inline constexpr std::array<std::string_view, 3> subtitle_ocr_input_channels {{" +
        ", ".join(json.dumps(value) for value in ocr_input["channels"]) + "}};",
        # Keep the manifest's decimal values as binary64 for canonical JSON provenance. HLSL
        # receives explicit float literals below; using float here would promote rounded binary32
        # values back to double when nlohmann serializes the dump descriptor.
        "  inline constexpr std::array<double, 3> subtitle_ocr_imagenet_mean {{" +
        ", ".join(json.dumps(value) for value in ocr_input["imagenet_mean"]) + "}};",
        "  inline constexpr std::array<double, 3> subtitle_ocr_imagenet_std {{" +
        ", ".join(json.dumps(value) for value in ocr_input["imagenet_std"]) + "}};",
        f"  inline constexpr std::string_view subtitle_ocr_output_name = "
        f"{json.dumps(ocr_output['name'])};",
        f"  inline constexpr std::string_view subtitle_ocr_output_dtype = "
        f"{json.dumps(ocr_output['dtype'])};",
        f"  inline constexpr std::string_view subtitle_ocr_output_layout = "
        f"{json.dumps(ocr_output['layout'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_output_n = {ocr_output['shape'][0]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_output_c = {ocr_output['shape'][1]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_output_height = {ocr_output['shape'][2]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_output_width = {ocr_output['shape'][3]}u;",
        *_classifier_declarations(subtitle_ocr["classifier"], "cpp"),

        f"  inline constexpr std::uint32_t subtitle_contour_palette_representative_index_bits = {subtitle_ocr['contour_policy']['palette_representative_index_bits']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_representative_index_mask = {subtitle_ocr['contour_policy']['palette_representative_index_mask']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_hue_channel_max = {subtitle_ocr['contour_policy']['palette_hue_channel_max']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_merge_hue_numerator = {subtitle_ocr['contour_policy']['palette_merge_hue_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_merge_hue_denominator = {subtitle_ocr['contour_policy']['palette_merge_hue_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_ordinary_search_inset_detector_pixels = {subtitle_ocr['contour_policy']['ordinary_search_inset_detector_pixels']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_ordinary_search_center_bias_detector_pixels = {subtitle_ocr['contour_policy']['ordinary_search_center_bias_detector_pixels']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_grid_columns = {subtitle_ocr['contour_policy']['palette_grid_columns']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_grid_rows = {subtitle_ocr['contour_policy']['palette_grid_rows']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_channel_max = {subtitle_ocr['contour_policy']['palette_channel_max']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_bin_count = {subtitle_ocr['contour_policy']['palette_bin_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_max_modes = {subtitle_ocr['contour_policy']['palette_max_modes']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_brightness_numerator = {subtitle_ocr['contour_policy']['palette_brightness_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_brightness_denominator = {subtitle_ocr['contour_policy']['palette_brightness_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_match_rgb_numerator = {subtitle_ocr['contour_policy']['palette_match_rgb_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_palette_match_rgb_denominator = {subtitle_ocr['contour_policy']['palette_match_rgb_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_outline_value_numerator = {subtitle_ocr['contour_policy']['outline_value_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_outline_value_denominator = {subtitle_ocr['contour_policy']['outline_value_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_outline_chroma_numerator = {subtitle_ocr['contour_policy']['outline_chroma_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_outline_chroma_denominator = {subtitle_ocr['contour_policy']['outline_chroma_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_outline_probe_radius_source_pixels = {subtitle_ocr['contour_policy']['outline_probe_radius_source_pixels']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_long_line_core_height_numerator = {subtitle_ocr['contour_policy']['long_line_core_height_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_contour_long_line_core_height_denominator = {subtitle_ocr['contour_policy']['long_line_core_height_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_record_schema = {ocr_record['schema']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_record_tag = 0x{ocr_record['tag']:08X}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_record_word_count = {ocr_record['word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_record_header_word_count = {ocr_record['header_word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_box_word_count = {ocr_record['box_word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_box_flag_ribbon = {ocr_record['box_flag_ribbon']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_box_known_flags = {ocr_record['box_known_flags']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_raw_box_offset = {ocr_record['raw_box_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_raw_box_capacity = {ocr_record['raw_box_capacity']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_final_box_offset = {ocr_record['final_box_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_final_box_capacity = {ocr_record['final_box_capacity']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_safe_row_top = "
        f"{field_policy['ocr_safe_row_top']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_safe_row_bottom = "
        f"{field_policy['ocr_safe_row_bottom']}u;",
        f"  inline constexpr float subtitle_ocr_active_probability_threshold = "
        f"{_float_literal(field_policy['detector_active_probability_threshold'])};",
        f"  inline constexpr float subtitle_ocr_min_mean_score = "
        f"{_float_literal(field_policy['detector_min_mean_score'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_crop_aspect_width = "
        f"{field_policy['source_crop_aspect_width']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_crop_aspect_height = "
        f"{field_policy['source_crop_aspect_height']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_text_join_gap_cells = "
        f"{field_policy['text_join_gap_cells']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ordinary_cover_vertical_guard_height_numerator = {field_policy['ordinary_cover_vertical_guard_height_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ordinary_cover_vertical_guard_height_denominator = {field_policy['ordinary_cover_vertical_guard_height_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ordinary_cover_horizontal_guard_cells = "
        f"{field_policy['ordinary_cover_horizontal_guard_cells']}u;",
        f"  inline constexpr std::string_view subtitle_ocr_ordinary_cover_horizontal_guard_policy = "
        f"{json.dumps(field_policy['ordinary_cover_horizontal_guard_policy'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_line_group_min_vertical_iou_numerator = "
        f"{field_policy['line_group_min_vertical_iou_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_line_group_min_vertical_iou_denominator = "
        f"{field_policy['line_group_min_vertical_iou_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_line_group_max_horizontal_gap_height_numerator = "
        f"{field_policy['line_group_max_horizontal_gap_height_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_line_group_max_horizontal_gap_height_denominator = "
        f"{field_policy['line_group_max_horizontal_gap_height_denominator']}u;",
        f"  inline constexpr std::string_view subtitle_ocr_line_group_horizontal_gap_metric = "
        f"{json.dumps(field_policy['line_group_horizontal_gap_metric'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_line_center_tolerance_numerator = "
        f"{field_policy['line_center_tolerance_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_line_center_tolerance_denominator = "
        f"{field_policy['line_center_tolerance_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_line_search_bottom_height_numerator = "
        f"{field_policy['line_search_bottom_height_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_line_search_bottom_height_denominator = "
        f"{field_policy['line_search_bottom_height_denominator']}u;",
        f"  inline constexpr std::string_view subtitle_ocr_line_search_region = "
        f"{json.dumps(field_policy['line_search_region'])};",
        f"  inline constexpr std::string_view subtitle_ocr_line_group_policy = "
        f"{json.dumps(field_policy['line_group_policy'])};",
        f"  inline constexpr std::string_view subtitle_ocr_line_center_reference = "
        f"{json.dumps(field_policy['line_center_reference'])};",
        f"  inline constexpr std::string_view subtitle_ocr_line_center_reference_params = "
        f"{json.dumps(field_policy['line_center_reference_params'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_join_gap_cells = "
        f"{field_policy['ribbon_join_gap_cells']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_structural_gap_min_cells = "
        f"{field_policy['ribbon_structural_gap_min_cells']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_min_structural_gaps = "
        f"{field_policy['ribbon_min_structural_gaps']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_min_width_numerator = "
        f"{field_policy['ribbon_min_width_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_min_width_denominator = "
        f"{field_policy['ribbon_min_width_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_bottom_tolerance_pixels = "
        f"{field_policy['ribbon_bottom_tolerance_pixels']}u;",
        f"  inline constexpr std::string_view subtitle_ocr_ribbon_bottom_tolerance_projection = "
        f"{json.dumps(field_policy['ribbon_bottom_tolerance_projection'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_cover_pad_limit = "
        f"{field_policy['ribbon_cover_pad_limit']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_state_schema = {plane_state['schema']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_state_word_count = {plane_state['word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_header_word_count = {plane_state['header_word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_cover_capacity = {plane_state['cover_capacity']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_flags_word = {plane_state['flags_word']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_current_valid_flag = {plane_state['current_valid_flag']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_current_slot_mask_word = {plane_state['current_slot_mask_word']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_frame_offset = {plane_state['frame_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_analysis_generation_offset = {plane_state['analysis_generation_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_observation_timestamp_offset = {plane_state['observation_timestamp_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_source_extent_offset = {plane_state['source_extent_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_field_extent_offset = {plane_state['field_extent_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_roi_offset = {plane_state['roi_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_content_offset = {plane_state['content_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_cut_epoch_word = {plane_state['cut_epoch_word']}u;",
        "  // Legacy reset-mask name: current-core probability cues block neighbor fallback only.",
        f"  inline constexpr std::uint32_t subtitle_plane_history_reset_mask_word = {plane_state['history_reset_mask_word']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_warmup_word = {plane_state['warmup_word']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_warmup_observations = {plane_state['warmup_observations']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_warmup_cue_bit = {plane_state['warmup_cue_bit']}u;",
        '  inline constexpr std::uint32_t subtitle_plane_warmup_remaining_mask = subtitle_plane_warmup_cue_bit - 1u;',
        f"  inline constexpr std::uint32_t subtitle_plane_reserved_offset = {plane_state['reserved_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_reserved_word_count = {plane_state['reserved_word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_adaptive_offset = {plane_state['adaptive_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_adaptive_word_count = {plane_state['adaptive_word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_adaptive_known_flags = {plane_state['adaptive_known_flags']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_adaptive_retired_approach_flag = {plane_state['adaptive_retired_approach_flag']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_adaptive_retired_retreat_flag = {plane_state['adaptive_retired_retreat_flag']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_retreat_time_constant_ms = {plane_state['retreat_time_constant_ms']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_adaptive_presented_flag = {plane_state['adaptive_presented_flag']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_smoothing_time_constant_ms = {plane_state['smoothing_time_constant_ms']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_slew_source_pixels_per_second = {plane_state['slew_source_pixels_per_second']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_max_tick_gap_ms = {plane_state['max_tick_gap_ms']}u;",
        f"  inline constexpr std::uint32_t subtitle_plane_state_tag = 0x{plane_state['tag']:08X}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_schema = "
        f"{condition_params['schema']}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_tag = "
        f"0x{condition_params['tag']:08X}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_word_count = "
        f"{condition_params['word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_current_count_word = {condition_params['current_count_word']}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_current_kind_mask_word = {condition_params['current_kind_mask_word']}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_active_word = {condition_params['active_word']}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_fully_active_value = {condition_params['fully_active_value']}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_target_word = {condition_params['target_word']}u;",
        '  static_assert(subtitle_plane_state_schema == 4u && subtitle_plane_state_tag == 0x34504955u);',
        '  static_assert(subtitle_plane_state_word_count == 40u && subtitle_plane_header_word_count == 24u);',
        '  static_assert(subtitle_plane_cover_capacity == subtitle_ocr_final_box_capacity);',
        '  static_assert(subtitle_plane_warmup_word == 22u && subtitle_plane_warmup_observations == 4u && subtitle_plane_warmup_cue_bit == 8u && subtitle_plane_warmup_remaining_mask == 7u);',
        '  static_assert(subtitle_plane_warmup_observations == subtitle_contour_history_confirmation_observations);',
        '  static_assert(subtitle_plane_reserved_offset + subtitle_plane_reserved_word_count == subtitle_plane_adaptive_offset);',
        '  static_assert(subtitle_plane_adaptive_offset == subtitle_plane_header_word_count);',
        '  static_assert(subtitle_plane_state_word_count == subtitle_plane_adaptive_offset + subtitle_plane_adaptive_word_count);',
        '  static_assert(subtitle_condition_param_schema == 4u && subtitle_condition_param_word_count == 6u);',
        '  static_assert(subtitle_condition_param_target_word + 1u == subtitle_condition_param_word_count);',
        '  static_assert(subtitle_contour_history_reserved_offset == 24u && subtitle_contour_history_reserved_word_count == 8u);',
        "  static_assert(subtitle_ocr_input_n == 1u && subtitle_ocr_input_c == 3u);",
        "  static_assert(subtitle_ocr_output_n == 1u && subtitle_ocr_output_c == 1u);",
        "  static_assert(subtitle_ocr_active_probability_threshold > 0.0f &&",
        "                subtitle_ocr_active_probability_threshold < 1.0f);",
        "  static_assert(subtitle_ocr_min_mean_score >",
        "                subtitle_ocr_active_probability_threshold);",
        "  static_assert(subtitle_ocr_min_mean_score <= 1.0f);",
        "  static_assert(subtitle_ocr_text_join_gap_cells > 0u);",
        "  static_assert(subtitle_ocr_text_join_gap_cells <",
        "                subtitle_ocr_ribbon_join_gap_cells);",
        "  static_assert(subtitle_ocr_ribbon_join_gap_cells >=",
        "                subtitle_ocr_ribbon_structural_gap_min_cells);",
        "  static_assert(subtitle_ocr_ribbon_bottom_tolerance_pixels <=",
        "                subtitle_ocr_safe_row_bottom);",
        "  static_assert(subtitle_ocr_final_box_offset == subtitle_ocr_raw_box_offset +",
        "                subtitle_ocr_raw_box_capacity * subtitle_ocr_box_word_count);",
        "  static_assert(subtitle_ocr_classifier_history_offset == subtitle_ocr_final_box_offset +",
        "                subtitle_ocr_final_box_capacity * subtitle_ocr_box_word_count);",
        "  static_assert(subtitle_ocr_record_word_count == subtitle_ocr_classifier_history_offset + subtitle_ocr_classifier_history_word_count);",
        "  static_assert(subtitle_condition_param_word_count == 6u);",
        "  inline constexpr std::uint32_t shader_source_closure_schema =",
        "    host_sbs_shader_cache::source_closure_schema;",
        "  inline constexpr std::uint32_t shader_source_compile_flags =",
        "    host_sbs_shader_cache::shader_compile_flags;",
        "  inline constexpr std::uint32_t shader_source_macro_count =",
        "    host_sbs_shader_cache::source_macro_count;",
        "  inline constexpr std::string_view shader_source_closure_sha256 =",
        "    host_sbs_shader_cache::parallax_v2_producer_source_closure_sha256;",
        "",
        "  using shader_source_spec_t = host_sbs_shader_cache::shader_spec;",
        "  inline constexpr auto &shader_source_specs =",
        "    host_sbs_shader_cache::parallax_v2_producer_specs;",
        "",
    ]
    lines.extend(
        (f"  inline constexpr std::uint32_t {_identifier(name)} = {defaults[name]}u;"
         if name == "adaptive_policy_id" else
         f"  inline constexpr float {_identifier(name)} = {_float_literal(defaults[name])};")
        for name in EXPECTED_DEFAULT_NAMES
    )
    lines.extend([
        f"  inline constexpr std::uint32_t limiter_serial_max_lines = {LIMITER_SERIAL_MAX_LINES}u;",
        f"  inline constexpr std::uint32_t limiter_group_lines = {LIMITER_GROUP_LINES}u;",
        f"  inline constexpr std::uint32_t limiter_line_chunks = {LIMITER_LINE_CHUNKS}u;",
        f"  inline constexpr std::uint32_t limiter_q_fraction_bits = "
        f"{LIMITER_Q_FRACTION_BITS}u;",
        f"  inline constexpr std::uint32_t limiter_q_scale = {LIMITER_Q_SCALE}u;",
        f"  inline constexpr std::int32_t limiter_container_q_limit = "
        f"{math.ceil(_float32(defaults['direct_container_limit']) * LIMITER_Q_SCALE)};",
        f"  inline constexpr std::uint32_t limiter_horizontal_step_q_numerator = "
        f"{int(float(defaults['max_horizontal_slope']) * LIMITER_Q_SCALE)}u;",
        f"  inline constexpr std::uint32_t limiter_vertical_step_q_numerator = "
        f"{int(float(defaults['max_vertical_shear']) * LIMITER_Q_SCALE)}u;",
        "  inline constexpr std::string_view direct_parallax_decode_expression = "
        f'"(encoded * 2 - 1) * {json.dumps(float(defaults["direct_container_limit"]))}";',
        "  static_assert(convergence_curve_default == 0.0f);",
        "  static_assert(max_horizontal_slope > 0.0f && max_horizontal_slope < 1.0f);",
        "  static_assert(max_vertical_shear > 0.0f);",
        "  static_assert(vertical_majorant_share > 0.0f && vertical_majorant_share < 1.0f);",
        "",
        "  struct model_preprocess_contract_t {",
        "    std::string_view profile;",
        "    std::uint32_t source_closure_schema;",
        "    std::string_view source_file;",
        "    std::string_view source_entrypoint;",
        "    std::string_view source_target;",
        "    std::uint32_t source_compile_flags;",
        "    std::uint32_t source_macro_count;",
        "    std::string_view source_closure_sha256;",
        "    std::uint32_t model_input_schema;",
        "    std::string_view dtype;",
        "    std::string_view layout;",
        "    std::array<std::string_view, 3> channels;",
        "    std::uint32_t patch_multiple;",
        "    std::uint32_t maximum_dimension;",
        "    std::array<float, 3> imagenet_mean;",
        "    std::array<float, 3> imagenet_std;",
        "    std::string_view stage;",
        "  };",
        "",
        "  struct model_calibration_t {",
        "    std::string_view calibration_id;",
        "    std::string_view depth_model;",
        "    std::string_view depth_model_url;",
        "    std::string_view onnx_sha256;",
        "    float raw_coordinate_scale;",
        "    model_preprocess_contract_t preprocess;",
        "  };",
        "",
        "  struct model_calibrated_shape_t {",
        "    std::string_view calibration_id;",
        "    std::uint32_t width;",
        "    std::uint32_t height;",
        "  };",
        "",
        f"  inline constexpr std::array<model_calibration_t, {len(calibrations)}> "
        "model_calibrations {{",
    ])
    for calibration in calibrations:
        preprocess = calibration["preprocess"]
        means = ", ".join(_float_literal(value) for value in preprocess["imagenet_mean"])
        stds = ", ".join(_float_literal(value) for value in preprocess["imagenet_std"])
        channels = ", ".join(json.dumps(value) for value in preprocess["channels"])
        lines.extend([
            "    {",
            f"      {json.dumps(calibration['calibration_id'])},",
            f"      {json.dumps(calibration['depth_model'])},",
            f"      {json.dumps(calibration['depth_model_url'])},",
            f"      {json.dumps(calibration['onnx_sha256'])},",
            f"      {_float_literal(calibration['raw_coordinate_scale'])},",
            "      {",
            f"        {json.dumps(preprocess['profile'])},",
            f"        {preprocess['source_closure_schema']}u,",
            f"        {json.dumps(preprocess['source_file'])},",
            f"        {json.dumps(preprocess['source_entrypoint'])},",
            f"        {json.dumps(preprocess['source_target'])},",
            f"        {preprocess['source_compile_flags']}u,",
            f"        {preprocess['source_macro_count']}u,",
            f"        {json.dumps(preprocess['source_closure_sha256'])},",
            f"        {preprocess['model_input_schema']}u,",
            f"        {json.dumps(preprocess['dtype'])},",
            f"        {json.dumps(preprocess['layout'])},",
            f"        {{{{{channels}}}}},",
            f"        {preprocess['patch_multiple']}u,",
            f"        {preprocess['maximum_dimension']}u,",
            f"        {{{{{means}}}}},",
            f"        {{{{{stds}}}}},",
            f"        {json.dumps(preprocess['stage'])},",
            "      },",
            "    },",
        ])
    lines.extend([
        "  }};",
        "",
        f"  inline constexpr std::array<model_calibrated_shape_t, {len(calibrated_shapes)}> "
        "model_calibrated_shapes {{",
    ])
    lines.extend(
        f"    {{{json.dumps(calibration_id)}, {width}u, {height}u}},"
        for calibration_id, width, height in calibrated_shapes
    )
    lines.extend([
        "  }};",
        "",
        f"  inline constexpr std::array<std::array<std::uint32_t, 2>, "
        f"{len(live_field_shapes)}> subtitle_ocr_live_field_shapes {{{{",
    ])
    lines.extend(
        f"    {{{{{width}u, {height}u}}}},"
        for width, height in live_field_shapes
    )
    lines.extend([
        "  }};",
        "",
        "  constexpr std::uint32_t subtitle_ocr_field_cell_scale(",
        "    const std::uint32_t width,",
        "    const std::uint32_t height",
        "  ) {",
        "    for (const auto &shape : model_calibrated_shapes) {",
        "      if (shape.width == width && shape.height == height) {",
        "        return 1u;",
        "      }",
        "    }",
        "    for (const auto &shape : subtitle_ocr_live_field_shapes) {",
        "      if (shape[0] == width && shape[1] == height) {",
        "        return 2u;",
        "      }",
        "    }",
        "    return 0u;",
        "  }",
        "",
        "  constexpr bool subtitle_ocr_field_is_calibrated(",
        "    const std::uint32_t width,",
        "    const std::uint32_t height",
        "  ) {",
        "    return subtitle_ocr_field_cell_scale(width, height) != 0u;",
        "  }",
        "",
        "  struct subtitle_ocr_roi_t {",
        "    std::uint32_t top;",
        "    std::uint32_t bottom;",
        "",
        "    constexpr explicit operator bool() const { return bottom > top; }",
        "  };",
        "",
        "  struct subtitle_ocr_projected_row_t {",
        "    std::uint32_t value;",
        "    bool valid;",
        "",
        "    constexpr explicit operator bool() const { return valid; }",
        "  };",
        "",
        "  constexpr std::uint64_t subtitle_ocr_ceil_div(",
        "    const std::uint64_t numerator,",
        "    const std::uint64_t denominator",
        "  ) {",
        "    return denominator == 0u ? 0u :",
        "      numerator / denominator + (numerator % denominator != 0u);",
        "  }",
        "",
        "  constexpr subtitle_ocr_projected_row_t subtitle_ocr_project_row_ceil(",
        "    const std::uint32_t source_width,",
        "    const std::uint32_t source_height,",
        "    const std::uint32_t field_width,",
        "    const std::uint32_t field_height,",
        "    const std::uint32_t detector_y",
        "  ) {",
        "    if (source_width == 0u || source_height == 0u ||",
        "        detector_y > subtitle_ocr_output_height ||",
        "        !subtitle_ocr_field_is_calibrated(field_width, field_height)) {",
        "      return {0u, false};",
        "    }",
        "    const auto requested_crop_height = subtitle_ocr_ceil_div(",
        "      static_cast<std::uint64_t>(source_width) * subtitle_ocr_crop_aspect_height,",
        "      subtitle_ocr_crop_aspect_width);",
        "    const auto crop_height = requested_crop_height < source_height ?",
        "      requested_crop_height : static_cast<std::uint64_t>(source_height);",
        "    const auto crop_top = static_cast<std::uint64_t>(source_height) - crop_height;",
        "    const auto denominator = static_cast<std::uint64_t>(source_height) *",
        "      subtitle_ocr_output_height;",
        "    const auto source_y_numerator = crop_top * subtitle_ocr_output_height +",
        "      static_cast<std::uint64_t>(detector_y) * crop_height;",
        "    const auto projected = subtitle_ocr_ceil_div(",
        "      source_y_numerator * field_height, denominator);",
        "    return {",
        "      static_cast<std::uint32_t>(projected < field_height ? projected : field_height),",
        "      true",
        "    };",
        "  }",
        "",
        "  constexpr subtitle_ocr_projected_row_t subtitle_ocr_ribbon_min_bottom(",
        "    const std::uint32_t source_width,",
        "    const std::uint32_t source_height,",
        "    const std::uint32_t field_width,",
        "    const std::uint32_t field_height",
        "  ) {",
        "    return subtitle_ocr_project_row_ceil(",
        "      source_width, source_height, field_width, field_height,",
        "      subtitle_ocr_safe_row_bottom - subtitle_ocr_ribbon_bottom_tolerance_pixels",
        "    );",
        "  }",
        "",
        "  constexpr subtitle_ocr_roi_t subtitle_ocr_dynamic_roi(",
        "    const std::uint32_t source_width,",
        "    const std::uint32_t source_height,",
        "    const std::uint32_t field_width,",
        "    const std::uint32_t field_height",
        "  ) {",
        "    const auto top = subtitle_ocr_project_row_ceil(",
        "      source_width, source_height, field_width, field_height, subtitle_ocr_safe_row_top);",
        "    const auto bottom = subtitle_ocr_project_row_ceil(",
        "      source_width, source_height, field_width, field_height,",
        "      subtitle_ocr_safe_row_bottom);",
        "    return top && bottom && top.value < bottom.value ?",
        "      subtitle_ocr_roi_t {top.value, bottom.value} : subtitle_ocr_roi_t {0u, 0u};",
        "  }",
        "",
        "  constexpr bool model_calibration_supports_shape(",
        "    const model_calibration_t &calibration,",
        "    const std::uint32_t width,",
        "    const std::uint32_t height",
        "  ) {",
        "    for (const auto &shape : model_calibrated_shapes) {",
        "      if (shape.calibration_id == calibration.calibration_id &&",
        "          shape.width == width && shape.height == height) {",
        "        return true;",
        "      }",
        "    }",
        "    return false;",
        "  }",
        "",
        "  enum class constant_word_e : std::size_t {",
    ])
    lines.extend(
        f"    {_identifier(field['name'])} = {field['index']}u," for field in constants
    )
    lines.extend([
        f"    count = {len(constants)}u,",
        "  };",
        "",
        "  constexpr std::size_t constant_index(const constant_word_e word) {",
        "    return static_cast<std::size_t>(word);",
        "  }",
        "",
        "  inline constexpr std::size_t constant_float_count =",
        "    constant_index(constant_word_e::count);",
        "  inline constexpr std::size_t constant_vector_count = constant_float_count / 4u;",
        "  inline constexpr std::array<const char *, constant_float_count> "
        "constant_field_names {{",
    ])
    lines.extend(f'    "{field["name"]}",' for field in constants)
    lines.extend([
        "  }};",
        "",
        "  struct alignas(16) constants_t {",
    ])
    lines.extend(
        f"    {'float' if field['type'] == 'float32' else 'std::uint32_t'} "
        f"{_identifier(field['name'])} {{}};" for field in constants)
    lines.extend([
        "  };",
        "",
        "  static_assert(std::is_standard_layout_v<constants_t>);",
        "  static_assert(std::is_trivially_copyable_v<constants_t>);",
        "  static_assert(alignof(constants_t) == 16u);",
        "  static_assert(sizeof(constants_t) == constant_float_count * sizeof(float));",
        "  static_assert(sizeof(constants_t) % 16u == 0u);",
    ])
    lines.extend(
        f"  static_assert(offsetof(constants_t, {_identifier(field['name'])}) == "
        f"constant_index(constant_word_e::{_identifier(field['name'])}) * sizeof(float));"
        for field in constants
    )
    lines.extend([
        "",
        "  enum class frame_stat_word_e : std::size_t {",
    ])
    lines.extend(
        f"    {_identifier(field['name'])} = {field['index']}u," for field in frame_stats
    )
    lines.extend([
        f"    count = {len(frame_stats)}u,",
        "  };",
        "",
        "  constexpr std::size_t frame_stat_index(const frame_stat_word_e word) {",
        "    return static_cast<std::size_t>(word);",
        "  }",
        "",
    ])
    lines.extend(
        f"  inline constexpr std::size_t frame_stat_{_identifier(field['name'])} = "
        f"frame_stat_index(frame_stat_word_e::{_identifier(field['name'])});"
        for field in frame_stats
    )
    lines.extend([
        "",
        "  inline constexpr std::size_t frame_stats_float_count =",
        "    frame_stat_index(frame_stat_word_e::count);",
        "  inline constexpr std::size_t frame_stats_vector_count =",
        "    frame_stats_float_count / 4u;",
        "  inline constexpr std::array<const char *, frame_stats_float_count> "
        "frame_stat_names {{",
    ])
    lines.extend(f'    "{field["name"]}",' for field in frame_stats)
    lines.extend([
        "  }};",
        "",
        "  enum class state_gpu_encoding_e {",
        "    float_value,",
        "    uint_bits,",
        "  };",
        "",
        "  enum class state_word_e : std::size_t {",
    ])
    lines.extend(
        f"    {_identifier(field['name'])} = {field['index']}u," for field in fields
    )
    lines.extend([
        f"    count = {len(fields)}u,",
        "  };",
        "",
        "  constexpr std::size_t state_index(const state_word_e word) {",
        "    return static_cast<std::size_t>(word);",
        "  }",
        "",
    ])
    lines.extend(
        f"  inline constexpr std::size_t {_identifier(field['name'])} = "
        f"state_index(state_word_e::{_identifier(field['name'])});"
        for field in fields
    )
    lines.extend([
        "",
        "  inline constexpr std::size_t state_float_count =",
        "    state_index(state_word_e::count);",
        f"  inline constexpr std::size_t state_vector_width = {contract['vector_width']}u;",
        "  inline constexpr std::size_t state_vector_count =",
        "    state_float_count / state_vector_width;",
        "  using state_words_t = std::array<std::uint32_t, state_float_count>;",
        "",
        "  struct state_field_descriptor_t {",
        "    state_word_e word;",
        "    std::string_view name;",
        "    std::string_view json_type;",
        "    state_gpu_encoding_e gpu_encoding;",
        "    std::uint32_t initial_word;",
        "  };",
        "",
        f"  inline constexpr std::array<state_field_descriptor_t, {len(fields)}> "
        "state_fields {{",
    ])
    for field in fields:
        encoding = "float_value" if field["gpu_encoding"] == "float" else "uint_bits"
        lines.append(
            "    {state_word_e::%s, \"%s\", \"%s\", state_gpu_encoding_e::%s, %s},"
            % (
                _identifier(field["name"]),
                field["name"],
                field["type"],
                encoding,
                _initial_word_cpp(field, tag),
            )
        )
    lines.extend([
        "  }};",
        "",
        "  inline constexpr std::array<const char *, state_float_count> "
        "state_field_names {{",
    ])
    lines.extend(f'    "{field["name"]}",' for field in fields)
    lines.extend([
        "  }};",
        "",
        "  inline constexpr state_words_t state_initial_words {{",
    ])
    lines.extend(f"    {_initial_word_cpp(field, tag)}," for field in fields)
    lines.extend([
        "  }};",
        "",
        "  constexpr bool state_field_layout_matches_indices() {",
        "    for (std::size_t position = 0; position < state_fields.size(); ++position) {",
        "      if (state_index(state_fields[position].word) != position) {",
        "        return false;",
        "      }",
        "    }",
        "    return true;",
        "  }",
        "",
        "  constexpr bool state_word_is_uint_bits(const state_word_e word) {",
        "    return state_fields[state_index(word)].gpu_encoding ==",
        "      state_gpu_encoding_e::uint_bits;",
        "  }",
        "",
        "  constexpr bool contract_tag_is_finite_normal(const std::uint32_t value) {",
        "    const auto exponent = (value >> 23u) & 0xffu;",
        "    return exponent > 0u && exponent < 0xffu;",
        "  }",
        "",
        "  static_assert(frame_stats_float_count % 4u == 0u);",
        "  static_assert(state_float_count % state_vector_width == 0u);",
        "  static_assert(state_field_layout_matches_indices());",
        "  static_assert(contract_tag != 0u);",
        "  static_assert(contract_tag_is_finite_normal(contract_tag));",
        "  static_assert(state_initial_words[state_index(state_word_e::contract_tag_bits)] ==",
        "                contract_tag);",
        "}  // namespace models::depth_coordinate_v2",
        "",
    ])
    return "\n".join(lines)


def render_hlsl(contract: dict[str, Any]) -> str:
    fields = contract["shadow_state"]["fields"]
    constant_buffer = contract["constant_buffer"]
    constants = constant_buffer["fields"]
    frame_stats = contract["frame_stats"]["fields"]
    defaults = contract["calibrated_defaults"]
    subtitle_ocr = contract["subtitle_ocr"]
    ocr_input = subtitle_ocr["input_tensor"]
    ocr_output = subtitle_ocr["output_tensor"]
    field_policy = subtitle_ocr["field_policy"]
    ocr_record = subtitle_ocr["ocr_record"]
    plane_state = subtitle_ocr["plane_state"]
    condition_params = subtitle_ocr["condition_params"]
    calibrated_shapes = [
        (shape["width"], shape["height"])
        for calibration in contract["model_calibrations"]
        for shape in calibration["calibrated_input_shapes"]
    ]
    live_field_shapes = [
        (shape["width"], shape["height"])
        for shape in field_policy["live_field_shapes"]
    ]
    calibrated_max_dimension = max(
        max(width, height) for width, height in calibrated_shapes
    )
    limiter_container_q_limit = math.ceil(
        _float32(defaults["direct_container_limit"]) * LIMITER_Q_SCALE)
    limiter_horizontal_step_q_numerator = int(
        float(defaults["max_horizontal_slope"]) * LIMITER_Q_SCALE)
    limiter_vertical_step_q_numerator = int(
        float(defaults["max_vertical_shear"]) * LIMITER_Q_SCALE)
    shape_macros = [
        macro
        for index, (width, height) in enumerate(calibrated_shapes)
        for macro in (
            f"#define V2_MODEL_CALIBRATED_SHAPE_WIDTH_{index} {width}u",
            f"#define V2_MODEL_CALIBRATED_SHAPE_HEIGHT_{index} {height}u",
        )
    ]
    live_field_shape_macros = [
        macro
        for index, (width, height) in enumerate(live_field_shapes)
        for macro in (
            f"#define V2_SUBTITLE_LIVE_FIELD_WIDTH_{index} {width}u",
            f"#define V2_SUBTITLE_LIVE_FIELD_HEIGHT_{index} {height}u",
        )
    ]
    coarse_shape_predicate = " ||\n           ".join(
        [
            f"(field_width == V2_MODEL_CALIBRATED_SHAPE_WIDTH_{index} && "
            f"field_height == V2_MODEL_CALIBRATED_SHAPE_HEIGHT_{index})"
            for index in range(len(calibrated_shapes))
        ]
    )
    live_shape_predicate = " ||\n           ".join(
        [
            f"(field_width == V2_SUBTITLE_LIVE_FIELD_WIDTH_{index} && "
            f"field_height == V2_SUBTITLE_LIVE_FIELD_HEIGHT_{index})"
            for index in range(len(live_field_shapes))
        ]
    )
    tag = contract_tag(contract)
    components = ("x", "y", "z", "w")
    lines = [
        "// Generated by tools/sbsbench/generate_depth_coordinate_v2_contract.py.",
        "// Edit tools/sbsbench/contracts/depth-coordinate-v2-v1.json instead.",
        "#ifndef DEPTH_COORDINATE_V2_CONTRACT_GENERATED_HLSL",
        "#define DEPTH_COORDINATE_V2_CONTRACT_GENERATED_HLSL",
        "",
        f"#define V2_CONTRACT_SCHEMA {contract['schema']}u",
        f"#define V2_CONTRACT_TAG 0x{tag:08X}u",
        f"#define V2_SHADOW_STATE_WORD_COUNT {len(fields)}u",
        f"#define V2_SHADOW_STATE_VECTOR_COUNT {len(fields) // contract['vector_width']}u",
        f"#define V2_SUBTITLE_OCR_CONTRACT_SCHEMA {subtitle_ocr['schema']}u",
        *_classifier_declarations(subtitle_ocr["classifier"], "hlsl"),

        f"#define V2_SUBTITLE_CONTOUR_PALETTE_REPRESENTATIVE_INDEX_BITS {subtitle_ocr['contour_policy']['palette_representative_index_bits']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_REPRESENTATIVE_INDEX_MASK {subtitle_ocr['contour_policy']['palette_representative_index_mask']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_HUE_CHANNEL_MAX {subtitle_ocr['contour_policy']['palette_hue_channel_max']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_MERGE_HUE_NUMERATOR {subtitle_ocr['contour_policy']['palette_merge_hue_numerator']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_MERGE_HUE_DENOMINATOR {subtitle_ocr['contour_policy']['palette_merge_hue_denominator']}u",
        f"#define V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_INSET_DETECTOR_PIXELS {subtitle_ocr['contour_policy']['ordinary_search_inset_detector_pixels']}u",
        f"#define V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_CENTER_BIAS_DETECTOR_PIXELS {subtitle_ocr['contour_policy']['ordinary_search_center_bias_detector_pixels']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_GRID_COLUMNS {subtitle_ocr['contour_policy']['palette_grid_columns']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_GRID_ROWS {subtitle_ocr['contour_policy']['palette_grid_rows']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_CHANNEL_MAX {subtitle_ocr['contour_policy']['palette_channel_max']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_BIN_COUNT {subtitle_ocr['contour_policy']['palette_bin_count']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_MAX_MODES {subtitle_ocr['contour_policy']['palette_max_modes']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_BRIGHTNESS_NUMERATOR {subtitle_ocr['contour_policy']['palette_brightness_numerator']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_BRIGHTNESS_DENOMINATOR {subtitle_ocr['contour_policy']['palette_brightness_denominator']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_MATCH_RGB_NUMERATOR {subtitle_ocr['contour_policy']['palette_match_rgb_numerator']}u",
        f"#define V2_SUBTITLE_CONTOUR_PALETTE_MATCH_RGB_DENOMINATOR {subtitle_ocr['contour_policy']['palette_match_rgb_denominator']}u",
        f"#define V2_SUBTITLE_CONTOUR_OUTLINE_VALUE_NUMERATOR {subtitle_ocr['contour_policy']['outline_value_numerator']}u",
        f"#define V2_SUBTITLE_CONTOUR_OUTLINE_VALUE_DENOMINATOR {subtitle_ocr['contour_policy']['outline_value_denominator']}u",
        f"#define V2_SUBTITLE_CONTOUR_OUTLINE_CHROMA_NUMERATOR {subtitle_ocr['contour_policy']['outline_chroma_numerator']}u",
        f"#define V2_SUBTITLE_CONTOUR_OUTLINE_CHROMA_DENOMINATOR {subtitle_ocr['contour_policy']['outline_chroma_denominator']}u",
        f"#define V2_SUBTITLE_CONTOUR_OUTLINE_PROBE_RADIUS_SOURCE_PIXELS {subtitle_ocr['contour_policy']['outline_probe_radius_source_pixels']}u",
        f"#define V2_SUBTITLE_CONTOUR_LONG_LINE_CORE_HEIGHT_NUMERATOR {subtitle_ocr['contour_policy']['long_line_core_height_numerator']}u",
        f"#define V2_SUBTITLE_CONTOUR_LONG_LINE_CORE_HEIGHT_DENOMINATOR {subtitle_ocr['contour_policy']['long_line_core_height_denominator']}u",
        f"#define V2_SUBTITLE_GLYPH_MATTE_SCHEMA {subtitle_ocr['glyph_matte_policy']['schema']}u",
        f"#define V2_SUBTITLE_GLYPH_MATTE_STEPS {subtitle_ocr['glyph_matte_policy']['steps']}u",
        f"#define V2_SUBTITLE_GLYPH_MATTE_RGB_DISTANCE {_float_literal(subtitle_ocr['glyph_matte_policy']['rgb_distance'])}",
        f"#define V2_SUBTITLE_GLYPH_MATTE_BRIGHTNESS_SPLIT {_float_literal(subtitle_ocr['glyph_matte_policy']['brightness_split'])}",
        f"#define V2_SUBTITLE_GLYPH_MATTE_COLLAR_RADIUS_SOURCE_PIXELS {subtitle_ocr['glyph_matte_policy']['collar_radius_source_pixels']}u",
        f"#define V2_SUBTITLE_GLYPH_MATTE_COLLAR_RADIUS_SQUARED {subtitle_ocr['glyph_matte_policy']['collar_radius_squared']}u",
        f"#define V2_SUBTITLE_CONTOUR_SCHEMA {subtitle_ocr['contour_policy']['schema']}u",
        f"#define V2_SUBTITLE_CONTOUR_RECIPE {subtitle_ocr['contour_policy']['recipe']}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_SCHEMA {subtitle_ocr['contour_policy']['history_schema']}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_TAG 0x{subtitle_ocr['contour_policy']['history_tag']:08X}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT {subtitle_ocr['contour_policy']['history_word_count']}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_SNAPSHOT_VALID_WORD {subtitle_ocr['contour_policy']['history_snapshot_valid_word']}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_RESERVED_OFFSET {subtitle_ocr['contour_policy']['history_reserved_offset']}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_RESERVED_WORD_COUNT {subtitle_ocr['contour_policy']['history_reserved_word_count']}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_VALID_BIT {subtitle_ocr['contour_policy']['history_feature_valid_bit']}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_MASK_BIT {subtitle_ocr['contour_policy']['history_feature_mask_bit']}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_RGB_DISTANCE {_float_literal(subtitle_ocr['contour_policy']['history_rgb_distance'])}",
        f"#define V2_SUBTITLE_CONTOUR_PROBABILITY_CHANGE_RATIO {_float_literal(subtitle_ocr['contour_policy']['probability_change_ratio'])}",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_CONFIRMATION_OBSERVATIONS {subtitle_ocr['contour_policy']['history_confirmation_observations']}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_FEATURE_COUNT_SHIFT {subtitle_ocr['contour_policy']['history_feature_count_shift']}u",
        f"#define V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT {subtitle_ocr['contour_policy']['source_max_extent']}u",
        f"#define V2_SUBTITLE_CONTOUR_HISTORY_JITTER_RADIUS_SOURCE_PIXELS {subtitle_ocr['contour_policy']['history_jitter_radius_source_pixels']}u",
        f"#define V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_NUMERATOR {subtitle_ocr['contour_policy']['ordinary_search_pad_height_numerator']}u",
        f"#define V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_DENOMINATOR {subtitle_ocr['contour_policy']['ordinary_search_pad_height_denominator']}u",
        f"#define V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RADIUS_SOURCE_PIXELS {subtitle_ocr['contour_policy']['spatial_edge_radius_source_pixels']}u",
        f"#define V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE {_float_literal(subtitle_ocr['contour_policy']['spatial_edge_rgb_distance'])}",
        f"#define V2_SUBTITLE_CONTOUR_DISTANCE_EMPTY_BITS {subtitle_ocr['contour_policy']['distance_empty_bits']}u",
        f"#define V2_SUBTITLE_CONTOUR_GAUSSIAN_CENTER_OFFSET_SOURCE_PIXELS {subtitle_ocr['contour_policy']['gaussian_center_offset_source_pixels']}u",
        f"#define V2_SUBTITLE_CONTOUR_GAUSSIAN_SIGMA_SHORT_SIDE_DIVISOR {_float_literal(subtitle_ocr['contour_policy']['gaussian_sigma_short_side_divisor'])}",
        f"#define V2_SUBTITLE_CONTOUR_DISTANCE_COMPLETION_SOURCE_PIXELS {subtitle_ocr['contour_policy']['distance_completion_source_pixels']}u",
        f"#define V2_SUBTITLE_CONTOUR_GAUSSIAN_WEIGHT_ABSOLUTE_ERROR_LIMIT {_float_literal(subtitle_ocr['contour_policy']['gaussian_weight_absolute_error_limit'])}",
        f"#define V2_OCR_INPUT_N {ocr_input['shape'][0]}u",
        f"#define V2_OCR_INPUT_C {ocr_input['shape'][1]}u",
        f"#define V2_OCR_INPUT_HEIGHT {ocr_input['shape'][2]}u",
        f"#define V2_OCR_INPUT_WIDTH {ocr_input['shape'][3]}u",
        f"#define V2_OCR_OUTPUT_N {ocr_output['shape'][0]}u",
        f"#define V2_OCR_OUTPUT_C {ocr_output['shape'][1]}u",
        f"#define V2_OCR_OUTPUT_HEIGHT {ocr_output['shape'][2]}u",
        f"#define V2_OCR_OUTPUT_WIDTH {ocr_output['shape'][3]}u",
        f"#define V2_OCR_IMAGENET_MEAN_B {_float_literal(ocr_input['imagenet_mean'][0])}",
        f"#define V2_OCR_IMAGENET_MEAN_G {_float_literal(ocr_input['imagenet_mean'][1])}",
        f"#define V2_OCR_IMAGENET_MEAN_R {_float_literal(ocr_input['imagenet_mean'][2])}",
        f"#define V2_OCR_IMAGENET_STD_B {_float_literal(ocr_input['imagenet_std'][0])}",
        f"#define V2_OCR_IMAGENET_STD_G {_float_literal(ocr_input['imagenet_std'][1])}",
        f"#define V2_OCR_IMAGENET_STD_R {_float_literal(ocr_input['imagenet_std'][2])}",
        f"#define V2_OCR_RECORD_SCHEMA {ocr_record['schema']}u",
        f"#define V2_OCR_RECORD_TAG 0x{ocr_record['tag']:08X}u",
        f"#define V2_OCR_RECORD_WORD_COUNT {ocr_record['word_count']}u",
        f"#define V2_OCR_RECORD_HEADER_WORD_COUNT {ocr_record['header_word_count']}u",
        f"#define V2_OCR_BOX_WORD_COUNT {ocr_record['box_word_count']}u",
        f"#define V2_OCR_BOX_FLAG_RIBBON {ocr_record['box_flag_ribbon']}u",
        f"#define V2_OCR_BOX_KNOWN_FLAGS {ocr_record['box_known_flags']}u",
        f"#define V2_OCR_RAW_BOX_OFFSET {ocr_record['raw_box_offset']}u",
        f"#define V2_OCR_RAW_BOX_CAPACITY {ocr_record['raw_box_capacity']}u",
        f"#define V2_OCR_FINAL_BOX_OFFSET {ocr_record['final_box_offset']}u",
        f"#define V2_OCR_FINAL_BOX_CAPACITY {ocr_record['final_box_capacity']}u",
        f"#define V2_MODEL_CALIBRATED_SHAPE_COUNT {len(calibrated_shapes)}u",
        f"#define V2_MODEL_CALIBRATED_MAX_DIMENSION {calibrated_max_dimension}u",
        f"#define V2_SUBTITLE_LIVE_FIELD_SHAPE_COUNT {len(live_field_shapes)}u",
        f"#define V2_LIMITER_SERIAL_MAX_LINES {LIMITER_SERIAL_MAX_LINES}u",
        f"#define V2_LIMITER_GROUP_LINES {LIMITER_GROUP_LINES}u",
        f"#define V2_LIMITER_LINE_CHUNKS {LIMITER_LINE_CHUNKS}u",
        f"#define V2_LIMITER_Q_FRACTION_BITS {LIMITER_Q_FRACTION_BITS}u",
        f"#define V2_LIMITER_Q_SCALE {LIMITER_Q_SCALE}.0f",
        f"#define V2_LIMITER_CONTAINER_Q_LIMIT {limiter_container_q_limit}",
        f"#define V2_LIMITER_HORIZONTAL_STEP_Q_NUMERATOR "
        f"{limiter_horizontal_step_q_numerator}u",
        f"#define V2_LIMITER_VERTICAL_STEP_Q_NUMERATOR "
        f"{limiter_vertical_step_q_numerator}u",
        *shape_macros,
        *live_field_shape_macros,
        f"#define V2_OCR_SAFE_ROW_TOP {field_policy['ocr_safe_row_top']}u",
        f"#define V2_OCR_SAFE_ROW_BOTTOM {field_policy['ocr_safe_row_bottom']}u",
        f"#define V2_OCR_ACTIVE_PROBABILITY_THRESHOLD "
        f"{_float_literal(field_policy['detector_active_probability_threshold'])}",
        f"#define V2_OCR_MIN_MEAN_SCORE "
        f"{_float_literal(field_policy['detector_min_mean_score'])}",
        f"#define V2_OCR_CROP_ASPECT_WIDTH "
        f"{field_policy['source_crop_aspect_width']}u",
        f"#define V2_OCR_CROP_ASPECT_HEIGHT "
        f"{field_policy['source_crop_aspect_height']}u",
        f"#define V2_OCR_TEXT_JOIN_GAP_CELLS "
        f"{field_policy['text_join_gap_cells']}u",
        f"#define V2_OCR_ORDINARY_COVER_VERTICAL_GUARD_HEIGHT_NUMERATOR {field_policy['ordinary_cover_vertical_guard_height_numerator']}u",
        f"#define V2_OCR_ORDINARY_COVER_VERTICAL_GUARD_HEIGHT_DENOMINATOR {field_policy['ordinary_cover_vertical_guard_height_denominator']}u",
        f"#define V2_OCR_ORDINARY_COVER_HORIZONTAL_GUARD_CELLS "
        f"{field_policy['ordinary_cover_horizontal_guard_cells']}u",
        f"#define V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_NUMERATOR "
        f"{field_policy['line_group_min_vertical_iou_numerator']}u",
        f"#define V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_DENOMINATOR "
        f"{field_policy['line_group_min_vertical_iou_denominator']}u",
        f"#define V2_OCR_LINE_GROUP_MAX_HORIZONTAL_GAP_HEIGHT_NUMERATOR "
        f"{field_policy['line_group_max_horizontal_gap_height_numerator']}u",
        f"#define V2_OCR_LINE_GROUP_MAX_HORIZONTAL_GAP_HEIGHT_DENOMINATOR "
        f"{field_policy['line_group_max_horizontal_gap_height_denominator']}u",
        f"#define V2_OCR_LINE_CENTER_TOLERANCE_NUMERATOR "
        f"{field_policy['line_center_tolerance_numerator']}u",
        f"#define V2_OCR_LINE_CENTER_TOLERANCE_DENOMINATOR "
        f"{field_policy['line_center_tolerance_denominator']}u",
        f"#define V2_OCR_LINE_SEARCH_BOTTOM_HEIGHT_NUMERATOR "
        f"{field_policy['line_search_bottom_height_numerator']}u",
        f"#define V2_OCR_LINE_SEARCH_BOTTOM_HEIGHT_DENOMINATOR "
        f"{field_policy['line_search_bottom_height_denominator']}u",
        f"#define V2_OCR_RIBBON_JOIN_GAP_CELLS "
        f"{field_policy['ribbon_join_gap_cells']}u",
        f"#define V2_OCR_RIBBON_STRUCTURAL_GAP_MIN_CELLS "
        f"{field_policy['ribbon_structural_gap_min_cells']}u",
        f"#define V2_OCR_RIBBON_MIN_STRUCTURAL_GAPS "
        f"{field_policy['ribbon_min_structural_gaps']}u",
        f"#define V2_OCR_RIBBON_MIN_WIDTH_NUMERATOR "
        f"{field_policy['ribbon_min_width_numerator']}u",
        f"#define V2_OCR_RIBBON_MIN_WIDTH_DENOMINATOR "
        f"{field_policy['ribbon_min_width_denominator']}u",
        f"#define V2_OCR_RIBBON_BOTTOM_TOLERANCE_PIXELS "
        f"{field_policy['ribbon_bottom_tolerance_pixels']}u",
        f"#define V2_OCR_RIBBON_COVER_PAD_LIMIT "
        f"{field_policy['ribbon_cover_pad_limit']}u",
        f"#define V2_SUBTITLE_PLANE_STATE_SCHEMA {plane_state['schema']}u",
        f"#define V2_SUBTITLE_PLANE_STATE_WORD_COUNT {plane_state['word_count']}u",
        f"#define V2_SUBTITLE_PLANE_HEADER_WORD_COUNT {plane_state['header_word_count']}u",
        f"#define V2_SUBTITLE_PLANE_COVER_CAPACITY {plane_state['cover_capacity']}u",
        f"#define V2_SUBTITLE_PLANE_FLAGS_WORD {plane_state['flags_word']}u",
        f"#define V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG {plane_state['current_valid_flag']}u",
        f"#define V2_SUBTITLE_PLANE_CURRENT_SLOT_MASK_WORD {plane_state['current_slot_mask_word']}u",
        f"#define V2_SUBTITLE_PLANE_FRAME_OFFSET {plane_state['frame_offset']}u",
        f"#define V2_SUBTITLE_PLANE_ANALYSIS_GENERATION_OFFSET {plane_state['analysis_generation_offset']}u",
        f"#define V2_SUBTITLE_PLANE_OBSERVATION_TIMESTAMP_OFFSET {plane_state['observation_timestamp_offset']}u",
        f"#define V2_SUBTITLE_PLANE_SOURCE_EXTENT_OFFSET {plane_state['source_extent_offset']}u",
        f"#define V2_SUBTITLE_PLANE_FIELD_EXTENT_OFFSET {plane_state['field_extent_offset']}u",
        f"#define V2_SUBTITLE_PLANE_ROI_OFFSET {plane_state['roi_offset']}u",
        f"#define V2_SUBTITLE_PLANE_CONTENT_OFFSET {plane_state['content_offset']}u",
        f"#define V2_SUBTITLE_PLANE_CUT_EPOCH_WORD {plane_state['cut_epoch_word']}u",
        "// Legacy reset-mask name: current-core probability cues block neighbor fallback only.",
        f"#define V2_SUBTITLE_PLANE_HISTORY_RESET_MASK_WORD {plane_state['history_reset_mask_word']}u",
        f"#define V2_SUBTITLE_PLANE_WARMUP_WORD {plane_state['warmup_word']}u",
        f"#define V2_SUBTITLE_PLANE_WARMUP_OBSERVATIONS {plane_state['warmup_observations']}u",
        f"#define V2_SUBTITLE_PLANE_WARMUP_CUE_BIT {plane_state['warmup_cue_bit']}u",
        '#define V2_SUBTITLE_PLANE_WARMUP_REMAINING_MASK (V2_SUBTITLE_PLANE_WARMUP_CUE_BIT - 1u)',
        f"#define V2_SUBTITLE_PLANE_RESERVED_OFFSET {plane_state['reserved_offset']}u",
        f"#define V2_SUBTITLE_PLANE_RESERVED_WORD_COUNT {plane_state['reserved_word_count']}u",
        f"#define V2_SUBTITLE_PLANE_ADAPTIVE_OFFSET {plane_state['adaptive_offset']}u",
        f"#define V2_SUBTITLE_PLANE_ADAPTIVE_WORD_COUNT {plane_state['adaptive_word_count']}u",
        f"#define V2_SUBTITLE_PLANE_ADAPTIVE_KNOWN_FLAGS {plane_state['adaptive_known_flags']}u",
        f"#define V2_SUBTITLE_PLANE_ADAPTIVE_RETIRED_APPROACH_FLAG {plane_state['adaptive_retired_approach_flag']}u",
        f"#define V2_SUBTITLE_PLANE_ADAPTIVE_RETIRED_RETREAT_FLAG {plane_state['adaptive_retired_retreat_flag']}u",
        f"#define V2_SUBTITLE_PLANE_RETREAT_TIME_CONSTANT_MS {plane_state['retreat_time_constant_ms']}u",
        f"#define V2_SUBTITLE_PLANE_ADAPTIVE_PRESENTED_FLAG {plane_state['adaptive_presented_flag']}u",
        f"#define V2_SUBTITLE_PLANE_SMOOTHING_TIME_CONSTANT_MS {plane_state['smoothing_time_constant_ms']}u",
        f"#define V2_SUBTITLE_PLANE_SLEW_SOURCE_PIXELS_PER_SECOND {plane_state['slew_source_pixels_per_second']}u",
        f"#define V2_SUBTITLE_PLANE_MAX_TICK_GAP_MS {plane_state['max_tick_gap_ms']}u",
        f"#define V2_SUBTITLE_PLANE_STATE_TAG 0x{plane_state['tag']:08X}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_SCHEMA {condition_params['schema']}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_TAG 0x{condition_params['tag']:08X}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_WORD_COUNT {condition_params['word_count']}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_CURRENT_COUNT_WORD {condition_params['current_count_word']}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_CURRENT_KIND_MASK_WORD {condition_params['current_kind_mask_word']}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_ACTIVE_WORD {condition_params['active_word']}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_FULLY_ACTIVE_VALUE {condition_params['fully_active_value']}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_TARGET_WORD {condition_params['target_word']}u",
        "#define V2_OCR_SHADER_CONTRACT_GENERATED 1u",
        f"#define V2_DIRECT_CONTAINER_LIMIT "
        f"{_float_literal(defaults['direct_container_limit'])}",
        f"#define V2_MAX_VERTICAL_SHEAR "
        f"{_float_literal(defaults['max_vertical_shear'])}",
        f"#define V2_VERTICAL_MAJORANT_SHARE "
        f"{_float_literal(defaults['vertical_majorant_share'])}",
        # The live pixel shader authenticates the published convergence curve against this
        # compile-time contract value. It must never read the producer constant buffer, which is
        # not bound at the warp draw's pixel stage.
        f"#define V2_CONVERGENCE_CURVE_DEFAULT "
        f"{_float_literal(defaults['convergence_curve_default'])}",
        f"#define V2_ADAPTIVE_POLICY_ID {defaults['adaptive_policy_id']}u",
        *[item for name in EXPECTED_DEFAULT_NAMES if name != "adaptive_policy_id" and name.startswith(("adaptive_", "host_percentile_"))
          for item in (
              f"#define V2_{name.upper()} {_float_literal(defaults[name])}",
              f"static const float v2_{name} = V2_{name.upper()};")],
        "static const float v2_max_vertical_shear = V2_MAX_VERTICAL_SHEAR;",
        "static const float v2_vertical_majorant_share = V2_VERTICAL_MAJORANT_SHARE;",
        "",
        "uint V2SubtitleOcrFieldCellScale(uint field_width, uint field_height) {",
        f"    if ({coarse_shape_predicate}) return 1u;",
        f"    if ({live_shape_predicate}) return 2u;",
        "    return 0u;",
        "}",
        "",
        "bool V2SubtitleOcrFieldIsCalibrated(uint field_width, uint field_height) {",
        "    return V2SubtitleOcrFieldCellScale(field_width, field_height) != 0u;",
        "}",
        "",
        "// Resolve the authenticated bottom crop without uint64 arithmetic.",
        "bool V2SubtitleOcrComputeCrop(",
        "    uint source_width, uint source_height, out uint crop_top, out uint crop_height) {",
        "    crop_top = 0u;",
        "    crop_height = 0u;",
        "    if (source_width == 0u || source_height == 0u) return false;",
        "    uint crop_quotient = source_width / V2_OCR_CROP_ASPECT_WIDTH;",
        "    uint crop_remainder = source_width % V2_OCR_CROP_ASPECT_WIDTH;",
        "    if (crop_quotient > 0xffffffffu / V2_OCR_CROP_ASPECT_HEIGHT ||",
        "        crop_remainder > 0xffffffffu / V2_OCR_CROP_ASPECT_HEIGHT) return false;",
        "    crop_height = min(",
        "        source_height,",
        "        crop_quotient * V2_OCR_CROP_ASPECT_HEIGHT +",
        "        (crop_remainder * V2_OCR_CROP_ASPECT_HEIGHT +",
        "         V2_OCR_CROP_ASPECT_WIDTH - 1u) / V2_OCR_CROP_ASPECT_WIDTH);",
        "    crop_top = source_height - crop_height;",
        "    return crop_height != 0u;",
        "}",
        "",
        "// Exact ceil projection of a detector row edge through that crop into the real",
        "// tensor-content rectangle.",
        "bool V2SubtitleOcrProjectContentRowCeil(",
        "    uint source_width, uint source_height, uint field_width, uint field_height,",
        "    uint4 tensor_content, uint detector_y, out uint projected_y) {",
        "    projected_y = 0u;",
        "    if (source_width == 0u || source_height == 0u || field_height == 0u ||",
        "        detector_y > V2_OCR_OUTPUT_HEIGHT ||",
        "        !V2SubtitleOcrFieldIsCalibrated(field_width, field_height) ||",
        "        tensor_content.x >= tensor_content.z || tensor_content.y >= tensor_content.w ||",
        "        tensor_content.z > field_width || tensor_content.w > field_height) return false;",
        "    uint crop_top;",
        "    uint crop_height;",
        "    if (!V2SubtitleOcrComputeCrop(",
        "            source_width, source_height, crop_top, crop_height)) return false;",
        "    if (source_height > 0xffffffffu / V2_OCR_OUTPUT_HEIGHT) return false;",
        "    uint denominator = V2_OCR_OUTPUT_HEIGHT * source_height;",
        "    uint content_height = tensor_content.w - tensor_content.y;",
        "    if (denominator == 0u || denominator > 0xffffffffu / content_height) return false;",
        "    uint source_y_numerator =",
        "        crop_top * V2_OCR_OUTPUT_HEIGHT + detector_y * crop_height;",
        "    uint scaled = source_y_numerator * content_height;",
        "    projected_y = tensor_content.y + min(",
        "        scaled / denominator + (scaled % denominator != 0u ? 1u : 0u),",
        "        content_height);",
        "    return true;",
        "}",
        "",
        f"cbuffer {constant_buffer['name']} : register({constant_buffer['register']}) {{",
    ]
    lines.extend(
        f"    {'float' if field['type'] == 'float32' else 'uint'} v2_{field['name']};"
        for field in constants)
    lines.extend([
        "};",
        "",
        f"#define V2_CONSTANT_WORD_COUNT {len(constants)}u",
        f"#define V2_CONSTANT_VECTOR_COUNT {len(constants) // contract['vector_width']}u",
        f"#define V2_FRAME_STATS_WORD_COUNT {len(frame_stats)}u",
        f"#define V2_FRAME_STATS_VECTOR_COUNT "
        f"{len(frame_stats) // contract['vector_width']}u",
        "",
    ])
    for field in frame_stats:
        upper = _upper_identifier(field["name"])
        vector = int(field["index"]) // contract["vector_width"]
        component = components[int(field["index"]) % contract["vector_width"]]
        lines.extend([
            f"#define V2_FRAME_STATS_WORD_{upper} {field['index']}u",
            f"#define V2_FRAME_STATS_VECTOR_{upper} {vector}u",
            f"#define V2_FRAME_STATS_{upper}(value) ((value).{component})",
        ])
    lines.extend([
        "",
    ])
    for field in fields:
        upper = _upper_identifier(field["name"])
        vector = int(field["index"]) // contract["vector_width"]
        component = components[int(field["index"]) % contract["vector_width"]]
        lines.extend([
            f"#define V2_STATE_WORD_{upper} {field['index']}u",
            f"#define V2_STATE_VECTOR_{upper} {vector}u",
            f"#define V2_STATE_{upper}(value) ((value).{component})",
        ])
    lines.extend(["", "#endif", ""])
    return "\n".join(lines)


def render_hlsl_ocr_assertions() -> str:
    """Render OCR8/UIP4/SCP4 consumer invariants, without historical ownership policy."""
    return "\n".join(['// Generated by tools/sbsbench/generate_depth_coordinate_v2_contract.py.',
 '// Edit tools/sbsbench/contracts/depth-coordinate-v2-v1.json instead.',
 '#ifndef DEPTH_COORDINATE_V2_OCR_ASSERT_GENERATED_HLSL',
 '#define DEPTH_COORDINATE_V2_OCR_ASSERT_GENERATED_HLSL',
 '',
 '#include "include/depth_coordinate_v2_contract.generated.hlsl"',
 '',
 '#if !defined(V2_OCR_SHADER_CONTRACT_GENERATED) || !defined(V2_SUBTITLE_PLANE_STATE_SCHEMA)',
 '#error "Complete generated V2 OCR8/UIP4 contract is required"',
 '#endif',
 '',
 '#if V2_ADAPTIVE_POLICY_ID != 3u || V2_MODEL_CALIBRATED_SHAPE_COUNT != 24 || \\',
 '    V2_OCR_SAFE_ROW_TOP >= V2_OCR_SAFE_ROW_BOTTOM || \\',
 '    V2_OCR_SAFE_ROW_BOTTOM > V2_OCR_OUTPUT_HEIGHT || \\',
 '    V2_OCR_CROP_ASPECT_WIDTH == 0u || V2_OCR_CROP_ASPECT_HEIGHT == 0u || \\',
 '    V2_OCR_CROP_ASPECT_HEIGHT > V2_OCR_CROP_ASPECT_WIDTH || \\',
 '    V2_OCR_LINE_SEARCH_BOTTOM_HEIGHT_NUMERATOR != 1u || V2_OCR_LINE_SEARCH_BOTTOM_HEIGHT_DENOMINATOR != 4u || \\',
 '    V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_NUMERATOR != 1u || V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_DENOMINATOR != 2u || \\',
 f'    V2_OCR_LINE_CENTER_TOLERANCE_NUMERATOR != {CANONICAL_SUBTITLE_OCR["field_policy"]["line_center_tolerance_numerator"]}u || '
 f'V2_OCR_LINE_CENTER_TOLERANCE_DENOMINATOR != {CANONICAL_SUBTITLE_OCR["field_policy"]["line_center_tolerance_denominator"]}u || \\',
 '    V2_OCR_BOX_FLAG_RIBBON != 1u || V2_OCR_BOX_KNOWN_FLAGS != 1u || \\',
 '    V2_OCR_TEXT_JOIN_GAP_CELLS == 0u || V2_OCR_TEXT_JOIN_GAP_CELLS >= V2_OCR_RIBBON_JOIN_GAP_CELLS || \\',
 '    V2_OCR_RIBBON_STRUCTURAL_GAP_MIN_CELLS == 0u || \\',
 '    V2_OCR_RIBBON_JOIN_GAP_CELLS < V2_OCR_RIBBON_STRUCTURAL_GAP_MIN_CELLS || \\',
 '    V2_OCR_RIBBON_MIN_STRUCTURAL_GAPS == 0u || \\',
 '    V2_OCR_RIBBON_MIN_WIDTH_DENOMINATOR == 0u || \\',
 '    V2_OCR_RIBBON_MIN_WIDTH_NUMERATOR >= V2_OCR_RIBBON_MIN_WIDTH_DENOMINATOR || \\',
 '    V2_OCR_RIBBON_BOTTOM_TOLERANCE_PIXELS >= V2_OCR_SAFE_ROW_BOTTOM || \\',
 '    V2_OCR_RIBBON_COVER_PAD_LIMIT == 0u || \\',
 '    V2_OCR_RECORD_HEADER_WORD_COUNT != V2_OCR_RAW_BOX_OFFSET || \\',
 '    V2_OCR_FINAL_BOX_OFFSET != V2_OCR_RAW_BOX_OFFSET + V2_OCR_RAW_BOX_CAPACITY * V2_OCR_BOX_WORD_COUNT || '
 '\\',
 '    V2_OCR_CLASSIFIER_HISTORY_OFFSET != V2_OCR_FINAL_BOX_OFFSET + V2_OCR_FINAL_BOX_CAPACITY * '
 'V2_OCR_BOX_WORD_COUNT || \\',
 '    V2_OCR_RECORD_WORD_COUNT != V2_OCR_CLASSIFIER_HISTORY_OFFSET + V2_OCR_CLASSIFIER_HISTORY_WORD_COUNT',
 '#error "Generated V2 OCR8 producer invariants are inconsistent"',
 '#endif',
 '',
 '#if V2_SUBTITLE_PLANE_STATE_SCHEMA != 4u || V2_SUBTITLE_PLANE_STATE_TAG != 0x34504955u || \\',
 '    V2_SUBTITLE_PLANE_STATE_WORD_COUNT != 40u || V2_SUBTITLE_PLANE_HEADER_WORD_COUNT != 24u || \\',
 '    V2_SUBTITLE_PLANE_COVER_CAPACITY != V2_OCR_FINAL_BOX_CAPACITY || \\',
 '    V2_SUBTITLE_PLANE_FLAGS_WORD != 2u || V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG != 1u || \\',
 '    V2_SUBTITLE_PLANE_CURRENT_SLOT_MASK_WORD != 3u || \\',
 '    V2_SUBTITLE_PLANE_FRAME_OFFSET != 4u || V2_SUBTITLE_PLANE_ANALYSIS_GENERATION_OFFSET != 6u || \\',
 '    V2_SUBTITLE_PLANE_OBSERVATION_TIMESTAMP_OFFSET != 8u || V2_SUBTITLE_PLANE_SOURCE_EXTENT_OFFSET != 10u '
 '|| \\',
 '    V2_SUBTITLE_PLANE_FIELD_EXTENT_OFFSET != 12u || V2_SUBTITLE_PLANE_ROI_OFFSET != 14u || \\',
 '    V2_SUBTITLE_PLANE_CONTENT_OFFSET != 16u || V2_SUBTITLE_PLANE_CUT_EPOCH_WORD != 20u || \\',
 '    V2_SUBTITLE_PLANE_HISTORY_RESET_MASK_WORD != 21u || V2_SUBTITLE_PLANE_WARMUP_WORD != 22u || V2_SUBTITLE_PLANE_WARMUP_OBSERVATIONS != 4u || V2_SUBTITLE_PLANE_WARMUP_OBSERVATIONS != V2_SUBTITLE_CONTOUR_HISTORY_CONFIRMATION_OBSERVATIONS || V2_SUBTITLE_PLANE_WARMUP_CUE_BIT != 8u || V2_SUBTITLE_PLANE_WARMUP_REMAINING_MASK != 7u || V2_SUBTITLE_PLANE_RESERVED_OFFSET != 23u || V2_SUBTITLE_PLANE_RESERVED_WORD_COUNT != 1u || \\',
 '    V2_SUBTITLE_PLANE_ADAPTIVE_OFFSET != 24u || V2_SUBTITLE_PLANE_ADAPTIVE_WORD_COUNT != 16u || \\',
 '    V2_SUBTITLE_PLANE_STATE_WORD_COUNT != V2_SUBTITLE_PLANE_ADAPTIVE_OFFSET + '
 'V2_SUBTITLE_PLANE_ADAPTIVE_WORD_COUNT || \\',
 '    V2_SUBTITLE_CONDITION_PARAM_SCHEMA != 4u || V2_SUBTITLE_CONDITION_PARAM_TAG != 0x34504353u || \\',
 '    V2_SUBTITLE_CONDITION_PARAM_WORD_COUNT != 6u || \\',
 '    V2_SUBTITLE_CONDITION_PARAM_CURRENT_COUNT_WORD != 2u || '
 'V2_SUBTITLE_CONDITION_PARAM_CURRENT_KIND_MASK_WORD != 3u || \\',
 '    V2_SUBTITLE_CONDITION_PARAM_ACTIVE_WORD != 4u || V2_SUBTITLE_CONDITION_PARAM_FULLY_ACTIVE_VALUE != 2u '
 '|| \\',
 '    V2_SUBTITLE_CONDITION_PARAM_TARGET_WORD != 5u || \\',
 '    V2_SUBTITLE_CONTOUR_SCHEMA != 5u || V2_SUBTITLE_CONTOUR_RECIPE != ' + str(CANONICAL_SUBTITLE_OCR["contour_policy"]["recipe"]) + 'u || \\',
 '    V2_SUBTITLE_CONTOUR_HISTORY_SCHEMA != 5u || V2_SUBTITLE_CONTOUR_HISTORY_TAG != 0x35544353u || \\',
 '    V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT != 32u || \\',
 '    V2_SUBTITLE_CONTOUR_HISTORY_RESERVED_OFFSET != 24u || V2_SUBTITLE_CONTOUR_HISTORY_RESERVED_WORD_COUNT '
 '!= 8u',
 '#error "Generated V2 UIP4/SCP4/SCT5 layout invariants are inconsistent"',
 '#endif',
 '',
 '#endif',
 ''])


def _write_or_check(path: Path, expected: str, check: bool) -> bool:
    if check:
        try:
            actual = path.read_text(encoding="utf-8")
        except OSError:
            print(f"missing generated contract: {path}", file=sys.stderr)
            return False
        if actual != expected:
            print(f"stale generated contract: {path}", file=sys.stderr)
            return False
        return True
    shader_manifest_generator._write_utf8_if_changed(path, expected)
    return True


def _write_hlsl_outputs(hlsl: str, ocr_assertions: str, check: bool) -> bool:
    valid = True
    for target in HLSL_TARGETS:
        valid &= _write_or_check(target, hlsl, check)
    for target in HLSL_OCR_ASSERT_TARGETS:
        valid &= _write_or_check(target, ocr_assertions, check)
    return valid


def _write_byte_mirrors(source: Path, targets: tuple[Path, ...], check: bool, label: str) -> bool:
    expected = source.read_bytes()
    valid = True
    for target in targets:
        current = target.read_bytes() if target.exists() else None
        if current == expected:
            continue
        if check:
            print(f"stale {label} mirror: {target}", file=sys.stderr)
            valid = False
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(expected)
    return valid


def _write_physical_black_model_mirrors(check: bool) -> bool:
    return _write_byte_mirrors(
        PHYSICAL_BLACK_MODEL_SOURCE, PHYSICAL_BLACK_MODEL_MIRRORS, check, "physical-black model")


def _write_cbc_unit_state_mirrors(check: bool) -> bool:
    return _write_byte_mirrors(
        CBC_UNIT_STATE_SOURCE, CBC_UNIT_STATE_MIRRORS, check, "CBC unit-state")


def _write_current_stroke_math_mirrors(check: bool) -> bool:
    return _write_byte_mirrors(
        CURRENT_STROKE_MATH_SOURCE, CURRENT_STROKE_MATH_TARGETS, check, "current stroke math")


def _write_current_outline_runtime_pins(
        check: bool, shader_root: Path = SHADER_ROOT,
        header: Path = CURRENT_OUTLINE_RUNTIME_HEADER) -> bool:
    source = header.read_text(encoding="utf-8")
    arrays = re.findall(
        r"inline constexpr std::array<cache::shader_spec,\s*(\d+)u>\s+(\w+)_specs\s*"
        r"\{\{(.*?)\}\};", source, re.DOTALL)
    pins = re.findall(r'inline constexpr std::string_view (\w+)_closure="([0-9a-f]{64})";', source)
    if (len(arrays) != len(CURRENT_OUTLINE_RUNTIME_ROOTS) or
            {name for _, name, _ in arrays} != set(CURRENT_OUTLINE_RUNTIME_ROOTS) or
            len(pins) != len(CURRENT_OUTLINE_RUNTIME_ROOTS) or
            {name for name, _ in pins} != set(CURRENT_OUTLINE_RUNTIME_ROOTS)):
        raise ValueError("current-outline runtime shader groups and pins are incomplete")
    for count, name, body in arrays:
        specs = tuple(re.findall(r'cache::shader_spec\{"([^"]+)","([^"]+)","([^"]+)"\}', body))
        if len(specs) != int(count):
            raise ValueError(f"current-outline runtime {name} shader specification is malformed")
        digest = shader_source_closure_sha256(
            shader_root / "current_outline" / CURRENT_OUTLINE_RUNTIME_ROOTS[name], specs,
            shader_compile_flags=SHADER_COMPILE_FLAGS)
        source = re.sub(
            rf'(inline constexpr std::string_view {name}_closure=")[0-9a-f]{{64}}(";)',
            rf'\g<1>{digest}\g<2>', source)
    return _write_or_check(header, source, check)


def _write_current_background_runtime_pin(
        check: bool, shader_root: Path = SHADER_ROOT,
        header: Path = CURRENT_BACKGROUND_RUNTIME_HEADER) -> bool:
    source = header.read_text(encoding="utf-8")
    array = re.search(
        r"inline constexpr std::array<cache::shader_spec,\s*(\d+)u>\s+specs\s*\{\{(.*?)\}\};",
        source, re.DOTALL)
    pin = re.search(r'inline constexpr std::string_view closure="[0-9a-f]{64}";', source)
    if array is None or pin is None:
        raise ValueError("current background runtime shader group or pin is missing")
    specs = tuple(re.findall(r'\{"([^"]+)","([^"]+)","([^"]+)"\}', array[2]))
    if len(specs) != int(array[1]):
        raise ValueError("current background runtime shader specification is malformed")
    digest = shader_source_closure_sha256(
        shader_root / "current_outline" / "background", specs, shader_compile_flags=SHADER_COMPILE_FLAGS)
    source = source[:pin.start()] + f'inline constexpr std::string_view closure="{digest}";' + source[pin.end():]
    return _write_or_check(header, source, check)


def shared_ui_warp_runtime_closures(
        shader_root: Path = SHADER_ROOT,
        header: Path = CURRENT_SHARED_UI_WARP_RUNTIME_HEADER) -> dict[str, str]:
    """Calculate the two confined Host adapters from their one runtime specification owner."""
    source = header.read_text(encoding="utf-8")
    result = {}
    for name, array_name in (("closure", "specs"), ("diagnostic_closure", "diagnostic_specs")):
        array = re.search(
            rf"inline constexpr std::array<cache::shader_spec,\s*(\d+)u>\s+{array_name}\s*\{{\{{(.*?)\}}\}};",
            source, re.DOTALL)
        if array is None:
            raise ValueError(f"shared UI warp {array_name} is missing")
        specs = tuple(re.findall(r'\{"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\}', array[2]))
        if len(specs) != int(array[1]):
            raise ValueError(f"shared UI warp {array_name} specification is malformed")
        result[name] = shader_source_closure_sha256(
            shader_root, specs, shader_compile_flags=SHADER_COMPILE_FLAGS)
    return result


def _write_shared_ui_warp_runtime_pin(
        check: bool, shader_root: Path = SHADER_ROOT,
        header: Path = CURRENT_SHARED_UI_WARP_RUNTIME_HEADER) -> bool:
    source = header.read_text(encoding="utf-8")
    for name, digest in shared_ui_warp_runtime_closures(shader_root, header).items():
        pattern = rf'(inline constexpr std::string_view {name}\s*=\s*")[0-9a-f]{{64}}(";)'
        if len(re.findall(pattern, source)) != 1:
            raise ValueError(f"shared UI warp {name} pin is missing or duplicated")
        source = re.sub(pattern, rf'\g<1>{digest}\g<2>', source)
    return _write_or_check(header, source, check)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed generated files differ from the manifest",
    )
    parser.add_argument(
        "--refresh-shader-identity",
        action="store_true",
        help=("regenerate the non-circular HLSL tag, refresh the independent complete "
              "shader closure digest in the manifest, then regenerate both language contracts"),
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check and args.refresh_shader_identity:
        parser.error("--check and --refresh-shader-identity are mutually exclusive")
    mirrors_valid = _write_physical_black_model_mirrors(args.check)
    mirrors_valid &= _write_cbc_unit_state_mirrors(args.check)
    mirrors_valid &= _write_current_stroke_math_mirrors(args.check)
    if not mirrors_valid:
        return 1
    if args.refresh_shader_identity:
        # First discard any hand-edited shader projection. The named closure manifest owns every
        # ordered root and pin; the DVC JSON stores only its generated semantic projection. Do not
        # validate the old pins yet: this is the command used precisely after a shader edit makes
        # one or more of them stale.
        shader_manifest = shader_manifest_generator.load_manifest()
        try:
            projected = shader_manifest_generator.render_dvc_manifest(
                shader_manifest, args.manifest.read_text(encoding="utf-8")
            )
        except OSError as exc:
            raise RuntimeError(
                f"could not read the DVC manifest for shader projection: {exc}"
            ) from exc
        shader_manifest_generator._write_utf8_if_changed(args.manifest, projected)
        contract = load_contract(
            args.manifest,
            verify_shader_source_closure=False,
            verify_preprocess_source_closure=False,
        )
        # The GPU tag deliberately excludes only the shader-body digest, so this first HLSL
        # generation is final. Refresh every named closure against that exact generated include;
        # the manifest generator then reprojects producer/preprocess identity into DVC.
        first_hlsl = render_hlsl(contract)
        first_ocr_assertions = render_hlsl_ocr_assertions()
        _write_hlsl_outputs(first_hlsl, first_ocr_assertions, False)
        _write_current_outline_runtime_pins(False)
        _write_current_background_runtime_pin(False)
        _write_shared_ui_warp_runtime_pin(False)
        if not shader_manifest_generator.generate(False, True):
            raise RuntimeError("could not refresh the named Host SBS shader closures")
        # generate() rewrites the closure manifest in this same process. Refresh the mutable
        # module view before validating its DVC projection; otherwise a real shader edit compares
        # the new JSON pin against the pre-refresh in-memory pin and incorrectly requires a second
        # invocation to converge.
        SHADER_MANIFEST.clear()
        SHADER_MANIFEST.update(shader_manifest_generator.load_manifest())
        if args.manifest != shader_manifest_generator.DVC_MANIFEST:
            refreshed_manifest = shader_manifest_generator.load_manifest()
            refreshed_projection = shader_manifest_generator.render_dvc_manifest(
                refreshed_manifest, args.manifest.read_text(encoding="utf-8")
            )
            shader_manifest_generator._write_utf8_if_changed(args.manifest, refreshed_projection)
        contract = load_contract(args.manifest)
        second_hlsl = render_hlsl(contract)
        second_ocr_assertions = render_hlsl_ocr_assertions()
        if (second_hlsl != first_hlsl or
                second_ocr_assertions != first_ocr_assertions):
            raise RuntimeError("shader identity refresh is not idempotent")
    else:
        contract = load_contract(args.manifest)
    valid = _write_or_check(CPP_TARGET, render_cpp(contract), args.check)
    valid &= _write_hlsl_outputs(
        render_hlsl(contract), render_hlsl_ocr_assertions(), args.check)
    valid &= _write_current_outline_runtime_pins(True)
    valid &= _write_current_background_runtime_pin(True)
    valid &= _write_shared_ui_warp_runtime_pin(True)
    try:
        validate_renderer_source_closure_pins()
    except ValueError as exc:
        print(f"stale: {exc}", file=sys.stderr)
        valid = False
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
