"""Generated Host SBS shader closure registry; do not edit by hand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

MANIFEST_SCHEMA = 1
SOURCE_CLOSURE_SCHEMA = 2
SHADER_COMPILE_FLAGS = 0x00008800
SOURCE_MACRO_COUNT = 0


@dataclass(frozen=True)
class ShaderSpec:
    source_file: str
    source_entrypoint: str
    source_target: str


@dataclass(frozen=True)
class ClosureGroup:
    name: str
    description: str
    specs: Tuple[ShaderSpec, ...]
    source_closure_sha256: str


BUFFER_TO_TEX = ShaderSpec(
    source_file="buffer_to_tex_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
BUFFER_TO_TEX_PAD = ShaderSpec(
    source_file="buffer_to_tex_cs.hlsl",
    source_entrypoint="pad_main",
    source_target="cs_5_0",
)
DEPTH_MINMAX_EMA = ShaderSpec(
    source_file="depth_minmax_ema_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_SCENE_CUT_EVIDENCE = ShaderSpec(
    source_file="depth_scene_cut_evidence_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_SCENE_CUT_RESOLVE = ShaderSpec(
    source_file="depth_scene_cut_resolve_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_MOMENTS = ShaderSpec(
    source_file="depth_coordinate_v2_moments_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_FRAME_RESOLVE = ShaderSpec(
    source_file="depth_coordinate_v2_frame_resolve_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_HISTOGRAM = ShaderSpec(
    source_file="depth_coordinate_v2_histogram_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_QUANTILES = ShaderSpec(
    source_file="depth_coordinate_v2_quantiles_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_STATE_RESOLVE = ShaderSpec(
    source_file="depth_coordinate_v2_state_resolve_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_MAP = ShaderSpec(
    source_file="depth_coordinate_v2_map_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_COORDINATE_DIAGNOSTIC = ShaderSpec(
    source_file="depth_coordinate_v2_map_cs.hlsl",
    source_entrypoint="coordinate_main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_VERTICAL_LIMIT = ShaderSpec(
    source_file="depth_coordinate_v2_vertical_limit_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_LIMIT = ShaderSpec(
    source_file="depth_coordinate_v2_limit_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
HOST_SBS_OCR_PREPROCESS = ShaderSpec(
    source_file="host_sbs_ocr_preprocess_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
HOST_SBS_OCR_CELLS = ShaderSpec(
    source_file="host_sbs_ocr_boxes_cs.hlsl",
    source_entrypoint="cells_main",
    source_target="cs_5_0",
)
HOST_SBS_OCR_RESOLVE = ShaderSpec(
    source_file="host_sbs_ocr_boxes_cs.hlsl",
    source_entrypoint="resolve_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_LOCATOR_RESOLVE = ShaderSpec(
    source_file="host_sbs_subtitle_locator_cs.hlsl",
    source_entrypoint="resolve_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_CONDITION = ShaderSpec(
    source_file="host_sbs_subtitle_locator_cs.hlsl",
    source_entrypoint="condition_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_CONTOUR_TEMPORAL = ShaderSpec(
    source_file="host_sbs_subtitle_contour_cs.hlsl",
    source_entrypoint="temporal_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_CONTOUR_DENSE = ShaderSpec(
    source_file="host_sbs_subtitle_contour_cs.hlsl",
    source_entrypoint="dense_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_CONTOUR_PROJECT = ShaderSpec(
    source_file="host_sbs_subtitle_contour_cs.hlsl",
    source_entrypoint="project_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_CONTOUR_HISTORY = ShaderSpec(
    source_file="host_sbs_subtitle_contour_cs.hlsl",
    source_entrypoint="history_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_CONTOUR_HISTORY_IDENTITY = ShaderSpec(
    source_file="host_sbs_subtitle_contour_cs.hlsl",
    source_entrypoint="history_identity_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_CONTOUR_HORIZONTAL = ShaderSpec(
    source_file="host_sbs_subtitle_contour_distance_cs.hlsl",
    source_entrypoint="horizontal_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_CONTOUR_VERTICAL = ShaderSpec(
    source_file="host_sbs_subtitle_contour_distance_cs.hlsl",
    source_entrypoint="vertical_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_CONTOUR_SNAPSHOT = ShaderSpec(
    source_file="host_sbs_subtitle_contour_cs.hlsl",
    source_entrypoint="snapshot_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_GAUSSIAN_VERTICAL_LIMIT = ShaderSpec(
    source_file="depth_coordinate_v2_vertical_limit_cs.hlsl",
    source_entrypoint="subtitle_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_GAUSSIAN_HORIZONTAL_LIMIT = ShaderSpec(
    source_file="depth_coordinate_v2_limit_cs.hlsl",
    source_entrypoint="subtitle_main",
    source_target="cs_5_0",
)
HOST_SBS_NEAR_IDENTICAL_FUSED_PREPROCESS = ShaderSpec(
    source_file="rgb_to_nchw_near_identical_cs.hlsl",
    source_entrypoint="fused_main",
    source_target="cs_5_0",
)
HOST_SBS_NEAR_IDENTICAL_RESOLVE = ShaderSpec(
    source_file="host_sbs_near_identical_detector_cs.hlsl",
    source_entrypoint="resolve_main",
    source_target="cs_5_0",
)
HOST_SBS_NEAR_IDENTICAL_SCENE_SEED = ShaderSpec(
    source_file="host_sbs_near_identical_detector_cs.hlsl",
    source_entrypoint="scene_seed_main",
    source_target="cs_5_0",
)
HOST_SBS_NEAR_IDENTICAL_FINALIZE = ShaderSpec(
    source_file="host_sbs_near_identical_detector_cs.hlsl",
    source_entrypoint="finalize_main",
    source_target="cs_5_0",
)
HOST_SBS_NEAR_IDENTICAL_REUSE_DEPTH = ShaderSpec(
    source_file="host_sbs_near_identical_detector_cs.hlsl",
    source_entrypoint="reuse_depth_main",
    source_target="cs_5_0",
)
HOST_SBS_GPU_TRACE = ShaderSpec(
    source_file="host_sbs_gpu_trace_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
PARALLAX_V2_LIVE_RENDERER = ShaderSpec(
    source_file="sbs_reprojection_v2_live_ps.hlsl",
    source_entrypoint="main_ps",
    source_target="ps_5_0",
)
PARALLAX_V2_P010_Y_RENDERER = ShaderSpec(
    source_file="sbs_reprojection_v2_p010_y_ps.hlsl",
    source_entrypoint="main_p010_y_ps",
    source_target="ps_5_0",
)
PARALLAX_V2_LIVE_MAPPING = ShaderSpec(
    source_file="sbs_reprojection_v2_diagnostics_ps.hlsl",
    source_entrypoint="mapping_ps",
    source_target="ps_5_0",
)
PARALLAX_V2_LIVE_MASK = ShaderSpec(
    source_file="sbs_reprojection_v2_diagnostics_ps.hlsl",
    source_entrypoint="mask_ps",
    source_target="ps_5_0",
)
SBS_REPROJECTION_VERTEX = ShaderSpec(
    source_file="sbs_reprojection_vs.hlsl",
    source_entrypoint="main_vs",
    source_target="vs_5_0",
)
HOST_SBS_SUBTITLE_COHORT_PREPARE = ShaderSpec(
    source_file="host_sbs_subtitle_cohort_cs.hlsl",
    source_entrypoint="prepare_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_COHORT_MEMBERS = ShaderSpec(
    source_file="host_sbs_subtitle_cohort_cs.hlsl",
    source_entrypoint="members_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_COHORT_FILTER = ShaderSpec(
    source_file="host_sbs_subtitle_cohort_cs.hlsl",
    source_entrypoint="filter_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_COHORT_SNAPSHOT = ShaderSpec(
    source_file="host_sbs_subtitle_cohort_snapshot_cs.hlsl",
    source_entrypoint="snapshot_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_MATTE_GUARD = ShaderSpec(
    source_file="host_sbs_subtitle_matte_guard_cs.hlsl",
    source_entrypoint="guard_main",
    source_target="cs_5_0",
)
SBS_FLAT_IDENTITY = ShaderSpec(
    source_file="sbs_flat_identity_ps.hlsl",
    source_entrypoint="main_ps",
    source_target="ps_5_0",
)
PARALLAX_V2_QUEUED_RENDERER = ShaderSpec(
    source_file="sbs_reprojection_v2_queued_ps.hlsl",
    source_entrypoint="main_ps",
    source_target="ps_5_0",
)
HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_PALETTE = ShaderSpec(
    source_file="host_sbs_subtitle_inner_colour_global_validation_cs.hlsl",
    source_entrypoint="palette_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_EMIT = ShaderSpec(
    source_file="host_sbs_subtitle_inner_colour_global_validation_cs.hlsl",
    source_entrypoint="emit_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_CLEAR = ShaderSpec(
    source_file="host_sbs_subtitle_inner_colour_global_validation_cs.hlsl",
    source_entrypoint="clear_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_ACCUMULATE = ShaderSpec(
    source_file="host_sbs_subtitle_inner_colour_global_validation_cs.hlsl",
    source_entrypoint="accumulate_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_VALIDATE = ShaderSpec(
    source_file="host_sbs_subtitle_inner_colour_global_validation_cs.hlsl",
    source_entrypoint="validate_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_INIT = ShaderSpec(
    source_file="host_sbs_subtitle_seeded_components_cs.hlsl",
    source_entrypoint="init_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_UNION = ShaderSpec(
    source_file="host_sbs_subtitle_seeded_components_cs.hlsl",
    source_entrypoint="union_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_REDUCE = ShaderSpec(
    source_file="host_sbs_subtitle_seeded_components_cs.hlsl",
    source_entrypoint="reduce_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_RESOLVE = ShaderSpec(
    source_file="host_sbs_subtitle_seeded_components_cs.hlsl",
    source_entrypoint="resolve_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_DILATE = ShaderSpec(
    source_file="host_sbs_subtitle_seeded_components_cs.hlsl",
    source_entrypoint="dilate_main",
    source_target="cs_5_0",
)
HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_DILATE_BOX = ShaderSpec(
    source_file="host_sbs_subtitle_seeded_components_cs.hlsl",
    source_entrypoint="dilate_box_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_MASK_HOLES_CLEAR = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_mask_holes_cs.hlsl",
    source_entrypoint="clear_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_MASK_HOLES_CAPTURE_UNIT = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_mask_holes_cs.hlsl",
    source_entrypoint="capture_unit_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_MASK_HOLES_PREPARE_HOLES = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_mask_holes_cs.hlsl",
    source_entrypoint="prepare_holes_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_MASK_HOLES_FREEZE_FINAL = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_mask_holes_cs.hlsl",
    source_entrypoint="freeze_final_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_MASK_HOLES_INIT = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_mask_holes_cs.hlsl",
    source_entrypoint="init_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_MASK_HOLES_UNION = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_mask_holes_cs.hlsl",
    source_entrypoint="union_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_MASK_HOLES_REDUCE = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_mask_holes_cs.hlsl",
    source_entrypoint="reduce_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_MASK_HOLES_PUBLISH = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_mask_holes_cs.hlsl",
    source_entrypoint="publish_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_MASK_HOLES_COUNTER = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_mask_holes_cs.hlsl",
    source_entrypoint="counter_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_STROKE_COVERAGE_CLEAR_OFFERS = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_stroke_coverage_cs.hlsl",
    source_entrypoint="clear_offers_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_STROKE_COVERAGE_PREPARE = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_stroke_coverage_cs.hlsl",
    source_entrypoint="prepare_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_STROKE_COVERAGE_SUPPORT = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_stroke_coverage_cs.hlsl",
    source_entrypoint="support_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_FOREGROUND_EDGE = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_foreground_edge_cs.hlsl",
    source_entrypoint="foreground_edge_main",
    source_target="cs_5_0",
)
HOST_SBS_CURRENT_TRUSTED_INNER = ShaderSpec(
    source_file="current_outline/hole/host_sbs_current_foreground_edge_cs.hlsl",
    source_entrypoint="trusted_inner_publish_main",
    source_target="cs_5_0",
)

SHADER_SPECS: Dict[str, ShaderSpec] = {
    "buffer_to_tex": BUFFER_TO_TEX,
    "buffer_to_tex_pad": BUFFER_TO_TEX_PAD,
    "depth_minmax_ema": DEPTH_MINMAX_EMA,
    "depth_scene_cut_evidence": DEPTH_SCENE_CUT_EVIDENCE,
    "depth_scene_cut_resolve": DEPTH_SCENE_CUT_RESOLVE,
    "depth_coordinate_v2_moments": DEPTH_COORDINATE_V2_MOMENTS,
    "depth_coordinate_v2_frame_resolve": DEPTH_COORDINATE_V2_FRAME_RESOLVE,
    "depth_coordinate_v2_histogram": DEPTH_COORDINATE_V2_HISTOGRAM,
    "depth_coordinate_v2_quantiles": DEPTH_COORDINATE_V2_QUANTILES,
    "depth_coordinate_v2_state_resolve": DEPTH_COORDINATE_V2_STATE_RESOLVE,
    "depth_coordinate_v2_map": DEPTH_COORDINATE_V2_MAP,
    "depth_coordinate_v2_coordinate_diagnostic": DEPTH_COORDINATE_V2_COORDINATE_DIAGNOSTIC,
    "depth_coordinate_v2_vertical_limit": DEPTH_COORDINATE_V2_VERTICAL_LIMIT,
    "depth_coordinate_v2_limit": DEPTH_COORDINATE_V2_LIMIT,
    "host_sbs_ocr_preprocess": HOST_SBS_OCR_PREPROCESS,
    "host_sbs_ocr_cells": HOST_SBS_OCR_CELLS,
    "host_sbs_ocr_resolve": HOST_SBS_OCR_RESOLVE,
    "host_sbs_subtitle_locator_resolve": HOST_SBS_SUBTITLE_LOCATOR_RESOLVE,
    "host_sbs_subtitle_condition": HOST_SBS_SUBTITLE_CONDITION,
    "host_sbs_subtitle_contour_temporal": HOST_SBS_SUBTITLE_CONTOUR_TEMPORAL,
    "host_sbs_subtitle_contour_dense": HOST_SBS_SUBTITLE_CONTOUR_DENSE,
    "host_sbs_subtitle_contour_project": HOST_SBS_SUBTITLE_CONTOUR_PROJECT,
    "host_sbs_subtitle_contour_history": HOST_SBS_SUBTITLE_CONTOUR_HISTORY,
    "host_sbs_subtitle_contour_history_identity": HOST_SBS_SUBTITLE_CONTOUR_HISTORY_IDENTITY,
    "host_sbs_subtitle_contour_horizontal": HOST_SBS_SUBTITLE_CONTOUR_HORIZONTAL,
    "host_sbs_subtitle_contour_vertical": HOST_SBS_SUBTITLE_CONTOUR_VERTICAL,
    "host_sbs_subtitle_contour_snapshot": HOST_SBS_SUBTITLE_CONTOUR_SNAPSHOT,
    "host_sbs_subtitle_gaussian_vertical_limit": HOST_SBS_SUBTITLE_GAUSSIAN_VERTICAL_LIMIT,
    "host_sbs_subtitle_gaussian_horizontal_limit": HOST_SBS_SUBTITLE_GAUSSIAN_HORIZONTAL_LIMIT,
    "host_sbs_near_identical_fused_preprocess": HOST_SBS_NEAR_IDENTICAL_FUSED_PREPROCESS,
    "host_sbs_near_identical_resolve": HOST_SBS_NEAR_IDENTICAL_RESOLVE,
    "host_sbs_near_identical_scene_seed": HOST_SBS_NEAR_IDENTICAL_SCENE_SEED,
    "host_sbs_near_identical_finalize": HOST_SBS_NEAR_IDENTICAL_FINALIZE,
    "host_sbs_near_identical_reuse_depth": HOST_SBS_NEAR_IDENTICAL_REUSE_DEPTH,
    "host_sbs_gpu_trace": HOST_SBS_GPU_TRACE,
    "parallax_v2_live_renderer": PARALLAX_V2_LIVE_RENDERER,
    "parallax_v2_p010_y_renderer": PARALLAX_V2_P010_Y_RENDERER,
    "parallax_v2_live_mapping": PARALLAX_V2_LIVE_MAPPING,
    "parallax_v2_live_mask": PARALLAX_V2_LIVE_MASK,
    "sbs_reprojection_vertex": SBS_REPROJECTION_VERTEX,
    "host_sbs_subtitle_cohort_prepare": HOST_SBS_SUBTITLE_COHORT_PREPARE,
    "host_sbs_subtitle_cohort_members": HOST_SBS_SUBTITLE_COHORT_MEMBERS,
    "host_sbs_subtitle_cohort_filter": HOST_SBS_SUBTITLE_COHORT_FILTER,
    "host_sbs_subtitle_cohort_snapshot": HOST_SBS_SUBTITLE_COHORT_SNAPSHOT,
    "host_sbs_subtitle_matte_guard": HOST_SBS_SUBTITLE_MATTE_GUARD,
    "sbs_flat_identity": SBS_FLAT_IDENTITY,
    "parallax_v2_queued_renderer": PARALLAX_V2_QUEUED_RENDERER,
    "host_sbs_subtitle_inner_colour_research_palette": HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_PALETTE,
    "host_sbs_subtitle_inner_colour_research_emit": HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_EMIT,
    "host_sbs_subtitle_inner_colour_research_clear": HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_CLEAR,
    "host_sbs_subtitle_inner_colour_research_accumulate": HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_ACCUMULATE,
    "host_sbs_subtitle_inner_colour_research_validate": HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_VALIDATE,
    "host_sbs_subtitle_inner_components_research_init": HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_INIT,
    "host_sbs_subtitle_inner_components_research_union": HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_UNION,
    "host_sbs_subtitle_inner_components_research_reduce": HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_REDUCE,
    "host_sbs_subtitle_inner_components_research_resolve": HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_RESOLVE,
    "host_sbs_subtitle_inner_components_research_dilate": HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_DILATE,
    "host_sbs_subtitle_inner_components_research_dilate_box": HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_DILATE_BOX,
    "host_sbs_current_mask_holes_clear": HOST_SBS_CURRENT_MASK_HOLES_CLEAR,
    "host_sbs_current_mask_holes_capture_unit": HOST_SBS_CURRENT_MASK_HOLES_CAPTURE_UNIT,
    "host_sbs_current_mask_holes_prepare_holes": HOST_SBS_CURRENT_MASK_HOLES_PREPARE_HOLES,
    "host_sbs_current_mask_holes_freeze_final": HOST_SBS_CURRENT_MASK_HOLES_FREEZE_FINAL,
    "host_sbs_current_mask_holes_init": HOST_SBS_CURRENT_MASK_HOLES_INIT,
    "host_sbs_current_mask_holes_union": HOST_SBS_CURRENT_MASK_HOLES_UNION,
    "host_sbs_current_mask_holes_reduce": HOST_SBS_CURRENT_MASK_HOLES_REDUCE,
    "host_sbs_current_mask_holes_publish": HOST_SBS_CURRENT_MASK_HOLES_PUBLISH,
    "host_sbs_current_mask_holes_counter": HOST_SBS_CURRENT_MASK_HOLES_COUNTER,
    "host_sbs_current_stroke_coverage_clear_offers": HOST_SBS_CURRENT_STROKE_COVERAGE_CLEAR_OFFERS,
    "host_sbs_current_stroke_coverage_prepare": HOST_SBS_CURRENT_STROKE_COVERAGE_PREPARE,
    "host_sbs_current_stroke_coverage_support": HOST_SBS_CURRENT_STROKE_COVERAGE_SUPPORT,
    "host_sbs_current_foreground_edge": HOST_SBS_CURRENT_FOREGROUND_EDGE,
    "host_sbs_current_trusted_inner": HOST_SBS_CURRENT_TRUSTED_INNER,
}

PREPROCESS_GROUP = ClosureGroup(
    name="preprocess",
    description=(
        "Model-calibration identity closure; production compiles the fused root from the full producer closure."
    ),
    specs=(
        HOST_SBS_NEAR_IDENTICAL_FUSED_PREPROCESS,
    ),
    source_closure_sha256="943f3295e6cdb490d0833d981b153a5cda9a5153696eb5c9ca0042e474d8d744",
)

PARALLAX_V2_PRODUCER_GROUP = ClosureGroup(
    name="parallax_v2_producer",
    description=(
        "Complete authenticated DAV2, OCR, subtitle-locator, and final-coordinate producer."
    ),
    specs=(
        HOST_SBS_NEAR_IDENTICAL_FUSED_PREPROCESS,
        BUFFER_TO_TEX,
        BUFFER_TO_TEX_PAD,
        DEPTH_MINMAX_EMA,
        DEPTH_SCENE_CUT_EVIDENCE,
        DEPTH_SCENE_CUT_RESOLVE,
        DEPTH_COORDINATE_V2_MOMENTS,
        DEPTH_COORDINATE_V2_FRAME_RESOLVE,
        DEPTH_COORDINATE_V2_HISTOGRAM,
        DEPTH_COORDINATE_V2_QUANTILES,
        DEPTH_COORDINATE_V2_STATE_RESOLVE,
        DEPTH_COORDINATE_V2_MAP,
        DEPTH_COORDINATE_V2_VERTICAL_LIMIT,
        DEPTH_COORDINATE_V2_LIMIT,
        HOST_SBS_OCR_PREPROCESS,
        HOST_SBS_OCR_CELLS,
        HOST_SBS_OCR_RESOLVE,
        HOST_SBS_SUBTITLE_LOCATOR_RESOLVE,
        HOST_SBS_SUBTITLE_CONDITION,
        HOST_SBS_SUBTITLE_CONTOUR_TEMPORAL,
        HOST_SBS_SUBTITLE_CONTOUR_DENSE,
        HOST_SBS_SUBTITLE_CONTOUR_PROJECT,
        HOST_SBS_SUBTITLE_CONTOUR_HISTORY,
        HOST_SBS_SUBTITLE_CONTOUR_HISTORY_IDENTITY,
        HOST_SBS_SUBTITLE_CONTOUR_HORIZONTAL,
        HOST_SBS_SUBTITLE_CONTOUR_VERTICAL,
        HOST_SBS_SUBTITLE_CONTOUR_SNAPSHOT,
        HOST_SBS_SUBTITLE_GAUSSIAN_VERTICAL_LIMIT,
        HOST_SBS_SUBTITLE_GAUSSIAN_HORIZONTAL_LIMIT,
    ),
    source_closure_sha256="f2c835b0e42ed2d9a550c0191b77102a07f2ff96fc9021564df72851a70fb548",
)

PARALLAX_V2_COORDINATE_DIAGNOSTIC_GROUP = ClosureGroup(
    name="parallax_v2_coordinate_diagnostic",
    description=(
        "Dump-only canonical-coordinate producer entry point."
    ),
    specs=(
        DEPTH_COORDINATE_V2_COORDINATE_DIAGNOSTIC,
    ),
    source_closure_sha256="4af4c677307e7ab977a52ecd4b4ca54ab55c62a322eca20890b51a38f7fd6314",
)

NEAR_IDENTICAL_DETECTOR_GROUP = ClosureGroup(
    name="near_identical_detector",
    description=(
        "Mandatory GPU reuse arbitration, scene seeding, finalization, and reuse-depth closure."
    ),
    specs=(
        HOST_SBS_NEAR_IDENTICAL_RESOLVE,
        HOST_SBS_NEAR_IDENTICAL_SCENE_SEED,
        HOST_SBS_NEAR_IDENTICAL_FINALIZE,
        HOST_SBS_NEAR_IDENTICAL_REUSE_DEPTH,
    ),
    source_closure_sha256="6a38ce04b5bb118467bbb3cc93c9b0d54c7982484b1101634f1c34797445a125",
)

GPU_TRACE_GROUP = ClosureGroup(
    name="gpu_trace",
    description=(
        "Optional diagnostic GPU completion trace."
    ),
    specs=(
        HOST_SBS_GPU_TRACE,
    ),
    source_closure_sha256="ac33c68d199105b142d2ebf6a4fc6437e6ebb10b1eeaf449d85bea5dbf561294",
)

PARALLAX_V2_LIVE_RENDERER_GROUP = ClosureGroup(
    name="parallax_v2_live_renderer",
    description=(
        "Authenticated live signed-parallax renderer and shared fullscreen vertex shader."
    ),
    specs=(
        PARALLAX_V2_LIVE_RENDERER,
        SBS_REPROJECTION_VERTEX,
    ),
    source_closure_sha256="aaeab938258701c2748a71888ad8a23e21a3fdd135a1a1faa022e8e48e7d4bf3",
)

PARALLAX_V2_P010_Y_GROUP = ClosureGroup(
    name="parallax_v2_p010_y",
    description=(
        "Optional fail-open P010 luma MRT renderer."
    ),
    specs=(
        PARALLAX_V2_P010_Y_RENDERER,
    ),
    source_closure_sha256="a047449579693a6cbbee7c56984166bcb3607c91713c53415fc5838ef6808efd",
)

SBS_FLAT_FALLBACK_GROUP = ClosureGroup(
    name="sbs_flat_fallback",
    description=(
        "Independent authenticated flat-identity fallback and shared fullscreen vertex shader."
    ),
    specs=(
        SBS_FLAT_IDENTITY,
        SBS_REPROJECTION_VERTEX,
    ),
    source_closure_sha256="7e45f7ca78b170c2d6c33ab5c5e20d9f45cece71a5c84e6e7fc4f0f42cfde8d4",
)

PARALLAX_V2_LIVE_DIAGNOSTIC_GROUP = ClosureGroup(
    name="parallax_v2_live_diagnostic",
    description=(
        "Dump-only live inverse-map and mask renderers."
    ),
    specs=(
        PARALLAX_V2_LIVE_MAPPING,
        PARALLAX_V2_LIVE_MASK,
    ),
    source_closure_sha256="a3ace036cb485ee61de78a57c4baa1c026adc6cd74050f653a89abb58eb73a3c",
)

SUBTITLE_COHORT_PROTOTYPE_RESEARCH_GROUP = ClosureGroup(
    name="subtitle_cohort_prototype_research",
    description=(
        "Opt-in cohort prototype provenance for the first three cohort passes; never production drawing authority."
    ),
    specs=(
        HOST_SBS_SUBTITLE_COHORT_PREPARE,
        HOST_SBS_SUBTITLE_COHORT_MEMBERS,
        HOST_SBS_SUBTITLE_COHORT_FILTER,
    ),
    source_closure_sha256="14d727e2633c941f5dda7fb899c53f526334f681ef2ec0a6d4ae31e96bfc3a46",
)

SUBTITLE_COHORT_RESEARCH_GROUP = ClosureGroup(
    name="subtitle_cohort_research",
    description=(
        "Opt-in private subtitle cohort experiment and coherent previous-observation snapshot; never rendering authority."
    ),
    specs=(
        HOST_SBS_SUBTITLE_COHORT_PREPARE,
        HOST_SBS_SUBTITLE_COHORT_MEMBERS,
        HOST_SBS_SUBTITLE_COHORT_FILTER,
        HOST_SBS_SUBTITLE_COHORT_SNAPSHOT,
    ),
    source_closure_sha256="20f83e0b0fddc179d7d1769e3cf393aa479ecd92cfba2dd4a00a3e88ed08aa80",
)

SUBTITLE_INNER_COLOUR_RESEARCH_GROUP = ClosureGroup(
    name="subtitle_inner_colour_research",
    description=(
        "Opt-in independent global inner-colour diagnostic; never replaces producer or drawing inputs."
    ),
    specs=(
        HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_PALETTE,
        HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_EMIT,
        HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_CLEAR,
        HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_ACCUMULATE,
        HOST_SBS_SUBTITLE_INNER_COLOUR_RESEARCH_VALIDATE,
    ),
    source_closure_sha256="f741ddfcfb5b489e7a10828d8857c80f7f3969fc9417b9e843fc465e2c0adf80",
)

SUBTITLE_INNER_COMPONENTS_RESEARCH_GROUP = ClosureGroup(
    name="subtitle_inner_components_research",
    description=(
        "Opt-in inner-colour six-pass component diagnostic dependency; separate from production current-outline ROI graphs."
    ),
    specs=(
        HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_INIT,
        HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_UNION,
        HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_REDUCE,
        HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_RESOLVE,
        HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_DILATE,
        HOST_SBS_SUBTITLE_INNER_COMPONENTS_RESEARCH_DILATE_BOX,
    ),
    source_closure_sha256="7e2f44690d64cebe3261133a3642fa558c8c25c7086bf19551f284a4cb13ebe0",
)

SUBTITLE_MATTE_GUARD_RESEARCH_GROUP = ClosureGroup(
    name="subtitle_matte_guard_research",
    description=(
        "Opt-in private final subtitle matte guard; separate from production drawing and cohort state."
    ),
    specs=(
        HOST_SBS_SUBTITLE_MATTE_GUARD,
    ),
    source_closure_sha256="8658c5dc385c3f87613bebd91951a5e0460a1dd2818f505dbdf34dff309805a3",
)

PARALLAX_V2_QUEUED_RENDERER_GROUP = ClosureGroup(
    name="parallax_v2_queued_renderer",
    description=(
        "Optional queued same-source renderer; exact root token and observation guard, never completed publication. Lazy and independent of ordinary renderer/prewarm."
    ),
    specs=(
        PARALLAX_V2_QUEUED_RENDERER,
    ),
    source_closure_sha256="ae3047c6d0a0eef72cb86d157d8932f3eec583bcc47a40133d477abda68ae840",
)

CURRENT_MASK_HOLES_GROUP = ClosureGroup(
    name="current_mask_holes",
    description=(
        "Optional current-source mask-hole refinement; independently authenticated and unavailable on any optional failure. Separate from the mandatory producer closure."
    ),
    specs=(
        HOST_SBS_CURRENT_MASK_HOLES_CLEAR,
        HOST_SBS_CURRENT_MASK_HOLES_CAPTURE_UNIT,
        HOST_SBS_CURRENT_MASK_HOLES_PREPARE_HOLES,
        HOST_SBS_CURRENT_MASK_HOLES_FREEZE_FINAL,
        HOST_SBS_CURRENT_MASK_HOLES_INIT,
        HOST_SBS_CURRENT_MASK_HOLES_UNION,
        HOST_SBS_CURRENT_MASK_HOLES_REDUCE,
        HOST_SBS_CURRENT_MASK_HOLES_PUBLISH,
        HOST_SBS_CURRENT_MASK_HOLES_COUNTER,
    ),
    source_closure_sha256="b4e48d5eecab2404951602ad649980001f21cf5f9768381acadc7c3973ef4668",
)

CURRENT_STROKE_COVERAGE_GROUP = ClosureGroup(
    name="current_stroke_coverage",
    description=(
        "Optional current-source stroke-coverage experiment; independently authenticated, with original-source proof and separate offers. Unavailable on optional failure and separate from the mandatory producer closure."
    ),
    specs=(
        HOST_SBS_CURRENT_STROKE_COVERAGE_CLEAR_OFFERS,
        HOST_SBS_CURRENT_STROKE_COVERAGE_PREPARE,
        HOST_SBS_CURRENT_STROKE_COVERAGE_SUPPORT,
    ),
    source_closure_sha256="1f03ba4279f6cadbaddf56b408cf20c0bc37776c4971e8e25830a8ffd3a3d063",
)

CURRENT_FOREGROUND_EDGE_GROUP = ClosureGroup(
    name="current_foreground_edge",
    description=(
        "Optional current-source foreground antialias recovery from an immutable post-completion mask, current qualified foreground seeds and a common foreground/outline RGB interval. Failure preserves the existing mask."
    ),
    specs=(
        HOST_SBS_CURRENT_FOREGROUND_EDGE,
    ),
    source_closure_sha256="968b3fce0bedb875df9f09540a04256e877ec2ae7cecbdbaf0a1ccd54065b315",
)

CURRENT_TRUSTED_INNER_GROUP = ClosureGroup(
    name="current_trusted_inner",
    description=(
        "Optional current-source repeated-colour inner-fill recovery, independently accumulated and published after existing outline and foreground-edge completion. Failure preserves the existing mask."
    ),
    specs=(
        HOST_SBS_CURRENT_TRUSTED_INNER,
    ),
    source_closure_sha256="2a6749300ff4907ae2b9b806854d285c1238b93f32f31e62ba25260c15e4f06e",
)

CLOSURE_GROUPS: Dict[str, ClosureGroup] = {
    "preprocess": PREPROCESS_GROUP,
    "parallax_v2_producer": PARALLAX_V2_PRODUCER_GROUP,
    "parallax_v2_coordinate_diagnostic": PARALLAX_V2_COORDINATE_DIAGNOSTIC_GROUP,
    "near_identical_detector": NEAR_IDENTICAL_DETECTOR_GROUP,
    "gpu_trace": GPU_TRACE_GROUP,
    "parallax_v2_live_renderer": PARALLAX_V2_LIVE_RENDERER_GROUP,
    "parallax_v2_p010_y": PARALLAX_V2_P010_Y_GROUP,
    "sbs_flat_fallback": SBS_FLAT_FALLBACK_GROUP,
    "parallax_v2_live_diagnostic": PARALLAX_V2_LIVE_DIAGNOSTIC_GROUP,
    "subtitle_cohort_prototype_research": SUBTITLE_COHORT_PROTOTYPE_RESEARCH_GROUP,
    "subtitle_cohort_research": SUBTITLE_COHORT_RESEARCH_GROUP,
    "subtitle_inner_colour_research": SUBTITLE_INNER_COLOUR_RESEARCH_GROUP,
    "subtitle_inner_components_research": SUBTITLE_INNER_COMPONENTS_RESEARCH_GROUP,
    "subtitle_matte_guard_research": SUBTITLE_MATTE_GUARD_RESEARCH_GROUP,
    "parallax_v2_queued_renderer": PARALLAX_V2_QUEUED_RENDERER_GROUP,
    "current_mask_holes": CURRENT_MASK_HOLES_GROUP,
    "current_stroke_coverage": CURRENT_STROKE_COVERAGE_GROUP,
    "current_foreground_edge": CURRENT_FOREGROUND_EDGE_GROUP,
    "current_trusted_inner": CURRENT_TRUSTED_INNER_GROUP,
}
