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
        HOST_SBS_SUBTITLE_CONTOUR_HORIZONTAL,
        HOST_SBS_SUBTITLE_CONTOUR_VERTICAL,
        HOST_SBS_SUBTITLE_CONTOUR_SNAPSHOT,
        HOST_SBS_SUBTITLE_GAUSSIAN_VERTICAL_LIMIT,
        HOST_SBS_SUBTITLE_GAUSSIAN_HORIZONTAL_LIMIT,
    ),
    source_closure_sha256="7ab5a59ec1a60b8d7764897df95ab28dde5f612b38dabdedc8f8be66541e476c",
)

PARALLAX_V2_COORDINATE_DIAGNOSTIC_GROUP = ClosureGroup(
    name="parallax_v2_coordinate_diagnostic",
    description=(
        "Dump-only canonical-coordinate producer entry point."
    ),
    specs=(
        DEPTH_COORDINATE_V2_COORDINATE_DIAGNOSTIC,
    ),
    source_closure_sha256="dfd32986c4fe9d3a9925d0c58c9b8827e3a25cd95d6b67726c7a1c4901fca134",
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
    source_closure_sha256="39f10cb7fce278d93d60898aaec16f6ac593fb629fbd517e361b1cc48e52fd19",
)

GPU_TRACE_GROUP = ClosureGroup(
    name="gpu_trace",
    description=(
        "Optional diagnostic GPU completion trace."
    ),
    specs=(
        HOST_SBS_GPU_TRACE,
    ),
    source_closure_sha256="44f412f253a201aabea111688f6c5b67769abe41a0d4b5697d103e42303a9ea9",
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
    source_closure_sha256="3914683f4b325648676935065e8029792819d85aae535ec446bb31feef240783",
)

PARALLAX_V2_P010_Y_GROUP = ClosureGroup(
    name="parallax_v2_p010_y",
    description=(
        "Optional fail-open P010 luma MRT renderer."
    ),
    specs=(
        PARALLAX_V2_P010_Y_RENDERER,
    ),
    source_closure_sha256="b365e2f96247e89280f73f42af72380087576b1b7253fabba916b272d106c2ce",
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
    source_closure_sha256="824ad9a4099801e38afc352a988deae5e5635eb69b922d396494b07f046a333c",
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
    source_closure_sha256="14873875da041fb60bbe1dd59196cfa316ce6b2dc537359bb7282c3dee18511a",
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
    source_closure_sha256="6d248d05f6c3cf56029e07493de747d36f25be20e57c892964be1a40b483c85c",
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
    source_closure_sha256="f77e92b943f7807ea90a8f3c5b8b34370e2ea0befa0a568b0b94966b24ecdbf4",
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
    source_closure_sha256="98a0ab18056da96db491aebd2de5090ad7e17e0c905a2ad11d0263b1a8b61bdb",
)

PARALLAX_V2_QUEUED_RENDERER_GROUP = ClosureGroup(
    name="parallax_v2_queued_renderer",
    description=(
        "Optional queued same-source renderer; exact root token and observation guard, never completed publication. Lazy and independent of ordinary renderer/prewarm."
    ),
    specs=(
        PARALLAX_V2_QUEUED_RENDERER,
    ),
    source_closure_sha256="f696d6d26829d8d4570e2dc121a4543d21b7d1e6b4482f776abfad283331693f",
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
}
