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
DEPTH_HIST = ShaderSpec(
    source_file="depth_hist_cs.hlsl",
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
PROD_ZIPDEPTH_CONVEX2X_LIVE_MAP = ShaderSpec(
    source_file="prod_zipdepth_convex2x_live_map_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_OWNERSHIP = ShaderSpec(
    source_file="depth_coordinate_v2_ownership_cs.hlsl",
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
SBS_FLAT_IDENTITY = ShaderSpec(
    source_file="sbs_flat_identity_ps.hlsl",
    source_entrypoint="main_ps",
    source_target="ps_5_0",
)

SHADER_SPECS: Dict[str, ShaderSpec] = {
    "buffer_to_tex": BUFFER_TO_TEX,
    "buffer_to_tex_pad": BUFFER_TO_TEX_PAD,
    "depth_minmax_ema": DEPTH_MINMAX_EMA,
    "depth_hist": DEPTH_HIST,
    "depth_scene_cut_evidence": DEPTH_SCENE_CUT_EVIDENCE,
    "depth_scene_cut_resolve": DEPTH_SCENE_CUT_RESOLVE,
    "depth_coordinate_v2_moments": DEPTH_COORDINATE_V2_MOMENTS,
    "depth_coordinate_v2_frame_resolve": DEPTH_COORDINATE_V2_FRAME_RESOLVE,
    "depth_coordinate_v2_state_resolve": DEPTH_COORDINATE_V2_STATE_RESOLVE,
    "depth_coordinate_v2_map": DEPTH_COORDINATE_V2_MAP,
    "prod_zipdepth_convex2x_live_map": PROD_ZIPDEPTH_CONVEX2X_LIVE_MAP,
    "depth_coordinate_v2_ownership": DEPTH_COORDINATE_V2_OWNERSHIP,
    "depth_coordinate_v2_coordinate_diagnostic": DEPTH_COORDINATE_V2_COORDINATE_DIAGNOSTIC,
    "depth_coordinate_v2_vertical_limit": DEPTH_COORDINATE_V2_VERTICAL_LIMIT,
    "depth_coordinate_v2_limit": DEPTH_COORDINATE_V2_LIMIT,
    "host_sbs_ocr_preprocess": HOST_SBS_OCR_PREPROCESS,
    "host_sbs_ocr_cells": HOST_SBS_OCR_CELLS,
    "host_sbs_ocr_resolve": HOST_SBS_OCR_RESOLVE,
    "host_sbs_subtitle_locator_resolve": HOST_SBS_SUBTITLE_LOCATOR_RESOLVE,
    "host_sbs_subtitle_condition": HOST_SBS_SUBTITLE_CONDITION,
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
    "sbs_flat_identity": SBS_FLAT_IDENTITY,
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
        DEPTH_HIST,
        DEPTH_SCENE_CUT_EVIDENCE,
        DEPTH_SCENE_CUT_RESOLVE,
        DEPTH_COORDINATE_V2_MOMENTS,
        DEPTH_COORDINATE_V2_FRAME_RESOLVE,
        DEPTH_COORDINATE_V2_STATE_RESOLVE,
        DEPTH_COORDINATE_V2_MAP,
        PROD_ZIPDEPTH_CONVEX2X_LIVE_MAP,
        DEPTH_COORDINATE_V2_OWNERSHIP,
        DEPTH_COORDINATE_V2_VERTICAL_LIMIT,
        DEPTH_COORDINATE_V2_LIMIT,
        HOST_SBS_OCR_PREPROCESS,
        HOST_SBS_OCR_CELLS,
        HOST_SBS_OCR_RESOLVE,
        HOST_SBS_SUBTITLE_LOCATOR_RESOLVE,
        HOST_SBS_SUBTITLE_CONDITION,
    ),
    source_closure_sha256="f6d850b391cf145908d9416c51c1ced23a03721dc754eb020cc52236a95629b3",
)

PARALLAX_V2_COORDINATE_DIAGNOSTIC_GROUP = ClosureGroup(
    name="parallax_v2_coordinate_diagnostic",
    description=(
        "Dump-only canonical-coordinate producer entry point."
    ),
    specs=(
        DEPTH_COORDINATE_V2_COORDINATE_DIAGNOSTIC,
    ),
    source_closure_sha256="9c52f0e3b70244d6ba7c2865ec14fe5367388fa36dbdcf51b1037f11af3ad5ad",
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
    source_closure_sha256="c4443d416964e1437dd7ca4d78ca4781aea2835b159b03e8856456890ef006a4",
)

GPU_TRACE_GROUP = ClosureGroup(
    name="gpu_trace",
    description=(
        "Optional diagnostic GPU completion trace."
    ),
    specs=(
        HOST_SBS_GPU_TRACE,
    ),
    source_closure_sha256="ec553c5cd1bb095ae84df2ccbf866413dc0bfe7b251e6dd3591f2a6c74ff96bf",
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
    source_closure_sha256="82f983e406d5ac5d034ce86745c923ad9266a8b93a27a598325b7c04c8dd2589",
)

PARALLAX_V2_P010_Y_GROUP = ClosureGroup(
    name="parallax_v2_p010_y",
    description=(
        "Optional fail-open P010 luma MRT renderer."
    ),
    specs=(
        PARALLAX_V2_P010_Y_RENDERER,
    ),
    source_closure_sha256="ab499f7edf78ba474d5b4e34137200e70391d65ebba43ef85914c8d712c66855",
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
    source_closure_sha256="1497c6a7b8bf42e0ec8485d5b810f817f8be9810bb57fed575a8a634973c746b",
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
}
