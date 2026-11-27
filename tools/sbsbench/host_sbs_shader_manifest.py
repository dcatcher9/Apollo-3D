"""Generated Host SBS shader closure registry; do not edit by hand."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

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
    root: str
    pinned_source_closure_sha256: Optional[str]

    @property
    def source_closure_sha256(self) -> str:
        """The checked-in pin, or the closure of the current shader sources."""
        if self.pinned_source_closure_sha256 is not None:
            return self.pinned_source_closure_sha256
        return _source_closure(self)


@lru_cache(maxsize=None)
def _source_closure(group: ClosureGroup) -> str:
    try:
        from . import generate_host_sbs_shader_manifest as generator
    except ImportError:
        import generate_host_sbs_shader_manifest as generator
    return generator.shader_source_closure_sha256(
        generator.SHADER_ROOT / group.root,
        tuple((spec.source_file, spec.source_entrypoint, spec.source_target)
              for spec in group.specs),
        shader_compile_flags=SHADER_COMPILE_FLAGS,
    )


BUFFER_TO_TEX = ShaderSpec(
    source_file="buffer_to_tex_cs.hlsl",
    source_entrypoint="domain_main",
    source_target="cs_5_0",
)
BUFFER_TO_TEX_PAD = ShaderSpec(
    source_file="buffer_to_tex_cs.hlsl",
    source_entrypoint="domain_pad_main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_MOMENTS = ShaderSpec(
    source_file="depth_coordinate_v2_moments_cs.hlsl",
    source_entrypoint="domain_main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_FRAME_RESOLVE = ShaderSpec(
    source_file="depth_coordinate_v2_frame_resolve_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_HISTOGRAM = ShaderSpec(
    source_file="depth_coordinate_v2_histogram_cs.hlsl",
    source_entrypoint="domain_main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_QUANTILES = ShaderSpec(
    source_file="depth_coordinate_v2_quantiles_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_PUBLISH = ShaderSpec(
    source_file="depth_coordinate_v2_quantiles_cs.hlsl",
    source_entrypoint="publish_main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_STATE_RESOLVE = ShaderSpec(
    source_file="depth_coordinate_v2_state_resolve_cs.hlsl",
    source_entrypoint="main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_MAP = ShaderSpec(
    source_file="depth_coordinate_v2_map_cs.hlsl",
    source_entrypoint="domain_main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_MAP_COMPACT = ShaderSpec(
    source_file="depth_coordinate_v2_map_cs.hlsl",
    source_entrypoint="compact_domain_main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_COORDINATE_DIAGNOSTIC = ShaderSpec(
    source_file="depth_coordinate_v2_map_cs.hlsl",
    source_entrypoint="domain_coordinate_main",
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
DEPTH_COORDINATE_V2_VERTICAL_LIMIT_COMPACT = ShaderSpec(
    source_file="depth_coordinate_v2_vertical_limit_cs.hlsl",
    source_entrypoint="compact_main",
    source_target="cs_5_0",
)
DEPTH_COORDINATE_V2_LIMIT_COMPACT = ShaderSpec(
    source_file="depth_coordinate_v2_limit_cs.hlsl",
    source_entrypoint="compact_main",
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
HOST_SBS_SUBTITLE_CONTOUR_PALETTE = ShaderSpec(
    source_file="host_sbs_subtitle_contour_cs.hlsl",
    source_entrypoint="shared_palette_main",
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
    source_entrypoint="fused_domain_main",
    source_target="cs_5_0",
)
HOST_SBS_NEAR_IDENTICAL_TILED_PREPROCESS = ShaderSpec(
    source_file="rgb_to_nchw_near_identical_cs.hlsl",
    source_entrypoint="fused_tiled_domain_main",
    source_target="cs_5_0",
)
HOST_SBS_NEAR_IDENTICAL_RESOLVE = ShaderSpec(
    source_file="host_sbs_near_identical_detector_cs.hlsl",
    source_entrypoint="resolve_main",
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
HOST_SBS_DIRECT_GLYPH_PLAN = ShaderSpec(
    source_file="host_sbs_subtitle_direct_glyph_cs.hlsl",
    source_entrypoint="plan_direct_main",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PUBLISH = ShaderSpec(
    source_file="host_sbs_subtitle_direct_glyph_cs.hlsl",
    source_entrypoint="publish_direct_main",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_PACK_RGB = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="pack_rgb",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_PACK_TILE = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="pack_tile",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_EXPORT_ROLES = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="export_roles",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_COARSE = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="coarse",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_MAKE_HIST = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="make_hist",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_SMOOTH_HIST = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="smooth_hist",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_CHOOSE_COLORS = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="choose_colors",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_MAKE_MOMENTS = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="make_moments",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_SMOOTH_MOMENTS = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="smooth_moments",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_PROTOTYPES = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="prototypes",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PALETTE_DECODE_PALETTE = ShaderSpec(
    source_file="host_sbs_direct_glyph_palette_cs.hlsl",
    source_entrypoint="decode_palette",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_BEGIN_HISTORY = ShaderSpec(
    source_file="host_sbs_direct_glyph_history_cs.hlsl",
    source_entrypoint="begin_history",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_CONFIRM_HISTORY = ShaderSpec(
    source_file="host_sbs_direct_glyph_history_cs.hlsl",
    source_entrypoint="confirm_history",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_FINISH_WINDOWS = ShaderSpec(
    source_file="host_sbs_direct_glyph_history_cs.hlsl",
    source_entrypoint="finish_windows",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_CONFIRM_MASK_HISTORY = ShaderSpec(
    source_file="host_sbs_direct_glyph_history_cs.hlsl",
    source_entrypoint="confirm_mask_history",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PUBLISH_LINE = ShaderSpec(
    source_file="host_sbs_direct_glyph_history_cs.hlsl",
    source_entrypoint="publish_line",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_FINISH_HISTORY = ShaderSpec(
    source_file="host_sbs_direct_glyph_history_cs.hlsl",
    source_entrypoint="finish_history",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_PREPARE = ShaderSpec(
    source_file="host_sbs_shared_ui_warp_direct_glyph.hlsl",
    source_entrypoint="prepare_direct_glyph_main",
    source_target="cs_5_0",
)
HOST_SBS_SHARED_UI_WARP_INITIALIZE = ShaderSpec(
    source_file="host_sbs_shared_ui_warp.hlsl",
    source_entrypoint="initialize_main",
    source_target="cs_5_0",
)
HOST_SBS_SHARED_UI_WARP_PIN = ShaderSpec(
    source_file="host_sbs_shared_ui_warp.hlsl",
    source_entrypoint="HostBinaryPinCS",
    source_target="cs_5_0",
)
HOST_SBS_SHARED_UI_WARP_PREPARE = ShaderSpec(
    source_file="host_sbs_shared_ui_warp.hlsl",
    source_entrypoint="prepare_main",
    source_target="cs_5_0",
)
HOST_SBS_SHARED_UI_WARP_DRAW = ShaderSpec(
    source_file="host_sbs_shared_ui_warp.hlsl",
    source_entrypoint="shared_pin_main_ps",
    source_target="ps_5_0",
)
HOST_SBS_SHARED_UI_WARP_MAPPING = ShaderSpec(
    source_file="host_sbs_shared_ui_warp.hlsl",
    source_entrypoint="shared_pin_mapping_ps",
    source_target="ps_5_0",
)
HOST_SBS_SHARED_UI_WARP_COVERAGE = ShaderSpec(
    source_file="host_sbs_shared_ui_warp.hlsl",
    source_entrypoint="shared_pin_coverage_ps",
    source_target="ps_5_0",
)
HOST_SBS_SHARED_UI_WARP_BOUNDARY = ShaderSpec(
    source_file="host_sbs_shared_ui_warp.hlsl",
    source_entrypoint="shared_pin_boundary_ps",
    source_target="ps_5_0",
)
HOST_SBS_TEXT_GAME_UI_WARP_INITIALIZE = ShaderSpec(
    source_file="host_sbs_text_game_ui_warp.hlsl",
    source_entrypoint="initialize_main",
    source_target="cs_5_0",
)
HOST_SBS_TEXT_GAME_UI_WARP_APPLY = ShaderSpec(
    source_file="host_sbs_text_game_ui_warp.hlsl",
    source_entrypoint="SunshineApplyUICS",
    source_target="cs_5_0",
)
HOST_SBS_TEXT_GAME_UI_WARP_DRAW = ShaderSpec(
    source_file="host_sbs_text_game_ui_warp.hlsl",
    source_entrypoint="shared_pin_main_ps",
    source_target="ps_5_0",
)
HOST_SBS_TEXT_GAME_UI_WARP_VERIFY = ShaderSpec(
    source_file="host_sbs_text_game_ui_warp.hlsl",
    source_entrypoint="text_game_verify_main",
    source_target="cs_5_0",
)
HOST_SBS_TEXT_GAME_UI_WARP_COLUMNS = ShaderSpec(
    source_file="host_sbs_text_game_ui_warp.hlsl",
    source_entrypoint="text_game_columns_main",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_EMPTY_GATE = ShaderSpec(
    source_file="host_sbs_raw_matte_empty_gate_cs.hlsl",
    source_entrypoint="prepare_empty_main",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_PREPARE = ShaderSpec(
    source_file="direct_source_audit.hlsl",
    source_entrypoint="prepare_main",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_COPY = ShaderSpec(
    source_file="direct_source_audit.hlsl",
    source_entrypoint="copy_main",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_FINALIZE = ShaderSpec(
    source_file="direct_source_audit.hlsl",
    source_entrypoint="finalize_main",
    source_target="cs_5_0",
)
HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_ENVELOPE = ShaderSpec(
    source_file="direct_source_audit.hlsl",
    source_entrypoint="envelope_main",
    source_target="cs_5_0",
)

SHADER_SPECS: Dict[str, ShaderSpec] = {
    "buffer_to_tex": BUFFER_TO_TEX,
    "buffer_to_tex_pad": BUFFER_TO_TEX_PAD,
    "depth_coordinate_v2_moments": DEPTH_COORDINATE_V2_MOMENTS,
    "depth_coordinate_v2_frame_resolve": DEPTH_COORDINATE_V2_FRAME_RESOLVE,
    "depth_coordinate_v2_histogram": DEPTH_COORDINATE_V2_HISTOGRAM,
    "depth_coordinate_v2_quantiles": DEPTH_COORDINATE_V2_QUANTILES,
    "depth_coordinate_v2_publish": DEPTH_COORDINATE_V2_PUBLISH,
    "depth_coordinate_v2_state_resolve": DEPTH_COORDINATE_V2_STATE_RESOLVE,
    "depth_coordinate_v2_map": DEPTH_COORDINATE_V2_MAP,
    "depth_coordinate_v2_map_compact": DEPTH_COORDINATE_V2_MAP_COMPACT,
    "depth_coordinate_v2_coordinate_diagnostic": DEPTH_COORDINATE_V2_COORDINATE_DIAGNOSTIC,
    "depth_coordinate_v2_vertical_limit": DEPTH_COORDINATE_V2_VERTICAL_LIMIT,
    "depth_coordinate_v2_limit": DEPTH_COORDINATE_V2_LIMIT,
    "depth_coordinate_v2_vertical_limit_compact": DEPTH_COORDINATE_V2_VERTICAL_LIMIT_COMPACT,
    "depth_coordinate_v2_limit_compact": DEPTH_COORDINATE_V2_LIMIT_COMPACT,
    "host_sbs_ocr_preprocess": HOST_SBS_OCR_PREPROCESS,
    "host_sbs_ocr_cells": HOST_SBS_OCR_CELLS,
    "host_sbs_ocr_resolve": HOST_SBS_OCR_RESOLVE,
    "host_sbs_subtitle_locator_resolve": HOST_SBS_SUBTITLE_LOCATOR_RESOLVE,
    "host_sbs_subtitle_condition": HOST_SBS_SUBTITLE_CONDITION,
    "host_sbs_subtitle_contour_palette": HOST_SBS_SUBTITLE_CONTOUR_PALETTE,
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
    "host_sbs_near_identical_tiled_preprocess": HOST_SBS_NEAR_IDENTICAL_TILED_PREPROCESS,
    "host_sbs_near_identical_resolve": HOST_SBS_NEAR_IDENTICAL_RESOLVE,
    "host_sbs_near_identical_finalize": HOST_SBS_NEAR_IDENTICAL_FINALIZE,
    "host_sbs_near_identical_reuse_depth": HOST_SBS_NEAR_IDENTICAL_REUSE_DEPTH,
    "host_sbs_gpu_trace": HOST_SBS_GPU_TRACE,
    "parallax_v2_live_renderer": PARALLAX_V2_LIVE_RENDERER,
    "parallax_v2_p010_y_renderer": PARALLAX_V2_P010_Y_RENDERER,
    "parallax_v2_live_mapping": PARALLAX_V2_LIVE_MAPPING,
    "parallax_v2_live_mask": PARALLAX_V2_LIVE_MASK,
    "sbs_reprojection_vertex": SBS_REPROJECTION_VERTEX,
    "sbs_flat_identity": SBS_FLAT_IDENTITY,
    "host_sbs_direct_glyph_plan": HOST_SBS_DIRECT_GLYPH_PLAN,
    "host_sbs_direct_glyph_publish": HOST_SBS_DIRECT_GLYPH_PUBLISH,
    "host_sbs_direct_glyph_palette_pack_rgb": HOST_SBS_DIRECT_GLYPH_PALETTE_PACK_RGB,
    "host_sbs_direct_glyph_palette_pack_tile": HOST_SBS_DIRECT_GLYPH_PALETTE_PACK_TILE,
    "host_sbs_direct_glyph_palette_export_roles": HOST_SBS_DIRECT_GLYPH_PALETTE_EXPORT_ROLES,
    "host_sbs_direct_glyph_palette_coarse": HOST_SBS_DIRECT_GLYPH_PALETTE_COARSE,
    "host_sbs_direct_glyph_palette_make_hist": HOST_SBS_DIRECT_GLYPH_PALETTE_MAKE_HIST,
    "host_sbs_direct_glyph_palette_smooth_hist": HOST_SBS_DIRECT_GLYPH_PALETTE_SMOOTH_HIST,
    "host_sbs_direct_glyph_palette_choose_colors": HOST_SBS_DIRECT_GLYPH_PALETTE_CHOOSE_COLORS,
    "host_sbs_direct_glyph_palette_make_moments": HOST_SBS_DIRECT_GLYPH_PALETTE_MAKE_MOMENTS,
    "host_sbs_direct_glyph_palette_smooth_moments": HOST_SBS_DIRECT_GLYPH_PALETTE_SMOOTH_MOMENTS,
    "host_sbs_direct_glyph_palette_prototypes": HOST_SBS_DIRECT_GLYPH_PALETTE_PROTOTYPES,
    "host_sbs_direct_glyph_palette_decode_palette": HOST_SBS_DIRECT_GLYPH_PALETTE_DECODE_PALETTE,
    "host_sbs_direct_glyph_begin_history": HOST_SBS_DIRECT_GLYPH_BEGIN_HISTORY,
    "host_sbs_direct_glyph_confirm_history": HOST_SBS_DIRECT_GLYPH_CONFIRM_HISTORY,
    "host_sbs_direct_glyph_finish_windows": HOST_SBS_DIRECT_GLYPH_FINISH_WINDOWS,
    "host_sbs_direct_glyph_confirm_mask_history": HOST_SBS_DIRECT_GLYPH_CONFIRM_MASK_HISTORY,
    "host_sbs_direct_glyph_publish_line": HOST_SBS_DIRECT_GLYPH_PUBLISH_LINE,
    "host_sbs_direct_glyph_finish_history": HOST_SBS_DIRECT_GLYPH_FINISH_HISTORY,
    "host_sbs_direct_glyph_prepare": HOST_SBS_DIRECT_GLYPH_PREPARE,
    "host_sbs_shared_ui_warp_initialize": HOST_SBS_SHARED_UI_WARP_INITIALIZE,
    "host_sbs_shared_ui_warp_pin": HOST_SBS_SHARED_UI_WARP_PIN,
    "host_sbs_shared_ui_warp_prepare": HOST_SBS_SHARED_UI_WARP_PREPARE,
    "host_sbs_shared_ui_warp_draw": HOST_SBS_SHARED_UI_WARP_DRAW,
    "host_sbs_shared_ui_warp_mapping": HOST_SBS_SHARED_UI_WARP_MAPPING,
    "host_sbs_shared_ui_warp_coverage": HOST_SBS_SHARED_UI_WARP_COVERAGE,
    "host_sbs_shared_ui_warp_boundary": HOST_SBS_SHARED_UI_WARP_BOUNDARY,
    "host_sbs_text_game_ui_warp_initialize": HOST_SBS_TEXT_GAME_UI_WARP_INITIALIZE,
    "host_sbs_text_game_ui_warp_apply": HOST_SBS_TEXT_GAME_UI_WARP_APPLY,
    "host_sbs_text_game_ui_warp_draw": HOST_SBS_TEXT_GAME_UI_WARP_DRAW,
    "host_sbs_text_game_ui_warp_verify": HOST_SBS_TEXT_GAME_UI_WARP_VERIFY,
    "host_sbs_text_game_ui_warp_columns": HOST_SBS_TEXT_GAME_UI_WARP_COLUMNS,
    "host_sbs_direct_glyph_empty_gate": HOST_SBS_DIRECT_GLYPH_EMPTY_GATE,
    "host_sbs_direct_glyph_source_audit_prepare": HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_PREPARE,
    "host_sbs_direct_glyph_source_audit_copy": HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_COPY,
    "host_sbs_direct_glyph_source_audit_finalize": HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_FINALIZE,
    "host_sbs_direct_glyph_source_audit_envelope": HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_ENVELOPE,
}

PREPROCESS_GROUP = ClosureGroup(
    name="preprocess",
    description=(
        "Model-calibration identity closure; production compiles the fused root from the full producer closure."
    ),
    specs=(
        HOST_SBS_NEAR_IDENTICAL_FUSED_PREPROCESS,
        HOST_SBS_NEAR_IDENTICAL_TILED_PREPROCESS,
    ),
    root="",
    pinned_source_closure_sha256="8e2de9ed76bd64a38a0c956e75dd7a13c8883d39d0ef0050ffe5aba9f7208672",
)

PARALLAX_V2_PRODUCER_GROUP = ClosureGroup(
    name="parallax_v2_producer",
    description=(
        "Complete authenticated DAV2, OCR, subtitle-locator, and final-coordinate producer."
    ),
    specs=(
        HOST_SBS_NEAR_IDENTICAL_FUSED_PREPROCESS,
        HOST_SBS_NEAR_IDENTICAL_TILED_PREPROCESS,
        BUFFER_TO_TEX,
        BUFFER_TO_TEX_PAD,
        DEPTH_COORDINATE_V2_MOMENTS,
        DEPTH_COORDINATE_V2_FRAME_RESOLVE,
        DEPTH_COORDINATE_V2_HISTOGRAM,
        DEPTH_COORDINATE_V2_PUBLISH,
        DEPTH_COORDINATE_V2_MAP,
        DEPTH_COORDINATE_V2_MAP_COMPACT,
        DEPTH_COORDINATE_V2_VERTICAL_LIMIT,
        DEPTH_COORDINATE_V2_LIMIT,
        DEPTH_COORDINATE_V2_VERTICAL_LIMIT_COMPACT,
        DEPTH_COORDINATE_V2_LIMIT_COMPACT,
        HOST_SBS_OCR_PREPROCESS,
        HOST_SBS_OCR_CELLS,
        HOST_SBS_OCR_RESOLVE,
        HOST_SBS_SUBTITLE_LOCATOR_RESOLVE,
        HOST_SBS_SUBTITLE_CONDITION,
        HOST_SBS_SUBTITLE_CONTOUR_PALETTE,
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
    root="",
    pinned_source_closure_sha256="e02a43748406bbd4a3e51745c00d594b1991d7e827e077fa024cd74e042469ce",
)

PARALLAX_V2_REPLAY_RESOLVE_GROUP = ClosureGroup(
    name="parallax_v2_replay_resolve",
    description=(
        "Raw-coordinate replay quantiles and camera resolve with synthetic depth health."
    ),
    specs=(
        DEPTH_COORDINATE_V2_QUANTILES,
        DEPTH_COORDINATE_V2_STATE_RESOLVE,
    ),
    root="",
    pinned_source_closure_sha256=None,
)

PARALLAX_V2_COORDINATE_DIAGNOSTIC_GROUP = ClosureGroup(
    name="parallax_v2_coordinate_diagnostic",
    description=(
        "Dump-only canonical-coordinate producer entry point."
    ),
    specs=(
        DEPTH_COORDINATE_V2_COORDINATE_DIAGNOSTIC,
    ),
    root="",
    pinned_source_closure_sha256=None,
)

NEAR_IDENTICAL_DETECTOR_GROUP = ClosureGroup(
    name="near_identical_detector",
    description=(
        "Mandatory GPU reuse arbitration, finalization, and reuse-depth closure."
    ),
    specs=(
        HOST_SBS_NEAR_IDENTICAL_RESOLVE,
        HOST_SBS_NEAR_IDENTICAL_FINALIZE,
        HOST_SBS_NEAR_IDENTICAL_REUSE_DEPTH,
    ),
    root="",
    pinned_source_closure_sha256=None,
)

GPU_TRACE_GROUP = ClosureGroup(
    name="gpu_trace",
    description=(
        "Optional diagnostic GPU completion trace."
    ),
    specs=(
        HOST_SBS_GPU_TRACE,
    ),
    root="",
    pinned_source_closure_sha256=None,
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
    root="",
    pinned_source_closure_sha256="7092c821696d20987e56ba65d74bf4c586601c43daaaac5a5db2c690864a567b",
)

PARALLAX_V2_P010_Y_GROUP = ClosureGroup(
    name="parallax_v2_p010_y",
    description=(
        "Optional fail-open P010 luma MRT renderer."
    ),
    specs=(
        PARALLAX_V2_P010_Y_RENDERER,
    ),
    root="",
    pinned_source_closure_sha256=None,
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
    root="",
    pinned_source_closure_sha256=None,
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
    root="",
    pinned_source_closure_sha256=None,
)

DIRECT_GLYPH_PLAN_GROUP = ClosureGroup(
    name="direct_glyph_plan",
    description=(
        "Projects current OCR cores inside their original covers and publishes the direct glyph matte with its current receipt. Adaptive publication also admits pixels whose original completed score windows own every required tap."
    ),
    specs=(
        HOST_SBS_DIRECT_GLYPH_PLAN,
        HOST_SBS_DIRECT_GLYPH_PUBLISH,
    ),
    root="",
    pinned_source_closure_sha256=None,
)

DIRECT_GLYPH_PALETTE_GROUP = ClosureGroup(
    name="direct_glyph_palette",
    description=(
        "Normalizes the full OCR line, stitches mixed encoder tiles and decodes fill-plus-outline alpha at native pixel centers."
    ),
    specs=(
        HOST_SBS_DIRECT_GLYPH_PALETTE_PACK_RGB,
        HOST_SBS_DIRECT_GLYPH_PALETTE_PACK_TILE,
        HOST_SBS_DIRECT_GLYPH_PALETTE_EXPORT_ROLES,
        HOST_SBS_DIRECT_GLYPH_PALETTE_COARSE,
        HOST_SBS_DIRECT_GLYPH_PALETTE_MAKE_HIST,
        HOST_SBS_DIRECT_GLYPH_PALETTE_SMOOTH_HIST,
        HOST_SBS_DIRECT_GLYPH_PALETTE_CHOOSE_COLORS,
        HOST_SBS_DIRECT_GLYPH_PALETTE_MAKE_MOMENTS,
        HOST_SBS_DIRECT_GLYPH_PALETTE_SMOOTH_MOMENTS,
        HOST_SBS_DIRECT_GLYPH_PALETTE_PROTOTYPES,
        HOST_SBS_DIRECT_GLYPH_PALETTE_DECODE_PALETTE,
        HOST_SBS_DIRECT_GLYPH_BEGIN_HISTORY,
        HOST_SBS_DIRECT_GLYPH_CONFIRM_HISTORY,
        HOST_SBS_DIRECT_GLYPH_FINISH_WINDOWS,
        HOST_SBS_DIRECT_GLYPH_CONFIRM_MASK_HISTORY,
        HOST_SBS_DIRECT_GLYPH_PUBLISH_LINE,
        HOST_SBS_DIRECT_GLYPH_FINISH_HISTORY,
    ),
    root="",
    pinned_source_closure_sha256=None,
)

DIRECT_GLYPH_PREPARE_GROUP = ClosureGroup(
    name="direct_glyph_prepare",
    description=(
        "Completed shared-warp preparation for the direct glyph matte receipt."
    ),
    specs=(
        HOST_SBS_DIRECT_GLYPH_PREPARE,
    ),
    root="",
    pinned_source_closure_sha256=None,
)

SHARED_UI_WARP_COMPLETED_GROUP = ClosureGroup(
    name="shared_ui_warp_completed",
    description=(
        "Completed shared UI warp: pin weights, pin field, draw preparation and the pinned draw."
    ),
    specs=(
        HOST_SBS_SHARED_UI_WARP_INITIALIZE,
        HOST_SBS_SHARED_UI_WARP_PIN,
        HOST_SBS_SHARED_UI_WARP_PREPARE,
        HOST_SBS_SHARED_UI_WARP_DRAW,
    ),
    root="",
    pinned_source_closure_sha256=None,
)

SHARED_UI_WARP_COMPLETED_DIAGNOSTIC_GROUP = ClosureGroup(
    name="shared_ui_warp_completed_diagnostic",
    description=(
        "Dump-only actual-map, coverage and boundary readback of the completed shared UI warp."
    ),
    specs=(
        HOST_SBS_SHARED_UI_WARP_MAPPING,
        HOST_SBS_SHARED_UI_WARP_COVERAGE,
        HOST_SBS_SHARED_UI_WARP_BOUNDARY,
    ),
    root="",
    pinned_source_closure_sha256=None,
)

TEXT_GAME_UI_WARP_GROUP = ClosureGroup(
    name="text_game_ui_warp",
    description=(
        "Game 3D text-game route: pins the coverage a game's text helper reports through the shared warp."
    ),
    specs=(
        HOST_SBS_TEXT_GAME_UI_WARP_INITIALIZE,
        HOST_SBS_TEXT_GAME_UI_WARP_APPLY,
        HOST_SBS_TEXT_GAME_UI_WARP_DRAW,
        HOST_SBS_TEXT_GAME_UI_WARP_VERIFY,
        HOST_SBS_TEXT_GAME_UI_WARP_COLUMNS,
    ),
    root="",
    pinned_source_closure_sha256=None,
)

DIRECT_GLYPH_EMPTY_GATE_GROUP = ClosureGroup(
    name="direct_glyph_empty_gate",
    description=(
        "Publishes the current zero Direct Glyph matte for an authenticated empty OCR observation."
    ),
    specs=(
        HOST_SBS_DIRECT_GLYPH_EMPTY_GATE,
    ),
    root="",
    pinned_source_closure_sha256=None,
)

DIRECT_GLYPH_SOURCE_AUDIT_GROUP = ClosureGroup(
    name="direct_glyph_source_audit",
    description=(
        "Audits the source snapshot the Direct Glyph scores read; its sources keep their own narrow root."
    ),
    specs=(
        HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_PREPARE,
        HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_COPY,
        HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_FINALIZE,
        HOST_SBS_DIRECT_GLYPH_SOURCE_AUDIT_ENVELOPE,
    ),
    root="direct_glyph_source",
    pinned_source_closure_sha256=None,
)

CLOSURE_GROUPS: Dict[str, ClosureGroup] = {
    "preprocess": PREPROCESS_GROUP,
    "parallax_v2_producer": PARALLAX_V2_PRODUCER_GROUP,
    "parallax_v2_replay_resolve": PARALLAX_V2_REPLAY_RESOLVE_GROUP,
    "parallax_v2_coordinate_diagnostic": PARALLAX_V2_COORDINATE_DIAGNOSTIC_GROUP,
    "near_identical_detector": NEAR_IDENTICAL_DETECTOR_GROUP,
    "gpu_trace": GPU_TRACE_GROUP,
    "parallax_v2_live_renderer": PARALLAX_V2_LIVE_RENDERER_GROUP,
    "parallax_v2_p010_y": PARALLAX_V2_P010_Y_GROUP,
    "sbs_flat_fallback": SBS_FLAT_FALLBACK_GROUP,
    "parallax_v2_live_diagnostic": PARALLAX_V2_LIVE_DIAGNOSTIC_GROUP,
    "direct_glyph_plan": DIRECT_GLYPH_PLAN_GROUP,
    "direct_glyph_palette": DIRECT_GLYPH_PALETTE_GROUP,
    "direct_glyph_prepare": DIRECT_GLYPH_PREPARE_GROUP,
    "shared_ui_warp_completed": SHARED_UI_WARP_COMPLETED_GROUP,
    "shared_ui_warp_completed_diagnostic": SHARED_UI_WARP_COMPLETED_DIAGNOSTIC_GROUP,
    "text_game_ui_warp": TEXT_GAME_UI_WARP_GROUP,
    "direct_glyph_empty_gate": DIRECT_GLYPH_EMPTY_GATE_GROUP,
    "direct_glyph_source_audit": DIRECT_GLYPH_SOURCE_AUDIT_GROUP,
}
