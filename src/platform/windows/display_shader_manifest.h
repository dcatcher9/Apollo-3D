/** @file Windows display shaders shared by startup and the build-time resource generator. */
#pragma once

#include "src/generated/host_sbs_shader_manifest.h"
#include <array>

#define WINDOWS_DISPLAY_SHADERS(X) \
  X(convert_yuv420_packed_uv_type0_ps, ps) \
  X(convert_yuv420_packed_uv_type0_ps_linear, ps) \
  X(convert_yuv420_packed_uv_type0_ps_perceptual_quantizer, ps) \
  X(convert_yuv420_packed_uv_type0_vs, vs) \
  X(convert_yuv420_packed_uv_type0s_ps, ps) \
  X(convert_yuv420_packed_uv_type0s_ps_linear, ps) \
  X(convert_yuv420_packed_uv_type0s_ps_perceptual_quantizer, ps) \
  X(convert_yuv420_packed_uv_type0s_vs, vs) \
  X(convert_yuv420_planar_y_ps, ps) \
  X(convert_yuv420_planar_y_ps_linear, ps) \
  X(convert_yuv420_planar_y_ps_perceptual_quantizer, ps) \
  X(convert_yuv420_planar_y_vs, vs) \
  X(convert_yuv420_packed_uv_type0_ps_pq_input_linear, ps) \
  X(convert_yuv420_packed_uv_type0_ps_pq_input_perceptual_quantizer, ps) \
  X(convert_yuv420_packed_uv_type0s_ps_pq_input_linear, ps) \
  X(convert_yuv420_packed_uv_type0s_ps_pq_input_perceptual_quantizer, ps) \
  X(convert_yuv420_planar_y_ps_pq_input_linear, ps) \
  X(convert_yuv420_planar_y_ps_pq_input_perceptual_quantizer, ps) \
  X(rgb_present_pq_input_to_linear_ps, ps) \
  X(rgb_present_pq_input_to_srgb_ps, ps) \
  X(sbs_packed_resample_pq_input_ps, ps) \
  X(pq_input_to_scrgb_ps, ps) \
  X(cursor_ps, ps) \
  X(cursor_ps_normalize_white, ps) \
  X(sbs_packed_resample_ps, ps) \
  X(rgb_present_linear_to_srgb_ps, ps) \
  X(rgb_present_hdr_to_srgb_ps, ps) \
  X(rgb_present_srgb_to_linear_ps, ps) \
  X(cursor_vs, vs) \
  X(p010_uv_tile_cs, cs)

namespace platf::dxgi {
  inline constexpr std::array display_shader_specs {
#define WINDOWS_DISPLAY_SHADER_SPEC(stem, stage) \
    models::host_sbs_shader_cache::shader_spec {#stem ".hlsl", "main_" #stage, #stage "_5_0"},
    WINDOWS_DISPLAY_SHADERS(WINDOWS_DISPLAY_SHADER_SPEC)
#undef WINDOWS_DISPLAY_SHADER_SPEC
  };
}  // namespace platf::dxgi
