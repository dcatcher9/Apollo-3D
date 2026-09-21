// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "game3d_stereo_contract.h"
#include "game3d_ui_plane.h"
#include <windows.h>
#include <reshade_api.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

namespace sunshine_game3d {
  // Byte-for-byte layout of game3d_native.hlsl b0. No ReShade FX uniforms.
  struct render_parameters {
    float strength = 50.f;
    std::int32_t depth_view = 0;
    std::uint32_t depth_ready = 0, camera_ready = 0;
    std::int32_t coordinate_basis = 0;
    float depth_scale = 0.f, strength_blend = 0.f;
    float disparity_limit_uv = default_disparity_limit_uv;
    std::array<float, 2> projection{}, raw_depth_range{0.f, 1.f};
    std::array<float, 2> convergence{}, jitter{};
    std::array<float, 4> depth_rect{0.f, 0.f, 1.f, 1.f};
  };
  static_assert(sizeof(render_parameters) == 80);
  static_assert(offsetof(render_parameters, disparity_limit_uv) == 28);

  // Borrowed only during the current render lease. Diagnostic copies must be
  // recorded before the next render reuses these textures.
  struct diagnostic_resources {
    reshade::api::resource source{}, linear_color{}, candidate{}, vertical_majorant{},
      vertical_field{}, final_field{}, sbs{}, ui_source{}, ui_plane_tiles{}, ui_plane_resolved{};
  };

  // One runtime owns its GPU working set. All passes run on ReShade's graphics
  // queue; borrowed depth is consumed entirely within the owner's read lease.
  class renderer {
  public:
    renderer();
    ~renderer();
    renderer(const renderer &) = delete;
    renderer &operator=(const renderer &) = delete;
    bool configure(reshade::api::effect_runtime *runtime, reshade::api::resource backbuffer,
      reshade::api::color_space color, std::string_view source_override = {});
    static std::string_view shader_source();
    std::string_view active_shader_source() const;
    bool render(reshade::api::command_list *commands, reshade::api::resource backbuffer,
      reshade::api::resource_view depth, const render_parameters &parameters, bool source_alpha_ui = false,
      reshade::api::resource_view alpha_source = {}, const ui_plane_parameters &plane = {});
    // Lazily allocated at the current color extent/format. Copies and reads use
    // the renderer queue; shader_resource is the resting state. Only alpha is
    // consumed, never this retained input's RGB.
    reshade::api::resource ui_source();
    reshade::api::resource_view ui_source_view() const;
    // The caller must first admit a current capture. Repeated presentations of
    // that immutable capture reuse our private texture in renderer queue order.
    // A failed replacement never leaves the old identity marked as uploaded.
    template<class Copy>
    reshade::api::resource_view prepare_ui_source(std::uint64_t capture_id, Copy &&copy) {
      if (!capture_id) return {};
      const auto destination = ui_source();
      if (!destination.handle) return {};
      if (ui_source_capture_ != capture_id) {
        ui_source_capture_ = 0;
        if (!std::forward<Copy>(copy)(destination)) return {};
        ui_source_capture_ = capture_id;
      }
      return ui_source_view();
    }
    reshade::api::resource output() const;
    diagnostic_resources diagnostics() const;
    render_parameters consumed_parameters() const;
    bool consumed_source_alpha_ui() const;
    // Submitted bits. In nearest-UI mode inverse_depth is the floor; only the
    // current-render diagnostic GPU scalar contains the resolved global plane.
    // Front-limit mode ignores this depth word and dispatches no UI reduction.
    ui_plane_parameters consumed_ui_plane() const;
    reshade::api::resource_view native_rtv(reshade::api::resource backbuffer);
    // Also bracket capture's D3D11 unbinds, not only our draw calls. D3D12
    // records to ReShade's dedicated immediate list and needs no app-state swap.
    void begin_frame_state();
    void end_frame_state();
    void finish_present();
    // ReShade's destroy_effect_runtime follows its GPU drain. No extra wait.
    void reset_after_runtime_drain();
  private:
    struct impl;
    std::unique_ptr<impl> data_;
    std::uint64_t ui_source_capture_ = 0;
  };
}
