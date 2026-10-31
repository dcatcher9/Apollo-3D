// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Per-Present provider entry and value-only shown-depth identity. The capture API retains
// ownership of all resources; this record holds only immutable provenance.
#include "depth_addon.h"
#include "game3d_ui_adaptive.h"
#include "raw_scene_policy.h"
#include <array>
#include <cstdint>
#include <type_traits>

namespace sunshine_game3d::depth_output {
  using source_identity = std::array<std::uint64_t, 4>;

  struct shown_depth {
    // These predicates intentionally differ for a fresh Generic UI observation.
    bool generic{}, generic_ui{}, ready{};
    source_identity scene_source{}; // epoch, source, sequence, viewport
    ui_adaptive::source ui_source;
    sunshine_raw_scene::selected_frame raw_source;
  };
  static_assert(std::is_trivially_destructible_v<shown_depth>);

  // True means the API provider owns this effects pass, including mono.
  // Generic capture switching and the early return stay at the caller.
  bool begin(reshade::api::effect_runtime *runtime, reshade::api::command_list *commands,
    std::uint64_t present, bool allowed);
  bool current(reshade::api::effect_runtime *runtime, sunshine_depth::frame_depth &output);

  shown_depth describe(const sunshine_depth::frame_depth &depth);
  bool same_capture(const source_identity &source, const source_identity &previous);
  bool same_source_except_sequence(const source_identity &previous, const source_identity &source);
  std::uint64_t depth_identity(const source_identity &source, bool borrowed_depth, bool depth_ready);
  ui_adaptive::source ui_observation(const shown_depth &shown, const ui_adaptive::source &previous,
    bool scene_ready, std::uint64_t generic_routing_epoch, std::uint64_t raw_basis_epoch, std::uint64_t now);
  sunshine_raw_scene::selected_frame raw_frame(const shown_depth &shown, std::uint64_t basis_epoch);
}

// Binding/readiness writers keep their private caches and definitions in place.
// bind_api is internal to provider depth_display.inl; bind_generic is static
// in depth_addon.cpp. The rebind_api_on_reload site keeps the existing external
// name provider::reload_effect_bindings. The other external writer names are
// provider::set_depth_ready and the FX path's depth_input::set_ready.
namespace sunshine_streamline::provider {
  void reload_effect_bindings(reshade::api::effect_runtime *runtime);
  void set_depth_ready(reshade::api::effect_runtime *runtime, bool ready);
}
namespace sunshine_game3d::depth_input {
  void set_ready(reshade::api::effect_runtime *runtime, bool ready);
}
