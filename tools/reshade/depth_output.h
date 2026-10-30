// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Value-only shown-depth identity for one Present. The capture API retains
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

  shown_depth describe(const sunshine_depth::frame_depth &depth);
  bool same_capture(const source_identity &source, const source_identity &previous);
  bool same_source_except_sequence(const source_identity &previous, const source_identity &source);
  std::uint64_t depth_identity(const source_identity &source, bool borrowed_depth, bool depth_ready);
  ui_adaptive::source ui_observation(const shown_depth &shown, const ui_adaptive::source &previous,
    bool scene_ready, std::uint64_t generic_routing_epoch, std::uint64_t raw_basis_epoch, std::uint64_t now);
  sunshine_raw_scene::selected_frame raw_frame(const shown_depth &shown, std::uint64_t basis_epoch);
}
