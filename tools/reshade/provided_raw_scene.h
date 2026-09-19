// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "scene_depth_source.h"
#include "raw_scene_policy.h"

// The API adapter owns a logical depth encoding across rotating GPU resources.
// Feed that identity to the shared reference controller; texture addresses,
// allocation padding and dynamic render size do not define a new encoding.
namespace sunshine_provided_raw {
  inline sunshine_raw_scene::selected_frame selected(const sunshine_scene_depth::frame &value,
      std::uint64_t basis_epoch, bool ready) noexcept {
    sunshine_raw_scene::selected_frame out;
    out.basis_epoch = basis_epoch;
    out.layout_epoch = value.epoch;
    // This is a logical normalized domain, never a native GPU resource handle.
    const auto generation = value.source_id ? value.source_id :
      value.provider == sunshine_scene_depth::provider_kind::streamline ? value.epoch : 0;
    out.source = {static_cast<std::uint64_t>(value.provider) + 1, generation,
      value.viewport, 1, 1, 0, 0, 1, 1};
    out.frame = {value.sequence, value.epoch};
    if (value.projection.supplied || value.projection.direction_supplied)
      out.direction = value.projection.reversed ? sunshine_depth::depth_orientation::reversed :
        sunshine_depth::depth_orientation::normal;
    out.depth_ready = ready;
    out.aligned_viewport_assumed = true; // Shader and readback both use the adapter's explicit active rect.
    out.feedback = value.feedback;
    return out;
  }

  inline sunshine_raw_scene::sample measured(const sunshine_scene_depth::frame &value,
      std::uint64_t id, std::uint64_t basis_epoch,
      const std::array<float, sunshine_raw_scene::grid_width * sunshine_raw_scene::grid_height> &raw) noexcept {
    sunshine_raw_scene::sample out;
    out.id = id;
    out.capture_ms = value.tick;
    out.metadata = selected(value, basis_epoch, true);
    out.readback_frame = out.metadata.frame;
    out.readback_source = out.metadata.source;
    out.readback_layout_epoch = out.metadata.layout_epoch;
    out.raw = raw;
    return out;
  }
}
