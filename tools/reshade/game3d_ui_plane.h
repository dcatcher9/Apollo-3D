// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstdint>
#include <cstring>

namespace sunshine_game3d {
  enum class ui_mask_channel : std::uint32_t { alpha = 0, red = 1 };
  enum class ui_plane_mode : std::uint32_t {
    screen = 0, depth_midpoint = 1, depth_midpoint_nearest_ui = 2, front_limit = 3, shallow_front = 4,
    display_fraction = 5
  };

  // The scene owner resolves this plane independently of the scene's zero.
  // Screen is the compatibility default for callers and historical dumps.
  // In nearest-UI mode inverse_depth is the midpoint floor; the GPU resolves
  // max(floor, nearest decoded depth under current selected UI coverage).
  // Front-limit mode uses the current positive display-disparity budget and
  // ignores inverse_depth, including malformed bits preserved for replay.
  // Shallow-front mode uses one quarter of that same authoritative bound.
  // Display-fraction mode freezes the independently resolved UI fraction in
  // the legacy inverse_depth word; valid fractions range from zero to 0.75.
  struct ui_plane_parameters {
    ui_plane_mode mode = ui_plane_mode::screen;
    float inverse_depth = 0.f;
  };

  // Exact 16-byte b1 layout shared by renderer and diagnostic replay. Preserve
  // submitted bits, including malformed inputs handled by the shader guards.
  inline std::array<std::uint32_t, 4> ui_parameter_words(bool enabled, const ui_plane_parameters &plane,
      ui_mask_channel channel = ui_mask_channel::alpha) {
    std::uint32_t inverse_bits{};
    static_assert(sizeof(inverse_bits) == sizeof(plane.inverse_depth));
    std::memcpy(&inverse_bits, &plane.inverse_depth, sizeof(inverse_bits));
    return {enabled ? 1u : 0u, static_cast<std::uint32_t>(plane.mode), inverse_bits, static_cast<std::uint32_t>(channel)};
  }
}
