// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstdint>
#include <cstring>

namespace sunshine_game3d {
  enum class ui_mask_channel : std::uint32_t { alpha = 0, red = 1 };
  // Values 1-4 named the midpoint, nearest-UI, front-limit and shallow-front
  // planes, removed on 11-25; replay refuses a package that recorded one, and
  // the values are never reused.
  enum class ui_plane_mode : std::uint32_t { screen = 0, display_fraction = 5 };

  // The scene owner resolves this plane independently of the scene's zero.
  // Screen is the default for callers and pins UI at zero disparity.
  // Display-fraction mode freezes the independently resolved UI fraction in
  // the inverse_depth word; valid fractions range from zero to 0.75.
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
