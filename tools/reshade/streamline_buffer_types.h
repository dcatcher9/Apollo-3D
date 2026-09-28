// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>

// Canonical Streamline buffer-type numbers consumed by Game 3D, as declared by
// the 2.12.0+ public headers. Raw SDK values are translated to these once, at
// the hook boundary, by streamline_buffer_contract.h; consumers compare these.
namespace sunshine_streamline::buffers {
  inline constexpr std::uint32_t depth = 0, hudless_color = 2, scaling_input_color = 3,
    scaling_output_color = 4, ui_color_and_alpha = 23, high_resolution_depth = 48, linear_depth = 49,
    backbuffer = 53, responsivity_mask = 68, ui_alpha = 69;
  inline constexpr std::uint32_t unknown = 0xffffffffu;

  constexpr bool is_depth(std::uint32_t type) {
    return type == depth || type == high_resolution_depth || type == linear_depth;
  }
}
