// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "sbs_cursor.h"
#include "src/reshade_bridge_protocol.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <d3d11.h>
#include <optional>

// The draws that resample a packed Game 3D export into the packed output raster, shared by the
// converter and its WARP tests.
namespace platf::packed_output {
  // One fullscreen-triangle draw of sbs_packed_resample_ps: `viewport` maps the whole packed
  // source over texture coordinates 0..1, and the shader clamps each output pixel to its own
  // source half. A pass that covers one eye only is limited to that eye by `scissor`.
  struct pass_t {
    D3D11_VIEWPORT viewport {};
    std::optional<D3D11_RECT> scissor;
  };

  struct passes_t {
    std::array<pass_t, 2> passes {};
    std::uint32_t count = 0;
    // Fill the output with black (zero in every transfer the intermediate holds: sRGB code values,
    // linear scRGB, and PQ decoded to scRGB) before the passes.
    bool clear = false;
  };

  // `placement` is ::reshade_bridge::place_output() for the same packed source and this
  // width x height output. `scaled` is one pass over the whole output. `fitted` is one pass per
  // eye: its viewport spans twice the fitted eye width, placed so that the eye's own source half
  // maps exactly onto the eye's fitted rectangle, and its scissor is that rectangle; the rest of
  // the output stays black. `exact` needs no resample and `refused` has no output: both are empty.
  [[nodiscard]] inline passes_t resample_passes(const ::reshade_bridge::output_placement_t &placement, std::uint32_t width, std::uint32_t height) {
    passes_t result;
    if (placement.fit == ::reshade_bridge::output_fit::scaled) {
      result.passes[0].viewport = {0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f};
      result.count = 1;
      return result;
    }
    const auto &eye = placement.eye;
    const std::uint32_t eye_width = width / 2;
    if (placement.fit != ::reshade_bridge::output_fit::fitted || !eye.width || !eye.height || eye.x + eye.width > eye_width || eye.y + eye.height > height) {
      return result;
    }
    for (std::uint32_t side = 0; side < 2; ++side) {
      const auto left = side * eye_width + eye.x;
      auto &pass = result.passes[side];
      // Left eye: its fitted rectangle is the first half of the viewport (coordinates below
      // 0.5). Right eye: the second half (coordinates above 0.5).
      pass.viewport = {float(left) - float(side * eye.width), float(eye.y), float(2 * eye.width), float(eye.height), 0.0f, 1.0f};
      pass.scissor = D3D11_RECT {LONG(left), LONG(eye.y), LONG(left + eye.width), LONG(eye.y + eye.height)};
    }
    result.count = 2;
    result.clear = true;
    return result;
  }

  // The smallest output-pixels-per-source-texel ratio of the passes (for sbs_cursor::patch_margin).
  [[nodiscard]] inline float minimum_scale(const passes_t &passes, std::uint32_t source_width, std::uint32_t source_height) {
    float scale = 0.0f;
    for (std::uint32_t i = 0; i < passes.count && source_width && source_height; ++i) {
      const auto &viewport = passes.passes[i].viewport;
      const float pass_scale = std::min(viewport.Width / float(source_width), viewport.Height / float(source_height));
      scale = i == 0 ? pass_scale : std::min(scale, pass_scale);
    }
    return scale;
  }

  // Output pixels of `pass` whose footprint may include a texel of `changed`: the pass's
  // sbs_cursor::output_region() clamped to its scissor. Re-running the pass from a cursor patch
  // over these pixels gives the same output as converting the composited frame.
  [[nodiscard]] inline std::optional<D3D11_RECT> pass_region(const pass_t &pass, const D3D11_RECT &changed, std::uint32_t source_width, std::uint32_t source_height) {
    auto region = sbs_cursor::output_region(changed, source_width, source_height, pass.viewport);
    if (region && pass.scissor) {
      region->left = std::max(region->left, pass.scissor->left);
      region->top = std::max(region->top, pass.scissor->top);
      region->right = std::min(region->right, pass.scissor->right);
      region->bottom = std::min(region->bottom, pass.scissor->bottom);
      if (region->left >= region->right || region->top >= region->bottom) {
        return std::nullopt;
      }
    }
    return region;
  }
}  // namespace platf::packed_output
