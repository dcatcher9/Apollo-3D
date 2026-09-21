// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>
#include <algorithm>
#include <cmath>

namespace sunshine_depth_statistics {
  inline constexpr std::uint32_t target_tiles = 576, maximum_tiles = 768;
  struct tile_layout { std::uint32_t x = 0, y = 0; };
  // Follow the active crop's aspect with approximately square tiles. Resolution
  // alone does not change the layout, except when tiny crops cap either axis.
  inline tile_layout tile_grid(std::uint32_t width, std::uint32_t height) noexcept {
    if (!width || !height) return {};
    const double aspect = double(width) / height;
    tile_layout out{
      std::clamp(std::uint32_t(std::sqrt(target_tiles * aspect) + .5), 1u, std::min(width, target_tiles)),
      std::clamp(std::uint32_t(std::sqrt(target_tiles / aspect) + .5), 1u, std::min(height, target_tiles))};
    while (std::uint64_t(out.x) * out.y > maximum_tiles) {
      if (out.x >= out.y) --out.x; else --out.y;
    }
    return out;
  }
  struct tile_interval {
    std::uint32_t first = 0, last = 0;
    std::uint32_t size() const noexcept { return last - first; }
    std::uint32_t center() const noexcept { return first + size() / 2; }
  };
  // Mirrors the shader's disjoint integer partition; callers must use the
  // frozen layout of the capture rather than recomputing it from a newer view.
  inline tile_interval tile_bounds(std::uint32_t index, std::uint32_t extent,
      std::uint32_t tiles) noexcept {
    if (!tiles || index >= tiles) return {};
    return {std::uint32_t(std::uint64_t(index) * extent / tiles),
      std::uint32_t((std::uint64_t(index) + 1) * extent / tiles)};
  }
  // Full active-rectangle moments of q = (raw - A) * inverseB, including zero
  // depth. The immutable coefficients belong to this capture, not its reader.
  // A supplied but invalid measurement must not become a sparse-grid fallback.
  struct moments {
    bool supplied = false, valid = false;
    float A = 0.f, inverseB = 1.f;
    std::uint64_t count = 0;
    double sum = 0., sum_squares = 0.;
    std::uint32_t tiles_x = 0, tiles_y = 0;
    // Production reduces (q - tile_min) directly and merges origins in double.
    // These are not reconstructed by subtracting nearly equal raw moments.
    // Supplied remains true on invalid results, so malformed centered evidence
    // cannot fall back to the cancellation-prone uncentered diagnostic sums.
    bool centered_supplied = false;
    double center = 0., sum_centered = 0., sum_centered_squares = 0.;
  };
}
