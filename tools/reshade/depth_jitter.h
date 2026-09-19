// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cmath>
#include "scene_depth_source.h"

namespace sunshine_depth_jitter {
  struct correction {
    std::array<float, 2> offset{};
  };

  // Jitter moves a rendered feature right/down in NVIDIA pixel coordinates.
  // Sample that position in the jittered depth for the resolved color pixel.
  // Convert via its render domain, then the active crop, never output size.
  inline correction make(const sunshine_scene_depth::jitter_offset &jitter,
      std::uint32_t width, std::uint32_t height, sunshine_scene_depth::extent area) {
    if (!jitter.supplied || !jitter.width || !jitter.height ||
        !std::isfinite(jitter.x) || !std::isfinite(jitter.y) ||
        std::abs(jitter.x) > .5f || std::abs(jitter.y) > .5f ||
        !width || !height || !area.width || !area.height ||
        area.left >= width || area.top >= height ||
        area.width > width - area.left || area.height > height - area.top)
      return {};
    return {{jitter.x / jitter.width * (float(area.width) / width),
             jitter.y / jitter.height * (float(area.height) / height)}};
  }
}
