// SPDX-License-Identifier: GPL-3.0-only
#include "depth_jitter.h"
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
  void require(bool condition, const char *message) {
    if (!condition) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
  }
  bool near(float a, float b) { return std::abs(a - b) < 1e-7f; }
}

int main() {
  using sunshine_depth_jitter::make;
  sunshine_scene_depth::jitter_offset jitter{.5f, -.25f, 1920, 1080, true};
  auto correction = make(jitter, 1920, 1080, {0, 0, 1920, 1080});
  require(near(correction.offset[0], .5f / 1920) && near(correction.offset[1], -.25f / 1080),
    "NVIDIA positive-right/down jitter sampling sign or render-pixel units changed");
  // Display-resolution depth uses the same UV shift, not half the render shift.
  const auto display = make(jitter, 3840, 2160, {0, 0, 3840, 2160});
  require(display.offset == correction.offset, "Display depth jitter was normalized by output pixels");
  correction = make(jitter, 2048, 1152, {32, 16, 1920, 1080});
  require(near(correction.offset[0], .5f / 2048) && near(correction.offset[1], -.25f / 1152),
    "Padded allocation changed jitter's render domain");
  const sunshine_depth_jitter::correction none;
  const auto inactive = [&](sunshine_scene_depth::jitter_offset value) {
    const auto got = make(value, 1920, 1080, {0, 0, 1920, 1080});
    require(got.offset == none.offset,
      "Unavailable/invalid jitter must be a sampling no-op, not a depth rejection");
  };
  inactive({});
  auto invalid = jitter; invalid.supplied = false; inactive(invalid);
  invalid = jitter; invalid.width = 0; inactive(invalid);
  invalid = jitter; invalid.y = std::numeric_limits<float>::quiet_NaN(); inactive(invalid);
  invalid = jitter; invalid.x = std::numeric_limits<float>::infinity(); inactive(invalid);
  invalid = jitter; invalid.x = 100.f; inactive(invalid);
  require(make(jitter, 2048, 1152, {2040, 0, 1920, 1080}).offset == none.offset,
    "Out-of-allocation crop produced a correction");
  jitter.x = jitter.y = 0;
  require(make(jitter, 1920, 1080, {0, 0, 1920, 1080}).offset == none.offset,
    "Zero jitter changed sampling");
  std::puts("PASS jitter sign, render/display domains, crop bounds, missing/invalid and zero metadata");
}
