// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <algorithm>
#include <cmath>

namespace sunshine_color_test {
  // Binary16 spacing, including its subnormal range. HDR values are absolute
  // scRGB light, so one fixed SDR-sized epsilon is not a valid rounding bound.
  inline float half_spacing(float value) {
    if (value == 0.f) return std::ldexp(1.f, -24);
    int exponent = 0;
    std::frexp(std::abs(value), &exponent);
    return std::ldexp(1.f, std::max(-24, exponent - 11));
  }

  template<class ReadFlat, class ReadFallback>
  float fallback_excess(unsigned width, unsigned height, unsigned color, ReadFlat flat, ReadFallback fallback) {
    float excess = 0;
    for (unsigned eye = 0; eye < 2; ++eye)
      for (unsigned y = height / 8; y < height * 7 / 8; ++y)
        for (unsigned x = width / 8; x < width * 7 / 8; ++x)
          for (unsigned c = 0; c < 3; ++c) {
            const float a = flat(eye * width + x, y, c), b = fallback(eye * width + x, y, c);
            if (!std::isfinite(a) || !std::isfinite(b)) return INFINITY;
            // The ready PQ path has an FP16 decoded source plus final FP16
            // export; the point fallback has only the final quantization.
            const float tolerance = color == 1 ? .003f : 2.f * std::max(half_spacing(a), half_spacing(b)) + 1.e-6f;
            excess = std::max(excess, std::abs(a - b) - tolerance);
          }
    return excess;
  }
}
