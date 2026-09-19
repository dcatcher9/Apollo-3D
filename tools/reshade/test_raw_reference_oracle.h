// SPDX-License-Identifier: GPL-3.0-only
// Independent numeric oracle over captured native DEPTH samples. No production
// statistics/policy implementation is included, and no value is injected.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace sunshine_raw_oracle {
  inline constexpr unsigned columns = 32, rows = 18, cells = columns * rows;
  struct result {
    std::array<float, cells> oriented {};
    double mean = 0, sigma = 0, center = 0, minimum = 1, maximum = 0;
    double expected_gain = 0;
  };
  inline bool legacy_center() {
    const char *value = std::getenv("SUNSHINE_RAW_REFERENCE_ORACLE");
    if (!value || !*value || !std::strcmp(value, "scene-sigma")) return false;
    if (!std::strcmp(value, "legacy-center")) return true;
    throw std::runtime_error("SUNSHINE_RAW_REFERENCE_ORACLE must be scene-sigma or explicit legacy-center control");
  }
  inline result calculate(const std::array<float, cells> &raw, bool normal) {
    result value;
    for (unsigned i = 0; i < cells; ++i) {
      if (!std::isfinite(raw[i]) || raw[i] < 0 || raw[i] > 1)
        throw std::runtime_error("Native reference oracle received invalid depth");
      // Match only the documented FP32 native-depth orientation. Moments use
      // a separate double two-pass calculation, not the production accumulator.
      const float t = normal ? 1.f - raw[i] : raw[i];
      value.oriented[i] = t;
      value.mean += double(t);
      value.minimum = std::min(value.minimum, double(t));
      value.maximum = std::max(value.maximum, double(t));
      const unsigned x = i % columns, y = i / columns;
      if (x >= 14 && x < 18 && y >= 7 && y < 11) value.center += double(t);
    }
    value.mean /= cells;
    value.center /= 16;
    double variance = 0;
    for (float t : value.oriented) variance += (double(t) - value.mean) * (double(t) - value.mean);
    value.sigma = std::sqrt(variance / cells);
    if (value.sigma <= 0 || value.center <= 0)
      throw std::runtime_error("Native reference oracle has no useful spread/center");
    value.expected_gain = 1. / (legacy_center() ? value.center : value.sigma);
    return value;
  }
}
