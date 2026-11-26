// SPDX-License-Identifier: GPL-3.0-only
// The exported source color's linear value, as the HDR mono checks of the
// runtime fixtures expect it (sRGB, scRGB or Rec.2020 PQ input).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace sunshine_depth3d_color {
  inline float srgb_linear(float encoded) {
    return encoded <= .04045f ? encoded / 12.92f : std::pow((encoded + .055f) / 1.055f, 2.4f);
  }

  // ST 2084 absolute-light decoding, followed by a linear Rec.2020 -> Rec.709
  // basis change. Values are scRGB units (80 cd/m2), not display tone mapping.
  inline double pq_linear(double code) {
    const double p = std::pow(code, 32.0 / 2523.0);
    return 125.0 * std::pow(std::max(p - 3424.0 / 4096.0, 0.0) / (2413.0 / 128.0 - (2392.0 / 128.0) * p), 16384.0 / 2610.0);
  }

  using rgb = std::array<double, 3>;

  inline rgb decoded(rgb value, unsigned color) {
    if (color == 1) {
      for (auto &v : value) {
        v = srgb_linear(float(v));
      }
    } else if (color == 3) {
      for (auto &v : value) {
        v = pq_linear(v);
      }
      const auto v = value;
      value = {1.660491002 * v[0] - .587641139 * v[1] - .072849863 * v[2], -.124550475 * v[0] + 1.132899897 * v[1] - .008349423 * v[2], -.018150763 * v[0] - .100578898 * v[1] + 1.118729661 * v[2]};
    }
    return value;
  }
}
