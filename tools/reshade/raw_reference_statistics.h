// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cmath>
#include <limits>

// Stateless reference statistics. Source/frame association, clock, calibration,
// target readiness and frozen gain belong exclusively to camera_scene_policy.
namespace sunshine_raw_reference {
  inline constexpr unsigned width = 32, height = 18, cell_count = width * height;
  using grid = std::array<float, cell_count>;
  // Shared experimental artistic target for sigma(H*t), not recovered camera
  // scale, measured binocular disparity, or a shipping quality calibration.
  inline constexpr double experimental_artistic_sigma_target = 1.0;
  // Admission criterion only: reject variance dominated by fewer than sixteen
  // effective cells. No input value is trimmed, rescaled or clipped.
  inline constexpr double minimum_effective_contributors = 16.0;
  struct statistics {
    bool valid = false;
    unsigned endpoint_cells = 0;
    double mean = 0.0, variance = 0.0, sigma = 0.0, effective_contributors = 0.0;
    // Diagnostics only. Neither value owns the screen-plane target.
    double center_mean = 0.0;
    double scene_mean_gain = 0.0, sigma_gain = 0.0;
  };

  // All equally spaced cells contribute, including exact 0 and 1. The input is
  // optionally oriented by the shader's exact FP32 1-raw expression. No
  // clipping, percentile selection, epsilon floor, log transform or fitted
  // input remapping is applied. The original grid remains immutable.
  inline statistics measure(const grid &depth, double artistic_sigma_target = experimental_artistic_sigma_target,
      bool flip_raw = false) noexcept {
    statistics result;
    if (!std::isfinite(artistic_sigma_target) || artistic_sigma_target <= 0.0) return result;
    double m2 = 0.0;
    for (unsigned i = 0; i < cell_count; ++i) {
      const float raw = depth[i];
      if (!std::isfinite(raw) || raw < 0.0f || raw > 1.0f) return result;
      const double t = flip_raw ? 1.0f - raw : raw;
      result.endpoint_cells += t == 0.0 || t == 1.0;
      const double delta = t - result.mean;
      result.mean += delta / (i + 1);
      m2 += delta * (t - result.mean);
      const unsigned x = i % width, y = i / width;
      if (x >= 14 && x < 18 && y >= 7 && y < 11) result.center_mean += t / 16.0;
    }
    // Values and work count are bounded, and accumulation stays in double.
    // A numerical failure invalidates the reference instead of modifying it.
    if (!std::isfinite(m2) || m2 < 0.0) return result;
    double m4 = 0.0;
    for (float raw : depth) {
      const double t = flip_raw ? 1.0f - raw : raw;
      const double residual = t - result.mean;
      const double square = residual * residual;
      m4 += square * square;
    }
    result.variance = m2 / cell_count;
    result.sigma = std::sqrt(result.variance);
    if (m4 > 0.0) result.effective_contributors = m2 * m2 / m4;
    result.scene_mean_gain = result.mean > 0.0 ? artistic_sigma_target / result.mean :
      std::numeric_limits<double>::infinity();
    result.sigma_gain = result.sigma > 0.0 ? artistic_sigma_target / result.sigma :
      std::numeric_limits<double>::infinity();
    result.valid = std::isfinite(m4) && std::isfinite(result.effective_contributors);
    return result;
  }

  inline bool usable_reference(const statistics &value) noexcept {
    return value.valid && value.endpoint_cells < cell_count && value.sigma > 0.0 &&
      value.effective_contributors >= minimum_effective_contributors;
  }
}
