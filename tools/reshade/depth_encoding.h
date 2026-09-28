// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cmath>

namespace sunshine_scene_depth {
  enum class depth_encoding { device, linear_distance };

  // The affine pair decodes inverse distance for device depth, or positive
  // view distance for linear depth. No metric conversion or near-plane scale.
  inline bool decode_inverse_distance(float raw, float A, float inverseB,
      depth_encoding encoding, float &q) noexcept {
    q = 0.f;
    if (!std::isfinite(raw)) return false;
    const float affine = (raw - A) * inverseB;
    if (!std::isfinite(affine)) return false;
    if (encoding == depth_encoding::linear_distance) {
      // D3D flushes denormals; admit only reciprocals reproducible on the GPU.
      if (!(affine > 0.f) || !std::isnormal(affine)) return false;
      const float inverse = 1.f / affine;
      if (!(inverse > 0.f) || !std::isnormal(inverse)) return false;
      q = inverse;
      return true;
    }
    if (encoding != depth_encoding::device || affine < 0.f) return false;
    q = affine;
    return true;
  }
}
