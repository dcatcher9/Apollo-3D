// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "camera_scene_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sunshine_projection_depth {
  // Camera coefficients reconstruct inverse game distance. Artistic scale is
  // selected separately by the shared scene-gain controller, never by the
  // clipping near plane or a per-game multiplier.
  inline constexpr float reference_zpd = sunshine_camera_scene::reference_zpd;

  enum class status { invalid_projection, unsupported_shader_domain, ready };
  struct coefficients {
    status reason{status::invalid_projection};
    float shader_A{}, inverseB{};
    float raw_min{}, raw_max{1.f};
    double near_plane{};
    bool valid() const noexcept { return reason == status::ready; }
  };

  // The caller supplies an already validated perspective encoding d=A+B/z
  // associated with its actual depth source. A/B alone cannot establish that
  // registration or distinguish a mislabeled orthographic/linear-depth input.
  // Optional resource precision converts stored raw to projection depth as
  // d=raw*raw_scale+raw_bias. Storage conventions cannot change reconstructed q.
  inline coefficients make(double A, double B, double raw_scale = 1.0, double raw_bias = 0.0) noexcept {
    coefficients out;
    if (!std::isfinite(A) || !std::isfinite(B) || B == 0.0 ||
        !std::isfinite(raw_scale) || !std::isfinite(raw_bias) || raw_scale == 0.0) return out;
    const double endpoint0 = (0.0 - A) / B, endpoint1 = (1.0 - A) / B;
    const double largest = std::max(endpoint0, endpoint1);
    if (!std::isfinite(endpoint0) || !std::isfinite(endpoint1) || endpoint0 < 0.0 || endpoint1 < 0.0 ||
        !(largest > 0.0)) return out;
    const double near_plane = 1.0 / largest;
    const double transformed_A = (A-raw_bias)/raw_scale;
    const float shader_A = static_cast<float>(transformed_A);
    const float inverseB = static_cast<float>(raw_scale/B);
    const double raw0 = -raw_bias/raw_scale, raw1 = (1.0-raw_bias)/raw_scale;
    const float raw_min = static_cast<float>(std::min(raw0, raw1));
    const float raw_max = static_cast<float>(std::max(raw0, raw1));
    out.reason = status::unsupported_shader_domain;
    if (!std::isfinite(near_plane) || !(near_plane > 0.0) ||
        !std::isfinite(shader_A) || (transformed_A != 0.0 && !std::isnormal(shader_A)) ||
        !std::isfinite(raw_min) || !std::isfinite(raw_max) || !(raw_min < raw_max) ||
        (raw_min != 0.f && !std::isnormal(raw_min)) || (raw_max != 0.f && !std::isnormal(raw_max)) ||
        !std::isnormal(inverseB)) return out;

    // Use the actual FP32 affine coefficients sent to SunshineCamera.fxh.
    // D3D arithmetic can flush subnormal FP32 operands to zero. Such a
    // coefficient must not pass only on the CPU and change meaning on the GPU.
    // The geometry owner verifies the decoded domain. The controller separately
    // checks its proposed gain against the whole domain before rendering.
    const float q0 = (raw_min - shader_A) * inverseB, q1 = (raw_max - shader_A) * inverseB;
    const float largest_q = std::max(q0, q1);
    if (!std::isfinite(q0) || !std::isfinite(q1) || q0 < 0.0f || q1 < 0.0f || !std::isnormal(largest_q) ||
        !(largest_q > 0.0f)) return out;
    out.reason = status::ready;
    out.shader_A = shader_A;
    out.inverseB = inverseB;
    out.near_plane = near_plane;
    out.raw_min = raw_min;
    out.raw_max = raw_max;
    return out;
  }

  // Mathematical conversion only. Exact endpoints can be real near/far depth
  // or an unwritten clear; the capture/controller owns that evidence decision.
  // A completed asynchronous sample uses its frozen coefficients, not a newer
  // camera fetched when readback completes. q and stored q0 stay in inverse
  // game-distance units across ordinary near/far clipping changes.
  inline bool inverse_distance(const coefficients &projection, float raw, float &q) noexcept {
    q = 0.0f;
    if (!projection.valid() || !std::isfinite(raw) || raw < projection.raw_min || raw > projection.raw_max) return false;
    const float converted = (raw - projection.shader_A) * projection.inverseB;
    if (!std::isfinite(converted) || converted < 0.0f) return false;
    q = converted;
    return true;
  }

  // The displayed scene gain must fit THIS frame's exact encoding before use.
  // A large clip change may temporarily require mono; never clamp raw pixels
  // or substitute old projection coefficients to make the transition fit.
  inline bool supports_scale(const coefficients &projection, float K) noexcept {
    if (!projection.valid() || !std::isnormal(K) || !(K > 0.f)) return false;
    const float q0 = (projection.raw_min - projection.shader_A) * projection.inverseB;
    const float q1 = (projection.raw_max - projection.shader_A) * projection.inverseB;
    return sunshine_camera_scene::shader_coordinate_domain(K, std::max(q0, q1), 0.0);
  }

  // The zero plane is an independent comfort target. It need not lie within
  // the current clip interval, but its FP32 displacement must remain finite.
  inline bool supports_zero(const coefficients &projection, float K, double q0) noexcept {
    if (!projection.valid() || !std::isnormal(K) || !(K > 0.f) ||
        !std::isfinite(q0) || q0 < 0.0 || q0 > std::numeric_limits<float>::max()) return false;
    const float stored = static_cast<float>(q0);
    if (q0 > 0.0 && !std::isnormal(stored)) return false;
    const float near_q = (projection.raw_min - projection.shader_A) * projection.inverseB;
    const float far_q = (projection.raw_max - projection.shader_A) * projection.inverseB;
    return sunshine_camera_scene::shader_coordinate_domain(K, std::max(near_q, far_q), stored);
  }
}
