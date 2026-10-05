// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

// Value types and constants shared by the camera sample binding, scene gain
// and raw controllers, NOT an admission path from the passive Streamline
// observer. A physical-camera caller must independently prove final-color/
// depth/frame/jitter registration before setting proof_admitted. (The
// physical-camera CPU policy that lived here had no live consumer.)
namespace sunshine_camera_scene {
  // Policy constants, distinct from the host AI scene-camera contract.
  inline constexpr unsigned grid_width = 32, grid_height = 18;
  inline constexpr std::uint64_t target_expiry_ms = 1500;
  inline constexpr std::uint64_t maximum_tick_gap_ms = 250;
  inline constexpr double time_constant_seconds = 0.5;
  inline constexpr float reference_zpd = 0.05f;
  // This reference is scene-relative, NOT absolute/game-independent strength:
  // a foreground object in the center changes q_ref even if perfectly stable.

  // The sole renderer stores bounded source-U in R32. Admit the finite
  // displacement expression, not the discarded reciprocal FP16 intermediates.
  inline bool shader_coordinate_domain(float scale, float largest_q, double zero_q) noexcept {
    const float q0 = static_cast<float>(zero_q);
    if (!std::isfinite(scale) || !(scale > 0.0f) || !std::isfinite(largest_q) ||
        !(largest_q > 0.0f) || !std::isfinite(q0) || q0 < 0.0f) return false;
    const float gain = reference_zpd * scale;
    return std::isfinite(gain) && std::isfinite(gain * q0) &&
      std::isfinite(gain * (q0 - largest_q));
  }

  struct frame_key {
    // Caller-owned monotonically increasing source-frame sequence within the
    // unit epoch, NOT an opaque FrameToken address or wrapping uint32 frame ID.
    std::uint64_t frame{}, token_generation{};
    bool operator==(const frame_key &v) const noexcept {
      return frame == v.frame && token_generation == v.token_generation;
    }
  };
  struct source_key {
    std::uint64_t native{}, lifetime{};
    std::uint32_t viewport{}, width{}, height{}, left{}, top{}, extent_width{}, extent_height{};
    bool operator==(const source_key &v) const noexcept {
      return native == v.native && lifetime == v.lifetime && viewport == v.viewport &&
        width == v.width && height == v.height && left == v.left && top == v.top &&
        extent_width == v.extent_width && extent_height == v.extent_height;
    }
  };
  struct registration {
    std::uint64_t unit_epoch{};
    frame_key frame;
    source_key source;
    // raw = A + B / positive view distance. This is the projection associated
    // with THIS registered frame, not mutable latest-camera state.
    double A{}, B{};
    bool proof_admitted{};
  };
}
