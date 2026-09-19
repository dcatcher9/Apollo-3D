// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "camera_scene_policy.h"
#include "scene_feedback.h"

// One shared screen plane and smoother for decoded inverse distance and the
// uncalibrated oriented-raw fallback. K/H are compatibility shader fields,
// derived from the same published float zero: gain*zero == 1 (rounding aside).
// There is no independent reference distance or persistent scene-strength gain.
namespace sunshine_scene_gain {
  inline constexpr unsigned reference_samples = 4;
  inline constexpr std::uint64_t reference_span_ms = 750, reference_expiry_ms = 1500;
  enum class refinement { holding, adapting, history_hold };
  inline const char *name(refinement value) noexcept {
    switch (value) {
      case refinement::holding: return "holding_screen_plane";
      case refinement::adapting: return "tracking_screen_plane";
      case refinement::history_hold: return "history_hold";
    }
    return "unknown";
  }
  enum class status { calibrating, ready, unsupported };

  class policy {
  public:
    void reset() noexcept { *this = {}; }
    void reset_reference() noexcept {
      const auto revision = revision_;
      *this = {};
      revision_ = revision;
    }
    void suspend() noexcept { motion_armed_ = false; }
    void invalidate() noexcept {
      suspend();
      have_target_ = false;
      refinement_ = refinement::holding;
      if (!initialized_) count_ = 0;
    }
    void advance(std::uint64_t now_ms) noexcept {
      if (have_wall_ && now_ms < wall_ms_) { invalidate(); return; }
      if (have_wall_ && now_ms-wall_ms_ > sunshine_camera_scene::maximum_tick_gap_ms) suspend();
      wall_ms_ = now_ms;
      have_wall_ = true;
      if (!initialized_ && count_ && now_ms >= first_ms_ && now_ms-first_ms_ >= reference_expiry_ms) count_ = 0;
      if (have_target_ && now_ms >= target_ms_ && now_ms-target_ms_ >= sunshine_camera_scene::target_expiry_ms) invalidate();
    }
    bool synchronize(const sunshine_scene_feedback::sample &feedback) noexcept {
      const bool changed = feedback.revision > revision_;
      if (changed) {
        revision_ = feedback.revision;
        invalidate();
        refinement_ = refinement::history_hold;
      }
      return changed;
    }
    bool accepts(const sunshine_scene_feedback::sample &feedback) const noexcept { return feedback.revision == revision_; }

    // Admission owners supply only fresh authenticated captures. Shader bounds
    // are checked by the renderer after update(), never against a future target:
    // valid numerical evidence must be able to recover an unsupported encoding.
    status observe(double center, std::uint64_t capture_ms,
        const sunshine_scene_feedback::sample &feedback = {}) noexcept {
      if (!accepts(feedback)) return initialized_ ? status::ready : status::calibrating;
      if (feedback.reset) { invalidate(); refinement_ = refinement::history_hold; return initialized_ ? status::ready : status::calibrating; }
      if (!std::isfinite(center) || !(center > 0.0)) { invalidate(); return status::unsupported; }
      if (!initialized_) {
        constexpr auto spacing = reference_span_ms/(reference_samples-1);
        if (count_ && (capture_ms <= reference_ms_ || capture_ms-reference_ms_ < spacing)) return status::calibrating;
        if (!count_) { first_ms_ = capture_ms; mean_ = 0.0; }
        reference_ms_ = capture_ms;
        mean_ += (center-mean_)/++count_;
        if (count_ < reference_samples) return status::calibrating;
        const float initial = static_cast<float>(mean_);
        if (!std::isnormal(initial) || !std::isnormal(1.f/initial)) { count_ = 0; return status::unsupported; }
        zero_ = mean_;
        initialized_ = true;
      }
      if (!have_target_) suspend();
      target_ = center;
      target_ms_ = capture_ms;
      have_target_ = true;
      return status::ready;
    }
    void update(std::uint64_t now_ms) noexcept {
      advance(now_ms);
      if (!initialized_ || !have_target_) return;
      if (motion_armed_ && now_ms >= last_tick_ms_ && now_ms-last_tick_ms_ <= sunshine_camera_scene::maximum_tick_gap_ms) {
        const double seconds = (now_ms-last_tick_ms_)*.001;
        const double delta = (target_-zero_)*-std::expm1(-seconds/sunshine_camera_scene::time_constant_seconds);
        // At most two current-zero units per second. With dt<=.25, the
        // downward step cannot cross zero; the positive exponential target
        // also prevents overshoot. Multiplying units scales the whole step.
        const double bound = sunshine_camera_scene::maximum_normalized_speed*seconds*zero_;
        zero_ += std::clamp(delta,-bound,bound);
      }
      last_tick_ms_ = now_ms;
      motion_armed_ = true;
      refinement_ = std::abs(target_-zero_) > zero_*1e-6 ? refinement::adapting : refinement::holding;
    }
    bool initialized() const noexcept { return initialized_; }
    bool has_target() const noexcept { return have_target_; }
    float zero() const noexcept { return initialized_ ? static_cast<float>(zero_) : 0.f; }
    float value() const noexcept { return initialized_ ? 1.f/zero() : 0.f; }
    double reference() const noexcept { return zero(); } // Compatibility diagnostic; current screen plane.
    double target() const noexcept { return have_target_ ? 1.f/static_cast<float>(target_) : 0.0; }
    unsigned samples() const noexcept { return count_; }
    refinement learning() const noexcept { return refinement_; }

  private:
    std::uint64_t first_ms_{}, reference_ms_{}, revision_{}, target_ms_{}, wall_ms_{}, last_tick_ms_{};
    double zero_{}, mean_{}, target_{};
    unsigned count_{};
    bool initialized_{}, have_target_{}, have_wall_{}, motion_armed_{};
    refinement refinement_{refinement::holding};
  };
}
