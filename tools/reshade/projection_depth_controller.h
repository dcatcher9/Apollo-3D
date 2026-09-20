// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "camera_scene_policy.h"
#include "projection_depth_scale.h"
#include "scene_gain.h"

namespace sunshine_projection_depth {
  namespace timing = sunshine_camera_scene;
  inline constexpr unsigned grid_width = timing::grid_width, grid_height = timing::grid_height;
  struct domain {
    std::uint64_t epoch{};
    std::uint32_t viewport{};
    bool operator==(const domain &other) const noexcept { return epoch == other.epoch && viewport == other.viewport; }
  };
  struct sample {
    std::uint64_t id{}, capture_ms{};
    domain logical_domain;
    coefficients projection;
    // Legacy point-only fixture input. Live captures carry full-image moments
    // decoded with this exact projection; extrema constrain safety separately.
    std::array<float, grid_width * grid_height> raw{};
    bool range_supplied{}, range_valid{};
    float range_min{}, range_max{};
    sunshine_depth_statistics::moments moments;
    sunshine_scene_feedback::sample feedback;
  };
  enum class center_status {
    waiting_for_center, ready, holding, invalid_domain, invalid_projection,
    stale_sample, invalid_depth, unsupported_shader_domain, clock_went_backwards
  };
  inline const char *name(center_status value) noexcept {
    switch (value) {
      case center_status::waiting_for_center: return "waiting_for_depth_range";
      case center_status::ready: return "ready";
      case center_status::holding: return "holding";
      case center_status::invalid_domain: return "invalid_domain";
      case center_status::invalid_projection: return "invalid_projection";
      case center_status::stale_sample: return "stale_sample";
      case center_status::invalid_depth: return "invalid_depth";
      case center_status::unsupported_shader_domain: return "unsupported_shader_domain";
      case center_status::clock_went_backwards: return "clock_went_backwards";
    }
    return "unknown";
  }
  struct center_output {
    bool initialized{}, ready{};
    center_status reason{center_status::waiting_for_center};
    float q0{}, K{};
    double target_K{};
    unsigned calibration_samples{};
    sunshine_scene_gain::refinement learning{};
    bool limited{};
    sunshine_scene_gain::depth_range depth_statistics{};
    bool has_depth_statistics{};
    double target_q0{};
  };

  // One zero-plane state per authenticated logical viewport/unit domain, not
  // per rotating depth resource. The caller proves sample/source association
  // and current capture readiness independently. Camera data reconstructs q;
  // one shared policy tracks nearest-depth gain and the observed range midpoint. Current A/B reconstruct
  // current pixels exactly, without interpolating camera coefficients.
  class controller {
  public:
    void configure(sunshine_scene_gain::limits budget) noexcept { gain_.configure(budget); }
    void reset(domain logical_domain, std::uint64_t now_ms) noexcept {
      *this = {};
      domain_ = logical_domain;
      wall_ms_ = capture_floor_ms_ = now_ms;
      have_wall_ = true;
    }
    bool reset_reference(domain logical_domain, std::uint64_t now_ms) noexcept {
      if (!domain_.epoch || !(logical_domain == domain_) || !advance(now_ms)) return false;
      capture_floor_ms_ = now_ms;
      strict_floor_ = true;
      // This explicit action replaces both reference and zero, while retaining
      // watermarks and feedback revision so delayed old work cannot reseed it.
      gain_.reset_reference();
      return true;
    }
    void suspend() noexcept { gain_.suspend(); }
    void synchronize_feedback(const sunshine_scene_feedback::sample &feedback) noexcept {
      gain_.synchronize(feedback);
    }

    center_status observe(const sample &value, std::uint64_t now_ms) noexcept {
      if (!domain_.epoch || !(value.logical_domain == domain_)) return center_status::invalid_domain;
      if (!advance(now_ms)) return center_status::clock_went_backwards;
      if (!value.id || !gain_.accepts(value.feedback) || value.capture_ms > now_ms || value.capture_ms < capture_floor_ms_ ||
          (strict_floor_ && value.capture_ms == capture_floor_ms_) || now_ms-value.capture_ms >= timing::target_expiry_ms ||
          (have_sample_ && (value.id <= last_id_ || value.capture_ms <= last_capture_ms_))) return center_status::stale_sample;
      if (!value.projection.valid()) return center_status::invalid_projection;
      have_sample_ = true;
      last_id_ = value.id;
      last_capture_ms_ = value.capture_ms;
      if (value.feedback.reset) {
        gain_.invalidate();
        return gain_.initialized() ? center_status::holding : center_status::waiting_for_center;
      }
      const auto range = sunshine_scene_gain::decode_range(value.raw, value.range_supplied,
        value.range_valid, value.range_min, value.range_max, value.projection.shader_A,
        value.projection.inverseB, value.projection.raw_min, value.projection.raw_max, value.moments);
      if (!range.valid()) {
        gain_.invalidate();
        return center_status::invalid_depth;
      }
      const auto gain_status = gain_.observe(range, value.capture_ms, value.feedback);
      if (gain_status == sunshine_scene_gain::status::unsupported) {
        return center_status::unsupported_shader_domain;
      }
      if (!gain_.initialized()) return center_status::waiting_for_center;
      return center_status::ready;
    }

    center_output evaluate(domain logical_domain, const coefficients &current, std::uint64_t now_ms) noexcept {
      if (!domain_.epoch || !(logical_domain == domain_)) { suspend(); return output(center_status::invalid_domain); }
      if (!advance(now_ms)) return output(center_status::clock_went_backwards);
      if (!current.valid()) { suspend(); return output(center_status::invalid_projection); }
      if (!gain_.initialized()) return output(center_status::waiting_for_center);
      // Progress the numerical plane before validating the current shader pair.
      // An unsupported old pair must not prevent convergence into a valid one.
      gain_.update(now_ms);
      const float K = gain_.value();
      if (!supports_scale(current, K) || !supports_zero(current, K, gain_.zero())) {
        return output(center_status::unsupported_shader_domain);
      }
      return output(gain_.has_target() ? center_status::ready : center_status::holding);
    }

  private:
    bool advance(std::uint64_t now_ms) noexcept {
      if (have_wall_ && now_ms < wall_ms_) { gain_.invalidate(); return false; }
      wall_ms_ = now_ms;
      have_wall_ = true;
      gain_.advance(now_ms);
      return true;
    }
    center_output output(center_status reason) const noexcept {
      center_output out{gain_.initialized(), reason == center_status::ready || reason == center_status::holding,
        reason, gain_.zero(), gain_.value(), gain_.target(), gain_.samples(), gain_.learning(), gain_.limited()};
      out.has_depth_statistics = gain_.depth_statistics(out.depth_statistics);
      out.target_q0 = gain_.target_zero();
      return out;
    }
    domain domain_;
    std::uint64_t wall_ms_{}, capture_floor_ms_{}, last_id_{}, last_capture_ms_{};
    sunshine_scene_gain::policy gain_;
    bool have_wall_{}, strict_floor_{}, have_sample_{};
  };
}
