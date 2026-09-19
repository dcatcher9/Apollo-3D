// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

// Physical-camera CPU policy, NOT an admission path from the passive
// Streamline observer. The raw controller shares value types and constants,
// but owns its state separately. A physical-camera caller must independently prove
// final-color/depth/frame/jitter registration before setting proof_admitted;
// that physical-camera path has no proven live consumer yet.
namespace sunshine_camera_scene {
  // Prototype policy constants, distinct from the host AI scene-camera contract.
  inline constexpr unsigned grid_width = 32, grid_height = 18;
  inline constexpr unsigned reference_samples = 4;
  inline constexpr std::uint64_t minimum_span_ms = 750;
  inline constexpr std::uint64_t calibration_timeout_ms = 5000;
  inline constexpr std::uint64_t target_expiry_ms = 1500;
  inline constexpr std::uint64_t maximum_tick_gap_ms = 250;
  inline constexpr double maximum_reference_cv = 0.05;
  inline constexpr double time_constant_seconds = 0.5;
  inline constexpr double maximum_normalized_speed = 2.0;
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
  struct sample {
    std::uint64_t id{}, capture_ms{};
    registration metadata;
    frame_key readback_frame;
    source_key readback_source;
    // Point-sampled immutable grid over the registered source extent. The
    // center 4x4 supplies both the startup reference and subsequent target.
    std::array<float, grid_width * grid_height> raw{};
  };
  enum class status {
    uninitialized, calibrating, reference_unstable, calibrated, ready,
    unit_epoch_mismatch, proof_missing, source_mismatch, frame_mismatch,
    invalid_source, invalid_projection, invalid_depth, clear_depth,
    duplicate_or_backward_sample, sample_in_future, stale_sample,
    clock_went_backwards, calibration_timeout, no_target, target_expired,
    unsupported_shader_domain
  };
  inline const char *name(status value) noexcept {
    switch (value) {
#define SUNSHINE_SCENE_STATUS(x) case status::x: return #x
      SUNSHINE_SCENE_STATUS(uninitialized); SUNSHINE_SCENE_STATUS(calibrating);
      SUNSHINE_SCENE_STATUS(reference_unstable); SUNSHINE_SCENE_STATUS(calibrated);
      SUNSHINE_SCENE_STATUS(ready); SUNSHINE_SCENE_STATUS(unit_epoch_mismatch);
      SUNSHINE_SCENE_STATUS(proof_missing); SUNSHINE_SCENE_STATUS(source_mismatch);
      SUNSHINE_SCENE_STATUS(frame_mismatch); SUNSHINE_SCENE_STATUS(invalid_source);
      SUNSHINE_SCENE_STATUS(invalid_projection); SUNSHINE_SCENE_STATUS(invalid_depth);
      SUNSHINE_SCENE_STATUS(clear_depth); SUNSHINE_SCENE_STATUS(duplicate_or_backward_sample);
      SUNSHINE_SCENE_STATUS(sample_in_future); SUNSHINE_SCENE_STATUS(stale_sample);
      SUNSHINE_SCENE_STATUS(clock_went_backwards); SUNSHINE_SCENE_STATUS(calibration_timeout);
      SUNSHINE_SCENE_STATUS(no_target); SUNSHINE_SCENE_STATUS(target_expired);
      SUNSHINE_SCENE_STATUS(unsupported_shader_domain);
#undef SUNSHINE_SCENE_STATUS
    }
    return "unknown";
  }
  struct output {
    bool calibrated{}, ready{};
    status reason{status::uninitialized};
    float A{}, inverseB{}, K{}, referenceZPD{reference_zpd}, q0{};
    double normalized_zero{}, reference_inverse_distance{};
    unsigned calibration_samples{};
  };

  class policy {
  public:
    // Explicit user calibration/unit-epoch operation is the ONLY way to change
    // an established K. Calling this again intentionally starts a calibration.
    void reset(std::uint64_t unit_epoch, std::uint64_t now_ms) noexcept {
      *this = policy{};
      epoch_ = unit_epoch;
      start_ms_ = wall_ms_ = now_ms;
    }

    status observe(const sample &value, std::uint64_t now_ms) noexcept {
      if (!epoch_) return status::uninitialized;
      if (value.metadata.unit_epoch != epoch_) return status::unit_epoch_mismatch;
      if (!advance_wall(now_ms)) return status::clock_went_backwards;
      if (!calibrated_ && timed_out(now_ms)) return status::calibration_timeout;
      if (!value.metadata.proof_admitted) return break_reference(status::proof_missing);
      if (!(value.readback_source == value.metadata.source)) return break_reference(status::source_mismatch);
      if (!(value.readback_frame == value.metadata.frame)) return break_reference(status::frame_mismatch);
      if (value.capture_ms > now_ms) return status::sample_in_future;
      if (now_ms - value.capture_ms >= target_expiry_ms || value.capture_ms < start_ms_ ||
          (have_cut_ && value.capture_ms <= cut_ms_)) return status::stale_sample;
      if (have_sample_ && (value.id <= last_id_ || value.capture_ms <= last_capture_ms_ ||
          value.metadata.frame.frame <= last_frame_.frame)) return status::duplicate_or_backward_sample;
      float A{}, inverseB{}, largest_q{};
      auto state = projection(value.metadata, A, inverseB, largest_q);
      if (state != status::ready) return break_reference(state);
      double center_q = 0.0;
      for (unsigned y = 7; y != 11; ++y) {
        for (unsigned x = 14; x != 18; ++x) {
          const float raw = value.raw[y * grid_width + x];
          if (!std::isfinite(raw) || raw < 0.0f || raw > 1.0f) return break_reference(status::invalid_depth);
          // An exact endpoint may be an unwritten clear even for finite-far
          // projections. Without per-cell written evidence reject the whole
          // target/reference, never discard cells and reweight the mean. Tiny
          // interior reversed-Z values remain valid; this is not an epsilon.
          if (raw == 0.0f || raw == 1.0f) return break_reference(status::clear_depth);
          // Evaluate the same float affine coefficients sent to the shader.
          const float q = (raw - A) * inverseB;
          if (!std::isfinite(q) || q < 0.0f) return break_reference(status::invalid_depth);
          center_q += static_cast<double>(q) / 16.0;
        }
      }
      if (!(center_q > 0.0) || !std::isfinite(center_q)) return break_reference(status::invalid_depth);
      if (calibrated_ && !shader_domain(K_, largest_q, center_q)) return status::unsupported_shader_domain;

      if (!calibrated_ && count_ && (!(history_source_ == value.metadata.source) ||
          value.capture_ms - last_capture_ms_ >= target_expiry_ms)) count_ = 0;
      // Malformed/duplicate readbacks never refresh target age or consume an ID.
      have_sample_ = true;
      last_id_ = value.id;
      last_capture_ms_ = value.capture_ms;
      last_frame_ = value.metadata.frame;
      if (!calibrated_) {
        history_source_ = value.metadata.source;
        // Observe EVERY accepted sample while waiting for the minimum span;
        // four retained endpoints must not hide a intervening depth excursion.
        // Running moments are bounded storage and require no sample allocation.
        if (!count_) {
          first_capture_ms_ = value.capture_ms;
          mean_q_ = squared_delta_ = 0.0;
        }
        if (count_ == std::numeric_limits<unsigned>::max()) return break_reference(status::reference_unstable);
        const double delta = center_q - mean_q_;
        mean_q_ += delta / ++count_;
        squared_delta_ += delta * (center_q - mean_q_);
        if (count_ < reference_samples || value.capture_ms - first_capture_ms_ < minimum_span_ms)
          return status::calibrating;
        // Input q is finite FP32; its square and moment sum fit in double even
        // at the bounded unsigned sample-count limit, including tiny units.
        const double cv = std::sqrt(std::max(0.0, squared_delta_) / count_) / mean_q_;
        if (!std::isfinite(cv) || cv > maximum_reference_cv) {
          count_ = 1;
          mean_q_ = center_q;
          squared_delta_ = 0.0;
          first_capture_ms_ = value.capture_ms;
          return status::reference_unstable;
        }
        const double requested_gain = 1.0 / mean_q_;
        if (!std::isfinite(requested_gain) || requested_gain <= 0.0 ||
            requested_gain > std::numeric_limits<float>::max()) return status::unsupported_shader_domain;
        const float candidate_K = static_cast<float>(requested_gain);
        if (!shader_domain(candidate_K, largest_q, mean_q_) || !shader_domain(candidate_K, largest_q, center_q))
          return status::unsupported_shader_domain;
        K_ = candidate_K;
        reference_q_ = mean_q_;
        q0_ = mean_q_;
        calibrated_ = true;
        timing_armed_ = false;
      }
      if (!have_target_ || now_ms - target_capture_ms_ >= target_expiry_ms ||
          !(target_source_ == value.metadata.source)) timing_armed_ = false;
      target_q0_ = center_q;
      target_capture_ms_ = value.capture_ms;
      target_source_ = value.metadata.source;
      have_target_ = true;
      return status::calibrated;
    }

    output tick(const registration &current, std::uint64_t now_ms) noexcept {
      output result;
      result.calibrated = calibrated_;
      result.K = K_;
      result.reference_inverse_distance = reference_q_;
      result.normalized_zero = static_cast<double>(K_) * q0_;
      result.q0 = calibrated_ ? static_cast<float>(q0_) : 0.0f;
      result.calibration_samples = count_;
      const auto fail = [&](status reason) {
        result.reason = reason;
        timing_armed_ = false;
        return result;
      };
      if (!epoch_) return fail(status::uninitialized);
      if (!advance_wall(now_ms)) return fail(status::clock_went_backwards);
      if (current.unit_epoch != epoch_) return fail(status::unit_epoch_mismatch);
      if (!current.proof_admitted) return fail(status::proof_missing);
      float largest_q{};
      auto state = projection(current, result.A, result.inverseB, largest_q);
      if (state != status::ready) return fail(state);
      if (!calibrated_) return fail(timed_out(now_ms) ? status::calibration_timeout : status::calibrating);
      if (!have_target_) return fail(status::no_target);
      if (now_ms < target_capture_ms_ || now_ms - target_capture_ms_ >= target_expiry_ms)
        return fail(status::target_expired);
      if (!(current.source == target_source_)) return fail(status::source_mismatch);
      if (current.frame.frame < last_frame_.frame ||
          (current.frame.frame == last_frame_.frame && !(current.frame == last_frame_)) ||
          (have_current_frame_ && (current.frame.frame < current_frame_.frame ||
            (current.frame.frame == current_frame_.frame && !(current.frame == current_frame_)))))
        return fail(status::frame_mismatch);
      if (!shader_domain(K_, largest_q, q0_) || !shader_domain(K_, largest_q, target_q0_))
        return fail(status::unsupported_shader_domain);

      // No accumulated time credit: invalid ticks disarm timing; a long gap
      // holds q0 on the first resumed tick even if a fresh readback just arrived.
      if (timing_armed_ && now_ms - last_tick_ms_ <= maximum_tick_gap_ms) {
        const double seconds = (now_ms - last_tick_ms_) * 0.001;
        const double delta = (target_q0_ - q0_) * -std::expm1(-seconds / time_constant_seconds);
        // Preserve the motion bound in normalized K*q units while storing q0 directly.
        const double bound = maximum_normalized_speed * seconds / static_cast<double>(K_);
        q0_ += std::max(-bound, std::min(bound, delta));
      }
      last_tick_ms_ = now_ms;
      timing_armed_ = true;
      current_frame_ = current.frame;
      have_current_frame_ = true;
      result.normalized_zero = static_cast<double>(K_) * q0_;
      result.q0 = static_cast<float>(q0_);
      result.reason = status::ready;
      result.ready = true;
      return result;
    }

    void scene_cut(std::uint64_t now_ms) noexcept {
      if (!epoch_ || !advance_wall(now_ms)) return;
      have_cut_ = true;
      cut_ms_ = now_ms;
      have_target_ = false;
      timing_armed_ = false;
      if (!calibrated_) count_ = 0;
      // Deliberately retain K, q_ref, q0 and the original wall-time deadline.
    }

  private:
    status break_reference(status reason) noexcept {
      if (!calibrated_) count_ = 0;
      return reason;
    }
    static bool valid_source(const source_key &v) noexcept {
      return v.native && v.lifetime && v.width && v.height && v.extent_width && v.extent_height &&
        v.left < v.width && v.top < v.height && v.extent_width <= v.width - v.left &&
        v.extent_height <= v.height - v.top;
    }
    static status projection(const registration &v, float &A, float &inverseB, float &largest_q) noexcept {
      if (!valid_source(v.source)) return status::invalid_source;
      if (!std::isfinite(v.A) || !std::isfinite(v.B) || v.B == 0.0) return status::invalid_projection;
      A = static_cast<float>(v.A);
      inverseB = static_cast<float>(1.0 / v.B);
      if (!std::isfinite(A) || !std::isfinite(inverseB) || inverseB == 0.0f) return status::invalid_projection;
      const float q0 = (0.0f - A) * inverseB, q1 = (1.0f - A) * inverseB;
      largest_q = std::max(q0, q1);
      if (!std::isfinite(q0) || !std::isfinite(q1) || q0 < 0.0f || q1 < 0.0f || !(largest_q > 0.0f))
        return status::invalid_projection;
      return status::ready;
    }
    static bool shader_domain(float K, float largest_q, double zero_q) noexcept {
      return shader_coordinate_domain(K, largest_q, zero_q);
    }
    bool advance_wall(std::uint64_t now_ms) noexcept {
      if (now_ms < wall_ms_) return false;
      wall_ms_ = now_ms;
      return true;
    }
    bool timed_out(std::uint64_t now_ms) const noexcept { return now_ms - start_ms_ > calibration_timeout_ms; }
    std::uint64_t epoch_{}, start_ms_{}, wall_ms_{}, last_id_{}, last_capture_ms_{}, target_capture_ms_{}, last_tick_ms_{}, cut_ms_{}, first_capture_ms_{};
    frame_key last_frame_, current_frame_;
    source_key target_source_, history_source_;
    unsigned count_{};
    float K_{};
    double reference_q_{}, q0_{}, target_q0_{}, mean_q_{}, squared_delta_{};
    bool have_sample_{}, calibrated_{}, have_target_{}, timing_armed_{}, have_cut_{}, have_current_frame_{};
  };
}
