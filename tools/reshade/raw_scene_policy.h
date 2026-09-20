// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "camera_scene_policy.h"
#include "native_depth_calibration.h"
#include "scene_gain.h"

// Relative controls for the selected depth/color pair. No camera matrices,
// meters or final-color registration are recovered. Whole-viewport extrema
// supply a tracked midpoint screen plane; the full-image maximum supplies
// its independent gain reference. Ratios cancel positive multiplicative units,
// not unknown additive offsets in an undeclared depth encoding.
// Unknown encoding changes remain a limitation.
namespace sunshine_raw_scene {
  // Shared value types/constants only; the raw controller does not invoke the
  // physical-camera policy or claim its projection/registration proof.
  namespace timing = sunshine_camera_scene;
  using source_key = timing::source_key;
  using frame_key = timing::frame_key;
  using orientation = sunshine_depth::depth_orientation;
  inline constexpr unsigned grid_width = timing::grid_width, grid_height = timing::grid_height;

  struct selected_frame {
    std::uint64_t basis_epoch{}, layout_epoch{}, convention_epoch{};
    source_key source;
    // Monotonic selected-scene capture sequence and its capture generation.
    frame_key frame;
    // Use detected(), not a clear hint or manual orientation override.
    orientation direction{orientation::automatic};
    bool depth_ready{}, copy_ambiguous{}, aligned_viewport_assumed{};
    sunshine_scene_feedback::sample feedback;
    // Zero means projection/encoding changes are unobservable. A reported
    // convention change starts a fresh reference, like a selected-source change.
  };
  struct sample {
    std::uint64_t id{}, capture_ms{};
    selected_frame metadata; // Immutable submission-time selected-depth facts.
    frame_key readback_frame;
    source_key readback_source;
    std::uint64_t readback_layout_epoch{};
    std::array<float, grid_width * grid_height> raw{}; // Unmodified selector points; legacy fixture fallback.
    bool range_supplied{}, range_valid{};
    float range_min{}, range_max{};
    sunshine_depth_statistics::moments moments;
  };
  enum class status {
    uninitialized, basis_epoch_mismatch, direction_unknown,
    depth_unavailable, ambiguous_copy, alignment_unavailable,
    invalid_source, source_mismatch, frame_mismatch,
    calibrating, calibrated, ready, holding_reference, invalid_depth, clear_depth,
    clock_went_backwards, no_target, target_expired,
    unsupported_shader_domain
  };
  inline const char *name(status value) noexcept {
    switch (value) {
#define SUNSHINE_RAW_STATUS(x) case status::x: return #x
      SUNSHINE_RAW_STATUS(uninitialized); SUNSHINE_RAW_STATUS(basis_epoch_mismatch);
      SUNSHINE_RAW_STATUS(direction_unknown);
      SUNSHINE_RAW_STATUS(depth_unavailable); SUNSHINE_RAW_STATUS(ambiguous_copy);
      SUNSHINE_RAW_STATUS(alignment_unavailable); SUNSHINE_RAW_STATUS(invalid_source);
      SUNSHINE_RAW_STATUS(source_mismatch); SUNSHINE_RAW_STATUS(frame_mismatch);
      SUNSHINE_RAW_STATUS(calibrating);
      SUNSHINE_RAW_STATUS(calibrated);
      SUNSHINE_RAW_STATUS(ready); SUNSHINE_RAW_STATUS(invalid_depth);
      SUNSHINE_RAW_STATUS(holding_reference);
      SUNSHINE_RAW_STATUS(clear_depth);
      SUNSHINE_RAW_STATUS(clock_went_backwards);
      SUNSHINE_RAW_STATUS(no_target); SUNSHINE_RAW_STATUS(target_expired);
      SUNSHINE_RAW_STATUS(unsupported_shader_domain);
#undef SUNSHINE_RAW_STATUS
    }
    return "unknown";
  }
  struct output {
    bool calibrated{}, ready{};
    status reason{status::uninitialized};
    // Synthetic shader basis, NOT recovered projection coefficients.
    float shader_A{}, shader_inverseB{}, H{}, referenceZPD{timing::reference_zpd}, t0{};
    double normalized_zero{}, reference_t{};
    unsigned calibration_samples{};
    double target_H{};
    sunshine_scene_gain::refinement learning{};
    bool limited{};
    sunshine_scene_gain::depth_range depth_statistics{};
    bool has_depth_statistics{};
    double target_t0{};
  };

  class policy {
  public:
    void configure(sunshine_scene_gain::limits budget) noexcept { budget_ = budget; gain_.configure(budget); }
    // Explicit reference reset discards numeric state and requests fresh packets
    // in a strictly newer caller-owned epoch. Gain and zero initialize separately.
    bool reset(std::uint64_t basis_epoch, std::uint64_t now_ms) noexcept {
      if (!basis_epoch || basis_epoch <= epoch_) return false;
      *this = policy{};
      epoch_ = basis_epoch;
      wall_ms_ = minimum_capture_ms_ = now_ms;
      have_wall_ = true;
      return true;
    }

    // The authoritative owner binds resource identity independently of whether
    // this present has a usable depth copy. Samples never choose their own basis.
    void bind(const selected_frame &basis, std::uint64_t now_ms) noexcept {
      if (!advance(now_ms) || basis.basis_epoch != epoch_ || !valid_source(basis)) return;
      if (have_basis_ && same_basis(basis, basis_)) {
        if (gain_.synchronize(basis.feedback)) {
          // A reset invalidates old comfort targets, not current depth pixels.
          evidence_reason_ = gain_.initialized() ? status::holding_reference : status::calibrating;
        }
        basis_.feedback = basis.feedback;
        return;
      }
      if (have_basis_) {
        minimum_capture_ms_ = now_ms;
        minimum_frame_ = basis.frame.frame;
      }
      clear_basis_state();
      basis_ = basis;
      have_basis_ = true;
      gain_.synchronize(basis.feedback);
    }

    // Ingest immutable completed measurements without consulting presentation
    // availability. The capture owner supplies an independent frame watermark.
    void observe(const sample &value, const frame_key &observed_frame, std::uint64_t now_ms) noexcept {
      if (!advance(now_ms) || !have_basis_ || !fresh_packet(value, observed_frame, now_ms)) return;
      last_id_ = value.id;
      last_capture_ms_ = value.capture_ms;
      last_sample_frame_ = value.metadata.frame;
      have_sample_ = true;
      const auto reject = [&](status reason) {
        invalidate(now_ms);
        evidence_reason_ = reason;
      };
      if (!(value.readback_frame == value.metadata.frame)) { reject(status::frame_mismatch); return; }
      if (!(value.readback_source == value.metadata.source) ||
          value.readback_layout_epoch != value.metadata.layout_epoch) { reject(status::source_mismatch); return; }
      const auto admitted = admission(value.metadata);
      if (admitted != status::ready) { reject(admitted); return; }
      if (value.metadata.feedback.reset) {
        gain_.invalidate();
        evidence_reason_ = gain_.initialized() ? status::holding_reference : status::calibrating;
        return;
      }
      const bool normal = value.metadata.direction == orientation::normal;
      const auto range = sunshine_scene_gain::decode_range(value.raw, value.range_supplied,
        value.range_valid, value.range_min, value.range_max, normal ? 1.f : 0.f, normal ? -1.f : 1.f,
        0.f, 1.f, value.moments);
      if (!range.valid()) { reject(status::invalid_depth); return; }
      const auto gain_status = gain_.observe(range, value.capture_ms, value.metadata.feedback);
      if (gain_status == sunshine_scene_gain::status::unsupported) {
        evidence_reason_ = status::unsupported_shader_domain;
        return;
      }
      if (!gain_.initialized()) {
        evidence_reason_ = status::calibrating;
        return;
      }
      target_frame_ = value.metadata.frame;
      evidence_reason_ = gain_.has_target() ? status::ready : status::holding_reference;
    }

    // Rendering uses only the actual current pair. Missing pixels suspend the
    // motion clock, not valid exact-source measurements still in flight.
    output evaluate(const selected_frame &current, std::uint64_t now_ms) noexcept {
      if (!epoch_) return result(status::uninitialized);
      if (!advance(now_ms)) return result(status::clock_went_backwards);
      if (current.basis_epoch != epoch_) {
        invalidate(now_ms);
        return result(status::basis_epoch_mismatch);
      }
      if (!current.frame.frame && !current.depth_ready) {
        suspend();
        return result(status::depth_unavailable);
      }
      if (!current.frame.frame || (have_current_ &&
          (current.frame.frame < current_frame_.frame ||
           (current.frame.frame == current_frame_.frame && !(current.frame == current_frame_))))) {
        suspend();
        return result(status::frame_mismatch);
      }
      current_frame_ = current.frame;
      have_current_ = true;
      const auto admitted = admission(current);
      if (admitted != status::ready) {
        suspend();
        return result(admitted);
      }
      if (!have_basis_ || !same_basis(current, basis_)) {
        suspend();
        return result(status::source_mismatch);
      }
      if (!gain_.initialized()) return result(evidence_reason_);
      if (target_frame_.frame > current.frame.frame || target_frame_.token_generation != current.frame.token_generation) {
        suspend();
        return result(status::frame_mismatch);
      }
      if (!gain_.has_target()) {
        // A missing comfort measurement is not missing depth. An initialized
        // exact-source reference can warp a proven current copy unchanged.
        // Only ordinary timeout permits this hold; explicit invalidation, cuts
        // and malformed evidence still require a fresh authenticated target.
        if (evidence_reason_ == status::target_expired || evidence_reason_ == status::holding_reference)
          return result(shader_domain(gain_.value(), gain_.zero()) ? status::holding_reference : status::unsupported_shader_domain);
        return result(evidence_reason_);
      }
      gain_.update(now_ms);
      if (!shader_domain(gain_.value(), gain_.zero())) {
        // Hold rendering, not numerical progress. A fresh valid target can move
        // the current zero into the shader domain without an explicit reset.
        return result(status::unsupported_shader_domain);
      }
      return result(status::ready);
    }

    // Compatibility convenience; production synchronizes/observes/evaluates
    // explicitly so even an off-turn source receives its completed measurements.
    output update(const selected_frame &current, const sample *packet, std::uint64_t now_ms) noexcept {
      bind(current, now_ms);
      if (packet) observe(*packet, current.frame, now_ms);
      return evaluate(current, now_ms);
    }

    // Monotonic time/expiry belongs to evidence, not to depth availability.
    bool advance(std::uint64_t now_ms) noexcept {
      if (!epoch_) return false;
      if (have_wall_ && now_ms < wall_ms_) { invalidate(wall_ms_); return false; }
      wall_ms_ = now_ms;
      have_wall_ = true;
      const auto pending_samples = gain_.samples();
      const bool had_target = gain_.has_target();
      gain_.advance(now_ms);
      if (!gain_.initialized() && pending_samples && !gain_.samples()) evidence_reason_ = status::calibrating;
      if (had_target && !gain_.has_target()) evidence_reason_ = status::target_expired;
      return true;
    }
    void suspend() noexcept { gain_.suspend(); }
    // Strict provenance failures and explicit cuts invalidate target evidence;
    // ordinary missing presents never move this asynchronous admission floor.
    void invalidate(std::uint64_t now_ms) noexcept {
      gain_.invalidate();
      minimum_capture_ms_ = std::max(minimum_capture_ms_, std::max(wall_ms_, now_ms));
      evidence_reason_ = gain_.initialized() ? status::no_target : status::calibrating;
    }

    // Hold scale and screen plane across an explicit cut, but require a target
    // captured strictly after it. No time from the interruption earns movement.
    void scene_cut(std::uint64_t now_ms) noexcept {
      if (!epoch_ || (have_wall_ && now_ms < wall_ms_)) return;
      wall_ms_ = cut_ms_ = now_ms;
      have_wall_ = have_cut_ = true;
      invalidate(now_ms);
    }

  private:
    static bool known_direction(orientation value) noexcept {
      return value == orientation::normal || value == orientation::reversed;
    }
    static bool valid_source(const selected_frame &value) noexcept {
      const auto &v = value.source;
      return value.layout_epoch && v.native && v.lifetime && v.width && v.height &&
        v.extent_width && v.extent_height && v.left < v.width && v.top < v.height &&
        v.extent_width <= v.width - v.left && v.extent_height <= v.height - v.top;
    }
    static status admission(const selected_frame &value) noexcept {
      if (!valid_source(value)) return status::invalid_source;
      if (!known_direction(value.direction)) return status::direction_unknown;
      if (!value.depth_ready) return status::depth_unavailable;
      if (value.copy_ambiguous) return status::ambiguous_copy;
      if (!value.aligned_viewport_assumed) return status::alignment_unavailable;
      return status::ready;
    }
    static bool same_basis(const selected_frame &a, const selected_frame &b) noexcept {
      return a.source == b.source && a.layout_epoch == b.layout_epoch &&
        a.convention_epoch == b.convention_epoch && a.direction == b.direction;
    }
    bool fresh_packet(const sample &value, const frame_key &observed_frame, std::uint64_t now_ms) const noexcept {
      if (!value.id || value.metadata.basis_epoch != epoch_ || !same_basis(value.metadata, basis_) ||
          !gain_.accepts(value.metadata.feedback) ||
          value.capture_ms > now_ms || value.capture_ms < minimum_capture_ms_ ||
          now_ms - value.capture_ms >= timing::target_expiry_ms ||
          (have_cut_ && value.capture_ms <= cut_ms_) || !value.metadata.frame.frame ||
          value.metadata.frame.frame < minimum_frame_ || value.metadata.frame.frame > observed_frame.frame ||
          value.metadata.frame.token_generation != observed_frame.token_generation) return false;
      return !have_sample_ || (value.id > last_id_ && value.capture_ms > last_capture_ms_ &&
        value.metadata.frame.frame > last_sample_frame_.frame);
    }
    static bool shader_domain(float H, double zero) noexcept {
      return timing::shader_coordinate_domain(H, 1.0f, zero);
    }
    void clear_basis_state() noexcept {
      have_current_ = have_sample_ = false;
      // Frame and sample counters belong to the selected encoding domain. SL
      // and NGX have independent sequences, so the previous owner's watermarks
      // cannot constrain this one. same_basis and the rebind capture floor still
      // reject delayed packets from the old owner or an earlier visit.
      current_frame_ = last_sample_frame_ = target_frame_ = {};
      last_id_ = last_capture_ms_ = 0;
      gain_.reset();
      gain_.configure(budget_);
      evidence_reason_ = status::calibrating;
      // reset/switch owns the capture floor. Initial adoption may consume a
      // valid asynchronous packet captured after reset but before this update.
    }
    float gain() const noexcept { return gain_.value(); }
    output result(status reason) const noexcept {
      output out;
      out.calibrated = gain_.initialized();
      out.ready = reason == status::ready || reason == status::holding_reference;
      out.reason = reason;
      out.shader_A = have_basis_ && basis_.direction == orientation::normal ? 1.0f : 0.0f;
      out.shader_inverseB = have_basis_ && basis_.direction == orientation::normal ? -1.0f : 1.0f;
      out.H = gain();
      out.t0 = gain_.zero();
      out.normalized_zero = static_cast<double>(out.H) * out.t0;
      out.reference_t = gain_.reference();
      out.calibration_samples = gain_.samples();
      out.target_H = gain_.target();
      out.learning = gain_.learning();
      out.limited = gain_.limited();
      out.has_depth_statistics = gain_.depth_statistics(out.depth_statistics);
      out.target_t0 = gain_.target_zero();
      return out;
    }

    selected_frame basis_;
    frame_key current_frame_, last_sample_frame_, target_frame_;
    std::uint64_t epoch_{}, wall_ms_{}, minimum_capture_ms_{}, minimum_frame_{},
      last_id_{}, last_capture_ms_{}, cut_ms_{};
    sunshine_scene_gain::policy gain_;
    sunshine_scene_gain::limits budget_;
    status evidence_reason_{status::calibrating};
    bool have_wall_{}, have_current_{}, have_basis_{}, have_sample_{}, have_cut_{};
  };
}
