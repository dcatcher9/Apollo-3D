// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "camera_scene_policy.h"
#include "scene_feedback.h"
#include "depth_moments.h"
#include "game3d_stereo_contract.h"

// Independent stereo gain and zero plane. Source admission and asynchronous
// readback ownership remain with the callers. No percentile depth clipping.
namespace sunshine_scene_gain {
  inline constexpr unsigned reference_samples = 4;
  inline constexpr std::uint64_t reference_span_ms = 750, reference_expiry_ms = 1500;
  // Maximum zero-plane translation in full-strength per-eye display budgets
  // per second. Independent of inverse-depth units and artistic strength.
  inline constexpr double zero_budget_per_second = 1.;
  struct depth_range {
    double minimum{}, maximum{}, mean{}, mean_square{};
    // Zero count identifies legacy point-only fixtures. Live geometry carries
    // the full active-rectangle measurement, independently of selector points.
    std::uint64_t pixel_count{};
    std::uint32_t tiles_x{}, tiles_y{};
    depth_range() = default;
    // Two-value construction is convenient for synthetic uniform fixtures.
    // Production decoding supplies the independently measured full-pixel mean.
    depth_range(double low, double high) noexcept : minimum(low), maximum(high), mean(low*.5+high*.5) {}
    depth_range(double low, double high, double average) noexcept : minimum(low), maximum(high), mean(average) {}
    bool valid() const noexcept {
      return std::isfinite(minimum) && std::isfinite(maximum) && std::isfinite(mean) &&
        minimum >= 0 && maximum >= minimum && mean >= minimum && mean <= maximum;
    }
  };
  // Production supplies full GPU moments and extrema with one frozen decoding
  // basis. Grid fallback supports older fixtures only; failed full statistics
  // must never turn into an apparently successful sparse measurement.
  template<std::size_t N>
  inline depth_range decode_range(const std::array<float, N> &grid, bool supplied, bool valid,
      float minimum, float maximum, float A, float inverseB, float raw_min = 0, float raw_max = 1,
      const sunshine_depth_statistics::moments &moments = {}) noexcept {
    static_assert(N != 0, "Scene reference needs a nonempty point grid");
    const auto invalid = [] { return depth_range{std::numeric_limits<double>::quiet_NaN(), 0}; };
    if (supplied && !valid) return invalid();
    if (moments.supplied && (!supplied || !moments.valid || !moments.count ||
        moments.A != A || moments.inverseB != inverseB ||
        !std::isfinite(moments.sum) || !std::isfinite(moments.sum_squares) ||
        moments.sum < 0. || moments.sum_squares < 0.)) return invalid();
    double mean = 0.;
    if (!supplied) {
      minimum = std::numeric_limits<float>::infinity();
      maximum = -minimum;
    }
    if (!moments.supplied) for (float value : grid) {
      if (!std::isfinite(value) || value < raw_min || value > raw_max) return invalid();
      const float decoded = (value-A)*inverseB;
      if (!std::isfinite(decoded) || decoded < 0) return invalid();
      mean += decoded;
      if (!supplied) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
      }
    }
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum < raw_min || maximum > raw_max || minimum > maximum)
      return invalid();
    const float a = (minimum-A)*inverseB, b = (maximum-A)*inverseB;
    if (!std::isfinite(a) || !std::isfinite(b) || a < 0.f || b < 0.f) return invalid();
    if (!moments.supplied) return {std::min(a,b), std::max(a,b), mean/N};
    mean = moments.sum / moments.count;
    const double mean_square = moments.sum_squares / moments.count;
    const double low = std::min(a,b), high = std::max(a,b);
    // GPU sums are FP32 reductions; restore only their rounding overshoot to
    // the measured range. No source pixel or depth extremum is clipped.
    constexpr double relative_roundoff = 256. * std::numeric_limits<float>::epsilon();
    const double bounded_mean = std::clamp(mean, low, high);
    if (!std::isfinite(mean) || !std::isfinite(mean_square) ||
        std::abs(mean-bounded_mean) > relative_roundoff*high ||
        mean_square < bounded_mean*bounded_mean*(1.-relative_roundoff) ||
        mean_square > high*bounded_mean*(1.+relative_roundoff)) return invalid();
    depth_range out{low, high, bounded_mean};
    out.mean_square = mean_square;
    out.pixel_count = moments.count;
    out.tiles_x = moments.tiles_x;
    out.tiles_y = moments.tiles_y;
    return out;
  }
  // Pre-warp displacement units, not a universal viewing-comfort guarantee.
  struct limits { double strength = .5, foreground = 1.5, background = 1.536, normalization = 1.; };
  inline limits render_limits(float strength, unsigned width, unsigned height) noexcept {
    const double container = width && height ? .04 * width * 2160.0 / (100.0 * height) : 1.536;
    const double foreground = std::min(1.5, container), background = std::min(2.5, container);
    const double display = double(sunshine_game3d::default_disparity_limit_uv) * container / .04;
    return {std::isfinite(strength) ? std::clamp(double(strength) * .01, 0.0, 1.0) : 0.0,
      foreground, background, std::min({foreground, background, display}) / sunshine_camera_scene::reference_zpd};
  }
  enum class refinement { holding, adapting, history_hold, limited };
  inline const char *name(refinement value) noexcept {
    switch (value) {
      case refinement::holding: return "holding_stereo_controls";
      case refinement::adapting: return "adjusting_stereo_controls";
      case refinement::history_hold: return "history_hold";
      case refinement::limited: return "approaching_gain_target";
    }
    return "unknown";
  }
  enum class status { calibrating, ready, unsupported };

  class policy {
  public:
    void reset() noexcept { *this = {}; }
    void reset_reference() noexcept {
      const auto revision = revision_;
      const auto budget = budget_;
      *this = {};
      revision_ = revision;
      budget_ = budget;
    }
    void configure(limits value) noexcept {
      if (std::isfinite(value.strength) && value.strength >= 0 && value.strength <= 1 &&
          std::isfinite(value.foreground) && value.foreground > 0 &&
          std::isfinite(value.background) && value.background > 0 &&
          std::isfinite(value.normalization) && value.normalization > 0) {
        budget_ = value;
        budget_.normalization = std::min({value.normalization,
          value.foreground / sunshine_camera_scene::reference_zpd,
          value.background / sunshine_camera_scene::reference_zpd});
      }
    }
    void suspend() noexcept { recovery_armed_ = false; }
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

    status observe(depth_range range, std::uint64_t capture_ms,
        const sunshine_scene_feedback::sample &feedback = {}) noexcept {
      if (!accepts(feedback)) return initialized_ ? status::ready : status::calibrating;
      if (feedback.reset) { invalidate(); refinement_ = refinement::history_hold; return initialized_ ? status::ready : status::calibrating; }
      if (!range.valid()) { invalidate(); return status::unsupported; }
      // Flat title/clear frames cannot seed gain. Once initialized, a positive
      // flat surface can still converge to the screen while gain stays held.
      // All-infinity frames provide neither a gain nor a finite zero anchor.
      const bool has_span = range.maximum > range.minimum;
      if (range.maximum == 0. || (!has_span && !initialized_)) {
        invalidate();
        return initialized_ ? status::ready : status::calibrating;
      }
      // The nearest observed depth supplies the ratio, independently of its
      // screen coverage. The GPU enforces the display budget on current pixels;
      // asynchronously observed extrema are never a current-frame guarantee.
      const double live_gain = has_span ? budget_.normalization/range.maximum : gain_;
      if (!std::isfinite(live_gain) || live_gain > std::numeric_limits<float>::max()) {
        invalidate();
        return status::unsupported;
      }
      float target = static_cast<float>(live_gain);
      if (target > live_gain) target = std::nextafter(target, 0.f);
      if (!std::isnormal(target) || !(target > 0)) { invalidate(); return status::unsupported; }
      if (!initialized_) {
        constexpr auto spacing = reference_span_ms/(reference_samples-1);
        if (count_ && (capture_ms <= reference_ms_ || capture_ms-reference_ms_ < spacing)) return status::calibrating;
        if (!count_) { first_ms_ = capture_ms; initial_zero_ = initial_maximum_ = 0.; }
        reference_ms_ = capture_ms;
        const double midpoint = range.minimum*.5 + range.maximum*.5;
        initial_zero_ += (midpoint-initial_zero_)/(count_+1);
        initial_maximum_ += (range.maximum-initial_maximum_)/(count_+1);
        if (++count_ < reference_samples) return status::calibrating;
        // Gain uses the nearest-depth reference, while zero has its own seed.
        // Neither the display budget nor subsequent gain changes move zero.
        const double gain = budget_.normalization/initial_maximum_;
        gain_ = static_cast<float>(gain);
        if (gain_ > gain) gain_ = std::nextafter(gain_, 0.f);
        if (!std::isnormal(gain_) || !(gain_ > 0)) { count_ = 0; return status::unsupported; }
        zero_ = initial_zero_;
        initialized_ = true;
      }
      requested_gain_ = has_span ? target : 0.f;
      range_ = range;
      target_ms_ = capture_ms;
      have_target_ = true;
      return status::ready;
    }
    void update(std::uint64_t now_ms) noexcept {
      advance(now_ms);
      if (!initialized_ || !have_target_) return;
      // Output extent can change without a new depth sample. Recompute only
      // the display multiplier; preserve the source-owned depth reference.
      const auto floor_float = [](double v) {
        float result = static_cast<float>(v);
        return result > v ? std::nextafter(result, -std::numeric_limits<float>::infinity()) : result;
      };
      if (has_gain_target()) {
        const double target = budget_.normalization/range_.maximum;
        if (!std::isfinite(target) || target > std::numeric_limits<float>::max()) { invalidate(); return; }
        requested_gain_ = floor_float(target);
        if (!std::isnormal(requested_gain_) || !(requested_gain_ > 0)) { invalidate(); return; }
      }
      if (budget_.strength == 0) {
        suspend();
        refinement_ = refinement::holding;
        return;
      }
      const float previous_gain = gain_;
      const double previous_zero = zero_;
      // Smooth in both directions: a brief close point must not immediately
      // flatten every other surface. Current pixels receive the GPU limit even
      // before this asynchronous target arrives or the gain reaches it.
      // Missing/expired depth and presentation gaps grant no catch-up time.
      if (recovery_armed_ && now_ms >= last_tick_ms_ &&
          now_ms-last_tick_ms_ <= sunshine_camera_scene::maximum_tick_gap_ms) {
        const double seconds = (now_ms-last_tick_ms_)*.001;
        const double alpha = -std::expm1(-seconds/sunshine_camera_scene::time_constant_seconds);
        if (has_gain_target()) {
          const double delta = alpha*std::log(double(requested_gain_)/gain_);
          const double bound = std::log(2.0)*seconds;
          const float next = static_cast<float>(double(gain_)*std::exp(std::clamp(delta, -bound, bound)));
          // Finish a sub-ULP exponential tail in representable steps instead of
          // getting permanently stuck a few floats short of the live target.
          gain_ = seconds > 0 && next == gain_ && gain_ != requested_gain_ ?
            std::nextafter(gain_, requested_gain_) : next;
        }
        // The range midpoint minimizes max |q-q0| at fixed gain. Limit its
        // movement in display units, so a cut or near outlier cannot teleport
        // the entire image. Never clamp the applied zero into the new range:
        // that would bypass this bound. The GPU still caps current disparity.
        const double bound = budget_.normalization/double(gain_) * zero_budget_per_second * seconds;
        zero_ += std::clamp(alpha*(target_zero()-zero_), -bound, bound);
      }
      last_tick_ms_ = now_ms;
      recovery_armed_ = true;
      refinement_ = limited() ? refinement::limited : zero_ == previous_zero && gain_ == previous_gain ? refinement::holding : refinement::adapting;
    }
    bool initialized() const noexcept { return initialized_; }
    bool has_target() const noexcept { return have_target_; }
    bool has_gain_target() const noexcept { return initialized_ && have_target_ && range_.maximum > range_.minimum; }
    // Copy only accepted, still-current measurements. The measured mean
    // is distinct from the reciprocal of the rounded FP32 gain target.
    bool depth_statistics(depth_range &out) const noexcept {
      if (!initialized_ || !have_target_) return false;
      out = range_;
      return true;
    }
    float zero() const noexcept { return initialized_ ? static_cast<float>(zero_) : 0.f; }
    float value() const noexcept { return initialized_ ? gain_ : 0.f; }
    double reference() const noexcept { return initialized_ && have_target_ ? range_.maximum : 0.0; }
    double target() const noexcept { return has_gain_target() ? requested_gain_ : 0.0; }
    double target_zero() const noexcept { return initialized_ && have_target_ ? range_.minimum*.5 + range_.maximum*.5 : 0.0; }
    double lower() const noexcept { return range_.minimum; }
    double upper() const noexcept { return range_.maximum; }
    bool limited() const noexcept { return has_gain_target() && gain_ < requested_gain_; }
    unsigned samples() const noexcept { return count_; }
    refinement learning() const noexcept { return refinement_; }

  private:
    std::uint64_t first_ms_{}, reference_ms_{}, revision_{}, target_ms_{}, wall_ms_{}, last_tick_ms_{};
    double zero_{}, initial_zero_{}, initial_maximum_{};
    float gain_{}, requested_gain_{};
    depth_range range_;
    limits budget_;
    unsigned count_{};
    bool initialized_{}, have_target_{}, have_wall_{}, recovery_armed_{};
    refinement refinement_{refinement::holding};
  };
}
