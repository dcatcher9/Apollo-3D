/** @file src/video_frame_diagnostics.h
 *  @brief Timestamp boundaries for optional host video diagnostics; never scheduling authority.
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <utility>

namespace video::detail {
  using diagnostic_clock_t = std::chrono::steady_clock;
  using diagnostic_timestamp_t = std::optional<diagnostic_clock_t::time_point>;

  /** An inactive optional must not construct trackers that read clocks or allocate strings. */
  template<class State>
  std::optional<State> make_diagnostic_state(bool enabled) {
    if (enabled) {
      return std::optional<State> {std::in_place};
    }
    return std::nullopt;
  }

  enum class diagnostic_content_e {
    unknown,
    new_content,
    repeated_content,
  };

  /** Retained/older pixels must not be reported as a new-content processing stall. */
  class diagnostic_content_tracker_t {
  public:
    diagnostic_content_e observe(
      const diagnostic_timestamp_t &content_timestamp,
      bool explicitly_repeated = false
    ) noexcept {
      if (!content_timestamp) {
        return diagnostic_content_e::unknown;
      }
      const bool repeated = explicitly_repeated ||
                            (latest_ && *content_timestamp <= *latest_);
      if (!latest_ || *content_timestamp > *latest_) {
        latest_ = content_timestamp;
      }
      return repeated ? diagnostic_content_e::repeated_content :
                        diagnostic_content_e::new_content;
    }

  private:
    diagnostic_timestamp_t latest_;
  };

  template<class Now>
  diagnostic_timestamp_t diagnostic_timestamp(bool enabled, const Now &now) {
    if (!enabled) {
      return std::nullopt;
    }
    return now();
  }

  /** Where the encode loop's time went, for one line every 20 s while diagnostics are enabled.
   *
   * Totals of one window, recorded on the encode thread without per-frame logging. The window
   * minus holding, waiting, converting and encoding is the loop's own work (control requests,
   * schedule and lifecycle checks).
   */
  class encode_loop_stats_t {
  public:
    using duration_t = std::chrono::nanoseconds;

    void iteration() noexcept {
      ++iterations_;
    }

    /** An exact sleep to an independent provider's poll target (provider_hold()). */
    void held(duration_t requested, duration_t actual) noexcept {
      ++holds_;
      hold_requested_ += requested;
      hold_actual_ += actual;
      hold_overshoot_max_ = std::max(hold_overshoot_max_, std::max(actual - requested, duration_t::zero()));
    }

    /** Whether image waits keep their bound on the deadline timer (detail::pop_encode_image()) or
     *  end at a scheduler tick; kept across reset(). */
    void set_deadline_timer(bool deadline_timer) noexcept {
      deadline_timer_ = deadline_timer;
    }

    /** A wait on the image event. One that lasted its whole bound timed out (nothing woke it);
     *  its overshoot past the bound is the deadline timer's, or without it the scheduler tick's. */
    void waited(duration_t requested, duration_t actual) noexcept {
      ++waits_;
      wait_requested_ += requested;
      wait_actual_ += actual;
      if (actual >= requested) {
        ++timeouts_;
        timeout_overshoot_ += actual - requested;
        timeout_overshoot_max_ = std::max(timeout_overshoot_max_, actual - requested);
      }
    }

    void converted(duration_t elapsed) noexcept {
      ++conversions_;
      converting_ += elapsed;
    }

    /** One encode step of the loop, with its NVENC picture submission and the time the loop waited
     *  for a picture's completion: its own with one picture at a time, else the oldest one's while
     *  every picture was in flight. `new_content` is the encoded content's identity
     *  (diagnostic_content_tracker_t), not whether a conversion ran: a keepalive that re-renders an
     *  unchanged export encodes repeated content. */
    void encoded(bool new_content, duration_t call, duration_t submit, duration_t completion_wait) noexcept {
      ++(new_content ? new_encodes_ : repeat_encodes_);
      encoding_ += call;
      submitting_ += submit;
      completion_waiting_ += completion_wait;
    }

    /** NVENC pictures that may be in flight; kept across reset(). */
    void set_pipeline_depth(unsigned depth) noexcept {
      depth_ = depth;
    }

    /** Pictures published: `behind` were submitted while an earlier one was still in flight, and
     *  each took from its submission to its packet `latency_total` in sum, `latency_max` at most. */
    void pictures(std::uint64_t behind, std::uint64_t published, duration_t latency_total, duration_t latency_max) noexcept {
      pictures_behind_ += behind;
      pictures_ += published;
      picture_latency_ += latency_total;
      picture_latency_max_ = std::max(picture_latency_max_, latency_max);
    }

    [[nodiscard]] std::uint64_t iterations() const noexcept {
      return iterations_;
    }

    [[nodiscard]] std::string report(duration_t window) const {
      const auto ms = [](duration_t value) {
        return std::chrono::duration<double, std::milli>(value).count();
      };
      const auto average = [&](duration_t total, std::uint64_t count) {
        return count ? ms(total) / static_cast<double>(count) : 0.0;
      };
      const auto busy = hold_actual_ + wait_actual_ + converting_ + encoding_;
      return std::format(
        "Video encode loop: {} iterations in {:.1f} s; {} new-content and {} repeated-content encodes; holding {:.1f} ms in {} exact holds "
        "(requested {:.1f} ms, overshoot avg {:.2f} max {:.2f} ms); waiting {:.1f} ms in {} image waits (requested {:.1f} ms; "
        "{} ran to their bound {}, overshoot avg {:.2f} max {:.2f} ms); converting {:.1f} ms in {} conversions; encoding {:.1f} ms "
        "(NVENC submit {:.1f} ms, completion wait {:.1f} ms; up to {} {} in flight, {} submitted behind another, submission to packet "
        "avg {:.2f} max {:.2f} ms); loop work {:.1f} ms.",
        iterations_,
        std::chrono::duration<double>(window).count(),
        new_encodes_,
        repeat_encodes_,
        ms(hold_actual_),
        holds_,
        ms(hold_requested_),
        average(hold_actual_ - hold_requested_, holds_),
        ms(hold_overshoot_max_),
        ms(wait_actual_),
        waits_,
        ms(wait_requested_),
        timeouts_,
        deadline_timer_ ? "on the deadline timer" : "at a scheduler tick",
        average(timeout_overshoot_, timeouts_),
        ms(timeout_overshoot_max_),
        ms(converting_),
        conversions_,
        ms(encoding_),
        ms(submitting_),
        ms(completion_waiting_),
        depth_,
        depth_ == 1 ? "picture" : "pictures",
        pictures_behind_,
        average(picture_latency_, pictures_),
        ms(picture_latency_max_),
        ms(std::max(window - busy, duration_t::zero()))
      );
    }

    void reset() noexcept {
      const auto depth = depth_;
      const bool deadline_timer = deadline_timer_;
      *this = {};
      depth_ = depth;
      deadline_timer_ = deadline_timer;
    }

  private:
    std::uint64_t iterations_ = 0, holds_ = 0, waits_ = 0, timeouts_ = 0, conversions_ = 0, new_encodes_ = 0, repeat_encodes_ = 0;
    duration_t hold_requested_ {}, hold_actual_ {}, hold_overshoot_max_ {};
    duration_t wait_requested_ {}, wait_actual_ {}, timeout_overshoot_ {}, timeout_overshoot_max_ {};
    duration_t converting_ {}, encoding_ {}, submitting_ {}, completion_waiting_ {};
    unsigned depth_ = 1;
    bool deadline_timer_ = false;
    std::uint64_t pictures_behind_ = 0, pictures_ = 0;
    duration_t picture_latency_ {}, picture_latency_max_ {};
  };

  /** Missing or regressed clocks are unavailable, not a fabricated zero-latency sample. */
  inline std::optional<double> diagnostic_elapsed_ms(
    const diagnostic_timestamp_t &started,
    diagnostic_clock_t::time_point ended
  ) noexcept {
    if (!started || ended < *started) {
      return std::nullopt;
    }
    return std::chrono::duration<double, std::milli>(ended - *started).count();
  }
}  // namespace video::detail
