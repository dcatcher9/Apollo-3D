#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

namespace platf::dxgi::detail {
  // A confirmed image can be held beyond its requested minimum dwell. Finite explicit eye
  // commands do not toggle during that hold. Bound the gap and retain raw QPC/counter agreement;
  // this never authorizes a shorter eye interval or infers an eye from refresh parity.
  constexpr bool held_refresh_delta_valid(unsigned repeats, std::uint32_t delta) noexcept {
    return repeats >= 1 && repeats <= 4 && delta >= repeats && delta <= 4 * repeats;
  }

  // The same bound for an eye that follows one the presenter deliberately showed for fewer
  // refreshes, to repay a late image: that earlier eye lasted previous_presents refreshes.
  constexpr bool eye_refresh_delta_valid(unsigned repeats, unsigned previous_presents, std::uint32_t delta) noexcept {
    return repeats >= 1 && repeats <= 4 && previous_presents >= 1 && previous_presents <= repeats &&
           delta >= previous_presents && delta <= 4 * repeats;
  }

  // A lost clock is reacquired with the emitter quiet, as often as it is lost: each recovery
  // holds the picture for a fraction of a second, and ending the session instead would turn a
  // disturbed few seconds, such as a program starting, into losing 3D altogether. What ends
  // the session is a clock that cannot be reacquired. Windows can withhold or blank
  // presentation statistics for a few seconds while it rearranges composition, and no eye
  // command is sent during acquisition, so the wait for one is generous.
  inline constexpr auto shutter_acquisition_timeout = std::chrono::seconds {6};

  enum class shutter_feedback_status_e {
    available,
    disjoint,
    unsupported,
  };

  struct shutter_feedback_t {
    shutter_feedback_status_e status = shutter_feedback_status_e::unsupported;
    std::uint32_t present_count = 0;
    std::uint32_t refresh_count = 0;
    std::uint32_t sync_refresh_count = 0;
    std::int64_t sync_qpc = 0;
    std::int32_t result_code = 0;
  };

  enum class shutter_feedback_tracker_status_e {
    acquiring,
    ready,
    accepted,
    rejected,
  };

  // Cold DWM feedback may stall or change counter domains before settling. Acquire a sequence
  // while shutters remain closed; after explicit arming, every sample must preserve the exact
  // submitted identity, physical cadence and caller-selected remaining emission window.
  class shutter_feedback_tracker_t {
  public:
    shutter_feedback_tracker_t(
      unsigned repeats,
      int refresh_millihz,
      std::int64_t qpc_frequency,
      std::int64_t host_age_budget_us,
      std::int64_t cadence_tolerance_us,
      unsigned required_samples = 13,
      // Once armed, the exact refresh-count difference already proves which refresh displayed
      // the image. The timestamp Windows attaches to that refresh can be several hundred
      // microseconds late under load, so the armed check may be wider than cold acquisition.
      // A negative value keeps the acquisition tolerance.
      std::int64_t armed_cadence_tolerance_us = -1,
      // Windows can publish a present slightly before the refresh timestamp it carries. Once
      // armed, a boundary no further ahead of the query clock than this is a current image.
      std::int64_t armed_future_tolerance_us = 0,
      // Presents each eye submits. The eye is identified by its first present, so the next
      // eye's identity is exactly this many present IDs later.
      unsigned present_stride = 1
    ) noexcept:
        repeats_(repeats),
        qpc_frequency_(qpc_frequency),
        host_age_budget_us_(host_age_budget_us),
        cadence_tolerance_us_(cadence_tolerance_us),
        armed_cadence_tolerance_us_(armed_cadence_tolerance_us < 0 ? cadence_tolerance_us : armed_cadence_tolerance_us),
        required_samples_(required_samples),
        present_stride_(present_stride) {
      failed_ = repeats < 1 || repeats > 4 || refresh_millihz <= 0 || qpc_frequency <= 0 ||
                host_age_budget_us < 0 || cadence_tolerance_us < 0 || required_samples == 0 ||
                armed_future_tolerance_us < 0 || armed_future_tolerance_us > 1000 ||
                present_stride < 1 || present_stride > 4;
      if (!failed_) {
        armed_future_ticks_ = static_cast<std::int64_t>(static_cast<long double>(armed_future_tolerance_us) * qpc_frequency / 1000000.0L);
        refresh_ticks_ = static_cast<long double>(qpc_frequency) * 1000.0L / refresh_millihz;
        // All later integer offsets must fit before conversion/addition. A QPC frequency too
        // coarse to distinguish this eye interval cannot establish scanout continuity either.
        failed_ = refresh_ticks_ * repeats < 1.0L ||
                  refresh_ticks_ * repeats >= static_cast<long double>(std::numeric_limits<std::int64_t>::max());
      }
    }

    // previous_eye_presents: how many presents the eye before this one submitted, when the
    // presenter shortened it to repay a late image. Zero means the usual present_stride.
    shutter_feedback_tracker_status_e observe(
      std::uint32_t requested_id,
      const shutter_feedback_t &feedback,
      std::int64_t query_qpc,
      unsigned previous_eye_presents = 0
    ) noexcept {
      using status_e = shutter_feedback_tracker_status_e;
      if (failed_) {
        return status_e::rejected;
      }
      // Only an armed sequence may contain a shortened eye; acquisition measures whole eyes.
      if (previous_eye_presents > present_stride_ || (previous_eye_presents != 0 && previous_eye_presents != present_stride_ && !armed_)) {
        return invalidate();
      }
      const unsigned stride = previous_eye_presents == 0 ? present_stride_ : previous_eye_presents;
      const unsigned shortest_delta = previous_eye_presents == 0 ? repeats_ : std::min(repeats_, previous_eye_presents);
      const auto present_qpc = reconstruct(requested_id, feedback);
      if (!present_qpc || feedback.sync_qpc - future_ticks() > query_qpc || !fresh(*present_qpc, query_qpc)) {
        return invalidate();
      }
      const auto refresh_delta = static_cast<std::uint32_t>(feedback.refresh_count - last_refresh_);
      // Once armed, the time between two boundaries says how many refreshes passed. Windows'
      // refresh counter was measured to advance by one fewer than the elapsed refresh periods
      // when an image is shown late, so the counter may trail the time by a little but never
      // lead it. Cold acquisition still requires counter and time to agree exactly.
      const auto timed_delta = armed_ && last_present_qpc_ ? timed_refreshes(*present_qpc) : std::nullopt;
      const bool continuous = last_present_qpc_ &&
                              requested_id == static_cast<std::uint32_t>(last_present_id_ + stride) &&
                              (armed_ ? timed_delta && eye_refresh_delta_valid(repeats_, shortest_delta, *timed_delta) &&
                                          refresh_delta >= 1 && refresh_delta <= *timed_delta && *timed_delta - refresh_delta <= 2 :
                                        refresh_delta == repeats_ && cadence_matches(*present_qpc, refresh_delta));
      interval_refreshes_ = continuous ? (armed_ ? *timed_delta : refresh_delta) : 0;
      if (last_present_qpc_ && !continuous) {
        if (armed_) {
          return invalidate();
        }
        // This is a valid boundary in a new sequence, not evidence that the old eye continued.
        streak_ = 0;
      }
      last_present_id_ = requested_id;
      last_refresh_ = feedback.refresh_count;
      last_present_qpc_ = *present_qpc;
      if (streak_ < required_samples_) {
        ++streak_;
      }
      return armed_ ? status_e::accepted : streak_ >= required_samples_ ? status_e::ready : status_e::acquiring;
    }

    bool arm() noexcept {
      if (failed_ || armed_ || streak_ < required_samples_) {
        return false;
      }
      armed_ = true;
      return true;
    }

    // Check again immediately before the emitter write, using the last accepted PRESENT
    // boundary rather than a newer sync sample. This does not consume another eye/sample.
    bool fresh_for_emit(std::int64_t now_qpc) noexcept {
      if (!armed_ || failed_) {
        return false;
      }
      if (!last_present_qpc_ || !fresh(*last_present_qpc_, now_qpc)) {
        invalidate();
        return false;
      }
      return true;
    }

    [[nodiscard]] unsigned streak() const noexcept { return streak_; }
    // Refreshes between the last accepted boundary and the one before it, by elapsed time once
    // armed. Zero when the last observation did not continue a sequence.
    [[nodiscard]] std::uint32_t interval_refreshes() const noexcept { return interval_refreshes_; }
    [[nodiscard]] std::optional<std::int64_t> last_present_qpc() const noexcept { return last_present_qpc_; }
    [[nodiscard]] bool armed() const noexcept { return armed_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }

  private:
    std::optional<std::int64_t> reconstruct(std::uint32_t requested_id, const shutter_feedback_t &feedback) const noexcept {
      if (feedback.status != shutter_feedback_status_e::available || feedback.present_count != requested_id || feedback.sync_qpc <= 0) {
        return std::nullopt;
      }
      const auto offset = static_cast<std::int32_t>(feedback.refresh_count - feedback.sync_refresh_count);
      if (offset < -static_cast<int>(repeats_) || offset > static_cast<int>(repeats_)) {
        return std::nullopt;
      }
      const auto rounded_shift = std::round(static_cast<long double>(offset) * refresh_ticks_);
      // Use floating point only for a bounded rational interval, then check signed integer
      // addition explicitly. Neither a corrupt timestamp nor an extreme QPC frequency may wrap.
      if (rounded_shift >= static_cast<long double>(std::numeric_limits<std::int64_t>::max()) ||
          rounded_shift <= static_cast<long double>(std::numeric_limits<std::int64_t>::min())) {
        return std::nullopt;
      }
      const auto shift = static_cast<std::int64_t>(rounded_shift);
      if ((shift > 0 && feedback.sync_qpc > std::numeric_limits<std::int64_t>::max() - shift) ||
          (shift < 0 && feedback.sync_qpc <= -shift)) {
        return std::nullopt;
      }
      return feedback.sync_qpc + shift;
    }

    // Zero until armed: cold acquisition never accepts a boundary ahead of the query clock.
    std::int64_t future_ticks() const noexcept {
      return armed_ ? armed_future_ticks_ : 0;
    }

    bool fresh(std::int64_t present_qpc, std::int64_t now_qpc) const noexcept {
      if (now_qpc < present_qpc) {
        // Both values are positive, so their signed difference cannot overflow.
        return present_qpc - now_qpc <= future_ticks();
      }
      return static_cast<long double>(now_qpc - present_qpc) * 1000000.0L / qpc_frequency_ <= host_age_budget_us_;
    }

    // The whole number of refresh periods since the last boundary, if the elapsed time is one.
    std::optional<std::uint32_t> timed_refreshes(std::int64_t present_qpc) const noexcept {
      if (present_qpc <= *last_present_qpc_) {
        return std::nullopt;
      }
      const auto elapsed = static_cast<long double>(present_qpc - *last_present_qpc_);
      const auto periods = std::round(elapsed / refresh_ticks_);
      if (periods < 1.0L || periods > 4.0L * repeats_) {
        return std::nullopt;
      }
      const auto error_ticks = std::fabs(elapsed - refresh_ticks_ * periods);
      if (error_ticks * 1000000.0L / qpc_frequency_ > armed_cadence_tolerance_us_) {
        return std::nullopt;
      }
      return static_cast<std::uint32_t>(periods);
    }

    bool cadence_matches(std::int64_t present_qpc, std::uint32_t refresh_delta) const noexcept {
      if (present_qpc <= *last_present_qpc_) {
        return false;
      }
      const auto error_ticks = std::fabs(static_cast<long double>(present_qpc - *last_present_qpc_) - refresh_ticks_ * refresh_delta);
      return error_ticks * 1000000.0L / qpc_frequency_ <= (armed_ ? armed_cadence_tolerance_us_ : cadence_tolerance_us_);
    }

    shutter_feedback_tracker_status_e invalidate() noexcept {
      if (armed_) {
        failed_ = true;
        return shutter_feedback_tracker_status_e::rejected;
      }
      streak_ = 0;
      last_present_qpc_.reset();
      return shutter_feedback_tracker_status_e::acquiring;
    }

    unsigned repeats_;
    std::int64_t qpc_frequency_;
    std::int64_t host_age_budget_us_;
    std::int64_t cadence_tolerance_us_;
    std::int64_t armed_cadence_tolerance_us_;
    std::int64_t armed_future_ticks_ = 0;
    unsigned required_samples_;
    unsigned present_stride_ = 1;
    long double refresh_ticks_ = 0;
    unsigned streak_ = 0;
    std::uint32_t last_present_id_ = 0;
    std::uint32_t last_refresh_ = 0;
    std::uint32_t interval_refreshes_ = 0;
    std::optional<std::int64_t> last_present_qpc_;
    bool armed_ = false;
    bool failed_ = false;
  };

  enum class shutter_feedback_wait_status_e {
    displayed,
    cancelled,
    timeout,
    unsupported,
    flush_failed,
    wait_failed,
  };

  struct shutter_feedback_wait_result_t {
    shutter_feedback_wait_status_e status;
    shutter_feedback_t feedback {};
    std::uint32_t queries = 0;
    std::uint32_t flushes = 0;
    std::uint32_t waits = 0;
  };

  inline bool shutter_feedback_is_timing_interruption(const shutter_feedback_wait_result_t &result) noexcept {
    return result.status == shutter_feedback_wait_status_e::timeout ||
           (result.status == shutter_feedback_wait_status_e::unsupported &&
             result.feedback.status == shutter_feedback_status_e::disjoint);
  }

  // An eye is several presents of the same image and is identified by the first. If the
  // statistics already report a later present of that same eye, the eye did reach the screen
  // but its boundary sample was not read in time. That is a missed measurement, which quiet
  // reacquisition can repair; a present beyond the eye is an unknown image and is not.
  inline bool shutter_feedback_superseded_within_eye(
    const shutter_feedback_wait_result_t &result,
    std::uint32_t requested_present,
    unsigned present_stride
  ) noexcept {
    if (result.status != shutter_feedback_wait_status_e::unsupported ||
        result.feedback.status != shutter_feedback_status_e::available || present_stride < 2 || present_stride > 4) {
      return false;
    }
    const auto ahead = static_cast<std::uint32_t>(result.feedback.present_count - requested_present);
    return ahead >= 1 && ahead < present_stride;
  }

  struct shutter_feedback_wait_io_t {
    using clock = std::chrono::steady_clock;
    std::function<clock::time_point()> now;
    std::function<bool()> cancelled;
    std::function<shutter_feedback_t()> query;
    std::function<bool()> flush;
    std::function<bool(clock::time_point)> wait_until;
  };

  // GetFrameStatistics is asynchronous feedback. Waiting another whole vblank whenever it is
  // stale can itself miss the next eye deadline. DwmFlush can block past the current shutter
  // window, so permit it only during unsignalled acquisition. Armed frames poll raw metadata
  // with the shared cancellable high-resolution waiter, without guessing another present or
  // falling back to a blocking flush. Flush-dependent systems must time out and recover quietly.
  inline shutter_feedback_wait_result_t await_shutter_present(
    const shutter_feedback_wait_io_t &io,
    std::uint32_t requested_present,
    bool acquiring,
    std::chrono::milliseconds timeout = std::chrono::milliseconds {250}
  ) {
    using result_e = shutter_feedback_wait_status_e;
    const auto deadline = io.now() + timeout;
    constexpr auto poll_interval = std::chrono::microseconds {200};
    // Windows can publish the requested present a moment before it fills in which refresh
    // displayed it: the count matches while the refresh counters and timestamp are all zero.
    // Poll a bounded number of times for the completed sample; if it stays empty, return it
    // as before so the caller's own validation rejects it.
    constexpr unsigned maximum_empty_samples = 8;
    unsigned empty_samples = 0;
    bool flushed = false;
    shutter_feedback_wait_result_t result {result_e::unsupported};
    const auto finish = [&](result_e status) {
      result.status = status;
      return result;
    };
    while (true) {
      if (io.cancelled()) {
        return finish(result_e::cancelled);
      }
      if (io.now() >= deadline) {
        return finish(result_e::timeout);
      }
      const auto feedback = io.query();
      result.feedback = feedback;
      ++result.queries;
      if (io.cancelled()) {
        return finish(result_e::cancelled);
      }
      if (io.now() >= deadline) {
        return finish(result_e::timeout);
      }
      if (feedback.status == shutter_feedback_status_e::available) {
        const bool empty = feedback.refresh_count == 0 && feedback.sync_refresh_count == 0 && feedback.sync_qpc == 0;
        if (feedback.present_count == requested_present && (!empty || ++empty_samples > maximum_empty_samples)) {
          return finish(result_e::displayed);
        }
        if (static_cast<std::int32_t>(feedback.present_count - requested_present) > 0) {
          return finish(result_e::unsupported);
        }
      } else if (feedback.status != shutter_feedback_status_e::disjoint || !acquiring) {
        return finish(result_e::unsupported);
      }
      if (acquiring && !flushed) {
        flushed = true;
        ++result.flushes;
        const bool published = io.flush();
        if (io.cancelled()) {
          return finish(result_e::cancelled);
        }
        if (!published) {
          return finish(result_e::flush_failed);
        }
        continue;
      }
      const auto next_poll = std::min(deadline, io.now() + poll_interval);
      ++result.waits;
      if (!io.wait_until(next_poll)) {
        return finish(io.cancelled() ? result_e::cancelled : result_e::wait_failed);
      }
    }
  }

  // Eye phase advances once for an explicitly confirmed submitted image. A bounded extra hold
  // does not replay a command or advance phase by refresh parity; the feedback tracker must
  // independently prove exact identity, raw counter/QPC agreement and current-frame freshness.
  class shutter_presentation_clock_t {
  public:
    explicit shutter_presentation_clock_t(unsigned eye_repeats):
        repeats_(eye_repeats) {}

    [[nodiscard]] bool left_eye() const noexcept {
      return left_;
    }

    [[nodiscard]] bool starts_pair() const noexcept {
      return left_;
    }

    // previous_eye_presents as for the feedback tracker: zero means a whole eye.
    bool observe_displayed(std::uint32_t refresh_count, unsigned previous_eye_presents = 0) noexcept {
      if (repeats_ < 1 || repeats_ > 4 || previous_eye_presents > repeats_) {
        return false;
      }
      const unsigned shortest_delta = previous_eye_presents == 0 ? repeats_ : previous_eye_presents;
      if (last_refresh_ && !eye_refresh_delta_valid(repeats_, shortest_delta, static_cast<std::uint32_t>(refresh_count - *last_refresh_))) {
        return false;
      }
      last_refresh_ = refresh_count;
      left_ = !left_;
      return true;
    }

  private:
    unsigned repeats_;
    bool left_ = true;
    std::optional<std::uint32_t> last_refresh_;
  };
}  // namespace platf::dxgi::detail
