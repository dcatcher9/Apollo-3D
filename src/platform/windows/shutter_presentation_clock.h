#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

namespace platf::dxgi::detail {
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
      unsigned required_samples = 13
    ) noexcept:
        repeats_(repeats),
        qpc_frequency_(qpc_frequency),
        host_age_budget_us_(host_age_budget_us),
        cadence_tolerance_us_(cadence_tolerance_us),
        required_samples_(required_samples) {
      failed_ = repeats < 1 || repeats > 4 || refresh_millihz <= 0 || qpc_frequency <= 0 ||
                host_age_budget_us < 0 || cadence_tolerance_us < 0 || required_samples == 0;
      if (!failed_) {
        refresh_ticks_ = static_cast<long double>(qpc_frequency) * 1000.0L / refresh_millihz;
        // All later integer offsets must fit before conversion/addition. A QPC frequency too
        // coarse to distinguish this eye interval cannot establish scanout continuity either.
        failed_ = refresh_ticks_ * repeats < 1.0L ||
                  refresh_ticks_ * repeats >= static_cast<long double>(std::numeric_limits<std::int64_t>::max());
      }
    }

    shutter_feedback_tracker_status_e observe(
      std::uint32_t requested_id,
      const shutter_feedback_t &feedback,
      std::int64_t query_qpc
    ) noexcept {
      using status_e = shutter_feedback_tracker_status_e;
      if (failed_) {
        return status_e::rejected;
      }
      const auto present_qpc = reconstruct(requested_id, feedback);
      if (!present_qpc || feedback.sync_qpc > query_qpc || !fresh(*present_qpc, query_qpc)) {
        return invalidate();
      }
      const bool continuous = last_present_qpc_ &&
                              requested_id == static_cast<std::uint32_t>(last_present_id_ + 1u) &&
                              static_cast<std::uint32_t>(feedback.refresh_count - last_refresh_) == repeats_ &&
                              cadence_matches(*present_qpc);
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

    bool fresh(std::int64_t present_qpc, std::int64_t now_qpc) const noexcept {
      return now_qpc >= present_qpc &&
             static_cast<long double>(now_qpc - present_qpc) * 1000000.0L / qpc_frequency_ <= host_age_budget_us_;
    }

    bool cadence_matches(std::int64_t present_qpc) const noexcept {
      if (present_qpc <= *last_present_qpc_) {
        return false;
      }
      const auto error_ticks = std::fabs(static_cast<long double>(present_qpc - *last_present_qpc_) - refresh_ticks_ * repeats_);
      return error_ticks * 1000000.0L / qpc_frequency_ <= cadence_tolerance_us_;
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
    unsigned required_samples_;
    long double refresh_ticks_ = 0;
    unsigned streak_ = 0;
    std::uint32_t last_present_id_ = 0;
    std::uint32_t last_refresh_ = 0;
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

  struct shutter_feedback_wait_io_t {
    using clock = std::chrono::steady_clock;
    std::function<clock::time_point()> now;
    std::function<bool()> cancelled;
    std::function<shutter_feedback_t()> query;
    std::function<bool()> flush;
    std::function<bool(clock::time_point)> wait_until;
  };

  // GetFrameStatistics is asynchronous feedback. Waiting another whole vblank whenever it is
  // stale can itself miss the next eye deadline. Publish outstanding DWM feedback once, then
  // service metadata at a bounded subframe cadence without issuing or guessing another present.
  // The production adapter uses DwmFlush and the shared cancellable high-resolution waiter.
  inline shutter_feedback_wait_result_t await_shutter_present(
    const shutter_feedback_wait_io_t &io,
    std::uint32_t requested_present,
    bool first_present,
    std::chrono::milliseconds timeout = std::chrono::milliseconds {250}
  ) {
    using result_e = shutter_feedback_wait_status_e;
    const auto deadline = io.now() + timeout;
    constexpr auto poll_interval = std::chrono::microseconds {200};
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
        if (feedback.present_count == requested_present) {
          return finish(result_e::displayed);
        }
        if (static_cast<std::int32_t>(feedback.present_count - requested_present) > 0) {
          return finish(result_e::unsupported);
        }
      } else if (feedback.status != shutter_feedback_status_e::disjoint || !first_present) {
        return finish(result_e::unsupported);
      }
      if (!flushed) {
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

  // Eye phase advances only when DXGI proves the requested present reached scanout. A late or
  // missing flip invalidates the session instead of guessing an eye from elapsed wall time.
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

    bool observe_displayed(std::uint32_t refresh_count) noexcept {
      if (repeats_ < 1 || repeats_ > 4 || (last_refresh_ && static_cast<std::uint32_t>(refresh_count - *last_refresh_) != repeats_)) {
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
