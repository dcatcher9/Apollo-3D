#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
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
