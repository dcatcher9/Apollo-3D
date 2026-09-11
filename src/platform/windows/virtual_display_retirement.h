/**
 * @file src/platform/windows/virtual_display_retirement.h
 * @brief Shared bounded retirement proof for local and remote virtual displays.
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <thread>

namespace VDISPLAY {

  enum class display_identity_state_e {
    indeterminate,
    absent,
    present,
  };

  enum class retirement_step_e {
    observe,
    complete,
    blocked,
  };

  struct retirement_timing_t {
    std::chrono::milliseconds poll_interval {50};
    std::chrono::milliseconds settle_interval {100};
  };

  /** The adapter retains its identity, serialization lock and platform cleanup ownership.
   * before_observation may retry safe driver removal, reject a superseded identity, or observe
   * that another owner already completed it. finish must retain failed cleanup for a later slice
   * and latch successful topology work before a subsequent query can become indeterminate.
   */
  struct retirement_callbacks_t {
    std::function<retirement_step_e()> before_observation;
    std::function<display_identity_state_e()> query;
    std::function<bool()> finish;
  };

  struct retirement_clock_t {
    auto now() const {
      return std::chrono::steady_clock::now();
    }

    void sleep_for(std::chrono::milliseconds delay) const {
      std::this_thread::sleep_for(delay);
    }
  };

  /** Publication history survives waiter timeouts; absence evidence does not.
   * Driver acknowledgement is never completion authority. Every slice must observe three
   * consecutive exact absences, respect late-publication quarantine, and recheck after settling.
   */
  struct retirement_record_t {
    bool was_published = false;
    std::chrono::steady_clock::time_point started {};

    template<class Clock>
    [[nodiscard]] bool wait_until(
      std::chrono::steady_clock::time_point deadline,
      retirement_timing_t timing,
      const retirement_callbacks_t &callbacks,
      Clock &clock
    ) const {
      using namespace std::chrono_literals;
      if (!callbacks.query || !callbacks.finish || timing.poll_interval <= 0ms || timing.settle_interval < 0ms) {
        return false;
      }

      unsigned consecutive_absent = 0u;
      // Even a zero-budget waiter gets one observation, but cannot skip the settle interval.
      while (true) {
        const auto step = callbacks.before_observation ?
                            callbacks.before_observation() :
                            retirement_step_e::observe;
        if (step != retirement_step_e::observe) {
          return step == retirement_step_e::complete;
        }
        if (callbacks.query() == display_identity_state_e::absent) {
          ++consecutive_absent;
        } else {
          // Indeterminate queries break the proof exactly as a confirmed reappearance does.
          consecutive_absent = 0u;
        }
        if (consecutive_absent >= 3u && (was_published || clock.now() - started >= 750ms)) {
          if (clock.now() + timing.settle_interval > deadline) {
            return false;
          }
          clock.sleep_for(timing.settle_interval);
          if (callbacks.query() == display_identity_state_e::absent) {
            return callbacks.finish();
          }
          consecutive_absent = 0u;
          if (clock.now() >= deadline) {
            return false;
          }
          continue;
        }

        const auto now = clock.now();
        if (now >= deadline) {
          return false;
        }
        clock.sleep_for(std::min(timing.poll_interval, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)));
      }
    }

    [[nodiscard]] bool wait_until(
      std::chrono::steady_clock::time_point deadline,
      retirement_timing_t timing,
      const retirement_callbacks_t &callbacks
    ) const {
      retirement_clock_t clock;
      return wait_until(deadline, timing, callbacks, clock);
    }
  };

}  // namespace VDISPLAY
