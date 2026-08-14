/**
 * @file src/platform/windows/audio_helpers.h
 * @brief Pure helpers for Windows audio capture timing and endpoint-volume fallback.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <span>

namespace platf::audio::detail {
  constexpr std::uint32_t reference_time_to_wait_ms(std::int64_t reference_time) {
    // REFERENCE_TIME is counted in 100 ns units. Round up so a partial millisecond never
    // becomes an early timeout, and retain a valid nonzero Win32 wait for invalid/zero periods.
    if (reference_time <= 0) {
      return 1;
    }

    constexpr std::int64_t units_per_millisecond = 10'000;
    const auto milliseconds = reference_time / units_per_millisecond +
                              (reference_time % units_per_millisecond != 0);
    return static_cast<std::uint32_t>(std::min<std::int64_t>(milliseconds, UINT32_MAX));
  }

  inline float endpoint_volume_gain(float decibels, bool muted) {
    if (muted) {
      return 0.0f;
    }
    if (!std::isfinite(decibels)) {
      return 1.0f;
    }
    return std::pow(10.0f, decibels / 20.0f);
  }

  inline float loopback_software_gain(bool native_post_volume, float decibels, bool muted) {
    return native_post_volume ? 1.0f : endpoint_volume_gain(decibels, muted);
  }

  inline void apply_endpoint_volume(std::span<float> samples, float gain) {
    if (gain == 1.0f) {
      return;
    }
    for (auto &sample : samples) {
      sample *= gain;
    }
  }

  /**
   * @brief Coordinates endpoint-volume refreshes between the COM callback and capture thread.
   *
   * Notifications only set an atomic pending bit. The capture thread consumes that bit when a
   * retry is due, re-arms it after a failed snapshot, and throttles warnings independently. This
   * keeps the COM callback nonblocking without busy-looping or losing transient read failures.
   */
  class endpoint_volume_refresh_state_t {
  public:
    using clock_t = std::chrono::steady_clock;
    static constexpr auto retry_interval = std::chrono::milliseconds {250};
    static constexpr auto warning_interval = std::chrono::seconds {5};

    void notify() noexcept {
      pending_.store(true, std::memory_order_release);
    }

    bool begin_if_due(clock_t::time_point now) noexcept {
      if (now < retry_not_before_) {
        return false;
      }
      return pending_.exchange(false, std::memory_order_acq_rel);
    }

    void failed(clock_t::time_point now) noexcept {
      retry_not_before_ = now + retry_interval;
      notify();
    }

    void succeeded() noexcept {
      retry_not_before_ = clock_t::time_point::min();
    }

    bool should_log_failure(clock_t::time_point now) noexcept {
      if (now < warning_not_before_) {
        return false;
      }
      warning_not_before_ = now + warning_interval;
      return true;
    }

  private:
    std::atomic_bool pending_ {true};
    // Only the capture thread reads or writes the deadlines. The COM callback touches pending_.
    clock_t::time_point retry_not_before_ {clock_t::time_point::min()};
    clock_t::time_point warning_not_before_ {clock_t::time_point::min()};
  };
}  // namespace platf::audio::detail
