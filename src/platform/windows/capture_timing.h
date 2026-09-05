#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace platf::dxgi::detail {

  // Windows.Foundation.TimeSpan uses 100 ns ticks independently of the QPC frequency.
  using wgc_timestamp_t = std::chrono::duration<std::int64_t, std::ratio<1, 10'000'000>>;

  inline std::chrono::nanoseconds wgc_frame_age(
    const std::int64_t current_qpc,
    const std::int64_t qpc_frequency,
    const wgc_timestamp_t frame_time
  ) noexcept {
    if (qpc_frequency <= 0) {
      return {};
    }
    const auto current_time = std::chrono::duration<long double> {
      static_cast<long double>(current_qpc) / qpc_frequency,
    };
    const auto frame_seconds = std::chrono::duration<long double> {frame_time};
    return std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - frame_seconds);
  }

  /** Retain and retry the newest local-presenter source without imposing a minimum-FPS loop.
   *
   * Source conversion and swapchain presentation are separate ownership steps: a busy latency
   * wait leaves conversion pending, while a busy Present leaves only the already-converted
   * backbuffer pending. This prevents both losing the final output and reconverting/re-enqueuing
   * the same pixels merely to retry Present.
   */
  class local_presenter_retry_state_t {
  public:
    constexpr void observe_source() noexcept {
      phase_ = phase_e::conversion_pending;
    }

    /** A conversion triggered by an asynchronous pipeline notification can happen after the
     * source was already presented once. Re-arm the output before Present so swapchain
     * backpressure cannot discard that newly rendered result. */
    constexpr void record_converted() noexcept {
      phase_ = phase_e::presentation_pending;
    }

    [[nodiscard]] constexpr bool should_process(
      const bool has_retained_source,
      const bool depth_pipeline_ready,
      const bool conversion_poll_pending
    ) const noexcept {
      return has_retained_source &&
             (phase_ != phase_e::idle || depth_pipeline_ready ||
              conversion_poll_pending);
    }

    [[nodiscard]] constexpr bool should_convert(
      const bool has_retained_source,
      const bool depth_pipeline_ready,
      const bool conversion_poll_pending
    ) const noexcept {
      return has_retained_source &&
             (phase_ == phase_e::conversion_pending || depth_pipeline_ready ||
              conversion_poll_pending);
    }

    constexpr void record_presented() noexcept {
      phase_ = phase_e::idle;
    }

    [[nodiscard]] constexpr bool presentation_pending() const noexcept {
      return phase_ == phase_e::presentation_pending;
    }

    [[nodiscard]] constexpr bool conversion_pending() const noexcept {
      return phase_ == phase_e::conversion_pending;
    }

  private:
    enum class phase_e {
      idle,
      conversion_pending,
      presentation_pending,
    };

    phase_e phase_ = phase_e::idle;
  };

  struct capture_wait_policy_t {
    bool pending_local_work = false;

    [[nodiscard]] constexpr std::chrono::milliseconds source_timeout() const noexcept {
      return std::chrono::milliseconds {pending_local_work ? 5 : 200};
    }

    [[nodiscard]] constexpr std::chrono::milliseconds idle_backoff() const noexcept {
      // Yield the capture-device lock even during local retries. Remote capture retains its
      // established starvation protection when no local completion needs servicing.
      return std::chrono::milliseconds {pending_local_work ? 1 : 10};
    }

    [[nodiscard]] constexpr std::chrono::steady_clock::duration pacing_sleep(
      const std::chrono::steady_clock::duration requested
    ) const noexcept {
      return pending_local_work ?
               std::min(requested, std::chrono::steady_clock::duration {source_timeout()}) :
               requested;
    }

    [[nodiscard]] constexpr bool retry_after_pacing_timeout() const noexcept {
      // A pending presenter must receive the timeout callback immediately after this short
      // snapshot; entering the ordinary idle snapshot here hid the final result for 210 ms.
      return !pending_local_work;
    }
  };

}  // namespace platf::dxgi::detail
