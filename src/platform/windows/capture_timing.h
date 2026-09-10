#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>

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

  /** Budget one local draw from its DXGI presentation-slot admission, not a free-running grid.
   * Call begin_frame only after acquiring the latency handle. A new admitted frame gets one
   * refresh interval for its work; this is a bounded render deadline, not a predicted vblank.
   * A busy output keeps that deadline even when a newer source replaces its pixels.
   */
  class local_presenter_schedule_t {
  public:
    explicit local_presenter_schedule_t(int refresh_millihz):
        interval_(std::chrono::nanoseconds {1'000'000'000'000LL / std::max(1000, refresh_millihz)}) {
    }

    std::chrono::steady_clock::time_point begin_frame(std::chrono::steady_clock::time_point now) {
      if (!active_target_) {
        // DXGI admission is the pacing authority. An unrelated clock phase can otherwise give
        // an admitted frame only the scraps of the previous frame's budget, repeatedly forcing
        // old packed output even though depth would complete well within one refresh interval.
        active_target_ = now + interval_;
      }
      return *active_target_;
    }

    void record_presented() noexcept {
      active_target_.reset();
    }

  private:
    std::chrono::steady_clock::duration interval_;
    std::optional<std::chrono::steady_clock::time_point> active_target_;
  };

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
      return has_retained_source && phase_ != phase_e::presentation_pending &&
             (phase_ == phase_e::conversion_pending || depth_pipeline_ready ||
              conversion_poll_pending);
    }

    [[nodiscard]] constexpr bool needs_presentation_slot() const noexcept {
      return !slot_acquired_;
    }

    constexpr void record_slot_acquired() noexcept {
      slot_acquired_ = true;
    }

    constexpr void record_presented() noexcept {
      phase_ = phase_e::idle;
      slot_acquired_ = false;
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
    bool slot_acquired_ = false;
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

    [[nodiscard]] constexpr bool rebase_after_pacing_snapshot(
      const std::chrono::steady_clock::duration requested
    ) const noexcept {
      // A successful early source probe is not a complete nominal frame interval. Counting it
      // against the old group would accumulate future pacing debt while the cursor is moving.
      return pacing_sleep(requested) < requested;
    }

    [[nodiscard]] constexpr bool retry_after_pacing_timeout() const noexcept {
      // A pending presenter must receive the timeout callback immediately after this short
      // snapshot; entering the ordinary idle snapshot here hid the final result for 210 ms.
      return !pending_local_work;
    }
  };

}  // namespace platf::dxgi::detail
