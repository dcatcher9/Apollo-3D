/**
 * @file src/atomic_presentation_test.h
 * @brief Device-free access to the real live-presentation worker and completion handoff.
 */
#pragma once

#ifdef SUNSHINE_TESTS
  #include "process.h"
  #include "stream.h"
  #include "video.h"

  #include <chrono>
  #include <cstdint>
  #include <functional>
  #include <memory>
  #include <optional>

namespace stream::testing {
  struct atomic_presentation_process_hooks_t {
    std::function<bool(int, int, int)> needs_display_change;
    std::function<proc::live_video_mode_result_e(int, int, int, std::uint32_t)> apply;
  };

  /** Own an inert session's actual worker, mail posts and effective publisher; no transport. */
  class atomic_presentation_fixture_t {
  public:
    atomic_presentation_fixture_t(video::effective_video_mode_t initial, atomic_presentation_process_hooks_t hooks);
    ~atomic_presentation_fixture_t();

    atomic_presentation_fixture_t(const atomic_presentation_fixture_t &) = delete;
    atomic_presentation_fixture_t &operator=(const atomic_presentation_fixture_t &) = delete;
    atomic_presentation_fixture_t(atomic_presentation_fixture_t &&) = delete;
    atomic_presentation_fixture_t &operator=(atomic_presentation_fixture_t &&) = delete;

    void submit(video::video_mode_change_t change);
    std::optional<video::video_mode_change_t> pop_encode_request(std::chrono::milliseconds timeout);
    void publish_effective_mode(video::effective_video_mode_t mode);
    void deliver_completion(video::video_mode_applied_t completion);
    std::optional<live_video_mode_ack_t> pop_ack(std::chrono::milliseconds timeout);

    // Gate the existing encoder-wake callback after commit and before the worker waits.
    void gate_next_encode_wake();
    bool wait_encode_wake(std::chrono::milliseconds timeout);
    void release_encode_wake();
    bool gate_timed_out() const;

    void stop();
    void join();
    bool stopping() const;
    std::uint32_t launch_session_id() const;
    bool pacing_diagnostics_rearmed() const;

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace stream::testing
#endif
