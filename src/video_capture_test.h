/**
 * @file src/video_capture_test.h
 * @brief Private test access to the production capture worker and encode loop.
 */
#pragma once

#ifdef SUNSHINE_TESTS
  #include "video.h"

  #include <chrono>
  #include <functional>
  #include <memory>
  #include <optional>
  #include <string>
  #include <vector>

namespace video::detail {
  struct native_encode_loop_test_result_t {
    bool session_created = false;
    unsigned pipeline_depth = 0;
    int next_frame_nr = 1;
    bool awaiting_desktop_after_export = false;
  };

  struct native_encode_loop_test_hooks_t {
    using picture_observer_t = std::function<void(
      int, std::optional<std::chrono::steady_clock::time_point>
    )>;

    picture_observer_t before_retrieve;
    picture_observer_t after_submit;
    std::function<void()> repeat_coalesced;
    std::function<void(bool has_retained_source, std::chrono::nanoseconds)> before_image_wait;
  };

  /** Synchronously create a normal native session and run the actual production encode loop.
   * The supplied display must have its original, unarmed make_unarmed_capture_display control
   * block. This bridge arms, publishes and acquires one real capture generation; it does not run
   * a capture worker. The fixture supplies captures/control/packet consumption and stops the loop
   * through its normal mail or image event. Only the unrelated HTTP launch-HDR check is omitted.
   * Joins native and per-device teardown before returning. Optional hooks observe the actual
   * picture lifecycle on its submitting and retrieving threads.
   */
  native_encode_loop_test_result_t run_native_encode_loop_for_test(
    std::shared_ptr<platf::display_t> display,
    safe::mail_t mail,
    img_event_t images,
    config_t config,
    packet_queue_t packets,
    const native_encode_loop_test_hooks_t &hooks = {}
  );

  struct capture_worker_test_access;

  /** Borrowed for one inert worker callback; no platform capture calls have run. */
  struct capture_worker_test_context_t {
    const config_t &config;
    const img_event_t &images;

    bool running() const noexcept;
    bool wait_for_cancel(std::chrono::milliseconds timeout) const noexcept;

  private:
    friend struct capture_worker_test_access;
    capture_worker_test_context_t(
      const config_t &config,
      const img_event_t &images,
      void *cancel_state,
      bool (*running)(const void *) noexcept,
      bool (*wait_for_cancel)(void *, std::chrono::milliseconds) noexcept
    );

    void *cancel_state_;
    bool (*running_)(const void *) noexcept;
    bool (*wait_for_cancel_)(void *, std::chrono::milliseconds) noexcept;
  };

  using capture_worker_test_callback_t = void (*)(void *, const capture_worker_test_context_t &) noexcept;

  /** Physical operations only; run() keeps its original capture/reinitialization policy. */
  struct capture_reinit_test_hooks_t {
    std::function<std::string()> preferred_display_name;
    std::function<void(const std::string &)> set_display_name;
    std::function<void(std::vector<std::string> &, int &, const std::string &)> refresh_displays;
    std::function<std::shared_ptr<platf::display_t>(const std::string &, const config_t &, platf::capture_backend_e, std::weak_ptr<platf::capture_display_retirement_t>)> open_display;
    std::function<void()> adjust_thread_priority;
    std::function<std::uint64_t()> display_change_epoch;
    std::function<bool()> reconcile_topology;
    std::function<void()> waited_for_display_release;
  };

  /** Run the full production capture worker against per-instance inert platform operations. */
  class capture_reinit_test_t {
  public:
    capture_reinit_test_t(config_t config, img_event_t images, capture_reinit_test_hooks_t hooks);
    ~capture_reinit_test_t();

    capture_reinit_test_t(const capture_reinit_test_t &) = delete;
    capture_reinit_test_t &operator=(const capture_reinit_test_t &) = delete;
    capture_reinit_test_t(capture_reinit_test_t &&) = delete;
    capture_reinit_test_t &operator=(capture_reinit_test_t &&) = delete;

    void stop();
    bool running() const;
    bool reinit_pending() const;
    std::shared_ptr<platf::display_t> current_display();
    void set_live_video_mode(std::optional<video_mode_change_t> mode);

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };

  /**
   * Test-only access to the production capture owner. It does not run
   * capture_async(), and its inert callback replaces the platform/display body of that worker.
   * The callback state must outlive this nonmovable facade. stop() is caller-serialized.
   */
  class capture_worker_test_t {
  public:
    capture_worker_test_t(
      config_t config,
      img_event_t images,
      capture_worker_test_callback_t callback,
      void *state,
      capture_worker_test_callback_t before_admission = nullptr
    );
    ~capture_worker_test_t();

    capture_worker_test_t(const capture_worker_test_t &) = delete;
    capture_worker_test_t &operator=(const capture_worker_test_t &) = delete;
    capture_worker_test_t(capture_worker_test_t &&) = delete;
    capture_worker_test_t &operator=(capture_worker_test_t &&) = delete;

    void stop();
    bool running() const;
    void set_live_video_mode(std::optional<video_mode_change_t> mode);
    std::optional<video_mode_change_t> live_video_mode() const;

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace video::detail
#endif
