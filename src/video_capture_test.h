/**
 * @file src/video_capture_test.h
 * @brief Private, device-free access to the real capture worker's lifetime.
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
