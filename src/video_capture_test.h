/**
 * @file src/video_capture_test.h
 * @brief Private, device-free access to the real capture worker's lifetime.
 */
#pragma once

#ifdef SUNSHINE_TESTS
  #include "video.h"

  #include <chrono>
  #include <memory>
  #include <optional>

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
