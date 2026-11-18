/**
 * @file src/video_capture_generation.h
 * @brief Private admission and revocation authority for a streaming capture generation.
 */
#pragma once

#include <atomic>
#include <memory>
#include <mutex>

namespace platf {
  class capture_display_retirement_t;
  class display_t;
}  // namespace platf

namespace video::detail {
  /** Owned by one capture_session_t; its original display owners retire independently. */
  class capture_generation_handoff_t {
    enum class phase_t {
      opening,
      reopening,
      ready,
      revoked,
    };

  public:
    class display_lease_t;

    class generation_t {
    public:
      const std::shared_ptr<platf::capture_display_retirement_t> &retirement() const noexcept;

    private:
      friend class capture_generation_handoff_t;
      friend class display_lease_t;
      generation_t();

      std::weak_ptr<platf::display_t> display_;
      std::atomic<phase_t> phase_ {phase_t::opening};
      std::shared_ptr<platf::capture_display_retirement_t> retirement_;
    };

    /** Keeps the original control block alive through encode_run's source-return guard. */
    class display_lease_t {
    public:
      display_lease_t() = default;

      explicit operator bool() const noexcept;
      const std::shared_ptr<platf::display_t> &display() const noexcept;
      bool revoked() const noexcept;

    private:
      friend class capture_generation_handoff_t;
      display_lease_t(std::shared_ptr<platf::display_t> display, std::shared_ptr<generation_t> generation);

      std::shared_ptr<platf::display_t> display_;
      std::shared_ptr<generation_t> generation_;
    };

    capture_generation_handoff_t() = default;
    ~capture_generation_handoff_t();
    capture_generation_handoff_t(const capture_generation_handoff_t &) = delete;
    capture_generation_handoff_t &operator=(const capture_generation_handoff_t &) = delete;

    /** Begin before opening. The preceding generation must already be revoked. */
    [[nodiscard]] std::shared_ptr<generation_t> begin();
    [[nodiscard]] bool publish(const std::shared_ptr<generation_t> &generation, const std::shared_ptr<platf::display_t> &display);
    [[nodiscard]] display_lease_t acquire();
    [[nodiscard]] bool revoke(const std::shared_ptr<generation_t> &generation);
    /** Permanently close admission and cancel the current retirement wait; never delete here. */
    void stop();
    bool reinit_pending() const;

  private:
    mutable std::mutex mutex_;
    std::shared_ptr<generation_t> current_;
    bool stopped_ = false;
  };
}  // namespace video::detail
