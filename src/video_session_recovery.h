/**
 * @file src/video_session_recovery.h
 * @brief Nonblocking completion gate for failed encoder-session cleanup.
 */
#pragma once

#include <chrono>
#include <future>
#include <utility>

namespace video::detail {
  /**
   * @brief Bound recovery waiting while a failed encoder session is cleaned up asynchronously.
   *
   * The caller polls from its 20 ms loop and checks session cancellation between polls. Cleanup
   * must own its resources independently and provide a promise/packaged_task future, never a
   * std::async future: destroying this gate must not join an unresolved worker. A timeout only
   * stops recovery waiting; it neither cancels cleanup nor releases resources still in use there.
   */
  class encoder_recovery_t {
  public:
    using clock = std::chrono::steady_clock;

    enum class state_t {
      ready,
      pending,
      timed_out,
      failed
    };

    void start(std::future<void> completion, clock::time_point now = clock::now()) {
      completion_ = std::move(completion);
      deadline_ = now + std::chrono::seconds {10};
      state_ = completion_.valid() ? state_t::pending : state_t::ready;
    }

    state_t poll(clock::time_point now = clock::now()) {
      // Terminal results remain stable until an explicit start() begins another recovery.
      if (state_ != state_t::pending) {
        return state_;
      }
      if (!completion_.valid()) {
        return state_ = state_t::ready;
      }
      // A cleanup already complete when observed is safe, including at the deadline.
      if (completion_.wait_for(std::chrono::seconds {0}) == std::future_status::ready) {
        try {
          completion_.get();
          state_ = state_t::ready;
        } catch (...) {
          state_ = state_t::failed;
        }
      } else if (now >= deadline_) {
        state_ = state_t::timed_out;
      }
      return state_;
    }

  private:
    std::future<void> completion_;
    clock::time_point deadline_ {};
    state_t state_ {state_t::ready};
  };
}  // namespace video::detail
