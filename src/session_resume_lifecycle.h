/**
 * @file src/session_resume_lifecycle.h
 * @brief Shared retention deadlines for remote and local presentation sessions.
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <optional>

namespace session_lifecycle {
  enum class phase_e {
    idle,
    active,
    retained,
    connecting,
  };

  /** The caller serializes state and supplies its own event delivery/clock observations. */
  class resume_state_t {
  public:
    using clock_t = std::chrono::steady_clock;
    using time_point = clock_t::time_point;
    using duration = clock_t::duration;

    void activate() {
      phase_ = phase_e::active;
      deadline_.reset();
    }

    void clear() {
      phase_ = phase_e::idle;
      deadline_.reset();
    }

    /** Repeated disconnect observations and restoration retries never renew the deadline. */
    void disconnect(time_point now, duration grace) {
      if (phase_ == phase_e::retained) {
        return;
      }
      phase_ = phase_e::retained;
      deadline_ = now + std::max(grace, duration::zero());
    }

    /** Only an accepted connection starts a new bounded handshake window. */
    void await_connection(time_point now, duration timeout) {
      phase_ = phase_e::connecting;
      deadline_ = now + std::max(timeout, duration::zero());
    }

    [[nodiscard]] phase_e phase() const {
      return phase_;
    }

    [[nodiscard]] bool retained() const {
      return phase_ == phase_e::retained;
    }

    [[nodiscard]] bool expired(time_point now) const {
      return deadline_ && now >= *deadline_;
    }

    [[nodiscard]] const std::optional<time_point> &deadline() const {
      return deadline_;
    }

  private:
    phase_e phase_ = phase_e::idle;
    std::optional<time_point> deadline_;
  };
}  // namespace session_lifecycle
