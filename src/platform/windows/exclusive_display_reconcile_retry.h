/**
 * @file src/platform/windows/exclusive_display_reconcile_retry.h
 * @brief Race-safe bounded retry state for exclusive display reconciliation.
 */
#pragma once

#include "primary_display.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace platf::primary_display::detail {
  enum class exclusive_reconcile_followup_e {
    finish,
    retry_immediate,
    retry_delayed,
    exhausted,
  };

  constexpr exclusive_reconcile_followup_e reconcile_followup(
    exclusive_reconcile_result_e result,
    bool dirty,
    bool can_retry
  ) {
    if (result == exclusive_reconcile_result_e::retry_active) {
      return can_retry ? exclusive_reconcile_followup_e::retry_delayed :
                         exclusive_reconcile_followup_e::exhausted;
    }
    if (result == exclusive_reconcile_result_e::settled && dirty) {
      return can_retry ? exclusive_reconcile_followup_e::retry_immediate :
                         exclusive_reconcile_followup_e::exhausted;
    }
    if (result == exclusive_reconcile_result_e::pending_recovery && dirty) {
      return can_retry ? exclusive_reconcile_followup_e::retry_delayed :
                         exclusive_reconcile_followup_e::exhausted;
    }
    return exclusive_reconcile_followup_e::finish;
  }

  /** Controller methods are serialized by the session monitor's reconciliation mutex. */
  class exclusive_reconcile_retry_t {
  public:
    using clock_t = std::chrono::steady_clock;

    struct ticket_t {
      std::uint64_t scheduler_generation = 0;
      std::uint64_t session_generation = 0;

      bool operator==(const ticket_t &) const = default;
    };

    explicit exclusive_reconcile_retry_t(
      std::size_t max_attempts,
      std::chrono::milliseconds retry_window
    ):
        max_attempts_(max_attempts),
        retry_window_(retry_window) {}

    /** Record a display notification in the controller's serialized event sequence. */
    void notify() {
      ++event_generation_;
    }

    /** Start a bounded cycle for an idle controller.
     * Notifications received while a worker or timer owns the cycle join that work instead of
     * resetting its retry budget.
     */
    std::optional<std::uint64_t> start() {
      if (phase_ != phase_e::idle || event_generation_ == handled_event_generation_) {
        return std::nullopt;
      }
      session_generation_.reset();
      attempts_ = 0;
      ++scheduler_generation_;
      phase_ = phase_e::worker;
      return scheduler_generation_;
    }

    std::uint64_t scheduler_generation() const {
      return scheduler_generation_;
    }

    std::uint64_t event_generation() const {
      return event_generation_;
    }

    bool begin_attempt(
      std::uint64_t scheduler_generation,
      std::uint64_t session_generation,
      clock_t::time_point now = clock_t::now()
    ) {
      if (phase_ != phase_e::worker || scheduler_generation != scheduler_generation_) {
        return false;
      }
      if (!session_generation_) {
        session_generation_ = session_generation;
        deadline_ = now + retry_window_;
      } else if (*session_generation_ != session_generation) {
        return false;
      }
      if (attempts_ >= max_attempts_ || (attempts_ > 0 && now >= *deadline_)) {
        handled_event_generation_ = event_generation_;
        reset_cycle();
        return false;
      }
      ++attempts_;
      return true;
    }

    bool can_retry(
      std::uint64_t scheduler_generation,
      std::uint64_t session_generation,
      clock_t::time_point now = clock_t::now()
    ) const {
      return phase_ == phase_e::worker && scheduler_generation == scheduler_generation_ && session_generation_ &&
             *session_generation_ == session_generation && deadline_ && now < *deadline_ &&
             attempts_ < max_attempts_;
    }

    std::optional<ticket_t> schedule(
      std::uint64_t scheduler_generation,
      std::uint64_t session_generation,
      std::uint64_t handled_event_generation,
      clock_t::time_point now = clock_t::now()
    ) {
      if (!can_retry(scheduler_generation, session_generation, now)) {
        return std::nullopt;
      }
      handled_event_generation_ = handled_event_generation;
      phase_ = phase_e::delayed;
      return ticket_t {scheduler_generation_, session_generation};
    }

    /** Claim one delayed callback for its follow-up worker.
     * The phase changes directly from delayed to worker, so notifications in the dispatch gap
     * cannot start another cycle or reset this cycle's budget.
     */
    std::optional<std::uint64_t> dispatch(
      ticket_t ticket,
      std::uint64_t current_session_generation,
      clock_t::time_point now = clock_t::now()
    ) {
      if (phase_ != phase_e::delayed || ticket.scheduler_generation != scheduler_generation_ || !session_generation_ || ticket.session_generation != *session_generation_) {
        return std::nullopt;
      }
      if (current_session_generation != ticket.session_generation) {
        reset_cycle();
        return std::nullopt;
      }
      if (!deadline_ || now >= *deadline_) {
        handled_event_generation_ = event_generation_;
        reset_cycle();
        return std::nullopt;
      }
      phase_ = phase_e::worker;
      return scheduler_generation_;
    }

    void acknowledge(std::uint64_t scheduler_generation, std::uint64_t event_generation) {
      if (phase_ == phase_e::worker && scheduler_generation == scheduler_generation_) {
        handled_event_generation_ = event_generation;
      }
    }

    void finish(std::uint64_t scheduler_generation, std::uint64_t event_generation) {
      if (phase_ != phase_e::stopped && scheduler_generation == scheduler_generation_) {
        handled_event_generation_ = event_generation;
        reset_cycle();
      }
    }

    void abandon(std::uint64_t scheduler_generation) {
      if (phase_ != phase_e::stopped && scheduler_generation == scheduler_generation_) {
        reset_cycle();
      }
    }

    void stop() {
      if (phase_ == phase_e::stopped) {
        return;
      }
      phase_ = phase_e::stopped;
      session_generation_.reset();
      deadline_.reset();
      attempts_ = 0;
      ++scheduler_generation_;
    }

    std::size_t attempts() const {
      return attempts_;
    }

    std::uint64_t handled_event_generation() const {
      return handled_event_generation_;
    }

  private:
    enum class phase_e {
      idle,
      worker,
      delayed,
      stopped,
    };

    void reset_cycle() {
      phase_ = phase_e::idle;
      session_generation_.reset();
      deadline_.reset();
      attempts_ = 0;
      ++scheduler_generation_;
    }

    const std::size_t max_attempts_;
    const std::chrono::milliseconds retry_window_;
    std::uint64_t scheduler_generation_ = 0;
    std::uint64_t event_generation_ = 0;
    std::uint64_t handled_event_generation_ = 0;
    std::optional<std::uint64_t> session_generation_;
    std::optional<clock_t::time_point> deadline_;
    std::size_t attempts_ = 0;
    phase_e phase_ = phase_e::idle;
  };
}  // namespace platf::primary_display::detail
