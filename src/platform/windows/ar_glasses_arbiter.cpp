/**
 * @file src/platform/windows/ar_glasses_arbiter.cpp
 * @brief Local and remote virtual-display ownership and the local retirement barrier.
 */
#include "ar_glasses_arbiter.h"

#include "ar_glasses_topology_journal.h"
#include "src/logging.h"
#include "src/utility.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace ar_glasses {
  using namespace std::literals;
  using namespace internal;

  namespace {
    constexpr auto ownership_release_timeout = 10s;

    std::optional<remote_virtual_display_lease_t> remote_session_active_lease;

    struct pending_remote_session_t {
      remote_virtual_display_lease_t lease = 0;
      std::chrono::steady_clock::time_point until {};
      bool handoff_in_progress = false;
      bool setup_in_progress = false;
    };

    std::optional<pending_remote_session_t> remote_session_pending;

    // Serialize the complete retirement operation, including its callbacks. A later waiter must
    // take a fresh record after an earlier waiter has certified cleanup and released ownership.
    std::timed_mutex local_virtual_display_retirement_mutex;

    void expire_remote_pending_locked(std::chrono::steady_clock::time_point now) {
      if (remote_session_pending && !remote_session_pending->handoff_in_progress && !remote_session_pending->setup_in_progress && now >= remote_session_pending->until) {
        remote_session_pending.reset();
      }
    }

    bool pending_remote_lease_matches_locked(remote_virtual_display_lease_t lease) {
      expire_remote_pending_locked(std::chrono::steady_clock::now());
      return remote_session_pending && remote_session_pending->lease == lease;
    }

    void clear_remote_lease_locked(remote_virtual_display_lease_t lease) {
      if (remote_session_active_lease == lease) {
        remote_session_active_lease.reset();
      }
      if (remote_session_pending && remote_session_pending->lease == lease) {
        remote_session_pending.reset();
      }
    }
  }  // namespace

  namespace internal {
    std::mutex ownership_mutex;
    std::condition_variable ownership_changed;
    bool local_session_present = false;
    std::optional<std::stop_source> local_session_construction_stop;

    std::optional<VDISPLAY::session_t> retired_local_virtual_display;

    bool remote_blocks_local_locked(std::chrono::steady_clock::time_point now) {
      expire_remote_pending_locked(now);
      return remote_session_active_lease.has_value() || remote_session_pending.has_value();
    }

    void release_local_virtual_display_claim_impl() {
      std::lock_guard lock(ownership_mutex);
      local_session_present = false;
      local_session_construction_stop.reset();
      ownership_changed.notify_all();
    }

    bool begin_local_virtual_display_retirement(
      VDISPLAY::session_t &&session,
      VDISPLAY::session_retirement_callbacks_t callbacks
    ) {
      std::lock_guard lock(ownership_mutex);
      if (retired_local_virtual_display) {
        if (retired_local_virtual_display->generation() == session.generation()) {
          return true;
        }
        BOOST_LOG(error) << "Refusing to overwrite a different local AR virtual-display retirement record."sv;
        return false;
      }
      session.begin_retirement(std::move(callbacks), {
                                                       .timing = {50ms, 100ms},
                                                       .retry_prepare_within_slice = true,
                                                       .retry_acknowledged_removal_each_slice = true,
                                                     });
      retired_local_virtual_display.emplace(std::move(session));
      ownership_changed.notify_all();
      return true;
    }

    bool wait_for_local_virtual_display_retirement_impl(
      std::chrono::milliseconds timeout
    ) {
      const auto deadline = std::chrono::steady_clock::now() + std::max(timeout, 0ms);
      {
        std::lock_guard lock(ownership_mutex);
        if (!retired_local_virtual_display) {
          return true;
        }
      }
      // Never hold ownership_mutex while waiting: remote launch publishes its reservation under
      // that lock before waiting for local retirement. Include lock contention in this budget.
      std::unique_lock retirement_lock(local_virtual_display_retirement_mutex, std::defer_lock);
      if (timeout <= 0ms ? !retirement_lock.try_lock() : !retirement_lock.try_lock_until(deadline)) {
        return false;
      }
      VDISPLAY::session_t *retiring = nullptr;
      {
        std::lock_guard lock(ownership_mutex);
        if (retired_local_virtual_display) {
          retiring = &*retired_local_virtual_display;
        }
      }
      if (!retiring) {
        return true;
      }

      // The retirement lock pins this move-only owner across callbacks. The shared owner latches
      // successful topology cleanup before its final absence query, including across wait slices.
      if (!retiring->retire_until(deadline)) {
        return false;
      }
      std::lock_guard lock(ownership_mutex);
      retired_local_virtual_display.reset();
      ownership_changed.notify_all();
      return true;
    }
  }  // namespace internal

  bool try_claim_local_virtual_display(const std::stop_source &construction_stop) {
    std::lock_guard lock(ownership_mutex);
    if (construction_stop.stop_requested() || local_session_present || retired_local_virtual_display || remote_blocks_local_locked(std::chrono::steady_clock::now())) {
      return false;
    }
    local_session_present = true;
    local_session_construction_stop = construction_stop;
    ownership_changed.notify_all();
    return true;
  }

  void release_local_virtual_display_claim() {
    release_local_virtual_display_claim_impl();
  }

  bool wait_for_local_virtual_display_retirement(std::chrono::milliseconds timeout) {
    return wait_for_local_virtual_display_retirement_impl(timeout);
  }

  bool remote_virtual_display_starting(
    remote_virtual_display_lease_t lease,
    std::chrono::milliseconds connect_timeout,
    bool setup_in_progress
  ) {
    if (lease == 0) {
      BOOST_LOG(error) << "Refusing a remote virtual-display reservation without a session lease."sv;
      return false;
    }
    std::unique_lock lock(ownership_mutex);
    const auto now = std::chrono::steady_clock::now();
    const auto release_deadline = now + ownership_release_timeout;
    const auto pending_duration = detail::remote_pending_duration(connect_timeout);
    // Cover whichever is longer: the bounded local teardown or one complete connection window.
    // Do not add the two durations; a quick handoff should not reserve the display for an extra
    // teardown timeout after a failed remote launch.
    const auto pending_until = std::max(release_deadline, now + pending_duration);
    if (remote_session_pending && remote_session_pending->lease == lease) {
      remote_session_pending->until = std::max(remote_session_pending->until, pending_until);
    } else {
      // A newer accepted HTTP launch supersedes an older handshake reservation. The session lease
      // prevents teardown from the older process lifecycle from clearing this replacement.
      remote_session_pending = pending_remote_session_t {
        lease,
        pending_until,
        true,
        setup_in_progress,
      };
    }
    remote_session_pending->handoff_in_progress = true;
    remote_session_pending->setup_in_progress =
      remote_session_pending->setup_in_progress || setup_in_progress;
    if (local_session_construction_stop) {
      local_session_construction_stop->request_stop();
    }
    ownership_changed.notify_all();

    if (ownership_changed.wait_until(lock, release_deadline, []() {
          return !local_session_present;
        })) {
      lock.unlock();
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        release_deadline - std::chrono::steady_clock::now()
      );
      const bool retirement_complete = wait_for_local_virtual_display_retirement(
        std::max(remaining, 0ms)
      );
      const bool topology_recovery_complete = retirement_complete &&
                                              recover_connected_saved_topologies_for_remote_handoff();
      lock.lock();
      if (!remote_session_pending || remote_session_pending->lease != lease) {
        BOOST_LOG(info) << "Remote virtual-display reservation was superseded during local handoff."sv;
        return false;
      }
      if (!topology_recovery_complete) {
        clear_remote_lease_locked(lease);
        ownership_changed.notify_all();
        BOOST_LOG(error) << (retirement_complete ? "The retired local AR display topology could not be recovered; refusing remote virtual-display creation."sv : "The retired local AR virtual display remained in Windows topology; refusing overlapping remote display creation."sv);
        return false;
      }
      // Teardown time must not consume the connection reservation. Give the remote path its full
      // configured connection window after local ownership has actually been released.
      remote_session_pending->until = std::chrono::steady_clock::now() + pending_duration;
      remote_session_pending->handoff_in_progress = false;
      return true;
    }

    clear_remote_lease_locked(lease);
    ownership_changed.notify_all();
    BOOST_LOG(error) << "Local AR did not release virtual-display ownership within "sv
                     << ownership_release_timeout.count() << " seconds."sv;
    return false;
  }

  void remote_virtual_display_awaiting_client(
    remote_virtual_display_lease_t lease,
    std::chrono::milliseconds connect_timeout
  ) {
    std::lock_guard lock(ownership_mutex);
    // Creating the virtual display, probing encoders, and running application preparation all
    // happen while proc_t owns its process lock and can legitimately outlive the initial lease.
    // Start a fresh connection window only after that work succeeds, before the process lock is
    // released and local AR is allowed to inspect ownership again.
    if (!pending_remote_lease_matches_locked(lease)) {
      BOOST_LOG(warning) << "Ignoring a stale remote virtual-display reservation renewal."sv;
      return;
    }
    remote_session_pending->until = std::max(
      remote_session_pending->until,
      std::chrono::steady_clock::now() + detail::remote_pending_duration(connect_timeout)
    );
    // Process setup deliberately pins this lease past its ordinary connect deadline. Once all
    // display creation, probing, and application work has completed, the normal bounded RTSP
    // connection window resumes and an abandoned launch can expire again.
    remote_session_pending->setup_in_progress = false;
    ownership_changed.notify_all();
  }

  bool remote_virtual_display_active(remote_virtual_display_lease_t lease) {
    std::lock_guard lock(ownership_mutex);
    if (remote_session_active_lease == lease) {
      return true;
    }
    if (!pending_remote_lease_matches_locked(lease)) {
      return false;
    }
    remote_session_active_lease = lease;
    remote_session_pending.reset();
    ownership_changed.notify_all();
    return true;
  }

  void remote_virtual_display_ended(remote_virtual_display_lease_t lease) {
    if (lease == 0) {
      return;
    }
    std::lock_guard lock(ownership_mutex);
    clear_remote_lease_locked(lease);
    ownership_changed.notify_all();
  }

  bool remote_virtual_display_blocks_local() {
    std::lock_guard lock(ownership_mutex);
    return remote_blocks_local_locked(std::chrono::steady_clock::now());
  }

#ifdef SUNSHINE_TESTS
  detail::local_retirement_transfer_result_t detail::local_retirement_transfer_for_test() {
    local_retirement_transfer_result_t result;
    {
      std::lock_guard lock(ownership_mutex);
      if (retired_local_virtual_display || local_session_present) {
        return result;
      }
    }
    auto now = std::chrono::steady_clock::now();
    bool final_query_indeterminate = false;
    VDISPLAY::session_t session({
      .remove = [&](const GUID &) {
        ++result.remove_calls;
        return true;
      },
      .now = [&]() {
        return now;
      },
      .sleep = [&](std::chrono::milliseconds delay) {
        now += delay;
      },
      .restore_cursor = [](const auto &) {},
    });
    VDISPLAY::creation_result_t binding {
      .display_name = L"test-local-display",
      .device_path = L"test-local-device",
      .identity = SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT {{42, 0}, 7},
    };
    session.adopt({}, std::move(binding));
    const auto generation = session.generation();
    result.transferred = begin_local_virtual_display_retirement(std::move(session), {
                                                                                      .prepare = [&](VDISPLAY::session_t &) {
                                                                                        ++result.prepare_calls;
                                                                                        return true;
                                                                                      },
                                                                                      .query = [&](const VDISPLAY::session_t &) {
                                                                                        if (std::exchange(final_query_indeterminate, false)) {
                                                                                          return VDISPLAY::display_identity_state_e::indeterminate;
                                                                                        }
                                                                                        return VDISPLAY::display_identity_state_e::absent;
                                                                                      },
                                                                                      .finish = [&](VDISPLAY::session_t &) {
                                                                                        ++result.finish_calls;
                                                                                        final_query_indeterminate = true;
                                                                                        return true;
                                                                                      },
                                                                                    });
    if (!result.transferred) {
      return result;
    }
    auto cleanup = util::fail_guard([]() {
      std::lock_guard lock(ownership_mutex);
      retired_local_virtual_display.reset();
      ownership_changed.notify_all();
    });
    result.source_released = !session.owns_display() && !session.prepared() && session.generation() == 0;
    result.first_wait_completed = wait_for_local_virtual_display_retirement_impl(1s);
    {
      std::lock_guard lock(ownership_mutex);
      result.barrier_retained = retired_local_virtual_display && retired_local_virtual_display->generation() == generation;
    }
    result.second_wait_completed = wait_for_local_virtual_display_retirement_impl(1s);
    {
      std::lock_guard lock(ownership_mutex);
      result.barrier_released = !retired_local_virtual_display;
    }
    return result;
  }
#endif
}  // namespace ar_glasses
