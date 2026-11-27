/**
 * @file src/stream_platform_lifecycle.cpp
 * @brief Definitions for the process-wide streaming platform lifecycle and resume grace.
 */

// standard includes
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

extern "C" {
  // clang-format off
#include <moonlight-common-c/src/Limelight-internal.h>
  // clang-format on
}

// local includes
#include "config.h"
#include "globals.h"
#include "logging.h"
#include "platform/common.h"
#include "process.h"
#include "rtsp.h"
#include "session_resume_lifecycle.h"
#include "stream.h"
#include "stream_internal.h"

using namespace std::literals;

namespace stream {
  namespace session {
    // Starting a Windows streaming session reapplies NVIDIA profile settings and can take
    // several seconds. Keep that process-wide platform state warm briefly after disconnect,
    // but also bound an accepted launch that never completes its RTSP handshake. The mutex owns
    // the single-active-session invariant and serializes both expiry paths with startup.
    std::mutex platform_lifecycle_mutex;
    bool remote_session_active {};

    namespace {
      std::uint64_t platform_lifecycle_generation {};
      task_pool_util::TaskPool::task_id_t pending_platform_stop {};
      session_lifecycle::resume_state_t platform_resume_state;
      task_pool_util::TaskPool::task_id_t pending_primary_restore {};
      bool platform_streaming_warm {};
      std::optional<std::uint64_t> warm_process_instance;

      void invalidate_pending_platform_stop_locked() {
        ++platform_lifecycle_generation;
        platform_resume_state.clear();
        if (pending_platform_stop) {
          task_pool.cancel(pending_platform_stop);
          pending_platform_stop = nullptr;
        }
        if (pending_primary_restore) {
          task_pool.cancel(pending_primary_restore);
          pending_primary_restore = nullptr;
        }
      }

      void schedule_primary_restore_locked(detail::primary_display_restore_retry_t retry) {
        pending_primary_restore = task_pool.pushDelayed(
          [retry]() {
            std::lock_guard delayed_lock(platform_lifecycle_mutex);
            // A cancelled callback may already be waiting on the lock. Never clear a new
            // generation's task handle, or restore its pending/active session's display.
            if (retry.generation != platform_lifecycle_generation) {
              return;
            }
            pending_primary_restore = nullptr;
            const auto action = retry.next_action(
              platform_lifecycle_generation,
              proc::proc.get_host_session_id(),
              warm_process_instance,
              remote_session_active,
              !rtsp_stream::launch_session_available(),
              platform_streaming_warm && pending_platform_stop
            );
            if (action == detail::primary_display_restore_retry_t::action_e::stop) {
              return;
            }
            if (action == detail::primary_display_restore_retry_t::action_e::wait || !proc::proc.pause_display_for_resume()) {
              // The existing grace timer remains authoritative; retries never extend it.
              schedule_primary_restore_locked(retry);
            }
          },
          1s
        );
      }

      void stop_warm_platform_locked() {
        invalidate_pending_platform_stop_locked();
        warm_process_instance.reset();
        if (!platform_streaming_warm) {
          return;
        }

        platf::streaming_will_stop();
        platform_streaming_warm = false;
      }

      bool cleanup_exited_idle_session_locked() {
        // running() owns command-exit cleanup, including prep-command undo and display retirement.
        // It is safe only after the media workers have joined, or before admitting an idle launch.
        if (remote_session_active || proc::proc.get_status().app_id == 0 || proc::proc.running() > 0) {
          return false;
        }
        rtsp_stream::clear_pending_launch_session();
        stop_warm_platform_locked();
        return true;
      }

      void schedule_platform_stop_locked(std::chrono::steady_clock::duration delay);
      void schedule_platform_stop_check_locked();

      void check_platform_stop_locked(std::uint64_t generation) {
        if (generation != platform_lifecycle_generation || remote_session_active || !platform_resume_state.deadline()) {
          return;
        }
        pending_platform_stop = nullptr;
        if (cleanup_exited_idle_session_locked()) {
          return;
        }
        if (!platform_resume_state.expired(std::chrono::steady_clock::now())) {
          // The retained driver target can reappear after a verified detach. Reuse this grace
          // poll to maintain the pause invariant, but never touch an accepted reconnect or a
          // replacement process. The absolute expiry and its generation remain unchanged.
          const detail::primary_display_restore_retry_t pause_check {generation, warm_process_instance.value_or(0)};
          if (!pending_primary_restore && pause_check.next_action(
                platform_lifecycle_generation,
                proc::proc.get_host_session_id(),
                warm_process_instance,
                remote_session_active,
                !rtsp_stream::launch_session_available(),
                platform_streaming_warm && platform_resume_state.retained()
              ) == detail::primary_display_restore_retry_t::action_e::restore) {
            (void) proc::proc.pause_display_for_resume();
          }
          // Reuse the grace task to observe command exit while disconnected. Polling preserves
          // both the absolute deadline and the generation that owns primary-display retries.
          schedule_platform_stop_check_locked();
          return;
        }

        // A new /launch can replace the retained process just before publishing its pending RTSP
        // handshake. Give that successor one fresh window without terminating it in the gap.
        const auto current_process_instance = proc::proc.get_host_session_id();
        if (current_process_instance != 0 && current_process_instance != warm_process_instance) {
          warm_process_instance = current_process_instance;
          schedule_platform_stop_locked(config::stream.ping_timeout);
          return;
        }

        invalidate_pending_platform_stop_locked();
        BOOST_LOG(info) << (platform_streaming_warm ? "Streaming session resume grace expired; terminating the retained app." : "Streaming launch handshake expired; terminating the unclaimed app.");
        rtsp_stream::clear_pending_launch_session();
        if (proc::proc.running() > 0) {
          proc::proc.terminate();
        }
        warm_process_instance.reset();
        if (platform_streaming_warm) {
          platf::streaming_will_stop();
          platform_streaming_warm = false;
        }
      }

      void schedule_platform_stop_check_locked() {
        const auto generation = platform_lifecycle_generation;
        const auto remaining = *platform_resume_state.deadline() - std::chrono::steady_clock::now();
        const auto delay = std::clamp<std::chrono::steady_clock::duration>(remaining, 0s, 1s);
        pending_platform_stop = task_pool.pushDelayed([generation]() {
          std::lock_guard delayed_lock(platform_lifecycle_mutex);
          check_platform_stop_locked(generation);
        },
                                                      delay);
      }

      void schedule_platform_stop_locked(std::chrono::steady_clock::duration delay) {
        invalidate_pending_platform_stop_locked();
        platform_resume_state.await_connection(std::chrono::steady_clock::now(), delay);
        schedule_platform_stop_check_locked();
      }
    }  // namespace

    bool activate_remote_session_locked(std::uint32_t remote_virtual_display_lease) {
      // Every accepted reconnect owns a new launch-session lease. Adopt it before potentially
      // slow platform startup.
      // Failure is terminal for this RTSP session: streaming without the lease would let local
      // AR reclaim and remove the virtual display while capture is running.
      if (!proc::proc.activate_remote_virtual_display_lease(remote_virtual_display_lease)) {
        BOOST_LOG(error) << "Refusing to start a streaming session without its virtual-display ownership lease."sv;
        return false;
      }

      invalidate_pending_platform_stop_locked();
      platform_resume_state.activate();
      if (platform_streaming_warm) {
        BOOST_LOG(info) << "Reusing warm streaming platform state after a short disconnect."sv;
      } else {
        platf::streaming_will_start();
        platform_streaming_warm = true;
      }

      warm_process_instance.reset();
      return true;
    }

    void retain_or_stop_session_locked() {
      if (remote_session_active) {
        return;
      }
      const auto disconnected_at = std::chrono::steady_clock::now();

      // Restore the physical desktop and detach the virtual source immediately. Retain the
      // app and driver device for reconnect, without leaving their desktop region active.
      BOOST_LOG(info) << "Remote streaming session is inactive; restoring the original display topology."sv;
      const bool primary_restored = proc::proc.pause_display_for_resume();
      if (cleanup_exited_idle_session_locked()) {
        return;
      }
      const auto process_status = proc::proc.get_status();
      if (process_status.app_id == 0) {
        // There is no app/session state worth retaining. Match the historical cleanup path.
        rtsp_stream::clear_pending_launch_session();
        stop_warm_platform_locked();
        return;
      }

      if (!platform_streaming_warm) {
        rtsp_stream::clear_pending_launch_session();
        proc::proc.terminate();
        stop_warm_platform_locked();
        return;
      }

      // Keep the app and remote virtual-display ownership active during the grace. Capture,
      // encoding, transport, and input are already stopped with the session; retaining process
      // ownership prevents another presentation path from claiming the display. The next
      // accepted reconnect reactivates that exact device and reapplies its display policy
      // before capture. A fresh launch can replace this retained ownership immediately.
      const auto host_session_id = proc::proc.get_host_session_id();
      warm_process_instance = host_session_id == 0 ? std::nullopt : std::optional<std::uint64_t> {host_session_id};
      // The current launch reservation may still be waiting for its control connection.
      // Preserve the platform for at least that RTSP handshake window even when reconnect
      // grace is off or configured shorter than ping_timeout.
      const bool validated_launch_pending = !rtsp_stream::launch_session_available();
      if (config::stream.session_resume_grace <= 0ms && !validated_launch_pending) {
        BOOST_LOG(info) << "Session resume grace is disabled; terminating the retained app."sv;
        proc::proc.terminate();
        stop_warm_platform_locked();
        return;
      }
      const auto retention = validated_launch_pending ?
                               std::max(config::stream.session_resume_grace, config::stream.ping_timeout) :
                               config::stream.session_resume_grace;
      if (!platform_resume_state.retained()) {
        invalidate_pending_platform_stop_locked();
        platform_resume_state.disconnect(disconnected_at, retention);
        schedule_platform_stop_check_locked();
      }
      if (!primary_restored && host_session_id != 0 && !pending_primary_restore) {
        schedule_primary_restore_locked({platform_lifecycle_generation, host_session_id});
      }
    }

    namespace {
      bool prepare_new_session_locked() {
        if (remote_session_active || !rtsp_stream::launch_session_available() || platform_resume_state.phase() == session_lifecycle::phase_e::connecting) {
          return false;
        }
        if (proc::proc.get_status().app_id > 0) {
          if (!platform_resume_state.retained()) {
            return false;
          }
          BOOST_LOG(info) << "A fresh connection is replacing the retained streaming session."sv;
          proc::proc.terminate(false, false);
        }
        stop_warm_platform_locked();
        return true;
      }
    }  // namespace

    struct platform_launch_guard_t::impl_t {
      impl_t():
          lock {platform_lifecycle_mutex},
          idle {!remote_session_active} {
        // A command can exit while disconnected, after the teardown-time check. Reap it before
        // HTTP reads the retained identity, without disturbing a live app's existing grace timer.
        cleanup_exited_idle_session_locked();
      }

      std::unique_lock<std::mutex> lock;
      bool idle;
      bool committed {};
    };

    platform_launch_guard_t::platform_launch_guard_t(std::unique_ptr<impl_t> impl):
        _impl {std::move(impl)} {
    }

    platform_launch_guard_t::platform_launch_guard_t(platform_launch_guard_t &&) noexcept = default;
    platform_launch_guard_t &platform_launch_guard_t::operator=(platform_launch_guard_t &&) noexcept = default;
    platform_launch_guard_t::~platform_launch_guard_t() = default;

    bool platform_launch_guard_t::idle() const {
      return _impl && _impl->idle;
    }

    void platform_launch_guard_t::commit() {
      if (!_impl || _impl->committed || !_impl->lock.owns_lock()) {
        return;
      }

      if (_impl->idle && !remote_session_active) {
        const auto current_process_instance = proc::proc.get_host_session_id();
        if (current_process_instance != 0) {
          warm_process_instance = current_process_instance;
        }
        const auto timeout = platform_streaming_warm ?
                               std::max(config::stream.session_resume_grace, config::stream.ping_timeout) :
                               config::stream.ping_timeout;
        schedule_platform_stop_locked(timeout);
        BOOST_LOG(debug) << "Committed streaming launch reservation with bounded lifetime."sv;
      }
      _impl->committed = true;
      _impl->lock.unlock();
    }

    bool platform_launch_guard_t::restore_primary_display() {
      if (!_impl || !_impl->idle || _impl->committed || !_impl->lock.owns_lock() || remote_session_active) {
        return false;
      }
      const bool launch_pending = !rtsp_stream::launch_session_available();
      if (!launch_pending && proc::proc.pause_display_for_resume()) {
        return true;
      }
      const auto host_session_id = proc::proc.get_host_session_id();
      const detail::primary_display_restore_retry_t retry {platform_lifecycle_generation, host_session_id};
      if (!pending_primary_restore && retry.next_action(platform_lifecycle_generation, host_session_id, warm_process_instance, remote_session_active, launch_pending, platform_streaming_warm && pending_platform_stop) != detail::primary_display_restore_retry_t::action_e::stop) {
        schedule_primary_restore_locked(retry);
      }
      return false;
    }

    bool platform_launch_guard_t::prepare_new_session() {
      return _impl && _impl->idle && !_impl->committed && _impl->lock.owns_lock() &&
             prepare_new_session_locked();
    }

    local_session_admission_e prepare_local_ar_session(bool replace_retained) {
      // Remote launch holds this lock while asking the local controller to stop. The controller
      // must remain able to service that request instead of waiting behind the remote caller.
      std::unique_lock lock(platform_lifecycle_mutex, std::try_to_lock);
      if (!lock.owns_lock()) {
        return local_session_admission_e::retry;
      }
      cleanup_exited_idle_session_locked();
      if (!replace_retained && platform_resume_state.retained() && proc::proc.get_status().app_id > 0) {
        return local_session_admission_e::remote_busy;
      }
      return prepare_new_session_locked() ? local_session_admission_e::ready : local_session_admission_e::remote_busy;
    }

    platform_launch_guard_t guard_platform_launch() {
      return platform_launch_guard_t {std::make_unique<platform_launch_guard_t::impl_t>()};
    }

    void flush_platform_state() {
      std::lock_guard lock(platform_lifecycle_mutex);
      stop_warm_platform_locked();
    }

#ifdef SUNSHINE_TESTS
    bool platform_lifecycle_lock_available_for_test() {
      bool available = false;
      std::thread probe([&]() {
        std::unique_lock lock {platform_lifecycle_mutex, std::try_to_lock};
        available = lock.owns_lock();
      });
      probe.join();
      return available;
    }

    bool claim_active_slot_for_test() {
      std::lock_guard lock(platform_lifecycle_mutex);
      if (remote_session_active) {
        return false;
      }
      remote_session_active = true;
      return true;
    }

    void release_active_slot_for_test() {
      std::lock_guard lock(platform_lifecycle_mutex);
      remote_session_active = false;
    }

    void retain_or_stop_session_for_test(bool platform_warm) {
      std::lock_guard lock(platform_lifecycle_mutex);
      // Model a previously started platform without acquiring any Windows scheduling/input state.
      platform_streaming_warm = platform_warm;
      retain_or_stop_session_locked();
    }

    std::uint64_t platform_lifecycle_generation_for_test() {
      std::lock_guard lock(platform_lifecycle_mutex);
      return platform_lifecycle_generation;
    }

    std::optional<std::chrono::steady_clock::time_point> platform_stop_deadline_for_test() {
      std::lock_guard lock(platform_lifecycle_mutex);
      return platform_resume_state.deadline();
    }

    void check_platform_stop_for_test(std::optional<std::uint64_t> dispatched_generation) {
      std::lock_guard lock(platform_lifecycle_mutex);
      const auto generation = dispatched_generation.value_or(platform_lifecycle_generation);
      if (generation == platform_lifecycle_generation && pending_platform_stop) {
        task_pool.cancel(pending_platform_stop);
      }
      check_platform_stop_locked(generation);
    }
#endif
  }  // namespace session
}  // namespace stream
