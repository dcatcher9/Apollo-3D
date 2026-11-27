/**
 * @file src/virtual_display_lifecycle.cpp
 * @brief The streaming session's virtual display: creation, binding, topology, mode and removal.
 *
 * process.cpp owns the public proc_t entry points and process_state_mutex. The proc_t members
 * here are the virtual-display steps those entry points take. Removal belongs to one retirement
 * owner, kept in this file, whose state outlives the proc_t that handed a display to it.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

#ifndef BOOST_PROCESS_VERSION
  #define BOOST_PROCESS_VERSION 1
#endif
// standard includes
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <thread>

// local includes
#include "config.h"
#include "display_device.h"
#include "logging.h"
#include "process.h"
#include "virtual_display_lifecycle.h"

#include "platform/windows/misc.h"
#include "platform/windows/primary_display.h"
#include "platform/windows/utils.h"

namespace proc {
  using namespace std::literals;

  namespace {
    // Topology removal outlives the proc_t configuration object that initiated it. Keep the
    // display owner here so refresh() cannot discard it while replacing proc.
    std::recursive_mutex retired_virtual_display_mutex;
    std::condition_variable_any retired_virtual_display_changed;
    std::optional<VDISPLAY::session_t> retired_virtual_display_session;
    bool retired_virtual_display_remove_ready = false;
    bool retired_virtual_display_deactivate_desktop = false;
    bool retired_virtual_display_requires_device_absence = false;
    bool retired_virtual_display_was_active_desktop_path = false;
    bool retired_virtual_display_final_teardown = false;
    bool retired_virtual_display_restart_explorer_after_removal = false;
    bool retired_virtual_display_explorer_repair_debt = false;
    bool retired_virtual_display_explorer_repair_in_progress = false;
    std::uint64_t retired_virtual_display_generation = 0;
    bool retired_virtual_display_cleanup_scheduled = false;
    bool retired_virtual_display_workers_stopping = false;
    std::shared_ptr<VDISPLAY::desktop_detach_context_t>
      retired_virtual_display_detach_context;
    std::jthread retired_virtual_display_cleanup_worker;
    std::jthread retired_virtual_display_explorer_repair_worker;

    bool retiredVirtualDisplayRemovalAllowed(
      bool desktop_deactivation_requested,
      bool detach_confirmed_ready
    ) {
      return !desktop_deactivation_requested || detach_confirmed_ready;
    }

    bool virtualDisplayRetirementHandoffMarksSession(
      bool has_bound_display,
      bool retiring_bound_display
    ) {
      // Failed recreation candidates are retired only after the old binding was cleared. They
      // still own the session GUID, so an unbound candidate needs the same teardown tombstone as a
      // bound display. A candidate must never mask a different live binding.
      return !has_bound_display || retiring_bound_display;
    }

    bool explorerRepairAllowedForRetirement(
      bool final_teardown,
      bool restart_enabled,
      bool active_desktop_removal_debt
    ) {
      // The Explorer workaround is deliberately narrower than driver cleanup. Only a final
      // session teardown consumes repair debt from a formerly active desktop path. The caller
      // evaluates this only after settled exact-identity absence, which is stronger evidence than
      // either the Remove IOCTL's return value or the creation-time GDI-name snapshot.
      return final_teardown &&
             restart_enabled &&
             active_desktop_removal_debt;
    }

    void scheduleExplorerRepairLocked() {
      if (retired_virtual_display_workers_stopping || retired_virtual_display_explorer_repair_in_progress) {
        return;
      }
      retired_virtual_display_explorer_repair_in_progress = true;
      try {
        retired_virtual_display_explorer_repair_worker =
          std::jthread([](std::stop_token) {
            (void) platf::restart_active_user_explorer();
            std::lock_guard retirement_lock(retired_virtual_display_mutex);
            retired_virtual_display_explorer_repair_in_progress = false;
            retired_virtual_display_changed.notify_all();
          });
      } catch (const std::system_error &thread_error) {
        retired_virtual_display_explorer_repair_in_progress = false;
        BOOST_LOG(error) << "Could not start the Explorer taskbar-repair worker: "
                         << thread_error.what();
      }
    }
  }  // namespace

  void stopRetiredVirtualDisplayCleanupWorker() {
    std::jthread cleanup_worker;
    {
      std::lock_guard retirement_lock(retired_virtual_display_mutex);
      retired_virtual_display_workers_stopping = true;
      retired_virtual_display_cleanup_worker.request_stop();
      retired_virtual_display_changed.notify_all();
      cleanup_worker = std::move(retired_virtual_display_cleanup_worker);
    }
    if (cleanup_worker.joinable()) {
      // Never join while holding retired_virtual_display_mutex: a bounded worker slice may be
      // completing its final topology observation.
      cleanup_worker.join();
    }

    std::jthread explorer_repair_worker;
    {
      std::lock_guard retirement_lock(retired_virtual_display_mutex);
      retired_virtual_display_explorer_repair_worker.request_stop();
      explorer_repair_worker =
        std::move(retired_virtual_display_explorer_repair_worker);
    }
    if (explorer_repair_worker.joinable()) {
      explorer_repair_worker.join();
    }
  }

  void mark_retirement_final_teardown() {
    std::lock_guard retirement_lock(retired_virtual_display_mutex);
    if (retired_virtual_display_session) {
      retired_virtual_display_final_teardown = true;
      retired_virtual_display_restart_explorer_after_removal =
        retired_virtual_display_restart_explorer_after_removal ||
        config::sunshine.virtual_display_restart_explorer;
    } else {
      const bool restart_explorer = explorerRepairAllowedForRetirement(
        true,
        config::sunshine.virtual_display_restart_explorer,
        retired_virtual_display_explorer_repair_debt
      );
      retired_virtual_display_explorer_repair_debt = false;
      if (restart_explorer) {
        scheduleExplorerRepairLocked();
      }
    }
  }

#ifdef SUNSHINE_TESTS
  bool retiredVirtualDisplayRemovalAllowedForTest(
    bool desktopDeactivationRequested,
    bool detachConfirmedReady
  ) {
    return retiredVirtualDisplayRemovalAllowed(
      desktopDeactivationRequested,
      detachConfirmedReady
    );
  }

  bool virtualDisplayRetirementHandoffMarksSessionForTest(
    bool hasBoundDisplay,
    bool retiringBoundDisplay
  ) {
    return virtualDisplayRetirementHandoffMarksSession(
      hasBoundDisplay,
      retiringBoundDisplay
    );
  }

  bool explorerRepairAllowedForRetirementForTest(
    bool finalTeardown,
    bool restartEnabled,
    bool activeDesktopRemovalDebt
  ) {
    return explorerRepairAllowedForRetirement(
      finalTeardown,
      restartEnabled,
      activeDesktopRemovalDebt
    );
  }
#endif

  VDISPLAY::session_t proc_t::create_retained_virtual_display(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t fps,
    const GUID &guid,
    const std::optional<LUID> &render_adapter
  ) {
    VDISPLAY::session_t owner;
#ifdef SUNSHINE_TESTS
    if (_test_edges.replaces_display_io()) {
      owner = VDISPLAY::session_t {_test_edges.display_io};
    }
#endif
    if (_display_context) {
      VDISPLAY::display_spec_t spec;
      spec.client_uid = _display_context->unique_id;
      spec.client_name = _display_context->device_name;
      spec.guid = guid;
      spec.width = width;
      spec.height = height;
      spec.refresh_millihz = fps;
      spec.render_adapter_luid = render_adapter;
      spec.exclusive = _virtual_display_only;
      owner.acquire(std::move(spec));
    }
    return owner;
  }

  bool proc_t::prepare_retired_virtual_display_for_removal(
    std::chrono::milliseconds timeout
  ) {
    std::lock_guard retirement_lock(retired_virtual_display_mutex);
    if (!retired_virtual_display_session) {
      return true;
    }
    // Restore the original displays before detach captures its survivor topology. The retirement
    // owner also resolves an unpublished Add's path without relying on a stale DISPLAY name.
    auto &owner = *retired_virtual_display_session;
    if (!owner.restore()) {
      BOOST_LOG(warning) << "Deferring virtual-display removal until the original display configuration is restored."sv;
      return false;
    }
    if (retired_virtual_display_remove_ready) {
      return true;
    }
    if (!retired_virtual_display_deactivate_desktop || !owner.binding().identity) {
      retired_virtual_display_remove_ready = true;
      return true;
    }

    const auto detach = VDISPLAY::deactivateVirtualDisplay(
      *owner.binding().identity,
      owner.binding().device_path,
      timeout,
      retired_virtual_display_detach_context
    );
    switch (detach.state) {
      case VDISPLAY::desktop_detach_state_e::detached:
        if (detach.active_mode_grace_expired) {
          BOOST_LOG(warning) << "Advanced Color user preferences were restored, but a "
                                "policy-controlled active mode did not reconverge before its "
                                "bounded observation grace expired."sv;
        }
        retired_virtual_display_requires_device_absence = true;
        retired_virtual_display_was_active_desktop_path = true;
        retired_virtual_display_remove_ready = true;
        BOOST_LOG(info) << "Virtual display detached from the Windows desktop before driver removal."sv;
        return true;
      case VDISPLAY::desktop_detach_state_e::already_inactive:
        if (detach.active_mode_grace_expired) {
          BOOST_LOG(warning) << "Advanced Color user preferences were restored, but a "
                                "policy-controlled active mode did not reconverge before its "
                                "bounded observation grace expired."sv;
        }
        retired_virtual_display_requires_device_absence = true;
        retired_virtual_display_remove_ready = true;
        BOOST_LOG(debug) << "Virtual display was already inactive before driver removal."sv;
        return true;
      case VDISPLAY::desktop_detach_state_e::skipped_only_active:
        // Removing the driver monitor is the only way to leave the headless topology. It was still
        // an active Explorer desktop, so final teardown carries repair debt even though there was
        // no surviving output on which to perform the safer two-phase detach.
        retired_virtual_display_requires_device_absence = false;
        retired_virtual_display_was_active_desktop_path = true;
        retired_virtual_display_remove_ready = true;
        BOOST_LOG(info) << "Keeping the virtual display active until driver removal because it is the only desktop output."sv;
        return true;
      case VDISPLAY::desktop_detach_state_e::color_restore_timeout:
        // The saved context owns the pre-detach color contract. Removing the driver monitor here
        // would discard the only restoration owner, even when path/shell settling already
        // completed, so keep observing/reconciling until the contract matches or the survivor is
        // confirmed gone.
        BOOST_LOG(error) << "Advanced Color restoration timed out; retaining the detached "
                            "virtual-display monitor and retrying the saved survivor contract."sv;
        return false;
      case VDISPLAY::desktop_detach_state_e::settle_timeout:
        BOOST_LOG(warning) << "The virtual-display detach did not produce a confirmed, settled "
                              "inactive path before the deadline; retrying without removing the driver monitor."sv;
        return false;
      case VDISPLAY::desktop_detach_state_e::ambiguous_identity:
        BOOST_LOG(warning) << "Could not identify exactly one active virtual-display path; "
                              "retrying instead of hot-removing an unverified active path."sv;
        return false;
      case VDISPLAY::desktop_detach_state_e::topology_query_failed:
        BOOST_LOG(warning) << "Could not prepare a verified desktop topology before virtual-display "
                              "removal; see the detach snapshot diagnostic. Retrying instead of "
                              "hot-removing an unverified active path."sv;
        return false;
      case VDISPLAY::desktop_detach_state_e::validation_failed:
        BOOST_LOG(warning) << "Windows rejected validation of the temporary virtual-display detach "
                              "(status "
                           << detach.windows_status
                           << "); retrying without driver removal."sv;
        return false;
      case VDISPLAY::desktop_detach_state_e::apply_failed:
        BOOST_LOG(warning) << "Windows could not detach the virtual display from the desktop "
                              "(status "
                           << detach.windows_status
                           << "); retrying without driver removal."sv;
        return false;
    }
    return false;
  }

  void proc_t::schedule_retired_virtual_display_cleanup() {
    std::lock_guard retirement_lock(retired_virtual_display_mutex);
    if (retired_virtual_display_workers_stopping || !retired_virtual_display_session) {
      return;
    }

    retired_virtual_display_cleanup_scheduled = true;
    try {
      if (!retired_virtual_display_cleanup_worker.joinable()) {
        retired_virtual_display_cleanup_worker = std::jthread([](std::stop_token stop) {
          std::unique_lock lock(retired_virtual_display_mutex);
          while (!stop.stop_requested()) {
            retired_virtual_display_changed.wait(
              lock,
              stop,
              []() {
                return retired_virtual_display_cleanup_scheduled;
              }
            );
            if (stop.stop_requested()) {
              break;
            }

            const auto generation = retired_virtual_display_generation;
            const bool state_changed = retired_virtual_display_changed.wait_for(
              lock,
              stop,
              1s,
              [generation]() {
                return generation != retired_virtual_display_generation ||
                       !retired_virtual_display_session ||
                       !retired_virtual_display_cleanup_scheduled;
              }
            );
            if (stop.stop_requested()) {
              break;
            }
            if (state_changed) {
              continue;
            }

            lock.unlock();
            // This dedicated worker never occupies Apollo's single shared task-pool thread and
            // never takes process_state_mutex. Each ownership slice is bounded; unresolved
            // transitions pause before retrying so foreground topology work is delayed by at most
            // one slice.
            const bool completed = proc_t::wait_for_retired_virtual_display(900ms, false);
            lock.lock();
            if (generation == retired_virtual_display_generation && (completed || !retired_virtual_display_session)) {
              retired_virtual_display_cleanup_scheduled = false;
            }
          }
        });
      }
    } catch (const std::system_error &thread_error) {
      retired_virtual_display_cleanup_scheduled = false;
      BOOST_LOG(error) << "Could not start autonomous virtual-display retirement cleanup: "
                       << thread_error.what();
      return;
    }
    retired_virtual_display_changed.notify_all();
  }

  bool proc_t::retire_virtual_display(
    VDISPLAY::session_t &owner,
    std::chrono::milliseconds timeout,
    bool deactivate_desktop,
    bool final_teardown,
    bool restart_explorer_after_removal
  ) {
#ifdef SUNSHINE_TESTS
    if (_test_edges.topology) {
      const bool completed = _test_edges.topology(
        display_topology_test_operation_e::retire,
        final_teardown
      );
      if (completed) {
        // The hook proves cleanup of the fake binding; consume its ownership as production does.
        (void) VDISPLAY::session_t(std::move(owner));
      }
      return completed;
    }
#endif
    if (!owner.owns_display() && !owner.prepared()) {
      return true;
    }

    // Never overwrite an unresolved owner. GUID reuse waits for its exact absence proof and
    // any Explorer repair that followed it.
    if (!wait_for_retired_virtual_display(3s)) {
      BOOST_LOG(error) << "A previous virtual-display retirement is still unresolved."sv;
      return false;
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::max(timeout, std::chrono::milliseconds::zero());
    const auto remaining_time = [&]() {
      const auto now = std::chrono::steady_clock::now();
      return now >= deadline ? 0ms :
                               std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    };

    {
      std::lock_guard retirement_lock(retired_virtual_display_mutex);
      if (retired_virtual_display_session) {
        return false;
      }
      const bool marks_session = virtualDisplayRetirementHandoffMarksSession(
        _display_session.owns_display(),
        &owner == &_display_session
      );
      retired_virtual_display_session.emplace(std::move(owner));
      retired_virtual_display_remove_ready = false;
      retired_virtual_display_deactivate_desktop = deactivate_desktop;
      retired_virtual_display_requires_device_absence = false;
      retired_virtual_display_was_active_desktop_path = false;
      retired_virtual_display_final_teardown = final_teardown;
      retired_virtual_display_restart_explorer_after_removal = restart_explorer_after_removal;
      retired_virtual_display_detach_context.reset();
      ++retired_virtual_display_generation;
      retired_virtual_display_cleanup_scheduled = false;
      if (marks_session) {
        _virtual_display_retirement_handed_off = true;
      }

      // The adapter prepares detach within each waiter's budget. The owner retains the resulting
      // preparation latch, Remove acknowledgement/retry time, and exact-absence proof.
      const VDISPLAY::session_retirement_callbacks_t callbacks {
        .prepare = [](VDISPLAY::session_t &) {
          return retired_virtual_display_remove_ready;
        },
        .before_remove = [](VDISPLAY::session_t &retired) {
          if (!retiredVirtualDisplayRemovalAllowed(
                retired_virtual_display_deactivate_desktop,
                retired_virtual_display_remove_ready
              )) {
            return false;
          }
          if (retired_virtual_display_deactivate_desktop && retired_virtual_display_requires_device_absence) {
            // Windows can reattach the exact target between bounded worker slices. A sole
            // remaining desktop intentionally bypasses this two-phase detach check.
            const auto &binding = retired.binding();
            const auto active_state = VDISPLAY::queryVirtualDisplayIdentity(*binding.identity, binding.device_path).state;
            if (!VDISPLAY::isDriverRemovalSafeAfterDesktopDetach(active_state)) {
              retired_virtual_display_remove_ready = false;
              retired_virtual_display_requires_device_absence = false;
              retired_virtual_display_detach_context.reset();
              BOOST_LOG(warning) << "Virtual-display active-path absence could not be revalidated "
                                    "immediately before driver removal; restarting safe detach."sv;
              return false;
            }
          }
          return true;
        },
        .query = [](const VDISPLAY::session_t &retired) {
          const auto &binding = retired.binding();
          if (!binding.identity) {
            return VDISPLAY::display_identity_state_e::absent;
          }
          if (retired_virtual_display_requires_device_absence) {
            // Desktop detach removes the active path before the driver monitor disappears.
            return VDISPLAY::queryVirtualDisplayRetirementState(*binding.identity, binding.device_path);
          }
          return VDISPLAY::queryVirtualDisplayIdentity(*binding.identity, binding.device_path).state;
        },
      };
      retired_virtual_display_session->begin_retirement(
        callbacks,
        {{deactivate_desktop ? 250ms : 50ms, 250ms}, 250ms}
      );
    }

    // Foreground teardown retains its longer initial detach budget; autonomous slices remain
    // bounded by wait_for_retired_virtual_display()'s smaller preparation budget.
    if (!prepare_retired_virtual_display_for_removal(std::min(remaining_time(), 5000ms))) {
      schedule_retired_virtual_display_cleanup();
      return false;
    }
    const bool completed = wait_for_retired_virtual_display(remaining_time(), false);
    if (!completed) {
      schedule_retired_virtual_display_cleanup();
    }
    return completed;
  }

  void proc_t::clear_virtual_display_binding() {
    stop_hdr_worker();
    _hdr_worker_state.reset();
    _saved_virtual_display_mode.reset();  // A replacement monitor has its own saved default.
    // Resource ownership has already moved to retirement. Never discard a live driver identity.
    if (!_display_session.owns_display() && !_display_session.prepared()) {
      _display_session = VDISPLAY::session_t {};
    }
    set_display_name_locked({});
  }

  bool proc_t::refresh_virtual_display_binding() {
    if (!_display_session.owns_display()) {
      return false;
    }
    for (int attempt = 0; attempt < 20; ++attempt) {
      if (_display_session.refresh() && !_display_session.binding().display_name.empty()) {
        set_display_name_locked(platf::to_utf8(_display_session.binding().display_name));
#ifdef SUNSHINE_TESTS
        if (_test_edges.topology) {
          return true;  // Tests publish the binding but never map a test name to an output.
        }
#endif
        config::video.output_name = display_device::map_display_name(display_name);
        return !config::video.output_name.empty();
      }
#ifdef SUNSHINE_TESTS
      if (_test_edges.topology) {
        break;
      }
#endif
      std::this_thread::sleep_for(50ms);
    }
    _display_session.invalidate_gdi();
    set_display_name_locked({});
    return false;
  }

  bool proc_t::prepare_retained_display_for_resume() {
    if (!_display_session.pause_requested()) {
      return refresh_virtual_display_binding();
    }
    _display_session.set_exclusive(_virtual_display_only);
    // An authorized resume can reactivate the virtual-only topology even when later verification
    // fails. Its rollback must wake the restored physical desktop once; maintenance polls never
    // enter this path and cannot rearm the notification after a transient query failure.
    _pause_wake_attempted = false;
    if (!_display_session.begin_resume()) {
      BOOST_LOG(warning) << "Could not reactivate the retained virtual display topology; the remote session remains paused."
                         << " owns_display=" << _display_session.owns_display()
                         << " retiring=" << _display_session.retiring()
                         << " pause_requested=" << _display_session.pause_requested()
                         << " paused=" << _display_session.paused()
                         << " has_retained=" << _display_session.has_retained();
      return false;
    }
    if (!refresh_virtual_display_binding()) {
      BOOST_LOG(warning) << "The retained virtual display topology was reactivated, but its active display binding could not be resolved; the remote session remains paused."sv;
      return false;
    }
    // Keep the recovery snapshot until mode, HDR and final topology promotion also succeed.
    BOOST_LOG(info) << "Retained virtual display reactivated before remote resume: " << display_name;
    return true;
  }

  bool proc_t::prepare_retained_display_after_settle() {
    // Restoring the physical desktop makes Windows re-detect its monitors, and for a few seconds
    // it can keep changing the active outputs by itself: a display query fails, or the detached
    // virtual display is active again in a different mode. A resume that arrives in that window
    // fails although nothing is wrong with the retained session. Return to the paused desktop,
    // let the topology settle, and try again before reporting the failure to the client.
    constexpr auto settle_window = 6s;
    constexpr auto settle_delay = 500ms;
    // Only a resume that started from a verified paused desktop retries. A pause that is itself
    // still unverified keeps its single attempt and its existing maintenance retry.
    const bool started_paused = _display_session.pause_requested() && _display_session.paused();
    if (prepare_retained_display_for_resume()) {
      return true;
    }
    if (!started_paused) {
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + settle_window;
    for (int retry_count = 1;; ++retry_count) {
      if (!_display_session.owns_display() || _display_session.retiring() || !_display_session.has_retained()) {
        return false;
      }
      // The next attempt repeats this pause when it could not be verified yet.
      pause_display_for_resume();
#ifdef SUNSHINE_TESTS
      const bool wait_for_settle = !_test_edges.topology;
#else
      const bool wait_for_settle = true;
#endif
      if (wait_for_settle) {
        std::this_thread::sleep_for(settle_delay);
      }
      if (prepare_retained_display_for_resume()) {
        BOOST_LOG(info) << "Retained virtual display reactivated after "sv
                        << retry_count << " topology-settle retry attempt(s)."sv;
        return true;
      }
#ifdef SUNSHINE_TESTS
      if (_test_edges.topology && retry_count >= 3) {
        return false;  // Tests do not wait, so they bound the retries by count.
      }
#endif
      if (std::chrono::steady_clock::now() + settle_delay >= deadline) {
        return false;
      }
    }
  }

  bool proc_t::promote_virtual_display(
    bool enable_hdr,
    std::chrono::milliseconds topology_retry_window
  ) {
    const bool had_hdr_worker = _hdr_worker.joinable();
    stop_hdr_worker();
    _hdr_worker_state.reset();
    const auto promote_once = [&]() {
#ifdef SUNSHINE_TESTS
      if (_test_edges.topology) {
        return _test_edges.topology(
          display_topology_test_operation_e::promote,
          _virtual_display_only
        );
      }
#endif
      _display_session.set_exclusive(_virtual_display_only);
      return _display_session.promote();
    };

    bool promoted = promote_once();
    int retry_count = 0;
    if (!promoted && topology_retry_window > 0ms) {
      const auto deadline = std::chrono::steady_clock::now() + topology_retry_window;
      BOOST_LOG(warning) << "Windows display topology was still settling after restore; retrying retained virtual-display promotion."sv;
      while (!promoted) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
          break;
        }
        std::this_thread::sleep_for(std::min(50ms, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)));
        ++retry_count;
        promoted = promote_once();
      }
      if (promoted) {
        BOOST_LOG(info) << "Retained virtual-display promotion succeeded after "sv
                        << retry_count << " topology-settle retry attempt(s)."sv;
      }
    }
    if (!promoted || !refresh_virtual_display_binding()) {
      return false;
    }
    if (had_hdr_worker && !request_hdr_state(enable_hdr, 6s)) {
      return false;
    }
    BOOST_LOG(info) << (_virtual_display_only ? "Virtual-display-only policy verified with the virtual display primary: " : "Virtual display verified as the Windows primary display: ") << display_name;
    return true;
  }

  void proc_t::adopt_virtual_display(
    VDISPLAY::session_t created_display,
    bool enable_hdr
  ) {
    _display_session = std::move(created_display);
    _virtual_display_retirement_handed_off = false;
    _virtual_display = true;
    set_display_name_locked(platf::to_utf8(_display_session.binding().display_name));
#ifdef SUNSHINE_TESTS
    if (_test_edges.map_display_name) {
      config::video.output_name = _test_edges.map_display_name(display_name);
    } else
#endif
      config::video.output_name = display_device::map_display_name(display_name);
    start_hdr_worker(enable_hdr);
  }

  std::optional<DEVMODEW> proc_t::query_virtual_display_mode() {
    // DISPLAYn is reusable. Resolve it from this session's exact driver identity
    // before reading a mode; a successful query of a stale name proves nothing.
    if (!_display_session.refresh()) {
      return std::nullopt;
    }
#ifdef SUNSHINE_TESTS
    if (_test_edges.query_mode) {
      return _test_edges.query_mode(_display_session.binding().display_name);
    }
#endif
    DEVMODEW mode {};
    return VDISPLAY::getDeviceSettings(_display_session.binding().display_name.c_str(), mode) ?
             std::optional<DEVMODEW> {mode} : std::nullopt;
  }

  LONG proc_t::configure_virtual_display_mode(const wchar_t *name, int width, int height, int fps_millihz, bool probe) {
#ifdef SUNSHINE_TESTS
    if (_test_edges.change_mode) {
      return _test_edges.change_mode(name, width, height, fps_millihz, probe);
    }
#endif
    if (probe) {
      return VDISPLAY::testDisplaySettings(name, width, height, fps_millihz);
    }
    // The mode set queries the active topology and re-applies it. Hold the display transaction
    // so the display-change worker's reconcile cannot interleave with that pair, as local AR
    // does through run_local_exclusive_update().
    return platf::primary_display::run_serialized_mode_change([&]() -> long {
      return VDISPLAY::changeDisplaySettings(name, width, height, fps_millihz, false);
    });
  }

  bool proc_t::display_settings_unavailable() {
#ifdef SUNSHINE_TESTS
    if (_test_edges.settings_unavailable) {
      return _test_edges.settings_unavailable();
    }
#endif
    return is_changing_settings_going_to_fail();
  }

  bool proc_t::virtual_display_baseline_topology_active() {
    const auto &device_path = _display_session.binding().device_path;
#ifdef SUNSHINE_TESTS
    if (_test_edges.baseline_active) {
      return _test_edges.baseline_active(device_path);
    }
#endif
    return !device_path.empty() && platf::primary_display::baseline_topology_active(device_path);
  }

  bool proc_t::save_virtual_display_session_mode(int width, int height, int fps_millihz) {
    const session_display_mode_t requested {width, height, fps_millihz};
    // Save only what Windows verifiably presents for this session, in Windows' own whole-Hz form.
    const auto active = query_virtual_display_mode();
    const auto record = active && live_display_mode_matches(*active, width, height, fps_millihz) ?
                          VDISPLAY::sessionModeRecord(*active) :
                          std::nullopt;
    if (!record) {
      BOOST_LOG(warning) << "The virtual display is not verifiably at its session mode "sv << width << 'x' << height
                         << " @ "sv << (static_cast<double>(fps_millihz) / 1000.0)
                         << " Hz; its default mode was not saved."sv;
      return false;
    }
    if (_saved_virtual_display_mode == requested) {
      return true;  // Already this monitor's saved default; rewriting it adds nothing.
    }
    // Windows does not document whether this registry save also records the active topology.
    // Write it only while that topology is still the user's own: never after promotion makes the
    // virtual display primary or disables physical outputs.
    if (!virtual_display_baseline_topology_active()) {
      BOOST_LOG(warning) << "The virtual display's session mode "sv << record->width << 'x' << record->height
                         << " @ "sv << record->refresh_hz
                         << " Hz was not saved as its default mode: a session display topology is active."sv;
      return false;
    }

    const auto &name = _display_session.binding().display_name;
    LONG status;
#ifdef SUNSHINE_TESTS
    if (_test_edges.record_mode) {
      status = _test_edges.record_mode(name, *record);
    } else
#endif
    {
      status = VDISPLAY::saveSessionModeRecord(name.c_str(), *record);
    }
    if (status != DISP_CHANGE_SUCCESSFUL) {
      _saved_virtual_display_mode.reset();  // A rejected write may leave any saved value.
      BOOST_LOG(warning) << "Could not save the virtual display's session mode "sv << record->width << 'x' << record->height
                         << " @ "sv << record->refresh_hz << " Hz as its default mode (status "sv << status
                         << "); the session keeps its temporary mode."sv;
      return false;
    }

    _saved_virtual_display_mode = requested;
    BOOST_LOG(info) << "Saved the virtual display's session mode "sv << record->width << 'x' << record->height
                    << " @ "sv << record->refresh_hz << " Hz as its default mode."sv;
    return true;
  }

  bool proc_t::drain_retired_virtual_display_for_shutdown(
    std::chrono::milliseconds timeout
  ) {
    const auto deadline =
      std::chrono::steady_clock::now() +
      std::max(timeout, std::chrono::milliseconds::zero());
    while (true) {
      const auto now = std::chrono::steady_clock::now();
      const auto remaining = now >= deadline ?
                               0ms :
                               std::chrono::duration_cast<std::chrono::milliseconds>(
                                 deadline - now
                               );
      if (wait_for_retired_virtual_display(remaining, true)) {
        return true;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }

      // prepare_retired_virtual_display_for_removal() intentionally returns promptly on an
      // indeterminate topology query or a settle/color timeout. During normal runtime the
      // autonomous worker retries that state. Orderly deinit must provide the same retry
      // ownership instead of interpreting the first transient result as a reason to stop it.
      std::unique_lock retirement_lock(retired_virtual_display_mutex);
      const auto generation = retired_virtual_display_generation;
      retired_virtual_display_changed.wait_for(
        retirement_lock,
        std::min(100ms, remaining),
        [generation]() {
          return generation != retired_virtual_display_generation ||
                 (!retired_virtual_display_session &&
                  !retired_virtual_display_explorer_repair_in_progress);
        }
      );
    }
  }

  bool proc_t::wait_for_retired_virtual_display(
    std::chrono::milliseconds timeout,
    bool wait_for_explorer_repair
  ) {
    std::unique_lock retirement_lock(retired_virtual_display_mutex);
    const auto deadline =
      std::chrono::steady_clock::now() +
      std::max(timeout, std::chrono::milliseconds::zero());
    if (retired_virtual_display_session) {
      if (!retired_virtual_display_remove_ready) {
        const auto now = std::chrono::steady_clock::now();
        const auto prepare_budget = now >= deadline ?
                                      0ms :
                                      std::min(
                                        1500ms,
                                        std::chrono::duration_cast<std::chrono::milliseconds>(
                                          deadline - now
                                        )
                                      );
        if (!prepare_retired_virtual_display_for_removal(prepare_budget)) {
          return false;
        }
      }

      if (!retired_virtual_display_session->retire_until(deadline)) {
        return false;
      }

      // Run adapter-only completion after the owner proves final absence. In particular, a
      // failed final query must not consume a later final-teardown Explorer request.
      retired_virtual_display_explorer_repair_debt =
        retired_virtual_display_explorer_repair_debt ||
        retired_virtual_display_was_active_desktop_path;
      const bool restart_explorer = explorerRepairAllowedForRetirement(
        retired_virtual_display_final_teardown,
        retired_virtual_display_restart_explorer_after_removal,
        retired_virtual_display_explorer_repair_debt
      );
      if (retired_virtual_display_final_teardown) {
        // Final teardown consumes the debt even if the user disabled shell repair.
        retired_virtual_display_explorer_repair_debt = false;
      }
      retired_virtual_display_session.reset();
      retired_virtual_display_remove_ready = false;
      retired_virtual_display_deactivate_desktop = false;
      retired_virtual_display_requires_device_absence = false;
      retired_virtual_display_was_active_desktop_path = false;
      retired_virtual_display_final_teardown = false;
      retired_virtual_display_restart_explorer_after_removal = false;
      retired_virtual_display_detach_context.reset();
      ++retired_virtual_display_generation;
      retired_virtual_display_cleanup_scheduled = false;
      if (restart_explorer) {
        scheduleExplorerRepairLocked();
      }
      retired_virtual_display_changed.notify_all();
    }

    // Driver retirement honors the caller's timeout independently of the Explorer worker.
    // New display owners may request this barrier before reusing the desktop topology.
    if (wait_for_explorer_repair && retired_virtual_display_explorer_repair_in_progress) {
      const bool repair_finished = retired_virtual_display_changed.wait_until(
        retirement_lock,
        deadline,
        []() {
          return !retired_virtual_display_explorer_repair_in_progress;
        }
      );
      return repair_finished && !retired_virtual_display_session;
    }
    return !retired_virtual_display_session;
  }

  bool proc_t::has_retired_virtual_display() {
    std::lock_guard retirement_lock(retired_virtual_display_mutex);
    return retired_virtual_display_session.has_value() ||
           retired_virtual_display_explorer_repair_in_progress;
  }
}  // namespace proc
