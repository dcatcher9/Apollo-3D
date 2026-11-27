/**
 * @file src/platform/windows/primary_display.cpp
 * @brief Temporary primary-display transactions with crash recovery: the process-wide entry
 * points, which serialize every transaction and own the session state around it.
 */
#include "primary_display.h"

#include "primary_display_journal.h"
#include "primary_display_manager.h"
#include "primary_display_win32.h"
#include "src/logging.h"

#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace platf::primary_display {
  namespace {
    using detail::journal_path;
    using display_config::same_device_path;
    using namespace std::literals;

    std::mutex transaction_mutex;
    detail::ownership_t process_ownership;
    detail::restore_observation_cache_t restore_observations;

    // Access is serialized by transaction_mutex. Session ownership is deliberately separate
    // from both the recovery journal and ClipCursor ownership: a virtual-only session with no
    // active AR output needs hotplug reconciliation, but does not need a cursor clip.
    detail::exclusive_session_state_t exclusive_session_state;

    bool acquire_ownership() {
      auto path = journal_path();
      path.replace_extension(L".lock");
      if (!process_ownership.acquire(path)) {
        BOOST_LOG(warning) << "Cannot acquire primary-display recovery ownership; another host may be using this configuration.";
        return false;
      }
      return true;
    }

    detail::manager_t manager() {
      return detail::manager_t(detail::system_io(), &restore_observations);
    }
  }  // namespace

  bool promote(std::wstring_view device_path, bool exclusive) {
    std::lock_guard lock(transaction_mutex);
    try {
      if (exclusive && !exclusive_session_state.begin_promotion(device_path)) {
        BOOST_LOG(warning) << "Refusing to promote a different display while an exclusive display session is pending or active.";
        return false;
      }
      const bool promoted = acquire_ownership() && manager().promote(device_path, exclusive);
      if (exclusive && promoted && !exclusive_session_state.promotion_succeeded(device_path)) {
        return false;
      }
      return promoted;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Temporary primary-display promotion failed: " << exception.what();
      return false;
    }
  }

  bool prepare(bool exclusive) {
    std::lock_guard lock(transaction_mutex);
    try {
      const bool prepared = acquire_ownership() && manager().prepare(exclusive);
      if (exclusive && prepared) {
        exclusive_session_state.prepared();
      }
      return prepared;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not prepare primary-display recovery: " << exception.what();
      return false;
    }
  }

  bool prepare_local_exclusive(std::wstring_view sink_device_path) {
    if (sink_device_path.empty()) {
      return false;
    }
    std::lock_guard lock(transaction_mutex);
    try {
      const bool prepared = acquire_ownership() && manager().prepare(true, sink_device_path);
      if (prepared) {
        exclusive_session_state.prepared();
      }
      return prepared;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not prepare local AR display recovery: " << exception.what();
      return false;
    }
  }

  bool prepare_local_monitor(std::wstring_view sink_device_path) {
    if (sink_device_path.empty()) {
      return false;
    }
    std::lock_guard lock(transaction_mutex);
    try {
      const bool prepared = acquire_ownership() && manager().prepare(true, sink_device_path, true);
      if (prepared) {
        exclusive_session_state.prepared();
      }
      return prepared;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not prepare physical-monitor replacement recovery: " << exception.what();
      return false;
    }
  }

  bool verify_local_monitor_replacement(std::wstring_view source_device_path, std::wstring_view sink_device_path) {
    std::lock_guard lock(transaction_mutex);
    const auto &active = exclusive_session_state.active_identity();
    if (!active || exclusive_session_state.pending_identity() || !same_device_path(*active, source_device_path)) {
      return false;
    }
    try {
      return manager().verify_local_monitor_replacement(source_device_path, sink_device_path);
    } catch (const std::exception &exception) {
      BOOST_LOG(warning) << "Could not verify physical-monitor replacement: " << exception.what();
      return false;
    }
  }

  bool run_local_exclusive_update(std::wstring_view source_device_path, const std::function<bool()> &update) {
    // The callback can refresh the caller's display-name/path strings after Windows
    // renumbers a source. Keep the transaction identity independent of that storage.
    const std::wstring source_identity(source_device_path);
    std::lock_guard lock(transaction_mutex);
    try {
      if (!update || !exclusive_session_state.can_use_identity(source_identity) || !acquire_ownership() || !manager().is_local_exclusive(source_identity) || !exclusive_session_state.begin_promotion(source_identity)) {
        return false;
      }
      const bool updated = update();
      const bool promoted = manager().promote(source_identity, true);
      if (promoted) {
        exclusive_session_state.promotion_succeeded(source_identity);
      }
      return updated && promoted;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Local AR display update failed: " << exception.what();
      return false;
    }
  }

  bool change_virtual_display_color(std::wstring_view device_path, const std::function<bool()> &change) {
    // The callback can outlive the caller's view of a renumbered display.
    const std::wstring identity(device_path);
    std::lock_guard lock(transaction_mutex);
    if (!change || !change()) {
      return false;
    }
    try {
      // Without a recorded exclusive state there is nothing to keep in step.
      if (!identity.empty() && acquire_ownership() && !manager().adopt_virtual_color(identity)) {
        BOOST_LOG(warning) << "The virtual display's new color state could not be recorded for its exclusive session; the next display change may restore the previous state."sv;
      }
    } catch (const std::exception &exception) {
      BOOST_LOG(warning) << "Could not record the virtual display's color state: " << exception.what();
    }
    return true;
  }

  long run_serialized_mode_change(const std::function<long()> &change) {
    std::lock_guard lock(transaction_mutex);
    return change();
  }

  bool bind_pending(std::wstring_view device_path) {
    std::lock_guard lock(transaction_mutex);
    try {
      if (exclusive_session_state.expected() && !exclusive_session_state.can_use_identity(device_path)) {
        BOOST_LOG(warning) << "Refusing to bind a different display to the pending exclusive display session.";
        return false;
      }
      const bool bound = acquire_ownership() && manager().bind_pending(device_path);
      if (bound) {
        exclusive_session_state.bound(device_path);
      }
      return bound;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not bind primary-display recovery identity: " << exception.what();
      return false;
    }
  }

  bool baseline_topology_active(std::wstring_view device_path) {
    std::lock_guard lock(transaction_mutex);
    try {
      return acquire_ownership() && manager().baseline_topology_active(device_path);
    } catch (const std::exception &exception) {
      BOOST_LOG(warning) << "Could not inspect the active display topology: " << exception.what();
      return false;
    }
  }

  bool restore(std::wstring_view expected_device_path) {
    std::lock_guard lock(transaction_mutex);
    try {
      const auto action = exclusive_session_state.restore_action(expected_device_path);
      if (action == detail::exclusive_restore_action_e::reject) {
        return false;
      }
      if (action == detail::exclusive_restore_action_e::disarm_before) {
        exclusive_session_state.disarm();
      }
      const bool restored = acquire_ownership() && manager().restore(expected_device_path);
      exclusive_session_state.restore_finished(action, restored);
      return restored;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Temporary primary-display recovery failed: " << exception.what();
      return false;
    }
  }

  bool recover() {
    return restore();
  }

  bool pause(std::wstring_view device_path, retained_display_ptr &retained, std::wstring_view local_sink) {
    std::lock_guard lock(transaction_mutex);
    try {
      const auto action = exclusive_session_state.restore_action(device_path);
      if (action == detail::exclusive_restore_action_e::reject) {
        return false;
      }
      if (action == detail::exclusive_restore_action_e::disarm_before) {
        exclusive_session_state.disarm();
      }
      const bool restored = acquire_ownership() && manager().pause(device_path, retained, local_sink);
      exclusive_session_state.restore_finished(action, restored);
      return restored;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not pause the retained virtual desktop: " << exception.what();
      return false;
    }
  }

  bool reactivate(const retained_display_ptr &retained, bool exclusive) {
    std::lock_guard lock(transaction_mutex);
    try {
      if (!retained) {
        BOOST_LOG(warning) << "Retained virtual-display resume rejected: no retained display state.";
        return false;
      }
      if (!exclusive_session_state.can_use_identity(retained->display.identity)) {
        BOOST_LOG(warning) << "Retained virtual-display resume rejected: session ownership belongs to a different display identity.";
        return false;
      }
      if (!acquire_ownership()) {
        return false;
      }
      // Reconciliation must defer until the caller completes HDR/mode setup and promotion.
      // Failed activation keeps this identity pending until its exact pause/restore rollback.
      exclusive_session_state.prepared();
      exclusive_session_state.bound(retained->display.identity);
      const bool activated = manager().reactivate(retained, exclusive);
      if (activated) {
        retained->last_resume_failure = {};
      }
      if (activated && !exclusive) {
        exclusive_session_state.disarm();
      }
      return activated;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not reactivate the retained virtual desktop: " << exception.what();
      return false;
    }
  }

  void restore_retained_cursor(const retained_display_ptr &retained) {
    std::lock_guard lock(transaction_mutex);
    try {
      if (retained && exclusive_session_state.can_use_identity(retained->display.identity)) {
        manager().restore_retained_cursor(retained);
      }
    } catch (...) {
      // A cursor failure must never undo an otherwise verified resume.
    }
  }

  bool refresh_exclusive_cursor_clip() {
    std::lock_guard lock(transaction_mutex);
    try {
      return detail::refresh_owned_cursor_clip();
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not refresh exclusive virtual-display cursor isolation: " << exception.what();
      return false;
    }
  }

  std::uint64_t exclusive_session_generation() {
    std::lock_guard lock(transaction_mutex);
    return exclusive_session_state.generation();
  }

  namespace {
    exclusive_reconcile_result_t reconcile_exclusive_display_topology_locked(
      std::optional<std::uint64_t> expected_generation
    ) {
      const auto generation = exclusive_session_state.generation();
      if (expected_generation && generation != *expected_generation) {
        return {exclusive_reconcile_result_e::settled, generation};
      }
      try {
        switch (exclusive_session_state.reconcile_action()) {
          case detail::exclusive_reconcile_action_e::defer:
            return {exclusive_reconcile_result_e::settled, generation};
          case detail::exclusive_reconcile_action_e::recover:
            // A crash-recovery journal can remain pending while a saved monitor is unplugged. A
            // later display notification is the earliest safe opportunity to finish that restore.
            return {
              acquire_ownership() && manager().recover_inactive_exclusive() ?
                exclusive_reconcile_result_e::settled :
                exclusive_reconcile_result_e::pending_recovery,
              generation,
            };
          case detail::exclusive_reconcile_action_e::reconcile:
            return {
              manager().reconcile_active_exclusive(*exclusive_session_state.active_identity()) ?
                exclusive_reconcile_result_e::settled :
                exclusive_reconcile_result_e::retry_active,
              generation,
            };
        }
      } catch (const std::exception &exception) {
        BOOST_LOG(error) << "Could not reassert exclusive virtual-display topology: " << exception.what();
        return {
          exclusive_session_state.reconcile_action() == detail::exclusive_reconcile_action_e::reconcile ?
            exclusive_reconcile_result_e::retry_active :
            exclusive_reconcile_result_e::pending_recovery,
          generation,
        };
      }
      return {exclusive_reconcile_result_e::pending_recovery, generation};
    }
  }  // namespace

  exclusive_reconcile_result_t reconcile_exclusive_display_topology() {
    std::lock_guard lock(transaction_mutex);
    return reconcile_exclusive_display_topology_locked(std::nullopt);
  }

  exclusive_reconcile_result_t reconcile_exclusive_display_topology(std::uint64_t expected_generation) {
    std::lock_guard lock(transaction_mutex);
    return reconcile_exclusive_display_topology_locked(expected_generation);
  }
}  // namespace platf::primary_display
