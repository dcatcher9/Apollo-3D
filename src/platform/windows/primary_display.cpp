/**
 * @file src/platform/windows/primary_display.cpp
 * @brief Temporary primary-display transactions with crash recovery.
 */
#include "primary_display.h"

#include "ar_glasses.h"
#include "display_config.h"
#include "misc.h"
#include "primary_display_journal.h"
#include "primary_display_manager.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/utility.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

namespace platf::primary_display {
  namespace {
    using detail::clear_journal;
    using detail::journal_path;
    using detail::load_journal;
    using detail::max_available_paths;
    using detail::max_displays;
    using detail::save_journal;
    using display_config::same_device_path;

    std::optional<detail::snapshot_t> query_snapshot_flags(UINT32 flags) {
      detail::snapshot_t result;
      // Every available target is listed, named or not; an active target must be named.
      const auto status = display_config::query_named_topology(
        flags | QDC_VIRTUAL_MODE_AWARE | QDC_VIRTUAL_REFRESH_RATE_AWARE,
        result,
        flags == QDC_ALL_PATHS ? display_config::unnamed_target_e::allowed : display_config::unnamed_target_e::rejected
      );
      if (status.unnamed_target) {
        return std::nullopt;
      }
      if (!status || result.paths.size() > (flags == QDC_ALL_PATHS ? max_available_paths : max_displays)) {
        BOOST_LOG(warning) << "Could not query displays for temporary primary-display change: " << status.status;
        return std::nullopt;
      }
      for (const auto &path : result.paths) {
        result.colors.push_back(path.flags & DISPLAYCONFIG_PATH_ACTIVE ? display_config::query_advanced_color(path.targetInfo.adapterId, path.targetInfo.id) : std::nullopt);
      }
      return result;
    }

    std::optional<detail::snapshot_t> query_snapshot() {
      return query_snapshot_flags(QDC_ONLY_ACTIVE_PATHS);
    }

    bool apply_color(const DISPLAYCONFIG_PATH_INFO &path, const display_config::advanced_color_state_t &color) {
      const auto current = display_config::query_advanced_color(path.targetInfo.adapterId, path.targetInfo.id);
      if (!current) {
        return false;
      }
      bool success = true;
      if (current->hdr_user_enabled != color.hdr_user_enabled) {
        success = display_config::set_hdr_state(path.targetInfo.adapterId, path.targetInfo.id, color.hdr_user_enabled) && success;
      }
      if (current->wcg_user_enabled != color.wcg_user_enabled) {
        success = display_config::set_wcg_state(path.targetInfo.adapterId, path.targetInfo.id, color.wcg_user_enabled) && success;
      }
      return success;
    }

    bool apply_snapshot(detail::snapshot_t snapshot, bool force_driver_mode_set) {
      // Never save the temporary primary/layout to the Windows topology database and never
      // permit CCD to choose replacement modes. Restore reconstructs inactive paths using
      // their saved modes and currently available source/target identities.
      // This Windows 11 host preserves virtual/physical refresh-rate distinctions in
      // both the query and apply directions, including BOOST_REFRESH_RATE/divider data.
      constexpr UINT32 flags = SDC_VIRTUAL_MODE_AWARE | SDC_VIRTUAL_REFRESH_RATE_AWARE;
      // NO_OPTIMIZATION is valid only for APPLY. A topology-only transition can otherwise
      // leave an already-active physical target's driver mode untouched after monitor wake.
      const auto result = display_config::validate_and_apply(snapshot, flags, force_driver_mode_set ? SDC_NO_OPTIMIZATION : 0);
      if (result.state == display_config::supplied_apply_e::invalid) {
        BOOST_LOG(warning) << "Temporary primary-display configuration is not valid: " << result.status;
      } else if (result.state == display_config::supplied_apply_e::failed) {
        BOOST_LOG(warning) << "Temporary primary-display configuration failed: " << result.status;
      }
      return result.state == display_config::supplied_apply_e::applied;
    }

    std::mutex transaction_mutex;
    detail::ownership_t process_ownership;
    detail::restore_observation_cache_t restore_observations;

    detail::cursor_clip_manager_t &exclusive_cursor_clip() {
      static detail::cursor_clip_manager_t cursor_clip({
        []() -> std::optional<detail::cursor_bounds_t> {
          RECT rect {};
          if (!GetClipCursor(&rect)) {
            return std::nullopt;
          }
          return detail::cursor_bounds_t {rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top};
        },
        []() -> std::optional<detail::cursor_bounds_t> {
          detail::cursor_bounds_t desktop {GetSystemMetrics(SM_XVIRTUALSCREEN), GetSystemMetrics(SM_YVIRTUALSCREEN), GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN)};
          return desktop.valid() ? std::optional {desktop} : std::nullopt;
        },
        [](std::optional<detail::cursor_bounds_t> clip) {
          if (!clip) {
            return ClipCursor(nullptr) != FALSE;
          }
          const RECT rect {clip->x, clip->y, clip->x + clip->width, clip->y + clip->height};
          return ClipCursor(&rect) != FALSE;
        },
      });
      return cursor_clip;
    }

    // Access is serialized by transaction_mutex. Session ownership is deliberately separate
    // from both the recovery journal and ClipCursor ownership: a virtual-only session with no
    // active AR output needs hotplug reconciliation, but does not need a cursor clip.
    detail::exclusive_session_state_t exclusive_session_state;
    std::optional<std::wstring> active_cursor_clip_identity;
    std::optional<detail::cursor_bounds_t> active_cursor_clip_bounds;

    bool set_exclusive_cursor_clip(std::wstring_view identity, std::optional<detail::cursor_bounds_t> bounds) {
      auto &cursor_clip = exclusive_cursor_clip();
      const bool existing_identity = active_cursor_clip_identity && same_device_path(*active_cursor_clip_identity, identity);
      const bool existing_bounds = existing_identity && active_cursor_clip_bounds == bounds;
      if (!bounds) {
        // A retiring older display may not release or cancel maintenance for a newer session.
        if (active_cursor_clip_identity && !identity.empty() && !existing_identity) {
          return false;
        }
        // Stop asynchronous maintenance before attempting release. If release itself fails,
        // restore() will retry it, while display/focus notifications must not trap a disconnected
        // local user by reacquiring the clip.
        active_cursor_clip_identity.reset();
        active_cursor_clip_bounds.reset();
        if (!cursor_clip.owns_clip()) {
          return true;
        }
      }
      if (!syncThreadDesktop()) {
        BOOST_LOG(warning) << "Could not synchronize the desktop for exclusive virtual-display cursor isolation.";
        return false;
      }
      const auto previous_dpi = SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
      auto restore_dpi = util::fail_guard([&]() {
        if (previous_dpi) {
          SetThreadDpiAwarenessContext(previous_dpi);
        }
      });
      if (GetAwarenessFromDpiAwarenessContext(GetThreadDpiAwarenessContext()) != DPI_AWARENESS_PER_MONITOR_AWARE) {
        return false;
      }
      const bool result = bounds ? cursor_clip.confine(identity, *bounds) : cursor_clip.release(identity);
      if (!result) {
        BOOST_LOG(warning) << "Could not " << (bounds ? "apply" : "release") << " exclusive virtual-display cursor isolation; retaining cursor ownership for cleanup.";
      } else if (bounds) {
        active_cursor_clip_identity = identity;
        active_cursor_clip_bounds = bounds;
        if (!existing_bounds) {
          BOOST_LOG(info) << "Windows cursor restricted to the virtual display for exclusive streaming: ["
                          << bounds->x << ',' << bounds->y << ',' << bounds->width << ',' << bounds->height << "].";
        }
      } else {
        BOOST_LOG(info) << "Released exclusive virtual-display cursor isolation.";
      }
      return result;
    }

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
      return detail::manager_t({query_snapshot, [](detail::snapshot_t snapshot) {
                                  return apply_snapshot(std::move(snapshot), false);
                                },
                                load_journal,
                                save_journal,
                                clear_journal,
                                [] {
                                  return query_snapshot_flags(QDC_ALL_PATHS);
                                },
                                apply_color,
                                [](const DISPLAYCONFIG_PATH_INFO &path, std::wstring_view) {
                                  return ar_glasses::preserve_during_remote_virtual_display(
                                    path.targetInfo.adapterId,
                                    path.targetInfo.id
                                  );
                                },
                                set_exclusive_cursor_clip,
                                []() -> std::optional<POINT> {
                                  POINT cursor {};
                                  // CCD rectangles use physical pixels regardless of thread DPI.
                                  return syncThreadDesktop() && GetPhysicalCursorPos(&cursor) ?
                                           std::optional {cursor} :
                                           std::nullopt;
                                },
                                [](POINT cursor) {
                                  return syncThreadDesktop() && SetPhysicalCursorPos(cursor.x, cursor.y);
                                },
                                [](detail::snapshot_t snapshot) {
                                  return apply_snapshot(std::move(snapshot), true);
                                },
                                [](std::chrono::milliseconds delay) {
                                  std::this_thread::sleep_for(delay);
                                }},
                               &restore_observations);
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
      if (!active_cursor_clip_identity) {
        return true;
      }
      if (!active_cursor_clip_bounds) {
        return false;
      }
      const auto identity = *active_cursor_clip_identity;
      const auto bounds = *active_cursor_clip_bounds;
      return set_exclusive_cursor_clip(identity, bounds);
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
