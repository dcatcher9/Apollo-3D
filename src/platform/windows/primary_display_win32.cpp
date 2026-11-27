/**
 * @file src/platform/windows/primary_display_win32.cpp
 * @brief The production adapter of the primary-display transactions.
 */
#include "primary_display_win32.h"

#include "ar_glasses.h"
#include "display_config.h"
#include "misc.h"
#include "src/logging.h"
#include "src/utility.h"

#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace platf::primary_display::detail {
  namespace {
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

    // The cursor clip this process owns for an exclusive session. Access is serialized by the
    // caller's transaction mutex, like every other use of this adapter.
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
  }  // namespace

  io_t system_io() {
    return {query_snapshot, [](detail::snapshot_t snapshot) {
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
            }};
  }

  bool refresh_owned_cursor_clip() {
    if (!active_cursor_clip_identity) {
      return true;
    }
    if (!active_cursor_clip_bounds) {
      return false;
    }
    const auto identity = *active_cursor_clip_identity;
    const auto bounds = *active_cursor_clip_bounds;
    return set_exclusive_cursor_clip(identity, bounds);
  }
}  // namespace platf::primary_display::detail
