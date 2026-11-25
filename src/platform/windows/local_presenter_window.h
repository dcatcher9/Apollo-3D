/**
 * @file src/platform/windows/local_presenter_window.h
 * @brief Local presenter window, pointer isolation and display identity helpers.
 */
#pragma once

// standard includes
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// platform includes
#include <windows.h>

// local includes
#include "display.h"
#include "local_monitor_pointer_geometry.h"
#include "misc.h"
#include "src/logging.h"
#include "src/utility.h"

namespace platf::dxgi {
  /** Internals of run_local_presenter(), for display_vram.cpp only. */
  namespace local_presenter {
    using namespace std::literals;

    constexpr wchar_t local_presenter_window_class[] = L"ApolloLocalArPresenter";

    void log_local_shutter_window_snapshot(HWND presenter, RECT target_rect, HMONITOR expected_monitor);
    bool same_local_presenter_rect(const RECT &left, const RECT &right);
    bool valid_local_presenter_rect(const RECT &rect);
    RECT local_presenter_virtual_screen_rect();
    std::optional<RECT> intersect_local_presenter_rects(const RECT &left, const RECT &right);
    std::optional<RECT> query_local_presenter_cursor_clip(int attempts = 1);

    struct deferred_cursor_restore_t {
      RECT previous_clip {};
      RECT owned_clip {};
      bool previous_clip_unbounded = false;
    };

    class deferred_cursor_restore_manager_t {
    public:
      deferred_cursor_restore_manager_t():
          worker_([this]() {
            run();
          }) {
      }

      ~deferred_cursor_restore_manager_t() {
        {
          std::lock_guard lock(mutex_);
          stopping_ = true;
        }
        changed_.notify_all();
        if (worker_.joinable()) {
          worker_.join();
        }
      }

      void defer(const deferred_cursor_restore_t &restore) {
        std::lock_guard lock(mutex_);
        if (pending_ && !same_local_presenter_rect(pending_->owned_clip, restore.owned_clip)) {
          BOOST_LOG(error) << "Local AR refused to overwrite an unresolved cursor-restoration record."sv;
          return;
        }
        pending_ = restore;
        retry_delay_ = 25ms;
        changed_.notify_all();
      }

      // A new owner must not save Apollo's unresolved clip as the user's prior restriction.
      bool ready_for_new_owner() {
        std::lock_guard lock(mutex_);
        resolve_once_locked();
        return !pending_;
      }

    private:
      bool resolve_once_locked() {
        if (!pending_) {
          return true;
        }
        const auto current = query_local_presenter_cursor_clip(3);
        if (!current) {
          return false;
        }
        if (!same_local_presenter_rect(*current, pending_->owned_clip)) {
          // Another application replaced Apollo's clip. Its state is authoritative, so restoring
          // the older rectangle now would corrupt that application's cursor confinement.
          pending_.reset();
          return true;
        }
        const BOOL restored = pending_->previous_clip_unbounded ?
                                ClipCursor(nullptr) :
                                ClipCursor(&pending_->previous_clip);
        if (!restored) {
          return false;
        }
        pending_.reset();
        return true;
      }

      void run() {
        std::unique_lock lock(mutex_);
        while (!stopping_) {
          changed_.wait(lock, [&]() {
            return stopping_ || pending_.has_value();
          });
          while (!stopping_ && pending_) {
            if (resolve_once_locked()) {
              retry_delay_ = 25ms;
              continue;
            }
            const auto delay = retry_delay_;
            retry_delay_ = std::min(retry_delay_ * 2, 1000ms);
            changed_.wait_for(lock, delay, [&]() {
              return stopping_ || !pending_;
            });
          }
        }
        // Make one final best effort while Win32 and the process desktop are still alive.
        resolve_once_locked();
      }

      std::mutex mutex_;
      std::condition_variable changed_;
      std::optional<deferred_cursor_restore_t> pending_;
      std::chrono::milliseconds retry_delay_ {25};
      bool stopping_ = false;
      std::thread worker_;
    };

    deferred_cursor_restore_manager_t &deferred_cursor_restore_manager();
    void defer_cursor_retry(local_presenter_cursor_clip_t &clip);
    void clear_cursor_retry(local_presenter_cursor_clip_t &clip);

    struct local_presenter_display_identity_t {
      LUID adapter_id {};
      UINT32 target_id = 0;
      std::wstring gdi_name;
      std::wstring device_path;
      std::wstring friendly_name;
    };

    bool same_local_presenter_luid(const LUID &left, const LUID &right);
    std::optional<local_presenter_display_identity_t> query_local_presenter_display_identity(
      std::wstring_view display_name
    );
    bool local_presenter_identity_is_coherent(
      const std::shared_ptr<local_presenter_config_t::target_t> &live_target,
      const local_presenter_display_identity_t &source,
      const local_presenter_display_identity_t &target,
      const LUID &expected_adapter
    );
    std::optional<RECT> query_local_presenter_display_rect(std::wstring_view display_name);

    struct local_presenter_pointer_isolation_t {
      struct clip_request_t {
        bool geometry_known = false;
        std::optional<RECT> clip;
      };

      std::wstring source_display_name;
      RECT source_fallback_rect {};
      RECT target_fallback_rect {};
      std::shared_ptr<local_presenter_config_t::target_t> live_target;
      std::shared_ptr<local_presenter_cursor_clip_t> cursor_clip;
      bool park_only = false;
      // Monitor presentation only: the sink as the mouse hook sees it, and the last pointer
      // position outside it.
      mutable std::mutex hook_mutex;
      mutable RECT hook_sink {};
      mutable std::optional<POINT> last_interactive_point;

      mutable std::vector<RECT> hook_monitors;  ///< Every monitor except the sink.

      void refresh_hook_sink() const {
        const auto sink = current_target_rect();
        auto monitors = desktop_monitor_rects().value_or(std::vector<RECT> {});
        std::erase_if(monitors, [&](const RECT &monitor) {
          return same_local_presenter_rect(monitor, sink);
        });
        std::lock_guard lock(hook_mutex);
        hook_sink = sink;
        hook_monitors = std::move(monitors);
        last_interactive_point.reset();
      }

      // Called for every mouse movement with the position the movement asks for, which Windows
      // has not yet limited to the desktop. Returns true when this has placed the pointer itself.
      // Windows carries a pointer pushed beyond an outer desktop edge over to another monitor;
      // with the sink parked at a corner that is the sink, and the pointer would come back at
      // the corner. So a movement that ends on no monitor in use is finished here instead: the
      // pointer goes to the nearest point of those monitors, which lets it slide along the edge
      // as it does on a single monitor.
      bool refuse_pointer_move(POINT destination) const {
        POINT stay {};
        {
          std::lock_guard lock(hook_mutex);
          if (!valid_local_presenter_rect(hook_sink) || hook_monitors.empty()) {
            return false;
          }
          long double nearest = std::numeric_limits<long double>::max();
          for (const auto &monitor : hook_monitors) {
            if (PtInRect(&monitor, destination)) {
              last_interactive_point = destination;
              return false;
            }
            const POINT candidate {
              std::clamp(destination.x, monitor.left, monitor.right - 1),
              std::clamp(destination.y, monitor.top, monitor.bottom - 1),
            };
            const auto dx = static_cast<long double>(candidate.x) - destination.x;
            const auto dy = static_cast<long double>(candidate.y) - destination.y;
            if (dx * dx + dy * dy < nearest) {
              nearest = dx * dx + dy * dy;
              stay = candidate;
            }
          }
          last_interactive_point = stay;
        }
        const auto previous_dpi = SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        SetCursorPos(stay.x, stay.y);
        if (previous_dpi) {
          SetThreadDpiAwarenessContext(previous_dpi);
        }
        return true;
      }

      RECT current_source_rect() const {
        if (const auto rect = query_local_presenter_display_rect(source_display_name)) {
          return *rect;
        }
        return source_fallback_rect;
      }

      RECT current_target_rect() const {
        RECT fallback = target_fallback_rect;
        std::wstring display_name;
        if (live_target) {
          std::lock_guard lock(live_target->mutex);
          fallback = live_target->rect;
          display_name = platf::from_utf8(live_target->display_name);
        }
        if (const auto rect = query_local_presenter_display_rect(display_name)) {
          return *rect;
        }
        return fallback;
      }

      // Every monitor of the desktop, in physical pixels.
      static std::optional<std::vector<RECT>> desktop_monitor_rects() {
        const auto previous_dpi = SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        auto restore_dpi = util::fail_guard([previous_dpi]() {
          if (previous_dpi) {
            SetThreadDpiAwarenessContext(previous_dpi);
          }
        });
        if (!AreDpiAwarenessContextsEqual(GetThreadDpiAwarenessContext(), DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
          return std::nullopt;
        }
        std::vector<RECT> monitors;
        const auto enumerated = EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR monitor, HDC, LPRECT, LPARAM parameter) -> BOOL {
          MONITORINFO info {sizeof(MONITORINFO)};
          if (!GetMonitorInfoW(monitor, &info)) {
            return FALSE;
          }
          reinterpret_cast<std::vector<RECT> *>(parameter)->push_back(info.rcMonitor);
          return TRUE;
        }, reinterpret_cast<LPARAM>(&monitors));
        if (!enumerated) {
          return std::nullopt;
        }
        return monitors;
      }

      clip_request_t desired_clip() const {
        const auto source_rect = current_source_rect();
        const auto target_rect = current_target_rect();
        if (!valid_local_presenter_rect(source_rect) || !valid_local_presenter_rect(target_rect)) {
          return {};
        }

        // The local-AR topology is interactive displays -> private virtual source -> physical
        // presentation sink. A rectangular cursor clip can exclude the sink without excluding an
        // interactive display only when the source and sink share that exact vertical boundary.
        const bool vertically_connected = std::max(source_rect.top, target_rect.top) <
                                          std::min(source_rect.bottom, target_rect.bottom);
        if (target_rect.left != source_rect.right || !vertically_connected) {
          return {true, std::nullopt};
        }

        RECT clip = local_presenter_virtual_screen_rect();
        clip.right = source_rect.right;
        if (!valid_local_presenter_rect(clip)) {
          return {true, std::nullopt};
        }
        return {true, clip};
      }

      void acquire_clip() {
        if (park_only || !cursor_clip || std::chrono::steady_clock::now() < cursor_clip->retry_after) {
          return;
        }
        if (!deferred_cursor_restore_manager().ready_for_new_owner()) {
          defer_cursor_retry(*cursor_clip);
          return;
        }
        if (cursor_clip->clip_yielded) {
          const auto current_clip = query_local_presenter_cursor_clip();
          if (!current_clip) {
            defer_cursor_retry(*cursor_clip);
            return;
          }
          const auto released_clip = cursor_clip->previous_clip_unbounded ?
                                       local_presenter_virtual_screen_rect() :
                                       cursor_clip->previous_clip;
          const auto unbounded_clip = local_presenter_virtual_screen_rect();
          const bool released_to_unbounded = same_local_presenter_rect(*current_clip, unbounded_clip);
          if (!same_local_presenter_rect(*current_clip, released_clip) && !released_to_unbounded) {
            cursor_clip->retry_after = std::chrono::steady_clock::now() + 250ms;
            return;  // another application still owns cursor confinement
          }
          if (released_to_unbounded) {
            // ClipCursor has no ownership stack. Once another application replaces the saved
            // restriction and releases to unbounded, that is the new state Apollo must preserve.
            cursor_clip->previous_clip = unbounded_clip;
            cursor_clip->previous_clip_unbounded = true;
          }
        }
        auto request = desired_clip();
        if (!request.geometry_known) {
          return;
        }
        auto requested_clip = request.clip;
        if (!requested_clip) {
          if (!cursor_clip->clip_unavailable_logged) {
            BOOST_LOG(debug) << "Local AR pointer confinement is unavailable for the current topology; "sv
                                "using event-driven edge clamping."sv;
            cursor_clip->clip_unavailable_logged = true;
          }
          return;
        }
        if (!cursor_clip->clip_saved) {
          const auto previous_clip = query_local_presenter_cursor_clip(3);
          if (!previous_clip) {
            BOOST_LOG(warning) << "Local AR presenter could not save the existing cursor clip: "sv
                               << GetLastError();
            defer_cursor_retry(*cursor_clip);
            return;
          }
          cursor_clip->previous_clip = *previous_clip;
          cursor_clip->clip_saved = true;
          cursor_clip->previous_clip_unbounded = same_local_presenter_rect(
            *previous_clip,
            local_presenter_virtual_screen_rect()
          );
        }
        if (!cursor_clip->previous_clip_unbounded) {
          requested_clip = intersect_local_presenter_rects(
            *requested_clip,
            cursor_clip->previous_clip
          );
          if (!requested_clip) {
            BOOST_LOG(debug) << "Local AR left another application's disjoint cursor confinement untouched."sv;
            cursor_clip->clip_yielded = true;
            cursor_clip->retry_after = std::chrono::steady_clock::now() + 250ms;
            return;
          }
        }
        if (!ClipCursor(&*requested_clip)) {
          BOOST_LOG(warning) << "Local AR presenter could not confine the cursor: "sv
                             << GetLastError();
          defer_cursor_retry(*cursor_clip);
          return;
        }

        // ClipCursor succeeded, so Apollo may own this clip even if the verification query is
        // transiently unavailable. Track tentative ownership so teardown can still restore it.
        cursor_clip->owned_clip = *requested_clip;
        cursor_clip->owns_clip = true;
        cursor_clip->clip_yielded = false;
        clear_cursor_retry(*cursor_clip);
        const auto applied_clip = query_local_presenter_cursor_clip(3);
        if (applied_clip && !same_local_presenter_rect(*applied_clip, *requested_clip)) {
          BOOST_LOG(warning) << "Local AR cursor confinement was overridden while being applied; "sv
                                "using event-driven edge clamping."sv;
          cursor_clip->owns_clip = false;
          cursor_clip->clip_yielded = true;
          return;
        }
        BOOST_LOG(debug) << "Local AR cursor confined through virtual-source right edge x="sv
                         << cursor_clip->owned_clip.right << '.';
      }

      void refresh_clip() {
        if (park_only || !cursor_clip) {
          return;
        }
        if (!cursor_clip->owns_clip) {
          acquire_clip();
          return;
        }

        const auto current_clip = query_local_presenter_cursor_clip();
        if (!current_clip) {
          // A query failure is not evidence that another application replaced Apollo's clip.
          // Keep tentative ownership and retry on the next pointer/topology event.
          defer_cursor_retry(*cursor_clip);
          return;
        }
        if (!same_local_presenter_rect(*current_clip, cursor_clip->owned_clip)) {
          // ClipCursor is shared system state. Never fight an application that replaces Apollo's
          // clip, and never restore over that application's state during teardown.
          cursor_clip->owns_clip = false;
          cursor_clip->clip_yielded = true;
          cursor_clip->retry_after = std::chrono::steady_clock::now() + 250ms;
          BOOST_LOG(debug) << "Local AR cursor confinement was replaced by another application; "sv
                              "leaving it untouched and using event-driven edge clamping."sv;
          return;
        }

        const auto request = desired_clip();
        if (!request.geometry_known) {
          // Display queries commonly fail while DXGI is rebuilding. Preserve the existing clip;
          // indeterminate geometry is not proof that the source/sink row stopped being valid.
          return;
        }
        const auto requested_clip = request.clip;
        if (!requested_clip) {
          if (!cursor_clip->restore()) {
            defer_cursor_retry(*cursor_clip);
          }
          cursor_clip->clip_unavailable_logged = false;
          return;
        }
        auto bounded_clip = requested_clip;
        if (!cursor_clip->previous_clip_unbounded) {
          bounded_clip = intersect_local_presenter_rects(*requested_clip, cursor_clip->previous_clip);
          if (!bounded_clip) {
            if (cursor_clip->restore()) {
              cursor_clip->retry_after = std::chrono::steady_clock::now() + 250ms;
            } else {
              defer_cursor_retry(*cursor_clip);
            }
            return;
          }
        }
        if (same_local_presenter_rect(*bounded_clip, cursor_clip->owned_clip)) {
          return;
        }
        if (!ClipCursor(&*bounded_clip)) {
          BOOST_LOG(warning) << "Local AR presenter could not update cursor confinement: "sv
                             << GetLastError();
          defer_cursor_retry(*cursor_clip);
          return;
        }

        cursor_clip->owned_clip = *bounded_clip;
        cursor_clip->owns_clip = true;
        clear_cursor_retry(*cursor_clip);
        const auto applied_clip = query_local_presenter_cursor_clip(3);
        if (applied_clip && !same_local_presenter_rect(*applied_clip, *bounded_clip)) {
          cursor_clip->owns_clip = false;
          cursor_clip->clip_yielded = true;
        }
      }

      bool clamp_pointer_to_source_edge() const {
        if (park_only) {
          // Monitor presentation parks only its physical scanout sink. Every other output remains
          // interactive; a global rectangular clip would incorrectly exclude those monitors, and
          // Windows drops a clip whenever another program comes to the front. The mouse hook
          // refuses entry; this moves back a pointer that got in while no hook was installed.
          const auto previous_dpi = SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
          auto restore_dpi = util::fail_guard([previous_dpi]() {
            if (previous_dpi) {
              SetThreadDpiAwarenessContext(previous_dpi);
            }
          });
          if (!AreDpiAwarenessContextsEqual(GetThreadDpiAwarenessContext(), DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
            return false;
          }
          const auto sink = current_target_rect();
          POINT cursor {};
          if (!valid_local_presenter_rect(sink) || !GetCursorPos(&cursor) || !PtInRect(&sink, cursor)) {
            return false;
          }
          const auto monitors = desktop_monitor_rects();
          if (!monitors) {
            return false;
          }
          {
            std::lock_guard lock(hook_mutex);
            if (last_interactive_point && !PtInRect(&sink, *last_interactive_point)) {
              return SetCursorPos(last_interactive_point->x, last_interactive_point->y) != FALSE;
            }
          }
          const auto clamped = detail::nearest_interactive_pointer_point(cursor, sink, *monitors);
          return clamped && SetCursorPos(clamped->x, clamped->y) != FALSE;
        }
        const auto source_rect = current_source_rect();
        const auto target_rect = current_target_rect();
        if (!valid_local_presenter_rect(source_rect) || !valid_local_presenter_rect(target_rect)) {
          return false;
        }

        POINT cursor {};
        if (!GetCursorPos(&cursor) || !PtInRect(&target_rect, cursor)) {
          return false;
        }

        const LONG clamped_x = std::clamp(cursor.x, source_rect.left, source_rect.right - 1);
        const LONG clamped_y = std::clamp(cursor.y, source_rect.top, source_rect.bottom - 1);
        return SetCursorPos(clamped_x, clamped_y) != FALSE;
      }

      void cancel_active_pointer_interaction(HWND sink_window) const {
        HWND recipient = GetCapture();
        if (!recipient || recipient == sink_window) {
          recipient = GetForegroundWindow();
        }
        if (!recipient || recipient == sink_window) {
          return;
        }
        DWORD_PTR ignored = 0;
        SendMessageTimeoutW(
          recipient,
          WM_CANCELMODE,
          0,
          0,
          SMTO_ABORTIFHUNG | SMTO_BLOCK,
          50,
          &ignored
        );
      }
    };

    extern std::atomic<const local_presenter_pointer_isolation_t *> hooked_pointer_isolation;

    LRESULT CALLBACK local_presenter_mouse_hook(int code, WPARAM wparam, LPARAM lparam);
    bool register_local_presenter_window_class();
  }  // namespace local_presenter
}  // namespace platf::dxgi
