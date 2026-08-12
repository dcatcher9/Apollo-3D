/**
 * @file src/platform/windows/foreground_window_region.cpp
 * @brief Synchronous, fail-closed foreground-window geometry for Host SBS.
 */

#include "foreground_window_region.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string_view>

#include <dwmapi.h>
#include <windows.h>

namespace platf::foreground_window {
  namespace {
    [[nodiscard]] constexpr bool contains(
      const rect_t outer,
      const rect_t inner
    ) noexcept {
      return outer.valid() && inner.valid() &&
             inner.left >= outer.left && inner.top >= outer.top &&
             inner.right <= outer.right && inner.bottom <= outer.bottom;
    }

    [[nodiscard]] constexpr bool intersects(
      const rect_t left,
      const rect_t right
    ) noexcept {
      return left.valid() && right.valid() &&
             std::max(left.left, right.left) < std::min(left.right, right.right) &&
             std::max(left.top, right.top) < std::min(left.bottom, right.bottom);
    }

    [[nodiscard]] constexpr bool observation_has_complete_geometry(
      const observation_t &observation
    ) noexcept {
      return carries_geometry(observation.status) && observation.window != 0 &&
             observation.process_id != 0 && observation.monitor != 0 &&
             observation.client_screen_rect.valid() &&
             observation.frame_screen_rect.valid() &&
             contains(observation.frame_screen_rect, observation.client_screen_rect) &&
             observation.observed_at.time_since_epoch().count() != 0;
    }

    [[nodiscard]] constexpr bool snapshot_has_complete_geometry(
      const snapshot_t &snapshot
    ) noexcept {
      return carries_geometry(snapshot.status) && snapshot.generation != 0 &&
             snapshot.window != 0 && snapshot.process_id != 0 && snapshot.monitor != 0 &&
             snapshot.client_screen_rect.valid() && snapshot.frame_screen_rect.valid() &&
             contains(snapshot.frame_screen_rect, snapshot.client_screen_rect) &&
             snapshot.observed_at.time_since_epoch().count() != 0 &&
             snapshot.geometry_valid_since.time_since_epoch().count() != 0;
    }

    [[nodiscard]] bool is_shell_class(const std::wstring_view name) noexcept {
      constexpr std::array<std::wstring_view, 4> shell_classes {
        L"Progman",
        L"WorkerW",
        L"Shell_TrayWnd",
        L"Shell_SecondaryTrayWnd",
      };
      for (const auto candidate : shell_classes) {
        if (
          name.size() == candidate.size() &&
          _wcsnicmp(name.data(), candidate.data(), name.size()) == 0
        ) {
          return true;
        }
      }
      return false;
    }

    class dpi_scope_t {
    public:
      dpi_scope_t() noexcept {
        previous_ = SetThreadDpiAwarenessContext(
          DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        );
        active_ = previous_ != nullptr && AreDpiAwarenessContextsEqual(
                                               GetThreadDpiAwarenessContext(),
                                               DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
                                             );
      }

      ~dpi_scope_t() {
        if (previous_) {
          (void) SetThreadDpiAwarenessContext(previous_);
        }
      }

      dpi_scope_t(const dpi_scope_t &) = delete;
      dpi_scope_t &operator=(const dpi_scope_t &) = delete;

      [[nodiscard]] bool active() const noexcept {
        return active_;
      }

    private:
      DPI_AWARENESS_CONTEXT previous_ {};
      bool active_ {};
    };
  }  // namespace

  const char *status_name(const status_e status) noexcept {
    switch (status) {
      case status_e::ok:
        return "ok";
      case status_e::no_foreground:
        return "no-foreground";
      case status_e::shell_surface:
        return "shell-surface";
      case status_e::self_process:
        return "self-process";
      case status_e::invalid_window:
        return "invalid-window";
      case status_e::not_visible:
        return "not-visible";
      case status_e::minimized:
        return "minimized";
      case status_e::cloaked:
        return "cloaked";
      case status_e::excluded_style:
        return "excluded-style";
      case status_e::dpi_unavailable:
        return "dpi-unavailable";
      case status_e::geometry_unavailable:
        return "geometry-unavailable";
      case status_e::foreground_changed:
        return "foreground-changed";
    }
    return "unknown";
  }

  namespace detail {
    observation_t classify(
      const raw_observation_t &raw,
      const std::chrono::steady_clock::time_point observed_at
    ) noexcept {
      const auto make_result = [&](const status_e status) {
        return observation_t {
          .status = status,
          .window = raw.window,
          .process_id = raw.process_id,
          .monitor = raw.monitor,
          .client_screen_rect = raw.client_screen_rect,
          .frame_screen_rect = raw.frame_screen_rect,
          .observed_at = observed_at,
        };
      };

      if (raw.window == 0) {
        return make_result(status_e::no_foreground);
      }
      if (!raw.dpi_aware) {
        return make_result(status_e::dpi_unavailable);
      }
      if (
        raw.window == raw.shell_window || raw.window == raw.desktop_window ||
        raw.shell_class
      ) {
        return make_result(status_e::shell_surface);
      }
      if (
        !raw.is_window || !raw.process_query_succeeded || raw.process_id == 0 ||
        !raw.class_query_succeeded || !raw.style_query_succeeded
      ) {
        return make_result(status_e::invalid_window);
      }
      if (raw.own_process_id != 0 && raw.process_id == raw.own_process_id) {
        return make_result(status_e::self_process);
      }
      if (!raw.visible) {
        return make_result(status_e::not_visible);
      }
      if (raw.minimized) {
        return make_result(status_e::minimized);
      }
      if (!raw.cloak_query_succeeded) {
        return make_result(status_e::geometry_unavailable);
      }
      if (raw.cloaked) {
        return make_result(status_e::cloaked);
      }
      constexpr std::uint64_t excluded_styles =
        WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED;
      if ((raw.extended_style & excluded_styles) != 0) {
        return make_result(status_e::excluded_style);
      }
      if (
        !raw.client_rect_succeeded || !raw.frame_rect_succeeded || raw.monitor == 0 ||
        !raw.client_screen_rect.valid() || !raw.frame_screen_rect.valid() ||
        !contains(raw.frame_screen_rect, raw.client_screen_rect)
      ) {
        return make_result(status_e::geometry_unavailable);
      }
      if (
        raw.window_after == 0 || raw.window_after != raw.window ||
        !raw.is_window_after || !raw.process_query_after_succeeded ||
        raw.process_id_after == 0 || raw.process_id_after != raw.process_id
      ) {
        return make_result(status_e::foreground_changed);
      }
      return make_result(status_e::ok);
    }
  }  // namespace detail

  observation_t sample() noexcept {
    const auto observed_at = std::chrono::steady_clock::now();
    dpi_scope_t dpi_scope;
    if (!dpi_scope.active()) {
      return {.status = status_e::dpi_unavailable, .observed_at = observed_at};
    }

    detail::raw_observation_t raw;
    raw.dpi_aware = true;
    const HWND foreground = GetForegroundWindow();
    if (!foreground) {
      return detail::classify(raw, observed_at);
    }
    const HWND window = GetAncestor(foreground, GA_ROOT);
    if (!window) {
      raw.window = reinterpret_cast<std::uintptr_t>(foreground);
      return detail::classify(raw, observed_at);
    }

    raw.window = reinterpret_cast<std::uintptr_t>(window);
    raw.shell_window = reinterpret_cast<std::uintptr_t>(GetShellWindow());
    raw.desktop_window = reinterpret_cast<std::uintptr_t>(GetDesktopWindow());
    raw.is_window = IsWindow(window) != FALSE;
    raw.visible = IsWindowVisible(window) != FALSE;
    raw.minimized = IsIconic(window) != FALSE;

    DWORD process_id = 0;
    raw.process_query_succeeded =
      GetWindowThreadProcessId(window, &process_id) != 0 && process_id != 0;
    raw.process_id = process_id;
    raw.own_process_id = GetCurrentProcessId();

    std::array<wchar_t, 256> class_name {};
    const int class_length = GetClassNameW(
      window,
      class_name.data(),
      static_cast<int>(class_name.size())
    );
    raw.class_query_succeeded = class_length > 0 &&
                                class_length < static_cast<int>(class_name.size());
    if (raw.class_query_succeeded) {
      raw.shell_class = is_shell_class(
        std::wstring_view(class_name.data(), static_cast<std::size_t>(class_length))
      );
    }

    SetLastError(ERROR_SUCCESS);
    const LONG_PTR extended_style = GetWindowLongPtrW(window, GWL_EXSTYLE);
    raw.style_query_succeeded =
      extended_style != 0 || GetLastError() == ERROR_SUCCESS;
    raw.extended_style = static_cast<std::uint64_t>(extended_style);

    DWORD cloaked = 0;
    raw.cloak_query_succeeded = SUCCEEDED(DwmGetWindowAttribute(
      window,
      DWMWA_CLOAKED,
      &cloaked,
      sizeof(cloaked)
    ));
    raw.cloaked = cloaked != 0;

    RECT client {};
    raw.client_rect_succeeded = GetClientRect(window, &client) != FALSE;
    if (raw.client_rect_succeeded) {
      SetLastError(ERROR_SUCCESS);
      raw.client_rect_succeeded =
        MapWindowPoints(
          window,
          nullptr,
          reinterpret_cast<POINT *>(&client),
          2
        ) != 0 ||
        GetLastError() == ERROR_SUCCESS;
      if (raw.client_rect_succeeded) {
        raw.client_screen_rect = {client.left, client.top, client.right, client.bottom};
      }
    }

    RECT frame {};
    raw.frame_rect_succeeded = SUCCEEDED(DwmGetWindowAttribute(
      window,
      DWMWA_EXTENDED_FRAME_BOUNDS,
      &frame,
      sizeof(frame)
    ));
    if (raw.frame_rect_succeeded) {
      raw.frame_screen_rect = {frame.left, frame.top, frame.right, frame.bottom};
    }

    if (raw.client_screen_rect.valid()) {
      raw.monitor = reinterpret_cast<std::uintptr_t>(MonitorFromRect(
        &client,
        MONITOR_DEFAULTTONULL
      ));
    }

    const HWND foreground_after = GetForegroundWindow();
    const HWND window_after = foreground_after ? GetAncestor(foreground_after, GA_ROOT) : nullptr;
    raw.window_after = reinterpret_cast<std::uintptr_t>(window_after);
    raw.is_window_after = window_after && IsWindow(window_after) != FALSE;
    DWORD process_id_after = 0;
    raw.process_query_after_succeeded =
      window_after && GetWindowThreadProcessId(window_after, &process_id_after) != 0 &&
      process_id_after != 0;
    raw.process_id_after = process_id_after;

    return detail::classify(raw, std::chrono::steady_clock::now());
  }

  snapshot_t continuity_tracker_t::update(const observation_t &observation) noexcept {
    if (!observation_has_complete_geometry(observation)) {
      key_.reset();
      geometry_valid_since_ = {};
      return {
        .status = observation.status,
        .observed_at = observation.observed_at,
      };
    }

    const key_t next_key {
      .status = observation.status,
      .window = observation.window,
      .process_id = observation.process_id,
      .monitor = observation.monitor,
      .client_screen_rect = observation.client_screen_rect,
      .frame_screen_rect = observation.frame_screen_rect,
    };
    if (
      !key_ || *key_ != next_key ||
      observation.observed_at < geometry_valid_since_
    ) {
      key_ = next_key;
      geometry_valid_since_ = observation.observed_at;
      if (++generation_ == 0) {
        ++generation_;
      }
    }

    return {
      .status = observation.status,
      .generation = generation_,
      .window = observation.window,
      .process_id = observation.process_id,
      .monitor = observation.monitor,
      .client_screen_rect = observation.client_screen_rect,
      .frame_screen_rect = observation.frame_screen_rect,
      .observed_at = observation.observed_at,
      .geometry_valid_since = geometry_valid_since_,
    };
  }

  void continuity_tracker_t::reset() noexcept {
    key_.reset();
    geometry_valid_since_ = {};
  }

  bool usable_for_content(
    const snapshot_t &snapshot,
    const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
    const std::chrono::steady_clock::time_point capture_now,
    const std::chrono::milliseconds maximum_age
  ) noexcept {
    return snapshot_has_complete_geometry(snapshot) && content_timestamp &&
           maximum_age.count() >= 0 && capture_now >= snapshot.observed_at &&
           capture_now - snapshot.observed_at <= maximum_age &&
           *content_timestamp >= snapshot.geometry_valid_since &&
           *content_timestamp <= capture_now;
  }

  const char *mapping_status_name(const mapping_status_e status) noexcept {
    switch (status) {
      case mapping_status_e::ok:
        return "ok";
      case mapping_status_e::invalid_observation:
        return "invalid-observation";
      case mapping_status_e::invalid_capture_target:
        return "invalid-capture-target";
      case mapping_status_e::unsupported_orientation:
        return "unsupported-orientation";
      case mapping_status_e::monitor_mismatch:
        return "monitor-mismatch";
      case mapping_status_e::outside_capture:
        return "outside-capture";
      case mapping_status_e::partially_outside_capture:
        return "partially-outside-capture";
    }
    return "unknown";
  }

  mapping_result_t map_to_capture(
    const snapshot_t &snapshot,
    const capture_target_t &target
  ) noexcept {
    if (!snapshot_has_complete_geometry(snapshot)) {
      return {.status = mapping_status_e::invalid_observation};
    }
    const auto target_width = static_cast<std::int64_t>(target.screen_rect.right) -
                              target.screen_rect.left;
    const auto target_height = static_cast<std::int64_t>(target.screen_rect.bottom) -
                               target.screen_rect.top;
    if (
      !target.screen_rect.valid() || target.width == 0 || target.height == 0 ||
      target.monitor == 0 || target.width > std::numeric_limits<std::int32_t>::max() ||
      target.height > std::numeric_limits<std::int32_t>::max() ||
      target_width != target.width || target_height != target.height
    ) {
      return {.status = mapping_status_e::invalid_capture_target};
    }
    if (!target.identity_orientation) {
      return {.status = mapping_status_e::unsupported_orientation};
    }
    if (snapshot.monitor != target.monitor) {
      return {.status = mapping_status_e::monitor_mismatch};
    }
    if (!contains(target.screen_rect, snapshot.client_screen_rect)) {
      return {
        .status = intersects(target.screen_rect, snapshot.client_screen_rect) ?
                    mapping_status_e::partially_outside_capture :
                    mapping_status_e::outside_capture,
      };
    }

    const rect_t capture_pixels {
      snapshot.client_screen_rect.left - target.screen_rect.left,
      snapshot.client_screen_rect.top - target.screen_rect.top,
      snapshot.client_screen_rect.right - target.screen_rect.left,
      snapshot.client_screen_rect.bottom - target.screen_rect.top,
    };
    const bool full_capture = snapshot.client_screen_rect == target.screen_rect;
    return {
      .status = mapping_status_e::ok,
      .route = full_capture ? route_e::full_capture : route_e::roi,
      .capture_pixels = capture_pixels,
    };
  }
}  // namespace platf::foreground_window
