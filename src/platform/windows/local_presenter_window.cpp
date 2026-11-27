/**
 * @file src/platform/windows/local_presenter_window.cpp
 * @brief Local presenter window, pointer isolation and display identity helpers.
 */
#include "local_presenter_window.h"

// standard includes
#include <array>
#include <cstdint>

// platform includes
#include <dwmapi.h>

// lib includes
#include <boost/algorithm/string/predicate.hpp>

// local includes
#include "display_config.h"
#include "utils.h"

namespace platf::dxgi {
  namespace local_presenter {
    void log_local_shutter_window_snapshot(HWND presenter, RECT target_rect, HMONITOR expected_monitor) {
      struct properties_t {
        unsigned scanned = 0;
        unsigned handoff = 0;
        bool limit_reached = false;
      };

      struct window_t {
        HWND hwnd {};
        DWORD pid = 0;
        DWORD tid = 0;
        std::wstring class_name;
        std::wstring process_name;
        LONG_PTR style = 0;
        LONG_PTR exstyle = 0;
        BOOL visible = FALSE;
        RECT raw_rect {};
        BOOL raw_rect_ok = FALSE;
        RECT physical_frame {};
        HRESULT frame_result = E_INVALIDARG;
        DWORD cloaked = 0;
        HRESULT cloaked_result = E_INVALIDARG;
        HMONITOR monitor {};
        DWORD_PTR title_length = 0;
        bool title_length_ok = false;
        properties_t properties;
        int properties_result = -1;
      };

      const auto counter = []() {
        LARGE_INTEGER value {};
        QueryPerformanceCounter(&value);
        return value.QuadPart;
      };
      const auto snapshot_start_qpc = counter();
      const POINT center {
        static_cast<LONG>((static_cast<std::int64_t>(target_rect.left) + target_rect.right) / 2),
        static_cast<LONG>((static_cast<std::int64_t>(target_rect.top) + target_rect.bottom) / 2),
      };
      const auto hit = WindowFromPhysicalPoint(center);
      const std::array<HWND, 4> handles {presenter, hit, GetAncestor(hit, GA_ROOT), GetForegroundWindow()};
      std::array<window_t, 4> windows;
      for (std::size_t index = 0; index < handles.size(); ++index) {
        auto &window = windows[index];
        window.hwnd = handles[index];
        if (!window.hwnd) {
          continue;
        }
        const auto previous = std::find(handles.begin(), handles.begin() + index, window.hwnd);
        if (previous != handles.begin() + index) {
          window = windows[static_cast<std::size_t>(previous - handles.begin())];
          continue;
        }
        window.tid = GetWindowThreadProcessId(window.hwnd, &window.pid);
        wchar_t name[256] {};
        GetClassNameW(window.hwnd, name, static_cast<int>(std::size(name)));
        window.class_name = name;
        window.style = GetWindowLongPtrW(window.hwnd, GWL_STYLE);
        window.exstyle = GetWindowLongPtrW(window.hwnd, GWL_EXSTYLE);
        window.visible = IsWindowVisible(window.hwnd);
        window.raw_rect_ok = GetWindowRect(window.hwnd, &window.raw_rect);
        // DWM's extended-frame bounds are physical pixels; GetWindowRect can be DPI virtualized.
        window.frame_result = DwmGetWindowAttribute(window.hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, &window.physical_frame, sizeof(window.physical_frame));
        window.cloaked_result = DwmGetWindowAttribute(window.hwnd, DWMWA_CLOAKED, &window.cloaked, sizeof(window.cloaked));
        window.monitor = MonitorFromWindow(window.hwnd, MONITOR_DEFAULTTONULL);
        window.title_length_ok = SendMessageTimeoutW(window.hwnd, WM_GETTEXTLENGTH, 0, 0, SMTO_ABORTIFHUNG | SMTO_BLOCK, 10, &window.title_length) != 0;
        const auto process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, window.pid);
        if (process) {
          wchar_t path[1024] {};
          DWORD count = static_cast<DWORD>(std::size(path));
          if (QueryFullProcessImageNameW(process, 0, path, &count)) {
            const std::wstring_view image(path, count);
            const auto separator = image.find_last_of(L"\\/");
            window.process_name = image.substr(separator == std::wstring_view::npos ? 0 : separator + 1);
          }
          CloseHandle(process);
        }
        window.properties_result = EnumPropsExW(window.hwnd, [](HWND, LPWSTR key, HANDLE, ULONG_PTR parameter) -> BOOL {
          auto &properties = *reinterpret_cast<properties_t *>(parameter);
          wchar_t atom_name[256] {};
          if (IS_INTRESOURCE(key)) {
            if (GlobalGetAtomNameW(static_cast<ATOM>(reinterpret_cast<ULONG_PTR>(key)), atom_name, static_cast<int>(std::size(atom_name)))) {
              key = atom_name;
            } else {
              key = nullptr;
            }
          }
          constexpr std::wstring_view prefix = L"Sunshine3D.MonitorDesktopHandoff.";
          if (key && std::wstring_view(key).starts_with(prefix)) {
            ++properties.handoff;
          }
          if (++properties.scanned >= 128) {
            properties.limit_reached = true;
            return FALSE;
          }
          return TRUE;
        },
                     reinterpret_cast<ULONG_PTR>(&window.properties));
      }
      const auto desktop_name = [](HDESK desktop) {
        wchar_t name[256] {};
        DWORD needed = 0;
        return desktop && GetUserObjectInformationW(desktop, UOI_NAME, name, sizeof(name), &needed) ? std::wstring(name) : std::wstring {};
      };
      const auto input_desktop = OpenInputDesktop(0, FALSE, DESKTOP_READOBJECTS);
      const auto input_name = desktop_name(input_desktop);
      if (input_desktop) {
        CloseDesktop(input_desktop);
      }
      const auto thread_name = desktop_name(GetThreadDesktop(GetCurrentThreadId()));
      const auto locked = query_console_session_locked();
      const auto snapshot_end_qpc = counter();
      BOOST_LOG(error) << "Local shutter failure visibility: snapshot_qpc=["sv << snapshot_start_qpc << ',' << snapshot_end_qpc
                       << "] target_rect=["sv << target_rect.left << ',' << target_rect.top << ',' << target_rect.right << ',' << target_rect.bottom
                       << "] expected_monitor="sv << expected_monitor << " center=["sv << center.x << ',' << center.y
                       << "] input_desktop="sv << platf::to_utf8(input_name) << " thread_desktop="sv << platf::to_utf8(thread_name)
                       << " wts_locked="sv << (locked ? (*locked ? "true" : "false") : "unknown") << '.';
      constexpr std::array roles {"presenter", "tv_center", "tv_center_root", "foreground"};
      for (std::size_t index = 0; index < windows.size(); ++index) {
        const auto &window = windows[index];
        BOOST_LOG(error) << "Local shutter failure window: role="sv << roles[index] << " hwnd="sv << window.hwnd
                         << " pid="sv << window.pid << " tid="sv << window.tid << " class="sv << platf::to_utf8(window.class_name)
                         << " process="sv << platf::to_utf8(window.process_name)
                         << " style="sv << util::log_hex(window.style) << " exstyle="sv << util::log_hex(window.exstyle)
                         << " visible="sv << window.visible << " cloaked="sv << window.cloaked << " cloaked_hr="sv << util::log_hex(window.cloaked_result)
                         << " physical_frame=["sv << window.physical_frame.left << ',' << window.physical_frame.top << ',' << window.physical_frame.right << ',' << window.physical_frame.bottom
                         << "] frame_hr="sv << util::log_hex(window.frame_result)
                         << " raw_window_rect=["sv << window.raw_rect.left << ',' << window.raw_rect.top << ',' << window.raw_rect.right << ',' << window.raw_rect.bottom
                         << "] raw_rect_ok="sv << window.raw_rect_ok << " monitor="sv << window.monitor
                         << " title_length="sv << window.title_length << " title_length_ok="sv << window.title_length_ok
                         << " handoff_property_present="sv << (window.properties.handoff != 0)
                         << " handoff_properties="sv << window.properties.handoff << " properties_scanned="sv << window.properties.scanned
                         << " property_limit_reached="sv << window.properties.limit_reached << " property_enum_result="sv << window.properties_result << '.';
      }
    }

    bool same_local_presenter_rect(const RECT &left, const RECT &right) {
      return left.left == right.left && left.top == right.top &&
             left.right == right.right && left.bottom == right.bottom;
    }

    bool valid_local_presenter_rect(const RECT &rect) {
      return rect.right > rect.left && rect.bottom > rect.top;
    }

    RECT local_presenter_virtual_screen_rect() {
      const LONG left = GetSystemMetrics(SM_XVIRTUALSCREEN);
      const LONG top = GetSystemMetrics(SM_YVIRTUALSCREEN);
      return {
        left,
        top,
        left + GetSystemMetrics(SM_CXVIRTUALSCREEN),
        top + GetSystemMetrics(SM_CYVIRTUALSCREEN),
      };
    }

    std::optional<RECT> intersect_local_presenter_rects(const RECT &left, const RECT &right) {
      RECT intersection {
        std::max(left.left, right.left),
        std::max(left.top, right.top),
        std::min(left.right, right.right),
        std::min(left.bottom, right.bottom),
      };
      if (!valid_local_presenter_rect(intersection)) {
        return std::nullopt;
      }
      return intersection;
    }

    std::optional<RECT> query_local_presenter_cursor_clip(int attempts) {
      RECT clip {};
      for (int attempt = 0; attempt < std::max(1, attempts); ++attempt) {
        if (GetClipCursor(&clip)) {
          return clip;
        }
        if (attempt + 1 < attempts) {
          Sleep(2);
        }
      }
      return std::nullopt;
    }

    deferred_cursor_restore_manager_t &deferred_cursor_restore_manager() {
      static deferred_cursor_restore_manager_t manager;
      return manager;
    }

    void defer_cursor_retry(local_presenter_cursor_clip_t &clip) {
      const unsigned shift = std::min(clip.retry_failures, 6u);
      const auto delay = std::min(20ms * (1u << shift), 1000ms);
      clip.retry_failures = std::min(clip.retry_failures + 1, 16u);
      clip.retry_after = std::chrono::steady_clock::now() + delay;
    }

    void clear_cursor_retry(local_presenter_cursor_clip_t &clip) {
      clip.retry_failures = 0;
      clip.retry_after = {};
    }

    std::optional<local_presenter_display_identity_t> query_local_presenter_display_identity(
      std::wstring_view display_name
    ) {
      if (display_name.empty()) {
        return std::nullopt;
      }
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      if (!platf::display_config::query_display_config(
            QDC_ONLY_ACTIVE_PATHS,
            paths,
            modes,
            {
              4,
              platf::display_config::retry_delay_e::none,
            }
          )) {
        return std::nullopt;
      }
      for (const auto &path : paths) {
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source_name {};
        source_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source_name.header.size = sizeof(source_name);
        source_name.header.adapterId = path.sourceInfo.adapterId;
        source_name.header.id = path.sourceInfo.id;
        if (DisplayConfigGetDeviceInfo(&source_name.header) != ERROR_SUCCESS || !platf::display_config::equal_device_paths(display_name, source_name.viewGdiDeviceName)) {
          continue;
        }

        DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
        target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
        target_name.header.size = sizeof(target_name);
        target_name.header.adapterId = path.targetInfo.adapterId;
        target_name.header.id = path.targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS || !target_name.monitorDevicePath[0]) {
          return std::nullopt;
        }
        return local_presenter_display_identity_t {
          path.targetInfo.adapterId,
          path.targetInfo.id,
          source_name.viewGdiDeviceName,
          target_name.monitorDevicePath,
          target_name.monitorFriendlyDeviceName,
        };
      }
      return std::nullopt;
    }

    bool local_presenter_identity_is_coherent(
      const std::shared_ptr<local_presenter_config_t::target_t> &live_target,
      const local_presenter_display_identity_t &source,
      const local_presenter_display_identity_t &target,
      const LUID &expected_adapter
    ) {
      // The source is a SudoVDA output: its CCD path lives on the IddCx adapter, whose LUID is
      // not the render GPU's. Only the physical sink is required to stay on the expected adapter;
      // the source is pinned by its driver-learned device path and Apollo-owned friendly name.
      if (!live_target || source.device_path.empty() || target.device_path.empty() || platf::display_config::equal_device_paths(source.device_path, target.device_path) || !platf::display_config::same_luid(target.adapter_id, expected_adapter) || !boost::algorithm::istarts_with(source.friendly_name, std::wstring(L"Apollo AR Des"))) {
        return false;
      }
      std::lock_guard lock(live_target->mutex);
      return !live_target->source_device_path.empty() &&
             !live_target->target_device_path.empty() &&
             platf::display_config::equal_device_paths(live_target->source_device_path, source.device_path) &&
             platf::display_config::equal_device_paths(live_target->target_device_path, target.device_path);
    }

    std::optional<RECT> query_local_presenter_display_rect(std::wstring_view display_name) {
      if (display_name.empty()) {
        return std::nullopt;
      }

      DEVMODEW mode {};
      mode.dmSize = sizeof(mode);
      const std::wstring owned_name {display_name};
      if (!EnumDisplaySettingsExW(owned_name.c_str(), ENUM_CURRENT_SETTINGS, &mode, 0)) {
        return std::nullopt;
      }

      RECT rect {
        mode.dmPosition.x,
        mode.dmPosition.y,
        mode.dmPosition.x + static_cast<LONG>(mode.dmPelsWidth),
        mode.dmPosition.y + static_cast<LONG>(mode.dmPelsHeight),
      };
      if (!valid_local_presenter_rect(rect)) {
        return std::nullopt;
      }
      return rect;
    }

    LRESULT CALLBACK local_presenter_window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
      if (message == WM_NCCREATE) {
        const auto *create = reinterpret_cast<const CREATESTRUCTW *>(lparam);
        SetWindowLongPtrW(
          hwnd,
          GWLP_USERDATA,
          reinterpret_cast<LONG_PTR>(create->lpCreateParams)
        );
      }
      auto *pointer_isolation = reinterpret_cast<local_presenter_pointer_isolation_t *>(
        GetWindowLongPtrW(hwnd, GWLP_USERDATA)
      );
      switch (message) {
        case WM_CLOSE:
          DestroyWindow(hwnd);
          return 0;
        case WM_DESTROY:
          PostQuitMessage(0);
          return 0;
        case WM_NCHITTEST:
          // ClipCursor is shared state and another application may replace it. Clamp at the first
          // presenter hit-test as an event-driven fallback, before a click can reach the sink.
          if (pointer_isolation) {
            pointer_isolation->refresh_clip();
            pointer_isolation->clamp_pointer_to_source_edge();
          }
          return HTCLIENT;
        case WM_MOUSEACTIVATE:
          return MA_NOACTIVATEANDEAT;
        case WM_MOUSEMOVE:
          if (pointer_isolation) {
            pointer_isolation->refresh_clip();
            pointer_isolation->clamp_pointer_to_source_edge();
          }
          return 0;
        case WM_DISPLAYCHANGE:
        case WM_SETTINGCHANGE:
        case WM_WINDOWPOSCHANGED:
          if (pointer_isolation) {
            pointer_isolation->refresh_clip();
            if (pointer_isolation->park_only) {
              pointer_isolation->refresh_hook_sink();
            }
          }
          break;
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
        case WM_XBUTTONUP:
          // If an application replaced ClipCursor during a drag, swallowing its release on the
          // presentation sink could leave the source application logically pressed. Cancel that
          // interaction explicitly before consuming the sink event.
          if (pointer_isolation) {
            pointer_isolation->cancel_active_pointer_interaction(hwnd);
            pointer_isolation->refresh_clip();
            pointer_isolation->clamp_pointer_to_source_edge();
          }
          return 0;
        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
        case WM_XBUTTONDOWN:
        case WM_MOUSEWHEEL:
        case WM_MOUSEHWHEEL:
          // The sink is presentation-only. If cursor confinement was replaced between movement
          // and this queued action, clamp it back and consume the sink-targeted action.
          if (pointer_isolation) {
            pointer_isolation->refresh_clip();
            pointer_isolation->clamp_pointer_to_source_edge();
          }
          return 0;
        default:
          break;
      }
      return DefWindowProcW(hwnd, message, wparam, lparam);
    }

    std::atomic<const local_presenter_pointer_isolation_t *> hooked_pointer_isolation {nullptr};

    LRESULT CALLBACK local_presenter_mouse_hook(int code, WPARAM wparam, LPARAM lparam) {
      if (code == HC_ACTION && wparam == WM_MOUSEMOVE) {
        const auto *isolation = hooked_pointer_isolation.load();
        const auto *info = reinterpret_cast<const MSLLHOOKSTRUCT *>(lparam);
        if (isolation && info && isolation->refuse_pointer_move(info->pt)) {
          return 1;
        }
      }
      return CallNextHookEx(nullptr, code, wparam, lparam);
    }

    bool register_local_presenter_window_class() {
      WNDCLASSEXW window_class {};
      window_class.cbSize = sizeof(window_class);
      window_class.lpfnWndProc = local_presenter_window_proc;
      window_class.hInstance = GetModuleHandleW(nullptr);
      window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
      window_class.lpszClassName = local_presenter_window_class;
      if (RegisterClassExW(&window_class)) {
        return true;
      }
      return GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
    }
  }  // namespace local_presenter

  using namespace local_presenter;

  bool local_presenter_cursor_clip_t::restore() {
    if (!owns_clip) {
      clip_saved = false;
      clip_yielded = false;
      previous_clip_unbounded = false;
      clear_cursor_retry(*this);
      return true;
    }

    const auto current_clip = query_local_presenter_cursor_clip(3);
    if (current_clip && same_local_presenter_rect(*current_clip, owned_clip)) {
      const BOOL restored = previous_clip_unbounded ? ClipCursor(nullptr) : ClipCursor(&previous_clip);
      if (!restored) {
        BOOST_LOG(warning) << "Local AR presenter could not restore the previous cursor clip: "sv
                           << GetLastError();
        return false;
      }
    } else if (!current_clip) {
      return false;
    }
    owns_clip = false;
    clip_saved = false;
    clip_yielded = false;
    previous_clip_unbounded = false;
    clear_cursor_retry(*this);
    return true;
  }

  local_presenter_cursor_clip_t::~local_presenter_cursor_clip_t() {
    for (int attempt = 0; attempt < 8; ++attempt) {
      if (restore()) {
        return;
      }
      Sleep(5);
    }
    BOOST_LOG(warning) << "Local AR presenter could not verify and restore cursor confinement after bounded retries; "sv
                          "deferring ownership-safe restoration until the desktop topology settles."sv;
    deferred_cursor_restore_manager().defer({previous_clip, owned_clip, previous_clip_unbounded});
    owns_clip = false;
  }

  void refresh_local_presenter_pointer_isolation(
    const std::string &source_display_name,
    const std::shared_ptr<local_presenter_config_t::target_t> &live_target,
    const std::shared_ptr<local_presenter_cursor_clip_t> &cursor_clip,
    bool frame_sequential
  ) {
    if (source_display_name.empty() || !live_target || (!cursor_clip && !frame_sequential)) {
      return;
    }
    RECT target_rect {};
    {
      std::lock_guard lock(live_target->mutex);
      target_rect = live_target->rect;
    }
    const auto source_name_wide = platf::from_utf8(source_display_name);
    local_presenter_pointer_isolation_t pointer_isolation {
      source_name_wide,
      query_local_presenter_display_rect(source_name_wide).value_or(RECT {}),
      target_rect,
      live_target,
      cursor_clip,
      frame_sequential,
    };
    pointer_isolation.refresh_clip();
    if (frame_sequential) {
      pointer_isolation.clamp_pointer_to_source_edge();
    }
  }
}  // namespace platf::dxgi
