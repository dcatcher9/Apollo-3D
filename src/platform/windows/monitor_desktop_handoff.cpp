#include "monitor_desktop_handoff.h"

#include "misc.h"
#include "src/logging.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cwctype>
#include <dwmapi.h>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace platf {
  namespace monitor_desktop_detail {
    bool is_security_surface(std::wstring_view window_class, std::wstring_view process_image) {
      const auto same_name = [](std::wstring_view left, std::wstring_view right) {
        return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(), [](wchar_t a, wchar_t b) {
          return std::towlower(a) == std::towlower(b);
        });
      };
      const auto separator = process_image.find_last_of(L"\\/");
      const auto process_name = process_image.substr(separator == std::wstring_view::npos ? 0 : separator + 1);
      return same_name(window_class, L"LockScreenInputOcclusionFrame") || same_name(process_name, L"LockApp.exe");
    }

    bool valid_rect(RECT rect) {
      const auto width = static_cast<std::int64_t>(rect.right) - rect.left;
      const auto height = static_cast<std::int64_t>(rect.bottom) - rect.top;
      return width > 0 && height > 0 && width <= std::numeric_limits<int>::max() && height <= std::numeric_limits<int>::max();
    }

    bool contains(RECT rect, POINT point) {
      return valid_rect(rect) && point.x >= rect.left && point.x < rect.right && point.y >= rect.top && point.y < rect.bottom;
    }

    bool contains_center(RECT monitor, RECT window) {
      if (!valid_rect(window) || !valid_rect(monitor)) {
        return false;
      }
      const auto twice_x = static_cast<std::int64_t>(window.left) + window.right;
      const auto twice_y = static_cast<std::int64_t>(window.top) + window.bottom;
      return twice_x >= static_cast<std::int64_t>(monitor.left) * 2 && twice_x < static_cast<std::int64_t>(monitor.right) * 2 &&
             twice_y >= static_cast<std::int64_t>(monitor.top) * 2 && twice_y < static_cast<std::int64_t>(monitor.bottom) * 2;
    }

    std::optional<RECT> map_rect(RECT window, RECT from, RECT to) {
      if (!valid_rect(window) || !valid_rect(from) || !valid_rect(to)) {
        return std::nullopt;
      }
      if (from.left == to.left && from.top == to.top && from.right == to.right && from.bottom == to.bottom) {
        // Monitor replacement keeps the same logical rectangle. Preserve windows spanning an
        // unchanged neighbor rather than resizing/recentering them into this one output.
        return window;
      }
      const auto width = std::min(static_cast<std::int64_t>(window.right) - window.left, static_cast<std::int64_t>(to.right) - to.left);
      const auto height = std::min(static_cast<std::int64_t>(window.bottom) - window.top, static_cast<std::int64_t>(to.bottom) - to.top);
      const auto left = std::clamp(static_cast<std::int64_t>(to.left) + window.left - from.left, static_cast<std::int64_t>(to.left), static_cast<std::int64_t>(to.right) - width);
      const auto top = std::clamp(static_cast<std::int64_t>(to.top) + window.top - from.top, static_cast<std::int64_t>(to.top), static_cast<std::int64_t>(to.bottom) - height);
      return RECT {static_cast<LONG>(left), static_cast<LONG>(top), static_cast<LONG>(left + width), static_cast<LONG>(top + height)};
    }

    std::optional<POINT> map_point(POINT point, RECT from, RECT to) {
      if (!valid_rect(from) || !valid_rect(to)) {
        return std::nullopt;
      }
      const auto x = std::clamp(static_cast<std::int64_t>(to.left) + point.x - from.left, static_cast<std::int64_t>(to.left), static_cast<std::int64_t>(to.right) - 1);
      const auto y = std::clamp(static_cast<std::int64_t>(to.top) + point.y - from.top, static_cast<std::int64_t>(to.top), static_cast<std::int64_t>(to.bottom) - 1);
      return POINT {static_cast<LONG>(x), static_cast<LONG>(y)};
    }

    std::optional<RECT> window_ownership_rect(RECT screen_rect, RECT normal_rect, bool minimized) {
      const auto selected = minimized ? normal_rect : screen_rect;
      return valid_rect(selected) ? std::optional<RECT> {selected} : std::nullopt;
    }

    std::optional<WINDOWPLACEMENT> placement_for_move(WINDOWPLACEMENT placement, RECT normal_workspace_rect, bool minimized, bool visible) {
      if (!valid_rect(placement.rcNormalPosition) || !valid_rect(normal_workspace_rect)) {
        return std::nullopt;
      }
      const auto dx = static_cast<std::int64_t>(normal_workspace_rect.left) - placement.rcNormalPosition.left;
      const auto dy = static_cast<std::int64_t>(normal_workspace_rect.top) - placement.rcNormalPosition.top;
      const auto translate_point = [dx, dy](POINT point) {
        const auto clamp = [](std::int64_t value) {
          return static_cast<LONG>(std::clamp(value, static_cast<std::int64_t>(std::numeric_limits<LONG>::min()), static_cast<std::int64_t>(std::numeric_limits<LONG>::max())));
        };
        return POINT {clamp(point.x + dx), clamp(point.y + dy)};
      };
      placement.length = sizeof(WINDOWPLACEMENT);
      // Keep WPF_RESTORETOMAXIMIZED/WPF_SETMINPOSITION rather than discarding their restoration
      // semantics when adding asynchronous delivery to another application's UI thread.
      placement.flags |= WPF_ASYNCWINDOWPLACEMENT;
      if (placement.ptMaxPosition.x != -1 || placement.ptMaxPosition.y != -1) {
        placement.ptMaxPosition = translate_point(placement.ptMaxPosition);
      }
      if (placement.flags & WPF_SETMINPOSITION) {
        placement.ptMinPosition = translate_point(placement.ptMinPosition);
      }
      placement.rcNormalPosition = normal_workspace_rect;
      if (!visible) {
        placement.showCmd = SW_HIDE;
      } else if (minimized) {
        // WPF_RESTORETOMAXIMIZED is documented for SW_SHOWMINIMIZED. Retain that combination so
        // handing off an iconic window does not unminimize it or lose its eventual maximization.
        placement.showCmd = SW_SHOWMINIMIZED;
      } else if (placement.showCmd != SW_SHOWMAXIMIZED) {
        placement.showCmd = SW_SHOWNOACTIVATE;
      }
      return placement;
    }
  }  // namespace monitor_desktop_detail

  namespace {
    class dpi_scope_t {
    public:
      dpi_scope_t():
          previous_(SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {}

      ~dpi_scope_t() {
        if (previous_) {
          SetThreadDpiAwarenessContext(previous_);
        }
      }

      bool active() const {
        return previous_ && AreDpiAwarenessContextsEqual(GetThreadDpiAwarenessContext(), DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
      }

    private:
      DPI_AWARENESS_CONTEXT previous_ {};
    };

    struct window_t {
      HWND hwnd {};
      DWORD pid = 0;
      DWORD tid = 0;
      ATOM window_class = 0;
      RECT screen_rect {};
      RECT normal_rect {};
      RECT desktop_rect {};
      WINDOWPLACEMENT placement {sizeof(WINDOWPLACEMENT)};
      bool minimized = false;
      bool visible = true;
      bool moved = false;
      std::optional<RECT> pending_source_normal;
      bool move_posted = false;
    };

    bool security_surface_window(HWND hwnd, std::wstring_view known_class = {}) {
      std::array<wchar_t, 256> name {};
      if (known_class.empty() && GetClassNameW(hwnd, name.data(), static_cast<int>(name.size()))) {
        known_class = name.data();
      }
      if (monitor_desktop_detail::is_security_surface(known_class, {})) {
        return true;
      }
      DWORD pid = 0;
      if (!GetWindowThreadProcessId(hwnd, &pid)) {
        return false;
      }
      const auto process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
      if (!process) {
        return false;
      }
      std::array<wchar_t, 1024> image {};
      DWORD length = static_cast<DWORD>(image.size());
      const bool queried = QueryFullProcessImageNameW(process, 0, image.data(), &length);
      CloseHandle(process);
      // A failed image query alone must not blacklist ordinary games or UWP applications.
      // The explicit lock-screen class above remains excluded independently of process access.
      return queried && monitor_desktop_detail::is_security_surface(known_class, {image.data(), length});
    }

    bool eligible_window(HWND hwnd) {
      if (!IsWindow(hwnd) || !IsWindowVisible(hwnd) || hwnd == GetShellWindow() || hwnd == GetDesktopWindow() || GetAncestor(hwnd, GA_ROOT) != hwnd) {
        return false;
      }
      const auto style = GetWindowLongPtrW(hwnd, GWL_STYLE);
      const auto extended_style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
      if ((style & WS_CHILD) || (extended_style & (WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE))) {
        return false;
      }
      DWORD pid = 0;
      if (!GetWindowThreadProcessId(hwnd, &pid) || pid == GetCurrentProcessId()) {
        return false;
      }
      DWORD cloaked = 0;
      if (SUCCEEDED(DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &cloaked, sizeof(cloaked))) && cloaked) {
        return false;
      }
      std::array<wchar_t, 256> name {};
      GetClassNameW(hwnd, name.data(), static_cast<int>(name.size()));
      const std::wstring_view class_name {name.data()};
      for (const auto *excluded : {L"Progman", L"WorkerW", L"Shell_TrayWnd", L"Shell_SecondaryTrayWnd", L"ApolloLocalArPresenter", L"DV2ControlHost", L"MultitaskingViewFrame"}) {
        if (class_name == excluded) {
          return false;
        }
      }
      // Normal application windows and borderless/fullscreen applications have a title or an
      // explicit app-window style. This leaves untitled shell overlays and menus in place.
      const bool application = (style & WS_CAPTION) || (extended_style & WS_EX_APPWINDOW) || GetWindowTextLengthW(hwnd) > 0;
      return application && !security_surface_window(hwnd, class_name);
    }

    std::optional<POINT> workspace_offset(HWND hwnd, RECT screen_rect) {
      if (GetWindowLongPtrW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW) {
        return POINT {};
      }
      MONITORINFO monitor {sizeof(MONITORINFO)};
      if (!GetMonitorInfoW(MonitorFromRect(&screen_rect, MONITOR_DEFAULTTONEAREST), &monitor)) {
        return std::nullopt;
      }
      return POINT {monitor.rcWork.left - monitor.rcMonitor.left, monitor.rcWork.top - monitor.rcMonitor.top};
    }

    std::optional<POINT> observed_workspace_offset(HWND hwnd) {
      MONITORINFO monitor {sizeof(MONITORINFO)};
      // MonitorFromWindow uses the pre-minimized window rectangle for iconic windows. Their
      // GetWindowRect result can be the shell's -32000 icon position on no actual output.
      if (!GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &monitor)) {
        return std::nullopt;
      }
      return POINT {monitor.rcWork.left - monitor.rcMonitor.left, monitor.rcWork.top - monitor.rcMonitor.top};
    }

    RECT translated(RECT rect, POINT offset, bool subtract = false) {
      const auto sign = subtract ? -1LL : 1LL;
      const auto clamp = [](std::int64_t value) {
        return static_cast<LONG>(std::clamp(value, static_cast<std::int64_t>(std::numeric_limits<LONG>::min()), static_cast<std::int64_t>(std::numeric_limits<LONG>::max())));
      };
      return {
        clamp(rect.left + sign * offset.x),
        clamp(rect.top + sign * offset.y),
        clamp(rect.right + sign * offset.x),
        clamp(rect.bottom + sign * offset.y),
      };
    }

    std::optional<window_t> observe_window(HWND hwnd, bool require_eligible = true) {
      window_t result;
      result.hwnd = hwnd;
      result.tid = GetWindowThreadProcessId(hwnd, &result.pid);
      result.window_class = static_cast<ATOM>(GetClassLongPtrW(hwnd, GCW_ATOM));
      if (!result.tid || (require_eligible && !eligible_window(hwnd)) || !GetWindowRect(hwnd, &result.screen_rect) || !GetWindowPlacement(hwnd, &result.placement)) {
        return std::nullopt;
      }
      result.minimized = IsIconic(hwnd);
      result.visible = IsWindowVisible(hwnd);
      if (result.minimized && (GetWindowLongPtrW(hwnd, GWL_STYLE) & WS_MAXIMIZE)) {
        result.placement.flags |= WPF_RESTORETOMAXIMIZED;
      }
      const auto offset = observed_workspace_offset(hwnd);
      if (!offset) {
        return std::nullopt;
      }
      result.normal_rect = translated(result.placement.rcNormalPosition, *offset);
      if (!monitor_desktop_detail::valid_rect(result.normal_rect)) {
        return std::nullopt;
      }
      const auto desktop_rect = monitor_desktop_detail::window_ownership_rect(result.screen_rect, result.normal_rect, result.minimized || !result.visible);
      if (!desktop_rect) {
        return std::nullopt;
      }
      result.desktop_rect = *desktop_rect;
      return result;
    }

    bool same_identity(const window_t &window) {
      DWORD pid = 0;
      return IsWindow(window.hwnd) && GetWindowThreadProcessId(window.hwnd, &pid) == window.tid && pid == window.pid &&
             static_cast<ATOM>(GetClassLongPtrW(window.hwnd, GCW_ATOM)) == window.window_class;
    }

    bool move_window(const window_t &window, RECT normal_rect, RECT destination, const std::wstring &property, HANDLE token) {
      if (!same_identity(window) || GetPropW(window.hwnd, property.c_str()) != token) {
        return false;
      }
      const auto destination_workspace = workspace_offset(window.hwnd, destination);
      if (!destination_workspace) {
        return false;
      }
      const auto placement = monitor_desktop_detail::placement_for_move(window.placement, translated(normal_rect, *destination_workspace, true), window.minimized, window.visible);
      return placement && !security_surface_window(window.hwnd) && SetWindowPlacement(window.hwnd, &*placement);
    }

    void retain_foreground(HWND foreground) {
      const auto current = GetForegroundWindow();
      if (foreground && IsWindow(foreground) && current != foreground && !security_surface_window(foreground) &&
          (!current || !security_surface_window(current))) {
        SetForegroundWindow(foreground);
      }
    }

    bool same_rect(RECT left, RECT right) {
      return left.left == right.left && left.top == right.top && left.right == right.right && left.bottom == right.bottom;
    }

    // Names a window by class and program only. Titles can hold private content and stay out of the log.
    std::string describe_window(const window_t &window) {
      std::array<wchar_t, 256> class_name {};
      GetClassNameW(window.hwnd, class_name.data(), static_cast<int>(class_name.size()));
      std::wstring program = L"unknown";
      if (const auto process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, window.pid)) {
        std::array<wchar_t, 1024> image {};
        DWORD length = static_cast<DWORD>(image.size());
        if (QueryFullProcessImageNameW(process, 0, image.data(), &length)) {
          const std::wstring_view path {image.data(), length};
          const auto separator = path.find_last_of(L"\\/");
          program = path.substr(separator == std::wstring_view::npos ? 0 : separator + 1);
        }
        CloseHandle(process);
      }
      return "program=" + to_utf8(program) + " class=" + to_utf8(class_name.data()) +
             (window.minimized ? " minimized" : window.placement.showCmd == SW_SHOWMAXIMIZED ? " maximized" : " normal") +
             (IsHungAppWindow(window.hwnd) ? " not-responding" : "");
    }

    bool await_moves(const std::vector<window_t> &windows, RECT destination) {
      // A large desktop with many maximized or busy applications needs more than a few hundred
      // milliseconds to acknowledge a move. The wait ends as soon as every window has arrived.
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds {2000};
      const auto arrived = [&](const auto &window) {
        if (!same_identity(window)) {
          return true;
        }
        const auto current = observe_window(window.hwnd, false);
        return current && monitor_desktop_detail::contains_center(destination, current->desktop_rect) &&
               (!window.visible ? !current->visible : window.minimized ? current->minimized : window.placement.showCmd != SW_SHOWMAXIMIZED || IsZoomed(window.hwnd));
      };
      while (true) {
        const bool complete = std::all_of(windows.begin(), windows.end(), arrived);
        if (complete) {
          return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          for (const auto &window : windows) {
            if (!arrived(window)) {
              const auto text = [](RECT rect) {
                return std::to_string(rect.left) + "," + std::to_string(rect.top) + "," + std::to_string(rect.right) + "," + std::to_string(rect.bottom);
              };
              const auto current = observe_window(window.hwnd, false);
              MONITORINFOEXW monitor {};
              monitor.cbSize = sizeof(monitor);
              GetMonitorInfoW(MonitorFromWindow(window.hwnd, MONITOR_DEFAULTTONEAREST), &monitor);
              BOOST_LOG(warning) << "Monitor desktop handoff: window did not arrive in time: " << describe_window(window)
                                 << " destination=" << text(destination) << " captured_normal=" << text(window.normal_rect)
                                 << (current ? " now_screen=" + text(current->screen_rect) + " now_normal=" + text(current->normal_rect) + " now_owner=" + text(current->desktop_rect) + (current->minimized ? " now_minimized" : "") + (IsZoomed(window.hwnd) ? " now_maximized" : "") : std::string {" now=unobservable"})
                                 << " on=" << to_utf8(monitor.szDevice) << " monitor=" << text(monitor.rcMonitor);
            }
          }
          return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds {10});
      }
    }

    bool private_source_monitor(HMONITOR monitor) {
      MONITORINFOEXW info {};
      info.cbSize = sizeof(info);
      if (!GetMonitorInfoW(monitor, &info)) {
        return false;
      }
      const auto private_name = [](std::wstring value) {
        std::transform(value.begin(), value.end(), value.begin(), [](wchar_t c) {
          return std::towlower(c);
        });
        return value.find(L"sudomaker") != std::wstring::npos || value.find(L"smkd1ce") != std::wstring::npos ||
               value.find(L"apollo ar des") != std::wstring::npos;
      };
      DISPLAY_DEVICEW child {};
      child.cb = sizeof(child);
      if (EnumDisplayDevicesW(info.szDevice, 0, &child, EDD_GET_DEVICE_INTERFACE_NAME) && (private_name(child.DeviceString) || private_name(child.DeviceID))) {
        return true;
      }
      for (DWORD index = 0; index < 128; ++index) {
        DISPLAY_DEVICEW adapter {};
        adapter.cb = sizeof(adapter);
        if (!EnumDisplayDevicesW(nullptr, index, &adapter, 0)) {
          break;
        }
        if (std::wstring_view {adapter.DeviceName} == info.szDevice) {
          return private_name(adapter.DeviceString);
        }
      }
      return false;
    }

    std::wstring monitor_device_path(RECT rect) {
      MONITORINFOEXW info {};
      info.cbSize = sizeof(info);
      if (!GetMonitorInfoW(MonitorFromRect(&rect, MONITOR_DEFAULTTONEAREST), &info)) {
        return {};
      }
      DISPLAY_DEVICEW child {};
      child.cb = sizeof(child);
      return EnumDisplayDevicesW(info.szDevice, 0, &child, EDD_GET_DEVICE_INTERFACE_NAME) ? child.DeviceID : std::wstring {};
    }

    std::optional<RECT> restore_destination(RECT requested) {
      MONITORINFO nearest {sizeof(MONITORINFO)};
      const auto nearest_monitor = MonitorFromRect(&requested, MONITOR_DEFAULTTONEAREST);
      if (GetMonitorInfoW(nearest_monitor, &nearest) && !private_source_monitor(nearest_monitor)) {
        return same_rect(nearest.rcMonitor, requested) ? requested : nearest.rcWork;
      }

      // The old physical target can disappear while the private source occupies its former
      // coordinates. Never choose that soon-to-be-retired source as the fallback destination.
      struct context_t {
        RECT requested;
        std::optional<RECT> best;
        long double distance = std::numeric_limits<long double>::max();
      } context {requested};

      EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR monitor, HDC, LPRECT, LPARAM data) -> BOOL {
        auto &context = *reinterpret_cast<context_t *>(data);
        MONITORINFO info {sizeof(MONITORINFO)};
        if (GetMonitorInfoW(monitor, &info) && !private_source_monitor(monitor)) {
          const auto dx = static_cast<long double>(info.rcMonitor.left) + info.rcMonitor.right - context.requested.left - context.requested.right;
          const auto dy = static_cast<long double>(info.rcMonitor.top) + info.rcMonitor.bottom - context.requested.top - context.requested.bottom;
          const auto distance = dx * dx + dy * dy;
          if (distance < context.distance) {
            context.distance = distance;
            context.best = info.rcWork;
          }
        }
        return TRUE;
      },
                          reinterpret_cast<LPARAM>(&context));
      return context.best;
    }
  }  // namespace

  struct monitor_desktop_handoff_t::impl_t {
    std::wstring property;
    std::wstring original_display_path;
    std::vector<window_t> windows;
    std::vector<window_t> restoring;
    RECT original {};
    RECT source {};
    std::optional<RECT> physical_sink;
    POINT original_cursor {};
    HWND original_foreground {};
    bool cursor_captured = false;
    bool captured = false;
    bool active = false;
    bool restore_prepared = false;
    POINT restore_cursor {};
    bool restore_cursor_captured = false;

    impl_t() {
      static std::atomic<std::uint64_t> generation {0};
      property = L"Sunshine3D.MonitorDesktopHandoff." + std::to_wstring(GetCurrentProcessId()) + L"." + std::to_wstring(++generation);
    }

    HANDLE token() const {
      return reinterpret_cast<HANDLE>(const_cast<impl_t *>(this));
    }

    bool owned(const window_t &window) const {
      return same_identity(window) && GetPropW(window.hwnd, property.c_str()) == token();
    }

    window_t *retain(const window_t &window) {
      if (security_surface_window(window.hwnd)) {
        return nullptr;
      }
      const auto previous = std::find_if(windows.begin(), windows.end(), [&](const auto &entry) {
        return entry.hwnd == window.hwnd;
      });
      if (previous != windows.end()) {
        if (owned(*previous)) {
          return &*previous;
        }
        windows.erase(previous);
      }
      if (!SetPropW(window.hwnd, property.c_str(), token())) {
        return nullptr;
      }
      if (!same_identity(window) || GetPropW(window.hwnd, property.c_str()) != token()) {
        if (GetPropW(window.hwnd, property.c_str()) == token()) {
          RemovePropW(window.hwnd, property.c_str());
        }
        return nullptr;
      }
      windows.push_back(window);
      return &windows.back();
    }

    void discard_security_windows() {
      const auto discard = [&](const window_t &window) {
        if (!same_identity(window) || !security_surface_window(window.hwnd)) {
          return false;
        }
        if (GetPropW(window.hwnd, property.c_str()) == token()) {
          RemovePropW(window.hwnd, property.c_str());
        }
        return true;
      };
      // Previously tagged/frozen entries are not permission to replay a security surface.
      std::erase_if(windows, discard);
      std::erase_if(restoring, discard);
    }

    void release_properties() {
      for (const auto &window : windows) {
        if (GetPropW(window.hwnd, property.c_str()) == token()) {
          RemovePropW(window.hwnd, property.c_str());
        }
      }
      windows.clear();
    }
  };

  monitor_desktop_handoff_t::monitor_desktop_handoff_t():
      impl_(std::make_unique<impl_t>()) {}

  monitor_desktop_handoff_t::~monitor_desktop_handoff_t() {
    if (impl_->active) {
      (void) restore(impl_->original);
    }
    impl_->release_properties();
  }

  bool monitor_desktop_handoff_t::capture(RECT original_monitor_rect) {
    if (impl_->active || !monitor_desktop_detail::valid_rect(original_monitor_rect) || !syncThreadDesktop()) {
      return false;
    }
    dpi_scope_t dpi;
    if (!dpi.active()) {
      return false;
    }
    impl_->release_properties();
    impl_->captured = false;
    impl_->restore_prepared = false;
    impl_->restoring.clear();
    impl_->physical_sink.reset();
    impl_->original = original_monitor_rect;
    impl_->original_display_path = monitor_device_path(original_monitor_rect);
    impl_->original_foreground = GetForegroundWindow();
    impl_->cursor_captured = GetCursorPos(&impl_->original_cursor) && monitor_desktop_detail::contains(original_monitor_rect, impl_->original_cursor);
    const auto success = EnumWindows([](HWND hwnd, LPARAM parameter) -> BOOL {
      auto &self = *reinterpret_cast<impl_t *>(parameter);
      auto window = observe_window(hwnd);
      if (window && monitor_desktop_detail::contains_center(self.original, window->desktop_rect)) {
        if (!SetPropW(hwnd, self.property.c_str(), self.token())) {
          return FALSE;
        }
        if (same_identity(*window)) {
          self.windows.push_back(*window);
        } else if (GetPropW(hwnd, self.property.c_str()) == self.token()) {
          RemovePropW(hwnd, self.property.c_str());
        }
      }
      return TRUE;
    },
                                     reinterpret_cast<LPARAM>(impl_.get()));
    if (!success) {
      impl_->release_properties();
      return false;
    }
    impl_->captured = true;
    return true;
  }

  bool monitor_desktop_handoff_t::activate(RECT virtual_source_rect) {
    if (impl_->active && same_rect(impl_->source, virtual_source_rect)) {
      return true;
    }
    if (!impl_->captured || !monitor_desktop_detail::valid_rect(virtual_source_rect) || !syncThreadDesktop()) {
      return false;
    }
    dpi_scope_t dpi;
    if (!dpi.active()) {
      return false;
    }
    const auto previous_source = impl_->active ? impl_->source : impl_->original;
    const bool first_activation = !impl_->active;
    impl_->source = virtual_source_rect;
    impl_->active = true;  // Keep failure cleanup responsible for every partially moved window.
    impl_->restore_prepared = false;
    impl_->restoring.clear();
    impl_->discard_security_windows();
    const HWND foreground = GetForegroundWindow();
    bool success = true;
    std::vector<window_t> moving;
    for (auto &window : impl_->windows) {
      if (!impl_->owned(window)) {
        continue;
      }
      const auto current = observe_window(window.hwnd);
      if (!current) {
        continue;
      }
      // For the initial handoff Windows may already have remapped windows during topology
      // normalization. Their captured placement remains the authoritative original position.
      // Subsequent source relocation follows only windows still on our previous source.
      if (!first_activation && !monitor_desktop_detail::contains_center(previous_source, current->desktop_rect)) {
        continue;
      }
      if (first_activation && !monitor_desktop_detail::contains_center(virtual_source_rect, current->desktop_rect) && (impl_->original_display_path.empty() || monitor_device_path(current->desktop_rect) != impl_->original_display_path)) {
        // A deliberate move onto another physical monitor since capture relinquishes this window.
        // A same-device Windows topology rebase or its automatic move to the new source does not.
        continue;
      }
      const auto mapped = monitor_desktop_detail::map_rect(first_activation ? window.normal_rect : current->normal_rect, previous_source, virtual_source_rect);
      if (mapped && move_window(first_activation ? window : *current, *mapped, virtual_source_rect, impl_->property, impl_->token())) {
        window.moved = true;
        moving.push_back(first_activation ? window : *current);
      } else {
        BOOST_LOG(warning) << "Monitor desktop handoff: window refused its move: " << describe_window(window);
        success = false;
      }
    }
    POINT cursor {};
    if (GetCursorPos(&cursor)) {
      const auto position = first_activation ? impl_->original_cursor : cursor;
      if ((first_activation && impl_->cursor_captured) || (!first_activation && monitor_desktop_detail::contains(previous_source, cursor))) {
        if (const auto mapped = monitor_desktop_detail::map_point(position, previous_source, virtual_source_rect)) {
          SetCursorPos(mapped->x, mapped->y);
        }
      }
    }
    const bool completed = await_moves(moving, virtual_source_rect);
    retain_foreground(foreground ? foreground : impl_->original_foreground);
    return success && completed;
  }

  bool monitor_desktop_handoff_t::refresh(RECT physical_sink_rect, RECT virtual_source_rect) {
    if (!impl_->active || impl_->restore_prepared || !monitor_desktop_detail::valid_rect(physical_sink_rect) ||
        !monitor_desktop_detail::valid_rect(virtual_source_rect) || same_rect(physical_sink_rect, virtual_source_rect) ||
        !same_rect(virtual_source_rect, impl_->source) || !syncThreadDesktop()) {
      return false;
    }
    dpi_scope_t dpi;
    if (!dpi.active()) {
      return false;
    }
    impl_->physical_sink = physical_sink_rect;
    std::erase_if(impl_->windows, [&](const auto &window) {
      return !impl_->owned(window);
    });

    struct context_t {
      RECT sink;
      RECT source;
      std::vector<window_t> candidates;
    } context {physical_sink_rect, virtual_source_rect};
    bool completed = EnumWindows([](HWND hwnd, LPARAM parameter) -> BOOL {
      auto &context = *reinterpret_cast<context_t *>(parameter);
      if (const auto window = observe_window(hwnd);
          window && monitor_desktop_detail::contains_center(context.sink, window->desktop_rect) &&
          !monitor_desktop_detail::contains_center(context.source, window->desktop_rect)) {
        context.candidates.push_back(*window);
      }
      return TRUE;
    },
                                 reinterpret_cast<LPARAM>(&context));

    const HWND foreground = GetForegroundWindow();
    bool attempted_move = false;
    for (const auto &candidate : context.candidates) {
      if (!same_identity(candidate)) {
        continue;
      }
      const auto current = observe_window(candidate.hwnd);
      if (!current || !monitor_desktop_detail::contains_center(physical_sink_rect, current->desktop_rect) ||
          monitor_desktop_detail::contains_center(virtual_source_rect, current->desktop_rect)) {
        continue;
      }
      auto *window = impl_->retain(*current);
      if (!window) {
        completed = false;
        continue;
      }
      if (window->pending_source_normal && window->move_posted) {
        // One posted placement per window. A hung UI thread must not accumulate another request
        // on every controller tick; acknowledgment is checked below without a blocking wait.
        continue;
      }
      const auto mapped = monitor_desktop_detail::map_rect(current->normal_rect, physical_sink_rect, virtual_source_rect);
      if (!mapped) {
        completed = false;
        continue;
      }
      *window = *current;
      window->pending_source_normal = *mapped;
      attempted_move = true;
      window->move_posted = move_window(*current, *mapped, virtual_source_rect, impl_->property, impl_->token());
      window->moved = window->move_posted;
      completed = window->move_posted && completed;
    }
    for (auto &window : impl_->windows) {
      if (!window.pending_source_normal || !impl_->owned(window)) {
        continue;
      }
      const auto current = observe_window(window.hwnd, false);
      if (!current) {
        completed = false;
        continue;
      }
      if (monitor_desktop_detail::contains_center(virtual_source_rect, current->desktop_rect)) {
        window.pending_source_normal.reset();
        window.move_posted = false;
        window.moved = true;
      } else if (monitor_desktop_detail::contains_center(physical_sink_rect, current->desktop_rect)) {
        completed = false;
      } else {
        // A user/app move to another interactive monitor relinquishes this pending migration.
        // prepare_restore likewise touches only source/sink-owned windows.
        window.pending_source_normal.reset();
        window.move_posted = false;
      }
    }
    if (attempted_move) {
      retain_foreground(foreground);
    }
    return completed;
  }

  void monitor_desktop_handoff_t::prepare_restore() {
    if (!impl_->active || impl_->restore_prepared || !syncThreadDesktop()) {
      return;
    }
    dpi_scope_t dpi;
    if (!dpi.active()) {
      return;
    }
    impl_->discard_security_windows();
    impl_->restoring.clear();
    const bool enumerated = EnumWindows([](HWND hwnd, LPARAM parameter) -> BOOL {
      auto &self = *reinterpret_cast<impl_t *>(parameter);
      const auto previous = std::find_if(self.windows.begin(), self.windows.end(), [&](const auto &window) {
        return window.hwnd == hwnd;
      });
      const bool already_owned = previous != self.windows.end() && self.owned(*previous);
      const auto current = observe_window(hwnd, !already_owned);
      if (!current) {
        // A live tagged window with an unreadable placement may still belong to the source or
        // have a queued sink migration. Keep the whole recovery snapshot retryable.
        return !already_owned;
      }
      const bool on_source = monitor_desktop_detail::contains_center(self.source, current->desktop_rect);
      const bool on_sink = self.physical_sink && monitor_desktop_detail::contains_center(*self.physical_sink, current->desktop_rect);
      if (!on_source && !on_sink) {
        return TRUE;
      }
      if (!self.retain(*current)) {
        return FALSE;
      }
      auto snapshot = *current;
      if (!on_source && on_sink) {
        // A failed/queued refresh still owns this window. Freeze its sink-relative restore
        // position in the same source coordinate domain as acknowledged migrations before CCD
        // restoration can move either output and Windows can normalize windows automatically.
        const auto mapped = monitor_desktop_detail::map_rect(current->normal_rect, *self.physical_sink, self.source);
        if (!mapped) {
          return FALSE;
        }
        snapshot.normal_rect = *mapped;
      }
      self.restoring.push_back(snapshot);
      return TRUE;
    },
                                        reinterpret_cast<LPARAM>(impl_.get()));
    impl_->restore_cursor_captured = GetCursorPos(&impl_->restore_cursor) && monitor_desktop_detail::contains(impl_->source, impl_->restore_cursor);
    impl_->restore_prepared = enumerated;
  }

  bool monitor_desktop_handoff_t::restore(RECT restored_physical_rect) {
    if (!impl_->active) {
      return true;
    }
    if (!monitor_desktop_detail::valid_rect(restored_physical_rect) || !syncThreadDesktop()) {
      return false;
    }
    dpi_scope_t dpi;
    if (!dpi.active()) {
      return false;
    }
    if (!impl_->restore_prepared) {
      prepare_restore();
      if (!impl_->restore_prepared) {
        return false;
      }
    }
    const auto destination = restore_destination(restored_physical_rect);
    if (!destination) {
      BOOST_LOG(warning) << "Monitor desktop handoff has no remaining physical display for window restoration.";
      return false;
    }
    const HWND foreground = GetForegroundWindow();
    impl_->discard_security_windows();
    bool success = true;
    std::vector<window_t> moving;
    for (const auto &window : impl_->restoring) {
      if (!impl_->owned(window)) {
        continue;
      }
      // The pre-topology snapshot identifies source windows even if Windows has already moved
      // them elsewhere after detaching the source. Preserve their latest source-relative layout.
      const auto mapped = monitor_desktop_detail::map_rect(window.normal_rect, impl_->source, *destination);
      if (mapped && move_window(window, *mapped, *destination, impl_->property, impl_->token())) {
        moving.push_back(window);
      } else {
        success = false;
      }
    }
    success = await_moves(moving, *destination) && success;
    POINT cursor {};
    if (impl_->restore_cursor_captured || (GetCursorPos(&cursor) && monitor_desktop_detail::contains(impl_->source, cursor))) {
      if (const auto mapped = monitor_desktop_detail::map_point(impl_->restore_cursor_captured ? impl_->restore_cursor : cursor, impl_->source, *destination)) {
        SetCursorPos(mapped->x, mapped->y);
      }
    }
    retain_foreground(foreground);
    if (!success) {
      BOOST_LOG(warning) << "Some monitor desktop windows did not acknowledge restoration; retaining their handoff snapshot for cleanup retry.";
      return false;
    }
    impl_->active = false;
    impl_->captured = false;
    impl_->release_properties();
    impl_->restoring.clear();
    impl_->restore_prepared = false;
    return true;
  }
}  // namespace platf
