/**
 * @file tools/virtual_desktop_launcher/main.cpp
 * @brief Hidden standard-user routing of native Windows launches onto the remote virtual display.
 */
#ifndef NOMINMAX
  #define NOMINMAX
#endif
#include "launcher_policy.h"
#include "native_shell_item.h"
#include "native_window_policy.h"

#include <array>
#include <atomic>
#include <chrono>
#include <compare>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <dwmapi.h>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shellapi.h>
#include <shlobj.h>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <windows.h>

namespace {
  using clock_t = std::chrono::steady_clock;
  using namespace std::chrono_literals;
  using desktop_launcher::rectangle_t;

  constexpr wchar_t window_class[] = L"Sunshine3DWindowRouter";
  constexpr UINT_PTR poll_timer = 1;
  constexpr UINT input_message = WM_APP + 1;

  class owned_handle_t {
  public:
    owned_handle_t() = default;

    explicit owned_handle_t(HANDLE value):
        value_(value) {}

    ~owned_handle_t() {
      if (*this) {
        CloseHandle(value_);
      }
    }

    owned_handle_t(const owned_handle_t &) = delete;
    owned_handle_t &operator=(const owned_handle_t &) = delete;

    owned_handle_t(owned_handle_t &&other) noexcept:
        value_(std::exchange(other.value_, nullptr)) {}

    owned_handle_t &operator=(owned_handle_t &&other) noexcept {
      if (this != &other) {
        owned_handle_t replacement(std::move(other));
        std::swap(value_, replacement.value_);
      }
      return *this;
    }

    explicit operator bool() const {
      return value_ && value_ != INVALID_HANDLE_VALUE;
    }

    HANDLE get() const {
      return value_;
    }

  private:
    HANDLE value_ = nullptr;
  };

  std::uint64_t file_time(const FILETIME &value) {
    return (static_cast<std::uint64_t>(value.dwHighDateTime) << 32) | value.dwLowDateTime;
  }

  std::optional<std::uint64_t> process_start(HANDLE process) {
    FILETIME created {}, exited {}, kernel {}, user {};
    if (!GetProcessTimes(process, &created, &exited, &kernel, &user)) {
      return std::nullopt;
    }
    return file_time(created);
  }

  rectangle_t rectangle(const RECT &value) {
    return {value.left, value.top, value.right, value.bottom};
  }

  RECT native_rectangle(const rectangle_t &value) {
    return {
      static_cast<LONG>(value.left),
      static_cast<LONG>(value.top),
      static_cast<LONG>(value.right),
      static_cast<LONG>(value.bottom),
    };
  }

  bool default_input_desktop() {
    const auto desktop = OpenInputDesktop(0, FALSE, DESKTOP_READOBJECTS);
    if (!desktop) {
      return false;
    }
    std::array<wchar_t, 128> name {};
    DWORD needed = 0;
    const bool result = GetUserObjectInformationW(
                          desktop,
                          UOI_NAME,
                          name.data(),
                          sizeof(name),
                          &needed
                        ) &&
                        _wcsicmp(name.data(), L"Default") == 0;
    CloseDesktop(desktop);
    return result;
  }

  struct target_t {
    HMONITOR monitor = nullptr;
    MONITORINFOEXW info {};
    LUID adapter {};
    UINT32 target_id = 0;
  };

  /** Resolve the exact device-instance path on every pass; GDI numbering is not identity. */
  std::optional<target_t> resolve_target(const std::wstring &device_path) {
    for (unsigned attempt = 0; attempt < 3; ++attempt) {
      UINT32 path_count = 0;
      UINT32 mode_count = 0;
      constexpr UINT32 flags = QDC_ONLY_ACTIVE_PATHS;
      if (GetDisplayConfigBufferSizes(flags, &path_count, &mode_count) != ERROR_SUCCESS || path_count > 256 || mode_count > 1024) {
        return std::nullopt;
      }
      std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
      std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
      const auto status = QueryDisplayConfig(
        flags,
        &path_count,
        paths.data(),
        &mode_count,
        modes.data(),
        nullptr
      );
      if (status == ERROR_INSUFFICIENT_BUFFER) {
        continue;
      }
      if (status != ERROR_SUCCESS) {
        return std::nullopt;
      }
      paths.resize(path_count);
      std::optional<target_t> found;
      std::wstring source_name;
      for (const auto &path : paths) {
        DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
        target_name.header = {
          DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME,
          sizeof(target_name),
          path.targetInfo.adapterId,
          path.targetInfo.id,
        };
        if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS) {
          // An incomplete snapshot cannot establish a unique monitor identity.
          return std::nullopt;
        }
        if (_wcsicmp(target_name.monitorDevicePath, device_path.c_str()) != 0) {
          continue;
        }
        if (found) {
          return std::nullopt;
        }
        // A cloned source is not a private destination.
        for (const auto &other : paths) {
          if (&other != &path && other.sourceInfo.id == path.sourceInfo.id && other.sourceInfo.adapterId.HighPart == path.sourceInfo.adapterId.HighPart && other.sourceInfo.adapterId.LowPart == path.sourceInfo.adapterId.LowPart) {
            return std::nullopt;
          }
        }
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source {};
        source.header = {
          DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME,
          sizeof(source),
          path.sourceInfo.adapterId,
          path.sourceInfo.id,
        };
        if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS) {
          return std::nullopt;
        }
        source_name = source.viewGdiDeviceName;
        found = target_t {};
        found->adapter = path.targetInfo.adapterId;
        found->target_id = path.targetInfo.id;
      }
      if (!found) {
        return std::nullopt;
      }

      struct enumeration_t {
        std::wstring_view name;
        target_t *target;
        unsigned matches = 0;
      } enumeration {source_name, &*found};

      const auto complete = EnumDisplayMonitors(
        nullptr,
        nullptr,
        [](HMONITOR monitor, HDC, LPRECT, LPARAM context) -> BOOL {
          auto &enumeration = *reinterpret_cast<enumeration_t *>(context);
          MONITORINFOEXW info {};
          info.cbSize = sizeof(info);
          if (!GetMonitorInfoW(monitor, &info)) {
            return FALSE;
          }
          if (enumeration.name == info.szDevice) {
            ++enumeration.matches;
            enumeration.target->monitor = monitor;
            enumeration.target->info = info;
          }
          return TRUE;
        },
        reinterpret_cast<LPARAM>(&enumeration)
      );
      if (!complete || enumeration.matches != 1) {
        return std::nullopt;
      }
      return found;
    }
    return std::nullopt;
  }

  bool same_target(const target_t &left, const target_t &right) {
    return left.adapter.HighPart == right.adapter.HighPart &&
           left.adapter.LowPart == right.adapter.LowPart &&
           left.target_id == right.target_id &&
           left.monitor == right.monitor &&
           EqualRect(&left.info.rcMonitor, &right.info.rcMonitor) &&
           EqualRect(&left.info.rcWork, &right.info.rcWork);
  }

  bool window_kind_allowed(HWND window) {
    if (!IsWindowVisible(window) || GetAncestor(window, GA_ROOT) != window) {
      return false;
    }
    const auto style = GetWindowLongPtrW(window, GWL_STYLE);
    const auto extended = GetWindowLongPtrW(window, GWL_EXSTYLE);
    if ((style & WS_CHILD) || (extended & (WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE))) {
      return false;
    }
    std::array<wchar_t, 128> class_name {};
    GetClassNameW(window, class_name.data(), static_cast<int>(class_name.size()));
    const std::wstring_view name(class_name.data());
    if (name == L"Progman" || name == L"WorkerW" || name == L"Shell_TrayWnd" || name == L"Shell_SecondaryTrayWnd" || name == L"#32768") {
      return false;
    }
    // Captioned dialogs and app popups are eligible; menus, tooltips and invisible owners are not.
    if ((style & WS_POPUP) && !(style & WS_CAPTION) && !(extended & WS_EX_APPWINDOW)) {
      return false;
    }
    DWORD cloaked = 0;
    return SUCCEEDED(DwmGetWindowAttribute(window, DWMWA_CLOAKED, &cloaked, sizeof(cloaked))) &&
           cloaked == 0;
  }

  bool interactive_move(HWND window) {
    GUITHREADINFO info {};
    info.cbSize = sizeof(info);
    return !GetGUIThreadInfo(GetWindowThreadProcessId(window, nullptr), &info) ||
           (info.flags & GUI_INMOVESIZE) != 0;
  }

  bool place_window(HWND window, const target_t &target) {
    RECT bounds {};
    if (!GetWindowRect(window, &bounds)) {
      return false;
    }
    if (!IsIconic(window) && MonitorFromWindow(window, MONITOR_DEFAULTTONULL) == target.monitor && (IsZoomed(window) || desktop_launcher::contains(rectangle(target.info.rcWork), rectangle(bounds)))) {
      return true;
    }
    if (IsZoomed(window) || IsIconic(window)) {
      WINDOWPLACEMENT placement {};
      placement.length = sizeof(placement);
      MONITORINFO source {};
      source.cbSize = sizeof(source);
      const auto source_monitor = MonitorFromWindow(window, MONITOR_DEFAULTTONULL);
      if (!GetWindowPlacement(window, &placement) || !source_monitor || !GetMonitorInfoW(source_monitor, &source)) {
        return false;
      }
      const auto restored = desktop_launcher::workspace_to_screen(
        rectangle(placement.rcNormalPosition),
        rectangle(source.rcMonitor),
        rectangle(source.rcWork)
      );
      const auto positioned = desktop_launcher::place_in_work_area(restored, rectangle(target.info.rcWork));
      if (!positioned) {
        return false;
      }
      placement.rcNormalPosition = native_rectangle(desktop_launcher::screen_to_workspace(*positioned, rectangle(target.info.rcMonitor), rectangle(target.info.rcWork)));
      placement.ptMaxPosition = {-1, -1};
      placement.flags |= WPF_ASYNCWINDOWPLACEMENT;
      if (IsIconic(window)) {
        placement.showCmd = (placement.flags & WPF_RESTORETOMAXIMIZED) ? SW_SHOWMAXIMIZED : SW_SHOWNOACTIVATE;
      }
      // Preserve maximized state while giving restored windows a position on the target.
      return SetWindowPlacement(window, &placement) != FALSE;
    }
    const auto positioned = desktop_launcher::place_in_work_area(rectangle(bounds), rectangle(target.info.rcWork));
    if (!positioned) {
      return false;
    }
    return SetWindowPos(
             window,
             nullptr,
             static_cast<int>(positioned->left),
             static_cast<int>(positioned->top),
             static_cast<int>(positioned->right - positioned->left),
             static_cast<int>(positioned->bottom - positioned->top),
             SWP_NOZORDER | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS | SWP_NOOWNERZORDER
           ) != FALSE;
  }

  // Expand Windows' 32-bit event timestamps in the same monotonic epoch as GetTickCount64.
  std::uint64_t event_tick(DWORD tick) {
    const auto now = GetTickCount64();
    const auto age = static_cast<std::int32_t>(static_cast<DWORD>(now) - tick);
    return age >= 0 && static_cast<std::uint64_t>(age) <= now ? now - age : now + static_cast<std::uint64_t>(-static_cast<std::int64_t>(age));
  }

  struct surface_t {
    HWND root = nullptr;
    DWORD pid = 0;
    std::array<wchar_t, 128> name {};
  };

  surface_t surface(HWND window) {
    surface_t result;
    result.root = GetAncestor(window, GA_ROOT);
    if (result.root) {
      GetWindowThreadProcessId(result.root, &result.pid);
      GetClassNameW(result.root, result.name.data(), static_cast<int>(result.name.size()));
    }
    return result;
  }

  enum class shell_kind_e {
    none,
    folder,
    infrastructure
  };

  shell_kind_e shell_kind(const surface_t &source) {
    DWORD session = 0, own_session = 0;
    if (!source.pid || !ProcessIdToSessionId(source.pid, &session) || !ProcessIdToSessionId(GetCurrentProcessId(), &own_session) || session != own_session) {
      return shell_kind_e::none;
    }
    owned_handle_t process(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, source.pid));
    std::array<wchar_t, 32768> path {}, windows {};
    DWORD size = static_cast<DWORD>(path.size());
    if (!process || !QueryFullProcessImageNameW(process.get(), 0, path.data(), &size) || !GetWindowsDirectoryW(windows.data(), static_cast<UINT>(windows.size()))) {
      return shell_kind_e::none;
    }
    const auto base = std::wstring(windows.data());
    const std::wstring_view name(source.name.data());
    if (_wcsicmp(path.data(), (base + L"\\explorer.exe").c_str()) == 0) {
      if (name == L"CabinetWClass" || name == L"ExploreWClass") {
        return shell_kind_e::folder;
      }
      if (name == L"Progman" || name == L"WorkerW" || name == L"Shell_TrayWnd" || name == L"Shell_SecondaryTrayWnd" || name == L"TaskListThumbnailWnd" || name == L"#32770" || name == L"#32768") {
        return shell_kind_e::infrastructure;
      }
    }
    // Exact Windows-owned locations, not an executable basename supplied by an application.
    for (const auto relative : {
           L"\\SystemApps\\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\\StartMenuExperienceHost.exe",
           L"\\SystemApps\\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\\SearchHost.exe",
           L"\\SystemApps\\Microsoft.Windows.Search_cw5n1h2txyewy\\SearchApp.exe",
           L"\\SystemApps\\ShellExperienceHost_cw5n1h2txyewy\\ShellExperienceHost.exe"
         }) {
      if (_wcsicmp(path.data(), (base + relative).c_str()) == 0) {
        return shell_kind_e::infrastructure;
      }
    }
    return shell_kind_e::none;
  }

  struct input_event_t {
    surface_t source;
    POINT point {};
    DWORD tick = 0;
    DWORD key = 0;
    UINT message = 0;
    std::uint64_t cancellation = 0;
    bool keyboard = false;
    bool trusted = false;
    std::uint64_t serial = 0;
    HWND foreground_before = nullptr;
    bool windows_key = false;
    bool control_key = false;
  };

  /** Low-level hooks only capture metadata. Slow process/display queries run on the router thread. */
  class input_observer_t {
  public:
    input_observer_t(HWND destination, std::uint64_t tag):
        destination_(destination),
        tag_(tag) {}

    ~input_observer_t() {
      if (thread_.joinable()) {
        PostThreadMessageW(thread_id_, WM_QUIT, 0, 0);
        thread_.join();
      }
    }

    bool start() {
      owned_handle_t ready(CreateEventW(nullptr, TRUE, FALSE, nullptr));
      if (!ready) {
        return false;
      }
      const auto event = ready.get();
      thread_ = std::thread([this, event] {
        active_ = this;
        thread_id_ = GetCurrentThreadId();
        MSG message {};
        PeekMessageW(&message, nullptr, 0, 0, PM_NOREMOVE);
        const auto keyboard = SetWindowsHookExW(WH_KEYBOARD_LL, keyboard_hook, GetModuleHandleW(nullptr), 0);
        const auto mouse = SetWindowsHookExW(WH_MOUSE_LL, mouse_hook, GetModuleHandleW(nullptr), 0);
        started_ = keyboard && mouse;
        SetEvent(event);
        if (started_) {
          while (GetMessageW(&message, nullptr, 0, 0) > 0) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
          }
        }
        if (keyboard) {
          UnhookWindowsHookEx(keyboard);
        }
        if (mouse) {
          UnhookWindowsHookEx(mouse);
        }
        active_ = nullptr;
      });
      WaitForSingleObject(ready.get(), INFINITE);
      return started_;
    }

    std::deque<input_event_t> take() {
      std::lock_guard lock(mutex_);
      std::deque<input_event_t> result;
      result.swap(queue_);
      return result;
    }

    std::uint64_t cancellation() const {
      return cancellation_.load();
    }

    std::uint64_t next_serial() {
      return ++serial_;
    }

    bool idle_unchanged() {
      LASTINPUTINFO info {sizeof(info)};
      std::lock_guard lock(mutex_);
      return queue_.empty() && GetLastInputInfo(&info) && info.dwTime == last_tick_.load();
    }

  private:
    void push(input_event_t event) {
      if (!event.trusted) {
        ++cancellation_;
      }
      event.serial = next_serial();
      event.foreground_before = GetAncestor(GetForegroundWindow(), GA_ROOT);
      event.cancellation = cancellation_.load();
      last_tick_ = event.tick;
      {
        std::lock_guard lock(mutex_);
        if (queue_.size() >= 256) {
          queue_.clear();
          event.cancellation = ++cancellation_;
        }
        queue_.push_back(event);
      }
      PostMessageW(destination_, input_message, 0, 0);
    }

    static LRESULT CALLBACK keyboard_hook(int code, WPARAM message, LPARAM data) {
      if (code == HC_ACTION && active_) {
        auto &observer = *active_;
        const auto &key = *reinterpret_cast<KBDLLHOOKSTRUCT *>(data);
        const bool down = message == WM_KEYDOWN || message == WM_SYSKEYDOWN;
        const bool trusted = !(key.flags & LLKHF_INJECTED) || key.dwExtraInfo == observer.tag_;
        // Track modifiers from the event stream: async key state has not yet been updated here.
        if (!trusted) {
          observer.keys_.fill(false);
        }
        if (key.vkCode < observer.keys_.size()) {
          observer.keys_[key.vkCode] = down && trusted;
        }
        input_event_t event;
        event.source = surface(GetForegroundWindow());
        GetCursorPos(&event.point);
        event.tick = key.time;
        event.key = key.vkCode;
        event.message = static_cast<UINT>(message);
        event.keyboard = true;
        event.trusted = trusted;
        event.windows_key = observer.keys_[VK_LWIN] || observer.keys_[VK_RWIN];
        event.control_key = observer.keys_[VK_CONTROL] || observer.keys_[VK_LCONTROL] || observer.keys_[VK_RCONTROL];
        observer.push(event);
      }
      return CallNextHookEx(nullptr, code, message, data);
    }

    static LRESULT CALLBACK mouse_hook(int code, WPARAM message, LPARAM data) {
      if (code == HC_ACTION && active_) {
        const auto &mouse = *reinterpret_cast<MSLLHOOKSTRUCT *>(data);
        input_event_t event;
        event.source = surface(WindowFromPoint(mouse.pt));
        event.point = mouse.pt;
        event.tick = mouse.time;
        event.message = static_cast<UINT>(message);
        event.trusted = !(mouse.flags & LLMHF_INJECTED) || mouse.dwExtraInfo == active_->tag_;
        active_->push(event);
      }
      return CallNextHookEx(nullptr, code, message, data);
    }

    static thread_local input_observer_t *active_;
    HWND destination_;
    std::uint64_t tag_;
    std::thread thread_;
    DWORD thread_id_ = 0;
    bool started_ = false;
    std::array<bool, 256> keys_ {};
    std::mutex mutex_;
    std::deque<input_event_t> queue_;
    std::atomic<std::uint64_t> cancellation_ {0};
    std::atomic<DWORD> last_tick_ {0};
    std::atomic<std::uint64_t> serial_ {0};
  };

  thread_local input_observer_t *input_observer_t::active_ = nullptr;

  // GA_ROOTOWNER follows GetParent semantics and can miss owners of overlapped windows.
  // Follow GW_OWNER explicitly, with a bound for corrupt/cyclic ownership chains.
  HWND root_owner(HWND window) {
    auto root = GetAncestor(window, GA_ROOT);
    for (unsigned depth = 0; root && depth < 32; ++depth) {
      const auto owner = GetWindow(root, GW_OWNER);
      if (!owner) {
        return root;
      }
      const auto next = GetAncestor(owner, GA_ROOT);
      if (!next || next == root) {
        return nullptr;
      }
      root = next;
    }
    return nullptr;
  }

  // A foreground root can own dialogs. An unrelated document in the same process is not family.
  bool same_family(HWND candidate, HWND root) {
    return candidate == root || root_owner(candidate) == root;
  }

  constexpr wchar_t family_property[] = L"Sunshine3DWindowRouterOwner";

  struct family_t {
    owned_handle_t process;
    HANDLE marker = nullptr;
    std::set<HWND> handled;
  };

  class router_t {
  public:
    router_t(std::wstring display_path, owned_handle_t parent, std::uint64_t tag):
        display_path_(std::move(display_path)),
        parent_(std::move(parent)),
        tag_(tag) {}

    ~router_t() {
      cancel();
      observer_.reset();
      if (foreground_hook_) {
        UnhookWinEvent(foreground_hook_);
      }
      if (show_hook_) {
        UnhookWinEvent(show_hook_);
      }
      if (desktop_hook_) {
        UnhookWinEvent(desktop_hook_);
      }
      if (window_) {
        DestroyWindow(window_);
      }
      active_ = nullptr;
    }

    bool create(HINSTANCE instance) {
      WNDCLASSW cls {};
      cls.lpfnWndProc = window_proc;
      cls.hInstance = instance;
      cls.lpszClassName = window_class;
      if (!RegisterClassW(&cls)) {
        return false;
      }
      active_ = this;
      // An ordinary hidden top-level window lets the host close this exact helper by PID.
      window_ = CreateWindowExW(WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, window_class, L"", WS_POPUP, 0, 0, 0, 0, nullptr, nullptr, instance, this);
      if (!window_) {
        return false;
      }
      observer_ = std::make_unique<input_observer_t>(window_, tag_);
      if (!observer_->start()) {
        return false;
      }
      foreground_hook_ = SetWinEventHook(EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, nullptr, win_event, 0, 0, WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS);
      desktop_hook_ = SetWinEventHook(EVENT_SYSTEM_DESKTOPSWITCH, EVENT_SYSTEM_DESKTOPSWITCH, nullptr, win_event, 0, 0, WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS);
      show_hook_ = SetWinEventHook(EVENT_OBJECT_SHOW, EVENT_OBJECT_SHOW, nullptr, win_event, 0, 0, WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS);
      return foreground_hook_ && desktop_hook_ && show_hook_ && SetTimer(window_, poll_timer, 250, nullptr);
    }

  private:
    bool refresh_target() {
      if (WaitForSingleObject(parent_.get(), 0) != WAIT_TIMEOUT) {
        PostMessageW(window_, WM_CLOSE, 0, 0);
        cancel();
        return false;
      }
      auto target = default_input_desktop() ? resolve_target(display_path_) : std::nullopt;
      if (!target || (target_ && !same_target(*target_, *target))) {
        cancel();
      }
      target_ = std::move(target);
      return target_.has_value();
    }

    void cancel() {
      intent_.cancel();
      virtual_navigation_ = 0;
      last_click_.reset();
      deferred_foreground_.reset();
      shell_items_.clear();
      for (const auto &[root, family] : families_) {
        if (GetPropW(root, family_property) == family.marker) {
          RemovePropW(root, family_property);
        }
      }
      families_.clear();
    }

    void warm_shell_cache() {
      POINT point {};
      if (!target_ || !GetCursorPos(&point) || !PtInRect(&target_->info.rcMonitor, point)) {
        return;
      }
      const auto source = surface(WindowFromPoint(point));
      if (shell_kind(source) != shell_kind_e::none || window_kind_allowed(source.root)) {
        shell_items_.request(source.root, point);
      }
    }

    void input() {
      const auto events = observer_->take();
      if (events.empty()) {
        return;
      }
      refresh_target();
      for (const auto &event : events) {
        if (event.cancellation != observer_->cancellation() || !event.trusted || !target_) {
          cancel();
          continue;
        }
        cancellation_ = event.cancellation;
        const bool pointer_inside = PtInRect(&target_->info.rcMonitor, event.point);
        if (!event.keyboard && !pointer_inside) {
          cancel();
          continue;
        }
        // Releases from the invocation gesture do not cancel its foreground result.
        if ((event.keyboard && (event.message == WM_KEYUP || event.message == WM_SYSKEYUP)) || (!event.keyboard && event.message == WM_LBUTTONUP)) {
          continue;
        }
        const auto tick = event_tick(event.tick);
        const auto shell = shell_kind(event.source);
        const bool application_source = window_kind_allowed(event.source.root) &&
                                        MonitorFromWindow(event.source.root, MONITOR_DEFAULTTONULL) == target_->monitor;
        const bool shortcut = event.keyboard && (event.key == VK_LWIN || event.key == VK_RWIN ||
                                                 (event.windows_key && (event.key == 'R' || event.key == 'E' || event.key == 'S')) ||
                                                 (event.control_key && event.key == VK_ESCAPE));
        const bool inside = event.keyboard && !shortcut ?
                              MonitorFromWindow(event.source.root, MONITOR_DEFAULTTONULL) == target_->monitor :
                              pointer_inside;
        const bool inherited = event.keyboard && shell != shell_kind_e::none && virtual_navigation_ &&
                               tick >= virtual_navigation_ && tick - virtual_navigation_ <= desktop_launcher::native_window_intent_t::timeout_ms;
        if (!inside && !inherited) {
          cancel();
          continue;
        }
        bool mouse_launch = false;
        if (!event.keyboard && (shell != shell_kind_e::none || application_source)) {
          if (event.message == WM_MOUSEMOVE || event.message == WM_LBUTTONDOWN) {
            shell_items_.request(event.source.root, event.point);
          }
          if (event.message == WM_LBUTTONDOWN) {
            const std::wstring_view name(event.source.name.data());
            const bool icons = shell == shell_kind_e::folder || name == L"Progman" || name == L"WorkerW";
            SHELLFLAGSTATE state {};
            SHGetSettings(&state, SSF_DOUBLECLICKINWEBVIEW);
            const bool double_click = last_click_ && last_click_->source.root == event.source.root &&
                                      tick >= event_tick(last_click_->tick) && tick - event_tick(last_click_->tick) <= GetDoubleClickTime() &&
                                      std::abs(event.point.x - last_click_->point.x) <= GetSystemMetrics(SM_CXDOUBLECLK) / 2 &&
                                      std::abs(event.point.y - last_click_->point.y) <= GetSystemMetrics(SM_CYDOUBLECLK) / 2;
            mouse_launch = shell_items_.result(event.source.root, event.point).value_or(false) &&
                           (!icons || !state.fDoubleClickInWebView || double_click);
            last_click_ = event;
          }
        }
        const auto action = event.keyboard ?
                              ((event.key == VK_RETURN || (event.windows_key && event.key == 'E')) ? desktop_launcher::native_input_action_e::launch : desktop_launcher::native_input_action_e::navigate) :
                              (mouse_launch                  ? desktop_launcher::native_input_action_e::launch :
                               event.message == WM_MOUSEMOVE ? desktop_launcher::native_input_action_e::motion :
                                                               desktop_launcher::native_input_action_e::navigate);
        // Standard app controls and terminal Enter can launch another executable too.
        // Their source window must itself be on the virtual monitor.
        const bool app_invocation = application_source &&
                                    (mouse_launch || (event.keyboard && event.key == VK_RETURN));
        const bool shell_source = shell != shell_kind_e::none || shortcut || app_invocation;
        if (action == desktop_launcher::native_input_action_e::launch) {
          const std::wstring_view name(event.source.name.data());
          taskbar_launch_ = !event.keyboard && (name == L"Shell_TrayWnd" || name == L"Shell_SecondaryTrayWnd" || name == L"TaskListThumbnailWnd");
          launch_foreground_ = event.foreground_before;
        }
        intent_.observe_input({tick, reinterpret_cast<std::uintptr_t>(event.source.root), action, true, true, inside, shell_source, inherited, event.serial, reinterpret_cast<std::uintptr_t>(event.foreground_before)});
        if (action != desktop_launcher::native_input_action_e::motion) {
          virtual_navigation_ = (shell != shell_kind_e::none || shortcut) && (inside || inherited) ? tick : 0;
        }
      }
    }

    void foreground(HWND window, DWORD timestamp, std::uint64_t observed_serial = 0) {
      if (!observed_serial) {
        observed_serial = observer_->next_serial();
      }
      input();
      if (!refresh_target()) {
        return;
      }
      route_owned(window);
      if (!intent_.armed()) {
        return;
      }
      // Clicking an active taskbar app can minimize it and expose an unrelated window.
      if (taskbar_launch_ && launch_foreground_ && (!IsWindow(launch_foreground_) || IsIconic(launch_foreground_))) {
        cancel();
        return;
      }
      const auto generation = intent_.generation();
      const auto source = surface(window);
      if (source.root != window || shell_kind(source) == shell_kind_e::infrastructure || !window_kind_allowed(window) || interactive_move(window)) {
        return;
      }
      DWORD own_session = 0, session = 0;
      if (!ProcessIdToSessionId(source.pid, &session) || !ProcessIdToSessionId(GetCurrentProcessId(), &own_session) || session != own_session) {
        return;
      }
      owned_handle_t process(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, source.pid));
      const auto created = process ? process_start(process.get()) : std::nullopt;
      if (!created) {
        return;
      }
      const bool idle = observer_->cancellation() == cancellation_ && observer_->idle_unchanged();
      if (!idle) {
        deferred_foreground_ = foreground_event_t {window, timestamp, observed_serial, generation};
        return;
      }
      deferred_foreground_.reset();
      if (!intent_.claim({event_tick(timestamp), GetTickCount64(), generation, reinterpret_cast<std::uintptr_t>(window), true, false, GetForegroundWindow() == window, true, idle, observed_serial})) {
        return;
      }
      virtual_navigation_ = 0;
      DWORD current_pid = 0;
      GetWindowThreadProcessId(window, &current_pid);
      if (current_pid != source.pid || WaitForSingleObject(process.get(), 0) != WAIT_TIMEOUT || GetForegroundWindow() != window || observer_->cancellation() != cancellation_ || !observer_->idle_unchanged()) {
        return;
      }
      // Move only the selected root and its owned dialogs, never all windows of a reused process.
      const auto owner = root_owner(window);
      const auto root = owner && window_kind_allowed(owner) ? owner : window;
      if (!place_window(root, *target_)) {
        return;
      }
      if (families_.size() >= 32) {
        cancel();
      }
      const auto marker = reinterpret_cast<HANDLE>(++family_serial_);
      DWORD root_pid = 0;
      GetWindowThreadProcessId(root, &root_pid);
      owned_handle_t root_process(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, root_pid));
      if (root_process && SetPropW(root, family_property, marker)) {
        families_.insert_or_assign(root, family_t {std::move(root_process), marker, {root}});
      }
      if (window != root && observer_->idle_unchanged() && observer_->cancellation() == cancellation_) {
        if (place_window(window, *target_) && families_.contains(root)) {
          families_.at(root).handled.insert(window);
        }
      }
    }

    void retry_foreground() {
      if (!deferred_foreground_) {
        return;
      }
      const auto event = *deferred_foreground_;
      deferred_foreground_.reset();
      if (event.generation == intent_.generation() && GetForegroundWindow() == event.window && GetTickCount64() - event_tick(event.timestamp) <= desktop_launcher::native_window_intent_t::timeout_ms) {
        foreground(event.window, event.timestamp, event.serial);
      }
    }

    void route_owned(HWND window) {
      const auto root = root_owner(window);
      const auto found = families_.find(root);
      if (found == families_.end() || window == root || !same_family(window, root)) {
        return;
      }
      auto &family = found->second;
      if (family.handled.contains(window) || family.handled.size() >= 4096 || GetPropW(root, family_property) != family.marker || WaitForSingleObject(family.process.get(), 0) != WAIT_TIMEOUT || !target_ || !window_kind_allowed(window) || interactive_move(window) || observer_->cancellation() != cancellation_ || !observer_->idle_unchanged()) {
        return;
      }
      if (place_window(window, *target_)) {
        family.handled.insert(window);
      }
    }

    static void CALLBACK win_event(HWINEVENTHOOK, DWORD event, HWND window, LONG object, LONG child, DWORD, DWORD timestamp) {
      if (!active_) {
        return;
      }
      if (event == EVENT_SYSTEM_DESKTOPSWITCH) {
        active_->cancel();
        return;
      }
      if (event == EVENT_OBJECT_SHOW && window && object == OBJID_WINDOW && child == CHILDID_SELF) {
        active_->input();
        if (!active_->families_.empty() && active_->refresh_target()) {
          active_->route_owned(window);
        }
      }
      if (event == EVENT_SYSTEM_FOREGROUND && window && object == OBJID_WINDOW && child == CHILDID_SELF) {
        active_->foreground(window, timestamp);
      }
    }

    static LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
      auto *router = reinterpret_cast<router_t *>(GetWindowLongPtrW(window, GWLP_USERDATA));
      if (message == WM_NCCREATE) {
        router = static_cast<router_t *>(reinterpret_cast<CREATESTRUCTW *>(lparam)->lpCreateParams);
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(router));
      }
      if (router) {
        if (message == input_message) {
          router->input();
          return 0;
        }
        if (message == WM_TIMER) {
          router->input();
          router->refresh_target();
          router->retry_foreground();
          router->route_owned(GetForegroundWindow());
          router->warm_shell_cache();
          return 0;
        }
        if (message == WM_CLOSE) {
          DestroyWindow(window);
          return 0;
        }
        if (message == WM_DESTROY) {
          router->window_ = nullptr;
          PostQuitMessage(0);
          return 0;
        }
      }
      return DefWindowProcW(window, message, wparam, lparam);
    }

    static router_t *active_;
    std::wstring display_path_;
    owned_handle_t parent_;
    std::uint64_t tag_;
    HWND window_ = nullptr;
    HWINEVENTHOOK foreground_hook_ = nullptr, desktop_hook_ = nullptr, show_hook_ = nullptr;
    std::unique_ptr<input_observer_t> observer_;
    desktop_launcher::native_shell_item_query_t shell_items_;
    std::optional<input_event_t> last_click_;
    std::map<HWND, family_t> families_;
    std::uintptr_t family_serial_ = 0;

    struct foreground_event_t {
      HWND window;
      DWORD timestamp;
      std::uint64_t serial, generation;
    };

    std::optional<foreground_event_t> deferred_foreground_;
    bool taskbar_launch_ = false;
    HWND launch_foreground_ = nullptr;
    std::optional<target_t> target_;
    desktop_launcher::native_window_intent_t intent_;
    std::uint64_t cancellation_ = 0, virtual_navigation_ = 0;
  };

  router_t *router_t::active_ = nullptr;

  /** Only self-created windows on a private, never switched-to desktop are mutated. */
  int self_test() {
    const auto original = GetThreadDesktop(GetCurrentThreadId());
    const auto name = L"SunshineWindowRoutingTest-" + std::to_wstring(GetCurrentProcessId());
    const auto desktop = CreateDesktopW(name.c_str(), nullptr, nullptr, 0, GENERIC_ALL, nullptr);
    if (!desktop || !SetThreadDesktop(desktop)) {
      if (desktop) {
        CloseDesktop(desktop);
      }
      return 10;
    }
    WNDCLASSW cls {};
    cls.lpfnWndProc = DefWindowProcW;
    cls.hInstance = GetModuleHandleW(nullptr);
    cls.lpszClassName = L"SunshineWindowRoutingFixture";
    RegisterClassW(&cls);
    const auto create = [&](HWND owner) {
      return CreateWindowExW(0, cls.lpszClassName, L"Routing fixture", WS_OVERLAPPEDWINDOW, 20, 30, 640, 480, owner, nullptr, cls.hInstance, nullptr);
    };
    const auto sentinel = create(nullptr), root = create(nullptr), dialog = create(root);
    int result = 0;
    RECT before {}, after {};
    GetWindowRect(sentinel, &before);
    target_t target;
    target.monitor = MonitorFromWindow(root, MONITOR_DEFAULTTONEAREST);
    const auto original_monitor = target.monitor;
    EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR monitor, HDC, LPRECT, LPARAM value) -> BOOL {
      auto &selected = *reinterpret_cast<HMONITOR *>(value);
      if (monitor != selected) {
        selected = monitor;
        return FALSE;
      }
      return TRUE;
    },
                        reinterpret_cast<LPARAM>(&target.monitor));
    target.info.cbSize = sizeof(target.info);
    if (!sentinel || !root || !dialog || !GetMonitorInfoW(target.monitor, &target.info)) {
      result = 11;
    }
    if (!result && (!same_family(dialog, root) || same_family(sentinel, root))) {
      result = 12;
    }
    for (const auto state : {SW_SHOWNOACTIVATE, SW_SHOWMAXIMIZED, SW_MINIMIZE}) {
      if (result) {
        break;
      }
      ShowWindow(root, state);
      if (!place_window(root, target)) {
        result = 13;
        break;
      }
      RECT bounds {};
      GetWindowRect(root, &bounds);
      if (IsIconic(root) || MonitorFromWindow(root, MONITOR_DEFAULTTONULL) != target.monitor || (!IsZoomed(root) && !desktop_launcher::contains(rectangle(target.info.rcWork), rectangle(bounds)))) {
        result = 14;
      }
    }
    GetWindowRect(sentinel, &after);
    if (!EqualRect(&before, &after)) {
      result = 15;
    }
    DestroyWindow(dialog);
    DestroyWindow(root);
    DestroyWindow(sentinel);
    UnregisterClassW(cls.lpszClassName, cls.hInstance);
    SetThreadDesktop(original);
    CloseDesktop(desktop);
    std::fwprintf(stderr, L"Native window placement self-test result=%d; cross-monitor=%ls.\n", result, target.monitor != original_monitor ? L"tested" : L"unavailable (one monitor)");
    return result;
  }

  std::optional<std::uint64_t> parse_unsigned(const wchar_t *text) {
    std::uint64_t value = 0;
    if (!*text) {
      return std::nullopt;
    }
    for (; *text; ++text) {
      if (*text < L'0' || *text > L'9' || value > (std::numeric_limits<std::uint64_t>::max() - (*text - L'0')) / 10) {
        return std::nullopt;
      }
      value = value * 10 + (*text - L'0');
    }
    return value ? std::optional {value} : std::nullopt;
  }
}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
  SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
  int count = 0;
  const auto arguments = CommandLineToArgvW(GetCommandLineW(), &count);
  if (!arguments) {
    return 2;
  }
  if (count == 2 && std::wstring_view(arguments[1]) == L"--self-test-shell-items") {
    LocalFree(arguments);
    return desktop_launcher::native_shell_item_self_test();
  }
  if (count == 2 && std::wstring_view(arguments[1]) == L"--self-test") {
    LocalFree(arguments);
    return self_test();
  }
  std::wstring display_path;
  std::optional<std::uint64_t> parent_id, parent_created, input_tag;
  bool valid = count == 9;
  for (int index = 1; valid && index + 1 < count; index += 2) {
    const std::wstring_view option(arguments[index]);
    if (option == L"--display-path" && display_path.empty()) {
      display_path = arguments[index + 1];
    } else if (option == L"--parent-pid" && !parent_id) {
      parent_id = parse_unsigned(arguments[index + 1]);
    } else if (option == L"--parent-start" && !parent_created) {
      parent_created = parse_unsigned(arguments[index + 1]);
    } else if (option == L"--input-tag" && !input_tag) {
      input_tag = parse_unsigned(arguments[index + 1]);
    } else {
      valid = false;
    }
  }
  LocalFree(arguments);
  if (!valid || display_path.empty() || !parent_id || *parent_id > MAXDWORD || !parent_created || !input_tag) {
    return 2;
  }
  owned_handle_t parent(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, static_cast<DWORD>(*parent_id)));
  if (!parent || WaitForSingleObject(parent.get(), 0) != WAIT_TIMEOUT || process_start(parent.get()) != parent_created || !default_input_desktop()) {
    return 3;
  }
  HANDLE raw_token = nullptr;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw_token)) {
    return 3;
  }
  owned_handle_t token(raw_token);
  TOKEN_ELEVATION elevation {};
  DWORD returned = 0;
  if (!GetTokenInformation(token.get(), TokenElevation, &elevation, sizeof(elevation), &returned) || elevation.TokenIsElevated) {
    return 3;
  }
  router_t router(std::move(display_path), std::move(parent), *input_tag);
  if (!router.create(instance)) {
    return 4;
  }
  MSG message {};
  BOOL status;
  while ((status = GetMessageW(&message, nullptr, 0, 0)) > 0) {
    TranslateMessage(&message);
    DispatchMessageW(&message);
  }
  return status == -1 ? 5 : 0;
}
