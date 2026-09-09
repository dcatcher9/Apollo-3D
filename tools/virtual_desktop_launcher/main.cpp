/**
 * @file tools/virtual_desktop_launcher/main.cpp
 * @brief Standard-user launcher for new applications on one exact remote virtual display.
 */
#ifndef NOMINMAX
  #define NOMINMAX
#endif
#include "launcher_policy.h"

#include <array>
#include <chrono>
#include <compare>
#include <cstdint>
#include <cstdio>
#include <dwmapi.h>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <shellapi.h>
#include <shobjidl.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <windows.h>

namespace {
  using clock_t = std::chrono::steady_clock;
  using namespace std::chrono_literals;
  using desktop_launcher::rectangle_t;

  constexpr wchar_t window_class[] = L"Sunshine3DDesktopLauncher";
  constexpr UINT_PTR poll_timer = 1;
  constexpr int executable_control = 101;
  constexpr int browse_control = 102;
  constexpr int arguments_control = 103;
  constexpr int launch_control = 104;
  constexpr std::size_t maximum_launches = 32;
  constexpr std::size_t maximum_windows = 4096;

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
    if (!IsWindowVisible(window) || IsIconic(window) || GetAncestor(window, GA_ROOT) != window) {
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
    if (MonitorFromWindow(window, MONITOR_DEFAULTTONULL) == target.monitor && (IsZoomed(window) || desktop_launcher::contains(rectangle(target.info.rcWork), rectangle(bounds)))) {
      return true;
    }
    if (IsZoomed(window)) {
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
      // Preserve showCmd: restoring a maximized app just to move it loses its intended state.
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

  std::wstring control_text(HWND control) {
    const auto length = GetWindowTextLengthW(control);
    std::wstring text(static_cast<std::size_t>(length) + 1, L'\0');
    text.resize(GetWindowTextW(control, text.data(), static_cast<int>(text.size())));
    return text;
  }

  std::wstring error_text(DWORD code) {
    wchar_t *message = nullptr;
    const auto length = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr,
      code,
      0,
      reinterpret_cast<wchar_t *>(&message),
      0,
      nullptr
    );
    std::wstring result = length ? std::wstring(message, length) : L"Windows error " + std::to_wstring(code);
    LocalFree(message);
    while (!result.empty() && (result.back() == L'\r' || result.back() == L'\n')) {
      result.pop_back();
    }
    return result;
  }

  struct window_key_t {
    std::uintptr_t window;
    DWORD process;
    std::uint64_t process_created;
    auto operator<=>(const window_key_t &) const = default;
  };

  struct observed_window_t {
    clock_t::time_point first_visible;
    bool handled = false;
  };

  struct launch_t {
    owned_handle_t job;
    std::uint64_t started = 0;
    clock_t::time_point launched;
    std::set<std::uintptr_t> baseline;
    std::map<window_key_t, observed_window_t> windows;
    bool placed_any = false;
    bool reported_no_window = false;
  };

  class attribute_list_t {
  public:
    explicit attribute_list_t(HANDLE *job) {
      SIZE_T size = 0;
      InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
      if (!size) {
        return;
      }
      storage_.resize(size);
      list_ = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage_.data());
      if (!InitializeProcThreadAttributeList(list_, 1, 0, &size)) {
        list_ = nullptr;
        return;
      }
      if (!UpdateProcThreadAttribute(list_, 0, PROC_THREAD_ATTRIBUTE_JOB_LIST, job, sizeof(*job), nullptr, nullptr)) {
        DeleteProcThreadAttributeList(list_);
        list_ = nullptr;
      }
    }

    ~attribute_list_t() {
      if (list_) {
        DeleteProcThreadAttributeList(list_);
      }
    }

    LPPROC_THREAD_ATTRIBUTE_LIST get() const {
      return list_;
    }

  private:
    std::vector<std::byte> storage_;
    LPPROC_THREAD_ATTRIBUTE_LIST list_ = nullptr;
  };

  class launcher_t {
  public:
    launcher_t(std::wstring display_path, owned_handle_t parent, std::uint64_t parent_created):
        display_path_(std::move(display_path)),
        parent_(std::move(parent)),
        parent_created_(parent_created) {}

    ~launcher_t() {
      if (font_) {
        DeleteObject(font_);
      }
      // Closing non-killing tracking jobs never terminates applications the user launched.
    }

    bool create(HINSTANCE instance, const target_t &target) {
      last_target_ = target;
      WNDCLASSW definition {};
      definition.hInstance = instance;
      definition.lpfnWndProc = dispatch;
      definition.lpszClassName = window_class;
      definition.hCursor = LoadCursorW(nullptr, IDC_ARROW);
      definition.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
      definition.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
      if (!RegisterClassW(&definition)) {
        return false;
      }
      const auto initial = desktop_launcher::place_in_work_area({0, 0, 800, 450}, rectangle(target.info.rcWork));
      if (!initial) {
        return false;
      }
      window_ = CreateWindowExW(
        0,
        window_class,
        L"Open in Virtual Desktop — Sunshine 3D",
        WS_OVERLAPPEDWINDOW,
        static_cast<int>(initial->left),
        static_cast<int>(initial->top),
        static_cast<int>(initial->right - initial->left),
        static_cast<int>(initial->bottom - initial->top),
        nullptr,
        nullptr,
        instance,
        this
      );
      if (!window_) {
        return false;
      }
      const auto dpi = GetDpiForWindow(window_);
      const auto sized = desktop_launcher::place_in_work_area(
        {0, 0, MulDiv(800, dpi ? dpi : 96, 96), MulDiv(520, dpi ? dpi : 96, 96)},
        rectangle(target.info.rcWork)
      );
      if (sized) {
        SetWindowPos(window_, nullptr, static_cast<int>(sized->left), static_cast<int>(sized->top), static_cast<int>(sized->right - sized->left), static_cast<int>(sized->bottom - sized->top), SWP_NOZORDER | SWP_NOACTIVATE);
      }
      ShowWindow(window_, SW_SHOWNOACTIVATE);
      return true;
    }

    HWND window() const {
      return window_;
    }

  private:
    static LRESULT CALLBACK dispatch(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
      auto *self = reinterpret_cast<launcher_t *>(GetWindowLongPtrW(window, GWLP_USERDATA));
      if (message == WM_NCCREATE) {
        self = static_cast<launcher_t *>(reinterpret_cast<CREATESTRUCTW *>(lparam)->lpCreateParams);
        self->window_ = window;
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
      }
      if (self) {
        return self->message(message, wparam, lparam);
      }
      return DefWindowProcW(window, message, wparam, lparam);
    }

    void status(const std::wstring &text) {
      SetWindowTextW(status_, text.c_str());
    }

    HWND add_control(const wchar_t *class_name, const wchar_t *text, DWORD style, int id = 0) {
      const auto control = CreateWindowExW(
        class_name == std::wstring_view(L"EDIT") ? WS_EX_CLIENTEDGE : 0,
        class_name,
        text,
        WS_CHILD | WS_VISIBLE | style,
        0,
        0,
        0,
        0,
        window_,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
        GetModuleHandleW(nullptr),
        nullptr
      );
      controls_.push_back(control);
      return control;
    }

    void layout() {
      RECT client {};
      GetClientRect(window_, &client);
      const auto monitor_dpi = GetDpiForWindow(window_);
      // A streamed portrait/landscape mode can be smaller than its requested Windows UI scale.
      // Keep every action reachable within this window instead of enforcing an oversized dialog.
      const int dpi = std::max(1, std::min({
                                    static_cast<int>(monitor_dpi ? monitor_dpi : 96),
                                    MulDiv(client.right, 96, 520),
                                    MulDiv(client.bottom, 96, 440),
                                  }));
      const auto scale = [dpi](int value) {
        return MulDiv(value, dpi, 96);
      };
      const auto old_font = font_;
      font_ = CreateFontW(
        -scale(18),
        0,
        0,
        0,
        FW_NORMAL,
        FALSE,
        FALSE,
        FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH,
        L"Segoe UI"
      );
      for (auto control : controls_) {
        SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(font_), TRUE);
      }
      if (old_font) {
        DeleteObject(old_font);
      }
      const int margin = scale(24);
      const int width = std::max(scale(200), static_cast<int>(client.right) - margin * 2);
      const auto place = [&](HWND control, int x, int y, int w, int h) {
        MoveWindow(control, x, scale(y), w, scale(h), TRUE);
      };
      place(heading_, margin, 20, width, 30);
      place(description_, margin, 57, width, 52);
      place(executable_label_, margin, 117, width, 26);
      place(executable_, margin, 147, std::max(scale(80), width - scale(135)), 35);
      place(browse_, margin + width - scale(125), 147, scale(125), 35);
      place(arguments_label_, margin, 198, width, 26);
      place(arguments_, margin, 228, width, 35);
      place(launch_, margin, 282, scale(195), 42);
      place(status_, margin, 340, width, std::max(70, MulDiv(client.bottom, 96, dpi) - 350));
    }

    void browse() {
      IFileOpenDialog *dialog = nullptr;
      const auto created = CoCreateInstance(
        CLSID_FileOpenDialog,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&dialog)
      );
      if (FAILED(created)) {
        status(L"The app picker could not open. Paste the full path to an .exe file instead.");
        return;
      }
      DWORD options = 0;
      dialog->GetOptions(&options);
      dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR);
      const COMDLG_FILTERSPEC filter {L"Applications (*.exe)", L"*.exe"};
      dialog->SetFileTypes(1, &filter);
      dialog->SetTitle(L"Choose an app for this virtual desktop");
      if (SUCCEEDED(dialog->Show(window_))) {
        IShellItem *selected = nullptr;
        if (SUCCEEDED(dialog->GetResult(&selected))) {
          PWSTR path = nullptr;
          if (SUCCEEDED(selected->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
            SetWindowTextW(executable_, path);
            CoTaskMemFree(path);
          }
          selected->Release();
        }
      }
      dialog->Release();
    }

    void launch() {
      const auto target = resolve_target(display_path_);
      if (!target || !default_input_desktop()) {
        status(L"The virtual desktop is temporarily unavailable. Reconnect and try again.");
        return;
      }
      if (launches_.size() >= maximum_launches) {
        status(L"Close an app launched here before opening another app.");
        return;
      }
      const std::filesystem::path executable(control_text(executable_));
      const auto executable_name = executable.wstring();
      const DWORD attributes = GetFileAttributesW(executable_name.c_str());
      if (!executable.is_absolute() || _wcsicmp(executable.extension().c_str(), L".exe") != 0 || attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_DIRECTORY) || executable_name.find(L'\"') != std::wstring::npos) {
        status(L"Choose an application (.exe), or paste its full path.");
        return;
      }
      auto command = L"\"" + executable_name + L"\"";
      const auto arguments = control_text(arguments_);
      if (!arguments.empty()) {
        command += L" " + arguments;
      }
      if (command.size() >= 32767) {
        status(L"The app arguments are too long. Shorten them and try again.");
        return;
      }
      launch_t launched;
      launched.job = owned_handle_t(CreateJobObjectW(nullptr, nullptr));
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
      // Pending launches are killed if this helper disappears while CreateProcess is returning.
      // Atomic job assignment below prevents a suspended child escaping that cleanup interval.
      limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      if (!launched.job || !SetInformationJobObject(launched.job.get(), JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        status(L"Could not prepare this app launch: " + error_text(GetLastError()));
        return;
      }
      const auto snapshot_complete = EnumWindows(
        [](HWND window, LPARAM context) -> BOOL {
          auto &baseline = *reinterpret_cast<std::set<std::uintptr_t> *>(context);
          if (baseline.size() >= maximum_windows) {
            return FALSE;
          }
          baseline.insert(reinterpret_cast<std::uintptr_t>(window));
          return TRUE;
        },
        reinterpret_cast<LPARAM>(&launched.baseline)
      );
      if (!snapshot_complete) {
        status(L"Windows could not provide a complete window snapshot. Try again.");
        return;
      }
      FILETIME now {};
      GetSystemTimeAsFileTime(&now);
      launched.started = file_time(now);
      launched.launched = clock_t::now();
      HANDLE job_handle = launched.job.get();
      attribute_list_t attributes_list(&job_handle);
      if (!attributes_list.get()) {
        status(L"Could not prepare this app launch: " + error_text(GetLastError()));
        return;
      }
      STARTUPINFOEXW startup {};
      startup.StartupInfo.cb = sizeof(startup);
      startup.StartupInfo.dwFlags = STARTF_USEPOSITION;
      startup.StartupInfo.dwX = static_cast<DWORD>(target->info.rcWork.left + 32);
      startup.StartupInfo.dwY = static_cast<DWORD>(target->info.rcWork.top + 32);
      startup.lpAttributeList = attributes_list.get();
      PROCESS_INFORMATION process {};
      const auto directory = executable.parent_path().wstring();
      if (!CreateProcessW(
            executable_name.c_str(),
            command.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
            nullptr,
            directory.c_str(),
            &startup.StartupInfo,
            &process
          )) {
        const auto failure = GetLastError();
        status(failure == ERROR_ELEVATION_REQUIRED ? L"This app requires administrator access. Choose its normal, non-administrator version; this launcher does not elevate apps." : L"The app could not start: " + error_text(failure));
        return;
      }
      owned_handle_t process_handle(process.hProcess);
      owned_handle_t thread_handle(process.hThread);
      // No app code runs until ownership is established. A reused existing instance cannot be
      // assigned to this job, even if the just-created process forwards its request to that app.
      if (ResumeThread(thread_handle.get()) == static_cast<DWORD>(-1)) {
        const auto failure = GetLastError();
        TerminateProcess(process_handle.get(), ERROR_PROCESS_ABORTED);
        status(L"The app could not start: " + error_text(failure));
        return;
      }
      // A completed launch has an independent lifetime. Permit app-specific child breakaways,
      // which intentionally receive no placement because their ownership cannot be established.
      limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_BREAKAWAY_OK;
      if (!SetInformationJobObject(launched.job.get(), JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        status(L"The app launch could not be completed: " + error_text(GetLastError()));
        return;  // The still-pending kill-on-close job cleans up this failed launch.
      }
      launches_.push_back(std::move(launched));
      status(L"Opening app… New windows from this launch will appear here.");
    }

    void examine_window(HWND window, const target_t &target) {
      if (window == window_ || !window_kind_allowed(window)) {
        return;
      }
      DWORD process_id = 0;
      GetWindowThreadProcessId(window, &process_id);
      if (!process_id) {
        return;
      }
      if (process_id == GetCurrentProcessId()) {
        // File picker and error dialogs belong to this helper, never to another app instance.
        if (GetAncestor(window, GA_ROOTOWNER) == window_ && !own_dialogs_.contains(window)) {
          const auto fresh_target = resolve_target(display_path_);
          if (fresh_target && same_target(target, *fresh_target) && default_input_desktop()) {
            own_dialogs_.insert(window);
            place_window(window, *fresh_target);
          }
        }
        return;
      }
      owned_handle_t process(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, process_id));
      if (!process || WaitForSingleObject(process.get(), 0) != WAIT_TIMEOUT) {
        return;
      }
      const auto created = process_start(process.get());
      if (!created) {
        return;
      }
      for (auto &launched : launches_) {
        BOOL member = FALSE;
        if (*created < launched.started || !IsProcessInJob(process.get(), launched.job.get(), &member) || !member) {
          continue;
        }
        const window_key_t key {reinterpret_cast<std::uintptr_t>(window), process_id, *created};
        if (launched.baseline.contains(key.window) || launched.windows.size() >= maximum_windows) {
          return;
        }
        auto [entry, inserted] = launched.windows.try_emplace(key, observed_window_t {clock_t::now()});
        auto &observed = entry->second;
        if (observed.handled) {
          return;
        }
        if (interactive_move(window)) {
          // A user already took control of this new window. Never fight or later undo their move.
          observed.handled = true;
          return;
        }
        if (clock_t::now() - observed.first_visible < 350ms) {
          return;
        }
        const auto fresh_target = resolve_target(display_path_);
        const bool target_available = fresh_target && same_target(target, *fresh_target);
        DWORD latest_process = 0;
        GetWindowThreadProcessId(window, &latest_process);
        const desktop_launcher::window_evidence_t evidence {
          .belongs_to_launch_job = latest_process == process_id,
          .existed_before_launch = launched.baseline.contains(key.window),
          .already_handled = observed.handled,
          .visible = IsWindowVisible(window) != FALSE,
          .top_level = GetAncestor(window, GA_ROOT) == window,
          .application_window = window_kind_allowed(window),
          .cloaked = false,  // Included in window_kind_allowed, rechecked immediately above.
          .minimized = IsIconic(window) != FALSE,
          .interactive_move = interactive_move(window),
          .exact_target_available = target_available,
          .default_input_desktop = default_input_desktop(),
          .process_created = *created,
          .launch_started = launched.started,
        };
        if (!desktop_launcher::may_place_window(evidence)) {
          return;
        }
        observed.handled = true;
        if (place_window(window, *fresh_target)) {
          launched.placed_any = true;
          status(L"New app window opened in this virtual desktop.");
        } else {
          status(L"The app opened, but Windows did not allow its new window to be placed here.");
        }
        return;
      }
    }

    void poll() {
      if (WaitForSingleObject(parent_.get(), 0) != WAIT_TIMEOUT || process_start(parent_.get()) != std::optional<std::uint64_t>(parent_created_)) {
        DestroyWindow(window_);
        return;
      }
      const auto target = resolve_target(display_path_);
      const bool available = target.has_value() && default_input_desktop();
      EnableWindow(launch_, available);
      EnableWindow(browse_, available);
      if (!available) {
        if (available_) {
          ShowWindow(window_, SW_HIDE);
          available_ = false;
        }
        return;
      }
      if (!available_) {
        place_window(window_, *target);
        ShowWindow(window_, SW_SHOWNOACTIVATE);
        status(L"Virtual desktop is ready. Choose an app to open here.");
        available_ = true;
      } else if (last_target_ && !same_target(*last_target_, *target) && !IsIconic(window_)) {
        // Geometry changes only re-contain our own launcher. Previously handled app windows are
        // never revisited, even if the user has since moved them to another monitor.
        place_window(window_, *target);
        layout();
      }
      last_target_ = target;
      for (auto iterator = own_dialogs_.begin(); iterator != own_dialogs_.end();) {
        iterator = IsWindow(*iterator) ? std::next(iterator) : own_dialogs_.erase(iterator);
      }

      struct enumeration_t {
        launcher_t *self;
        const target_t *target;
        std::size_t seen = 0;
      } context {this, &*target};

      EnumWindows(
        [](HWND window, LPARAM data) -> BOOL {
          auto &context = *reinterpret_cast<enumeration_t *>(data);
          if (++context.seen > maximum_windows) {
            return FALSE;
          }
          context.self->examine_window(window, *context.target);
          return TRUE;
        },
        reinterpret_cast<LPARAM>(&context)
      );
      for (auto iterator = launches_.begin(); iterator != launches_.end();) {
        auto &launched = *iterator;
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting {};
        const bool inactive = QueryInformationJobObject(
                                launched.job.get(),
                                JobObjectBasicAccountingInformation,
                                &accounting,
                                sizeof(accounting),
                                nullptr
                              ) &&
                              accounting.ActiveProcesses == 0;
        if (!launched.placed_any && !launched.reported_no_window && (clock_t::now() - launched.launched >= 15s || inactive)) {
          status(L"No new window could be placed. The app may have reused an existing instance or opened through a Windows app broker. Existing windows stay where they are.");
          launched.reported_no_window = true;
        }
        if (inactive) {
          iterator = launches_.erase(iterator);
        } else {
          ++iterator;
        }
      }
    }

    LRESULT message(UINT message, WPARAM wparam, LPARAM lparam) {
      switch (message) {
        case WM_CREATE:
          heading_ = add_control(L"STATIC", L"Open an app in this desktop", 0);
          description_ = add_control(L"STATIC", L"Choose an app to start here. Existing windows on your PC stay where they are.", 0);
          executable_label_ = add_control(L"STATIC", L"Application (.exe)", 0);
          executable_ = add_control(L"EDIT", L"", ES_AUTOHSCROLL | WS_TABSTOP, executable_control);
          browse_ = add_control(L"BUTTON", L"Browse…", BS_PUSHBUTTON | WS_TABSTOP, browse_control);
          arguments_label_ = add_control(L"STATIC", L"Arguments (optional)", 0);
          arguments_ = add_control(L"EDIT", L"", ES_AUTOHSCROLL | WS_TABSTOP, arguments_control);
          launch_ = add_control(L"BUTTON", L"Open in this desktop", BS_DEFPUSHBUTTON | WS_TABSTOP, launch_control);
          status_ = add_control(L"STATIC", L"Choose an app above to get started.", 0);
          SendMessageW(executable_, EM_SETLIMITTEXT, 32760, 0);
          SendMessageW(arguments_, EM_SETLIMITTEXT, 32760, 0);
          SetTimer(window_, poll_timer, 250, nullptr);
          layout();
          return 0;
        case WM_COMMAND:
          if (HIWORD(wparam) == BN_CLICKED) {
            if (LOWORD(wparam) == browse_control) {
              browse();
              return 0;
            }
            if (LOWORD(wparam) == launch_control || LOWORD(wparam) == IDOK) {
              launch();
              return 0;
            }
          }
          break;
        case WM_SIZE:
          if (wparam != SIZE_MINIMIZED) {
            layout();
          }
          return 0;
        case WM_DPICHANGED:
          {
            const auto &suggested = *reinterpret_cast<RECT *>(lparam);
            SetWindowPos(window_, nullptr, suggested.left, suggested.top, suggested.right - suggested.left, suggested.bottom - suggested.top, SWP_NOZORDER | SWP_NOACTIVATE);
            layout();
            return 0;
          }
        case WM_GETMINMAXINFO:
          {
            auto *limits = reinterpret_cast<MINMAXINFO *>(lparam);
            const auto dpi = GetDpiForWindow(window_);
            limits->ptMinTrackSize = {MulDiv(520, dpi ? dpi : 96, 96), MulDiv(470, dpi ? dpi : 96, 96)};
            if (last_target_) {
              const auto &work = last_target_->info.rcWork;
              limits->ptMinTrackSize.x = std::min(limits->ptMinTrackSize.x, work.right - work.left);
              limits->ptMinTrackSize.y = std::min(limits->ptMinTrackSize.y, work.bottom - work.top);
            }
            return 0;
          }
        case WM_TIMER:
          if (wparam == poll_timer) {
            poll();
          }
          return 0;
        case WM_CLOSE:
          DestroyWindow(window_);
          return 0;
        case WM_DESTROY:
          KillTimer(window_, poll_timer);
          PostQuitMessage(0);
          return 0;
      }
      return DefWindowProcW(window_, message, wparam, lparam);
    }

    std::wstring display_path_;
    owned_handle_t parent_;
    std::uint64_t parent_created_ = 0;
    HWND window_ = nullptr;
    HWND heading_ = nullptr;
    HWND description_ = nullptr;
    HWND executable_label_ = nullptr;
    HWND executable_ = nullptr;
    HWND browse_ = nullptr;
    HWND arguments_label_ = nullptr;
    HWND arguments_ = nullptr;
    HWND launch_ = nullptr;
    HWND status_ = nullptr;
    HFONT font_ = nullptr;
    bool available_ = true;
    std::optional<target_t> last_target_;
    std::vector<HWND> controls_;
    std::vector<launch_t> launches_;
    std::set<HWND> own_dialogs_;
  };

  std::optional<std::uint64_t> unsigned_argument(std::wstring_view value) {
    if (value.empty()) {
      return std::nullopt;
    }
    std::uint64_t result = 0;
    for (const auto character : value) {
      if (character < L'0' || character > L'9' || result > (std::numeric_limits<std::uint64_t>::max() - (character - L'0')) / 10) {
        return std::nullopt;
      }
      result = result * 10 + (character - L'0');
    }
    return result;
  }

  HWND create_fixture_window() {
    constexpr wchar_t fixture_class[] = L"SunshineDesktopLauncherHiddenFixture";
    WNDCLASSW definition {};
    definition.hInstance = GetModuleHandleW(nullptr);
    definition.lpfnWndProc = DefWindowProcW;
    definition.lpszClassName = fixture_class;
    RegisterClassW(&definition);
    // Hidden from creation through destruction: this mode never creates a visible test window.
    return CreateWindowExW(0, fixture_class, L"Launcher hidden test fixture", WS_OVERLAPPEDWINDOW, 37, 41, 220, 160, nullptr, nullptr, definition.hInstance, nullptr);
  }

  std::wstring own_executable() {
    std::array<wchar_t, 32768> path {};
    const auto length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    return length && length < path.size() ? std::wstring(path.data(), length) : std::wstring {};
  }

  int fixture_child(const wchar_t *ready_name, const wchar_t *stop_name, const wchar_t *descendant_ready) {
    owned_handle_t ready(OpenEventW(EVENT_MODIFY_STATE, FALSE, ready_name));
    owned_handle_t stop(OpenEventW(SYNCHRONIZE, FALSE, stop_name));
    if (!ready || !stop) {
      return 10;
    }
    const auto window = create_fixture_window();
    if (!window) {
      return 11;
    }
    std::array<wchar_t, 128> desktop_name {};
    DWORD desktop_name_bytes = 0;
    if (GetUserObjectInformationW(GetThreadDesktop(GetCurrentThreadId()), UOI_NAME, desktop_name.data(), sizeof(desktop_name), &desktop_name_bytes) && std::wstring_view(desktop_name.data()).starts_with(L"SunshineLauncherTest-")) {
      // This desktop is never made the input desktop. The maximized fixture is invisible to the
      // user while exercising real WINDOWPLACEMENT behavior across the machine's monitors.
      ShowWindow(window, SW_SHOWMAXIMIZED);
    }
    owned_handle_t descendant;
    if (descendant_ready) {
      const auto executable = own_executable();
      auto command = L"\"" + executable + L"\" --self-test-child \"" + descendant_ready + L"\" \"" + stop_name + L"\"";
      STARTUPINFOW startup {};
      startup.cb = sizeof(startup);
      PROCESS_INFORMATION process {};
      if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup, &process)) {
        DestroyWindow(window);
        return 12;
      }
      descendant = owned_handle_t(process.hProcess);
      CloseHandle(process.hThread);
    }
    SetEvent(ready.get());
    const auto deadline = clock_t::now() + 15s;
    while (WaitForSingleObject(stop.get(), 20) == WAIT_TIMEOUT && clock_t::now() < deadline) {
      MSG message {};
      while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
    }
    DestroyWindow(window);
    if (descendant) {
      WaitForSingleObject(descendant.get(), 1000);
    }
    return 0;
  }

  /** Native regression fixture. Only windows and processes created by this mode are mutated. */
  int self_test() {
    const auto original_desktop = GetThreadDesktop(GetCurrentThreadId());
    const auto desktop_name = L"SunshineLauncherTest-" + std::to_wstring(GetCurrentProcessId());
    const auto desktop = CreateDesktopW(desktop_name.c_str(), nullptr, nullptr, 0, GENERIC_ALL, nullptr);
    if (!desktop || !SetThreadDesktop(desktop)) {
      if (desktop) {
        CloseDesktop(desktop);
      }
      return 19;
    }
    const auto sentinel = create_fixture_window();
    if (!sentinel) {
      SetThreadDesktop(original_desktop);
      CloseDesktop(desktop);
      return 20;
    }
    RECT sentinel_before {};
    GetWindowRect(sentinel, &sentinel_before);
    const auto prefix = L"Local\\SunshineDesktopLauncherTest-" + std::to_wstring(GetCurrentProcessId()) +
                        L"-" + std::to_wstring(GetTickCount64());
    const auto ready_name = prefix + L"-ready";
    const auto descendant_name = prefix + L"-descendant";
    const auto stop_name = prefix + L"-stop";
    owned_handle_t ready(CreateEventW(nullptr, TRUE, FALSE, ready_name.c_str()));
    owned_handle_t descendant_ready(CreateEventW(nullptr, TRUE, FALSE, descendant_name.c_str()));
    owned_handle_t stop(CreateEventW(nullptr, TRUE, FALSE, stop_name.c_str()));
    owned_handle_t job(CreateJobObjectW(nullptr, nullptr));
    owned_handle_t root_process;
    std::vector<owned_handle_t> fixture_processes;
    bool cross_monitor_tested = false;
    const auto run = [&]() -> int {
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
      limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      if (!ready || !descendant_ready || !stop || !job || !SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        return 21;
      }
      HANDLE job_value = job.get();
      attribute_list_t attributes(&job_value);
      STARTUPINFOEXW startup {};
      startup.StartupInfo.cb = sizeof(startup);
      auto desktop_path = L"WinSta0\\" + desktop_name;
      startup.StartupInfo.lpDesktop = desktop_path.data();
      startup.lpAttributeList = attributes.get();
      if (!startup.lpAttributeList) {
        return 22;
      }
      const auto executable = own_executable();
      auto command = L"\"" + executable + L"\" --self-test-child \"" + ready_name + L"\" \"" + stop_name + L"\" \"" + descendant_name + L"\"";
      PROCESS_INFORMATION process {};
      if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &startup.StartupInfo, &process)) {
        return 23;
      }
      root_process = owned_handle_t(process.hProcess);
      owned_handle_t initial_thread(process.hThread);
      BOOL belongs = FALSE;
      if (!IsProcessInJob(root_process.get(), job.get(), &belongs) || !belongs || ResumeThread(initial_thread.get()) == static_cast<DWORD>(-1)) {
        return 24;
      }
      if (WaitForSingleObject(ready.get(), 4000) != WAIT_OBJECT_0 || WaitForSingleObject(descendant_ready.get(), 4000) != WAIT_OBJECT_0) {
        return 25;
      }
      struct fixtures_t {
        HANDLE job;
        std::vector<std::pair<HWND, owned_handle_t>> windows;
      } fixtures {job.get(), {}};
      EnumWindows(
        [](HWND window, LPARAM value) -> BOOL {
          auto &fixtures = *reinterpret_cast<fixtures_t *>(value);
          std::array<wchar_t, 128> class_name {};
          GetClassNameW(window, class_name.data(), static_cast<int>(class_name.size()));
          if (std::wstring_view(class_name.data()) != L"SunshineDesktopLauncherHiddenFixture") {
            return TRUE;
          }
          DWORD process_id = 0;
          GetWindowThreadProcessId(window, &process_id);
          owned_handle_t process(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE | PROCESS_TERMINATE, FALSE, process_id));
          BOOL member = FALSE;
          if (process && IsProcessInJob(process.get(), fixtures.job, &member) && member) {
            fixtures.windows.emplace_back(window, std::move(process));
          }
          return TRUE;
        },
        reinterpret_cast<LPARAM>(&fixtures)
      );
      if (fixtures.windows.size() != 2) {
        return 26;
      }
      BOOL sentinel_member = TRUE;
      if (!IsProcessInJob(GetCurrentProcess(), job.get(), &sentinel_member) || sentinel_member) {
        return 27;
      }
      const auto fixture_window = fixtures.windows.front().first;
      target_t target;
      target.monitor = MonitorFromWindow(sentinel, MONITOR_DEFAULTTONEAREST);
      // Prefer another existing monitor so maximized transfer is exercised when available.
      struct monitor_choice_t {
        HMONITOR original;
        HMONITOR selected;
      } choice {target.monitor, target.monitor};
      EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR monitor, HDC, LPRECT, LPARAM value) -> BOOL {
        auto &choice = *reinterpret_cast<monitor_choice_t *>(value);
        if (monitor != choice.original) {
          choice.selected = monitor;
          return FALSE;
        }
        return TRUE;
      },
                          reinterpret_cast<LPARAM>(&choice));
      target.monitor = choice.selected;
      cross_monitor_tested = choice.selected != choice.original;
      target.info.cbSize = sizeof(target.info);
      if (!GetMonitorInfoW(target.monitor, &target.info) || !place_window(fixture_window, target)) {
        return 28;
      }
      bool contained = false;
      const auto deadline = clock_t::now() + 1s;
      while (clock_t::now() < deadline) {
        RECT positioned {};
        GetWindowRect(fixture_window, &positioned);
        if (IsZoomed(fixture_window) && MonitorFromWindow(fixture_window, MONITOR_DEFAULTTONULL) == target.monitor) {
          contained = true;
          break;
        }
        Sleep(20);
      }
      RECT sentinel_after {};
      GetWindowRect(sentinel, &sentinel_after);
      if (!contained || !EqualRect(&sentinel_before, &sentinel_after)) {
        return 29;
      }
      WINDOWPLACEMENT placement {};
      placement.length = sizeof(placement);
      if (!GetWindowPlacement(fixture_window, &placement) || !desktop_launcher::contains(rectangle(target.info.rcWork), desktop_launcher::workspace_to_screen(rectangle(placement.rcNormalPosition), rectangle(target.info.rcMonitor), rectangle(target.info.rcWork)))) {
        return 32;
      }
      limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_BREAKAWAY_OK;
      if (!SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        return 30;
      }
      for (auto &fixture : fixtures.windows) {
        fixture_processes.push_back(std::move(fixture.second));
      }
      job = owned_handle_t();
      for (const auto &process : fixture_processes) {
        if (WaitForSingleObject(process.get(), 0) != WAIT_TIMEOUT) {
          return 31;
        }
      }
      return 0;
    };
    const int result = run();
    if (stop) {
      SetEvent(stop.get());
    }
    if (root_process && WaitForSingleObject(root_process.get(), 2000) == WAIT_TIMEOUT) {
      TerminateProcess(root_process.get(), ERROR_PROCESS_ABORTED);
    }
    for (const auto &process : fixture_processes) {
      if (WaitForSingleObject(process.get(), 2000) == WAIT_TIMEOUT) {
        TerminateProcess(process.get(), ERROR_PROCESS_ABORTED);
      }
    }
    DestroyWindow(sentinel);
    SetThreadDesktop(original_desktop);
    CloseDesktop(desktop);
    if (result == 0) {
      std::fwprintf(stderr, L"Desktop launcher native self-test passed; maximized cross-monitor transfer %ls.\n", cross_monitor_tested ? L"verified" : L"not exercised (one monitor)");
    } else {
      std::fwprintf(stderr, L"Desktop launcher native self-test failed: %d.\n", result);
    }
    return result;
  }
}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
  SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
  int argument_count = 0;
  const auto arguments = CommandLineToArgvW(GetCommandLineW(), &argument_count);
  if (!arguments) {
    return 2;
  }
  if (argument_count == 2 && std::wstring_view(arguments[1]) == L"--self-test") {
    LocalFree(arguments);
    return self_test();
  }
  if ((argument_count == 4 || argument_count == 5) && std::wstring_view(arguments[1]) == L"--self-test-child") {
    const int result = fixture_child(arguments[2], arguments[3], argument_count == 5 ? arguments[4] : nullptr);
    LocalFree(arguments);
    return result;
  }
  std::wstring display_path;
  std::optional<std::uint64_t> parent_id;
  std::optional<std::uint64_t> parent_created;
  bool valid_arguments = argument_count == 7;
  for (int index = 1; valid_arguments && index + 1 < argument_count; index += 2) {
    const std::wstring_view option(arguments[index]);
    if (option == L"--display-path" && display_path.empty()) {
      display_path = arguments[index + 1];
    } else if (option == L"--parent-pid" && !parent_id) {
      parent_id = unsigned_argument(arguments[index + 1]);
    } else if (option == L"--parent-start" && !parent_created) {
      parent_created = unsigned_argument(arguments[index + 1]);
    } else {
      valid_arguments = false;
    }
  }
  LocalFree(arguments);
  if (!valid_arguments || display_path.empty() || !parent_id || !*parent_id || *parent_id > std::numeric_limits<DWORD>::max() || !parent_created || !*parent_created) {
    return 2;
  }
  owned_handle_t parent(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, static_cast<DWORD>(*parent_id)));
  if (!parent || WaitForSingleObject(parent.get(), 0) != WAIT_TIMEOUT || process_start(parent.get()) != parent_created || !default_input_desktop()) {
    return 3;
  }
  // This UI must never be an elevation bridge, including when started manually.
  owned_handle_t token;
  HANDLE raw_token = nullptr;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw_token)) {
    return 3;
  }
  token = owned_handle_t(raw_token);
  TOKEN_ELEVATION elevation {};
  DWORD returned = 0;
  if (!GetTokenInformation(token.get(), TokenElevation, &elevation, sizeof(elevation), &returned) || elevation.TokenIsElevated != 0) {
    return 3;
  }
  const auto target = resolve_target(display_path);
  if (!target) {
    return 4;
  }
  const auto com = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
  if (FAILED(com)) {
    return 5;
  }
  int result = 0;
  {
    launcher_t launcher(std::move(display_path), std::move(parent), *parent_created);
    if (!launcher.create(instance, *target)) {
      result = 6;
    } else {
      MSG message {};
      while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageW(launcher.window(), &message)) {
          TranslateMessage(&message);
          DispatchMessageW(&message);
        }
      }
    }
  }
  CoUninitialize();
  return result;
}
