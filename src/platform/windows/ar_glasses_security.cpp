#include "ar_glasses_security.h"

#include "ar_glasses.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cwctype>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>

namespace ar_glasses {
  using namespace std::literals;
  using namespace internal;

  namespace {
    bool same_windows_name(std::wstring_view left, std::wstring_view right) {
      return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(), [](wchar_t a, wchar_t b) {
               return std::towlower(a) == std::towlower(b);
             });
    }

    std::wstring window_process_name(HWND window) {
      DWORD process_id = 0;
      if (!window || !GetWindowThreadProcessId(window, &process_id)) {
        return {};
      }
      const auto process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id);
      if (!process) {
        return {};
      }
      std::array<wchar_t, 1024> path {};
      DWORD size = static_cast<DWORD>(path.size());
      const bool queried = QueryFullProcessImageNameW(process, 0, path.data(), &size);
      CloseHandle(process);
      if (!queried) {
        return {};
      }
      const std::wstring_view image(path.data(), size);
      const auto separator = image.find_last_of(L"\\/");
      return std::wstring(image.substr(separator == std::wstring_view::npos ? 0 : separator + 1));
    }

    bool lock_screen_overlay_present(std::optional<RECT> selected_monitor) {
      if (same_windows_name(window_process_name(GetForegroundWindow()), L"LockApp.exe")) {
        return true;
      }
      if (!selected_monitor) {
        return false;
      }
      const POINT center {
        static_cast<LONG>((static_cast<std::int64_t>(selected_monitor->left) + selected_monitor->right) / 2),
        static_cast<LONG>((static_cast<std::int64_t>(selected_monitor->top) + selected_monitor->bottom) / 2),
      };
      const auto hit = WindowFromPhysicalPoint(center);
      const auto shell_lock_window = [](HWND window) {
        wchar_t class_name[256] {};
        return window && GetClassNameW(window, class_name, static_cast<int>(std::size(class_name))) &&
               same_windows_name(class_name, L"LockScreenInputOcclusionFrame") &&
               same_windows_name(window_process_name(window), L"explorer.exe");
      };
      return shell_lock_window(hit) || shell_lock_window(GetAncestor(hit, GA_ROOT));
    }

    std::string_view monitor_desktop_block_reason(
      std::optional<bool> session_locked,
      std::wstring_view input_desktop,
      std::wstring_view presentation_desktop,
      bool lock_overlay_present
    ) {
      if (session_locked && *session_locked) {
        return "Unlock Windows and return to your desktop before starting Monitor 3D again."sv;
      }
      if (lock_overlay_present) {
        return "Dismiss the Windows lock or sign-in screen, then start Monitor 3D again."sv;
      }
      if (!session_locked) {
        return "Windows session state could not be verified. Return to your unlocked desktop, then start Monitor 3D again."sv;
      }
      if (!same_windows_name(input_desktop, L"Default") || !same_windows_name(presentation_desktop, L"Default")) {
        return "Monitor 3D returned to 2D for a Windows sign-in or permission screen. Start 3D again after closing it."sv;
      }
      return {};
    }
  }  // namespace

  namespace internal {
    std::wstring desktop_object_name(HANDLE desktop) {
      wchar_t name[256] {};
      DWORD size = 0;
      return desktop && GetUserObjectInformationW(desktop, UOI_NAME, name, sizeof(name), &size) ?
               std::wstring(name) :
               std::wstring {};
    }

    std::string_view query_monitor_desktop_block_reason(std::wstring_view presentation_desktop, std::optional<RECT> selected_monitor) {
      const auto locked = query_console_session_locked();
      const auto desktop = OpenInputDesktop(0, FALSE, DESKTOP_READOBJECTS);
      const auto input_name = desktop_object_name(desktop);
      if (desktop) {
        CloseDesktop(desktop);
      }
      return monitor_desktop_block_reason(locked, input_name, presentation_desktop, lock_screen_overlay_present(selected_monitor));
    }
  }  // namespace internal

#ifdef SUNSHINE_TESTS
  std::string detail::monitor_desktop_block_reason_for_test(
    std::optional<bool> session_locked,
    std::wstring_view input_desktop,
    std::wstring_view presentation_desktop,
    bool lock_overlay_present
  ) {
    return std::string(monitor_desktop_block_reason(session_locked, input_desktop, presentation_desktop, lock_overlay_present));
  }
#endif
}  // namespace ar_glasses
