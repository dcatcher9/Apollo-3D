/**
 * @file src/platform/windows/explorer_restart.cpp
 * @brief Bounded restart of the active user's Explorer shell on Windows.
 */
// standard includes
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

// prevent clang format from "optimizing" the header include order
// clang-format off
#include <Windows.h>
#include <UserEnv.h>
#include <WinUser.h>
#include <WtsApi32.h>
// clang-format on

// local includes
#include "misc.h"
#include "src/logging.h"
#include "src/utility.h"
#include "user_token.h"

namespace {

  std::mutex explorer_restart_mutex;

}  // namespace

using namespace std::literals;

namespace platf {
  namespace {
    constexpr DWORD invalid_console_session = 0xFFFFFFFF;
    constexpr auto explorer_auto_restart_grace = std::chrono::milliseconds(3000);
    constexpr auto explorer_fallback_start_grace = std::chrono::milliseconds(8000);

    const char *explorerRestartResultLabel(explorer_restart_result_e result) {
      switch (result) {
        case explorer_restart_result_e::restarted_automatically:
          return "Windows restarted Explorer automatically";
        case explorer_restart_result_e::relaunched:
          return "a replacement Explorer reached taskbar readiness";
        case explorer_restart_result_e::skipped_no_shell:
          return "no active Explorer shell was available";
        case explorer_restart_result_e::skipped_identity:
          return "the active shell identity could not be verified";
        case explorer_restart_result_e::preflight_failed:
          return "Explorer restart preflight failed";
        case explorer_restart_result_e::terminate_failed:
          return "the verified Explorer process could not be terminated";
        case explorer_restart_result_e::exit_timeout:
          return "the verified Explorer process did not exit before the timeout";
        case explorer_restart_result_e::relaunch_failed:
          return "Explorer did not restart automatically and the fallback launch failed";
        case explorer_restart_result_e::replacement_timeout:
          return "a replacement Explorer taskbar was not confirmed before the timeout";
      }
      return "unknown Explorer restart result";
    }

    bool samePathCaseInsensitive(
      const std::wstring_view lhs,
      const std::wstring_view rhs
    ) {
      return CompareStringOrdinal(
               lhs.data(),
               static_cast<int>(lhs.size()),
               rhs.data(),
               static_cast<int>(rhs.size()),
               TRUE
             ) == CSTR_EQUAL;
    }

    std::optional<std::wstring> processImagePath(HANDLE process) {
      std::array<wchar_t, 32768> path {};
      DWORD path_length = static_cast<DWORD>(path.size());
      if (!QueryFullProcessImageNameW(
            process,
            0,
            path.data(),
            &path_length
          )) {
        return std::nullopt;
      }
      return std::wstring {path.data(), path_length};
    }

    bool tokenBelongsToSession(HANDLE token, DWORD expected_session) {
      DWORD token_session = invalid_console_session;
      DWORD returned = 0;
      return GetTokenInformation(
               token,
               TokenSessionId,
               &token_session,
               sizeof(token_session),
               &returned
             ) &&
             token_session == expected_session;
    }

    std::optional<LUID> tokenAuthenticationId(HANDLE token) {
      TOKEN_STATISTICS statistics {};
      DWORD returned = 0;
      if (!GetTokenInformation(
            token,
            TokenStatistics,
            &statistics,
            sizeof(statistics),
            &returned
          )) {
        return std::nullopt;
      }
      return statistics.AuthenticationId;
    }

    bool sameLuid(const LUID &lhs, const LUID &rhs) {
      return lhs.HighPart == rhs.HighPart &&
             lhs.LowPart == rhs.LowPart;
    }

    bool currentProcessUserMatchesToken(HANDLE other_token) {
      HANDLE process_token = nullptr;
      if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &process_token)) {
        return false;
      }
      auto process_token_guard = util::fail_guard([&]() {
        CloseHandle(process_token);
      });
      const auto process_user = token_user_id(process_token);
      const auto other_user = token_user_id(other_token);
      return process_user &&
             other_user &&
             *process_user == *other_user;
    }

    class explorer_restart_operations_t {
    public:
      explorer_restart_operations_t() = default;
      explorer_restart_operations_t(const explorer_restart_operations_t &) = delete;
      explorer_restart_operations_t &operator=(const explorer_restart_operations_t &) = delete;

      ~explorer_restart_operations_t() {
        if (environment_) {
          DestroyEnvironmentBlock(environment_);
        }
        if (user_token_) {
          CloseHandle(user_token_);
        }
        if (shell_process_) {
          CloseHandle(shell_process_);
        }
      }

      std::optional<explorer_restart_result_e> preflight() {
        active_session_ = WTSGetActiveConsoleSessionId();
        if (active_session_ == invalid_console_session) {
          BOOST_LOG(warning) << "Skipping Explorer taskbar repair because no active console "
                                "session exists."sv;
          return explorer_restart_result_e::skipped_no_shell;
        }

        const auto shell_window = GetShellWindow();
        const auto tray_window = FindWindowW(L"Shell_TrayWnd", nullptr);
        DWORD tray_pid = 0;
        if (shell_window) {
          GetWindowThreadProcessId(shell_window, &shell_pid_);
        }
        if (tray_window) {
          GetWindowThreadProcessId(tray_window, &tray_pid);
        }
        if (!shell_window || shell_pid_ == 0) {
          BOOST_LOG(warning) << "Skipping Explorer taskbar repair because Windows did not expose "
                                "an active desktop shell."sv;
          return explorer_restart_result_e::skipped_no_shell;
        }
        // Shell_TrayWnd may itself be missing while the taskbar is damaged. When present it is
        // useful corroboration, but the path/session/token-validated GetShellWindow owner remains
        // authoritative.
        if ((tray_pid != 0 && shell_pid_ != tray_pid) ||
            !shellPidBelongsToOriginalSession(shell_pid_)) {
          BOOST_LOG(error) << "Skipping Explorer taskbar repair because the desktop shell and "
                              "taskbar evidence conflict in the active console session."sv;
          return explorer_restart_result_e::skipped_identity;
        }

        shell_process_ = OpenProcess(
          PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE | SYNCHRONIZE,
          FALSE,
          shell_pid_
        );
        if (!shell_process_) {
          BOOST_LOG(error) << "Could not open the active Explorer shell for taskbar repair: "
                           << GetLastError();
          return explorer_restart_result_e::preflight_failed;
        }

        std::array<wchar_t, MAX_PATH + 1> windows_directory {};
        const auto windows_length = GetWindowsDirectoryW(
          windows_directory.data(),
          static_cast<UINT>(windows_directory.size())
        );
        if (windows_length == 0 || windows_length >= windows_directory.size()) {
          BOOST_LOG(error) << "Could not resolve the Windows directory before restarting Explorer: "
                           << GetLastError();
          return explorer_restart_result_e::preflight_failed;
        }
        working_directory_ = std::wstring {
          windows_directory.data(),
          windows_length
        };
        explorer_path_ =
          (std::filesystem::path {working_directory_} / L"explorer.exe").wstring();
        const auto observed_path = processImagePath(shell_process_);
        if (!observed_path || !samePathCaseInsensitive(*observed_path, explorer_path_)) {
          BOOST_LOG(error) << "Skipping Explorer taskbar repair because the active shell image "
                              "is not the Windows Explorer executable."sv;
          return explorer_restart_result_e::skipped_identity;
        }

        HANDLE shell_token = nullptr;
        if (!OpenProcessToken(
              shell_process_,
              TOKEN_QUERY | TOKEN_DUPLICATE,
              &shell_token
            )) {
          BOOST_LOG(error) << "Could not inspect the active Explorer shell token: "
                           << GetLastError();
          return explorer_restart_result_e::preflight_failed;
        }
        auto shell_token_guard = util::fail_guard([&]() {
          CloseHandle(shell_token);
        });
        TOKEN_ELEVATION shell_elevation {};
        DWORD returned = 0;
        if (!GetTokenInformation(
              shell_token,
              TokenElevation,
              &shell_elevation,
              sizeof(shell_elevation),
              &returned
            ) ||
            !tokenBelongsToSession(shell_token, active_session_)) {
          BOOST_LOG(error) << "Skipping Explorer taskbar repair because the shell does not use "
                              "the expected token in the active console session."sv;
          return explorer_restart_result_e::skipped_identity;
        }
        shell_elevated_ = shell_elevation.TokenIsElevated != 0;
        const auto shell_user = token_user_id(shell_token);
        const auto shell_authentication = tokenAuthenticationId(shell_token);
        if (!shell_user || !shell_authentication) {
          BOOST_LOG(error) << "Could not identify the active Explorer shell logon."sv;
          return explorer_restart_result_e::preflight_failed;
        }
        if (!is_running_as_system() &&
            !currentProcessUserMatchesToken(shell_token)) {
          BOOST_LOG(error) << "Skipping Explorer taskbar repair because Sunshine 3D does not "
                              "belong to the active shell's user account."sv;
          return explorer_restart_result_e::skipped_identity;
        }

        // Duplicate the exact validated shell token. This preserves standard UAC sessions and
        // deliberately also supports UAC-disabled administrators whose Explorer is elevated.
        if (!DuplicateTokenEx(
              shell_token,
              MAXIMUM_ALLOWED,
              nullptr,
              SecurityImpersonation,
              TokenPrimary,
              &user_token_
            )) {
          BOOST_LOG(error) << "Could not capture the active Explorer token before restarting it: "
                           << GetLastError();
          return explorer_restart_result_e::preflight_failed;
        }
        user_id_ = token_user_id(user_token_).value_or(std::string {});
        authentication_id_ = tokenAuthenticationId(user_token_);
        if (user_id_.empty() ||
            user_id_ != *shell_user ||
            !authentication_id_ ||
            !sameLuid(*authentication_id_, *shell_authentication) ||
            !tokenBelongsToSession(user_token_, active_session_)) {
          BOOST_LOG(error) << "Skipping Explorer taskbar repair because the captured fallback "
                              "token does not match the active shell."sv;
          return explorer_restart_result_e::skipped_identity;
        }
        if (!CreateEnvironmentBlock(&environment_, user_token_, FALSE)) {
          BOOST_LOG(error) << "Could not capture the active user's environment before restarting "
                              "Explorer: " << GetLastError();
          return explorer_restart_result_e::preflight_failed;
        }

        // Close the last PID-reuse/session-switch window before the controller is allowed to kill.
        if (!currentWindowsIdentifyShell(shell_pid_)) {
          BOOST_LOG(warning) << "Skipping Explorer taskbar repair because the active shell changed "
                                "during preflight."sv;
          return explorer_restart_result_e::skipped_identity;
        }
        return std::nullopt;
      }

      bool terminate_exact_shell() {
        if (!shell_process_ || !currentWindowsIdentifyShell(shell_pid_)) {
          return false;
        }
        if (WaitForSingleObject(shell_process_, 0) == WAIT_OBJECT_0) {
          return true;
        }
        if (!TerminateProcess(shell_process_, ERROR_PROCESS_ABORTED)) {
          const auto terminate_error = GetLastError();
          if (WaitForSingleObject(shell_process_, 0) == WAIT_OBJECT_0) {
            return true;
          }
          BOOST_LOG(error) << "Could not terminate the verified Explorer shell (PID "
                           << shell_pid_ << "): " << terminate_error;
          return false;
        }
        BOOST_LOG(info) << "Restarting the active user's Explorer shell (PID "
                        << shell_pid_ << ") after confirmed virtual-display removal."sv;
        return true;
      }

      bool wait_for_old_shell_exit() {
        return shell_process_ &&
               WaitForSingleObject(shell_process_, 5000) == WAIT_OBJECT_0;
      }

      bool wait_for_replacement(bool after_fallback_launch) {
        const auto timeout = after_fallback_launch ?
                               explorer_fallback_start_grace :
                               explorer_auto_restart_grace;
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true) {
          if (replacementShellPresent(true)) {
            return true;
          }
          const auto now = std::chrono::steady_clock::now();
          if (now >= deadline) {
            return false;
          }
          std::this_thread::sleep_for(
            std::min<std::chrono::steady_clock::duration>(100ms, deadline - now)
          );
        }
      }

      bool replacement_present_now() const {
        return replacementShellPresent(false);
      }

      bool launch_captured_shell() {
        if (!user_token_ || !environment_ || explorer_path_.empty()) {
          return false;
        }
        if (WTSGetActiveConsoleSessionId() != active_session_) {
          BOOST_LOG(warning) << "Refusing to relaunch Explorer because the active console session "
                                "changed during taskbar repair."sv;
          return false;
        }
        // Close the remaining AutoRestartShell race after the controller's explicit recheck.
        if (replacementShellPresent(false)) {
          return true;
        }

        STARTUPINFOW startup_info {};
        startup_info.cb = sizeof(startup_info);
        wchar_t interactive_desktop[] = L"winsta0\\default";
        startup_info.lpDesktop = interactive_desktop;
        PROCESS_INFORMATION process_info {};
        const DWORD creation_flags =
          CREATE_UNICODE_ENVIRONMENT | CREATE_BREAKAWAY_FROM_JOB;
        if (!CreateProcessAsUserW(
              user_token_,
              explorer_path_.c_str(),
              nullptr,
              nullptr,
              nullptr,
              FALSE,
              creation_flags,
              environment_,
              working_directory_.c_str(),
              &startup_info,
              &process_info
            )) {
          BOOST_LOG(error) << "Could not launch the Explorer fallback in the active user's "
                              "desktop session: " << GetLastError();
          return false;
        }
        CloseHandle(process_info.hThread);
        CloseHandle(process_info.hProcess);
        BOOST_LOG(info) << "Windows did not auto-restart Explorer; launched one fallback with "
                           "the captured shell token."sv;
        return true;
      }

    private:
      bool shellPidBelongsToOriginalSession(DWORD pid) const {
        DWORD process_session = invalid_console_session;
        return pid != 0 &&
               ProcessIdToSessionId(pid, &process_session) &&
               process_session == active_session_;
      }

      bool currentWindowsIdentifyShell(DWORD expected_pid) const {
        if (WTSGetActiveConsoleSessionId() != active_session_) {
          return false;
        }
        const auto shell_window = GetShellWindow();
        const auto tray_window = FindWindowW(L"Shell_TrayWnd", nullptr);
        DWORD current_shell_pid = 0;
        DWORD current_tray_pid = 0;
        if (shell_window) {
          GetWindowThreadProcessId(shell_window, &current_shell_pid);
        }
        if (tray_window) {
          GetWindowThreadProcessId(tray_window, &current_tray_pid);
        }
        return current_shell_pid == expected_pid &&
               (current_tray_pid == 0 || current_tray_pid == expected_pid) &&
               shellPidBelongsToOriginalSession(expected_pid);
      }

      bool replacementShellPresent(bool require_taskbar) const {
        if (WTSGetActiveConsoleSessionId() != active_session_) {
          return false;
        }
        const auto shell_window = GetShellWindow();
        const auto tray_window = FindWindowW(L"Shell_TrayWnd", nullptr);
        DWORD replacement_pid = 0;
        DWORD tray_pid = 0;
        if (shell_window) {
          GetWindowThreadProcessId(shell_window, &replacement_pid);
        }
        if (tray_window) {
          GetWindowThreadProcessId(tray_window, &tray_pid);
        }
        if (replacement_pid == 0 ||
            replacement_pid == shell_pid_ ||
            (require_taskbar ?
               replacement_pid != tray_pid :
               (tray_pid != 0 && replacement_pid != tray_pid)) ||
            !shellPidBelongsToOriginalSession(replacement_pid)) {
          return false;
        }

        HANDLE replacement_process = OpenProcess(
          PROCESS_QUERY_LIMITED_INFORMATION,
          FALSE,
          replacement_pid
        );
        if (!replacement_process) {
          return false;
        }
        auto process_guard = util::fail_guard([&]() {
          CloseHandle(replacement_process);
        });
        const auto observed_path = processImagePath(replacement_process);
        if (!observed_path || !samePathCaseInsensitive(*observed_path, explorer_path_)) {
          return false;
        }

        HANDLE replacement_token = nullptr;
        if (!OpenProcessToken(replacement_process, TOKEN_QUERY, &replacement_token)) {
          return false;
        }
        auto token_guard = util::fail_guard([&]() {
          CloseHandle(replacement_token);
        });
        TOKEN_ELEVATION replacement_elevation {};
        DWORD returned = 0;
        const auto replacement_authentication =
          tokenAuthenticationId(replacement_token);
        return GetTokenInformation(
                 replacement_token,
                 TokenElevation,
                 &replacement_elevation,
                 sizeof(replacement_elevation),
                 &returned
               ) &&
               (replacement_elevation.TokenIsElevated != 0) == shell_elevated_ &&
               replacement_authentication &&
               authentication_id_ &&
               sameLuid(*replacement_authentication, *authentication_id_) &&
               tokenBelongsToSession(replacement_token, active_session_) &&
               token_user_id(replacement_token) == std::optional<std::string> {user_id_};
      }

      DWORD active_session_ = invalid_console_session;
      DWORD shell_pid_ = 0;
      HANDLE shell_process_ = nullptr;
      HANDLE user_token_ = nullptr;
      PVOID environment_ = nullptr;
      std::wstring explorer_path_;
      std::wstring working_directory_;
      std::string user_id_;
      std::optional<LUID> authentication_id_;
      bool shell_elevated_ = false;
    };
  }  // namespace

  explorer_restart_result_e restart_active_user_explorer() {
    std::lock_guard restart_lock {explorer_restart_mutex};
    explorer_restart_operations_t operations;
    const auto result = detail::run_explorer_restart(operations);
    switch (result) {
      case explorer_restart_result_e::restarted_automatically:
      case explorer_restart_result_e::relaunched:
        BOOST_LOG(info) << "Explorer taskbar repair completed: "
                        << explorerRestartResultLabel(result) << '.';
        break;
      default:
        BOOST_LOG(warning) << "Explorer taskbar repair was not confirmed: "
                           << explorerRestartResultLabel(result) << '.';
        break;
    }
    return result;
  }
}  // namespace platf
