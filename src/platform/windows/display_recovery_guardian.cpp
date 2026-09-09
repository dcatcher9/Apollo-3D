/**
 * @file src/platform/windows/display_recovery_guardian.cpp
 * @brief Hidden process guardian for recoverable virtual-only display sessions.
 */
#include "display_recovery_guardian.h"

#include "misc.h"
#include "primary_display.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/utility.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <shellapi.h>
#include <vector>

namespace platf::display_recovery_guardian {
  namespace {
    constexpr std::wstring_view worker_command = L"--display-recovery-guardian";

    struct guardian_t {
      std::mutex mutex;
      HANDLE process = nullptr;
      std::filesystem::path config_path;

      ~guardian_t() {
        // Closing our observation handle must not terminate the independently running worker.
        if (process) {
          CloseHandle(process);
        }
      }
    };

    guardian_t guardian;

    bool launch_error(const char *operation, DWORD error_code) {
      BOOST_LOG(error) << "Display recovery guardian " << operation << " failed: " << error_code;
      return false;
    }

    std::optional<std::filesystem::path> current_config_path() {
      if (config::sunshine.config_file.empty()) {
        return std::nullopt;
      }
      std::error_code error;
      const auto path = std::filesystem::absolute(std::filesystem::path(from_utf8(config::sunshine.config_file)), error);
      if (error || path.filename().empty()) {
        return std::nullopt;
      }
      // Match primary_display's lock identity. Resolving the filename itself could change the
      // journal name if the configured file is a symlink.
      const auto parent = std::filesystem::weakly_canonical(path.parent_path(), error);
      return error ? std::nullopt : std::make_optional(parent / path.filename());
    }
  }  // namespace

  namespace detail {
    std::wstring quote_argument(std::wstring_view value) {
      std::wstring result = L"\"";
      size_t backslashes = 0;
      for (const auto character : value) {
        if (character == L'\\') {
          ++backslashes;
          continue;
        }
        result.append(character == L'"' ? backslashes * 2 + 1 : backslashes, L'\\');
        result += character;
        backslashes = 0;
      }
      result.append(backslashes * 2, L'\\');
      result += L'"';
      return result;
    }

    std::optional<HANDLE> parse_handle(std::wstring_view value) {
      if (value.empty()) {
        return std::nullopt;
      }
      std::uintptr_t parsed = 0;
      for (const auto character : value) {
        if (character < L'0' || character > L'9') {
          return std::nullopt;
        }
        const auto digit = static_cast<std::uintptr_t>(character - L'0');
        if (parsed > (std::numeric_limits<std::uintptr_t>::max() - digit) / 10) {
          return std::nullopt;
        }
        parsed = parsed * 10 + digit;
      }
      if (parsed == 0 || parsed == std::numeric_limits<std::uintptr_t>::max()) {
        return std::nullopt;
      }
      return reinterpret_cast<HANDLE>(parsed);
    }

    DWORD watch_parent(HANDLE parent, HANDLE ready, const std::function<bool()> &recover, DWORD retry_ms) {
      const auto parent_id = GetProcessId(parent);
      if (!parent_id || parent_id == GetCurrentProcessId() || !recover || retry_ms == 0) {
        return ERROR_INVALID_PARAMETER;
      }
      // Validate SYNCHRONIZE before advertising readiness. A dead parent is valid: the host
      // may have crashed between creating this process and its first instruction.
      const auto initial = WaitForSingleObject(parent, 0);
      if (initial != WAIT_TIMEOUT && initial != WAIT_OBJECT_0) {
        return ERROR_INVALID_HANDLE;
      }
      if (!SetEvent(ready)) {
        return GetLastError();
      }
      if (WaitForSingleObject(parent, INFINITE) != WAIT_OBJECT_0) {
        return ERROR_INVALID_HANDLE;
      }
      // recover() owns the existing exclusive configuration lock. A restarted normal host can
      // win that lock first; the guardian then waits rather than touching its live topology.
      while (!recover()) {
        Sleep(retry_ms);
      }
      return ERROR_SUCCESS;
    }
  }  // namespace detail

  bool ensure_running() {
    try {
      std::lock_guard lock(guardian.mutex);
      const auto config_path = current_config_path();
      if (!config_path) {
        return launch_error("configuration identity", ERROR_INVALID_NAME);
      }
      if (guardian.process) {
        if (WaitForSingleObject(guardian.process, 0) == WAIT_TIMEOUT) {
          return guardian.config_path == *config_path;
        }
        CloseHandle(guardian.process);
        guardian.process = nullptr;
      }

      std::wstring executable(32768, L'\0');
      const auto length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
      if (length == 0 || length >= executable.size()) {
        return launch_error("executable identity", ERROR_BAD_PATHNAME);
      }
      executable.resize(length);

      HANDLE parent = nullptr;
      if (!DuplicateHandle(GetCurrentProcess(), GetCurrentProcess(), GetCurrentProcess(), &parent, SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, TRUE, 0)) {
        return launch_error("parent handle", GetLastError());
      }
      const auto close_parent = util::fail_guard([&]() {
        CloseHandle(parent);
      });
      SECURITY_ATTRIBUTES inheritable {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
      const auto ready = CreateEventW(&inheritable, TRUE, FALSE, nullptr);
      if (!ready) {
        return launch_error("readiness event", GetLastError());
      }
      const auto close_ready = util::fail_guard([&]() {
        CloseHandle(ready);
      });

      SIZE_T bytes = 0;
      InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
      if (bytes == 0) {
        return launch_error("handle-list size", GetLastError());
      }
      std::vector<std::byte> attribute_storage(bytes);
      const auto attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_storage.data());
      if (!InitializeProcThreadAttributeList(attributes, 1, 0, &bytes)) {
        return launch_error("handle-list initialization", GetLastError());
      }
      const auto delete_attributes = util::fail_guard([&]() {
        DeleteProcThreadAttributeList(attributes);
      });
      std::array<HANDLE, 2> inherited {parent, ready};
      if (!UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(), sizeof(inherited), nullptr, nullptr)) {
        return launch_error("handle-list restriction", GetLastError());
      }

      auto command = detail::quote_argument(executable) + L" " + std::wstring(worker_command) + L" " +
                     std::to_wstring(reinterpret_cast<std::uintptr_t>(parent)) + L" " +
                     std::to_wstring(reinterpret_cast<std::uintptr_t>(ready)) + L" " +
                     detail::quote_argument(config_path->wstring());
      if (command.size() >= 32767) {
        return launch_error("command length", ERROR_FILENAME_EXCED_RANGE);
      }
      STARTUPINFOEXW startup {};
      startup.StartupInfo.cb = sizeof(startup);
      startup.StartupInfo.dwFlags = STARTF_USESHOWWINDOW;
      startup.StartupInfo.wShowWindow = SW_HIDE;
      startup.lpAttributeList = attributes;
      PROCESS_INFORMATION child {};
      // Never inherit the host's kill-on-close job or its primary-display ownership file handle.
      // If breakaway is prohibited, physical outputs must remain enabled.
      constexpr DWORD flags = CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT |
                              EXTENDED_STARTUPINFO_PRESENT | CREATE_BREAKAWAY_FROM_JOB;
      if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE, flags, nullptr, config_path->parent_path().c_str(), &startup.StartupInfo, &child)) {
        return launch_error("independent process creation", GetLastError());
      }
      CloseHandle(child.hThread);
      auto stop_unready_child = util::fail_guard([&]() {
        // This is only the child created above, and its parent is still alive: it cannot yet
        // be recovering displays. Do not leave an unacknowledged guardian behind on failure.
        TerminateProcess(child.hProcess, ERROR_PROCESS_ABORTED);
        WaitForSingleObject(child.hProcess, 2000);
        CloseHandle(child.hProcess);
      });
      std::array<HANDLE, 2> startup_wait {child.hProcess, ready};
      const auto result = WaitForMultipleObjects(static_cast<DWORD>(startup_wait.size()), startup_wait.data(), FALSE, 20000);
      const auto wait_error = result == WAIT_FAILED ? GetLastError() : ERROR_SUCCESS;
      if (result != WAIT_OBJECT_0 + 1 || WaitForSingleObject(child.hProcess, 0) != WAIT_TIMEOUT) {
        DWORD exit_code = STILL_ACTIVE;
        const auto have_exit_code = GetExitCodeProcess(child.hProcess, &exit_code);
        BOOST_LOG(error) << "Display recovery guardian readiness handshake failed: wait=" << result
                         << ", wait error=" << wait_error << ", worker exit="
                         << (have_exit_code ? std::to_string(exit_code) : "unavailable")
                         << ". Recovery log: " << to_utf8(config_path->wstring())
                         << ".display-recovery-" << child.dwProcessId << ".log";
        return false;
      }
      BOOL in_job = TRUE;
      if (!IsProcessInJob(child.hProcess, nullptr, &in_job) || in_job) {
        return launch_error("job independence", ERROR_ACCESS_DENIED);
      }
      guardian.config_path = *config_path;
      guardian.process = child.hProcess;
      stop_unready_child.disable();
      BOOST_LOG(info) << "Independent display recovery guardian is ready (PID " << child.dwProcessId << ").";
      return true;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not establish display recovery guardian: " << exception.what();
      return false;
    }
  }

  std::optional<int> run_if_requested() {
    int count = 0;
    const auto arguments = CommandLineToArgvW(GetCommandLineW(), &count);
    if (!arguments) {
      return std::nullopt;
    }
    const auto free_arguments = util::fail_guard([&]() {
      LocalFree(arguments);
    });
    if (count < 2 || std::wstring_view(arguments[1]) != worker_command) {
      return std::nullopt;
    }
    if (count != 5) {
      return ERROR_INVALID_PARAMETER;
    }
    const auto parent = detail::parse_handle(arguments[2]);
    const auto ready = detail::parse_handle(arguments[3]);
    const std::filesystem::path config_path(arguments[4]);
    if (!parent || !ready || *parent == *ready || !config_path.is_absolute() || config_path.filename().empty()) {
      return ERROR_INVALID_PARAMETER;
    }
    const auto close_handles = util::fail_guard([&]() {
      CloseHandle(*parent);
      CloseHandle(*ready);
    });
    config::sunshine.config_file = to_utf8(config_path.wstring());
    const auto recovery_log = config::sunshine.config_file + ".display-recovery-" + std::to_string(GetCurrentProcessId()) + ".log";
    auto log_guard = logging::init(2, recovery_log);
    if (!log_guard) {
      return ERROR_OPEN_FAILED;
    }
    DWORD parent_flags = 0, ready_flags = 0;
    if (!GetHandleInformation(*parent, &parent_flags) || !(parent_flags & HANDLE_FLAG_INHERIT) || !GetHandleInformation(*ready, &ready_flags) || !(ready_flags & HANDLE_FLAG_INHERIT)) {
      BOOST_LOG(error) << "Display recovery guardian requires valid inherited parent and readiness handles.";
      logging::log_flush();
      return ERROR_ACCESS_DENIED;
    }
    if (!SetHandleInformation(*parent, HANDLE_FLAG_INHERIT, 0) || !SetHandleInformation(*ready, HANDLE_FLAG_INHERIT, 0)) {
      const auto error_code = GetLastError();
      BOOST_LOG(error) << "Display recovery guardian could not protect its inherited handles: " << error_code;
      logging::log_flush();
      return static_cast<int>(error_code);
    }
    BOOL in_job = TRUE;
    if (!IsProcessInJob(GetCurrentProcess(), nullptr, &in_job) || in_job) {
      // Breakaway can succeed for an inner job while an outer job still contains us. The
      // immediate job's limits cannot prove that every ancestor will survive the host.
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
      const auto have_limits = QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &limits, sizeof(limits), nullptr);
      BOOST_LOG(error) << "Display recovery guardian could not establish job independence; refusing readiness. "
                          "An enclosing process job may terminate the worker with the host. Immediate job limit flags: "
                       << (have_limits ? std::to_string(limits.BasicLimitInformation.LimitFlags) : "unavailable") << '.';
      logging::log_flush();
      return ERROR_ACCESS_DENIED;
    }
    BOOST_LOG(info) << "Display recovery guardian is waiting for host PID " << GetProcessId(*parent) << " to exit.";
    logging::log_flush();
    bool recovery_started = false;
    const auto result = detail::watch_parent(*parent, *ready, [&]() {
      if (!recovery_started) {
        BOOST_LOG(info) << "Host process exited; attempting saved display recovery.";
        recovery_started = true;
      }
      return primary_display::recover();
    });
    BOOST_LOG(info) << "Display recovery guardian finished with status " << result << '.';
    logging::log_flush();
    return static_cast<int>(result);
  }
}  // namespace platf::display_recovery_guardian
