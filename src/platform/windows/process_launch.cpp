/**
 * @file src/platform/windows/process_launch.cpp
 * @brief Command resolution, user process launch, and process groups on Windows.
 */
// standard includes
#include <array>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <io.h>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#ifndef BOOST_PROCESS_VERSION
 #define BOOST_PROCESS_VERSION 1
#endif

// lib includes
#include <boost/algorithm/string.hpp>
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/group.hpp>
#include <boost/process/v1/environment.hpp>
#include <boost/program_options/parsers.hpp>

// prevent clang format from "optimizing" the header include order
// clang-format off
#include <Windows.h>
#include <UserEnv.h>
#include <WinUser.h>
#include <WtsApi32.h>
// clang-format on

// Boost overrides NTDDI_VERSION, so we re-override it here
#undef NTDDI_VERSION
#define NTDDI_VERSION NTDDI_WIN10
#include <Shlwapi.h>

// local includes
#include "misc.h"
#include "process_environment.h"
#include "src/entry_handler.h"
#include "src/logging.h"
#include "src/platform/common.h"
#include "src/utility.h"
#include "user_token.h"

// PROC_THREAD_ATTRIBUTE_JOB_LIST is currently missing from MinGW headers
#ifndef PROC_THREAD_ATTRIBUTE_JOB_LIST
  #define PROC_THREAD_ATTRIBUTE_JOB_LIST ProcThreadAttributeValue(13, FALSE, TRUE, FALSE)
#endif

namespace {

  std::mutex command_resolution_mutex;

}  // namespace

namespace bp = boost::process;

using namespace std::literals;

namespace platf {
  namespace {
    /**
     * Temporarily enable privileges on the process token and restore their exact prior state.
     *
     * Process creation must not rely on Windows implicitly selecting or enabling privileges. In
     * particular, an impersonating thread has a different effective token from the process token
     * whose privileges authorize the cross-token launch. Keep that distinction explicit and fail
     * closed when a requested privilege is not present in the process token.
     */
    class scoped_process_privileges_t {
    public:
      explicit scoped_process_privileges_t(
        std::initializer_list<const char *> privilege_names
      ) {
        // Reserve before changing token state so allocation failure cannot strand a privilege in
        // its enabled state while this object's constructor is still incomplete.
        previous_states_.reserve(privilege_names.size());
        if (!OpenProcessToken(
              GetCurrentProcess(),
              TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
              &process_token_
            )) {
          error_ = GetLastError();
          return;
        }

        for (const auto privilege_name : privilege_names) {
          LUID privilege_luid {};
          if (!LookupPrivilegeValueA(nullptr, privilege_name, &privilege_luid)) {
            error_ = GetLastError();
            return;
          }

          TOKEN_PRIVILEGES requested {};
          requested.PrivilegeCount = 1;
          requested.Privileges[0].Luid = privilege_luid;
          requested.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

          TOKEN_PRIVILEGES previous {};
          DWORD previous_size = sizeof(previous);
          SetLastError(ERROR_SUCCESS);
          if (!AdjustTokenPrivileges(
                process_token_,
                FALSE,
                &requested,
                sizeof(previous),
                &previous,
                &previous_size
              )) {
            error_ = GetLastError();
            return;
          }
          const auto adjust_error = GetLastError();
          if (adjust_error == ERROR_NOT_ALL_ASSIGNED) {
            error_ = adjust_error;
            return;
          }
          if (adjust_error != ERROR_SUCCESS) {
            error_ = adjust_error;
            return;
          }
          if (previous.PrivilegeCount != 0) {
            previous_states_.push_back(previous);
          }

          PRIVILEGE_SET required {};
          required.PrivilegeCount = 1;
          required.Control = PRIVILEGE_SET_ALL_NECESSARY;
          required.Privilege[0].Luid = privilege_luid;
          required.Privilege[0].Attributes = SE_PRIVILEGE_ENABLED;
          BOOL enabled = FALSE;
          if (!PrivilegeCheck(process_token_, &required, &enabled)) {
            error_ = GetLastError();
            return;
          }
          if (!enabled) {
            error_ = ERROR_PRIVILEGE_NOT_HELD;
            return;
          }
        }
      }

      scoped_process_privileges_t(const scoped_process_privileges_t &) = delete;
      scoped_process_privileges_t &operator=(const scoped_process_privileges_t &) = delete;

      ~scoped_process_privileges_t() {
        if (!process_token_) {
          return;
        }
        for (auto state = previous_states_.rbegin(); state != previous_states_.rend(); ++state) {
          if (!AdjustTokenPrivileges(
                process_token_,
                FALSE,
                &*state,
                0,
                nullptr,
                nullptr
              )) {
            BOOST_LOG(error) << "Failed to restore a process-token privilege after user launch: "sv
                             << GetLastError();
          }
        }
        CloseHandle(process_token_);
      }

      [[nodiscard]] explicit operator bool() const {
        return error_ == ERROR_SUCCESS;
      }

      [[nodiscard]] DWORD win32_error() const {
        return error_;
      }

    private:
      HANDLE process_token_ = nullptr;
      DWORD error_ = ERROR_SUCCESS;
      std::vector<TOKEN_PRIVILEGES> previous_states_;
    };

    std::optional<DWORD> get_token_session_id(HANDLE token) {
      DWORD session_id = 0xFFFFFFFF;
      DWORD returned = 0;
      if (!GetTokenInformation(
            token,
            TokenSessionId,
            &session_id,
            sizeof(session_id),
            &returned
          )) {
        return std::nullopt;
      }
      return session_id;
    }
  }  // namespace

  bool merge_user_environment_block(bp::environment &env, HANDLE shell_token) {
    // Get the target user's environment block
    PVOID env_block;
    if (!CreateEnvironmentBlock(&env_block, shell_token, FALSE)) {
      return false;
    }

    // Parse the environment block and populate env
    for (auto c = (PWCHAR) env_block; *c != UNICODE_NULL; c += wcslen(c) + 1) {
      // Environment variable entries end with a null-terminator, so std::wstring() will get an entire entry.
      std::string env_tuple = to_utf8(std::wstring {c});
      std::string env_name = env_tuple.substr(0, env_tuple.find('='));
      std::string env_val = env_tuple.substr(env_tuple.find('=') + 1);

      // Perform a case-insensitive search to see if this variable name already exists
      auto itr = std::find_if(env.cbegin(), env.cend(), [&](const auto &e) {
        return boost::iequals(e.get_name(), env_name);
      });
      if (itr != env.cend()) {
        // Use this existing name if it is already present to ensure we merge properly
        env_name = itr->get_name();
      }

      // For the PATH variable, we will merge the values together
      if (boost::iequals(env_name, "PATH")) {
        const auto existing_path = env[env_name].to_string();
        env[env_name] = existing_path.empty() ?
                          env_val :
                          env_val + ";" + existing_path;
      } else {
        // Other variables will be superseded by those in the user's environment block
        env[env_name] = env_val;
      }
    }

    DestroyEnvironmentBlock(env_block);
    return true;
  }

  std::wstring create_environment_block(bp::environment &env) {
    std::wstring env_block;
    for (const auto &entry : env) {
      // Construct the NAME=VAL\0 string
      env_block.append(from_utf8(entry.get_name()));
      env_block.push_back(L'=');
      env_block.append(from_utf8(entry.to_string()));
      env_block.push_back(L'\0');
    }

    // Append a final null terminator
    env_block.push_back(L'\0');

    return env_block;
  }

  LPPROC_THREAD_ATTRIBUTE_LIST allocate_proc_thread_attr_list(DWORD attribute_count) {
    SIZE_T size;
    InitializeProcThreadAttributeList(nullptr, attribute_count, 0, &size);

    auto list = (LPPROC_THREAD_ATTRIBUTE_LIST) HeapAlloc(GetProcessHeap(), 0, size);
    if (list == nullptr) {
      return nullptr;
    }

    if (!InitializeProcThreadAttributeList(list, attribute_count, 0, &size)) {
      HeapFree(GetProcessHeap(), 0, list);
      return nullptr;
    }

    return list;
  }

  void free_proc_thread_attr_list(LPPROC_THREAD_ATTRIBUTE_LIST list) {
    DeleteProcThreadAttributeList(list);
    HeapFree(GetProcessHeap(), 0, list);
  }

  /**
   * @brief Create a `bp::child` object from the results of launching a process.
   * @param process_launched A boolean indicating if the launch was successful.
   * @param cmd The command that was used to launch the process.
   * @param ec A reference to an `std::error_code` object that will store any error that occurred during the launch.
   * @param process_info A reference to a `PROCESS_INFORMATION` structure that contains information about the new process.
   * @return A `bp::child` object representing the new process, or an empty `bp::child` object if the launch failed.
   */
  bp::child create_boost_child_from_results(
    bool process_launched,
    const DWORD process_launch_error,
    const std::string &cmd,
    std::error_code &ec,
    PROCESS_INFORMATION &process_info
  ) {
    // Use RAII to ensure the process is closed when we're done with it, even if there was an error.
    auto close_process_handles = util::fail_guard([process_launched, process_info]() {
      if (process_launched) {
        CloseHandle(process_info.hThread);
        CloseHandle(process_info.hProcess);
      }
    });

    if (ec) {
      // If there was an error, return an empty bp::child object
      return bp::child();
    }

    if (process_launched) {
      // If the launch was successful, create a new bp::child object representing the new process
      auto child = bp::child((bp::pid_t) process_info.dwProcessId);
      BOOST_LOG(info) << cmd << " running with PID "sv << child.id();
      return child;
    } else {
      const auto winerror = process_launch_error == ERROR_SUCCESS ?
                              ERROR_GEN_FAILURE :
                              process_launch_error;
      BOOST_LOG(error) << "Failed to launch process: "sv << winerror;
      ec = std::error_code {
        static_cast<int>(winerror),
        std::system_category(),
      };
      // We must NOT attach the failed process here, since this case can potentially be induced by ACL
      // manipulation (denying yourself execute permission) to cause an escalation of privilege.
      // So to protect ourselves against that, we'll return an empty child process instead.
      return bp::child();
    }
  }

  /**
   * @brief Create a `STARTUPINFOEXW` structure for launching a process.
   * @param file A pointer to a `FILE` object that will be used as the standard output and error for the new process, or null if not needed.
   * @param job A job object handle to insert the new process into. This pointer must remain valid for the life of this startup info!
   * @param ec A reference to a `std::error_code` object that will store any error that occurred during the creation of the structure.
   * @return A structure that contains information about how to launch the new process.
   */
  STARTUPINFOEXW create_startup_info(FILE *file, HANDLE *job, std::error_code &ec) {
    // Initialize a zeroed-out STARTUPINFOEXW structure and set its size
    STARTUPINFOEXW startup_info = {};
    startup_info.StartupInfo.cb = sizeof(startup_info);

    // Allocate a process attribute list with space for 2 elements
    startup_info.lpAttributeList = allocate_proc_thread_attr_list(2);
    if (startup_info.lpAttributeList == nullptr) {
      // If the allocation failed, set ec to an appropriate error code and return the structure
      ec = std::make_error_code(std::errc::not_enough_memory);
      return startup_info;
    }
    const auto fail_attribute_setup = [&](const DWORD windows_error) {
      free_proc_thread_attr_list(startup_info.lpAttributeList);
      startup_info.lpAttributeList = nullptr;
      ec = std::error_code {
        static_cast<int>(windows_error),
        std::system_category(),
      };
    };

    if (file) {
      // If a file was provided, get its handle and use it as the standard output and error for the new process
      HANDLE log_file_handle = (HANDLE) _get_osfhandle(_fileno(file));
      HANDLE inheritable_log_handle = INVALID_HANDLE_VALUE;
      if (
        log_file_handle == INVALID_HANDLE_VALUE ||
        !DuplicateHandle(
          GetCurrentProcess(),
          log_file_handle,
          GetCurrentProcess(),
          &inheritable_log_handle,
          0,
          TRUE,
          DUPLICATE_SAME_ACCESS
        )
      ) {
        fail_attribute_setup(
          log_file_handle == INVALID_HANDLE_VALUE ?
            ERROR_INVALID_HANDLE :
            GetLastError()
        );
        return startup_info;
      }

      // Populate std handles if the caller gave us a log file to use
      startup_info.StartupInfo.dwFlags |= STARTF_USESTDHANDLES;
      startup_info.StartupInfo.hStdInput = nullptr;
      startup_info.StartupInfo.hStdOutput = inheritable_log_handle;
      startup_info.StartupInfo.hStdError = inheritable_log_handle;

      // Allow the log file handle to be inherited by the child process (without inheriting all of
      // our inheritable handles, such as our own log file handle created by SunshineSvc).
      //
      // Note: The value we point to here must be valid for the lifetime of the attribute list,
      // so we need to point into the STARTUPINFO instead of our log_file_variable on the stack.
      if (
        !UpdateProcThreadAttribute(
          startup_info.lpAttributeList,
          0,
          PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
          &startup_info.StartupInfo.hStdOutput,
          sizeof(startup_info.StartupInfo.hStdOutput),
          nullptr,
          nullptr
        )
      ) {
        const auto windows_error = GetLastError();
        CloseHandle(inheritable_log_handle);
        startup_info.StartupInfo.hStdOutput = nullptr;
        startup_info.StartupInfo.hStdError = nullptr;
        fail_attribute_setup(windows_error);
        return startup_info;
      }
    }

    if (job) {
      // Atomically insert the new process into the specified job.
      //
      // Note: The value we point to here must be valid for the lifetime of the attribute list,
      // so we take a HANDLE* instead of just a HANDLE to use the caller's stack storage.
      if (!UpdateProcThreadAttribute(
            startup_info.lpAttributeList,
            0,
            PROC_THREAD_ATTRIBUTE_JOB_LIST,
            job,
            sizeof(*job),
            nullptr,
            nullptr
          )) {
        fail_attribute_setup(GetLastError());
        return startup_info;
      }
    }

    return startup_info;
  }

  /**
   * @brief This function overrides HKEY_CURRENT_USER and HKEY_CLASSES_ROOT using the provided token.
   * @param token The primary token identifying the user to use, or `nullptr` to restore original keys.
   * @return `true` if the override or restore operation was successful.
   */
  bool override_per_user_predefined_keys(HANDLE token) {
    HKEY user_classes_root = nullptr;
    if (token) {
      auto err = RegOpenUserClassesRoot(token, 0, GENERIC_ALL, &user_classes_root);
      if (err != ERROR_SUCCESS) {
        BOOST_LOG(error) << "Failed to open classes root for target user: "sv << err;
        return false;
      }
    }
    auto close_classes_root = util::fail_guard([user_classes_root]() {
      if (user_classes_root) {
        RegCloseKey(user_classes_root);
      }
    });

    HKEY user_key = nullptr;
    if (token) {
      impersonate_current_user(token, [&]() {
        // RegOpenCurrentUser() doesn't take a token. It assumes we're impersonating the desired user.
        auto err = RegOpenCurrentUser(GENERIC_ALL, &user_key);
        if (err != ERROR_SUCCESS) {
          BOOST_LOG(error) << "Failed to open user key for target user: "sv << err;
          user_key = nullptr;
        }
      });
      if (!user_key) {
        return false;
      }
    }
    auto close_user = util::fail_guard([user_key]() {
      if (user_key) {
        RegCloseKey(user_key);
      }
    });

    auto err = RegOverridePredefKey(HKEY_CLASSES_ROOT, user_classes_root);
    if (err != ERROR_SUCCESS) {
      BOOST_LOG(error) << "Failed to override HKEY_CLASSES_ROOT: "sv << err;
      return false;
    }

    err = RegOverridePredefKey(HKEY_CURRENT_USER, user_key);
    if (err != ERROR_SUCCESS) {
      BOOST_LOG(error) << "Failed to override HKEY_CURRENT_USER: "sv << err;
      RegOverridePredefKey(HKEY_CLASSES_ROOT, nullptr);
      return false;
    }

    return true;
  }

  /**
   * @brief Quote/escape an argument according to the Windows parsing convention.
   * @param argument The raw argument to process.
   * @return An argument string suitable for use by CreateProcess().
   */
  std::wstring escape_argument(const std::wstring &argument) {
    // If there are no characters requiring quoting/escaping, we're done
    if (argument.find_first_of(L" \t\n\v\"") == argument.npos) {
      return argument;
    }

    // The algorithm implemented here comes from a MSDN blog post:
    // https://web.archive.org/web/20120201194949/http://blogs.msdn.com/b/twistylittlepassagesallalike/archive/2011/04/23/everyone-quotes-arguments-the-wrong-way.aspx
    std::wstring escaped_arg;
    escaped_arg.push_back(L'"');
    for (auto it = argument.begin();; it++) {
      auto backslash_count = 0U;
      while (it != argument.end() && *it == L'\\') {
        it++;
        backslash_count++;
      }

      if (it == argument.end()) {
        escaped_arg.append(backslash_count * 2, L'\\');
        break;
      } else if (*it == L'"') {
        escaped_arg.append(backslash_count * 2 + 1, L'\\');
      } else {
        escaped_arg.append(backslash_count, L'\\');
      }

      escaped_arg.push_back(*it);
    }
    escaped_arg.push_back(L'"');
    return escaped_arg;
  }

  /**
   * @brief Escape an argument according to cmd's parsing convention.
   * @param argument An argument already escaped by `escape_argument()`.
   * @return An argument string suitable for use by cmd.exe.
   */
  std::wstring escape_argument_for_cmd(const std::wstring &argument) {
    // Start with the original string and modify from there
    std::wstring escaped_arg = argument;

    // Look for the next cmd metacharacter
    size_t match_pos = 0;
    while ((match_pos = escaped_arg.find_first_of(L"()%!^\"<>&|", match_pos)) != std::wstring::npos) {
      // Insert an escape character and skip past the match
      escaped_arg.insert(match_pos, 1, L'^');
      match_pos += 2;
    }

    return escaped_arg;
  }

  /**
   * @brief Resolve the given raw command into a proper command string for CreateProcess().
   * @details This converts URLs and non-executable file paths into a runnable command like ShellExecute().
   * @param raw_cmd The raw command provided by the user.
   * @param working_dir The working directory for the new process.
   * @param token The user token currently being impersonated or `nullptr` if running as ourselves.
   * @param creation_flags The creation flags for CreateProcess(), which may be modified by this function.
   * @return A command string suitable for use by CreateProcess().
   */
  std::wstring resolve_command_string(const std::string &raw_cmd, const std::wstring &working_dir, HANDLE token, DWORD &creation_flags) {
    std::wstring raw_cmd_w = from_utf8(raw_cmd);

    // First, convert the given command into parts so we can get the executable/file/URL without parameters
    auto raw_cmd_parts = boost::program_options::split_winmain(raw_cmd_w);
    if (raw_cmd_parts.empty()) {
      // This is highly unexpected, but we'll just return the raw string and hope for the best.
      BOOST_LOG(warning) << "Failed to split command string: "sv << raw_cmd;
      return from_utf8(raw_cmd);
    }

    auto raw_target = raw_cmd_parts.at(0);
    if (PathIsURLW(raw_target.c_str())) {
      // If the target is a URL, handle it directly with rundll32.exe
      std::wstring cmd = L"rundll32.exe url.dll,FileProtocolHandler " + raw_target;
      return cmd;
    }

    // If the target is not a URL, assume it's a regular file path
    auto extension = PathFindExtensionW(raw_target.c_str());
    if (extension == nullptr || *extension == 0) {
      // If the file has no extension, assume it's a command and allow CreateProcess()
      // to try to find it via PATH
      return from_utf8(raw_cmd);
    }
    else if (boost::iequals(extension, L".exe")) {
      // If the file has an .exe extension, we will bypass the resolution here and
      // directly pass the unmodified command string to CreateProcess(). The argument
      // escaping rules are subtly different between CreateProcess() and ShellExecute(),
      // and we want to preserve backwards compatibility with older configs.
      return from_utf8(raw_cmd);
    }

    // For regular files, the class is found using the file extension (including the dot)
    std::wstring lookup_string = extension;
    HRESULT res;

    std::array<WCHAR, MAX_PATH> shell_command_string;
    bool needs_cmd_escaping = false;
    {
      // Overriding these predefined keys affects process-wide state, so serialize all calls
      // to ensure the handle state is consistent while we perform the command query.
      static std::mutex per_user_key_mutex;
      auto lg = std::lock_guard(per_user_key_mutex);

      // Override HKEY_CLASSES_ROOT and HKEY_CURRENT_USER to ensure we query the correct class info
      if (!override_per_user_predefined_keys(token)) {
        return from_utf8(raw_cmd);
      }

      // Find the command string for the specified class
      DWORD out_len = shell_command_string.size();
      res = AssocQueryStringW(ASSOCF_NOTRUNCATE, ASSOCSTR_COMMAND, lookup_string.c_str(), L"open", shell_command_string.data(), &out_len);

      // In some cases (UWP apps), we might not have a command for this target. If that happens,
      // we'll have to launch via cmd.exe. This prevents proper job tracking, but that was already
      // broken for UWP apps anyway due to how they are started by Windows. Even 'start /wait'
      // doesn't work properly for UWP, so really no termination tracking seems to work at all.
      //
      // FIXME: Maybe we can improve this in the future.
      if (res == HRESULT_FROM_WIN32(ERROR_NO_ASSOCIATION)) {
        BOOST_LOG(warning) << "Using trampoline to handle target: "sv << raw_cmd;
        std::wcscpy(shell_command_string.data(), L"cmd.exe /c start \"\" /wait \"%1\" %*");
        needs_cmd_escaping = true;

        // We must suppress the console window that would otherwise appear when starting cmd.exe.
        creation_flags &= ~CREATE_NEW_CONSOLE;
        creation_flags |= CREATE_NO_WINDOW;

        res = S_OK;
      }

      // Reset per-user keys back to the original value
      override_per_user_predefined_keys(nullptr);
    }

    if (res != S_OK) {
      BOOST_LOG(warning) << "Failed to query command string for raw command: "sv << raw_cmd << " ["sv << util::hex(res).to_string_view() << ']';
      return from_utf8(raw_cmd);
    }

    // Finally, construct the real command string that will be passed into CreateProcess().
    // We support common substitutions (%*, %1, %2, %L, %W, %V, etc), but there are other
    // uncommon ones that are unsupported here.
    //
    // https://web.archive.org/web/20111002101214/http://msdn.microsoft.com/en-us/library/windows/desktop/cc144101(v=vs.85).aspx
    std::wstring cmd_string {shell_command_string.data()};
    size_t match_pos = 0;
    while ((match_pos = cmd_string.find_first_of(L'%', match_pos)) != std::wstring::npos) {
      std::wstring match_replacement;

      // If no additional character exists after the match, the dangling '%' is stripped
      if (match_pos + 1 == cmd_string.size()) {
        cmd_string.erase(match_pos, 1);
        break;
      }

      // Shell command replacements are strictly '%' followed by a single non-'%' character
      auto next_char = std::tolower(cmd_string.at(match_pos + 1));
      switch (next_char) {
        // Escape character
        case L'%':
          match_replacement = L'%';
          break;

        // Argument replacements
        case L'0':
        case L'1':
        case L'2':
        case L'3':
        case L'4':
        case L'5':
        case L'6':
        case L'7':
        case L'8':
        case L'9':
          {
            // Arguments numbers are 1-based, except for %0 which is equivalent to %1
            int index = next_char - L'0';
            if (next_char != L'0') {
              index--;
            }

            // Replace with the matching argument, or nothing if the index is invalid
            if (index < raw_cmd_parts.size()) {
              match_replacement = raw_cmd_parts.at(index);
            }
            break;
          }

        // All arguments following the target
        case L'*':
          for (int i = 1; i < raw_cmd_parts.size(); i++) {
            // Insert a space before arguments after the first one
            if (i > 1) {
              match_replacement += L' ';
            }

            // Argument escaping applies only to %*, not the single substitutions like %2
            auto escaped_argument = escape_argument(raw_cmd_parts.at(i));
            if (needs_cmd_escaping) {
              // If we're using the cmd.exe trampoline, we'll need to add additional escaping
              escaped_argument = escape_argument_for_cmd(escaped_argument);
            }
            match_replacement += escaped_argument;
          }
          break;

        // Long file path of target
        case L'l':
        case L'd':
        case L'v':
          {
            std::array<WCHAR, MAX_PATH> path;
            std::array<PCWCHAR, 2> other_dirs {working_dir.c_str(), nullptr};

            // PathFindOnPath() is a little gross because it uses the same
            // buffer for input and output, so we need to copy our input
            // into the path array.
            std::wcsncpy(path.data(), raw_target.c_str(), path.size());
            if (path[path.size() - 1] != 0) {
              // The path was so long it was truncated by this copy. We'll
              // assume it was an absolute path (likely) and use it unmodified.
              match_replacement = raw_target;
            }
            // See if we can find the path on our search path or working directory
            else if (PathFindOnPathW(path.data(), other_dirs.data())) {
              match_replacement = std::wstring {path.data()};
            } else {
              // We couldn't find the target, so we'll just hope for the best
              match_replacement = raw_target;
            }
            break;
          }

        // Working directory
        case L'w':
          match_replacement = working_dir;
          break;

        default:
          BOOST_LOG(warning) << "Unsupported argument replacement: %%" << next_char;
          break;
      }

      // Replace the % and following character with the match replacement
      cmd_string.replace(match_pos, 2, match_replacement);

      // Skip beyond the match replacement itself to prevent recursive replacement
      match_pos += match_replacement.size();
    }

    BOOST_LOG(info) << "Resolved user-provided command '"sv << raw_cmd << "' to '"sv << cmd_string << '\'';
    return cmd_string;
  }

  /**
   * @brief Run a command on the users profile.
   *
   * Launches a child process as the user, using the current user's environment and a specific working directory.
   *
   * @param elevated Specify whether to elevate the process.
   * @param interactive Specify whether this will run in a window or hidden.
   * @param cmd The command to run.
   * @param working_dir The working directory for the new process.
   * @param env The environment variables to use for the new process.
   * @param file A file object to redirect the child process's output to (may be `nullptr`).
   * @param ec An error code, set to indicate any errors that occur during the launch process.
   * @param group A pointer to a `bp::group` object to which the new process should belong (may be `nullptr`).
   * @return A `bp::child` object representing the new process, or an empty `bp::child` object if the launch fails.
   */
  namespace {
    bp::child run_command_impl(
      bool elevated,
      bool force_unelevated,
      bool interactive,
      const std::string &cmd,
      boost::filesystem::path &working_dir,
      const bp::environment &env,
      FILE *file,
      std::error_code &ec,
      bp::group *group,
      const std::optional<std::string> &expected_user_id
    ) {
    // resolve_command_string() historically relies on a temporary process-global PATH.
    // Serialize the complete mutation/restore interval so concurrent probes or app
    // launches cannot observe or restore another launch's intermediate value.
    std::lock_guard command_resolution_lock {command_resolution_mutex};

    // This parameter is an output. A stale error from a previous command must not suppress a
    // later launch while startup structures are being created.
    ec.clear();

    std::wstring start_dir = from_utf8(working_dir.string());
    HANDLE job = group ? group->native_handle() : nullptr;
    STARTUPINFOEXW startup_info = create_startup_info(file, job ? &job : nullptr, ec);
    PROCESS_INFORMATION process_info {};
    auto inherited_log_close = util::fail_guard([&]() {
      if (
        file &&
        startup_info.StartupInfo.hStdOutput &&
        startup_info.StartupInfo.hStdOutput != INVALID_HANDLE_VALUE
      ) {
        CloseHandle(startup_info.StartupInfo.hStdOutput);
      }
    });

    // Clone the environment to create a local copy. Boost.Process (bp) shares the environment with all spawned processes.
    // Since we're going to modify the 'env' variable by merging user-specific environment variables into it,
    // we make a clone to prevent side effects to the shared environment.
    bp::environment cloned_env = env;
    if (force_unelevated) {
      // Crossing from an elevated/service process into the standard-user media sandbox is a
      // security boundary. Do not leak parent-only variables (credentials, service paths, or
      // injected configuration); merge_user_environment_block() below reconstructs the child's
      // environment solely from the selected standard token.
      cloned_env.clear();
    }

    if (ec) {
      // In the event that startup_info failed, return a blank child process.
      return bp::child();
    }

    // Use RAII to ensure the attribute list is freed when we're done with it
    auto attr_list_free = util::fail_guard([list = startup_info.lpAttributeList]() {
      free_proc_thread_attr_list(list);
    });

    DWORD creation_flags =
      EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT;
    // Ordinary launched applications retain the historical attempt to escape a parent
    // service job. A caller-supplied group uses PROC_THREAD_ATTRIBUTE_JOB_LIST for atomic
    // assignment to that explicit job, so requesting breakaway at the same time is both
    // contradictory and rejected when the parent job disallows it.
    if (!group) {
      creation_flags |= CREATE_BREAKAWAY_FROM_JOB;
    }

    // Create a new console for interactive processes and use no console for non-interactive processes
    creation_flags |= interactive ? CREATE_NEW_CONSOLE : CREATE_NO_WINDOW;

    // Find the PATH variable in our environment block using a case-insensitive search
    auto sunshine_wenv = boost::this_process::wenvironment();
    std::wstring path_var_name {L"PATH"};
    std::wstring old_path_val;
    auto itr = std::find_if(sunshine_wenv.cbegin(), sunshine_wenv.cend(), [&](const auto &e) {
      return boost::iequals(e.get_name(), path_var_name);
    });
    if (itr != sunshine_wenv.cend()) {
      // Use the existing variable if it exists, since Boost treats these as case-sensitive.
      path_var_name = itr->get_name();
      old_path_val = sunshine_wenv[path_var_name].to_string();
    }

    // Temporarily prepend the specified working directory to PATH to ensure CreateProcess()
    // will (preferentially) find binaries that reside in the working directory.
    sunshine_wenv[path_var_name].assign(
      force_unelevated ?
        start_dir :
        start_dir + L";" + old_path_val
    );

    // Restore the old PATH value for our process when we're done here
    auto restore_path = util::fail_guard([&]() {
      if (old_path_val.empty()) {
        sunshine_wenv[path_var_name].clear();
      } else {
        sunshine_wenv[path_var_name].assign(old_path_val);
      }
    });

    const bool running_as_system = is_running_as_system();
    bool launch_with_user_token = running_as_system;
    if (force_unelevated && !running_as_system) {
      const auto active_identity =
        current_process_matches_active_standard_user();
      if (!active_identity || !*active_identity) {
        ec = std::make_error_code(std::errc::permission_denied);
        return bp::child();
      }
      const auto elevated = current_process_token_is_elevated();
      if (!elevated) {
        ec = std::make_error_code(std::errc::permission_denied);
        return bp::child();
      }
      launch_with_user_token = *elevated;
    }

    BOOL ret = FALSE;
    DWORD process_launch_error = ERROR_SUCCESS;
    if (launch_with_user_token) {
      // A service uses the active console user's token. An elevated tray process must
      // explicitly select its linked standard token instead of inheriting elevation.
      HANDLE user_token = running_as_system ?
                            retrieve_users_token(force_unelevated ? false : elevated) :
                            retrieve_current_users_standard_token();
      if (!user_token) {
        // Fail the launch rather than risking launching with Sunshine's permissions unmodified.
        ec = std::make_error_code(std::errc::permission_denied);
        return bp::child();
      }
      if (expected_user_id) {
        const auto observed_user_id = token_user_id(user_token);
        if (
          !observed_user_id ||
          *observed_user_id != *expected_user_id
        ) {
          ec = std::make_error_code(std::errc::permission_denied);
          CloseHandle(user_token);
          return bp::child();
        }
      }

      // Use RAII to ensure the shell token is closed when we're done with it
      auto token_close = util::fail_guard([user_token]() {
        CloseHandle(user_token);
      });

      // Populate env with user-specific environment variables
      if (!merge_user_environment_block(cloned_env, user_token)) {
        ec = std::make_error_code(std::errc::not_enough_memory);
        return bp::child();
      }

      std::wstring env_block = create_environment_block(cloned_env);
      std::wstring wcmd;
      if (running_as_system) {
        // CreateProcessAsUserW is the cross-session service path. Enable the privileges on the
        // SYSTEM process token explicitly, while retaining target impersonation so executable,
        // working-directory, association, and network-share access is evaluated as the user.
        scoped_process_privileges_t process_privileges {
          SE_INCREASE_QUOTA_NAME,
          SE_ASSIGNPRIMARYTOKEN_NAME,
        };
        if (!process_privileges) {
          const auto privilege_error = process_privileges.win32_error();
          BOOST_LOG(error) << "Could not enable the SYSTEM process privileges required for a "
                              "user launch: "sv
                           << privilege_error;
          ec = std::error_code {
            static_cast<int>(privilege_error),
            std::system_category(),
          };
        } else {
          ec = impersonate_current_user(user_token, [&]() {
            wcmd = resolve_command_string(cmd, start_dir, user_token, creation_flags);
            ret = CreateProcessAsUserW(
              user_token,
              nullptr,
              wcmd.data(),
              nullptr,
              nullptr,
              !!(startup_info.StartupInfo.dwFlags & STARTF_USESTDHANDLES),
              creation_flags,
              env_block.data(),
              start_dir.empty() ? nullptr : start_dir.c_str(),
              reinterpret_cast<LPSTARTUPINFOW>(&startup_info),
              &process_info
            );
            if (!ret) {
              // Capture before impersonation cleanup or logging can overwrite it.
              process_launch_error = GetLastError();
            }
          });
        }
      } else {
        // CreateProcessWithTokenW launches in the caller's session. Verify that this elevated tray
        // and its validated Explorer token are in the same active session before using it.
        HANDLE process_token = nullptr;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &process_token)) {
          const auto token_error = GetLastError();
          ec = std::error_code {
            static_cast<int>(token_error),
            std::system_category(),
          };
        } else {
          auto process_token_close = util::fail_guard([&]() {
            CloseHandle(process_token);
          });
          const auto process_session = get_token_session_id(process_token);
          const auto user_session = get_token_session_id(user_token);
          const auto active_session = WTSGetActiveConsoleSessionId();
          if (
            !process_session ||
            !user_session ||
            *process_session != *user_session ||
            active_session == 0xFFFFFFFF ||
            *process_session != active_session
          ) {
            BOOST_LOG(error) << "Refusing a standard-user launch because the elevated tray and "
                                "Explorer tokens are not in the same active session."sv;
            ec = std::make_error_code(std::errc::permission_denied);
          }
        }

        if (!ec) {
          scoped_process_privileges_t process_privileges {
            SE_IMPERSONATE_NAME,
          };
          if (!process_privileges) {
            const auto privilege_error = process_privileges.win32_error();
            BOOST_LOG(error) << "Could not enable SeImpersonatePrivilege for a standard-user "
                                "worker launch: "sv
                             << privilege_error;
            ec = std::error_code {
              static_cast<int>(privilege_error),
              std::system_category(),
            };
          } else {
            // Resolve all per-user command state under the target token. The actual API call must
            // occur after restoring the elevated effective token: CreateProcessWithTokenW checks
            // SeImpersonatePrivilege on the caller's effective token and would otherwise see the
            // standard user's unprivileged thread token.
            ec = impersonate_current_user(user_token, [&]() {
              wcmd = resolve_command_string(cmd, start_dir, user_token, creation_flags);
            });
            if (!ec) {
              // CreateProcessWithTokenW exposes no bInheritHandles parameter, so the extended
              // HANDLE_LIST contract cannot be satisfied; its documented STARTF_USESTDHANDLES
              // path copies the duplicated inheritable standard handles directly. Use plain
              // STARTUPINFO, create the process suspended, and assign it to the hardened job
              // before allowing its first instruction to run.
              const auto token_launch = detail::create_process_with_token_plan(
                creation_flags,
                job != nullptr
              );
              auto token_startup_info = startup_info.StartupInfo;
              token_startup_info.cb = token_launch.startup_info_size;
              ret = CreateProcessWithTokenW(
                user_token,
                0,
                nullptr,
                wcmd.data(),
                token_launch.creation_flags,
                env_block.data(),
                start_dir.empty() ? nullptr : start_dir.c_str(),
                &token_startup_info,
                &process_info
              );
              if (!ret) {
                // Capture before privilege restoration or logging can overwrite it.
                process_launch_error = GetLastError();
              } else if (token_launch.assign_job_after_create) {
                struct token_job_launch_operations_t {
                  HANDLE job;
                  PROCESS_INFORMATION &process;

                  bool assign_to_job() {
                    return AssignProcessToJobObject(job, process.hProcess) != FALSE;
                  }

                  bool resume_initial_thread() {
                    return ResumeThread(process.hThread) != static_cast<DWORD>(-1);
                  }

                  DWORD last_error() const {
                    return GetLastError();
                  }

                  void terminate_process(const DWORD exit_code) {
                    TerminateProcess(process.hProcess, exit_code);
                  }

                  void close_initial_thread() {
                    CloseHandle(process.hThread);
                    process.hThread = nullptr;
                  }

                  void close_process() {
                    CloseHandle(process.hProcess);
                    process.hProcess = nullptr;
                  }
                } operations {job, process_info};
                const DWORD post_create_error =
                  detail::finish_suspended_job_launch(operations);
                if (post_create_error != ERROR_SUCCESS) {
                  process_info = {};
                  process_launch_error = post_create_error;
                  ret = FALSE;
                }
              }
            }
          }
        }
      }
    }
    // Otherwise, launch the process using CreateProcessW()
    // This will inherit the elevation of whatever the user launched Sunshine with.
    else {
      // Open our current token to resolve environment variables
      HANDLE process_token;
      if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_DUPLICATE, &process_token)) {
        ec = std::make_error_code(std::errc::permission_denied);
        return bp::child();
      }
      auto token_close = util::fail_guard([process_token]() {
        CloseHandle(process_token);
      });
      if (expected_user_id) {
        const auto observed_user_id = token_user_id(process_token);
        if (
          !observed_user_id ||
          *observed_user_id != *expected_user_id
        ) {
          ec = std::make_error_code(std::errc::permission_denied);
          return bp::child();
        }
      }

      // Populate env with user-specific environment variables
      if (!merge_user_environment_block(cloned_env, process_token)) {
        ec = std::make_error_code(std::errc::not_enough_memory);
        return bp::child();
      }

      std::wstring env_block = create_environment_block(cloned_env);
      std::wstring wcmd = resolve_command_string(cmd, start_dir, nullptr, creation_flags);
      ret = CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, !!(startup_info.StartupInfo.dwFlags & STARTF_USESTDHANDLES), creation_flags, env_block.data(), start_dir.empty() ? nullptr : start_dir.c_str(), reinterpret_cast<LPSTARTUPINFOW>(&startup_info), &process_info);
      if (!ret) {
        // Capture before logging, handle cleanup, or another Win32 call can overwrite it.
        process_launch_error = GetLastError();
      }
    }

    // Use the results of the launch to create a bp::child object
    return create_boost_child_from_results(
      ret,
      process_launch_error,
      cmd,
      ec,
      process_info
    );
  }
  }  // namespace

  bp::child run_command(bool elevated, bool interactive, const std::string &cmd, boost::filesystem::path &working_dir, const bp::environment &env, FILE *file, std::error_code &ec, bp::group *group) {
    return run_command_impl(
      elevated,
      false,
      interactive,
      cmd,
      working_dir,
      env,
      file,
      ec,
      group,
      std::nullopt
    );
  }

  bp::child run_command_unelevated(
    bool interactive,
    const std::string &cmd,
    boost::filesystem::path &working_dir,
    const bp::environment &env,
    FILE *file,
    std::error_code &ec,
    bp::group *group,
    const std::optional<std::string> &expected_user_id
  ) {
    return run_command_impl(
      false,
      true,
      interactive,
      cmd,
      working_dir,
      env,
      file,
      ec,
      group,
      expected_user_id
    );
  }

  /**
   * @brief Open a url in the default web browser.
   * @param url The url to open.
   */
  void open_url(const std::string &url) {
    boost::process::v1::environment _env = boost::this_process::environment();
    auto working_dir = boost::filesystem::path();
    std::error_code ec;

    auto child = run_command(false, false, url, working_dir, _env, nullptr, ec, nullptr);
    if (ec) {
      BOOST_LOG(warning) << "Couldn't open url ["sv << url << "]: System: "sv << ec.message();
    } else {
      BOOST_LOG(info) << "Opened url ["sv << url << "]"sv;
      child.detach();
    }
  }

  void restart_on_exit() {
    STARTUPINFOEXW startup_info {};
    startup_info.StartupInfo.cb = sizeof(startup_info);

    WCHAR executable[MAX_PATH];
    if (GetModuleFileNameW(nullptr, executable, ARRAYSIZE(executable)) == 0) {
      auto winerr = GetLastError();
      BOOST_LOG(fatal) << "Failed to get Sunshine path: "sv << winerr;
      return;
    }

    PROCESS_INFORMATION process_info;
    if (!CreateProcessW(executable, GetCommandLineW(), nullptr, nullptr, false, CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, (LPSTARTUPINFOW) &startup_info, &process_info)) {
      auto winerr = GetLastError();
      BOOST_LOG(fatal) << "Unable to restart Sunshine: "sv << winerr;
      return;
    }

    CloseHandle(process_info.hProcess);
    CloseHandle(process_info.hThread);
  }

  void restart() {
    // If we're running standalone, we have to respawn ourselves via CreateProcess().
    // If we're running from the service, we should just exit and let it respawn us.
    if (GetConsoleWindow() != nullptr) {
      // Avoid racing with the new process by waiting until we're exiting to start it.
      atexit(restart_on_exit);
    }

    // We use an async exit call here because we can't block the HTTP thread or we'll hang shutdown.
    lifetime::exit_sunshine(0, true);
  }

  struct enum_wnd_context_t {
    std::set<DWORD> process_ids;
    bool requested_exit;
  };

  static BOOL CALLBACK prgrp_enum_windows(HWND hwnd, LPARAM lParam) {
    auto enum_ctx = (enum_wnd_context_t *) lParam;

    // Find the owner PID of this window
    DWORD wnd_process_id;
    if (!GetWindowThreadProcessId(hwnd, &wnd_process_id)) {
      // Continue enumeration
      return TRUE;
    }

    // Check if this window is owned by a process we want to terminate
    if (enum_ctx->process_ids.find(wnd_process_id) != enum_ctx->process_ids.end()) {
      // Send an async WM_CLOSE message to this window
      if (SendNotifyMessageW(hwnd, WM_CLOSE, 0, 0)) {
        BOOST_LOG(debug) << "Sent WM_CLOSE to PID: "sv << wnd_process_id;
        enum_ctx->requested_exit = true;
      } else {
        auto error = GetLastError();
        BOOST_LOG(warning) << "Failed to send WM_CLOSE to PID ["sv << wnd_process_id << "]: " << error;
      }
    }

    // Continue enumeration
    return TRUE;
  }

  bool request_process_group_exit(std::uintptr_t native_handle) {
    auto job_handle = (HANDLE) native_handle;

    // Get list of all processes in our job object
    bool success;
    DWORD required_length = sizeof(JOBOBJECT_BASIC_PROCESS_ID_LIST);
    auto process_id_list = (PJOBOBJECT_BASIC_PROCESS_ID_LIST) calloc(1, required_length);
    auto fg = util::fail_guard([&process_id_list]() {
      free(process_id_list);
    });
    while (!(success = QueryInformationJobObject(job_handle, JobObjectBasicProcessIdList, process_id_list, required_length, &required_length)) &&
           GetLastError() == ERROR_MORE_DATA) {
      free(process_id_list);
      process_id_list = (PJOBOBJECT_BASIC_PROCESS_ID_LIST) calloc(1, required_length);
      if (!process_id_list) {
        return false;
      }
    }

    if (!success) {
      auto err = GetLastError();
      BOOST_LOG(warning) << "Failed to enumerate processes in group: "sv << err;
      return false;
    } else if (process_id_list->NumberOfProcessIdsInList == 0) {
      // If all processes are already dead, treat it as a success
      return true;
    }

    enum_wnd_context_t enum_ctx = {};
    enum_ctx.requested_exit = false;
    for (DWORD i = 0; i < process_id_list->NumberOfProcessIdsInList; i++) {
      enum_ctx.process_ids.emplace(process_id_list->ProcessIdList[i]);
    }

    // Enumerate all windows belonging to processes in the list
    EnumWindows(prgrp_enum_windows, (LPARAM) &enum_ctx);

    // Return success if we told at least one window to close
    return enum_ctx.requested_exit;
  }

  bool process_group_running(std::uintptr_t native_handle) {
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting_info;

    if (!QueryInformationJobObject((HANDLE) native_handle, JobObjectBasicAccountingInformation, &accounting_info, sizeof(accounting_info), nullptr)) {
      auto err = GetLastError();
      BOOST_LOG(error) << "Failed to get job accounting info: "sv << err;
      return false;
    }

    return accounting_info.ActiveProcesses != 0;
  }
}  // namespace platf

#ifdef BOOST_PROCESS_VERSION
  #undef BOOST_PROCESS_VERSION
#endif
