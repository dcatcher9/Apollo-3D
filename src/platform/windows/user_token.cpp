/**
 * @file src/platform/windows/user_token.cpp
 * @brief Interactive-user tokens, identity checks, and impersonation on Windows.
 */
// standard includes
#include <cstddef>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

// prevent clang format from "optimizing" the header include order
// clang-format off
#include <Windows.h>
#include <WtsApi32.h>
#include <sddl.h>
#include <ShlObj.h>
// clang-format on

// local includes
#include "misc.h"
#include "src/logging.h"
#include "src/platform/common.h"
#include "src/utility.h"
#include "user_token.h"

namespace {

  std::mutex active_identity_cache_mutex;
  std::optional<DWORD> approved_active_identity_session;

}  // namespace

using namespace std::literals;

namespace platf {
  std::optional<bool> token_is_admin(HANDLE user_token) {
    SID_IDENTIFIER_AUTHORITY NtAuthority = SECURITY_NT_AUTHORITY;
    PSID AdministratorsGroup;
    if (!AllocateAndInitializeSid(
          &NtAuthority,
          2,
          SECURITY_BUILTIN_DOMAIN_RID,
          DOMAIN_ALIAS_RID_ADMINS,
          0,
          0,
          0,
          0,
          0,
          0,
          &AdministratorsGroup
        )) {
      BOOST_LOG(error) << "Unable to allocate SID to check administrative access: " << GetLastError();
      return std::nullopt;
    }
    auto sid_guard = util::fail_guard([&]() {
      FreeSid(AdministratorsGroup);
    });
    // CheckTokenMembership() accepts only an impersonation token, and the callers hold primary
    // tokens. An identification-level duplicate is enough to read the groups.
    HANDLE identification_token = nullptr;
    if (!DuplicateToken(user_token, SecurityIdentification, &identification_token)) {
      BOOST_LOG(error) << "Failed to duplicate the token to verify administrative access: " << GetLastError();
      return std::nullopt;
    }
    auto token_guard = util::fail_guard([&]() {
      CloseHandle(identification_token);
    });
    BOOL member = FALSE;
    if (!CheckTokenMembership(identification_token, AdministratorsGroup, &member)) {
      BOOST_LOG(error) << "Failed to verify token membership for administrative access: " << GetLastError();
      return std::nullopt;
    }
    return member != FALSE;
  }

  bool IsUserAdmin(HANDLE user_token) {
    return token_is_admin(user_token).value_or(false);
  }

  /**
   * @brief Obtain the current sessions user's primary token with elevated privileges.
   * @return The user's token. If user has admin capability it will be elevated, otherwise it will be a limited token. On error, `nullptr`.
   */
  HANDLE retrieve_users_token(bool elevated) {
    DWORD consoleSessionId;
    HANDLE userToken;
    TOKEN_ELEVATION_TYPE elevationType;
    DWORD dwSize;

    // Get the session ID of the active console session
    consoleSessionId = WTSGetActiveConsoleSessionId();
    if (0xFFFFFFFF == consoleSessionId) {
      // If there is no active console session, log a warning and return null
      BOOST_LOG(warning) << "There isn't an active user session, therefore it is not possible to execute commands under the users profile.";
      return nullptr;
    }

    // Get the user token for the active console session
    if (!WTSQueryUserToken(consoleSessionId, &userToken)) {
      BOOST_LOG(debug) << "QueryUserToken failed, this would prevent commands from launching under the users profile.";
      return nullptr;
    }

    // We need to know if this is an elevated token or not.
    // Get the elevation type of the user token
    // Elevation - Default: User is not an admin, UAC enabled/disabled does not matter.
    // Elevation - Limited: User is an admin, has UAC enabled.
    // Elevation - Full:    User is an admin, has UAC disabled.
    if (!GetTokenInformation(userToken, TokenElevationType, &elevationType, sizeof(TOKEN_ELEVATION_TYPE), &dwSize)) {
      BOOST_LOG(debug) << "Retrieving token information failed: " << GetLastError();
      CloseHandle(userToken);
      return nullptr;
    }

    // User is currently not an administrator
    // The documentation for this scenario is conflicting, so we'll double check to see if user is actually an admin.
    if (elevated && (elevationType == TokenElevationTypeDefault && !IsUserAdmin(userToken))) {
      // We don't have to strip the token or do anything here, but let's give the user a warning so they're aware what is happening.
      BOOST_LOG(warning) << "This command requires elevation and the current user account logged in does not have administrator rights. "
                         << "For security reasons Sunshine will retain the same access level as the current user and will not elevate it.";
    }

    // User has a limited token, this means they have UAC enabled and is an Administrator
    if (elevated && elevationType == TokenElevationTypeLimited) {
      TOKEN_LINKED_TOKEN linkedToken;
      // Retrieve the administrator token that is linked to the limited token
      if (!GetTokenInformation(userToken, TokenLinkedToken, reinterpret_cast<void *>(&linkedToken), sizeof(TOKEN_LINKED_TOKEN), &dwSize)) {
        // If the retrieval failed, log an error message and return null
        BOOST_LOG(error) << "Retrieving linked token information failed: " << GetLastError();
        CloseHandle(userToken);

        // There is no scenario where this should be hit, except for an actual error.
        return nullptr;
      }

      // Since we need the elevated token, we'll replace it with their administrative token.
      CloseHandle(userToken);
      userToken = linkedToken.LinkedToken;
    }

    // We don't need to do anything for TokenElevationTypeFull users here, because they're already elevated.
    return userToken;
  }

  std::optional<std::vector<std::byte>> token_user_information(
    HANDLE token
  ) {
    DWORD bytes = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &bytes);
    if (bytes == 0 || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
      return std::nullopt;
    }
    std::vector<std::byte> buffer(bytes);
    if (!GetTokenInformation(
          token,
          TokenUser,
          buffer.data(),
          static_cast<DWORD>(buffer.size()),
          &bytes
        )) {
      return std::nullopt;
    }
    return buffer;
  }

  std::optional<std::string> token_user_id(HANDLE token) {
    const auto buffer = token_user_information(token);
    if (!buffer) {
      return std::nullopt;
    }
    const auto token_user =
      reinterpret_cast<const TOKEN_USER *>(buffer->data());
    LPWSTR sid_text = nullptr;
    if (!ConvertSidToStringSidW(token_user->User.Sid, &sid_text) || !sid_text) {
      return std::nullopt;
    }
    auto sid_guard = util::fail_guard([&]() {
      LocalFree(sid_text);
    });
    return to_utf8(sid_text);
  }

  /**
   * Capture the standard primary token of the active desktop shell for an elevated tray process.
   *
   * TokenLinkedToken is not sufficient here. On this system the elevated process receives an
   * identification-constrained linked handle; using it for impersonation leaves profile access
   * failing with ERROR_BAD_IMPERSONATION_LEVEL. Explorer already owns the usable standard primary
   * token. The linked handle remains useful only as an identity witness for that shell token.
   */
  HANDLE duplicate_active_shell_standard_token(HANDLE process_token) {
    const DWORD active_session = WTSGetActiveConsoleSessionId();
    if (active_session == 0xFFFFFFFF) {
      BOOST_LOG(error) << "Could not capture the standard user token because no active console "
                          "session exists."sv;
      return nullptr;
    }

    DWORD process_session = 0xFFFFFFFF;
    DWORD returned = 0;
    if (
      !GetTokenInformation(
        process_token,
        TokenSessionId,
        &process_session,
        sizeof(process_session),
        &returned
      ) ||
      process_session != active_session
    ) {
      BOOST_LOG(error) << "The elevated tray process is not in the active console session; "
                          "refusing to borrow another session's shell token."sv;
      return nullptr;
    }

    const auto shell_window = GetShellWindow();
    DWORD shell_pid = 0;
    if (shell_window) {
      GetWindowThreadProcessId(shell_window, &shell_pid);
    }
    DWORD shell_session = 0xFFFFFFFF;
    if (
      shell_pid == 0 ||
      !ProcessIdToSessionId(shell_pid, &shell_session) ||
      shell_session != active_session
    ) {
      BOOST_LOG(error) << "Could not identify the active desktop shell for de-elevation."sv;
      return nullptr;
    }

    HANDLE shell_process = OpenProcess(
      PROCESS_QUERY_LIMITED_INFORMATION,
      FALSE,
      shell_pid
    );
    if (!shell_process) {
      BOOST_LOG(error) << "Could not open the active desktop shell for de-elevation: "sv
                       << GetLastError();
      return nullptr;
    }
    auto shell_process_guard = util::fail_guard([&]() {
      CloseHandle(shell_process);
    });

    HANDLE shell_token = nullptr;
    if (!OpenProcessToken(
          shell_process,
          TOKEN_QUERY | TOKEN_DUPLICATE,
          &shell_token
        )) {
      BOOST_LOG(error) << "Could not open the active desktop shell token for de-elevation: "sv
                       << GetLastError();
      return nullptr;
    }
    auto shell_token_guard = util::fail_guard([&]() {
      CloseHandle(shell_token);
    });

    TOKEN_ELEVATION shell_elevation {};
    TOKEN_TYPE shell_type = TokenImpersonation;
    TOKEN_STATISTICS shell_statistics {};
    DWORD shell_token_session = 0xFFFFFFFF;
    TOKEN_LINKED_TOKEN linked {};
    if (
      !GetTokenInformation(
        shell_token,
        TokenElevation,
        &shell_elevation,
        sizeof(shell_elevation),
        &returned
      ) ||
      !GetTokenInformation(
        shell_token,
        TokenType,
        &shell_type,
        sizeof(shell_type),
        &returned
      ) ||
      !GetTokenInformation(
        shell_token,
        TokenStatistics,
        &shell_statistics,
        sizeof(shell_statistics),
        &returned
      ) ||
      !GetTokenInformation(
        shell_token,
        TokenSessionId,
        &shell_token_session,
        sizeof(shell_token_session),
        &returned
      ) ||
      !GetTokenInformation(
        process_token,
        TokenLinkedToken,
        &linked,
        sizeof(linked),
        &returned
      )
    ) {
      BOOST_LOG(error) << "Could not validate the active desktop shell token: "sv
                       << GetLastError();
      return nullptr;
    }
    auto linked_token_guard = util::fail_guard([&]() {
      CloseHandle(linked.LinkedToken);
    });

    TOKEN_STATISTICS linked_statistics {};
    TOKEN_ELEVATION linked_elevation {};
    DWORD linked_session = 0xFFFFFFFF;
    if (
      !GetTokenInformation(
        linked.LinkedToken,
        TokenStatistics,
        &linked_statistics,
        sizeof(linked_statistics),
        &returned
      ) ||
      !GetTokenInformation(
        linked.LinkedToken,
        TokenElevation,
        &linked_elevation,
        sizeof(linked_elevation),
        &returned
      ) ||
      !GetTokenInformation(
        linked.LinkedToken,
        TokenSessionId,
        &linked_session,
        sizeof(linked_session),
        &returned
      )
    ) {
      BOOST_LOG(error) << "Could not validate the linked standard-user identity: "sv
                       << GetLastError();
      return nullptr;
    }

    const auto shell_user = token_user_information(shell_token);
    const auto process_user = token_user_information(process_token);
    const auto linked_user = token_user_information(linked.LinkedToken);
    if (!shell_user || !process_user || !linked_user) {
      BOOST_LOG(error) << "Could not identify the active desktop shell account."sv;
      return nullptr;
    }
    const auto shell_token_user =
      reinterpret_cast<const TOKEN_USER *>(shell_user->data());
    const auto process_token_user =
      reinterpret_cast<const TOKEN_USER *>(process_user->data());
    const auto linked_token_user =
      reinterpret_cast<const TOKEN_USER *>(linked_user->data());
    const bool same_user =
      EqualSid(
        shell_token_user->User.Sid,
        process_token_user->User.Sid
      ) != FALSE &&
      EqualSid(
        shell_token_user->User.Sid,
        linked_token_user->User.Sid
      ) != FALSE;
    const bool same_logon =
      shell_statistics.AuthenticationId.HighPart ==
        linked_statistics.AuthenticationId.HighPart &&
      shell_statistics.AuthenticationId.LowPart ==
        linked_statistics.AuthenticationId.LowPart;
    if (
      shell_elevation.TokenIsElevated != 0 ||
      shell_type != TokenPrimary ||
      shell_token_session != active_session ||
      linked_elevation.TokenIsElevated != 0 ||
      linked_session != active_session ||
      !same_user ||
      !same_logon
    ) {
      BOOST_LOG(error) << "The active desktop shell is not the matching standard-user logon; "
                          "refusing de-elevation."sv;
      return nullptr;
    }

    HANDLE duplicate = nullptr;
    if (!DuplicateTokenEx(
          shell_token,
          MAXIMUM_ALLOWED,
          nullptr,
          SecurityImpersonation,
          TokenPrimary,
          &duplicate
        )) {
      BOOST_LOG(error) << "Could not duplicate the active standard-user shell token: "sv
                       << GetLastError();
      return nullptr;
    }
    return duplicate;
  }

  /**
   * Reject over-the-shoulder elevation and service-account/session mismatches.
   *
   * A standard desktop token is safe to select only when the elevated process belongs to the same
   * account as the active shell. Otherwise "de-elevation" would target the administrator account
   * rather than the user operating the Web UI.
   */
  std::optional<bool> current_process_matches_active_standard_user() {
    const DWORD active_session = WTSGetActiveConsoleSessionId();
    if (active_session == 0xFFFFFFFF) {
      return std::nullopt;
    }
    {
      std::lock_guard cache_lock {active_identity_cache_mutex};
      if (
        approved_active_identity_session &&
        *approved_active_identity_session == active_session
      ) {
        return true;
      }
    }

    const auto shell_window = GetShellWindow();
    if (!shell_window) {
      return std::nullopt;
    }
    DWORD shell_pid = 0;
    GetWindowThreadProcessId(shell_window, &shell_pid);
    if (shell_pid == 0) {
      return std::nullopt;
    }
    DWORD shell_session = 0;
    if (
      !ProcessIdToSessionId(shell_pid, &shell_session) ||
      shell_session != active_session
    ) {
      return false;
    }

    HANDLE shell_process = OpenProcess(
      PROCESS_QUERY_LIMITED_INFORMATION,
      FALSE,
      shell_pid
    );
    if (!shell_process) {
      return std::nullopt;
    }
    auto shell_process_guard = util::fail_guard([&]() {
      CloseHandle(shell_process);
    });
    HANDLE shell_token = nullptr;
    if (!OpenProcessToken(shell_process, TOKEN_QUERY, &shell_token)) {
      return std::nullopt;
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
        )) {
      return std::nullopt;
    }
    if (shell_elevation.TokenIsElevated != 0) {
      // A UAC-disabled administrator has no lower-integrity interactive token.
      return false;
    }

    HANDLE process_token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &process_token)) {
      return std::nullopt;
    }
    auto process_token_guard = util::fail_guard([&]() {
      CloseHandle(process_token);
    });
    const auto shell_user = token_user_information(shell_token);
    const auto process_user = token_user_information(process_token);
    if (!shell_user || !process_user) {
      return std::nullopt;
    }
    const auto shell_token_user =
      reinterpret_cast<const TOKEN_USER *>(shell_user->data());
    const auto process_token_user =
      reinterpret_cast<const TOKEN_USER *>(process_user->data());
    const bool matches =
      EqualSid(
        shell_token_user->User.Sid,
        process_token_user->User.Sid
      ) != FALSE;
    if (matches) {
      std::lock_guard cache_lock {active_identity_cache_mutex};
      approved_active_identity_session = active_session;
    }
    return matches;
  }

  /**
   * Return a primary standard-user token for the current interactive account.
   *
   * Sunshine commonly runs elevated from the tray. CreateProcessW() would therefore silently pass
   * administrative rights to media parsers. For a split-token account, capture the matching active
   * desktop shell's standard primary token. A UAC-disabled administrator has no standard token, so
   * callers that require de-elevation must fail closed.
   */
  HANDLE retrieve_current_users_standard_token() {
    const auto active_identity =
      current_process_matches_active_standard_user();
    if (!active_identity || !*active_identity) {
      BOOST_LOG(error)
        << "The elevated process account does not match the active standard-user shell; "sv
           "refusing cross-account media access."sv;
      return nullptr;
    }
    HANDLE process_token = nullptr;
    if (!OpenProcessToken(
          GetCurrentProcess(),
          TOKEN_QUERY | TOKEN_DUPLICATE,
          &process_token
        )) {
      BOOST_LOG(error) << "Could not open the current process token for de-elevation: "sv
                       << GetLastError();
      return nullptr;
    }
    auto process_token_guard = util::fail_guard([&]() {
      CloseHandle(process_token);
    });

    TOKEN_ELEVATION_TYPE elevation_type {};
    DWORD returned = 0;
    if (!GetTokenInformation(
          process_token,
          TokenElevationType,
          &elevation_type,
          sizeof(elevation_type),
          &returned
        )) {
      BOOST_LOG(error) << "Could not inspect the current process elevation type: "sv
                       << GetLastError();
      return nullptr;
    }

    if (elevation_type == TokenElevationTypeFull) {
      return duplicate_active_shell_standard_token(process_token);
    }

    if (
      elevation_type == TokenElevationTypeDefault &&
      IsUserAdmin(process_token)
    ) {
      BOOST_LOG(error)
        << "Cannot launch an untrusted media worker safely because this administrator "sv
           "account has no UAC standard token."sv;
      return nullptr;
    }

    HANDLE duplicate = nullptr;
    if (!DuplicateTokenEx(
          process_token,
          MAXIMUM_ALLOWED,
          nullptr,
          SecurityImpersonation,
          TokenPrimary,
          &duplicate
        )) {
      BOOST_LOG(error) << "Could not duplicate the current standard-user token: "sv
                       << GetLastError();
      return nullptr;
    }
    return duplicate;
  }

  std::optional<bool> current_process_token_is_elevated() {
    HANDLE process_token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &process_token)) {
      return std::nullopt;
    }
    auto token_guard = util::fail_guard([&]() {
      CloseHandle(process_token);
    });
    TOKEN_ELEVATION elevation {};
    DWORD returned = 0;
    if (!GetTokenInformation(
          process_token,
          TokenElevation,
          &elevation,
          sizeof(elevation),
          &returned
        )) {
      return std::nullopt;
    }
    return elevation.TokenIsElevated != 0;
  }

  std::filesystem::path user_local_appdata() {
    HANDLE user_token = nullptr;
    if (is_running_as_system()) {
      user_token = retrieve_users_token(false);
      if (!user_token) {
        return {};
      }
    } else {
      const auto active_identity =
        current_process_matches_active_standard_user();
      if (!active_identity || !*active_identity) {
        return {};
      }
      const auto elevated = current_process_token_is_elevated();
      if (!elevated) {
        return {};
      }
      if (*elevated) {
        // Resolve the profile root with the same linked standard-user token
        // that run_as_active_user() and run_command_unelevated() use. Passing
        // nullptr here would make SHGetKnownFolderPath inspect the elevated
        // process token even though every later workspace access is performed
        // while impersonating the linked token.
        user_token = retrieve_current_users_standard_token();
        if (!user_token) {
          return {};
        }
      }
    }
    auto token_guard = util::fail_guard([&]() {
      if (user_token) {
        CloseHandle(user_token);
      }
    });

    PWSTR value = nullptr;
    const auto result = SHGetKnownFolderPath(
      FOLDERID_LocalAppData,
      KF_FLAG_DEFAULT,
      user_token,
      &value
    );
    if (FAILED(result) || !value) {
      return {};
    }
    auto value_guard = util::fail_guard([&]() {
      CoTaskMemFree(value);
    });
    return std::filesystem::path {value};
  }

  std::optional<std::string> active_user_id() {
    HANDLE user_token = nullptr;
    if (is_running_as_system()) {
      user_token = retrieve_users_token(false);
    } else {
      const auto active_identity =
        current_process_matches_active_standard_user();
      if (
        !active_identity ||
        !*active_identity ||
        !OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &user_token)
      ) {
        user_token = nullptr;
      }
    }
    if (!user_token) {
      BOOST_LOG(warning)
        << "Could not identify the active user for offline job isolation."sv;
      return std::nullopt;
    }
    auto token_guard = util::fail_guard([&]() {
      CloseHandle(user_token);
    });

    return token_user_id(user_token);
  }

  std::error_code run_as_active_user(
    const std::function<void()> &callback,
    const std::optional<std::string> &expected_user_id
  ) {
    // Offline media state is intentionally user-writable. Even a tray-launched host
    // normally runs elevated, so direct callbacks would turn every path race/reparse
    // into an administrative filesystem operation. Always perform these accesses with
    // the same standard token used by the de-elevated worker.
    const bool running_as_system = is_running_as_system();
    if (!running_as_system) {
      const auto active_identity =
        current_process_matches_active_standard_user();
      if (!active_identity || !*active_identity) {
        return std::make_error_code(std::errc::permission_denied);
      }
      const auto elevated = current_process_token_is_elevated();
      if (!elevated) {
        return std::make_error_code(std::errc::permission_denied);
      }
      if (!*elevated) {
        if (expected_user_id) {
          HANDLE process_token = nullptr;
          if (
            !OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &process_token)
          ) {
            return std::make_error_code(std::errc::permission_denied);
          }
          auto process_token_guard = util::fail_guard([&]() {
            CloseHandle(process_token);
          });
          const auto observed_user_id = token_user_id(process_token);
          if (
            !observed_user_id ||
            *observed_user_id != *expected_user_id
          ) {
            return std::make_error_code(std::errc::permission_denied);
          }
        }
        callback();
        return {};
      }
    }
    HANDLE user_token = running_as_system ?
                          retrieve_users_token(false) :
                          retrieve_current_users_standard_token();
    if (!user_token) {
      return std::make_error_code(std::errc::permission_denied);
    }
    auto token_guard = util::fail_guard([&]() {
      CloseHandle(user_token);
    });
    if (expected_user_id) {
      const auto observed_user_id = token_user_id(user_token);
      if (
        !observed_user_id ||
        *observed_user_id != *expected_user_id
      ) {
        return std::make_error_code(std::errc::permission_denied);
      }
    }
    return impersonate_current_user(user_token, callback);
  }

  /**
   * @brief Check if the current process is running with system-level privileges.
   * @return `true` if the current process has system-level privileges, `false` otherwise.
   */
  bool is_running_as_system() {
    BOOL ret;
    PSID SystemSid;
    DWORD dwSize = SECURITY_MAX_SID_SIZE;

    // Allocate memory for the SID structure
    SystemSid = LocalAlloc(LMEM_FIXED, dwSize);
    if (SystemSid == nullptr) {
      BOOST_LOG(error) << "Failed to allocate memory for the SID structure: " << GetLastError();
      return false;
    }

    // Create a SID for the local system account
    ret = CreateWellKnownSid(WinLocalSystemSid, nullptr, SystemSid, &dwSize);
    if (ret) {
      // Check if the current process token contains this SID
      if (!CheckTokenMembership(nullptr, SystemSid, &ret)) {
        BOOST_LOG(error) << "Failed to check token membership: " << GetLastError();
        ret = false;
      }
    } else {
      BOOST_LOG(error) << "Failed to create a SID for the local system account. This may happen if the system is out of memory or if the SID buffer is too small: " << GetLastError();
    }

    // Free the memory allocated for the SID structure
    LocalFree(SystemSid);
    return ret;
  }

  /**
   * @brief Impersonate the current user and invoke the callback function.
   * @param user_token A handle to the user's token that was obtained from the shell.
   * @param callback A function that will be executed while impersonating the user.
   * @return Object that will store any error that occurred during the impersonation
   */
  std::error_code impersonate_current_user(HANDLE user_token, std::function<void()> callback) {
    std::error_code ec;
    HANDLE previous_thread_token = nullptr;
    const bool had_previous_thread_token = OpenThreadToken(
      GetCurrentThread(),
      TOKEN_IMPERSONATE | TOKEN_QUERY,
      TRUE,
      &previous_thread_token
    ) != FALSE;
    const auto previous_token_error = had_previous_thread_token ?
                                        ERROR_SUCCESS :
                                        GetLastError();
    if (!had_previous_thread_token && previous_token_error != ERROR_NO_TOKEN) {
      const auto winerror = previous_token_error;
      BOOST_LOG(error) << "Failed to preserve the current thread impersonation token: "sv
                       << winerror;
      return std::error_code {
        static_cast<int>(winerror),
        std::system_category(),
      };
    }
    auto previous_thread_token_guard = util::fail_guard([&]() {
      if (previous_thread_token) {
        CloseHandle(previous_thread_token);
      }
    });

    // Impersonate the user when launching the process. This will ensure that appropriate access
    // checks are done against the user token, not our SYSTEM token. It will also allow network
    // shares and mapped network drives to be used as launch targets, since those credentials
    // are stored per-user.
    if (!ImpersonateLoggedOnUser(user_token)) {
      auto winerror = GetLastError();
      // Log the failure of impersonating the user and its error code
      BOOST_LOG(error) << "Failed to impersonate user: "sv << winerror;
      ec = std::make_error_code(std::errc::permission_denied);
      return ec;
    }

    // Restore the exact prior effective token, including when the callback throws. RevertToSelf()
    // is not stack-safe: an inner impersonation would otherwise discard its caller's thread token.
    auto revert_guard = util::fail_guard([&]() {
      const bool restored = had_previous_thread_token ?
                              SetThreadToken(nullptr, previous_thread_token) != FALSE :
                              RevertToSelf() != FALSE;
      if (!restored) {
        const auto winerror = GetLastError();
        BOOST_LOG(fatal) << "Failed to restore the prior thread token after impersonation: "sv
                         << winerror;
        DebugBreak();
      }
    });

    HANDLE thread_token = nullptr;
    if (!OpenThreadToken(
          GetCurrentThread(),
          TOKEN_QUERY,
          TRUE,
          &thread_token
        )) {
      BOOST_LOG(error) << "Could not validate the installed user impersonation token: "sv
                       << GetLastError();
      return std::make_error_code(std::errc::permission_denied);
    }
    auto thread_token_guard = util::fail_guard([&]() {
      CloseHandle(thread_token);
    });
    SECURITY_IMPERSONATION_LEVEL impersonation_level = SecurityAnonymous;
    DWORD returned = 0;
    if (
      !GetTokenInformation(
        thread_token,
        TokenImpersonationLevel,
        &impersonation_level,
        sizeof(impersonation_level),
        &returned
      ) ||
      impersonation_level < SecurityImpersonation
    ) {
      BOOST_LOG(error) << "Refusing user callback because Windows installed an insufficient "
                          "impersonation level."sv;
      return std::make_error_code(std::errc::permission_denied);
    }

    callback();
    return ec;
  }
}  // namespace platf
