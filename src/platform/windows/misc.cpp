/**
 * @file src/platform/windows/misc.cpp
 * @brief Miscellaneous definitions for Windows.
 */
// standard includes
#include <array>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <mutex>
#include <set>
#include <sstream>

#ifndef BOOST_PROCESS_VERSION
 #define BOOST_PROCESS_VERSION 1
#endif

// lib includes
#include <boost/algorithm/string.hpp>
#include <boost/asio/ip/address.hpp>
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/group.hpp>
#include <boost/process/v1/environment.hpp>
#include <boost/program_options/parsers.hpp>

// prevent clang format from "optimizing" the header include order
// clang-format off
#include <dwmapi.h>
#include <iphlpapi.h>
#include <iterator>
#include <timeapi.h>
#include <UserEnv.h>
#include <WinSock2.h>
#include <Windows.h>
#include <WinUser.h>
#include <wlanapi.h>
#include <WS2tcpip.h>
#include <WtsApi32.h>
#include <sddl.h>
#include <ShlObj.h>
// clang-format on

// Boost overrides NTDDI_VERSION, so we re-override it here
#undef NTDDI_VERSION
#define NTDDI_VERSION NTDDI_WIN10
#include <Shlwapi.h>

// local includes
#include "misc.h"
#include "utils.h"
#include "nvprefs/nvprefs_interface.h"
#include "src/entry_handler.h"
#include "src/globals.h"
#include "src/logging.h"
#include "src/platform/common.h"
#include "src/utility.h"

// UDP_SEND_MSG_SIZE was added in the Windows 10 20H1 SDK
#ifndef UDP_SEND_MSG_SIZE
  #define UDP_SEND_MSG_SIZE 2
#endif

// PROC_THREAD_ATTRIBUTE_JOB_LIST is currently missing from MinGW headers
#ifndef PROC_THREAD_ATTRIBUTE_JOB_LIST
  #define PROC_THREAD_ATTRIBUTE_JOB_LIST ProcThreadAttributeValue(13, FALSE, TRUE, FALSE)
#endif

#include <qos2.h>

#ifndef WLAN_API_MAKE_VERSION
  #define WLAN_API_MAKE_VERSION(_major, _minor) (((DWORD) (_minor)) << 16 | (_major))
#endif

#include <winternl.h>
extern "C" {
  NTSTATUS NTAPI NtSetTimerResolution(ULONG DesiredResolution, BOOLEAN SetResolution, PULONG CurrentResolution);
}

namespace {

  std::atomic<bool> used_nt_set_timer_resolution = false;
  std::mutex command_resolution_mutex;
  std::mutex active_identity_cache_mutex;
  std::mutex explorer_restart_mutex;
  std::optional<DWORD> approved_active_identity_session;
  std::mutex mouse_keys_mutex;
  platf::detail::mouse_keys_controller_t mouse_keys_controller;
  std::chrono::steady_clock::time_point mouse_keys_retry_after;

  bool nt_set_timer_resolution_max() {
    ULONG minimum, maximum, current;
    if (!NT_SUCCESS(NtQueryTimerResolution(&minimum, &maximum, &current)) ||
        !NT_SUCCESS(NtSetTimerResolution(maximum, TRUE, &current))) {
      return false;
    }
    return true;
  }

  bool nt_set_timer_resolution_min() {
    ULONG minimum, maximum, current;
    if (!NT_SUCCESS(NtQueryTimerResolution(&minimum, &maximum, &current)) ||
        !NT_SUCCESS(NtSetTimerResolution(minimum, TRUE, &current))) {
      return false;
    }
    return true;
  }

}  // namespace

namespace bp = boost::process;

static std::string ensureCrLf(const std::string& utf8Str);
static std::wstring getClipboardData();
static int setClipboardData(const std::wstring& utf16Str);

using namespace std::literals;

namespace platf {
  using adapteraddrs_t = util::c_ptr<IP_ADAPTER_ADDRESSES>;

  bool is_running_as_system();

  std::error_code impersonate_current_user(
    HANDLE user_token,
    std::function<void()> callback
  );

  HANDLE qos_handle = nullptr;

  decltype(QOSCreateHandle) *fn_QOSCreateHandle = nullptr;
  decltype(QOSAddSocketToFlow) *fn_QOSAddSocketToFlow = nullptr;
  decltype(QOSRemoveSocketFromFlow) *fn_QOSRemoveSocketFromFlow = nullptr;

  HANDLE wlan_handle = nullptr;

  decltype(WlanOpenHandle) *fn_WlanOpenHandle = nullptr;
  decltype(WlanCloseHandle) *fn_WlanCloseHandle = nullptr;
  decltype(WlanFreeMemory) *fn_WlanFreeMemory = nullptr;
  decltype(WlanEnumInterfaces) *fn_WlanEnumInterfaces = nullptr;
  decltype(WlanSetInterface) *fn_WlanSetInterface = nullptr;

  std::filesystem::path appdata() {
    WCHAR sunshine_path[MAX_PATH];
    GetModuleFileNameW(nullptr, sunshine_path, _countof(sunshine_path));
    return std::filesystem::path {sunshine_path}.remove_filename() / L"config"sv;
  }

  std::string from_sockaddr(const sockaddr *const socket_address) {
    char data[INET6_ADDRSTRLEN] = {};

    auto family = socket_address->sa_family;
    if (family == AF_INET6) {
      inet_ntop(AF_INET6, &((sockaddr_in6 *) socket_address)->sin6_addr, data, INET6_ADDRSTRLEN);
    } else if (family == AF_INET) {
      inet_ntop(AF_INET, &((sockaddr_in *) socket_address)->sin_addr, data, INET_ADDRSTRLEN);
    }

    return std::string {data};
  }

  std::pair<std::uint16_t, std::string> from_sockaddr_ex(const sockaddr *const ip_addr) {
    char data[INET6_ADDRSTRLEN] = {};

    auto family = ip_addr->sa_family;
    std::uint16_t port = 0;
    if (family == AF_INET6) {
      inet_ntop(AF_INET6, &((sockaddr_in6 *) ip_addr)->sin6_addr, data, INET6_ADDRSTRLEN);
      port = ((sockaddr_in6 *) ip_addr)->sin6_port;
    } else if (family == AF_INET) {
      inet_ntop(AF_INET, &((sockaddr_in *) ip_addr)->sin_addr, data, INET_ADDRSTRLEN);
      port = ((sockaddr_in *) ip_addr)->sin_port;
    }

    return {port, std::string {data}};
  }

  adapteraddrs_t get_adapteraddrs() {
    adapteraddrs_t info {nullptr};
    ULONG size = 0;

    while (GetAdaptersAddresses(AF_UNSPEC, 0, nullptr, info.get(), &size) == ERROR_BUFFER_OVERFLOW) {
      info.reset((PIP_ADAPTER_ADDRESSES) malloc(size));
    }

    return info;
  }

  std::string get_mac_address(const std::string_view &address) {
    adapteraddrs_t info = get_adapteraddrs();
    for (auto adapter_pos = info.get(); adapter_pos != nullptr; adapter_pos = adapter_pos->Next) {
      for (auto addr_pos = adapter_pos->FirstUnicastAddress; addr_pos != nullptr; addr_pos = addr_pos->Next) {
        if (adapter_pos->PhysicalAddressLength != 0 && address == from_sockaddr(addr_pos->Address.lpSockaddr)) {
          std::stringstream mac_addr;
          mac_addr << std::hex;
          for (int i = 0; i < adapter_pos->PhysicalAddressLength; i++) {
            if (i > 0) {
              mac_addr << ':';
            }
            mac_addr << std::setw(2) << std::setfill('0') << (int) adapter_pos->PhysicalAddress[i];
          }
          return mac_addr.str();
        }
      }
    }
    BOOST_LOG(debug) << "Unable to find MAC address for "sv << address << ", is this a virtual network adapter?";
    return "00:00:00:00:00:00"s;
  }

  std::string get_local_ip_for_gateway() {
    PIP_ADAPTER_INFO pAdapterInfo;
    PIP_ADAPTER_INFO pAdapter = nullptr;
    DWORD dwRetVal = 0;
    ULONG ulOutBufLen = sizeof(IP_ADAPTER_INFO);

    pAdapterInfo = (IP_ADAPTER_INFO *)malloc(sizeof(IP_ADAPTER_INFO));
    if (pAdapterInfo == nullptr) {
      BOOST_LOG(warning) << "Error allocating memory needed to call GetAdaptersInfo";
      return "";
    }

    // Make an initial call to GetAdaptersInfo to get the necessary size into the ulOutBufLen variable
    if (GetAdaptersInfo(pAdapterInfo, &ulOutBufLen) == ERROR_BUFFER_OVERFLOW) {
      free(pAdapterInfo);
      pAdapterInfo = (IP_ADAPTER_INFO *)malloc(ulOutBufLen);
      if (pAdapterInfo == nullptr) {
        BOOST_LOG(warning) << "Error allocating memory needed to call GetAdaptersInfo";
        return "";
      }
    }

    if ((dwRetVal = GetAdaptersInfo(pAdapterInfo, &ulOutBufLen)) != NO_ERROR) {
      if (pAdapterInfo) {
        free(pAdapterInfo);
      }
      BOOST_LOG(warning) << "GetAdaptersInfo failed with error: " + std::to_string(dwRetVal);
      return "";
    }

    pAdapter = pAdapterInfo;
    std::string local_ip;

    // Iterate through the list of adapters
    while (pAdapter) {
      IP_ADDR_STRING* pGateway = &pAdapter->GatewayList;
      if (pGateway && pGateway->IpAddress.String[0] != '\0') {
        // This adapter has a default gateway, use its IP address
        local_ip = pAdapter->IpAddressList.IpAddress.String;
        break;
      }
      pAdapter = pAdapter->Next;
    }

    if (pAdapterInfo) {
      free(pAdapterInfo);
    }

    if (local_ip.empty()) {
      BOOST_LOG(warning) << "No associated IP address found for the default gateway";
    }

    return local_ip;
  }

  HDESK syncThreadDesktop() {
    auto hDesk = OpenInputDesktop(DF_ALLOWOTHERACCOUNTHOOK, FALSE, GENERIC_ALL);
    if (!hDesk) {
      auto err = GetLastError();
      BOOST_LOG(error) << "Failed to Open Input Desktop [0x"sv << util::hex(err).to_string_view() << ']';

      return nullptr;
    }

    if (!SetThreadDesktop(hDesk)) {
      auto err = GetLastError();
      BOOST_LOG(error) << "Failed to sync desktop to thread [0x"sv << util::hex(err).to_string_view() << ']';
    }

    CloseDesktop(hDesk);

    return hDesk;
  }

  void print_status(const std::string_view &prefix, HRESULT status) {
    char err_string[1024];

    DWORD bytes = FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, status, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), err_string, sizeof(err_string), nullptr);

    BOOST_LOG(error) << prefix << ": "sv << std::string_view {err_string, bytes};
  }

  bool IsUserAdmin(HANDLE user_token) {
    WINBOOL ret;
    SID_IDENTIFIER_AUTHORITY NtAuthority = SECURITY_NT_AUTHORITY;
    PSID AdministratorsGroup;
    ret = AllocateAndInitializeSid(
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
    );
    if (ret) {
      if (!CheckTokenMembership(user_token, AdministratorsGroup, &ret)) {
        ret = false;
        BOOST_LOG(error) << "Failed to verify token membership for administrative access: " << GetLastError();
      }
      FreeSid(AdministratorsGroup);
    } else {
      BOOST_LOG(error) << "Unable to allocate SID to check administrative access: " << GetLastError();
    }

    return ret;
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

  // Note: This does NOT append a null terminator
  void append_string_to_environment_block(wchar_t *env_block, int &offset, const std::wstring &wstr) {
    std::memcpy(&env_block[offset], wstr.data(), wstr.length() * sizeof(wchar_t));
    offset += wstr.length();
  }

  std::wstring create_environment_block(bp::environment &env) {
    int size = 0;
    for (const auto &entry : env) {
      auto name = entry.get_name();
      auto value = entry.to_string();
      size += from_utf8(name).length() + 1 /* L'=' */ + from_utf8(value).length() + 1 /* L'\0' */;
    }

    size += 1 /* L'\0' */;

    wchar_t env_block[size];
    int offset = 0;
    for (const auto &entry : env) {
      auto name = entry.get_name();
      auto value = entry.to_string();

      // Construct the NAME=VAL\0 string
      append_string_to_environment_block(env_block, offset, from_utf8(name));
      env_block[offset++] = L'=';
      append_string_to_environment_block(env_block, offset, from_utf8(value));
      env_block[offset++] = L'\0';
    }

    // Append a final null terminator
    env_block[offset++] = L'\0';

    return std::wstring(env_block, offset);
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

  void adjust_thread_priority(thread_priority_e priority) {
    int win32_priority;

    switch (priority) {
      case thread_priority_e::low:
        win32_priority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
      case thread_priority_e::normal:
        win32_priority = THREAD_PRIORITY_NORMAL;
        break;
      case thread_priority_e::high:
        win32_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
      case thread_priority_e::critical:
        win32_priority = THREAD_PRIORITY_HIGHEST;
        break;
      default:
        BOOST_LOG(error) << "Unknown thread priority: "sv << (int) priority;
        return;
    }

    if (!SetThreadPriority(GetCurrentThread(), win32_priority)) {
      auto winerr = GetLastError();
      BOOST_LOG(warning) << "Unable to set thread priority to "sv << win32_priority << ": "sv << winerr;
    }
  }

  void streaming_will_start() {
    static std::once_flag load_wlanapi_once_flag;
    std::call_once(load_wlanapi_once_flag, []() {
      // wlanapi.dll is not installed by default on Windows Server, so we load it dynamically
      HMODULE wlanapi = LoadLibraryExA("wlanapi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
      if (!wlanapi) {
        BOOST_LOG(debug) << "wlanapi.dll is not available on this OS"sv;
        return;
      }

      fn_WlanOpenHandle = (decltype(fn_WlanOpenHandle)) GetProcAddress(wlanapi, "WlanOpenHandle");
      fn_WlanCloseHandle = (decltype(fn_WlanCloseHandle)) GetProcAddress(wlanapi, "WlanCloseHandle");
      fn_WlanFreeMemory = (decltype(fn_WlanFreeMemory)) GetProcAddress(wlanapi, "WlanFreeMemory");
      fn_WlanEnumInterfaces = (decltype(fn_WlanEnumInterfaces)) GetProcAddress(wlanapi, "WlanEnumInterfaces");
      fn_WlanSetInterface = (decltype(fn_WlanSetInterface)) GetProcAddress(wlanapi, "WlanSetInterface");

      if (!fn_WlanOpenHandle || !fn_WlanCloseHandle || !fn_WlanFreeMemory || !fn_WlanEnumInterfaces || !fn_WlanSetInterface) {
        BOOST_LOG(error) << "wlanapi.dll is missing exports?"sv;

        fn_WlanOpenHandle = nullptr;
        fn_WlanCloseHandle = nullptr;
        fn_WlanFreeMemory = nullptr;
        fn_WlanEnumInterfaces = nullptr;
        fn_WlanSetInterface = nullptr;

        FreeLibrary(wlanapi);
        return;
      }
    });

    // Enable MMCSS scheduling for DWM
    DwmEnableMMCSS(true);

    // Reduce timer period to 0.5ms
    if (nt_set_timer_resolution_max()) {
      used_nt_set_timer_resolution = true;
    } else {
      BOOST_LOG(error) << "NtSetTimerResolution() failed, falling back to timeBeginPeriod()";
      timeBeginPeriod(1);
      used_nt_set_timer_resolution = false;
    }

    // Promote ourselves to high priority class
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Modify NVIDIA control panel settings again, in case they have been changed externally since sunshine launch
    if (nvprefs_instance.load()) {
      if (!nvprefs_instance.owning_undo_file()) {
        nvprefs_instance.restore_from_and_delete_undo_file_if_exists();
      }
      nvprefs_instance.modify_application_profile();
      nvprefs_instance.modify_global_profile();
      nvprefs_instance.unload();
    }

    // Enable low latency mode on all connected WLAN NICs if wlanapi.dll is available
    if (fn_WlanOpenHandle) {
      DWORD negotiated_version;

      if (fn_WlanOpenHandle(WLAN_API_MAKE_VERSION(2, 0), nullptr, &negotiated_version, &wlan_handle) == ERROR_SUCCESS) {
        PWLAN_INTERFACE_INFO_LIST wlan_interface_list;

        if (fn_WlanEnumInterfaces(wlan_handle, nullptr, &wlan_interface_list) == ERROR_SUCCESS) {
          for (DWORD i = 0; i < wlan_interface_list->dwNumberOfItems; i++) {
            if (wlan_interface_list->InterfaceInfo[i].isState == wlan_interface_state_connected) {
              // Enable media streaming mode for 802.11 wireless interfaces to reduce latency and
              // unnecessary background scanning operations that cause packet loss and jitter.
              //
              // https://docs.microsoft.com/en-us/windows-hardware/drivers/network/oid-wdi-set-connection-quality
              // https://docs.microsoft.com/en-us/previous-versions/windows/hardware/wireless/native-802-11-media-streaming
              BOOL value = TRUE;
              auto error = fn_WlanSetInterface(wlan_handle, &wlan_interface_list->InterfaceInfo[i].InterfaceGuid, wlan_intf_opcode_media_streaming_mode, sizeof(value), &value, nullptr);
              if (error == ERROR_SUCCESS) {
                BOOST_LOG(info) << "WLAN interface "sv << i << " is now in low latency mode"sv;
              }
            }
          }

          fn_WlanFreeMemory(wlan_interface_list);
        } else {
          fn_WlanCloseHandle(wlan_handle, nullptr);
          wlan_handle = nullptr;
        }
      }
    }

    {
      std::lock_guard lock(mouse_keys_mutex);
      mouse_keys_retry_after = {};
    }
    refresh_mouse_keys();
  }

  void refresh_mouse_keys() {
    std::lock_guard lock(mouse_keys_mutex);
    const auto now = std::chrono::steady_clock::now();
    if (now < mouse_keys_retry_after) {
      return;
    }

    bool operation_failed = false;

    const auto enabled = mouse_keys_controller.refresh(
      GetSystemMetrics(SM_MOUSEPRESENT) != 0,
      [&](MOUSEKEYS &state) {
        if (SystemParametersInfoW(SPI_GETMOUSEKEYS, sizeof(MOUSEKEYS), &state, 0)) {
          return true;
        }

        operation_failed = true;
        BOOST_LOG(warning) << "Unable to get current state of Mouse Keys: "sv << GetLastError();
        return false;
      },
      [&](MOUSEKEYS &state) {
        if (SystemParametersInfoW(SPI_SETMOUSEKEYS, sizeof(MOUSEKEYS), &state, 0)) {
          return true;
        }

        operation_failed = true;
        BOOST_LOG(warning) << "Unable to enable Mouse Keys: "sv << GetLastError();
        return false;
      }
    );

    mouse_keys_retry_after = operation_failed ? now + 30s : std::chrono::steady_clock::time_point {};

    if (enabled) {
      BOOST_LOG(info) << "A mouse was not detected. Sunshine 3D enabled Mouse Keys while streaming to keep the remote cursor visible."sv;
    }
  }

  void streaming_will_stop() {
    // Demote ourselves back to normal priority class
    SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);

    // End our 0.5ms timer request
    if (used_nt_set_timer_resolution) {
      used_nt_set_timer_resolution = false;
      if (!nt_set_timer_resolution_min()) {
        BOOST_LOG(error) << "nt_set_timer_resolution_min() failed even though nt_set_timer_resolution_max() succeeded";
      }
    } else {
      timeEndPeriod(1);
    }

    // Disable MMCSS scheduling for DWM
    DwmEnableMMCSS(false);

    // Closing our WLAN client handle will undo our optimizations
    if (wlan_handle != nullptr) {
      fn_WlanCloseHandle(wlan_handle, nullptr);
      wlan_handle = nullptr;
    }

    // Restore Mouse Keys back to the previous settings if we turned it on. The controller keeps
    // ownership after a failure so a later stream stop can retry instead of losing the snapshot.
    std::lock_guard lock(mouse_keys_mutex);
    mouse_keys_controller.restore([](MOUSEKEYS &state) {
      if (SystemParametersInfoW(SPI_SETMOUSEKEYS, sizeof(MOUSEKEYS), &state, 0)) {
        return true;
      }

      BOOST_LOG(warning) << "Unable to restore original state of Mouse Keys: "sv << GetLastError();
      return false;
    });
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

  int set_env(const std::string &name, const std::string &value) {
    return _putenv_s(name.c_str(), value.c_str());
  }

  int unset_env(const std::string &name) {
    return _putenv_s(name.c_str(), "");
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

  SOCKADDR_IN to_sockaddr(boost::asio::ip::address_v4 address, uint16_t port) {
    SOCKADDR_IN saddr_v4 = {};

    saddr_v4.sin_family = AF_INET;
    saddr_v4.sin_port = htons(port);

    auto addr_bytes = address.to_bytes();
    memcpy(&saddr_v4.sin_addr, addr_bytes.data(), sizeof(saddr_v4.sin_addr));

    return saddr_v4;
  }

  SOCKADDR_IN6 to_sockaddr(boost::asio::ip::address_v6 address, uint16_t port) {
    SOCKADDR_IN6 saddr_v6 = {};

    saddr_v6.sin6_family = AF_INET6;
    saddr_v6.sin6_port = htons(port);
    saddr_v6.sin6_scope_id = address.scope_id();

    auto addr_bytes = address.to_bytes();
    memcpy(&saddr_v6.sin6_addr, addr_bytes.data(), sizeof(saddr_v6.sin6_addr));

    return saddr_v6;
  }

  // Use UDP segmentation offload if it is supported by the OS. If the NIC is capable, this will use
  // hardware acceleration to reduce CPU usage. Support for USO was introduced in Windows 10 20H1.
  bool send_batch(batched_send_info_t &send_info) {
    WSAMSG msg;

    // Convert the target address into a SOCKADDR
    SOCKADDR_IN taddr_v4;
    SOCKADDR_IN6 taddr_v6;
    if (send_info.target_address.is_v6()) {
      taddr_v6 = to_sockaddr(send_info.target_address.to_v6(), send_info.target_port);

      msg.name = (PSOCKADDR) &taddr_v6;
      msg.namelen = sizeof(taddr_v6);
    } else {
      taddr_v4 = to_sockaddr(send_info.target_address.to_v4(), send_info.target_port);

      msg.name = (PSOCKADDR) &taddr_v4;
      msg.namelen = sizeof(taddr_v4);
    }

    auto const max_bufs_per_msg = send_info.payload_buffers.size() + (send_info.headers ? 1 : 0);

    WSABUF bufs[(send_info.headers ? send_info.block_count : 1) * max_bufs_per_msg];
    DWORD bufcount = 0;
    if (send_info.headers) {
      // Interleave buffers for headers and payloads
      for (auto i = 0; i < send_info.block_count; i++) {
        bufs[bufcount].buf = (char *) &send_info.headers[(send_info.block_offset + i) * send_info.header_size];
        bufs[bufcount].len = send_info.header_size;
        bufcount++;
        auto payload_desc = send_info.buffer_for_payload_offset((send_info.block_offset + i) * send_info.payload_size);
        bufs[bufcount].buf = (char *) payload_desc.buffer;
        bufs[bufcount].len = send_info.payload_size;
        bufcount++;
      }
    } else {
      // Translate buffer descriptors into WSABUFs
      auto payload_offset = send_info.block_offset * send_info.payload_size;
      auto payload_length = payload_offset + (send_info.block_count * send_info.payload_size);
      while (payload_offset < payload_length) {
        auto payload_desc = send_info.buffer_for_payload_offset(payload_offset);
        bufs[bufcount].buf = (char *) payload_desc.buffer;
        bufs[bufcount].len = std::min(payload_desc.size, payload_length - payload_offset);
        payload_offset += bufs[bufcount].len;
        bufcount++;
      }
    }

    msg.lpBuffers = bufs;
    msg.dwBufferCount = bufcount;
    msg.dwFlags = 0;

    // At most, one DWORD option and one PKTINFO option
    char cmbuf[WSA_CMSG_SPACE(sizeof(DWORD)) + std::max(WSA_CMSG_SPACE(sizeof(IN6_PKTINFO)), WSA_CMSG_SPACE(sizeof(IN_PKTINFO)))] = {};
    ULONG cmbuflen = 0;

    msg.Control.buf = cmbuf;
    msg.Control.len = sizeof(cmbuf);

    auto cm = WSA_CMSG_FIRSTHDR(&msg);
    if (send_info.source_address.is_v6()) {
      IN6_PKTINFO pktInfo;

      SOCKADDR_IN6 saddr_v6 = to_sockaddr(send_info.source_address.to_v6(), 0);
      pktInfo.ipi6_addr = saddr_v6.sin6_addr;
      pktInfo.ipi6_ifindex = 0;

      cmbuflen += WSA_CMSG_SPACE(sizeof(pktInfo));

      cm->cmsg_level = IPPROTO_IPV6;
      cm->cmsg_type = IPV6_PKTINFO;
      cm->cmsg_len = WSA_CMSG_LEN(sizeof(pktInfo));
      memcpy(WSA_CMSG_DATA(cm), &pktInfo, sizeof(pktInfo));
    } else {
      IN_PKTINFO pktInfo;

      SOCKADDR_IN saddr_v4 = to_sockaddr(send_info.source_address.to_v4(), 0);
      pktInfo.ipi_addr = saddr_v4.sin_addr;
      pktInfo.ipi_ifindex = 0;

      cmbuflen += WSA_CMSG_SPACE(sizeof(pktInfo));

      cm->cmsg_level = IPPROTO_IP;
      cm->cmsg_type = IP_PKTINFO;
      cm->cmsg_len = WSA_CMSG_LEN(sizeof(pktInfo));
      memcpy(WSA_CMSG_DATA(cm), &pktInfo, sizeof(pktInfo));
    }

    if (send_info.block_count > 1) {
      cmbuflen += WSA_CMSG_SPACE(sizeof(DWORD));

      cm = WSA_CMSG_NXTHDR(&msg, cm);
      cm->cmsg_level = IPPROTO_UDP;
      cm->cmsg_type = UDP_SEND_MSG_SIZE;
      cm->cmsg_len = WSA_CMSG_LEN(sizeof(DWORD));
      *((DWORD *) WSA_CMSG_DATA(cm)) = send_info.header_size + send_info.payload_size;
    }

    msg.Control.len = cmbuflen;

    // If USO is not supported, this will fail and the caller will fall back to unbatched sends.
    DWORD bytes_sent;
    return WSASendMsg((SOCKET) send_info.native_socket, &msg, 0, &bytes_sent, nullptr, nullptr) != SOCKET_ERROR;
  }

  bool send(send_info_t &send_info) {
    WSAMSG msg;

    // Convert the target address into a SOCKADDR
    SOCKADDR_IN taddr_v4;
    SOCKADDR_IN6 taddr_v6;
    if (send_info.target_address.is_v6()) {
      taddr_v6 = to_sockaddr(send_info.target_address.to_v6(), send_info.target_port);

      msg.name = (PSOCKADDR) &taddr_v6;
      msg.namelen = sizeof(taddr_v6);
    } else {
      taddr_v4 = to_sockaddr(send_info.target_address.to_v4(), send_info.target_port);

      msg.name = (PSOCKADDR) &taddr_v4;
      msg.namelen = sizeof(taddr_v4);
    }

    WSABUF bufs[2];
    DWORD bufcount = 0;
    if (send_info.header) {
      bufs[bufcount].buf = (char *) send_info.header;
      bufs[bufcount].len = send_info.header_size;
      bufcount++;
    }
    bufs[bufcount].buf = (char *) send_info.payload;
    bufs[bufcount].len = send_info.payload_size;
    bufcount++;

    msg.lpBuffers = bufs;
    msg.dwBufferCount = bufcount;
    msg.dwFlags = 0;

    char cmbuf[std::max(WSA_CMSG_SPACE(sizeof(IN6_PKTINFO)), WSA_CMSG_SPACE(sizeof(IN_PKTINFO)))] = {};
    ULONG cmbuflen = 0;

    msg.Control.buf = cmbuf;
    msg.Control.len = sizeof(cmbuf);

    auto cm = WSA_CMSG_FIRSTHDR(&msg);
    if (send_info.source_address.is_v6()) {
      IN6_PKTINFO pktInfo;

      SOCKADDR_IN6 saddr_v6 = to_sockaddr(send_info.source_address.to_v6(), 0);
      pktInfo.ipi6_addr = saddr_v6.sin6_addr;
      pktInfo.ipi6_ifindex = 0;

      cmbuflen += WSA_CMSG_SPACE(sizeof(pktInfo));

      cm->cmsg_level = IPPROTO_IPV6;
      cm->cmsg_type = IPV6_PKTINFO;
      cm->cmsg_len = WSA_CMSG_LEN(sizeof(pktInfo));
      memcpy(WSA_CMSG_DATA(cm), &pktInfo, sizeof(pktInfo));
    } else {
      IN_PKTINFO pktInfo;

      SOCKADDR_IN saddr_v4 = to_sockaddr(send_info.source_address.to_v4(), 0);
      pktInfo.ipi_addr = saddr_v4.sin_addr;
      pktInfo.ipi_ifindex = 0;

      cmbuflen += WSA_CMSG_SPACE(sizeof(pktInfo));

      cm->cmsg_level = IPPROTO_IP;
      cm->cmsg_type = IP_PKTINFO;
      cm->cmsg_len = WSA_CMSG_LEN(sizeof(pktInfo));
      memcpy(WSA_CMSG_DATA(cm), &pktInfo, sizeof(pktInfo));
    }

    msg.Control.len = cmbuflen;

    DWORD bytes_sent;
    if (WSASendMsg((SOCKET) send_info.native_socket, &msg, 0, &bytes_sent, nullptr, nullptr) == SOCKET_ERROR) {
      auto winerr = WSAGetLastError();
      BOOST_LOG(warning) << "WSASendMsg() failed: "sv << winerr;
      return false;
    }

    return true;
  }

  class qos_t: public deinit_t {
  public:
    qos_t(QOS_FLOWID flow_id):
        flow_id(flow_id) {
    }

    virtual ~qos_t() {
      if (!fn_QOSRemoveSocketFromFlow(qos_handle, (SOCKET) nullptr, flow_id, 0)) {
        auto winerr = GetLastError();
        BOOST_LOG(warning) << "QOSRemoveSocketFromFlow() failed: "sv << winerr;
      }
    }

  private:
    QOS_FLOWID flow_id;
  };

  /**
   * @brief Enables QoS on the given socket for traffic to the specified destination.
   * @param native_socket The native socket handle.
   * @param address The destination address for traffic sent on this socket.
   * @param port The destination port for traffic sent on this socket.
   * @param data_type The type of traffic sent on this socket.
   * @param dscp_tagging Specifies whether to enable DSCP tagging on outgoing traffic.
   */
  std::unique_ptr<deinit_t> enable_socket_qos(uintptr_t native_socket, boost::asio::ip::address &address, uint16_t port, qos_data_type_e data_type, bool dscp_tagging) {
    SOCKADDR_IN saddr_v4;
    SOCKADDR_IN6 saddr_v6;
    PSOCKADDR dest_addr;
    bool using_connect_hack = false;

    // Windows doesn't support any concept of traffic priority without DSCP tagging
    if (!dscp_tagging) {
      return nullptr;
    }

    static std::once_flag load_qwave_once_flag;
    std::call_once(load_qwave_once_flag, []() {
      // qWAVE is not installed by default on Windows Server, so we load it dynamically
      HMODULE qwave = LoadLibraryExA("qwave.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
      if (!qwave) {
        BOOST_LOG(debug) << "qwave.dll is not available on this OS"sv;
        return;
      }

      fn_QOSCreateHandle = (decltype(fn_QOSCreateHandle)) GetProcAddress(qwave, "QOSCreateHandle");
      fn_QOSAddSocketToFlow = (decltype(fn_QOSAddSocketToFlow)) GetProcAddress(qwave, "QOSAddSocketToFlow");
      fn_QOSRemoveSocketFromFlow = (decltype(fn_QOSRemoveSocketFromFlow)) GetProcAddress(qwave, "QOSRemoveSocketFromFlow");

      if (!fn_QOSCreateHandle || !fn_QOSAddSocketToFlow || !fn_QOSRemoveSocketFromFlow) {
        BOOST_LOG(error) << "qwave.dll is missing exports?"sv;

        fn_QOSCreateHandle = nullptr;
        fn_QOSAddSocketToFlow = nullptr;
        fn_QOSRemoveSocketFromFlow = nullptr;

        FreeLibrary(qwave);
        return;
      }

      QOS_VERSION qos_version {1, 0};
      if (!fn_QOSCreateHandle(&qos_version, &qos_handle)) {
        auto winerr = GetLastError();
        BOOST_LOG(warning) << "QOSCreateHandle() failed: "sv << winerr;
        return;
      }
    });

    // If qWAVE is unavailable, just return
    if (!fn_QOSAddSocketToFlow || !qos_handle) {
      return nullptr;
    }

    auto disconnect_fg = util::fail_guard([&]() {
      if (using_connect_hack) {
        SOCKADDR_IN6 empty = {};
        empty.sin6_family = AF_INET6;
        if (connect((SOCKET) native_socket, (PSOCKADDR) &empty, sizeof(empty)) < 0) {
          auto wsaerr = WSAGetLastError();
          BOOST_LOG(error) << "qWAVE dual-stack workaround failed: "sv << wsaerr;
        }
      }
    });

    if (address.is_v6()) {
      auto address_v6 = address.to_v6();

      saddr_v6 = to_sockaddr(address_v6, port);
      dest_addr = (PSOCKADDR) &saddr_v6;

      // qWAVE doesn't properly support IPv4-mapped IPv6 addresses, nor does it
      // correctly support IPv4 addresses on a dual-stack socket (despite MSDN's
      // claims to the contrary). To get proper QoS tagging when hosting in dual
      // stack mode, we will temporarily connect() the socket to allow qWAVE to
      // successfully initialize a flow, then disconnect it again so WSASendMsg()
      // works later on.
      if (address_v6.is_v4_mapped()) {
        if (connect((SOCKET) native_socket, (PSOCKADDR) &saddr_v6, sizeof(saddr_v6)) < 0) {
          auto wsaerr = WSAGetLastError();
          BOOST_LOG(error) << "qWAVE dual-stack workaround failed: "sv << wsaerr;
        } else {
          BOOST_LOG(debug) << "Using qWAVE connect() workaround for QoS tagging"sv;
          using_connect_hack = true;
          dest_addr = nullptr;
        }
      }
    } else {
      saddr_v4 = to_sockaddr(address.to_v4(), port);
      dest_addr = (PSOCKADDR) &saddr_v4;
    }

    QOS_TRAFFIC_TYPE traffic_type;
    switch (data_type) {
      case qos_data_type_e::audio:
        traffic_type = QOSTrafficTypeVoice;
        break;
      case qos_data_type_e::video:
        traffic_type = QOSTrafficTypeAudioVideo;
        break;
      default:
        BOOST_LOG(error) << "Unknown traffic type: "sv << (int) data_type;
        return nullptr;
    }

    QOS_FLOWID flow_id = 0;
    if (!fn_QOSAddSocketToFlow(qos_handle, (SOCKET) native_socket, dest_addr, traffic_type, QOS_NON_ADAPTIVE_FLOW, &flow_id)) {
      auto winerr = GetLastError();
      BOOST_LOG(warning) << "QOSAddSocketToFlow() failed: "sv << winerr;
      return nullptr;
    }

    return std::make_unique<qos_t>(flow_id);
  }

  int64_t qpc_counter() {
    LARGE_INTEGER performance_counter;
    if (QueryPerformanceCounter(&performance_counter)) {
      return performance_counter.QuadPart;
    }
    return 0;
  }

  std::chrono::nanoseconds qpc_time_difference(int64_t performance_counter1, int64_t performance_counter2) {
    auto get_frequency = []() {
      LARGE_INTEGER frequency;
      frequency.QuadPart = 0;
      QueryPerformanceFrequency(&frequency);
      return frequency.QuadPart;
    };
    static const std::int64_t frequency = get_frequency();
    if (frequency > 0) {
      const auto seconds = std::chrono::duration<long double> {
        static_cast<long double>(performance_counter1 - performance_counter2) /
        static_cast<long double>(frequency)
      };
      return std::chrono::duration_cast<std::chrono::nanoseconds>(seconds);
    }
    return {};
  }

  std::wstring from_utf8(const std::string_view &string) {
    // No conversion needed if the string is empty
    if (string.empty()) {
      return {};
    }

    // Get the output size required to store the string
    auto output_size = MultiByteToWideChar(CP_UTF8, 0, string.data(), string.size(), nullptr, 0);
    if (output_size == 0) {
      auto winerr = GetLastError();
      BOOST_LOG(error) << "Failed to get UTF-16 buffer size: "sv << winerr;
      return {};
    }

    // Perform the conversion
    std::wstring output(output_size, L'\0');
    output_size = MultiByteToWideChar(CP_UTF8, 0, string.data(), string.size(), output.data(), output.size());
    if (output_size == 0) {
      auto winerr = GetLastError();
      BOOST_LOG(error) << "Failed to convert string to UTF-16: "sv << winerr;
      return {};
    }

    return output;
  }

  std::string to_utf8(const std::wstring_view &string) {
    // No conversion needed if the string is empty
    if (string.empty()) {
      return {};
    }

    // Get the output size required to store the string
    auto output_size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, string.data(), string.size(), nullptr, 0, nullptr, nullptr);
    if (output_size == 0) {
      auto winerr = GetLastError();
      BOOST_LOG(error) << "Failed to get UTF-8 buffer size: "sv << winerr;
      return {};
    }

    // Perform the conversion
    std::string output(output_size, '\0');
    output_size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, string.data(), string.size(), output.data(), output.size(), nullptr, nullptr);
    if (output_size == 0) {
      auto winerr = GetLastError();
      BOOST_LOG(error) << "Failed to convert string to UTF-8: "sv << winerr;
      return {};
    }

    return output;
  }

  std::string get_host_name() {
    WCHAR hostname[256];
    if (GetHostNameW(hostname, ARRAYSIZE(hostname)) == SOCKET_ERROR) {
      BOOST_LOG(error) << "GetHostNameW() failed: "sv << WSAGetLastError();
      return "Sunshine"s;
    }
    return to_utf8(hostname);
  }

  class win32_high_precision_timer: public high_precision_timer {
  public:
    win32_high_precision_timer() {
      // Use CREATE_WAITABLE_TIMER_HIGH_RESOLUTION if supported (Windows 10 1809+)
      timer = CreateWaitableTimerEx(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
      if (!timer) {
        timer = CreateWaitableTimerEx(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
        if (!timer) {
          BOOST_LOG(error) << "Unable to create high_precision_timer, CreateWaitableTimerEx() failed: " << GetLastError();
        }
      }
    }

    ~win32_high_precision_timer() {
      if (timer) {
        CloseHandle(timer);
      }
    }

    void sleep_for(const std::chrono::nanoseconds &duration) override {
      if (!timer) {
        BOOST_LOG(error) << "Attempting high_precision_timer::sleep_for() with uninitialized timer";
        return;
      }
      if (duration < 0s) {
        BOOST_LOG(error) << "Attempting high_precision_timer::sleep_for() with negative duration";
        return;
      }
      if (duration > 5s) {
        BOOST_LOG(error) << "Attempting high_precision_timer::sleep_for() with unexpectedly large duration (>5s)";
        return;
      }

      LARGE_INTEGER due_time;
      due_time.QuadPart = duration.count() / -100;
      SetWaitableTimer(timer, &due_time, 0, nullptr, nullptr, false);
      WaitForSingleObject(timer, INFINITE);
    }

    operator bool() override {
      return timer != nullptr;
    }

  private:
    HANDLE timer = nullptr;
  };

  std::unique_ptr<high_precision_timer> create_high_precision_timer() {
    return std::make_unique<win32_high_precision_timer>();
  }

  std::string
  get_clipboard() {
    std::string currentClipboard = to_utf8(getClipboardData());
    return currentClipboard;
  }

  bool
  set_clipboard(const std::string& content) {
    std::wstring cpContent = from_utf8(ensureCrLf(content));
    return !setClipboardData(cpContent);
  }
}  // namespace platf

static std::string ensureCrLf(const std::string& utf8Str) {
    std::string result;
    result.reserve(utf8Str.size() + utf8Str.size() / 2); // Reserve extra space

    for (size_t i = 0; i < utf8Str.size(); ++i) {
        if (utf8Str[i] == '\n' && (i == 0 || utf8Str[i - 1] != '\r')) {
            result += '\r'; // Add \r before \n if not present
        }
        result += utf8Str[i]; // Always add the current character
    }

    return result;
}

static std::wstring getClipboardData() {
  if (!OpenClipboard(nullptr)) {
    BOOST_LOG(warning) << "Failed to open clipboard.";
    return L"";
  }

  HANDLE hData = GetClipboardData(CF_UNICODETEXT);
  if (hData == nullptr) {
    BOOST_LOG(warning) << "No text data in clipboard or failed to get data.";
    CloseClipboard();
    return L"";
  }

  wchar_t* pszText = static_cast<wchar_t*>(GlobalLock(hData));
  if (pszText == nullptr) {
    BOOST_LOG(warning) << "Failed to lock clipboard data.";
    CloseClipboard();
    return L"";
  }

  std::wstring ret = pszText;

  GlobalUnlock(hData);
  CloseClipboard();

  return ret;
}

static int setClipboardData(const std::wstring& utf16Str) {
  if (!OpenClipboard(nullptr)) {
    BOOST_LOG(warning) << "Failed to open clipboard.";
    return 1;
  }

  if (!EmptyClipboard()) {
    BOOST_LOG(warning) << "Failed to empty clipboard.";
    CloseClipboard();
    return 1;
  }

  // Allocate global memory for the clipboard text
  HGLOBAL hGlobal = GlobalAlloc(GMEM_MOVEABLE, utf16Str.size() * 2 + 2);
  if (hGlobal == nullptr) {
    BOOST_LOG(warning) << "Failed to allocate global memory.";
    CloseClipboard();
    return 1;
  }

  // Lock the global memory and copy the text
  char* pGlobal = static_cast<char*>(GlobalLock(hGlobal));
  if (pGlobal == nullptr) {
    BOOST_LOG(warning) << "Failed to lock global memory.";
    GlobalFree(hGlobal);
    CloseClipboard();
    return 1;
  }

  memcpy(pGlobal, utf16Str.c_str(), utf16Str.size() * 2 + 2);
  GlobalUnlock(hGlobal);

  // Set the clipboard data
  if (SetClipboardData(CF_UNICODETEXT, hGlobal) == nullptr) {
    BOOST_LOG(warning) << "Failed to set clipboard data.";
    GlobalFree(hGlobal);
    CloseClipboard();
    return 1;
  }

  CloseClipboard();

  return 0;
}

#ifdef BOOST_PROCESS_VERSION
  #undef BOOST_PROCESS_VERSION
#endif
