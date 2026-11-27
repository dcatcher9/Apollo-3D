/**
 * @file src/platform/windows/user_token.h
 * @brief Interactive-user tokens shared by Windows process launch and Explorer repair.
 */
#pragma once

// standard includes
#include <functional>
#include <optional>
#include <string>
#include <system_error>

// platform includes
#include <Windows.h>

namespace platf {
  /** The active console user's primary token, linked to its administrator token when elevated is requested, or null. */
  HANDLE retrieve_users_token(bool elevated);

  /** A primary standard-user token for the current interactive account, or null when none is safe. */
  HANDLE retrieve_current_users_standard_token();

  /** Whether this process belongs to the account of the active desktop shell, or nullopt when unknown. */
  std::optional<bool> current_process_matches_active_standard_user();

  /** Whether this process token is elevated, or nullopt when unknown. */
  std::optional<bool> current_process_token_is_elevated();

  /** The string SID of the token's user, or nullopt. */
  std::optional<std::string> token_user_id(HANDLE token);

  /** Whether the token's enabled groups include Administrators, or nullopt when Windows cannot say. Accepts a primary token. */
  std::optional<bool> token_is_admin(HANDLE user_token);

  /** Whether this process runs as the local system account. */
  bool is_running_as_system();

  /** Run callback while impersonating user_token, then restore the thread's prior token. */
  std::error_code impersonate_current_user(
    HANDLE user_token,
    std::function<void()> callback
  );
}  // namespace platf
