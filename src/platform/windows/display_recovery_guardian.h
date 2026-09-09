/**
 * @file src/platform/windows/display_recovery_guardian.h
 * @brief Independent recovery after a host with disabled physical displays exits.
 */
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <windows.h>

namespace platf::display_recovery_guardian {
  /** Establish a ready worker outside the host's jobs before disabling physical outputs. */
  bool ensure_running();

  /** Private command dispatch, before normal host configuration or platform initialization. */
  std::optional<int> run_if_requested();

  namespace detail {
    std::wstring quote_argument(std::wstring_view value);
    std::optional<HANDLE> parse_handle(std::wstring_view value);

    /** Injectable recovery action; waits for a real process handle, never a reusable PID. */
    DWORD watch_parent(HANDLE parent, HANDLE ready, const std::function<bool()> &recover, DWORD retry_ms = 5000);
  }  // namespace detail
}  // namespace platf::display_recovery_guardian
