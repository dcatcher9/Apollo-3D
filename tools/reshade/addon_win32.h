// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <Windows.h>

// Win32 helpers shared by the add-on's hooks and callbacks.
namespace sunshine_addon_win32 {
  // Pins the module whose image holds address until process exit and reports it in owner.
  // Hooks pin their own code and the code they call, so a later unload never pulls it away
  // from a call already in flight.
  template<class T>
  inline bool pin_module_of(T *address, HMODULE *owner = nullptr) noexcept {
    HMODULE module{};
    const bool pinned = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
      reinterpret_cast<LPCWSTR>(address), &module) != FALSE;
    if (owner) *owner = module;
    return pinned;
  }

  // Sets the thread's last-error code back when it leaves scope, so code the game did not call
  // never changes what the game's next GetLastError() reads. It saves the current code, or the
  // code given, such as one a detour read on entry.
  struct last_error_guard {
    DWORD value{GetLastError()};
    ~last_error_guard() { SetLastError(value); }
  };
}
