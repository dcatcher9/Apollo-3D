// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <Windows.h>
#include <MinHook.h>

namespace sunshine_game3d::hook_host {
  inline bool minhook_ready(bool &flag) {
    if (!flag) {
      const auto status = MH_Initialize();
      flag = status == MH_OK || status == MH_ERROR_ALREADY_INITIALIZED;
    }
    return flag;
  }

  inline bool executable_image(const MEMORY_BASIC_INFORMATION &memory) {
    return memory.State == MEM_COMMIT && memory.Type == MEM_IMAGE &&
      (memory.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) &&
      !(memory.Protect & (PAGE_GUARD | PAGE_NOACCESS));
  }

  // Per-Present source maintenance, sequenced under the Streamline-owned lock.
  void poll();
  void initialize(HMODULE addon);
  void shutdown();
}
