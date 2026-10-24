// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Every hook a test binary's add-on code creates through MinHook, in order:
// the baseline of the installed hook set (refactor plan P4). CMake's
// sunshine_minhook_ledger(<target>) links the binary with
// -Wl,--wrap=MH_CreateHook and defines SUNSHINE_MINHOOK_LEDGER; then this
// header routes each MH_CreateHook through the ledger. Include it in exactly
// one file of such a binary. Without the ledger (another toolchain) it records
// nothing and available() is false.
#include <cstddef>
#include <MinHook.h>
#include <mutex>
#include <vector>

namespace minhook_ledger {
  struct hook {
    void *target;
    void *detour;
  };

  inline std::mutex lock;
  inline std::vector<hook> created;

  constexpr bool available() {
#ifdef SUNSHINE_MINHOOK_LEDGER
    return true;
#else
    return false;
#endif
  }

  // The position of the next hook, for created_since.
  inline std::size_t mark() {
    std::lock_guard guard(lock);
    return created.size();
  }

  // The targets of the hooks created since a mark, in order.
  inline std::vector<void *> created_since(std::size_t from) {
    std::lock_guard guard(lock);
    std::vector<void *> out;
    for (std::size_t i = from; i < created.size(); ++i) {
      out.push_back(created[i].target);
    }
    return out;
  }
}  // namespace minhook_ledger

#ifdef SUNSHINE_MINHOOK_LEDGER
extern "C" MH_STATUS WINAPI __real_MH_CreateHook(LPVOID target, LPVOID detour, LPVOID *original);

extern "C" MH_STATUS WINAPI __wrap_MH_CreateHook(LPVOID target, LPVOID detour, LPVOID *original) {
  const auto status = __real_MH_CreateHook(target, detour, original);
  if (status == MH_OK) {
    std::lock_guard guard(minhook_ledger::lock);
    minhook_ledger::created.push_back({target, detour});
  }
  return status;
}
#endif
