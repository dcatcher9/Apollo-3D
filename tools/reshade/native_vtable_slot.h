// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <Windows.h>
#include <cstddef>
#include <cstdint>

namespace sunshine_native_vtable {
  struct slot {
    void **address{};
    void *original{};
    HMODULE table_owner{}, code_owner{};
  };

  namespace detail {
    // Separate observers may edit different slots on the same read-only page.
    // Share the protection/CAS/restore critical section across translation units.
    inline SRWLOCK mutation_lock = SRWLOCK_INIT;
    struct restore_error {
      DWORD value{GetLastError()};
      ~restore_error() { SetLastError(value); }
    };
    inline bool read(const void *source, void *output, std::size_t size) {
      SIZE_T copied{};
      return source && ReadProcessMemory(GetCurrentProcess(), source, output, size, &copied) && copied == size;
    }
    inline bool image(const void *address, bool executable) {
      MEMORY_BASIC_INFORMATION memory{};
      return VirtualQuery(address, &memory, sizeof(memory)) == sizeof(memory) && memory.State == MEM_COMMIT &&
        memory.Type == MEM_IMAGE && !(memory.Protect & (PAGE_GUARD | PAGE_NOACCESS)) &&
        (!executable || (memory.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)));
    }
    inline bool retain(const void *address, HMODULE &owner) {
      return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(address), &owner) != FALSE;
    }
    inline bool pin(const void *address, HMODULE expected = nullptr) {
      HMODULE owner{};
      return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
        reinterpret_cast<LPCWSTR>(address), &owner) && (!expected || owner == expected);
    }
    inline bool exchange(const slot &value, void *expected, void *replacement) {
      AcquireSRWLockExclusive(&mutation_lock);
      struct unlock { ~unlock() { ReleaseSRWLockExclusive(&mutation_lock); } } held;
      MEMORY_BASIC_INFORMATION memory{};
      if (VirtualQuery(value.address, &memory, sizeof(memory)) != sizeof(memory) || memory.State != MEM_COMMIT ||
          memory.Type != MEM_IMAGE || (memory.Protect & (PAGE_GUARD | PAGE_NOACCESS))) return false;
      const bool executable = (memory.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;
      DWORD previous{};
      if (!VirtualProtect(value.address, sizeof(void *), executable ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE, &previous)) return false;
      auto *address = reinterpret_cast<void *volatile *>(value.address);
      const bool exchanged = InterlockedCompareExchangePointer(address, replacement, expected) == expected;
      DWORD ignored{};
      if (VirtualProtect(value.address, sizeof(void *), previous, &ignored)) return exchanged;
      // Fail closed if protection restoration failed. Do not overwrite a newer
      // third-party value during rollback; an entered detour still has its
      // previously published original and permanently pinned code.
      if (exchanged) InterlockedCompareExchangePointer(address, expected, replacement);
      VirtualProtect(value.address, sizeof(void *), previous, &ignored);
      return false;
    }
  }

  inline void release(slot &value) {
    const detail::restore_error incoming;
    if (value.table_owner) FreeLibrary(value.table_owner);
    if (value.code_owner) FreeLibrary(value.code_owner);
    value = {};
  }
  // Caller first authenticates the exact COM interface and holds its reference
  // while discovering. The output must be empty; it owns two module references.
  inline bool discover(std::uint64_t object, std::size_t index, slot &output) {
    const detail::restore_error incoming;
    std::uintptr_t table{};
    if (!detail::read(reinterpret_cast<const void *>(object), &table, sizeof(table)) || !table ||
        index > (UINTPTR_MAX - table) / sizeof(void *)) return false;
    const auto address = table + index * sizeof(void *);
    if (address % alignof(void *) || !detail::image(reinterpret_cast<const void *>(address), false)) return false;
    slot candidate;
    candidate.address = reinterpret_cast<void **>(address);
    if (!detail::read(candidate.address, &candidate.original, sizeof(candidate.original)) ||
        !detail::image(candidate.original, true) || !detail::retain(candidate.address, candidate.table_owner) ||
        !detail::retain(candidate.original, candidate.code_owner)) {
      release(candidate);
      return false;
    }
    output = candidate;
    return true;
  }
  inline bool matches(const slot &value, void *detour) {
    const detail::restore_error incoming;
    void *current{};
    return value.address && detail::read(value.address, &current, sizeof(current)) && current == detour;
  }
  // Publish the callable original before install: another thread can enter the
  // detour immediately after the CAS. Installed hooks remain for process life.
  inline bool install(const slot &value, void *detour) {
    const detail::restore_error incoming;
    return value.address && value.original && detour && detail::pin(value.address, value.table_owner) &&
      detail::pin(value.original, value.code_owner) && detail::pin(detour) && detail::exchange(value, value.original, detour);
  }
  // Quiescent fixture teardown only. A third-party replacement is left intact.
  inline bool restore(const slot &value, void *detour) {
    const detail::restore_error incoming;
    return value.address && detour && detail::exchange(value, detour, value.original);
  }
}
