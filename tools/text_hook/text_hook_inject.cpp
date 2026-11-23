// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook_inject.cpp
 * @brief Load the text helper into one application of the same bitness.
 *
 * usage: text_hook_inject <process id> <creation time> <absolute path of the helper DLL>
 * The creation time is the process's, as a FILETIME in decimal: a process id alone may by now
 * name another program.
 * exit: 0 loaded, 2 usage, 3 bitness mismatch, 4 cannot open or not that process, 5 load failed.
 */
#include <cstdlib>
#include <cwchar>
#include <windows.h>

#include <psapi.h>

namespace {
  /** 1 for a 32-bit process, 0 for a 64-bit one, -1 when it cannot be told. */
  int is_32_bit(HANDLE process) {
    SYSTEM_INFO native {};
    GetNativeSystemInfo(&native);
    if (native.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_INTEL) {
      return 1;
    }
    BOOL wow64 = FALSE;
    return IsWow64Process(process, &wow64) ? (wow64 ? 1 : 0) : -1;
  }

  unsigned long long creation_time(HANDLE process) {
    FILETIME created {}, exited {}, kernel {}, user {};
    if (!GetProcessTimes(process, &created, &exited, &kernel, &user)) {
      return 0u;
    }
    return (static_cast<unsigned long long>(created.dwHighDateTime) << 32u) | created.dwLowDateTime;
  }

  /** True when the process has a module of this file name. The remote thread's exit code is
   * only the low half of the module's address in a 64-bit process, so it cannot tell. */
  bool has_module(HANDLE process, const wchar_t *library) {
    const wchar_t *slash = std::wcsrchr(library, L'\\');
    const wchar_t *name = slash ? slash + 1 : library;
    static HMODULE modules[2048];
    DWORD bytes = 0u;
    if (!K32EnumProcessModules(process, modules, sizeof(modules), &bytes)) {
      return false;
    }
    const DWORD count = bytes / sizeof(HMODULE) < 2048u ? bytes / sizeof(HMODULE) : 2048u;
    for (DWORD index = 0u; index < count; ++index) {
      wchar_t base[MAX_PATH];
      if (K32GetModuleBaseNameW(process, modules[index], base, MAX_PATH) != 0u && _wcsicmp(base, name) == 0) {
        return true;
      }
    }
    return false;
  }
}  // namespace

int wmain(int argc, wchar_t **argv) {
  if (argc != 4) {
    return 2;
  }
  const DWORD process_id = std::wcstoul(argv[1], nullptr, 10);
  const unsigned long long created = std::wcstoull(argv[2], nullptr, 10);
  const wchar_t *library = argv[3];
  if (process_id == 0u || created == 0u || GetFileAttributesW(library) == INVALID_FILE_ATTRIBUTES) {
    return 2;
  }
  HANDLE process = OpenProcess(
    PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ | SYNCHRONIZE,
    FALSE, process_id
  );
  if (!process) {
    return 4;
  }
  int result = 5;
  const int narrow = is_32_bit(process);
  if (narrow < 0 || creation_time(process) != created) {
    result = 4;
  } else if ((narrow == 1) != (sizeof(void *) == 4u)) {
    result = 3;
  } else {
    const SIZE_T bytes = (std::wcslen(library) + 1u) * sizeof(wchar_t);
    void *remote = VirtualAllocEx(process, nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    bool finished = true;
    if (remote && WriteProcessMemory(process, remote, library, bytes, nullptr)) {
      // kernel32 is mapped at the same address in every process of one bitness.
      const auto entry = reinterpret_cast<LPTHREAD_START_ROUTINE>(
        reinterpret_cast<void *>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW"))
      );
      if (HANDLE thread = CreateRemoteThread(process, nullptr, 0u, entry, remote, 0u, nullptr)) {
        finished = WaitForSingleObject(thread, 10000u) == WAIT_OBJECT_0;
        if (finished && WaitForSingleObject(process, 0u) == WAIT_TIMEOUT && has_module(process, library)) {
          result = 0;
        }
        CloseHandle(thread);
      }
    }
    // A thread that has not run yet still needs the path: a few bytes stay behind instead.
    if (remote && finished) {
      VirtualFreeEx(process, remote, 0u, MEM_RELEASE);
    }
  }
  CloseHandle(process);
  return result;
}
