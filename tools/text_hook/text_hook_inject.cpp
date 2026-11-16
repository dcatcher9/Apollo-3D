// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook_inject.cpp
 * @brief Load the text hook into one application of the same bitness.
 *
 * usage: text_hook_inject <process id> <absolute path of the hook DLL>
 * exit: 0 loaded, 2 usage, 3 bitness mismatch, 4 cannot open, 5 load failed.
 */
#include <cstdlib>
#include <cwchar>
#include <windows.h>

namespace {
  bool is_32_bit(HANDLE process) {
    BOOL wow64 = FALSE;
    SYSTEM_INFO native {};
    GetNativeSystemInfo(&native);
    if (native.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_INTEL) {
      return true;
    }
    return IsWow64Process(process, &wow64) && wow64;
  }
}  // namespace

int wmain(int argc, wchar_t **argv) {
  if (argc != 3) {
    return 2;
  }
  const DWORD process_id = std::wcstoul(argv[1], nullptr, 10);
  const wchar_t *library = argv[2];
  if (process_id == 0u || GetFileAttributesW(library) == INVALID_FILE_ATTRIBUTES) {
    return 2;
  }
  HANDLE process = OpenProcess(
    PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
    FALSE, process_id
  );
  if (!process) {
    return 4;
  }
  int result = 5;
  if (is_32_bit(process) != (sizeof(void *) == 4u)) {
    result = 3;
  } else {
    const SIZE_T bytes = (std::wcslen(library) + 1u) * sizeof(wchar_t);
    void *remote = VirtualAllocEx(process, nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (remote && WriteProcessMemory(process, remote, library, bytes, nullptr)) {
      // kernel32 is mapped at the same address in every process of one bitness.
      const auto entry = reinterpret_cast<LPTHREAD_START_ROUTINE>(
        reinterpret_cast<void *>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW"))
      );
      if (HANDLE thread = CreateRemoteThread(process, nullptr, 0u, entry, remote, 0u, nullptr)) {
        DWORD module = 0u;
        if (WaitForSingleObject(thread, 10000u) == WAIT_OBJECT_0 && GetExitCodeThread(thread, &module) && module != 0u) {
          result = 0;
        }
        CloseHandle(thread);
      }
    }
    if (remote) {
      VirtualFreeEx(process, remote, 0u, MEM_RELEASE);
    }
  }
  CloseHandle(process);
  return result;
}
