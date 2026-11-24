// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook.cpp
 * @brief In-application observer of a game's text drawing for Game 3D: the published picture.
 *
 * Loaded only into an application its owner confirmed. The Direct2D part (text_hook_d2d.cpp) and
 * the GDI part (text_hook_gdi.cpp) watch the application's text calls and forward each one
 * unchanged; this part owns the picture of that text and its publication to the host. The helper
 * keeps no text content and changes nothing the application draws.
 */
#include "text_hook_core.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cwchar>

namespace sunshine_text_hook::helper {
  CRITICAL_SECTION state_lock;
  surface_t surface;

  namespace {
    snapshot_t *shared = nullptr;
    std::uint32_t *surface_pixels = nullptr;
    HANDLE surface_section = nullptr;
    // The size that could not be created, so that it is not tried again for every text draw.
    std::uint32_t failed_width = 0u, failed_height = 0u;
    DWORD nesting_slot = TLS_OUT_OF_INDEXES;
    FILE *log_file = nullptr;
    unsigned log_lines = 0u;

    void release_surface() {
      if (surface_pixels) {
        UnmapViewOfFile(surface_pixels);
        surface_pixels = nullptr;
      }
      if (surface_section) {
        CloseHandle(surface_section);
        surface_section = nullptr;
      }
      surface = {};
    }

    bool create_surface(const std::uint32_t width, const std::uint32_t height) {
      wchar_t name[128];
      std::swprintf(name, 128, L"%ls%lu-%ux%u", coverage_prefix, GetCurrentProcessId(), width, height);
      const std::size_t bytes = static_cast<std::size_t>(width) * height * 4u;
      surface_section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0u, static_cast<DWORD>(bytes), name);
      surface_pixels = surface_section ? static_cast<std::uint32_t *>(MapViewOfFile(surface_section, FILE_MAP_ALL_ACCESS, 0u, 0u, bytes)) : nullptr;
      if (!surface_pixels) {
        return false;
      }
      // A section of this name may have outlived an earlier picture of the same size.
      std::memset(surface_pixels, 0, bytes);
      surface = {width, height, surface_pixels};
      return true;
    }

    // The host reads between two even values of the sequence that are equal.
    void begin_publication() {
      InterlockedIncrement(reinterpret_cast<volatile LONG *>(&shared->sequence));
    }

    void end_publication(const source_e source, const std::uint32_t text_top, const std::uint32_t text_bottom, const HWND window) {
      shared->source = source;
      shared->surface_width = surface.width;
      shared->surface_height = surface.height;
      shared->text_top = text_top;
      shared->text_bottom = text_bottom;
      shared->window = reinterpret_cast<std::uintptr_t>(window);
      InterlockedIncrement(reinterpret_cast<volatile LONG *>(&shared->sequence));
    }
  }  // namespace

  void log_line(const char *format, ...) {
    if (!log_file || log_lines >= 4000u) {
      return;
    }
    ++log_lines;
    va_list arguments;
    va_start(arguments, format);
    std::vfprintf(log_file, format, arguments);
    va_end(arguments);
    std::fputc('\n', log_file);
    std::fflush(log_file);
  }

  bool logging() {
    return log_file != nullptr && log_lines < 4000u;
  }

  bool nested() {
    return nesting_slot != TLS_OUT_OF_INDEXES && TlsGetValue(nesting_slot) != nullptr;
  }

  void enter_call() {
    if (nesting_slot != TLS_OUT_OF_INDEXES) {
      TlsSetValue(nesting_slot, reinterpret_cast<void *>(reinterpret_cast<std::uintptr_t>(TlsGetValue(nesting_slot)) + 1u));
    }
  }

  void leave_call() {
    if (nesting_slot != TLS_OUT_OF_INDEXES) {
      TlsSetValue(nesting_slot, reinterpret_cast<void *>(reinterpret_cast<std::uintptr_t>(TlsGetValue(nesting_slot)) - 1u));
    }
  }

  void publish(const std::uint32_t *pixels, std::uint32_t top, std::uint32_t bottom, const source_e source, std::uint32_t text_top, std::uint32_t text_bottom, const HWND window) {
    if (!shared || !surface_pixels || !pixels) {
      return;
    }
    bottom = min_of(bottom, surface.height);
    top = min_of(top, bottom);
    if (text_bottom <= text_top || text_bottom > surface.height) {
      text_top = text_bottom = 0u;
    }
    const std::size_t offset = static_cast<std::size_t>(top) * surface.width;
    const std::size_t bytes = static_cast<std::size_t>(bottom - top) * surface.width * 4u;
    // An application that draws the same text again and again publishes it once.
    if (shared->source == static_cast<std::uint32_t>(source) && shared->text_top == text_top && shared->text_bottom == text_bottom &&
        shared->window == reinterpret_cast<std::uintptr_t>(window) && std::memcmp(surface_pixels + offset, pixels + offset, bytes) == 0) {
      return;
    }
    begin_publication();
    std::memcpy(surface_pixels + offset, pixels + offset, bytes);
    end_publication(source, text_top, text_bottom, window);
  }

  bool ensure_surface(const std::uint32_t width, const std::uint32_t height) {
    if (!shared || width == 0u || height == 0u || width > max_surface_extent || height > max_surface_extent) {
      return false;
    }
    if (surface.pixels && width == surface.width && height == surface.height) {
      return true;
    }
    if (width == failed_width && height == failed_height) {
      return false;
    }
    // The host must not take the old size for the new picture, nor a cleared picture for a
    // complete one: the change of size is a publication of its own.
    begin_publication();
    release_surface();
    const bool created = create_surface(width, height);
    if (!created) {
      release_surface();
      failed_width = width;
      failed_height = height;
    } else {
      failed_width = failed_height = 0u;
    }
    end_publication(source_none, 0u, 0u, nullptr);
    log_line("picture %ux%u %s", width, height, created ? "ready" : "unavailable");
    return created;
  }

  namespace {
    void open_log() {
      wchar_t log_path[MAX_PATH];
      if (GetEnvironmentVariableW(L"SUNSHINE_TEXT_HOOK_LOG", log_path, MAX_PATH) > 0u) {
        log_file = _wfopen(log_path, L"w");
        return;
      }
      // A diagnostic log is written only where its owner already put the file.
      const DWORD length = GetTempPathW(MAX_PATH, log_path);
      if (length != 0u && length + 24u < MAX_PATH) {
        std::wcscat(log_path, L"SunshineTextHook.log");
        if (GetFileAttributesW(log_path) != INVALID_FILE_ATTRIBUTES) {
          log_file = _wfopen(log_path, L"w");
        }
      }
    }

    DWORD WINAPI run(void *) {
      wchar_t name[96];
      std::swprintf(name, 96, L"%ls%lu", section_prefix, GetCurrentProcessId());
      if (HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0u, sizeof(snapshot_t), name)) {
        shared = static_cast<snapshot_t *>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0u, 0u, sizeof(snapshot_t)));
      }
      open_log();
      if (!shared) {
        return 1u;
      }
      FILETIME created {}, exited {}, kernel {}, user {};
      GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user);

      EnterCriticalSection(&state_lock);
      nesting_slot = TlsAlloc();
      // A section of this name may be one the host still holds from an earlier process.
      begin_publication();
      shared->process_created = (static_cast<std::uint64_t>(created.dwHighDateTime) << 32u) | created.dwLowDateTime;
      end_publication(source_none, 0u, 0u, nullptr);
      bool direct2d = d2d::install();
      gdi::install();
      // Only now is this a helper the host can rely on.
      shared->version = version;
      shared->magic = magic;
      LeaveCriticalSection(&state_lock);
      log_line("helper started, Direct2D %s", direct2d ? "hooked" : "not loaded yet");
      // Learned from a window and a device of the helper's own, which load libraries: not under
      // the lock that the application's threads wait for inside the helper's hooks.
      bool direct3d = d3d11::install();

      // The application's own end of drawing normally publishes its text. Text that no such call
      // follows is published from here once it has been quiet for a moment. The same thread
      // covers what the application loads later.
      const DWORD step_ms = static_cast<DWORD>(longest_batch_ms / 2u);
      for (unsigned step = 1u;; ++step) {
        Sleep(step_ms);
        EnterCriticalSection(&state_lock);
        const std::uint64_t tick = GetTickCount64();
        d2d::idle(tick);
        gdi::idle(tick);
        if (step % 40u == 0u) {
          if (!direct2d && (direct2d = d2d::install())) {
            log_line("Direct2D hooked");
          }

          gdi::install();
        }
        LeaveCriticalSection(&state_lock);
        if (step % 40u == 0u) {
          direct3d = direct3d || d3d11::install();
        }
      }
    }
  }  // namespace
}  // namespace sunshine_text_hook::helper

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, void *) {
  if (reason == DLL_PROCESS_ATTACH) {
    DisableThreadLibraryCalls(instance);
    InitializeCriticalSection(&sunshine_text_hook::helper::state_lock);
    // Direct2D, DirectWrite and the module list must not be entered under the loader lock.
    if (HANDLE thread = CreateThread(nullptr, 0u, sunshine_text_hook::helper::run, nullptr, 0u, nullptr)) {
      CloseHandle(thread);
    }
  }
  return TRUE;
}
