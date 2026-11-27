/**
 * @file src/virtual_display_lifecycle.h
 * @brief What process.cpp uses of virtual_display_lifecycle.cpp besides proc_t's own members.
 */
#pragma once

// standard includes
#include <cstdint>
#include <cstdlib>

// platform includes
#include <windows.h>

namespace proc {
  inline bool live_display_mode_matches(const DEVMODEW &mode, int width, int height, int fps_millihz) {
    // GDI reports integer Hz; use the same tolerance for live changes and retained resumes.
    return width > 0 && height > 0 && fps_millihz > 0 &&
           mode.dmPelsWidth == static_cast<DWORD>(width) &&
           mode.dmPelsHeight == static_cast<DWORD>(height) &&
           std::abs(static_cast<std::int64_t>(mode.dmDisplayFrequency) * 1000 - fps_millihz) < 1000;
  }

  /**
   * Final teardown of a session whose display the retirement owner already holds. The owner
   * removes it as a final teardown, with the Explorer repair the configuration asks for; when
   * that removal already finished, the repair it left owed is scheduled now.
   */
  void mark_retirement_final_teardown();

  /** Host shutdown: stop the retirement and Explorer-repair workers and join them. */
  void stopRetiredVirtualDisplayCleanupWorker();
}  // namespace proc
