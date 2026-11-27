/**
 * @file src/platform/windows/ar_glasses_security.h
 * @brief The desktop security guard: Monitor 3D may replace a desktop only while Windows shows
 * the unlocked interactive desktop, never a lock, sign-in or permission screen.
 */
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <windows.h>

namespace ar_glasses {
  namespace internal {
    std::wstring desktop_object_name(HANDLE desktop);

    /** The reason the selected monitor must return to 2D now, or empty while it may present. */
    std::string_view query_monitor_desktop_block_reason(std::wstring_view presentation_desktop, std::optional<RECT> selected_monitor = std::nullopt);
  }  // namespace internal
}  // namespace ar_glasses
