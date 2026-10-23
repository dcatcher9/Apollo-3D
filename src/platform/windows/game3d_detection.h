// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "foreground_window_region.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <windows.h>

namespace platf::game3d {
  struct detected_game {
    std::filesystem::path executable;
    std::uint32_t process_id {};
    std::uint64_t creation_time {};
    std::string name;
    std::uintptr_t window {};
  };

  // Worker-thread observation of the foreground application on the captured display. One
  // general rule, without store-library or engine-layout lists: a verified, live native x64
  // process outside the Windows directory and the non-game suppression list, with D3D11/12
  // loaded, whose foreground client exactly covers the captured output. A candidate only selects
  // what setup may offer; this never injects into the process or changes presentation mode.
  // Process access/enumeration failure and a changing foreground identity return no candidate.
  [[nodiscard]] std::optional<detected_game> detect_game(RECT capture_rect) noexcept;

  // A candidate must persist across worker polls for two seconds before setup acts on it.
  // Missing observations, identity/display changes and stale polls reset it.
  class detection_confirmation {
  public:
    [[nodiscard]] std::optional<detected_game> observe(
      std::optional<detected_game> candidate,
      RECT capture_rect,
      std::chrono::steady_clock::time_point now
    );
    void reset() noexcept;

  private:
    std::optional<detected_game> provisional_;
    RECT capture_rect_ {};
    std::chrono::steady_clock::time_point first_seen_ {};
    std::chrono::steady_clock::time_point last_seen_ {};
  };

  namespace detail {
    // Pure admission policy, shared with focused tests. Native process lifetime, x64, Windows
    // directory and module census checks are performed before calling this in detect_game().
    [[nodiscard]] bool is_game_candidate(
      const std::filesystem::path &executable,
      std::span<const std::filesystem::path> loaded_modules,
      const foreground_window::observation_t &foreground,
      RECT capture_rect
    );
  }  // namespace detail
}  // namespace platf::game3d
