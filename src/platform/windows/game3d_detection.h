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
  enum class detection_evidence_e {
    known_runtime,
    fullscreen_direct3d,
  };

  struct detected_game {
    std::filesystem::path executable;
    std::uint32_t process_id {};
    std::uint64_t creation_time {};
    std::string name;
    detection_evidence_e evidence {detection_evidence_e::known_runtime};
    std::uintptr_t window {};
  };

  // Worker-thread heuristic for the foreground application on the captured display. Requires
  // a verified, live native x64 process and D3D11/12. A recognizable game runtime or Unreal
  // shipping layout supports windowed games; otherwise the client must fill the captured display.
  // This selects automatic setup for the next launch; it never injects into the
  // running process or changes presentation mode.
  // Process access/enumeration failure and a changing foreground identity return no candidate.
  [[nodiscard]] std::optional<detected_game> detect_game(RECT capture_rect) noexcept;

  // The broader fullscreen heuristic must persist across worker polls before setup can write
  // beside an executable. Missing observations, identity/display changes and stale polls reset it.
  // Known-runtime evidence retains immediate setup for ordinary windowed games.
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
    // Pure admission policy, shared with focused tests. Native process lifetime, x64 and module
    // census checks are performed before calling this function in detect_game().
    [[nodiscard]] std::optional<detection_evidence_e> classify_game_candidate(
      const std::filesystem::path &executable,
      std::span<const std::filesystem::path> loaded_modules,
      const foreground_window::observation_t &foreground,
      RECT capture_rect
    );

    [[nodiscard]] bool is_game_candidate(
      const std::filesystem::path &executable,
      std::span<const std::filesystem::path> loaded_modules,
      const foreground_window::observation_t &foreground,
      RECT capture_rect
    );
  }  // namespace detail
}  // namespace platf::game3d
