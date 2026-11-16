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
  /**
   * What a game can supply to Game 3D. The host supplies the rest: an add-on game sends finished
   * stereo; a text game has no depth, so it sends only where its text is and the host adds depth.
   */
  enum class game_kind_e {
    addon,  ///< 64-bit D3D11/D3D12 game: the ReShade add-on supplies depth, UI mask and stereo.
    text_hook,  ///< 32-bit DirectWrite text game: the in-application text hook supplies a text mask.
  };

  struct detected_game {
    std::filesystem::path executable;
    std::uint32_t process_id {};
    std::uint64_t creation_time {};
    std::string name;
    std::uintptr_t window {};
    game_kind_e kind {game_kind_e::addon};
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

    // A 32-bit process can never load the add-on. It is a text-game candidate when DirectWrite
    // and Direct2D are loaded, which is how such a game draws its dialogue, and its whole client
    // area is on the captured output. It may be a window: the host converts a text game from the
    // captured picture, so unlike an add-on game it need not cover the output.
    [[nodiscard]] bool is_text_game_candidate(
      const std::filesystem::path &executable,
      std::span<const std::filesystem::path> loaded_modules,
      const foreground_window::observation_t &foreground,
      RECT capture_rect
    );
  }  // namespace detail
}  // namespace platf::game3d
