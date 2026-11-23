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
    text_hook,  ///< 2D text game: the host converts its picture; its text helper supplies a text mask.
  };

  struct detected_game {
    std::filesystem::path executable;
    std::uint32_t process_id {};
    std::uint64_t creation_time {};
    std::string name;
    std::uintptr_t window {};
    game_kind_e kind {game_kind_e::addon};
  };

  // Worker-thread observation of the foreground application on the captured display. General
  // rules, without store-library or engine-layout lists, for a verified, live process outside the
  // Windows directory and the non-game suppression list. An add-on game is a native x64 process
  // with D3D11/12 loaded whose foreground client exactly covers the captured output. A text game
  // is one the rules of detail::is_text_game_candidate describe; a 64-bit process that fits both
  // is a text game unless the add-on is already loaded in it. A candidate only selects what
  // setup may offer; this never injects into the process or changes presentation mode.
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

    /** What a game's executable itself imports, read from its file. */
    struct executable_imports_t {
      bool direct2d_text {};  ///< Direct2D and DirectWrite: it draws its picture and text with them.
      bool gdi_text {};  ///< A GDI function that draws text.
    };

    // Reads the import table of an executable file, 32-bit or 64-bit, without loading it. A file
    // that cannot be read, or whose imports a protector has hidden, imports nothing.
    [[nodiscard]] executable_imports_t inspect_imports(const std::filesystem::path &executable) noexcept;

    // A sound, input or older graphics library that games load and ordinary programs rarely do.
    [[nodiscard]] bool game_library_loaded(std::span<const std::filesystem::path> loaded_modules);

    // A text game is converted by the host from the captured picture, so unlike an add-on game it
    // may be a window: its whole client area must be on the captured output. What marks it as one:
    // - 32-bit, which can never load the add-on, with DirectWrite and Direct2D loaded: this is how
    //   such a game draws its dialogue;
    // - 32-bit, its executable importing a GDI text function, with a game library loaded;
    // - 64-bit, its executable itself importing Direct2D and DirectWrite, with a game library
    //   loaded. Loaded modules alone say nothing there: many 64-bit programs load both.
    [[nodiscard]] bool is_text_game_candidate(
      const std::filesystem::path &executable,
      std::span<const std::filesystem::path> loaded_modules,
      const foreground_window::observation_t &foreground,
      RECT capture_rect,
      bool native,
      const executable_imports_t &imports
    );
  }  // namespace detail
}  // namespace platf::game3d
