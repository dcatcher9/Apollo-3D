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
    text_hook,  ///< 2D text game: the host converts its picture; a text helper may supply a text mask.
  };

  struct detected_game {
    std::filesystem::path executable;
    std::uint32_t process_id {};
    std::uint64_t creation_time {};
    std::string name;
    std::uintptr_t window {};
    game_kind_e kind {game_kind_e::addon};
    /// A text game whose text the text helper can follow. Nothing is loaded into any other
    /// text game, and its text is found in the picture.
    bool text_helper {};
    /// A text game because its owner chose it on the setup card, where no rule says it is one.
    bool chosen {};
  };

  /** The foreground application on the captured display, as the rules read it. */
  struct foreground_application {
    /// What setup offers where a rule describes the application. Otherwise the application as
    /// far as it is known: its process, executable, name and window.
    detected_game game;
    /// A rule describes it as a game.
    bool recognised {};
    /// Its owner may choose it for conversion as a 2D game: no rule takes it for a text game
    /// already, and the add-on is not loaded in it. `as_chosen` is the game it then is.
    bool choosable {};
    /// Whether the text helper could follow its text once it is chosen.
    bool text_helper {};
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

  // The same observation, also of an application no rule describes. Such an application is one
  // its owner may choose for conversion: a verified, live process outside the Windows directory
  // and the non-game suppression list whose whole client area is on the captured output. No
  // candidate where even that does not hold.
  [[nodiscard]] std::optional<foreground_application> observe_foreground(RECT capture_rect) noexcept;

  // The text game an application is once its owner has chosen it.
  [[nodiscard]] detected_game as_chosen(const foreground_application &application);

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
      bool gdi_glyphs {};  ///< `GetGlyphOutline`: it takes glyph bitmaps from GDI and composes its text itself.
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
    // - 32-bit, its executable importing a GDI text or glyph function, with a game library loaded;
    // - 64-bit, its executable itself importing Direct2D and DirectWrite, or the GDI glyph
    //   function, with a game library loaded. Loaded modules alone say nothing there: many 64-bit
    //   programs load Direct2D, and ordinary 3D games import a GDI text function too.
    [[nodiscard]] bool is_text_game_candidate(
      const std::filesystem::path &executable,
      std::span<const std::filesystem::path> loaded_modules,
      const foreground_window::observation_t &foreground,
      RECT capture_rect,
      bool native,
      const executable_imports_t &imports
    );

    // Whether an application may be chosen by its owner for conversion, as far as its window
    // tells: the same placement a text game needs, with no mark of a game asked for.
    [[nodiscard]] bool is_choosable(
      const std::filesystem::path &executable,
      const foreground_window::observation_t &foreground,
      RECT capture_rect
    );

    // Whether the text helper can follow a text game's text: the game draws it with Direct2D or
    // with a GDI text function, or it draws with Direct3D 11, where the helper follows the
    // texture a self-composed text is loaded into. A game that composes its text and draws by
    // other means is not followed. Asked only about a game the rules above took for a text game.
    [[nodiscard]] bool text_helper_serves(
      std::span<const std::filesystem::path> loaded_modules,
      bool native,
      const executable_imports_t &imports
    );
  }  // namespace detail
}  // namespace platf::game3d
