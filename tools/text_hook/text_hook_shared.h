// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook_shared.h
 * @brief Shared-memory contract between the in-application text helper and the Game 3D host.
 *
 * The helper runs inside a game its owner confirmed for Game 3D (docs/reshade-sbs.md, Text
 * games). It reports a picture of the text the game has drawn, in the pixels of the surface the
 * game draws on, and the rows of that picture that hold text. It never reports the text itself.
 * The layout is identical for 32-bit and 64-bit builds.
 */
#pragma once

#include <cstdint>

namespace sunshine_text_hook {
  /** State section name: this prefix followed by the decimal process id of the application. */
  inline constexpr wchar_t section_prefix[] = L"Local\\SunshineTextHook-";
  /**
   * Coverage section name: this prefix, the process id, '-', surface width, 'x', surface height.
   * It holds width*height BGRA pixels, top row first, premultiplied: the text as the application
   * drew it, in its own colours, on nothing. Alpha is text coverage.
   */
  inline constexpr wchar_t coverage_prefix[] = L"Local\\SunshineTextHookCoverage-";
  inline constexpr std::uint32_t magic = 0x34485453u;  // "STH4"
  inline constexpr std::uint32_t version = 4u;
  /// Neither side of the picture is longer than this.
  inline constexpr std::uint32_t max_surface_extent = 8192u;

  /** How the helper saw the text. Informative: the picture means the same either way. */
  enum source_e : std::uint32_t {
    source_none = 0u,
    source_direct2d = 1u,  ///< Direct2D text calls, repeated on the helper's own render target.
    source_gdi = 2u,  ///< GDI text calls, followed through the game's blits to its window.
    /// Text the game composed itself from GDI glyphs, followed from its texture through the
    /// game's Direct3D 11 draws to its back buffer.
    source_direct3d11 = 3u,
    /// The helper follows something it does not take for text, and reports no text: the host
    /// is to find the text in the picture.
    source_unsure = 4u,
  };

  /**
   * The text currently drawn. A writer makes `sequence` odd, updates the fields and the coverage
   * pixels, then makes it even; a reader retries while it is odd or changed. `magic` and
   * `version` are written last at start-up, once the helper has installed what it can.
   */
  struct snapshot_t {
    std::uint32_t magic;
    std::uint32_t version;
    volatile std::uint32_t sequence;
    std::uint32_t source;  ///< source_e of the text now in the picture.
    /// Size of the picture. Zero until the application's drawing surface is known; nonzero only
    /// while the coverage section of exactly this size exists.
    std::uint32_t surface_width;
    std::uint32_t surface_height;
    /// The rows of the picture that may hold text, [text_top, text_bottom); equal without text.
    /// Every text pixel lies at least two rows inside them, or they end at the picture's edge.
    std::uint32_t text_top;
    std::uint32_t text_bottom;
    /// The window whose client area the picture covers, as a handle value, or zero when the
    /// helper cannot tell (Direct2D draws on a surface, not on a window).
    std::uint64_t window;
    /// Creation time of the helper's process (FILETIME as one number). A section that outlived
    /// its process cannot pass for the helper of a later process with the same id.
    std::uint64_t process_created;
  };

  static_assert(sizeof(snapshot_t) == 48u);
}  // namespace sunshine_text_hook
