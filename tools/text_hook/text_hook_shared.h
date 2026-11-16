// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook_shared.h
 * @brief Shared-memory contract between the in-application text helper and the Game 3D host.
 *
 * The helper runs inside a 2D text game its owner confirmed for Game 3D (docs/reshade-sbs.md, Text
 * games). It reports where the game draws text and a picture of that text, both in the pixels of
 * the surface the game draws on. It never reports the text itself. The layout is identical for
 * 32-bit and 64-bit builds.
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
  inline constexpr std::uint32_t magic = 0x32485453u;  // "STH2"
  inline constexpr std::uint32_t version = 3u;
  inline constexpr std::uint32_t max_boxes = 512u;

  /** One drawn piece of text, in pixels of the drawing surface. Half-open: [left, right). */
  struct box_t {
    float left;
    float top;
    float right;
    float bottom;
  };

  /**
   * The text currently drawn. A writer makes `sequence` odd, updates the fields and the coverage
   * pixels, then makes it even; a reader retries while it is odd or changed. `generation`
   * increases whenever the text is replaced rather than extended, and never returns to zero.
   */
  struct snapshot_t {
    std::uint32_t magic;
    std::uint32_t version;
    volatile std::uint32_t sequence;
    std::uint32_t generation;
    std::uint64_t change_tick_ms;  ///< GetTickCount64() of the last change.
    std::uint32_t surface_width;  ///< Zero until the application's drawing surface is known.
    std::uint32_t surface_height;
    std::uint32_t box_count;
    std::uint32_t coverage_available;  ///< Nonzero once the coverage section exists for this surface size.
    box_t boxes[max_boxes];
  };

  static_assert(sizeof(box_t) == 16u);
  static_assert(sizeof(snapshot_t) == 40u + 16u * max_boxes);
}  // namespace sunshine_text_hook
