// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook_core.h
 * @brief What the parts of the in-application text helper share: the lock, the published
 * picture and its publication.
 *
 * The helper is linked into another application without a C++ runtime library: plain arrays,
 * no exceptions, no allocation through `new`.
 */
#pragma once

#include "text_hook_shared.h"

#include <windows.h>

namespace sunshine_text_hook::helper {
  /// Text drawn in many calls is published as one piece: when the drawing it belongs to ends, or
  /// after this long, whichever comes first.
  inline constexpr std::uint64_t longest_batch_ms = 50u;

  template<class T>
  constexpr T min_of(const T a, const T b) {
    return a < b ? a : b;
  }

  template<class T>
  constexpr T max_of(const T a, const T b) {
    return a > b ? a : b;
  }

  /// Guards everything below and all state of the Direct2D and GDI parts. Recursive.
  extern CRITICAL_SECTION state_lock;

  /** One line into the diagnostic log, which exists only where its owner asked for it. */
  void log_line(const char *format, ...);
  /** True where there is such a log, for work that is done only to fill it. */
  bool logging();

  /**
   * Per thread: true inside an application's text call the helper is forwarding, and inside the
   * helper's own drawing. A hooked function reached from there is the same text once more, or
   * the helper's own, so only a call that is not nested is observed.
   */
  bool nested();
  void enter_call();
  void leave_call();

  /**
   * The published picture: premultiplied BGRA, top row first, shared with the host. Each part
   * draws on a picture of its own and publishes from it, so the host never sees one half drawn.
   */
  struct surface_t {
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    const std::uint32_t *pixels = nullptr;  ///< Null while there is no picture.
  };
  extern surface_t surface;

  /**
   * Make the published picture this size. A new size starts empty and is announced to the host
   * at once.
   * @return false when there is no picture of this size, now or after an earlier failure.
   */
  bool ensure_surface(std::uint32_t width, std::uint32_t height);

  /**
   * Publish the rows [top, bottom) of `pixels`, a picture of the published size, and the rows
   * that hold text. The host is told nothing when nothing differs from what it has.
   */
  void publish(const std::uint32_t *pixels, std::uint32_t top, std::uint32_t bottom, source_e source, std::uint32_t text_top, std::uint32_t text_bottom, HWND window);

  namespace d2d {
    /// Patch the Direct2D render-target vtables. False while the application has not loaded
    /// Direct2D and DirectWrite; worth calling again later.
    bool install();
    /// True once the application has drawn text with Direct2D: the picture is then Direct2D's.
    bool active();
    /// Publish a batch that has seen no text draw for a while.
    void idle(std::uint64_t tick);
  }  // namespace d2d

  namespace d3d11 {
    /// Patch the Direct3D 11 vtables. False while the application has not loaded Direct3D 11;
    /// worth calling again later.
    bool install();
    /// The application took a glyph's bitmap or outline from GDI: text is being composed.
    void glyphs_taken();
  }  // namespace d3d11

  namespace gdi {
    /// Point the GDI imports of the application's modules at the hooks. Modules loaded since the
    /// last call are covered by calling it again.
    void install();
    /// Publish changes to the window's text that no present-like call has published.
    void idle(std::uint64_t tick);
    /// Direct2D text has appeared: forget everything followed so far.
    void retire();
  }  // namespace gdi
}  // namespace sunshine_text_hook::helper
