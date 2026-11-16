// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file src/platform/windows/game3d_text_hook.h
 * @brief Host side of the in-application text hook for Game 3D text games.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace platf::game3d::text_hook {
  enum class load_state_e {
    loaded,  ///< The hook is running in the process and publishing.
    not_running,  ///< The process has ended; the hook loads when the game is started again.
    unavailable,  ///< The bundled hook for this process is missing or the process cannot be opened.
    failed,  ///< The injector ran and the hook did not come up.
  };

  struct load_result {
    load_state_e state {load_state_e::unavailable};
    std::string detail;
  };

  // Always resolves beside the host executable, independent of its working directory.
  [[nodiscard]] std::filesystem::path bundled_directory();

  /**
   * The owner's standing decision for one running game, as the setup worker last applied it.
   * The capture path converts a game only while this is true: a helper section alone proves
   * neither that the owner approved the game nor that the approval still stands.
   */
  void set_consent(std::uint32_t process_id, std::uint64_t creation_time, bool consented);
  [[nodiscard]] bool consented(std::uint32_t process_id, std::uint64_t creation_time);

  /** True when the hook's state section for this process exists and carries the current contract. */
  [[nodiscard]] bool loaded(std::uint32_t process_id);

  /**
   * Load the hook into one running 32-bit process. Nothing is written to the game's folder: the
   * hook lives only in that process and ends with it, so this runs again at every game start.
   * The caller owns the user's consent and the anti-cheat refusal. Worker thread only.
   */
  [[nodiscard]] load_result load(std::uint32_t process_id, const std::filesystem::path &bundle_directory = bundled_directory());

  /** Where the text is on the game's drawing surface. Empty (bottom <= top) without text. */
  struct text_bounds_t {
    float left = 0.0f, top = 0.0f, right = 0.0f, bottom = 0.0f;  ///< Around every piece of text.
    float line_height = 0.0f;  ///< Height of the shortest piece: about one line of text.

    [[nodiscard]] bool empty() const {
      return !(bottom > top) || !(right > left);
    }
  };

  /** The text on the game's drawing surface, and the picture of that text. */
  struct text_t {
    std::uint32_t sequence = 0u;  ///< Changes whenever the text or its picture changes.
    std::uint32_t surface_width = 0u;
    std::uint32_t surface_height = 0u;
    text_bounds_t bounds;
    std::uint32_t top = 0u;  ///< First surface row with text.
    std::uint32_t bottom = 0u;  ///< One past the last surface row with text; equal to top without text.
    /// surface_width * surface_height BGRA pixels, top row first; alpha is text coverage. The
    /// game keeps drawing into them, so a copy is only as consistent as `still_current` says.
    const std::uint8_t *pixels = nullptr;
  };

  /**
   * Reads what one game's text helper publishes. It only opens memory the helper shares; it never
   * waits for the game, so a paused or closing game cannot hold the stream.
   */
  class reader_t {
  public:
    reader_t() = default;
    reader_t(const reader_t &) = delete;
    reader_t &operator=(const reader_t &) = delete;
    ~reader_t();

    /**
     * The text of this process now. While the game is in the middle of drawing, which a line
     * typed out glyph by glyph is most of the time, this is the last complete state instead.
     */
    [[nodiscard]] bool read(std::uint32_t process_id, text_t &text);
    /** True when nothing changed since `text` was read, so pixels copied meanwhile belong to it. */
    [[nodiscard]] bool still_current(const text_t &text) const;
    void close();

  private:
    std::uint32_t process_id_ = 0u;
    void *state_section_ = nullptr;
    const void *state_ = nullptr;
    void *coverage_section_ = nullptr;
    const void *coverage_ = nullptr;
    std::uint32_t coverage_width_ = 0u;
    std::uint32_t coverage_height_ = 0u;
    text_t last_;
  };

  namespace detail {
    /**
     * The rectangle around the helper's boxes (left, top, right, bottom each), clipped to the
     * surface. A damaged or empty box, or one off the surface, is skipped.
     */
    [[nodiscard]] text_bounds_t text_bounds(const float *boxes, std::uint32_t box_count, std::uint32_t surface_width, std::uint32_t surface_height);

    /** Injector exit codes of tools/text_hook/text_hook_inject.cpp. */
    [[nodiscard]] std::string injector_failure(std::uint32_t exit_code);

    /**
     * The decision around one load, without process access: skip a loaded hook, run the injector
     * once, then require the hook to report in.
     */
    [[nodiscard]] load_result load_with(
      const std::function<bool()> &is_loaded,
      const std::function<bool()> &bundle_present,
      const std::function<std::uint32_t()> &run_injector,
      const std::function<bool()> &process_running = [] {
        return true;
      }
    );
  }  // namespace detail
}  // namespace platf::game3d::text_hook
