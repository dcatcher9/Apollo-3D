// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file src/platform/windows/game3d_text_hook.h
 * @brief Host side of the in-application text helper for Game 3D text games.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace platf::game3d::text_hook {
  enum class load_state_e {
    loaded,  ///< The helper is running in the process and publishing.
    not_running,  ///< The process has ended; the helper loads when the game is started again.
    unavailable,  ///< This build has no helper for a game of this kind.
    failed,  ///< The loader ran and the helper did not come up.
  };

  struct load_result {
    load_state_e state {load_state_e::unavailable};
    std::string detail;
  };

  // Always resolves beside the host executable, independent of its working directory.
  [[nodiscard]] std::filesystem::path bundled_directory();

  /**
   * The owner's standing decision for one running game, as the setup worker last applied it.
   * The capture path converts a game only while this is true. A helper section proves neither
   * that the owner approved the game nor that the approval still stands, and a game the helper
   * could not be loaded into is still one the owner asked to have converted.
   */
  void set_consent(std::uint32_t process_id, std::uint64_t creation_time, bool consented);
  [[nodiscard]] bool consented(std::uint32_t process_id, std::uint64_t creation_time);

  /**
   * True when the helper's state section for this process exists, carries the current contract
   * and was written by the process created at this time.
   */
  [[nodiscard]] bool loaded(std::uint32_t process_id, std::uint64_t creation_time);

  /**
   * Load the helper into one running process, 32-bit or 64-bit. Nothing is written to the game's
   * folder: the helper lives only in that process and ends with it, so this runs again at every
   * game start. The caller owns the user's consent and the anti-cheat refusal. Worker thread only.
   */
  [[nodiscard]] load_result load(std::uint32_t process_id, std::uint64_t creation_time, const std::filesystem::path &bundle_directory = bundled_directory());

  /** The text on the game's drawing surface, as a picture of it. */
  struct text_t {
    std::uint32_t sequence = 0u;  ///< Changes whenever the text or its picture changes.
    std::uint32_t surface_width = 0u;
    std::uint32_t surface_height = 0u;
    std::uint32_t top = 0u;  ///< First surface row that may hold text.
    std::uint32_t bottom = 0u;  ///< One past the last such row; equal to top without text.
    /// The window whose client area the surface is, or zero when the helper cannot tell.
    std::uint64_t window = 0u;
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
     * The text of this process now. While the helper is in the middle of publishing, this is
     * the last complete state instead. False without a helper of this very process.
     */
    [[nodiscard]] bool read(std::uint32_t process_id, std::uint64_t creation_time, text_t &text);
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
    /** Loader exit codes of tools/text_hook/text_hook_inject.cpp. */
    [[nodiscard]] std::string injector_failure(std::uint32_t exit_code);

    /**
     * The decision around one load, without process access: skip a loaded helper, run the loader
     * once, then require the helper to report in. `bits` is the game's, for the messages.
     */
    [[nodiscard]] load_result load_with(
      unsigned bits,
      const std::function<bool()> &is_loaded,
      const std::function<bool()> &bundle_present,
      const std::function<std::uint32_t()> &run_injector,
      const std::function<bool()> &process_running = [] {
        return true;
      }
    );
  }  // namespace detail
}  // namespace platf::game3d::text_hook
