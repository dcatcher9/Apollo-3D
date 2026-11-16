/**
 * @file src/platform/windows/host3d_subtitle_apps.h
 * @brief Owner-selected applications whose bottom text Host 3D protects as subtitles.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace platf::host3d_subtitle_apps {
  /** Configuration key holding a JSON array of enabled executable paths. */
  inline constexpr std::string_view config_key = "sbs_3d_subtitle_apps";

  struct app_t {
    std::string executable;  ///< Absolute path, UTF-8, forward slashes.
    std::string name;  ///< File name for display.
    bool enabled = false;
    bool recent = false;  ///< Seen in front of a Host 3D stream since the host started.

    bool operator==(const app_t &) const = default;
  };

  /**
   * Pure selection state. Protection is off for every application until its owner enables that
   * executable; nothing is inferred from the application itself. Recently observed foreground
   * applications are remembered only so the owner can pick one, and are never persisted.
   */
  class selection_t {
  public:
    static constexpr std::size_t max_enabled = 64u;
    static constexpr std::size_t max_recent = 12u;

    /** Replace the enabled set from a JSON array of paths. Invalid input or entries are dropped. */
    void load(std::string_view json_array);
    /** JSON array of the enabled paths, in the order they were enabled. */
    [[nodiscard]] std::string serialize() const;
    /** False when the path is not an absolute .exe path or the enabled set is full. */
    bool set_enabled(std::string_view executable, bool enabled);
    /** Remember a foreground application and report whether its owner enabled it. */
    bool observe(std::string_view executable);
    [[nodiscard]] bool enabled(std::string_view executable) const;
    /** Enabled applications first, then other recent ones, most recent first. */
    [[nodiscard]] std::vector<app_t> list() const;

  private:
    std::vector<std::string> enabled_;
    std::vector<std::string> recent_;
  };

  /** Load the enabled set from the configuration value read at start. */
  void load(std::string_view json_array);
  [[nodiscard]] std::string serialize();
  [[nodiscard]] std::vector<app_t> list();
  /** Change one application for the running host at once. The caller persists `serialize()`. */
  bool set_enabled(std::string_view executable, bool enabled);
  /**
   * Resolve the current executable of a validated foreground process, remember it as recent and
   * report whether it is enabled. The caller owns foreground HWND/PID validation; no PID is cached.
   */
  [[nodiscard]] bool observe_process(std::uint32_t process_id) noexcept;
}  // namespace platf::host3d_subtitle_apps
