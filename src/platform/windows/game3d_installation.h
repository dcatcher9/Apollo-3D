// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <filesystem>
#include <string>

namespace platf::game3d {
  enum class installation_state_e {
    missing_reshade,
    missing_addon,
    current,
    update_available,
    update_pending,
    updated,
    installed,
    unavailable,
  };

  struct installation_status {
    installation_state_e state {installation_state_e::unavailable};
    std::filesystem::path executable;
    std::filesystem::path addon;
    std::string detail;
    bool reshade_present {};
    bool addon_present {};
  };

  // Always resolves beside the host executable, independent of its working directory.
  std::filesystem::path bundled_addon_directory();

  // Host shutdown only. Cancels a running installer and prevents further work.
  void cancel_installation_work();

  // Disk evidence only: the proxy's ReShade product identity plus ReShade.ini do
  // not prove that the full-add-on runtime loaded into the running game.
  installation_status inspect_installation(
    const std::filesystem::path &game_executable,
    const std::filesystem::path &bundle_directory = bundled_addon_directory()
  );

  // Replaces only an already installed add-on, never ReShade, presets or preferences.
  // Defers while the game is running, process state cannot be verified, or files are locked.
  // A successful atomic replacement retains the old DLL at SunshineSBS.addon64.previous.
  installation_status update_existing_addon(
    const std::filesystem::path &game_executable,
    const std::filesystem::path &bundle_directory = bundled_addon_directory()
  );

  // Installs only a missing Sunshine add-on through the bundled installer. An
  // already running game keeps its current modules until its next launch.
  // ReShade's runtime/configuration must already exist; existing DLLs are never
  // replaced through this path.
  installation_status install_addon_for_next_launch(
    const std::filesystem::path &game_executable,
    const std::filesystem::path &bundle_directory = bundled_addon_directory()
  );

  // Runs the pinned, bundled ReShade runtime installer. It refuses foreign proxy
  // DLLs and prepares a fresh runtime for the game's next launch.
  installation_status install_reshade_for_next_launch(
    const std::filesystem::path &game_executable,
    const std::filesystem::path &bundle_directory = bundled_addon_directory()
  );
}  // namespace platf::game3d
