// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <windows.h>

namespace platf::game3d {
  // A live presentation owns a watch. Process discovery and all disk work run on a
  // separate worker; retiring an encoder never waits for an add-on file operation.
  [[nodiscard]] std::shared_ptr<void> watch_game_setup(RECT capture_rect);

  // Read-only, authenticated Web UI status. Retains the most recently detected
  // game so opening the browser does not discard the setup instructions.
  [[nodiscard]] nlohmann::json setup_status();

  // Join before the host's logging and tray services stop. Pending updates survive
  // mode switches/disconnects within this host process, but never host shutdown.
  void shutdown_setup();
}  // namespace platf::game3d
