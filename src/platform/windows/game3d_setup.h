// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_installation.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

namespace platf::game3d {
  // A live presentation owns a watch. Process discovery and all disk work run on a
  // separate worker; retiring an encoder never waits for an add-on file operation.
  [[nodiscard]] std::shared_ptr<void> watch_game_setup(RECT capture_rect);

  // Read-only, authenticated Web UI status. Retains the most recently detected
  // game so opening the browser does not discard the setup instructions.
  [[nodiscard]] nlohmann::json setup_status();

  // Authenticated Web UI choice for the executable the status currently offers: "install"
  // or "never". The executable path and identity must match that offer exactly. The choice
  // persists across host restarts; installation itself runs on the setup worker.
  // Returns an error message when the offer no longer matches or the choice cannot be saved.
  [[nodiscard]] std::optional<std::string> decide_setup(std::string_view executable, std::string_view identity, std::string_view decision);

  // Join before the host's logging and tray services stop. Pending updates survive
  // mode switches/disconnects within this host process, but never host shutdown.
  void shutdown_setup();

  namespace detail {
    enum class setup_decision_e {
      none,
      install,
      never,
    };

    // The user's explicit per-executable choices, in one JSON file in the host's config
    // directory. "never" applies to the executable path; "install" applies only while the
    // executable's file identity matches the one confirmed. An unreadable file holds no
    // choices, so it can only cause a new offer, never a write.
    class setup_decisions {
    public:
      explicit setup_decisions(std::filesystem::path file);

      [[nodiscard]] setup_decision_e lookup(const std::filesystem::path &executable, std::string_view identity);
      // none removes the executable's choice. False when the file cannot be written.
      [[nodiscard]] bool record(const std::filesystem::path &executable, std::string_view identity, setup_decision_e decision);

    private:
      struct entry_t {
        std::filesystem::path executable;
        std::string identity;
        setup_decision_e decision {};
      };

      void load_locked();
      [[nodiscard]] bool save_locked(const std::vector<entry_t> &entries) const;

      std::mutex mutex_;
      std::filesystem::path file_;
      bool loaded_ {};
      std::vector<entry_t> entries_;
    };

    struct setup_outcome {
      installation_status installation;
      // Published state: an installation_state_e name, or ready_to_install / declined.
      std::string state;
      // Identity of the executable a ready_to_install or declined state offers.
      std::string identity;
    };

    // One confirmed detection. Writes only what the folder's state and the user's choice
    // permit: nothing for anti-cheat, an unconfirmed executable without the Sunshine add-on
    // (ready_to_install) or a declined executable; after an install confirmation, ReShade plus
    // the add-on in a fresh folder, or only the add-on where ReShade already exists; and an
    // update of an add-on that is already installed. Never creates a proxy otherwise.
    [[nodiscard]] setup_outcome apply_setup(
      const std::filesystem::path &executable,
      setup_decisions &decisions,
      const std::function<void(const std::string &detail)> &progress,
      const std::filesystem::path &bundle_directory = bundled_addon_directory()
    );
  }  // namespace detail
}  // namespace platf::game3d
