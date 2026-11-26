// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_detection.h"
#include "game3d_installation.h"
#include "game3d_text_hook.h"

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
  // "convert" is the owner's own choice of an application no rule offers: one of those the
  // status lists as lately in front on the streamed display, named by the path and identity
  // listed. It is converted as a text game from then on, at every start, until it is stopped.
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
    // choices, so it can only cause a new offer, never a write. It is read again on the next
    // use, and the next choice saved first moves it aside (game3d_setup.json.bad-<time>)
    // instead of replacing it; when it cannot be moved, that choice is not saved.
    // An "install" may carry the mark that the owner chose the executable where no rule offered
    // it. That mark is what makes the host take the application for a text game at all, so it
    // follows the executable path alone: after a game update the application is still found,
    // and the changed file is offered for confirmation like any other.
    class setup_decisions {
    public:
      explicit setup_decisions(std::filesystem::path file);

      [[nodiscard]] setup_decision_e lookup(const std::filesystem::path &executable, std::string_view identity);
      // True for an executable its owner chose for conversion and has not stopped since.
      [[nodiscard]] bool chosen(const std::filesystem::path &executable);
      // none removes the executable's choice. False when the file cannot be written.
      [[nodiscard]] bool record(const std::filesystem::path &executable, std::string_view identity, setup_decision_e decision, bool chosen = false);

    private:
      struct entry_t {
        std::filesystem::path executable;
        std::string identity;
        setup_decision_e decision {};
        bool chosen {};
      };

      void load_locked();
      [[nodiscard]] bool set_aside_locked();
      [[nodiscard]] bool save_locked(const std::vector<entry_t> &entries) const;

      std::mutex mutex_;
      std::filesystem::path file_;
      bool loaded_ {};
      bool unreadable_ {};
      std::vector<entry_t> entries_;
    };

    // The applications an owner may choose for conversion: those lately in front on the
    // streamed display that no rule offers. One entry per executable, the latest first, and no
    // more than `most`: a list to pick a game from, not a history. Not thread-safe.
    class choosable_applications {
    public:
      static constexpr std::size_t most = 4;

      struct entry_t {
        detected_game game;  ///< As it is once chosen.
        std::string executable;  ///< The path as the status lists it.
        std::string identity;  ///< Of the executable's file when it was last in front.
      };

      // An application is in front. One whose file cannot be identified is not listed.
      void observe(const detected_game &game, std::string executable, std::string identity);
      void forget(const std::filesystem::path &executable);
      void keep_if(const std::function<bool(const detected_game &game)> &keep);
      // The entry listed under exactly this path and identity.
      [[nodiscard]] std::optional<entry_t> find(std::string_view executable, std::string_view identity) const;

      [[nodiscard]] const std::vector<entry_t> &entries() const {
        return entries_;
      }

    private:
      std::vector<entry_t> entries_;
    };

    // What the setup card shows for a game: the installation's own state, the user's open
    // choice about it, or a confirmed installation in progress.
    enum class published_state_e {
      installation,  ///< The installation state, published under its own name.
      ready_to_install,  ///< Waits for the user's confirmation; nothing is written before it.
      declined,  ///< The user chose "never" for this executable.
      installing,  ///< A confirmed installation is running.
    };

    struct setup_outcome {
      installation_status installation;
      published_state_e state {published_state_e::installation};
      // Identity of the executable a ready_to_install or declined state offers.
      std::string identity;
      // Text game: the owner's approval stands and the game is running, so it is converted,
      // with the helper's text or, where the helper is not running, with the picture's alone.
      bool convert {};
      // Text game: the helper did not come up this time and loading it is worth another try.
      bool retry_load {};
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

    // One confirmed detection of a text game. Nothing is ever written to its folder: the text
    // helper is loaded into the running process and ends with it. It is loaded only after the
    // user confirmed this executable, and never where `refusal` names a reason, which is where
    // anti-cheat software was found or could not be ruled out. Unlike an add-on installation
    // the confirmation is kept, because the helper must be loaded again at every start of the
    // game; "never" withdraws it. A game the helper cannot be loaded into is converted all the
    // same: the helper only adds the game's own text to what the host finds in the picture.
    // A game that composes its text itself (`helper` false) has nothing the helper could watch:
    // nothing is loaded into it, so nothing refuses it, and `load` only tells whether it runs.
    [[nodiscard]] setup_outcome apply_text_hook_setup(
      const std::filesystem::path &executable,
      setup_decisions &decisions,
      const std::function<std::string(const std::filesystem::path &executable)> &refusal,
      const std::function<text_hook::load_result()> &load,
      bool helper = true
    );

    // The state name the Web UI's Game 3D setup card reads: an installation_state_e name,
    // "ready_to_install", "declined" or "installing".
    [[nodiscard]] std::string_view published_state_name(const setup_outcome &outcome);
  }  // namespace detail
}  // namespace platf::game3d
