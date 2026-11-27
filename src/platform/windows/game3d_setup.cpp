// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_win32_util.h"
#include "game3d_setup.h"

#include "src/config.h"
#include "src/logging.h"
#include "src/system_tray.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iterator>
#include <map>
#include <nlohmann/json.hpp>
#include <thread>
#include <utility>

namespace platf::game3d {
  namespace detail {
    namespace {
      using win32::utf8;

      bool same_executable(const std::filesystem::path &a, const std::filesystem::path &b) {
        return win32::same_name(a.lexically_normal().native(), b.lexically_normal().native());
      }

      constexpr std::size_t max_decisions = 4096;
    }  // namespace

    setup_decisions::setup_decisions(std::filesystem::path file):
        file_(std::move(file)) {}

    void setup_decisions::load_locked() {
      if (loaded_) {
        return;
      }
      entries_.clear();
      std::error_code error;
      const bool present = !file_.empty() && std::filesystem::exists(file_, error);
      if (!error && !present) {
        loaded_ = true;
        unreadable_ = false;
        return;
      }
      try {
        if (error) {
          throw std::runtime_error(error.message());
        }
        const auto size = std::filesystem::file_size(file_, error);
        if (error || size > 1024 * 1024) {
          throw std::runtime_error("unexpected size");
        }
        std::ifstream input(file_, std::ios::binary);
        if (!input) {
          throw std::runtime_error("the file cannot be opened");
        }
        const std::string text {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        if (input.bad()) {
          throw std::runtime_error("the file cannot be read");
        }
        const auto document = nlohmann::json::parse(text);
        if (document.at("version").get<int>() != 1) {
          throw std::runtime_error("unsupported version");
        }
        std::vector<entry_t> entries;
        for (const auto &item : document.at("executables")) {
          const auto decision = item.at("decision").get<std::string>();
          entry_t entry {win32::path_from_utf8(item.at("path").get<std::string>()), item.value("identity", std::string {})};
          entry.chosen = item.value("chosen", false);
          if (decision == "install") {
            entry.decision = setup_decision_e::install;
          } else if (decision == "never") {
            entry.decision = setup_decision_e::never;
          }
          if (entry.decision == setup_decision_e::none || !entry.executable.is_absolute() || (entry.decision == setup_decision_e::install && entry.identity.empty()) || entries.size() >= max_decisions) {
            throw std::runtime_error("invalid entry");
          }
          entries.push_back(std::move(entry));
        }
        entries_ = std::move(entries);
        loaded_ = true;
        unreadable_ = false;
      } catch (const std::exception &failure) {
        // No choice is safer than a partial one: the executable is offered again and nothing
        // is written until the user confirms it again. The file is read again on the next use,
        // since another program may only hold it open for a moment.
        if (!unreadable_) {
          BOOST_LOG(warning) << "Game 3D setup: ignoring unreadable setup choices in \"" << utf8(file_) << "\": " << failure.what();
        }
        unreadable_ = true;
      }
    }

    bool setup_decisions::set_aside_locked() {
      SYSTEMTIME now {};
      GetLocalTime(&now);
      std::array<wchar_t, 64> suffix {};
      swprintf(suffix.data(), suffix.size(), L".bad-%04u%02u%02u-%02u%02u%02u-%03u", now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond, now.wMilliseconds);
      auto aside = file_;
      aside += suffix.data();
      // No replacement: a file this cannot move, for example one another program holds open,
      // keeps its choices and refuses the new one.
      if (MoveFileExW(file_.c_str(), aside.c_str(), MOVEFILE_WRITE_THROUGH)) {
        BOOST_LOG(warning) << "Game 3D setup: kept the unreadable setup choices as \"" << utf8(aside) << "\" and started a new file";
      } else if (GetLastError() != ERROR_FILE_NOT_FOUND) {
        return false;
      }
      entries_.clear();
      loaded_ = true;
      unreadable_ = false;
      return true;
    }

    bool setup_decisions::save_locked(const std::vector<entry_t> &entries) const {
      auto document = nlohmann::json {{"version", 1}, {"executables", nlohmann::json::array()}};
      for (const auto &entry : entries) {
        nlohmann::json item {
          {"path", utf8(entry.executable)},
          {"identity", entry.identity},
          {"decision", entry.decision == setup_decision_e::install ? "install" : "never"},
        };
        if (entry.chosen) {
          item["chosen"] = true;
        }
        document["executables"].push_back(std::move(item));
      }
      const auto text = document.dump(2);
      auto temporary = file_;
      temporary += L".tmp";
      const auto output = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (output == INVALID_HANDLE_VALUE) {
        return false;
      }
      // Durable before the rename: a power loss must not leave a replaced file without content.
      DWORD written = 0;
      const bool complete = WriteFile(output, text.data(), static_cast<DWORD>(text.size()), &written, nullptr) && written == text.size() && FlushFileBuffers(output);
      CloseHandle(output);
      if (!complete || !MoveFileExW(temporary.c_str(), file_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        DeleteFileW(temporary.c_str());
        return false;
      }
      return true;
    }

    setup_decision_e setup_decisions::lookup(const std::filesystem::path &executable, const std::string_view identity) {
      std::lock_guard lock(mutex_);
      load_locked();
      const auto entry = std::find_if(entries_.begin(), entries_.end(), [&](const auto &candidate) {
        return same_executable(candidate.executable, executable);
      });
      if (entry == entries_.end()) {
        return setup_decision_e::none;
      }
      if (entry->decision == setup_decision_e::install && (identity.empty() || entry->identity != identity)) {
        // The confirmed file changed (for example a game update): confirm it again.
        return setup_decision_e::none;
      }
      return entry->decision;
    }

    bool setup_decisions::chosen(const std::filesystem::path &executable) {
      std::lock_guard lock(mutex_);
      load_locked();
      return std::any_of(entries_.begin(), entries_.end(), [&](const auto &candidate) {
        return candidate.chosen && candidate.decision == setup_decision_e::install && same_executable(candidate.executable, executable);
      });
    }

    void choosable_applications::observe(const detected_game &game, std::string executable, std::string identity) {
      if (identity.empty()) {
        return;
      }
      forget(game.executable);
      entries_.insert(entries_.begin(), entry_t {game, std::move(executable), std::move(identity)});
      if (entries_.size() > most) {
        entries_.resize(most);
      }
    }

    void choosable_applications::forget(const std::filesystem::path &executable) {
      std::erase_if(entries_, [&](const auto &entry) {
        return same_executable(entry.game.executable, executable);
      });
    }

    void choosable_applications::keep_if(const std::function<bool(const detected_game &game)> &keep) {
      std::erase_if(entries_, [&](const auto &entry) {
        return !keep(entry.game);
      });
    }

    std::optional<choosable_applications::entry_t> choosable_applications::find(const std::string_view executable, const std::string_view identity) const {
      const auto entry = std::find_if(entries_.begin(), entries_.end(), [&](const auto &candidate) {
        return candidate.executable == executable && candidate.identity == identity;
      });
      return entry == entries_.end() ? std::nullopt : std::optional(*entry);
    }

    bool setup_decisions::record(const std::filesystem::path &executable, const std::string_view identity, const setup_decision_e decision, const bool chosen) {
      std::lock_guard lock(mutex_);
      load_locked();
      if (!executable.is_absolute() || (decision == setup_decision_e::install && identity.empty())) {
        return false;
      }
      // Never replace choices that could not be read with only this one.
      if (unreadable_ && !set_aside_locked()) {
        return false;
      }
      auto entries = entries_;
      std::erase_if(entries, [&](const auto &candidate) {
        return same_executable(candidate.executable, executable);
      });
      if (decision != setup_decision_e::none) {
        if (entries.size() >= max_decisions) {
          return false;
        }
        // Only a standing approval makes an application a game; a refusal needs no such mark.
        entries.push_back({executable.lexically_normal(), std::string {identity}, decision, chosen && decision == setup_decision_e::install});
      }
      if (!save_locked(entries)) {
        return false;
      }
      entries_ = std::move(entries);
      return true;
    }

    setup_outcome apply_setup(
      const std::filesystem::path &executable,
      setup_decisions &decisions,
      const std::function<void(const std::string &detail)> &progress,
      const std::filesystem::path &bundle_directory
    ) {
      setup_outcome outcome {inspect_installation(executable, bundle_directory)};
      auto &status = outcome.installation;
      if (status.state == installation_state_e::anti_cheat || status.state == installation_state_e::unavailable) {
        return outcome;
      }
      const bool writes = status.state == installation_state_e::missing_reshade || status.state == installation_state_e::missing_addon || status.state == installation_state_e::update_available;
      if (!writes) {
        return outcome;
      }
      outcome.identity = file_identity(status.executable);
      if (outcome.identity.empty()) {
        status.state = installation_state_e::unavailable;
        status.detail = "The game executable could not be identified, so Sunshine 3D does not change its folder.";
        return outcome;
      }
      const auto decision = decisions.lookup(status.executable, outcome.identity);
      if (decision == setup_decision_e::never) {
        outcome.state = published_state_e::declined;
        status.detail = "You chose not to set up this game, so Sunshine 3D does not change its folder. Choose Set up this game to change that.";
        return outcome;
      }
      // A first installation writes new files and settings into the game's folders, also where
      // the user already installed ReShade: the add-on installer disables ReShade's Generic
      // Depth add-on and enables Game 3D in ReShade.ini. It needs the user's confirmation.
      if (status.state == installation_state_e::missing_reshade || status.state == installation_state_e::missing_addon) {
        const bool fresh = status.state == installation_state_e::missing_reshade;
        if (decision != setup_decision_e::install) {
          outcome.state = published_state_e::ready_to_install;
          status.detail = fresh ? "Sunshine 3D can install ReShade with full add-on support and its Game 3D add-on beside this executable. Nothing is written until you confirm this game." :
                                  "ReShade is installed for this executable. Sunshine 3D can add its Game 3D add-on, which also disables ReShade's Generic Depth add-on and enables Game 3D in ReShade.ini. Nothing is written until you confirm this game.";
          return outcome;
        }
        progress(fresh ? "Installing ReShade with full add-on support and the Sunshine 3D add-on beside the game executable. Keep playing; setup takes effect on the next launch." : "Installing the Sunshine 3D add-on for the next launch. Your current game can stay open.");
        status = fresh ? install_reshade_and_addon_for_next_launch(status.executable, bundle_directory) : install_addon_for_next_launch(status.executable, bundle_directory);
        // The confirmation covered this installation. A failure keeps it, so the next launch
        // retries; success consumes it, so a later removal is offered again.
        if (status.state == installation_state_e::installed && !decisions.record(status.executable, {}, setup_decision_e::none)) {
          BOOST_LOG(warning) << "Game 3D setup: could not clear the completed install confirmation for \"" << utf8(status.executable) << '"';
        }
        return outcome;
      }
      // The Sunshine add-on is already installed, by a confirmed or a manual setup: keep it
      // matched to this host. A confirmation is no longer needed.
      if (decision == setup_decision_e::install && !decisions.record(status.executable, {}, setup_decision_e::none)) {
        BOOST_LOG(warning) << "Game 3D setup: could not clear a no longer needed install confirmation for \"" << utf8(status.executable) << '"';
      }
      status = update_existing_addon(status.executable, bundle_directory);
      return outcome;
    }

    setup_outcome apply_text_hook_setup(
      const std::filesystem::path &executable,
      setup_decisions &decisions,
      const std::function<std::string(const std::filesystem::path &executable)> &refusal,
      const std::function<text_hook::load_result()> &load,
      const bool helper
    ) {
      setup_outcome outcome;
      auto &status = outcome.installation;
      status.executable = executable;
      if (auto reason = helper ? refusal(executable) : std::string {}; !reason.empty()) {
        status.state = installation_state_e::anti_cheat;
        status.detail = std::move(reason);
        return outcome;
      }
      outcome.identity = file_identity(executable);
      if (outcome.identity.empty()) {
        status.state = installation_state_e::unavailable;
        status.detail = "The game executable could not be identified, so Sunshine 3D does not load anything into it.";
        return outcome;
      }
      const auto decision = decisions.lookup(executable, outcome.identity);
      if (decision == setup_decision_e::never) {
        outcome.state = published_state_e::declined;
        status.detail = "You chose not to convert this game, so Sunshine 3D does not load anything into it. Choose Convert this game to change that.";
        return outcome;
      }
      if (decision != setup_decision_e::install) {
        outcome.state = published_state_e::ready_to_install;
        status.detail = helper ? "This looks like a 2D text game. Sunshine 3D can convert its picture to 3D itself and load a small text helper into the game while it runs, so the game's own text stays readable. Nothing is written to the game's folder, and nothing is loaded until you confirm this game." :
                                 "This looks like a 2D text game that composes its text itself. Sunshine 3D can convert its picture to 3D itself and keep the text readable from the picture alone. Nothing is loaded into the game or written to its folder, and nothing is converted until you confirm this game.";
        return outcome;
      }
      const auto loaded = load();
      status.detail = loaded.detail;
      status.addon_present = loaded.state == text_hook::load_state_e::loaded;
      if (loaded.state == text_hook::load_state_e::not_running) {
        outcome.state = published_state_e::approved;
        return outcome;
      }
      // The approval is for converting the game. The helper is how the game's own text is found
      // exactly; without it the host still finds text in the picture.
      outcome.state = published_state_e::converting;
      outcome.convert = true;
      if (!status.addon_present) {
        outcome.retry_load = helper && loaded.state == text_hook::load_state_e::failed;
        status.detail += helper ? " The game is converted without it: its text is found in the picture alone." : " Its text is found in the picture alone.";
      }
      return outcome;
    }

    std::string_view published_state_name(const setup_outcome &outcome) {
      switch (outcome.state) {
        case published_state_e::installation:
          break;
        case published_state_e::ready_to_install:
          return "ready_to_install";
        case published_state_e::declined:
          return "declined";
        case published_state_e::installing:
          return "installing";
        case published_state_e::converting:
          return "converting";
        case published_state_e::approved:
          return "approved";
      }
      switch (outcome.installation.state) {
        case installation_state_e::missing_reshade:
          return "missing_reshade";
        case installation_state_e::missing_addon:
          return "missing_addon";
        case installation_state_e::current:
          return "current";
        case installation_state_e::update_available:
          return "update_available";
        case installation_state_e::update_pending:
          return "update_pending";
        case installation_state_e::updated:
          return "updated";
        case installation_state_e::installed:
          return "installed";
        case installation_state_e::anti_cheat:
          return "anti_cheat";
        case installation_state_e::unavailable:
          return "unavailable";
      }
      return "unavailable";
    }
  }  // namespace detail

  namespace {
    using namespace std::chrono_literals;
    using detail::published_state_e;

    using win32::utf8;

    std::filesystem::path decisions_file() {
      return win32::path_from_utf8(config::sunshine.config_file).parent_path() / "game3d_setup.json";
    }

    class setup_service {
    public:
      ~setup_service() {
        stop();
      }

      std::uint64_t watch(RECT rect) {
        std::lock_guard lock(mutex_);
        if (stopping_) {
          return 0;
        }
        const auto token = ++next_token_;
        watches_.emplace(token, rect);
        ++watch_revision_;
        start_locked();
        changed_ = true;
        wake_.notify_one();
        return token;
      }

      void unwatch(std::uint64_t token) {
        std::lock_guard lock(mutex_);
        watches_.erase(token);
        ++watch_revision_;
        changed_ = true;
        wake_.notify_one();
      }

      nlohmann::json status() {
        nlohmann::json result;
        std::vector<detail::choosable_applications::entry_t> choosable;
        {
          std::lock_guard lock(mutex_);
          result = status_;
          result["active"] = !watches_.empty();
          // What keeps the converted game's text readable, as the capture path found it within
          // the last seconds: the game's own text, or the fallback and why. Empty when the game
          // is not being converted now.
          if (status_game_) {
            const auto state = text_hook::protection(status_game_->first, status_game_->second);
            result["text_protection"] = state.protection == text_hook::protection_e::game_text ? "game" : state.protection == text_hook::protection_e::picture ? "fallback" :
                                                                                                                                                                 "";
            result["text_protection_detail"] = text_hook::describe(state);
          }
          choosable = choosable_.entries();
        }
        // The applications the owner may choose for conversion, as long as they run. The worker
        // drops the closed ones only while a stream is watched. A text game the card itself
        // shows, as one that was stopped is, is chosen again there.
        const bool shown_as_text_game = result.value("detected", false) && result.value("kind", std::string {}) == "text_hook";
        result["choosable"] = nlohmann::json::array();
        for (const auto &entry : choosable) {
          if (!(shown_as_text_game && result.value("executable", std::string {}) == entry.executable) && text_hook::running(entry.game.process_id, entry.game.creation_time)) {
            result["choosable"].push_back({{"game_name", entry.game.name}, {"executable", entry.executable}, {"identity", entry.identity}, {"text_helper", entry.game.text_helper}});
          }
        }
        return result;
      }

      // The owner's own choice of an application that no rule offers.
      std::optional<std::string> choose(std::string_view executable, std::string_view identity) {
        std::optional<detail::choosable_applications::entry_t> entry;
        {
          std::lock_guard lock(mutex_);
          if (stopping_) {
            return "Sunshine 3D is shutting down.";
          }
          // Only an application the host itself saw in front on the streamed display can be
          // chosen, and only while its file is the one listed.
          entry = choosable_.find(executable, identity);
        }
        if (!entry) {
          return "The application is no longer listed. Bring it to the front on the streamed display, refresh the page and choose it again.";
        }
        if (!decisions_.record(entry->game.executable, entry->identity, detail::setup_decision_e::install, true)) {
          return "The setup choice could not be saved in the host's configuration directory.";
        }
        BOOST_LOG(info) << "Game 3D setup: user chose to convert \"" << utf8(entry->game.executable) << "\", which no rule offers";
        std::lock_guard lock(mutex_);
        if (stopping_) {
          return std::nullopt;
        }
        choosable_.forget(entry->game.executable);
        // The worker applies the choice as it applies a confirmed offer.
        requests_.push_back(entry->game);
        start_locked();
        changed_ = true;
        wake_.notify_one();
        return std::nullopt;
      }

      std::optional<std::string> decide(std::string_view executable, std::string_view identity, std::string_view decision_name) {
        if (decision_name == "convert") {
          return choose(executable, identity);
        }
        const auto decision = decision_name == "install" ? detail::setup_decision_e::install :
                              decision_name == "never"   ? detail::setup_decision_e::never :
                                                           detail::setup_decision_e::none;
        if (decision == detail::setup_decision_e::none) {
          return "Choose install, never or convert.";
        }
        std::optional<offer_t> offer;
        {
          std::lock_guard lock(mutex_);
          if (stopping_) {
            return "Sunshine 3D is shutting down.";
          }
          // Only the executable the host itself detected and offered can be confirmed, and only
          // while its file is the one shown. A stale page or another path cannot authorize writes.
          if (!offered_ || offered_->published_executable != executable || offered_->identity != identity) {
            return "The offered game has changed. Refresh the page and confirm again.";
          }
          offer = offered_;
        }
        // An application its owner chose stays chosen when a changed file is confirmed again.
        if (!decisions_.record(offer->installation_executable, offer->identity, decision, offer->game.chosen)) {
          return "The setup choice could not be saved in the host's configuration directory.";
        }
        BOOST_LOG(info) << "Game 3D setup: user chose " << decision_name << " for \"" << utf8(offer->installation_executable) << '"';
        std::lock_guard lock(mutex_);
        if (stopping_) {
          return std::nullopt;
        }
        // The worker re-inspects the folder and applies the recorded choice: installation for
        // "install", the declined state for "never". It never runs on this request thread.
        requests_.push_back(offer->game);
        // One confirmation is one request; the worker publishes the resulting state.
        offered_.reset();
        start_locked();
        changed_ = true;
        wake_.notify_one();
        return std::nullopt;
      }

      void stop() {
        {
          std::lock_guard lock(mutex_);
          stopping_ = true;
          wake_.notify_one();
        }
        cancel_installation_work();
        if (worker_.joinable()) {
          worker_.join();
        }
      }

    private:
      struct pending_update {
        detected_game game;
        std::chrono::steady_clock::time_point retry_at {};
        int attempts {};
      };

      struct offer_t {
        detected_game game;
        std::string published_executable;
        std::filesystem::path installation_executable;
        std::string identity;
      };

      void start_locked() {
        if (!worker_.joinable()) {
          worker_ = std::thread([this]() {
            run();
          });
        }
      }

      // `announce` false updates the card alone: no log line and no tray notification.
      void publish(const detected_game &game, const detail::setup_outcome &outcome, const bool announce = true) {
        const auto &installation = outcome.installation;
        const auto executable = utf8(game.executable);
        const auto state = outcome.state;
        const std::string state_name {detail::published_state_name(outcome)};
        const bool converting = state == published_state_e::converting;
        // An add-on installation consumes its confirmation. A text game's confirmation stands
        // for every later start, so its active state stays an offer the user can withdraw.
        const bool offer = state == published_state_e::ready_to_install || state == published_state_e::declined || converting || state == published_state_e::approved;
        // Worker-owned. Only a running text game is shown as being converted.
        converting_ = converting ? std::optional(game) : std::nullopt;
        {
          std::lock_guard lock(mutex_);
          status_game_ = converting_ ? std::optional(std::pair {game.process_id, game.creation_time}) : std::nullopt;
          status_ = {
            {"detected", true},
            {"game_name", game.name},
            {"executable", executable},
            {"kind", game.kind == game_kind_e::text_hook ? "text_hook" : "addon"},
            {"text_helper", game.text_helper},
            {"chosen", game.chosen},
            {"state", state_name},
            {"detail", installation.detail},
            {"reshade_present", installation.reshade_present},
            {"install_script", utf8(bundled_addon_directory() / "install.ps1")},
            {"runtime_install_script", utf8(bundled_addon_directory() / "install-runtime.ps1")},
          };
          // Only the offer on display can be confirmed.
          if (offer) {
            status_["identity"] = outcome.identity;
            offered_ = offer_t {game, executable, installation.executable, outcome.identity};
          } else {
            offered_.reset();
          }
        }
        // Focus changes and encoder rebuilds must not repeat the same suggestion. An offer that
        // is still unanswered is made again when the game is started again: that is a new
        // occasion to decide, and "never" ends it.
        const auto notification_key = executable + "\n" + state_name +
                                      (state == published_state_e::ready_to_install ? "\n" + std::to_string(game.creation_time) : std::string {});
        if (!announce || announced_.contains(notification_key)) {
          return;
        }
        announced_.emplace(notification_key, true);
        BOOST_LOG(info) << "Game 3D setup: " << game.name << " [" << state_name << "] executable=\"" << executable
                        << "\" pid=" << game.process_id << " creation_time=" << game.creation_time << ": " << installation.detail;
        // The tray only points at the Web UI, where the user confirms; a notification click
        // never installs. Declined, anti-cheat, current and converting states have nothing to act on.
        const bool settled = state == published_state_e::installation && (installation.state == installation_state_e::current || installation.state == installation_state_e::anti_cheat);
        if (!settled && state != published_state_e::declined && !converting) {
          const auto target_name = game.name + " (" + utf8(game.executable.parent_path().filename()) + ")";
          system_tray::update_tray_game3d_setup(target_name, state == published_state_e::ready_to_install ? "Game 3D can be set up for this game; confirm or decline it in Sunshine 3D." : installation.detail);
        }
      }

      detail::setup_outcome apply(const detected_game &game, bool publish_progress) {
        if (game.kind == game_kind_e::text_hook) {
          auto outcome = detail::apply_text_hook_setup(game.executable, decisions_, injection_refusal, [&]() -> text_hook::load_result {
            if (!game.text_helper) {
              if (!text_hook::running(game.process_id, game.creation_time)) {
                return {text_hook::load_state_e::not_running, text_hook::approved_for_next_start};
              }
              return {text_hook::load_state_e::unavailable, "This game composes its text itself, so nothing is loaded into it."};
            }
            const auto result = text_hook::load(game.process_id, game.creation_time);
            // Every load is on record, not only the first of a host run.
            BOOST_LOG(info) << "Game 3D setup: text helper for " << game.name << " pid=" << game.process_id << ": " << result.detail;
            return result;
          },
                                                       game.text_helper);
          if (game.chosen && outcome.state == published_state_e::ready_to_install) {
            // Found by its path, which the owner chose; this file is not the one confirmed.
            outcome.installation.detail = "You chose to convert this application, and its file has changed since then. Nothing is loaded or converted until you confirm it again.";
          }
          // The capture path converts this process only while the owner's approval stands;
          // "never" withdraws it at once, for the running game too.
          text_hook::set_consent(game.process_id, game.creation_time, outcome.convert);
          return outcome;
        }
        return detail::apply_setup(game.executable, decisions_, [&](const std::string &detail) {
          if (publish_progress) {
            publish(game, progress_outcome(detail));
          }
        });
      }

      static detail::setup_outcome progress_outcome(const std::string &detail) {
        detail::setup_outcome outcome;
        outcome.installation.detail = detail;
        outcome.state = published_state_e::installing;
        return outcome;
      }

      static bool retry(const installation_state_e state) {
        return state == installation_state_e::update_pending || state == installation_state_e::update_available || state == installation_state_e::unavailable;
      }

      // A text helper that did not come up is tried twice more for this run of the game: the
      // game may only have been busy starting. An add-on update is retried by its own rules.
      void retry_text_helper(const detected_game &game, const detail::setup_outcome &outcome) {
        if (game.kind == game_kind_e::text_hook && outcome.retry_load) {
          pending_[game.executable] = {game, std::chrono::steady_clock::now() + 5s, 1};
        }
      }

      void tick(const std::optional<RECT> &capture_rect, std::vector<detected_game> requests) {
        // Explicit Web UI choices first, even after Game 3D or the stream has ended.
        for (const auto &game : requests) {
          const auto outcome = apply(game, true);
          retry_text_helper(game, outcome);
          publish(game, outcome);
        }
        // A text game that has closed is no longer being converted. Applying it once more
        // finds it gone, withdraws its consent and shows it as approved for its next start.
        if (converting_ && !text_hook::running(converting_->process_id, converting_->creation_time)) {
          const auto game = *converting_;
          converting_.reset();
          const auto outcome = apply(game, false);
          BOOST_LOG(info) << "Game 3D setup: " << game.name << " pid=" << game.process_id << " has closed.";
          // Closing a game is not an occasion to notify its owner about it.
          publish(game, outcome, false);
        }
        const auto now = std::chrono::steady_clock::now();
        // Once a detected installation needs an update, keep retrying after the
        // stream ends. The updater independently checks every process using the
        // executable and refuses a loaded or locked DLL.
        for (auto it = pending_.begin(); it != pending_.end();) {
          if (now < it->second.retry_at) {
            ++it;
            continue;
          }
          const bool selected = !last_game_ || last_game_->executable == it->second.game.executable;
          const auto outcome = apply(it->second.game, selected);
          // A background game's update must not replace another game's instructions.
          if (selected) {
            publish(it->second.game, outcome);
          }
          if (it->second.game.kind == game_kind_e::text_hook) {
            if (outcome.retry_load && ++it->second.attempts < 3) {
              it->second.retry_at = std::chrono::steady_clock::now() + 5s;
              ++it;
            } else {
              it = pending_.erase(it);
            }
          } else if (retry(outcome.installation.state) && outcome.state != published_state_e::ready_to_install && outcome.state != published_state_e::declined) {
            it->second.retry_at = std::chrono::steady_clock::now() + (outcome.installation.state == installation_state_e::unavailable ? 30s : 2s);
            ++it;
          } else {
            it = pending_.erase(it);
          }
        }
        if (!capture_rect) {
          confirmation_.reset();
          return;
        }
        // An application its owner chose is a text game wherever the rules say nothing of it,
        // or take it for an add-on game it is not. Any other application the rules do not take
        // for a text game is listed as one the owner may choose.
        const auto application = observe_foreground(*capture_rect);
        const bool chosen = application && application->choosable && decisions_.chosen(application->game.executable);
        const auto candidate = chosen ? std::optional(as_chosen(*application)) : application && application->recognised ? std::optional(application->game) :
                                                                                                                          std::nullopt;
        const auto listed = application && application->choosable && !chosen ? std::optional(as_chosen(*application)) : std::nullopt;
        const auto listed_identity = listed ? file_identity(listed->executable) : std::string {};
        std::vector<detail::choosable_applications::entry_t> known;
        {
          std::lock_guard lock(mutex_);
          known = choosable_.entries();
        }
        std::erase_if(known, [](const auto &entry) {
          return text_hook::running(entry.game.process_id, entry.game.creation_time);
        });
        {
          std::lock_guard lock(mutex_);
          for (const auto &closed : known) {
            choosable_.keep_if([&](const detected_game &game) {
              return game.process_id != closed.game.process_id || game.creation_time != closed.game.creation_time;
            });
          }
          if (listed) {
            choosable_.observe(*listed, utf8(listed->executable), listed_identity);
          } else if (chosen) {
            choosable_.forget(application->game.executable);
          }
        }
        const auto game = confirmation_.observe(candidate, *capture_rect, std::chrono::steady_clock::now());
        if (!game) {
          return;
        }
        if (last_game_ && last_game_->process_id == game->process_id && last_game_->creation_time == game->creation_time && last_game_->executable == game->executable) {
          return;
        }
        last_game_ = game;
        const auto outcome = apply(*game, true);
        // Only an existing installation's add-on update is retried automatically. A confirmed
        // fresh installation that failed retries on the game's next launch.
        if (game->kind == game_kind_e::text_hook) {
          retry_text_helper(*game, outcome);
        } else if (outcome.installation.state == installation_state_e::update_pending || outcome.installation.state == installation_state_e::update_available) {
          pending_[game->executable] = {*game, std::chrono::steady_clock::now() + 2s};
        }
        publish(*game, outcome);
      }

      void run() {
        std::unique_lock lock(mutex_);
        std::uint64_t observed_watch_revision = 0;
        while (!stopping_) {
          // A text game shown as being converted is watched until it closes, stream or not.
          if (watches_.empty() && pending_.empty() && requests_.empty() && !converting_) {
            wake_.wait(lock, [this]() {
              return stopping_ || !watches_.empty() || !requests_.empty();
            });
          }
          if (stopping_) {
            break;
          }
          // Reconfiguration can briefly overlap converters. The newest owns the
          // selected display; the refcount keeps monitoring through that overlap.
          const auto rect = watches_.empty() ? std::nullopt : std::optional<RECT>(watches_.rbegin()->second);
          if (observed_watch_revision != watch_revision_) {
            confirmation_.reset();
            observed_watch_revision = watch_revision_;
          }
          auto requests = std::exchange(requests_, {});
          changed_ = false;
          lock.unlock();
          try {
            tick(rect, std::move(requests));
          } catch (const std::exception &error) {
            BOOST_LOG(warning) << "Game 3D setup check failed: " << error.what();
          }
          lock.lock();
          wake_.wait_for(lock, 2s, [this]() {
            return stopping_ || changed_;
          });
        }
      }

      std::mutex mutex_;
      std::condition_variable wake_;
      std::thread worker_;
      bool stopping_ {};
      bool changed_ {};
      std::uint64_t next_token_ {};
      std::uint64_t watch_revision_ {};
      std::map<std::uint64_t, RECT> watches_;
      std::vector<detected_game> requests_;
      std::optional<offer_t> offered_;
      detail::choosable_applications choosable_;  ///< Guarded by mutex_.
      nlohmann::json status_ {{"detected", false}};
      std::optional<std::pair<std::uint32_t, std::uint64_t>> status_game_;  ///< The text game `status_` shows as converted.
      // Thread-safe; written by the request thread (choices) and the worker (completion).
      detail::setup_decisions decisions_ {decisions_file()};
      // Worker-owned state: no filesystem/process operation holds mutex_.
      std::optional<detected_game> last_game_;
      std::optional<detected_game> converting_;
      detection_confirmation confirmation_;
      std::map<std::filesystem::path, pending_update> pending_;
      std::map<std::string, bool> announced_;
    };

    setup_service &service() {
      static setup_service instance;
      return instance;
    }
  }  // namespace

  std::shared_ptr<void> watch_game_setup(RECT capture_rect) {
    const auto token = service().watch(capture_rect);
    return std::shared_ptr<void>(new std::uint64_t(token), [](void *value) {
      const auto token = static_cast<std::uint64_t *>(value);
      service().unwatch(*token);
      delete token;
    });
  }

  nlohmann::json setup_status() {
    return service().status();
  }

  std::optional<std::string> decide_setup(std::string_view executable, std::string_view identity, std::string_view decision) {
    return service().decide(executable, identity, decision);
  }

  void shutdown_setup() {
    service().stop();
  }
}  // namespace platf::game3d
