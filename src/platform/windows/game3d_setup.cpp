// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_setup.h"

#include "game3d_detection.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/system_tray.h"

#include <algorithm>
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
      std::string utf8(const std::filesystem::path &path) {
        const auto value = path.u8string();
        return {value.begin(), value.end()};
      }

      std::filesystem::path from_utf8(const std::string &value) {
        return std::filesystem::path {std::u8string(value.begin(), value.end())};
      }

      bool same_executable(const std::filesystem::path &a, const std::filesystem::path &b) {
        const auto left = a.lexically_normal().wstring();
        const auto right = b.lexically_normal().wstring();
        return CompareStringOrdinal(left.c_str(), static_cast<int>(left.size()), right.c_str(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL;
      }

      constexpr std::size_t max_decisions = 4096;
    }  // namespace

    setup_decisions::setup_decisions(std::filesystem::path file):
        file_(std::move(file)) {}

    void setup_decisions::load_locked() {
      if (loaded_) {
        return;
      }
      loaded_ = true;
      entries_.clear();
      std::error_code error;
      if (file_.empty() || !std::filesystem::exists(file_, error)) {
        return;
      }
      try {
        const auto size = std::filesystem::file_size(file_, error);
        if (error || size > 1024 * 1024) {
          throw std::runtime_error("unexpected size");
        }
        std::ifstream input(file_, std::ios::binary);
        const auto document = nlohmann::json::parse(std::string {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()});
        if (document.at("version").get<int>() != 1) {
          throw std::runtime_error("unsupported version");
        }
        for (const auto &item : document.at("executables")) {
          const auto decision = item.at("decision").get<std::string>();
          entry_t entry {from_utf8(item.at("path").get<std::string>()), item.value("identity", std::string {})};
          if (decision == "install") {
            entry.decision = setup_decision_e::install;
          } else if (decision == "never") {
            entry.decision = setup_decision_e::never;
          }
          if (entry.decision == setup_decision_e::none || !entry.executable.is_absolute() || (entry.decision == setup_decision_e::install && entry.identity.empty()) || entries_.size() >= max_decisions) {
            throw std::runtime_error("invalid entry");
          }
          entries_.push_back(std::move(entry));
        }
      } catch (const std::exception &failure) {
        // No choice is safer than a partial one: the executable is offered again and nothing
        // is written until the user confirms it again.
        entries_.clear();
        BOOST_LOG(warning) << "Game 3D setup: ignoring unreadable setup choices in \"" << utf8(file_) << "\": " << failure.what();
      }
    }

    bool setup_decisions::save_locked(const std::vector<entry_t> &entries) const {
      auto document = nlohmann::json {{"version", 1}, {"executables", nlohmann::json::array()}};
      for (const auto &entry : entries) {
        document["executables"].push_back({
          {"path", utf8(entry.executable)},
          {"identity", entry.identity},
          {"decision", entry.decision == setup_decision_e::install ? "install" : "never"},
        });
      }
      auto temporary = file_;
      temporary += L".tmp";
      {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output << document.dump(2);
        output.flush();
        if (!output) {
          output.close();
          std::error_code ignored;
          std::filesystem::remove(temporary, ignored);
          return false;
        }
      }
      if (!MoveFileExW(temporary.c_str(), file_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
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

    bool setup_decisions::record(const std::filesystem::path &executable, const std::string_view identity, const setup_decision_e decision) {
      std::lock_guard lock(mutex_);
      load_locked();
      if (!executable.is_absolute() || (decision == setup_decision_e::install && identity.empty())) {
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
        entries.push_back({executable.lexically_normal(), std::string {identity}, decision});
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
        outcome.state = "declined";
        status.detail = "You chose not to set up this game, so Sunshine 3D does not change its folder. Choose Set up this game to change that.";
        return outcome;
      }
      if (status.state == installation_state_e::missing_reshade) {
        if (decision != setup_decision_e::install) {
          outcome.state = "ready_to_install";
          status.detail = "Sunshine 3D can install ReShade with full add-on support and its Game 3D add-on beside this executable. Nothing is written until you confirm this game.";
          return outcome;
        }
        progress("Installing ReShade with full add-on support and the Sunshine 3D add-on beside the game executable. Keep playing; setup takes effect on the next launch.");
        status = install_reshade_and_addon_for_next_launch(status.executable, bundle_directory);
        // The confirmation covered this installation. A failure keeps it, so the next launch
        // retries; success consumes it, so a later removal of ReShade is offered again.
        if (status.state == installation_state_e::installed && !decisions.record(status.executable, {}, setup_decision_e::none)) {
          BOOST_LOG(warning) << "Game 3D setup: could not clear the completed install confirmation for \"" << utf8(status.executable) << '"';
        }
        return outcome;
      }
      // ReShade already exists in this folder: keep the Sunshine add-on present and matched. A
      // confirmation for the runtime is no longer needed once a runtime is present.
      if (decision == setup_decision_e::install && !decisions.record(status.executable, {}, setup_decision_e::none)) {
        BOOST_LOG(warning) << "Game 3D setup: could not clear a no longer needed install confirmation for \"" << utf8(status.executable) << '"';
      }
      if (status.state == installation_state_e::missing_addon) {
        progress("Installing the Sunshine 3D add-on for the next launch. Your current game can stay open.");
        status = install_addon_for_next_launch(status.executable, bundle_directory);
        return outcome;
      }
      status = update_existing_addon(status.executable, bundle_directory);
      return outcome;
    }
  }  // namespace detail

  namespace {
    using namespace std::chrono_literals;

    std::string utf8(const std::filesystem::path &path) {
      const auto value = path.u8string();
      return {value.begin(), value.end()};
    }

    const char *state_name(installation_state_e state) {
      switch (state) {
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

    std::filesystem::path decisions_file() {
      const auto &config_file = config::sunshine.config_file;
      return std::filesystem::path {std::u8string(config_file.begin(), config_file.end())}.parent_path() / "game3d_setup.json";
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
        std::lock_guard lock(mutex_);
        auto result = status_;
        result["active"] = !watches_.empty();
        return result;
      }

      std::optional<std::string> decide(std::string_view executable, std::string_view identity, std::string_view decision_name) {
        const auto decision = decision_name == "install" ? detail::setup_decision_e::install :
                              decision_name == "never"   ? detail::setup_decision_e::never :
                                                           detail::setup_decision_e::none;
        if (decision == detail::setup_decision_e::none) {
          return "Choose install or never.";
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
        if (!decisions_.record(offer->installation_executable, offer->identity, decision)) {
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

      void publish(const detected_game &game, const detail::setup_outcome &outcome, const std::string &state) {
        const auto &installation = outcome.installation;
        const auto executable = utf8(game.executable);
        const bool offer = state == "ready_to_install" || state == "declined";
        {
          std::lock_guard lock(mutex_);
          status_ = {
            {"detected", true},
            {"game_name", game.name},
            {"executable", executable},
            {"state", state},
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
        // Focus changes and encoder rebuilds must not repeat the same suggestion.
        const auto notification_key = executable + "\n" + state;
        if (announced_.contains(notification_key)) {
          return;
        }
        announced_.emplace(notification_key, true);
        BOOST_LOG(info) << "Game 3D setup: " << game.name << " [" << state << "] executable=\"" << executable
                        << "\" pid=" << game.process_id << " creation_time=" << game.creation_time << ": " << installation.detail;
#if defined(SUNSHINE_TRAY) && SUNSHINE_TRAY >= 1
        // The tray only points at the Web UI, where the user confirms; a notification click
        // never installs. Declined and anti-cheat states have nothing to act on.
        if (installation.state != installation_state_e::current && installation.state != installation_state_e::anti_cheat && state != "declined") {
          const auto target_name = game.name + " (" + utf8(game.executable.parent_path().filename()) + ")";
          system_tray::update_tray_game3d_setup(target_name, state == "ready_to_install" ? "Game 3D can be set up for this game; confirm or decline it in Sunshine 3D." : installation.detail);
        }
#endif
      }

      detail::setup_outcome apply(const detected_game &game, bool publish_progress) {
        auto outcome = detail::apply_setup(game.executable, decisions_, [&](const std::string &detail) {
          if (publish_progress) {
            publish(game, progress_outcome(detail), "installing");
          }
        });
        if (outcome.state.empty()) {
          outcome.state = state_name(outcome.installation.state);
        }
        return outcome;
      }

      static detail::setup_outcome progress_outcome(const std::string &detail) {
        detail::setup_outcome outcome;
        outcome.installation.detail = detail;
        return outcome;
      }

      static bool retry(const installation_state_e state) {
        return state == installation_state_e::update_pending || state == installation_state_e::update_available || state == installation_state_e::unavailable;
      }

      void tick(const std::optional<RECT> &capture_rect, std::vector<detected_game> requests) {
        // Explicit Web UI choices first, even after Game 3D or the stream has ended.
        for (const auto &game : requests) {
          const auto outcome = apply(game, true);
          publish(game, outcome, outcome.state);
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
            publish(it->second.game, outcome, outcome.state);
          }
          if (retry(outcome.installation.state) && outcome.state != "ready_to_install" && outcome.state != "declined") {
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
        const auto game = confirmation_.observe(detect_game(*capture_rect), *capture_rect, std::chrono::steady_clock::now());
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
        if (outcome.installation.state == installation_state_e::update_pending || outcome.installation.state == installation_state_e::update_available) {
          pending_[game->executable] = {*game, std::chrono::steady_clock::now() + 2s};
        }
        publish(*game, outcome, outcome.state);
      }

      void run() {
        std::unique_lock lock(mutex_);
        std::uint64_t observed_watch_revision = 0;
        while (!stopping_) {
          if (watches_.empty() && pending_.empty() && requests_.empty()) {
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
      nlohmann::json status_ {{"detected", false}};
      // Thread-safe; written by the request thread (choices) and the worker (completion).
      detail::setup_decisions decisions_ {decisions_file()};
      // Worker-owned state: no filesystem/process operation holds mutex_.
      std::optional<detected_game> last_game_;
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
