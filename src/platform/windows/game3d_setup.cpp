// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_setup.h"

#include "game3d_detection.h"
#include "game3d_installation.h"
#include "src/logging.h"
#include "src/system_tray.h"

#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace platf::game3d {
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
        case installation_state_e::unavailable:
          return "unavailable";
      }
      return "unavailable";
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
        if (!worker_.joinable()) {
          worker_ = std::thread([this]() {
            run();
          });
        }
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

      void publish(const detected_game &game, const installation_status &installation, const char *state = nullptr) {
        const auto executable = utf8(game.executable);
        const auto label = state ? state : state_name(installation.state);
        {
          std::lock_guard lock(mutex_);
          status_ = {
            {"detected", true},
            {"game_name", game.name},
            {"executable", executable},
            {"state", label},
            {"detail", installation.detail},
            {"install_script", utf8(bundled_addon_directory() / "install.ps1")},
            {"runtime_install_script", utf8(bundled_addon_directory() / "install-runtime.ps1")},
          };
        }
        // Focus changes and encoder rebuilds must not repeat the same suggestion.
        const auto notification_key = executable + "\n" + label;
        if (announced_.contains(notification_key)) {
          return;
        }
        announced_.emplace(notification_key, true);
        BOOST_LOG(info) << "Game 3D setup: " << game.name << " [" << label << "] executable=\"" << executable
                        << "\" pid=" << game.process_id << " creation_time=" << game.creation_time
                        << " evidence=" << (game.evidence == detection_evidence_e::fullscreen_direct3d ? "fullscreen_direct3d" : "known_runtime")
                        << ": " << installation.detail;
#if defined(SUNSHINE_TRAY) && SUNSHINE_TRAY >= 1
        if (installation.state != installation_state_e::current) {
          const auto target_name = game.name + " (" + utf8(game.executable.parent_path().filename()) + ")";
          system_tray::update_tray_game3d_setup(target_name, installation.detail);
        }
#endif
      }

      installation_status ensure_installation(const detected_game &game, bool publish_progress = true) {
        auto result = inspect_installation(game.executable);
        if (result.state == installation_state_e::missing_reshade) {
          auto installing = result;
          installing.detail = "Installing ReShade with full add-on support beside the game executable. Keep playing; setup takes effect on the next launch.";
          if (publish_progress) {
            publish(game, installing, "installing");
          }
          result = install_reshade_for_next_launch(game.executable);
          if (result.state != installation_state_e::installed) {
            return result;
          }
          result = install_addon_for_next_launch(game.executable);
          if (result.state == installation_state_e::installed) {
            result.detail = "ReShade and the Sunshine 3D add-on are installed beside the game executable. Keep playing; Game 3D takes effect on the next launch.";
          }
          return result;
        }
        if (result.state == installation_state_e::missing_addon) {
          auto installing = result;
          installing.detail = "Installing the Sunshine 3D add-on for the next launch. Your current game can stay open.";
          if (publish_progress) {
            publish(game, installing, "installing");
          }
          return install_addon_for_next_launch(game.executable);
        }
        if (result.state == installation_state_e::update_available) {
          return update_existing_addon(game.executable);
        }
        return result;
      }

      void tick(const std::optional<RECT> &capture_rect) {
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
          const auto result = ensure_installation(it->second.game, selected);
          // A background game's update must not replace another game's instructions.
          if (selected) {
            publish(it->second.game, result);
          }
          if (result.state == installation_state_e::update_pending || result.state == installation_state_e::update_available || result.state == installation_state_e::unavailable) {
            it->second.retry_at = std::chrono::steady_clock::now() + (result.state == installation_state_e::unavailable ? 30s : 2s);
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
        const auto result = ensure_installation(*game);
        if (result.state == installation_state_e::update_pending || result.state == installation_state_e::update_available || result.state == installation_state_e::unavailable) {
          pending_[game->executable] = {*game, std::chrono::steady_clock::now() + (result.state == installation_state_e::unavailable ? 30s : 2s)};
        }
        publish(*game, result);
      }

      void run() {
        std::unique_lock lock(mutex_);
        std::uint64_t observed_watch_revision = 0;
        while (!stopping_) {
          if (watches_.empty() && pending_.empty()) {
            wake_.wait(lock, [this]() {
              return stopping_ || !watches_.empty();
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
          changed_ = false;
          lock.unlock();
          try {
            tick(rect);
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
      // Worker-owned state: no filesystem/process operation holds mutex_.
      std::optional<detected_game> last_game_;
      detection_confirmation confirmation_;
      std::map<std::filesystem::path, pending_update> pending_;
      std::map<std::string, bool> announced_;
      nlohmann::json status_ {{"detected", false}};
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

  void shutdown_setup() {
    service().stop();
  }
}  // namespace platf::game3d
