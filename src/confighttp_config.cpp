/**
 * @file src/confighttp_config.cpp
 * @brief Web UI routes for the configuration file, logs and the host's own lifetime.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <sstream>
#include <stdexcept>
#include <string>

// lib includes
#include <nlohmann/json.hpp>

// local includes
#include "config.h"
#include "confighttp.h"
#include "confighttp_common.h"
#include "confighttp_routes.h"
#include "entry_handler.h"
#include "file_handler.h"
#include "logging.h"
#include "platform/common.h"
#include "process.h"

#include "platform/windows/ar_glasses.h"
#include "platform/windows/host3d_subtitle_apps.h"
#include "platform/windows/utils.h"

using namespace std::literals;

namespace confighttp {
  void erase_retired_config_options(nlohmann::json &config_tree) {
    if (!config_tree.is_object()) {
      return;
    }

    for (auto option = config_tree.begin(); option != config_tree.end();) {
      if (config::is_retired_config_option(option.key())) {
        option = config_tree.erase(option);
      } else {
        ++option;
      }
    }
  }

  /**
   * @brief Get the configuration settings.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void getConfig(resp_https_t response, req_https_t request) {
    nlohmann::json output_tree;
    output_tree["status"] = true;
    output_tree["version"] = PROJECT_VERSION;
    output_tree["vdisplayStatus"] = (int) proc::vDisplayDriverStatus.load();
    auto vars = config::parse_config(file_handler::read_file(config::sunshine.config_file.c_str()));
    for (auto &[name, value] : vars) {
      output_tree[name] = value;
    }
    send_response(response, output_tree);
  }

  /**
   * @brief Get the locale setting.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/configLocale| GET| null}
   */
  void getLocale(resp_https_t response, req_https_t request) {
    print_req(request);

    nlohmann::json output_tree;
    output_tree["status"] = true;
    output_tree["locale"] = config::sunshine.locale;
    send_response(response, output_tree);
  }

  /**
   * @brief Get the host's version, which every page's navigation bar shows.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/version| GET| null}
   */
  void getVersion(resp_https_t response, req_https_t request) {
    nlohmann::json output_tree;
    output_tree["status"] = true;
    output_tree["version"] = PROJECT_VERSION;
    send_response(response, output_tree);
  }

  /**
   * @brief List the applications Host 3D may protect subtitles in.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * Enabled applications, then applications recently seen in front of a Host 3D stream.
   *
   * @api_examples{/api/host3d/subtitle-apps| GET| null}
   */
  void getHost3dSubtitleApps(resp_https_t response, req_https_t request) {
    nlohmann::json output_tree;
    output_tree["status"] = true;
    output_tree["apps"] = nlohmann::json::array();
    for (const auto &app : platf::host3d_subtitle_apps::list()) {
      output_tree["apps"].push_back({
        {"executable", app.executable},
        {"name", app.name},
        {"enabled", app.enabled},
        {"recent", app.recent},
      });
    }
    send_response(response, output_tree);
  }

  /**
   * @brief Turn Host 3D subtitle protection on or off for one application.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * The running host changes at once and the selection is saved to the configuration file.
   *
   * @api_examples{/api/host3d/subtitle-apps| POST| {"executable":"C:/Games/Example/game.exe","enabled":true}}
   */
  void setHost3dSubtitleApp(resp_https_t response, req_https_t request) {
    try {
      std::stringstream body;
      body << request->content.rdbuf();
      const auto input = nlohmann::json::parse(body);
      const auto executable = input.find("executable");
      const auto enabled = input.find("enabled");
      if (executable == input.end() || !executable->is_string() || enabled == input.end() || !enabled->is_boolean() ||
          !platf::host3d_subtitle_apps::set_enabled(executable->get<std::string>(), enabled->get<bool>())) {
        bad_request(response, request, "An absolute .exe path and a boolean 'enabled' are required");
        return;
      }
      BOOST_LOG(info) << "Host 3D subtitle protection "sv << (enabled->get<bool>() ? "enabled"sv : "disabled"sv)
                      << " for ["sv << executable->get<std::string>() << "] from the Web UI."sv;
      // The live selection already changed. A failed save only loses it at the next start.
      const bool saved = ar_glasses::persist_config_value(
        platf::host3d_subtitle_apps::config_key, platf::host3d_subtitle_apps::serialize()
      );
      send_response(response, nlohmann::json {{"status", true}, {"saved", saved}});
    } catch (const std::exception &error) {
      bad_request(response, request, error.what());
    }
  }

  /**
   * @brief Save the configuration settings.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   * The body for the post request should be JSON serialized in the following format:
   * @code{.json}
   * {
   *   "key": "value"
   * }
   * @endcode
   *
   * @attention{It is recommended to ONLY save the config settings that differ from the default behavior.}
   *
   * @api_examples{/api/config| POST| {"key":"value"}}
   */
  void saveConfig(resp_https_t response, req_https_t request) {
    try {
      // TODO: Input Validation
      std::stringstream config_stream;
      nlohmann::json output_tree;
      nlohmann::json input_tree = nlohmann::json::parse(request->content);
      erase_retired_config_options(input_tree);
      // This option is live-managed by the AR display decision API rather than the general form.
      input_tree.erase("ar_glass_devices");
      // The per-application subtitle selection is live-managed by its own API as well. Write the
      // current selection so a general save cannot drop or revive an entry.
      // The 3D source of each local surface is switched live next to Start 3D. Write the
      // current choice so a general save cannot bring back the one the form was loaded with.
      input_tree["local_monitor_mode"] = ar_glasses::monitor_source_is_game() ? "game" : "host";
      input_tree["sbs_reshade"] = ar_glasses::ar_source_is_game() ? "enabled" : "disabled";
      input_tree.erase(std::string {platf::host3d_subtitle_apps::config_key});
      if (const auto subtitle_apps = nlohmann::json::parse(platf::host3d_subtitle_apps::serialize());
          !subtitle_apps.empty()) {
        input_tree[std::string {platf::host3d_subtitle_apps::config_key}] = subtitle_apps;
      }
      for (const auto &[k, v] : input_tree.items()) {
        if (v.is_null() || (v.is_string() && v.get<std::string>().empty())) {
          continue;
        }

        // v.dump() will dump valid json, which we do not want for strings in the config right now
        // we should migrate the config file to straight json and get rid of all this nonsense
        config_stream << k << " = " << (v.is_string() ? v.get<std::string>() : v.dump()) << std::endl;
      }
      if (!ar_glasses::write_config_with_devices(config_stream.str())) {
        throw std::runtime_error("Could not atomically save the configuration file");
      }
      output_tree["status"] = true;
      send_response(response, output_tree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "SaveConfig: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }

  /**
   * @brief Get the logs from the log file.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/logs| GET| null}
   */
  void getLogs(resp_https_t response, req_https_t request) {
    std::string content = file_handler::read_file(config::sunshine.log_file.c_str());
    SimpleWeb::CaseInsensitiveMultimap headers;
    std::string contentType = "text/plain";
    contentType += "; charset=";
    contentType += currentCodePageToCharset();
    headers.emplace("Content-Type", contentType);
    add_frame_guard(headers);
    response->write(SimpleWeb::StatusCode::success_ok, content, headers);
  }

  /**
   * @brief Restart Apollo.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/restart| POST| null}
   */
  void restart(resp_https_t response, req_https_t request) {
    // We may not return from this call
    platf::restart();
  }

  /**
   * @brief Quit Apollo.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * On Windows, if running in a service, a special shutdown code is returned.
   */
  void quit(resp_https_t response, req_https_t request) {
    BOOST_LOG(warning) << "Requested quit from config page!"sv;

    // Complete the handler-owned response before raising shutdown. Retaining this Response beyond
    // the handler would retain a deleter that references the local HTTPS server after it stops.
    response->write();

    if (GetConsoleWindow() == NULL) {
      lifetime::exit_sunshine(ERROR_SHUTDOWN_IN_PROGRESS, true);
    } else {
      lifetime::exit_sunshine(0, true);
    }
  }
}  // namespace confighttp
