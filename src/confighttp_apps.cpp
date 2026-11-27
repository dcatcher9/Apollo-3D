/**
 * @file src/confighttp_apps.cpp
 * @brief Web UI routes for the app library and its cover images.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// lib includes
#include <nlohmann/json.hpp>
#include <Simple-Web-Server/crypto.hpp>

// local includes
#include "apps_store.h"
#include "config.h"
#include "confighttp.h"
#include "confighttp_common.h"
#include "confighttp_routes.h"
#include "file_handler.h"
#include "httpcommon.h"
#include "logging.h"
#include "nvhttp.h"
#include "platform/common.h"
#include "process.h"

using namespace std::literals;

namespace confighttp {
  /**
   * @brief Get the list of available applications.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/apps| GET| null}
   */
  void getApps(resp_https_t response, req_https_t request) {
    try {
      std::string content = file_handler::read_file(config::stream.file_apps.c_str());
      nlohmann::json file_tree = nlohmann::json::parse(content);

      // Runtime-generated launch targets are deliberately not persisted in apps.json. Expose
      // them separately so the Library can show them without allowing an edit, delete, or
      // reorder operation to create a duplicate persisted entry.
      file_tree["builtin_apps"] = nlohmann::json::array();
      for (const auto &app : proc::proc.get_app_list()) {
        if (!app.synthetic_virtual_display) {
          continue;
        }

        file_tree["builtin_apps"].push_back({
          {"uuid", app.uuid},
          {"name", app.name},
          {"image-path", app.image_path},
          {"virtual-display", true}
        });
      }

      file_tree["current_app"] = proc::proc.get_running_app_uuid();
      file_tree["host_uuid"] = http::unique_id;
      file_tree["host_name"] = config::nvhttp.sunshine_name;

      send_response(response, file_tree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "GetApps: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }

  /**
   * @brief Save an application. To save a new application the UUID must be empty.
   *        To update an existing application, you must provide the current UUID of the application.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   * The body for the post request should be JSON serialized in the following format:
   * @code{.json}
   * {
   *   "name": "Application Name",
   *   "output": "Log Output Path",
   *   "cmd": "Command to run the application",
   *   "exclude-global-prep-cmd": false,
   *   "elevated": false,
   *   "auto-detach": true,
   *   "wait-all": true,
   *   "exit-timeout": 5,
   *   "prep-cmd": [
   *     {
   *       "do": "Command to prepare",
   *       "undo": "Command to undo preparation",
   *       "elevated": false
   *     }
   *   ],
   *   "detached": [
   *     "Detached command"
   *   ],
   *   "image-path": "Full path to the application image. Must be a png file.",
   *   "uuid": "aaaa-bbbb"
   * }
   * @endcode
   *
   * @api_examples{/api/apps| POST| {"name":"Hello, World!","uuid": "aaaa-bbbb"}}
   */
  void saveApp(resp_https_t response, req_https_t request) {
    BOOST_LOG(info) << config::stream.file_apps;
    try {
      // TODO: Input Validation

      // Read the input JSON from the request body.
      nlohmann::json inputTree = nlohmann::json::parse(request->content);

      // Read the existing apps file.
      std::string content = file_handler::read_file(config::stream.file_apps.c_str());
      nlohmann::json fileTree = nlohmann::json::parse(content);

      // Migrate/merge the new app into the file tree.
      proc::migrate_apps(&fileTree, &inputTree);

      // Write the updated file tree back to disk.
      file_handler::write_file(config::stream.file_apps.c_str(), fileTree.dump(4));
      nvhttp::reload_apps(config::stream.file_apps);

      // Prepare and send the output response.
      nlohmann::json outputTree;
      outputTree["status"] = true;
      send_response(response, outputTree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "SaveApp: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }

  /**
   * @brief Close the currently running application.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/apps/close| POST| null}
   */
  void closeApp(resp_https_t response, req_https_t request) {
    nvhttp::terminate_active_session();
    nlohmann::json output_tree;
    output_tree["status"] = true;
    send_response(response, output_tree);
  }

  nlohmann::json reorder_app_list(nlohmann::json original_apps_list, const nlohmann::json &order_uuids) {
    nlohmann::json reordered_apps_list = nlohmann::json::array();
    std::vector<bool> item_moved(original_apps_list.size(), false);

    // Place the first remaining app for each requested UUID, including duplicate UUIDs.
    for (const auto &uuid_json_value : order_uuids) {
      if (!uuid_json_value.is_string()) {
        BOOST_LOG(warning) << "ReorderApps: Encountered a non-string UUID in the 'order' array. Skipping this entry.";
        continue;
      }
      const auto &target_uuid = uuid_json_value.get_ref<const nlohmann::json::string_t &>();
      bool found_match_for_ordered_uuid = false;

      for (size_t i = 0; i < original_apps_list.size(); ++i) {
        if (item_moved[i]) {
          continue;
        }

        auto &app_item = original_apps_list[i];
        if (app_item.is_object() && app_item.contains("uuid") && app_item["uuid"].is_string()) {
          if (app_item["uuid"].get_ref<const nlohmann::json::string_t &>() == target_uuid) {
            reordered_apps_list.push_back(std::move(app_item));
            item_moved[i] = true;
            found_match_for_ordered_uuid = true;
            break;
          }
        }
      }

      if (!found_match_for_ordered_uuid) {
        BOOST_LOG(debug) << "ReorderApps: UUID '" << target_uuid << "' from 'order' array not found in available apps list or its matching app was already processed. Omitting.";
      }
    }

    // Preserve the original order and data of every unspecified or malformed entry.
    for (size_t i = 0; i < original_apps_list.size(); ++i) {
      if (!item_moved[i]) {
        reordered_apps_list.push_back(std::move(original_apps_list[i]));
      }
    }
    return reordered_apps_list;
  }

  /**
   * @brief Reorder applications.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/apps/reorder| POST| {"order": ["aaaa-bbbb", "cccc-dddd"]}}
   */
  void reorderApps(resp_https_t response, req_https_t request) {
    try {
      nlohmann::json input_tree = nlohmann::json::parse(request->content);
      nlohmann::json output_tree;

      // Read the existing apps file.
      std::string content = file_handler::read_file(config::stream.file_apps.c_str());
      nlohmann::json fileTree = nlohmann::json::parse(content);

      // Get the desired order of UUIDs from the request.
      if (!input_tree.contains("order") || !input_tree["order"].is_array()) {
        throw std::runtime_error("Missing or invalid 'order' array in request body");
      }
      const auto &order_uuids_json = input_tree["order"];

      // Get the original apps array from the fileTree.
      // Default to an empty array if "apps" key is missing or if it's present but not an array (after logging an error).
      nlohmann::json original_apps_list = nlohmann::json::array();
      if (fileTree.contains("apps")) {
        if (fileTree["apps"].is_array()) {
          original_apps_list = std::move(fileTree["apps"]);
        } else {
          // "apps" key exists but is not an array. This is a malformed state.
          BOOST_LOG(error) << "ReorderApps: 'apps' key in apps configuration file ('" << config::stream.file_apps
                           << "') is present but not an array.";
          throw std::runtime_error("'apps' in file is not an array, cannot reorder.");
        }
      } else {
        // "apps" key is missing. Treat as an empty list. Reordering an empty list is valid.
        BOOST_LOG(debug) << "ReorderApps: 'apps' key missing in apps configuration file ('" << config::stream.file_apps
                         << "'). Treating as an empty list for reordering.";
        // original_apps_list is already an empty array, so no specific action needed here.
      }

      // Update the fileTree with the new, reordered list of apps.
      fileTree["apps"] = reorder_app_list(std::move(original_apps_list), order_uuids_json);

      // Write the modified fileTree back to the apps configuration file.
      file_handler::write_file(config::stream.file_apps.c_str(), fileTree.dump(4));

      // Notify relevant parts of the system that the apps configuration has changed.
      nvhttp::reload_apps(config::stream.file_apps);

      output_tree["status"] = true;
      send_response(response, output_tree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "ReorderApps: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }

  /**
   * @brief Delete an application.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/apps/delete | POST| { uuid: 'aaaa-bbbb' }}
   */
  void deleteApp(resp_https_t response, req_https_t request) {
    try {
      nlohmann::json input_tree = nlohmann::json::parse(request->content);

      // Check for required uuid field in body
      if (!input_tree.contains("uuid") || !input_tree["uuid"].is_string()) {
        bad_request(response, request, "Missing or invalid uuid in request body");
        return;
      }
      auto uuid = input_tree["uuid"].get<std::string>();

      // Read the apps file into a nlohmann::json object.
      std::string content = file_handler::read_file(config::stream.file_apps.c_str());
      nlohmann::json fileTree = nlohmann::json::parse(content);

      // Remove any app with the matching uuid directly from the "apps" array.
      if (fileTree.contains("apps") && fileTree["apps"].is_array()) {
        auto &apps = fileTree["apps"];
        apps.erase(
          std::remove_if(apps.begin(), apps.end(), [&uuid](const nlohmann::json &app) {
            return app.value("uuid", "") == uuid;
          }),
          apps.end()
        );
      }

      // Write the updated JSON back to the file.
      file_handler::write_file(config::stream.file_apps.c_str(), fileTree.dump(4));
      nvhttp::reload_apps(config::stream.file_apps);

      // Prepare and send the response.
      nlohmann::json outputTree;
      outputTree["status"] = true;
      send_response(response, outputTree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "DeleteApp: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }

  /**
   * @brief Upload a cover image.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/covers/upload| POST| {"key":"igdb_1234","url":"https://images.igdb.com/igdb/image/upload/t_cover_big_2x/abc123.png"}}
   */
  void uploadCover(resp_https_t response, req_https_t request) {
    try {
      nlohmann::json input_tree = nlohmann::json::parse(request->content);
      nlohmann::json output_tree;
      std::string key = input_tree.value("key", "");
      if (key.empty()) {
        bad_request(response, request, "Cover key is required");
        return;
      }
      std::string url = input_tree.value("url", "");
      const std::string coverdir = platf::appdata().string() + "/covers/";
      file_handler::make_directory(coverdir);
      std::string path = coverdir + http::url_escape(key) + ".png";
      if (!url.empty()) {
        if (http::url_get_host(url) != "images.igdb.com") {
          bad_request(response, request, "Only images.igdb.com is allowed");
          return;
        }
        if (!http::download_file(url, path)) {
          bad_request(response, request, "Failed to download cover");
          return;
        }
      } else {
        auto data = SimpleWeb::Crypto::Base64::decode(input_tree.value("data", ""));
        // Binary: a text-mode stream would write every 0x0A of the image as 0x0D 0x0A.
        std::ofstream imgfile(path, std::ios::binary | std::ios::trunc);
        imgfile.write(data.data(), static_cast<std::streamsize>(data.size()));
        imgfile.close();
        if (!imgfile) {
          bad_request(response, request, "Failed to save cover");
          return;
        }
      }
      output_tree["status"] = true;
      output_tree["path"] = path;
      send_response(response, output_tree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "UploadCover: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }
}  // namespace confighttp
