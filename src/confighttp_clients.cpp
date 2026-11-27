/**
 * @file src/confighttp_clients.cpp
 * @brief Web UI routes for paired clients and pairing.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <stdexcept>
#include <string>

// lib includes
#include <nlohmann/json.hpp>

// local includes
#include "config.h"
#include "confighttp_common.h"
#include "confighttp_routes.h"
#include "crypto.h"
#include "logging.h"
#include "nvhttp.h"
#include "platform/common.h"

using namespace std::literals;

namespace confighttp {
  /**
   * @brief Get the list of paired clients.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/clients/list| GET| null}
   */
  void getClients(resp_https_t response, req_https_t request) {
    nlohmann::json output_tree;
    output_tree["named_certs"] = nvhttp::get_all_clients();
    output_tree["status"] = true;
    send_response(response, output_tree);
  }

  /**
   * @brief Update client information.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * The body for the POST request should be JSON serialized in the following format:
   * @code{.json}
   * {
   *   "uuid": "<uuid>",
   *   "name": "<Friendly Name>",
   *   "do": [ { "cmd": "<command>", "elevated": false }, ... ],
   *   "undo": [ { "cmd": "<command>", "elevated": false }, ... ],
   *   "perm": <uint32_t>
   * }
   * @endcode
   */
  void updateClient(resp_https_t response, req_https_t request) {
    try {
      nlohmann::json input_tree = nlohmann::json::parse(request->content);
      nlohmann::json output_tree;
      std::string uuid = input_tree.value("uuid", "");
      std::string name = input_tree.value("name", "");
      auto perm = static_cast<crypto::PERM>(input_tree.value("perm", static_cast<uint32_t>(crypto::PERM::_no)) & static_cast<uint32_t>(crypto::PERM::_all));
      output_tree["status"] = nvhttp::update_device_info(
        uuid,
        name,
        perm
      );
      send_response(response, output_tree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "Update Client: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }

  /**
   * @brief Unpair a client.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * The body for the POST request should be JSON serialized in the following format:
   * @code{.json}
   * {
   *  "uuid": "<uuid>"
   * }
   * @endcode
   *
   * @api_examples{/api/clients/unpair| POST| {"uuid":"1234"}}
   */
  void unpair(resp_https_t response, req_https_t request) {
    try {
      nlohmann::json input_tree = nlohmann::json::parse(request->content);
      nlohmann::json output_tree;
      std::string uuid = input_tree.value("uuid", "");
      output_tree["status"] = nvhttp::unpair_client(uuid);
      send_response(response, output_tree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "Unpair: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }

  /**
   * @brief Unpair all clients.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/clients/unpair-all| POST| null}
   */
  void unpairAll(resp_https_t response, req_https_t request) {
    nvhttp::erase_all_clients();
    nlohmann::json output_tree;
    output_tree["status"] = true;
    send_response(response, output_tree);
  }

  /**
   * @brief Get a one-time password (OTP).
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @api_examples{/api/otp| GET| null}
   */
  void getOTP(resp_https_t response, req_https_t request) {
    nlohmann::json output_tree;
    try {
      nlohmann::json input_tree = nlohmann::json::parse(request->content);

      std::string passphrase = input_tree.value("passphrase", "");
      if (passphrase.empty()) {
        throw std::runtime_error("Passphrase not provided!");
      }
      if (passphrase.size() < 4) {
        throw std::runtime_error("Passphrase too short!");
      }

      std::string deviceName = input_tree.value("deviceName", "");
      output_tree["otp"] = nvhttp::request_otp(passphrase, deviceName);
      output_tree["ip"] = platf::get_local_ip_for_gateway();
      output_tree["name"] = config::nvhttp.sunshine_name;
      output_tree["status"] = true;
      output_tree["message"] = "OTP created, effective within 3 minutes.";
      send_response(response, output_tree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "OTP creation failed: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }

  /**
   * @brief Send a PIN code to the host.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * The body for the POST request should be JSON serialized in the following format:
   * @code{.json}
   * {
   *   "pin": "<pin>",
   *   "name": "Friendly Client Name"
   * }
   * @endcode
   *
   * @api_examples{/api/pin| POST| {"pin":"1234","name":"My PC"}}
   */
  void savePin(resp_https_t response, req_https_t request) {
    try {
      nlohmann::json input_tree = nlohmann::json::parse(request->content);
      nlohmann::json output_tree;
      std::string pin = input_tree.value("pin", "");
      std::string name = input_tree.value("name", "");
      output_tree["status"] = nvhttp::pin(pin, name);
      send_response(response, output_tree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "SavePin: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }

  /**
   * @brief Disconnect a client.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void disconnect(resp_https_t response, req_https_t request) {
    try {
      nlohmann::json output_tree;
      nlohmann::json input_tree = nlohmann::json::parse(request->content);
      std::string uuid = input_tree.value("uuid", "");
      output_tree["status"] = nvhttp::find_and_stop_session(uuid, true);
      send_response(response, output_tree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "Disconnect: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }
}  // namespace confighttp
