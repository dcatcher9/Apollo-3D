/**
 * @file src/confighttp_game3d.cpp
 * @brief Web UI routes for Game 3D setup.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <string>

// lib includes
#include <nlohmann/json.hpp>

// local includes
#include "confighttp_common.h"
#include "confighttp_routes.h"

#include "platform/windows/game3d_setup.h"

namespace confighttp {
  void getGame3dSetup(resp_https_t response, req_https_t request) {
    send_response(response, platf::game3d::setup_status());
  }

  /**
   * @brief Record the user's install/never choice for the executable Game 3D setup offers.
   *
   * The host never writes into a game folder without this confirmation. The body names the
   * offered executable and its identity exactly as the status reported them.
   */
  void decideGame3dSetup(resp_https_t response, req_https_t request) {
    try {
      const auto input = nlohmann::json::parse(request->content);
      const auto failure = platf::game3d::decide_setup(input.at("executable").get<std::string>(), input.at("identity").get<std::string>(), input.at("decision").get<std::string>());
      if (failure) {
        bad_request(response, request, *failure);
        return;
      }
      send_response(response, nlohmann::json {{"status", true}, {"setup", platf::game3d::setup_status()}});
    } catch (const std::exception &error) {
      bad_request(response, request, error.what());
    }
  }
}  // namespace confighttp
