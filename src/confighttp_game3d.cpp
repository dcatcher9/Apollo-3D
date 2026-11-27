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
  nlohmann::json game3d_setup_json(const platf::game3d::setup_status_t &status) {
    using platf::game3d::game_kind_e;
    using platf::game3d::text_hook::protection_e;

    nlohmann::json result {{"detected", false}};
    if (status.card) {
      const auto &card = *status.card;
      result = {
        {"detected", true},
        {"game_name", card.game_name},
        {"executable", card.executable},
        {"kind", card.kind == game_kind_e::text_hook ? "text_hook" : "addon"},
        {"text_helper", card.text_helper},
        {"chosen", card.chosen},
        {"state", card.state},
        {"detail", card.detail},
        {"reshade_present", card.reshade_present},
        {"install_script", card.install_script},
        {"runtime_install_script", card.runtime_install_script},
      };
      if (card.identity) {
        result["identity"] = *card.identity;
      }
    }
    result["active"] = status.active;
    if (status.text_protection) {
      const auto protection = status.text_protection->protection;
      result["text_protection"] = protection == protection_e::game_text ? "game" : protection == protection_e::picture ? "fallback" :
                                                                                                                         "";
      result["text_protection_detail"] = status.text_protection->detail;
    }
    result["choosable"] = nlohmann::json::array();
    for (const auto &application : status.choosable) {
      result["choosable"].push_back({{"game_name", application.game_name}, {"executable", application.executable}, {"identity", application.identity}, {"text_helper", application.text_helper}});
    }
    return result;
  }

  void getGame3dSetup(resp_https_t response, req_https_t request) {
    send_response(response, game3d_setup_json(platf::game3d::setup_status()));
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
      send_response(response, nlohmann::json {{"status", true}, {"setup", game3d_setup_json(platf::game3d::setup_status())}});
    } catch (const std::exception &error) {
      bad_request(response, request, error.what());
    }
  }
}  // namespace confighttp
