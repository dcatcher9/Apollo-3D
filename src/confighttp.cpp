/**
 * @file src/confighttp.cpp
 * @brief Definitions for the Web UI Config HTTPS server.
 *
 * @todo Authentication, better handling of routes common to nvhttp, cleanup
 *
 * The route table and the server live here. Each handler lives in the confighttp_*.cpp file
 * of its domain, and confighttp_common.cpp holds the guards and responses they share.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

// lib includes
#include <boost/asio/ip/address.hpp>
#include <Simple-Web-Server/server_https.hpp>

// local includes
#include "config.h"
#include "confighttp.h"
#include "confighttp_common.h"
#include "confighttp_routes.h"
#include "globals.h"
#include "logging.h"
#include "network.h"

using namespace std::literals;

namespace confighttp {
  using https_server_t = SimpleWeb::Server<SimpleWeb::HTTPS>;

  bounded_content_length_e validate_bounded_content_length(
    std::optional<std::string_view> value,
    const std::size_t maximum
  ) {
    if (!value) {
      return bounded_content_length_e::absent_or_within_limit;
    }

    auto text = *value;
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
      text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) {
      text.remove_suffix(1);
    }
    if (text.empty()) {
      return bounded_content_length_e::invalid;
    }

    std::size_t length = 0;
    const auto parsed = std::from_chars(
      text.data(),
      text.data() + text.size(),
      length
    );
    if (parsed.ec != std::errc {} || parsed.ptr != text.data() + text.size()) {
      return bounded_content_length_e::invalid;
    }
    return length > maximum ?
             bounded_content_length_e::exceeds_limit :
             bounded_content_length_e::absent_or_within_limit;
  }

  std::size_t request_content_length_limit(
    const std::string_view method,
    const std::string_view path
  ) {
    if (
      (method == "POST" || method == "DELETE") &&
      (
        path == "/api/offline-sbs/browse" ||
        path == "/api/offline-sbs/jobs" ||
        path.starts_with("/api/offline-sbs/jobs/")
      )
    ) {
      return OFFLINE_SBS_REQUEST_MAX_BYTES;
    }
    return WEB_UI_REQUEST_STREAM_MAX_BYTES;
  }

  /** Who may call a route. Routes registered without guarded() are public. */
  enum class access_e {
    page,  ///< Signed-in user; anyone else is sent to the login page.
    api,  ///< Signed-in user; anyone else gets 401.
    json_api,  ///< A JSON body first, then a signed-in user.
  };

  using route_handler_t = void (*)(resp_https_t, req_https_t);

  /** Checks a route's access, logs the admitted request, then runs its handler. */
  auto guarded(const access_e access, const route_handler_t handler) {
    return [access, handler](resp_https_t response, req_https_t request) {
      if (access == access_e::json_api && !validateContentType(response, request, "application/json")) {
        return;
      }
      if (!authenticate(response, request, access == access_e::page)) {
        return;
      }
      print_req(request);
      handler(std::move(response), std::move(request));
    };
  }

  /** Registers every Web UI route with its access. Tests register the same table. */
  void add_routes(SimpleWeb::ServerBase<SimpleWeb::HTTPS> &server) {
    server.default_resource["DELETE"] = [](resp_https_t response, req_https_t request) {
      bad_request(response, request);
    };
    server.default_resource["PATCH"] = [](resp_https_t response, req_https_t request) {
      bad_request(response, request);
    };
    server.default_resource["POST"] = [](resp_https_t response, req_https_t request) {
      bad_request(response, request);
    };
    server.default_resource["PUT"] = [](resp_https_t response, req_https_t request) {
      bad_request(response, request);
    };
    server.default_resource["GET"] = not_found;
    server.resource["^/$"]["GET"] = guarded(access_e::page, getIndexPage);
    server.resource["^/pin/?$"]["GET"] = guarded(access_e::page, getPinPage);
    server.resource["^/apps/?$"]["GET"] = guarded(access_e::page, getAppsPage);
    server.resource["^/config/?$"]["GET"] = guarded(access_e::page, getConfigPage);
    server.resource["^/welcome/?$"]["GET"] = getWelcomePage;
    server.resource["^/login/?$"]["GET"] = getLoginPage;
    server.resource["^/troubleshooting/?$"]["GET"] = guarded(access_e::page, getTroubleshootingPage);
    server.resource["^/offline-conversion/?$"]["GET"] = guarded(access_e::page, getOfflineConversionPage);
    server.resource["^/api/login"]["POST"] = login;
    server.resource["^/api/pin$"]["POST"] = guarded(access_e::json_api, savePin);
    server.resource["^/api/otp$"]["POST"] = guarded(access_e::json_api, getOTP);
    server.resource["^/api/apps$"]["GET"] = guarded(access_e::api, getApps);
    server.resource["^/api/apps$"]["POST"] = guarded(access_e::json_api, saveApp);
    server.resource["^/api/apps/reorder$"]["POST"] = guarded(access_e::json_api, reorderApps);
    server.resource["^/api/apps/delete$"]["POST"] = guarded(access_e::json_api, deleteApp);
    server.resource["^/api/apps/close$"]["POST"] = guarded(access_e::json_api, closeApp);
    server.resource["^/api/logs$"]["GET"] = guarded(access_e::api, getLogs);
    server.resource["^/api/offline-sbs/browse$"]["POST"] =
      guarded(access_e::api, browseOfflineSbsFilesystem);
    server.resource["^/api/offline-sbs/capabilities$"]["GET"] =
      guarded(access_e::api, getOfflineSbsCapabilities);
    server.resource["^/api/offline-sbs/jobs$"]["GET"] = guarded(access_e::api, listOfflineSbsJobs);
    server.resource["^/api/offline-sbs/jobs$"]["POST"] = guarded(access_e::api, createOfflineSbsJob);
    server.resource["^/api/offline-sbs/jobs/([0-9A-Fa-f-]+)/cancel$"]["POST"] =
      guarded(access_e::api, cancelOfflineSbsJob);
    server.resource["^/api/offline-sbs/jobs/([0-9A-Fa-f-]+)$"]["DELETE"] =
      guarded(access_e::api, clearOfflineSbsJob);
    server.resource["^/api/offline-sbs/jobs/([0-9A-Fa-f-]+)$"]["GET"] =
      guarded(access_e::api, getOfflineSbsJob);
    server.resource["^/api/config$"]["GET"] = guarded(access_e::api, getConfig);
    server.resource["^/api/config$"]["POST"] = guarded(access_e::json_api, saveConfig);
    server.resource["^/api/game3d/setup$"]["GET"] = guarded(access_e::api, getGame3dSetup);
    server.resource["^/api/game3d/setup/decision$"]["POST"] = guarded(access_e::json_api, decideGame3dSetup);
    server.resource["^/api/ar-glasses$"]["GET"] = guarded(access_e::api, getArGlassDevices);
    server.resource["^/api/ar-glasses$"]["POST"] = guarded(access_e::json_api, setArGlassDevice);
    server.resource["^/api/local-monitor$"]["GET"] = guarded(access_e::api, getLocalMonitor);
    server.resource["^/api/local-monitor$"]["POST"] = guarded(access_e::json_api, setLocalMonitor);
    server.resource["^/api/host3d/subtitle-apps$"]["GET"] = guarded(access_e::api, getHost3dSubtitleApps);
    server.resource["^/api/host3d/subtitle-apps$"]["POST"] = guarded(access_e::json_api, setHost3dSubtitleApp);
    server.resource["^/api/configLocale$"]["GET"] = getLocale;
    server.resource["^/api/version$"]["GET"] = guarded(access_e::api, getVersion);
    server.resource["^/api/restart$"]["POST"] = guarded(access_e::json_api, restart);
    server.resource["^/api/quit$"]["POST"] = guarded(access_e::api, quit);
    server.resource["^/api/password$"]["POST"] = savePassword;
    server.resource["^/api/clients/unpair-all$"]["POST"] = guarded(access_e::json_api, unpairAll);
    server.resource["^/api/clients/list$"]["GET"] = guarded(access_e::api, getClients);
    server.resource["^/api/clients/update$"]["POST"] = guarded(access_e::json_api, updateClient);
    server.resource["^/api/clients/unpair$"]["POST"] = guarded(access_e::json_api, unpair);
    server.resource["^/api/clients/disconnect$"]["POST"] = guarded(access_e::json_api, disconnect);
    server.resource["^/api/covers/upload$"]["POST"] = guarded(access_e::json_api, uploadCover);
    server.resource["^/images/sunshine3d.ico$"]["GET"] = getFaviconImage;
    server.resource["^/images/logo-sunshine3d-45.png$"]["GET"] = getProductLogoImage;
    // Preserve the old asset routes for cached pages during in-place upgrades.
    server.resource["^/images/apollo.ico$"]["GET"] = getFaviconImage;
    server.resource["^/images/logo-apollo-45.png$"]["GET"] = getProductLogoImage;
    server.resource["^/assets\\/.+$"]["GET"] = getNodeModules;
  }

  /**
   * @brief Start the HTTPS server.
   */
  void start() {
    auto shutdown_event = mail::man->event<bool>(mail::shutdown);
    auto port_https = net::map_port(PORT_HTTPS);
    auto address_family = net::af_from_enum_string(config::sunshine.address_family);
    https_server_t server {config::nvhttp.cert, config::nvhttp.pkey};
    add_routes(server);
    server.config.reuse_address = true;
    // The finite global ceiling preserves existing app/config/cover payloads. The header-time
    // selector applies the stricter 64 KiB offline contract before declared-body transfer or
    // cumulative chunk acceptance, including on a keep-alive connection.
    server.config.max_request_streambuf_size = WEB_UI_REQUEST_STREAM_MAX_BYTES;
    server.config.request_content_length_limit = [](const auto &request) {
      return request_content_length_limit(request.method, request.path);
    };
    const auto bind_address = net::get_bind_address(address_family);
    if (!bind_address) {
      BOOST_LOG(fatal) << "Configuration UI refused invalid bind_address ["sv << config::sunshine.bind_address << ']';
      shutdown_event->raise(true);
      return;
    }
    server.config.address = *bind_address;
    server.config.port = port_https;

    std::string display_address = "localhost";
    if (!config::sunshine.bind_address.empty()) {
      boost::system::error_code ec;
      const auto address = boost::asio::ip::make_address(server.config.address, ec);
      if (!ec) {
        display_address = net::addr_to_url_escaped_string(address);
      }
    }

    auto accept_and_run = [&, display_address = std::move(display_address)](auto *server) {
      try {
        server->start([display_address](unsigned short port) {
          BOOST_LOG(info) << "Configuration UI available at [https://"sv << display_address << ':' << port << ']';
        });
      } catch (boost::system::system_error &err) {
        // It's possible the exception gets thrown after calling server->stop() from a different thread
        if (shutdown_event->peek()) {
          return;
        }
        BOOST_LOG(fatal) << "Couldn't start Configuration HTTPS server on port ["sv << port_https << "]: "sv << err.what();
        shutdown_event->raise(true);
        return;
      }
    };
    std::thread tcp {accept_and_run, &server};

    // Wait for any event
    shutdown_event->view();

    server.stop();

    tcp.join();
  }
}  // namespace confighttp
