/**
 * @file src/confighttp_pages.cpp
 * @brief Web UI pages and static assets.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

// local includes
#include "confighttp.h"
#include "confighttp_common.h"
#include "confighttp_routes.h"
#include "file_handler.h"
#include "logging.h"
#include "network.h"

using namespace std::literals;

namespace confighttp {
  namespace {
    const std::map<std::string, std::string> mime_types = {
      {"css", "text/css"},
      {"gif", "image/gif"},
      {"htm", "text/html"},
      {"html", "text/html"},
      {"ico", "image/x-icon"},
      {"jpeg", "image/jpeg"},
      {"jpg", "image/jpeg"},
      {"js", "application/javascript"},
      {"json", "application/json"},
      {"png", "image/png"},
      {"svg", "image/svg+xml"},
      {"ttf", "font/ttf"},
      {"txt", "text/plain"},
      {"woff2", "font/woff2"},
      {"xml", "text/xml"},
    };
  }  // namespace

  namespace fs = std::filesystem;

  /**
   * @brief Get the index page.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void getIndexPage(resp_https_t response, req_https_t request) {
    std::string content = file_handler::read_file(WEB_DIR "index.html");
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "text/html; charset=utf-8");
    add_frame_guard(headers);
    response->write(content, headers);
  }

  /**
   * @brief Get the PIN page.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void getPinPage(resp_https_t response, req_https_t request) {
    std::string content = file_handler::read_file(WEB_DIR "pin.html");
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "text/html; charset=utf-8");
    add_frame_guard(headers);
    response->write(content, headers);
  }

  /**
   * @brief Get the apps page.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void getAppsPage(resp_https_t response, req_https_t request) {
    std::string content = file_handler::read_file(WEB_DIR "apps.html");
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "text/html; charset=utf-8");
    add_frame_guard(headers);
    headers.emplace("Access-Control-Allow-Origin", "https://images.igdb.com/");
    response->write(content, headers);
  }

  /**
   * @brief Get the configuration page.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void getConfigPage(resp_https_t response, req_https_t request) {
    std::string content = file_handler::read_file(WEB_DIR "config.html");
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "text/html; charset=utf-8");
    add_frame_guard(headers);
    response->write(content, headers);
  }

  /**
   * @brief Get the login page.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @todo Combine this function with getWelcomePage if appropriate.
   */
  void getLoginPage(resp_https_t response, req_https_t request) {
    const auto peer_class = checkIPOrigin(response, request);
    if (!peer_class) {
      return;
    }

    if (*peer_class <= net::LAN) {
      send_redirect(response, request, "/");
      return;
    }

    if (!credentials_configured()) {
      send_redirect(response, request, "/welcome");
      return;
    }

    std::string content = file_handler::read_file(WEB_DIR "login.html");
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "text/html; charset=utf-8");
    headers.emplace("Cache-Control", "no-store");
    add_frame_guard(headers);
    response->write(content, headers);
  }

  /**
   * @brief Get the welcome page.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void getWelcomePage(resp_https_t response, req_https_t request) {
    if (!checkIPOrigin(response, request)) {
      return;
    }

    print_req(request);

    if (credentials_configured()) {
      send_redirect(response, request, "/");
      return;
    }

    std::string content = file_handler::read_file(WEB_DIR "welcome.html");
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "text/html; charset=utf-8");
    add_frame_guard(headers);
    response->write(content, headers);
  }

  /**
   * @brief Get the troubleshooting page.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void getTroubleshootingPage(resp_https_t response, req_https_t request) {
    std::string content = file_handler::read_file(WEB_DIR "troubleshooting.html");
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "text/html; charset=utf-8");
    add_frame_guard(headers);
    response->write(content, headers);
  }

  void getOfflineConversionPage(resp_https_t response, req_https_t request) {
    const std::string content =
      file_handler::read_file(WEB_DIR "offline-conversion.html");
    const SimpleWeb::CaseInsensitiveMultimap headers {
      {"Content-Type", "text/html; charset=utf-8"},
      {"Cache-Control", "no-store"},
      {"X-Frame-Options", "DENY"},
      {"Content-Security-Policy", "frame-ancestors 'none';"}
    };
    response->write(content, headers);
  }

  /**
   * @brief Get the favicon image.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void getFaviconImage(resp_https_t response, req_https_t request) {
    print_req(request);

    std::ifstream in(WEB_DIR "images/sunshine3d.ico", std::ios::binary);
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "image/x-icon");
    add_frame_guard(headers);
    response->write(SimpleWeb::StatusCode::success_ok, in, headers);
  }

  /**
   * @brief Get the Sunshine 3D logo image.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * @todo combine function with getFaviconImage and possibly getNodeModules
   * @todo use mime_types map
   */
  void getProductLogoImage(resp_https_t response, req_https_t request) {
    print_req(request);

    std::ifstream in(WEB_DIR "images/logo-sunshine3d-45.png", std::ios::binary);
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "image/png");
    add_frame_guard(headers);
    response->write(SimpleWeb::StatusCode::success_ok, in, headers);
  }

  /**
   * @brief Check if a path is a child of another path.
   * @param base The base path.
   * @param query The path to check.
   * @return True if the path is a child of the base path, false otherwise.
   */
  bool isChildPath(fs::path const &base, fs::path const &query) {
    auto relPath = fs::relative(base, query);
    return *(relPath.begin()) != fs::path("..");
  }

  /**
   * @brief Get an asset from the node_modules directory.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void getNodeModules(resp_https_t response, req_https_t request) {
    print_req(request);

    fs::path webDirPath(WEB_DIR);
    fs::path nodeModulesPath(webDirPath / "assets");

    // .relative_path is needed to shed any leading slash that might exist in the request path
    auto filePath = fs::weakly_canonical(webDirPath / fs::path(request->path).relative_path());

    // Don't do anything if file does not exist or is outside the assets directory
    if (!isChildPath(filePath, nodeModulesPath)) {
      BOOST_LOG(warning) << "Someone requested a path " << filePath << " that is outside the assets folder";
      bad_request(response, request);
      return;
    }

    if (!fs::exists(filePath)) {
      not_found(response, request);
      return;
    }

    auto relPath = fs::relative(filePath, webDirPath);
    // get the mime type from the file extension mime_types map
    // remove the leading period from the extension
    auto mimeType = mime_types.find(relPath.extension().string().substr(1));
    if (mimeType == mime_types.end()) {
      bad_request(response, request);
      return;
    }
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", mimeType->second);
    add_frame_guard(headers);
    std::ifstream in(filePath.string(), std::ios::binary);
    response->write(SimpleWeb::StatusCode::success_ok, in, headers);
  }
}  // namespace confighttp
