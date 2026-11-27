/**
 * @file src/confighttp_common.cpp
 * @brief Web UI request guards and responses, and the session they authenticate against.
 *
 * The login and password routes live here because they own that session.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <chrono>
#include <format>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

// lib includes
#include <boost/algorithm/string.hpp>
#include <nlohmann/json.hpp>

// local includes
#include "config.h"
#include "confighttp.h"
#include "confighttp_common.h"
#include "confighttp_routes.h"
#include "crypto.h"
#include "http_request_logging.h"
#include "httpcommon.h"
#include "logging.h"
#include "network.h"
#include "utility.h"

using namespace std::literals;

namespace confighttp {
  // SESSION COOKIE
  static std::string sessionCookie;
  static std::chrono::time_point<std::chrono::steady_clock> cookie_creation_time;
  static std::mutex auth_mutex;

  bool credentials_configured() {
    std::lock_guard lock {auth_mutex};
    return !config::sunshine.username.empty();
  }

  std::string make_auth_cookie(const std::string_view token) {
    const auto max_age = std::chrono::duration_cast<std::chrono::seconds>(SESSION_EXPIRE_DURATION).count();
    return std::format("auth={}; Secure; HttpOnly; SameSite=Strict; Max-Age={}; Path=/", token, max_age);
  }

  /**
   * @brief Log the request details.
   * @param request The HTTP request object.
   */
  void print_req(const req_https_t &request) {
    http::log_request_debug(*request);
  }

  /** No Web UI response may be shown inside another page's frame. Every reply carries both headers. */
  void add_frame_guard(SimpleWeb::CaseInsensitiveMultimap &headers) {
    headers.emplace("X-Frame-Options", "DENY");
    headers.emplace("Content-Security-Policy", "frame-ancestors 'none';");
  }

  /**
   * @brief Send a response.
   * @param response The HTTP response object.
   * @param output_tree The JSON tree to send.
   */
  void send_response(resp_https_t response, const nlohmann::json &output_tree) {
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "application/json");
    add_frame_guard(headers);
    response->write(output_tree.dump(), headers);
  }

  void send_json_response(
    resp_https_t response,
    SimpleWeb::StatusCode status,
    const nlohmann::json &body
  ) {
    SimpleWeb::CaseInsensitiveMultimap headers {
      {"Content-Type", "application/json"},
      {"Cache-Control", "no-store"},
      {"X-Content-Type-Options", "nosniff"},
    };
    add_frame_guard(headers);
    response->write(status, body.dump(), headers);
  }

  /**
   * @brief Send a 401 Unauthorized response.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void send_unauthorized(resp_https_t response, req_https_t request) {
    auto address = net::addr_to_normalized_string(request->remote_endpoint().address());
    BOOST_LOG(info) << "Web UI: ["sv << address << "] -- not authorized"sv;
    constexpr SimpleWeb::StatusCode code = SimpleWeb::StatusCode::client_error_unauthorized;
    nlohmann::json tree;
    tree["status_code"] = code;
    tree["status"] = false;
    tree["error"] = "Unauthorized";
    SimpleWeb::CaseInsensitiveMultimap headers {
      {"Content-Type", "application/json"},
    };
    add_frame_guard(headers);
    response->write(code, tree.dump(), headers);
  }

  /**
   * @brief Send a redirect response.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   * @param path The path to redirect to.
   */
  void send_redirect(resp_https_t response, req_https_t request, const char *path) {
    auto address = net::addr_to_normalized_string(request->remote_endpoint().address());
    BOOST_LOG(info) << "Web UI: ["sv << address << "] -- redirecting"sv;
    SimpleWeb::CaseInsensitiveMultimap headers {
      {"Location", path},
    };
    add_frame_guard(headers);
    response->write(SimpleWeb::StatusCode::redirection_temporary_redirect, headers);
  }

  /**
   * @brief Retrieve the value of a key from a cookie string.
   * @param cookieString The cookie header string.
   * @param key The key to search.
   * @return The value if found, empty string otherwise.
   */
  std::string getCookieValue(const std::string &cookieString, const std::string &key) {
    std::string keyWithEqual = key + "=";
    std::size_t startPos = cookieString.find(keyWithEqual);
    if (startPos == std::string::npos) {
      return "";
    }
    startPos += keyWithEqual.length();
    std::size_t endPos = cookieString.find(";", startPos);
    if (endPos == std::string::npos) {
      return cookieString.substr(startPos);
    }
    return cookieString.substr(startPos, endPos - startPos);
  }

  /**
   * @brief Check if the IP origin is allowed.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   * @return The admitted peer class, or no value after sending the denial response.
   */
  std::optional<net::net_e> checkIPOrigin(resp_https_t response, req_https_t request) {
    const auto address = request->remote_endpoint().address();
    auto ip_type = net::from_address(address);
    if (ip_type > http::origin_web_ui_allowed) {
      BOOST_LOG(info) << "Web UI: ["sv << net::addr_to_normalized_string(address) << "] -- denied"sv;
      response->write(SimpleWeb::StatusCode::client_error_forbidden);
      return std::nullopt;
    }
    return ip_type;
  }

  request_header_t get_single_request_header(const req_https_t &request, const std::string_view name) {
    const auto range = request->header.equal_range(std::string {name});
    if (range.first == range.second) {
      return {true, std::nullopt};
    }

    auto entry = range.first;
    const std::string_view value {entry->second};
    if (++entry != range.second) {
      return {false, std::nullopt};
    }
    return {true, value};
  }

  namespace {
    bool is_safe_http_method(const std::string_view method) {
      return boost::iequals(method, "GET") ||
             boost::iequals(method, "HEAD") ||
             boost::iequals(method, "OPTIONS");
    }
  }  // namespace

  /**
   * @brief Reject unsafe browser requests that did not originate from this exact Web UI origin.
   */
  bool checkBrowserOrigin(resp_https_t response, req_https_t request) {
    if (is_safe_http_method(request->method)) {
      return true;
    }

    const auto host = get_single_request_header(request, "Host");
    const auto origin = get_single_request_header(request, "Origin");
    const auto referer = get_single_request_header(request, "Referer");
    const auto sec_fetch_site = get_single_request_header(request, "Sec-Fetch-Site");
    if (!host.unique || !host.value || !origin.unique || !referer.unique || !sec_fetch_site.unique || !http::web_ui_origin_allowed(*host.value, origin.value, referer.value, sec_fetch_site.value)) {
      const auto address = net::addr_to_normalized_string(request->remote_endpoint().address());
      BOOST_LOG(warning) << "Web UI: ["sv << address << "] -- rejected cross-origin "sv
                         << request->method << ' ' << request->path;
      response->write(SimpleWeb::StatusCode::client_error_forbidden);
      return false;
    }

    return true;
  }

  /**
   * @brief Authenticate the request.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   * @param needsRedirect Whether to redirect on failure.
   * @return True if authenticated, false otherwise.
   *
   * This function uses session cookies (if set) and ensures they have not expired.
   */
  bool authenticate(resp_https_t response, req_https_t request, bool needsRedirect) {
    const auto peer_class = checkIPOrigin(response, request);
    if (!peer_class || !checkBrowserOrigin(response, request)) {
      return false;
    }

    // Local control should feel appliance-like: the network boundary is the trust prompt.
    // Unsafe browser requests still pass the strict same-origin validation above.
    if (*peer_class <= net::LAN) {
      return true;
    }

    // If credentials not set, redirect to welcome.
    if (!credentials_configured()) {
      send_redirect(response, request, "/welcome");
      return false;
    }
    // Guard: on failure, redirect if requested.
    auto fg = util::fail_guard([&]() {
      if (needsRedirect) {
        std::string redir_path = "/login?redir=.";
        redir_path += request->path;
        send_redirect(response, request, redir_path.c_str());
      } else {
        send_unauthorized(response, request);
      }
    });
    auto cookies = request->header.find("cookie");
    if (cookies == request->header.end()) {
      return false;
    }
    auto authCookie = getCookieValue(cookies->second, "auth");
    if (authCookie.empty()) {
      return false;
    }
    {
      std::lock_guard lock {auth_mutex};
      if (sessionCookie.empty()) {
        return false;
      }
      // Check for expiry
      if (std::chrono::steady_clock::now() - cookie_creation_time > SESSION_EXPIRE_DURATION) {
        sessionCookie.clear();
        return false;
      }
      if (util::hex(crypto::hash(authCookie + config::sunshine.salt)).to_string() != sessionCookie) {
        return false;
      }
    }
    fg.disable();
    return true;
  }

  /**
   * @brief Send a 404 Not Found response.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   */
  void not_found(resp_https_t response, [[maybe_unused]] req_https_t request) {
    constexpr SimpleWeb::StatusCode code = SimpleWeb::StatusCode::client_error_not_found;
    nlohmann::json tree;
    tree["status_code"] = static_cast<int>(code);
    tree["error"] = "Not Found";
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "application/json");
    add_frame_guard(headers);

    response->write(code, tree.dump(), headers);
  }

  std::string make_bad_request_body(std::string_view error_message) {
    nlohmann::json tree;
    tree["status_code"] = static_cast<int>(SimpleWeb::StatusCode::client_error_bad_request);
    tree["status"] = false;
    tree["error"] = error_message;
    return tree.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
  }

  /**
   * @brief Send a 400 Bad Request response.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   * @param error_message The error message.
   */
  void bad_request(resp_https_t response, [[maybe_unused]] req_https_t request, std::string_view error_message) {
    constexpr SimpleWeb::StatusCode code = SimpleWeb::StatusCode::client_error_bad_request;
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "application/json");
    add_frame_guard(headers);

    response->write(code, make_bad_request_body(error_message), headers);
  }

  /**
   * @brief Validate the request content type and send bad request when mismatch.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   * @param contentType The required content type.
   */
  bool validateContentType(resp_https_t response, req_https_t request, const std::string_view &contentType) {
    auto requestContentType = request->header.find("content-type");
    if (requestContentType == request->header.end()) {
      bad_request(response, request, "Content type not provided");
      return false;
    }

    // Extract the media type part before any parameters (e.g., charset)
    std::string actualContentType = requestContentType->second;
    size_t semicolonPos = actualContentType.find(';');
    if (semicolonPos != std::string::npos) {
      actualContentType = actualContentType.substr(0, semicolonPos);
    }

    // Trim whitespace and convert to lowercase for case-insensitive comparison
    boost::algorithm::trim(actualContentType);
    boost::algorithm::to_lower(actualContentType);

    std::string expectedContentType(contentType);
    boost::algorithm::to_lower(expectedContentType);

    if (actualContentType != expectedContentType) {
      bad_request(response, request, "Content type mismatch");
      return false;
    }
    return true;
  }

  /**
   * @brief Update existing credentials.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * The body for the POST request should be JSON serialized in the following format:
   * @code{.json}
   * {
   *   "currentUsername": "Current Username",
   *   "currentPassword": "Current Password",
   *   "newUsername": "New Username",
   *   "newPassword": "New Password",
   *   "confirmNewPassword": "Confirm New Password"
   * }
   * @endcode
   *
   * @api_examples{/api/password| POST| {"currentUsername":"admin","currentPassword":"admin","newUsername":"admin","newPassword":"admin","confirmNewPassword":"admin"}}
   */
  void savePassword(resp_https_t response, req_https_t request) {
    if (!credentials_configured()) {
      if (!checkIPOrigin(response, request) || !checkBrowserOrigin(response, request)) {
        return;
      }
    } else if (!authenticate(response, request)) {
      return;
    }
    if (!validateContentType(response, request, "application/json")) {
      return;
    }
    print_req(request);
    std::vector<std::string> errors;
    try {
      nlohmann::json input_tree = nlohmann::json::parse(request->content);
      nlohmann::json output_tree;
      std::string username = input_tree.value("currentUsername", "");
      std::string newUsername = input_tree.value("newUsername", "");
      std::string password = input_tree.value("currentPassword", "");
      std::string newPassword = input_tree.value("newPassword", "");
      std::string confirmPassword = input_tree.value("confirmNewPassword", "");
      if (newUsername.empty()) {
        newUsername = username;
      }
      if (newUsername.empty()) {
        errors.push_back("Invalid Username");
      } else {
        std::lock_guard lock {auth_mutex};
        auto hash = util::hex(crypto::hash(password + config::sunshine.salt)).to_string();
        if (config::sunshine.username.empty() || (boost::iequals(username, config::sunshine.username) && hash == config::sunshine.password)) {
          if (newPassword.empty() || newPassword != confirmPassword) {
            errors.push_back("Password Mismatch");
          } else {
            http::save_user_creds(config::sunshine.credentials_file, newUsername, newPassword);
            http::reload_user_creds(config::sunshine.credentials_file);
            sessionCookie.clear();  // force re-login
            output_tree["status"] = true;
          }
        } else {
          errors.push_back("Invalid Current Credentials");
        }
      }
      if (!errors.empty()) {
        std::string error = std::accumulate(errors.begin(), errors.end(), std::string(), [](const std::string &a, const std::string &b) {
          return a.empty() ? b : a + ", " + b;
        });
        bad_request(response, request, error);
        return;
      }
      send_response(response, output_tree);
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "SavePassword: "sv << e.what();
      bad_request(response, request, e.what());
    }
  }

  /**
   * @brief Login the user.
   * @param response The HTTP response object.
   * @param request The HTTP request object.
   *
   * The body for the POST request should be JSON serialized in the following format:
   * @code{.json}
   * {
   *   "username": "<username>",
   *   "password": "<password>"
   * }
   * @endcode
   */
  void login(resp_https_t response, req_https_t request) {
    if (!checkIPOrigin(response, request) || !checkBrowserOrigin(response, request) || !validateContentType(response, request, "application/json")) {
      return;
    }

    auto fg = util::fail_guard([&] {
      response->write(SimpleWeb::StatusCode::client_error_unauthorized);
    });

    try {
      nlohmann::json input_tree = nlohmann::json::parse(request->content);
      std::string username = input_tree.value("username", "");
      std::string password = input_tree.value("password", "");
      std::string sessionCookieRaw = crypto::rand_alphabet(64);
      {
        std::lock_guard lock {auth_mutex};
        std::string hash = util::hex(crypto::hash(password + config::sunshine.salt)).to_string();
        if (!boost::iequals(username, config::sunshine.username) || hash != config::sunshine.password) {
          return;
        }
        sessionCookie = util::hex(crypto::hash(sessionCookieRaw + config::sunshine.salt)).to_string();
        cookie_creation_time = std::chrono::steady_clock::now();
      }
      const SimpleWeb::CaseInsensitiveMultimap headers {
        {"Set-Cookie", make_auth_cookie(sessionCookieRaw)}
      };
      response->write(headers);
      fg.disable();
    } catch (std::exception &e) {
      BOOST_LOG(warning) << "Web UI Login failed: ["sv << net::addr_to_normalized_string(request->remote_endpoint().address())
                         << "]: "sv << e.what();
      response->write(SimpleWeb::StatusCode::server_error_internal_server_error);
      fg.disable();
      return;
    }
  }
}  // namespace confighttp
