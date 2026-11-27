/**
 * @file src/confighttp_common.h
 * @brief Request guards and responses shared by the Web UI routes.
 */
#pragma once

// standard includes
#include <memory>
#include <optional>
#include <string>
#include <string_view>

// lib includes
#include <nlohmann/json_fwd.hpp>
#include <Simple-Web-Server/server_https.hpp>

// local includes
#include "network.h"

namespace confighttp {
  using resp_https_t = std::shared_ptr<typename SimpleWeb::ServerBase<SimpleWeb::HTTPS>::Response>;
  using req_https_t = std::shared_ptr<typename SimpleWeb::ServerBase<SimpleWeb::HTTPS>::Request>;

  /** A header that may appear at most once. `unique` is false when the request repeats it. */
  struct request_header_t {
    bool unique;
    std::optional<std::string_view> value;
  };

  request_header_t get_single_request_header(const req_https_t &request, std::string_view name);

  // Responses. Every Web UI reply carries the frame guard.
  void add_frame_guard(SimpleWeb::CaseInsensitiveMultimap &headers);
  void send_response(resp_https_t response, const nlohmann::json &output_tree);
  void send_json_response(resp_https_t response, SimpleWeb::StatusCode status, const nlohmann::json &body);
  void send_unauthorized(resp_https_t response, req_https_t request);
  void send_redirect(resp_https_t response, req_https_t request, const char *path);
  void not_found(resp_https_t response, req_https_t request);
  void bad_request(resp_https_t response, req_https_t request, std::string_view error_message = "Bad Request");
  bool validateContentType(resp_https_t response, req_https_t request, const std::string_view &contentType);
  void print_req(const req_https_t &request);

  // Access checks. A check that refuses has already sent its denial response.
  std::optional<net::net_e> checkIPOrigin(resp_https_t response, req_https_t request);
  bool checkBrowserOrigin(resp_https_t response, req_https_t request);
  bool credentials_configured();
  bool authenticate(resp_https_t response, req_https_t request, bool needsRedirect = false);
}  // namespace confighttp
