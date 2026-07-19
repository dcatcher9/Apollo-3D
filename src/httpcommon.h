/**
 * @file src/httpcommon.h
 * @brief Declarations for common HTTP.
 */
#pragma once

// standard includes
#include <optional>
#include <string>
#include <string_view>

// lib includes
#include <curl/curl.h>

// local includes
#include "network.h"
#include "thread_safe.h"
#include "uuid.h"

namespace http {

  int init();
  int create_creds(const std::string &pkey, const std::string &cert);
  int save_user_creds(
    const std::string &file,
    const std::string &username,
    const std::string &password,
    bool run_our_mouth = false
  );

  int reload_user_creds(const std::string &file);
  bool download_file(const std::string &url, const std::string &file, long ssl_version = CURL_SSLVERSION_TLSv1_2);
  std::string url_escape(const std::string &url);
  std::string url_get_host(const std::string &url);

  /**
   * @brief Validate browser source metadata for an unsafe Web UI request.
   * @details The target is always HTTPS. The Host header must exactly match every supplied
   * source origin, including the effective port. At least one trustworthy browser source
   * signal is required.
   */
  bool web_ui_origin_allowed(
    std::string_view host,
    std::optional<std::string_view> origin,
    std::optional<std::string_view> referer,
    std::optional<std::string_view> sec_fetch_site
  );

  /**
   * @brief Redact credential-bearing request headers before logging.
   */
  std::string_view redact_request_header(std::string_view name, std::string_view value);

  /**
   * @brief Redact credential-bearing query parameters before logging.
   */
  std::string_view redact_query_parameter(std::string_view name, std::string_view value);

  extern std::string unique_id;
  extern uuid_util::uuid_t uuid;
  extern net::net_e origin_web_ui_allowed;

}  // namespace http
