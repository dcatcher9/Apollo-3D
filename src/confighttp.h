/**
 * @file src/confighttp.h
 * @brief Declarations for the Web UI Config HTTP server.
 */
#pragma once

// standard includes
#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

// local includes
#include "thread_safe.h"

#define WEB_DIR SUNSHINE_ASSETS_DIR "/web/"

using namespace std::chrono_literals;

namespace confighttp {
  constexpr auto PORT_HTTPS = 1;
  constexpr auto SESSION_EXPIRE_DURATION = 24h * 15;

  constexpr std::size_t OFFLINE_SBS_REQUEST_MAX_BYTES = 64ull * 1024ull;
  // Preserve existing app/config and cover-upload payloads while placing a finite transport
  // ceiling on the Web UI. A header-time route selector applies the stricter offline limit before
  // declared-body transfer or cumulative chunk acceptance.
  constexpr std::size_t WEB_UI_REQUEST_STREAM_MAX_BYTES =
    32ull * 1024ull * 1024ull;

  enum class bounded_content_length_e {
    absent_or_within_limit,
    invalid,
    exceeds_limit,
  };

  [[nodiscard]] bounded_content_length_e validate_bounded_content_length(
    std::optional<std::string_view> value,
    std::size_t maximum
  );

  [[nodiscard]] std::size_t request_content_length_limit(
    std::string_view method,
    std::string_view path
  );

  std::string make_auth_cookie(std::string_view token);
  void start();
}  // namespace confighttp

// mime types map
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
