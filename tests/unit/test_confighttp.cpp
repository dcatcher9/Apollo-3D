/**
 * @file tests/unit/test_confighttp.cpp
 * @brief Test pure Web UI authorization and request-boundary helpers.
 */

#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include <src/confighttp.h>

#include "../tests_common.h"

using confighttp::bounded_content_length_e;

TEST(ConfigHttpRequestBoundary, AcceptsMissingAndInRangeContentLength) {
  EXPECT_EQ(
    confighttp::validate_bounded_content_length(std::nullopt, 65536),
    bounded_content_length_e::absent_or_within_limit
  );
  EXPECT_EQ(
    confighttp::validate_bounded_content_length("0", 65536),
    bounded_content_length_e::absent_or_within_limit
  );
  EXPECT_EQ(
    confighttp::validate_bounded_content_length("65536", 65536),
    bounded_content_length_e::absent_or_within_limit
  );
  EXPECT_EQ(
    confighttp::validate_bounded_content_length(" \t42\t ", 65536),
    bounded_content_length_e::absent_or_within_limit
  );
}

TEST(ConfigHttpRequestBoundary, RejectsOversizeContentLength) {
  EXPECT_EQ(
    confighttp::validate_bounded_content_length("65537", 65536),
    bounded_content_length_e::exceeds_limit
  );
  EXPECT_EQ(
    confighttp::validate_bounded_content_length(
      std::to_string(std::numeric_limits<std::size_t>::max()),
      65536
    ),
    bounded_content_length_e::exceeds_limit
  );
}

TEST(ConfigHttpRequestBoundary, RejectsMalformedContentLength) {
  for (const std::string_view value : {
         "",
         " ",
         "-1",
         "+1",
         "1x",
         "1, 1",
         "184467440737095516160",
       }) {
    EXPECT_EQ(
      confighttp::validate_bounded_content_length(value, 65536),
      bounded_content_length_e::invalid
    ) << value;
  }
}

TEST(ConfigHttpRequestBoundary, AppliesOfflineLimitBeforeRouteDispatch) {
  EXPECT_EQ(
    confighttp::WEB_UI_REQUEST_STREAM_MAX_BYTES,
    32ull * 1024ull * 1024ull
  );
  EXPECT_GT(
    confighttp::WEB_UI_REQUEST_STREAM_MAX_BYTES,
    confighttp::OFFLINE_SBS_REQUEST_MAX_BYTES
  );
  EXPECT_EQ(
    confighttp::request_content_length_limit(
      "POST",
      "/api/offline-sbs/jobs"
    ),
    confighttp::OFFLINE_SBS_REQUEST_MAX_BYTES
  );
  EXPECT_EQ(
    confighttp::request_content_length_limit(
      "POST",
      "/api/offline-sbs/jobs/00000000-0000-0000-0000-000000000000/cancel"
    ),
    confighttp::OFFLINE_SBS_REQUEST_MAX_BYTES
  );
  EXPECT_EQ(
    confighttp::request_content_length_limit(
      "POST",
      "/api/covers/upload"
    ),
    confighttp::WEB_UI_REQUEST_STREAM_MAX_BYTES
  );
  EXPECT_EQ(
    confighttp::request_content_length_limit(
      "GET",
      "/api/offline-sbs/jobs"
    ),
    confighttp::WEB_UI_REQUEST_STREAM_MAX_BYTES
  );
}
