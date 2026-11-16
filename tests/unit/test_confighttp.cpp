/**
 * @file tests/unit/test_confighttp.cpp
 * @brief Test pure Web UI authorization and request-boundary helpers.
 */

#include <array>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>
#include <src/confighttp.h>

#include "../tests_common.h"

using confighttp::bounded_content_length_e;

TEST(ConfigHttpErrorResponse, PreservesValidMessagesAndWireFields) {
  const std::array<std::string, 4> messages {
    "Bad Request",
    "quotes: \"value\" and \\ and control\r\n\t",
    "UTF-8: caf\xc3\xa9 \xe2\x98\x83",
    std::string {"embedded\0nul", 12},
  };
  for (const auto &message : messages) {
    SCOPED_TRACE(message);
    const nlohmann::json original {
      {"status_code", 400},
      {"status", false},
      {"error", message}
    };
    EXPECT_EQ(confighttp::make_bad_request_body(message), original.dump());
  }
}

TEST(ConfigHttpErrorResponse, ReplacesInvalidUtf8WithoutLosingReadableText) {
  struct case_t {
    const char *name;
    std::string message;
    std::string expected;
  };

  const std::array cases {
    case_t {"invalid lead", std::string {"before"} + static_cast<char>(0xff) + "after",
            "before"
            "\xef\xbf\xbd"
            "after"},
    case_t {"invalid continuation", "before\xc3(after",
            "before"
            "\xef\xbf\xbd"
            "(after"},
    case_t {"truncated sequence", "before\xe2\x98",
            "before"
            "\xef\xbf\xbd"},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    const auto body = nlohmann::json::parse(confighttp::make_bad_request_body(test.message));
    EXPECT_EQ(body, (nlohmann::json {{"status_code", 400}, {"status", false}, {"error", test.expected}}));
  }
}

TEST(ConfigHttpErrorResponse, SerializesTheRealMalformedInputDiagnostic) {
  try {
    const auto unexpected = nlohmann::json::parse("{\"value\":\"\xc3(\"}");
    FAIL() << "Malformed UTF-8 unexpectedly parsed: " << unexpected;
  } catch (const nlohmann::json::parse_error &error) {
    const nlohmann::json original {
      {"status_code", 400},
      {"status", false},
      {"error", error.what()}
    };
    // The former serializer throws while already handling a request parse failure.
    EXPECT_THROW(original.dump(), nlohmann::json::type_error);
    const auto body = nlohmann::json::parse(confighttp::make_bad_request_body(error.what()));
    EXPECT_EQ(body.at("status_code"), 400);
    EXPECT_EQ(body.at("status"), false);
    const auto &message = body.at("error").get_ref<const std::string &>();
    EXPECT_TRUE(message.starts_with("[json.exception.parse_error.101]"));
    EXPECT_NE(message.find("\xef\xbf\xbd"), std::string::npos);
  }
}

TEST(ConfigHttpMigration, RetiredVirtualDisplayOnlyCannotBeSaved) {
  nlohmann::json payload {
    {"sunshine_name", "XR host"},
    {"virtual_display_only", "on"},
    {"virtual_display_restart_explorer", "enabled"},
  };

  confighttp::erase_retired_config_options(payload);

  EXPECT_FALSE(payload.contains("virtual_display_only"));
  EXPECT_EQ(payload.at("sunshine_name"), "XR host");
  EXPECT_EQ(payload.at("virtual_display_restart_explorer"), "enabled");
}

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
      "/api/offline-sbs/browse"
    ),
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
      "DELETE",
      "/api/offline-sbs/jobs/00000000-0000-0000-0000-000000000000"
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
