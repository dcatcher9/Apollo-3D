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
#include <src/confighttp_routes.h>
#include <src/platform/windows/game3d_setup.h>

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

TEST(ConfigHttpMigration, RetiredOptionCannotBeSaved) {
  nlohmann::json payload {
    {"sunshine_name", "XR host"},
    {"sbs_3d_foreground_subtitles", "enabled"},
    {"virtual_display_restart_explorer", "enabled"},
  };

  confighttp::erase_retired_config_options(payload);

  EXPECT_FALSE(payload.contains("sbs_3d_foreground_subtitles"));
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

namespace {
  using setup_status_t = platf::game3d::setup_status_t;

  setup_status_t::card_t addon_offer() {
    return {
      .game_name = "Example",
      .executable = "C:/Games/Example/game.exe",
      .kind = platf::game3d::game_kind_e::addon,
      .text_helper = false,
      .chosen = false,
      .state = "ready_to_install",
      .detail = "Confirm this game.",
      .reshade_present = true,
      .install_script = "C:/Host/tools/reshade/install.ps1",
      .runtime_install_script = "C:/Host/tools/reshade/install-runtime.ps1",
      .identity = "0011",
    };
  }
}  // namespace

TEST(ConfigHttpGame3dSetup, UndetectedCardCarriesOnlyActivityAndChoosableApplications) {
  setup_status_t status;
  EXPECT_EQ(confighttp::game3d_setup_json(status).dump(), R"({"active":false,"choosable":[],"detected":false})");

  status.active = true;
  status.choosable.push_back({"Visual Novel", "C:/Games/Novel/novel.exe", "00aa", true});
  EXPECT_EQ(
    confighttp::game3d_setup_json(status).dump(),
    R"({"active":true,"choosable":[{"executable":"C:/Games/Novel/novel.exe","game_name":"Visual Novel","identity":"00aa","text_helper":true}],"detected":false})"
  );
}

TEST(ConfigHttpGame3dSetup, DetectedCardNamesItsFieldsAndAnOfferIdentityOnly) {
  setup_status_t status;
  status.card = addon_offer();
  EXPECT_EQ(
    confighttp::game3d_setup_json(status).dump(),
    R"({"active":false,"choosable":[],"chosen":false,"detail":"Confirm this game.","detected":true,"executable":"C:/Games/Example/game.exe","game_name":"Example","identity":"0011","install_script":"C:/Host/tools/reshade/install.ps1","kind":"addon","reshade_present":true,"runtime_install_script":"C:/Host/tools/reshade/install-runtime.ps1","state":"ready_to_install","text_helper":false})"
  );

  // A card that shows no offer has nothing to confirm, so it names no identity.
  status.card->identity.reset();
  status.card->state = "current";
  const auto settled = confighttp::game3d_setup_json(status);
  EXPECT_FALSE(settled.contains("identity"));
  EXPECT_EQ(settled.at("state"), "current");
}

TEST(ConfigHttpGame3dSetup, TextProtectionNamesTheGameTextTheFallbackOrNothing) {
  using platf::game3d::text_hook::protection_e;
  setup_status_t status;
  status.card = addon_offer();
  status.card->kind = platf::game3d::game_kind_e::text_hook;
  status.card->text_helper = true;
  status.card->chosen = true;
  status.card->state = "converting";
  EXPECT_EQ(confighttp::game3d_setup_json(status).at("kind"), "text_hook");
  EXPECT_FALSE(confighttp::game3d_setup_json(status).contains("text_protection"));

  for (const auto &[protection, name] : std::array<std::pair<protection_e, std::string_view>, 3> {{
         {protection_e::game_text, "game"},
         {protection_e::picture, "fallback"},
         {protection_e::unknown, ""},
       }}) {
    SCOPED_TRACE(name);
    status.text_protection = setup_status_t::text_protection_t {protection, "why"};
    const auto json = confighttp::game3d_setup_json(status);
    EXPECT_EQ(json.at("text_protection"), name);
    EXPECT_EQ(json.at("text_protection_detail"), "why");
    EXPECT_EQ(json.at("chosen"), true);
    EXPECT_EQ(json.at("text_helper"), true);
  }
}
