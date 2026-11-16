/**
 * @file tests/unit/test_httpcommon.cpp
 * @brief Test src/httpcommon.*.
 */
// test imports
#include "../tests_common.h"

// standard imports
#include <array>
#include <sstream>
#include <string_view>
#include <utility>

// lib imports
#include <boost/log/expressions.hpp>
#include <boost/make_shared.hpp>
#include <curl/curl.h>

// local imports
#include <src/confighttp.h>
#include <src/http_request_logging.h>
#include <src/httpcommon.h>
#include <src/utility.h>

struct UrlEscapeTest: testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(UrlEscapeTest, Run) {
  const auto &[input, expected] = GetParam();
  ASSERT_EQ(http::url_escape(input), expected);
}

INSTANTIATE_TEST_SUITE_P(
  UrlEscapeTests,
  UrlEscapeTest,
  testing::Values(
    std::make_tuple("igdb_0123456789", "igdb_0123456789"),
    std::make_tuple("../../../", "..%2F..%2F..%2F"),
    std::make_tuple("..*\\", "..%2A%5C")
  )
);

struct UrlGetHostTest: testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(UrlGetHostTest, Run) {
  const auto &[input, expected] = GetParam();
  ASSERT_EQ(http::url_get_host(input), expected);
}

INSTANTIATE_TEST_SUITE_P(
  UrlGetHostTests,
  UrlGetHostTest,
  testing::Values(
    std::make_tuple("https://images.igdb.com/example.txt", "images.igdb.com"),
    std::make_tuple("http://localhost:8080", "localhost"),
    std::make_tuple("nonsense!!}{::", "")
  )
);

namespace {
  using optional_string_view = std::optional<std::string_view>;

  optional_string_view header(const std::string_view value) {
    return value;
  }
}  // namespace

TEST(WebUiOriginTest, AcceptsExactHttpsOrigins) {
  EXPECT_TRUE(http::web_ui_origin_allowed(
    "apollo.local:47990",
    header("HTTPS://APOLLO.LOCAL:47990"),
    std::nullopt,
    header("same-origin")
  ));
  EXPECT_TRUE(http::web_ui_origin_allowed(
    "192.168.1.10:47990",
    header("https://192.168.1.10:47990"),
    std::nullopt,
    std::nullopt
  ));
  EXPECT_TRUE(http::web_ui_origin_allowed(
    "[2001:0db8:0:0::1]:47990",
    header("https://[2001:db8::1]:47990"),
    std::nullopt,
    std::nullopt
  ));
  EXPECT_TRUE(http::web_ui_origin_allowed(
    "apollo.local:47990",
    std::nullopt,
    header("https://apollo.local:47990/config?tab=general"),
    std::nullopt
  ));
}

TEST(WebUiOriginTest, NormalizesDefaultHttpsPort) {
  EXPECT_TRUE(http::web_ui_origin_allowed(
    "apollo.local",
    header("https://apollo.local:443"),
    std::nullopt,
    std::nullopt
  ));
  EXPECT_TRUE(http::web_ui_origin_allowed(
    "apollo.local:443",
    header("https://apollo.local"),
    std::nullopt,
    std::nullopt
  ));
}

TEST(WebUiOriginTest, AcceptsSameOriginFetchMetadataWithoutUrlHeaders) {
  EXPECT_TRUE(http::web_ui_origin_allowed(
    "apollo.local:47990",
    std::nullopt,
    std::nullopt,
    header("same-origin")
  ));
}

TEST(WebUiOriginTest, RejectsMissingOrCrossSiteSourceMetadata) {
  EXPECT_FALSE(http::web_ui_origin_allowed(
    "apollo.local:47990",
    std::nullopt,
    std::nullopt,
    std::nullopt
  ));
  for (const auto value : {"same-site", "cross-site", "none", "unknown"}) {
    EXPECT_FALSE(http::web_ui_origin_allowed(
      "apollo.local:47990",
      std::nullopt,
      std::nullopt,
      header(value)
    ));
  }
}

TEST(WebUiOriginTest, RejectsOriginTupleMismatches) {
  for (const auto value : {
         "http://apollo.local:47990",
         "https://evil-apollo.local:47990",
         "https://apollo.local.evil.example:47990",
         "https://apollo.local:47991",
         "null",
         "https://user@apollo.local:47990",
         "https://apollo.local:47990/path",
       }) {
    SCOPED_TRACE(value);
    EXPECT_FALSE(http::web_ui_origin_allowed(
      "apollo.local:47990",
      header(value),
      std::nullopt,
      std::nullopt
    ));
  }
}

TEST(WebUiOriginTest, RejectsConflictingOriginAndReferer) {
  EXPECT_FALSE(http::web_ui_origin_allowed(
    "apollo.local:47990",
    header("https://evil.example:47990"),
    header("https://apollo.local:47990/config"),
    header("same-origin")
  ));
  EXPECT_FALSE(http::web_ui_origin_allowed(
    "apollo.local:47990",
    header("https://apollo.local:47990"),
    header("https://evil.example:47990/config"),
    header("same-origin")
  ));
}

TEST(WebUiOriginTest, RejectsMalformedHosts) {
  for (const auto value : {
         "",
         "apollo.local/evil",
         "user@apollo.local",
         "2001:db8::1",
         "apollo.local:",
         "apollo.local:0",
         "apollo.local:+443",
         "apollo.local:invalid",
         "apollo.local:65536",
         "apollo.local,evil.example",
       }) {
    SCOPED_TRACE(value);
    EXPECT_FALSE(http::web_ui_origin_allowed(
      value,
      header("https://apollo.local"),
      std::nullopt,
      std::nullopt
    ));
  }
}

TEST(HttpRequestLoggingTest, RedactsCredentialHeadersCaseInsensitively) {
  for (const auto name : {
         "Authorization",
         "authorization",
         "PrOxY-AuThOrIzAtIoN",
         "Cookie",
         "Set-Cookie",
         "X-CSRF-Token",
       }) {
    EXPECT_EQ(http::redact_request_header(name, "top-secret"), "CREDENTIALS REDACTED");
  }
  EXPECT_EQ(http::redact_request_header("User-Agent", "Artemis"), "Artemis");
}

TEST(HttpRequestLoggingTest, RedactsGameStreamSecretsCaseInsensitively) {
  for (const auto name : {
         "rikey",
         "RIKEYID",
         "salt",
         "otpauth",
         "clientchallenge",
         "serverchallengeresp",
         "clientpairingsecret",
         "clientcert",
         "csrf_token",
       }) {
    EXPECT_EQ(http::redact_query_parameter(name, "top-secret"), "CREDENTIALS REDACTED");
  }
  EXPECT_EQ(http::redact_query_parameter("mode", "1920x1080x60"), "1920x1080x60");
}

namespace {
  struct request_log_probe_t {
    std::string_view method = "GET";
    std::string_view path = "/launch";
    std::array<std::pair<std::string_view, std::string_view>, 2> header {{
      {"Cookie", "header-secret"},
      {"User-Agent", "Artemis"},
    }};
    mutable unsigned query_parses = 0;

    auto parse_query_string() const {
      ++query_parses;
      return std::array<std::pair<std::string_view, std::string_view>, 2> {{
        {"rikey", "query-secret"},
        {"mode", "1920x1080x60"},
      }};
    }
  };

  class request_log_capture_t {
  public:
    explicit request_log_capture_t(int minimum_severity):
        output_(boost::make_shared<std::ostringstream>()),
        sink_(boost::make_shared<sink_t>()) {
      sink_->locked_backend()->add_stream(output_);
      sink_->set_formatter(boost::log::expressions::stream << boost::log::expressions::smessage);
      // The native harness also has a diagnostic sink accepting every level. Apply the
      // temporary admission filter at the core so that sink cannot admit rejected debug lines.
      boost::log::core::get()->set_filter(boost::log::expressions::attr<int>("Severity") >= minimum_severity);
      boost::log::core::get()->add_sink(sink_);
    }

    ~request_log_capture_t() {
      boost::log::core::get()->remove_sink(sink_);
      boost::log::core::get()->reset_filter();
    }

    std::string output() const {
      return output_->str();
    }

  private:
    using sink_t = boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>;
    boost::shared_ptr<std::ostringstream> output_;
    boost::shared_ptr<sink_t> sink_;
  };
}  // namespace

TEST(HttpRequestLoggingTest, RejectedDebugDoesNotParseQueriesOrEmitAdmissionRecords) {
  const auto old_level = config::sunshine.min_log_level;
  const auto restore_level = util::fail_guard([old_level]() {
    config::sunshine.min_log_level = old_level;
  });
  config::sunshine.min_log_level = debug.default_severity();
  request_log_capture_t capture(debug.default_severity() + 1);
  request_log_probe_t request;

  http::log_request_debug(request);
  http::log_request_debug(request, "HTTPS");

  EXPECT_EQ(request.query_parses, 0u);
  EXPECT_TRUE(capture.output().empty());
}

TEST(HttpRequestLoggingTest, AdmittedDebugParsesOnceAndPreservesRedactedLinesAndTunnel) {
  const auto old_level = config::sunshine.min_log_level;
  const auto restore_level = util::fail_guard([old_level]() {
    config::sunshine.min_log_level = old_level;
  });
  config::sunshine.min_log_level = debug.default_severity() + 1;
  const std::string expected =
    "METHOD :: GET\n"
    "DESTINATION :: /launch\n"
    "Cookie -- CREDENTIALS REDACTED\n"
    "User-Agent -- Artemis\n"
    " [--] \n"
    "rikey -- CREDENTIALS REDACTED\n"
    "mode -- 1920x1080x60\n"
    " [--] \n";

  for (const auto tunnel : std::array<std::optional<std::string_view>, 3> {std::nullopt, "NONE", "HTTPS"}) {
    SCOPED_TRACE(tunnel.value_or("Web UI"));
    request_log_capture_t capture(debug.default_severity());
    request_log_probe_t request;

    http::log_request_debug(request, tunnel);

    EXPECT_EQ(request.query_parses, 1u);
    const auto prefix = tunnel ? "TUNNEL :: " + std::string(*tunnel) + "\n" : std::string {};
    EXPECT_EQ(capture.output(), prefix + expected);
  }
}

TEST(AuthCookieTest, UsesBrowserHardeningAndServerLifetime) {
  EXPECT_EQ(
    confighttp::make_auth_cookie("session-token"),
    "auth=session-token; Secure; HttpOnly; SameSite=Strict; Max-Age=1296000; Path=/"
  );
}

struct DownloadFileTest: testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(DownloadFileTest, Run) {
  const auto &[url, filename] = GetParam();
  const std::string test_dir = platf::appdata().string() + "/tests/";
  std::string path = test_dir + filename;
  ASSERT_TRUE(http::download_file(url, path, CURL_SSLVERSION_TLSv1_0));
}

#ifdef SUNSHINE_BUILD_FLATPAK
// requires running `npm run serve` prior to running the tests
constexpr const char *URL_1 = "http://0.0.0.0:3000/hello.txt";
constexpr const char *URL_2 = "http://0.0.0.0:3000/hello-redirect.txt";
#else
constexpr const char *URL_1 = "https://httpbin.org/base64/aGVsbG8h";
constexpr const char *URL_2 = "https://httpbin.org/redirect-to?url=/base64/aGVsbG8h";
#endif

INSTANTIATE_TEST_SUITE_P(
  DownloadFileTests,
  DownloadFileTest,
  testing::Values(
    std::make_tuple(URL_1, "hello.txt"),
    std::make_tuple(URL_2, "hello-redirect.txt")
  )
);
