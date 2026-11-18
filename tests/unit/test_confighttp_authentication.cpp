/**
 * @file tests/unit/test_confighttp_authentication.cpp
 * @brief Exercise real Web UI authentication with fixture-owned HTTPS request objects.
 */
#include "../tests_common.h"
#include "src/httpcommon.h"
#include "src/network.h"

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/make_shared.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <Simple-Web-Server/server_https.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace confighttp {
  bool authenticate(
    std::shared_ptr<SimpleWeb::ServerBase<SimpleWeb::HTTPS>::Response> response,
    std::shared_ptr<SimpleWeb::ServerBase<SimpleWeb::HTTPS>::Request> request,
    bool needs_redirect
  );
  void getLoginPage(
    std::shared_ptr<SimpleWeb::ServerBase<SimpleWeb::HTTPS>::Response> response,
    std::shared_ptr<SimpleWeb::ServerBase<SimpleWeb::HTTPS>::Request> request
  );
}  // namespace confighttp

namespace {
  using https_base_t = SimpleWeb::ServerBase<SimpleWeb::HTTPS>;
  using tcp = boost::asio::ip::tcp;

  // Only the TCP endpoint is connected. No server start, certificate, TLS handshake, or HTTP
  // transport is needed: the library's protected factory constructs its real Request/Response.
  class auth_request_t: public https_base_t {
  public:
    explicit auth_request_t(bool loopback):
        https_base_t(0),
        tls_(boost::asio::ssl::context::tls_server),
        client_(io_) {
      connection_ = create_connection(io_, tls_);
      if (loopback) {
        connect_loopback();
      }
      session_ = std::make_shared<Session>(4096, connection_);
      std::function<void(std::shared_ptr<Response>, std::shared_ptr<Request>)> capture =
        [this](auto response, auto request) {
          response_ = std::move(response);
          request_ = std::move(request);
        };
      write(session_, capture);
      if (!response_ || !request_) {
        throw std::runtime_error("Unable to construct the HTTPS authentication request fixture");
      }
      request_->method = "GET";
      request_->path = "/api/test";
    }

    ~auth_request_t() override {
      // Response's factory deleter posts an SSL send. Stop admission before releasing it, then
      // drain that callback; it exits at continue_lock() without starting transport I/O.
      handler_runner->stop();
      response_.reset();
      io_.restart();
      io_.poll();
    }

    std::string output() {
      std::ostringstream result;
      result << response_->rdbuf();
      return result.str();
    }

    std::shared_ptr<Response> response_;
    std::shared_ptr<Request> request_;

  private:
    void accept() override {
    }

    void connect_loopback() {
      tcp::acceptor acceptor(io_, {boost::asio::ip::address_v4::loopback(), 0});
      boost::asio::steady_timer deadline(io_, std::chrono::seconds {3});
      boost::system::error_code failure;
      unsigned completed = 0;
      const auto finish = [&](const boost::system::error_code &error) {
        if (error && !failure) {
          failure = error;
          boost::system::error_code ignored;
          acceptor.close(ignored);
          client_.close(ignored);
          connection_->socket->lowest_layer().close(ignored);
        }
        if (++completed == 2) {
          deadline.cancel();
        }
      };
      deadline.async_wait([&](const boost::system::error_code &error) {
        if (!error) {
          failure = boost::asio::error::timed_out;
          boost::system::error_code ignored;
          acceptor.close(ignored);
          client_.close(ignored);
          connection_->socket->lowest_layer().close(ignored);
        }
      });
      acceptor.async_accept(connection_->socket->lowest_layer(), finish);
      client_.async_connect(acceptor.local_endpoint(), finish);
      io_.run();
      if (failure) {
        throw boost::system::system_error(failure);
      }
    }

    boost::asio::io_context io_;
    boost::asio::ssl::context tls_;
    tcp::socket client_;
    std::shared_ptr<Connection> connection_;
    std::shared_ptr<Session> session_;
  };

  class auth_log_capture_t {
  public:
    auth_log_capture_t():
        output_(boost::make_shared<std::ostringstream>()),
        sink_(boost::make_shared<sink_t>()) {
      sink_->locked_backend()->add_stream(output_);
      sink_->set_formatter(boost::log::expressions::stream << boost::log::expressions::smessage);
      boost::log::core::get()->add_sink(sink_);
    }

    ~auth_log_capture_t() {
      boost::log::core::get()->remove_sink(sink_);
    }

    std::string output() const {
      return output_->str();
    }

  private:
    using sink_t = boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>;
    boost::shared_ptr<std::ostringstream> output_;
    boost::shared_ptr<sink_t> sink_;
  };

  class ConfigHttpAuthenticationTest: public testing::Test {
  protected:
    void SetUp() override {
      config::sunshine.username.clear();
      http::origin_web_ui_allowed = net::LAN;
    }

    ~ConfigHttpAuthenticationTest() override {
      config::sunshine.username = std::move(original_username_);
      http::origin_web_ui_allowed = original_origin_;
    }

    void set_username(bool configured) {
      config::sunshine.username = configured ? "fixture-admin" : "";
    }

    static void expect_redirect(const std::string &output, const char *location) {
      EXPECT_TRUE(output.starts_with("HTTP/1.1 307 Temporary Redirect\r\n")) << output;
      EXPECT_NE(output.find(std::string {"Location: "} + location + "\r\n"), std::string::npos) << output;
    }

  private:
    std::string original_username_ = config::sunshine.username;
    net::net_e original_origin_ = http::origin_web_ui_allowed;
  };
}  // namespace

TEST_F(ConfigHttpAuthenticationTest, LoopbackPcZeroBypassesMissingAndConfiguredCredentials) {
  static_assert(net::PC == 0);
  for (const bool configured : {false, true}) {
    SCOPED_TRACE(configured);
    set_username(configured);
    auth_request_t request(true);
    ASSERT_EQ(request.request_->remote_endpoint().address(), boost::asio::ip::address_v4::loopback());
    EXPECT_TRUE(confighttp::authenticate(request.response_, request.request_, false));
    EXPECT_TRUE(request.output().empty());
  }
}

TEST_F(ConfigHttpAuthenticationTest, UnsafeBrowserOriginIsRejectedBeforeTheLocalBypass) {
  for (const bool configured : {false, true}) {
    SCOPED_TRACE(configured);
    set_username(configured);
    auth_request_t request(true);
    request.request_->method = "POST";
    request.request_->header.emplace("Host", "apollo.test:47990");
    request.request_->header.emplace("Origin", "https://other.test:47990");
    request.request_->header.emplace("Sec-Fetch-Site", "cross-site");
    EXPECT_FALSE(confighttp::authenticate(request.response_, request.request_, false));
    const auto output = request.output();
    EXPECT_TRUE(output.starts_with("HTTP/1.1 403 Forbidden\r\n")) << output;
    EXPECT_EQ(output.find("Location:"), std::string::npos);
  }
}

TEST_F(ConfigHttpAuthenticationTest, SameOriginPostStillUsesTheLocalBypass) {
  for (const bool configured : {false, true}) {
    SCOPED_TRACE(configured);
    set_username(configured);
    auth_request_t request(true);
    request.request_->method = "POST";
    request.request_->header.emplace("Host", "apollo.test:47990");
    request.request_->header.emplace("Origin", "https://apollo.test:47990");
    request.request_->header.emplace("Sec-Fetch-Site", "same-origin");
    EXPECT_TRUE(confighttp::authenticate(request.response_, request.request_, false));
    EXPECT_TRUE(request.output().empty());
  }
}

TEST_F(ConfigHttpAuthenticationTest, FailedEndpointIpDenialPrecedesBrowserOriginChecking) {
  auth_request_t request(false);
  ASSERT_TRUE(request.request_->remote_endpoint().address().is_unspecified());
  request.request_->method = "POST";
  request.request_->header.emplace("Host", "apollo.test:47990");
  request.request_->header.emplace("Origin", "https://other.test:47990");
  request.request_->header.emplace("Sec-Fetch-Site", "cross-site");
  auth_log_capture_t capture;
  EXPECT_FALSE(confighttp::authenticate(request.response_, request.request_, false));
  EXPECT_TRUE(request.output().starts_with("HTTP/1.1 403 Forbidden\r\n"));
  EXPECT_NE(capture.output().find("-- denied"), std::string::npos);
  EXPECT_EQ(capture.output().find("rejected cross-origin"), std::string::npos);
}

TEST_F(ConfigHttpAuthenticationTest, AllowedWanWithoutCookieKeepsTheCredentialResponses) {
  http::origin_web_ui_allowed = net::WAN;
  {
    auth_request_t request(false);
    EXPECT_FALSE(confighttp::authenticate(request.response_, request.request_, false));
    expect_redirect(request.output(), "/welcome");
  }
  set_username(true);
  for (const bool redirect : {false, true}) {
    SCOPED_TRACE(redirect);
    auth_request_t request(false);
    EXPECT_FALSE(confighttp::authenticate(request.response_, request.request_, redirect));
    const auto output = request.output();
    if (redirect) {
      expect_redirect(output, "/login?redir=./api/test");
    } else {
      EXPECT_TRUE(output.starts_with("HTTP/1.1 401 Unauthorized\r\n")) << output;
      EXPECT_NE(output.find("\"error\":\"Unauthorized\""), std::string::npos);
    }
  }
}

TEST_F(ConfigHttpAuthenticationTest, LoginPageRedirectsLocalPeersAndUnconfiguredWanPeers) {
  for (const bool configured : {false, true}) {
    SCOPED_TRACE(configured);
    set_username(configured);
    auth_request_t request(true);
    request.request_->path = "/login";
    confighttp::getLoginPage(request.response_, request.request_);
    expect_redirect(request.output(), "/");
  }
  set_username(false);
  http::origin_web_ui_allowed = net::WAN;
  auth_request_t request(false);
  request.request_->path = "/login";
  confighttp::getLoginPage(request.response_, request.request_);
  expect_redirect(request.output(), "/welcome");
}
