/** @file tests/unit/test_http_json_body.cpp */
#include <algorithm>
#include <array>
#include <boost/asio.hpp>
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <Simple-Web-Server/server_http.hpp>
#include <src/confighttp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {
  using http_server_t = SimpleWeb::Server<SimpleWeb::HTTP>;
  using json = nlohmann::json;
  using tcp = boost::asio::ip::tcp;

  class json_body_server_t {
  public:
    json_body_server_t() {
      server.config.port = 0;
      server.config.address = "127.0.0.1";
      server.config.thread_pool_size = 1;
      server.config.max_request_streambuf_size = 128 * 1024;
      // An owned external context stays stopped even if the worker starts after a startup timeout.
      server.io_service = std::make_shared<boost::asio::io_context>();
      server.resource["^/(direct|copied-string|copied-stream)$"]["POST"] =
        [](const auto &response, const auto &request) {
          try {
            json document;
            if (request->path == "/direct") {
              document = json::parse(request->content);
            } else {
              std::stringstream body;
              body << request->content.rdbuf();
              document = request->path == "/copied-string" ?
                           json::parse(body.str()) :
                           json::parse(body);
            }
            response->write(
              SimpleWeb::StatusCode::success_ok,
              json {{"ok", true}, {"document", std::move(document)}}.dump()
            );
          } catch (const json::parse_error &error) {
            // Diagnostics may retain malformed UTF-8 from the input. Transport their exact
            // bytes so JSON serialization cannot hide a parser/framing result behind its own error.
            const std::string_view message {error.what()};
            response->write(
              SimpleWeb::StatusCode::client_error_bad_request,
              json {{"ok", false}, {"error_id", error.id}, {"error_byte", error.byte}, {"error_bytes", std::vector<unsigned char> {message.begin(), message.end()}}}.dump()
            );
          }
        };
      server.resource["^/bad-request$"]["POST"] = [](const auto &response, const auto &request) {
        try {
          const auto document = json::parse(request->content);
          response->write(json {{"document", document}}.dump());
        } catch (const json::parse_error &error) {
          response->write(
            SimpleWeb::StatusCode::client_error_bad_request,
            confighttp::make_bad_request_body(error.what()),
            {{"Content-Type", "application/json"}}
          );
        }
      };
      server.resource["^/ping$"]["GET"] = [](const auto &response, const auto &) {
        response->write(json {{"pong", true}}.dump());
      };

      auto ready = std::make_shared<std::promise<unsigned short>>();
      auto future = ready->get_future();
      thread = std::thread([this, ready]() {
        try {
          server.start([ready](unsigned short assigned_port) {
            ready->set_value(assigned_port);
          });
          server.io_service->run();
        } catch (...) {
          try {
            ready->set_exception(std::current_exception());
          } catch (...) {
          }
        }
      });
      if (future.wait_for(3s) != std::future_status::ready) {
        server.io_service->stop();
        server.stop();
        thread.join();
        throw std::runtime_error("timed out starting HTTP JSON test server");
      }
      try {
        port = future.get();
      } catch (...) {
        server.io_service->stop();
        server.stop();
        thread.join();
        throw;
      }
    }

    ~json_body_server_t() {
      server.io_service->stop();
      server.stop();
      if (thread.joinable()) {
        thread.join();
      }
    }

    http_server_t server;
    std::thread thread;
    unsigned short port = 0;
  };

  struct http_response_t {
    std::string status;
    json body;
  };

  // Every exchange ends with a pipelined Connection: close request. The deadline covers connect,
  // write and reading both responses; cancellation drains before the client objects are destroyed.
  std::vector<http_response_t> exchange_json_requests(unsigned short port, const std::string &request) {
    boost::asio::io_context context;
    tcp::socket socket {context};
    boost::asio::steady_timer deadline {context, 3s};
    std::string wire;
    boost::system::error_code exchange_error;
    bool timed_out = false;
    const auto finish = [&](const boost::system::error_code &error) {
      exchange_error = error;
      deadline.cancel();
    };
    deadline.async_wait([&](const boost::system::error_code &error) {
      if (!error) {
        timed_out = true;
        boost::system::error_code ignored;
        socket.close(ignored);
      }
    });
    socket.async_connect({boost::asio::ip::make_address_v4("127.0.0.1"), port}, [&](const boost::system::error_code &error) {
      if (error) {
        finish(error);
        return;
      }
      boost::asio::async_write(socket, boost::asio::buffer(request), [&](const boost::system::error_code &error, std::size_t) {
        if (error) {
          finish(error);
          return;
        }
        boost::asio::async_read(socket, boost::asio::dynamic_buffer(wire, 1024 * 1024), [&](const boost::system::error_code &error, std::size_t) {
          finish(error);
        });
      });
    });
    context.run();
    if (timed_out) {
      throw std::runtime_error("timed out exchanging HTTP JSON requests");
    }
    if (exchange_error != boost::asio::error::eof) {
      throw std::runtime_error("HTTP JSON exchange failed: " + exchange_error.message());
    }

    std::vector<http_response_t> responses;
    std::size_t position = 0;
    while (position < wire.size()) {
      const auto header_end = wire.find("\r\n\r\n", position);
      if (header_end == std::string::npos) {
        throw std::runtime_error("incomplete HTTP JSON response headers");
      }
      std::istringstream headers(wire.substr(position, header_end + 4 - position));
      std::string version;
      http_response_t response;
      SimpleWeb::CaseInsensitiveMultimap fields;
      if (!SimpleWeb::ResponseMessage::parse(headers, version, response.status, fields)) {
        throw std::runtime_error("invalid HTTP JSON response headers");
      }
      const auto length = fields.find("Content-Length");
      if (length == fields.end()) {
        throw std::runtime_error("HTTP JSON response lacks Content-Length");
      }
      position = header_end + 4;
      const auto size = std::stoull(length->second);
      if (size > wire.size() - position) {
        throw std::runtime_error("incomplete HTTP JSON response body");
      }
      response.body = json::parse(wire.substr(position, size));
      responses.push_back(std::move(response));
      position += size;
    }
    return responses;
  }

  std::string pipelined_request(std::string_view path, const std::string &body, bool chunked) {
    std::string request = "POST " + std::string {path} + " HTTP/1.1\r\nHost: localhost\r\n";
    if (chunked) {
      request += "Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n";
      for (std::size_t position = 0; position < body.size();) {
        const auto size = std::min<std::size_t>(1021, body.size() - position);
        std::ostringstream length;
        length << std::hex << size;
        request += length.str() + "\r\n";
        request.append(body, position, size);
        request += "\r\n";
        position += size;
      }
      request += "0\r\n\r\n";
    } else {
      request += "Content-Length: " + std::to_string(body.size()) + "\r\nConnection: keep-alive\r\n\r\n";
      request += body;
    }
    request += "GET /ping HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    return request;
  }

  void expect_parser_parity(json_body_server_t &server, const std::string &body, bool valid, bool chunked) {
    SCOPED_TRACE(chunked ? "chunked" : "Content-Length");
    http_response_t copied;
    for (const auto path : {"/copied-string", "/copied-stream", "/direct"}) {
      SCOPED_TRACE(path);
      const auto responses = exchange_json_requests(server.port, pipelined_request(path, body, chunked));
      ASSERT_EQ(responses.size(), 2u);
      EXPECT_EQ(responses[1].status, "200 OK");
      EXPECT_EQ(responses[1].body, (json {{"pong", true}}));
      EXPECT_EQ(responses[0].status, valid ? "200 OK" : "400 Bad Request");
      EXPECT_EQ(responses[0].body.at("ok"), valid);
      if (valid) {
        EXPECT_EQ(responses[0].body.at("document"), json::parse(body));
      } else {
        EXPECT_EQ(responses[0].body.at("error_id"), 101);
        EXPECT_GT(responses[0].body.at("error_byte").get<std::size_t>(), 0u);
      }
      if (std::string_view {path} == "/copied-string") {
        copied = responses[0];
      } else {
        EXPECT_EQ(responses[0].status, copied.status);
        EXPECT_EQ(responses[0].body, copied.body);
      }
    }
  }
}  // namespace

TEST(HttpJsonBody, ActualContentMatchesBothCopiedParsers) {
  json_body_server_t server;

  struct body_case_t {
    const char *name;
    std::string body;
    bool valid;
  };

  std::string raw_nul = "{\"value\":\"before";
  raw_nul.push_back('\0');
  raw_nul += "after\"}";
  const std::array cases {
    body_case_t {"object", R"({"nested":[true,null,{"value":42}],"text":"hello"})", true},
    body_case_t {"scalar", "true", true},
    body_case_t {"trailing whitespace", "{\"value\":7}\r\n\t ", true},
    body_case_t {"UTF-8 and escapes", "{\"utf8\":\"caf\xc3\xa9 \xe2\x98\x83\",\"escaped\":\"\\uD83D\\uDE80\\u0000\"}", true},
    body_case_t {"empty", "", false},
    body_case_t {"malformed", R"({"items":[1,]})", false},
    body_case_t {"truncated JSON", R"({"items":[1,2)", false},
    body_case_t {"trailing input", R"({"value":1}garbage)", false},
    body_case_t {"second document", "{}[]", false},
    body_case_t {"invalid UTF-8", "{\"value\":\"\xc3(\"}", false},
    body_case_t {"raw NUL", raw_nul, false},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    for (const bool chunked : {false, true}) {
      expect_parser_parity(server, test.body, test.valid, chunked);
    }
  }
}

TEST(HttpJsonBody, LargeBodyMatchesBothCopiedParsers) {
  json_body_server_t server;
  const auto body = json {{"text", std::string(70 * 1024, 'x')}}.dump();
  for (const bool chunked : {false, true}) {
    expect_parser_parity(server, body, true, chunked);
  }
}

TEST(HttpJsonBody, EarlyParseFailureLeavesFollowingPipelinedRequestIntact) {
  json_body_server_t server;
  // The parser fails before consuming most of this body. Its unread content must remain separate
  // from HTTP framing, even when it contains text resembling another request.
  const std::string body = "{]" + std::string(70 * 1024, 'x') +
                           "\r\nGET /not-a-request HTTP/1.1\r\nHost: localhost\r\n\r\n";
  for (const bool chunked : {false, true}) {
    expect_parser_parity(server, body, false, chunked);
  }
}

TEST(HttpJsonBody, InvalidUtf8BadRequestUsesProductionSerializerAndKeepsPipeline) {
  json_body_server_t server;
  const std::string body = "{\"value\":\"\xc3(\"}";
  for (const bool chunked : {false, true}) {
    SCOPED_TRACE(chunked ? "chunked" : "Content-Length");
    const auto responses = exchange_json_requests(server.port, pipelined_request("/bad-request", body, chunked));
    ASSERT_EQ(responses.size(), 2u);
    EXPECT_EQ(responses[0].status, "400 Bad Request");
    const auto &failure = responses[0].body;
    ASSERT_TRUE(failure.is_object());
    EXPECT_EQ(failure.size(), 3u);
    EXPECT_EQ(failure.at("status_code"), 400);
    EXPECT_EQ(failure.at("status"), false);
    ASSERT_TRUE(failure.at("error").is_string());
    const auto message = failure.at("error").get<std::string>();
    EXPECT_NE(message.find("parse_error.101"), std::string::npos);
    EXPECT_NE(message.find("syntax error"), std::string::npos);
    EXPECT_NE(message.find("\xef\xbf\xbd"), std::string::npos);
    EXPECT_EQ(responses[1].status, "200 OK");
    EXPECT_EQ(responses[1].body, (json {{"pong", true}}));
  }
}

TEST(HttpJsonBody, BadRequestPreservesNormalParseMessageAndAcceptsValidJson) {
  json_body_server_t server;
  const std::string malformed = R"({"items":[1,]})";
  std::string expected_message;
  try {
    const auto unexpected = json::parse(malformed);
    FAIL() << "Malformed JSON unexpectedly parsed: " << unexpected;
  } catch (const json::parse_error &error) {
    expected_message = error.what();
  }
  ASSERT_FALSE(expected_message.empty());
  const json document {{"value", "accepted"}};
  for (const bool chunked : {false, true}) {
    SCOPED_TRACE(chunked ? "chunked" : "Content-Length");
    const auto rejected = exchange_json_requests(server.port, pipelined_request("/bad-request", malformed, chunked));
    ASSERT_EQ(rejected.size(), 2u);
    EXPECT_EQ(rejected[0].status, "400 Bad Request");
    EXPECT_EQ(rejected[0].body, (json {{"status_code", 400}, {"status", false}, {"error", expected_message}}));
    EXPECT_EQ(rejected[1].status, "200 OK");
    EXPECT_EQ(rejected[1].body, (json {{"pong", true}}));

    const auto accepted = exchange_json_requests(server.port, pipelined_request("/bad-request", document.dump(), chunked));
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_EQ(accepted[0].status, "200 OK");
    EXPECT_EQ(accepted[0].body, (json {{"document", document}}));
    EXPECT_EQ(accepted[1].status, "200 OK");
    EXPECT_EQ(accepted[1].body, (json {{"pong", true}}));
  }
}
