#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include <boost/asio.hpp>
#include <gtest/gtest.h>

#include <Simple-Web-Server/server_http.hpp>

using namespace std::chrono_literals;

namespace {
  using http_server_t = SimpleWeb::Server<SimpleWeb::HTTP>;

  struct exchange_result_t {
    std::string response;
    bool peer_closed = false;
  };

  class request_limit_server_t {
  public:
    request_limit_server_t() {
      server.config.port = 0;
      server.config.address = "127.0.0.1";
      server.config.thread_pool_size = 1;
      server.config.max_request_streambuf_size = global_limit;
      server.config.request_content_length_limit = [](const auto &request) {
        return request.path == "/limited" ? offline_limit : global_limit;
      };
      server.resource["^/limited$"]["POST"] =
        [this](const auto &response, const auto &request) {
          ++limited_calls;
          last_content_size = request->content.size();
          response->write("accepted");
        };
      server.resource["^/large$"]["POST"] =
        [this](const auto &response, const auto &request) {
          ++large_calls;
          last_content_size = request->content.size();
          response->write("accepted");
        };
      server.resource["^/ping$"]["GET"] =
        [this](const auto &response, const auto &) {
          ++ping_calls;
          response->write("pong");
        };

      auto ready = std::make_shared<std::promise<unsigned short>>();
      auto future = ready->get_future();
      thread = std::thread([this, ready]() {
        try {
          server.start([ready](const unsigned short assigned_port) {
            ready->set_value(assigned_port);
          });
        } catch (...) {
          try {
            ready->set_exception(std::current_exception());
          } catch (...) {
          }
        }
      });
      if (future.wait_for(3s) != std::future_status::ready) {
        server.stop();
        if (thread.joinable()) {
          thread.join();
        }
        throw std::runtime_error("timed out starting request-limit test server");
      }
      try {
        port = future.get();
      } catch (...) {
        if (thread.joinable()) {
          thread.join();
        }
        throw;
      }
    }

    ~request_limit_server_t() {
      server.stop();
      if (thread.joinable()) {
        thread.join();
      }
    }

    static constexpr std::size_t offline_limit = 64ull * 1024ull;
    static constexpr std::size_t global_limit = 128ull * 1024ull;

    http_server_t server;
    std::thread thread;
    unsigned short port = 0;
    std::atomic_int limited_calls {0};
    std::atomic_int large_calls {0};
    std::atomic_int ping_calls {0};
    std::atomic_size_t last_content_size {0};
  };

  exchange_result_t raw_exchange(
    const unsigned short port,
    const std::string &request
  ) {
    boost::asio::io_context context;
    boost::asio::ip::tcp::socket socket {context};
    socket.connect({
      boost::asio::ip::make_address_v4("127.0.0.1"),
      port,
    });
    boost::asio::write(socket, boost::asio::buffer(request));
    boost::system::error_code ignored;
    socket.shutdown(boost::asio::ip::tcp::socket::shutdown_send, ignored);
    socket.non_blocking(true, ignored);

    exchange_result_t result;
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    std::array<char, 4096> buffer {};
    while (std::chrono::steady_clock::now() < deadline) {
      boost::system::error_code error;
      const auto received =
        socket.read_some(boost::asio::buffer(buffer), error);
      if (!error) {
        result.response.append(buffer.data(), received);
        continue;
      }
      if (
        error == boost::asio::error::would_block ||
        error == boost::asio::error::try_again
      ) {
        std::this_thread::sleep_for(2ms);
        continue;
      }
      result.peer_closed =
        error == boost::asio::error::eof ||
        error == boost::asio::error::connection_reset;
      break;
    }
    socket.close(ignored);
    return result;
  }

  std::size_t count_occurrences(
    const std::string &text,
    const std::string_view needle
  ) {
    std::size_t count = 0;
    std::size_t position = 0;
    while ((position = text.find(needle, position)) != std::string::npos) {
      ++count;
      position += needle.size();
    }
    return count;
  }
}  // namespace

TEST(SimpleWebRequestLimit, RejectsDeclaredOversizeWith413AndCloses) {
  request_limit_server_t server;
  const auto result = raw_exchange(
    server.port,
    "POST /limited HTTP/1.1\r\n"
    "Host: localhost\r\n"
    "Content-Length: 65537\r\n"
    "Connection: keep-alive\r\n\r\n"
  );
  EXPECT_NE(result.response.find("HTTP/1.1 413"), std::string::npos);
  EXPECT_TRUE(result.peer_closed);
  EXPECT_EQ(server.limited_calls, 0);
}

TEST(SimpleWebRequestLimit, RejectsCumulativeChunkOverflowWith413AndCloses) {
  request_limit_server_t server;
  std::string request =
    "POST /limited HTTP/1.1\r\n"
    "Host: localhost\r\n"
    "Transfer-Encoding: chunked\r\n"
    "Connection: keep-alive\r\n\r\n"
    "9C40\r\n";
  request.append(40000, 'a');
  request += "\r\n7530\r\n";
  const auto result = raw_exchange(server.port, request);
  EXPECT_NE(result.response.find("HTTP/1.1 413"), std::string::npos);
  EXPECT_TRUE(result.peer_closed);
  EXPECT_EQ(server.limited_calls, 0);
}

TEST(SimpleWebRequestLimit, PreservesLargerRoutesAndKeepAliveOverreads) {
  request_limit_server_t server;
  std::string large_request =
    "POST /large HTTP/1.1\r\n"
    "Host: localhost\r\n"
    "Content-Length: 70000\r\n"
    "Connection: close\r\n\r\n";
  large_request.append(70000, 'b');
  const auto large = raw_exchange(server.port, large_request);
  EXPECT_NE(large.response.find("HTTP/1.1 200"), std::string::npos);
  EXPECT_EQ(server.large_calls, 1);
  EXPECT_EQ(server.last_content_size, 70000u);

  const auto pipelined = raw_exchange(
    server.port,
    "POST /limited HTTP/1.1\r\n"
    "Host: localhost\r\n"
    "Content-Length: 2\r\n"
    "Connection: keep-alive\r\n\r\n"
    "{}"
    "GET /ping HTTP/1.1\r\n"
    "Host: localhost\r\n"
    "Connection: close\r\n\r\n"
  );
  EXPECT_EQ(count_occurrences(pipelined.response, "HTTP/1.1 200"), 2u);
  EXPECT_NE(pipelined.response.find("accepted"), std::string::npos);
  EXPECT_NE(pipelined.response.find("pong"), std::string::npos);
  EXPECT_EQ(server.limited_calls, 1);
  EXPECT_EQ(server.ping_calls, 1);
}

TEST(SimpleWebRequestLimit, RejectsAmbiguousFramingAndAcceptsMixedCaseChunked) {
  request_limit_server_t server;
  for (const auto &request : {
         std::string {
           "POST /limited HTTP/1.1\r\n"
           "Host: localhost\r\n"
           "Content-Length: 2\r\n"
           "Transfer-Encoding: chunked\r\n"
           "Connection: keep-alive\r\n\r\n"
         },
         std::string {
           "POST /limited HTTP/1.1\r\n"
           "Host: localhost\r\n"
           "Content-Length: 2\r\n"
           "Content-Length: 3\r\n"
           "Connection: keep-alive\r\n\r\n"
         },
       }) {
    const auto rejected = raw_exchange(server.port, request);
    EXPECT_NE(rejected.response.find("HTTP/1.1 400"), std::string::npos);
    EXPECT_TRUE(rejected.peer_closed);
  }
  EXPECT_EQ(server.limited_calls, 0);

  const auto accepted = raw_exchange(
    server.port,
    "POST /limited HTTP/1.1\r\n"
    "Host: localhost\r\n"
    "Transfer-Encoding: ChUnKeD\r\n"
    "Connection: close\r\n\r\n"
    "2\r\n{}\r\n0\r\n\r\n"
  );
  EXPECT_NE(accepted.response.find("HTTP/1.1 200"), std::string::npos);
  EXPECT_EQ(server.limited_calls, 1);
  EXPECT_EQ(server.last_content_size, 2u);
}
