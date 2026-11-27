/** Actual offline handlers/authentication with an owned service and finite startup gate. */
#include <gtest/gtest.h>

#include "src/config.h"
#include "src/confighttp.h"
#include "src/httpcommon.h"
#include "src/network.h"
#include "src/offline_sbs_job.h"
#include "src/offline_sbs_service_test.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <Simple-Web-Server/server_https.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace confighttp {
  using offline_http_base_t = SimpleWeb::ServerBase<SimpleWeb::HTTPS>;
  void getOfflineSbsCapabilities(std::shared_ptr<offline_http_base_t::Response>, std::shared_ptr<offline_http_base_t::Request>);
  void browseOfflineSbsFilesystem(std::shared_ptr<offline_http_base_t::Response>, std::shared_ptr<offline_http_base_t::Request>);
  void listOfflineSbsJobs(std::shared_ptr<offline_http_base_t::Response>, std::shared_ptr<offline_http_base_t::Request>);
  void createOfflineSbsJob(std::shared_ptr<offline_http_base_t::Response>, std::shared_ptr<offline_http_base_t::Request>);
  void getOfflineSbsJob(std::shared_ptr<offline_http_base_t::Response>, std::shared_ptr<offline_http_base_t::Request>);
  void cancelOfflineSbsJob(std::shared_ptr<offline_http_base_t::Response>, std::shared_ptr<offline_http_base_t::Request>);
  void clearOfflineSbsJob(std::shared_ptr<offline_http_base_t::Response>, std::shared_ptr<offline_http_base_t::Request>);
  void add_routes(offline_http_base_t &server);
}  // namespace confighttp

namespace {
  namespace fs = std::filesystem;
  using namespace std::chrono_literals;
  using https_base_t = confighttp::offline_http_base_t;
  using tcp = boost::asio::ip::tcp;
  using json = nlohmann::json;
  using owner_t = offline_sbs::detail::service_owner_t;
  using access_t = offline_sbs::testing::service_owner_test_access;
  using hooks_t = offline_sbs::testing::service_initialization_hooks_t;
  using stage_t = offline_sbs::testing::service_initialization_stage_e;
  using owner_scope_t = offline_sbs::testing::scoped_service_owner_override_t;

  // Same protected Request/Response factory and bounded loopback endpoint used by the existing
  // authentication fixture. No HTTP server, certificate, TLS handshake or media child starts.
  class offline_http_request_t: public https_base_t {
  public:
    explicit offline_http_request_t(bool loopback):
        https_base_t(0),
        tls_(boost::asio::ssl::context::tls_server),
        client_(io_) {
      connection_ = create_connection(io_, tls_);
      if (loopback) {
        connect_loopback();
      }
      session_ = std::make_shared<Session>(128 * 1024, connection_);
      std::function<void(std::shared_ptr<Response>, std::shared_ptr<Request>)> capture =
        [this](auto response, auto request) {
          response_ = std::move(response);
          request_ = std::move(request);
        };
      write(session_, capture);
      if (!response_ || !request_) {
        throw std::runtime_error("Unable to construct the offline HTTPS request fixture");
      }
    }

    ~offline_http_request_t() override {
      // Factory deletion posts a send; close its admission and drain it without TLS I/O.
      handler_runner->stop();
      response_.reset();
      io_.restart();
      io_.poll();
    }

    void body(std::string_view bytes, std::optional<std::string_view> content_type = "application/json") {
      if (content_type) {
        request_->header.emplace("Content-Type", std::string(*content_type));
      }
      std::ostream output(request_->content.rdbuf());
      output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      if (!output) {
        throw std::runtime_error("Unable to fill the actual request body stream");
      }
    }

    std::string output() const {
      if (!output_) {
        std::ostringstream result;
        result << response_->rdbuf();
        output_ = result.str();
      }
      return *output_;
    }

    json document() const {
      const auto bytes = output();
      const auto separator = bytes.find("\r\n\r\n");
      if (separator == std::string::npos) {
        throw std::runtime_error("Offline handler did not produce an HTTP document");
      }
      return json::parse(bytes.substr(separator + 4));
    }

    std::shared_ptr<Response> response_;
    std::shared_ptr<Request> request_;

  private:
    void accept() override {}

    void connect_loopback() {
      tcp::acceptor acceptor(io_, {boost::asio::ip::address_v4::loopback(), 0});
      boost::asio::steady_timer deadline(io_, 3s);
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
    mutable std::optional<std::string> output_;
  };

  struct route_t {
    const char *method;
    const char *path;
    const char *pattern;
    void (*handler)(std::shared_ptr<https_base_t::Response>, std::shared_ptr<https_base_t::Request>);
    bool json_body;
  };

  const std::array<route_t, 7> routes {{
    {"GET", "/api/offline-sbs/capabilities", "^/api/offline-sbs/capabilities$", confighttp::getOfflineSbsCapabilities, false},
    {"POST", "/api/offline-sbs/browse", "^/api/offline-sbs/browse$", confighttp::browseOfflineSbsFilesystem, true},
    {"GET", "/api/offline-sbs/jobs", "^/api/offline-sbs/jobs$", confighttp::listOfflineSbsJobs, false},
    {"POST", "/api/offline-sbs/jobs", "^/api/offline-sbs/jobs$", confighttp::createOfflineSbsJob, true},
    {"GET", "/api/offline-sbs/jobs/11111111-2222-3333-4444-555555555555", "^/api/offline-sbs/jobs/([0-9A-Fa-f-]+)$", confighttp::getOfflineSbsJob, false},
    {"POST", "/api/offline-sbs/jobs/11111111-2222-3333-4444-555555555555/cancel", "^/api/offline-sbs/jobs/([0-9A-Fa-f-]+)/cancel$", confighttp::cancelOfflineSbsJob, false},
    {"DELETE", "/api/offline-sbs/jobs/11111111-2222-3333-4444-555555555555", "^/api/offline-sbs/jobs/([0-9A-Fa-f-]+)$", confighttp::clearOfflineSbsJob, false},
  }};

  void select_route(offline_http_request_t &request, const route_t &route) {
    request.request_->method = route.method;
    request.request_->path = route.path;
    request.request_->header.emplace("Host", "apollo.test:47990");
    if (request.request_->method == "POST" || request.request_->method == "DELETE") {
      request.request_->header.emplace("Origin", "https://apollo.test:47990");
      request.request_->header.emplace("Sec-Fetch-Site", "same-origin");
    }
    if (!SimpleWeb::regex::regex_match(request.request_->path, request.request_->path_match, SimpleWeb::regex::regex(route.pattern))) {
      throw std::runtime_error("Fixture path does not match the actual handler route");
    }
  }

  // Runs the route as the server registers it, with its access check, not the bare handler.
  void serve(offline_http_request_t &request, const route_t &route) {
    confighttp::add_routes(request);
    for (const auto &[pattern, methods] : request.resource) {
      if (pattern.str != route.pattern) {
        continue;
      }
      const auto handler = methods.find(route.method);
      if (handler == methods.end()) {
        break;
      }
      handler->second(request.response_, request.request_);
      return;
    }
    throw std::runtime_error("The server registers no such route");
  }

  class temporary_tree_t {
  public:
    temporary_tree_t() {
      static std::atomic_uint64_t next {0};
      base_ = fs::weakly_canonical(fs::temp_directory_path());
      path = base_ / ("sunshine3d-offline-http-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(next.fetch_add(1)));
      if (!fs::create_directory(path)) {
        throw std::runtime_error("HTTP temporary child already exists; ownership was not acquired");
      }
    }

    ~temporary_tree_t() {
      std::error_code error;
      const auto resolved = fs::weakly_canonical(path, error);
      if (error || !resolved.is_absolute() || resolved.parent_path() != base_ || !resolved.filename().string().starts_with("sunshine3d-offline-http-")) {
        ADD_FAILURE() << "Refusing cleanup outside the owned HTTP temporary child: " << path;
        return;
      }
      fs::remove_all(resolved, error);
      EXPECT_FALSE(error) << error.message();
    }

    fs::path path;

  private:
    fs::path base_;
  };

  offline_sbs::service_config_t service_config(const fs::path &root) {
    for (const auto *name : {"sunshine.exe", "sunshine.conf", "ffmpeg.exe", "ffprobe.exe"}) {
      std::ofstream output(root / name, std::ios::binary);
      output.exceptions(std::ios::badbit | std::ios::failbit);
      output << "fixture placeholder, never executed";
    }
    return {
      .state_root = root / "state",
      .worker_root = root / "worker",
      .sunshine_executable = root / "sunshine.exe",
      .sunshine_config = root / "sunshine.conf",
      .ffmpeg_executable = root / "ffmpeg.exe",
      .ffprobe_executable = root / "ffprobe.exe",
      .max_retained_jobs = 13,
      .max_retained_artifact_bytes = 3 * 1024 * 1024,
      .process_poll_interval = 5ms,
      .probe_media_tools = false,
    };
  }

  struct gate_t {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
    bool timed_out = false;

    void hold() {
      std::unique_lock lock(mutex);
      entered = true;
      changed.notify_all();
      timed_out = !changed.wait_for(lock, 10s, [&] {
        return released;
      });
    }

    bool wait_entered() {
      std::unique_lock lock(mutex);
      return changed.wait_for(lock, 3s, [&] {
        return entered;
      });
    }

    void release() {
      std::lock_guard lock(mutex);
      released = true;
      changed.notify_all();
    }

    bool timeout_observed() {
      std::lock_guard lock(mutex);
      return timed_out;
    }
  };

  struct release_gate_t {
    gate_t &gate;

    ~release_gate_t() {
      gate.release();
    }
  };

  class local_owner_t {
  public:
    local_owner_t():
        config(service_config(tree.path)),
        owner(access_t::create(config, hooks())),
        release {gate},
        scope(*owner) {}

    void expect_dormant() {
      EXPECT_EQ(launches.load(), 0);
      EXPECT_EQ(entered.load(), 0);
      EXPECT_EQ(allocations.load(), 0);
      EXPECT_EQ(started.load(), 0);
      EXPECT_EQ(exited.load(), 0);
      EXPECT_FALSE(owner->ready_service());
      EXPECT_FALSE(fs::exists(config.state_root));
      EXPECT_FALSE(fs::exists(config.worker_root));
    }

    void release_and_join() {
      gate.release();
      access_t::join(*owner);
      EXPECT_EQ(launches.load(), 1);
      EXPECT_EQ(entered.load(), 1);
      EXPECT_EQ(allocations.load(), 1);
      EXPECT_EQ(started.load(), 1);
      EXPECT_EQ(exited.load(), 1);
      EXPECT_FALSE(gate.timeout_observed());
      EXPECT_TRUE(owner->ready_service());
      EXPECT_TRUE(owner->capabilities().at("available").get<bool>()) << owner->capabilities().dump();
    }

    std::string create_body() const {
      const auto path = (tree.path / "missing-input.mkv").generic_u8string();
      return json {{"input_path", std::string(reinterpret_cast<const char *>(path.data()), path.size())}, {"operation", "evaluate"}}.dump();
    }

    temporary_tree_t tree;
    offline_sbs::service_config_t config;
    std::atomic_int launches {0}, allocations {0}, entered {0}, started {0}, exited {0};
    gate_t gate;
    std::unique_ptr<owner_t> owner;
    // Destruction order: restore caller TLS, release gate, join/destroy real owner, then files.
    release_gate_t release;
    owner_scope_t scope;

  private:
    hooks_t hooks() {
      return {
        .observe = [&](stage_t stage) {
          switch (stage) {
            case stage_t::worker_enter:
              ++entered;
              gate.hold();
              break;
            case stage_t::service_start_finished:
              ++started;
              break;
            case stage_t::worker_exit:
              ++exited;
              break;
          }
        },
        .before_thread_launch = [&] {
          ++launches;
        },
        .before_service_allocation = [&] {
          ++allocations;
        },
      };
    }
  };

  class OfflineSbsHttpDemandTest: public testing::Test {
  protected:
    void SetUp() override {
      config::sunshine.username = "offline-http-fixture-admin";
      http::origin_web_ui_allowed = net::WAN;
    }

    ~OfflineSbsHttpDemandTest() override {
      config::sunshine.username = std::move(username_);
      http::origin_web_ui_allowed = origin_;
    }

    static void expect_status(const offline_http_request_t &request, std::string_view status) {
      EXPECT_TRUE(request.output().starts_with(status)) << request.output();
    }

  private:
    std::string username_ = config::sunshine.username;
    net::net_e origin_ = http::origin_web_ui_allowed;
  };

  TEST_F(OfflineSbsHttpDemandTest, PureFacadeSnapshotsDoNotStartTheConfiguredOwner) {
    local_owner_t local;
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      EXPECT_FALSE(offline_sbs::capabilities().at("available").get<bool>());
      EXPECT_TRUE(offline_sbs::list().empty());
      EXPECT_FALSE(offline_sbs::has_active_job());
    }
    local.expect_dormant();
  }

  TEST_F(OfflineSbsHttpDemandTest, AllSevenUnauthorizedHandlersRejectBeforeDemand) {
    local_owner_t local;
    for (const auto &route : routes) {
      SCOPED_TRACE(route.path);
      SCOPED_TRACE(route.method);
      offline_http_request_t request(false);
      select_route(request, route);
      if (route.json_body) {
        request.body(route.handler == confighttp::createOfflineSbsJob ? local.create_body() : "{}");
      }
      serve(request, route);
      expect_status(request, "HTTP/1.1 401 Unauthorized\r\n");
      local.expect_dormant();
    }
  }

  TEST_F(OfflineSbsHttpDemandTest, InvalidJsonTypeAndBothSizeGuardsRejectBeforeDemand) {
    local_owner_t local;

    struct invalid_t {
      std::string body;
      std::optional<std::string_view> type;
      std::optional<std::string> length;
      const char *status;
    };

    const std::array<invalid_t, 7> invalid {{
      {"{", "application/json", {}, "HTTP/1.1 400 Bad Request\r\n"},
      {"[]", "application/json", {}, "HTTP/1.1 400 Bad Request\r\n"},
      {"{\"unknown-fixture-field\":true}", "application/json", {}, "HTTP/1.1 400 Bad Request\r\n"},
      {"{}", {}, {}, "HTTP/1.1 400 Bad Request\r\n"},
      {"{}", "text/plain", {}, "HTTP/1.1 400 Bad Request\r\n"},
      {"{}", "application/json", std::to_string(confighttp::OFFLINE_SBS_REQUEST_MAX_BYTES + 1), "HTTP/1.1 413 Payload Too Large\r\n"},
      {std::string(confighttp::OFFLINE_SBS_REQUEST_MAX_BYTES + 1, 'x'), "application/json", {}, "HTTP/1.1 413 Payload Too Large\r\n"},
    }};
    for (const auto &route : routes) {
      if (!route.json_body) {
        continue;
      }
      for (const auto &value : invalid) {
        SCOPED_TRACE(route.path);
        SCOPED_TRACE(value.status);
        offline_http_request_t request(true);
        select_route(request, route);
        request.body(value.body, value.type);
        if (value.length) {
          request.request_->header.emplace("Content-Length", *value.length);
        }
        serve(request, route);
        expect_status(request, value.status);
        local.expect_dormant();
      }
    }
  }

  TEST_F(OfflineSbsHttpDemandTest, RetiredCacheLimitFieldsAreRejectedBeforeDemand) {
    local_owner_t local;
    const auto &create = routes[3];
    ASSERT_EQ(create.handler, confighttp::createOfflineSbsJob);
    const json retired {
      {"scene_cache_max_bytes", 4ull * 1024ull * 1024ull * 1024ull},
      {"cache_budget_policy", "fail"},
    };
    for (const auto &[name, value] : retired.items()) {
      SCOPED_TRACE(name);
      auto body = json::parse(local.create_body());
      body[name] = value;
      offline_http_request_t request(true);
      select_route(request, create);
      request.body(body.dump());
      serve(request, create);
      expect_status(request, "HTTP/1.1 400 Bad Request\r\n");
      local.expect_dormant();
    }
  }

  TEST_F(OfflineSbsHttpDemandTest, FirstAuthorizedGetStartsOnceReturns503ThenActualReady200) {
    local_owner_t local;
    local.expect_dormant();
    {
      offline_http_request_t request(true);
      select_route(request, routes[0]);
      serve(request, routes[0]);
      expect_status(request, "HTTP/1.1 503 Service Unavailable\r\n");
      EXPECT_FALSE(request.document().at("status").get<bool>());
    }
    ASSERT_TRUE(local.gate.wait_entered());
    EXPECT_EQ(local.launches.load(), 1);
    EXPECT_EQ(local.entered.load(), 1);
    EXPECT_EQ(local.allocations.load(), 0);
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      offline_http_request_t request(true);
      select_route(request, routes[0]);
      serve(request, routes[0]);
      expect_status(request, "HTTP/1.1 503 Service Unavailable\r\n");
      EXPECT_FALSE(offline_sbs::has_active_job());
      EXPECT_TRUE(offline_sbs::list().empty());
      EXPECT_EQ(local.launches.load(), 1);
    }
    local.release_and_join();
    for (const auto index : {0u, 2u}) {
      offline_http_request_t request(true);
      select_route(request, routes[index]);
      serve(request, routes[index]);
      expect_status(request, "HTTP/1.1 200 OK\r\n");
      EXPECT_TRUE(request.document().at("status").get<bool>());
    }
    EXPECT_EQ(local.launches.load(), 1);
    EXPECT_EQ(local.started.load(), 1);
  }

  TEST_F(OfflineSbsHttpDemandTest, EveryAuthorizedOfflineHandlerCanRequestTheSingleInitializer) {
    for (const auto &route : routes) {
      SCOPED_TRACE(route.path);
      SCOPED_TRACE(route.method);
      local_owner_t local;
      offline_http_request_t request(true);
      select_route(request, route);
      if (route.json_body) {
        request.body(route.handler == confighttp::createOfflineSbsJob ? local.create_body() : "{}");
      }
      serve(request, route);
      expect_status(request, "HTTP/1.1 503 Service Unavailable\r\n");
      ASSERT_TRUE(local.gate.wait_entered());
      EXPECT_EQ(local.launches.load(), 1);
      EXPECT_EQ(local.entered.load(), 1);
      EXPECT_EQ(local.allocations.load(), 0);
      local.release_and_join();
    }
  }
}  // namespace
