/**
 * @file tests/unit/test_nvhttp_launch_transaction.cpp
 * @brief Complete launch/resume handlers with real admission and device-free native edges.
 */
#include "../tests_common.h"
#include "src/network.h"
#include "src/nvhttp_launch_test.h"
#include "src/stream.h"

#include <boost/property_tree/xml_parser.hpp>
#include <chrono>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
  using namespace std::chrono_literals;
  using tcp = boost::asio::ip::tcp;
  using https_base_t = SimpleWeb::ServerBase<nvhttp::SunshineHTTPS>;
  using launch_t = rtsp_stream::launch_session_t;
  using process_access = proc::launch_transaction_test_access;
  using ptree = boost::property_tree::ptree;

  class launch_request_t: public https_base_t {
  private:
    // I/O services outlive every owner of their sockets, including constructor-failure cleanup.
    boost::asio::io_context io_;
    boost::asio::ssl::context tls_;
    tcp::socket client_;
    std::shared_ptr<Connection> connection_;
    std::shared_ptr<Session> session_;

  public:
    launch_request_t(std::string query, const crypto::p_named_cert_t &identity):
        https_base_t(0),
        tls_(boost::asio::ssl::context::tls_server),
        client_(io_) {
      try {
        connection_ = create_connection(io_, tls_);
        connect_loopback();
        session_ = std::make_shared<Session>(4096, connection_);
        std::function<void(std::shared_ptr<Response>, std::shared_ptr<Request>)> capture =
          [this](auto response, auto request) {
            response_ = std::move(response);
            request_ = std::move(request);
          };
        write(session_, capture);
        if (!response_ || !request_) {
          throw std::runtime_error("Unable to construct the launch request fixture");
        }
        request_->method = "GET";
        request_->path = "/launch";
        request_->query_string = std::move(query);
        request_->userp = identity;
      } catch (...) {
        close_transport();
        throw;
      }
    }

    ~launch_request_t() override {
      close_transport();
    }

    void close_transport() {
      // The response factory posts its send on deletion. Refuse it before draining the callback;
      // the connected TCP endpoint is used only for local_endpoint(), never for TLS or HTTP I/O.
      handler_runner->stop();
      response_.reset();
      io_.restart();
      io_.poll();
      request_.reset();
      session_.reset();
      connection_.reset();
    }

    ptree result() {
      std::ostringstream output;
      output << response_->rdbuf();
      const auto wire = output.str();
      EXPECT_TRUE(wire.starts_with("HTTP/1.1 200 OK\r\n")) << wire;
      const auto body = wire.find("\r\n\r\n");
      if (body == std::string::npos) {
        throw std::runtime_error("Launch response has no HTTP body");
      }
      std::istringstream input(wire.substr(body + 4));
      ptree result;
      boost::property_tree::read_xml(input, result);
      EXPECT_TRUE(response_->close_connection_after_response);
      return result;
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
      auto cancel_and_drain = util::fail_guard([&] {
        // Keep the referenced local completion state alive while cancelling partial startup.
        deadline.cancel();
        boost::system::error_code ignored;
        acceptor.close(ignored);
        client_.close(ignored);
        connection_->socket->lowest_layer().close(ignored);
        io_.restart();
        io_.run();
      });
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
      cancel_and_drain.disable();
    }
  };

  struct edge_t {
    std::string name;
    bool platform_locked;
    bool live_gpu;
  };

  class NvHttpLaunchTransactionTest: public ::testing::Test {
  protected:
    void SetUp() override {
      ASSERT_FALSE(rtsp_stream::active_session_uuid());
      ASSERT_TRUE(rtsp_stream::launch_session_available());
      ASSERT_FALSE(gpu_workload::live_stream_active());
      ASSERT_FALSE(gpu_workload::offline_sbs_active());
      ASSERT_EQ(proc::proc.get_status().app_id, 0);
      {
        auto guard = stream::session::guard_platform_launch();
        ASSERT_TRUE(guard.idle());
      }
      stream::session::flush_platform_state();
      saved_process_ = std::move(proc::proc);
      owns_process_ = true;
      app_.id = "7";
      app_.uuid = "2bb2e4cf-3df4-48e6-a153-b2467fce5da0";
      app_.name = "Launch transaction fixture";
      auto other_app = app_;
      other_app.id = "8";
      other_app.uuid = "b85b69ab-caca-4cf5-a230-77bb7c9b4f04";
      proc::proc = proc::proc_t {boost::this_process::environment(), std::vector<proc::ctx_t> {app_, std::move(other_app)}};
      config::sunshine.flags[config::flag::FRESH_STATE] = true;
      config::stream.ping_timeout = 20s;
      config::stream.session_resume_grace = 60s;
      static const auto certificate = crypto::gen_creds("launch-transaction-test", 2048).x509;
      nvhttp::add_authorized_client_for_test("Launch transaction test", certificate);
      identity_ = nvhttp::get_authorized_client_for_test(certificate);
      ASSERT_TRUE(identity_);
      owns_client_ = true;
      set_permissions(crypto::PERM::_all);
      install_hooks();
    }

    void TearDown() override {
      if (!owns_process_) {
        return;
      }
      if (candidate_) {
        rtsp_stream::finish_launch_session_for_test(*candidate_, false);
      }
      if (rival_) {
        rtsp_stream::finish_launch_session_for_test(*rival_, false);
      }
      rtsp_stream::clear_pending_launch_session();
      stream::session::release_active_slot_for_test();
      stream::session::flush_platform_state();
      process_access::clear(proc::proc);
      candidate_.reset();
      original_.reset();
      rival_.reset();
      offline_.reset();
      if (owns_client_) {
        EXPECT_TRUE(nvhttp::unpair_client(identity_->uuid));
      }
      identity_.reset();
      proc::proc = std::move(saved_process_);
      config::video.output_name = saved_output_;
      config::sunshine.flags[config::flag::FRESH_STATE] = saved_fresh_state_;
      config::stream.ping_timeout = saved_ping_timeout_;
      config::stream.session_resume_grace = saved_resume_grace_;
      EXPECT_TRUE(rtsp_stream::launch_session_available());
      EXPECT_FALSE(gpu_workload::live_stream_active());
      EXPECT_FALSE(gpu_workload::offline_sbs_active());
      EXPECT_TRUE(stream::session::platform_lifecycle_lock_available_for_test());
    }

    void install_hooks() {
      hooks_.execute = [&](proc::ctx_t &&selected, std::shared_ptr<launch_t> launch, bool probe) {
        edge("execute");
        EXPECT_TRUE(probe);
        EXPECT_EQ(selected.id, app_.id);
        EXPECT_EQ(selected.uuid, app_.uuid);
        candidate_ = std::move(launch);
        if (throw_execute_) {
          throw std::runtime_error("fixture execute failure");
        }
        if (execute_result_ != 0) {
          return execute_result_;
        }
        process_access::seed(proc::proc, std::move(selected), candidate_, fresh_token, candidate_->virtual_display, [this] {
          edge("primary");
        });
        fail_publication_if_requested();
        return 0;
      };
      hooks_.terminate = [&] {
        edge("terminate");
        process_access::clear(proc::proc);
      };
      hooks_.probe_encoders = [&] {
        edge("probe");
        return probe_result_;
      };
      hooks_.reconfigure = [&](std::shared_ptr<launch_t> launch) {
        edge("reconfigure");
        candidate_ = std::move(launch);
        if (reconfigure_result_ != 0) {
          return reconfigure_result_;
        }
        // Native success commits its own contract before HTTP attempts RTSP publication.
        candidate_->virtual_display = proc::proc.get_status().virtual_display;
        process_access::publish_resume(proc::proc, *candidate_);
        fail_publication_if_requested();
        return 0;
      };
      hooks_.remote_starting = [&](std::uint64_t id, std::chrono::milliseconds timeout, bool setup) {
        edge("ar-start");
        EXPECT_NE(id, 0U);
        EXPECT_EQ(timeout, config::stream.ping_timeout);
        EXPECT_TRUE(setup);
        remote_id_ = id;
        return remote_start_result_;
      };
      hooks_.remote_awaiting = [&](std::uint64_t id, std::chrono::milliseconds timeout) {
        edge("ar-await");
        EXPECT_EQ(id, remote_id_);
        EXPECT_EQ(timeout, config::stream.ping_timeout);
      };
      hooks_.remote_ended = [&](std::uint64_t id) {
        edge("ar-end");
        EXPECT_EQ(id, remote_id_);
      };
    }

    void set_permissions(crypto::PERM permissions) {
      ASSERT_TRUE(nvhttp::update_device_info(identity_->uuid, identity_->name, permissions));
      identity_ = nvhttp::get_authorized_client_for_test(identity_->cert);
      ASSERT_TRUE(identity_);
    }

    static std::string options() {
      return "corever=1&localAudioPlayMode=0&mode=1920x1080x60&rikey=000102030405060708090A0B0C0D0E0F&rikeyid=9&sops=1&surroundAudioInfo=2";
    }

    std::string query(bool resume = false, std::string extra = {}) const {
      return options() + "&appid=7&appuuid=" + app_.uuid +
             (resume ? "&hostSessionId=" + std::to_string(retained_token) : "") + extra;
    }

    ptree dispatch(bool resume, std::string query, bool authorized = true) {
      launch_request_t request(std::move(query), authorized ? identity_ : nullptr);
      request.request_->path = resume ? "/resume" : "/launch";
      nvhttp::testing::invoke_launch_handler_for_test(resume, request.response_, request.request_, hooks_);
      return request.result();
    }

    static void expect_error(const ptree &tree, int status, std::string_view message, const char *result_field = "resume") {
      EXPECT_EQ(tree.get<int>("root.<xmlattr>.status_code"), status);
      EXPECT_EQ(tree.get<std::string>("root.<xmlattr>.status_message"), message);
      if (result_field) {
        EXPECT_EQ(tree.get<int>(std::string("root.") + result_field), 0);
      }
      EXPECT_FALSE(tree.get_optional<std::string>("root.sessionUrl0"));
      EXPECT_FALSE(tree.get_optional<std::string>("root.hostsessionid"));
    }

    void expect_events(std::initializer_list<const char *> expected) const {
      std::vector<std::string> names;
      for (const auto &entry : edges_) {
        names.push_back(entry.name);
        EXPECT_EQ(entry.platform_locked, entry.name != "ar-end") << entry.name;
        EXPECT_EQ(entry.live_gpu, entry.name != "ar-end") << entry.name;
      }
      std::vector<std::string> wanted(expected.begin(), expected.end());
      EXPECT_EQ(names, wanted);
    }

    void expect_unpublished() const {
      EXPECT_TRUE(rtsp_stream::launch_session_available());
      EXPECT_FALSE(gpu_workload::live_stream_active());
      EXPECT_TRUE(stream::session::platform_lifecycle_lock_available_for_test());
      if (candidate_) {
        EXPECT_FALSE(candidate_->pending_live_gpu_lease);
      }
    }

    void retain(bool virtual_display) {
      original_ = std::make_shared<launch_t>();
      original_->id = 0x70000001U;
      original_->width = 1280;
      original_->height = 720;
      original_->fps = 30000;
      original_->scale_factor = 100;
      original_->unique_id = identity_->uuid;
      original_->virtual_display = virtual_display;
      process_access::seed(proc::proc, app_, original_, retained_token, virtual_display, [this] {
        edge("primary");
      });
    }

    auto existing_deadline() {
      auto guard = stream::session::guard_platform_launch();
      guard.commit();
      return stream::session::platform_stop_deadline_for_test();
    }

    void edge(std::string name) {
      edges_.push_back({std::move(name), !stream::session::platform_lifecycle_lock_available_for_test(), gpu_workload::live_stream_active()});
    }

    void fail_publication_if_requested() {
      if (revoke_publication_) {
        candidate_->revoke_reservation();
      }
      if (rival_publication_) {
        rival_ = std::make_shared<launch_t>();
        rival_->id = 0x70000002U;
        rival_->gcm_key.assign(16, 0);
        rival_->rtsp_cipher.emplace(rival_->gcm_key, false);
        rival_->av_ping_payload = "0123456789abcdef";
        EXPECT_TRUE(rtsp_stream::launch_session_raise(rival_));
      }
    }

    static constexpr std::uint64_t retained_token = 1234;
    static constexpr std::uint64_t fresh_token = 9001;
    proc::ctx_t app_ {};
    crypto::p_named_cert_t identity_;
    nvhttp::testing::launch_native_hooks_t hooks_;
    std::vector<edge_t> edges_;
    std::shared_ptr<launch_t> candidate_;
    std::shared_ptr<launch_t> original_;
    std::shared_ptr<launch_t> rival_;
    std::optional<gpu_workload::lease_t> offline_;
    int execute_result_ = 0;
    int probe_result_ = 0;
    int reconfigure_result_ = 0;
    bool remote_start_result_ = true;
    bool throw_execute_ = false;
    bool revoke_publication_ = false;
    bool rival_publication_ = false;
    std::uint64_t remote_id_ = 0;

  private:
    proc::proc_t saved_process_;
    bool owns_process_ = false;
    bool owns_client_ = false;
    std::string saved_output_ = config::video.output_name;
    bool saved_fresh_state_ = config::sunshine.flags[config::flag::FRESH_STATE];
    std::chrono::milliseconds saved_ping_timeout_ = config::stream.ping_timeout;
    std::chrono::milliseconds saved_resume_grace_ = config::stream.session_resume_grace;
  };
}  // namespace

TEST_F(NvHttpLaunchTransactionTest, UnauthorizedRequestsKeepTheExactEarlyXmlEnvelope) {
  for (const bool resume : {false, true}) {
    const auto tree = dispatch(resume, "appid=bad", false);
    expect_error(tree, 401, "The client is no longer authorized", nullptr);
    EXPECT_FALSE(tree.get_optional<int>("root.gamesession"));
    EXPECT_FALSE(tree.get_optional<int>("root.resume"));
  }
  expect_events({});
  expect_unpublished();
}

TEST_F(NvHttpLaunchTransactionTest, FreshLookupErrorsPrecedeLaunchPermissionAndKeepTheirFields) {
  set_permissions(crypto::PERM::_no);
  const auto missing = dispatch(false, options() + "&appid=9");
  expect_error(missing, 404, "Cannot find requested application", "gamesession");
  EXPECT_FALSE(missing.get_optional<int>("root.resume"));
  const auto mismatch = dispatch(false, options() + "&appid=8&appuuid=" + app_.uuid);
  expect_error(mismatch, 400, "Application ID and UUID identify different applications", "gamesession");
  const auto denied = dispatch(false, query());
  expect_error(denied, 403, "Permission denied");
  EXPECT_FALSE(denied.get_optional<int>("root.gamesession"));
  expect_events({});
  expect_unpublished();
}

TEST_F(NvHttpLaunchTransactionTest, ResumePermissionAndTokenChecksPrecedeMalformedOptions) {
  retain(false);
  const auto deadline = existing_deadline();
  set_permissions(crypto::PERM::_no);
  expect_error(dispatch(true, "hostSessionId=bad&appid=bad"), 403, "Permission denied");
  set_permissions(crypto::PERM::_all);
  expect_error(dispatch(true, "hostSessionId=0&appid=bad"), 409, "Missing or stale host session ID");
  expect_error(dispatch(true, "hostSessionId=1234&appid=bad"), 400, "Invalid application ID");
  expect_error(dispatch(true, "hostSessionId=1234&appid=7"), 400, "Invalid resume parameters");
  expect_error(dispatch(true, options() + "&hostSessionId=1234&appid=8"), 409, "The requested application does not match the retained host session");
  EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
  EXPECT_EQ(proc::proc.get_host_session_id(), retained_token);
  expect_events({});
  expect_unpublished();
}

TEST_F(NvHttpLaunchTransactionTest, ActiveSessionRejectionPrecedesPendingAndOfflineAdmission) {
  retain(false);
  ASSERT_TRUE(stream::session::claim_active_slot_for_test());
  offline_ = gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs);
  ASSERT_TRUE(offline_);
  for (const bool resume : {false, true}) {
    expect_error(dispatch(resume, query(resume)), 503, "Another streaming session is already active");
  }
  expect_events({});
}

TEST_F(NvHttpLaunchTransactionTest, ExistingHandshakeIsPreservedBeforeOfflineOrNativeWork) {
  retain(false);
  candidate_ = std::make_shared<launch_t>();
  rival_publication_ = true;
  fail_publication_if_requested();
  offline_ = gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs);
  ASSERT_TRUE(offline_);
  for (const bool resume : {false, true}) {
    expect_error(dispatch(resume, query(resume)), 503, "Another streaming handshake is already pending");
  }
  ASSERT_TRUE(rival_);
  EXPECT_EQ(rival_->reservation(), rtsp_stream::launch_reservation_state_e::pending);
  EXPECT_TRUE(rtsp_stream::claim_launch_session_for_test(*rival_));
  expect_events({});
}

TEST_F(NvHttpLaunchTransactionTest, OfflineOwnershipRejectsBothHandlersBeforeNativeWork) {
  retain(false);
  offline_ = gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs);
  ASSERT_TRUE(offline_);
  for (const bool resume : {false, true}) {
    expect_error(dispatch(resume, query(resume)), 503, "Offline SBS conversion is currently using the GPU");
  }
  EXPECT_EQ(proc::proc.get_host_session_id(), retained_token);
  expect_events({});
}

TEST_F(NvHttpLaunchTransactionTest, FreshExecuteFailureReleasesGpuWithoutOuterTermination) {
  for (const auto error : {503, 521}) {
    execute_result_ = error;
    const auto tree = dispatch(false, query());
    expect_error(tree, error, error == 503 ? "Failed to initialize video capture/encoding. Is a display connected and turned on?" : "Failed to start the specified application", "gamesession");
    EXPECT_FALSE(tree.get_optional<int>("root.resume"));
    expect_events({"execute"});
    expect_unpublished();
    EXPECT_EQ(proc::proc.get_status().app_id, 0);
    edges_.clear();
  }
}

TEST_F(NvHttpLaunchTransactionTest, FreshExecuteExceptionUnwindsGpuAndThePlatformLock) {
  throw_execute_ = true;
  launch_request_t request(query(), identity_);
  EXPECT_THROW(nvhttp::testing::invoke_launch_handler_for_test(false, request.response_, request.request_, hooks_), std::runtime_error);
  expect_events({"execute"});
  expect_unpublished();
  EXPECT_EQ(proc::proc.get_status().app_id, 0);
}

TEST_F(NvHttpLaunchTransactionTest, FreshPublicationFailureTerminatesBeforeReleasingGpuAndLock) {
  rival_publication_ = true;
  const auto tree = dispatch(false, query());
  expect_error(tree, 503, "Another streaming handshake is already pending");
  EXPECT_FALSE(tree.get_optional<int>("root.gamesession"));
  expect_events({"execute", "terminate"});
  ASSERT_TRUE(candidate_);
  EXPECT_FALSE(candidate_->pending_live_gpu_lease);
  EXPECT_FALSE(gpu_workload::live_stream_active());
  EXPECT_EQ(proc::proc.get_status().app_id, 0);
  ASSERT_TRUE(rival_);
  EXPECT_TRUE(rtsp_stream::claim_launch_session_for_test(*rival_));
  EXPECT_FALSE(rtsp_stream::claim_launch_session_for_test(*candidate_));
}

TEST_F(NvHttpLaunchTransactionTest, FreshSuccessPublishesTheSameEncryptedLaunchAndTransfersItsLease) {
  const auto tree = dispatch(false, query());
  EXPECT_EQ(tree.get<int>("root.<xmlattr>.status_code"), 200);
  EXPECT_EQ(tree.get<int>("root.gamesession"), 1);
  EXPECT_FALSE(tree.get_optional<int>("root.resume"));
  EXPECT_EQ(tree.get<std::uint64_t>("root.hostsessionid"), fresh_token);
  EXPECT_EQ(tree.get<std::string>("root.sessionUrl0"), "rtspenc://127.0.0.1:" + std::to_string(net::map_port(rtsp_stream::RTSP_SETUP_PORT)));
  expect_events({"execute"});
  ASSERT_TRUE(candidate_);
  EXPECT_TRUE(candidate_->rtsp_cipher);
  EXPECT_EQ(candidate_->gcm_key.size(), 16U);
  EXPECT_EQ(candidate_->av_ping_payload.size(), 16U);
  EXPECT_EQ(candidate_->width, 1920);
  EXPECT_EQ(candidate_->height, 1080);
  EXPECT_EQ(candidate_->fps, 60000);
  ASSERT_TRUE(candidate_->pending_live_gpu_lease);
  EXPECT_TRUE(stream::session::platform_stop_deadline_for_test());
  EXPECT_TRUE(stream::session::platform_lifecycle_lock_available_for_test());
  ASSERT_TRUE(rtsp_stream::claim_launch_session_for_test(*candidate_));
  auto lease = rtsp_stream::take_claimed_live_gpu_lease_for_test(*candidate_);
  ASSERT_TRUE(lease);
  EXPECT_FALSE(candidate_->pending_live_gpu_lease);
  EXPECT_FALSE(gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs));
  lease.reset();
  EXPECT_FALSE(gpu_workload::live_stream_active());
}

TEST_F(NvHttpLaunchTransactionTest, PhysicalProbeFailureLeavesRetainedContractAndDeadlineUntouched) {
  retain(false);
  const auto deadline = existing_deadline();
  probe_result_ = -1;
  expect_error(dispatch(true, query(true, "&virtualDisplay=1")), 503, "Failed to initialize video capture/encoding. Is a display connected and turned on?");
  expect_events({"probe"});
  expect_unpublished();
  EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
  EXPECT_EQ(proc::proc.get_host_session_id(), retained_token);
  EXPECT_EQ(process_access::active_transport(proc::proc), original_->id);
  EXPECT_EQ(original_->width, 1280);
}

TEST_F(NvHttpLaunchTransactionTest, VirtualArAdmissionFailureDoesNotPauseOrReleaseAnUnreservedArLease) {
  retain(true);
  const auto deadline = existing_deadline();
  remote_start_result_ = false;
  expect_error(dispatch(true, query(true)), 503, "Local AR display ownership could not be released");
  expect_events({"ar-start"});
  expect_unpublished();
  EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
  EXPECT_EQ(process_access::retained_launch(proc::proc), original_);
}

TEST_F(NvHttpLaunchTransactionTest, FailedVirtualReconfigurePausesWithGpuHeldThenEndsArAfterUnlock) {
  retain(true);
  const auto deadline = existing_deadline();
  reconfigure_result_ = 409;
  expect_error(dispatch(true, query(true)), 409, "Failed to reconfigure the retained streaming session");
  expect_events({"ar-start", "reconfigure", "primary", "ar-end"});
  expect_unpublished();
  EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
  EXPECT_EQ(proc::proc.get_host_session_id(), retained_token);
  EXPECT_EQ(original_->width, 1280);
  EXPECT_EQ(process_access::active_transport(proc::proc), original_->id);
}

TEST_F(NvHttpLaunchTransactionTest, RejectedResumePublicationKeepsTheNewContractAndRestoresPrimary) {
  retain(true);
  const auto deadline = existing_deadline();
  revoke_publication_ = true;
  expect_error(dispatch(true, query(true)), 503, "Another streaming handshake is already pending");
  expect_events({"ar-start", "reconfigure", "ar-await", "primary", "ar-end"});
  expect_unpublished();
  ASSERT_TRUE(candidate_);
  EXPECT_EQ(candidate_->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
  EXPECT_EQ(process_access::retained_launch(proc::proc), original_);
  EXPECT_EQ(proc::proc.get_host_session_id(), retained_token);
  EXPECT_EQ(original_->width, 1920);
  EXPECT_EQ(original_->height, 1080);
  EXPECT_EQ(original_->fps, 60000);
  EXPECT_EQ(process_access::active_transport(proc::proc), candidate_->id);
  EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
}

TEST_F(NvHttpLaunchTransactionTest, RivalHandshakeDefersResumePrimaryRollbackAndIsNotReplaced) {
  retain(true);
  const auto deadline = existing_deadline();
  rival_publication_ = true;
  expect_error(dispatch(true, query(true)), 503, "Another streaming handshake is already pending");
  // The real guard defers primary recovery while any validated handshake is pending.
  expect_events({"ar-start", "reconfigure", "ar-await", "ar-end"});
  EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
  EXPECT_EQ(original_->width, 1920);
  ASSERT_TRUE(candidate_);
  EXPECT_EQ(process_access::active_transport(proc::proc), candidate_->id);
  EXPECT_FALSE(candidate_->pending_live_gpu_lease);
  EXPECT_FALSE(gpu_workload::live_stream_active());
  ASSERT_TRUE(rival_);
  EXPECT_TRUE(rtsp_stream::claim_launch_session_for_test(*rival_));
  EXPECT_FALSE(rtsp_stream::claim_launch_session_for_test(*candidate_));
}

TEST_F(NvHttpLaunchTransactionTest, ResumeSuccessUsesRetainedDisplayKindAndRenewsThePublishedHandshake) {
  for (const bool virtual_display : {false, true}) {
    SCOPED_TRACE(virtual_display);
    retain(virtual_display);
    const auto tree = dispatch(true, query(true));
    EXPECT_EQ(tree.get<int>("root.<xmlattr>.status_code"), 200);
    EXPECT_EQ(tree.get<int>("root.resume"), 1);
    EXPECT_FALSE(tree.get_optional<int>("root.gamesession"));
    EXPECT_EQ(tree.get<std::uint64_t>("root.hostsessionid"), retained_token);
    EXPECT_EQ(tree.get<std::string>("root.sessionUrl0"), "rtspenc://127.0.0.1:" + std::to_string(net::map_port(rtsp_stream::RTSP_SETUP_PORT)));
    if (virtual_display) {
      // Requested virtualDisplay defaults to false, but the retained source still needs AR.
      expect_events({"ar-start", "reconfigure", "ar-await"});
    } else {
      expect_events({"probe", "reconfigure"});
    }
    ASSERT_TRUE(candidate_);
    EXPECT_EQ(candidate_->virtual_display, virtual_display);
    EXPECT_EQ(process_access::retained_launch(proc::proc), original_);
    EXPECT_EQ(process_access::active_transport(proc::proc), candidate_->id);
    EXPECT_EQ(proc::proc.get_host_session_id(), retained_token);
    EXPECT_TRUE(candidate_->pending_live_gpu_lease);
    EXPECT_TRUE(stream::session::platform_stop_deadline_for_test());
    ASSERT_TRUE(rtsp_stream::claim_launch_session_for_test(*candidate_));
    rtsp_stream::finish_launch_session_for_test(*candidate_, false);
    EXPECT_EQ(candidate_->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
    expect_unpublished();
    stream::session::flush_platform_state();
    process_access::clear(proc::proc);
    edges_.clear();
  }
}
