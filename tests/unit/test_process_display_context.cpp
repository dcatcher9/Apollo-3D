/**
 * @file tests/unit/test_process_display_context.cpp
 * @brief Actual process/RTSP retained-display alias and transport boundaries.
 */
// Keep Winsock2 ahead of every Windows include, including the shared test declarations.
#include <boost/asio.hpp>

#include "../tests_common.h"

#include "src/process_display_context_test.h"
#include "src/uuid.h"
#include "src/video.h"

#include <boost/process/v1/environment.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace {
  using access = proc::process_display_context_test_access;
  using snapshot_t = proc::testing::display_snapshot_t;
  using operation_e = proc::display_topology_test_operation_e;
  using tcp = boost::asio::ip::tcp;

  void expect_display(const snapshot_t &actual, const snapshot_t &expected) {
    EXPECT_EQ(actual.device_name, expected.device_name);
    EXPECT_EQ(actual.unique_id, expected.unique_id);
    EXPECT_EQ(actual.width, expected.width);
    EXPECT_EQ(actual.height, expected.height);
    EXPECT_EQ(actual.fps, expected.fps);
    EXPECT_EQ(actual.enable_hdr, expected.enable_hdr);
    EXPECT_EQ(actual.virtual_display, expected.virtual_display);
    EXPECT_EQ(actual.virtual_display_only, expected.virtual_display_only);
    EXPECT_EQ(actual.scale_factor, expected.scale_factor);
    EXPECT_EQ(actual.sbs_mode, expected.sbs_mode);
    EXPECT_EQ(std::memcmp(&actual.display_guid, &expected.display_guid, sizeof(GUID)), 0);
  }

  std::shared_ptr<rtsp_stream::launch_session_t> make_launch(std::uint32_t id) {
    auto launch = std::make_shared<rtsp_stream::launch_session_t>();
    launch->id = id;
    snapshot_t display {};
    display.device_name = "original client";
    display.unique_id = "01234567-89ab-cdef-0123-456789abcdef";
    display.width = 1920;
    display.height = 1080;
    display.fps = 60000;
    display.scale_factor = 100;
    access::set_display(*launch, display);
    launch->gcm_key.assign(16, 0x36);
    launch->rtsp_cipher.emplace(launch->gcm_key, false);
    launch->rtsp_iv_counter = 0;
    launch->av_ping_payload = "0123456789abcdef";
    launch->host_audio = false;
    launch->surround_info = 2;
    return launch;
  }

  proc::ctx_t desktop(bool synthetic_virtual = false) {
    proc::ctx_t app {};
    app.id = "7";
    app.name = "inert context desktop";
    app.uuid = "8f15f45a-5555-4444-8888-636f6e746578";
    app.synthetic_virtual_display = synthetic_virtual;
    app.exit_timeout = 0s;
    return app;
  }

  // All asynchronous operations are local and canceled/drained before their borrowed state dies.
  class encrypted_peer_t {
  public:
    encrypted_peer_t():
        acceptor_(io_, {boost::asio::ip::address_v4::loopback(), 0}),
        sender_(io_),
        receiver_(io_) {
      bool accepted = false;
      bool connected = false;
      boost::system::error_code accept_error;
      boost::system::error_code connect_error;
      auto drain = util::fail_guard([&]() {
        close();
        io_.restart();
        io_.run();
      });
      acceptor_.async_accept(sender_, [&](auto error) {
        accept_error = error;
        accepted = true;
      });
      receiver_.async_connect(acceptor_.local_endpoint(), [&](auto error) {
        connect_error = error;
        connected = true;
      });
      io_.run_for(3s);
      if (!accepted || !connected || accept_error || connect_error) {
        throw std::runtime_error("The inert RTSP loopback did not connect before its deadline");
      }
      // The real sender cannot block indefinitely even if this fixture breaks its receive side.
      sender_.non_blocking(true);
      drain.disable();
    }

    ~encrypted_peer_t() {
      close();
      io_.restart();
      io_.run();
    }

    int announce(rtsp_stream::launch_session_t &launch, std::string_view payload) {
      rtsp_stream::invoke_stale_announce_for_test(sender_, launch, payload);
      std::array<std::uint8_t, 24> header {};
      read(header.data(), header.size());
      std::uint32_t type_and_length;
      std::uint32_t sequence;
      std::memcpy(&type_and_length, header.data(), sizeof(type_and_length));
      std::memcpy(&sequence, header.data() + 4, sizeof(sequence));
      type_and_length = util::endian::big(type_and_length);
      sequence = util::endian::big(sequence);
      if (!(type_and_length & 0x80000000U)) {
        throw std::runtime_error("RTSP response was not encrypted");
      }
      const auto size = type_and_length & 0x7fffffffU;
      if (size == 0 || size > 4096) {
        throw std::runtime_error("RTSP response length is outside the fixture bound");
      }
      std::vector<std::uint8_t> tagged(16 + size);
      std::copy(header.begin() + 8, header.end(), tagged.begin());
      read(tagged.data() + 16, size);
      crypto::aes_t iv(12);
      std::memcpy(iv.data(), &sequence, sizeof(sequence));
      iv[10] = 'H';
      iv[11] = 'R';
      crypto::cipher::gcm_t decoder {launch.gcm_key, false};
      std::vector<std::uint8_t> plaintext;
      if (decoder.decrypt(std::string_view {reinterpret_cast<const char *>(tagged.data()), tagged.size()}, plaintext, &iv) != 0) {
        throw std::runtime_error("RTSP response authentication failed");
      }
      const std::string response(plaintext.begin(), plaintext.end());
      if (response.starts_with("RTSP/1.0 400 ")) {
        return 400;
      }
      if (response.starts_with("RTSP/1.0 454 ")) {
        return 454;
      }
      throw std::runtime_error("Unexpected stale RTSP response: " + response);
    }

  private:
    void close() {
      boost::system::error_code ignored;
      acceptor_.close(ignored);
      sender_.close(ignored);
      receiver_.close(ignored);
    }

    void read(std::uint8_t *data, std::size_t size) {
      bool finished = false;
      boost::system::error_code result;
      auto drain = util::fail_guard([&]() {
        close();
        io_.restart();
        io_.run();
      });
      io_.restart();
      boost::asio::async_read(receiver_, boost::asio::buffer(data, size), [&](auto error, auto) {
        result = error;
        finished = true;
      });
      io_.run_for(3s);
      if (!finished || result) {
        throw std::runtime_error("RTSP response did not arrive before its deadline");
      }
      drain.disable();
    }

    boost::asio::io_context io_;
    tcp::acceptor acceptor_;
    tcp::socket sender_;
    tcp::socket receiver_;
  };

  class ProcessDisplayContextTest: public testing::Test {
  protected:
    void SetUp() override {
      ASSERT_EQ(proc::proc.get_status().app_id, 0);
      saved_global_ = std::move(proc::proc);
      proc::proc = proc::proc_t {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
      saved_driver_ = proc::vDisplayDriverStatus.exchange(VDISPLAY::DRIVER_STATUS::OK);
      saved_apps_ = config::stream.file_apps;
      saved_output_ = config::video.output_name;
      globals_saved_ = true;
      apps_ = std::filesystem::temp_directory_path() /
              ("apollo-context-" + uuid_util::uuid_t::generate().string() + ".json");
      std::ofstream file(apps_);
      ASSERT_TRUE(file.is_open());
      file << R"({"version":2,"env":{},"apps":[]})";
      file.close();
      config::stream.file_apps = apps_.string();
      mode_.dmPelsWidth = 1920;
      mode_.dmPelsHeight = 1080;
      mode_.dmDisplayFrequency = 60;

      edges_.encoder_probing_allowed = [&]() {
        return probing_allowed_;
      };
      edges_.recover_primary = [&]() {
        ++recoveries_;
        return true;
      };
      edges_.remote_starting = [&](auto, auto, bool setup) {
        EXPECT_TRUE(setup);
        ++remote_starts_;
        return true;
      };
      edges_.remote_awaiting = [&](auto, auto) {
        ++remote_awaiting_;
      };
      edges_.remote_ended = [&](auto) {
        ++remote_ends_;
      };
      edges_.map_display_name = [](const auto &) {
        return "inert-context-device";
      };
      edges_.capture_display_names = [] {
        return std::vector<std::string> {};
      };
      edges_.start_hdr_worker = [&](bool enabled) {
        hdr_starts_.push_back(enabled);
      };
      edges_.display_io.prepare = [&](const auto &spec) {
        prepared_.push_back(spec);
        if (auto launch = observed_.lock()) {
          shared_at_prepare_ = access::shares_display_context(process_, *launch);
          alias_at_prepare_ = access::retained(process_);
        }
        return true;
      };
      edges_.display_io.create = [&](const auto &) {
        ++creations_;
        VDISPLAY::creation_result_t result;
        if (acquire_succeeds_ && creations_ == expected_creation_) {
          result.identity.emplace();
          result.identity->TargetId = 37;
          result.display_name = L"inert-context-display";
          result.device_path = L"inert-context-path";
        }
        return result;
      };
      edges_.display_io.bind = [](auto) {
        return true;
      };
      edges_.display_io.promote = [](auto, bool) {
        return true;
      };
      edges_.display_io.restore = [](auto) {
        return true;
      };
      edges_.display_io.pause = [&](auto, auto &retained, auto) {
        ++pauses_;
        auto storage = std::make_shared<int>(0);
        retained = platf::primary_display::retained_display_ptr(storage, reinterpret_cast<platf::primary_display::retained_display_t *>(storage.get()));
        return true;
      };
      edges_.display_io.reactivate = [](const auto &, bool) {
        return true;
      };
      edges_.display_io.query = [](const auto &) {
        VDISPLAY::display_identity_query_t result;
        result.state = VDISPLAY::display_identity_state_e::present;
        result.display_name = L"inert-context-display";
        result.device_path = L"inert-context-path";
        return result;
      };
      edges_.display_io.query_retirement = [](const auto &) {
        return VDISPLAY::display_identity_state_e::absent;
      };
      edges_.display_io.remove = [](const auto &) {
        return true;
      };
      edges_.display_io.now = [] {
        return std::chrono::steady_clock::now();
      };
      edges_.display_io.sleep = [](auto) {
      };
      edges_.display_io.restore_cursor = [](const auto &) {
      };
      edges_.topology = [&](operation_e operation, bool desired) {
        operations_.push_back(operation);
        return !(fail_hdr_ && operation == operation_e::request_hdr && desired);
      };
      edges_.query_mode = [&](auto) {
        return std::optional {mode_};
      };
      edges_.change_mode = [&](auto, int width, int height, int fps, bool probe) {
        if (!probe) {
          mode_.dmPelsWidth = width;
          mode_.dmPelsHeight = height;
          mode_.dmDisplayFrequency = (fps + 500) / 1000;
        }
        return DISP_CHANGE_SUCCESSFUL;
      };
      edges_.record_mode = [](auto, const auto &) {
        return DISP_CHANGE_SUCCESSFUL;
      };
      edges_.baseline_active = [](auto) {
        return true;
      };
      edges_.settings_unavailable = [] {
        return false;
      };
      access::configure(process_, edges_);
    }

    void TearDown() override {
      if (!globals_saved_) {
        return;
      }
      for (const auto &launch : published_) {
        rtsp_stream::finish_launch_session_for_test(*launch, false);
      }
      rtsp_stream::clear_pending_launch_session();
      process_.terminate(false, false);
      proc::proc.terminate(false, false);
      proc::proc = std::move(saved_global_);
      proc::vDisplayDriverStatus.store(saved_driver_);
      config::stream.file_apps = saved_apps_;
      config::video.output_name = saved_output_;
      if (!apps_.empty()) {
        std::error_code ignored;
        std::filesystem::remove(apps_, ignored);
      }
    }

    int execute(const std::shared_ptr<rtsp_stream::launch_session_t> &launch, bool synthetic = false) {
      observed_ = launch;
      expected_creation_ = creations_ + 1;
      return access::execute(process_, desktop(synthetic), launch);
    }

    void publish(const std::shared_ptr<rtsp_stream::launch_session_t> &launch) {
      published_.push_back(launch);
      ASSERT_TRUE(rtsp_stream::launch_session_raise(launch));
    }

    proc::testing::native_edges_t edges_;
    proc::proc_t saved_global_;
    bool globals_saved_ = false;
    VDISPLAY::DRIVER_STATUS saved_driver_ {};
    std::string saved_apps_;
    std::string saved_output_;
    std::filesystem::path apps_;
    std::weak_ptr<rtsp_stream::launch_session_t> observed_;
    std::vector<std::shared_ptr<rtsp_stream::launch_session_t>> published_;
    bool probing_allowed_ = true;
    bool acquire_succeeds_ = true;
    bool fail_hdr_ = false;
    bool shared_at_prepare_ = false;
    std::optional<snapshot_t> alias_at_prepare_;
    DEVMODEW mode_ {};
    int creations_ = 0;
    int expected_creation_ = 1;
    int recoveries_ = 0;
    int pauses_ = 0;
    int remote_starts_ = 0;
    int remote_awaiting_ = 0;
    int remote_ends_ = 0;
    std::vector<bool> hdr_starts_;
    std::vector<VDISPLAY::display_spec_t> prepared_;
    std::vector<operation_e> operations_;
    proc::proc_t process_ {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  };

  TEST_F(ProcessDisplayContextTest, ActualExecutePublishesScaledAliasAndOneShotAudioEnvironment) {
    for (const bool host_audio : {false, true}) {
      for (const auto [channels, label] : std::array {std::pair {2, "2.0"}, std::pair {0x10006, "5.1"}, std::pair {8, "7.1"}}) {
        SCOPED_TRACE(channels);
        SCOPED_TRACE(host_audio);
        auto launch = make_launch(11);
        auto expected = access::display(*launch);
        expected.scale_factor = 75;
        expected.enable_hdr = !host_audio;
        access::set_display(*launch, expected);
        launch->host_audio = host_audio;
        launch->surround_info = channels;
        ASSERT_EQ(execute(launch), 0);
        expected.width = 1440;
        expected.height = 810;
        ASSERT_TRUE(access::retained(process_));
        expect_display(*access::retained(process_), expected);
        expect_display(access::display(*launch), expected);
        EXPECT_TRUE(access::shares_display_context(process_, *launch));
        EXPECT_EQ(access::active_transport(process_), 11U);
        EXPECT_NE(process_.get_host_session_id(), 0U);
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_UUID"), expected.unique_id);
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_NAME"), expected.device_name);
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_WIDTH"), "1440");
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_HEIGHT"), "810");
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_RENDER_WIDTH"), "1440");
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_RENDER_HEIGHT"), "810");
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_SCALE_FACTOR"), "75");
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_FPS"), "60.000");
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_HDR"), host_audio ? "false" : "true");
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_HOST_AUDIO"), host_audio ? "true" : "false");
        EXPECT_EQ(access::environment(process_, "APOLLO_CLIENT_AUDIO_CONFIGURATION"), label);
        process_.terminate(false, false);
      }
    }
    EXPECT_EQ(creations_, 0);
    EXPECT_EQ(remote_starts_, 0);
    EXPECT_EQ(recoveries_, 0);
  }

  TEST_F(ProcessDisplayContextTest, ForcedVirtualExecuteSharesNormalizationGuidAndActualOwnershipWrites) {
    for (const bool synthetic : {false, true}) {
      probing_allowed_ = synthetic;
      auto launch = make_launch(11);
      auto expected = access::display(*launch);
      expected.width = 1600;
      expected.height = 900;
      expected.scale_factor = 80;
      expected.virtual_display_only = synthetic;
      access::set_display(*launch, expected);
      ASSERT_EQ(execute(launch, synthetic), 0);
      expected.width = 1280;
      expected.height = 720;
      const auto uuid = uuid_util::uuid_t::parse(expected.unique_id);
      std::memcpy(&expected.display_guid, &uuid, sizeof(GUID));
      ASSERT_TRUE(alias_at_prepare_);
      EXPECT_TRUE(shared_at_prepare_);
      expect_display(*alias_at_prepare_, expected);
      expected.virtual_display = true;
      expect_display(access::display(*launch), expected);
      const auto retained = access::retained(process_);
      ASSERT_TRUE(retained);
      expect_display(*retained, expected);
      ASSERT_FALSE(prepared_.empty());
      EXPECT_EQ(prepared_.back().width, 1280U);
      EXPECT_EQ(prepared_.back().height, 720U);
      EXPECT_EQ(prepared_.back().refresh_millihz, 60000U);
      EXPECT_EQ(prepared_.back().exclusive, synthetic);
      EXPECT_EQ(std::memcmp(&prepared_.back().guid, &expected.display_guid, sizeof(GUID)), 0);
      EXPECT_TRUE(process_.get_status().virtual_display);
      process_.terminate(false, false);
    }
    EXPECT_EQ(creations_, 2);
    EXPECT_EQ(remote_starts_, 2);
    EXPECT_EQ(remote_awaiting_, 2);
    EXPECT_EQ(remote_ends_, 2);
  }

  TEST_F(ProcessDisplayContextTest, FailedActualAcquireLeavesCallerWritesButClearsProcessOwnership) {
    acquire_succeeds_ = false;
    auto launch = make_launch(11);
    auto expected = access::display(*launch);
    expected.virtual_display = true;
    expected.scale_factor = 50;
    access::set_display(*launch, expected);
    ASSERT_EQ(execute(launch), 503);
    EXPECT_TRUE(shared_at_prepare_);
    expected.width = 960;
    expected.height = 540;
    const auto uuid = uuid_util::uuid_t::parse(expected.unique_id);
    std::memcpy(&expected.display_guid, &uuid, sizeof(GUID));
    expected.virtual_display = false;
    expect_display(access::display(*launch), expected);
    EXPECT_FALSE(access::retained(process_));
    EXPECT_EQ(process_.get_status().app_id, 0);
    EXPECT_EQ(process_.get_host_session_id(), 0U);
    EXPECT_EQ(access::active_transport(process_), 0U);
    EXPECT_EQ(creations_, 1);
    EXPECT_EQ(remote_ends_, 1);
    EXPECT_EQ(recoveries_, 2);
  }

  TEST_F(ProcessDisplayContextTest, RealResumeUpdatesOnlyMutableFieldsAcrossOriginalAndIncomingAliases) {
    auto original = make_launch(11);
    ASSERT_EQ(execute(original, true), 0);
    const auto host_token = process_.get_host_session_id();
    const auto identity = access::display(*original);
    auto resumed = make_launch(22);
    auto requested = access::display(*resumed);
    requested.unique_id = "fedcba98-7654-3210-fedc-ba9876543210";
    requested.device_name = "successor client";
    requested.width = 1600;
    requested.height = 900;
    requested.fps = 90000;
    requested.enable_hdr = true;
    requested.virtual_display_only = true;
    requested.scale_factor = 80;
    requested.sbs_mode = video::SBS_GAME_MONO;
    access::set_display(*resumed, requested);
    resumed->stream_gamma = video::stream_gamma_mode_e::gamma_2_4;
    ASSERT_EQ(access::reconfigure(process_, resumed), 0);
    auto expected = requested;
    expected.width = 1280;
    expected.height = 720;
    expected.virtual_display = true;
    expected.display_guid = identity.display_guid;
    expect_display(access::display(*resumed), expected);
    expected.unique_id = identity.unique_id;
    expected.device_name = identity.device_name;
    expect_display(access::display(*original), expected);
    const auto retained = access::retained(process_);
    ASSERT_TRUE(retained);
    expect_display(*retained, expected);
    EXPECT_TRUE(access::shares_display_context(process_, *original));
    EXPECT_FALSE(access::shares_display_context(process_, *resumed));
    EXPECT_EQ(original->id, 11U);
    EXPECT_EQ(resumed->id, 22U);
    EXPECT_EQ(original->stream_gamma, video::stream_gamma_mode_e::windows_default);
    EXPECT_EQ(process_.get_host_session_id(), host_token);
    EXPECT_EQ(access::active_transport(process_), 22U);
    EXPECT_EQ(creations_, 1);
  }

  TEST_F(ProcessDisplayContextTest, FailedRealResumeRestoresContractAndRejectsFailedTransportIdentity) {
    auto original = make_launch(11);
    ASSERT_EQ(execute(original, true), 0);
    const auto before = access::display(*original);
    const auto host_token = process_.get_host_session_id();
    auto resumed = make_launch(22);
    auto requested = access::display(*resumed);
    requested.width = 1280;
    requested.height = 720;
    requested.fps = 90000;
    requested.enable_hdr = true;
    requested.virtual_display_only = true;
    requested.sbs_mode = video::SBS_GAME_MONO;
    access::set_display(*resumed, requested);
    fail_hdr_ = true;
    const auto pauses_before = pauses_;
    ASSERT_EQ(access::reconfigure(process_, resumed), 503);
    expect_display(access::display(*original), before);
    const auto retained = access::retained(process_);
    ASSERT_TRUE(retained);
    expect_display(*retained, before);
    expect_display(access::display(*resumed), requested);
    EXPECT_EQ(process_.get_host_session_id(), host_token);
    EXPECT_EQ(access::active_transport(process_), 11U);
    EXPECT_EQ(process_.apply_live_video_mode(1920, 1080, 60000, 22), proc::live_video_mode_result_e::needs_reconnect);
    EXPECT_EQ(creations_, 1);
    EXPECT_GT(pauses_, pauses_before);
    EXPECT_EQ(process_.get_status().virtual_display_only, before.virtual_display_only);
  }

  TEST_F(ProcessDisplayContextTest, LiveModeWritesOriginalSharedRecordAndOldTransportCannotResizeSuccessor) {
    auto original = make_launch(11);
    ASSERT_EQ(execute(original, true), 0);
    auto resumed = make_launch(22);
    ASSERT_EQ(access::reconfigure(process_, resumed), 0);
    const auto before = access::display(*original);
    EXPECT_EQ(process_.apply_live_video_mode(1280, 720, 90000, 11), proc::live_video_mode_result_e::needs_reconnect);
    expect_display(access::display(*original), before);
    ASSERT_EQ(process_.apply_live_video_mode(1280, 720, 90000, 22), proc::live_video_mode_result_e::applied);
    auto expected = before;
    expected.width = 1280;
    expected.height = 720;
    expected.fps = 90000;
    expect_display(access::display(*original), expected);
    const auto retained = access::retained(process_);
    ASSERT_TRUE(retained);
    expect_display(*retained, expected);
    // The incoming resume transport was a separate projection, never the retained authority.
    EXPECT_EQ(access::display(*resumed).fps, 60000);
    EXPECT_EQ(creations_, 1);
  }

  TEST_F(ProcessDisplayContextTest, RealReservationsRejectOldAliasAndKeepSuccessorClaimIndependent) {
    auto original = make_launch(11);
    ASSERT_EQ(execute(original), 0);
    publish(original);
    ASSERT_TRUE(rtsp_stream::claim_launch_session_for_test(*original));
    rtsp_stream::finish_launch_session_for_test(*original, true);
    rtsp_stream::clear_pending_launch_session();
    auto resumed = make_launch(22);
    ASSERT_EQ(access::reconfigure(process_, resumed), 0);
    publish(resumed);
    EXPECT_FALSE(rtsp_stream::claim_launch_session_for_test(*original));
    EXPECT_TRUE(rtsp_stream::claim_launch_session_for_test(*resumed));
    EXPECT_TRUE(access::shares_display_context(process_, *original));
    EXPECT_FALSE(access::shares_display_context(process_, *resumed));
    EXPECT_EQ(access::active_transport(process_), 22U);
    EXPECT_EQ(original->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
  }

  TEST_F(ProcessDisplayContextTest, StaleEncryptedAnnounceObservesRealResumeBeforeRejectingItsReservation) {
    auto original = make_launch(11);
    auto display = access::display(*original);
    display.sbs_mode = video::SBS_GAME_MONO;
    access::set_display(*original, display);
    ASSERT_EQ(execute(original), 0);
    publish(original);
    rtsp_stream::clear_pending_launch_session();
    ASSERT_EQ(original->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
    const std::string payload =
      "a=x-nv-audio.surround.numChannels:2\r\n"
      "a=x-nv-audio.surround.channelMask:3\r\n"
      "a=x-nv-aqos.packetDuration:5\r\n"
      "a=x-nv-audio.surround.AudioQuality:0\r\n"
      "a=x-nv-general.useReliableUdp:13\r\n"
      "a=x-nv-video[0].packetSize:1392\r\n"
      "a=x-nv-vqos[0].fec.minRequiredFecPackets:2\r\n"
      "a=x-ml-general.featureFlags:2\r\n"
      "a=x-nv-aqos.qosTrafficType:0\r\n"
      "a=x-nv-vqos[0].qosTrafficType:0\r\n"
      "a=x-ss-general.encryptionEnabled:7\r\n"
      "a=x-nv-video[0].clientViewportHt:1080\r\n"
      "a=x-nv-video[0].clientViewportWd:1920\r\n"
      "a=x-nv-video[0].maxFPS:60\r\n"
      "a=x-nv-video[0].clientRefreshRateX100:6000\r\n"
      "a=x-nv-vqos[0].bw.maximumBitrateKbps:10000\r\n"
      "a=x-nv-video[0].videoEncoderSlicesPerFrame:1\r\n"
      "a=x-nv-video[0].maxNumReferenceFrames:1\r\n"
      "a=x-nv-video[0].encoderCscMode:0\r\n"
      "a=x-nv-vqos[0].bitStreamFormat:0\r\n"
      "a=x-nv-video[0].dynamicRangeMode:0\r\n"
      "a=x-ss-video[0].chromaSamplingType:0\r\n"
      "a=x-ml-video.configuredBitrateKbps:10000\r\n";
    encrypted_peer_t peer;
    EXPECT_EQ(peer.announce(*original, payload), 400);
    auto resumed = make_launch(22);
    ASSERT_EQ(access::reconfigure(process_, resumed), 0);
    EXPECT_EQ(access::display(*original).sbs_mode, video::SBS_OFF);
    EXPECT_EQ(peer.announce(*original, payload), 454);
    EXPECT_EQ(original->rtsp_iv_counter, 2U);
    EXPECT_EQ(original->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
    EXPECT_EQ(access::active_transport(process_), 22U);
  }

  TEST_F(ProcessDisplayContextTest, RetainedDisplayOutlivesRevokedTransportWithoutRetainingItsResources) {
    auto original = make_launch(11);
    ASSERT_EQ(execute(original, true), 0);
    const auto before = access::retained(process_);
    ASSERT_TRUE(before);
    const auto token = process_.get_host_session_id();
    std::weak_ptr transport = original;
    std::weak_ptr context = original->display_context;
    ASSERT_TRUE(original->reserve_live_gpu());
    EXPECT_TRUE(gpu_workload::live_stream_active());
    publish(original);
    ASSERT_TRUE(rtsp_stream::claim_launch_session_for_test(*original));
    rtsp_stream::finish_launch_session_for_test(*original, false);
    rtsp_stream::clear_pending_launch_session();
    // GPU revocation is explicit existing RTSP policy, before transport lifetime changes.
    EXPECT_FALSE(gpu_workload::live_stream_active());
    EXPECT_FALSE(transport.expired());
    published_.clear();
    original.reset();
    EXPECT_TRUE(transport.expired());
    EXPECT_FALSE(context.expired());
    const auto retained = access::retained(process_);
    ASSERT_TRUE(retained);
    expect_display(*retained, *before);
    EXPECT_EQ(process_.get_host_session_id(), token);
    EXPECT_EQ(access::active_transport(process_), 11U);
    EXPECT_TRUE(process_.get_status().virtual_display);
    ASSERT_EQ(process_.apply_live_video_mode(1280, 720, 90000, 11), proc::live_video_mode_result_e::applied);
    const auto changed = access::retained(process_);
    ASSERT_TRUE(changed);
    auto expected = *before;
    expected.width = 1280;
    expected.height = 720;
    expected.fps = 90000;
    expect_display(*changed, expected);
    process_.terminate(false, false);
    EXPECT_TRUE(context.expired());
  }
}  // namespace
