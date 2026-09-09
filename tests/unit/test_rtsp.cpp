/**
 * @file tests/unit/test_rtsp.cpp
 * @brief Tests for RTSP request parsing helpers.
 */

#include <algorithm>
#include <atomic>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <winsock2.h>

// Keep Boost.Asio/Winsock2 ahead of rtsp.h's Windows headers.
// clang-format off
#include <src/stream.h>
#include <src/rtsp.h>
#include "../tests_common.h"
// clang-format on

namespace rtsp_stream {
  namespace detail {
    std::unordered_map<std::string_view, std::string_view> parse_announce_attributes(std::string_view payload);
  }
}  // namespace rtsp_stream

namespace {
  std::shared_ptr<rtsp_stream::launch_session_t> make_modern_launch_session(
    std::uint32_t id,
    const std::string &unique_id
  ) {
    auto session = std::make_shared<rtsp_stream::launch_session_t>();
    session->id = id;
    session->unique_id = unique_id;
    session->gcm_key.assign(16, 0);
    session->rtsp_cipher.emplace(session->gcm_key, false);
    session->rtsp_iv_counter = 0;
    session->av_ping_payload = "0123456789abcdef";
    return session;
  }

  class lifecycle_microphone_sink_t: public microphone::pcm_sink_t {
  public:
    explicit lifecycle_microphone_sink_t(std::shared_ptr<std::atomic<unsigned>> destroyed):
        destroyed_(std::move(destroyed)) {}

    ~lifecycle_microphone_sink_t() override {
      ++*destroyed_;
    }

    bool open(const std::string &, std::string &) override {
      return true;
    }

    int write(std::span<const std::int16_t> pcm) override {
      return static_cast<int>(pcm.size());
    }

  private:
    std::shared_ptr<std::atomic<unsigned>> destroyed_;
  };

  std::shared_ptr<microphone::session_t> make_lifecycle_microphone(
    const rtsp_stream::launch_session_t &launch,
    std::shared_ptr<std::atomic<unsigned>> destroyed,
    std::uint16_t port = 0
  ) {
    microphone::options_t options;
    options.bind_address = "127.0.0.1";
    options.remote_address = "127.0.0.1";
    options.sink_endpoint = "injected-lifecycle-test-sink";
    options.port = port;
    options.encrypted = true;
    std::copy_n(launch.av_ping_payload.begin(), 16, options.ping_payload.begin());
    std::string error;
    auto receiver = microphone::session_t::start(options, error, [destroyed] {
      return std::make_unique<lifecycle_microphone_sink_t>(destroyed);
    });
    EXPECT_TRUE(receiver) << error;
    return receiver;
  }
}  // namespace

TEST(RtspAnnounceParsingTest, HandlesEmptyAndMalformedAttributes) {
  const auto args = rtsp_stream::detail::parse_announce_attributes("a=empty:\r\na=missing-colon\r\na=:value\r\n");

  const auto empty = args.find("empty");
  ASSERT_NE(empty, args.end());
  EXPECT_TRUE(empty->second.empty());
  EXPECT_EQ(args.count("missing-colon"), 0);
  EXPECT_EQ(args.count(""), 0);
}

TEST(RtspAnnounceParsingTest, RetainsFinalLineAndTrimsTrailingSpaces) {
  const auto args = rtsp_stream::detail::parse_announce_attributes(
    "v=0\r\n\r\na=first:1  \r\na=last:2"
  );

  ASSERT_EQ(args.size(), 2);
  EXPECT_EQ(args.at("first"), "1");
  EXPECT_EQ(args.at("last"), "2");
}

TEST(RtspAnnounceParsingTest, EnforcesNumericSyntaxAndProtocolBounds) {
  using field = rtsp_stream::detail::announce_int_field;
  using rtsp_stream::detail::parse_announce_int;

  EXPECT_EQ(parse_announce_int(field::audio_channels, "2"), 2);
  EXPECT_EQ(parse_announce_int(field::audio_channels, "6"), 6);
  EXPECT_EQ(parse_announce_int(field::audio_channels, "8"), 8);
  EXPECT_FALSE(parse_announce_int(field::audio_channels, "7"));
  EXPECT_EQ(parse_announce_int(field::audio_packet_duration, "10"), 10);
  EXPECT_FALSE(parse_announce_int(field::audio_packet_duration, "20"));
  EXPECT_EQ(parse_announce_int(field::control_protocol, "13"), 13);
  EXPECT_FALSE(parse_announce_int(field::control_protocol, "1"));
  EXPECT_FALSE(parse_announce_int(field::control_protocol, "0"));
  EXPECT_EQ(parse_announce_int(field::video_format, "2"), 2);
  EXPECT_FALSE(parse_announce_int(field::video_format, "3"));
  EXPECT_EQ(parse_announce_int(field::binary_option, "1"), 1);
  EXPECT_FALSE(parse_announce_int(field::binary_option, "2"));
  EXPECT_EQ(parse_announce_int(field::encryption_flags, "7"), 7);
  EXPECT_EQ(parse_announce_int(field::encryption_flags, "15"), 15);
  EXPECT_FALSE(parse_announce_int(field::encryption_flags, "16"));
  EXPECT_FALSE(parse_announce_int(field::max_fps, "0"));
  EXPECT_EQ(parse_announce_int(field::max_fps, "1000000"), 1000000);
  EXPECT_FALSE(parse_announce_int(field::max_fps, "1000001"));
  EXPECT_EQ(parse_announce_int(field::client_refresh_x100, "0"), 0);
  EXPECT_EQ(parse_announce_int(field::client_refresh_x100, "100000"), 100000);
  EXPECT_FALSE(parse_announce_int(field::client_refresh_x100, "100001"));
  EXPECT_FALSE(parse_announce_int(field::bitrate_kbps, "12kbps"));
  EXPECT_FALSE(parse_announce_int(field::feature_flags, "-1"));
  EXPECT_FALSE(parse_announce_int(field::viewport_dimension, "16385"));
  EXPECT_FALSE(parse_announce_int(field::configured_bitrate_kbps, "2147483648"));
}

TEST(RtspAnnounceParsingTest, UsesClientRefreshOnlyWhenItMatchesStreamCadence) {
  using rtsp_stream::detail::validated_client_refresh_x100;

  EXPECT_EQ(validated_client_refresh_x100(60, 5994), 5994);
  EXPECT_EQ(validated_client_refresh_x100(59940, 5994), 5994);
  EXPECT_EQ(validated_client_refresh_x100(120, 11988), 11988);
  EXPECT_EQ(validated_client_refresh_x100(119, 12000), 0);
  EXPECT_EQ(validated_client_refresh_x100(60, 9000), 0);
  EXPECT_EQ(validated_client_refresh_x100(240, 6000), 0);
  EXPECT_EQ(validated_client_refresh_x100(239760, 5994), 0);
  EXPECT_EQ(validated_client_refresh_x100(1001, 100), 0);
  EXPECT_EQ(validated_client_refresh_x100(4000, 400), 0);
  EXPECT_EQ(validated_client_refresh_x100(60, 0), 0);
}

TEST(RtspAnnounceParsingTest, EnforcesWarpAndEncoderArithmeticBounds) {
  using rtsp_stream::detail::calculate_warp_bitrate_factor;
  using rtsp_stream::detail::is_safe_encoder_bitrate;

  EXPECT_EQ(calculate_warp_bitrate_factor(60, 60000), 1);
  EXPECT_EQ(calculate_warp_bitrate_factor(240, 60000), 4);
  EXPECT_EQ(calculate_warp_bitrate_factor(300, 60000), 1);
  EXPECT_EQ(calculate_warp_bitrate_factor(60, 0), 1);

  constexpr auto max_safe_kbps = std::numeric_limits<int>::max() / 1000;
  EXPECT_TRUE(is_safe_encoder_bitrate(max_safe_kbps));
  EXPECT_FALSE(is_safe_encoder_bitrate(max_safe_kbps + 1LL));
  EXPECT_FALSE(is_safe_encoder_bitrate(0));
}

TEST(RtspAnnounceParsingTest, ReservesFecAsParityRelativeToVideoData) {
  using rtsp_stream::detail::calculate_video_bitrate_budget;
  using rtsp_stream::detail::reserve_video_bitrate_for_fec;

  EXPECT_EQ(reserve_video_bitrate_for_fec(100'000, 0), 100'000);
  EXPECT_EQ(reserve_video_bitrate_for_fec(100'000, 10), 90'909);
  EXPECT_EQ(reserve_video_bitrate_for_fec(100'000, 20), 83'333);
  EXPECT_EQ(reserve_video_bitrate_for_fec(100'000, 255), 28'169);
  EXPECT_EQ(reserve_video_bitrate_for_fec(0, 20), 0);

  // 131000 - 512 Kbps audio - 500 Kbps control/packet allowance = 129988 Kbps
  // of video wire budget, then 20% parity relative to data => floor(129988 / 1.2).
  EXPECT_EQ(calculate_video_bitrate_budget(131'000, 20, 512), 108'323);
  // Low budgets retain the established 20% audio and 10% fixed-overhead caps.
  EXPECT_EQ(calculate_video_bitrate_budget(1'000, 20, 10'000), 600);
}

TEST(RtspAnnounceParsingTest, AcceptsNegotiatedPacketSizeAndRejectsSilentClamp) {
  using rtsp_stream::detail::announced_packet_size_allowed;

  EXPECT_TRUE(announced_packet_size_allowed(1392, 0));
  EXPECT_TRUE(announced_packet_size_allowed(1392, 1456));
  EXPECT_FALSE(announced_packet_size_allowed(1392, 1346));
  EXPECT_TRUE(announced_packet_size_allowed(1346, 1346));
  EXPECT_TRUE(announced_packet_size_allowed(200, 200));
  EXPECT_FALSE(announced_packet_size_allowed(1392, stream::VIDEO_PACKET_SIZE_MIN));
  EXPECT_FALSE(announced_packet_size_allowed(1392, stream::VIDEO_PACKET_SIZE_MIN - 1));
  EXPECT_FALSE(announced_packet_size_allowed(1392, stream::VIDEO_PACKET_SIZE_MAX + 1));
  EXPECT_FALSE(announced_packet_size_allowed(0, 0));
}

TEST(RtspSetupParsingTest, RejectsTargetsWithoutAStreamSelector) {
  using rtsp_stream::detail::parse_setup_stream_type;

  EXPECT_EQ(parse_setup_stream_type("stream=video/0"), std::optional<std::string_view> {"video"});
  EXPECT_EQ(parse_setup_stream_type("stream=audio"), std::optional<std::string_view> {"audio"});
  EXPECT_FALSE(parse_setup_stream_type("stream="));
  EXPECT_FALSE(parse_setup_stream_type("stream/video"));
  EXPECT_FALSE(parse_setup_stream_type(""));
}

TEST(RtspLaunchReservationTest, PreservesTheFirstPendingHandshake) {
  auto first = make_modern_launch_session(101, "first");
  auto second = make_modern_launch_session(102, "second");

  ASSERT_TRUE(rtsp_stream::launch_session_raise(first));
  EXPECT_FALSE(rtsp_stream::launch_session_raise(second));

  rtsp_stream::launch_session_clear(first->id);
  EXPECT_EQ(first->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
  EXPECT_FALSE(first->try_claim_reservation());
  EXPECT_TRUE(rtsp_stream::launch_session_raise(second));
  rtsp_stream::launch_session_clear(second->id);
}

TEST(RtspAnnounceParsingTest, RejectsUnsupportedCodecAndBitDepthCombinations) {
  using rtsp_stream::detail::is_video_mode_supported;

  EXPECT_TRUE(is_video_mode_supported(0, 0, true, true, true, true));
  EXPECT_FALSE(is_video_mode_supported(0, 1, true, true, true, true));

  EXPECT_FALSE(is_video_mode_supported(1, 0, false, false, true, true));
  EXPECT_TRUE(is_video_mode_supported(1, 0, true, false, true, true));
  EXPECT_FALSE(is_video_mode_supported(1, 1, true, false, true, true));
  EXPECT_TRUE(is_video_mode_supported(1, 1, true, true, true, true));

  EXPECT_FALSE(is_video_mode_supported(2, 0, true, true, false, false));
  EXPECT_TRUE(is_video_mode_supported(2, 0, true, true, true, false));
  EXPECT_FALSE(is_video_mode_supported(2, 1, true, true, true, false));
  EXPECT_TRUE(is_video_mode_supported(2, 1, true, true, true, true));
  EXPECT_FALSE(is_video_mode_supported(3, 0, true, true, true, true));
}

TEST(RtspLaunchReservationTest, RejectsLegacyRtspAndPingIdentity) {
  auto plaintext = std::make_shared<rtsp_stream::launch_session_t>();
  plaintext->id = 103;
  plaintext->av_ping_payload = "0123456789abcdef";
  EXPECT_FALSE(rtsp_stream::launch_session_raise(plaintext));

  auto malformed_ping = make_modern_launch_session(104, "malformed-ping");
  malformed_ping->av_ping_payload = "PING";
  EXPECT_FALSE(rtsp_stream::launch_session_raise(malformed_ping));
  EXPECT_TRUE(rtsp_stream::launch_session_available());
}

TEST(RtspLaunchReservationTest, ExplicitTeardownClearsPendingHandshake) {
  auto first = make_modern_launch_session(201, "first");
  auto replacement = make_modern_launch_session(202, "replacement");

  ASSERT_TRUE(first->reserve_live_gpu());
  ASSERT_TRUE(rtsp_stream::launch_session_raise(first));
  // Keep a reference just as an already accepted RTSP socket does. The production teardown
  // entry point must invalidate that accepted reservation as well as emptying the pending queue.
  const auto accepted = first;
  ASSERT_TRUE(accepted);
  rtsp_stream::terminate_session();
  EXPECT_EQ(accepted->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
  EXPECT_FALSE(accepted->try_claim_reservation());
  EXPECT_FALSE(gpu_workload::live_stream_active());
  EXPECT_TRUE(gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs))
    << "A retained accepted socket must not retain a revoked pending launch's GPU lease";
  EXPECT_TRUE(rtsp_stream::launch_session_raise(replacement));
  rtsp_stream::launch_session_clear(replacement->id);
}

TEST(RtspLaunchReservationTest, TimeoutReleasesLeaseRetainedByAcceptedSocket) {
  auto launch = make_modern_launch_session(203, "expired");
  ASSERT_TRUE(launch->reserve_live_gpu());
  ASSERT_TRUE(rtsp_stream::launch_session_raise(launch));
  const auto accepted = launch;

  rtsp_stream::expire_launch_session_for_test(launch->id);
  EXPECT_EQ(accepted->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
  EXPECT_FALSE(gpu_workload::live_stream_active());
  EXPECT_TRUE(gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs))
    << "A timed-out reservation must release its lease even while a socket retains it";
  EXPECT_TRUE(rtsp_stream::launch_session_available());
}

TEST(RtspLaunchReservationTest, ReservationCanBeClaimedOnlyOnce) {
  auto launch = make_modern_launch_session(301, "single-claim");

  ASSERT_TRUE(rtsp_stream::launch_session_available());
  ASSERT_TRUE(rtsp_stream::launch_session_raise(launch));
  EXPECT_FALSE(rtsp_stream::launch_session_available());
  EXPECT_TRUE(rtsp_stream::claim_launch_session_for_test(*launch));
  EXPECT_FALSE(rtsp_stream::claim_launch_session_for_test(*launch));
  EXPECT_EQ(launch->reservation(), rtsp_stream::launch_reservation_state_e::claimed);

  // This is the real control ordering: the control connection clears the pending record while
  // ANNOUNCE startup still owns the claimed reservation. Clearing must not revoke that startup,
  // and the claimed slot must remain unavailable until startup publishes its final result.
  rtsp_stream::launch_session_clear(launch->id);
  EXPECT_EQ(launch->reservation(), rtsp_stream::launch_reservation_state_e::claimed);
  EXPECT_FALSE(rtsp_stream::launch_session_available());

  rtsp_stream::finish_launch_session_for_test(*launch, true);
  EXPECT_EQ(launch->reservation(), rtsp_stream::launch_reservation_state_e::claimed);
  EXPECT_TRUE(rtsp_stream::launch_session_available());
}

TEST(RtspLaunchReservationTest, TeardownRevokesClaimedStartupBeforeReplacement) {
  auto launch = make_modern_launch_session(401, "claimed-before-teardown");

  ASSERT_TRUE(launch->reserve_live_gpu());
  ASSERT_TRUE(rtsp_stream::launch_session_raise(launch));
  ASSERT_TRUE(rtsp_stream::claim_launch_session_for_test(*launch));
  rtsp_stream::terminate_session();
  EXPECT_EQ(launch->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
  EXPECT_FALSE(rtsp_stream::launch_session_available());
  EXPECT_FALSE(gpu_workload::live_stream_active());

  // The startup path observes revocation, rolls back, and releases the claimed slot.
  rtsp_stream::finish_launch_session_for_test(*launch, false);
  EXPECT_TRUE(rtsp_stream::launch_session_available());
}

TEST(RtspLaunchReservationTest, TransferredClaimedLeaseSurvivesConcurrentRevocation) {
  auto launch = make_modern_launch_session(402, "claimed-and-transferred");

  ASSERT_TRUE(launch->reserve_live_gpu());
  ASSERT_TRUE(rtsp_stream::launch_session_raise(launch));
  ASSERT_TRUE(rtsp_stream::claim_launch_session_for_test(*launch));
  auto startup_lease = rtsp_stream::take_claimed_live_gpu_lease_for_test(*launch);
  ASSERT_TRUE(startup_lease);

  rtsp_stream::terminate_session();
  EXPECT_EQ(launch->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
  EXPECT_TRUE(gpu_workload::live_stream_active());
  EXPECT_FALSE(gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs))
    << "Revocation must not open a gap after ownership moved into startup";

  rtsp_stream::finish_launch_session_for_test(*launch, false);
  startup_lease.reset();
  EXPECT_FALSE(gpu_workload::live_stream_active());
  EXPECT_TRUE(rtsp_stream::launch_session_available());
}

TEST(RtspLaunchReservationTest, FailedClaimedStartupRevokesAndReleasesReservation) {
  auto failed = make_modern_launch_session(501, "failed-startup");

  ASSERT_TRUE(rtsp_stream::launch_session_raise(failed));
  ASSERT_TRUE(rtsp_stream::claim_launch_session_for_test(*failed));
  rtsp_stream::finish_launch_session_for_test(*failed, false);

  EXPECT_EQ(failed->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
  EXPECT_FALSE(failed->try_claim_reservation());
  EXPECT_TRUE(rtsp_stream::launch_session_available());
  EXPECT_FALSE(rtsp_stream::launch_session_raise(failed));
}

TEST(RtspMicrophoneLifecycleTest, PendingTeardownClosesReceiverAndStaleLaunchCannotStopReplacement) {
  auto cleanup = util::fail_guard([] {
    rtsp_stream::terminate_session();
  });
  auto first = make_modern_launch_session(601, "microphone-first");
  auto destroyed = std::make_shared<std::atomic<unsigned>>(0);
  first->microphone_receiver = make_lifecycle_microphone(*first, destroyed);
  ASSERT_TRUE(first->microphone_receiver);
  const auto retained_receiver = first->microphone_receiver;
  const auto retained_socket_launch = first;
  const auto reusable_port = retained_receiver->bound_port();
  ASSERT_TRUE(rtsp_stream::launch_session_raise(first));

  rtsp_stream::terminate_session();
  EXPECT_FALSE(retained_receiver->running());
  EXPECT_FALSE(retained_socket_launch->microphone_receiver);
  EXPECT_EQ(destroyed->load(), 1u);

  auto replacement = make_modern_launch_session(602, "microphone-replacement");
  replacement->microphone_receiver = make_lifecycle_microphone(*replacement, destroyed, reusable_port);
  ASSERT_TRUE(replacement->microphone_receiver) << "Teardown must release the bound UDP port";
  ASSERT_TRUE(rtsp_stream::launch_session_raise(replacement));
  retained_socket_launch->revoke_reservation();
  EXPECT_TRUE(replacement->microphone_receiver->running());
  EXPECT_EQ(destroyed->load(), 1u);
  rtsp_stream::launch_session_clear(replacement->id);
  EXPECT_EQ(destroyed->load(), 2u);
}

TEST(RtspMicrophoneLifecycleTest, PendingTimeoutStopsReceiverDespiteRetainedSocketAndReceiverReferences) {
  auto cleanup = util::fail_guard([] {
    rtsp_stream::terminate_session();
  });
  auto launch = make_modern_launch_session(603, "microphone-timeout");
  auto destroyed = std::make_shared<std::atomic<unsigned>>(0);
  launch->microphone_receiver = make_lifecycle_microphone(*launch, destroyed);
  ASSERT_TRUE(launch->microphone_receiver);
  const auto retained_receiver = launch->microphone_receiver;
  ASSERT_TRUE(rtsp_stream::launch_session_raise(launch));
  rtsp_stream::expire_launch_session_for_test(launch->id);
  EXPECT_FALSE(retained_receiver->running());
  EXPECT_EQ(destroyed->load(), 1u);
  EXPECT_EQ(launch->reservation(), rtsp_stream::launch_reservation_state_e::revoked);
  EXPECT_TRUE(rtsp_stream::launch_session_available());
}

TEST(RtspMicrophoneLifecycleTest, TransferPreservesReceiverAndActivePermissionRevocationStopsOnlyMicrophone) {
  auto launch = make_modern_launch_session(604, "microphone-policy");
  launch->iv.resize(16);
  launch->perm = crypto::PERM::_all;
  auto destroyed = std::make_shared<std::atomic<unsigned>>(0);
  launch->microphone_receiver = make_lifecycle_microphone(*launch, destroyed);
  ASSERT_TRUE(launch->microphone_receiver);
  const auto retained_receiver = launch->microphone_receiver;
  stream::config_t config {};
  auto active = stream::session::alloc(config, *launch);
  ASSERT_TRUE(active);
  EXPECT_FALSE(launch->microphone_receiver);
  retained_receiver->activate();
  stream::session::set_state_for_test(*active, stream::session::state_e::RUNNING);
  launch->revoke_reservation();
  EXPECT_TRUE(retained_receiver->running()) << "The active stream owns the transferred receiver";

  const auto without_microphone = static_cast<crypto::PERM>(
    static_cast<std::uint32_t>(crypto::PERM::_all) & ~static_cast<std::uint32_t>(crypto::PERM::input_microphone)
  );
  EXPECT_EQ(stream::session::update_client_policy(*active, 1, "microphone-policy", without_microphone, false), stream::session::client_policy_result_e::updated);
  EXPECT_FALSE(retained_receiver->running());
  EXPECT_EQ(destroyed->load(), 1u);
  EXPECT_EQ(stream::session::state(*active), stream::session::state_e::RUNNING);
  EXPECT_EQ(stream::session::permissions(*active), without_microphone);
  EXPECT_EQ(stream::session::update_client_policy(*active, 0, "stale-policy", crypto::PERM::_all, false), stream::session::client_policy_result_e::ignored);
  EXPECT_FALSE(retained_receiver->running());
}
