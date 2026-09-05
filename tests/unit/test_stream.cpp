/**
 * @file tests/unit/test_stream.cpp
 * @brief Test src/stream.*
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <src/process.h>
#include <src/stream.h>
#include <src/utility.h>
#include <string>
#include <vector>

namespace stream {
  std::vector<uint8_t> concat_and_insert(uint64_t insert_size, uint64_t slice_size, const std::string_view &data1, const std::string_view &data2);
}

#include "../tests_common.h"

using namespace std::chrono_literals;

static_assert(stream::CONTROL_OUTGOING_MAX_WAIT <= 10ms);

TEST(SbsDebugDumpRequestTest, RequiresDiagnosticsAndRuntimeHostSbsOwnership) {
  EXPECT_TRUE(stream::sbs_debug_dump_request_allowed(true, video::SBS_AI, true));
  EXPECT_FALSE(stream::sbs_debug_dump_request_allowed(false, video::SBS_AI, true));
  EXPECT_FALSE(stream::sbs_debug_dump_request_allowed(true, video::SBS_OFF, true));
  EXPECT_FALSE(stream::sbs_debug_dump_request_allowed(true, video::SBS_AI, false));
}

TEST(PlatformLaunchGuardTest, SerializesConcurrentLaunchPreparation) {
  std::future<bool> second;
  {
    auto first = stream::session::guard_platform_launch();
    EXPECT_TRUE(first.idle());

    second = std::async(std::launch::async, []() {
      auto guard = stream::session::guard_platform_launch();
      return guard.idle();
    });

    EXPECT_EQ(second.wait_for(20ms), std::future_status::timeout);
  }
  EXPECT_EQ(second.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(second.get());
}

TEST(PlatformLaunchGuardTest, CommitReleasesLaunchPreparationLock) {
  auto first = stream::session::guard_platform_launch();
  EXPECT_TRUE(first.idle());
  first.commit();

  auto second = std::async(std::launch::async, []() {
    auto guard = stream::session::guard_platform_launch();
    return guard.idle();
  });

  EXPECT_EQ(second.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(second.get());

  // Accepted HTTP paths may defensively commit during cleanup; this must remain harmless.
  first.commit();
}

TEST(PlatformLaunchGuardTest, ActiveSlotRejectsSecondSessionAndMakesHostNonIdle) {
  ASSERT_TRUE(stream::session::claim_active_slot_for_test());
  auto cleanup = util::fail_guard([]() {
    stream::session::release_active_slot_for_test();
  });

  EXPECT_FALSE(stream::session::claim_active_slot_for_test());

  auto guard = stream::session::guard_platform_launch();
  EXPECT_FALSE(guard.idle());
}

TEST(SessionWorkerStartTest, RollsBackWhenSecondThreadCannotStart) {
  EXPECT_TRUE(stream::session::worker_start_rollback_for_test());
}

TEST(ConcatAndInsertTests, ConcatNoInsertionTest) {
  char b1[] = {'a', 'b'};
  char b2[] = {'c', 'd', 'e'};
  auto res = stream::concat_and_insert(0, 2, std::string_view {b1, sizeof(b1)}, std::string_view {b2, sizeof(b2)});
  auto expected = std::vector<uint8_t> {'a', 'b', 'c', 'd', 'e'};
  ASSERT_EQ(res, expected);
}

TEST(ConcatAndInsertTests, ConcatLargeStrideTest) {
  char b1[] = {'a', 'b'};
  char b2[] = {'c', 'd', 'e'};
  auto res = stream::concat_and_insert(1, sizeof(b1) + sizeof(b2) + 1, std::string_view {b1, sizeof(b1)}, std::string_view {b2, sizeof(b2)});
  auto expected = std::vector<uint8_t> {0, 'a', 'b', 'c', 'd', 'e'};
  ASSERT_EQ(res, expected);
}

TEST(ConcatAndInsertTests, ConcatSmallStrideTest) {
  char b1[] = {'a', 'b'};
  char b2[] = {'c', 'd', 'e'};
  auto res = stream::concat_and_insert(1, 1, std::string_view {b1, sizeof(b1)}, std::string_view {b2, sizeof(b2)});
  auto expected = std::vector<uint8_t> {0, 'a', 0, 'b', 0, 'c', 0, 'd', 0, 'e'};
  ASSERT_EQ(res, expected);
}

TEST(ConcatAndInsertTests, ConcatAcrossBufferBoundaryAndUnevenFinalSlice) {
  char b1[] = {'a', 'b', 'c'};
  char b2[] = {'d', 'e', 'f', 'g'};
  auto res = stream::concat_and_insert(1, 4, std::string_view {b1, sizeof(b1)}, std::string_view {b2, sizeof(b2)});
  auto expected = std::vector<uint8_t> {0, 'a', 'b', 'c', 'd', 0, 'e', 'f', 'g'};
  ASSERT_EQ(res, expected);
}

TEST(ConcatAndInsertTests, RejectsZeroStrideAndSizeOverflow) {
  constexpr char data[] = {'a'};

  EXPECT_TRUE(stream::concat_and_insert(1, 0, std::string_view {data, sizeof(data)}, {}).empty());
  EXPECT_TRUE(
    stream::concat_and_insert(
      std::numeric_limits<std::uint64_t>::max(),
      1,
      std::string_view {data, sizeof(data)},
      {}
    )
      .empty()
  );
}

TEST(ConcatAndInsertTests, EmptyInputProducesEmptyOutput) {
  EXPECT_TRUE(stream::concat_and_insert(1, 1, {}, {}).empty());
}

TEST(VideoTransportConfigTests, EnforcesPacketAndFecWireBounds) {
  EXPECT_FALSE(stream::is_valid_video_packet_size(stream::VIDEO_PACKET_SIZE_MIN - 1));
  EXPECT_TRUE(stream::is_valid_video_packet_size(stream::VIDEO_PACKET_SIZE_MIN));
  EXPECT_TRUE(stream::is_valid_video_packet_size(1392));
  EXPECT_TRUE(stream::is_valid_video_packet_size(stream::VIDEO_PACKET_SIZE_MAX));
  EXPECT_FALSE(stream::is_valid_video_packet_size(stream::VIDEO_PACKET_SIZE_MAX + 1));

  EXPECT_FALSE(stream::is_valid_video_transport_config(stream::VIDEO_PACKET_SIZE_MIN - 1, 0));
  EXPECT_TRUE(stream::is_valid_video_transport_config(stream::VIDEO_PACKET_SIZE_MIN, 0));
  EXPECT_TRUE(stream::is_valid_video_transport_config(1392, stream::MIN_REQUIRED_FEC_PACKETS_MAX));
  EXPECT_TRUE(stream::is_valid_video_transport_config(stream::VIDEO_PACKET_SIZE_MAX, 1));
  EXPECT_FALSE(stream::is_valid_video_transport_config(stream::VIDEO_PACKET_SIZE_MAX + 1, 0));
  EXPECT_FALSE(stream::is_valid_video_transport_config(1392, -1));
  EXPECT_FALSE(stream::is_valid_video_transport_config(1392, stream::MIN_REQUIRED_FEC_PACKETS_MAX + 1));
}

TEST(ControlFeedbackPolicyTests, OutgoingQueuesArePolledWithinTenMilliseconds) {
  EXPECT_GT(stream::CONTROL_OUTGOING_MAX_WAIT, 0ms);
  EXPECT_LE(stream::CONTROL_OUTGOING_MAX_WAIT, 10ms);
}

TEST(VideoTransportConfigTests, EnforcesTenBitFecPacketIndex) {
  constexpr std::size_t block_size = 1408;
  constexpr std::size_t largest_valid_payload = stream::FEC_PACKET_INDEX_MAX * block_size;

  EXPECT_EQ(stream::fec_packet_count(largest_valid_payload, block_size), stream::FEC_PACKET_INDEX_MAX);
  EXPECT_TRUE(stream::is_valid_fec_block_size(largest_valid_payload, block_size));
  EXPECT_EQ(stream::fec_packet_count(largest_valid_payload + 1, block_size), stream::FEC_PACKET_INDEX_MAX + 1);
  EXPECT_FALSE(stream::is_valid_fec_block_size(largest_valid_payload + 1, block_size));
  EXPECT_FALSE(stream::is_valid_fec_block_size(1, 0));
}

TEST(VideoTransportConfigTests, EstimatesFecShardsIncludingMinimumParity) {
  constexpr std::size_t block_size = 1000;

  EXPECT_EQ(stream::video_fec_shard_count(10'000, block_size, 20, 0), 12);
  EXPECT_EQ(stream::video_fec_shard_count(1, block_size, 1, 2), 3);
  EXPECT_EQ(stream::video_fec_shard_count(1, block_size, 0, 2), 1);
  EXPECT_EQ(stream::video_fec_shard_count(0, block_size, 20, 2), 0);
}

TEST(VideoPacingTests, UsesBitrateFecCadenceAndBoundedBatches) {
  constexpr std::size_t payload_packet_bytes = 1376;
  constexpr std::size_t wire_packet_bytes = 1440;
  constexpr std::size_t estimated_data_packets = 104;
  constexpr std::size_t estimated_wire_packets = 125;

  const auto plan = stream::make_video_pacing_plan(
    104'108,
    90'000,
    estimated_data_packets,
    estimated_wire_packets,
    payload_packet_bytes,
    wire_packet_bytes
  );

  EXPECT_GT(plan.target_wire_bps, 104'108'000);
  EXPECT_LE(plan.target_wire_bps, stream::VIDEO_PACING_MAX_WIRE_BPS);
  EXPECT_GT(plan.packets_per_quantum, 1);
  EXPECT_LE(
    stream::video_pacing_offset(estimated_wire_packets, plan.packets_per_second).count(),
    plan.max_frame_span_ns
  );
  EXPECT_EQ(plan.max_frame_span_ns, plan.frame_interval_ns / 2);
  EXPECT_LE(plan.max_frame_span_ns, 5'555'556);
}

TEST(VideoPacingTests, NinetyFpsFrameSpanIsAtMostHalfAnInterval) {
  const auto plan = stream::make_video_pacing_plan(
    200'000,
    90'000,
    200,
    240,
    1376,
    1440
  );

  EXPECT_EQ(plan.frame_interval_ns, 11'111'111);
  EXPECT_EQ(plan.max_frame_span_ns, 5'555'555);
  EXPECT_LE(plan.max_frame_span_ns, 5'555'556);
}

TEST(VideoPacketizationWorkspaceTests, ReusesCapacityAndClearsOnlyInsertedPrefixes) {
  std::vector<std::uint8_t> workspace(128, 0xA5);
  const auto original_capacity = workspace.capacity();
  const std::array<std::uint8_t, 3> first {1, 2, 3};
  const std::array<std::uint8_t, 2> second {4, 5};

  const auto size = stream::concat_and_insert_into(
    workspace,
    2,
    3,
    {reinterpret_cast<const char *>(first.data()), first.size()},
    {reinterpret_cast<const char *>(second.data()), second.size()}
  );

  ASSERT_EQ(size, 9u);
  EXPECT_EQ(workspace.capacity(), original_capacity);
  const std::array<std::uint8_t, 9> expected {0, 0, 1, 2, 3, 0, 0, 4, 5};
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), workspace.begin()));
  EXPECT_EQ(workspace[9], 0xA5);
}

TEST(VideoPacingTests, LowBitrateThirtyFpsStreamRemainsNegotiatedRateAware) {
  const auto low_rate = stream::make_video_pacing_plan(
    10'000,
    30'000,
    30,
    36,
    1376,
    1440
  );
  EXPECT_GT(low_rate.target_wire_bps, 10'000'000);
  EXPECT_LT(low_rate.target_wire_bps, 30'000'000);
  EXPECT_EQ(low_rate.target_wire_bps, 24'883'201);
  EXPECT_LE(
    stream::video_pacing_offset(36, low_rate.packets_per_second).count(),
    low_rate.max_frame_span_ns
  );
}

TEST(VideoPacingTests, UsesFallbackOnlyForInvalidBitrateAndCapsPathologicalFrames) {
  const auto invalid_bitrate = stream::make_video_pacing_plan(
    0,
    60'000,
    60,
    72,
    1376,
    1440
  );
  EXPECT_GT(invalid_bitrate.target_wire_bps, stream::VIDEO_PACING_FALLBACK_ENCODED_BPS);
  EXPECT_LT(invalid_bitrate.target_wire_bps, stream::VIDEO_PACING_MAX_WIRE_BPS);

  const auto oversized_frame = stream::make_video_pacing_plan(
    1'000'000,
    120'000,
    3410,
    4092,
    1376,
    1440
  );
  EXPECT_EQ(oversized_frame.target_wire_bps, stream::VIDEO_PACING_MAX_WIRE_BPS);
}

TEST(VideoPacingTests, BoundsLateScheduleCatchupToOneQuantum) {
  EXPECT_EQ(stream::video_pacing_rebase_ns(5'000'000, 5'500'000), 0);
  EXPECT_EQ(stream::video_pacing_rebase_ns(5'000'000, 6'000'000), 0);
  EXPECT_EQ(stream::video_pacing_rebase_ns(5'000'000, 25'000'000), 19'000'000);
  EXPECT_EQ(
    5'000'000 + stream::video_pacing_rebase_ns(5'000'000, 25'000'000),
    25'000'000 - stream::VIDEO_PACING_MAX_CATCHUP_NS
  );
}

TEST(VideoPacingTests, RetainsFractionalCadenceAndBoundsQueueAge) {
  EXPECT_EQ(stream::video_frame_interval_ns(60'000), 16'666'666);
  EXPECT_EQ(stream::video_frame_interval_ns(59'940), 16'683'350);
  EXPECT_GT(stream::video_frame_interval_ns(59'940), stream::video_frame_interval_ns(60'000));

  EXPECT_EQ(stream::video_packet_max_queue_age_ns(90'000), 50'000'000);
  EXPECT_EQ(stream::video_packet_max_queue_age_ns(30'000), 99'999'999);
}

TEST(CheckedIntegerParsingTests, RejectsPartialAndOverflowingValues) {
  EXPECT_EQ(util::from_view_checked<int>("1392"), 1392);
  EXPECT_EQ(util::from_view_checked<int>("-1"), -1);
  EXPECT_FALSE(util::from_view_checked<int>(""));
  EXPECT_FALSE(util::from_view_checked<int>("12x"));
  EXPECT_FALSE(util::from_view_checked<int>(" 12"));
  EXPECT_FALSE(util::from_view_checked<int>("+12"));
  EXPECT_FALSE(util::from_view_checked<int>("2147483648"));
  EXPECT_FALSE(util::from_view_checked<int>("999999999999999999999999999999999999"));
}

TEST(ControlPayloadValidationTests, EnforcesOuterPacketBounds) {
  EXPECT_FALSE(stream::is_valid_control_packet_size(sizeof(std::uint16_t) - 1));
  EXPECT_TRUE(stream::is_valid_control_packet_size(sizeof(std::uint16_t)));
  EXPECT_TRUE(stream::is_valid_control_packet_size(stream::CONTROL_PACKET_SIZE_MAX));
  EXPECT_FALSE(stream::is_valid_control_packet_size(stream::CONTROL_PACKET_SIZE_MAX + 1));
}

TEST(ControlPayloadValidationTests, EnforcesEncryptedEnvelopeLength) {
  constexpr auto minimum_length = stream::CONTROL_ENCRYPTED_MIN_LENGTH;

  EXPECT_FALSE(stream::is_valid_encrypted_control_payload(minimum_length + 1, minimum_length - 1));
  EXPECT_TRUE(stream::is_valid_encrypted_control_payload(stream::CONTROL_ENCRYPTED_LENGTH_FIELD_SIZE + minimum_length, minimum_length));
  EXPECT_FALSE(stream::is_valid_encrypted_control_payload(stream::CONTROL_ENCRYPTED_LENGTH_FIELD_SIZE + minimum_length - 1, minimum_length));
  EXPECT_FALSE(stream::is_valid_encrypted_control_payload(stream::CONTROL_ENCRYPTED_LENGTH_FIELD_SIZE + minimum_length + 1, minimum_length));
  EXPECT_TRUE(stream::is_valid_encrypted_control_payload(stream::CONTROL_ENCRYPTED_LENGTH_FIELD_SIZE + std::numeric_limits<std::uint16_t>::max(), std::numeric_limits<std::uint16_t>::max()));
}

TEST(ControlPayloadValidationTests, MinimumEnvelopeAlwaysContainsTheSplitHeader) {
  EXPECT_EQ(
    stream::CONTROL_ENCRYPTED_MIN_LENGTH - stream::CONTROL_ENCRYPTED_SEQUENCE_SIZE,
    stream::CONTROL_GCM_TAG_SIZE + stream::CONTROL_HEADER_V2_SIZE
  );
  EXPECT_FALSE(stream::is_valid_encrypted_control_payload(
    stream::CONTROL_ENCRYPTED_LENGTH_FIELD_SIZE +
      stream::CONTROL_ENCRYPTED_MIN_LENGTH - 1,
    stream::CONTROL_ENCRYPTED_MIN_LENGTH - 1
  ));
}

TEST(ControlPayloadValidationTests, EnforcesDecryptedInnerLength) {
  EXPECT_TRUE(stream::is_valid_decrypted_control_payload(stream::CONTROL_HEADER_V2_SIZE, 0));
  EXPECT_TRUE(stream::is_valid_decrypted_control_payload(stream::CONTROL_HEADER_V2_SIZE + 17, 17));
  EXPECT_FALSE(stream::is_valid_decrypted_control_payload(stream::CONTROL_HEADER_V2_SIZE - 1, 0));
  EXPECT_FALSE(stream::is_valid_decrypted_control_payload(stream::CONTROL_HEADER_V2_SIZE + 17, 16));
  EXPECT_FALSE(stream::is_valid_decrypted_control_payload(stream::CONTROL_HEADER_V2_SIZE + 17, 18));
}

TEST(ControlPayloadValidationTests, RecognizesOnlyTheFixedFrameFecStatusBody) {
  EXPECT_EQ(stream::FRAME_FEC_STATUS_PAYLOAD_SIZE, 21U);
  EXPECT_TRUE(stream::is_valid_frame_fec_status_payload_size(21));
  // Five bytes is the opposite-direction RGB LED body that shares packet type 0x5502.
  EXPECT_FALSE(stream::is_valid_frame_fec_status_payload_size(5));
  EXPECT_FALSE(stream::is_valid_frame_fec_status_payload_size(20));
  EXPECT_FALSE(stream::is_valid_frame_fec_status_payload_size(22));
}

TEST(ControlPayloadValidationTests, EnforcesLiveVideoModeGeometryBounds) {
  EXPECT_TRUE(stream::is_valid_live_video_mode_dimension(1920));
  EXPECT_TRUE(stream::is_valid_live_video_mode_dimension(stream::LIVE_VIDEO_MODE_DIMENSION_MIN));
  EXPECT_TRUE(stream::is_valid_live_video_mode_dimension(stream::LIVE_VIDEO_MODE_DIMENSION_MAX));

  // 4:2:0 subsampling runs through the whole encode path, so odd dimensions are never encodable.
  EXPECT_FALSE(stream::is_valid_live_video_mode_dimension(1921));
  EXPECT_FALSE(stream::is_valid_live_video_mode_dimension(0));
  EXPECT_FALSE(stream::is_valid_live_video_mode_dimension(-1920));
  EXPECT_FALSE(stream::is_valid_live_video_mode_dimension(stream::LIVE_VIDEO_MODE_DIMENSION_MAX + 2));

  // Widths beyond any codec's capability stay valid on the wire: the encode loop caps them, and
  // capping cannot fail where rejecting a creatable mode would deny a deliverable one.
  EXPECT_TRUE(stream::is_valid_live_video_mode_dimension(10240));
}

TEST(ControlPayloadValidationTests, EnforcesLiveVideoModeFrameRateBounds) {
  EXPECT_TRUE(stream::is_valid_live_video_mode_framerate_x100(6000));
  EXPECT_TRUE(stream::is_valid_live_video_mode_framerate_x100(2397));
  EXPECT_TRUE(stream::is_valid_live_video_mode_framerate_x100(2997));
  EXPECT_TRUE(stream::is_valid_live_video_mode_framerate_x100(stream::LIVE_VIDEO_MODE_FRAMERATE_X100_MIN));
  EXPECT_TRUE(stream::is_valid_live_video_mode_framerate_x100(stream::LIVE_VIDEO_MODE_FRAMERATE_X100_MAX));

  EXPECT_FALSE(stream::is_valid_live_video_mode_framerate_x100(0));
  EXPECT_FALSE(stream::is_valid_live_video_mode_framerate_x100(99));
  EXPECT_FALSE(stream::is_valid_live_video_mode_framerate_x100(-6000));
  EXPECT_FALSE(stream::is_valid_live_video_mode_framerate_x100(stream::LIVE_VIDEO_MODE_FRAMERATE_X100_MAX + 1));
}

TEST(AtomicPresentationWireTests, DecodesExactV2GoldenVector) {
  EXPECT_EQ(stream::ATOMIC_PRESENTATION_V2_REQUEST_PAYLOAD_SIZE, 20U);
  EXPECT_EQ(stream::ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE, 28U);
  EXPECT_EQ(stream::ATOMIC_PRESENTATION_VERSION, 2U);

  const std::array<std::uint8_t, 20> v2 {
    0x02, 0x01, 0x00, 0x00,  // version, AI, flags=0
    0xEF, 0xCD, 0xAB, 0x89,  // u32 request id
    0x00, 0x14,  // source width 5120
    0x70, 0x08,  // source height 2160
    0x6A, 0x17, 0x00, 0x00,  // 59.94 Hz
    0x50, 0xC3, 0x00, 0x00,  // 50000 kbps
  };
  stream::live_video_mode_wire_request_t request;
  const auto decoded = stream::decode_live_video_mode_request_payload(
    {reinterpret_cast<const char *>(v2.data()), v2.size()},
    request
  );
  ASSERT_EQ(decoded, stream::live_video_mode_request_decode_e::v2);
  EXPECT_EQ(request.protocol_version, stream::ATOMIC_PRESENTATION_VERSION);
  EXPECT_EQ(request.desired_sbs_mode, video::SBS_AI);
  EXPECT_EQ(request.flags, 0);
  EXPECT_EQ(request.request_id, 0x89ABCDEFu);
  EXPECT_EQ(request.source_width, 5120);
  EXPECT_EQ(request.source_height, 2160);
  EXPECT_EQ(request.framerate_x100, 5994u);
  EXPECT_EQ(request.bitrate_kbps, 50000u);
}

TEST(AtomicPresentationWireTests, RejectsEveryNonExactBodyAndUnsupportedVersion) {
  // The retired v1 body was 12 bytes; it is deliberately just another invalid size now.
  for (const std::size_t size : {0U, 12U, 19U, 21U}) {
    std::array<char, 21> bytes {};
    stream::live_video_mode_wire_request_t request;
    EXPECT_EQ(
      stream::decode_live_video_mode_request_payload({bytes.data(), size}, request),
      stream::live_video_mode_request_decode_e::invalid
    ) << "size=" << size;
  }

  std::array<char, stream::ATOMIC_PRESENTATION_V2_REQUEST_PAYLOAD_SIZE> future {};
  future[0] = 3;
  stream::live_video_mode_wire_request_t request;
  EXPECT_EQ(
    stream::decode_live_video_mode_request_payload(
      {future.data(), future.size()},
      request
    ),
    stream::live_video_mode_request_decode_e::unsupported_version
  );
  EXPECT_EQ(request.protocol_version, 3);
}

TEST(AtomicPresentationNegotiationTests, PublishesMatchingHostAndClientFeatureBits) {
  EXPECT_EQ(stream::CLIENT_FEATURE_ATOMIC_PRESENTATION_V2, 0x08u);
  EXPECT_EQ(platf::platform_caps::atomic_presentation_v2, 0x20000000u);
#ifdef _WIN32
  EXPECT_NE(
    platf::get_capabilities() & platf::platform_caps::atomic_presentation_v2,
    0u
  );
#endif
}

TEST(LiveVideoModeSerialGateTests, CoalescesOnlyPendingRequestsAndWaitsThroughCompletion) {
  stream::detail::live_video_mode_serial_gate_t gate;

  const auto first = gate.submit();
  ASSERT_NE(first.transaction_id, 0U);
  EXPECT_FALSE(first.superseded_transaction_id);

  // The first request has not begun, so a newer request can replace it without performing either
  // display or encoder work.
  const auto second = gate.submit();
  EXPECT_EQ(second.superseded_transaction_id, first.transaction_id);
  ASSERT_TRUE(gate.can_begin());
  ASSERT_EQ(gate.begin_next(), second.transaction_id);
  EXPECT_EQ(gate.active(), second.transaction_id);

  // Once B begins, C remains pending. A stale completion (including one correlated by a duplicate
  // client request id in production) cannot release B or let C overtake it.
  const auto third = gate.submit();
  EXPECT_FALSE(third.superseded_transaction_id);
  EXPECT_FALSE(gate.can_begin());
  EXPECT_FALSE(gate.finish(first.transaction_id));
  EXPECT_EQ(gate.active(), second.transaction_id);

  // Merely observing B's encoder completion is not enough: the worker retains the gate while it
  // restores a rejected desktop. `finish()` is called only after that rollback attempt ends.
  EXPECT_TRUE(gate.accepts_completion(second.transaction_id));
  EXPECT_FALSE(gate.can_begin());
  EXPECT_TRUE(gate.finish(second.transaction_id));
  ASSERT_TRUE(gate.can_begin());
  EXPECT_EQ(gate.begin_next(), third.transaction_id);
  EXPECT_TRUE(gate.finish(third.transaction_id));
  EXPECT_FALSE(gate.active());
}

TEST(LiveVideoModeSerialGateTests, CoalescesAQueuedBurstBehindAnActiveTransaction) {
  stream::detail::live_video_mode_serial_gate_t gate;

  const auto active = gate.submit();
  ASSERT_EQ(gate.begin_next(), active.transaction_id);
  const auto pending_a = gate.submit();
  const auto pending_b = gate.submit();
  const auto pending_c = gate.submit();

  EXPECT_FALSE(pending_a.superseded_transaction_id);
  EXPECT_EQ(pending_b.superseded_transaction_id, pending_a.transaction_id);
  EXPECT_EQ(pending_c.superseded_transaction_id, pending_b.transaction_id);
  EXPECT_FALSE(gate.can_begin());

  ASSERT_TRUE(gate.finish(active.transaction_id));
  ASSERT_EQ(gate.begin_next(), pending_c.transaction_id);
  EXPECT_TRUE(gate.finish(pending_c.transaction_id));
}

TEST(LiveVideoModeAckTests, MapsEveryDisplayOutcomeToItsWireStatus) {
  // A desktop that already presented the geometry still delivers the mode the client asked for,
  // so it must read as success and not as a reason to reconnect.
  EXPECT_EQ(
    stream::live_video_mode_ack_status(proc::live_video_mode_result_e::applied),
    stream::live_video_mode_ack_e::applied
  );
  EXPECT_EQ(
    stream::live_video_mode_ack_status(proc::live_video_mode_result_e::unchanged),
    stream::live_video_mode_ack_e::applied
  );
  EXPECT_EQ(
    stream::live_video_mode_ack_status(proc::live_video_mode_result_e::needs_reconnect),
    stream::live_video_mode_ack_e::rejected_needs_reconnect
  );
  EXPECT_EQ(
    stream::live_video_mode_ack_status(proc::live_video_mode_result_e::failed),
    stream::live_video_mode_ack_e::failed
  );

  // An outcome the mapping does not know must degrade to the retryable status, never to a
  // permanent refusal that would make a client give up on a deliverable mode.
  EXPECT_EQ(
    stream::live_video_mode_ack_status(static_cast<proc::live_video_mode_result_e>(99)),
    stream::live_video_mode_ack_e::failed
  );
}

TEST(LiveVideoModeAckTests, WireStatusValuesAreFrozen) {
  // These numbers are on the wire and are mirrored by the client's Limelight.h. Renumbering them
  // silently changes what an existing client believes happened.
  EXPECT_EQ(static_cast<std::uint16_t>(stream::live_video_mode_ack_e::applied), 0);
  EXPECT_EQ(static_cast<std::uint16_t>(stream::live_video_mode_ack_e::rejected_invalid), 1);
  EXPECT_EQ(static_cast<std::uint16_t>(stream::live_video_mode_ack_e::rejected_needs_reconnect), 2);
  EXPECT_EQ(static_cast<std::uint16_t>(stream::live_video_mode_ack_e::failed), 3);
}

TEST(AtomicPresentationAckTests, EncodesCompleteAppliedStateGoldenVector) {
  const video::effective_video_mode_t effective {
    4096,
    1728,
    5994,
    38000,
    5120,
    2160,
    8192,
    1728,
    video::SBS_AI,
    0x10203040u,
  };
  const auto ack = stream::make_live_video_mode_ack(
    0x89ABCDEFu,
    stream::live_video_mode_ack_e::applied,
    effective
  );
  std::uint8_t payload[stream::ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_atomic_presentation_ack_payload(ack, payload));

  const std::uint8_t expected[stream::ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE] = {
    0x02, 0x00, 0x01, 0x00,  // version, applied, AI, flags=0
    0xEF, 0xCD, 0xAB, 0x89,  // request id
    0x40, 0x30, 0x20, 0x10,  // applied generation
    0x00, 0x14, 0x70, 0x08,  // source 5120x2160
    0x00, 0x20, 0xC0, 0x06,  // exact encoded 8192x1728
    0x6A, 0x17, 0x00, 0x00,  // 59.94 Hz
    0x70, 0x94, 0x00, 0x00,  // encoder bitrate 38000 kbps
  };
  EXPECT_TRUE(std::equal(std::begin(payload), std::end(payload), std::begin(expected)));

  // The internal per-eye geometry remains available on the effective state, while the wire
  // publishes the unmodified source and exact packed encoder extents.
  EXPECT_EQ(effective.width, 4096);
  EXPECT_EQ(ack.applied_source_width, 5120);
  EXPECT_EQ(ack.applied_encoded_width, 8192);
}

TEST(AtomicPresentationAckTests, RequiresACompleteProvenGeneration) {
  const video::effective_video_mode_t unproven {
    1920, 1080, 6000, 20000, 1920, 1080, 1920, 1080, video::SBS_OFF, 0,
  };
  const auto ack = stream::make_live_video_mode_ack(
    std::numeric_limits<std::uint32_t>::max(),
    stream::live_video_mode_ack_e::failed,
    unproven
  );
  std::uint8_t payload[stream::ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE] {};
  EXPECT_FALSE(stream::encode_atomic_presentation_ack_payload(ack, payload));

  auto proven = unproven;
  proven.generation = 7;
  const auto refusal = stream::make_live_video_mode_ack(
    std::numeric_limits<std::uint32_t>::max(),
    stream::live_video_mode_ack_e::rejected_invalid,
    proven
  );
  ASSERT_TRUE(stream::encode_atomic_presentation_ack_payload(refusal, payload));
  EXPECT_EQ(payload[4], 0xFF);
  EXPECT_EQ(payload[5], 0xFF);
  EXPECT_EQ(payload[6], 0xFF);
  EXPECT_EQ(payload[7], 0xFF);
  EXPECT_EQ(payload[8], 7);  // unchanged current generation on rejection
}

TEST(AtomicPresentationAckTests, DefersOnlyPreProofRefusalsAndFailsClosedOnImpossibleApply) {
  stream::live_video_mode_ack_t ack {
    7,
    stream::live_video_mode_ack_e::failed,
    6000,
    20000,
    video::SBS_OFF,
    1920,
    1080,
    1920,
    1080,
    0,
  };
  EXPECT_EQ(
    stream::classify_live_video_mode_ack_delivery(ack),
    stream::live_video_mode_ack_delivery_e::defer_until_initial_proof
  );
  ack.status = stream::live_video_mode_ack_e::applied;
  EXPECT_EQ(
    stream::classify_live_video_mode_ack_delivery(ack),
    stream::live_video_mode_ack_delivery_e::fail_closed
  );
  ack.applied_generation = 1;
  EXPECT_EQ(
    stream::classify_live_video_mode_ack_delivery(ack),
    stream::live_video_mode_ack_delivery_e::send
  );
}

TEST(AtomicPresentationAckTests, ReportsAClampedApplyWithExactSourceAndEncodedGeometry) {
  // The whole point of the applied_* fields: a per-eye 5120x2160 SBS request packs to 10240, which
  // exceeds the 8192 codec ceiling, so the host installs 4096 per eye with an aspect-scaled height.
  // That is a successful apply, not a failure, and the client must be told the clamped geometry so
  // its confirmation check does not mistake the clamp for a refusal and revert.
  const auto packed = video::host_sbs_output_dimensions(5120, 2160, 2, 8192, 8192);
  ASSERT_EQ(packed.width, 8192);
  ASSERT_EQ(packed.height, 1728);

  const video::effective_video_mode_t effective {
    packed.width / 2,
    packed.height,
    6000,
    38000,
    5120,
    2160,
    packed.width,
    packed.height,
    video::SBS_AI,
    5,
  };
  const auto ack = stream::make_live_video_mode_ack(
    7,
    stream::live_video_mode_ack_e::applied,
    effective
  );

  std::uint8_t payload[stream::ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_atomic_presentation_ack_payload(ack, payload));
  EXPECT_EQ(payload[1], 0);  // applied
  EXPECT_EQ(payload[2], video::SBS_AI);
  EXPECT_EQ(static_cast<int>(payload[12]) | (static_cast<int>(payload[13]) << 8), 5120);
  EXPECT_EQ(static_cast<int>(payload[14]) | (static_cast<int>(payload[15]) << 8), 2160);
  EXPECT_EQ(static_cast<int>(payload[16]) | (static_cast<int>(payload[17]) << 8), 8192);
  EXPECT_EQ(static_cast<int>(payload[18]) | (static_cast<int>(payload[19]) << 8), 1728);
}

TEST(AtomicPresentationAckTests, RefusesValuesThatDoNotFitTheV2WireContract) {
  const video::effective_video_mode_t effective {
    1920, 1080, 6000, 20000, 1920, 1080, 1920, 1080, video::SBS_OFF, 1,
  };
  const auto valid = stream::make_live_video_mode_ack(
    1,
    stream::live_video_mode_ack_e::applied,
    effective
  );
  std::uint8_t payload[stream::ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE];
  const auto reject = [&](auto mutate) {
    auto ack = valid;
    mutate(ack);
    std::fill(std::begin(payload), std::end(payload), 0xA5);
    EXPECT_FALSE(stream::encode_atomic_presentation_ack_payload(ack, payload));
    EXPECT_TRUE(std::all_of(std::begin(payload), std::end(payload), [](auto byte) {
      return byte == 0xA5;
    }));
  };

  reject([](auto &ack) { ack.request_id = -1; });
  reject([](auto &ack) { ack.request_id = 4294967296LL; });
  reject([](auto &ack) { ack.status = static_cast<stream::live_video_mode_ack_e>(4); });
  reject([](auto &ack) { ack.applied_sbs_mode = 2; });
  reject([](auto &ack) { ack.applied_source_width = 65536; });
  reject([](auto &ack) { ack.applied_source_height = -1; });
  reject([](auto &ack) { ack.applied_encoded_width = 65536; });
  reject([](auto &ack) { ack.applied_encoded_height = -1; });
  reject([](auto &ack) { ack.applied_framerate_x100 = -1; });
  reject([](auto &ack) { ack.applied_bitrate_kbps = 4294967296LL; });
  reject([](auto &ack) { ack.applied_generation = 0; });
}

TEST(AtomicPresentationAckTests, RefusalsReportTheLastProvenMode) {
  // A refusal reports the last proven mode rather than zeros or the rejected request. Status 2
  // still makes reconnect authoritative because a failed desktop rollback can invalidate it.
  video::effective_video_mode_publisher_t publisher {{
    1920, 1080, 6000, 20000, 1920, 1080, 1920, 1080, video::SBS_OFF, 1,
  }};
  EXPECT_EQ(publisher.current().width, 1920);

  publisher.publish({
    2560, 1440, 12000, 45000, 2560, 1440, 2560, 1440, video::SBS_OFF, 2,
  });
  const auto in_effect = publisher.current();
  EXPECT_EQ(in_effect.width, 2560);
  EXPECT_EQ(in_effect.height, 1440);
  EXPECT_EQ(in_effect.framerateX100, 12000);
  EXPECT_EQ(in_effect.bitrate, 45000);

  const auto ack = stream::make_live_video_mode_ack(
    42,
    stream::live_video_mode_ack_e::rejected_needs_reconnect,
    in_effect
  );
  std::uint8_t payload[stream::ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_atomic_presentation_ack_payload(ack, payload));
  EXPECT_EQ(payload[1], 2);  // needs reconnect
  EXPECT_EQ(static_cast<int>(payload[4]) | (static_cast<int>(payload[5]) << 8), 42);
  EXPECT_EQ(static_cast<int>(payload[12]) | (static_cast<int>(payload[13]) << 8), 2560);
}

TEST(SbsTelemetryWireTests, VersionSizesFlagsAndStatusesAreFrozen) {
  EXPECT_EQ(stream::SBS_TELEMETRY_VERSION, 1);
  EXPECT_EQ(stream::SBS_TELEMETRY_SUBSCRIPTION_PAYLOAD_SIZE, 8);
  EXPECT_EQ(stream::SBS_TELEMETRY_STATE_PAYLOAD_SIZE, 88);
  EXPECT_EQ(stream::SBS_TELEMETRY_SUBSCRIBE_ENABLED, 1u << 0);
  EXPECT_EQ(stream::SBS_TELEMETRY_SUBSCRIBE_FOCUSED, 1u << 1);
  EXPECT_EQ(stream::CLIENT_FEATURE_SBS_TELEMETRY, 0x04u);

  // These are public protocol values, not the renderer's private sample-state enum.
  EXPECT_EQ(static_cast<std::uint8_t>(stream::sbs_telemetry_status_e::ok), 0);
  EXPECT_EQ(static_cast<std::uint8_t>(stream::sbs_telemetry_status_e::unavailable), 1);
  EXPECT_EQ(static_cast<std::uint8_t>(stream::sbs_telemetry_status_e::unsupported_version), 2);
  EXPECT_EQ(static_cast<std::uint8_t>(stream::sbs_telemetry_status_e::failed), 3);
}

TEST(SbsTelemetryWireTests, DecodesTheExactLittleEndianSubscriptionBody) {
  const std::array<std::uint8_t, stream::SBS_TELEMETRY_SUBSCRIPTION_PAYLOAD_SIZE> payload {
    stream::SBS_TELEMETRY_VERSION,
    stream::SBS_TELEMETRY_SUBSCRIBE_ENABLED | stream::SBS_TELEMETRY_SUBSCRIBE_FOCUSED,
    0xEF,
    0xBE,
    0xFA,
    0x00,
    0x00,
    0x00,
  };
  stream::sbs_telemetry_subscription_request_t request;
  const auto result = stream::decode_sbs_telemetry_subscription_payload(
    std::string_view {reinterpret_cast<const char *>(payload.data()), payload.size()},
    request
  );

  EXPECT_EQ(result, stream::sbs_telemetry_subscription_decode_e::ok);
  EXPECT_TRUE(request.enabled());
  EXPECT_TRUE(request.focused());
  EXPECT_EQ(request.request_id, 0xBEEF);
  EXPECT_EQ(request.interval_ms, 250);
}

TEST(SbsTelemetryWireTests, RejectsEverySubscriptionBodyThatIsNotExactlyEightBytes) {
  std::array<std::uint8_t, stream::SBS_TELEMETRY_SUBSCRIPTION_PAYLOAD_SIZE + 1> payload {};
  payload[0] = stream::SBS_TELEMETRY_VERSION;
  stream::sbs_telemetry_subscription_request_t request {
    stream::SBS_TELEMETRY_SUBSCRIBE_ENABLED,
    0xBEEF,
    250,
  };

  EXPECT_EQ(
    stream::decode_sbs_telemetry_subscription_payload(
      std::string_view {reinterpret_cast<const char *>(payload.data()), payload.size() - 2},
      request
    ),
    stream::sbs_telemetry_subscription_decode_e::invalid
  );
  EXPECT_EQ(request.flags, 0);
  EXPECT_EQ(request.request_id, 0);
  EXPECT_EQ(request.interval_ms, 0);

  EXPECT_EQ(
    stream::decode_sbs_telemetry_subscription_payload(
      std::string_view {reinterpret_cast<const char *>(payload.data()), payload.size()},
      request
    ),
    stream::sbs_telemetry_subscription_decode_e::invalid
  );
  EXPECT_EQ(request.flags, 0);
  EXPECT_EQ(request.request_id, 0);
  EXPECT_EQ(request.interval_ms, 0);
}

TEST(SbsTelemetryWireTests, ReportsUnsupportedVersionAndPreservesItsCorrelationId) {
  const std::array<std::uint8_t, stream::SBS_TELEMETRY_SUBSCRIPTION_PAYLOAD_SIZE> payload {
    static_cast<std::uint8_t>(stream::SBS_TELEMETRY_VERSION + 1),
    stream::SBS_TELEMETRY_SUBSCRIBE_ENABLED,
    0xEF,
    0xBE,
    0xFA,
    0x00,
    0x00,
    0x00,
  };
  stream::sbs_telemetry_subscription_request_t request;
  EXPECT_EQ(
    stream::decode_sbs_telemetry_subscription_payload(
      std::string_view {reinterpret_cast<const char *>(payload.data()), payload.size()},
      request
    ),
    stream::sbs_telemetry_subscription_decode_e::unsupported_version
  );
  EXPECT_EQ(request.request_id, 0xBEEF);
  EXPECT_EQ(request.interval_ms, 250);
}

TEST(SbsTelemetryWireTests, RejectsUnknownSubscriptionFlagsAndNonzeroReservedBytes) {
  std::array<std::uint8_t, stream::SBS_TELEMETRY_SUBSCRIPTION_PAYLOAD_SIZE> payload {
    stream::SBS_TELEMETRY_VERSION,
    0x80,
    0xEF,
    0xBE,
    0x00,
    0x00,
    0x00,
    0x00,
  };
  stream::sbs_telemetry_subscription_request_t request;
  const auto decode = [&](const auto &body) {
    return stream::decode_sbs_telemetry_subscription_payload(
      std::string_view {reinterpret_cast<const char *>(body.data()), body.size()},
      request
    );
  };

  EXPECT_EQ(decode(payload), stream::sbs_telemetry_subscription_decode_e::invalid);
  payload[1] = 0;
  payload[6] = 1;
  EXPECT_EQ(decode(payload), stream::sbs_telemetry_subscription_decode_e::invalid);
  payload[6] = 0;
  payload[7] = 1;
  EXPECT_EQ(decode(payload), stream::sbs_telemetry_subscription_decode_e::invalid);
}

TEST(SbsTelemetryWireTests, DefaultsAndClampsTheRequestedSamplingInterval) {
  stream::sbs_telemetry_subscription_request_t request;
  EXPECT_EQ(
    stream::effective_sbs_telemetry_interval_ms(request),
    stream::SBS_TELEMETRY_INTERVAL_BACKGROUND_MS
  );

  request.flags = stream::SBS_TELEMETRY_SUBSCRIBE_FOCUSED;
  EXPECT_EQ(
    stream::effective_sbs_telemetry_interval_ms(request),
    stream::SBS_TELEMETRY_INTERVAL_FOCUSED_MS
  );

  request.interval_ms = stream::SBS_TELEMETRY_INTERVAL_MIN_MS - 1;
  EXPECT_EQ(
    stream::effective_sbs_telemetry_interval_ms(request),
    stream::SBS_TELEMETRY_INTERVAL_MIN_MS
  );
  request.interval_ms = stream::SBS_TELEMETRY_INTERVAL_MIN_MS;
  EXPECT_EQ(
    stream::effective_sbs_telemetry_interval_ms(request),
    stream::SBS_TELEMETRY_INTERVAL_MIN_MS
  );
  request.interval_ms = 250;
  EXPECT_EQ(stream::effective_sbs_telemetry_interval_ms(request), 250);
  request.interval_ms = stream::SBS_TELEMETRY_INTERVAL_MAX_MS;
  EXPECT_EQ(
    stream::effective_sbs_telemetry_interval_ms(request),
    stream::SBS_TELEMETRY_INTERVAL_MAX_MS
  );
  request.interval_ms = stream::SBS_TELEMETRY_INTERVAL_MAX_MS + 1;
  EXPECT_EQ(
    stream::effective_sbs_telemetry_interval_ms(request),
    stream::SBS_TELEMETRY_INTERVAL_MAX_MS
  );
  request.interval_ms = std::numeric_limits<std::uint16_t>::max();
  EXPECT_EQ(
    stream::effective_sbs_telemetry_interval_ms(request),
    stream::SBS_TELEMETRY_INTERVAL_MAX_MS
  );
}

TEST(SbsTelemetryWireTests, ExplicitlyMapsRendererStatusToTheWireStatus) {
  EXPECT_EQ(
    stream::sbs_telemetry_wire_status(video::sbs_telemetry_sample_status_e::ready),
    stream::sbs_telemetry_status_e::ok
  );
  EXPECT_EQ(
    stream::sbs_telemetry_wire_status(video::sbs_telemetry_sample_status_e::unavailable),
    stream::sbs_telemetry_status_e::unavailable
  );
  EXPECT_EQ(
    stream::sbs_telemetry_wire_status(video::sbs_telemetry_sample_status_e::failed),
    stream::sbs_telemetry_status_e::failed
  );
  EXPECT_EQ(
    stream::sbs_telemetry_wire_status(
      static_cast<video::sbs_telemetry_sample_status_e>(0xFF)
    ),
    stream::sbs_telemetry_status_e::failed
  );
}

TEST(SbsTelemetryWireTests, PreRendererFallbackDoesNotFabricateConfigOrGeneration) {
  const auto snapshot = stream::unavailable_sbs_telemetry_snapshot();

  EXPECT_EQ(
    snapshot.status,
    video::sbs_telemetry_sample_status_e::unavailable
  );
  EXPECT_EQ(snapshot.generation, 0u);
  EXPECT_EQ(snapshot.sequence, 0u);
  EXPECT_EQ(snapshot.valid_fields, 0u);
  EXPECT_EQ(snapshot.runtime_flags, 0u);
  EXPECT_EQ(snapshot.depth_width, 0u);
  EXPECT_EQ(snapshot.depth_height, 0u);
  EXPECT_EQ(snapshot.zero_plane_mode, 0u);
  EXPECT_FLOAT_EQ(snapshot.pop_floor, 0.0f);
  EXPECT_FLOAT_EQ(snapshot.pop_ceiling, 0.0f);
  EXPECT_FLOAT_EQ(snapshot.effective_pop, 0.0f);
}

TEST(SbsTelemetryWireTests, EncodesEveryStateFieldAtItsFrozenLittleEndianOffset) {
  stream::sbs_telemetry_state_t state;
  state.status = stream::sbs_telemetry_status_e::unsupported_version;
  state.request_id = 0xBEEF;
  state.snapshot.generation = 0x11223344;
  state.snapshot.sequence = 0x55667788;
  state.snapshot.valid_fields = 0xA1B2C3D4;
  state.snapshot.runtime_flags = 0x10203040;
  state.snapshot.depth_width = 770;
  state.snapshot.depth_height = 434;
  state.snapshot.zero_plane_mode = 3;
  state.snapshot.pop_floor = 1.2f;
  state.snapshot.pop_ceiling = 2.0f;
  state.snapshot.effective_pop = 1.5f;
  state.snapshot.edge_fraction = 0.125f;
  state.snapshot.change_fraction = 0.25f;
  state.snapshot.zero_anchor_shift_px = -2.5f;
  state.snapshot.subject_depth = 0.5f;
  state.snapshot.valid_depth_fraction = 0.75f;
  state.snapshot.effective_range_width = 0.0625f;
  state.snapshot.scene_age = 0x01020304;
  state.snapshot.hard_cut_count = 0x11121314;
  state.snapshot.external_cut_count = 0x21222324;
  state.snapshot.empty_raw_count = 0x31323334;
  state.snapshot.collapsed_raw_count = 0x41424344;
  state.snapshot.sampled_frame_id = 0xDEADBEEF;

  const std::array<std::uint8_t, stream::SBS_TELEMETRY_STATE_PAYLOAD_SIZE> expected {
    0x01, 0x02, 0xEF, 0xBE,
    0x44, 0x33, 0x22, 0x11,
    0x88, 0x77, 0x66, 0x55,
    0xD4, 0xC3, 0xB2, 0xA1,
    0x40, 0x30, 0x20, 0x10,
    0x02, 0x03, 0xB2, 0x01,
    0x03, 0x00, 0x00, 0x00,
    0x9A, 0x99, 0x99, 0x3F,
    0x00, 0x00, 0x00, 0x40,
    0x00, 0x00, 0xC0, 0x3F,
    0x00, 0x00, 0x00, 0x3E,
    0x00, 0x00, 0x80, 0x3E,
    0x00, 0x00, 0x20, 0xC0,
    0x00, 0x00, 0x00, 0x3F,
    0x00, 0x00, 0x40, 0x3F,
    0x00, 0x00, 0x80, 0x3D,
    0x04, 0x03, 0x02, 0x01,
    0x14, 0x13, 0x12, 0x11,
    0x24, 0x23, 0x22, 0x21,
    0x34, 0x33, 0x32, 0x31,
    0x44, 0x43, 0x42, 0x41,
    0xEF, 0xBE, 0xAD, 0xDE,
  };
  std::uint8_t encoded[stream::SBS_TELEMETRY_STATE_PAYLOAD_SIZE] {};

  ASSERT_TRUE(stream::encode_sbs_telemetry_state_payload(state, encoded));
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), encoded));
}

TEST(SbsTelemetryWireTests, FailedEncodeLeavesTheCallerBufferUntouched) {
  stream::sbs_telemetry_state_t state;
  state.status = stream::sbs_telemetry_status_e::ok;
  std::uint8_t encoded[stream::SBS_TELEMETRY_STATE_PAYLOAD_SIZE];
  std::fill(std::begin(encoded), std::end(encoded), 0xA5);

  state.snapshot.edge_fraction = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(stream::encode_sbs_telemetry_state_payload(state, encoded));
  EXPECT_TRUE(std::all_of(std::begin(encoded), std::end(encoded), [](std::uint8_t byte) {
    return byte == 0xA5;
  }));

  state.snapshot.edge_fraction = 0.0f;
  state.status = static_cast<stream::sbs_telemetry_status_e>(0xFF);
  EXPECT_FALSE(stream::encode_sbs_telemetry_state_payload(state, encoded));
  EXPECT_TRUE(std::all_of(std::begin(encoded), std::end(encoded), [](std::uint8_t byte) {
    return byte == 0xA5;
  }));
}
