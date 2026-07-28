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
  EXPECT_LT(low_rate.target_wire_bps, 20'000'000);
  EXPECT_EQ(low_rate.target_wire_bps, 16'588'801);
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

TEST(ControlPayloadValidationTests, EnforcesDecryptedInnerLength) {
  EXPECT_TRUE(stream::is_valid_decrypted_control_payload(stream::CONTROL_HEADER_V2_SIZE, 0));
  EXPECT_TRUE(stream::is_valid_decrypted_control_payload(stream::CONTROL_HEADER_V2_SIZE + 17, 17));
  EXPECT_FALSE(stream::is_valid_decrypted_control_payload(stream::CONTROL_HEADER_V2_SIZE - 1, 0));
  EXPECT_FALSE(stream::is_valid_decrypted_control_payload(stream::CONTROL_HEADER_V2_SIZE + 17, 16));
  EXPECT_FALSE(stream::is_valid_decrypted_control_payload(stream::CONTROL_HEADER_V2_SIZE + 17, 18));
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

TEST(LiveVideoModeAckTests, EncodesTheAppliedModeInWireOrder) {
  const stream::live_video_mode_ack_t ack {
    0xBEEF,
    stream::live_video_mode_ack_e::rejected_needs_reconnect,
    1920,
    1080,
    2997,
    40000,
  };

  std::uint8_t payload[stream::LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_live_video_mode_ack_payload(ack, payload));

  // u16 request_id, u16 status, u16 applied_width, u16 applied_height,
  // u16 applied_framerate_x100, u32 applied_bitrate_kbps -- all little-endian, no padding. The
  // trailing u32 lands on an odd multiple of two on purpose; the body is packed.
  const std::uint8_t expected[stream::LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE] = {
    0xEF,
    0xBE,  // request_id 0xBEEF echoed verbatim
    0x02,
    0x00,  // rejected_needs_reconnect
    0x80,
    0x07,  // 1920
    0x38,
    0x04,  // 1080
    0xB5,
    0x0B,  // 2997
    0x40,
    0x9C,
    0x00,
    0x00,  // 40000
  };
  EXPECT_TRUE(std::equal(std::begin(payload), std::end(payload), std::begin(expected)));
}

TEST(LiveVideoModeAckTests, ReportsAClampedApplyAsAppliedWithTheClampedMode) {
  // The whole point of the applied_* fields: a per-eye 5120x2160 SBS request packs to 10240, which
  // exceeds the 8192 codec ceiling, so the host installs 4096 per eye with an aspect-scaled height.
  // That is a successful apply, not a failure, and the client must be told the clamped geometry so
  // its confirmation check does not mistake the clamp for a refusal and revert.
  const auto packed = video::host_sbs_output_dimensions(5120, 2160, 2, 8192, 8192);
  ASSERT_EQ(packed.width, 8192);
  ASSERT_EQ(packed.height, 1728);

  // The encode loop reports the BASE per-eye geometry, before SBS doubling.
  const video::effective_video_mode_t effective {packed.width / 2, packed.height, 6000, 38000};
  const video::video_mode_applied_t report {
    {{7, 41}},
    true,
    effective,
    {5120, 2160, 60000},
  };

  const auto status = report.applied ?
                        stream::live_video_mode_ack_e::applied :
                        stream::live_video_mode_ack_e::failed;
  EXPECT_EQ(status, stream::live_video_mode_ack_e::applied);

  const stream::live_video_mode_ack_t ack {
    report.requests.front().request_id,
    status,
    report.mode.width,
    report.mode.height,
    report.mode.framerateX100,
    report.mode.bitrate,
  };
  EXPECT_NE(ack.applied_width, 5120);
  EXPECT_EQ(ack.applied_width, 4096);
  EXPECT_EQ(ack.applied_height, 1728);
  EXPECT_EQ(report.desktop_mode.width, 5120);
  EXPECT_EQ(report.desktop_mode.height, 2160);
  EXPECT_EQ(report.desktop_mode.framerate_millihz, 60000);

  std::uint8_t payload[stream::LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_live_video_mode_ack_payload(ack, payload));
  EXPECT_EQ(payload[2], 0x00);  // status applied
  EXPECT_EQ(payload[3], 0x00);
  EXPECT_EQ(payload[4], 0x00);  // 4096
  EXPECT_EQ(payload[5], 0x10);
  EXPECT_EQ(payload[6], 0xC0);  // 1728
  EXPECT_EQ(payload[7], 0x06);
}

TEST(LiveVideoModeAckTests, ReportsANonSbsClampAsAppliedWithTheClampedMode) {
  // Same contract without SBS: H.264 tops out at 4096, so a 5120-wide live request installs 4096
  // with an aspect-scaled height and still reports success.
  const auto clamped = video::clamp_encode_dimensions(5120, 2160, 0, 4096);
  ASSERT_EQ(clamped.width, 4096);
  ASSERT_EQ(clamped.height, 1728);

  const stream::live_video_mode_ack_t ack {
    3,
    stream::live_video_mode_ack_e::applied,
    clamped.width,
    clamped.height,
    6000,
    25000,
  };

  std::uint8_t payload[stream::LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_live_video_mode_ack_payload(ack, payload));
  EXPECT_EQ(static_cast<int>(payload[2]) | (static_cast<int>(payload[3]) << 8), 0);
  EXPECT_EQ(static_cast<int>(payload[4]) | (static_cast<int>(payload[5]) << 8), 4096);
  EXPECT_EQ(static_cast<int>(payload[6]) | (static_cast<int>(payload[7]) << 8), 1728);
}

TEST(LiveVideoModeAckTests, EncodesTheWireFieldExtremes) {
  const stream::live_video_mode_ack_t ack {
    std::numeric_limits<std::uint16_t>::max(),
    stream::live_video_mode_ack_e::failed,
    std::numeric_limits<std::uint16_t>::max(),
    std::numeric_limits<std::uint16_t>::max(),
    std::numeric_limits<std::uint16_t>::max(),
    std::numeric_limits<std::uint32_t>::max(),
  };

  std::uint8_t payload[stream::LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_live_video_mode_ack_payload(ack, payload));
  EXPECT_EQ(payload[0], 0xFF);
  EXPECT_EQ(payload[1], 0xFF);
  EXPECT_EQ(payload[2], 0x03);
  EXPECT_EQ(payload[3], 0x00);
  for (std::size_t i = 4; i < stream::LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE; ++i) {
    EXPECT_EQ(payload[i], 0xFF);
  }

  const stream::live_video_mode_ack_t zeroed {0, stream::live_video_mode_ack_e::applied, 0, 0, 0, 0};
  ASSERT_TRUE(stream::encode_live_video_mode_ack_payload(zeroed, payload));
  for (const auto byte : payload) {
    EXPECT_EQ(byte, 0x00);
  }
}

TEST(LiveVideoModeAckTests, RefusesValuesThatCannotHaveComeOffTheWire) {
  // Every reachable acknowledgement carries an id decoded from the request's own u16 field and a
  // mode the encoder actually ran, so these are host bugs rather than client input. The encoder
  // must not truncate them silently.
  std::uint8_t payload[stream::LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE] {};
  const auto reject = [&](const stream::live_video_mode_ack_t &ack) {
    EXPECT_FALSE(stream::encode_live_video_mode_ack_payload(ack, payload));
  };

  const auto status = stream::live_video_mode_ack_e::applied;
  reject({65536, status, 1920, 1080, 6000, 20000});
  reject({-1, status, 1920, 1080, 6000, 20000});
  reject({1, status, 65536, 1080, 6000, 20000});
  reject({1, status, 1920, 65536, 6000, 20000});
  reject({1, status, 1920, 1080, 65536, 20000});
  reject({1, status, 1920, 1080, 6000, 4294967296LL});
  reject({1, status, -1, 1080, 6000, 20000});
  reject({1, status, 1920, -1, 6000, 20000});
  reject({1, status, 1920, 1080, -1, 20000});
  reject({1, status, 1920, 1080, 6000, -1});

  // A refused encode leaves the caller's buffer untouched rather than half-written.
  for (const auto byte : payload) {
    EXPECT_EQ(byte, 0x00);
  }
}

TEST(LiveVideoModeAckTests, RefusalsReportTheModeStillInEffect) {
  // A refused request must never be answered with zeros or with the rejected request echoed back:
  // the client resynchronizes its UI from these fields.
  video::effective_video_mode_publisher_t publisher {{1920, 1080, 6000, 20000}};
  EXPECT_EQ(publisher.current().width, 1920);

  publisher.publish({2560, 1440, 12000, 45000});
  const auto in_effect = publisher.current();
  EXPECT_EQ(in_effect.width, 2560);
  EXPECT_EQ(in_effect.height, 1440);
  EXPECT_EQ(in_effect.framerateX100, 12000);
  EXPECT_EQ(in_effect.bitrate, 45000);

  const stream::live_video_mode_ack_t ack {
    42,
    stream::live_video_mode_ack_e::rejected_needs_reconnect,
    in_effect.width,
    in_effect.height,
    in_effect.framerateX100,
    in_effect.bitrate,
  };
  std::uint8_t payload[stream::LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_live_video_mode_ack_payload(ack, payload));
  EXPECT_EQ(static_cast<int>(payload[0]) | (static_cast<int>(payload[1]) << 8), 42);
  EXPECT_EQ(static_cast<int>(payload[4]) | (static_cast<int>(payload[5]) << 8), 2560);
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
