/**
 * @file tests/unit/test_stream.cpp
 * @brief Test src/stream.*
 */

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
  const video::video_mode_applied_t report {{7}, true, effective};

  const auto status = report.applied ?
                        stream::live_video_mode_ack_e::applied :
                        stream::live_video_mode_ack_e::failed;
  EXPECT_EQ(status, stream::live_video_mode_ack_e::applied);

  const stream::live_video_mode_ack_t ack {
    report.request_ids.front(),
    status,
    report.mode.width,
    report.mode.height,
    report.mode.framerateX100,
    report.mode.bitrate,
  };
  EXPECT_NE(ack.applied_width, 5120);
  EXPECT_EQ(ack.applied_width, 4096);
  EXPECT_EQ(ack.applied_height, 1728);

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
