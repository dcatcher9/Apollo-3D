/**
 * @file tests/unit/test_stream.cpp
 * @brief Test src/stream.*
 */

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
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
