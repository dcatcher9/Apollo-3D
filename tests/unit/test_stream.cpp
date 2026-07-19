/**
 * @file tests/unit/test_stream.cpp
 * @brief Test src/stream.*
 */

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <src/stream.h>
#include <src/utility.h>

namespace stream {
  std::vector<uint8_t> concat_and_insert(uint64_t insert_size, uint64_t slice_size, const std::string_view &data1, const std::string_view &data2);
}

#include "../tests_common.h"

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
