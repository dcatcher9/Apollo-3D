/**
 * @file tests/unit/test_rtsp.cpp
 * @brief Tests for RTSP request parsing helpers.
 */

#include <limits>
#include <string_view>
#include <unordered_map>

#include <src/stream.h>
#include <src/rtsp.h>

#include "../tests_common.h"

namespace rtsp_stream {
  namespace detail {
    std::unordered_map<std::string_view, std::string_view> parse_announce_attributes(std::string_view payload);
  }
}

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
  EXPECT_FALSE(parse_announce_int(field::control_protocol, "0"));
  EXPECT_EQ(parse_announce_int(field::video_format, "2"), 2);
  EXPECT_FALSE(parse_announce_int(field::video_format, "3"));
  EXPECT_EQ(parse_announce_int(field::binary_option, "1"), 1);
  EXPECT_FALSE(parse_announce_int(field::binary_option, "2"));
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

TEST(RtspAnnounceParsingTest, AppliesOnlyValidLowerPacketSizeLimits) {
  using rtsp_stream::detail::apply_packet_size_limit;

  EXPECT_EQ(apply_packet_size_limit(1392, 0), 1392);
  EXPECT_EQ(apply_packet_size_limit(1392, 1456), 1392);
  EXPECT_EQ(apply_packet_size_limit(1392, 1346), 1346);
  EXPECT_EQ(apply_packet_size_limit(1392, stream::VIDEO_PACKET_SIZE_MIN), stream::VIDEO_PACKET_SIZE_MIN);
  EXPECT_EQ(apply_packet_size_limit(1392, stream::VIDEO_PACKET_SIZE_MIN - 1), 1392);
  EXPECT_EQ(apply_packet_size_limit(1392, stream::VIDEO_PACKET_SIZE_MAX + 1), 1392);
}

TEST(RtspPlaintextParsingTest, FindsHeaderDelimiterAcrossReadBoundary) {
  using rtsp_stream::detail::find_plaintext_header_end;

  constexpr std::string_view request = "OPTIONS rtsp://host RTSP/1.0\r\nCSeq: 1\r\n\r\nbody";
  const auto split = request.find("\r\n\r\n") + 2;
  const auto end = find_plaintext_header_end(request, split);

  ASSERT_TRUE(end);
  EXPECT_EQ(*end, request.find("body"));
  EXPECT_FALSE(find_plaintext_header_end(request.substr(0, split), split));
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
  auto first = std::make_shared<rtsp_stream::launch_session_t>();
  first->id = 101;
  first->unique_id = "first";
  auto second = std::make_shared<rtsp_stream::launch_session_t>();
  second->id = 102;
  second->unique_id = "second";

  ASSERT_TRUE(rtsp_stream::launch_session_raise(first));
  EXPECT_FALSE(rtsp_stream::launch_session_raise(second));

  rtsp_stream::launch_session_clear(first->id);
  EXPECT_TRUE(rtsp_stream::launch_session_raise(second));
  rtsp_stream::launch_session_clear(second->id);
}
