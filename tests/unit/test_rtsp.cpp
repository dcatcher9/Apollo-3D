/**
 * @file tests/unit/test_rtsp.cpp
 * @brief Tests for RTSP request parsing helpers.
 */

#include <string_view>
#include <unordered_map>

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
