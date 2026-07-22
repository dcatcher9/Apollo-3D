/**
 * @file tests/unit/test_nvhttp_launch.cpp
 * @brief Tests for GameStream launch parameter parsing.
 */

#include "../tests_common.h"

#include <cstdint>
#include <limits>

#include <src/nvhttp.h>

TEST(NvHttpLaunchParsingTest, AcceptsDeploymentModesAndFractionalRates) {
  EXPECT_EQ(nvhttp::parse_launch_mode("5120x2160x120"), (nvhttp::launch_mode_t {5120, 2160, 120000}));
  EXPECT_EQ(nvhttp::parse_launch_mode("3552x3840x59940"), (nvhttp::launch_mode_t {3552, 3840, 59940}));
  EXPECT_EQ(nvhttp::parse_launch_mode("1920x1080x59.94"), (nvhttp::launch_mode_t {1920, 1080, 59940}));
  EXPECT_EQ(nvhttp::parse_launch_mode("1x1x1"), (nvhttp::launch_mode_t {1, 1, 1000}));
  EXPECT_EQ(nvhttp::parse_launch_mode("16384x16384x1000000"), (nvhttp::launch_mode_t {16384, 16384, 1000000}));
}

TEST(NvHttpLaunchParsingTest, RejectsMalformedAndOutOfRangeModes) {
  for (const auto mode : {
         "",
         "1920x1080",
         "1920x1080x60x1",
         "1920xx60",
         " 1920x1080x60",
         "1920x1080x60fps",
         "0x1080x60",
         "1920x-1x60",
         "16385x1080x60",
         "1920x1080x0",
         "1920x1080x0.5",
         "1920x1080xNaN",
         "1920x1080xInf",
         "1920x1080x1000001",
       }) {
    EXPECT_FALSE(nvhttp::parse_launch_mode(mode)) << mode;
  }
}

TEST(NvHttpLaunchParsingTest, ParsesExactRemoteInputEncryptionValues) {
  const auto key = nvhttp::parse_remote_input_key("000102030405060708090A0B0C0D0E0F");
  ASSERT_TRUE(key);
  ASSERT_EQ(key->size(), 16);
  for (std::size_t index = 0; index < key->size(); ++index) {
    EXPECT_EQ((*key)[index], index);
  }

  EXPECT_EQ(nvhttp::parse_remote_input_key_id("0"), 0U);
  EXPECT_EQ(nvhttp::parse_remote_input_key_id("2147483647"), 0x7FFFFFFFU);
  EXPECT_EQ(nvhttp::parse_remote_input_key_id("2147483648"), 0x80000000U);
  EXPECT_EQ(nvhttp::parse_remote_input_key_id("-2147483648"), 0x80000000U);
  EXPECT_EQ(nvhttp::parse_remote_input_key_id("-1"), 0xFFFFFFFFU);
  EXPECT_EQ(nvhttp::parse_remote_input_key_id("4294967295"), 0xFFFFFFFFU);
}

TEST(NvHttpLaunchParsingTest, RejectsMalformedRemoteInputEncryptionValues) {
  EXPECT_FALSE(nvhttp::parse_remote_input_key(""));
  EXPECT_FALSE(nvhttp::parse_remote_input_key("000102030405060708090A0B0C0D0E"));
  EXPECT_FALSE(nvhttp::parse_remote_input_key("000102030405060708090A0B0C0D0E0G"));
  EXPECT_FALSE(nvhttp::parse_remote_input_key_id(""));
  EXPECT_FALSE(nvhttp::parse_remote_input_key_id("+1"));
  EXPECT_FALSE(nvhttp::parse_remote_input_key_id("1x"));
  EXPECT_FALSE(nvhttp::parse_remote_input_key_id("-2147483649"));
  EXPECT_FALSE(nvhttp::parse_remote_input_key_id("4294967296"));
}

TEST(NvHttpLaunchParsingTest, ValidatesRetainedHostSessionTokensExactly) {
  EXPECT_EQ(nvhttp::parse_host_session_id("1"), 1U);
  EXPECT_EQ(nvhttp::parse_host_session_id("18446744073709551615"), std::numeric_limits<std::uint64_t>::max());
  EXPECT_FALSE(nvhttp::parse_host_session_id(""));
  EXPECT_FALSE(nvhttp::parse_host_session_id("0"));
  EXPECT_FALSE(nvhttp::parse_host_session_id("-1"));
  EXPECT_FALSE(nvhttp::parse_host_session_id("1x"));
  EXPECT_FALSE(nvhttp::parse_host_session_id("18446744073709551616"));

  EXPECT_TRUE(nvhttp::detail::host_session_matches(42, 42));
  EXPECT_FALSE(nvhttp::detail::host_session_matches(0, 0));
  EXPECT_FALSE(nvhttp::detail::host_session_matches(42, 0));
  EXPECT_FALSE(nvhttp::detail::host_session_matches(42, 41));
}

TEST(NvHttpLaunchParsingTest, MatchesCanonicalApplicationIdentity) {
  EXPECT_TRUE(nvhttp::detail::app_identity_matches(7, "ABC-DEF", 7, "abc-def"));
  EXPECT_TRUE(nvhttp::detail::app_identity_matches(std::nullopt, "ABC-DEF", 7, "abc-def"));
  EXPECT_TRUE(nvhttp::detail::app_identity_matches(7, "", 7, "abc-def"));
  EXPECT_FALSE(nvhttp::detail::app_identity_matches(8, "ABC-DEF", 7, "abc-def"));
  EXPECT_FALSE(nvhttp::detail::app_identity_matches(7, "OTHER", 7, "abc-def"));
  EXPECT_FALSE(nvhttp::detail::app_identity_matches(std::nullopt, "", 7, "abc-def"));
}

TEST(NvHttpLaunchParsingTest, EnforcesScalarLaunchOptionContracts) {
  using field = nvhttp::launch_int_field;
  using nvhttp::parse_launch_int;

  EXPECT_EQ(parse_launch_int(field::binary_option, "0"), 0);
  EXPECT_EQ(parse_launch_int(field::binary_option, "1"), 1);
  EXPECT_FALSE(parse_launch_int(field::binary_option, "2"));
  EXPECT_EQ(parse_launch_int(field::scale_factor, "20"), 20);
  EXPECT_EQ(parse_launch_int(field::scale_factor, "200"), 200);
  EXPECT_FALSE(parse_launch_int(field::scale_factor, "19"));
  EXPECT_FALSE(parse_launch_int(field::scale_factor, "201"));
  EXPECT_EQ(parse_launch_int(field::app_id, "0"), 0);
  EXPECT_FALSE(parse_launch_int(field::app_id, "-1"));
  EXPECT_EQ(parse_launch_int(field::surround_info, "196610"), 196610);
  EXPECT_FALSE(parse_launch_int(field::surround_info, "0"));
  EXPECT_FALSE(parse_launch_int(field::core_version, "-1"));
  EXPECT_FALSE(parse_launch_int(field::core_version, "0"));
  EXPECT_FALSE(parse_launch_int(field::core_version, "1x"));
  EXPECT_EQ(parse_launch_int(field::sbs_mode, "0"), 0);
  EXPECT_EQ(parse_launch_int(field::sbs_mode, "1"), 1);
  EXPECT_FALSE(parse_launch_int(field::sbs_mode, "-1"));
  EXPECT_FALSE(parse_launch_int(field::sbs_mode, "2"));
}
