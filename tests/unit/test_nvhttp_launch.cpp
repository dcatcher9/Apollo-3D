/**
 * @file tests/unit/test_nvhttp_launch.cpp
 * @brief Tests for GameStream launch parameter parsing.
 */

#include <src/nvhttp.h>

#include "../tests_common.h"

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
