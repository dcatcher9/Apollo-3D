/**
 * @file tests/unit/test_config_migration.cpp
 * @brief Tests for retired host configuration boundaries.
 */

#include <src/config.h>

#include "../tests_common.h"

TEST(ConfigMigration, RetiredOptionIsConsumedDuringParsing) {
  const auto parsed = config::parse_config(
    "sunshine_name = XR host\n"
    "sbs_3d_foreground_subtitles = enabled\n"
    "virtual_display_restart_explorer = enabled\n"
  );

  EXPECT_FALSE(parsed.contains("sbs_3d_foreground_subtitles"));
  EXPECT_EQ(parsed.at("sunshine_name"), "XR host");
  EXPECT_EQ(parsed.at("virtual_display_restart_explorer"), "enabled");
}

TEST(ConfigMigration, OnlyTheExactRetiredOptionIsScrubbed) {
  EXPECT_TRUE(config::is_retired_config_option("sbs_3d_foreground_subtitles"));
  EXPECT_FALSE(config::is_retired_config_option("sbs_3d_subtitle_apps"));
  EXPECT_FALSE(config::is_retired_config_option("sbs_3d_foreground"));
  EXPECT_FALSE(config::is_retired_config_option("SBS_3D_Foreground_Subtitles"));
}
