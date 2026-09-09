/**
 * @file tests/unit/test_config_migration.cpp
 * @brief Tests for retired host configuration boundaries.
 */

#include <src/config.h>

#include "../tests_common.h"

TEST(ConfigMigration, RetiredVirtualDisplayOnlyIsConsumedDuringParsing) {
  const auto parsed = config::parse_config(
    "sunshine_name = XR host\n"
    "virtual_display_only = on\n"
    "virtual_display_restart_explorer = enabled\n"
  );

  EXPECT_FALSE(parsed.contains("virtual_display_only"));
  EXPECT_EQ(parsed.at("sunshine_name"), "XR host");
  EXPECT_EQ(parsed.at("virtual_display_restart_explorer"), "enabled");
}

TEST(ConfigMigration, OnlyTheExactRetiredOptionIsScrubbed) {
  EXPECT_TRUE(config::is_retired_config_option("virtual_display_only"));
  EXPECT_FALSE(config::is_retired_config_option("virtual_display"));
  EXPECT_FALSE(config::is_retired_config_option("virtual_display_restart_explorer"));
  EXPECT_FALSE(config::is_retired_config_option("Virtual_Display_Only"));
}
