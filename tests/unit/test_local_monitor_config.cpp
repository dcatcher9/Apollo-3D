#include "src/config.h"

#include <gtest/gtest.h>

namespace {
  std::unordered_map<std::string, std::string> enabled_monitor() {
    return {
      {"local_monitor_enabled", "enabled"},
      {"local_monitor_display_id", R"(\\?\DISPLAY#GSM1234#physical)"},
      {"local_monitor_emitter_port", "COM7"},
    };
  }
}  // namespace

TEST(LocalMonitorConfig, DefaultsKeepTheMonitorDisabled) {
  std::unordered_map<std::string, std::string> values {{"sunshine_name", "My host"}};
  const auto monitor = config::parse_local_monitor_config(values);
  EXPECT_FALSE(monitor.enabled);
  EXPECT_EQ(monitor.mode, "host");
  EXPECT_EQ(monitor.eye_repeats, 1);
  EXPECT_EQ(values.at("sunshine_name"), "My host");
}

TEST(LocalMonitorConfig, AcceptsBothProvidersAndConsumesOnlySurfaceKeys) {
  for (const auto *mode : {"host", "game"}) {
    auto values = enabled_monitor();
    values["local_monitor_mode"] = mode;
    values["local_monitor_eye_repeats"] = "2";
    values["local_monitor_frame_delay_us"] = "0";
    values["local_monitor_flip_eyes"] = "true";
    values["output_name"] = "Other display";
    const auto monitor = config::parse_local_monitor_config(values);
    EXPECT_TRUE(monitor.enabled);
    EXPECT_EQ(monitor.mode, mode);
    EXPECT_EQ(monitor.eye_repeats, 2);
    EXPECT_EQ(monitor.frame_delay_us, 0);
    EXPECT_TRUE(monitor.flip_eyes);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_EQ(values.at("output_name"), "Other display");
  }
}

TEST(LocalMonitorConfig, MalformedOrOverflowTimingCannotActivateTheMonitor) {
  for (const auto *timing : {"", "2000junk", "1.5", "-1", "65001", "99999999999999999999999999"}) {
    auto values = enabled_monitor();
    values["local_monitor_frame_duration_us"] = timing;
    EXPECT_FALSE(config::parse_local_monitor_config(values).enabled) << timing;
    EXPECT_TRUE(values.empty());
  }
}

TEST(LocalMonitorConfig, RejectsUnsupportedProtocolAndRepeatCounts) {
  for (const auto &[key, value] : std::vector<std::pair<std::string, std::string>> {
         {"local_monitor_ir_protocol", "7"},
         {"local_monitor_eye_repeats", "0"},
         {"local_monitor_eye_repeats", "5"},
         {"local_monitor_frame_duration_us", "0"},
         {"local_monitor_frame_delay_us", "-1"},
         {"local_monitor_signal_spacing_us", "65001"},
       }) {
    auto values = enabled_monitor();
    values[key] = value;
    EXPECT_FALSE(config::parse_local_monitor_config(values).enabled) << key << '=' << value;
  }
}

TEST(LocalMonitorConfig, InvalidModeAndBooleanFailClosed) {
  for (const auto &[key, value] : std::vector<std::pair<std::string, std::string>> {
         {"local_monitor_mode", "sbs"},
         {"local_monitor_mode", ""},
         {"local_monitor_enabled", "perhaps"},
         {"local_monitor_flip_eyes", "perhaps"},
       }) {
    auto values = enabled_monitor();
    values[key] = value;
    EXPECT_FALSE(config::parse_local_monitor_config(values).enabled) << key << '=' << value;
  }
}

TEST(LocalMonitorConfig, RequiresBothExactMonitorSelectionAndEmitterPort) {
  for (const auto *key : {"local_monitor_display_id", "local_monitor_emitter_port"}) {
    auto values = enabled_monitor();
    values.erase(key);
    EXPECT_FALSE(config::parse_local_monitor_config(values).enabled) << key;
  }
}
