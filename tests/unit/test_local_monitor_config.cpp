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
    values["local_monitor_frame_delay_us"] = "500";
    values["local_monitor_flip_eyes"] = "true";
    values["output_name"] = "Other display";
    const auto monitor = config::parse_local_monitor_config(values);
    EXPECT_TRUE(monitor.enabled);
    EXPECT_EQ(monitor.mode, mode);
    EXPECT_EQ(monitor.eye_repeats, 2);
    EXPECT_EQ(monitor.frame_delay_us, 500);
    EXPECT_TRUE(monitor.flip_eyes);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_EQ(values.at("output_name"), "Other display");
  }
}

TEST(LocalMonitorConfig, MalformedOrOverflowTimingCannotActivateTheMonitor) {
  for (const auto *timing : {"", "2000junk", "1.5", "-1", "32768", "40000", "99999999999999999999999999"}) {
    auto values = enabled_monitor();
    values["local_monitor_frame_duration_us"] = timing;
    EXPECT_FALSE(config::parse_local_monitor_config(values).enabled) << timing;
    EXPECT_TRUE(values.empty());
  }
}

TEST(LocalMonitorConfig, RejectsUnsupportedProtocolAndRepeatCounts) {
  for (const auto &[key, value] : std::vector<std::pair<std::string, std::string>> {
         {"local_monitor_ir_protocol", "7"},
         {"local_monitor_ir_protocol", "0"},
         {"local_monitor_ir_protocol", "1"},
         {"local_monitor_ir_protocol", "3"},
         {"local_monitor_eye_repeats", "0"},
         {"local_monitor_eye_repeats", "5"},
         {"local_monitor_frame_duration_us", "0"},
         {"local_monitor_frame_delay_us", "-1"},
         {"local_monitor_frame_delay_us", "0"},
         {"local_monitor_signal_spacing_us", "0"},
         {"local_monitor_signal_spacing_us", "1024"},
       }) {
    auto values = enabled_monitor();
    values[key] = value;
    EXPECT_FALSE(config::parse_local_monitor_config(values).enabled) << key << '=' << value;
  }
}

TEST(LocalMonitorConfig, AcceptsOnlyFiniteShutterProtocolsAtTheTimerBoundary) {
  struct profile_t {
    const char *protocol;
    int minimum_delay;
    int maximum_spacing;
  };
  for (const auto &[protocol, minimum_delay, maximum_spacing] : std::vector<profile_t> {
         {"2", 150, 882}, {"4", 100, 483}, {"5", 100, 803}, {"6", 100, 803},
       }) {
    auto values = enabled_monitor();
    values["local_monitor_ir_protocol"] = protocol;
    values["local_monitor_frame_delay_us"] = std::to_string(minimum_delay);
    values["local_monitor_frame_duration_us"] = std::to_string(32767 - minimum_delay);
    values["local_monitor_signal_spacing_us"] = std::to_string(maximum_spacing);
    const auto monitor = config::parse_local_monitor_config(values);
    EXPECT_TRUE(monitor.enabled) << protocol;
    EXPECT_EQ(monitor.frame_delay_us + monitor.frame_duration_us, 32767);
    EXPECT_EQ(monitor.signal_spacing_us, maximum_spacing);
  }
}

TEST(LocalMonitorConfig, RejectsProtocolSpecificPulseAndSpacingViolations) {
  struct invalid_t {
    const char *protocol;
    const char *key;
    const char *value;
  };
  for (const auto &[protocol, key, value] : std::vector<invalid_t> {
         {"2", "local_monitor_frame_delay_us", "149"},
         {"4", "local_monitor_frame_delay_us", "99"},
         {"5", "local_monitor_frame_delay_us", "1"},
         {"2", "local_monitor_frame_duration_us", "329"},
         {"4", "local_monitor_frame_duration_us", "669"},
         {"5", "local_monitor_frame_duration_us", "349"},
         {"6", "local_monitor_frame_duration_us", "349"},
         {"2", "local_monitor_signal_spacing_us", "883"},
         {"4", "local_monitor_signal_spacing_us", "484"},
         {"5", "local_monitor_signal_spacing_us", "804"},
       }) {
    auto values = enabled_monitor();
    values["local_monitor_ir_protocol"] = protocol;
    values[key] = value;
    EXPECT_FALSE(config::parse_local_monitor_config(values).enabled) << protocol << ':' << key << '=' << value;
  }
}

TEST(LocalMonitorConfig, RejectsCombinedTimerOverflowEvenWhenBothFieldsFitIndividually) {
  auto values = enabled_monitor();
  values["local_monitor_frame_delay_us"] = "16000";
  values["local_monitor_frame_duration_us"] = "16768";
  EXPECT_FALSE(config::parse_local_monitor_config(values).enabled);

  values = enabled_monitor();
  values["local_monitor_frame_delay_us"] = "32367";
  values["local_monitor_frame_duration_us"] = "400";
  EXPECT_TRUE(config::parse_local_monitor_config(values).enabled);
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

TEST(LocalDisplaySurface, DefaultsToGlassesAndReadsTheEarlierMonitorSwitch) {
  using surface_e = config::video_t::local_surface_e;
  std::unordered_map<std::string, std::string> empty;
  surface_e surface = surface_e::off;
  EXPECT_FALSE(config::parse_local_monitor_config(empty, &surface).enabled);
  EXPECT_EQ(surface, surface_e::ar_glasses);

  std::unordered_map<std::string, std::string> earlier {
    {"local_monitor_enabled", "enabled"},
    {"local_monitor_display_id", "monitor"},
    {"local_monitor_emitter_port", "COM4"},
  };
  EXPECT_TRUE(config::parse_local_monitor_config(earlier, &surface).enabled);
  EXPECT_EQ(surface, surface_e::shutter_monitor);
  EXPECT_TRUE(earlier.empty());
}

TEST(LocalDisplaySurface, TheChoiceDecidesOverTheEarlierSwitch) {
  using surface_e = config::video_t::local_surface_e;
  surface_e surface = surface_e::off;
  std::unordered_map<std::string, std::string> glasses {
    {"local_display_surface", "ar_glasses"},
    {"local_monitor_enabled", "enabled"},
    {"local_monitor_display_id", "monitor"},
    {"local_monitor_emitter_port", "COM4"},
  };
  EXPECT_FALSE(config::parse_local_monitor_config(glasses, &surface).enabled);
  EXPECT_EQ(surface, surface_e::ar_glasses);

  std::unordered_map<std::string, std::string> monitor {
    {"local_display_surface", "shutter_monitor"},
    {"local_monitor_display_id", "monitor"},
    {"local_monitor_emitter_port", "COM4"},
  };
  EXPECT_TRUE(config::parse_local_monitor_config(monitor, &surface).enabled);
  EXPECT_EQ(surface, surface_e::shutter_monitor);

  std::unordered_map<std::string, std::string> off {{"local_display_surface", "off"}};
  EXPECT_FALSE(config::parse_local_monitor_config(off, &surface).enabled);
  EXPECT_EQ(surface, surface_e::off);
}

TEST(LocalDisplaySurface, AnInvalidChoiceOrMonitorCannotStartAnything) {
  using surface_e = config::video_t::local_surface_e;
  surface_e surface = surface_e::ar_glasses;
  std::unordered_map<std::string, std::string> unknown {{"local_display_surface", "projector"}};
  EXPECT_FALSE(config::parse_local_monitor_config(unknown, &surface).enabled);
  EXPECT_EQ(surface, surface_e::off);

  // A chosen monitor with no emitter must not fall back to glasses.
  surface = surface_e::ar_glasses;
  std::unordered_map<std::string, std::string> incomplete {
    {"local_display_surface", "shutter_monitor"},
    {"local_monitor_display_id", "monitor"},
  };
  EXPECT_FALSE(config::parse_local_monitor_config(incomplete, &surface).enabled);
  EXPECT_EQ(surface, surface_e::off);
}

TEST(LocalMonitorConfig, EmitterAveragingIsOffUnlessEnabled) {
  auto values = enabled_monitor();
  EXPECT_FALSE(config::parse_local_monitor_config(values).emitter_averaging);
  values = enabled_monitor();
  values["local_monitor_emitter_averaging"] = "enabled";
  const auto monitor = config::parse_local_monitor_config(values);
  EXPECT_TRUE(monitor.enabled);
  EXPECT_TRUE(monitor.emitter_averaging);
  values = enabled_monitor();
  values["local_monitor_emitter_averaging"] = "perhaps";
  EXPECT_FALSE(config::parse_local_monitor_config(values).enabled);
}
