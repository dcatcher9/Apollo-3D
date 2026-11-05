#ifdef _WIN32

  #include "src/platform/windows/open3d_emitter.h"

  #include <gtest/gtest.h>

namespace {
  using namespace platf::open3d;

  constexpr std::string_view parameters = "parameters 20,6,500,7000,30,7500,10,128,0,1,1,0,0,0,32,1,8333,0";

  TEST(Open3DEmitter, RestrictsDeviceAccessToExplicitComPorts) {
    EXPECT_EQ(detail::normalize_port("com12"), "COM12");
    EXPECT_EQ(detail::normalize_port(R"(\\.\COM123)"), "COM123");
    for (const auto *port : {"", "COM0", "COM65536", "COM1garbage", "COM1\n", "COM-1", "\\\\.\\PhysicalDrive0", "LPT1"}) {
      EXPECT_FALSE(detail::normalize_port(port)) << port;
    }
  }

  TEST(Open3DEmitter, RejectsUnprovenProtocolShapesAndMalformedNumbers) {
    EXPECT_TRUE(detail::parse_parameters(parameters));
    for (const auto *line : {
           "parameters 19,6,500,7000,30,7500,10,128,0,1,1,0,0,0,32,1,8333,0",
           "parameters 20,6,500,7000,30,7500,10,128,0,1,1,0,0,0,32,1,8333",
           "parameters 20,6,500,7000,30,7500,10,128,0,1,1,0,0,0,32,1,8333,0junk",
           "parameters 20,6,500,7000,30,7500,10,128,0,1,1,0,0,0,32,1,-1,0",
           "parameters 20,6,500,7000,30,7500,10,128,0,1,1,0,0,0,32,1,999999999999,0",
           "parameters 20,6,500,7000,30,7500,10,128,0,1,1,0,0,0,32,1,8333,",
           "other 20,6,500,7000,30,7500,10,128,0,1,1,0,0,0,32,1,8333,0"
         }) {
      EXPECT_FALSE(detail::parse_parameters(line)) << line;
    }
  }

  TEST(Open3DEmitter, PreservesReadbackFieldsWhileApplyingVolatilePcSerialProfile) {
    auto previous = detail::parse_parameters(std::string(parameters) + ",42,99");
    ASSERT_TRUE(previous);
    settings_t settings;
    settings.port = "COM12";
    settings.protocol = 5;
    settings.frame_delay_us = 350;
    settings.frame_duration_us = 1800;
    settings.signal_spacing_us = 20;
    settings.flip_eyes = true;
    settings.target_frametime_us = 6061;
    const auto command = detail::configuration_command(*previous, settings);
    ASSERT_TRUE(command);
    EXPECT_EQ(*command, "0,5,350,1800,20,7500,10,128,0,0,0,0,0,1,32,0,6061,1,42,99");
    const auto applied = detail::parse_parameters("parameters 20" + command->substr(1));
    ASSERT_TRUE(applied);
    EXPECT_TRUE(detail::configuration_matches(*applied, *previous, settings));
    auto changed = *applied;
    changed.values[16] = 0;
    EXPECT_FALSE(detail::configuration_matches(changed, *previous, settings));
    changed = *applied;
    changed.values[4] = 123;
    EXPECT_FALSE(detail::configuration_matches(changed, *previous, settings));
    changed = *applied;
    ++changed.firmware_version;
    EXPECT_FALSE(detail::configuration_matches(changed, *previous, settings));
  }

  TEST(Open3DEmitter, RejectsInvalidSettingsBeforeBuildingOrOpening) {
    const auto previous = detail::parse_parameters(parameters);
    ASSERT_TRUE(previous);
    settings_t settings;
    settings.port = "COM4";
    EXPECT_TRUE(detail::valid_settings(settings));
    settings.frame_duration_us = 0;
    EXPECT_FALSE(detail::configuration_command(*previous, settings));
    settings.frame_duration_us = 2000;
    settings.target_frametime_us = 65536;
    EXPECT_FALSE(detail::configuration_command(*previous, settings));
    emitter_t emitter;
    EXPECT_FALSE(emitter.healthy());
    EXPECT_FALSE(emitter.signal(true));
    EXPECT_FALSE(emitter.connect(settings));
    emitter.close();
    emitter.close();
    EXPECT_FALSE(emitter.healthy());
  }

  TEST(Open3DEmitter, CancellationBeforeConnectNeverOpensTheSelectedPort) {
    settings_t settings;
    settings.port = "COM65535";
    std::stop_source stop;
    stop.request_stop();
    emitter_t emitter;
    EXPECT_FALSE(emitter.connect(settings, stop.get_token()));
    EXPECT_FALSE(emitter.healthy());
    EXPECT_FALSE(emitter.signal(true));
    emitter.close();
  }
}  // namespace

#endif
