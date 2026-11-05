#ifdef _WIN32

  #include "src/platform/windows/open3d_emitter.h"

  #include <gtest/gtest.h>
  #include <utility>

namespace {
  using namespace platf::open3d;

  constexpr std::string_view parameters = "parameters 20,6,500,7000,30,7500,10,128,0,1,1,0,0,0,32,1,8333,0";

  struct quiesce_fixture_t {
    using clock = detail::quiesce_io_t::clock;
    detail::parameters_t expected = detail::parse_parameters(
      "parameters 22,6,500,2000,30,7500,10,128,0,0,0,0,0,0,32,0,0,1,42,99"
    ).value();
    std::optional<detail::parameters_t> actual = expected;
    clock::time_point now {};
    std::vector<std::string> steps;
    clock::time_point finish_deadline {}, query_deadline {}, drain_deadline {};
    clock::duration finish_elapsed = std::chrono::milliseconds {100};
    clock::duration query_elapsed = std::chrono::milliseconds {200};
    bool cancelled = false;
    int cancel_after_step = -1;
    bool finish_ok = true;
    bool wait_ok = true;
    bool complete_drain = true;

    detail::quiesce_io_t io() {
      return {
        .now = [this]() { return now; },
        .cancelled = [this]() { return cancelled; },
        .finish_signal = [this](auto deadline) {
          steps.emplace_back("finish_signal");
          finish_deadline = deadline;
          now += finish_elapsed;
          cancelled |= cancel_after_step == 1;
          return finish_ok;
        },
        .read_parameters = [this](auto deadline) {
          steps.emplace_back("query0");
          query_deadline = deadline;
          now += query_elapsed;
          cancelled |= cancel_after_step == 2;
          return actual;
        },
        .wait_until = [this](auto deadline) {
          steps.emplace_back("finite_drain");
          drain_deadline = deadline;
          if (complete_drain) {
            now = deadline;
          }
          cancelled |= cancel_after_step == 3;
          return wait_ok;
        },
      };
    }
  };

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
    settings.signal_spacing_us = 40;
    settings.flip_eyes = true;
    settings.target_frametime_us = 6061;
    const auto command = detail::configuration_command(*previous, settings);
    ASSERT_TRUE(command);
    EXPECT_EQ(*command, "0,5,350,1800,40,7500,10,128,0,0,0,0,0,1,32,0,0,1,42,99");
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
    settings.target_frametime_us = -1;
    EXPECT_FALSE(detail::configuration_command(*previous, settings));
    emitter_t emitter;
    EXPECT_FALSE(emitter.healthy());
    EXPECT_FALSE(emitter.signal(true));
    EXPECT_FALSE(emitter.connect(settings));
    emitter.close();
    emitter.close();
    EXPECT_FALSE(emitter.healthy());
  }

  TEST(Open3DEmitter, AdmitsOnlyProtocolsWithIndependentlyScheduledFiniteCloseTokens) {
    settings_t settings;
    settings.port = "COM4";
    for (const int protocol : {2, 4, 5, 6}) {
      settings.protocol = protocol;
      EXPECT_TRUE(detail::valid_settings(settings)) << protocol;
    }
    for (const int protocol : {-1, 0, 1, 3, 7, 8}) {
      settings.protocol = protocol;
      EXPECT_FALSE(detail::valid_settings(settings)) << protocol;
    }
  }

  TEST(Open3DEmitter, RejectsTimerWrapAndZeroDelayRatherThanSilentlyChangingPulseTiming) {
    settings_t settings;
    settings.port = "COM4";
    settings.frame_delay_us = 32417;
    settings.frame_duration_us = 350;
    EXPECT_TRUE(detail::valid_settings(settings));
    settings.frame_delay_us = 32418;
    EXPECT_FALSE(detail::valid_settings(settings));  // Complete window would wrap at32768us.
    settings.frame_delay_us = 40000;
    EXPECT_FALSE(detail::valid_settings(settings));
    settings.frame_delay_us = 0;
    EXPECT_FALSE(detail::valid_settings(settings));
    settings.frame_delay_us = 1;
    EXPECT_FALSE(detail::valid_settings(settings));  // Not enough time to arm the comparator.
    settings.frame_delay_us = 100;
    settings.frame_duration_us = 32667;
    EXPECT_TRUE(detail::valid_settings(settings));
    settings.frame_duration_us = 32668;
    EXPECT_FALSE(detail::valid_settings(settings));
    settings.frame_duration_us = 40000;
    EXPECT_FALSE(detail::valid_settings(settings));
    settings.frame_duration_us = 0;
    EXPECT_FALSE(detail::valid_settings(settings));
  }

  TEST(Open3DEmitter, KeepsFiniteWindowInsideTheSelectedEyeInterval) {
    settings_t settings;
    settings.port = "COM4";
    settings.target_frametime_us = 2500;
    EXPECT_TRUE(detail::valid_settings(settings));
    settings.target_frametime_us = 2499;
    EXPECT_FALSE(detail::valid_settings(settings));
    settings.target_frametime_us = 0;
    EXPECT_TRUE(detail::valid_settings(settings));  // Unknown cadence still enforces Timer1 wrap.
    settings.target_frametime_us = -1;
    EXPECT_FALSE(detail::valid_settings(settings));
  }

  TEST(Open3DEmitter, RestrictsSignalSpacingToThePositiveTimer4Range) {
    settings_t settings;
    settings.port = "COM4";
    settings.signal_spacing_us = 1;
    EXPECT_FALSE(detail::valid_settings(settings));
    settings.signal_spacing_us = 29;
    EXPECT_FALSE(detail::valid_settings(settings));
    settings.signal_spacing_us = 30;
    EXPECT_TRUE(detail::valid_settings(settings));
    settings.signal_spacing_us = 803;
    EXPECT_TRUE(detail::valid_settings(settings));
    settings.signal_spacing_us = 804;
    EXPECT_FALSE(detail::valid_settings(settings));
    settings.signal_spacing_us = 40000;
    EXPECT_FALSE(detail::valid_settings(settings));
    settings.signal_spacing_us = 0;
    EXPECT_FALSE(detail::valid_settings(settings));
  }

  TEST(Open3DEmitter, DerivesProtocolSpecificLeadDurationAndCompleteTimer4TokenBudgets) {
    struct profile_t {
      int protocol;
      int lead;
      int minimum_duration;
      int maximum_spacing;
      int period_tail;
    };

    for (const auto &profile : {profile_t {2, 150, 330, 882, 241}, profile_t {4, 100, 670, 483, 560}, profile_t {5, 100, 350, 803, 320}, profile_t {6, 100, 350, 803, 320}}) {
      const auto timing = timing_limits(profile.protocol, 30);
      ASSERT_TRUE(timing);
      EXPECT_EQ(timing->min_delay_us, profile.lead);
      EXPECT_EQ(timing->min_duration_us, profile.minimum_duration);
      EXPECT_EQ(timing->min_period_tail_base_us, profile.period_tail);
      EXPECT_EQ(timing->max_signal_spacing_us, profile.maximum_spacing);
      settings_t settings;
      settings.port = "COM4";
      settings.protocol = profile.protocol;
      settings.frame_delay_us = profile.lead;
      settings.frame_duration_us = profile.minimum_duration;
      EXPECT_TRUE(detail::valid_settings(settings));
      --settings.frame_delay_us;
      EXPECT_FALSE(detail::valid_settings(settings));
      ++settings.frame_delay_us;
      --settings.frame_duration_us;
      EXPECT_FALSE(detail::valid_settings(settings));
      settings.frame_duration_us = 2000;
      settings.signal_spacing_us = profile.maximum_spacing;
      EXPECT_TRUE(detail::valid_settings(settings));
      ++settings.signal_spacing_us;
      EXPECT_FALSE(detail::valid_settings(settings));
    }
  }

  TEST(Open3DEmitter, RejectsIrCloseQueueTailsThatWouldCrossTheNextEyeBoundary) {
    settings_t settings;
    settings.port = "COM4";
    settings.frame_delay_us = 100;
    settings.target_frametime_us = 2350;
    EXPECT_TRUE(detail::valid_settings(settings));
    settings.target_frametime_us = 2349;
    EXPECT_FALSE(detail::valid_settings(settings));  // Delay+duration fits, but close-token tail does not.
  }

  TEST(Open3DEmitter, MonitorFeedbackBudgetRejectsFirmwareValidProfilesWithNoHostTimeRemaining) {
    settings_t settings;
    settings.port = "COM4";
    settings.target_frametime_us = 8333;
    settings.frame_duration_us = 7800;
    ASSERT_TRUE(detail::valid_settings(settings));
    EXPECT_FALSE(detail::host_feedback_budget_us(settings));
    settings.frame_duration_us = 7483;
    ASSERT_TRUE(detail::valid_settings(settings));
    EXPECT_FALSE(detail::host_feedback_budget_us(settings));  // Exactly zero host-feedback time.
    settings.frame_duration_us = 7482;
    EXPECT_EQ(detail::host_feedback_budget_us(settings), 1);
    settings.frame_duration_us = 2000;
    EXPECT_EQ(detail::host_feedback_budget_us(settings), 5483);
  }

  TEST(Open3DEmitter, MonitorFeedbackBudgetUsesEachProtocolCloseMarginAndTheActualEyeInterval) {
    settings_t settings;
    settings.port = "COM4";
    for (const auto [protocol, expected_budget] : std::array {
           std::pair {2, 5562}, std::pair {4, 5243}, std::pair {5, 5483}, std::pair {6, 5483}}) {
      SCOPED_TRACE(protocol);
      settings.protocol = protocol;
      settings.target_frametime_us = 8333;
      EXPECT_EQ(detail::host_feedback_budget_us(settings), expected_budget);
      settings.target_frametime_us = 16667;
      EXPECT_EQ(detail::host_feedback_budget_us(settings), expected_budget + 8334);
    }
  }

  TEST(Open3DEmitter, MonitorFeedbackBudgetRequiresAKnownCadenceAndAValidFirmwareProfile) {
    settings_t settings;
    settings.port = "COM4";
    ASSERT_TRUE(detail::valid_settings(settings));
    EXPECT_FALSE(detail::host_feedback_budget_us(settings));
    settings.target_frametime_us = 8333;
    settings.protocol = 1;
    EXPECT_FALSE(detail::host_feedback_budget_us(settings));
    settings.protocol = 6;
    settings.frame_delay_us = 0;
    EXPECT_FALSE(detail::host_feedback_budget_us(settings));
  }

  TEST(Open3DEmitter, QuiescesOldUnsupportedProfileWithoutChangingItsProtocolOrLiveTimings) {
    const auto previous = detail::parse_parameters("parameters 22,1,40000,40000,1000,7500,10,128,0,1,1,0,0,1,32,1,8333,0");
    ASSERT_TRUE(previous);
    const auto command = detail::quiesce_command(*previous);
    ASSERT_TRUE(command);
    EXPECT_EQ(*command, "0,1,40000,40000,1000,7500,10,128,0,0,0,0,0,1,32,0,8333,1");
    auto quiet = detail::parse_parameters("parameters 22" + command->substr(1));
    ASSERT_TRUE(quiet);
    EXPECT_TRUE(detail::quiesce_matches(*quiet, *previous));
    quiet->values[0] = 6;
    EXPECT_FALSE(detail::quiesce_matches(*quiet, *previous));
  }

  TEST(Open3DEmitter, SupportsSlowHostEyeIntervalsWithoutWrappingTheUnusedFirmwareFilter) {
    const auto previous = detail::parse_parameters(parameters);
    ASSERT_TRUE(previous);
    settings_t settings;
    settings.port = "COM4";
    settings.target_frametime_us = 66667;
    EXPECT_TRUE(detail::valid_settings(settings));
    const auto command = detail::configuration_command(*previous, settings);
    ASSERT_TRUE(command);
    const auto actual = detail::parse_parameters("parameters 20" + command->substr(1));
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->values[14], 0);
    EXPECT_EQ(actual->values[15], 0);
    EXPECT_TRUE(detail::configuration_matches(*actual, *previous, settings));
  }

  TEST(Open3DEmitter, ComputesTheSharedEyePeriodWithoutOverflowOrAnArtificialUint16Limit) {
    EXPECT_EQ(eye_period_us(330000, 2), 6061);
    EXPECT_EQ(eye_period_us(60000, 4), 66667);
    EXPECT_FALSE(eye_period_us(0, 1));
    EXPECT_FALSE(eye_period_us(-60000, 1));
    EXPECT_FALSE(eye_period_us(60000, 0));
    EXPECT_FALSE(eye_period_us(60000, 5));
    EXPECT_FALSE(eye_period_us(1, 4));
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

  TEST(Open3DEmitter, QuiescentBarrierFinishesTheEyeThenVerifiesReadbackBeforeFiniteDrain) {
    quiesce_fixture_t fixture;
    const auto original = fixture.expected;
    EXPECT_TRUE(detail::quiesce_with_io(fixture.expected, fixture.io()));
    EXPECT_EQ(fixture.steps, (std::vector<std::string> {"finish_signal", "query0", "finite_drain"}));
    EXPECT_EQ(fixture.finish_deadline, fixture.query_deadline);
    EXPECT_EQ(fixture.finish_deadline, quiesce_fixture_t::clock::time_point {} + std::chrono::milliseconds {1500});
    EXPECT_EQ(fixture.drain_deadline, quiesce_fixture_t::clock::time_point {} + fixture.finish_elapsed +
                                      fixture.query_elapsed + limits::startup_settle);
    EXPECT_EQ(fixture.expected.firmware_version, original.firmware_version);
    EXPECT_EQ(fixture.expected.values, original.values);  // Includes unknown newer-firmware fields.
  }

  TEST(Open3DEmitter, QuiescentBarrierNeverReadsAfterAnIncompleteEyeOrDrainsAfterMissingReadback) {
    quiesce_fixture_t fixture;
    fixture.finish_ok = false;
    EXPECT_FALSE(detail::quiesce_with_io(fixture.expected, fixture.io()));
    EXPECT_EQ(fixture.steps, (std::vector<std::string> {"finish_signal"}));
    fixture = {};
    fixture.actual.reset();
    EXPECT_FALSE(detail::quiesce_with_io(fixture.expected, fixture.io()));
    EXPECT_EQ(fixture.steps, (std::vector<std::string> {"finish_signal", "query0"}));
  }

  TEST(Open3DEmitter, QuiescentBarrierRejectsAnyFirmwareOrParameterChange) {
    for (const int changed_field : {-1, 0, 1, 2, 3, 14, 16, 18}) {
      SCOPED_TRACE(changed_field);
      quiesce_fixture_t fixture;
      if (changed_field < 0) {
        ++fixture.actual->firmware_version;
      } else {
        ++fixture.actual->values[changed_field];
      }
      EXPECT_FALSE(detail::quiesce_with_io(fixture.expected, fixture.io()));
      EXPECT_EQ(fixture.steps, (std::vector<std::string> {"finish_signal", "query0"}));
    }
  }

  TEST(Open3DEmitter, QuiescentBarrierHonorsCancellationAtEveryPhaseWithoutAuthorizingRestart) {
    for (const int phase : {0, 1, 2, 3}) {
      SCOPED_TRACE(phase);
      quiesce_fixture_t fixture;
      fixture.cancelled = phase == 0;
      fixture.cancel_after_step = phase;
      EXPECT_FALSE(detail::quiesce_with_io(fixture.expected, fixture.io()));
      EXPECT_EQ(fixture.steps.size(), static_cast<std::size_t>(phase));
    }
  }

  TEST(Open3DEmitter, QuiescentBarrierUsesOneDeadlineForPendingWriteAndFirmwareReceipt) {
    quiesce_fixture_t fixture;
    fixture.finish_elapsed = std::chrono::milliseconds {1500};
    EXPECT_FALSE(detail::quiesce_with_io(fixture.expected, fixture.io()));
    EXPECT_EQ(fixture.steps, (std::vector<std::string> {"finish_signal"}));
    fixture = {};
    fixture.query_elapsed = std::chrono::milliseconds {1400};
    EXPECT_FALSE(detail::quiesce_with_io(fixture.expected, fixture.io()));
    EXPECT_EQ(fixture.steps, (std::vector<std::string> {"finish_signal", "query0"}));
    fixture = {};
    fixture.query_elapsed = std::chrono::milliseconds {1399};
    EXPECT_TRUE(detail::quiesce_with_io(fixture.expected, fixture.io()));
    EXPECT_EQ(fixture.finish_deadline, fixture.query_deadline);
  }

  TEST(Open3DEmitter, QuiescentBarrierRequiresTheWholeFiniteDrainEvenIfWaitReportsSuccess) {
    quiesce_fixture_t fixture;
    fixture.complete_drain = false;
    EXPECT_FALSE(detail::quiesce_with_io(fixture.expected, fixture.io()));
    fixture = {};
    fixture.wait_ok = false;
    EXPECT_FALSE(detail::quiesce_with_io(fixture.expected, fixture.io()));
  }

  TEST(Open3DEmitter, QuiescentBarrierCannotCertifyAnAutonomousOrUnsupportedPriorProfile) {
    for (const auto [field, value] : std::array {std::pair {0, 3}, std::pair {14, 1}, std::pair {16, 0}}) {
      quiesce_fixture_t fixture;
      fixture.expected.values[field] = value;
      EXPECT_FALSE(detail::quiesce_with_io(fixture.expected, fixture.io()));
      EXPECT_TRUE(fixture.steps.empty());
    }
    emitter_t emitter;
    EXPECT_FALSE(emitter.quiesce());  // An unopened emitter must not access any COM device.
    EXPECT_FALSE(emitter.healthy());
    EXPECT_FALSE(emitter.signal(true));
  }
}  // namespace

#endif
