#include <gtest/gtest.h>
#include <tools/virtual_desktop_launcher/native_window_policy.h>

namespace {
  using desktop_launcher::native_foreground_t;
  using desktop_launcher::native_input_action_e;
  using desktop_launcher::native_input_t;
  using desktop_launcher::native_window_intent_t;

  native_input_t shell_input(std::uint64_t tick, native_input_action_e action = native_input_action_e::launch) {
    return {
      .tick = tick,
      .source_root = 100,
      .action = action,
      .trusted_input = true,
      .target_available = true,
      .inside_target = true,
      .shell_surface = true,
    };
  }

  native_foreground_t foreground(const native_window_intent_t &intent, std::uint64_t event_tick = 101) {
    return {
      .event_tick = event_tick,
      .now_tick = event_tick + 1,
      .expected_generation = intent.generation(),
      .root_window = 200,
      .eligible_application = true,
      .is_current_foreground = true,
      .target_available = true,
      .input_idle_unchanged = true,
    };
  }
}  // namespace

TEST(NativeWindowRoutingTest, TrustedShellLaunchAdmitsOneForegroundIncludingReusedWindow) {
  native_window_intent_t intent;
  intent.observe_input(shell_input(100));
  ASSERT_TRUE(intent.armed());
  EXPECT_EQ(intent.source_root(), 100u);
  const auto candidate = foreground(intent);
  EXPECT_TRUE(intent.claim(candidate));
  EXPECT_FALSE(intent.armed());
  EXPECT_FALSE(intent.claim(candidate));
}

TEST(NativeWindowRoutingTest, MotionAndShellNavigationCannotAuthorizeWindowMovement) {
  native_window_intent_t intent;
  intent.observe_input(shell_input(100, native_input_action_e::motion));
  EXPECT_FALSE(intent.armed());
  EXPECT_FALSE(intent.claim(foreground(intent)));
  intent.observe_input(shell_input(102, native_input_action_e::navigate));
  EXPECT_FALSE(intent.armed());
  EXPECT_FALSE(intent.claim(foreground(intent, 103)));
}

TEST(NativeWindowRoutingTest, PhysicalShellLaunchOnVirtualMonitorIsAdmitted) {
  native_window_intent_t intent;
  // The adapter grants the same trust to ordinary physical input and our remote injection.
  // Monitor/source attribution decides where the resulting window belongs.
  const auto physical_launch = shell_input(100);
  intent.observe_input(physical_launch);
  EXPECT_TRUE(intent.claim(foreground(intent)));
}

TEST(NativeWindowRoutingTest, PhysicalInputOnMainMonitorCancelsPendingLaunch) {
  for (const auto action : {native_input_action_e::motion, native_input_action_e::launch}) {
    native_window_intent_t intent;
    intent.observe_input(shell_input(100));
    const auto queued_foreground = foreground(intent, 103);
    auto physical_input = shell_input(102, action);
    physical_input.inside_target = false;
    intent.observe_input(physical_input);
    EXPECT_FALSE(intent.armed());
    EXPECT_FALSE(intent.claim(queued_foreground));
  }
}

TEST(NativeWindowRoutingTest, ForeignInjectedInputCancelsPendingLaunch) {
  native_window_intent_t intent;
  intent.observe_input(shell_input(100));
  const auto queued_foreground = foreground(intent, 103);
  auto foreign_motion = shell_input(102, native_input_action_e::motion);
  foreign_motion.trusted_input = false;
  intent.observe_input(foreign_motion);
  EXPECT_FALSE(intent.armed());
  EXPECT_FALSE(intent.claim(queued_foreground));
}

TEST(NativeWindowRoutingTest, OutsideInputTargetLossAndNonshellInteractionCancel) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    native_window_intent_t intent;
    intent.observe_input(shell_input(100));
    auto next = shell_input(102);
    if (scenario == 0) {
      next.inside_target = false;
    }
    if (scenario == 1) {
      next.target_available = false;
    }
    if (scenario == 2) {
      next.shell_surface = false;
    }
    intent.observe_input(next);
    EXPECT_FALSE(intent.armed());
    EXPECT_FALSE(intent.claim(foreground(intent, 103)));
  }
}

TEST(NativeWindowRoutingTest, BackgroundShowSourceShellAndOldForegroundNeverQualify) {
  native_window_intent_t intent;
  intent.observe_input(shell_input(100));
  auto candidate = foreground(intent);
  candidate.is_current_foreground = false;
  EXPECT_FALSE(intent.claim(candidate));
  candidate = foreground(intent);
  candidate.root_window = 100;
  EXPECT_FALSE(intent.claim(candidate));
  candidate = foreground(intent);
  candidate.shell_surface = true;
  EXPECT_FALSE(intent.claim(candidate));
  candidate = foreground(intent, 100);
  EXPECT_FALSE(intent.claim(candidate));
  EXPECT_TRUE(intent.claim(foreground(intent, 102)));
}

TEST(NativeWindowRoutingTest, StaleGenerationCannotClaimOrCancelNewerLaunch) {
  native_window_intent_t intent;
  intent.observe_input(shell_input(100));
  const auto stale = foreground(intent, 104);
  intent.observe_input(shell_input(102));
  EXPECT_FALSE(intent.claim(stale));
  EXPECT_TRUE(intent.armed());
  EXPECT_TRUE(intent.claim(foreground(intent, 104)));
}

TEST(NativeWindowRoutingTest, IntentExpiresAndMotionCannotKeepItAlive) {
  native_window_intent_t intent;
  intent.observe_input(shell_input(100));
  intent.observe_input(shell_input(5100, native_input_action_e::motion));
  EXPECT_FALSE(intent.claim(foreground(intent, 5101)));
  EXPECT_FALSE(intent.armed());
}

TEST(NativeWindowRoutingTest, InterveningInputOrMissingTargetAtPlacementCancels) {
  for (bool target_present : {false, true}) {
    native_window_intent_t intent;
    intent.observe_input(shell_input(100));
    auto candidate = foreground(intent);
    candidate.target_available = target_present;
    candidate.input_idle_unchanged = !target_present;
    EXPECT_FALSE(intent.claim(candidate));
    EXPECT_FALSE(intent.armed());
  }
}

TEST(NativeWindowRoutingTest, KeyboardShellHopRetainsExplicitVirtualNavigationOrigin) {
  native_window_intent_t intent;
  // Qualified keyboard navigation starts on the virtual desktop.
  intent.observe_input(shell_input(100, native_input_action_e::navigate));
  auto enter = shell_input(300);
  enter.source_root = 110;  // Start/Search may use a different native shell root.
  enter.inside_target = false;
  enter.inherits_virtual_shell_origin = true;
  intent.observe_input(enter);
  ASSERT_TRUE(intent.armed());
  EXPECT_EQ(intent.source_root(), 110u);
  EXPECT_TRUE(intent.claim(foreground(intent, 301)));
}

TEST(NativeWindowRoutingTest, KeyboardOriginCannotBeInventedFromMotionOrExpiredNavigation) {
  for (const auto initial : {native_input_action_e::motion, native_input_action_e::navigate}) {
    native_window_intent_t intent;
    intent.observe_input(shell_input(100, initial));
    auto enter = shell_input(initial == native_input_action_e::motion ? 101 : 5101);
    enter.inside_target = false;
    enter.inherits_virtual_shell_origin = true;
    intent.observe_input(enter);
    EXPECT_FALSE(intent.armed());
  }
}

TEST(NativeWindowRoutingTest, OutOfOrderInputCannotResurrectCancelledGesture) {
  native_window_intent_t intent;
  auto foreign = shell_input(200, native_input_action_e::motion);
  foreign.trusted_input = false;
  intent.observe_input(foreign);
  intent.observe_input(shell_input(100));
  EXPECT_FALSE(intent.armed());
  EXPECT_FALSE(intent.claim(foreground(intent, 201)));
}

TEST(NativeWindowRoutingTest, EqualTickTransitionRequiresSerialAndPreInputForegroundEvidence) {
  native_window_intent_t intent;
  auto launch = shell_input(100);
  launch.observation_serial = 10;
  launch.foreground_root_before_input = 300;
  intent.observe_input(launch);
  auto candidate = foreground(intent, 100);
  candidate.observation_serial = 11;
  EXPECT_TRUE(intent.claim(candidate));
}

TEST(NativeWindowRoutingTest, DelayedEqualTickEventForPreviouslyForegroundAppIsRejected) {
  native_window_intent_t intent;
  auto launch = shell_input(100);
  launch.observation_serial = 10;
  launch.foreground_root_before_input = 200;
  intent.observe_input(launch);
  auto candidate = foreground(intent, 100);
  candidate.observation_serial = 11;
  EXPECT_FALSE(intent.claim(candidate));
  EXPECT_TRUE(intent.armed());
}

TEST(NativeWindowRoutingTest, SerialCannotAuthorizeOlderEventOrMissingTransitionEvidence) {
  for (int scenario = 0; scenario < 4; ++scenario) {
    native_window_intent_t intent;
    auto launch = shell_input(100);
    launch.observation_serial = scenario == 0 ? 0 : 10;
    launch.foreground_root_before_input = scenario == 1 ? 0 : 300;
    intent.observe_input(launch);
    auto candidate = foreground(intent, scenario == 2 ? 99 : 100);
    candidate.now_tick = 101;
    candidate.observation_serial = scenario == 3 ? 10 : 11;
    EXPECT_FALSE(intent.claim(candidate));
  }
}
