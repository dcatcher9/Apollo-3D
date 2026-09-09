/**
 * @file tests/unit/test_virtual_desktop_launcher.cpp
 * @brief New-window ownership and monitor geometry; no real desktop windows are touched.
 */
#include "../tests_common.h"
#include "tools/virtual_desktop_launcher/launcher_policy.h"

namespace {
  desktop_launcher::window_evidence_t eligible_window() {
    return {
      .belongs_to_launch_job = true,
      .visible = true,
      .top_level = true,
      .application_window = true,
      .exact_target_available = true,
      .default_input_desktop = true,
      .process_created = 150,
      .launch_started = 100,
    };
  }
}  // namespace

TEST(VirtualDesktopLauncher, AdmitsOnlyNewWindowsFromExplicitLaunchJob) {
  auto evidence = eligible_window();
  EXPECT_TRUE(desktop_launcher::may_place_window(evidence));
  evidence.belongs_to_launch_job = false;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
  evidence.belongs_to_launch_job = true;
  evidence.existed_before_launch = true;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
  evidence.existed_before_launch = false;
  evidence.process_created = 99;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
}

TEST(VirtualDesktopLauncher, ExistingInstanceAndBrokerWindowsAreNeverInferredFromFocus) {
  auto evidence = eligible_window();
  // A fresh HWND created by an existing browser or broker has no ownership in the new job.
  evidence.belongs_to_launch_job = false;
  evidence.process_created = 10;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
}

TEST(VirtualDesktopLauncher, StopsAtMonitorAbsenceAndSecureDesktop) {
  auto evidence = eligible_window();
  evidence.exact_target_available = false;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
  evidence.exact_target_available = true;
  evidence.default_input_desktop = false;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
}

TEST(VirtualDesktopLauncher, NeverRepositionsHandledOrManuallyDraggedWindows) {
  auto evidence = eligible_window();
  evidence.already_handled = true;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
  evidence.already_handled = false;
  evidence.interactive_move = true;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
}

TEST(VirtualDesktopLauncher, ExcludesHiddenMinimizedCloakedAndUtilityWindows) {
  const auto base = eligible_window();
  auto evidence = base;
  evidence.visible = false;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
  evidence = base;
  evidence.minimized = true;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
  evidence = base;
  evidence.cloaked = true;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
  evidence = base;
  evidence.application_window = false;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
  evidence = base;
  evidence.top_level = false;
  EXPECT_FALSE(desktop_launcher::may_place_window(evidence));
}

TEST(VirtualDesktopLauncher, CentersInsideNegativeCoordinateWorkArea) {
  const desktop_launcher::rectangle_t work {-1920, 40, 0, 1080};
  const auto placed = desktop_launcher::place_in_work_area({100, 100, 900, 700}, work);
  ASSERT_TRUE(placed);
  EXPECT_EQ(*placed, (desktop_launcher::rectangle_t {-1360, 260, -560, 860}));
  EXPECT_TRUE(desktop_launcher::contains(work, *placed));
}

TEST(VirtualDesktopLauncher, OversizedWindowsFitPortraitDesktop) {
  const desktop_launcher::rectangle_t work {3840, -800, 4920, 1120};
  const auto placed = desktop_launcher::place_in_work_area({0, 0, 3840, 2160}, work);
  ASSERT_TRUE(placed);
  EXPECT_EQ(*placed, work);
  EXPECT_FALSE(desktop_launcher::place_in_work_area({0, 0, 0, 200}, work));
  EXPECT_FALSE(desktop_launcher::place_in_work_area(work, {0, 0, 0, 0}));
}

TEST(VirtualDesktopLauncher, RestorePlacementAccountsForTopAndLeftTaskbars) {
  const desktop_launcher::rectangle_t monitor {-1920, -400, 0, 680};
  const desktop_launcher::rectangle_t work {-1880, -350, 0, 680};
  const desktop_launcher::rectangle_t screen {-1700, -200, -900, 400};
  const auto workspace = desktop_launcher::screen_to_workspace(screen, monitor, work);
  EXPECT_EQ(workspace, (desktop_launcher::rectangle_t {-1740, -250, -940, 350}));
  EXPECT_EQ(desktop_launcher::workspace_to_screen(workspace, monitor, work), screen);
}
