/**
 * @file tests/unit/test_virtual_desktop_launcher.cpp
 * @brief Monitor geometry; no real desktop windows are touched.
 */
#include "../tests_common.h"
#include "tools/virtual_desktop_launcher/launcher_policy.h"

TEST(NativeWindowPlacement, CentersInsideNegativeCoordinateWorkArea) {
  const desktop_launcher::rectangle_t work {-1920, 40, 0, 1080};
  const auto placed = desktop_launcher::place_in_work_area({100, 100, 900, 700}, work);
  ASSERT_TRUE(placed);
  EXPECT_EQ(*placed, (desktop_launcher::rectangle_t {-1360, 260, -560, 860}));
  EXPECT_TRUE(desktop_launcher::contains(work, *placed));
}

TEST(NativeWindowPlacement, OversizedWindowsFitPortraitDesktop) {
  const desktop_launcher::rectangle_t work {3840, -800, 4920, 1120};
  const auto placed = desktop_launcher::place_in_work_area({0, 0, 3840, 2160}, work);
  ASSERT_TRUE(placed);
  EXPECT_EQ(*placed, work);
  EXPECT_FALSE(desktop_launcher::place_in_work_area({0, 0, 0, 200}, work));
  EXPECT_FALSE(desktop_launcher::place_in_work_area(work, {0, 0, 0, 0}));
}

TEST(NativeWindowPlacement, RestorePlacementAccountsForTopAndLeftTaskbars) {
  const desktop_launcher::rectangle_t monitor {-1920, -400, 0, 680};
  const desktop_launcher::rectangle_t work {-1880, -350, 0, 680};
  const desktop_launcher::rectangle_t screen {-1700, -200, -900, 400};
  const auto workspace = desktop_launcher::screen_to_workspace(screen, monitor, work);
  EXPECT_EQ(workspace, (desktop_launcher::rectangle_t {-1740, -250, -940, 350}));
  EXPECT_EQ(desktop_launcher::workspace_to_screen(workspace, monitor, work), screen);
}
