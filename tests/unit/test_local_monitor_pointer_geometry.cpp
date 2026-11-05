#ifdef _WIN32

  #include "src/platform/windows/local_monitor_pointer_geometry.h"

  #include <array>
  #include <gtest/gtest.h>
  #include <limits>

namespace {
  using platf::dxgi::detail::nearest_interactive_pointer_point;

  void expect_point(std::optional<POINT> actual, LONG x, LONG y) {
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->x, x);
    EXPECT_EQ(actual->y, y);
  }

  TEST(LocalMonitorPointerGeometry, UsesNearestInteractiveMonitorRatherThanAlwaysReturningToSource) {
    const RECT sink {8000, 0, 9920, 1080};
    const std::array<RECT, 3> monitors {RECT {0, 0, 1920, 1080}, RECT {4000, 0, 8000, 2160}, sink};
    expect_point(nearest_interactive_pointer_point({8010, 500}, sink, monitors), 7999, 500);
    EXPECT_FALSE(nearest_interactive_pointer_point({7900, 500}, sink, monitors));
  }

  TEST(LocalMonitorPointerGeometry, ClampsToTheMonitorUnionRatherThanItsBoundingRectangle) {
    const RECT sink {0, 0, 1920, 1080};
    const std::array<RECT, 3> monitors {RECT {-1920, -1080, 0, 0}, RECT {1920, 700, 3840, 1780}, sink};
    expect_point(nearest_interactive_pointer_point({1900, 100}, sink, monitors), 1920, 700);
    expect_point(nearest_interactive_pointer_point({10, 10}, sink, monitors), -1, -1);
  }

  TEST(LocalMonitorPointerGeometry, HonorsHalfOpenOuterEdgesAndNegativeCoordinates) {
    const RECT sink {-1920, -1080, 0, 0};
    const std::array<RECT, 3> monitors {RECT {0, -1080, 1920, 0}, RECT {-1920, 0, 0, 1080}, sink};
    expect_point(nearest_interactive_pointer_point({-1, -500}, sink, monitors), 0, -500);
    expect_point(nearest_interactive_pointer_point({-500, -1}, sink, monitors), -500, 0);
    EXPECT_FALSE(nearest_interactive_pointer_point({0, -1}, sink, monitors));
  }

  TEST(LocalMonitorPointerGeometry, ExcludesSinkPixelsDuringOverlappingTopologyObservations) {
    const RECT sink {100, 100, 200, 200};
    const std::array<RECT, 2> monitors {RECT {0, 0, 300, 300}, sink};
    expect_point(nearest_interactive_pointer_point({199, 150}, sink, monitors), 200, 150);
    const std::array<RECT, 2> unusable {sink, RECT {1, 1, 1, 100}};
    EXPECT_FALSE(nearest_interactive_pointer_point({150, 150}, sink, unusable));
  }

  TEST(LocalMonitorPointerGeometry, AvoidsDistanceOverflowAtDesktopCoordinateLimits) {
    const auto lowest = std::numeric_limits<LONG>::min();
    const auto highest = std::numeric_limits<LONG>::max();
    const RECT sink {highest - 100, highest - 100, highest, highest};
    const std::array<RECT, 3> monitors {RECT {lowest, lowest, lowest + 100, lowest + 100}, RECT {highest - 200, highest - 100, highest - 100, highest}, sink};
    expect_point(nearest_interactive_pointer_point({highest - 1, highest - 1}, sink, monitors), highest - 101, highest - 1);
  }
}  // namespace

#endif
