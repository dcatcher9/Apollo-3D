/**
 * @file tests/unit/test_foreground_window_region.cpp
 * @brief Pure foreground-window admission, continuity, and monitor-mapping tests.
 */

#include "src/platform/windows/foreground_window_region.h"

#include <chrono>
#include <gtest/gtest.h>
#include <windows.h>

namespace {
  using namespace std::chrono_literals;
  namespace foreground = platf::foreground_window;

  foreground::detail::raw_observation_t available_window() {
    return {
      .window = 0x100u,
      .window_after = 0x100u,
      .shell_window = 0x900u,
      .desktop_window = 0x901u,
      .process_id = 42u,
      .process_id_after = 42u,
      .own_process_id = 7u,
      .monitor = 0x200u,
      .client_screen_rect = {-1800, 100, -200, 1000},
      .frame_screen_rect = {-1810, 70, -190, 1010},
      .dpi_aware = true,
      .is_window = true,
      .is_window_after = true,
      .visible = true,
      .cloak_query_succeeded = true,
      .process_query_succeeded = true,
      .process_query_after_succeeded = true,
      .client_rect_succeeded = true,
      .frame_rect_succeeded = true,
      .style_query_succeeded = true,
      .class_query_succeeded = true,
    };
  }

  foreground::snapshot_t available_snapshot() {
    const auto now = std::chrono::steady_clock::time_point {10s};
    return {
      .status = foreground::status_e::ok,
      .generation = 5u,
      .window = 0x100u,
      .process_id = 42u,
      .monitor = 0x200u,
      .client_screen_rect = {-1800, 100, -200, 1000},
      .frame_screen_rect = {-1810, 70, -190, 1010},
      .observed_at = now,
      .geometry_valid_since = now - 1s,
    };
  }

  TEST(ForegroundWindowPolicy, AcceptsOnlyCompleteStableOrdinaryWindow) {
    const auto observed_at = std::chrono::steady_clock::time_point {10s};
    const auto result = foreground::detail::classify(
      available_window(),
      observed_at
    );
    EXPECT_EQ(result.status, foreground::status_e::ok);
    EXPECT_EQ(result.window, 0x100u);
    EXPECT_EQ(result.process_id, 42u);
    EXPECT_EQ(result.monitor, 0x200u);
    EXPECT_EQ(result.client_screen_rect, (foreground::rect_t {-1800, 100, -200, 1000}));
    EXPECT_EQ(result.frame_screen_rect, (foreground::rect_t {-1810, 70, -190, 1010}));
    EXPECT_EQ(result.observed_at, observed_at);
  }

  TEST(ForegroundWindowPolicy, RejectsNoForegroundAndShellSurfaces) {
    const auto now = std::chrono::steady_clock::time_point {10s};
    auto raw = available_window();
    raw.window = 0;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::no_foreground
    );

    raw = available_window();
    raw.shell_window = raw.window;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::shell_surface
    );
    raw = available_window();
    raw.desktop_window = raw.window;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::shell_surface
    );
    raw = available_window();
    raw.shell_class = true;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::shell_surface
    );
  }

  TEST(ForegroundWindowPolicy, RejectsUnavailableOrTransientWindows) {
    const auto now = std::chrono::steady_clock::time_point {10s};
    auto raw = available_window();
    raw.dpi_aware = false;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::dpi_unavailable
    );
    raw = available_window();
    raw.is_window = false;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::invalid_window
    );
    raw = available_window();
    raw.process_query_succeeded = false;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::invalid_window
    );
    raw = available_window();
    raw.own_process_id = raw.process_id;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::self_process
    );
    raw = available_window();
    raw.class_query_succeeded = false;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::invalid_window
    );
    raw = available_window();
    raw.visible = false;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::not_visible
    );
    raw = available_window();
    raw.minimized = true;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::minimized
    );
    raw = available_window();
    raw.cloaked = true;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::cloaked
    );
  }

  TEST(ForegroundWindowPolicy, RejectsToolNoActivateAndLayeredStyles) {
    const auto now = std::chrono::steady_clock::time_point {10s};
    constexpr std::uint64_t excluded_styles[] {
      WS_EX_TOOLWINDOW,
      WS_EX_NOACTIVATE,
      WS_EX_LAYERED,
      WS_EX_TOOLWINDOW | WS_EX_LAYERED,
    };
    for (const auto style : excluded_styles) {
      auto raw = available_window();
      raw.extended_style = style;
      EXPECT_EQ(
        foreground::detail::classify(raw, now).status,
        foreground::status_e::excluded_style
      ) << style;
    }
  }

  TEST(ForegroundWindowPolicy, RequiresDwmFrameClientAndMonitorGeometry) {
    const auto now = std::chrono::steady_clock::time_point {10s};
    auto raw = available_window();
    raw.cloak_query_succeeded = false;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::geometry_unavailable
    );
    raw = available_window();
    raw.client_rect_succeeded = false;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::geometry_unavailable
    );
    raw = available_window();
    raw.frame_rect_succeeded = false;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::geometry_unavailable
    );
    raw = available_window();
    raw.monitor = 0;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::geometry_unavailable
    );
    raw = available_window();
    raw.frame_screen_rect = {-1700, 200, -300, 900};
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::geometry_unavailable
    );
  }

  TEST(ForegroundWindowPolicy, RechecksForegroundHandleAndProcessAfterGeometry) {
    const auto now = std::chrono::steady_clock::time_point {10s};
    auto raw = available_window();
    raw.window_after += 1u;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::foreground_changed
    );
    raw = available_window();
    raw.process_id_after += 1u;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::foreground_changed
    );
    raw = available_window();
    raw.is_window_after = false;
    EXPECT_EQ(
      foreground::detail::classify(raw, now).status,
      foreground::status_e::foreground_changed
    );
  }

  TEST(ForegroundWindowContinuity, PreservesExactRunAndRearmsEveryDiscontinuity) {
    foreground::continuity_tracker_t tracker;
    const auto first_at = std::chrono::steady_clock::time_point {10s};
    auto observation = foreground::detail::classify(available_window(), first_at);
    const auto first = tracker.update(observation);
    ASSERT_EQ(first.status, foreground::status_e::ok);
    ASSERT_NE(first.generation, 0u);
    EXPECT_EQ(first.geometry_valid_since, first_at);

    observation.observed_at += 50ms;
    const auto heartbeat = tracker.update(observation);
    EXPECT_EQ(heartbeat.generation, first.generation);
    EXPECT_EQ(heartbeat.geometry_valid_since, first.geometry_valid_since);
    EXPECT_EQ(heartbeat.observed_at, first_at + 50ms);

    observation.client_screen_rect.left += 1;
    observation.client_screen_rect.right += 1;
    observation.frame_screen_rect.left += 1;
    observation.frame_screen_rect.right += 1;
    observation.observed_at += 50ms;
    const auto moved = tracker.update(observation);
    EXPECT_GT(moved.generation, first.generation);
    EXPECT_EQ(moved.geometry_valid_since, first_at + 100ms);

    const foreground::observation_t unavailable {
      .status = foreground::status_e::no_foreground,
      .observed_at = first_at + 150ms,
    };
    const auto invalid = tracker.update(unavailable);
    EXPECT_EQ(invalid.generation, 0u);
    EXPECT_EQ(invalid.geometry_valid_since, std::chrono::steady_clock::time_point {});

    observation.observed_at = first_at + 200ms;
    const auto reacquired = tracker.update(observation);
    EXPECT_GT(reacquired.generation, moved.generation);
    EXPECT_EQ(reacquired.geometry_valid_since, first_at + 200ms);

    tracker.reset();
    observation.observed_at = first_at + 250ms;
    const auto after_reset = tracker.update(observation);
    EXPECT_GT(after_reset.generation, reacquired.generation);
    EXPECT_EQ(after_reset.geometry_valid_since, first_at + 250ms);
  }

  TEST(ForegroundWindowContinuity, ChangedIdentityMonitorOrFrameStartsNewGeneration) {
    const auto start = std::chrono::steady_clock::time_point {10s};
    constexpr auto step = 10ms;
    const auto expect_change = [&](auto mutate) {
      foreground::continuity_tracker_t tracker;
      auto observation = foreground::detail::classify(available_window(), start);
      const auto first = tracker.update(observation);
      mutate(observation);
      observation.observed_at += step;
      const auto changed = tracker.update(observation);
      EXPECT_GT(changed.generation, first.generation);
      EXPECT_EQ(changed.geometry_valid_since, start + step);
    };
    expect_change([](auto &value) {
      ++value.window;
    });
    expect_change([](auto &value) {
      ++value.process_id;
    });
    expect_change([](auto &value) {
      ++value.monitor;
    });
    expect_change([](auto &value) {
      --value.frame_screen_rect.left;
    });
  }

  TEST(ForegroundWindowCausality, RequiresFreshGeometryBeforeDesktopPresentation) {
    foreground::continuity_tracker_t tracker;
    const auto first_at = std::chrono::steady_clock::time_point {10s};
    auto observation = foreground::detail::classify(available_window(), first_at);
    const auto first = tracker.update(observation);
    EXPECT_FALSE(foreground::usable_for_content(first, first_at - 1ms, first_at + 1ms));
    EXPECT_FALSE(foreground::usable_for_content(first, std::nullopt, first_at + 1ms));

    observation.observed_at = first_at + 100ms;
    const auto later = tracker.update(observation);
    ASSERT_EQ(later.geometry_valid_since, first_at);
    EXPECT_TRUE(foreground::usable_for_content(
      later,
      first_at + 50ms,
      first_at + 150ms
    ));
    EXPECT_FALSE(foreground::usable_for_content(
      later,
      first_at + 50ms,
      first_at + 351ms
    ));
    EXPECT_FALSE(foreground::usable_for_content(
      later,
      first_at + 200ms,
      first_at + 150ms
    ));
  }

  TEST(ForegroundWindowMapping, MapsNegativeRawDesktopCoordinatesWithoutClipping) {
    const auto mapped = foreground::map_to_capture(
      available_snapshot(),
      {
        .screen_rect = {-1920, 0, 0, 1080},
        .width = 1920,
        .height = 1080,
        .monitor = 0x200u,
      }
    );
    ASSERT_TRUE(mapped);
    EXPECT_EQ(mapped.route, foreground::route_e::roi);
    EXPECT_EQ(mapped.capture_pixels, (foreground::rect_t {120, 100, 1720, 1000}));
  }

  TEST(ForegroundWindowMapping, ExactClientExtentSelectsCanonicalFullCapture) {
    auto snapshot = available_snapshot();
    snapshot.client_screen_rect = {-1920, 0, 0, 1080};
    snapshot.frame_screen_rect = snapshot.client_screen_rect;
    const auto mapped = foreground::map_to_capture(
      snapshot,
      {
        .screen_rect = {-1920, 0, 0, 1080},
        .width = 1920,
        .height = 1080,
        .monitor = 0x200u,
      }
    );
    ASSERT_TRUE(mapped);
    EXPECT_EQ(mapped.route, foreground::route_e::full_capture);
    EXPECT_EQ(mapped.capture_pixels, (foreground::rect_t {0, 0, 1920, 1080}));
  }

  TEST(ForegroundWindowMapping, RejectsOtherMonitorAndSpanningOrOffscreenRects) {
    const foreground::capture_target_t target {
      .screen_rect = {0, 0, 1920, 1080},
      .width = 1920,
      .height = 1080,
      .monitor = 0x300u,
    };
    auto snapshot = available_snapshot();
    EXPECT_EQ(
      foreground::map_to_capture(snapshot, target).status,
      foreground::mapping_status_e::monitor_mismatch
    );

    snapshot.monitor = target.monitor;
    snapshot.client_screen_rect = {-100, 100, 1000, 900};
    snapshot.frame_screen_rect = {-110, 70, 1010, 910};
    EXPECT_EQ(
      foreground::map_to_capture(snapshot, target).status,
      foreground::mapping_status_e::partially_outside_capture
    );

    snapshot.client_screen_rect = {-1800, 100, -200, 900};
    snapshot.frame_screen_rect = {-1810, 70, -190, 910};
    EXPECT_EQ(
      foreground::map_to_capture(snapshot, target).status,
      foreground::mapping_status_e::outside_capture
    );
  }

  TEST(ForegroundWindowMapping, RejectsInvalidTargetRotationAndInvalidObservation) {
    auto target = foreground::capture_target_t {
      .screen_rect = {-1920, 0, 0, 1080},
      .width = 1919,
      .height = 1080,
      .monitor = 0x200u,
    };
    auto snapshot = available_snapshot();
    EXPECT_EQ(
      foreground::map_to_capture(snapshot, target).status,
      foreground::mapping_status_e::invalid_capture_target
    );
    target.width = 1920;
    target.identity_orientation = false;
    EXPECT_EQ(
      foreground::map_to_capture(snapshot, target).status,
      foreground::mapping_status_e::unsupported_orientation
    );
    target.identity_orientation = true;
    snapshot.status = foreground::status_e::minimized;
    EXPECT_EQ(
      foreground::map_to_capture(snapshot, target).status,
      foreground::mapping_status_e::invalid_observation
    );
  }
}  // namespace
