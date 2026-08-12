/**
 * @file tests/unit/test_video_dom_client.cpp
 * @brief Host-side video-DOM protocol and coordinate mapping tests.
 */

#include "src/platform/windows/video_dom_client.h"

#include <chrono>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <string_view>

namespace {
  using namespace std::chrono_literals;
  namespace video_dom = platf::video_dom;

  constexpr std::string_view ok_record =
    "SUNSHINE_VIDEO_DOM_V1\t1\tok\t4660\t991\t-200\t-19121\t29\t203\t1802\t1201";
  constexpr std::string_view ok_fullscreen_record =
    "SUNSHINE_VIDEO_DOM_V1\t2\tok-fullscreen\t4660\t991\t-200\t-19121\t0\t0\t1920\t1080";

  TEST(VideoDomClientProtocol, ParsesExactOkRecord) {
    const auto parsed = video_dom::detail::parse_protocol_record(ok_record);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed->status, video_dom::status_e::ok);
    EXPECT_EQ(parsed->sequence, 1u);
    EXPECT_EQ(parsed->window, 4660u);
    EXPECT_EQ(parsed->process_id, 991u);
    EXPECT_EQ(parsed->document_id, -200);
    EXPECT_EQ(parsed->video_id, -19121);
    EXPECT_EQ(parsed->screen_rect, (video_dom::rect_t {29, 203, 1802, 1201}));
  }

  TEST(VideoDomClientProtocol, ParsesFullscreenProvenanceWithoutChangingFieldCount) {
    const auto parsed = video_dom::detail::parse_protocol_record(ok_fullscreen_record);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed->status, video_dom::status_e::ok_fullscreen);
    EXPECT_EQ(parsed->sequence, 2u);
    EXPECT_EQ(parsed->window, 4660u);
    EXPECT_EQ(parsed->process_id, 991u);
    EXPECT_EQ(parsed->document_id, -200);
    EXPECT_EQ(parsed->video_id, -19121);
    EXPECT_EQ(parsed->screen_rect, (video_dom::rect_t {0, 0, 1920, 1080}));
    EXPECT_STREQ(
      video_dom::status_name(video_dom::status_e::ok_fullscreen),
      "ok-fullscreen"
    );
  }

  TEST(VideoDomClientProtocol, AcceptsEveryDeclaredUnavailableStatusOnlyWithZeroPayload) {
    constexpr std::string_view statuses[] {
      "no-foreground",
      "unsupported",
      "unavailable",
      "accessibility",
      "warming",
      "incomplete",
      "changed",
      "no-video",
      "ambiguous",
    };
    std::uint64_t sequence = 1;
    for (const auto status : statuses) {
      const auto line =
        "SUNSHINE_VIDEO_DOM_V1\t" + std::to_string(sequence++) + "\t" +
        std::string(status) + "\t0\t0\t0\t0\t0\t0\t0\t0";
      const auto parsed = video_dom::detail::parse_protocol_record(line);
      ASSERT_TRUE(parsed) << status;
      EXPECT_FALSE(video_dom::carries_video_geometry(parsed->status));
    }
  }

  TEST(VideoDomClientProtocol, RejectsUnknownLooseAndContradictoryRecords) {
    EXPECT_FALSE(video_dom::detail::parse_protocol_record("SUNSHINE_VIDEO_DOM_V1\t1\tfuture\t0\t0\t0\t0\t0\t0\t0\t0"));
    EXPECT_FALSE(video_dom::detail::parse_protocol_record("SUNSHINE_VIDEO_DOM_V1\t1\tno-video\t1\t0\t0\t0\t0\t0\t0\t0"));
    EXPECT_FALSE(video_dom::detail::parse_protocol_record("SUNSHINE_VIDEO_DOM_V1\t1\tok\t4660\t991\t-200\t-19121\t29\t203\t29\t1201"));
    EXPECT_FALSE(video_dom::detail::parse_protocol_record("SUNSHINE_VIDEO_DOM_V1\t1\tok-fullscreen\t0\t0\t0\t0\t0\t0\t0\t0"));
    EXPECT_FALSE(video_dom::detail::parse_protocol_record(std::string(ok_record) + "\textra"));
    EXPECT_FALSE(video_dom::detail::parse_protocol_record(std::string(ok_record) + "\r"));
    EXPECT_FALSE(video_dom::detail::parse_protocol_record("SUNSHINE_VIDEO_DOM_V1\t+1\tok\t4660\t991\t-200\t-19121\t29\t203\t1802\t1201"));
  }

  TEST(VideoDomClientProtocol, RequiresExactMonotonicSequence) {
    EXPECT_TRUE(video_dom::detail::next_sequence(0, 1));
    EXPECT_TRUE(video_dom::detail::next_sequence(1, 2));
    EXPECT_FALSE(video_dom::detail::next_sequence(0, 2));
    EXPECT_FALSE(video_dom::detail::next_sequence(2, 2));
    EXPECT_FALSE(video_dom::detail::next_sequence(2, 4));
    EXPECT_FALSE(video_dom::detail::next_sequence(std::numeric_limits<std::uint64_t>::max(), 0));
  }

  TEST(VideoDomClientMapping, MapsNegativeDesktopCoordinatesAsHalfOpenPixels) {
    const auto mapped = video_dom::map_screen_rect_to_capture(
      {-1800, 100, -200, 1000},
      {-1920, 0, 0, 1080},
      1920,
      1080
    );
    ASSERT_TRUE(mapped);
    EXPECT_EQ(mapped.capture_pixels, (video_dom::rect_t {120, 100, 1720, 1000}));
    EXPECT_FLOAT_EQ(mapped.normalized.left, 120.0f / 1920.0f);
    EXPECT_FLOAT_EQ(mapped.normalized.top, 100.0f / 1080.0f);
    EXPECT_FLOAT_EQ(mapped.normalized.right, 1720.0f / 1920.0f);
    EXPECT_FLOAT_EQ(mapped.normalized.bottom, 1000.0f / 1080.0f);
  }

  TEST(VideoDomClientMapping, ClipsOnePixelEndpointRounding) {
    const auto mapped = video_dom::map_screen_rect_to_capture(
      {-1, -1, 1921, 1081},
      {0, 0, 1920, 1080},
      1920,
      1080
    );
    ASSERT_TRUE(mapped);
    EXPECT_EQ(mapped.capture_pixels, (video_dom::rect_t {0, 0, 1920, 1080}));
    EXPECT_EQ(mapped.normalized, (video_dom::normalized_rect_t {0, 0, 1, 1}));
  }

  TEST(VideoDomClientMapping, RejectsLargerOverflowExtentMismatchAndRotation) {
    EXPECT_EQ(
      video_dom::map_screen_rect_to_capture(
        {-1920, 0, 0, 1080},
        {0, 0, 1920, 1080},
        1920,
        1080
      )
        .status,
      video_dom::mapping_status_e::outside_capture
    );
    EXPECT_EQ(
      video_dom::map_screen_rect_to_capture(
        {-2, 0, 1920, 1080},
        {0, 0, 1920, 1080},
        1920,
        1080
      )
        .status,
      video_dom::mapping_status_e::outside_capture
    );
    EXPECT_EQ(
      video_dom::map_screen_rect_to_capture(
        {0, 0, 1920, 1080},
        {0, 0, 2560, 1440},
        1920,
        1080
      )
        .status,
      video_dom::mapping_status_e::extent_mismatch
    );
    EXPECT_EQ(
      video_dom::map_screen_rect_to_capture(
        {0, 0, 1920, 1080},
        {0, 0, 1920, 1080},
        1920,
        1080,
        video_dom::rotation_e::rotate_90
      )
        .status,
      video_dom::mapping_status_e::unsupported_rotation
    );
  }

  TEST(VideoDomClientSnapshot, UsabilityRequiresCompleteFreshOkIdentity) {
    const auto now = std::chrono::steady_clock::now();
    video_dom::snapshot_t snapshot {
      .status = video_dom::status_e::ok,
      .generation = 9,
      .helper_sequence = 3,
      .window = 7,
      .process_id = 8,
      .document_id = -9,
      .video_id = -10,
      .screen_rect = {0, 0, 1920, 1080},
      .received_at = now - 2s,
      .geometry_valid_since = now - 4s,
    };
    EXPECT_TRUE(video_dom::usable(snapshot, now));
    snapshot.status = video_dom::status_e::ok_fullscreen;
    EXPECT_TRUE(video_dom::usable(snapshot, now));
    EXPECT_FALSE(video_dom::usable(snapshot, now - 2001ms));
    EXPECT_FALSE(video_dom::usable(snapshot, now + 501ms));
    snapshot.status = video_dom::status_e::warming;
    EXPECT_FALSE(video_dom::usable(snapshot, now));
  }

  TEST(VideoDomClientSnapshot, PausedContentUsesContinuousGeometryNotLatestHeartbeat) {
    const auto helper_received = std::chrono::steady_clock::now();
    video_dom::snapshot_t snapshot {
      .status = video_dom::status_e::ok,
      .generation = 1,
      .helper_sequence = 1,
      .window = 7,
      .process_id = 8,
      .document_id = -9,
      .video_id = -10,
      .screen_rect = {0, 0, 1920, 1080},
      .received_at = helper_received,
      .geometry_valid_since = helper_received - 5s,
    };
    const auto paused_desktop = helper_received - 1s;
    const auto later_cursor = helper_received + 1s;
    EXPECT_TRUE(
      video_dom::detail::usable_for_content(snapshot, paused_desktop, later_cursor)
    );
    EXPECT_FALSE(
      video_dom::detail::usable_for_content(snapshot, std::nullopt, later_cursor)
    );
    EXPECT_TRUE(video_dom::usable(snapshot, later_cursor));
  }

  TEST(VideoDomClientSnapshot, GeometryMustPredateContentAndLatestHeartbeatMustStayFresh) {
    const auto now = std::chrono::steady_clock::now();
    video_dom::snapshot_t snapshot {
      .status = video_dom::status_e::ok,
      .generation = 1,
      .helper_sequence = 1,
      .window = 7,
      .process_id = 8,
      .document_id = -9,
      .video_id = -10,
      .screen_rect = {0, 0, 1920, 1080},
      .received_at = now - 100ms,
      .geometry_valid_since = now - 1s,
    };
    EXPECT_FALSE(
      video_dom::detail::usable_for_content(snapshot, now - 2s, now)
    );
    EXPECT_FALSE(
      video_dom::detail::usable_for_content(snapshot, now + 1ms, now)
    );

    snapshot.received_at = now - 2501ms;
    EXPECT_FALSE(
      video_dom::detail::usable_for_content(snapshot, now - 500ms, now)
    );
  }

  TEST(VideoDomClientSnapshot, ContinuousGeometryResetsOnAnyIdentityRectOrStatusChange) {
    const auto first_seen = std::chrono::steady_clock::now();
    const video_dom::detail::protocol_record_t record {
      .status = video_dom::status_e::ok,
      .sequence = 2,
      .window = 7,
      .process_id = 8,
      .document_id = -9,
      .video_id = -10,
      .screen_rect = {0, 0, 1920, 1080},
    };
    video_dom::snapshot_t previous {
      .status = video_dom::status_e::ok,
      .window = record.window,
      .process_id = record.process_id,
      .document_id = record.document_id,
      .video_id = record.video_id,
      .screen_rect = record.screen_rect,
      .received_at = first_seen,
      .geometry_valid_since = first_seen,
    };
    const auto heartbeat = first_seen + 1s;
    EXPECT_EQ(
      video_dom::detail::continued_geometry_valid_since(&previous, record, heartbeat),
      first_seen
    );

    auto changed = record;
    changed.screen_rect.right += 1;
    EXPECT_EQ(
      video_dom::detail::continued_geometry_valid_since(&previous, changed, heartbeat),
      heartbeat
    );
    changed = record;
    changed.window += 1;
    EXPECT_EQ(
      video_dom::detail::continued_geometry_valid_since(&previous, changed, heartbeat),
      heartbeat
    );
    changed = record;
    changed.process_id += 1;
    EXPECT_EQ(
      video_dom::detail::continued_geometry_valid_since(&previous, changed, heartbeat),
      heartbeat
    );
    changed = record;
    changed.document_id -= 1;
    EXPECT_EQ(
      video_dom::detail::continued_geometry_valid_since(&previous, changed, heartbeat),
      heartbeat
    );
    changed = record;
    changed.video_id -= 1;
    EXPECT_EQ(
      video_dom::detail::continued_geometry_valid_since(&previous, changed, heartbeat),
      heartbeat
    );
    changed = record;
    changed.status = video_dom::status_e::ok_fullscreen;
    EXPECT_EQ(
      video_dom::detail::continued_geometry_valid_since(&previous, changed, heartbeat),
      heartbeat
    );
    previous.status = video_dom::status_e::ok_fullscreen;
    EXPECT_EQ(
      video_dom::detail::continued_geometry_valid_since(&previous, changed, heartbeat),
      first_seen
    );
    previous.status = video_dom::status_e::ok;
    changed = record;
    changed.status = video_dom::status_e::changed;
    EXPECT_EQ(
      video_dom::detail::continued_geometry_valid_since(&previous, changed, heartbeat),
      std::chrono::steady_clock::time_point {}
    );
    previous.status = video_dom::status_e::stale;
    EXPECT_EQ(
      video_dom::detail::continued_geometry_valid_since(&previous, record, heartbeat),
      heartbeat
    );
  }

  TEST(VideoDomClientLifecycle, ReacquireDuringJoinRestartsOnlyAfterOldWorkerStops) {
    video_dom::detail::lifecycle_t lifecycle;

    EXPECT_EQ(
      lifecycle.acquire(),
      video_dom::detail::lifecycle_action_e::start_worker
    );
    EXPECT_EQ(
      lifecycle.release(),
      video_dom::detail::lifecycle_action_e::stop_worker
    );
    EXPECT_EQ(
      lifecycle.acquire(),
      video_dom::detail::lifecycle_action_e::none
    );
    EXPECT_EQ(
      lifecycle.worker_stopped(),
      video_dom::detail::lifecycle_action_e::start_worker
    );
  }

  TEST(VideoDomClientLifecycle, LeaseDroppedDuringJoinLeavesControllerStopped) {
    video_dom::detail::lifecycle_t lifecycle;

    EXPECT_EQ(
      lifecycle.acquire(),
      video_dom::detail::lifecycle_action_e::start_worker
    );
    EXPECT_EQ(
      lifecycle.release(),
      video_dom::detail::lifecycle_action_e::stop_worker
    );
    EXPECT_EQ(
      lifecycle.acquire(),
      video_dom::detail::lifecycle_action_e::none
    );
    EXPECT_EQ(
      lifecycle.release(),
      video_dom::detail::lifecycle_action_e::none
    );
    EXPECT_EQ(
      lifecycle.worker_stopped(),
      video_dom::detail::lifecycle_action_e::publish_stopped
    );
    EXPECT_EQ(
      lifecycle.acquire(),
      video_dom::detail::lifecycle_action_e::start_worker
    );
  }

  TEST(VideoDomClientLifecycle, FailedFirstStartCanBeRolledBackAndRetried) {
    video_dom::detail::lifecycle_t lifecycle;

    EXPECT_EQ(
      lifecycle.acquire(),
      video_dom::detail::lifecycle_action_e::start_worker
    );
    lifecycle.worker_start_failed();
    EXPECT_EQ(
      lifecycle.release(),
      video_dom::detail::lifecycle_action_e::publish_stopped
    );
    EXPECT_EQ(
      lifecycle.acquire(),
      video_dom::detail::lifecycle_action_e::start_worker
    );
  }

  TEST(VideoDomClientLifecycle, FailedRestartPreservesExistingLeaseAndAllowsRetry) {
    video_dom::detail::lifecycle_t lifecycle;

    EXPECT_EQ(
      lifecycle.acquire(),
      video_dom::detail::lifecycle_action_e::start_worker
    );
    EXPECT_EQ(
      lifecycle.release(),
      video_dom::detail::lifecycle_action_e::stop_worker
    );
    EXPECT_EQ(
      lifecycle.acquire(),
      video_dom::detail::lifecycle_action_e::none
    );
    EXPECT_EQ(
      lifecycle.worker_stopped(),
      video_dom::detail::lifecycle_action_e::start_worker
    );
    lifecycle.worker_start_failed();
    EXPECT_EQ(
      lifecycle.acquire(),
      video_dom::detail::lifecycle_action_e::start_worker
    );
  }
}  // namespace
