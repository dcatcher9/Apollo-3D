/**
 * @file tests/unit/platform/test_sbs_debug_dump_border.cpp
 * @brief Fail-closed contract tests for the optional matched-frame video-border artifact.
 */
#include "../../tests_common.h"

#ifdef _WIN32
  #include <src/platform/windows/sbs_debug_dump_border.h>

namespace {
  using platf::sbs_debug::validate_window_video_border;
  using platf::sbs_debug::window_video_border_error;
  using platf::sbs_debug::window_video_border_snapshot;

  window_video_border_snapshot valid_border() {
    window_video_border_snapshot border;
    border.matched_frame_id = 41;
    border.source_width = 3840;
    border.source_height = 2160;
    border.left = 320;
    border.top = 180;
    border.right = 3520;
    border.bottom = 1980;
    border.hwnd = 0x1234u;
    border.process_id = 55;
    border.document_id = -7;
    border.video_id = -9;
    border.generation = 3;
    border.latest_heartbeat_age_ms_at_capture = 120;
    border.maximum_heartbeat_age_ms = 1000;
    border.geometry_continuity_ms_at_capture = 5000;
    border.source_content_age_ms_at_capture = 1000;
    return border;
  }

  TEST(SbsDebugDumpBorderTest, AcceptsExactHalfOpenMatchedFrameRectangle) {
    const auto border = valid_border();
    EXPECT_EQ(
      validate_window_video_border(border, 41, 3840, 2160),
      window_video_border_error::none
    );
  }

  TEST(SbsDebugDumpBorderTest, RejectsCurrentBorderForAnOlderMatchedFrame) {
    const auto border = valid_border();
    EXPECT_EQ(
      validate_window_video_border(border, 40, 3840, 2160),
      window_video_border_error::frame_mismatch
    );
  }

  TEST(SbsDebugDumpBorderTest, RejectsExtentAndRectangleMismatches) {
    auto border = valid_border();
    EXPECT_EQ(
      validate_window_video_border(border, 41, 1920, 1080),
      window_video_border_error::source_extent_mismatch
    );

    border = valid_border();
    border.right = 3841;
    EXPECT_EQ(
      validate_window_video_border(border, 41, 3840, 2160),
      window_video_border_error::empty_or_out_of_bounds_rect
    );

    border = valid_border();
    border.bottom = border.top;
    EXPECT_EQ(
      validate_window_video_border(border, 41, 3840, 2160),
      window_video_border_error::empty_or_out_of_bounds_rect
    );
  }

  TEST(SbsDebugDumpBorderTest, RejectsMissingIdentityAndStaleEvidence) {
    auto border = valid_border();
    border.video_id = 0;
    EXPECT_EQ(
      validate_window_video_border(border, 41, 3840, 2160),
      window_video_border_error::missing_identity
    );

    border = valid_border();
    border.latest_heartbeat_age_ms_at_capture = 1001;
    EXPECT_EQ(
      validate_window_video_border(border, 41, 3840, 2160),
      window_video_border_error::stale
    );

    border = valid_border();
    border.geometry_continuity_ms_at_capture = 999;
    EXPECT_EQ(
      validate_window_video_border(border, 41, 3840, 2160),
      window_video_border_error::noncausal_geometry
    );
  }
}  // namespace
#endif
