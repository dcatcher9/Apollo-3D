/**
 * @file tests/unit/platform/test_sbs_debug_dump_border.cpp
 * @brief Fail-closed contract tests for optional matched-frame window-region provenance.
 */
#include "../../tests_common.h"

#ifdef _WIN32
  #include <src/platform/windows/sbs_debug_dump_border.h>

namespace {
  using platf::sbs_debug::validate_window_region;
  using platf::sbs_debug::window_region_authority_kind_e;
  using platf::sbs_debug::window_region_error;
  using platf::sbs_debug::window_region_snapshot;

  window_region_snapshot valid_chromium_region() {
    window_region_snapshot region;
    region.matched_frame_id = 41;
    region.source_width = 3840;
    region.source_height = 2160;
    region.left = 320;
    region.top = 180;
    region.right = 3520;
    region.bottom = 1980;
    region.hwnd = 0x1234u;
    region.process_id = 55;
    region.document_id = -7;
    region.video_id = -9;
    region.generation = 3;
    region.latest_observation_age_ms_at_capture = 120;
    region.maximum_observation_age_ms = 1000;
    region.geometry_continuity_ms_at_capture = 5000;
    region.source_content_age_ms_at_capture = 1000;
    return region;
  }

  TEST(SbsDebugDumpBorderTest, AcceptsChromiumAndForegroundIdentityShapes) {
    auto region = valid_chromium_region();
    EXPECT_EQ(
      validate_window_region(region, 41, 3840, 2160),
      window_region_error::none
    );

    region.authority_kind = window_region_authority_kind_e::foreground_client;
    region.document_id = 0;
    region.video_id = 0;
    EXPECT_EQ(
      validate_window_region(region, 41, 3840, 2160),
      window_region_error::none
    );
  }

  TEST(SbsDebugDumpBorderTest, RejectsCurrentBorderForAnOlderMatchedFrame) {
    const auto region = valid_chromium_region();
    EXPECT_EQ(
      validate_window_region(region, 40, 3840, 2160),
      window_region_error::frame_mismatch
    );
  }

  TEST(SbsDebugDumpBorderTest, RejectsExtentAndRectangleMismatches) {
    auto region = valid_chromium_region();
    EXPECT_EQ(
      validate_window_region(region, 41, 1920, 1080),
      window_region_error::source_extent_mismatch
    );

    region = valid_chromium_region();
    region.right = 3841;
    EXPECT_EQ(
      validate_window_region(region, 41, 3840, 2160),
      window_region_error::empty_or_out_of_bounds_rect
    );

    region = valid_chromium_region();
    region.bottom = region.top;
    EXPECT_EQ(
      validate_window_region(region, 41, 3840, 2160),
      window_region_error::empty_or_out_of_bounds_rect
    );
  }

  TEST(SbsDebugDumpBorderTest, RejectsMissingIdentityAndStaleEvidence) {
    auto region = valid_chromium_region();
    region.video_id = 0;
    EXPECT_EQ(
      validate_window_region(region, 41, 3840, 2160),
      window_region_error::missing_identity
    );

    region = valid_chromium_region();
    region.authority_kind = window_region_authority_kind_e::foreground_client;
    EXPECT_EQ(
      validate_window_region(region, 41, 3840, 2160),
      window_region_error::unexpected_dom_identity
    );

    region = valid_chromium_region();
    region.authority_kind = static_cast<window_region_authority_kind_e>(0xFFu);
    EXPECT_EQ(
      validate_window_region(region, 41, 3840, 2160),
      window_region_error::unknown_authority_kind
    );

    region = valid_chromium_region();
    region.latest_observation_age_ms_at_capture = 1001;
    EXPECT_EQ(
      validate_window_region(region, 41, 3840, 2160),
      window_region_error::stale
    );

    region = valid_chromium_region();
    region.geometry_continuity_ms_at_capture = 999;
    EXPECT_EQ(
      validate_window_region(region, 41, 3840, 2160),
      window_region_error::noncausal_geometry
    );
  }
}  // namespace
#endif
