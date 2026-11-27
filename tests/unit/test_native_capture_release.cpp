/**
 * @file tests/unit/test_native_capture_release.cpp
 * @brief Exercise the concrete native encoder's capture-release boundary without GPU initialization.
 */
#include "../tests_common.h"

#include "src/platform/windows/display.h"

namespace {
  void expect_capture_released_while_native_teardown_is_held(const platf::dxgi::native_capture_release_report_t &report) {
    ASSERT_TRUE(report.valid);
    EXPECT_NE(report.own_token, 0u);
    EXPECT_TRUE(report.gate_entered);
    EXPECT_TRUE(report.owner_consumed);
    EXPECT_TRUE(report.retirement_pending_before_release);
    EXPECT_TRUE(report.display_expired_before_release);
    EXPECT_EQ(report.final_claim, 0u);
    EXPECT_TRUE(report.completed_after_release);
    EXPECT_FALSE(report.gate_timed_out);
  }
}  // namespace

TEST(NativeCaptureReleaseTest, ReleasesItsOwnClaimAndDisplayBeforeNativeMemberTeardown) {
  const auto report = platf::dxgi::native_capture_release_for_test(false);
  expect_capture_released_while_native_teardown_is_held(report);
  EXPECT_EQ(report.foreign_token, 0u);
  EXPECT_EQ(report.claim_after_capture_release, 0u);
}

TEST(NativeCaptureReleaseTest, PreservesAnotherConvertersClaimUntilThatConverterReleasesIt) {
  const auto report = platf::dxgi::native_capture_release_for_test(true);
  expect_capture_released_while_native_teardown_is_held(report);
  EXPECT_NE(report.foreign_token, 0u);
  EXPECT_NE(report.foreign_token, report.own_token);
  EXPECT_EQ(report.claim_after_capture_release, report.foreign_token);
}
