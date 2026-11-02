#include <gtest/gtest.h>
#ifdef _WIN32
// Match the existing Windows platform fixture prerequisite before display.h.
typedef long NTSTATUS;
#include "src/platform/windows/display.h"

namespace {
  namespace d = platf::dxgi::detail;
  DXGI_OUTDUPL_FRAME_INFO present(UINT count) {
    DXGI_OUTDUPL_FRAME_INFO info {};
    info.LastPresentTime.QuadPart = 100;
    info.AccumulatedFrames = count;
    return info;
  }
  auto history() { return std::make_shared<d::ddup_damage_history_t>(); }
}

TEST(WindowsDdupCaptureIntervalTest, ActualOnePresentIsSeparateFromDamageMetadata) {
  const auto info = present(1u);
  EXPECT_TRUE(d::ddup_frame_has_one_desktop_present(info));
  EXPECT_FALSE(d::ddup_frame_damage_metadata_available(info));
}
TEST(WindowsDdupCaptureIntervalTest, CoalescedAndMissingPresentCountsAreUnknown) {
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(present(0u)));
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(present(2u)));
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(present(UINT_MAX)));
}
TEST(WindowsDdupCaptureIntervalTest, CursorOnlyAndInvalidPresentTimesAreUnknown) {
  auto info = present(1u);
  info.LastPresentTime.QuadPart = 0;
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(info));
  info.LastPresentTime.QuadPart = -1;
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(info));
}
TEST(WindowsDdupCaptureIntervalTest, ProtectedContentCannotSupplyContinuity) {
  auto info = present(1u);
  info.ProtectedContentMaskedOut = TRUE;
  EXPECT_FALSE(d::ddup_frame_has_one_desktop_present(info));
}
TEST(WindowsDdupCaptureIntervalTest, DamageOnlyAuthoredUpdatesDoNotGainPresentEvidence) {
  auto h = history();
  const auto a = h->commit({true, {}});
  const auto b = h->commit({true, {}});
  EXPECT_FALSE(b.single_desktop_present);
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(a, b));
}
TEST(WindowsDdupCaptureIntervalTest, OneObservedPresentDoesNotRequireKnownDirtyRects) {
  auto h = history();
  const auto a = h->commit({false, {}, false});
  const auto b = h->commit({false, {}, true});
  EXPECT_TRUE(d::ddup_adjacent_desktop_present_observed(a, b));
  EXPECT_EQ(d::query_ddup_damage_between(a, b, RECT {0, 0, 100, 100}),
            d::ddup_damage_intersection_e::unknown);
}
TEST(WindowsDdupCaptureIntervalTest, SkippedCommittedSurfacesNeverProveCompleteObservation) {
  auto h = history();
  const auto a = h->commit({true, {}, true});
  (void)h->commit({true, {}, true});
  const auto c = h->commit({true, {}, true});
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(a, c));
}
TEST(WindowsDdupCaptureIntervalTest, SameCaptureRetryIsNotANewPresent) {
  auto h = history();
  const auto a = h->commit({true, {}, true});
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(a, a));
}
TEST(WindowsDdupCaptureIntervalTest, RegressedAndCrossNamespaceSnapshotsAreUnknown) {
  auto h = history();
  const auto a = h->commit({true, {}, true});
  const auto b = h->commit({true, {}, true});
  auto other = history();
  const auto c = other->commit({true, {}, true});
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(b, a));
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(a, c));
}
TEST(WindowsDdupCaptureIntervalTest, MissingAndOverflowedIdentitiesAreUnknown) {
  auto h = history();
  d::ddup_damage_snapshot_t a {h, UINT64_MAX, true};
  d::ddup_damage_snapshot_t b {h, 0u, true};
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(a, b));
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(std::nullopt, a));
  EXPECT_FALSE(d::ddup_adjacent_desktop_present_observed(b, a));
}

#endif // _WIN32
