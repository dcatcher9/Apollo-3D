#include "src/platform/windows/exclusive_cursor_clip.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
  using namespace platf::primary_display::detail;

  struct fake_clip_t {
    cursor_bounds_t desktop {-1920, 0, 3840, 1080};
    cursor_bounds_t current = desktop;
    bool query_ok = true;
    bool apply_ok = true;
    bool fail_verification = false;
    unsigned queries = 0;
    std::vector<std::optional<cursor_bounds_t>> applied;

    cursor_clip_io_t io() {
      return {
        [&]() -> std::optional<cursor_bounds_t> {
          ++queries;
          return query_ok && !(fail_verification && queries == 2) ? std::optional {current} : std::nullopt;
        },
        [&]() {
          return std::optional {desktop};
        },
        [&](std::optional<cursor_bounds_t> bounds) {
          applied.push_back(bounds);
          if (apply_ok) {
            current = bounds.value_or(desktop);
          }
          return apply_ok;
        },
      };
    }
  };

  TEST(ExclusiveCursorClip, NativeBoundaryExcludesLeftArAndRefreshesAfterResize) {
    fake_clip_t fake;
    cursor_clip_manager_t guard(fake.io());
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    EXPECT_EQ(fake.current, (cursor_bounds_t {0, 0, 1920, 1080}));
    EXPECT_FALSE(cursor_bounds_intersection(fake.current, {-1, 500, 1, 1}));
    EXPECT_TRUE(cursor_bounds_intersection(fake.current, {0, 500, 1, 1}));
    fake.desktop = {-1920, 0, 5760, 2160};
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 3840, 2160}));
    EXPECT_EQ(fake.current, (cursor_bounds_t {0, 0, 3840, 2160}));
    ASSERT_TRUE(guard.release(L"virtual"));
    EXPECT_FALSE(fake.applied.back());
    EXPECT_EQ(fake.current, fake.desktop);
  }

  TEST(ExclusiveCursorClip, ReappliesAfterWindowsResetsClipToExpandedDesktop) {
    fake_clip_t fake;
    fake.desktop = {0, 0, 1920, 1080};
    fake.current = fake.desktop;
    cursor_clip_manager_t guard(fake.io());
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    const auto initial_apply_count = fake.applied.size();

    // Display activation resets the shared Windows clip to the expanded virtual desktop.
    fake.desktop = {-1920, 0, 3840, 1080};
    fake.current = fake.desktop;
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    ASSERT_EQ(fake.applied.size(), initial_apply_count + 1);
    EXPECT_EQ(fake.current, (cursor_bounds_t {0, 0, 1920, 1080}));
    EXPECT_FALSE(cursor_bounds_intersection(fake.current, {-1, 500, 1, 1}));
  }

  TEST(ExclusiveCursorClip, RedundantForegroundRefreshDoesNotRewriteOwnedClip) {
    fake_clip_t fake;
    cursor_clip_manager_t guard(fake.io());
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    const auto initial_apply_count = fake.applied.size();
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    EXPECT_EQ(fake.applied.size(), initial_apply_count);
  }

  TEST(ExclusiveCursorClip, PreservesAndRestoresExistingAppClipWithoutWidening) {
    fake_clip_t fake;
    fake.current = {100, 100, 400, 300};
    const auto original = fake.current;
    cursor_clip_manager_t guard(fake.io());
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    EXPECT_EQ(fake.current, original);
    ASSERT_TRUE(guard.release());
    EXPECT_EQ(fake.applied.back(), original);
  }

  TEST(ExclusiveCursorClip, OverridesDisjointApplicationClipAndRestoresItOnRelease) {
    fake_clip_t fake;
    fake.current = {-500, 0, 400, 300};
    const auto original = fake.current;
    cursor_clip_manager_t guard(fake.io());
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    EXPECT_TRUE(guard.owns_clip());
    EXPECT_EQ(fake.current, (cursor_bounds_t {0, 0, 1920, 1080}));
    ASSERT_TRUE(guard.release());
    EXPECT_EQ(fake.current, original);
  }

  TEST(ExclusiveCursorClip, DoesNotRestoreOverAnotherApplicationsReplacement) {
    fake_clip_t fake;
    cursor_clip_manager_t guard(fake.io());
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    const auto calls = fake.applied.size();
    fake.current = {100, 100, 400, 300};
    ASSERT_TRUE(guard.release());
    EXPECT_EQ(fake.applied.size(), calls);
    EXPECT_EQ(fake.current, (cursor_bounds_t {100, 100, 400, 300}));
  }

  TEST(ExclusiveCursorClip, RefreshPreservesTheLatestApplicationRestriction) {
    fake_clip_t fake;
    cursor_clip_manager_t guard(fake.io());
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    fake.current = {100, 100, 400, 300};
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    ASSERT_TRUE(guard.release());
    EXPECT_EQ(fake.current, (cursor_bounds_t {100, 100, 400, 300}));
  }

  TEST(ExclusiveCursorClip, FailedVerificationRetainsOwnershipForCleanup) {
    fake_clip_t fake;
    fake.fail_verification = true;
    cursor_clip_manager_t guard(fake.io());
    EXPECT_FALSE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    EXPECT_TRUE(guard.owns_clip());
    ASSERT_TRUE(guard.release());
    EXPECT_EQ(fake.current, fake.desktop);
    EXPECT_FALSE(guard.owns_clip());
  }

  TEST(ExclusiveCursorClip, FailedReleaseKeepsOwnershipUntilRetrySucceeds) {
    fake_clip_t fake;
    cursor_clip_manager_t guard(fake.io());
    ASSERT_TRUE(guard.confine(L"virtual", {0, 0, 1920, 1080}));
    fake.apply_ok = false;
    EXPECT_FALSE(guard.release());
    EXPECT_TRUE(guard.owns_clip());
    fake.apply_ok = true;
    fake.query_ok = false;
    EXPECT_FALSE(guard.release());
    EXPECT_TRUE(guard.owns_clip());
    fake.query_ok = true;
    EXPECT_TRUE(guard.release());
    EXPECT_FALSE(guard.owns_clip());
  }

  TEST(ExclusiveCursorClip, OldSessionCannotReleaseOrReplaceANewerDisplayClip) {
    fake_clip_t fake;
    cursor_clip_manager_t guard(fake.io());
    ASSERT_TRUE(guard.confine(L"new-virtual", {0, 0, 1920, 1080}));
    const auto calls = fake.applied.size();
    EXPECT_FALSE(guard.release(L"old-virtual"));
    EXPECT_FALSE(guard.confine(L"old-virtual", {0, 0, 640, 480}));
    EXPECT_EQ(fake.applied.size(), calls);
    EXPECT_TRUE(guard.release(L"NEW-VIRTUAL"));
  }
}  // namespace
