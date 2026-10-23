#include <gtest/gtest.h>

#include <src/host_subtitle_context.h>

#include <array>

namespace {
  namespace context = models::host_subtitle_context;

  constexpr context::key_t browser_video() {
    return {
      .kind = context::kind_e::browser_video,
      .window = 100u,
      .process_id = 200u,
      .foreground_generation = 3u,
      .browser_epoch = 4u,
    };
  }

  constexpr context::key_t video_player() {
    auto key = browser_video();
    key.kind = context::kind_e::registered_video_player;
    key.browser_epoch = 0u;
    return key;
  }

  TEST(HostSubtitleContextTest, StartsDisabledWithNonzeroGeneration) {
    context::tracker_t tracker;
    EXPECT_FALSE(tracker.eligible());
    EXPECT_EQ(tracker.kind(), context::kind_e::disabled);
    EXPECT_EQ(tracker.generation(), 1u);
    EXPECT_TRUE(tracker.matches(1u));
    EXPECT_FALSE(tracker.matches(0u));
    EXPECT_FALSE(tracker.update({}));
    EXPECT_EQ(tracker.generation(), 1u);
  }

  TEST(HostSubtitleContextTest, BrowserAndNativeVideoUseSemanticAuthorityIncludingFullscreen) {
    context::tracker_t tracker;
    // Fullscreen can canonicalize depth geometry to the whole source. This policy still uses
    // the authenticated video identity, without requiring a smaller video rectangle.
    ASSERT_TRUE(tracker.update(browser_video()));
    EXPECT_TRUE(tracker.eligible());
    EXPECT_EQ(tracker.kind(), context::kind_e::browser_video);
    EXPECT_EQ(tracker.generation(), 2u);
    EXPECT_FALSE(tracker.update(browser_video()));
    ASSERT_TRUE(tracker.update(video_player()));
    EXPECT_TRUE(tracker.eligible());
    EXPECT_EQ(tracker.kind(), context::kind_e::registered_video_player);
    EXPECT_EQ(tracker.generation(), 3u);
  }

  TEST(HostSubtitleContextTest, DisabledIdentityNoiseDoesNotCreateTransitions) {
    context::tracker_t tracker;
    auto disabled = browser_video();
    disabled.kind = context::kind_e::disabled;
    EXPECT_FALSE(tracker.update(disabled));
    EXPECT_EQ(tracker.generation(), 1u);
    ASSERT_TRUE(tracker.update(browser_video()));
    ASSERT_TRUE(tracker.update(disabled));
    EXPECT_FALSE(tracker.eligible());
    const auto disabled_generation = tracker.generation();
    disabled.window++;
    disabled.browser_epoch++;
    EXPECT_FALSE(tracker.update(disabled));
    EXPECT_EQ(tracker.generation(), disabled_generation);
  }

  TEST(HostSubtitleContextTest, AuthorityIdentityChangesRevokePreviousGeneration) {
    const auto original = browser_video();
    auto window_changed = original;
    ++window_changed.window;
    auto process_changed = original;
    ++process_changed.process_id;
    auto foreground_changed = original;
    ++foreground_changed.foreground_generation;
    auto browser_changed = original;
    ++browser_changed.browser_epoch;
    for (const auto changed : std::array {window_changed, process_changed, foreground_changed, browser_changed}) {
      context::tracker_t tracker;
      ASSERT_TRUE(tracker.update(original));
      const auto submitted_generation = tracker.generation();
      EXPECT_TRUE(tracker.update(changed));
      EXPECT_FALSE(tracker.matches(submitted_generation));
      EXPECT_EQ(tracker.generation(), submitted_generation + 1u);
      EXPECT_FALSE(tracker.update(changed));
    }
  }

  TEST(HostSubtitleContextTest, VideoDesktopVideoCannotReauthorizePendingSubmission) {
    for (const auto video : std::array {browser_video(), video_player()}) {
      context::tracker_t tracker;
      ASSERT_TRUE(tracker.update(video));
      const auto pending_generation = tracker.generation();
      ASSERT_TRUE(tracker.update({}));
      const auto desktop_generation = tracker.generation();
      EXPECT_FALSE(tracker.matches(pending_generation));
      ASSERT_TRUE(tracker.update(video));
      EXPECT_TRUE(tracker.eligible());
      EXPECT_FALSE(tracker.matches(pending_generation));
      EXPECT_FALSE(tracker.matches(desktop_generation));
      EXPECT_EQ(tracker.generation(), pending_generation + 2u);
    }
  }

  TEST(HostSubtitleContextTest, RegistryReadinessAndLossAreEligibilityTransitions) {
    context::tracker_t tracker;
    // The same foreground window is disabled until registration evidence is ready.
    auto unavailable = video_player();
    unavailable.kind = context::kind_e::disabled;
    EXPECT_FALSE(tracker.update(unavailable));
    ASSERT_TRUE(tracker.update(video_player()));
    const auto ready_generation = tracker.generation();
    EXPECT_TRUE(tracker.eligible());
    ASSERT_TRUE(tracker.update(unavailable));
    EXPECT_FALSE(tracker.eligible());
    EXPECT_FALSE(tracker.matches(ready_generation));
  }

  TEST(HostSubtitleContextTest, MalformedOrUnknownEligibleIdentityFailsClosed) {
    for (const auto valid : std::array {browser_video(), video_player()}) {
      auto missing_window = valid;
      missing_window.window = 0u;
      auto missing_process = valid;
      missing_process.process_id = 0u;
      auto missing_foreground = valid;
      missing_foreground.foreground_generation = 0u;
      auto unknown_kind = valid;
      unknown_kind.kind = static_cast<context::kind_e>(255u);
      for (const auto malformed : std::array {missing_window, missing_process, missing_foreground, unknown_kind}) {
        context::tracker_t tracker;
        ASSERT_TRUE(tracker.update(valid));
        EXPECT_TRUE(tracker.update(malformed));
        EXPECT_FALSE(tracker.eligible());
        EXPECT_EQ(tracker.kind(), context::kind_e::disabled);
        EXPECT_FALSE(tracker.update({}));
      }
    }
    auto missing_browser_epoch = browser_video();
    missing_browser_epoch.browser_epoch = 0u;
    context::tracker_t tracker;
    EXPECT_FALSE(tracker.update(missing_browser_epoch));
    EXPECT_FALSE(tracker.eligible());
  }

  TEST(HostSubtitleContextTest, NativePlayerIgnoresBrowserEpoch) {
    context::tracker_t tracker;
    ASSERT_TRUE(tracker.update(video_player()));
    const auto player_generation = tracker.generation();
    auto unrelated_browser_change = video_player();
    unrelated_browser_change.browser_epoch = 99u;
    EXPECT_FALSE(tracker.update(unrelated_browser_change));
    EXPECT_EQ(tracker.generation(), player_generation);
    EXPECT_TRUE(tracker.eligible());
  }
}  // namespace
