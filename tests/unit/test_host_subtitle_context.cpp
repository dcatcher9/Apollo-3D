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

#ifdef _WIN32
  #include "src/platform/windows/subtitle_context_policy.h"

namespace {
  namespace foreground = platf::foreground_window;
  namespace dom = platf::video_dom;
  namespace policy = platf::subtitle_context_policy;
  using namespace std::chrono_literals;

  class BrowserSubtitleContextPolicyTest: public testing::Test {
  protected:
    const std::chrono::steady_clock::time_point now {10s};
    std::optional<std::chrono::steady_clock::time_point> content {now - 5ms};
    foreground::capture_target_t target {
      .screen_rect = {0, 0, 1920, 1080}, .width = 1920, .height = 1080, .monitor = 0x200u,
    };
    foreground::snapshot_t root {
      .status = foreground::status_e::ok, .generation = 5u, .window = 0x100u,
      .process_id = 42u, .monitor = 0x200u, .monitor_screen_rect = {0, 0, 1920, 1080},
      .client_screen_rect = {-100, 50, 1850, 1040},
      .frame_screen_rect = {-110, 20, 1860, 1050},
      .observed_at = now - 1ms, .geometry_valid_since = now - 1s,
    };
    dom::snapshot_t browser {
      .status = dom::status_e::ok, .generation = 9u, .helper_sequence = 3u,
      .window = 0x100u, .process_id = 42u, .document_id = 7, .video_id = 8,
      .screen_rect = {200, 200, 1600, 900},
      .received_at = now - 100ms, .geometry_valid_since = now - 1s,
    };

    policy::evaluation_t evaluate(const std::uint64_t epoch = 4u) {
      const auto client_mapping = foreground::map_to_capture(root, target);
      return policy::evaluate(&root, static_cast<bool>(client_mapping), client_mapping, &browser, epoch,
        {target.screen_rect.left, target.screen_rect.top, target.screen_rect.right, target.screen_rect.bottom},
        target.width, target.height, content, now);
    }

    void expect_disabled() {
      const auto result = evaluate();
      EXPECT_EQ(result.browser_key.kind, context::kind_e::disabled);
      context::tracker_t tracker;
      EXPECT_FALSE(tracker.update(result.browser_key));
      EXPECT_FALSE(tracker.eligible());
    }
  };

  TEST_F(BrowserSubtitleContextPolicyTest, ContainedDomVideoSurvivesPartiallyOutsideClient) {
    const auto client = foreground::map_to_capture(root, target);
    ASSERT_EQ(client.status, foreground::mapping_status_e::partially_outside_capture);
    ASSERT_FALSE(client);
    ASSERT_TRUE(foreground::usable_for_content(root, content, now));
    const auto video = dom::map_screen_rect_to_capture(browser.screen_rect, {0, 0, 1920, 1080}, 1920, 1080);
    ASSERT_TRUE(video);
    EXPECT_EQ(video.capture_pixels, browser.screen_rect);
    const auto result = evaluate();
    EXPECT_TRUE(result.foreground_usable);
    EXPECT_TRUE(result.browser_identity_matches);
    EXPECT_TRUE(result.browser_usable);
    EXPECT_TRUE(result.mapping_succeeded);
    EXPECT_FALSE(result.registered_player_candidate);
    context::tracker_t tracker;
    ASSERT_TRUE(tracker.update(result.browser_key));
    EXPECT_TRUE(tracker.eligible());
    EXPECT_EQ(tracker.kind(), context::kind_e::browser_video);
    EXPECT_EQ(result.browser_key, (context::key_t {context::kind_e::browser_video, 0x100u, 42u, 5u, 4u}));
  }

  TEST_F(BrowserSubtitleContextPolicyTest, ContainedClientKeepsSameBrowserKeyAndNativeFallback) {
    const auto clipped_key = evaluate().browser_key;
    root.client_screen_rect.left = 100;
    root.frame_screen_rect.left = 90;
    ASSERT_TRUE(foreground::map_to_capture(root, target));
    EXPECT_EQ(evaluate().browser_key, clipped_key);
    const auto native = policy::evaluate(&root, true, foreground::map_to_capture(root, target), nullptr, 0u, {0, 0, 1920, 1080}, 1920, 1080, content, now);
    EXPECT_TRUE(native.foreground_usable);
    EXPECT_TRUE(native.registered_player_candidate);
    root.client_screen_rect.left = -100;
    root.frame_screen_rect.left = -110;
    const bool client_mapped = static_cast<bool>(foreground::map_to_capture(root, target));
    ASSERT_FALSE(client_mapped);
    const auto clipped_native = policy::evaluate(&root, client_mapped, foreground::map_to_capture(root, target), nullptr, 0u,
      {0, 0, 1920, 1080}, 1920, 1080, content, now);
    EXPECT_TRUE(clipped_native.foreground_usable);
    EXPECT_FALSE(clipped_native.registered_player_candidate);
  }

  TEST_F(BrowserSubtitleContextPolicyTest, OutsideVideoAndInvalidCaptureStayDisabled) {
    browser.screen_rect.left = -2;
    ASSERT_EQ(dom::map_screen_rect_to_capture(browser.screen_rect, {0, 0, 1920, 1080}, 1920, 1080).status,
      dom::mapping_status_e::outside_capture);
    expect_disabled();
    browser.screen_rect.left = 200;
    target.width = 1919;
    EXPECT_EQ(foreground::map_to_capture(root, target).status, foreground::mapping_status_e::invalid_capture_target);
    EXPECT_FALSE(evaluate().foreground_target_usable);
    EXPECT_FALSE(evaluate().mapping_attempted);
    expect_disabled();
  }

  TEST_F(BrowserSubtitleContextPolicyTest, WrongIdentityOrMissingBrowserAuthorityStaysDisabled) {
    browser.process_id++;
    expect_disabled();
    browser.process_id--;
    browser.window++;
    expect_disabled();
    browser.window--;
    EXPECT_EQ(evaluate(0u).browser_key.kind, context::kind_e::disabled);
    const auto missing_root = policy::evaluate(nullptr, false, foreground::map_to_capture(root, target), &browser, 4u,
      {0, 0, 1920, 1080}, 1920, 1080, content, now);
    EXPECT_FALSE(missing_root.foreground_usable);
    EXPECT_EQ(missing_root.browser_key.kind, context::kind_e::disabled);
  }

  TEST_F(BrowserSubtitleContextPolicyTest, InteractiveMoveStaleAndNoncausalContentStayDisabled) {
    root.status = foreground::status_e::interactive_move_size;
    expect_disabled();
    root.status = foreground::status_e::ok;
    root.observed_at = now - 251ms;
    expect_disabled();
    root.observed_at = now - 1ms;
    browser.received_at = now - 2501ms;
    expect_disabled();
    browser.received_at = now - 100ms;
    root.geometry_valid_since = *content + 1ms;
    expect_disabled();
    root.geometry_valid_since = now - 1s;
    browser.geometry_valid_since = *content + 1ms;
    expect_disabled();
    browser.geometry_valid_since = now - 1s;
    content = now + 1ms;
    expect_disabled();
    content.reset();
    expect_disabled();
  }

  TEST_F(BrowserSubtitleContextPolicyTest, FullscreenClaimStillRequiresWholeCapture) {
    browser.status = dom::status_e::ok_fullscreen;
    EXPECT_TRUE(evaluate().mapping_succeeded);
    EXPECT_FALSE(evaluate().fullscreen_match);
    expect_disabled();
    browser.screen_rect = {0, 0, 1920, 1080};
    EXPECT_EQ(evaluate().browser_key.kind, context::kind_e::browser_video);
  }

  TEST_F(BrowserSubtitleContextPolicyTest, NoncausalBrowserRevokesPreviousContextWithoutNativeFallback) {
    root.client_screen_rect.left = 100;
    root.frame_screen_rect.left = 90;
    context::tracker_t tracker;
    ASSERT_TRUE(tracker.update(evaluate().browser_key));
    const auto accepted_generation = tracker.generation();
    browser.received_at = now - 1ms;
    browser.geometry_valid_since = now - 1ms;
    const auto rejected = evaluate();
    EXPECT_FALSE(rejected.registered_player_candidate);
    ASSERT_TRUE(tracker.update(rejected.browser_key));
    EXPECT_FALSE(tracker.eligible());
    EXPECT_FALSE(tracker.matches(accepted_generation));
  }

  TEST_F(BrowserSubtitleContextPolicyTest, RealManual139GeometryRemainsBrowserEligible) {
    target = {{0, 0, 3840, 2160}, 3840, 2160, 0x200u, true};
    root.window = 0x140B14u;
    root.process_id = 3172u;
    root.generation = 8u;
    root.monitor_screen_rect = target.screen_rect;
    root.client_screen_rect = {875, 10, 3949, 1739};
    root.frame_screen_rect = {865, -20, 3959, 1749};
    browser.window = root.window;
    browser.process_id = root.process_id;
    browser.document_id = -35710;
    browser.video_id = -35868;
    browser.screen_rect = {1089, 425, 2984, 1492};
    browser.received_at = now - 426ms;
    browser.geometry_valid_since = now - 223208ms;
    content = now;
    ASSERT_EQ(foreground::map_to_capture(root, target).status,
      foreground::mapping_status_e::partially_outside_capture);
    const auto result = evaluate();
    EXPECT_TRUE(result.foreground_target_usable);
    EXPECT_EQ(result.browser_key.kind, context::kind_e::browser_video);
    EXPECT_EQ(result.browser_key.window, root.window);
    EXPECT_EQ(result.browser_key.process_id, root.process_id);
  }

  TEST_F(BrowserSubtitleContextPolicyTest, MonitorMismatchStaysDisabledButExactCloneRemainsEligible) {
    target.monitor = 0x201u;
    root.monitor_screen_rect = {-1920, 0, 0, 1080};
    ASSERT_EQ(foreground::map_to_capture(root, target).status, foreground::mapping_status_e::monitor_mismatch);
    const auto wrong_monitor = evaluate();
    EXPECT_FALSE(wrong_monitor.foreground_target_usable);
    EXPECT_FALSE(wrong_monitor.mapping_attempted);
    expect_disabled();
    root.monitor_screen_rect = target.screen_rect;
    ASSERT_EQ(foreground::map_to_capture(root, target).status,
      foreground::mapping_status_e::partially_outside_capture);
    EXPECT_EQ(evaluate().browser_key.kind, context::kind_e::browser_video);
    root.client_screen_rect.left = 100;
    root.frame_screen_rect.left = 90;
    ASSERT_TRUE(foreground::map_to_capture(root, target));
    EXPECT_EQ(evaluate().browser_key.kind, context::kind_e::browser_video);
  }

  TEST_F(BrowserSubtitleContextPolicyTest, UnsupportedOrientationExtentAndFailedTargetStayDisabled) {
    // Covers180 degrees and square90 degrees: matching extents alone never grant orientation.
    for (const auto square : {false, true}) {
      target.identity_orientation = false;
      if (square) {
        target.screen_rect = {0, 0, 1920, 1920};
        target.height = 1920;
        root.monitor_screen_rect = target.screen_rect;
      }
      ASSERT_EQ(foreground::map_to_capture(root, target).status, foreground::mapping_status_e::unsupported_orientation);
      EXPECT_FALSE(evaluate().foreground_target_usable);
      EXPECT_FALSE(evaluate().mapping_attempted);
      expect_disabled();
    }
    target.identity_orientation = true;
    target.width--;
    ASSERT_EQ(foreground::map_to_capture(root, target).status, foreground::mapping_status_e::invalid_capture_target);
    expect_disabled();
    target.width++;
    // Failed GetDesc must remain invalid even if COM wrote apparently valid output coordinates.
    const foreground::mapping_result_t failed_target {.status = foreground::mapping_status_e::invalid_capture_target};
    const auto rejected = policy::evaluate(&root, false, failed_target, &browser, 4u,
      {0, 0, 1920, 1920}, 1920, 1920, content, now);
    EXPECT_FALSE(rejected.foreground_target_usable);
    EXPECT_FALSE(rejected.mapping_attempted);
    EXPECT_EQ(rejected.browser_key.kind, context::kind_e::disabled);
  }
}  // namespace
#endif
