#include <gtest/gtest.h>

#include "src/platform/windows/capture_timing.h"

namespace {
  using namespace std::chrono_literals;
  using namespace platf::dxgi::detail;

  TEST(WindowsQpcWgcTest, DelayedFrameHasSameAgeAtDifferentCounterFrequencies) {
    // Frame at 9.975 s after boot, observed at 10 s. The ABI timestamp is independent of QPC Hz.
    const wgc_timestamp_t frame_time {99'750'000};
    for (const std::int64_t frequency : {3'125'000LL, 10'000'000LL, 24'000'000LL}) {
      EXPECT_NEAR(wgc_frame_age(10 * frequency, frequency, frame_time).count(),
                  std::chrono::nanoseconds {25ms}.count(), 1);
    }
  }

  TEST(WindowsQpcWgcTest, UptimeDoesNotBecomeAnArtificialFutureFrame) {
    constexpr std::int64_t frequency = 3'125'000;
    constexpr std::int64_t uptime_seconds = 7 * 24 * 60 * 60;
    const auto timestamp = wgc_timestamp_t {std::chrono::seconds {uptime_seconds}} -
                           wgc_timestamp_t {123ms};
    EXPECT_NEAR(wgc_frame_age(uptime_seconds * frequency, frequency, timestamp).count(),
                std::chrono::nanoseconds {123ms}.count(), 1);
  }

  TEST(WindowsQpcWgcTest, SubMicrosecondResolutionAndSignedAgeArePreserved) {
    EXPECT_NEAR(wgc_frame_age(10'000'000, 10'000'000, wgc_timestamp_t {9'999'999}).count(),
                100, 1);
    EXPECT_NEAR(wgc_frame_age(10'000'000, 10'000'000, wgc_timestamp_t {10'000'001}).count(),
                -100, 1);
    EXPECT_EQ(wgc_frame_age(1, 0, wgc_timestamp_t {1}), 0ns);
  }

  TEST(WindowsLocalPresenterTimingTest, FinalDepthCompletionIsServicedWithoutAnotherSource) {
    local_presenter_retry_state_t presenter;
    presenter.observe_source();
    presenter.record_converted();
    presenter.record_presented();
    // The last flat frame has been presented, but its depth root is still in flight. A capture
    // timeout must return to the same owner promptly even though no new image ever arrives.
    const capture_wait_policy_t wait {presenter.should_process(true, false, true)};
    EXPECT_LE(wait.source_timeout() + wait.idle_backoff(), 6ms);
    EXPECT_LE(wait.pacing_sleep(1s), 5ms);
    EXPECT_FALSE(wait.retry_after_pacing_timeout());
    EXPECT_TRUE(presenter.should_convert(true, false, true));
    presenter.record_converted();
    presenter.record_presented();
    EXPECT_FALSE(presenter.should_process(true, false, false));
  }

  TEST(WindowsLocalPresenterTimingTest, BusyFinalPresentRetriesWithoutReconverting) {
    local_presenter_retry_state_t presenter;
    presenter.observe_source();
    presenter.record_converted();
    for (int busy_flip = 0; busy_flip < 3; ++busy_flip) {
      const capture_wait_policy_t wait {presenter.should_process(true, false, false)};
      EXPECT_LE(wait.source_timeout() + wait.idle_backoff(), 6ms);
      EXPECT_FALSE(presenter.should_convert(true, false, false));
      EXPECT_TRUE(presenter.presentation_pending());
    }
    presenter.record_presented();
    EXPECT_FALSE(presenter.should_process(true, false, false));
  }

  TEST(WindowsLocalPresenterTimingTest, IdleAndRemoteCaptureRetainLongWaitAndLockYield) {
    local_presenter_retry_state_t presenter;
    const capture_wait_policy_t idle {presenter.should_process(true, false, false)};
    const capture_wait_policy_t remote {};
    for (const auto &wait : {idle, remote}) {
      EXPECT_EQ(wait.source_timeout(), 200ms);
      EXPECT_EQ(wait.idle_backoff(), 10ms);
      EXPECT_EQ(wait.pacing_sleep(20ms), 20ms);
      EXPECT_TRUE(wait.retry_after_pacing_timeout());
    }
  }

  TEST(WindowsLocalPresenterTimingTest, ReadyNotificationAndNewSourceRearmBoundedWork) {
    local_presenter_retry_state_t presenter;
    presenter.record_presented();
    EXPECT_TRUE(presenter.should_process(true, true, false));
    EXPECT_TRUE(presenter.should_convert(true, true, false));
    presenter.observe_source();
    EXPECT_TRUE(presenter.should_process(true, false, false));
    EXPECT_TRUE(presenter.should_convert(true, false, false));
    EXPECT_FALSE(presenter.should_process(false, true, true));
    const capture_wait_policy_t wait {true};
    EXPECT_EQ(wait.pacing_sleep(-1ms), -1ms);
    EXPECT_EQ(wait.pacing_sleep(1ms), 1ms);
  }

  TEST(WindowsLocalPresenterTimingTest, MovingCursorDuringPendingWorkDoesNotAccumulateFutureSleep) {
    constexpr auto frame_interval = std::chrono::nanoseconds {1s} / 60;
    const capture_wait_policy_t pending {true};
    const capture_wait_policy_t idle {};
    auto now = std::chrono::steady_clock::time_point {};
    auto group_start = now;
    std::uint32_t group_frames = 1;

    // A continuously moving cursor can satisfy every short source probe before a 60 Hz frame
    // is due. After one second of these successful probes, completing depth must not reveal
    // seconds of nominal frame intervals counted ahead of the actual capture timeline.
    for (int cursor_update = 0; cursor_update < 200; ++cursor_update) {
      const auto requested_sleep = group_start + group_frames * frame_interval - now;
      now += pending.pacing_sleep(requested_sleep);
      if (pending.rebase_after_pacing_snapshot(requested_sleep)) {
        group_start = now;
        group_frames = 1;
      } else {
        ++group_frames;
      }
    }

    EXPECT_EQ(now.time_since_epoch(), 1s);
    EXPECT_EQ(idle.pacing_sleep(group_start + group_frames * frame_interval - now), frame_interval);
  }

  TEST(WindowsLocalPresenterTimingTest, OnlyShortenedLocalPacingProbesRebaseTheGroup) {
    const capture_wait_policy_t pending {true};
    EXPECT_TRUE(pending.rebase_after_pacing_snapshot(16ms));
    EXPECT_TRUE(pending.rebase_after_pacing_snapshot(5ms + 1ns));
    EXPECT_FALSE(pending.rebase_after_pacing_snapshot(5ms));
    EXPECT_FALSE(pending.rebase_after_pacing_snapshot(1ms));
    EXPECT_FALSE(pending.rebase_after_pacing_snapshot(-1ms));

    // Remote capture and local idle capture preserve their original fractional-rate group.
    const capture_wait_policy_t ordinary {};
    EXPECT_FALSE(ordinary.rebase_after_pacing_snapshot(16ms));
    EXPECT_FALSE(ordinary.rebase_after_pacing_snapshot(1s));
  }
}  // namespace
