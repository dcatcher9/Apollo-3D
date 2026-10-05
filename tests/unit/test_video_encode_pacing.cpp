#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
#include <vector>
#include <src/video.h>
#include <src/video_encode_pacing.h>

namespace {
  using namespace std::chrono_literals;

  // Model the image-event boundary with a deterministic clock. A depth completion does not
  // signal this event: only a new captured image can interrupt its wait, as in encode_run().
  struct image_event {
    std::chrono::nanoseconds now {};
    std::optional<std::chrono::nanoseconds> next_image;
    bool running = true;
    int polls = 0;

    std::optional<int> pop(std::chrono::nanoseconds wait) {
      ++polls;
      if (!running) {
        return std::nullopt;
      }
      if (next_image && *next_image <= now + wait) {
        now = std::max(now, *next_image);
        next_image.reset();
        return 42;
      }
      now += wait;
      return std::nullopt;
    }
  };

  TEST(RemoteEncodeCompletionTest, LastFrameCompletionDoesNotWaitForIdleHeartbeat) {
    image_event images;
    constexpr auto frame_interval = 16666666ns;
    constexpr auto idle_interval = 83333333ns;
    constexpr auto gpu_ready = 20ms;
    bool pending = true;
    while (pending) {
      ASSERT_FALSE(video::detail::wait_for_encode_image(
        images, idle_interval, frame_interval, true, false, pending
      ));
      pending = images.now < gpu_ready;
      ASSERT_LE(images.polls, 2);
    }
    EXPECT_EQ(images.now, 2 * frame_interval);
    EXPECT_LT(images.now, idle_interval);

    // Once the completion is consumed, there is no longer a reason to encode at stream cadence.
    video::detail::wait_for_encode_image(images, idle_interval, frame_interval, true, false, pending);
    EXPECT_EQ(images.now, 2 * frame_interval + idle_interval);
  }

  TEST(RemoteEncodeCompletionTest, NewCaptureInterruptsPendingWait) {
    image_event images;
    images.next_image = 3ms;
    EXPECT_EQ(video::detail::wait_for_encode_image(images, 83ms, 16ms, true, false, true), 42);
    EXPECT_EQ(images.now, 3ms);
  }

  TEST(RemoteEncodeCompletionTest, ReadyPipelineNeedsARealRetainedSource) {
    image_event images;
    video::detail::wait_for_encode_image(images, 83ms, 16ms, false, true, true);
    EXPECT_EQ(images.now, 83ms);
    video::detail::wait_for_encode_image(images, 83ms, 16ms, true, true, true);
    EXPECT_EQ(images.now, 83ms);
  }

  TEST(RemoteEncodeCompletionTest, BusyInferenceCannotSpinOrSuppressFasterKeepalives) {
    image_event images;
    for (int poll = 0; poll < 8; ++poll) {
      video::detail::wait_for_encode_image(images, 83ms, 0ns, true, false, true);
    }
    EXPECT_EQ(images.now, 8ms);
    video::detail::wait_for_encode_image(images, 2ms, 16ms, true, false, true);
    EXPECT_EQ(images.now, 10ms);
  }

  TEST(RemoteEncodeCompletionTest, CaptureShutdownReturnsWithoutWaitingOrProducingAnImage) {
    image_event images;
    images.running = false;
    EXPECT_FALSE(video::detail::wait_for_encode_image(images, 83ms, 16ms, true, false, true));
    EXPECT_EQ(images.now, 0ns);
  }

  struct captured_source {
    int pixels;
    std::chrono::steady_clock::time_point captured_at;
  };
  using source_owner = video::detail::latest_encode_source_t<std::shared_ptr<captured_source>>;

  std::chrono::steady_clock::time_point at(std::chrono::nanoseconds elapsed) {
    return std::chrono::steady_clock::time_point {elapsed};
  }

  TEST(RemoteEncodeProviderPacingTest, EncodeCostIsIncludedInThePresentationInterval) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    constexpr auto interval = 13888888ns; // 72 Hz
    constexpr auto encode_cost = 6700us;
    auto next_target = at(interval);
    images.now = encode_cost;
    for (int frame = 1; frame <= 6; ++frame) {
      ASSERT_FALSE(video::detail::wait_for_encode_image(
        images, 69ms, interval, true, false, true,
        source.remaining_wait(at(images.now), next_target, true)
      ));
      EXPECT_EQ(images.now, frame * interval);
      ASSERT_TRUE(source.due(at(images.now), next_target, false, true));
      const auto schedule = video::detail::select_encode_frame_schedule(
        at(images.now), next_target, interval, interval / 4
      );
      EXPECT_EQ(schedule.presentation_timestamp, at(frame * interval));
      next_target = schedule.next_encode_target;
      source.converted();
      images.now += encode_cost;
      EXPECT_EQ(source.remaining_wait(at(images.now), next_target, true), interval - encode_cost);
      EXPECT_FALSE(source.due(at(images.now), next_target, false, true));
    }
    EXPECT_EQ(images.polls, 6);
    EXPECT_EQ(source.latest()->captured_at, at(0ns)); // No invented source timestamp.
  }

  TEST(RemoteEncodeProviderPacingTest, EarlyCaptureSharesTheRetainedProviderDeadline) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    images.now = 6ms;
    images.next_image = 8ms;
    auto next_target = at(16ms);
    ASSERT_EQ(video::detail::wait_for_encode_image(
      images, 80ms, 16ms, true, false, true,
      source.remaining_wait(at(images.now), next_target, true)
    ), 42);
    source.observe(std::make_shared<captured_source>(2, at(images.now)));
    EXPECT_FALSE(source.due(at(images.now), next_target));
    ASSERT_FALSE(video::detail::wait_for_encode_image(
      images, 80ms, 16ms, true, false, true,
      source.remaining_wait(at(images.now), next_target, true)
    ));
    EXPECT_EQ(images.now, 16ms); // Not 8+16: no second full-frame wait.
    ASSERT_TRUE(source.due(at(images.now), next_target));
    const auto schedule = video::detail::select_encode_frame_schedule(at(images.now), next_target, 16ms, 4ms);
    next_target = schedule.next_encode_target;
    source.converted();
    images.now += 6ms;
    EXPECT_EQ(source.remaining_wait(at(images.now), next_target, true), 10ms);
    EXPECT_FALSE(source.pending());
  }

  TEST(RemoteEncodeProviderPacingTest, OverduePollRebasesAndCannotBusyLoop) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    images.now = 40ms; // Previous encode exceeded the interval.
    auto next_target = at(16ms);
    EXPECT_EQ(source.remaining_wait(at(images.now), next_target, true), 0ns);
    EXPECT_TRUE(source.due(at(images.now), next_target, false, true));
    const auto schedule = video::detail::select_encode_frame_schedule(at(images.now), next_target, 16ms, 4ms);
    EXPECT_EQ(schedule.presentation_timestamp, at(40ms));
    next_target = schedule.next_encode_target;
    source.converted();
    EXPECT_FALSE(source.due(at(images.now), next_target, false, true));
    EXPECT_EQ(source.remaining_wait(at(images.now), next_target, true), 16ms);
  }

  TEST(RemoteEncodeProviderPacingTest, FasterHeartbeatDoesNotAdvanceProviderAndOptOutIsUnchanged) {
    source_owner source;
    EXPECT_FALSE(source.remaining_wait(at(0ns), at(16ms), true));
    EXPECT_FALSE(source.due(at(20ms), at(16ms), true, true));
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    images.now = 6ms;
    const auto next_target = at(16ms);
    ASSERT_FALSE(video::detail::wait_for_encode_image(
      images, 2ms, 16ms, true, false, true,
      source.remaining_wait(at(images.now), next_target, true)
    ));
    EXPECT_EQ(images.now, 8ms);
    EXPECT_FALSE(source.due(at(images.now), next_target, false, true));
    EXPECT_EQ(source.remaining_wait(at(images.now), next_target, true), 8ms);
    EXPECT_TRUE(source.due(at(images.now), next_target, true, true)); // IDR may service now.
    EXPECT_FALSE(source.remaining_wait(at(images.now), next_target));
    EXPECT_FALSE(source.due(at(20ms), next_target));
    video::detail::wait_for_encode_image(images, 80ms, 16ms, true, false, true);
    EXPECT_EQ(images.now, 24ms); // Existing Host AI completion wait remains cadence-sized.
  }

  TEST(RemoteEncodeProviderPacingTest, NonpollingBackendDoesNotSpinAtExpiredProviderDeadline) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    image_event images;
    images.now = 40ms;
    const auto expired_target = at(16ms);
    constexpr bool independent_provider = true;
    constexpr bool conversion_poll_pending = false;
    for (int heartbeat = 0; heartbeat < 3; ++heartbeat) {
      const auto wait = source.remaining_wait(
        at(images.now),
        expired_target,
        independent_provider && conversion_poll_pending
      );
      EXPECT_FALSE(wait);
      EXPECT_FALSE(video::detail::wait_for_encode_image(images, 80ms, 16ms, true, false, conversion_poll_pending, wait));
    }
    EXPECT_EQ(images.now, 280ms);

    // Ordinary pending capture work still has its deadline, even without provider polling.
    source.observe(std::make_shared<captured_source>(2, at(images.now)));
    EXPECT_EQ(source.remaining_wait(at(images.now), expired_target, independent_provider && conversion_poll_pending), 0ns);
    EXPECT_TRUE(source.due(at(images.now), expired_target));
  }

  TEST(RemoteEncodeProviderPacingTest, UnchangedMetadataCaptureNeitherSchedulesNorCancelsConversion) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    // Metadata for pixels a live export owns, with an unchanged cursor: retained, not converted.
    source.observe(std::make_shared<captured_source>(2, at(1ms)), false);
    EXPECT_EQ(source.latest()->pixels, 2);
    EXPECT_FALSE(source.pending());
    EXPECT_FALSE(source.remaining_wait(at(1ms), at(16ms)));
    EXPECT_FALSE(source.due(at(20ms), at(16ms)));
    // A conversion still owed for an earlier capture survives a later unchanged one, which then
    // supplies the newest cursor metadata.
    source.observe(std::make_shared<captured_source>(3, at(2ms)));
    source.observe(std::make_shared<captured_source>(4, at(3ms)), false);
    EXPECT_TRUE(source.pending());
    EXPECT_EQ(source.latest()->pixels, 4);
  }

  TEST(RemoteEncodeProviderPacingTest, KeepaliveIsMeasuredFromTheLastEncode) {
    EXPECT_EQ(video::detail::provider_keepalive_wait(at(10ms), at(0ns), 55ms), 45ms);
    EXPECT_EQ(video::detail::provider_keepalive_wait(at(55ms), at(0ns), 55ms), 0ns);
    EXPECT_EQ(video::detail::provider_keepalive_wait(at(80ms), at(0ns), 55ms), 0ns);
  }

  // The provider wake path of encode_run(): an export is converted when it is ready, or once the
  // time reaches its target minus the variation threshold, and a newer export supersedes one that
  // is still waiting. Returns the wait of each converted export.
  std::vector<std::chrono::nanoseconds> woken_provider_waits(
    std::chrono::nanoseconds game_interval,
    std::chrono::nanoseconds stream_interval,
    std::chrono::nanoseconds phase,
    int exports
  ) {
    const auto threshold = stream_interval / 4;
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(0ns)));
    source.converted();
    auto target = at(0ns);
    std::chrono::nanoseconds now {};
    std::optional<std::chrono::nanoseconds> ready;
    std::vector<std::chrono::nanoseconds> waits;
    int next = 0;
    while (next < exports || ready) {
      const auto arrival = phase + next * game_interval;
      if (!ready) {
        now = arrival;
        ready = arrival;
        ++next;
        continue;
      }
      const auto poll_target = target - threshold;
      const auto due_at = std::max(now, poll_target.time_since_epoch());
      if (next < exports && arrival <= due_at) {
        now = arrival;
        ready = arrival;  // The waiting export is superseded, never queued.
        ++next;
        continue;
      }
      now = due_at;
      EXPECT_TRUE(source.due(at(now), poll_target, false, true));
      EXPECT_EQ(source.remaining_wait(at(now), poll_target, true), 0ns);
      const auto schedule = video::detail::select_encode_frame_schedule(at(now), target, stream_interval, threshold);
      EXPECT_GE(schedule.next_encode_target - target, stream_interval);  // Never above stream rate.
      target = schedule.next_encode_target;
      source.converted();
      waits.push_back(now - *ready);
      ready.reset();
    }
    return waits;
  }

  TEST(RemoteEncodeProviderPacingTest, WokenProviderFramesConvertWhenReadyAtOrBelowStreamRate) {
    constexpr auto stream = 11111111ns;  // 90 Hz
    for (const std::chrono::nanoseconds phase : {0ns, 3000000ns, 7000000ns, 10000000ns}) {
      // A game at or below the stream rate is converted the moment each export is ready, in any
      // phase relative to the stream's earlier deadlines.
      for (const std::chrono::nanoseconds game : {stream, 16666666ns, 33333333ns}) {
        const auto waits = woken_provider_waits(game, stream, phase, 60);
        ASSERT_EQ(waits.size(), 60u);
        for (const auto wait : waits) {
          EXPECT_EQ(wait, 0ns);
        }
      }
    }
  }

  TEST(RemoteEncodeProviderPacingTest, FasterGameIsCappedToTheStreamRateWithBoundedWaits) {
    constexpr auto stream = 11111111ns;  // 90 Hz
    constexpr auto game = 8333333ns;  // 120 fps
    const auto waits = woken_provider_waits(game, stream, 1ms, 120);  // 1 s of exports
    // At most one conversion per stream interval on average. A converted export never waited a
    // whole game frame: a newer export would have superseded it.
    EXPECT_LE(waits.size(), 91u);
    EXPECT_GE(waits.size(), 85u);
    std::chrono::nanoseconds total {};
    for (const auto wait : waits) {
      EXPECT_LT(wait, game);
      total += wait;
    }
    EXPECT_LT(total / static_cast<int>(waits.size()), stream / 2);
  }

  TEST(RemoteEncodePendingSourceTest, FinalEarlySourceSurvivesUntilItsPresentationDeadline) {
    source_owner source;
    image_event images;
    images.now = 5ms;
    const auto next_target = at(16666666ns);
    auto final_image = std::make_shared<captured_source>(17, at(5ms));
    const std::weak_ptr<captured_source> retained = final_image;
    source.observe(std::move(final_image));
    ASSERT_FALSE(final_image);
    ASSERT_FALSE(retained.expired());
    ASSERT_TRUE(source.pending());
    EXPECT_FALSE(source.due(at(images.now), next_target));

    // No subsequent capture arrives. The real image-event wait reaches the source deadline,
    // rather than waiting 83 ms and re-encoding the older persistent encoder input.
    EXPECT_FALSE(video::detail::wait_for_encode_image(
      images, 83333333ns, 16666666ns, true, false, false,
      source.remaining_wait(at(images.now), next_target)
    ));
    EXPECT_EQ(images.now, 16666666ns);
    ASSERT_TRUE(source.due(at(images.now), next_target));
    ASSERT_EQ(source.latest()->pixels, 17);
    EXPECT_EQ(source.latest()->captured_at, at(5ms));

    source.converted();
    EXPECT_FALSE(source.pending());
    EXPECT_FALSE(source.due(at(images.now), next_target));
    EXPECT_FALSE(source.remaining_wait(at(images.now), next_target));
    EXPECT_FALSE(retained.expired());  // Still available to depth completion / same-display rebuild.

    video::detail::wait_for_encode_image(
      images, 83333333ns, 16666666ns, true, false, false,
      source.remaining_wait(at(images.now), next_target)
    );
    EXPECT_EQ(images.now, 99999999ns);
    EXPECT_EQ(images.polls, 2);  // One due-source wait, then the ordinary idle heartbeat.
  }

  TEST(RemoteEncodePendingSourceTest, ANewCaptureReplacesTheDeferredOwnerWithoutGrowingAQueue) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(1, at(3ms)));
    const std::weak_ptr<captured_source> superseded = source.latest();
    image_event images;
    images.now = 3ms;
    images.next_image = 8ms;
    EXPECT_EQ(video::detail::wait_for_encode_image(
      images, 83ms, 16ms, true, false, false, source.remaining_wait(at(images.now), at(16ms))
    ), 42);
    EXPECT_EQ(images.now, 8ms);
    source.observe(std::make_shared<captured_source>(2, at(8ms)));
    EXPECT_TRUE(superseded.expired());
    EXPECT_EQ(source.latest()->pixels, 2);
    EXPECT_EQ(source.remaining_wait(at(images.now), at(16ms)), 8ms);
    EXPECT_TRUE(source.due(at(16ms), at(16ms)));
  }

  TEST(RemoteEncodePendingSourceTest, StaleSourceTimestampCannotStarveDuePresentation) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(9, at(1ms)));
    EXPECT_TRUE(source.due(at(40ms), at(16ms)));
    EXPECT_EQ(source.remaining_wait(at(40ms), at(16ms)), 0ns);
    EXPECT_EQ(source.latest()->captured_at, at(1ms));
    source.converted();
    EXPECT_FALSE(source.due(at(41ms), at(16ms)));
  }

  TEST(RemoteEncodePendingSourceTest, IdrUsesPendingSourceImmediatelyAndDoesNotInventOne) {
    source_owner source;
    EXPECT_FALSE(source.due(at(2ms), at(16ms), true));
    source.observe(std::make_shared<captured_source>(3, at(2ms)));
    EXPECT_FALSE(source.due(at(2ms), at(16ms)));
    EXPECT_TRUE(source.due(at(2ms), at(16ms), true));
    source.converted();
    EXPECT_FALSE(source.due(at(2ms), at(16ms), true));
    ASSERT_TRUE(source.latest());  // An ordinary IDR may still encode the existing input surface.
  }

  TEST(RemoteEncodePendingSourceTest, FasterConfiguredHeartbeatDoesNotRetirePendingSource) {
    source_owner source;
    source.observe(std::make_shared<captured_source>(4, at(5ms)));
    image_event images;
    images.now = 5ms;
    video::detail::wait_for_encode_image(
      images, 2ms, 16ms, true, false, false, source.remaining_wait(at(images.now), at(16ms))
    );
    EXPECT_EQ(images.now, 7ms);
    EXPECT_FALSE(source.due(at(images.now), at(16ms)));
    EXPECT_TRUE(source.pending());
    EXPECT_EQ(source.remaining_wait(at(images.now), at(16ms)), 9ms);
  }

  TEST(RemoteEncodePendingSourceTest, SameDisplayHandoffTransfersNewestPendingOwnerExactlyOnce) {
    source_owner retiring;
    retiring.observe(std::make_shared<captured_source>(5, at(5ms)));
    const std::weak_ptr<captured_source> image = retiring.latest();
    source_owner replacement;
    replacement.observe(retiring.release());
    EXPECT_FALSE(retiring.latest());
    EXPECT_FALSE(retiring.pending());
    EXPECT_TRUE(replacement.pending());
    EXPECT_EQ(replacement.latest()->pixels, 5);
    EXPECT_FALSE(image.expired());
    EXPECT_FALSE(retiring.release());
  }

  TEST(RemoteEncodePendingSourceTest, ShutdownDoesNotWaitOrConvertAndDropsRetainedOwnerOnExit) {
    std::weak_ptr<captured_source> image;
    {
      source_owner source;
      source.observe(std::make_shared<captured_source>(6, at(5ms)));
      image = source.latest();
      image_event images;
      images.now = 5ms;
      images.running = false;
      EXPECT_FALSE(video::detail::wait_for_encode_image(
        images, 83ms, 16ms, true, false, false, source.remaining_wait(at(images.now), at(16ms))
      ));
      EXPECT_EQ(images.now, 5ms);
      EXPECT_TRUE(source.pending());
      EXPECT_FALSE(image.expired());
    }
    EXPECT_TRUE(image.expired());
  }

  TEST(RemoteEncodePendingSourceTest, FailedEncoderReturnsStaticDesktopToReplacement) {
    safe::event_t<std::shared_ptr<captured_source>> images;
    source_owner retiring;
    retiring.observe(std::make_shared<captured_source>(7, at(5ms)));
    retiring.converted();  // No more desktop changes arrive before the encoder times out.
    retiring.return_for_rebuild(images, false, false);
    EXPECT_FALSE(retiring.latest());

    source_owner replacement;
    replacement.observe(images.pop(0ms));
    ASSERT_TRUE(replacement.latest());
    EXPECT_EQ(replacement.latest()->pixels, 7);
    EXPECT_EQ(replacement.latest()->captured_at, at(5ms));
    EXPECT_TRUE(replacement.pending());
    EXPECT_FALSE(images.peek());
  }

  TEST(RemoteEncodePendingSourceTest, FailureHandoffPreservesNewerQueuedCapture) {
    safe::event_t<std::shared_ptr<captured_source>> images;
    source_owner retiring;
    retiring.observe(std::make_shared<captured_source>(7, at(5ms)));
    const std::weak_ptr<captured_source> old_image = retiring.latest();
    images.raise(std::make_shared<captured_source>(8, at(6ms)));

    retiring.return_for_rebuild(images, false, false);
    EXPECT_TRUE(old_image.expired());
    const auto next = images.pop(0ms);
    ASSERT_TRUE(next);
    EXPECT_EQ(next->pixels, 8);
    EXPECT_EQ(next->captured_at, at(6ms));
  }

  TEST(RemoteEncodePendingSourceTest, FailureHandoffDoesNotKeepAnOldDisplayAliveAfterCancellation) {
    for (int cancellation = 0; cancellation < 3; ++cancellation) {
      safe::event_t<std::shared_ptr<captured_source>> images;
      std::weak_ptr<captured_source> old_image;
      {
        source_owner retiring;
        retiring.observe(std::make_shared<captured_source>(7, at(5ms)));
        old_image = retiring.latest();
        if (cancellation == 0) {
          images.stop();
        }
        retiring.return_for_rebuild(images, cancellation == 1, cancellation == 2);
        EXPECT_FALSE(images.try_pop());
      }
      EXPECT_TRUE(old_image.expired());
    }
  }
}  // namespace
