#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
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
}  // namespace
