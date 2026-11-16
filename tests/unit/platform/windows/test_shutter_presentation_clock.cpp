#include "src/platform/windows/shutter_presentation_clock.h"

#include <gtest/gtest.h>
#include <limits>
#include <vector>

using platf::dxgi::detail::shutter_presentation_clock_t;

TEST(ShutterRecoveryBudget, AllowsAFewTransientLossesButBoundsARepeatedFailureLoop) {
  using budget_t = platf::dxgi::detail::shutter_recovery_budget_t;
  using namespace std::chrono_literals;
  budget_t budget;
  const budget_t::clock::time_point start {};
  for (int attempt = 0; attempt < static_cast<int>(budget_t::maximum_attempts); ++attempt) {
    EXPECT_TRUE(budget.begin(start + attempt * 100ms));
  }
  EXPECT_FALSE(budget.begin(start + 9999ms));
  EXPECT_TRUE(budget.begin(start + 10s));
  // Expiring the first attempt does not expire the newer second attempt.
  EXPECT_FALSE(budget.begin(start + 10099ms));
  EXPECT_TRUE(budget.begin(start + 10100ms));
}

TEST(ShutterRecoveryBudget, ClockRegressionCannotResetTheFailureLimit) {
  using budget_t = platf::dxgi::detail::shutter_recovery_budget_t;
  using namespace std::chrono_literals;
  budget_t budget;
  const auto start = budget_t::clock::time_point {} + 10s;
  EXPECT_TRUE(budget.begin(start));
  EXPECT_FALSE(budget.begin(start - 1ms));
  for (std::size_t attempt = 1; attempt < budget_t::maximum_attempts; ++attempt) {
    EXPECT_TRUE(budget.begin(start));
  }
  EXPECT_FALSE(budget.begin(start));
}

TEST(ShutterPresentationClock, HoldsOneStereoPairThroughRepeatedEyeScanouts) {
  shutter_presentation_clock_t clock(2);
  EXPECT_TRUE(clock.starts_pair());
  EXPECT_TRUE(clock.left_eye());
  ASSERT_TRUE(clock.observe_displayed(100));
  EXPECT_FALSE(clock.starts_pair());
  EXPECT_FALSE(clock.left_eye());
  ASSERT_TRUE(clock.observe_displayed(102));
  EXPECT_TRUE(clock.starts_pair());
  EXPECT_TRUE(clock.left_eye());
}

TEST(ShutterPresentationClock, HeldRefreshesAdvanceOneExplicitEyeWithoutUsingRefreshParity) {
  shutter_presentation_clock_t clock(1);
  ASSERT_TRUE(clock.observe_displayed(42));
  EXPECT_FALSE(clock.left_eye());
  ASSERT_TRUE(clock.observe_displayed(44));  // Two refreshes still mean one newly displayed eye.
  EXPECT_TRUE(clock.left_eye());
  ASSERT_TRUE(clock.observe_displayed(47));
  EXPECT_FALSE(clock.left_eye());
  ASSERT_TRUE(clock.observe_displayed(51));
  EXPECT_TRUE(clock.left_eye());
}

TEST(ShutterPresentationClock, ShortRepeatedBackwardsAndUnboundedHoldsDoNotAdvancePhase) {
  for (unsigned repeats : {1u, 2u, 3u, 4u}) {
    for (const auto delta : {0u, repeats - 1u, 4u * repeats + 1u, UINT32_MAX}) {
      SCOPED_TRACE(repeats);
      SCOPED_TRACE(delta);
      shutter_presentation_clock_t clock(repeats);
      ASSERT_TRUE(clock.observe_displayed(100));
      EXPECT_FALSE(clock.observe_displayed(100u + delta));
      EXPECT_FALSE(clock.left_eye());
      // A rejected boundary did not replace the last accepted refresh or consume an eye.
      ASSERT_TRUE(clock.observe_displayed(100u + repeats));
      EXPECT_TRUE(clock.left_eye());
    }
  }
}

TEST(ShutterPresentationClock, RefreshCounterWrapPreservesPhase) {
  shutter_presentation_clock_t clock(2);
  ASSERT_TRUE(clock.observe_displayed(UINT32_MAX - 1));
  ASSERT_TRUE(clock.observe_displayed(0));
  EXPECT_TRUE(clock.left_eye());
}

TEST(ShutterPresentationClock, InvalidRepeatCountCannotStart) {
  EXPECT_FALSE(shutter_presentation_clock_t(0).observe_displayed(1));
  EXPECT_FALSE(shutter_presentation_clock_t(5).observe_displayed(1));
}

namespace {
  using namespace std::chrono_literals;
  namespace shutter = platf::dxgi::detail;

  struct feedback_probe_t {
    using clock = std::chrono::steady_clock;
    clock::time_point now {};
    clock::time_point displayed_at = now + 6061us;
    clock::time_point feedback_at = displayed_at + 150us;
    std::uint32_t present_id = 41;
    std::uint32_t refresh = 100;
    unsigned queries = 0;
    unsigned flushes = 0;
    bool cancelled = false;
    std::vector<clock::duration> waits;

    shutter::shutter_feedback_wait_io_t io() {
      return {
        .now = [&]() {
          return now;
        },
        .cancelled = [&]() {
          return cancelled;
        },
        .query = [&]() {
          ++queries;
          return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::available, now >= feedback_at ? present_id : present_id - 1u, refresh};
        },
        .flush = [&]() {
          ++flushes;
          // The real flush can wait for this actual present, while its asynchronous metadata
          // still arrives a fraction of a refresh after the drawing has completed.
          now = std::max(now, displayed_at);
          return true;
        },
        .wait_until = [&](auto deadline) {
          waits.push_back(deadline - now);
          now = deadline;
          return true;
        },
      };
    }
  };
}  // namespace

TEST(ShutterPresentFeedback, LateMetadataDoesNotSpendAnotherRefreshForSingleOrRepeatedEyes) {
  for (unsigned repeats : {1u, 2u}) {
    SCOPED_TRACE(repeats);
    feedback_probe_t probe;
    auto io = probe.io();
    shutter_presentation_clock_t eye_clock(repeats);
    for (unsigned frame = 0; frame < 6; ++frame) {
      SCOPED_TRACE(frame);
      probe.present_id = 41u + frame;
      probe.refresh = 100u + frame * repeats;
      probe.displayed_at = feedback_probe_t::clock::time_point {} + 6061us * (1u + frame * repeats);
      probe.feedback_at = probe.displayed_at + 150us;
      const auto confirmed = shutter::await_shutter_present(io, probe.present_id, frame == 0);
      ASSERT_EQ(confirmed.status, shutter::shutter_feedback_wait_status_e::displayed);
      EXPECT_LT(probe.now, probe.displayed_at + 6061us);
      EXPECT_EQ(eye_clock.left_eye(), frame % 2 == 0);
      ASSERT_TRUE(eye_clock.observe_displayed(confirmed.feedback.refresh_count));
    }
    EXPECT_EQ(probe.flushes, 1u);  // Only the unsignalled acquisition call may flush.
    EXPECT_GT(probe.waits.size(), 6u);
    for (auto wait : probe.waits) {
      EXPECT_EQ(wait, 200us);
    }
  }
}

TEST(ShutterPresentFeedback, AnEmptySampleForTheRequestedPresentIsPolledUntilItsRefreshDataArrives) {
  feedback_probe_t probe;
  probe.now = probe.feedback_at;
  auto io = probe.io();
  unsigned empty_left = 3;
  io.query = [&]() {
    ++probe.queries;
    if (empty_left) {
      --empty_left;
      return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::available, probe.present_id, 0, 0, 0};
    }
    return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::available, probe.present_id, probe.refresh, probe.refresh, 123456};
  };
  const auto result = shutter::await_shutter_present(io, probe.present_id, false);
  EXPECT_EQ(result.status, shutter::shutter_feedback_wait_status_e::displayed);
  EXPECT_EQ(result.feedback.refresh_count, probe.refresh);
  EXPECT_EQ(result.feedback.sync_qpc, 123456);
  EXPECT_EQ(probe.queries, 4u);
  EXPECT_EQ(probe.waits.size(), 3u);
  EXPECT_EQ(probe.flushes, 0u);
}

TEST(ShutterPresentFeedback, APersistentlyEmptySampleIsReturnedAfterABoundedPollForTheCallerToReject) {
  feedback_probe_t probe;
  probe.now = probe.feedback_at;
  auto io = probe.io();
  io.query = [&]() {
    ++probe.queries;
    return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::available, probe.present_id, 0, 0, 0};
  };
  const auto started = probe.now;
  const auto result = shutter::await_shutter_present(io, probe.present_id, false);
  EXPECT_EQ(result.status, shutter::shutter_feedback_wait_status_e::displayed);
  EXPECT_EQ(result.feedback.sync_qpc, 0);
  EXPECT_EQ(probe.queries, 9u);
  EXPECT_EQ(probe.now - started, 1600us);  // Eight 200 us polls, far inside the 250 ms timeout.
}

TEST(ShutterPresentFeedback, ReadyFeedbackDoesNotFlushOrWaitAgain) {
  feedback_probe_t probe;
  probe.now = probe.feedback_at;
  const auto result = shutter::await_shutter_present(probe.io(), probe.present_id, false);
  EXPECT_EQ(result.status, shutter::shutter_feedback_wait_status_e::displayed);
  EXPECT_EQ(probe.queries, 1u);
  EXPECT_EQ(probe.flushes, 0u);
  EXPECT_TRUE(probe.waits.empty());
}

TEST(ShutterPresentFeedback, PresentIdWrapTreatsPreviousMaximumAsPending) {
  feedback_probe_t probe;
  probe.present_id = 0;
  const auto result = shutter::await_shutter_present(probe.io(), 0, false);
  EXPECT_EQ(result.status, shutter::shutter_feedback_wait_status_e::displayed);
  EXPECT_EQ(result.feedback.present_count, 0u);
  EXPECT_EQ(probe.flushes, 0u);
}

TEST(ShutterPresentFeedback, InitialDisjointStartsASequenceButLiveDisjointStopsIt) {
  feedback_probe_t initial;
  auto io = initial.io();
  const auto query = io.query;
  io.query = [&]() {
    if (initial.queries == 0) {
      ++initial.queries;
      return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::disjoint};
    }
    return query();
  };
  EXPECT_EQ(shutter::await_shutter_present(io, initial.present_id, true).status, shutter::shutter_feedback_wait_status_e::displayed);
  EXPECT_EQ(initial.flushes, 1u);

  feedback_probe_t live;
  io = live.io();
  io.query = []() {
    return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::disjoint};
  };
  EXPECT_EQ(shutter::await_shutter_present(io, live.present_id, false).status, shutter::shutter_feedback_wait_status_e::unsupported);
  EXPECT_EQ(live.flushes, 0u);
}

TEST(ShutterPresentFeedback, FuturePresentAndUnsupportedFeedbackCannotAuthorizeAnEye) {
  feedback_probe_t probe;
  auto io = probe.io();
  io.query = [&]() {
    return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::available, probe.present_id + 1u, probe.refresh};
  };
  EXPECT_EQ(shutter::await_shutter_present(io, probe.present_id, false).status, shutter::shutter_feedback_wait_status_e::unsupported);
  io.query = []() {
    return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::unsupported};
  };
  EXPECT_EQ(shutter::await_shutter_present(io, probe.present_id, false).status, shutter::shutter_feedback_wait_status_e::unsupported);
  EXPECT_EQ(probe.flushes, 0u);
}

TEST(ShutterPresentFeedback, UnboundedHeldRefreshStillFailsAfterMetadataBecomesAvailable) {
  feedback_probe_t probe;
  probe.refresh = 105;
  shutter_presentation_clock_t eye_clock(1);
  ASSERT_TRUE(eye_clock.observe_displayed(100));
  const auto result = shutter::await_shutter_present(probe.io(), probe.present_id, false);
  ASSERT_EQ(result.status, shutter::shutter_feedback_wait_status_e::displayed);
  EXPECT_FALSE(eye_clock.observe_displayed(result.feedback.refresh_count));
  EXPECT_FALSE(eye_clock.left_eye());
}

TEST(ShutterPresentFeedback, StaleLiveFeedbackPollsWithoutFlushingAndStopsAtABoundedDeadline) {
  feedback_probe_t probe;
  auto io = probe.io();
  probe.displayed_at = probe.now;
  io.query = [&]() {
    ++probe.queries;
    return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::available, probe.present_id - 1u, 0, 73, 1234567, 0};
  };
  const auto result = shutter::await_shutter_present(io, probe.present_id, false);
  EXPECT_EQ(result.status, shutter::shutter_feedback_wait_status_e::timeout);
  EXPECT_EQ(result.feedback.present_count, probe.present_id - 1u);
  EXPECT_EQ(result.feedback.sync_refresh_count, 73u);
  EXPECT_EQ(result.feedback.sync_qpc, 1234567);
  EXPECT_EQ(result.feedback.result_code, 0);
  EXPECT_EQ(result.queries, probe.queries);
  EXPECT_EQ(result.flushes, 0u);
  EXPECT_EQ(result.waits, 1250u);
  EXPECT_EQ(probe.now.time_since_epoch(), 250ms);
  EXPECT_EQ(probe.flushes, 0u);
  EXPECT_EQ(probe.waits.size(), 1250u);
}

TEST(ShutterPresentFeedback, UnsupportedFeedbackRetainsTheRawFailureWithoutPublishingOrAuthorizing) {
  feedback_probe_t probe;
  auto io = probe.io();
  constexpr auto dxgi_unsupported = static_cast<std::int32_t>(0x887A0004u);
  io.query = [&]() {
    return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::unsupported, 0, 0, 0, 0, dxgi_unsupported};
  };
  const auto result = shutter::await_shutter_present(io, probe.present_id, true);
  EXPECT_EQ(result.status, shutter::shutter_feedback_wait_status_e::unsupported);
  EXPECT_EQ(result.feedback.result_code, dxgi_unsupported);
  EXPECT_EQ(result.queries, 1u);
  EXPECT_EQ(result.flushes, 0u);
  EXPECT_EQ(result.waits, 0u);
}

TEST(ShutterPresentFeedback, OnlyTimeoutOrRawDisjointCanEnterQuietClockRecovery) {
  using status_e = shutter::shutter_feedback_wait_status_e;
  shutter::shutter_feedback_wait_result_t result {status_e::timeout};
  EXPECT_TRUE(shutter::shutter_feedback_is_timing_interruption(result));
  result.status = status_e::unsupported;
  result.feedback.status = shutter::shutter_feedback_status_e::disjoint;
  EXPECT_TRUE(shutter::shutter_feedback_is_timing_interruption(result));
  // A newer ID and an unsupported HRESULT share the wait status; neither is a disjoint clock.
  for (const auto raw : {shutter::shutter_feedback_status_e::available, shutter::shutter_feedback_status_e::unsupported}) {
    result.feedback.status = raw;
    EXPECT_FALSE(shutter::shutter_feedback_is_timing_interruption(result));
  }
  result.feedback.status = shutter::shutter_feedback_status_e::disjoint;
  for (const auto status : {status_e::displayed, status_e::cancelled, status_e::flush_failed, status_e::wait_failed}) {
    result.status = status;
    EXPECT_FALSE(shutter::shutter_feedback_is_timing_interruption(result));
  }
}

TEST(ShutterPresentFeedback, CancellationInterruptsSubframeWaitAndAvoidsAnyFurtherQuery) {
  feedback_probe_t probe;
  auto io = probe.io();
  io.wait_until = [&](auto) {
    probe.now += 50us;
    probe.cancelled = true;
    return true;
  };
  EXPECT_EQ(shutter::await_shutter_present(io, probe.present_id, false).status, shutter::shutter_feedback_wait_status_e::cancelled);
  EXPECT_EQ(probe.flushes, 0u);
  EXPECT_EQ(probe.queries, 1u);
  const auto queries = probe.queries;
  EXPECT_EQ(shutter::await_shutter_present(io, probe.present_id, false).status, shutter::shutter_feedback_wait_status_e::cancelled);
  EXPECT_EQ(probe.queries, queries);
}

TEST(ShutterPresentFeedback, FailedFlushOrTimerDoesNotAuthorizeAnEye) {
  feedback_probe_t probe;
  auto io = probe.io();
  io.flush = []() {
    return false;
  };
  EXPECT_EQ(shutter::await_shutter_present(io, probe.present_id, true).status, shutter::shutter_feedback_wait_status_e::flush_failed);
  io = probe.io();
  io.wait_until = [](auto) {
    return false;
  };
  EXPECT_EQ(shutter::await_shutter_present(io, probe.present_id, false).status, shutter::shutter_feedback_wait_status_e::wait_failed);
}

TEST(ShutterPresentFeedback, LiveMetadataNeverEntersAFlushThatWouldSpendTheShutterWindow) {
  feedback_probe_t probe;
  auto io = probe.io();
  io.flush = [&]() {
    ++probe.flushes;
    probe.now += 18ms;  // Observed live DwmFlush stall exceeds an entire 120Hz eye interval.
    return true;
  };
  const auto result = shutter::await_shutter_present(io, probe.present_id, false, 10ms);
  ASSERT_EQ(result.status, shutter::shutter_feedback_wait_status_e::displayed);
  EXPECT_EQ(result.feedback.present_count, probe.present_id);
  EXPECT_EQ(probe.flushes, 0u);
  EXPECT_EQ(result.flushes, 0u);
  EXPECT_GE(probe.now, probe.feedback_at);
  EXPECT_LT(probe.now - probe.displayed_at, 1ms);
}

TEST(ShutterPresentFeedback, FlushDependentPublisherCannotFallBackToALiveBlockingFlush) {
  for (bool acquiring : {true, false}) {
    SCOPED_TRACE(acquiring);
    feedback_probe_t probe;
    auto io = probe.io();
    bool published = false;
    io.query = [&]() {
      ++probe.queries;
      return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::available,
        published ? probe.present_id : probe.present_id - 1u, probe.refresh};
    };
    io.flush = [&]() {
      ++probe.flushes;
      published = true;
      return true;
    };
    const auto result = shutter::await_shutter_present(io, probe.present_id, acquiring);
    EXPECT_EQ(result.status, acquiring ? shutter::shutter_feedback_wait_status_e::displayed :
                                        shutter::shutter_feedback_wait_status_e::timeout);
    EXPECT_EQ(probe.flushes, acquiring ? 1u : 0u);
    EXPECT_EQ(published, acquiring);
    if (!acquiring) {
      EXPECT_EQ(probe.now.time_since_epoch(), 250ms);
      EXPECT_NE(result.feedback.present_count, probe.present_id);
    }
  }
}

TEST(ShutterPresentFeedback, ColdFlushReturningPastDeadlineCannotAuthorizeAnEye) {
  feedback_probe_t probe;
  auto io = probe.io();
  io.flush = [&]() {
    ++probe.flushes;
    probe.now += 18ms;
    return true;
  };
  const auto result = shutter::await_shutter_present(io, probe.present_id, true, 10ms);
  EXPECT_EQ(result.status, shutter::shutter_feedback_wait_status_e::timeout);
  EXPECT_EQ(probe.flushes, 1u);
  EXPECT_EQ(probe.queries, 1u);
  EXPECT_NE(result.feedback.present_count, probe.present_id);
}

namespace {
  using tracker_status = shutter::shutter_feedback_tracker_status_e;
  using tracker_t = shutter::shutter_feedback_tracker_t;
  constexpr std::int64_t tracker_frequency = 12000000;
  constexpr std::int64_t tracker_refresh_ticks = 100000;
  constexpr std::int64_t tracker_origin = 10000000;

  shutter::shutter_feedback_t tracker_feedback(
    std::uint32_t id,
    std::uint32_t refresh,
    std::int64_t present_qpc,
    std::int32_t refresh_offset = 0
  ) {
    return {shutter::shutter_feedback_status_e::available, id, refresh,
      refresh - static_cast<std::uint32_t>(refresh_offset),
      present_qpc - refresh_offset * tracker_refresh_ticks, 0};
  }

  void acquire_tracker(tracker_t &tracker, unsigned repeats = 1) {
    ASSERT_EQ(tracker.observe(10, tracker_feedback(10, 1000, tracker_origin), tracker_origin + 1200), tracker_status::acquiring);
    const auto next_qpc = tracker_origin + repeats * tracker_refresh_ticks;
    ASSERT_EQ(tracker.observe(11, tracker_feedback(11, 1000 + repeats, next_qpc), next_qpc + 1200), tracker_status::ready);
    ASSERT_TRUE(tracker.arm());
  }
}  // namespace

TEST(ShutterFeedbackTracker, ArmedTimestampJitterIsToleratedOnlyWithAnExactRefreshCount) {
  // 12 ticks per microsecond: 3600 ticks = 300 us, 7200 ticks = 600 us.
  tracker_t tracker(1, 120000, tracker_frequency, 2000, 100, 2, 500);
  acquire_tracker(tracker);
  const auto late = tracker_origin + 2 * tracker_refresh_ticks + 3600;
  EXPECT_EQ(tracker.observe(12, tracker_feedback(12, 1002, late), late + 1200), tracker_status::accepted);
  // The following timestamp back on the nominal grid makes a 300 us short interval.
  const auto nominal = tracker_origin + 3 * tracker_refresh_ticks;
  EXPECT_EQ(tracker.observe(13, tracker_feedback(13, 1003, nominal), nominal + 1200), tracker_status::accepted);
  const auto too_late = tracker_origin + 4 * tracker_refresh_ticks + 7200;
  EXPECT_EQ(tracker.observe(14, tracker_feedback(14, 1004, too_late), too_late + 1200), tracker_status::rejected);
}

TEST(ShutterFeedbackTracker, ColdAcquisitionKeepsItsStrictCadenceWhenTheArmedToleranceIsWider) {
  tracker_t tracker(1, 120000, tracker_frequency, 2000, 100, 2, 500);
  ASSERT_EQ(tracker.observe(10, tracker_feedback(10, 1000, tracker_origin), tracker_origin + 1200), tracker_status::acquiring);
  const auto late = tracker_origin + tracker_refresh_ticks + 3600;
  EXPECT_EQ(tracker.observe(11, tracker_feedback(11, 1001, late), late + 1200), tracker_status::acquiring);
  EXPECT_FALSE(tracker.arm());
}

TEST(ShutterFeedbackTracker, RequiresTwelveMeasuredIntervalsAndExplicitArmingBeforeAnEye) {
  tracker_t tracker(1, 120000, tracker_frequency, 2000, 100);
  for (unsigned frame = 0; frame < 13; ++frame) {
    const auto present = tracker_origin + frame * tracker_refresh_ticks;
    EXPECT_EQ(tracker.observe(100 + frame, tracker_feedback(100 + frame, 1000 + frame, present), present + 1200),
      frame == 12 ? tracker_status::ready : tracker_status::acquiring);
    EXPECT_FALSE(tracker.fresh_for_emit(present + 1200));
    if (frame < 12) { EXPECT_FALSE(tracker.arm()); }
  }
  ASSERT_TRUE(tracker.arm());
  EXPECT_FALSE(tracker.arm());
  const auto accepted_boundary = tracker.last_present_qpc();
  EXPECT_TRUE(tracker.fresh_for_emit(*accepted_boundary + 1200));
  EXPECT_TRUE(tracker.fresh_for_emit(*accepted_boundary + 2400));
  EXPECT_EQ(tracker.last_present_qpc(), accepted_boundary);
  EXPECT_EQ(tracker.streak(), 13u);
  const auto first_eye = tracker_origin + 13 * tracker_refresh_ticks;
  EXPECT_EQ(tracker.observe(113, tracker_feedback(113, 1013, first_eye), first_eye + 1200), tracker_status::accepted);
}

TEST(ShutterFeedbackTracker, ColdStallsDisjointAndCounterEpochChangesRequireANewStableSequence) {
  tracker_t tracker(1, 120000, tracker_frequency, 2000, 100);
  ASSERT_EQ(tracker.observe(1, tracker_feedback(1, 100, tracker_origin), tracker_origin + 1200), tracker_status::acquiring);
  ASSERT_EQ(tracker.observe(2, tracker_feedback(2, 101, tracker_origin + tracker_refresh_ticks), tracker_origin + tracker_refresh_ticks + 1200), tracker_status::acquiring);
  EXPECT_EQ(tracker.observe(3, tracker_feedback(2, 101, tracker_origin + tracker_refresh_ticks), tracker_origin + 2 * tracker_refresh_ticks), tracker_status::acquiring);
  EXPECT_EQ(tracker.streak(), 0u);
  EXPECT_FALSE(tracker.last_present_qpc());
  EXPECT_EQ(tracker.observe(3, {shutter::shutter_feedback_status_e::disjoint}, tracker_origin + 2 * tracker_refresh_ticks), tracker_status::acquiring);

  for (unsigned frame = 0; frame < 4; ++frame) {
    const auto present = tracker_origin + (frame + 3) * tracker_refresh_ticks;
    tracker.observe(4 + frame, tracker_feedback(4 + frame, 1000 + frame, present), present + 1200);
  }
  ASSERT_EQ(tracker.streak(), 4u);
  // A new refresh counter domain with an otherwise fresh exact ID starts at one, not five.
  for (unsigned frame = 0; frame < 13; ++frame) {
    const auto present = tracker_origin + (frame + 7) * tracker_refresh_ticks;
    EXPECT_EQ(tracker.observe(8 + frame, tracker_feedback(8 + frame, 500 + frame, present), present + 1200),
      frame == 12 ? tracker_status::ready : tracker_status::acquiring);
    EXPECT_EQ(tracker.streak(), frame + 1);
  }
  EXPECT_TRUE(tracker.arm());
}

TEST(ShutterFeedbackTracker, ValidButSkippedIdsOrWrongPhysicalCadenceCannotAccumulateColdSamples) {
  tracker_t tracker(1, 120000, tracker_frequency, 2000, 100);
  tracker.observe(1, tracker_feedback(1, 100, tracker_origin), tracker_origin + 1200);
  const auto second = tracker_origin + tracker_refresh_ticks;
  tracker.observe(3, tracker_feedback(3, 101, second), second + 1200);
  EXPECT_EQ(tracker.streak(), 1u);
  // Counts increment once, but a 60 Hz clock is not a 120 Hz scanout interval.
  const auto third = second + 2 * tracker_refresh_ticks;
  tracker.observe(4, tracker_feedback(4, 102, third), third + 1200);
  EXPECT_EQ(tracker.streak(), 1u);
  const auto fourth = third + tracker_refresh_ticks;
  tracker.observe(5, tracker_feedback(5, 103, fourth), fourth + 1200);
  EXPECT_EQ(tracker.streak(), 2u);
}

TEST(ShutterFeedbackTracker, RepeatedEyesAndBothCounterWrapsPreserveExactContinuity) {
  for (unsigned repeats : {1u, 2u, 3u, 4u}) {
    SCOPED_TRACE(repeats);
    tracker_t tracker(repeats, 120000, tracker_frequency, 2000, 100, 2);
    const auto first_refresh = UINT32_MAX - repeats + 1;
    EXPECT_EQ(tracker.observe(UINT32_MAX, tracker_feedback(UINT32_MAX, first_refresh, tracker_origin), tracker_origin + 1200), tracker_status::acquiring);
    const auto second = tracker_origin + repeats * tracker_refresh_ticks;
    EXPECT_EQ(tracker.observe(0, tracker_feedback(0, 0, second), second + 1200), tracker_status::ready);
    ASSERT_TRUE(tracker.arm());
    const auto third = second + repeats * tracker_refresh_ticks;
    EXPECT_EQ(tracker.observe(1, tracker_feedback(1, repeats, third), third + 1200), tracker_status::accepted);
  }
}

TEST(ShutterFeedbackTracker, ArmedMeasuredHoldsPreserveTheExplicitLeftRightSequence) {
  for (unsigned repeats : {1u, 2u, 3u, 4u}) {
    SCOPED_TRACE(repeats);
    tracker_t tracker(repeats, 120000, tracker_frequency, 2000, 100, 2);
    acquire_tracker(tracker, repeats);
    shutter_presentation_clock_t eye_clock(repeats);
    std::uint32_t present_id = 11;
    std::uint32_t refresh = 1000 + repeats;
    auto boundary = tracker_origin + repeats * tracker_refresh_ticks;
    unsigned accepted_eyes = 0;
    for (unsigned delta = repeats; delta <= 4u * repeats; ++delta) {
      SCOPED_TRACE(delta);
      ++present_id;
      refresh += delta;
      boundary += delta * tracker_refresh_ticks;
      ASSERT_EQ(tracker.observe(present_id, tracker_feedback(present_id, refresh, boundary), boundary + 1200), tracker_status::accepted);
      ASSERT_TRUE(tracker.fresh_for_emit(boundary + 1200));
      EXPECT_EQ(tracker.last_present_qpc(), boundary);
      // Includes odd holds for repeated eyes: refresh parity never chooses the emitted eye.
      EXPECT_EQ(eye_clock.left_eye(), accepted_eyes % 2u == 0);
      EXPECT_EQ(eye_clock.starts_pair(), accepted_eyes % 2u == 0);
      ASSERT_TRUE(eye_clock.observe_displayed(refresh));
      ++accepted_eyes;
      EXPECT_EQ(eye_clock.left_eye(), accepted_eyes % 2u == 0);
    }
  }
}

TEST(ShutterFeedbackTracker, BoundedHeldRefreshesAlsoCrossBothCounterWraps) {
  tracker_t tracker(1, 120000, tracker_frequency, 2000, 100, 2);
  ASSERT_EQ(tracker.observe(UINT32_MAX - 1u, tracker_feedback(UINT32_MAX - 1u, UINT32_MAX - 2u, tracker_origin), tracker_origin + 1200), tracker_status::acquiring);
  const auto second = tracker_origin + tracker_refresh_ticks;
  ASSERT_EQ(tracker.observe(UINT32_MAX, tracker_feedback(UINT32_MAX, UINT32_MAX - 1u, second), second + 1200), tracker_status::ready);
  ASSERT_TRUE(tracker.arm());
  const auto held = second + 3 * tracker_refresh_ticks;
  EXPECT_EQ(tracker.observe(0, tracker_feedback(0, 1, held), held + 1200), tracker_status::accepted);
  EXPECT_TRUE(tracker.fresh_for_emit(held + 1200));
}

TEST(ShutterFeedbackTracker, ColdMeasuredHoldsRestartAllThirteenAcquisitionSamples) {
  for (unsigned repeats : {1u, 2u, 3u, 4u}) {
    SCOPED_TRACE(repeats);
    tracker_t tracker(repeats, 120000, tracker_frequency, 2000, 100);
    std::uint32_t present_id = 100;
    std::uint32_t refresh = 500;
    auto boundary = tracker_origin;
    for (unsigned sample = 0; sample < 12; ++sample) {
      ASSERT_EQ(tracker.observe(present_id, tracker_feedback(present_id, refresh, boundary), boundary + 1200), tracker_status::acquiring);
      ++present_id;
      refresh += repeats;
      boundary += repeats * tracker_refresh_ticks;
    }
    ASSERT_EQ(tracker.streak(), 12u);
    // One additional held refresh interval is valid feedback, but is not a cold continuous clock.
    refresh += repeats;
    boundary += repeats * tracker_refresh_ticks;
    ASSERT_EQ(tracker.observe(present_id, tracker_feedback(present_id, refresh, boundary), boundary + 1200), tracker_status::acquiring);
    EXPECT_EQ(tracker.streak(), 1u);
    EXPECT_FALSE(tracker.arm());
    for (unsigned interval = 1; interval <= 12; ++interval) {
      ++present_id;
      refresh += repeats;
      boundary += repeats * tracker_refresh_ticks;
      EXPECT_EQ(tracker.observe(present_id, tracker_feedback(present_id, refresh, boundary), boundary + 1200),
        interval == 12 ? tracker_status::ready : tracker_status::acquiring);
    }
    EXPECT_TRUE(tracker.arm());
  }
}

TEST(ShutterFeedbackTracker, ArmedHoldsRejectShortRepeatedBackwardsAndExcessiveRefreshDeltas) {
  for (unsigned repeats : {1u, 2u, 3u, 4u}) {
    for (const auto delta : {0u, repeats - 1u, 4u * repeats + 1u, UINT32_MAX}) {
      SCOPED_TRACE(repeats);
      SCOPED_TRACE(delta);
      tracker_t tracker(repeats, 120000, tracker_frequency, 2000, 100, 2);
      acquire_tracker(tracker, repeats);
      const auto elapsed = delta == UINT32_MAX ? repeats : delta;
      const auto boundary = tracker_origin + (repeats + elapsed) * tracker_refresh_ticks;
      EXPECT_EQ(tracker.observe(12, tracker_feedback(12, 1000u + repeats + delta, boundary), boundary + 1200), tracker_status::rejected);
      EXPECT_TRUE(tracker.failed());
      EXPECT_FALSE(tracker.fresh_for_emit(boundary + 1200));
    }
  }
}

TEST(ShutterFeedbackTracker, ArmedBoundarySlightlyAheadOfTheQueryClockIsCurrentOnlyWithinItsTolerance) {
  // 12 ticks per microsecond: 250 us = 3000 ticks.
  tracker_t tracker(1, 120000, tracker_frequency, 2000, 100, 2, 100, 250);
  acquire_tracker(tracker);
  const auto next = tracker_origin + 2 * tracker_refresh_ticks;
  EXPECT_EQ(tracker.observe(12, tracker_feedback(12, 1002, next), next - 3000), tracker_status::accepted);
  EXPECT_TRUE(tracker.fresh_for_emit(next - 3000));
  const auto after = tracker_origin + 3 * tracker_refresh_ticks;
  EXPECT_EQ(tracker.observe(13, tracker_feedback(13, 1003, after), after - 3001), tracker_status::rejected);
  EXPECT_TRUE(tracker.failed());
}

TEST(ShutterFeedbackTracker, ColdAcquisitionNeverAcceptsABoundaryAheadOfTheQueryClock) {
  tracker_t tracker(1, 120000, tracker_frequency, 2000, 100, 2, 100, 250);
  ASSERT_EQ(tracker.observe(10, tracker_feedback(10, 1000, tracker_origin), tracker_origin + 1200), tracker_status::acquiring);
  const auto next = tracker_origin + tracker_refresh_ticks;
  EXPECT_EQ(tracker.observe(11, tracker_feedback(11, 1001, next), next - 1), tracker_status::acquiring);
  EXPECT_FALSE(tracker.arm());
  EXPECT_TRUE(tracker_t(1, 120000, tracker_frequency, 2000, 100, 2, 100, 1001).failed());
}

TEST(ShutterFeedbackTracker, HeldRefreshesCannotHideSkippedIdsClockDomainChangesOrExpiredFrames) {
  enum class fault { skipped_id, old_id, wrong_qpc, counter_domain, stale, future };
  for (auto error : {fault::skipped_id, fault::old_id, fault::wrong_qpc, fault::counter_domain, fault::stale, fault::future}) {
    SCOPED_TRACE(static_cast<int>(error));
    tracker_t tracker(1, 120000, tracker_frequency, 2000, 100, 2);
    acquire_tracker(tracker);
    const auto boundary = tracker_origin + 3 * tracker_refresh_ticks;
    std::uint32_t requested = 12;
    auto feedback = tracker_feedback(requested, 1003, boundary);
    auto query = boundary + 1200;
    switch (error) {
      case fault::skipped_id: requested = feedback.present_count = 13; break;
      case fault::old_id: requested = feedback.present_count = 10; break;
      case fault::wrong_qpc: feedback.sync_qpc += 1212; query += 1212; break;  // 101 us beyond measured hold.
      case fault::counter_domain: feedback.refresh_count += 1000; feedback.sync_refresh_count += 1000; break;
      case fault::stale: query = boundary + 24001; break;
      case fault::future: query = boundary - 1; break;
    }
    EXPECT_EQ(tracker.observe(requested, feedback, query), tracker_status::rejected);
    EXPECT_TRUE(tracker.failed());
    EXPECT_FALSE(tracker.arm());
    // Even a subsequent valid held boundary cannot revive a failed armed clock.
    EXPECT_EQ(tracker.observe(12, tracker_feedback(12, 1003, boundary), boundary + 1200), tracker_status::rejected);
  }
}

TEST(ShutterFeedbackTracker, SignedSyncOffsetsReconstructTheAcceptedPresentRatherThanTheNewerSyncSample) {
  tracker_t tracker(2, 120000, tracker_frequency, 10000, 100, 2);
  const auto first = tracker_origin;
  // The sync sample was one physical refresh earlier than this present.
  EXPECT_EQ(tracker.observe(1, tracker_feedback(1, 100, first, 1), first + 1200), tracker_status::acquiring);
  EXPECT_EQ(tracker.last_present_qpc(), first);
  const auto second = first + 2 * tracker_refresh_ticks;
  // The sync sample is newer, so the accepted present is older by one physical refresh.
  EXPECT_EQ(tracker.observe(2, tracker_feedback(2, 102, second, -1), second + tracker_refresh_ticks + 1200), tracker_status::ready);
  ASSERT_TRUE(tracker.arm());
  EXPECT_TRUE(tracker.fresh_for_emit(second + 120000));  // Exactly 10000 us from PRESENT.
  EXPECT_FALSE(tracker.fresh_for_emit(second + 120001));
  EXPECT_TRUE(tracker.failed());
  EXPECT_EQ(tracker.last_present_qpc(), second);
}

TEST(ShutterFeedbackTracker, CallerBudgetAcceptsItsBoundaryAndRejectsDelayBeforeEmitterWithoutAdvancingPhase) {
  tracker_t tracker(1, 120000, tracker_frequency, 180, 100, 2);
  acquire_tracker(tracker);
  const auto boundary = *tracker.last_present_qpc();
  EXPECT_TRUE(tracker.fresh_for_emit(boundary + 180 * 12));
  EXPECT_EQ(tracker.streak(), 2u);
  EXPECT_FALSE(tracker.fresh_for_emit(boundary + 180 * 12 + 1));
  EXPECT_TRUE(tracker.failed());
  EXPECT_FALSE(tracker.arm());
  const auto next = boundary + tracker_refresh_ticks;
  EXPECT_EQ(tracker.observe(12, tracker_feedback(12, 1002, next), next + 1200), tracker_status::rejected);
}

TEST(ShutterFeedbackTracker, EveryArmedFeedbackViolationLatchesInsteadOfReacquiringOrRearming) {
  enum class fault { missing, disjoint, unsupported, wrong_id, unmeasured_refresh, wrong_cadence, zero_qpc, future_present, future_sync, stale, oversized_offset, offset_min };
  for (auto error : {fault::missing, fault::disjoint, fault::unsupported, fault::wrong_id, fault::unmeasured_refresh, fault::wrong_cadence,
         fault::zero_qpc, fault::future_present, fault::future_sync, fault::stale, fault::oversized_offset, fault::offset_min}) {
    SCOPED_TRACE(static_cast<int>(error));
    tracker_t tracker(1, 120000, tracker_frequency, 2000, 100, 2);
    acquire_tracker(tracker);
    const auto next = tracker_origin + 2 * tracker_refresh_ticks;
    auto feedback = tracker_feedback(12, 1002, next);
    auto query = next + 1200;
    switch (error) {
      case fault::missing: feedback.present_count = 11; break;
      case fault::disjoint: feedback.status = shutter::shutter_feedback_status_e::disjoint; break;
      case fault::unsupported: feedback.status = shutter::shutter_feedback_status_e::unsupported; break;
      case fault::wrong_id: feedback.present_count = 13; break;
      case fault::unmeasured_refresh: ++feedback.refresh_count; ++feedback.sync_refresh_count; break;
      case fault::wrong_cadence: feedback.sync_qpc += tracker_refresh_ticks; query += tracker_refresh_ticks; break;
      case fault::zero_qpc: feedback.sync_qpc = 0; break;
      case fault::future_present: query = next - 1; break;
      case fault::future_sync: feedback.sync_qpc += tracker_refresh_ticks; ++feedback.sync_refresh_count; break;
      case fault::stale: query = next + 24001; break;
      case fault::oversized_offset: feedback.sync_refresh_count -= 2; break;
      case fault::offset_min: feedback.sync_refresh_count = feedback.refresh_count + 0x80000000u; break;
    }
    EXPECT_EQ(tracker.observe(12, feedback, query), tracker_status::rejected);
    EXPECT_TRUE(tracker.failed());
    EXPECT_FALSE(tracker.arm());
    EXPECT_FALSE(tracker.fresh_for_emit(next));
    EXPECT_EQ(tracker.observe(12, tracker_feedback(12, 1002, next), next + 1200), tracker_status::rejected);
  }
}

TEST(ShutterFeedbackTracker, InvalidPoliciesAndExtremeArithmeticNeverAuthorizeASequence) {
  for (tracker_t tracker : {
         tracker_t(0, 120000, tracker_frequency, 2000, 100), tracker_t(5, 120000, tracker_frequency, 2000, 100),
         tracker_t(1, 0, tracker_frequency, 2000, 100), tracker_t(1, 120000, 0, 2000, 100),
         tracker_t(1, 120000, tracker_frequency, -1, 100), tracker_t(1, 120000, tracker_frequency, 2000, -1),
         tracker_t(1, 120000, tracker_frequency, 2000, 100, 0),
         tracker_t(4, 1, std::numeric_limits<std::int64_t>::max(), 2000, 100)}) {
    EXPECT_EQ(tracker.observe(1, tracker_feedback(1, 1, tracker_origin), tracker_origin), tracker_status::rejected);
    EXPECT_FALSE(tracker.arm());
  }
  tracker_t tracker(1, 120000, tracker_frequency, 2000, 100);
  auto overflow = tracker_feedback(1, 100, tracker_origin);
  overflow.sync_qpc = std::numeric_limits<std::int64_t>::max() - 1;
  --overflow.sync_refresh_count;
  EXPECT_EQ(tracker.observe(1, overflow, std::numeric_limits<std::int64_t>::max()), tracker_status::acquiring);
  EXPECT_EQ(tracker.streak(), 0u);
  auto underflow = tracker_feedback(1, 100, tracker_origin);
  underflow.sync_qpc = 1;
  ++underflow.sync_refresh_count;
  EXPECT_EQ(tracker.observe(1, underflow, tracker_origin), tracker_status::acquiring);
  EXPECT_EQ(tracker.streak(), 0u);
}

TEST(ShutterFeedbackTracker, FractionalQpcRefreshShiftRoundsOnceAndKeepsTheMeasuredBoundary) {
  tracker_t tracker(1, 120000, 10000001, 2000, 1, 2);
  auto first = tracker_feedback(1, 100, tracker_origin);
  first.sync_qpc = tracker_origin - 83333;
  --first.sync_refresh_count;
  EXPECT_EQ(tracker.observe(1, first, tracker_origin + 1000), tracker_status::acquiring);
  EXPECT_EQ(tracker.last_present_qpc(), tracker_origin);
  auto second = first;
  second.present_count = 2;
  ++second.refresh_count;
  ++second.sync_refresh_count;
  second.sync_qpc += 83334;
  EXPECT_EQ(tracker.observe(2, second, tracker_origin + 83334 + 1000), tracker_status::ready);
  EXPECT_EQ(tracker.last_present_qpc(), tracker_origin + 83334);
}

namespace {
  using phase_status = shutter::shutter_emit_phase_wait_status_e;

  struct phase_wait_probe_t {
    using clock = shutter::shutter_emit_phase_wait_io_t::clock;
    std::int64_t qpc = 1000000;
    clock::time_point now = clock::time_point {} + 1s;
    bool cancelled = false;
    unsigned waits = 0, relaxes = 0, qpc_queries = 0;

    shutter::shutter_emit_phase_wait_io_t io() {
      return {
        .now_qpc = [&]() { ++qpc_queries; return qpc; },
        .now = [&]() { return now; },
        .cancelled = [&]() { return cancelled; },
        .wait_until = [&](auto deadline) {
          ++waits;
          qpc += std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();
          now = deadline;
          return true;
        },
        .relax = [&]() {
          ++relaxes;
          ++qpc;
          now += 1us;
        },
      };
    }
  };
}

TEST(ShutterEmitPhase, BudgetIncludesTheWholeLatenessWindowAndZeroStaysDisabled) {
  const auto edge = shutter::plan_shutter_emit_phase(1000000, 1000000, 1000, 1250);
  ASSERT_TRUE(edge);
  EXPECT_EQ(edge->target_qpc, 1001000);
  EXPECT_EQ(edge->latest_write_qpc, 1001250);
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(1000000, 1000000, 1000, 1249));
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(1000000, 1000000, 0, 1250));
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(1000000, 1000000, -1, 1250));
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(0, 1000000, 1000, 1250));
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(1000000, 0, 1000, 1250));
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(1000000, 1000000, 1000, -1));
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(1000000, 1000000, 1000, 1250, -1));
  const auto exact = shutter::plan_shutter_emit_phase(1000000, 1000000, 1000, 1000, 0);
  ASSERT_TRUE(exact);
  EXPECT_EQ(exact->target_qpc, exact->latest_write_qpc);
}

TEST(ShutterEmitPhase, FractionalQpcRoundingCannotAdvanceTheTargetOrWidenTheLimit) {
  const auto plan = shutter::plan_shutter_emit_phase(1000000, 10000001, 1000, 1250);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->target_qpc, 1010001);
  EXPECT_EQ(plan->latest_write_qpc, 1012500);
  // No representable tick lies inside this sub-tick requested window.
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(1, 1, 1, 251));
}

TEST(ShutterEmitPhase, ExtremeInputsCannotOverflowProductsSumsOrTheAcceptedBoundary) {
  constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(maximum - 1000, 1000000, 1000, 1250));
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(1, maximum, maximum, maximum, 0));
  EXPECT_FALSE(shutter::plan_shutter_emit_phase(1, 1000000, maximum, maximum));
  const auto large = shutter::plan_shutter_emit_phase(1, maximum, 999999, 1000000, 1);
  EXPECT_FALSE(large);  // The exact one-second latest boundary cannot be added to 1.
  const auto safe = shutter::plan_shutter_emit_phase(1, maximum, 1, 251);
  ASSERT_TRUE(safe);  // Large factors with a representable result remain valid.
  EXPECT_GT(safe->latest_write_qpc, safe->target_qpc);
}

TEST(ShutterEmitPhase, WaitRechecksEarlyNotificationsAndNeverReturnsBeforeTheTarget) {
  phase_wait_probe_t probe;
  auto io = probe.io();
  const auto ordinary_wait = io.wait_until;
  io.wait_until = [&](auto deadline) {
    if (probe.waits == 0) {
      ++probe.waits;
      probe.qpc += 50;
      probe.now += 50us;
      return true;
    }
    return ordinary_wait(deadline);
  };
  const auto plan = shutter::plan_shutter_emit_phase(probe.qpc, 1000000, 1000, 1250);
  ASSERT_TRUE(plan);
  EXPECT_EQ(shutter::await_shutter_emit_phase(io, *plan, 1000000), phase_status::ready);
  EXPECT_EQ(probe.qpc, plan->target_qpc);
  EXPECT_GT(probe.waits, 1u);
  EXPECT_GT(probe.relaxes, 0u);
}

TEST(ShutterEmitPhase, CoarseTimerQuantumCannotReplaceTheFineTailWithALateWake) {
  // Model a 500-us timer quantum that reproduces the late wake with feedback 422 us old. Repeated
  // 200-us OS waits alone first reach a 2000-us target at age 2422, past its 2250-us limit.
  constexpr std::int64_t feedback_age = 422;
  phase_wait_probe_t timer_only;
  const auto boundary = timer_only.qpc - feedback_age;
  const auto plan = shutter::plan_shutter_emit_phase(boundary, 1000000, 2000, 3317);
  ASSERT_TRUE(plan);
  const auto coarse_io = [](phase_wait_probe_t &probe) {
    auto io = probe.io();
    io.wait_until = [&](auto deadline) {
      ++probe.waits;
      const auto requested = std::chrono::ceil<std::chrono::microseconds>(deadline - probe.now).count();
      const auto actual = std::chrono::microseconds {((requested + 499) / 500) * 500};
      probe.qpc += actual.count();
      probe.now += actual;
      return true;
    };
    return io;
  };
  auto previous_io = coarse_io(timer_only);
  while (timer_only.qpc < plan->target_qpc) {
    ASSERT_TRUE(previous_io.wait_until(timer_only.now + 200us));
  }
  EXPECT_EQ(timer_only.qpc - boundary, 2422);
  EXPECT_GT(timer_only.qpc, plan->latest_write_qpc);

  phase_wait_probe_t precise;
  auto io = coarse_io(precise);
  const auto quantum_wait = io.wait_until;
  io.wait_until = [&](auto deadline) {
    const auto requested = std::chrono::duration_cast<std::chrono::microseconds>(deadline - precise.now).count();
    const auto remaining = plan->target_qpc - precise.qpc;
    EXPECT_GT(remaining, 750);
    EXPECT_LE(requested, 200);
    EXPECT_LE(requested, remaining - 750);
    return quantum_wait(deadline);
  };
  EXPECT_EQ(shutter::await_shutter_emit_phase(io, *plan, 1000000), phase_status::ready);
  EXPECT_EQ(precise.qpc, plan->target_qpc);
  EXPECT_EQ(precise.qpc - boundary, 2000);
  EXPECT_EQ(precise.waits, 2u);
  EXPECT_GT(precise.relaxes, 0u);
  EXPECT_LE(precise.qpc, plan->latest_write_qpc);
}

TEST(ShutterEmitPhase, FineTailAllowsFastPollingBeyondTheCoarseTimerRetryLimit) {
  phase_wait_probe_t probe;
  auto io = probe.io();
  io.relax = [&]() {
    ++probe.relaxes;
    ++probe.qpc;
    probe.now += 100ns;
  };
  const auto plan = shutter::plan_shutter_emit_phase(probe.qpc, 10000000, 500, 3317);
  ASSERT_TRUE(plan);
  EXPECT_EQ(shutter::await_shutter_emit_phase(io, *plan, 10000000), phase_status::ready);
  EXPECT_EQ(probe.qpc, plan->target_qpc);
  EXPECT_EQ(probe.waits, 0u);
  EXPECT_EQ(probe.relaxes, 5000u);
}

TEST(ShutterEmitPhase, FineTailCancellationAndClockFaultsNeverAuthorizeAnEye) {
  for (int fault : {0, 1, 2, 3, 4, 5}) {
    SCOPED_TRACE(fault);
    phase_wait_probe_t probe;
    auto io = probe.io();
    const auto plan = shutter::plan_shutter_emit_phase(probe.qpc, 1000000, 500, 3317);
    ASSERT_TRUE(plan);
    io.relax = [&]() {
      ++probe.relaxes;
      if (fault == 0) {
        probe.cancelled = true;
      } else if (fault == 1) {
        --probe.qpc;
      } else if (fault == 2) {
        probe.now -= 1us;
      } else if (fault == 3) {
        probe.now += 1us;  // Frozen QPC still expires at the steady hard deadline.
      } else if (fault == 5) {
        probe.qpc = plan->latest_write_qpc + 1;
        probe.now += 751us;
      }
      // fault 4: both clocks are frozen, so the independent spin count must terminate.
    };
    const auto status = shutter::await_shutter_emit_phase(io, *plan, 1000000);
    const auto expected = fault == 0 ? phase_status::cancelled : fault < 3 ? phase_status::invalid :
                          fault == 4 ? phase_status::wait_failed : phase_status::missed;
    EXPECT_EQ(status, expected);
    EXPECT_EQ(probe.waits, 0u);
    EXPECT_LE(probe.relaxes, 131072u);
    unsigned writes = 0;
    if (status == phase_status::ready) {
      ++writes;
    }
    EXPECT_EQ(writes, 0u);
  }
}

TEST(ShutterEmitPhase, LateFeedbackCannotCatchUpEvenInsideTheWakeLatenessAllowance) {
  phase_wait_probe_t probe;
  const shutter::shutter_emit_phase_plan_t plan {probe.qpc - 1, probe.qpc + 249};
  EXPECT_EQ(shutter::await_shutter_emit_phase(probe.io(), plan, 1000000), phase_status::missed);
  EXPECT_EQ(probe.waits, 0u);
  const shutter::shutter_emit_phase_plan_t at_target {probe.qpc, probe.qpc + 250};
  EXPECT_EQ(shutter::await_shutter_emit_phase(probe.io(), at_target, 1000000), phase_status::ready);
  EXPECT_EQ(probe.waits, 0u);
}

TEST(ShutterEmitPhase, WakeAtTheFinalWriteLimitIsAllowedButOneTickLaterIsMissed) {
  for (const std::int64_t overshoot : {0, 1}) {
    phase_wait_probe_t probe;
    auto io = probe.io();
    const auto plan = shutter::plan_shutter_emit_phase(probe.qpc, 1000000, 1000, 1250);
    ASSERT_TRUE(plan);
    io.wait_until = [&](auto) {
      ++probe.waits;
      probe.qpc = plan->latest_write_qpc + overshoot;
      probe.now += std::chrono::microseconds {1250 + overshoot};
      return true;
    };
    EXPECT_EQ(shutter::await_shutter_emit_phase(io, *plan, 1000000),
      overshoot ? phase_status::missed : phase_status::ready);
    EXPECT_EQ(probe.waits, 1u);
  }
}

TEST(ShutterEmitPhase, CancellationAndFailedWaitsCannotAuthorizeAnEye) {
  phase_wait_probe_t probe;
  const auto plan = shutter::plan_shutter_emit_phase(probe.qpc, 1000000, 1000, 1250);
  ASSERT_TRUE(plan);
  probe.cancelled = true;
  EXPECT_EQ(shutter::await_shutter_emit_phase(probe.io(), *plan, 1000000), phase_status::cancelled);
  EXPECT_EQ(probe.qpc_queries, 0u);
  probe.cancelled = false;
  auto io = probe.io();
  io.wait_until = [&](auto) { probe.cancelled = true; return false; };
  EXPECT_EQ(shutter::await_shutter_emit_phase(io, *plan, 1000000), phase_status::cancelled);
  probe.cancelled = false;
  io.wait_until = [](auto) { return false; };
  EXPECT_EQ(shutter::await_shutter_emit_phase(io, *plan, 1000000), phase_status::wait_failed);
}

TEST(ShutterEmitPhase, RegressingClocksAndStalledQpcFailClosed) {
  for (int fault : {0, 1, 2, 3}) {
    phase_wait_probe_t probe;
    auto io = probe.io();
    const auto plan = shutter::plan_shutter_emit_phase(probe.qpc, 1000000, 1000, 1250);
    ASSERT_TRUE(plan);
    io.wait_until = [&](auto deadline) {
      ++probe.waits;
      if (fault == 0) {
        --probe.qpc;
      } else if (fault == 1) {
        probe.now -= 1us;
      } else if (fault == 2) {
        probe.now = deadline;  // A frozen QPC cannot extend the steady hard deadline.
      }
      return true;  // fault 3: a broken notifier advances neither clock.
    };
    const auto expected = fault < 2 ? phase_status::invalid : fault == 2 ? phase_status::missed : phase_status::wait_failed;
    EXPECT_EQ(shutter::await_shutter_emit_phase(io, *plan, 1000000), expected);
    EXPECT_LE(probe.waits, 4096u);
  }
}

TEST(ShutterEmitPhase, InvalidPlansAndSteadyDeadlineOverflowNeverWait) {
  phase_wait_probe_t probe;
  auto missing_relax = probe.io();
  missing_relax.relax = {};
  EXPECT_EQ(shutter::await_shutter_emit_phase(missing_relax, {1001000, 1001250}, 1000000), phase_status::invalid);
  EXPECT_EQ(shutter::await_shutter_emit_phase(probe.io(), {}, 1000000), phase_status::invalid);
  EXPECT_EQ(shutter::await_shutter_emit_phase(probe.io(), {1001000, 1000999}, 1000000), phase_status::invalid);
  EXPECT_EQ(shutter::await_shutter_emit_phase(probe.io(), {1001000, 1001250}, 0), phase_status::invalid);
  probe.now = phase_wait_probe_t::clock::time_point::max() - 1us;
  EXPECT_EQ(shutter::await_shutter_emit_phase(probe.io(), {1001000, 1001250}, 1000000), phase_status::invalid);
  probe.now = phase_wait_probe_t::clock::time_point {};
  probe.qpc = 1;
  constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
  EXPECT_EQ(shutter::await_shutter_emit_phase(probe.io(), {maximum - 1, maximum}, 1), phase_status::invalid);
  EXPECT_EQ(probe.waits, 0u);
}

TEST(ShutterEmitPhase, CapturedLogicalSlotKeepsZeroPhaseImmediateAcrossOddHeldRefreshes) {
  shutter_presentation_clock_t eye_clock(1);
  phase_wait_probe_t probe;
  const std::array<std::int64_t, 2> phases {0, 1000};
  unsigned immediate = 0, delayed = 0;
  for (const auto refresh : {40u, 43u, 44u, 48u}) {
    const bool displayed_left = eye_clock.left_eye();
    ASSERT_TRUE(eye_clock.observe_displayed(refresh));
    const auto phase = phases[displayed_left ? 0 : 1];
    const auto prior_queries = probe.qpc_queries;
    if (phase == 0) {
      ++immediate;  // Presenter contract: preserve the original path without consulting IO.
      EXPECT_EQ(probe.qpc_queries, prior_queries);
    } else {
      const auto plan = shutter::plan_shutter_emit_phase(probe.qpc, 1000000, phase, 1250);
      ASSERT_TRUE(plan);
      ASSERT_EQ(shutter::await_shutter_emit_phase(probe.io(), *plan, 1000000), phase_status::ready);
      ++delayed;
    }
    EXPECT_NE(displayed_left, eye_clock.left_eye());
  }
  EXPECT_EQ(immediate, 2u);
  EXPECT_EQ(delayed, 2u);
  EXPECT_TRUE(eye_clock.starts_pair());
}

TEST(ShutterFeedbackTracker, AnEyeOfSeveralPresentsIsIdentifiedByItsFirstPresentOneStrideLater) {
  // Three presents per eye: eyes are 10, 13, 16 and each spans three refreshes.
  tracker_t tracker(3, 120000, tracker_frequency, 2000, 100, 2, 100, 0, 3);
  const auto at = [](std::uint32_t refresh) {
    return tracker_origin + static_cast<std::int64_t>(refresh - 1000u) * tracker_refresh_ticks;
  };
  ASSERT_EQ(tracker.observe(10, tracker_feedback(10, 1000, at(1000)), at(1000) + 1200), tracker_status::acquiring);
  ASSERT_EQ(tracker.observe(13, tracker_feedback(13, 1003, at(1003)), at(1003) + 1200), tracker_status::ready);
  ASSERT_TRUE(tracker.arm());
  EXPECT_EQ(tracker.observe(16, tracker_feedback(16, 1006, at(1006)), at(1006) + 1200), tracker_status::accepted);
  // The second present of that eye is not an eye boundary.
  EXPECT_EQ(tracker.observe(17, tracker_feedback(17, 1007, at(1007)), at(1007) + 1200), tracker_status::rejected);
  EXPECT_TRUE(tracker.failed());
  EXPECT_TRUE(tracker_t(3, 120000, tracker_frequency, 2000, 100, 2, 100, 0, 0).failed());
  EXPECT_TRUE(tracker_t(3, 120000, tracker_frequency, 2000, 100, 2, 100, 0, 5).failed());
}

TEST(ShutterFeedbackTracker, ColdAcquisitionWithAPresentStrideRejectsTheOldSingleStep) {
  tracker_t tracker(3, 120000, tracker_frequency, 2000, 100, 2, 100, 0, 3);
  ASSERT_EQ(tracker.observe(10, tracker_feedback(10, 1000, tracker_origin), tracker_origin + 1200), tracker_status::acquiring);
  const auto next = tracker_origin + 3 * tracker_refresh_ticks;
  EXPECT_EQ(tracker.observe(11, tracker_feedback(11, 1003, next), next + 1200), tracker_status::acquiring);
  EXPECT_FALSE(tracker.arm());
}

TEST(ShutterPresentFeedback, ALaterPresentOfTheSameEyeIsAMissedMeasurementButABeyondEyePresentIsNot) {
  using status_e = shutter::shutter_feedback_wait_status_e;
  shutter::shutter_feedback_wait_result_t result {status_e::unsupported};
  result.feedback.status = shutter::shutter_feedback_status_e::available;
  for (std::uint32_t ahead : {1u, 2u}) {
    result.feedback.present_count = 100u + ahead;
    EXPECT_TRUE(shutter::shutter_feedback_superseded_within_eye(result, 100, 3));
  }
  for (std::uint32_t count : {100u, 103u, 104u, 99u}) {
    result.feedback.present_count = count;
    EXPECT_FALSE(shutter::shutter_feedback_superseded_within_eye(result, 100, 3));
  }
  result.feedback.present_count = 101;
  EXPECT_FALSE(shutter::shutter_feedback_superseded_within_eye(result, 100, 1));  // One present per eye has no later one.
  result.feedback.status = shutter::shutter_feedback_status_e::disjoint;
  EXPECT_FALSE(shutter::shutter_feedback_superseded_within_eye(result, 100, 3));
  result.feedback.status = shutter::shutter_feedback_status_e::available;
  result.status = status_e::displayed;
  EXPECT_FALSE(shutter::shutter_feedback_superseded_within_eye(result, 100, 3));
  // Present IDs wrap.
  result.status = status_e::unsupported;
  result.feedback.present_count = 0;
  EXPECT_TRUE(shutter::shutter_feedback_superseded_within_eye(result, UINT32_MAX, 3));
}

TEST(ShutterFeedbackTracker, AShortIntervalAfterALateImageIsStillAFault) {
  // With one present per refresh a late image shifts every later image, so nothing is made up.
  tracker_t tracker(3, 120000, tracker_frequency, 2000, 100, 2);
  acquire_tracker(tracker, 3);  // Boundaries at refresh 1000 and 1003.
  const auto at = [](std::uint32_t refresh) {
    return tracker_origin + static_cast<std::int64_t>(refresh - 1000u) * tracker_refresh_ticks;
  };
  ASSERT_EQ(tracker.observe(12, tracker_feedback(12, 1007, at(1007)), at(1007) + 1200), tracker_status::accepted);  // One late.
  EXPECT_EQ(tracker.observe(13, tracker_feedback(13, 1009, at(1009)), at(1009) + 1200), tracker_status::rejected);
  EXPECT_TRUE(tracker.failed());
  shutter_presentation_clock_t clock(3);
  ASSERT_TRUE(clock.observe_displayed(1003));
  ASSERT_TRUE(clock.observe_displayed(1007));
  EXPECT_FALSE(clock.observe_displayed(1009));
}

TEST(ShutterPresentFeedback, AnEmptySampleFollowedByALaterPresentOfTheEyeIsARecoverableMiss) {
  feedback_probe_t probe;
  probe.now = probe.feedback_at;
  auto io = probe.io();
  unsigned empty_left = 2;
  io.query = [&]() {
    ++probe.queries;
    if (empty_left) {
      --empty_left;
      return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::available, probe.present_id, 0, 0, 0};
    }
    return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::available, probe.present_id + 1u, probe.refresh + 1u, probe.refresh + 1u, 123456};
  };
  const auto result = shutter::await_shutter_present(io, probe.present_id, false);
  EXPECT_EQ(result.status, shutter::shutter_feedback_wait_status_e::unsupported);
  EXPECT_FALSE(shutter::shutter_feedback_is_timing_interruption(result));
  EXPECT_TRUE(shutter::shutter_feedback_superseded_within_eye(result, probe.present_id, 3));
  EXPECT_FALSE(shutter::shutter_feedback_superseded_within_eye(result, probe.present_id, 1));
}
