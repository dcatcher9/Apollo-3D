#include "src/platform/windows/shutter_presentation_clock.h"

#include <gtest/gtest.h>
#include <limits>
#include <vector>

using platf::dxgi::detail::shutter_presentation_clock_t;

TEST(ShutterRecoveryBudget, AllowsTwoTransientLossesButBoundsARepeatedFailureLoop) {
  using budget_t = platf::dxgi::detail::shutter_recovery_budget_t;
  using namespace std::chrono_literals;
  budget_t budget;
  const budget_t::clock::time_point start {};
  EXPECT_TRUE(budget.begin(start));
  EXPECT_TRUE(budget.begin(start + 500ms));
  EXPECT_FALSE(budget.begin(start + 9999ms));
  EXPECT_TRUE(budget.begin(start + 10s));
  // Expiring the first attempt does not expire the newer second attempt.
  EXPECT_FALSE(budget.begin(start + 10499ms));
  EXPECT_TRUE(budget.begin(start + 10500ms));
}

TEST(ShutterRecoveryBudget, ClockRegressionCannotResetTheFailureLimit) {
  using budget_t = platf::dxgi::detail::shutter_recovery_budget_t;
  using namespace std::chrono_literals;
  budget_t budget;
  const auto start = budget_t::clock::time_point {} + 10s;
  EXPECT_TRUE(budget.begin(start));
  EXPECT_FALSE(budget.begin(start - 1ms));
  EXPECT_TRUE(budget.begin(start));
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

TEST(ShutterPresentationClock, MissingOrRepeatedScanoutDoesNotGuessNextEye) {
  shutter_presentation_clock_t clock(1);
  ASSERT_TRUE(clock.observe_displayed(42));
  EXPECT_FALSE(clock.observe_displayed(44));
  EXPECT_FALSE(clock.left_eye());
  EXPECT_FALSE(clock.observe_displayed(42));
  EXPECT_FALSE(clock.left_eye());
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

TEST(ShutterPresentFeedback, SkippedActualRefreshStillFailsAfterMetadataBecomesAvailable) {
  feedback_probe_t probe;
  probe.refresh = 102;
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
  enum class fault { missing, disjoint, unsupported, wrong_id, skipped_refresh, wrong_cadence, zero_qpc, future_present, future_sync, stale, oversized_offset, offset_min };
  for (auto error : {fault::missing, fault::disjoint, fault::unsupported, fault::wrong_id, fault::skipped_refresh, fault::wrong_cadence,
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
      case fault::skipped_refresh: ++feedback.refresh_count; ++feedback.sync_refresh_count; break;
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
