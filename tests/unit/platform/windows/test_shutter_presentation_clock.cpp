#include "src/platform/windows/shutter_presentation_clock.h"

#include <gtest/gtest.h>
#include <vector>

using platf::dxgi::detail::shutter_presentation_clock_t;

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
    EXPECT_EQ(probe.flushes, 6u);
    EXPECT_EQ(probe.waits.size(), 6u);
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
  EXPECT_EQ(probe.flushes, 1u);
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

TEST(ShutterPresentFeedback, StaleFeedbackFlushesOnceAndStopsAtABoundedDeadline) {
  feedback_probe_t probe;
  auto io = probe.io();
  probe.displayed_at = probe.now;
  io.query = [&]() {
    return shutter::shutter_feedback_t {shutter::shutter_feedback_status_e::available, probe.present_id - 1u, 0};
  };
  EXPECT_EQ(shutter::await_shutter_present(io, probe.present_id, false).status, shutter::shutter_feedback_wait_status_e::timeout);
  EXPECT_EQ(probe.now.time_since_epoch(), 250ms);
  EXPECT_EQ(probe.flushes, 1u);
  EXPECT_EQ(probe.waits.size(), 1250u);
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
  EXPECT_EQ(probe.flushes, 1u);
  EXPECT_EQ(probe.queries, 2u);
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
  EXPECT_EQ(shutter::await_shutter_present(io, probe.present_id, false).status, shutter::shutter_feedback_wait_status_e::flush_failed);
  io = probe.io();
  io.wait_until = [](auto) {
    return false;
  };
  EXPECT_EQ(shutter::await_shutter_present(io, probe.present_id, false).status, shutter::shutter_feedback_wait_status_e::wait_failed);
}
