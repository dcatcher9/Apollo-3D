#include <gtest/gtest.h>

#include <src/host_sbs_gpu_outcomes.h>

namespace {
  namespace outcomes = models::host_sbs_gpu_outcomes;

  TEST(HostSbsGpuOutcomesTest, CumulativeSnapshotsPreserveSkippedReadsAndDrainOnce) {
    outcomes::delta_tracker_t tracker;
    EXPECT_FALSE(tracker.observe({7u, 3u, 1u}));
    // Many GPU dispatches can pass without a successful CPU sample.
    EXPECT_FALSE(tracker.observe({1007u, 503u, 4u}));
    const auto first = tracker.take_delta();
    EXPECT_EQ(first.infer, 1007u);
    EXPECT_EQ(first.reuse, 503u);
    EXPECT_EQ(first.invalid, 4u);
    const auto empty = tracker.take_delta();
    EXPECT_EQ(empty.infer, 0u);
    EXPECT_EQ(empty.reuse, 0u);
    EXPECT_EQ(empty.invalid, 0u);
    EXPECT_FALSE(tracker.observe({1007u, 503u, 4u}));
    EXPECT_FALSE(tracker.observe({1010u, 510u, 6u}));
    const auto next = tracker.take_delta();
    EXPECT_EQ(next.infer, 3u);
    EXPECT_EQ(next.reuse, 7u);
    EXPECT_EQ(next.invalid, 2u);
    EXPECT_DOUBLE_EQ(next.reuse_percent(), 70.0);
    EXPECT_EQ(tracker.totals().infer, 1010u);
    EXPECT_EQ(tracker.totals().reuse, 510u);
  }

  TEST(HostSbsGpuOutcomesTest, DecodesLowWordCarryBeforeComputingDelta) {
    auto words = outcomes::initial_words;
    words[2u] = 0xfffffffeu;
    words[3u] = 12u;
    const auto before = outcomes::decode(words);
    ASSERT_TRUE(before);
    outcomes::delta_tracker_t tracker;
    EXPECT_FALSE(tracker.observe(*before));
    (void) tracker.take_delta();
    words[2u] = 3u;
    words[3u] = 13u;
    const auto after = outcomes::decode(words);
    ASSERT_TRUE(after);
    EXPECT_FALSE(tracker.observe(*after));
    EXPECT_EQ(tracker.take_delta().infer, 5u);
    EXPECT_EQ(tracker.totals().infer, (13ull << 32u) + 3u);
  }

  TEST(HostSbsGpuOutcomesTest, SessionResetAndRegressedCountersStartIndependentEpochs) {
    outcomes::delta_tracker_t tracker;
    EXPECT_FALSE(tracker.observe({500u, 100u, 2u}));
    tracker.reset();
    EXPECT_FALSE(tracker.observe({2u, 3u, 0u}));
    auto delta = tracker.take_delta();
    EXPECT_EQ(delta.infer, 2u);
    EXPECT_EQ(delta.reuse, 3u);
    EXPECT_EQ(delta.invalid, 0u);
    EXPECT_FALSE(tracker.observe({4u, 8u, 1u}));
    // A regression of any counter invalidates the entire old epoch, including pending deltas.
    EXPECT_TRUE(tracker.observe({5u, 0u, 0u}));
    delta = tracker.take_delta();
    EXPECT_EQ(delta.infer, 5u);
    EXPECT_EQ(delta.reuse, 0u);
    EXPECT_EQ(delta.invalid, 0u);
    EXPECT_DOUBLE_EQ(delta.reuse_percent(), 0.0);
  }

  TEST(HostSbsGpuOutcomesTest, RejectsUnknownCounterBuffersAndExcludesInvalidFromPercentage) {
    auto words = outcomes::initial_words;
    words[2u] = 3u;
    words[4u] = 1u;
    words[6u] = 100u;
    const auto counts = outcomes::decode(words);
    ASSERT_TRUE(counts);
    EXPECT_DOUBLE_EQ(counts->reuse_percent(), 25.0);
    EXPECT_DOUBLE_EQ(outcomes::counts_t {}.reuse_percent(), 0.0);
    words[0u] += 1u;
    EXPECT_FALSE(outcomes::decode(words));
    words = outcomes::initial_words;
    words[1u] ^= 1u;
    EXPECT_FALSE(outcomes::decode(words));
  }
}  // namespace
