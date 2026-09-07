#include <gtest/gtest.h>
#include <src/host_sbs_telemetry_perf.h>
#include <thread>

namespace {
  using namespace host_sbs_telemetry;

  host_sbs_telemetry::clock_t::time_point at(std::uint64_t ms) {
    return host_sbs_telemetry::clock_t::time_point {std::chrono::milliseconds(ms)};
  }
}  // namespace

TEST(HostSbsTelemetryPerfTests, MissingAndInvalidMeasurementsAreNotMeasuredZero) {
  collector perf;
  EXPECT_EQ(perf.snapshot().valid_fields, 0u);
  perf.record(stage::warp, -1.0, at(100));
  perf.record(stage::warp, std::numeric_limits<double>::infinity(), at(100));
  perf.record(stage::warp, std::numeric_limits<double>::quiet_NaN(), at(100));
  EXPECT_EQ(perf.snapshot().valid_fields, 0u);
  perf.record(stage::warp, 0.0, at(101));
  auto sample = perf.snapshot();
  EXPECT_EQ(sample.valid_fields, stage_valid(3));
  EXPECT_EQ(sample.stages[3].count, 1u);
  EXPECT_FLOAT_EQ(sample.stages[3].mean_ms, 0.0f);
  EXPECT_EQ(sample.stages[3].latest_host_ms, 101u);
}

TEST(HostSbsTelemetryPerfTests, IndependentStagesAverageSinceGenerationAndDoNotRefreshOnRead) {
  collector current;
  collector retiring;
  current.record(stage::conversion, 2.0, at(100));
  current.record(stage::conversion, 4.0, at(200));
  current.record(stage::encode, 8.0, at(250));
  retiring.record(stage::conversion, 99.0, at(300));
  const auto sample = current.snapshot();
  EXPECT_FLOAT_EQ(sample.stages[0].mean_ms, 3.0f);
  EXPECT_EQ(sample.stages[0].count, 2u);
  EXPECT_EQ(sample.stages[0].latest_host_ms, 200u);
  EXPECT_FLOAT_EQ(sample.stages[1].mean_ms, 8.0f);
  EXPECT_EQ(current.snapshot().stages[0].latest_host_ms, 200u);
}

TEST(HostSbsTelemetryPerfTests, OutcomeCopiesAreNonConsumingAndRestartsAdvanceEpoch) {
  collector perf;
  perf.record_outcomes({20, 80, 2}, at(1000));
  auto first = perf.snapshot();
  EXPECT_EQ(first.outcome_epoch, 1u);
  EXPECT_EQ(first.outcome_sequence, 1u);
  EXPECT_EQ(perf.snapshot().outcomes.reuse, 80u);
  EXPECT_EQ(perf.snapshot().outcome_sequence, 1u);
  perf.record_outcomes({20, 90, 2}, at(2000));
  auto second = perf.snapshot();
  EXPECT_EQ(second.outcome_epoch, first.outcome_epoch);
  EXPECT_EQ(second.outcome_sequence, 2u);
  EXPECT_EQ(second.outcome_host_ms - first.outcome_host_ms, 1000u);
  EXPECT_EQ(second.outcomes.reuse - first.outcomes.reuse, 10u);
  perf.record_outcomes({1, 1, 0}, at(3000));
  EXPECT_EQ(perf.snapshot().outcome_epoch, 2u);
  perf.invalidate_outcomes();
  EXPECT_EQ(perf.snapshot().valid_fields & outcomes_valid, 0u);
  perf.record_outcomes({50, 50, 0}, at(4000));
  EXPECT_EQ(perf.snapshot().outcome_epoch, 3u);
}

TEST(HostSbsTelemetryPerfTests, OutputCountersSeparateDepthReuseFromPackedPresentation) {
  collector perf;
  perf.record_output(false, true, at(10));
  perf.record_output(true, false, at(20));
  perf.record_output(false, false, at(30));
  const auto sample = perf.snapshot();
  EXPECT_EQ(sample.warped, 1u);
  EXPECT_EQ(sample.repeated, 1u);
  EXPECT_EQ(sample.flat, 1u);
  EXPECT_EQ(sample.output_host_ms, 30u);
  EXPECT_EQ(sample.valid_fields, outputs_valid);
}

TEST(HostSbsTelemetryPerfTests, CrossThreadOwnersPublishCoherentCounts) {
  collector perf;
  std::thread sender([&] {
    for (int i = 0; i < 1000; ++i) {
      perf.record(stage::new_content_age, 7.0, at(i));
    }
  });
  for (int i = 0; i < 1000; ++i) {
    perf.record(stage::conversion, 3.0, at(i));
    const auto sample = perf.snapshot();
    if (sample.stages[2].count > 0) {
      EXPECT_FLOAT_EQ(sample.stages[2].mean_ms, 7.0f);
    }
  }
  sender.join();
  EXPECT_EQ(perf.snapshot().stages[0].count, 1000u);
  EXPECT_EQ(perf.snapshot().stages[2].count, 1000u);
}

TEST(HostSbsTelemetryPerfTests, HostTimestampWrapKeepsShortUnsignedIntervals) {
  const auto before = host_ms(at(0xFFFFFFF0ull));
  const auto after = host_ms(at(0x100000010ull));
  EXPECT_EQ(after - before, 32u);
}
