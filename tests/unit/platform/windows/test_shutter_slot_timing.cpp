#include "src/platform/windows/shutter_slot_timing.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using platf::dxgi::detail::shutter_timing_aggregate_t;

TEST(ShutterSlotTiming, InvalidMeasurementsCannotBiasTheTimingSummary) {
  shutter_timing_aggregate_t timing;
  timing.observe(-1);
  timing.observe(std::numeric_limits<double>::quiet_NaN());
  timing.observe(std::numeric_limits<double>::infinity());
  EXPECT_EQ(timing.count(), 0);
  EXPECT_EQ(timing.invalid_count(), 3);
  EXPECT_EQ(timing.percentile(95).upper_bound_us, 0);
  timing.observe(120);
  timing.observe(380);
  EXPECT_EQ(timing.count(), 2);
  EXPECT_EQ(timing.invalid_count(), 3);
  EXPECT_DOUBLE_EQ(timing.minimum_us(), 120);
  EXPECT_DOUBLE_EQ(timing.mean_us(), 250);
  EXPECT_DOUBLE_EQ(timing.maximum_us(), 380);
}

TEST(ShutterSlotTiming, PercentilesIncludeLongOutliersInsteadOfClippingThemIntoAValidBin) {
  shutter_timing_aggregate_t timing;
  for (int i = 0; i < 98; ++i) {
    timing.observe(250);
  }
  timing.observe(12799.9);
  timing.observe(25000);
  EXPECT_EQ(timing.overflow_count(), 1);
  EXPECT_FALSE(timing.percentile(95).overflow);
  EXPECT_DOUBLE_EQ(timing.percentile(95).upper_bound_us, 300);
  EXPECT_FALSE(timing.percentile(99).overflow);
  EXPECT_DOUBLE_EQ(timing.percentile(99).upper_bound_us, 12800);
  EXPECT_TRUE(timing.percentile(100).overflow);
  EXPECT_DOUBLE_EQ(timing.percentile(100).upper_bound_us, 12800);
  EXPECT_DOUBLE_EQ(timing.maximum_us(), 25000);
  EXPECT_NEAR(timing.mean_us(), (98 * 250 + 12799.9 + 25000) / 100, 0.000001);
}

TEST(ShutterSlotTiming, HistogramRanksHandleZeroAndExactBinBoundaries) {
  shutter_timing_aggregate_t timing;
  for (double value : {0.0, 99.9, 100.0, 199.9, 12799.9, 12800.0}) {
    timing.observe(value);
  }
  EXPECT_DOUBLE_EQ(timing.minimum_us(), 0);
  EXPECT_DOUBLE_EQ(timing.percentile(0).upper_bound_us, 100);
  EXPECT_DOUBLE_EQ(timing.percentile(50).upper_bound_us, 200);
  EXPECT_DOUBLE_EQ(timing.percentile(80).upper_bound_us, 12800);
  EXPECT_FALSE(timing.percentile(80).overflow);
  EXPECT_TRUE(timing.percentile(101).overflow);
}

TEST(ShutterSlotTiming, MeanDoesNotOverflowWhenFiniteSamplesWouldOverflowASum) {
  shutter_timing_aggregate_t timing;
  const auto large = std::numeric_limits<double>::max();
  timing.observe(large);
  timing.observe(large);
  EXPECT_TRUE(std::isfinite(timing.mean_us()));
  EXPECT_DOUBLE_EQ(timing.mean_us(), large);
  EXPECT_TRUE(timing.percentile(95).overflow);
}
