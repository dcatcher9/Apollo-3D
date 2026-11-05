#include "src/platform/windows/shutter_presentation_clock.h"

#include <gtest/gtest.h>

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
