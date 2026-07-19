#ifdef _WIN32

  #include "src/platform/windows/ar_glasses.h"

  #include <chrono>

  #include <gtest/gtest.h>

using namespace std::chrono_literals;

TEST(ArGlassesMode, SelectsNormalForNativeTwoDimensionalMode) {
  EXPECT_EQ(
    ar_glasses::classify_mode(1920, 1080),
    ar_glasses::presentation_mode_e::normal
  );
}

TEST(ArGlassesMode, SelectsFullSbsForDoubleWidthMode) {
  EXPECT_EQ(
    ar_glasses::classify_mode(3840, 1080),
    ar_glasses::presentation_mode_e::sbs_ai
  );
}

TEST(ArGlassesMode, RejectsUnrecognizedModes) {
  EXPECT_EQ(
    ar_glasses::classify_mode(2560, 1080),
    ar_glasses::presentation_mode_e::unsupported
  );
  EXPECT_EQ(
    ar_glasses::classify_mode(3840, 2160),
    ar_glasses::presentation_mode_e::unsupported
  );
}

TEST(ArGlassesDiscovery, RecognizesSpecificModelsAndNames) {
  EXPECT_TRUE(ar_glasses::is_recognized_ar_display("DISPLAY:TCL03D4", "Generic Monitor"));
  EXPECT_TRUE(ar_glasses::is_recognized_ar_display("DISPLAY:ABC1234", "XREAL Air 2 Pro"));
  EXPECT_TRUE(ar_glasses::is_recognized_ar_display("DISPLAY:ABC1234", "SmartGlasses"));
}

TEST(ArGlassesDiscovery, DoesNotGuessFromOrdinaryMonitorNames) {
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:SMKD1CE", "Apollo AR Des"));
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:GSM1234", "LG ULTRAGEAR"));
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:AUS4321", "ROG PG32UCDM"));
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:ACI9999", "ARZOPA Portable Monitor"));
}

TEST(ArGlassesOwnership, RenewedRemoteConnectWindowBlocksLocalPresentation) {
  ar_glasses::remote_virtual_display_ended();
  ASSERT_TRUE(ar_glasses::remote_virtual_display_starting(0ms));

  // This is called after display creation, encoder probing, and app preparation. Its fresh lease
  // must remain visible to the local topology controller until RTSP activates or the lease ends.
  ar_glasses::remote_virtual_display_awaiting_client(0ms);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_blocks_local());

  ar_glasses::remote_virtual_display_ended();
  EXPECT_FALSE(ar_glasses::remote_virtual_display_blocks_local());
}

TEST(ArGlassesOwnership, ReservationTracksLongConfiguredPingTimeout) {
  EXPECT_EQ(
    ar_glasses::detail::remote_pending_duration(120s),
    122s
  );
  EXPECT_EQ(
    ar_glasses::detail::remote_pending_duration(-1ms),
    2s
  );
}

#endif
