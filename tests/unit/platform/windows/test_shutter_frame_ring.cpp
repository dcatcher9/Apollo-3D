#ifdef _WIN32
  #include "src/platform/windows/display.h"

  #include <gtest/gtest.h>

TEST(LocalShutterPresentationGpu, BoundsPublicationAndRetainsAnExactStereoPair) {
  const auto result = platf::dxgi::probe_local_shutter_frame_ring();
  if (!result.available) {
    GTEST_SKIP() << "D3D11 shared keyed textures are unavailable on this adapter";
  }
  ASSERT_TRUE(result.consumer_progress_during_publication);
  EXPECT_TRUE(result.retained_pair_not_republished);
  EXPECT_TRUE(result.publication_bounded);
  EXPECT_TRUE(result.current_pair_immutable);
  EXPECT_TRUE(result.eye_crop_exact);
  EXPECT_TRUE(result.next_pair_published);
}

TEST(LocalShutterPresentationGpu, SwitchesOnlyToAnAuthorizedCompletePairFromAnotherProducer) {
  const auto result = platf::dxgi::probe_local_shutter_pair_handoff();
  if (!result.available) {
    GTEST_SKIP() << "Shared keyed textures and fences are unavailable on this adapter";
  }
  RecordProperty("driver", result.used_warp ? "WARP" : "hardware");
  EXPECT_TRUE(result.unpublished_retained);
  EXPECT_TRUE(result.unfenced_retained);
  EXPECT_TRUE(result.denied_retained);
  EXPECT_TRUE(result.adopted_exact_pair);
}
#endif
