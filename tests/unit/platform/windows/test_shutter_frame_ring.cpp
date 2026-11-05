#ifdef _WIN32
  #include "src/platform/windows/display.h"

  #include <gtest/gtest.h>

TEST(LocalShutterPresentationGpu, BoundsPublicationAndRetainsAnExactStereoPair) {
  const auto result = platf::dxgi::probe_local_shutter_frame_ring();
  if (!result.available) {
    GTEST_SKIP() << "D3D11 shared keyed textures are unavailable on this adapter";
  }
  EXPECT_TRUE(result.publication_bounded);
  EXPECT_TRUE(result.current_pair_immutable);
  EXPECT_TRUE(result.eye_crop_exact);
  EXPECT_TRUE(result.next_pair_published);
}
#endif
