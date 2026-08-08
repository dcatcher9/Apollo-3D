/**
 * @file tests/unit/test_host_sbs_resolution.cpp
 * @brief Fast, CPU-only checks for Host SBS V2 resolution preflight.
 */
#include "../tests_common.h"

#include <array>
#include <cstdint>
#include <src/host_sbs_resolution.h>

TEST(HostSbsResolutionTest, EveryStandardMoonlight3dChoiceMapsToAuthenticatedShape) {
  struct resolution_case_t {
    std::uint32_t source_width;
    std::uint32_t source_height;
    int depth_width;
    int depth_height;
  };
  constexpr std::array cases {
    resolution_case_t {1920u, 1080u, 770, 434},
    resolution_case_t {2560u, 1440u, 770, 434},
    resolution_case_t {3840u, 2160u, 770, 434},
    resolution_case_t {2560u, 1080u, 1022, 434},
    resolution_case_t {3440u, 1440u, 1036, 434},
    resolution_case_t {5120u, 2160u, 1022, 434},
    resolution_case_t {1080u, 1920u, 434, 770},
    resolution_case_t {1440u, 2560u, 434, 770},
    resolution_case_t {2160u, 3840u, 434, 770},
    resolution_case_t {1080u, 2560u, 434, 1022},
    resolution_case_t {1440u, 3440u, 434, 1036},
    resolution_case_t {2160u, 5120u, 434, 1022},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(
      std::to_string(test_case.source_width) + "x" +
      std::to_string(test_case.source_height)
    );
    const auto fitted = models::fit_host_sbs_v2_depth_tensor_shape(
      test_case.source_width,
      test_case.source_height
    );
    EXPECT_EQ(fitted.width, test_case.depth_width);
    EXPECT_EQ(fitted.height, test_case.depth_height);
    EXPECT_TRUE(models::host_sbs_v2_depth_shape_is_authenticated(fitted));
    EXPECT_TRUE(models::host_sbs_v2_source_resolution_is_supported(
      test_case.source_width,
      test_case.source_height
    ));
    EXPECT_TRUE(models::host_sbs_v2_source_resolution_rejection_reason(
      test_case.source_width,
      test_case.source_height
    ).empty());
  }
}

TEST(HostSbsResolutionTest, UsesTensorAuthenticationInsteadOfAStreamSizeAllowlist) {
  // A smaller 16:9 custom stream is safe because it produces the same authenticated 770x434
  // tensor as the standard 1080p/1440p/4K choices.
  EXPECT_EQ(
    models::fit_host_sbs_v2_depth_tensor_shape(1280u, 720u),
    (models::depth_tensor_shape_t {770, 434})
  );
  EXPECT_TRUE(models::host_sbs_v2_source_resolution_is_supported(1280u, 720u));

  // These valid stream rasters fit uncalibrated tensor shapes and must therefore be rejected
  // before Host SBS creates an estimator or silently settles on terminal-flat output.
  EXPECT_FALSE(models::host_sbs_v2_source_resolution_is_supported(1920u, 1200u));
  EXPECT_FALSE(models::host_sbs_v2_source_resolution_is_supported(3840u, 1080u));
  EXPECT_FALSE(models::host_sbs_v2_source_resolution_is_supported(640u, 360u));
  EXPECT_FALSE(models::host_sbs_v2_source_resolution_is_supported(0u, 2160u));
  EXPECT_FALSE(models::host_sbs_v2_source_resolution_is_supported(3840u, 0u));

  // A huge raster can fit the same calibrated tensor aspect, but exact-area preprocessing scales
  // with source pixels. Keep live work bounded to the largest shipped 5K ultrawide choice.
  EXPECT_FALSE(models::host_sbs_v2_source_resolution_is_supported(7680u, 4320u));
  EXPECT_FALSE(models::host_sbs_v2_source_resolution_is_supported(5121u, 2160u));
  EXPECT_FALSE(models::host_sbs_v2_source_resolution_is_supported(2160u, 5121u));

  EXPECT_EQ(
    models::host_sbs_v2_source_resolution_rejection_reason(7680u, 4320u),
    "source raster exceeds the exact-area preprocessing budget"
  );
  EXPECT_EQ(
    models::host_sbs_v2_source_resolution_rejection_reason(1920u, 1200u),
    "fitted depth tensor is not authenticated"
  );
  EXPECT_EQ(
    models::host_sbs_v2_source_resolution_rejection_reason(0u, 2160u),
    "source extent is empty"
  );
}

TEST(HostSbsResolutionTest, TrimsMeasuredBrowserVideoToTheActiveTensorAspect) {
  const models::depth_source_rect_t video {820u, 510u, 2471u, 1439u};
  const auto plan = models::plan_host_sbs_v2_video_region(
    video,
    3840u,
    2160u,
    {770, 434}
  );
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->tensor_shape, (models::depth_tensor_shape_t {770, 434}));
  EXPECT_GT(plan->trimmed_area_fraction, 0.0f);
  EXPECT_LE(
    plan->trimmed_area_fraction,
    models::host_sbs_v2_max_video_region_trim_fraction
  );
  EXPECT_GE(plan->source_rect.left, video.left);
  EXPECT_GE(plan->source_rect.top, video.top);
  EXPECT_LE(plan->source_rect.right, video.right);
  EXPECT_LE(plan->source_rect.bottom, video.bottom);
  EXPECT_TRUE(
    plan->source_rect.left > video.left || plan->source_rect.top > video.top ||
    plan->source_rect.right < video.right || plan->source_rect.bottom < video.bottom
  );
}

TEST(HostSbsResolutionTest, KeepsAnExactTensorAspectRectangle) {
  const models::depth_source_rect_t video {100u, 200u, 1640u, 1068u};
  const auto plan = models::plan_host_sbs_v2_video_region(
    video,
    3840u,
    2160u,
    {770, 434}
  );
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->source_rect, video);
  EXPECT_FLOAT_EQ(plan->trimmed_area_fraction, 0.0f);
}

TEST(HostSbsResolutionTest, KeepsAFullCaptureVideoOnTheOrdinaryFullFrameRoute) {
  EXPECT_FALSE(models::plan_host_sbs_v2_video_region(
    {0u, 0u, 3840u, 2160u},
    3840u,
    2160u,
    {770, 434}
  ));
}

TEST(HostSbsResolutionTest, TrimsOnlyAProvablySmallNearAspectMiss) {
  // This slightly-wide rectangle rounds to an uncalibrated patch shape as-is. A narrow inward
  // trim removes the player edge while mapping through the ordinary fitter to 16:9.
  const models::depth_source_rect_t near_miss {100u, 100u, 1730u, 1010u};
  ASSERT_NE(
    models::fit_host_sbs_v2_depth_tensor_shape(
      near_miss.width(), near_miss.height()),
    (models::depth_tensor_shape_t {770, 434})
  );
  const auto plan = models::plan_host_sbs_v2_video_region(
    near_miss,
    3840u,
    2160u,
    {770, 434}
  );
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->tensor_shape, (models::depth_tensor_shape_t {770, 434}));
  EXPECT_GT(plan->trimmed_area_fraction, 0.0f);
  EXPECT_LE(
    plan->trimmed_area_fraction,
    models::host_sbs_v2_max_video_region_trim_fraction
  );
  EXPECT_GE(plan->source_rect.left, near_miss.left);
  EXPECT_GE(plan->source_rect.top, near_miss.top);
  EXPECT_LE(plan->source_rect.right, near_miss.right);
  EXPECT_LE(plan->source_rect.bottom, near_miss.bottom);
}

TEST(HostSbsResolutionTest, RejectsMaterialMismatchSmallVideoAndWrongDomainShape) {
  EXPECT_FALSE(models::plan_host_sbs_v2_video_region(
    {100u, 100u, 1700u, 1300u}, 3840u, 2160u, {770, 434}
  ));
  EXPECT_FALSE(models::plan_host_sbs_v2_video_region(
    {0u, 0u, 640u, 360u}, 3840u, 2160u, {770, 434}
  ));
  EXPECT_FALSE(models::plan_host_sbs_v2_video_region(
    {0u, 0u, 1920u, 1080u}, 3840u, 2160u, {1022, 434}
  ));
  EXPECT_FALSE(models::plan_host_sbs_v2_video_region(
    {0u, 0u, 4000u, 2160u}, 3840u, 2160u, {770, 434}
  ));
}
