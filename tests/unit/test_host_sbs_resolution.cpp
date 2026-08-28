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

TEST(HostSbsResolutionTest, ProjectsSubtitleSafeRowsIntoEveryAuthenticatedField) {
  struct geometry_case_t {
    std::uint32_t source_width;
    std::uint32_t source_height;
    int field_width;
    int field_height;
    std::uint32_t roi_top;
    std::uint32_t roi_bottom;
  };
  constexpr std::array cases {
    geometry_case_t {1920u, 1080u, 770, 434, 325u, 430u},
    geometry_case_t {2560u, 1440u, 770, 434, 325u, 430u},
    geometry_case_t {3840u, 2160u, 770, 434, 325u, 430u},
    geometry_case_t {2560u, 1080u, 1022, 434, 289u, 429u},
    geometry_case_t {3440u, 1440u, 1036, 434, 287u, 429u},
    geometry_case_t {5120u, 2160u, 1022, 434, 289u, 429u},
    geometry_case_t {1080u, 1920u, 434, 770, 709u, 768u},
    geometry_case_t {1440u, 2560u, 434, 770, 709u, 768u},
    geometry_case_t {2160u, 3840u, 434, 770, 709u, 768u},
    geometry_case_t {1080u, 2560u, 434, 1022, 961u, 1020u},
    geometry_case_t {1440u, 3440u, 434, 1036, 975u, 1034u},
    geometry_case_t {2160u, 5120u, 434, 1022, 961u, 1020u},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(
      std::to_string(test_case.source_width) + "x" +
      std::to_string(test_case.source_height)
    );
    const auto geometry = models::fit_subtitle_analysis_geometry(
      test_case.source_width,
      test_case.source_height,
      {test_case.field_width, test_case.field_height}
    );
    ASSERT_TRUE(geometry.valid());
    EXPECT_EQ(geometry.field_width, static_cast<std::uint32_t>(test_case.field_width));
    EXPECT_EQ(geometry.field_height, static_cast<std::uint32_t>(test_case.field_height));
    EXPECT_EQ(geometry.roi_top, test_case.roi_top);
    EXPECT_EQ(geometry.roi_bottom, test_case.roi_bottom);
  }

  EXPECT_FALSE(models::fit_subtitle_analysis_geometry(1920u, 1080u, {1008, 434}).valid());
  EXPECT_FALSE(models::fit_subtitle_analysis_geometry(0u, 1080u, {770, 434}).valid());
  EXPECT_FALSE(models::fit_subtitle_analysis_geometry(1920u, 0u, {770, 434}).valid());
}

TEST(HostSbsResolutionTest, AdmitsAllExactConvex2xSubtitleFields) {
  struct geometry_case_t {
    std::uint32_t source_width;
    std::uint32_t source_height;
    int field_width;
    int field_height;
  };
  constexpr std::array cases {
    geometry_case_t {1920u, 1080u, 1540, 868},
    geometry_case_t {2560u, 1080u, 2044, 868},
    geometry_case_t {3440u, 1440u, 2072, 868},
    geometry_case_t {1080u, 1920u, 868, 1540},
    geometry_case_t {1080u, 2560u, 868, 2044},
    geometry_case_t {1440u, 3440u, 868, 2072},
  };
  for (const auto &test_case : cases) {
    const models::depth_tensor_shape_t field {
      test_case.field_width, test_case.field_height,
    };
    const models::depth_tensor_content_rect_t content {
      0u,
      0u,
      static_cast<std::uint32_t>(test_case.field_width),
      static_cast<std::uint32_t>(test_case.field_height),
    };
    const auto geometry = models::fit_subtitle_analysis_geometry(
      test_case.source_width, test_case.source_height, field, content
    );
    ASSERT_TRUE(geometry.valid());
    EXPECT_EQ(geometry.field_width, static_cast<std::uint32_t>(test_case.field_width));
    EXPECT_EQ(geometry.field_height, static_cast<std::uint32_t>(test_case.field_height));
    EXPECT_EQ(geometry.tensor_content, content);
  }

  EXPECT_FALSE(models::fit_subtitle_analysis_geometry(
    1920u, 1080u, {1542, 868}, {0u, 0u, 1542u, 868u}
  ).valid()) << "the live-field allowlist must not become a generic 2x shape predicate";
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

TEST(HostSbsResolutionTest, LetterboxesMeasuredWindowWithoutCroppingIt) {
  const models::depth_source_rect_t video {820u, 510u, 2471u, 1439u};
  const auto plan = models::plan_host_sbs_v2_video_region(
    video,
    3840u,
    2160u,
    {770, 434}
  );
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->tensor_shape, (models::depth_tensor_shape_t {770, 434}));
  EXPECT_EQ(plan->source_rect, video);
  EXPECT_GT(plan->padded_area_fraction, 0.0f);
  EXPECT_TRUE(plan->tensor_content.valid(plan->tensor_shape));
  EXPECT_TRUE(
    plan->tensor_content.left > 0u || plan->tensor_content.top > 0u ||
    plan->tensor_content.right < plan->tensor_shape.width ||
    plan->tensor_content.bottom < plan->tensor_shape.height
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
  EXPECT_TRUE(plan->tensor_content.full(plan->tensor_shape));
  EXPECT_FLOAT_EQ(plan->padded_area_fraction, 0.0f);
}

TEST(HostSbsResolutionTest, KeepsAFullCaptureVideoOnTheOrdinaryFullFrameRoute) {
  EXPECT_FALSE(models::plan_host_sbs_v2_video_region(
    {0u, 0u, 3840u, 2160u},
    3840u,
    2160u,
    {770, 434}
  ));
}

TEST(HostSbsResolutionTest, SupportsArbitraryWindowAspectsAndSmallWindows) {
  struct case_t {
    models::depth_source_rect_t rect;
    bool pillarbox;
  };
  constexpr std::array cases {
    case_t {{100u, 100u, 1700u, 1300u}, true},   // 4:3
    case_t {{100u, 100u, 1000u, 1000u}, true},   // square
    case_t {{100u, 100u, 700u, 1100u}, true},    // portrait
    case_t {{100u, 100u, 2500u, 1000u}, false},  // ultrawide
    case_t {{0u, 0u, 640u, 360u}, false},        // exact-aspect upscale
  };
  for (const auto &test_case : cases) {
    const auto plan = models::plan_host_sbs_v2_video_region(
      test_case.rect, 3840u, 2160u, {770, 434}
    );
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->source_rect, test_case.rect);
    EXPECT_TRUE(plan->tensor_content.valid(plan->tensor_shape));
    if (test_case.pillarbox) {
      EXPECT_GT(plan->tensor_content.left, 0u);
      EXPECT_EQ(plan->tensor_content.top, 0u);
      EXPECT_EQ(plan->tensor_content.bottom, 434u);
    } else if (!plan->tensor_content.full(plan->tensor_shape)) {
      EXPECT_EQ(plan->tensor_content.left, 0u);
      EXPECT_EQ(plan->tensor_content.right, 770u);
      EXPECT_LT(plan->tensor_content.bottom - plan->tensor_content.top, 434u);
    }
  }
}

TEST(HostSbsResolutionTest, ContentRectUsesDeterministicContainFitAndTrailingOddPad) {
  const auto square = models::plan_host_sbs_v2_video_region(
    {100u, 100u, 1000u, 1000u}, 3840u, 2160u, {770, 434}
  );
  ASSERT_TRUE(square);
  EXPECT_EQ(
    square->tensor_content,
    (models::depth_tensor_content_rect_t {168u, 0u, 602u, 434u})
  );

  // 640x360 differs from the authenticated 770x434 aspect by less than one tensor row. The
  // integer contain fit keeps every source pixel and assigns the odd one-row pad to the trailing
  // edge, so all components reproduce the same half-open rectangle without float rounding.
  const auto almost_exact = models::plan_host_sbs_v2_video_region(
    {0u, 0u, 640u, 360u}, 3840u, 2160u, {770, 434}
  );
  ASSERT_TRUE(almost_exact);
  EXPECT_EQ(
    almost_exact->tensor_content,
    (models::depth_tensor_content_rect_t {0u, 0u, 770u, 433u})
  );
}

TEST(HostSbsResolutionTest, SubtitleProjectionIsOffsetInsideTensorContent) {
  constexpr models::depth_tensor_content_rect_t content {168u, 0u, 602u, 434u};
  constexpr auto geometry = models::fit_subtitle_analysis_geometry(
    900u, 900u, {770, 434}, content
  );
  static_assert(geometry.valid());
  EXPECT_EQ(geometry.tensor_content, content);
  EXPECT_GE(geometry.roi_top, content.top);
  EXPECT_LE(geometry.roi_bottom, content.bottom);
  EXPECT_GT(geometry.ribbon_min_bottom, geometry.roi_top);
  EXPECT_LE(geometry.ribbon_min_bottom, geometry.roi_bottom);
  EXPECT_EQ(geometry.field_width, 770u);
  EXPECT_EQ(geometry.field_height, 434u);
}

TEST(HostSbsResolutionTest, SubtitleProjectionUsesExactVerticallyPaddedContent) {
  constexpr models::depth_tensor_content_rect_t content {0u, 0u, 770u, 433u};
  constexpr auto geometry = models::fit_subtitle_analysis_geometry(
    2536u, 1427u, {770, 434}, content
  );
  static_assert(geometry.valid());
  EXPECT_EQ(geometry.roi_top, 324u);
  EXPECT_EQ(geometry.ribbon_min_bottom, 428u);
  EXPECT_EQ(geometry.roi_bottom, 429u);
}

TEST(HostSbsResolutionTest, RejectsInvalidGeometryAndUnauthenticatedTensorOnly) {
  EXPECT_FALSE(models::plan_host_sbs_v2_video_region(
    {0u, 0u, 4000u, 2160u}, 3840u, 2160u, {770, 434}
  ));
  EXPECT_FALSE(models::plan_host_sbs_v2_video_region(
    {0u, 0u, 1920u, 1080u}, 3840u, 2160u, {1008, 434}
  ));
}
