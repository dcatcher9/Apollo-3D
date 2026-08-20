#include "src/sbs_bench_harness.h"

#include <limits>
#include <string>

#include <gtest/gtest.h>

#ifdef _WIN32
namespace {
  TEST(SbsBenchGeometryTest, ResolvesOrdinaryPackedGeometry) {
    std::string error;
    const auto geometry = sbs_bench::detail::resolve_sbs_geometry(
      1920, 1080, 0, 0, 1.0, 8192, error
    );

    ASSERT_TRUE(geometry) << error;
    EXPECT_EQ(geometry->eye_width, 1920u);
    EXPECT_EQ(geometry->eye_height, 1080u);
    EXPECT_EQ(geometry->sbs_width, 3840u);
    EXPECT_EQ(geometry->sbs_height, 1080u);
    EXPECT_FLOAT_EQ(geometry->content_scale_x, 1.0f);
    EXPECT_FLOAT_EQ(geometry->content_scale_y, 1.0f);
  }

  TEST(SbsBenchGeometryTest, AppliesPackedWidthCapBeforeWireBounds) {
    std::string error;
    const auto geometry = sbs_bench::detail::resolve_sbs_geometry(
      1920, 1080, 0, 20000, 1.0, 8192, error
    );

    ASSERT_TRUE(geometry) << error;
    EXPECT_EQ(geometry->eye_width, 4096u);
    EXPECT_EQ(geometry->eye_height, 2304u);
    EXPECT_EQ(geometry->sbs_width, 8192u);
    EXPECT_EQ(geometry->sbs_height, 2304u);
  }

  TEST(SbsBenchGeometryTest, RejectsOversizedAdaptiveHeight) {
    std::string error;
    const auto geometry = sbs_bench::detail::resolve_sbs_geometry(
      1920, 1080, 1, 20000, 1.0, 8192, error
    );

    EXPECT_FALSE(geometry);
    EXPECT_NE(error.find("2x20000"), std::string::npos);
    EXPECT_NE(error.find("16384x16384"), std::string::npos);
  }

  TEST(SbsBenchGeometryTest, RejectsOversizedDerivedPortraitHeight) {
    std::string error;
    const auto geometry = sbs_bench::detail::resolve_sbs_geometry(
      100, 1000, 0, 20000, 1.0, 8192, error
    );

    EXPECT_FALSE(geometry);
    EXPECT_NE(error.find("4000x20000"), std::string::npos);
  }

  TEST(SbsBenchGeometryTest, RejectsExtremeInputsWithoutIntegerOverflow) {
    std::string error;
    const auto geometry = sbs_bench::detail::resolve_sbs_geometry(
      1,
      16384,
      std::numeric_limits<int>::max(),
      0,
      1.0,
      std::numeric_limits<int>::max(),
      error
    );

    EXPECT_FALSE(geometry);
    EXPECT_NE(error.find("16384x16384"), std::string::npos);
  }
}  // namespace
#endif
