#include "src/sbs_bench_harness.h"

#include <limits>
#include <string>

#include <gtest/gtest.h>

namespace {
  TEST(SbsBenchGeometryTest, ResolvesOrdinaryPackedGeometry) {
    std::string error;
    const auto geometry = sbs_bench::detail::resolve_sbs_geometry(
      1920, 1080, 8192, error
    );

    ASSERT_TRUE(geometry) << error;
    EXPECT_EQ(geometry->eye_width, 1920u);
    EXPECT_EQ(geometry->eye_height, 1080u);
    EXPECT_EQ(geometry->sbs_width, 3840u);
    EXPECT_EQ(geometry->sbs_height, 1080u);
    EXPECT_FLOAT_EQ(geometry->content_scale_x, 1.0f);
    EXPECT_FLOAT_EQ(geometry->content_scale_y, 1.0f);
  }

  TEST(SbsBenchGeometryTest, AppliesPackedWidthCap) {
    std::string error;
    const auto geometry = sbs_bench::detail::resolve_sbs_geometry(
      3840, 2160, 3840, error
    );

    ASSERT_TRUE(geometry) << error;
    EXPECT_EQ(geometry->eye_width, 1920u);
    EXPECT_EQ(geometry->eye_height, 1080u);
    EXPECT_EQ(geometry->sbs_width, 3840u);
    EXPECT_EQ(geometry->sbs_height, 1080u);
    EXPECT_FLOAT_EQ(geometry->content_scale_x, 1.0f);
    EXPECT_FLOAT_EQ(geometry->content_scale_y, 1.0f);
  }

  TEST(SbsBenchGeometryTest, RejectsPackedWidthBeyondWireBound) {
    std::string error;
    const auto geometry = sbs_bench::detail::resolve_sbs_geometry(
      16384, 16384, std::numeric_limits<int>::max(), error
    );

    EXPECT_FALSE(geometry);
    EXPECT_NE(error.find("32768x16384"), std::string::npos);
    EXPECT_NE(error.find("16384x16384"), std::string::npos);
  }

  TEST(SbsBenchGeometryTest, RejectsInvalidSourceAndCap) {
    std::string error;
    EXPECT_FALSE(sbs_bench::detail::resolve_sbs_geometry(0, 1080, 8192, error));
    EXPECT_FALSE(sbs_bench::detail::resolve_sbs_geometry(16385, 1080, 8192, error));
    EXPECT_FALSE(sbs_bench::detail::resolve_sbs_geometry(1920, 1080, 0, error));
    EXPECT_EQ(error, "source dimensions or output controls are invalid");
  }
}  // namespace
