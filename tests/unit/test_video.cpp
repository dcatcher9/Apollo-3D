/**
 * @file tests/unit/test_video.cpp
 * @brief Test src/video.*.
 */
#include "../tests_common.h"

#include <tuple>

#include <src/video.h>
#include <src/video_colorspace.h>

namespace {
  float apply_color_vector(const float (&color_vector)[4], float red, float green, float blue) {
    return color_vector[0] * red + color_vector[1] * green + color_vector[2] * blue + color_vector[3];
  }
}  // namespace

TEST(ColorVectorsTest, LimitedRangeUnormUsesTargetBitDepth) {
  for (const auto bit_depth : {8u, 10u}) {
    const auto *vectors = video::color_vectors_from_colorspace(
      {video::colorspace_e::rec709, false, bit_depth},
      true
    );
    ASSERT_NE(vectors, nullptr);

    const auto max_value = static_cast<float>((1u << bit_depth) - 1u);
    const auto scale = static_cast<float>(1u << (bit_depth - 8u));
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 0.0f, 0.0f, 0.0f), 16.0f * scale / max_value, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 1.0f, 1.0f, 1.0f), 235.0f * scale / max_value, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_u, 0.0f, 0.0f, 0.0f), 128.0f * scale / max_value, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_v, 1.0f, 1.0f, 1.0f), 128.0f * scale / max_value, 1e-6f);
  }
}

TEST(ColorVectorsTest, FullRangeUnormUsesTargetBitDepth) {
  for (const auto bit_depth : {8u, 10u}) {
    const auto *vectors = video::color_vectors_from_colorspace(
      {video::colorspace_e::rec709, true, bit_depth},
      true
    );
    ASSERT_NE(vectors, nullptr);

    const auto max_value = static_cast<float>((1u << bit_depth) - 1u);
    const auto neutral_chroma = static_cast<float>(1u << (bit_depth - 1u)) / max_value;
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 0.0f, 0.0f, 0.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 1.0f, 1.0f, 1.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_u, 0.0f, 0.0f, 0.0f), neutral_chroma, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_v, 1.0f, 1.0f, 1.0f), neutral_chroma, 1e-6f);
  }
}

TEST(ColorVectorsTest, UintOutputRoundsBt2020LimitedValues) {
  const auto *vectors = video::color_vectors_from_colorspace(
    {video::colorspace_e::bt2020, false, 10},
    false
  );
  ASSERT_NE(vectors, nullptr);

  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_y, 0.0f, 0.0f, 0.0f)), 64u);
  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_y, 1.0f, 1.0f, 1.0f)), 940u);
  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_u, 0.0f, 0.0f, 0.0f)), 512u);
  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_v, 1.0f, 0.0f, 0.0f)), 960u);
}

struct EncoderTest: PlatformTestSuite, testing::WithParamInterface<video::encoder_t *> {
  void SetUp() override {
    auto &encoder = *GetParam();
    if (!video::validate_encoder(encoder, false)) {
      // Encoder failed validation,
      // if it's software - fail, otherwise skip
      if (encoder.name == "software") {
        FAIL() << "Software encoder not available";
      } else {
        GTEST_SKIP() << "Encoder not available";
      }
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
  EncoderVariants,
  EncoderTest,
  testing::Values(
#if !defined(__APPLE__)
    &video::nvenc,
#endif
#ifdef _WIN32
    &video::amdvce,
    &video::quicksync,
#endif
#ifdef __linux__
    &video::vaapi,
#endif
#ifdef __APPLE__
    &video::videotoolbox,
#endif
    &video::software
  ),
  [](const auto &info) {
    return std::string(info.param->name);
  }
);

TEST_P(EncoderTest, ValidateEncoder) {
  // todo:: test something besides fixture setup
}

struct FramerateX100Test: testing::TestWithParam<std::tuple<std::int32_t, AVRational>> {};

TEST_P(FramerateX100Test, ConvertsToExpectedRational) {
  const auto &[x100, expected] = GetParam();
  const auto actual = video::framerate_x100_to_rational(x100);
  EXPECT_EQ(av_cmp_q(actual, expected), 0)
    << "expected " << expected.num << '/' << expected.den
    << ", got " << actual.num << '/' << actual.den;
}

INSTANTIATE_TEST_SUITE_P(
  FramerateX100Tests,
  FramerateX100Test,
  testing::Values(
    std::make_tuple(2397, AVRational {24000, 1001}),
    std::make_tuple(2398, AVRational {24000, 1001}),
    std::make_tuple(2500, AVRational {25, 1}),
    std::make_tuple(2997, AVRational {30000, 1001}),
    std::make_tuple(3000, AVRational {30, 1}),
    std::make_tuple(5994, AVRational {60000, 1001}),
    std::make_tuple(6000, AVRational {60, 1}),
    std::make_tuple(11988, AVRational {120000, 1001}),
    std::make_tuple(23976, AVRational {240000, 1001}),
    std::make_tuple(9498, AVRational {4749, 50})
  )
);
