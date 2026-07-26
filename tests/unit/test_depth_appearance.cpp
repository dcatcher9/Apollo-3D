#include "../tests_common.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace {
  float srgb_oetf(float value) {
    const float x = std::clamp(value, 0.0f, 1.0f);
    return x <= 0.0031308f ?
             x * 12.92f :
             1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
  }

  float post_tonemap_max_rgb(std::array<float, 3> color, float exposure) {
    for (auto &channel : color) {
      channel = std::max(channel * exposure, 0.0f);
    }
    const float luminance =
      color[0] * 0.2126f + color[1] * 0.7152f + color[2] * 0.0722f;
    for (auto &channel : color) {
      channel /= 1.0f + std::max(luminance, 0.0f);
    }
    const float peak = std::max({color[0], color[1], color[2]});
    for (auto &channel : color) {
      channel /= std::max(peak, 1.0f);
      channel = srgb_oetf(channel);
    }
    return std::max({color[0], color[1], color[2]});
  }

  float capture_max_rgb(const std::array<float, 3> &color, float exposure) {
    return std::max({color[0], color[1], color[2]}) * exposure;
  }
}  // namespace

TEST(HostSbsSceneCutTest, CaptureOrdinalSurvivesHdrExposureThatReversesTonemappedRank) {
  // These differently coloured scRGB pixels reverse maxRGB order after Apollo's
  // luminance-preserving Reinhard mapping when global exposure changes. This is not a numeric
  // edge case: the per-pixel luminance divisor is intentionally chroma dependent.
  constexpr std::array<float, 3> warm {
    1.22392758f, 0.78133029f, 0.02648114f
  };
  constexpr std::array<float, 3> neutral {
    3.46400915f, 3.41940613f, 2.77509378f
  };

  EXPECT_LT(post_tonemap_max_rgb(warm, 1.0f), post_tonemap_max_rgb(neutral, 1.0f));
  EXPECT_GT(post_tonemap_max_rgb(warm, 3.0f), post_tonemap_max_rgb(neutral, 3.0f));

  // Point-sampled capture maxRGB remains a positive scalar multiple, so its ordering cannot
  // reverse. The production shader stores this signal before tone mapping and spatial filtering.
  EXPECT_LT(capture_max_rgb(warm, 1.0f), capture_max_rgb(neutral, 1.0f));
  EXPECT_LT(capture_max_rgb(warm, 3.0f), capture_max_rgb(neutral, 3.0f));
}
