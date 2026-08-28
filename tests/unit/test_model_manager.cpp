/**
 * @file tests/unit/test_model_manager.cpp
 * @brief Test bounded, identity-complete TensorRT cache names.
 */
#include "../tests_common.h"

#include <algorithm>
#include <string>
#include <string_view>

#include <src/model_manager.h>

TEST(ModelManagerTest, DepthEngineRecipesKeepLegacyAndFusedPlansIsolated) {
  const config::depth_model_info model {
    .name = "depth_anything_v2_fp16",
    .url = {},
  };
  constexpr std::string_view compatibility_tag = "trt11-sm120-onnxsha";
  using enum models::prod_zipdepth_convex2x::engine_io_e;

  const auto legacy = models::engine_filename(model, compatibility_tag);
  EXPECT_EQ(
    legacy,
    models::engine_filename(model, production_dav2, compatibility_tag)
  );
  EXPECT_EQ(
    legacy,
    "depth_anything_v2_fp16.trt-opt770x434-max1036-level5-v3."
    "trt11-sm120-onnxsha.engine"
  );

  const auto fused = models::engine_filename(
    model, production_dav2_zipdepth_convex2x, compatibility_tag
  );
  EXPECT_EQ(
    fused,
    "depth_anything_v2_fp16.trt-six-point-profiles-level5-v1."
    "trt11-sm120-onnxsha.engine"
  );
  EXPECT_NE(fused, legacy);
  EXPECT_TRUE(models::engine_filename(model, invalid, compatibility_tag).empty());
  EXPECT_EQ(models::depth_engine_builder_level, 5);
}

TEST(ModelManagerTest, OcrEngineFilenameIsBoundedAndCommitsTheCompleteIdentity) {
  const std::string compatibility_tag =
    "trt11_2_1_2-sm120-gpu0123456789abcdef-onnx" + std::string(64, 'a');
  const auto filename = models::ocr_engine_filename(compatibility_tag);

  EXPECT_EQ(
    filename,
    "ppocrv6_tiny_det_modelopt_fp16.cache-"
    "ddea1dd1468273af92bf7f29accba1c2d4972261bd312d2a88b052810f25acd6.engine"
  );
  EXPECT_EQ(filename, models::ocr_engine_filename(compatibility_tag));
  EXPECT_NE(filename, models::ocr_engine_filename(compatibility_tag + "-changed"));
  EXPECT_LT(filename.size() + std::string_view(".part").size(), 128u);

  constexpr std::string_view prefix = "ppocrv6_tiny_det_modelopt_fp16.cache-";
  constexpr std::string_view suffix = ".engine";
  ASSERT_TRUE(filename.starts_with(prefix));
  ASSERT_TRUE(filename.ends_with(suffix));
  const std::string_view token(filename.data() + prefix.size(),
                               filename.size() - prefix.size() - suffix.size());
  EXPECT_EQ(token.size(), 64u);
  EXPECT_TRUE(std::all_of(token.begin(), token.end(), [](const char ch) {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
  }));
}
