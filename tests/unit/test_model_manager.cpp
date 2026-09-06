/**
 * @file tests/unit/test_model_manager.cpp
 * @brief Test bounded, identity-complete TensorRT cache names.
 */
#include "../tests_common.h"

#include <algorithm>
#include <string>
#include <string_view>

#include <src/model_manager.h>

TEST(ModelManagerTest, DepthEngineFilenameIsBoundedAndUsesTheFusedRecipe) {
  const config::depth_model_info model {
    .name = "depth_anything_v2_fp16",
    .url = {},
  };
  constexpr std::string_view compatibility_tag = "trt11-sm120-onnxsha";
  const auto filename = models::engine_filename(model, compatibility_tag);
  EXPECT_TRUE(filename.starts_with("depth_anything_v2_fp16.trt-6high-point-l5-v2.cache-"));
  EXPECT_TRUE(filename.ends_with(".engine"));
  EXPECT_EQ(filename, models::engine_filename(model, compatibility_tag));
  EXPECT_NE(filename, models::engine_filename(model, std::string(compatibility_tag) + "-changed"));
  EXPECT_LT(filename.size() + 5u, 128u);
  EXPECT_EQ(models::engine_filename(model),
            "depth_anything_v2_fp16.trt-6high-point-l5-v2.engine");
  EXPECT_EQ(models::depth_engine_builder_level, 5);
}

TEST(ModelManagerTest, CurrentDepthFilenameRejectsTruncatedHashesAndPaths) {
  const config::depth_model_info model {
    .name = std::string(models::prod_zipdepth_convex2x::logical_model), .url = {},
  };
  const auto filename = models::engine_filename(model, std::string(4096u, 'a'));
  EXPECT_TRUE(models::is_current_depth_engine_filename(filename));
  EXPECT_FALSE(models::is_current_depth_engine_filename(filename.substr(1)));
  EXPECT_FALSE(models::is_current_depth_engine_filename("../" + filename));
  EXPECT_FALSE(models::is_current_depth_engine_filename("..\\" + filename));
  EXPECT_FALSE(models::is_current_depth_engine_filename(filename + ".part"));
  auto nonhex = filename;
  nonhex[nonhex.size() - std::string_view(".engine").size() - 1u] = 'g';
  EXPECT_FALSE(models::is_current_depth_engine_filename(nonhex));
  const std::filesystem::path build_assets {
    "E:/Git/Repo/Apollo-3D/cmake-build-relwithdebinfo/assets"
  };
  EXPECT_LT((build_assets / (filename + ".part")).native().size(), 260u);
  EXPECT_TRUE(models::engine_cache_path_fits_native_limit(build_assets / filename));
  EXPECT_FALSE(models::engine_cache_path_fits_native_limit({}));
#ifdef _WIN32
  const std::filesystem::path prefix {"C:/"};
  EXPECT_TRUE(models::engine_cache_path_fits_native_limit(prefix / std::string(251u, 'a')));
  EXPECT_FALSE(models::engine_cache_path_fits_native_limit(prefix / std::string(252u, 'a')));
#endif
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
