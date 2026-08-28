/**
 * @file tests/unit/test_host_sbs_convex2x_contract.cpp
 * @brief Pure shape and named-I/O tests for the frozen convex-2x integration boundary.
 */

#include <src/host_sbs_resolution.h>
#include <src/model_manager.h>
#include <src/prod_zipdepth_convex2x.h>
#include <src/video_depth_estimator.h>

#include <array>
#include <gtest/gtest.h>

namespace {

  TEST(HostSbsConvex2xContract, DoublesEveryAuthenticatedProductionShape) {
    using models::depth_tensor_shape_t;
    constexpr std::array coarse_shapes {
      depth_tensor_shape_t {770, 434},
      depth_tensor_shape_t {1022, 434},
      depth_tensor_shape_t {1036, 434},
      depth_tensor_shape_t {434, 770},
      depth_tensor_shape_t {434, 1022},
      depth_tensor_shape_t {434, 1036},
    };
    for (const auto coarse : coarse_shapes) {
      const auto refined = models::host_sbs_convex2x_field_shape(coarse);
      ASSERT_TRUE(refined.has_value());
      EXPECT_EQ(refined->width, 2 * coarse.width);
      EXPECT_EQ(refined->height, 2 * coarse.height);
      EXPECT_TRUE(models::host_sbs_convex2x_shape_relation(coarse, *refined));
      EXPECT_TRUE(models::host_sbs_convex2x_operator_shape_is_supported(coarse));
      EXPECT_TRUE(
        models::prod_zipdepth_convex2x::live_geometry_shape_relation(
          static_cast<std::uint32_t>(coarse.width),
          static_cast<std::uint32_t>(coarse.height),
          static_cast<std::uint32_t>(refined->width),
          static_cast<std::uint32_t>(refined->height)
        )
      );
    }
  }

  TEST(HostSbsConvex2xContract, RejectsUnauthenticatedShapesAndRelations) {
    EXPECT_FALSE(models::host_sbs_convex2x_field_shape({1008, 434}));
    EXPECT_FALSE(models::host_sbs_convex2x_operator_shape_is_supported({1008, 434}));
    EXPECT_FALSE(models::host_sbs_convex2x_field_shape({0, 434}));
    EXPECT_FALSE(models::host_sbs_convex2x_shape_relation(
      {770, 434}, {1540, 867}
    ));
  }

  TEST(HostSbsConvex2xContract, DoublesOddContentEdgesWithoutRefitting) {
    constexpr models::depth_tensor_shape_t coarse_shape {770, 434};
    constexpr models::depth_tensor_content_rect_t coarse {17u, 9u, 752u, 421u};
    const auto refined = models::host_sbs_convex2x_content_rect(
      coarse, coarse_shape
    );
    ASSERT_TRUE(refined.has_value());
    EXPECT_EQ(
      *refined,
      (models::depth_tensor_content_rect_t {34u, 18u, 1504u, 842u})
    );
    const auto refined_shape = models::host_sbs_convex2x_field_shape(coarse_shape);
    ASSERT_TRUE(refined_shape.has_value());
    EXPECT_TRUE(refined->valid(*refined_shape));
    EXPECT_EQ(
      static_cast<std::uint64_t>(coarse.left) * refined_shape->width,
      static_cast<std::uint64_t>(refined->left) * coarse_shape.width
    );
    EXPECT_EQ(
      static_cast<std::uint64_t>(coarse.bottom) * refined_shape->height,
      static_cast<std::uint64_t>(refined->bottom) * coarse_shape.height
    );
  }

  TEST(HostSbsConvex2xContract, RejectsInvalidContent) {
    EXPECT_FALSE(models::host_sbs_convex2x_content_rect(
      {0u, 0u, 771u, 434u}, {770, 434}
    ));
    EXPECT_FALSE(models::host_sbs_convex2x_content_rect(
      {0u, 0u, 770u, 434u}, {1008, 434}
    ));
  }

  TEST(HostSbsConvex2xContract, RefinedRealizationDoesNotChangeAnalysisDomain) {
    models::depth_input_region_t first {
      .source_width = 3840u,
      .source_height = 2160u,
      .left = 101u,
      .top = 77u,
      .right = 2021u,
      .bottom = 1157u,
      .tensor_content = {0u, 0u, 770u, 433u},
      .analysis_generation = 9u,
      .authority = models::depth_analysis_authority_e::foreground_client,
    };
    const auto refined = models::host_sbs_convex2x_content_rect(
      first.tensor_content, {770, 434}
    );
    ASSERT_TRUE(refined.has_value());
    const auto second = first;
    EXPECT_TRUE(first.same_analysis_domain(second));
    EXPECT_EQ(*refined, (models::depth_tensor_content_rect_t {0u, 0u, 1540u, 866u}));
  }

  TEST(HostSbsConvex2xContract, ClassifiesOnlyExactNamedIoSets) {
    using enum models::prod_zipdepth_convex2x::engine_io_e;
    using models::prod_zipdepth_convex2x::classify_named_io;
    EXPECT_EQ(
      classify_named_io(2u, true, true, false),
      production_dav2
    );
    EXPECT_EQ(
      classify_named_io(2u, true, false, true),
      production_dav2_zipdepth_convex2x
    );
    EXPECT_EQ(classify_named_io(3u, true, true, true), invalid);
    EXPECT_EQ(classify_named_io(2u, true, true, true), invalid);
    EXPECT_EQ(classify_named_io(2u, false, false, true), invalid);
  }

  TEST(HostSbsConvex2xContract, FreezesOneHighPointProfilePerProductionShape) {
    using namespace models::prod_zipdepth_convex2x;
    static_assert(fixed_profile_shapes.size() == 6u);
    constexpr std::array expected {
      high_shape_t {1540u, 868u},
      high_shape_t {2044u, 868u},
      high_shape_t {2072u, 868u},
      high_shape_t {868u, 1540u},
      high_shape_t {868u, 2044u},
      high_shape_t {868u, 2072u},
    };
    EXPECT_EQ(fixed_profile_shapes, expected);
    for (std::uint32_t index = 0u; index < fixed_profile_shapes.size(); ++index) {
      const auto shape = fixed_profile_shapes[index];
      EXPECT_EQ(fixed_profile_index(shape.width, shape.height), index);
    }
    EXPECT_FALSE(fixed_profile_index(434u, 434u));
    EXPECT_FALSE(fixed_profile_index(1540u, 870u));
  }

  TEST(HostSbsConvex2xContract, EveryProfileFitsCheckedHighGridCapacities) {
    using namespace models::prod_zipdepth_convex2x;
    constexpr std::array expected_tile_counts {
      5335u, 7040u, 7150u, 5335u, 7040u, 7150u,
    };
    std::uint32_t observed_maximum = 0u;
    for (std::size_t index = 0u; index < fixed_profile_shapes.size(); ++index) {
      const auto shape = fixed_profile_shapes[index];
      const auto layout = models::detail::checked_near_identical_tile_layout(
        shape.width, shape.height
      );
      ASSERT_TRUE(layout);
      EXPECT_EQ(layout->group_count, expected_tile_counts[index]);
      EXPECT_EQ(layout->group_count, near_identical_tile_group_count(shape));
      EXPECT_EQ(layout->word_count, 4u * layout->group_count);
      EXPECT_EQ(layout->byte_count, 16u * layout->group_count);
      EXPECT_LE(layout->group_width, 65535u);
      EXPECT_LE(layout->group_height, 65535u);
      EXPECT_LE(shape.width, 65535u);
      EXPECT_LE(shape.height, 65535u);
      observed_maximum = std::max(observed_maximum, layout->group_count);
    }
    EXPECT_EQ(observed_maximum, near_identical_max_tile_group_count);
    EXPECT_EQ(observed_maximum, 7150u);
    EXPECT_TRUE(models::fused_depth_profile_contracts_match);
    EXPECT_FALSE(models::detail::checked_near_identical_tile_layout(0u, 868u));
    EXPECT_FALSE(models::detail::checked_near_identical_tile_layout(
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<std::uint32_t>::max()
    ));
  }

  TEST(HostSbsConvex2xContract, FreezesCompositeAndEmbeddedModelIdentity) {
    using namespace models::prod_zipdepth_convex2x;
    EXPECT_EQ(contract_schema, 2u);
    EXPECT_EQ(
      logical_model,
      "prod_dav2_zipdepth_c2x_high_opset18"
    );
    EXPECT_EQ(
      engine_recipe,
      "trt-6high-point-l5-v2"
    );
    EXPECT_EQ(
      dav2_onnx_sha256,
      "2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04"
    );
    EXPECT_EQ(
      fused_onnx_sha256,
      "0547dd046dead55057bb34a356d987559b2d93248e84600245f02df828d8bbb7"
    );
  }

  TEST(HostSbsConvex2xContract, LocalCompositeSelectionFallsBackOnlyWhenAbsent) {
    using enum models::detail::composite_depth_asset_resolution_e;
    using models::detail::resolve_composite_depth_asset;

    EXPECT_EQ(resolve_composite_depth_asset(false, false, false), legacy);
    EXPECT_EQ(resolve_composite_depth_asset(true, true, true), fused);
    EXPECT_EQ(resolve_composite_depth_asset(true, true, false), fail);
    EXPECT_EQ(resolve_composite_depth_asset(true, false, false), fail);
    EXPECT_EQ(resolve_composite_depth_asset(true, false, true), fail);
  }

  TEST(HostSbsConvex2xContract, PadsOnlyTensorRtOutputStorageTo512Bytes) {
    using models::detail::checked_tensorrt_output_allocation_bytes;
    EXPECT_EQ(models::detail::tensorrt_output_allocation_alignment, 512u);

    constexpr std::size_t coarse_logical = 770u * 434u * sizeof(float);
    constexpr std::size_t refined_logical = 1540u * 868u * sizeof(float);
    const auto coarse = checked_tensorrt_output_allocation_bytes(coarse_logical);
    const auto refined = checked_tensorrt_output_allocation_bytes(refined_logical);
    ASSERT_TRUE(coarse);
    ASSERT_TRUE(refined);
    EXPECT_EQ(coarse_logical, 1'336'720u);
    EXPECT_EQ(*coarse, 1'336'832u);
    EXPECT_EQ(refined_logical, 5'346'880u);
    EXPECT_EQ(*refined, 5'347'328u);
    const auto already_aligned =
      checked_tensorrt_output_allocation_bytes(1024u);
    ASSERT_TRUE(already_aligned);
    EXPECT_EQ(*already_aligned, 1024u);
    EXPECT_FALSE(checked_tensorrt_output_allocation_bytes(
      std::numeric_limits<std::size_t>::max()
    ));
  }

}  // namespace
