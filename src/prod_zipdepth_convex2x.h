/**
 * @file src/prod_zipdepth_convex2x.h
 * @brief Frozen single-high-I/O DAV2 + ZipDepth convex-2x boundary names.
 */
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace models::prod_zipdepth_convex2x {

  inline constexpr std::uint32_t contract_schema = 2u;
  inline constexpr std::uint32_t scale = 2u;
  inline constexpr std::uint32_t neighbor_count = 9u;
  inline constexpr std::uint32_t subpixel_count = scale * scale;
  inline constexpr std::uint32_t mask_channel_count =
    neighbor_count * subpixel_count;

  inline constexpr std::string_view logical_model =
    "prod_dav2_zipdepth_c2x_high_opset18";
  inline constexpr std::string_view engine_recipe =
    "trt-6high-point-l5-v2";
  inline constexpr std::string_view dav2_onnx_sha256 =
    "2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04";
  inline constexpr std::string_view zipdepth_checkpoint_sha256 =
    "a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65";
  inline constexpr std::string_view fused_onnx_sha256 =
    "26684c5da8fdd4bdc5f1c9cf919cec8d1e2d027fbe95705a454f85d31eee2c23";

  inline constexpr std::string_view input = "pixel_values";
  inline constexpr std::string_view refined_output = "refined_depth";

  struct high_shape_t {
    std::uint32_t width;
    std::uint32_t height;

    constexpr bool operator==(const high_shape_t &) const = default;
  };

  /** Fixed HIGH TensorRT optimization-profile order for the dynamic fused graph.
   *
   * TensorRT 11.2 cannot compile the dynamic convex tail under a ranged H/W profile. Keeping
   * all six authenticated public 2x shapes as point profiles in one engine preserves one
   * input, one output, one context, and one enqueue while making every tail dimension constant.
   */
  inline constexpr std::array fixed_profile_shapes {
    high_shape_t {1540u, 868u},
    high_shape_t {2044u, 868u},
    high_shape_t {2072u, 868u},
    high_shape_t {868u, 1540u},
    high_shape_t {868u, 2044u},
    high_shape_t {868u, 2072u},
  };

  inline constexpr std::optional<std::uint32_t> fixed_profile_index(
    const std::uint32_t width,
    const std::uint32_t height
  ) noexcept {
    for (std::uint32_t index = 0u; index < fixed_profile_shapes.size(); ++index) {
      if (fixed_profile_shapes[index] == high_shape_t {width, height}) {
        return index;
      }
    }
    return std::nullopt;
  }

  inline constexpr std::uint32_t near_identical_tile_group_count(
    const high_shape_t shape
  ) noexcept {
    const auto groups_x = shape.width / 16u + (shape.width % 16u != 0u);
    const auto groups_y = shape.height / 16u + (shape.height % 16u != 0u);
    return groups_x * groups_y;
  }

  inline constexpr std::uint32_t near_identical_max_tile_group_count = []() {
    std::uint32_t maximum = 0u;
    for (const auto shape : fixed_profile_shapes) {
      const auto count = near_identical_tile_group_count(shape);
      maximum = count > maximum ? count : maximum;
    }
    return maximum;
  }();

  static_assert(near_identical_max_tile_group_count == 7150u);

  /** All six production orientations have authenticated refined-field limiters. */
  inline constexpr bool live_geometry_coarse_shape_is_supported(
    const std::uint32_t width,
    const std::uint32_t height
  ) noexcept {
    return width <= std::numeric_limits<std::uint32_t>::max() / scale &&
           height <= std::numeric_limits<std::uint32_t>::max() / scale &&
           fixed_profile_index(scale * width, scale * height).has_value();
  }

  inline constexpr bool live_geometry_field_shape_is_supported(
    const std::uint32_t width,
    const std::uint32_t height
  ) noexcept {
    return fixed_profile_index(width, height).has_value();
  }

  inline constexpr bool live_geometry_shape_relation(
    const std::uint32_t coarse_width,
    const std::uint32_t coarse_height,
    const std::uint32_t field_width,
    const std::uint32_t field_height
  ) noexcept {
    return live_geometry_coarse_shape_is_supported(coarse_width, coarse_height) &&
           field_width == scale * coarse_width &&
           field_height == scale * coarse_height &&
           live_geometry_field_shape_is_supported(field_width, field_height);
  }

  enum class engine_io_e : std::uint8_t {
    invalid,
    production_dav2_zipdepth_convex2x,
  };

  /** Pure named-I/O classification used before any TensorRT shape/address binding.
   *
   * Dtype and input/output-mode validation stays at the TensorRT boundary.  Exact tensor counts
   * prevent a partially exported or debug-output graph from being mistaken for production.
   */
  inline constexpr engine_io_e classify_named_io(
    const std::uint32_t tensor_count,
    const bool has_pixel_values,
    const bool has_refined_output
  ) noexcept {
    if (tensor_count == 2u && has_pixel_values && has_refined_output) {
      return engine_io_e::production_dav2_zipdepth_convex2x;
    }
    return engine_io_e::invalid;
  }

}  // namespace models::prod_zipdepth_convex2x
