/**
 * @file src/prod_zipdepth_convex2x.h
 * @brief Frozen one-engine DAV2 + ZipDepth convex-2x boundary names.
 */
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace models::prod_zipdepth_convex2x {

  inline constexpr std::uint32_t contract_schema = 1u;
  inline constexpr std::uint32_t scale = 2u;
  inline constexpr std::uint32_t neighbor_count = 9u;
  inline constexpr std::uint32_t subpixel_count = scale * scale;
  inline constexpr std::uint32_t mask_channel_count =
    neighbor_count * subpixel_count;

  inline constexpr std::string_view logical_model =
    "prod_dav2_zipdepth_convex2x_dynamic_opset18";
  inline constexpr std::string_view engine_recipe =
    "trt-six-point-profiles-level5-v1";
  inline constexpr std::string_view dav2_onnx_sha256 =
    "2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04";
  inline constexpr std::string_view zipdepth_checkpoint_sha256 =
    "a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65";
  inline constexpr std::string_view fused_onnx_sha256 =
    "959fc90097d7055b9c56cb140f432e0f5aed533476e8cedd6ec2baae097b287f";

  inline constexpr std::string_view dav2_input = "pixel_values";
  inline constexpr std::string_view guidance_input = "zip_pixel_values";
  inline constexpr std::string_view coarse_output = "predicted_depth";
  inline constexpr std::string_view refined_output = "refined_depth";

  struct coarse_shape_t {
    std::uint32_t width;
    std::uint32_t height;

    constexpr bool operator==(const coarse_shape_t &) const = default;
  };

  /** Fixed TensorRT optimization-profile order for the dynamic-shape fused graph.
   *
   * TensorRT 11.2 cannot compile the dynamic convex tail under a ranged H/W profile. Keeping
   * all six authenticated production shapes as point profiles in one engine preserves one
   * engine/context/enqueue while making every convex-tail dimension constant per profile.
   */
  inline constexpr std::array fixed_profile_shapes {
    coarse_shape_t {770u, 434u},
    coarse_shape_t {1022u, 434u},
    coarse_shape_t {1036u, 434u},
    coarse_shape_t {434u, 770u},
    coarse_shape_t {434u, 1022u},
    coarse_shape_t {434u, 1036u},
  };

  /** Initial Stage-2 live spatial authority is deliberately landscape-only.
   *
   * The fused operator supports all six point profiles, but the current vertical limiter cannot
   * authenticate a 2072-cell refined portrait column.  Keep that operator capability distinct
   * from the smaller set allowed to publish a live 2x parallax field.
   */
  inline constexpr bool live_geometry_coarse_shape_is_supported(
    const std::uint32_t width,
    const std::uint32_t height
  ) noexcept {
    return height == 434u &&
           (width == 770u || width == 1022u || width == 1036u);
  }

  inline constexpr bool live_geometry_field_shape_is_supported(
    const std::uint32_t width,
    const std::uint32_t height
  ) noexcept {
    return height == 868u &&
           (width == 1540u || width == 2044u || width == 2072u);
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

  inline constexpr std::optional<std::uint32_t> fixed_profile_index(
    const std::uint32_t width,
    const std::uint32_t height
  ) noexcept {
    for (std::uint32_t index = 0u; index < fixed_profile_shapes.size(); ++index) {
      if (fixed_profile_shapes[index] == coarse_shape_t {width, height}) {
        return index;
      }
    }
    return std::nullopt;
  }

  enum class engine_io_e : std::uint8_t {
    invalid,
    production_dav2,
    production_dav2_zipdepth_convex2x,
  };

  /** Pure named-I/O classification used before any TensorRT shape/address binding.
   *
   * Dtype and input/output-mode validation stays at the TensorRT boundary.  Exact tensor counts
   * prevent a partially exported or debug-output graph from being mistaken for production.
   */
  inline constexpr engine_io_e classify_named_io(
    const std::uint32_t tensor_count,
    const bool has_dav2_input,
    const bool has_guidance_input,
    const bool has_coarse_output,
    const bool has_refined_output
  ) noexcept {
    if (tensor_count == 2u && has_dav2_input && !has_guidance_input &&
        has_coarse_output && !has_refined_output) {
      return engine_io_e::production_dav2;
    }
    if (tensor_count == 4u && has_dav2_input && has_guidance_input &&
        has_coarse_output && has_refined_output) {
      return engine_io_e::production_dav2_zipdepth_convex2x;
    }
    return engine_io_e::invalid;
  }

}  // namespace models::prod_zipdepth_convex2x
