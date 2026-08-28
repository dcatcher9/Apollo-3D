#pragma once
#include <string>
#include <string_view>
#include <filesystem>

#include "config.h"
#include "generated/depth_coordinate_v2_contract.h"
#include "prod_zipdepth_convex2x.h"

namespace models {
    // TensorRT tactic-selection target for the shipping landscape DA-V2 path. Both dimensions
    // are patch-aligned (14 px), and match the 16:9 production tensor selected from the default
    // short side. The recipe tag must change whenever the serialized-engine build contract does.
    inline constexpr int depth_engine_opt_width = 770;
    inline constexpr int depth_engine_opt_height = 434;
    inline constexpr int depth_engine_builder_level = 5;
    // Upper bound of the dynamic-shape profile, and the cap applied when fitting the source
    // aspect. 1036 = 74 patches, chosen so the two ultrawide production cases reach the full
    // configured short side instead of dropping a patch row: 21:9 needs 1036x434 and 5K2K
    // 1022x434, both of which the previous 1008 bound forced down to a 420 short side.
    inline constexpr int depth_engine_max_dim = 1036;
    inline constexpr char depth_engine_recipe[] = "trt-opt770x434-max1036-level5-v3";
    // TensorRT 11.2 cannot compile the dynamic convex gather under the legacy ranged profile.
    // The fused graph instead owns the six authenticated HIGH public shapes as point profiles in
    // the order frozen by prod_zipdepth_convex2x::fixed_profile_shapes. Keep this cache recipe
    // distinct from both the legacy DAV2 plan and the retired two-input/two-output fused plan.
    inline constexpr bool fused_depth_profile_contracts_match = []() {
      if (prod_zipdepth_convex2x::fixed_profile_shapes.size() !=
            depth_coordinate_v2::model_calibrated_shapes.size() ||
          prod_zipdepth_convex2x::fixed_profile_shapes.size() !=
            depth_coordinate_v2::subtitle_ocr_live_field_shapes.size()) {
        return false;
      }
      for (std::size_t index = 0u;
           index < prod_zipdepth_convex2x::fixed_profile_shapes.size();
           ++index) {
        const auto high = prod_zipdepth_convex2x::fixed_profile_shapes[index];
        const auto coarse = depth_coordinate_v2::model_calibrated_shapes[index];
        const auto live = depth_coordinate_v2::subtitle_ocr_live_field_shapes[index];
        if (high.width != prod_zipdepth_convex2x::scale * coarse.width ||
            high.height != prod_zipdepth_convex2x::scale * coarse.height ||
            high.width != live[0] || high.height != live[1]) {
          return false;
        }
      }
      return true;
    }();
    static_assert(
      fused_depth_profile_contracts_match,
      "DAV2 calibration, fused TensorRT profiles, and high-field policy must stay identical"
    );
    static_assert(
      prod_zipdepth_convex2x::fixed_profile_shapes.front() ==
      prod_zipdepth_convex2x::high_shape_t {
        static_cast<std::uint32_t>(prod_zipdepth_convex2x::scale) *
          static_cast<std::uint32_t>(depth_engine_opt_width),
        static_cast<std::uint32_t>(prod_zipdepth_convex2x::scale) *
          static_cast<std::uint32_t>(depth_engine_opt_height),
      }
    );
    inline constexpr std::string_view depth_engine_recipe_for(
        const prod_zipdepth_convex2x::engine_io_e kind) noexcept {
      switch (kind) {
        case prod_zipdepth_convex2x::engine_io_e::production_dav2:
          return depth_engine_recipe;
        case prod_zipdepth_convex2x::engine_io_e::production_dav2_zipdepth_convex2x:
          return prod_zipdepth_convex2x::engine_recipe;
        default:
          return {};
      }
    }

    // Production burned-in text detector. The generated DVC2 contract owns its artifact,
    // provenance, boundary dimensions, and engine recipe; keep only the host builder level here.
    inline constexpr int ocr_engine_width =
        static_cast<int>(depth_coordinate_v2::subtitle_ocr_input_width);
    inline constexpr int ocr_engine_height =
        static_cast<int>(depth_coordinate_v2::subtitle_ocr_input_height);
    static_assert(depth_coordinate_v2::subtitle_ocr_output_width ==
                  depth_coordinate_v2::subtitle_ocr_input_width);
    static_assert(depth_coordinate_v2::subtitle_ocr_output_height ==
                  depth_coordinate_v2::subtitle_ocr_input_height);
    inline constexpr int ocr_engine_builder_level = 5;
    inline constexpr std::string_view ocr_model_name =
        depth_coordinate_v2::subtitle_ocr_model_name;
    inline constexpr std::string_view ocr_model_asset_path =
        depth_coordinate_v2::subtitle_ocr_asset_path;
    inline constexpr std::string_view ocr_model_artifact_onnx_sha256 =
        depth_coordinate_v2::subtitle_ocr_artifact_onnx_sha256;
    inline constexpr std::string_view ocr_engine_recipe =
        depth_coordinate_v2::subtitle_ocr_engine_recipe;

    /**
     * @brief Recipe-specific cached TensorRT engine filename.
     */
    std::string engine_filename(const config::depth_model_info& model, std::string_view compatibility_tag = {});

    /** Recipe-specific cached engine filename for an already classified depth-model I/O graph. */
    std::string engine_filename(
        const config::depth_model_info& model,
        prod_zipdepth_convex2x::engine_io_e kind,
        std::string_view compatibility_tag);

    /**
     * Bounded cached PP-OCRv6 engine filename. The compatibility tag must bind the complete
     * TensorRT runtime, GPU, and authenticated ONNX identity used to build the plan.
     */
    std::string ocr_engine_filename(std::string_view compatibility_tag = {});

    /** Ensure the ONNX source exists locally, downloading it atomically when necessary. */
    std::filesystem::path ensure_onnx_available(
        const std::filesystem::path& assets_dir,
        const std::string& model_name,
        const std::string& model_url);

    /** Full SHA-256 of a file, as lowercase hexadecimal. Empty means the file could not be read. */
    std::string file_sha256_hex(const std::filesystem::path& path);
}
