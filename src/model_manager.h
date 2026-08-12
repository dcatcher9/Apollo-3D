#pragma once
#include <string>
#include <string_view>
#include <filesystem>

#include "config.h"
#include "generated/depth_coordinate_v2_contract.h"

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
