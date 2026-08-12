#pragma once
#include <string>
#include <string_view>
#include <filesystem>

#include "config.h"

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

    // Production burned-in text detector. The official FP32 ONNX retains FP32 graph/boundary
    // types, with TF32 acceleration explicitly allowed by the builder recipe. A later FP16 path
    // must authenticate a separately converted graph instead of relabeling these source bytes.
    inline constexpr int ocr_engine_width = 960;
    inline constexpr int ocr_engine_height = 160;
    inline constexpr int ocr_engine_builder_level = 5;
    inline constexpr char ocr_model_name[] = "ppocrv6_tiny_det";
    inline constexpr char ocr_model_url[] =
        "https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/"
        "2ba1506c0380b8f0b03dd142459aac66d4421f6c/inference.onnx?download=true";
    inline constexpr char ocr_model_onnx_sha256[] =
        "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8";
    inline constexpr char ocr_engine_recipe[] =
        "trt-strong-fp32-tf32-fixed960x160-level5-v1";

    /**
     * @brief Recipe-specific cached TensorRT engine filename.
     */
    std::string engine_filename(const config::depth_model_info& model, std::string_view compatibility_tag = {});

    /** Recipe-specific cached PP-OCRv6 tiny detector engine filename. */
    std::string ocr_engine_filename(std::string_view compatibility_tag = {});

    /** Ensure the ONNX source exists locally, downloading it atomically when necessary. */
    std::filesystem::path ensure_onnx_available(
        const std::filesystem::path& assets_dir,
        const std::string& model_name,
        const std::string& model_url);

    /** Full SHA-256 of a file, as lowercase hexadecimal. Empty means the file could not be read. */
    std::string file_sha256_hex(const std::filesystem::path& path);
}
