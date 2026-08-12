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

    // Production burned-in text detector. The bundled ONNX is derived from the pinned official
    // PP-OCRv6 tiny source with NVIDIA ModelOpt AutoCast: the graph runs in FP16 while its D3D/CUDA
    // input and output boundaries remain FP32. The artifact path, both hashes, conversion recipe,
    // and TensorRT build recipe are authenticated by the generated DVC2 contract.
    inline constexpr int ocr_engine_width = 960;
    inline constexpr int ocr_engine_height = 160;
    inline constexpr int ocr_engine_builder_level = 5;
    inline constexpr char ocr_model_name[] = "ppocrv6_tiny_det_modelopt_fp16";
    inline constexpr char ocr_model_asset_path[] =
        "models/ppocrv6_tiny_det_modelopt045_mixed_fp16_fp32io.onnx";
    inline constexpr char ocr_model_artifact_onnx_sha256[] =
        "169a233ba0ff7cac27f8ec7dccb6a406e614b25b21fe6a5638c423bf2118bb44";
    inline constexpr char ocr_model_source_url[] =
        "https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/"
        "2ba1506c0380b8f0b03dd142459aac66d4421f6c/inference.onnx?download=true";
    inline constexpr char ocr_model_source_onnx_sha256[] =
        "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8";
    inline constexpr char ocr_model_conversion_tool[] = "nvidia-modelopt";
    inline constexpr char ocr_model_conversion_version[] = "0.45.0";
    inline constexpr char ocr_model_conversion_recipe[] =
        "nvidia-modelopt-autocast-fp16-keep-io-fp32-v1";
    inline constexpr char ocr_model_conversion_calibration_profile[] =
        "apollo-live8-bottom960x160-v1";
    inline constexpr char ocr_engine_recipe[] =
        "trt-strong-modelopt045-fp16-iofp32-tf32-fixed960x160-level5-v2";

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
