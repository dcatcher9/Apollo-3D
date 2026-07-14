#pragma once
#include <string>
#include <filesystem>

#include "config.h"

namespace models {
    // TensorRT tactic-selection target for the shipping landscape DA-V2 path. Both dimensions
    // are patch-aligned (14 px), and match the 16:9 production tensor selected from the default
    // short side. The recipe tag must change whenever the serialized-engine build contract does.
    inline constexpr int depth_engine_opt_width = 770;
    inline constexpr int depth_engine_opt_height = 434;
    inline constexpr char depth_engine_recipe[] = "trt-opt770x434-v1";

    /**
     * @brief Recipe-specific cached TensorRT engine filename.
     */
    std::string engine_filename(const config::depth_model_info& model);

    /**
     * @brief Ensures the given depth model is available locally.
     * Looks for the engine (engine_name, or <model_name>.engine if empty), then
     * <model_name>.onnx, in the assets directory. If neither exists, downloads the ONNX from
     * model_url and saves it as <model_name>.onnx. The name is decoupled from the URL so
     * different models (not just variants) can coexist, each with its own cached engine.
     *
     * @param assets_dir The base assets directory path
     * @param model_name Local file stem identifying the model (e.g. "depth_anything_v2_fp16").
     * @param model_url  URL to download the .onnx from if it isn't present locally.
     * @param engine_name Engine filename to look for (see engine_filename()).
     *                    Empty = "<model_name>.engine".
     * @return The path to the usable model file (.engine preferred over .onnx), or empty if failed.
     */
    std::filesystem::path ensure_model_available(const std::filesystem::path& assets_dir, const std::string& model_name, const std::string& model_url, const std::string& engine_name = "");
}
