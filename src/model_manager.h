#pragma once
#include <string>
#include <filesystem>

#include "config.h"

namespace models {
    /**
     * @brief Cached-engine filename for a depth model, encoding the build recipe.
     *
     * A cached <stem>.engine is only valid for the exact build recipe it was compiled with:
     * the input rank and which outputs were kept both change the engine's I/O, so a stale
     * engine from a different recipe would bind mismatched tensors. The recipe is folded into
     * the filename so a recipe change forces a rebuild (and variants coexist on disk). The
     * legacy DA-V2 recipe (rank-4, no confidence) yields the bare "<name>.engine", leaving
     * existing engines valid; DA-V3 (rank-5) becomes "<name>.r5.engine", "+confidence" adds a
     * "c" (e.g. "<name>.r5c.engine").
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
     * @param engine_name Recipe-specific engine filename to look for (see engine_filename()).
     *                    Empty = the legacy "<model_name>.engine".
     * @return The path to the usable model file (.engine preferred over .onnx), or empty if failed.
     */
    std::filesystem::path ensure_model_available(const std::filesystem::path& assets_dir, const std::string& model_name, const std::string& model_url, const std::string& engine_name = "");
}
