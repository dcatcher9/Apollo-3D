#pragma once
#include <string>
#include <filesystem>

namespace models {
    /**
     * @brief Ensures the Depth Anything V2 model is available locally.
     * Checks if the .engine or .onnx file exists in the assets directory.
     * If not, downloads the .onnx file from the internet.
     * 
     * @param assets_dir The base assets directory path
     * @return The path to the usable model file (.engine preferred over .onnx), or empty if failed.
     */
    std::filesystem::path ensure_model_available(const std::filesystem::path& assets_dir);
}
