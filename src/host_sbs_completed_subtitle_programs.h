/** @file Shared pre-input setup for the completed Host SBS subtitle renderer. */
#pragma once

#ifdef _WIN32
  #include "host_sbs_shared_ui_warp_runtime.h"
  #include "video_depth_estimator.h"

  #include <exception>
  #include <filesystem>
  #include <string>
  #include <utility>

namespace models::host_sbs_completed_subtitle_programs {
  /** The live and Convert Video route: adaptive OCR work with the native palette decoder. */
  inline subtitle_direct_glyph_configuration_t canonical_configuration(const std::filesystem::path &shader_root) {
    subtitle_direct_glyph_configuration_t configuration;
    configuration.shader_root = shader_root.string();
    configuration.adaptive_context = true;
    configuration.line_palette = true;
    return configuration;
  }

  // The estimator accepts one pre-input configuration. Authenticate the matching
  // renderer first, and expose the pair only after the estimator accepts it.
  inline bool prepare(ID3D11Device *device, video_depth_estimator &estimator,
                      const subtitle_direct_glyph_configuration_t &configuration,
                      host_sbs_shared_ui_warp::programs_t &output, std::string &error) {
    if (configuration.shader_root.empty()) {
      error = "Selected rendering requires the shared subtitle warp";
      return false;
    }
    try {
      const std::filesystem::path shader_root(configuration.shader_root);
      host_sbs_shared_ui_warp::programs_t programs;
      if (!programs.load(device, shader_root, error) ||
          !programs.load_direct_glyph_completed(device, shader_root, error) ||
          !estimator.configure_subtitle_direct_glyph(configuration, error)) {
        return false;
      }
      output = std::move(programs);
      return true;
    } catch (const std::exception &exception) {
      error = exception.what();
      return false;
    }
  }
}  // namespace models::host_sbs_completed_subtitle_programs
#endif
