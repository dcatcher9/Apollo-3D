/** @file Shared pre-input setup for the completed Host SBS subtitle renderer. */
#pragma once

#ifdef _WIN32
  #include "host_sbs_current_outline_runtime_programs.h"
  #include "host_sbs_shared_ui_warp_runtime.h"
  #include "video_depth_estimator.h"

  #include <exception>
  #include <filesystem>
  #include <string>
  #include <utility>

namespace models::host_sbs_completed_subtitle_programs {
  inline subtitle_selected_matte_research_configuration_t canonical_configuration(
    const std::filesystem::path &shader_root, bool direct_glyph,
    bool adaptive_context = false
  ) {
    const auto outline_root = shader_root / "current_outline";
    subtitle_selected_matte_research_configuration_t configuration;
    configuration.current_outline = true;
    configuration.current_outline_shader_root = outline_root.string();
    configuration.selector_shader_root = current_outline_runtime::selector_root(outline_root).string();
    configuration.selector_source_closure_sha256 = current_outline_runtime::selector_closure;
    if (direct_glyph) {
      configuration.current_outline_direct_glyph_shader_root = shader_root.string();
      configuration.direct_glyph_adaptive_context = adaptive_context;
      configuration.direct_glyph_line_palette = adaptive_context;
    }
    return configuration;
  }

  // The estimator accepts one pre-input configuration. Authenticate the matching
  // renderer first, and expose the pair only after the estimator accepts it.
  inline bool prepare(ID3D11Device *device, video_depth_estimator &estimator,
                      const subtitle_selected_matte_research_configuration_t &configuration,
                      host_sbs_shared_ui_warp::programs_t &output, std::string &error) {
    if (!configuration.current_outline || configuration.current_outline_shader_root.empty()) {
      error = "Selected rendering requires the current-outline shared warp";
      return false;
    }
    try {
      const auto shader_root = std::filesystem::path(configuration.current_outline_shader_root).parent_path();
      host_sbs_shared_ui_warp::programs_t programs;
      const bool direct = !configuration.current_outline_direct_glyph_shader_root.empty();
      if (!programs.load(device, shader_root, error) ||
          ((direct || !configuration.current_outline_raw_matte_reduced_shader_root.empty()) &&
           !programs.load_raw_model_completed(device, shader_root, error, direct)) ||
          !estimator.configure_subtitle_selected_matte_research(configuration, error)) {
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
