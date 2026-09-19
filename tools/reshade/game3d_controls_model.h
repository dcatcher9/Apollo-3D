// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_controls.h"

#include <cmath>

namespace sunshine_game3d {
  inline constexpr const char *config_section = "SUNSHINE_GAME3D";
  enum class control : unsigned { strength, depth_view, count };

  struct settings_state {
    render_settings values;
    bool alive = true;
  };

  template<class Backend>
  render_settings load_settings(Backend &config) {
    render_settings result;
    config.read("Strength", result.strength);
    config.read("DepthView", result.depth_view);
    config.read("Enabled", result.enabled);
    if (!std::isfinite(result.strength)) result.strength = default_strength;
    if (result.depth_view < 0 || result.depth_view > 2) result.depth_view = 0;
    // Preserve exact finite saved strength, including old off-range values.
    // Rendering applies its 0..100 limit; opening the UI never rewrites config.
    return result;
  }

  template<class Backend>
  bool edit_strength(settings_state &settings, float value, Backend &config) {
    if (!settings.alive || !std::isfinite(value) || value == settings.values.strength) return false;
    config.write("Strength", value);
    if (!settings.alive) return false;
    settings.values.strength = value;
    return true;
  }

  template<class Backend>
  bool edit_depth_view(settings_state &settings, int value, Backend &config) {
    if (!settings.alive || value < 0 || value > 2 || value == settings.values.depth_view) return false;
    config.write("DepthView", value);
    if (!settings.alive) return false;
    settings.values.depth_view = value;
    return true;
  }

  template<class Backend>
  bool edit_enabled(settings_state &settings, bool value, Backend &config) {
    if (!settings.alive || value == settings.values.enabled) return false;
    config.write("Enabled", value);
    if (!settings.alive) return false;
    settings.values.enabled = value;
    return true;
  }

  inline const char *output_status_text(const automatic_status &status, const render_settings &settings) {
    if (!settings.enabled) return "2D: Game 3D is disabled";
    switch (status.phase) {
      case automatic_phase::unavailable: return "2D: automatic depth unavailable";
      case automatic_phase::waiting_for_depth: return "2D: waiting for scene depth";
      case automatic_phase::calibrating: return "2D: setting the screen plane";
      case automatic_phase::suspended: return "2D: automatic depth paused";
      case automatic_phase::unsupported_resolution: return "2D: source resolution exceeds 3840 x 3840";
      case automatic_phase::renderer_unavailable: return "2D: Game 3D renderer unavailable for this display mode";
      case automatic_phase::ready: break;
    }
    // Normal depth remains visible at zero strength. Stereo depth follows the
    // same mono gate as the game image; match the renderer's ordering here.
    if (settings.depth_view == 2) return "Depth preview: normal depth";
    if (!std::isfinite(settings.strength) || settings.strength <= 0.f) return "2D: 3D strength is off";
    if (settings.depth_view == 1) return "Depth preview: stereo depth";
    return "3D active";
  }
}  // namespace sunshine_game3d
