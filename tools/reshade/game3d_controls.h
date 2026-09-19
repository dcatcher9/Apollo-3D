// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cmath>

namespace reshade::api {
  struct effect_runtime;
}

namespace sunshine_game3d {
  inline constexpr float default_strength = 50.f;
  struct render_settings {
    float strength = default_strength;
    int depth_view = 0;
    bool enabled = true;
  };

  // Add-on-owned settings, independent of effects, presets and Performance Mode.
  // Each actual edit is persisted to this runtime's ReShade configuration.
  render_settings query_render_settings(reshade::api::effect_runtime *runtime);

  enum class automatic_phase { unavailable, waiting_for_depth, calibrating, ready, suspended, unsupported_resolution, renderer_unavailable };
  enum class automatic_scale_basis { unknown, camera_matrix, relative_depth };
  enum class automatic_scale_state { unavailable, pending, active, held };
  struct automatic_scale {
    automatic_scale_basis basis = automatic_scale_basis::unknown;
    float value = 0.f;
    bool active = false;
    float target_value = 0.f;
    double conversion_multiplier = 0.0;
    double conversion_offset = 0.0;
    bool has_projection_conversion = false;

    bool has_value() const {
      return basis != automatic_scale_basis::unknown && std::isfinite(value) && value > 0.f;
    }
    bool has_target() const {
      return active && has_value() && std::isfinite(target_value) && target_value > 0.f;
    }
    bool projection_conversion_valid() const {
      return basis == automatic_scale_basis::camera_matrix && has_projection_conversion &&
        std::isfinite(conversion_multiplier) && conversion_multiplier != 0.0 && std::isfinite(conversion_offset);
    }
    automatic_scale_state state() const {
      if (basis == automatic_scale_basis::unknown) return automatic_scale_state::unavailable;
      if (!has_value()) return automatic_scale_state::pending;
      return active ? automatic_scale_state::active : automatic_scale_state::held;
    }
  };

  inline automatic_scale retain_automatic_scale(automatic_scale previous, automatic_scale current) {
    // Only a value actually applied to a ready frame establishes a displayable
    // scale. Missing depth retains that last value as held, never as active.
    if (current.active && current.has_value()) return current;
    if (previous.has_value() && (current.basis == automatic_scale_basis::unknown || current.basis == previous.basis)) {
      previous.active = false;
      previous.target_value = 0.f;
      return previous;
    }
    // Camera distance and raw coordinates are different representations.
    // Never relabel held values.
    // Valid current projection conversion may be known before scene centering.
    current.value = current.target_value = 0.f;
    current.active = false;
    return current;
  }

  struct automatic_status {
    automatic_phase phase = automatic_phase::unavailable;
    bool can_recalibrate = false;
    automatic_scale scale;
  };

  // Exporter-owned policy state. Values contain no borrowed GPU objects.
  automatic_status query_automatic(reshade::api::effect_runtime *runtime);
  bool recalibrate_automatic(reshade::api::effect_runtime *runtime);

  void initialize();
  void shutdown();
  // False means a synchronous edit invalidated the runtime snapshot. Stop
  // drawing before accessing any earlier runtime state.
  bool draw(reshade::api::effect_runtime *runtime);
  // Optional depth-preview and recenter actions, drawn only inside the depth
  // owner's troubleshooting section. Read-only status stays in the main panel.
  // Does not request candidate enumeration.
  bool draw_diagnostics(reshade::api::effect_runtime *runtime);
}  // namespace sunshine_game3d
