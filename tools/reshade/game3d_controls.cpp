// SPDX-License-Identifier: GPL-3.0-only
#include "addon_lifetime.h"
#include "game3d_controls.h"
#include "game3d_controls_model.h"
#include "game3d_stereo_contract.h"

// COM declares MinGW's __uuidof support before ReShade's API templates.
#include <Windows.h>
#include <Unknwn.h>

#include <algorithm>
#include <atomic>
#include <imgui.h>
#include <limits>
#include <memory>
#include <mutex>
#include <reshade.hpp>
#include <unordered_map>

namespace sunshine_game3d {
  namespace api = reshade::api;

  namespace {
    std::recursive_mutex state_mutex;
    std::unordered_map<api::effect_runtime *, std::shared_ptr<settings_state>> runtimes;
    std::atomic_bool registered {false};

    struct config_backend {
      api::effect_runtime *runtime;
      template<class T> void read(const char *key, T &value) {
        // The installer migrates ReShade.ini. Secondary runtimes may use a
        // separate ReShade[index].ini; inherit only keys they do not override.
        if (!reshade::get_config_value(runtime, config_section, key, value))
          reshade::get_config_value(nullptr, config_section, key, value);
      }
      template<class T> void write(const char *key, T value) {
        // ReShade saves its configuration independently of shader presets and
        // their Auto Save setting. There is no effect reload or uniform write.
        reshade::set_config_value(runtime, config_section, key, value);
      }
    };

    std::shared_ptr<settings_state> get_state(api::effect_runtime *runtime) {
      auto &data = runtimes[runtime];
      if (!data) {
        data = std::make_shared<settings_state>();
        config_backend config {runtime};
        data->values = load_settings(config);
      }
      return data;
    }

    void destroy(api::effect_runtime *runtime) {
      std::lock_guard<std::recursive_mutex> lock(state_mutex);
      if (auto it = runtimes.find(runtime); it != runtimes.end()) {
        it->second->alive = false;
        runtimes.erase(it);
      }
    }

    float reserve_reset_button() {
      // ReShade is built with MSVC. Avoid ImVec2-returning ImGui calls across
      // MinGW's ABI; scalar returns and style references are safe.
      const auto &style = ImGui::GetStyle();
      const float width = 3.5f * ImGui::GetFontSize() + style.FramePadding.x * 2;
      ImGui::SetNextItemWidth(-(width + style.ItemSpacing.x));
      return width;
    }

    void strength_control(settings_state &settings, config_backend &config) {
      float value = settings.values.strength;
      ImGui::PushID("Strength");
      ImGui::TextUnformatted("3D strength");
      const float reset_width = reserve_reset_button();
      if (ImGui::SliderFloat("##value", &value, 0, 100, "%.0f%%", ImGuiSliderFlags_NoRoundToFormat))
        edit_strength(settings, value, config);
      ImGui::SetItemTooltip("Stereo separation. 0%% shows the same image to both eyes. The default is 50%%. Changes are saved automatically for this game.");
      if (settings.alive && settings.values.strength != default_strength) {
        ImGui::SameLine();
        if (ImGui::Button("Reset", ImVec2(reset_width, 0))) edit_strength(settings, default_strength, config);
        ImGui::SetItemTooltip("Restore 50%% strength");
      }
      ImGui::PopID();
    }

    void depth_view_control(settings_state &settings, config_backend &config) {
      const char *views[] {"Game image", "Stereo depth", "Normal depth"};
      const int value = settings.values.depth_view;
      ImGui::PushID("DepthView");
      ImGui::TextUnformatted("Depth view");
      const float reset_width = reserve_reset_button();
      if (ImGui::BeginCombo("##value", views[value])) {
        for (int i = 0; i < 3; ++i) {
          if (ImGui::Selectable(views[i], value == i)) edit_depth_view(settings, i, config);
          if (!settings.alive) break;
        }
        ImGui::EndCombo();
      }
      ImGui::SetItemTooltip("Normal depth checks scene capture. Stereo depth shows the field used for the 3D effect. Choose Game image to return to normal viewing.");
      if (settings.alive && settings.values.depth_view != 0) {
        ImGui::SameLine();
        if (ImGui::Button("Reset", ImVec2(reset_width, 0))) edit_depth_view(settings, 0, config);
        ImGui::SetItemTooltip("Restore Game image");
      }
      ImGui::PopID();
    }

    void camera_hint(const automatic_status &status, bool enabled,
        const frame_generation_mode &fg) {
      if (status.scale.basis != automatic_scale_basis::relative_depth || !enabled) return;
      if (fg.known && fg.enabled)
        ImGui::TextWrapped("Camera data unavailable with this game's Frame Generation.");
      else
        ImGui::TextWrapped("Camera data unavailable. Try Frame Generation 2x if supported.");
      ImGui::SetItemTooltip("Some games provide Streamline camera data only with Frame Generation. This is not guaranteed. Without it, relative depth assumes an infinite far plane. Sunshine does not change game settings.");
    }

    void calibration_status(const automatic_status &status, float strength) {
      const auto &scale = status.scale;
      const auto scale_state = scale.state();
      const bool camera_scale = scale.basis == automatic_scale_basis::camera_matrix;
      const bool held = scale_state == automatic_scale_state::held;
      const bool has_zero = scale.has_zero && std::isfinite(scale.zero_inverse) && scale.zero_inverse >= 0.f;
      const double zero_plane = camera_scale ? scale.zero_inverse > 0.f ? 1.0 / scale.zero_inverse :
        std::numeric_limits<double>::infinity() : scale.zero_inverse;
      const double target_zero_plane = camera_scale ? scale.target_zero_inverse > 0. ? 1.0 / scale.target_zero_inverse :
        std::numeric_limits<double>::infinity() : scale.target_zero_inverse;
      ImGui::TextWrapped("Depth conversion: %s", camera_scale ? "camera projection matrix" :
        scale.basis == automatic_scale_basis::relative_depth ? "relative depth (assumed infinite far plane)" : "waiting for depth data");
      ImGui::TextWrapped("Depth q: %s", camera_scale ? "inverse game units (1/Z); larger is nearer" :
        scale.basis == automatic_scale_basis::relative_depth ? "relative inverse depth; larger is nearer" : "unavailable");
      if (held) ImGui::TextDisabled("Last applied values; tracking paused");
      else if (!scale.has_value()) ImGui::TextDisabled("Screen plane: %s", scale_state == automatic_scale_state::pending ? "centering" : "unavailable");
      if (ImGui::BeginTable("ScaleDetails", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn("Parameter", ImGuiTableColumnFlags_WidthStretch, 1.8f);
        ImGui::TableSetupColumn("Current", ImGuiTableColumnFlags_WidthStretch, 1.f);
        ImGui::TableSetupColumn("Target", ImGuiTableColumnFlags_WidthStretch, 1.f);
        ImGui::TableHeadersRow();
        const auto row = [](const char *label, const char *tooltip, bool has_value, double value, bool has_target = false, double target = 0) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextWrapped("%s", label);
          ImGui::SetItemTooltip("%s", tooltip);
          ImGui::TableNextColumn();
          if (has_value && std::isinf(value)) ImGui::TextUnformatted("Infinity");
          else if (has_value) ImGui::Text("%.6g", value); else ImGui::TextDisabled("--");
          ImGui::TableNextColumn();
          if (has_target && std::isinf(target)) ImGui::TextUnformatted("Infinity");
          else if (has_target) ImGui::Text("%.6g", target); else ImGui::TextDisabled("--");
        };
        const bool has_depth = scale.depth_statistics_valid();
        row("Nearest reference Q", "Largest converted inverse depth in the full active depth rectangle of the accepted measurement. Target stereo gain is L/Q. This is a scene measurement, not the camera's near clipping plane; even a single nearest pixel contributes.",
          has_depth, scale.reference_inverse);
        row("Farthest (q min)", "Smallest converted inverse depth in the full active depth rectangle of the same accepted measurement. Zero denotes infinite distance in this depth model. This is not the camera's far clipping plane.",
          has_depth, scale.minimum_inverse);
        row("Zero plane q0", "Inverse depth where objects appear at screen depth. Trial placement balances the farthest depth with a depth-contrast-weighted foreground reference. This is not subject recognition. Current zero follows gradually with a parallax-based speed limit; it can remain outside the new range during a transition. Changing zero does not redefine gain.",
          has_zero, scale.zero_inverse, scale.has_zero_target(), scale.target_zero_inverse);
        row("Stereo gain K", "Target is L/Q, independently of the zero plane and strength slider. Current gain follows it gradually in either direction. The renderer limits each pixel's final parallax during adaptation. A positive flat scene can update zero while holding gain. K is also called H in relative-depth diagnostics; 1/K is not the measured reference Q.",
          scale.has_value(), scale.value, scale.has_target(), scale.target_value);
        row("UI midpoint q", "UI plane follows the independently smoothed inverse-depth midpoint (q min + q max) / 2. This is separate from the scene zero plane and is not a physical-distance midpoint. Current scene gain, zero, strength and stereo blend map this depth to displayed parallax.",
          scale.has_ui_midpoint, scale.ui_midpoint_inverse, scale.has_ui_midpoint_target(), scale.target_ui_midpoint_inverse);
        row("Normalization L", "Full-strength normalization captured for this output shape and parallax limit. Target K=L/Q; L is dimensionless and does not recover a physical camera baseline.",
          scale.has_value() && std::isfinite(scale.normalization) && scale.normalization > 0., scale.normalization);
        if (camera_scale)
          row("Zero-plane distance", "1/q0 in game units, not necessarily meters. The relative-depth path cannot recover this distance.",
            has_zero, zero_plane, scale.has_zero_target(), target_zero_plane);
        row("Conversion scale", "Calculated from the projection matrix and raw-depth packing: inverse distance = scale * raw depth + offset. This is not estimated from scene content.", scale.projection_conversion_valid(), scale.conversion_multiplier);
        row("Conversion offset", "The offset in the camera's affine inverse-depth conversion. It has no scene target; the current validated matrix determines it.", scale.projection_conversion_valid(), scale.conversion_offset);
        ImGui::EndTable();
      }
      if (scale.depth_statistics_valid() && scale.depth_pixel_count)
        ImGui::TextDisabled("Full depth: %llu pixels, %u x %u tiles",
          static_cast<unsigned long long>(scale.depth_pixel_count), scale.depth_tiles_x, scale.depth_tiles_y);
      const float bounded_strength = std::isfinite(strength) ? std::clamp(strength, 0.f, 100.f) : 0.f;
      ImGui::TextWrapped("Per-eye parallax limit: %.3g%% of image width at current strength",
        double(default_disparity_limit_uv) * bounded_strength);
      ImGui::SetItemTooltip("Maximum horizontal shift per eye in the source image. Re-entry and reused-frame strength protection can reduce it further. This limits rendered parallax, not raw depth values.");
      if (scale.gain_below_target && scale.has_target()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.75f, 0.25f, 1.0f));
        ImGui::TextWrapped("Stereo gain is below its current target.");
        ImGui::PopStyleColor();
      } else {
        ImGui::TextWrapped("Trial zero plane: between the farthest depth and a representative foreground depth. Gain remains independent.");
      }
    }
  }  // namespace

  void initialize() {
    if (registered.exchange(true)) return;
    reshade::register_event<reshade::addon_event::destroy_effect_runtime>(sunshine_addon_lifetime::guarded<destroy>);
  }

  void shutdown() {
    if (registered.exchange(false))
      reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(sunshine_addon_lifetime::guarded<destroy>);
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    for (const auto &entry : runtimes) entry.second->alive = false;
    runtimes.clear();
  }

  render_settings query_render_settings(api::effect_runtime *runtime) {
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!runtime || !registered) return {default_strength, 0, false};
    return get_state(runtime)->values;
  }

  bool draw(api::effect_runtime *runtime) {
    if (!runtime) return false;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return true;
    const auto data = get_state(runtime);
    config_backend config {runtime};
    const auto automatic = query_automatic(runtime);
    ImGui::PushID("SunshineGame3DControls");
    const auto finish = [] { ImGui::Spacing(); ImGui::PopID(); };
    ImGui::TextUnformatted("Game 3D");
    ImGui::TextWrapped("%s", output_status_text(automatic, data->values));
    bool enabled = data->values.enabled;
    if (ImGui::Checkbox("Enable Game 3D", &enabled)) edit_enabled(*data, enabled, config);
    ImGui::SetItemTooltip("Enable native SBS rendering and export. This setting is independent of ReShade effects and presets.");
    if (!data->alive) { finish(); return false; }
    ImGui::Spacing();

    const auto source_alpha = query_source_alpha_ui(runtime);
    const auto &fg = source_alpha.fg;
    if (data->values.enabled && fg.known && fg.enabled) {
      const char *automatic_mode = fg.automatic ? " (Auto)" : "";
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.75f, 0.25f, 1.0f));
      if (fg.generated_frames >= 2) {
        const auto multiplier = static_cast<unsigned long long>(fg.generated_frames) + 1;
        ImGui::TextWrapped("Frame Generation %llux%s: use Off or 2x to reduce artifacts and stutter.", multiplier, automatic_mode);
      } else if (fg.generated_frames == 1) {
        ImGui::TextWrapped("Frame Generation 2x%s: turn off if artifacts or stutter appear.", automatic_mode);
      } else {
        ImGui::TextWrapped("Frame Generation enabled%s: turn off if artifacts or stutter appear.", automatic_mode);
      }
      ImGui::SetItemTooltip("Change Frame Generation in the game's settings. DLSS Super Resolution can stay on. This is the game's requested setting, not a measurement of generated frames. Sunshine does not change it automatically.");
      ImGui::PopStyleColor();
    }
    strength_control(*data, config);
    if (!data->alive) { finish(); return false; }
    bool source_alpha_ui = data->values.source_alpha_ui;
    if (ImGui::Checkbox("Keep UI on a separate plane (source alpha)", &source_alpha_ui))
      edit_source_alpha_ui(*data, source_alpha_ui, config);
    ImGui::SetItemTooltip("Enable only when this game's real-frame alpha represents UI. White and gray keep composited UI on one flat plane at the smoothed inverse-depth midpoint, independently of the scene zero target. Its displayed position follows the current scene mapping and 3D strength. Black keeps scene depth. All white makes the entire frame flat at the UI plane. Frame Generation reuses the latest completed real-input alpha; current RGB stays current. Fast-changing UI may briefly lag. If that input is unavailable, UI protection waits instead of using generated output alpha.");
    if (!data->alive) { finish(); return false; }
    if (source_alpha.blocked_by_fg()) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.75f, 0.25f, 1.0f));
      if (source_alpha.input_state == source_alpha_input_state::state_conflict)
        ImGui::TextWrapped("UI protection: real-frame alpha capture is blocked by a resource-state conflict. Use Dump 3D to record the details.");
      else if (source_alpha.input_state == source_alpha_input_state::rejected)
        ImGui::TextWrapped("UI protection: real-frame input was received, but alpha capture was rejected. Use Dump 3D to record the reason.");
      else if (source_alpha.input_state == source_alpha_input_state::seen)
        ImGui::TextWrapped("UI protection: real-frame input received; waiting for usable alpha.");
      else
        ImGui::TextWrapped("UI protection: waiting for real-frame alpha input. This FG path must expose its input color.");
      ImGui::PopStyleColor();
    } else if (source_alpha.effective()) {
      ImGui::TextWrapped("UI protection: %s", source_alpha.retained_alpha_ready ?
        "using latest real-frame alpha (reused during FG)" : "using current real-frame alpha");
    }
    camera_hint(automatic, data->values.enabled, fg);
    if (data->values.depth_view != 0) {
      ImGui::TextWrapped("Depth preview is selected.");
      if (ImGui::Button("Return to game image")) edit_depth_view(*data, 0, config);
      if (!data->alive) { finish(); return false; }
    }
    ImGui::Spacing();
    calibration_status(automatic, data->values.strength);
    finish();
    return data->alive;
  }

  bool draw_diagnostics(api::effect_runtime *runtime) {
    if (!runtime) return false;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return true;
    const auto data = get_state(runtime);
    config_backend config {runtime};
    ImGui::PushID("SunshineGame3DDiagnostics");
    depth_view_control(*data, config);
    ImGui::PopID();
    return data->alive;
  }

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  // Scalar-only adapters retain the existing test ABI while exercising native
  // config ownership. No test path reflects, modifies or saves shader presets.
  extern "C" __declspec(dllexport) BOOL SunshineGame3DTestQuery(api::effect_runtime *runtime, unsigned index, unsigned *flags, float *value) {
    if (!runtime || index >= unsigned(control::count) || !flags || !value) return FALSE;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return FALSE;
    const auto data = get_state(runtime);
    *flags = 3u; // Both native controls are always available and editable.
    *value = index == unsigned(control::strength) ? data->values.strength : float(data->values.depth_view);
    return TRUE;
  }

  extern "C" __declspec(dllexport) BOOL SunshineGame3DTestEditFloat(api::effect_runtime *runtime, unsigned index, float value) {
    if (!runtime || index != unsigned(control::strength)) return FALSE;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return FALSE;
    const auto data = get_state(runtime);
    config_backend config {runtime};
    return edit_strength(*data, value, config) ? TRUE : FALSE;
  }

  extern "C" __declspec(dllexport) BOOL SunshineGame3DTestEditInt(api::effect_runtime *runtime, unsigned index, int value) {
    if (!runtime || index != unsigned(control::depth_view)) return FALSE;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return FALSE;
    const auto data = get_state(runtime);
    config_backend config {runtime};
    return edit_depth_view(*data, value, config) ? TRUE : FALSE;
  }

  extern "C" __declspec(dllexport) BOOL SunshineGame3DTestSetEnabled(api::effect_runtime *runtime, BOOL enabled) {
    if (!runtime) return FALSE;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return FALSE;
    const auto data = get_state(runtime);
    config_backend config {runtime};
    return edit_enabled(*data, enabled != FALSE, config) ? TRUE : FALSE;
  }

  extern "C" __declspec(dllexport) BOOL SunshineGame3DTestSave(api::effect_runtime *runtime) {
    // Compatibility observation only: every accepted native edit saves itself.
    return runtime && registered ? TRUE : FALSE;
  }

  extern "C" __declspec(dllexport) BOOL SunshineGame3DTestQueryAutomatic(api::effect_runtime *runtime, unsigned *flags) {
    if (!runtime || !flags) return FALSE;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return FALSE;
    const auto settings = get_state(runtime)->values;
    const auto automatic = query_automatic(runtime);
    *flags = (settings.enabled ? 1u : 0u) |
      (settings.enabled && automatic.phase == automatic_phase::ready ? 4u : 0u) |
      (settings.enabled && automatic.can_recalibrate ? 8u : 0u);
    return TRUE;
  }

  extern "C" __declspec(dllexport) BOOL SunshineGame3DTestQueryScale(api::effect_runtime *runtime,
      unsigned *basis, unsigned *state, float *value) {
    if (!runtime || !basis || !state || !value || !registered) return FALSE;
    const auto scale = query_automatic(runtime).scale;
    *basis = unsigned(scale.basis);
    *state = unsigned(scale.state());
    *value = scale.has_value() ? scale.value : 0.f;
    return TRUE;
  }

  extern "C" __declspec(dllexport) BOOL SunshineGame3DTestRecalibrate(api::effect_runtime *runtime) {
    if (!runtime) return FALSE;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered || !get_state(runtime)->values.enabled || !query_automatic(runtime).can_recalibrate) return FALSE;
    return recalibrate_automatic(runtime) ? TRUE : FALSE;
  }
#endif
}  // namespace sunshine_game3d
