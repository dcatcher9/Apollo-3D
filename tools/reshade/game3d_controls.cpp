// SPDX-License-Identifier: GPL-3.0-only
#include "addon_lifetime.h"
#include "game3d_controls.h"
#include "game3d_controls_model.h"
#include "streamline_camera_probe.h"

// COM declares MinGW's __uuidof support before ReShade's API templates.
#include <Windows.h>
#include <Unknwn.h>

#include <atomic>
#include <imgui.h>
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
        const sunshine_streamline::frame_generation_snapshot &fg,
        sunshine_streamline::frame_generation_query_status fg_status) {
      if (status.scale.basis != automatic_scale_basis::relative_depth || !enabled) return;
      using fg_query = sunshine_streamline::frame_generation_query_status;
      if (fg_status == fg_query::observed && fg.enabled)
        ImGui::TextWrapped("Camera data unavailable with this game's Frame Generation.");
      else if (fg_status == fg_query::unavailable || (fg_status == fg_query::observed && !fg.enabled))
        ImGui::TextWrapped("Camera data unavailable. Try Frame Generation 2x if supported.");
      else
        ImGui::TextWrapped("Camera data unavailable for this depth source.");
      ImGui::SetItemTooltip("Some games provide Streamline camera data only with Frame Generation. This is not guaranteed. Without it, relative depth assumes an infinite far plane. Sunshine does not change game settings.");
    }

    void calibration_status(const automatic_status &status) {
      const auto &scale = status.scale;
      const auto scale_state = scale.state();
      const bool camera_scale = scale.basis == automatic_scale_basis::camera_matrix;
      const bool held = scale_state == automatic_scale_state::held;
      ImGui::TextWrapped("Depth conversion: %s", camera_scale ? "camera projection matrix" :
        scale.basis == automatic_scale_basis::relative_depth ? "relative depth (assumed infinite far plane)" : "waiting for depth data");
      ImGui::TextWrapped("Zero-plane units: %s", camera_scale ? "game units, not necessarily meters" :
        scale.basis == automatic_scale_basis::relative_depth ? "relative raw depth, not distance" : "unavailable");
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
          if (has_value) ImGui::Text("%.6g", value); else ImGui::TextDisabled("--");
          ImGui::TableNextColumn();
          if (has_target) ImGui::Text("%.6g", target); else ImGui::TextDisabled("--");
        };
        row("Conversion scale", "Calculated from the projection matrix and raw-depth packing: inverse distance = scale * raw depth + offset. This is not estimated from scene content.", scale.projection_conversion_valid(), scale.conversion_multiplier);
        row("Conversion offset", "The offset in the camera's affine inverse-depth conversion. It has no scene target; the current validated matrix determines it.", scale.projection_conversion_valid(), scale.conversion_offset);
        row("Zero plane", "Objects here appear at screen depth. Current is smoothed; Target follows the center of the scene. See the units above.",
          scale.has_value(), camera_scale ? scale.value : scale.has_value() ? 1.0 / scale.value : 0,
          scale.has_target(), camera_scale ? scale.target_value : scale.has_target() ? 1.0 / scale.target_value : 0);
        row("Stereo normalization", "Derived from the screen plane: K = 1/q0 with matrix depth, H = 1/t0 with relative depth. 3D strength is a separate user multiplier afterward.",
          scale.has_value(), scale.value, scale.has_target(), scale.target_value);
        ImGui::EndTable();
      }
      ImGui::TextWrapped("Zero-plane tracking also changes stereo normalization and apparent strength.");
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

    sunshine_streamline::frame_generation_snapshot fg;
    auto fg_status = sunshine_streamline::frame_generation_query_status::unavailable;
    if (data->values.enabled && sunshine_streamline::query_frame_generation(UINT32_MAX, fg, &fg_status) && fg.enabled) {
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
    camera_hint(automatic, data->values.enabled, fg, fg_status);
    if (data->values.depth_view != 0) {
      ImGui::TextWrapped("Depth preview is selected.");
      if (ImGui::Button("Return to game image")) edit_depth_view(*data, 0, config);
      if (!data->alive) { finish(); return false; }
    }
    ImGui::Spacing();
    calibration_status(automatic);
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
    if (data->alive) {
      const auto status = query_automatic(runtime);
      ImGui::BeginDisabled(!data->values.enabled || !status.can_recalibrate);
      if (ImGui::Button("Recenter screen plane")) recalibrate_automatic(runtime);
      ImGui::SetItemTooltip("Set the screen plane from fresh captures of the current view, then continue tracking smoothly. This also changes stereo normalization and apparent strength. It does not reset your 3D strength setting.");
      ImGui::EndDisabled();
    }
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
