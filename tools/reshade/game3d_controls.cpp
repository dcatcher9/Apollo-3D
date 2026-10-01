// SPDX-License-Identifier: GPL-3.0-only
#include "addon_lifetime.h"
#include "game3d_controls.h"
#include "game3d_controls_model.h"
#include "game3d_stereo_contract.h"
#include "game3d_ui_adaptive.h"
#include "streamline_buffer_contract.h"

// COM declares MinGW's __uuidof support before ReShade's API templates.
#include <Windows.h>
#include <Unknwn.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <imgui.h>
#include <limits>
#include <memory>
#include <mutex>
#include <reshade.hpp>
#include "async_log.h" // After imgui.h and reshade.hpp, which emits the ImGui wrappers.
#include "game3d_ui_input_provider.h"
#include <unordered_map>

namespace sunshine_game3d {
  namespace api = reshade::api;

  namespace {
    std::recursive_mutex state_mutex;
    std::unordered_map<api::effect_runtime *, std::shared_ptr<settings_state>> runtimes;
    std::atomic_bool registered {false};
    source_alpha_mode process_alpha_mode = source_alpha_mode::automatic;
    bool alpha_mode_loaded = false;

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    float control_positions[5][2] {};
#endif

    void record_control_position(unsigned index) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      control_positions[index][0] = ImGui::GetCursorPosX();
      control_positions[index][1] = ImGui::GetCursorPosY();
#else
      (void)index;
#endif
    }

    bool begin_controls(const char *id) {
      if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) return false;
      ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 7.f * ImGui::GetFontSize());
      ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthStretch);
      return true;
    }

    void control_row(const char *label) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(label);
      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-1.f);
    }

    // Whether a channel carries UI coverage is a property of the game, so trust
    // outlives the session in this game's ReShade.ini (docs/reshade-sbs.md).
    constexpr const char *trusted_alpha_key = "TrustedUIAlpha";

    const std::shared_ptr<alpha_auto_policy> &alpha_session() {
      // Manual On/Off is shared per game; live Auto qualification is per runtime.
      static const auto session = [] {
        auto policy = std::make_shared<alpha_auto_policy>();
        unsigned int remembered = 0;
        if (reshade::get_config_value(nullptr, config_section, trusted_alpha_key, remembered) && (remembered & 15u)) {
          policy->restore_trusted_alpha(remembered);
          char text[160];
          std::snprintf(text, sizeof(text),
            "Sunshine UI protection: restored alpha trust 0x%x from an earlier session of this game", remembered & 15u);
          sunshine_log::message(reshade::log::level::info, text);
        }
        policy->on_trust_change([](std::uint32_t trusted) {
          reshade::set_config_value(nullptr, config_section, trusted_alpha_key, static_cast<unsigned int>(trusted));
          char text[160];
          std::snprintf(text, sizeof(text),
            "Sunshine UI protection: alpha trust is now 0x%x; remembered for later sessions of this game", trusted);
          sunshine_log::message(reshade::log::level::info, text);
        });
        return policy;
      }();
      return session;
    }

    struct config_backend {
      api::effect_runtime *runtime;
      template<class T> void read(const char *key, T &value) {
        // UI protection is one game-process choice, including games with more
        // than one runtime. Always reload it from the same per-game file.
        if (std::strcmp(key, "SourceAlphaUIMode") == 0) {
          reshade::get_config_value(nullptr, config_section, key, value);
          return;
        }
        // The installer migrates ReShade.ini. Secondary runtimes may use a
        // separate ReShade[index].ini; inherit only keys they do not override.
        if (!reshade::get_config_value(runtime, config_section, key, value))
          reshade::get_config_value(nullptr, config_section, key, value);
      }
      template<class T> void write(const char *key, T value) {
        // ReShade saves its configuration independently of shader presets and
        // their Auto Save setting. There is no effect reload or uniform write.
        reshade::set_config_value(std::strcmp(key, "SourceAlphaUIMode") == 0 ? nullptr : runtime,
          config_section, key, value);
      }
    };

    std::shared_ptr<settings_state> get_state(api::effect_runtime *runtime) {
      auto &data = runtimes[runtime];
      if (!data) {
        data = std::make_shared<settings_state>();
        config_backend config {runtime};
        data->values = load_settings(config);
        data->alpha_session = alpha_session();
        if (!alpha_mode_loaded) {
          process_alpha_mode = data->values.ui_protection;
          if (process_alpha_mode != source_alpha_mode::automatic)
            data->alpha_session->set_manual(process_alpha_mode == source_alpha_mode::on);
          alpha_mode_loaded = true;
        }
      }
      data->values.ui_protection = process_alpha_mode;
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
      control_row("3D strength");
      const float reset_width = reserve_reset_button();
      record_control_position(1);
      if (ImGui::SliderFloat("##value", &value, 0, 100, "%.0f%%", ImGuiSliderFlags_NoRoundToFormat))
        edit_strength(settings, value, config);
      ImGui::SetItemTooltip("Stereo separation. 0%% shows the same image to both eyes. The default is 50%%. Changes are saved automatically for this game.");
      if (settings.alive) {
        ImGui::SameLine();
        ImGui::BeginDisabled(settings.values.strength == default_strength);
        if (ImGui::Button("Reset", ImVec2(reset_width, 0))) edit_strength(settings, default_strength, config);
        ImGui::SetItemTooltip("Restore 50%% strength");
        ImGui::EndDisabled();
      }
      ImGui::PopID();
    }

    void depth_view_control(settings_state &settings, config_backend &config) {
      const char *views[] {"Game image", "Stereo depth", "Normal depth"};
      const int value = settings.values.depth_view;
      ImGui::PushID("DepthView");
      control_row("Depth view");
      const float reset_width = reserve_reset_button();
      record_control_position(3);
      if (ImGui::BeginCombo("##value", views[value])) {
        for (int i = 0; i < 3; ++i) {
          if (ImGui::Selectable(views[i], value == i)) edit_depth_view(settings, i, config);
          if (!settings.alive) break;
        }
        ImGui::EndCombo();
      }
      ImGui::SetItemTooltip("Normal depth checks scene capture. Stereo depth shows the field used for the 3D effect. Choose Game image to return to normal viewing.");
      if (settings.alive) {
        ImGui::SameLine();
        ImGui::BeginDisabled(settings.values.depth_view == 0);
        if (ImGui::Button("Reset", ImVec2(reset_width, 0))) edit_depth_view(settings, 0, config);
        ImGui::SetItemTooltip("Restore Game image");
        ImGui::EndDisabled();
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
      const bool distance_scale = camera_scale || scale.basis == automatic_scale_basis::linear_distance;
      const bool held = scale_state == automatic_scale_state::held;
      const bool has_zero = scale.has_zero && std::isfinite(scale.zero_inverse) && scale.zero_inverse >= 0.f;
      const double zero_plane = distance_scale ? scale.zero_inverse > 0.f ? 1.0 / scale.zero_inverse :
        std::numeric_limits<double>::infinity() : scale.zero_inverse;
      const double target_zero_plane = distance_scale ? scale.target_zero_inverse > 0. ? 1.0 / scale.target_zero_inverse :
        std::numeric_limits<double>::infinity() : scale.target_zero_inverse;
      ImGui::TextWrapped("Depth conversion: %s", camera_scale ? "camera projection matrix" :
        scale.basis == automatic_scale_basis::linear_distance ? "linear game-distance (1/Z)" :
        scale.basis == automatic_scale_basis::relative_depth ? "relative depth (assumed infinite far plane)" : "waiting for depth data");
      ImGui::TextWrapped("Depth q: %s", distance_scale ? "inverse game units (1/Z); larger is nearer" :
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
        row("Reference midpoint q", "Independently smoothed inverse-depth midpoint (q min + q max) / 2, retained for comparison with the earlier UI placement. Adaptive UI chooses a fraction of the display parallax limit instead of this depth.",
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
      ImGui::TextWrapped("UI plane: adaptive, with limited separation");
      ImGui::SetItemTooltip("Prefers screen depth. Inside the central 75%% of image width and height, moves nearer when foreground conflicts cover more than %u%% of UI or %.1f%% of the central image. Per-eye movement is limited to 0.35%% of image width and 50%% of the scene front limit. Retreat requires both overlap measures to clear and is delayed and smoothed. The system cursor follows UI; ReShade controls stay at screen depth. The limits may leave overlap.",
        static_cast<unsigned>(ui_adaptive::entry_conflict_percent), double(ui_adaptive::entry_area_per_mille) / 10.);
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
    alpha_session();
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
    const auto data = get_state(runtime);
    auto result = data->values;
    result.source_alpha_ui = result.ui_protection == source_alpha_mode::on;
    result.source_alpha_monitoring = false;
    result.source_alpha_probe_interval_ms = 0;
    return result;
  }

  alpha_auto_policy &source_alpha_startup_policy() { return *alpha_session(); }

  bool draw(api::effect_runtime *runtime) {
    if (!runtime) return false;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return true;
    const auto data = get_state(runtime);
    config_backend config {runtime};
    ImGui::PushID("SunshineGame3DControls");
    ImGui::TextUnformatted("Game 3D");
    // Keep every primary control ahead of live text. Neither a capture gap,
    // wrapped warning nor the active diagnostics tab may move these targets.
    bool enabled = data->values.enabled;
    record_control_position(0);
    if (ImGui::Checkbox("Enable Game 3D", &enabled)) edit_enabled(*data, enabled, config);
    ImGui::SetItemTooltip("Enable native SBS rendering and export. This setting is independent of ReShade effects and presets.");
    if (data->alive && begin_controls("MainControls")) {
      strength_control(*data, config);
      if (data->alive) {
        control_row("UI protection");
        // Display order is On / Off / Auto; persisted enum values stay stable.
        const source_alpha_mode modes[]{source_alpha_mode::on, source_alpha_mode::off, source_alpha_mode::automatic};
        int choice = data->values.ui_protection == source_alpha_mode::on ? 0 : data->values.ui_protection == source_alpha_mode::off ? 1 : 2;
        record_control_position(2);
        if (ImGui::Combo("##UIProtection", &choice, "On\0Off\0Auto\0")) {
          if (edit_source_alpha_mode(*data, modes[choice], config, GetTickCount64()))
            process_alpha_mode = data->values.ui_protection;
        }
        ImGui::SetItemTooltip("On and Off are saved per game. Auto finds a compatible UI source and checks its quality automatically. Frame Generation requires submitted usable input pixels.");
      }
      ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::PopID();
    return data->alive;
  }

  void draw_status(api::effect_runtime *runtime) {
    if (!runtime) return;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return;
    const auto data = get_state(runtime);
    const auto automatic = query_automatic(runtime);
    const auto source_alpha = query_source_alpha_ui(runtime);
    const auto &fg = source_alpha.fg;
    const auto source = ui_input::source_status(runtime);
    ImGui::TextWrapped("%s", output_status_text(automatic, data->values));
    if (!data->values.enabled)
      ImGui::TextUnformatted("UI protection: inactive while Game 3D is disabled");
    else if (data->values.ui_protection == source_alpha_mode::off)
      ImGui::TextUnformatted("UI protection: off");
    const bool dedicated_ui = source_alpha.dedicated_ui_active(data->values.ui_protection, data->values.enabled);
    if (data->values.enabled && data->values.ui_protection == source_alpha_mode::automatic)
      ImGui::TextWrapped("UI protection: %s", source_alpha_detection_text(source, source_alpha));
    else if (data->values.enabled && data->values.ui_protection == source_alpha_mode::on)
      if (const auto blocked = source_alpha_capture_block_text(source_alpha_capture_block_for(source.selected, fg)))
        ImGui::TextWrapped("UI protection: %s", blocked);
    if (source.candidate.source != ui_qualification::choice::automatic)
      ImGui::TextWrapped("Candidate: %s", ui_qualification::name(source.candidate.source));
    if (source.selected == ui_qualification::choice::sl_ui_alpha && data->values.ui_protection == source_alpha_mode::automatic &&
        !sunshine_streamline::buffers::ui_alpha_authenticated(sunshine_streamline::buffers::active()))
      ImGui::TextWrapped("UI alpha tag meaning is unverified for this game's Streamline version. This source cannot pass automatic detection.");
    if (data->values.enabled && data->values.ui_protection != source_alpha_mode::off &&
        source_alpha.mode == data->values.ui_protection && source_alpha.blocked_by_fg()) {
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
    } else if (data->values.ui_protection == source_alpha_mode::on &&
        source_alpha.applied_for(data->values.ui_protection, data->values.enabled)) {
      ImGui::TextWrapped("UI protection: %s", source_alpha.input == source_alpha_input::sl_hudless_difference ?
        "using HUDless color difference" : dedicated_ui ? "using a dedicated UI mask" : source_alpha.retained_alpha_ready ?
        "using captured input alpha" : "using source alpha");
    }

    ImGui::Spacing();
    if (!fg.known) ImGui::TextUnformatted("Frame Generation: unknown");
    else if (!fg.enabled) ImGui::TextUnformatted("Frame Generation: off");
    if (fg.known && fg.enabled) {
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
    namespace buffers = sunshine_streamline::buffers;
    if (const auto contract = buffers::active(); contract.known) {
      char text[128]{};
      const auto &v = contract.version;
      if (contract.raw_ui_alpha != buffers::unknown)
        std::snprintf(text, sizeof(text), "Streamline %u.%u.%u: UI alpha tag %u (%s)", v.major, v.minor, v.patch,
          contract.raw_ui_alpha, buffers::ui_alpha_authenticated(contract) ? "automatic" : "manual only");
      else
        std::snprintf(text, sizeof(text), "Streamline %u.%u.%u: this SDK has no UI alpha tag", v.major, v.minor, v.patch);
      ImGui::TextUnformatted(text);
    }

    camera_hint(automatic, data->values.enabled, fg);
    if (data->values.depth_view != 0)
      ImGui::TextWrapped("Depth preview is selected. To return to gameplay, choose Troubleshooting > Depth view > Game image.");
    ImGui::Spacing();
  }

  bool draw_diagnostics(api::effect_runtime *runtime) {
    if (!runtime) return false;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return true;
    const auto data = get_state(runtime);
    config_backend config {runtime};
    ImGui::PushID("SunshineGame3DDiagnostics");
    if (begin_controls("DiagnosticControls")) {
      depth_view_control(*data, config);
      if (data->alive) {
        control_row("UI mask source");
        const auto source = ui_input::source_status(runtime);
        int choice = static_cast<int>(source.selected);
        record_control_position(4);
        if (ImGui::Combo("##UIMaskSource", &choice,
            "Discover automatically\0Current color alpha\0Streamline UI alpha\0Streamline UI color alpha\0Streamline real-input alpha\0HUDless color difference\0"))
          ui_input::select_source(runtime, static_cast<ui_qualification::choice>(choice));
        ImGui::SetItemTooltip("Automatic discovery chooses a usable UI source. Select a specific source to troubleshoot it; selection does not bypass capture or quality checks.\nSelected: %s", ui_qualification::name(source.selected));
      }
      ImGui::EndTable();
    }
    ImGui::PopID();
    return data->alive;
  }

  void draw_calibration(api::effect_runtime *runtime) {
    if (!runtime) return;
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    if (!registered) return;
    calibration_status(query_automatic(runtime), get_state(runtime)->values.strength);
  }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  // Scalar-only adapters retain the existing test ABI while exercising native
  // config ownership. No test path reflects, modifies or saves shader presets.
  extern "C" __declspec(dllexport) BOOL SunshineGame3DTestControlPosition(unsigned index, float *x, float *y) {
    if (index >= 5 || !x || !y) return FALSE;
    *x = control_positions[index][0];
    *y = control_positions[index][1];
    return TRUE;
  }

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
