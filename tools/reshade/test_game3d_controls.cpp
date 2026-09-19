// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_controls_model.h"

#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {
  using namespace sunshine_game3d;

  void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }

  struct fake_config {
    std::map<std::string, std::variant<float, int, bool>> values;
    std::vector<std::string> writes;
    std::function<void()> on_write;

    template<class T> void read(const char *key, T &value) {
      const auto found = values.find(key);
      if (found != values.end() && std::holds_alternative<T>(found->second)) value = std::get<T>(found->second);
    }
    template<class T> void write(const char *key, T value) {
      if (on_write) on_write();
      values[key] = value;
      writes.emplace_back(key);
    }
  };

  void native_defaults_and_saved_values() {
    fake_config config;
    auto settings = load_settings(config);
    check(settings.enabled && settings.strength == 50.f && settings.depth_view == 0 && config.writes.empty(),
      "Native defaults or passive-load persistence differ");
    config.values = {{"Strength", 0.f}, {"DepthView", 2}, {"Enabled", false}, {"Unrelated", 17}};
    settings = load_settings(config);
    check(!settings.enabled && settings.strength == 0.f && settings.depth_view == 2 && config.writes.empty(),
      "Saved zero, preview, disabled state or passive load were altered");
    config.values["Strength"] = 150.12345f;
    settings = load_settings(config);
    check(settings.strength == 150.12345f && config.writes.empty(), "Opening native controls clamped a finite saved value");
    config.values["Strength"] = std::numeric_limits<float>::quiet_NaN();
    config.values["DepthView"] = 9;
    settings = load_settings(config);
    check(settings.strength == 50.f && settings.depth_view == 0 && config.writes.empty(),
      "Invalid config reached rendering or inspection rewrote the user's file");
  }

  void edits_save_exactly_one_owned_setting() {
    fake_config config;
    config.values = {{"Strength", 0.f}, {"DepthView", 2}, {"Enabled", false}, {"Unrelated", 17}};
    settings_state settings {load_settings(config)};
    const auto original = config.values;
    check(edit_strength(settings, default_strength, config), "Strength reset failed");
    check(settings.values.strength == 50.f && std::get<float>(config.values.at("Strength")) == 50.f &&
      config.writes == std::vector<std::string>{"Strength"} && config.values.at("DepthView") == original.at("DepthView") &&
      config.values.at("Enabled") == original.at("Enabled") && config.values.at("Unrelated") == original.at("Unrelated"),
      "Strength reset changed another field or failed automatic persistence");
    check(!edit_strength(settings, 50.f, config) && config.writes.size() == 1, "Reset at default performed redundant saving");
    check(edit_depth_view(settings, 0, config) && settings.values.depth_view == 0 &&
      std::get<int>(config.values.at("DepthView")) == 0 && settings.values.strength == 50.f,
      "Preview reset changed strength or did not save");
    check(edit_enabled(settings, true, config) && settings.values.enabled && std::get<bool>(config.values.at("Enabled")) &&
      !edit_enabled(settings, true, config) && config.writes.size() == 3, "Enable state did not persist exactly once");
    check(edit_strength(settings, 37.250123f, config) && settings.values.strength == 37.250123f &&
      std::get<float>(config.values.at("Strength")) == 37.250123f, "Native persistence lost scalar precision");
  }

  void invalid_edits_do_not_reach_config() {
    fake_config config;
    settings_state settings;
    check(!edit_strength(settings, std::numeric_limits<float>::quiet_NaN(), config) &&
      !edit_strength(settings, std::numeric_limits<float>::infinity(), config) &&
      !edit_depth_view(settings, -1, config) && !edit_depth_view(settings, 3, config) && config.writes.empty(),
      "An invalid renderer setting was persisted");
    settings.alive = false;
    check(!edit_strength(settings, 10.f, config) && !edit_depth_view(settings, 2, config) &&
      !edit_enabled(settings, false, config) && config.writes.empty(), "Destroyed runtime accepted an edit");
  }

  void persistence_survives_runtime_recreation() {
    fake_config first_config, other_config;
    settings_state first {load_settings(first_config)}, other {load_settings(other_config)};
    check(edit_strength(first, 23.4567f, first_config) && edit_depth_view(first, 1, first_config) &&
      edit_enabled(first, false, first_config), "Native persistence setup failed");
    first.alive = false;
    settings_state recreated {load_settings(first_config)};
    check(recreated.values.strength == 23.4567f && recreated.values.depth_view == 1 && !recreated.values.enabled &&
      first_config.writes.size() == 3, "Runtime recreation lost settings or generated duplicate writes");
    check(other.values.strength == 50.f && other.values.depth_view == 0 && other.values.enabled && other_config.writes.empty(),
      "One runtime's settings leaked into another configuration");
  }

  void config_failure_and_lifetime_do_not_commit_stale_values() {
    fake_config config;
    settings_state settings;
    config.on_write = [] { throw std::runtime_error("Synthetic config failure"); };
    bool threw = false;
    try { edit_strength(settings, 20.f, config); } catch (const std::runtime_error &) { threw = true; }
    check(threw && settings.values.strength == 50.f && config.writes.empty(), "Failed persistence changed the applied setting");
    config.on_write = [&] { settings.alive = false; };
    check(!edit_strength(settings, 20.f, config) && settings.values.strength == 50.f,
      "A destroyed runtime's retained state was changed after config persistence");
  }

  void output_status_distinguishes_flat_preview_and_stereo() {
    automatic_status status;
    status.phase = automatic_phase::ready;
    render_settings settings;
    const auto label = [&] { return std::string_view(output_status_text(status, settings)); };
    check(label() == "3D active", "Ready stereo was not reported active");
    settings.enabled = false;
    check(label() == "2D: Game 3D is disabled", "Disabled native rendering reported active stereo");
    settings.enabled = true;
    settings.strength = 0.f;
    check(label() == "2D: 3D strength is off", "Flat strength reported active stereo");
    settings.depth_view = 2;
    check(label() == "Depth preview: normal depth", "Normal depth was hidden at zero strength");
    settings.depth_view = 1;
    check(label() == "2D: 3D strength is off", "Zero-strength stereo diagnostic bypassed the mono gate");
    settings.strength = 50.f;
    check(label() == "Depth preview: stereo depth", "Stereo depth was labeled as the game image");
    for (auto phase : {automatic_phase::unavailable, automatic_phase::waiting_for_depth,
        automatic_phase::calibrating, automatic_phase::suspended, automatic_phase::unsupported_resolution,
        automatic_phase::renderer_unavailable}) {
      status.phase = phase;
      for (int view = 0; view < 3; ++view) {
        settings.depth_view = view;
        check(label().substr(0, 3) == "2D:", "Unavailable depth reported active stereo or a usable preview");
      }
    }
  }

  void automatic_scale_tracks_applied_basis() {
    using basis = automatic_scale_basis;
    using availability = automatic_scale_state;
    automatic_scale scale;
    check(scale.state() == availability::unavailable && !scale.has_value(), "Uninitialized scale was displayed as active zero");
    scale = retain_automatic_scale(scale, {basis::camera_matrix, 0.f, true});
    check(scale.state() == availability::pending && !scale.has_value(), "Temporary camera default became an active scale");
    scale = retain_automatic_scale(scale, {basis::camera_matrix, 512.f, false});
    check(scale.state() == availability::pending && !scale.has_value(), "Unapplied camera target was displayed as a rendered scale");
    scale = retain_automatic_scale(scale, {basis::camera_matrix, 257.125f, true, 128.f});
    check(scale.state() == availability::active && scale.value == 257.125f, "Display did not retain the applied stereo reference");
    check(scale.has_target() && scale.target_value == 128.f, "Stereo-reference target did not accompany the applied value");
    scale = retain_automatic_scale(scale, {});
    check(!scale.has_target() && scale.target_value == 0.f, "Missing depth kept a stale stereo-reference target visible");
    check(scale.state() == availability::held && scale.basis == basis::camera_matrix && scale.value == 257.125f,
      "Unavailable current depth reset/relabelled the last applied camera scale");
    scale = retain_automatic_scale(scale, {basis::camera_matrix, std::numeric_limits<float>::quiet_NaN(), true});
    check(scale.state() == availability::held && scale.value == 257.125f, "Invalid scale replaced a valid retained value");
    scale = retain_automatic_scale(scale, {basis::relative_depth, 0.f, false});
    check(scale.state() == availability::pending && scale.basis == basis::relative_depth && !scale.has_value(),
      "Basis switch relabeled a camera reference as uncalibrated relative depth");
    scale = retain_automatic_scale(scale, {basis::relative_depth, 950.75f, true});
    check(scale.state() == availability::active && scale.value == 950.75f, "Relative reference did not match the applied value");
    scale = retain_automatic_scale(scale, {basis::relative_depth, 0.f, false});
    check(scale.state() == availability::held && scale.value == 950.75f, "Recalibration wait displayed a false zero reset");
    scale = retain_automatic_scale(scale, {basis::relative_depth, 949.875f, true});
    check(scale.state() == availability::active && scale.value == 949.875f, "Explicitly reset reference did not replace its held predecessor");
    scale = retain_automatic_scale(scale, {basis::camera_matrix, 0.f, false});
    check(scale.state() == availability::pending && !scale.has_value(), "Raw H was incorrectly reused for camera K");
    check(automatic_status{}.scale.state() == availability::unavailable, "A new runtime inherited another runtime's scale");
  }

  void projection_conversion_is_independent_and_retained_with_its_reference() {
    using basis = automatic_scale_basis;
    using availability = automatic_scale_state;
    automatic_scale current{basis::camera_matrix, 0.f, false, 0.f, 16.0, -.5, true};
    auto shown = retain_automatic_scale({}, current);
    check(shown.state() == availability::pending && !shown.has_value() && shown.projection_conversion_valid() &&
      shown.conversion_multiplier == 16.0 && shown.conversion_offset == -.5,
      "Known projection conversion was hidden while scene scale was still being estimated");

    current.value = 4.f;
    current.target_value = 2.f;
    current.active = true;
    shown = retain_automatic_scale(shown, current);
    check(shown.has_target() && shown.value == 4.f && shown.target_value == 2.f &&
      shown.conversion_multiplier == 16.0 && shown.conversion_offset == -.5,
      "Stereo-reference values were confused with independent projection conversion coefficients");

    current.value = 0.f;
    current.active = false;
    current.conversion_multiplier = 8.0;
    current.conversion_offset = .25;
    shown = retain_automatic_scale(shown, current);
    check(shown.state() == availability::held && !shown.has_target() && shown.value == 4.f &&
      shown.projection_conversion_valid() && shown.conversion_multiplier == 16.0 && shown.conversion_offset == -.5,
      "Held stereo reference was relabeled using unapplied current projection coefficients");
    shown = retain_automatic_scale(shown, {});
    check(shown.state() == availability::held && shown.projection_conversion_valid() && shown.conversion_multiplier == 16.0,
      "Missing depth destroyed the last-applied projection/reference pairing");

    current.value = 3.f;
    current.active = true;
    shown = retain_automatic_scale(shown, current);
    check(shown.state() == availability::active && shown.value == 3.f &&
      shown.conversion_multiplier == 8.0 && shown.conversion_offset == .25,
      "New applied projection/reference pair did not replace held values together");

    shown = retain_automatic_scale(shown, {basis::relative_depth, 0.f, false});
    check(shown.state() == availability::pending && !shown.projection_conversion_valid() &&
      !shown.has_projection_conversion && shown.conversion_multiplier == 0.0 && shown.conversion_offset == 0.0,
      "Relative-depth source inherited or fabricated a projection conversion");
    current.basis = basis::relative_depth;
    check(!current.projection_conversion_valid(), "Relative raw depth was mislabeled as a calculated projection");
    current.basis = basis::camera_matrix;
    current.conversion_multiplier = -16.0;
    current.conversion_offset = 16.0;
    check(current.projection_conversion_valid(), "Normal-depth conversion rejected a valid negative multiplier");
    current.conversion_offset = 0.0;
    check(current.projection_conversion_valid(), "Reversed infinite projection rejected its valid zero offset");
    current.conversion_multiplier = 0.0;
    check(!current.projection_conversion_valid(), "Default zero multiplier looked like a calculated projection");
    current.conversion_multiplier = std::numeric_limits<double>::infinity();
    check(!current.projection_conversion_valid(), "Nonfinite projection multiplier was displayable");
    current.conversion_multiplier = 16.0;
    current.conversion_offset = std::numeric_limits<double>::quiet_NaN();
    check(!current.projection_conversion_valid(), "Nonfinite projection offset was displayable");
    current.conversion_offset = 0.0;
    current.has_projection_conversion = false;
    check(!current.projection_conversion_valid(), "Unverified numeric defaults were displayed as calculated coefficients");
  }

}  // namespace

int main() {
  try {
    native_defaults_and_saved_values();
    edits_save_exactly_one_owned_setting();
    invalid_edits_do_not_reach_config();
    persistence_survives_runtime_recreation();
    config_failure_and_lifetime_do_not_commit_stale_values();
    output_status_distinguishes_flat_preview_and_stereo();
    automatic_scale_tracks_applied_basis();
    projection_conversion_is_independent_and_retained_with_its_reference();
    std::puts("PASS native Game 3D controls: defaults, automatic persistence, independent resets, runtime lifetime, output status and scale/conversion");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
