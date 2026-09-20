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
    check(settings.enabled && settings.strength == 50.f && settings.depth_view == 0 && !settings.source_alpha_ui && config.writes.empty(),
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

  void source_alpha_ui_is_explicit_persistent_and_runtime_scoped() {
    fake_config config, other_config;
    config.values = {{"Strength", 23.5f}, {"DepthView", 2}, {"Enabled", false}, {"Unrelated", 17}};
    const auto original = config.values;
    settings_state settings {load_settings(config)};
    check(!settings.values.source_alpha_ui && config.writes.empty(), "Source alpha UI was enabled implicitly");
    check(edit_source_alpha_ui(settings, true, config) && settings.values.source_alpha_ui &&
      std::get<bool>(config.values.at("SourceAlphaUI")) && config.writes == std::vector<std::string>{"SourceAlphaUI"},
      "Source alpha interpretation did not persist as an explicit boolean");
    for (const auto &[key, value] : original)
      check(config.values.at(key) == value, "Source alpha edit changed an unrelated setting");
    check(settings.values.strength == 23.5f && settings.values.depth_view == 2 && !settings.values.enabled &&
      !edit_source_alpha_ui(settings, true, config) && config.writes.size() == 1,
      "Source alpha edit changed rendering controls or saved redundantly");
    settings_state recreated {load_settings(config)};
    check(recreated.values.source_alpha_ui && config.writes.size() == 1 && !load_settings(other_config).source_alpha_ui,
      "Source alpha interpretation was lost on recreation or leaked into another game");
    check(edit_source_alpha_ui(recreated, false, config) && !load_settings(config).source_alpha_ui &&
      !std::get<bool>(config.values.at("SourceAlphaUI")) && config.writes.size() == 2,
      "Disabling source alpha UI did not persist");
    recreated.alive = false;
    check(!edit_source_alpha_ui(recreated, true, config) && config.writes.size() == 2,
      "Destroyed runtime accepted a source alpha edit");
  }

  void source_alpha_ui_persistence_failure_does_not_commit() {
    fake_config config;
    settings_state settings;
    config.on_write = [] { throw std::runtime_error("Synthetic config failure"); };
    bool threw = false;
    try { edit_source_alpha_ui(settings, true, config); } catch (const std::runtime_error &) { threw = true; }
    check(threw && !settings.values.source_alpha_ui && config.writes.empty(),
      "Failed source alpha persistence changed the applied setting");
    config.on_write = [&] { settings.alive = false; };
    check(!edit_source_alpha_ui(settings, true, config) && !settings.values.source_alpha_ui,
      "Source alpha callback committed after runtime destruction");
  }

  void source_alpha_fg_mode_survives_observation_gaps() {
    source_alpha_ui_policy policy;
    check(policy.update(true, {}, true).effective(), "An unobserved FG mode disabled an explicit non-FG alpha preference");
    frame_generation_mode fg {true, true, false, 1, 0, 7, 41};
    const auto initial = policy.update(true, fg, true);
    check(initial.blocked_by_fg() && !initial.effective(), "Confirmed FG did not block presentation alpha");
    auto retained = initial;
    retained.retained_alpha_ready = true;
    check(retained.effective() && !retained.blocked_by_fg(), "Completed real alpha did not enable FG UI protection");
    retained.requested = false;
    check(!retained.effective(), "Retained alpha overrode the user's disabled preference");
    // Busy, ambiguous and missing observations all carry no confirmed mode.
    // None is an FG-off event, including while Generic depth is manually pinned.
    for (unsigned missing = 0; missing < 3; ++missing) {
      const auto held = policy.update(true, {}, true);
      check(held.blocked_by_fg() && !held.effective() && held.fg.epoch == 7 && held.fg.sequence == 41,
        "Observation loss reenables alpha or loses the confirmed mode's identity");
    }
    const auto disabled = policy.update(false, {}, true);
    check(!disabled.requested && !disabled.effective() && disabled.fg.enabled,
      "Disabling the UI preference lost the independent FG observation");
    fg.enabled = false; fg.sequence = 42;
    check(policy.update(true, fg, true).effective(), "Explicit FG Off did not restore the saved preference");
    fg.enabled = true; fg.epoch = 8; fg.sequence = 1;
    check(policy.update(true, fg, true).blocked_by_fg(), "New confirmed observer scope was not accepted");
    const auto stopped = policy.update(true, {}, false);
    check(!stopped.fg.known && stopped.effective(), "Observer shutdown retained a previous scope's FG mode");
    source_alpha_ui_policy recreated;
    check(!recreated.update(true, {}, true).fg.known, "A new runtime inherited the old runtime's mode");
    check(initial.fg.enabled && !initial.effective(), "Later mode changes mutated an already frozen presentation");
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

  void measured_statistics_follow_applied_and_held_scale_ownership() {
    using basis = automatic_scale_basis;
    automatic_scale current{basis::camera_matrix, 4.f, true, 2.f};
    current.reference_inverse = 2.;
    current.minimum_inverse = 0.;
    current.maximum_inverse = 2.;
    current.mean_inverse = .3;
    current.normalization = 4.;
    current.zero_inverse = .25f;
    current.has_zero = true;
    current.target_zero_inverse = 1.;
    current.zero_target_available = true;
    current.mean_square_inverse = .2;
    current.depth_pixel_count = 3072 * 2304;
    current.depth_tiles_x = 28; current.depth_tiles_y = 21;
    current.has_depth_statistics = true;
    auto shown = retain_automatic_scale({}, current);
    check(shown.depth_statistics_valid() && shown.reference_inverse == 2. && shown.mean_inverse == .3 &&
        shown.normalization == 4. && shown.minimum_inverse == 0. &&
        shown.has_zero_target() && shown.zero_inverse == .25f && shown.target_zero_inverse == 1. &&
        shown.reference_inverse != 1. / shown.value && shown.reference_inverse != 1. / shown.target_value,
      "Measured nearest reference was confused with the mean, reciprocal gain or infinite-far endpoint");

    auto pending = current;
    pending.active = false;
    pending.reference_inverse = 1.;
    pending.maximum_inverse = 1.;
    pending.mean_inverse = .8;
    pending.normalization = 16.;
    pending.target_zero_inverse = .5;
    shown = retain_automatic_scale(shown, pending);
    check(shown.state() == automatic_scale_state::held && shown.depth_statistics_valid() && shown.reference_inverse == 2. &&
        shown.mean_inverse == .3 && shown.normalization == 4. &&
        !shown.has_zero_target() && !shown.zero_target_available && shown.target_zero_inverse == 0. &&
        shown.zero_inverse == .25f &&
        shown.depth_pixel_count == 3072 * 2304 && shown.depth_tiles_x == 28 && shown.mean_square_inverse == .2,
      "Held applied scale acquired statistics from an unapplied frame");
    shown = retain_automatic_scale(shown, {});
    check(shown.state() == automatic_scale_state::held && shown.depth_statistics_valid() && shown.maximum_inverse == 2. &&
        shown.mean_inverse == .3 && shown.normalization == 4.,
      "Missing depth lost the last-applied held statistics");
    current.target_value = 0.f;
    current.has_depth_statistics = false;
    shown = retain_automatic_scale(shown, current);
    check(shown.state() == automatic_scale_state::active && !shown.depth_statistics_valid() && !shown.has_zero_target(),
      "Active frame with expired target resurrected earlier fresh statistics");
    pending.basis = basis::relative_depth;
    shown = retain_automatic_scale(shown, pending);
    check(shown.state() == automatic_scale_state::pending && !shown.has_depth_statistics &&
        shown.reference_inverse == 0. && shown.minimum_inverse == 0. && shown.maximum_inverse == 0. &&
        shown.mean_inverse == 0. && shown.normalization == 0. &&
        shown.target_zero_inverse == 0. && !shown.zero_target_available && !shown.has_zero_target() &&
        shown.depth_pixel_count == 0 && shown.depth_tiles_x == 0 && shown.depth_tiles_y == 0 && shown.mean_square_inverse == 0.,
      "Pending basis change retained camera statistics as raw statistics");
    check(!retain_automatic_scale({}, pending).depth_statistics_valid(),
      "Statistics from a never-applied scale became visible during startup");

    current.has_depth_statistics = true;
    for (unsigned fault = 0; fault != 17; ++fault) {
      auto invalid = current;
      switch (fault) {
        case 0: invalid.reference_inverse = std::numeric_limits<double>::quiet_NaN(); break;
        case 1: invalid.minimum_inverse = -1.; break;
        case 2: invalid.maximum_inverse = std::numeric_limits<double>::infinity(); break;
        case 3: invalid.minimum_inverse = 3.; break;
        case 4: invalid.reference_inverse = -.1; break;
        case 5: invalid.reference_inverse = 2.1; break;
        case 6: invalid.basis = basis::unknown; break;
        case 7: invalid.mean_square_inverse = std::numeric_limits<double>::quiet_NaN(); break;
        case 8: invalid.depth_tiles_x = 0; break;
        case 9: invalid.reference_inverse = .3; break;
        case 10: invalid.mean_inverse = std::numeric_limits<double>::quiet_NaN(); break;
        case 11: invalid.mean_inverse = -.1; break;
        case 12: invalid.mean_inverse = 2.1; break;
        case 13: invalid.normalization = 0.; break;
        case 14: invalid.normalization = -1.; break;
        case 15: invalid.normalization = std::numeric_limits<double>::infinity(); break;
        case 16: invalid.normalization = std::numeric_limits<double>::quiet_NaN(); break;
      }
      check(!invalid.depth_statistics_valid(), "Invalid depth statistics passed UI validation");
    }
    check(!automatic_scale{}.depth_statistics_valid(), "Default statistic zeros were shown as measured data");
  }

  void positive_flat_zero_target_does_not_invent_gain_tracking() {
    automatic_scale measured{automatic_scale_basis::relative_depth, 4.f, true, 0.f};
    measured.has_zero = true;
    measured.zero_inverse = .1f;
    measured.reference_inverse = measured.minimum_inverse = measured.maximum_inverse = .4;
    measured.mean_inverse = .4;
    measured.normalization = 7.68;
    measured.has_depth_statistics = true;
    measured.target_zero_inverse = .4;
    measured.zero_target_available = true;
    const auto shown = retain_automatic_scale({}, measured);
    check(shown.state() == automatic_scale_state::active && shown.depth_statistics_valid() &&
        shown.has_zero_target() && shown.target_zero_inverse == .4 && shown.zero_inverse == .1f &&
        !shown.has_target() && shown.value == 4.f,
      "A positive flat scene hid its zero target or fabricated a gain target");
    const auto held = retain_automatic_scale(shown, {});
    check(held.value == shown.value && held.zero_inverse == shown.zero_inverse &&
        !held.has_target() && !held.has_zero_target() && held.target_zero_inverse == 0.,
      "Missing depth exposed a stale flat-scene zero target");
    for (double bad_target : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
      auto invalid = measured;
      invalid.target_zero_inverse = bad_target;
      check(!invalid.has_zero_target(), "Invalid zero target passed UI validation");
    }
    measured.zero_target_available = false;
    check(!measured.has_zero_target(), "Unavailable zero target was inferred from a default numeric value");
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
    source_alpha_ui_is_explicit_persistent_and_runtime_scoped();
    source_alpha_ui_persistence_failure_does_not_commit();
    source_alpha_fg_mode_survives_observation_gaps();
    persistence_survives_runtime_recreation();
    config_failure_and_lifetime_do_not_commit_stale_values();
    output_status_distinguishes_flat_preview_and_stereo();
    automatic_scale_tracks_applied_basis();
    measured_statistics_follow_applied_and_held_scale_ownership();
    positive_flat_zero_target_does_not_invent_gain_tracking();
    projection_conversion_is_independent_and_retained_with_its_reference();
    std::puts("PASS native Game 3D controls: defaults, automatic persistence, independent resets, runtime lifetime, output status and scale/conversion");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
