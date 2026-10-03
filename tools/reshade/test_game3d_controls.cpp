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
    check(settings.enabled && settings.strength == 50.f && settings.depth_view == 0 &&
      settings.ui_protection == source_alpha_mode::automatic && !settings.source_alpha_ui && config.writes.empty(),
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

  void source_alpha_modes_are_persistent_and_game_scoped() {
    fake_config config, other_config;
    config.values = {{"Strength", 23.5f}, {"DepthView", 2}, {"Enabled", false}, {"SourceAlphaUI", false}, {"Unrelated", 17}};
    const auto original = config.values;
    settings_state settings {load_settings(config)};
    check(settings.values.ui_protection == source_alpha_mode::off && !settings.values.source_alpha_ui && config.writes.empty(),
      "Legacy disabled source alpha did not migrate to Off without rewriting config");
    unsigned writes = 0;
    for (const auto mode : {source_alpha_mode::on, source_alpha_mode::off, source_alpha_mode::automatic}) {
      check(edit_source_alpha_mode(settings, mode, config), "Source alpha mode edit did not apply");
      ++writes;
      check(settings.values.ui_protection == mode && settings.values.source_alpha_ui == (mode == source_alpha_mode::on) &&
        std::get<int>(config.values.at("SourceAlphaUIMode")) == int(mode) && config.writes.size() == writes &&
        config.writes.back() == "SourceAlphaUIMode", "Source alpha mode was not saved to its owned integer setting");
      for (const auto &[key, value] : original)
        check(config.values.at(key) == value, "Source alpha mode edit changed a legacy or unrelated setting");
      check(settings.values.strength == 23.5f && settings.values.depth_view == 2 && !settings.values.enabled &&
        !edit_source_alpha_mode(settings, mode, config) && config.writes.size() == writes,
        "Source alpha edit changed rendering controls or saved redundantly");
      settings_state recreated {load_settings(config)};
      check(recreated.values.ui_protection == mode && recreated.values.source_alpha_ui == (mode == source_alpha_mode::on) &&
        config.writes.size() == writes, "Source alpha mode changed during passive runtime recreation");
    }
    const auto other = load_settings(other_config);
    check(other.ui_protection == source_alpha_mode::automatic && !other.source_alpha_ui && other_config.writes.empty(),
      "Another game inherited the edited game's source alpha preference");
    check(!edit_source_alpha_mode(settings, source_alpha_mode(-1), config) &&
      !edit_source_alpha_mode(settings, source_alpha_mode(3), config) && config.writes.size() == writes,
      "An invalid source alpha mode reached persistence");
    settings.alive = false;
    check(!edit_source_alpha_mode(settings, source_alpha_mode::on, config) && config.writes.size() == writes,
      "Destroyed runtime accepted a source alpha mode edit");
  }

  void source_alpha_mode_migration_and_precedence() {
    for (bool legacy : {false, true}) {
      fake_config config;
      config.values["SourceAlphaUI"] = legacy;
      const auto migrated = load_settings(config);
      check(migrated.ui_protection == (legacy ? source_alpha_mode::automatic : source_alpha_mode::off) &&
        !migrated.source_alpha_ui && config.writes.empty(), "Legacy source alpha migration changed meaning or rewrote settings");
      for (int mode : {0, 1, 2}) {
        config.values["SourceAlphaUIMode"] = mode;
        const auto explicit_mode = load_settings(config);
        check(explicit_mode.ui_protection == source_alpha_mode(mode) && explicit_mode.source_alpha_ui == (mode == 1) &&
          config.writes.empty(), "Legacy checkbox overrode the saved three-way source alpha mode");
      }
      for (int invalid : {-1, 3, 99}) {
        config.values["SourceAlphaUIMode"] = invalid;
        const auto fallback = load_settings(config);
        check(fallback.ui_protection == source_alpha_mode::automatic && !fallback.source_alpha_ui && config.writes.empty(),
          "Invalid source alpha configuration enabled protection or caused a passive write");
      }
    }
  }

  void source_alpha_mode_persistence_failure_does_not_commit() {
    fake_config config;
    settings_state settings;
    settings.alpha_session = std::make_shared<alpha_auto_policy>();
    config.on_write = [] { throw std::runtime_error("Synthetic config failure"); };
    bool threw = false;
    try { edit_source_alpha_mode(settings, source_alpha_mode::on, config); } catch (const std::runtime_error &) { threw = true; }
    check(threw && settings.values.ui_protection == source_alpha_mode::automatic && !settings.values.source_alpha_ui &&
      settings.alpha_session->decision().state == alpha_auto_state::waiting_for_source && config.writes.empty(),
      "Failed source alpha persistence changed the applied mode or session latch");
    config.on_write = [&] { settings.alive = false; };
    check(!edit_source_alpha_mode(settings, source_alpha_mode::on, config) &&
      settings.values.ui_protection == source_alpha_mode::automatic && !settings.values.source_alpha_ui &&
      settings.alpha_session->decision().state == alpha_auto_state::waiting_for_source,
      "Source alpha callback committed a mode or manual latch after runtime destruction");
  }

  void source_alpha_mode_edits_control_the_session() {
    fake_config config;
    settings_state settings;
    settings.alpha_session = std::make_shared<alpha_auto_policy>();
    check(edit_source_alpha_mode(settings, source_alpha_mode::off, config), "Manual Off edit was rejected");
    auto applied = settings.alpha_session->decision();
    check(!applied.enabled && applied.state == alpha_auto_state::manual_off, "Manual Off did not apply immediately");
    check(edit_source_alpha_mode(settings, source_alpha_mode::on, config), "Manual On edit was rejected");
    applied = settings.alpha_session->decision();
    check(applied.enabled && applied.state == alpha_auto_state::manual_on, "Manual On did not apply immediately");
    check(edit_source_alpha_mode(settings, source_alpha_mode::automatic, config), "Return to Auto was rejected");
    applied = settings.alpha_session->decision();
    check(!applied.enabled && applied.state == alpha_auto_state::waiting_for_source, "Returning to Auto kept the manual mode");
    check(load_settings(config).ui_protection == source_alpha_mode::automatic && !load_settings(config).source_alpha_ui,
      "Loading Auto confused its selected mode with the previous session's applied result");
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

  void source_alpha_selection_reports_capture_prerequisite() {
    using choice = ui_qualification::choice;
    using block = source_alpha_capture_block;
    source_alpha_ui_decision value;
    for (const auto selected : {choice::sl_ui_alpha, choice::sl_ui_color_alpha, choice::sl_backbuffer, choice::sl_hudless}) {
      value.qualification.selected = selected;
      for (bool known : {false, true}) for (bool enabled : {false, true}) {
        value.fg = {}; value.fg.known = known; value.fg.enabled = enabled;
        check(source_alpha_capture_block_for(selected, value.fg) == block::none &&
            std::string_view(value.source_availability()) == "source_unavailable" &&
            std::string_view(source_alpha_detection_text(value.qualification, value)).find("Finding a usable UI source") != std::string_view::npos,
          "Captured Streamline source incorrectly required Frame Generation or looked ready before capture");
      }
    }
    value.qualification.selected = choice::current_color;
    value.fg.known = value.fg.enabled = true;
    check(source_alpha_capture_block_for(choice::current_color, value.fg) == block::generated_current_color &&
        std::string_view(source_alpha_detection_text(value.qualification, value)).find("unavailable during Frame Generation") != std::string_view::npos,
      "Generated current-color alpha was offered as a real UI source");
    value.fg = {};
    for (const auto selected : {choice::automatic, choice::current_color}) {
      value.qualification.selected = selected;
      for (bool known : {false, true}) {
        value.fg = {}; value.fg.known = known;
        check(source_alpha_capture_block_for(selected, value.fg) == block::none &&
            std::string_view(value.source_availability()) == "source_unavailable",
          "Current color discovery incorrectly required enabled FG");
      }
    }
    value.qualification.available = true;
    check(std::string_view(value.source_availability()) == "checking_quality" &&
        std::string_view(source_alpha_detection_text(value.qualification, value)) == "Checking source quality",
      "Pending diagnostic feedback claimed the current GPU mask was off");
    value.coverage.state = alpha_auto_state::automatic_off;
    check(std::string_view(value.source_availability()) == "quality_rejected" &&
        std::string_view(source_alpha_detection_text(value.qualification, value)) == "No usable UI mask detected",
      "Completed quality feedback claimed to control the current GPU mask");
    value.coverage.state = alpha_auto_state::automatic_on;
    value.coverage.enabled = true;
    check(std::string_view(value.source_availability()) == "detected" &&
        std::string_view(source_alpha_detection_text(value.qualification, value)).find("detected automatically") != std::string_view::npos,
      "Automatic detection still required a human approval flag");
    auto replacement = value.qualification;
    replacement.candidate.revision++;
    check(std::string_view(source_alpha_detection_text(replacement, value)) == "Checking source quality",
      "A replacement source inherited the previous source's automatic detection status");
    value.qualification.available = false;
    check(std::string_view(value.source_availability()) == "source_unavailable" &&
        std::string_view(source_alpha_detection_text(value.qualification, value)).find("Finding a usable UI source") != std::string_view::npos,
      "An automatically detected source with lost pixels was still described as available");
  }

  void ui_protection_warning_follows_unprotected_rendered_frames() {
    // Stellar Blade in SDR without FG: its UI layer holds the scene image (no
    // alpha, color on 60%, so V1-invalid) and current alpha covers
    // everything, so no source decides and no dedicated candidate shows no UI.
    source_alpha_ui_decision value;
    value.requested = value.rendered = true;
    value.mode = source_alpha_mode::automatic;
    value.qualification.available = true;
    value.coverage.state = alpha_auto_state::automatic_off;
    value.coverage.pixels = 1000;
    auto &evidence = value.coverage.evidence;
    evidence.candidates = 0x40 | 8;
    evidence.alpha_covered = {0, 0, 0, 1000}; evidence.layer_invalid = 600;
    check(value.unprotected() && value.layer_without_alpha(), "A layer without alpha beside full current alpha was protected");
    auto tolerated = value;
    tolerated.coverage.evidence.layer_invalid = 10;
    check(!tolerated.layer_without_alpha(), "A layer with 1% invalid pixels was reported without alpha");
    auto with_alpha = value; // Dead Space in SDR: a scene buffer whose luma-like alpha sits below its color.
    with_alpha.coverage.evidence.layer_covered = 900;
    check(with_alpha.unprotected() && !with_alpha.layer_without_alpha(), "An invalid layer with alpha was reported without alpha");
    auto unoffered = value;
    unoffered.coverage.evidence.candidates = 8;
    check(!unoffered.layer_without_alpha(), "A layer that was not offered was reported without alpha");
    auto searching = value;
    searching.qualification.available = false; searching.coverage.state = alpha_auto_state::waiting_for_source;
    check(searching.unprotected(), "No UI source offered was protected");
    auto clean = value;
    clean.coverage.evidence.layer_invalid = 0;
    check(!clean.unprotected() && !clean.layer_without_alpha(), "A clean empty UI layer (no UI on screen) was unprotected");
    auto tolerant = value;
    tolerant.coverage.evidence.layer_invalid = 1;
    check(tolerant.unprotected(), "An empty UI layer with invalid pixels showed that no UI was on screen");
    for (const std::uint32_t tag : {1u, 2u}) {
      auto declared = value;
      declared.coverage.evidence.candidates = tag | 8u;
      check(!declared.unprotected(), "A clean empty UIAlpha or UI color tag (no UI on screen) was unprotected");
      declared.coverage.evidence.alpha_invalid[tag - 1] = 1;
      check(declared.unprotected(), "An empty UIAlpha or UI color tag with invalid pixels showed that no UI was on screen");
    }
    auto presented = value;
    presented.coverage.evidence.candidates = 4u | 8u; presented.coverage.evidence.alpha_covered = {};
    check(presented.unprotected(), "Empty presented alpha showed that no UI was on screen");
    auto decided = value;
    decided.coverage.state = alpha_auto_state::automatic_on; decided.coverage.enabled = true;
    decided.coverage.evidence.alpha_covered = {};
    check(!decided.unprotected(), "A trusted channel deciding no UI was unprotected");
    auto checking = value;
    checking.coverage.state = alpha_auto_state::collecting;
    check(!checking.unprotected(), "Checking a source counted as unprotected");
    for (const auto mode : {source_alpha_mode::on, source_alpha_mode::off}) {
      auto manual = value; manual.mode = mode;
      check(!manual.unprotected(), "A manual mode counted as unprotected");
    }
    auto skipped = value; skipped.rendered = false;
    check(!skipped.unprotected(), "A Present that rendered nothing counted as unprotected");

    // The run starts with the first unprotected rendered frame, survives
    // Presents that rendered nothing or still collect their status sample, and
    // ends on a decided rendered frame, a manual mode, UI protection off or
    // Game 3D off.
    check(next_unprotected_since(value, true, 0, 5000) == 5000 && next_unprotected_since(value, true, 4000, 5000) == 4000,
      "An unprotected frame did not start or keep the run");
    check(next_unprotected_since(skipped, true, 4000, 5000) == 4000 && !next_unprotected_since(skipped, true, 0, 5000),
      "A Present that rendered nothing started or ended the run");
    auto searching_sample = value;
    searching_sample.coverage.state = alpha_auto_state::waiting_for_source;
    check(next_unprotected_since(checking, true, 4000, 5000) == 4000 && !next_unprotected_since(checking, true, 0, 5000) &&
        next_unprotected_since(searching_sample, true, 4000, 5000) == 4000,
      "A Present whose status sample was pending started or ended the run");
    check(!next_unprotected_since(decided, true, 4000, 5000) && !next_unprotected_since(clean, true, 4000, 5000),
      "A protected rendered frame did not end the run");
    auto off = skipped; off.mode = source_alpha_mode::off;
    auto unrequested = skipped; unrequested.requested = false;
    check(!next_unprotected_since(off, true, 4000, 5000) && !next_unprotected_since(unrequested, true, 4000, 5000),
      "A mode change did not end the run");
    check(!next_unprotected_since(skipped, false, 4000, 5000) && !next_unprotected_since(value, false, 4000, 5000),
      "Game 3D off kept the run");

    // Shown after alpha_trust_span_ms of the run, while the game shows 3D in Auto.
    render_settings settings;
    value.unprotected_since_ms = 10000;
    value.sdr_output = true;
    value.fg.known = true;
    const auto at = [&](std::uint64_t now, const source_alpha_ui_decision &decision, const render_settings &edited,
        automatic_phase phase = automatic_phase::ready) {
      return ui_protection_warning_for(decision, edited, phase, now);
    };
    check(!at(11999, value, settings).show && at(12000, value, settings).show, "The warning ignored the 2 s span");
    auto fresh = value; fresh.unprotected_since_ms = 0;
    check(!at(20000, fresh, settings).show, "A protected frame showed the warning");
    const auto sdr = at(12000, value, settings);
    check(sdr.try_hdr && sdr.try_fg, "SDR with a layer without alpha and FG off did not hint at HDR and FG");
    auto hdr = value; hdr.sdr_output = false; hdr.fg.enabled = true; hdr.retained_alpha_ready = true;
    const auto hdr_fg = at(12000, hdr, settings);
    check(hdr_fg.show && !hdr_fg.try_hdr && !hdr_fg.try_fg, "HDR with FG on hinted at HDR or FG");
    auto opaque_scene = value; // SDR without a layer (Dead Space): HDR would not help.
    opaque_scene.coverage.evidence.candidates = 8;
    check(!at(12000, opaque_scene, settings).try_hdr, "SDR without a layer hinted at HDR");
    auto scene_layer = value; // An SDR layer with alpha but V1-invalid (glow or a scene buffer): HDR would not help.
    scene_layer.coverage.evidence.layer_covered = 900;
    check(at(12000, scene_layer, settings).show && !at(12000, scene_layer, settings).try_hdr, "An invalid layer with alpha hinted at HDR");
    auto unknown_fg = value; unknown_fg.fg.known = false;
    check(!at(12000, unknown_fg, settings).try_fg, "Unknown FG hinted at FG");
    auto blocked = value;
    blocked.fg.enabled = true; blocked.retained_alpha_ready = false;
    check(blocked.blocked_by_fg() && !at(12000, blocked, settings).show, "A blocked FG capture showed this warning too");
    auto edited = settings;
    edited.depth_view = 1;
    check(!at(12000, value, edited).show, "Depth preview showed the warning");
    edited = settings; edited.strength = 0.f;
    check(!at(12000, value, edited).show, "2D at zero strength showed the warning");
    edited = settings; edited.enabled = false;
    check(!at(12000, value, edited).show, "Disabled Game 3D showed the warning");
    edited = settings; edited.ui_protection = source_alpha_mode::on;
    check(!at(12000, value, edited).show, "Manual On showed the warning");
    check(!at(12000, value, settings, automatic_phase::calibrating).show, "A 2D phase showed the warning");
    const std::string_view headline = ui_protection_warning_text;
    char span[32];
    std::snprintf(span, sizeof(span), "for %g s.", double(alpha_trust_span_ms) / 1000.0);
    check(headline.find("HUD") != std::string_view::npos && std::string_view(ui_protection_warning_tooltip()).find(span) != std::string_view::npos,
      "The warning text lost what the user sees or its span");
  }

  void source_alpha_applied_status_is_transient_and_mode_scoped() {
    alpha_auto_policy session;
    const auto generic = session.decision();
    check(generic.state == alpha_auto_state::waiting_for_source && !generic.enabled,
      "Status regression did not begin with an Auto session that detected nothing");
    source_alpha_ui_decision presented;
    presented.mode = source_alpha_mode::automatic;
    presented.automatic = presented.requested = presented.retained_alpha_ready = true;
    presented.fg = {true, true, false, 1, 0, 7, 41};
    presented.coverage = generic;
    presented.coverage.enabled = true;
    presented.coverage.state = alpha_auto_state::dedicated_ui;
    presented.rendered = presented.applied = true;
    for (const auto input : {source_alpha_input::sl_ui_color_alpha, source_alpha_input::sl_ui_alpha}) {
      presented.input = input;
      check(presented.dedicated_ui_active(source_alpha_mode::automatic, true),
        "An Auto session without detection hid an actually rendered dedicated UI mask");
    }
    const auto still_generic = session.decision();
    check(still_generic.state == alpha_auto_state::waiting_for_source && !still_generic.enabled,
      "Showing a dedicated mask changed the session's mode");

    check(!presented.applied_for(source_alpha_mode::off, true) &&
      !presented.dedicated_ui_active(source_alpha_mode::off, true) &&
      !presented.dedicated_ui_active(source_alpha_mode::automatic, false),
      "A previously rendered UI mask overrode a newly disabled setting");
    check(!presented.dedicated_ui_active(source_alpha_mode::on, true),
      "A mode edit reused the previous mode's rendered decision");
    presented.mode = source_alpha_mode::on;
    check(presented.applied_for(source_alpha_mode::on, true) &&
      !presented.dedicated_ui_active(source_alpha_mode::automatic, true),
      "Returning to Auto reused a manual On presentation");
    presented.mode = source_alpha_mode::automatic;

    auto next = presented;
    next.rendered = false;
    check(!next.applied_for(source_alpha_mode::automatic, true) &&
      !next.dedicated_ui_active(source_alpha_mode::automatic, true),
      "Available mask metadata reported protection after a failed render");
    next = presented;
    next.applied = false;
    next.coverage = generic;
    check(!next.dedicated_ui_active(source_alpha_mode::automatic, true),
      "Unapplied input inherited the previous frame's dedicated status");
    for (const auto input : {source_alpha_input::none, source_alpha_input::present_alpha, source_alpha_input::sl_backbuffer_alpha}) {
      next = presented;
      next.input = input;
      check(!next.dedicated_ui_active(source_alpha_mode::automatic, true),
        "A replacement generic input was mislabeled as a dedicated UI mask");
    }
    source_alpha_ui_decision missing;
    check(!missing.applied_for(source_alpha_mode::automatic, true) &&
      !missing.dedicated_ui_active(source_alpha_mode::automatic, true),
      "Missing presentation evidence reported active UI protection");
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
        automatic_phase::renderer_unavailable, automatic_phase::renderer_preparing}) {
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
    current.ui_midpoint_inverse = .75f;
    current.target_ui_midpoint_inverse = 1.;
    current.has_ui_midpoint = current.ui_midpoint_target_available = true;
    current.mean_square_inverse = .2;
    current.depth_pixel_count = 3072 * 2304;
    current.depth_tiles_x = 28; current.depth_tiles_y = 21;
    current.has_depth_statistics = true;
    auto shown = retain_automatic_scale({}, current);
    check(shown.depth_statistics_valid() && shown.reference_inverse == 2. && shown.mean_inverse == .3 &&
        shown.normalization == 4. && shown.minimum_inverse == 0. &&
        shown.has_zero_target() && shown.zero_inverse == .25f && shown.target_zero_inverse == 1. &&
        shown.has_ui_midpoint_target() && shown.ui_midpoint_inverse == .75f && shown.target_ui_midpoint_inverse == 1. &&
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
        shown.has_ui_midpoint && shown.ui_midpoint_inverse == .75f && !shown.has_ui_midpoint_target() &&
        !shown.ui_midpoint_target_available && shown.target_ui_midpoint_inverse == 0. &&
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
        !shown.has_ui_midpoint && shown.ui_midpoint_inverse == 0.f && !shown.has_ui_midpoint_target() &&
        !shown.ui_midpoint_target_available && shown.target_ui_midpoint_inverse == 0. &&
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
    source_alpha_modes_are_persistent_and_game_scoped();
    source_alpha_mode_migration_and_precedence();
    source_alpha_mode_persistence_failure_does_not_commit();
    source_alpha_mode_edits_control_the_session();
    source_alpha_fg_mode_survives_observation_gaps();
    source_alpha_selection_reports_capture_prerequisite();
    ui_protection_warning_follows_unprotected_rendered_frames();
    source_alpha_applied_status_is_transient_and_mode_scoped();
    persistence_survives_runtime_recreation();
    config_failure_and_lifetime_do_not_commit_stale_values();
    output_status_distinguishes_flat_preview_and_stereo();
    automatic_scale_tracks_applied_basis();
    measured_statistics_follow_applied_and_held_scale_ownership();
    positive_flat_zero_target_does_not_invent_gain_tracking();
    projection_conversion_is_independent_and_retained_with_its_reference();
    std::puts("PASS native Game 3D controls: defaults, automatic persistence, independent resets, runtime lifetime, output status, UI protection warning and scale/conversion");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
