// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_controls.h"
#include "game3d_diagnostics.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace sunshine_game3d {
  inline constexpr const char *config_section = "SUNSHINE_GAME3D";
  enum class control : unsigned { strength, depth_view, count };

  struct settings_state {
    render_settings values;
    bool alive = true;
    std::shared_ptr<alpha_auto_policy> alpha_session{};
  };

  template<class Backend>
  render_settings load_settings(Backend &config) {
    render_settings result;
    config.read("Strength", result.strength);
    config.read("DepthView", result.depth_view);
    config.read("Enabled", result.enabled);
    // The old checkbox enabled heuristics, so legacy true migrates to Auto.
    // Preserve an explicitly disabled old checkbox until a three-way edit.
    bool legacy_alpha = true;
    config.read("SourceAlphaUI", legacy_alpha);
    int alpha_mode = legacy_alpha ? 0 : 2;
    config.read("SourceAlphaUIMode", alpha_mode);
    result.ui_protection = alpha_mode >= 0 && alpha_mode <= 2 ? source_alpha_mode(alpha_mode) : source_alpha_mode::automatic;
    result.source_alpha_ui = result.ui_protection == source_alpha_mode::on;
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

  template<class Backend>
  bool edit_source_alpha_mode(settings_state &settings, source_alpha_mode value, Backend &config) {
    if (!settings.alive || int(value) < 0 || int(value) > 2 || value == settings.values.ui_protection) return false;
    config.write("SourceAlphaUIMode", int(value));
    if (!settings.alive) return false;
    settings.values.ui_protection = value;
    settings.values.source_alpha_ui = value == source_alpha_mode::on;
    if (settings.alpha_session) {
      if (value == source_alpha_mode::automatic) settings.alpha_session->set_automatic();
      else settings.alpha_session->set_manual(value == source_alpha_mode::on);
    }
    return true;
  }

  inline const char *source_alpha_capture_block_text(source_alpha_capture_block blocked) {
    switch (blocked) {
      case source_alpha_capture_block::generated_current_color:
        return "Current color alpha is unavailable during Frame Generation. Choose Discover automatically to use captured game inputs.";
      default: return nullptr;
    }
  }

  // Forget learned UI sources (the overlay's Forget, A3): clears the UI
  // sources Auto accepted for this game, in this session and, through the
  // session's change listener, in TrustedUISources. Returns whether anything
  // was accepted; `cleared` receives the keys for the caller's log line.
  inline bool forget_learned_ui_sources(settings_state &settings, std::string *cleared = nullptr) {
    if (!settings.alive || !settings.alpha_session) return false;
    auto keys = settings.alpha_session->forget();
    const bool any = !keys.empty();
    if (cleared) *cleared = std::move(keys);
    return any;
  }

  // The first-run shadow of hidden-scene evidence (UISceneShadow) was
  // removed: a ReShade.ini that still carries the key loads unchanged and it
  // is ignored, like UIPinOnlyUI and UIFlattenStillScreens below.

  // Rule H2 (still screens without a UI source) was removed: a ReShade.ini
  // that still carries UIFlattenStillScreens loads unchanged and the key is
  // ignored, like UIPinOnlyUI.
  // The add-on's Diagnostics switch (game3d_diagnostics.h; docs/reshade-sbs.md,
  // Diagnostics switch): per game in the global ReShade.ini, process-wide.
  // Absent writes 0 so the key is discoverable; 1 turns it on, anything else
  // keeps it off. Loading sets the process switch.
  template<class Backend>
  bool load_diagnostics(Backend &config) {
    int value = -1;
    config.read(diagnostics::key, value);
    if (value == -1) config.write(diagnostics::key, 0);
    diagnostics::set_enabled(value == 1);
    return value == 1;
  }
  // The panel's checkbox: saves the key, then switches the process.
  template<class Backend>
  bool edit_diagnostics(bool value, Backend &config) {
    if (value == diagnostics::enabled()) return false;
    config.write(diagnostics::key, value ? 1 : 0);
    diagnostics::set_enabled(value);
    return true;
  }
  // The log line, at load and on every edit.
  inline std::string diagnostics_log_text(bool enabled) {
    return enabled ? "Sunshine Game 3D: diagnostics on (Diagnostics=1)" : "Sunshine Game 3D: diagnostics off (Diagnostics=0)";
  }

  // The overlay name of a candidate bit (ui_detection::candidate), empty for
  // none.
  inline const char *ui_source_name(std::uint32_t candidate_bit) {
    namespace candidate = ui_detection::candidate;
    switch (candidate_bit) {
      case candidate::ui_alpha: return "UI alpha tag";
      case candidate::ui_color: return "UI color tag";
      case candidate::layer: return "UI layer";
      case candidate::backbuffer: return "real-input alpha";
      case candidate::current: return "current color alpha";
      case candidate::hudless: return "HUD-less difference";
      default: return "";
    }
  }

  // F1: why a sample decided no UI mask, from its own decision's reason and
  // refused candidate (decision texel 9); empty when it decided a source or
  // carries no reason.
  inline std::string ui_protection_reason_text(const alpha_auto_decision &decision) {
    const auto &evidence = decision.evidence;
    if (decision.source_kind) return {};
    if (evidence.frame_reason >= ui_no_mask::count) return {};
    const std::string source = ui_source_name(evidence.refused);
    const auto the = [&](const char *fallback) { return "the " + (source.empty() ? std::string(fallback) : source); };
    switch (evidence.frame_reason) {
      case ui_no_mask::unaccepted: return "learning " + the("UI source");
      case ui_no_mask::ambiguous: return the("UI source") + " shows no UI or the whole frame";
      case ui_no_mask::trusted_invalid: return the("UI source") + " is unusable in this frame";
      case ui_no_mask::presented_blocked: return "the UI tag is unusable in this frame and holds back " + the("inferred alpha");
      case ui_no_mask::layer_aside: return "the UI layer holds no UI alpha";
      case ui_no_mask::difference_failed: return "the HUD-less difference does not isolate UI";
      case ui_no_mask::gate_no_hold: return "checking whether a full-screen menu hides the scene";
      case ui_no_mask::no_candidate: return "no UI source offered";
      default: return "no UI source qualifies";
    }
  }

  inline std::string source_alpha_detection_text(const ui_qualification::status &source, const source_alpha_ui_decision &decision) {
    if (const auto blocked = source_alpha_capture_block_text(source_alpha_capture_block_for(source.selected, decision.fg))) return blocked;
    if (!source.available) return "Finding a usable UI source; protection off";
    // A source edit can happen before the next rendered decision. Never label
    // the replacement using quality evidence from the previous source.
    if (source.selected != decision.qualification.selected || source.candidate != decision.qualification.candidate)
      return "Checking source quality";
    if (decision.coverage.enabled) return "UI source detected automatically";
    if (decision.coverage.state == alpha_auto_state::automatic_off) {
      const auto reason = ui_protection_reason_text(decision.coverage);
      return reason.empty() ? std::string("No usable UI mask detected") : "No usable UI mask (" + reason + ")";
    }
    return "Checking source quality";
  }

  // The Status tab's UI protection warning (docs/reshade-sbs.md, UI
  // protection): Auto has rendered without a UI mask for alpha_trust_span_ms,
  // the span over which detection itself earns or loses confidence, while the
  // game shows 3D. Hints name the game settings that commonly provide a mask,
  // only where the evidence fits them: HDR when the SDR frame's offscreen UI
  // layer had color but no alpha, Frame Generation when the game reports it
  // off. A blocked FG capture has its own text.
  struct ui_protection_warning {
    bool show = false, try_hdr = false, try_fg = false;
  };
  inline constexpr const char *ui_protection_warning_text = "No UI mask in this game mode: HUD and menus take the scene's depth.";
  inline constexpr const char *ui_protection_hdr_hint = "Try HDR output: some games draw their UI on a separate layer only in HDR.";
  inline constexpr const char *ui_protection_fg_hint = "Try Frame Generation: some games provide their UI buffers only while it is on.";
  // The tooltip states the span from alpha_trust_span_ms, its one owner.
  inline const char *ui_protection_warning_tooltip() {
    static const std::string text = [] {
      char span[32];
      std::snprintf(span, sizeof(span), "%g", double(alpha_trust_span_ms) / 1000.0);
      return std::string("Sunshine keeps UI flat only with a usable UI mask from the game: a Streamline UI buffer, an "
                         "offscreen UI layer or alpha that marks UI. None has been usable for ") +
        span + " s. Protection resumes by itself when one appears. Sunshine does not change game settings.";
    }();
    return text.c_str();
  }
  inline ui_protection_warning ui_protection_warning_for(const source_alpha_ui_decision &decision, const render_settings &settings,
      automatic_phase phase, std::uint64_t now_ms) {
    ui_protection_warning result;
    result.show = settings.enabled && settings.ui_protection == source_alpha_mode::automatic &&
      decision.mode == source_alpha_mode::automatic && phase == automatic_phase::ready && settings.depth_view == 0 &&
      std::isfinite(settings.strength) && settings.strength > 0.f && !decision.blocked_by_fg() &&
      decision.unprotected_since_ms && now_ms >= decision.unprotected_since_ms &&
      now_ms - decision.unprotected_since_ms >= alpha_trust_span_ms;
    result.try_hdr = result.show && decision.sdr_output && decision.layer_without_alpha();
    result.try_fg = result.show && decision.fg.known && !decision.fg.enabled;
    return result;
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
      case automatic_phase::renderer_preparing: return "2D: preparing Game 3D shaders";
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
