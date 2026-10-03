// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_alpha_auto.h"
#include "game3d_ui_qualification.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace reshade::api {
  struct effect_runtime;
}

namespace sunshine_game3d {
  inline constexpr float default_strength = 50.f;
  enum class source_alpha_mode : int { automatic = 0, on = 1, off = 2 };
  struct render_settings {
    float strength = default_strength;
    int depth_view = 0;
    bool enabled = true;
    source_alpha_mode ui_protection = source_alpha_mode::automatic;
    bool source_alpha_ui = false;
  };

  // Present alpha is a game-specific convention, not an FG output contract.
  // Generated output alpha is never a mask. FG requires a retained, completed
  // real-input alpha copy. Automatic quality checks are separate from availability;
  // explicit replay still honors white masks.
  inline bool source_alpha_ui_for_present(bool requested, bool frame_generation_active, bool retained_alpha_ready = false) {
    return requested && (!frame_generation_active || retained_alpha_ready);
  }

  // Game-requested FG mode, independent of which depth source is selected.
  // This does not classify the current presentation as real or generated.
  struct frame_generation_mode {
    bool known = false, enabled = false, automatic = false;
    std::uint32_t generated_frames = 0, viewport = 0;
    std::uint64_t epoch = 0, sequence = 0;
  };
  enum class source_alpha_capture_block { none, generated_current_color };
  inline source_alpha_capture_block source_alpha_capture_block_for(ui_qualification::choice selected,
      const frame_generation_mode &fg) {
    using choice = ui_qualification::choice;
    return selected == choice::current_color && fg.known && fg.enabled ?
      source_alpha_capture_block::generated_current_color : source_alpha_capture_block::none;
  }
  inline const char *name(source_alpha_capture_block value) {
    switch (value) {
      case source_alpha_capture_block::generated_current_color: return "generated_current_color";
      default: return "none";
    }
  }
  enum class source_alpha_input_state { not_observed, seen, rejected, state_conflict };
  enum class source_alpha_input { none, present_alpha, sl_backbuffer_alpha, sl_ui_color_alpha, sl_ui_alpha, sl_hudless_difference, automatic_mask };
  inline const char *name(source_alpha_input value) {
    switch (value) {
      case source_alpha_input::present_alpha: return "present_alpha";
      case source_alpha_input::sl_backbuffer_alpha: return "sl_backbuffer_alpha";
      case source_alpha_input::sl_ui_color_alpha: return "sl_ui_color_alpha";
      case source_alpha_input::sl_ui_alpha: return "sl_ui_alpha";
      case source_alpha_input::sl_hudless_difference: return "sl_hudless_difference";
      case source_alpha_input::automatic_mask: return "automatic_gpu_mask";
      default: return "none";
    }
  }
  inline const char *name(source_alpha_input_state value) {
    switch (value) {
      case source_alpha_input_state::seen: return "input_seen";
      case source_alpha_input_state::rejected: return "capture_rejected";
      case source_alpha_input_state::state_conflict: return "resource_state_conflict";
      default: return "not_observed";
    }
  }
  struct source_alpha_ui_decision {
    bool requested = false;
    frame_generation_mode fg;
    bool retained_alpha_ready = false;
    // Latest input attempt observed at this render; it is not the identity of
    // a retained mask, and failure does not revoke an otherwise usable copy.
    source_alpha_input_state input_state = source_alpha_input_state::not_observed;
    bool automatic = false;
    alpha_auto_decision coverage;
    ui_qualification::status qualification;
    // Presentation telemetry, independent of source quality and manual mode.
    // Early exits leave rendered/applied false; available input alone is not On.
    source_alpha_mode mode = source_alpha_mode::automatic;
    source_alpha_input input = source_alpha_input::none;
    bool rendered = false, applied = false;
    // The swapchain presents SDR (sRGB), for the overlay's hints only.
    bool sdr_output = false;
    // Since when (GetTickCount64) consecutive rendered frames were unprotected;
    // zero while protected. The exporter owns it (next_unprotected_since).
    std::uint64_t unprotected_since_ms = 0;
    bool fg_active() const { return fg.known && fg.enabled; }
    bool effective() const { return source_alpha_ui_for_present(requested, fg_active(), retained_alpha_ready) && (!automatic || coverage.enabled); }
    bool blocked_by_fg() const { return requested && fg_active() && !retained_alpha_ready; }
    bool applied_for(source_alpha_mode current_mode, bool game3d_enabled) const {
      return game3d_enabled && current_mode != source_alpha_mode::off && current_mode == mode && rendered && applied;
    }
    bool dedicated_ui_active(source_alpha_mode current_mode, bool game3d_enabled) const {
      return applied_for(current_mode, game3d_enabled) &&
        (input == source_alpha_input::sl_ui_color_alpha || input == source_alpha_input::sl_ui_alpha);
    }
    const char *source_availability() const {
      const auto blocked = source_alpha_capture_block_for(qualification.selected, fg);
      if (blocked != source_alpha_capture_block::none) return name(blocked);
      if (!qualification.available) return "source_unavailable";
      if (coverage.enabled) return "detected";
      return coverage.state == alpha_auto_state::automatic_off ? "quality_rejected" : "checking_quality";
    }
    // Auto rendered this frame without a UI mask while UI may be on screen: no
    // source decided (coverage.enabled), and either no UI source was offered
    // or no dedicated UI candidate (UIAlpha, the UI color tag or the
    // offscreen UI layer) was offered clean and empty, which would mean no UI
    // on screen. An accepted candidate deciding empty is detected, not
    // unprotected. tools/reshade/game3d_log_report.py mirrors this definition
    // (docs/reshade-sbs.md, UI protection).
    bool unprotected() const {
      if (mode != source_alpha_mode::automatic || !requested || !rendered || coverage.enabled) return false;
      if (!qualification.available) return true;
      if (coverage.state != alpha_auto_state::automatic_off) return false;
      const auto &evidence = coverage.evidence;
      using ui_selection::kind;
      for (const auto k : {kind::ui_alpha, kind::ui_color, kind::ui_layer}) {
        const auto counts = alpha_counts_of(evidence, k);
        if ((evidence.candidates & ui_selection::bit(k)) && !counts.covered && !counts.invalid) return false;
      }
      return true;
    }
    // The latest sample's offscreen UI layer had color but no alpha: no alpha
    // anywhere and more than 1% of its pixels beyond the premultiplied bound.
    // V1 then rejects it, so it neither decides nor blocks; in SDR, Stellar
    // Blade draws its scene image there. A V1-invalid layer with some alpha
    // (a scene buffer's luma-like alpha, glow beyond the bound) is not this:
    // HDR output would not change it.
    bool layer_without_alpha() const {
      const auto &evidence = coverage.evidence;
      return (evidence.candidates & ui_detection::candidate::layer) && !evidence.layer_covered &&
        !ui_selection::alpha_valid(evidence.layer_invalid, coverage.pixels);
    }
  };
  // The start of the current unprotected run after publishing `value`: kept
  // while unprotected, cleared by a decided rendered frame (a mask, or a clean
  // empty UI channel showing no UI), a manual mode, UI protection off or Game
  // 3D off. Neither a Present that rendered nothing nor one whose status
  // sample is still pending (collecting after a change of the winning source,
  // revision or viewport) starts or ends the run, so depth gaps and sample churn do
  // not restart it.
  inline std::uint64_t next_unprotected_since(const source_alpha_ui_decision &value, bool game3d_enabled,
      std::uint64_t previous, std::uint64_t now_ms) {
    if (!game3d_enabled || value.mode != source_alpha_mode::automatic || !value.requested) return 0;
    if (value.unprotected()) return previous ? previous : std::max<std::uint64_t>(now_ms, 1);
    if (!value.rendered) return previous;
    return value.coverage.enabled || value.coverage.state == alpha_auto_state::automatic_off ? 0 : previous;
  }
  struct source_alpha_ui_policy {
    source_alpha_ui_decision update(bool requested, frame_generation_mode observed, bool observer_active) {
      // Contention or ambiguous/lost observations cannot turn a known FG mode
      // off. Only an observed mode change or genuine observer/runtime shutdown
      // clears it. In particular, manually pinning Generic depth is not FG Off.
      if (!observer_active) mode = {};
      else if (observed.known) mode = observed;
      source_alpha_ui_decision result;
      result.requested = requested;
      result.fg = mode;
      return result;
    }
  private:
    frame_generation_mode mode;
  };

  // The same frozen presentation decision consumed by rendering and Dump 3D.
  source_alpha_ui_decision query_source_alpha_ui(reshade::api::effect_runtime *runtime);

  // Add-on-owned settings, independent of effects, presets and Performance Mode.
  // Edits persist in ReShade configuration; UI protection uses the per-game
  // global file so all of the game's runtimes share the saved choice.
  render_settings query_render_settings(reshade::api::effect_runtime *runtime);
  // The game process's UI protection mode and alpha trust. Live Auto checks
  // source quality independently on the GPU.
  alpha_auto_policy &source_alpha_session();

  enum class automatic_phase { unavailable, waiting_for_depth, calibrating, ready, suspended, unsupported_resolution, renderer_unavailable, renderer_preparing };
  enum class automatic_scale_basis { unknown, camera_matrix, relative_depth, linear_distance };
  enum class automatic_scale_state { unavailable, pending, active, held };
  struct automatic_scale {
    automatic_scale_basis basis = automatic_scale_basis::unknown;
    float value = 0.f;
    bool active = false;
    float target_value = 0.f;
    double conversion_multiplier = 0.0;
    double conversion_offset = 0.0;
    bool has_projection_conversion = false;
    float zero_inverse = 0.f;
    bool has_zero = false, gain_below_target = false;
    // Accepted nearest reference Q and far/near extrema in zero_inverse's
    // decoded inverse-depth basis. These are measurements, not clipping planes.
    double reference_inverse{}, minimum_inverse{}, maximum_inverse{};
    double mean_square_inverse{};
    bool has_centered_depth_statistics = false;
    double mean_contrast_inverse{}, mean_contrast_square_inverse{};
    std::uint64_t depth_pixel_count{};
    std::uint32_t depth_tiles_x{}, depth_tiles_y{};
    bool has_depth_statistics = false;
    // Raw moments remain diagnostics; stable centered moments define zero.
    // Captured L is dimensionless and sets target K=L/Q for this output shape.
    double mean_inverse{}, normalization{};
    double target_zero_inverse{};
    bool zero_target_available = false;
    float ui_midpoint_inverse{};
    double target_ui_midpoint_inverse{};
    bool has_ui_midpoint = false, ui_midpoint_target_available = false;

    bool has_value() const {
      return basis != automatic_scale_basis::unknown && std::isfinite(value) && value > 0.f;
    }
    bool has_target() const {
      return active && has_value() && std::isfinite(target_value) && target_value > 0.f;
    }
    bool has_zero_target() const {
      return active && has_value() && zero_target_available && depth_statistics_valid() &&
        std::isfinite(target_zero_inverse) && target_zero_inverse >= 0.;
    }
    bool has_ui_midpoint_target() const {
      return active && has_value() && ui_midpoint_target_available && depth_statistics_valid() &&
        std::isfinite(target_ui_midpoint_inverse) && target_ui_midpoint_inverse >= 0.;
    }
    bool projection_conversion_valid() const {
      return basis == automatic_scale_basis::camera_matrix && has_projection_conversion &&
        std::isfinite(conversion_multiplier) && conversion_multiplier != 0.0 && std::isfinite(conversion_offset);
    }
    bool depth_statistics_valid() const {
      return basis != automatic_scale_basis::unknown && has_depth_statistics &&
        std::isfinite(reference_inverse) && std::isfinite(minimum_inverse) && std::isfinite(maximum_inverse) &&
        minimum_inverse >= 0.0 && maximum_inverse >= minimum_inverse &&
        reference_inverse == maximum_inverse &&
        std::isfinite(mean_inverse) && mean_inverse >= minimum_inverse && mean_inverse <= maximum_inverse &&
        std::isfinite(normalization) && normalization > 0. &&
        (!depth_pixel_count || (depth_tiles_x && depth_tiles_y &&
          std::isfinite(mean_square_inverse) && mean_square_inverse >= 0.));
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
      previous.target_zero_inverse = 0.;
      previous.zero_target_available = false;
      previous.target_ui_midpoint_inverse = 0.;
      previous.ui_midpoint_target_available = false;
      return previous;
    }
    // Camera distance and raw coordinates are different representations.
    // Never relabel held values.
    // Valid current projection conversion may be known before scene centering.
    current.value = current.target_value = 0.f;
    current.active = false;
    current.reference_inverse = current.minimum_inverse = current.maximum_inverse = 0.0;
    current.mean_square_inverse = 0.;
    current.has_centered_depth_statistics = false;
    current.mean_contrast_inverse = current.mean_contrast_square_inverse = 0.;
    current.mean_inverse = current.normalization = 0.;
    current.target_zero_inverse = 0.;
    current.zero_target_available = false;
    current.ui_midpoint_inverse = 0.f;
    current.target_ui_midpoint_inverse = 0.;
    current.has_ui_midpoint = current.ui_midpoint_target_available = false;
    current.depth_pixel_count = current.depth_tiles_x = current.depth_tiles_y = 0;
    current.has_depth_statistics = false;
    return current;
  }

  struct automatic_status {
    automatic_phase phase = automatic_phase::unavailable;
    // Capability for test-only reset adapters; the product has no manual reset.
    bool can_recalibrate = false;
    automatic_scale scale;
  };

  // Exporter-owned policy state. Values contain no borrowed GPU objects.
  automatic_status query_automatic(reshade::api::effect_runtime *runtime);
#if defined(SUNSHINE_SBS_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
  bool recalibrate_automatic(reshade::api::effect_runtime *runtime);
#endif

  void initialize();
  void shutdown();
  // False means a synchronous edit invalidated the runtime snapshot. Stop
  // drawing before accessing any earlier runtime state.
  bool draw(reshade::api::effect_runtime *runtime);
  // Read-only tabs never insert controls or change the primary layout.
  void draw_status(reshade::api::effect_runtime *runtime);
  void draw_calibration(reshade::api::effect_runtime *runtime);
  // Preview and source overrides, ahead of live diagnostics. Does not request
  // candidate enumeration; the depth owner handles its manual buffer list.
  bool draw_diagnostics(reshade::api::effect_runtime *runtime);
}  // namespace sunshine_game3d
