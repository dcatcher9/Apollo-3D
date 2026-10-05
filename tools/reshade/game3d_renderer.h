// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "game3d_stereo_contract.h"
#include "game3d_ui_plane.h"
#include "game3d_alpha_auto.h"
#include "game3d_ui_adaptive.h"
#include <windows.h>
#include <reshade_api.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>

namespace sunshine_game3d {
  // Byte-for-byte layout of game3d_native.hlsl b0. No ReShade FX uniforms.
  struct render_parameters {
    float strength = 50.f;
    std::int32_t depth_view = 0;
    std::uint32_t depth_ready = 0, camera_ready = 0;
    std::int32_t coordinate_basis = 0;
    float depth_scale = 0.f, strength_blend = 0.f;
    float disparity_limit_uv = default_disparity_limit_uv;
    std::array<float, 2> projection{}, raw_depth_range{0.f, 1.f};
    std::array<float, 2> convergence{}, jitter{};
    std::array<float, 4> depth_rect{0.f, 0.f, 1.f, 1.f};
  };
  static_assert(sizeof(render_parameters) == 80);
  static_assert(offsetof(render_parameters, disparity_limit_uv) == 28);

  // Semantic input contract. API versions, feature IDs and generated-frame
  // capture/reuse rules belong to providers, not stereo rendering. An absent
  // captured view is unavailable; it must never fall back to current color.
  enum class ui_input_kind { unavailable, current_color_alpha, captured_color_alpha, dedicated_mask, hudless_difference };
  struct ui_detection_inputs {
    // Provider-admitted, current native captures, each candidate in its own
    // slot (UI framework E1): masks are the UIAlpha tag (R), the tagged
    // UIColorAndAlpha (A, never the offscreen UI layer) and the Backbuffer tag
    // (A); layer is the offscreen UI layer copy (A). Selection among them is
    // the shader's (ui_selection::decide), not this order.
    std::array<reshade::api::resource_view, 3> masks{}; // UI alpha R, UI color tag A, backbuffer A.
    reshade::api::resource_view layer{};
    // Stored Sunshine_UIDetectionFlags of the layer slot: ui_layer::detection_flags
    // of the offered layer (game3d_ui_detection_contract.h).
    std::uint32_t layer_flags = 0;
    // The acceptance signature of each offered kind (A1): its typed DXGI format
    // and the swapchain color space. A kind left at format zero, or a zero color
    // space, takes the renderer's own view format and color space.
    candidate_signatures signatures;
    reshade::api::resource_view hudless{};
    bool current_color = false;
    // A generated Present by Present counting (UI framework T1): one that
    // offers nothing within the reported generated count of the last Present
    // that offered a UI tag (ui_mask::generated_without_input). It never
    // detects: it shows the last real decision, or has no mask
    // (ui_temporal::detection_state::arbitrate).
    bool hold_previous = false;
    // The game's own final color from the HUD-less image's tag batch (Streamline
    // Backbuffer). When present, HUD-less is compared with it exactly, and
    // hudless_presents_ago is not used.
    reshade::api::resource_view hudless_pair{};
    // Otherwise counting proposes the color presented this many Presents ago
    // (ui_mask::pair_hudless_present): the retained color, or the current one
    // for zero. Such a pair is inexact; only V2 validates its pixels.
    std::uint32_t hudless_presents_ago = 0;
    static constexpr std::uint32_t max_retained_presents = 2;
    // The pair is known to show one game frame: a same-batch Backbuffer
    // (E2). Only then may a difference covering the whole frame mean
    // full-screen UI rather than a mismatched pair.
    bool hudless_exact = false;
  };
  struct ui_render_input {
    ui_input_kind kind = ui_input_kind::unavailable;
    reshade::api::resource_view view{};
    ui_plane_parameters plane;
    const alpha_auto_source *automatic{};
    const ui_adaptive::source *adaptive{};
    ui_mask_channel channel = ui_mask_channel::alpha;
    const ui_detection_inputs *detection{};
    bool available() const {
      return kind == ui_input_kind::current_color_alpha ||
        ((kind == ui_input_kind::captured_color_alpha || kind == ui_input_kind::dedicated_mask ||
          kind == ui_input_kind::hudless_difference) && view.handle) ||
        (detection && (detection->current_color || detection->hudless.handle || detection->hold_previous ||
          detection->masks[0].handle || detection->masks[1].handle || detection->masks[2].handle ||
          detection->layer.handle));
    }
  };
  struct render_frame_input {
    reshade::api::resource color{};
    reshade::api::resource_view depth{};
    // Resolved depth encoding, paired camera conversion, jitter and stereo
    // controls; readiness belongs to this frame, never to an SDK capability.
    render_parameters scene;
    ui_render_input ui;
    // The depth was captured for this frame rather than reused from an
    // earlier one. Hidden-scene evidence is valid only for current depth.
    bool depth_current = true;
    // Dump 3D is armed for this Present: a pack recorded for it also records
    // the full conditioning, even where the pack would show the source mono,
    // so that the dump's diagnostic textures describe this render.
    bool diagnostic_armed = false;
    // Nonzero only when the depth view's content is immutable for this value
    // (the provider's capture epoch, source and sequence). Equal values with
    // the same view and parameters let the renderer keep its depth candidate
    // and vertical field from the last recorded conditioning; zero (unknown)
    // always records them.
    std::uint64_t depth_identity = 0;
  };

  // Current positive display bound after strength and blend. Scene admission
  // is a separate concern; keep this arithmetic shared by UI observation and
  // export, in SunshineBoundFinalParallax's multiplication order.
  inline float display_parallax_cap_uv(const render_parameters &p) {
    if (!std::isfinite(p.strength) || !std::isfinite(p.strength_blend) ||
        !std::isfinite(p.disparity_limit_uv) || p.disparity_limit_uv <= 0.f || p.disparity_limit_uv > .04f)
      return 0.f;
    float cap = p.disparity_limit_uv * std::clamp(p.strength, 0.f, 100.f);
    cap *= .01f;
    cap *= std::clamp(p.strength_blend, 0.f, 1.f);
    return cap;
  }

  // Export-only conversion of an already admitted live scene and its consumed
  // UI fraction. Scene admission remains owned by the scene controller; this
  // does not infer depth, choose a plane, or advance an adaptive policy. The
  // multiplication order matches SunshineBoundFinalParallax in the shader.
  // Diagnostic views and unsupported dimensions have no shifted game UI.
  inline float admitted_ui_parallax_uv(const render_parameters &p, const ui_plane_parameters &plane,
      bool scene_admitted, bool ui_enabled, std::uint32_t width, std::uint32_t height) {
    if (!scene_admitted || !ui_enabled || !width || !height || width > 3840 || height > 3840 ||
        !p.depth_ready || !p.camera_ready || p.depth_view != 0 || plane.mode != ui_plane_mode::display_fraction ||
        !std::isfinite(plane.inverse_depth) || plane.inverse_depth < 0.f || plane.inverse_depth > .75f)
      return 0.f;
    return plane.inverse_depth * display_parallax_cap_uv(p);
  }

  // Borrowed only during the current render lease. Diagnostic copies must be
  // recorded before the next render reuses these textures. The conditioning
  // textures (linear color, candidate, vertical, final field, nearest UI
  // plane) hold the last RECORDED conditioning: a render whose pack is owed
  // records it with that pack, and an owed pack that shows the source mono
  // records none unless Dump 3D is armed (render_frame_input::diagnostic_armed).
  struct diagnostic_resources {
    reshade::api::resource source{}, linear_color{}, candidate{}, vertical_majorant{},
      vertical_field{}, final_field{}, sbs{}, ui_source{}, ui_plane_tiles{}, ui_plane_resolved{};
    // sbs holds 10-bit PQ code values (renderer::set_pq_output: Rec.2020,
    // ST 2084, R10G10B10A2_UNORM) instead of FP16 scRGB.
    bool sbs_pq{};
  };

  // GPU time of the live render stages over a reporting window, in ms. Inputs
  // covers work the caller records after begin_gpu_profile (depth and UI
  // captures); pack is zero on frames nothing consumed. Recorded only while
  // the add-on's Diagnostics switch is on (game3d_diagnostics.h).
  struct gpu_timing {
    // Conditioning is split into PQ linearization, depth candidate, and the
    // vertical and horizontal scans (the latter includes UI placement passes).
    // A pack records the conditioning it owes first, so these stages time the
    // conditioning itself wherever it was recorded; they are zero on frames
    // that recorded none (no pack, or a pack that shows the source mono).
    enum stage { inputs, source, detection, linearize, candidate, vertical, horizontal, eyes, pack, total, stage_count };
    std::uint32_t frames{};
    std::array<double, stage_count> mean_ms{}, max_ms{};
    // Why marked frames did not count. The profile may never have started,
    // or the Diagnostics switch is off (disabled: no timestamps are recorded);
    // a frame's slot can be reused before its completion fence passed, or
    // after it passed with results never readable (a resolve copy that had not
    // executed, or, on D3D11, results ReShade did not return); or a read frame
    // lacked its render mark.
    enum class profile_state : std::uint8_t { not_started, ready, no_timestamp_frequency, no_query_heap, disabled };
    profile_state state = profile_state::not_started;
    std::uint32_t dropped_fence_pending{}, dropped_unresolved{}, incomplete{};
  };
  inline const char *name(gpu_timing::profile_state value) {
    switch (value) {
      case gpu_timing::profile_state::ready: return "ready";
      case gpu_timing::profile_state::no_timestamp_frequency: return "no_timestamp_frequency";
      case gpu_timing::profile_state::no_query_heap: return "no_query_heap";
      case gpu_timing::profile_state::disabled: return "disabled";
      default: return "not_started";
    }
  }

  // The UI detection constants (b2) behind the mask a render consumed, for
  // Dump 3D replay. A held mask (a generated Present showing a real frame's
  // decision, T1) reports the run that made it.
  struct ui_detection_snapshot {
    enum class run_state : std::uint8_t { inactive, ran, held };
    run_state state = run_state::inactive;
    // accepted: the pushed Sunshine_UIAcceptedCandidates (b2 word 2), the
    // accepted candidates in candidate-bit positions.
    std::uint32_t candidates{}, threshold_bits{}, accepted{};
    // flags is the full pushed Sunshine_UIDetectionFlags word; stored_flags is
    // the offscreen UI layer slot's stored part, never a per-frame bit.
    std::uint32_t flags{}, stored_flags{};
    std::uint32_t held_presents{}; // Consecutive generated Presents that applied the mask.
    // b2 word 4 (Sunshine_UIPreUIThreshold, float32 bits): the offscreen UI
    // layer's pair threshold with the presented color, zero without a layer,
    // when the two are not comparable or on a frame that is not a detection
    // sample (H1 d).
    std::uint32_t pre_ui_threshold_bits{};
    // b2 word 5 is reserved and pushed as zero (ui_detection::b2_words).
  };
  inline const char *name(ui_detection_snapshot::run_state value) {
    switch (value) {
      case ui_detection_snapshot::run_state::ran: return "ran";
      case ui_detection_snapshot::run_state::held: return "held";
      default: return "inactive";
    }
  }

  struct alpha_probe_counters {
    std::uint64_t submitted = 0, mapped = 0; // Recorded probes and native Map calls.
    std::uint64_t scene_evidence = 0; // Recorded hidden-scene evidence passes (one per evaluated sample).
  };

  // One runtime owns its GPU working set. All passes run on ReShade's graphics
  // queue; borrowed depth is consumed entirely within the owner's read lease.
  class renderer {
  public:
    renderer();
    ~renderer();
    renderer(const renderer &) = delete;
    renderer &operator=(const renderer &) = delete;
    // Background preparation (production shader only) compiles missing
    // entry points on the thread pool and returns false while they are
    // pending; preparing() then distinguishes that from an unusable mode.
    bool configure(reshade::api::effect_runtime *runtime, reshade::api::resource backbuffer,
      reshade::api::color_space color, std::string_view source_override = {}, bool prepare_in_background = false);
    bool preparing() const { return preparing_; }
    static std::string_view shader_source();
    std::string_view active_shader_source() const;
    // Mode-5 live adaptation uses caller-owned source identity only. Omitting
    // adaptive freezes the submitted fraction for deterministic replay.
    // defer_pack leaves the final SBS pack owed until a consumer calls pack(),
    // into an export slot or output(). An unconsumed pack is never recorded, and
    // output() then keeps an earlier image. Detection and every CPU decision
    // happen here; the depth conditioning only the pack, a dump or the
    // adaptive probe read is owed with the pack (recorded here for a probe
    // frame, an older two-pass replay shader or without defer_pack, in full),
    // and an owed pack that shows the source mono records none unless Dump 3D
    // is armed. A Present's owed work is dropped by the next render or
    // begin_present.
    bool render(reshade::api::command_list *commands, const render_frame_input &input, bool defer_pack = false);
    // Records the owed pack. A null target writes output(); an export target
    // receives the SBS image directly, avoiding a full-frame copy. Views of
    // export targets are cached for one export generation only.
    bool pack(reshade::api::command_list *commands, reshade::api::resource export_target = {},
      std::uint64_t export_generation = 0);
    // Lazily allocated at the current extent and exact input format. At most
    // three formats are retained; copies never reinterpret a different format.
    // The renderer queue reads only the explicitly selected mask channel.
    reshade::api::resource ui_source(reshade::api::format format = reshade::api::format::unknown);
    reshade::api::resource_view ui_source_view() const;
    reshade::api::resource ui_candidate(unsigned slot, reshade::api::format format);
    reshade::api::resource_view ui_candidate_view(unsigned slot) const;
    // Direct binding (docs/reshade-sbs.md, direct candidate binding): a
    // candidate the caller keeps readable and unwritten through this
    // Present's last read (a leased Streamline snapshot, a live UI layer
    // copy), bound as is, recording no copy. resource is the view's texture:
    // a leased snapshot's view is a raw descriptor that
    // device::get_resource_from_view cannot resolve. Returns the view, or
    // nothing (the caller then copies). Cleared at begin_present.
    reshade::api::resource_view bind_ui_candidate(unsigned slot, std::uint64_t capture_id,
      reshade::api::resource_view view, reshade::api::resource resource);
    // The same for a leased Streamline snapshot (D3D12 only): the renderer
    // describes it itself, a typed DXGI view_format of its storage, in an
    // original-device descriptor that push_descriptors can copy.
    reshade::api::resource_view bind_ui_snapshot(unsigned slot, std::uint64_t capture_id,
      reshade::api::resource resource, std::uint32_t view_format);
    // The completion fence of this renderer's submissions and the value that
    // its next signal (the current Present's finish_present, or a later one
    // for a Present that missed it) sets: a resource read by work recorded so
    // far is free once the fence reaches it.
    reshade::api::fence completion_fence() const;
    std::uint64_t completion_value() const;
    // The copy callback takes the destination texture and records the copy.
    template<class Copy>
    reshade::api::resource_view prepare_ui_candidate(unsigned slot, std::uint64_t capture_id, Copy &&copy,
        reshade::api::format format) {
      if (slot >= ui_candidate_captures_.size() || !capture_id) return {};
      const auto destination = ui_candidate(slot, format);
      if (!destination.handle) return {};
      if (ui_candidate_captures_[slot] != capture_id) {
        ui_candidate_captures_[slot] = 0;
        if (!std::forward<Copy>(copy)(destination)) return {};
        ui_candidate_captures_[slot] = capture_id;
      }
      return ui_candidate_view(slot);
    }
    // The caller must first admit a current capture. Repeated presentations of
    // that immutable capture reuse our private texture in renderer queue order.
    // A failed replacement never leaves the old identity marked as uploaded.
    template<class Copy>
    reshade::api::resource_view prepare_ui_source(std::uint64_t capture_id, Copy &&copy,
        reshade::api::format format = reshade::api::format::unknown) {
      if (!capture_id) return {};
      const auto destination = ui_source(format);
      if (!destination.handle) return {};
      if (ui_source_capture_ != capture_id) {
        ui_source_capture_ = 0;
        if (!std::forward<Copy>(copy)(destination)) return {};
        ui_source_capture_ = capture_id;
      }
      return ui_source_view();
    }
    reshade::api::resource output() const;
    // The export transfer (docs/reshade-sbs.md, PQ wire transfer): true packs
    // an HDR10 source as 10-bit PQ code values (Rec.2020, ST 2084, encoded as
    // the host's scRGBTo2100PQ) into an R10G10B10A2_UNORM target, output()
    // included; false packs FP16 scRGB. Only an HDR10 swapchain with a
    // per-tap PQ shader supports it; returns the transfer now in effect.
    bool set_pq_output(bool pq);
    bool pq_output() const;
    bool pq_output_supported() const;
    diagnostic_resources diagnostics() const;
    render_parameters consumed_parameters() const;
    bool consumed_source_alpha_ui() const;
    ui_mask_channel consumed_ui_channel() const;
    // Process-owned toggle decision, independent of current FG input eligibility.
    // The caller keeps automatic->session alive for this render; the renderer
    // owns only bounded GPU observation storage, never the detection deadline.
    alpha_auto_decision consumed_alpha_auto() const;
    ui_detection_snapshot consumed_detection() const;
    alpha_probe_counters alpha_probe_activity() const;
    // Value copy: policy evidence for the fraction consumed by this render.
    // Explicit mode-5 replay reports the supplied fraction with status frozen.
    ui_adaptive::decision consumed_ui_adaptive() const;
    // Recorded conflict probes, independent of readback acceptance or scope.
    std::uint64_t ui_probe_submissions() const;
    // Submitted bits. In nearest-UI mode inverse_depth is the floor; only the
    // current-render diagnostic GPU scalar contains the resolved global plane.
    // Front-limit mode ignores this depth word and dispatches no UI reduction.
    // Display-fraction mode freezes the resolved fraction in the same word.
    ui_plane_parameters consumed_ui_plane() const;
    reshade::api::resource_view native_rtv(reshade::api::resource backbuffer);
    // Also bracket capture's D3D11 unbinds, not only our draw calls. D3D12
    // records to ReShade's dedicated immediate list and needs no app-state swap.
    void begin_frame_state();
    void end_frame_state();
    // A new presentation begins. Work recorded for a Present that never reached
    // finish_present stays unsignaled until the next signal covers it, so this
    // presentation can render while resource replacement remains blocked. It
    // also numbers Presents for colors retained for late HUD-less pairing.
    void begin_present();
    void finish_present();
    // Marks the start of this presentation's input work for the GPU profile.
    void begin_gpu_profile(reshade::api::command_list *commands);
    // Completed-frame GPU stage times since the last call; resets the window.
    // No CPU wait: frames still in flight are reported by a later call.
    bool take_gpu_timing(gpu_timing &out);
    // ReShade's destroy_effect_runtime follows its GPU drain. No extra wait.
    void reset_after_runtime_drain();
    // The same, for a runtime reset that may be a swapchain resize (an SDR and
    // HDR toggle): the renderer and its cached one are kept for the runtime's
    // reinitialisation; only the views of the swapchain's back buffers go.
    // The next configure starts the decision state as a new renderer would.
    void park_after_runtime_drain();
    // The device of the current renderer (nullptr before the first configure).
    reshade::api::device *device() const;
    // Recorded conditioning since configure: recorded in full, skipped for a
    // pack that shows the source mono, and recorded with the depth candidate
    // and vertical field kept from the last one (an exact repeat of its
    // depth view, nonzero depth identity and parameters).
    struct conditioning_counters {
      std::uint64_t recorded = 0, mono = 0, memo = 0;
    };
    conditioning_counters conditioning_activity() const;
  private:
    struct impl;
    std::unique_ptr<impl> data_;
    // The renderer of the previous colour transfer (SDR and HDR toggles),
    // reused on a toggle back and released once idle for cached_idle_ms.
    std::unique_ptr<impl> cached_;
    std::uint64_t cached_since_ = 0;
    static constexpr std::uint64_t cached_idle_ms = 60000;
    std::uint64_t ui_source_capture_ = 0;
    std::array<std::uint64_t, 5> ui_candidate_captures_{};
    bool preparing_ = false;
    bool resume_pending_ = false;
  };
}
