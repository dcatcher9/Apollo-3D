// SPDX-License-Identifier: GPL-3.0-only
// The scene placement values (docs/reshade-sbs.md, Scene placement): the
// value-only decision of one effects pass (scene_parameters_t,
// frame_decision_t), the placement reports, and the exact-source reuse and
// hold helpers that exporter/scene_placement.cpp defines. Runtime calibration
// fields stay in runtime_t (exporter_internal.h).
#pragma once
#include "export_ring.h"

namespace sunshine_exporter {
  // Value-only decision for one effects pass. Calibration resolves these values
  // first; shader uniforms, UI and export consume the same completed decision.
  // No borrowed textures or independently mutable copies of readiness live here.
  struct scene_parameters_t {
    bool ready = false;
    std::uint64_t generic_routing_epoch = 0;
    int basis = 0;
    float scale = 0.f, blend = 0.f;
    // Held-depth presentations reuse the geometry admitted at this strength.
    // A decrease is immediate; an increase waits for the next real-depth frame.
    float admitted_strength = sunshine_game3d::default_strength;
    std::array<float, 2> projection{}, zero{}, raw_range{0.f, 1.f};
    std::array<float, 4> rect{0.f, 0.f, 1.f, 1.f};
    // Temporary scene unavailability suspends this live policy; it must not
    // masquerade as an explicit switch to a different placement mode.
    sunshine_game3d::ui_plane_parameters ui_plane{sunshine_game3d::ui_plane_mode::display_fraction, 0.f};
    sunshine_game3d::automatic_status ui;
    // When the projection controller last resolved this scene from a capture
    // with its own camera; zero for raw placement. Copies keep the original time.
    std::uint64_t projection_ms = 0;
  };
  struct frame_decision_t {
    bool prepared = false, depth_ready = false, reused_depth = false;
    depth_gap gap = depth_gap::none;
    // Feature observations are reporting only. Scene admission consumes ready,
    // reused_depth and the frozen capture identity, never a feature switch.
    struct { bool frame_generation_active = false; } diagnostics;
    sunshine_game3d::source_alpha_ui_decision source_alpha;
    // Identify the real capture for which scene parameters were resolved. A
    // later pending/generated presentation may reuse that scene, never another source's.
    // A Generic capture is runtime epoch, source, frame and layout | the Generic marker.
    std::array<std::uint64_t, 4> scene_source{}; // epoch, source, sequence, viewport
    sunshine_game3d::ui_adaptive::source ui_source;
    scene_parameters_t scene;
    sunshine_depth_jitter::correction jitter;
    // When this Present's depth was prepared (the game clock) and its
    // runtime_t::presentation_ordinal: hold_projection_scene ages a held
    // placement by the time it was shown.
    std::uint64_t decided_ms = 0, present_ordinal = 0;
  };

  sunshine_game3d::ui_adaptive::source resolve_ui_observation(const sunshine_game3d::depth_output::shown_depth &shown,
      const sunshine_game3d::ui_adaptive::source &previous, const scene_parameters_t &scene,
      std::uint64_t raw_basis_epoch, std::uint64_t now);

  // Raw and projection controllers report the same placement quantities under
  // different names; the panel and dumps describe both through one mapping.
  struct placement_report {
    float gain{}, zero{}, ui_midpoint{};
    double target_gain{}, target_zero{}, target_ui_midpoint{};
    bool ready{}, placed{}, limited{}, has_statistics{};
    sunshine_scene_gain::depth_range statistics{};
  };
  placement_report report(const sunshine_raw_scene::output &state);
  placement_report report(const sunshine_projection_depth::center_output &center);
  sunshine_game3d::automatic_scale describe_scale(sunshine_game3d::automatic_scale_basis basis,
      const placement_report &placement, double normalization);

  void restore_reused_scene(frame_decision_t &frame, const frame_decision_t &previous);

  // A fresh API capture that arrived without its own camera renders with the
  // placement the projection controller resolved for the same source at most
  // the source age earlier (freshness.h), without advancing calibration. Projection
  // coefficients are camera constants; switching this one frame to the raw
  // controller instead rendered it mono (raw calibration never completed) and
  // restarted the 500 ms strength ramp. The Witcher 3 misses the camera of a
  // few SR evaluations every second.
  // That age is shown time: measured to the previous Present, which showed
  // this placement with ready depth of the same source, when this Present
  // is the one right after it (presentation ordinals), so a pause in
  // Presents alone (a game hitch) does not drop the placement (owner ruling
  // 10-27; until the 10-27 review a camera-less frame after a pause of the
  // source age switched to the raw basis, mono, and faded in again). A run
  // of camera-less Presents still ends one Present after the source age of
  // shown time, and a hold never spans two Presents of which neither had
  // the camera: after a second pause the age runs to a Present that only
  // held it.
  bool hold_projection_scene(frame_decision_t &frame, const frame_decision_t &previous,
      const sunshine_depth::frame_depth &depth, std::uint64_t now);
}  // namespace sunshine_exporter
