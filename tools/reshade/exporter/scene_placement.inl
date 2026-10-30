// SPDX-License-Identifier: GPL-3.0-only
// Part of exporter.cpp, included at the original decision-type/helper position,
// before runtime_t. Internal linkage and the source order are unchanged.
// Runtime calibration fields stay in runtime_t at their original positions;
// exporter/scene_placement_impl.inl holds the inline member policy bodies.
// Docs: docs/reshade-sbs.md, Scene placement.
  // Value-only decision for one effects pass. Calibration resolves these values
  // first; shader uniforms, UI and export consume the same completed decision.
  // No borrowed textures or independently mutable copies of readiness live here.
  struct scene_parameters_t {
    bool owned = false, ready = false;
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
    // When this Present's depth was prepared (GetTickCount64) and its
    // runtime_t::presentation_ordinal: hold_projection_scene ages a held
    // placement by the time it was shown.
    std::uint64_t decided_ms = 0, present_ordinal = 0;
  };

  sunshine_game3d::ui_adaptive::source resolve_ui_observation(const sunshine_game3d::depth_output::shown_depth &shown,
      const sunshine_game3d::ui_adaptive::source &previous, const scene_parameters_t &scene,
      std::uint64_t raw_basis_epoch, std::uint64_t now) {
    return sunshine_game3d::depth_output::ui_observation(shown, previous, scene.ready,
      scene.generic_routing_epoch, raw_basis_epoch, now);
  }

  // Raw and projection controllers report the same placement quantities under
  // different names; the panel and dumps describe both through one mapping.
  struct placement_report {
    float gain{}, zero{}, ui_midpoint{};
    double target_gain{}, target_zero{}, target_ui_midpoint{};
    bool ready{}, placed{}, limited{}, has_statistics{};
    sunshine_scene_gain::depth_range statistics{};
  };
  placement_report report(const sunshine_raw_scene::output &state) {
    return {state.H, state.t0, state.ui_midpoint_q, state.target_H, state.target_t0, state.target_ui_midpoint_q,
      state.ready, state.calibrated, state.limited, state.has_depth_statistics, state.depth_statistics};
  }
  placement_report report(const sunshine_projection_depth::center_output &center) {
    return {center.K, center.q0, center.ui_midpoint_q, center.target_K, center.target_q0, center.target_ui_midpoint_q,
      center.ready, center.initialized, center.limited, center.has_depth_statistics, center.depth_statistics};
  }
  sunshine_game3d::automatic_scale describe_scale(sunshine_game3d::automatic_scale_basis basis,
      const placement_report &placement, double normalization) {
    sunshine_game3d::automatic_scale scale{basis, placement.gain, placement.ready, static_cast<float>(placement.target_gain)};
    const auto &statistics = placement.statistics;
    scale.zero_inverse = placement.zero;
    scale.has_zero = placement.placed;
    scale.target_zero_inverse = placement.target_zero;
    scale.zero_target_available = placement.has_statistics;
    scale.ui_midpoint_inverse = placement.ui_midpoint;
    scale.target_ui_midpoint_inverse = placement.target_ui_midpoint;
    scale.has_ui_midpoint = placement.placed;
    scale.ui_midpoint_target_available = placement.has_statistics;
    scale.gain_below_target = placement.limited;
    scale.reference_inverse = statistics.maximum;
    scale.mean_inverse = statistics.mean;
    scale.normalization = normalization;
    scale.minimum_inverse = statistics.minimum;
    scale.maximum_inverse = statistics.maximum;
    scale.has_depth_statistics = placement.has_statistics;
    scale.mean_square_inverse = statistics.mean_square;
    scale.has_centered_depth_statistics = statistics.has_centered_moments;
    scale.mean_contrast_inverse = statistics.mean_contrast;
    scale.mean_contrast_square_inverse = statistics.mean_contrast_square;
    scale.depth_pixel_count = statistics.pixel_count;
    scale.depth_tiles_x = statistics.tiles_x;
    scale.depth_tiles_y = statistics.tiles_y;
    return scale;
  }

  void restore_reused_scene(frame_decision_t &frame, const frame_decision_t &previous, bool supported) {
    // Only the depth source can authorize old pixels: an API provider's held
    // or pending copy, or Generic's hold of the capture it showed on the
    // previous Present. Their already resolved scene travels with the exact
    // real capture, for NGX pending copies as well as FG and Generic holds.
    // This path never evaluates calibration or advances its motion clock.
    // The scene is copied as the real frame left it, placed or still calibrating:
    // a generated frame must look like its real frame. Dropping its depth while
    // the real frame still shows depth (range calibration after FG starts, or a
    // camera cut) flickered generated frames flat at up to four per real frame.
    frame.scene.owned = supported;
    if (frame.reused_depth && frame.depth_ready && previous.depth_ready &&
        sunshine_game3d::depth_output::same_capture(frame.scene_source, previous.scene_source))
      frame.scene = previous.scene;
    else if (supported) {
      frame.scene.ui = {sunshine_game3d::automatic_phase::waiting_for_depth, true};
      if (frame.depth_ready) frame.gap = previous.depth_ready ? depth_gap::reuse_other_source : depth_gap::reuse_after_gap;
      frame.depth_ready = false;
    }
  }

  // Part declarations: functions defined in exporter/present_diagnostics.inl (included at
  // the end of this file) that code here calls.
  sunshine_game3d::vram::reading query_vram(IDXGIAdapter3 *adapter);
  output_sample classify_output(const frame_decision_t &frame);

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
      const sunshine_depth::frame_depth &depth, std::uint64_t now) {
    const bool own_camera = depth.projection.supplied ||
      depth.projection.encoding == sunshine_scene_depth::depth_encoding::linear_distance;
    const auto held = previous.scene.projection_ms;
    const bool next_present = previous.present_ordinal && frame.present_ordinal == previous.present_ordinal + 1;
    const auto shown_until = next_present && previous.decided_ms <= now ? previous.decided_ms : now;
    if (own_camera || !depth.ready || !previous.depth_ready || !held ||
        !sunshine_freshness::fresh_le(held, shown_until, sunshine_freshness::source_age_ms) || !frame.scene_source[0] ||
        !sunshine_game3d::depth_output::same_source_except_sequence(previous.scene_source, frame.scene_source)) return false;
    frame.scene = previous.scene;
    return true;
  }

