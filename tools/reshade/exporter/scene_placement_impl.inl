// SPDX-License-Identifier: GPL-3.0-only
// Function-only part of exporter.cpp, included at the end of that same TU.
// Former in-class member bodies retain inline; declarations/defaults and all
// runtime fields stay in the shell. exporter/scene_placement.inl supplies the
// decision types and free helpers at their original position.
// Docs: docs/reshade-sbs.md, Scene placement.
namespace {
  // Every scene the stereo reentry ramp places tells it this Present's
  // readiness (reentry_update), and a reused or held scene repeats its real
  // frame's as that frame left it.
  inline float publisher_t::reentry_update(runtime_t &proof, bool ready, std::uint64_t now,
      sunshine_raw_scene::stereo_basis basis) {
    proof.reentry_heard = true;
    return proof.raw_reentry.update(ready, now, basis);
  }

  // At the start of each Present of a runtime, under the lock: the previous
  // Present, if no scene told the ramp about it, was shown without Game 3D
  // stereo (Game 3D disabled, the renderer unavailable or preparing, a
  // failed native lease, an FX path that cannot place the scene: no FX
  // uniforms, or cropped depth an older FX cannot sample), and the ramp
  // hears it now as not ready at its own time, before this Present's scene
  // does. Presents shown without ready depth for more than 250 ms then
  // restart the 500 ms fade (owner ruling 10-27), while a pause with no
  // Present at all tells it nothing. Until the 10-27 review those exits
  // returned before the ramp heard of them, so stereo came back at full
  // strength after seconds of Game 3D disabled.
  inline void publisher_t::note_shown_present(runtime_t &proof, std::uint64_t now) {
    if (!std::exchange(proof.reentry_heard, false) && proof.reentry_present_ms)
      proof.raw_reentry.shown_without_depth(proof.reentry_present_ms);
    proof.reentry_present_ms = now;
  }

  inline void publisher_t::decide_scene_frame(api::effect_runtime *runtime, runtime_t &proof, frame_decision_t &frame,
      const frame_decision_t &previous_frame, const sunshine_depth::frame_depth &depth,
      bool game_enabled, std::uint64_t now) {
    frame.decided_ms = now;
    frame.present_ordinal = proof.presentation_ordinal;
    frame.depth_ready = depth.ready;
    if (!depth.ready) frame.gap = depth_gap::unavailable;
    if (depth.ready)
      frame.jitter = sunshine_depth_jitter::make(depth.provided.jitter, depth.width, depth.height,
        {depth.x, depth.y, depth.active_width, depth.active_height});
    frame.reused_depth = depth.reused_depth;
    const auto shown = sunshine_game3d::depth_output::describe(depth);
    frame.scene_source = shown.scene_source;
    // Source choice and bounded reuse are resolved once by the depth provider.
    // Another fallible metadata query here could contradict this same frame.
    frame.diagnostics.frame_generation_active = game_enabled && depth.frame_generation_active;
    if (depth.reused_depth) {
      // The depth source owns the short, invalidation-aware depth reuse window.
      // Render current color with those pixels and their previously resolved
      // geometry, without advancing calibration, samples or readiness ramps.
      // Strength reductions remain immediate; increases wait until the next
      // real-depth frame re-evaluates this scene's disparity interval.
      restore_reused_scene(frame, previous_frame, proof.raw_supported);
      // Its real frame's scene as that frame left it: the ramp heard that
      // frame, and this Present neither adds to nor ends its exposure.
      proof.reentry_heard = proof.reentry_heard || (proof.raw_supported && frame.depth_ready);
    } else {
      const bool provided = sunshine_game3d::depth_input::provider_selected(runtime);
      const sunshine_game3d::slow_step step("scene placement");
      if (provided && proof.raw_supported && game_enabled && hold_projection_scene(frame, previous_frame, depth, now)) {
        ++proof.camera_missing_holds;
        proof.reentry_heard = true; // The held scene, as for reused depth.
      } else frame.scene = resolve_raw_scene(runtime, proof, depth, shown, game_enabled, provided, now);
    }
    frame.ui_source = resolve_ui_observation(shown, previous_frame.ui_source, frame.scene, proof.raw_basis_epoch, now);
  }

  inline scene_parameters_t publisher_t::resolve_raw_scene(api::effect_runtime *runtime, runtime_t &proof,
      const sunshine_depth::frame_depth &depth, const sunshine_game3d::depth_output::shown_depth &shown,
      bool game_enabled, bool api_selected, std::uint64_t now) {
    scene_parameters_t scene;
    scene.admitted_strength = proof.frame_strength;
    scene.owned = proof.raw_supported;
    const bool enabled = proof.raw_supported && game_enabled;
    const bool provided = enabled && api_selected;
    const bool projection_provided = provided && (depth.projection.supplied ||
      depth.projection.encoding == sunshine_scene_depth::depth_encoding::linear_distance);
    const auto restart_reference = [&proof, now] {
      // A new basis invalidates queued samples. The raw controller owns
      // ordinary source changes and initializes each source only once.
      proof.raw_basis_epoch = next_raw_basis.fetch_add(1, std::memory_order_relaxed);
      proof.raw_policy.reset(proof.raw_basis_epoch, now);
      proof.provided_raw_policy.reset(proof.raw_basis_epoch, now);
      proof.raw_last_status = sunshine_raw_scene::status::uninitialized;
      proof.raw_log = {};
      proof.raw_reentry = {};
    };
#if defined(SUNSHINE_SBS_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
    if (take_recalibration(runtime)) {
      if (projection_provided) {
        const sunshine_projection_depth::domain domain{depth.projection.epoch, depth.projection.viewport,
          depth.projection.encoding, depth.provided.provider, depth.provided.source_id};
        if (!(domain == proof.projection_domain)) {
          proof.projection_domain = domain;
          proof.projection_policy.reset(domain, now);
        }
        proof.projection_policy.reset_reference(domain, now);
        proof.provided_raw_policy.discard(now);
        proof.raw_reentry = {};
        log(reshade::log::level::info, "Sunshine 3D Streamline: user requested Recenter; acquiring a fresh nearest-depth gain reference and screen plane");
      } else {
        restart_reference();
        // latest_center may still describe a capture from the action's same
        // millisecond. Rewrapping it in the new raw epoch must not make it new.
        proof.raw_policy.scene_cut(now);
        proof.provided_raw_policy.scene_cut(now);
        log(reshade::log::level::info, "Sunshine 3D raw automation: user requested Recenter; acquiring a fresh nearest-depth gain reference and screen plane");
      }
    }
#endif
    if (enabled && !proof.raw_basis_epoch)
      restart_reference();
    const auto budget = sunshine_scene_gain::render_limits(proof.frame_strength, proof.width, proof.height);
    proof.raw_policy.configure(budget);
    proof.provided_raw_policy.configure(budget);
    sunshine_depth::set_raw_scene_request(runtime, proof.raw_basis_epoch, enabled && !provided);
    using phase = sunshine_game3d::automatic_phase;
    if (!proof.raw_supported) {
      return scene; // No competing camera-uniform writer.
    }
    const bool cropped = depth.ready && (depth.x || depth.y || depth.active_width != depth.width || depth.active_height != depth.height);
    // Native constants always carry the active rectangle. Only reference
    // effects need to prove that their older uniform interface supports it.
    if (proof.addon_native || fx_reference_has_rect(proof)) {
      if (depth.ready && depth.width && depth.height) {
        scene.rect = {float(depth.x) / depth.width, float(depth.y) / depth.height,
          float(depth.active_width) / depth.width, float(depth.active_height) / depth.height};
      }
    } else if (cropped) {
      return scene; // An older shader must not sample allocation padding as scene depth.
    }
    if (projection_provided) {
      // A raw encoding returning from here must see a fresh target first.
      proof.provided_raw_policy.leave();
      resolve_projection_scene(runtime, proof, depth, now, budget, scene);
      return scene;
    }
    proof.projection_policy.suspend();
    if (!enabled || !proof.raw_basis_epoch) {
      proof.raw_policy.scene_cut(now);
      proof.provided_raw_policy.scene_cut(now);
      reentry_update(proof, false, now);
      scene.ui = {phase::waiting_for_depth, false};
      return scene;
    }

    sunshine_raw_scene::output state;
    bool source_changed = false;
    if (provided) {
      const auto current = sunshine_provided_raw::selected(depth.provided, proof.raw_basis_epoch, depth.ready);
      proof.provided_raw_policy.bind(current, now);
      sunshine_game3d::depth_input::sample measured;
      if (sunshine_game3d::depth_input::latest_sample(runtime, measured) && measured.moments.supplied) {
        sunshine_raw_scene::sample sample;
        sample.id = measured.id;
        sample.capture_ms = measured.tick;
        sample.metadata = sunshine_provided_raw::selected(measured.metadata, proof.raw_basis_epoch, true);
        sample.readback_frame = sample.metadata.frame;
        sample.readback_source = sample.metadata.source;
        sample.readback_layout_epoch = sample.metadata.layout_epoch;
        sample.range_supplied = true;
        sample.range_valid = measured.range_valid;
        sample.range_min = measured.range_min;
        sample.range_max = measured.range_max;
        sample.moments = measured.moments;
        proof.provided_raw_policy.observe(sample, current.frame, now);
      }
      state = proof.provided_raw_policy.evaluate(current, now);
    } else {
      const auto current = sunshine_game3d::depth_output::raw_frame(shown, proof.raw_basis_epoch);
      sunshine_depth::raw_source_roster roster;
      if (sunshine_depth::get_raw_source_roster(runtime, roster))
        scene.generic_routing_epoch = roster.routing_epoch;
      // Capture slots are transient; only authoritative resource/basis changes
      // invalidate retained calibration. Validate all bounded histories at once.
      const auto history = proof.raw_policy.history_sources();
      proof.raw_policy.retain_history(sunshine_depth::validate_raw_history(runtime, history));
      source_changed = depth.source_id && !proof.raw_policy.contains(current);
      // Capture owns sample identity and source lifetime; calibration consumes
      // that evidence independently of which source can render this present.
      // A missing current copy never prevents a completed peer sample from
      // reaching its own controller. Membership cannot be adopted from a packet.
      if (proof.raw_policy.synchronize(roster, now)) {
        std::array<sunshine_raw_scene::sample, sunshine_raw_scene::maximum_sources> samples;
        const auto count = sunshine_depth::get_raw_samples(runtime, samples);
        for (unsigned i = 0; i < count; ++i) {
          proof.raw_policy.observe(samples[i], now);
        }
      }
      // Rendering separately requires this present's selected, usable copy.
      // The policy retains numeric history when this evaluation returns mono.
      state = proof.raw_policy.evaluate(current, now);
      }
    scene.blend = reentry_update(proof, state.ready, now);
    {
      using reason = sunshine_raw_scene::status;
      phase display = phase::waiting_for_depth;
      if (state.ready) display = phase::ready;
      else if (state.reason == reason::calibrating)
        display = phase::calibrating;
      else if (state.reason == reason::unsupported_shader_domain)
        display = phase::suspended;
      scene.ui = {display, true, describe_scale(depth.ready ? sunshine_game3d::automatic_scale_basis::relative_depth :
        sunshine_game3d::automatic_scale_basis::unknown, report(state), budget.normalization)};
    }
    const bool status_changed = state.reason != proof.raw_last_status;
    // Rotation through already admitted members must not bypass the log gate
    // every frame. A newly admitted physical basis is still reported promptly.
    const bool suspended = status_changed && state.reason == sunshine_raw_scene::status::unsupported_shader_domain;
    if (proof.raw_log.due(now, status_changed, source_changed || suspended, state.ready)) {
      char message[640] {};
      std::snprintf(message, sizeof(message), "Sunshine 3D raw automation: %s; source=%llu; samples=%u stereo_scale=%.9g target_scale=%.9g t0=%.9g target_t0=%.9g reference_Q=%.9g normalization_L=%.9g reference_valid=%u plane_state=%s; uncalibrated relative raw basis; depth_path=%s feedback_revision=%llu",
        sunshine_raw_scene::name(state.reason), static_cast<unsigned long long>(depth.source_id),
        state.calibration_samples, state.H, state.target_H, state.t0, state.target_t0, scene.ui.scale.reference_inverse,
        scene.ui.scale.normalization, unsigned(state.has_depth_statistics), sunshine_scene_gain::name(state.learning), provided ?
          (depth.provided.provider == sunshine_scene_depth::provider_kind::ngx ? "NGX" : "SL") : "Generic",
        static_cast<unsigned long long>(provided ? depth.provided.feedback.revision : 0));
      log(reshade::log::level::info, message);
      proof.raw_last_status = state.reason;
    }
    scene.ready = state.ready;
    scene.basis = 1; // Explicitly not recovered camera coefficients.
    scene.projection = {state.shader_A, state.shader_inverseB};
    scene.zero = {state.referenceZPD, state.t0};
    if (state.calibrated)
      scene.ui_plane = {sunshine_game3d::ui_plane_mode::display_fraction, 0.f};
    scene.scale = state.H;
    return scene;
  }

  inline void publisher_t::resolve_projection_scene(api::effect_runtime *runtime, runtime_t &proof,
      const sunshine_depth::frame_depth &depth, std::uint64_t now,
      const sunshine_scene_gain::limits &budget, scene_parameters_t &scene) {
    namespace projection = sunshine_projection_depth;
    using phase = sunshine_game3d::automatic_phase;
    bool ready = false;
    projection::coefficients coefficients;
    projection::center_output center;
    const bool linear = depth.projection.encoding == sunshine_scene_depth::depth_encoding::linear_distance;
    if (depth.ready && (depth.projection.supplied || linear)) {
      const projection::domain domain{depth.projection.epoch, depth.projection.viewport,
        depth.projection.encoding, depth.provided.provider, depth.provided.source_id};
      if (!(domain == proof.projection_domain)) {
        proof.projection_domain = domain;
        proof.projection_policy.reset(domain, now);
        proof.raw_reentry = {};
      }
      coefficients = projection::make(depth.projection.encoding, depth.projection.A, depth.projection.B,
        depth.projection.raw_scale, depth.projection.raw_bias);
      if (fx_reference_rejects_linear(proof, linear)) coefficients = {}; // Legacy FX has no reciprocal input basis.
      proof.projection_policy.configure(budget);
      if (!proof.addon_native && !fx_reference_has_raw_range(proof) &&
          (depth.projection.raw_scale != 1.0 || depth.projection.raw_bias != 0.0)) coefficients = {};
      proof.projection_policy.synchronize_feedback(depth.provided.feedback);
      sunshine_game3d::depth_input::sample measured;
      if (sunshine_game3d::depth_input::latest_sample(runtime, measured) && measured.moments.supplied) {
        projection::sample sample;
        sample.id = measured.id; sample.capture_ms = measured.tick;
        sample.logical_domain = {measured.projection.epoch, measured.projection.viewport,
          measured.projection.encoding, measured.metadata.provider, measured.metadata.source_id};
        sample.projection = projection::make(measured.projection.encoding, measured.projection.A, measured.projection.B,
          measured.projection.raw_scale, measured.projection.raw_bias);
        sample.range_supplied = true;
        sample.range_valid = measured.range_valid;
        sample.range_min = measured.range_min;
        sample.range_max = measured.range_max;
        sample.moments = measured.moments;
        sample.feedback = measured.metadata.feedback;
        proof.projection_policy.observe(sample, now);
      }
      center = proof.projection_policy.evaluate(domain, coefficients, now);
      ready = center.ready;
    } else {
      proof.projection_policy.suspend();
    }
    scene.blend = reentry_update(proof, ready, now, sunshine_raw_scene::stereo_basis::projection);
    const bool unsupported = center.reason == projection::center_status::unsupported_shader_domain ||
      center.reason == projection::center_status::invalid_projection;
    auto placement = report(center);
    placement.ready = ready; // A suspended controller keeps its last center but is not ready.
    scene.ui = {
      ready ? phase::ready : !depth.ready ? phase::waiting_for_depth :
        unsupported ? phase::suspended : phase::calibrating, true,
      describe_scale(linear ? sunshine_game3d::automatic_scale_basis::linear_distance :
        sunshine_game3d::automatic_scale_basis::camera_matrix, placement, budget.normalization)};
    if (coefficients.valid() && !linear) {
      // Show the same affine conversion the shader applies, including any
      // packed raw-depth transform. Its units differ from the scene-estimated
      // stereo normalization K; neither display value drives rendering.
      scene.ui.scale.conversion_multiplier = coefficients.inverseB;
      scene.ui.scale.conversion_offset = -static_cast<double>(coefficients.shader_A) * coefficients.inverseB;
      scene.ui.scale.has_projection_conversion = true;
    }
    if (proof.projection_log.due(now, ready != proof.projection_ready, false, ready)) {
      char message[896]{};
      if (!depth.ready) {
        // No coefficients or center were evaluated on this pass. Reporting
        // their default zeros would falsely imply that calibration reset.
        std::snprintf(message, sizeof(message), "Sunshine 3D Streamline scale: waiting_for_depth; viewport=%u; retaining calibration state",
          depth.projection.viewport);
      } else {
        std::snprintf(message, sizeof(message), "Sunshine 3D Streamline scale: %s; viewport=%u encoding=%s A=%.9g B=%.9g near=%.9g stereo_scale=%.9g target_scale=%.9g q0=%.9g target_q0=%.9g reference_Q=%.9g normalization_L=%.9g reference_valid=%u conversion_scale=%.9g conversion_offset=%.9g samples=%u plane_state=%s feedback_revision=%llu camera_missing_holds=%llu; inverse-distance depth, independent stereo gain and zero plane",
          ready ? "ready" : projection::name(center.reason), depth.projection.viewport,
          linear ? "linear_distance" : "device",
          depth.projection.A, depth.projection.B, coefficients.near_plane, center.K, center.target_K, center.q0, center.target_q0,
          scene.ui.scale.reference_inverse, scene.ui.scale.normalization, unsigned(center.has_depth_statistics),
          scene.ui.scale.conversion_multiplier, scene.ui.scale.conversion_offset,
          center.calibration_samples, sunshine_scene_gain::name(center.learning),
          static_cast<unsigned long long>(depth.provided.feedback.revision),
          static_cast<unsigned long long>(proof.camera_missing_holds));
      }
      log(reshade::log::level::info, message);
      proof.projection_ready = ready;
    }
    scene.ready = ready;
    scene.projection_ms = depth.ready ? now : 0;
    scene.basis = linear ? 2 : 0;
    scene.projection = {coefficients.shader_A, coefficients.inverseB};
    scene.zero = {projection::reference_zpd, center.q0};
    if (center.initialized)
      scene.ui_plane = {sunshine_game3d::ui_plane_mode::display_fraction, 0.f};
    scene.raw_range = {coefficients.raw_min, coefficients.raw_max};
    scene.scale = center.K;
  }

}
