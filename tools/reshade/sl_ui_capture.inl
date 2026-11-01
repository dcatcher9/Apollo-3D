// SPDX-License-Identifier: GPL-3.0-only
// Part of streamline_camera_probe.cpp (sl_adapter): SL tag and DLSS-G UI-input
// capture through ui_mask's public API. The shared DLSS-G wrapper keeps its
// observation, serial, clock and depth execution; these helpers run exactly at
// the original UI boundaries. The shell keeps types, state and thread_local.
// One translation unit, included after sl_frame_generation.inl.

namespace sunshine_streamline {
  SUNSHINE_PROBE_INTERNAL {
    // later: where a ValidUntilPresent UI tag is kept to be copied before its
    // frame's Present (R2, the matrix's U2 and U5).
    sunshine_game3d::ui_mask::attempt capture_ui_tag(const batch &out, void *commands,
        const tag_record &tag, std::uint64_t source_epoch, bool fg_enabled, deferred_ui *later) {
      namespace mask = sunshine_game3d::ui_mask;
      depth_capture::input input;
      input.epoch = source_epoch; input.sequence = out.serial; input.tick = out.tick;
      input.viewport = out.viewport; input.observation_revision = out.loss;
      input.source_id = sunshine_scene_depth::fg_source_id(out.viewport);
      input.frame_generation_input = fg_enabled;
      input.source_frame_generation = out.frame.generation;
      input.source_frame_token = tag.token;
      input.source_frame_numeric = out.frame.numeric;
      input.source_frame_has_numeric = out.frame.has_numeric;
      input.source_frame_explicit = tag.source == origin::frame;
      input.resource.native = tag.supported ? tag.value.native_resource : 0;
      input.resource.width = tag.value.resource_width; input.resource.height = tag.value.resource_height;
      input.resource.native_format = tag.value.native_format;
      input.resource.area = {tag.value.area.left, tag.value.area.top, tag.value.area.width, tag.value.area.height};
      // An undeclared state (UINT_MAX) is no claim: only this recording's
      // observation can supply it, under either state policy. A UI tag is a
      // 2.x tag, copied inside its own tag call.
      sunshine_api_contract::apply(sl_contract::map(declaration(tag, false), sl_contract::copy_point::tag_call), input);
      input.force_snapshot = true;
      depth_capture::record_diagnostic retained;
      input.source = depth_capture::retain_source(input.resource.native, &retained);
      input.source_present_generation = depth_capture::source_present_generation(input.source);
      // Validate actual allocation size before recording, even if SL omitted
      // the optional resource dimensions. Partial output extents are rejected.
      if (input.source) { input.resource.width = retained.width; input.resource.height = retained.height; }
      mask::boundary where;
      where.source = input; where.command = reinterpret_cast<std::uint64_t>(commands);
      where.tag_scope = tag.source == origin::frame ? 1u : tag.source == origin::local ? 3u : 0u;
      where.kind = static_cast<mask::source_kind>(tag.value.type);
      where.source_present_generation = input.source_present_generation;
      where.format = input.source ? retained.format : 0;
      where.refusal = tag.value.refusal;
      if (later && input.source && tag.supported) {
        // A tag call copies its UI tag where it copies a frame-generation
        // depth tag (deferred_copy, the matrix's U5 and U2): inside the call
        // from the state its list records, which a declaration of the tag's
        // later use does not refuse (tag_call_proof, S11), else a
        // ValidUntilPresent tag (no list, no recorded state, or a write
        // state) before its frame's Present; an UntilEvaluate tag whose list
        // records none is refused at the call (missing_state), since no
        // evaluate copies a UI tag.
        bool writing = false;
        const bool recorded = commands &&
          depth_capture::recorded_state(reinterpret_cast<std::uint64_t>(commands), input.source, &writing);
        if (sl_contract::deferred_copy(tag.value.lifecycle, commands != nullptr, recorded, writing) ==
            sl_contract::tag_call_copy::present) {
          // Its kind's precedence over DLSS-G's input stays as it is until the
          // copy is made or dropped (present_deferred). While the kind's last
          // deferred copy was refused within the source age, DLSS-G's input
          // supplies it and no copy is deferred: one would only be refused
          // again, refreshing that refusal every frame.
          if (!ui_deferral_refused(source_epoch, out.viewport, mask::capture_slot(where.kind))) {
            later->used = true; later->where = where; later->input = input;
          }
          return {};
        }
        input.proof = sl_contract::tag_call_proof(tag.value.lifecycle, input.proof);
        where.source = input;
      }
      auto attempt = mask::begin(where, input);
      // A recorded copy keeps its kind's precedence over DLSS-G's input of
      // the same kind (ui_tag_copy); a refusal hands that kind over at once.
      if (mask::source_mask(where.kind))
        remember_ui_tag_copy(source_epoch, out.viewport, mask::capture_slot(where.kind), static_cast<bool>(attempt.ticket));
      return attempt;
    }
    // Whether a batch carries a UI input.
    bool carries_ui(const batch &out) {
      return std::any_of(ui_priority.begin(), ui_priority.end(), [&](unsigned index) { return out.tags[index].present; });
    }
    // The UI inputs of a tag call, or an evaluate's local ones (owner ruling
    // R4, the matrix's U3), each copied on that call's list.
    alpha_attempts capture_ui_inputs(const batch &out, void *commands, deferred_uis *later) {
      namespace mask = sunshine_game3d::ui_mask;
      const auto source_epoch = epoch.load(std::memory_order_acquire);
      mask::capture_gate_observation gate;
      gate.epoch = source_epoch; gate.revision = out.loss; gate.sequence = out.serial; gate.tick = out.tick;
      gate.viewport = out.valid_viewport ? out.viewport : UINT32_MAX;
      for (const auto index : ui_priority) if (out.tags[index].present)
        gate.seen_kinds |= mask::source_mask(static_cast<mask::source_kind>(out.tags[index].value.type));
      const bool interested = mask::interested(source_epoch, out.loss, out.viewport, out.tick, &gate.matching_requests);
      if (!out.valid_viewport) gate.state = mask::capture_gate::invalid_viewport;
      else if (!current(out.observation)) gate.state = mask::capture_gate::inactive_observation;
      else if (out.loss != loss_revision.load(std::memory_order_acquire)) gate.state = mask::capture_gate::observation_changed;
      else if (!interested) gate.state = gate.matching_requests ? mask::capture_gate::ambiguous_request : mask::capture_gate::no_matching_request;
      else gate.state = gate.seen_kinds ? mask::capture_gate::admitted : mask::capture_gate::no_ui_tags;
      mask::observe_gate(gate);
      if (gate.state != mask::capture_gate::admitted) return {};
      frame_generation_snapshot fg;
      const bool fg_enabled = query_frame_generation(out.viewport, fg) && fg.enabled;
      // Every source uses its own synchronous command/state/lifetime boundary.
      // Capture availability is independent of its eventual per-frame mask
      // validation, and HUD-less color remains a separate scene-color input.
      alpha_attempts attempts;
      for (unsigned i = 0; i != ui_priority.size(); ++i) {
        const auto &tag = out.tags[ui_priority[i]];
        if (tag.present) attempts[i] = capture_ui_tag(out, commands, tag, source_epoch, fg_enabled,
          later ? &(*later)[mask::capture_slot(static_cast<mask::source_kind>(tag.value.type))] : nullptr);
      }
      return attempts;
    }
    // After the SDK call that supplied them: kept only when that call
    // succeeded within the observation the copies were recorded in.
    void finish_ui_inputs(const alpha_attempts &attempts, bool kept) {
      for (const auto &attempt : attempts) {
        sunshine_game3d::ui_mask::finish(attempt, kept);
        // A revoked copy no longer keeps its kind's precedence (ui_tag_copy).
        if (!kept && attempt.ticket) forget_ui_tag_copy(sunshine_game3d::ui_mask::capture_slot(attempt.kind));
      }
    }
    void initialize_frame_generation_ui(const frame_generation::inputs &value, frame_generation::attempt &out) {
      using frame_generation::parameter;
      using frame_generation::ui_outcome;
    for (unsigned i = 0; i != value.ui.size(); ++i)
      out.ui_result[i] = value.ui[i].state == parameter::absent ? ui_outcome::absent : ui_outcome::no_fg_scope;
    }
    void capture_frame_generation_ui(const frame_generation::inputs &value, frame_generation::attempt &out,
        std::uint64_t source_epoch, std::uint64_t loss, const frame_generation_snapshot &fg,
        std::uint64_t serial, std::uint64_t tick, std::uint64_t source_id) {
      namespace mask = sunshine_game3d::ui_mask;
      using frame_generation::parameter;
      using frame_generation::ui_outcome;
    if (std::any_of(value.ui.begin(), value.ui.end(), [](const auto &item) { return item.state != parameter::absent; })) {
      const bool interested = mask::interested(source_epoch, loss, fg.viewport, tick);
      for (unsigned i = 0; i != value.ui.size(); ++i) {
        const auto &item = value.ui[i];
        auto &result = out.ui_result[i];
        if (item.state == parameter::absent) continue;
        if (!interested) { result = ui_outcome::no_request; continue; }
        if (!item.area_valid) { result = ui_outcome::subrect; continue; }
        // The game's own tag of this kind copied it first, or its copy waits
        // for this frame's Present: no second copy, and its exact pairs stay
        // the tags' own (ui_tags_supply).
        if (ui_tags_supply(source_epoch, fg.viewport, i)) {
          result = ui_outcome::tag_copied;
          continue;
        }
        depth_capture::input input;
        input.epoch = source_epoch; input.sequence = serial; input.tick = tick;
        input.viewport = fg.viewport; input.observation_revision = loss;
        input.source_id = source_id; input.frame_generation_input = true;
        input.resource.native = item.state == parameter::present ? item.native : 0;
        input.resource.area = item.area;
        sunshine_api_contract::apply(sunshine_api_contract::ngx::frame_generation_input, input);
        input.force_snapshot = true; input.at_tag_call = true;
        depth_capture::record_diagnostic retained;
        input.source = depth_capture::retain_source(input.resource.native, &retained);
        input.source_present_generation = depth_capture::source_present_generation(input.source);
        if (input.source) { input.resource.width = retained.width; input.resource.height = retained.height; }
        mask::boundary where;
        where.source = input; where.command = value.command; where.tag_scope = 2;
        where.kind = mask::capture_order[i];
        where.source_present_generation = input.source_present_generation;
        where.format = input.source ? retained.format : 0;
        where.frame_generation_evaluation = true;
        out.ui[i] = mask::begin(where, input);
        result = item.state == parameter::present ? ui_outcome::offered : ui_outcome::null_resource;
      }
    }
    }
    void finish_frame_generation_ui(const frame_generation::attempt &value, bool successful, bool still_current) {
    const bool ui_current = successful && still_current && value.loss == loss_revision.load(std::memory_order_acquire);
    for (const auto &attempt : value.ui) sunshine_game3d::ui_mask::finish(attempt, ui_current);
    }
  }
}
