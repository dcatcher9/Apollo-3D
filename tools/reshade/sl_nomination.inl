// SPDX-License-Identifier: GPL-3.0-only
// Part of streamline_camera_probe.cpp (sl_adapter): Streamline depth admission
// (source and tag rejection, depth frame normalization), the FG routing
// questions (frame_generation_route, source_path_selected,
// owns_nested_depth_capture), nomination and the v1/v2 evaluate hooks. One
// translation unit with its shell, which keeps every namespace-scope variable,
// thread_local and type and includes this part at its end; the shell declares
// each function below where it was defined.

namespace sunshine_streamline {
  SUNSHINE_PROBE_INTERNAL {
    bool unframed_tag_source(const evaluation_snapshot &snapshot) {
      // A synchronous ordinary tag can identify its own pixels without an SDK
      // frame identity. It supplies no camera, numeric frame or FG association.
      return snapshot.tag_boundary && !snapshot.frame_generation_input &&
        snapshot.frame.kind == frame_identity_kind::unavailable && !snapshot.frame_correlated;
    }
    // The first failing source gate, or nullptr when this observation may name a depth source.
    const char *source_rejection(const evaluation_snapshot &snapshot) {
      // A depth tag names the renderer's source independently of whether its
      // optional camera matrices can supply a scale. Preserve the stricter
      // camera status rather than upgrading it.
      switch (snapshot.status) {
        case evidence_status::source_associated_evaluation:
        case evidence_status::camera_reset:
        case evidence_status::missing_constants:
        case evidence_status::rejected_constants:
        case evidence_status::invalid_camera: break;
        case evidence_status::untracked_frame:
          if (!unframed_tag_source(snapshot)) return "untracked_frame";
          break;
        default: return name(snapshot.status);
      }
      if (!snapshot.epoch || !snapshot.sequence) return "no_observation";
      // A frame-generation tag call may come without a command list when its
      // tags are ValidUntilPresent (sl_core_api.h); tag_rejection judges each
      // tag, and nominate_depth_source copies them before the frame's Present.
      if (snapshot.tag_boundary && !snapshot.command_buffer && !snapshot.frame_generation_input)
        return "boundary_without_commands";
      if (snapshot.loss_revision != loss_revision.load(std::memory_order_acquire)) return "observation_lost";
      if (snapshot.frame_correlated && snapshot.decoded == decode_status::ok) {
        const bool current_reset = snapshot.camera.reset == 1 && snapshot.projection.valid() &&
          snapshot.status == evidence_status::camera_reset;
        if (snapshot.camera.reset != 0 && !current_reset) return "camera_reset";
        if (snapshot.frame.kind == frame_identity_kind::v1_numeric && snapshot.camera.not_rendering_game_frames != 0)
          return "not_rendering_game_frames";
      }
      auto pin = read_metadata();
      if (!pin && metadata.has_publication()) return "metadata_busy";
      static const metadata_state empty;
      const auto &state = pin ? *pin : empty;
      const bool same_epoch = snapshot.epoch == (pin ? pin->epoch : epoch.load(std::memory_order_acquire));
      const bool framed = unframed_tag_source(snapshot) || token_current(state, snapshot.frame);
      pin.reset();
      if (!same_epoch) return "epoch_changed";
      if (!framed) return "frame_token_not_current";
      return snapshot.loss_revision == loss_revision.load(std::memory_order_acquire) ? nullptr : "observation_lost";
    }
    bool direct_source_admissible(const evaluation_snapshot &snapshot) { return !source_rejection(snapshot); }
    // Where this observation copies a depth tag (sl_contract::copied_at):
    // inside the call that supplied it (a tag boundary, or the evaluate a
    // local input came with) or at a later evaluate.
    sl_contract::copy_point copy_point(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag) {
      return sl_contract::copied_at(snapshot.tag_boundary, tag.scope == tag_scope::evaluation_local);
    }
    // A tag's declaration as api_contract.h reads it. version_one: a 1.x
    // evaluate's tag, whose state 0 is chi eUnknown.
    template<class Tag> sl_contract::tag_declaration declaration(const Tag &tag, bool version_one) {
      sl_contract::tag_declaration out;
      out.lifecycle = tag.value.lifecycle; out.state = tag.native_state; out.version_one = version_one;
      out.supported = tag.supported; out.state_requires_observation = tag.state_requires_observation;
      return out;
    }
    // The first failing tag gate, or nullptr when the tag can supply this
    // observation's depth. matched: match()'s refusal when that gate failed.
    const char *tag_rejection(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag,
        match_status *matched) {
      if (!tag.present) return "absent";
      if (!tag.supported) {
        switch (tag.value.refusal) {
          case tag_refusal::unreadable: return "unsupported_unreadable";
          case tag_refusal::header_version: return "unsupported_header_version";
          case tag_refusal::extension: return "unsupported_extension";
          case tag_refusal::precision: return "unsupported_precision";
          case tag_refusal::resource_type: return "unsupported_resource_type";
          case tag_refusal::lifecycle: return "unsupported_lifecycle";
          default: return "unsupported";
        }
      }
      if (!buffers::is_depth(tag.value.type)) return "not_depth";
      // Only a ValidUntilPresent tag may come without a command list
      // (sl_core_api.h). Another tag of a frame-generation tag call without
      // one, and a stored UntilEvaluate tag whose call had none (illegal
      // input), have no point to copy at (the matrix's S2).
      if (snapshot.tag_boundary ? !snapshot.command_buffer && tag.value.lifecycle != sl_contract::valid_until_present :
          tag.scope != tag_scope::evaluation_local && !tag.commanded && tag.value.lifecycle == sl_contract::valid_until_evaluate)
        return "boundary_without_commands";
      if (snapshot.tag_boundary && !snapshot.frame_generation_input && tag.value.lifecycle != sl_contract::only_valid_now)
        return "lifecycle_outlives_boundary";
      if (snapshot.frame.kind == frame_identity_kind::v2_constants_call &&
          (!snapshot.tag_boundary || tag.scope != tag_scope::active_global || tag.value.lifecycle != sl_contract::only_valid_now))
        return "constants_call_frame";
      // UntilEvaluate is usable at this synchronous boundary only, and
      // OnlyValidNow inside the call that supplied it only: its tag boundary,
      // or the evaluate it is a local input of (sl_contract::lifecycle_at).
      auto at_evaluation = tag.value;
      at_evaluation.lifecycle = sl_contract::lifecycle_at(tag.value.lifecycle, copy_point(snapshot, tag));
      const auto status = match({snapshot.viewport, snapshot.epoch, 0, false}, at_evaluation);
      if (status != match_status::same_epoch_only) {
        if (matched) *matched = status;
        return name(status);
      }
      // match() reads no allocation, and the Resource's declared size bounds
      // no extent (S16d): where the tag's allocation is retained, an extent
      // outside it is refused here (the capture checks it again,
      // depth_capture::source_region).
      std::uint32_t width{}, height{};
      return depth_capture::source_extent(tag.direct_source, width, height) &&
          !sunshine_api_contract::extent_in_allocation(tag.value.area, width, height, 0, 0) ?
        name(match_status::invalid_extent) : nullptr;
    }
    bool direct_tag_admissible(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag) {
      return !tag_rejection(snapshot, tag);
    }
    // Streamline's untag (sl_core_api.h: a null resource removes the tag):
    // every depth tag of this call names no native resource. It ends those
    // tags and is not a capture attempt, so it names no source to withdraw. A
    // malformed or loss-marked tag is unsupported and still withdraws.
    bool depth_untag(const evaluation_snapshot &snapshot) {
      bool any = false;
      for (const auto &tag : snapshot.tags) {
        if (!tag.present) continue;
        if (!tag.supported || tag.value.native_resource) return false;
        any = true;
      }
      return any;
    }
    sunshine_scene_depth::jitter_offset depth_frame_jitter(const evaluation_snapshot &snapshot,
        const evaluated_depth_tag &tag) {
      if (!snapshot.frame_correlated || snapshot.decoded != decode_status::ok || !snapshot.camera.jitter_supplied)
        return {};
      // The jitter denominator is the active region: the tag's extent when
      // set, else (a null extent) the whole allocation (its declared size, or
      // the retained allocation's own when undeclared). The allocation bounds
      // the extent.
      const auto dimensions = [&](const evaluated_depth_tag &input) {
        const auto &source = input.value;
        sunshine_scene_depth::jitter_offset result;
        if (source.viewport != snapshot.viewport || !source.native_resource) return result;
        std::uint32_t full_width = source.resource_width, full_height = source.resource_height;
        std::uint32_t allocated_width{}, allocated_height{};
        if (depth_capture::source_extent(input.direct_source, allocated_width, allocated_height)) {
          full_width = allocated_width; full_height = allocated_height;
        }
        const auto active = sunshine_api_contract::normalized_extent(source.area, full_width, full_height);
        const auto width = active.width, height = active.height;
        if (!width || !height || width > 16384 || height > 16384 ||
            (full_width && (active.left > full_width || width > full_width-active.left)) ||
            (full_height && (active.top > full_height || height > full_height-active.top))) return result;
        return sunshine_scene_depth::jitter_offset{snapshot.camera.jitter_offset[0], snapshot.camera.jitter_offset[1], width, height, true};
      };
      if (tag.value.type == buffers::depth || tag.value.type == buffers::linear_depth) return dimensions(tag);
      // High-resolution depth is still jittered in render pixels. Its own
      // extent is not the denominator. Accept only a render input explicitly
      // frozen with this frame/batch; an older global tag may have another size.
      const auto &raw = snapshot.tags[0];
      const bool same_frame = raw.scope == tag_scope::explicit_frame || raw.scope == tag_scope::evaluation_local ||
        raw.observation_sequence == snapshot.sequence;
      return same_frame && direct_tag_admissible(snapshot, raw) ? dimensions(raw) : sunshine_scene_depth::jitter_offset{};
    }
    sunshine_scene_depth::frame normalize_depth_frame(const evaluation_snapshot &snapshot,
        const evaluated_depth_tag &tag, bool version_one) {
      report_declared_size(tag);
      sunshine_scene_depth::frame value;
      value.epoch = snapshot.epoch; value.sequence = snapshot.sequence; value.tick = GetTickCount64();
      value.observation_revision = snapshot.loss_revision;
      value.feedback = snapshot.feedback;
      value.viewport = snapshot.viewport;
      value.source_frame_generation = snapshot.frame.generation;
      value.source_frame_token = snapshot.frame.token;
      value.source_frame_numeric = snapshot.frame.numeric;
      value.source_frame_has_numeric = snapshot.frame.has_numeric;
      value.source_frame_explicit = snapshot.frame.kind != frame_identity_kind::unavailable &&
        (!snapshot.tag_boundary || tag.scope == tag_scope::explicit_frame);
      value.frame_generation_input = snapshot.frame_generation_input;
      value.source_id = snapshot.frame_generation_input ?
        sunshine_scene_depth::fg_source_id(snapshot.viewport) : 0;
      const bool valid = (snapshot.status == evidence_status::source_associated_evaluation ||
          snapshot.status == evidence_status::camera_reset) &&
        snapshot.projection.valid() && snapshot.camera.reset <= 1 && tag.present && tag.supported &&
        tag.value.viewport == snapshot.viewport && buffers::is_depth(tag.value.type);
      // Without the frame's own camera, the viewport's steady depth coefficients
      // (see evaluation_snapshot::viewport_projection_ready).
      const bool steady = !valid && !snapshot.frame_correlated && snapshot.viewport_projection_ready &&
        tag.present && tag.supported && tag.value.viewport == snapshot.viewport && buffers::is_depth(tag.value.type);
      const auto &coefficients = valid ? snapshot.projection : snapshot.viewport_projection;
      const bool correlated_direction = snapshot.frame_correlated && snapshot.decoded == decode_status::ok &&
        snapshot.camera.depth_inverted <= 1;
      const bool direction = correlated_direction || snapshot.viewport_direction;
      const auto inverted = correlated_direction ? snapshot.camera.depth_inverted : snapshot.viewport_inverted;
      value.projection = {valid || steady ? coefficients.depth_offset : 0.0,
        valid || steady ? coefficients.depth_scale : 0.0, valid || steady,
        direction && ((inverted != 0) != (tag.precision_scale < 0)), direction};
      value.projection.raw_scale = tag.precision_scale;
      value.projection.raw_bias = tag.precision_bias;
      if (tag.value.type == buffers::linear_depth) {
        value.projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
        value.projection.direction_supplied = true;
        value.projection.reversed = tag.precision_scale < 0;
      }
      value.jitter = depth_frame_jitter(snapshot, tag);
      value.resource = {tag.value.native_resource, tag.value.resource_width, tag.value.resource_height,
        {tag.value.area.left, tag.value.area.top, tag.value.area.width, tag.value.area.height},
        tag.value.type == buffers::high_resolution_depth ? sunshine_scene_depth::resource_kind::display_depth :
          sunshine_scene_depth::resource_kind::raw_depth, tag.value.native_format};
      sunshine_api_contract::apply(sl_contract::map(declaration(tag, version_one), copy_point(snapshot, tag)), value);
      return value;
    }
    fg_route frame_generation_route(std::uint32_t viewport, bool *disabled) {
      if (disabled) *disabled = false;
      if (!fg_available.load(std::memory_order_acquire) || !current_fg_target.load(std::memory_order_acquire)) return fg_route::none;
      // The viewport's committed FG options (the FG options store).
      const auto fg = read_fg_mode(viewport);
      const bool current = fg.known && fg.epoch == epoch.load(std::memory_order_acquire) &&
        fg.loss_revision == fg_loss_revision.load(std::memory_order_acquire);
      if (disabled) *disabled = current && !fg.enabled;
      if (!current || !fg.enabled) return fg_route::none;
      return depth_capture::frame_generation_scope_live(fg.epoch, viewport, fg.enabled_tick) ? fg_route::owned : fg_route::silent;
    }
    // FG has an explicit lifecycle for its rendered-frame inputs. While its
    // scope is live the same viewport's SR/RR evaluations remain diagnostics
    // and must not overwrite its newer FG source with a competing
    // renderer-input watermark. A silent FG scope leaves them eligible
    // (Assassin's Creed Shadows 10-22 had FG on and no FG depth captured).
    bool source_path_selected(const evaluation_snapshot &snapshot) {
      return snapshot.frame_generation_input || frame_generation_route(snapshot.viewport) != fg_route::owned;
    }
    bool owns_nested_depth_capture(const evaluation_snapshot &snapshot, bool copy_recorded) {
      if (!source_requested.load(std::memory_order_acquire) || !depth_capture::active()) return false;
      frame_generation_snapshot fg;
      // Confirmed same-viewport FG ownership also suppresses nested SR/RR work.
      // An ambiguous FG read is never ownership evidence, and neither is a
      // silent FG scope (frame_generation_route).
      if (query_frame_generation(snapshot.viewport, fg) && fg.enabled &&
          depth_capture::frame_generation_scope_live(fg.epoch, snapshot.viewport, fg.enabled_tick)) return true;
      // A retained resource or metadata-only nomination is not a depth copy.
      // Allow nested NGX to provide usable pixels when native SL admission fails.
      if (!copy_recorded || !source_path_selected(snapshot) || !direct_source_admissible(snapshot)) return false;
      return std::any_of(snapshot.tags.begin(), snapshot.tags.end(), [&](const evaluated_depth_tag &tag) {
        return tag.direct_source && direct_tag_admissible(snapshot, tag);
      });
    }
    // Why no tag of this observation offered a candidate: the first present
    // depth tag's failing gate in priority order, no_retained_source when it
    // passed every gate with no retained resource, or absent.
    const char *candidate_rejection(const evaluation_snapshot &snapshot) {
      for (const unsigned index : depth_priority) if (snapshot.tags[index].present) {
        const auto *rejected = tag_rejection(snapshot, snapshot.tags[index]);
        return rejected ? rejected : "no_retained_source";
      }
      return "absent";
    }
    // Whether this evaluate refuses only tags its frame's tag call decided:
    // every present depth tag an OnlyValidNow tag of the frame it was set for,
    // refused here for its lifetime (sl_contract::keeps_tag_call_copy), or an
    // untag of such a tag after its call (the evaluate reads no depth then,
    // and judges the call the untag ended in its place). Its frame is this
    // one where the evaluate names it (own_frame) or no Present came since
    // the tag call, as the capture judges a later use of a tag
    // (source_lifetime_current).
    bool tag_call_decided(const evaluation_snapshot &snapshot, bool version_one) {
      const auto decided = [&](const evaluated_depth_tag &tag) {
        auto matched = match_status::same_epoch_only;
        const bool same_frame = tag.own_frame || (tag.source_present_generation &&
          tag.source_present_generation == depth_capture::source_present_generation(tag.direct_source));
        return tag_rejection(snapshot, tag, &matched) && matched == match_status::unsupported_lifecycle &&
          sl_contract::keeps_tag_call_copy(declaration(tag, version_one), copy_point(snapshot, tag), same_frame);
      };
      bool any = false;
      for (const unsigned index : depth_priority) {
        const auto &tag = snapshot.tags[index];
        if (!tag.present) continue;
        if (tag.ended.present) {
          const auto &ended = tag.ended;
          evaluated_depth_tag call;
          call.value = ended.value; call.scope = ended.scope; call.present = true; call.supported = ended.supported;
          call.native_state = ended.native_state; call.state_requires_observation = ended.state_requires_observation;
          call.commanded = ended.commanded; call.direct_source = ended.direct_source;
          call.source_present_generation = ended.source_present_generation; call.own_frame = ended.own_frame;
          if (!decided(call)) return false;
        } else if (!decided(tag)) {
          return false;
        }
        any = true;
      }
      return any;
    }
    // deferred: a frame-generation tag call's depth that is copied before its
    // frame's Present instead (R2, deferred_copy); observe_tag_call keeps it
    // once the SDK accepted the call.
    std::uint64_t nominate_depth_source(const evaluation_snapshot &snapshot, bool version_one, bool *copy_recorded,
        deferred_depth *deferred) {
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      if (nomination_tap_armed) tap_nomination(snapshot, version_one);
#endif
      if (copy_recorded) *copy_recorded = false;
      if (deferred) *deferred = {};
      if (!source_requested.load(std::memory_order_acquire) || !depth_capture::active()) return 0;
      bool fg_disabled = false;
      const auto route = frame_generation_route(snapshot.viewport, &fg_disabled);
      if (!snapshot.frame_generation_input && route == fg_route::owned) {
        // source_path_selected. Still a later frame's call: an earlier
        // frame's presenting window ends here (until 10-28 a hitch frame
        // without an FG call, after a frame whose generated Present DLSS-G
        // skipped, showed that frame's copy past the source age as its own).
        depth_capture::evaluation_called(snapshot.epoch);
        return 0;
      }
      // Observation discovers interfaces; only an accepted submitted capture
      // establishes ownership in the shared capture layer.
      depth_capture::observe_command(snapshot.command_buffer);
      // An ordinary tag boundary, and ordinary evaluation of a viewport whose
      // FG scope is silent or whose FG is observed off, replaces that
      // viewport's FG observation; an FG nomination replaces its ordinary one
      // (source 0). With FG off no FG view is legitimate: until 10-28 an FG
      // view made after the Off (a late Off's deferred FG copy, DLSS-G's
      // evaluation or an FG tag call racing the options call) stayed beside
      // the SR view, so every Present read ambiguous views and revoked the
      // readable SR copies (the 10-28 review's P1). Not where FG is merely
      // unknown or ambiguous, which keeps a live scope's views.
      const bool fg_silent = route == fg_route::silent;
      const auto fg_source = sunshine_scene_depth::fg_source_id(snapshot.viewport);
      const auto superseded = snapshot.frame_generation_input ? 0 :
        snapshot.tag_boundary || fg_silent || fg_disabled ? fg_source : UINT64_MAX;
      // A silent FG scope has no FG observation to withdraw: its refused
      // boundaries leave the ordinary source of the viewport in place, and
      // only end an earlier frame's presenting window (below). A
      // withdrawn attempt is the namespace's newest nomination without a copy,
      // a missing frame of its display lineage that revokes no finished copy
      // (depth_capture::select_capture). A refused FG boundary also leaves its
      // frame without an FG copy (miss_frame_generation), which makes a live
      // scope silent.
      const bool withdraws = !(snapshot.frame_generation_input && fg_silent);
      const auto withdraw_invalid_attempt = [&] {
        // A silent scope's refused FG boundary is still its frame's call: it
        // ends an earlier frame's presenting window (frame_presenting), as a
        // live scope's withdrawal does from its begin and DLSS-G's refused
        // attempt does (10-31). Until 11-01 it ended none, so a generated
        // Present inside that frame's recording after it still presented the
        // earlier frame by depth's window: the UI held that frame's decision
        // there as a generated Present of it (PD7; soak seed
        // f0b594e3ad951a96 at frame 99), and depth could hold that frame's
        // copy there past the source age as its own (own_frame).
        if (!withdraws) {
          depth_capture::evaluation_called(snapshot.epoch);
          return;
        }
        depth_capture::begin_evaluation(snapshot.epoch, snapshot.sequence, snapshot.viewport,
          sunshine_scene_depth::provider_kind::streamline, snapshot.frame_generation_input ? fg_source : 0,
          superseded);
      };
      // Pixels already copied stay bounded by the source age and the consumed
      // watermark; the probe metadata still reports no current depth tag.
      // Assassin's Creed Shadows (10-22) sends a null FG depth tag every
      // frame, which used to withdraw its FG source each time.
      if (snapshot.tag_boundary && depth_untag(snapshot)) return 0;
      const bool fg_boundary = snapshot.tag_boundary && snapshot.frame_generation_input;
      if (const auto rejected = source_rejection(snapshot)) {
        if (fg_boundary) {
          report_fg_refusal(snapshot, rejected, withdraws, route == fg_route::owned);
          frame_missed(snapshot.epoch, snapshot.viewport, snapshot.sequence, evaluation_supplies(snapshot.epoch, snapshot.viewport));
        }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        remember_refusal(snapshot, sunshine_depth_refusal::gate::source, rejected);
#endif
        // An evaluate that reads no depth because the frame's untag ended its
        // OnlyValidNow tag after the tag call is no newer attempt either: it
        // keeps that call's outcome (tag_call_decided).
        if (!(snapshot.status == evidence_status::missing_depth && !snapshot.tag_boundary &&
            tag_call_decided(snapshot, version_one)))
          withdraw_invalid_attempt();
        return 0;
      }
      // The adapter defines semantic priority; the capture owner tries this
      // ordered set as one evaluation, including its in-progress lifetime.
      std::array<depth_capture::input, depth_priority.size()> candidates;
      std::size_t count = 0;
      // A frame-generation tag call's UntilEvaluate tag whose list records no
      // state: its evaluate copies it, not this call (S11c).
      bool awaits_evaluate = false;
      for (const unsigned index : depth_priority) {
        const auto &tag = snapshot.tags[index];
        if (!direct_tag_admissible(snapshot, tag) || !tag.direct_source) continue;
        auto &value = candidates[count];
        static_cast<sunshine_scene_depth::frame &>(value) = normalize_depth_frame(snapshot, tag, version_one);
        value.source = tag.direct_source;
        value.source_present_generation = tag.source_present_generation;
        value.force_snapshot = value.valid_until == sunshine_scene_depth::lifetime::at_call;
        // Inside the call that supplied the tag, a local input's evaluate
        // included: no Present can come between (source_lifetime_current).
        value.at_tag_call = copy_point(snapshot, tag) == sl_contract::copy_point::tag_call;
        value.own_frame = tag.own_frame;
        v1_resource_state::read_details state_details;
        const auto state = v1_resource_state::resolve(value.native_state, version_one,
          v1_private_state_ready.load(std::memory_order_acquire) ? reinterpret_cast<IUnknown *>(tag.value.native_resource) : nullptr,
          &state_details);
        if (version_one && state.proof == sunshine_scene_depth::state_proof::observed_nonzero) {
          // Bound failure reporting at the immutable evaluation boundary. An
          // enabled version adapter alone does not prove a usable state read.
          static std::atomic<std::uint64_t> next_state_report{};
          report_v1_state_lookup(snapshot, tag, value, state_details, next_state_report);
        }
        value.native_state = state.native;
        if (!tag.state_requires_observation) value.proof = state.proof;
        if (fg_boundary) {
          // Where this call copies the tag (R2): its declaration describes
          // the tag's use, so only a state its list records proves it here,
          // whatever the declaration names, and a ValidUntilPresent tag its
          // list records in a write state is copied before its frame's
          // Present instead (S11).
          bool writing = false;
          const bool recorded = snapshot.command_buffer &&
            depth_capture::recorded_state(snapshot.command_buffer, tag.direct_source, &writing);
          const auto point = sl_contract::deferred_copy(tag.value.lifecycle, snapshot.command_buffer != 0, recorded, writing);
          if (point != sl_contract::tag_call_copy::at_call) {
            if (point == sl_contract::tag_call_copy::present && deferred)
              deferred->candidates[deferred->count++] = std::move(value);
            else awaits_evaluate = true;
            value = {};
            continue;
          }
          value.proof = sl_contract::tag_call_proof(tag.value.lifecycle, value.proof);
        }
        ++count;
      }
      if (count && deferred) *deferred = {};
      if (deferred && deferred->count) {
        // Nothing is nominated, withdrawn or refused at this call: the copy
        // before the frame's Present decides (present_deferred).
        deferred->superseded = superseded;
        return 0;
      }
      depth_capture::record_diagnostic diagnostic;
      if (!count && awaits_evaluate) {
        diagnostic.result = depth_capture::status::missing_state;
        diagnostic.stage = depth_capture::record_stage::missing_state;
      }
      if (count) if (const auto ticket = depth_capture::nominate_evaluation(snapshot.command_buffer,
          candidates.data(), count, superseded, &diagnostic)) {
        if (copy_recorded) *copy_recorded = diagnostic.result == depth_capture::status::recorded;
        if (fg_boundary) remember_tag_copy(snapshot.epoch, snapshot.viewport, diagnostic.result == depth_capture::status::recorded,
          snapshot.sequence, snapshot.frame);
        if (fg_boundary && diagnostic.result == depth_capture::status::recorded)
          count_tag_call_copy(candidates.data(), count, diagnostic);
        else if (fg_boundary)
          miss_frame_generation(snapshot.epoch, snapshot.viewport, snapshot.sequence, evaluation_supplies(snapshot.epoch, snapshot.viewport));
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        // A metadata-only nomination keeps the source; its copy was refused.
        if (diagnostic.result != depth_capture::status::recorded)
          remember_refusal(snapshot, sunshine_depth_refusal::gate::capture, nullptr, &diagnostic);
#endif
        return ticket;
      }
      const bool attempted = count || awaits_evaluate;
      // A command-less FG tag call whose tags offer no copy point withdraws
      // nothing while DLSS-G's own evaluation supplies this live scope's
      // copies (observe_frame_generation_inputs): withdrawing would leave
      // every Present between the tag and that evaluation without a current
      // FG nomination. That evaluation, not the tag, then decides the frame
      // (its frame is not missed). One read of the supply decides both
      // (evaluation_supplies, PD10).
      const bool evaluation_supplied = fg_boundary && evaluation_supplies(snapshot.epoch, snapshot.viewport);
      const bool supplied = fg_boundary && !attempted && withdraws &&
        !std::strcmp(candidate_rejection(snapshot), "boundary_without_commands") && evaluation_supplied;
      if (fg_boundary) {
        frame_missed(snapshot.epoch, snapshot.viewport, snapshot.sequence, evaluation_supplied);
        char reason[96]{};
        if (attempted) std::snprintf(reason, sizeof(reason), "capture_%s/%s",
          depth_capture::name(diagnostic.result), depth_capture::name(diagnostic.stage));
        else std::snprintf(reason, sizeof(reason), "%s", candidate_rejection(snapshot));
        report_fg_refusal(snapshot, reason, withdraws && !supplied, route == fg_route::owned, supplied);
      }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      if (attempted) remember_refusal(snapshot, sunshine_depth_refusal::gate::capture, nullptr, &diagnostic);
      else remember_refusal(snapshot, sunshine_depth_refusal::gate::input, candidate_rejection(snapshot));
#endif
      // An evaluate that refuses only its frame's OnlyValidNow tags for their
      // lifetime (S3) is no newer attempt than their tag call (S1): it keeps
      // that call's copy, or its refusal, as the frame's.
      const bool kept = !attempted && tag_call_decided(snapshot, version_one);
      if (!supplied && !kept) withdraw_invalid_attempt();
      return 0;
    }
    // A copy survives a successful SDK call. Unrelated metadata observed or
    // lost while the original ran does not change the pixels it copied.
    void finish_capture(std::uint64_t captured, bool successful, std::uint64_t ticket) {
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      if (nomination_tap_armed) finish_tap(successful, ticket);
#endif
      if (!captured) return;
      depth_capture::finish(captured, successful && current(ticket), depth_capture::capture_failure::evaluation_failed);
    }
    bool hook_evaluate_v1(void *commands, std::uint32_t feature, std::uint32_t frame, std::uint32_t viewport) {
      const DWORD incoming = GetLastError();
      sunshine_upscaler_trace::streamline_scope call_trace(feature, SUNSHINE_UPSCALER_CALLER);
      const auto ticket = observation_ticket();
      // Reflex's fourth argument is a timing marker, not a viewport. Other
      // unknown legacy features need commands and an exact observed numeric
      // camera/depth tuple before that untyped argument may act as a viewport.
      const bool known_contract = feature == 0;
      const bool inspect = ticket && feature != 3 && (known_contract || commands);
      const loss_context_scope loss_scope(0, inspect ? viewport : UINT32_MAX, feature);
      if (ticket) ++evaluation_calls;
      evaluation_snapshot snapshot;
      if (inspect) {
        const auto serial = ++call_sequence;
        loss_context.sequence = serial;
        snapshot = capture_evaluation(viewport, feature, numeric_frame(frame), commands, ticket,
          serial, loss_revision.load(std::memory_order_acquire), nullptr, nullptr, false);
      }
      const bool sample = inspect && (known_contract || (snapshot.frame_correlated &&
        snapshot.decoded == decode_status::ok && snapshot.projection.valid() && direct_source_admissible(snapshot) &&
        std::any_of(snapshot.tags.begin(), snapshot.tags.end(),
          [&](const evaluated_depth_tag &tag) { return direct_tag_admissible(snapshot, tag); })));
      bool copy_recorded{};
      const auto captured = sample ? nominate_depth_source(snapshot, true, &copy_recorded) : 0;
      call_trace.claim_depth_capture(sample && owns_nested_depth_capture(snapshot, copy_recorded));
      const auto diagnostic = dump_stamp(ticket, snapshot.sequence, GetTickCount64(), viewport, 0, frame, true, reinterpret_cast<std::uint64_t>(commands));
      SetLastError(incoming);
      const bool result = original_evaluate_v1(commands, feature, frame, viewport);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result ? 1 : 0; loss_context.has_sdk_result = true;
      if (sample && diagnostic.session) dump_metadata::observe_sl_evaluation(diagnostic, feature, result);
      call_trace.finish(result);
      finish_capture(captured, result, ticket);
      if (sample) finish_evaluation(snapshot, result, ticket);
      SetLastError(outgoing);
      return result;
    }
    std::int32_t hook_evaluate_v2(std::uint32_t feature, const abi_v2::frame_token &frame,
        const base_structure **inputs, std::uint32_t count, void *commands) {
      const DWORD incoming = GetLastError();
      sunshine_upscaler_trace::streamline_scope call_trace(feature, SUNSHINE_UPSCALER_CALLER);
      const auto ticket = observation_ticket();
      // Known renderer profiles retain their fail-closed malformed-input
      // behavior. Other features are admitted by a validated typed depth tuple,
      // never by a feature-ID whitelist; opaque/non-renderer inputs pass through.
      const bool known_contract = known_renderer_contract(feature);
      if (ticket) ++evaluation_calls;
      const auto serial = ticket ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial, UINT32_MAX, feature);
      const auto loss = loss_revision.load(std::memory_order_acquire);
      const auto entry_tick = ticket ? GetTickCount64() : 0;
      batch out;
      if (ticket) out = read_local_inputs(inputs, count, reinterpret_cast<std::uintptr_t>(&frame), known_contract);
      const auto identity = ticket && (known_contract || out.valid_viewport) ?
        observed_frame(reinterpret_cast<std::uintptr_t>(&frame)) : frame_identity{};
      out.observation = ticket;
      out.serial = serial;
      out.loss = loss;
      out.frame = identity;
      out.tick = entry_tick;
      // Local UI inputs, copied on the evaluate's list inside the call that
      // supplied them. An evaluate without one leaves the hook gate to the
      // tag calls.
      const bool ui_inputs = ticket && out.valid_viewport && carries_ui(out);
      const auto alpha = ui_inputs ? capture_ui_inputs(out, commands) : alpha_attempts{};
      evaluation_snapshot snapshot;
      if (ticket && out.valid_viewport)
        snapshot = capture_evaluation(out.viewport, feature, identity, commands, ticket, serial, loss, &out, nullptr, false);
      const bool sample = ticket && (known_contract || (commands && out.valid_viewport &&
        direct_source_admissible(snapshot) && std::any_of(snapshot.tags.begin(), snapshot.tags.end(),
          [&](const evaluated_depth_tag &tag) { return direct_tag_admissible(snapshot, tag); })));
      const auto diagnostic = dump_stamp(ticket, serial, entry_tick, out.valid_viewport ? out.viewport : UINT32_MAX,
        reinterpret_cast<std::uintptr_t>(&frame), identity.numeric, identity.has_numeric, reinterpret_cast<std::uint64_t>(commands));
      if (sample && diagnostic.session) dump_metadata::observe_sl_inputs_v2(diagnostic, inputs, count);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      if (entry_policy_armed) {
        entry_policy_armed = false;
        entry_policy_observed = true;
        entry_path_selected = source_path_selected(snapshot);
        entry_source_admitted = direct_source_admissible(snapshot);
      }
#endif
      bool copy_recorded{};
      const auto captured = sample && out.valid_viewport ? nominate_depth_source(snapshot, false, &copy_recorded) : 0;
      call_trace.claim_depth_capture(sample && out.valid_viewport && owns_nested_depth_capture(snapshot, copy_recorded));
      if (sample && !out.valid_viewport && source_requested.load(std::memory_order_acquire) && depth_capture::active()) {
        depth_capture::observe_command(reinterpret_cast<std::uintptr_t>(commands));
        depth_capture::begin_evaluation(0, serial, 0xffffffffu);
      }
      SetLastError(incoming);
      const auto result = original_evaluate_v2(feature, frame, inputs, count, commands);
      const DWORD outgoing = GetLastError();
      // The depth copy and camera recorded before forwarding are the game's
      // own; an over-budget warning leaves them as valid as a plain success.
      const bool sdk_ok = sdk_accepted(result);
      loss_context.sdk_result = result; loss_context.has_sdk_result = true;
      if (sample && diagnostic.session) {
        dump_metadata::finish_sl_call(diagnostic, sdk_ok);
        dump_metadata::observe_sl_evaluation(diagnostic, feature, sdk_ok);
      }
      call_trace.finish(sdk_ok);
      finish_capture(captured, sdk_ok, ticket);
      finish_ui_inputs(alpha, sdk_ok && current(ticket) && loss == loss_revision.load(std::memory_order_acquire));
      if (sample && out.valid_viewport) {
        if (sdk_ok && out.count) store_tags(out);
        finish_evaluation(snapshot, sdk_ok, ticket);
      } else if (sample && current(ticket)) {
        // An unresolvable evaluation viewport cannot leave any older camera
        // evidence looking applicable to a new successful call.
        lose(!sdk_ok ? loss_diagnostics::reason::sdk_failure : loss_diagnostics::reason::invalid_input, __func__, 3027);
      } else if (ui_inputs && sdk_ok) {
        // Its UI inputs name the viewport whose UI the renderer requests
        // (query_ui_scope), whether or not it supplies depth.
        store_tags(out);
      }
      SetLastError(outgoing);
      return result;
    }
  }
}
