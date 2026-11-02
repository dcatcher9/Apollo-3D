// SPDX-License-Identifier: GPL-3.0-only
// Part of streamline_camera_probe.cpp (sl_adapter): the Streamline ABI decode,
// observation tickets, the tag/camera store, evaluation assembly and the
// present-thread queries. One translation unit with its shell, which keeps
// every namespace-scope variable and thread_local and includes this part at
// its end; the shell declares each function below where it was defined.

namespace sunshine_streamline {
  SUNSHINE_PROBE_INTERNAL {
    bool known_renderer_contract(std::uint32_t feature) { return feature == 0 || feature == 1001; }
    // Another thread requesting a new frame's token at the same moment owns the
    // writer only for a short copy (repeated requests for a frame never write).
    // Failing at once turned each collision into an observation loss (depth
    // dropped for ~5 presents, UI placement reset). A bounded retry keeps the
    // hook non-blocking: a preempted writer still fails after the bound.
    token_owner::transaction write_tokens_briefly() {
      auto update = token_metadata.try_write();
      for (unsigned attempt = 0; !update && update.failure() == observation::write_failure::owner_busy && attempt < 64; ++attempt) {
        if (attempt < 48) YieldProcessor();
        else SwitchToThread();
        update = token_metadata.try_write();
      }
      return update;
    }
    void release_camera_records() {
      // Lifecycle only: callbacks are already closed. Readers never delay this
      // clear; retired source leases are released outside mutation ownership.
      while (!metadata.clear()) SwitchToThread();
      clear_fg_modes();
      // A restart ends the scope of the process's FG view as well: the next
      // Present reads FG as a new process does. Until 11-01 the view kept
      // its UI projection, so FG On confirmed before a device recreation
      // stayed the UI's mode for the rest of the process when the game did
      // not set its options again (F26: Present alpha refused as FG_active,
      // the overlay's unprotected-HUD warning suppressed).
      AcquireSRWLockExclusive(&fg_view_lock);
      fg_state = {};
      ReleaseSRWLockExclusive(&fg_view_lock);
      drop_deferred();
    }

    bool same(const guid &a, const guid &b) { return std::memcmp(&a, &b, sizeof(guid)) == 0; }
    bool read_bytes(const void *source, void *destination, std::size_t size) {
      SIZE_T copied = 0;
      return source && ReadProcessMemory(GetCurrentProcess(), source, destination, size, &copied) && copied == size;
    }
    template<class T> bool read(const T *source, T &destination) { return read_bytes(source, &destination, sizeof(T)); }
    void message(const char *text) { sunshine_log::message(reshade::log::level::info, text); }
    // `line` is a frozen site id, not a live line number: each loss site below
    // passes the literal that its source line number had at ff50be30 (where it
    // was __LINE__). Never renumber them; they drift from the source on purpose.
    std::uint64_t lose(loss_diagnostics::reason cause, const char *site, std::uint32_t line,
        loss_diagnostics::context details) {
      const auto previous = loss_revision.fetch_add(1, std::memory_order_acq_rel);
      const DWORD saved_error = GetLastError();
      // A contended or overwritten diagnostic slot is simply unavailable. It
      // must never itself lose source observations or alter the revision count.
      loss_journal.publish({previous + 1, sunshine_game3d::clock::game_now(), GetCurrentThreadId(), line, cause, site, details});
      SetLastError(saved_error);
      return previous;
    }
    void invalid_observation(loss_diagnostics::reason cause, const char *site, std::uint32_t line) {
      ++invalid; lose(cause, site, line);
    }
    void dropped_observation(loss_diagnostics::reason cause, const char *site, std::uint32_t line,
        loss_diagnostics::context details) {
      ++dropped; lose(cause, site, line, details);
    }
    std::uint64_t observation_ticket() {
      const auto ticket = observation_generation.load(std::memory_order_acquire);
      if (!middleware_requested() || !observing.load(std::memory_order_acquire)) return 0;
      return observation_generation.load(std::memory_order_acquire) == ticket ? ticket : 0;
    }
    bool current(std::uint64_t ticket) {
      return ticket && middleware_requested() && observing.load(std::memory_order_acquire) &&
        observation_generation.load(std::memory_order_acquire) == ticket;
    }

    // briefly: retry a read that failed on every attempt
    // (observer_snapshot.h: the publication changed under it, or the slot's
    // reader count changed between its load and its pin, because another
    // reader pinned or unpinned the slot or a writer pinned its base) a
    // bounded number of times, like write_metadata, for a call's own camera
    // evidence, which the game never repeats for its frame (capture_evaluation,
    // capture_depth_tag). Until 11-01 such a read refused DLSS-G's camera
    // evidence (scope_changed, no copy), made an SL evaluation's snapshot busy
    // and dropped a whole depth tag boundary under a burst of camera, tag and
    // token calls on other threads. A publication of an earlier epoch or
    // observation reads as none at once.
    metadata_owner::read_pin read_metadata(bool briefly) {
      const auto read = [] {
#if defined(SUNSHINE_STREAMLINE_PROBE_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
        for (auto count = metadata_read_failures.load(std::memory_order_acquire); count;)
          if (metadata_read_failures.compare_exchange_weak(count, count - 1, std::memory_order_acq_rel))
            return metadata_owner::read_pin{};
#endif
        return metadata.read();
      };
      auto pin = read();
      for (unsigned attempt = 0; briefly && !pin && metadata.has_publication() && attempt < 64; ++attempt) {
        if (attempt < 48) YieldProcessor();
        else SwitchToThread();
        pin = read();
      }
      if (pin && (pin->epoch != epoch.load(std::memory_order_acquire) ||
          pin->observation != observation_generation.load(std::memory_order_acquire))) pin.reset();
      return pin;
    }
    // briefly: retry a contended writer a bounded number of times, like
    // write_tokens_briefly, for a publication the game may never repeat.
    metadata_owner::transaction write_metadata(std::uint64_t ticket, bool briefly) {
      auto update = metadata.try_write();
      for (unsigned attempt = 0; briefly && !update && update.failure() == observation::write_failure::owner_busy && attempt < 64; ++attempt) {
        if (attempt < 48) YieldProcessor();
        else SwitchToThread();
        update = metadata.try_write();
      }
      if (update) {
        if (!current(ticket)) update.reset();
        else { update->epoch = epoch.load(std::memory_order_acquire); update->observation = ticket; }
      }
      return update;
    }
    void metadata_write_lost(const metadata_owner::transaction &update, const char *site,
        std::uint32_t line, loss_diagnostics::context details) {
      dropped_observation(update.failure() == observation::write_failure::storage_busy ?
        loss_diagnostics::reason::metadata_storage_busy : loss_diagnostics::reason::records_busy, site, line, details);
    }
    viewport_record &slot(metadata_state &state, std::uint32_t viewport) {
      for (auto &record : state.records) if (record.used && record.viewport == viewport) return record;
      auto *chosen = &state.records[0];
      for (auto &record : state.records) {
        if (!record.used) { chosen = &record; break; }
        if (record.changed < chosen->changed) chosen = &record;
      }
      // The replaced viewport's FG mode goes with its record (drop_fg_mode).
      if (chosen->used) drop_fg_mode(chosen->viewport);
      *chosen = {};
      chosen->used = true;
      chosen->viewport = viewport;
      return *chosen;
    }
    bool same_frame(const frame_identity &a, const frame_identity &b) {
      return a.kind != frame_identity_kind::unavailable && a.kind == b.kind &&
        (a.kind == frame_identity_kind::v1_numeric ? a.numeric == b.numeric :
          a.generation == b.generation && a.token == b.token);
    }
    frame_identity numeric_frame(std::uint32_t frame) {
      return {frame_identity_kind::v1_numeric, 0, frame, 0, true};
    }
    bool token_current(const metadata_state &state, const frame_identity &frame) {
      // Constants-call identity is checked in the same immutable metadata
      // version; token mint identity has a separate bounded owner.
      if (frame.kind == frame_identity_kind::v1_numeric) return true;
      if (frame.kind != frame_identity_kind::v2_constants_call &&
          frame.kind != frame_identity_kind::v2_observed_token) return false;
      auto pin = token_metadata.read();
      if ((!pin && token_metadata.has_publication()) ||
          (pin && pin->observation != observation_generation.load(std::memory_order_acquire))) return false;
      bool current_token = frame.kind == frame_identity_kind::v2_constants_call;
      if (pin) for (const auto &token : pin->tokens) {
        if (frame.kind == frame_identity_kind::v2_constants_call) {
          if (token.frame.token == frame.token && token.frame.generation > frame.generation) {
            current_token = false; break;
          }
        } else if (same_frame(token.frame, frame)) {
          current_token = true; break;
        }
      }
      if (!current_token) return false;
      // Only a synchronous global depth tag may use this latest successful
      // constants-call identity. It is never an SDK frame number/token mint.
      if (frame.kind == frame_identity_kind::v2_constants_call) {
        for (const auto &record : state.records)
          if (record.used && same_frame(record.camera_frame, frame)) return true;
        return false;
      }
      return true;
    }
    // The frame a tag call's tags were set for: its frame token's, or for a
    // global tag (slSetTag names no frame) the viewport's newest constants
    // when they came first on a minted frame token; otherwise none. An
    // evaluation of that frame may use the tag after Presents of earlier
    // frames (evaluated_depth_tag::own_frame), and a copy deferred to that
    // frame's Present is cancelled by its untag (R7).
    frame_identity tag_frame(const metadata_state &state, const viewport_record &record, const frame_identity &call,
        bool global, std::uint64_t serial) {
      if (!global || call.kind != frame_identity_kind::unavailable) return call;
      if (record.camera_frame.kind == frame_identity_kind::v2_observed_token && record.camera_serial < serial &&
          token_current(state, record.camera_frame)) return record.camera_frame;
      return {};
    }
    // The frame DLSS-G evaluates for an FG viewport: the viewport's newest,
    // begun by its constants (store_camera), whose own FG tag copy alone keeps
    // DLSSG.Depth out (tags_supply_frame). Without its constants, or with
    // busy metadata, nothing marks it, and any tag copy younger than the
    // source age keeps DLSSG.Depth out, as every one did before 10-26.
    frame_generation::frame_mark evaluated_frame(std::uint32_t viewport) {
      auto pin = read_metadata();
      if (pin) for (const auto &record : pin->records)
        if (record.used && record.viewport == viewport && record.has_camera)
          return {frame_key(record.camera_frame), record.frame_serial};
      return {};
    }
    frame_identity observed_frame(std::uintptr_t address) {
      frame_identity result;
      auto pin = token_metadata.read();
      if (!pin) {
        if (token_metadata.has_publication()) dropped_observation(loss_diagnostics::reason::tokens_busy, __func__, 862);
        return result;
      }
      if (pin->observation != observation_generation.load(std::memory_order_acquire)) return result;
      for (const auto &token : pin->tokens) if (token.frame.token == address && address) { result = token.frame; break; }
      return result;
    }
    frame_record *frame_slot(metadata_state &state, std::uint32_t viewport, const frame_identity &identity, std::uint64_t serial, bool create) {
      if (identity.kind == frame_identity_kind::unavailable) return nullptr;
      for (auto &frame : state.frames) if (frame.used && frame.viewport == viewport && same_frame(frame.frame, identity)) return &frame;
      if (!create) return nullptr;
      auto *chosen = &state.frames[0];
      for (auto &frame : state.frames) {
        if (!frame.used) { chosen = &frame; break; }
        if (frame.changed < chosen->changed) chosen = &frame;
      }
      *chosen = {};
      chosen->used = true;
      chosen->viewport = viewport;
      chosen->frame = identity;
      chosen->changed = serial;
      return chosen;
    }
    const frame_record *find_frame(const metadata_state &state, std::uint32_t viewport, const frame_identity &identity) {
      for (const auto &frame : state.frames)
        if (frame.used && frame.viewport == viewport && same_frame(frame.frame, identity)) return &frame;
      return nullptr;
    }
    void store_camera(std::uint32_t viewport, const camera_data &camera, decode_status decoded,
        std::uint64_t frame, bool explicit_frame, std::uintptr_t token, std::uint64_t ticket,
        const frame_identity &identity, std::uint64_t serial, std::uint64_t loss, std::uint64_t entry_tick) {
      if (!current(ticket)) return;
      const bool invalid_camera = decoded != decode_status::ok || camera.reset > 1 ||
        (explicit_frame && camera.not_rendering_game_frames != 0) || !validate(camera).valid();
      auto details = loss_context;
      details.sequence = serial; details.viewport = viewport;
      details.reset = decoded == decode_status::ok ? camera.reset : UINT32_MAX;
      auto update = write_metadata(ticket);
      if (!update) {
        if (current(ticket)) { ++history_loss; metadata_write_lost(update, __func__, 1226, details); }
        return;
      }
      auto &state = *update;
      auto &record = slot(state, viewport);
      if (serial < record.camera_serial) return;
      // Invalid constants and resets are observed facts, not lost evidence: the
      // camera is optional for depth. A reset ends only this viewport's
      // temporal history; it never revokes depth or other viewports.
      if (!invalid_camera && camera.reset == 1) ++record.history;
      if (invalid_camera) record.steady_frames = 0;
      else {
        const auto checked = validate(camera);
        // Engines that rebuild the matrix every frame differ in the last bits.
        const auto close = [](double a, double b) { return std::abs(a - b) <= 1e-6 * std::max(std::abs(a), std::abs(b)); };
        const bool same = record.steady_frames && close(checked.depth_offset, record.steady_projection.depth_offset) &&
          close(checked.depth_scale, record.steady_projection.depth_scale);
        record.steady_projection = checked;
        record.steady_frames = same ? record.steady_frames + 1 : 1;
      }
      const auto camera_identity = identity.kind == frame_identity_kind::unavailable &&
          installed_abi == abi::v2 && token ?
        frame_identity{frame_identity_kind::v2_constants_call, serial, 0, token, false} : identity;
      record.camera = camera;
      record.decoded = decoded;
      record.key = {viewport, state.epoch, frame, explicit_frame};
      record.camera_token = token;
      record.camera_frame = camera_identity;
      record.camera_tick = entry_tick;
      record.camera_serial = serial;
      record.has_camera = true;
      record.changed = ++state.revision;
      // The viewport's frame begins with its constants: its enabled FG scope
      // expects the frame to record an FG copy (depth_capture::frame_generation_frame),
      // and DLSS-G's evaluation of it stands aside only for that frame's own
      // FG tag copy (evaluated_frame).
      const auto fg = read_fg_mode(viewport);
      const bool fg_frame = fg_available.load(std::memory_order_acquire) &&
        current_fg_target.load(std::memory_order_acquire) && fg.known && fg.enabled && fg.epoch == state.epoch &&
        fg.loss_revision == fg_loss_revision.load(std::memory_order_acquire);
      const auto fg_epoch = fg.epoch, fg_enabled_tick = fg.enabled_tick;
      const auto key = frame_key(camera_identity);
      const bool begins = frame_generation::frame_begins(record.frame_key, key);
      if (begins) record.frame_serial = serial;
      record.frame_key = key;
      if (token_current(state, camera_identity)) if (auto *saved = frame_slot(state, viewport, camera_identity, serial, true)) {
        if (serial >= saved->camera_serial) {
          saved->camera = camera;
          saved->decoded = decoded;
          saved->camera_serial = saved->changed = serial;
          saved->camera_tick = record.camera_tick;
          saved->loss = loss;
          saved->history = record.history;
          saved->has_camera = true;
        }
      }
      if (!current(ticket)) return;
      if (!update.commit()) { ++history_loss; metadata_write_lost(update, __func__, 1283, details); return; }
      if (fg_frame && begins) depth_capture::frame_generation_frame(fg_epoch, viewport, fg_enabled_tick);
    }
    void store_tags(batch &values) {
      if (!values.valid_viewport || !current(values.observation)) return;
      auto details = loss_context;
      details.sequence = values.serial; details.viewport = values.viewport;
      auto update = write_metadata(values.observation);
      if (!update) { if (current(values.observation)) metadata_write_lost(update, __func__, 1291, details); return; }
      auto &state = *update;
      auto &record = slot(state, values.viewport);
      // A null, replaced or unusable tag is itself observed and stored below;
      // later evaluations read it. Earlier evaluations and their finished
      // copies keep the tag they were given, so nothing is revoked here.
      for (unsigned i = 0; i < values.count; ++i) {
        auto value = values.tags[i];
        if (!value.present) continue;
        value.value.observation_epoch = state.epoch;
        value.tick = values.tick;
        value.serial = values.serial;
        value.loss = values.loss;
        value.frame = tag_frame(state, record, values.frame, value.source == origin::global, values.serial);
        auto &destination = value.source == origin::local ? record.local_tags : record.global_tags;
        destination[i] = value;
        if (i < record.depth_calls.size() && value.source != origin::local && value.supported && value.value.native_resource &&
            value.serial >= record.depth_calls[i].serial) record.depth_calls[i] = value;
        if (value.source == origin::global && value.serial >= record.active_tags[i].serial) record.active_tags[i] = value;
        if (value.source == origin::frame && token_current(state, values.frame)) {
          if (auto *saved = frame_slot(state, values.viewport, values.frame, values.serial, true)) {
            if (value.serial >= saved->tags[i].serial) saved->tags[i] = value;
            saved->changed = std::max(saved->changed, values.serial);
          }
        }
      }
      record.changed = ++state.revision;
      if (current(values.observation) && !update.commit()) metadata_write_lost(update, __func__, 1318, details);
    }
    evaluation_snapshot capture_evaluation(std::uint32_t viewport, std::uint32_t feature,
        const frame_identity &identity, void *commands, std::uint64_t ticket, std::uint64_t serial,
        std::uint64_t loss, const batch *local, const metadata_state *selected_state,
        bool tag_boundary) {
      evaluation_snapshot out;
      out.viewport = viewport;
      out.feature = feature;
      out.tag_boundary = tag_boundary;
      out.frame = identity;
      out.sequence = serial;
      out.command_buffer = reinterpret_cast<std::uintptr_t>(commands);
      out.loss_revision = loss;
      out.status = evidence_status::source_associated_evaluation;
      if (!current(ticket)) { out.status = evidence_status::inactive; return out; }
      // Read before the metadata, like loss: a later lost constants call
      // belongs to the next evaluation.
      const auto history_revision = history_loss.load(std::memory_order_acquire);
      auto pin = selected_state ? metadata_owner::read_pin{} : read_metadata(true);
      if (!selected_state && pin) selected_state = &*pin;
      if (!selected_state && metadata.has_publication()) { out.status = evidence_status::busy; return out; }
      // The first evaluation may provide local raw depth before any constants
      // or stored tags. It still owns its explicit inputs and minted identity.
      static const metadata_state empty_state;
      if (!selected_state) selected_state = &empty_state;
      const auto &state = *selected_state;
      if (!current(ticket) || (state.observation && state.observation != ticket)) { out.status = evidence_status::inactive; return out; }
      out.epoch = state.observation ? state.epoch : epoch.load(std::memory_order_acquire);
      static const viewport_record empty;
      const viewport_record *selected = &empty;
      for (const auto &candidate : state.records)
        if (candidate.used && candidate.viewport == viewport) { selected = &candidate; break; }
      const auto &record = *selected;
      const auto *saved = find_frame(state, viewport, identity);
      if (!token_current(state, identity)) out.status = evidence_status::untracked_frame;
      else if (!saved || !saved->has_camera) out.status = evidence_status::missing_constants;
      else {
        out.camera = saved->camera;
        out.camera_tick = saved->camera_tick;
        out.camera_sequence = saved->camera_serial;
        out.decoded = saved->decoded;
        out.frame_correlated = true;
        if (saved->loss != loss) out.status = evidence_status::observation_lost;
        else if (out.decoded != decode_status::ok) out.status = evidence_status::rejected_constants;
      }
      if (record.has_camera && record.decoded == decode_status::ok && record.camera.depth_inverted <= 1 &&
          record.camera.reset <= 1 && validate(record.camera).valid()) {
        const auto now = sunshine_game3d::clock::game_now();
        if (sunshine_freshness::fresh_lt(record.camera_tick, now, sunshine_freshness::source_age_ms)) {
          out.viewport_direction = true;
          out.viewport_inverted = record.camera.depth_inverted;
          if (record.steady_frames >= steady_projection_frames && record.steady_projection.valid()) {
            out.viewport_projection_ready = true;
            out.viewport_projection = record.steady_projection;
          }
        }
      }
      // Freeze history continuity with this camera/frame tuple before the native
      // evaluation. It ends at this viewport's reset, at a lost constants call
      // (which may have hidden a reset) or at an observation restart; other
      // transient losses cannot hide a reset. Both counters only increase.
      out.feedback.revision = history_revision + 1 + (out.frame_correlated ? saved->history : record.history);
      out.feedback.reset = out.frame_correlated && out.decoded == decode_status::ok && out.camera.reset == 1;
      bool has_depth = false;
      for (unsigned i = 0; i != out.tags.size(); ++i) {
        // A standalone tag call supplies its complete synchronous input.
        // Older global tags must not contaminate this call's loss/validity or
        // outrank its explicit candidates before capture_depth_tag returns.
        if (tag_boundary && local && !local->tags[i].present) continue;
        const auto *tag = &record.active_tags[i];
        if (saved && saved->tags[i].present) tag = &saved->tags[i];
        if (local && local->tags[i].present) tag = &local->tags[i];
        if (!tag->present) continue;
        const bool at_boundary = local && tag == &local->tags[i];
        auto &value = out.tags[i];
        value = {tag->value, tag->source == origin::local ? tag_scope::evaluation_local :
          tag->source == origin::frame ? tag_scope::explicit_frame : tag_scope::active_global, true, tag->supported, {}, {}};
        value.value.observation_epoch = out.epoch;
        value.observed_tick = at_boundary ? local->tick : tag->tick;
        value.observation_sequence = at_boundary ? serial : tag->serial;
        value.native_state = tag->native_state;
        value.state_requires_observation = tag->state_requires_observation;
        value.commanded = tag->commanded;
        value.direct_source = tag->direct_source;
        value.source_present_generation = tag->source_present_generation;
        // S7: this evaluation names the frame its stored tag was set for.
        value.own_frame = !at_boundary && identity.kind == frame_identity_kind::v2_observed_token &&
          (tag->source == origin::frame ? saved && tag == &saved->tags[i] :
            tag->source == origin::global && same_frame(tag->frame, identity));
        // A stored untag: the viewport's newest tag call of this kind that
        // named a resource, when it came before the untag. A loss since that
        // call leaves it unsupported.
        if (const auto &call = record.depth_calls[i]; !at_boundary && tag->supported && !tag->value.native_resource &&
            call.present && call.serial < tag->serial) {
          auto &ended = value.ended;
          ended.present = true;
          ended.value = call.value;
          ended.value.observation_epoch = out.epoch;
          ended.scope = call.source == origin::frame ? tag_scope::explicit_frame : tag_scope::active_global;
          ended.supported = call.supported && call.loss == loss;
          ended.state_requires_observation = call.state_requires_observation;
          ended.commanded = call.commanded;
          ended.native_state = call.native_state;
          ended.direct_source = call.direct_source;
          ended.source_present_generation = call.source_present_generation;
          ended.own_frame = identity.kind == frame_identity_kind::v2_observed_token && same_frame(call.frame, identity);
        }
        value.precision_scale = tag->precision_scale;
        value.precision_bias = tag->precision_bias;
        const auto tag_loss = at_boundary ? local->loss : tag->loss;
        if (tag_loss != loss) {
          if (out.status == evidence_status::source_associated_evaluation) out.status = evidence_status::observation_lost;
          value.supported = false;
        }
        if (value.supported && value.value.native_resource) has_depth = true;
      }
      pin.reset();
      if (out.frame_correlated) {
        out.projection = validate(out.camera);
        if (out.decoded != decode_status::ok) out.status = evidence_status::rejected_constants;
        else if (out.camera.reset == 1 && out.projection.valid() &&
            out.status == evidence_status::source_associated_evaluation) out.status = evidence_status::camera_reset;
        else if (!out.projection.valid() || out.camera.reset > 1 ||
            (identity.kind == frame_identity_kind::v1_numeric && out.camera.not_rendering_game_frames != 0))
          out.status = evidence_status::invalid_camera;
        else if (out.status == evidence_status::source_associated_evaluation && !has_depth) out.status = evidence_status::missing_depth;
      }
      if (out.status == evidence_status::source_associated_evaluation && loss != loss_revision.load(std::memory_order_acquire))
        out.status = evidence_status::observation_lost;
      return out;
    }
    void finish_evaluation(evaluation_snapshot value, bool success, std::uint64_t ticket) {
      if (!current(ticket)) return;
      auto details = loss_context;
      details.sequence = value.sequence; details.viewport = value.viewport; details.feature = value.feature;
      details.reset = value.frame_correlated && value.decoded == decode_status::ok ? value.camera.reset : UINT32_MAX;
      if (!success) {
        lose(loss_diagnostics::reason::sdk_failure, __func__, 1486, details);
      }
      // The game never repeats an evaluation: a writer busy for a moment
      // (another thread's camera or FG update) must not lose it (RE
      // Requiem 10-22 lost one as records_busy).
      auto update = write_metadata(ticket, true);
      if (!update) { if (current(ticket)) metadata_write_lost(update, __func__, 1494, details); return; }
      if (current(ticket)) {
        auto &state = *update;
        auto &record = slot(state, value.viewport);
        // Lifecycle2 ends at this return even when the feature failed. Strong
        // COM references do not extend pixel validity. Never consume a newer
        // concurrent tag or turn an evaluation-local tag into a global one.
        for (unsigned i = 0; i != value.tags.size(); ++i) {
          // A tag call ends no UntilEvaluate tag, a frame-generation one
          // included: the slEvaluateFeature call that uses it does (its
          // same-frame evaluate copies a tag its call could not, S11c). A
          // mixed batch may carry that tag beside the OnlyValidNow copy.
          if (value.tag_boundary) break;
          const auto &used = value.tags[i];
          if (!used.present || used.value.lifecycle != sl_contract::valid_until_evaluate ||
              used.scope == tag_scope::evaluation_local) continue;
          tag_record *original = nullptr;
          if (used.scope == tag_scope::active_global) original = &record.active_tags[i];
          else if (auto *saved = frame_slot(state, value.viewport, value.frame, 0, false)) original = &saved->tags[i];
          if (original && original->present && original->serial == used.observation_sequence) {
            original->supported = false;
            original->direct_source.reset();
          }
        }
        if (success) {
          record.changed = ++state.revision;
        }
      }
      if (current(ticket) && !update.commit()) metadata_write_lost(update, __func__, 1527, details);
    }
    bool viewport_value(const abi_v2::viewport &value, std::uint32_t &output) {
      abi_v2::viewport copy{};
      constexpr auto prefix = offsetof(abi_v2::viewport, value) + sizeof(copy.value);
      return read_bytes(&value, &copy, prefix) && decode_v2_viewport(&copy, prefix, output) == decode_status::ok;
    }
    // An evaluate's inputs may form one chain ("the chained structures
    // providing the input data (viewport, tags, constants etc)",
    // sl_core_api.h slEvaluateFeature; SLA:250-262). There a node's next may
    // be the next input, which is never an extension of the node: a
    // viewport, a tag or constants (the matrix's S15).
    bool evaluate_input(const void *node) {
      base_structure base;
      return node && read(static_cast<const base_structure *>(node), base) &&
        (same(base.type, viewport_guid) || same(base.type, tag_guid) || same(base.type, constants_guid));
    }
    // Streamline structures follow sl_struct.h's versioning rule: a newer
    // version only appends fields, so a structure with the expected GUID and
    // version 1 or later is read by its version-1 fields alone, never past them.
    // The only extension a tag may chain is PrecisionInfo, which chains none;
    // in an evaluate's input chain (input_chain) its next may be the next
    // input.
    tag_refusal decode_precision(const void *extension, double &scale, double &bias, bool input_chain) {
      scale = 1; bias = 0;
      if (!extension) return tag_refusal::none;
      precision_info precision{};
      constexpr auto prefix = offsetof(precision_info, scale) + sizeof(precision.scale);
      if (!read_bytes(extension, &precision, prefix)) return tag_refusal::unreadable;
      if (!same(precision.base.type, precision_guid)) return tag_refusal::extension;
      if (precision.base.version < 1) return tag_refusal::header_version;
      if (precision.base.next && !(input_chain && evaluate_input(precision.base.next))) return tag_refusal::extension;
      if (precision.formula == 0) return tag_refusal::none;
      if (precision.formula != 1 || !std::isfinite(precision.scale) || !std::isfinite(precision.bias) || precision.scale == 0)
        return tag_refusal::precision;
      scale = precision.scale; bias = precision.bias;
      return tag_refusal::none;
    }
    void append_tag(batch &out, const abi_v2::resource_tag &tag, origin source, std::uintptr_t token, bool report_invalid) {
      const auto type = buffers::canonical(installed_contract, tag.type);
      const unsigned index = tag_index(type);
      if (index == observed_tag_types.size()) return;
      abi_v2::resource resource{};
      const bool header_ok = same(tag.base.type, tag_guid) && tag.base.version >= 1;
      constexpr auto resource_prefix = offsetof(abi_v2::resource, height) + sizeof(resource.height);
      bool resource_ok = header_ok && (!tag.resource_ptr || read_bytes(tag.resource_ptr, &resource, resource_prefix));
      // nativeFormat follows height in every typed 2.x Resource; the
      // zero-header variant below declares no layout, so it reads as 0.
      if (resource_ok && tag.resource_ptr && resource.base.next == nullptr && same(resource.base.type, resource_guid) &&
          resource.base.version >= 1)
        resource_ok = read_bytes(&tag.resource_ptr->native_format, &resource.native_format, sizeof(resource.native_format));
      if (!resource_ok) {
        if (report_invalid) invalid_observation(loss_diagnostics::reason::invalid_input, __func__, 1579);
        resource = {};
      }
      tag_record value{};
      value.present = true;
      // An evaluate's input (origin::local) may be chained: as for the
      // viewport (read_local_inputs), a next that is the next input is no
      // extension of the tag, and a PrecisionInfo there is the tag's own (S15).
      const bool input_chain = source == origin::local;
      const auto precision = decode_precision(input_chain && evaluate_input(tag.base.next) ? nullptr : tag.base.next,
        value.precision_scale, value.precision_bias, input_chain);
      const bool precision_ok = precision == tag_refusal::none;
      // Some 2.x integrations zero the whole Resource, header included, then
      // set its fields, while constructing a correctly typed ResourceTag. No
      // SDK constructor leaves that header (sl_struct.h), and a later version
      // only appends members, so the fields are read by version 1's layout
      // (the contract matrix's S14). Accept that one layout variant, not
      // arbitrary GUIDs/versions or extensions. The native
      // IID/device/description checks still belong to the shared owner.
      const guid empty_guid{};
      const bool zero_resource_header = installed_abi == abi::v2 && resource.base.next == nullptr &&
        same(resource.base.type, empty_guid) && resource.base.version == 0;
      const bool resource_header = resource.base.next == nullptr &&
        ((same(resource.base.type, resource_guid) && resource.base.version >= 1) ||
          zero_resource_header);
      // eTex2d, or eUnknown where the installed release declares it (S12).
      value.supported = resource_ok && precision_ok && sl_contract::defined_lifecycle(tag.lifecycle) && (!tag.resource_ptr ||
        (resource_header && (resource.type == 0 || (resource.type == 8 && installed_contract.unknown_resource_type))));
      value.value.refusal = value.supported ? tag_refusal::none : !header_ok ? tag_refusal::header_version :
        !resource_ok ? tag_refusal::unreadable : !precision_ok ? precision :
        !sl_contract::defined_lifecycle(tag.lifecycle) ? tag_refusal::lifecycle :
        !resource_header ? (resource.base.next ? tag_refusal::extension : tag_refusal::header_version) :
        tag_refusal::resource_type;
      value.value.native_format = value.supported ? resource.native_format : 0;
      value.source = source;
      value.token = token;
      value.value.native_resource = value.supported ? reinterpret_cast<std::uintptr_t>(resource.native) : 0;
      value.value.viewport = out.viewport;
      value.value.type = type;
      value.value.lifecycle = tag.lifecycle;
      value.value.area = tag.area;
      value.value.resource_width = resource.width;
      value.value.resource_height = resource.height;
      value.native_state = resource.state;
      // Its nonzero state is a declaration, as a typed Resource's (S14); a 0
      // there may be the zeroing rather than COMMON, so it takes the shared
      // observed-state path. A typed Resource's 0 declares COMMON.
      value.state_requires_observation = zero_resource_header && resource.state == 0;
      if (value.supported && index < 3)
        value.direct_source = depth_capture::retain_source(value.value.native_resource);
      value.source_present_generation = depth_capture::source_present_generation(value.direct_source);
      // An opaque FrameToken address can be reused. Its identity is diagnostic
      // only and must never be promoted to an exact numeric frame identifier.
      value.value.explicit_frame = false;
      report_tag_resource(out, tag, source, value, resource, index, zero_resource_header);
      if (!value.supported && index < 3) {
        static std::atomic<std::uint64_t> next_tag_report{};
        report_rejected_tag(out, tag, value, resource, precision_ok, resource_ok, next_tag_report);
      }
      out.tags[index] = value;
      out.count = std::max(out.count, index + 1);
    }
    batch read_tags(const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags,
        std::uint32_t count, origin source, std::uintptr_t token) {
      batch out;
      out.valid_viewport = viewport_value(viewport, out.viewport);
      if (!out.valid_viewport) {
        invalid_observation(loss_diagnostics::reason::invalid_input, __func__, 1689); return out;
      }
      loss_context.viewport = out.viewport;
      std::array<abi_v2::resource_tag, tag_chunk> copy{};
      for (std::uint32_t first = 0; first < count; first += tag_chunk) {
        const auto size = std::min<std::uint32_t>(count - first, tag_chunk);
        // A ResourceTag above version 1 may append fields, so past its first
        // element the array's stride is unknown: such a multi-tag call is read
        // as unreadable. One tag, and tags chained in evaluate inputs, are read.
        if (!read_bytes(tags + first, copy.data(), size * sizeof(copy[0])) ||
            (!first && count > 1 && same(copy[0].base.type, tag_guid) && copy[0].base.version > 1)) {
          invalid_observation(loss_diagnostics::reason::invalid_input, __func__, 1700);
          // A successful call with an unreadable/unsupported tag array must not
          // leave an older resource, or a tag of this call read before the
          // unreadable element, looking current for this known viewport.
          out.count = static_cast<unsigned>(out.tags.size());
          for (unsigned i = 0; i < out.count; ++i) {
            out.tags[i] = {};
            out.tags[i].present = true;
            out.tags[i].source = source;
            out.tags[i].value.viewport = out.viewport;
            out.tags[i].value.type = observed_tag_types[i];
          }
          return out;
        }
        for (unsigned i = 0; i < size; ++i) append_tag(out, copy[i], source, token);
      }
      return out;
    }
    std::int32_t hook_constants_v2(const abi_v2::constants &values, const abi_v2::frame_token &frame, const abi_v2::viewport &viewport) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const bool sample = ticket != 0;
      const auto serial = sample ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial);
      const auto loss = loss_revision.load(std::memory_order_acquire);
      const auto identity = sample ? observed_frame(reinterpret_cast<std::uintptr_t>(&frame)) : frame_identity{};
      const auto entry_tick = sample ? sunshine_game3d::clock::game_now() : 0;
      auto diagnostic = dump_stamp(ticket, serial, entry_tick, UINT32_MAX, reinterpret_cast<std::uintptr_t>(&frame), identity.numeric, identity.has_numeric);
      camera_data camera;
      decode_status decoded = decode_status::short_buffer;
      std::uint32_t id{};
      bool valid_viewport = false;
      if (sample) {
        ++constant_calls;
        abi_v2::constants copy{};
        constexpr auto prefix = offsetof(abi_v2::constants, motion_vectors_jittered) + 1;
        const bool readable = read_bytes(&values, &copy, prefix);
        if (readable) decoded = decode_v2_constants(&copy, prefix, camera);
        valid_viewport = viewport_value(viewport, id);
        loss_context.viewport = valid_viewport ? id : UINT32_MAX;
        diagnostic.viewport = valid_viewport ? id : UINT32_MAX;
        if (diagnostic.session) {
          const bool tail_read = decoded == decode_status::ok && copy.base.version >= 2 &&
            read_bytes(reinterpret_cast<const unsigned char *>(&values) + offsetof(abi_v2::constants, min_relative_linear_depth_object_separation),
              &copy.min_relative_linear_depth_object_separation, sizeof(copy.min_relative_linear_depth_object_separation));
          dump_metadata::observe_sl_constants(diagnostic, copy, readable, decoded, tail_read);
        }
      }
      SetLastError(incoming);
      const auto result = original_constants_v2(values, frame, viewport);
      const DWORD outgoing = GetLastError();
      const bool sdk_ok = sdk_accepted(result);
      loss_context.sdk_result = result; loss_context.has_sdk_result = true;
      if (diagnostic.session) dump_metadata::finish_sl_call(diagnostic, sdk_ok);
      if (sample && sdk_ok && valid_viewport)
        store_camera(id, camera, decoded, 0, false, reinterpret_cast<std::uintptr_t>(&frame), ticket, identity, serial, loss, entry_tick);
      else if (sample && current(ticket))
        lose(!sdk_ok ? loss_diagnostics::reason::sdk_failure : loss_diagnostics::reason::invalid_input, __func__, 2474);
      SetLastError(outgoing);
      return result;
    }
    evaluation_snapshot capture_depth_tag(const batch &out, void *commands) {
      evaluation_snapshot snapshot;
      if (!out.valid_viewport || !source_requested.load(std::memory_order_acquire) ||
          (!out.tags[0].present && !out.tags[1].present && !out.tags[2].present)) return snapshot;
      frame_identity identity = out.frame;
      auto pin = read_metadata(true);
      if (!pin && metadata.has_publication()) return snapshot;
      // The viewport's FG role: its committed options (the FG options store).
      const auto fg = read_fg_mode(out.viewport);
      const bool enabled = fg_available.load(std::memory_order_acquire) && current_fg_target.load(std::memory_order_acquire) &&
        fg.known && fg.enabled && fg.epoch == epoch.load(std::memory_order_acquire) &&
        fg.loss_revision == fg_loss_revision.load(std::memory_order_acquire);
      if (pin) for (const auto &record : pin->records) if (record.used && record.viewport == out.viewport) {
        const auto &first_tag = out.tags[0].present ? out.tags[0] : out.tags[1].present ? out.tags[1] : out.tags[2];
        const bool synchronous_global_depth = std::any_of(out.tags.begin(), out.tags.begin() + 3, [](const tag_record &tag) {
          return tag.present && tag.source == origin::global && tag.value.lifecycle == sl_contract::only_valid_now;
        });
        const auto *camera_frame = find_frame(*pin, out.viewport, record.camera_frame);
        if (identity.kind == frame_identity_kind::unavailable && first_tag.source == origin::global &&
            (record.camera_frame.kind != frame_identity_kind::v2_constants_call || synchronous_global_depth) &&
            record.has_camera && sunshine_freshness::fresh_le(record.camera_tick, out.tick, sunshine_freshness::camera_age_ms) &&
            camera_frame && camera_frame->camera_serial < out.serial && camera_frame->loss == out.loss &&
            token_current(*pin, record.camera_frame))
          identity = record.camera_frame;
        break;
      }
      if (!enabled && std::none_of(out.tags.begin(), out.tags.begin() + 3, [](const tag_record &tag) {
          return tag.present && tag.value.lifecycle == sl_contract::only_valid_now;
        })) return snapshot;
      snapshot = capture_evaluation(out.viewport, enabled ? 1000 : UINT32_MAX, identity, commands,
        out.observation, out.serial, out.loss, &out, pin ? &*pin : nullptr, true);
      snapshot.tag_boundary = true;
      snapshot.frame_generation_input = enabled;
      if (snapshot.frame_correlated && (snapshot.camera_sequence >= out.serial ||
          !sunshine_freshness::fresh_le(snapshot.camera_tick, out.tick, sunshine_freshness::camera_age_ms))) {
        snapshot.status = evidence_status::stale;
        snapshot.frame_correlated = false;
        if (!enabled) {
          // A stale camera cannot calibrate this call, but the call still owns
          // its explicit OnlyValidNow depth. Keep that raw-only fallback free
          // of stale camera, jitter and SDK frame claims; the viewport's depth
          // direction and steady coefficients are not frame claims.
          snapshot.frame = {}; snapshot.camera = {}; snapshot.projection = {};
          snapshot.decoded = decode_status::short_buffer;
          snapshot.camera_tick = snapshot.camera_sequence = 0;
          snapshot.feedback.reset = false;
          snapshot.status = evidence_status::untracked_frame;
        }
      }
      return snapshot;
    }
    // slSetTag and slSetTagForFrame differ only in their frame token and tag
    // scope. Capture, UI admission, loss accounting and completion are shared.
    template<class Forward>
    std::int32_t observe_tag_call(const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags, std::uint32_t count,
        void *commands, const abi_v2::frame_token *frame, Forward &&forward) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const bool sample = ticket != 0;
      const auto serial = sample ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial);
      const auto loss = loss_revision.load(std::memory_order_acquire);
      const auto token = reinterpret_cast<std::uintptr_t>(frame);
      const auto identity = sample && frame ? observed_frame(token) : frame_identity{};
      const auto entry_tick = sample ? sunshine_game3d::clock::game_now() : 0;
      const auto diagnostic = dump_stamp(ticket, serial, entry_tick, UINT32_MAX, token, identity.numeric, identity.has_numeric,
        reinterpret_cast<std::uint64_t>(commands));
      if (sample && diagnostic.session) dump_metadata::observe_sl_tags_v2(diagnostic, viewport, tags, count,
        frame ? dump_metadata::tag_scope::frame : dump_metadata::tag_scope::global);
      batch out;
      if (sample) {
        ++tag_calls; out = read_tags(viewport, tags, count, frame ? origin::frame : origin::global, token);
        for (auto &tag : out.tags) tag.commanded = commands != nullptr;
      }
      out.observation = ticket;
      out.serial = serial;
      out.loss = loss;
      out.frame = identity;
      out.tick = entry_tick;
      deferred_uis ui_later;
      const auto alpha = sample ? capture_ui_inputs(out, commands, &ui_later) : alpha_attempts{};
      const auto snapshot = sample ? capture_depth_tag(out, commands) : evaluation_snapshot{};
      if (snapshot.tag_boundary) loss_context.feature = snapshot.feature;
      deferred_depth depth_later;
      const auto captured = snapshot.tag_boundary ? nominate_depth_source(snapshot, false, nullptr, &depth_later) : 0;
      SetLastError(incoming);
      const std::int32_t result = forward();
      const DWORD outgoing = GetLastError();
      const bool sdk_ok = sdk_accepted(result);
      loss_context.sdk_result = result; loss_context.has_sdk_result = true;
      if (diagnostic.session) dump_metadata::finish_sl_call(diagnostic, sdk_ok);
      finish_capture(captured, sdk_ok, ticket);
      if (sample && sdk_ok && out.count) store_tags(out);
      else if (sample && (!sdk_ok || !out.valid_viewport) && current(ticket))
        lose(!sdk_ok ? loss_diagnostics::reason::sdk_failure : loss_diagnostics::reason::invalid_input, __func__, 2857);
      if (snapshot.tag_boundary) finish_evaluation(snapshot, sdk_ok, ticket);
      const bool ui_kept = sdk_ok && current(ticket) && loss == loss_revision.load(std::memory_order_acquire);
      finish_ui_inputs(alpha, ui_kept);
      if (sample && sdk_ok && current(ticket) && out.valid_viewport) defer_tag_call(out, frame != nullptr, depth_later, ui_later, ui_kept);
      SetLastError(outgoing);
      return result;
    }
    std::int32_t hook_tag_v2(const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags, std::uint32_t count, void *commands) {
      return observe_tag_call(viewport, tags, count, commands, nullptr,
        [&] { return original_tag_v2(viewport, tags, count, commands); });
    }
    std::int32_t hook_frame_tag_v2(const abi_v2::frame_token &frame, const abi_v2::viewport &viewport,
        const abi_v2::resource_tag *tags, std::uint32_t count, void *commands) {
      return observe_tag_call(viewport, tags, count, commands, &frame,
        [&] { return original_frame_tag_v2(frame, viewport, tags, count, commands); });
    }
    batch read_local_inputs(const base_structure **inputs, std::uint32_t count, std::uintptr_t token, bool report_invalid) {
      batch out;
      const auto reject = [&] {
        out.valid_viewport = false;
        if (report_invalid) invalid_observation(loss_diagnostics::reason::invalid_input, __func__, 2878);
      };
      // Visits every node of every input, in order: the inputs a chunk at a
      // time, each chain to its end (numInputs and a chain have no limit,
      // S18). A chain that returns to a node it passed is malformed (Brent's
      // cycle check, with no length cap). failed: a read or the visitor
      // refused a node.
      enum class walked { all, failed, cyclic };
      const auto walk = [&](auto &&visit) {
        std::array<const base_structure *, tag_chunk> roots{};
        for (std::uint32_t first = 0; first < count; first += tag_chunk) {
          const auto size = std::min<std::uint32_t>(count - first, tag_chunk);
          if (!read_bytes(inputs + first, roots.data(), size * sizeof(roots[0]))) return walked::failed;
          for (unsigned i = 0; i < size; ++i) {
            const void *mark{};
            std::uint64_t steps{}, span{1};
            for (const auto *current = roots[i]; current;) {
              base_structure base;
              if (!read(current, base) || !visit(current, base)) return walked::failed;
              current = static_cast<const base_structure *>(base.next);
              if (current && current == mark) return walked::cyclic;
              if (++steps == span) { mark = current; steps = 0; span *= 2; }
            }
          }
        }
        return walked::all;
      };
      // The first walk finds the viewport and keeps the first tag_chunk tags;
      // a later walk appends the rest, once the viewport is known.
      std::array<abi_v2::resource_tag, tag_chunk> tags{};
      std::uint64_t tag_count = 0;
      bool viewport_refused = false;
      const auto first_walk = walk([&](const base_structure *current, const base_structure &base) {
        if (same(base.type, viewport_guid)) {
          abi_v2::viewport viewport{};
          std::uint32_t id{};
          constexpr auto prefix = offsetof(abi_v2::viewport, value) + sizeof(viewport.value);
          // In an evaluate input list 'next' is the documented input chain,
          // which the walk traverses. It does not change the viewport
          // identifier's known prefix layout.
          viewport_refused = !read_bytes(current, &viewport, prefix);
          viewport.base.next = nullptr;
          viewport_refused = viewport_refused || decode_v2_viewport(&viewport, sizeof(viewport), id) != decode_status::ok ||
            (out.valid_viewport && out.viewport != id);
          if (viewport_refused) return false;
          out.viewport = id;
          out.valid_viewport = true;
          loss_context.viewport = id;
        } else if (same(base.type, tag_guid)) {
          abi_v2::resource_tag later{};
          if (!read_bytes(current, tag_count < tags.size() ? &tags[tag_count] : &later, sizeof(later))) return false;
          ++tag_count;
        }
        return true;
      });
      if (first_walk != walked::all) {
        if (viewport_refused || first_walk == walked::cyclic) loss_context.viewport = UINT32_MAX;
        reject(); return out;
      }
      if (!out.valid_viewport) return out;
      for (std::uint64_t i = 0; i < std::min<std::uint64_t>(tag_count, tags.size()); ++i)
        append_tag(out, tags[i], origin::local, token, report_invalid);
      if (tag_count > tags.size()) {
        std::uint64_t seen = 0;
        const auto later_walk = walk([&](const base_structure *current, const base_structure &base) {
          if (!same(base.type, tag_guid) || seen++ < tags.size()) return true;
          abi_v2::resource_tag later{};
          if (!read_bytes(current, &later, sizeof(later))) return false;
          append_tag(out, later, origin::local, token, report_invalid);
          return true;
        });
        // The game changed its inputs during its own call.
        if (later_walk != walked::all) reject();
      }
      return out;
    }
    std::int32_t hook_new_token_v2(abi_v2::frame_token *&token, const std::uint32_t *frame_index) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const auto serial = ticket ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial);
      std::uint32_t numeric{};
      const bool index_ok = !ticket || !frame_index || read(frame_index, numeric);
      SetLastError(incoming);
      const auto result = original_new_token_v2(token, frame_index);
      const DWORD outgoing = GetLastError();
      const bool sdk_ok = sdk_accepted(result);
      loss_context.sdk_result = result; loss_context.has_sdk_result = true;
      if (current(ticket)) {
        abi_v2::frame_token *returned{};
        if (!sdk_ok || !index_ok || !read(&token, returned) || !returned) {
          invalid_observation(!sdk_ok ? loss_diagnostics::reason::sdk_failure : loss_diagnostics::reason::invalid_input,
            __func__, 3053);
        } else {
          const auto address = reinterpret_cast<std::uintptr_t>(returned);
          // Requesting the same caller-supplied index again returns that frame's
          // token: it is the same frame, so it keeps its identity and the tags and
          // constants already recorded for it still reach its evaluation. The
          // optional numeric index is caller-owned; absent means unknown, never zero.
          const auto same_frame = [&](const frame_identity &frame) {
            return frame_index && frame.token == address && frame.has_numeric && frame.numeric == numeric;
          };
          // Such a repeat changes nothing, so it takes no writer. The Witcher 3
          // repeats each request about 30 times a frame from several threads.
          const bool repeated = frame_index && [&] {
            const auto pin = token_metadata.read();
            if (!pin || pin->observation != ticket) return false;
            for (const auto &candidate : pin->tokens)
              if (candidate.frame.token == address) return same_frame(candidate.frame);
            return false;
          }();
          if (!repeated) {
            auto update = write_tokens_briefly();
            if (!update) {
              if (current(ticket)) dropped_observation(update.failure() == observation::write_failure::storage_busy ?
                loss_diagnostics::reason::metadata_storage_busy : loss_diagnostics::reason::tokens_busy, __func__, 3076);
            } else if (current(ticket)) {
              update->observation = ticket;
              auto *chosen = &update->tokens[0];
              for (auto &candidate : update->tokens) {
                if (candidate.frame.token == address) { chosen = &candidate; break; }
                if (!candidate.frame.token || candidate.frame.generation < chosen->frame.generation) chosen = &candidate;
              }
              // A recycled address is a NEW observer generation. The same frame
              // can still arrive here when its repeat raced the read above.
              if (!same_frame(chosen->frame)) {
                if (chosen->frame.token != address || serial > chosen->frame.generation)
                  chosen->frame = {frame_identity_kind::v2_observed_token, serial, numeric, address, frame_index != nullptr};
                else lose(loss_diagnostics::reason::token_replaced, __func__, 3089); // Concurrent out-of-order allocation of the same address.
              }
              if (current(ticket) && !update.commit())
                dropped_observation(loss_diagnostics::reason::metadata_storage_busy, __func__, 3092);
            }
          }
        }
      }
      SetLastError(outgoing);
      return result;
    }
  }

  bool query_ui_scope(std::uint32_t viewport, ui_observation_scope &output, std::uint32_t max_age_ms) {
    output = {};
    const auto ticket = observation_ticket();
    if (!ticket) return false;
    auto pin = read_metadata();
    if (!pin) return false;
    const auto loss = loss_revision.load(std::memory_order_acquire);
    const auto now = sunshine_game3d::clock::game_now();
    bool found = false, ambiguous = false;
    for (const auto &record : pin->records) {
      if (!record.used || (viewport != UINT32_MAX && record.viewport != viewport)) continue;
      const tag_record *newest{};
      // A UI tag, or a local UI input of an evaluate (U3), names the scope.
      for (const auto *group : {&record.global_tags, &record.local_tags}) for (const auto index : ui_priority) {
        const auto &tag = (*group)[index];
        if (!tag.present || !tag.supported || !tag.serial || tag.loss != loss ||
            tag.value.observation_epoch != pin->epoch || !tag.tick || !sunshine_freshness::fresh_lt(tag.tick, now, max_age_ms)) continue;
        if (!newest || tag.serial > newest->serial) newest = &tag;
      }
      if (!newest) continue;
      if (found) { ambiguous = true; break; }
      output = {pin->epoch, loss, newest->serial, newest->tick, record.viewport};
      found = true;
    }
    pin.reset();
    if (!found || ambiguous || !current(ticket) || loss != loss_revision.load(std::memory_order_acquire)) {
      output = {}; return false;
    }
    return true;
  }

  namespace {
    // query_frame_generation over one copy of the FG options store: store, a
    // copy the caller read (take_fg's one read per Present), or null to read
    // the store here.
    bool read_frame_generation(const fg_mode_table *store, std::uint32_t viewport, frame_generation_snapshot &output,
        frame_generation_query_status *status) {
      output = {};
      if (status) *status = frame_generation_query_status::unavailable;
      const auto ticket = observation_ticket();
      if (!ticket || !fg_available.load(std::memory_order_acquire) || !current_fg_target.load(std::memory_order_acquire)) return false;
      // The FG options store: every read gets the last committed options.
      fg_mode_table read;
      if (!store) { read = read_fg_modes(); store = &read; }
      const auto &modes = *store;
      bool found = false, ambiguous = false;
      const frame_generation_snapshot *off = nullptr;
      const auto options_loss = fg_loss_revision.load(std::memory_order_acquire);
      const auto source_epoch = epoch.load(std::memory_order_acquire);
      for (const auto &slot : modes) {
        const auto &state = slot.state;
        if (!slot.used || slot.observation != ticket || !state.sequence || (viewport != UINT32_MAX && state.viewport != viewport)) continue;
        if (!state.known || state.epoch != source_epoch || state.loss_revision != options_loss) { ambiguous = true; continue; }
        // A viewport's confirmed Off binds no FG scope: beside another
        // viewport's enabled options it leaves that one the FG scope, and
        // with none enabled the read is the Off of the lowest viewport id.
        // Until 11-01 any two viewports' options made the any-viewport read
        // ambiguous, so a game that sets Off on its second SR viewport each
        // frame never had a confirmed FG scope: no tie-break of its two
        // active views (owner ruling R5), no FG input and mono for as long
        // as both rendered (R5-3). Then, until R5-3's fix review, the all-Off
        // read was the newest Off: every in-order call rewrites its slot's
        // sequence, so with Off set on two viewports each frame it named
        // either one by where the Present's read fell between the calls, and
        // each flip moved the UI's FG viewport, which reset its source
        // qualification and detection scope.
        if (!state.enabled) {
          if (!off || state.viewport < off->viewport) off = &state;
          continue;
        }
        if (found) { ambiguous = true; continue; }
        output = state; found = true;
      }
      if (!found && off) { output = *off; found = true; }
      if (!found || ambiguous || !current(ticket) || options_loss != fg_loss_revision.load(std::memory_order_acquire)) {
        if (status && ambiguous) *status = frame_generation_query_status::ambiguous;
        output = {}; return false;
      }
      if (status) *status = frame_generation_query_status::observed;
      return true;
    }
  }

  bool query_frame_generation(std::uint32_t viewport, frame_generation_snapshot &output,
      frame_generation_query_status *status) {
    return read_frame_generation(nullptr, viewport, output, status);
  }
}
