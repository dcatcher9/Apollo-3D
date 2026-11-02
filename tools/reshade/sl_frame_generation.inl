// SPDX-License-Identifier: GPL-3.0-only
// Part of streamline_camera_probe.cpp (sl_adapter): FG options, deferred tag
// copies and the shared DLSS-G input wrappers. The UI capture part is called
// at the original UI statement boundaries. The shell keeps every type,
// namespace-scope variable and thread_local;
// this function-only part is included at its end in the same translation unit.

namespace sunshine_streamline {
  SUNSHINE_PROBE_INTERNAL {
    template<unsigned Index> std::int32_t hook_fg_options(const abi_v2::viewport &viewport, const fg_options_prefix &options) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      // A getter wrapper may have loaded the native entry just before the
      // deferred detour was enabled. Its nested entry must only forward, not
      // publish the same SDK call twice. No TLS object needs destruction.
      const bool sample = !fg_forwarding_depth && ticket && fg_available.load(std::memory_order_acquire) &&
        current_fg_target.load(std::memory_order_acquire) == fg_targets[Index].entry;
      const auto serial = sample ? ++call_sequence : 0;
      fg_options_prefix copy{};
      std::uint32_t id{};
      const bool valid_viewport = sample && viewport_value(viewport, id);
      const bool valid = valid_viewport && read(&options, copy) &&
        same(copy.base.type, fg_options_guid) && copy.base.version >= 1 &&
        !copy.base.next && copy.mode <= 3 && (!copy.mode || copy.mode == 3 || copy.generated_frames != 0);
      SetLastError(incoming);
      const auto options_loss = fg_loss_revision.load(std::memory_order_acquire);
      ++fg_forwarding_depth;
      const auto result = fg_targets[Index].original.load(std::memory_order_acquire)(viewport, options);
      --fg_forwarding_depth;
      const DWORD outgoing = GetLastError();
      // Only the sampling entry judges (and counts) the result.
      const bool sdk_ok = sample && sdk_accepted(result);
      std::uint64_t retired_epoch{}, turned_on{}, turned_on_epoch{};
      // The retiring call observed FG off (a successful mode 0 call), not
      // a failed or unknown one (retire_source's confirmed_off).
      bool confirmed_off{};
      if (sample) dump_metadata::observe_sl_fg(dump_stamp(ticket, serial, sunshine_game3d::clock::game_now(), valid_viewport ? id : UINT32_MAX), copy.mode, copy.generated_frames, valid, sdk_ok);
      bool changed = false;
      // The FG options store (presentation_scope_mode.inl), whose only writer
      // of options this is: the write cannot be lost to the camera and tag
      // writers, so a game that sets its options once is never left with an
      // unknown mode. The commit and its transition entries below run under
      // fg_transition_lock, so two options calls' transitions follow their
      // commits' order.
      const bool transition = sample && current(ticket);
      if (transition) {
        AcquireSRWLockExclusive(&fg_transition_lock);
        AcquireSRWLockExclusive(&fg_mode_lock);
        if (current(ticket)) {
          const auto source_epoch = epoch.load(std::memory_order_acquire);
          if (!valid_viewport) {
            for (auto &slot : fg_modes) {
              changed |= slot.used && slot.observation == ticket && slot.state.known;
              slot = {};
            }
          } else {
            auto &state = fg_mode_slot_of(id, ticket).state;
            if (serial >= state.sequence) {
              const bool known = valid && sdk_ok;
              changed = !state.sequence || state.known != known ||
                (known && (state.mode != copy.mode || state.generated_frames != copy.generated_frames));
              // Games repeat their options every frame: FG stays enabled since
              // the call that turned it on in this epoch and options revision.
              const auto enabled_since = state.known && state.enabled && state.epoch == source_epoch &&
                state.loss_revision == options_loss ? state.enabled_tick : 0;
              // And an Off (or a failed call) that repeats the committed one in
              // this epoch and options revision retires nothing new: the
              // first one retired the FG role, which stays retired until FG On,
              // and every input since is the viewport's ordinary source. Until
              // 11-01 each repeated Off ran retire_source again under the
              // capture lock (once per frame in a game played with FG off).
              const bool retired_already = state.sequence && !state.enabled && state.known == known &&
                state.epoch == source_epoch && state.loss_revision == options_loss;
              const auto now = sunshine_game3d::clock::game_now();
              state = {source_epoch, serial, now, options_loss, id, copy.mode, copy.generated_frames,
                known, known && copy.mode != 0, valid && copy.mode >= 2};
              if (state.enabled) state.enabled_tick = enabled_since ? enabled_since : now;
              if (state.enabled && !enabled_since) { turned_on = now; turned_on_epoch = source_epoch; }
              if ((!known || copy.mode == 0) && !retired_already) { retired_epoch = source_epoch; confirmed_off = known; }
            }
          }
        }
        ReleaseSRWLockExclusive(&fg_mode_lock);
        if (!current(ticket)) retired_epoch = turned_on = 0;
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
        if (fg_transition_seam) fg_transition_seam();
#endif
      }
      // The one FG options-transition entry (docs/reshade-sbs.md, Presentation
      // scope). After the store write (fg_mode_lock released), still under
      // fg_transition_lock, in this order: frame_generation_enabled,
      // frame_generation_restored, revoke_evaluation_supply, retire_source,
      // ui_mask::invalidate_scope or invalidate_all; then, after the lock,
      // the "Sunshine Streamline frame generation:" line. Until 11-01 the
      // entries ran after the store's lock with nothing ordering them: an Off
      // whose retire_source got the capture lock after a newer On's
      // frame_generation_restored left the FG role retired while the store
      // said On, and every later On repeats the mode, so nothing restored it
      // until the next FG Off-to-On toggle (F81).
      // FG's first Present moves Present to DLSS-G's queue after this frame's
      // copies on the game's queue were submitted: those are copied out for
      // the source age from here (U-C1b, depth_capture::frame_generation_enabled).
      if (turned_on) depth_capture::frame_generation_enabled(turned_on);
      // FG On ends the viewport's FG role retirement (retire_source), and
      // DLSS-G's supply from before it supplies nothing of the new period.
      if (turned_on) depth_capture::frame_generation_restored(turned_on_epoch, id);
      if (turned_on) revoke_evaluation_supply(turned_on_epoch, id, turned_on);
      // Revoke the exact FG scope, including tags recorded while the failed
      // SDK call was in flight. A concurrent newer On commits and restores
      // after this (fg_transition_lock).
      if (retired_epoch) depth_capture::retire_source(sunshine_scene_depth::provider_kind::streamline,
        retired_epoch, sunshine_scene_depth::fg_source_id(id), serial, confirmed_off);
      // Only an actual FG-mode/knowledge transition revokes UI scope. Games
      // can call unsupported Off options after every successful HUDless tag;
      // repeated failure is not a new failure of those independent input tags.
      if (sample && changed && current(ticket) && (!valid || !sdk_ok || retired_epoch)) {
        if (valid_viewport) sunshine_game3d::ui_mask::invalidate_scope(epoch, id);
        else sunshine_game3d::ui_mask::invalidate_all();
      }
      if (transition) ReleaseSRWLockExclusive(&fg_transition_lock);
      if (changed) {
        char text[256]{};
        std::snprintf(text, sizeof(text), "Sunshine Streamline frame generation: viewport=%u mode=%u generated_frames=%u supported=%u success=%u sdk_result=%d",
          id, copy.mode, copy.generated_frames, valid, sdk_ok, static_cast<int>(result));
        message(text);
      }
      SetLastError(outgoing);
      return result;
    }
    // report_deferrals is defined in sl_report.inl.
    // Requires evaluation_supply_lock. Recounts the used deferral entries.
    void count_deferred() {
      unsigned used = 0;
      for (const auto &entry : deferred_frames) used += entry.used;
      deferral_pending.store(used, std::memory_order_release);
    }
    // The kinds a deferral entry holds, for the precedence handed back when
    // they are dropped: its depth (bit 0) and its UI capture slots (bit 1 +
    // slot).
    unsigned deferred_kinds(const deferred_frame &entry) {
      unsigned kinds = entry.depth.count ? 1u : 0u;
      for (unsigned slot = 0; slot != entry.ui.size(); ++slot) if (entry.ui[slot].used) kinds |= 2u << slot;
      return kinds;
    }
    unsigned kind_count(unsigned kinds) {
      unsigned count = 0;
      for (; kinds; kinds &= kinds - 1) ++count;
      return count;
    }
    // A dropped deferral no longer keeps its kinds' precedence over DLSS-G's
    // own inputs (tag_copy, ui_tag_copy): this frame has no tag copy, so its
    // FG scope owns no depth for it either (miss_frame_generation). sequence:
    // the frame's newest tag call.
    void hand_over(std::uint64_t scope_epoch, std::uint32_t viewport, unsigned kinds, std::uint64_t sequence) {
      if (kinds & 1u) {
        frame_missed(scope_epoch, viewport, sequence, evaluation_supplies(scope_epoch, viewport));
      }
      for (unsigned slot = 0; slot != 4; ++slot)
        if (kinds & (2u << slot)) remember_ui_tag_copy(scope_epoch, viewport, slot, false);
    }
    // After a 2.x tag call the SDK accepted: keep its copies deferred to its
    // frame's Present (R2), and let its untags cancel the viewport's deferred
    // copies of their frame (R7, api_contract.h untag_cancels). tokened: the
    // call named a frame token (slSetTagForFrame). ui_current: its UI inputs
    // were admitted under the current observation.
    void defer_tag_call(const batch &out, bool tokened, deferred_depth &depth, deferred_uis &ui, bool ui_current) {
      namespace mask = sunshine_game3d::ui_mask;
      if (!ui_current) for (auto &value : ui) value = {};
      const bool keeps_ui = std::any_of(ui.begin(), ui.end(), [](const deferred_ui &value) { return value.used; });
      // An untag: the call's depth tags all name no native (as depth_untag), or
      // a UI kind's tag names none.
      bool untags_depth = false, depth_named = false;
      for (unsigned i = 0; i != 3; ++i) if (out.tags[i].present) {
        if (out.tags[i].supported && !out.tags[i].value.native_resource) untags_depth = true;
        else depth_named = true;
      }
      untags_depth = untags_depth && !depth_named;
      unsigned ui_untags = 0;
      for (unsigned i = 3; i != out.tags.size(); ++i) {
        const auto &tag = out.tags[i];
        const auto kind = static_cast<mask::source_kind>(tag.value.type);
        if (tag.present && tag.supported && !tag.value.native_resource && mask::source_mask(kind))
          ui_untags |= 2u << mask::capture_slot(kind);
      }
      if (!depth.count && !keeps_ui && !untags_depth && !ui_untags) return;
      // The game's work executed before the frame's Present proves the
      // sources' states there (the declaration must agree).
      for (std::size_t i = 0; i != depth.count; ++i) depth_capture::track_deferred_source(depth.candidates[i].source);
      for (const auto &item : ui) if (item.used) depth_capture::track_deferred_source(item.input.source);
      frame_identity frame;
      if (auto pin = read_metadata()) for (const auto &record : pin->records)
        if (record.used && record.viewport == out.viewport) { frame = tag_frame(*pin, record, out.frame, !tokened, out.serial); break; }
      const bool framed = frame.kind != frame_identity_kind::unavailable;
      const auto source_epoch = epoch.load(std::memory_order_acquire);
      const auto executions = depth_capture::game_executions();
      unsigned cancelled = 0, dropped = 0;
      std::uint64_t dropped_epoch{}, cancelled_serial{}, dropped_serial{};
      std::uint32_t dropped_viewport{};
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      deferred_frame *entry = nullptr;
      for (auto &value : deferred_frames) if (value.used && value.viewport == out.viewport) { entry = &value; break; }
      if (entry && (untags_depth || ui_untags) &&
          sl_contract::untag_cancels(tokened && framed, entry->frame.kind != frame_identity_kind::unavailable,
            same_frame(frame, entry->frame))) {
        cancelled = deferred_kinds(*entry) & ((untags_depth ? 1u : 0u) | ui_untags);
        cancelled_serial = entry->serial;
        if (cancelled & 1u) entry->depth = {};
        for (unsigned slot = 0; slot != entry->ui.size(); ++slot) if (cancelled & (2u << slot)) entry->ui[slot] = {};
      }
      if (depth.count || keeps_ui) {
        bool begins = false;
        // Another frame's deferral of the viewport lapses: its Present is past.
        if (entry && (entry->epoch != source_epoch || entry->observation != out.observation ||
            !(same_frame(entry->frame, frame) || (!framed && entry->frame.kind == frame_identity_kind::unavailable)))) {
          dropped = deferred_kinds(*entry); dropped_epoch = entry->epoch; dropped_viewport = entry->viewport;
          dropped_serial = entry->serial;
          *entry = {};
          entry = nullptr;
        }
        if (!entry) {
          // A free entry, or the viewport entry tagged longest ago (lapsed).
          for (auto &value : deferred_frames) if (!value.used) { entry = &value; break; }
          if (!entry) {
            entry = &deferred_frames[0];
            for (auto &value : deferred_frames) if (value.serial < entry->serial) entry = &value;
            dropped |= deferred_kinds(*entry); dropped_epoch = entry->epoch; dropped_viewport = entry->viewport;
            dropped_serial = entry->serial;
            *entry = {};
          }
          entry->used = true; entry->observation = out.observation; entry->epoch = source_epoch;
          entry->viewport = out.viewport; entry->frame = frame;
          begins = true;
        }
        // The frame's newest tag call: its submitted work (and under frame
        // generation DLSS-G's evaluation of it, while DLSS-G evaluated the
        // viewport's previous deferred frame) must follow it.
        entry->serial = out.serial; entry->tick = out.tick; entry->executions = executions;
        entry->evaluations = fg_evaluation.tag(source_epoch, out.viewport, out.serial, begins, entry->evaluations,
          depth_capture::next_present_ordinal());
        if (depth.count) { entry->depth = std::move(depth); deferrals.depth.fetch_add(1, std::memory_order_relaxed); }
        for (unsigned slot = 0; slot != ui.size(); ++slot) if (ui[slot].used) {
          entry->ui[slot] = std::move(ui[slot]);
          deferrals.ui.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (entry && !deferred_kinds(*entry)) *entry = {};
      count_deferred();
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
      if (cancelled) {
        deferrals.cancelled.fetch_add(kind_count(cancelled), std::memory_order_relaxed);
        hand_over(source_epoch, out.viewport, cancelled, cancelled_serial);
      }
      if (dropped) {
        deferrals.lapsed.fetch_add(kind_count(dropped), std::memory_order_relaxed);
        hand_over(dropped_epoch, dropped_viewport, dropped, dropped_serial);
      }
      report_deferrals();
    }
  }

  namespace {
    presentation_scope::present_fg take_fg(std::uint64_t swapchain) {
      presentation_scope::present_fg out;
      out.swapchain = swapchain;
      // One read of the FG options store gives the any-viewport read (so
      // depth and the UI) and each viewport's committed options
      // (present_deferred), and it is taken under fg_view_lock, so the
      // process's one view applies reads in store order: of two present
      // threads, the one that read later updates the view later. Until 11-01
      // the store was read twice, outside the lock: an options call between
      // the two reads gave present_deferred another mode than depth and the
      // UI, and a present thread could apply its older read after another
      // thread's newer one, which an ambiguous or unavailable next read kept.
      AcquireSRWLockExclusive(&fg_view_lock);
      const auto modes = read_fg_modes();
      read_frame_generation(&modes, UINT32_MAX, out.fg, &out.read);
      out.observer_active = source_enabled();
      const auto observation = observation_generation.load(std::memory_order_acquire);
      for (std::size_t i = 0; i != modes.size(); ++i)
        if (modes[i].used && modes[i].observation == observation) out.modes[i] = modes[i].state;
      out.depth = fg_state.update(out.read, out.fg, out.observer_active);
      out.generated_frames = fg_state.generated_frames();
      out.ui = fg_state.ui();
      ReleaseSRWLockExclusive(&fg_view_lock);
      return out;
    }
  }

  void take_present_fg(std::uint64_t swapchain) {
    present_snapshot = take_fg(swapchain);
  }

  void end_present_fg() {
    present_snapshot = {};
  }

  const presentation_scope::present_fg &present_frame_generation(std::uint64_t swapchain) {
    if (swapchain && present_snapshot.swapchain == swapchain) return present_snapshot;
    // A consumer with no snapshot taken for its swapchain's Present on this
    // thread reads its own, for this call only: it is not kept for a later
    // one. Until 11-01 it became the thread's snapshot, so where depth's
    // present events frame no Present (depth not registered: a launch that
    // asks for a restart, a legacy add-on) every later Present of that
    // swapchain on the thread read the first one, and the fallback counter
    // rose once (F68).
    fg_snapshot_fallbacks.fetch_add(1, std::memory_order_relaxed);
    fallback_snapshot = take_fg(swapchain);
    return fallback_snapshot;
  }

  std::uint64_t present_fg_fallbacks() { return fg_snapshot_fallbacks.load(std::memory_order_relaxed); }

  void present_depth_frame(std::uint64_t swapchain, std::uint64_t frame) {
    if (swapchain && present_snapshot.swapchain == swapchain) present_snapshot.depth_frame = frame;
  }

  bool present_deferred(std::uint64_t swapchain, std::uint64_t queue, std::uint64_t immediate) {
    if (!deferral_pending.load(std::memory_order_acquire) || !queue || !immediate) return false;
    namespace mask = sunshine_game3d::ui_mask;
    const DWORD incoming = GetLastError();
    const auto ticket = observation_ticket();
    const auto executions = depth_capture::game_executions();
    const auto queue_executions = depth_capture::game_executions_on(queue);
    const auto now = sunshine_game3d::clock::game_now();
    // This Present's ordinal: it is counted after this call (on_present).
    const auto ordinal = depth_capture::next_present_ordinal();
    const auto source_epoch = epoch.load(std::memory_order_acquire);
    std::array<deferred_frame, slot_count> due;
    unsigned waited = 0;
    unsigned due_count = 0;
    struct ended_frame { std::uint64_t epoch{}, serial{}; std::uint32_t viewport{}; unsigned kinds{}; };
    std::array<ended_frame, slot_count> ended;
    unsigned ended_count = 0;
    // The viewports' FG options as this Present read them (its FG snapshot),
    // and the camera metadata only for whether a later frame began.
    const auto &current = present_frame_generation(swapchain);
    auto pin = read_metadata();
    AcquireSRWLockExclusive(&evaluation_supply_lock);
    for (auto &entry : deferred_frames) {
      if (!entry.used) continue;
      // Its frame's Present is past once the observation changed, the
      // viewport's constants named a later frame, or the source age passed
      // at a Present that cannot be the frame's own (expired: a hitch of the
      // source age between the frame's calls and its Present keeps it; until
      // 10-28 it lapsed there, and that Present was mono).
      bool past = !ticket || entry.observation != ticket || entry.epoch != source_epoch;
      // Whether the viewport generates frames, and how many per real one.
      // Until 11-01 a lost metadata read counted as generating with none
      // (PD5): an FG-off viewport's copy waited for the FG rule, and an MFG
      // frame's copy lapsed after a hitch at a Present still its own.
      const auto mode = current.mode(entry.viewport);
      const bool generating = mode.known && mode.enabled;
      const std::uint32_t generated = generating ? mode.generated_presents() : 0;
      if (!past && pin) for (const auto &record : pin->records) if (record.used && record.viewport == entry.viewport) {
        past = record.camera_serial > entry.serial && !same_frame(record.camera_frame, entry.frame);
        break;
      }
      past = past || fg_evaluation.expired(entry.epoch, entry.viewport, entry.evaluations, entry.tick, generated, ordinal, now);
      // Its frame's Present. Without frame generation the game's Present
      // reaches this one directly, so the first after a game list executed
      // following the tag (one with none since is an earlier frame's). With
      // it, earlier frames' Presents (paced or generated) come while the game
      // submits this frame's work on any queue: the first after DLSS-G
      // evaluated a frame following the tag, while DLSS-G evaluates every
      // frame in this scope; else the first after a game list executed on
      // this presenting queue since the tag (deferral_evaluations).
      const bool reached = fg_evaluation.reached(generating, entry.epoch, entry.viewport, entry.evaluations, entry.executions,
        executions, queue_executions, now, ordinal);
      if (!past && !reached) {
        waited += generating;
        continue;
      }
      if (past) ended[ended_count++] = {entry.epoch, entry.serial, entry.viewport, deferred_kinds(entry)};
      else due[due_count++] = std::move(entry);
      entry = {};
    }
    count_deferred();
    ReleaseSRWLockExclusive(&evaluation_supply_lock);
    pin.reset();
    if (waited) deferrals.waited.fetch_add(waited, std::memory_order_relaxed);
    for (unsigned n = 0; n != ended_count; ++n) {
      deferrals.lapsed.fetch_add(kind_count(ended[n].kinds), std::memory_order_relaxed);
      hand_over(ended[n].epoch, ended[n].viewport, ended[n].kinds, ended[n].serial);
    }
    bool recorded = false;
    // ReShade's own list records these copies; its list events never report it.
    if (due_count) depth_capture::observe_runtime_list(immediate, queue);
    for (unsigned n = 0; n != due_count; ++n) {
      auto &entry = due[n];
      const auto sequence = ++call_sequence;
      if (entry.depth.count) {
        // A refusal changes nothing: the earlier capture holds (S6), and the
        // FG observation stays as it was (pixels_required).
        depth_capture::record_diagnostic diagnostic;
        bool copied = false;
        // FG Off retired the FG role of the viewport's lineage after this
        // frame's tag call (a late Off) or keeps it off: the copy is the
        // frame's own depth of the lineage, its ordinary source (source 0,
        // superseding the FG view), and no FG scope's (the 10-28 review's
        // P1; until then it became the FG view again beside the SR view).
        const bool retired = entry.depth.candidates[0].frame_generation_input &&
          depth_capture::frame_generation_retired(entry.epoch, entry.viewport, entry.serial);
        const auto fg_key = sunshine_scene_depth::fg_source_id(entry.viewport);
        for (std::size_t i = 0; i != entry.depth.count && !copied; ++i) {
          auto value = entry.depth.candidates[i];
          // The copy is made here, at its frame's Present: its tick, from
          // which the source age of every later hold of it counts, is this
          // Present's. Until 11-01 it kept its tag call's, so a hitch or a
          // preempted game thread between the frame's calls and its Present
          // aged the held copy by that delay, and a later Present that could
          // only hold it went mono early.
          value.sequence = sequence; value.tick = now; value.present_queue = queue;
          value.own_frame = true; value.at_tag_call = false; value.pixels_required = true;
          if (retired) { value.frame_generation_input = false; value.source_id = 0; }
          const auto captured = depth_capture::nominate(immediate, value, &diagnostic);
          if (captured && diagnostic.result == depth_capture::status::recorded) {
            depth_capture::begin_evaluation(value.epoch, sequence, value.viewport, value.provider, value.source_id,
              retired ? fg_key : entry.depth.superseded);
            depth_capture::finish(captured, true);
            copied = true;
          } else if (captured) depth_capture::finish(captured, false);
        }
        // A refused copy is still this frame's call at its own Present: it
        // ends an earlier frame's presenting window (frame_presenting), as a
        // recorded copy's begin_evaluation does. DLSS-G's evaluation of the
        // frame stood aside for it and began nothing. Until 11-01 nothing
        // ended that window here, so where DLSS-G presented fewer of the
        // earlier frame's generated Presents than declared, a hitch of the
        // source age before this Present held the earlier frame's copy past
        // the source age as this frame's own (F2).
        if (!copied) depth_capture::evaluation_called(entry.epoch);
        if (!retired) {
          remember_tag_copy(entry.epoch, entry.viewport, copied, entry.serial, entry.frame);
          if (!copied) miss_frame_generation(entry.epoch, entry.viewport, entry.serial, evaluation_supplies(entry.epoch, entry.viewport));
        }
        if (copied) {
          recorded = true;
          deferrals.depth_copied.fetch_add(1, std::memory_order_relaxed);
        } else {
          deferrals.depth_refused.fetch_add(1, std::memory_order_relaxed);
          deferrals.refusal.store(unsigned(diagnostic.stage), std::memory_order_relaxed);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
          evaluation_snapshot refused;
          refused.epoch = entry.epoch; refused.sequence = sequence; refused.viewport = entry.viewport;
          refused.frame_generation_input = true;
          remember_refusal(refused, sunshine_depth_refusal::gate::capture, nullptr, &diagnostic);
#endif
        }
      }
      // One new observation sequence per tag call, in call order, so a call's
      // UI kinds stay one batch (exact HUD-less pairs) and no two calls merge.
      std::array<std::uint64_t, 4> calls{}, renumbered{};
      unsigned call_count = 0;
      for (const auto &item : entry.ui) if (item.used) {
        unsigned at = 0;
        while (at != call_count && calls[at] < item.input.sequence) ++at;
        if (at != call_count && calls[at] == item.input.sequence) continue;
        for (unsigned i = call_count; i != at; --i) calls[i] = calls[i - 1];
        calls[at] = item.input.sequence; ++call_count;
      }
      for (unsigned i = 0; i != call_count; ++i) renumbered[i] = i ? ++call_sequence : sequence;
      for (unsigned slot = 0; slot != entry.ui.size(); ++slot) {
        auto &item = entry.ui[slot];
        if (!item.used) continue;
        unsigned call = 0;
        while (calls[call] != item.input.sequence) ++call;
        item.input.sequence = renumbered[call];
        item.input.present_queue = queue; item.input.own_frame = true;
        item.where.source = item.input;
        item.where.command = immediate;
        const auto attempt = mask::begin(item.where, item.input);
        mask::finish(attempt, true);
        remember_ui_tag_copy(entry.epoch, entry.viewport, slot, static_cast<bool>(attempt.ticket), true);
        if (attempt.ticket) {
          recorded = true;
          deferrals.ui_copied.fetch_add(1, std::memory_order_relaxed);
        } else deferrals.ui_refused.fetch_add(1, std::memory_order_relaxed);
      }
    }
    report_deferrals();
    SetLastError(incoming);
    return recorded;
  }

  frame_generation::attempt observe_frame_generation_inputs(const frame_generation::inputs &value) {
    namespace mask = sunshine_game3d::ui_mask;
    using frame_generation::depth_outcome;
    using frame_generation::parameter;
    using frame_generation::ui_outcome;
    const DWORD incoming = GetLastError();
    frame_generation::attempt out;
    // An absent key changes nothing and is reported as absent in any scope.
    out.depth = value.depth.state == parameter::absent ? depth_outcome::absent :
      value.depth.state == parameter::null ? depth_outcome::null_resource : depth_outcome::not_offered;
    initialize_frame_generation_ui(value, out);
    // The scope: the one viewport whose FG options were observed enabled in
    // this epoch. Without it no input belongs to any Streamline FG scope.
    const auto ticket = observation_ticket();
    frame_generation_snapshot fg;
    if (!ticket || !query_frame_generation(UINT32_MAX, fg) || !fg.enabled) {
      if (out.depth == depth_outcome::not_offered) out.depth = depth_outcome::no_fg_scope;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      remember_frame_generation_refusal(out, 0, 0, UINT32_MAX);
#endif
      SetLastError(incoming);
      return out;
    }
    const auto loss = loss_revision.load(std::memory_order_acquire);
    const auto source_epoch = epoch.load(std::memory_order_acquire);
    if (fg.epoch != source_epoch || !current(ticket)) {
      if (out.depth == depth_outcome::not_offered) out.depth = depth_outcome::scope_changed;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      remember_frame_generation_refusal(out, fg.epoch, 0, fg.viewport);
#endif
      SetLastError(incoming);
      return out;
    }
    out.observation = ticket; out.loss = loss;
    // One observation sequence for every input of this evaluation, ordered
    // with every Streamline call (the capture owner's FG view and the UI
    // request's newest boundary compare it with tag sequences).
    const auto serial = ++call_sequence;
    const auto tick = sunshine_game3d::clock::game_now();
    // This frame's Present comes next (present_deferred).
    AcquireSRWLockExclusive(&evaluation_supply_lock);
    fg_evaluation.evaluated(fg.epoch, fg.viewport, tick, depth_capture::next_present_ordinal(), fg.generated_presents());
    ReleaseSRWLockExclusive(&evaluation_supply_lock);
    const auto source_id = sunshine_scene_depth::fg_source_id(fg.viewport);
    if (out.depth == depth_outcome::not_offered) {
      // Every evaluation of the scope that offered its depth is its frame's
      // call: one that makes no copy attempt (a malformed subrect, a camera
      // read lost again after the retry) ends an earlier frame's presenting
      // window as one that tries does (below; F2, 11-01). One whose frame's
      // tag copy waits for the frame's Present leaves that to the Present
      // (present_deferred).
      if (!value.depth.area_valid) {
        out.depth = depth_outcome::subrect;
        depth_capture::evaluation_called(fg.epoch);
      } else if (!source_requested.load(std::memory_order_acquire) || !depth_capture::active()) out.depth = depth_outcome::inactive;
      // The frame's own FG tag copied it first, or its copy waits for the
      // frame's Present: no second copy (a later frame without an FG tag of
      // its own is DLSS-G's).
      else if (bool recorded{}; tags_supply_frame(fg.epoch, fg.viewport, evaluated_frame(fg.viewport), &recorded)) {
        out.depth = depth_outcome::tag_copied;
        // Recorded at the frame's tag call: DLSS-G evaluates the frame inside
        // its Present, so its Presents begin at the next one, which anchors
        // its presenting window (frame_presenting) also where its copy's
        // submission on a presenting queue is not seen. A copy that waits for
        // the frame's Present is stamped there itself.
        if (recorded) depth_capture::frame_generation_evaluated(fg.epoch);
      }
      // The viewport's own camera evidence, exactly as for an unframed FG tag
      // (normalize_depth_frame): its history revision, its recent camera's
      // depth direction and its steady depth coefficients. No frame
      // correlation, jitter or reset pulse is claimed. Busy or lost metadata
      // refuses, as a tag boundary does, after a read that failed on every
      // attempt (read_metadata, which retries it briefly first).
      else if (const auto viewport = capture_evaluation(fg.viewport, 1000, frame_identity{},
                 reinterpret_cast<void *>(value.command), ticket, serial, loss, nullptr, nullptr, false);
               viewport.status == evidence_status::busy || viewport.status == evidence_status::inactive ||
               viewport.epoch != fg.epoch) {
        out.depth = depth_outcome::scope_changed;
        depth_capture::evaluation_called(fg.epoch);
      } else {
        // An evaluation that tries its own copy is its frame's call whatever
        // its outcome: it ends an earlier frame's presenting window
        // (frame_presenting), as a live scope's nomination does from its
        // begin. Until 10-31 a silent scope's refused attempt began none, so
        // with FG 4x declared the Present after a hitch held an earlier
        // frame's copy past the source age as its own, and whether the scope
        // was still live there followed the clock (soak seed
        // 4547e4874aef62bb, frame 80, stale under CPU load only).
        depth_capture::evaluation_called(fg.epoch);
        depth_capture::observe_command(value.command);
        depth_capture::input input;
        input.provider = sunshine_scene_depth::provider_kind::streamline;
        input.epoch = fg.epoch; input.sequence = serial; input.tick = tick; input.viewport = fg.viewport;
        input.observation_revision = loss; input.feedback = viewport.feedback;
        input.frame_generation_input = true; input.source_id = source_id;
        // Device depth (DLSS-G linearizes 1/(1-d), or 1/d when inverted).
        // DLSSG.DepthInverted names its direction; without it the viewport
        // camera's does. The steady coefficients apply only when both agree
        // (or only the camera names it): DLSSG.CameraNear/Far are not used.
        input.projection.encoding = sunshine_scene_depth::depth_encoding::device;
        const bool camera_inverted = viewport.viewport_inverted != 0;
        const bool agree = !value.inverted_known || !viewport.viewport_direction || value.inverted == camera_inverted;
        input.projection.direction_supplied = value.inverted_known || viewport.viewport_direction;
        input.projection.reversed = value.inverted_known ? value.inverted : viewport.viewport_direction && camera_inverted;
        if (agree && viewport.viewport_direction && viewport.viewport_projection_ready) {
          input.projection.depth_offset = viewport.viewport_projection.depth_offset;
          input.projection.depth_scale = viewport.viewport_projection.depth_scale;
          input.projection.supplied = true;
        }
        input.resource.native = value.depth.native;
        input.resource.area = value.depth.area;
        input.resource.kind = sunshine_scene_depth::resource_kind::raw_depth;
        // No API states this resource's state at the call: only a state
        // observed on the evaluation's own recording is acceptable.
        sunshine_api_contract::apply(sunshine_api_contract::ngx::frame_generation_input, input);
        input.at_tag_call = true; // Copied inside the call that supplied it.
        input.pixels_required = true;
        depth_capture::record_diagnostic diagnostic;
        input.source = depth_capture::retain_source(input.resource.native, &diagnostic);
        if (!input.source) {
          out.depth = depth_outcome::capture; out.depth_stage = diagnostic.stage;
        } else {
          input.source_present_generation = depth_capture::source_present_generation(input.source);
          // A live scope's attempt replaces its FG observation first, exactly
          // as an FG tag boundary does, so a refusal withdraws the observation,
          // never finished pixels: it is a missing frame, for which the scope's
          // newest unshown copy stands in (depth_capture::select_capture). A
          // silent scope's refusal changes nothing: only a recorded copy
          // becomes the viewport's FG observation (replacing its ordinary one).
          // Either way a refused frame has no FG copy, so the scope owns no
          // depth for it (frame_generation_missed).
          // A live scope's attempt is a nomination in progress from its begin.
          // An evaluation that began before FG Off retired the viewport's FG
          // role (DLSS-G's thread racing the game's options call) or while it
          // stays retired is the frame's own depth of the display lineage,
          // its ordinary source, superseding the FG view (the 10-28 review's
          // P1; until then it became the FG view again beside the SR view).
          const bool retired = depth_capture::frame_generation_retired(fg.epoch, fg.viewport, serial);
          if (retired) { input.frame_generation_input = false; input.source_id = 0; }
          const bool live = !retired && depth_capture::frame_generation_scope_live(fg.epoch, fg.viewport, fg.enabled_tick);
          depth_capture::evaluation_nomination nominating;
          if (live) nominating.begin(fg.epoch, serial, fg.viewport, sunshine_scene_depth::provider_kind::streamline, source_id, 0);
          const auto captured = nominating.nominate(value.command, input, &diagnostic);
          nominating.end();
          if (captured && diagnostic.result == depth_capture::status::recorded) {
            if (!live) depth_capture::begin_evaluation(fg.epoch, serial, fg.viewport,
              sunshine_scene_depth::provider_kind::streamline, input.source_id, retired ? source_id : 0);
            if (!retired) remember_evaluation_supply(fg.epoch, fg.viewport, tick);
            out.depth_ticket = captured; out.depth = depth_outcome::copied;
          } else {
            if (captured) depth_capture::finish(captured, false);
            if (!retired) depth_capture::frame_generation_missed(fg.epoch, fg.viewport, serial);
            out.depth = depth_outcome::capture; out.depth_stage = diagnostic.stage;
          }
        }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        remember_frame_generation_refusal(out, fg.epoch, serial, fg.viewport, diagnostic.result);
#endif
      }
    }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    if (out.depth != depth_outcome::capture) remember_frame_generation_refusal(out, fg.epoch, serial, fg.viewport);
#endif
    capture_frame_generation_ui(value, out, source_epoch, loss, fg, serial, tick, source_id);
    SetLastError(incoming);
    return out;
  }
  void finish_frame_generation_inputs(const frame_generation::attempt &value, bool successful) {
    if (!value.observation) return;
    const DWORD incoming = GetLastError();
    // A copy survives a successful SDK call; the UI inputs also need the
    // observation they were admitted under, as Streamline tags do.
    const bool still_current = current(value.observation);
    if (value.depth_ticket)
      depth_capture::finish(value.depth_ticket, successful && still_current, depth_capture::capture_failure::evaluation_failed);
    finish_frame_generation_ui(value, successful, still_current);
    SetLastError(incoming);
  }

}
