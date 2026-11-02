// SPDX-License-Identifier: GPL-3.0-only
// Present ordinals and frame-generation scope; functions-only part of streamline_depth_capture.cpp.
// Types, namespace-scope state and test variables stay in the shell.

namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    void miss_invalidated_frame(const slot &value, capture_failure why) {
      // An FG copy recorded on a COMPUTE list (DLSS-G's own input, or an FG
      // tag's: S22, N7) whose producer can no longer supply its frame (its
      // recording discarded unsubmitted, as when its queue has no place,
      // replayed, or its queue gone) leaves that frame without an FG copy,
      // exactly as frame_generation_missed records: an accepted copy that
      // never reaches a consumer keeps no SR depth out.
      const auto &m = value.metadata;
      if (value.compute_list && value.source_nominated && m.frame_generation_input &&
          m.provider == provider_kind::streamline && m.epoch && m.sequence &&
          why != capture_failure::consumer_signal_failed && why != capture_failure::consumer_capacity &&
          why != capture_failure::consumer_queue_changed)
        for (auto &view : evaluations[provider_index(m.provider)].views)
          if (view.epoch == m.epoch && view.viewport == m.viewport && view.source_id == m.source_id)
            view.missed_sequence = std::max(view.missed_sequence, m.sequence);
    }

    std::uint64_t present_ordinal() { return scope.presents.load(std::memory_order_acquire) + 1; }

    // Requires mutex. Whether a stamp is the frame the Present being decided
    // presents: no other Present came since its call (above), or, for a
    // frame that made a frame-generation call (its DLSS-G call is what adds
    // Presents for it), one of the `generated` Presents after its real one, as
    // long as no later evaluation began. With FG on DLSS-G presents a frame's generated
    // Presents and its real one in either order, so a hitch between a
    // frame's evaluation and its last Present made every Present after its
    // first mono (inactive_views, retained_depth_expired; the 10-27 review's
    // F1 and F2, open items 1 and 2), and the first Present after the pause
    // without depth restarted the stereo fade.
    // The frame's stamps run from its first frame-generation stamp
    // (frame_generation_present) to its anchor, the first Present after its
    // last submission seen (frame_generation_anchor), and its window from
    // its first stamp to `generated` Presents after the later of the two.
    // DLSS-G presents earlier frames on its own thread while the game records
    // this one (Assassin's Creed Shadows inside its SR evaluation), so up to
    // `generated` Presents of the earlier frame come after this frame's stamp
    // and before its real Present. Until 10-31 the window counted from the
    // stamp, so those Presents used it up: after a hitch before its real
    // Present, its own generated Presents went mono (MFG 4x: three, a 2D flash
    // right after the freeze; soak seed d40ba4d5b7c456ee at frame 126), and
    // an evaluation of the frame after them (its SR evaluation) ended its
    // window as a later frame's would. An evaluation that began after the
    // anchor is the next frame's and ends the window, so a game that keeps
    // presenting without evaluating reaches the source age at the
    // (generated + 2)-th Present after its anchor, as before.
    bool frame_presenting(std::uint64_t ordinal) {
      if (!ordinal) return false;
      const auto n = scope.presents.load(std::memory_order_acquire);
      if (ordinal >= n) return true;
      const auto first = scope.frame_generation_present, generated = std::uint64_t(scope.presenting_generated);
      const auto last = std::max(first, scope.frame_generation_anchor);
      return generated && first && ordinal >= first && ordinal <= last &&
        scope.newest_evaluation_present <= last && n <= last + generated;
    }

    // Requires mutex. The presenting floor (acquisition_decision): while this
    // Present still presents the newest evaluated frame (frame_presenting),
    // the first stamp of that frame when the newest evaluation is one of its
    // stamps (from its first FG stamp to its anchor: its own evaluations up to
    // the anchor, as frame_presenting counts them), else the newest
    // evaluation's ordinal; 0 otherwise. A copy of that frame stamped at its
    // call (presented_by: no submission on the reader seen) before the
    // frame's own later evaluation is that frame's own (own_frame). Until
    // 11-01 the floor was the newest evaluation's ordinal, so after the
    // earlier frame's generated Presents and the frame's own SR evaluation
    // its call-stamped FG copy lost its hold after a hitch (F63).
    std::uint64_t presenting_floor() {
      const auto newest = scope.newest_evaluation_present;
      if (!frame_presenting(newest)) return 0;
      const auto first = scope.frame_generation_present;
      return scope.presenting_generated && first && newest >= first &&
        newest <= std::max(first, scope.frame_generation_anchor) ? first : newest;
    }

    // Requires mutex. A frame-generation stamp at ordinal (an FG input's
    // evaluation began, or its copy was published): the newest FG frame's
    // own while it is no later than that frame's stamps and anchor (its FG
    // inputs that DLSS-G's evaluation copies at its Present), else the first
    // stamp of the next FG frame, which has no anchor yet.
    void stamp_frame_generation(std::uint64_t ordinal) {
      if (ordinal <= std::max(scope.frame_generation_present, scope.frame_generation_anchor)) return;
      scope.frame_generation_present = ordinal;
      scope.frame_generation_anchor = 0;
    }

    // Requires mutex. ordinal: the first Present after a submission of the
    // frame stamped at call (a copy's producer list on a presenting queue,
    // slot::submitted_present), or after DLSS-G's evaluation of the newest
    // FG frame (frame_generation_evaluated). The frame's real Present comes
    // at or after it: it anchors that frame's window (frame_presenting) when
    // call is one of its stamps and at most `generated` Presents (the earlier
    // frame's that DLSS-G may still present) came since its first. A later
    // frame's submission or evaluation is further away, so it never extends
    // an earlier frame's window, nor reopens a window a later evaluation
    // ended.
    void anchor_frame_generation(std::uint64_t call, std::uint64_t ordinal) {
      const auto first = scope.frame_generation_present;
      if (!first || call < first || call > std::max(first, scope.frame_generation_anchor) ||
          ordinal > first + scope.presenting_generated) return;
      scope.frame_generation_anchor = std::max(scope.frame_generation_anchor, ordinal);
    }

    // The present ordinal of the frame a copy is: its submission's on a
    // presenting queue when seen (slot::submitted_present), else its call's.
    std::uint64_t presented_by(const slot &value) {
      return value.submitted_present ? value.submitted_present : value.present;
    }

    // Requires mutex. The viewport's FG retirement in epoch, or null.
    fg_retirement *fg_retirement_of(std::uint64_t epoch, std::uint32_t viewport) {
      if (!epoch) return nullptr;
      for (auto &mark : scope.fg_retirements) if (mark.epoch == epoch && mark.viewport == viewport) return &mark;
      return nullptr;
    }

    // Requires mutex. While FG Off keeps a viewport's FG role retired (no FG
    // On since), no input is the viewport's FG input: a copy of a frame from
    // before the Off that reaches the capture owner after it (a deferred FG
    // tag copy made at a late Off's Present, DLSS-G's evaluation racing the
    // options call) is the frame's own depth of the display lineage, so it
    // is recorded and evaluated as the viewport's ordinary source (source
    // 0), and every evaluation of the viewport supersedes its FG view. Until
    // 10-28 such a copy became the viewport's FG view again beside its SR
    // view: every Present read two active views (ambiguous_views), moved the
    // gap watermark over each readable SR copy and held the stale FG copy
    // until it expired (the 10-28 review's P1).
    bool frame_generation_role_retired(provider_kind provider, std::uint64_t epoch, std::uint32_t viewport) {
      if (provider != provider_kind::streamline) return false;
      const auto *mark = fg_retirement_of(epoch, viewport);
      return mark && mark->active;
    }

    // Requires mutex. value's metadata with its FG role retired: the
    // viewport's ordinary source.
    void retire_frame_generation_role(input &value) {
      if (value.frame_generation_input && frame_generation_role_retired(value.provider, value.epoch, value.viewport)) {
        value.frame_generation_input = false;
        value.source_id = 0;
      }
    }

    // Requires mutex. A depth copy recorded on a COMPUTE list is its FG
    // scope's accepted copy (frame_generation_scope_live) only once the
    // Present queue can read it: its SDK call succeeded, its submission
    // carries the capture owner's fence, it was copied out on its compute
    // queue (or, without a copy-out, its recording was reset) and that fence
    // passed (completed_snapshot's rule for another queue's copy). The scope
    // then turns live with readable FG pixels, so its handover drops no SR
    // depth for a copy still in flight. Streamline resets its DLSS-G
    // recording frames later, which before copy-out (10-27) kept each such
    // copy unreadable for three of its evaluations. One never submitted, as
    // when its queue has no place, or never readable keeps no other depth out
    // of its viewport. Called at the copy's call, its submission and each
    // liveness query (frame_generation_scope_live).
    // Compute-copy acceptance needs a producer fence, but does not test nomination_invalid.
    void accept_compute_copy(const slot &value) {
      const auto &m = value.metadata;
      if (!value.compute_list || !value.source_nominated || !value.finished || !value.success || value.invalid ||
          value.nomination_only || value.preservation_only || !value.producer_submitted || !value.producer_fence ||
          !valid_provider(m.provider))
        return;
      auto &views = evaluations[provider_index(m.provider)].views;
      const auto owned = [&](const auto &view) {
        return view.epoch == m.epoch && view.viewport == m.viewport && view.source_id == m.source_id;
      };
      if (std::none_of(views.begin(), views.end(), [&](const auto &view) {
            return owned(view) && view.accepted_sequence < m.sequence; }) ||
          !immutable_for_other_queues(value))
        return;
      // Readable on the queue that reads depth now (U-C1: ordered there by the
      // game's own fences, or complete).
      if (!readable_elsewhere(value, reader_queue(), producer_progress(value))) return;
      for (auto &view : views)
        if (owned(view)) {
          view.accepted_tick = std::max(view.accepted_tick, m.tick);
          view.accepted_sequence = std::max(view.accepted_sequence, m.sequence);
        }
    }

    std::uint64_t scope_now() { return sunshine_game3d::clock::game_now(); }

    // Requires mutex. The FG scope's view when the scope is live
    // (frame_generation_scope_live), else null.
    evaluation_head *live_frame_generation_scope(std::uint64_t epoch, std::uint32_t viewport,
        std::uint64_t enabled_tick) {
      const auto now = scope_now();
      const auto source = sunshine_scene_depth::fg_source_id(viewport);
      // A compute-list copy of this scope counts once readable (accept_compute_copy).
      if (compute_copies_recorded)
        for (const auto &value : capture_pool.slots)
          if (value.id && value.compute_list && value.metadata.epoch == epoch && value.metadata.viewport == viewport &&
              value.metadata.source_id == source)
            accept_compute_copy(value);
      // A tick slightly ahead of this read (another thread's) is fresh.
      for (auto &view : evaluations[provider_index(provider_kind::streamline)].views)
        if (view.epoch == epoch && view.viewport == viewport && view.source_id == source && view.accepted_tick &&
            view.accepted_tick >= enabled_tick && view.accepted_sequence > view.missed_sequence &&
            sunshine_freshness::ahead_tolerant_le(view.accepted_tick, now, sunshine_freshness::fg_silent_ms))
          return &view;
      return nullptr;
    }

    // Requires mutex. value, a copy just recorded, supplies its FG scope's
    // current frame (frame_generation_frame).
    void record_frame_generation_copy(const input &value) {
      if (!value.frame_generation_input || !valid_provider(value.provider)) return;
      for (auto &view : evaluations[provider_index(value.provider)].views)
        if (view.epoch == value.epoch && view.viewport == value.viewport && view.source_id == value.source_id) {
          view.recorded_sequence = std::max(view.recorded_sequence, value.sequence);
          view.awaiting = false;
        }
    }
  }

  bool frame_generation_scope_live(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t enabled_tick) {
    if (!requested.load() || !epoch) return false;
    std::lock_guard lock(mutex);
    return live_frame_generation_scope(epoch, viewport, enabled_tick) != nullptr;
  }

  void frame_generation_enabled(std::uint64_t enabled_tick) {
    scope.frame_generation_on_moves.store(reader_moves.load(std::memory_order_relaxed), std::memory_order_relaxed);
    scope.frame_generation_on_tick.store(enabled_tick, std::memory_order_relaxed);
    if (!enabled_tick) scope.frame_generation_run_tick.store(0, std::memory_order_relaxed);
  }

  void frame_generation_activity(std::uint64_t now) {
    const auto last = scope.frame_generation_activity_tick.exchange(now, std::memory_order_relaxed);
    const auto on = scope.frame_generation_on_tick.load(std::memory_order_relaxed);
    if (!last || now < last || now - last >= sunshine_freshness::fg_silent_ms || (on && on >= last)) {
      scope.frame_generation_run_moves.store(reader_moves.load(std::memory_order_relaxed), std::memory_order_relaxed);
      scope.frame_generation_run_tick.store(now, std::memory_order_relaxed);
    }
  }

  void frame_generation_frame(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t enabled_tick) {
    if (!requested.load() || !epoch) return;
    frame_generation_activity(sunshine_game3d::clock::game_now());
    std::lock_guard lock(mutex);
    auto *view = live_frame_generation_scope(epoch, viewport, enabled_tick);
    if (!view || !std::exchange(view->awaiting, true)) return;
    // No copy recorded since the previous frame began: that frame had no FG
    // copy. Every copy recorded so far, accepted or still in flight, is of
    // an earlier frame (frame_generation_missed's ordering).
    view->missed_sequence = std::max({view->missed_sequence, view->recorded_sequence, view->accepted_sequence});
    view->awaiting = false;
  }

  void frame_generation_missed(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t sequence) {
    if (!requested.load() || !epoch || !sequence) return;
    std::lock_guard lock(mutex);
    const auto source = sunshine_scene_depth::fg_source_id(viewport);
    for (auto &view : evaluations[provider_index(provider_kind::streamline)].views)
      if (view.epoch == epoch && view.viewport == viewport && view.source_id == source)
        view.missed_sequence = std::max(view.missed_sequence, sequence);
  }

  void frame_generation_restored(std::uint64_t epoch, std::uint32_t viewport) {
    if (!requested.load() || !epoch) return;
    std::lock_guard lock(mutex);
    if (auto *mark = fg_retirement_of(epoch, viewport)) mark->active = false;
    // The FG scope's copies the Off kept as the viewport's ordinary source
    // (retire_source) never show again: an FG Off/On transition never
    // resurrects the scope before it.
    for (auto &value : capture_pool.slots)
      if (value.id && value.retired_fg_copy && value.metadata.epoch == epoch && value.metadata.viewport == viewport)
        invalidate(value, capture_failure::source_retired);
  }

  bool frame_generation_retired(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t serial) {
    if (!requested.load() || !epoch) return false;
    std::lock_guard lock(mutex);
    const auto *mark = fg_retirement_of(epoch, viewport);
    return mark && (mark->active || (serial && serial <= mark->serial));
  }

  std::uint64_t game_executions() { return capture_queues.game_execution_count.load(std::memory_order_acquire); }

  std::uint64_t next_present_ordinal() { return present_ordinal(); }

  void evaluation_called(std::uint64_t epoch) {
    if (!requested.load() || !epoch) return;
    std::lock_guard lock(mutex);
    scope.newest_evaluation_present = std::max(scope.newest_evaluation_present, present_ordinal());
  }

  void frame_generation_evaluated(std::uint64_t epoch) {
    if (!requested.load() || !epoch) return;
    std::lock_guard lock(mutex);
    anchor_frame_generation(scope.frame_generation_present, present_ordinal());
  }

  void present(std::uint64_t native, std::uint32_t generated_presents) {
    // Counted for every Present, also one the capture owner does not serve
    // (frame_presenting must never take a frame older than the last Present
    // for the current one).
    scope.presents.fetch_add(1, std::memory_order_acq_rel);
    // Its thread is the present role of the capture lock's waits.
    lock_counters.present_thread.store(GetCurrentThreadId(), std::memory_order_relaxed);
    if (!requested.load() || !native) return;
    std::lock_guard lock(mutex);
    // The Presents DLSS-G adds per real frame, from this Present's FG
    // snapshot: one grant per Present, which every frame_presenting until the
    // next Present reads (the selection of this Present, a copy's record,
    // FG Off's retire_source on the options thread). Until 11-01 each
    // acquisition wrote it from its own runtime's policy (select_provider), so
    // another runtime's acquisition overwrote the FG frame's grant (PD8).
    scope.presenting_generated = generated_presents;
    const auto *owner = queue(native);
    if (!owner) return;
    const auto device = device_cookie(owner->device.p);
    if (!device) return;
    for (const auto &weak : capture_pool.sources) if (const auto source = weak.lock())
      if (source->device_identity == device) ++source->present_generation;
  }
}
