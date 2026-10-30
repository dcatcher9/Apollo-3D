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
    // Presents for it), one of the `generated` Presents after its first, as
    // long as no later evaluation began. With FG on DLSS-G presents a frame's generated
    // Presents and its real one in either order, so a hitch between a
    // frame's evaluation and its last Present made every Present after its
    // first mono (inactive_views, retained_depth_expired; the 10-27 review's
    // F1 and F2, open items 1 and 2), and the first Present after the pause
    // without depth restarted the stereo fade. At most generated + 1
    // Presents count, ended early by the next frame's first evaluation, so a
    // game that keeps presenting without evaluating reaches the source age
    // at its (generated + 2)-th Present as before.
    bool frame_presenting(std::uint64_t ordinal) {
      if (!ordinal) return false;
      const auto n = scope.presents.load(std::memory_order_acquire);
      if (ordinal >= n) return true;
      const std::uint64_t generated = ordinal == scope.frame_generation_present ? scope.presenting_generated : 0;
      return generated && ordinal >= scope.newest_evaluation_present && n - ordinal <= generated;
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

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    std::uint64_t scope_now() {
      const auto fixed = scope_clock.load(std::memory_order_relaxed);
      return fixed ? fixed : GetTickCount64();
    }

#else
    std::uint64_t scope_now() { return GetTickCount64(); }
#endif

    // Requires mutex. The FG scope's view when the scope is live
    // (frame_generation_scope_live), else null.
    evaluation_head *live_frame_generation_scope(std::uint64_t epoch, std::uint32_t viewport,
        std::uint64_t enabled_tick) {
      const auto now = scope_now();
      const auto source = sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(viewport));
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
    frame_generation_activity(GetTickCount64());
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
    const auto source = sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(viewport));
    for (auto &view : evaluations[provider_index(provider_kind::streamline)].views)
      if (view.epoch == epoch && view.viewport == viewport && view.source_id == source)
        view.missed_sequence = std::max(view.missed_sequence, sequence);
  }

  void frame_generation_restored(std::uint64_t epoch, std::uint32_t viewport) {
    if (!requested.load() || !epoch) return;
    std::lock_guard lock(mutex);
    if (auto *mark = fg_retirement_of(epoch, viewport)) mark->active = false;
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

  void present(std::uint64_t native) {
    // Counted for every Present, also one the capture owner does not serve
    // (frame_presenting must never take a frame older than the last Present
    // for the current one).
    scope.presents.fetch_add(1, std::memory_order_acq_rel);
    if (!requested.load() || !native) return;
    std::lock_guard lock(mutex);
    const auto *owner = queue(native);
    if (!owner) return;
    const auto device = device_cookie(owner->device.p);
    if (!device) return;
    for (const auto &weak : capture_pool.sources) if (const auto source = weak.lock())
      if (source->device_identity == device) ++source->present_generation;
  }
}
