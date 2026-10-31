// SPDX-License-Identifier: GPL-3.0-only
// Part of streamline_camera_probe.cpp: evaluation/tag/UI supply, deferred
// frames and their lock. Included at their ORIGINAL position inside the
// shell's existing private namespace; no namespace or using-directive here.
// This preserves non-trivial global initialization/destruction order and all
// existing lock regions/modes. Execution stays in the FG/UI adapter parts.

    // The newest FG copy DLSS-G's own NGX evaluation recorded: its scope (the
    // observation epoch and viewport) and tick (observe_frame_generation_inputs).
    // Within the FG scope's second a command-less FG tag boundary of that
    // scope whose tags offer no copy point withdraws nothing: the evaluation,
    // not the tag, is where its depth is copied (Assassin's Creed Shadows
    // 10-23 tags its FG depth without a command list and untags it on every
    // frame; the untag cancels the copy deferred to the frame's Present, R7).
    SRWLOCK evaluation_supply_lock = SRWLOCK_INIT;
    struct { std::uint64_t epoch{}, tick{}; std::uint32_t viewport{}; } evaluation_supply;
    void remember_evaluation_supply(std::uint64_t scope_epoch, std::uint32_t viewport, std::uint64_t tick) {
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      if (tick >= evaluation_supply.tick || scope_epoch != evaluation_supply.epoch || viewport != evaluation_supply.viewport)
        evaluation_supply = {scope_epoch, tick, viewport};
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    bool evaluation_supplies(std::uint64_t scope_epoch, std::uint32_t viewport) {
      if (!TryAcquireSRWLockShared(&evaluation_supply_lock)) return false; // Busy: withdraw as before.
      const auto supply = evaluation_supply;
      ReleaseSRWLockShared(&evaluation_supply_lock);
      const auto now = GetTickCount64();
      return supply.tick && supply.epoch == scope_epoch && supply.viewport == viewport &&
        sunshine_freshness::ahead_tolerant_le(supply.tick, now, sunshine_freshness::fg_silent_ms);
    }
    // The newest FG tag boundary's outcome per scope: when it recorded a copy
    // and for which tag call, or none when it was refused. A Streamline FG tag
    // that copies keeps precedence over DLSS-G's evaluation of the same frame,
    // which then adds no second copy (observe_frame_generation_inputs,
    // frame_generation::depth_tag_precedence).
    frame_generation::depth_tag_precedence tag_copy;
    // sequence: the tag call's; frame: the frame it named or was associated
    // with (tag_frame).
    void remember_tag_copy(std::uint64_t scope_epoch, std::uint32_t viewport, bool recorded, std::uint64_t sequence = 0,
        const frame_identity &frame = {}) {
      const frame_generation::frame_mark tag{frame_key(frame), sequence};
      const auto now = GetTickCount64();
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      tag_copy.remember(scope_epoch, viewport, recorded, tag, now);
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    // A frame whose FG tag left it without a copy (refused, or its copy
    // deferred to the frame's Present cancelled, refused or lapsed): the FG
    // scope owns no depth for it (depth_capture::frame_generation_missed), so
    // the viewport's ordinary SR/RR depth may supply it. sequence is the
    // frame's tag call. While DLSS-G's own evaluation supplies the scope's
    // copies (remember_evaluation_supply), that evaluation, not the tag,
    // decides the frame.
    void miss_frame_generation(std::uint64_t scope_epoch, std::uint32_t viewport, std::uint64_t sequence) {
      AcquireSRWLockShared(&evaluation_supply_lock);
      const auto supply = evaluation_supply;
      ReleaseSRWLockShared(&evaluation_supply_lock);
      if (supply.tick && supply.epoch == scope_epoch && supply.viewport == viewport &&
          sunshine_freshness::ahead_tolerant_le(supply.tick, GetTickCount64(), sunshine_freshness::fg_silent_ms)) return;
      depth_capture::frame_generation_missed(scope_epoch, viewport, sequence);
    }
    // A precedence per UI kind (ui_mask::capture_slot,
    // ui_mask::tag_precedence), not keyed to the frame as depth's is, since a
    // kind's snapshots pair exactly only within one origin (docs/reshade-sbs.md,
    // DLSS-G's inputs): the newest Streamline
    // UI tag copy of that kind for the scope within the source age, and the
    // newest refusal of one deferred to its frame's
    // Present. DLSS-G's input of a kind the game's own tags copy adds no second
    // copy (observe_frame_generation_inputs), so one origin feeds UI detection,
    // its exact pairs included; a kind the tags leave out or cannot copy is
    // DLSS-G's. A tag's copy deferred to its frame's Present (present_deferred)
    // counts when it is made, so DLSS-G's evaluation of the next frame stands
    // aside; while it is pending, DLSS-G's evaluation of its own frame stands
    // aside too (ui_tags_supply), unless the kind's last deferred copy was
    // refused within the source age: then the tags defer none of the kind
    // (capture_ui_tag) and DLSS-G's input supplies it until that refusal ages.
    sunshine_game3d::ui_mask::tag_precedence ui_tag_copy;
    void remember_ui_tag_copy(std::uint64_t scope_epoch, std::uint32_t viewport, unsigned slot, bool recorded,
        bool deferred = false) {
      const auto now = GetTickCount64();
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      ui_tag_copy.remember(scope_epoch, viewport, slot, recorded, deferred, now);
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    void forget_ui_tag_copy(unsigned slot) {
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      ui_tag_copy.forget(slot);
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    bool ui_deferral_refused(std::uint64_t scope_epoch, std::uint32_t viewport, unsigned slot) {
      const auto now = GetTickCount64();
      AcquireSRWLockShared(&evaluation_supply_lock);
      const bool refused = ui_tag_copy.deferral_refused(scope_epoch, viewport, slot, now);
      ReleaseSRWLockShared(&evaluation_supply_lock);
      return refused;
    }
    // Copies deferred to their frame's Present (owner ruling R2,
    // api_contract.h deferred_copy; the matrix's S6, S11b and U2): per
    // viewport, the newest frame's frame-generation depth candidates (in
    // priority order) and UI tags that their tag call could not copy.
    // present_deferred makes them at that frame's Present (the first after its
    // work was submitted, or under frame generation after DLSS-G evaluated
    // it, while DLSS-G evaluates every frame), or drops them. Guarded by
    // evaluation_supply_lock; deferral_pending counts used entries.
    struct deferred_depth {
      std::array<depth_capture::input, depth_priority.size()> candidates;
      std::size_t count{};
      std::uint64_t superseded{};
    };
    struct deferred_ui {
      bool used{};
      sunshine_game3d::ui_mask::boundary where;
      depth_capture::input input; // Its sequence is its tag call's.
    };
    struct deferred_frame {
      bool used{};
      // executions: the game executions counted at the frame's newest tag
      // call; evaluations: DLSS-G's (fg_evaluation) there and at the
      // viewport's previous deferred frame's.
      std::uint64_t observation{}, epoch{}, serial{}, tick{}, executions{};
      frame_generation::deferral_evaluations::marks evaluations;
      std::uint32_t viewport{};
      // The frame the tag call named (tag_frame); unavailable when none.
      frame_identity frame;
      deferred_depth depth;
      std::array<deferred_ui, 4> ui{}; // By ui_mask::capture_slot.
    };
    std::array<deferred_frame, slot_count> deferred_frames{};
    std::atomic<unsigned> deferral_pending{};
    // Whether DLSS-G's input of this UI kind stands aside for the game's tags
    // (ui_tag_copy): their copy was made within the source age, or one waits
    // for the scope's frame Present (U2, U5), which is that frame's tag copy.
    // Otherwise both would copy the kind, and at that Present DLSS-G's copy of
    // the frame (still pending) and its previous one hold the kind's two
    // snapshots, so the deferred copy finds no reservation and DLSS-G keeps
    // the kind for good. One shared acquisition per kind.
    bool ui_tags_supply(std::uint64_t scope_epoch, std::uint32_t viewport, unsigned slot) {
      if (slot >= ui_tag_copy.copied.size()) return false;
      const bool deferrals = deferral_pending.load(std::memory_order_acquire) != 0;
      const auto now = GetTickCount64();
      AcquireSRWLockShared(&evaluation_supply_lock);
      bool pending = false;
      if (deferrals)
        for (const auto &entry : deferred_frames)
          pending = pending || (entry.used && entry.epoch == scope_epoch && entry.viewport == viewport && entry.ui[slot].used);
      const bool supply = ui_tag_copy.tags_supply(scope_epoch, viewport, slot, pending, now);
      ReleaseSRWLockShared(&evaluation_supply_lock);
      return supply;
    }
    // Whether DLSSG.Depth of the frame DLSS-G evaluates stands aside for that
    // frame's own FG tag copy (tag_copy): recorded for it, or deferred to its
    // Present and waiting there (the viewport's deferral's newest tag call).
    // evaluated: the frame (evaluated_frame).
    bool tags_supply_frame(std::uint64_t scope_epoch, std::uint32_t viewport, const frame_generation::frame_mark &evaluated) {
      const bool deferrals = deferral_pending.load(std::memory_order_acquire) != 0;
      const auto now = GetTickCount64();
      AcquireSRWLockShared(&evaluation_supply_lock);
      frame_generation::frame_mark pending;
      bool waiting = false;
      if (deferrals)
        for (const auto &entry : deferred_frames)
          if (entry.used && entry.depth.count && entry.epoch == scope_epoch && entry.viewport == viewport) {
            pending = {frame_key(entry.frame), entry.serial};
            waiting = true;
          }
      const bool supply = tag_copy.tags_supply(scope_epoch, viewport, evaluated, waiting ? &pending : nullptr, now);
      ReleaseSRWLockShared(&evaluation_supply_lock);
      return supply;
    }
    // DLSS-G's NGX evaluations of a real frame (observe_frame_generation_inputs
    // with an FG scope): their count, and the newest one's scope and tick.
    // DLSS-G evaluates a frame in that frame's Present, so under frame
    // generation a Present after an evaluation that followed a tag call is
    // the tag's frame's (or a later one's); a Present before it is an earlier
    // frame's, paced or generated, while DLSS-G evaluates every frame
    // (frame_generation::deferral_evaluations). Guarded by
    // evaluation_supply_lock.
    frame_generation::deferral_evaluations fg_evaluation;
    // Totals for the bounded deferral line (report_deferrals).
    struct {
      std::atomic<std::uint64_t> depth{}, ui{}, depth_copied{}, ui_copied{}, depth_refused{}, ui_refused{}, cancelled{},
        lapsed{};
      // Presents at which a pending deferral of a frame-generating viewport
      // was not yet its frame's (an earlier frame's Present).
      std::atomic<std::uint64_t> waited{};
      std::atomic<unsigned> refusal{}; // The newest refused depth copy's record_stage.
    } deferrals;
    std::atomic<std::uint64_t> next_deferral_report{}, reported_deferrals{};
    // Totals for the bounded tag-call copy line (count_tag_call_copy): depth
    // copies recorded inside a frame-generation tag call, and those
    // made from the state the call's list recorded where the declaration of
    // the tag's later use named another (S11).
    struct {
      std::atomic<std::uint64_t> copied{}, recorded_over_declared{};
    } tag_call_copies;
    std::atomic<std::uint64_t> next_tag_call_report{};
    void drop_deferred() {
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      for (auto &entry : deferred_frames) entry = {};
      deferral_pending.store(0, std::memory_order_release);
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    // The three default-argument missed-frame sites keep their two calls
    // and separate lock holds in the same order.
    void frame_missed(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t sequence) {
      remember_tag_copy(epoch, viewport, false);
      miss_frame_generation(epoch, viewport, sequence);
    }
