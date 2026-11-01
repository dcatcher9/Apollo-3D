// SPDX-License-Identifier: GPL-3.0-only
// Part of streamline_camera_probe.cpp: the FG options store. Included at the
// shell's state position inside its existing private namespace, like
// presentation_scope_supply.inl; no namespace or using-directive here.

    // The FG mode (docs/reshade-sbs.md, Presentation scope): each viewport's
    // newest committed FG options, kept apart from the per-call camera and
    // tag metadata (metadata_state), which every camera, tag and token call
    // republishes. The options hook (hook_fg_options) is the only writer of
    // options and cannot lose its write (drop_fg_mode and clear_fg_modes only
    // erase slots); a reader always gets the last committed options.
    // Until 11-01 the mode lived in the camera records: a read that lost the
    // writers' race was "busy", and each reader had its own rule for it (the
    // provider required FG before any confirmation, the SL route took the
    // viewport as FG's, nested NGX did not, NGX copied, the UI kept its mode,
    // present_deferred took the viewport as generating, DLSS-G's inputs found
    // no FG scope, the SL UI tags took FG as off); an options call that lost
    // its write to a camera writer bumped fg_loss_revision, which left the
    // mode ambiguous for every reader until the game's next options call. A
    // viewport keeps its mode as long as its camera record (drop_fg_mode).
    // fg_mode_lock is a leaf: nothing else is taken while it is held.
    struct fg_mode_slot {
      frame_generation_snapshot state;
      // The observation generation its options call was sampled in: a slot of
      // an earlier observation reads as absent (as a metadata pin of one does).
      std::uint64_t observation{};
      bool used{};
    };
    using fg_mode_table = std::array<fg_mode_slot, slot_count>;
    SRWLOCK fg_mode_lock = SRWLOCK_INIT;
    fg_mode_table fg_modes{};
    // A copy of the store.
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    // Test seam (testing::set_fg_store_read_seam): runs after each copy of
    // the store, outside fg_mode_lock.
    std::function<void()> fg_store_read_seam;
#endif
    fg_mode_table read_fg_modes() {
      AcquireSRWLockShared(&fg_mode_lock);
      const auto out = fg_modes;
      ReleaseSRWLockShared(&fg_mode_lock);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      if (fg_store_read_seam) fg_store_read_seam();
#endif
      return out;
    }
    // The viewport's committed options in the current observation, or none
    // (a default snapshot: sequence 0, not known).
    frame_generation_snapshot read_fg_mode(std::uint32_t viewport) {
      const auto observation = observation_generation.load(std::memory_order_acquire);
      frame_generation_snapshot out;
      AcquireSRWLockShared(&fg_mode_lock);
      for (const auto &slot : fg_modes)
        if (slot.used && slot.observation == observation && slot.state.viewport == viewport) { out = slot.state; break; }
      ReleaseSRWLockShared(&fg_mode_lock);
      return out;
    }
    // Requires fg_mode_lock exclusive. The viewport's slot of observation; a
    // new viewport takes a free slot, one of an earlier observation, or the
    // slot whose options were set longest ago (more than four FG viewports).
    fg_mode_slot &fg_mode_slot_of(std::uint32_t viewport, std::uint64_t observation) {
      for (auto &slot : fg_modes)
        if (slot.used && slot.observation == observation && slot.state.viewport == viewport) return slot;
      auto *chosen = &fg_modes[0];
      for (auto &slot : fg_modes) {
        if (!slot.used || slot.observation != observation) { chosen = &slot; break; }
        if (slot.state.sequence < chosen->state.sequence) chosen = &slot;
      }
      *chosen = {};
      chosen->used = true;
      chosen->observation = observation;
      chosen->state.viewport = viewport;
      return *chosen;
    }
    // A viewport whose camera record a newer viewport's replaced (slot(): the
    // game renders it no longer) loses its FG mode with it, as when the mode
    // lived in the record: until 11-01, and again since the same day's fix,
    // a retired viewport's mode never keeps the any-viewport query ambiguous
    // beside the game's new FG viewport.
    void drop_fg_mode(std::uint32_t viewport) {
      AcquireSRWLockExclusive(&fg_mode_lock);
      for (auto &slot : fg_modes) if (slot.used && slot.state.viewport == viewport) slot = {};
      ReleaseSRWLockExclusive(&fg_mode_lock);
    }
    // Lifecycle (release_camera_records): a restart reads every mode as unavailable.
    void clear_fg_modes() {
      AcquireSRWLockExclusive(&fg_mode_lock);
      fg_modes = {};
      ReleaseSRWLockExclusive(&fg_mode_lock);
    }

    // The process's one FG view (presentation_scope::fg_view), updated once
    // per Present on its present thread (take_present_fg) under fg_view_lock,
    // held there across that Present's one store read, so the lock order is
    // fg_view_lock, then fg_mode_lock (shared), and reset at a restart
    // (release_camera_records); and this thread's Present's snapshot.
    SRWLOCK fg_view_lock = SRWLOCK_INIT;
    presentation_scope::fg_view fg_state;
    thread_local presentation_scope::present_fg present_snapshot;
    // Consumers that found no snapshot taken for their Present and took
    // their own (present_frame_generation, present_fg_fallbacks), and the
    // newest such read on this thread, which no later call reuses.
    thread_local presentation_scope::present_fg fallback_snapshot;
    std::atomic<std::uint64_t> fg_snapshot_fallbacks{};
