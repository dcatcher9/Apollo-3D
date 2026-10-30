// SPDX-License-Identifier: GPL-3.0-only
// The capture owner's observation of the game's command recording, a part of
// streamline_depth_capture.cpp (included at its end, in the same translation
// unit): command-list and recording identity, per-recording source states and
// barriers (legacy and enhanced), Reset/Close/render-pass lifecycle, the
// executed-list state proof (aliasing, coverage, apply_executed), ReShade's
// list and runtime-list events, Present observation, command-list
// destruction, and the depth-texture lifecycle (creation, destruction, deferred
// sources, depth clears, legacy render-target copies). Every global these
// functions use stays in the shell in its original order; the shell's part
// declarations name the ones shell code calls.
namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    command_storage::handle command(std::uint64_t native, std::uint64_t cookie, bool create) {
      if (!native || !cookie) return {};
      auto entry = command_storage::acquire(reinterpret_cast<ID3D12Object *>(native), create);
      if (!entry) return {};
      if (!entry->cookie && create) {
        entry->cookie = cookie; entry->observation_generation = capture_observation.command_generation;
        entry.life()->cookie.store(cookie, std::memory_order_release);
      }
      if (entry->cookie != cookie) return {};
      if (entry->observation_generation != capture_observation.command_generation) {
        if (!entry->invalid) entry->invalidation = recording_loss::global_observation_loss;
        entry->invalid = true;
      }
      return entry;
    }
    void associate_recording(std::uint64_t native, std::uint64_t *out_cookie) {
      if (!out_cookie) return;
      *out_cookie = 0;
      // Native method entry authenticates a live command object. A fresh list
      // is already open after CreateCommandList and need not call Reset first.
      // Never initialize from a post-call zero-cookie observation: Reset could
      // have changed its lifetime while the original operation was executing.
      std::lock_guard lock(mutex);
      if (!native || !native_observer::recording_cookie_absent(native)) return;
      auto *object = reinterpret_cast<ID3D12Object *>(native);
      if (command_storage::acquire(object, false)) return;
      auto owner = command_storage::acquire(object, true);
      if (!owner || owner->cookie) return;
      const auto cookie = ++serial;
      if (!native_observer::set_recording_cookie(native, cookie)) {
        invalidate(*owner, recording_loss::source_identity_unavailable);
        return;
      }
      owner->restart(cookie, capture_observation.command_generation);
      owner.life()->cookie.store(cookie, std::memory_order_release);
      *out_cookie = cookie;
    }
    source_state *state(command_state &owner, std::uint64_t identity, bool create) {
      if (!identity) return nullptr;
      for (auto &value : owner.states) if (value.source == identity) return &value;
      if (create) {
        try {
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
          if (fail_state_growth && owner.states.size() == owner.states.capacity()) throw std::bad_alloc();
#endif
          if (owner.states.capacity() == 0) owner.states.reserve(16);
          owner.states.push_back({identity});
          return &owner.states.back();
        } catch (...) {
          // Hook callbacks must not throw. Existing callers invalidate the
          // recording if preserving a source's state/block actually fails.
          return nullptr;
        }
      }
      return nullptr;
    }
    void invalidate(command_state &owner, recording_loss why) {
      if (!owner.invalid) owner.invalidation = why;
      owner.invalid = true;
    }
    void block_source(command_state &owner, ID3D12Resource *resource) {
      // Exceptional barriers can precede the first tag. Retain identity on the
      // actual resource, without registering ordinary game resources globally.
      const auto identity = retain_source_cookie(resource);
      auto *value = state(owner, identity, true);
      if (!value) { invalidate(owner, identity ? recording_loss::source_state_capacity : recording_loss::source_identity_unavailable); return; }
      value->blocked = true;
    }
    void invalidate_all() {
      std::lock_guard lock(mutex);
      for_each_capture([](auto &value) { invalidate(value, capture_failure::observer_loss); });
      ++capture_observation.command_generation;
      // Lists executed meanwhile may have gone unobserved.
      capture_observation.proven.unprove_all();
      reason = status::unavailable;
    }
    // Every game ResourceBarrier/Barrier arrives here. A recording's payload is
    // read and written only on the thread recording that list (its lifecycle
    // events, record_impl and this hook; a list is recorded by one thread at
    // a time), so barrier states take no capture lock. A list ReShade never
    // reported cannot admit a capture (lifecycle_view), so nothing is tracked
    // for it and its states cannot grow without a Reset.
    void barriers(std::uint64_t native, std::uint64_t cookie, std::uint32_t count, const D3D12_RESOURCE_BARRIER *values) {
      auto owner = command(native, cookie, true);
      if (!owner) return;
      // A closed recording records nothing: the list was Reset unreported.
      if (owner->closed) { owner->missed_reset = true; return; }
      if (!owner->reported) return;
      ++owner->native_barrier_calls;
      owner->last_barrier_command = native;
      for (unsigned n = 0; n != count; ++n) {
        const auto &value = values[n];
        // An aliasing barrier, named or wildcard (NULL: any placed or reserved
        // resource), changes no resource state; it only moves heap memory
        // between overlapping resources. Every capture here copies where the
        // SDK consumes the source (its tag call or evaluation), so the game
        // must have that source active and initialized there. Transient-memory
        // engines (RE Engine) alias every frame; blocking on them refused all
        // depth. The generic bind-switch preservation copies at no such point
        // and keeps its own alias rule (depth_addon.cpp). The end-of-frame
        // copy copies at no such point either: an aliasing barrier is marked
        // for the per-resource proven state alone, which it revokes
        // (proven_state.h). A resource without an identity cannot be tracked
        // there, so none is created for it.
        if (value.Type == D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {
          const auto order = ++owner->barrier_order;
          const auto &alias = value.Aliasing;
          if (!alias.pResourceBefore && !alias.pResourceAfter) { owner->wildcard_alias = order; continue; }
          for (auto *named : {alias.pResourceBefore, alias.pResourceAfter}) {
            const auto identity = named ? source_cookie(named) : 0;
            if (!identity) continue;
            try {
              if (owner->aliases.capacity() == 0) owner->aliases.reserve(8);
              owner->aliases.push_back({identity, order});
              owner->aliases_sorted = false;
            } catch (...) { invalidate(*owner, recording_loss::source_state_capacity); }
          }
          continue;
        }
        if (value.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
          ++owner->native_transition_count;
          // Only subresource 0 (or all) defines the copied state; a split or
          // transition of another subresource (a stencil plane) leaves it.
          if (value.Transition.Subresource != 0 && value.Transition.Subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) continue;
          const auto flags = value.Flags;
          // Any other flag combination is no transition this owner can follow.
          // Preserve that rejection until Reset without poisoning unrelated depth.
          if (flags != D3D12_RESOURCE_BARRIER_FLAG_NONE && flags != D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY &&
              flags != D3D12_RESOURCE_BARRIER_FLAG_END_ONLY) { block_source(*owner, value.Transition.pResource); continue; }
          // A resource can reach its first SDK nomination after all of its
          // transitions were recorded. Preserve that live object's identity
          // now, including a split that precedes the first tag/cookie. The
          // observed state remains local to this exact recording; it is never
          // a global hint.
          const auto identity = observed_source_cookie(value.Transition.pResource);
          if (!identity) { invalidate(*owner, recording_loss::source_identity_unavailable); continue; }
          auto *known = state(*owner, identity, true);
          if (!known) { invalidate(*owner, recording_loss::source_state_capacity); continue; }
          if (known->blocked) continue;
          // A legacy barrier inside an enhanced split is outside the D3D12 contract.
          if (known->pending_split && known->enhanced) { known->blocked = true; continue; }
          // So is an END_ONLY whose StateAfter is not its BEGIN_ONLY's, which
          // pending_split keeps (MS Learn, Split Barriers: the barrier that
          // ends a split has the same before and after states; the contract
          // matrix's D4c): the state is unknown.
          if (known->pending_split && flags == D3D12_RESOURCE_BARRIER_FLAG_END_ONLY &&
              value.Transition.StateAfter != known->value) { known->blocked = true; continue; }
          // BEGIN_ONLY leaves the state in transition; END_ONLY, like a
          // plain transition, completes in StateAfter.
          known->pending_split = flags == D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
          known->known = !known->pending_split && copy_state::supported(value.Transition.StateAfter);
          known->value = value.Transition.StateAfter;
          known->enhanced = false;
          known->order = ++owner->barrier_order;
        }
      }
    }
    void retire_recording(slot &value, std::uint64_t old) {
      if (!old) return;
      // An unsubmitted recording was discarded. Submitted resources still wait
      // for their actual queue fence; reset never proves GPU completion.
      if (value.command == old) {
        if (!value.producer_submitted) { invalidate(value, capture_failure::discarded_recording); value.finished = true; }
        value.command = 0;
        value.producer_recording.reset();
      }
      for (unsigned i = 0; i != value.consumers.size(); ++i) if (value.consumers[i] == old) {
        value.consumers[i] = 0; value.consumer_recordings[i].reset();
      }
    }
    void retire_dead_recordings(slot &value) {
      // A native-only list may never emit ReShade's destruction event. The
      // object-owned token expires without taking this mutex or retaining COM.
      // This retires CPU identities only; reclaimable still requires GPU fences.
      if (value.command && value.producer_recording &&
          value.producer_recording->cookie.load(std::memory_order_acquire) != value.command)
        retire_recording(value, value.command);
      for (unsigned i = 0; i != value.consumers.size(); ++i)
        if (value.consumers[i] && value.consumer_recordings[i] &&
            value.consumer_recordings[i]->cookie.load(std::memory_order_acquire) != value.consumers[i])
          retire_recording(value, value.consumers[i]);
    }
    // reported: ReShade's lifecycle started the new recording.
    void reset(std::uint64_t native, std::uint64_t old, HRESULT result, bool reported = false) {
      if (FAILED(result)) return;
      std::lock_guard lock(mutex);
      for_each_capture([&](auto &value) { retire_recording(value, old); });
      if (!native) return;
      auto owner = command_storage::acquire(reinterpret_cast<ID3D12Object *>(native), true);
      if (!owner) return;
      owner.life()->cookie.store(0, std::memory_order_release);
      const auto cookie = ++serial;
      if (!native_observer::set_recording_cookie(native, cookie)) {
        invalidate(*owner, recording_loss::source_identity_unavailable); return;
      }
      owner->restart(cookie, capture_observation.command_generation);
      owner->reported = reported;
      owner.life()->cookie.store(cookie, std::memory_order_release);
    }
    void close(std::uint64_t native, std::uint64_t cookie, HRESULT result) {
      std::lock_guard lock(mutex);
      if (auto value = command(native, cookie, true)) {
        value->closed = SUCCEEDED(result);
        if (FAILED(result)) invalidate(*value, recording_loss::close_failed);
      }
      if (FAILED(result)) for_each_capture([&](auto &value) {
        if (value.command == cookie) invalidate(value, capture_failure::close_failed);
      });
    }
    // The legacy state equivalent to a generic enhanced barrier layout, for
    // diagnostics only. Queue-specific and video layouts have none.
    std::optional<std::uint32_t> legacy_state(std::uint32_t layout) {
      switch (static_cast<D3D12_BARRIER_LAYOUT>(layout)) {
        case D3D12_BARRIER_LAYOUT_COMMON: return D3D12_RESOURCE_STATE_COMMON;
        case D3D12_BARRIER_LAYOUT_GENERIC_READ: return D3D12_RESOURCE_STATE_GENERIC_READ;
        case D3D12_BARRIER_LAYOUT_RENDER_TARGET: return D3D12_RESOURCE_STATE_RENDER_TARGET;
        case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS: return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE: return D3D12_RESOURCE_STATE_DEPTH_WRITE;
        case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ: return D3D12_RESOURCE_STATE_DEPTH_READ;
        case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE:
          return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        case D3D12_BARRIER_LAYOUT_COPY_SOURCE: return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case D3D12_BARRIER_LAYOUT_COPY_DEST: return D3D12_RESOURCE_STATE_COPY_DEST;
        case D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE: return D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
        case D3D12_BARRIER_LAYOUT_RESOLVE_DEST: return D3D12_RESOURCE_STATE_RESOLVE_DEST;
        default: return std::nullopt;
      }
    }
    // An enhanced texture barrier that includes subresource 0 sets that
    // texture's layout and access on this recording; other subresources (a
    // stencil plane) keep subresource 0's. The copy then leaves and restores
    // exactly that layout and access with enhanced barriers. A layout no copy
    // can use (neither copy_state::layouts::admissible nor a compute queue's
    // own) or NO_ACCESS refuses the copy until a later barrier names another,
    // and so does one outside the recording's list type
    // (copy_state::list_basis); a SyncAfter of SPLIT leaves the layout in
    // transition until the split's end barrier.
    void enhanced_textures(std::uint64_t native, std::uint64_t cookie, unsigned count,
        const native_observer::enhanced_texture *textures) {
      if (count && !capture_observation.enhanced_barriers_seen.load(std::memory_order_relaxed))
        capture_observation.enhanced_barriers_seen.store(true, std::memory_order_relaxed);
      // No capture lock, exactly as barriers().
      auto owner = command(native, cookie, true);
      if (!owner) return;
      if (owner->closed) { owner->missed_reset = true; return; }
      if (!owner->reported) return;
      for (unsigned i = 0; i != count; ++i) {
        const auto &texture = textures[i];
        if (!texture.resource || !texture.first_subresource) continue;
        const auto identity = observed_source_cookie(texture.resource);
        if (!identity) { invalidate(*owner, recording_loss::source_identity_unavailable); continue; }
        auto *known = state(*owner, identity, true);
        if (!known) { invalidate(*owner, recording_loss::source_state_capacity); continue; }
        known->enhanced_named = true;
        if (known->blocked) continue;
        // An enhanced barrier inside a legacy split is outside the D3D12 contract.
        if (known->pending_split && !known->enhanced) { known->blocked = true; continue; }
        const auto layout = texture.layout_after;
        known->enhanced = true;
        known->layout = layout; known->access = texture.access_after;
        known->value = legacy_state(copy_state::layouts::generic(layout)).value_or(UINT32_MAX);
        known->pending_split = texture.sync_after == D3D12_BARRIER_SYNC_SPLIT;
        known->known = !known->pending_split &&
          (copy_state::layouts::admissible(layout) || copy_state::layouts::compute_queue(layout)) &&
          texture.access_after != D3D12_BARRIER_ACCESS_NO_ACCESS;
        known->order = ++owner->barrier_order;
      }
    }
    // The capture's state evidence for a source from its entry on the recording.
    copy_state::observation observation_of(const source_state *observed) {
      return observed ? copy_state::observation{true, observed->known, observed->blocked, observed->value, observed->pending_split,
        observed->enhanced, observed->layout, observed->access} : copy_state::observation{};
    }
    // Diagnostic provenance (state_basis) of an admitted copy's pre-copy state: an entry on the
    // recording (the copy used or matched it), else the SDK contract or the
    // per-call declaration. It never changes which state the copy uses.
    state_basis copy_basis(const source_state *observed, const copy_state::decision &chosen) {
      if (observed) return observed->enhanced ? state_basis::observed_enhanced : state_basis::observed_legacy;
      return chosen.used_contract ? state_basis::contract : state_basis::declared;
    }
    void render_pass(std::uint64_t native, std::uint64_t cookie, bool inside) {
      std::lock_guard lock(mutex);
      if (auto owner = command(native, cookie, true)) owner->render_pass = inside;
    }
    // Requires mutex. An aliasing barrier on the recording named source after
    // its barrier order (0: anywhere on the recording). On an executed
    // recording (sort_aliases) a binary search finds the source's last mark.
    bool aliased_after(const command_state &owner, std::uint64_t source, std::uint32_t order) {
      if (owner.aliases_sorted) {
        const auto end = std::upper_bound(owner.aliases.begin(), owner.aliases.end(), source,
          [](std::uint64_t key, const resource_mark &mark) { return key < mark.source; });
        return end != owner.aliases.begin() && std::prev(end)->source == source && std::prev(end)->order > order;
      }
      for (const auto &mark : owner.aliases) if (mark.source == source && mark.order > order) return true;
      return false;
    }
    // Requires mutex, on a recording being executed (closed, so nothing
    // appends to it until its Reset): its aliasing marks sorted once, so each
    // held texture's lookup in apply_executed costs a binary search, not a
    // scan of every mark (transient-memory engines alias every frame).
    void sort_aliases(command_state &owner) {
      if (owner.aliases_sorted) return;
      std::sort(owner.aliases.begin(), owner.aliases.end(), [](const resource_mark &a, const resource_mark &b) {
        return a.source != b.source ? a.source < b.source : a.order < b.order;
      });
      owner.aliases_sorted = true;
    }
    // Requires mutex. Whether the barrier hooks saw every barrier the list
    // recorded: its method table is hooked (states_ready). A list being
    // executed is alive, so its table pointer is a plain read; the answer is
    // remembered per table until coverage changes (command_generation), and
    // at most 16 new tables a second are checked (beyond that a list counts
    // as unobserved). Inside the submission callback, where a lost hook's
    // invalidation is suppressed like any nested notification.
    bool states_covered(std::uint64_t native) {
      const auto table = *reinterpret_cast<const std::uintptr_t *>(native);
      const auto generation = capture_observation.command_generation.load();
      for (const auto &entry : covered_tables)
        if (entry.table == table && entry.generation == generation) return entry.covered;
      const auto now = GetTickCount64();
      if (now - coverage_window >= 1000) { coverage_window = now; coverage_checks = 0; }
      if (++coverage_checks > 16) return false;
      const bool covered = native_observer::states_ready(native);
      covered_tables[covered_next++ % covered_tables.size()] = {table, generation, covered};
      return covered;
    }
    // Requires mutex. A copy recorded from its source's proven state
    // (record_impl) is checked when its recording is executed: the proof must
    // be unchanged and the queue the one it was proven for, or the copy is
    // discarded (held, then mono). A replay is invalidated as such.
    void verify_proven_copies(std::uint64_t queue, std::uint64_t cookie) {
      if (!cookie) return;
      for_each_capture([&](slot &value) {
        if (!value.id || !value.proven_version || value.command != cookie) return;
        if (value.proven_queue != queue || capture_observation.proven.version(value.proven_source) != value.proven_version) {
          invalidate(value, capture_failure::proven_state_changed);
          capture_observation.proven_counters.changed_copies.fetch_add(1, std::memory_order_relaxed);
        }
        value.proven_version = 0;
      });
    }
    // Requires mutex. The lists of one ExecuteCommandLists call on queue, in
    // submission order (proven_state.h): copies from a proven state are
    // checked before their list's own barriers apply. A list the barrier
    // hooks did not fully observe (not reported by ReShade's lifecycle, Reset
    // without it, invalidated, or of an unhooked method table) may have
    // changed any texture: none stays proven. Otherwise every held texture
    // its barriers named takes the state its last barrier left it in, on this
    // queue, unless an aliasing barrier followed (any placed or reserved
    // resource for one naming neither), a split is still open or a barrier
    // was blocked; a depth clear after its last barrier proves DEPTH_WRITE,
    // or the DEPTH_STENCIL_WRITE layout where enhanced barriers own it. An
    // enhanced barrier naming it latches its layout (proven_state.h).
    void apply_executed(std::uint64_t queue, std::uint32_t count, const native_observer::command_identity *values) {
      for (std::uint32_t n = 0; n != count; ++n) {
        verify_proven_copies(queue, values[n].recording_cookie);
        if (!capture_observation.proven.size()) continue;
        const auto native = values[n].native_command;
        // A list that never had a recording state, while the hooks cover its
        // method table, recorded no barrier through them (the first would
        // have given it one): DXGI's internal Present list. It changes no
        // texture's state.
        if (!values[n].recording_cookie && native && native_observer::recording_cookie_absent(native) && states_covered(native))
          continue;
        auto owner = command(native, values[n].recording_cookie, false);
        auto &c = capture_observation.proven_counters;
        auto *unobserved = !owner ? &c.unrecorded : !owner->reported ? &c.unreported : owner->invalid ? &c.invalidated :
          owner->missed_reset ? &c.missed_reset : !states_covered(native) ? &c.uncovered : nullptr;
        if (unobserved) {
          capture_observation.proven.unprove_all();
          unobserved->fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        if (owner->wildcard_alias) capture_observation.proven.unprove_all();
        sort_aliases(*owner);
        // Each aliased texture once (the marks are sorted by source).
        for (std::size_t i = 0; i != owner->aliases.size(); ++i)
          if (!i || owner->aliases[i].source != owner->aliases[i - 1].source) capture_observation.proven.unprove(owner->aliases[i].source);
        for (const auto &entry : owner->states) {
          if (!capture_observation.proven.held(entry.source)) continue;
          // Any enhanced barrier naming it latches its layout (D6), whatever
          // state the list left it in.
          if (entry.enhanced_named) capture_observation.proven.set_latch(entry.source);
          if (!entry.known || entry.blocked || entry.pending_split || owner->wildcard_alias > entry.order ||
              aliased_after(*owner, entry.source, entry.order))
            capture_observation.proven.unprove(entry.source);
          else if (entry.enhanced) capture_observation.proven.prove_layout(entry.source, queue, entry.layout, entry.access);
          else capture_observation.proven.prove(entry.source, queue, entry.value);
        }
        // A depth clear after a texture's last barrier proves what the D3D12
        // clear contract requires, in recording order, unless aliased after
        // it: DEPTH_WRITE, or the DEPTH_STENCIL_WRITE layout where enhanced
        // barriers own the layout (that last barrier on this list, else the
        // texture's proof; proven_state.h clear_regime). A legacy proof there
        // would let a copy record legacy barriers on an enhanced layout (D6).
        for (const auto &mark : owner->clears) {
          if (!capture_observation.proven.held(mark.source)) continue;
          const auto entry = std::find_if(owner->states.begin(), owner->states.end(),
            [&](const source_state &value) { return value.source == mark.source; });
          if (entry != owner->states.end() && entry->order > mark.order) continue;
          using sunshine_proven_state::regime;
          const auto kind = entry == owner->states.end() ? capture_observation.proven.clear_regime(mark.source) :
            entry->enhanced ? regime::enhanced : regime::legacy;
          if (kind == regime::unknown || owner->wildcard_alias > mark.order || aliased_after(*owner, mark.source, mark.order))
            capture_observation.proven.unprove(mark.source);
          else if (kind == regime::enhanced)
            capture_observation.proven.prove_layout(mark.source, queue, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE);
          else capture_observation.proven.prove(mark.source, queue, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        }
      }
    }
  }
  void observe_command(std::uint64_t native) {
    if (!requested.load() || !native) return;
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native, IID_ID3D12GraphicsCommandList, checked)) return;
    native = reinterpret_cast<std::uint64_t>(checked.p);
    auto cookie = native_observer::get_recording_cookie(native);
    // Nomination uses the same strict first-recording association as native
    // method entry. Failed/malformed private data must never be revived here.
    if (!cookie) associate_recording(native, &cookie);
    {
      std::lock_guard lock(mutex);
      // Attach before hook discovery can invalidate the current recording.
      // Late payload allocation must not erase a loss already observed here.
      command(native, cookie, true);
    }
    native_observer::observe_command(native);
  }
  namespace {
    bool runtime_list(std::uint64_t native) {
      if (!native || !capture_observation.runtime_lists_known.load(std::memory_order_acquire)) return false;
      AcquireSRWLockShared(&capture_observation.runtime_lock);
      const bool found = std::find(capture_observation.runtime_lists.begin(), capture_observation.runtime_lists.end(), native) != capture_observation.runtime_lists.end();
      ReleaseSRWLockShared(&capture_observation.runtime_lock);
      return found;
    }
    // ReShade reports only game lists, so a reported list at a runtime list's
    // address is a game list that reused it.
    void forget_runtime_list(std::uint64_t native) {
      if (!runtime_list(native)) return;
      AcquireSRWLockExclusive(&capture_observation.runtime_lock);
      for (auto &value : capture_observation.runtime_lists) if (value == native) value = 0;
      ReleaseSRWLockExclusive(&capture_observation.runtime_lock);
    }
    // ReShade's lifecycle is the only source of a list's recording. created: a
    // recording opens (a new list's first, or a registered runtime list's
    // current one); reset: the previous one ends and the next opens. native is
    // the list object itself, whose private data holds its recording.
    void lifecycle_recording(std::uint64_t native, list_event event) {
      const auto cookie = native_observer::get_recording_cookie(native);
      switch (event) {
        case list_event::created:
          if (cookie) { reset(native, cookie, S_OK, true); break; }
          {
            std::uint64_t first{};
            associate_recording(native, &first);
            std::lock_guard lock(mutex);
            if (auto owner = command(native, first, true)) owner->reported = true;
          }
          break;
        case list_event::reset: reset(native, cookie, S_OK, true); break;
        case list_event::closed: close(native, cookie, S_OK); break;
        case list_event::pass_begin: render_pass(native, cookie, true); break;
        case list_event::pass_end: render_pass(native, cookie, false); break;
        default: break;
      }
    }
  }
  void observe_list_event(std::uint64_t native, list_event event) {
    if (!native || event >= list_event::count) return;
    switch (event) {
      case list_event::destroyed: command_destroyed(native); return;
      case list_event::created: forget_runtime_list(native); break;
      default: break;
    }
    if (!requested.load()) return;
    try { lifecycle_recording(native, event); } catch (...) {}
  }
  void observe_runtime_list(std::uint64_t native, std::uint64_t queue) {
    if (!native || !requested.load() || runtime_list(native)) return;
    // Its recordings advance only through submissions observed on its queue.
    if (!native_observer::queue_ready(queue)) return;
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native, IID_ID3D12GraphicsCommandList, checked) || reinterpret_cast<std::uint64_t>(checked.p) != native) return;
    bool added = false;
    AcquireSRWLockExclusive(&capture_observation.runtime_lock);
    if (std::find(capture_observation.runtime_lists.begin(), capture_observation.runtime_lists.end(), native) == capture_observation.runtime_lists.end())
      for (auto &value : capture_observation.runtime_lists) if (!value) { value = native; added = true; break; }
    if (added) capture_observation.runtime_lists_known.store(true, std::memory_order_release);
    ReleaseSRWLockExclusive(&capture_observation.runtime_lock);
    // Registered while ReShade records it: its current recording is open.
    if (added) try { lifecycle_recording(native, list_event::created); } catch (...) {}
  }
  void observe_present(std::uint64_t command_native, std::uint64_t queue_native, std::uint64_t immediate) {
    if (!requested.load()) return;
    // The registered immediate list was checked for this exact interface at
    // registration; any other list (or an unregistered one) is queried.
    std::uint64_t list{}, cookie{};
    if (command_native && command_native == immediate && runtime_list(command_native)) {
      list = command_native;
      cookie = native_observer::get_recording_cookie(list);
    }
    if (!list && command_native) {
      com_ptr<ID3D12GraphicsCommandList> checked;
      if (query_native(command_native, IID_ID3D12GraphicsCommandList, checked)) {
        list = reinterpret_cast<std::uint64_t>(checked.p);
        cookie = native_observer::get_recording_cookie(list);
      }
    }
    // The same strict first-recording association as observe_command.
    if (list && !cookie) associate_recording(list, &cookie);
    bool known_queue_present = false;
    {
      pooled_storage discard;
      // The wait for the capture lock alone: a game thread's submission
      // callback (submitted) holds it across its copy-out driver calls.
      std::unique_lock lock(mutex, std::defer_lock);
      {
        const sunshine_game3d::slow_step step("depth present observation lock");
        lock.lock();
      }
      if (list) command(list, cookie, true);
      if (queue_native && known_queue(queue_native)) {
        known_queue_present = true;
        collect_retired(discard);
      }
    }
    if (list) native_observer::observe_command(list);
    if (queue_native) {
      if (known_queue_present) native_observer::observe_queue(queue_native);
      else observe_queue(queue_native);
    }
    if (immediate) observe_runtime_list(immediate, queue_native);
  }
  void command_destroyed(std::uint64_t native) {
    if (!native) return;
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native, IID_ID3D12GraphicsCommandList, checked)) return;
    native = reinterpret_cast<std::uint64_t>(checked.p);
    // Destruction retires this recording but cannot complete GPU work. The
    // same fence/ownership requirements still protect every captured texture.
    std::lock_guard lock(mutex);
    const auto cookie = native_observer::get_recording_cookie(native);
    for_each_capture([&](auto &value) {
      retire_recording(value, cookie);
      // Never submitted, so its recorded read can no longer execute.
      if (value.pending_immediate == native) value.pending_immediate = 0;
    });
    if (auto owner = command_storage::acquire(checked.p, false)) {
      owner.life()->cookie.store(0, std::memory_order_release);
      owner->closed = true;
      owner->reported = false;
    }
  }
  bool recorded_state(std::uint64_t native_command, const source_ref &source, bool *writing) {
    if (writing) *writing = false;
    if (!requested.load() || !native_command || !source) return false;
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native_command, IID_ID3D12GraphicsCommandList, checked)) return false;
    const auto native = reinterpret_cast<std::uint64_t>(checked.p);
    if (!native_observer::states_ready(native)) return false;
    const auto cookie = native_observer::get_recording_cookie(native);
    std::lock_guard lock(mutex);
    auto owner = command(native, cookie, false);
    const auto *entry = owner && !owner->invalid ? state(*owner, source->cookie, false) : nullptr;
    if (entry && writing) *writing = copy_state::writes(observation_of(entry));
    return entry != nullptr;
  }
  void resource_created(std::uint64_t resource, std::uint32_t initial_state) {
    if (!requested.load() || !resource) return;
    com_ptr<ID3D12Resource> checked;
    if (!query_native(resource, IID_ID3D12Resource, checked)) return;
    // Buffers and simultaneous-access textures decay to COMMON when each
    // execution completes (MS Learn, Implicit State Transitions): no barrier
    // proves their state between executions.
    const auto desc = checked->GetDesc();
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)) return;
    const auto identity = retain_source_cookie(checked.p);
    if (!identity) return;
    std::lock_guard lock(mutex);
    forget_destroyed();
    capture_observation.proven.track(identity, initial_state != UINT32_MAX, initial_state);
  }
  void resource_destroyed(std::uint64_t identity) {
    if (!identity || !requested.load()) return;
    for (auto &slot : capture_observation.destroyed) {
      std::uint64_t free{};
      if (slot.compare_exchange_strong(free, identity, std::memory_order_acq_rel)) {
        capture_observation.destroyed_pending.store(true, std::memory_order_release);
        return;
      }
    }
  }
  void track_deferred_source(const source_ref &source) {
    if (!requested.load() || !source || !source->cookie || source->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ||
        (source->desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)) return;
    std::lock_guard lock(mutex);
    forget_destroyed();
    capture_observation.proven.adopt(source->cookie);
  }
  void observe_depth_clear(std::uint64_t native, std::uint64_t resource) {
    if (!requested.load() || !native || !resource) return;
    // No capture lock, exactly as barriers(): the recording thread owns it.
    auto owner = command(native, native_observer::get_recording_cookie(native), false);
    if (!owner || owner->closed || !owner->reported) return;
    // A tracked texture already has its identity.
    const auto identity = source_cookie(reinterpret_cast<ID3D12Object *>(resource));
    if (!identity) return;
    try {
      if (owner->clears.capacity() == 0) owner->clears.reserve(4);
      owner->clears.push_back({identity, ++owner->barrier_order});
    } catch (...) { invalidate(*owner, recording_loss::source_state_capacity); }
  }
  bool legacy_render_target_copy(std::uint64_t native, std::uint64_t resource) {
    com_ptr<ID3D12Resource> checked;
    if (!query_native(resource, IID_ID3D12Resource, checked) ||
        (checked->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)) return false;
    if (!requested.load()) return true;
    const auto identity = retain_source_cookie(checked.p);
    if (!identity) return false;
    // No capture lock for the recording, exactly as barriers().
    com_ptr<ID3D12GraphicsCommandList> list;
    if (query_native(native, IID_ID3D12GraphicsCommandList, list)) {
      native = reinterpret_cast<std::uint64_t>(list.p);
      auto owner = native_observer::states_ready(native) ?
        command(native, native_observer::get_recording_cookie(native), false) : command_storage::handle{};
      if (owner && owner->reported && !owner->closed && !owner->invalid)
        if (const auto *entry = state(*owner, identity, false))
          return !entry->enhanced && entry->known && !entry->blocked && !entry->pending_split &&
            entry->value == D3D12_RESOURCE_STATE_RENDER_TARGET;
    }
    std::lock_guard lock(mutex);
    forget_destroyed();
    capture_observation.proven.watch(identity);
    return !capture_observation.proven.latched(identity).latched;
  }
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  // Test view only (streamline_depth_capture_test_access.h); no shipping
  // code reads it.
  bool last_proven_state(std::uint64_t resource, std::uint64_t queue, proven_state &out) {
    out = {};
    com_ptr<ID3D12Resource> checked;
    com_ptr<ID3D12CommandQueue> checked_queue;
    if (!requested.load() || !query_native(resource, IID_ID3D12Resource, checked) ||
        !query_native(queue, IID_ID3D12CommandQueue, checked_queue)) return false;
    queue = reinterpret_cast<std::uint64_t>(checked_queue.p);
    const auto identity = source_cookie(checked.p);
    if (!identity || !native_observer::queue_ready(queue)) return false;
    std::lock_guard lock(mutex);
    out = capture_observation.proven.query(identity, queue, !capture_observation.enhanced_barriers_seen.load(std::memory_order_relaxed));
    return bool(out);
  }
#endif
}
