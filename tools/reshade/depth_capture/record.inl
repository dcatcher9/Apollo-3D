// SPDX-License-Identifier: GPL-3.0-only
// Evaluation nomination and capture recording; included at the end of
// streamline_depth_capture.cpp. All state and inline test hooks keep their owners.

namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    unsigned provider_index(provider_kind value) { return static_cast<unsigned>(value); }

    bool valid_provider(provider_kind value) { return provider_index(value) < provider_count; }

    void set_attempt(provider_kind provider, status value) {
      reason = value;
      if (valid_provider(provider)) attempt_status[provider_index(provider)] = value;
    }

    // A nomination's identity: its evaluation's epoch, sequence, source and
    // viewport.
    evaluation_head nomination_of(const input &value) {
      return {value.epoch, value.sequence, 0, value.source_id, value.viewport};
    }

    bool same_nomination(const evaluation_head &a, const evaluation_head &b) {
      return a.epoch == b.epoch && a.sequence == b.sequence && a.source_id == b.source_id && a.viewport == b.viewport;
    }

    // Ends provider's nomination in progress when it is still this one.
    void end_nomination(provider_kind provider, const evaluation_head &nomination) {
      std::lock_guard lock(mutex);
      if (valid_provider(provider)) {
        auto &pending = evaluations[provider_index(provider)].pending_nomination;
        if (same_nomination(pending, nomination)) pending = {};
      }
    }

    void end_nomination(const input &value) { end_nomination(value.provider, nomination_of(value)); }

    // A metadata-only nomination: source authority without GPU storage.
    std::uint64_t record_nomination(const input &value, std::uint64_t normalized_source,
        std::uint64_t cookie, const recording_ref &recording) {
      // Keep the GPU-capable pool available while trying alternate tags in one
      // evaluation. A metadata-only high-resolution nomination must not occupy
      // the last slot that could capture the ordinary depth tag.
      const auto pixel_end = capture_pool.slots.begin() + pixel_slot_limit;
      auto found = std::find_if(pixel_end, capture_pool.slots.end(), reclaimable);
      if (found == capture_pool.slots.end()) {
        found = std::find_if(capture_pool.slots.begin(), pixel_end, reclaimable);
        if (found == pixel_end) return 0;
      }
      *found = {};
      found->nomination_only = true;
      found->metadata = value; found->metadata.resource.native = normalized_source;
      retire_frame_generation_role(found->metadata);
      found->id = ++serial; found->command = cookie; found->producer_recording = recording;
      return found->id;
    }

    bool reusable_preserved(const slot &entry, const input &value, std::uint64_t recording) {
      return entry.id && entry.preservation_only && !entry.invalid && !entry.producer_submitted &&
        !entry.acquired && entry.command == recording && entry.metadata.source && value.source &&
        entry.metadata.source->cookie == value.source->cookie &&
        std::none_of(entry.consumers.begin(), entry.consumers.end(), [](auto id) { return id != 0; });
    }
  }

  void retire_source(provider_kind provider, std::uint64_t epoch, std::uint64_t source_id, std::uint64_t retiring_serial,
      bool confirmed_off) {
    if (!valid_provider(provider) || !epoch || !source_id) return;
    std::lock_guard lock(mutex);
    const auto matches = [&](const input &value) {
      return value.provider == provider && value.epoch == epoch && value.source_id == source_id;
    };
    // FG Off (slDLSSGSetOptions mode=0, or an options call that failed)
    // retires only the FG role of its viewport's display lineage: the
    // viewport's SR/RR copies of the same epoch are the lineage's successor
    // (depth_hold.h, lineage_successor), and nothing else of it is
    // revoked. The lineage's owner (the queue whose display shows it,
    // established on the FG role or already on the viewport's ordinary
    // source: a Present selects a refused FG frame's SR head before the
    // Off, seed c40a42f921da5ba0 at frame 148) is floored at the copy its
    // display last copied (shown_capture), so no copy published before the
    // one it showed returns. Until 10-31 the floor was the owner's last
    // selected copy, matched only on the FG role: an owner the Present
    // before had moved to the ordinary source took no floor (frame 149
    // showed frame 144's SR copy after frame 146's FG copy), and a pending
    // FG copy never shown (DLSS-G's compute copy in flight at its own
    // Present) became the floor. The FG role's owner also keeps its
    // consumed frame (last_epoch, last_sequence: the viewport's FG and SR
    // inputs share one sequence per epoch), so no copy of an earlier frame
    // returns; present and present_capture stay, so a repeated effect of the
    // same presentation selects nothing new, and present_capture stays the
    // handover floor of establish_provider. Its association ends as before
    // (C8 holds the display's copy within the source age, Generic may
    // follow). Until 10-28 the watermark moved to serial here, which revoked
    // the viewport's SR copies recorded since the shown FG copy, the
    // lineage's legal successor: a Present before its next SR copy was
    // readable held the older FG copy (completed_before_gap), or went mono
    // once that passed the source age. Any other retired source (an NGX
    // feature released) still ends its presentation: copies recorded before
    // the release never re-establish it (the next provider must submit a new
    // frame; IDs use pre-increment, so serial is the last issued ID).
    std::uint32_t viewport {};
    const bool fg_role = provider == provider_kind::streamline &&
      sunshine_scene_depth::fg_depth_viewport(source_id, viewport);
    const auto lineage_owner = [&](const queue_state &owner) {
      return fg_role && owner.provider_established && owner.provider == provider && owner.nominated_epoch == epoch &&
        owner.viewport == viewport && (owner.source_id == source_id || !owner.source_id);
    };
    std::uint64_t shown {};
    for (const auto &owner : capture_queues.queues)
      if (owner && lineage_owner(*owner)) shown = std::max(shown, owner->shown_capture);
    auto &current = evaluations[provider_index(provider)];
    const auto now = GetTickCount64();
    // Another viewport's view in this epoch is active (or presenting): the
    // viewport's picks stay ambiguous, so neither its FG copies nor its FG
    // nomination pass to its ordinary source (below).
    const bool other_viewport_active = fg_role &&
      std::any_of(current.views.begin(), current.views.end(), [&](const evaluation_head &view) {
        return view.sequence && view.epoch == epoch && view.viewport != viewport &&
          (sunshine_freshness::fresh_lt(view.tick, now, sunshine_freshness::source_age_ms) || view_presenting(view, provider));
      });
    // The role stays retired until FG On (frame_generation_restored): the
    // inputs admitted before this call (serial), and any while FG stays off,
    // are the viewport's ordinary source (frame_generation_role_retired,
    // frame_generation_retired). retiring_serial: the options call's.
    if (fg_role) {
      auto *mark = fg_retirement_of(epoch, viewport);
      const auto kept = mark ? mark->serial : 0;
      if (!mark) {
        mark = &scope.fg_retirements[0];
        for (auto &value : scope.fg_retirements)
          if (value.epoch < mark->epoch || (value.epoch == mark->epoch && value.serial < mark->serial)) mark = &value;
      }
      *mark = {epoch, std::max(kept, retiring_serial), viewport, true};
    }
    // The retired source's copies, recorded or still recording. At a
    // confirmed FG Off those published after the copy the lineage's display
    // showed are the frames' own depth of the display lineage, the
    // viewport's ordinary source, as an input of the role admitted after the
    // Off is (retire_frame_generation_role), so the newest readable one
    // replaces the held copy (C8), until FG On restores the role
    // (frame_generation_restored drops them: an FG Off/On transition never
    // resurrects the FG scope before it). The others, and every copy of a
    // failed or unknown options call's scope or of any other retired source,
    // never show again. Until 10-31 every copy was invalidated, so DLSS-G's
    // compute copy in flight at its own Present never showed after the Off
    // and the display held the older copy beside it (seed 40e5fd9a22d273bc
    // at frames 107 to 109, 64863a396a5d6dd5 at frames 139 to 142), or the
    // queue lost its association at a late Off (117a6b99131447b1 at frame
    // 57).
    for (auto &value : capture_pool.slots) if (value.id && !value.preservation_only && matches(value.metadata)) {
      if (fg_role && confirmed_off && !other_viewport_active && value.id > shown) {
        retire_frame_generation_role(value.metadata);
        value.retired_fg_copy = true;
      } else {
        invalidate(value, capture_failure::source_retired);
      }
    }
    for (auto &owner : capture_queues.queues) if (owner && lineage_owner(*owner) && owner->source_id != source_id)
      owner->admission_after = std::max(owner->admission_after, owner->shown_capture);
    for (auto &owner : capture_queues.queues) if (owner && owner->provider_established && owner->provider == provider &&
        owner->nominated_epoch == epoch && owner->source_id == source_id) {
      owner->provider_established = false;
      if (fg_role) {
        owner->source_id = owner->last_source_tick = owner->nominated_epoch = 0;
        owner->admission_after = std::max(owner->admission_after, owner->shown_capture);
        continue;
      }
      owner->source_id = owner->last_epoch = owner->last_sequence = owner->last_source_tick = owner->nominated_epoch = owner->present_capture = 0;
      owner->admission_after = std::max(owner->admission_after, serial.load(std::memory_order_relaxed));
    }
    // The FG role's newest nomination, the namespace's newest, passes to the
    // viewport's ordinary source (Streamline SR/RR: source 0) as a head
    // without a capture: the next Present before the viewport's next SR
    // nomination is a missing frame of the lineage (select_capture), whose
    // newest legal copy stands in, and moves no watermark
    // (interrupts_admission). Until 10-28 the namespace kept a head no view
    // named (its FG nomination had replaced the viewport's SR view), so that
    // Present read inactive or ambiguous views and moved acquire's gap
    // watermark, which revoked the same SR copies. Not while another
    // viewport's view in this epoch is active (select_latest_capture): a
    // source-0 view beside it keeps the picks ambiguous, so the view goes as
    // before. An inactive one (a viewport that stopped evaluating, such as
    // the session timelines' earlier viewports) makes nothing ambiguous.
    const evaluation_head *head = nullptr;
    if (fg_role && current.epoch == epoch && current.source_id == source_id && !other_viewport_active)
      for (const auto &view : current.views)
        if (view.epoch == epoch && view.source_id == source_id && view.sequence == current.sequence) head = &view;
    for (auto &view : current.views) {
      if (&view == head) {
        // Its call (sequence, tick, present) without the FG scope's facts.
        evaluation_head ordinary {epoch, view.sequence, view.tick, 0, viewport};
        ordinary.present = view.present;
        view = ordinary;
      } else if (view.epoch == epoch && (view.source_id == source_id || (head && view.viewport == viewport && !view.source_id))) {
        view = {};
      }
    }
    if (current.pending_nomination.epoch == epoch && current.pending_nomination.source_id == source_id)
      current.pending_nomination = {};
    if (current.epoch == epoch && current.source_id == source_id) {
      // Preserve the namespace's monotonic sequence watermark; releasing a
      // feature does not authorize replay of its earlier evaluation. Source 0
      // names the re-keyed head above, or no view.
      current.source_id = 0;
      attempt_status[provider_index(provider)] = status::unavailable;
    }
  }

  static void begin_evaluation_locked(std::uint64_t epoch, std::uint64_t sequence, std::uint32_t viewport,
      provider_kind provider, std::uint64_t source_id, std::uint64_t superseded_source_id) {
    if (!requested.load() || !valid_provider(provider)) return;
    auto &current = evaluations[provider_index(provider)];
    if (epoch != current.epoch) { current = {}; current.epoch = epoch; }
    if (frame_generation_role_retired(provider, epoch, viewport)) {
      const auto fg_key = sunshine_scene_depth::fg_source_id(viewport);
      if (source_id == fg_key) source_id = 0;
      if (!source_id) superseded_source_id = fg_key;
    }
    if (sequence <= current.sequence) return;
    current.sequence = sequence;
    current.source_id = source_id;
    current.pending_nomination = {};
    attempt_status[provider_index(provider)] = status::unavailable;
    // An explicit adapter role change (SR -> FG) replaces one logical source
    // for this same viewport. Other viewports remain independently ambiguous;
    // this changes neither captured GPU lifetimes nor provider ownership.
    if (superseded_source_id != UINT64_MAX && superseded_source_id != source_id)
      for (auto &value : current.views)
        if (value.epoch == epoch && value.viewport == viewport && value.source_id == superseded_source_id) value = {};
    const auto now = GetTickCount64();
    const auto ordinal = present_ordinal();
    if (epoch) {
      scope.newest_evaluation_present = std::max(scope.newest_evaluation_present, ordinal);
      if (provider == provider_kind::streamline &&
          sunshine_scene_depth::is_fg_source(source_id, viewport))
        stamp_frame_generation(ordinal);
    }
    for (auto &value : current.views) if (value.epoch == epoch && value.viewport == viewport && value.source_id == source_id) {
      // A new call does not end a run of rejected ones; only its result can.
      value.sequence = sequence; value.tick = now; value.present = ordinal; return;
    }
    auto oldest = std::min_element(current.views.begin(), current.views.end(), [](const auto &a, const auto &b) { return a.sequence < b.sequence; });
    *oldest = {epoch, sequence, now, source_id, viewport};
    oldest->present = ordinal;
  }

  void begin_evaluation(std::uint64_t epoch, std::uint64_t sequence, std::uint32_t viewport,
      provider_kind provider, std::uint64_t source_id, std::uint64_t superseded_source_id) {
    std::lock_guard lock(mutex);
    begin_evaluation_locked(epoch, sequence, viewport, provider, source_id, superseded_source_id);
  }

  namespace {
    // Requires mutex. value, its provider's newest evaluation, is a nomination
    // in progress (pending_nomination) until its copy is recorded or refused
    // (end_nomination). Every evaluation's, SR's as well as FG's: a Present
    // meanwhile, another thread's, does not read that attempt as refused,
    // which moved acquire's gap watermark and so revoked the source's earlier
    // snapshots still in flight on their queue (Assassin's Creed Shadows
    // 10-25: with FG on, FG Presents landed while the SR evaluate allocated
    // storage for each dynamic-resolution size, and each revoked SR depth for
    // a frame or more); the snapshot completed before it stands in
    // (select_capture), else the Present holds. false: value is not the newest
    // evaluation, or (unless renewed) its nomination is already in progress
    // and its own caller ends it.
    bool mark_pending(evaluation_namespace &current, const evaluation_head &nomination, bool renewed) {
      if (current.epoch != nomination.epoch || current.sequence != nomination.sequence ||
          current.source_id != nomination.source_id || !nomination.epoch || !nomination.sequence ||
          (!renewed && same_nomination(current.pending_nomination, nomination)))
        return false;
      current.pending_nomination = nomination;
      current.pending_nomination.tick = GetTickCount64();
      current.pending_nomination.present = present_ordinal();
      return true;
    }

    // An adapter's input names its retained source and resource as well.
    bool mark_pending(evaluation_namespace &current, const input &value, bool renewed) {
      return value.source && value.resource.native && mark_pending(current, nomination_of(value), renewed);
    }
  }

  SUNSHINE_CAPTURE_STATIC bool begin_nomination(const input &value, std::uint64_t superseded_source_id) {
    std::lock_guard lock(mutex);
    begin_evaluation_locked(value.epoch, value.sequence, value.viewport, value.provider, value.source_id, superseded_source_id);
    if (!requested.load() || !valid_provider(value.provider)) return false;
    return mark_pending(evaluations[provider_index(value.provider)], value, true);
  }

  void evaluation_nomination::begin(std::uint64_t epoch, std::uint64_t sequence, std::uint32_t viewport,
      provider_kind provider, std::uint64_t source_id, std::uint64_t superseded_source_id) {
    end();
    std::lock_guard lock(mutex);
    begin_evaluation_locked(epoch, sequence, viewport, provider, source_id, superseded_source_id);
    if (!requested.load() || !valid_provider(provider)) return;
    const evaluation_head nomination{epoch, sequence, 0, source_id, viewport};
    if (mark_pending(evaluations[provider_index(provider)], nomination, true))
      nominated = {provider, epoch, sequence, source_id, viewport, true};
  }

  void evaluation_nomination::end() {
    if (!std::exchange(nominated.pending, false)) return;
    end_nomination(nominated.provider, {nominated.epoch, nominated.sequence, 0, nominated.source_id, nominated.viewport});
  }

  SUNSHINE_CAPTURE_STATIC bool source_lifetime_current(const input &value, std::uint64_t current_generation) {
    // A copy recorded synchronously inside the authenticated call that
    // supplied the resource (a tag call, or an evaluation's own local input)
    // is current by construction, for OnlyValidNow and UntilPresent alike: an
    // independent FG Present of an earlier frame cannot end this call's
    // lifetime. With frame generation such Presents land between reading the
    // input and recording its copy on a fraction of real frames, more often
    // while new storage is allocated without the lock (Assassin's Creed
    // Shadows 10-24: its SR evaluation's local depth input, each dynamic
    // resolution change). A later use of the tag's own frame (own_frame) is current for
    // the same reason: a Present since the tag was an earlier frame's (the
    // matrix's S7). Any other later use, and a delayed preservation copy,
    // still require an unchanged generation.
    if (value.at_tag_call || value.own_frame ||
        (value.valid_until == sunshine_scene_depth::lifetime::at_call && value.force_snapshot))
      return true;
    return value.source_present_generation && value.source_present_generation == current_generation;
  }

  // proven_queue: the queue that executes native, for a last_proven proof.
  static std::uint64_t record_impl(std::uint64_t native, const input &value, record_diagnostic *diagnostic,
      bool preservation_only, preservation_ticket *preserved = nullptr, bool nominate_source = false,
      bool diagnostic_copy = false, bool reusable_storage = false,
      texture_state_policy state_policy = texture_state_policy::source_contract, std::uint64_t proven_queue = 0) {
    bool valid_nomination = false;
    std::uint64_t nomination_command{}, nomination_source{};
    recording_ref nomination_recording;
    if (diagnostic) {
      *diagnostic = {};
      diagnostic->command = native; diagnostic->resource = value.resource.native;
      diagnostic->expected_generation = value.source_present_generation;
      diagnostic->native_state = value.native_state;
      if (value.source) {
        diagnostic->expected_device_identity = value.source->device_identity;
        diagnostic->width = static_cast<std::uint32_t>(value.source->desc.Width);
        diagnostic->height = value.source->desc.Height; diagnostic->format = value.source->desc.Format;
        diagnostic->flags = value.source->desc.Flags;
        diagnostic->dimension = value.source->desc.Dimension; diagnostic->mip_levels = value.source->desc.MipLevels;
        diagnostic->array_size = value.source->desc.DepthOrArraySize; diagnostic->samples = value.source->desc.SampleDesc.Count;
      }
    }
    const auto reject = [&](status result, record_stage stage) -> std::uint64_t {
      if (diagnostic) { diagnostic->result = result; diagnostic->stage = stage; }
      if (!preservation_only && !value.pixels_required) set_attempt(value.provider, result);
      if (nominate_source && valid_nomination && !value.pixels_required) {
        // Pixel capability must not determine source authority. This carries
        // the validated source through original evaluation completion and real
        // submission without fabricating an owned copy or another source.
        const auto ticket = record_nomination(value, nomination_source, nomination_command, nomination_recording);
        if (ticket) {
          for (auto &entry : capture_pool.slots) if (entry.id == ticket) {
            entry.nomination_only = entry.source_nominated = true;
            entry.pixel_failure = result; break;
          }
        }
        return ticket;
      }
      return 0;
    };
    if (!requested.load()) return reject(status::malformed, record_stage::inactive);
    const bool prefer_observed_recording = state_policy == texture_state_policy::prefer_observed_recording;
    if (state_policy != texture_state_policy::source_contract &&
        (!prefer_observed_recording || !preservation_only || !diagnostic_copy ||
          value.provider != provider_kind::streamline || !value.force_snapshot ||
          value.valid_until != sunshine_scene_depth::lifetime::at_call))
      return reject(status::malformed, record_stage::malformed_input);
    if (!native) return reject(status::malformed, record_stage::malformed_input);
    // A source released with its device (release_device): its resource, which
    // value.resource names, may be gone.
    if (!value.source || value.source->released.load(std::memory_order_acquire))
      return reject(status::malformed, record_stage::missing_source);
    if ((!preservation_only && (!valid_provider(value.provider) || !valid_projection(value.projection) ||
        !value.epoch || !value.sequence)) || !value.resource.native)
      return reject(status::malformed, record_stage::malformed_input);
    if (value.valid_until != sunshine_scene_depth::lifetime::until_present &&
        value.valid_until != sunshine_scene_depth::lifetime::until_evaluation &&
        !(value.valid_until == sunshine_scene_depth::lifetime::at_call && value.force_snapshot))
      return reject(status::unsupported_lifetime, record_stage::unsupported_lifetime);
    // Every capture copies this call's contents and needs a state proof.
    const auto state_rule = copy_state::select(value.proof, state_policy);
    if (state_rule == copy_state::rule::none)
      return reject(status::unsupported_state, record_stage::unsupported_proof);
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native, IID_ID3D12GraphicsCommandList, checked))
      return reject(status::unsupported_queue, record_stage::command_interface);
    com_ptr<ID3D12Resource> checked_source;
    if (!query_native(value.resource.native, IID_ID3D12Resource, checked_source) ||
        source_cookie(checked_source.p) != value.source->cookie)
      return reject(status::unsupported_resource, record_stage::source_identity);
    native = reinterpret_cast<std::uint64_t>(checked.p);
    if (diagnostic) diagnostic->command = native;
    observe_command(native);
    // The barrier hooks supply the list's resource states only when they see it.
    const bool states_observed = native_observer::states_ready(native);
    // A proven state holds for a queue whose executions the hooks report. It
    // supplies a last_proven proof, and checks the declaration of a copy
    // deferred to its frame's Present (input::present_queue) as an
    // observation would: one that disagrees refuses the copy.
    com_ptr<ID3D12CommandQueue> present_queue;
    if (!proven_queue && query_native(value.present_queue, IID_ID3D12CommandQueue, present_queue))
      proven_queue = reinterpret_cast<std::uint64_t>(present_queue.p); // As the submission hook reports it.
    const bool proven_basis_allowed = (value.proof == sunshine_scene_depth::state_proof::last_proven ||
        (value.present_queue && value.proof == sunshine_scene_depth::state_proof::declared)) && states_observed &&
      proven_queue && native_observer::queue_ready(proven_queue);
    auto *list = checked.p;
    const auto command_type = list->GetType();
    if (diagnostic) diagnostic->command_type = command_type;
    // NGX and Streamline evaluate on any non-copy queue (the matrix's S22 and
    // N7), so a copy is recorded on a DIRECT or COMPUTE list, from a basis
    // that list may name (copy_state::list_basis).
    const bool on_compute = command_type == D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (command_type != D3D12_COMMAND_LIST_TYPE_DIRECT && !on_compute)
      return reject(status::unsupported_queue, record_stage::command_type);
    com_ptr<ID3D12Device> list_device;
    if (FAILED(list->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(list_device.put()))))
      return reject(status::unsupported_queue, record_stage::command_device);
    const auto device_identity = device_cookie(list_device.p);
    if (diagnostic) diagnostic->device_identity = device_identity;
    if (device_identity != value.source->device_identity) return reject(status::unsupported_queue, record_stage::device_identity);
    const auto cookie = native_observer::get_recording_cookie(native);
    if (diagnostic) diagnostic->recording_cookie = cookie;
    // Game threads never hold the capture lock while the driver allocates.
    // An attempt that needs new storage reserves a slot (and its diagnostic
    // budget) under the lock, allocates unlocked, then repeats every
    // admission check from the start under the lock before it publishes. A
    // reservation is only a preference: publication never trusts it.
    std::shared_ptr<texture_reference> prepared, prepared_published;
    // Copy-out storage this call failed to allocate or register.
    bool copy_out_refused = false;
    std::uint64_t reservation{};
    // Storage replaced under the lock is released after it, on a pool thread
    // (Release of a large committed resource can take milliseconds on the
    // game's recording thread), except what is released before an allocation
    // to bound its peak (discarded.clear() below).
    pooled_storage discarded;
    for (unsigned attempt = 0;; ++attempt) {
      std::unique_lock lock(mutex);
      valid_nomination = false;
      // Every exit below holds the lock, and a reservation never outlives the
      // call: only the retry after an unlocked allocation keeps it.
      struct reservation_release {
        std::uint64_t &token;
        bool keep{};
        ~reservation_release() { if (token && !keep) clear_reservation(std::exchange(token, 0)); }
      } release_on_exit{reservation};
      // A compute list's copy is accepted only once its queue holds a place
      // (register_compute_producer): with none left for it and no compute
      // queue holding one, no GPU copy is recorded that could never be read.
      if (on_compute && compute_unplaced && !compute_producer_placed()) {
        if (!free_place()) return reject(status::exhausted, record_stage::capacity);
        compute_unplaced = false;
      }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      if (nominate_source && !preservation_only && !diagnostic_copy &&
          present_before_next_copy.exchange(false, std::memory_order_acq_rel))
        ++value.source->present_generation;
#endif
      if (!value.source->resource.p) return reject(status::malformed, record_stage::missing_source);
      const auto current_generation = source_present_generation(value.source);
      if (diagnostic) diagnostic->current_generation = current_generation;
      if (!source_lifetime_current(value, current_generation))
        return reject(status::unsupported_lifetime, record_stage::source_lifetime);
      auto owner = command(native, cookie, true);
      // ReShade's lifecycle decides coverage: the recording is open outside a
      // render pass.
      const auto view = lifecycle_view(owner ? &*owner : nullptr);
      if (!attempt) count_admission(view, states_observed);
      if (view == list_view::unknown) return reject(status::unavailable, record_stage::observer_coverage);
      if (diagnostic) {
        diagnostic->loss = owner->invalidation; diagnostic->recording_closed = owner->closed;
        diagnostic->recording_invalid = owner->invalid; diagnostic->render_pass = owner->render_pass;
        diagnostic->native_barrier_calls = owner->native_barrier_calls;
        diagnostic->native_transition_count = owner->native_transition_count;
        diagnostic->tracked_source_count = owner->states.size();
        diagnostic->last_barrier_command = owner->last_barrier_command;
        diagnostic->source_cookie = value.source->cookie;
      }
      if (view == list_view::closed) return reject(status::unavailable, record_stage::recording_closed);
      if (view == list_view::pass) return reject(status::unavailable, record_stage::recording_render_pass);
      const auto &desc = value.source->desc;
      copy_region region;
      const auto source_status = source_region(desc, value.resource, region);
      if (source_status != status::ready) return reject(source_status, record_stage::resource_region);
      valid_nomination = nominate_source;
      nomination_command = cookie; nomination_source = reinterpret_cast<std::uint64_t>(checked_source.p);
      nomination_recording = owner.life();
      const auto region_status = diagnostic_copy ?
        (diagnostic_description(desc, value.resource.native_format) ? source_region(desc, value.resource, region) :
          status::unsupported_resource) :
        capture_region(desc, value.resource, region);
      if (region_status != status::ready) return reject(region_status, record_stage::resource_region);
      // A depth snapshot is read through one view, resolved now; a linear
      // R16_TYPELESS without a declared format has none.
      const auto snapshot_view = diagnostic_copy ? DXGI_FORMAT_UNKNOWN : depth_view(desc.Format, value);
      if (!diagnostic_copy && snapshot_view == DXGI_FORMAT_UNKNOWN)
        return reject(status::unsupported_resource, record_stage::resource_region);
      if (owner->invalid) return reject(status::unavailable, record_stage::recording_invalid);
      // Unobserved lists (refused or moved method tables) use the declared state.
      const auto *observed = states_observed ? state(*owner, value.source->cookie, false) : nullptr;
      if (diagnostic && observed) {
        diagnostic->observed = true; diagnostic->observed_state = observed->value; diagnostic->blocked = observed->blocked;
        diagnostic->pending_split = observed->pending_split;
        diagnostic->observed_enhanced = observed->enhanced;
        diagnostic->observed_layout = observed->layout; diagnostic->observed_access = observed->access;
      }
      // A last proven state stands in for the observation this recording
      // lacks: the source's state proven on the copy's queue, unless this
      // recording aliased it before the copy (proven_state.h).
      sunshine_proven_state::state proven_basis;
      if (proven_basis_allowed && !observed && !owner->wildcard_alias && !aliased_after(*owner, value.source->cookie, 0))
        proven_basis = capture_observation.proven.query(value.source->cookie, proven_queue, !capture_observation.enhanced_barriers_seen.load(std::memory_order_relaxed));
      // Otherwise a source an enhanced barrier on an earlier executed list
      // latched (proven_state.h) takes no legacy barrier from its claim: the
      // claim is checked against its proof from that list (copy_state's
      // earlier observation, the matrix's D6 and N8). That proof stands only
      // as one above would: the hooks see this recording, which did not alias
      // it before the copy, and it was made on the copy's queue when that is
      // known; else the copy is refused.
      sunshine_proven_state::latch earlier;
      if (!observed && !proven_basis) {
        earlier = capture_observation.proven.latched(value.source->cookie);
        if (!states_observed || owner->wildcard_alias || aliased_after(*owner, value.source->cookie, 0) ||
            (proven_queue && earlier.proven.queue != proven_queue))
          earlier.proven = {};
      }
      const auto &proof = proven_basis ? proven_basis : earlier.proven;
      if (diagnostic) diagnostic->earlier_layout = earlier.latched;
      if (diagnostic && proof) {
        diagnostic->used_proven_state = true; diagnostic->observed_state = proof.value;
        diagnostic->observed_enhanced = proof.enhanced;
        diagnostic->observed_layout = proof.layout; diagnostic->observed_access = proof.access;
      }
      auto evidence = proof ? copy_state::observation{true, true, false, proof.value, false,
        proof.enhanced, proof.layout, proof.access} : observation_of(observed);
      evidence.earlier = earlier.latched;
      const auto chosen = copy_state::resolve(state_rule, value.native_state, evidence, desc.Flags);
      if (!chosen.admitted) {
        if (earlier.latched) capture_observation.proven_counters.earlier_refused.fetch_add(1, std::memory_order_relaxed);
        return reject(chosen.result, chosen.stage);
      }
      // A compute list's copy leaves and restores only a state or layout that
      // list may name, a direct list's none of a compute queue's own layouts
      // (S22b).
      if (!copy_state::list_basis(chosen, on_compute))
        return reject(status::unsupported_state, record_stage::unsupported_state);
      // The D3D12 contract forbids a legacy barrier on a texture in an
      // enhanced layout: that copy needs this list's CommandList7 Barrier.
      com_ptr<ID3D12GraphicsCommandList7> enhanced_list;
      if (chosen.enhanced && !query_native(native, IID_ID3D12GraphicsCommandList7, enhanced_list))
        return reject(status::unsupported_queue, record_stage::command_interface);
      if (diagnostic) diagnostic->enhanced_copy = chosen.enhanced;
      const auto before = chosen.state;
      const auto basis = proof ? state_basis::proven : copy_basis(observed, chosen);
      if (diagnostic) {
        diagnostic->copy_state = before; diagnostic->copy_state_known = true;
        diagnostic->used_observed_state = chosen.used_observed;
        diagnostic->used_contract_state = chosen.used_contract;
      }
      // Multiple legal clear boundaries in one unsubmitted recording may replace
      // the same source snapshot before anyone reads it. Old tickets are retired
      // by ID; their CPU leases alone do not represent recorded GPU reads.
      if (diagnostic_copy) collect_diagnostics(discarded);
      auto *pixel_begin = diagnostic_copy ? capture_pool.diagnostic_slots.data() : capture_pool.slots.data();
      auto *pixel_end = pixel_begin + (diagnostic_copy ? diagnostic_slot_limit : pixel_slot_limit);
      const auto texture_format = diagnostic_copy ? diagnostic_storage_format(desc.Format, value.resource.native_format) :
        typeless(desc.Format);
      // A shared copy is opened by the host: only formats its table knows.
      if (diagnostic_copy && !reusable_storage && !game3d_debug::pixel_bytes(static_cast<unsigned>(texture_format)))
        return reject(status::unsupported_resource, record_stage::resource_region);
      const auto compatible_storage = [&](const auto &texture) {
        if (!texture || texture->device_identity != value.source->device_identity) return false;
        if (diagnostic_copy && bool(texture->shared_handle) == reusable_storage) return false;
        // A compute list cannot leave the pixel-shader state a depth
        // snapshot rests in; a direct list may use storage of either rest.
        if (on_compute && texture->rest != D3D12_RESOURCE_STATE_COMMON) return false;
        const auto existing = texture->resource->GetDesc();
        return existing.Width == region.width && existing.Height == region.height && existing.Format == texture_format &&
          (diagnostic_copy || texture->view_format == snapshot_view);
      };
      // Another attempt's live reservation is not available to this one.
      const auto free_for_us = [&](const slot &entry) { return !reserved_by_other(entry, reservation); };
      auto *found = preservation_only && !diagnostic_copy ? std::find_if(pixel_begin, pixel_end, [&](const auto &entry) {
        return reusable_preserved(entry, value, cookie);
      }) : pixel_end;
      // Prefer the slot this attempt reserved, if it is still empty and free.
      if (found == pixel_end && reservation) found = std::find_if(pixel_begin, pixel_end, [&](auto &entry) {
        return entry.reservation == reservation && !entry.id;
      });
      if (found == pixel_end && diagnostic_copy && reusable_storage) found = std::find_if(pixel_begin, pixel_end, [&](const auto &entry) {
        return !entry.id && free_for_us(entry) && compatible_storage(entry.texture);
      });
      // Do not overwrite the only newly completed API snapshot with its pending
      // successor before the effects queue can consume it. This borrows an
      // existing pool slot; no extra owner, queue, or resource copy is introduced.
      const auto *completed = nominate_source ? completed_snapshot(value, GetTickCount64()) : nullptr;
      // A Dump 3D copy (never cached) takes only a slot without live storage:
      // a live snapshot reuses its cache every frame. A full pool is exhausted.
      const bool dump_copy = diagnostic_copy && !reusable_storage;
      if (found == pixel_end) found = std::find_if(pixel_begin, pixel_end, [&](auto &entry) {
        return &entry != completed && free_for_us(entry) && reclaimable(entry) &&
          !(dump_copy && entry.texture && live_storage(entry));
      });
      if (found == pixel_end) {
        capacity_refused(diagnostic_copy ? auxiliary_pool : snapshot_pool);
        return reject(status::exhausted, record_stage::capacity);
      }
      auto texture = found->texture;
      // Copy-out storage (slot::published) for a copy of a source another
      // queue reads (copy_out): the snapshot's shape, view and rest. Each
      // comes from the slot's cache, else from an earlier attempt's
      // allocation (prepared), else is allocated now. U-C1b: the queue a
      // recording executes on is known only at its submission, so a copy
      // also takes it on a COMPUTE list, which never executes on the
      // presenting queue that reads depth, and on a list not known to
      // execute on the reader's queue: never executed yet, or last executed
      // on another queue, as the render queue's lists are once FG On moved
      // Present to DLSS-G's queue (the idle sweep releases it when the
      // submission is the reader's after all). A list never executed is
      // predicted to run where the newest DIRECT-list copy ran
      // (last_direct_copy_queue): with FG Off, the reader's own queue, so a
      // game that pools or creates its lists allocates nothing for them
      // (until 10-27 each new list's first copy allocated copy-out storage on
      // the game's thread, released unused on the Present thread). Storage
      // this attempt could not allocate is not sought again (copy_out
      // failed): the copy keeps the Reset rule. A list predicted to run on
      // the reader's queue takes it too while FG may move Present off that
      // queue (reader_may_move: the copies of FG's first Present).
      const auto record_tick = GetTickCount64();
      const bool read_elsewhere_before = read_elsewhere_recently(*value.source, record_tick);
      // Cross-section read: storage prediction consults the current consumer
      // reader; a queue's diagnostic chosen flag is not this ownership fact.
      const bool predicted = !read_elsewhere_before && !on_compute && reader_live() &&
        ((owner->executed_queue ? owner->executed_queue != reader : last_direct_copy_queue != reader) ||
          reader_may_move(record_tick));
      const bool wants_published = !diagnostic_copy && !preservation_only && !copy_out_refused &&
        (read_elsewhere_before || on_compute || predicted);
      auto published = wants_published ? found->published : nullptr;
      // New storage's rest (texture_reference::rest).
      const auto resting_state = diagnostic_copy || on_compute ? D3D12_RESOURCE_STATE_COMMON : sampled_state;
      const bool texture_cached = compatible_storage(texture);
      const bool published_cached = !wants_published || compatible_storage(published);
      const bool texture_prepared = !texture_cached && prepared && compatible_storage(prepared);
      const bool published_prepared = !published_cached && prepared_published && compatible_storage(prepared_published);
      const bool need_texture = !texture_cached && !texture_prepared;
      bool need_published = !published_cached && !published_prepared;
      if (need_published && copy_out_failure.tick && copy_out_failure.device_identity == value.source->device_identity &&
          copy_out_failure.width == region.width && copy_out_failure.height == region.height &&
          copy_out_failure.format == texture_format &&
          sunshine_freshness::fresh_lt(copy_out_failure.tick, record_tick, sunshine_freshness::source_age_ms)) {
        // An allocation of this shape failed within the source age: the copy
        // keeps the Reset rule unless copy_out finds idle storage for it.
        need_published = false;
        published = nullptr;
        copy_out_refused = true;
        capture_queues.copy_out_counters.skipped.fetch_add(1, std::memory_order_relaxed);
      }
      if (!need_texture && !need_published) {
        if (texture_prepared) {
          // Storage allocated by an earlier attempt. Its budget was reserved on
          // the slot it reserved; a different slot must admit it again.
          if (diagnostic_copy && found->reservation != reservation &&
              !reserve_diagnostic_bytes(capture_pool.diagnostic_slots, found, prepared->allocation_bytes, reusable_storage, reservation, &discarded))
            return reject(status::exhausted, record_stage::capacity);
          texture = std::move(prepared);
        }
        if (published_prepared) published = std::move(prepared_published);
      } else {
        // Bounded: a second allocation means the shape changed meanwhile.
        if (attempt >= 2) return reject(status::exhausted, record_stage::capacity);
        storage_plan plan;
        plan.device = value.source->device;
        plan.device_identity = value.source->device_identity;
        auto &destination = plan.desc;
        destination = desc; destination.Format = texture_format; destination.Flags = D3D12_RESOURCE_FLAG_NONE;
        destination.Width = region.width; destination.Height = region.height;
        destination.MipLevels = 1; destination.DepthOrArraySize = 1; // Subresource 0 only.
        plan.state = resting_state;
        if (diagnostic_copy) {
          destination.Alignment = 0;
          destination.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
          // D3D11's host OpenSharedResource1 requires the D3D12 shared color
          // allocation to be RT-capable. This is a private-copy bind capability,
          // never a requirement on the application's mask/input allocation.
          D3D12_FEATURE_DATA_FORMAT_SUPPORT support{texture_format};
          if (FAILED(plan.device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
              !(support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET))
            return reject(status::unsupported_resource, record_stage::resource_region);
          destination.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
          const auto allocation = plan.device->GetResourceAllocationInfo(0, 1, &destination);
          if (!reserve_diagnostic_bytes(capture_pool.diagnostic_slots, found, allocation.SizeInBytes, reusable_storage, reservation, &discarded))
            return reject(status::exhausted, record_stage::capacity);
          plan.allocation_bytes = allocation.SizeInBytes;
          plan.shared = !reusable_storage;
          // Local snapshots can be bound directly (lease_local_view).
          plan.descriptors = reusable_storage;
          // The replacement is retired too. Release its incompatible storage
          // before allocation so even the replacement peak respects the budget.
          discarded.push_back(std::move(found->texture));
        } else {
          plan.descriptors = true;
          plan.view = snapshot_view;
          // Sizes the idle-storage budget (release_idle_slots).
          plan.allocation_bytes = plan.device->GetResourceAllocationInfo(0, 1, &destination).SizeInBytes;
        }
        clear_reservation(reservation);
        reservation = ++reservation_serial;
        found->reservation = reservation; found->reservation_tick = GetTickCount64();
        found->reserved_bytes = plan.allocation_bytes; found->reserved_live = diagnostic_copy && reusable_storage;
        lock.unlock();
        discarded.clear();
        if (need_texture) prepared.reset();
        if (need_published) prepared_published.reset();
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
        if (auto *hook = testing::allocation_hook.exchange(nullptr)) hook(testing::allocation_context.load());
#endif
        auto stage = need_texture ? allocate_storage(plan, prepared) : record_stage::recorded;
        // Copy-out storage is optional: a copy without it keeps the Reset
        // rule, as before U-C1b, and its submission records its list's
        // queue, so a list never executed is not predicted again.
        bool published_failed = false;
        if (stage == record_stage::recorded && need_published) {
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
          if (testing::fail_copy_out_allocation()) published_failed = true;
          else
#endif
          published_failed = allocate_storage(plan, prepared_published) != record_stage::recorded;
        }
        lock.lock();
        if (stage != record_stage::recorded) return reject(status::failed, stage);
        // Released with its device whoever holds it (release_device).
        if (need_texture && !register_storage(prepared)) return reject(status::failed, record_stage::texture_allocation);
        if (need_published && !published_failed && !register_storage(prepared_published)) published_failed = true;
        if (published_failed) {
          discarded.push_back(std::move(prepared_published));
          copy_out_refused = true;
          capture_queues.copy_out_counters.failed.fetch_add(1, std::memory_order_relaxed);
          copy_out_failure = {plan.device_identity, plan.desc.Width, plan.desc.Height, GetTickCount64(), plan.desc.Format};
        }
        release_on_exit.keep = true;
        continue;
      }
      // Publication: this attempt passed every check above with the lock held.
      discarded.push_back(std::move(found->texture));
      discarded.push_back(std::move(found->published));
      *found = {};
      if (reservation) clear_reservation(std::exchange(reservation, 0));
      found->texture = std::move(texture); found->published = std::move(published); found->metadata = value; found->metadata.native_state = before;
      retire_frame_generation_role(found->metadata);
      found->predicted_storage = predicted && found->published;
      if (found->predicted_storage) capture_queues.copy_out_counters.predicted.fetch_add(1, std::memory_order_relaxed);
      found->id = ++serial; found->command = cookie; found->producer_recording = owner.life();
      found->present = present_ordinal();
      if (found->metadata.frame_generation_input) stamp_frame_generation(found->present);
      found->basis = basis; found->compute_list = on_compute;
      compute_copies_recorded |= on_compute;
      if (proof) {
        found->proven_source = value.source->cookie; found->proven_version = proof.version;
        found->proven_queue = proven_basis ? proven_queue : proof.queue;
      }
      if (earlier.latched) capture_observation.proven_counters.earlier_copied.fetch_add(1, std::memory_order_relaxed);
      found->preservation_only = preservation_only;
      found->diagnostic_only = diagnostic_copy;
      found->diagnostic_reusable = reusable_storage;
      if (diagnostic_copy) capture_pool.diagnostic_slots_active = true;
      found->source_nominated = nominate_source;
      found->finished = found->success = preservation_only && !diagnostic_copy;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      if (nominate_source && !preservation_only && !diagnostic_copy)
        newest_published_copy.store(found->id | (found->metadata.frame_generation_input ? 1ull << 63 : 0), std::memory_order_release);
#endif
      native_observer::suppression_scope suppress;
      D3D12_RESOURCE_BARRIER transitions[2]{};
      transitions[0].Type = transitions[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      transitions[0].Transition = {value.source->resource.p, 0, static_cast<D3D12_RESOURCE_STATES>(before), D3D12_RESOURCE_STATE_COPY_SOURCE};
      transitions[1].Transition = {found->texture->resource.p, 0, found->texture->rest, D3D12_RESOURCE_STATE_COPY_DEST};
      // An enhanced source leaves its exact layout and access for the copy
      // after all prior work (subresource 0: the zero range is index form).
      D3D12_TEXTURE_BARRIER source_layout{};
      D3D12_BARRIER_GROUP source_group{};
      if (chosen.enhanced) {
        source_layout.SyncBefore = D3D12_BARRIER_SYNC_ALL; source_layout.SyncAfter = D3D12_BARRIER_SYNC_COPY;
        source_layout.AccessBefore = static_cast<D3D12_BARRIER_ACCESS>(chosen.access);
        source_layout.AccessAfter = D3D12_BARRIER_ACCESS_COPY_SOURCE;
        source_layout.LayoutBefore = static_cast<D3D12_BARRIER_LAYOUT>(chosen.layout);
        source_layout.LayoutAfter = static_cast<D3D12_BARRIER_LAYOUT>(copy_state::layouts::copy_source(chosen.layout));
        source_layout.pResource = value.source->resource.p;
        source_group.Type = D3D12_BARRIER_TYPE_TEXTURE; source_group.NumBarriers = 1; source_group.pTextureBarriers = &source_layout;
        list->ResourceBarrier(1, &transitions[1]);
        enhanced_list->Barrier(1, &source_group);
      } else if (before == D3D12_RESOURCE_STATE_COPY_SOURCE) list->ResourceBarrier(1, &transitions[1]);
      else list->ResourceBarrier(2, transitions);
      D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
      source.pResource = value.source->resource.p; source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      destination.pResource = found->texture->resource.p; destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      // Subresource zero is mip zero/array zero/plane zero. capture_region rejects
      // multi-mip, arrays and MSAA. Packed stencil planes remain untouched.
      list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
      std::swap(transitions[0].Transition.StateBefore, transitions[0].Transition.StateAfter);
      std::swap(transitions[1].Transition.StateBefore, transitions[1].Transition.StateAfter);
      if (chosen.enhanced) {
        // Restore the exact layout and access. SyncAfter ALL makes every later
        // command wait for the copy: the enhanced form of the legacy write
        // boundary below, so a later game write cannot overlap the read.
        std::swap(source_layout.AccessBefore, source_layout.AccessAfter);
        std::swap(source_layout.LayoutBefore, source_layout.LayoutAfter);
        source_layout.SyncBefore = D3D12_BARRIER_SYNC_COPY; source_layout.SyncAfter = D3D12_BARRIER_SYNC_ALL;
        enhanced_list->Barrier(1, &source_group);
      } else if (before != D3D12_RESOURCE_STATE_COPY_SOURCE && !(before & write_states)) {
        // The asynchronous copy/reuse regression exposes partial later writes
        // when restoring only a read state here. Complete the injected source
        // read at an explicit write-state boundary before restoring the game's
        // exact state. This adds ordering, never a write to the game resource.
        auto completion = transitions[0];
        completion.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        list->ResourceBarrier(1, &completion);
        transitions[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      }
      if (chosen.enhanced || before == D3D12_RESOURCE_STATE_COPY_SOURCE) list->ResourceBarrier(1, &transitions[1]);
      else list->ResourceBarrier(2, transitions);
      if (!preservation_only) set_attempt(value.provider, status::recorded);
      if (nominate_source && !preservation_only) record_frame_generation_copy(found->metadata);
      if (diagnostic) { diagnostic->result = status::recorded; diagnostic->stage = record_stage::recorded; }
      if (preserved) *preserved = {found->id, reinterpret_cast<std::uint64_t>(found->texture->resource.p),
        found->texture->identity, found->texture};
      return found->id;
    }
  }

  std::uint64_t record(std::uint64_t command, const input &value, record_diagnostic *diagnostic) {
    return record_impl(command, value, diagnostic, false);
  }

  static diagnostic_ticket record_auxiliary_texture(std::uint64_t command, const input &value,
      record_diagnostic *diagnostic, bool reusable_storage,
      texture_state_policy state_policy = texture_state_policy::source_contract) {
    try {
      const auto id = record_impl(command, value, diagnostic, true, nullptr, false, true, reusable_storage, state_policy);
      if (!id) return {};
      std::lock_guard lock(mutex);
      for (const auto &entry : capture_pool.diagnostic_slots) if (entry.id == id) return {id, entry.texture};
    } catch (...) {
      // Like depth preservation, every allocation precedes the injected copy.
      // An allocation failure never escapes into the application's API call.
      if (diagnostic) { diagnostic->result = status::exhausted; diagnostic->stage = record_stage::capacity; }
    }
    return {};
  }

  diagnostic_ticket record_diagnostic_texture(std::uint64_t command, const input &value, record_diagnostic *diagnostic,
      texture_state_policy state_policy) {
    return record_auxiliary_texture(command, value, diagnostic, false, state_policy);
  }

  diagnostic_ticket record_local_texture(std::uint64_t command, const input &value, record_diagnostic *diagnostic,
      texture_state_policy state_policy) {
    return record_auxiliary_texture(command, value, diagnostic, true, state_policy);
  }

  std::uint64_t nominate(std::uint64_t command, const input &value, record_diagnostic *diagnostic) {
    // An API identifies the contents at this call, not whichever partial pass
    // Generic later preserves from the same resource. Keep both adapters on
    // the shared immutable snapshot/retirement path; never substitute a
    // clear-count heuristic for the middleware's capture boundary. An
    // undeclared state needs this recording's observation (copy_state::select).
    // The copy of an evaluation begun earlier (begin_evaluation: NGX, DLSS-G)
    // is its nomination in progress as nominate_evaluation's is.
    bool pending = false;
    if (requested.load() && valid_provider(value.provider)) {
      std::lock_guard lock(mutex);
      pending = mark_pending(evaluations[provider_index(value.provider)], value, false);
    }
    struct nomination_scope {
      const input &value;
      bool pending;
      ~nomination_scope() { if (pending) end_nomination(value); }
    } completion{value, pending};
    return record_impl(command, value, diagnostic, false, nullptr, true);
  }

  std::uint64_t evaluation_nomination::nominate(std::uint64_t command, const input &value, record_diagnostic *diagnostic) {
    // Its own copy was published at begin; another value (or none begun)
    // takes nominate's own path.
    if (nominated.pending && nominated.provider == value.provider &&
        same_nomination({nominated.epoch, nominated.sequence, 0, nominated.source_id, nominated.viewport}, nomination_of(value)))
      return record_impl(command, value, diagnostic, false, nullptr, true);
    return depth_capture::nominate(command, value, diagnostic);
  }

  std::uint64_t nominate_evaluation(std::uint64_t command, const input &value,
      std::uint64_t superseded_source_id, record_diagnostic *diagnostic) {
    return nominate_evaluation(command, &value, 1, superseded_source_id, diagnostic);
  }

  std::uint64_t nominate_evaluation(std::uint64_t command, const input *candidates, std::size_t count,
      std::uint64_t superseded_source_id, record_diagnostic *diagnostic) {
    const auto malformed = [&] {
      if (diagnostic) {
        *diagnostic = {}; diagnostic->command = command;
        diagnostic->result = status::malformed; diagnostic->stage = record_stage::malformed_input;
      }
      return std::uint64_t{};
    };
    if (!candidates || !count || count > 3) return malformed();
    const auto &value = candidates[0];
    for (std::size_t i = 1; i < count; ++i) {
      const auto &other = candidates[i];
      if (other.provider != value.provider || other.epoch != value.epoch || other.sequence != value.sequence ||
          other.source_id != value.source_id || other.viewport != value.viewport ||
          other.frame_generation_input != value.frame_generation_input) return malformed();
    }
    const bool pending = begin_nomination(value, superseded_source_id);
    struct nomination_scope {
      const input &value;
      bool pending;
      ~nomination_scope() { if (pending) end_nomination(value); }
    } completion{value, pending};
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    if (pending) wait_nomination_gate();
#endif
    struct fallback_nomination {
      std::uint64_t ticket{};
      record_diagnostic diagnostic;
      ~fallback_nomination() { if (ticket) finish(ticket, false); }
      std::uint64_t release() { return std::exchange(ticket, 0); }
    } fallback;
    for (std::size_t i = 0; i < count; ++i) {
      record_diagnostic captured;
      // A candidate of the nomination published above needs no second
      // capture-lock round trip to find it published.
      const auto ticket = pending ? record_impl(command, candidates[i], &captured, false, nullptr, true) :
        nominate(command, candidates[i], &captured);
      if (diagnostic) *diagnostic = captured;
      if (!ticket) continue;
      if (captured.result == status::recorded) return ticket;
      if (!fallback.ticket) { fallback.ticket = ticket; fallback.diagnostic = captured; }
      else finish(ticket, false);
    }
    if (fallback.ticket && diagnostic) *diagnostic = fallback.diagnostic;
    return fallback.release();
  }

  preservation_ticket record_preserved(std::uint64_t command, std::uint64_t source,
      const sunshine_api_contract::contract &boundary, std::uint64_t queue, record_diagnostic *diagnostic) {
    try {
      input value;
      value.source = retain_source(source, diagnostic);
      if (!value.source) return {};
      // A depth-stencil source is tracked from its creation; one a full
      // table forgot is tracked again, unproven until its next barrier or
      // clear proves it.
      if (value.source->desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) track_deferred_source(value.source);
      value.resource.native = source;
      sunshine_api_contract::apply(boundary, value);
      value.source_present_generation = source_present_generation(value.source);
      // The same queue pointer the submission hook reports.
      com_ptr<ID3D12CommandQueue> checked_queue;
      queue = queue && query_native(queue, IID_ID3D12CommandQueue, checked_queue) ?
        reinterpret_cast<std::uint64_t>(checked_queue.p) : 0;
      preservation_ticket ticket;
      record_impl(command, value, diagnostic, true, &ticket, false, false, false, texture_state_policy::source_contract, queue);
      return ticket;
    } catch (...) {
      // Allocation failure must not escape a ReShade recording callback. All
      // allocations precede recording the copy; no partial ticket is published.
      if (diagnostic) {
        *diagnostic = {}; diagnostic->command = command; diagnostic->resource = source;
        diagnostic->result = status::exhausted; diagnostic->stage = record_stage::capacity;
      }
      return {};
    }
  }

  preservation_ticket record_preserved(std::uint64_t command, std::uint64_t source,
      std::uint32_t native_state, record_diagnostic *diagnostic) {
    return record_preserved(command, source, sunshine_api_contract::generic::preserved(native_state), 0, diagnostic);
  }

  void finish(std::uint64_t ticket, bool successful, capture_failure failure) {
    std::lock_guard lock(mutex);
    for (auto &value : capture_pool.slots) if (value.id == ticket) {
      value.finished = true; value.success = successful;
      if (!successful) invalidate(value, failure == capture_failure::none ? capture_failure::evaluation_failed : failure);
      // The view's run of rejected evaluations (expire_association) and its
      // newest accepted copy (frame_generation_scope_live).
      const auto &m = value.metadata;
      if (!value.preservation_only && valid_provider(m.provider))
        for (auto &view : evaluations[provider_index(m.provider)].views) {
          if (view.epoch != m.epoch || view.viewport != m.viewport || view.source_id != m.source_id) continue;
          // A compute-list copy is accepted once readable (accept_compute_copy).
          if (successful && !value.nomination_only && !value.invalid && !value.compute_list) {
            view.accepted_tick = std::max(view.accepted_tick, m.tick);
            view.accepted_sequence = std::max(view.accepted_sequence, m.sequence);
          }
          if (m.sequence < view.result_sequence) continue;
          view.result_sequence = m.sequence;
          if (successful) view.rejected_since = 0;
          else if (!view.rejected_since) view.rejected_since = GetTickCount64();
        }
      accept_compute_copy(value);
      return;
    }
  }
}
