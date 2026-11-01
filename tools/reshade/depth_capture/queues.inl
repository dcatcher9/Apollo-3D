// SPDX-License-Identifier: GPL-3.0-only
// The capture owner's queues, a part of streamline_depth_capture.cpp
// (included at its end, in the same translation unit): queue places (lookup,
// registration, reclaiming an idle place, COMPUTE producers), producer and
// consumer submissions and their fences, whether another queue may read a
// copy (readable_elsewhere, the reader queue), copy-out on the producer
// queue, the submission hook, retired queues, and each queue's newest game
// execution. Every global these functions use stays in the shell in its
// original order; the shell's part declarations name the ones shell code
// calls. The Step 4 queue-order reads (readable_elsewhere's ordered branch,
// the producer positions in submitted) are calls into
// depth_capture/queue_order_consumer.inl.
namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    queue_state *known_queue(std::uint64_t native) {
      for (auto &value : capture_queues.queues) if (value && reinterpret_cast<std::uint64_t>(value->queue.p) == native) {
        value->last_use = ++capture_queues.place_clock;
        return value.get();
      }
      return nullptr;
    }
    // The caller owns mutex. Keep the one lookup and unlock/observe/relock order.
    void ensure_queue(std::unique_lock<std::mutex> &lock, std::uint64_t native) {
      if (!known_queue(native)) {
        lock.unlock();
        observe_queue(native);
        lock.lock();
      }
    }
    queue_state *queue(std::uint64_t native) {
      if (auto *value = known_queue(native)) return value;
      // A COM wrapper and its native queue share private data, just like the
      // command-recording cookie. Never substitute device identity for a queue.
      com_ptr<ID3D12CommandQueue> object;
      std::uint64_t cookie{}; UINT size = sizeof(cookie);
      if (!query_native(native, IID_ID3D12CommandQueue, object) ||
          FAILED(object->GetPrivateData(queue_guid, &size, &cookie)) || size != sizeof(cookie) || !cookie) return nullptr;
      for (auto &value : capture_queues.queues) if (value && value->cookie == cookie) {
        value->last_use = ++capture_queues.place_clock;
        return value.get();
      }
      return nullptr;
    }
    void producer_submitted(slot &value, std::uint64_t native, std::uint64_t fence, bool signaled) {
      if (value.producer_submitted && value.queue != native) {
        invalidate(value, capture_failure::producer_queue_changed);
        value.retirement_unknown = true;
        return; // Never relabel the original producer's fence as another queue.
      }
      value.retirement_unknown |= !signaled;
      // Submitted on a presenting queue (one that acquired depth): the
      // Present that shows this copy is decided by this submission's place
      // in that queue's order, not by when the CPU recorded it. A game that
      // records frame N+1 (its evaluation) before its submission thread
      // presents frame N stamped N+1's copy with Present N's ordinal, so its
      // own Present after a hitch never counted it as current (the 10-27
      // review's P2). The call stamp stays for a copy whose submission on
      // a presenting queue was not seen (presented_by).
      // Cross-section read: producer submission uses this native place's
      // diagnostic chosen flag as evidence that it has presented; reader differs.
      // It is also the newest FG frame's anchor when the copy is that
      // frame's (anchor_frame_generation): its real Present comes after.
      if (!value.producer_submitted && !value.submitted_present)
        if (const auto *presenting = known_queue(native); presenting && presenting->chosen) {
          value.submitted_present = present_ordinal();
          anchor_frame_generation(value.present, value.submitted_present);
        }
      value.producer_submitted = true;
      value.queue = native;
      value.producer_fence = signaled ? fence : 0;
      if (!signaled) invalidate(value, capture_failure::producer_signal_failed);
      // Submission and evaluation completion are independent. A pending result
      // remains unavailable to acquire(), but a later success can admit it.
      else if (value.finished && !value.success) invalidate(value, capture_failure::evaluation_failed);
    }
    // Submission to this queue accepts metadata-only nominations and otherwise needs its fence.
    bool submitted_success(const slot &value, std::uint64_t native) {
      return value.id && !value.invalid && value.finished && value.success &&
        value.producer_submitted && value.queue == native && (value.nomination_only || value.producer_fence);
    }
    queue_progress producer_progress(const slot &value) {
      queue_progress out;
      if (const auto *owner = known_queue(value.queue)) {
        out.device_identity = owner->device_identity;
        if (!value.nomination_only && owner->fence.p) { out.completed = owner->fence->GetCompletedValue(); out.queried = true; }
      }
      return out;
    }
    bool producer_recording_retired(const slot &value) {
      return !value.command || (value.producer_recording &&
        value.producer_recording->cookie.load(std::memory_order_acquire) != value.command);
    }
    // Another queue reads a snapshot only where no later execution of its
    // producer list can write it: its copy-out (copied_out), else its own
    // storage once that recording retired (Reset or destruction; the rule
    // where no copy-out was made). Its producer fence must complete as well.
    bool immutable_for_other_queues(const slot &value) {
      return value.copied_out || producer_recording_retired(value);
    }
    // The storage every consumer reads: the copy-out once made (on the
    // producer queue it follows the copy in queue order, as later
    // submissions there do), else the snapshot's own.
    const std::shared_ptr<texture_reference> &read_storage(const slot &value) {
      return value.copied_out && value.published ? value.published : value.texture;
    }
    // U-C1 (docs/reshade-sbs.md, Queue order): another queue reads an
    // immutable snapshot once its producer fence completed, or before that
    // once the queue-order model shows that every call consumer submits from
    // now on is ordered after the snapshot's producer position (its copy-out
    // list, else the game submission of a recording that has since retired).
    // consumer 0 keeps the completion rule. The add-on never waits.
    bool readable_elsewhere(const slot &value, std::uint64_t consumer, const queue_progress &progress,
        sunshine_queue_order::verdict *relied) {
      if (!progress.valid() || !value.producer_fence) return false;
      if (progress.completed >= value.producer_fence) return true;
      return ordered_after_producer(value, consumer, relied);
    }
    // Requires mutex. Whether the newest acquisition's queue (reader) is a
    // live place.
    bool reader_live() {
      return reader && std::any_of(capture_queues.queues.begin(), capture_queues.queues.end(), [](const auto &entry) {
        return entry && entry->cookie == reader && !entry->retiring;
      });
    }
    // Requires mutex. The native queue of the newest acquisition (reader).
    std::uint64_t reader_queue() {
      for (const auto &entry : capture_queues.queues)
        if (entry && reader && entry->cookie == reader && !entry->retiring) return reinterpret_cast<std::uint64_t>(entry->queue.p);
      return 0;
    }
    void consumer_submitted(slot &value, std::uint64_t native, std::uint64_t fence, bool signaled) {
      if (value.consumer_queue && value.consumer_queue != native) invalidate(value, capture_failure::consumer_queue_changed);
      if (!signaled) {
        invalidate(value, capture_failure::consumer_signal_failed); value.retirement_unknown = true; return;
      }
      auto found = std::find_if(value.retirements.begin(), value.retirements.end(),
        [native](const auto &point) { return point.queue == native; });
      if (found == value.retirements.end()) found = std::find_if(value.retirements.begin(), value.retirements.end(),
        [](const auto &point) { return !point.queue; });
      if (found == value.retirements.end()) {
        invalidate(value, capture_failure::consumer_capacity); value.retirement_unknown = true; return;
      }
      found->queue = native; found->value = std::max(found->value, fence);
    }
    // The runtime resets its immediate list after submitting it, so this one
    // observed submission (now followed by the queue fence) carries the read.
    void submitted_immediate(slot &value, std::uint32_t count, const native_observer::command_identity *values) {
      for (unsigned n = 0; n != count; ++n)
        if (value.pending_immediate && values[n].native_command == value.pending_immediate) value.pending_immediate = 0;
    }
    bool uses_queue(const slot &value, std::uint64_t native) {
      return value.queue == native || value.consumer_queue == native || std::any_of(value.retirements.begin(), value.retirements.end(),
        [native](const auto &point) { return point.queue == native; });
    }
    bool needs_submission_fence(const std::array<bool, slot_limit> &producers,
        const std::array<bool, slot_limit> &consumers) {
      for (unsigned i = 0; i != capture_pool.slots.size(); ++i)
        if ((!capture_pool.slots[i].nomination_only && producers[i]) || consumers[i]) return true;
      return false;
    }
    void capacity_refused(capacity_bound bound) { capacity_hits.fetch_or(bound, std::memory_order_relaxed); }
    // Requires mutex. A place for another queue: a free one, else the place of
    // the longest unused idle queue, which no capture names, which owns no
    // provider and whose signalled fence values have all completed (the
    // device's removal completes them). The idle queue registers again when
    // it next carries a copy (register_compute_producer) or presents
    // (observe_present), so no number of queues created over a session
    // exhausts the table.
    std::unique_ptr<queue_state> *free_place() {
      for (auto &entry : capture_queues.queues) if (!entry) return &entry;
      std::unique_ptr<queue_state> *idle = nullptr;
      for (auto &entry : capture_queues.queues) {
        if (entry->retiring || entry->provider_established || (idle && (*idle)->last_use <= entry->last_use)) continue;
        const auto native = reinterpret_cast<std::uint64_t>(entry->queue.p);
        bool used = false;
        for_each_capture([&](const slot &value) { used = used || (value.id && uses_queue(value, native)); });
        if (!used && (!entry->fence.p || entry->fence->GetCompletedValue() >= entry->next_fence)) idle = &entry;
      }
      return idle;
    }
    // Requires mutex. A queue's place: its device and the fence the capture
    // owner signals after the queue's submissions that carry a copy. DIRECT
    // queues register as ReShade reports them (observe_queue);
    // compute_producer also admits a COMPUTE queue, which only ever produces
    // (a copy recorded on a compute list: S22, N7).
    queue_state *register_queue(ID3D12CommandQueue *value, bool compute_producer) {
      const auto type = value->GetDesc().Type;
      if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && !(compute_producer && type == D3D12_COMMAND_LIST_TYPE_COMPUTE)) {
        reason = status::unsupported_queue; return nullptr;
      }
      auto *entry = free_place();
      if (!entry) {
        capacity_refused(queue_places);
        reason = status::exhausted;
        return nullptr;
      }
      if (*entry) { entry->reset(); places_reclaimed = true; }
      auto next = std::make_unique<queue_state>();
      if (FAILED(value->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(next->device.put()))) ||
          FAILED(next->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void **>(next->fence.put())))) {
        reason = status::unsupported_queue; return nullptr;
      }
      next->device_identity = device_cookie(next->device.p, true);
      if (!next->device_identity) { reason = status::unsupported_queue; return nullptr; }
      value->AddRef(); next->queue.p = value;
      next->compute_producer = type == D3D12_COMMAND_LIST_TYPE_COMPUTE;
      next->cookie = ++serial;
      next->last_use = ++capture_queues.place_clock;
      if (FAILED(value->SetPrivateData(queue_guid, sizeof(next->cookie), &next->cookie))) { reason = status::unsupported_queue; return nullptr; }
      *entry = std::move(next); return entry->get();
    }
    // Requires mutex. A COMPUTE queue that ReShade reported (or not) takes a
    // place at its first submission of a copy awaiting its producer: a copy
    // recorded on a compute list (record_impl). Without one, such a copy is
    // never submitted, and its recording's Reset discards it. So does any
    // queue whose idle place was reclaimed. While a compute queue found no
    // place (compute_unplaced), record_impl refuses compute-list copies until
    // one is free again (collect_retired, free_place).
    bool compute_producer_placed() {
      return std::any_of(capture_queues.queues.begin(), capture_queues.queues.end(),
        [](const auto &value) { return value && !value->retiring && value->compute_producer; });
    }
    queue_state *register_compute_producer(std::uint64_t native, std::uint32_t count,
        const native_observer::command_identity *values) {
      if (!compute_copies_recorded && !places_reclaimed) return nullptr;
      const auto awaits = [&](const auto &captures) {
        for (std::uint32_t n = 0; n != count; ++n) {
          const auto cookie = values[n].recording_cookie;
          if (!cookie) continue;
          for (const auto &value : captures)
            if (value.id && !value.producer_submitted && value.command == cookie) return true;
        }
        return false;
      };
      if (!awaits(capture_pool.slots) && !(capture_pool.diagnostic_slots_active && awaits(capture_pool.diagnostic_slots))) return nullptr;
      com_ptr<ID3D12CommandQueue> checked;
      if (!query_native(native, IID_ID3D12CommandQueue, checked)) return nullptr;
      auto *owner = register_queue(checked.p, true);
      if (!owner && reason == status::exhausted) compute_unplaced = true;
      return owner;
    }
    // Requires mutex. Depth is read on a queue other than owner: the newest
    // acquisition (reader) was another live queue's. Not whether owner ever
    // acquired: a render queue that presented with FG off reads elsewhere
    // once FG presents on DLSS-G's queue, and no longer once FG is off again.
    bool read_elsewhere(const queue_state &owner) {
      return reader && reader != owner.cookie && std::any_of(capture_queues.queues.begin(), capture_queues.queues.end(), [&](const auto &entry) {
        return entry && entry->cookie == reader && !entry->retiring;
      });
    }
    // U-C1b: FG options turned on within the source age, so Present may move
    // to DLSS-G's queue at FG's first Present, after the frame's copies on
    // the queue that reads depth now were submitted (until 10-27 those kept
    // the Reset rule, and that first FG Present was flat). Bounded: a game
    // whose Present stays on its queue with FG on pays one copy-out per copy
    // for that age after each FG On.
    bool reader_may_move(std::uint64_t now) {
      const auto moves = reader_moves.load(std::memory_order_relaxed);
      const auto open = [&](const std::atomic<std::uint64_t> &tick, const std::atomic<std::uint64_t> &at) {
        const auto start = tick.load(std::memory_order_relaxed);
        return start && at.load(std::memory_order_relaxed) == moves &&
          sunshine_freshness::fresh_lt(start, now, sunshine_freshness::source_age_ms);
      };
      return open(scope.frame_generation_on_tick, scope.frame_generation_on_moves) ||
        open(scope.frame_generation_run_tick, scope.frame_generation_run_moves);
    }
    // Requires mutex. The newest acquisition's queue (place cookie).
    void read_on(std::uint64_t cookie, bool pixels) {
      if (reader && reader != cookie) {
        reader_moves.fetch_add(1, std::memory_order_relaxed);
        if (!pixels) reader_move_unreadable.fetch_add(1, std::memory_order_relaxed);
      }
      reader = cookie;
    }
    // Requires mutex. Copy-out on the producer queue (docs/reshade-sbs.md):
    // in the submission hook, right after the game's ExecuteCommandLists and
    // before the producer fence value (fence) is signalled. A first
    // submission of a copy that another queue reads (read_elsewhere) stamps
    // its source's foreign_tick, so that its copies recorded within the
    // source age take copy-out storage (record_impl,
    // read_elsewhere_recently); each such copy is copied into that storage
    // with one list of the add-on's own on this queue, at most one per game
    // submission. Returns the slots copied out; the caller sets copied_out
    // once the fence after them was signalled. A copy not copied out keeps
    // the Reset rule (immutable_for_other_queues).
    bool reserved_by_other(const slot &entry, std::uint64_t own);
    // Requires mutex. U-C1b: a copy submitted on a queue other than the
    // reader without copy-out storage (recorded before its source was read
    // elsewhere, as when FG On moves Present to DLSS-G's queue) takes an idle
    // slot's cached storage of its snapshot's shape, format, view and device
    // (rest COMMON on a COMPUTE queue), so the game's submission thread never
    // allocates under the capture lock. The idle slot allocates again at its
    // next record, without the lock. False when no idle slot holds such
    // storage: the copy keeps the Reset rule, as before.
    bool take_idle_storage(slot &value, bool compute) {
      const auto &own = *value.texture;
      const auto shape = own.resource->GetDesc();
      const auto fits = [&](const std::shared_ptr<texture_reference> &storage) {
        if (!storage || storage.use_count() != 1 || !storage->resource.p || storage->device_identity != own.device_identity ||
            storage->view_format != own.view_format || (compute && storage->rest != D3D12_RESOURCE_STATE_COMMON))
          return false;
        const auto desc = storage->resource->GetDesc();
        return desc.Width == shape.Width && desc.Height == shape.Height && desc.Format == shape.Format;
      };
      for (auto &entry : capture_pool.slots) {
        if (entry.id || reserved_by_other(entry, 0)) continue;
        for (auto *storage : {&entry.published, &entry.texture})
          if (fits(*storage)) {
            value.published = std::move(*storage);
            return true;
          }
      }
      return false;
    }
    std::array<bool, slot_limit> copy_out(queue_state &owner, const std::array<bool, slot_limit> &producers,
        std::uint64_t fence) {
      std::array<bool, slot_limit> out{};
      if (!fence) return out;
      const auto now = GetTickCount64();
      // The reader's own queue copies out too while FG may move Present off
      // it (reader_may_move); only a copy read elsewhere stamps its source.
      // Cross-section read: producer copy-out consults current consumer reader,
      // not this place's persistent diagnostic chosen flag.
      const bool elsewhere = read_elsewhere(owner);
      if (!elsewhere && !(reader == owner.cookie && !owner.compute_producer && reader_may_move(now))) return out;
      bool any = false;
      for (unsigned i = 0; i != capture_pool.slots.size(); ++i) {
        auto &value = capture_pool.slots[i];
        if (!producers[i] || value.nomination_only || value.preservation_only || value.producer_submitted) continue;
        if (elsewhere && value.metadata.source) value.metadata.source->foreign_tick.store(now, std::memory_order_relaxed);
        if (value.invalid || !value.texture || value.copied_out) continue;
        if (!value.published) {
          const bool taken = take_idle_storage(value, owner.compute_producer);
          (taken ? capture_queues.copy_out_counters.taken : capture_queues.copy_out_counters.unstored).fetch_add(1, std::memory_order_relaxed);
          if (!taken) continue;
        }
        // A COMPUTE queue names only COMMON of these (record_impl's rest).
        if (owner.compute_producer && (value.texture->rest != D3D12_RESOURCE_STATE_COMMON ||
            value.published->rest != D3D12_RESOURCE_STATE_COMMON)) continue;
        out[i] = any = true;
      }
      if (!any) return out;
      const auto refuse = [&](std::atomic<std::uint64_t> &counter) {
        counter.fetch_add(1, std::memory_order_relaxed);
        out.fill(false);
        return out;
      };
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      if (refuse_copy_outs.load(std::memory_order_acquire)) return refuse(capture_queues.copy_out_counters.ring_busy);
#endif
      // A list whose last execution completed (its fence value passed).
      const auto completed = owner.fence->GetCompletedValue();
      copy_out_list *entry = nullptr;
      for (auto &candidate : owner.copy_out)
        if (!candidate.list.p || (completed != UINT64_MAX && completed >= candidate.fence)) { entry = &candidate; break; }
      if (!entry) return refuse(capture_queues.copy_out_counters.ring_busy);
      const auto type = owner.compute_producer ? D3D12_COMMAND_LIST_TYPE_COMPUTE : D3D12_COMMAND_LIST_TYPE_DIRECT;
      const bool opened = entry->list.p ?
        SUCCEEDED(entry->allocator->Reset()) && SUCCEEDED(entry->list->Reset(entry->allocator.p, nullptr)) :
        SUCCEEDED(owner.device->CreateCommandAllocator(type, __uuidof(ID3D12CommandAllocator),
          reinterpret_cast<void **>(entry->allocator.put()))) &&
        SUCCEEDED(owner.device->CreateCommandList(0, type, entry->allocator.p, nullptr, __uuidof(ID3D12GraphicsCommandList),
          reinterpret_cast<void **>(entry->list.put())));
      auto *list = entry->list.p;
      if (opened) for (unsigned i = 0; i != capture_pool.slots.size(); ++i) {
        if (!out[i]) continue;
        const auto &value = capture_pool.slots[i];
        D3D12_RESOURCE_BARRIER transitions[2]{};
        transitions[0].Type = transitions[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transitions[0].Transition = {value.texture->resource.p, 0, value.texture->rest, D3D12_RESOURCE_STATE_COPY_SOURCE};
        transitions[1].Transition = {value.published->resource.p, 0, value.published->rest, D3D12_RESOURCE_STATE_COPY_DEST};
        list->ResourceBarrier(2, transitions);
        list->CopyResource(value.published->resource.p, value.texture->resource.p);
        for (auto &transition : transitions) std::swap(transition.Transition.StateBefore, transition.Transition.StateAfter);
        list->ResourceBarrier(2, transitions);
      }
      if (!opened || FAILED(list->Close())) {
        entry->list.reset(); entry->allocator.reset(); // Not executed since its last fence completed.
        return refuse(capture_queues.copy_out_counters.failed);
      }
      ID3D12CommandList *lists[]{list};
      owner.queue->ExecuteCommandLists(1, lists); // The hooks are suppressed in this callback.
      entry->fence = fence;
      return out;
    }
    void submitted(std::uint64_t native, std::uint32_t count, const native_observer::command_identity *values) {
      const auto game_at = game_submission_position();
      std::lock_guard lock(mutex);
      apply_executed(native, count, values);
      auto *owner = queue(native);
      if (!owner) owner = register_compute_producer(native, count, values);
      if (!owner) return;
      native = reinterpret_cast<std::uint64_t>(owner->queue.p);
      std::array<bool, slot_limit> producers{}, consumers{};
      const bool depth_used = identify_submissions(capture_pool.slots, count, values, producers, consumers);
      std::array<bool, diagnostic_slot_limit> auxiliary_producers, auxiliary_consumers;
      bool auxiliary_used = false;
      if (capture_pool.diagnostic_slots_active) {
        auxiliary_producers.fill(false); auxiliary_consumers.fill(false);
        auxiliary_used = identify_submissions(capture_pool.diagnostic_slots, count, values, auxiliary_producers, auxiliary_consumers);
      }
      if (!depth_used && !auxiliary_used) return;
      // The queue of each list that carried a depth copy, which its next
      // copy's copy-out storage is predicted from (U-C1b, record_impl): one
      // private-data read per such list.
      if (depth_used) for (std::uint32_t n = 0; n != count; ++n) {
        const auto cookie = values[n].recording_cookie;
        bool carried = false;
        for (unsigned i = 0; i != capture_pool.slots.size() && !carried; ++i) carried = producers[i] && cookie && capture_pool.slots[i].command == cookie;
        if (!carried) continue;
        if (auto list = command_storage::acquire(reinterpret_cast<ID3D12Object *>(values[n].native_command), false))
          list->executed_queue = owner->cookie;
        if (!owner->compute_producer) last_direct_copy_queue = owner->cookie;
      }
      const bool needs_fence = auxiliary_used || needs_submission_fence(producers, consumers);
      const auto fence = needs_fence ? ++owner->next_fence : 0;
      const auto copied = depth_used ? copy_out(*owner, producers, fence) : std::array<bool, slot_limit>{};
      const auto copy_at = copy_out_position(copied);
      // This callback runs AFTER the actual native ExecuteCommandLists. Signal
      // therefore follows the copied/consumed pixels (and copy-outs) on this queue.
      const bool signaled = !needs_fence || SUCCEEDED(owner->queue->Signal(owner->fence.p, fence));
      if (!signaled)
        for (auto &entry : owner->copy_out) if (entry.fence == fence) entry.fence = UINT64_MAX; // Never reused.
      for (unsigned i = 0; i != capture_pool.slots.size(); ++i) {
        auto &value = capture_pool.slots[i];
        if (producers[i]) {
          producer_submitted(value, native, value.nomination_only ? 0 : fence, value.nomination_only || signaled);
          set_producer_position(value, copied[i], signaled, copy_at, game_at);
          if (copied[i] && signaled) {
            value.copied_out = true;
            capture_queues.copy_out_counters.made.fetch_add(1, std::memory_order_relaxed);
          }
          // An executed copy-out without its fence is never released unused.
          if (copied[i]) value.predicted_storage = false;
          accept_compute_copy(value);
        }
        if (consumers[i]) {
          consumer_submitted(value, native, fence, signaled);
          // Keep the recording reference until Reset/destroy. A legal replay of
          // an unchanged command list can consume this texture again.
          submitted_immediate(value, count, values);
        }
      }
      if (auxiliary_used) for (unsigned i = 0; i != capture_pool.diagnostic_slots.size(); ++i) {
        if (auxiliary_producers[i]) producer_submitted(capture_pool.diagnostic_slots[i], native, fence, signaled);
        if (auxiliary_consumers[i]) {
          consumer_submitted(capture_pool.diagnostic_slots[i], native, fence, signaled);
          submitted_immediate(capture_pool.diagnostic_slots[i], count, values);
        }
      }
    }
    // Requires mutex. The retired slots' storage moves to discard.
    void collect_retired(std::vector<std::shared_ptr<texture_reference>> &discard) {
      for (auto &owner : capture_queues.queues) {
        if (!owner || !owner->retiring) continue;
        const auto native = reinterpret_cast<std::uint64_t>(owner->queue.p);
        const bool removed = FAILED(owner->device->GetDeviceRemovedReason());
        bool retained = false;
        for_each_capture([&](auto &value) {
          if (value.id && uses_queue(value, native)) {
            if (removed || reclaimable(value)) {
              retire_storage(discard, value.texture);
              retire_storage(discard, value.published);
              value = {};
            } else retained = true;
          }
        });
        if (!retained) {
          owner.reset();
          compute_unplaced = false;
        }
      }
      collect_diagnostics(discard);
    }
  }
  namespace {
    void raise_execution(queue_execution &entry, std::uint64_t newest) {
      std::atomic_ref<std::uint64_t> value(entry.newest);
      auto seen = value.load(std::memory_order_relaxed);
      while (seen < newest && !value.compare_exchange_weak(seen, newest, std::memory_order_relaxed)) {}
    }
    queue_execution *find_queue_execution(std::uint64_t queue) {
      for (std::size_t i = 0; i != capture_queues.queue_execution_count; ++i)
        if (capture_queues.queue_executions[i].queue == queue) return &capture_queues.queue_executions[i];
      return nullptr;
    }
    void note_queue_execution(std::uint64_t queue, std::uint64_t newest) {
      if (!queue) return;
      AcquireSRWLockShared(&capture_queues.queue_execution_lock);
      auto *known = find_queue_execution(queue);
      if (known) raise_execution(*known, newest);
      ReleaseSRWLockShared(&capture_queues.queue_execution_lock);
      if (known) return;
      AcquireSRWLockExclusive(&capture_queues.queue_execution_lock);
      auto *entry = find_queue_execution(queue);
      if (!entry) {
        if (capture_queues.queue_execution_count != capture_queues.queue_executions.size()) entry = &capture_queues.queue_executions[capture_queues.queue_execution_count++];
        else {
          entry = &capture_queues.queue_executions.front();
          for (auto &value : capture_queues.queue_executions) if (value.newest < entry->newest) entry = &value;
        }
        *entry = {queue, 0};
      }
      entry->newest = std::max(entry->newest, newest);
      ReleaseSRWLockExclusive(&capture_queues.queue_execution_lock);
    }
    void forget_queue_execution(std::uint64_t queue) {
      if (!queue) return;
      AcquireSRWLockExclusive(&capture_queues.queue_execution_lock);
      if (auto *entry = find_queue_execution(queue)) {
        *entry = capture_queues.queue_executions[--capture_queues.queue_execution_count];
        capture_queues.queue_executions[capture_queues.queue_execution_count] = {};
      }
      ReleaseSRWLockExclusive(&capture_queues.queue_execution_lock);
    }
    void observed_submission(std::uint64_t queue, std::uint32_t count, const native_observer::command_identity *values) {
      submitted(queue, count, values);
      // ReShade resets its own list right after submitting it: the submitted
      // recording ends and the next one begins. Any other list with a
      // recording is the game's (game_executions).
      bool game = false;
      for (std::uint32_t i = 0; i != count; ++i)
        if (runtime_list(values[i].native_command))
          try { lifecycle_recording(values[i].native_command, list_event::reset); } catch (...) {}
        else if (values[i].recording_cookie) game = true;
      if (game) note_queue_execution(queue, capture_queues.game_execution_count.fetch_add(1, std::memory_order_acq_rel) + 1);
    }
  }
  void observe_queue(std::uint64_t native) {
    if (!requested.load() || !native) return;
    com_ptr<ID3D12CommandQueue> checked;
    if (!query_native(native, IID_ID3D12CommandQueue, checked)) { reason = status::unsupported_queue; return; }
    native = reinterpret_cast<std::uint64_t>(checked.p);
    native_observer::observe_queue(native);
    pooled_storage discard;
    std::lock_guard lock(mutex);
    collect_retired(discard);
    if (queue(native)) return;
    register_queue(reinterpret_cast<ID3D12CommandQueue *>(native), false);
  }
  void retire_queue(std::uint64_t native) {
    pooled_storage discard;
    std::lock_guard lock(mutex);
    if (auto *owner = queue(native)) {
      owner->retiring = true;
      native = reinterpret_cast<std::uint64_t>(owner->queue.p);
      for_each_capture([&](auto &value) {
        if (!uses_queue(value, native)) return;
        invalidate(value, capture_failure::queue_retired);
        // The runtime's immediate list is destroyed with its queue; an
        // unsubmitted read can no longer execute.
        if (value.consumer_queue == native) value.pending_immediate = 0;
      });
    }
    collect_retired(discard);
    // Its newest game execution goes with it, freeing that entry.
    forget_queue_execution(native);
  }
}
