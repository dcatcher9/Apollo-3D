// SPDX-License-Identifier: GPL-3.0-only
// Capture storage and source lifetime; included by streamline_depth_capture.cpp
// after the shell. Namespace-scope types and state retain their original positions.

namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    bool current_source_nomination(const slot &value) {
      if (!value.id || !value.source_nominated || value.preservation_only || value.invalid ||
          value.nomination_invalid || (value.finished && !value.success) || !valid_provider(value.metadata.provider))
        return false;
      const auto &head = evaluations[provider_index(value.metadata.provider)];
      return value.metadata.epoch == head.epoch && value.metadata.sequence == head.sequence &&
        value.metadata.source_id == head.source_id;
    }

    source_retention logical_source_retention(slot &value) {
      if (value.nomination_only) {
        // Metadata-only nominations carry source authority but no GPU storage.
        const auto &head = evaluations[provider_index(value.metadata.provider)];
        const bool current = !value.invalid && (!value.finished || value.success) &&
          value.metadata.epoch == head.epoch && value.metadata.sequence == head.sequence &&
          value.metadata.source_id == head.source_id;
        return current ? source_retention::current_nomination : source_retention::none;
      }
      if (!value.source_nominated || value.preservation_only || value.invalid || value.nomination_invalid ||
          !valid_provider(value.metadata.provider)) return source_retention::none;
      // GPU retirement does not retire the provider's nomination authority.
      // A different provider (or Generic preservation) shares this pool and
      // must not erase the current head while its namespace still names it.
      if (current_source_nomination(value)) return source_retention::current_nomination;
      // The provider's newest nomination: its capture, valid or not (a
      // failed or refused one is a missing frame, select_capture), else the
      // view that names it when it was refused before its capture existed.
      const auto provider = value.metadata.provider;
      const auto &head = evaluations[provider_index(provider)];
      const slot *latest = nullptr;
      for (const auto &candidate : slots)
        if (candidate.id && !candidate.preservation_only && candidate.metadata.provider == provider &&
            candidate.metadata.epoch == head.epoch && candidate.metadata.sequence == head.sequence &&
            candidate.metadata.source_id == head.source_id && (!latest || candidate.id > latest->id)) latest = &candidate;
      const evaluation_head *view = nullptr;
      if (!latest && head.epoch && head.sequence)
        for (const auto &candidate : head.views)
          if (candidate.epoch == head.epoch && candidate.sequence == head.sequence && candidate.source_id == head.source_id)
            view = &candidate;
      if (!latest && !view) return source_retention::none;
      // Preserve exactly the existing eligible completed fallback too, beneath
      // a pending successor, a nomination in progress or a missing frame
      // alike, and beneath an FG input the SR/RR snapshot of its lineage that
      // stands in for it (lineage_snapshot). Their age, reset/revision,
      // source and GPU-completion requirements remain in completed_snapshot
      // and lineage_fallback; retaining storage grants no extra admission.
      // Until 10-28 that SR snapshot was kept by neither rule: on another
      // queue it becomes readable and reclaimable at once, so the Present's
      // idle sweep (poll) or another record could recycle it before the next
      // Present, which then held an older copy or went mono.
      const auto identity = latest ? latest->metadata : head_identity(*view, provider);
      const auto now = GetTickCount64();
      return completed_snapshot(identity, now) == &value || lineage_fallback(identity, now) == &value ?
        source_retention::completed_fallback : source_retention::none;
    }

    bool snapshot_storage_retired(const slot &value) {
      if (!value.id) return true;
      if (value.diagnostic_only && !value.diagnostic_released) return false;
      // No copy allocation or submitted GPU obligation belongs to metadata.
      if (value.nomination_only) return true;
      if (value.retirement_unknown) return false;
      if (value.command) return false; // A closed producer may legally replay.
      if (value.texture.use_count() > 1 || value.published.use_count() > 1) return false;
      if (std::any_of(value.consumers.begin(), value.consumers.end(), [](auto v) { return v != 0; })) return false;
      if (value.pending_immediate) return false;
      if (!value.producer_submitted) return value.finished && value.command == 0;
      return gpu_retired(value, [](const fence_point &point) {
        const auto *owner = known_queue(point.queue);
        if (!owner || !owner->fence.p) return false;
        const auto complete = owner->fence->GetCompletedValue();
        return complete != UINT64_MAX && complete >= point.value;
      });
    }

    bool reclaimable(slot &value) {
      retire_dead_recordings(value);
      if (!value.id) return true;
      // Logical source lifetime and GPU storage lifetime are independent. A
      // retired allocation can still carry authority; withdrawing authority
      // never discharges an unfinished recording, CPU lease or queue fence.
      return logical_source_retention(value) == source_retention::none && snapshot_storage_retired(value);
    }

    // Leaves the count last: a waiter may then release the device.
    void release_storage_job(storage_job &job, bool pooled) {
#include "report_storage_release.inl"
      {
        std::lock_guard lock(storage_jobs.mutex);
        --storage_jobs.outstanding;
      }
      storage_jobs.idle.notify_all();
    }

    void CALLBACK storage_release_task(PTP_CALLBACK_INSTANCE, void *context) {
      const std::unique_ptr<storage_job> job(static_cast<storage_job *>(context));
      release_storage_job(*job, true);
    }

    // Never under the capture lock. Without a pool thread (or memory for the
    // job) the storage is released here, as before 10-27, and the step is
    // named.
    void release_storage(storage_list &values) noexcept {
      // Only storage dropped last goes to the pool: a reference another holder
      // keeps (a slot, a packet) is only a decrement, dropped here, so a
      // pooled job never holds a slot's storage as a lease would (a slot
      // whose storage someone else holds is not reclaimable).
      for (auto &value : values) if (value.use_count() > 1) value.reset();
      values.erase(std::remove(values.begin(), values.end(), nullptr), values.end());
      if (values.empty()) return;
      std::unique_ptr<storage_job> job(new (std::nothrow) storage_job);
      if (job) {
        job->values.swap(values);
        job->caller = GetCurrentThreadId();
        {
          std::lock_guard lock(storage_jobs.mutex);
          ++storage_jobs.outstanding;
        }
        if (TrySubmitThreadpoolCallback(storage_release_task, job.get(), nullptr)) {
          job.release();
          return;
        }
        const sunshine_game3d::slow_step step("depth capture storage release");
        release_storage_job(*job, false);
        return;
      }
      const sunshine_game3d::slow_step step("depth capture storage release");
      values.clear();
    }

    // Requires mutex. Moves value's storage to discard (released after the
    // lock); without memory for it, here.
    void retire_storage(storage_list &discard, std::shared_ptr<texture_reference> &value) noexcept {
      if (!value) return;
      try {
        discard.push_back(std::move(value));
      } catch (...) {
        value.reset();
      }
    }

    // Teardown (release_device, shutdown). Once ExitProcess has begun, every
    // other thread (the pool's too) has ended: a job that has not run never
    // will, and none runs.
    void wait_storage_releases() {
      using shutdown_query = BOOLEAN(NTAPI *)();
      static const auto shutting_down = reinterpret_cast<shutdown_query>(
        reinterpret_cast<void *>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlDllShutdownInProgress")));
      if (sunshine_addon_lifetime::stopping() || (shutting_down && shutting_down())) return;
      std::unique_lock lock(storage_jobs.mutex);
      storage_jobs.idle.wait(lock, [] { return !storage_jobs.outstanding; });
    }

    void collect_diagnostics(storage_list &discard) {
      if (!diagnostic_slots_active) return;
      bool retained = false;
      const auto now = GetTickCount64();
      for (auto &value : diagnostic_slots) {
        if (value.id && reclaimable(value)) {
          // Retire the capture identity/leases, not its reusable allocation.
          // Live FG alpha uses this owner every real frame; destroying storage
          // here would turn the bounded ring into a per-frame GPU allocator.
          auto storage = value.diagnostic_reusable ? std::move(value.texture) : nullptr;
          retire_storage(discard, value.texture);
          value = {};
          value.texture = std::move(storage);
          value.idle_tick = now;
        }
        retained |= value.id != 0;
      }
      diagnostic_slots_active = retained;
    }

    // Requires the lock. A live reservation of another record attempt.
    bool reserved_by_other(const slot &entry, std::uint64_t own) {
      return entry.reservation && entry.reservation != own &&
        GetTickCount64() - entry.reservation_tick < reservation_timeout_ms;
    }

    // Requires the lock.
    void clear_reservation(std::uint64_t token) {
      if (!token) return;
      const auto clear = [token](slot &entry) {
        if (entry.reservation == token) { entry.reservation = entry.reservation_tick = entry.reserved_bytes = 0; entry.reserved_live = false; }
      };
      for (auto &entry : slots) clear(entry);
      for (auto &entry : diagnostic_slots) clear(entry);
    }

    // Idle cached storage is always live: dump copies are never cached.
    bool live_storage(const slot &entry) { return entry.id ? entry.diagnostic_reusable : bool(entry.texture); }

    // own: the caller's reservation, whose reserved bytes are its own. Evicted
    // idle storage moves to discard (released by the caller after the lock).
    bool reserve_diagnostic_bytes(std::array<slot, diagnostic_slot_limit> &pool,
        const slot *replacement, std::uint64_t bytes, bool live, std::uint64_t own,
        std::vector<std::shared_ptr<texture_reference>> *discard) {
      const auto limit = live ? std::max(diagnostic_byte_limit, live_snapshot_reserve * bytes) : diagnostic_byte_limit;
      if (!bytes || bytes > limit) return false;
      std::uint64_t allocated = 0;
      for (const auto &entry : pool) {
        if (&entry != replacement && entry.texture && live_storage(entry) == live) allocated += entry.texture->allocation_bytes;
        // Storage another attempt is allocating without the lock right now.
        if (reserved_by_other(entry, own) && entry.reserved_live == live) allocated += entry.reserved_bytes;
      }
      // A format/size change may need different storage. Evict only idle cache
      // entries; live tickets and any unfinished GPU obligations retain theirs.
      for (auto &entry : pool) {
        if (allocated <= limit - bytes) return true;
        if (live && &entry != replacement && !entry.id && entry.texture) {
          allocated -= entry.texture->allocation_bytes;
          if (discard) discard->push_back(std::move(entry.texture));
          entry.texture.reset();
        }
      }
      return allocated <= limit - bytes;
    }

    // Never called with the capture lock: the driver may take milliseconds.
    // The device is the recording's retained source's, alive while the game
    // records with it; release_device releases this storage before it goes.
    record_stage allocate_storage(const storage_plan &plan, std::shared_ptr<texture_reference> &out) {
      out.reset();
      try {
        auto texture = std::make_shared<texture_reference>();
        texture->device = plan.device;
        texture->device_identity = plan.device_identity;
        texture->allocation_bytes = plan.allocation_bytes;
        texture->rest = plan.state;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        native_observer::suppression_scope suppress;
        if (FAILED(plan.device->CreateCommittedResource(&heap, plan.shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE,
            &plan.desc, plan.state, nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void **>(texture->resource.put()))))
          return record_stage::texture_allocation;
        texture->identity = retain_source_cookie(texture->resource.p);
        if (!texture->identity) return record_stage::texture_allocation;
        if (plan.shared &&
            FAILED(plan.device->CreateSharedHandle(texture->resource.p, nullptr, GENERIC_ALL, nullptr, &texture->shared_handle)))
          return record_stage::texture_allocation;
        if (plan.descriptors) {
          D3D12_DESCRIPTOR_HEAP_DESC descriptors{};
          descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; descriptors.NumDescriptors = 1;
          if (FAILED(plan.device->CreateDescriptorHeap(&descriptors, __uuidof(ID3D12DescriptorHeap),
              reinterpret_cast<void **>(texture->views.put())))) return record_stage::descriptor_allocation;
          if (plan.view != DXGI_FORMAT_UNKNOWN) {
            const auto handle = texture->views->GetCPUDescriptorHandleForHeapStart();
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = plan.view; view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels = 1;
            plan.device->CreateShaderResourceView(texture->resource.p, &view, handle);
            texture->shader_resource = handle.ptr;
            texture->view_format = plan.view;
          }
        }
        out = std::move(texture);
        return record_stage::recorded;
      } catch (...) {
        return record_stage::texture_allocation;
      }
    }

    // Requires mutex. New storage joins the registry (storage).
    bool register_storage(const std::shared_ptr<texture_reference> &value) {
      try {
        if (storage.size() == storage.capacity())
          storage.erase(std::remove_if(storage.begin(), storage.end(), [](const auto &entry) { return entry.expired(); }),
            storage.end());
        storage.push_back(value);
        return true;
      } catch (...) {
        return false;
      }
    }
  }

  void shutdown() {
    requested = false;
    // No owner keeps a consumed frame or a shown copy without its source
    // (only retire_source's FG role does, select_latest_capture).
    { std::lock_guard lock(mutex);
      for (auto &owner : queues) if (owner) {
        owner->provider_established = false;
        owner->last_epoch = owner->last_sequence = owner->present_capture = 0;
      } }
    native_observer::shutdown();
    invalidate_all();
    // No storage release job outlives the add-on's capture owner.
    wait_storage_releases();
    // Pinned hooks and bounded in-flight allocations remain alive: unloading or
    // releasing uncompleted GPU work would be unsafe. No shutdown wait is added.
    reason = status::inactive;
  }

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    void set_release_wait(std::uint64_t ms) { release_wait_ms = ms ? ms : release_wait_bound_ms; }
  }
#endif

  bool release_device(std::uint64_t native, char *out, std::size_t size) {
    if (out && size) *out = 0;
    com_ptr<ID3D12Device> checked;
    if (!native || !query_native(native, IID_ID3D12Device, checked)) return false;
    const auto device = device_cookie(checked.p);
    if (!device) return false;
    // Every pooled storage release finished first: none releases a resource
    // of this device after the device is gone.
    wait_storage_releases();
    // Its queue places take no further work, and each signals its fence after
    // everything submitted to it. Every copy into or read of a snapshot that
    // was submitted was submitted on a placed queue (an unplaced queue's copy
    // is never submitted, so never read; the game drained its own queues).
    std::array<std::pair<ID3D12Fence *, std::uint64_t>, queue_limit> fences{};
    unsigned waits = 0;
    bool finished = true;
    {
      std::lock_guard lock(mutex);
      // The add-on's own queue calls under the capture mutex are never game
      // calls to the order hooks (design finding 8).
      native_observer::suppression_scope suppress;
      for (auto &owner : queues) if (owner && owner->device_identity == device) {
        owner->retiring = true;
        owner->provider_established = false;
        const auto value = ++owner->next_fence;
        if (SUCCEEDED(owner->queue->Signal(owner->fence.p, value))) {
          owner->fence->AddRef();
          fences[waits++] = {owner->fence.p, value};
        } else if (SUCCEEDED(owner->device->GetDeviceRemovedReason())) finished = false; // A removed device runs nothing.
      }
    }
    forget_queue_order(checked.p);
    const auto started = GetTickCount64();
    const std::uint64_t bound = release_wait_ms;
    const HANDLE event = waits ? CreateEventW(nullptr, FALSE, FALSE, nullptr) : nullptr;
    for (unsigned i = 0; i != waits; ++i) {
      const auto [fence, value] = fences[i];
      // A removed device's fences read UINT64_MAX.
      if (finished && fence->GetCompletedValue() < value) {
        const auto elapsed = GetTickCount64() - started;
        finished = event && elapsed < bound && SUCCEEDED(fence->SetEventOnCompletion(value, event)) &&
          WaitForSingleObject(event, static_cast<DWORD>(bound - elapsed)) == WAIT_OBJECT_0;
      }
      fence->Release();
    }
    if (event) CloseHandle(event);
    const auto waited = GetTickCount64() - started;
    // Every object of the device leaves the capture owner: its snapshots and
    // queue places, and the resources of its sources and storage wherever
    // they are still held (an adapter's tag, a packet, a ticket), which are
    // released after the lock. Unfinished, they are kept for good instead.
    std::vector<std::shared_ptr<texture_reference>> storage_held;
    std::vector<source_ref> sources_held;
    std::vector<std::unique_ptr<queue_state>> places_held;
    std::vector<IUnknown *> released;
    unsigned snapshots = 0, places = 0, released_sources = 0, released_storage = 0;
    bool reserved = false;
    {
      std::lock_guard lock(mutex);
      // Everything below moves without allocating.
      try {
        constexpr std::size_t slot_count = slot_limit + diagnostic_slot_limit;
        storage_held.reserve(slot_count + storage.size());
        sources_held.reserve(2 * slot_count + source_limit);
        places_held.reserve(queue_limit);
        released.reserve(source_limit + 2 * storage.size());
        if (!finished) {
          if (!kept) kept = new kept_objects;
          kept->storage.reserve(kept->storage.size() + storage_held.capacity());
          kept->sources.reserve(kept->sources.size() + sources_held.capacity());
          kept->places.reserve(kept->places.size() + places_held.capacity());
        }
        reserved = true;
      } catch (...) {
        finished = false; // Out of memory: nothing of the device moved.
      }
      if (reserved) {
        std::array<std::uint64_t, queue_limit> natives{};
        for (auto &owner : queues) if (owner && owner->device_identity == device) {
          const auto queue = reinterpret_cast<std::uint64_t>(owner->queue.p);
          natives[places++] = queue;
          forget_queue_execution(queue);
          places_held.push_back(std::move(owner));
        }
        const auto of_device = [&](const slot &value) {
          return (value.texture && value.texture->device_identity == device) ||
            (value.published && value.published->device_identity == device) ||
            (value.metadata.source && value.metadata.source->device_identity == device) ||
            (value.auxiliary_destination && value.auxiliary_destination->device_identity == device) ||
            std::any_of(natives.begin(), natives.begin() + places, [&](auto queue) { return uses_queue(value, queue); });
        };
        const auto release_slot = [&](slot &value) {
          if (!of_device(value)) return;
          snapshots += value.id != 0;
          if (value.texture) storage_held.push_back(std::move(value.texture));
          if (value.published) storage_held.push_back(std::move(value.published));
          if (value.metadata.source) sources_held.push_back(std::move(value.metadata.source));
          if (value.auxiliary_destination) sources_held.push_back(std::move(value.auxiliary_destination));
          slot empty;
          // A record attempt may be allocating for this slot without the lock.
          empty.reservation = value.reservation; empty.reservation_tick = value.reservation_tick;
          empty.reserved_bytes = value.reserved_bytes; empty.reserved_live = value.reserved_live;
          value = std::move(empty);
        };
        for (auto &value : slots) release_slot(value);
        for (auto &value : diagnostic_slots) release_slot(value);
        for (auto &weak : sources) if (const auto value = weak.lock()) if (value->device_identity == device) {
          weak.reset();
          ++released_sources;
          if (!finished) { sources_held.push_back(value); continue; }
          value->released.store(true, std::memory_order_release);
          value->device = nullptr;
          released.push_back(std::exchange(value->resource.p, nullptr));
        }
        for (auto &weak : storage) if (const auto value = weak.lock()) if (value->device_identity == device) {
          weak.reset();
          ++released_storage;
          if (!finished) { storage_held.push_back(value); continue; }
          value->released.store(true, std::memory_order_release);
          value->device = nullptr;
          value->shader_resource = 0;
          released.push_back(std::exchange(value->resource.p, nullptr));
          released.push_back(std::exchange(value->views.p, nullptr));
        }
        if (!finished) {
          for (auto &value : storage_held) kept->storage.push_back(std::move(value));
          for (auto &value : sources_held) kept->sources.push_back(std::move(value));
          for (auto &value : places_held) kept->places.push_back(std::move(value));
        }
      }
    }
    // A device recreated over the same native object starts with a new identity.
    checked->SetPrivateData(sunshine_native_identity::device_guid, 0, nullptr);
    for (auto *value : released) if (value) value->Release();
    const auto u = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    if (out && size) {
      if (!reserved)
        std::snprintf(out, size, "Sunshine capture: device %llu: out of memory at its release; its objects stay", u(device));
      else if (finished)
        std::snprintf(out, size, "Sunshine capture: released device %llu: queue_places=%u snapshots=%u sources=%u "
          "storage=%u gpu_wait_ms=%llu", u(device), places, snapshots, released_sources, released_storage, u(waited));
      else
        std::snprintf(out, size, "Sunshine capture: kept device %llu: its queues had not finished within %llu ms, so its "
          "queue_places=%u snapshots=%u sources=%u storage=%u are kept for good", u(device), u(waited), places, snapshots,
          released_sources, released_storage);
    }
    return finished;
  }

  namespace {
    // Requires mutex. A reclaimable slot keeps only its allocation: the game
    // resource, its device and recording references go now rather than when
    // the slot is next reused (after a hitch many slots can be idle). Released
    // storage moves to discard, for release after the lock.
    void release_idle_slots(std::vector<std::shared_ptr<texture_reference>> &discard) {
      // An API namespace's newest nomination stays: selection reports its
      // failure or pending state.
      const auto head = [](const slot &value) {
        const auto &metadata = value.metadata;
        if (value.preservation_only || !metadata.epoch || !valid_provider(metadata.provider)) return false;
        const auto &current = evaluations[provider_index(metadata.provider)];
        return metadata.epoch == current.epoch && metadata.sequence == current.sequence && metadata.source_id == current.source_id;
      };
      std::uint64_t idle = 0;
      for (auto &value : slots) {
        // Predicted storage (U-C1b) whose submission made no copy-out: the
        // reader's queue executed it after all, or the copy keeps the Reset
        // rule. No list wrote it.
        if (value.id && value.predicted_storage && value.producer_submitted && !value.copied_out &&
            value.published.use_count() == 1) {
          discard.push_back(std::move(value.published));
          value.predicted_storage = false;
          copy_out_counters.released.fetch_add(1, std::memory_order_relaxed);
        }
        if (value.id && !head(value) && reclaimable(value)) {
          slot idle_slot;
          idle_slot.texture = std::move(value.texture);
          idle_slot.published = std::move(value.published);
          // A record attempt may be allocating for this slot without the lock.
          idle_slot.reservation = value.reservation; idle_slot.reservation_tick = value.reservation_tick;
          idle_slot.reserved_bytes = value.reserved_bytes; idle_slot.reserved_live = value.reserved_live;
          value = std::move(idle_slot);
        }
        if (value.id) continue;
        // Cached copy-out storage counts as the snapshot's (record_impl).
        for (auto *storage : {&value.texture, &value.published}) {
          if (!*storage) continue;
          if (idle + (*storage)->allocation_bytes > idle_pixel_byte_limit) discard.push_back(std::move(*storage));
          else idle += (*storage)->allocation_bytes;
        }
      }
    }

    // Requires mutex. Its live budget is sized from the request, so idle storage
    // ages out instead of waiting for budget pressure. Released storage moves to
    // discard, for release after the lock.
    void release_idle_diagnostics(std::vector<std::shared_ptr<texture_reference>> &discard, std::uint64_t now) {
      for (auto &value : diagnostic_slots)
        if (!value.id && value.texture && !reserved_by_other(value, 0) && now - value.idle_tick >= idle_live_storage_ms)
          discard.push_back(std::move(value.texture));
    }
  }

  // On the present thread ("depth input observation"); each part names its
  // own hitch. The storage it drops goes to a pool thread (release_storage).
  void poll() {
    if (requested.load()) {
      const sunshine_game3d::slow_step step("depth capture hook installation");
      native_observer::install_pending();
    }
    pooled_storage discard;
    std::unique_lock lock(mutex, std::defer_lock);
    {
      const sunshine_game3d::slow_step step("depth capture poll lock");
      lock.lock();
    }
    const sunshine_game3d::slow_step step("depth capture idle sweep");
    collect_retired(discard);
    try {
      release_idle_slots(discard);
      release_idle_diagnostics(discard, GetTickCount64());
    } catch (...) {}
    // discard is released after the lock (declared before it).
  }

  source_ref retain_source(std::uint64_t native, record_diagnostic *diagnostic) {
    if (diagnostic) { *diagnostic = {}; diagnostic->resource = native; }
    const auto result = [&](source_ref value, status outcome, record_stage stage) {
      if (diagnostic) {
        diagnostic->result = outcome; diagnostic->stage = stage;
        if (value) {
          diagnostic->expected_device_identity = value->device_identity;
          diagnostic->width = static_cast<std::uint32_t>(value->desc.Width);
          diagnostic->height = value->desc.Height; diagnostic->format = value->desc.Format;
          diagnostic->flags = value->desc.Flags;
          diagnostic->dimension = value->desc.Dimension; diagnostic->mip_levels = value->desc.MipLevels;
          diagnostic->array_size = value->desc.DepthOrArraySize; diagnostic->samples = value->desc.SampleDesc.Count;
        }
      }
      return value;
    };
    if (!active()) return result({}, status::inactive, record_stage::inactive);
    if (!native) return result({}, status::malformed, record_stage::malformed_input);
    com_ptr<ID3D12Resource> checked;
    if (!query_native(native, IID_ID3D12Resource, checked)) return result({}, status::unsupported_resource, record_stage::source_interface);
    auto *resource = checked.p;
    const auto retained = [](std::uint64_t cookie) -> source_ref {
      if (cookie) for (const auto &weak : sources) if (auto value = weak.lock()) if (value->cookie == cookie) return value;
      return {};
    };
    {
      std::lock_guard lock(mutex);
      if (auto value = retained(source_cookie(resource))) return result(value, status::ready, record_stage::retained);
    }
    // A new source's device is read without the lock: under ReShade the
    // GetDevice hook and the proxy's Release take ReShade's device lock, which
    // it holds while it destroys a device (release_device takes this lock).
    auto value = std::make_shared<source_reference>();
    if (FAILED(resource->QueryInterface(__uuidof(ID3D12Resource), reinterpret_cast<void **>(value->resource.put()))))
      return result({}, status::unsupported_resource, record_stage::source_interface);
    ID3D12Device *device{};
    if (FAILED(resource->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(&device))) || !device)
      return result({}, status::unsupported_queue, record_stage::source_device);
    // Not owned (source_reference): the game keeps its device while it
    // presents, and release_device drops this source before it is destroyed.
    value->device = device;
    value->device_identity = device_cookie(device, true);
    device->Release();
    value->desc = resource->GetDesc();
    if (diagnostic) {
      diagnostic->width = static_cast<std::uint32_t>(value->desc.Width);
      diagnostic->height = value->desc.Height; diagnostic->format = value->desc.Format;
      diagnostic->flags = value->desc.Flags;
      diagnostic->dimension = value->desc.Dimension; diagnostic->mip_levels = value->desc.MipLevels;
      diagnostic->array_size = value->desc.DepthOrArraySize; diagnostic->samples = value->desc.SampleDesc.Count;
    }
    if (!value->device_identity) { reason = status::unsupported_queue; return result({}, status::unsupported_queue, record_stage::device_identity); }
    // Retention proves object identity/lifetime, not capture capability. The
    // common opportunity recorder validates the actual copy description.
    value->cookie = retain_source_cookie(resource);
    if (!value->cookie) return result({}, status::unsupported_resource, record_stage::source_cookie);
    std::lock_guard lock(mutex);
    // Another call may have retained it meanwhile.
    if (auto existing = retained(value->cookie)) return result(existing, status::ready, record_stage::retained);
    auto entry = std::find_if(sources.begin(), sources.end(), [](const auto &value) { return value.expired(); });
    if (entry == sources.end()) {
      capacity_refused(source_table);
      reason = status::exhausted; return result({}, status::exhausted, record_stage::capacity);
    }
    // An adapter's source is watched for the enhanced-layout latch from now
    // on (proven_state.h); buffers and simultaneous-access textures keep no
    // layout between executions.
    if (value->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER &&
        !(value->desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)) {
      forget_destroyed();
      proven.watch(value->cookie);
    }
    *entry = value;
    return result(value, status::ready, record_stage::retained);
  }

  std::uint64_t source_present_generation(const source_ref &source) {
    return source ? source->present_generation.load(std::memory_order_acquire) : 0;
  }

  bool source_extent(const source_ref &source, std::uint32_t &width, std::uint32_t &height) {
    width = height = 0;
    if (!source || !source->desc.Width || source->desc.Width > UINT32_MAX) return false;
    width = static_cast<std::uint32_t>(source->desc.Width); height = source->desc.Height;
    return true;
  }
}
