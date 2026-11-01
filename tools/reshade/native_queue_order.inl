// SPDX-License-Identifier: GPL-3.0-only
// Step 4 host functions, included at the end of streamline_native_observer.cpp.
// Every order:: object and thread_local remains in that translation-unit shell.

namespace sunshine_streamline::native_observer {
  namespace {
    // Queue Signal: logged, with its poison checks, BEFORE it is forwarded
    // (design finding 7); a failed call leaves a conservative entry. Only a
    // call that is logged may wait on the order lock for its queue's record:
    // the add-on's own (suppressed) calls and ReShade's Signals of its own
    // fences (own_signal) take TryAcquire, as Execute does.
    // caller: the return address into whoever called the hooked slot, where
    // the second-signaller diagnostic starts its list of calling modules.
    template<unsigned Index> HRESULT STDMETHODCALLTYPE signal_detour(ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value) {
      const DWORD incoming = GetLastError();
      void *const caller = __builtin_return_address(0);
      auto &slot = targets[static_cast<unsigned>(method::signal)][Index];
      invocation call(method::signal, slot.status.load(std::memory_order_acquire));
      restore_error outgoing{incoming};
      const bool logged = call.eligible && !(fence && order_on() && own_signal(fence));
      order_enter(call, queue, logged);
      if (logged && !call.inherited && fence && order_on()) order_signal(queue, call, fence, value, caller);
      SetLastError(incoming);
      const HRESULT result = reinterpret_cast<queue_fence_fn>(slot.original.load(std::memory_order_acquire))(queue, fence, value);
      outgoing.value = GetLastError();
      return result;
    }

    // Queue Wait: logged only AFTER it returned success, with its quiet bit.
    template<unsigned Index> HRESULT STDMETHODCALLTYPE wait_detour(ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::wait)][Index];
      invocation call(method::wait, slot.status.load(std::memory_order_acquire));
      restore_error outgoing{incoming};
      order_enter(call, queue, call.eligible);
      SetLastError(incoming);
      const HRESULT result = reinterpret_cast<queue_fence_fn>(slot.original.load(std::memory_order_acquire))(queue, fence, value);
      outgoing.value = GetLastError();
      if (SUCCEEDED(result) && call.eligible && !call.inherited && call.ticket && fence && order_on())
        order_wait(call, fence, value, order_quiet(call));
      return result;
    }

    // CPU ID3D12Fence::Signal: poisons the fence before it is forwarded,
    // whatever its result.
    template<unsigned Index> HRESULT STDMETHODCALLTYPE fence_signal_detour(ID3D12Fence *fence, UINT64 value) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::fence_signal)][Index];
      invocation call(method::fence_signal, slot.status.load(std::memory_order_acquire));
      restore_error outgoing{incoming};
      if (call.eligible && !call.inherited && order_on()) order_cpu_signal(fence);
      SetLastError(incoming);
      const HRESULT result = reinterpret_cast<fence_signal_fn>(slot.original.load(std::memory_order_acquire))(fence, value);
      outgoing.value = GetLastError();
      return result;
    }

    // -- Queue order host --------------------------------------------------
    bool order_on() { return enabled() && order::requested.load(std::memory_order_acquire); }

    order::ticket_slot *ticket_of(std::uint64_t id) {
      const auto index = (id & 0xff) - 1;
      if (!id || index >= order::queue_limit) return nullptr;
      auto &slot = order::tickets[index];
      return slot.id.load(std::memory_order_acquire) == id ? &slot : nullptr;
    }

    // Requires order::lock exclusive: applies a deferred coverage reset.
    void apply_order_reset() {
      if (!order::reset_pending.load(std::memory_order_acquire) || !order::reset_pending.exchange(false, std::memory_order_acq_rel)) return;
      order::model.reset(GetTickCount64());
      for (auto &slot : order::tickets) slot.id.store(0, std::memory_order_release);
      ++order::resets;
    }

    bool private_bytes(void *object, const GUID &guid, void *output, UINT size) {
      auto *value = reinterpret_cast<ID3D12Object *>(object);
      UINT read = size;
      return SUCCEEDED(value->lpVtbl->GetPrivateData(value, guid, &read, output)) && read == size;
    }

    std::uint64_t device_of(ID3D12DeviceChild *child) {
      ID3D12Device *device{};
      if (FAILED(child->lpVtbl->GetDevice(child, IID_ID3D12Device, reinterpret_cast<void **>(&device))) || !device) return 0;
      const auto key = reinterpret_cast<std::uintptr_t>(device);
      device->lpVtbl->Release(device);
      return key;
    }

    // The queue's record id, sighting it (GetDevice, slot addresses, then
    // SetPrivateData outside the lock). Execute never blocks on the lock.
    std::uint64_t queue_id(ID3D12CommandQueue *queue, bool blocking) {
      order::queue_memory memory{};
      const bool found = private_bytes(queue, order_queue_guid, &memory, sizeof(memory));
      if (found && ticket_of(memory.id)) return memory.id;
      if (!found || !memory.cookie) memory.cookie = ++order::queue_cookies;
      const auto device = device_of(reinterpret_cast<ID3D12DeviceChild *>(queue));
      if (!device) return 0;
      std::array<void **, 3> slots{};
      const unsigned kinds[3]{static_cast<unsigned>(method::execute), static_cast<unsigned>(method::signal), static_cast<unsigned>(method::wait)};
      for (unsigned i = 0; i != 3; ++i)
        if (!read_slot_address(reinterpret_cast<std::uintptr_t>(queue), kinds[i], slots[i])) slots[i] = nullptr;
      if (blocking) AcquireSRWLockExclusive(&order::lock);
      else if (!TryAcquireSRWLockExclusive(&order::lock)) { ++order::contention; return 0; }
      apply_order_reset();
      memory.id = order::model.add_queue(device, memory.cookie);
      auto &slot = order::tickets[(memory.id & 0xff) - 1];
      slot.id.store(0, std::memory_order_release);
      slot.entered.store(0, std::memory_order_relaxed);
      slot.slots = slots;
      slot.cookie = memory.cookie;
      slot.type.store(0, std::memory_order_relaxed);
      slot.id.store(memory.id, std::memory_order_release);
      ReleaseSRWLockExclusive(&order::lock);
      auto *object = reinterpret_cast<ID3D12Object *>(queue);
      object->lpVtbl->SetPrivateData(object, order_queue_guid, sizeof(memory), &memory);
      return memory.id;
    }

    void order_enter(invocation &call, ID3D12CommandQueue *queue, bool blocking) {
      if (call.inherited || !queue || !order_on()) return;
      const restore_error incoming{GetLastError()};
      const auto id = queue_id(queue, blocking);
      if (auto *slot = ticket_of(id)) {
        call.order_queue = id;
        call.ticket = slot->entered.fetch_add(1, std::memory_order_acq_rel) + 1;
      }
    }

    // No later ticket of the queue was taken before this call returned.
    bool order_quiet(const invocation &call) {
      const auto *slot = ticket_of(call.order_queue);
      return slot && slot->entered.load(std::memory_order_acquire) == call.ticket;
    }

    sunshine_queue_order::fence_memory read_fence(ID3D12Fence *fence) {
      order::fence_bytes bytes{};
      if (!private_bytes(fence, order_fence_guid, &bytes, sizeof(bytes))) return {};
      return {bytes.id, bytes.signaller, static_cast<sunshine_queue_order::poison>(bytes.why)};
    }

    // Requires order::lock exclusive (SetPrivateData is a leaf runtime call):
    // the private data always reflects the model state it was written from,
    // so a stale write never replaces a newer poison or signaller.
    void write_fence(ID3D12Fence *fence, const sunshine_queue_order::fence_memory &value) {
      const order::fence_bytes bytes{value.id, value.signaller, static_cast<std::uint64_t>(value.why)};
      auto *object = reinterpret_cast<ID3D12Object *>(fence);
      object->lpVtbl->SetPrivateData(object, order_fence_guid, sizeof(bytes), &bytes);
    }

    // ReShade's Signal of a fence of its own: one it signals in the add-on's
    // flush of its immediate list (own_flush_scope), marked there in the
    // fence's private data, so that its later Signals (ReShade's own flush at
    // the end of a Present, a game thread's Execute flushing the list) are
    // known too. Never logged: no game queue waits on such a fence, and once
    // marked it is never logged again, so its record (if a Signal before the
    // mark made one) never reads a value no logged signal set. Until 10-27
    // each such Signal waited on the order lock, on the present thread at
    // every flush the add-on made.
    bool own_signal(ID3D12Fence *fence) {
      const restore_error incoming{GetLastError()};
      std::uint8_t mark{};
      if (private_bytes(fence, own_fence_guid, &mark, sizeof(mark)) && mark) {
        ++order::own_signals;
        return true;
      }
      if (!own_flush_depth) return false;
      mark = 1;
      auto *object = reinterpret_cast<ID3D12Object *>(fence);
      object->lpVtbl->SetPrivateData(object, own_fence_guid, sizeof(mark), &mark);
      ++order::own_signals;
      return true;
    }

    // The fence's record id, sighting it inside a game call on it: its CPU
    // Signal slot must be installed first (no record otherwise: its CPU
    // signals would go unseen), its creation flags and device are checked and
    // its completed value is the baseline. A re-sight keeps what its private
    // data remembers (poison, signaller). lost: the calling queue's record was
    // dropped (evicted or reset) since the call took its ticket.
    std::uint64_t fence_id(ID3D12Fence *fence, std::uint64_t queue, bool *lost) {
      auto memory = read_fence(fence);
      if (memory.id) {
        AcquireSRWLockShared(&order::lock);
        const bool known = order::model.known_fence(memory.id);
        ReleaseSRWLockShared(&order::lock);
        if (known) return memory.id;
      }
      const auto kind = static_cast<unsigned>(method::fence_signal);
      // The fence is the game's argument, alive in its call (read_fence just
      // called through its vtable): its slot's address needs no checked read.
      void **const address = reinterpret_cast<void **>(fence->lpVtbl) + vtable_slots[kind];
      bool hooked = false, seen = false;
      if (TryAcquireSRWLockShared(&targets_lock)) {
        if (auto *slot = find(kind, address)) {
          seen = true;
          hooked = slot->status.load(std::memory_order_acquire) == state::installed &&
            sunshine_native_vtable::matches(slot->location, detours[kind][slot - targets[kind].data()]);
        } else seen = refused_before(kind, reinterpret_cast<std::uintptr_t>(address));
        ReleaseSRWLockShared(&targets_lock);
      } else seen = true;  // Discovery in progress: the next call looks again.
      if (!hooked) {
        ++order::cpu_unhooked;
        // A slot pending, rejected or refused is never discovered again per
        // call: only a slot never seen is.
        if (!seen) observe(reinterpret_cast<std::uintptr_t>(fence), method::fence_signal);
        return 0;
      }
      auto why = sunshine_queue_order::poison::none;
      const auto mark = [&](sunshine_queue_order::poison value) { if (why == sunshine_queue_order::poison::none) why = value; };
      ID3D12Fence1 *fence1{};
      if (SUCCEEDED(fence->lpVtbl->QueryInterface(fence, IID_ID3D12Fence1, reinterpret_cast<void **>(&fence1))) && fence1) {
        const auto flags = fence1->lpVtbl->GetCreationFlags(fence1);
        fence1->lpVtbl->Release(fence1);
        if (flags & (D3D12_FENCE_FLAG_SHARED | D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER | D3D12_FENCE_FLAG_NON_MONITORED))
          mark(sunshine_queue_order::poison::shared);
      } else mark(sunshine_queue_order::poison::shared);
      const auto device = device_of(reinterpret_cast<ID3D12DeviceChild *>(fence));
      const auto baseline = fence->lpVtbl->GetCompletedValue(fence);
      AcquireSRWLockExclusive(&order::lock);
      apply_order_reset();
      // A call whose queue record was dropped since it took its ticket: the
      // caller re-sights its queue (order_signal) or the wait is not logged.
      if (!order::model.known_queue(queue)) {
        ReleaseSRWLockExclusive(&order::lock);
        if (lost) *lost = true;
        return 0;
      }
      // What the private data says now, under the lock its writers hold:
      // another thread may have sighted or poisoned the fence since.
      memory = read_fence(fence);
      if (!order::model.known_fence(memory.id)) {
        if (!device || device != order::model.queue_device(queue)) mark(sunshine_queue_order::poison::foreign_device);
        memory.id = order::model.add_fence(device, baseline, {0, memory.signaller, memory.why}, why, GetTickCount64());
        memory = order::model.memory(memory.id);
        write_fence(fence, memory);
      }
      ReleaseSRWLockExclusive(&order::lock);
      return memory.id;
    }

    template<class Operation> void order_update(ID3D12Fence *fence, std::uint64_t id, Operation operation) {
      AcquireSRWLockExclusive(&order::lock);
      apply_order_reset();
      if (order::model.known_fence(id)) {
        const auto before = order::model.memory(id);
        operation();
        const auto after = order::model.memory(id);
        if (after.signaller != before.signaller || after.why != before.why) write_fence(fence, after);
      }
      ReleaseSRWLockExclusive(&order::lock);
    }

    // A Signal that poisoned its fence as the fence's second signaller: the
    // fence's record id and its first signaller's queue cookie.
    struct second_signal {
      std::uint64_t fence{}, first{};
    };

    // Logs one queue Signal. False only with retry, when the call's queue
    // record was dropped (evicted, or a reset) since it took its ticket. A
    // signalling queue without a record poisons a fence already known.
    bool log_signal(const invocation &call, ID3D12Fence *fence, UINT64 value, bool retry, second_signal &poisoned) {
      bool lost = false;
      auto id = call.ticket ? fence_id(fence, call.order_queue, &lost) : read_fence(fence).id;
      if (lost && retry) return false;
      if (lost) id = read_fence(fence).id;
      if (!id) return true;
      const auto completed = fence->lpVtbl->GetCompletedValue(fence);
      bool dropped = false;
      order_update(fence, id, [&] {
        const bool recorded = call.ticket && !lost && order::model.known_queue(call.order_queue);
        if (call.ticket && !recorded && retry) { dropped = true; return; }
        const auto before = order::model.memory(id);
        order::model.signal(recorded ? call.order_queue : 0, id, value, recorded ? call.ticket : 0, completed, GetTickCount64());
        if (before.why == sunshine_queue_order::poison::none &&
            order::model.memory(id).why == sunshine_queue_order::poison::second_signaller)
          poisoned = {id, before.signaller};
      });
      return !dropped;
    }

    void order_signal(ID3D12CommandQueue *queue, invocation &call, ID3D12Fence *fence, UINT64 value, void *caller) {
      const restore_error incoming{GetLastError()};
      // The queue's type, once per record, for the second-signaller
      // diagnostic (a getter; no lock).
      if (auto *slot = ticket_of(call.order_queue); slot && !slot->type.load(std::memory_order_relaxed)) {
        D3D12_COMMAND_QUEUE_DESC desc{};
        queue->lpVtbl->GetDesc(queue, &desc);
        slot->type.store(static_cast<std::uint8_t>(desc.Type + 1), std::memory_order_relaxed);
      }
      // The native call is not forwarded yet: a record dropped since the call
      // took its ticket is re-sighted with a fresh ticket, so a signal never
      // goes unlogged against a live fence record (a pending wait would
      // resolve to a later signal of its queue, after work this one precedes).
      second_signal poisoned;
      if (!log_signal(call, fence, value, true, poisoned)) {
        call.order_queue = call.ticket = 0;
        order_enter(call, queue, true);
        log_signal(call, fence, value, false, poisoned);
      }
      if (poisoned.fence) report_second_signaller(queue, poisoned, value, caller);
    }

    const char *queue_type_name(std::uint8_t type) {
      switch (type) {
        case D3D12_COMMAND_LIST_TYPE_DIRECT + 1: return "direct";
        case D3D12_COMMAND_LIST_TYPE_COMPUTE + 1: return "compute";
        case D3D12_COMMAND_LIST_TYPE_COPY + 1: return "copy";
        case 0: return "unknown";
        default: return "other";
      }
    }

    // The modules of the calling stack, from the frame that called the hooked
    // slot (caller) outwards, consecutive frames of one module merged:
    // "a<b<c" (a called the slot, c is the outermost listed). A stack the
    // walk does not reach caller on starts after this module's own frames.
    void calling_modules(void *caller, char *out, std::size_t size) {
      void *frames[48]{};
      const auto count = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
      const auto module_of = [](void *address) {
        HMODULE module{};
        return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(address), &module) ? module : nullptr;
      };
      unsigned first = 0;
      while (first < count && frames[first] != caller) ++first;
      if (first == count) {
        const auto own = module_of(reinterpret_cast<void *>(&calling_modules));
        first = 0;
        while (first < count && module_of(frames[first]) == own) ++first;
      }
      std::size_t used = 0;
      out[0] = 0;
      HMODULE previous{};
      for (unsigned i = first, listed = 0; i < count && listed < 8 && used + 1 < size; ++i) {
        const auto module = module_of(frames[i]);
        if (listed && module == previous) continue;
        previous = module;
        char path[MAX_PATH]{};
        const char *name = "?";
        if (module && GetModuleFileNameA(module, path, MAX_PATH)) {
          name = path;
          for (const char *at = path; *at; ++at)
            if (*at == '\\' || *at == '/') name = at + 1;
        }
        const int wrote = std::snprintf(out + used, size - used, "%s%s", listed ? "<" : "", name);
        if (wrote < 0) break;
        used = std::min(size - 1, used + static_cast<std::size_t>(wrote));
        ++listed;
      }
    }

    // One report per queue that poisons a fence as its second signaller
    // (its cookie), at most order::report_limit queues: both queues and the
    // second one's calling modules (who signals the fence besides its first
    // signaller: the game, Streamline or DLSS-G, ReShade or the add-on).
    // Called after the order lock was released; costs a stack walk only for a
    // queue not reported yet, and each fence is poisoned once.
    void report_second_signaller(ID3D12CommandQueue *queue, const second_signal &poisoned, UINT64 value, void *caller) {
      order::queue_memory memory{};
      if (!private_bytes(queue, order_queue_guid, &memory, sizeof(memory)) || !memory.cookie) return;
      const auto known = [&] {
        return std::find(order::reported.begin(), order::reported.begin() + order::report_count, memory.cookie) !=
          order::reported.begin() + order::report_count;
      };
      AcquireSRWLockShared(&order::report_lock);
      const bool seen = known(), full = order::report_count == order::report_limit;
      ReleaseSRWLockShared(&order::report_lock);
      if (seen || full) return;
      std::uint8_t first_type = 0, second_type = 0;
      AcquireSRWLockShared(&order::lock);
      for (const auto &slot : order::tickets) {
        if (!slot.id.load(std::memory_order_acquire)) continue;
        if (slot.cookie == poisoned.first) first_type = slot.type.load(std::memory_order_relaxed);
        if (slot.cookie == memory.cookie) second_type = slot.type.load(std::memory_order_relaxed);
      }
      ReleaseSRWLockShared(&order::lock);
      char callers[256];
      calling_modules(caller, callers, sizeof(callers));
      std::array<char, order::report_size> text{};
      AcquireSRWLockExclusive(&order::report_lock);
      if (known() || order::report_count == order::report_limit) {
        ReleaseSRWLockExclusive(&order::report_lock);
        return;
      }
      std::snprintf(text.data(), text.size(),
        "Sunshine 3D queue order: second signaller fence=%llu value=%llu first={queue=%llu type=%s} "
        "second={queue=%llu type=%s callers=%s} report=%u/%u",
        static_cast<unsigned long long>(poisoned.fence), static_cast<unsigned long long>(value),
        static_cast<unsigned long long>(poisoned.first), queue_type_name(first_type),
        static_cast<unsigned long long>(memory.cookie), queue_type_name(second_type), callers,
        order::report_count + 1, order::report_limit);
      order::reported[order::report_count] = memory.cookie;
      order::reports[order::report_count++] = text;
      ReleaseSRWLockExclusive(&order::report_lock);
    }

    void order_wait(const invocation &call, ID3D12Fence *fence, UINT64 value, bool quiet) {
      const restore_error incoming{GetLastError()};
      const auto id = fence_id(fence, call.order_queue);
      const auto completed = fence->lpVtbl->GetCompletedValue(fence);
      AcquireSRWLockExclusive(&order::lock);
      apply_order_reset();
      // An unknown fence still logs the wait (refused as untracked).
      const bool known = id && order::model.known_fence(id);
      const auto before = known ? order::model.memory(id) : sunshine_queue_order::fence_memory{};
      order::model.wait(call.order_queue, id, value, call.ticket, quiet, completed, GetTickCount64());
      if (known) {
        const auto after = order::model.memory(id);
        if (after.signaller != before.signaller || after.why != before.why) write_fence(fence, after);
      }
      ReleaseSRWLockExclusive(&order::lock);
    }

    void order_cpu_signal(ID3D12Fence *fence) {
      const restore_error incoming{GetLastError()};
      AcquireSRWLockExclusive(&order::lock);
      apply_order_reset();
      // Read under the lock its writers hold. Never sighted: its baseline at
      // sight covers this signal.
      auto memory = read_fence(fence);
      if (memory.id && order::model.known_fence(memory.id)) {
        order::model.cpu_signal(memory.id, GetTickCount64());
        const auto after = order::model.memory(memory.id);
        if (after.signaller != memory.signaller || after.why != memory.why) write_fence(fence, after);
      } else if (memory.id && memory.why == sunshine_queue_order::poison::none) {
        memory.why = sunshine_queue_order::poison::cpu_signal;
        write_fence(fence, memory);
      }
      ReleaseSRWLockExclusive(&order::lock);
    }

    // Whether an installed slot still holds its detour now. Its table's
    // module is pinned at install (sunshine_native_vtable::install), so a
    // plain read of the slot is safe and costs no syscall.
    bool holds(const target &slot, unsigned kind) {
      return slot.status.load(std::memory_order_acquire) == state::installed &&
        *reinterpret_cast<void *const volatile *>(slot.location.address) == detours[kind][&slot - targets[kind].data()];
    }

    // Whether a queue's Execute, Signal and Wait slots are all installed and
    // still hold their detours. Requires targets_lock shared.
    bool covered(std::uint64_t id) {
      const auto *slot = ticket_of(id);
      if (!slot) return false;
      const unsigned kinds[3]{static_cast<unsigned>(method::execute), static_cast<unsigned>(method::signal), static_cast<unsigned>(method::wait)};
      for (unsigned i = 0; i != 3; ++i) {
        const auto *target = slot->slots[i] ? find(kinds[i], slot->slots[i]) : nullptr;
        if (!target || !holds(*target, kinds[i])) return false;
      }
      return true;
    }

    // Requires targets_lock shared.
    unsigned order_pending() {
      unsigned count = 0;
      for (unsigned kind = 0; kind != method_count; ++kind)
        if (order_family(kind))
          for (unsigned i = 0; i != target_counts[kind]; ++i)
            count += targets[kind][i].status.load(std::memory_order_acquire) == state::pending;
      return count;
    }

    slot_check verify_order_slots() {
      bool lost_any = false;
      if (!TryAcquireSRWLockShared(&targets_lock)) return slot_check::busy;
      for (unsigned kind = 0; kind != method_count; ++kind)
        if (order_family(kind))
          for (unsigned i = 0; i != target_counts[kind]; ++i) {
            auto &slot = targets[kind][i];
            auto expected = state::installed;
            if (slot.status.load(std::memory_order_acquire) == state::installed && !holds(slot, kind) &&
                slot.status.compare_exchange_strong(expected, state::rejected, std::memory_order_acq_rel))
              lost_any = true;
          }
      ReleaseSRWLockShared(&targets_lock);
      if (!lost_any) return slot_check::verified;
      ++rejected; coverage_changed(static_cast<unsigned>(method::signal));
      return slot_check::lost;
    }
  }

  void set_queue_order(bool enabled) {
    order::requested.store(enabled, std::memory_order_release);
    order::reset_pending.store(true, std::memory_order_release);
  }

  bool queue_order_enabled() { return order::requested.load(std::memory_order_acquire); }

  position last_execute_position() { return order::last_execute; }

  sunshine_queue_order::verdict queue_ordered(std::uint64_t reader, position producer) {
    using sunshine_queue_order::refusal;
    sunshine_queue_order::verdict out;
    const auto refuse = [&](refusal why) {
      out.why = why;
      ++order::refused[static_cast<unsigned>(why)];
      return out;
    };
    if (!order_on() || !reader) return refuse(refusal::disabled);
    const restore_error incoming{GetLastError()};
    // Every installed order slot still holds its detour now, not only at the
    // last poll: a CPU Signal or a queue Signal that bypassed a replaced slot
    // since then was never logged.
    switch (verify_order_slots()) {
      case slot_check::busy: return refuse(refusal::pending_hooks);
      case slot_check::lost: return refuse(refusal::uncovered);
      default: break;
    }
    order::queue_memory memory{};
    const bool tracked = private_bytes(reinterpret_cast<void *>(reader), order_queue_guid, &memory, sizeof(memory)) && ticket_of(memory.id);
    if (!TryAcquireSRWLockShared(&targets_lock)) return refuse(refusal::pending_hooks);
    const unsigned pending = order_pending();
    const bool reader_covered = tracked && covered(memory.id), producer_covered = covered(producer.queue);
    ReleaseSRWLockShared(&targets_lock);
    if (pending) return refuse(refusal::pending_hooks);
    // A coverage change not yet applied: the model is about to be reset.
    if (order::reset_pending.load(std::memory_order_acquire)) return refuse(refusal::disabled);
    if (!tracked) return refuse(refusal::untracked);
    if (!reader_covered || !producer_covered) return refuse(refusal::uncovered);
    // The present thread never waits for a game thread's hold: a verdict
    // the lock is busy for is refused, so the completion rule decides.
    if (!TryAcquireSRWLockShared(&order::lock)) {
      ++order::contention;
      out.why = refusal::pending_hooks;
      return out;
    }
    out = order::model.ordered(memory.id, producer);
    ReleaseSRWLockShared(&order::lock);
    if (out.ordered) ++order::ordered_count;
    else ++order::refused[static_cast<unsigned>(out.why)];
    return out;
  }

  bool queue_order_rely(const sunshine_queue_order::verdict &value, std::uint64_t tag) {
    if (!value.ordered) return false;
    // Busy: the read could not be protected, so it is not made (fail-safe).
    if (!TryAcquireSRWLockExclusive(&order::lock)) {
      ++order::contention;
      return false;
    }
    // A pending reset forgets the verdict's fences, which conflicts it here.
    apply_order_reset();
    const bool stands = order::model.rely(value, GetTickCount64(), tag);
    ReleaseSRWLockExclusive(&order::lock);
    return stands;
  }

  bool queue_order_conflicted(std::uint64_t tag) {
    if (!tag) return false;
    // Busy: the last answer for this tag. A conflict logged during that
    // hold is seen at the next Present, as one logged just after this call.
    if (!TryAcquireSRWLockShared(&order::lock)) {
      ++order::contention;
      return order::conflicted_seen.load(std::memory_order_relaxed) == tag;
    }
    const bool out = order::model.conflicted(tag);
    ReleaseSRWLockShared(&order::lock);
    if (out) order::conflicted_seen.store(tag, std::memory_order_relaxed);
    return out;
  }

  void queue_order_forget_device(std::uint64_t device) {
    if (!device) return;
    AcquireSRWLockExclusive(&order::lock);
    order::model.forget_device(device);
    for (auto &slot : order::tickets)
      if (const auto id = slot.id.load(std::memory_order_acquire); id && !order::model.known_queue(id)) slot.id.store(0, std::memory_order_release);
    ReleaseSRWLockExclusive(&order::lock);
  }

  queue_order_status queue_order_counts() {
    queue_order_status out;
    out.enabled = order::requested.load(std::memory_order_acquire);
    if (TryAcquireSRWLockShared(&targets_lock)) {
      const auto installed_of = [](method value) {
        unsigned count = 0;
        const auto kind = static_cast<unsigned>(value);
        for (unsigned i = 0; i != target_counts[kind]; ++i) count += targets[kind][i].status.load(std::memory_order_acquire) == state::installed;
        return count;
      };
      out.signal_hooks = installed_of(method::signal);
      out.wait_hooks = installed_of(method::wait);
      out.fence_signal_hooks = installed_of(method::fence_signal);
      out.pending = order_pending();
      ReleaseSRWLockShared(&targets_lock);
    }
    // The status line never waits for a game thread's hold.
    if (TryAcquireSRWLockShared(&order::lock)) {
      out.model = order::model.snapshot();
      ReleaseSRWLockShared(&order::lock);
    } else out.busy = true;
    out.ordered = order::ordered_count.load(std::memory_order_relaxed);
    out.resets = order::resets.load(std::memory_order_relaxed);
    out.cpu_unhooked = order::cpu_unhooked.load(std::memory_order_relaxed);
    out.contention = order::contention.load(std::memory_order_relaxed);
    out.own_signals = order::own_signals.load(std::memory_order_relaxed);
    for (unsigned i = 0; i != order::refused.size(); ++i) out.refused[i] = order::refused[i].load(std::memory_order_relaxed);
    return out;
  }

  bool queue_order_signaller_line(char *out, std::size_t size) {
    if (!size || !TryAcquireSRWLockExclusive(&order::report_lock)) return false;
    const bool pending = order::report_read < order::report_count;
    if (pending) std::snprintf(out, size, "%s", order::reports[order::report_read++].data());
    ReleaseSRWLockExclusive(&order::report_lock);
    return pending;
  }

  HRESULT signal_own(ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value) {
    const suppression_scope own;
    return queue->lpVtbl->Signal(queue, fence, value);
  }
}
