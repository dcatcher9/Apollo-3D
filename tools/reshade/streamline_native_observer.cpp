// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_native_observer.h"
#include "addon_lifetime.h"
#define CINTERFACE
#include <d3d12.h>
#include "native_vtable_slot.h"
#include <array>
#include <atomic>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <new>

namespace sunshine_streamline::native_observer {
  namespace {
    // Command-list lifecycle (Reset, Close, render passes, bundles) comes from
    // ReShade's events. The hooks observe resource states and submission
    // (the state family) and queue order (the order family: signal, wait,
    // fence_signal; execute also takes tickets for it).
    enum class method : unsigned { barriers, execute, enhanced, signal, wait, fence_signal, count };
    constexpr unsigned method_count = static_cast<unsigned>(method::count);
    constexpr unsigned target_limit = 8, command_chunk = 64, inline_barriers = 256;
    constexpr std::array<std::size_t, method_count> vtable_slots{
      offsetof(ID3D12GraphicsCommandListVtbl, ResourceBarrier) / sizeof(void *),
      offsetof(ID3D12CommandQueueVtbl, ExecuteCommandLists) / sizeof(void *),
      offsetof(ID3D12GraphicsCommandList7Vtbl, Barrier) / sizeof(void *),
      offsetof(ID3D12CommandQueueVtbl, Signal) / sizeof(void *),
      offsetof(ID3D12CommandQueueVtbl, Wait) / sizeof(void *),
      offsetof(ID3D12FenceVtbl, Signal) / sizeof(void *)};
    static_assert(vtable_slots[0] == 26 && vtable_slots[1] == 10 && vtable_slots[2] == 80, "Unexpected D3D12 native COM ABI");
    static_assert(vtable_slots[3] == 14 && vtable_slots[4] == 15 && vtable_slots[5] == 10, "Unexpected D3D12 native COM ABI");
    static_assert(offsetof(ID3D12FenceVtbl, GetCompletedValue) / sizeof(void *) == 8 &&
      offsetof(ID3D12Fence1Vtbl, GetCreationFlags) / sizeof(void *) == 11, "Unexpected D3D12 native COM ABI");
    constexpr bool order_family(unsigned kind) { return kind >= static_cast<unsigned>(method::signal); }
    constexpr hooked_method hooked_table[method_count]{
      {"ID3D12GraphicsCommandList", "ResourceBarrier", 26}, {"ID3D12CommandQueue", "ExecuteCommandLists", 10},
      {"ID3D12GraphicsCommandList7", "Barrier", 80}, {"ID3D12CommandQueue", "Signal", 14},
      {"ID3D12CommandQueue", "Wait", 15}, {"ID3D12Fence", "Signal", 10}};
    using barrier_fn = void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *, UINT, const D3D12_RESOURCE_BARRIER *);
    using execute_fn = void (STDMETHODCALLTYPE *)(ID3D12CommandQueue *, UINT, ID3D12CommandList *const *);
    using enhanced_fn = void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList7 *, UINT32, const D3D12_BARRIER_GROUP *);
    using queue_fence_fn = HRESULT (STDMETHODCALLTYPE *)(ID3D12CommandQueue *, ID3D12Fence *, UINT64);
    using fence_signal_fn = HRESULT (STDMETHODCALLTYPE *)(ID3D12Fence *, UINT64);
    constexpr GUID recording_guid{0x92c957c6, 0x88de, 0x42b3, {0x9f, 0x3e, 0x31, 0xde, 0x63, 0x05, 0x7b, 0x29}};
    // Queue order identities (never SetPrivateDataInterface: plain bytes, so
    // no game object points into the add-on).
    constexpr GUID order_queue_guid{0x3c1d7a52, 0x6b0e, 0x4f43, {0x8d, 0x27, 0x5a, 0x90, 0xe1, 0x44, 0x0b, 0x6c}};
    constexpr GUID order_fence_guid{0x7e64b1a9, 0x2f58, 0x4c0d, {0xa3, 0x91, 0x0c, 0x6f, 0x5d, 0x28, 0xe4, 0x17}};
    enum class state { empty, pending, installed, rejected };
    struct target {
      sunshine_native_vtable::slot location;
      std::atomic<void *> original{};
      std::atomic<state> status{state::empty};
    };
    std::array<std::array<target, target_limit>, method_count> targets;
    std::array<unsigned, method_count> target_counts{};
    // Every later object sharing a refused vtable slot is refused identically.
    // Report that coverage change once; repeating it would revoke all evidence
    // on every call. QueryInterface is per object (wrappers share vtables), so
    // its refusals are never remembered.
    constexpr unsigned refused_limit = 16;
    std::array<std::array<std::uintptr_t, refused_limit>, method_count> refused{};
    std::array<unsigned, method_count> refused_counts{};
    SRWLOCK targets_lock = SRWLOCK_INIT;
    std::atomic<bool> requested{}, active{};
    std::atomic<std::uint64_t> activation_epoch{};
    std::atomic<decltype(callbacks::barriers)> barrier_callback{};
    std::atomic<decltype(callbacks::submitted)> submitted_callback{};
    std::atomic<decltype(callbacks::invalidated)> invalidated_callback{};
    std::atomic<decltype(callbacks::enhanced_textures)> enhanced_textures_callback{};
    std::atomic<decltype(callbacks::associate_recording)> associate_recording_callback{};
    std::atomic<submission_listener> listener_callback{};
    std::atomic<std::uint64_t> calls{}, observed{}, unreadable{}, dropped{}, installed{}, rejected{}, nested{}, suppressed{};
    std::atomic<std::uint64_t> barrier_overflow{}, submission_overflow{}, discovery_contention{};
    thread_local unsigned suppression_depth{};
    struct invocation;
    thread_local std::array<invocation *, method_count> current_invocations{};

    // Queue order state. Tickets live in a lock-free table indexed by the
    // queue record's slot; the model and everything else under order_lock.
    namespace order {
      using model_type = sunshine_queue_order::model<>;
      constexpr unsigned queue_limit = 32;
      struct ticket_slot {
        std::atomic<std::uint64_t> id{}, entered{};
        // The queue's Execute, Signal and Wait vtable slots at first sight
        // (coverage); written under order_lock before id is published.
        std::array<void **, 3> slots{};
      };
      std::array<ticket_slot, queue_limit> tickets;
      SRWLOCK lock = SRWLOCK_INIT;
      model_type model;
      std::atomic<bool> requested{true}, reset_pending{};
      std::atomic<std::uint64_t> queue_cookies{}, resets{}, ordered_count{}, cpu_unhooked{}, contention{};
      std::array<std::atomic<std::uint64_t>, static_cast<unsigned>(sunshine_queue_order::refusal::count)> refused{};
      thread_local position last_execute{};
      struct queue_memory { std::uint64_t cookie, id; };
      struct fence_bytes { std::uint64_t id, signaller, why; };
    }

    struct restore_error {
      DWORD value;
      ~restore_error() { SetLastError(value); }
    };
    bool enabled() {
      return !sunshine_addon_lifetime::stopping() && requested.load(std::memory_order_acquire) && active.load(std::memory_order_acquire);
    }
    struct invocation {
      unsigned kind;
      invocation *parent;
      bool child{};
      std::uint64_t epoch;
      bool eligible;
      unsigned depth;
      // A proxy-to-native call at the same suppression depth (inherited)
      // shares its parent's ticket; a suppressed call made inside a game
      // call (the add-on's copy-out) is never its child and takes its own.
      bool inherited{};
      std::uint64_t order_queue{}, ticket{};
      invocation(method value, state status) : kind(static_cast<unsigned>(value)), parent(current_invocations[kind]),
          epoch(activation_epoch.load(std::memory_order_acquire)), eligible(status == state::installed && enabled() && !suppression_depth),
          depth(suppression_depth) {
        ++calls;
        if (suppression_depth) ++suppressed;
        // Only the deepest hooked native implementation reports an operation.
        // In particular a proxy's post-call must not report the submission twice.
        if (parent && parent->depth == depth) {
          parent->child = true; ++nested;
          inherited = true; order_queue = parent->order_queue; ticket = parent->ticket;
        }
        current_invocations[kind] = this;
      }
      ~invocation() { current_invocations[kind] = parent; }
      bool notify() const { return eligible && !child && enabled() && epoch == activation_epoch.load(std::memory_order_acquire); }
    };
    bool read(const void *source, void *output, std::size_t size) {
      SIZE_T copied{};
      return source && ReadProcessMemory(GetCurrentProcess(), source, output, size, &copied) && copied == size;
    }
    template<class T, std::size_t Inline> struct batch_snapshot {
      // Keep ordinary calls allocation-free, without treating inline storage
      // size as an API limit. Each invocation owns its snapshot independently
      // so nested proxy forwarding cannot overwrite an outer call's evidence.
      std::array<T, Inline> local;
      std::unique_ptr<T[]> overflow;
      bool prepare(std::size_t count) noexcept {
        if (count <= Inline) return true;
        if (count > SIZE_MAX / sizeof(T)) return false;
        try { overflow.reset(new (std::nothrow) T[count]); }
        catch (...) { return false; }
        return bool(overflow);
      }
      T *data() noexcept { return overflow ? overflow.get() : local.data(); }
    };
    template<class T> bool read_elements(const T *source, std::size_t offset, std::size_t count, T *output) {
      const auto address = reinterpret_cast<std::uintptr_t>(source);
      if (!address || offset > (UINTPTR_MAX - address) / sizeof(T)) return false;
      const auto begin = address + offset * sizeof(T);
      if (count > (UINTPTR_MAX - begin) / sizeof(T)) return false;
      return read(reinterpret_cast<const void *>(begin), output, count * sizeof(T));
    }
    template<class Callback, class... Args> void notify(Callback callback, Args... args) noexcept {
      if (!callback || !enabled() || suppression_depth) return;
      suppression_scope scope;
      try { ++observed; callback(args...); }
      catch (...) {
        active.store(false, std::memory_order_release); ++activation_epoch; ++dropped;
        // Fail closed even if a callback threw after partially updating owner
        // state. Do not recurse through notify if invalidation itself throws.
        if (const auto failed = sunshine_addon_lifetime::stopping() ? nullptr : invalidated_callback.load(std::memory_order_acquire)) {
          try { failed(); }
          catch (...) { ++dropped; }
        }
      }
    }
    // The submission listener: the same post-call point and suppression as
    // notify, without its fail-closed invalidation (the listener is noexcept).
    void listen(std::uintptr_t queue, unsigned count, const command_identity *commands) noexcept {
      const auto listener = listener_callback.load(std::memory_order_acquire);
      if (!listener || !enabled() || suppression_depth) return;
      suppression_scope scope;
      listener(queue, count, commands);
    }
    void invalidate() noexcept { notify(invalidated_callback.load(std::memory_order_acquire)); }
    void drop(bool unreadable_input) noexcept {
      if (unreadable_input) ++unreadable;
      else ++dropped;
      invalidate();
    }

    std::uint64_t operation_cookie(std::uint64_t command) {
      auto cookie = get_recording_cookie(command);
      if (!cookie && recording_cookie_absent(command))
        notify(associate_recording_callback.load(std::memory_order_acquire), command, &cookie);
      return cookie;
    }

    // Defined after the slot helpers below.
    bool order_on();
    void order_enter(invocation &call, ID3D12CommandQueue *queue, bool blocking);
    bool order_quiet(const invocation &call);
    void order_signal(ID3D12CommandQueue *queue, invocation &call, ID3D12Fence *fence, UINT64 value);
    void order_wait(const invocation &call, ID3D12Fence *fence, UINT64 value, bool quiet);
    void order_cpu_signal(ID3D12Fence *fence);

    template<unsigned Index> void STDMETHODCALLTYPE barrier_detour(ID3D12GraphicsCommandList *command,
        UINT count, const D3D12_RESOURCE_BARRIER *values) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::barriers)][Index];
      invocation call(method::barriers, slot.status.load(std::memory_order_acquire));
      restore_error outgoing{incoming}; // Restore after snapshot storage is freed.
      batch_snapshot<D3D12_RESOURCE_BARRIER, inline_barriers> copied;
      std::uint64_t cookie{};
      bool complete = true, allocated = true;
      if (call.eligible) {
        cookie = operation_cookie(reinterpret_cast<std::uintptr_t>(command));
        allocated = copied.prepare(count);
        complete = allocated && (!count || read_elements(values, 0, count, copied.data()));
      }
      SetLastError(incoming);
      reinterpret_cast<barrier_fn>(slot.original.load(std::memory_order_acquire))(command, count, values);
      outgoing.value = GetLastError();
      if (!call.notify()) return;
      if (!complete) { if (!allocated) ++barrier_overflow; drop(allocated); return; }
      notify(barrier_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(command), cookie, count, copied.data());
    }
    template<unsigned Index> void STDMETHODCALLTYPE execute_detour(ID3D12CommandQueue *queue,
        UINT count, ID3D12CommandList *const *commands) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::execute)][Index];
      invocation call(method::execute, slot.status.load(std::memory_order_acquire));
      restore_error outgoing{incoming}; // Restore after snapshot storage is freed.
      std::array<ID3D12CommandList *, command_chunk> copied;
      batch_snapshot<command_identity, command_chunk> identities;
      bool complete = true, allocated = true;
      if (call.eligible) {
        allocated = identities.prepare(count);
        complete = allocated;
        // Snapshot every identity before forwarding. Chunk only the CPU read;
        // the game's original ExecuteCommandLists remains one unchanged call.
        for (UINT offset = 0; complete && offset < count;) {
          const UINT length = std::min(count - offset, command_chunk);
          complete = read_elements(commands, offset, length, copied.data());
          if (!complete) break;
          for (UINT i = 0; i != length; ++i) {
            auto &identity = identities.data()[offset + i];
            identity.native_command = reinterpret_cast<std::uintptr_t>(copied[i]);
            identity.recording_cookie = get_recording_cookie(identity.native_command);
          }
          offset += length;
        }
      }
      // A ticket, suppressed or not (the add-on's copy-out lists get positions).
      order_enter(call, queue, false);
      SetLastError(incoming);
      reinterpret_cast<execute_fn>(slot.original.load(std::memory_order_acquire))(queue, count, commands);
      outgoing.value = GetLastError();
      order::last_execute = call.ticket ? position{call.order_queue, call.ticket, order_quiet(call)} : position{};
      if (!call.notify()) return;
      if (!complete) { if (!allocated) ++submission_overflow; drop(allocated); return; }
      notify(submitted_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(queue), count, identities.data());
      listen(reinterpret_cast<std::uintptr_t>(queue), count, identities.data());
    }
    template<unsigned Index> void STDMETHODCALLTYPE enhanced_detour(ID3D12GraphicsCommandList7 *command, UINT32 count, const D3D12_BARRIER_GROUP *groups) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::enhanced)][Index];
      invocation call(method::enhanced, slot.status.load(std::memory_order_acquire));
      restore_error outgoing{incoming}; // Restore after snapshot storage is freed.
      // Only texture barriers change a texture's layout. Snapshot the textures
      // they name and their new layouts before forwarding, as the legacy
      // barrier hook does.
      batch_snapshot<enhanced_texture, inline_barriers> textures;
      std::uint64_t cookie{};
      std::size_t named = 0;
      bool complete = true, allocated = true;
      if (call.eligible) {
        cookie = operation_cookie(reinterpret_cast<std::uintptr_t>(command));
        batch_snapshot<D3D12_BARRIER_GROUP, 16> copied;
        complete = (allocated = copied.prepare(count)) && (!count || read_elements(groups, 0, count, copied.data()));
        for (UINT32 i = 0; complete && i != count; ++i)
          if (copied.data()[i].Type == D3D12_BARRIER_TYPE_TEXTURE) named += copied.data()[i].NumBarriers;
        complete = complete && (allocated = textures.prepare(named));
        std::size_t at = 0;
        for (UINT32 i = 0; complete && i != count; ++i) {
          const auto &group = copied.data()[i];
          if (group.Type != D3D12_BARRIER_TYPE_TEXTURE || !group.NumBarriers) continue;
          batch_snapshot<D3D12_TEXTURE_BARRIER, 16> texture_barriers;
          complete = (allocated = texture_barriers.prepare(group.NumBarriers)) &&
            read_elements(group.pTextureBarriers, 0, group.NumBarriers, texture_barriers.data());
          for (UINT32 b = 0; complete && b != group.NumBarriers; ++b) {
            const auto &barrier = texture_barriers.data()[b];
            const auto &range = barrier.Subresources;
            // 0xffffffff names every subresource; NumMipLevels 0 names one by index.
            const bool first = range.IndexOrFirstMipLevel == UINT32_MAX ||
              (range.NumMipLevels == 0 ? range.IndexOrFirstMipLevel == 0 :
                range.IndexOrFirstMipLevel == 0 && range.FirstArraySlice == 0 && range.FirstPlane == 0 &&
                range.NumArraySlices != 0 && range.NumPlanes != 0);
            textures.data()[at++] = {barrier.pResource, static_cast<std::uint32_t>(barrier.LayoutAfter), first,
              static_cast<std::uint32_t>(barrier.AccessAfter), static_cast<std::uint32_t>(barrier.SyncBefore),
              static_cast<std::uint32_t>(barrier.SyncAfter)};
          }
        }
      }
      SetLastError(incoming);
      reinterpret_cast<enhanced_fn>(slot.original.load(std::memory_order_acquire))(command, count, groups);
      outgoing.value = GetLastError();
      if (!call.notify()) return;
      if (!complete) { if (!allocated) ++barrier_overflow; drop(allocated); return; }
      if (named) notify(enhanced_textures_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(command),
        cookie, static_cast<unsigned>(named), static_cast<const enhanced_texture *>(textures.data()));
    }
    // Queue Signal: logged, with its poison checks, BEFORE it is forwarded
    // (design finding 7); a failed call leaves a conservative entry.
    template<unsigned Index> HRESULT STDMETHODCALLTYPE signal_detour(ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::signal)][Index];
      invocation call(method::signal, slot.status.load(std::memory_order_acquire));
      restore_error outgoing{incoming};
      order_enter(call, queue, true);
      if (call.eligible && !call.inherited && fence && order_on()) order_signal(queue, call, fence, value);
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
      order_enter(call, queue, true);
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
    const std::array<std::array<void *, target_limit>, method_count> detours{{
      {reinterpret_cast<void *>(barrier_detour<0>), reinterpret_cast<void *>(barrier_detour<1>), reinterpret_cast<void *>(barrier_detour<2>), reinterpret_cast<void *>(barrier_detour<3>), reinterpret_cast<void *>(barrier_detour<4>), reinterpret_cast<void *>(barrier_detour<5>), reinterpret_cast<void *>(barrier_detour<6>), reinterpret_cast<void *>(barrier_detour<7>)},
      {reinterpret_cast<void *>(execute_detour<0>), reinterpret_cast<void *>(execute_detour<1>), reinterpret_cast<void *>(execute_detour<2>), reinterpret_cast<void *>(execute_detour<3>), reinterpret_cast<void *>(execute_detour<4>), reinterpret_cast<void *>(execute_detour<5>), reinterpret_cast<void *>(execute_detour<6>), reinterpret_cast<void *>(execute_detour<7>)},
      {reinterpret_cast<void *>(enhanced_detour<0>), reinterpret_cast<void *>(enhanced_detour<1>), reinterpret_cast<void *>(enhanced_detour<2>), reinterpret_cast<void *>(enhanced_detour<3>), reinterpret_cast<void *>(enhanced_detour<4>), reinterpret_cast<void *>(enhanced_detour<5>), reinterpret_cast<void *>(enhanced_detour<6>), reinterpret_cast<void *>(enhanced_detour<7>)},
      {reinterpret_cast<void *>(signal_detour<0>), reinterpret_cast<void *>(signal_detour<1>), reinterpret_cast<void *>(signal_detour<2>), reinterpret_cast<void *>(signal_detour<3>), reinterpret_cast<void *>(signal_detour<4>), reinterpret_cast<void *>(signal_detour<5>), reinterpret_cast<void *>(signal_detour<6>), reinterpret_cast<void *>(signal_detour<7>)},
      {reinterpret_cast<void *>(wait_detour<0>), reinterpret_cast<void *>(wait_detour<1>), reinterpret_cast<void *>(wait_detour<2>), reinterpret_cast<void *>(wait_detour<3>), reinterpret_cast<void *>(wait_detour<4>), reinterpret_cast<void *>(wait_detour<5>), reinterpret_cast<void *>(wait_detour<6>), reinterpret_cast<void *>(wait_detour<7>)},
      {reinterpret_cast<void *>(fence_signal_detour<0>), reinterpret_cast<void *>(fence_signal_detour<1>), reinterpret_cast<void *>(fence_signal_detour<2>), reinterpret_cast<void *>(fence_signal_detour<3>), reinterpret_cast<void *>(fence_signal_detour<4>), reinterpret_cast<void *>(fence_signal_detour<5>), reinterpret_cast<void *>(fence_signal_detour<6>), reinterpret_cast<void *>(fence_signal_detour<7>)}
    }};
    target *find(unsigned kind, void **address) {
      for (unsigned i = 0; i != target_counts[kind]; ++i) if (targets[kind][i].location.address == address) return &targets[kind][i];
      return nullptr;
    }
    // Caller holds targets_lock.
    bool refused_before(unsigned row, std::uintptr_t key) {
      const auto begin = refused[row].begin(), end = begin + refused_counts[row];
      return std::find(begin, end, key) != end;
    }
    // A coverage change of one family: the state family invalidates the
    // capture owner's evidence; the order family only resets the order model
    // (deferred to its next locked operation, never inside a detour's lock).
    void coverage_changed(unsigned kind) {
      if (order_family(kind)) order::reset_pending.store(true, std::memory_order_release);
      else invalidate();
    }
    void lost(unsigned kind, bool unreadable_input) {
      if (!order_family(kind)) { drop(unreadable_input); return; }
      if (unreadable_input) ++unreadable;
      else ++dropped;
      coverage_changed(kind);
    }
    void report_refusal(unsigned kind = 0) { ++rejected; coverage_changed(kind); }
    // Reports only a new slot refusal; a full table keeps reporting every one.
    // With locked, the caller's exclusive targets_lock is released here.
    void refuse(unsigned kind, std::uintptr_t key, bool locked = false) {
      if (!locked && !TryAcquireSRWLockExclusive(&targets_lock)) { ++discovery_contention; lost(kind, false); return; }
      const bool known = refused_before(kind, key);
      if (!known && refused_counts[kind] != refused_limit) refused[kind][refused_counts[kind]++] = key;
      ReleaseSRWLockExclusive(&targets_lock);
      if (!known) report_refusal(kind);
    }
    bool read_vtable_entry(std::uint64_t object, std::size_t index, void *&entry) {
      std::uintptr_t vtable{};
      const auto offset = index * sizeof(void *);
      return read(reinterpret_cast<void *>(object), &vtable, sizeof(vtable)) && vtable && vtable <= UINTPTR_MAX - offset &&
        read(reinterpret_cast<void *>(vtable + offset), &entry, sizeof(entry)) && entry;
    }
    bool read_slot_address(std::uint64_t object, unsigned kind, void **&address) {
      std::uintptr_t table{};
      const auto offset = vtable_slots[kind] * sizeof(void *);
      if (!read(reinterpret_cast<const void *>(object), &table, sizeof(table)) || !table || table > UINTPTR_MAX - offset) return false;
      address = reinterpret_cast<void **>(table + offset);
      return true;
    }
    template<class Interface> Interface *required_interface(std::uint64_t address, REFIID iid, bool discover) {
      // A live IUnknown may have unrelated methods at the command-list/queue
      // offsets. Never inspect or register those offsets until QI establishes
      // the exact interface, and use the returned pointer (which may differ).
      void *query{};
      std::uintptr_t table{};
      if (!read_vtable_entry(address, 0, query) || !read(reinterpret_cast<const void *>(address), &table, sizeof(table))) {
        if (discover) drop(true);
        return nullptr;
      }
      auto *object = reinterpret_cast<ID3D12Object *>(address);
      Interface *result{};
      const HRESULT status = object->lpVtbl->QueryInterface(object, iid, reinterpret_cast<void **>(&result));
      if (FAILED(status) || !result) {
        if (discover) report_refusal();
        return nullptr;
      }
      return result;
    }
    void observe(std::uint64_t object, method value) {
      if (!requested.load(std::memory_order_acquire) || suppression_depth) return;
      const restore_error incoming{GetLastError()};
      const auto kind = static_cast<unsigned>(value);
      void **address{};
      if (!read_slot_address(object, kind, address)) { lost(kind, true); return; }
      if (!TryAcquireSRWLockShared(&targets_lock)) { ++discovery_contention; lost(kind, false); return; }
      const auto key = reinterpret_cast<std::uintptr_t>(address);
      const bool seen = find(kind, address) != nullptr || refused_before(kind, key);
      ReleaseSRWLockShared(&targets_lock);
      if (seen) return;
      sunshine_native_vtable::slot location;
      if (!sunshine_native_vtable::discover(object, vtable_slots[kind], location)) { refuse(kind, key); return; }
      if (!TryAcquireSRWLockExclusive(&targets_lock)) { sunshine_native_vtable::release(location); ++discovery_contention; lost(kind, false); return; }
      if (find(kind, location.address)) { ReleaseSRWLockExclusive(&targets_lock); sunshine_native_vtable::release(location); return; }
      if (target_counts[kind] == target_limit) {
        refuse(kind, key, true);
        sunshine_native_vtable::release(location);
        return;
      }
      auto &slot = targets[kind][target_counts[kind]++];
      slot.location = location;
      slot.status.store(state::pending, std::memory_order_release);
      ReleaseSRWLockExclusive(&targets_lock);
      coverage_changed(kind);
    }
    bool ready(std::uint64_t object, method value) {
      if (!enabled()) return false;
      const restore_error incoming{GetLastError()};
      const auto kind = static_cast<unsigned>(value);
      void **address{};
      if (!read_slot_address(object, kind, address) || !TryAcquireSRWLockShared(&targets_lock)) return false;
      auto *slot = find(kind, address);
      bool result = false, lost = false;
      if (slot && slot->status.load(std::memory_order_acquire) == state::installed) {
        result = sunshine_native_vtable::matches(slot->location, detours[kind][slot - targets[kind].data()]);
        if (!result) {
          auto expected = state::installed;
          lost = slot->status.compare_exchange_strong(expected, state::rejected, std::memory_order_acq_rel);
        }
      }
      ReleaseSRWLockShared(&targets_lock);
      // Missing observations cannot be recovered by a third party restoring
      // our pointer later. Retire coverage and invalidate the owner's proofs
      // outside the target lock; retained originals remain safe passthroughs.
      if (lost) { ++rejected; coverage_changed(kind); }
      return result;
    }
    // A list that supports CommandList7 can also change states with enhanced
    // Barrier, which makes the legacy states it observed unusable.
    bool enhanced_coverage(std::uint64_t command, bool discover) {
      const restore_error incoming{GetLastError()};
      auto *object = reinterpret_cast<ID3D12Object *>(command);
      ID3D12GraphicsCommandList7 *version7{};
      const HRESULT result7 = object->lpVtbl->QueryInterface(object, IID_ID3D12GraphicsCommandList7, reinterpret_cast<void **>(&version7));
      bool complete = result7 == E_NOINTERFACE;
      if (SUCCEEDED(result7) && version7) {
        const auto address = reinterpret_cast<std::uintptr_t>(version7);
        if (discover) observe(address, method::enhanced);
        complete = ready(address, method::enhanced);
        version7->lpVtbl->Release(version7);
      }
      if (discover && ((FAILED(result7) && result7 != E_NOINTERFACE) || (SUCCEEDED(result7) && !version7))) invalidate();
      return complete;
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
      order::model.reset();
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
    void write_fence(ID3D12Fence *fence, const sunshine_queue_order::fence_memory &value) {
      const order::fence_bytes bytes{value.id, value.signaller, static_cast<std::uint64_t>(value.why)};
      auto *object = reinterpret_cast<ID3D12Object *>(fence);
      object->lpVtbl->SetPrivateData(object, order_fence_guid, sizeof(bytes), &bytes);
    }
    // The fence's record id, sighting it inside a game call on it: its CPU
    // Signal slot must be installed first (no record otherwise: its CPU
    // signals would go unseen), its creation flags and device are checked and
    // its completed value is the baseline. A re-sight keeps what its private
    // data remembers (poison, signaller). lost: the calling queue's record was
    // dropped (evicted or reset) since the call took its ticket.
    std::uint64_t fence_id(ID3D12Fence *fence, std::uint64_t queue, bool *lost = nullptr) {
      auto memory = read_fence(fence);
      if (memory.id) {
        AcquireSRWLockShared(&order::lock);
        const bool known = order::model.known_fence(memory.id);
        ReleaseSRWLockShared(&order::lock);
        if (known) return memory.id;
      }
      const auto kind = static_cast<unsigned>(method::fence_signal);
      void **address{};
      if (!read_slot_address(reinterpret_cast<std::uintptr_t>(fence), kind, address)) return 0;
      bool hooked = false;
      if (TryAcquireSRWLockShared(&targets_lock)) {
        if (auto *slot = find(kind, address))
          hooked = slot->status.load(std::memory_order_acquire) == state::installed &&
            sunshine_native_vtable::matches(slot->location, detours[kind][slot - targets[kind].data()]);
        ReleaseSRWLockShared(&targets_lock);
      }
      if (!hooked) {
        ++order::cpu_unhooked;
        observe(reinterpret_cast<std::uintptr_t>(fence), method::fence_signal);
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
      if (!device || device != order::model.queue_device(queue)) mark(sunshine_queue_order::poison::foreign_device);
      memory.id = order::model.add_fence(device, baseline, {0, memory.signaller, memory.why}, why);
      memory = order::model.memory(memory.id);
      ReleaseSRWLockExclusive(&order::lock);
      write_fence(fence, memory);
      return memory.id;
    }
    template<class Operation> void order_update(ID3D12Fence *fence, std::uint64_t id, Operation operation) {
      sunshine_queue_order::fence_memory before{}, after{};
      AcquireSRWLockExclusive(&order::lock);
      apply_order_reset();
      if (order::model.known_fence(id)) {
        before = order::model.memory(id);
        operation();
        after = order::model.memory(id);
      }
      ReleaseSRWLockExclusive(&order::lock);
      if (after.id && (after.signaller != before.signaller || after.why != before.why)) write_fence(fence, after);
    }
    // Logs one queue Signal. False only with retry, when the call's queue
    // record was dropped (evicted, or a reset) since it took its ticket. A
    // signalling queue without a record poisons a fence already known.
    bool log_signal(const invocation &call, ID3D12Fence *fence, UINT64 value, bool retry) {
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
        order::model.signal(recorded ? call.order_queue : 0, id, value, recorded ? call.ticket : 0, completed, GetTickCount64());
      });
      return !dropped;
    }
    void order_signal(ID3D12CommandQueue *queue, invocation &call, ID3D12Fence *fence, UINT64 value) {
      const restore_error incoming{GetLastError()};
      // The native call is not forwarded yet: a record dropped since the call
      // took its ticket is re-sighted with a fresh ticket, so a signal never
      // goes unlogged against a live fence record (a pending wait would
      // resolve to a later signal of its queue, after work this one precedes).
      if (log_signal(call, fence, value, true)) return;
      call.order_queue = call.ticket = 0;
      order_enter(call, queue, true);
      log_signal(call, fence, value, false);
    }
    void order_wait(const invocation &call, ID3D12Fence *fence, UINT64 value, bool quiet) {
      const restore_error incoming{GetLastError()};
      const auto id = fence_id(fence, call.order_queue);
      const auto completed = fence->lpVtbl->GetCompletedValue(fence);
      AcquireSRWLockExclusive(&order::lock);
      apply_order_reset();
      // An unknown fence still logs the wait (refused as untracked).
      sunshine_queue_order::fence_memory before{}, after{};
      if (id && order::model.known_fence(id)) before = order::model.memory(id);
      order::model.wait(call.order_queue, id, value, call.ticket, quiet, completed, GetTickCount64());
      if (id && order::model.known_fence(id)) after = order::model.memory(id);
      ReleaseSRWLockExclusive(&order::lock);
      if (after.id && (after.signaller != before.signaller || after.why != before.why)) write_fence(fence, after);
    }
    void order_cpu_signal(ID3D12Fence *fence) {
      const restore_error incoming{GetLastError()};
      auto memory = read_fence(fence);
      if (!memory.id) return;  // Never sighted: its baseline at sight covers this signal.
      bool known = false;
      order_update(fence, memory.id, [&] { known = true; order::model.cpu_signal(memory.id, GetTickCount64()); });
      if (!known && memory.why == sunshine_queue_order::poison::none) {
        memory.why = sunshine_queue_order::poison::cpu_signal;
        write_fence(fence, memory);
      }
    }
    // Whether a queue's Execute, Signal and Wait slots are all installed.
    // Requires targets_lock shared.
    bool covered(std::uint64_t id) {
      const auto *slot = ticket_of(id);
      if (!slot) return false;
      const unsigned kinds[3]{static_cast<unsigned>(method::execute), static_cast<unsigned>(method::signal), static_cast<unsigned>(method::wait)};
      for (unsigned i = 0; i != 3; ++i) {
        const auto *target = slot->slots[i] ? find(kinds[i], slot->slots[i]) : nullptr;
        if (!target || target->status.load(std::memory_order_acquire) != state::installed) return false;
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
    // Installed order slots a third party replaced since: their coverage is
    // lost for good (as ready() retires a state slot).
    void verify_order_slots() {
      bool lost_any = false;
      if (!TryAcquireSRWLockShared(&targets_lock)) return;
      for (unsigned kind = 0; kind != method_count; ++kind)
        if (order_family(kind))
          for (unsigned i = 0; i != target_counts[kind]; ++i) {
            auto &slot = targets[kind][i];
            auto expected = state::installed;
            if (slot.status.load(std::memory_order_acquire) == state::installed &&
                !sunshine_native_vtable::matches(slot.location, detours[kind][i]) &&
                slot.status.compare_exchange_strong(expected, state::rejected, std::memory_order_acq_rel))
              lost_any = true;
          }
      ReleaseSRWLockShared(&targets_lock);
      if (lost_any) { ++rejected; coverage_changed(static_cast<unsigned>(method::signal)); }
    }
  }

  void initialize(callbacks value) {
    if (sunshine_addon_lifetime::stopping()) return;
    active.store(false, std::memory_order_release);
    ++activation_epoch;
    barrier_callback.store(value.barriers, std::memory_order_release);
    submitted_callback.store(value.submitted, std::memory_order_release);
    invalidated_callback.store(value.invalidated, std::memory_order_release);
    enhanced_textures_callback.store(value.enhanced_textures, std::memory_order_release);
    associate_recording_callback.store(value.associate_recording, std::memory_order_release);
    requested.store(true, std::memory_order_release);
  }
  void set_active(bool value) {
    value = value && !sunshine_addon_lifetime::stopping();
    if (active.load(std::memory_order_acquire) != value) {
      ++activation_epoch;
      active.store(value, std::memory_order_release);
    }
  }
  void shutdown() {
    active.store(false, std::memory_order_release);
    requested.store(false, std::memory_order_release);
    ++activation_epoch;
  }
  void set_submission_listener(submission_listener value) {
    listener_callback.store(value, std::memory_order_release);
  }
  void observe_command(std::uint64_t command) {
    if (sunshine_addon_lifetime::stopping() || !requested.load(std::memory_order_acquire) || suppression_depth) return;
    const restore_error incoming{GetLastError()};
    auto *object = required_interface<ID3D12GraphicsCommandList>(command, IID_ID3D12GraphicsCommandList, true);
    if (!object) return;
    const auto normalized = reinterpret_cast<std::uintptr_t>(object);
    observe(normalized, method::barriers);
    enhanced_coverage(normalized, true);
    object->lpVtbl->Release(object);
  }
  void observe_queue(std::uint64_t queue) {
    if (sunshine_addon_lifetime::stopping() || !requested.load(std::memory_order_acquire) || suppression_depth) return;
    const restore_error incoming{GetLastError()};
    auto *object = required_interface<ID3D12CommandQueue>(queue, IID_ID3D12CommandQueue, true);
    if (!object) return;
    observe(reinterpret_cast<std::uintptr_t>(object), method::execute);
    if (order::requested.load(std::memory_order_acquire)) {
      observe(reinterpret_cast<std::uintptr_t>(object), method::signal);
      observe(reinterpret_cast<std::uintptr_t>(object), method::wait);
    }
    object->lpVtbl->Release(object);
  }
  bool states_ready(std::uint64_t command) {
    if (!enabled()) return false;
    const restore_error incoming{GetLastError()};
    auto *object = required_interface<ID3D12GraphicsCommandList>(command, IID_ID3D12GraphicsCommandList, false);
    if (!object) return false;
    const auto normalized = reinterpret_cast<std::uintptr_t>(object);
    const bool result = ready(normalized, method::barriers) && enhanced_coverage(normalized, false);
    object->lpVtbl->Release(object);
    return result;
  }
  bool queue_ready(std::uint64_t queue) {
    if (!enabled()) return false;
    const restore_error incoming{GetLastError()};
    auto *object = required_interface<ID3D12CommandQueue>(queue, IID_ID3D12CommandQueue, false);
    if (!object) return false;
    const bool result = ready(reinterpret_cast<std::uintptr_t>(object), method::execute);
    object->lpVtbl->Release(object);
    return result;
  }
  bool install_pending() {
    if (!enabled()) return false;
    const restore_error incoming{GetLastError()};
    // Caller serializes deferred installation outside event/loader callbacks.
    // The shared helper serializes page-protection changes across observers.
    verify_order_slots();
    std::array<std::array<unsigned, 2>, method_count * target_limit> pending{};
    unsigned count{}, state_count{}, order_count{};
    if (!TryAcquireSRWLockShared(&targets_lock)) { ++discovery_contention; drop(false); return false; }
    for (unsigned kind = 0; kind != method_count; ++kind)
      for (unsigned i = 0; i != target_counts[kind]; ++i)
        if (targets[kind][i].status.load(std::memory_order_acquire) == state::pending) {
          pending[count++] = {kind, i};
          ++(order_family(kind) ? order_count : state_count);
        }
    ReleaseSRWLockShared(&targets_lock);
    const auto changed = [&] {
      if (state_count) invalidate();
      if (order_count) coverage_changed(static_cast<unsigned>(method::signal));
    };
    changed();
    bool okay = true;
    for (unsigned n = 0; n != count; ++n) {
      const auto kind = pending[n][0], index = pending[n][1];
      auto &slot = targets[kind][index];
      // A thread can enter the detour immediately after the slot CAS, before
      // status becomes installed. Its callable original must already exist.
      slot.original.store(slot.location.original, std::memory_order_release);
      if (!sunshine_native_vtable::install(slot.location, detours[kind][index])) {
        slot.status.store(state::rejected, std::memory_order_release); ++rejected; okay = false; continue;
      }
      slot.status.store(state::installed, std::memory_order_release);
      ++installed;
    }
    changed();
    return okay;
  }
  bool set_recording_cookie(std::uint64_t command, std::uint64_t cookie) {
    if (sunshine_addon_lifetime::stopping() || !command) return false;
    const restore_error incoming{GetLastError()};
    auto *object = reinterpret_cast<ID3D12Object *>(command);
    return SUCCEEDED(object->lpVtbl->SetPrivateData(object, recording_guid, sizeof(cookie), &cookie));
  }
  std::uint64_t get_recording_cookie(std::uint64_t command) {
    if (sunshine_addon_lifetime::stopping() || !command) return 0;
    const restore_error incoming{GetLastError()};
    auto *object = reinterpret_cast<ID3D12Object *>(command);
    UINT size = sizeof(std::uint64_t);
    std::uint64_t cookie{};
    return SUCCEEDED(object->lpVtbl->GetPrivateData(object, recording_guid, &size, &cookie)) && size == sizeof(cookie) ? cookie : 0;
  }
  bool recording_cookie_absent(std::uint64_t command) {
    if (sunshine_addon_lifetime::stopping() || !command) return false;
    const restore_error incoming{GetLastError()};
    auto *object = reinterpret_cast<ID3D12Object *>(command);
    UINT size = sizeof(std::uint64_t);
    std::uint64_t cookie{};
    return object->lpVtbl->GetPrivateData(object, recording_guid, &size, &cookie) == DXGI_ERROR_NOT_FOUND;
  }
  counters counts() {
    counters result{calls.load(), observed.load(), unreadable.load(), dropped.load(), 0, installed.load(), rejected.load(), nested.load(), suppressed.load(),
      barrier_overflow.load(), submission_overflow.load(), discovery_contention.load()};
    if (TryAcquireSRWLockShared(&targets_lock)) {
      for (const auto count : target_counts) result.targets += count;
      ReleaseSRWLockShared(&targets_lock);
    }
    return result;
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
    AcquireSRWLockShared(&order::lock);
    out = order::model.ordered(memory.id, producer);
    ReleaseSRWLockShared(&order::lock);
    if (out.ordered) ++order::ordered_count;
    else ++order::refused[static_cast<unsigned>(out.why)];
    return out;
  }
  void queue_order_rely(const sunshine_queue_order::verdict &value) {
    if (!value.ordered) return;
    AcquireSRWLockExclusive(&order::lock);
    order::model.rely(value, GetTickCount64());
    ReleaseSRWLockExclusive(&order::lock);
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
    AcquireSRWLockShared(&order::lock);
    out.model = order::model.snapshot();
    ReleaseSRWLockShared(&order::lock);
    out.ordered = order::ordered_count.load(std::memory_order_relaxed);
    out.resets = order::resets.load(std::memory_order_relaxed);
    out.cpu_unhooked = order::cpu_unhooked.load(std::memory_order_relaxed);
    for (unsigned i = 0; i != order::refused.size(); ++i) out.refused[i] = order::refused[i].load(std::memory_order_relaxed);
    return out;
  }
  const hooked_method *hooked_methods(unsigned &count) {
    count = method_count;
    return hooked_table;
  }
  suppression_scope::suppression_scope() { ++suppression_depth; }
  suppression_scope::~suppression_scope() { --suppression_depth; }

#ifdef SUNSHINE_STREAMLINE_NATIVE_OBSERVER_TEST
  void reset_after_hooks_removed() {
    shutdown();
    bool restored = true;
    for (unsigned kind = 0; kind != method_count; ++kind)
      for (unsigned i = 0; i != target_counts[kind]; ++i) {
        const auto &location = targets[kind][i].location;
        if (sunshine_native_vtable::matches(location, detours[kind][i]) &&
            !sunshine_native_vtable::restore(location, detours[kind][i])) restored = false;
      }
    // Preserve callable originals if a fixture's protection restore failed.
    if (!restored) return;
    for (unsigned kind = 0; kind != method_count; ++kind) {
      for (unsigned i = 0; i != target_counts[kind]; ++i) {
        auto &slot = targets[kind][i];
        sunshine_native_vtable::release(slot.location);
        slot.original = nullptr; slot.status = state::empty;
      }
      target_counts[kind] = 0;
    }
    refused_counts = {};
    order::model.reset();
    for (auto &slot : order::tickets) { slot.id = 0; slot.entered = 0; slot.slots = {}; }
    order::requested = true; order::reset_pending = false;
    order::resets = order::ordered_count = order::cpu_unhooked = order::contention = 0;
    for (auto &value : order::refused) value = 0;
    calls = observed = unreadable = dropped = installed = rejected = nested = suppressed = 0;
    barrier_overflow = submission_overflow = discovery_contention = 0;
    barrier_callback = nullptr; submitted_callback = nullptr; invalidated_callback = nullptr;
    enhanced_textures_callback = nullptr; associate_recording_callback = nullptr; listener_callback = nullptr;
  }
#endif
}
