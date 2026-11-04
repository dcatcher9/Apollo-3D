// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_native_observer.h"
#include "addon_lifetime.h"
#define CINTERFACE
#include <d3d12.h>
#include "native_vtable_slot.h"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <new>
#include "game3d_clock.h"

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
    // A fence ReShade signalled in the add-on's flush of its immediate list
    // (own_flush_scope): ReShade's own, which no game queue waits on.
    constexpr GUID own_fence_guid{0x52a0c4e7, 0x93d1, 0x4b6f, {0x8e, 0x15, 0x2c, 0x71, 0xa9, 0x0d, 0x36, 0xf8}};
    // A list whose own (object-owned) method table the execution hook wrote a
    // hook into: the recording cookie it held right after the write
    // (states_dispatch::after). Plain bytes, as the recording cookie.
    constexpr GUID dispatch_guid{0x1f6b9e3d, 0x47a2, 0x4c58, {0xb0, 0x6e, 0x93, 0x2d, 0x7c, 0x15, 0xa8, 0x4f}};
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
    std::atomic<std::uint64_t> barrier_overflow{}, submission_overflow{}, discovery_contention{}, written{};
    // An invalidation raised inside a suppressed callback (a coverage change
    // found there, where notify drops every notification): delivered at the
    // next install_pending, so no loss is ever dropped.
    std::atomic<bool> invalidation_pending{};
    // enhanced_barrier_use's evidence, sticky: a Barrier call entered a
    // Barrier hook; an executed list's recording may have recorded barriers
    // past the hooks.
    std::atomic<bool> enhanced_used{}, barrier_blind{};
    thread_local unsigned suppression_depth{};
    thread_local unsigned own_flush_depth{};
    struct invocation;
    thread_local std::array<invocation *, method_count> current_invocations{};

    // Step 4 state, deleted with native_queue_order.inl.
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
        // The queue's stable cookie and its D3D12_COMMAND_LIST_TYPE + 1
        // (read at its sighting), both written with slots: the
        // second-signaller diagnostic names both.
        std::uint64_t cookie{};
        std::atomic<std::uint8_t> type{};
      };
      std::array<ticket_slot, queue_limit> tickets;
      SRWLOCK lock = SRWLOCK_INIT;
      model_type model;
      std::atomic<bool> requested{true}, reset_pending{};
      std::atomic<std::uint64_t> queue_cookies{}, resets{}, ordered_count{}, cpu_unhooked{}, contention{}, own_signals{};
      // The newest tag queue_order_conflicted found conflicted: its answer
      // when the lock is held elsewhere (the present thread never waits).
      std::atomic<std::uint64_t> conflicted_seen{};
      std::array<std::atomic<std::uint64_t>, static_cast<unsigned>(sunshine_queue_order::refusal::count)> refused{};
      thread_local position last_execute{};
      struct queue_memory { std::uint64_t cookie, id; };
      struct fence_bytes { std::uint64_t id, signaller, why; };
      // The second-signaller diagnostic (queue_order_signaller_line): a queue
      // whose Signal poisoned a fence as its second signaller is reported
      // once (by its cookie), at most report_limit queues per process; the
      // reports are formatted on the signalling thread and handed to the
      // present thread's status step one at a time.
      constexpr unsigned report_limit = 8;
      constexpr std::size_t report_size = 512;
      std::array<std::uint64_t, report_limit> reported{};
      std::array<std::array<char, report_size>, report_limit> reports{};
      unsigned report_count{}, report_read{};
      SRWLOCK report_lock = SRWLOCK_INIT;
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
    void invalidate() noexcept {
      const auto callback = invalidated_callback.load(std::memory_order_acquire);
      if (callback && suppression_depth && enabled()) { invalidation_pending.store(true, std::memory_order_release); return; }
      notify(callback);
    }
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
    bool cover(std::uint64_t command, std::uint64_t cookie);
    bool cookieless_covered(std::uint64_t command);
    bool order_on();
    bool own_signal(ID3D12Fence *fence);
    void order_enter(invocation &call, ID3D12CommandQueue *queue, bool blocking);
    bool order_quiet(const invocation &call);
    void order_signal(ID3D12CommandQueue *queue, invocation &call, ID3D12Fence *fence, UINT64 value, void *caller);
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
      // Each executed list a recording cookie names, in the table it now
      // dispatches through: a module slot first seen is discovered, and an
      // object-owned table gets the hook (cover), before the owner judges
      // this execution (which began before any write here) from the verdict
      // each identity now carries (command_identity::covered). A list whose
      // recording the barrier hooks may have missed part of leaves the
      // process's enhanced-barrier use unknown (enhanced_barrier_use).
      for (UINT i = 0; i != count; ++i) {
        auto &identity = identities.data()[i];
        identity.covered = identity.recording_cookie ? cover(identity.native_command, identity.recording_cookie) :
          cookieless_covered(identity.native_command);
        if (!identity.covered && !barrier_blind.load(std::memory_order_relaxed)) barrier_blind.store(true, std::memory_order_relaxed);
      }
      notify(submitted_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(queue), count, identities.data());
      listen(reinterpret_cast<std::uintptr_t>(queue), count, identities.data());
    }
    template<unsigned Index> void STDMETHODCALLTYPE enhanced_detour(ID3D12GraphicsCommandList7 *command, UINT32 count, const D3D12_BARRIER_GROUP *groups) {
      const DWORD incoming = GetLastError();
      // Any call, whatever it names (enhanced_barrier_use).
      if (!enhanced_used.load(std::memory_order_relaxed)) enhanced_used.store(true, std::memory_order_relaxed);
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
              static_cast<std::uint32_t>(barrier.SyncAfter),
              barrier.LayoutBefore == D3D12_BARRIER_LAYOUT_UNDEFINED || (barrier.Flags & D3D12_TEXTURE_BARRIER_FLAG_DISCARD) != 0};
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
    template<unsigned Index> HRESULT STDMETHODCALLTYPE signal_detour(ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value);
    template<unsigned Index> HRESULT STDMETHODCALLTYPE wait_detour(ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value);
    template<unsigned Index> HRESULT STDMETHODCALLTYPE fence_signal_detour(ID3D12Fence *fence, UINT64 value);
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
#ifdef SUNSHINE_STREAMLINE_NATIVE_OBSERVER_TEST
    std::atomic<std::uint64_t> slot_reads{};
    std::atomic<std::uintptr_t> test_own_object{}, test_own_table{};
#endif
    bool read_slot_address(std::uint64_t object, unsigned kind, void **&address) {
#ifdef SUNSHINE_STREAMLINE_NATIVE_OBSERVER_TEST
      ++slot_reads;
#endif
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
    // The exact interface of a live list the caller holds a reference to (an
    // executed list inside its ExecuteCommandLists, ReShade's native list at
    // a list event or a call recorded on it): its QueryInterface is called
    // directly, without required_interface's checked reads (a system call
    // each, three per list).
    template<class Interface> Interface *live_interface(std::uint64_t address, REFIID iid) {
      auto *object = reinterpret_cast<ID3D12Object *>(address);
      Interface *result{};
      if (FAILED(object->lpVtbl->QueryInterface(object, iid, reinterpret_cast<void **>(&result)))) return nullptr;
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
      // A table outside every module image (a runtime's per-object table,
      // one per list) is no slot to share or remember: the execution hook
      // covers it (cover). Refusing it would revoke evidence for every list.
      if (kind == static_cast<unsigned>(method::barriers) || kind == static_cast<unsigned>(method::enhanced))
        if (!sunshine_native_vtable::detail::image(address, false)) return;
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

    // -- Dispatch coverage (states_coverage): judged by the value each barrier
    // slot of the list's current method table holds, so a list a runtime
    // moved to another table (its own, or another module table) is judged by
    // the table it uses. Lock-free: detours never change, and a hook's
    // location and original are written before its status publishes it.
    int hook_index(unsigned kind, const void *value) {
      for (unsigned i = 0; i != target_limit; ++i)
        if (detours[kind][i] == value && targets[kind][i].status.load(std::memory_order_acquire) == state::installed)
          return static_cast<int>(i);
      return -1;
    }
    bool our_detour(const void *value) {
      for (const auto &row : detours)
        for (const auto *entry : row)
          if (entry == value) return true;
      return false;
    }
    // The slot of kind's method in the live object's current table, and what
    // it holds: plain reads (no system call per executed list), valid because
    // the caller holds a reference to the object through the exact interface
    // QueryInterface returned, whose table has that method.
    void **live_slot(std::uint64_t object, unsigned kind, void *&current) {
      const auto table = *reinterpret_cast<void **const volatile *>(object);
      if (!table) return nullptr;
      auto **address = table + vtable_slots[kind];
      current = *reinterpret_cast<void *const volatile *>(address);
      return address;
    }
    dispatch combine(dispatch a, dispatch b) {
      if (a == dispatch::none || b == dispatch::none) return dispatch::none;
      return a == dispatch::patched || b == dispatch::patched ? dispatch::patched : dispatch::hooked;
    }
    // Whether kind's slot at address lies in the object's own method table:
    // a table inside the object itself, within its first 64 KiB (D3D12Core
    // 1.619 builds it at object+0x5c8). No other list shares such a table,
    // so the list's dispatch stamp speaks for every write into it. A table
    // elsewhere outside every image (a hooking library's shadow table, a
    // heap table several lists share) may be rewritten by another list's
    // execution or restored by a third party during this list's recording,
    // which no stamp of this list would show: it is never written and never
    // covers (none).
    constexpr std::uintptr_t own_table_extent = 0x10000;
    bool owned_table(std::uint64_t object, unsigned kind, void **address) {
      const auto slot = reinterpret_cast<std::uintptr_t>(address);
      const auto table = slot - vtable_slots[kind] * sizeof(void *);
#ifdef SUNSHINE_STREAMLINE_NATIVE_OBSERVER_TEST
      if (object && object == test_own_object.load() && table == test_own_table.load()) return true;
#endif
      return table > object && slot < object + own_table_extent;
    }
    // A registered, installed module slot at address that no longer holds
    // its hook is retired for good (missed calls cannot be recovered by a
    // later restore) and invalidates evidence; inside a suppressed callback
    // that invalidation is delivered at the next install_pending.
    void retire_lost(unsigned kind, void **address) {
      for (auto &slot : targets[kind]) {
        if (slot.status.load(std::memory_order_acquire) != state::installed || slot.location.address != address) continue;
        auto expected = state::installed;
        if (slot.status.compare_exchange_strong(expected, state::rejected, std::memory_order_acq_rel)) { ++rejected; coverage_changed(kind); }
        break;
      }
    }
    // How one barrier slot of object (kind's interface) dispatches now.
    dispatch slot_dispatch(std::uint64_t object, unsigned kind) {
      void *current{};
      auto **const address = live_slot(object, kind, current);
      if (!address) return dispatch::none;
      const int index = hook_index(kind, current);
      if (index >= 0)
        return targets[kind][index].location.address == address ? dispatch::hooked :
          owned_table(object, kind, address) ? dispatch::patched : dispatch::none;
      retire_lost(kind, address);
      return dispatch::none;
    }
    // The CommandList7 Barrier slot where the object supports it; an object
    // without CommandList7 has no such method (hooked: neutral in combine).
    dispatch enhanced_dispatch(std::uint64_t command) {
      auto *object = reinterpret_cast<ID3D12Object *>(command);
      ID3D12GraphicsCommandList7 *version7{};
      const HRESULT result7 = object->lpVtbl->QueryInterface(object, IID_ID3D12GraphicsCommandList7, reinterpret_cast<void **>(&version7));
      if (result7 == E_NOINTERFACE) return dispatch::hooked;
      if (FAILED(result7) || !version7) return dispatch::none;
      const auto value = slot_dispatch(reinterpret_cast<std::uintptr_t>(version7), static_cast<unsigned>(method::enhanced));
      version7->lpVtbl->Release(version7);
      return value;
    }
    bool dispatch_stamp(std::uint64_t command, std::uint64_t &after) {
      auto *object = reinterpret_cast<ID3D12Object *>(command);
      UINT size = sizeof(after);
      return SUCCEEDED(object->lpVtbl->GetPrivateData(object, dispatch_guid, &size, &after)) && size == sizeof(after);
    }
    // The hook that forwards to value (a function in a loaded module image):
    // an installed hook whose original it is (a module slot's, whose table the
    // runtime copied), else a new dispatch-only hook of its own (no slot; its
    // code module pinned for process life), at most target_limit a method.
    int dispatch_target(unsigned kind, void *value) {
      for (unsigned i = 0; i != target_limit; ++i)
        if (targets[kind][i].status.load(std::memory_order_acquire) == state::installed &&
            targets[kind][i].original.load(std::memory_order_acquire) == value) return static_cast<int>(i);
      if (!TryAcquireSRWLockExclusive(&targets_lock)) { ++discovery_contention; return -1; }
      int index = -1;
      for (unsigned i = 0; i != target_counts[kind] && index < 0; ++i)
        if (targets[kind][i].status.load(std::memory_order_acquire) == state::installed &&
            targets[kind][i].original.load(std::memory_order_acquire) == value) index = static_cast<int>(i);
      if (index < 0 && target_counts[kind] != target_limit) {
        const auto next = target_counts[kind];
        HMODULE code{};
        if (sunshine_native_vtable::detail::retain(value, code) && sunshine_native_vtable::detail::pin(value, code) &&
            sunshine_native_vtable::detail::pin(detours[kind][next])) {
          auto &slot = targets[kind][next];
          slot.location = {};
          slot.location.original = value; slot.location.code_owner = code;
          slot.original.store(value, std::memory_order_release);
          slot.status.store(state::installed, std::memory_order_release);
          target_counts[kind] = next + 1;
          ++installed;
          index = static_cast<int>(next);
        } else if (code) FreeLibrary(code);
      }
      ReleaseSRWLockExclusive(&targets_lock);
      return index;
    }
    // The index of kind's detour value whose slot is no longer installed (a
    // module slot retired once a third party replaced its hook), or -1.
    int retired_index(unsigned kind, const void *value) {
      for (unsigned i = 0; i != target_limit; ++i)
        if (detours[kind][i] == value && targets[kind][i].status.load(std::memory_order_acquire) != state::installed)
          return static_cast<int>(i);
      return -1;
    }
    // Write a hook into the list's own table slot (owned_table) that
    // dispatches to current: only a committed plain read-write private page
    // (a runtime's per-object table; never a page whose protection would have
    // to change), and only to a function in a loaded image that is not ours.
    // A detour of ours left there by a slot since retired (a module slot whose
    // hook a third party replaced, whose index this table reused) forwards to
    // that slot's original but is no longer eligible: it is replaced by a hook
    // that forwards to the same original (a fresh write, stamped as one).
    bool write_hook(unsigned kind, void **address, void *current) {
      MEMORY_BASIC_INFORMATION memory{};
      if (VirtualQuery(address, &memory, sizeof(memory)) != sizeof(memory) || memory.State != MEM_COMMIT ||
          memory.Type != MEM_PRIVATE || memory.Protect != PAGE_READWRITE) return false;
      void *forward = current;
      if (our_detour(current)) {
        const int retired = retired_index(kind, current);
        forward = retired < 0 ? nullptr : targets[kind][retired].original.load(std::memory_order_acquire);
        if (!forward) return false;
      }
      if (!sunshine_native_vtable::detail::image(forward, true)) return false;
      const int index = dispatch_target(kind, forward);
      if (index < 0) return false;
      if (InterlockedCompareExchangePointer(reinterpret_cast<void *volatile *>(address), detours[kind][index], current) != current)
        return false;
      ++written;
      return true;
    }
    struct slot_cover {
      dispatch kind{dispatch::none};
      bool wrote{};
    };
    slot_cover cover_slot(std::uint64_t object, method value) {
      const auto kind = static_cast<unsigned>(value);
      void *current{};
      auto **const address = live_slot(object, kind, current);
      if (!address) return {};
      const int index = hook_index(kind, current);
      if (index >= 0)
        return {targets[kind][index].location.address == address ? dispatch::hooked :
          owned_table(object, kind, address) ? dispatch::patched : dispatch::none};
      // A module table: a registered slot that lost its hook is retired (as
      // slot_dispatch does; the owner's proof takes this verdict); one first
      // seen here is discovered, installed by the next install_pending (which
      // invalidates the recordings it straddles).
      if (sunshine_native_vtable::detail::image(address, false)) {
        retire_lost(kind, address);
        observe(object, value);
        return {};
      }
      // Any other table that is not the list's own is never written.
      if (!owned_table(object, kind, address) || !write_hook(kind, address, current)) return {};
      return {dispatch::patched, true};
    }
    // An executed list (cover in execute_detour): the slots of the table it
    // dispatches through now. A list patched now, or patched without a stamp,
    // is stamped with the recording cookie it holds after every write: only
    // a recording that begins later (a greater cookie) is covered. Whether
    // the hooks saw every barrier of the executed recording (cookie) before
    // any write here: its slots hold module hooks, or hooks written into its
    // own table before that recording began (enhanced_barrier_use).
    bool cover(std::uint64_t command, std::uint64_t cookie) {
      const restore_error incoming{GetLastError()};
      auto *object = live_interface<ID3D12GraphicsCommandList>(command, IID_ID3D12GraphicsCommandList);
      if (!object) return false;
      const auto normalized = reinterpret_cast<std::uintptr_t>(object);
      const auto barriers = cover_slot(normalized, method::barriers);
      slot_cover enhanced{dispatch::hooked};
      auto *raw = reinterpret_cast<ID3D12Object *>(normalized);
      ID3D12GraphicsCommandList7 *version7{};
      const HRESULT result7 = raw->lpVtbl->QueryInterface(raw, IID_ID3D12GraphicsCommandList7, reinterpret_cast<void **>(&version7));
      if (SUCCEEDED(result7) && version7) {
        enhanced = cover_slot(reinterpret_cast<std::uintptr_t>(version7), method::enhanced);
        version7->lpVtbl->Release(version7);
      } else if (result7 != E_NOINTERFACE) enhanced = {};
      std::uint64_t after{};
      const auto kind = combine(barriers.kind, enhanced.kind);
      bool covered = kind == dispatch::hooked;
      if (kind == dispatch::patched) {
        const bool stamped = !barriers.wrote && !enhanced.wrote && dispatch_stamp(normalized, after);
        covered = stamped && cookie > after;
        if (!stamped) {
          after = get_recording_cookie(normalized);
          if (after) object->lpVtbl->SetPrivateData(object, dispatch_guid, sizeof(after), &after);
        }
      }
      object->lpVtbl->Release(object);
      return covered;
    }
    // An executed list without a recording cookie made no call through a
    // hook (the first would have given it one): its barriers were all seen
    // only where installed module hooks cover its table (as apply_executed's
    // exemption of DXGI's internal Present list requires).
    bool cookieless_covered(std::uint64_t command) {
      const restore_error incoming{GetLastError()};
      auto *object = live_interface<ID3D12GraphicsCommandList>(command, IID_ID3D12GraphicsCommandList);
      if (!object) return false;
      const auto normalized = reinterpret_cast<std::uintptr_t>(object);
      const bool covered = combine(slot_dispatch(normalized, static_cast<unsigned>(method::barriers)), enhanced_dispatch(normalized)) ==
        dispatch::hooked;
      object->lpVtbl->Release(object);
      return covered;
    }

    // The uncovered-dispatch diagnostic (note_uncovered, coverage_line).
    namespace uncovered {
      constexpr unsigned limit = 8;
      struct report {
        std::uintptr_t key{}, object{}, table{};
        bool image{}, version7{};
        void *barrier{}, *enhanced{};
      };
      std::array<report, limit> reports{};
      unsigned count{}, read{};
      // coverage_line's count of written hooks: the last logged, and when.
      std::uint64_t logged_written{}, logged_at{};
      SRWLOCK lock = SRWLOCK_INIT;
    }
    // The module and offset of address ("none" outside every module).
    void describe(const void *address, char *out, std::size_t size) {
      HMODULE module{};
      char path[MAX_PATH]{};
      if (!address || !GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCSTR>(address), &module) || !GetModuleFileNameA(module, path, MAX_PATH)) {
        std::snprintf(out, size, "%p(none)", address);
        return;
      }
      const char *name = path;
      for (const char *at = path; *at; ++at)
        if (*at == '\\' || *at == '/') name = at + 1;
      std::snprintf(out, size, "%p(%s+0x%llx)", address, name,
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(address) - reinterpret_cast<std::uintptr_t>(module)));
    }

    bool order_on();
    order::ticket_slot *ticket_of(std::uint64_t id);
    void apply_order_reset();
    bool private_bytes(void *object, const GUID &guid, void *output, UINT size);
    std::uint64_t device_of(ID3D12DeviceChild *child);
    std::uint64_t queue_id(ID3D12CommandQueue *queue, bool blocking);
    void order_enter(invocation &call, ID3D12CommandQueue *queue, bool blocking);
    bool order_quiet(const invocation &call);
    sunshine_queue_order::fence_memory read_fence(ID3D12Fence *fence);
    void write_fence(ID3D12Fence *fence, const sunshine_queue_order::fence_memory &value);
    bool own_signal(ID3D12Fence *fence);
    std::uint64_t fence_id(ID3D12Fence *fence, std::uint64_t queue, bool *lost = nullptr);
    template<class Operation> void order_update(ID3D12Fence *fence, std::uint64_t id, Operation operation);
    struct second_signal;
    bool log_signal(const invocation &call, ID3D12Fence *fence, UINT64 value, bool retry, second_signal &poisoned);
    void order_signal(ID3D12CommandQueue *queue, invocation &call, ID3D12Fence *fence, UINT64 value, void *caller);
    void report_second_signaller(ID3D12CommandQueue *queue, const second_signal &poisoned, UINT64 value, void *caller);
    void order_wait(const invocation &call, ID3D12Fence *fence, UINT64 value, bool quiet);
    void order_cpu_signal(ID3D12Fence *fence);
    bool holds(const target &slot, unsigned kind);
    bool covered(std::uint64_t id);
    unsigned order_pending();
    // Installed order slots a third party replaced since: their coverage is
    // lost for good (as ready() retires a state slot). Run by each poll and
    // by each verdict, so no verdict rests on a slot replaced before it.
    enum class slot_check { verified, busy, lost };
    slot_check verify_order_slots();
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
  states_dispatch states_coverage(std::uint64_t command) {
    if (!enabled() || !command) return {};
    const restore_error incoming{GetLastError()};
    auto *object = live_interface<ID3D12GraphicsCommandList>(command, IID_ID3D12GraphicsCommandList);
    if (!object) return {};
    const auto normalized = reinterpret_cast<std::uintptr_t>(object);
    states_dispatch result{slot_dispatch(normalized, static_cast<unsigned>(method::barriers))};
    if (result.kind != dispatch::none) result.kind = combine(result.kind, enhanced_dispatch(normalized));
    if (result.kind == dispatch::patched && !dispatch_stamp(normalized, result.after)) result = {};
    object->lpVtbl->Release(object);
    return result;
  }
  bool states_ready(std::uint64_t command) { return states_coverage(command).kind == dispatch::hooked; }
  enhanced_use enhanced_barrier_use() {
    if (enhanced_used.load(std::memory_order_relaxed)) return enhanced_use::used;
    if (!enabled() || barrier_blind.load(std::memory_order_relaxed)) return enhanced_use::unknown;
    return enhanced_use::unused;
  }
  void note_dispatch_gap() {
    if (!barrier_blind.load(std::memory_order_relaxed)) barrier_blind.store(true, std::memory_order_relaxed);
  }
  void note_uncovered(std::uint64_t command) {
    if (sunshine_addon_lifetime::stopping() || !command) return;
    const restore_error incoming{GetLastError()};
    auto *object = required_interface<ID3D12GraphicsCommandList>(command, IID_ID3D12GraphicsCommandList, false);
    if (!object) return;
    uncovered::report value;
    value.object = reinterpret_cast<std::uintptr_t>(object);
    void **slot{};
    if (read(object, &value.table, sizeof(value.table)) && read_slot_address(value.object, static_cast<unsigned>(method::barriers), slot))
      read(slot, &value.barrier, sizeof(value.barrier));
    value.image = value.table && sunshine_native_vtable::detail::image(reinterpret_cast<const void *>(value.table), false);
    auto *raw = reinterpret_cast<ID3D12Object *>(object);
    ID3D12GraphicsCommandList7 *version7{};
    if (SUCCEEDED(raw->lpVtbl->QueryInterface(raw, IID_ID3D12GraphicsCommandList7, reinterpret_cast<void **>(&version7))) && version7) {
      value.version7 = true;
      void **enhanced{};
      if (read_slot_address(reinterpret_cast<std::uintptr_t>(version7), static_cast<unsigned>(method::enhanced), enhanced))
        read(enhanced, &value.enhanced, sizeof(value.enhanced));
      version7->lpVtbl->Release(version7);
    }
    object->lpVtbl->Release(object);
    // Every object-owned table is one class (one table per list).
    value.key = value.image ? value.table : 0;
    if (!TryAcquireSRWLockExclusive(&uncovered::lock)) return;
    bool known = false;
    for (unsigned i = 0; i != uncovered::count && !known; ++i) known = uncovered::reports[i].key == value.key;
    if (!known && uncovered::count != uncovered::limit) uncovered::reports[uncovered::count++] = value;
    ReleaseSRWLockExclusive(&uncovered::lock);
  }
  bool coverage_line(std::uint64_t now_ms, char *out, std::size_t size) {
    if (!size || !TryAcquireSRWLockExclusive(&uncovered::lock)) return false;
    const bool pending = uncovered::read != uncovered::count;
    const auto value = pending ? uncovered::reports[uncovered::read++] : uncovered::report{};
    bool count_due = false;
    const auto total = written.load(std::memory_order_relaxed);
    if (!pending && total != uncovered::logged_written && (!uncovered::logged_at || now_ms - uncovered::logged_at >= 10000)) {
      uncovered::logged_written = total; uncovered::logged_at = now_ms; count_due = true;
    }
    ReleaseSRWLockExclusive(&uncovered::lock);
    if (count_due) {
      std::snprintf(out, size, "Sunshine 3D barrier coverage: written=%llu (hooks written into lists' own method tables "
        "at their execution: about one per list where the runtime keeps that table across Reset, one per execution "
        "where it rebuilds it)", static_cast<unsigned long long>(total));
      return true;
    }
    if (!pending) return false;
    char table[160], barrier[160], enhanced[160];
    if (value.image) describe(reinterpret_cast<const void *>(value.table), table, sizeof(table));
    else std::snprintf(table, sizeof(table), "%p(object-owned, object%+lld)", reinterpret_cast<const void *>(value.table),
      static_cast<long long>(value.table - value.object));
    describe(value.barrier, barrier, sizeof(barrier));
    if (value.version7) describe(value.enhanced, enhanced, sizeof(enhanced));
    else std::snprintf(enhanced, sizeof(enhanced), "none");
    std::snprintf(out, size, "Sunshine 3D barrier coverage: a list no barrier hook covered: table=%s ResourceBarrier=%s Barrier=%s; "
      "its executions count as uncovered; logged once per method table (object-owned tables once)", table, barrier, enhanced);
    return true;
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
    if (!suppression_depth && invalidation_pending.exchange(false, std::memory_order_acq_rel)) invalidate();
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
      barrier_overflow.load(), submission_overflow.load(), discovery_contention.load(), written.load()};
    if (TryAcquireSRWLockShared(&targets_lock)) {
      for (const auto count : target_counts) result.targets += count;
      ReleaseSRWLockShared(&targets_lock);
    }
    return result;
  }
  own_flush_scope::own_flush_scope() { ++own_flush_depth; }
  own_flush_scope::~own_flush_scope() { --own_flush_depth; }
  const hooked_method *hooked_methods(unsigned &count) {
    count = method_count;
    return hooked_table;
  }
  suppression_scope::suppression_scope() { ++suppression_depth; }
  suppression_scope::~suppression_scope() { --suppression_depth; }

#ifdef SUNSHINE_STREAMLINE_NATIVE_OBSERVER_TEST
  std::uint64_t slot_reads_for_testing() { return slot_reads.load(); }
  void forget_enhanced_barrier_use() { enhanced_used = false; barrier_blind = false; }
  void own_table_for_testing(std::uint64_t object, const void *table) {
    test_own_object = table ? object : 0;
    test_own_table = reinterpret_cast<std::uintptr_t>(table);
  }
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
    for (auto &slot : order::tickets) { slot.id = 0; slot.entered = 0; slot.slots = {}; slot.cookie = 0; slot.type = 0; }
    order::requested = true; order::reset_pending = false;
    order::resets = order::ordered_count = order::cpu_unhooked = order::contention = order::own_signals = 0;
    order::conflicted_seen = 0;
    order::reported = {}; order::report_count = order::report_read = 0;
    for (auto &value : order::refused) value = 0;
    calls = observed = unreadable = dropped = installed = rejected = nested = suppressed = slot_reads = 0;
    barrier_overflow = submission_overflow = discovery_contention = written = 0;
    invalidation_pending = false;
    forget_enhanced_barrier_use();
    uncovered::reports = {}; uncovered::count = uncovered::read = 0;
    uncovered::logged_written = uncovered::logged_at = 0;
    barrier_callback = nullptr; submitted_callback = nullptr; invalidated_callback = nullptr;
    enhanced_textures_callback = nullptr; associate_recording_callback = nullptr; listener_callback = nullptr;
  }
#endif
}

#include "native_queue_order.inl"
