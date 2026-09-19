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
    enum class method : unsigned { barriers, reset, close, execute, bundle, enhanced, begin_pass, end_pass, count };
    constexpr unsigned method_count = static_cast<unsigned>(method::count);
    constexpr unsigned target_limit = 8, command_chunk = 64, inline_barriers = 256;
    constexpr std::array<std::size_t, method_count> vtable_slots{
      offsetof(ID3D12GraphicsCommandListVtbl, ResourceBarrier) / sizeof(void *),
      offsetof(ID3D12GraphicsCommandListVtbl, Reset) / sizeof(void *),
      offsetof(ID3D12GraphicsCommandListVtbl, Close) / sizeof(void *),
      offsetof(ID3D12CommandQueueVtbl, ExecuteCommandLists) / sizeof(void *),
      offsetof(ID3D12GraphicsCommandListVtbl, ExecuteBundle) / sizeof(void *),
      offsetof(ID3D12GraphicsCommandList7Vtbl, Barrier) / sizeof(void *),
      offsetof(ID3D12GraphicsCommandList4Vtbl, BeginRenderPass) / sizeof(void *),
      offsetof(ID3D12GraphicsCommandList4Vtbl, EndRenderPass) / sizeof(void *)};
    static_assert(vtable_slots[0] == 26 && vtable_slots[1] == 10 && vtable_slots[2] == 9 && vtable_slots[3] == 10,
      "Unexpected D3D12 native COM ABI");
    static_assert(vtable_slots[4] == 27 && vtable_slots[5] == 80 && vtable_slots[6] == 68 && vtable_slots[7] == 69,
      "Unexpected extended D3D12 native COM ABI");
    using barrier_fn = void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *, UINT, const D3D12_RESOURCE_BARRIER *);
    using reset_fn = HRESULT (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *, ID3D12CommandAllocator *, ID3D12PipelineState *);
    using close_fn = HRESULT (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *);
    using execute_fn = void (STDMETHODCALLTYPE *)(ID3D12CommandQueue *, UINT, ID3D12CommandList *const *);
    using bundle_fn = void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *, ID3D12GraphicsCommandList *);
    using enhanced_fn = void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList7 *, UINT32, const D3D12_BARRIER_GROUP *);
    using begin_pass_fn = void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList4 *, UINT, const D3D12_RENDER_PASS_RENDER_TARGET_DESC *, const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC *, D3D12_RENDER_PASS_FLAGS);
    using end_pass_fn = void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList4 *);
    constexpr GUID recording_guid{0x92c957c6, 0x88de, 0x42b3, {0x9f, 0x3e, 0x31, 0xde, 0x63, 0x05, 0x7b, 0x29}};
    enum class state { empty, pending, installed, rejected };
    struct target {
      sunshine_native_vtable::slot location;
      std::atomic<void *> original{};
      std::atomic<state> status{state::empty};
    };
    std::array<std::array<target, target_limit>, method_count> targets;
    std::array<unsigned, method_count> target_counts{};
    SRWLOCK targets_lock = SRWLOCK_INIT;
    std::atomic<bool> requested{}, active{};
    std::atomic<std::uint64_t> activation_epoch{};
    std::atomic<decltype(callbacks::barriers)> barrier_callback{};
    std::atomic<decltype(callbacks::reset)> reset_callback{};
    std::atomic<decltype(callbacks::close)> close_callback{};
    std::atomic<decltype(callbacks::submitted)> submitted_callback{};
    std::atomic<decltype(callbacks::invalidated)> invalidated_callback{};
    std::atomic<decltype(callbacks::invalidated_command)> invalidated_command_callback{};
    std::atomic<decltype(callbacks::render_pass)> render_pass_callback{};
    std::atomic<std::uint64_t> calls{}, observed{}, unreadable{}, dropped{}, installed{}, rejected{}, nested{}, suppressed{};
    std::atomic<std::uint64_t> barrier_overflow{}, submission_overflow{}, discovery_contention{};
    thread_local unsigned suppression_depth{};
    struct invocation;
    thread_local std::array<invocation *, method_count> current_invocations{};

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
      invocation(method value, state status) : kind(static_cast<unsigned>(value)), parent(current_invocations[kind]),
          epoch(activation_epoch.load(std::memory_order_acquire)), eligible(status == state::installed && enabled() && !suppression_depth) {
        ++calls;
        if (suppression_depth) ++suppressed;
        // Only the deepest hooked native implementation reports an operation.
        // In particular a proxy's post-call must not report the submission twice.
        if (parent) { parent->child = true; ++nested; }
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
    void invalidate() noexcept { notify(invalidated_callback.load(std::memory_order_acquire)); }
    void drop(bool unreadable_input) noexcept {
      if (unreadable_input) ++unreadable;
      else ++dropped;
      invalidate();
    }

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
        cookie = get_recording_cookie(reinterpret_cast<std::uintptr_t>(command));
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
    template<unsigned Index> HRESULT STDMETHODCALLTYPE reset_detour(ID3D12GraphicsCommandList *command,
        ID3D12CommandAllocator *allocator, ID3D12PipelineState *initial_state) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::reset)][Index];
      invocation call(method::reset, slot.status.load(std::memory_order_acquire));
      const auto cookie = call.eligible ? get_recording_cookie(reinterpret_cast<std::uintptr_t>(command)) : 0;
      SetLastError(incoming);
      const HRESULT result = reinterpret_cast<reset_fn>(slot.original.load(std::memory_order_acquire))(command, allocator, initial_state);
      const restore_error outgoing{GetLastError()};
      if (call.notify()) notify(reset_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(command), cookie, result);
      return result;
    }
    template<unsigned Index> HRESULT STDMETHODCALLTYPE close_detour(ID3D12GraphicsCommandList *command) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::close)][Index];
      invocation call(method::close, slot.status.load(std::memory_order_acquire));
      const auto cookie = call.eligible ? get_recording_cookie(reinterpret_cast<std::uintptr_t>(command)) : 0;
      SetLastError(incoming);
      const HRESULT result = reinterpret_cast<close_fn>(slot.original.load(std::memory_order_acquire))(command);
      const restore_error outgoing{GetLastError()};
      if (call.notify()) notify(close_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(command), cookie, result);
      return result;
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
      SetLastError(incoming);
      reinterpret_cast<execute_fn>(slot.original.load(std::memory_order_acquire))(queue, count, commands);
      outgoing.value = GetLastError();
      if (!call.notify()) return;
      if (!complete) { if (!allocated) ++submission_overflow; drop(allocated); return; }
      notify(submitted_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(queue), count, identities.data());
    }
    template<unsigned Index> void STDMETHODCALLTYPE bundle_detour(ID3D12GraphicsCommandList *command, ID3D12GraphicsCommandList *bundle) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::bundle)][Index];
      invocation call(method::bundle, slot.status.load(std::memory_order_acquire));
      const auto cookie = call.eligible ? get_recording_cookie(reinterpret_cast<std::uintptr_t>(command)) : 0;
      SetLastError(incoming);
      reinterpret_cast<bundle_fn>(slot.original.load(std::memory_order_acquire))(command, bundle);
      const restore_error outgoing{GetLastError()};
      if (call.notify()) notify(invalidated_command_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(command), cookie);
    }
    template<unsigned Index> void STDMETHODCALLTYPE enhanced_detour(ID3D12GraphicsCommandList7 *command, UINT32 count, const D3D12_BARRIER_GROUP *groups) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::enhanced)][Index];
      invocation call(method::enhanced, slot.status.load(std::memory_order_acquire));
      const auto cookie = call.eligible ? get_recording_cookie(reinterpret_cast<std::uintptr_t>(command)) : 0;
      SetLastError(incoming);
      reinterpret_cast<enhanced_fn>(slot.original.load(std::memory_order_acquire))(command, count, groups);
      const restore_error outgoing{GetLastError()};
      if (call.notify()) notify(invalidated_command_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(command), cookie);
    }
    template<unsigned Index> void STDMETHODCALLTYPE begin_pass_detour(ID3D12GraphicsCommandList4 *command, UINT count,
        const D3D12_RENDER_PASS_RENDER_TARGET_DESC *render_targets, const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC *depth, D3D12_RENDER_PASS_FLAGS flags) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::begin_pass)][Index];
      invocation call(method::begin_pass, slot.status.load(std::memory_order_acquire));
      const auto cookie = call.eligible ? get_recording_cookie(reinterpret_cast<std::uintptr_t>(command)) : 0;
      SetLastError(incoming);
      reinterpret_cast<begin_pass_fn>(slot.original.load(std::memory_order_acquire))(command, count, render_targets, depth, flags);
      const restore_error outgoing{GetLastError()};
      if (call.notify()) notify(render_pass_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(command), cookie, true);
    }
    template<unsigned Index> void STDMETHODCALLTYPE end_pass_detour(ID3D12GraphicsCommandList4 *command) {
      const DWORD incoming = GetLastError();
      auto &slot = targets[static_cast<unsigned>(method::end_pass)][Index];
      invocation call(method::end_pass, slot.status.load(std::memory_order_acquire));
      const auto cookie = call.eligible ? get_recording_cookie(reinterpret_cast<std::uintptr_t>(command)) : 0;
      SetLastError(incoming);
      reinterpret_cast<end_pass_fn>(slot.original.load(std::memory_order_acquire))(command);
      const restore_error outgoing{GetLastError()};
      if (call.notify()) notify(render_pass_callback.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(command), cookie, false);
    }
    const std::array<std::array<void *, target_limit>, method_count> detours{{
      {reinterpret_cast<void *>(barrier_detour<0>), reinterpret_cast<void *>(barrier_detour<1>), reinterpret_cast<void *>(barrier_detour<2>), reinterpret_cast<void *>(barrier_detour<3>), reinterpret_cast<void *>(barrier_detour<4>), reinterpret_cast<void *>(barrier_detour<5>), reinterpret_cast<void *>(barrier_detour<6>), reinterpret_cast<void *>(barrier_detour<7>)},
      {reinterpret_cast<void *>(reset_detour<0>), reinterpret_cast<void *>(reset_detour<1>), reinterpret_cast<void *>(reset_detour<2>), reinterpret_cast<void *>(reset_detour<3>), reinterpret_cast<void *>(reset_detour<4>), reinterpret_cast<void *>(reset_detour<5>), reinterpret_cast<void *>(reset_detour<6>), reinterpret_cast<void *>(reset_detour<7>)},
      {reinterpret_cast<void *>(close_detour<0>), reinterpret_cast<void *>(close_detour<1>), reinterpret_cast<void *>(close_detour<2>), reinterpret_cast<void *>(close_detour<3>), reinterpret_cast<void *>(close_detour<4>), reinterpret_cast<void *>(close_detour<5>), reinterpret_cast<void *>(close_detour<6>), reinterpret_cast<void *>(close_detour<7>)},
      {reinterpret_cast<void *>(execute_detour<0>), reinterpret_cast<void *>(execute_detour<1>), reinterpret_cast<void *>(execute_detour<2>), reinterpret_cast<void *>(execute_detour<3>), reinterpret_cast<void *>(execute_detour<4>), reinterpret_cast<void *>(execute_detour<5>), reinterpret_cast<void *>(execute_detour<6>), reinterpret_cast<void *>(execute_detour<7>)},
      {reinterpret_cast<void *>(bundle_detour<0>), reinterpret_cast<void *>(bundle_detour<1>), reinterpret_cast<void *>(bundle_detour<2>), reinterpret_cast<void *>(bundle_detour<3>), reinterpret_cast<void *>(bundle_detour<4>), reinterpret_cast<void *>(bundle_detour<5>), reinterpret_cast<void *>(bundle_detour<6>), reinterpret_cast<void *>(bundle_detour<7>)},
      {reinterpret_cast<void *>(enhanced_detour<0>), reinterpret_cast<void *>(enhanced_detour<1>), reinterpret_cast<void *>(enhanced_detour<2>), reinterpret_cast<void *>(enhanced_detour<3>), reinterpret_cast<void *>(enhanced_detour<4>), reinterpret_cast<void *>(enhanced_detour<5>), reinterpret_cast<void *>(enhanced_detour<6>), reinterpret_cast<void *>(enhanced_detour<7>)},
      {reinterpret_cast<void *>(begin_pass_detour<0>), reinterpret_cast<void *>(begin_pass_detour<1>), reinterpret_cast<void *>(begin_pass_detour<2>), reinterpret_cast<void *>(begin_pass_detour<3>), reinterpret_cast<void *>(begin_pass_detour<4>), reinterpret_cast<void *>(begin_pass_detour<5>), reinterpret_cast<void *>(begin_pass_detour<6>), reinterpret_cast<void *>(begin_pass_detour<7>)},
      {reinterpret_cast<void *>(end_pass_detour<0>), reinterpret_cast<void *>(end_pass_detour<1>), reinterpret_cast<void *>(end_pass_detour<2>), reinterpret_cast<void *>(end_pass_detour<3>), reinterpret_cast<void *>(end_pass_detour<4>), reinterpret_cast<void *>(end_pass_detour<5>), reinterpret_cast<void *>(end_pass_detour<6>), reinterpret_cast<void *>(end_pass_detour<7>)}
    }};
    target *find(unsigned kind, void **address) {
      for (unsigned i = 0; i != target_counts[kind]; ++i) if (targets[kind][i].location.address == address) return &targets[kind][i];
      return nullptr;
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
      if (!read_vtable_entry(address, 0, query)) {
        if (discover) drop(true);
        return nullptr;
      }
      auto *object = reinterpret_cast<ID3D12Object *>(address);
      Interface *result{};
      const HRESULT status = object->lpVtbl->QueryInterface(object, iid, reinterpret_cast<void **>(&result));
      if (FAILED(status) || !result) {
        if (discover) { ++rejected; invalidate(); }
        return nullptr;
      }
      return result;
    }
    void observe(std::uint64_t object, method value) {
      if (!requested.load(std::memory_order_acquire) || suppression_depth) return;
      const restore_error incoming{GetLastError()};
      const auto kind = static_cast<unsigned>(value);
      void **address{};
      if (!read_slot_address(object, kind, address)) { drop(true); return; }
      if (!TryAcquireSRWLockShared(&targets_lock)) { ++discovery_contention; drop(false); return; }
      const bool seen = find(kind, address) != nullptr;
      ReleaseSRWLockShared(&targets_lock);
      if (seen) return;
      sunshine_native_vtable::slot location;
      if (!sunshine_native_vtable::discover(object, vtable_slots[kind], location)) {
        ++rejected; invalidate(); return;
      }
      if (!TryAcquireSRWLockExclusive(&targets_lock)) { sunshine_native_vtable::release(location); ++discovery_contention; drop(false); return; }
      if (find(kind, location.address)) { ReleaseSRWLockExclusive(&targets_lock); sunshine_native_vtable::release(location); return; }
      if (target_counts[kind] == target_limit) {
        ReleaseSRWLockExclusive(&targets_lock); sunshine_native_vtable::release(location); ++rejected; invalidate(); return;
      }
      auto &slot = targets[kind][target_counts[kind]++];
      slot.location = location;
      slot.status.store(state::pending, std::memory_order_release);
      ReleaseSRWLockExclusive(&targets_lock);
      invalidate();
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
      if (lost) { ++rejected; invalidate(); }
      return result;
    }
    bool optional_coverage(std::uint64_t command, bool discover) {
      const restore_error incoming{GetLastError()};
      auto *object = reinterpret_cast<ID3D12Object *>(command);
      ID3D12GraphicsCommandList4 *version4{};
      const HRESULT result4 = object->lpVtbl->QueryInterface(object, IID_ID3D12GraphicsCommandList4, reinterpret_cast<void **>(&version4));
      bool complete = result4 == E_NOINTERFACE;
      if (SUCCEEDED(result4) && version4) {
        const auto address = reinterpret_cast<std::uintptr_t>(version4);
        if (discover) { observe(address, method::begin_pass); observe(address, method::end_pass); }
        complete = ready(address, method::begin_pass) && ready(address, method::end_pass);
        version4->lpVtbl->Release(version4);
      }
      ID3D12GraphicsCommandList7 *version7{};
      const HRESULT result7 = object->lpVtbl->QueryInterface(object, IID_ID3D12GraphicsCommandList7, reinterpret_cast<void **>(&version7));
      if (SUCCEEDED(result7) && version7) {
        const auto address = reinterpret_cast<std::uintptr_t>(version7);
        if (discover) observe(address, method::enhanced);
        complete = ready(address, method::enhanced) && complete;
        version7->lpVtbl->Release(version7);
      } else if (result7 != E_NOINTERFACE) complete = false;
      if (discover && ((FAILED(result4) && result4 != E_NOINTERFACE) || (SUCCEEDED(result4) && !version4) ||
          (FAILED(result7) && result7 != E_NOINTERFACE) || (SUCCEEDED(result7) && !version7))) invalidate();
      return complete;
    }
  }

  void initialize(callbacks value) {
    if (sunshine_addon_lifetime::stopping()) return;
    active.store(false, std::memory_order_release);
    ++activation_epoch;
    barrier_callback.store(value.barriers, std::memory_order_release);
    reset_callback.store(value.reset, std::memory_order_release);
    close_callback.store(value.close, std::memory_order_release);
    submitted_callback.store(value.submitted, std::memory_order_release);
    invalidated_callback.store(value.invalidated, std::memory_order_release);
    invalidated_command_callback.store(value.invalidated_command, std::memory_order_release);
    render_pass_callback.store(value.render_pass, std::memory_order_release);
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
  void observe_command(std::uint64_t command) {
    if (sunshine_addon_lifetime::stopping() || !requested.load(std::memory_order_acquire) || suppression_depth) return;
    const restore_error incoming{GetLastError()};
    auto *object = required_interface<ID3D12GraphicsCommandList>(command, IID_ID3D12GraphicsCommandList, true);
    if (!object) return;
    const auto normalized = reinterpret_cast<std::uintptr_t>(object);
    observe(normalized, method::barriers); observe(normalized, method::reset); observe(normalized, method::close); observe(normalized, method::bundle);
    optional_coverage(normalized, true);
    object->lpVtbl->Release(object);
  }
  void observe_queue(std::uint64_t queue) {
    if (sunshine_addon_lifetime::stopping() || !requested.load(std::memory_order_acquire) || suppression_depth) return;
    const restore_error incoming{GetLastError()};
    auto *object = required_interface<ID3D12CommandQueue>(queue, IID_ID3D12CommandQueue, true);
    if (!object) return;
    observe(reinterpret_cast<std::uintptr_t>(object), method::execute);
    object->lpVtbl->Release(object);
  }
  bool command_ready(std::uint64_t command) {
    if (!enabled()) return false;
    const restore_error incoming{GetLastError()};
    auto *object = required_interface<ID3D12GraphicsCommandList>(command, IID_ID3D12GraphicsCommandList, false);
    if (!object) return false;
    const auto normalized = reinterpret_cast<std::uintptr_t>(object);
    const bool result = ready(normalized, method::barriers) && ready(normalized, method::reset) && ready(normalized, method::close) &&
      ready(normalized, method::bundle) && optional_coverage(normalized, false);
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
    std::array<std::array<unsigned, 2>, method_count * target_limit> pending{};
    unsigned count{};
    if (!TryAcquireSRWLockShared(&targets_lock)) { ++discovery_contention; drop(false); return false; }
    for (unsigned kind = 0; kind != method_count; ++kind)
      for (unsigned i = 0; i != target_counts[kind]; ++i)
        if (targets[kind][i].status.load(std::memory_order_acquire) == state::pending) pending[count++] = {kind, i};
    ReleaseSRWLockShared(&targets_lock);
    if (count) invalidate();
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
    if (count) invalidate();
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
  counters counts() {
    counters result{calls.load(), observed.load(), unreadable.load(), dropped.load(), 0, installed.load(), rejected.load(), nested.load(), suppressed.load(),
      barrier_overflow.load(), submission_overflow.load(), discovery_contention.load()};
    if (TryAcquireSRWLockShared(&targets_lock)) {
      for (const auto count : target_counts) result.targets += count;
      ReleaseSRWLockShared(&targets_lock);
    }
    return result;
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
    calls = observed = unreadable = dropped = installed = rejected = nested = suppressed = 0;
    barrier_overflow = submission_overflow = discovery_contention = 0;
    barrier_callback = nullptr; reset_callback = nullptr; close_callback = nullptr; submitted_callback = nullptr; invalidated_callback = nullptr;
    invalidated_command_callback = nullptr; render_pass_callback = nullptr;
  }
#endif
}
