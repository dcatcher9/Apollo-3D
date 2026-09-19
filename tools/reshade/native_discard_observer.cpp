// SPDX-License-Identifier: GPL-3.0-only
#include "native_discard_observer.h"
#include "addon_lifetime.h"
#include "native_vtable_slot.h"
#define CINTERFACE
#include <d3d12.h>
#include <array>
#include <atomic>
#include <cstddef>

namespace sunshine_streamline::native_discard {
  namespace {
    constexpr std::size_t discard_slot = offsetof(ID3D12GraphicsCommandListVtbl, DiscardResource) / sizeof(void *);
    static_assert(discard_slot == 51, "Unexpected ID3D12GraphicsCommandList COM ABI");
    using discard_fn = void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *, ID3D12Resource *, const D3D12_DISCARD_REGION *);
    enum class state { empty, pending, installed, rejected };
    struct target {
      sunshine_native_vtable::slot binding;
      std::atomic<discard_fn> original{};
      std::atomic<state> status{state::empty};
    };
    constexpr unsigned target_limit = 8;
    std::array<target, target_limit> targets;
    SRWLOCK targets_lock = SRWLOCK_INIT;
    std::atomic<bool> requested{}, active{};
    std::atomic<invalidate_callback> callback{};
    std::atomic<std::uint64_t> calls{}, observed{}, unreadable{}, dropped{}, installed{}, rejected{};
    unsigned target_count{};

    bool enabled() {
      return !sunshine_addon_lifetime::stopping() && requested.load(std::memory_order_acquire) && active.load(std::memory_order_acquire);
    }

    bool read(const void *source, void *output, std::size_t size) {
      SIZE_T copied{};
      return source && ReadProcessMemory(GetCurrentProcess(), source, output, size, &copied) && copied == size;
    }
    void invalidate(std::uint64_t command, std::uint64_t resource) noexcept {
      if (!enabled()) return;
      if (const auto fn = callback.load(std::memory_order_acquire)) {
        try { fn(command, resource); }
        catch (...) { active.store(false, std::memory_order_release); ++dropped; }
      }
    }
    template<unsigned Index> void STDMETHODCALLTYPE detour(ID3D12GraphicsCommandList *command,
        ID3D12Resource *resource, const D3D12_DISCARD_REGION *region) {
      const DWORD incoming = GetLastError();
      ++calls;
      auto &slot = targets[Index];
      if (slot.status.load(std::memory_order_acquire) == state::installed && enabled()) {
        ++observed;
        // A partial discard also invalidates whole-resource evidence. Neither
        // the optional region nor its rectangle array is ever dereferenced.
        invalidate(reinterpret_cast<std::uintptr_t>(command), reinterpret_cast<std::uintptr_t>(resource));
      }
      SetLastError(incoming);
      // Published before enabling this detour and immutable while callable.
      slot.original.load(std::memory_order_acquire)(command, resource, region);
    }
    constexpr std::array<discard_fn, target_limit> detours{
      detour<0>, detour<1>, detour<2>, detour<3>, detour<4>, detour<5>, detour<6>, detour<7>};
    bool existing(void **address) {
      for (unsigned i = 0; i != target_count; ++i) if (targets[i].binding.address == address) return true;
      return false;
    }
  }
  void initialize(invalidate_callback value) {
    if (sunshine_addon_lifetime::stopping()) return;
    active.store(false, std::memory_order_release);
    callback.store(value, std::memory_order_release);
    requested.store(true, std::memory_order_release);
  }
  void set_active(bool value) { active.store(value && !sunshine_addon_lifetime::stopping(), std::memory_order_release); }
  void shutdown() {
    active.store(false, std::memory_order_release);
    requested.store(false, std::memory_order_release);
  }
  void observe_command(std::uint64_t native) {
    if (sunshine_addon_lifetime::stopping() || !requested.load(std::memory_order_acquire)) return;
    const DWORD incoming = GetLastError();
    struct restore { DWORD value; ~restore() { SetLastError(value); } } restore_error{incoming};
    std::uintptr_t vtable{}; void *query{};
    if (!read(reinterpret_cast<void *>(native), &vtable, sizeof(vtable)) || !vtable ||
        !read(reinterpret_cast<void *>(vtable), &query, sizeof(query)) || !query) {
      ++unreadable; invalidate(native, 0); return;
    }
    auto *unknown = reinterpret_cast<ID3D12Object *>(native);
    ID3D12GraphicsCommandList *command{};
    if (FAILED(unknown->lpVtbl->QueryInterface(unknown, IID_ID3D12GraphicsCommandList,
          reinterpret_cast<void **>(&command))) || !command) {
      ++rejected; invalidate(native, 0); return;
    }
    struct release_interface { ID3D12GraphicsCommandList *value; ~release_interface() { value->lpVtbl->Release(value); } } release{command};
    if (!read(command, &vtable, sizeof(vtable)) || !vtable || vtable > UINTPTR_MAX - discard_slot * sizeof(void *)) {
      ++unreadable; invalidate(native, 0); return;
    }
    const auto address = reinterpret_cast<void **>(vtable + discard_slot * sizeof(void *));
    if (!TryAcquireSRWLockShared(&targets_lock)) { ++dropped; invalidate(native, 0); return; }
    const bool seen = existing(address);
    ReleaseSRWLockShared(&targets_lock);
    if (seen) return;
    sunshine_native_vtable::slot binding;
    if (!sunshine_native_vtable::discover(reinterpret_cast<std::uintptr_t>(command), discard_slot, binding)) {
      ++rejected; invalidate(native, 0); return;
    }
    if (!TryAcquireSRWLockExclusive(&targets_lock)) {
      sunshine_native_vtable::release(binding); ++dropped; invalidate(native, 0); return;
    }
    if (existing(binding.address)) { ReleaseSRWLockExclusive(&targets_lock); sunshine_native_vtable::release(binding); return; }
    if (target_count == target_limit) {
      ReleaseSRWLockExclusive(&targets_lock); sunshine_native_vtable::release(binding); ++rejected; invalidate(native, 0); return;
    }
    auto &slot = targets[target_count++];
    slot.binding = binding;
    slot.status.store(state::pending, std::memory_order_release);
    ReleaseSRWLockExclusive(&targets_lock);
  }
  bool install_pending() {
    if (!enabled()) return false;
    // Caller serializes installation. The target is a module-owned vtable slot,
    // not globally shared code; unrelated interfaces can use the same function.
    std::array<unsigned, target_limit> pending{}; unsigned count{};
    if (!TryAcquireSRWLockShared(&targets_lock)) { ++dropped; invalidate(0, 0); return false; }
    bool okay = true, changed = false;
    for (unsigned i = 0; i != target_count; ++i) {
      auto &slot = targets[i];
      const auto status = slot.status.load(std::memory_order_acquire);
      if (status == state::pending) pending[count++] = i;
      else if (status == state::installed && !sunshine_native_vtable::matches(slot.binding, reinterpret_cast<void *>(detours[i]))) {
        slot.status.store(state::rejected, std::memory_order_release); ++rejected; okay = false;
      } else if (status == state::rejected) okay = false;
    }
    ReleaseSRWLockShared(&targets_lock);
    for (unsigned n = 0; n != count; ++n) {
      auto &slot = targets[pending[n]];
      // Publish the correctly typed original before making our slot callable.
      // Partial publication failures retain it and all lifetime pins.
      slot.original.store(reinterpret_cast<discard_fn>(slot.binding.original), std::memory_order_release);
      if (!sunshine_native_vtable::install(slot.binding, reinterpret_cast<void *>(detours[pending[n]]))) {
        slot.status.store(state::rejected, std::memory_order_release); ++rejected; okay = false; continue;
      }
      slot.status.store(state::installed, std::memory_order_release);
      ++installed; changed = true;
    }
    if (changed || !okay) invalidate(0, 0);
    return okay;
  }
  counters counts() {
    counters out{calls.load(), observed.load(), unreadable.load(), dropped.load(), 0, installed.load(), rejected.load()};
    if (TryAcquireSRWLockShared(&targets_lock)) { out.targets = target_count; ReleaseSRWLockShared(&targets_lock); }
    return out;
  }
  unsigned discard_vtable_slot() { return static_cast<unsigned>(discard_slot); }
#ifdef SUNSHINE_NATIVE_DISCARD_TEST
  void reset_after_hooks_removed() {
    shutdown();
    bool restored = true;
    for (unsigned i = 0; i != target_count; ++i)
      if (sunshine_native_vtable::matches(targets[i].binding, reinterpret_cast<void *>(detours[i])))
        restored = sunshine_native_vtable::restore(targets[i].binding, reinterpret_cast<void *>(detours[i])) && restored;
    // A failed protection change can leave our slot callable. Preserve every
    // original/binding in that case, even during quiescent fixture cleanup.
    if (!restored) return;
    for (unsigned i = 0; i != target_count; ++i) {
      sunshine_native_vtable::release(targets[i].binding);
      targets[i].original = nullptr;
      targets[i].status.store(state::empty, std::memory_order_release);
    }
    target_count = 0; calls = observed = unreadable = dropped = installed = rejected = 0;
    callback = nullptr;
  }
#endif
}
