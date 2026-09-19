// SPDX-License-Identifier: GPL-3.0-only
// Real COM vtable-slot interception on isolated fixtures. No graphics API/GPU.
#include "native_discard_observer.h"
#define CINTERFACE
#include <d3d12.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <thread>

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline, noipa))
#else
#define NOINLINE __declspec(noinline)
#endif
namespace {
  namespace observer = sunshine_streamline::native_discard;
  using function = void (STDMETHODCALLTYPE *)(void *, void *, const void *);
  struct object { void **vtable{}; object *queried{}; ULONG references{1}; bool graphics{true}; };
  constexpr std::size_t header_slot = offsetof(ID3D12GraphicsCommandListVtbl, DiscardResource) / sizeof(void *);
  // Production accepts module-owned vtables only; these are image data rather
  // than temporary stack/heap arrays whose lifetime the observer cannot own.
  std::array<std::array<void *, header_slot + 1>, 10> tables{};
  std::array<void *, header_slot + 1> unrelated_table{};
  std::array<void *, header_slot + 1> rejected_table{};
  struct invocation { void *command{}, *resource{}; const void *region{}; DWORD incoming{}; unsigned target{}; } invoked;
  constexpr DWORD incoming_error = 0x12344321, outgoing_error = 0xabcdef01;
  std::atomic<unsigned> original_calls{}, mutations{}, epochs{};
  std::atomic<bool> block_original{}, entered{}, release_original{};
  std::uint64_t observed_command{}, observed_resource{};
  bool throw_callback{};
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  NOINLINE HRESULT STDMETHODCALLTYPE query(ID3D12Object *native, REFIID iid, void **output) {
    if (!output) return E_POINTER;
    *output = nullptr;
    auto *value = reinterpret_cast<object *>(native);
    if (IsEqualGUID(iid, IID_IUnknown)) { *output = value; ++value->references; return S_OK; }
    if (value->queried) value = value->queried;
    if (value->graphics && IsEqualGUID(iid, IID_ID3D12GraphicsCommandList)) {
      *output = value; ++value->references; return S_OK;
    }
    return E_NOINTERFACE;
  }
  NOINLINE ULONG STDMETHODCALLTYPE add_ref(ID3D12Object *native) { return ++reinterpret_cast<object *>(native)->references; }
  NOINLINE ULONG STDMETHODCALLTYPE release(ID3D12Object *native) { return --reinterpret_cast<object *>(native)->references; }
  template<unsigned Index> NOINLINE void STDMETHODCALLTYPE original(void *command, void *resource, const void *region) {
    invoked = {command, resource, region, GetLastError(), Index};
    ++original_calls;
    if (block_original.load(std::memory_order_acquire)) {
      entered.store(true, std::memory_order_release);
      while (!release_original.load(std::memory_order_acquire)) std::this_thread::yield();
    }
    SetLastError(outgoing_error);
  }
  constexpr std::array<function, 10> originals{original<0>, original<1>, original<2>, original<3>, original<4>,
    original<5>, original<6>, original<7>, original<8>, original<9>};
  void observe_mutation(std::uint64_t command, std::uint64_t resource) {
    SetLastError(0xbadf00d);
    if (throw_callback) throw std::runtime_error("fixture observation failure");
    if (!command && !resource) { ++epochs; return; }
    observed_command = command; observed_resource = resource; ++mutations;
  }
  void call(object &value, void *resource, const void *region) {
    SetLastError(incoming_error);
    reinterpret_cast<function>(value.vtable[observer::discard_vtable_slot()])(&value, resource, region);
    require(GetLastError() == outgoing_error, "detour changed original outgoing LastError");
    require(invoked.command == &value && invoked.resource == resource && invoked.region == region && invoked.incoming == incoming_error,
      "detour changed native this/resource/region or incoming LastError");
  }
}
int main() {
  try {
    require(header_slot == 51 && observer::discard_vtable_slot() == header_slot, "DiscardResource slot differs from COM header ABI");
    observer::initialize(observe_mutation);
    std::array<object, 10> objects{};
    for (unsigned i = 0; i != objects.size(); ++i) {
      tables[i][0] = reinterpret_cast<void *>(query); tables[i][1] = reinterpret_cast<void *>(add_ref);
      tables[i][2] = reinterpret_cast<void *>(release);
      tables[i][header_slot] = reinterpret_cast<void *>(originals[i]); objects[i].vtable = tables[i].data();
    }
    // Two distinct authentic command-list vtables may share their code entry.
    // Only the observed slot may change, and each slot needs its own owner.
    tables[1][header_slot] = tables[0][header_slot];
    unrelated_table = tables[0];
    object unrelated{unrelated_table.data(), nullptr, 1, false};
    observer::observe_command(reinterpret_cast<std::uintptr_t>(&unrelated));
    require(observer::counts().targets == 0, "unrelated IUnknown was accepted as a command list");
    object base_interface{unrelated_table.data(), &objects[0], 1, false};
    observer::observe_command(reinterpret_cast<std::uintptr_t>(&base_interface));
    require(objects[0].references == 1, "normalized QI interface reference leaked");
    observer::observe_command(reinterpret_cast<std::uintptr_t>(&objects[0]));
    observer::observe_command(reinterpret_cast<std::uintptr_t>(&objects[0]));
    require(observer::counts().targets == 1 && !observer::install_pending(), "inactive discovery duplicated or enabled hooks");
    int resource{};
    const void *unreadable_region = reinterpret_cast<void *>(1); // Never dereferenced by observer or fixture original.
    call(objects[0], &resource, unreadable_region);
    require(original_calls == 1 && mutations == 0, "inactive unhooked call was altered");
    observer::set_active(true);
    require(observer::install_pending() && observer::counts().installed == 1 && epochs == 1, "deferred target installation/coverage reset failed");
    call(objects[0], &resource, unreadable_region);
    require(original_calls == 2 && mutations == 1 && observed_command == reinterpret_cast<std::uintptr_t>(&objects[0]) &&
        observed_resource == reinterpret_cast<std::uintptr_t>(&resource), "native discard did not forward exactly once and observe resource");
    object unknown{tables[0].data()};
    call(unknown, nullptr, nullptr);
    require(original_calls == 3 && mutations == 2 && observed_command == reinterpret_cast<std::uintptr_t>(&unknown) && !observed_resource,
      "unknown native instance sharing entry was not forwarded conservatively");
    observer::set_active(false);
    call(objects[0], &resource, nullptr);
    require(original_calls == 4 && mutations == 2, "inactive detour observed mutation or skipped original");
    observer::set_active(true);
    const auto before_unrelated = mutations.load();
    call(unrelated, &resource, unreadable_region);
    require(mutations == before_unrelated && unrelated_table[header_slot] == reinterpret_cast<void *>(originals[0]),
      "global function hook altered an unobserved interface sharing the same code address");
    observer::observe_command(1);
    require(observer::counts().unreadable > 0 && observed_command == 1 && !observed_resource,
      "unreadable COM identity did not reject discovery");
    rejected_table = tables[0]; rejected_table[header_slot] = &resource;
    object invalid_object{rejected_table.data()};
    const auto before_invalid = observer::counts().rejected;
    observer::observe_command(reinterpret_cast<std::uintptr_t>(&invalid_object));
    require(observer::counts().rejected > before_invalid, "non-executable target was accepted");
    auto temporary_table = unrelated_table;
    object temporary{temporary_table.data()};
    const auto before_temporary = observer::counts().rejected;
    observer::observe_command(reinterpret_cast<std::uintptr_t>(&temporary));
    require(observer::counts().rejected > before_temporary, "temporary non-module vtable was accepted");
    for (unsigned i = 1; i != objects.size(); ++i) observer::observe_command(reinterpret_cast<std::uintptr_t>(&objects[i]));
    require(observer::counts().targets == 8 && observer::counts().rejected >= 3, "bounded native target capacity was not enforced");
    require(observer::install_pending() && observer::counts().installed == 8, "multiple native entry trampolines failed installation");
    for (unsigned i = 0; i != objects.size(); ++i) {
      const auto before = original_calls.load();
      call(objects[i], &resource, unreadable_region);
      require(original_calls == before + 1 && invoked.target == (i == 1 ? 0 : i), "per-slot hook forwarded to wrong implementation");
    }
    // Shutdown never removes a trampoline while the original is in flight.
    block_original = true;
    std::thread running([&] { call(objects[0], &resource, unreadable_region); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool did_enter = entered.load(std::memory_order_acquire);
    observer::shutdown();
    release_original = true; running.join(); block_original = false;
    require(did_enter, "blocked native original did not enter");
    const auto before_shutdown_call = mutations.load();
    call(objects[0], &resource, nullptr);
    require(mutations == before_shutdown_call, "shutdown detour did not remain pure pass-through");
    observer::initialize(observe_mutation); observer::set_active(true);
    call(objects[0], &resource, nullptr);
    require(mutations == before_shutdown_call + 1, "reinitialized observer lost process-lifetime trampoline");
    throw_callback = true;
    const auto before_throw = original_calls.load();
    call(objects[0], &resource, unreadable_region);
    require(original_calls == before_throw + 1 && observer::counts().dropped > 0, "observation exception escaped or skipped original");
    throw_callback = false;
    observer::set_active(true);
    tables[1][header_slot] = reinterpret_cast<void *>(originals[9]);
    const auto before_replacement = epochs.load();
    require(!observer::install_pending() && epochs > before_replacement && tables[1][header_slot] == reinterpret_cast<void *>(originals[9]),
      "another slot owner's replacement was overwritten or accepted as complete observation");
    observer::shutdown();
    observer::reset_after_hooks_removed();
    require(tables[0][header_slot] == reinterpret_cast<void *>(originals[0]) && tables[1][header_slot] == reinterpret_cast<void *>(originals[9]),
      "quiescent cleanup failed to restore its own slot or overwrote another owner");
    std::puts("PASS native discard exact COM interface, shared-code slot isolation, bounded module-owned targets, unchanged arguments/LastError, inactive/unknown/shutdown forwarding");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    observer::shutdown();
    observer::reset_after_hooks_removed();
    return 1;
  }
}
