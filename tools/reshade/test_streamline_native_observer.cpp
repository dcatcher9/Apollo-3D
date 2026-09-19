// SPDX-License-Identifier: GPL-3.0-only
// Real native hooks on COM-shaped CPU fixtures. No graphics device or GPU.
#include "streamline_native_observer.h"
#include "addon_lifetime.h"
#define CINTERFACE
#include <d3d12.h>
#include <MinHook.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline, noipa))
#else
#define NOINLINE __declspec(noinline)
#endif
namespace {
  namespace observer = sunshine_streamline::native_observer;
  constexpr auto barrier_slot = offsetof(ID3D12GraphicsCommandListVtbl, ResourceBarrier) / sizeof(void *);
  constexpr auto reset_slot = offsetof(ID3D12GraphicsCommandListVtbl, Reset) / sizeof(void *);
  constexpr auto close_slot = offsetof(ID3D12GraphicsCommandListVtbl, Close) / sizeof(void *);
  constexpr auto allocator_slot = offsetof(ID3D12DeviceVtbl, CreateCommandAllocator) / sizeof(void *);
  constexpr auto execute_slot = offsetof(ID3D12CommandQueueVtbl, ExecuteCommandLists) / sizeof(void *);
  constexpr auto bundle_slot = offsetof(ID3D12GraphicsCommandListVtbl, ExecuteBundle) / sizeof(void *);
  constexpr auto enhanced_slot = offsetof(ID3D12GraphicsCommandList7Vtbl, Barrier) / sizeof(void *);
  constexpr auto begin_pass_slot = offsetof(ID3D12GraphicsCommandList4Vtbl, BeginRenderPass) / sizeof(void *);
  constexpr auto end_pass_slot = offsetof(ID3D12GraphicsCommandList4Vtbl, EndRenderPass) / sizeof(void *);
  using barrier_fn = void (STDMETHODCALLTYPE *)(void *, UINT, const D3D12_RESOURCE_BARRIER *);
  using reset_fn = HRESULT (STDMETHODCALLTYPE *)(void *, void *, void *);
  using close_fn = HRESULT (STDMETHODCALLTYPE *)(void *);
  using execute_fn = void (STDMETHODCALLTYPE *)(void *, UINT, void *const *);
  using bundle_fn = void (STDMETHODCALLTYPE *)(void *, void *);
  using enhanced_fn = void (STDMETHODCALLTYPE *)(void *, UINT32, const D3D12_BARRIER_GROUP *);
  using begin_pass_fn = void (STDMETHODCALLTYPE *)(void *, UINT, const D3D12_RENDER_PASS_RENDER_TARGET_DESC *, const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC *, D3D12_RENDER_PASS_FLAGS);
  using end_pass_fn = void (STDMETHODCALLTYPE *)(void *);
  enum class object_kind { unknown, command, queue, device };
  struct object {
    void **vtable{};
    std::uint64_t cookie{};
    object *forwarded{};
    bool extended{};
    object_kind kind{};
    object *queried{};
    unsigned references{1};
  };
#if defined(__GNUC__) && defined(__x86_64__)
  static_assert(offsetof(object, forwarded) == 16);
  // Streamline 1.1.1 uses exactly this ABI-transparent slot-9 forwarding thunk
  // for BOTH GraphicsCommandList::Close and Device::CreateCommandAllocator.
  // Its shared code has no single C++ signature; arguments pass through intact.
  __attribute__((naked, noinline)) void shared_slot9() {
    __asm__("mov 16(%rcx), %rcx\n\t"
            "mov (%rcx), %rax\n\t"
            "jmp *72(%rax)");
  }
#endif
  constexpr DWORD incoming_error = 0x12344321, outgoing_error = 0xabcdef01;
  std::atomic<unsigned> original_submissions{}, original_barriers{}, original_resets{}, original_closes{};
  std::atomic<bool> block_original{}, entered{}, release_original{};
  unsigned barrier_events{}, reset_events{}, close_events{}, submit_events{}, invalidations{};
  unsigned command_invalidations{}, render_pass_events{}, original_bundles{}, original_enhanced{}, original_passes{};
  bool inside_pass{};
  unsigned last_queue_target{};
  HRESULT native_result = S_OK, observed_result{};
  std::uint64_t observed_command{}, observed_cookie{}, observed_queue{};
  std::vector<observer::command_identity> observed_commands;
  std::vector<D3D12_RESOURCE_BARRIER> observed_barriers;
  std::vector<char> batch_order;
  unsigned observed_count{};
  D3D12_RESOURCE_BARRIER observed_barrier{};
  bool error_preserved = true, callback_after_original = true, mutate_cookie{}, inject_capture{}, throw_callback{};
  bool trace_batch{}, mutate_barriers{};
  unsigned original_submit_count{}, original_barrier_count{}, original_forwarded_submissions{};
  void *const *original_submit_pointer{}, *const *original_wrapper_pointer{};
  const D3D12_RESOURCE_BARRIER *original_barrier_pointer{};
  object *capture_command{}, *capture_queue{};
  unsigned unrelated_calls{};

  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  NOINLINE HRESULT STDMETHODCALLTYPE query(ID3D12Object *native, REFIID iid, void **output) {
    if (!output) return E_POINTER;
    auto *value = reinterpret_cast<object *>(native);
    *output = nullptr;
    if (IsEqualGUID(iid, IID_IUnknown)) {
      *output = value;
      ++value->references;
      return S_OK;
    }
    if (value->queried) value = value->queried;
    if ((value->kind == object_kind::command && (IsEqualGUID(iid, IID_ID3D12GraphicsCommandList) ||
        (value->extended && (IsEqualGUID(iid, IID_ID3D12GraphicsCommandList4) || IsEqualGUID(iid, IID_ID3D12GraphicsCommandList7))))) ||
        (value->kind == object_kind::queue && IsEqualGUID(iid, IID_ID3D12CommandQueue)) ||
        (value->kind == object_kind::device && IsEqualGUID(iid, IID_ID3D12Device))) {
      *output = value;
      ++value->references;
      return S_OK;
    }
    return E_NOINTERFACE;
  }
  NOINLINE ULONG STDMETHODCALLTYPE add_ref(ID3D12Object *native) { return ++reinterpret_cast<object *>(native)->references; }
  NOINLINE ULONG STDMETHODCALLTYPE release(ID3D12Object *native) { return --reinterpret_cast<object *>(native)->references; }
  NOINLINE HRESULT STDMETHODCALLTYPE unrelated_method(void *native, D3D12_COMMAND_LIST_TYPE type, REFIID iid, void **output) {
    ++unrelated_calls;
    require(native && type == D3D12_COMMAND_LIST_TYPE_COMPUTE && IsEqualGUID(iid, IID_ID3D12CommandAllocator) && output,
      "device CreateCommandAllocator received altered arguments");
    *output = native;
    SetLastError(outgoing_error);
    return S_FALSE;
  }
  NOINLINE HRESULT STDMETHODCALLTYPE get_data(ID3D12Object *native, REFGUID, UINT *size, void *output) {
    auto *value = reinterpret_cast<object *>(native);
    if (value->forwarded) value = value->forwarded;
    if (!size || *size != sizeof(value->cookie) || !output) return E_INVALIDARG;
    std::memcpy(output, &value->cookie, sizeof(value->cookie));
    SetLastError(0xbadf00d);
    return S_OK;
  }
  NOINLINE HRESULT STDMETHODCALLTYPE set_data(ID3D12Object *native, REFGUID, UINT size, const void *input) {
    auto *value = reinterpret_cast<object *>(native);
    if (value->forwarded) value = value->forwarded;
    if (size != sizeof(value->cookie) || !input) return E_INVALIDARG;
    std::memcpy(&value->cookie, input, sizeof(value->cookie));
    SetLastError(0xbadf00d);
    return S_OK;
  }
  template<unsigned Index> NOINLINE void STDMETHODCALLTYPE original_barrier(void *native, UINT count, const D3D12_RESOURCE_BARRIER *barriers) {
    ++original_barriers;
    if (trace_batch) batch_order.push_back(Index == 1 ? 'P' : 'B');
    error_preserved = error_preserved && GetLastError() == incoming_error;
    auto *value = static_cast<object *>(native);
    if constexpr (Index == 1) {
      reinterpret_cast<barrier_fn>(value->forwarded->vtable[barrier_slot])(value->forwarded, count, barriers);
    } else {
      original_barrier_count = count;
      original_barrier_pointer = barriers;
      if (mutate_barriers) for (UINT i = 0; i != count; ++i)
        const_cast<D3D12_RESOURCE_BARRIER *>(barriers)[i].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    }
    SetLastError(outgoing_error);
  }
  template<unsigned Index> NOINLINE HRESULT STDMETHODCALLTYPE original_reset(void *native, void *allocator, void *initial_state) {
    ++original_resets;
    error_preserved = error_preserved && GetLastError() == incoming_error;
    auto *value = static_cast<object *>(native);
    HRESULT result = native_result;
    if constexpr (Index == 1) result = reinterpret_cast<reset_fn>(value->forwarded->vtable[reset_slot])(value->forwarded, allocator, initial_state);
    SetLastError(outgoing_error);
    return result;
  }
  template<unsigned Index> NOINLINE HRESULT STDMETHODCALLTYPE original_close(void *native) {
    ++original_closes;
    error_preserved = error_preserved && GetLastError() == incoming_error;
    auto *value = static_cast<object *>(native);
    HRESULT result = native_result;
    if constexpr (Index == 1) result = reinterpret_cast<close_fn>(value->forwarded->vtable[close_slot])(value->forwarded);
    SetLastError(outgoing_error);
    return result;
  }
  template<unsigned Index> NOINLINE void STDMETHODCALLTYPE original_execute(void *native, UINT count, void *const *commands) {
    if (trace_batch) batch_order.push_back(Index == 1 ? 'W' : 'E');
    error_preserved = error_preserved && GetLastError() == incoming_error;
    auto *value = static_cast<object *>(native);
    if constexpr (Index == 1) {
      ++original_forwarded_submissions;
      original_wrapper_pointer = commands;
      std::vector<void *> unwrapped(count);
      for (UINT i = 0; i != count; ++i) {
        auto *command = static_cast<object *>(commands[i]);
        unwrapped[i] = command->forwarded ? command->forwarded : command;
      }
      reinterpret_cast<execute_fn>(value->forwarded->vtable[execute_slot])(value->forwarded, count, unwrapped.data());
    } else {
      ++original_submissions;
      original_submit_count = count;
      original_submit_pointer = commands;
      last_queue_target = Index;
      if (block_original.load(std::memory_order_acquire)) {
        entered.store(true, std::memory_order_release);
        while (!release_original.load(std::memory_order_acquire)) std::this_thread::yield();
      }
      if (mutate_cookie) for (UINT i = 0; i != count; ++i) static_cast<object *>(commands[i])->cookie = 9999;
    }
    SetLastError(outgoing_error);
  }
  template<unsigned Index> NOINLINE void STDMETHODCALLTYPE original_bundle(void *native, void *bundle) {
    ++original_bundles;
    error_preserved = error_preserved && GetLastError() == incoming_error;
    auto *value = static_cast<object *>(native);
    if constexpr (Index == 1) reinterpret_cast<bundle_fn>(value->forwarded->vtable[bundle_slot])(value->forwarded, bundle);
    SetLastError(outgoing_error);
  }
  NOINLINE void STDMETHODCALLTYPE original_enhanced_barrier(void *, UINT32, const D3D12_BARRIER_GROUP *) {
    ++original_enhanced;
    error_preserved = error_preserved && GetLastError() == incoming_error;
    SetLastError(outgoing_error);
  }
  NOINLINE void STDMETHODCALLTYPE original_begin_pass(void *, UINT, const D3D12_RENDER_PASS_RENDER_TARGET_DESC *, const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC *, D3D12_RENDER_PASS_FLAGS) {
    ++original_passes;
    error_preserved = error_preserved && GetLastError() == incoming_error;
    SetLastError(outgoing_error);
  }
  NOINLINE void STDMETHODCALLTYPE original_end_pass(void *) {
    ++original_passes;
    error_preserved = error_preserved && GetLastError() == incoming_error;
    SetLastError(outgoing_error);
  }
  constexpr std::array<execute_fn, 10> execute_originals{original_execute<0>, original_execute<1>, original_execute<2>, original_execute<3>, original_execute<4>,
    original_execute<5>, original_execute<6>, original_execute<7>, original_execute<8>, original_execute<9>};
  void on_barrier(std::uint64_t command, std::uint64_t cookie, unsigned count, const D3D12_RESOURCE_BARRIER *barriers) {
    ++barrier_events; observed_command = command; observed_cookie = cookie;
    if (trace_batch) batch_order.push_back('O');
    callback_after_original = callback_after_original && original_barriers > 0;
    observed_barriers.clear();
    if (count) observed_barriers.assign(barriers, barriers + count);
    if (count) observed_barrier = barriers[0];
    SetLastError(0xbadf00d);
  }
  void on_reset(std::uint64_t command, std::uint64_t cookie, HRESULT result) {
    ++reset_events; observed_command = command; observed_cookie = cookie; observed_result = result;
    if (SUCCEEDED(result)) observer::set_recording_cookie(command, cookie + 1);
    SetLastError(0xbadf00d);
  }
  void on_close(std::uint64_t command, std::uint64_t cookie, HRESULT result) {
    ++close_events; observed_command = command; observed_cookie = cookie; observed_result = result;
    SetLastError(0xbadf00d);
  }
  void on_submit(std::uint64_t queue, unsigned count, const observer::command_identity *commands) {
    ++submit_events; observed_queue = queue; observed_count = count;
    if (trace_batch) batch_order.push_back('S');
    callback_after_original = callback_after_original && original_submissions > 0;
    observed_commands.clear();
    if (count) observed_commands.assign(commands, commands + count);
    if (throw_callback) throw std::runtime_error("fixture callback failure");
    if (inject_capture) {
      // Internal barrier, Reset, Close and queue calls must not notify at all.
      D3D12_RESOURCE_BARRIER barrier{};
      SetLastError(incoming_error);
      reinterpret_cast<barrier_fn>(capture_command->vtable[barrier_slot])(capture_command, 1, &barrier);
      SetLastError(incoming_error);
      reinterpret_cast<reset_fn>(capture_command->vtable[reset_slot])(capture_command, nullptr, nullptr);
      SetLastError(incoming_error);
      reinterpret_cast<close_fn>(capture_command->vtable[close_slot])(capture_command);
      SetLastError(incoming_error);
      void *command = capture_command;
      reinterpret_cast<execute_fn>(capture_queue->vtable[execute_slot])(capture_queue, 1, &command);
    }
    SetLastError(0xbadf00d);
  }
  void on_invalidated() { ++invalidations; SetLastError(0xbadf00d); }
  void on_invalidated_command(std::uint64_t command, std::uint64_t cookie) {
    ++command_invalidations; observed_command = command; observed_cookie = cookie;
    SetLastError(0xbadf00d);
  }
  void on_render_pass(std::uint64_t command, std::uint64_t cookie, bool inside) {
    ++render_pass_events; observed_command = command; observed_cookie = cookie; inside_pass = inside;
    SetLastError(0xbadf00d);
  }
  void submit(object &queue, UINT count, void *const *commands) {
    SetLastError(incoming_error);
    reinterpret_cast<execute_fn>(queue.vtable[execute_slot])(&queue, count, commands);
    require(GetLastError() == outgoing_error, "post-submit callback changed outgoing LastError");
  }
  void barrier(object &command, UINT count, const D3D12_RESOURCE_BARRIER *barriers) {
    SetLastError(incoming_error);
    reinterpret_cast<barrier_fn>(command.vtable[barrier_slot])(&command, count, barriers);
    require(GetLastError() == outgoing_error, "post-barrier callback changed outgoing LastError");
  }
  HRESULT reset(object &command) {
    SetLastError(incoming_error);
    const auto result = reinterpret_cast<reset_fn>(command.vtable[reset_slot])(&command, nullptr, nullptr);
    require(GetLastError() == outgoing_error, "post-reset callback changed outgoing LastError");
    return result;
  }
  HRESULT close(object &command) {
    SetLastError(incoming_error);
    const auto result = reinterpret_cast<close_fn>(command.vtable[close_slot])(&command);
    require(GetLastError() == outgoing_error, "post-close callback changed outgoing LastError");
    return result;
  }
  std::uint64_t address(object &value) { return reinterpret_cast<std::uintptr_t>(&value); }
  void large_batch_cases(std::array<object, 2> &commands, std::array<object, 10> &queues) {
    // Exercise actual native hooks, including proxy unwrapping, with complete
    // valid calls beyond the old fixed snapshot capacities. Cookies change in
    // the original call to prove the callback uses the pre-submit recording.
    for (const unsigned count : {65u, 193u}) for (const bool nested : {false, true}) {
      std::vector<object> native_lists(count, commands[0]), proxy_lists(count, commands[1]);
      std::vector<void *> input(count);
      for (unsigned i = 0; i != count; ++i) {
        native_lists[i].cookie = 0x10000 + i;
        proxy_lists[i].forwarded = &native_lists[i];
        input[i] = nested ? &proxy_lists[i] : &native_lists[i];
      }
      const auto events_before = submit_events;
      const auto originals_before = original_submissions.load();
      const auto wrappers_before = original_forwarded_submissions;
      const auto invalid_before = invalidations;
      batch_order.clear(); trace_batch = true; mutate_cookie = true;
      submit(queues[nested ? 1 : 0], count, input.data());
      mutate_cookie = false; trace_batch = false;
      require(submit_events == events_before + 1 && original_submissions == originals_before + 1 &&
          original_forwarded_submissions == wrappers_before + (nested ? 1 : 0) && invalidations == invalid_before,
        "valid large submission was dropped, forwarded twice or notified by both proxy and native hooks");
      require(observed_queue == address(queues[0]) && observed_count == count && observed_commands.size() == count &&
          original_submit_count == count && (nested ? original_wrapper_pointer : original_submit_pointer) == input.data(),
        "large submission changed original arguments, selected the wrapper or truncated observed commands");
      require(batch_order == (nested ? std::vector<char>{'W', 'E', 'S'} : std::vector<char>{'E', 'S'}),
        "large submission callback was not exactly after the innermost original");
      for (unsigned i = 0; i != count; ++i) require(
          observed_commands[i].native_command == address(native_lists[i]) &&
          observed_commands[i].recording_cookie == 0x10000 + i && native_lists[i].cookie == 9999,
        "large submission lost command order or read a cookie after the original modified it");
    }
    for (const unsigned count : {257u, 1031u}) for (const bool nested : {false, true}) {
      std::vector<D3D12_RESOURCE_BARRIER> input(count);
      for (unsigned i = 0; i != count; ++i) {
        input[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        input[i].Flags = i % 2 ? D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY : D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
        input[i].Transition.pResource = reinterpret_cast<ID3D12Resource *>(std::uintptr_t{0x100000} + i);
        input[i].Transition.Subresource = i;
        input[i].Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        input[i].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      }
      const auto expected = input;
      const auto events_before = barrier_events;
      const auto originals_before = original_barriers.load();
      const auto invalid_before = invalidations;
      const auto expected_cookie = observer::get_recording_cookie(address(commands[0]));
      batch_order.clear(); trace_batch = true; mutate_barriers = true;
      barrier(commands[nested ? 1 : 0], count, input.data());
      mutate_barriers = false; trace_batch = false;
      require(barrier_events == events_before + 1 && original_barriers == originals_before + (nested ? 2 : 1) &&
          invalidations == invalid_before && observed_command == address(commands[0]) && observed_cookie == expected_cookie,
        "valid large barrier call was dropped, forwarded twice or notified by both proxy and native hooks");
      require(original_barrier_count == count && original_barrier_pointer == input.data() && observed_barriers.size() == count &&
          std::memcmp(observed_barriers.data(), expected.data(), count * sizeof(input[0])) == 0,
        "large barrier call changed original arguments or lost the pre-original ordered snapshot");
      require(batch_order == (nested ? std::vector<char>{'P', 'B', 'O'} : std::vector<char>{'B', 'O'}),
        "large barrier callback was not exactly after the innermost original");
      for (const auto &value : input) require(value.Transition.StateAfter == D3D12_RESOURCE_STATE_COPY_DEST,
        "large barrier mutation fixture did not exercise every original element");
    }
    // A readable prefix must never produce a partial notification when a later
    // pointer is unreadable. The original API still receives its complete call.
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    struct page_allocation {
      void *value{};
      ~page_allocation() { if (value) VirtualFree(value, 0, MEM_RELEASE); }
    } pages{VirtualAlloc(nullptr, 2 * system.dwPageSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)};
    require(pages.value && system.dwPageSize >= 64 * sizeof(void *), "guarded submission fixture allocation failed");
    DWORD old_protection{};
    auto *boundary = static_cast<unsigned char *>(pages.value) + system.dwPageSize;
    require(VirtualProtect(boundary, system.dwPageSize, PAGE_NOACCESS, &old_protection), "guarded submission fixture protection failed");
    auto **partial = reinterpret_cast<void **>(boundary - 64 * sizeof(void *));
    for (unsigned i = 0; i != 64; ++i) partial[i] = &commands[0];
    const auto events_before = submit_events;
    const auto originals_before = original_submissions.load();
    const auto invalid_before = invalidations;
    const auto unreadable_before = observer::counts().unreadable;
    submit(queues[0], 65, partial);
    require(submit_events == events_before && original_submissions == originals_before + 1 && invalidations == invalid_before + 1 &&
        observer::counts().unreadable == unreadable_before + 1 && original_submit_count == 65 && original_submit_pointer == partial,
      "partially readable command array published incomplete evidence or changed the original call");
    std::puts("PASS large native submissions/barrier arrays preserve complete pre-original snapshots, exact call order and innermost-only notifications");
  }
}
int main() {
  bool minhook = false;
  try {
    require(barrier_slot == 26 && reset_slot == 10 && close_slot == 9 && execute_slot == 10 && allocator_slot == close_slot,
      "D3D12 COM ABI differs from pinned headers");
    require(MH_Initialize() == MH_OK, "MinHook init failed"); minhook = true;
    const observer::callbacks callbacks{on_barrier, on_reset, on_close, on_submit, on_invalidated, on_invalidated_command, on_render_pass};
    observer::initialize(callbacks);
    static std::array<void *, enhanced_slot + 1> unrelated_table{};
    unrelated_table.fill(reinterpret_cast<void *>(unrelated_method));
    unrelated_table[0] = reinterpret_cast<void *>(query);
    unrelated_table[1] = reinterpret_cast<void *>(add_ref);
    unrelated_table[2] = reinterpret_cast<void *>(release);
    object unrelated{};
    unrelated.vtable = unrelated_table.data();
    unrelated.kind = object_kind::device;
    observer::set_active(true);
    SetLastError(incoming_error);
    observer::observe_command(address(unrelated)); observer::observe_queue(address(unrelated));
    require(GetLastError() == incoming_error && !observer::command_ready(address(unrelated)) && !observer::queue_ready(address(unrelated)) &&
        observer::counts().targets == 0 && observer::install_pending() && observer::counts().installed == 0,
      "device IUnknown registered or installed a command-list/queue hook");
    using unrelated_fn = HRESULT (STDMETHODCALLTYPE *)(void *, D3D12_COMMAND_LIST_TYPE, REFIID, void **);
    void *unrelated_output{};
    require(reinterpret_cast<unrelated_fn>(unrelated.vtable[allocator_slot])(&unrelated, D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_ID3D12CommandAllocator, &unrelated_output) == S_FALSE && unrelated_output == &unrelated &&
        GetLastError() == outgoing_error && unrelated_calls == 1,
      "rejected device CreateCommandAllocator was not forwarded unchanged");
    observer::set_active(false);
    static std::array<std::array<void *, enhanced_slot + 1>, 2> command_tables{};
    std::array<object, 2> commands{};
    for (unsigned i = 0; i != commands.size(); ++i) {
      command_tables[i][0] = reinterpret_cast<void *>(query); command_tables[i][1] = reinterpret_cast<void *>(add_ref);
      command_tables[i][2] = reinterpret_cast<void *>(release);
      command_tables[i][3] = reinterpret_cast<void *>(get_data); command_tables[i][4] = reinterpret_cast<void *>(set_data);
      command_tables[i][barrier_slot] = i ? reinterpret_cast<void *>(original_barrier<1>) : reinterpret_cast<void *>(original_barrier<0>);
      command_tables[i][reset_slot] = i ? reinterpret_cast<void *>(original_reset<1>) : reinterpret_cast<void *>(original_reset<0>);
      command_tables[i][close_slot] = i ? reinterpret_cast<void *>(original_close<1>) : reinterpret_cast<void *>(original_close<0>);
      command_tables[i][bundle_slot] = i ? reinterpret_cast<void *>(original_bundle<1>) : reinterpret_cast<void *>(original_bundle<0>);
      command_tables[i][enhanced_slot] = reinterpret_cast<void *>(original_enhanced_barrier);
      command_tables[i][begin_pass_slot] = reinterpret_cast<void *>(original_begin_pass);
      command_tables[i][end_pass_slot] = reinterpret_cast<void *>(original_end_pass);
      commands[i].vtable = command_tables[i].data();
      commands[i].kind = object_kind::command;
    }
    commands[1].forwarded = &commands[0];
    static std::array<std::array<void *, execute_slot + 1>, 10> queue_tables{};
    std::array<object, 10> queues{};
    for (unsigned i = 0; i != queues.size(); ++i) {
      queue_tables[i][0] = reinterpret_cast<void *>(query); queue_tables[i][1] = reinterpret_cast<void *>(add_ref);
      queue_tables[i][2] = reinterpret_cast<void *>(release);
      queue_tables[i][execute_slot] = reinterpret_cast<void *>(execute_originals[i]); queues[i].vtable = queue_tables[i].data();
      queues[i].kind = object_kind::queue;
    }
    queues[1].forwarded = &queues[0];
    object command_alias{}, queue_alias{};
    command_alias.vtable = queue_alias.vtable = unrelated_table.data();
    command_alias.queried = &commands[0]; queue_alias.queried = &queues[0];
    observer::observe_command(address(command_alias)); observer::observe_queue(address(queue_alias));
    require(observer::counts().targets == 5 && commands[0].references == 1 && queues[0].references == 1,
      "discovery did not normalize or release the QI-returned interfaces");
    for (auto &command : commands) observer::observe_command(address(command));
    observer::observe_queue(address(queues[0])); observer::observe_queue(address(queues[1]));
    observer::observe_queue(address(queues[0]));
    require(observer::counts().targets == 10 && !observer::install_pending(), "inactive discovery duplicated targets or installed hooks");
    require(!observer::command_ready(address(commands[0])) && !observer::queue_ready(address(queues[0])), "pending hooks reported ready");
    observer::set_active(true);
    require(observer::install_pending() && observer::counts().installed == 10, "deferred hook install failed");
    require(observer::command_ready(address(commands[1])) && observer::queue_ready(address(queues[1])), "installed hooks not ready");
    require(observer::command_ready(address(command_alias)) && observer::queue_ready(address(queue_alias)) &&
        commands[0].references == 1 && queues[0].references == 1 && command_alias.references == 1 && queue_alias.references == 1,
      "readiness did not normalize or release the QI-returned interfaces");
    require(reinterpret_cast<unrelated_fn>(command_alias.vtable[allocator_slot])(&command_alias, D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_ID3D12CommandAllocator, &unrelated_output) == S_FALSE && unrelated_output == &command_alias && unrelated_calls == 2,
      "normalization hooked an unrelated entry from the incoming interface");
    SetLastError(incoming_error);
    require(observer::set_recording_cookie(address(commands[1]), 1234) && observer::get_recording_cookie(address(commands[0])) == 1234 &&
        GetLastError() == incoming_error, "proxy/native private-data cookie or LastError failed");
    D3D12_RESOURCE_BARRIER native_barrier{};
    native_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    native_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
    native_barrier.Transition.pResource = reinterpret_cast<ID3D12Resource *>(std::uintptr_t{0x123456});
    native_barrier.Transition.Subresource = 3;
    native_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    native_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier(commands[1], 1, &native_barrier);
    require(barrier_events == 1 && original_barriers == 2 && observed_command == address(commands[0]) && observed_cookie == 1234 &&
        std::memcmp(&observed_barrier, &native_barrier, sizeof(native_barrier)) == 0, "nested barrier lost native fields or notified twice");
    native_result = E_FAIL;
    require(reset(commands[1]) == E_FAIL && reset_events == 1 && observed_result == E_FAIL && observed_cookie == 1234 &&
        observer::get_recording_cookie(address(commands[0])) == 1234, "failed Reset changed recording identity or HRESULT");
    native_result = S_OK;
    require(reset(commands[1]) == S_OK && reset_events == 2 && observed_cookie == 1234 &&
        observer::get_recording_cookie(address(commands[0])) == 1235, "successful Reset callback did not advance native recording once");
    native_result = E_FAIL;
    require(close(commands[1]) == E_FAIL && close_events == 1 && observed_result == E_FAIL && observed_cookie == 1235, "Close result or cookie lost");
    native_result = S_OK;
    require(close(commands[1]) == S_OK && close_events == 2, "successful Close not forwarded");
    void *submitted_command = &commands[1];
    mutate_cookie = true;
    submit(queues[1], 1, &submitted_command);
    mutate_cookie = false;
    require(submit_events == 1 && original_submissions == 1 && observed_queue == address(queues[0]) && observed_count == 1 &&
        observed_commands[0].native_command == address(commands[0]) && observed_commands[0].recording_cookie == 1235 &&
        observer::get_recording_cookie(address(commands[0])) == 9999, "post-submit snapshot read a later recording or wrapper notified twice");
    capture_command = &commands[0]; capture_queue = &queues[0]; inject_capture = true;
    submit(queues[0], 1, &submitted_command);
    inject_capture = false;
    require(submit_events == 2 && original_submissions == 3 && barrier_events == 1 && reset_events == 2 && close_events == 2 &&
        observer::counts().suppressed >= 4, "capture reentrancy changed source observations");
    const auto before_scope = submit_events;
    { observer::suppression_scope scope; submit(queues[0], 1, &submitted_command); barrier(commands[0], 1, &native_barrier); }
    require(submit_events == before_scope && barrier_events == 1, "explicit capture suppression failed");
    SetLastError(incoming_error);
    reinterpret_cast<bundle_fn>(commands[1].vtable[bundle_slot])(&commands[1], reinterpret_cast<void *>(1));
    require(GetLastError() == outgoing_error && command_invalidations == 1 && original_bundles == 2 &&
        observed_command == address(commands[0]) && observed_cookie == 9999, "ExecuteBundle did not invalidate native recording exactly once");
    commands[0].extended = true;
    require(!observer::command_ready(address(commands[0])), "unobserved supported CL4/CL7 hooks reported ready");
    observer::observe_command(address(commands[0]));
    require(!observer::command_ready(address(commands[0])) && observer::install_pending() && observer::command_ready(address(commands[0])),
      "optional interface discovery/readiness did not require deferred coverage");
    SetLastError(incoming_error);
    reinterpret_cast<enhanced_fn>(commands[0].vtable[enhanced_slot])(&commands[0], 1, reinterpret_cast<const D3D12_BARRIER_GROUP *>(1));
    require(GetLastError() == outgoing_error && command_invalidations == 2 && original_enhanced == 1 && observed_cookie == 9999,
      "enhanced Barrier did not invalidate native recording");
    SetLastError(incoming_error);
    reinterpret_cast<begin_pass_fn>(commands[0].vtable[begin_pass_slot])(&commands[0], 0, nullptr, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
    require(GetLastError() == outgoing_error && render_pass_events == 1 && inside_pass && observed_cookie == 9999, "render-pass begin was not observed");
    SetLastError(incoming_error);
    reinterpret_cast<end_pass_fn>(commands[0].vtable[end_pass_slot])(&commands[0]);
    require(GetLastError() == outgoing_error && render_pass_events == 2 && !inside_pass && original_passes == 2, "render-pass end was not observed");
    const auto before_invalid = invalidations;
    const auto barriers_before_invalid = original_barriers.load();
    const auto submissions_before_invalid = original_submissions.load();
    barrier(commands[0], 257, reinterpret_cast<const D3D12_RESOURCE_BARRIER *>(1));
    submit(queues[0], 65, reinterpret_cast<void *const *>(1));
    barrier(commands[0], 1, reinterpret_cast<const D3D12_RESOURCE_BARRIER *>(1));
    submit(queues[0], 1, reinterpret_cast<void *const *>(1));
    require(invalidations == before_invalid + 4 && barrier_events == 1 && submit_events == before_scope &&
        original_barriers == barriers_before_invalid + 2 && original_submissions == submissions_before_invalid + 2,
      "small and large unreadable input did not fail closed after exactly one original call");
    large_batch_cases(commands, queues);
    observer::observe_queue(1);
    for (unsigned i = 2; i != queues.size(); ++i) observer::observe_queue(address(queues[i]));
    require(observer::counts().targets == 19 && observer::counts().rejected >= 2 && !observer::queue_ready(address(queues[9])), "per-method target bounds failed");
    require(observer::install_pending(), "additional deferred queue hooks failed");
    for (unsigned i = 2; i != queues.size(); ++i) {
      submit(queues[i], 0, nullptr);
      require(last_queue_target == i, "per-target trampoline called wrong original");
    }
    block_original = true;
    std::thread running([&] { submit(queues[0], 1, &submitted_command); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool did_enter = entered.load(std::memory_order_acquire);
    const auto before_shutdown = submit_events;
    observer::shutdown();
    release_original = true; running.join(); block_original = false;
    require(did_enter && submit_events == before_shutdown, "shutdown dispatched an in-flight stale post-submit callback");
    submit(queues[0], 1, &submitted_command);
    require(submit_events == before_shutdown, "shutdown hook was not pure pass-through");
    observer::initialize(callbacks); observer::set_active(true);
    submit(queues[0], 1, &submitted_command);
    require(submit_events == before_shutdown + 1, "reinitialized observer lost retained hooks");
    throw_callback = true;
    const auto before_throw = original_submissions.load();
    submit(queues[0], 1, &submitted_command);
    throw_callback = false;
    require(original_submissions == before_throw + 1 && !observer::queue_ready(address(queues[0])), "callback exception escaped or observer did not deactivate");
    require(callback_after_original && error_preserved, "observer notified before original or changed incoming LastError");
    // Process detach can start inside another DLL while an original native
    // method is still on the stack. Keep its forwarding contract, but forbid
    // both that late completion and later cached entries from notifying owners.
    observer::initialize(callbacks); observer::set_active(true);
    block_original = true; entered = false; release_original = false;
    const auto detach_submissions = original_submissions.load();
    const auto detach_events = submit_events;
    std::thread detaching_call([&] { submit(queues[0], 1, &submitted_command); });
    const auto detach_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < detach_deadline) std::this_thread::yield();
    const bool entered_at_detach = entered.load(std::memory_order_acquire);
    const auto cached_submit = reinterpret_cast<execute_fn>(queues[0].vtable[execute_slot]);
    const auto cached_barrier = reinterpret_cast<barrier_fn>(commands[0].vtable[barrier_slot]);
    const auto detach_barriers = original_barriers.load();
    const auto detach_barrier_events = barrier_events;
    sunshine_addon_lifetime::begin_detach();
    release_original = true; detaching_call.join(); block_original = false;
    SetLastError(incoming_error); cached_submit(&queues[0], 1, &submitted_command);
    const bool submit_error_preserved = GetLastError() == outgoing_error;
    SetLastError(incoming_error); cached_barrier(&commands[0], 1, &native_barrier);
    const bool detach_ok = entered_at_detach && submit_error_preserved && GetLastError() == outgoing_error &&
      original_submissions == detach_submissions + 2 && original_barriers == detach_barriers + 1 &&
      submit_events == detach_events && barrier_events == detach_barrier_events &&
      !observer::command_ready(address(commands[0])) && !observer::queue_ready(address(queues[0]));
    // Test-only reset: the actual process detach fence is one-way.
    sunshine_addon_lifetime::detaching.store(false, std::memory_order_release);
    require(detach_ok, "process detach observed late native completion or changed cached original forwarding");
    std::puts("PASS process-detach fence preserves cached native forwarding and suppresses in-flight completion callbacks");
#if defined(__GNUC__) && defined(__x86_64__)
    // Reset target banks while every fixture is still alive, so this regression
    // is independent of the bounded-capacity tests above.
    observer::shutdown();
    require(MH_DisableHook(MH_ALL_HOOKS) == MH_OK, "fixture hook disable before shared thunk failed");
    require(MH_Uninitialize() == MH_OK, "fixture MinHook teardown before shared thunk failed"); minhook = false;
    observer::reset_after_hooks_removed();
    require(MH_Initialize() == MH_OK, "MinHook shared-thunk init failed"); minhook = true;
    observer::initialize(callbacks); observer::set_active(true);
    static std::array<void *, enhanced_slot + 1> shared_command_table{}, shared_device_table{};
    shared_command_table = command_tables[0];
    shared_device_table = unrelated_table;
    shared_command_table[close_slot] = shared_device_table[allocator_slot] = reinterpret_cast<void *>(shared_slot9);
    object shared_command{}, shared_device{};
    shared_command.vtable = shared_command_table.data(); shared_command.kind = object_kind::command;
    shared_command.forwarded = &commands[0];
    shared_device.vtable = shared_device_table.data(); shared_device.kind = object_kind::device;
    shared_device.forwarded = &unrelated;
    observer::observe_command(address(shared_command));
    require(observer::install_pending() && observer::command_ready(address(shared_command)),
      "shared-thunk command interface was not observed");
    const auto shared_close_before = close_events;
    SetLastError(incoming_error);
    require(reinterpret_cast<unrelated_fn>(shared_device.vtable[allocator_slot])(&shared_device, D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_ID3D12CommandAllocator, &unrelated_output) == S_FALSE && unrelated_output == &unrelated &&
        GetLastError() == outgoing_error && close_events == shared_close_before,
      "shared code interception changed the unrelated device method's signature or notification");
    require(close(shared_command) == S_OK && close_events == shared_close_before + 1,
      "scoped Close hook did not preserve the shared thunk's command-list forwarding");
    const auto retained_close = shared_command_table[close_slot];
    const auto invalidations_before_replacement = invalidations;
    shared_command_table[close_slot] = reinterpret_cast<void *>(shared_slot9);
    require(!observer::command_ready(address(shared_command)) && invalidations == invalidations_before_replacement + 1,
      "a replaced vtable slot incorrectly retained capture evidence");
    shared_command_table[close_slot] = retained_close;
    require(!observer::command_ready(address(shared_command)) && invalidations == invalidations_before_replacement + 1,
      "restoring a replaced slot revived evidence after missed commands");
    std::puts("PASS shared Streamline slot-9 thunk preserves device allocator arguments and separately observes command Close");
#endif
    observer::shutdown();
    require(MH_DisableHook(MH_ALL_HOOKS) == MH_OK, "fixture hook disable failed");
    require(MH_Uninitialize() == MH_OK, "fixture MinHook teardown failed"); minhook = false;
    observer::reset_after_hooks_removed();
    std::puts("PASS required COM interface validation/normalization, native submit order/cookies, nested proxies, barriers/results, bundle/enhanced/pass coverage, suppression, bounds, shutdown and LastError");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    observer::shutdown();
    if (minhook) { MH_DisableHook(MH_ALL_HOOKS); MH_Uninitialize(); }
    return 1;
  }
}
