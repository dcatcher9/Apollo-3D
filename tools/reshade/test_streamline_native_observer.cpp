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
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>
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
    bool cookie_present{true};
    object *enhanced{};
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
  unsigned barrier_events{}, submit_events{}, invalidations{};
  unsigned enhanced_reports{}, original_bundles{}, original_enhanced{}, original_passes{};
  unsigned observed_texture_count{};
  ID3D12Resource *observed_texture{};
  std::uint32_t observed_layout{}, observed_access{}, observed_sync_before{}, observed_sync_after{};
  bool observed_first_subresource{};
  unsigned last_queue_target{};
  HRESULT native_result = S_OK;
  std::uint64_t observed_command{}, observed_cookie{}, observed_queue{};
  std::vector<observer::command_identity> observed_commands;
  std::vector<D3D12_RESOURCE_BARRIER> observed_barriers;
  std::vector<char> batch_order;
  unsigned observed_count{};
  D3D12_RESOURCE_BARRIER observed_barrier{};
  bool error_preserved = true, callback_after_original = true, mutate_cookie{}, inject_capture{}, throw_callback{};
  bool trace_batch{}, mutate_barriers{};
  unsigned association_events{};
  bool check_entry_cookie{}, entry_cookie_present{};
  std::uint64_t replace_barrier_cookie{};
  unsigned original_submit_count{}, original_barrier_count{}, original_forwarded_submissions{};
  void *const *original_submit_pointer{}, *const *original_wrapper_pointer{};
  const D3D12_RESOURCE_BARRIER *original_barrier_pointer{};
  object *capture_command{}, *capture_queue{};
  unsigned unrelated_calls{};

  // -- Queue order (Step 4) fixture: queues, fences and a device with their
  // own private data by GUID, the queue's Signal (14) and Wait (15) and the
  // fence's CPU Signal (10). A queue Signal completes at once (a fast GPU).
  constexpr auto queue_signal_slot = offsetof(ID3D12CommandQueueVtbl, Signal) / sizeof(void *);
  constexpr auto queue_wait_slot = offsetof(ID3D12CommandQueueVtbl, Wait) / sizeof(void *);
  constexpr auto fence_signal_slot = offsetof(ID3D12FenceVtbl, Signal) / sizeof(void *);
  using queue_fence_fn = HRESULT (STDMETHODCALLTYPE *)(void *, void *, UINT64);
  using fence_signal_fn = HRESULT (STDMETHODCALLTYPE *)(void *, UINT64);
  struct order_object {
    void **vtable{};
    object_kind kind{};
    order_object *device{};
    std::uint64_t value{};
    D3D12_FENCE_FLAGS flags{};
    unsigned references{1};
    std::vector<std::pair<GUID, std::vector<unsigned char>>> data;
  };
  HRESULT queue_signal_result = S_OK, queue_wait_result = S_OK, cpu_signal_result = S_OK;
  std::atomic<bool> order_block{}, order_entered{}, order_release{};
  unsigned order_executions{}, order_signals{}, order_waits{}, order_cpu_signals{};
  NOINLINE HRESULT STDMETHODCALLTYPE order_query(void *native, REFIID iid, void **output) {
    auto *value = static_cast<order_object *>(native);
    *output = nullptr;
    if (IsEqualGUID(iid, IID_IUnknown) || (value->kind == object_kind::queue && IsEqualGUID(iid, IID_ID3D12CommandQueue)) ||
        (value->kind == object_kind::device && IsEqualGUID(iid, IID_ID3D12Device)) ||
        (value->kind == object_kind::unknown && (IsEqualGUID(iid, IID_ID3D12Fence) || IsEqualGUID(iid, IID_ID3D12Fence1)))) {
      *output = value;
      ++value->references;
      return S_OK;
    }
    return E_NOINTERFACE;
  }
  NOINLINE ULONG STDMETHODCALLTYPE order_add_ref(void *native) { return ++static_cast<order_object *>(native)->references; }
  NOINLINE ULONG STDMETHODCALLTYPE order_unref(void *native) { return --static_cast<order_object *>(native)->references; }
  NOINLINE HRESULT STDMETHODCALLTYPE order_get_data(void *native, REFGUID guid, UINT *size, void *output) {
    for (const auto &[key, bytes] : static_cast<order_object *>(native)->data)
      if (IsEqualGUID(key, guid)) {
        if (*size < bytes.size()) { *size = UINT(bytes.size()); return DXGI_ERROR_MORE_DATA; }
        std::memcpy(output, bytes.data(), bytes.size());
        *size = UINT(bytes.size());
        return S_OK;
      }
    return DXGI_ERROR_NOT_FOUND;
  }
  NOINLINE HRESULT STDMETHODCALLTYPE order_set_data(void *native, REFGUID guid, UINT size, const void *input) {
    auto &data = static_cast<order_object *>(native)->data;
    const auto *bytes = static_cast<const unsigned char *>(input);
    for (auto &[key, stored] : data)
      if (IsEqualGUID(key, guid)) { stored.assign(bytes, bytes + size); return S_OK; }
    data.emplace_back(guid, std::vector<unsigned char>(bytes, bytes + size));
    return S_OK;
  }
  NOINLINE HRESULT STDMETHODCALLTYPE order_get_device(void *native, REFIID, void **output) {
    auto *device = static_cast<order_object *>(native)->device;
    *output = device;
    if (!device) return E_FAIL;
    ++device->references;
    return S_OK;
  }
  // Runs once, inside the next GetCompletedValue (another thread's work
  // between a hooked call's steps).
  std::function<void()> order_on_completed;
  // A slow GPU: queue Signals do not complete.
  bool order_defer_signals{};
  NOINLINE UINT64 STDMETHODCALLTYPE order_completed(void *native) {
    if (auto hook = std::exchange(order_on_completed, {})) hook();
    return static_cast<order_object *>(native)->value;
  }
  NOINLINE D3D12_FENCE_FLAGS STDMETHODCALLTYPE order_flags(void *native) { return static_cast<order_object *>(native)->flags; }
  NOINLINE void STDMETHODCALLTYPE order_execute(void *, UINT, void *const *) {
    ++order_executions;
    if (order_block.load(std::memory_order_acquire)) {
      order_entered.store(true, std::memory_order_release);
      while (!order_release.load(std::memory_order_acquire)) std::this_thread::yield();
    }
  }
  NOINLINE HRESULT STDMETHODCALLTYPE order_queue_signal(void *, void *fence, UINT64 value) {
    ++order_signals;
    if (SUCCEEDED(queue_signal_result) && !order_defer_signals) static_cast<order_object *>(fence)->value = value;
    return queue_signal_result;
  }
  NOINLINE HRESULT STDMETHODCALLTYPE order_queue_wait(void *, void *, UINT64) { ++order_waits; return queue_wait_result; }
  NOINLINE HRESULT STDMETHODCALLTYPE order_cpu_signal(void *fence, UINT64 value) {
    ++order_cpu_signals;
    if (SUCCEEDED(cpu_signal_result)) static_cast<order_object *>(fence)->value = value;
    return cpu_signal_result;
  }

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
    if (value->enhanced && IsEqualGUID(iid, IID_ID3D12GraphicsCommandList7)) {
      *output = value->enhanced;
      ++value->enhanced->references;
      return S_OK;
    }
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
    if (!value->cookie_present) { SetLastError(0xbadf00d); return DXGI_ERROR_NOT_FOUND; }
    std::memcpy(output, &value->cookie, sizeof(value->cookie));
    SetLastError(0xbadf00d);
    return S_OK;
  }
  NOINLINE HRESULT STDMETHODCALLTYPE set_data(ID3D12Object *native, REFGUID, UINT size, const void *input) {
    auto *value = reinterpret_cast<object *>(native);
    if (value->forwarded) value = value->forwarded;
    if (size != sizeof(value->cookie) || !input) return E_INVALIDARG;
    std::memcpy(&value->cookie, input, sizeof(value->cookie));
    value->cookie_present = true;
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
      if (check_entry_cookie) entry_cookie_present = value->cookie_present && value->cookie != 0;
      if (replace_barrier_cookie) value->cookie = replace_barrier_cookie;
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
  void on_associate_recording(std::uint64_t command, std::uint64_t *cookie) {
    ++association_events;
    *cookie = 0;
    if (observer::recording_cookie_absent(command) && observer::set_recording_cookie(command, 100000 + association_events))
      *cookie = 100000 + association_events;
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
  // The submission listener (the UI layer's cross-queue fence).
  unsigned listen_events{}, listened_count{}, listened_owner_events{}, listened_originals{};
  std::uint64_t listened_queue{}, listened_command{};
  void on_listen(std::uint64_t queue, unsigned count, const observer::command_identity *commands) noexcept {
    ++listen_events; listened_queue = queue; listened_count = count;
    listened_command = count ? commands[0].native_command : 0;
    listened_owner_events = submit_events; listened_originals = original_submissions;
    SetLastError(0xbadf00d);
  }
  void on_enhanced_textures(std::uint64_t command, std::uint64_t cookie, unsigned count, const observer::enhanced_texture *textures) {
    ++enhanced_reports; observed_command = command; observed_cookie = cookie;
    observed_texture_count = count; observed_texture = count ? textures[0].resource : nullptr;
    observed_layout = count ? textures[0].layout_after : UINT32_MAX;
    observed_first_subresource = count && textures[0].first_subresource;
    observed_access = count ? textures[0].access_after : UINT32_MAX;
    observed_sync_before = count ? textures[0].sync_before : UINT32_MAX;
    observed_sync_after = count ? textures[0].sync_after : UINT32_MAX;
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

  // Queue order (Step 4): the order family's hooks on COM-shaped fixtures.
  void order_cases() {
    observer::set_queue_order(true);
    unsigned method_count{};
    const auto *methods = observer::hooked_methods(method_count);
    require(method_count == 6 && methods[3].slot == 14 && methods[4].slot == 15 && methods[5].slot == 10 &&
        std::strcmp(methods[3].method, "Signal") == 0 && std::strcmp(methods[4].method, "Wait") == 0 &&
        std::strcmp(methods[5].interface_name, "ID3D12Fence") == 0, "the hooked method table changed");
    static std::array<void *, queue_wait_slot + 1> queue_table{}, unordered_table{};
    static std::array<void *, 12> fence_table{};
    static std::array<void *, 3> device_table{};
    for (auto *table : {queue_table.data(), unordered_table.data(), fence_table.data(), device_table.data()}) {
      table[0] = reinterpret_cast<void *>(order_query); table[1] = reinterpret_cast<void *>(order_add_ref);
      table[2] = reinterpret_cast<void *>(order_unref);
    }
    for (auto *table : {queue_table.data(), unordered_table.data(), fence_table.data()}) {
      table[3] = reinterpret_cast<void *>(order_get_data); table[4] = reinterpret_cast<void *>(order_set_data);
      table[7] = reinterpret_cast<void *>(order_get_device);
    }
    for (auto *table : {queue_table.data(), unordered_table.data()}) {
      table[execute_slot] = reinterpret_cast<void *>(order_execute);
      table[queue_signal_slot] = reinterpret_cast<void *>(order_queue_signal);
      table[queue_wait_slot] = reinterpret_cast<void *>(order_queue_wait);
    }
    fence_table[8] = reinterpret_cast<void *>(order_completed);
    fence_table[fence_signal_slot] = reinterpret_cast<void *>(order_cpu_signal);
    fence_table[11] = reinterpret_cast<void *>(order_flags);
    order_object device{device_table.data(), object_kind::device};
    order_object render{queue_table.data(), object_kind::queue, &device}, present{queue_table.data(), object_kind::queue, &device};
    order_object fence{fence_table.data(), object_kind::unknown, &device}, shared{fence_table.data(), object_kind::unknown, &device};
    shared.flags = D3D12_FENCE_FLAG_SHARED;
    const auto execute = [](order_object &queue) {
      reinterpret_cast<execute_fn>(queue.vtable[execute_slot])(&queue, 0, nullptr);
      return observer::last_execute_position();
    };
    const auto signal = [](order_object &queue, order_object &on, UINT64 value) {
      return reinterpret_cast<queue_fence_fn>(queue.vtable[queue_signal_slot])(&queue, &on, value);
    };
    const auto wait = [](order_object &queue, order_object &on, UINT64 value) {
      return reinterpret_cast<queue_fence_fn>(queue.vtable[queue_wait_slot])(&queue, &on, value);
    };
    const auto ordered = [&](order_object &reader, const observer::position &at) {
      return observer::queue_ordered(reinterpret_cast<std::uintptr_t>(&reader), at);
    };
    observer::observe_queue(reinterpret_cast<std::uintptr_t>(&render));
    require(observer::install_pending(), "the queue's order slots were not installed");
    auto status = observer::queue_order_counts();
    require(status.enabled && status.signal_hooks == 1 && status.wait_hooks == 1 && status.fence_signal_hooks == 0,
      "observe_queue did not discover the queue's Signal and Wait");
    // A fence is first seen in a Signal: its CPU Signal slot is not hooked
    // yet, so it gets no record, and verdicts refuse while that slot waits.
    const auto invalid_before = invalidations;
    require(signal(render, fence, 1) == S_OK && fence.value == 1 && observer::queue_order_counts().cpu_unhooked == 1 &&
        observer::queue_order_counts().pending == 1, "a fence was recorded before its CPU Signal slot was hooked");
    require(ordered(present, execute(render)).why == sunshine_queue_order::refusal::pending_hooks,
      "a verdict was given while an order slot was pending");
    require(observer::install_pending() && observer::queue_order_counts().fence_signal_hooks == 1 && invalidations == invalid_before,
      "the fence's CPU Signal slot was not installed, or order coverage revoked capture evidence");
    // Settle (the first post-sight signal completes), then an edge.
    require(signal(render, fence, 2) == S_OK, "settling signal failed");
    const auto producer = execute(render);
    require(producer && producer.quiet, "a sequential ExecuteCommandLists had no quiet position");
    require(signal(render, fence, 3) == S_OK && wait(present, fence, 3) == S_OK, "fence calls failed");
    auto verdict = ordered(present, producer);
    require(verdict.ordered && verdict.fences[0], "a reader waiting on the producer's later signal is not ordered after it");
    require(!ordered(present, execute(render)).ordered, "a later submission is ordered before a wait that preceded it");
    // Cost: each hooked call against its original on these fixtures (their
    // private data is a short list, as the runtime's is), on a fence of its own.
    {
      order_object bench{fence_table.data(), object_kind::unknown, &device};
      constexpr unsigned calls = 100000;
      const auto per_call = [](auto &&call) {
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i != calls; ++i) call(i);
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / calls;
      };
      require(signal(render, bench, 1) == S_OK, "bench fence sight failed");
      const auto hooked_signal = per_call([&](unsigned i) { signal(render, bench, i + 2); });
      const auto original_signal = per_call([&](unsigned i) { order_queue_signal(&render, &bench, calls + i + 2); });
      const auto hooked_wait = per_call([&](unsigned i) { wait(present, bench, i + 2); });
      const auto original_wait = per_call([&](unsigned i) { order_queue_wait(&present, &bench, i + 2); });
      const auto hooked_execute = per_call([&](unsigned) { execute(render); });
      const auto original_execute = per_call([&](unsigned) { order_execute(&render, 0, nullptr); });
      std::printf("MEASURE queue-order hook ns/call: signal=%.0f (original %.0f) wait=%.0f (original %.0f) execute=%.0f (original %.0f)\n",
        hooked_signal, original_signal, hooked_wait, original_wait, hooked_execute, original_execute);
    }
    // A suppressed call (the add-on's copy-out) takes its own ticket.
    const auto game = execute(render);
    observer::position own;
    { observer::suppression_scope scope; own = execute(render); }
    require(own && own.ticket == game.ticket + 1 && own.queue == game.queue, "a suppressed ExecuteCommandLists took no ticket of its own");
    // The quiet bit: an earlier call still in its native code when another
    // thread takes a ticket on the same queue is not quiet.
    observer::position blocked;
    order_block = true; order_entered = false; order_release = false;
    std::thread other([&] { blocked = execute(render); });
    while (!order_entered.load(std::memory_order_acquire)) std::this_thread::yield();
    order_block = false;
    const auto overtaking = execute(render);
    order_release = true;
    other.join();
    require(blocked && !blocked.quiet && overtaking.quiet && overtaking.ticket == blocked.ticket + 1,
      "overlapping calls on one queue were not told apart by the quiet bit");
    require(signal(render, fence, 4) == S_OK && wait(present, fence, 4) == S_OK &&
        ordered(present, blocked).why == sunshine_queue_order::refusal::overlap, "a call that was not quiet was the earlier side of an edge");
    // A failed Wait is never logged.
    const auto waits_before = observer::queue_order_counts().model.waits;
    queue_wait_result = E_FAIL;
    require(wait(present, fence, 4) == E_FAIL && observer::queue_order_counts().model.waits == waits_before, "a failed Wait was logged");
    queue_wait_result = S_OK;
    // A shared fence never proves.
    require(signal(render, shared, 1) == S_OK && signal(render, shared, 2) == S_OK, "shared fence signals failed");
    const auto before_shared = execute(render);
    require(signal(render, shared, 3) == S_OK && wait(present, shared, 3) == S_OK && !ordered(present, before_shared).ordered &&
        observer::queue_order_counts().model.poisoned[static_cast<unsigned>(sunshine_queue_order::poison::shared)] == 1,
      "a shared fence proved an edge");
    // A CPU Signal poisons, even one that fails; the poison outlives the
    // model's record of the fence (its private data).
    cpu_signal_result = E_FAIL;
    require(reinterpret_cast<fence_signal_fn>(fence.vtable[fence_signal_slot])(&fence, 99) == E_FAIL, "the CPU Signal result changed");
    cpu_signal_result = S_OK;
    status = observer::queue_order_counts();
    require(status.model.cpu_signals == 1 && status.model.poisoned[static_cast<unsigned>(sunshine_queue_order::poison::cpu_signal)] == 1 &&
        !ordered(present, producer).ordered, "a failed CPU Signal did not poison its fence");
    observer::queue_order_forget_device(reinterpret_cast<std::uintptr_t>(&device));
    const auto after_forget = execute(render);
    require(signal(render, fence, 100) == S_OK && wait(present, fence, 100) == S_OK && !ordered(present, after_forget).ordered,
      "a fence re-sighted after its records were dropped lost its poison");
    // A third party replacing the Signal slot resets the order model only.
    const auto resets_before = observer::queue_order_counts().resets;
    const auto evidence_before = invalidations;
    const auto detour = queue_table[queue_signal_slot];
    queue_table[queue_signal_slot] = reinterpret_cast<void *>(order_queue_signal);
    observer::install_pending();
    status = observer::queue_order_counts();
    require(status.signal_hooks == 0 && invalidations == evidence_before, "a replaced Signal slot kept its coverage or revoked capture evidence");
    require(wait(present, fence, 100) == S_OK && observer::queue_order_counts().resets == resets_before + 1,
      "a replaced Signal slot did not reset the order model");
    require(wait(present, fence, 100) == S_OK && ordered(present, execute(render)).why == sunshine_queue_order::refusal::uncovered,
      "an uncovered queue proved an edge");
    queue_table[queue_signal_slot] = detour;
    // QueueFenceOrder=0: no order slot is discovered, Execute takes no ticket.
    observer::set_queue_order(false);
    order_object unordered{unordered_table.data(), object_kind::queue, &device};
    const auto targets_before = observer::counts().targets;
    observer::observe_queue(reinterpret_cast<std::uintptr_t>(&unordered));
    require(observer::counts().targets == targets_before + 1 && observer::install_pending(), "QueueFenceOrder=0 discovered an order slot");
    require(!execute(unordered) && !execute(render) &&
        ordered(present, producer).why == sunshine_queue_order::refusal::disabled, "QueueFenceOrder=0 still took tickets or gave verdicts");
    std::puts("PASS queue order: Signal/Wait/CPU Signal hooks, tickets and the quiet bit, suppressed tickets, failed calls, shared and CPU-signalled fences, persisted poison, slot replacement and QueueFenceOrder=0");
  }

  // The order host's review cells (10-27), on vtables of their own.
  struct order_tables {
    std::array<void *, queue_wait_slot + 1> queue{};
    std::array<void *, 12> fence{};
    std::array<void *, 3> device{};
    order_tables() {
      for (auto *table : {queue.data(), fence.data(), device.data()}) {
        table[0] = reinterpret_cast<void *>(order_query); table[1] = reinterpret_cast<void *>(order_add_ref);
        table[2] = reinterpret_cast<void *>(order_unref);
      }
      for (auto *table : {queue.data(), fence.data()}) {
        table[3] = reinterpret_cast<void *>(order_get_data); table[4] = reinterpret_cast<void *>(order_set_data);
        table[7] = reinterpret_cast<void *>(order_get_device);
      }
      queue[execute_slot] = reinterpret_cast<void *>(order_execute);
      queue[queue_signal_slot] = reinterpret_cast<void *>(order_queue_signal);
      queue[queue_wait_slot] = reinterpret_cast<void *>(order_queue_wait);
      fence[8] = reinterpret_cast<void *>(order_completed);
      fence[fence_signal_slot] = reinterpret_cast<void *>(order_cpu_signal);
      fence[11] = reinterpret_cast<void *>(order_flags);
    }
  };
  void order_review_cases() {
    observer::set_queue_order(true);
    static order_tables tables;
    const auto execute = [](order_object &queue) {
      reinterpret_cast<execute_fn>(queue.vtable[execute_slot])(&queue, 0, nullptr);
      return observer::last_execute_position();
    };
    const auto signal = [](order_object &queue, order_object &on, UINT64 value) {
      return reinterpret_cast<queue_fence_fn>(queue.vtable[queue_signal_slot])(&queue, &on, value);
    };
    const auto wait = [](order_object &queue, order_object &on, UINT64 value) {
      return reinterpret_cast<queue_fence_fn>(queue.vtable[queue_wait_slot])(&queue, &on, value);
    };
    const auto ordered = [](order_object &reader, const observer::position &at) {
      return observer::queue_ordered(reinterpret_cast<std::uintptr_t>(&reader), at);
    };
    order_object device{tables.device.data(), object_kind::device};
    order_object producer{tables.queue.data(), object_kind::queue, &device}, reader{tables.queue.data(), object_kind::queue, &device};
    order_object fence{tables.fence.data(), object_kind::unknown, &device};
    observer::observe_queue(reinterpret_cast<std::uintptr_t>(&producer));
    require(observer::install_pending(), "the review queue's order slots were not installed");
    require(signal(producer, fence, 1) == S_OK && observer::install_pending(), "the review fence's CPU Signal slot was not installed");
    // The installation's model reset applies at the next locked call: a
    // reader's first wait (unsettled, refused). Then the fence settles.
    require(wait(reader, fence, 1) == S_OK, "the reader's first wait failed");
    const auto settling = execute(producer);
    require(signal(producer, fence, 2) == S_OK && wait(reader, fence, 2) == S_OK && ordered(reader, settling).ordered,
      "the review fence did not settle");

    // A queue record evicted between a Signal's ticket and its log: the
    // native call is not forwarded yet, so the signal is logged on the
    // re-sighted record. Dropped, the reader's wait for it would resolve to
    // the next signal, after a copy-out that this one precedes.
    order_defer_signals = true;
    require(signal(producer, fence, 3) == S_OK, "a slow signal failed");
    std::vector<order_object> others(32, order_object{tables.queue.data(), object_kind::queue, &device});
    order_on_completed = [&] { for (auto &other : others) execute(other); };
    require(signal(producer, fence, 4) == S_OK && !order_on_completed, "the evicting sightings did not run inside the Signal");
    require(wait(reader, fence, 4) == S_OK, "the reader's wait failed");
    const auto copy = execute(producer);
    require(copy && copy.quiet && signal(producer, fence, 5) == S_OK, "the re-sighted producer took no position");
    require(!ordered(reader, copy).ordered,
      "a Signal dropped with its evicted queue record let the reader's wait resolve to a signal after the copy-out");
    require(wait(reader, fence, 5) == S_OK && ordered(reader, copy).ordered, "a wait for the signal after the copy-out did not order it");
    order_defer_signals = false;
    std::puts("PASS queue order review: a Signal whose queue record was evicted after its ticket is logged on the re-sighted record");
  }
}
int main() {
  bool minhook = false;
  try {
    require(barrier_slot == 26 && reset_slot == 10 && close_slot == 9 && execute_slot == 10 && allocator_slot == close_slot,
      "D3D12 COM ABI differs from pinned headers");
    require(MH_Initialize() == MH_OK, "MinHook init failed"); minhook = true;
    const observer::callbacks callbacks{on_barrier, on_submit, on_invalidated, on_enhanced_textures, on_associate_recording};
    observer::initialize(callbacks);
    // The state family's cells below run without the order family (its own
    // cells are order_cases).
    observer::set_queue_order(false);
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
    require(GetLastError() == incoming_error && !observer::states_ready(address(unrelated)) && !observer::queue_ready(address(unrelated)) &&
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
    // Command lists hook only ResourceBarrier (and CL7 Barrier); their
    // lifecycle comes from ReShade's events.
    require(observer::counts().targets == 2 && commands[0].references == 1 && queues[0].references == 1,
      "discovery did not normalize or release the QI-returned interfaces");
    for (auto &command : commands) observer::observe_command(address(command));
    observer::observe_queue(address(queues[0])); observer::observe_queue(address(queues[1]));
    observer::observe_queue(address(queues[0]));
    require(observer::counts().targets == 4 && !observer::install_pending(), "inactive discovery duplicated targets or installed hooks");
    require(!observer::states_ready(address(commands[0])) && !observer::queue_ready(address(queues[0])), "pending hooks reported ready");
    observer::set_active(true);
    require(observer::install_pending() && observer::counts().installed == 4, "deferred hook install failed");
    require(observer::states_ready(address(commands[1])) && observer::queue_ready(address(queues[1])), "installed hooks not ready");
    require(observer::states_ready(address(command_alias)) && observer::queue_ready(address(queue_alias)) &&
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
    // Reset and Close are not hooked: ReShade's lifecycle events own the
    // recording. They pass through unchanged and leave the identity alone.
    const auto resets_before = original_resets.load(), closes_before = original_closes.load();
    native_result = E_FAIL;
    require(reset(commands[1]) == E_FAIL && close(commands[1]) == E_FAIL && barrier_events == 1 &&
        observer::get_recording_cookie(address(commands[0])) == 1234, "unhooked Reset/Close changed an HRESULT or the identity");
    native_result = S_OK;
    require(reset(commands[1]) == S_OK && close(commands[1]) == S_OK &&
        original_resets == resets_before + 4 && original_closes == closes_before + 4 &&
        observer::get_recording_cookie(address(commands[0])) == 1234, "unhooked Reset/Close were not forwarded through the proxy once each");
    observer::set_recording_cookie(address(commands[0]), 1235); // As ReShade's reset event would.
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
    require(submit_events == 2 && original_submissions == 3 && barrier_events == 1 &&
        observer::counts().suppressed >= 2, "capture reentrancy changed source observations");
    const auto before_scope = submit_events;
    { observer::suppression_scope scope; submit(queues[0], 1, &submitted_command); barrier(commands[0], 1, &native_barrier); }
    require(submit_events == before_scope && barrier_events == 1, "explicit capture suppression failed");
    SetLastError(incoming_error);
    reinterpret_cast<bundle_fn>(commands[1].vtable[bundle_slot])(&commands[1], reinterpret_cast<void *>(1));
    require(GetLastError() == outgoing_error && enhanced_reports == 0 && original_bundles == 2,
      "ExecuteBundle (a ReShade lifecycle event) was hooked or not forwarded through the proxy");
    commands[0].extended = true;
    require(!observer::states_ready(address(commands[0])), "unobserved supported CL7 Barrier reported ready");
    observer::observe_command(address(commands[0]));
    require(!observer::states_ready(address(commands[0])) && observer::install_pending() && observer::states_ready(address(commands[0])),
      "enhanced Barrier discovery/readiness did not require deferred coverage");
    // Only texture barriers can change a texture's layout: the hook reports
    // exactly the textures they name, never global or buffer barriers.
    const auto enhanced = reinterpret_cast<enhanced_fn>(commands[0].vtable[enhanced_slot]);
    int named_texture_storage{};
    D3D12_TEXTURE_BARRIER texture_barrier{};
    texture_barrier.pResource = reinterpret_cast<ID3D12Resource *>(&named_texture_storage);
    texture_barrier.LayoutAfter = D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ;
    texture_barrier.AccessAfter = D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ;
    texture_barrier.SyncBefore = D3D12_BARRIER_SYNC_ALL; texture_barrier.SyncAfter = D3D12_BARRIER_SYNC_SPLIT;
    texture_barrier.Subresources.IndexOrFirstMipLevel = UINT32_MAX; // Every subresource.
    D3D12_GLOBAL_BARRIER global_barrier{};
    D3D12_BARRIER_GROUP groups[2]{};
    groups[0].Type = D3D12_BARRIER_TYPE_GLOBAL; groups[0].NumBarriers = 1; groups[0].pGlobalBarriers = &global_barrier;
    groups[1].Type = D3D12_BARRIER_TYPE_TEXTURE; groups[1].NumBarriers = 1; groups[1].pTextureBarriers = &texture_barrier;
    SetLastError(incoming_error);
    enhanced(&commands[0], 2, groups);
    require(GetLastError() == outgoing_error && enhanced_reports == 1 && observed_texture_count == 1 &&
        observed_texture == texture_barrier.pResource && original_enhanced == 1 && observed_cookie == 9999 &&
        observed_layout == D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ && observed_first_subresource &&
        observed_access == D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ && observed_sync_before == D3D12_BARRIER_SYNC_ALL &&
        observed_sync_after == D3D12_BARRIER_SYNC_SPLIT,
      "enhanced Barrier did not report exactly its named texture, layout, access, sync and subresource range");
    SetLastError(incoming_error);
    enhanced(&commands[0], 1, groups);
    require(enhanced_reports == 1 && original_enhanced == 2, "a global-only enhanced Barrier reported a texture");
    // An unreadable barrier array cannot name its textures: fail closed.
    const auto unreadable_before = invalidations;
    SetLastError(incoming_error);
    enhanced(&commands[0], 1, reinterpret_cast<const D3D12_BARRIER_GROUP *>(1));
    require(invalidations == unreadable_before + 1 && enhanced_reports == 1 && original_enhanced == 3,
      "an unreadable enhanced Barrier did not fail closed or was not forwarded");
    {
      // D3D12Core can return CommandList7 as a per-object interface whose
      // table lives in heap memory. That slot is refused once, and the list's
      // states stay unobserved (its declared state applies).
      std::vector<void *> heap_table(commands[0].vtable, commands[0].vtable + enhanced_slot + 1);
      object torn_off{}; torn_off.vtable = heap_table.data(); torn_off.kind = object_kind::command;
      object split{}; split.vtable = commands[0].vtable; split.kind = object_kind::command;
      split.extended = true; split.enhanced = &torn_off; split.cookie = 9999;
      const auto invalid_before = invalidations;
      for (unsigned repeat = 0; repeat != 3; ++repeat) observer::observe_command(address(split));
      require(invalidations == invalid_before + 1 && !observer::states_ready(address(split)) &&
          split.references == 1 && torn_off.references == 1,
        "per-object CommandList7 table was not refused once, reported observed states, or leaked references");
    }
    // Render passes are ReShade lifecycle events too: forwarded, never hooked.
    SetLastError(incoming_error);
    reinterpret_cast<begin_pass_fn>(commands[0].vtable[begin_pass_slot])(&commands[0], 0, nullptr, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
    require(GetLastError() == outgoing_error, "render-pass begin changed LastError");
    SetLastError(incoming_error);
    reinterpret_cast<end_pass_fn>(commands[0].vtable[end_pass_slot])(&commands[0]);
    require(GetLastError() == outgoing_error && original_passes == 2 && enhanced_reports == 1,
      "render passes were hooked or not forwarded");
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
    object first{}; first.vtable = commands[0].vtable; first.kind = object_kind::command; first.cookie_present = false;
    object first_proxy{}; first_proxy.vtable = commands[1].vtable; first_proxy.kind = object_kind::command; first_proxy.forwarded = &first;
    const auto associations_before = association_events;
    const auto events_before_first = barrier_events;
    check_entry_cookie = true;
    barrier(first_proxy, 1, &native_barrier);
    check_entry_cookie = false;
    const auto first_cookie = observer::get_recording_cookie(address(first));
    require(first_cookie && entry_cookie_present && association_events == associations_before + 1 &&
        barrier_events == events_before_first + 1 && observed_cookie == first_cookie && observed_command == address(first),
      "first barrier without Reset did not associate the forwarded native recording before the original call");
    replace_barrier_cookie = first_cookie + 100;
    barrier(first, 1, &native_barrier);
    replace_barrier_cookie = 0;
    require(association_events == associations_before + 1 && observed_cookie == first_cookie &&
        observer::get_recording_cookie(address(first)) == first_cookie + 100,
      "post-call observation rebound to a replacement recording instead of retaining its entry cookie");
    object malformed{}; malformed.vtable = commands[0].vtable; malformed.kind = object_kind::command;
    barrier(malformed, 1, &native_barrier);
    require(association_events == associations_before + 1 && observed_cookie == 0 && !observer::recording_cookie_absent(address(malformed)),
      "malformed zero private data was mistaken for an absent first recording");
    object never_recorded{}; never_recorded.vtable = commands[0].vtable; never_recorded.kind = object_kind::command; never_recorded.cookie_present = false;
    void *never_recorded_pointer = &never_recorded;
    submit(queues[0], 1, &never_recorded_pointer);
    require(association_events == associations_before + 1 && observed_commands.size() == 1 && observed_commands[0].recording_cookie == 0 &&
        observer::recording_cookie_absent(address(never_recorded)), "submission invented an unobserved producer recording");
    { observer::suppression_scope scope; barrier(never_recorded, 1, &native_barrier); }
    require(association_events == associations_before + 1 && observer::recording_cookie_absent(address(never_recorded)),
      "suppressed native work associated a source recording");
    std::puts("PASS first native recording associates before forwarding, freezes identity, and rejects malformed/suppressed/submission bootstrap");
    observer::observe_queue(1);
    for (unsigned i = 2; i != queues.size(); ++i) observer::observe_queue(address(queues[i]));
    require(observer::counts().targets == 11 && observer::counts().rejected >= 2 && !observer::queue_ready(address(queues[9])), "per-method target bounds failed");
    {
      // A game submits through the same refused vtable every frame. Repeating
      // a known refusal must not revoke every capture on every call.
      const auto rejected_before = observer::counts().rejected;
      const auto invalid_before = invalidations;
      for (unsigned repeat = 0; repeat != 3; ++repeat) observer::observe_queue(address(queues[9]));
      require(observer::counts().rejected == rejected_before && invalidations == invalid_before &&
          !observer::queue_ready(address(queues[9])),
        "known refused vtable was reported again or became ready");
    }
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
    {
      // The submission listener sees each submission once, innermost, after
      // the original and the capture owner's callback, with the same native
      // identities; never under suppression nor once removed.
      observer::set_submission_listener(on_listen);
      const auto listened = listen_events;
      submit(queues[1], 1, &submitted_command);
      require(listen_events == listened + 1 && listened_queue == address(queues[0]) && listened_count == 1 &&
          listened_command == address(commands[0]) && listened_owner_events == submit_events &&
          listened_originals == original_submissions, "the submission listener missed, repeated or preceded a submission");
      { observer::suppression_scope scope; submit(queues[0], 1, &submitted_command); }
      require(listen_events == listened + 1, "the submission listener ran under suppression");
      observer::set_submission_listener(nullptr);
      submit(queues[0], 1, &submitted_command);
      require(listen_events == listened + 1, "a removed submission listener still ran");
    }
    std::puts("PASS submission listener: once per submission, innermost, after the original and the owner's callback, never suppressed or after removal");
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
      !observer::states_ready(address(commands[0])) && !observer::queue_ready(address(queues[0]));
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
    observer::set_queue_order(false);
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
    require(observer::install_pending() && observer::states_ready(address(shared_command)),
      "shared-thunk command interface was not observed");
    // Close is no longer hooked, so the shared slot-9 code is never touched:
    // both the device method and the command-list Close forward unchanged.
    const auto shared_closes_before = original_closes.load();
    SetLastError(incoming_error);
    require(reinterpret_cast<unrelated_fn>(shared_device.vtable[allocator_slot])(&shared_device, D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_ID3D12CommandAllocator, &unrelated_output) == S_FALSE && unrelated_output == &unrelated &&
        GetLastError() == outgoing_error && shared_command_table[close_slot] == reinterpret_cast<void *>(shared_slot9),
      "the shared slot-9 thunk was intercepted or changed the unrelated device method");
    require(close(shared_command) == S_OK && original_closes == shared_closes_before + 1,
      "the shared thunk's command-list Close was not forwarded");
    const auto retained_barrier = shared_command_table[barrier_slot];
    const auto invalidations_before_replacement = invalidations;
    shared_command_table[barrier_slot] = reinterpret_cast<void *>(original_barrier<0>);
    require(!observer::states_ready(address(shared_command)) && invalidations == invalidations_before_replacement + 1,
      "a replaced barrier slot incorrectly retained capture evidence");
    shared_command_table[barrier_slot] = retained_barrier;
    require(!observer::states_ready(address(shared_command)) && invalidations == invalidations_before_replacement + 1,
      "restoring a replaced slot revived evidence after missed commands");
    std::puts("PASS shared Streamline slot-9 thunk is left untouched; a replaced barrier slot revokes state evidence");
#endif
    order_cases();
    order_review_cases();
    observer::shutdown();
    require(MH_DisableHook(MH_ALL_HOOKS) == MH_OK, "fixture hook disable failed");
    require(MH_Uninitialize() == MH_OK, "fixture MinHook teardown failed"); minhook = false;
    observer::reset_after_hooks_removed();
    std::puts("PASS required COM interface validation/normalization, native submit order/cookies, nested proxies, barriers, enhanced coverage, unhooked lifecycle methods, suppression, bounds, shutdown and LastError");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    observer::shutdown();
    if (minhook) { MH_DisableHook(MH_ALL_HOOKS); MH_Uninitialize(); }
    return 1;
  }
}
