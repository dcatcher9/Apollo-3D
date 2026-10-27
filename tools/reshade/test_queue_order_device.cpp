// SPDX-License-Identifier: GPL-3.0-only
// The queue-order model on real queues (L2; README, Queue order device
// cells; docs/reshade-sbs.md, Queue order): the native observer's hooks on a
// D3D12 device's queues and fences under the debug layer, as the capture
// owner installs them for a game. The producer queue renders behind a gate
// the CPU opens only after the verdict, so the read is still in flight when
// the model decides; the reader's read-back proves what the GPU ordered. The
// negative cells must refuse: a CPU signal, a second signaller and a shared
// fence each leave the reader unordered, and a wait satisfied by a signal
// before the producer orders nothing after it. Exit 77 without the debug
// layer (Windows Graphics Tools).
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <wrl/client.h>

namespace {
  namespace capture = sunshine_streamline::depth_capture;
  namespace observer = sunshine_streamline::native_observer;
  using Microsoft::WRL::ComPtr;
  using sunshine_queue_order::refusal;

  template<class T>
  std::uint64_t native(T *value) {
    return reinterpret_cast<std::uint64_t>(value);
  }

  void require(bool value, const std::string &message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  void check(HRESULT value, const char *message) {
    if (FAILED(value)) {
      char text[256];
      std::snprintf(text, sizeof(text), "%s: 0x%08lx", message, static_cast<unsigned long>(value));
      throw std::runtime_error(text);
    }
  }

  struct gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12InfoQueue> messages;
    // producer renders, reader waits and reads back, third signals as a
    // second signaller.
    ComPtr<ID3D12CommandQueue> producer, reader, third;
    ComPtr<ID3D12CommandAllocator> producer_allocator, reader_allocator;
    ComPtr<ID3D12GraphicsCommandList> producer_list, reader_list;
    ComPtr<ID3D12Resource> target, readback;
    ComPtr<ID3D12DescriptorHeap> heap;
    // gate holds the producer queue until the CPU opens it; idle is the
    // reader's own completion fence, read by the CPU only.
    ComPtr<ID3D12Fence> gate, idle;
    std::uint64_t gate_value {}, idle_value {};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};

    explicit gpu(IDXGIAdapter1 *adapter) {
      check(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)), "device");
      device.As(&messages);
      D3D12_COMMAND_QUEUE_DESC desc {};
      desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&producer)), "producer queue");
      check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&reader)), "reader queue");
      check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&third)), "third queue");
      check(device->CreateCommandAllocator(desc.Type, IID_PPV_ARGS(&producer_allocator)), "producer allocator");
      check(device->CreateCommandAllocator(desc.Type, IID_PPV_ARGS(&reader_allocator)), "reader allocator");
      check(device->CreateCommandList(0, desc.Type, producer_allocator.Get(), nullptr, IID_PPV_ARGS(&producer_list)), "producer list");
      check(device->CreateCommandList(0, desc.Type, reader_allocator.Get(), nullptr, IID_PPV_ARGS(&reader_list)), "reader list");
      check(producer_list->Close(), "close producer list");
      check(reader_list->Close(), "close reader list");
      D3D12_HEAP_PROPERTIES properties {};
      properties.Type = D3D12_HEAP_TYPE_DEFAULT;
      D3D12_RESOURCE_DESC texture {};
      texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      texture.Width = texture.Height = 1;
      texture.DepthOrArraySize = texture.MipLevels = texture.SampleDesc.Count = 1;
      texture.Format = DXGI_FORMAT_R32_FLOAT;
      texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
      check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &texture, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target)), "target");
      UINT64 bytes {};
      device->GetCopyableFootprints(&texture, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
      properties.Type = D3D12_HEAP_TYPE_READBACK;
      D3D12_RESOURCE_DESC buffer {};
      buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      buffer.Width = bytes;
      buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
      buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)), "readback");
      D3D12_DESCRIPTOR_HEAP_DESC heap_desc {};
      heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
      heap_desc.NumDescriptors = 1;
      check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "RTV heap");
      device->CreateRenderTargetView(target.Get(), nullptr, heap->GetCPUDescriptorHandleForHeapStart());
      check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)), "gate");
      check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&idle)), "idle");
    }

    static void transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
      D3D12_RESOURCE_BARRIER barrier {};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      barrier.Transition = {resource, 0, before, after};
      list->ResourceBarrier(1, &barrier);
    }

    // The producer queue waits for the reader's previous read (as a game's
    // next frame follows its Present) and for the gate, then renders value
    // into the target: the position of that submission.
    observer::position render(float value) {
      if (idle_value) {
        check(producer->Wait(idle.Get(), idle_value), "order the producer after the previous read");
      }
      check(producer->Wait(gate.Get(), ++gate_value), "gate the producer");
      check(producer_allocator->Reset(), "producer allocator reset");
      check(producer_list->Reset(producer_allocator.Get(), nullptr), "producer list reset");
      transition(producer_list.Get(), target.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
      const float color[4] {value, 0, 0, 0};
      producer_list->ClearRenderTargetView(heap->GetCPUDescriptorHandleForHeapStart(), color, 0, nullptr);
      transition(producer_list.Get(), target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
      check(producer_list->Close(), "close producer list");
      ID3D12CommandList *lists[] {producer_list.Get()};
      producer->ExecuteCommandLists(1, lists);
      return observer::last_execute_position();
    }

    // The reader queue copies the target out, then the CPU opens the gate
    // and waits for the reader: the value it read. A negative cell's reader
    // copies nothing (its read would race the render it is not ordered
    // after) and only completes.
    float read(bool copy = true) {
      if (!copy) {
        check(reader->Signal(idle.Get(), ++idle_value), "reader completion");
        check(gate->Signal(gate_value), "open the gate");
        check(idle->SetEventOnCompletion(idle_value, nullptr), "wait for the reader");
        check(producer->Signal(idle.Get(), ++idle_value), "producer completion");
        check(idle->SetEventOnCompletion(idle_value, nullptr), "wait for the producer");
        return 0;
      }
      check(reader_allocator->Reset(), "reader allocator reset");
      check(reader_list->Reset(reader_allocator.Get(), nullptr), "reader list reset");
      transition(reader_list.Get(), target.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
      D3D12_TEXTURE_COPY_LOCATION to {readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
      to.PlacedFootprint = footprint;
      D3D12_TEXTURE_COPY_LOCATION from {target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
      reader_list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
      transition(reader_list.Get(), target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
      check(reader_list->Close(), "close reader list");
      ID3D12CommandList *lists[] {reader_list.Get()};
      reader->ExecuteCommandLists(1, lists);
      check(reader->Signal(idle.Get(), ++idle_value), "reader completion");
      check(gate->Signal(gate_value), "open the gate");
      check(idle->SetEventOnCompletion(idle_value, nullptr), "wait for the reader");
      float value {};
      void *mapped {};
      const D3D12_RANGE range {0, sizeof(float)};
      check(readback->Map(0, &range, &mapped), "map readback");
      std::memcpy(&value, mapped, sizeof(value));
      const D3D12_RANGE none {};
      readback->Unmap(0, &none);
      return value;
    }

    // A fence settled for the model: a first signal of the producer queue
    // above its value at first sight, read complete by a later hooked call.
    ComPtr<ID3D12Fence> settled_fence(D3D12_FENCE_FLAGS flags = D3D12_FENCE_FLAG_NONE) {
      ComPtr<ID3D12Fence> fence;
      check(device->CreateFence(0, flags, IID_PPV_ARGS(&fence)), "cell fence");
      check(producer->Signal(fence.Get(), 1), "settling signal");
      check(fence->SetEventOnCompletion(1, nullptr), "settling completion");
      return fence;
    }

    void require_clean() {
      if (!messages) {
        return;
      }
      std::string errors;
      for (UINT64 i = 0; i != messages->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
        SIZE_T size {};
        check(messages->GetMessage(i, nullptr, &size), "debug message size");
        std::vector<unsigned char> storage(size);
        auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
        check(messages->GetMessage(i, message, &size), "debug message");
        if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR || message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
          errors += std::string(message->pDescription) + "\n";
        }
      }
      require(errors.empty(), "D3D12 validation errors:\n" + errors);
    }
  };

  void cells(gpu &g) {
    const auto reader = native(g.reader.Get());
    const auto ordered = [&](const observer::position &at) {
      return observer::queue_ordered(reader, at);
    };
    // The fences' CPU Signal slot is discovered at the first fence's sight
    // and installed at the next poll, as a game's first frames do.
    g.settled_fence();
    capture::poll();
    require(observer::queue_order_counts().fence_signal_hooks >= 1 && !observer::queue_order_counts().pending, "the fence's CPU Signal slot was not installed");
    float value = 1;

    // Positive, after-signal: the producer renders behind the gate and
    // signals; the reader waits for that signal. Ordered while the render is
    // still in flight, and the reader reads what the render wrote.
    {
      auto fence = g.settled_fence();
      const auto at = g.render(++value);
      require(at && at.quiet, "the producer's submission took no quiet position");
      check(g.producer->Signal(fence.Get(), 2), "producer signal");
      check(g.reader->Wait(fence.Get(), 2), "reader wait");
      const auto verdict = ordered(at);
      require(verdict.ordered && verdict.fences[0], "after-signal: a reader waiting for the producer's signal is not ordered after it");
      require(g.read() == value, "after-signal: the reader read before the producer's render");
    }
    // Positive, wait before signal: the reader's wait is submitted before
    // the producer's signal and resolves to it.
    {
      auto fence = g.settled_fence();
      const auto at = g.render(++value);
      check(g.reader->Wait(fence.Get(), 2), "reader wait before the signal");
      check(g.producer->Signal(fence.Get(), 2), "producer signal after the wait");
      require(ordered(at).ordered, "wait-before-signal: the wait did not resolve to the producer's later signal");
      require(g.read() == value, "wait-before-signal: the reader read before the producer's render");
    }
    // Negative, earlier signal: the wait is satisfied by a signal the
    // producer made before its render.
    {
      auto fence = g.settled_fence();
      check(g.producer->Signal(fence.Get(), 2), "producer signal before its render");
      const auto at = g.render(++value);
      check(g.producer->Signal(fence.Get(), 3), "producer signal after its render");
      check(g.reader->Wait(fence.Get(), 2), "reader wait for the earlier signal");
      require(!ordered(at).ordered, "earlier-signal: a wait satisfied before the producer ordered the reader after it");
      g.read(false);
    }
    // Negative, CPU signal: the CPU raises the fence to the waited value.
    {
      auto fence = g.settled_fence();
      const auto at = g.render(++value);
      check(g.producer->Signal(fence.Get(), 2), "producer signal");
      check(fence->Signal(2), "CPU signal");
      check(g.reader->Wait(fence.Get(), 2), "reader wait");
      const auto verdict = ordered(at);
      require(!verdict.ordered && verdict.why == refusal::poisoned, "cpu-signal: a CPU-signalled fence ordered the reader");
      g.read(false);
    }
    // Negative, second signaller: another queue signals the waited value.
    {
      auto fence = g.settled_fence();
      const auto at = g.render(++value);
      check(g.producer->Signal(fence.Get(), 2), "producer signal");
      check(g.third->Signal(fence.Get(), 2), "second signaller");
      check(g.reader->Wait(fence.Get(), 2), "reader wait");
      const auto verdict = ordered(at);
      require(!verdict.ordered && verdict.why == refusal::poisoned, "second-signaller: a fence with a second signaller ordered the reader");
      g.read(false);
    }
    // Negative, shared fence: another process or adapter may signal it.
    {
      auto fence = g.settled_fence(D3D12_FENCE_FLAG_SHARED);
      const auto at = g.render(++value);
      check(g.producer->Signal(fence.Get(), 2), "producer signal");
      check(g.reader->Wait(fence.Get(), 2), "reader wait");
      const auto verdict = ordered(at);
      require(!verdict.ordered && verdict.why == refusal::poisoned, "shared: a shared fence ordered the reader");
      require(observer::queue_order_counts().model.poisoned[static_cast<unsigned>(sunshine_queue_order::poison::shared)] >= 1, "shared: the shared fence was not poisoned as shared");
      g.read(false);
    }
    g.require_clean();
  }
}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    ComPtr<ID3D12Debug> debug;
    if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
      std::puts("SKIP the D3D12 debug layer (Windows Graphics Tools) is not installed");
      return 77;
    }
    debug->EnableDebugLayer();
    ComPtr<IDXGIFactory4> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
    ComPtr<IDXGIAdapter1> adapter, chosen;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
      DXGI_ADAPTER_DESC1 desc {};
      adapter->GetDesc1(&desc);
      if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
        chosen = adapter;
        break;
      }
    }
    if (!chosen) {
      ComPtr<IDXGIAdapter> warp;
      check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "WARP adapter");
      check(warp.As(&chosen), "WARP adapter 1");
    }
    gpu g(chosen.Get());
    capture::initialize(true);
    for (auto *queue : {g.producer.Get(), g.reader.Get(), g.third.Get()}) {
      capture::observe_queue(native(queue));
    }
    capture::poll();
    const auto status = observer::queue_order_counts();
    require(status.enabled && status.signal_hooks >= 1 && status.wait_hooks >= 1 && !status.pending, "the queues' Signal and Wait slots were not installed");
    cells(g);
    capture::shutdown();
    std::puts(
      "PASS queue order on real queues: after-signal and wait-before-signal reads in flight ordered and read back exact; "
      "an earlier signal, a CPU signal, a second signaller and a shared fence refused; D3D12 validation clean"
    );
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
