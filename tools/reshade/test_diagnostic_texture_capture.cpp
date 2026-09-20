// SPDX-License-Identifier: GPL-3.0-only
// Functional GPU regression for the actual dump snapshot owner. No timing claim.
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"
#include <d3d11_1.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
  namespace capture = sunshine_streamline::depth_capture;
  namespace observer = sunshine_streamline::native_observer;
  using Microsoft::WRL::ComPtr;
  template<class T> std::uint64_t native(T *value) { return reinterpret_cast<std::uint64_t>(value); }
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  void check(HRESULT value, const char *message) {
    if (FAILED(value)) { char text[256]; std::snprintf(text, sizeof(text), "%s: 0x%08lx", message, static_cast<unsigned long>(value)); throw std::runtime_error(text); }
  }
  D3D12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE type) { D3D12_HEAP_PROPERTIES value{}; value.Type = type; return value; }
  void transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, 0, before, after}; list->ResourceBarrier(1, &barrier);
  }
  struct fixture {
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue, foreign_queue;
    ComPtr<ID3D12CommandAllocator> allocator, next_allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D11Device> host;
    ComPtr<ID3D11Device1> host1;
    ComPtr<ID3D11DeviceContext> context;
    std::uint64_t completion{};
    fixture() {
      check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
      for (unsigned index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        DXGI_ADAPTER_DESC1 desc{}; adapter->GetDesc1(&desc);
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) break;
        adapter.Reset();
      }
      require(device != nullptr, "hardware D3D12 device unavailable");
      D3D12_COMMAND_QUEUE_DESC queue_desc{}; queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "producer queue");
      check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&foreign_queue)), "independent queue");
      check(device->CreateCommandAllocator(queue_desc.Type, IID_PPV_ARGS(&allocator)), "allocator");
      check(device->CreateCommandAllocator(queue_desc.Type, IID_PPV_ARGS(&next_allocator)), "next allocator");
      check(device->CreateCommandList(0, queue_desc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "command list");
      check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "completion fence");
      check(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &host, nullptr, &context), "host D3D11 device");
      check(host.As(&host1), "host D3D11.1");
      capture::initialize(false);
      const auto prior = observer::counts();
      capture::input empty;
      require(!capture::record_diagnostic_texture(native(list.Get()), empty), "inactive capture accepted");
      const auto after = observer::counts();
      require(prior.observed == after.observed && prior.calls == after.calls, "inactive capture touched native observation");
      capture::initialize(true);
      capture::observe_queue(native(queue.Get())); capture::observe_queue(native(foreign_queue.Get()));
      capture::observe_command(native(list.Get())); capture::poll();
      require(observer::command_ready(native(list.Get())) && observer::queue_ready(native(queue.Get())), "native observer not ready");
      check(list->Close(), "initial close");
      check(allocator->Reset(), "initial allocator reset");
      check(list->Reset(allocator.Get(), nullptr), "initial list reset");
    }
    void wait() {
      check(queue->Signal(fence.Get(), ++completion), "test signal");
      HANDLE event = CreateEventW(nullptr, false, false, nullptr);
      require(event != nullptr, "test event");
      const auto hr = fence->SetEventOnCompletion(completion, event);
      const auto outcome = SUCCEEDED(hr) ? WaitForSingleObject(event, 10000) : WAIT_FAILED;
      CloseHandle(event); require(outcome == WAIT_OBJECT_0, "test GPU completion timeout");
    }
    void submit() { check(list->Close(), "close"); ID3D12CommandList *lists[]{list.Get()}; queue->ExecuteCommandLists(1, lists); }
    void reset() {
      check(next_allocator->Reset(), "spare allocator reset");
      check(list->Reset(next_allocator.Get(), nullptr), "reset producer recording");
      std::swap(allocator, next_allocator); capture::poll();
    }
    ~fixture() { capture::shutdown(); }
  };
  struct texture_case {
    ComPtr<ID3D12Resource> source, upload;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    unsigned bpp{}, width{64}, height{48};
    capture::input input;
    texture_case(fixture &gpu, DXGI_FORMAT format, unsigned bytes) : bpp(bytes) {
      D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = width; desc.Height = height; desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1; desc.Format = format;
      auto properties = heap(D3D12_HEAP_TYPE_DEFAULT);
      check(gpu.device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&source)), "source texture");
      UINT64 size{}; gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &size);
      D3D12_RESOURCE_DESC buffer{}; buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; buffer.Width = size * 2;
      buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1; buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      properties = heap(D3D12_HEAP_TYPE_UPLOAD);
      check(gpu.device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)), "upload buffer");
      unsigned char *mapped{}; D3D12_RANGE no_read{}; check(upload->Map(0, &no_read, reinterpret_cast<void **>(&mapped)), "upload map");
      std::memset(mapped, 0, size * 2);
      for (unsigned image = 0; image != 2; ++image) for (unsigned y = 0; y != height; ++y) for (unsigned x = 0; x != width * bpp; ++x)
        mapped[image * size + y * footprint.Footprint.RowPitch + x] = static_cast<unsigned char>((x * 7 + y * 11 + image * 97) & 255);
      upload->Unmap(0, nullptr);
      input.source = capture::retain_source(native(source.Get())); require(bool(input.source), "retain typed source");
      input.resource.native = native(source.Get()); input.resource.area = {4, 3, width - 8, height - 6};
      input.valid_until = sunshine_scene_depth::lifetime::at_call; input.force_snapshot = true;
      input.proof = sunshine_scene_depth::state_proof::declared;
      input.native_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      copy(gpu, 0);
      transition(gpu.list.Get(), source.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    void copy(fixture &gpu, unsigned image) {
      const auto desc = source->GetDesc(); UINT64 size{}; gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, nullptr, nullptr, nullptr, &size);
      D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
      src.pResource = upload.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = footprint; src.PlacedFootprint.Offset = image * size;
      dst.pResource = source.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    void verify_host(fixture &gpu, const capture::diagnostic_texture &snapshot, unsigned image = 0) {
      require(snapshot.width == width && snapshot.height == height && snapshot.area.left == 4 && snapshot.area.width == width - 8,
        "allocation or tagged crop lost");
      ComPtr<ID3D11Texture2D> opened, staging;
      check(gpu.host1->OpenSharedResource1(reinterpret_cast<HANDLE>(snapshot.shared_handle), IID_PPV_ARGS(&opened)), "host shared texture open");
      D3D11_TEXTURE2D_DESC desc{}; opened->GetDesc(&desc); require(desc.Format == source->GetDesc().Format, "typed format changed");
      desc.Usage = D3D11_USAGE_STAGING; desc.BindFlags = desc.MiscFlags = 0; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      check(gpu.host->CreateTexture2D(&desc, nullptr, &staging), "host staging");
      gpu.context->CopyResource(staging.Get(), opened.Get());
      D3D11_MAPPED_SUBRESOURCE mapped{}; check(gpu.context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "host readback");
      bool exact = true;
      for (unsigned y = 0; y != height; ++y) for (unsigned x = 0; x != width * bpp; ++x)
        exact &= static_cast<unsigned char *>(mapped.pData)[y * mapped.RowPitch + x] == static_cast<unsigned char>((x * 7 + y * 11 + image * 97) & 255);
      gpu.context->Unmap(staging.Get(), 0); require(exact, "snapshot changed with source mutation or lost raw bytes");
    }
  };
  struct consumer_fixture {
    fixture &gpu;
    ComPtr<ID3D12CommandAllocator> allocator, spare;
    ComPtr<ID3D12GraphicsCommandList> list;
    explicit consumer_fixture(fixture &value): gpu(value) {
      check(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "consumer allocator");
      check(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&spare)), "consumer spare allocator");
      check(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "consumer list");
      capture::observe_command(native(list.Get())); capture::poll();
      check(list->Close(), "consumer initial close");
      reset();
    }
    void reset() {
      check(spare->Reset(), "consumer allocator reset");
      check(list->Reset(spare.Get(), nullptr), "consumer recording reset");
      std::swap(allocator, spare); capture::poll();
    }
    void submit() {
      check(list->Close(), "consumer close");
      ID3D12CommandList *values[]{list.Get()}; gpu.foreign_queue->ExecuteCommandLists(1, values);
    }
    void wait() {
      ComPtr<ID3D12Fence> fence;
      check(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "consumer completion fence");
      check(gpu.foreign_queue->Signal(fence.Get(), 1), "consumer completion signal");
      HANDLE event = CreateEventW(nullptr, false, false, nullptr); require(event != nullptr, "consumer event");
      const auto hr = fence->SetEventOnCompletion(1, event);
      const auto outcome = SUCCEEDED(hr) ? WaitForSingleObject(event, 10000) : WAIT_FAILED;
      CloseHandle(event); require(outcome == WAIT_OBJECT_0, "consumer GPU completion timeout");
    }
  };
  ComPtr<ID3D12Resource> destination(ID3D12Device *device, DXGI_FORMAT format, unsigned width = 64, unsigned height = 48) {
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width; desc.Height = height; desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1; desc.Format = format;
    const auto properties = heap(D3D12_HEAP_TYPE_DEFAULT);
    ComPtr<ID3D12Resource> result;
    check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc,
      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&result)), "consumer destination");
    return result;
  }
  struct gate {
    ComPtr<ID3D12Fence> value;
    gate(ID3D12Device *device, ID3D12CommandQueue *queue) {
      check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&value)), "GPU test gate");
      check(queue->Wait(value.Get(), 1), "GPU test wait");
    }
    void open() { check(value->Signal(1), "release GPU test gate"); }
    ~gate() { value->Signal(1); } // An assertion failure must not strand the GPU.
  };
  struct readback {
    ComPtr<ID3D12Resource> bytes;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    readback(fixture &gpu, ID3D12Resource *texture) {
      const auto desc = texture->GetDesc(); UINT64 size{};
      gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &size);
      D3D12_RESOURCE_DESC buffer{}; buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; buffer.Width = size;
      buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
      buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      const auto properties = heap(D3D12_HEAP_TYPE_READBACK);
      check(gpu.device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&bytes)), "consumer readback");
    }
    void record(ID3D12GraphicsCommandList *list, ID3D12Resource *texture) {
      transition(list, texture, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
      D3D12_TEXTURE_COPY_LOCATION from{}, to{};
      from.pResource = texture; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      to.pResource = bytes.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint = footprint;
      list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
      transition(list, texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
    void verify(unsigned width, unsigned height, unsigned bpp) {
      unsigned char *mapped{}; check(bytes->Map(0, nullptr, reinterpret_cast<void **>(&mapped)), "consumer readback map");
      bool exact = true;
      for (unsigned y = 0; y != height; ++y) for (unsigned x = 0; x != width * bpp; ++x)
        exact &= mapped[y * footprint.Footprint.RowPitch + x] == static_cast<unsigned char>((x * 7 + y * 11) & 255);
      D3D12_RANGE no_write{}; bytes->Unmap(0, &no_write);
      require(exact, "auxiliary consumer copy changed native RGBA bytes");
    }
  };

  void live_auxiliary_copy(fixture &gpu, DXGI_FORMAT format, unsigned bpp, DXGI_FORMAT target_format = DXGI_FORMAT_UNKNOWN) {
    if (target_format == DXGI_FORMAT_UNKNOWN) target_format = format;
    consumer_fixture consumer(gpu);
    texture_case image(gpu, format, bpp);
    auto target = destination(gpu.device.Get(), target_format);
    capture::consumer_diagnostic diagnostic;
    auto ticket = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input);
    require(bool(ticket), "live auxiliary producer capture");
    const auto copy = [&](std::uint64_t output, std::uint64_t queue = 0) {
      return capture::copy_diagnostic_texture(native(consumer.list.Get()), queue ? queue : native(gpu.foreign_queue.Get()),
        ticket, output, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &diagnostic);
    };
    require(!copy(native(target.Get())) && diagnostic.result == capture::consumer_status::capture_not_ready,
      "unsubmitted auxiliary copy accepted");
    capture::finish_diagnostic_texture(ticket, true);
    {
      gate delayed(gpu.device.Get(), gpu.queue.Get()); gpu.submit(); gpu.reset();
      const auto before = std::chrono::steady_clock::now();
      require(!copy(native(target.Get())) && diagnostic.result == capture::consumer_status::capture_not_ready,
        "pending producer was consumed after Reset but before completion");
      require(std::chrono::steady_clock::now() - before < std::chrono::milliseconds(100), "auxiliary copy waited for producer GPU");
      delayed.open(); gpu.wait();
    }
    auto wrong_format = destination(gpu.device.Get(), target_format == DXGI_FORMAT_R8G8B8A8_UNORM ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM);
    require(!copy(native(wrong_format.Get())) && diagnostic.result == capture::consumer_status::invalid_destination,
      "different typed color formats were considered copy compatible");
    auto wrong_size = destination(gpu.device.Get(), target_format, 65);
    require(!copy(native(wrong_size.Get())) && diagnostic.result == capture::consumer_status::invalid_destination,
      "different destination extent was accepted");
    ComPtr<IDXGIAdapter> warp; ComPtr<ID3D12Device> other_device;
    check(gpu.factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "WARP identity test adapter");
    check(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&other_device)), "WARP identity test device");
    auto wrong_device = destination(other_device.Get(), target_format);
    require(!copy(native(wrong_device.Get())) && diagnostic.result == capture::consumer_status::device_mismatch,
      "foreign-device destination accepted");
    require(copy(native(target.Get())), "completed auxiliary copy was not accepted on independent consumer queue");
    auto rebound = destination(gpu.device.Get(), target_format);
    require(!copy(native(rebound.Get())) && diagnostic.result == capture::consumer_status::invalid_destination,
      "one auxiliary ticket rebound to a second destination");
    require(!copy(native(target.Get()), native(gpu.queue.Get())) && diagnostic.result == capture::consumer_status::ownership_mismatch,
      "one auxiliary ticket rebound to a second consumer queue");
    require(copy(native(target.Get())), "same auxiliary ticket could not be reused in the same recording");
    readback actual(gpu, target.Get()); actual.record(consumer.list.Get(), target.Get());
    std::weak_ptr<const capture::texture_reference> source_lifetime = ticket.ownership;
    auto retained_destination = capture::retain_source(native(target.Get()));
    std::weak_ptr<const capture::source_reference> target_lifetime = retained_destination;
    retained_destination = {}; target.Reset();
    capture::release_diagnostic_texture(ticket); ticket = {};
    {
      gate delayed(gpu.device.Get(), gpu.foreign_queue.Get()); consumer.submit(); consumer.reset(); capture::poll();
      require(!source_lifetime.expired() && !target_lifetime.expired(),
        "consumer Reset released source or destination before GPU completion");
      // The pending consumer must occupy one of the existing 32 auxiliary slots.
      std::vector<capture::diagnostic_ticket> other;
      for (unsigned i = 0; i != 31; ++i) {
        auto next = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input);
        require(bool(next), "pending consumer consumed more than one bounded auxiliary slot"); other.push_back(std::move(next));
      }
      capture::record_diagnostic exhausted;
      require(!capture::record_diagnostic_texture(native(gpu.list.Get()), image.input, &exhausted) && exhausted.result == capture::status::exhausted,
        "pending auxiliary consumer was reclaimed before its queue fence");
      for (auto &next : other) capture::release_diagnostic_texture(next);
      check(gpu.list->Close(), "cancel auxiliary pool recording"); gpu.reset(); other.clear(); capture::poll();
      delayed.open(); consumer.wait(); actual.verify(image.width, image.height, bpp); capture::poll();
      require(source_lifetime.expired() && target_lifetime.expired(), "completed auxiliary consumer ownership was not reclaimed");
    }
    std::printf("PASS live auxiliary format %u->%u: exact cross-queue copy, delayed producer/consumer, destination ownership, bounded reuse\n", unsigned(format), unsigned(target_format));
  }

  void auxiliary_consumer_replay_and_discard(fixture &gpu) {
    consumer_fixture consumer(gpu);
    texture_case image(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    auto target = destination(gpu.device.Get(), DXGI_FORMAT_R8G8B8A8_UNORM);
    const auto capture_ready = [&] {
      auto ticket = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input);
      require(bool(ticket), "auxiliary replay source capture");
      capture::finish_diagnostic_texture(ticket, true); gpu.submit(); gpu.wait(); gpu.reset(); return ticket;
    };
    const auto copy = [&](const capture::diagnostic_ticket &ticket) {
      return capture::copy_diagnostic_texture(native(consumer.list.Get()), native(gpu.foreign_queue.Get()), ticket,
        native(target.Get()), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    };
    auto discarded = capture_ready(); require(copy(discarded), "auxiliary discarded consumer setup");
    std::weak_ptr<const capture::texture_reference> discarded_lifetime = discarded.ownership;
    capture::release_diagnostic_texture(discarded); discarded = {};
    check(consumer.list->Close(), "unsubmitted consumer close"); consumer.reset(); capture::poll();
    require(discarded_lifetime.expired(), "unsubmitted consumer Reset invented an unretirable GPU fence");

    auto ticket = capture_ready(); require(copy(ticket), "auxiliary replay consumer setup");
    consumer.submit(); consumer.wait();
    std::weak_ptr<const capture::texture_reference> retained = ticket.ownership;
    capture::diagnostic_texture pixels;
    ID3D12CommandList *values[]{consumer.list.Get()};
    gpu.foreign_queue->ExecuteCommandLists(1, values); consumer.wait();
    require(capture::acquire_diagnostic_texture(ticket, pixels) == capture::status::ready,
      "same-queue replay of immutable auxiliary reads was rejected"); pixels = {};
    gpu.queue->ExecuteCommandLists(1, values); gpu.wait();
    require(capture::acquire_diagnostic_texture(ticket, pixels) == capture::status::failed &&
      pixels.failure == capture::capture_failure::consumer_queue_changed,
      "auxiliary consumer replay on a different queue did not invalidate the source"); pixels = {};
    capture::release_diagnostic_texture(ticket); ticket = {}; capture::poll();
    require(!retained.expired(), "closed consumer recording was released while it could still replay");
    consumer.reset(); capture::poll();
    require(retained.expired(), "retired replay consumer did not release its source");

    ticket = capture_ready(); require(copy(ticket), "queue retirement consumer setup");
    consumer.submit(); consumer.wait();
    capture::retire_queue(native(gpu.foreign_queue.Get()));
    require(capture::acquire_diagnostic_texture(ticket, pixels) == capture::status::failed &&
      pixels.failure == capture::capture_failure::queue_retired, "retired auxiliary consumer queue remained available");
    pixels = {}; capture::release_diagnostic_texture(ticket); ticket = {}; consumer.reset(); capture::poll();
    capture::observe_queue(native(gpu.foreign_queue.Get())); capture::poll();
    std::puts("PASS auxiliary consumer discard, same-queue replay, cross-queue replay rejection and queue retirement");
  }
  void captured_mutation(fixture &gpu, DXGI_FORMAT format, unsigned bpp) {
    texture_case image(gpu, format, bpp);
    auto wrong = image.input; wrong.native_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    capture::record_diagnostic diagnostic;
    require(!capture::record_diagnostic_texture(native(gpu.list.Get()), wrong, &diagnostic) &&
      diagnostic.result == capture::status::conflicting_state, "wrong declared state accepted");
    auto ticket = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input, &diagnostic);
    if (!ticket) { std::fprintf(stderr, "record result=%s stage=%s\n", capture::name(diagnostic.result), capture::name(diagnostic.stage)); }
    require(bool(ticket), "typed diagnostic copy not recorded");
    capture::diagnostic_texture snapshot;
    require(capture::acquire_diagnostic_texture(ticket, snapshot) == capture::status::recorded && !snapshot.texture, "unsubmitted texture exposed");
    transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    image.copy(gpu, 1);
    transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto after_mutation = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input);
    require(bool(after_mutation), "state restoration prevented later capture");
    capture::finish_diagnostic_texture(after_mutation, true);
    capture::finish_diagnostic_texture(ticket, true);
    ComPtr<ID3D12Fence> gate; check(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)), "gate fence");
    check(gpu.queue->Wait(gate.Get(), 1), "test GPU gate");
    gpu.submit();
    const auto start = std::chrono::steady_clock::now();
    require(capture::acquire_diagnostic_texture(ticket, snapshot) == capture::status::submitted && !snapshot.texture, "pending copy exposed");
    require(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(100), "acquire waited for GPU");
    gpu.reset(); // Different allocator: retirement is not GPU completion.
    require(capture::acquire_diagnostic_texture(ticket, snapshot) == capture::status::submitted, "reset assumed GPU completion");
    check(gate->Signal(1), "release test gate"); gpu.wait();
    require(capture::acquire_diagnostic_texture(ticket, snapshot) == capture::status::ready, "completed immutable copy not ready");
    image.verify_host(gpu, snapshot);
    capture::diagnostic_texture changed;
    require(capture::acquire_diagnostic_texture(after_mutation, changed) == capture::status::ready, "later capture unavailable");
    image.verify_host(gpu, changed, 1); // The snapshot must not interfere with the game's later write.
    capture::release_diagnostic_texture(after_mutation); changed = {}; after_mutation = {};
    capture::release_diagnostic_texture(ticket);
    capture::diagnostic_texture released;
    require(capture::acquire_diagnostic_texture(ticket, released) == capture::status::stale, "released ticket resurrected");
    image.verify_host(gpu, snapshot); // Existing consumer lease remains valid.
    snapshot = {}; ticket = {}; capture::poll();
    std::printf("PASS typed format %u: pre-mutation full bytes, crop, independent D3D11 host, nonblocking retirement\n", unsigned(format));
  }
  void replay_and_cancel(fixture &gpu) {
    texture_case image(gpu, DXGI_FORMAT_R8_UNORM, 1);
    gpu.submit(); gpu.wait(); gpu.reset(); // Initial upload is not part of the replayed recording.
    auto ticket = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input); require(bool(ticket), "replay capture");
    capture::finish_diagnostic_texture(ticket, true); gpu.submit(); gpu.wait();
    capture::diagnostic_texture snapshot;
    require(capture::acquire_diagnostic_texture(ticket, snapshot) == capture::status::submitted, "replayable closed recording exposed");
    ID3D12CommandList *lists[]{gpu.list.Get()}; gpu.queue->ExecuteCommandLists(1, lists); gpu.wait();
    require(capture::acquire_diagnostic_texture(ticket, snapshot) == capture::status::failed && snapshot.failure == capture::capture_failure::replay,
      "replayed snapshot accepted");
    gpu.reset(); capture::release_diagnostic_texture(ticket); ticket = {}; capture::poll();
    std::vector<capture::diagnostic_ticket> pending;
    for (unsigned i = 0; i != 32; ++i) {
      auto next = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input);
      require(bool(next), "auxiliary pool filled before own 32-slot limit"); pending.push_back(std::move(next));
    }
    capture::record_diagnostic failure;
    require(!capture::record_diagnostic_texture(native(gpu.list.Get()), image.input, &failure) && failure.result == capture::status::exhausted,
      "auxiliary pool exceeded bound");
    for (auto &next : pending) { capture::release_diagnostic_texture(next); capture::finish_diagnostic_texture(next, true); }
    check(gpu.list->Close(), "discarded recording close"); gpu.reset(); pending.clear(); capture::poll();
    auto recovered = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input); require(bool(recovered), "canceled pool did not recover");
    capture::finish_diagnostic_texture(recovered, false);
    require(capture::acquire_diagnostic_texture(recovered, snapshot) == capture::status::failed, "failed API call accepted");
    capture::release_diagnostic_texture(recovered); check(gpu.list->Close(), "canceled final close"); gpu.reset(); recovered = {}; capture::poll();
    std::puts("PASS replay rejection, canceled discarded recording, independent bounded pool and failed API result");
  }
}
int main() {
  try {
    require(capture::testing::diagnostic_snapshot_regression(), "diagnostic retirement/fence/format policy regression");
    fixture gpu;
    captured_mutation(gpu, DXGI_FORMAT_R8_UNORM, 1);
    captured_mutation(gpu, DXGI_FORMAT_R8_UINT, 1);
    captured_mutation(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    captured_mutation(gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, 8);
    live_auxiliary_copy(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    live_auxiliary_copy(gpu, DXGI_FORMAT_R10G10B10A2_UNORM, 4);
    live_auxiliary_copy(gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, 8);
    live_auxiliary_copy(gpu, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 4, DXGI_FORMAT_R8G8B8A8_UNORM);
    live_auxiliary_copy(gpu, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, 4, DXGI_FORMAT_B8G8R8A8_UNORM);
    auxiliary_consumer_replay_and_discard(gpu);
    replay_and_cancel(gpu);
    std::puts("PASS diagnostic texture capture functional GPU regression (no performance measurement)");
    return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
