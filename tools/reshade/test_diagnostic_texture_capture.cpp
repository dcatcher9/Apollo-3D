// SPDX-License-Identifier: GPL-3.0-only
// Functional GPU regression for the actual dump snapshot owner. No timing claim.
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"
#include "native_resource_identity.h"
#include <d3d11_1.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
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
    ComPtr<ID3D12InfoQueue> debug_messages;
    ComPtr<ID3D12CommandQueue> queue, foreign_queue;
    ComPtr<ID3D12CommandAllocator> allocator, next_allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D11Device> host;
    ComPtr<ID3D11Device1> host1;
    ComPtr<ID3D11DeviceContext> context;
    std::uint64_t completion{};
    fixture() {
      ComPtr<ID3D12Debug> debug;
      if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
      check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
      for (unsigned index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        DXGI_ADAPTER_DESC1 desc{}; adapter->GetDesc1(&desc);
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) break;
        adapter.Reset();
      }
      require(device != nullptr, "hardware D3D12 device unavailable");
      device.As(&debug_messages);
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
    void check_debug_errors() {
      if (!debug_messages) return;
      bool clean = true;
      for (UINT64 i = 0; i != debug_messages->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
        SIZE_T size{}; check(debug_messages->GetMessage(i, nullptr, &size), "debug message size");
        std::vector<unsigned char> storage(size);
        auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
        check(debug_messages->GetMessage(i, message, &size), "debug message");
        if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR || message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
          std::fprintf(stderr, "D3D12 error %u: %s\n", unsigned(message->ID), message->pDescription); clean = false;
        }
      }
      require(clean, "observed-recording color snapshot produced D3D12 validation errors");
    }
    ~fixture() { capture::shutdown(); }
  };
  struct texture_case {
    ComPtr<ID3D12Resource> source, upload;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    unsigned bpp{}, width{64}, height{48};
    capture::input input;
    texture_case(fixture &gpu, DXGI_FORMAT format, unsigned bytes,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE, bool retain_before_barrier = true) : bpp(bytes) {
      D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = width; desc.Height = height; desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1; desc.Format = format;
      desc.Flags = flags;
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
      if (retain_before_barrier) {
        input.source = capture::retain_source(native(source.Get())); require(bool(input.source), "retain typed source");
      }
      input.resource.native = native(source.Get()); input.resource.area = {4, 3, width - 8, height - 6};
      input.valid_until = sunshine_scene_depth::lifetime::at_call; input.force_snapshot = true;
      input.proof = sunshine_scene_depth::state_proof::declared;
      input.native_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      copy(gpu, 0);
      if (retain_before_barrier)
        transition(gpu.list.Get(), source.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    void copy(fixture &gpu, unsigned image) {
      const auto desc = source->GetDesc(); UINT64 size{}; gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, nullptr, nullptr, nullptr, &size);
      D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
      src.pResource = upload.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = footprint; src.PlacedFootprint.Offset = image * size;
      dst.pResource = source.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    void verify_host(fixture &gpu, const capture::diagnostic_texture &snapshot, unsigned image = 0,
        ID3D11Texture2D *already_open = nullptr, DXGI_FORMAT expected_format = DXGI_FORMAT_UNKNOWN) {
      require(snapshot.width == width && snapshot.height == height && snapshot.area.left == 4 && snapshot.area.width == width - 8,
        "allocation or tagged crop lost");
      ComPtr<ID3D11Texture2D> opened, staging;
      if (already_open) opened = already_open;
      else check(gpu.host1->OpenSharedResource1(reinterpret_cast<HANDLE>(snapshot.shared_handle), IID_PPV_ARGS(&opened)), "host shared texture open");
      if (expected_format == DXGI_FORMAT_UNKNOWN) expected_format = source->GetDesc().Format;
      D3D11_TEXTURE2D_DESC desc{}; opened->GetDesc(&desc);
      require(desc.Format == expected_format && snapshot.format == unsigned(expected_format),
        "shared host texture does not expose the captured typed format");
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
    void verify(unsigned width, unsigned height, unsigned bpp, unsigned image = 0) {
      unsigned char *mapped{}; check(bytes->Map(0, nullptr, reinterpret_cast<void **>(&mapped)), "consumer readback map");
      bool exact = true;
      for (unsigned y = 0; y != height; ++y) for (unsigned x = 0; x != width * bpp; ++x)
        exact &= mapped[y * footprint.Footprint.RowPitch + x] == static_cast<unsigned char>((x * 7 + y * 11 + image * 97) & 255);
      D3D12_RANGE no_write{}; bytes->Unmap(0, &no_write);
      require(exact, "auxiliary consumer copy changed native RGBA bytes");
    }
  };

  capture::diagnostic_ticket auxiliary_snapshot(fixture &gpu, bool shared, const capture::input &input,
      capture::record_diagnostic *diagnostic, capture::texture_state_policy policy) {
    return shared ? capture::record_diagnostic_texture(native(gpu.list.Get()), input, diagnostic, policy) :
      capture::record_local_texture(native(gpu.list.Get()), input, diagnostic, policy);
  }

  void first_recording_depth_capture(fixture &gpu) {
    if (gpu.debug_messages) gpu.debug_messages->ClearStoredMessages();
    consumer_fixture consumer(gpu);
    texture_case depth(gpu, DXGI_FORMAT_R32_FLOAT, 4, D3D12_RESOURCE_FLAG_NONE, false);
    gpu.submit(); gpu.wait(); gpu.reset(); // Finish the source upload on its own recording.
    require(!depth.input.source && !sunshine_native_identity::resource_cookie(depth.source.Get()),
      "first-resource fixture retained source identity before the observed barrier");

    // Hooks are already installed, but these native lists have never been
    // observed explicitly or Reset. CreateCommandList returns an open recording.
    ComPtr<ID3D12CommandAllocator> empty_allocator, source_allocator;
    ComPtr<ID3D12GraphicsCommandList> empty_list, source_list;
    const auto fresh_list = [&](ComPtr<ID3D12CommandAllocator> &allocator,
        ComPtr<ID3D12GraphicsCommandList> &list) {
      check(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
        "first-recording allocator");
      check(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
        IID_PPV_ARGS(&list)), "first-recording command list");
      require(observer::get_recording_cookie(native(list.Get())) == 0,
        "fresh native command list already has a recording cookie");
      require(observer::command_ready(native(list.Get())), "fresh native list did not share installed hooks");
    };
    fresh_list(empty_allocator, empty_list);
    fresh_list(source_allocator, source_list);

    // Neither this open command list nor this native resource has been
    // registered. Preserve the first ordinary transition before NGX supplies
    // the resource to retain_source. Upload left the source in COPY_DEST.
    transition(source_list.Get(), depth.source.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
      D3D12_RESOURCE_STATE_COPY_SOURCE);
    const auto first_cookie = observer::get_recording_cookie(native(source_list.Get()));
    const auto first_source_cookie = sunshine_native_identity::resource_cookie(depth.source.Get());

    auto input = depth.input;
    input.source = capture::retain_source(native(depth.source.Get()));
    require(bool(input.source), "retain source after its first ordinary barrier");
    input.provider = sunshine_scene_depth::provider_kind::ngx;
    input.epoch = 73; input.sequence = 1; input.source_id = 91; input.viewport = 0;
    input.tick = GetTickCount64();
    input.native_state = 0; // NGX supplies no resource-state declaration.
    input.proof = sunshine_scene_depth::state_proof::observed_nonzero;
    capture::record_diagnostic diagnostic;
    const auto empty_ticket = capture::nominate_evaluation(native(empty_list.Get()), input, UINT64_MAX, &diagnostic);
    if (empty_ticket) capture::finish(empty_ticket, false); // Metadata-only nomination never authorizes pixels.
    require(diagnostic.result == capture::status::missing_state && diagnostic.stage == capture::record_stage::missing_state &&
        !diagnostic.observed && !diagnostic.copy_state_known,
      "fresh recording imported NGX state from another command list's source barrier");
    check(empty_list->Close(), "empty first-recording close");
    capture::command_destroyed(native(empty_list.Get()));

    input.sequence = 2; input.tick = GetTickCount64();
    const auto ticket = capture::nominate_evaluation(native(source_list.Get()), input, UINT64_MAX, &diagnostic);
    if (ticket) capture::finish(ticket, diagnostic.result == capture::status::recorded);
    if (diagnostic.result != capture::status::recorded) std::fprintf(stderr,
      "first native barrier: result=%s stage=%s before_nomination_cookie=%llu nomination_cookie=%llu source_cookie_at_barrier=%llu observed=%u\n",
      capture::name(diagnostic.result), capture::name(diagnostic.stage), static_cast<unsigned long long>(first_cookie),
      static_cast<unsigned long long>(diagnostic.recording_cookie), static_cast<unsigned long long>(first_source_cookie), unsigned(diagnostic.observed));
    require(first_cookie && first_source_cookie && ticket && diagnostic.result == capture::status::recorded && diagnostic.observed &&
        !diagnostic.blocked && diagnostic.observed_state == D3D12_RESOURCE_STATE_COPY_SOURCE &&
        diagnostic.copy_state_known && diagnostic.copy_state == D3D12_RESOURCE_STATE_COPY_SOURCE &&
        diagnostic.used_observed_state && diagnostic.recording_cookie == first_cookie,
      "NGX nomination lost the first pre-retention resource barrier or replaced its recording identity");
    transition(source_list.Get(), depth.source.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); // Validate exact source-state restoration.
    check(source_list->Close(), "first-recording source close");
    ID3D12CommandList *lists[]{source_list.Get()}; gpu.queue->ExecuteCommandLists(1, lists); gpu.wait();
    check(source_allocator->Reset(), "first-recording completed allocator reset");
    check(source_list->Reset(source_allocator.Get(), nullptr), "first-recording source retirement");
    capture::poll();

    capture::packet packet;
    capture::capture_diagnostic acquired;
    require(capture::acquire(native(gpu.foreign_queue.Get()), 1, packet, &acquired) && packet.pixel_ready &&
        !packet.shared_preservation && packet.capture_id == ticket &&
        packet.metadata.provider == sunshine_scene_depth::provider_kind::ngx && !packet.metadata.frame_generation_input,
      "completed first-recording NGX depth was unavailable or changed provider role");
    auto target = destination(gpu.device.Get(), DXGI_FORMAT_R32_FLOAT);
    readback actual(gpu, target.Get());
    require(capture::copy_current(native(consumer.list.Get()), packet, native(target.Get()),
      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE), "first-recording NGX depth readback copy failed");
    actual.record(consumer.list.Get(), target.Get());
    capture::complete_frame(packet, 1);
    consumer.submit(); consumer.wait(); consumer.reset();
    actual.verify(depth.width, depth.height, depth.bpp);
    packet = {};
    capture::retire_source(input.provider, input.epoch, input.source_id);
    capture::command_destroyed(native(source_list.Get()));
    capture::command_destroyed(native(consumer.list.Get()));
    capture::poll();
    gpu.check_debug_errors();
    std::printf("PASS first native recording and resource: barrier before registration/Reset/retain_source preserves observed-only NGX depth and exact pixels; cross-recording state rejected%s\n",
      gpu.debug_messages ? "; D3D12 validation clean" : "; D3D12 debug layer unavailable");
  }

  // NVIDIA's DLSS guide fixes every D3D12 input at NON_PIXEL_SHADER_RESOURCE for
  // the evaluation call. Engines often transition depth on an earlier command
  // list, so the evaluating recording has no entry for it (Hogwarts Legacy logged
  // missing_state for every NGX evaluation). Observed state still takes priority.
  void ngx_contract_depth_capture(fixture &gpu) {
    if (gpu.debug_messages) gpu.debug_messages->ClearStoredMessages();
    consumer_fixture consumer(gpu);
    texture_case depth(gpu, DXGI_FORMAT_R32_FLOAT, 4);
    // The source reaches its contract state on a separate, retired recording.
    gpu.submit(); gpu.wait(); gpu.reset();
    auto input = depth.input;
    input.provider = sunshine_scene_depth::provider_kind::ngx;
    input.epoch = 74; input.source_id = 92; input.viewport = 0;
    input.valid_until = sunshine_scene_depth::lifetime::until_evaluation; input.force_snapshot = false;
    input.source_present_generation = capture::source_present_generation(input.source);
    input.native_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    input.proof = sunshine_scene_depth::state_proof::sdk_contract;
    capture::record_diagnostic diagnostic;
    const auto nominate = [&](std::uint64_t sequence) {
      input.sequence = sequence; input.tick = GetTickCount64();
      return capture::nominate_evaluation(native(gpu.list.Get()), input, UINT64_MAX, &diagnostic);
    };

    // An unobserved source uses the SDK contract; the debug layer validates it.
    const auto ticket = nominate(1);
    require(ticket && diagnostic.result == capture::status::recorded && !diagnostic.observed &&
        diagnostic.copy_state_known && diagnostic.copy_state == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE &&
        diagnostic.used_contract_state && !diagnostic.used_observed_state,
      "NGX contract state did not admit an unobserved evaluation input");
    capture::finish(ticket, true);
    gpu.submit(); gpu.wait(); gpu.reset();
    capture::packet packet;
    capture::capture_diagnostic acquired;
    require(capture::acquire(native(gpu.foreign_queue.Get()), 11, packet, &acquired) && packet.pixel_ready &&
        packet.capture_id == ticket && packet.metadata.provider == sunshine_scene_depth::provider_kind::ngx,
      "contract-state NGX depth was unavailable to its consumer");
    auto target = destination(gpu.device.Get(), DXGI_FORMAT_R32_FLOAT);
    readback actual(gpu, target.Get());
    require(capture::copy_current(native(consumer.list.Get()), packet, native(target.Get()),
      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE), "contract-state NGX depth consumer copy failed");
    actual.record(consumer.list.Get(), target.Get());
    capture::complete_frame(packet, 11);
    consumer.submit(); consumer.wait(); consumer.reset();
    actual.verify(depth.width, depth.height, depth.bpp);
    packet = {};

    // A state observed on the evaluating recording overrides the contract.
    transition(gpu.list.Get(), depth.source.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
      D3D12_RESOURCE_STATE_COPY_SOURCE);
    const auto observed = nominate(2);
    require(observed && diagnostic.result == capture::status::recorded && diagnostic.observed &&
        diagnostic.copy_state == D3D12_RESOURCE_STATE_COPY_SOURCE && diagnostic.used_observed_state &&
        !diagnostic.used_contract_state, "NGX contract replaced state observed on the same recording");
    capture::finish(observed, true);
    // Observed COMMON is evidence against the contract, not an absent entry.
    transition(gpu.list.Get(), depth.source.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    const auto common = nominate(3);
    if (common) capture::finish(common, false);
    require(diagnostic.result == capture::status::missing_state && diagnostic.observed &&
        diagnostic.observed_state == 0 && !diagnostic.copy_state_known,
      "NGX contract admitted a source observed in COMMON");
    transition(gpu.list.Get(), depth.source.Get(), D3D12_RESOURCE_STATE_COMMON,
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.submit(); gpu.wait(); gpu.reset();
    // A contract proof without a usable state value remains unavailable.
    for (const auto hint : {0u, UINT32_MAX}) {
      input.native_state = hint;
      const auto missing = nominate(4 + hint % 2);
      if (missing) capture::finish(missing, false);
      require(diagnostic.result == capture::status::missing_state && !diagnostic.observed && !diagnostic.copy_state_known,
        "NGX contract proof without a state value admitted capture");
    }
    capture::retire_source(input.provider, input.epoch, input.source_id);
    capture::command_destroyed(native(consumer.list.Get()));
    capture::poll();
    gpu.check_debug_errors();
    std::printf("PASS NGX SDK-contract input state: unobserved source uses NON_PIXEL_SHADER_RESOURCE with exact pixels; observed state wins; COMMON and absent values rejected%s\n",
      gpu.debug_messages ? "; D3D12 validation clean" : "; D3D12 debug layer unavailable");
  }

  void observed_recording_color_capture(fixture &gpu, DXGI_FORMAT format, bool shared,
      D3D12_RESOURCE_STATES observed_state = D3D12_RESOURCE_STATE_RENDER_TARGET,
      DXGI_FORMAT snapshot_format = DXGI_FORMAT_UNKNOWN,
      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      bool frame_generation = true,
      D3D12_RESOURCE_STATES declared_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
    if (snapshot_format == DXGI_FORMAT_UNKNOWN) snapshot_format = format;
    consumer_fixture consumer(gpu);
    if (gpu.debug_messages) gpu.debug_messages->ClearStoredMessages();
    constexpr auto policy = capture::texture_state_policy::prefer_observed_recording;
    const auto rt = observed_state;
    texture_case image(gpu, format, format == DXGI_FORMAT_R8_UNORM ? 1 : 4, flags);
    auto input = image.input; input.frame_generation_input = frame_generation; input.native_state = declared_state;
    transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, rt);
    capture::record_diagnostic diagnostic;
    const auto strict_rejection = [&] {
      require(diagnostic.result == capture::status::conflicting_state &&
        diagnostic.stage == capture::record_stage::conflicting_state && diagnostic.observed && !diagnostic.blocked &&
        diagnostic.native_state == declared_state && diagnostic.observed_state == rt && diagnostic.recording_cookie &&
        !diagnostic.copy_state_known && !diagnostic.used_observed_state,
        "strict capture changed admission or lost declared/observed state evidence");
    };
    require(!capture::record_diagnostic_texture(native(gpu.list.Get()), input, &diagnostic),
      "optional shared diagnostic accepted conflicting tag state"); strict_rejection();
    require(!capture::record_local_texture(native(gpu.list.Get()), input, &diagnostic),
      "default local capture accepted conflicting tag state"); strict_rejection();

    // Both storage lifetimes use the same synchronous Streamline input policy;
    // FG status is not part of the resource-state proof.
    const auto reject_role = [&](const capture::input &wrong) {
      require(!auxiliary_snapshot(gpu, shared, wrong, &diagnostic, policy) &&
        diagnostic.result == capture::status::malformed && diagnostic.stage == capture::record_stage::malformed_input &&
        !diagnostic.copy_state_known, "observed-recording policy accepted a different capture role");
    };
    auto wrong = input; wrong.provider = sunshine_scene_depth::provider_kind::ngx; reject_role(wrong);
    wrong = input; wrong.force_snapshot = false; reject_role(wrong);
    for (const auto lifetime : {sunshine_scene_depth::lifetime::until_present,
        sunshine_scene_depth::lifetime::until_evaluation, sunshine_scene_depth::lifetime::unsupported}) {
      wrong = input; wrong.valid_until = lifetime; reject_role(wrong);
    }
    require(!auxiliary_snapshot(gpu, shared, input, &diagnostic,
      static_cast<capture::texture_state_policy>(99)) && diagnostic.result == capture::status::malformed,
      "unknown auxiliary state policy silently selected a fallback");

    // Ordinary depth still requires its source contract to agree with evidence.
    texture_case depth(gpu, DXGI_FORMAT_R32_FLOAT, 4);
    auto depth_input = depth.input; depth_input.epoch = 1; depth_input.sequence = 1; depth_input.native_state = declared_state;
    require(!capture::record(native(gpu.list.Get()), depth_input, &diagnostic) &&
      diagnostic.result == capture::status::conflicting_state && !diagnostic.copy_state_known,
      "ordinary depth admission was weakened by the auxiliary color policy");

    const auto record = [&] {
      auto ticket = auxiliary_snapshot(gpu, shared, input, &diagnostic, policy);
      require(bool(ticket) && diagnostic.result == capture::status::recorded && diagnostic.copy_state_known &&
        diagnostic.used_observed_state && diagnostic.copy_state == rt && diagnostic.native_state == declared_state &&
        diagnostic.observed && diagnostic.observed_state == rt && !diagnostic.blocked && diagnostic.recording_cookie &&
        diagnostic.format == unsigned(format) && diagnostic.flags == unsigned(flags),
        "auxiliary policy did not preserve observed state, original format and declared-state provenance");
      capture::finish_diagnostic_texture(ticket, true); return ticket;
    };
    auto original = record();
    // This real write requires capture to restore its observed input state.
    transition(gpu.list.Get(), image.source.Get(), rt, D3D12_RESOURCE_STATE_COPY_DEST);
    image.copy(gpu, 1);
    transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_COPY_DEST, rt);
    auto changed = record();
    auto correct = input; correct.native_state = rt;
    auto strict_after = auxiliary_snapshot(gpu, shared, correct, &diagnostic, capture::texture_state_policy::source_contract);
    require(bool(strict_after) && diagnostic.copy_state_known && diagnostic.copy_state == rt &&
      !diagnostic.used_observed_state, "auxiliary override changed later declared-state admission");
    capture::finish_diagnostic_texture(strict_after, true);
    gpu.submit(); gpu.wait(); gpu.reset();

    auto first_target = destination(gpu.device.Get(), snapshot_format);
    auto next_target = destination(gpu.device.Get(), snapshot_format);
    readback first_bytes(gpu, first_target.Get()), next_bytes(gpu, next_target.Get());
    ComPtr<ID3D11Texture2D> host_lease;
    capture::diagnostic_texture host_snapshot;
    std::weak_ptr<const capture::texture_reference> host_storage;
    const auto read = [&](const capture::diagnostic_ticket &ticket, ID3D12Resource *target, readback &bytes, unsigned version) {
      capture::diagnostic_texture pixels;
      require(capture::acquire_diagnostic_texture(ticket, pixels) == capture::status::ready &&
          bool(pixels.shared_handle) == shared && pixels.format == unsigned(snapshot_format),
        "observed-state auxiliary copy lost its storage lifetime or typed format");
      if (shared) {
        image.verify_host(gpu, pixels, version, nullptr, snapshot_format);
        if (!version) {
          host_snapshot = pixels;
          host_storage = pixels.ownership;
          check(gpu.host1->OpenSharedResource1(reinterpret_cast<HANDLE>(pixels.shared_handle), IID_PPV_ARGS(&host_lease)),
            "retain observed-state shared host snapshot");
        }
      }
      require(capture::copy_diagnostic_texture(native(consumer.list.Get()), native(gpu.foreign_queue.Get()),
        ticket, native(target), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE), "observed-state auxiliary readback copy failed");
      bytes.record(consumer.list.Get(), target);
    };
    read(original, first_target.Get(), first_bytes, 0); read(changed, next_target.Get(), next_bytes, 1);
    consumer.submit(); consumer.wait(); consumer.reset();
    first_bytes.verify(image.width, image.height, image.bpp, 0);
    next_bytes.verify(image.width, image.height, image.bpp, 1);
    capture::release_diagnostic_texture(original); capture::release_diagnostic_texture(changed);
    capture::release_diagnostic_texture(strict_after); original = {}; changed = {}; strict_after = {}; capture::poll();
    host_snapshot.ownership = {};
    capture::poll();
    if (shared) {
      require(host_storage.expired(), "shared observed-state storage retained an add-on lease after host acknowledgement");
      image.verify_host(gpu, host_snapshot, 0, host_lease.Get(), snapshot_format);
    }

    // An observation-required proof must still reject after Reset. The
    // explicit-declaration compatibility path is tested separately in UAV.
    const auto reject_state = [&](capture::status result, capture::record_stage stage) {
      require(!auxiliary_snapshot(gpu, shared, input, &diagnostic, policy) &&
        diagnostic.result == result && diagnostic.stage == stage && !diagnostic.copy_state_known &&
        !diagnostic.used_observed_state, "observed-recording state guard admitted missing or unsafe evidence");
    };
    input.proof = sunshine_scene_depth::state_proof::observed_nonzero;
    reject_state(capture::status::missing_state, capture::record_stage::missing_state);
    require(!diagnostic.observed, "reset retained state from an older recording");
    input.proof = sunshine_scene_depth::state_proof::declared;
    transition(gpu.list.Get(), image.source.Get(), rt, D3D12_RESOURCE_STATE_COMMON);
    reject_state(capture::status::missing_state, capture::record_stage::missing_state);
    require(diagnostic.observed && !diagnostic.blocked && diagnostic.observed_state == 0,
      "observed COMMON rejection was mistaken for absent evidence");
    transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RESOLVE_DEST);
    reject_state(capture::status::incomplete_state, capture::record_stage::incomplete_state);
    require(diagnostic.observed_state == D3D12_RESOURCE_STATE_RESOLVE_DEST, "unsupported legacy state evidence was lost");
    transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_RESOLVE_DEST, rt);
    D3D12_RESOURCE_BARRIER split{}; split.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    split.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
    split.Transition = {image.source.Get(), 0, rt, D3D12_RESOURCE_STATE_COPY_SOURCE};
    gpu.list->ResourceBarrier(1, &split);
    reject_state(capture::status::incomplete_state, capture::record_stage::incomplete_state);
    require(diagnostic.blocked, "pending split did not retain its blocked state");
    split.Flags = D3D12_RESOURCE_BARRIER_FLAG_END_ONLY; gpu.list->ResourceBarrier(1, &split);
    reject_state(capture::status::incomplete_state, capture::record_stage::incomplete_state);
    gpu.submit(); gpu.wait(); gpu.reset();
    transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, rt);
    D3D12_RESOURCE_BARRIER alias{}; alias.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    gpu.list->ResourceBarrier(1, &alias); // Legal wildcard alias, unknowable affected source set.
    reject_state(capture::status::unavailable, capture::record_stage::recording_invalid);
    require(diagnostic.recording_invalid && diagnostic.loss == capture::recording_loss::wildcard_alias,
      "lost recording evidence was accepted or misreported");
    gpu.submit(); gpu.wait(); gpu.reset();
    gpu.check_debug_errors();
    std::printf("PASS %s observed-state policy format %u->%u flags=%u FG=%u: declared%u/observed%u exact bytes and restoration; strict depth/default, role, missing/COMMON, split and loss guards%s\n",
      shared ? "shared" : "local", unsigned(format), unsigned(snapshot_format), unsigned(flags), unsigned(frame_generation),
      unsigned(declared_state), unsigned(observed_state), gpu.debug_messages ? "; D3D12 validation clean" : "; D3D12 debug layer unavailable");
  }

  void declared_recording_color_capture(fixture &gpu, DXGI_FORMAT format, bool shared,
      DXGI_FORMAT snapshot_format = DXGI_FORMAT_UNKNOWN,
      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      bool frame_generation = true) {
    if (snapshot_format == DXGI_FORMAT_UNKNOWN) snapshot_format = format;
    consumer_fixture consumer(gpu);
    if (gpu.debug_messages) gpu.debug_messages->ClearStoredMessages();
    constexpr auto policy = capture::texture_state_policy::prefer_observed_recording;
    constexpr auto uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    texture_case image(gpu, format, 4, flags);
    texture_case ordinary(gpu, format, 4); // Deliberately lacks RT/UAV/depth capabilities.
    auto input = image.input; input.frame_generation_input = frame_generation; input.native_state = uav;
    // A UAV write state does not decay when the original recording is submitted.
    // Capture will run on a fresh recording without any source-state observation.
    transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, uav);
    gpu.submit(); gpu.wait(); gpu.reset();
    capture::record_diagnostic diagnostic;
    const auto reject = [&](const capture::input &candidate, capture::status result, capture::record_stage stage) {
      require(!auxiliary_snapshot(gpu, shared, candidate, &diagnostic, policy) &&
        diagnostic.result == result && diagnostic.stage == stage && !diagnostic.observed &&
        !diagnostic.copy_state_known && !diagnostic.used_observed_state,
        "absent-observation fallback accepted an absent, malformed or incompatible declaration");
    };
    for (const auto hint : {0u, UINT32_MAX}) {
      auto wrong = input; wrong.native_state = hint;
      reject(wrong, capture::status::missing_state, capture::record_stage::missing_state);
    }
    auto wrong = input; wrong.proof = sunshine_scene_depth::state_proof::observed_nonzero;
    reject(wrong, capture::status::missing_state, capture::record_stage::missing_state);
    wrong = input; wrong.proof = sunshine_scene_depth::state_proof::unavailable;
    reject(wrong, capture::status::unsupported_state, capture::record_stage::unsupported_proof);
    for (const auto hint : {std::uint32_t(D3D12_RESOURCE_STATE_RESOLVE_DEST),
        std::uint32_t(D3D12_RESOURCE_STATE_RENDER_TARGET | uav)}) {
      wrong = input; wrong.native_state = hint;
      reject(wrong, capture::status::unsupported_state, capture::record_stage::unsupported_state);
    }
    for (const auto hint : {D3D12_RESOURCE_STATE_RENDER_TARGET, uav,
        D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE}) {
      wrong = ordinary.input; wrong.frame_generation_input = frame_generation; wrong.native_state = hint;
      reject(wrong, capture::status::unsupported_state, capture::record_stage::unsupported_state);
    }
    const auto record = [&] {
      auto ticket = auxiliary_snapshot(gpu, shared, input, &diagnostic, policy);
      require(bool(ticket) && diagnostic.result == capture::status::recorded && !diagnostic.observed &&
        diagnostic.copy_state_known && diagnostic.copy_state == uav && diagnostic.native_state == uav &&
        !diagnostic.used_observed_state && diagnostic.recording_cookie &&
        diagnostic.format == unsigned(format) && diagnostic.flags == unsigned(flags),
        "absent recording evidence did not preserve explicit UAV declaration provenance");
      capture::finish_diagnostic_texture(ticket, true); return ticket;
    };
    auto original = record();
    // The copy must restore UAV for this real subsequent write to be legal.
    transition(gpu.list.Get(), image.source.Get(), uav, D3D12_RESOURCE_STATE_COPY_DEST);
    image.copy(gpu, 1);
    transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_COPY_DEST, uav);
    gpu.submit(); gpu.wait(); gpu.reset();
    auto changed = record(); // Same declaration on another fresh, unobserved recording.
    gpu.submit(); gpu.wait(); gpu.reset();

    auto first_target = destination(gpu.device.Get(), snapshot_format);
    auto next_target = destination(gpu.device.Get(), snapshot_format);
    readback first_bytes(gpu, first_target.Get()), next_bytes(gpu, next_target.Get());
    ComPtr<ID3D11Texture2D> host_lease;
    capture::diagnostic_texture host_snapshot;
    std::weak_ptr<const capture::texture_reference> host_storage;
    const auto read = [&](const capture::diagnostic_ticket &ticket, ID3D12Resource *target, readback &bytes, unsigned version) {
      capture::diagnostic_texture pixels;
      require(capture::acquire_diagnostic_texture(ticket, pixels) == capture::status::ready && bool(pixels.shared_handle) == shared &&
          pixels.format == unsigned(snapshot_format),
        "declared-state auxiliary copy lost its typed storage or sharing policy");
      if (shared) {
        image.verify_host(gpu, pixels, version, nullptr, snapshot_format);
        if (!version) {
          host_snapshot = pixels;
          host_storage = pixels.ownership;
          check(gpu.host1->OpenSharedResource1(reinterpret_cast<HANDLE>(pixels.shared_handle), IID_PPV_ARGS(&host_lease)),
            "retain declared-state shared host snapshot");
        }
      }
      require(capture::copy_diagnostic_texture(native(consumer.list.Get()), native(gpu.foreign_queue.Get()),
        ticket, native(target), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE), "declared-state auxiliary readback copy failed");
      bytes.record(consumer.list.Get(), target);
    };
    read(original, first_target.Get(), first_bytes, 0); read(changed, next_target.Get(), next_bytes, 1);
    consumer.submit(); consumer.wait(); consumer.reset();
    first_bytes.verify(image.width, image.height, image.bpp, 0);
    next_bytes.verify(image.width, image.height, image.bpp, 1);
    capture::release_diagnostic_texture(original); capture::release_diagnostic_texture(changed);
    original = {}; changed = {}; capture::poll();
    host_snapshot.ownership = {};
    capture::poll();
    if (shared) {
      require(host_storage.expired(), "shared declared-state storage retained an add-on lease after host acknowledgement");
      image.verify_host(gpu, host_snapshot, 0, host_lease.Get(), snapshot_format);
    }
    gpu.check_debug_errors();
    std::printf("PASS %s declared-state compatibility format %u->%u flags=%u FG=%u: prior-recording UAV, exact bytes/restoration, absent/malformed/flag guards%s\n",
      shared ? "shared" : "local", unsigned(format), unsigned(snapshot_format), unsigned(flags), unsigned(frame_generation),
      gpu.debug_messages ? "; D3D12 validation clean" : "; D3D12 debug layer unavailable");
  }

  void live_auxiliary_copy(fixture &gpu, DXGI_FORMAT format, unsigned bpp,
      DXGI_FORMAT target_format = DXGI_FORMAT_UNKNOWN, bool local = false,
      sunshine_scene_depth::lifetime lifetime = sunshine_scene_depth::lifetime::at_call) {
    if (target_format == DXGI_FORMAT_UNKNOWN) target_format = format;
    const bool typeless = format == DXGI_FORMAT_R8G8B8A8_TYPELESS || format == DXGI_FORMAT_B8G8R8A8_TYPELESS;
    if (typeless && gpu.debug_messages) gpu.debug_messages->ClearStoredMessages();
    consumer_fixture consumer(gpu);
    texture_case image(gpu, format, bpp);
    image.input.valid_until = lifetime;
    image.input.source_present_generation = capture::source_present_generation(image.input.source);
    auto target = destination(gpu.device.Get(), target_format);
    capture::consumer_diagnostic diagnostic;
    capture::record_diagnostic producer_diagnostic;
    const auto record = [&](const capture::input &input) {
      return local ? capture::record_local_texture(native(gpu.list.Get()), input, &producer_diagnostic) :
        capture::record_diagnostic_texture(native(gpu.list.Get()), input, &producer_diagnostic);
    };
    if (lifetime != sunshine_scene_depth::lifetime::at_call) {
      auto expired = image.input;
      expired.source_present_generation = 0;
      require(!record(expired) && producer_diagnostic.result == capture::status::unsupported_lifetime,
        "long-lived auxiliary source bypassed its presentation generation");
    }
    auto wrong_state = image.input;
    wrong_state.native_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    require(!record(wrong_state) && producer_diagnostic.result == capture::status::conflicting_state &&
        producer_diagnostic.observed_state == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE &&
        !producer_diagnostic.copy_state_known,
      "typed auxiliary normalization weakened the source-state guard");
    auto ticket = record(image.input);
    require(bool(ticket), "live auxiliary producer capture");
    require(producer_diagnostic.format == unsigned(format), "producer diagnostic lost original source format");
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
    capture::diagnostic_texture snapshot;
    const auto snapshot_format = typeless ? target_format : format;
    require(capture::acquire_diagnostic_texture(ticket, snapshot) == capture::status::ready &&
        snapshot.format == unsigned(snapshot_format) &&
        bool(snapshot.shared_handle) != local,
      "completed auxiliary snapshot lost its typed format or sharing policy");
    snapshot = {};
    if (typeless) {
      auto typeless_target = destination(gpu.device.Get(), format);
      require(!copy(native(typeless_target.Get())) && diagnostic.result == capture::consumer_status::invalid_destination,
        "normalized source allowed a typeless consumer destination");
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
      require((local ? source_lifetime.use_count() == 1 : source_lifetime.expired()) && target_lifetime.expired(),
        "completed auxiliary consumer retained a lease beyond the optional local storage cache");
    }
    if (typeless) gpu.check_debug_errors();
    std::printf("PASS %s auxiliary format %u->%u: exact cross-queue RGB/alpha, delayed producer/consumer, destination/state guards, bounded reuse\n",
      local ? "local" : "shared", unsigned(format), unsigned(target_format));
  }

  void local_same_queue_ordered_capture(fixture &gpu) {
    consumer_fixture consumer(gpu);
    texture_case image(gpu, DXGI_FORMAT_R10G10B10A2_UNORM, 4);
    auto target = destination(gpu.device.Get(), DXGI_FORMAT_R10G10B10A2_UNORM);
    auto other_target = destination(gpu.device.Get(), DXGI_FORMAT_R10G10B10A2_UNORM);
    auto local = capture::record_local_texture(native(gpu.list.Get()), image.input);
    auto shared = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input);
    auto failed = capture::record_local_texture(native(gpu.list.Get()), image.input);
    require(local && shared && failed, "same-queue ordered auxiliary capture setup failed");
    capture::finish_diagnostic_texture(shared, true);
    capture::finish_diagnostic_texture(failed, false);
    capture::diagnostic_texture pixels;
    capture::consumer_diagnostic diagnostic;
    const auto copy = [&](const capture::diagnostic_ticket &ticket, std::uint64_t queue, ID3D12Resource *output) {
      return capture::copy_local_texture(native(consumer.list.Get()), queue, ticket,
        native(output), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &diagnostic);
    };
    require(capture::acquire_local_texture(local, native(gpu.queue.Get()), pixels) == capture::status::recorded &&
        !copy(local, native(gpu.queue.Get()), target.Get()), "unsubmitted local snapshot became GPU-readable");
    readback actual(gpu, target.Get());
    std::weak_ptr<const capture::texture_reference> source_lifetime = local.ownership;
    std::weak_ptr<const capture::source_reference> target_lifetime;
    {
      gate delayed(gpu.device.Get(), gpu.queue.Get());
      gpu.submit(); // Keep the producer closed and its GPU submission blocked.
      const auto before = std::chrono::steady_clock::now();
      require(capture::acquire_local_texture(local, native(gpu.queue.Get()), pixels) == capture::status::recorded &&
          !copy(local, native(gpu.queue.Get()), target.Get()), "unfinished SDK call admitted local pixels");
      capture::finish_diagnostic_texture(local, true);
      require(capture::acquire_diagnostic_texture(local, pixels) == capture::status::submitted &&
          capture::acquire_local_texture(local, native(gpu.foreign_queue.Get()), pixels) == capture::status::submitted &&
          !copy(local, native(gpu.foreign_queue.Get()), target.Get()),
        "shared/foreign acquisition bypassed in-flight producer completion or recording retirement");
      require(capture::acquire_local_texture(shared, native(gpu.queue.Get()), pixels) == capture::status::unsupported_resource &&
          !copy(shared, native(gpu.queue.Get()), target.Get()), "shared dump ticket opted into local-only ordered reads");
      require(capture::acquire_local_texture(failed, native(gpu.queue.Get()), pixels) == capture::status::failed &&
          !copy(failed, native(gpu.queue.Get()), target.Get()), "failed SDK capture became readable through local ordering");
      require(capture::acquire_local_texture(local, native(gpu.queue.Get()), pixels) == capture::status::ready &&
          !pixels.producer_recording_retired && pixels.producer_fence && !pixels.shared_handle &&
          pixels.producer_queue == native(gpu.queue.Get()) && pixels.format == DXGI_FORMAT_R10G10B10A2_UNORM,
        "submitted local same-queue snapshot still required CPU completion or producer Reset");
      require(!capture::copy_diagnostic_texture(native(consumer.list.Get()), native(gpu.queue.Get()), local,
          native(target.Get()), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &diagnostic),
        "strict diagnostic copy was weakened by local snapshot support");
      require(copy(local, native(gpu.queue.Get()), target.Get()), "same-queue GPU-ordered local copy was rejected");
      // Observe the wrapper retained by the successful copy. The source
      // registry itself is weak; an earlier temporary wrapper is a different
      // lifetime and can expire before this destination lease is established.
      auto target_reference = capture::retain_source(native(target.Get()));
      target_lifetime = target_reference;
      target_reference = {};
      require(!copy(local, native(gpu.queue.Get()), other_target.Get()) &&
          !copy(local, native(gpu.foreign_queue.Get()), target.Get()),
        "ordered local copy rebound its destination or consuming queue");
      require(std::chrono::steady_clock::now() - before < std::chrono::milliseconds(100),
        "local auxiliary admission/copy waited for GPU completion");
      actual.record(consumer.list.Get(), target.Get());
      check(consumer.list->Close(), "ordered consumer close");
      ID3D12CommandList *submitted[]{consumer.list.Get()};
      gpu.queue->ExecuteCommandLists(1, submitted);
      pixels = {};
      capture::release_diagnostic_texture(local); local = {};
      capture::release_diagnostic_texture(shared); shared = {};
      capture::release_diagnostic_texture(failed); failed = {};
      target.Reset(); gpu.reset(); consumer.reset(); capture::poll();
      require(!source_lifetime.expired() && !target_lifetime.expired(),
        "ordered consumer Reset released source/destination before queued GPU work completed");
      delayed.open(); gpu.wait(); actual.verify(image.width, image.height, 4); capture::poll();
      require(target_lifetime.expired(), "ordered local consumer retained destination after GPU retirement");
    }
    gpu.check_debug_errors();
    std::puts("PASS local same-queue pending GPU copy: strict shared/foreign rejection, exact R10 bytes, no wait, fixed consumer/destination and fence retirement");
  }

  void immediate_consumer_without_list_hooks(fixture &gpu) {
    // Another tool can swap a consumer list's vtable for a per-object heap
    // copy that is never hooked (seen in The Witcher 3). A runtime immediate
    // list needs no list hook: its observed submission and the queue fence
    // retire the read, and nothing is released before that GPU work.
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "immediate allocator");
    check(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
      "immediate list");
    auto **object = reinterpret_cast<void ***>(list.Get());
    void **const original = *object;
    constexpr std::size_t list7_methods = 81; // ID3D12GraphicsCommandList7::Barrier is slot 80.
    std::vector<void *> swapped(original, original + list7_methods);
    *object = swapped.data();
    struct restore_vtable { void ***object; void **table; ~restore_vtable() { *object = table; } } restore{object, original};
    // The first sighting reports the refused slots once, revoking recordings
    // open at that moment. Later recordings and captures persist.
    capture::observe_command(native(list.Get())); capture::poll();
    check(gpu.list->Close(), "close recording open during refusal"); gpu.reset();
    texture_case image(gpu, DXGI_FORMAT_R10G10B10A2_UNORM, 4);
    auto target = destination(gpu.device.Get(), DXGI_FORMAT_R10G10B10A2_UNORM);
    auto local = capture::record_local_texture(native(gpu.list.Get()), image.input);
    require(bool(local), "immediate consumer source capture");
    gpu.submit();
    capture::finish_diagnostic_texture(local, true);
    capture::diagnostic_texture pixels;
    capture::consumer_diagnostic diagnostic;
    require(capture::acquire_local_texture(local, native(gpu.queue.Get()), pixels) == capture::status::ready,
      "submitted same-queue local snapshot was not readable");
    const auto copy = [&](bool immediate) {
      return capture::copy_local_texture(native(list.Get()), native(gpu.queue.Get()), local,
        native(target.Get()), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &diagnostic, immediate);
    };
    require(!copy(false) && diagnostic.result == capture::consumer_status::observer_not_ready,
      "a list with a swapped heap vtable became hook-ready");
    require(copy(true) && copy(true), "immediate consumer without list hooks was rejected");
    readback actual(gpu, target.Get());
    actual.record(list.Get(), target.Get());
    auto target_reference = capture::retain_source(native(target.Get()));
    std::weak_ptr<const capture::source_reference> target_lifetime = target_reference;
    target_reference = {};
    pixels = {};
    capture::release_diagnostic_texture(local); local = {};
    target.Reset(); capture::poll();
    require(!target_lifetime.expired(), "an unsubmitted immediate read released its destination");
    check(list->Close(), "immediate close");
    ID3D12CommandList *values[]{list.Get()};
    gpu.queue->ExecuteCommandLists(1, values);
    gpu.wait(); actual.verify(image.width, image.height, 4); gpu.reset(); capture::poll();
    require(target_lifetime.expired(), "the submitted immediate read was not retired by its queue fence");
    gpu.check_debug_errors();
    std::puts("PASS immediate consumer with an unhookable swapped vtable: exact R10 bytes, retained until its observed submission and fence");
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
  void captured_mutation(fixture &gpu, DXGI_FORMAT format, unsigned bpp,
      DXGI_FORMAT snapshot_format = DXGI_FORMAT_UNKNOWN) {
    if (snapshot_format == DXGI_FORMAT_UNKNOWN) snapshot_format = format;
    if (snapshot_format != format && gpu.debug_messages) gpu.debug_messages->ClearStoredMessages();
    texture_case image(gpu, format, bpp);
    auto wrong = image.input; wrong.native_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    capture::record_diagnostic diagnostic;
    require(!capture::record_diagnostic_texture(native(gpu.list.Get()), wrong, &diagnostic) &&
      diagnostic.result == capture::status::conflicting_state && diagnostic.observed && !diagnostic.blocked &&
      diagnostic.native_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS &&
      diagnostic.observed_state == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE && diagnostic.recording_cookie,
      "wrong declared UAV state accepted or rejection lost the observed state");
    auto ticket = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input, &diagnostic);
    if (!ticket) { std::fprintf(stderr, "record result=%s stage=%s\n", capture::name(diagnostic.result), capture::name(diagnostic.stage)); }
    require(bool(ticket), "typed diagnostic copy not recorded");
    require(diagnostic.format == unsigned(format), "snapshot normalization changed original source diagnostic format");
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
    image.verify_host(gpu, snapshot, 0, nullptr, snapshot_format);
    capture::diagnostic_texture changed;
    require(capture::acquire_diagnostic_texture(after_mutation, changed) == capture::status::ready, "later capture unavailable");
    image.verify_host(gpu, changed, 1, nullptr, snapshot_format); // The snapshot must not interfere with the game's later write.
    capture::release_diagnostic_texture(after_mutation); changed = {}; after_mutation = {};
    capture::release_diagnostic_texture(ticket);
    capture::diagnostic_texture released;
    require(capture::acquire_diagnostic_texture(ticket, released) == capture::status::stale, "released ticket resurrected");
    image.verify_host(gpu, snapshot, 0, nullptr, snapshot_format); // Existing consumer lease remains valid.
    snapshot = {}; ticket = {}; capture::poll();
    if (snapshot_format != format) gpu.check_debug_errors();
    std::printf("PASS source format %u->%u: immutable RGB/alpha bytes, crop, typed shared D3D11 host, state guard, nonblocking retirement\n",
      unsigned(format), unsigned(snapshot_format));
  }
  void retired_storage_reuse(fixture &gpu) {
    texture_case image(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    consumer_fixture consumer(gpu);
    auto target = destination(gpu.device.Get(), DXGI_FORMAT_R8G8B8A8_UNORM);
    readback actual(gpu, target.Get());
    // The host can acknowledge opening a dump before its own readback retires.
    // Its COM reference is invisible to addon shared_ptr ownership, so an
    // exported allocation must remain immutable after all addon leases expire.
    auto exported = capture::record_diagnostic_texture(native(gpu.list.Get()), image.input);
    require(bool(exported), "immutable exported capture setup");
    capture::finish_diagnostic_texture(exported, true); gpu.submit(); gpu.wait(); gpu.reset();
    capture::diagnostic_texture original;
    require(capture::acquire_diagnostic_texture(exported, original) == capture::status::ready, "exported capture unavailable");
    ComPtr<ID3D11Texture2D> host_pixels;
    check(gpu.host1->OpenSharedResource1(reinterpret_cast<HANDLE>(original.shared_handle), IID_PPV_ARGS(&host_pixels)), "host opens before ack");
    std::weak_ptr<const capture::texture_reference> exported_owner = exported.ownership;
    capture::release_diagnostic_texture(exported); exported = {}; original.ownership = {}; capture::poll();
    require(exported_owner.expired(), "exported snapshot was cached for unsafe overwrite after host ack");
    std::weak_ptr<const capture::texture_reference> storage;
    std::uint64_t previous_id{};
    for (unsigned frame = 0; frame != 8; ++frame) {
      if (frame) {
        transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        image.copy(gpu, frame % 2);
        transition(gpu.list.Get(), image.source.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      }
      auto ticket = capture::record_local_texture(native(gpu.list.Get()), image.input);
      require(bool(ticket) && ticket.id != previous_id, "reused storage kept a stale capture identity");
      if (frame) require(ticket.ownership == storage.lock(), "retired matching storage was allocated again");
      capture::finish_diagnostic_texture(ticket, true); gpu.submit(); gpu.wait(); gpu.reset();
      capture::diagnostic_texture pixels;
      require(capture::acquire_diagnostic_texture(ticket, pixels) == capture::status::ready, "reused capture unavailable");
      require(!pixels.shared_handle, "reusable local pixels exposed an unsafe IPC handle");
      require(capture::copy_diagnostic_texture(native(consumer.list.Get()), native(gpu.foreign_queue.Get()),
        ticket, native(target.Get()), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE), "reused local snapshot copy unavailable");
      actual.record(consumer.list.Get(), target.Get()); consumer.submit(); consumer.wait(); consumer.reset();
      actual.verify(image.width, image.height, image.bpp, frame % 2);
      image.verify_host(gpu, original, 0, host_pixels.Get());
      storage = ticket.ownership; previous_id = ticket.id;
      capture::release_diagnostic_texture(ticket); ticket = {}; pixels = {}; capture::poll();
      require(storage.use_count() == 1, "retired capture did not retain exactly one cache owner");
    }
    std::puts("PASS local snapshot reuse with fresh IDs/exact pixels; exported pixels stay immutable after host ack");
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
    first_recording_depth_capture(gpu);
    ngx_contract_depth_capture(gpu);
    local_same_queue_ordered_capture(gpu);
    immediate_consumer_without_list_hooks(gpu);
    for (const bool shared : {false, true}) {
      observed_recording_color_capture(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, shared);
      observed_recording_color_capture(gpu, DXGI_FORMAT_R10G10B10A2_UNORM, shared);
      // Exact Hogwarts tag53 conflict: format24, flags5, declared UAV8 / observed192.
      observed_recording_color_capture(gpu, DXGI_FORMAT_R10G10B10A2_UNORM, shared,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      // HUDLess tag2 has the same state-proof contract as UI tags. Mask semantics
      // and absent application RT/UAV flags must not change that contract either.
      observed_recording_color_capture(gpu, DXGI_FORMAT_R10G10B10A2_UNORM, shared,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        DXGI_FORMAT_R10G10B10A2_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        false, D3D12_RESOURCE_STATE_COPY_DEST);
      observed_recording_color_capture(gpu, DXGI_FORMAT_R8_UNORM, shared,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_FLAG_NONE, false, D3D12_RESOURCE_STATE_COPY_DEST);
      observed_recording_color_capture(gpu, DXGI_FORMAT_B8G8R8A8_TYPELESS, shared,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        DXGI_FORMAT_B8G8R8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, false);
      declared_recording_color_capture(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, shared);
      declared_recording_color_capture(gpu, DXGI_FORMAT_R10G10B10A2_UNORM, shared);
      declared_recording_color_capture(gpu, DXGI_FORMAT_B8G8R8A8_TYPELESS, shared, DXGI_FORMAT_B8G8R8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, false);
    }
    retired_storage_reuse(gpu);
    captured_mutation(gpu, DXGI_FORMAT_R8_UNORM, 1);
    captured_mutation(gpu, DXGI_FORMAT_R8_UINT, 1);
    captured_mutation(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    captured_mutation(gpu, DXGI_FORMAT_R8G8B8A8_TYPELESS, 4, DXGI_FORMAT_R8G8B8A8_UNORM);
    captured_mutation(gpu, DXGI_FORMAT_B8G8R8A8_TYPELESS, 4, DXGI_FORMAT_B8G8R8A8_UNORM);
    captured_mutation(gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, 8);
    live_auxiliary_copy(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    live_auxiliary_copy(gpu, DXGI_FORMAT_R10G10B10A2_UNORM, 4);
    live_auxiliary_copy(gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, 8);
    live_auxiliary_copy(gpu, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 4, DXGI_FORMAT_R8G8B8A8_UNORM);
    live_auxiliary_copy(gpu, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, 4, DXGI_FORMAT_B8G8R8A8_UNORM);
    live_auxiliary_copy(gpu, DXGI_FORMAT_R8G8B8A8_TYPELESS, 4, DXGI_FORMAT_R8G8B8A8_UNORM, true);
    live_auxiliary_copy(gpu, DXGI_FORMAT_B8G8R8A8_TYPELESS, 4, DXGI_FORMAT_B8G8R8A8_UNORM, true);
    for (const auto lifetime : {sunshine_scene_depth::lifetime::until_present, sunshine_scene_depth::lifetime::until_evaluation})
      for (const bool local : {false, true}) {
        live_auxiliary_copy(gpu, DXGI_FORMAT_B8G8R8A8_TYPELESS, 4, DXGI_FORMAT_B8G8R8A8_UNORM, local, lifetime);
        std::printf("PASS auxiliary lifetime=%u %s: strict-state admission/rejection, generation guard and exact BGRA bytes\n",
          unsigned(lifetime), local ? "local" : "shared");
      }
    auxiliary_consumer_replay_and_discard(gpu);
    replay_and_cancel(gpu);
    std::puts("PASS diagnostic texture capture functional GPU regression (no performance measurement)");
    return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
