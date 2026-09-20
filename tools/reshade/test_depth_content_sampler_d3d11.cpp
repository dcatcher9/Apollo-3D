// SPDX-License-Identifier: GPL-3.0-only
// Opt-in native GPU fixture. Includes the production implementation to exercise
// its exact format, coordinate, and private-command-list path without a mock SDK.
// Never run automatically as part of the ordinary host test suite.
#include "depth_content_sampler.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
  using sunshine_depth::com_ptr;
  using sunshine_depth::pipeline_cache_t;
  namespace api = reshade::api;

  void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }
  void checked(HRESULT result, const char *message) {
    if (FAILED(result)) {
      char text[256];
      std::snprintf(text, sizeof(text), "%s (0x%08lx)", message, static_cast<unsigned long>(result));
      throw std::runtime_error(text);
    }
  }

  constexpr UINT source_width = 96, source_height = 54;
#ifdef SUNSHINE_SAMPLER_BASELINE
  constexpr UINT sample_pixel_bytes = sizeof(float);
  constexpr UINT output_logical_bytes = 32 * 18 * sample_pixel_bytes;
#else
  constexpr UINT sample_pixel_bytes = 4 * sizeof(float);
  constexpr UINT output_logical_bytes = sunshine_depth_statistics::maximum_tiles * 2 * sample_pixel_bytes;
#endif
  UINT point_coordinate(UINT index, UINT extent, UINT tiles) {
#ifdef SUNSHINE_SAMPLER_BASELINE
    return UINT((std::uint64_t(2 * index + 1) * extent) / (2 * tiles));
#else
    const UINT first = UINT(std::uint64_t(index) * extent / tiles);
    const UINT last = UINT(std::uint64_t(index + 1) * extent / tiles);
    return first + (last - first) / 2;
#endif
  }
  float scene_depth(UINT x, UINT y) {
    return .1f + .6f * float(x) / source_width + .2f * float(y) / source_height;
  }

  void measure_cache(api::device_api kind, std::uint64_t native, ID3DBlob *shader) {
    pipeline_cache_t cache;
    int identity{};
    auto *owner = reinterpret_cast<api::effect_runtime *>(&identity);
    require(bool(cache.acquire(kind, native, owner, shader)), "Warm native pipeline timing");
    constexpr unsigned fresh_count = 8, reused_count = 1024;
    const auto fresh_start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < fresh_count; ++i) {
      cache.retire(owner);
      require(bool(cache.acquire(kind, native, owner, shader)), "Create fresh native pipeline during timing");
    }
    const auto fresh_end = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < reused_count; ++i)
      require(bool(cache.acquire(kind, native, owner, shader)), "Acquire cached native pipeline during timing");
    const auto reused_end = std::chrono::steady_clock::now();
    const double fresh_us = std::chrono::duration<double, std::micro>(fresh_end - fresh_start).count() / fresh_count;
    const double reused_us = std::chrono::duration<double, std::micro>(reused_end - fresh_end).count() / reused_count;
    std::printf("MEASURE immutable native %s program: fresh_mean_us=%.6f (%u), reused_mean_us=%.6f (%u); CPU wall time, no speed threshold\n",
      kind == api::device_api::d3d12 ? "D3D12" : "D3D11", fresh_us, fresh_count, reused_us, reused_count);
  }

  void run_format(ID3D11Device *device, ID3D11DeviceContext *context, ID3DBlob *shader,
                  pipeline_cache_t &cache, api::effect_runtime *owner,
                  DXGI_FORMAT format, unsigned pixel_bytes, bool use_viewport) {
    const UINT x0 = use_viewport ? 8 : 0, y0 = use_viewport ? 5 : 0;
    const UINT region_width = use_viewport ? 64 : source_width;
    const UINT region_height = use_viewport ? 36 : source_height;
    std::vector<unsigned char> source_data(size_t(source_width) * source_height * pixel_bytes);
    for (UINT y = 0; y < source_height; ++y) for (UINT x = 0; x < source_width; ++x) {
      const float value = x >= x0 && x < x0 + region_width && y >= y0 && y < y0 + region_height ? scene_depth(x, y) : 0.f;
      auto *pixel = source_data.data() + (size_t(y) * source_width + x) * pixel_bytes;
      if (format == DXGI_FORMAT_R16_TYPELESS) {
        const auto encoded = std::uint16_t(std::lround(value * 65535.f));
        std::memcpy(pixel, &encoded, 2);
      } else if (format == DXGI_FORMAT_R24G8_TYPELESS) {
        const auto encoded = std::uint32_t(std::lround(double(value) * 16777215.0)) | 0xA5000000u;
        std::memcpy(pixel, &encoded, 4);
      } else {
        std::memcpy(pixel, &value, 4);
        if (pixel_bytes == 8) pixel[4] = 0xA5; // Depth SRV must ignore stencil.
      }
    }
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = source_width;
    desc.Height = source_height;
    desc.ArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Format = format;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {source_data.data(), source_width * pixel_bytes, 0};
    com_ptr<ID3D11Texture2D> source;
    checked(device->CreateTexture2D(&desc, &initial, source.put()), "Create depth backup source");

    sunshine_depth::sample_request request;
    request.token = 77;
    request.source.handle = reinterpret_cast<std::uint64_t>(source.get());
    request.source_width = source_width;
    request.source_height = source_height;
    request.x = x0; request.y = y0;
    request.width = region_width; request.height = region_height;
#ifndef SUNSHINE_SAMPLER_BASELINE
    request.collect_range = use_viewport;
#endif
    sunshine_depth::sample_t sample;
    sample.native_queue = reinterpret_cast<std::uint64_t>(context);
    sample.pipeline = cache.acquire(api::device_api::d3d11, reinterpret_cast<std::uint64_t>(device), owner, shader);
    require(sunshine_depth::prepare11(sample, request), "Production D3D11 sampler preparation failed");

    // Install different COM objects in every CS slot the private list changes.
    // Checking those identities after execution detects leaking sampler state.
    com_ptr<ID3D11ComputeShader> sentinel_shader;
    com_ptr<ID3D11Buffer> sentinel_constants;
    com_ptr<ID3D11ShaderResourceView> sentinel_srv;
    com_ptr<ID3D11Texture2D> sentinel_output;
    com_ptr<ID3D11UnorderedAccessView> sentinel_uav;
    checked(device->CreateComputeShader(shader->GetBufferPointer(), shader->GetBufferSize(), nullptr, sentinel_shader.put()), "Create sentinel shader");
    D3D11_BUFFER_DESC constant_desc {};
    constant_desc.ByteWidth = 16;
    constant_desc.Usage = D3D11_USAGE_DEFAULT;
    constant_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    checked(device->CreateBuffer(&constant_desc, nullptr, sentinel_constants.put()), "Create sentinel constants");
    D3D11_SHADER_RESOURCE_VIEW_DESC view {};
    view.Format = sunshine_depth::depth_srv_format(format);
    view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    view.Texture2D.MipLevels = 1;
    checked(device->CreateShaderResourceView(source.get(), &view, sentinel_srv.put()), "Create sentinel depth SRV");
    desc.Width = desc.Height = 8;
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    checked(device->CreateTexture2D(&desc, nullptr, sentinel_output.put()), "Create sentinel output");
    checked(device->CreateUnorderedAccessView(sentinel_output.get(), nullptr, sentinel_uav.put()), "Create sentinel UAV");
    auto *srv = sentinel_srv.get();
    auto *uav = sentinel_uav.get();
    auto *constants = sentinel_constants.get();
    context->CSSetShader(sentinel_shader.get(), nullptr, 0);
    context->CSSetShaderResources(0, 1, &srv);
    context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
    context->CSSetConstantBuffers(0, 1, &constants);
    context->ExecuteCommandList(sample.commands11.get(), TRUE);
    com_ptr<ID3D11ComputeShader> actual_shader;
    com_ptr<ID3D11Buffer> actual_constants;
    com_ptr<ID3D11ShaderResourceView> actual_srv;
    com_ptr<ID3D11UnorderedAccessView> actual_uav;
    context->CSGetShader(actual_shader.put(), nullptr, nullptr);
    context->CSGetConstantBuffers(0, 1, actual_constants.put());
    context->CSGetShaderResources(0, 1, actual_srv.put());
    context->CSGetUnorderedAccessViews(0, 1, actual_uav.put());
    require(actual_shader.get() == sentinel_shader.get() && actual_constants.get() == sentinel_constants.get() &&
      actual_srv.get() == sentinel_srv.get() && actual_uav.get() == sentinel_uav.get(), "Sampler changed caller's compute state");

    // Waiting belongs only to this explicit fixture. Production polling never
    // flushes or waits for a query and never maps before confirmed completion.
    context->Flush();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    BOOL complete = FALSE;
    while (!complete && std::chrono::steady_clock::now() < deadline) {
      checked(context->GetData(sample.query11.get(), &complete, sizeof(complete), D3D11_ASYNC_GETDATA_DONOTFLUSH), "Poll sample event");
      if (!complete) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(complete, "Sampler event did not finish");
    D3D11_MAPPED_SUBRESOURCE mapped {};
    checked(context->Map(sample.readback11.get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped), "Map finished sample");
    float max_error = 0;
    for (UINT y = 0; y < 18; ++y) for (UINT x = 0; x < 32; ++x) {
      float actual;
      std::memcpy(&actual, static_cast<const unsigned char *>(mapped.pData) + y * mapped.RowPitch + x * sample_pixel_bytes, sizeof(float));
      const UINT source_x = x0 + ((2 * x + 1) * region_width) / 64;
      const UINT source_y = y0 + ((2 * y + 1) * region_height) / 36;
      require(std::isfinite(actual), "Depth sampler returned non-finite data");
      max_error = std::max(max_error, std::abs(actual - scene_depth(source_x, source_y)));
    }
    context->Unmap(sample.readback11.get(), 0);
    require(max_error < .00002f, "Depth grid differs from independent coordinate/format oracle");
#ifndef SUNSHINE_SAMPLER_BASELINE
    sunshine_depth::read_sample(sample);
    require(sample.result.valid && sample.result.range_valid == use_viewport,
      "Optional range changed point-grid readiness or was collected without request");
    if (use_viewport) require(std::abs(sample.result.range_min - scene_depth(x0, y0)) < .00002f &&
      std::abs(sample.result.range_max - scene_depth(x0 + region_width - 1, y0 + region_height - 1)) < .00002f,
      "Full-range reduction lost source-format precision or sampled allocation padding");
#endif
    require(sample.result.token == 77 && sample.result.viewport_width == region_width && sample.result.viewport_height == region_height,
      "Sampler lost source identity or viewport metadata");
    std::printf("PASS D3D11 sampler format=%u viewport=%s grid oracle max_error=%.9g, caller CS state restored\n", unsigned(format), use_viewport ? "padded" : "full", max_error);
    context->ClearState();
  }

  void verify_device_cache(ID3D11Device *device, ID3DBlob *shader) {
    pipeline_cache_t cache;
    // These addresses are owner identities only; the production cache never
    // calls through them. Every device/program below is a real native object.
    int identity_a{}, identity_b{};
    auto *owner_a = reinterpret_cast<api::effect_runtime *>(&identity_a);
    auto *owner_b = reinterpret_cast<api::effect_runtime *>(&identity_b);
    const auto native = reinterpret_cast<std::uint64_t>(device);
    auto first = cache.acquire(api::device_api::d3d11, native, owner_a, shader);
    require(first && first->device11.get() == device && first->shader11.get(), "Initial native shader cache creation failed");
    for (unsigned i = 0; i < 32; ++i) {
      const auto reused = cache.acquire(api::device_api::d3d11, native, owner_a, shader);
      require(reused == first && reused->shader11.get() == first->shader11.get(), "Repeated sample recreated immutable D3D11 program");
    }
    require(cache.acquire(api::device_api::d3d11, native, owner_b, shader) == first, "Same-device runtimes did not share immutable program");
    cache.retire(owner_a);
    require(cache.acquire(api::device_api::d3d11, native, owner_b, shader) == first, "Retiring prior runtime evicted active borrower's program");
    std::weak_ptr<sunshine_depth::device_pipeline_t> retired = first;
    cache.retire(owner_b);
    require(!retired.expired(), "Retirement discarded a sample-held program");
    const auto replacement = cache.acquire(api::device_api::d3d11, native, owner_a, shader);
    require(replacement && replacement != first, "Retired runtime retained an idle cache entry");
    first.reset();
    require(retired.expired(), "Unreferenced retired program still pins native device");

    com_ptr<ID3D11Device> other_device;
    com_ptr<ID3D11DeviceContext> other_context;
    checked(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
      D3D11_SDK_VERSION, other_device.put(), nullptr, other_context.put()), "Create second native D3D11 device");
    require(other_device.get() != device, "Fixture requires distinct native D3D11 devices");
    const auto other = cache.acquire(api::device_api::d3d11, reinterpret_cast<std::uint64_t>(other_device.get()), owner_a, shader);
    require(other && other != replacement && other->device11.get() == other_device.get(), "New native device reused old device's program");
    require(replacement->device11.get() == device && replacement->shader11.get(), "Cache replacement discarded outstanding old-device owner");
    cache.retire(owner_a);
    std::puts("PASS native immutable cache: repeated shader reuse, shared-runtime retirement, new-device replacement and final-owner release");
    measure_cache(api::device_api::d3d11, native, shader);
  }

  std::weak_ptr<sunshine_depth::device_pipeline_t> run_d3d12(ID3DBlob *shader) {
    com_ptr<ID3D12Device> device;
    checked(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())), "Create native D3D12 device");
    measure_cache(api::device_api::d3d12, reinterpret_cast<std::uint64_t>(device.get()), shader);
    com_ptr<ID3D12CommandQueue> queue;
    const D3D12_COMMAND_QUEUE_DESC queue_desc {D3D12_COMMAND_LIST_TYPE_DIRECT};
    checked(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(queue.put())), "Create sampler fixture queue");
    D3D12_RESOURCE_DESC source_desc {};
    source_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    source_desc.Width = source_width;
    source_desc.Height = source_height;
    source_desc.DepthOrArraySize = source_desc.MipLevels = source_desc.SampleDesc.Count = 1;
    source_desc.Format = DXGI_FORMAT_R32_FLOAT;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    com_ptr<ID3D12Resource> source;
    checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &source_desc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(source.put())), "Create native D3D12 depth backup");
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
    UINT64 upload_bytes = 0;
    device->GetCopyableFootprints(&source_desc, 0, 1, 0, &footprint, nullptr, nullptr, &upload_bytes);
    D3D12_RESOURCE_DESC upload_desc {};
    upload_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    upload_desc.Width = upload_bytes;
    upload_desc.Height = upload_desc.DepthOrArraySize = upload_desc.MipLevels = upload_desc.SampleDesc.Count = 1;
    upload_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    com_ptr<ID3D12Resource> upload;
    checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(upload.put())), "Create depth upload resource");
    void *upload_data = nullptr;
    const D3D12_RANGE empty {0, 0};
    checked(upload->Map(0, &empty, &upload_data), "Map depth upload resource");
    for (UINT y = 0; y < source_height; ++y) for (UINT x = 0; x < source_width; ++x) {
      const float value =
#ifndef SUNSHINE_SAMPLER_BASELINE
        x == 8 && y == 5 ? 0.f : x == 70 && y == 5 ? 1.f :
        x == 0 && y == 0 ? std::numeric_limits<float>::quiet_NaN() :
#endif
        scene_depth(x, y);
      std::memcpy(static_cast<unsigned char *>(upload_data) + footprint.Offset + y * footprint.Footprint.RowPitch + x * sizeof(float), &value, sizeof(value));
    }
    upload->Unmap(0, nullptr);
    com_ptr<ID3D12CommandAllocator> allocator;
    com_ptr<ID3D12GraphicsCommandList> commands;
    checked(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.put())), "Create upload allocator");
    checked(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.get(), nullptr,
      IID_PPV_ARGS(commands.put())), "Create upload command list");
    D3D12_TEXTURE_COPY_LOCATION from {}, to {};
    from.pResource = upload.get(); from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
    to.pResource = source.get(); to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    checked(commands->Close(), "Close depth upload command list");

    pipeline_cache_t cache;
    int identity{};
    auto *owner = reinterpret_cast<api::effect_runtime *>(&identity);
    const auto native = reinterpret_cast<std::uint64_t>(device.get());
    sunshine_depth::sample_request request;
    request.token = 91;
    request.source = {reinterpret_cast<std::uint64_t>(source.get())};
    request.source_width = source_width; request.source_height = source_height;
    request.x = 8; request.y = 5; request.width = 64; request.height = 36;
#ifndef SUNSHINE_SAMPLER_BASELINE
    request.collect_moments = true; // Implies exact range in the same dispatch.
    request.moments_A = 1.f;
    request.moments_inverseB = -3.f;
#endif
    sunshine_depth::sample_t sample;
    sample.native_queue = reinterpret_cast<std::uint64_t>(queue.get());
    sample.pipeline = cache.acquire(api::device_api::d3d12, native, owner, shader);
    require(sample.pipeline && sample.pipeline->root12.get() && sample.pipeline->pipeline12.get(), "Create cached native D3D12 program");
    for (unsigned i = 0; i < 32; ++i) {
      const auto reused = cache.acquire(api::device_api::d3d12, native, owner, shader);
      require(reused == sample.pipeline && reused->root12.get() == sample.pipeline->root12.get() &&
        reused->pipeline12.get() == sample.pipeline->pipeline12.get(), "Repeated sample recreated immutable D3D12 root/PSO");
    }
    require(sunshine_depth::prepare12(sample, request), "Prepare production D3D12 sample with cached program");
    ID3D12CommandList *lists[] {commands.get(), sample.commands12.get()};
    queue->ExecuteCommandLists(2, lists);
    checked(queue->Signal(sample.fence12.get(), 1), "Signal native D3D12 sample");
    std::weak_ptr<sunshine_depth::device_pipeline_t> retired = sample.pipeline;
    cache.retire(owner);
    require(!retired.expired(), "Retirement freed submitted sample's D3D12 program");
    const auto replacement = cache.acquire(api::device_api::d3d12, native, owner, shader);
    require(replacement && replacement != sample.pipeline, "D3D12 retirement failed to release idle cache");

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (sample.fence12->GetCompletedValue() < 1 && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto completed = sample.fence12->GetCompletedValue();
    require(completed != UINT64_MAX && completed >= 1, "Native D3D12 sample did not complete");
    const D3D12_RANGE read_range {0, SIZE_T(sample.bytes12)};
    void *readback = nullptr;
    checked(sample.readback12->Map(0, &read_range, &readback), "Map native D3D12 sample");
    float max_error = 0;
    for (UINT y = 0; y < 18; ++y) for (UINT x = 0; x < 32; ++x) {
      float actual = 0;
      std::memcpy(&actual, static_cast<const unsigned char *>(readback) + sample.footprint12.Offset +
        y * sample.footprint12.Footprint.RowPitch + x * sample_pixel_bytes, sizeof(actual));
      const float expected = scene_depth(request.x + ((2 * x + 1) * request.width) / 64,
        request.y + ((2 * y + 1) * request.height) / 36);
      require(std::isfinite(actual), "D3D12 sample returned nonfinite data");
      max_error = std::max(max_error, std::abs(actual - expected));
    }
    sample.readback12->Unmap(0, &empty);
    require(max_error == 0.f && sample.result.token == 91, "Cached D3D12 sample failed independent coordinate/value oracle");
#ifndef SUNSHINE_SAMPLER_BASELINE
    sunshine_depth::read_sample(sample);
    require(sample.result.valid && sample.result.range_valid && sample.result.range_min == 0.f && sample.result.range_max == 1.f,
      "D3D12 exact range missed one-pixel endpoints between point samples or included nonfinite padding");
    double expected_sum = 0., expected_squares = 0.;
    for (UINT y = request.y; y < request.y + request.height; ++y)
      for (UINT x = request.x; x < request.x + request.width; ++x) {
        const float raw = x == 8 && y == 5 ? 0.f : x == 70 && y == 5 ? 1.f : scene_depth(x, y);
        const float q = (raw - request.moments_A) * request.moments_inverseB;
        expected_sum += q;
        expected_squares += double(q) * q;
      }
    const auto &moments = sample.result.moments;
    require(moments.supplied && moments.valid && moments.count == 64 * 36 &&
      moments.tiles_x == 32 && moments.tiles_y == 18 &&
      std::abs(moments.sum - expected_sum) <= expected_sum * 3e-5 &&
      std::abs(moments.sum_squares - expected_squares) <= expected_squares * 3e-5,
      "D3D12 full-image decoded moments differ from the independent per-pixel oracle");
#endif
    std::printf("PASS native D3D12 cached root/PSO: repeated reuse, retirement during submitted sample, replacement and padded grid oracle max_error=%.9g\n", max_error);
    return retired;
  }
  struct native_source {
    com_ptr<ID3D11Texture2D> texture11;
    com_ptr<ID3D12Resource> texture12;
    UINT width = 0, height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    unsigned index = 0;
    bool *destroyed = nullptr;
    std::uint64_t native() const {
      return texture11.get() ? reinterpret_cast<std::uint64_t>(texture11.get()) :
        reinterpret_cast<std::uint64_t>(texture12.get());
    }
    float expected(UINT x, UINT y) const {
      const float ramp = .6f * float(x) / width + .2f * float(y) / height;
      return index ? .9f - ramp : .1f + ramp;
    }
  };

  template<class T> void reset_com(com_ptr<T> &value) {
#ifdef SUNSHINE_SAMPLER_BASELINE
    if (value.get()) value->Release();
    *value.put() = nullptr;
#else
    value.reset();
#endif
  }

  UINT64 fence_target(const sunshine_depth::sample_t &sample) {
#ifdef SUNSHINE_SAMPLER_BASELINE
    (void)sample;
    return 1;
#else
    return sample.fence_value12;
#endif
  }

  // This fixture owns all waits. Production submits once, polls without flushing
  // and never recycles its bundle until the matching query/fence has completed.
  void wait_sample(sunshine_depth::sample_t &sample) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    if (sample.fence12.get()) {
      UINT64 completed = 0;
      do {
        completed = sample.fence12->GetCompletedValue();
        require(completed != UINT64_MAX, "D3D12 device removed during repeated samples");
        if (completed >= fence_target(sample)) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (std::chrono::steady_clock::now() < deadline);
    } else {
      sample.context11->Flush();
      BOOL ready = FALSE;
      do {
        checked(sample.context11->GetData(sample.query11.get(), &ready, sizeof(ready),
          D3D11_ASYNC_GETDATA_DONOTFLUSH), "Repeated sample query");
        if (ready) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (std::chrono::steady_clock::now() < deadline);
    }
    throw std::runtime_error("Repeated sample did not complete");
  }

#ifndef SUNSHINE_SAMPLER_BASELINE
  void run_exact_range(ID3D11Device *device, ID3D11DeviceContext *context, ID3DBlob *shader) {
    pipeline_cache_t cache;
    int identity{};
    auto *owner = reinterpret_cast<api::effect_runtime *>(&identity);
    auto pipeline = cache.acquire(api::device_api::d3d11, reinterpret_cast<std::uint64_t>(device), owner, shader);
    std::unique_ptr<sunshine_depth::sample_t> spare;
    // The odd larger extent exercises strided tile scans and uneven integer
    // partitions; a single texel caps the common grid to one nonempty tile.
    for (const auto dimensions : {std::array<UINT, 2>{96, 54}, std::array<UINT, 2>{641, 359}}) {
      const UINT width = dimensions[0], height = dimensions[1];
      for (unsigned scenario = 0; scenario != 6; ++scenario) {
        sunshine_depth::sample_request request;
        request.source_width = width; request.source_height = height;
        request.x = 8; request.y = 5;
        request.width = scenario == 5 ? 1 : width - 32;
        request.height = scenario == 5 ? 1 : height - 18;
        std::vector<float> pixels(size_t(width) * height, .25f);
        pixels[0] = std::numeric_limits<float>::infinity(); // Outside the active crop.
        const auto first = size_t(request.y) * width + request.x;
        const auto last = size_t(request.y) * width + request.x + request.width - 1;
        pixels[first] = 0.f;
        if (scenario != 5) pixels[last] = 1.f;
        if (scenario == 1) { pixels[first] = -4.f; pixels[last] = 9.f; } // Packed raw domain, not [0,1].
        if (scenario == 2) pixels[first] = std::numeric_limits<float>::quiet_NaN();
        if (scenario == 3) pixels[first] = std::numeric_limits<float>::infinity();
        if (scenario == 4) pixels[first] = -std::numeric_limits<float>::infinity();
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = width; desc.Height = height;
        desc.ArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Format = DXGI_FORMAT_R32_FLOAT;
        desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial{pixels.data(), width * UINT(sizeof(float)), 0};
        com_ptr<ID3D11Texture2D> source;
        checked(device->CreateTexture2D(&desc, &initial, source.put()), "Create exact-range test source");
        request.source = {reinterpret_cast<std::uint64_t>(source.get())};
        for (bool collect_range : {false, true}) {
          request.collect_range = collect_range;
          auto sample = sunshine_depth::take_sample(spare, pipeline);
          sample->native_queue = reinterpret_cast<std::uint64_t>(context);
          require(sunshine_depth::prepare11(*sample, request), "Prepare exact-range D3D11 sample");
          context->ExecuteCommandList(sample->commands11.get(), TRUE);
          wait_sample(*sample);
          sunshine_depth::read_sample(*sample);
          const auto &result = sample->result;
          const auto layout = sunshine_depth_statistics::tile_grid(request.width, request.height);
          require(result.valid && result.width == layout.x && result.height == layout.y &&
            result.values.size() == size_t(layout.x) * layout.y, "Exact range altered point-grid readiness");
          for (UINT y = 0; y < result.height; ++y) for (UINT x = 0; x < result.width; ++x) {
            const UINT source_x = request.x + point_coordinate(x, request.width, result.width);
            const UINT source_y = request.y + point_coordinate(y, request.height, result.height);
            const float expected = pixels[size_t(source_y) * width + source_x];
            require(result.values[y * result.width + x] == expected, "Exact-range mode changed selector point samples");
          }
          const bool expected_valid = collect_range && (scenario < 2 || scenario == 5);
          require(result.range_valid == expected_valid, "Exact range skipped a nonfinite texel or included invalid padding");
          if (expected_valid) require(result.range_min == (scenario == 1 ? -4.f : 0.f) &&
            result.range_max == (scenario == 1 ? 9.f : scenario == 5 ? 0.f : 1.f),
            "Exact range trimmed sparse extrema/endpoints or failed a one-texel crop");
          sample->clear_capture();
          spare = std::move(sample);
        }
      }
    }
    std::puts("PASS exact depth range: sparse endpoints, arbitrary finite raw domain, NaN/Inf rejection, crop/padding, uneven/tiny extents and unchanged optional point grid");
  }

  void run_full_moments(ID3D11Device *device, ID3D11DeviceContext *context, ID3DBlob *shader) {
    using sunshine_depth_statistics::tile_grid;
    using sunshine_depth_statistics::tile_bounds;
    for (const auto shape : {std::array<UINT, 4>{96, 54, 32, 18}, {192, 108, 32, 18},
        {64, 48, 28, 21}, {128, 96, 28, 21}, {36, 64, 18, 32}, {72, 128, 18, 32},
        {1, 1, 1, 1}, {3, 2, 3, 2}, {1, 1024, 1, 576}, {1024, 1, 576, 1}}) {
      const auto layout = tile_grid(shape[0], shape[1]);
      require(layout.x == shape[2] && layout.y == shape[3], "Shared tile layout changed with resolution or lost aspect/tiny-crop bounds");
      for (UINT axis = 0; axis < 2; ++axis) {
        const UINT extent = shape[axis], tiles = axis ? layout.y : layout.x;
        UINT previous = 0;
        for (UINT i = 0; i < tiles; ++i) {
          const auto interval = tile_bounds(i, extent, tiles);
          require(interval.first == previous && interval.last > interval.first &&
            interval.center() >= interval.first && interval.center() < interval.last,
            "Shared integer tile partition has a gap, overlap, empty tile or outside point");
          previous = interval.last;
        }
        require(previous == extent, "Shared tiles failed to cover an entire crop axis");
      }
    }
    for (const auto shape : {std::array<UINT, 2>{1, UINT32_MAX}, {UINT32_MAX, 1}, {UINT32_MAX, UINT32_MAX}}) {
      const auto layout = tile_grid(shape[0], shape[1]);
      require(layout.x && layout.y && std::uint64_t(layout.x) * layout.y <= sunshine_depth_statistics::maximum_tiles,
        "Extreme aspect escaped the bounded tile record capacity");
    }
    require(tile_grid(0, 10).x == 0 && tile_grid(10, 0).y == 0, "Empty crop fabricated a tile layout");

    enum class pattern { gradient, sparse, zero, normal, normal_far, packed, negative, nan, infinity, overflow, maximum };
    struct fixture { const char *name; UINT width, height; pattern content; float inverseB = 1.f; };
    const fixture fixtures[] {
      {"sparse between selector points", 96, 54, pattern::sparse},
      {"same-aspect larger sparse", 192, 108, pattern::sparse},
      {"four by three", 64, 48, pattern::gradient},
      {"four by three larger", 128, 96, pattern::gradient},
      {"portrait", 36, 64, pattern::gradient},
      {"portrait larger", 72, 128, pattern::gradient},
      {"uneven cropped partitions", 609, 341, pattern::gradient},
      {"4K full-pixel reduction", 3840, 2160, pattern::gradient},
      {"ultrawide near far endpoint", 3440, 1440, pattern::normal_far},
      {"tiny positive", 1, 1, pattern::gradient},
      {"tiny zero", 1, 1, pattern::zero},
      {"tiny nondivisible", 3, 2, pattern::gradient},
      {"extreme portrait", 1, 1024, pattern::gradient},
      {"extreme landscape", 1024, 1, pattern::gradient},
      {"all zero", 96, 54, pattern::zero},
      {"normal depth", 96, 54, pattern::normal},
      {"normal depth near far endpoint", 96, 54, pattern::normal_far},
      {"packed affine basis", 96, 54, pattern::packed},
      {"small depth units", 96, 54, pattern::gradient, 1e-30f},
      {"large depth units", 96, 54, pattern::gradient, 1e30f},
      {"largest finite decoded depth", 96, 54, pattern::maximum},
      {"negative decoded depth", 96, 54, pattern::negative},
      {"nonfinite raw NaN", 96, 54, pattern::nan},
      {"nonfinite raw infinity", 96, 54, pattern::infinity},
      {"overflowing decoded depth", 96, 54, pattern::overflow},
      {"invalid zero decode coefficient", 96, 54, pattern::gradient, 0.f},
      {"unrepresentable subnormal coefficient", 96, 54, pattern::gradient, std::numeric_limits<float>::denorm_min()},
      {"recovery after invalid and changed aspect", 64, 48, pattern::gradient}
    };
    pipeline_cache_t cache;
    int identity{};
    auto *owner = reinterpret_cast<api::effect_runtime *>(&identity);
    auto pipeline = cache.acquire(api::device_api::d3d11, reinterpret_cast<std::uint64_t>(device), owner, shader);
    std::unique_ptr<sunshine_depth::sample_t> spare;
    const ID3D11Texture2D *scratch = nullptr;
    for (const auto &fixture : fixtures) {
      try {
        sunshine_depth::sample_request request;
        request.source_width = fixture.width + 13; request.source_height = fixture.height + 9;
        request.x = 5; request.y = 3; request.width = fixture.width; request.height = fixture.height;
        request.moments_inverseB = fixture.inverseB;
        if (fixture.content == pattern::normal || fixture.content == pattern::normal_far) {
          request.moments_A = 1.f; request.moments_inverseB = -1.f;
        } else if (fixture.content == pattern::packed) {
          request.moments_A = -2.f; request.moments_inverseB = .5f;
        } else if (fixture.content == pattern::overflow) request.moments_inverseB = std::numeric_limits<float>::max();
        // Invalid allocation padding must not enter extrema or either moment.
        std::vector<float> pixels(size_t(request.source_width) * request.source_height,
          std::numeric_limits<float>::quiet_NaN());
        double expected_sum = 0., expected_squares = 0.;
        bool valid_raw = true, valid_moments = std::isnormal(request.moments_inverseB);
        float lower = std::numeric_limits<float>::max(), upper = -std::numeric_limits<float>::max();
        for (UINT y = 0; y < fixture.height; ++y) for (UINT x = 0; x < fixture.width; ++x) {
          float raw = float(1 + (x * 17 + y * 11) % 31) / 32.f;
          switch (fixture.content) {
          case pattern::sparse: raw = y == 0 && x == 0 ? .75f : y == 0 && x == 2 ? .5f : 0.f; break;
          case pattern::zero: raw = 0.f; break;
          case pattern::normal: raw = 1.f - raw; break;
          case pattern::normal_far: raw = 1.f - std::ldexp(float((x + 3 * y) % 4), -24); break;
          case pattern::packed: raw = -2.f + 2.f * raw; break;
          case pattern::negative: if (x == 0 && y == 0) raw = -.25f; break;
          case pattern::nan: if (x == 0 && y == 0) raw = std::numeric_limits<float>::quiet_NaN(); break;
          case pattern::infinity: if (x == 0 && y == 0) raw = std::numeric_limits<float>::infinity(); break;
          case pattern::overflow: raw = 2.f; break;
          case pattern::maximum: raw = std::numeric_limits<float>::max(); break;
          default: break;
          }
          pixels[size_t(request.y + y) * request.source_width + request.x + x] = raw;
          valid_raw &= std::isfinite(raw);
          lower = std::min(lower, raw); upper = std::max(upper, raw);
          // Match the actual FP32 decode, then independently sum every pixel in
          // double. No affine expansion of raw squared sums is used as oracle.
          const float delta = raw - request.moments_A;
          const float q = delta * request.moments_inverseB;
          valid_moments &= std::isfinite(q) && q >= 0.f;
          expected_sum += q;
          expected_squares += double(q) * q;
        }
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = request.source_width; desc.Height = request.source_height;
        desc.ArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Format = DXGI_FORMAT_R32_FLOAT;
        desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial{pixels.data(), request.source_width * UINT(sizeof(float)), 0};
        com_ptr<ID3D11Texture2D> source;
        checked(device->CreateTexture2D(&desc, &initial, source.put()), "Create full-moment test source");
        request.source = {reinterpret_cast<std::uint64_t>(source.get())};
        const auto layout = tile_grid(request.width, request.height);
        std::vector<float> selector_only;
        // Selector-only, old exact-range request, and moments-only request all
        // publish bit-identical selector points from the one common tile grid.
        for (unsigned mode = 0; mode < 3; ++mode) {
          request.collect_range = mode == 1; request.collect_moments = mode == 2;
          const auto submitted = request;
          auto sample = sunshine_depth::take_sample(spare, pipeline);
          sample->native_queue = reinterpret_cast<std::uint64_t>(context);
          require(sunshine_depth::prepare11(*sample, request), "Prepare full-moment D3D11 sample");
          if (!scratch) scratch = sample->output11.get();
          require(sample->output11.get() == scratch, "Changing aspect or moment mode reallocated completed scratch");
          request.moments_A = 123.f; request.moments_inverseB = 456.f;
          request.width = request.height = 1; // Pending capture must own its basis/layout.
          context->ExecuteCommandList(sample->commands11.get(), TRUE);
          wait_sample(*sample);
          sunshine_depth::read_sample(*sample);
          request = submitted;
          const auto &result = sample->result;
          require(result.valid && result.width == layout.x && result.height == layout.y &&
            result.values.size() == size_t(layout.x) * layout.y,
            "Full moments corrupted point-grid readiness or frozen layout");
          for (UINT y = 0; y < result.height; ++y) for (UINT x = 0; x < result.width; ++x) {
            const UINT px = submitted.x + point_coordinate(x, submitted.width, result.width);
            const UINT py = submitted.y + point_coordinate(y, submitted.height, result.height);
            const float expected = pixels[size_t(py) * submitted.source_width + px];
            const float actual = result.values[size_t(y) * result.width + x];
            require(actual == expected || (std::isnan(actual) && std::isnan(expected)),
              "Shared selector points differ from independent integer-partition center oracle");
          }
          if (!mode) selector_only = result.values;
          else require(std::memcmp(selector_only.data(), result.values.data(), result.values.size() * sizeof(float)) == 0,
            "Adding a range or moments scan changed raw selector point bits");
          if (fixture.content == pattern::sparse) require(std::all_of(result.values.begin(), result.values.end(),
            [](float value) { return value == 0.f; }), "Sparse full-image fixture accidentally hit selector points");
          require(result.range_valid == (mode != 0 && valid_raw), "Moment decode validity leaked into raw range validity");
          if (result.range_valid) require(result.range_min == lower && result.range_max == upper,
            "Full-image moments changed exact extrema or included padding");
          const auto &moments = result.moments;
          require(moments.supplied == (mode == 2) && moments.valid == (mode == 2 && valid_moments),
            "Invalid moments were accepted, or valid moments failed independently of raw points");
          if (mode == 2) {
            require(moments.A == submitted.moments_A && moments.inverseB == submitted.moments_inverseB &&
              moments.count == std::uint64_t(submitted.width) * submitted.height &&
              moments.tiles_x == layout.x && moments.tiles_y == layout.y,
              "Moment basis, count or layout was not frozen with the full active crop");
            if (valid_moments) {
              const auto close = [](double actual, double expected) {
                return expected == 0. ? actual == 0. : std::abs(actual - expected) <= std::abs(expected) * 3e-5;
              };
              require(close(moments.sum, expected_sum) && close(moments.sum_squares, expected_squares),
                "Stable full-image moments differ from independent double accumulation");
            }
          } else require(moments.count == 0 && moments.tiles_x == 0 && moments.tiles_y == 0,
            "A raw-only request retained a previous capture's moment facts");
          if (!moments.valid) require(moments.sum == 0. && moments.sum_squares == 0.,
            "Invalid or unrequested moments retained numeric data from an older capture");
          sample->clear_capture();
          spare = std::move(sample);
        }
      } catch (const std::exception &error) {
        throw std::runtime_error(std::string(fixture.name) + ": " + error.what());
      }
    }
    std::puts("PASS full-image decoded moments: sparse geometry, shared aspect grid, resolution/crop reuse, zero counts, frozen basis, normal/reverse depth, cancellation, extreme units and independent invalidation");
  }
#endif

  std::vector<float> read_grid(sunshine_depth::sample_t &sample) {
#ifndef SUNSHINE_SAMPLER_BASELINE
    sunshine_depth::read_sample(sample);
    require(sample.result.valid, "Production completed-sample reader failed");
    return sample.result.values;
#else
    std::vector<float> values(32 * 18);
    const unsigned char *bytes = nullptr;
    UINT pitch = 0;
    if (sample.readback12.get()) {
      const D3D12_RANGE range {0, SIZE_T(sample.bytes12)};
      void *mapped = nullptr;
      checked(sample.readback12->Map(0, &range, &mapped), "Map completed repeated D3D12 sample");
      bytes = static_cast<const unsigned char *>(mapped) + sample.footprint12.Offset;
      pitch = sample.footprint12.Footprint.RowPitch;
    } else {
      D3D11_MAPPED_SUBRESOURCE mapped {};
      checked(sample.context11->Map(sample.readback11.get(), 0, D3D11_MAP_READ,
        D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped), "Map completed repeated D3D11 sample");
      bytes = static_cast<const unsigned char *>(mapped.pData);
      pitch = mapped.RowPitch;
    }
    for (UINT y = 0; y < 18; ++y)
      std::memcpy(values.data() + y * 32, bytes + y * pitch, 32 * sizeof(float));
    if (sample.readback12.get()) {
      const D3D12_RANGE written {0, 0};
      sample.readback12->Unmap(0, &written);
    } else sample.context11->Unmap(sample.readback11.get(), 0);
    return values;
#endif
  }

  // Attach a real COM lifetime witness to the source. It proves that completed
  // command lists / private-context bindings do not keep retired sources alive.
  class source_lifetime final : public IUnknown {
  public:
    explicit source_lifetime(bool &destroyed) : destroyed_(destroyed) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id, void **out) override {
      if (!out) return E_POINTER;
      *out = nullptr;
      if (id != IID_IUnknown) return E_NOINTERFACE;
      *out = static_cast<IUnknown *>(this); AddRef(); return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++references_; }
    ULONG STDMETHODCALLTYPE Release() override {
      const ULONG remaining = --references_;
      if (!remaining) delete this;
      return remaining;
    }
  private:
    ~source_lifetime() { destroyed_ = true; }
    std::atomic<ULONG> references_ {1};
    bool &destroyed_;
  };
  const GUID lifetime_guid {0x31196c3f, 0x997e, 0x483f, {0xbd, 0x6e, 0xd2, 0x82, 0x09, 0x7b, 0x49, 0x31}};

  std::vector<unsigned char> source_pixels(const native_source &source) {
    const UINT bytes_per_pixel = source.format == DXGI_FORMAT_R16_UNORM ? 2 : 4;
    std::vector<unsigned char> bytes(size_t(source.width) * source.height * bytes_per_pixel);
    for (UINT y = 0; y < source.height; ++y) for (UINT x = 0; x < source.width; ++x) {
      auto *pixel = bytes.data() + (size_t(y) * source.width + x) * bytes_per_pixel;
      const float value = source.expected(x, y);
      if (bytes_per_pixel == 2) {
        const auto encoded = std::uint16_t(std::lround(value * 65535.f));
        std::memcpy(pixel, &encoded, sizeof(encoded));
      } else std::memcpy(pixel, &value, sizeof(value));
    }
    return bytes;
  }

  struct scratch_measurement {
    double prepare_us = 0, submit_us = 0, readback_us = 0;
    std::vector<double> prepare_samples_us;
    unsigned created_bundles = 0;
    UINT64 readback_bytes = 0;
  };

  scratch_measurement exercise_samples(api::device_api kind, std::uint64_t device, std::uint64_t queue,
      ID3DBlob *shader, std::array<native_source, 2> &sources, unsigned count, bool reuse) {
    pipeline_cache_t cache;
    int identity {};
    auto *owner = reinterpret_cast<api::effect_runtime *>(&identity);
    auto pipeline = cache.acquire(kind, device, owner, shader);
    require(bool(pipeline), "Create program for repeated sampler workload");
    scratch_measurement measured;
    std::unique_ptr<sunshine_depth::sample_t> spare;
    std::vector<sunshine_depth::sample_result> completed;
    std::array<const void *, 6> first_objects {};
    UINT64 previous_fence = 0;
    for (unsigned i = 0; i < count; ++i) {
      auto &source = sources[i % sources.size()];
      sunshine_depth::sample_request request;
      request.token = 700 + i;
      request.capture_id = 1000 + i;
      request.source_lifetime = 2000 + i;
      request.capture_scope = {3, 4 + i, device, 6};
      request.source = {source.native()};
      request.source_width = source.width; request.source_height = source.height;
      const bool cropped = (i / 2) % 2 != 0;
      request.x = cropped ? 7 : 0; request.y = cropped ? 3 : 0;
      request.width = cropped ? source.width - 19 : source.width;
      request.height = cropped ? source.height - 11 : source.height;
#ifdef SUNSHINE_SAMPLER_BASELINE
      // The frozen implementation destroys completed bundles. Both labels run
      // the identical capture workload, but neither pretends old code reuses.
      auto sample = std::make_unique<sunshine_depth::sample_t>();
      sample->pipeline = pipeline;
#else
      auto sample = sunshine_depth::take_sample(spare, pipeline);
#endif
      sample->runtime = owner;
      sample->native_queue = queue;
      const bool fresh = !sample->output11.get() && !sample->output12.get();
      const auto prepare_start = std::chrono::steady_clock::now();
      require(kind == api::device_api::d3d12 ? sunshine_depth::prepare12(*sample, request) :
        sunshine_depth::prepare11(*sample, request), "Prepare repeated native sample");
      const auto prepare_end = std::chrono::steady_clock::now();
      const double prepare_us = std::chrono::duration<double, std::micro>(prepare_end - prepare_start).count();
      measured.prepare_us += prepare_us;
      measured.prepare_samples_us.push_back(prepare_us);
      measured.created_bundles += fresh;
      measured.readback_bytes = sample->readback12.get() ? sample->bytes12 : output_logical_bytes;
      // Mutable submission metadata must not alias a pending result's facts.
      const auto submitted = request;
      request.token = request.capture_id = request.source_lifetime = 999999;
      request.capture_scope = {}; request.x = request.y = request.width = request.height = 0;
      if (reuse && i + 1 == count) {
        // Retire the owner's source before submitting this last capture. The
        // sampler must keep it alive through GPU completion and readback.
        reset_com(source.texture11); reset_com(source.texture12);
        require(!*source.destroyed, "Prepared sample failed to retain its source until GPU completion");
      }
#ifndef SUNSHINE_SAMPLER_BASELINE
      const std::array<const void *, 6> objects = kind == api::device_api::d3d12 ?
        std::array<const void *, 6> {sample->output12.get(), sample->readback12.get(), sample->descriptors12.get(),
          sample->allocator12.get(), sample->commands12.get(), sample->fence12.get()} :
        std::array<const void *, 6> {sample->output11.get(), sample->readback11.get(), sample->uav11.get(),
          sample->constants11.get(), sample->query11.get(), sample->deferred11.get()};
      if (reuse && i) require(objects == first_objects, "Same-device capture recreated fixed scratch objects");
      if (!i) first_objects = objects;
      if (sample->fence12.get() && reuse) {
        require(sample->fence_value12 > previous_fence, "Reused D3D12 fence did not advance");
        require(sample->fence12->GetCompletedValue() < sample->fence_value12,
          "Old completed D3D12 fence value could prematurely admit new readback");
        previous_fence = sample->fence_value12;
      }
#endif
      const auto submit_start = std::chrono::steady_clock::now();
      if (sample->queue12.get()) {
        ID3D12CommandList *list = sample->commands12.get();
        sample->queue12->ExecuteCommandLists(1, &list);
        checked(sample->queue12->Signal(sample->fence12.get(), fence_target(*sample)), "Signal repeated sample");
      } else sample->context11->ExecuteCommandList(sample->commands11.get(), TRUE);
      const auto submit_end = std::chrono::steady_clock::now();
      measured.submit_us += std::chrono::duration<double, std::micro>(submit_end - submit_start).count();
      wait_sample(*sample);
      const auto read_start = std::chrono::steady_clock::now();
      sample->result.values = read_grid(*sample);
      sample->result.valid = true;
      const auto read_end = std::chrono::steady_clock::now();
      measured.readback_us += std::chrono::duration<double, std::micro>(read_end - read_start).count();
      const auto &result = sample->result;
      require(result.token == submitted.token && result.capture_id == submitted.capture_id &&
        result.source_lifetime == submitted.source_lifetime && result.capture_scope == submitted.capture_scope &&
        result.source.handle == submitted.source.handle && result.source_format == unsigned(source.format) &&
        result.source_width == source.width && result.source_height == source.height &&
        result.viewport_x == submitted.x && result.viewport_y == submitted.y &&
        result.viewport_width == submitted.width && result.viewport_height == submitted.height,
        "Repeated sample reused old provenance, source, format or crop");
      float max_error = 0;
      for (UINT y = 0; y < result.height; ++y) for (UINT x = 0; x < result.width; ++x) {
        const float actual = result.values[y * result.width + x];
        const float expected = source.expected(submitted.x + point_coordinate(x, submitted.width, result.width),
          submitted.y + point_coordinate(y, submitted.height, result.height));
        require(std::isfinite(actual), "Repeated sample returned nonfinite depth");
        max_error = std::max(max_error, std::abs(actual - expected));
      }
      require(max_error < .00002f, "Repeated sample read stale source/format/crop contents");
      completed.push_back(std::move(sample->result));
#ifndef SUNSHINE_SAMPLER_BASELINE
      if (reuse) {
        sample->clear_capture();
        require(!sample->source11.get() && !sample->source12.get() && !sample->srv11.get() &&
          !sample->commands11.get() && sample->result.values.empty(), "Idle scratch retained capture objects/facts");
        spare = std::move(sample);
      }
#endif
    }
    for (unsigned i = 0; i < count; ++i)
      require(completed[i].valid && completed[i].capture_id == 1000 + i &&
        completed[i].values.size() == size_t(completed[i].width) * completed[i].height,
        "Later scratch reuse mutated an earlier immutable result");
    if (reuse) {
      // Keep the completed spare alive while dropping the remaining owner refs.
      // A stale private-context binding or recorded list would fail this check.
      for (auto &source : sources) {
        reset_com(source.texture11); reset_com(source.texture12);
        require(*source.destroyed, "Idle scratch retained a retired source through old bindings");
      }
    }
#ifndef SUNSHINE_SAMPLER_BASELINE
    require(measured.created_bundles == (reuse ? 1u : count), "Unexpected fixed scratch allocation count");
    if (spare) {
      // Retiring the immutable program also prevents reuse of its scratch on a
      // replacement runtime, even if the physical device happens to be shared.
      std::weak_ptr<sunshine_depth::device_pipeline_t> previous = pipeline;
      cache.retire(owner);
      pipeline.reset();
      auto replacement = cache.acquire(kind, device, owner, shader);
      auto next = sunshine_depth::take_sample(spare, replacement);
      require(previous.expired() && !next->output11.get() && !next->output12.get(),
        "Retired program reused completed scratch or retained the old native owner");
    }
#else
    require(measured.created_bundles == count, "Frozen sampler unexpectedly retained scratch");
#endif
    const char *implementation =
#ifdef SUNSHINE_SAMPLER_BASELINE
      "frozen-before";
#else
      "current";
#endif
    auto ordered_prepare_us = measured.prepare_samples_us;
    std::sort(ordered_prepare_us.begin(), ordered_prepare_us.end());
    const double prepare_p95_us = ordered_prepare_us[(count * 95 + 99) / 100 - 1];
    std::printf("MEASURE scratch api=%s implementation=%s workload=%s samples=%u prepare_mean_us=%.6f prepare_p95_us=%.6f prepare_max_us=%.6f submit_mean_us=%.6f readback_mean_us=%.6f created_bundles=%u scratch_resource_creations=%u output_logical_bytes=%u readback_bytes=%llu; CPU wall time, waits/source creation excluded, bytes exclude driver allocation overhead\n",
      kind == api::device_api::d3d12 ? "D3D12" : "D3D11", implementation, reuse ? "persistent" : "fresh", count,
      measured.prepare_us / count, prepare_p95_us, ordered_prepare_us.back(), measured.submit_us / count,
      measured.readback_us / count, measured.created_bundles,
      measured.created_bundles * (kind == api::device_api::d3d12 ? 2 : 3), output_logical_bytes,
      static_cast<unsigned long long>(measured.readback_bytes));
    return measured;
  }

  void run_repeated_samples(ID3D11Device *device11, ID3D11DeviceContext *context11, ID3DBlob *shader, unsigned count) {
    std::array<bool, 2> destroyed {};
    std::array<native_source, 2> sources;
    for (unsigned i = 0; i < sources.size(); ++i) {
      auto &source = sources[i];
      source.width = 96 + 32 * i; source.height = 54 + 18 * i;
      source.format = i ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R32_FLOAT;
      source.index = i;
      source.destroyed = &destroyed[i];
      const auto pixels = source_pixels(source);
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = source.width; desc.Height = source.height;
      desc.ArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Format = source.format; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      const D3D11_SUBRESOURCE_DATA initial {pixels.data(), source.width * (i ? 2u : 4u), 0};
      checked(device11->CreateTexture2D(&desc, &initial, source.texture11.put()), "Create repeated D3D11 source");
      auto *witness = new source_lifetime(destroyed[i]);
      checked(source.texture11->SetPrivateDataInterface(lifetime_guid, witness), "Track repeated D3D11 source lifetime");
      witness->Release();
    }
    for (bool reuse : {false, true}) exercise_samples(api::device_api::d3d11,
      reinterpret_cast<std::uint64_t>(device11), reinterpret_cast<std::uint64_t>(context11), shader, sources, count, reuse);
    for (auto &source : sources) reset_com(source.texture11);
    require(destroyed[0] && destroyed[1], "Completed D3D11 scratch still retains old source/context bindings");

    com_ptr<ID3D12Device> device12;
    checked(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device12.put())), "Create repeated D3D12 device");
    com_ptr<ID3D12CommandQueue> queue12;
    const D3D12_COMMAND_QUEUE_DESC queue_desc {D3D12_COMMAND_LIST_TYPE_DIRECT};
    checked(device12->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(queue12.put())), "Create repeated D3D12 queue");
    com_ptr<ID3D12CommandAllocator> upload_allocator;
    com_ptr<ID3D12GraphicsCommandList> upload_commands;
    com_ptr<ID3D12Fence> upload_fence;
    checked(device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(upload_allocator.put())), "Create repeated upload allocator");
    checked(device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, upload_allocator.get(), nullptr,
      IID_PPV_ARGS(upload_commands.put())), "Create repeated upload list");
    checked(device12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(upload_fence.put())), "Create upload completion fence");
    std::array<com_ptr<ID3D12Resource>, 2> uploads;
    destroyed = {};
    for (unsigned i = 0; i < sources.size(); ++i) {
      auto &source = sources[i];
      D3D12_RESOURCE_DESC desc {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = source.width; desc.Height = source.height;
      desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Format = source.format;
      D3D12_HEAP_PROPERTIES heap {};
      heap.Type = D3D12_HEAP_TYPE_DEFAULT;
      heap.CreationNodeMask = heap.VisibleNodeMask = 1;
      checked(device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(source.texture12.put())), "Create repeated D3D12 source");
      auto *witness = new source_lifetime(destroyed[i]);
      checked(source.texture12->SetPrivateDataInterface(lifetime_guid, witness), "Track repeated D3D12 source lifetime");
      witness->Release();
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
      UINT64 bytes = 0;
      device12->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
      D3D12_RESOURCE_DESC upload_desc {};
      upload_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      upload_desc.Width = bytes;
      upload_desc.Height = upload_desc.DepthOrArraySize = upload_desc.MipLevels = upload_desc.SampleDesc.Count = 1;
      upload_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      heap.Type = D3D12_HEAP_TYPE_UPLOAD;
      checked(device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &upload_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr, IID_PPV_ARGS(uploads[i].put())), "Create repeated D3D12 upload");
      void *mapped = nullptr;
      const D3D12_RANGE empty {0, 0};
      checked(uploads[i]->Map(0, &empty, &mapped), "Map repeated source upload");
      const auto pixels = source_pixels(source);
      const auto pitch = source.width * (i ? 2u : 4u);
      for (UINT y = 0; y < source.height; ++y)
        std::memcpy(static_cast<unsigned char *>(mapped) + footprint.Offset + y * footprint.Footprint.RowPitch,
          pixels.data() + size_t(y) * pitch, pitch);
      uploads[i]->Unmap(0, nullptr);
      D3D12_TEXTURE_COPY_LOCATION from {}, to {};
      from.pResource = uploads[i].get(); from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
      to.pResource = source.texture12.get(); to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      upload_commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    checked(upload_commands->Close(), "Close repeated source upload");
    ID3D12CommandList *upload_list = upload_commands.get();
    queue12->ExecuteCommandLists(1, &upload_list);
    checked(queue12->Signal(upload_fence.get(), 1), "Signal repeated source upload");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (upload_fence->GetCompletedValue() < 1 && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(upload_fence->GetCompletedValue() != UINT64_MAX && upload_fence->GetCompletedValue() >= 1, "Source upload did not finish");
    reset_com(upload_commands); reset_com(upload_allocator);
    for (bool reuse : {false, true}) exercise_samples(api::device_api::d3d12,
      reinterpret_cast<std::uint64_t>(device12.get()), reinterpret_cast<std::uint64_t>(queue12.get()), shader, sources, count, reuse);
    for (auto &source : sources) reset_com(source.texture12);
    require(destroyed[0] && destroyed[1], "Completed D3D12 scratch still retains old source bindings");
    std::puts("PASS repeated native samples: changing sources, dimensions, formats and crops; immutable capture facts; completion reuse and source release");
  }
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    require(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0), "Usage: test_depth_content_sampler_d3d11 [--benchmark]");
    const unsigned repeated_count = argc == 2 ? 64 : 8;
    com_ptr<ID3D11Device> device;
    com_ptr<ID3D11DeviceContext> context;
    checked(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
      D3D11_SDK_VERSION, device.put(), nullptr, context.put()), "Create hardware D3D11 device");
    com_ptr<ID3DBlob> shader, errors;
    const auto compiled = D3DCompile(sunshine_depth::shader_source, sizeof(sunshine_depth::shader_source) - 1, nullptr,
      nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, shader.put(), errors.put());
    if (FAILED(compiled) && errors.get()) std::fprintf(stderr, "%s\n", static_cast<const char *>(errors->GetBufferPointer()));
    checked(compiled, "Compile actual sampler shader");
    verify_device_cache(device.get(), shader.get());
#ifndef SUNSHINE_SAMPLER_BASELINE
    run_exact_range(device.get(), context.get(), shader.get());
    run_full_moments(device.get(), context.get(), shader.get());
#endif
    pipeline_cache_t cache;
    int identity{};
    auto *owner = reinterpret_cast<api::effect_runtime *>(&identity);
    const auto first = cache.acquire(api::device_api::d3d11, reinterpret_cast<std::uint64_t>(device.get()), owner, shader.get());
    for (bool viewport : {false, true}) {
      run_format(device.get(), context.get(), shader.get(), cache, owner, DXGI_FORMAT_R32_TYPELESS, 4, viewport);
      run_format(device.get(), context.get(), shader.get(), cache, owner, DXGI_FORMAT_R32G8X24_TYPELESS, 8, viewport);
      run_format(device.get(), context.get(), shader.get(), cache, owner, DXGI_FORMAT_R24G8_TYPELESS, 4, viewport);
      run_format(device.get(), context.get(), shader.get(), cache, owner, DXGI_FORMAT_R16_TYPELESS, 2, viewport);
      require(cache.acquire(api::device_api::d3d11, reinterpret_cast<std::uint64_t>(device.get()), owner, shader.get()) == first,
        "Format or crop changes recreated immutable native shader");
    }
    require(run_d3d12(shader.get()).expired(), "Finished D3D12 sample retained retired program/device");
    run_repeated_samples(device.get(), context.get(), shader.get(), repeated_count);
    std::puts("PASS production D3D11 depth sampler: all four depth format families, full/padded viewport, independent value oracle and exact CS state preservation");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
