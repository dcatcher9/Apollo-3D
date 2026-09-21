// SPDX-License-Identifier: GPL-3.0-only
#include "depth_content_sampler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <d3d11.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

static_assert(RESHADE_API_VERSION == 20, "Use the pinned ReShade 6.8 SDK");

namespace sunshine_depth {
namespace {
  namespace api = reshade::api;
  constexpr UINT storage_width = 32;
  constexpr UINT tile_capacity = sunshine_depth_statistics::maximum_tiles;
  static_assert(tile_capacity == 768 && tile_capacity % storage_width == 0,
    "Keep shader record addressing synchronized with the fixed scratch layout");

  template<class T> class com_ptr {
  public:
    com_ptr() = default;
    ~com_ptr() { if (value_) value_->Release(); }
    com_ptr(const com_ptr &) = delete;
    com_ptr &operator=(const com_ptr &) = delete;
    T *get() const { return value_; }
    T *operator->() const { return value_; }
    void reset() { if (value_) value_->Release(); value_ = nullptr; }
    T **put() { reset(); return &value_; }
    void retain(T *value) { if (value) value->AddRef(); reset(); value_ = value; }
  private:
    T *value_ = nullptr;
  };

  void warning(const char *message) {
    reshade::log::message(reshade::log::level::warning, message);
  }

  // Integer texel loads avoid filtering across geometry and padded viewport edges.
  // Selector points and extrema remain raw. Optional moments decode only the
  // immutable capture's depth basis. Moments make a second tile traversal after
  // reducing its minimum, inside this same dispatch and scratch/readback layout.
  constexpr char shader_source[] = R"(
Texture2D<float> source_depth : register(t0);
RWTexture2D<float4> sampled_depth : register(u0);
cbuffer Region : register(b0) {
  uint4 region;
  uint collect_range, collect_moments;
  float moments_A, moments_inverseB;
  uint2 tile_dimensions;
};
groupshared float2 tile_range[64];
groupshared uint tile_invalid[64];
groupshared float3 tile_moments[64];
groupshared uint tile_moments_invalid[64];
bool finite_value(float value) {
  return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}
float3 merge_moments(float3 a, float3 b) {
  float scale = max(a.x, b.x);
  if (scale == 0.0) return 0.0;
  float ra = a.x / scale, rb = b.x / scale;
  return float3(scale, a.y * ra + b.y * rb, a.z * ra * ra + b.z * rb * rb);
}
float3 add_moment(float3 state, float q) {
  if (q == 0.0) return state;
  // A power-of-two scale keeps both the scale and its reciprocal normal,
  // even near FLT_MAX. Squared normalized values stay below sixteen.
  float scale = asfloat(min(max(asuint(q) & 0x7f800000u, 0x00800000u), 0x7e800000u));
  float normalized = q / scale;
  return merge_moments(state, float3(scale, normalized, normalized * normalized));
}
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID,
    uint3 local : SV_GroupThreadID, uint lane : SV_GroupIndex) {
  // One frozen layout defines both selector points and full-image partitions.
  // Scratch coordinates only pack records and never determine the tile grid.
  const uint storage_width = 32;
  if (!collect_range) {
    if (all(id.xy < tile_dimensions)) {
      uint2 first = id.xy * region.zw / tile_dimensions;
      uint2 last = (id.xy + 1) * region.zw / tile_dimensions;
      uint2 pixel = region.xy + first + (last - first) / 2;
      uint tile = id.y * tile_dimensions.x + id.x;
      sampled_depth[uint2(tile % storage_width, tile / storage_width)] =
        float4(source_depth.Load(int3(pixel, 0)), 0, 0, 0);
    }
  } else {
    // Disjoint integer partitions cover the entire active rectangle, including
    // tiny/thin geometry between point samples. Padding is never inspected.
    uint2 first = region.xy + group.xy * region.zw / tile_dimensions;
    uint2 last = region.xy + (group.xy + 1) * region.zw / tile_dimensions;
    float lower = 3.402823466e+38, upper = -3.402823466e+38;
    uint invalid = 0;
    for (uint y = first.y + local.y; y < last.y; y += 8)
      for (uint x = first.x + local.x; x < last.x; x += 8) {
        float value = source_depth.Load(int3(x, y, 0));
        if (!finite_value(value)) invalid = 1;
        else { lower = min(lower, value); upper = max(upper, value); }
      }
    tile_range[lane] = float2(lower, upper);
    tile_invalid[lane] = invalid;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32; stride; stride >>= 1) {
      if (lane < stride) {
        tile_range[lane] = float2(min(tile_range[lane].x, tile_range[lane + stride].x),
          max(tile_range[lane].y, tile_range[lane + stride].y));
        tile_invalid[lane] |= tile_invalid[lane + stride];
      }
      GroupMemoryBarrierWithGroupSync();
    }
    float3 moments = 0.0;
    uint moments_invalid = 0;
    if (collect_moments) {
      // Decode the near/far endpoint with the same FP32 operations as every
      // texel. Subtract before summing so nearly flat q near one retains its
      // tiny contrast; E[q*q] - 2*b*E[q] + b*b would lose it in FP32.
      float raw_origin = moments_inverseB > 0.0 ? tile_range[0].x : tile_range[0].y;
      precise float origin = (raw_origin - moments_A) * moments_inverseB;
      moments_invalid = tile_invalid[0] || !finite_value(origin) || origin < 0.0;
      for (uint y = first.y + local.y; y < last.y; y += 8)
        for (uint x = first.x + local.x; x < last.x; x += 8) {
          float value = source_depth.Load(int3(x, y, 0));
          precise float q = (value - moments_A) * moments_inverseB;
          precise float centered = q - origin;
          if (!finite_value(q) || q < 0.0 || !finite_value(centered) || centered < 0.0)
            moments_invalid = 1;
          else moments = add_moment(moments, centered);
        }
    }
    tile_moments[lane] = moments;
    tile_moments_invalid[lane] = moments_invalid;
    GroupMemoryBarrierWithGroupSync();
    if (collect_moments) {
      for (uint stride = 32; stride; stride >>= 1) {
        if (lane < stride) {
          tile_moments[lane] = merge_moments(tile_moments[lane], tile_moments[lane + stride]);
          tile_moments_invalid[lane] |= tile_moments_invalid[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
      }
    }
    if (!lane) {
      uint tile = group.y * tile_dimensions.x + group.x;
      uint2 output = uint2(tile % storage_width, tile / storage_width);
      uint2 pixel = first + (last - first) / 2;
      float point_value = source_depth.Load(int3(pixel, 0));
      float status = tile_invalid[0] ? 0.0 : any(first == last) ? 2.0 : 1.0;
      sampled_depth[output] = float4(point_value, tile_range[0], status);
      float moments_status = !collect_moments || tile_moments_invalid[0] ? 0.0 : any(first == last) ? 2.0 : 1.0;
      sampled_depth[uint2((tile + 768) % storage_width, (tile + 768) / storage_width)] =
        float4(tile_moments[0], moments_status);
    }
  }
}
)";

  DXGI_FORMAT depth_srv_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
      return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
      return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
      return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_UNORM:
      return DXGI_FORMAT_R16_UNORM;
    default:
      return DXGI_FORMAT_UNKNOWN;
    }
  }

  // Only immutable, device-owned objects are shared. Mutable scratch objects
  // belong to one sample at a time and can be reused only after GPU completion.
  struct device_pipeline_t {
    com_ptr<ID3D11Device> device11;
    com_ptr<ID3D11ComputeShader> shader11;
    com_ptr<ID3D12Device> device12;
    com_ptr<ID3D12RootSignature> root12;
    com_ptr<ID3D12PipelineState> pipeline12;
  };

  class pipeline_cache_t {
  public:
    std::shared_ptr<device_pipeline_t> acquire(api::device_api kind, std::uint64_t native,
        api::effect_runtime *owner, ID3DBlob *shader) {
      if (!native || !owner || !shader) return {};
      if (cached_ && ((kind == api::device_api::d3d11 &&
            reinterpret_cast<std::uint64_t>(cached_->device11.get()) == native) ||
          (kind == api::device_api::d3d12 &&
            reinterpret_cast<std::uint64_t>(cached_->device12.get()) == native))) {
        owner_ = owner;
        return cached_;
      }
      auto next = std::make_shared<device_pipeline_t>();
      if (kind == api::device_api::d3d11) {
        next->device11.retain(reinterpret_cast<ID3D11Device *>(native));
        if (FAILED(next->device11->CreateComputeShader(shader->GetBufferPointer(), shader->GetBufferSize(),
            nullptr, next->shader11.put()))) return {};
      } else if (kind == api::device_api::d3d12) {
        next->device12.retain(reinterpret_cast<ID3D12Device *>(native));
        D3D12_DESCRIPTOR_RANGE ranges[2] {};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = 1;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 1;
        ranges[1].OffsetInDescriptorsFromTableStart = 1;
        D3D12_ROOT_PARAMETER parameters[2] {};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[0].DescriptorTable = {2, ranges};
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[1].Constants.Num32BitValues = 12;
        D3D12_ROOT_SIGNATURE_DESC root_desc {};
        root_desc.NumParameters = 2;
        root_desc.pParameters = parameters;
        com_ptr<ID3DBlob> signature, errors;
        if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, signature.put(), errors.put())) ||
            FAILED(next->device12->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
              IID_PPV_ARGS(next->root12.put())))) return {};
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc {};
        pipeline_desc.pRootSignature = next->root12.get();
        pipeline_desc.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
        if (FAILED(next->device12->CreateComputePipelineState(&pipeline_desc, IID_PPV_ARGS(next->pipeline12.put())))) return {};
      } else return {};
      // The native device is retained, so a reused COM address cannot falsely
      // match this entry. Replacement releases only the cache's ownership;
      // any submitted sample still owns its original program and device.
      cached_ = std::move(next);
      owner_ = owner;
      return cached_;
    }

    void retire(api::effect_runtime *owner) {
      // The most recent runtime borrowing this one-entry cache owns its idle
      // lifetime. Retiring another runtime on the same device must not evict
      // an active borrower's entry. No runtime is ever dereferenced here.
      if (owner_ == owner) {
        cached_.reset();
        owner_ = nullptr;
      }
    }

  private:
    std::shared_ptr<device_pipeline_t> cached_;
    api::effect_runtime *owner_ = nullptr;
  };

  struct sample_t {
    api::effect_runtime *runtime = nullptr; // Identity only; may have been destroyed.
    std::uint64_t swapchain = 0, native_queue = 0;
    sample_result result;
    std::array<UINT, 12> region {};
    bool executed = false, signalled = false, abandoned = false;
    bool retired = false, quarantined = false;

    // Declared before the private objects so it is released after them. A
    // quarantined sample retains this reference along with all GPU resources.
    std::shared_ptr<device_pipeline_t> pipeline;

    com_ptr<ID3D11DeviceContext> context11, deferred11;
    com_ptr<ID3D11Texture2D> source11, output11, readback11;
    com_ptr<ID3D11ShaderResourceView> srv11;
    com_ptr<ID3D11UnorderedAccessView> uav11;
    com_ptr<ID3D11Buffer> constants11;
    com_ptr<ID3D11CommandList> commands11;
    com_ptr<ID3D11Query> query11;

    com_ptr<ID3D12CommandQueue> queue12;
    com_ptr<ID3D12Resource> source12, output12, readback12;
    com_ptr<ID3D12DescriptorHeap> descriptors12;
    com_ptr<ID3D12CommandAllocator> allocator12;
    com_ptr<ID3D12GraphicsCommandList> commands12;
    com_ptr<ID3D12Fence> fence12;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint12 {};
    UINT64 bytes12 = 0, fence_value12 = 0;

    // Called only after completion. Drop source bindings and immutable capture
    // facts before keeping the fixed-size scratch bundle idle. The runtime is
    // retained as an identity so its destruction can release this idle bundle.
    void clear_capture() {
      commands11.reset();
      srv11.reset();
      source11.reset();
      source12.reset();
      context11.reset();
      queue12.reset();
      swapchain = native_queue = 0;
      result = {};
      region = {};
      executed = signalled = abandoned = retired = quarantined = false;
    }
  };

  std::unique_ptr<sample_t> take_sample(std::unique_ptr<sample_t> &spare,
      std::shared_ptr<device_pipeline_t> pipeline) {
    auto sample = std::move(spare);
    // The retained program includes its native device identity. A different
    // device or a retired runtime's replacement program starts a fresh bundle.
    if (!sample || sample->pipeline != pipeline) sample = std::make_unique<sample_t>();
    sample->pipeline = std::move(pipeline);
    return sample;
  }

  bool set_region(sample_t &sample, const sample_request &request, UINT width, UINT height) {
    if (!width || !height || width > 65536 || height > 65536 ||
        (request.source_width && request.source_width != width) ||
        (request.source_height && request.source_height != height)) return false;
    auto &out = sample.result;
    out.token = request.token;
    out.capture_id = request.capture_id;
    out.capture_scope = request.capture_scope;
    out.source_lifetime = request.source_lifetime;
    out.source_width = width;
    out.source_height = height;
    const bool inside = request.width && request.height && request.x < width && request.y < height &&
      request.width <= width - request.x && request.height <= height - request.y;
    out.viewport_x = inside ? request.x : 0;
    out.viewport_y = inside ? request.y : 0;
    out.viewport_width = inside ? request.width : width;
    out.viewport_height = inside ? request.height : height;
    sample.region = {out.viewport_x, out.viewport_y, out.viewport_width, out.viewport_height,
      request.collect_range || request.collect_moments ? 1u : 0u, request.collect_moments ? 1u : 0u};
    std::memcpy(&sample.region[6], &request.moments_A, sizeof(float));
    std::memcpy(&sample.region[7], &request.moments_inverseB, sizeof(float));
    const auto tiles = sunshine_depth_statistics::tile_grid(out.viewport_width, out.viewport_height);
    out.width = tiles.x;
    out.height = tiles.y;
    sample.region[8] = tiles.x;
    sample.region[9] = tiles.y;
    out.moments = {};
    out.moments.supplied = request.collect_moments;
    out.moments.centered_supplied = request.collect_moments;
    out.moments.A = request.moments_A;
    out.moments.inverseB = request.moments_inverseB;
    out.moments.count = request.collect_moments ? std::uint64_t(out.viewport_width) * out.viewport_height : 0;
    out.moments.tiles_x = request.collect_moments ? tiles.x : 0;
    out.moments.tiles_y = request.collect_moments ? tiles.y : 0;
    return true;
  }

  bool prepare11(sample_t &sample, const sample_request &request) {
    if (!sample.pipeline || !sample.pipeline->device11.get() || !sample.pipeline->shader11.get()) return false;
    auto *device = sample.pipeline->device11.get();
    sample.source11.retain(reinterpret_cast<ID3D11Texture2D *>(request.source.handle));
    D3D11_TEXTURE2D_DESC source_desc {};
    sample.source11->GetDesc(&source_desc);
    sample.result.source = {reinterpret_cast<std::uint64_t>(sample.source11.get())};
    sample.result.source_format = static_cast<std::uint32_t>(source_desc.Format);
    const DXGI_FORMAT format = depth_srv_format(source_desc.Format);
    if (format == DXGI_FORMAT_UNKNOWN || source_desc.ArraySize != 1 || source_desc.MipLevels != 1 ||
        source_desc.SampleDesc.Count != 1 || !(source_desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) ||
        !set_region(sample, request, source_desc.Width, source_desc.Height)) return false;

    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc {};
    srv_desc.Format = format;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    if (FAILED(device->CreateShaderResourceView(sample.source11.get(), &srv_desc, sample.srv11.put()))) return false;

    if (!sample.output11.get()) {
      D3D11_TEXTURE2D_DESC output_desc {};
      output_desc.Width = storage_width;
      output_desc.Height = tile_capacity * 2 / storage_width;
      output_desc.ArraySize = output_desc.MipLevels = 1;
      output_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
      output_desc.SampleDesc.Count = 1;
      output_desc.Usage = D3D11_USAGE_DEFAULT;
      output_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      if (FAILED(device->CreateTexture2D(&output_desc, nullptr, sample.output11.put())) ||
          FAILED(device->CreateUnorderedAccessView(sample.output11.get(), nullptr, sample.uav11.put()))) return false;
      output_desc.Usage = D3D11_USAGE_STAGING;
      output_desc.BindFlags = 0;
      output_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      if (FAILED(device->CreateTexture2D(&output_desc, nullptr, sample.readback11.put()))) return false;

      D3D11_BUFFER_DESC constants_desc {};
      constants_desc.ByteWidth = sizeof(sample.region);
      constants_desc.Usage = D3D11_USAGE_DEFAULT;
      constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const D3D11_QUERY_DESC query_desc {D3D11_QUERY_EVENT, 0};
      if (FAILED(device->CreateBuffer(&constants_desc, nullptr, sample.constants11.put())) ||
          FAILED(device->CreateQuery(&query_desc, sample.query11.put())) ||
          FAILED(device->CreateDeferredContext(0, sample.deferred11.put()))) return false;
    }

    // A private deferred context plus RestoreContextState=TRUE preserves every
    // application and ReShade binding, including slots we do not know about.
    auto *deferred = sample.deferred11.get();
    deferred->UpdateSubresource(sample.constants11.get(), 0, nullptr, sample.region.data(), 0, 0);
    ID3D11ShaderResourceView *srv = sample.srv11.get();
    ID3D11UnorderedAccessView *uav = sample.uav11.get();
    ID3D11Buffer *constants = sample.constants11.get();
    deferred->CSSetShader(sample.pipeline->shader11.get(), nullptr, 0);
    deferred->CSSetShaderResources(0, 1, &srv);
    deferred->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
    deferred->CSSetConstantBuffers(0, 1, &constants);
    deferred->Dispatch(sample.region[4] ? sample.region[8] : (sample.region[8] + 7) / 8,
      sample.region[4] ? sample.region[9] : (sample.region[9] + 7) / 8, 1);
    uav = nullptr;
    deferred->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
    deferred->CopyResource(sample.readback11.get(), sample.output11.get());
    deferred->End(sample.query11.get());
    if (FAILED(deferred->FinishCommandList(FALSE, sample.commands11.put()))) return false;
    sample.context11.retain(reinterpret_cast<ID3D11DeviceContext *>(sample.native_queue));
    return sample.context11->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE;
  }

  bool prepare12(sample_t &sample, const sample_request &request) {
    if (!sample.pipeline || !sample.pipeline->device12.get() || !sample.pipeline->root12.get() ||
        !sample.pipeline->pipeline12.get()) return false;
    // These are the states the integration owns for its dedicated backup. Do not
    // guess state for arbitrary application resources or accept a partial mask.
    D3D12_RESOURCE_STATES before;
    if (request.before == api::resource_usage::copy_dest) before = D3D12_RESOURCE_STATE_COPY_DEST;
    else if (request.before == api::resource_usage::shader_resource)
      before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    else return false;
    auto *device = sample.pipeline->device12.get();
    sample.source12.retain(reinterpret_cast<ID3D12Resource *>(request.source.handle));
    const D3D12_RESOURCE_DESC source_desc = sample.source12->GetDesc(); // Native MinGW COM adapter.
    sample.result.source = {reinterpret_cast<std::uint64_t>(sample.source12.get())};
    sample.result.source_format = static_cast<std::uint32_t>(source_desc.Format);
    const DXGI_FORMAT format = depth_srv_format(source_desc.Format);
    if (format == DXGI_FORMAT_UNKNOWN || source_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        source_desc.DepthOrArraySize != 1 || source_desc.MipLevels != 1 || source_desc.SampleDesc.Count != 1 ||
        source_desc.Width > UINT32_MAX || (source_desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) ||
        !set_region(sample, request, UINT(source_desc.Width), source_desc.Height)) return false;

    if (!sample.output12.get()) {
      D3D12_RESOURCE_DESC output_desc {};
      output_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      output_desc.Width = storage_width;
      output_desc.Height = tile_capacity * 2 / storage_width;
      output_desc.DepthOrArraySize = output_desc.MipLevels = 1;
      output_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
      output_desc.SampleDesc.Count = 1;
      output_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
      D3D12_HEAP_PROPERTIES heap_properties {};
      heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
      heap_properties.CreationNodeMask = heap_properties.VisibleNodeMask = 1;
      if (FAILED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &output_desc,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(sample.output12.put())))) return false;
      device->GetCopyableFootprints(&output_desc, 0, 1, 0, &sample.footprint12, nullptr, nullptr, &sample.bytes12);
      D3D12_RESOURCE_DESC readback_desc {};
      readback_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      readback_desc.Width = sample.bytes12;
      readback_desc.Height = readback_desc.DepthOrArraySize = readback_desc.MipLevels = 1;
      readback_desc.SampleDesc.Count = 1;
      readback_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      heap_properties.Type = D3D12_HEAP_TYPE_READBACK;
      if (FAILED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &readback_desc,
          D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(sample.readback12.put())))) return false;

      const D3D12_DESCRIPTOR_HEAP_DESC descriptors_desc {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
      if (FAILED(device->CreateDescriptorHeap(&descriptors_desc, IID_PPV_ARGS(sample.descriptors12.put()))) ||
          FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(sample.allocator12.put()))) ||
          FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, sample.allocator12.get(), sample.pipeline->pipeline12.get(), IID_PPV_ARGS(sample.commands12.put()))) ||
          FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(sample.fence12.put())))) return false;

      auto uav_handle = sample.descriptors12->GetCPUDescriptorHandleForHeapStart();
      uav_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
      D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc {};
      uav_desc.Format = output_desc.Format;
      uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      device->CreateUnorderedAccessView(sample.output12.get(), nullptr, &uav_desc, uav_handle);
    } else {
      // Reuse is reached only through a completed sample; neither the allocator
      // nor its descriptors can still be referenced by submitted GPU commands.
      if (FAILED(sample.allocator12->Reset()) || FAILED(sample.commands12->Reset(
          sample.allocator12.get(), sample.pipeline->pipeline12.get()))) return false;
    }
    if (sample.fence_value12 >= UINT64_MAX - 1) return false;
    ++sample.fence_value12;

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc {};
    srv_desc.Format = format;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Texture2D.MipLevels = 1;
    auto cpu_handle = sample.descriptors12->GetCPUDescriptorHandleForHeapStart();
    device->CreateShaderResourceView(sample.source12.get(), &srv_desc, cpu_handle);
    D3D12_RESOURCE_BARRIER source_barrier {};
    source_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    source_barrier.Transition = {sample.source12.get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
      before, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    sample.commands12->ResourceBarrier(1, &source_barrier);
    ID3D12DescriptorHeap *descriptor_heap = sample.descriptors12.get();
    sample.commands12->SetDescriptorHeaps(1, &descriptor_heap);
    sample.commands12->SetComputeRootSignature(sample.pipeline->root12.get());
    sample.commands12->SetComputeRootDescriptorTable(0, sample.descriptors12->GetGPUDescriptorHandleForHeapStart());
    sample.commands12->SetComputeRoot32BitConstants(1, UINT(sample.region.size()), sample.region.data(), 0);
    sample.commands12->Dispatch(sample.region[4] ? sample.region[8] : (sample.region[8] + 7) / 8,
      sample.region[4] ? sample.region[9] : (sample.region[9] + 7) / 8, 1);
    std::swap(source_barrier.Transition.StateBefore, source_barrier.Transition.StateAfter);
    D3D12_RESOURCE_BARRIER output_barrier {};
    output_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    output_barrier.Transition = {sample.output12.get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
    const D3D12_RESOURCE_BARRIER barriers[] = {source_barrier, output_barrier};
    sample.commands12->ResourceBarrier(2, barriers);
    D3D12_TEXTURE_COPY_LOCATION src {}, dst {};
    src.pResource = sample.output12.get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = sample.readback12.get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = sample.footprint12;
    sample.commands12->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    // Start and finish every executed list in UAV state. An abandoned list has
    // not changed that state, so no per-submission state tracking is required.
    std::swap(output_barrier.Transition.StateBefore, output_barrier.Transition.StateAfter);
    sample.commands12->ResourceBarrier(1, &output_barrier);
    if (FAILED(sample.commands12->Close())) return false;
    sample.queue12.retain(reinterpret_cast<ID3D12CommandQueue *>(sample.native_queue));
    return sample.queue12->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT;
  }

  void read_sample(sample_t &sample) {
    const UINT tiles_x = sample.region[8], tiles_y = sample.region[9];
    if (!tiles_x || !tiles_y || std::uint64_t(tiles_x) * tiles_y > tile_capacity) return;
    std::vector<float> values(tiles_x * tiles_y);
    bool valid_range = sample.region[4] != 0, have_range = false;
    auto &moments = sample.result.moments;
    bool valid_moments = moments.supplied && std::isfinite(moments.A) &&
      (moments.A == 0.f || std::isnormal(moments.A)) && std::isnormal(moments.inverseB);
    double center = 0., sum_centered = 0., sum_centered_squares = 0.;
    std::uint64_t moment_count = 0;
    float range_min = std::numeric_limits<float>::max(), range_max = -std::numeric_limits<float>::max();
    const auto copy_rows = [&](const std::uint8_t *bytes, UINT pitch) {
      const auto record = [&](UINT index) {
        std::array<float, 4> cell;
        std::memcpy(cell.data(), bytes + (index / storage_width) * pitch +
          (index % storage_width) * sizeof(cell), sizeof(cell));
        return cell;
      };
      for (UINT tile = 0; tile < values.size(); ++tile) {
        const auto cell = record(tile);
        values[tile] = cell[0];
        if (!sample.region[4]) continue;
        const auto horizontal = sunshine_depth_statistics::tile_bounds(tile % tiles_x,
          sample.result.viewport_width, tiles_x);
        const auto vertical = sunshine_depth_statistics::tile_bounds(tile / tiles_x,
          sample.result.viewport_height, tiles_y);
        const auto count = std::uint64_t(horizontal.size()) * vertical.size();
        const bool tile_valid = cell[3] == 1.f && count && std::isfinite(cell[1]) &&
          std::isfinite(cell[2]) && cell[1] <= cell[2];
        if (moments.supplied) {
          const auto measured = record(tile + tile_capacity);
          if (measured[3] == 2.f && cell[3] == 2.f && count == 0) {
            // Empty partitions contribute neither samples nor a fabricated zero.
          } else if (!tile_valid || measured[3] != 1.f || !std::isfinite(measured[0]) || measured[0] < 0.f ||
              !std::isfinite(measured[1]) || measured[1] < 0.f || !std::isfinite(measured[2]) || measured[2] < 0.f ||
              (measured[0] == 0.f && (measured[1] != 0.f || measured[2] != 0.f))) {
            valid_moments = false;
          } else {
            // Match the shader's decoded tile minimum in FP32. The remaining
            // arithmetic only shifts nonnegative moments to a smaller origin;
            // no subtraction of large uncentered sums occurs.
            const float raw_origin = moments.inverseB > 0.f ? cell[1] : cell[2];
            const float delta = raw_origin - moments.A;
            const float decoded_origin = delta * moments.inverseB;
            const double scale = measured[0];
            const double tile_sum = scale * measured[1];
            const double tile_squares = scale * scale * measured[2];
            if (!std::isfinite(decoded_origin) || decoded_origin < 0.f) valid_moments = false;
            else {
              if (!moment_count) center = decoded_origin;
              if (decoded_origin < center) {
                const double shift = center - decoded_origin;
                sum_centered_squares += 2. * shift * sum_centered + double(moment_count) * shift * shift;
                sum_centered += double(moment_count) * shift;
                center = decoded_origin;
              }
              const double shift = double(decoded_origin) - center;
              sum_centered_squares += tile_squares + 2. * shift * tile_sum + double(count) * shift * shift;
              sum_centered += tile_sum + double(count) * shift;
              moment_count += count;
            }
          }
        }
        if (cell[3] == 2.f && !count) continue;
        if (!tile_valid) {
          valid_range = false;
          continue;
        }
        range_min = std::min(range_min, cell[1]);
        range_max = std::max(range_max, cell[2]);
        have_range = true;
      }
    };
    if (sample.readback12.get()) {
      const D3D12_RANGE range {0, SIZE_T(sample.bytes12)};
      void *mapped = nullptr;
      if (FAILED(sample.readback12->Map(0, &range, &mapped)) || !mapped) return;
      copy_rows(static_cast<const std::uint8_t *>(mapped) + sample.footprint12.Offset, sample.footprint12.Footprint.RowPitch);
      const D3D12_RANGE written {0, 0};
      sample.readback12->Unmap(0, &written);
    } else {
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(sample.context11->Map(sample.readback11.get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped)) || !mapped.pData) return;
      copy_rows(static_cast<const std::uint8_t *>(mapped.pData), mapped.RowPitch);
      sample.context11->Unmap(sample.readback11.get(), 0);
    }
    sample.result.values = std::move(values);
    sample.result.valid = true;
    sample.result.range_valid = valid_range && have_range;
    sample.result.range_min = sample.result.range_valid ? range_min : 0.f;
    sample.result.range_max = sample.result.range_valid ? range_max : 0.f;
    // Legacy sums remain diagnostics, reconstructed in double from the stable
    // centered measurement. Consumers needing contrast use the fields directly.
    const double sum = sum_centered + double(moment_count) * center;
    const double sum_squares = sum_centered_squares + 2. * center * sum_centered + double(moment_count) * center * center;
    moments.valid = valid_moments && sample.result.range_valid && moment_count == moments.count &&
      std::isfinite(center) && std::isfinite(sum_centered) && std::isfinite(sum_centered_squares) &&
      std::isfinite(sum) && std::isfinite(sum_squares);
    moments.center = moments.valid ? center : 0.;
    moments.sum_centered = moments.valid ? sum_centered : 0.;
    moments.sum_centered_squares = moments.valid ? sum_centered_squares : 0.;
    moments.sum = moments.valid ? sum : 0.;
    moments.sum_squares = moments.valid ? sum_squares : 0.;
  }

  class sampler_t {
  public:
    ~sampler_t() {
      // Never destroy objects referenced by submitted commands whose completion
      // could not be proved. At most one bundle is intentionally retained until
      // process exit; quarantine prevents accumulating any further work.
      if (pending_ && pending_->executed && !complete(*pending_)) pending_.release();
    }

    bool submit(api::effect_runtime *runtime, api::command_list *commands, const sample_request &request) {
      std::lock_guard<std::mutex> lock(mutex_);
      reap_retired();
      if (pending_ || disabled_ || !runtime || !commands || !request.source.handle) return false;
      auto *queue = runtime->get_command_queue();
      if (!queue || commands != queue->get_immediate_command_list()) return false;
      const auto api = runtime->get_device()->get_api();
      if (api != api::device_api::d3d11 && api != api::device_api::d3d12) return false;
      if (!shader_.get()) {
        com_ptr<ID3DBlob> errors;
        if (FAILED(D3DCompile(shader_source, std::strlen(shader_source), "SunshineDepthContent", nullptr, nullptr,
            "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, shader_.put(), errors.put()))) {
          warning("Sunshine depth selection: content sampler shader compilation failed; sampling disabled");
          disabled_ = true;
          return false;
        }
      }
      auto pipeline = pipelines_.acquire(api, runtime->get_device()->get_native(), runtime, shader_.get());
      if (!pipeline) return false;
      auto sample = take_sample(spare_, std::move(pipeline));
      sample->runtime = runtime;
      sample->swapchain = runtime->get_native();
      sample->native_queue = queue->get_native();
      const bool prepared = api == api::device_api::d3d12 ?
        prepare12(*sample, request) : prepare11(*sample, request);
      if (!prepared) return false;
      pending_ = std::move(sample);
      return true;
    }

    bool busy() {
      std::lock_guard<std::mutex> lock(mutex_);
      reap_retired();
      return disabled_ || bool(pending_);
    }

    void begin_present(api::swapchain *swapchain) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (pending_ && !pending_->executed && pending_->swapchain == swapchain->get_native()) {
        // A private list that was never executed has no GPU lifetime to protect.
        // Report a failed sample; do not later execute it against another frame.
        pending_->abandoned = true;
      }
      reap_retired();
    }

    void finish_present(api::command_queue *queue, api::swapchain *swapchain) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!pending_ || pending_->executed || pending_->abandoned || pending_->retired ||
          pending_->swapchain != swapchain->get_native() || pending_->native_queue != queue->get_native()) return;
      auto &sample = *pending_;
      sample.executed = true;
      if (sample.queue12.get()) {
        ID3D12CommandList *list = sample.commands12.get();
        sample.queue12->ExecuteCommandLists(1, &list);
        if (FAILED(sample.queue12->Signal(sample.fence12.get(), sample.fence_value12))) {
          quarantine(sample, "Sunshine depth selection: sampler fence signal failed; one GPU bundle retained and sampling disabled");
          return;
        }
      } else {
        sample.context11->ExecuteCommandList(sample.commands11.get(), TRUE);
      }
      sample.signalled = true;
    }

    std::optional<sample_result> poll(api::effect_runtime *runtime) {
      std::lock_guard<std::mutex> lock(mutex_);
      reap_retired();
      if (!pending_ || pending_->runtime != runtime || !complete(*pending_)) return std::nullopt;
      auto sample = std::move(pending_);
      if (!sample->abandoned) read_sample(*sample);
      auto result = std::move(sample->result);
      if (!sample->abandoned) {
        sample->clear_capture();
        spare_ = std::move(sample);
      }
      return result;
    }

    void retire(api::effect_runtime *runtime) {
      std::lock_guard<std::mutex> lock(mutex_);
      pipelines_.retire(runtime);
      if (spare_ && spare_->runtime == runtime) spare_.reset();
      if (pending_ && pending_->runtime == runtime) pending_->retired = true;
      reap_retired();
    }

  private:
    void quarantine(sample_t &sample, const char *message) {
      if (!sample.quarantined) warning(message);
      sample.quarantined = true;
      disabled_ = true;
    }

    bool complete(sample_t &sample) {
      if (sample.abandoned) return true;
      if (!sample.executed || !sample.signalled || sample.quarantined) return false;
      if (sample.fence12.get()) {
        const UINT64 completed = sample.fence12->GetCompletedValue();
        if (completed == UINT64_MAX) {
          quarantine(sample, "Sunshine depth selection: sampler device removed; one GPU bundle retained and sampling disabled");
          return false;
        }
        return completed >= sample.fence_value12;
      }
      BOOL ready = FALSE;
      const HRESULT hr = sample.context11->GetData(sample.query11.get(), &ready, sizeof(ready), D3D11_ASYNC_GETDATA_DONOTFLUSH);
      if (FAILED(hr)) {
        quarantine(sample, "Sunshine depth selection: sampler query failed; one GPU bundle retained and sampling disabled");
        return false;
      }
      return hr == S_OK && ready;
    }

    void reap_retired() {
      if (pending_ && pending_->retired && (!pending_->executed || complete(*pending_))) pending_.reset();
    }

    std::mutex mutex_;
    com_ptr<ID3DBlob> shader_;
    pipeline_cache_t pipelines_;
    std::unique_ptr<sample_t> pending_;
    // At most one submitted bundle. Reuse its fixed-size scratch after completion.
    std::unique_ptr<sample_t> spare_;
    bool disabled_ = false;
  };

  sampler_t sampler;
}

bool submit(api::effect_runtime *runtime, api::command_list *commands, const sample_request &request) {
  return sampler.submit(runtime, commands, request);
}
bool busy() { return sampler.busy(); }
std::optional<sample_result> poll(api::effect_runtime *runtime) { return sampler.poll(runtime); }
void begin_present(api::swapchain *swapchain) { sampler.begin_present(swapchain); }
void finish_present(api::command_queue *queue, api::swapchain *swapchain) { sampler.finish_present(queue, swapchain); }
void retire_runtime(api::effect_runtime *runtime) { sampler.retire(runtime); }
}
