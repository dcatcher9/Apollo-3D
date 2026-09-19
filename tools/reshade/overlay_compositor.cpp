// SPDX-License-Identifier: GPL-3.0-only
#include "overlay_compositor.h"

#include "overlay_shaders.h"

#include <array>
#include <cstring>
#include <d3d11_4.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <imgui.h>
#include <mutex>
#include <reshade.hpp>
#include <utility>

static_assert(IMGUI_VERSION_NUM == 19250, "The native GUI callback ABI is pinned to ReShade 6.8.0");
static_assert(RESHADE_API_VERSION == 20);

namespace sunshine::overlay {
  namespace api = reshade::api;

  namespace {
    template<class T>
    struct com_t {
      T *p = nullptr;

      ~com_t() {
        reset();
      }

      com_t() = default;
      com_t(const com_t &) = delete;
      com_t &operator=(const com_t &) = delete;

      T *operator->() const {
        return p;
      }

      T **put() {
        reset();
        return &p;
      }

      void reset(T *v = nullptr) {
        if (p) {
          p->Release();
        }
        p = v;
      }

      void retain(T *v) {
        if (v) {
          v->AddRef();
        }
        reset(v);
      }
    };

    DXGI_FORMAT typed(DXGI_FORMAT f) {
      switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
          return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
          return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
          return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
          return DXGI_FORMAT_R16G16B16A16_FLOAT;
        default:
          return f;
      }
    }

    bool compile(const char *source, const char *entry, const char *target, bool hdr, com_t<ID3DBlob> &code) {
      const D3D_SHADER_MACRO macros[] = {{"EXPORT_HDR", hdr ? "1" : "0"}, {nullptr, nullptr}};
      com_t<ID3DBlob> errors;
      if (SUCCEEDED(D3DCompile(source, std::strlen(source), "Sunshine overlay compositor", macros, nullptr, entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, code.put(), errors.put()))) {
        return true;
      }
      if (errors.p) {
        reshade::log::message(reshade::log::level::error, static_cast<const char *>(errors->GetBufferPointer()));
      }
      return false;
    }

    D3D12_RENDER_TARGET_BLEND_DESC blend12(bool enabled) {
      D3D12_RENDER_TARGET_BLEND_DESC d {};
      d.BlendEnable = enabled;
      d.SrcBlend = D3D12_BLEND_SRC_ALPHA;
      d.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
      d.BlendOp = D3D12_BLEND_OP_ADD;
      d.SrcBlendAlpha = D3D12_BLEND_ONE;
      d.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
      d.BlendOpAlpha = D3D12_BLEND_OP_ADD;
      d.LogicOp = D3D12_LOGIC_OP_NOOP;
      d.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
      return d;
    }

    D3D12_RASTERIZER_DESC raster12() {
      D3D12_RASTERIZER_DESC d {};
      d.FillMode = D3D12_FILL_MODE_SOLID;
      d.CullMode = D3D12_CULL_MODE_NONE;
      d.DepthClipEnable = TRUE;
      return d;
    }

    void barrier12(ID3D12GraphicsCommandList *cmd, ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
      D3D12_RESOURCE_BARRIER b {};
      b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
      cmd->ResourceBarrier(1, &b);
    }
  }  // namespace

  struct compositor_t::impl_t {
    std::mutex callback_mutex;
    api::effect_runtime *runtime = nullptr;
    api::command_list *commands = nullptr;
    api::device_api backend = api::device_api::d3d11;
    UINT width = 0, height = 0;
    DXGI_FORMAT native_format = DXGI_FORMAT_UNKNOWN, export_format = DXGI_FORMAT_UNKNOWN;
    bool hdr = false, initialized = false, armed = false, appended = false, captured = false;

    com_t<ID3D11Device1> device11;
    com_t<ID3D11DeviceContext1> context11;
    com_t<ID3DDeviceContextState> state11;
    com_t<ID3D11Texture2D> layer11, source11, destination11;
    com_t<ID3D11RenderTargetView> native_rtv11, layer_rtv11, destination_rtv11;
    com_t<ID3D11ShaderResourceView> layer_srv11, source_srv11;
    com_t<ID3D11PixelShader> capture_ps11, composite_ps11;
    com_t<ID3D11VertexShader> composite_vs11;
    com_t<ID3D11BlendState> capture_blend11;
    com_t<ID3D11RasterizerState> raster11;
    com_t<ID3D11DepthStencilState> depth11;

    com_t<ID3D12Device> device12;
    com_t<ID3D12GraphicsCommandList> command12;
    com_t<ID3D12GraphicsCommandList4> command4;
    com_t<ID3D12Resource> layer12, source12, destination12, native_target12;
    com_t<ID3D12DescriptorHeap> rtv_heap12, srv_heap12;
    com_t<ID3D12RootSignature> gui_root12, composite_root12;
    com_t<ID3D12PipelineState> capture_pso12, composite_pso12;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv12[3] {};
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu12 {};
    UINT srv_stride12 = 0;
    bool native_render_pass = false;

    void disarm() {
      armed = false;
      appended = false;
      captured = false;
      runtime = nullptr;
      commands = nullptr;
    }

    bool create11() {
      D3D11_TEXTURE2D_DESC texture {};
      texture.Width = width;
      texture.Height = height;
      texture.MipLevels = texture.ArraySize = texture.SampleDesc.Count = 1;
      texture.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      texture.Usage = D3D11_USAGE_DEFAULT;
      texture.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
      if (FAILED(device11->CreateTexture2D(&texture, nullptr, layer11.put())) || FAILED(device11->CreateRenderTargetView(layer11.p, nullptr, layer_rtv11.put())) || FAILED(device11->CreateShaderResourceView(layer11.p, nullptr, layer_srv11.put()))) {
        return false;
      }

      com_t<ID3DBlob> capture, vs, ps;
      if (!compile(gui_shader, "gui_ps", "ps_5_0", hdr, capture) || !compile(composite_shader, "fullscreen_vs", "vs_5_0", hdr, vs) || !compile(composite_shader, "composite_ps", "ps_5_0", hdr, ps) || FAILED(device11->CreatePixelShader(capture->GetBufferPointer(), capture->GetBufferSize(), nullptr, capture_ps11.put())) || FAILED(device11->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(), nullptr, composite_vs11.put())) || FAILED(device11->CreatePixelShader(ps->GetBufferPointer(), ps->GetBufferSize(), nullptr, composite_ps11.put()))) {
        return false;
      }

      D3D11_BLEND_DESC blend {};
      blend.IndependentBlendEnable = TRUE;
      for (auto &rt : blend.RenderTarget) {
        rt.BlendEnable = TRUE;
        rt.SrcBlend = D3D11_BLEND_SRC_ALPHA;
        rt.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        rt.BlendOp = D3D11_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D11_BLEND_ONE;
        rt.DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
        rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
        rt.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
      }
      D3D11_RASTERIZER_DESC raster {};
      raster.FillMode = D3D11_FILL_SOLID;
      raster.CullMode = D3D11_CULL_NONE;
      raster.DepthClipEnable = TRUE;
      D3D11_DEPTH_STENCIL_DESC depth {};
      depth.DepthFunc = D3D11_COMPARISON_ALWAYS;
      depth.FrontFace.StencilFunc = depth.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
      depth.FrontFace.StencilFailOp = depth.FrontFace.StencilDepthFailOp = depth.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
      depth.BackFace = depth.FrontFace;
      const D3D_FEATURE_LEVEL level = device11->GetFeatureLevel();
      const UINT state_flags = (device11->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED) != 0 ?
                                 D3D11_1_CREATE_DEVICE_CONTEXT_STATE_SINGLETHREADED :
                                 0;
      return SUCCEEDED(device11->CreateBlendState(&blend, capture_blend11.put())) &&
             SUCCEEDED(device11->CreateRasterizerState(&raster, raster11.put())) &&
             SUCCEEDED(device11->CreateDepthStencilState(&depth, depth11.put())) &&
             SUCCEEDED(device11->CreateDeviceContextState(state_flags, &level, 1, D3D11_SDK_VERSION, __uuidof(ID3D11Device), nullptr, state11.put()));
    }

    bool create_roots12() {
      // Exact native GUI layout (API 20), without switching the bound native
      // signature at capture time. Changing signatures would discard its b0/s0.
      D3D12_DESCRIPTOR_RANGE1 ranges[2] {};
      ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, 0};
      ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE, 0};
      D3D12_ROOT_PARAMETER1 params[3] {};
      for (UINT i = 0; i < 2; ++i) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[i].DescriptorTable = {1, &ranges[i]};
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      }
      params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      params[2].Constants = {0, 0, 18};
      params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      D3D12_VERSIONED_ROOT_SIGNATURE_DESC root {};
      root.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
      root.Desc_1_1.NumParameters = 3;
      root.Desc_1_1.pParameters = params;
      root.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;
      D3D12_FEATURE_DATA_D3D12_OPTIONS7 options {};
      if (SUCCEEDED(device12->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options))) && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED) {
        root.Desc_1_1.Flags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_AMPLIFICATION_SHADER_ROOT_ACCESS |
                               D3D12_ROOT_SIGNATURE_FLAG_DENY_MESH_SHADER_ROOT_ACCESS;
      }
      com_t<ID3DBlob> serialized, errors;
      if (FAILED(D3D12SerializeVersionedRootSignature(&root, serialized.put(), errors.put())) || FAILED(device12->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(gui_root12.put())))) {
        return false;
      }

      ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE, 0};
      params[0].DescriptorTable = {1, &ranges[0]};
      root.Desc_1_1.NumParameters = 1;
      if (FAILED(D3D12SerializeVersionedRootSignature(&root, serialized.put(), errors.put())) || FAILED(device12->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(composite_root12.put())))) {
        return false;
      }
      return true;
    }

    bool create12() {
      D3D12_HEAP_PROPERTIES heap {};
      heap.Type = D3D12_HEAP_TYPE_DEFAULT;
      heap.CreationNodeMask = heap.VisibleNodeMask = 1;
      D3D12_RESOURCE_DESC texture {};
      texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      texture.Width = width;
      texture.Height = height;
      texture.DepthOrArraySize = texture.MipLevels = texture.SampleDesc.Count = 1;
      texture.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
      D3D12_CLEAR_VALUE clear {};
      clear.Format = texture.Format;
      if (FAILED(device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear, IID_PPV_ARGS(layer12.put())))) {
        return false;
      }
      D3D12_DESCRIPTOR_HEAP_DESC descriptors {};
      descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
      descriptors.NumDescriptors = 3;
      if (FAILED(device12->CreateDescriptorHeap(&descriptors, IID_PPV_ARGS(rtv_heap12.put())))) {
        return false;
      }
      const auto base = rtv_heap12->GetCPUDescriptorHandleForHeapStart();
      const auto stride = device12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
      for (UINT i = 0; i < 3; ++i) {
        rtv12[i].ptr = base.ptr + i * stride;
      }
      device12->CreateRenderTargetView(layer12.p, nullptr, rtv12[1]);
      descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
      descriptors.NumDescriptors = 2;
      descriptors.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
      if (FAILED(device12->CreateDescriptorHeap(&descriptors, IID_PPV_ARGS(srv_heap12.put())))) {
        return false;
      }
      srv_gpu12 = srv_heap12->GetGPUDescriptorHandleForHeapStart();
      srv_stride12 = device12->GetDescriptorHandleIncrementSize(descriptors.Type);
      auto cpu = srv_heap12->GetCPUDescriptorHandleForHeapStart();
      cpu.ptr += srv_stride12;
      D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
      srv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      srv.Texture2D.MipLevels = 1;
      device12->CreateShaderResourceView(layer12.p, &srv, cpu);
      if (!create_roots12()) {
        return false;
      }

      com_t<ID3DBlob> capture_vs, capture_ps, vs, ps;
      if (!compile(gui_shader, "gui_vs", "vs_5_0", hdr, capture_vs) || !compile(gui_shader, "gui_ps", "ps_5_0", hdr, capture_ps) || !compile(composite_shader, "fullscreen_vs", "vs_5_0", hdr, vs) || !compile(composite_shader, "composite_ps", "ps_5_0", hdr, ps)) {
        return false;
      }
      const D3D12_INPUT_ELEMENT_DESC inputs[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(ImDrawVert, pos), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(ImDrawVert, uv), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, offsetof(ImDrawVert, col), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      };
      D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline {};
      pipeline.pRootSignature = gui_root12.p;
      pipeline.VS = {capture_vs->GetBufferPointer(), capture_vs->GetBufferSize()};
      pipeline.PS = {capture_ps->GetBufferPointer(), capture_ps->GetBufferSize()};
      pipeline.BlendState.IndependentBlendEnable = TRUE;
      for (auto &rt : pipeline.BlendState.RenderTarget) {
        rt = blend12(true);
      }
      pipeline.SampleMask = UINT_MAX;
      pipeline.RasterizerState = raster12();
      pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
      pipeline.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
      pipeline.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
      pipeline.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
      pipeline.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
      pipeline.DepthStencilState.BackFace = pipeline.DepthStencilState.FrontFace;
      pipeline.InputLayout = {inputs, 3};
      pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
      pipeline.NumRenderTargets = 2;
      pipeline.RTVFormats[0] = native_format;
      pipeline.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
      pipeline.SampleDesc.Count = 1;
      if (FAILED(device12->CreateGraphicsPipelineState(&pipeline, IID_PPV_ARGS(capture_pso12.put())))) {
        return false;
      }
      pipeline.pRootSignature = composite_root12.p;
      pipeline.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
      pipeline.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
      for (auto &rt : pipeline.BlendState.RenderTarget) {
        rt = blend12(false);
      }
      pipeline.InputLayout = {};
      pipeline.NumRenderTargets = 1;
      pipeline.RTVFormats[0] = export_format;
      pipeline.RTVFormats[1] = DXGI_FORMAT_UNKNOWN;
      return SUCCEEDED(device12->CreateGraphicsPipelineState(&pipeline, IID_PPV_ARGS(composite_pso12.put())));
    }

    bool bind11(api::resource_view native_rtv, api::resource source, api::resource destination) {
      native_rtv11.retain(reinterpret_cast<ID3D11RenderTargetView *>(native_rtv.handle));
      source11.retain(reinterpret_cast<ID3D11Texture2D *>(source.handle));
      destination11.retain(reinterpret_cast<ID3D11Texture2D *>(destination.handle));
      D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
      srv.Format = export_format;
      srv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
      srv.Texture2D.MipLevels = 1;
      D3D11_RENDER_TARGET_VIEW_DESC rtv {};
      rtv.Format = export_format;
      rtv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
      return SUCCEEDED(device11->CreateShaderResourceView(source11.p, &srv, source_srv11.put())) &&
             SUCCEEDED(device11->CreateRenderTargetView(destination11.p, &rtv, destination_rtv11.put()));
    }

    bool bind12(api::resource_view native_rtv, api::resource source, api::resource destination) {
      source12.retain(reinterpret_cast<ID3D12Resource *>(source.handle));
      destination12.retain(reinterpret_cast<ID3D12Resource *>(destination.handle));
      device12->CopyDescriptorsSimple(1, rtv12[0], {static_cast<SIZE_T>(native_rtv.handle)}, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
      D3D12_RENDER_TARGET_VIEW_DESC rtv {};
      rtv.Format = export_format;
      rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
      device12->CreateRenderTargetView(destination12.p, &rtv, rtv12[2]);
      D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
      srv.Format = export_format;
      srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      srv.Texture2D.MipLevels = 1;
      device12->CreateShaderResourceView(source12.p, &srv, srv_heap12->GetCPUDescriptorHandleForHeapStart());
      return true;
    }

    static void capture(const ImDrawList *, const ImDrawCmd *draw) {
      auto *self = static_cast<impl_t *>(draw->UserCallbackData);
      if (!self) {
        return;
      }
      std::lock_guard<std::mutex> lock(self->callback_mutex);
      if (!self->armed || self->captured) {
        return;
      }
      // Keep the runtime's own root signature, descriptor heaps, VB/IB, b0/s0,
      // viewport and all subsequent native draw/scissor/texture commands intact.
      self->commands->end_render_pass();
      if (self->backend == api::device_api::d3d11) {
        ID3D11RenderTargetView *rtvs[] = {self->native_rtv11.p, self->layer_rtv11.p};
        const float clear[4] {};
        self->context11->ClearRenderTargetView(self->layer_rtv11.p, clear);
        self->context11->OMSetRenderTargets(2, rtvs, nullptr);
        self->context11->OMSetBlendState(self->capture_blend11.p, nullptr, UINT_MAX);
        self->context11->PSSetShader(self->capture_ps11.p, nullptr, 0);
      } else {
        if (self->native_render_pass) {
          D3D12_RENDER_PASS_RENDER_TARGET_DESC targets[2] {};
          for (UINT i = 0; i < 2; ++i) {
            targets[i].cpuDescriptor = self->rtv12[i];
            targets[i].BeginningAccess.Type = i == 0 ? D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_PRESERVE : D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
            targets[i].BeginningAccess.Clear.ClearValue.Format = i == 0 ? self->native_format : DXGI_FORMAT_R16G16B16A16_FLOAT;
            targets[i].EndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
          }
          self->command4->BeginRenderPass(2, targets, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
        } else {
          const float clear[4] {};
          self->command12->ClearRenderTargetView(self->rtv12[1], clear, 0, nullptr);
          self->command12->OMSetRenderTargets(2, self->rtv12, FALSE, nullptr);
        }
        self->command12->SetPipelineState(self->capture_pso12.p);
      }
      // The native renderer closes this replacement pass after all real UI draws,
      // including windows/tooltips/cursor added after our background callback.
      self->captured = true;
    }

    bool finish11() {
      // Windows 11's native context-state exchange preserves ALL application
      // bindings, including class instances/UAVs/SO, rather than a partial save.
      com_t<ID3DDeviceContextState> previous;
      context11->SwapDeviceContextState(state11.p, previous.put());
      context11->OMSetRenderTargets(1, &destination_rtv11.p, nullptr);
      context11->OMSetBlendState(nullptr, nullptr, UINT_MAX);
      context11->OMSetDepthStencilState(depth11.p, 0);
      context11->RSSetState(raster11.p);
      const D3D11_VIEWPORT viewport {0, 0, static_cast<float>(width * 2), static_cast<float>(height), 0, 1};
      context11->RSSetViewports(1, &viewport);
      context11->IASetInputLayout(nullptr);
      context11->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      context11->VSSetShader(composite_vs11.p, nullptr, 0);
      context11->PSSetShader(composite_ps11.p, nullptr, 0);
      context11->GSSetShader(nullptr, nullptr, 0);
      context11->HSSetShader(nullptr, nullptr, 0);
      context11->DSSetShader(nullptr, nullptr, 0);
      ID3D11ShaderResourceView *srvs[] = {source_srv11.p, layer_srv11.p};
      context11->PSSetShaderResources(0, 2, srvs);
      context11->Draw(3, 0);
      context11->SwapDeviceContextState(previous.p, nullptr);
      return true;
    }

    bool finish12() {
      // ReShade owns this immediate command list; it is separate from game lists
      // and naturally submitted after reshade_present. Never flush it here.
      barrier12(command12.p, layer12.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      barrier12(command12.p, destination12.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
      command12->OMSetRenderTargets(1, &rtv12[2], FALSE, nullptr);
      command12->SetPipelineState(composite_pso12.p);
      command12->SetGraphicsRootSignature(composite_root12.p);
      command12->SetDescriptorHeaps(1, &srv_heap12.p);
      command12->SetGraphicsRootDescriptorTable(0, srv_gpu12);
      const D3D12_VIEWPORT viewport {0, 0, static_cast<float>(width * 2), static_cast<float>(height), 0, 1};
      const D3D12_RECT scissor {0, 0, static_cast<LONG>(width * 2), static_cast<LONG>(height)};
      command12->RSSetViewports(1, &viewport);
      command12->RSSetScissorRects(1, &scissor);
      command12->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      command12->DrawInstanced(3, 1, 0, 0);
      command12->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
      barrier12(command12.p, destination12.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
      barrier12(command12.p, layer12.p, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
      return true;
    }
  };

  compositor_t::compositor_t():
      impl_(std::make_unique<impl_t>()) {}

  compositor_t::~compositor_t() = default;

  bool compositor_t::prepare(api::effect_runtime *runtime, api::resource_view native_rtv, api::resource source, api::resource destination, std::uint32_t width, std::uint32_t height, api::color_space source_color, api::color_space export_color) {
    cancel();
    if (!runtime || !native_rtv.handle || !source.handle || !destination.handle || source.handle == destination.handle || !width || width > 8192 || !height || height > 16384 || (export_color != api::color_space::srgb && export_color != api::color_space::scrgb) || (source_color != api::color_space::srgb && source_color != api::color_space::scrgb && source_color != api::color_space::hdr10_pq) || ((source_color == api::color_space::srgb) != (export_color == api::color_space::srgb))) {
      return false;
    }
    auto *device = runtime->get_device();
    auto *commands = runtime->get_command_queue()->get_immediate_command_list();
    const auto backend = device->get_api();
    if (backend != api::device_api::d3d11 && backend != api::device_api::d3d12) {
      return false;
    }
    const auto native_resource = device->get_resource_from_view(native_rtv);
    if (!native_resource.handle) {
      return false;
    }
    DXGI_FORMAT native_format, export_format;
    if (backend == api::device_api::d3d11) {
      D3D11_TEXTURE2D_DESC native {}, input {}, output {};
      reinterpret_cast<ID3D11Texture2D *>(native_resource.handle)->GetDesc(&native);
      reinterpret_cast<ID3D11Texture2D *>(source.handle)->GetDesc(&input);
      reinterpret_cast<ID3D11Texture2D *>(destination.handle)->GetDesc(&output);
      if (native.Width != width || native.Height != height || native.SampleDesc.Count != 1 || input.Width != width * 2 || input.Height != height || input.SampleDesc.Count != 1 || input.ArraySize != 1 || output.Width != width * 2 || output.Height != height || output.SampleDesc.Count != 1 || output.ArraySize != 1 || typed(input.Format) != typed(output.Format)) {
        return false;
      }
      native_format = typed(native.Format);
      export_format = typed(input.Format);
    } else {
      const auto native = reinterpret_cast<ID3D12Resource *>(native_resource.handle)->GetDesc();
      const auto input = reinterpret_cast<ID3D12Resource *>(source.handle)->GetDesc();
      const auto output = reinterpret_cast<ID3D12Resource *>(destination.handle)->GetDesc();
      if (native.Width != width || native.Height != height || native.SampleDesc.Count != 1 || input.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || input.Width != width * 2 || input.Height != height || input.SampleDesc.Count != 1 || input.DepthOrArraySize != 1 || output.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || output.Width != width * 2 || output.Height != height || output.SampleDesc.Count != 1 || output.DepthOrArraySize != 1 || typed(input.Format) != typed(output.Format)) {
        return false;
      }
      native_format = typed(native.Format);
      export_format = typed(input.Format);
    }
    const bool hdr = export_color == api::color_space::scrgb;
    if ((hdr && export_format != DXGI_FORMAT_R16G16B16A16_FLOAT) || (!hdr && export_format != DXGI_FORMAT_R10G10B10A2_UNORM && export_format != DXGI_FORMAT_R8G8B8A8_UNORM && export_format != DXGI_FORMAT_B8G8R8A8_UNORM)) {
      return false;
    }
    if (!impl_->initialized || impl_->backend != backend || impl_->width != width || impl_->height != height || impl_->native_format != native_format || impl_->export_format != export_format || impl_->hdr != hdr) {
      impl_ = std::make_unique<impl_t>();
      auto &s = *impl_;
      s.width = width;
      s.height = height;
      s.backend = backend;
      s.native_format = native_format;
      s.export_format = export_format;
      s.hdr = hdr;
      if (backend == api::device_api::d3d11) {
        if (FAILED(reinterpret_cast<ID3D11Device *>(device->get_native())->QueryInterface(IID_PPV_ARGS(s.device11.put()))) || FAILED(reinterpret_cast<ID3D11DeviceContext *>(commands->get_native())->QueryInterface(IID_PPV_ARGS(s.context11.put()))) || !s.create11()) {
          return false;
        }
      } else {
        s.device12.retain(reinterpret_cast<ID3D12Device *>(device->get_native()));
        if (!s.create12()) {
          return false;
        }
      }
      s.initialized = true;
    }
    auto &s = *impl_;
    s.runtime = runtime;
    s.commands = commands;
    if (backend == api::device_api::d3d11) {
      if (!s.bind11(native_rtv, source, destination)) {
        return false;
      }
    } else {
      s.native_target12.retain(reinterpret_cast<ID3D12Resource *>(native_resource.handle));
      s.command12.retain(reinterpret_cast<ID3D12GraphicsCommandList *>(commands->get_native()));
      s.command4.reset();
      s.native_render_pass = SUCCEEDED(s.command12->QueryInterface(IID_PPV_ARGS(s.command4.put())));
      // Match the pinned runtime's VKD3D exclusion when choosing its pass type.
      constexpr GUID vkd3d_list = {0x77a86b09, 0x2bea, 0x4801, {0xb8, 0x9a, 0x37, 0x64, 0x8e, 0x10, 0x4a, 0xf1}};
      com_t<IUnknown> extension;
      if (s.command12->QueryInterface(vkd3d_list, reinterpret_cast<void **>(extension.put())) == S_OK) {
        s.native_render_pass = false;
      }
      if (!s.bind12(native_rtv, source, destination)) {
        return false;
      }
    }
    {
      std::lock_guard<std::mutex> lock(s.callback_mutex);
      s.armed = true;
    }
    return true;
  }

  void compositor_t::append_capture_callback() {
    auto &s = *impl_;
    std::lock_guard<std::mutex> lock(s.callback_mutex);
    if (s.armed && !s.appended) {
      ImGui::GetBackgroundDrawList(nullptr)->AddCallback(impl_t::capture, &s);
      s.appended = true;
    }
  }

  bool compositor_t::finish() {
    auto &s = *impl_;
    std::lock_guard<std::mutex> lock(s.callback_mutex);
    if (!s.armed || !s.captured) {
      s.disarm();
      return false;
    }
    const bool result = s.backend == api::device_api::d3d11 ? s.finish11() : s.finish12();
    s.disarm();
    return result;
  }

  void compositor_t::cancel() {
    std::lock_guard<std::mutex> lock(impl_->callback_mutex);
    impl_->disarm();
    // Retain all native objects; recorded GPU work and a cancelled ImDrawCmd can
    // still refer to this instance until the owner's generation fence completes.
  }
}  // namespace sunshine::overlay
