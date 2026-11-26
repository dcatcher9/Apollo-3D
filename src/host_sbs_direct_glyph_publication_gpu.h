/** @file Direct Glyph planning and selected-matte publication on the GPU. */
#pragma once

#ifdef _WIN32
  #include "generated/depth_coordinate_v2_contract.h"
  #include "host_sbs_compute_state_guard.h"
  #include "host_sbs_direct_glyph_tiles.h"
  #include "host_sbs_display_snapshot_rgb8_analysis_gpu.h"
  #include "host_sbs_shader_cache.h"

  #include <array>
  #include <bit>
  #include <cstdint>
  #include <d3d11.h>
  #include <filesystem>
  #include <string>
  #include <string_view>
  #include <utility>
  #include <wrl/client.h>

namespace models::host_sbs_direct_glyph_publication {
  using Microsoft::WRL::ComPtr;
  namespace shader_cache = host_sbs_shader_cache;

  inline constexpr UINT cutoff_alpha8 = 38u;
  inline constexpr UINT direct_work_word_count = 36u;
  inline constexpr UINT direct_fringe_request_bytes = 144u;
  inline constexpr UINT direct_fringe_flag = 32u;
  inline constexpr std::string_view empty_gate_shader_source_closure_sha256 =
    "f4e4ce4d14378bca175db941f5b216ebfc806dca8effaa2007f016a7df9ffdb2";
  inline constexpr std::array<shader_cache::shader_spec, 1u> empty_gate_shader_specs {{
    {"host_sbs_raw_matte_empty_gate_cs.hlsl", "prepare_empty_main", "cs_5_0"},
  }};

  // All views must belong to the same ordered observation as the actual C24 and Root.
  struct request_t {
    UINT width = 0u, height = 0u;
    std::uint64_t frame = 0u, observation_us = 0u, analysis_generation = 0u, token = 0u;
    ID3D11Buffer *current_constants = nullptr;
    ID3D11ShaderResourceView *ocr = nullptr, *sct = nullptr, *condition = nullptr, *root = nullptr;
    UINT postprocess_mode = 0u, expected_work = 0u;
    // Pass the authenticated contour producer's dense_main and its actual t0/t5/t12.
    ID3D11ComputeShader *legacy_dense_shader = nullptr;
    ID3D11ShaderResourceView *legacy_rgb = nullptr, *legacy_support = nullptr, *legacy_observation = nullptr;
    ID3D11ShaderResourceView *uip = nullptr; // Actual current UIP40, before the source envelope.
    bool use_empty_gate = false;
  };

  struct views_t {
    ComPtr<ID3D11ShaderResourceView> matte, receipt, alpha8;
    // The Direct Glyph Work header and line records, direct_work_word_count words.
    ComPtr<ID3D11Buffer> work;
  };

  // The compute slots the publication passes bind, restored around each queue.
  using compute_state_t = host_sbs_compute_state::guard_t<25u, 8u, 3u>;

  class resources_t {
    struct texture_t {
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    struct buffer_t {
      ComPtr<ID3D11Buffer> data;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    using srvs_t = std::array<ID3D11ShaderResourceView *, 25u>;
    using uavs_t = std::array<ID3D11UnorderedAccessView *, 8u>;

    ComPtr<ID3D11Device> device_;
    std::array<ComPtr<ID3D11ComputeShader>, shader_cache::direct_glyph_plan_specs.size()> plan_shaders_;
    ComPtr<ID3D11ComputeShader> empty_gate_shader_;
    buffer_t request_, work_, evidence_, receipt_, publish_args_, legacy_args_;
    buffer_t producer_schedule_, empty_witness_;
    texture_t matte_, alpha8_;
    UINT width_ = 0u, height_ = 0u;
    bool published_ = false;
    bool empty_gate_pending_ = false;
    bool direct_pending_ = false;

    static std::array<UINT, 12u> early_parameters(const request_t &r, UINT third, UINT fourth) {
      const auto lo = [](std::uint64_t v) { return static_cast<UINT>(v); };
      const auto hi = [](std::uint64_t v) { return static_cast<UINT>(v >> 32u); };
      return {r.width, r.height, third, fourth,
              lo(r.frame), hi(r.frame), lo(r.analysis_generation), hi(r.analysis_generation),
              lo(r.observation_us), hi(r.observation_us), lo(r.token), hi(r.token)};
    }

    static bool fail(std::string &error, const char *message) {
      error = message;
      return false;
    }

    bool same_device(ID3D11DeviceChild *object) const {
      if (!object) return false;
      ComPtr<ID3D11Device> owner;
      object->GetDevice(&owner);
      return owner.Get() == device_.Get();
    }

    bool matches_texture(ID3D11ShaderResourceView *view, DXGI_FORMAT format, UINT w, UINT h) const {
      if (!same_device(view)) return false;
      D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
      view->GetDesc(&srv);
      if (srv.Format != format || srv.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D ||
          srv.Texture2D.MostDetailedMip != 0u || srv.Texture2D.MipLevels != 1u) return false;
      ComPtr<ID3D11Resource> resource;
      ComPtr<ID3D11Texture2D> texture;
      view->GetResource(&resource);
      if (FAILED(resource.As(&texture))) return false;
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      return desc.Format == format && desc.Width == w && desc.Height == h &&
             desc.MipLevels == 1u && desc.ArraySize == 1u && desc.SampleDesc.Count == 1u;
    }

    bool matches_structured(ID3D11ShaderResourceView *view, UINT elements, UINT stride = 4u) const {
      if (!same_device(view)) return false;
      D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
      view->GetDesc(&srv);
      if (srv.Format != DXGI_FORMAT_UNKNOWN || srv.ViewDimension != D3D11_SRV_DIMENSION_BUFFER ||
          srv.Buffer.FirstElement != 0u || srv.Buffer.NumElements != elements) return false;
      ComPtr<ID3D11Resource> resource;
      ComPtr<ID3D11Buffer> buffer;
      view->GetResource(&resource);
      if (FAILED(resource.As(&buffer))) return false;
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      return desc.MiscFlags == D3D11_RESOURCE_MISC_BUFFER_STRUCTURED &&
             desc.StructureByteStride == stride && desc.ByteWidth == elements * stride;
    }

    bool matches_raw(ID3D11ShaderResourceView *view, UINT bytes) const {
      if (!same_device(view)) return false;
      D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
      view->GetDesc(&srv);
      if (srv.Format != DXGI_FORMAT_R32_TYPELESS || srv.ViewDimension != D3D11_SRV_DIMENSION_BUFFEREX ||
          srv.BufferEx.FirstElement != 0u || srv.BufferEx.NumElements != bytes / 4u ||
          srv.BufferEx.Flags != D3D11_BUFFEREX_SRV_FLAG_RAW) return false;
      ComPtr<ID3D11Resource> resource;
      ComPtr<ID3D11Buffer> buffer;
      view->GetResource(&resource);
      if (FAILED(resource.As(&buffer))) return false;
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      return desc.ByteWidth == bytes && (desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS) != 0u;
    }

    bool matches_c24(ID3D11Buffer *buffer) const {
      if (!same_device(buffer)) return false;
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      return desc.ByteWidth == 96u && (desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) != 0u;
    }

    bool matches_lazy_legacy(const request_t &r) const {
      constexpr UINT observation_records = 1u + models::depth_coordinate_v2::subtitle_plane_cover_capacity *
        (4u + models::depth_coordinate_v2::subtitle_contour_palette_max_modes);
      return matches_c24(r.current_constants) && matches_structured(r.sct, 32u) &&
             same_device(r.legacy_dense_shader) &&
             matches_texture(r.legacy_rgb, DXGI_FORMAT_R16G16B16A16_FLOAT, r.width, r.height) &&
             matches_texture(r.legacy_support, DXGI_FORMAT_R32_UINT, r.width, r.height) &&
             matches_structured(r.legacy_observation, observation_records, 16u);
    }

    bool make_texture(texture_t &target, UINT width, UINT height) const {
      const D3D11_TEXTURE2D_DESC desc {
        width, height, 1u, 1u, DXGI_FORMAT_R32_UINT, {1u, 0u}, D3D11_USAGE_DEFAULT,
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0u, 0u
      };
      ComPtr<ID3D11Texture2D> texture;
      return SUCCEEDED(device_->CreateTexture2D(&desc, nullptr, &texture)) &&
             SUCCEEDED(device_->CreateShaderResourceView(texture.Get(), nullptr, &target.srv)) &&
             SUCCEEDED(device_->CreateUnorderedAccessView(texture.Get(), nullptr, &target.uav));
    }

    bool make_buffer(buffer_t &target, UINT bytes, UINT binds, UINT misc,
                     const void *initial = nullptr) const {
      const bool structured = (misc & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) != 0u;
      const bool raw = (misc & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS) != 0u;
      const bool immutable = initial && (binds & D3D11_BIND_UNORDERED_ACCESS) == 0u;
      const D3D11_BUFFER_DESC desc {
        bytes, immutable ? D3D11_USAGE_IMMUTABLE : D3D11_USAGE_DEFAULT,
        binds, 0u, misc, structured ? 4u : 0u
      };
      const D3D11_SUBRESOURCE_DATA data {initial, 0u, 0u};
      if (FAILED(device_->CreateBuffer(&desc, initial ? &data : nullptr, &target.data))) return false;
      if ((binds & D3D11_BIND_SHADER_RESOURCE) != 0u) {
        D3D11_SHADER_RESOURCE_VIEW_DESC view {};
        view.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
        view.ViewDimension = raw ? D3D11_SRV_DIMENSION_BUFFEREX : D3D11_SRV_DIMENSION_BUFFER;
        if (raw) {
          view.BufferEx.NumElements = bytes / 4u;
          view.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        } else {
          view.Buffer.NumElements = bytes / 4u;
        }
        if (FAILED(device_->CreateShaderResourceView(target.data.Get(), &view, &target.srv))) return false;
      }
      if ((binds & D3D11_BIND_UNORDERED_ACCESS) != 0u) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC view {};
        view.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
        view.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        view.Buffer.NumElements = bytes / 4u;
        view.Buffer.Flags = raw ? static_cast<UINT>(D3D11_BUFFER_UAV_FLAG_RAW) : 0u;
        if (FAILED(device_->CreateUnorderedAccessView(target.data.Get(), &view, &target.uav))) return false;
      }
      return true;
    }

    bool ensure(UINT width, UINT height) {
      if (width_ == width && height_ == height) return true;
      texture_t matte, alpha8;
      if (!make_texture(matte, width, height) || !make_texture(alpha8, width, height)) return false;
      matte_ = std::move(matte);
      alpha8_ = std::move(alpha8);
      width_ = width;
      height_ = height;
      return true;
    }

    static void unbind(ID3D11DeviceContext *context) {
      const srvs_t srvs {};
      const uavs_t uavs {};
      context->CSSetShaderResources(0u, UINT(srvs.size()), srvs.data());
      context->CSSetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data(), nullptr);
    }

    void run_plan(ID3D11DeviceContext *context, UINT entry, const srvs_t &srvs, const uavs_t &uavs) const {
      context->CSSetShader(plan_shaders_[entry].Get(), nullptr, 0u);
      context->CSSetShaderResources(0u, UINT(srvs.size()), srvs.data());
      context->CSSetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data(), nullptr);
      if (entry == 0u) {
        context->Dispatch(1u, 1u, 1u);
      } else {
        context->DispatchIndirect(publish_args_.data.Get(), 0u);
      }
      unbind(context);
    }

    void run_legacy_dense(ID3D11DeviceContext *context, const request_t &r) const {
      srvs_t inputs {};
      inputs[0u] = r.legacy_rgb;
      inputs[5u] = r.legacy_support;
      inputs[12u] = r.legacy_observation;
      ID3D11Buffer *const constants[2] {r.current_constants, nullptr};
      auto *output = matte_.uav.Get();
      context->CSSetShader(r.legacy_dense_shader, nullptr, 0u);
      context->CSSetConstantBuffers(0u, 2u, constants);
      context->CSSetShaderResources(0u, UINT(inputs.size()), inputs.data());
      context->CSSetUnorderedAccessViews(6u, 1u, &output, nullptr);
      context->DispatchIndirect(legacy_args_.data.Get(), 0u);
      unbind(context);
    }

  public:
    bool load_direct(ID3D11Device *device, const std::filesystem::path &shader_root, std::string &error,
                     bool resolved_fringe) {
      error.clear();
      static_assert(std::endian::native == std::endian::little);
      if (!device) return fail(error, "Direct glyph publication device unavailable");
      resources_t next;
      next.device_ = device;
      const UINT su = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      const auto plan_snapshot = shader_cache::snapshot_sources(shader_root, shader_cache::direct_glyph_plan_specs);
      if (!plan_snapshot ||
          shader_cache::source_closure_sha256(plan_snapshot) != shader_cache::direct_glyph_plan_source_closure_sha256)
        return fail(error, "Direct glyph plan shader package unavailable");
      for (UINT i = 0u; i < shader_cache::direct_glyph_plan_specs.size(); ++i) {
        const auto code = shader_cache::get(plan_snapshot, shader_cache::direct_glyph_plan_specs[i]);
        if (!code || FAILED(device->CreateComputeShader(code->data(), code->size(), nullptr, &next.plan_shaders_[i])))
          return fail(error, "Direct glyph plan shader package unavailable");
      }
      auto gate_snapshot = shader_cache::snapshot_sources(shader_root, empty_gate_shader_specs);
      if (!gate_snapshot || shader_cache::source_closure_sha256(gate_snapshot) != empty_gate_shader_source_closure_sha256)
        return fail(error, "Subtitle empty gate shader package identity differs");
      const auto gate = shader_cache::get(gate_snapshot, empty_gate_shader_specs[0]);
      if (!gate || FAILED(device->CreateComputeShader(gate->data(), gate->size(), nullptr, &next.empty_gate_shader_)))
        return fail(error, "Subtitle empty gate shader creation failed");
      const UINT indirect = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      if (!next.make_buffer(next.publish_args_, 16u, D3D11_BIND_UNORDERED_ACCESS, indirect) ||
          !next.make_buffer(next.legacy_args_, 16u, D3D11_BIND_UNORDERED_ACCESS, indirect) ||
          !next.make_buffer(next.producer_schedule_, 256u, D3D11_BIND_UNORDERED_ACCESS, indirect) ||
          !next.make_buffer(next.empty_witness_, 128u, su, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED))
        return fail(error, "Direct glyph publication arguments allocation failed");
      // A first Root0 has no previously published current receipt to hold.
      constexpr std::array<UINT, 64u> unavailable_receipt {0x31534d53u, 1u, 1u, 1u};
      if (!next.make_buffer(next.request_, 48u, D3D11_BIND_CONSTANT_BUFFER, 0u) ||
          !next.make_buffer(next.work_, direct_work_word_count * 4u, su,
                            D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) ||
          (resolved_fringe && !next.make_buffer(next.evidence_, direct_fringe_request_bytes, su,
                                               D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)) ||
          !next.make_buffer(next.receipt_, 256u, su, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, unavailable_receipt.data()))
        return fail(error, "Direct glyph publication buffer allocation failed");
      *this = std::move(next);
      return true;
    }

    // Submit the frame's empty classifier before planning and return its private scheduling twin
    // at the usual Root offsets. Keep r.root as the actual Root for plan_direct. The twin and
    // witness belong to this serialized observation through the next plan_direct call.
    [[nodiscard]] ID3D11Buffer *prepare_empty_gate(ID3D11DeviceContext *context, const request_t &r, std::string &error) {
      error.clear();
      empty_gate_pending_ = false;
      if (!r.use_empty_gate || !empty_gate_shader_ || !same_device(context) ||
          context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !matches_raw(r.root, 256u) ||
          !r.width || !r.height || r.width > 5120u || r.height > 5120u) {
        fail(error, "Subtitle empty gate requires a ready same-device immediate context, actual Root and native extent");
        return nullptr;
      }

      const auto parameters = early_parameters(r, r.postprocess_mode, r.expected_work);
      const compute_state_t saved(context);
      unbind(context);
      context->UpdateSubresource(request_.data.Get(), 0u, nullptr, parameters.data(), 0u, 0u);
      ID3D11Buffer *const constants[3] {matches_c24(r.current_constants) ? r.current_constants : nullptr,
                                       nullptr, request_.data.Get()};
      ID3D11ShaderResourceView *const inputs[5] {
        r.root, matches_structured(r.ocr, 256u) ? r.ocr : nullptr,
        matches_structured(r.sct, 32u) ? r.sct : nullptr,
        matches_structured(r.condition, 6u) ? r.condition : nullptr,
        matches_structured(r.uip, 40u) ? r.uip : nullptr
      };
      ID3D11UnorderedAccessView *const outputs[2] {producer_schedule_.uav.Get(), empty_witness_.uav.Get()};
      context->CSSetShader(empty_gate_shader_.Get(), nullptr, 0u);
      context->CSSetConstantBuffers(0u, 3u, constants);
      context->CSSetShaderResources(0u, 5u, inputs);
      context->CSSetUnorderedAccessViews(0u, 2u, outputs, nullptr);
      context->Dispatch(1u, 1u, 1u);
      unbind(context);
      empty_gate_pending_ = true;
      return producer_schedule_.data.Get();
    }

    // Opens one observation. Keep its inputs and Work bound to that owner until publish_direct.
    [[nodiscard]] ID3D11Buffer *plan_direct(
      ID3D11DeviceContext *context, const request_t &r,
      const models::display_snapshot_rgb8_analysis_gpu_v1::direct_audit_submission_t *audit,
      bool adaptive_context, std::string &error, bool detector_caps = false
    ) {
      error.clear();
      const bool gate_queued = std::exchange(empty_gate_pending_, false);
      if (!plan_shaders_[0] || direct_pending_ || !same_device(context) ||
          context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !matches_raw(r.root, 256u) ||
          !r.width || !r.height || r.width > 5120u || r.height > 5120u || !ensure(r.width, r.height)) {
        fail(error, "Direct glyph planning requires a current same-device Root and native extent");
        return nullptr;
      }
      published_ = false;
      const bool lazy_legacy = matches_lazy_legacy(r);
      const UINT flags = (lazy_legacy ? 4u : 0u) | (r.use_empty_gate && gate_queued ? 8u : 0u) |
                         (adaptive_context ? 16u : 0u) |
                         (detector_caps ? 64u : (evidence_.uav ? direct_fringe_flag : 0u)) |
                         (r.postprocess_mode << 8u) | (r.expected_work << 16u);
      const auto parameters = early_parameters(r, cutoff_alpha8, flags);
      const compute_state_t saved(context);
      unbind(context);
      context->UpdateSubresource(request_.data.Get(), 0u, nullptr, parameters.data(), 0u, 0u);
      ID3D11Buffer *const constants[2] {request_.data.Get(), matches_c24(r.current_constants) ? r.current_constants : nullptr};
      context->CSSetConstantBuffers(0u, 2u, constants);
      if (!lazy_legacy) {
        constexpr std::array<UINT, 4u> zero {};
        context->ClearUnorderedAccessViewUint(legacy_args_.uav.Get(), zero.data());
      }

      srvs_t inputs {};
      inputs[12u] = matches_structured(r.ocr, 256u) ? r.ocr : nullptr;
      inputs[13u] = matches_structured(r.sct, 32u) ? r.sct : nullptr;
      inputs[14u] = matches_structured(r.condition, 6u) ? r.condition : nullptr;
      inputs[16u] = r.root;
      auto *envelope = audit ? audit->uip_envelope() : nullptr;
      inputs[17u] = matches_structured(envelope, models::display_snapshot_rgb8_analysis_gpu_v1::envelope_words) ? envelope : nullptr;
      inputs[21u] = r.use_empty_gate && gate_queued ? empty_witness_.srv.Get() : nullptr;
      run_plan(context, 0u, inputs, {nullptr, legacy_args_.uav.Get(), work_.uav.Get(), nullptr,
                                     nullptr, nullptr, evidence_.uav.Get(), publish_args_.uav.Get()});
      direct_pending_ = true;
      return work_.data.Get();
    }

    [[nodiscard]] ID3D11ShaderResourceView *direct_detector_caps() const noexcept {
      return direct_pending_ ? evidence_.srv.Get() : nullptr;
    }

    bool publish_direct(ID3D11DeviceContext *context, const request_t &r, ID3D11ShaderResourceView *alpha8,
                        ID3D11ShaderResourceView *completed_placement) {
      if (!direct_pending_ || !same_device(context)) return false;
      direct_pending_ = false;
      const compute_state_t saved(context);
      unbind(context);
      const bool alpha8_available = matches_texture(alpha8, DXGI_FORMAT_R32_UINT, r.width, r.height);
      if (!alpha8_available) {
        // An attested Work header cannot authenticate a missing score plane.
        constexpr D3D11_BOX score_ready_word {12u, 0u, 0u, 16u, 1u, 1u};
        constexpr UINT unavailable = 0u;
        context->UpdateSubresource(work_.data.Get(), 0u, &score_ready_word, &unavailable, 0u, 0u);
      }
      ID3D11Buffer *const constants[2] {request_.data.Get(), matches_c24(r.current_constants) ? r.current_constants : nullptr};
      context->CSSetConstantBuffers(0u, 2u, constants);
      srvs_t inputs {};
      inputs[7u] = work_.srv.Get();
      inputs[13u] = matches_structured(r.sct, 32u) ? r.sct : nullptr;
      inputs[14u] = matches_structured(r.condition, 6u) ? r.condition : nullptr;
      inputs[15u] = alpha8_available ? alpha8 : nullptr;
      inputs[16u] = r.root;
      if (evidence_.srv && alpha8_available &&
          matches_structured(completed_placement, models::host_sbs_direct_glyph_gpu::placement_words)) {
        inputs[22u] = evidence_.srv.Get();
        inputs[23u] = completed_placement;
      }
      run_plan(context, 1u, inputs, {matte_.uav.Get(), legacy_args_.uav.Get(), nullptr, alpha8_.uav.Get(),
                                     nullptr, receipt_.uav.Get()});
      if (matches_lazy_legacy(r)) run_legacy_dense(context, r);
      published_ = true;
      return true;
    }

    [[nodiscard]] views_t views() const {
      if (!published_) return {};
      return {matte_.srv, receipt_.srv, alpha8_.srv, work_.data};
    }

    [[nodiscard]] ID3D11Buffer *direct_legacy_dispatch_for_evaluation() const noexcept {
      return published_ ? legacy_args_.data.Get() : nullptr;
    }
  };
}
#endif
