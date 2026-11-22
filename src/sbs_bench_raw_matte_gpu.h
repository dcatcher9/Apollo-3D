/** @file Host SBS subtitle GPU inference and selected-matte publication. */
#pragma once

#ifdef _WIN32
  #include "generated/depth_coordinate_v2_contract.h"
  #include "host_sbs_direct_glyph_tiles.h"
  #include "host_sbs_display_snapshot_rgb8_analysis_gpu.h"
  #include "host_sbs_shader_cache.h"
  #include "host_sbs_subtitle_half_context_weights.h"
  #include "sbs_bench_raw_matte_weights.h"

  #include <array>
  #include <bit>
  #include <cstdint>
  #include <d3d11.h>
  #include <filesystem>
  #include <string>
  #include <string_view>
  #include <utility>
  #include <wrl/client.h>

namespace sbs_bench::raw_matte_gpu {
  using Microsoft::WRL::ComPtr;
  namespace shader_cache = models::host_sbs_shader_cache;

  inline constexpr UINT cutoff_alpha8 = 38u;
  inline constexpr UINT work_word_count = 60u;
  inline constexpr UINT direct_work_word_count = 36u;
  inline constexpr UINT direct_fringe_request_bytes = 144u;
  inline constexpr UINT direct_fringe_flag = 32u;
  inline constexpr std::string_view shader_source_closure_sha256 =
    "b9f09c1f413cdbdf8368f7d21d23b2b51ccad06d0707e49f444eecf9e39705fd";
  inline constexpr std::array<shader_cache::shader_spec, 4u> shader_specs {{
    {"host_sbs_bench_raw_matte_cs.hlsl", "prepare_main", "cs_5_0"},
    {"host_sbs_bench_raw_matte_cs.hlsl", "features_main", "cs_5_0"},
    {"host_sbs_bench_raw_matte_cs.hlsl", "score_main", "cs_5_0"},
    {"host_sbs_bench_raw_matte_cs.hlsl", "publish_main", "cs_5_0"},
  }};
  inline constexpr std::string_view half_context_shader_source_closure_sha256 =
    "53653ad9b23a7f787c3de5c655f0176c5014ba5f09fcd278257bb3b5b12e4a99";
  inline constexpr std::array<shader_cache::shader_spec, 5u> half_context_shader_specs {{
    {"host_sbs_subtitle_half_context_cs.hlsl", "init_domain", "cs_5_0"},
    {"host_sbs_subtitle_half_context_cs.hlsl", "geometry", "cs_5_0"},
    {"host_sbs_subtitle_half_context_cs.hlsl", "extent", "cs_5_0"},
    {"host_sbs_subtitle_half_context_cs.hlsl", "trunk", "cs_5_0"},
    {"host_sbs_subtitle_half_context_cs.hlsl", "finish", "cs_5_0"},
  }};
  inline constexpr std::string_view early_shader_source_closure_sha256 =
    "138a61e967c36b87f1c9f4c51b061fa387c3524d248ea7c566bf853129b6ce04";
  inline constexpr std::string_view empty_gate_shader_source_closure_sha256 =
    "f4e4ce4d14378bca175db941f5b216ebfc806dca8effaa2007f016a7df9ffdb2";
  inline constexpr std::array<shader_cache::shader_spec, 2u> early_shader_specs {{
    {"host_sbs_raw_matte_early_cs.hlsl", "prepare_early_main", "cs_5_0"},
    {"host_sbs_raw_matte_early_cs.hlsl", "publish_early_main", "cs_5_0"},
  }};
  inline constexpr std::array<shader_cache::shader_spec, 1u> empty_gate_shader_specs {{
    {"host_sbs_raw_matte_empty_gate_cs.hlsl", "prepare_empty_main", "cs_5_0"},
  }};

  // The caller provides the serialized, GPU-completed composition and actual renderer t3/t7/SCT
  // owners. The probe's compute validates their current receipts and never maps production inputs.
  struct request_t {
    UINT width = 0u, height = 0u;
    std::uint64_t frame = 0u, observation_us = 0u, analysis_generation = 0u, token = 0u;
    ID3D11Buffer *current_constants = nullptr;
    ID3D11ShaderResourceView *rgb = nullptr, *cover = nullptr, *body = nullptr, *selected = nullptr, *raw = nullptr;
    ID3D11ShaderResourceView *facts = nullptr, *trusted = nullptr, *rule = nullptr, *rule_receipt = nullptr;
    ID3D11ShaderResourceView *ocr = nullptr, *sct = nullptr, *condition = nullptr;
  };

  // Bind the literal outputs of the reduced Source5/CBC5/RawM/Inside producer.
  // All views must belong to the same ordered observation as the actual C24 and Root.
  struct early_request_t {
    UINT width = 0u, height = 0u;
    std::uint64_t frame = 0u, observation_us = 0u, analysis_generation = 0u, token = 0u;
    ID3D11Buffer *current_constants = nullptr;
    ID3D11ShaderResourceView *rgb = nullptr, *cover = nullptr, *body = nullptr, *selected = nullptr, *raw = nullptr;
    ID3D11ShaderResourceView *floor = nullptr, *ocr = nullptr, *sct = nullptr, *condition = nullptr;
    ID3D11ShaderResourceView *root = nullptr, *source_envelope = nullptr, *cbc_meta = nullptr, *raw_meta = nullptr,
                             *floor_data = nullptr;
    ID3D11ShaderResourceView *legacy = nullptr, *legacy_receipt = nullptr;
    UINT postprocess_mode = 0u, expected_work = 0u;
    // Pass the authenticated contour producer's dense_main and its actual t0/t5/t12.
    // A lazy request does not also provide an eagerly materialized legacy pair.
    ID3D11ComputeShader *legacy_dense_shader = nullptr;
    ID3D11ShaderResourceView *legacy_rgb = nullptr, *legacy_support = nullptr, *legacy_observation = nullptr;
    ID3D11ShaderResourceView *uip = nullptr; // Actual current UIP40, before the Source5 envelope.
    bool use_empty_gate = false;
  };

  struct views_t {
    ComPtr<ID3D11ShaderResourceView> matte, receipt, alpha8;
    // [0..15] two indirect arguments/rectangles, [16..19] disposition, slots, reason, cutoff,
    // [20..51] current covers, [52] current legacy available (early only), [53] planned
    // disposition, [54..56] actual groups and fault bits, [57..59] lazy legacy dispatch dimensions.
    // Dispositions: 0 fallback/unavailable,
    // 1 exact empty, 2 active, 3 held previous publication (early Root0 only).
    // If disposition is 1, work[4] distinguishes fast empty (1) from ordinary empty (0).
    ComPtr<ID3D11Buffer> work;
  };

  class resources_t {
    enum class route_e { benchmark, half_context, direct };
    struct texture_t {
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    struct buffer_t {
      ComPtr<ID3D11Buffer> data;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    using shaders_t = std::array<ComPtr<ID3D11ComputeShader>, shader_specs.size()>;
    using srvs_t = std::array<ID3D11ShaderResourceView *, 25u>;
    using uavs_t = std::array<ID3D11UnorderedAccessView *, 8u>;

    struct saved_cs_t {
      ID3D11DeviceContext *context;
      srvs_t srvs {};
      uavs_t uavs {};
      std::array<ID3D11Buffer *, 3u> constants {};
      std::array<ID3D11ClassInstance *, 256u> instances {};
      ID3D11ComputeShader *shader = nullptr;
      ID3D11Predicate *predicate = nullptr;
      BOOL predicate_value = FALSE;
      UINT instance_count = instances.size();

      explicit saved_cs_t(ID3D11DeviceContext *c): context(c) {
        context->CSGetShaderResources(0u, UINT(srvs.size()), srvs.data());
        context->CSGetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data());
        context->CSGetConstantBuffers(0u, UINT(constants.size()), constants.data());
        context->CSGetShader(&shader, instances.data(), &instance_count);
        context->GetPredication(&predicate, &predicate_value);
        context->SetPredication(nullptr, FALSE);
      }
      saved_cs_t(const saved_cs_t &) = delete;
      saved_cs_t &operator=(const saved_cs_t &) = delete;

      ~saved_cs_t() {
        context->CSSetShaderResources(0u, UINT(srvs.size()), srvs.data());
        context->CSSetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data(), nullptr);
        context->CSSetConstantBuffers(0u, UINT(constants.size()), constants.data());
        context->CSSetShader(shader, instance_count == 0u ? nullptr : instances.data(), instance_count);
        context->SetPredication(predicate, predicate_value);
        for (auto *p : srvs) if (p) p->Release();
        for (auto *p : uavs) if (p) p->Release();
        for (auto *p : constants) if (p) p->Release();
        for (UINT i = 0u; i < instance_count; ++i) if (instances[i]) instances[i]->Release();
        if (shader) shader->Release();
        if (predicate) predicate->Release();
      }
    };

    ComPtr<ID3D11Device> device_;
    shaders_t shaders_;
    std::array<ComPtr<ID3D11ComputeShader>, half_context_shader_specs.size()> half_context_shaders_;
    std::array<ComPtr<ID3D11ComputeShader>, early_shader_specs.size()> early_shaders_;
    ComPtr<ID3D11ComputeShader> empty_gate_shader_;
    buffer_t weights_, request_, work_, evidence_, receipt_, publish_args_, legacy_args_;
    buffer_t producer_schedule_, empty_witness_;
    buffer_t model_domain_, model_geometry_args_, model_activations_;
    texture_t geometry_, candidate_, matte_, alpha8_, axial_;
    UINT width_ = 0u, height_ = 0u;
    route_e route_ = route_e::benchmark;
    bool published_ = false;
    bool empty_gate_pending_ = false;
    bool direct_pending_ = false;

    static std::array<UINT, 12u> early_parameters(const early_request_t &r, UINT third, UINT fourth) {
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

    bool matches_lazy_legacy(const early_request_t &r) const {
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
      texture_t geometry, candidate, matte, alpha8;
      if (!make_texture(matte, width, height) || !make_texture(alpha8, width, height) ||
          (route_ != route_e::direct && (!make_texture(geometry, width, height) ||
                                        !make_texture(candidate, width, height)))) return false;
      geometry_ = std::move(geometry);
      candidate_ = std::move(candidate);
      matte_ = std::move(matte);
      alpha8_ = std::move(alpha8);
      axial_ = {};
      width_ = width;
      height_ = height;
      return true;
    }

    void run_half_context(ID3D11DeviceContext *context, const early_request_t &r) const {
      const auto dispatch = [&](UINT entry, const srvs_t &inputs, const uavs_t &outputs,
                                ID3D11Buffer *args = nullptr, UINT offset = 0u) {
        context->CSSetShader(half_context_shaders_[entry].Get(), nullptr, 0u);
        context->CSSetShaderResources(0u, UINT(inputs.size()), inputs.data());
        context->CSSetUnorderedAccessViews(0u, UINT(outputs.size()), outputs.data(), nullptr);
        if (args) context->DispatchIndirect(args, offset);
        else context->Dispatch(1u, 1u, 1u);
        unbind(context);
      };
      auto *evidence = evidence_.uav.Get();
      auto *domain = model_domain_.uav.Get();
      auto *geometry_args = model_geometry_args_.data.Get();
      auto *domain_args = model_domain_.data.Get();
      srvs_t inputs {r.rgb, nullptr, r.cover, r.body, r.selected, r.raw, weights_.srv.Get(), work_.srv.Get(),
                     nullptr, r.floor};
      dispatch(0u, inputs, {nullptr, nullptr, domain, nullptr, nullptr, nullptr, nullptr, model_geometry_args_.uav.Get()});
      dispatch(1u, inputs, {candidate_.uav.Get(), axial_.uav.Get(), domain, nullptr, geometry_.uav.Get(), nullptr, evidence},
               geometry_args);
      dispatch(2u, {}, {nullptr, nullptr, domain, nullptr, nullptr, nullptr, evidence, work_.uav.Get()});
      inputs[8u] = geometry_.srv.Get();
      inputs[22u] = axial_.srv.Get();
      inputs[24u] = model_domain_.srv.Get();
      dispatch(3u, inputs, {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, evidence, model_activations_.uav.Get()},
               domain_args, 16u);
      inputs[23u] = model_activations_.srv.Get();
      dispatch(4u, inputs, {candidate_.uav.Get(), nullptr, nullptr, alpha8_.uav.Get(), nullptr, nullptr, evidence},
               domain_args, 16u);
    }

    static void unbind(ID3D11DeviceContext *context) {
      const srvs_t srvs {};
      const uavs_t uavs {};
      context->CSSetShaderResources(0u, UINT(srvs.size()), srvs.data());
      context->CSSetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data(), nullptr);
    }

    void run(ID3D11DeviceContext *context, UINT entry, const srvs_t &srvs, const uavs_t &uavs,
             UINT indirect_offset) const {
      context->CSSetShader(shaders_[entry].Get(), nullptr, 0u);
      context->CSSetShaderResources(0u, UINT(srvs.size()), srvs.data());
      context->CSSetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data(), nullptr);
      if (entry == 1u || entry == 2u) {
        context->DispatchIndirect(work_.data.Get(), indirect_offset);
      } else if (entry == 0u) {
        context->Dispatch(1u, 1u, 1u);
      } else {
        context->Dispatch((width_ + 15u) / 16u, (height_ + 7u) / 8u, 1u);
      }
      unbind(context);
    }

    void run_early(ID3D11DeviceContext *context, UINT entry, const srvs_t &srvs, const uavs_t &uavs) const {
      context->CSSetShader(early_shaders_[entry].Get(), nullptr, 0u);
      context->CSSetShaderResources(0u, UINT(srvs.size()), srvs.data());
      context->CSSetUnorderedAccessViews(0u, UINT(uavs.size()), uavs.data(), nullptr);
      if (entry == 0u) {
        context->Dispatch(1u, 1u, 1u);
      } else {
        context->DispatchIndirect(publish_args_.data.Get(), 0u);
      }
      unbind(context);
    }

    void run_legacy_dense(ID3D11DeviceContext *context, const early_request_t &r) const {
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

    bool load_route(ID3D11Device *device, const std::filesystem::path &shader_root, std::string &error, route_e route,
                    bool resolved_fringe = false) {
      error.clear();
      static_assert(std::endian::native == std::endian::little);
      if (!device) return fail(error, "RawMatte device unavailable");
      resources_t next;
      next.device_ = device;
      next.route_ = route;
      const bool early = route != route_e::benchmark;
      const bool direct = route == route_e::direct;
      const UINT su = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      const auto load_group = [&](const auto &specs, std::string_view closure, auto &shaders) {
        const auto snapshot = shader_cache::snapshot_sources(shader_root, specs);
        if (!snapshot || shader_cache::source_closure_sha256(snapshot) != closure) return false;
        for (UINT i = 0u; i < specs.size(); ++i) {
          const auto code = shader_cache::get(snapshot, specs[i]);
          if (!code || FAILED(device->CreateComputeShader(code->data(), code->size(), nullptr, &shaders[i]))) return false;
        }
        return true;
      };
      if (early) {
        const auto &specs = direct ? shader_cache::direct_glyph_plan_specs : early_shader_specs;
        const auto closure = direct ? shader_cache::direct_glyph_plan_source_closure_sha256 : early_shader_source_closure_sha256;
        if (!load_group(specs, closure, next.early_shaders_))
          return fail(error, "RawMatte early shader package unavailable");
        auto gate_snapshot = shader_cache::snapshot_sources(shader_root, empty_gate_shader_specs);
        if (!gate_snapshot || shader_cache::source_closure_sha256(gate_snapshot) != empty_gate_shader_source_closure_sha256)
          return fail(error, "RawMatte empty gate shader package identity differs");
        const auto gate = shader_cache::get(gate_snapshot, empty_gate_shader_specs[0]);
        if (!gate || FAILED(device->CreateComputeShader(gate->data(), gate->size(), nullptr, &next.empty_gate_shader_)))
          return fail(error, "RawMatte empty gate shader creation failed");
        const UINT indirect = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
        // Keep in step with NativeCapacity and the twelve full-resolution channels in the shader.
        constexpr UINT activation_bytes = 98304u * 12u * sizeof(float);
        if (!direct &&
            (!load_group(half_context_shader_specs, half_context_shader_source_closure_sha256, next.half_context_shaders_) ||
             !next.make_buffer(next.model_activations_, activation_bytes, su, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) ||
             !next.make_buffer(next.model_domain_, 80u, su, indirect) ||
             !next.make_buffer(next.model_geometry_args_, 16u, D3D11_BIND_UNORDERED_ACCESS, indirect)))
          return fail(error, "Subtitle half-context scratch allocation failed");
        if (!next.make_buffer(next.publish_args_, 16u, D3D11_BIND_UNORDERED_ACCESS, indirect) ||
            !next.make_buffer(next.legacy_args_, 16u, D3D11_BIND_UNORDERED_ACCESS, indirect) ||
            !next.make_buffer(next.producer_schedule_, 256u, D3D11_BIND_UNORDERED_ACCESS, indirect) ||
            !next.make_buffer(next.empty_witness_, 128u, su, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED))
          return fail(error, "RawMatte early arguments allocation failed");
      } else if (!load_group(shader_specs, shader_source_closure_sha256, next.shaders_)) {
        return fail(error, "RawMatte benchmark shader package unavailable");
      }
      // A first Root0 has no previously published current receipt to hold.
      constexpr std::array<UINT, 64u> unavailable_receipt {0x31534d53u, 1u, 1u, 1u};
      const UINT weights_bytes = early ? sizeof(half_context_weight_words) : sizeof(weight_words);
      const auto *weights = early ? half_context_weight_words.data() : weight_words.data();
      if ((!direct && !next.make_buffer(next.weights_, weights_bytes, D3D11_BIND_SHADER_RESOURCE,
                                        D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, weights)) ||
          !next.make_buffer(next.request_, 48u, D3D11_BIND_CONSTANT_BUFFER, 0u) ||
          !next.make_buffer(next.work_, (direct ? direct_work_word_count : work_word_count) * 4u, su,
                            D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) ||
          (!direct && !next.make_buffer(next.evidence_, 16u, D3D11_BIND_UNORDERED_ACCESS,
                                        D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)) ||
          (resolved_fringe && !next.make_buffer(next.evidence_, direct_fringe_request_bytes, su,
                                               D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)) ||
          !next.make_buffer(next.receipt_, 256u, su, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, unavailable_receipt.data()))
        return fail(error, "RawMatte buffer allocation failed");
      *this = std::move(next);
      return true;
    }

  public:
    bool load(ID3D11Device *device, const std::filesystem::path &shader_root, std::string &error, bool early = false) {
      return load_route(device, shader_root, error, early ? route_e::half_context : route_e::benchmark);
    }

    bool load_direct(ID3D11Device *device, const std::filesystem::path &shader_root, std::string &error,
                     bool resolved_fringe) {
      return load_route(device, shader_root, error, route_e::direct, resolved_fringe);
    }

    bool queue(ID3D11DeviceContext *context, const request_t &r, std::string &error) {
      error.clear();
      published_ = false;
      empty_gate_pending_ = false;
      if (!shaders_[0] || !same_device(context) || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE ||
          !r.width || !r.height || r.width > 5120u || r.height > 5120u)
        return fail(error, "RawMatte requires a ready same-device immediate context and native extent");
      if (!matches_texture(r.rule, DXGI_FORMAT_R32_UINT, r.width, r.height) ||
          !matches_structured(r.rule_receipt, 64u))
        return fail(error, "RawMatte current Rule fallback pair has wrong storage");
      if (!ensure(r.width, r.height)) return fail(error, "RawMatte native plane allocation failed");
      bool valid = r.frame && r.observation_us && r.token && matches_c24(r.current_constants) &&
                   matches_texture(r.rgb, DXGI_FORMAT_R16G16B16A16_FLOAT, r.width, r.height) &&
                   matches_structured(r.ocr, 256u) && matches_structured(r.sct, 32u) &&
                   matches_structured(r.condition, 6u);
      for (auto *view : {r.cover, r.body, r.selected, r.raw, r.facts, r.trusted})
        valid = valid && matches_texture(view, DXGI_FORMAT_R32_UINT, r.width, r.height);

      const auto lo = [](std::uint64_t v) { return static_cast<UINT>(v); };
      const auto hi = [](std::uint64_t v) { return static_cast<UINT>(v >> 32u); };
      const std::array<UINT, 12u> parameters {
        r.width, r.height, cutoff_alpha8, valid ? 0u : 1u,
        lo(r.frame), hi(r.frame), lo(r.analysis_generation), hi(r.analysis_generation),
        lo(r.observation_us), hi(r.observation_us), lo(r.token), hi(r.token)
      };
      const saved_cs_t saved(context);
      unbind(context);
      context->UpdateSubresource(request_.data.Get(), 0u, nullptr, parameters.data(), 0u, 0u);
      ID3D11Buffer *const constants[2] {request_.data.Get(), valid ? r.current_constants : nullptr};
      context->CSSetConstantBuffers(0u, 2u, constants);
      constexpr UINT zero[4] {};
      context->ClearUnorderedAccessViewUint(alpha8_.uav.Get(), zero);

      srvs_t inputs {};
      inputs[10] = r.rule;
      inputs[11] = r.rule_receipt;
      if (valid) {
        inputs = {r.rgb, r.facts, r.cover, r.body, r.selected, r.raw, weights_.srv.Get(), nullptr,
                  nullptr, r.trusted, r.rule, r.rule_receipt, r.ocr, r.sct, r.condition};
      }
      run(context, 0u, inputs, {nullptr, nullptr, work_.uav.Get(), nullptr, nullptr, nullptr, evidence_.uav.Get()}, 0u);
      if (valid) {
        inputs[7] = work_.srv.Get();
        run(context, 1u, inputs, {nullptr, nullptr, nullptr, nullptr, geometry_.uav.Get(), nullptr, evidence_.uav.Get()}, 32u);
        inputs[8] = geometry_.srv.Get();
        run(context, 2u, inputs, {candidate_.uav.Get(), nullptr, nullptr, alpha8_.uav.Get(),
                                 nullptr, nullptr, evidence_.uav.Get()}, 0u);
      }
      srvs_t final_inputs {};
      final_inputs[2] = valid ? r.cover : nullptr;
      final_inputs[10] = r.rule;
      final_inputs[11] = r.rule_receipt;
      final_inputs[15] = candidate_.srv.Get();
      run(context, 3u, final_inputs, {matte_.uav.Get(), nullptr, work_.uav.Get(), alpha8_.uav.Get(),
                                     nullptr, receipt_.uav.Get(), evidence_.uav.Get()}, 0u);
      published_ = true;
      return true;
    }

    // Submit the frame's classifier before the reduced producer and return its private scheduling
    // twin at the usual Root offsets. Keep r.root as the actual Root for queue_early. The twin and
    // witness belong to this serialized observation through the next queue_early call.
    [[nodiscard]] ID3D11Buffer *prepare_empty_gate(ID3D11DeviceContext *context, const early_request_t &r, std::string &error) {
      error.clear();
      empty_gate_pending_ = false;
      if (!r.use_empty_gate || !empty_gate_shader_ || !same_device(context) ||
          context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !matches_raw(r.root, 256u) ||
          !r.width || !r.height || r.width > 5120u || r.height > 5120u) {
        fail(error, "RawMatte empty gate requires a ready same-device immediate context, actual Root and native extent");
        return nullptr;
      }

      const auto parameters = early_parameters(r, r.postprocess_mode, r.expected_work);
      const saved_cs_t saved(context);
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

    bool queue_early(ID3D11DeviceContext *context, const early_request_t &r, std::string &error) {
      error.clear();
      published_ = false;
      const bool gate_queued = std::exchange(empty_gate_pending_, false);
      if (route_ != route_e::half_context || !early_shaders_[0] || !same_device(context) ||
          context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE ||
          !r.width || !r.height || r.width > 5120u || r.height > 5120u || !matches_raw(r.root, 256u))
        return fail(error, "RawMatte early requires a ready same-device immediate context, actual Root and native extent");
      if (!ensure(r.width, r.height)) return fail(error, "RawMatte native plane allocation failed");

      const bool c24_ok = matches_c24(r.current_constants);
      const bool sct_ok = matches_structured(r.sct, 32u);
      const bool lazy_legacy = r.legacy_dense_shader || r.legacy_rgb || r.legacy_support || r.legacy_observation;
      if (lazy_legacy && (r.legacy || r.legacy_receipt))
        return fail(error, "RawMatte early requires one legacy fallback source");
      const bool legacy_ok = matches_texture(r.legacy, DXGI_FORMAT_R32_UINT, r.width, r.height) &&
                             matches_structured(r.legacy_receipt, 64u);
      const bool mode_ok = (r.postprocess_mode == 1u || r.postprocess_mode == 2u || r.postprocess_mode == 4u) &&
                           r.expected_work <= 2u;
      const bool lazy_ok = mode_ok && matches_lazy_legacy(r);
      bool valid = r.frame && r.observation_us && r.token && c24_ok && sct_ok && mode_ok &&
                   matches_texture(r.rgb, DXGI_FORMAT_R16G16B16A16_FLOAT, r.width, r.height) &&
                   matches_structured(r.ocr, 256u) && matches_structured(r.condition, 6u) &&
                   matches_structured(r.source_envelope, 104u) && matches_structured(r.cbc_meta, 512u) &&
                   matches_structured(r.raw_meta, 512u) && matches_raw(r.floor_data, 68608u * 4u);
      for (auto *view : {r.cover, r.body, r.selected, r.raw, r.floor})
        valid = valid && matches_texture(view, DXGI_FORMAT_R32_UINT, r.width, r.height);
      if (valid && !axial_.srv) {
        texture_t axial;
        valid = make_texture(axial, r.width, r.height);
        if (valid) axial_ = std::move(axial);
      }

      const bool empty_witness = r.use_empty_gate && gate_queued;
      const UINT mode = mode_ok ? r.postprocess_mode | (r.expected_work << 8u) : 0u;
      const UINT flags = (valid ? 0u : 1u) | (legacy_ok ? 2u : 0u) | (lazy_ok ? 4u : 0u) |
                        (empty_witness ? 8u : 0u) | (mode << 8u);
      const auto parameters = early_parameters(r, cutoff_alpha8, flags);
      const saved_cs_t saved(context);
      unbind(context);
      context->UpdateSubresource(request_.data.Get(), 0u, nullptr, parameters.data(), 0u, 0u);
      ID3D11Buffer *const constants[2] {request_.data.Get(), c24_ok ? r.current_constants : nullptr};
      context->CSSetConstantBuffers(0u, 2u, constants);

      srvs_t inputs {};
      inputs[11] = legacy_ok ? r.legacy_receipt : nullptr;
      inputs[13] = sct_ok ? r.sct : nullptr;
      inputs[16] = r.root;
      inputs[21] = empty_witness ? empty_witness_.srv.Get() : nullptr;
      if (valid) {
        inputs[12] = r.ocr;
        inputs[14] = r.condition;
        inputs[17] = r.source_envelope;
        inputs[18] = r.cbc_meta;
        inputs[19] = r.raw_meta;
        inputs[20] = r.floor_data;
      }
      auto *legacy_arguments = lazy_ok ? legacy_args_.uav.Get() : nullptr;
      run_early(context, 0u, inputs, {nullptr, legacy_arguments, work_.uav.Get(), nullptr, nullptr, nullptr,
                                     evidence_.uav.Get(), publish_args_.uav.Get()});
      if (valid) run_half_context(context, r);
      srvs_t final_inputs {};
      final_inputs[2] = valid ? r.cover : nullptr;
      final_inputs[8] = geometry_.srv.Get();
      final_inputs[9] = valid ? r.floor : nullptr;
      final_inputs[10] = legacy_ok ? r.legacy : nullptr;
      final_inputs[11] = legacy_ok ? r.legacy_receipt : nullptr;
      final_inputs[13] = sct_ok ? r.sct : nullptr;
      final_inputs[15] = candidate_.srv.Get();
      run_early(context, 1u, final_inputs, {matte_.uav.Get(), legacy_arguments, work_.uav.Get(), alpha8_.uav.Get(),
                                       nullptr, receipt_.uav.Get(), evidence_.uav.Get()});
      // This is ordered before every consumer of the selected pair. Zero arguments make the
      // original legacy shader do no work on a held, empty or accepted model observation.
      if (lazy_ok) run_legacy_dense(context, r);
      published_ = true;
      return true;
    }

    // Opens one observation. Keep its inputs and Work bound to that owner until publish_direct.
    [[nodiscard]] ID3D11Buffer *plan_direct(
      ID3D11DeviceContext *context, const early_request_t &r,
      const models::display_snapshot_rgb8_analysis_gpu_v1::direct_audit_submission_t *audit,
      bool adaptive_context, std::string &error, bool detector_caps = false
    ) {
      error.clear();
      const bool gate_queued = std::exchange(empty_gate_pending_, false);
      if (route_ != route_e::direct || direct_pending_ || !same_device(context) ||
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
      const saved_cs_t saved(context);
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
      run_early(context, 0u, inputs, {nullptr, legacy_args_.uav.Get(), work_.uav.Get(), nullptr,
                                     nullptr, nullptr, evidence_.uav.Get(), publish_args_.uav.Get()});
      direct_pending_ = true;
      return work_.data.Get();
    }

    [[nodiscard]] ID3D11ShaderResourceView *direct_detector_caps() const noexcept {
      return direct_pending_ ? evidence_.srv.Get() : nullptr;
    }

    bool publish_direct(ID3D11DeviceContext *context, const early_request_t &r, ID3D11ShaderResourceView *alpha8,
                        ID3D11ShaderResourceView *completed_placement) {
      if (!direct_pending_ || !same_device(context)) return false;
      direct_pending_ = false;
      const saved_cs_t saved(context);
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
      run_early(context, 1u, inputs, {matte_.uav.Get(), legacy_args_.uav.Get(), nullptr, alpha8_.uav.Get(),
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
      return route_ == route_e::direct && published_ ? legacy_args_.data.Get() : nullptr;
    }
  };
}
#endif
