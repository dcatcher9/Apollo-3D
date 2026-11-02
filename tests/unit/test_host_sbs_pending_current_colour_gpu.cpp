/**
 * @file tests/unit/test_host_sbs_v2_live_warp.cpp
 * @brief Executable D3D11-WARP checks for the authenticated Host SBS V2 live renderer.
 */
#include "../tests_common.h"
#include "src/host_sbs_pending_current_colour_lane.h"

#ifdef _WIN32

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstddef>
  #include <cstdint>
  #include <cstdlib>
  #include <cstring>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <fstream>
  #include <limits>
  #include <nlohmann/json.hpp>
  #include <src/crypto.h>
  #include <src/depth_coordinate_v2.h>
  #include <src/host_sbs_queued_render_guard.h>
  #include <src/host_sbs_shader_cache.h>
  #include <src/host_sbs_subtitle_render.h>
  #include <src/host_sbs_v2_geometry.h>
  #include <src/platform/windows/host_sbs_v2_renderer.h>
  #include <sstream>
  #include <stdexcept>
  #include <string>
  #include <string_view>
  #include <tests/fixtures/host_sbs_adaptive_camera_state.h>
  #include <utility>
  #include <vector>
  #include <wrl/client.h>

class HostSbsV2LiveWarpGpuTest_QueuedRenderRecorderScopeMatchesLegacyOrIndependentFlat_Test;

namespace {
  using Microsoft::WRL::ComPtr;

  struct rgba32f_t {
    float r;
    float g;
    float b;
    float a;
  };

  struct rgba16_bits_t {
    std::uint16_t r;
    std::uint16_t g;
    std::uint16_t b;
    std::uint16_t a;

    bool operator==(const rgba16_bits_t &) const = default;
  };

  struct bgra8_t {
    std::uint8_t b;
    std::uint8_t g;
    std::uint8_t r;
    std::uint8_t a;

    bool operator==(const bgra8_t &) const = default;
  };

  using models::host_sbs_v2_geometry_t;

  struct direct_parallax_constants_t {
    float source_u_limit = models::depth_coordinate_v2::direct_container_limit;
    float reserved_0 = 0.0f;
    float reserved_1 = 0.0f;
    float reserved_2 = 0.0f;
  };

  static_assert(sizeof(direct_parallax_constants_t) == 16u);

  enum class warp_shader_e {
    live_main,
    live_mapping,
    live_mask,
    direct_main,
    direct_mapping,
    flat_main,
  };

  struct subtitle_inputs_t {
    models::host_sbs_subtitle_render_constants_t constants;
    UINT width = 0u;
    UINT height = 0u;
    std::vector<std::uint32_t> matte;
    std::vector<rgba16_bits_t> snapshot;
    std::array<std::uint32_t, 6u> condition {};
    std::array<std::uint32_t, 32u> identity {};
  };

  class live_v2_warp_fixture_t {
    friend class ::HostSbsV2LiveWarpGpuTest_QueuedRenderRecorderScopeMatchesLegacyOrIndependentFlat_Test;
  public:
    bool initialize(
      std::string &error,
      const warp_shader_e shader = warp_shader_e::live_main,
      const bool fixed_eleven_reference = false,
      const bool hardware = false
    ) {
      mapping_output =
        shader == warp_shader_e::live_mapping ||
        shader == warp_shader_e::direct_mapping;
      direct_replay =
        shader == warp_shader_e::direct_main ||
        shader == warp_shader_e::direct_mapping;
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      auto status = D3D11CreateDevice(
        nullptr,
        hardware ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP,
        nullptr,
        0,
        requested,
        static_cast<UINT>(std::size(requested)),
        D3D11_SDK_VERSION,
        &device,
        &actual,
        &context
      );
      if (FAILED(status) || actual < D3D_FEATURE_LEVEL_11_0) {
        error = "D3D11 feature level 11 initialization failed";
        return false;
      }

      const std::filesystem::path shader_root = SUNSHINE_SHADERS_DIR;
      if (shader == warp_shader_e::flat_main) {
        ComPtr<ID3DBlob> flat_bytecode;
        ComPtr<ID3DBlob> flat_errors;
        status = D3DCompileFromFile(
          (shader_root / "sbs_flat_identity_ps.hlsl").c_str(),
          nullptr,
          D3D_COMPILE_STANDARD_FILE_INCLUDE,
          "main_ps",
          "ps_5_0",
          D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
          0,
          &flat_bytecode,
          &flat_errors
        );
        if (FAILED(status) || !flat_bytecode ||
            FAILED(device->CreatePixelShader(
              flat_bytecode->GetBufferPointer(),
              flat_bytecode->GetBufferSize(),
              nullptr,
              &pixel_shader
            ))) {
          error = flat_errors ?
                    static_cast<const char *>(flat_errors->GetBufferPointer()) :
                    "could not create the independent flat-SBS pixel shader";
          return false;
        }
      } else if (shader == warp_shader_e::live_main && !fixed_eleven_reference) {
        const auto live_sources =
          models::host_sbs_shader_cache::snapshot_sources(
            shader_root,
            models::host_sbs_shader_cache::parallax_v2_live_renderer_specs
          );
        if (!live_sources) {
          error = "could not snapshot the live V2 renderer source closure";
          return false;
        }
        const auto closure =
          models::host_sbs_shader_cache::source_closure_sha256(live_sources);
        if (closure !=
            models::host_sbs_shader_cache::
              parallax_v2_live_renderer_source_closure_sha256) {
          error = "live V2 renderer source closure does not match its authenticated pin";
          return false;
        }
        const auto pixel_bytecode = models::host_sbs_shader_cache::get(
          live_sources,
          models::host_sbs_shader_cache::parallax_v2_live_renderer
        );
        if (!pixel_bytecode ||
            FAILED(device->CreatePixelShader(
              pixel_bytecode->data(),
              pixel_bytecode->size(),
              nullptr,
              &pixel_shader
            ))) {
          error = "could not create the authenticated live V2 pixel shader";
          return false;
        }
      } else if ((shader == warp_shader_e::live_mapping || shader == warp_shader_e::live_mask) && !fixed_eleven_reference) {
        const auto diagnostic_sources =
          models::host_sbs_shader_cache::snapshot_sources(
            shader_root,
            models::host_sbs_shader_cache::parallax_v2_live_diagnostic_specs
          );
        if (!diagnostic_sources) {
          error = "could not snapshot the live V2 diagnostic source closure";
          return false;
        }
        const auto closure =
          models::host_sbs_shader_cache::source_closure_sha256(diagnostic_sources);
        if (closure !=
            models::host_sbs_shader_cache::
              parallax_v2_live_diagnostic_source_closure_sha256) {
          error = "live V2 diagnostic source closure does not match its authenticated pin";
          return false;
        }
        const auto pixel_bytecode = models::host_sbs_shader_cache::get(
          diagnostic_sources,
          shader == warp_shader_e::live_mask ?
            models::host_sbs_shader_cache::parallax_v2_live_mask :
            models::host_sbs_shader_cache::parallax_v2_live_mapping
        );
        if (!pixel_bytecode || FAILED(device->CreatePixelShader(pixel_bytecode->data(), pixel_bytecode->size(), nullptr, &pixel_shader))) {
          error = "could not create the authenticated live V2 mapping pixel shader";
          return false;
        }
      } else {
        const bool mask_output = shader == warp_shader_e::live_mask;
        const auto source_file = direct_replay ?
                                   "sbs_direct_replay_ps.hlsl" :
                                 mapping_output || mask_output ?
                                   "sbs_reprojection_v2_diagnostics_ps.hlsl" :
                                   "sbs_reprojection_v2_live_ps.hlsl";
        const auto entrypoint = mask_output      ? "mask_ps" :
                                mapping_output   ? "mapping_ps" :
                                                   "main_ps";
        constexpr D3D_SHADER_MACRO fixed_eleven_macros[] = {
          {"HOST_SBS_TEST_FIXED_ELEVEN_REFERENCE", "1"},
          {nullptr, nullptr},
        };
        ComPtr<ID3DBlob> reference_bytecode;
        ComPtr<ID3DBlob> reference_errors;
        status = D3DCompileFromFile(
          (shader_root / source_file).c_str(),
          fixed_eleven_reference ? fixed_eleven_macros : nullptr,
          D3D_COMPILE_STANDARD_FILE_INCLUDE,
          entrypoint,
          "ps_5_0",
          D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
          0,
          &reference_bytecode,
          &reference_errors
        );
        if (FAILED(status) || !reference_bytecode ||
            FAILED(device->CreatePixelShader(
              reference_bytecode->GetBufferPointer(),
              reference_bytecode->GetBufferSize(),
              nullptr,
              &pixel_shader
            ))) {
          error = reference_errors ?
                    static_cast<const char *>(reference_errors->GetBufferPointer()) :
                    "could not create the direct or fixed-eleven reference pixel shader";
          return false;
        }
      }

      ComPtr<ID3DBlob> vertex_bytecode;
      ComPtr<ID3DBlob> vertex_errors;
      status = D3DCompileFromFile(
        (shader_root / "sbs_reprojection_vs.hlsl").c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main_vs",
        "vs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &vertex_bytecode,
        &vertex_errors
      );
      if (FAILED(status) || !vertex_bytecode ||
          FAILED(device->CreateVertexShader(
            vertex_bytecode->GetBufferPointer(),
            vertex_bytecode->GetBufferSize(),
            nullptr,
            &vertex_shader
          ))) {
        error = vertex_errors ?
                  static_cast<const char *>(vertex_errors->GetBufferPointer()) :
                  "could not create the SBS fullscreen vertex shader";
        return false;
      }

      D3D11_SAMPLER_DESC sampler_desc {};
      sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
      sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
      if (FAILED(device->CreateSamplerState(&sampler_desc, &sampler))) {
        error = "could not create the live V2 linear-clamp sampler";
        return false;
      }

      return true;
    }

    nlohmann::json adapter_identity() const {
      ComPtr<IDXGIDevice> dxgi_device;
      ComPtr<IDXGIAdapter> adapter;
      DXGI_ADAPTER_DESC desc {};
      if (FAILED(device.As(&dxgi_device)) || FAILED(dxgi_device->GetAdapter(&adapter)) ||
          FAILED(adapter->GetDesc(&desc))) {
        throw std::runtime_error("Cannot identify replay D3D11 adapter");
      }
      return {{"description", std::filesystem::path(desc.Description).string()},
              {"vendor_id", desc.VendorId}, {"device_id", desc.DeviceId},
              {"adapter_luid_low", desc.AdapterLuid.LowPart},
              {"adapter_luid_high", desc.AdapterLuid.HighPart}};
    }

    // Research-only: compile a pinned private shader and own immutable guard operands.
    bool configure_private_guard(const std::filesystem::path &root, const std::string &closure, const models::research_queued_render_guard_v1::expected_t *expected, const std::vector<std::uint32_t> *transaction, std::string &error, std::string_view source_file = "sbs_reprojection_v2_live_ps.hlsl") {
      if (!device || !context) {
        error = "private guard requires initialized fixture";
        return false;
      }
      const std::array specs {models::host_sbs_shader_cache::shader_spec {
        source_file,
        "main_ps",
        "ps_5_0"
      }};
      const auto sources = models::host_sbs_shader_cache::snapshot_sources(root, specs);
      if (!sources || models::host_sbs_shader_cache::source_closure_sha256(sources) != closure) {
        error = "private queued guard closure mismatch";
        return false;
      }
      const auto code = models::host_sbs_shader_cache::get(sources, specs[0]);
      ComPtr<ID3D11PixelShader> ps;
      if (!code || FAILED(device->CreatePixelShader(code->data(), code->size(), nullptr, &ps))) {
        error = "private queued guard PS unavailable";
        return false;
      }
      ComPtr<ID3D11Buffer> cb, packet;
      ComPtr<ID3D11ShaderResourceView> srv;
      if (expected) {
        D3D11_BUFFER_DESC d {};
        d.ByteWidth = sizeof(*expected);
        d.Usage = D3D11_USAGE_IMMUTABLE;
        d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA init {expected, 0u, 0u};
        if (FAILED(device->CreateBuffer(&d, &init, &cb))) {
          error = "private queued CB failed";
          return false;
        }
      }
      if (transaction && !transaction->empty()) {
        D3D11_BUFFER_DESC d {};
        d.ByteWidth = static_cast<UINT>(transaction->size() * sizeof(std::uint32_t));
        d.Usage = D3D11_USAGE_IMMUTABLE;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA init {transaction->data(), 0u, 0u};
        D3D11_SHADER_RESOURCE_VIEW_DESC view {};
        view.Format = DXGI_FORMAT_R32_TYPELESS;
        view.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        view.BufferEx.NumElements = static_cast<UINT>(transaction->size());
        view.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        if (FAILED(device->CreateBuffer(&d, &init, &packet)) || FAILED(device->CreateShaderResourceView(packet.Get(), &view, &srv))) {
          error = "private queued raw packet failed";
          return false;
        }
      }
      pixel_shader = ps;
      private_guard_cb = cb;
      private_guard_packet = packet;
      private_guard_srv = srv;
      private_guard_configured = true;
      return true;
    }

    bool private_guard_slots_clear() const {
      ID3D11Buffer *cb = nullptr;
      ID3D11ShaderResourceView *srv = nullptr;
      context->PSGetConstantBuffers(5u, 1u, &cb);
      context->PSGetShaderResources(10u, 1u, &srv);
      const bool clear = cb == nullptr && srv == nullptr;
      if (cb) {
        cb->Release();
      }
      if (srv) {
        srv->Release();
      }
      return clear;
    }

    bool render(
      const DXGI_FORMAT color_format,
      const UINT color_bytes_per_pixel,
      const UINT source_width,
      const UINT source_height,
      const void *source_pixels,
      const std::vector<float> &final_parallax,
      const std::vector<float> &candidate_parallax,
      const models::depth_coordinate_v2::state_words_t &state,
      const std::array<float, 4> &clear_color,
      std::vector<std::byte> &packed_output,
      std::string &error,
      const UINT depth_width = 0u,
      const UINT depth_height = 0u,
      const host_sbs_v2_geometry_t &geometry = host_sbs_v2_geometry_t {},
      const subtitle_inputs_t *subtitle = nullptr,
      ComPtr<ID3D11Texture2D> *gpu_output = nullptr,
      const bool readback_output = true
    ) {
      const UINT resolved_depth_width = depth_width == 0u ? source_width : depth_width;
      const UINT resolved_depth_height = depth_height == 0u ? source_height : depth_height;
      if (!source_pixels || source_width == 0u || source_height == 0u ||
          (depth_width == 0u) != (depth_height == 0u) ||
          final_parallax.size() !=
            static_cast<std::size_t>(resolved_depth_width) * resolved_depth_height ||
          candidate_parallax.size() !=
            static_cast<std::size_t>(resolved_depth_width) * resolved_depth_height) {
        error = "invalid live V2 render input";
        return false;
      }

      ComPtr<ID3D11Texture2D> color_texture;
      ComPtr<ID3D11ShaderResourceView> color_srv;
      if (!create_immutable_texture_srv(
            color_format,
            source_width,
            source_height,
            color_bytes_per_pixel,
            source_pixels,
            color_texture,
            color_srv
          )) {
        error = "could not create the mono color texture";
        return false;
      }

      ComPtr<ID3D11Texture2D> parallax_texture;
      ComPtr<ID3D11ShaderResourceView> parallax_srv;
      if (!create_immutable_texture_srv(
            DXGI_FORMAT_R32_FLOAT,
            resolved_depth_width,
            resolved_depth_height,
            sizeof(float),
            final_parallax.data(),
            parallax_texture,
            parallax_srv
          )) {
        error = "could not create the signed parallax texture";
        return false;
      }

      ComPtr<ID3D11Buffer> state_buffer;
      ComPtr<ID3D11ShaderResourceView> state_srv;
      D3D11_BUFFER_DESC state_desc {};
      state_desc.ByteWidth = sizeof(state);
      state_desc.Usage = D3D11_USAGE_IMMUTABLE;
      state_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      state_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      state_desc.StructureByteStride =
        models::depth_coordinate_v2::state_vector_width * sizeof(float);
      D3D11_SUBRESOURCE_DATA state_data {};
      state_data.pSysMem = state.data();
      D3D11_SHADER_RESOURCE_VIEW_DESC state_srv_desc {};
      state_srv_desc.Format = DXGI_FORMAT_UNKNOWN;
      state_srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      state_srv_desc.Buffer.FirstElement = 0;
      state_srv_desc.Buffer.NumElements =
        models::depth_coordinate_v2::state_vector_count;
      if (FAILED(device->CreateBuffer(&state_desc, &state_data, &state_buffer)) ||
          FAILED(device->CreateShaderResourceView(
            state_buffer.Get(),
            &state_srv_desc,
            &state_srv
          ))) {
        error = "could not create the authenticated V2 state buffer";
        return false;
      }

      // Exact 48-byte mirror of HostSbsV2Geometry at b2. Creating one immutable buffer for each
      // draw also proves that ROI geometry belongs to that exact matched render rather than to
      // mutable fixture state retained from an earlier frame.
      ComPtr<ID3D11Buffer> geometry_buffer;
      D3D11_BUFFER_DESC geometry_desc {};
      geometry_desc.ByteWidth = sizeof(geometry);
      geometry_desc.Usage = D3D11_USAGE_IMMUTABLE;
      geometry_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA geometry_data {};
      geometry_data.pSysMem = &geometry;
      if (FAILED(device->CreateBuffer(
            &geometry_desc,
            &geometry_data,
            &geometry_buffer
          ))) {
        error = "could not create live V2 render constants";
        return false;
      }

      ComPtr<ID3D11Texture2D> matte_texture, snapshot_texture;
      ComPtr<ID3D11ShaderResourceView> matte_srv, snapshot_srv, condition_srv, identity_srv;
      ComPtr<ID3D11Buffer> subtitle_buffer, condition_buffer, identity_buffer;
      if (subtitle) {
        if (subtitle->matte.size() != static_cast<std::size_t>(subtitle->width) * subtitle->height ||
            subtitle->snapshot.size() != subtitle->matte.size() ||
            !create_immutable_texture_srv(DXGI_FORMAT_R32_UINT, subtitle->width, subtitle->height,
                                         sizeof(std::uint32_t), subtitle->matte.data(), matte_texture, matte_srv) ||
            !create_immutable_texture_srv(DXGI_FORMAT_R16G16B16A16_FLOAT, subtitle->width, subtitle->height,
                                         sizeof(rgba16_bits_t), subtitle->snapshot.data(), snapshot_texture, snapshot_srv)) {
          error = "could not create native glyph matte/snapshot operands";
          return false;
        }
        const auto words_srv = [&](const auto &words, ComPtr<ID3D11Buffer> &buffer,
                                   ComPtr<ID3D11ShaderResourceView> &srv) {
          D3D11_BUFFER_DESC desc {};
          desc.ByteWidth = static_cast<UINT>(sizeof(words));
          desc.Usage = D3D11_USAGE_IMMUTABLE;
          desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
          desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
          desc.StructureByteStride = sizeof(std::uint32_t);
          D3D11_SUBRESOURCE_DATA initial {};
          initial.pSysMem = words.data();
          return SUCCEEDED(device->CreateBuffer(&desc, &initial, &buffer)) &&
            SUCCEEDED(device->CreateShaderResourceView(buffer.Get(), nullptr, &srv));
        };
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = sizeof(subtitle->constants);
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial {};
        initial.pSysMem = &subtitle->constants;
        if (!words_srv(subtitle->condition, condition_buffer, condition_srv) ||
            !words_srv(subtitle->identity, identity_buffer, identity_srv) ||
            FAILED(device->CreateBuffer(&desc, &initial, &subtitle_buffer))) {
          error = "could not create glyph owner/plane/constants operands";
          return false;
        }
      }

      ComPtr<ID3D11Buffer> direct_constants_buffer;
      if (direct_replay) {
        const direct_parallax_constants_t direct_constants {};
        D3D11_BUFFER_DESC direct_desc {};
        direct_desc.ByteWidth = sizeof(direct_constants);
        direct_desc.Usage = D3D11_USAGE_IMMUTABLE;
        direct_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA direct_data {};
        direct_data.pSysMem = &direct_constants;
        if (FAILED(device->CreateBuffer(
              &direct_desc,
              &direct_data,
              &direct_constants_buffer
            ))) {
          error = "could not create direct-parallax constants";
          return false;
        }
      }

      const UINT packed_width = source_width * 2u;
      const DXGI_FORMAT output_format =
        mapping_output ? DXGI_FORMAT_R32_FLOAT : color_format;
      const UINT output_bytes_per_pixel =
        mapping_output ? sizeof(float) : color_bytes_per_pixel;
      D3D11_TEXTURE2D_DESC output_desc {};
      output_desc.Width = packed_width;
      output_desc.Height = source_height;
      output_desc.MipLevels = 1u;
      output_desc.ArraySize = 1u;
      output_desc.Format = output_format;
      output_desc.SampleDesc.Count = 1u;
      output_desc.Usage = D3D11_USAGE_DEFAULT;
      output_desc.BindFlags = D3D11_BIND_RENDER_TARGET |
        (gpu_output ? D3D11_BIND_SHADER_RESOURCE : 0u);
      ComPtr<ID3D11Texture2D> output_texture;
      ComPtr<ID3D11RenderTargetView> output_rtv;
      if (FAILED(device->CreateTexture2D(
            &output_desc,
            nullptr,
            &output_texture
          )) ||
          FAILED(device->CreateRenderTargetView(
            output_texture.Get(),
            nullptr,
            &output_rtv
          ))) {
        error = "could not create the packed V2 render target";
        return false;
      }

      context->ClearRenderTargetView(output_rtv.Get(), clear_color.data());
      const D3D11_VIEWPORT viewport {
        0.0f,
        0.0f,
        static_cast<float>(packed_width),
        static_cast<float>(source_height),
        0.0f,
        1.0f,
      };
      if (direct_constants_buffer) {
        ID3D11Buffer *direct_constants[] = {direct_constants_buffer.Get()};
        context->PSSetConstantBuffers(4, 1, direct_constants);
      }
      const platf::dxgi::host_sbs_v2_draw_command_t draw_command {
        .render_targets = {output_rtv.Get(), nullptr},
        .render_target_count = 1u,
        .vertex_shader = vertex_shader.Get(),
        .pixel_shader = pixel_shader.Get(),
        .viewport = viewport,
        .sampler = sampler.Get(),
        .shader_resources = {
          color_srv.Get(),
          parallax_srv.Get(),
          state_srv.Get(),
          matte_srv.Get(),
          snapshot_srv.Get(),
          condition_srv.Get(),
          identity_srv.Get(),
        },
        .geometry_constants = geometry_buffer.Get(),
        .subtitle_constants = subtitle_buffer.Get(),
        .queued_guard_override = use_recorder_guard_scope,
        .queued_guard_constants = use_recorder_guard_scope ? private_guard_cb.Get() : nullptr,
        .queued_guard_transaction = use_recorder_guard_scope ? private_guard_srv.Get() : nullptr,
      };
      bool submitted = false;
      recorder_flat_fallback_used = false;
      {
        models::research_pending_current_colour_v1::scoped_negative_t negative;
        if(pending_scope && !negative.bind(context.Get(),negative_scope_cb.Get(),negative_scope_srv.Get(),current_scope_cb.Get(),current_scope_srv.Get())) {error="pending scope rejected";return false;}
        models::research_queued_render_guard_v1::scoped_binding_t guard;
        if (private_guard_configured && !use_recorder_guard_scope && !guard.bind(context.Get(), private_guard_cb.Get(), private_guard_srv.Get())) {
          error = "private queued guard binding rejected";
          return false;
        }
        submitted = platf::dxgi::record_host_sbs_v2_draw(context.Get(), draw_command);
      }
      if (!submitted && use_recorder_guard_scope && recorder_flat_fallback_ps) {
        ComPtr<ID3D11Predicate> previous_predicate;
        BOOL previous_predicate_value = FALSE;
        context->GetPredication(&previous_predicate, &previous_predicate_value);
        context->SetPredication(nullptr, FALSE);
        auto fallback = draw_command;
        fallback.pixel_shader = recorder_flat_fallback_ps.Get();
        fallback.shader_resources = {color_srv.Get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        fallback.subtitle_constants = nullptr;
        fallback.queued_guard_override = false;
        fallback.queued_guard_constants = nullptr;
        fallback.queued_guard_transaction = nullptr;
        submitted = platf::dxgi::record_host_sbs_v2_draw(context.Get(), fallback);
        context->SetPredication(previous_predicate.Get(), previous_predicate_value);
        recorder_flat_fallback_used = submitted;
      }
      ID3D11Buffer *null_constants = nullptr;
      context->PSSetConstantBuffers(4u, 1u, &null_constants);
      if (!submitted) {
        error = "shared Host SBS V2 draw recorder rejected complete operands";
        return false;
      }

      std::array<ID3D11ShaderResourceView *, 7u> remaining_resources {};
      context->PSGetShaderResources(0u, static_cast<UINT>(remaining_resources.size()), remaining_resources.data());
      ID3D11Buffer *remaining_subtitle = nullptr;
      context->PSGetConstantBuffers(3u, 1u, &remaining_subtitle);
      bool unbound = remaining_subtitle == nullptr;
      if (remaining_subtitle) remaining_subtitle->Release();
      for (auto *resource : remaining_resources) {
        unbound = unbound && resource == nullptr;
        if (resource) resource->Release();
      }
      if (!unbound) {
        error = "shared renderer left native subtitle resources or b3 bound";
        return false;
      }

      if (gpu_output) *gpu_output = output_texture;
      if (!readback_output) { packed_output.clear(); return true; }

      auto staging_desc = output_desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging))) {
        error = "could not create the packed V2 staging texture";
        return false;
      }
      context->CopyResource(staging.Get(), output_texture.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        error = "could not read the packed V2 render result";
        return false;
      }
      const std::size_t row_bytes =
        static_cast<std::size_t>(packed_width) * output_bytes_per_pixel;
      packed_output.resize(row_bytes * source_height);
      for (UINT y = 0; y < source_height; ++y) {
        std::memcpy(
          packed_output.data() + static_cast<std::size_t>(y) * row_bytes,
          static_cast<const std::byte *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          row_bytes
        );
      }
      context->Unmap(staging.Get(), 0);
      return true;
    }

    struct private_layer_output_t {
      std::vector<std::byte> reconstructed_rgba8;
      std::vector<std::byte> base_background_bgra8;
      std::vector<std::byte> reference_rgba8;
      std::vector<std::byte> shifted_foreground_rgba32f;
      std::vector<std::byte> shifted_body_alpha_f32;
    };

    // External diagnostic extension: the production Base draw remains the only
    // background inverse warp. All reference premultiplication, UI resampling and
    // compositing occur on this device; no CPU-derived warped image is uploaded.
    bool render_private_layer_reference(
      UINT width, UINT height, const std::vector<bgra8_t> &background,
      const std::vector<std::array<std::uint8_t, 4u>> &foreground_rgba,
      const std::vector<std::uint8_t> &effective_body_alpha,
      const std::vector<float> &base,
      const models::depth_coordinate_v2::state_words_t &state,
      UINT field_width, UINT field_height, const host_sbs_v2_geometry_t &geometry,
      const subtitle_inputs_t &subtitle, const std::filesystem::path &reference_root,
      const std::string &expected_closure, private_layer_output_t &output, std::string &error
    ) {
      namespace cache = models::host_sbs_shader_cache;
      constexpr std::array<cache::shader_spec, 3u> specs {{
        {"private_layer_reference_cs.hlsl", "premultiply_main", "cs_5_0"},
        {"private_layer_reference_cs.hlsl", "reconstruct_main", "cs_5_0"},
        {"private_layer_reference_cs.hlsl", "reference_main", "cs_5_0"},
      }};
      const auto sources = cache::snapshot_sources(reference_root, specs);
      if (!sources || cache::source_closure_sha256(sources) != expected_closure) {
        error = "Private layered-reference shader closure does not match its separate pin";
        return false;
      }
      std::array<ComPtr<ID3D11ComputeShader>, 3u> shaders;
      for (std::size_t i = 0u; i < specs.size(); ++i) {
        const auto code = cache::get(sources, specs[i]);
        if (!code || FAILED(device->CreateComputeShader(code->data(), code->size(), nullptr, &shaders[i]))) {
          error = "Could not create private layered-reference compute shader";
          return false;
        }
      }
      if (background.size() != std::size_t(width) * height || foreground_rgba.size() != background.size() ||
          effective_body_alpha.size() != background.size()) {
        error = "Private reference requires full-source authored background and straight RGBA";
        return false;
      }
      auto disabled = subtitle;
      disabled.constants.enabled = 0u;
      ComPtr<ID3D11Texture2D> base_background;
      std::vector<std::byte> unused_readback;
      if (!render(DXGI_FORMAT_B8G8R8A8_UNORM, sizeof(bgra8_t), width, height, background.data(),
            base, base, state, {}, unused_readback, error, field_width, field_height, geometry,
            &disabled, &base_background, false)) return false;

      ComPtr<ID3D11Texture2D> foreground_texture, background_texture, body_alpha_texture;
      ComPtr<ID3D11ShaderResourceView> foreground_srv, background_srv, base_background_srv, body_alpha_srv;
      if (!create_immutable_texture_srv(DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 4u,
            foreground_rgba.data(), foreground_texture, foreground_srv) ||
          !create_immutable_texture_srv(DXGI_FORMAT_B8G8R8A8_UNORM, width, height, 4u,
            background.data(), background_texture, background_srv) ||
          !create_immutable_texture_srv(DXGI_FORMAT_R8_UNORM, width, height, 1u,
            effective_body_alpha.data(), body_alpha_texture, body_alpha_srv) ||
          FAILED(device->CreateShaderResourceView(base_background.Get(), nullptr, &base_background_srv))) {
        error = "Could not create private authored-layer source views";
        return false;
      }
      struct plane_t {
        ComPtr<ID3D11Texture2D> texture;
        ComPtr<ID3D11ShaderResourceView> srv;
        ComPtr<ID3D11UnorderedAccessView> uav;
      };
      const auto plane = [&](UINT w, UINT h, DXGI_FORMAT format, plane_t &value) {
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = w; desc.Height = h; desc.MipLevels = desc.ArraySize = 1u;
        desc.Format = format; desc.SampleDesc.Count = 1u;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        return SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &value.texture)) &&
          SUCCEEDED(device->CreateShaderResourceView(value.texture.Get(), nullptr, &value.srv)) &&
          SUCCEEDED(device->CreateUnorderedAccessView(value.texture.Get(), nullptr, &value.uav));
      };
      plane_t premult, reconstruction, reference, shifted, shifted_body;
      if (!plane(width, height, DXGI_FORMAT_R32G32B32A32_FLOAT, premult) ||
          !plane(width, height, DXGI_FORMAT_R8G8B8A8_UNORM, reconstruction) ||
          !plane(width * 2u, height, DXGI_FORMAT_R8G8B8A8_UNORM, reference) ||
          !plane(width * 2u, height, DXGI_FORMAT_R32G32B32A32_FLOAT, shifted) ||
          !plane(width * 2u, height, DXGI_FORMAT_R32_FLOAT, shifted_body)) {
        error = "Could not create private reference outputs";
        return false;
      }
      ComPtr<ID3D11Buffer> reference_cb, geometry_cb, condition_buffer;
      ComPtr<ID3D11ShaderResourceView> condition_srv;
      const std::array<std::uint32_t, 4u> reference_words {width, height, subtitle.constants.source_width, 0u};
      const auto immutable_buffer = [&](const void *data, UINT size, UINT bind, UINT stride,
                                       ComPtr<ID3D11Buffer> &buffer) {
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = size; desc.Usage = D3D11_USAGE_IMMUTABLE; desc.BindFlags = bind;
        if (stride) { desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride = stride; }
        D3D11_SUBRESOURCE_DATA initial {}; initial.pSysMem = data;
        return SUCCEEDED(device->CreateBuffer(&desc, &initial, &buffer));
      };
      if (!immutable_buffer(reference_words.data(), sizeof(reference_words), D3D11_BIND_CONSTANT_BUFFER, 0u, reference_cb) ||
          !immutable_buffer(&geometry, sizeof(geometry), D3D11_BIND_CONSTANT_BUFFER, 0u, geometry_cb) ||
          !immutable_buffer(subtitle.condition.data(), sizeof(subtitle.condition), D3D11_BIND_SHADER_RESOURCE, 4u, condition_buffer) ||
          FAILED(device->CreateShaderResourceView(condition_buffer.Get(), nullptr, &condition_srv))) {
        error = "Could not create private reference constants/plane view";
        return false;
      }

      // A private fixture owns its context. Explicit unbinds separate each SRV/UAV
      // role; production's shared recorder has already unbound all PS readers/RTVs.
      ID3D11ShaderResourceView *null_srvs[6] {};
      ID3D11UnorderedAccessView *null_uavs[5] {};
      const auto unbind = [&]() {
        context->CSSetShaderResources(0u, 6u, null_srvs);
        context->CSSetUnorderedAccessViews(0u, 5u, null_uavs, nullptr);
      };
      ID3D11Buffer *b0 = reference_cb.Get(), *b2 = geometry_cb.Get();
      ID3D11SamplerState *linear = sampler.Get();
      context->CSSetConstantBuffers(0u, 1u, &b0);
      context->CSSetConstantBuffers(2u, 1u, &b2);
      context->CSSetSamplers(0u, 1u, &linear);
      ID3D11ShaderResourceView *srvs[6] {foreground_srv.Get(), background_srv.Get(),
        base_background_srv.Get(), condition_srv.Get(), nullptr, body_alpha_srv.Get()};
      ID3D11UnorderedAccessView *uavs[5] {premult.uav.Get(), nullptr, nullptr, nullptr, nullptr};
      context->CSSetShaderResources(0u, 6u, srvs);
      context->CSSetUnorderedAccessViews(0u, 5u, uavs, nullptr);
      context->CSSetShader(shaders[0].Get(), nullptr, 0u);
      context->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);
      unbind();

      srvs[4] = premult.srv.Get();
      uavs[0] = nullptr; uavs[1] = reconstruction.uav.Get();
      context->CSSetShaderResources(0u, 6u, srvs);
      context->CSSetUnorderedAccessViews(0u, 5u, uavs, nullptr);
      context->CSSetShader(shaders[1].Get(), nullptr, 0u);
      context->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);
      unbind();

      uavs[1] = nullptr; uavs[2] = reference.uav.Get(); uavs[3] = shifted.uav.Get();
      uavs[4] = shifted_body.uav.Get();
      context->CSSetShaderResources(0u, 6u, srvs);
      context->CSSetUnorderedAccessViews(0u, 5u, uavs, nullptr);
      context->CSSetShader(shaders[2].Get(), nullptr, 0u);
      context->Dispatch((width * 2u + 15u) / 16u, (height + 15u) / 16u, 1u);
      unbind();
      ID3D11Buffer *null_buffer = nullptr;
      context->CSSetConstantBuffers(0u, 1u, &null_buffer);
      context->CSSetConstantBuffers(2u, 1u, &null_buffer);
      ID3D11SamplerState *null_sampler = nullptr;
      context->CSSetSamplers(0u, 1u, &null_sampler);
      context->CSSetShader(nullptr, nullptr, 0u);

      // These are diagnostic transport/readbacks after all reference commands.
      // None is uploaded again or used to choose a reference pixel.
      const auto readback = [&](ID3D11Texture2D *texture, UINT bytes_per_pixel, std::vector<std::byte> &bytes) {
        D3D11_TEXTURE2D_DESC desc {}; texture->GetDesc(&desc);
        desc.Usage = D3D11_USAGE_STAGING; desc.BindFlags = 0u; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> staging;
        if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) return false;
        context->CopyResource(staging.Get(), texture);
        D3D11_MAPPED_SUBRESOURCE mapped {};
        if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
        const auto row = std::size_t(desc.Width) * bytes_per_pixel;
        bytes.resize(row * desc.Height);
        for (UINT y = 0u; y < desc.Height; ++y) {
          std::memcpy(bytes.data() + y * row, static_cast<const std::byte *>(mapped.pData) + y * mapped.RowPitch, row);
        }
        context->Unmap(staging.Get(), 0u);
        return true;
      };
      if (!readback(reconstruction.texture.Get(), 4u, output.reconstructed_rgba8) ||
          !readback(base_background.Get(), 4u, output.base_background_bgra8) ||
          !readback(reference.texture.Get(), 4u, output.reference_rgba8) ||
          !readback(shifted.texture.Get(), 16u, output.shifted_foreground_rgba32f) ||
          !readback(shifted_body.texture.Get(), 4u, output.shifted_body_alpha_f32)) {
        error = "Could not read private layered-reference result";
        return false;
      }
      return true;
    }

    bool render_private_selector(
      const DXGI_FORMAT color_format,
      const UINT color_bytes_per_pixel,
      const UINT source_width,
      const UINT source_height,
      const void *source_pixels,
      const std::vector<float> &final_parallax,
      const std::vector<float> &candidate_parallax,
      const models::depth_coordinate_v2::state_words_t &state,
      const std::array<float, 4> &clear_color,
      std::vector<std::byte> &packed_output,
      std::string &error,
      const std::filesystem::path &private_shader_root,
      const std::string &expected_private_closure,
      std::vector<std::byte> &repeat_colour,
      std::array<std::vector<std::byte>, 2u> &diagnostic_output,
      const UINT depth_width = 0u,
      const UINT depth_height = 0u,
      const host_sbs_v2_geometry_t &geometry = host_sbs_v2_geometry_t {},
      const subtitle_inputs_t *subtitle = nullptr,
      ComPtr<ID3D11Texture2D> *gpu_output = nullptr,
      const bool readback_output = true
    ) {
      namespace cache = models::host_sbs_shader_cache;
      constexpr std::array<cache::shader_spec, 2u> private_specs {{
        {"private_two_sided_selector_ps.hlsl", "private_selector_ps", "ps_5_0"},
        {"sbs_reprojection_vs.hlsl", "main_vs", "vs_5_0"},
      }};
      const auto private_sources = cache::snapshot_sources(private_shader_root, private_specs);
      if (!private_sources || cache::source_closure_sha256(private_sources) != expected_private_closure) {
        error = "Private two-sided selector closure differs from its explicit research pin";
        return false;
      }
      const auto private_code = cache::get(private_sources, private_specs[0]);
      ComPtr<ID3D11PixelShader> private_pixel_shader;
      if (!private_code || FAILED(device->CreatePixelShader(private_code->data(), private_code->size(),
          nullptr, &private_pixel_shader))) {
        error = "Cannot compile the pinned private two-sided selector";
        return false;
      }
      if (color_format != DXGI_FORMAT_B8G8R8A8_UNORM || color_bytes_per_pixel != 4u ||
          mapping_output || direct_replay || !subtitle || !readback_output) {
        error = "Private selector requires full SDR BGRA8 colour, current subtitle tuple, and readback";
        return false;
      }
      const UINT resolved_depth_width = depth_width == 0u ? source_width : depth_width;
      const UINT resolved_depth_height = depth_height == 0u ? source_height : depth_height;
      if (!source_pixels || source_width == 0u || source_height == 0u ||
          (depth_width == 0u) != (depth_height == 0u) ||
          final_parallax.size() !=
            static_cast<std::size_t>(resolved_depth_width) * resolved_depth_height ||
          candidate_parallax.size() !=
            static_cast<std::size_t>(resolved_depth_width) * resolved_depth_height) {
        error = "invalid live V2 render input";
        return false;
      }

      ComPtr<ID3D11Texture2D> color_texture;
      ComPtr<ID3D11ShaderResourceView> color_srv;
      if (!create_immutable_texture_srv(
            color_format,
            source_width,
            source_height,
            color_bytes_per_pixel,
            source_pixels,
            color_texture,
            color_srv
          )) {
        error = "could not create the mono color texture";
        return false;
      }

      ComPtr<ID3D11Texture2D> parallax_texture;
      ComPtr<ID3D11ShaderResourceView> parallax_srv;
      if (!create_immutable_texture_srv(
            DXGI_FORMAT_R32_FLOAT,
            resolved_depth_width,
            resolved_depth_height,
            sizeof(float),
            final_parallax.data(),
            parallax_texture,
            parallax_srv
          )) {
        error = "could not create the signed parallax texture";
        return false;
      }

      ComPtr<ID3D11Buffer> state_buffer;
      ComPtr<ID3D11ShaderResourceView> state_srv;
      D3D11_BUFFER_DESC state_desc {};
      state_desc.ByteWidth = sizeof(state);
      state_desc.Usage = D3D11_USAGE_IMMUTABLE;
      state_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      state_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      state_desc.StructureByteStride =
        models::depth_coordinate_v2::state_vector_width * sizeof(float);
      D3D11_SUBRESOURCE_DATA state_data {};
      state_data.pSysMem = state.data();
      D3D11_SHADER_RESOURCE_VIEW_DESC state_srv_desc {};
      state_srv_desc.Format = DXGI_FORMAT_UNKNOWN;
      state_srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      state_srv_desc.Buffer.FirstElement = 0;
      state_srv_desc.Buffer.NumElements =
        models::depth_coordinate_v2::state_vector_count;
      if (FAILED(device->CreateBuffer(&state_desc, &state_data, &state_buffer)) ||
          FAILED(device->CreateShaderResourceView(
            state_buffer.Get(),
            &state_srv_desc,
            &state_srv
          ))) {
        error = "could not create the authenticated V2 state buffer";
        return false;
      }

      // Exact 48-byte mirror of HostSbsV2Geometry at b2. Creating one immutable buffer for each
      // draw also proves that ROI geometry belongs to that exact matched render rather than to
      // mutable fixture state retained from an earlier frame.
      ComPtr<ID3D11Buffer> geometry_buffer;
      D3D11_BUFFER_DESC geometry_desc {};
      geometry_desc.ByteWidth = sizeof(geometry);
      geometry_desc.Usage = D3D11_USAGE_IMMUTABLE;
      geometry_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA geometry_data {};
      geometry_data.pSysMem = &geometry;
      if (FAILED(device->CreateBuffer(
            &geometry_desc,
            &geometry_data,
            &geometry_buffer
          ))) {
        error = "could not create live V2 render constants";
        return false;
      }

      ComPtr<ID3D11Texture2D> matte_texture, snapshot_texture;
      ComPtr<ID3D11ShaderResourceView> matte_srv, snapshot_srv, condition_srv, identity_srv;
      ComPtr<ID3D11Buffer> subtitle_buffer, condition_buffer, identity_buffer;
      if (subtitle) {
        if (subtitle->matte.size() != static_cast<std::size_t>(subtitle->width) * subtitle->height ||
            subtitle->snapshot.size() != subtitle->matte.size() ||
            !create_immutable_texture_srv(DXGI_FORMAT_R32_UINT, subtitle->width, subtitle->height,
                                         sizeof(std::uint32_t), subtitle->matte.data(), matte_texture, matte_srv) ||
            !create_immutable_texture_srv(DXGI_FORMAT_R16G16B16A16_FLOAT, subtitle->width, subtitle->height,
                                         sizeof(rgba16_bits_t), subtitle->snapshot.data(), snapshot_texture, snapshot_srv)) {
          error = "could not create native glyph matte/snapshot operands";
          return false;
        }
        const auto words_srv = [&](const auto &words, ComPtr<ID3D11Buffer> &buffer,
                                   ComPtr<ID3D11ShaderResourceView> &srv) {
          D3D11_BUFFER_DESC desc {};
          desc.ByteWidth = static_cast<UINT>(sizeof(words));
          desc.Usage = D3D11_USAGE_IMMUTABLE;
          desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
          desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
          desc.StructureByteStride = sizeof(std::uint32_t);
          D3D11_SUBRESOURCE_DATA initial {};
          initial.pSysMem = words.data();
          return SUCCEEDED(device->CreateBuffer(&desc, &initial, &buffer)) &&
            SUCCEEDED(device->CreateShaderResourceView(buffer.Get(), nullptr, &srv));
        };
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = sizeof(subtitle->constants);
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial {};
        initial.pSysMem = &subtitle->constants;
        if (!words_srv(subtitle->condition, condition_buffer, condition_srv) ||
            !words_srv(subtitle->identity, identity_buffer, identity_srv) ||
            FAILED(device->CreateBuffer(&desc, &initial, &subtitle_buffer))) {
          error = "could not create glyph owner/plane/constants operands";
          return false;
        }
      }

      ComPtr<ID3D11Buffer> direct_constants_buffer;
      if (direct_replay) {
        const direct_parallax_constants_t direct_constants {};
        D3D11_BUFFER_DESC direct_desc {};
        direct_desc.ByteWidth = sizeof(direct_constants);
        direct_desc.Usage = D3D11_USAGE_IMMUTABLE;
        direct_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA direct_data {};
        direct_data.pSysMem = &direct_constants;
        if (FAILED(device->CreateBuffer(
              &direct_desc,
              &direct_data,
              &direct_constants_buffer
            ))) {
          error = "could not create direct-parallax constants";
          return false;
        }
      }

      const UINT packed_width = source_width * 2u;
      const DXGI_FORMAT output_format =
        mapping_output ? DXGI_FORMAT_R32_FLOAT : color_format;
      const UINT output_bytes_per_pixel =
        mapping_output ? sizeof(float) : color_bytes_per_pixel;
      D3D11_TEXTURE2D_DESC output_desc {};
      output_desc.Width = packed_width;
      output_desc.Height = source_height;
      output_desc.MipLevels = 1u;
      output_desc.ArraySize = 1u;
      output_desc.Format = output_format;
      output_desc.SampleDesc.Count = 1u;
      output_desc.Usage = D3D11_USAGE_DEFAULT;
      output_desc.BindFlags = D3D11_BIND_RENDER_TARGET |
        (gpu_output ? D3D11_BIND_SHADER_RESOURCE : 0u);
      ComPtr<ID3D11Texture2D> output_texture;
      ComPtr<ID3D11RenderTargetView> output_rtv;
      if (FAILED(device->CreateTexture2D(
            &output_desc,
            nullptr,
            &output_texture
          )) ||
          FAILED(device->CreateRenderTargetView(
            output_texture.Get(),
            nullptr,
            &output_rtv
          ))) {
        error = "could not create the packed V2 render target";
        return false;
      }

      // Two unchanged production-recorder draws; b5 chooses only the diagnostic MRT layout.
      std::array<ComPtr<ID3D11Texture2D>, 2u> diagnostic_textures;
      std::array<ComPtr<ID3D11RenderTargetView>, 2u> diagnostic_rtvs;
      auto diagnostic_desc = output_desc;
      diagnostic_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
      diagnostic_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
      for (std::size_t diagnostic_index = 0u; diagnostic_index < 2u; ++diagnostic_index) {
        if (FAILED(device->CreateTexture2D(&diagnostic_desc, nullptr, &diagnostic_textures[diagnostic_index])) ||
            FAILED(device->CreateRenderTargetView(diagnostic_textures[diagnostic_index].Get(), nullptr,
              &diagnostic_rtvs[diagnostic_index]))) {
          error = "Cannot allocate private selector UV/weight diagnostic MRT";
          return false;
        }
        context->ClearRenderTargetView(diagnostic_rtvs[diagnostic_index].Get(), clear_color.data());
      }
      ComPtr<ID3D11Texture2D> repeat_colour_texture;
      ComPtr<ID3D11RenderTargetView> repeat_colour_rtv;
      if (FAILED(device->CreateTexture2D(&output_desc, nullptr, &repeat_colour_texture)) ||
          FAILED(device->CreateRenderTargetView(repeat_colour_texture.Get(), nullptr, &repeat_colour_rtv))) {
        error = "Cannot allocate repeated candidate colour proof target";
        return false;
      }
      context->ClearRenderTargetView(repeat_colour_rtv.Get(), clear_color.data());
      std::array<ComPtr<ID3D11Buffer>, 2u> diagnostic_constants;
      for (std::uint32_t diagnostic_mode = 0u; diagnostic_mode < 2u; ++diagnostic_mode) {
        const std::array<std::uint32_t, 4u> words {diagnostic_mode, 0u, 0u, 0u};
        D3D11_BUFFER_DESC diagnostic_cb_desc {};
        diagnostic_cb_desc.ByteWidth = sizeof(words);
        diagnostic_cb_desc.Usage = D3D11_USAGE_IMMUTABLE;
        diagnostic_cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA diagnostic_cb_data {};
        diagnostic_cb_data.pSysMem = words.data();
        if (FAILED(device->CreateBuffer(&diagnostic_cb_desc, &diagnostic_cb_data, &diagnostic_constants[diagnostic_mode]))) {
          error = "Cannot allocate private selector diagnostic b5";
          return false;
        }
      }
      context->ClearRenderTargetView(output_rtv.Get(), clear_color.data());
      const D3D11_VIEWPORT viewport {
        0.0f,
        0.0f,
        static_cast<float>(packed_width),
        static_cast<float>(source_height),
        0.0f,
        1.0f,
      };
      if (direct_constants_buffer) {
        ID3D11Buffer *direct_constants[] = {direct_constants_buffer.Get()};
        context->PSSetConstantBuffers(4, 1, direct_constants);
      }
      const platf::dxgi::host_sbs_v2_draw_command_t draw_command {
        .render_targets = {output_rtv.Get(), nullptr},
        .render_target_count = 1u,
        .vertex_shader = vertex_shader.Get(),
        .pixel_shader = private_pixel_shader.Get(),
        .viewport = viewport,
        .sampler = sampler.Get(),
        .shader_resources = {
          color_srv.Get(),
          parallax_srv.Get(),
          state_srv.Get(),
          matte_srv.Get(),
          snapshot_srv.Get(),
          condition_srv.Get(),
          identity_srv.Get(),
        },
        .geometry_constants = geometry_buffer.Get(),
        .subtitle_constants = subtitle_buffer.Get(),
      };
      struct restore_private_b5_t {
        ID3D11DeviceContext *context;
        ComPtr<ID3D11Buffer> previous;
        explicit restore_private_b5_t(ID3D11DeviceContext *selected) : context(selected) {
          context->PSGetConstantBuffers(5u, 1u, previous.GetAddressOf());
        }
        ~restore_private_b5_t() {
          auto *value = previous.Get();
          context->PSSetConstantBuffers(5u, 1u, &value);
        }
      };
      bool submitted = true;
      for (std::size_t diagnostic_pass = 0u; diagnostic_pass < 2u; ++diagnostic_pass) {
        restore_private_b5_t restore_private_b5(context.Get());
        auto private_command = draw_command;
        private_command.render_target_count = 2u;
        private_command.render_targets = {
          diagnostic_pass == 0u ? output_rtv.Get() : repeat_colour_rtv.Get(),
          diagnostic_rtvs[diagnostic_pass].Get()};
        auto *diagnostic_cb = diagnostic_constants[diagnostic_pass].Get();
        context->PSSetConstantBuffers(5u, 1u, &diagnostic_cb);
        submitted = platf::dxgi::record_host_sbs_v2_draw(context.Get(), private_command) && submitted;
      }
      ID3D11Buffer *null_constants = nullptr;
      context->PSSetConstantBuffers(4u, 1u, &null_constants);
      if (!submitted) {
        error = "shared Host SBS V2 draw recorder rejected complete operands";
        return false;
      }

      std::array<ID3D11ShaderResourceView *, 7u> remaining_resources {};
      context->PSGetShaderResources(0u, static_cast<UINT>(remaining_resources.size()), remaining_resources.data());
      ID3D11Buffer *remaining_subtitle = nullptr;
      context->PSGetConstantBuffers(3u, 1u, &remaining_subtitle);
      bool unbound = remaining_subtitle == nullptr;
      if (remaining_subtitle) remaining_subtitle->Release();
      for (auto *resource : remaining_resources) {
        unbound = unbound && resource == nullptr;
        if (resource) resource->Release();
      }
      if (!unbound) {
        error = "shared renderer left native subtitle resources or b3 bound";
        return false;
      }

      if (gpu_output) *gpu_output = output_texture;
      if (!readback_output) { packed_output.clear(); return true; }

      auto staging_desc = output_desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging))) {
        error = "could not create the packed V2 staging texture";
        return false;
      }
      context->CopyResource(staging.Get(), output_texture.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        error = "could not read the packed V2 render result";
        return false;
      }
      const std::size_t row_bytes =
        static_cast<std::size_t>(packed_width) * output_bytes_per_pixel;
      packed_output.resize(row_bytes * source_height);
      for (UINT y = 0; y < source_height; ++y) {
        std::memcpy(
          packed_output.data() + static_cast<std::size_t>(y) * row_bytes,
          static_cast<const std::byte *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          row_bytes
        );
      }
      context->Unmap(staging.Get(), 0);
      for (std::size_t diagnostic_index = 0u; diagnostic_index < 2u; ++diagnostic_index) {
        auto diagnostic_staging_desc = diagnostic_desc;
        diagnostic_staging_desc.Usage = D3D11_USAGE_STAGING;
        diagnostic_staging_desc.BindFlags = 0u;
        diagnostic_staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> diagnostic_staging;
        if (FAILED(device->CreateTexture2D(&diagnostic_staging_desc, nullptr, &diagnostic_staging))) {
          error = "Cannot allocate private selector diagnostic staging";
          return false;
        }
        context->CopyResource(diagnostic_staging.Get(), diagnostic_textures[diagnostic_index].Get());
        D3D11_MAPPED_SUBRESOURCE diagnostic_mapped {};
        if (FAILED(context->Map(diagnostic_staging.Get(), 0u, D3D11_MAP_READ, 0u, &diagnostic_mapped))) {
          error = "Cannot read private selector diagnostics";
          return false;
        }
        const std::size_t diagnostic_row_bytes = std::size_t(packed_width) * 16u;
        diagnostic_output[diagnostic_index].resize(diagnostic_row_bytes * source_height);
        for (UINT diagnostic_row = 0u; diagnostic_row < source_height; ++diagnostic_row) {
          std::memcpy(diagnostic_output[diagnostic_index].data() + std::size_t(diagnostic_row) * diagnostic_row_bytes,
            static_cast<const std::byte *>(diagnostic_mapped.pData) + std::size_t(diagnostic_row) * diagnostic_mapped.RowPitch,
            diagnostic_row_bytes);
        }
        context->Unmap(diagnostic_staging.Get(), 0u);
      }
      context->CopyResource(staging.Get(), repeat_colour_texture.Get());
      D3D11_MAPPED_SUBRESOURCE repeat_mapped {};
      if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &repeat_mapped))) {
        error = "Cannot read repeated candidate colour";
        return false;
      }
      repeat_colour.resize(packed_output.size());
      for (UINT repeat_row = 0u; repeat_row < source_height; ++repeat_row) {
        std::memcpy(repeat_colour.data() + std::size_t(repeat_row) * row_bytes,
          static_cast<const std::byte *>(repeat_mapped.pData) + std::size_t(repeat_row) * repeat_mapped.RowPitch,
          row_bytes);
      }
      context->Unmap(staging.Get(), 0u);
      if (repeat_colour != packed_output) {
        error = "The two diagnostic draws produced different candidate colour bytes";
        return false;
      }
      return true;
    }


  public:
    bool create_immutable_texture_srv(
      const DXGI_FORMAT format,
      const UINT width,
      const UINT height,
      const UINT bytes_per_pixel,
      const void *pixels,
      ComPtr<ID3D11Texture2D> &texture,
      ComPtr<ID3D11ShaderResourceView> &srv
    ) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = 1u;
      desc.ArraySize = 1u;
      desc.Format = format;
      desc.SampleDesc.Count = 1u;
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA data {};
      data.pSysMem = pixels;
      data.SysMemPitch = width * bytes_per_pixel;
      return SUCCEEDED(device->CreateTexture2D(&desc, &data, &texture)) &&
             SUCCEEDED(device->CreateShaderResourceView(
               texture.Get(),
               nullptr,
               &srv
             ));
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11VertexShader> vertex_shader;
    ComPtr<ID3D11PixelShader> pixel_shader;
    ComPtr<ID3D11SamplerState> sampler;
    bool mapping_output = false;
    bool direct_replay = false;
    bool private_guard_configured = false;
    bool pending_scope=false;
    ComPtr<ID3D11Buffer> negative_scope_cb,current_scope_cb;
    ComPtr<ID3D11ShaderResourceView> negative_scope_srv,current_scope_srv;
    bool use_recorder_guard_scope = false;
    ComPtr<ID3D11PixelShader> recorder_flat_fallback_ps;
    bool recorder_flat_fallback_used = false;
    ComPtr<ID3D11Buffer> private_guard_cb, private_guard_packet;
    ComPtr<ID3D11ShaderResourceView> private_guard_srv;
  };

  models::depth_coordinate_v2::state_words_t make_live_state(
    const bool frame_valid,
    const bool known_contract
  ) {
    namespace v2 = models::depth_coordinate_v2;
    auto state = v2::state_initial_words;
    const float center_value = 0.25f;
    const float inverse_scale_value =
      1.0f / v2::model_calibrations.front().raw_coordinate_scale;
    state[v2::center] = std::bit_cast<std::uint32_t>(center_value);
    state[v2::inverse_scale] =
      std::bit_cast<std::uint32_t>(inverse_scale_value);
    state[v2::convergence_curve] =
      std::bit_cast<std::uint32_t>(v2::convergence_curve_default);
    state[v2::container_scale] = std::bit_cast<std::uint32_t>(1.0f);
    state[v2::calibration_revision] = 1u;
    state[v2::frame_valid] =
      std::bit_cast<std::uint32_t>(frame_valid ? 1.0f : 0.0f);
    state[v2::contract_tag_bits] =
      known_contract ? v2::contract_tag : (v2::contract_tag ^ 1u);
    state[v2::renderer_authorization_bits] = frame_valid ?
      (known_contract ? v2::contract_tag : (v2::contract_tag ^ 1u)) : 0u;
    state[v2::joint_plane_mode_bits] = v2::adaptive_policy_id;
    state[v2::mapping_state_reserved_2] = 0u;
    host_sbs_test::seal_adaptive_camera(state);
    return state;
  }

  subtitle_inputs_t make_subtitle_inputs(
    const UINT width, const UINT height,
    const models::depth_coordinate_v2::state_words_t &state,
    const float plane, const UINT field_width = 0u, const UINT field_height = 0u
  ) {
    namespace v2 = models::depth_coordinate_v2;
    subtitle_inputs_t input;
    input.width = width;
    input.height = height;
    input.constants = {1u, 0u, 7u, 0u, 0u, 0u, width, height};
    input.matte.resize(static_cast<std::size_t>(width) * height);
    input.snapshot.resize(input.matte.size(), {0u, 0u, 0u, 0x3c00u});
    input.condition = {v2::subtitle_condition_param_schema, v2::subtitle_condition_param_tag,
      1u, 0u, v2::subtitle_condition_param_fully_active_value, std::bit_cast<std::uint32_t>(plane)};
    auto &id = input.identity;
    id[0] = v2::subtitle_contour_schema;
    id[1] = v2::subtitle_contour_history_tag;
    id[2] = 1u;
    id[3] = state[v2::confirmed_cut_count];
    id[4] = 1u;
    id[6] = 7u;
    id[8] = width;
    id[9] = height;
    id[11] = height;
    id[12] = field_width == 0u ? width : field_width;
    id[13] = field_height == 0u ? height : field_height;
    id[15] = id[13];
    id[18] = id[12];
    id[19] = id[13];
    id[20] = state[v2::gain_last_observation_low];
    id[21] = state[v2::gain_last_observation_high];
    id[22] = v2::subtitle_contour_recipe;
    id[23] = 1u;
    return input;
  }


} // namespace

namespace {
  namespace pc=models::research_pending_current_colour_v1;
  namespace pr=models::research_pending_source_pattern_retirement_v1;
  namespace nr=models::research_source_pattern_retirement_v1;
  namespace cs=models::research_current_source_snapshot_v1;
  constexpr const char *pending_ps_closure="3fc3444621bc1539200790e7a3f3f08e87b318643cad1a3ed4eaab0bbf48d2fc";
  constexpr const char *pending_cs_closure="9c956a43074bd4f52b8a3bf865d8b487c3cfd5331070b693faf4da56a6c55cf2";
  void pc_require(bool value,const char *error){if(!value)throw std::runtime_error(error);}
  void pc_hr(HRESULT value){pc_require(SUCCEEDED(value),"Pending-colour fixture D3D failure");}
  struct word_buffer_t {ComPtr<ID3D11Buffer> buffer;ComPtr<ID3D11ShaderResourceView> srv;};
  word_buffer_t words(ID3D11Device *d,const std::vector<std::uint32_t> &v,bool raw=false,bool args=false) {
    D3D11_BUFFER_DESC b{};b.ByteWidth=static_cast<UINT>(v.size()*4u);b.Usage=D3D11_USAGE_DEFAULT;
    b.BindFlags=args?0u:D3D11_BIND_SHADER_RESOURCE;b.MiscFlags=args?D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS:raw?D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS:D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;b.StructureByteStride=(raw||args)?0u:4u;
    D3D11_SUBRESOURCE_DATA data{v.data(),0,0};word_buffer_t out;pc_hr(d->CreateBuffer(&b,&data,&out.buffer));
    if(!args){D3D11_SHADER_RESOURCE_VIEW_DESC view{};view.ViewDimension=raw?D3D11_SRV_DIMENSION_BUFFEREX:D3D11_SRV_DIMENSION_BUFFER;view.Format=raw?DXGI_FORMAT_R32_TYPELESS:DXGI_FORMAT_UNKNOWN;
      if(raw){view.BufferEx.NumElements=static_cast<UINT>(v.size());view.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;}else view.Buffer.NumElements=static_cast<UINT>(v.size());
      pc_hr(d->CreateShaderResourceView(out.buffer.Get(),&view,&out.srv));}return out;
  }
  template<class T>ComPtr<ID3D11Buffer> fixture_cb(ID3D11Device *d,const T &v){static_assert(sizeof(T)%16u==0u);D3D11_BUFFER_DESC b{};b.ByteWidth=sizeof(T);b.Usage=D3D11_USAGE_DEFAULT;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA data{&v,0,0};ComPtr<ID3D11Buffer> out;pc_hr(d->CreateBuffer(&b,&data,&out));return out;}
  std::vector<std::uint32_t> read_words(ID3D11Device *d,ID3D11DeviceContext *c,ID3D11Resource *r) {
    ComPtr<ID3D11Buffer>b;pc_hr(r->QueryInterface(IID_PPV_ARGS(&b)));D3D11_BUFFER_DESC desc{};b->GetDesc(&desc);desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=desc.MiscFlags=desc.StructureByteStride=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> stage;pc_hr(d->CreateBuffer(&desc,nullptr,&stage));c->CopyResource(stage.Get(),r);D3D11_MAPPED_SUBRESOURCE map{};pc_hr(c->Map(stage.Get(),0,D3D11_MAP_READ,0,&map));std::vector<std::uint32_t> v(desc.ByteWidth/4u);std::memcpy(v.data(),map.pData,desc.ByteWidth);c->Unmap(stage.Get(),0);return v;
  }
  std::vector<std::uint32_t> root_packet(const models::research_queued_render_guard_v1::expected_t &e,UINT w,UINT h){std::vector<std::uint32_t> p(64);p[0]=1;p[7]=0x52434f4f;p[1]=1u^0xd1ec15a5u^p[7];p[2]=p[8]=e.root[0];p[3]=p[9]=e.root[1];p[4]=p[10]=e.root[0]^0xa3756c91u;p[5]=p[11]=e.root[1]^0x5c8a936eu;p[6]=0x47524243u;p[12]=0x54535152u;p[13]=1;p[14]=1u^0x6f435257u;p[20]=p[21]=p[22]=1;p[24]=(w+15u)/16u;p[25]=(h+15u)/16u;p[26]=1;return p;}
  std::vector<std::uint32_t> source_receipt(const cs::constants_t &c,UINT inputW,UINT inputH){std::vector<std::uint32_t> r(64);r[0]=cs::tag;r[1]=r[2]=1;r[4]=c.source_rect[2];r[5]=c.band_field[1];r[6]=r[4]*r[5];std::memcpy(r.data()+8,&c,sizeof(c));r[40]=inputW;r[41]=inputH;r[42]=r[4];r[43]=r[5];r[45]=1;return r;}
  cs::constants_t source_context(UINT w=128u,UINT h=16u){cs::constants_t c;c.control={1u,cs::tag,1u,0u};c.source_rect={0,0,w,h};c.band_field={0,h,w,h};c.content={0,0,w,h};c.current={2,0,3000,0};c.scope={91,0,7,0};c.domain={7,0,31,0};return c;}
  bool selected_hardware(){const auto *h=std::getenv("SUNSHINE_TEST_PENDING_CURRENT_COLOUR_HARDWARE");return h&&std::string_view(h)=="1";}
  std::filesystem::path pending_root(){const auto *p=std::getenv("SUNSHINE_TEST_PENDING_CURRENT_COLOUR_SHADER_ROOT");return p?std::filesystem::path(p):std::filesystem::path{};}
  models::host_sbs_gpu_completion_receipt::expected_t owner_key(std::uint64_t frame=1u){models::host_sbs_gpu_completion_receipt::expected_t p;p.estimator_generation=11;p.publication_sequence=frame;p.frame_id=frame;p.observation_timestamp_us=1000u+frame;p.analysis_generation=7;p.domain_tag=31;p.transaction_token=0x12345000ull+frame;return p;}
  std::vector<std::uint32_t> negative_packet(const nr::expected_t &e,std::uint32_t status,std::uint32_t best,std::uint32_t cut){std::vector<std::uint32_t> n(64);n[0]=nr::tag;n[1]=1;n[2]=status;n[4]=8;n[5]=best;n[6]=255;n[7]=5;std::memcpy(n.data()+8,&e,sizeof(e));n[48]=cut;return n;}
  void configure_source_scope(live_v2_warp_fixture_t &f,const cs::constants_t &c,const nr::expected_t &e,const std::vector<std::uint32_t> &receipt,const std::vector<std::uint32_t> &negative){f.current_scope_cb=fixture_cb(f.device.Get(),c);f.negative_scope_cb=fixture_cb(f.device.Get(),e);f.current_scope_srv=words(f.device.Get(),receipt).srv;f.negative_scope_srv=words(f.device.Get(),negative).srv;f.pending_scope=true;}
}

TEST(PendingCurrentColourGpu, UnknownAndPartialMatchPreserveLegacyRetiredUsesIndependentBase) {
  const auto root=pending_root();if(root.empty())GTEST_SKIP()<<"Explicit private source root required";
  constexpr UINT W=128,H=16;constexpr std::uint64_t clock=1001;
  auto state=make_live_state(true,true);namespace v2=models::depth_coordinate_v2;
  state[v2::gain_last_observation_low]=state[v2::gain_seed_first_low]=state[v2::gain_seed_last_low]=static_cast<std::uint32_t>(clock);
  state[v2::gain_last_observation_high]=state[v2::gain_seed_first_high]=state[v2::gain_seed_last_high]=0;state[v2::camera_center_integrity_bits]=v2::camera_center_integrity_for_state_words(state);
  std::vector<rgba32f_t> current(W*H);for(UINT y=0;y<H;++y)for(UINT x=0;x<W;++x)current[y*W+x]={float(x)/W,float(y)/H,.2f,1};
  current[8*W+64]={1,1,1,1};const std::vector<float> base(W*H,-2.0f/W);auto ui=make_subtitle_inputs(W,H,state,2.5f/W);ui.matte[8*W+64]=1;ui.snapshot[8*W+64]={0x3c00,0x3c00,0x3c00,0x3c00};
  live_v2_warp_fixture_t legacy,flat,candidate;std::string error;
  ASSERT_TRUE(legacy.initialize(error,warp_shader_e::live_main,false,selected_hardware()))<<error;
  ASSERT_TRUE(flat.initialize(error,warp_shader_e::flat_main,false,selected_hardware()))<<error;
  ASSERT_TRUE(candidate.initialize(error,warp_shader_e::live_main,false,selected_hardware()))<<error;
  const auto root_expected=models::research_queued_render_guard_v1::make_expected(owner_key().transaction_token,1,clock);auto raw=root_packet(root_expected,W,H);
  ASSERT_TRUE(candidate.configure_private_guard(root,pending_ps_closure,&root_expected,&raw,error,"sbs_reprojection_v2_pending_current_colour_ps.hlsl"))<<error;
  const auto draw=[&](live_v2_warp_fixture_t &f,const subtitle_inputs_t *sub){std::vector<std::byte> out;pc_require(f.render(DXGI_FORMAT_R32G32B32A32_FLOAT,sizeof(rgba32f_t),W,H,current.data(),base,base,state,{},out,error,0,0,{},sub),error.c_str());return out;};
  const auto old=draw(legacy,&ui),scene=draw(legacy,nullptr),identity=draw(flat,nullptr);
  ASSERT_NE(old,scene);
  ASSERT_NE(scene,identity);
  for(const auto mode:{0u,1u,2u}) {
    auto c=source_context(W,H);c.control[3]=mode;ui.constants.color_mode=mode;const auto e=nr::make_expected(c,owner_key());const auto receipt=source_receipt(c,W,H);
    const auto current_legacy=draw(legacy,&ui);
    for(const auto status:{0u,1u}){SCOPED_TRACE("negative status "+std::to_string(status)+" colour "+std::to_string(mode));configure_source_scope(candidate,c,e,receipt,negative_packet(e,status,8,state[v2::confirmed_cut_count]));EXPECT_EQ(draw(candidate,&ui),current_legacy);}
    configure_source_scope(candidate,c,e,receipt,negative_packet(e,2u,0,state[v2::confirmed_cut_count]));EXPECT_EQ(draw(candidate,&ui),scene);
  }
  RecordProperty("negative_only_not_current_UI_or_EMPTY",true);RecordProperty("authored_packet_mechanics_not_model_receipt",true);RecordProperty("hardware",selected_hardware());
}

TEST(PendingCurrentColourGpu, InvalidRootOrCurrentSourceCannotUseGeometryOrUi) {
  const auto root=pending_root();if(root.empty())GTEST_SKIP()<<"Explicit private source root required";
  constexpr UINT W=128,H=16;auto state=make_live_state(true,true);namespace v2=models::depth_coordinate_v2;
  state[v2::gain_last_observation_low]=state[v2::gain_seed_first_low]=state[v2::gain_seed_last_low]=1001;state[v2::gain_last_observation_high]=state[v2::gain_seed_first_high]=state[v2::gain_seed_last_high]=0;state[v2::camera_center_integrity_bits]=v2::camera_center_integrity_for_state_words(state);
  std::vector<rgba32f_t> current(W*H);for(UINT y=0;y<H;++y)for(UINT x=0;x<W;++x)current[y*W+x]={float(x)/W,float(y)/H,.25f,1};const std::vector<float> base(W*H,-2.0f/W);auto ui=make_subtitle_inputs(W,H,state,2.5f/W);
  live_v2_warp_fixture_t flat,candidate;std::string error;
  ASSERT_TRUE(flat.initialize(error,warp_shader_e::flat_main,false,selected_hardware()))<<error;
  ASSERT_TRUE(candidate.initialize(error,warp_shader_e::live_main,false,selected_hardware()))<<error;
  const auto draw=[&](live_v2_warp_fixture_t &f){std::vector<std::byte> out;pc_require(f.render(DXGI_FORMAT_R32G32B32A32_FLOAT,sizeof(rgba32f_t),W,H,current.data(),base,base,state,{},out,error,0,0,{},&ui),error.c_str());return out;};
  const auto identity=draw(flat);const auto expected=models::research_queued_render_guard_v1::make_expected(owner_key().transaction_token,1,1001);const auto good=root_packet(expected,W,H);const auto c=source_context(W,H);const auto e=nr::make_expected(c,owner_key());const auto receipt=source_receipt(c,W,H);const auto n=negative_packet(e,0,8,state[v2::confirmed_cut_count]);
  for(UINT bad=0;bad<8;++bad){SCOPED_TRACE(bad);auto root_e=expected;auto raw=good;auto context=c;auto r=receipt;
    if(bad==0)raw[6]=0;if(bad==1)raw[25]^=1;if(bad==2)root_e.control[0]=0;if(bad==3)r[2]=0;if(bad==4)r[3]=2;if(bad==5)r[24]^=1;if(bad==6)context.scope[2]^=1;if(bad==7)context.control[3]=3;
    ASSERT_TRUE(candidate.configure_private_guard(root,pending_ps_closure,&root_e,&raw,error,"sbs_reprojection_v2_pending_current_colour_ps.hlsl"))<<error;
    configure_source_scope(candidate,context,e,r,n);EXPECT_EQ(draw(candidate),identity);
  }
  RecordProperty("all_rejects_compare_independent_GPU_flat_current",true);
}

namespace {
  struct pattern_owner_t {models::completed_ocr_conditioner_research_views_t warm;models::pending_ocr_conditioner_research_views_t pending;};
  ComPtr<ID3D11ShaderResourceView> half_band(live_v2_warp_fixture_t &f,const std::vector<rgba16_bits_t> &pixels){ComPtr<ID3D11Texture2D> t;ComPtr<ID3D11ShaderResourceView> v;pc_require(f.create_immutable_texture_srv(DXGI_FORMAT_R16G16B16A16_FLOAT,128,32,8,pixels.data(),t,v),"Pattern half source allocation failed");return v;}
  std::vector<rgba16_bits_t> stripe_band(){std::vector<rgba16_bits_t> v(128*32);for(UINT y=0;y<32;++y)for(UINT x=0;x<128;++x){const std::uint16_t q=x%4u<2u?0x3c00u:0u;v[y*128+x]={q,q,q,0x3c00u};}return v;}
  pattern_owner_t pattern_owner(live_v2_warp_fixture_t &f,std::uint64_t sequence,bool strong=true) {
    pattern_owner_t out;auto &w=out.warm;auto &p=out.pending;w.publication=owner_key(sequence);w.input_region.source_width=128;w.input_region.source_height=64;w.input_region.right=128;w.input_region.bottom=64;w.input_region.analysis_generation=7;w.input_region.authority=models::depth_analysis_authority_e::foreground_client;w.input_region.tensor_content={0,0,128,64};w.field_width=128;w.field_height=64;w.field_content={0,0,128,64};
    pc_require(w.input_region.valid(),"Authored pattern geometry invalid");
    std::array<std::uint32_t,24> cb{{128,64,32,32,128,64,32,64,0,0,128,64,static_cast<std::uint32_t>(sequence),0,7,0,static_cast<std::uint32_t>(w.publication.observation_timestamp_us),0,0,0,1,0,0,0}};
    std::vector<std::uint32_t> ocr(256);ocr[0]=4;ocr[1]=0x3852434f;ocr[2]=ocr[3]=ocr[4]=1;std::copy(cb.begin()+12,cb.begin()+16,ocr.begin()+5);std::copy(cb.begin(),cb.begin()+2,ocr.begin()+9);std::copy(cb.begin()+4,cb.begin()+8,ocr.begin()+11);ocr[16]=8;ocr[17]=36;ocr[18]=120;ocr[19]=60;
    std::vector<std::uint32_t> sct(32);sct[0]=5;sct[1]=0x35544353;sct[2]=1;std::copy(cb.begin()+12,cb.begin()+16,sct.begin()+4);std::copy(cb.begin(),cb.begin()+12,sct.begin()+8);sct[20]=cb[16];sct[21]=cb[17];sct[22]=27;sct[23]=1;
    const auto proof=models::research_queued_render_guard_v1::make_expected(w.publication.transaction_token,1,w.publication.observation_timestamp_us);const auto raw=root_packet(proof,128,64);
    w.source_band=half_band(f,stripe_band());w.probability_snapshot=words(f.device.Get(),std::vector<std::uint32_t>(960*160,strong?0x3f800000u:0u)).srv;w.ocr_record=words(f.device.Get(),ocr).srv;w.uip=words(f.device.Get(),std::vector<std::uint32_t>(40)).srv;w.sct=words(f.device.Get(),sct).srv;w.constants=fixture_cb(f.device.Get(),cb);w.transaction=words(f.device.Get(),raw,true).buffer;w.dispatch_arguments=words(f.device.Get(),raw,false,true).buffer;
    p.ticket_sequence=sequence;p.submitted_frame_id=sequence;p.observation_timestamp_us=w.publication.observation_timestamp_us;p.transaction_token=w.publication.transaction_token;p.subtitle_context_generation=7;p.expected_work=1;p.postprocess_request_flags=4;p.planned_owner=w.publication;p.input_region=w.input_region;p.field_width=w.field_width;p.field_height=w.field_height;p.field_content=w.field_content;p.source_band=w.source_band;p.probability_snapshot=w.probability_snapshot;p.ocr_record=w.ocr_record;p.uip=w.uip;p.sct=w.sct;p.constants=w.constants;p.transaction=w.transaction;p.dispatch_arguments=w.dispatch_arguments;
    D3D11_SHADER_RESOURCE_VIEW_DESC v{};v.Format=DXGI_FORMAT_R32_TYPELESS;v.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;v.BufferEx.NumElements=64;v.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;pc_hr(f.device->CreateShaderResourceView(p.transaction.Get(),&v,&p.transaction_raw));return out;
  }
  constexpr std::array<std::array<std::uint32_t,2>,256> authored_coordinates{{{8,5},{12,5},{17,5},{21,5},{24,5},{28,5},{29,5},{33,5},{36,5},{40,5},{45,5},{49,5},{52,5},{56,5},{57,5},{61,5},{64,5},{68,5},{73,5},{77,5},{80,5},{84,5},{85,5},{89,5},{92,5},{96,5},{101,5},{105,5},{108,5},{112,5},{113,5},{117,5},{8,8},{12,8},{17,8},{21,8},{24,8},{28,8},{29,8},{33,8},{36,8},{40,8},{45,8},{49,8},{52,8},{56,8},{57,8},{61,8},{64,8},{68,8},{73,8},{77,8},{80,8},{84,8},{85,8},{89,8},{92,8},{96,8},{101,8},{105,8},{108,8},{112,8},{113,8},{117,8},{8,11},{12,11},{17,11},{21,11},{24,11},{28,11},{29,11},{33,11},{36,11},{40,11},{45,11},{49,11},{52,11},{56,11},{57,11},{61,11},{64,11},{68,11},{73,11},{77,11},{80,11},{84,11},{85,11},{89,11},{92,11},{96,11},{101,11},{105,11},{108,11},{112,11},{113,11},{117,11},{8,14},{12,14},{17,14},{21,14},{24,14},{28,14},{29,14},{33,14},{36,14},{40,14},{45,14},{49,14},{52,14},{56,14},{57,14},{61,14},{64,14},{68,14},{73,14},{77,14},{80,14},{84,14},{85,14},{89,14},{92,14},{96,14},{101,14},{105,14},{108,14},{112,14},{113,14},{117,14},{8,17},{12,17},{17,17},{21,17},{24,17},{28,17},{29,17},{33,17},{36,17},{40,17},{45,17},{49,17},{52,17},{56,17},{57,17},{61,17},{64,17},{68,17},{73,17},{77,17},{80,17},{84,17},{85,17},{89,17},{92,17},{96,17},{101,17},{105,17},{108,17},{112,17},{113,17},{117,17},{8,20},{12,20},{17,20},{21,20},{24,20},{28,20},{29,20},{33,20},{36,20},{40,20},{45,20},{49,20},{52,20},{56,20},{57,20},{61,20},{64,20},{68,20},{73,20},{77,20},{80,20},{84,20},{85,20},{89,20},{92,20},{96,20},{101,20},{105,20},{108,20},{112,20},{113,20},{117,20},{8,23},{12,23},{17,23},{21,23},{24,23},{28,23},{29,23},{33,23},{36,23},{40,23},{45,23},{49,23},{52,23},{56,23},{57,23},{61,23},{64,23},{68,23},{73,23},{77,23},{80,23},{84,23},{85,23},{89,23},{92,23},{96,23},{101,23},{105,23},{108,23},{112,23},{113,23},{117,23},{8,26},{12,26},{17,26},{21,26},{24,26},{28,26},{29,26},{33,26},{36,26},{40,26},{45,26},{49,26},{52,26},{56,26},{57,26},{61,26},{64,26},{68,26},{73,26},{77,26},{80,26},{84,26},{85,26},{89,26},{92,26},{96,26},{101,26},{105,26},{108,26},{112,26},{113,26},{117,26}}};
  constexpr std::array<std::uint32_t,256> authored_grid_indices{{0u,2u,5u,7u,9u,11u,12u,14u,16u,18u,21u,23u,25u,27u,28u,30u,32u,34u,37u,39u,41u,43u,44u,46u,48u,50u,53u,55u,57u,59u,60u,62u,64u,66u,69u,71u,73u,75u,76u,78u,80u,82u,85u,87u,89u,91u,92u,94u,96u,98u,101u,103u,105u,107u,108u,110u,112u,114u,117u,119u,121u,123u,124u,126u,128u,130u,133u,135u,137u,139u,140u,142u,144u,146u,149u,151u,153u,155u,156u,158u,160u,162u,165u,167u,169u,171u,172u,174u,176u,178u,181u,183u,185u,187u,188u,190u,192u,194u,197u,199u,201u,203u,204u,206u,208u,210u,213u,215u,217u,219u,220u,222u,224u,226u,229u,231u,233u,235u,236u,238u,240u,242u,245u,247u,249u,251u,252u,254u,256u,258u,261u,263u,265u,267u,268u,270u,272u,274u,277u,279u,281u,283u,284u,286u,288u,290u,293u,295u,297u,299u,300u,302u,304u,306u,309u,311u,313u,315u,316u,318u,320u,322u,325u,327u,329u,331u,332u,334u,336u,338u,341u,343u,345u,347u,348u,350u,352u,354u,357u,359u,361u,363u,364u,366u,368u,370u,373u,375u,377u,379u,380u,382u,384u,386u,389u,391u,393u,395u,396u,398u,400u,402u,405u,407u,409u,411u,412u,414u,416u,418u,421u,423u,425u,427u,428u,430u,432u,434u,437u,439u,441u,443u,444u,446u,448u,450u,453u,455u,457u,459u,460u,462u,464u,466u,469u,471u,473u,475u,476u,478u,480u,482u,485u,487u,489u,491u,492u,494u,496u,498u,501u,503u,505u,507u,508u,510u}};
  void expect_authored_owner(live_v2_warp_fixture_t &f,pr::resources_t &helper,UINT bank) {
    const auto resources=helper.export_resources(bank);const auto owner=read_words(f.device.Get(),f.context.Get(),resources[8].Get());
    ASSERT_EQ(owner.size(),256u);
    EXPECT_EQ(owner[3],0u);EXPECT_EQ(owner[4],1u);EXPECT_EQ(owner[5],256u);EXPECT_EQ(owner[6],255u);EXPECT_EQ(owner[7],8u);EXPECT_EQ(owner[8],1u);EXPECT_EQ(owner[81],1u);
    const std::array<std::uint32_t,8> mode{{65535,65535,65535,65535,256,0,63,0}};
    for(UINT i=0;i<8;++i)EXPECT_EQ(owner[96+i],mode[i]);
    const auto anchors=read_words(f.device.Get(),f.context.Get(),resources[9].Get());
    ASSERT_EQ(anchors.size(),4096u);
    for(UINT i=0;i<256;++i){EXPECT_EQ(anchors[8*i],authored_coordinates[i][0]);EXPECT_EQ(anchors[8*i+1],authored_coordinates[i][1]);for(UINT c=2;c<5;++c)EXPECT_EQ(anchors[8*i+c],0x3f800000u);EXPECT_EQ(anchors[8*i+5],authored_grid_indices[i]);EXPECT_EQ(anchors[8*i+6],0u);EXPECT_EQ(anchors[8*i+7],0u);}
    for(UINT i=256u*8u;i<4096u;++i)EXPECT_EQ(anchors[i],0u);
  }
  struct pattern_runtime_t {
    std::array<ComPtr<ID3D11ComputeShader>,8> shaders;pr::resources_t helper;
    void prepare(live_v2_warp_fixture_t &f,const models::completed_ocr_conditioner_research_views_t &warm){std::array<models::host_sbs_shader_cache::shader_spec,8> specs;for(UINT i=0;i<8;++i)specs[i]={"host_sbs_pending_source_pattern_retirement_cs.hlsl",pr::entries[i],"cs_5_0"};const auto sources=models::host_sbs_shader_cache::snapshot_sources(pending_root(),specs);pc_require(models::host_sbs_shader_cache::source_closure_sha256(sources)==pending_cs_closure,"Pattern CS source closure mismatch");pr::shader_set_t set;for(UINT i=0;i<8;++i){const auto code=models::host_sbs_shader_cache::get(sources,specs[i]);pc_require(bool(code),"Pattern CS bytecode missing");pc_hr(f.device->CreateComputeShader(code->data(),code->size(),nullptr,&shaders[i]));set.shaders[i]=shaders[i].Get();}std::string error;pc_require(helper.prepare(f.device.Get(),f.context.Get(),warm,set,error),error.c_str());}
    std::optional<UINT> capture(live_v2_warp_fixture_t &f,const pattern_owner_t &owner,const cs::constants_t &c,std::string &error){return helper.capture_pending_owner(f.context.Get(),owner.pending,nr::make_expected(c,owner.pending.planned_owner),error);}
    std::vector<std::uint32_t> observe(live_v2_warp_fixture_t &f,UINT bank,const pattern_owner_t &owner,const cs::constants_t &c,const std::vector<rgba16_bits_t> &pixels){const auto source=half_band(f,pixels);const auto receipt=words(f.device.Get(),source_receipt(c,128,64));const auto cb=fixture_cb(f.device.Get(),c);std::string error;pc_require(helper.observe(bank,f.context.Get(),source.Get(),receipt.srv.Get(),cb.Get(),nr::make_expected(c,owner.pending.planned_owner),error),error.c_str());return read_words(f.device.Get(),f.context.Get(),helper.export_resources(bank)[12].Get());}
  };
  cs::constants_t pattern_current(std::uint64_t serial=2,std::uint64_t clock=3000){auto c=source_context(128,64);c.band_field={32,32,128,64};c.current={static_cast<std::uint32_t>(serial),static_cast<std::uint32_t>(serial>>32),static_cast<std::uint32_t>(clock),static_cast<std::uint32_t>(clock>>32)};return c;}
}

TEST(PendingCurrentColourGpu, ActualPartialPatternDoesNotRetireAndInsufficientAnchorsRemainUnknown) {
  if(pending_root().empty())GTEST_SKIP()<<"Explicit private source root required";
  live_v2_warp_fixture_t f;std::string error;
  ASSERT_TRUE(f.initialize(error,warp_shader_e::live_main,false,selected_hardware()))<<error;
  const auto owner=pattern_owner(f,1);pattern_runtime_t rt;rt.prepare(f,owner.warm);auto c=pattern_current();const auto bank=rt.capture(f,owner,c,error);
  ASSERT_TRUE(bank)<<error;
  ASSERT_NO_FATAL_FAILURE(expect_authored_owner(f,rt.helper,*bank));
  auto partial=stripe_band();for(UINT y=0;y<32;++y)for(UINT x=0;x<48;++x)partial[y*128+x]={0x3800,0x3800,0x3800,0x3c00};const auto a=rt.observe(f,*bank,owner,c,partial);
  ASSERT_EQ(a[3],0u);
  ASSERT_EQ(a[2],1u)<<"Partial/shared pattern must preserve old overlay capability";
  EXPECT_EQ(a[4],256u);EXPECT_EQ(a[5],168u);
  const std::array<std::uint32_t,25> expected_scores{{0,88,168,80,0,0,88,168,80,0,0,88,168,80,0,0,88,168,80,0,0,88,168,80,0}};
  const auto scores=read_words(f.device.Get(),f.context.Get(),rt.helper.export_resources(*bank)[11].Get());
  for(UINT i=0;i<25;++i)EXPECT_EQ(scores[i],expected_scores[i]);
  ASSERT_GE(2u*a[5],a[4]);
  const auto weak=pattern_owner(f,2,false);const auto weak_bank=rt.capture(f,weak,c,error);
  ASSERT_TRUE(weak_bank)<<error;
  const auto b=rt.observe(f,*weak_bank,weak,c,partial);
  EXPECT_EQ(b[3],0u);
  EXPECT_EQ(b[2],0u)<<"Insufficient own anchors are UNKNOWN, never EMPTY/retired";
  EXPECT_EQ(b[4],0u);
  RecordProperty("fixed_7227_math_authored_mechanics_not_current_glyph_or_EMPTY",true);
}

TEST(PendingCurrentColourGpu, ThreeOwnersRecycleTwoBanksWithoutRearmingOrResettingLatestColour) {
  if(pending_root().empty())GTEST_SKIP()<<"Explicit private source root required";
  live_v2_warp_fixture_t f;std::string error;
  ASSERT_TRUE(f.initialize(error,warp_shader_e::live_main,false,selected_hardware()))<<error;
  const std::array owners{pattern_owner(f,1),pattern_owner(f,2),pattern_owner(f,3)};pattern_runtime_t rt;rt.prepare(f,owners[0].warm);const auto c=pattern_current();const std::vector<rgba16_bits_t> current(128*32,{0x3800,0x3800,0x3800,0x3c00});std::vector<std::uint32_t> watermark;
  UINT last_bank=99;
  for(UINT i=0;i<3;++i){const auto bank=rt.capture(f,owners[i],c,error);
    ASSERT_TRUE(bank)<<error;
    ASSERT_NO_FATAL_FAILURE(expect_authored_owner(f,rt.helper,*bank));
    EXPECT_EQ(*bank,i%2u);last_bank=*bank;const auto n=rt.observe(f,*bank,owners[i],c,current);
    ASSERT_EQ(n[3],0u);
    ASSERT_EQ(n[2],2u);
    EXPECT_EQ(n[4],256u);EXPECT_EQ(n[5],0u);
    const auto mark=read_words(f.device.Get(),f.context.Get(),rt.helper.export_resources(*bank)[14].Get());
    if(i==0)watermark=mark;else EXPECT_EQ(mark,watermark)<<"Same retained C evaluated once for each NEW owner without invented age";
  }
  const auto before=read_words(f.device.Get(),f.context.Get(),rt.helper.export_resources(last_bank)[8].Get());
  ASSERT_EQ(before[9],1u);
  const auto same=rt.capture(f,owners[2],c,error);
  ASSERT_TRUE(same)<<error;
  EXPECT_EQ(*same,last_bank);
  EXPECT_EQ(read_words(f.device.Get(),f.context.Get(),rt.helper.export_resources(last_bank)[8].Get()),before)<<"Same owner capture must not rebuild or clear latch";
  EXPECT_FALSE(rt.capture(f,owners[0],c,error))<<"Overwritten old owner cannot resurrect";
  EXPECT_EQ(read_words(f.device.Get(),f.context.Get(),rt.helper.export_resources(last_bank)[8].Get()),before);
  EXPECT_EQ(read_words(f.device.Get(),f.context.Get(),rt.helper.export_resources(last_bank)[14].Get()),watermark);
}

TEST(PendingCurrentColourGpu, ChangedDomainIsUnknownAndCannotAdvanceRetirementWatermark) {
  if(pending_root().empty())GTEST_SKIP()<<"Explicit private source root required";
  live_v2_warp_fixture_t f;std::string error;
  ASSERT_TRUE(f.initialize(error,warp_shader_e::live_main,false,selected_hardware()))<<error;
  const auto owner=pattern_owner(f,1);pattern_runtime_t rt;rt.prepare(f,owner.warm);auto c=pattern_current();const auto bank=rt.capture(f,owner,c,error);
  ASSERT_TRUE(bank)<<error;
  const auto current=stripe_band();const auto good=rt.observe(f,*bank,owner,c,current);
  ASSERT_EQ(good[3],0u);
  const auto before=read_words(f.device.Get(),f.context.Get(),rt.helper.export_resources(*bank)[14].Get());
  c.scope[2]^=1u;c.current[0]=3;c.current[2]=4000;const auto bad=rt.observe(f,*bank,owner,c,current);
  EXPECT_EQ(bad[2],0u);
  EXPECT_NE(bad[3],0u);
  EXPECT_EQ(read_words(f.device.Get(),f.context.Get(),rt.helper.export_resources(*bank)[14].Get()),before);
}
TEST(PendingCurrentColourGpu, NonzeroRoiUsesExactCurrentRectangleAndPreservesOutsideScene) {
  const auto root=pending_root();if(root.empty())GTEST_SKIP()<<"Explicit private source root required";
  constexpr UINT W=256,H=32,RW=128,RH=16;auto state=make_live_state(true,true);namespace v2=models::depth_coordinate_v2;
  state[v2::gain_last_observation_low]=state[v2::gain_seed_first_low]=state[v2::gain_seed_last_low]=1001;state[v2::gain_last_observation_high]=state[v2::gain_seed_first_high]=state[v2::gain_seed_last_high]=0;state[v2::camera_center_integrity_bits]=v2::camera_center_integrity_for_state_words(state);
  auto ui=make_subtitle_inputs(RW,RH,state,2.5f/RW,RW,RH);ui.constants.source_left=16;ui.constants.source_top=8;ui.matte[8*RW+64]=1;ui.snapshot[8*RW+64]={0x3c00,0x3c00,0x3c00,0x3c00};
  models::host_sbs_v2_geometry_t geometry;geometry.video_roi_active=1;geometry.video_roi_left=16.0f/W;geometry.video_roi_top=8.0f/H;geometry.video_roi_right=144.0f/W;geometry.video_roi_bottom=24.0f/H;geometry.tensor_content_left=geometry.tensor_content_top=0;geometry.tensor_content_right=RW;geometry.tensor_content_bottom=RH;
  std::vector<rgba32f_t> current(W*H);for(UINT y=0;y<H;++y)for(UINT x=0;x<W;++x)current[y*W+x]={float(x)/W,float(y)/H,.2f,1};current[16*W+80]={1,1,1,1};const std::vector<float> base(RW*RH,-2.0f/RW);
  live_v2_warp_fixture_t legacy,candidate;std::string error;
  ASSERT_TRUE(legacy.initialize(error,warp_shader_e::live_main,false,selected_hardware()))<<error;
  ASSERT_TRUE(candidate.initialize(error,warp_shader_e::live_main,false,selected_hardware()))<<error;
  const auto expected=models::research_queued_render_guard_v1::make_expected(owner_key().transaction_token,1,1001);const auto raw=root_packet(expected,RW,RH);
  ASSERT_TRUE(candidate.configure_private_guard(root,pending_ps_closure,&expected,&raw,error,"sbs_reprojection_v2_pending_current_colour_ps.hlsl"))<<error;
  auto c=source_context(RW,RH);c.source_rect={16,8,RW,RH};const auto e=nr::make_expected(c,owner_key());const auto receipt=source_receipt(c,W,H);
  const auto draw=[&](live_v2_warp_fixture_t &f,const subtitle_inputs_t *sub){std::vector<std::byte> out;pc_require(f.render(DXGI_FORMAT_R32G32B32A32_FLOAT,sizeof(rgba32f_t),W,H,current.data(),base,base,state,{},out,error,RW,RH,geometry,sub),error.c_str());return out;};
  const auto old=draw(legacy,&ui),scene=draw(legacy,nullptr);
  ASSERT_NE(old,scene);
  configure_source_scope(candidate,c,e,receipt,negative_packet(e,1,8,state[v2::confirmed_cut_count]));EXPECT_EQ(draw(candidate,&ui),old);
  configure_source_scope(candidate,c,e,receipt,negative_packet(e,2,0,state[v2::confirmed_cut_count]));EXPECT_EQ(draw(candidate,&ui),scene);
  RecordProperty("nonzero_ROI_current_source_not_restamped_full_source",true);
}

#include "pending_current_colour_roi_join_cases.inc"

#endif // _WIN32
