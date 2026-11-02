// EXTERNAL PREPARED TEST: not compiled or executed.
// Exact existing GPU fixture extraction; no CPU warp or model inference.
/**
 * @file tests/unit/test_host_sbs_v2_live_warp.cpp
 * @brief Executable D3D11-WARP checks for the authenticated Host SBS V2 live renderer.
 */
#include <tests/tests_common.h>
#include <src/video_depth_estimator.h>

#ifdef _WIN32

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <src/depth_coordinate_v2.h>
#include <src/crypto.h>
#include <src/host_sbs_shader_cache.h>
#include <src/host_sbs_v2_geometry.h>
#include <src/host_sbs_subtitle_render.h>
#include <src/platform/windows/host_sbs_v2_renderer.h>
#include <tests/fixtures/host_sbs_adaptive_camera_state.h>

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
      const subtitle_inputs_t *subtitle = nullptr
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
      output_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
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
      };
      const bool submitted = platf::dxgi::record_host_sbs_v2_draw(context.Get(), draw_command);
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

  private:
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


  struct subtitle_trajectory_t {
    static constexpr UINT width = 128u, height = 8u, row = 4u, a_x = 56u, b_x = 72u;
    models::depth_coordinate_v2::state_words_t state = make_live_state(true, true);
    std::vector<float> base = std::vector<float>(width * height, -2.0f / width);
    std::vector<rgba32f_t> empty = std::vector<rgba32f_t>(width * height);
    std::vector<rgba32f_t> a, b;
    subtitle_inputs_t subtitle_a = make_subtitle_inputs(width, height, state, 2.0f / width);
    live_v2_warp_fixture_t renderer;
    std::string error;
    bool initialize() {
      const char *hardware = std::getenv("SUNSHINE_TEST_PRESENT_TRAJECTORY_HARDWARE");
      if (!renderer.initialize(error, warp_shader_e::live_main, false,
              hardware && std::string_view {hardware} == "1")) return false;
      for (UINT y = 0u; y < height; ++y) {
        for (UINT x = 0u; x < width; ++x) {
          empty[y * width + x] = {float(x) / 512.0f, float(y) / 64.0f, 0.125f, 1.0f};
        }
      }
      a = empty;
      b = empty;
      a[row * width + a_x] = {1.0f, 1.0f, 1.0f, 1.0f};
      b[row * width + b_x] = {0.0f, 1.0f, 1.0f, 1.0f};
      subtitle_a.identity[4] = 17u;
      subtitle_a.matte[row * width + a_x] = 1u;
      subtitle_a.snapshot[row * width + a_x] = {0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u};
      return true;
    }
    bool draw(const std::vector<rgba32f_t> &rgb, const subtitle_inputs_t *subtitle,
              std::vector<std::byte> &out) {
      return renderer.render(DXGI_FORMAT_R32G32B32A32_FLOAT, sizeof(rgba32f_t),
        width, height, rgb.data(), base, base, state, {}, out, error,
        0u, 0u, {}, subtitle);
    }
  };

  std::size_t gpu_white_pixels(const std::vector<std::byte> &bytes) {
    std::size_t count = 0u;
    for (std::size_t offset = 0u; offset + sizeof(rgba32f_t) <= bytes.size(); offset += sizeof(rgba32f_t)) {
      rgba32f_t p {};
      std::memcpy(&p, bytes.data() + offset, sizeof(p));
      if (p.r == 1.0f && p.g == 1.0f && p.b == 1.0f) ++count;
    }
    return count;
  }
}  // namespace

TEST(HostSbsSubtitlePresentTrajectoryGpuTest, HeldTupleUsesActualCurrentBAndEmptyPixels) {
  subtitle_trajectory_t t;
  ASSERT_TRUE(t.initialize()) << t.error;
  const auto identity_before = t.subtitle_a.identity;
  const auto matte_before = t.subtitle_a.matte;
  const auto source_before = t.subtitle_a.snapshot;
  std::vector<std::byte> a_ui, a_base, b_held, b_base, empty_held, empty_base;
  ASSERT_TRUE(t.draw(t.a, &t.subtitle_a, a_ui)) << t.error;
  ASSERT_TRUE(t.draw(t.a, nullptr, a_base)) << t.error;
  ASSERT_NE(a_ui, a_base) << "A must actually exercise native UI selection.";
  ASSERT_GT(gpu_white_pixels(a_ui), 0u);
  ASSERT_TRUE(t.draw(t.b, &t.subtitle_a, b_held)) << t.error;
  ASSERT_TRUE(t.draw(t.b, nullptr, b_base)) << t.error;
  EXPECT_EQ(b_held, b_base) << "Held A cannot borrow age or move new B.";
  EXPECT_EQ(gpu_white_pixels(b_held), 0u);
  ASSERT_TRUE(t.draw(t.empty, &t.subtitle_a, empty_held)) << t.error;
  ASSERT_TRUE(t.draw(t.empty, nullptr, empty_base)) << t.error;
  EXPECT_EQ(empty_held, empty_base) << "Current empty RGB revokes old A at the real pixel shader.";
  EXPECT_EQ(gpu_white_pixels(empty_held), 0u);
  EXPECT_EQ(t.subtitle_a.identity, identity_before);
  EXPECT_EQ(t.subtitle_a.matte, matte_before);
  EXPECT_EQ(t.subtitle_a.snapshot, source_before);
}

TEST(HostSbsSubtitlePresentTrajectoryGpuTest, SharedPackedPolicyRetainsGpuAWhileChangedSourceHasNoCompletion) {
  using namespace std::chrono_literals;
  subtitle_trajectory_t t;
  ASSERT_TRUE(t.initialize()) << t.error;
  std::vector<std::byte> packed_a, b_current, empty_current;
  ASSERT_TRUE(t.draw(t.a, &t.subtitle_a, packed_a)) << t.error;
  ASSERT_TRUE(t.draw(t.b, &t.subtitle_a, b_current)) << t.error;
  ASSERT_TRUE(t.draw(t.empty, &t.subtitle_a, empty_current)) << t.error;
  ASSERT_NE(packed_a, b_current);
  ASSERT_NE(packed_a, empty_current);
  // These are authored policy observations, not measured elapsed GPU time and not a CUDA mock.
  // Pixel values on both sides are genuine executions of the pinned production renderer.
  for (const auto coded_age : {1ms, 33ms, models::host_sbs_v2_max_matched_repeat_age}) {
    SCOPED_TRACE(coded_age.count());
    const bool repeat = models::host_sbs_should_repeat_packed_presentation(
      models::host_sbs_renderer_e::parallax_v2, false, true, true, coded_age, false);
    ASSERT_TRUE(repeat);
    for (const auto *current : {&b_current, &empty_current}) {
      const auto &selected_gpu_bytes = repeat ? packed_a : *current;
      EXPECT_EQ(selected_gpu_bytes, packed_a);
      EXPECT_GT(gpu_white_pixels(selected_gpu_bytes), 0u);
      EXPECT_EQ(gpu_white_pixels(*current), 0u);
    }
  }
  EXPECT_FALSE(models::host_sbs_should_repeat_packed_presentation(
    models::host_sbs_renderer_e::parallax_v2, false, true, true,
    models::host_sbs_v2_max_matched_repeat_age + 1ms, false));
  EXPECT_FALSE(models::host_sbs_should_repeat_packed_presentation(
    models::host_sbs_renderer_e::parallax_v2, false, true, false, 1ms, false));
}

TEST(HostSbsSubtitlePresentTrajectoryGpuTest, FreshCurrentFlatClearsPackedAWithoutRestampingTheOwner) {
  subtitle_trajectory_t t;
  ASSERT_TRUE(t.initialize()) << t.error;
  live_v2_warp_fixture_t flat;
  ASSERT_TRUE(flat.initialize(t.error, warp_shader_e::flat_main)) << t.error;
  std::vector<std::byte> packed_a, flat_b, flat_empty;
  ASSERT_TRUE(t.draw(t.a, &t.subtitle_a, packed_a)) << t.error;
  const auto old_clock = std::array {t.subtitle_a.identity[20], t.subtitle_a.identity[21]};
  for (const auto *rgb : {&t.b, &t.empty}) {
    auto &out = rgb == &t.b ? flat_b : flat_empty;
    ASSERT_TRUE(flat.render(DXGI_FORMAT_R32G32B32A32_FLOAT, sizeof(rgba32f_t),
      t.width, t.height, rgb->data(), t.base, t.base, t.state, {}, out, t.error,
      0u, 0u, {}, &t.subtitle_a)) << t.error;
    EXPECT_NE(out, packed_a);
    EXPECT_EQ(gpu_white_pixels(out), 0u);
  }
  EXPECT_EQ((std::array {t.subtitle_a.identity[20], t.subtitle_a.identity[21]}), old_clock);
}

TEST(HostSbsSubtitlePresentTrajectoryGpuTest, CurrentUiClockCannotBorrowCachedBaseObservation) {
  subtitle_trajectory_t t;
  ASSERT_TRUE(t.initialize()) << t.error;
  auto fresh_b = t.subtitle_a;
  fresh_b.identity[4] = 18u;
  fresh_b.identity[20] ^= 1u;
  std::fill(fresh_b.matte.begin(), fresh_b.matte.end(), 0u);
  fresh_b.matte[t.row * t.width + t.b_x] = 1u;
  fresh_b.snapshot[t.row * t.width + t.b_x] = {0u, 0x3c00u, 0x3c00u, 0x3c00u};
  std::vector<std::byte> rejected, base;
  ASSERT_TRUE(t.draw(t.b, &fresh_b, rejected)) << t.error;
  ASSERT_TRUE(t.draw(t.b, nullptr, base)) << t.error;
  EXPECT_EQ(rejected, base) << "A separately-clocked UI needs an explicit UI/Base join, never a forged old clock.";
}

TEST(HostSbsSubtitlePresentTrajectoryGpuTest, ExactUnchangedSourceMayKeepItsGpu3DWithoutAnAgeExpiry) {
  using namespace std::chrono_literals;
  subtitle_trajectory_t t;
  ASSERT_TRUE(t.initialize()) << t.error;
  std::vector<std::byte> first, current_same, base;
  ASSERT_TRUE(t.draw(t.a, &t.subtitle_a, first)) << t.error;
  ASSERT_TRUE(t.draw(t.a, &t.subtitle_a, current_same)) << t.error;
  ASSERT_TRUE(t.draw(t.a, nullptr, base)) << t.error;
  EXPECT_EQ(first, current_same);
  EXPECT_NE(current_same, base);
  EXPECT_TRUE(models::host_sbs_should_repeat_packed_presentation(
    models::host_sbs_renderer_e::parallax_v2, false, true, true, 1h, true));
}

#endif  // _WIN32
