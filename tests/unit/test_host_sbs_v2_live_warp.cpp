/**
 * @file tests/unit/test_host_sbs_v2_live_warp.cpp
 * @brief Executable D3D11-WARP checks for the authenticated Host SBS V2 live renderer.
 */
#include "../tests_common.h"

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
  #include <d3d11shader.h>
  #include <filesystem>
  #include <fstream>
  #include <iterator>
  #include <limits>
  #include <map>
  #include <nlohmann/json.hpp>
  #include <src/crypto.h>
  #include <src/depth_coordinate_v2.h>
  #include <src/host_sbs_current_outline_runtime_programs.h>
  #include <src/host_sbs_queued_render_guard.h>
  #include <src/host_sbs_current_outline_selected_matte_gpu.h>
  #include <src/host_sbs_shared_ui_warp_runtime.h>
  #include <src/host_sbs_shader_cache.h>
  #include <src/host_sbs_subtitle_render.h>
  #include <src/host_sbs_v2_geometry.h>
  #include <src/platform/windows/host_sbs_v2_renderer.h>
  #include "subtitle_helper_source_pins.h"
  #include <sstream>
  #include <stdexcept>
  #include <string>
  #include <string_view>
  #include <type_traits>
  #include <tests/fixtures/host_sbs_adaptive_camera_state.h>
  #include <utility>
  #include <vector>
  #include <wrl/client.h>

class HostSbsV2LiveWarpGpuTest_SharedUiPinProductionOwnerRestoresContextAndRejectsMissingInputs_Test;
class HostSbsV2LiveWarpGpuTest_SharedUiPinProductionRejectsInvalidRowExtentAndDiagnosticDevice_Test;
class HostSbsV2LiveWarpGpuTest_SharedUiPinQueuedProductionRequiresCurrentGuardAtPrepareAndDraw_Test;

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
    selected_outline,
    selected_outline_mapping,
    selected_outline_boundary,
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

  struct saved_background_inputs_t {
    std::vector<rgba32f_t> pixels;
    std::array<std::uint32_t, 96u> proof {};
    std::array<std::uint32_t, 16u> expected {};
  };

  struct shared_ui_pin_diagnostics_t {
    std::array<UINT, 4u> shape {}; // full W/H, band top/height
    std::vector<float> field;
    std::vector<float> authorized_weights;
    std::vector<std::uint32_t> active_rows;
    bool selected_receipt_authorized = false;
    std::array<std::uint32_t,1u> selected_receipt_header {};
    std::vector<std::byte> mapping_rgba32f;
    std::vector<std::byte> ui_coverage_rgba32f;
  };

  struct current_outline_inputs_t {
    std::vector<std::uint32_t> matte;
    std::array<std::uint32_t, 64u> receipt {};
    bool shared_ui_pin = false;
    bool production_shared_ui_pin = false;
    bool production_shared_ui_pin_queued = false;
    ID3D11ShaderResourceView *shared_ui_pin_draw_guard_override = nullptr;
    bool shared_ui_pin_resample_only = false;
    shared_ui_pin_diagnostics_t *shared_ui_pin_diagnostic_output = nullptr;
    double *render_gpu_ms = nullptr;
  };

  class live_v2_warp_fixture_t {
    friend class ::HostSbsV2LiveWarpGpuTest_SharedUiPinProductionOwnerRestoresContextAndRejectsMissingInputs_Test;
    friend class ::HostSbsV2LiveWarpGpuTest_SharedUiPinProductionRejectsInvalidRowExtentAndDiagnosticDevice_Test;
    friend class ::HostSbsV2LiveWarpGpuTest_SharedUiPinQueuedProductionRequiresCurrentGuardAtPrepareAndDraw_Test;
  public:
    bool initialize(
      std::string &error,
      const warp_shader_e shader = warp_shader_e::live_main,
      const bool fixed_eleven_reference = false,
      const bool hardware = false
    ) {
      if (const auto *cache_root = std::getenv("SUNSHINE_TEST_HOST_SBS_SHADER_CACHE")) {
        models::host_sbs_shader_cache::configure_persistent_cache(cache_root);
      }
      mapping_output =
        shader == warp_shader_e::live_mapping ||
        shader == warp_shader_e::direct_mapping ||
        shader == warp_shader_e::selected_outline_mapping;
      direct_replay =
        shader == warp_shader_e::direct_main ||
        shader == warp_shader_e::direct_mapping;
      selected_outline = shader == warp_shader_e::selected_outline ||
        shader == warp_shader_e::selected_outline_mapping || shader == warp_shader_e::selected_outline_boundary;
      shared_boundary_output = shader == warp_shader_e::selected_outline_boundary;
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
      } else if ((shader == warp_shader_e::live_main || selected_outline) && !fixed_eleven_reference) {
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

    // Immutable current guard operands; the shared renderer owns its pixel program.
    bool configure_guard_operands(const models::research_queued_render_guard_v1::expected_t *expected,
                                  const std::vector<std::uint32_t> *transaction, std::string &error) {
      if (!device || !context) {error = "queued guard requires initialized fixture";return false;}
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
      private_guard_cb = cb;
      private_guard_packet = packet;
      private_guard_srv = srv;
      private_guard_configured = true;
      return true;
    }

    static constexpr std::array<models::host_sbs_shader_cache::shader_spec, 5u> shared_ui_pin_specs {{
      {"host_sbs_shared_ui_warp.hlsl", "initialize_main", "cs_5_0"},
      {"host_sbs_shared_ui_warp.hlsl", "SunshineApplyUICS", "cs_5_0"},
      {"host_sbs_shared_ui_warp.hlsl", "shared_pin_main_ps", "ps_5_0"},
      {"host_sbs_shared_ui_warp.hlsl", "shared_pin_mapping_ps", "ps_5_0"},
      {"host_sbs_shared_ui_warp.hlsl", "shared_pin_coverage_ps", "ps_5_0"},
    }};

    bool configure_shared_ui_pin(const std::filesystem::path &root, const std::string &closure, std::string &error) {
      const auto sources = models::host_sbs_shader_cache::snapshot_sources(root, shared_ui_pin_specs);
      if (!device || !sources || closure.size() != 64u || models::host_sbs_shader_cache::source_closure_sha256(sources) != closure) {
        error = "shared UI pin experiment requires initialized fixture and exact pinned source closure";
        return false;
      }
      std::array<ComPtr<ID3D11ComputeShader>, 2u> cs;
      std::array<ComPtr<ID3D11PixelShader>, 3u> ps;
      for (std::size_t i=0u;i<shared_ui_pin_specs.size();++i) {
        const auto code = models::host_sbs_shader_cache::get(sources,shared_ui_pin_specs[i]);
        if (!code || (i<2u ? FAILED(device->CreateComputeShader(code->data(),code->size(),nullptr,&cs[i])) :
                            FAILED(device->CreatePixelShader(code->data(),code->size(),nullptr,&ps[i-2u])))) {
          error = "shared UI pin experimental shader creation failed";
          return false;
        }
      }
      shared_ui_pin_cs = std::move(cs);shared_ui_pin_ps = std::move(ps);
      shared_ui_pin_root = root;shared_ui_pin_closure = closure;
      return true;
    }

    bool shared_ui_pin_source_unchanged() const {
      const auto sources = models::host_sbs_shader_cache::snapshot_sources(shared_ui_pin_root,shared_ui_pin_specs);
      return sources && models::host_sbs_shader_cache::source_closure_sha256(sources) == shared_ui_pin_closure;
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
      const bool readback_output = true,
      const current_outline_inputs_t *outline = nullptr
    ) {
      // An absent selected receipt retains the shared renderer's ordinary Base path.
      current_outline_inputs_t unavailable_outline;
      if (selected_outline && subtitle && !outline) {
        unavailable_outline.matte = subtitle->matte;
        outline = &unavailable_outline;
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

      ComPtr<ID3D11Texture2D> matte_texture, snapshot_texture, selected_matte_texture;
      ComPtr<ID3D11ShaderResourceView> matte_srv, snapshot_srv, condition_srv, identity_srv;
      ComPtr<ID3D11ShaderResourceView> selected_matte_srv, selected_receipt_srv;
      ComPtr<ID3D11Buffer> subtitle_buffer, condition_buffer, identity_buffer, selected_receipt_buffer;
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
        if (selected_outline) {
          const std::array<std::uint32_t, 64u> unavailable_receipt {};
          const auto &selected = outline ? outline->matte : subtitle->matte;
          const auto &receipt = outline ? outline->receipt : unavailable_receipt;
          if (selected.size() != subtitle->matte.size() || !create_immutable_texture_srv(DXGI_FORMAT_R32_UINT, subtitle->width, subtitle->height, sizeof(std::uint32_t), selected.data(), selected_matte_texture, selected_matte_srv) || !words_srv(receipt, selected_receipt_buffer, selected_receipt_srv)) {
            error = "could not create selected current-outline matte/receipt operands";
            return false;
          }
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
      const bool production_shared_pin = selected_outline && subtitle &&
        (!outline || outline->production_shared_ui_pin || !outline->shared_ui_pin);
      const bool shared_pin = outline && (outline->shared_ui_pin || production_shared_pin);
      if (shared_pin && (!selected_outline || !subtitle ||
          (!production_shared_pin && (!shared_ui_pin_ps[0] || !shared_ui_pin_ps[1])))) {
        error = "shared UI pin route needs a configured current tuple";
        return false;
      }
      if (production_shared_pin && !production_shared_ui_pin_programs.ready() &&
          !production_shared_ui_pin_programs.load(device.Get(), SUNSHINE_SHADERS_DIR, error)) return false;
      const bool production_queued_pin = production_shared_pin && outline && outline->production_shared_ui_pin_queued;
      if (production_shared_pin && (mapping_output || shared_boundary_output) &&
          !production_shared_ui_pin_programs.load_diagnostics(device.Get(), SUNSHINE_SHADERS_DIR, error)) return false;
      platf::dxgi::host_sbs_v2_draw_command_t draw_command {
        .render_targets = {output_rtv.Get(), nullptr},
        .render_target_count = 1u,
        .vertex_shader = vertex_shader.Get(),
        .pixel_shader = production_shared_pin ?
                        shared_boundary_output ? production_shared_ui_pin_programs.boundary(production_queued_pin) :
                        mapping_output ? production_shared_ui_pin_programs.mapping(production_queued_pin) :
                        production_shared_ui_pin_programs.program(production_queued_pin).pixel :
                        shared_pin ? shared_ui_pin_ps[0].Get() : pixel_shader.Get(),
        .viewport = viewport,
        .sampler = sampler.Get(),
        .shader_resources = {
          color_srv.Get(),
          parallax_srv.Get(),
          state_srv.Get(),
          selected_outline ? selected_matte_srv.Get() : matte_srv.Get(),
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
      struct pin_plane_t {
        ComPtr<ID3D11Texture2D> texture;
        ComPtr<ID3D11ShaderResourceView> srv;
        ComPtr<ID3D11UnorderedAccessView> uav;
      };
      pin_plane_t pin_field, pin_weights, pin_rows;
      models::host_sbs_shared_ui_warp::views_t production_pin_views;
      ComPtr<ID3D11Buffer> pin_geometry;
      std::array<UINT,4u> pin_shape {};
      ComPtr<ID3D11Query> timing_disjoint,timing_start,timing_end;
      if (outline && outline->render_gpu_ms) {
        *outline->render_gpu_ms = -1.0;
        D3D11_QUERY_DESC query {D3D11_QUERY_TIMESTAMP_DISJOINT,0u};
        if (FAILED(device->CreateQuery(&query,&timing_disjoint))) return false;
        query.Query=D3D11_QUERY_TIMESTAMP;
        if (FAILED(device->CreateQuery(&query,&timing_start)) || FAILED(device->CreateQuery(&query,&timing_end))) return false;
      }
      bool timing_started=false;
      const auto begin_timing=[&]() {
        if (timing_disjoint && !timing_started) {
          context->Begin(timing_disjoint.Get());context->End(timing_start.Get());timing_started=true;
        }
      };
      if (shared_pin) {
        const UINT crop_top=subtitle->constants.source_top+subtitle->identity[10u];
        const UINT crop_end=std::min(source_height,crop_top+subtitle->height);
        if (crop_top>=source_height || crop_end<=crop_top) {
          error="shared UI pin native band is outside source";return false;
        }
        pin_shape={source_width,source_height,crop_top>0u?crop_top-1u:0u,0u};
        pin_shape[3u]=std::min(source_height,crop_end+1u)-pin_shape[2u];
        if (production_shared_pin) {
          namespace shared = models::host_sbs_shared_ui_warp;
          shared::request_t request;
          std::copy(draw_command.shader_resources.begin(), draw_command.shader_resources.end(), request.draw_srvs.begin());
          request.draw_srvs[7u] = selected_receipt_srv.Get();
          request.geometry_constants = geometry_buffer.Get();request.subtitle_constants = subtitle_buffer.Get();
          request.sampler = sampler.Get();
          request.queued = production_queued_pin;
          request.queued_guard_constants = production_queued_pin ? private_guard_cb.Get() : nullptr;
          request.queued_guard_transaction = production_queued_pin ? private_guard_srv.Get() : nullptr;
          if (!production_shared_ui_pin_owner.ensure(device.Get(), pin_shape, error)) return false;
          begin_timing();
          if (!production_shared_ui_pin_owner.queue(context.Get(), production_shared_ui_pin_programs.program(production_queued_pin), request, error) ||
              !production_shared_ui_pin_owner.matches(context.Get(), request)) return false;
          // Explicit test-only mutation after the pin field has been issued. The queued PS
          // must authenticate its actual draw guard rather than trust the preparation header.
          if (outline->shared_ui_pin_draw_guard_override)
            draw_command.queued_guard_transaction = outline->shared_ui_pin_draw_guard_override;
          production_pin_views = production_shared_ui_pin_owner.views();
          const auto retain = [](ID3D11ShaderResourceView *view, pin_plane_t &plane) {
            ComPtr<ID3D11Resource> resource;view->GetResource(&resource);plane.srv = view;
            return SUCCEEDED(resource.As(&plane.texture));
          };
          if (!retain(production_pin_views.field.Get(), pin_field) || !retain(production_pin_views.weights.Get(), pin_weights) ||
              !retain(production_pin_views.rows.Get(), pin_rows)) return false;
          pin_geometry = production_pin_views.geometry;
        } else {
        const auto plane=[&](UINT w,UINT h,DXGI_FORMAT format,pin_plane_t &p) {
          D3D11_TEXTURE2D_DESC d {};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=1u;
          d.Format=format;d.SampleDesc.Count=1u;d.Usage=D3D11_USAGE_DEFAULT;
          d.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
          return SUCCEEDED(device->CreateTexture2D(&d,nullptr,&p.texture)) &&
            SUCCEEDED(device->CreateShaderResourceView(p.texture.Get(),nullptr,&p.srv)) &&
            SUCCEEDED(device->CreateUnorderedAccessView(p.texture.Get(),nullptr,&p.uav));
        };
        if (!plane(source_width,pin_shape[3u],DXGI_FORMAT_R32_FLOAT,pin_field) ||
            !plane(source_width,pin_shape[3u],DXGI_FORMAT_R32_FLOAT,pin_weights) ||
            !plane(1u,pin_shape[3u]+1u,DXGI_FORMAT_R32_UINT,pin_rows)) {
          error="shared UI pin native planes unavailable";return false;
        }
        D3D11_BUFFER_DESC d {};d.ByteWidth=sizeof(pin_shape);d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial {pin_shape.data(),0u,0u};
        if (FAILED(device->CreateBuffer(&d,&initial,&pin_geometry))) return false;
        begin_timing();
        const UINT clear[4] {};context->ClearUnorderedAccessViewUint(pin_rows.uav.Get(),clear);
        ID3D11ShaderResourceView *srvs[15] {color_srv.Get(),parallax_srv.Get(),state_srv.Get(),selected_matte_srv.Get(),
          snapshot_srv.Get(),condition_srv.Get(),identity_srv.Get(),selected_receipt_srv.Get(),matte_srv.Get(),
          nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
        ID3D11Buffer *b2=geometry_buffer.Get(),*b3=subtitle_buffer.Get(),*b5=pin_geometry.Get();
        ID3D11SamplerState *linear=sampler.Get();
        ID3D11UnorderedAccessView *uavs[3] {pin_field.uav.Get(),pin_weights.uav.Get(),pin_rows.uav.Get()};
        context->CSSetConstantBuffers(2u,1u,&b2);context->CSSetConstantBuffers(3u,1u,&b3);context->CSSetConstantBuffers(6u,1u,&b5);
        context->CSSetSamplers(0u,1u,&linear);context->CSSetShaderResources(0u,15u,srvs);
        context->CSSetUnorderedAccessViews(0u,3u,uavs,nullptr);context->CSSetShader(shared_ui_pin_cs[0].Get(),nullptr,0u);
        context->Dispatch((source_width+15u)/16u,(pin_shape[3u]+15u)/16u,1u);
        ID3D11UnorderedAccessView *null_uavs[3] {};context->CSSetUnorderedAccessViews(0u,3u,null_uavs,nullptr);
        if (!outline->shared_ui_pin_resample_only) {
          srvs[12u]=pin_weights.srv.Get();context->CSSetShaderResources(0u,15u,srvs);
          uavs[1u]=uavs[2u]=nullptr;context->CSSetUnorderedAccessViews(0u,3u,uavs,nullptr);
          context->CSSetShader(shared_ui_pin_cs[1].Get(),nullptr,0u);context->Dispatch((pin_shape[3u]+7u)/8u,1u,1u);
          context->CSSetUnorderedAccessViews(0u,3u,null_uavs,nullptr);
        }
        ID3D11ShaderResourceView *null_srvs[15] {};context->CSSetShaderResources(0u,15u,null_srvs);
        ID3D11Buffer *null_cb=nullptr;ID3D11SamplerState *null_sampler=nullptr;
        context->CSSetConstantBuffers(2u,1u,&null_cb);context->CSSetConstantBuffers(3u,1u,&null_cb);context->CSSetConstantBuffers(6u,1u,&null_cb);
        context->CSSetSamplers(0u,1u,&null_sampler);context->CSSetShader(nullptr,nullptr,0u);
        ID3D11ShaderResourceView *pin_srvs[3] {pin_weights.srv.Get(),pin_field.srv.Get(),pin_rows.srv.Get()};
        context->PSSetShaderResources(12u,3u,pin_srvs);context->PSSetConstantBuffers(6u,1u,&b5);
        }
      }
      begin_timing();
      bool submitted = false;
      ComPtr<ID3D11Buffer> previous_auxiliary_constants;
      std::array<ComPtr<ID3D11ShaderResourceView>, 2u> previous_auxiliary_resources;
      if (selected_outline) {
        context->PSGetConstantBuffers(4u, 1u, previous_auxiliary_constants.GetAddressOf());
        context->PSGetShaderResources(9u, 1u, &previous_auxiliary_resources[0u]);
        context->PSGetShaderResources(11u, 1u, &previous_auxiliary_resources[1u]);
      }
      {
        models::research_queued_render_guard_v1::scoped_binding_t guard;
        if (private_guard_configured && !use_recorder_guard_scope && !guard.bind(context.Get(), private_guard_cb.Get(), private_guard_srv.Get())) {
          error = "private queued guard binding rejected";
          return false;
        }
        models::research_current_outline_selected_matte_v2::scoped_receipt_t selected_scope;
        if (selected_outline && subtitle && !selected_scope.bind(context.Get(), selected_receipt_srv.Get())) {
          error = "production selected-matte receipt scope rejected draw operands";
          return false;
        }
        models::host_sbs_shared_ui_warp::scoped_binding_t pin_scope;
        if (production_shared_pin && !pin_scope.bind(context.Get(), &production_pin_views)) return false;
        submitted = platf::dxgi::record_host_sbs_v2_draw(context.Get(), draw_command);
      }
      if (selected_outline) {
        ComPtr<ID3D11Buffer> restored_auxiliary_constants;
        context->PSGetConstantBuffers(4u, 1u, restored_auxiliary_constants.GetAddressOf());
        if (restored_auxiliary_constants.Get() != previous_auxiliary_constants.Get()) {
          error = "shared warp changed unrelated b4 state";
          return false;
        }
      }
      if (timing_disjoint) {
        context->End(timing_end.Get());context->End(timing_disjoint.Get());
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint {};UINT64 start=0u,end=0u;
        while (context->GetData(timing_disjoint.Get(),&disjoint,sizeof(disjoint),0u)==S_FALSE) Sleep(1u);
        while (context->GetData(timing_start.Get(),&start,sizeof(start),0u)==S_FALSE) Sleep(1u);
        while (context->GetData(timing_end.Get(),&end,sizeof(end),0u)==S_FALSE) Sleep(1u);
        if (disjoint.Disjoint || disjoint.Frequency==0u || end<start) {error="shared comparison GPU timestamp invalid";return false;}
        *outline->render_gpu_ms=double(end-start)*1000.0/double(disjoint.Frequency);
      }
      if (shared_pin && outline->shared_ui_pin_diagnostic_output) {
        // Production renders do not configure the separate private pin shaders.
        // Read their diagnostics through the canonical same-device program tuple.
        if (production_shared_pin &&
            !production_shared_ui_pin_programs.load_diagnostics(device.Get(), SUNSHINE_SHADERS_DIR, error)) return false;
        auto &out=*outline->shared_ui_pin_diagnostic_output;out={};out.shape=pin_shape;
        std::vector<std::byte> bytes;
        if (!read_diagnostic_texture(pin_field.texture.Get(),4u,bytes)) {error="shared UI pin field diagnostic readback failed";return false;}
        out.field.resize(bytes.size()/4u);std::memcpy(out.field.data(),bytes.data(),bytes.size());
        if (!read_diagnostic_texture(pin_weights.texture.Get(),4u,bytes)) {error="shared UI pin weights diagnostic readback failed";return false;}
        out.authorized_weights.resize(bytes.size()/4u);std::memcpy(out.authorized_weights.data(),bytes.data(),bytes.size());
        if (!read_diagnostic_texture(pin_rows.texture.Get(),4u,bytes)) {error="shared UI pin rows diagnostic readback failed";return false;}
        out.active_rows.resize(bytes.size()/4u);std::memcpy(out.active_rows.data(),bytes.data(),bytes.size());
        out.selected_receipt_header[0u]=out.active_rows.back();out.selected_receipt_authorized=out.active_rows.back()!=0u;out.active_rows.pop_back();
        D3D11_TEXTURE2D_DESC d=output_desc;d.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;
        ComPtr<ID3D11Texture2D> map;ComPtr<ID3D11RenderTargetView> map_rtv;
        if (FAILED(device->CreateTexture2D(&d,nullptr,&map)) || FAILED(device->CreateRenderTargetView(map.Get(),nullptr,&map_rtv))) {error="shared UI pin diagnostic render target unavailable";return false;}
        auto command=draw_command;
        command.pixel_shader=production_shared_pin?production_shared_ui_pin_programs.mapping(production_queued_pin):
          shared_ui_pin_ps[1u].Get();
        command.render_targets[0u]=map_rtv.Get();
        models::research_current_outline_selected_matte_v2::scoped_receipt_t coverage_scope;
        if (!coverage_scope.bind(context.Get(),selected_receipt_srv.Get())) {error="shared UI pin diagnostic receipt binding failed";return false;}
        models::host_sbs_shared_ui_warp::scoped_binding_t pin_scope;
        if (production_shared_pin && !pin_scope.bind(context.Get(), &production_pin_views)) {error="production shared UI pin diagnostic binding failed";return false;}
        if (!platf::dxgi::record_host_sbs_v2_draw(context.Get(),command) || !read_diagnostic_texture(map.Get(),16u,out.mapping_rgba32f)) {error="shared UI pin mapping diagnostic draw/readback failed";return false;}
        command.pixel_shader=production_shared_pin?production_shared_ui_pin_programs.coverage(production_queued_pin):
          shared_ui_pin_ps[2u].Get();
        if (!platf::dxgi::record_host_sbs_v2_draw(context.Get(),command) || !read_diagnostic_texture(map.Get(),16u,out.ui_coverage_rgba32f)) {error="shared UI pin coverage diagnostic draw/readback failed";return false;}
      }
      if (shared_pin && !production_shared_pin) {
        ID3D11ShaderResourceView *null_srvs[3] {};context->PSSetShaderResources(12u,3u,null_srvs);
        ID3D11Buffer *null_cb=nullptr;context->PSSetConstantBuffers(6u,1u,&null_cb);
      }
      ID3D11Buffer *null_constants = nullptr;
      if (direct_constants_buffer) context->PSSetConstantBuffers(4u, 1u, &null_constants);
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
      if (selected_outline) {
        std::array<ID3D11ShaderResourceView *, 1u> private_resources {};
        context->PSGetShaderResources(7u, static_cast<UINT>(private_resources.size()), private_resources.data());
        for (auto *resource : private_resources) {
          unbound = unbound && resource == nullptr;
          if (resource) {
            resource->Release();
          }
        }
        if (!unbound) {
          error = "shared warp leaked its selected receipt binding";
          return false;
        }
        constexpr std::array<UINT, 2u> auxiliary_slots {9u, 11u};
        for (std::size_t index = 0u; index < auxiliary_slots.size(); ++index) {
          ComPtr<ID3D11ShaderResourceView> resource;
          context->PSGetShaderResources(auxiliary_slots[index], 1u, &resource);
          if (resource.Get() != previous_auxiliary_resources[index].Get()) {
            error = "shared warp changed unrelated auxiliary resources";
            return false;
          }
        }
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

    bool read_diagnostic_texture(ID3D11Texture2D *texture, UINT bytes_per_pixel, std::vector<std::byte> &bytes) {
      D3D11_TEXTURE2D_DESC d {};texture->GetDesc(&d);
      d.Usage=D3D11_USAGE_STAGING;d.BindFlags=d.MiscFlags=0u;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device->CreateTexture2D(&d,nullptr,&staging))) return false;
      context->CopyResource(staging.Get(),texture);D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(),0u,D3D11_MAP_READ,0u,&mapped))) return false;
      const auto row=std::size_t(d.Width)*bytes_per_pixel;bytes.resize(row*d.Height);
      for (UINT y=0u;y<d.Height;++y) std::memcpy(bytes.data()+std::size_t(y)*row,
        static_cast<const std::byte *>(mapped.pData)+std::size_t(y)*mapped.RowPitch,row);
      context->Unmap(staging.Get(),0u);return true;
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
      if (direct_constants_buffer) context->PSSetConstantBuffers(4u, 1u, &null_constants);
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
    bool selected_outline = false;
    bool shared_boundary_output = false;
    std::array<ComPtr<ID3D11ComputeShader>,2u> shared_ui_pin_cs;
    std::array<ComPtr<ID3D11PixelShader>,3u> shared_ui_pin_ps;
    std::filesystem::path shared_ui_pin_root;
    std::string shared_ui_pin_closure;
    models::host_sbs_shared_ui_warp::programs_t production_shared_ui_pin_programs;
    models::host_sbs_shared_ui_warp::resources_t production_shared_ui_pin_owner;
    bool private_guard_configured = false;
    bool use_recorder_guard_scope = false;
    ComPtr<ID3D11Buffer> private_guard_cb, private_guard_packet;
    ComPtr<ID3D11ShaderResourceView> private_guard_srv;
  };

  template <typename Pixel>
  bool copy_texture_region_exact(
    const DXGI_FORMAT format,
    const UINT source_width,
    const UINT source_height,
    const std::vector<Pixel> &source_pixels,
    const D3D11_BOX &source_box,
    std::vector<Pixel> &cropped_pixels,
    std::string &error
  ) {
    const UINT crop_width = source_box.right - source_box.left;
    const UINT crop_height = source_box.bottom - source_box.top;
    if (source_width == 0u || source_height == 0u || crop_width == 0u ||
        crop_height == 0u || source_box.right > source_width ||
        source_box.bottom > source_height || source_box.front != 0u ||
        source_box.back != 1u ||
        source_pixels.size() !=
          static_cast<std::size_t>(source_width) * source_height) {
      error = "invalid D3D11 crop-copy test input";
      return false;
    }

    constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL actual {};
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    if (FAILED(D3D11CreateDevice(
          nullptr,
          D3D_DRIVER_TYPE_WARP,
          nullptr,
          0,
          requested,
          static_cast<UINT>(std::size(requested)),
          D3D11_SDK_VERSION,
          &device,
          &actual,
          &context
        )) ||
        actual < D3D_FEATURE_LEVEL_11_0) {
      error = "D3D11 WARP feature level 11 crop-copy initialization failed";
      return false;
    }

    D3D11_TEXTURE2D_DESC source_desc {};
    source_desc.Width = source_width;
    source_desc.Height = source_height;
    source_desc.MipLevels = 1u;
    source_desc.ArraySize = 1u;
    source_desc.Format = format;
    source_desc.SampleDesc.Count = 1u;
    source_desc.Usage = D3D11_USAGE_DEFAULT;
    D3D11_SUBRESOURCE_DATA source_data {};
    source_data.pSysMem = source_pixels.data();
    source_data.SysMemPitch = source_width * sizeof(Pixel);
    ComPtr<ID3D11Texture2D> source_texture;
    if (FAILED(device->CreateTexture2D(
          &source_desc,
          &source_data,
          &source_texture
        ))) {
      error = "could not create the source texture for exact crop copy";
      return false;
    }

    auto crop_desc = source_desc;
    crop_desc.Width = crop_width;
    crop_desc.Height = crop_height;
    ComPtr<ID3D11Texture2D> crop_texture;
    if (FAILED(device->CreateTexture2D(&crop_desc, nullptr, &crop_texture))) {
      error = "could not create the same-format crop texture";
      return false;
    }
    context->CopySubresourceRegion(
      crop_texture.Get(),
      0,
      0,
      0,
      0,
      source_texture.Get(),
      0,
      &source_box
    );

    auto staging_desc = crop_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging))) {
      error = "could not create the exact crop staging texture";
      return false;
    }
    context->CopyResource(staging.Get(), crop_texture.Get());

    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
      error = "could not read the exact crop-copy result";
      return false;
    }
    cropped_pixels.resize(static_cast<std::size_t>(crop_width) * crop_height);
    const std::size_t row_bytes = static_cast<std::size_t>(crop_width) * sizeof(Pixel);
    for (UINT y = 0; y < crop_height; ++y) {
      std::memcpy(
        cropped_pixels.data() + static_cast<std::size_t>(y) * crop_width,
        static_cast<const std::byte *>(mapped.pData) +
          static_cast<std::size_t>(y) * mapped.RowPitch,
        row_bytes
      );
    }
    context->Unmap(staging.Get(), 0);
    return true;
  }

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

  current_outline_inputs_t make_current_outline_inputs(const subtitle_inputs_t &subtitle) {
    // This is the published selector ABI consumed by the production shared warp, not a
    // substitute renderer. The compute-selector producer has separate executable coverage.
    current_outline_inputs_t input;
    input.matte = subtitle.matte;
    auto &receipt = input.receipt;
    receipt[0u] = models::research_current_outline_selected_matte_v2::tag;
    receipt[1u] = models::research_current_outline_selected_matte_v2::schema;
    receipt[2u] = 3u;
    std::copy_n(subtitle.identity.begin() + 8u, 12u, receipt.begin() + 4u);
    std::copy_n(subtitle.identity.begin() + 4u, 4u, receipt.begin() + 16u);
    std::copy_n(subtitle.identity.begin() + 20u, 2u, receipt.begin() + 20u);
    receipt[24u] = 1u;
    receipt[25u] = models::depth_coordinate_v2::subtitle_contour_recipe;
    std::copy(subtitle.identity.begin(), subtitle.identity.end(), receipt.begin() + 28u);
    receipt[60u] = 71u;
    receipt[61u] = 9u;
    receipt[62u] = models::research_current_outline_selected_matte_v2::algorithm;
    return input;
  }

  std::vector<rgba32f_t> unpack_rgba32f(const std::vector<std::byte> &bytes) {
    std::vector<rgba32f_t> result(bytes.size() / sizeof(rgba32f_t));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
  }

  std::vector<rgba16_bits_t> unpack_rgba16(const std::vector<std::byte> &bytes) {
    std::vector<rgba16_bits_t> result(bytes.size() / sizeof(rgba16_bits_t));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
  }

  void expect_flat_identity(
    const std::vector<rgba32f_t> &source,
    const std::vector<rgba32f_t> &packed,
    const UINT width
  ) {
    ASSERT_EQ(source.size(), width);
    ASSERT_EQ(packed.size(), width * 2u);
    for (UINT eye = 0; eye < 2u; ++eye) {
      for (UINT x = 0; x < width; ++x) {
        const auto &expected = source[x];
        const auto &actual = packed[eye * width + x];
        EXPECT_NEAR(actual.r, expected.r, 2.0e-6f)
          << "eye=" << eye << ", x=" << x;
        EXPECT_NEAR(actual.g, expected.g, 2.0e-6f)
          << "eye=" << eye << ", x=" << x;
        EXPECT_NEAR(actual.b, expected.b, 2.0e-6f)
          << "eye=" << eye << ", x=" << x;
        EXPECT_NEAR(actual.a, expected.a, 2.0e-6f)
          << "eye=" << eye << ", x=" << x;
      }
    }
  }

  rgba32f_t lerp_rgba(const rgba32f_t &a, const rgba32f_t &b, const float t) {
    return {
      a.r + t * (b.r - a.r),
      a.g + t * (b.g - a.g),
      a.b + t * (b.b - a.b),
      a.a + t * (b.a - a.a),
    };
  }

  rgba32f_t sample_linear_clamp(
    const std::vector<rgba32f_t> &source,
    const UINT width,
    const UINT height,
    const float u,
    const float v
  ) {
    const float x = std::clamp(u, 0.0f, 1.0f) * static_cast<float>(width) - 0.5f;
    const float y = std::clamp(v, 0.0f, 1.0f) * static_cast<float>(height) - 0.5f;
    const int x0_unclamped = static_cast<int>(std::floor(x));
    const int y0_unclamped = static_cast<int>(std::floor(y));
    const UINT x0 = static_cast<UINT>(std::clamp(x0_unclamped, 0, static_cast<int>(width) - 1));
    const UINT x1 = static_cast<UINT>(std::clamp(x0_unclamped + 1, 0, static_cast<int>(width) - 1));
    const UINT y0 = static_cast<UINT>(std::clamp(y0_unclamped, 0, static_cast<int>(height) - 1));
    const UINT y1 = static_cast<UINT>(std::clamp(y0_unclamped + 1, 0, static_cast<int>(height) - 1));
    const float tx = x - std::floor(x);
    const float ty = y - std::floor(y);
    const auto at = [&](const UINT sx, const UINT sy) -> const rgba32f_t & {
      return source[static_cast<std::size_t>(sy) * width + sx];
    };
    return lerp_rgba(
      lerp_rgba(at(x0, y0), at(x1, y0), tx),
      lerp_rgba(at(x0, y1), at(x1, y1), tx),
      ty
    );
  }

  float sample_linear_clamp(
    const std::vector<float> &source,
    const UINT width,
    const UINT height,
    const float u,
    const float v
  ) {
    const float x = std::clamp(u, 0.0f, 1.0f) * static_cast<float>(width) - 0.5f;
    const float y = std::clamp(v, 0.0f, 1.0f) * static_cast<float>(height) - 0.5f;
    const int x0_unclamped = static_cast<int>(std::floor(x));
    const int y0_unclamped = static_cast<int>(std::floor(y));
    const UINT x0 = static_cast<UINT>(std::clamp(x0_unclamped, 0, static_cast<int>(width) - 1));
    const UINT x1 = static_cast<UINT>(std::clamp(x0_unclamped + 1, 0, static_cast<int>(width) - 1));
    const UINT y0 = static_cast<UINT>(std::clamp(y0_unclamped, 0, static_cast<int>(height) - 1));
    const UINT y1 = static_cast<UINT>(std::clamp(y0_unclamped + 1, 0, static_cast<int>(height) - 1));
    const float tx = x - std::floor(x);
    const float ty = y - std::floor(y);
    const auto at = [&](const UINT sx, const UINT sy) {
      return source[static_cast<std::size_t>(sy) * width + sx];
    };
    const float top = at(x0, y0) + tx * (at(x1, y0) - at(x0, y0));
    const float bottom = at(x0, y1) + tx * (at(x1, y1) - at(x0, y1));
    return top + ty * (bottom - top);
  }

  float sample_effective_parallax(
    const std::vector<float> &roi_local_parallax,
    const UINT depth_width,
    const UINT depth_height,
    const float source_u,
    const float source_v,
    const host_sbs_v2_geometry_t &geometry
  ) {
    if (geometry.video_roi_active == 0.0f) {
      return sample_linear_clamp(
        roi_local_parallax,
        depth_width,
        depth_height,
        source_u,
        source_v
      );
    }

    const float projected_u = std::clamp(
      source_u,
      geometry.video_roi_left,
      geometry.video_roi_right
    );
    const float projected_v = std::clamp(
      source_v,
      geometry.video_roi_top,
      geometry.video_roi_bottom
    );
    const float roi_width = geometry.video_roi_right - geometry.video_roi_left;
    const float roi_height = geometry.video_roi_bottom - geometry.video_roi_top;
    const float local_u =
      (projected_u - geometry.video_roi_left) / roi_width;
    const float local_v =
      (projected_v - geometry.video_roi_top) / roi_height;
    const float tensor_u =
      (static_cast<float>(geometry.tensor_content_left) +
       local_u * static_cast<float>(
         geometry.tensor_content_right - geometry.tensor_content_left)) /
      static_cast<float>(depth_width);
    const float tensor_v =
      (static_cast<float>(geometry.tensor_content_top) +
       local_v * static_cast<float>(
         geometry.tensor_content_bottom - geometry.tensor_content_top)) /
      static_cast<float>(depth_height);
    const float local_parallax = sample_linear_clamp(
      roi_local_parallax,
      depth_width,
      depth_height,
      tensor_u,
      tensor_v
    );
    const float full_source_parallax = roi_width * local_parallax;
    const float vertical_budget_per_source_v =
      models::depth_coordinate_v2::max_vertical_shear * (roi_width / roi_height) *
      (static_cast<float>(
         geometry.tensor_content_bottom - geometry.tensor_content_top) /
       static_cast<float>(
         geometry.tensor_content_right - geometry.tensor_content_left));
    const float collar_budget =
      models::depth_coordinate_v2::max_horizontal_slope *
        std::abs(source_u - projected_u) +
      vertical_budget_per_source_v * std::abs(source_v - projected_v);
    const float magnitude = std::abs(full_source_parallax);
    if (collar_budget >= magnitude) {
      return 0.0f;
    }
    return std::copysign(magnitude - collar_budget, full_source_parallax);
  }

  rgba32f_t one_tap_reference(
    const std::vector<rgba32f_t> &source,
    const UINT width,
    const UINT height,
    const float u,
    const float v
  ) {
    return sample_linear_clamp(source, width, height, u, v);
  }

  void expect_rgba_near(
    const rgba32f_t &actual,
    const rgba32f_t &expected,
    const float tolerance,
    const std::string &where
  ) {
    EXPECT_NEAR(actual.r, expected.r, tolerance) << where << " r";
    EXPECT_NEAR(actual.g, expected.g, tolerance) << where << " g";
    EXPECT_NEAR(actual.b, expected.b, tolerance) << where << " b";
    EXPECT_NEAR(actual.a, expected.a, tolerance) << where << " a";
  }

}  // namespace

TEST(HostSbsVideoRegionCropGpuTest, CopySubresourceRegionPreservesExactBgra8Pixels) {
  constexpr UINT source_width = 11u;
  constexpr UINT source_height = 8u;
  constexpr D3D11_BOX crop_box {
    2u, 1u, 0u,
    9u, 7u, 1u,
  };
  constexpr UINT crop_width = crop_box.right - crop_box.left;
  constexpr UINT crop_height = crop_box.bottom - crop_box.top;
  constexpr bgra8_t outside_sentinel {0xdeu, 0xadu, 0xbeu, 0x4du};
  std::vector<bgra8_t> source(
    static_cast<std::size_t>(source_width) * source_height,
    outside_sentinel
  );
  std::vector<bgra8_t> expected;
  expected.reserve(static_cast<std::size_t>(crop_width) * crop_height);
  for (UINT y = crop_box.top; y < crop_box.bottom; ++y) {
    for (UINT x = crop_box.left; x < crop_box.right; ++x) {
      const bgra8_t pixel {
        static_cast<std::uint8_t>(3u + 17u * x + 11u * y),
        static_cast<std::uint8_t>(5u + 7u * x + 29u * y),
        static_cast<std::uint8_t>(9u + 31u * x + 13u * y),
        0xffu,
      };
      source[static_cast<std::size_t>(y) * source_width + x] = pixel;
      expected.push_back(pixel);
    }
  }

  std::string error;
  std::vector<bgra8_t> cropped;
  ASSERT_TRUE(copy_texture_region_exact(
    DXGI_FORMAT_B8G8R8A8_UNORM,
    source_width,
    source_height,
    source,
    crop_box,
    cropped,
    error
  )) << error;
  ASSERT_EQ(cropped.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(cropped[index], expected[index]) << "crop pixel " << index;
    EXPECT_NE(cropped[index], outside_sentinel) << "crop pixel " << index;
  }
}

TEST(HostSbsVideoRegionCropGpuTest, CopySubresourceRegionPreservesExactRgba16fBits) {
  constexpr UINT source_width = 10u;
  constexpr UINT source_height = 9u;
  constexpr D3D11_BOX crop_box {
    1u, 2u, 0u,
    8u, 7u, 1u,
  };
  constexpr UINT crop_width = crop_box.right - crop_box.left;
  constexpr UINT crop_height = crop_box.bottom - crop_box.top;
  constexpr rgba16_bits_t outside_sentinel {
    0x7bffu,
    0xfbffu,
    0x0001u,
    0x0000u,
  };
  std::vector<rgba16_bits_t> source(
    static_cast<std::size_t>(source_width) * source_height,
    outside_sentinel
  );
  std::vector<rgba16_bits_t> expected;
  expected.reserve(static_cast<std::size_t>(crop_width) * crop_height);
  for (UINT y = crop_box.top; y < crop_box.bottom; ++y) {
    for (UINT x = crop_box.left; x < crop_box.right; ++x) {
      const rgba16_bits_t pixel {
        static_cast<std::uint16_t>(0x3000u + 13u * x + 7u * y),
        static_cast<std::uint16_t>(0x3400u + 5u * x + 17u * y),
        static_cast<std::uint16_t>(0x3800u + 19u * x + 3u * y),
        0x3c00u,
      };
      source[static_cast<std::size_t>(y) * source_width + x] = pixel;
      expected.push_back(pixel);
    }
  }

  std::string error;
  std::vector<rgba16_bits_t> cropped;
  ASSERT_TRUE(copy_texture_region_exact(
    DXGI_FORMAT_R16G16B16A16_FLOAT,
    source_width,
    source_height,
    source,
    crop_box,
    cropped,
    error
  )) << error;
  ASSERT_EQ(cropped.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(cropped[index], expected[index]) << "crop pixel " << index;
    EXPECT_NE(cropped[index], outside_sentinel) << "crop pixel " << index;
  }
}

TEST(HostSbsV2LiveWarpGpuTest, ExecutesAuthenticatedPixelContract) {
  constexpr UINT width = 8u;
  constexpr UINT height = 1u;
  const std::array<float, 4> sentinel_clear {0.91f, 0.13f, 0.77f, 0.42f};
  std::string error;
  live_v2_warp_fixture_t warp;
  ASSERT_TRUE(warp.initialize(error)) << error;

  std::vector<rgba32f_t> source(width);
  for (UINT x = 0; x < width; ++x) {
    source[x] = {
      (static_cast<float>(x) + 0.5f) / static_cast<float>(width),
      0.15f + 0.02f * static_cast<float>(x),
      0.8f - 0.03f * static_cast<float>(x),
      1.0f,
    };
  }

  // A valid zero field must preserve the current mono frame independently in both eyes.
  std::vector<std::byte> packed_bytes;
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    width,
    height,
    source.data(),
    std::vector<float>(width, 0.0f),
    std::vector<float>(width, 0.0f),
    make_live_state(true, true),
    sentinel_clear,
    packed_bytes,
    error
  )) << error;
  const auto zero_output = unpack_rgba32f(packed_bytes);
  expect_flat_identity(source, zero_output, width);

  // Positive signed parallax looks up a lower source U in the left eye and a higher source U in
  // the right eye. With an increasing red ramp this directly proves the production eye signs.
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    width,
    height,
    source.data(),
    std::vector<float>(width, 0.10f),
    std::vector<float>(width, 0.10f),
    make_live_state(true, true),
    sentinel_clear,
    packed_bytes,
    error
  )) << error;
  const auto signed_output = unpack_rgba32f(packed_bytes);
  constexpr UINT probe_x = 3u;
  ASSERT_EQ(signed_output.size(), width * 2u);
  EXPECT_LT(signed_output[probe_x].r, zero_output[probe_x].r - 0.08f);
  EXPECT_GT(
    signed_output[width + probe_x].r,
    zero_output[width + probe_x].r + 0.08f
  );

  // Invalid depth retains only the camera, not stale color or geometry. The live renderer must
  // overwrite the sentinel target with the current matched color through flat identity.
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    width,
    height,
    source.data(),
    std::vector<float>(width, 0.10f),
    std::vector<float>(width, -3.0f),
    make_live_state(false, true),
    sentinel_clear,
    packed_bytes,
    error
  )) << error;
  expect_flat_identity(source, unpack_rgba32f(packed_bytes), width);

  // An unknown same-sized state contract is equally untrusted and must fail to deterministic
  // current-color identity rather than retaining whatever happened to be in the target.
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    width,
    height,
    source.data(),
    std::vector<float>(width, 0.10f),
    std::vector<float>(width, -3.0f),
    make_live_state(true, false),
    sentinel_clear,
    packed_bytes,
    error
  )) << error;
  expect_flat_identity(source, unpack_rgba32f(packed_bytes), width);

  // The renderer consumes the compact authorization sealed by state resolve. A missing or
  // corrupt seal must fail to current-color identity without repeating the full camera checksum
  // for every output pixel.
  auto corrupt_authorization_state = make_live_state(true, true);
  corrupt_authorization_state[models::depth_coordinate_v2::renderer_authorization_bits] ^= 1u;
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    width,
    height,
    source.data(),
    std::vector<float>(width, 0.10f),
    std::vector<float>(width, -3.0f),
    corrupt_authorization_state,
    sentinel_clear,
    packed_bytes,
    error
  )) << error;
  expect_flat_identity(source, unpack_rgba32f(packed_bytes), width);

  // The V2 warp is geometry-only. FP16 linear scRGB values, including negative and >1 channels,
  // must pass through byte-for-byte at zero parallax; tone mapping belongs only to model input.
  constexpr rgba16_bits_t scrgb_pixel {
    0xb800u,  // -0.5
    0x3800u,  //  0.5
    0x4400u,  //  4.0
    0x3c00u,  //  1.0
  };
  const std::vector<rgba16_bits_t> scrgb_source(width, scrgb_pixel);
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R16G16B16A16_FLOAT,
    sizeof(rgba16_bits_t),
    width,
    height,
    scrgb_source.data(),
    std::vector<float>(width, 0.0f),
    std::vector<float>(width, -3.0f),
    make_live_state(true, true),
    sentinel_clear,
    packed_bytes,
    error
  )) << error;
  const auto scrgb_output = unpack_rgba16(packed_bytes);
  ASSERT_EQ(scrgb_output.size(), width * 2u);
  for (std::size_t index = 0; index < scrgb_output.size(); ++index) {
    EXPECT_EQ(scrgb_output[index], scrgb_pixel) << "packed pixel " << index;
  }
}

TEST(HostSbsV2LiveWarpGpuTest, RoiLocalFieldUsesMinimumSlopeConstrainedOutsideCollar) {
  namespace v2 = models::depth_coordinate_v2;
  constexpr UINT source_width = 512u;
  constexpr UINT depth_width = 64u;
  constexpr UINT depth_height = 32u;
  constexpr float local_parallax = 0.04f;
  constexpr host_sbs_v2_geometry_t geometry {
    .video_roi_active = 1.0f,
    .video_roi_left = 0.25f,
    .video_roi_top = 0.25f,
    .video_roi_right = 0.75f,
    .video_roi_bottom = 0.75f,
    .tensor_content_right = depth_width,
    .tensor_content_bottom = depth_height,
  };
  const std::vector<float> positive_field(
    static_cast<std::size_t>(depth_width) * depth_height,
    local_parallax
  );
  const std::vector<float> negative_field(
    positive_field.size(),
    -local_parallax
  );

  const auto effective = [&](const std::vector<float> &field, float u, float v) {
    return sample_effective_parallax(
      field,
      depth_width,
      depth_height,
      u,
      v,
      geometry
    );
  };

  constexpr float roi_width = 0.5f;
  constexpr float expected_full_source_parallax =
    roi_width * local_parallax;
  // ROI-local q is not tapered or otherwise changed. Only the coordinate-unit conversion from
  // local video U to full-source U scales its renderer displacement.
  EXPECT_FLOAT_EQ(effective(positive_field, 0.50f, 0.50f), expected_full_source_parallax);
  EXPECT_FLOAT_EQ(
    effective(positive_field, geometry.video_roi_left, 0.50f),
    expected_full_source_parallax
  );
  EXPECT_FLOAT_EQ(
    effective(positive_field, 0.50f, 0.50f) / roi_width,
    local_parallax
  );
  EXPECT_FLOAT_EQ(
    effective(negative_field, 0.50f, 0.50f),
    -expected_full_source_parallax
  );

  auto narrow_geometry = geometry;
  narrow_geometry.video_roi_left = 0.375f;
  narrow_geometry.video_roi_right = 0.625f;
  EXPECT_FLOAT_EQ(
    sample_effective_parallax(
      positive_field,
      depth_width,
      depth_height,
      0.50f,
      0.50f,
      narrow_geometry
    ),
    0.25f * local_parallax
  );

  // The left/right collar is the fastest continuous decay admitted by the authenticated 0.5
  // row slope. At |p|=0.02 its exact support is 0.04 full-source U beyond either ROI edge.
  constexpr float left_zero =
    geometry.video_roi_left -
    expected_full_source_parallax / v2::max_horizontal_slope;
  constexpr float right_zero =
    geometry.video_roi_right +
    expected_full_source_parallax / v2::max_horizontal_slope;
  EXPECT_NEAR(effective(positive_field, left_zero, 0.50f), 0.0f, 1.0e-7f);
  EXPECT_NEAR(effective(negative_field, left_zero, 0.50f), 0.0f, 1.0e-7f);
  EXPECT_NEAR(effective(positive_field, right_zero, 0.50f), 0.0f, 1.0e-7f);
  EXPECT_FLOAT_EQ(effective(positive_field, left_zero - 0.001f, 0.50f), 0.0f);
  EXPECT_FLOAT_EQ(effective(negative_field, right_zero + 0.001f, 0.50f), 0.0f);
  EXPECT_NEAR(
    effective(positive_field, geometry.video_roi_left - 0.01f, 0.50f),
    expected_full_source_parallax - 0.01f * v2::max_horizontal_slope,
    1.0e-7f
  );
  EXPECT_NEAR(
    effective(negative_field, geometry.video_roi_left - 0.01f, 0.50f),
    -expected_full_source_parallax + 0.01f * v2::max_horizontal_slope,
    1.0e-7f
  );

  // Dense finite differences cover both ROI boundaries and both zero joins. They prove that the
  // local-to-full conversion plus signed collar never exceeds the same global row slope that
  // makes the production fixed-point inverse contractive.
  constexpr float sample_step = 1.0f / 4096.0f;
  for (const auto *field : {&positive_field, &negative_field}) {
    float previous = effective(*field, 0.0f, 0.50f);
    for (float u = sample_step; u <= 1.0f; u += sample_step) {
      const float current = effective(*field, u, 0.50f);
      EXPECT_LE(
        std::abs(current - previous),
        v2::max_horizontal_slope * sample_step + 2.0e-7f
      ) << "u=" << u;
      previous = current;
    }
  }

  // Top/bottom use source-width units just like the producer's vertical shear. With this 2:1
  // tensor-content aspect, 0.02 parallax reaches zero 0.02 source-V outside the edge. A corner
  // spends the horizontal and vertical budgets together, producing the minimum anisotropic diamond.
  constexpr float top_zero = geometry.video_roi_top - 0.02f;
  EXPECT_NEAR(effective(positive_field, 0.50f, top_zero), 0.0f, 1.0e-7f);
  EXPECT_FLOAT_EQ(effective(positive_field, 0.50f, top_zero - 0.001f), 0.0f);
  EXPECT_FLOAT_EQ(effective(positive_field, 0.229f, 0.239f), 0.0f);

  // Run the real GPU shader against a source-U color ramp. The complete packed output must match
  // an independent CPU inverse using the same ROI-local texture and effective collar field.
  constexpr UINT render_height = 1u;
  std::vector<rgba32f_t> source(source_width);
  for (UINT x = 0; x < source_width; ++x) {
    const float u = (static_cast<float>(x) + 0.5f) /
                    static_cast<float>(source_width);
    source[x] = {u, 0.2f + 0.3f * u, 0.8f - 0.2f * u, 1.0f};
  }
  std::string error;
  live_v2_warp_fixture_t warp;
  ASSERT_TRUE(warp.initialize(error)) << error;
  std::vector<std::byte> packed_bytes;
  const std::array<float, 4> sentinel_clear {0.91f, 0.13f, 0.77f, 0.42f};
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    source_width,
    render_height,
    source.data(),
    positive_field,
    std::vector<float>(positive_field.size(), 0.0f),
    make_live_state(true, true),
    sentinel_clear,
    packed_bytes,
    error,
    depth_width,
    depth_height,
    geometry
  )) << error;
  const auto packed = unpack_rgba32f(packed_bytes);
  ASSERT_EQ(packed.size(), source_width * 2u);
  for (UINT eye = 0; eye < 2u; ++eye) {
    const float eye_sign = eye == 0u ? -1.0f : 1.0f;
    for (UINT x = 0; x < source_width; ++x) {
      const float destination_u =
        (static_cast<float>(x) + 0.5f) / static_cast<float>(source_width);
      float source_u = destination_u;
      for (int iteration = 0; iteration < 11; ++iteration) {
        source_u = destination_u + eye_sign * sample_effective_parallax(
          positive_field,
          depth_width,
          depth_height,
          source_u,
          0.5f,
          geometry
        );
      }
      const auto expected = sample_linear_clamp(
        source,
        source_width,
        render_height,
        std::clamp(source_u, 0.0f, 1.0f),
        0.5f
      );
      const std::size_t packed_index = eye * source_width + x;
      expect_rgba_near(
        packed[packed_index],
        expected,
        3.0e-5f,
        "ROI collar eye=" + std::to_string(eye) + ", x=" + std::to_string(x)
      );
    }
  }

  constexpr UINT center_x = source_width / 2u - 1u;
  const float center_destination_u =
    (static_cast<float>(center_x) + 0.5f) / static_cast<float>(source_width);
  EXPECT_NEAR(
    (center_destination_u - packed[center_x].r) / roi_width,
    local_parallax,
    4.0e-5f
  );
  EXPECT_NEAR(
    (packed[source_width + center_x].r - center_destination_u) / roi_width,
    local_parallax,
    4.0e-5f
  );
  constexpr UINT identity_x = 48u;
  expect_rgba_near(
    packed[identity_x],
    source[identity_x],
    3.0e-6f,
    "left eye beyond collar"
  );
  expect_rgba_near(
    packed[source_width + identity_x],
    source[identity_x],
    3.0e-6f,
    "right eye beyond collar"
  );
}

TEST(HostSbsV2LiveWarpGpuTest, RoiMapsOnlyTheIntegerTensorContentRectangle) {
  constexpr UINT source_width = 128u;
  constexpr UINT source_height = 64u;
  constexpr UINT depth_width = 64u;
  constexpr UINT depth_height = 32u;
  constexpr host_sbs_v2_geometry_t geometry {
    .video_roi_active = 1.0f,
    .video_roi_left = 0.25f,
    .video_roi_top = 0.0f,
    .video_roi_right = 0.75f,
    .video_roi_bottom = 1.0f,
    .tensor_content_left = 16u,
    .tensor_content_top = 0u,
    .tensor_content_right = 48u,
    .tensor_content_bottom = 32u,
  };

  // The synthetic pillarbox is the exact nearest-boundary extension produced by the real map
  // pipeline. A varying interior makes sampling the whole tensor observably wrong even though no
  // invalid padding value is present.
  std::vector<float> field(static_cast<std::size_t>(depth_width) * depth_height);
  for (UINT y = 0u; y < depth_height; ++y) {
    for (UINT x = 0u; x < depth_width; ++x) {
      const UINT content_x = std::clamp(x, 16u, 47u);
      field[static_cast<std::size_t>(y) * depth_width + x] =
        0.004f + 0.006f * static_cast<float>(content_x - 16u) / 31.0f;
    }
  }
  std::vector<rgba32f_t> source(
    static_cast<std::size_t>(source_width) * source_height
  );
  for (UINT y = 0u; y < source_height; ++y) {
    for (UINT x = 0u; x < source_width; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / source_width;
      const float v = (static_cast<float>(y) + 0.5f) / source_height;
      source[static_cast<std::size_t>(y) * source_width + x] =
        {u, v, 0.25f + 0.5f * u, 1.0f};
    }
  }

  std::string error;
  live_v2_warp_fixture_t warp;
  ASSERT_TRUE(warp.initialize(error)) << error;
  std::vector<std::byte> packed_bytes;
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    source_width,
    source_height,
    source.data(),
    field,
    std::vector<float>(field.size(), 0.0f),
    make_live_state(true, true),
    {0.91f, 0.13f, 0.77f, 0.42f},
    packed_bytes,
    error,
    depth_width,
    depth_height,
    geometry
  )) << error;
  const auto packed = unpack_rgba32f(packed_bytes);
  ASSERT_EQ(packed.size(), source.size() * 2u);
  constexpr UINT y = source_height / 2u;
  const float destination_v = (static_cast<float>(y) + 0.5f) / source_height;
  for (UINT eye = 0u; eye < 2u; ++eye) {
    const float eye_sign = eye == 0u ? -1.0f : 1.0f;
    for (UINT x = 8u; x + 8u < source_width; x += 7u) {
      const float destination_u = (static_cast<float>(x) + 0.5f) / source_width;
      float source_u = destination_u;
      for (int iteration = 0; iteration < 11; ++iteration) {
        source_u = destination_u + eye_sign * sample_effective_parallax(
          field,
          depth_width,
          depth_height,
          source_u,
          destination_v,
          geometry
        );
      }
      const auto expected = sample_linear_clamp(
        source,
        source_width,
        source_height,
        std::clamp(source_u, 0.0f, 1.0f),
        destination_v
      );
      const auto packed_index = static_cast<std::size_t>(y) * source_width * 2u +
                                eye * source_width + x;
      expect_rgba_near(
        packed[packed_index],
        expected,
        4.0e-5f,
        "letterbox content mapping eye=" + std::to_string(eye) +
          ", x=" + std::to_string(x)
      );
    }
  }
}

TEST(HostSbsV2LiveWarpGpuTest, AdaptiveContinuousCollarSamplesGradientAndPreservesSignedSpill) {
  namespace v2 = models::depth_coordinate_v2;
  constexpr UINT width = 512u;
  constexpr UINT height = 64u;
  constexpr UINT depth_width = 64u;
  constexpr UINT depth_height = 32u;
  constexpr host_sbs_v2_geometry_t geometry {
    .video_roi_active = 1.0f,
    .video_roi_left = 0.25f,
    .video_roi_top = 0.25f,
    .video_roi_right = 0.75f,
    .video_roi_bottom = 0.75f,
    .tensor_content_right = depth_width,
    .tensor_content_bottom = depth_height,
  };
  std::vector<rgba32f_t> source(static_cast<std::size_t>(width) * height);
  for (UINT y = 0u; y < height; ++y) {
    for (UINT x = 0u; x < width; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / width;
      const float v = (static_cast<float>(y) + 0.5f) / height;
      source[static_cast<std::size_t>(y) * width + x] = rgba32f_t {u, v, 3.0f + u - v, 1.0f};
    }
  }
  auto state = make_live_state(true, true);
  state[v2::joint_plane_mode_bits] = 3u;
  state[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(state);
  std::string error;
  live_v2_warp_fixture_t color;
  live_v2_warp_fixture_t mapping;
  ASSERT_TRUE(color.initialize(error)) << error;
  ASSERT_TRUE(mapping.initialize(error, warp_shader_e::live_mapping)) << error;
  for (const float sign : {1.0f, -1.0f}) {
    SCOPED_TRACE(sign);
    const std::vector<float> field(static_cast<std::size_t>(depth_width) * depth_height,
      sign * 0.03125f);
    const auto render = [&](live_v2_warp_fixture_t &fixture) {
      std::vector<std::byte> bytes;
      EXPECT_TRUE(fixture.render(DXGI_FORMAT_R32G32B32A32_FLOAT, sizeof(rgba32f_t),
        width, height, source.data(), field, field, state, {0.9f, 0.1f, 0.7f, 0.4f},
        bytes, error, depth_width, depth_height, geometry)) << error;
      return bytes;
    };
    const auto pixels = unpack_rgba32f(render(color));
    const auto map_bytes = render(mapping);
    std::vector<float> map(map_bytes.size() / sizeof(float));
    std::memcpy(map.data(), map_bytes.data(), map_bytes.size());
    ASSERT_EQ(map.size(), source.size() * 2u);
    ASSERT_EQ(pixels.size(), map.size());
    const auto at = [&](UINT eye, UINT x, UINT y) {
      return static_cast<std::size_t>(y) * width * 2u + eye * width + x;
    };
    for (UINT eye = 0u; eye < 2u; ++eye) {
      for (UINT y = 0u; y < height; ++y) {
        for (UINT x = 0u; x < width; ++x) {
          const auto index = at(eye, x, y);
          const auto expected = sample_linear_clamp(source, width, height, map[index],
            (static_cast<float>(y) + 0.5f) / height);
          expect_rgba_near(pixels[index], expected, 3.0e-5f, "one full-capture sample");
          // Both signs, horizontal joins, top/bottom joins and their corners. The inverse of
          // a 0.5-Lipschitz row is increasing and has adjacent slope between 2/3 and 2.
          if (x > 0u) {
            const float step = (map[index] - map[index - 1u]) * width;
            EXPECT_GE(step, 0.665f);
            EXPECT_LE(step, 2.001f);
          }
          if (y > 0u) {
            EXPECT_LE(std::abs(map[index] - map[at(eye, x, y - 1u)]),
              2.0f / height + 2.0e-6f);
          }
        }
      }
      for (const UINT x : {32u, 480u}) {
        EXPECT_NEAR(map[at(eye, x, 32u)], (static_cast<float>(x) + 0.5f) / width, 2.0e-7f);
        expect_rgba_near(pixels[at(eye, x, 32u)], source[32u * width + x],
          3.0e-6f, "far outside collar");
      }
      for (const UINT y : {2u, 61u}) {
        EXPECT_NEAR(map[at(eye, 256u, y)], 256.5f / width, 2.0e-7f);
      }
    }
    // The source ROI still projects beyond its rectangle. Near positive geometry moves right
    // in the left eye and left in the right eye; negative geometry has the opposite direction.
    const UINT right_spill_eye = sign > 0.0f ? 0u : 1u;
    const UINT left_spill_eye = 1u - right_spill_eye;
    EXPECT_LT(map[at(right_spill_eye, 386u, 32u)], geometry.video_roi_right);
    EXPECT_GE(map[at(right_spill_eye, 386u, 32u)], geometry.video_roi_left);
    EXPECT_GT(map[at(left_spill_eye, 125u, 32u)], geometry.video_roi_left);
    // The retracted right-side band samples distinct captured browser columns. The old one
    // adjacent-column fill would make every U and ramp color in this interval identical.
    const UINT retract_eye = 1u - right_spill_eye;
    EXPECT_GT(map[at(retract_eye, 378u, 32u)], geometry.video_roi_right);
    for (UINT x = 379u; x < 384u; ++x) {
      EXPECT_GT(map[at(retract_eye, x, 32u)], map[at(retract_eye, x - 1u, 32u)]);
      EXPECT_GT(pixels[at(retract_eye, x, 32u)].r, pixels[at(retract_eye, x - 1u, 32u)].r);
    }
    EXPECT_GT((map[at(retract_eye, 383u, 32u)] - map[at(retract_eye, 378u, 32u)]) * width, 3.0f);
  }
}

TEST(HostSbsV2LiveWarpGpuTest, AdaptiveContinuousMaskHasOnlyFiniteImageBoundaryAuthority) {
  namespace v2 = models::depth_coordinate_v2;
  constexpr UINT width = 64u;
  constexpr UINT height = 32u;
  constexpr host_sbs_v2_geometry_t geometry {
    .video_roi_active = 1.0f,
    .video_roi_right = 1.0f,
    .video_roi_bottom = 1.0f,
    .tensor_content_right = width,
    .tensor_content_bottom = height,
  };
  const std::vector<bgra8_t> source(static_cast<std::size_t>(width) * height, {192u,128u,64u,255u});
  const std::vector<float> field(source.size(), 0.03125f);
  auto state = make_live_state(true, true);
  state[v2::joint_plane_mode_bits] = 3u;
  state[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(state);
  std::string error;
  live_v2_warp_fixture_t mask;
  live_v2_warp_fixture_t mapping;
  ASSERT_TRUE(mask.initialize(error, warp_shader_e::live_mask)) << error;
  ASSERT_TRUE(mapping.initialize(error, warp_shader_e::live_mapping)) << error;
  const auto render = [&](live_v2_warp_fixture_t &fixture) {
    std::vector<std::byte> bytes;
    EXPECT_TRUE(fixture.render(DXGI_FORMAT_B8G8R8A8_UNORM, sizeof(bgra8_t),
      width, height, source.data(), field, field, state, {0,0,0,0},
      bytes, error, width, height, geometry)) << error;
    return bytes;
  };
  const auto mask_bytes = render(mask);
  const auto map_bytes = render(mapping);
  std::vector<bgra8_t> encoded(mask_bytes.size() / sizeof(bgra8_t));
  std::vector<float> selected(map_bytes.size() / sizeof(float));
  std::memcpy(encoded.data(), mask_bytes.data(), mask_bytes.size());
  std::memcpy(selected.data(), map_bytes.data(), map_bytes.size());
  ASSERT_EQ(encoded.size(), source.size() * 2u);
  ASSERT_EQ(selected.size(), encoded.size());
  std::size_t boundary_count = 0u;
  for (std::size_t index = 0u; index < encoded.size(); ++index) {
    const bool outside = selected[index] < 0.0f || selected[index] > 1.0f;
    EXPECT_EQ(encoded[index].r, outside ? 255u : 0u);
    EXPECT_EQ(encoded[index].g, 0u);
    EXPECT_EQ(encoded[index].b, 0u);
    EXPECT_EQ(encoded[index].a, 255u);
    boundary_count += outside ? 1u : 0u;
  }
  EXPECT_GT(boundary_count, 0u);
}

TEST(HostSbsV2LiveWarpGpuTest, AdaptiveContinuousRoiUsesDeclaredPaddedFieldEmbedding) {
  namespace v2 = models::depth_coordinate_v2;
  constexpr UINT width = 256u;
  constexpr UINT height = 64u;
  constexpr UINT depth_width = 64u;
  constexpr UINT depth_height = 32u;
  constexpr host_sbs_v2_geometry_t geometry {
    .video_roi_active = 1.0f,
    .video_roi_left = 0.25f,
    .video_roi_top = 0.25f,
    .video_roi_right = 0.75f,
    .video_roi_bottom = 0.75f,
    .tensor_content_left = 16u,
    .tensor_content_top = 8u,
    .tensor_content_right = 48u,
    .tensor_content_bottom = 24u,
  };
  // The production sampler maps the real-content rectangle boundary, preserving its native
  // bilinear footprint there. Padding is slope-valid, so a half-tap at that boundary does not
  // fabricate an impossible sign cliff or assert the removed mode3 texel-center clamp.
  std::vector<float> field(static_cast<std::size_t>(depth_width) * depth_height);
  for (UINT y = 0u; y < depth_height; ++y) {
    for (UINT x = 0u; x < depth_width; ++x) {
      field[static_cast<std::size_t>(y) * depth_width + x] =
        0.001f * (static_cast<float>(x) - 31.5f);
    }
  }
  const std::vector<rgba32f_t> source(static_cast<std::size_t>(width) * height,
    {0.25f, 0.5f, 0.75f, 1});
  std::string error;
  live_v2_warp_fixture_t mapping;
  ASSERT_TRUE(mapping.initialize(error, warp_shader_e::live_mapping)) << error;
  const auto render = [&](std::uint32_t mode, const host_sbs_v2_geometry_t &placement) {
    auto state = make_live_state(true, true);
    state[v2::joint_plane_mode_bits] = mode;
    state[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(state);
    std::vector<std::byte> bytes;
    EXPECT_TRUE(mapping.render(DXGI_FORMAT_R32G32B32A32_FLOAT, sizeof(rgba32f_t),
      width, height, source.data(), field, field, state, {0, 0, 0, 0},
      bytes, error, depth_width, depth_height, placement)) << error;
    return bytes;
  };
  const auto adaptive_bytes = render(3u, geometry);
  ASSERT_EQ(adaptive_bytes.size(), source.size() * 2u * sizeof(float));
  // Declaring padding as real content changes its spatial embedding. Comparing two actual GPU
  // maps makes this observable independently of mode parity and needs no CPU inverse oracle.
  auto wrong_embedding = geometry;
  wrong_embedding.tensor_content_left = 0u;
  wrong_embedding.tensor_content_top = 0u;
  wrong_embedding.tensor_content_right = depth_width;
  wrong_embedding.tensor_content_bottom = depth_height;
  const auto wrong_bytes = render(3u, wrong_embedding);
  ASSERT_EQ(wrong_bytes.size(), adaptive_bytes.size());
  std::vector<float> selected(adaptive_bytes.size() / sizeof(float));
  std::vector<float> wrong(wrong_bytes.size() / sizeof(float));
  std::memcpy(selected.data(), adaptive_bytes.data(), adaptive_bytes.size());
  std::memcpy(wrong.data(), wrong_bytes.data(), wrong_bytes.size());
  float maximum_difference_pixels = 0.0f;
  for (std::size_t index = 0u; index < selected.size(); ++index) {
    maximum_difference_pixels = std::max(maximum_difference_pixels,
      std::abs(selected[index] - wrong[index]) * width);
  }
  EXPECT_GT(maximum_difference_pixels, 1.0f);
}

TEST(HostSbsV2LiveWarpGpuTest, AdaptiveRoiInvalidGeometryAndMissingAuthorityStayIdentity) {
  namespace v2 = models::depth_coordinate_v2;
  constexpr UINT width = 64u;
  constexpr UINT height = 32u;
  constexpr UINT depth_width = 16u;
  constexpr UINT depth_height = 8u;
  constexpr host_sbs_v2_geometry_t geometry {
    .video_roi_active = 1.0f,
    .video_roi_left = 0.25f,
    .video_roi_top = 0.25f,
    .video_roi_right = 0.75f,
    .video_roi_bottom = 0.75f,
    .tensor_content_right = depth_width,
    .tensor_content_bottom = depth_height,
  };
  std::vector<rgba32f_t> source(static_cast<std::size_t>(width) * height);
  for (UINT y = 0u; y < height; ++y) {
    for (UINT x = 0u; x < width; ++x) {
      source[static_cast<std::size_t>(y) * width + x] = {
        (static_cast<float>(x) + 0.5f) / width,
        (static_cast<float>(y) + 0.5f) / height,
        0.625f,
        1.0f
      };
    }
  }
  auto state = make_live_state(true, true);
  state[v2::joint_plane_mode_bits] = 3u;
  state[v2::camera_center_integrity_bits] =
    v2::camera_center_integrity_for_state_words(state);
  const std::vector<float> field(static_cast<std::size_t>(depth_width) * depth_height, 0.04f);
  std::string error;
  live_v2_warp_fixture_t color;
  ASSERT_TRUE(color.initialize(error)) << error;
  const auto expect_identity = [&](const host_sbs_v2_geometry_t &tested_geometry, const v2::state_words_t &tested_state) {
    std::vector<std::byte> bytes;
    ASSERT_TRUE(color.render(
      DXGI_FORMAT_R32G32B32A32_FLOAT,
      sizeof(rgba32f_t),
      width,
      height,
      source.data(),
      field,
      field,
      tested_state,
      {0.91f, 0.13f, 0.77f, 0.42f},
      bytes,
      error,
      depth_width,
      depth_height,
      tested_geometry
    )) << error;
    const auto packed = unpack_rgba32f(bytes);
    ASSERT_EQ(packed.size(), source.size() * 2u);
    for (UINT y = 0u; y < height; ++y) {
      for (UINT eye = 0u; eye < 2u; ++eye) {
        for (UINT x = 0u; x < width; ++x) {
          expect_rgba_near(
            packed[static_cast<std::size_t>(y) * width * 2u + eye * width + x],
            source[static_cast<std::size_t>(y) * width + x],
            3.0e-6f,
            "invalid adaptive ROI identity"
          );
        }
      }
    }
  };
  auto invalid = geometry;
  invalid.video_roi_left = std::numeric_limits<float>::quiet_NaN();
  expect_identity(invalid, state);
  invalid = geometry;
  invalid.video_roi_right = 1.125f;
  expect_identity(invalid, state);
  invalid = geometry;
  invalid.video_roi_bottom = invalid.video_roi_top;
  expect_identity(invalid, state);
  invalid = geometry;
  invalid.tensor_content_right = depth_width + 1u;
  expect_identity(invalid, state);
  invalid = geometry;
  invalid.tensor_content_bottom = 0u;
  expect_identity(invalid, state);
  invalid = geometry;
  invalid.video_roi_active = 0.5f;
  expect_identity(invalid, state);
  invalid = geometry;
  invalid.reserved = 1.0f;
  expect_identity(invalid, state);
  auto unauthenticated = state;
  unauthenticated[v2::renderer_authorization_bits] = 0u;
  expect_identity(geometry, unauthenticated);
}

TEST(HostSbsV2LiveWarpGpuTest, CandidateFieldCannotBlurOrAlterLiveColor) {
  constexpr UINT width = 67u;
  constexpr UINT height = 53u;
  const std::array<float, 4> sentinel_clear {0.91f, 0.13f, 0.77f, 0.42f};
  std::string error;
  live_v2_warp_fixture_t warp;
  ASSERT_TRUE(warp.initialize(error)) << error;

  // Prime-sized, non-periodic, spatially varying values expose any accidental extra color tap.
  // Negative and >1 values also exercise the active linear-scRGB policy.
  std::vector<rgba32f_t> source(static_cast<std::size_t>(width) * height);
  for (UINT y = 0; y < height; ++y) {
    for (UINT x = 0; x < width; ++x) {
      source[static_cast<std::size_t>(y) * width + x] = {
        static_cast<float>((17u * x + 29u * y + 3u) % 113u) / 37.0f - 0.8f,
        static_cast<float>((31u * x + 11u * y + 7u) % 127u) / 41.0f - 0.6f,
        static_cast<float>((13u * x + 43u * y + 5u) % 109u) / 35.0f - 0.7f,
        1.0f,
      };
    }
  }

  const std::vector<float> zero_final(source.size(), 0.0f);
  auto render_candidate = [&](const float deviation_source_px) {
    std::vector<std::byte> bytes;
    const std::vector<float> candidate(
      source.size(),
      -deviation_source_px / static_cast<float>(width)
    );
    EXPECT_TRUE(warp.render(
      DXGI_FORMAT_R32G32B32A32_FLOAT,
      sizeof(rgba32f_t),
      width,
      height,
      source.data(),
      zero_final,
      candidate,
      make_live_state(true, true),
      sentinel_clear,
      bytes,
      error
    )) << error;
    return unpack_rgba32f(bytes);
  };

  const auto inactive = render_candidate(0.0f);
  const auto onset = render_candidate(4.0f);
  const auto quarter_response = render_candidate(8.0f);
  const auto active = render_candidate(20.0f);
  const auto saturated = render_candidate(40.0f);
  ASSERT_EQ(inactive.size(), source.size() * 2u);
  ASSERT_EQ(inactive.size(), active.size());
  ASSERT_EQ(inactive.size(), onset.size());
  ASSERT_EQ(inactive.size(), quarter_response.size());
  ASSERT_EQ(active.size(), saturated.size());

  float active_max_difference = 0.0f;
  // smoothstep(4, 20, 8) = smoothstep(0, 1, 0.25) = 0.15625.
  constexpr float expected_quarter_response = 0.15625f;
  for (UINT eye = 0; eye < 2u; ++eye) {
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        const std::size_t packed_index =
          static_cast<std::size_t>(y) * width * 2u + eye * width + x;
        const std::size_t source_index = static_cast<std::size_t>(y) * width + x;
        const std::string where =
          "eye=" + std::to_string(eye) + ", x=" + std::to_string(x) +
          ", y=" + std::to_string(y);
        const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        const auto exact_blur = one_tap_reference(
          source, width, height, u, v);
        expect_rgba_near(inactive[packed_index], source[source_index], 3.0e-6f, where);
        expect_rgba_near(onset[packed_index], inactive[packed_index], 3.0e-6f, where);
        // Candidate displacement is diagnostic geometry evidence only. It must not soften or
        // otherwise alter the one-tap live color sample at any deviation magnitude.
        expect_rgba_near(active[packed_index], exact_blur, 1.0e-3f, where);
        expect_rgba_near(saturated[packed_index], exact_blur, 1.0e-3f, where);
        expect_rgba_near(
          quarter_response[packed_index],
          lerp_rgba(inactive[packed_index], exact_blur, expected_quarter_response),
          1.0e-3f,
          where
        );
        active_max_difference = std::max(
          active_max_difference,
          std::abs(active[packed_index].r - inactive[packed_index].r)
        );
      }
    }
  }
  EXPECT_LT(active_max_difference, 3.0e-6f);

  // A negative final-minus-candidate deviation is foreground compression and remains on the
  // exact one-tap path even when its magnitude is above the positive collar threshold.
  std::vector<std::byte> negative_bytes;
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    width,
    height,
    source.data(),
    zero_final,
    std::vector<float>(source.size(), 20.0f / static_cast<float>(width)),
    make_live_state(true, true),
    sentinel_clear,
    negative_bytes,
    error
  )) << error;
  const auto negative = unpack_rgba32f(negative_bytes);
  ASSERT_EQ(negative.size(), inactive.size());
  for (std::size_t index = 0; index < inactive.size(); ++index) {
    expect_rgba_near(negative[index], inactive[index], 3.0e-6f, "negative mask");
  }

  // Validate mask/filter sampling after a real, nonzero inverse warp. A constant one-pixel
  // parallax has an exact inverse: left samples x-1 and right samples x+1.
  const float one_pixel_parallax = 1.0f / static_cast<float>(width);
  std::vector<std::byte> shifted_bytes;
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    width,
    height,
    source.data(),
    std::vector<float>(source.size(), one_pixel_parallax),
    std::vector<float>(
      source.size(), one_pixel_parallax - 20.0f / static_cast<float>(width)),
    make_live_state(true, true),
    sentinel_clear,
    shifted_bytes,
    error
  )) << error;
  const auto shifted = unpack_rgba32f(shifted_bytes);
  ASSERT_EQ(shifted.size(), inactive.size());
  for (UINT eye = 0; eye < 2u; ++eye) {
    const float eye_sign = eye == 0u ? -1.0f : 1.0f;
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        const std::size_t packed_index =
          static_cast<std::size_t>(y) * width * 2u + eye * width + x;
        const float u = std::clamp(
          (static_cast<float>(x) + 0.5f) / static_cast<float>(width) +
            eye_sign * one_pixel_parallax,
          0.0f,
          1.0f
        );
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        expect_rgba_near(
          shifted[packed_index],
          one_tap_reference(source, width, height, u, v),
          1.5e-3f,
          "nonzero inverse warp"
        );
      }
    }
  }

  // Production binds the depth fields at the smaller DAV2 resolution. Exercise nonconstant
  // geometry and a spatially varying candidate delta so coordinate normalization, bilinear depth
  // sampling, the 11-step inverse, and source-pixel mask scaling are all independently checked.
  constexpr UINT depth_width = 19u;
  constexpr UINT depth_height = 13u;
  std::vector<rgba32f_t> smooth_source(source.size());
  for (UINT y = 0; y < height; ++y) {
    for (UINT x = 0; x < width; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
      const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
      smooth_source[static_cast<std::size_t>(y) * width + x] = {
        0.15f + 0.55f * u + 0.12f * std::sin(12.5663706f * u),
        0.20f + 0.45f * v + 0.10f * std::cos(9.4247780f * v),
        0.10f + 0.30f * u + 0.25f * v +
          0.08f * std::sin(6.2831853f * (u + v)),
        1.0f,
      };
    }
  }
  std::vector<float> final_depth(static_cast<std::size_t>(depth_width) * depth_height);
  std::vector<float> candidate_depth(final_depth.size());
  for (UINT y = 0; y < depth_height; ++y) {
    for (UINT x = 0; x < depth_width; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(depth_width);
      const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(depth_height);
      const std::size_t index = static_cast<std::size_t>(y) * depth_width + x;
      final_depth[index] = 0.008f * (u - 0.5f) + 0.003f * (v - 0.5f);
      const float deviation_source_px = 2.0f + 24.0f * u * (1.0f - 0.35f * v);
      candidate_depth[index] =
        final_depth[index] - deviation_source_px / static_cast<float>(width);
    }
  }
  std::vector<std::byte> production_ratio_bytes;
  ASSERT_TRUE(warp.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    width,
    height,
    smooth_source.data(),
    final_depth,
    candidate_depth,
    make_live_state(true, true),
    sentinel_clear,
    production_ratio_bytes,
    error,
    depth_width,
    depth_height
  )) << error;
  const auto production_ratio = unpack_rgba32f(production_ratio_bytes);
  ASSERT_EQ(production_ratio.size(), inactive.size());
  float production_max_defocus = 0.0f;
  for (UINT eye = 0; eye < 2u; ++eye) {
    const float eye_sign = eye == 0u ? -1.0f : 1.0f;
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        const std::size_t packed_index =
          static_cast<std::size_t>(y) * width * 2u + eye * width + x;
        const float destination_u =
          (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        float source_u = destination_u;
        for (int iteration = 0; iteration < 11; ++iteration) {
          source_u = destination_u + eye_sign * sample_linear_clamp(
            final_depth, depth_width, depth_height, source_u, v);
        }
        source_u = std::clamp(source_u, 0.0f, 1.0f);
        const float final_sample = sample_linear_clamp(
          final_depth, depth_width, depth_height, source_u, v);
        const float candidate_sample = sample_linear_clamp(
          candidate_depth, depth_width, depth_height, source_u, v);
        const float deviation_source_px =
          std::max(final_sample - candidate_sample, 0.0f) * static_cast<float>(width);
        const float response_t = std::clamp((deviation_source_px - 4.0f) / 16.0f, 0.0f, 1.0f);
        const float response = response_t * response_t * (3.0f - 2.0f * response_t);
        const auto center = sample_linear_clamp(
          smooth_source, width, height, source_u, v);
        const auto blur = one_tap_reference(
          smooth_source, width, height, source_u, v);
        production_max_defocus = std::max(
          production_max_defocus,
          std::abs(blur.r - center.r)
        );
        expect_rgba_near(
          production_ratio[packed_index],
          lerp_rgba(center, blur, response),
          8.0e-4f,
          "production depth/color ratio"
        );
      }
    }
  }
  EXPECT_LT(production_max_defocus, 3.0e-6f);
}

TEST(HostSbsV2LiveWarpGpuTest, ExactEarlyExitMatchesUnconditionalElevenIterationsBitForBit) {
  namespace v2 = models::depth_coordinate_v2;
  constexpr UINT source_width = 32u;
  constexpr UINT source_height = 12u;
  constexpr UINT depth_width = 16u;
  constexpr UINT depth_height = 8u;
  const std::array<float, 4> sentinel_clear {0.91f, 0.13f, 0.77f, 0.42f};

  std::vector<bgra8_t> source(
    static_cast<std::size_t>(source_width) * source_height
  );
  for (UINT y = 0; y < source_height; ++y) {
    for (UINT x = 0; x < source_width; ++x) {
      source[static_cast<std::size_t>(y) * source_width + x] = {
        static_cast<std::uint8_t>((17u * x + 31u * y + 11u) & 0xffu),
        static_cast<std::uint8_t>((47u * x + 13u * y + 23u) & 0xffu),
        static_cast<std::uint8_t>((29u * x + 53u * y + 37u) & 0xffu),
        0xffu,
      };
    }
  }

  const std::size_t depth_elements =
    static_cast<std::size_t>(depth_width) * depth_height;
  const std::vector<float> zero_field(depth_elements, 0.0f);
  const std::vector<float> constant_field(depth_elements, 0.011f);
  std::vector<float> complex_field(depth_elements);
  std::vector<float> worst_slope_field(depth_elements);
  for (UINT y = 0; y < depth_height; ++y) {
    for (UINT x = 0; x < depth_width; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) /
        static_cast<float>(depth_width);
      const float v = (static_cast<float>(y) + 0.5f) /
        static_cast<float>(depth_height);
      complex_field[static_cast<std::size_t>(y) * depth_width + x] =
        0.014f * (u - 0.5f) + 0.005f * (v - 0.5f) +
        0.002f * std::sin(6.2831853f * (u + v));
      worst_slope_field[static_cast<std::size_t>(y) * depth_width + x] =
        std::clamp(0.5f * (u - 0.5f), -0.04f, 0.04f);
    }
  }

  host_sbs_v2_geometry_t roi_geometry {};
  roi_geometry.video_roi_active = 1.0f;
  roi_geometry.video_roi_left = 0.1875f;
  roi_geometry.video_roi_top = 1.0f / 6.0f;
  roi_geometry.video_roi_right = 0.8125f;
  roi_geometry.video_roi_bottom = 5.0f / 6.0f;
  roi_geometry.tensor_content_left = 2u;
  roi_geometry.tensor_content_top = 1u;
  roi_geometry.tensor_content_right = 14u;
  roi_geometry.tensor_content_bottom = 7u;

  live_v2_warp_fixture_t live_main;
  live_v2_warp_fixture_t live_main_reference;
  live_v2_warp_fixture_t live_mapping;
  live_v2_warp_fixture_t live_mapping_reference;
  live_v2_warp_fixture_t direct_main;
  live_v2_warp_fixture_t direct_main_reference;
  live_v2_warp_fixture_t direct_mapping;
  live_v2_warp_fixture_t direct_mapping_reference;
  std::string error;
  ASSERT_TRUE(live_main.initialize(error, warp_shader_e::live_main)) << error;
  ASSERT_TRUE(live_main_reference.initialize(
    error, warp_shader_e::live_main, true
  )) << error;
  ASSERT_TRUE(live_mapping.initialize(error, warp_shader_e::live_mapping)) << error;
  ASSERT_TRUE(live_mapping_reference.initialize(
    error, warp_shader_e::live_mapping, true
  )) << error;
  ASSERT_TRUE(direct_main.initialize(error, warp_shader_e::direct_main)) << error;
  ASSERT_TRUE(direct_main_reference.initialize(
    error, warp_shader_e::direct_main, true
  )) << error;
  ASSERT_TRUE(direct_mapping.initialize(
    error, warp_shader_e::direct_mapping
  )) << error;
  ASSERT_TRUE(direct_mapping_reference.initialize(
    error, warp_shader_e::direct_mapping, true
  )) << error;

  const auto compare_exact_and_reference = [&] (
    live_v2_warp_fixture_t &exact,
    live_v2_warp_fixture_t &reference,
    const std::vector<float> &field,
    const host_sbs_v2_geometry_t &geometry,
    const std::string &where,
    const std::uint32_t mode = v2::adaptive_policy_id
  ) {
    SCOPED_TRACE(where);
    auto state = make_live_state(true, true);
    state[v2::joint_plane_mode_bits] = mode;
    state[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(state);
    std::vector<std::byte> exact_bytes;
    std::vector<std::byte> reference_bytes;
    ASSERT_TRUE(exact.render(
      DXGI_FORMAT_B8G8R8A8_UNORM,
      sizeof(bgra8_t),
      source_width,
      source_height,
      source.data(),
      field,
      zero_field,
      state,
      sentinel_clear,
      exact_bytes,
      error,
      depth_width,
      depth_height,
      geometry
    )) << error;
    ASSERT_TRUE(reference.render(
      DXGI_FORMAT_B8G8R8A8_UNORM,
      sizeof(bgra8_t),
      source_width,
      source_height,
      source.data(),
      field,
      zero_field,
      state,
      sentinel_clear,
      reference_bytes,
      error,
      depth_width,
      depth_height,
      geometry
    )) << error;
    EXPECT_EQ(exact_bytes, reference_bytes);
  };

  // Compare both the packed color and the R32 inverse coordinate. The R32 mapping witnesses
  // coordinate bits directly, so a texture sample cannot hide a difference at a flat color or
  // within one bilinear footprint. Zero and constant fields settle early; the spatial field
  // exercises ordinary varying geometry, and the clamped 0.5-slope ramp covers the authenticated
  // worst contraction where the hard 11-step cap remains necessary.
  const std::array<std::pair<const char *, const std::vector<float> *>, 4>
    signed_fields {{
      {"zero", &zero_field},
      {"constant", &constant_field},
      {"complex", &complex_field},
      {"worst-slope", &worst_slope_field},
    }};
  for (const auto &[name, field] : signed_fields) {
    compare_exact_and_reference(
      live_main,
      live_main_reference,
      *field,
      host_sbs_v2_geometry_t {},
      std::string("live full packed ") + name
    );
    compare_exact_and_reference(
      live_mapping,
      live_mapping_reference,
      *field,
      host_sbs_v2_geometry_t {},
      std::string("live full mapping ") + name
    );
    compare_exact_and_reference(
      live_main,
      live_main_reference,
      *field,
      roi_geometry,
      std::string("live ROI packed ") + name
    );
    compare_exact_and_reference(
      live_mapping,
      live_mapping_reference,
      *field,
      roi_geometry,
      std::string("live ROI mapping ") + name
    );
  }

  for (const auto &[name, field] : signed_fields) {
    compare_exact_and_reference(live_main, live_main_reference, *field,
      host_sbs_v2_geometry_t {}, std::string("mode3 full packed ") + name, 3u);
    compare_exact_and_reference(live_mapping, live_mapping_reference, *field,
      host_sbs_v2_geometry_t {}, std::string("mode3 full mapping ") + name, 3u);
    compare_exact_and_reference(live_main, live_main_reference, *field,
      roi_geometry, std::string("mode3 ROI packed ") + name, 3u);
    compare_exact_and_reference(live_mapping, live_mapping_reference, *field,
      roi_geometry, std::string("mode3 ROI mapping ") + name, 3u);
  }

  const auto direct_encode = [] (const std::vector<float> &field) {
    std::vector<float> encoded(field.size());
    for (std::size_t index = 0; index < field.size(); ++index) {
      encoded[index] = 0.5f + field[index] /
        (2.0f * v2::direct_container_limit);
    }
    return encoded;
  };
  for (const auto &[name, signed_field] : signed_fields) {
    const auto encoded_field = direct_encode(*signed_field);
    compare_exact_and_reference(
      direct_main,
      direct_main_reference,
      encoded_field,
      host_sbs_v2_geometry_t {},
      std::string("direct packed ") + name
    );
    compare_exact_and_reference(
      direct_mapping,
      direct_mapping_reference,
      encoded_field,
      host_sbs_v2_geometry_t {},
      std::string("direct mapping ") + name
    );
  }
}

TEST(HostSbsV2LiveWarpGpuTest, GeometryFastPathPreservesDormantAndFailFlatSemantics) {
  constexpr UINT source_width = 32u;
  constexpr UINT source_height = 8u;
  constexpr UINT depth_width = 16u;
  constexpr UINT depth_height = 8u;
  const std::size_t depth_elements =
    static_cast<std::size_t>(depth_width) * depth_height;
  const std::vector<float> parallax(depth_elements, 0.011f);
  const std::vector<float> candidate(depth_elements, 0.0f);
  std::vector<bgra8_t> source(
    static_cast<std::size_t>(source_width) * source_height
  );
  for (UINT y = 0u; y < source_height; ++y) {
    for (UINT x = 0u; x < source_width; ++x) {
      source[static_cast<std::size_t>(y) * source_width + x] = {
        static_cast<std::uint8_t>((17u * x + 31u * y + 11u) & 0xffu),
        static_cast<std::uint8_t>((47u * x + 13u * y + 23u) & 0xffu),
        static_cast<std::uint8_t>((29u * x + 53u * y + 37u) & 0xffu),
        0xffu,
      };
    }
  }

  host_sbs_v2_geometry_t valid_roi {};
  valid_roi.video_roi_active = 1.0f;
  valid_roi.video_roi_left = 0.1875f;
  valid_roi.video_roi_top = 0.125f;
  valid_roi.video_roi_right = 0.8125f;
  valid_roi.video_roi_bottom = 0.875f;
  valid_roi.tensor_content_left = 2u;
  valid_roi.tensor_content_top = 1u;
  valid_roi.tensor_content_right = 14u;
  valid_roi.tensor_content_bottom = 7u;

  struct geometry_case_t {
    const char *name;
    host_sbs_v2_geometry_t geometry;
    bool full_warp;
    bool flat_identity;
  };
  std::array<geometry_case_t, 12u> cases;
  cases[0] = {"valid full", host_sbs_v2_geometry_t {}, true, false};
  auto dormant_malformed = host_sbs_v2_geometry_t {};
  dormant_malformed.video_roi_active = std::bit_cast<float>(0x80000000u);
  dormant_malformed.reserved = std::bit_cast<float>(0x80000000u);
  dormant_malformed.video_roi_left = std::numeric_limits<float>::quiet_NaN();
  dormant_malformed.video_roi_top = std::numeric_limits<float>::infinity();
  dormant_malformed.video_roi_right = -std::numeric_limits<float>::infinity();
  dormant_malformed.video_roi_bottom = std::numeric_limits<float>::quiet_NaN();
  dormant_malformed.tensor_content_left = UINT32_MAX;
  dormant_malformed.tensor_content_top = UINT32_MAX;
  dormant_malformed.tensor_content_right = 0u;
  dormant_malformed.tensor_content_bottom = 0u;
  cases[1] = {"dormant malformed signed-zero full", dormant_malformed, true, false};
  cases[2] = {"valid ROI", valid_roi, false, false};
  auto invalid_mode = valid_roi;
  invalid_mode.video_roi_active = 0.5f;
  cases[3] = {"invalid mode", invalid_mode, false, true};
  auto invalid_mode_nan = valid_roi;
  invalid_mode_nan.video_roi_active = std::numeric_limits<float>::quiet_NaN();
  cases[4] = {"invalid mode NaN", invalid_mode_nan, false, true};
  auto invalid_reserved = valid_roi;
  invalid_reserved.reserved = std::numeric_limits<float>::quiet_NaN();
  cases[5] = {"invalid reserved NaN", invalid_reserved, false, true};
  auto invalid_roi_nan = valid_roi;
  invalid_roi_nan.video_roi_left = std::numeric_limits<float>::quiet_NaN();
  cases[6] = {"invalid ROI NaN", invalid_roi_nan, false, true};
  auto invalid_roi_inf = valid_roi;
  invalid_roi_inf.video_roi_bottom = std::numeric_limits<float>::infinity();
  cases[7] = {"invalid ROI infinity", invalid_roi_inf, false, true};
  auto invalid_roi_bounds = valid_roi;
  invalid_roi_bounds.video_roi_left = -0.01f;
  invalid_roi_bounds.video_roi_right = 1.01f;
  cases[8] = {"invalid ROI bounds", invalid_roi_bounds, false, true};
  auto invalid_roi_degenerate = valid_roi;
  invalid_roi_degenerate.video_roi_right = invalid_roi_degenerate.video_roi_left;
  cases[9] = {"invalid ROI degenerate", invalid_roi_degenerate, false, true};
  auto invalid_content_degenerate = valid_roi;
  invalid_content_degenerate.tensor_content_right =
    invalid_content_degenerate.tensor_content_left;
  cases[10] = {
    "invalid content degenerate", invalid_content_degenerate, false, true
  };
  auto invalid_content_extent = valid_roi;
  invalid_content_extent.tensor_content_right = depth_width + 1u;
  invalid_content_extent.tensor_content_bottom = UINT32_MAX;
  cases[11] = {"invalid content extent", invalid_content_extent, false, true};

  std::string error;
  live_v2_warp_fixture_t main_warp;
  live_v2_warp_fixture_t mapping_warp;
  ASSERT_TRUE(main_warp.initialize(error, warp_shader_e::live_main)) << error;
  ASSERT_TRUE(mapping_warp.initialize(error, warp_shader_e::live_mapping)) << error;
  const auto state = make_live_state(true, true);
  const std::array<float, 4> sentinel_clear {0.91f, 0.13f, 0.77f, 0.42f};
  const auto render_case = [&] (
    live_v2_warp_fixture_t &warp,
    const host_sbs_v2_geometry_t &geometry
  ) {
    std::vector<std::byte> bytes;
    EXPECT_TRUE(warp.render(
      DXGI_FORMAT_B8G8R8A8_UNORM,
      sizeof(bgra8_t),
      source_width,
      source_height,
      source.data(),
      parallax,
      candidate,
      state,
      sentinel_clear,
      bytes,
      error,
      depth_width,
      depth_height,
      geometry
    )) << error;
    return bytes;
  };

  const auto full_main = render_case(main_warp, cases[0].geometry);
  const auto full_mapping = render_case(mapping_warp, cases[0].geometry);
  const auto roi_main = render_case(main_warp, valid_roi);
  const auto roi_mapping = render_case(mapping_warp, valid_roi);
  const auto identity_main = render_case(main_warp, invalid_mode);
  const auto identity_mapping = render_case(mapping_warp, invalid_mode);
  ASSERT_FALSE(full_main.empty());
  ASSERT_FALSE(full_mapping.empty());
  EXPECT_NE(full_main, identity_main);
  EXPECT_NE(full_mapping, identity_mapping);
  EXPECT_NE(roi_main, identity_main);
  EXPECT_NE(roi_mapping, identity_mapping);

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto actual_main = render_case(main_warp, test_case.geometry);
    const auto actual_mapping = render_case(mapping_warp, test_case.geometry);
    if (test_case.full_warp) {
      EXPECT_EQ(actual_main, full_main);
      EXPECT_EQ(actual_mapping, full_mapping);
    } else if (test_case.flat_identity) {
      EXPECT_EQ(actual_main, identity_main);
      EXPECT_EQ(actual_mapping, identity_mapping);
    } else {
      EXPECT_EQ(actual_main, roi_main);
      EXPECT_EQ(actual_mapping, roi_mapping);
    }
  }
}

TEST(HostSbsV2LiveWarpGpuTest, FullFrameBytecodeSkipsTheRoiDimensionQuery) {
  const auto shader_path = std::filesystem::path(SUNSHINE_SHADERS_DIR) /
    "sbs_reprojection_v2_live_ps.hlsl";
  ComPtr<ID3DBlob> bytecode;
  ComPtr<ID3DBlob> errors;
  auto status = D3DCompileFromFile(
    shader_path.c_str(),
    nullptr,
    D3D_COMPILE_STANDARD_FILE_INCLUDE,
    "main_ps",
    "ps_5_0",
    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
    0,
    &bytecode,
    &errors
  );
  ASSERT_TRUE(SUCCEEDED(status) && bytecode) << (errors ?
    static_cast<const char *>(errors->GetBufferPointer()) : "compile failed");
  ComPtr<ID3DBlob> disassembly;
  ASSERT_TRUE(SUCCEEDED(D3DDisassemble(
    bytecode->GetBufferPointer(), bytecode->GetBufferSize(), 0, nullptr, &disassembly
  )));
  const std::string assembly(
    static_cast<const char *>(disassembly->GetBufferPointer()),
    disassembly->GetBufferSize()
  );
  std::vector<std::string> lines;
  std::istringstream stream(assembly);
  for (std::string line; std::getline(stream, line);) {
    const auto first = line.find_first_not_of(" \t\r");
    const auto last = line.find_last_not_of(" \t\r");
    lines.push_back(first == std::string::npos ?
      std::string {} : line.substr(first, last - first + 1u));
  }
  std::size_t active_compare = std::string::npos;
  std::string active_zero_register;
  for (std::size_t index = 0u; index < lines.size(); ++index) {
    const auto &line = lines[index];
    if (line.starts_with("eq ") && line.find("cb2[0].z") != std::string::npos &&
        line.find("l(0.000000, 1.000000") != std::string::npos) {
      const auto destination_end = line.find(',');
      ASSERT_NE(destination_end, std::string::npos);
      const auto destination = line.substr(3u, destination_end - 3u);
      const auto component = destination.find('.');
      ASSERT_NE(component, std::string::npos);
      ASSERT_GT(destination.size(), component + 1u);
      active_zero_register = destination.substr(0u, component + 1u) +
        destination[component + 1u];
      active_compare = index;
      break;
    }
  }
  ASSERT_NE(active_compare, std::string::npos);

  std::size_t full_branch = std::string::npos;
  for (std::size_t index = active_compare + 1u; index < lines.size(); ++index) {
    if (lines[index] == "if_nz " + active_zero_register) {
      full_branch = index;
      break;
    }
  }
  ASSERT_NE(full_branch, std::string::npos);
  int branch_depth = 0;
  bool roi_else = false;
  std::size_t full_resinfo = 0u;
  std::size_t roi_resinfo = 0u;
  std::size_t source_resinfo = 0u;
  std::size_t full_source_resinfo = 0u;
  std::size_t base_divisions = 0u;
  std::size_t full_base_samples = 0u;
  std::size_t roi_base_samples = 0u;
  for (std::size_t index = full_branch; index < lines.size(); ++index) {
    const auto &line = lines[index];
    if (line.starts_with("if_") || line.starts_with("loop")) {
      ++branch_depth;
    } else if (line == "else" && branch_depth == 1) {
      roi_else = true;
    } else if (line == "endif" || line == "endloop") {
      --branch_depth;
      if (branch_depth == 0) break;
    }
    base_divisions += line.starts_with("div ") ? 1u : 0u;
    if (line.starts_with("sample_l") && line.find("t1.") != std::string::npos) {
      if (roi_else) {
        ++roi_base_samples;
      } else {
        ++full_base_samples;
      }
    }
    if (line.find("resinfo") != std::string::npos &&
        line.find("t1.") != std::string::npos) {
      if (roi_else) {
        ++roi_resinfo;
      } else {
        ++full_resinfo;
      }
    }
    if (line.find("resinfo") != std::string::npos &&
        line.find("t0.") != std::string::npos) {
      ++source_resinfo;
      full_source_resinfo += roi_else ? 0u : 1u;
    }
  }
  EXPECT_TRUE(roi_else);
  EXPECT_EQ(full_resinfo, 0u);
  EXPECT_EQ(roi_resinfo, 1u);
  EXPECT_EQ(full_source_resinfo, 0u);
  EXPECT_EQ(source_resinfo, 0u);
  // Inspect the ordinary Base inverse. Current-glyph authority and display/HDR
  // revocation belong to the separately tested shared subtitle renderer.
  EXPECT_LE(base_divisions, 8u);
  EXPECT_GE(full_base_samples, 1u);
  EXPECT_LE(full_base_samples, 11u);
  EXPECT_GE(roi_base_samples, 1u);
  EXPECT_LE(roi_base_samples, 11u);
}

TEST(HostSbsV2LiveWarpGpuTest, OrdinaryShaderBytecodeHasNoSubtitleBindings) {
  namespace fs = std::filesystem;
  for (const auto &[file, entry] : {
         std::pair {"sbs_reprojection_v2_live_ps.hlsl", "main_ps"},
         std::pair {"sbs_reprojection_v2_queued_ps.hlsl", "main_ps"},
         std::pair {"sbs_reprojection_v2_diagnostics_ps.hlsl", "mapping_ps"},
         std::pair {"sbs_reprojection_v2_diagnostics_ps.hlsl", "mask_ps"},
         std::pair {"sbs_reprojection_v2_p010_y_ps.hlsl", "main_p010_y_ps"},
       }) {
    SCOPED_TRACE(std::string(file) + ":" + entry);
    ComPtr<ID3DBlob> bytecode, errors;
    const auto status = D3DCompileFromFile((fs::path(SUNSHINE_SHADERS_DIR) / file).c_str(),
      nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, "ps_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0u, &bytecode, &errors);
    ASSERT_TRUE(SUCCEEDED(status) && bytecode) << (errors ?
      static_cast<const char *>(errors->GetBufferPointer()) : "ordinary shader compile failed");
    ComPtr<ID3D11ShaderReflection> reflection;
    ASSERT_TRUE(SUCCEEDED(D3DReflect(bytecode->GetBufferPointer(), bytecode->GetBufferSize(),
      IID_ID3D11ShaderReflection, reinterpret_cast<void **>(reflection.GetAddressOf()))));
    D3D11_SHADER_DESC desc {};
    ASSERT_TRUE(SUCCEEDED(reflection->GetDesc(&desc)));
    for (UINT i = 0u; i < desc.BoundResources; ++i) {
      D3D11_SHADER_INPUT_BIND_DESC resource {};
      ASSERT_TRUE(SUCCEEDED(reflection->GetResourceBindingDesc(i, &resource)));
      if (resource.Type == D3D_SIT_CBUFFER) {
        EXPECT_NE(resource.BindPoint, 3u) << resource.Name;
      } else if (resource.Type != D3D_SIT_SAMPLER) {
        // Queued transaction authentication at t10 is independent of subtitles.
        EXPECT_TRUE(resource.BindPoint < 3u || resource.BindPoint == 10u) << resource.Name;
      }
    }
  }
}

TEST(HostSbsV2LiveWarpGpuTest, IndependentFlatShaderIgnoresV2Geometry) {
  constexpr UINT width = 8u;
  constexpr UINT height = 1u;
  const std::array<float, 4> sentinel_clear {0.91f, 0.13f, 0.77f, 0.42f};
  std::string error;
  live_v2_warp_fixture_t flat;
  ASSERT_TRUE(flat.initialize(error, warp_shader_e::flat_main)) << error;

  std::vector<rgba32f_t> source(width);
  for (UINT x = 0; x < width; ++x) {
    source[x] = {
      (static_cast<float>(x) + 0.5f) / static_cast<float>(width),
      0.2f,
      0.7f,
      1.0f,
    };
  }
  std::vector<std::byte> packed_bytes;
  ASSERT_TRUE(flat.render(
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    sizeof(rgba32f_t),
    width,
    height,
    source.data(),
    std::vector<float>(width, 0.4f),
    std::vector<float>(width, -3.0f),
    make_live_state(true, true),
    sentinel_clear,
    packed_bytes,
    error
  )) << error;
  expect_flat_identity(source, unpack_rgba32f(packed_bytes), width);
}


TEST(HostSbsV2LiveWarpGpuTest, OrdinaryRendererUsesBaseWithValidSubtitleInputs) {
  constexpr UINT width = 128u, height = 4u, row = 2u, glyph_x = 64u;
  const auto state = make_live_state(true, true);
  const std::vector<float> base(width * height, -2.0f / width);
  std::vector<rgba32f_t> source(width * height, {0.0f, 0.0f, 0.0f, 1.0f});
  source[row * width + glyph_x] = {1.0f, 1.0f, 1.0f, 1.0f};
  std::string error;
  live_v2_warp_fixture_t color, mapping, flat;
  ASSERT_TRUE(color.initialize(error)) << error;
  ASSERT_TRUE(mapping.initialize(error, warp_shader_e::live_mapping)) << error;
  ASSERT_TRUE(flat.initialize(error, warp_shader_e::flat_main)) << error;
  const auto render = [&](live_v2_warp_fixture_t &fixture, const subtitle_inputs_t *subtitle) {
    std::vector<std::byte> output;
    EXPECT_TRUE(fixture.render(DXGI_FORMAT_R32G32B32A32_FLOAT, sizeof(rgba32f_t),
      width, height, source.data(), base, base, state, {}, output, error,
      0u, 0u, {}, subtitle)) << error;
    return output;
  };
  const auto base_color = render(color, nullptr), base_map = render(mapping, nullptr);
  EXPECT_NE(base_color, render(flat, nullptr)) << "Usable Base depth must survive without subtitle rendering";
  for (const float shift : {2.0f, 2.5f}) {
    auto subtitle = make_subtitle_inputs(width, height, state, shift / width);
    subtitle.matte[row * width + glyph_x] = 1u;
    subtitle.snapshot[row * width + glyph_x] = {0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u};
    EXPECT_EQ(render(color, &subtitle), base_color);
    EXPECT_EQ(render(mapping, &subtitle), base_map);
  }
}

TEST(HostSbsV2LiveWarpGpuTest, OrdinaryRendererIgnoresInvalidSubtitleInputs) {
  constexpr UINT width = 128u, height = 4u, row = 2u, glyph_x = 64u;
  const auto state = make_live_state(true, true);
  const std::array<float, 4> clear {};
  const std::vector<float> base(width * height, -2.0f / width);
  std::vector<rgba32f_t> source(width * height, {0.0f, 0.0f, 0.0f, 1.0f});
  source[row * width + glyph_x] = {1.0f, 1.0f, 1.0f, 1.0f};
  auto original = make_subtitle_inputs(width, height, state, 2.0f / width);
  original.matte[row * width + glyph_x] = 1u;
  original.snapshot[row * width + glyph_x] = {0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u};
  std::string error;
  live_v2_warp_fixture_t color, mapping;
  ASSERT_TRUE(color.initialize(error)) << error;
  ASSERT_TRUE(mapping.initialize(error, warp_shader_e::live_mapping)) << error;
  const auto render = [&](live_v2_warp_fixture_t &fixture, const subtitle_inputs_t *subtitle) {
    std::vector<std::byte> output;
    EXPECT_TRUE(fixture.render(DXGI_FORMAT_R32G32B32A32_FLOAT, sizeof(rgba32f_t),
      width, height, source.data(), base, base, state, clear, output, error,
      0u, 0u, {}, subtitle)) << error;
    return output;
  };
  for (UINT fault = 0u; fault < 12u; ++fault) {
    SCOPED_TRACE(fault);
    auto subtitle = original;
    source[row * width + glyph_x] = {1.0f, 1.0f, 1.0f, 1.0f};
    switch (fault) {
      case 0u: subtitle.constants.enabled = 0u; break;
      case 1u: subtitle.condition[4] = 0u; break;
      case 2u: subtitle.condition[2] = 0u; break;
      case 3u: subtitle.identity[3] ^= 1u; break;
      case 4u: subtitle.constants.analysis_generation_low ^= 1u; break;
      case 5u: subtitle.identity[20] ^= 1u; break;
      case 6u: subtitle.snapshot[row * width + glyph_x].a = 0u; break;
      case 7u: subtitle.identity[22] ^= 1u; break;
      case 8u: subtitle.identity[24] = 1u; break;
      case 9u: subtitle.condition[5] = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN()); break;
      case 10u: subtitle.matte[row * width + glyph_x] = 0u; break;
      case 11u: source[row * width + glyph_x] = {0.2f, 0.0f, 0.0f, 1.0f}; break;
    }
    EXPECT_EQ(render(color, &subtitle), render(color, nullptr));
    EXPECT_EQ(render(mapping, &subtitle), render(mapping, nullptr));
  }
}

TEST(HostSbsV2LiveWarpGpuTest, OrdinaryRoiRendererIgnoresSubtitleCropAndPlane) {
  constexpr UINT width = 128u, height = 16u, field_width = 16u, field_height = 4u;
  constexpr UINT glyph_x = 64u, glyph_y = 10u;
  const auto state = make_live_state(true, true);
  const std::vector<float> base(field_width * field_height, 0.025f);
  std::vector<rgba32f_t> source(width * height, {0.0f, 0.0f, 0.0f, 1.0f});
  source[glyph_y * width + glyph_x] = {1.0f, 1.0f, 1.0f, 1.0f};
  host_sbs_v2_geometry_t geometry {
    .video_roi_active = 1.0f,
    .video_roi_left = 0.25f, .video_roi_top = 0.25f,
    .video_roi_right = 0.75f, .video_roi_bottom = 0.75f,
    .tensor_content_right = field_width, .tensor_content_bottom = field_height,
  };
  auto subtitle = make_subtitle_inputs(64u, 8u, state, 2.0f / 64u, field_width, field_height);
  subtitle.constants.source_left = 32u;
  subtitle.constants.source_top = 4u;
  subtitle.height = 4u;
  subtitle.identity[10u] = 4u;
  subtitle.identity[11u] = 4u;
  subtitle.matte.resize(64u * 4u);
  subtitle.snapshot.resize(64u * 4u);
  const auto native_index = (glyph_y - 8u) * 64u + (glyph_x - 32u);
  subtitle.matte[native_index] = 1u;
  subtitle.snapshot[native_index] = {0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u};
  std::string error;
  live_v2_warp_fixture_t color, mapping;
  ASSERT_TRUE(color.initialize(error)) << error;
  ASSERT_TRUE(mapping.initialize(error, warp_shader_e::live_mapping)) << error;
  const auto render = [&](live_v2_warp_fixture_t &fixture, const subtitle_inputs_t *input) {
    std::vector<std::byte> bytes;
    EXPECT_TRUE(fixture.render(DXGI_FORMAT_R32G32B32A32_FLOAT, sizeof(rgba32f_t),
      width, height, source.data(), base, base, state, {}, bytes, error,
      field_width, field_height, geometry, input)) << error;
    return bytes;
  };
  EXPECT_EQ(render(color, &subtitle), render(color, nullptr));
  EXPECT_EQ(render(mapping, &subtitle), render(mapping, nullptr));
}

TEST(HostSbsV2LiveWarpGpuTest, OrdinaryRendererIgnoresSubtitleSnapshotsInEveryColourMode) {
  constexpr UINT width = 128u, height = 2u, glyph_x = 64u, row = 1u;
  const auto state = make_live_state(true, true);
  const std::vector<float> base(width * height, -2.0f / width);
  std::string error;
  live_v2_warp_fixture_t warp;
  ASSERT_TRUE(warp.initialize(error)) << error;
  for (UINT mode = 0u; mode < 3u; ++mode) {
    SCOPED_TRACE(mode);
    auto subtitle = make_subtitle_inputs(width, height, state, 2.0f / width);
    subtitle.constants.color_mode = mode;
    // For encoded SDR, native .5 stays .5. Linear .5 and scRGB1 both map to
    // display-sRGB OETF(.5), whose stored FP16 is 0x39e2.
    const std::uint16_t native = mode == 2u ? 0x3c00u : 0x3800u;
    const std::uint16_t display = mode == 0u ? 0x3800u : 0x39e2u;
    std::vector<rgba16_bits_t> source(width * height, {0u, 0u, 0u, 0x3c00u});
    source[row * width + glyph_x] = {native, native, native, 0x3c00u};
    subtitle.matte[row * width + glyph_x] = 1u;
    subtitle.snapshot[row * width + glyph_x] = {display, display, display, 0x3c00u};
    std::vector<std::byte> bytes;
    ASSERT_TRUE(warp.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t),
      width, height, source.data(), base, base, state, {}, bytes, error,
      0u, 0u, {}, &subtitle)) << error;
    std::vector<std::byte> baseline;
    ASSERT_TRUE(warp.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t),
      width, height, source.data(), base, base, state, {}, baseline, error)) << error;
    EXPECT_EQ(bytes, baseline);
  }
}

TEST(HostSbsV2LiveWarpGpuTest, SavedCurrentHdrCompletedDrawReplay) {
  const auto *input_env = std::getenv("SUNSHINE_TEST_SAVED_CURRENT_OUTLINE_CASES");
  if (!input_env || !*input_env) GTEST_SKIP() << "No saved-current HDR specification selected";
  const auto *output_env = std::getenv("SUNSHINE_TEST_SAVED_CURRENT_DRAW_OUTPUT");
  ASSERT_TRUE(output_env && *output_env);
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;
  using json = nlohmann::json;
  const auto require = [](bool valid, const std::string &message) {
    if (!valid) throw std::runtime_error(message);
  };
  const auto read = [&](const fs::path &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(bool(stream) && stream.tellg() >= 0, "Cannot read saved-current draw operand");
    std::vector<char> bytes(static_cast<std::size_t>(stream.tellg()));stream.seekg(0);
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));require(bool(stream), "Incomplete saved-current draw operand");
    return bytes;
  };
  const auto sha256 = [](const auto &bytes) {
    const auto hash = crypto::hash(std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
    constexpr char digits[] = "0123456789abcdef";std::string result;
    for (auto byte : hash) {result.push_back(digits[byte >> 4u]);result.push_back(digits[byte & 15u]);}
    return result;
  };
  try {
    for (const auto *retired : {"SUNSHINE_TEST_SAVED_CURRENT_BG_SHADER_ROOT",
                               "SUNSHINE_TEST_SAVED_CURRENT_BG_SHADER_PIN",
                               "SUNSHINE_TEST_SHARED_UI_PIN_PERF_SAMPLES"}) {
      require(!std::getenv(retired),
        "Retired literal/repair pixel-shader modes are unavailable; use the shared warp replay");
    }
    const fs::path input(input_env), output(output_env);
    const auto specification_bytes = read(input);
    const auto spec = json::parse(specification_bytes.begin(), specification_bytes.end());
    require(spec.at("schema") == 1u && (spec.at("driver") == "hardware" || spec.at("driver") == "warp"), "Saved-current draw schema/driver");
    require(!fs::exists(output) || (fs::is_directory(output) && fs::is_empty(output)), "Saved-current draw output must be new or empty");
    const auto &cases = spec.at("cases");require(cases.is_array() && !cases.empty() && cases.size() <= 8u, "Saved-current draw case count");
    // Contract identity includes the current renderer source inventory. Historical
    // captured states stay immutable and cannot be retagged to authorize this build.
    // Reject unsupported captures before creating a device or submitting GPU work.
    for (const auto &entry : cases) {
      const auto &binding = entry.at("artifacts").at("state");
      const auto bytes = read(binding.at("path").get<std::string>());
      require(bytes.size() == sizeof(v2::state_words_t) && binding.at("byte_count") == bytes.size() &&
        binding.at("sha256") == sha256(bytes), "Saved-current preflight state binding");
      v2::state_words_t state {};std::memcpy(state.data(), bytes.data(), bytes.size());
      const auto &typed = entry.at("provenance").at("typed_state_json");
      const auto typed_bytes = read(typed.at("path").get<std::string>());
      require(typed.at("byte_count") == typed_bytes.size() && typed.at("sha256") == sha256(typed_bytes),
        "Saved-current preflight typed-state binding");
      const auto document = json::parse(typed_bytes.begin(), typed_bytes.end());
      const float scale = document.at("constants").at("raw_coordinate_scale").get<float>();
      require(v2::parallax_state_words_are_authenticated(state, scale, v2::adaptive_policy_id) &&
        state[v2::renderer_authorization_bits] == v2::contract_tag,
        "Saved-current captured renderer contract is retired; immutable captures cannot authorize the current shared warp");
    }
    fs::create_directories(output);
    const bool hardware = spec.at("driver") == "hardware";
    live_v2_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, warp_shader_e::selected_outline, false, hardware)) << error;
    std::array<wchar_t, 32768u> exe_path {};
    const auto exe_size = GetModuleFileNameW(nullptr, exe_path.data(), DWORD(exe_path.size()));
    require(exe_size > 0u && exe_size < exe_path.size(), "Cannot locate current draw replay executable");
    const fs::path executable(std::wstring(exe_path.data(), exe_size));
    const auto executable_bytes = read(executable);
    const auto fixture_path = fs::path(SUNSHINE_SOURCE_DIR) / "tests/unit/test_host_sbs_v2_live_warp.cpp";
    const auto fixture_bytes = read(fixture_path);
    json report {{"schema", 1u}, {"driver", spec.at("driver")}, {"adapter", fixture.adapter_identity()},
      {"input_specification", {{"path", input.string()}, {"sha256", sha256(specification_bytes)}}},
      {"test_executable", {{"path", executable.string()}, {"sha256", sha256(executable_bytes)}}},
      {"fixture_source", {{"path", fixture_path.string()}, {"sha256", sha256(fixture_bytes)}}},
      {"shared_shader_closure", models::host_sbs_shared_ui_warp::closure},
      {"scope", "Immutable saved native-HDR source/producer inputs rendered by the current shared warp; captured renderer identity remains provenance only"},
      {"pixel_route_map", "not exported; colour crops and whole-output hashes only"},
      {"quality_claim", false}, {"performance_claim", false}, {"cases", json::array()}};
    const auto read_binding = [&](const json &binding) {
      const auto bytes = read(binding.at("path").get<std::string>());
      require(binding.at("byte_count") == bytes.size() && binding.at("sha256") == sha256(bytes), "Candidate draw binding drift");
      return bytes;
    };
    const auto *candidate_env = std::getenv("SUNSHINE_TEST_SAVED_CURRENT_DRAW_CANDIDATE_CASES");
    const bool has_candidate = candidate_env && *candidate_env;
    std::vector<char> candidate_specification_bytes, candidate_producer_bytes;
    json candidate_specification, candidate_producer;
    json fixture_f2_binding;
    if (has_candidate) {
      candidate_specification_bytes = read(candidate_env);
      candidate_specification = json::parse(candidate_specification_bytes.begin(), candidate_specification_bytes.end());
      require(candidate_specification.at("schema") == 1u && candidate_specification.at("cases").is_array() &&
        candidate_specification.at("cases").size() == cases.size(), "Candidate draw specification schema/cases");
      candidate_producer_bytes = read_binding(candidate_specification.at("producer_report"));
      candidate_producer = json::parse(candidate_producer_bytes.begin(), candidate_producer_bytes.end());
      require(candidate_producer.at("schema") == 1u && candidate_producer.at("driver") == spec.at("driver"), "Candidate producer schema/driver");
      require(candidate_producer.at("test_executable").at("sha256") == sha256(executable_bytes), "Candidate producer and draw must use the same test executable");
      namespace runtime = models::current_outline_runtime;
      const bool compiled_hole_ready = candidate_producer.at("hole_policy").at("production_program_ready").get<bool>();
      const bool compiled_coverage_ready = candidate_producer.at("coverage_policy").at("production_program_ready").get<bool>();
      require(candidate_producer.at("coverage_policy").at("compiled_source_closure") == runtime::coverage_closure,
        "Candidate coverage source inventory mismatch");
      require(candidate_producer.at("hole_policy").at("compiled_source_closure") == runtime::hole_closure &&
        candidate_producer.at("hole_policy").at("compiled_helper_pin") == subtitle_test_source_pins::current_mask_holes,
        "Candidate hole source/helper inventory mismatch");
      const std::array<std::pair<const char *, std::string_view>, 11u> compiled_closures {{
        {"source5", runtime::source5_closure}, {"colour", runtime::colour_closure}, {"components", runtime::components_closure},
        {"enclosure", runtime::enclosure_closure}, {"roi", runtime::roi_closure}, {"outline", runtime::outline_closure},
        {"inside", runtime::inside_closure}, {"selector", runtime::selector_closure}, {"stability", runtime::stability_closure},
        {"hole", compiled_hole_ready ? runtime::hole_closure : std::string_view()},
        {"coverage", compiled_coverage_ready ? runtime::coverage_closure : std::string_view()}}};
      const bool fixture_f2 = candidate_producer.value("prototype_not_production_qualified", false);
      json expected_compiled = json::object();for (const auto &[name, closure] : compiled_closures) expected_compiled[name] = closure;
      if (fixture_f2) {
        const auto *opt_in = std::getenv("SUNSHINE_TEST_SAVED_CURRENT_OUTLINE_FIXTURE_F2");
        require(opt_in && std::string_view(opt_in) == "1", "Prototype F2 draw requires explicit fixture opt-in");
        require(candidate_producer.at("compiled_production_closures") == expected_compiled &&
          candidate_producer.at("effective_composition_closures") == candidate_producer.at("closures") &&
          candidate_producer.at("closures").size() == compiled_closures.size(), "Prototype F2 compiled/effective inventory mismatch");
        const auto &override = candidate_producer.at("fixture_f2_override");
        require(override.at("compiled_outline_helper_pin") == subtitle_test_source_pins::current_outline_support, "Prototype F2 helper ABI mismatch");
        fixture_f2_binding = json::object();
        const auto check_group = [&](const char *name, const char *root_env, const char *pin_env, const auto &specs) {
          const auto *root = std::getenv(root_env), *pin = std::getenv(pin_env);
          require(root && *root && pin && *pin, "Prototype F2 requires actual external root and pin");
          const auto sources = models::host_sbs_shader_cache::snapshot_sources(root, specs);
          require(sources && models::host_sbs_shader_cache::source_closure_sha256(sources) == pin &&
            candidate_producer.at("closures").at(name) == pin && override.at(std::string(name) + "_closure") == pin &&
            fs::weakly_canonical(root) == fs::weakly_canonical(override.at(std::string(name) + "_root").get<std::string>()), "Prototype F2 source/root/closure mismatch");
          fixture_f2_binding[name] = {{"root", root}, {"source_closure_sha256", pin}};
        };
        check_group("outline", "SUNSHINE_TEST_OUTLINE_CLOSURE_SHADER_ROOT", "SUNSHINE_TEST_OUTLINE_CLOSURE_SHADER_PIN", runtime::outline_specs);
        check_group("stability", "SUNSHINE_TEST_BLACK_STABILITY_SHADER_ROOT", "SUNSHINE_TEST_BLACK_STABILITY_SHADER_PIN", runtime::stability_specs);
        if (!candidate_producer.at("closures").at("hole").get<std::string>().empty())
          check_group("hole", "SUNSHINE_TEST_CURRENT_HOLE_SHADER_ROOT", "SUNSHINE_TEST_CURRENT_HOLE_SHADER_PIN", runtime::hole_specs);
        if (!candidate_producer.at("closures").at("coverage").get<std::string>().empty())
          check_group("coverage", "SUNSHINE_TEST_STROKE_COVERAGE_SHADER_ROOT", "SUNSHINE_TEST_STROKE_COVERAGE_SHADER_PIN", runtime::coverage_specs);
        report["prototype_not_production_qualified"] = true;
        report["effective_producer_closures"] = candidate_producer.at("closures");
        report["compiled_production_closures"] = expected_compiled;
        report["fixture_f2_source_bindings"] = fixture_f2_binding;
      }
      for (const auto &[name, expected] : compiled_closures) {
        if (fixture_f2 && (std::string_view(name) == "outline" || std::string_view(name) == "stability" ||
                           std::string_view(name) == "hole" || std::string_view(name) == "coverage")) continue;
        require(candidate_producer.at("closures").at(name) == expected, "Candidate producer runtime closure mismatch");
      }
      report["candidate_input_specification"] = {{"path", candidate_env}, {"sha256", sha256(candidate_specification_bytes)}};
      report["candidate_producer_report"] = candidate_specification.at("producer_report");
    }
    const auto *shared_pin_root=std::getenv("SUNSHINE_TEST_SHARED_UI_PIN_SHADER_ROOT");
    const auto *shared_pin_pin=std::getenv("SUNSHINE_TEST_SHARED_UI_PIN_SHADER_PIN");
    const bool has_shared_pin=shared_pin_root && *shared_pin_root;
    require(has_shared_pin==bool(shared_pin_pin && *shared_pin_pin),"Shared UI pin requires root and exact closure pin");
    if (has_shared_pin) {
      ASSERT_TRUE(fixture.configure_shared_ui_pin(shared_pin_root,shared_pin_pin,error)) << error;
      report["shared_ui_pin_experiment"]={{"root",shared_pin_root},{"source_closure_sha256",shared_pin_pin},
        {"not_production_qualified",true},{"mask_policy","Same selected M, receipt and pointwise current-RGB gate as matched current compositor"},
        {"field_policy","Native whole-source-width subtitle band; shared Game 3D pin CS; exact original Base inverse on rows without positive bilinear active-row contribution"},
        {"resample_control","Same current M and initialized native h, with shared pin dispatch omitted"}};
    }
    for (std::size_t index = 0u; index < cases.size(); ++index) {
      const auto &entry = cases.at(index);SCOPED_TRACE(entry.at("name").get<std::string>());
      const auto source_shape = entry.at("source").get<std::array<UINT, 2u>>();
      const auto field = entry.at("field").get<std::array<UINT, 2u>>();
      const auto shape = entry.at("analysis_shape").get<std::array<UINT, 4u>>();
      const UINT width = source_shape[0], height = source_shape[1];
      require(width > 0u && width <= 8192u && height > 0u && height <= 16384u &&
        field[0] > 0u && field[0] <= 5120u && field[1] > 0u && field[1] <= 5120u &&
        shape[0] > 0u && shape[0] <= 5120u && shape[1] > shape[2] && shape[1] <= 5120u &&
        shape[3] == shape[1] - shape[2], "Saved-current draw extents");
      const std::size_t native_pixels = std::size_t(shape[0]) * shape[3];
      std::map<std::string, std::vector<char>> imported;
      json bindings = json::object();
      const auto operand = [&](const char *key, std::size_t size) {
        const auto &binding = entry.at("artifacts").at(key);
        const auto bytes = read(binding.at("path").get<std::string>());
        require(bytes.size() == size && binding.at("byte_count") == size && binding.at("sha256") == sha256(bytes), std::string("Saved-current draw binding: ") + key);
        imported.emplace(key, bytes);bindings[key] = binding;return bytes;
      };
      const auto object = [&]<typename T>(const char *key, T &target) {
        const auto bytes = operand(key, sizeof(target));std::memcpy(&target, bytes.data(), bytes.size());
      };
      const auto vector = [&]<typename T>(const char *key, std::size_t count) {
        const auto bytes = operand(key, count * sizeof(T));std::vector<T> values(count);
        std::memcpy(values.data(), bytes.data(), bytes.size());return values;
      };
      const auto source = vector.template operator()<rgba16_bits_t>("source_rgba16f", std::size_t(width) * height);
      const auto base = vector.template operator()<float>("base", std::size_t(field[0]) * field[1]);
      models::depth_coordinate_v2::state_words_t state {};
      host_sbs_v2_geometry_t geometry;
      subtitle_inputs_t subtitle;
      current_outline_inputs_t outline;
      saved_background_inputs_t background;
      object("state", state);object("geometry", geometry);object("subtitle_constants", subtitle.constants);
      object("condition", subtitle.condition);object("current_sct", subtitle.identity);object("selected_receipt", outline.receipt);
      object("background_proof", background.proof);object("background_expected", background.expected);
      std::array<std::uint32_t, 24u> contour_constants {};
      object("constants", contour_constants);
      subtitle.width = shape[0];subtitle.height = shape[3];
      subtitle.matte = vector.template operator()<std::uint32_t>("legacy_matte", native_pixels);
      subtitle.snapshot = vector.template operator()<rgba16_bits_t>("display_snapshot", native_pixels);
      outline.matte = vector.template operator()<std::uint32_t>("expected_selected_matte", native_pixels);
      background.pixels = vector.template operator()<rgba32f_t>("background", native_pixels);
      const auto bound_provenance = [&](const char *key) {
        const auto &binding = entry.at("provenance").at(key);
        const auto bytes = read(binding.at("path").get<std::string>());
        require(binding.at("byte_count") == bytes.size() && binding.at("sha256") == sha256(bytes), "Saved-current draw provenance binding");
        imported.emplace(key, bytes);bindings[key] = binding;
        return json::parse(bytes.begin(), bytes.end());
      };
      const auto state_document = bound_provenance("typed_state_json");
      const auto actual_draw = bound_provenance("actual_draw_evidence");
      const float raw_scale = state_document.at("constants").at("raw_coordinate_scale").get<float>();
      // Historical capture provenance only. This closure is never loaded or used
      // as the current renderer identity; the report pins the shared warp above.
      constexpr std::string_view captured_literal_shader_closure =
        "c8053c9ed5b1f3969e2852f63df3f811671e88f8b1ec8e9dae3b7fc616a3ebe6";
      require(actual_draw.at("pixel_shader_closure") == captured_literal_shader_closure &&
        actual_draw.at("program") == "current-outline-selected", "Saved-current captured renderer provenance");
      require(subtitle.constants.color_mode == 2u && subtitle.constants.enabled == 1u &&
        subtitle.constants.source_width == shape[0] && subtitle.constants.source_height == shape[1] &&
        subtitle.identity[10u] == shape[2] && subtitle.identity[11u] == shape[3] &&
        subtitle.constants.source_left <= width && shape[0] <= width - subtitle.constants.source_left &&
        subtitle.constants.source_top <= height && shape[1] <= height - subtitle.constants.source_top, "Saved-current HDR draw geometry");
      require(v2::parallax_state_words_are_authenticated(state, raw_scale, v2::adaptive_policy_id) &&
        state[v2::renderer_authorization_bits] == v2::contract_tag, "Saved-current draw state authorization");
      require(std::all_of(base.begin(), base.end(), [](float x) {return std::isfinite(x) && std::abs(x) <= v2::direct_container_limit;}), "Saved-current draw Base values");
      for (const auto *mask : {&subtitle.matte, &outline.matte}) require(std::all_of(mask->begin(), mask->end(), [](auto x) {return x <= 1u;}), "Saved-current draw binary mask");
      require(outline.receipt[0u] == 0x31534d53u && outline.receipt[1u] == 1u && outline.receipt[2u] == 3u &&
        outline.receipt[3u] == 0u && outline.receipt[62u] == 2u && outline.receipt[63u] == 0u &&
        std::equal(contour_constants.begin(), contour_constants.end(), outline.receipt.begin() + 4u) &&
        std::equal(subtitle.identity.begin(), subtitle.identity.end(), outline.receipt.begin() + 28u), "Saved-current draw selected owner");
      current_outline_inputs_t candidate_outline;
      json candidate_binding;
      if (has_candidate) {
        const auto &mapping = candidate_specification.at("cases").at(index);
        require(mapping.at("name") == entry.at("name"), "Candidate draw case order/name mismatch");
        const auto producer_index = mapping.at("producer_case_index").get<std::size_t>();
        require(producer_index < candidate_producer.at("cases").size(), "Candidate producer case index");
        const auto &produced = candidate_producer.at("cases").at(producer_index);
        require(produced.at("name") == entry.at("name") && produced.at("current_outline_authorized") == true &&
          produced.at("model_frame") == entry.at("model_frame") && produced.at("observation_us") == entry.at("observation_us") &&
          produced.at("analysis_generation") == entry.at("analysis_generation") && produced.at("token") == entry.at("token"), "Candidate producer current identity");
        // Bind every model input as well as the output. These bytes never become substitute PS
        // inputs; they establish that the candidate mask came from this exact saved current case.
        constexpr std::array<const char *, 11u> producer_input_names {"source_rgba16f", "display_snapshot", "probability", "support",
          "ocr_record", "current_uip", "current_sct", "constants", "root_arguments", "legacy_matte", "expected_selected_matte"};
        require(produced.at("input_bindings").is_object() && produced.at("input_bindings").size() == producer_input_names.size(), "Candidate producer input coverage");
        for (const auto *name : producer_input_names) {
          const auto &binding = produced.at("input_bindings").at(name);
          const auto &original = entry.at("artifacts").at(name);
          require(binding.at("sha256") == original.at("sha256") && binding.at("byte_count") == original.at("byte_count"), "Candidate producer input differs from frozen tuple");
          const auto bytes = read_binding(binding);
          const auto key = std::string("candidate_producer_input:") + name;imported.emplace(key, bytes);bindings[key] = binding;
        }
        for (const auto &[key, artifact] : std::array<std::pair<const char *, const char *>, 2u> {{
          {"selected_matte", "selected_M.u32"}, {"selected_receipt", "selected_receipt.u32"}}}) {
          const auto &binding = mapping.at(key), &actual = produced.at("artifacts").at(artifact);
          require(binding.at("sha256") == actual.at("sha256") && binding.at("byte_count") == actual.at("byte_count"), "Candidate mask/receipt not joined to producer output");
          const auto bytes = read_binding(binding);
          const auto input_key = std::string("candidate_") + key;imported.emplace(input_key, bytes);bindings[input_key] = binding;
          if (std::string_view(key) == "selected_matte") {
            require(bytes.size() == native_pixels * 4u, "Candidate native matte size");
            candidate_outline.matte.resize(native_pixels);std::memcpy(candidate_outline.matte.data(), bytes.data(), bytes.size());
          } else {
            require(bytes.size() == sizeof(candidate_outline.receipt), "Candidate selected receipt size");
            std::memcpy(candidate_outline.receipt.data(), bytes.data(), bytes.size());
          }
        }
        require(std::all_of(candidate_outline.matte.begin(), candidate_outline.matte.end(), [](auto x) {return x <= 1u;}), "Candidate matte is not literal binary");
        const auto candidate_count = std::count(candidate_outline.matte.begin(), candidate_outline.matte.end(), 1u);
        require(produced.at("actual_selected_pixels") == candidate_count, "Candidate producer matte count mismatch");
        require(candidate_outline.receipt[0u] == 0x31534d53u && candidate_outline.receipt[1u] == 1u &&
          candidate_outline.receipt[2u] == 3u && candidate_outline.receipt[3u] == 0u &&
          candidate_outline.receipt[62u] == 2u && candidate_outline.receipt[63u] == 0u &&
          candidate_outline.receipt[60u] == outline.receipt[60u] && candidate_outline.receipt[61u] == outline.receipt[61u] &&
          std::equal(contour_constants.begin(), contour_constants.end(), candidate_outline.receipt.begin() + 4u) &&
          std::equal(subtitle.identity.begin(), subtitle.identity.end(), candidate_outline.receipt.begin() + 28u), "Candidate selected current receipt");
        candidate_binding = {{"producer_report", candidate_specification.at("producer_report")}, {"producer_case_index", producer_index},
          {"selected_matte", mapping.at("selected_matte")}, {"selected_receipt", mapping.at("selected_receipt")},
          {"render_policy", "Current shared warp consumes the actual candidate M and unchanged captured source/receipt"}};
      }
      const auto expected = outline.matte;
      UINT left = width, top = height, right = 0u, bottom = 0u;
      for (UINT y = 0u; y < shape[3]; ++y) for (UINT x = 0u; x < shape[0]; ++x) if (expected[std::size_t(y) * shape[0] + x]) {
        const UINT fx = subtitle.constants.source_left + x, fy = subtitle.constants.source_top + shape[2] + y;
        left = std::min(left, fx);top = std::min(top, fy);right = std::max(right, fx + 1u);bottom = std::max(bottom, fy + 1u);
      }
      require(left < right && top < bottom && right <= width && bottom <= height, "Saved-current caption crop");
      left = left > 128u ? left - 128u : 0u;top = top > 32u ? top - 32u : 0u;
      right = std::min(width, right + 128u);bottom = std::min(height, bottom + 32u);
      const auto case_output = output / ("case_" + std::to_string(index));fs::create_directory(case_output);
      json result {{"name", entry.at("name")}, {"crop_ltrb", {left, top, right, bottom}}, {"crop_layout", "eye-major RGBA16F tightly packed"},
        {"input_bindings", bindings}, {"routes", json::object()}};
      if (has_candidate) result["candidate"] = candidate_binding;
      std::vector<std::string> routes {"shared_ui_pin", "base", "ui_reference"};
      if (has_candidate) routes.emplace_back("candidate_shared_ui_pin");
      if (has_shared_pin) routes.emplace_back("shared_ui_pin_resample_only");
      for (const auto &route : routes) {
        auto render_subtitle = subtitle;auto render_outline = outline;auto render_geometry = geometry;
        auto render_base = base;
        if (route == "candidate_shared_ui_pin") render_outline = candidate_outline;
        shared_ui_pin_diagnostics_t pin_diagnostic;
        const bool pin_route=route=="shared_ui_pin" || route=="candidate_shared_ui_pin" || route=="shared_ui_pin_resample_only";
        render_outline.production_shared_ui_pin=true;
        if (pin_route) {
          render_outline.production_shared_ui_pin=!has_shared_pin;render_outline.shared_ui_pin=has_shared_pin;
          render_outline.shared_ui_pin_resample_only=route=="shared_ui_pin_resample_only";
          render_outline.shared_ui_pin_diagnostic_output=&pin_diagnostic;
        }
        double gpu_ms=-1.0;render_outline.render_gpu_ms=&gpu_ms;
        if (route == "base" || route == "ui_reference") render_subtitle.constants.enabled = 0u;
        if (route == "ui_reference") {
          const float target = std::bit_cast<float>(subtitle.condition[5u]) * (float(shape[0]) / float(width));
          require(std::isfinite(target) && target >= 0.0f && target <= v2::direct_container_limit, "Saved-current UI reference target");
          std::fill(render_base.begin(), render_base.end(), target);render_geometry.video_roi_active = 0.0f;
        }
        std::vector<std::byte> pixels;
        ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t), width, height,
          source.data(), render_base, render_base, state, {}, pixels, error, field[0], field[1], render_geometry,
          &render_subtitle, nullptr, true, &render_outline)) << route << ": " << error;
        ASSERT_EQ(pixels.size(), std::size_t(width) * height * 2u * sizeof(rgba16_bits_t));
        const std::size_t row_bytes = std::size_t(right - left) * sizeof(rgba16_bits_t);
        std::vector<std::byte> crop(row_bytes * (bottom - top) * 2u);
        for (UINT eye = 0u; eye < 2u; ++eye) for (UINT y = top; y < bottom; ++y) {
          const auto src = (std::size_t(y) * width * 2u + eye * width + left) * sizeof(rgba16_bits_t);
          const auto dst = (std::size_t(eye) * (bottom - top) + y - top) * row_bytes;
          std::memcpy(crop.data() + dst, pixels.data() + src, row_bytes);
        }
        const auto path = case_output / (route + ".rgba16f");
        std::ofstream stream(path, std::ios::binary);stream.write(reinterpret_cast<const char *>(crop.data()), static_cast<std::streamsize>(crop.size()));
        require(bool(stream), "Cannot write saved-current draw crop");
        result["routes"][route] = {{"whole_output_sha256", sha256(pixels)}, {"crop", {{"path", path.string()}, {"byte_count", crop.size()}, {"sha256", sha256(crop)}}}};
        if (has_shared_pin) result["routes"][route]["prepare_and_draw_gpu_ms"]=gpu_ms;
        if (pin_route) {
          const auto artifact=[&](const std::string &suffix,const void *data,std::size_t size) {
            const auto destination=case_output/(route+suffix);std::vector<char> bytes(size);std::memcpy(bytes.data(),data,size);
            std::ofstream stream(destination,std::ios::binary);stream.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));stream.close();
            require(bool(stream),"Cannot write shared UI pin diagnostic");
            return json{{"path",destination.string()},{"byte_count",bytes.size()},{"sha256",sha256(bytes)}};
          };
          const auto crop_float4=[&](const std::vector<std::byte> &whole) {
            require(whole.size()==std::size_t(width)*height*2u*16u,"Shared UI pin GPU diagnostic shape");
            const auto row=std::size_t(right-left)*16u;std::vector<std::byte> cropped(row*(bottom-top)*2u);
            for(UINT eye=0u;eye<2u;++eye)for(UINT y=top;y<bottom;++y)std::memcpy(cropped.data()+(std::size_t(eye)*(bottom-top)+y-top)*row,
              whole.data()+(std::size_t(y)*width*2u+eye*width+left)*16u,row);
            return cropped;
          };
          const auto mapping=crop_float4(pin_diagnostic.mapping_rgba32f),coverage=crop_float4(pin_diagnostic.ui_coverage_rgba32f);
          result["routes"][route]["shared_ui_pin"]={{"shape",pin_diagnostic.shape},
            {"field",artifact(".native-field.f32",pin_diagnostic.field.data(),pin_diagnostic.field.size()*4u)},
            {"authorized_weights",artifact(".authorized-weights.f32",pin_diagnostic.authorized_weights.data(),pin_diagnostic.authorized_weights.size()*4u)},
            {"active_rows",artifact(".active-rows.u32",pin_diagnostic.active_rows.data(),pin_diagnostic.active_rows.size()*4u)},
            {"selected_receipt_authorized",pin_diagnostic.selected_receipt_authorized},
            {"row_header_schema",1u},{"row_header",artifact(".row-header.u32",pin_diagnostic.selected_receipt_header.data(),sizeof(pin_diagnostic.selected_receipt_header))},
            {"mapping_crop",artifact(".mapping-crop.rgba32f",mapping.data(),mapping.size())},
            {"ui_coverage_crop",artifact(".ui-coverage-crop.rgba32f",coverage.data(),coverage.size())},
            {"mapping_layout","eye-major cropped RGBA32F: unsaturated sourceU,V, active-row route, valid"},
            {"coverage_layout","eye-major cropped RGBA32F: actual GPU current selected-M bilinear UI coverage, full-coverage flag,0,1"},
            {"native_plane_bytes",std::uint64_t(pin_diagnostic.field.size())*8u+(std::uint64_t(pin_diagnostic.active_rows.size())+1u)*4u}};
        }
      }
      if (entry.contains("provenance")) result["provenance"] = entry.at("provenance");
      for (const auto &[key, bytes] : imported) require(read(bindings.at(key).at("path").get<std::string>()) == bytes, "Saved-current draw input changed during replay");
      report["cases"].push_back(std::move(result));
    }
    require(read(input) == specification_bytes && read(executable) == executable_bytes && read(fixture_path) == fixture_bytes, "Saved-current draw implementation/specification changed");
    if (has_candidate) {
      require(read(candidate_env) == candidate_specification_bytes &&
        read_binding(candidate_specification.at("producer_report")) == candidate_producer_bytes, "Candidate draw specification/producer report changed");
    }
    if (has_shared_pin) require(fixture.shared_ui_pin_source_unchanged(),"Shared UI pin source changed during replay");
    if (!fixture_f2_binding.is_null()) {
      const auto unchanged = [&](const char *name, const auto &specs) {
        const auto &binding = fixture_f2_binding.at(name);
        const auto sources = models::host_sbs_shader_cache::snapshot_sources(binding.at("root").get<std::string>(), specs);
        require(sources && models::host_sbs_shader_cache::source_closure_sha256(sources) == binding.at("source_closure_sha256").get<std::string>(), "Prototype F2 source changed during draw replay");
      };
      unchanged("outline", models::current_outline_runtime::outline_specs);
      unchanged("stability", models::current_outline_runtime::stability_specs);
      if (fixture_f2_binding.contains("hole")) unchanged("hole", models::current_outline_runtime::hole_specs);
    }
    std::ofstream stream(output / "report.json", std::ios::binary);stream << report.dump(2);require(bool(stream), "Cannot write saved-current draw report");
  } catch (const std::exception &failure) {
    FAIL() << failure.what();
  }
}

// Opt-in schema-2 shared replay. All producer/source/receipt operands are literal saved bytes;
// historical schema-1 analytic evidence remains tied to its frozen executable.
TEST(HostSbsV2LiveWarpGpuTest, SavedGlyphTupleReplay) {
  const auto *cases_env = std::getenv("SUNSHINE_TEST_GLYPH_TUPLE_REPLAY_CASES");
  if (!cases_env || !*cases_env) GTEST_SKIP() << "No saved glyph tuple specification selected";
  const auto *output_env = std::getenv("SUNSHINE_TEST_GLYPH_TUPLE_REPLAY_OUTPUT");
  ASSERT_TRUE(output_env && *output_env) << "Saved replay requires an explicit output directory";
  namespace v2 = models::depth_coordinate_v2;
  namespace cache = models::host_sbs_shader_cache;
  namespace fs = std::filesystem;
  using json = nlohmann::json;
  try {
    const auto require = [](bool valid, const std::string &message) {
      if (!valid) throw std::runtime_error(message);
    };
    const auto read_bytes = [](const fs::path &path) {
      std::ifstream input(path, std::ios::binary | std::ios::ate);
      if (!input || input.tellg() < 0) throw std::runtime_error("Cannot read replay input: " + path.string());
      std::vector<char> bytes(static_cast<std::size_t>(input.tellg()));
      input.seekg(0);
      if (!bytes.empty()) input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      if (!input) throw std::runtime_error("Incomplete replay input: " + path.string());
      return bytes;
    };
    const auto sha256 = [](const void *data, std::size_t size) {
      const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), size));
      constexpr char digits[] = "0123456789abcdef";
      std::string hex;
      for (const auto byte : digest) {
        hex.push_back(digits[byte >> 4u]);
        hex.push_back(digits[byte & 15u]);
      }
      return hex;
    };
    const auto write_bytes = [](const fs::path &path, const void *data, std::size_t size) {
      std::ofstream output(path, std::ios::binary);
      output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
      if (!output) throw std::runtime_error("Cannot write replay output: " + path.string());
    };
    const fs::path cases_path(cases_env), output_root(output_env);
    require(!fs::exists(output_root) || (fs::is_directory(output_root) && fs::is_empty(output_root)),
            "Replay output must be a new or empty directory");
    const auto specification_bytes = read_bytes(cases_path);
    const auto specification = json::parse(specification_bytes.begin(), specification_bytes.end());
    require(specification.at("schema") == 2 && specification.at("driver") == "hardware",
            "Current shared glyph replay requires schema 2 and hardware D3D11; schema-1 analytic evidence needs its frozen executable");
    const auto &cases = specification.at("cases");
    require(cases.is_array() && !cases.empty() && cases.size() <= 64u, "Replay requires 1..64 cases");
    std::string error;
    live_v2_warp_fixture_t color, mapping, mask;
    ASSERT_TRUE(color.initialize(error, warp_shader_e::selected_outline, false, true)) << error;
    ASSERT_TRUE(mapping.initialize(error, warp_shader_e::selected_outline_mapping, false, true)) << error;
    ASSERT_TRUE(mask.initialize(error, warp_shader_e::selected_outline_boundary, false, true)) << error;
    const auto adapter = color.adapter_identity();
    require(mapping.adapter_identity() == adapter && mask.adapter_identity() == adapter,
            "Reference routes selected different adapters");
    std::array<wchar_t, 32768u> executable_path {};
    const auto executable_path_size = GetModuleFileNameW(nullptr, executable_path.data(), static_cast<DWORD>(executable_path.size()));
    require(executable_path_size > 0u && executable_path_size < executable_path.size(), "Cannot locate replay executable");
    const auto executable_sha256 = [&]() {
      const auto bytes = read_bytes(fs::path(executable_path.data()));
      return sha256(bytes.data(), bytes.size());
    }();
    const fs::path fixture_path = fs::path(SUNSHINE_SOURCE_DIR) / "tests/unit/test_host_sbs_v2_live_warp.cpp";
    const auto fixture_bytes = read_bytes(fixture_path);
    fs::create_directories(output_root);
    json report {{"schema", 2}, {"driver", "hardware"}, {"adapter", adapter},
      {"test_executable_path", fs::path(executable_path.data()).string()},
      {"test_executable_sha256", executable_sha256},
      {"fixture_source_path", fixture_path.string()},
      {"fixture_source_sha256", sha256(fixture_bytes.data(), fixture_bytes.size())},
      {"input_specification_path", cases_path.string()},
      {"input_specification_sha256", sha256(specification_bytes.data(), specification_bytes.size())},
      {"coordinate_contract_schema", v2::contract_schema},
      {"coordinate_contract_sha256", v2::contract_canonical_sha256},
      {"live_renderer_source_closure_sha256", cache::parallax_v2_live_renderer_source_closure_sha256},
      {"live_diagnostic_source_closure_sha256", cache::parallax_v2_live_diagnostic_source_closure_sha256},
      {"shared_renderer_source_closure_sha256", models::host_sbs_shared_ui_warp::closure},
      {"shared_diagnostic_source_closure_sha256", models::host_sbs_shared_ui_warp::diagnostic_closure},
      {"scope", "Literal saved current matte/receipt and source tuple, actual shared D3D11 renderer; no CPU inverse or color warp"},
      {"route_definitions", {
        {"base", "Saved Base/state/b2 and all saved subtitle resources; only b3.enabled set to zero"},
        {"ui", "Constant GPU field at SCPtarget*(analysis_width/full_width); saved state and PackedToSource content scales; ROI Base embedding disabled; b3.enabled zero"},
        {"selected", "Actual shared warp with the supplied selected matte/receipt and unchanged source/Base/state/b2/b3"}}},
      {"cases", json::array()}};
    for (std::size_t case_index = 0u; case_index < cases.size(); ++case_index) {
      const auto &entry = cases[case_index];
      const auto name = entry.at("name").get<std::string>();
      SCOPED_TRACE(name);
      const auto source_shape = entry.at("source").get<std::array<UINT, 2u>>();
      const auto field_shape = entry.at("field").get<std::array<UINT, 2u>>();
      const auto [width, height] = source_shape;
      const auto [field_width, field_height] = field_shape;
      require(width > 0u && width <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION / 2u &&
              height > 0u && height <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
              field_width > 0u && field_width <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
              field_height > 0u && field_height <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION,
              "Invalid source or field shape");
      json input_bindings = json::object();
      const auto operand = [&](const char *key, std::size_t expected_size) {
        const auto &binding = entry.at(key);
        fs::path path(binding.at("path").get<std::string>());
        if (path.is_relative()) path = cases_path.parent_path() / path;
        const auto bytes = read_bytes(path);
        require(bytes.size() == expected_size, std::string(key) + " byte count/shape mismatch");
        const auto hash = sha256(bytes.data(), bytes.size());
        require(hash == binding.at("sha256").get<std::string>(), std::string(key) + " SHA-256 mismatch");
        input_bindings[key] = {{"path", path.string()}, {"sha256", hash}, {"byte_count", bytes.size()}};
        return bytes;
      };
      const auto load_object = [&]<typename T>(const char *key, T &value) {
        const auto bytes = operand(key, sizeof(value));
        std::memcpy(&value, bytes.data(), sizeof(value));
      };
      const auto load_vector = [&]<typename T>(const char *key, std::size_t count) {
        const auto bytes = operand(key, count * sizeof(T));
        std::vector<T> values(count);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
      };
      models::depth_coordinate_v2::state_words_t state {};
      host_sbs_v2_geometry_t geometry;
      subtitle_inputs_t subtitle;
      load_object("state", state);
      load_object("geometry", geometry);
      load_object("subtitle_constants", subtitle.constants);
      load_object("condition", subtitle.condition);
      load_object("identity", subtitle.identity);
      const auto source = load_vector.template operator()<bgra8_t>("source_bgra8", std::size_t(width) * height);
      const auto base = load_vector.template operator()<float>("base", std::size_t(field_width) * field_height);
      const auto &id = subtitle.identity;
      const auto &scp = subtitle.condition;
      const auto &b3 = subtitle.constants;
      require(v2::parallax_state_words_are_authenticated(state, entry.at("raw_coordinate_scale").get<float>(), v2::adaptive_policy_id),
              "Saved ParallaxState is not authenticated");
      require(state[v2::renderer_authorization_bits] == v2::contract_tag, "Saved frame has no authorized Base");
      require(std::all_of(base.begin(), base.end(), [](float value) {
        return std::isfinite(value) && std::abs(value) <= v2::direct_container_limit;
      }), "Saved Base has nonfinite or out-of-container values");
      require(b3.enabled == 1u && b3.color_mode == 0u && b3.source_width > 0u && b3.source_height > 0u &&
              b3.source_width <= v2::subtitle_contour_source_max_extent &&
              b3.source_height <= v2::subtitle_contour_source_max_extent &&
              b3.source_left <= width && b3.source_top <= height &&
              b3.source_width <= width - b3.source_left && b3.source_height <= height - b3.source_top,
              "Saved b3 must bind an exact SDR8 analysis source inside full color");
      require(std::isfinite(geometry.content_scale_x) && geometry.content_scale_x > 0.0f && geometry.content_scale_x <= 1.0f &&
              std::isfinite(geometry.content_scale_y) && geometry.content_scale_y > 0.0f && geometry.content_scale_y <= 1.0f &&
              geometry.reserved == 0.0f && (geometry.video_roi_active == 0.0f || geometry.video_roi_active == 1.0f),
              "Invalid saved b2 geometry");
      require(id[0] == v2::subtitle_contour_schema && id[1] == v2::subtitle_contour_history_tag &&
              id[2] <= 1u && id[23] == 1u && id[22] == v2::subtitle_contour_recipe &&
              (id[4] != 0u || id[5] != 0u) && id[6] == b3.analysis_generation_low && id[7] == b3.analysis_generation_high &&
              id[8] == b3.source_width && id[9] == b3.source_height && id[11] > 0u &&
              id[10] <= id[9] && id[11] <= id[9] - id[10] &&
              id[12] == field_width && id[13] == field_height &&
              id[16] < id[18] && id[17] < id[19] && id[18] <= field_width && id[19] <= field_height &&
              id[3] == state[v2::confirmed_cut_count] &&
              (id[20] != 0u || id[21] != 0u) && id[20] == state[v2::gain_last_observation_low] &&
              id[21] == state[v2::gain_last_observation_high] &&
              std::all_of(id.begin() + v2::subtitle_contour_history_reserved_offset, id.end(), [](auto value) { return value == 0u; }),
              "Saved SCT owner/observation/cut/generation/shape binding is invalid");
      subtitle.width = id[8];
      subtitle.height = id[11];
      subtitle.matte = load_vector.template operator()<std::uint32_t>("matte", std::size_t(subtitle.width) * subtitle.height);
      subtitle.snapshot = load_vector.template operator()<rgba16_bits_t>("snapshot", subtitle.matte.size());
      current_outline_inputs_t selected;
      selected.matte = load_vector.template operator()<std::uint32_t>("selected_matte", subtitle.matte.size());
      load_object("selected_receipt", selected.receipt);
      require(std::all_of(selected.matte.begin(), selected.matte.end(), [](auto value) { return value <= 1u; }),
              "Saved selected matte is not literal binary R32_UINT");
      for (const auto *key : {"expected_selected_color", "expected_selected_map", "expected_selected_mask"})
        require(!entry.contains(key), "Historical analytic golden keys cannot qualify the current shared renderer");
      require(std::all_of(subtitle.matte.begin(), subtitle.matte.end(), [](auto value) { return value <= 1u; }),
              "Saved matte is not literal binary R32_UINT");
      const float target = std::bit_cast<float>(scp[5]);
      const bool active = scp[2] > 0u;
      require(scp[0] == v2::subtitle_condition_param_schema && scp[1] == v2::subtitle_condition_param_tag &&
              std::isfinite(target) && target >= 0.0f && target <= v2::direct_container_limit &&
              scp[2] <= v2::subtitle_plane_cover_capacity && (scp[3] & ~((1u << scp[2]) - 1u)) == 0u &&
              (active ? scp[4] == v2::subtitle_condition_param_fully_active_value && id[2] == 1u :
                        scp[3] == 0u && scp[4] == 0u && scp[5] == 0u && id[2] == 0u),
              "Saved SCP activity/target disagrees with SCT");
      require(active || std::all_of(subtitle.matte.begin(), subtitle.matte.end(), [](auto value) { return value == 0u; }),
              "Inactive saved tuple retains a positive glyph matte");
      require(active || std::all_of(selected.matte.begin(), selected.matte.end(), [](auto value) { return value == 0u; }),
              "Inactive saved tuple retains a positive selected matte");
      if (geometry.video_roi_active == 1.0f) {
        require(geometry.tensor_content_left == id[16] && geometry.tensor_content_top == id[17] &&
                geometry.tensor_content_right == id[18] && geometry.tensor_content_bottom == id[19],
                "Saved ROI b2 content differs from SCT");
        const std::array actual_roi {geometry.video_roi_left, geometry.video_roi_top, geometry.video_roi_right, geometry.video_roi_bottom};
        const std::array expected_roi {float(b3.source_left) / width, float(b3.source_top) / height,
          float(b3.source_left + b3.source_width) / width, float(b3.source_top + b3.source_height) / height};
        for (std::size_t i = 0u; i < actual_roi.size(); ++i) {
          require(std::isfinite(actual_roi[i]) && actual_roi[i] >= 0.0f && actual_roi[i] <= 1.0f &&
                  std::abs(actual_roi[i] - expected_roi[i]) * (i % 2u == 0u ? width : height) <= 0.25f,
                  "Saved ROI b2 and integer b3 source rectangles differ");
        }
      } else {
        require(b3.source_left == 0u && b3.source_top == 0u && b3.source_width == width && b3.source_height == height,
                "Full-source saved tuple has a conflicting b3 analysis source");
      }
      // A constant production GPU field is the UI reference. Only the Base ROI embedding is
      // disabled: PackedToSource's content scale and the exact saved ParallaxState are unchanged.
      // The full-source displacement is rounded once to FP32, matching the shader's SCP product.
      const float full_source_target = target * (float(b3.source_width) / float(width));
      const std::vector<float> ui_field(base.size(), full_source_target);
      auto ui_geometry = geometry;
      ui_geometry.video_roi_active = 0.0f;
      auto disabled_subtitle = subtitle;
      disabled_subtitle.constants.enabled = 0u;
      const auto case_dir = output_root / ("case_" + std::to_string(case_index));
      fs::create_directory(case_dir);
      json result {{"name", name}, {"source", source_shape}, {"field", field_shape},
        {"packed", {width * 2u, height}}, {"input_bindings", input_bindings},
        {"formats", {{"map", "float32-le raw unclamped source U, packed row-major"},
                     {"color", "BGRA8_UNORM packed row-major"},
                     {"mask", "BGRA8_UNORM packed row-major; red marks raw U outside [0,1]"}}},
        {"provenance", entry.at("provenance")}, {"ui_target_bits", scp[5]},
        {"full_source_ui_target_bits", std::bit_cast<std::uint32_t>(full_source_target)},
        {"state_uploaded_without_modification", true}, {"routes", json::object()}};
      for (const auto route : {"base", "ui", "selected"}) {
        const bool is_ui = std::string_view(route) == "ui";
        const bool is_selected = std::string_view(route) == "selected";
        const auto &field = is_ui ? ui_field : base;
        const auto &render_geometry = is_ui ? ui_geometry : geometry;
        const auto *render_subtitle = is_selected ? &subtitle : &disabled_subtitle;
        for (const auto kind : {"map", "color", "mask"}) {
          const bool is_map = std::string_view(kind) == "map";
          auto &fixture = is_map ? mapping : std::string_view(kind) == "mask" ? mask : color;
          std::vector<std::byte> output;
          shared_ui_pin_diagnostics_t diagnostic;
          selected.shared_ui_pin_diagnostic_output = is_selected ? &diagnostic : nullptr;
          ASSERT_TRUE(fixture.render(DXGI_FORMAT_B8G8R8A8_UNORM, sizeof(bgra8_t), width, height,
            source.data(), field, field, state, {}, output, error, field_width, field_height,
            render_geometry, render_subtitle, nullptr, true, &selected)) << route << '/' << kind << ": " << error;
          if (is_selected) require(diagnostic.selected_receipt_authorized,
            "Saved current shared replay has an invalid selected receipt");
          const auto path = case_dir / (std::string(route) + "_" + kind + (is_map ? ".f32" : ".bgra8"));
          write_bytes(path, output.data(), output.size());
          const auto hash = sha256(output.data(), output.size());
          result["routes"][route][kind] = {{"path", path.string()}, {"sha256", hash}, {"byte_count", output.size()}};
          const std::string expected_key = std::string("expected_shared_") + kind;
          if (is_selected && entry.contains(expected_key)) {
            const auto expected = operand(expected_key.c_str(), output.size());
            const auto expected_hash = sha256(expected.data(), expected.size());
            result["routes"][route][kind]["expected_sha256"] = expected_hash;
            result["routes"][route][kind]["expected_bitexact"] = hash == expected_hash;
            EXPECT_EQ(hash, expected_hash) << "Saved actual shared renderer " << kind << " differs from selected replay";
          }
        }
      }
      result["input_bindings"] = input_bindings;
      const auto result_text = result.dump(2);
      write_bytes(case_dir / "metadata.json", result_text.data(), result_text.size());
      report["cases"].push_back(std::move(result));
    }
    const auto report_text = report.dump(2);
    write_bytes(output_root / "report.json", report_text.data(), report_text.size());
  } catch (const std::exception &failure) {
    FAIL() << failure.what();
  }
}


// Research-only opt-in extension, separate from canonical SavedGlyphTupleReplay.
// Small GPU-reference checks, independent of authored segmentation or scores.
TEST(HostSbsV2LiveWarpGpuTest, PrivateLayerReferenceZeroOpaqueAndPremultipliedSampling) {
  const auto *cases_env = std::getenv("SUNSHINE_TEST_PRIVATE_LAYER_REPLAY_CASES");
  if (!cases_env || !*cases_env) GTEST_SKIP() << "No private reference shader specification selected";
  std::ifstream stream(cases_env);
  ASSERT_TRUE(stream);
  const auto specification = nlohmann::json::parse(stream);
  const bool hardware = specification.at("driver") == "hardware";
  live_v2_warp_fixture_t fixture;
  std::string error;
  ASSERT_TRUE(fixture.initialize(error, warp_shader_e::live_main, false, hardware)) << error;
  constexpr UINT width = 32u, height = 2u;
  const auto state = make_live_state(true, true);
  const std::vector<bgra8_t> background(width * height, {20u, 40u, 80u, 255u});
  const std::vector<float> base(width * height, 0.0f);
  auto subtitle = make_subtitle_inputs(width, height, state, 0.0f);
  std::vector<std::array<std::uint8_t, 4u>> foreground(width * height, {0u, 0u, 255u, 0u});
  std::vector<std::uint8_t> body_alpha(width * height, 0u);
  live_v2_warp_fixture_t::private_layer_output_t result;
  const auto run = [&]() {
    return fixture.render_private_layer_reference(width, height, background, foreground, body_alpha,
      base, state, width, height, {}, subtitle,
      specification.at("private_layer_shader_root").get<std::string>(),
      specification.at("private_layer_source_closure_sha256").get<std::string>(), result, error);
  };
  ASSERT_TRUE(run()) << error;
  ASSERT_EQ(result.reference_rgba8.size(), width * 2u * height * 4u);
  for (std::size_t pixel = 0u; pixel < width * 2u * height; ++pixel) {
    EXPECT_EQ(std::to_integer<unsigned int>(result.reference_rgba8[pixel * 4u]), 80u);
    EXPECT_EQ(std::to_integer<unsigned int>(result.reference_rgba8[pixel * 4u + 1u]), 40u);
    EXPECT_EQ(std::to_integer<unsigned int>(result.reference_rgba8[pixel * 4u + 2u]), 20u);
    EXPECT_EQ(std::to_integer<unsigned int>(result.reference_rgba8[pixel * 4u + 3u]), 255u);
  }
  // Alpha-zero blue texels must not bleed blue into a red half-pixel edge.
  foreground[16u] = {255u, 0u, 0u, 255u};
  body_alpha[16u] = 255u;
  subtitle.condition[5u] = std::bit_cast<std::uint32_t>(0.5f / width);
  ASSERT_TRUE(run()) << error;
  ASSERT_EQ(result.reconstructed_rgba8.size(), width * height * 4u);
  EXPECT_EQ(std::to_integer<unsigned int>(result.reconstructed_rgba8[16u * 4u]), 255u);
  EXPECT_EQ(std::to_integer<unsigned int>(result.reconstructed_rgba8[16u * 4u + 1u]), 0u);
  EXPECT_EQ(std::to_integer<unsigned int>(result.reconstructed_rgba8[16u * 4u + 2u]), 0u);
  for (const auto pixel : {17u, width + 16u}) {
    EXPECT_LE(std::abs(int(std::to_integer<unsigned int>(result.reference_rgba8[pixel * 4u])) - 168), 1);
    EXPECT_LE(std::abs(int(std::to_integer<unsigned int>(result.reference_rgba8[pixel * 4u + 1u])) - 20), 1);
    EXPECT_LE(std::abs(int(std::to_integer<unsigned int>(result.reference_rgba8[pixel * 4u + 2u])) - 10), 1);
    EXPECT_EQ(std::to_integer<unsigned int>(result.reference_rgba8[pixel * 4u + 3u]), 255u);
  }
}

TEST(HostSbsV2LiveWarpGpuTest, PrivateNativeMatteLayerReplay) {
  const auto *cases_env = std::getenv("SUNSHINE_TEST_PRIVATE_LAYER_REPLAY_CASES");
  if (!cases_env || !*cases_env) GTEST_SKIP() << "No saved glyph tuple specification selected";
  const auto *output_env = std::getenv("SUNSHINE_TEST_PRIVATE_LAYER_REPLAY_OUTPUT");
  ASSERT_TRUE(output_env && *output_env) << "Saved replay requires an explicit output directory";
  namespace v2 = models::depth_coordinate_v2;
  namespace cache = models::host_sbs_shader_cache;
  namespace fs = std::filesystem;
  using json = nlohmann::json;
  try {
    const auto require = [](bool valid, const std::string &message) {
      if (!valid) throw std::runtime_error(message);
    };
    const auto read_bytes = [](const fs::path &path) {
      std::ifstream input(path, std::ios::binary | std::ios::ate);
      if (!input || input.tellg() < 0) throw std::runtime_error("Cannot read replay input: " + path.string());
      std::vector<char> bytes(static_cast<std::size_t>(input.tellg()));
      input.seekg(0);
      if (!bytes.empty()) input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      if (!input) throw std::runtime_error("Incomplete replay input: " + path.string());
      return bytes;
    };
    const auto sha256 = [](const void *data, std::size_t size) {
      const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), size));
      constexpr char digits[] = "0123456789abcdef";
      std::string hex;
      for (const auto byte : digest) {
        hex.push_back(digits[byte >> 4u]);
        hex.push_back(digits[byte & 15u]);
      }
      return hex;
    };
    const auto write_bytes = [](const fs::path &path, const void *data, std::size_t size) {
      std::ofstream output(path, std::ios::binary);
      output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
      if (!output) throw std::runtime_error("Cannot write replay output: " + path.string());
    };
    const fs::path cases_path(cases_env), output_root(output_env);
    require(!fs::exists(output_root) || (fs::is_directory(output_root) && fs::is_empty(output_root)),
            "Replay output must be a new or empty directory");
    const auto specification_bytes = read_bytes(cases_path);
    const auto specification = json::parse(specification_bytes.begin(), specification_bytes.end());
    require(specification.at("schema") == 2 && (specification.at("driver") == "hardware" || specification.at("driver") == "warp"),
            "Current shared layered replay requires schema 2 and an explicit hardware/warp driver; schema-1 evidence needs its frozen executable");
    const auto &cases = specification.at("cases");
    require(cases.is_array() && !cases.empty() && cases.size() <= 8u, "Private replay requires 1..8 matched cases");
    std::string error;
    const bool hardware = specification.at("driver") == "hardware";
    live_v2_warp_fixture_t color, mapping;
    ASSERT_TRUE(color.initialize(error, warp_shader_e::selected_outline, false, hardware)) << error;
    ASSERT_TRUE(mapping.initialize(error, warp_shader_e::selected_outline_mapping, false, hardware)) << error;
    const auto adapter = color.adapter_identity();
    require(mapping.adapter_identity() == adapter,
            "Reference routes selected different adapters");
    std::array<wchar_t, 32768u> executable_path {};
    const auto executable_path_size = GetModuleFileNameW(nullptr, executable_path.data(), static_cast<DWORD>(executable_path.size()));
    require(executable_path_size > 0u && executable_path_size < executable_path.size(), "Cannot locate replay executable");
    const auto executable_sha256 = [&]() {
      const auto bytes = read_bytes(fs::path(executable_path.data()));
      return sha256(bytes.data(), bytes.size());
    }();
    const fs::path fixture_path = fs::path(SUNSHINE_SOURCE_DIR) / "tests/unit/test_host_sbs_v2_live_warp.cpp";
    const auto fixture_bytes = read_bytes(fixture_path);
    fs::create_directories(output_root);
    json report {{"schema", 2}, {"driver", specification.at("driver")}, {"adapter", adapter},
      {"test_executable_path", fs::path(executable_path.data()).string()},
      {"test_executable_sha256", executable_sha256},
      {"fixture_source_path", fixture_path.string()},
      {"fixture_source_sha256", sha256(fixture_bytes.data(), fixture_bytes.size())},
      {"input_specification_path", cases_path.string()},
      {"input_specification_sha256", sha256(specification_bytes.data(), specification_bytes.size())},
      {"coordinate_contract_schema", v2::contract_schema},
      {"coordinate_contract_sha256", v2::contract_canonical_sha256},
      {"live_renderer_source_closure_sha256", cache::parallax_v2_live_renderer_source_closure_sha256},
      {"live_diagnostic_source_closure_sha256", cache::parallax_v2_live_diagnostic_source_closure_sha256},
      {"scope", "PRIVATE saved-input old/new native-M rendering through the actual shared warp; supplied receipt is context for counterfactual masks; authored GPU layer reference; no live/producer qualification"},
      {"production_tuple_claim_for_candidate", false},
      {"shared_renderer_source_closure_sha256", models::host_sbs_shared_ui_warp::closure},
      {"shared_diagnostic_source_closure_sha256", models::host_sbs_shared_ui_warp::diagnostic_closure},
      {"private_layer_shader_root", specification.at("private_layer_shader_root")},
      {"private_layer_source_closure_sha256", specification.at("private_layer_source_closure_sha256")},
      {"cases", json::array()}};
    for (std::size_t case_index = 0u; case_index < cases.size(); ++case_index) {
      const auto &entry = cases[case_index];
      const auto name = entry.at("name").get<std::string>();
      SCOPED_TRACE(name);
      const auto source_shape = entry.at("source").get<std::array<UINT, 2u>>();
      const auto field_shape = entry.at("field").get<std::array<UINT, 2u>>();
      const auto [width, height] = source_shape;
      const auto [field_width, field_height] = field_shape;
      require(width > 0u && width <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION / 2u &&
              height > 0u && height <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
              field_width > 0u && field_width <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
              field_height > 0u && field_height <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION,
              "Invalid source or field shape");
      json input_bindings = json::object();
      const auto operand = [&](const char *key, std::size_t expected_size) {
        const auto &binding = entry.at(key);
        fs::path path(binding.at("path").get<std::string>());
        if (path.is_relative()) path = cases_path.parent_path() / path;
        const auto bytes = read_bytes(path);
        require(bytes.size() == expected_size, std::string(key) + " byte count/shape mismatch");
        const auto hash = sha256(bytes.data(), bytes.size());
        require(hash == binding.at("sha256").get<std::string>(), std::string(key) + " SHA-256 mismatch");
        input_bindings[key] = {{"path", path.string()}, {"sha256", hash}, {"byte_count", bytes.size()}};
        return bytes;
      };
      const auto load_object = [&]<typename T>(const char *key, T &value) {
        const auto bytes = operand(key, sizeof(value));
        std::memcpy(&value, bytes.data(), sizeof(value));
      };
      const auto load_vector = [&]<typename T>(const char *key, std::size_t count) {
        const auto bytes = operand(key, count * sizeof(T));
        std::vector<T> values(count);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
      };
      models::depth_coordinate_v2::state_words_t state {};
      host_sbs_v2_geometry_t geometry;
      subtitle_inputs_t subtitle;
      load_object("state", state);
      load_object("geometry", geometry);
      load_object("subtitle_constants", subtitle.constants);
      load_object("condition", subtitle.condition);
      load_object("identity", subtitle.identity);
      const auto source = load_vector.template operator()<bgra8_t>("source_bgra8", std::size_t(width) * height);
      const auto base = load_vector.template operator()<float>("base", std::size_t(field_width) * field_height);
      const auto &id = subtitle.identity;
      const auto &scp = subtitle.condition;
      const auto &b3 = subtitle.constants;
      require(v2::parallax_state_words_are_authenticated(state, entry.at("raw_coordinate_scale").get<float>(), v2::adaptive_policy_id),
              "Saved ParallaxState is not authenticated");
      require(state[v2::renderer_authorization_bits] == v2::contract_tag, "Saved frame has no authorized Base");
      require(std::all_of(base.begin(), base.end(), [](float value) {
        return std::isfinite(value) && std::abs(value) <= v2::direct_container_limit;
      }), "Saved Base has nonfinite or out-of-container values");
      require(b3.enabled == 1u && b3.color_mode == 0u && b3.source_width > 0u && b3.source_height > 0u &&
              b3.source_width <= v2::subtitle_contour_source_max_extent &&
              b3.source_height <= v2::subtitle_contour_source_max_extent &&
              b3.source_left <= width && b3.source_top <= height &&
              b3.source_width <= width - b3.source_left && b3.source_height <= height - b3.source_top,
              "Saved b3 must bind an exact SDR8 analysis source inside full color");
      require(std::isfinite(geometry.content_scale_x) && geometry.content_scale_x > 0.0f && geometry.content_scale_x <= 1.0f &&
              std::isfinite(geometry.content_scale_y) && geometry.content_scale_y > 0.0f && geometry.content_scale_y <= 1.0f &&
              geometry.reserved == 0.0f && (geometry.video_roi_active == 0.0f || geometry.video_roi_active == 1.0f),
              "Invalid saved b2 geometry");
      require(id[0] == v2::subtitle_contour_schema && id[1] == v2::subtitle_contour_history_tag &&
              id[2] <= 1u && id[23] == 1u && id[22] == v2::subtitle_contour_recipe &&
              (id[4] != 0u || id[5] != 0u) && id[6] == b3.analysis_generation_low && id[7] == b3.analysis_generation_high &&
              id[8] == b3.source_width && id[9] == b3.source_height && id[11] > 0u &&
              id[10] <= id[9] && id[11] <= id[9] - id[10] &&
              id[12] == field_width && id[13] == field_height &&
              id[16] < id[18] && id[17] < id[19] && id[18] <= field_width && id[19] <= field_height &&
              id[3] == state[v2::confirmed_cut_count] &&
              (id[20] != 0u || id[21] != 0u) && id[20] == state[v2::gain_last_observation_low] &&
              id[21] == state[v2::gain_last_observation_high] &&
              std::all_of(id.begin() + v2::subtitle_contour_history_reserved_offset, id.end(), [](auto value) { return value == 0u; }),
              "Saved SCT owner/observation/cut/generation/shape binding is invalid");
      subtitle.width = id[8];
      subtitle.height = id[11];
      subtitle.matte = load_vector.template operator()<std::uint32_t>("matte", std::size_t(subtitle.width) * subtitle.height);
      subtitle.snapshot = load_vector.template operator()<rgba16_bits_t>("snapshot", subtitle.matte.size());
      current_outline_inputs_t selected;
      load_object("selected_receipt", selected.receipt);
      for (const auto *key : {"expected_selected_color", "expected_selected_map"})
        require(!entry.contains(key), "Historical analytic golden keys cannot qualify current shared mask experiments");
      require(std::all_of(subtitle.matte.begin(), subtitle.matte.end(), [](auto value) { return value <= 1u; }),
              "Saved matte is not literal binary R32_UINT");
      const float target = std::bit_cast<float>(scp[5]);
      const bool active = scp[2] > 0u;
      require(scp[0] == v2::subtitle_condition_param_schema && scp[1] == v2::subtitle_condition_param_tag &&
              std::isfinite(target) && target >= 0.0f && target <= v2::direct_container_limit &&
              scp[2] <= v2::subtitle_plane_cover_capacity && (scp[3] & ~((1u << scp[2]) - 1u)) == 0u &&
              (active ? scp[4] == v2::subtitle_condition_param_fully_active_value && id[2] == 1u :
                        scp[3] == 0u && scp[4] == 0u && scp[5] == 0u && id[2] == 0u),
              "Saved SCP activity/target disagrees with SCT");
      require(active || std::all_of(subtitle.matte.begin(), subtitle.matte.end(), [](auto value) { return value == 0u; }),
              "Inactive saved tuple retains a positive glyph matte");
      if (geometry.video_roi_active == 1.0f) {
        require(geometry.tensor_content_left == id[16] && geometry.tensor_content_top == id[17] &&
                geometry.tensor_content_right == id[18] && geometry.tensor_content_bottom == id[19],
                "Saved ROI b2 content differs from SCT");
        const std::array actual_roi {geometry.video_roi_left, geometry.video_roi_top, geometry.video_roi_right, geometry.video_roi_bottom};
        const std::array expected_roi {float(b3.source_left) / width, float(b3.source_top) / height,
          float(b3.source_left + b3.source_width) / width, float(b3.source_top + b3.source_height) / height};
        for (std::size_t i = 0u; i < actual_roi.size(); ++i) {
          require(std::isfinite(actual_roi[i]) && actual_roi[i] >= 0.0f && actual_roi[i] <= 1.0f &&
                  std::abs(actual_roi[i] - expected_roi[i]) * (i % 2u == 0u ? width : height) <= 0.25f,
                  "Saved ROI b2 and integer b3 source rectangles differ");
        }
      } else {
        require(b3.source_left == 0u && b3.source_top == 0u && b3.source_width == width && b3.source_height == height,
                "Full-source saved tuple has a conflicting b3 analysis source");
      }
      const auto candidate_identity_bytes = operand("candidate_native_identity", entry.at("candidate_native_identity").at("byte_count").get<std::size_t>());
      require(!candidate_identity_bytes.empty(), "Private mask needs an actual native-run identity binding");
      std::array<std::uint32_t, 24u> candidate_constants {};
      load_object("candidate_constants", candidate_constants);
      require(candidate_constants[0] == id[8] && candidate_constants[1] == id[9] &&
              candidate_constants[2] == id[10] && candidate_constants[3] == id[11] &&
              candidate_constants[4] == id[12] && candidate_constants[5] == id[13] &&
              candidate_constants[6] == id[14] && candidate_constants[7] == id[15] &&
              std::equal(candidate_constants.begin() + 8u, candidate_constants.begin() + 12u, id.begin() + 16u) &&
              candidate_constants[12] == id[4] && candidate_constants[13] == id[5] &&
              candidate_constants[14] == id[6] && candidate_constants[15] == id[7] &&
              candidate_constants[16] == id[20] && candidate_constants[17] == id[21] &&
              candidate_constants[19] == 0u && candidate_constants[22] == 0u && candidate_constants[23] == 0u,
              "Private native mask CB differs from the saved source/crop/field/content/frame/generation/clock context");
      auto candidate_subtitle = subtitle;
      candidate_subtitle.matte = load_vector.template operator()<std::uint32_t>("candidate_matte", subtitle.matte.size());
      require(std::all_of(candidate_subtitle.matte.begin(), candidate_subtitle.matte.end(), [](auto value) { return value <= 1u; }),
              "Private candidate matte is not literal binary R32_UINT");
      require(active || std::all_of(candidate_subtitle.matte.begin(), candidate_subtitle.matte.end(), [](auto value) { return value == 0u; }),
              "Inactive frame cannot publish a positive private candidate matte");
      const auto background = load_vector.template operator()<bgra8_t>("background_bgra8", std::size_t(width) * height);
      const auto foreground = load_vector.template operator()<std::array<std::uint8_t, 4u>>("foreground_rgba8", background.size());
      const auto effective_body_alpha = load_vector.template operator()<std::uint8_t>("effective_body_alpha_u8", background.size());
      require(std::all_of(background.begin(), background.end(), [](const auto &value) { return value.a == 255u; }) &&
              std::all_of(source.begin(), source.end(), [](const auto &value) { return value.a == 255u; }),
              "Private SDR reference currently requires an opaque authored background and source");
      const auto case_dir = output_root / ("case_" + std::to_string(case_index));
      fs::create_directory(case_dir);
      json result {{"name", name}, {"source", source_shape}, {"field", field_shape},
        {"packed", {width * 2u, height}}, {"input_bindings", input_bindings},
        {"provenance", entry.at("provenance")}, {"ui_target_bits", scp[5]},
        {"state_uploaded_without_modification", true}, {"geometry_uploaded_without_modification", true},
        {"recipe27_identity_is_context_only_for_candidate", true},
        {"candidate_is_canonical_recipe27_matte", false},
        {"candidate_matte_transport", "literal previously GPU-produced native mask; no CPU inferred pixels"},
        {"routes", json::object()}, {"reference", json::object()}};
      const auto save = [&](const std::string &filename, const std::vector<std::byte> &bytes) {
        const auto path = case_dir / filename;
        write_bytes(path, bytes.data(), bytes.size());
        return json {{"path", path.string()}, {"sha256", sha256(bytes.data(), bytes.size())}, {"byte_count", bytes.size()}};
      };
      auto disabled_subtitle = subtitle;
      disabled_subtitle.constants.enabled = 0u;
      for (const auto route : {"base_composite", "old_matte", "candidate_matte"}) {
        const bool candidate_route = std::string_view(route) == "candidate_matte";
        const bool old_route = std::string_view(route) == "old_matte";
        const auto *render_subtitle = candidate_route ? &candidate_subtitle : old_route ? &subtitle : &disabled_subtitle;
        auto operands = selected;operands.matte = render_subtitle->matte;
        for (const auto kind : {"map", "color"}) {
          const bool is_map = std::string_view(kind) == "map";
          auto &fixture = is_map ? mapping : color;
          std::vector<std::byte> output;
          shared_ui_pin_diagnostics_t diagnostic;
          operands.shared_ui_pin_diagnostic_output = &diagnostic;
          ASSERT_TRUE(fixture.render(DXGI_FORMAT_B8G8R8A8_UNORM, sizeof(bgra8_t), width, height,
            source.data(), base, base, state, {}, output, error, field_width, field_height,
            geometry, render_subtitle, nullptr, true, &operands)) << route << '/' << kind << ": " << error;
          require(diagnostic.selected_receipt_authorized,
            "Private shared mask replay needs a valid common context receipt");
          const auto filename = std::string(route) + "_" + kind + (is_map ? ".f32" : ".bgra8");
          result["routes"][route][kind] = save(filename, output);
          const std::string expected_key = std::string("expected_shared_") + kind;
          if (old_route && entry.contains(expected_key)) {
            const auto expected = operand(expected_key.c_str(), output.size());
            const auto expected_hash = sha256(expected.data(), expected.size());
            result["routes"][route][kind]["expected_bitexact"] = sha256(output.data(), output.size()) == expected_hash;
            EXPECT_EQ(sha256(output.data(), output.size()), expected_hash) << "Baseline shared replay differs from its supplied current shared output";
          }
        }
      }
      live_v2_warp_fixture_t::private_layer_output_t layered;
      ASSERT_TRUE(color.render_private_layer_reference(width, height, background, foreground, effective_body_alpha, base, state,
        field_width, field_height, geometry, subtitle,
        specification.at("private_layer_shader_root").get<std::string>(),
        specification.at("private_layer_source_closure_sha256").get<std::string>(), layered, error)) << error;
      require(layered.reconstructed_rgba8.size() == source.size() * 4u, "Reconstruction shape differs from full source");
      std::array<std::uint64_t, 256u> identity_max_error_histogram {};
      std::uint32_t maximum_identity_rgb_error = 0u;
      for (std::size_t pixel = 0u; pixel < source.size(); ++pixel) {
        const std::array<std::uint8_t, 3u> expected {source[pixel].r, source[pixel].g, source[pixel].b};
        std::uint32_t maximum = 0u;
        for (std::size_t channel = 0u; channel < 3u; ++channel) {
          const auto actual = std::to_integer<unsigned int>(layered.reconstructed_rgba8[pixel * 4u + channel]);
          maximum = std::max(maximum, static_cast<std::uint32_t>(std::abs(int(actual) - int(expected[channel]))));
        }
        ++identity_max_error_histogram[maximum];
        maximum_identity_rgb_error = std::max(maximum_identity_rgb_error, maximum);
      }
      result["reference"] = {
        {"reconstructed_source", save("reference_reconstructed_source.rgba8", layered.reconstructed_rgba8)},
        {"base_background", save("reference_base_background.bgra8", layered.base_background_bgra8)},
        {"layered_color", save("reference_layered.rgba8", layered.reference_rgba8)},
        {"ui_foreground_premultiplied", save("reference_ui_foreground.rgba32f", layered.shifted_foreground_rgba32f)},
        {"ui_effective_body_alpha", save("reference_ui_effective_body_alpha.f32", layered.shifted_body_alpha_f32)},
        {"identity_max_rgb8_error", maximum_identity_rgb_error},
        {"identity_rgb8_max_error_histogram", identity_max_error_histogram},
        {"predeclared_identity_tolerance_rgb8", 1u},
        {"authored_layer_identity_pass", maximum_identity_rgb_error <= 1u},
        {"body_and_optional_black_outline_metrics_are_separate", true},
        {"color_domain", "SDR UNORM numeric RGB, matching production SourceColor; not linear-light conversion"},
        {"scope", "Production Base warp of authored background plus premultiplied authored RGBA sampled at current UI plane, composited entirely on GPU"}};
      result["input_bindings"] = input_bindings;
      const auto result_text = result.dump(2);
      write_bytes(case_dir / "metadata.json", result_text.data(), result_text.size());
      report["cases"].push_back(std::move(result));
      // A failed plate/source reconstruction is evidence, never permission to
      // reinterpret the reference or silently repair the supplied author layers.
      EXPECT_LE(maximum_identity_rgb_error, 1u) << "Authored layers do not reconstruct the actual current source under the predeclared GPU compositor";
    }
    const auto report_text = report.dump(2);
    write_bytes(output_root / "report.json", report_text.data(), report_text.size());
  } catch (const std::exception &failure) {
    FAIL() << failure.what();
  }
}


// Independent authored layers exercise renderer differences, not OCR or mask-estimator quality.
TEST(HostSbsV2LiveWarpGpuTest, SharedUiPinAuthoredStressComparison) {
  const auto *cases_env=std::getenv("SUNSHINE_TEST_SHARED_UI_PIN_STRESS_CASES");
  if(!cases_env || !*cases_env)GTEST_SKIP()<<"No shared UI pin authored stress specification selected";
  const auto *output_env=std::getenv("SUNSHINE_TEST_SHARED_UI_PIN_STRESS_OUTPUT");
  const auto *shader_root=std::getenv("SUNSHINE_TEST_SHARED_UI_PIN_SHADER_ROOT");
  const auto *shader_pin=std::getenv("SUNSHINE_TEST_SHARED_UI_PIN_SHADER_PIN");
  const auto *reference_env=std::getenv("SUNSHINE_TEST_PRIVATE_LAYER_REPLAY_CASES");
  ASSERT_TRUE(output_env && shader_root && shader_pin && reference_env);
  namespace fs=std::filesystem;using json=nlohmann::json;
  try {
    const auto require=[](bool pass,const std::string &message){if(!pass)throw std::runtime_error(message);};
    const auto read=[&](const fs::path &path){
      std::ifstream stream(path,std::ios::binary|std::ios::ate);require(bool(stream)&&stream.tellg()>=0,"Cannot read authored stress operand");
      std::vector<std::byte> bytes(static_cast<std::size_t>(stream.tellg()));stream.seekg(0);
      stream.read(reinterpret_cast<char *>(bytes.data()),static_cast<std::streamsize>(bytes.size()));require(bool(stream),"Incomplete authored stress operand");return bytes;
    };
    const auto sha=[](const auto &bytes){
      const auto hash=crypto::hash(std::string_view(reinterpret_cast<const char *>(bytes.data()),bytes.size()));
      constexpr char digits[]="0123456789abcdef";std::string value;
      for(auto b:hash){value.push_back(digits[b>>4u]);value.push_back(digits[b&15u]);}return value;
    };
    const auto specification_bytes=read(cases_env),reference_bytes=read(reference_env);
    const auto spec=json::parse(std::string_view(reinterpret_cast<const char *>(specification_bytes.data()),specification_bytes.size()));
    const auto reference_spec=json::parse(std::string_view(reinterpret_cast<const char *>(reference_bytes.data()),reference_bytes.size()));
    require(spec.at("schema")==1u && spec.at("driver")=="hardware" && spec.at("cases").size()>=1u && spec.at("cases").size()<=16u,"Authored stress schema/driver/case count");
    const fs::path output(output_env);require(!fs::exists(output)||(fs::is_directory(output)&&fs::is_empty(output)),"Authored stress output must be new or empty");fs::create_directories(output);
    live_v2_warp_fixture_t fixture;std::string error;
    ASSERT_TRUE(fixture.initialize(error,warp_shader_e::selected_outline,false,true))<<error;
    ASSERT_TRUE(fixture.configure_shared_ui_pin(shader_root,shader_pin,error))<<error;
    json report={{"schema",1u},{"driver","hardware"},{"adapter",fixture.adapter_identity()},
      {"input_specification",{{"path",cases_env},{"sha256",sha(specification_bytes)}}},
      {"reference_specification",{{"path",reference_env},{"sha256",sha(reference_bytes)}}},
      {"shared_ui_pin_source_closure_sha256",shader_pin},
      {"scope","Authored fixed-mask shared warp with independent actual-GPU known-layer reference; retired repair route is not executed"},
      {"cases",json::array()}};
    for(std::size_t index=0u;index<spec.at("cases").size();++index) {
      const auto &entry=spec.at("cases").at(index);SCOPED_TRACE(entry.at("name").get<std::string>());
      const auto shape=entry.at("source").get<std::array<UINT,2u>>(),field=entry.at("field").get<std::array<UINT,2u>>();
      const auto rect=entry.at("analysis_rect").get<std::array<UINT,4u>>();const UINT width=shape[0u],height=shape[1u];
      require(width>=32u&&height>=2u&&width<=5120u&&height<=5120u&&rect[2u]>0u&&rect[3u]>0u&&rect[0u]+rect[2u]<=width&&rect[1u]+rect[3u]<=height,"Authored stress geometry");
      const auto crop=entry.value("native_crop",std::array<UINT,3u>{rect[2u],0u,rect[3u]});
      require(crop[0u]==rect[2u]&&crop[2u]>0u&&crop[1u]<=rect[3u]&&crop[2u]<=rect[3u]-crop[1u],"Authored native crop facts");
      json bindings=json::object();
      const auto load=[&]<typename T>(const char *name,std::size_t count){
        const auto &binding=entry.at(name);const auto bytes=read(binding.at("path").get<std::string>());
        require(bytes.size()==count*sizeof(T)&&binding.at("byte_count")==bytes.size()&&binding.at("sha256")==sha(bytes),std::string("Authored stress binding: ")+name);
        bindings[name]=binding;std::vector<T> values(count);std::memcpy(values.data(),bytes.data(),bytes.size());return values;
      };
      const auto source=load.template operator()<bgra8_t>("source_bgra8",std::size_t(width)*height);
      const auto background=load.template operator()<bgra8_t>("background_bgra8",source.size());
      const auto foreground=load.template operator()<std::array<std::uint8_t,4u>>("foreground_rgba8",source.size());
      const auto body=load.template operator()<std::uint8_t>("effective_body_alpha_u8",source.size());
      const auto base=load.template operator()<float>("base",std::size_t(field[0u])*field[1u]);
      auto state=make_live_state(true,true);
      const float target=entry.contains("ui_local_target_bits")?std::bit_cast<float>(entry.at("ui_local_target_bits").get<std::uint32_t>()):entry.at("ui_local_target").get<float>();
      auto subtitle=make_subtitle_inputs(rect[2u],rect[3u],state,target,field[0u],field[1u]);
      subtitle.constants.source_left=rect[0u];subtitle.constants.source_top=rect[1u];subtitle.identity[4u]=static_cast<UINT>(index+1u);
      subtitle.width=crop[0u];subtitle.height=crop[2u];subtitle.identity[10u]=crop[1u];subtitle.identity[11u]=crop[2u];
      if(entry.contains("content_rect")) {
        const auto content=entry.at("content_rect").get<std::array<UINT,4u>>();std::copy(content.begin(),content.end(),subtitle.identity.begin()+16u);
      }
      if(entry.contains("field_rows")) {
        const auto rows=entry.at("field_rows").get<std::array<UINT,2u>>();std::copy(rows.begin(),rows.end(),subtitle.identity.begin()+14u);
      }
      subtitle.matte=load.template operator()<std::uint32_t>("matte",std::size_t(crop[0u])*crop[2u]);
      subtitle.snapshot=load.template operator()<rgba16_bits_t>("snapshot",subtitle.matte.size());
      host_sbs_v2_geometry_t geometry;const auto scale=entry.at("content_scale").get<std::array<float,2u>>();
      geometry.content_scale_x=scale[0u];geometry.content_scale_y=scale[1u];
      if(entry.at("roi").get<bool>()) {
        geometry.video_roi_active=1.0f;geometry.video_roi_left=float(rect[0u])/width;geometry.video_roi_top=float(rect[1u])/height;
        geometry.video_roi_right=float(rect[0u]+rect[2u])/width;geometry.video_roi_bottom=float(rect[1u]+rect[3u])/height;
        geometry.tensor_content_left=0u;geometry.tensor_content_top=0u;geometry.tensor_content_right=field[0u];geometry.tensor_content_bottom=field[1u];
      }
      if(entry.contains("geometry")) {
        const auto bytes=load.template operator()<std::byte>("geometry",sizeof(geometry));std::memcpy(&geometry,bytes.data(),sizeof(geometry));
      }
      auto selected=make_current_outline_inputs(subtitle);
      const auto folder=output/("case_"+std::to_string(index));fs::create_directory(folder);
      const auto save=[&](const std::string &name,const void *data,std::size_t size){
        std::vector<std::byte> bytes(size);std::memcpy(bytes.data(),data,size);const auto path=folder/name;
        std::ofstream stream(path,std::ios::binary);stream.write(reinterpret_cast<const char *>(bytes.data()),static_cast<std::streamsize>(size));stream.close();require(bool(stream),"Cannot save authored stress artifact");
        return json{{"path",path.string()},{"byte_count",size},{"sha256",sha(bytes)}};
      };
      const auto save_vector=[&](const std::string &name,const auto &values){return save(name,values.data(),values.size()*sizeof(typename std::decay_t<decltype(values)>::value_type));};
      bindings["state"]=save("state.u32",state.data(),sizeof(state));bindings["geometry"]=save("geometry.b2",&geometry,sizeof(geometry));
      bindings["subtitle_constants"]=save("subtitle.b3",&subtitle.constants,sizeof(subtitle.constants));
      bindings["condition"]=save("condition.u32",subtitle.condition.data(),sizeof(subtitle.condition));
      bindings["identity"]=save("identity.u32",subtitle.identity.data(),sizeof(subtitle.identity));
      bindings["comparison_selected_matte"]=bindings.at("matte");
      bindings["comparison_selected_receipt"]=save("selected_receipt.u32",selected.receipt.data(),sizeof(selected.receipt));
      json result={{"name",entry.at("name")},{"source",shape},{"field",field},{"packed",{width*2u,height}},
        {"input_bindings",bindings},{"authored_parameters",entry},{"routes",json::object()},
        {"comparison_preparation","Current SCT/SCP/receipt explicitly authored by fixture; all renderer operands common, no current-producer claim"}};
      if(entry.contains("original_input_bindings"))result["original_input_bindings"]=entry.at("original_input_bindings");
      for(const auto *route:{"shared_ui_pin","shared_ui_pin_resample_only","base_composite"}) {
        const bool pin=std::string_view(route)=="shared_ui_pin"||std::string_view(route)=="shared_ui_pin_resample_only";
        auto operands=selected;auto ui=subtitle;operands.shared_ui_pin=pin;
        operands.shared_ui_pin_resample_only=std::string_view(route)=="shared_ui_pin_resample_only";
        if(std::string_view(route)=="base_composite")ui.constants.enabled=0u;
        shared_ui_pin_diagnostics_t diagnostic;if(pin)operands.shared_ui_pin_diagnostic_output=&diagnostic;
        double elapsed=-1.0;operands.render_gpu_ms=&elapsed;std::vector<std::byte> pixels;
        ASSERT_TRUE(fixture.render(DXGI_FORMAT_B8G8R8A8_UNORM,4u,width,height,source.data(),base,base,state,{},pixels,error,
          field[0u],field[1u],geometry,&ui,nullptr,true,&operands))<<route<<": "<<error;
        result["routes"][route]={{"color",save_vector(std::string(route)+".bgra8",pixels)},{"prepare_and_draw_gpu_ms",elapsed}};
        if(pin)result["routes"][route]["shared_ui_pin"]={{"shape",diagnostic.shape},
          {"field",save_vector(std::string(route)+".field.f32",diagnostic.field)},
          {"authorized_weights",save_vector(std::string(route)+".weights.f32",diagnostic.authorized_weights)},
          {"active_rows",save_vector(std::string(route)+".rows.u32",diagnostic.active_rows)},
          {"selected_receipt_authorized",diagnostic.selected_receipt_authorized},
          {"row_header_schema",1u},{"row_header",save_vector(std::string(route)+".row-header.u32",diagnostic.selected_receipt_header)},
          {"mapping",save_vector(std::string(route)+".mapping.rgba32f",diagnostic.mapping_rgba32f)},
          {"ui_coverage",save_vector(std::string(route)+".coverage.rgba32f",diagnostic.ui_coverage_rgba32f)}};
      }
      live_v2_warp_fixture_t::private_layer_output_t layered;
      ASSERT_TRUE(fixture.render_private_layer_reference(width,height,background,foreground,body,base,state,field[0u],field[1u],geometry,subtitle,
        reference_spec.at("private_layer_shader_root").get<std::string>(),reference_spec.at("private_layer_source_closure_sha256").get<std::string>(),layered,error))<<error;
      UINT maximum=0u;for(std::size_t pixel=0u;pixel<source.size();++pixel)for(UINT channel=0u;channel<3u;++channel) {
        const auto expected=channel==0u?source[pixel].r:channel==1u?source[pixel].g:source[pixel].b;
        maximum=std::max(maximum,UINT(std::abs(int(std::to_integer<unsigned int>(layered.reconstructed_rgba8[pixel*4u+channel]))-int(expected))));
      }
      result["reference"]={{"reconstructed_source",save_vector("reference_source.rgba8",layered.reconstructed_rgba8)},
        {"base_background",save_vector("reference_base_background.bgra8",layered.base_background_bgra8)},
        {"layered_color",save_vector("reference_layered.rgba8",layered.reference_rgba8)},
        {"ui_foreground_premultiplied",save_vector("reference_foreground.rgba32f",layered.shifted_foreground_rgba32f)},
        {"ui_effective_body_alpha",save_vector("reference_body.f32",layered.shifted_body_alpha_f32)},
        {"identity_max_rgb8_error",maximum},{"predeclared_identity_tolerance_rgb8",1u},{"authored_layer_identity_pass",maximum<=1u}};
      EXPECT_LE(maximum,1u)<<"Author layers must reconstruct source independently";
      // Invalid authority, explicit empty M, and current-RGB revocation must retain exact Base.
      if(index==0u) {
        json adversaries=json::array();
        for(const auto *kind:{"empty","stale_sct","revoked_rgb","stale_receipt"}) {
          auto ui=subtitle;auto operands=selected;operands.shared_ui_pin=true;
          if(std::string_view(kind)=="empty")std::fill(operands.matte.begin(),operands.matte.end(),0u);
          if(std::string_view(kind)=="stale_sct")ui.identity[20u]^=1u;
          if(std::string_view(kind)=="revoked_rgb")for(auto &pixel:ui.snapshot)pixel={0u,0x3c00u,0x3c00u,0x3c00u};
          if(std::string_view(kind)=="stale_receipt")operands.receipt[28u+20u]^=1u;
          std::vector<std::byte> actual,expected;
          ASSERT_TRUE(fixture.render(DXGI_FORMAT_B8G8R8A8_UNORM,4u,width,height,source.data(),base,base,state,{},actual,error,
            field[0u],field[1u],geometry,&ui,nullptr,true,&operands))<<kind<<": "<<error;
          // Every rejected shared receipt/source tuple retains ordinary Base.
          operands.shared_ui_pin=false;ui.constants.enabled=0u;
          ASSERT_TRUE(fixture.render(DXGI_FORMAT_B8G8R8A8_UNORM,4u,width,height,source.data(),base,base,state,{},expected,error,
            field[0u],field[1u],geometry,&ui,nullptr,true,&operands))<<error;
          EXPECT_EQ(actual,expected)<<kind;adversaries.push_back({{"kind",kind},{"actual_sha256",sha(actual)},{"reference_sha256",sha(expected)},{"bitexact",actual==expected}});
        }
        result["authority_adversaries"]=adversaries;
      }
      for(const auto &[key,binding]:bindings.items())if(entry.contains(key)) {
        const auto bytes=read(binding.at("path").get<std::string>());
        require(bytes.size()==binding.at("byte_count").get<std::size_t>()&&sha(bytes)==binding.at("sha256").get<std::string>(),"Authored stress input changed during run");
      }
      const auto text=result.dump(2);save("metadata.json",text.data(),text.size());report["cases"].push_back(std::move(result));
    }
    require(read(cases_env)==specification_bytes&&read(reference_env)==reference_bytes&&fixture.shared_ui_pin_source_unchanged(),"Authored stress source/specification drift");
    const auto text=report.dump(2);std::ofstream stream(output/"report.json",std::ios::binary);stream<<text;require(bool(stream),"Cannot save authored stress report");
  }catch(const std::exception &failure){FAIL()<<failure.what();}
}

TEST(HostSbsV2LiveWarpGpuTest, SharedUiPinPreservesNativeGlyphAndAuthorityAcrossColourRoiAndFractionalSampling) {
  const auto *external_root=std::getenv("SUNSHINE_TEST_SHARED_UI_PIN_SHADER_ROOT"),*external_pin=std::getenv("SUNSHINE_TEST_SHARED_UI_PIN_SHADER_PIN");
  const std::filesystem::path root=external_root&&*external_root?external_root:SUNSHINE_SHADERS_DIR;
  const auto source=models::host_sbs_shader_cache::snapshot_sources(root,live_v2_warp_fixture_t::shared_ui_pin_specs);
  ASSERT_TRUE(source);
  const std::string pin=external_root&&*external_root?(external_pin?external_pin:""):
    models::host_sbs_shader_cache::source_closure_sha256(source);
  ASSERT_EQ(pin.size(),64u);live_v2_warp_fixture_t fixture;std::string error;
  ASSERT_TRUE(fixture.initialize(error,warp_shader_e::selected_outline,false,true))<<error;
  ASSERT_TRUE(fixture.configure_shared_ui_pin(root,pin,error))<<error;
  constexpr UINT W=128u,H=32u,FW=24u,FH=10u;const auto state=make_live_state(true,true);
  for(bool roi:{false,true})for(float sign:{-1.0f,1.0f})for(float shift:{2.0f,2.5f})for(UINT mode:{0u,1u,2u})for(float scale_y:{1.0f,0.8125f}) {
    SCOPED_TRACE(std::to_string(roi)+"/"+std::to_string(sign)+"/"+std::to_string(shift)+"/"+std::to_string(mode)+"/"+std::to_string(scale_y));
    const UINT aw=roi?64u:W,ah=roi?16u:H,left=roi?32u:0u,top=roi?8u:0u;
    auto ui=make_subtitle_inputs(aw,ah,state,shift/float(aw),FW,FH);ui.constants.color_mode=mode;
    ui.constants.source_left=left;ui.constants.source_top=top;
    host_sbs_v2_geometry_t geometry;geometry.content_scale_y=scale_y;
    if(roi){geometry.video_roi_active=1.0f;geometry.video_roi_left=geometry.video_roi_top=0.25f;
      geometry.video_roi_right=geometry.video_roi_bottom=0.75f;geometry.tensor_content_right=FW;geometry.tensor_content_bottom=FH;}
    std::vector<rgba16_bits_t> source(W*H,{0x3400u,0x3400u,0x3400u,0x3c00u});
    const std::uint16_t raw=mode==2u?0x3c00u:0x3800u,display=mode==0u?0x3800u:0x39e2u;
    const UINT first_y=mode==0u?0u:mode==2u?ah-7u:ah/2u-3u;
    for(UINT y=first_y;y<first_y+7u;++y)for(UINT x=aw/2u-4u;x<aw/2u+5u;++x) {
      ui.matte[std::size_t(y)*aw+x]=1u;ui.snapshot[std::size_t(y)*aw+x]={display,display,display,0x3c00u};
      source[std::size_t(y+top)*W+x+left]={raw,raw,raw,0x3c00u};
    }
    const std::vector<float> base(FW*FH,sign*1.25f/float(aw));auto operands=make_current_outline_inputs(ui);
    operands.shared_ui_pin=true;shared_ui_pin_diagnostics_t diagnostic;operands.shared_ui_pin_diagnostic_output=&diagnostic;
    std::vector<std::byte> actual,production,constant_ui,base_output,revoked;
    ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},actual,error,FW,FH,geometry,&ui,nullptr,true,&operands))<<error;
    auto production_operands=operands;production_operands.production_shared_ui_pin=true;
    shared_ui_pin_diagnostics_t production_diagnostic;production_operands.shared_ui_pin_diagnostic_output=&production_diagnostic;
    ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},production,error,FW,FH,geometry,&ui,nullptr,true,&production_operands))<<error;
    EXPECT_EQ(production,actual)<<"Production owner must preserve the qualified shared warp pixels";
    EXPECT_EQ(production_diagnostic.shape,diagnostic.shape);
    EXPECT_EQ(production_diagnostic.field,diagnostic.field);
    EXPECT_EQ(production_diagnostic.authorized_weights,diagnostic.authorized_weights);
    EXPECT_EQ(production_diagnostic.active_rows,diagnostic.active_rows);
    EXPECT_EQ(production_diagnostic.selected_receipt_header,diagnostic.selected_receipt_header);
    auto disabled=ui;disabled.constants.enabled=0u;auto flat_geometry=geometry;flat_geometry.video_roi_active=0.0f;
    const float full_plane=(shift/float(aw))*(float(aw)/float(W));const std::vector<float> constant(FW*FH,full_plane);
    auto ordinary=make_current_outline_inputs(disabled);
    ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),constant,constant,state,{},constant_ui,error,FW,FH,flat_geometry,&disabled,nullptr,true,&ordinary))<<error;
    ASSERT_EQ(diagnostic.ui_coverage_rgba32f.size(),std::size_t(W)*H*2u*16u);
    const auto coverage=unpack_rgba32f(diagnostic.ui_coverage_rgba32f);std::size_t full=0u;
    for(std::size_t i=0u;i<coverage.size();++i)if(coverage[i].g==1.0f) {
      ++full;EXPECT_EQ(std::memcmp(actual.data()+i*8u,constant_ui.data()+i*8u,8u),0)<<"Full marked footprint must retain exact rigid native sample at pixel "<<i;
    }
    EXPECT_GT(full,0u);EXPECT_TRUE(diagnostic.selected_receipt_authorized);
    for(UINT y=0u;y<diagnostic.shape[3u];++y)for(UINT x=1u;x<W;++x)
      EXPECT_LE(std::abs(double(diagnostic.field[std::size_t(y)*W+x])-double(diagnostic.field[std::size_t(y)*W+x-1u]))*W,0.50001);
    ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},base_output,error,FW,FH,geometry,&disabled,nullptr,true,&ordinary))<<error;
    auto bad=ui;for(auto &pixel:bad.snapshot)pixel={0u,0x3c00u,0x3c00u,0x3c00u};
    operands.shared_ui_pin_diagnostic_output=nullptr;
    ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},revoked,error,FW,FH,geometry,&bad,nullptr,true,&operands))<<error;
    EXPECT_EQ(revoked,base_output)<<"All current glyph RGB was revoked";
    production_operands.shared_ui_pin_diagnostic_output=nullptr;
    ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},production,error,FW,FH,geometry,&bad,nullptr,true,&production_operands))<<error;
    EXPECT_EQ(production,base_output)<<"Production owner must recheck current glyph RGB";
  }
  EXPECT_TRUE(fixture.shared_ui_pin_source_unchanged());
}

TEST(HostSbsV2LiveWarpGpuTest, SharedUiPinProductionRetainsAuthorityAndClearsReusedRows) {
  constexpr UINT width = 128u, height = 32u, field_width = 24u, field_height = 10u;
  const auto state = make_live_state(true, true);
  const std::vector<float> base(field_width * field_height, -1.25f / width);
  std::vector<rgba16_bits_t> source(width * height, {0x3400u, 0x3400u, 0x3400u, 0x3c00u});
  auto ui = make_subtitle_inputs(width, height, state, 2.5f / width, field_width, field_height);
  for (const UINT x : {56u, 64u}) {
    const auto pixel = 15u * width + x;
    source[pixel] = ui.snapshot[pixel] = {0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u};
  }
  ui.matte[15u * width + 56u] = 1u;
  auto selected = make_current_outline_inputs(ui);
  selected.matte.assign(width * height, 0u);
  selected.matte[15u * width + 64u] = 1u;
  selected.production_shared_ui_pin = true;
  std::string error;
  live_v2_warp_fixture_t fixture;
  ASSERT_TRUE(fixture.initialize(error, warp_shader_e::selected_outline)) << error;
  const auto render = [&](const subtitle_inputs_t &subtitle, const current_outline_inputs_t &operands) {
    std::vector<std::byte> output;
    EXPECT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t), width, height,
      source.data(), base, base, state, {}, output, error, field_width, field_height, {}, &subtitle,
      nullptr, true, &operands)) << error;
    return output;
  };
  const auto current = render(ui, selected);
  auto disabled = ui;disabled.constants.enabled = 0u;
  auto baseline = selected;baseline.production_shared_ui_pin = false;
  const auto base_output = render(disabled, baseline);
  ASSERT_NE(current, base_output);
  for (const auto *fault : {"empty", "stale SCT", "inactive SCP", "changed RGB", "stale receipt"}) {
    SCOPED_TRACE(fault);
    auto changed = ui;
    auto operands = selected;
    if (std::string_view(fault) == "empty") operands.matte.assign(width * height, 0u);
    else if (std::string_view(fault) == "stale SCT") changed.identity[20u] ^= 1u;
    else if (std::string_view(fault) == "inactive SCP") changed.condition[2u] = changed.condition[4u] = changed.condition[5u] = 0u;
    else if (std::string_view(fault) == "changed RGB") {
      for (auto &pixel : changed.snapshot) pixel = {0u, 0u, 0u, 0x3c00u};
    } else operands.receipt[63u] = 1u;
    const auto actual = render(changed, operands);
    if (std::string_view(fault) == "stale receipt") {
      EXPECT_EQ(actual, base_output) << "Receipt rejection must retain ordinary Base";
    } else EXPECT_EQ(actual, base_output);
    EXPECT_EQ(render(ui, selected), current) << "A rejected/empty draw must not poison a later current draw";
  }
  // A receipt that mirrors a malformed tuple cannot grant source/plane authority.
  // These were previously exercised through the retired literal pixel program.
  for (UINT fault = 0u; fault < 11u; ++fault) {
    SCOPED_TRACE("independent tuple authority " + std::to_string(fault));
    auto changed = ui;
    switch (fault) {
      case 0u: changed.constants.enabled = 0u; break;
      case 1u: changed.condition[4u] = 0u; break;
      case 2u: changed.condition[2u] = 0u; break;
      case 3u: changed.identity[3u] ^= 1u; break;
      case 4u: changed.constants.analysis_generation_low ^= 1u; break;
      case 5u: changed.identity[20u] ^= 1u; break;
      case 6u: changed.snapshot[15u * width + 64u].a = 0u; break;
      case 7u: changed.identity[22u] ^= 1u; break;
      case 8u: changed.identity[24u] = 1u; break;
      case 9u: changed.condition[5u] = 0x7fc00000u; break;
      default: break;
    }
    auto operands = make_current_outline_inputs(changed);
    operands.matte = selected.matte;
    operands.production_shared_ui_pin = true;
    if (fault == 10u) std::fill(operands.matte.begin(), operands.matte.end(), 0u);
    EXPECT_EQ(render(changed, operands), base_output);
    EXPECT_EQ(render(ui, selected), current);
  }
  for (const UINT word : {0u, 1u, 3u, 4u, 16u, 20u, 22u, 23u, 24u, 25u, 26u, 28u, 62u, 63u}) {
    SCOPED_TRACE("receipt word " + std::to_string(word));
    auto operands = selected;operands.receipt[word] ^= word == 22u ? 2u : 1u;
    EXPECT_EQ(render(ui, operands), base_output);
    EXPECT_EQ(render(ui, selected), current);
  }
  for (const bool zero_token : {false, true}) {
    auto operands = selected;
    if (zero_token) operands.receipt[60u] = operands.receipt[61u] = 0u;
    else operands.receipt[2u] = 1u;
    EXPECT_EQ(render(ui, operands), base_output);
  }
  std::vector<std::byte> absent_receipt;
  ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t), width, height,
    source.data(), base, base, state, {}, absent_receipt, error, field_width, field_height, {}, &ui)) << error;
  EXPECT_EQ(absent_receipt, base_output) << "An absent receipt cannot select the ordinary renderer's legacy subtitle warp";

  auto unauthorized = state;
  unauthorized[models::depth_coordinate_v2::renderer_authorization_bits] ^= 1u;
  live_v2_warp_fixture_t flat;
  ASSERT_TRUE(flat.initialize(error, warp_shader_e::flat_main)) << error;
  std::vector<std::byte> rejected, flat_pixels;
  ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t), width, height,
    source.data(), base, base, unauthorized, {}, rejected, error, field_width, field_height, {}, &ui,
    nullptr, true, &selected)) << error;
  ASSERT_TRUE(flat.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t), width, height,
    source.data(), base, base, unauthorized, {}, flat_pixels, error, field_width, field_height)) << error;
  EXPECT_EQ(rejected, flat_pixels) << "A selected receipt cannot bypass renderer authorization";
}

TEST(HostSbsV2LiveWarpGpuTest, SharedUiPinProductionOwnerRestoresContextAndRejectsMissingInputs) {
  namespace shared = models::host_sbs_shared_ui_warp;
  live_v2_warp_fixture_t fixture;
  std::string error;
  ASSERT_TRUE(fixture.initialize(error, warp_shader_e::selected_outline)) << error;
  ASSERT_TRUE(fixture.production_shared_ui_pin_programs.load(fixture.device.Get(), SUNSHINE_SHADERS_DIR, error)) << error;
  constexpr UINT width = 128u, height = 16u;
  const auto state = make_live_state(true, true);
  const std::vector<float> base(width * height, -1.0f / width);
  auto subtitle = make_subtitle_inputs(width, height, state, 2.5f / width);
  subtitle.matte[8u * width + 64u] = 1u;
  subtitle.snapshot[8u * width + 64u] = {0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u};
  std::vector<rgba16_bits_t> source(width * height, {0x3400u, 0x3400u, 0x3400u, 0x3c00u});
  source[8u * width + 64u] = subtitle.snapshot[8u * width + 64u];
  auto operands = make_current_outline_inputs(subtitle);operands.production_shared_ui_pin = true;

  ComPtr<ID3D11Texture2D> sentinel_texture, sentinel_uav_texture;
  ComPtr<ID3D11ShaderResourceView> sentinel_srv;
  ASSERT_TRUE(fixture.create_immutable_texture_srv(DXGI_FORMAT_R32_FLOAT, width, height, 4u,
    base.data(), sentinel_texture, sentinel_srv));
  D3D11_TEXTURE2D_DESC texture_desc {};
  sentinel_texture->GetDesc(&texture_desc);
  texture_desc.Usage = D3D11_USAGE_DEFAULT;texture_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
  ComPtr<ID3D11UnorderedAccessView> sentinel_uav;
  ASSERT_TRUE(SUCCEEDED(fixture.device->CreateTexture2D(&texture_desc, nullptr, &sentinel_uav_texture)));
  ASSERT_TRUE(SUCCEEDED(fixture.device->CreateUnorderedAccessView(sentinel_uav_texture.Get(), nullptr, &sentinel_uav)));
  D3D11_BUFFER_DESC desc {};desc.ByteWidth = 48u;desc.Usage = D3D11_USAGE_DEFAULT;desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  ComPtr<ID3D11Buffer> sentinel_constants;
  ASSERT_TRUE(SUCCEEDED(fixture.device->CreateBuffer(&desc, nullptr, &sentinel_constants)));
  std::array<ID3D11ShaderResourceView *, 15u> old_srvs;old_srvs.fill(sentinel_srv.Get());
  std::array<ID3D11Buffer *, 7u> old_constants;old_constants.fill(sentinel_constants.Get());
  ID3D11UnorderedAccessView *old_uavs[3] {sentinel_uav.Get(), nullptr, nullptr};
  ID3D11ShaderResourceView *old_pixel[3] {sentinel_srv.Get(), sentinel_srv.Get(), sentinel_srv.Get()};
  auto *old_buffer = sentinel_constants.Get();auto *old_sampler = fixture.sampler.Get();
  auto *old_shader = fixture.production_shared_ui_pin_programs.program().initialize;
  fixture.context->CSSetShaderResources(0u, 15u, old_srvs.data());
  fixture.context->CSSetConstantBuffers(0u, 7u, old_constants.data());
  fixture.context->CSSetUnorderedAccessViews(0u, 3u, old_uavs, nullptr);
  fixture.context->CSSetSamplers(0u, 1u, &old_sampler);fixture.context->CSSetShader(old_shader, nullptr, 0u);
  fixture.context->PSSetShaderResources(12u, 3u, old_pixel);fixture.context->PSSetConstantBuffers(6u, 1u, &old_buffer);
  auto *old_background = sentinel_srv.Get();
  fixture.context->PSSetShaderResources(9u, 1u, &old_background);
  fixture.context->PSSetShaderResources(11u, 1u, &old_background);
  fixture.context->PSSetConstantBuffers(4u, 1u, &old_buffer);
  std::vector<std::byte> output;
  ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t), width, height,
    source.data(), base, base, state, {}, output, error, 0u, 0u, {}, &subtitle, nullptr, true, &operands)) << error;
  ID3D11ShaderResourceView *restored_srvs[15] {}, *restored_pixel[3] {};
  ID3D11Buffer *restored_constants[7] {}, *restored_pixel_constant = nullptr;
  ID3D11UnorderedAccessView *restored_uavs[3] {};
  ComPtr<ID3D11SamplerState> restored_sampler;ComPtr<ID3D11ComputeShader> restored_shader;
  fixture.context->CSGetShaderResources(0u, 15u, restored_srvs);fixture.context->CSGetConstantBuffers(0u, 7u, restored_constants);
  fixture.context->CSGetUnorderedAccessViews(0u, 3u, restored_uavs);fixture.context->CSGetSamplers(0u, 1u, &restored_sampler);
  fixture.context->CSGetShader(&restored_shader, nullptr, nullptr);fixture.context->PSGetShaderResources(12u, 3u, restored_pixel);
  fixture.context->PSGetConstantBuffers(6u, 1u, &restored_pixel_constant);
  for (UINT i = 0u; i < 15u; ++i) {EXPECT_EQ(restored_srvs[i], old_srvs[i]);if(restored_srvs[i])restored_srvs[i]->Release();}
  for (UINT i = 0u; i < 7u; ++i) {EXPECT_EQ(restored_constants[i], old_constants[i]);if(restored_constants[i])restored_constants[i]->Release();}
  for (UINT i = 0u; i < 3u; ++i) {
    EXPECT_EQ(restored_uavs[i], old_uavs[i]);if(restored_uavs[i])restored_uavs[i]->Release();
    EXPECT_EQ(restored_pixel[i], old_pixel[i]);if(restored_pixel[i])restored_pixel[i]->Release();
  }
  EXPECT_EQ(restored_sampler.Get(), old_sampler);EXPECT_EQ(restored_shader.Get(), old_shader);
  EXPECT_EQ(restored_pixel_constant, old_buffer);if(restored_pixel_constant)restored_pixel_constant->Release();
  ComPtr<ID3D11ShaderResourceView> restored_background, restored_proof;
  fixture.context->PSGetShaderResources(9u, 1u, &restored_background);
  fixture.context->PSGetShaderResources(11u, 1u, &restored_proof);
  EXPECT_EQ(restored_background.Get(), old_background);
  EXPECT_EQ(restored_proof.Get(), old_background);
  ComPtr<ID3D11Buffer> restored_auxiliary_constants;
  fixture.context->PSGetConstantBuffers(4u, 1u, &restored_auxiliary_constants);
  EXPECT_EQ(restored_auxiliary_constants.Get(), old_buffer);
  shared::request_t missing;
  EXPECT_FALSE(fixture.production_shared_ui_pin_owner.queue(fixture.context.Get(),
    fixture.production_shared_ui_pin_programs.program(), missing, error));
  EXPECT_FALSE(fixture.production_shared_ui_pin_owner.matches(fixture.context.Get(), missing));
  EXPECT_FALSE(fixture.production_shared_ui_pin_owner.views().field);
}

TEST(HostSbsV2LiveWarpGpuTest, SharedUiPinProductionRejectsInvalidRowExtentAndDiagnosticDevice) {
  namespace shared = models::host_sbs_shared_ui_warp;
  live_v2_warp_fixture_t fixture;
  std::string error;
  ASSERT_TRUE(fixture.initialize(error, warp_shader_e::selected_outline)) << error;
  shared::resources_t owner;
  EXPECT_FALSE(owner.ensure(fixture.device.Get(), {8u, 16384u, 0u, 16384u}, error));
  EXPECT_EQ(error, "Shared UI warp native source domain unavailable");
  EXPECT_FALSE(owner.views().field);

  shared::programs_t programs;
  ASSERT_TRUE(programs.load(fixture.device.Get(), SUNSHINE_SHADERS_DIR, error)) << error;
  ASSERT_TRUE(programs.load_diagnostics(fixture.device.Get(), SUNSHINE_SHADERS_DIR, error)) << error;
  EXPECT_FALSE(programs.load_diagnostics(nullptr, SUNSHINE_SHADERS_DIR, error));
  EXPECT_FALSE(error.empty());
  ComPtr<ID3D11Device> foreign_device;
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, nullptr, 0u,
    D3D11_SDK_VERSION, foreign_device.GetAddressOf(), nullptr, nullptr)));
  EXPECT_FALSE(programs.load_diagnostics(foreign_device.Get(), SUNSHINE_SHADERS_DIR, error));
  error = "previous diagnostic failure";
  ASSERT_TRUE(programs.load_diagnostics(fixture.device.Get(), SUNSHINE_SHADERS_DIR, error)) << error;
  EXPECT_TRUE(error.empty());
}

TEST(HostSbsV2LiveWarpGpuTest, SharedUiPinQueuedProductionRequiresCurrentGuardAtPrepareAndDraw) {
  namespace guard = models::research_queued_render_guard_v1;
  namespace v2 = models::depth_coordinate_v2;
  constexpr UINT width = 128u, height = 16u;
  const auto state = make_live_state(true, true);
  const std::uint64_t clock = std::uint64_t(state[v2::gain_last_observation_low]) |
                              (std::uint64_t(state[v2::gain_last_observation_high]) << 32u);
  const std::uint64_t token = 71ull | (9ull << 32u);
  const auto expected = guard::make_expected(token, 1u, clock);
  auto ui = make_subtitle_inputs(width, height, state, 2.5f / width);
  ui.matte[8u * width + 64u] = 1u;
  ui.snapshot[8u * width + 64u] = {0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u};
  auto selected = make_current_outline_inputs(ui);selected.production_shared_ui_pin = true;
  std::vector<rgba16_bits_t> source(width * height, {0x3400u, 0x3400u, 0x3400u, 0x3c00u});
  source[8u * width + 64u] = ui.snapshot[8u * width + 64u];
  const std::vector<float> base(width * height, -1.25f / width);
  std::string error;
  live_v2_warp_fixture_t fixture, flat;
  ASSERT_TRUE(fixture.initialize(error, warp_shader_e::selected_outline)) << error;
  ASSERT_TRUE(flat.initialize(error, warp_shader_e::flat_main)) << error;
  const auto packet_for = [&](const guard::expected_t &e) {
    std::vector<std::uint32_t> packet(64u);
    packet[0] = 1u;packet[1] = 1u ^ 0xd1ec15a5u ^ 0x52434f4fu;
    packet[2] = e.root[0];packet[3] = e.root[1];
    packet[4] = e.root[0] ^ 0xa3756c91u;packet[5] = e.root[1] ^ 0x5c8a936eu;
    packet[6] = 0x47524243u;packet[7] = 0x52434f4fu;
    std::copy_n(packet.begin() + 2u, 4u, packet.begin() + 8u);
    packet[12] = 0x54535152u;packet[13] = e.root[2];packet[14] = e.root[3];
    packet[20] = packet[21] = packet[22] = packet[26] = 1u;
    packet[24] = (width + 15u) / 16u;packet[25] = (height + 15u) / 16u;
    return packet;
  };
  const auto good = packet_for(expected);
  const auto configure = [&](const guard::expected_t &owner, const std::vector<std::uint32_t> &packet) {
    return fixture.configure_guard_operands(&owner, &packet, error);
  };
  const auto render = [&](live_v2_warp_fixture_t &renderer, const current_outline_inputs_t *operands) {
    std::vector<std::byte> pixels;
    EXPECT_TRUE(renderer.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t), width, height,
      source.data(), base, base, state, {}, pixels, error, 0u, 0u, {}, &ui, nullptr, true, operands)) << error;
    return pixels;
  };
  const auto completed = render(fixture, &selected), identity = render(flat, nullptr);
  ASSERT_NE(completed, identity);
  fixture.use_recorder_guard_scope = true;
  selected.production_shared_ui_pin_queued = true;
  ASSERT_TRUE(configure(expected, good)) << error;
  EXPECT_EQ(render(fixture, &selected), completed);
  auto bad = good;bad[6u] = 0u;
  ASSERT_TRUE(configure(expected, bad)) << error;
  const auto bad_draw_packet = fixture.private_guard_srv;
  EXPECT_EQ(render(fixture, &selected), identity) << "A rejected prepare guard must produce current flat pixels";
  ASSERT_TRUE(configure(expected, good)) << error;
  selected.shared_ui_pin_draw_guard_override = bad_draw_packet.Get();
  EXPECT_EQ(render(fixture, &selected), identity) << "Revocation after preparation must reject the issued pin field";
  selected.shared_ui_pin_draw_guard_override = nullptr;
  auto stale_clock = expected;stale_clock.observation[0u] ^= 1u;
  ASSERT_TRUE(configure(stale_clock, good)) << error;
  EXPECT_EQ(render(fixture, &selected), identity);
  struct malformed_guard_t {
    std::string name;
    guard::expected_t owner;
    std::vector<std::uint32_t> packet;
  };
  std::vector<malformed_guard_t> malformed;
  const auto add = [&](const std::string &name) {
    malformed.push_back({name, expected, good});return &malformed.back();
  };
  add("raw packet wrong size")->packet.resize(63u);
  add("bad enable")->owner.control[0u] = 2u;
  add("bad tag")->owner.control[1u] ^= 1u;
  add("bad schema")->owner.control[2u] = 2u;
  add("reserved control")->owner.control[3u] = 1u;
  add("zero token")->owner.root[0u] = 0u;malformed.back().owner.root[1u] = 0u;
  add("invalid work")->owner.root[2u] = 4u;
  add("bad work cookie")->owner.root[3u] ^= 1u;
  add("zero clock")->owner.observation = {0u, 0u, 0u, 0u};
  add("reserved observation")->owner.observation[2u] = 1u;
  for (const UINT word : {8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u})
    add("request word " + std::to_string(word))->packet[word] ^= 1u;
  add("mixed request work")->packet[13u] = 3u;
  malformed.back().packet[14u] = 3u ^ 0x6f435257u;
  add("invalid request work")->packet[13u] = 4u;
  malformed.back().packet[14u] = 4u ^ 0x6f435257u;
  add("pre-child marker")->packet[6u] = 0u;
  add("proposal marker")->packet[6u] = 0x504f5250u;
  add("reuse is not inference")->packet[0u] = 0u;
  malformed.back().packet[1u] = 0xd1ec15a5u;malformed.back().packet[7u] = 0u;
  for (const UINT word : {1u, 2u, 3u, 4u, 5u, 7u})
    add("receipt word " + std::to_string(word))->packet[word] ^= 1u;
  add("OCR marker with wrong work")->owner = guard::make_expected(token, 2u, clock);
  malformed.back().packet = packet_for(malformed.back().owner);
  for (const UINT word : {20u, 21u, 22u, 23u, 24u, 25u, 26u, 27u})
    add("finalizer word " + std::to_string(word))->packet[word] ^= 1u;
  for (const auto &fault : malformed) {
    SCOPED_TRACE(fault.name);
    ASSERT_TRUE(configure(fault.owner, fault.packet)) << error;
    EXPECT_EQ(render(fixture, &selected), identity);
    EXPECT_TRUE(fixture.private_guard_slots_clear());
  }
  // Missing guard resources are rejected before queueing any shared preparation.
  for (const bool missing_constants : {false, true}) {
    ASSERT_TRUE(configure(expected, good)) << error;
    if (missing_constants) fixture.private_guard_cb.Reset();else fixture.private_guard_srv.Reset();
    std::vector<std::byte> pixels;
    EXPECT_FALSE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t), width, height,
      source.data(), base, base, state, {}, pixels, error, 0u, 0u, {}, &ui, nullptr, true, &selected));
    EXPECT_EQ(error, "Shared UI warp program/context/draw operands unavailable");
    EXPECT_TRUE(fixture.private_guard_slots_clear());
  }
  ASSERT_TRUE(configure(expected, good)) << error;
  auto old_state_clock = state;
  old_state_clock[v2::gain_last_observation_low] ^= 1u;
  old_state_clock[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(old_state_clock);
  std::vector<std::byte> old_clock_pixels;
  ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t), width, height,
    source.data(), base, base, old_state_clock, {}, old_clock_pixels, error, 0u, 0u, {}, &ui,
    nullptr, true, &selected)) << error;
  EXPECT_EQ(old_clock_pixels, identity);

  ComPtr<ID3D11Predicate> predicate;
  D3D11_QUERY_DESC predicate_desc {D3D11_QUERY_OCCLUSION_PREDICATE, 0u};
  ASSERT_TRUE(SUCCEEDED(fixture.device->CreatePredicate(&predicate_desc, &predicate)));
  fixture.context->SetPredication(predicate.Get(), TRUE);
  std::vector<std::byte> predicated_pixels;
  EXPECT_FALSE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT, sizeof(rgba16_bits_t), width, height,
    source.data(), base, base, state, {}, predicated_pixels, error, 0u, 0u, {}, &ui,
    nullptr, true, &selected));
  EXPECT_EQ(error, "Shared UI warp requires an unpredicated draw");
  ComPtr<ID3D11Predicate> restored_predicate;BOOL restored_predicate_value = FALSE;
  fixture.context->GetPredication(&restored_predicate, &restored_predicate_value);
  EXPECT_EQ(restored_predicate.Get(), predicate.Get());EXPECT_EQ(restored_predicate_value, TRUE);
  fixture.context->SetPredication(nullptr, FALSE);
  ASSERT_TRUE(configure(expected, good)) << error;
  EXPECT_EQ(render(fixture, &selected), completed) << "Rejected guards must not poison a later current queued draw";
  EXPECT_TRUE(fixture.private_guard_slots_clear());
}

// Opt-in external two-sided32 selector experiment, independent of production qualification.
TEST(HostSbsV2LiveWarpGpuTest, PrivateNativeMatteTwoSidedSelectorReplay) {
  const auto *cases_env = std::getenv("SUNSHINE_TEST_PRIVATE_SELECTOR_REPLAY_CASES");
  if (!cases_env || !*cases_env) GTEST_SKIP() << "No saved glyph tuple specification selected";
  const auto *output_env = std::getenv("SUNSHINE_TEST_PRIVATE_SELECTOR_REPLAY_OUTPUT");
  ASSERT_TRUE(output_env && *output_env) << "Saved replay requires an explicit output directory";
  namespace v2 = models::depth_coordinate_v2;
  namespace cache = models::host_sbs_shader_cache;
  namespace fs = std::filesystem;
  using json = nlohmann::json;
  try {
    const auto require = [](bool valid, const std::string &message) {
      if (!valid) throw std::runtime_error(message);
    };
    const auto read_bytes = [](const fs::path &path) {
      std::ifstream input(path, std::ios::binary | std::ios::ate);
      if (!input || input.tellg() < 0) throw std::runtime_error("Cannot read replay input: " + path.string());
      std::vector<char> bytes(static_cast<std::size_t>(input.tellg()));
      input.seekg(0);
      if (!bytes.empty()) input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      if (!input) throw std::runtime_error("Incomplete replay input: " + path.string());
      return bytes;
    };
    const auto sha256 = [](const void *data, std::size_t size) {
      const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), size));
      constexpr char digits[] = "0123456789abcdef";
      std::string hex;
      for (const auto byte : digest) {
        hex.push_back(digits[byte >> 4u]);
        hex.push_back(digits[byte & 15u]);
      }
      return hex;
    };
    const auto write_bytes = [](const fs::path &path, const void *data, std::size_t size) {
      std::ofstream output(path, std::ios::binary);
      output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
      if (!output) throw std::runtime_error("Cannot write replay output: " + path.string());
    };
    const fs::path cases_path(cases_env), output_root(output_env);
    require(!fs::exists(output_root) || (fs::is_directory(output_root) && fs::is_empty(output_root)),
            "Replay output must be a new or empty directory");
    const auto specification_bytes = read_bytes(cases_path);
    const auto specification = json::parse(specification_bytes.begin(), specification_bytes.end());
    require(specification.at("schema") == 1 && (specification.at("driver") == "hardware" || specification.at("driver") == "warp"),
            "Private layered replay requires schema 1 and an explicit hardware/warp driver");
    const auto selector_mode_kind = specification.value("selector_mode_kind", std::string {"blend"});
    require(selector_mode_kind == "blend" || selector_mode_kind == "nearest", "Unknown private selector semantic mode");
    const bool nearest_selector = selector_mode_kind == "nearest";
    const auto &cases = specification.at("cases");
    require(cases.is_array() && !cases.empty() && cases.size() <= 8u, "Private replay requires 1..8 matched cases");
    std::string error;
    const bool hardware = specification.at("driver") == "hardware";
    live_v2_warp_fixture_t color, mapping, selector;
    ASSERT_TRUE(color.initialize(error, warp_shader_e::live_main, false, hardware)) << error;
    ASSERT_TRUE(mapping.initialize(error, warp_shader_e::live_mapping, false, hardware)) << error;
    ASSERT_TRUE(selector.initialize(error, warp_shader_e::live_main, false, hardware)) << error;
    const auto adapter = color.adapter_identity();
    require(mapping.adapter_identity() == adapter && selector.adapter_identity() == adapter,
            "Reference routes selected different adapters");
    std::array<wchar_t, 32768u> executable_path {};
    const auto executable_path_size = GetModuleFileNameW(nullptr, executable_path.data(), static_cast<DWORD>(executable_path.size()));
    require(executable_path_size > 0u && executable_path_size < executable_path.size(), "Cannot locate replay executable");
    const auto executable_sha256 = [&]() {
      const auto bytes = read_bytes(fs::path(executable_path.data()));
      return sha256(bytes.data(), bytes.size());
    }();
    const fs::path fixture_path = fs::path(SUNSHINE_SOURCE_DIR) / "tests/unit/test_host_sbs_v2_live_warp.cpp";
    const auto fixture_bytes = read_bytes(fixture_path);
    require(sha256(fixture_bytes.data(), fixture_bytes.size()) == specification.at("expected_fixture_source_sha256").get<std::string>(),
      "Fixture source is not the exact independently frozen proposal");
    fs::create_directories(output_root);
    json report {{"schema", 1}, {"driver", specification.at("driver")}, {"adapter", adapter},
      {"test_executable_path", fs::path(executable_path.data()).string()},
      {"test_executable_sha256", executable_sha256},
      {"fixture_source_path", fixture_path.string()},
      {"fixture_source_sha256", sha256(fixture_bytes.data(), fixture_bytes.size())},
      {"input_specification_path", cases_path.string()},
      {"input_specification_sha256", sha256(specification_bytes.data(), specification_bytes.size())},
      {"coordinate_contract_schema", v2::contract_schema},
      {"coordinate_contract_sha256", v2::contract_canonical_sha256},
      {"live_renderer_source_closure_sha256", cache::parallax_v2_live_renderer_source_closure_sha256},
      {"live_diagnostic_source_closure_sha256", cache::parallax_v2_live_diagnostic_source_closure_sha256},
      {"scope", "PRIVATE saved-input two-sided32 selector experiment; canonical old/new M routes unchanged; authored GPU reference evaluation only"},
      {"production_tuple_claim_for_candidate", false},
      {"private_layer_shader_root", specification.at("private_layer_shader_root")},
      {"private_layer_source_closure_sha256", specification.at("private_layer_source_closure_sha256")},
      {"private_selector_shader_root", specification.at("private_selector_shader_root")},
      {"private_selector_source_closure_sha256", specification.at("private_selector_source_closure_sha256")},
      {"selector_mode_kind", selector_mode_kind},
      {"selector_rule", nearest_selector ?
        "UI-hit preserved; Base-only-hit bilateral clear footprints within32 then nearer CURRENT sample, LEFT ties; missing side oldUI fallback" :
        "UI-hit preserved; Base-only-hit nearest bilateral clear footprints within32 then current-colour lerp; missing side oldUI fallback"},
      {"selector_is_private_not_production", true},
      {"candidate_single_source_U_map_exists", nearest_selector},
      {"cases", json::array()}};
    for (std::size_t case_index = 0u; case_index < cases.size(); ++case_index) {
      const auto &entry = cases[case_index];
      const auto name = entry.at("name").get<std::string>();
      SCOPED_TRACE(name);
      const auto source_shape = entry.at("source").get<std::array<UINT, 2u>>();
      const auto field_shape = entry.at("field").get<std::array<UINT, 2u>>();
      const auto [width, height] = source_shape;
      const auto [field_width, field_height] = field_shape;
      require(width > 0u && width <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION / 2u &&
              height > 0u && height <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
              field_width > 0u && field_width <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
              field_height > 0u && field_height <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION,
              "Invalid source or field shape");
      json input_bindings = json::object();
      const auto operand = [&](const char *key, std::size_t expected_size) {
        const auto &binding = entry.at(key);
        fs::path path(binding.at("path").get<std::string>());
        if (path.is_relative()) path = cases_path.parent_path() / path;
        const auto bytes = read_bytes(path);
        require(bytes.size() == expected_size, std::string(key) + " byte count/shape mismatch");
        const auto hash = sha256(bytes.data(), bytes.size());
        require(hash == binding.at("sha256").get<std::string>(), std::string(key) + " SHA-256 mismatch");
        input_bindings[key] = {{"path", path.string()}, {"sha256", hash}, {"byte_count", bytes.size()}};
        return bytes;
      };
      const auto load_object = [&]<typename T>(const char *key, T &value) {
        const auto bytes = operand(key, sizeof(value));
        std::memcpy(&value, bytes.data(), sizeof(value));
      };
      const auto load_vector = [&]<typename T>(const char *key, std::size_t count) {
        const auto bytes = operand(key, count * sizeof(T));
        std::vector<T> values(count);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
      };
      models::depth_coordinate_v2::state_words_t state {};
      host_sbs_v2_geometry_t geometry;
      subtitle_inputs_t subtitle;
      load_object("state", state);
      load_object("geometry", geometry);
      load_object("subtitle_constants", subtitle.constants);
      load_object("condition", subtitle.condition);
      load_object("identity", subtitle.identity);
      const auto source = load_vector.template operator()<bgra8_t>("source_bgra8", std::size_t(width) * height);
      const auto base = load_vector.template operator()<float>("base", std::size_t(field_width) * field_height);
      const auto &id = subtitle.identity;
      const auto &scp = subtitle.condition;
      const auto &b3 = subtitle.constants;
      require(v2::parallax_state_words_are_authenticated(state, entry.at("raw_coordinate_scale").get<float>(), v2::adaptive_policy_id),
              "Saved ParallaxState is not authenticated");
      require(state[v2::renderer_authorization_bits] == v2::contract_tag, "Saved frame has no authorized Base");
      require(std::all_of(base.begin(), base.end(), [](float value) {
        return std::isfinite(value) && std::abs(value) <= v2::direct_container_limit;
      }), "Saved Base has nonfinite or out-of-container values");
      require(b3.enabled == 1u && b3.color_mode == 0u && b3.source_width > 0u && b3.source_height > 0u &&
              b3.source_width <= v2::subtitle_contour_source_max_extent &&
              b3.source_height <= v2::subtitle_contour_source_max_extent &&
              b3.source_left <= width && b3.source_top <= height &&
              b3.source_width <= width - b3.source_left && b3.source_height <= height - b3.source_top,
              "Saved b3 must bind an exact SDR8 analysis source inside full color");
      require(std::isfinite(geometry.content_scale_x) && geometry.content_scale_x > 0.0f && geometry.content_scale_x <= 1.0f &&
              std::isfinite(geometry.content_scale_y) && geometry.content_scale_y > 0.0f && geometry.content_scale_y <= 1.0f &&
              geometry.reserved == 0.0f && (geometry.video_roi_active == 0.0f || geometry.video_roi_active == 1.0f),
              "Invalid saved b2 geometry");
      require(id[0] == v2::subtitle_contour_schema && id[1] == v2::subtitle_contour_history_tag &&
              id[2] <= 1u && id[23] == 1u && id[22] == v2::subtitle_contour_recipe &&
              (id[4] != 0u || id[5] != 0u) && id[6] == b3.analysis_generation_low && id[7] == b3.analysis_generation_high &&
              id[8] == b3.source_width && id[9] == b3.source_height && id[11] > 0u &&
              id[10] <= id[9] && id[11] <= id[9] - id[10] &&
              id[12] == field_width && id[13] == field_height &&
              id[16] < id[18] && id[17] < id[19] && id[18] <= field_width && id[19] <= field_height &&
              id[3] == state[v2::confirmed_cut_count] &&
              (id[20] != 0u || id[21] != 0u) && id[20] == state[v2::gain_last_observation_low] &&
              id[21] == state[v2::gain_last_observation_high] &&
              std::all_of(id.begin() + v2::subtitle_contour_history_reserved_offset, id.end(), [](auto value) { return value == 0u; }),
              "Saved SCT owner/observation/cut/generation/shape binding is invalid");
      subtitle.width = id[8];
      subtitle.height = id[11];
      subtitle.matte = load_vector.template operator()<std::uint32_t>("matte", std::size_t(subtitle.width) * subtitle.height);
      subtitle.snapshot = load_vector.template operator()<rgba16_bits_t>("snapshot", subtitle.matte.size());
      require(std::all_of(subtitle.matte.begin(), subtitle.matte.end(), [](auto value) { return value <= 1u; }),
              "Saved matte is not literal binary R32_UINT");
      const float target = std::bit_cast<float>(scp[5]);
      const bool active = scp[2] > 0u;
      require(scp[0] == v2::subtitle_condition_param_schema && scp[1] == v2::subtitle_condition_param_tag &&
              std::isfinite(target) && target >= 0.0f && target <= v2::direct_container_limit &&
              scp[2] <= v2::subtitle_plane_cover_capacity && (scp[3] & ~((1u << scp[2]) - 1u)) == 0u &&
              (active ? scp[4] == v2::subtitle_condition_param_fully_active_value && id[2] == 1u :
                        scp[3] == 0u && scp[4] == 0u && scp[5] == 0u && id[2] == 0u),
              "Saved SCP activity/target disagrees with SCT");
      require(active || std::all_of(subtitle.matte.begin(), subtitle.matte.end(), [](auto value) { return value == 0u; }),
              "Inactive saved tuple retains a positive glyph matte");
      if (geometry.video_roi_active == 1.0f) {
        require(geometry.tensor_content_left == id[16] && geometry.tensor_content_top == id[17] &&
                geometry.tensor_content_right == id[18] && geometry.tensor_content_bottom == id[19],
                "Saved ROI b2 content differs from SCT");
        const std::array actual_roi {geometry.video_roi_left, geometry.video_roi_top, geometry.video_roi_right, geometry.video_roi_bottom};
        const std::array expected_roi {float(b3.source_left) / width, float(b3.source_top) / height,
          float(b3.source_left + b3.source_width) / width, float(b3.source_top + b3.source_height) / height};
        for (std::size_t i = 0u; i < actual_roi.size(); ++i) {
          require(std::isfinite(actual_roi[i]) && actual_roi[i] >= 0.0f && actual_roi[i] <= 1.0f &&
                  std::abs(actual_roi[i] - expected_roi[i]) * (i % 2u == 0u ? width : height) <= 0.25f,
                  "Saved ROI b2 and integer b3 source rectangles differ");
        }
      } else {
        require(b3.source_left == 0u && b3.source_top == 0u && b3.source_width == width && b3.source_height == height,
                "Full-source saved tuple has a conflicting b3 analysis source");
      }
      const auto candidate_identity_bytes = operand("candidate_native_identity", entry.at("candidate_native_identity").at("byte_count").get<std::size_t>());
      require(!candidate_identity_bytes.empty(), "Private mask needs an actual native-run identity binding");
      std::array<std::uint32_t, 24u> candidate_constants {};
      load_object("candidate_constants", candidate_constants);
      require(candidate_constants[0] == id[8] && candidate_constants[1] == id[9] &&
              candidate_constants[2] == id[10] && candidate_constants[3] == id[11] &&
              candidate_constants[4] == id[12] && candidate_constants[5] == id[13] &&
              candidate_constants[6] == id[14] && candidate_constants[7] == id[15] &&
              std::equal(candidate_constants.begin() + 8u, candidate_constants.begin() + 12u, id.begin() + 16u) &&
              candidate_constants[12] == id[4] && candidate_constants[13] == id[5] &&
              candidate_constants[14] == id[6] && candidate_constants[15] == id[7] &&
              candidate_constants[16] == id[20] && candidate_constants[17] == id[21] &&
              candidate_constants[19] == 0u && candidate_constants[22] == 0u && candidate_constants[23] == 0u,
              "Private native mask CB differs from the saved source/crop/field/content/frame/generation/clock context");
      auto candidate_subtitle = subtitle;
      candidate_subtitle.matte = load_vector.template operator()<std::uint32_t>("candidate_matte", subtitle.matte.size());
      require(std::all_of(candidate_subtitle.matte.begin(), candidate_subtitle.matte.end(), [](auto value) { return value <= 1u; }),
              "Private candidate matte is not literal binary R32_UINT");
      require(active || std::all_of(candidate_subtitle.matte.begin(), candidate_subtitle.matte.end(), [](auto value) { return value == 0u; }),
              "Inactive frame cannot publish a positive private candidate matte");
      const auto background = load_vector.template operator()<bgra8_t>("background_bgra8", std::size_t(width) * height);
      const auto foreground = load_vector.template operator()<std::array<std::uint8_t, 4u>>("foreground_rgba8", background.size());
      const auto effective_body_alpha = load_vector.template operator()<std::uint8_t>("effective_body_alpha_u8", background.size());
      require(std::all_of(background.begin(), background.end(), [](const auto &value) { return value.a == 255u; }) &&
              std::all_of(source.begin(), source.end(), [](const auto &value) { return value.a == 255u; }),
              "Private SDR reference currently requires an opaque authored background and source");
      const auto case_dir = output_root / ("case_" + std::to_string(case_index));
      fs::create_directory(case_dir);
      json result {{"name", name}, {"source", source_shape}, {"field", field_shape},
        {"packed", {width * 2u, height}}, {"input_bindings", input_bindings},
        {"provenance", entry.at("provenance")}, {"ui_target_bits", scp[5]},
        {"state_uploaded_without_modification", true}, {"geometry_uploaded_without_modification", true},
        {"recipe27_identity_is_context_only_for_candidate", true},
        {"candidate_is_canonical_recipe27_matte", false},
        {"candidate_matte_transport", "literal previously GPU-produced native mask; no CPU inferred pixels"},
        {"routes", json::object()}, {"reference", json::object()}};
      const auto save = [&](const std::string &filename, const std::vector<std::byte> &bytes) {
        const auto path = case_dir / filename;
        write_bytes(path, bytes.data(), bytes.size());
        return json {{"path", path.string()}, {"sha256", sha256(bytes.data(), bytes.size())}, {"byte_count", bytes.size()}};
      };
      auto disabled_subtitle = subtitle;
      disabled_subtitle.constants.enabled = 0u;
      for (const auto route : {"base_composite", "old_matte", "candidate_matte"}) {
        const bool candidate_route = std::string_view(route) == "candidate_matte";
        const bool old_route = std::string_view(route) == "old_matte";
        const auto *render_subtitle = candidate_route ? &candidate_subtitle : old_route ? &subtitle : &disabled_subtitle;
        for (const auto kind : {"map", "color"}) {
          const bool is_map = std::string_view(kind) == "map";
          auto &fixture = is_map ? mapping : color;
          std::vector<std::byte> output;
          ASSERT_TRUE(fixture.render(DXGI_FORMAT_B8G8R8A8_UNORM, sizeof(bgra8_t), width, height,
            source.data(), base, base, state, {}, output, error, field_width, field_height,
            geometry, render_subtitle)) << route << '/' << kind << ": " << error;
          const auto filename = std::string(route) + "_" + kind + (is_map ? ".f32" : ".bgra8");
          result["routes"][route][kind] = save(filename, output);
          const std::string expected_key = std::string("expected_selected_") + kind;
          if (old_route && entry.contains(expected_key)) {
            const auto expected = operand(expected_key.c_str(), output.size());
            const auto expected_hash = sha256(expected.data(), expected.size());
            result["routes"][route][kind]["expected_bitexact"] = sha256(output.data(), output.size()) == expected_hash;
            EXPECT_EQ(sha256(output.data(), output.size()), expected_hash) << "Baseline replay differs from actual harness output";
          }
          if (candidate_route) {
            const std::string candidate_key = std::string("expected_candidate_selected_") + kind;
            const auto expected_candidate = operand(candidate_key.c_str(), output.size());
            const auto candidate_expected_hash = sha256(expected_candidate.data(), expected_candidate.size());
            result["routes"][route][kind]["prior_hardware_expected_bitexact"] =
              sha256(output.data(), output.size()) == candidate_expected_hash;
            EXPECT_EQ(sha256(output.data(), output.size()), candidate_expected_hash)
              << "Unchanged M5 route differs from the prior authenticated hardware replay";
          }

        }
      }
      std::vector<std::byte> selected_colour, repeated_selected_colour;
      std::array<std::vector<std::byte>, 2u> selected_diagnostics;
      ASSERT_TRUE(selector.render_private_selector(DXGI_FORMAT_B8G8R8A8_UNORM, sizeof(bgra8_t), width, height,
        source.data(), base, base, state, {}, selected_colour, error,
        specification.at("private_selector_shader_root").get<std::string>(),
        specification.at("private_selector_source_closure_sha256").get<std::string>(), repeated_selected_colour, selected_diagnostics,
        field_width, field_height, geometry, &candidate_subtitle)) << error;
      require(selected_diagnostics[0].size() == std::size_t(width) * 2u * height * 16u &&
              selected_diagnostics[1].size() == selected_diagnostics[0].size(), "Private selector diagnostic extent mismatch");
      // Diagnostic semantics only: renderer calls, operands and numerical shader branches are unchanged.
      for (std::size_t pixel = 0u; pixel < std::size_t(width) * 2u * height; ++pixel) {
        std::array<float, 4u> selection {};
        std::memcpy(selection.data(), selected_diagnostics[1].data() + pixel * sizeof(selection), sizeof(selection));
        require(std::all_of(selection.begin(), selection.end(), [](float x) { return std::isfinite(x); }), "Nonfinite selector diagnostics");
        require(selection[1] >= 0.0f && selection[1] <= 3.0f && std::floor(selection[1]) == selection[1], "Malformed selector mode");
        if (selection[1] == 2.0f) {
          require(selection[2] >= 1.0f && selection[2] <= 32.0f && std::floor(selection[2]) == selection[2] &&
                  selection[3] >= 1.0f && selection[3] <= 32.0f && std::floor(selection[3]) == selection[3], "Malformed bilateral distance");
          if (nearest_selector) {
            require(selection[0] == (selection[2] <= selection[3] ? 0.0f : 1.0f), "Nearest endpoint weight/LEFT tie differs from declared rule");
          } else {
            require(selection[0] > 0.0f && selection[0] < 1.0f, "Blend mode requires strict interior weight");
          }
        } else {
          require(selection[0] == 0.0f && selection[2] == 0.0f && selection[3] == 0.0f, "Single mode contains bilateral diagnostics");
        }
      }
      result["routes"]["two_sided32"] = {
        {"color", save("two_sided32_color.bgra8", selected_colour)},
        {"color_repeat", save("two_sided32_color_repeat.bgra8", repeated_selected_colour)},
        {"two_diagnostic_draw_colours_byte_exact", repeated_selected_colour == selected_colour},
        {"sample_uvs", save("two_sided32_sample_uvs.rgba32f", selected_diagnostics[0])},
        {"selection", save("two_sided32_selection.rgba32f", selected_diagnostics[1])},
        {"single_source_map", nearest_selector},
        {"single_source_map_exported", false},
        {"sample_uvs_layout", "leftU,leftV,rightU,rightV"},
        {"selection_layout", "weightRight,mode,leftDistance,rightDistance"},
        {"mode_values", {{"0", "Base"}, {"1", "UI footprint"}, {"2", nearest_selector ? "nearest clear endpoint; LEFT ties" : "two-sided blend"}, {"3", "missing-side UI fallback"}}}
      };
      live_v2_warp_fixture_t::private_layer_output_t layered;
      ASSERT_TRUE(color.render_private_layer_reference(width, height, background, foreground, effective_body_alpha, base, state,
        field_width, field_height, geometry, subtitle,
        specification.at("private_layer_shader_root").get<std::string>(),
        specification.at("private_layer_source_closure_sha256").get<std::string>(), layered, error)) << error;
      require(layered.reconstructed_rgba8.size() == source.size() * 4u, "Reconstruction shape differs from full source");
      std::array<std::uint64_t, 256u> identity_max_error_histogram {};
      std::uint32_t maximum_identity_rgb_error = 0u;
      for (std::size_t pixel = 0u; pixel < source.size(); ++pixel) {
        const std::array<std::uint8_t, 3u> expected {source[pixel].r, source[pixel].g, source[pixel].b};
        std::uint32_t maximum = 0u;
        for (std::size_t channel = 0u; channel < 3u; ++channel) {
          const auto actual = std::to_integer<unsigned int>(layered.reconstructed_rgba8[pixel * 4u + channel]);
          maximum = std::max(maximum, static_cast<std::uint32_t>(std::abs(int(actual) - int(expected[channel]))));
        }
        ++identity_max_error_histogram[maximum];
        maximum_identity_rgb_error = std::max(maximum_identity_rgb_error, maximum);
      }
      result["reference"] = {
        {"reconstructed_source", save("reference_reconstructed_source.rgba8", layered.reconstructed_rgba8)},
        {"base_background", save("reference_base_background.bgra8", layered.base_background_bgra8)},
        {"layered_color", save("reference_layered.rgba8", layered.reference_rgba8)},
        {"ui_foreground_premultiplied", save("reference_ui_foreground.rgba32f", layered.shifted_foreground_rgba32f)},
        {"ui_effective_body_alpha", save("reference_ui_effective_body_alpha.f32", layered.shifted_body_alpha_f32)},
        {"identity_max_rgb8_error", maximum_identity_rgb_error},
        {"identity_rgb8_max_error_histogram", identity_max_error_histogram},
        {"predeclared_identity_tolerance_rgb8", 1u},
        {"authored_layer_identity_pass", maximum_identity_rgb_error <= 1u},
        {"body_and_optional_black_outline_metrics_are_separate", true},
        {"color_domain", "SDR UNORM numeric RGB, matching production SourceColor; not linear-light conversion"},
        {"scope", "Production Base warp of authored background plus premultiplied authored RGBA sampled at current UI plane, composited entirely on GPU"}};
      result["input_bindings"] = input_bindings;
      const auto result_text = result.dump(2);
      write_bytes(case_dir / "metadata.json", result_text.data(), result_text.size());
      report["cases"].push_back(std::move(result));
      // A failed plate/source reconstruction is evidence, never permission to
      // reinterpret the reference or silently repair the supplied author layers.
      EXPECT_LE(maximum_identity_rgb_error, 1u) << "Authored layers do not reconstruct the actual current source under the predeclared GPU compositor";
    }
    const auto report_text = report.dump(2);
    write_bytes(output_root / "report.json", report_text.data(), report_text.size());
  } catch (const std::exception &failure) {
    FAIL() << failure.what();
  }
}

#endif  // _WIN32
