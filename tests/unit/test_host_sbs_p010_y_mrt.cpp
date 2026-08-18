/**
 * @file tests/unit/test_host_sbs_p010_y_mrt.cpp
 * @brief Exact D3D11-WARP parity checks for the optional Host SBS HDR luma MRT.
 */
#include "../tests_common.h"

#ifdef _WIN32

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <src/depth_coordinate_v2.h>
#include <src/host_sbs_shader_cache.h>
#include <src/host_sbs_v2_geometry.h>
#include <src/video_colorspace.h>

namespace {
  using Microsoft::WRL::ComPtr;
  using models::host_sbs_v2_geometry_t;

  struct rgba16_bits_t {
    std::uint16_t r;
    std::uint16_t g;
    std::uint16_t b;
    std::uint16_t a;

    bool operator==(const rgba16_bits_t &) const = default;
  };

  struct rotate_constants_t {
    std::int32_t rotate_texture_steps = 0;
    std::array<std::int32_t, 3> padding {};
  };

  static_assert(sizeof(rotate_constants_t) == 16u);

  std::string hresult_string(const HRESULT status) {
    std::ostringstream text;
    text << "0x" << std::hex << std::uppercase
         << static_cast<std::uint32_t>(status);
    return text.str();
  }

  models::depth_coordinate_v2::state_words_t make_authenticated_state() {
    namespace v2 = models::depth_coordinate_v2;
    auto state = v2::state_initial_words;
    state[v2::center] = std::bit_cast<std::uint32_t>(0.25f);
    state[v2::inverse_scale] = std::bit_cast<std::uint32_t>(
      1.0f / v2::model_calibrations.front().raw_coordinate_scale
    );
    state[v2::convergence_curve] =
      std::bit_cast<std::uint32_t>(v2::convergence_curve_default);
    state[v2::container_scale] = std::bit_cast<std::uint32_t>(1.0f);
    state[v2::calibration_revision] = 1u;
    state[v2::frame_valid] = std::bit_cast<std::uint32_t>(1.0f);
    state[v2::contract_tag_bits] = v2::contract_tag;
    state[v2::renderer_authorization_bits] = v2::contract_tag;
    state[v2::mapping_state_reserved_1] = 0u;
    state[v2::mapping_state_reserved_2] = 0u;
    state[v2::camera_center_integrity_bits] =
      v2::camera_center_integrity_for_words(
        state[v2::center],
        state[v2::inverse_scale],
        state[v2::convergence_curve],
        state[v2::calibration_revision]
      );
    return state;
  }

  class p010_y_mrt_fixture_t {
  public:
    bool initialize(std::string &error) {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      auto status = D3D11CreateDevice(
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
      );
      if (FAILED(status) || actual < D3D_FEATURE_LEVEL_11_0) {
        error = "D3D11 WARP feature-level 11 creation failed: " +
          hresult_string(status);
        return false;
      }

      const std::filesystem::path shader_root = SUNSHINE_SHADERS_DIR;
      namespace cache = models::host_sbs_shader_cache;
      const auto live_sources = cache::snapshot_sources(
        shader_root,
        cache::parallax_v2_live_renderer_specs
      );
      if (!live_sources) {
        error = "could not snapshot the authenticated live V2 renderer closure";
        return false;
      }
      const auto live_closure = cache::source_closure_sha256(live_sources);
      if (live_closure != cache::parallax_v2_live_renderer_source_closure_sha256) {
        error = "live V2 renderer closure pin mismatch: got " + live_closure;
        return false;
      }
      const auto live_ps_bytecode = cache::get(
        live_sources,
        cache::parallax_v2_live_renderer
      );
      const auto live_vs_bytecode = cache::get(
        live_sources,
        cache::sbs_reprojection_vertex
      );
      if (!live_ps_bytecode || !live_vs_bytecode) {
        error = "could not compile the authenticated live V2 renderer closure";
        return false;
      }
      status = device->CreatePixelShader(
        live_ps_bytecode->data(),
        live_ps_bytecode->size(),
        nullptr,
        &live_pixel_shader
      );
      if (FAILED(status)) {
        error = "could not create the authenticated live V2 pixel shader: " +
          hresult_string(status);
        return false;
      }
      status = device->CreateVertexShader(
        live_vs_bytecode->data(),
        live_vs_bytecode->size(),
        nullptr,
        &fullscreen_vertex_shader
      );
      if (FAILED(status)) {
        error = "could not create the authenticated SBS vertex shader: " +
          hresult_string(status);
        return false;
      }

      const auto mrt_sources = cache::snapshot_sources(
        shader_root,
        cache::parallax_v2_p010_y_specs
      );
      if (!mrt_sources) {
        error = "could not snapshot the optional P010-Y renderer closure";
        return false;
      }
      const auto mrt_closure = cache::source_closure_sha256(mrt_sources);
      if (mrt_closure != cache::parallax_v2_p010_y_source_closure_sha256) {
        error = "P010-Y renderer closure pin mismatch: got " + mrt_closure;
        return false;
      }
      const auto mrt_ps_bytecode = cache::get(
        mrt_sources,
        cache::parallax_v2_p010_y_renderer
      );
      if (!mrt_ps_bytecode) {
        error = "could not compile the optional P010-Y MRT pixel shader";
        return false;
      }
      status = device->CreatePixelShader(
        mrt_ps_bytecode->data(),
        mrt_ps_bytecode->size(),
        nullptr,
        &mrt_pixel_shader
      );
      if (FAILED(status)) {
        error = "could not create the optional P010-Y MRT pixel shader: " +
          hresult_string(status);
        return false;
      }

      if (!compile_shader(
            shader_root / "convert_yuv420_planar_y_vs.hlsl",
            "main_vs",
            "vs_5_0",
            y_vertex_shader,
            error
          ) ||
          !compile_shader(
            shader_root /
              "convert_yuv420_planar_y_ps_perceptual_quantizer.hlsl",
            "main_ps",
            "ps_5_0",
            y_pixel_shader,
            error
          )) {
        return false;
      }

      D3D11_SAMPLER_DESC sampler_desc {};
      sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
      sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
      status = device->CreateSamplerState(&sampler_desc, &linear_sampler);
      if (FAILED(status)) {
        error = "could not create the linear-clamp sampler: " +
          hresult_string(status);
        return false;
      }
      return true;
    }

    bool compare_paths(
      const UINT source_width,
      const UINT source_height,
      const std::vector<rgba16_bits_t> &source_pixels,
      const UINT depth_width,
      const UINT depth_height,
      const std::vector<float> &parallax,
      const models::depth_coordinate_v2::state_words_t &state,
      const host_sbs_v2_geometry_t &geometry,
      const video::sunshine_colorspace_t &colorspace,
      std::vector<rgba16_bits_t> &reference_rgb,
      std::vector<rgba16_bits_t> &mrt_rgb,
      std::vector<std::uint16_t> &reference_y,
      std::vector<std::uint16_t> &mrt_y,
      std::string &error,
      const bool actual_p010 = false,
      bool *p010_unsupported = nullptr
    ) {
      if (p010_unsupported) {
        *p010_unsupported = false;
      }
      if (source_width == 0u || source_height == 0u ||
          source_pixels.size() !=
            static_cast<std::size_t>(source_width) * source_height ||
          parallax.size() !=
            static_cast<std::size_t>(depth_width) * depth_height) {
        error = "invalid P010-Y parity input";
        return false;
      }
      const UINT packed_width = source_width * 2u;

      ComPtr<ID3D11Texture2D> source_texture;
      ComPtr<ID3D11ShaderResourceView> source_srv;
      if (!create_immutable_texture_srv(
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            source_width,
            source_height,
            sizeof(rgba16_bits_t),
            source_pixels.data(),
            source_texture,
            source_srv,
            error
          )) {
        return false;
      }
      ComPtr<ID3D11Texture2D> parallax_texture;
      ComPtr<ID3D11ShaderResourceView> parallax_srv;
      if (!create_immutable_texture_srv(
            DXGI_FORMAT_R32_FLOAT,
            depth_width,
            depth_height,
            sizeof(float),
            parallax.data(),
            parallax_texture,
            parallax_srv,
            error
          )) {
        return false;
      }

      ComPtr<ID3D11Buffer> state_buffer;
      ComPtr<ID3D11ShaderResourceView> state_srv;
      if (!create_state_srv(state, state_buffer, state_srv, error)) {
        return false;
      }
      ComPtr<ID3D11Buffer> geometry_buffer;
      if (!create_constant_buffer(geometry, geometry_buffer, error)) {
        return false;
      }
      const auto *color_vectors =
        video::color_vectors_from_colorspace(colorspace, true);
      if (!color_vectors) {
        error = "missing HDR color vectors";
        return false;
      }
      ComPtr<ID3D11Buffer> color_buffer;
      if (!create_constant_buffer(*color_vectors, color_buffer, error)) {
        return false;
      }
      const rotate_constants_t rotate_constants {};
      ComPtr<ID3D11Buffer> rotate_buffer;
      if (!create_constant_buffer(rotate_constants, rotate_buffer, error)) {
        return false;
      }

      ComPtr<ID3D11Texture2D> reference_rgb_texture;
      ComPtr<ID3D11RenderTargetView> reference_rgb_rtv;
      ComPtr<ID3D11ShaderResourceView> reference_rgb_srv;
      if (!create_render_target(
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            packed_width,
            source_height,
            true,
            reference_rgb_texture,
            reference_rgb_rtv,
            reference_rgb_srv,
            error
          )) {
        return false;
      }
      ComPtr<ID3D11Texture2D> mrt_rgb_texture;
      ComPtr<ID3D11RenderTargetView> mrt_rgb_rtv;
      ComPtr<ID3D11ShaderResourceView> unused_mrt_rgb_srv;
      if (!create_render_target(
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            packed_width,
            source_height,
            false,
            mrt_rgb_texture,
            mrt_rgb_rtv,
            unused_mrt_rgb_srv,
            error
          )) {
        return false;
      }
      ComPtr<ID3D11Texture2D> reference_y_texture;
      ComPtr<ID3D11RenderTargetView> reference_y_rtv;
      ComPtr<ID3D11ShaderResourceView> unused_reference_y_srv;
      ComPtr<ID3D11Texture2D> mrt_y_texture;
      ComPtr<ID3D11RenderTargetView> mrt_y_rtv;
      ComPtr<ID3D11ShaderResourceView> unused_mrt_y_srv;
      if (actual_p010) {
        if (!create_p010_y_render_target(
              packed_width,
              source_height,
              reference_y_texture,
              reference_y_rtv,
              error
            ) ||
            !create_p010_y_render_target(
              packed_width,
              source_height,
              mrt_y_texture,
              mrt_y_rtv,
              error
            )) {
          if (p010_unsupported) {
            *p010_unsupported = true;
          }
          return false;
        }
      } else {
        if (!create_render_target(
              DXGI_FORMAT_R16_UNORM,
              packed_width,
              source_height,
              false,
              reference_y_texture,
              reference_y_rtv,
              unused_reference_y_srv,
              error
            ) ||
            !create_render_target(
              DXGI_FORMAT_R16_UNORM,
              packed_width,
              source_height,
              false,
              mrt_y_texture,
              mrt_y_rtv,
              unused_mrt_y_srv,
              error
            )) {
          return false;
        }
      }

      const D3D11_VIEWPORT viewport {
        0.0f,
        0.0f,
        static_cast<float>(packed_width),
        static_cast<float>(source_height),
        0.0f,
        1.0f,
      };
      context->IASetInputLayout(nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      context->RSSetViewports(1, &viewport);

      ID3D11ShaderResourceView *live_srvs[] = {
        source_srv.Get(),
        parallax_srv.Get(),
        state_srv.Get(),
      };
      ID3D11SamplerState *samplers[] = {linear_sampler.Get()};
      ID3D11Buffer *geometry_buffers[] = {geometry_buffer.Get()};

      // Established reference: authenticated warp into FP16, then a distinct PQ luma draw.
      context->OMSetRenderTargets(1, reference_rgb_rtv.GetAddressOf(), nullptr);
      context->VSSetShader(fullscreen_vertex_shader.Get(), nullptr, 0);
      context->PSSetShader(live_pixel_shader.Get(), nullptr, 0);
      context->PSSetShaderResources(
        0,
        static_cast<UINT>(std::size(live_srvs)),
        live_srvs
      );
      context->PSSetSamplers(0, 1, samplers);
      context->PSSetConstantBuffers(2, 1, geometry_buffers);
      context->Draw(3, 0);
      unbind_outputs_and_srvs();

      context->OMSetRenderTargets(1, reference_y_rtv.GetAddressOf(), nullptr);
      context->VSSetShader(y_vertex_shader.Get(), nullptr, 0);
      context->PSSetShader(y_pixel_shader.Get(), nullptr, 0);
      ID3D11ShaderResourceView *reference_y_srvs[] = {reference_rgb_srv.Get()};
      ID3D11Buffer *color_buffers[] = {color_buffer.Get()};
      ID3D11Buffer *rotate_buffers[] = {rotate_buffer.Get()};
      context->PSSetShaderResources(0, 1, reference_y_srvs);
      context->PSSetSamplers(0, 1, samplers);
      context->PSSetConstantBuffers(0, 1, color_buffers);
      context->VSSetConstantBuffers(1, 1, rotate_buffers);
      context->Draw(3, 0);
      unbind_outputs_and_srvs();

      // Candidate: identical authenticated warp with FP16 RGB and R16_UNORM Y bound together.
      ID3D11RenderTargetView *mrt_targets[] = {
        mrt_rgb_rtv.Get(),
        mrt_y_rtv.Get(),
      };
      context->OMSetRenderTargets(
        static_cast<UINT>(std::size(mrt_targets)),
        mrt_targets,
        nullptr
      );
      context->VSSetShader(fullscreen_vertex_shader.Get(), nullptr, 0);
      context->PSSetShader(mrt_pixel_shader.Get(), nullptr, 0);
      context->PSSetShaderResources(
        0,
        static_cast<UINT>(std::size(live_srvs)),
        live_srvs
      );
      context->PSSetSamplers(0, 1, samplers);
      context->PSSetConstantBuffers(0, 1, color_buffers);
      context->PSSetConstantBuffers(2, 1, geometry_buffers);
      context->Draw(3, 0);
      unbind_outputs_and_srvs();

      if (!readback_texture(
            reference_rgb_texture.Get(),
            packed_width,
            source_height,
            reference_rgb,
            error
          ) ||
          !readback_texture(
            mrt_rgb_texture.Get(),
            packed_width,
            source_height,
            mrt_rgb,
            error
          )) {
        return false;
      }
      if (actual_p010) {
        return readback_p010_y(
                 reference_y_texture.Get(),
                 packed_width,
                 source_height,
                 reference_y,
                 error
               ) &&
               readback_p010_y(
                 mrt_y_texture.Get(),
                 packed_width,
                 source_height,
                 mrt_y,
                 error
               );
      }
      return readback_texture(
               reference_y_texture.Get(),
               packed_width,
               source_height,
               reference_y,
               error
             ) &&
             readback_texture(
               mrt_y_texture.Get(),
               packed_width,
               source_height,
               mrt_y,
               error
             );
    }

  private:
    template <typename ShaderInterface>
    bool compile_shader(
      const std::filesystem::path &path,
      const char *entrypoint,
      const char *target,
      ComPtr<ShaderInterface> &shader,
      std::string &error
    ) {
      ComPtr<ID3DBlob> bytecode;
      ComPtr<ID3DBlob> errors;
      const auto status = D3DCompileFromFile(
        path.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entrypoint,
        target,
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &bytecode,
        &errors
      );
      if (FAILED(status) || !bytecode) {
        error = errors ?
          static_cast<const char *>(errors->GetBufferPointer()) :
          "shader compilation failed: " + hresult_string(status);
        return false;
      }
      HRESULT create_status = E_NOINTERFACE;
      if constexpr (std::is_same_v<ShaderInterface, ID3D11VertexShader>) {
        create_status = device->CreateVertexShader(
          bytecode->GetBufferPointer(),
          bytecode->GetBufferSize(),
          nullptr,
          &shader
        );
      } else {
        create_status = device->CreatePixelShader(
          bytecode->GetBufferPointer(),
          bytecode->GetBufferSize(),
          nullptr,
          &shader
        );
      }
      if (FAILED(create_status)) {
        error = "shader creation failed for " + path.string() + ": " +
          hresult_string(create_status);
        return false;
      }
      return true;
    }

    bool create_immutable_texture_srv(
      const DXGI_FORMAT format,
      const UINT width,
      const UINT height,
      const UINT bytes_per_pixel,
      const void *pixels,
      ComPtr<ID3D11Texture2D> &texture,
      ComPtr<ID3D11ShaderResourceView> &srv,
      std::string &error
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
      const D3D11_SUBRESOURCE_DATA data {
        pixels,
        width * bytes_per_pixel,
        0u,
      };
      auto status = device->CreateTexture2D(&desc, &data, &texture);
      if (FAILED(status)) {
        error = "could not create immutable shader texture " +
          std::to_string(static_cast<unsigned>(format)) + ": " +
          hresult_string(status);
        return false;
      }
      status = device->CreateShaderResourceView(texture.Get(), nullptr, &srv);
      if (FAILED(status)) {
        error = "could not create immutable texture SRV: " +
          hresult_string(status);
        return false;
      }
      return true;
    }

    bool create_state_srv(
      const models::depth_coordinate_v2::state_words_t &state,
      ComPtr<ID3D11Buffer> &buffer,
      ComPtr<ID3D11ShaderResourceView> &srv,
      std::string &error
    ) {
      D3D11_BUFFER_DESC buffer_desc {};
      buffer_desc.ByteWidth = sizeof(state);
      buffer_desc.Usage = D3D11_USAGE_IMMUTABLE;
      buffer_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      buffer_desc.StructureByteStride =
        models::depth_coordinate_v2::state_vector_width * sizeof(float);
      const D3D11_SUBRESOURCE_DATA data {state.data(), 0u, 0u};
      auto status = device->CreateBuffer(&buffer_desc, &data, &buffer);
      if (FAILED(status)) {
        error = "could not create the authenticated state buffer: " +
          hresult_string(status);
        return false;
      }
      D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc {};
      srv_desc.Format = DXGI_FORMAT_UNKNOWN;
      srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      srv_desc.Buffer.FirstElement = 0u;
      srv_desc.Buffer.NumElements =
        models::depth_coordinate_v2::state_vector_count;
      status = device->CreateShaderResourceView(buffer.Get(), &srv_desc, &srv);
      if (FAILED(status)) {
        error = "could not create the authenticated state SRV: " +
          hresult_string(status);
        return false;
      }
      return true;
    }

    template <typename Constants>
    bool create_constant_buffer(
      const Constants &constants,
      ComPtr<ID3D11Buffer> &buffer,
      std::string &error
    ) {
      static_assert(sizeof(Constants) % 16u == 0u);
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(Constants);
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const D3D11_SUBRESOURCE_DATA data {&constants, 0u, 0u};
      const auto status = device->CreateBuffer(&desc, &data, &buffer);
      if (FAILED(status)) {
        error = "could not create constant buffer: " + hresult_string(status);
        return false;
      }
      return true;
    }

    bool create_render_target(
      const DXGI_FORMAT format,
      const UINT width,
      const UINT height,
      const bool shader_resource,
      ComPtr<ID3D11Texture2D> &texture,
      ComPtr<ID3D11RenderTargetView> &rtv,
      ComPtr<ID3D11ShaderResourceView> &srv,
      std::string &error
    ) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = 1u;
      desc.ArraySize = 1u;
      desc.Format = format;
      desc.SampleDesc.Count = 1u;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_RENDER_TARGET |
        (shader_resource ? D3D11_BIND_SHADER_RESOURCE : 0u);
      auto status = device->CreateTexture2D(&desc, nullptr, &texture);
      if (FAILED(status)) {
        UINT support = 0u;
        device->CheckFormatSupport(format, &support);
        error = "WARP could not create MRT texture format=" +
          std::to_string(static_cast<unsigned>(format)) + " support=" +
          hresult_string(static_cast<HRESULT>(support)) + " status=" +
          hresult_string(status);
        return false;
      }
      status = device->CreateRenderTargetView(texture.Get(), nullptr, &rtv);
      if (FAILED(status)) {
        error = "WARP could not create MRT render-target view format=" +
          std::to_string(static_cast<unsigned>(format)) + ": " +
          hresult_string(status);
        return false;
      }
      if (shader_resource) {
        status = device->CreateShaderResourceView(texture.Get(), nullptr, &srv);
        if (FAILED(status)) {
          error = "could not create render-target SRV: " +
            hresult_string(status);
          return false;
        }
      }
      return true;
    }

    bool create_p010_y_render_target(
      const UINT width,
      const UINT height,
      ComPtr<ID3D11Texture2D> &texture,
      ComPtr<ID3D11RenderTargetView> &y_rtv,
      std::string &error
    ) {
      if ((width & 1u) != 0u || (height & 1u) != 0u) {
        error = "P010 probe dimensions must be even";
        return false;
      }
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = 1u;
      desc.ArraySize = 1u;
      desc.Format = DXGI_FORMAT_P010;
      desc.SampleDesc.Count = 1u;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_RENDER_TARGET;
      auto status = device->CreateTexture2D(&desc, nullptr, &texture);
      if (FAILED(status)) {
        UINT support = 0u;
        const auto support_status =
          device->CheckFormatSupport(DXGI_FORMAT_P010, &support);
        error = "WARP does not support an actual P010 render-target texture: status=" +
          hresult_string(status) + " check-status=" +
          hresult_string(support_status) + " support-bits=" +
          hresult_string(static_cast<HRESULT>(support));
        return false;
      }

      D3D11_RENDER_TARGET_VIEW_DESC rtv_desc {};
      rtv_desc.Format = DXGI_FORMAT_R16_UNORM;
      rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
      rtv_desc.Texture2D.MipSlice = 0u;
      status = device->CreateRenderTargetView(
        texture.Get(),
        &rtv_desc,
        &y_rtv
      );
      if (FAILED(status)) {
        error = "WARP cannot expose the actual P010 Y plane as R16_UNORM RTV: " +
          hresult_string(status);
        return false;
      }
      return true;
    }

    template <typename Pixel>
    bool readback_texture(
      ID3D11Texture2D *texture,
      const UINT width,
      const UINT height,
      std::vector<Pixel> &pixels,
      std::string &error
    ) {
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      auto staging_desc = desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0u;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      auto status = device->CreateTexture2D(&staging_desc, nullptr, &staging);
      if (FAILED(status)) {
        error = "could not create parity staging texture: " +
          hresult_string(status);
        return false;
      }
      context->CopyResource(staging.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      status = context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped);
      if (FAILED(status)) {
        error = "could not map parity staging texture: " +
          hresult_string(status);
        return false;
      }
      pixels.resize(static_cast<std::size_t>(width) * height);
      const std::size_t row_bytes = static_cast<std::size_t>(width) * sizeof(Pixel);
      for (UINT y = 0u; y < height; ++y) {
        std::memcpy(
          pixels.data() + static_cast<std::size_t>(y) * width,
          static_cast<const std::byte *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          row_bytes
        );
      }
      context->Unmap(staging.Get(), 0u);
      return true;
    }

    bool readback_p010_y(
      ID3D11Texture2D *texture,
      const UINT width,
      const UINT height,
      std::vector<std::uint16_t> &pixels,
      std::string &error
    ) {
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      if (desc.Format != DXGI_FORMAT_P010 || desc.Width != width ||
          desc.Height != height) {
        error = "invalid actual P010 texture passed to raw Y-plane readback";
        return false;
      }
      auto staging_desc = desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0u;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      auto status = device->CreateTexture2D(&staging_desc, nullptr, &staging);
      if (FAILED(status)) {
        error = "could not create actual P010 staging texture: " +
          hresult_string(status);
        return false;
      }
      context->CopyResource(staging.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      status = context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped);
      if (FAILED(status)) {
        error = "could not map actual P010 texture for raw Y readback: " +
          hresult_string(status);
        return false;
      }
      pixels.resize(static_cast<std::size_t>(width) * height);
      const std::size_t row_bytes =
        static_cast<std::size_t>(width) * sizeof(std::uint16_t);
      // D3D11 planar staging maps the full resource with Y first, followed by interleaved UV.
      // Copy only the first `height` rows so this comparison observes raw stored Y-plane bits.
      for (UINT y = 0u; y < height; ++y) {
        std::memcpy(
          pixels.data() + static_cast<std::size_t>(y) * width,
          static_cast<const std::byte *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          row_bytes
        );
      }
      context->Unmap(staging.Get(), 0u);
      return true;
    }

    void unbind_outputs_and_srvs() {
      std::array<ID3D11RenderTargetView *, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT>
        null_targets {};
      std::array<ID3D11ShaderResourceView *, 3> null_srvs {};
      context->OMSetRenderTargets(
        static_cast<UINT>(null_targets.size()),
        null_targets.data(),
        nullptr
      );
      context->PSSetShaderResources(
        0,
        static_cast<UINT>(null_srvs.size()),
        null_srvs.data()
      );
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11VertexShader> fullscreen_vertex_shader;
    ComPtr<ID3D11VertexShader> y_vertex_shader;
    ComPtr<ID3D11PixelShader> live_pixel_shader;
    ComPtr<ID3D11PixelShader> mrt_pixel_shader;
    ComPtr<ID3D11PixelShader> y_pixel_shader;
    ComPtr<ID3D11SamplerState> linear_sampler;
  };

  std::vector<rgba16_bits_t> make_hdr_source(
    const UINT width,
    const UINT height
  ) {
    // Exact half-float codes cover negative scRGB, dark values, diffuse white, and highlights.
    constexpr std::array<std::uint16_t, 12> values {
      0xb800u,  // -0.5
      0x0000u,  //  0.0
      0x2c00u,  //  0.0625
      0x3000u,  //  0.125
      0x3400u,  //  0.25
      0x3800u,  //  0.5
      0x3a00u,  //  0.75
      0x3c00u,  //  1.0 / 80 nit
      0x3e00u,  //  1.5
      0x4000u,  //  2.0
      0x4400u,  //  4.0
      0x4800u,  //  8.0
    };
    std::vector<rgba16_bits_t> pixels(
      static_cast<std::size_t>(width) * height
    );
    for (UINT y = 0u; y < height; ++y) {
      for (UINT x = 0u; x < width; ++x) {
        auto &pixel = pixels[static_cast<std::size_t>(y) * width + x];
        pixel = {
          values[(x + 3u * y) % values.size()],
          values[(5u * x + y + 4u) % values.size()],
          values[(7u * x + 2u * y + 7u) % values.size()],
          0x3c00u,
        };
      }
    }
    // High-contrast source edges and the two source columns adjacent to the packed-eye seam
    // make clamping, eye selection, and bilinear interpolation visible in both output planes.
    for (UINT y = 0u; y < height; ++y) {
      pixels[static_cast<std::size_t>(y) * width] = {
        0x0000u, 0x4800u, 0xb800u, 0x3c00u
      };
      pixels[static_cast<std::size_t>(y) * width + width - 1u] = {
        0x4800u, 0xb800u, 0x0000u, 0x3c00u
      };
      pixels[static_cast<std::size_t>(y) * width + width / 2u - 1u] = {
        0x2c00u, 0x4400u, 0x3800u, 0x3c00u
      };
      pixels[static_cast<std::size_t>(y) * width + width / 2u] = {
        0x4400u, 0x2c00u, 0x4000u, 0x3c00u
      };
    }
    return pixels;
  }
}  // namespace

TEST(HostSbsP010YMrtGpuTest, MatchesEstablishedFp16ThenPqLumaPathBitForBit) {
  constexpr UINT source_width = 16u;
  constexpr UINT source_height = 10u;
  constexpr UINT depth_width = 8u;
  constexpr UINT depth_height = 6u;
  const auto source = make_hdr_source(source_width, source_height);
  const auto state = make_authenticated_state();
  const std::vector<float> zero_parallax(
    static_cast<std::size_t>(depth_width) * depth_height,
    0.0f
  );
  std::vector<float> spatial_parallax(zero_parallax.size());
  for (UINT y = 0u; y < depth_height; ++y) {
    for (UINT x = 0u; x < depth_width; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) /
        static_cast<float>(depth_width);
      const float v = (static_cast<float>(y) + 0.5f) /
        static_cast<float>(depth_height);
      spatial_parallax[static_cast<std::size_t>(y) * depth_width + x] =
        0.052f + 0.012f * (u - 0.5f) - 0.006f * (v - 0.5f);
    }
  }

  host_sbs_v2_geometry_t aspect_bars {};
  aspect_bars.content_scale_x = 0.75f;
  aspect_bars.content_scale_y = 0.6f;
  host_sbs_v2_geometry_t video_roi {};
  video_roi.video_roi_active = 1.0f;
  video_roi.video_roi_left = 0.1875f;
  video_roi.video_roi_top = 0.2f;
  video_roi.video_roi_right = 0.8125f;
  video_roi.video_roi_bottom = 0.8f;
  video_roi.tensor_content_right = depth_width;
  video_roi.tensor_content_bottom = depth_height;

  struct parity_case_t {
    const char *name;
    const std::vector<float> *parallax;
    host_sbs_v2_geometry_t geometry;
    bool full_range;
  };
  const std::array cases {
    parity_case_t {
      "zero parallax limited range", &zero_parallax, {}, false
    },
    parity_case_t {
      "authenticated spatial warp limited range", &spatial_parallax, {}, false
    },
    parity_case_t {
      "authenticated warp with aspect bars full range",
      &spatial_parallax,
      aspect_bars,
      true
    },
    parity_case_t {
      "authenticated video ROI warp limited range",
      &spatial_parallax,
      video_roi,
      false
    },
  };

  p010_y_mrt_fixture_t fixture;
  std::string error;
  ASSERT_TRUE(fixture.initialize(error)) << error;

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    std::vector<rgba16_bits_t> reference_rgb;
    std::vector<rgba16_bits_t> mrt_rgb;
    std::vector<std::uint16_t> reference_y;
    std::vector<std::uint16_t> mrt_y;
    ASSERT_TRUE(fixture.compare_paths(
      source_width,
      source_height,
      source,
      depth_width,
      depth_height,
      *test_case.parallax,
      state,
      test_case.geometry,
      {video::colorspace_e::bt2020, test_case.full_range, 10u},
      reference_rgb,
      mrt_rgb,
      reference_y,
      mrt_y,
      error
    )) << error;
    ASSERT_EQ(reference_rgb.size(), mrt_rgb.size());
    ASSERT_EQ(reference_y.size(), mrt_y.size());
    ASSERT_EQ(reference_rgb.size(), reference_y.size());

    bool observed_nonzero_luma = false;
    for (std::size_t index = 0u; index < reference_rgb.size(); ++index) {
      EXPECT_EQ(reference_rgb[index], mrt_rgb[index])
        << "packed FP16 pixel=" << index;
      EXPECT_EQ(reference_y[index], mrt_y[index])
        << "R16_UNORM luma pixel=" << index;
      observed_nonzero_luma |= reference_y[index] != 0u;
    }
    EXPECT_TRUE(observed_nonzero_luma);
  }
}

TEST(HostSbsP010YMrtGpuTest, ActualP010YPlaneMatchesAtProductionPackedWidth) {
  // A 7680-wide packed target exercises the real 4K-SBS texcoord precision. A short height keeps
  // this software-rasterizer qualification focused on 1:1 sampling and planar resource behavior.
  constexpr UINT source_width = 3840u;
  constexpr UINT source_height = 4u;
  constexpr UINT depth_width = 64u;
  constexpr UINT depth_height = 4u;
  const auto source = make_hdr_source(source_width, source_height);
  const auto state = make_authenticated_state();
  std::vector<float> parallax(
    static_cast<std::size_t>(depth_width) * depth_height
  );
  for (UINT y = 0u; y < depth_height; ++y) {
    for (UINT x = 0u; x < depth_width; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) /
        static_cast<float>(depth_width);
      const float v = (static_cast<float>(y) + 0.5f) /
        static_cast<float>(depth_height);
      parallax[static_cast<std::size_t>(y) * depth_width + x] =
        0.018f + 0.009f * (u - 0.5f) + 0.004f * (v - 0.5f);
    }
  }

  p010_y_mrt_fixture_t fixture;
  std::string error;
  ASSERT_TRUE(fixture.initialize(error)) << error;
  std::vector<rgba16_bits_t> reference_rgb;
  std::vector<rgba16_bits_t> mrt_rgb;
  std::vector<std::uint16_t> reference_y;
  std::vector<std::uint16_t> mrt_y;
  bool p010_unsupported = false;
  if (!fixture.compare_paths(
        source_width,
        source_height,
        source,
        depth_width,
        depth_height,
        parallax,
        state,
        {},
        {video::colorspace_e::bt2020, false, 10u},
        reference_rgb,
        mrt_rgb,
        reference_y,
        mrt_y,
        error,
        true,
        &p010_unsupported
      )) {
    if (p010_unsupported) {
      GTEST_SKIP() << error;
    }
    FAIL() << error;
  }

  ASSERT_EQ(reference_rgb.size(), mrt_rgb.size());
  ASSERT_EQ(reference_y.size(), mrt_y.size());
  ASSERT_EQ(reference_rgb.size(), reference_y.size());
  ASSERT_FALSE(reference_y.empty());
  bool observed_nonzero_luma = false;
  bool observed_multiple_luma_codes = false;
  for (std::size_t index = 0u; index < reference_rgb.size(); ++index) {
    EXPECT_EQ(reference_rgb[index], mrt_rgb[index])
      << "7680-wide packed FP16 pixel=" << index;
    EXPECT_EQ(reference_y[index], mrt_y[index])
      << "raw actual-P010 Y pixel=" << index;
    observed_nonzero_luma |= reference_y[index] != 0u;
    observed_multiple_luma_codes |= reference_y[index] != reference_y.front();
  }
  EXPECT_TRUE(observed_nonzero_luma);
  EXPECT_TRUE(observed_multiple_luma_codes);
}

#endif  // _WIN32
