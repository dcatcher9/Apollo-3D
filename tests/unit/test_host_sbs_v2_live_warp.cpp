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
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <src/depth_coordinate_v2.h>
#include <src/host_sbs_shader_cache.h>

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

  class live_v2_warp_fixture_t {
  public:
    bool initialize(std::string &error, const bool flat_identity = false) {
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
        error = "D3D11 WARP feature level 11 initialization failed";
        return false;
      }

      const std::filesystem::path shader_root = SUNSHINE_SHADERS_DIR;
      if (flat_identity) {
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
      } else {
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

      // Exact 16-byte mirror of HostSbsV2Geometry at b2. Unit content scale maps each packed eye
      // to the entire mono source; the reserved words must not become hidden geometry authority.
      constexpr std::array<float, 4> constants {
        1.0f, 1.0f, 0.0f, 0.0f,
      };
      D3D11_BUFFER_DESC constant_desc {};
      constant_desc.ByteWidth = sizeof(constants);
      constant_desc.Usage = D3D11_USAGE_IMMUTABLE;
      constant_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA constant_data {};
      constant_data.pSysMem = constants.data();
      if (FAILED(device->CreateBuffer(
            &constant_desc,
            &constant_data,
            &constant_buffer
          ))) {
        error = "could not create live V2 render constants";
        return false;
      }
      return true;
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
      const UINT depth_height = 0u
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

      ComPtr<ID3D11Texture2D> candidate_texture;
      ComPtr<ID3D11ShaderResourceView> candidate_srv;
      if (!create_immutable_texture_srv(
            DXGI_FORMAT_R32_FLOAT,
            resolved_depth_width,
            resolved_depth_height,
            sizeof(float),
            candidate_parallax.data(),
            candidate_texture,
            candidate_srv
          )) {
        error = "could not create the candidate parallax texture";
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

      const UINT packed_width = source_width * 2u;
      D3D11_TEXTURE2D_DESC output_desc {};
      output_desc.Width = packed_width;
      output_desc.Height = source_height;
      output_desc.MipLevels = 1u;
      output_desc.ArraySize = 1u;
      output_desc.Format = color_format;
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
      ID3D11ShaderResourceView *srvs[] = {
        color_srv.Get(),
        parallax_srv.Get(),
        state_srv.Get(),
        candidate_srv.Get(),
        nullptr,
        nullptr,
      };
      ID3D11SamplerState *samplers[] = {sampler.Get()};
      ID3D11Buffer *constants[] = {constant_buffer.Get()};
      context->IASetInputLayout(nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      context->VSSetShader(vertex_shader.Get(), nullptr, 0);
      context->PSSetShader(pixel_shader.Get(), nullptr, 0);
      context->PSSetShaderResources(0, static_cast<UINT>(std::size(srvs)), srvs);
      context->PSSetSamplers(0, 1, samplers);
      context->PSSetConstantBuffers(2, 1, constants);
      context->RSSetViewports(1, &viewport);
      context->OMSetRenderTargets(1, output_rtv.GetAddressOf(), nullptr);
      context->Draw(3, 0);

      ID3D11RenderTargetView *null_rtv = nullptr;
      std::array<ID3D11ShaderResourceView *, 6> null_srvs {};
      context->OMSetRenderTargets(1, &null_rtv, nullptr);
      context->PSSetShaderResources(
        0,
        static_cast<UINT>(null_srvs.size()),
        null_srvs.data()
      );

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
        static_cast<std::size_t>(packed_width) * color_bytes_per_pixel;
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
    ComPtr<ID3D11Buffer> constant_buffer;
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

TEST(HostSbsV2LiveWarpGpuTest, IndependentFlatShaderIgnoresV2Geometry) {
  constexpr UINT width = 8u;
  constexpr UINT height = 1u;
  const std::array<float, 4> sentinel_clear {0.91f, 0.13f, 0.77f, 0.42f};
  std::string error;
  live_v2_warp_fixture_t flat;
  ASSERT_TRUE(flat.initialize(error, true)) << error;

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

#endif  // _WIN32
