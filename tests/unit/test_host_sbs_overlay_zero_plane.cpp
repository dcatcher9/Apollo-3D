/**
 * @file tests/unit/test_host_sbs_overlay_zero_plane.cpp
 * @brief Pure-plan and D3D11-WARP checks for burned-in overlay final-field conditioning.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <src/depth_coordinate_v2.h>
#include <src/host_sbs_overlay_geometry.h>

namespace {
  using Microsoft::WRL::ComPtr;

  struct depth_constants_t {
    std::uint32_t target_w = 0u;
    std::uint32_t target_h = 0u;
    std::uint32_t color_mode = 0u;
    float ema_alpha = 0.0f;
    float minmax_alpha = 0.0f;
    std::uint32_t reduce_threads = 0u;
    float ema_edge_change = 0.0f;
    float ema_edge_gradient = 0.0f;
    float ema_edge_strength = 0.0f;
    std::array<float, 3> reserved {};
  };
  static_assert(sizeof(depth_constants_t) == 48u);

  template<class T>
  bool create_constant_buffer(
    ID3D11Device *device,
    const T &value,
    ComPtr<ID3D11Buffer> &buffer
  ) {
    static_assert(sizeof(T) % 16u == 0u);
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = static_cast<UINT>(sizeof(T));
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data {};
    data.pSysMem = &value;
    return SUCCEEDED(device->CreateBuffer(&desc, &data, &buffer));
  }

  class overlay_zero_plane_fixture_t {
  public:
    bool initialize(std::string &error) {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      const HRESULT device_status = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr,
        0u,
        requested,
        static_cast<UINT>(std::size(requested)),
        D3D11_SDK_VERSION,
        &device,
        &actual,
        &context
      );
      if (FAILED(device_status) || actual < D3D_FEATURE_LEVEL_11_0) {
        error = "D3D11 WARP feature level 11 initialization failed";
        return false;
      }

      const auto shader_path = std::filesystem::path(SUNSHINE_SHADERS_DIR) /
        "depth_coordinate_v2_overlay_zero_plane_cs.hlsl";
      ComPtr<ID3DBlob> bytecode;
      ComPtr<ID3DBlob> diagnostics;
      const HRESULT compile_status = D3DCompileFromFile(
        shader_path.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main",
        "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0u,
        &bytecode,
        &diagnostics
      );
      if (FAILED(compile_status) || !bytecode ||
          FAILED(device->CreateComputeShader(
            bytecode->GetBufferPointer(),
            bytecode->GetBufferSize(),
            nullptr,
            &shader
          ))) {
        error = diagnostics ?
                  static_cast<const char *>(diagnostics->GetBufferPointer()) :
                  "overlay zero-plane compute shader creation failed";
        return false;
      }
      return true;
    }

    bool dispatch(
      const std::uint32_t width,
      const std::uint32_t height,
      const std::uint32_t analysis_width,
      const std::uint32_t analysis_height,
      const std::vector<float> &input,
      const std::span<const models::depth_source_rect_t> rectangles,
      std::vector<float> &output,
      std::string &error,
      const std::uint32_t advertised_rect_count = 0u,
      const std::uint32_t overlay_reserved = 0u
    ) {
      if (width == 0u || height == 0u ||
          input.size() != static_cast<std::size_t>(width) * height) {
        error = "invalid overlay zero-plane test input";
        return false;
      }

      D3D11_TEXTURE2D_DESC input_desc {};
      input_desc.Width = width;
      input_desc.Height = height;
      input_desc.MipLevels = 1u;
      input_desc.ArraySize = 1u;
      input_desc.Format = DXGI_FORMAT_R32_FLOAT;
      input_desc.SampleDesc.Count = 1u;
      input_desc.Usage = D3D11_USAGE_IMMUTABLE;
      input_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA input_data {};
      input_data.pSysMem = input.data();
      input_data.SysMemPitch = static_cast<UINT>(width * sizeof(float));
      ComPtr<ID3D11Texture2D> input_texture;
      ComPtr<ID3D11ShaderResourceView> input_srv;
      if (FAILED(device->CreateTexture2D(
            &input_desc,
            &input_data,
            &input_texture
          )) ||
          FAILED(device->CreateShaderResourceView(
            input_texture.Get(),
            nullptr,
            &input_srv
          ))) {
        error = "could not create overlay input texture";
        return false;
      }

      auto output_desc = input_desc;
      output_desc.Usage = D3D11_USAGE_DEFAULT;
      output_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      ComPtr<ID3D11Texture2D> output_texture;
      ComPtr<ID3D11UnorderedAccessView> output_uav;
      if (FAILED(device->CreateTexture2D(
            &output_desc,
            nullptr,
            &output_texture
          )) ||
          FAILED(device->CreateUnorderedAccessView(
            output_texture.Get(),
            nullptr,
            &output_uav
          ))) {
        error = "could not create overlay output texture";
        return false;
      }

      ComPtr<ID3D11Buffer> rect_buffer;
      ComPtr<ID3D11ShaderResourceView> rect_srv;
      if (!rectangles.empty()) {
        D3D11_BUFFER_DESC rect_desc {};
        rect_desc.ByteWidth = static_cast<UINT>(
          rectangles.size_bytes()
        );
        rect_desc.Usage = D3D11_USAGE_IMMUTABLE;
        rect_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        rect_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        rect_desc.StructureByteStride = sizeof(models::depth_source_rect_t);
        D3D11_SUBRESOURCE_DATA rect_data {};
        rect_data.pSysMem = rectangles.data();
        D3D11_SHADER_RESOURCE_VIEW_DESC rect_srv_desc {};
        rect_srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        rect_srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        rect_srv_desc.Buffer.FirstElement = 0u;
        rect_srv_desc.Buffer.NumElements = static_cast<UINT>(rectangles.size());
        if (FAILED(device->CreateBuffer(
              &rect_desc,
              &rect_data,
              &rect_buffer
            )) ||
            FAILED(device->CreateShaderResourceView(
              rect_buffer.Get(),
              &rect_srv_desc,
              &rect_srv
            ))) {
          error = "could not create overlay rectangle buffer";
          return false;
        }
      }

      const depth_constants_t depth_constants {
        .target_w = width,
        .target_h = height,
      };
      auto v2_constants = models::depth_coordinate_v2::constants_t {};
      v2_constants.max_horizontal_slope =
        models::depth_coordinate_v2::max_horizontal_slope;
      const models::host_sbs_overlay_zero_plane_constants_t overlay_constants {
        .analysis_width = analysis_width,
        .analysis_height = analysis_height,
        .loose_rect_count = advertised_rect_count == 0u ?
                              static_cast<std::uint32_t>(rectangles.size()) :
                              advertised_rect_count,
        .reserved = overlay_reserved,
      };
      ComPtr<ID3D11Buffer> depth_cbuffer;
      ComPtr<ID3D11Buffer> v2_cbuffer;
      ComPtr<ID3D11Buffer> overlay_cbuffer;
      if (!create_constant_buffer(device.Get(), depth_constants, depth_cbuffer) ||
          !create_constant_buffer(device.Get(), v2_constants, v2_cbuffer) ||
          !create_constant_buffer(device.Get(), overlay_constants, overlay_cbuffer)) {
        error = "could not create overlay constant buffers";
        return false;
      }

      ID3D11ShaderResourceView *srvs[] = {input_srv.Get(), rect_srv.Get()};
      ID3D11Buffer *cbuffers[] = {
        depth_cbuffer.Get(),
        v2_cbuffer.Get(),
        overlay_cbuffer.Get(),
      };
      context->CSSetShader(shader.Get(), nullptr, 0u);
      context->CSSetShaderResources(0u, static_cast<UINT>(std::size(srvs)), srvs);
      context->CSSetUnorderedAccessViews(0u, 1u, output_uav.GetAddressOf(), nullptr);
      context->CSSetConstantBuffers(
        0u,
        static_cast<UINT>(std::size(cbuffers)),
        cbuffers
      );
      context->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);

      ID3D11ShaderResourceView *null_srvs[2] = {nullptr, nullptr};
      ID3D11UnorderedAccessView *null_uav = nullptr;
      ID3D11Buffer *null_cbuffers[3] = {nullptr, nullptr, nullptr};
      context->CSSetShaderResources(0u, 2u, null_srvs);
      context->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);
      context->CSSetConstantBuffers(0u, 3u, null_cbuffers);

      D3D11_TEXTURE2D_DESC staging_desc = output_desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0u;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging))) {
        error = "could not create overlay staging texture";
        return false;
      }
      context->CopyResource(staging.Get(), output_texture.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
        error = "could not read overlay conditioned field";
        return false;
      }
      output.resize(input.size());
      for (std::uint32_t y = 0u; y < height; ++y) {
        const auto *row = reinterpret_cast<const float *>(
          static_cast<const std::byte *>(mapped.pData) +
          static_cast<std::size_t>(y) * mapped.RowPitch
        );
        std::copy_n(
          row,
          width,
          output.begin() + static_cast<std::size_t>(y) * width
        );
      }
      context->Unmap(staging.Get(), 0u);
      return true;
    }

  private:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11ComputeShader> shader;
  };

  float sample_linear_clamp(
    const std::vector<float> &field,
    const std::uint32_t width,
    const std::uint32_t height,
    const float u,
    const float v
  ) {
    const float x = std::clamp(u, 0.0f, 1.0f) * static_cast<float>(width) - 0.5f;
    const float y = std::clamp(v, 0.0f, 1.0f) * static_cast<float>(height) - 0.5f;
    const int x0_unclamped = static_cast<int>(std::floor(x));
    const int y0_unclamped = static_cast<int>(std::floor(y));
    const auto x0 = static_cast<std::uint32_t>(
      std::clamp(x0_unclamped, 0, static_cast<int>(width) - 1)
    );
    const auto x1 = static_cast<std::uint32_t>(
      std::clamp(x0_unclamped + 1, 0, static_cast<int>(width) - 1)
    );
    const auto y0 = static_cast<std::uint32_t>(
      std::clamp(y0_unclamped, 0, static_cast<int>(height) - 1)
    );
    const auto y1 = static_cast<std::uint32_t>(
      std::clamp(y0_unclamped + 1, 0, static_cast<int>(height) - 1)
    );
    const float tx = x - std::floor(x);
    const float ty = y - std::floor(y);
    const auto at = [&](const std::uint32_t sx, const std::uint32_t sy) {
      return field[static_cast<std::size_t>(sy) * width + sx];
    };
    const float top = at(x0, y0) + tx * (at(x1, y0) - at(x0, y0));
    const float bottom = at(x0, y1) + tx * (at(x1, y1) - at(x0, y1));
    return top + ty * (bottom - top);
  }
}  // namespace

TEST(HostSbsOverlayGeometryTest, FixedPayloadRejectsOutOfBoundsAndStaleSlots) {
  models::host_sbs_overlay_geometry_t geometry;
  EXPECT_TRUE(geometry.valid(1920u, 1080u));
  EXPECT_TRUE(geometry.empty());

  geometry.loose_rect_count = 2u;
  geometry.loose_rects[0] = {120u, 700u, 1800u, 1030u};
  geometry.loose_rects[1] = {300u, 40u, 1500u, 220u};
  EXPECT_TRUE(geometry.valid(1920u, 1080u));
  EXPECT_FALSE(geometry.empty());

  auto invalid = geometry;
  invalid.loose_rects[1].right = 1921u;
  EXPECT_FALSE(invalid.valid(1920u, 1080u));

  invalid = geometry;
  invalid.loose_rect_count = 1u;
  EXPECT_FALSE(invalid.valid(1920u, 1080u));

  invalid = geometry;
  invalid.loose_rect_count = static_cast<std::uint32_t>(
    models::host_sbs_overlay_max_loose_rects + 1u
  );
  EXPECT_FALSE(invalid.valid(1920u, 1080u));
  EXPECT_FALSE(geometry.valid(0u, 1080u));

  models::host_sbs_overlay_geometry_t full_capacity;
  full_capacity.loose_rect_count = static_cast<std::uint32_t>(
    models::host_sbs_overlay_max_loose_rects
  );
  std::fill(
    full_capacity.loose_rects.begin(),
    full_capacity.loose_rects.end(),
    models::depth_source_rect_t {120u, 700u, 1800u, 1030u}
  );
  EXPECT_TRUE(full_capacity.valid(1920u, 1080u));
}

TEST(HostSbsOverlayZeroPlaneGpuTest, AbsentOrMalformedPlanLeavesFieldExact) {
  constexpr std::uint32_t width = 19u;
  constexpr std::uint32_t height = 11u;
  std::vector<float> input(static_cast<std::size_t>(width) * height);
  for (std::uint32_t y = 0u; y < height; ++y) {
    for (std::uint32_t x = 0u; x < width; ++x) {
      input[static_cast<std::size_t>(y) * width + x] =
        -0.035f + 0.003f * static_cast<float>((x * 5u + y * 7u) % 23u);
    }
  }

  overlay_zero_plane_fixture_t fixture;
  std::string error;
  ASSERT_TRUE(fixture.initialize(error)) << error;

  std::vector<float> output;
  ASSERT_TRUE(fixture.dispatch(
    width,
    height,
    1900u,
    1100u,
    input,
    {},
    output,
    error
  )) << error;
  EXPECT_EQ(output, input);

  const std::array malformed_rects {
    models::depth_source_rect_t {100u, 100u, 2000u, 400u},
  };
  ASSERT_TRUE(fixture.dispatch(
    width,
    height,
    1900u,
    1100u,
    input,
    malformed_rects,
    output,
    error
  )) << error;
  EXPECT_EQ(output, input);

  const std::array valid_rects {
    models::depth_source_rect_t {100u, 100u, 800u, 400u},
  };
  ASSERT_TRUE(fixture.dispatch(
    width,
    height,
    1900u,
    1100u,
    input,
    valid_rects,
    output,
    error,
    1u,
    1u
  )) << error;
  EXPECT_EQ(output, input);
}

TEST(HostSbsOverlayZeroPlaneGpuTest, CoreIsExactZeroAndCollarPreservesBothSlopeBounds) {
  namespace v2 = models::depth_coordinate_v2;
  constexpr std::uint32_t width = 32u;
  constexpr std::uint32_t height = 16u;
  constexpr std::uint32_t analysis_width = 3200u;
  constexpr std::uint32_t analysis_height = 1600u;
  constexpr float magnitude = 0.04f;
  const std::array rectangles {
    models::depth_source_rect_t {1200u, 400u, 2000u, 1200u},
    models::depth_source_rect_t {400u, 100u, 800u, 300u},
    // Partially overlap the large rectangle and extend it rightward. Same-plane overlaps are
    // intentionally safe and use the union of their exact-zero cores.
    models::depth_source_rect_t {1800u, 800u, 2400u, 1100u},
  };

  overlay_zero_plane_fixture_t fixture;
  std::string error;
  ASSERT_TRUE(fixture.initialize(error)) << error;

  for (const float sign : {1.0f, -1.0f}) {
    const std::vector<float> input(
      static_cast<std::size_t>(width) * height,
      sign * magnitude
    );
    std::vector<float> output;
    ASSERT_TRUE(fixture.dispatch(
      width,
      height,
      analysis_width,
      analysis_height,
      input,
      rectangles,
      output,
      error
    )) << error;
    ASSERT_EQ(output.size(), input.size());

    const auto at = [&](const std::uint32_t x, const std::uint32_t y) {
      return output[static_cast<std::size_t>(y) * width + x];
    };

    // The large rectangle itself, plus a complete model-texel sampling guard on every side, is
    // exact zero. Its requested X range is [12,20); x=11 and x=20 are the outside guard.
    for (std::uint32_t y = 4u; y < 12u; ++y) {
      for (std::uint32_t x = 12u; x < 20u; ++x) {
        EXPECT_FLOAT_EQ(at(x, y), 0.0f) << "x=" << x << ", y=" << y;
      }
    }
    EXPECT_FLOAT_EQ(at(11u, 8u), 0.0f);
    EXPECT_FLOAT_EQ(at(20u, 8u), 0.0f);
    EXPECT_FLOAT_EQ(at(16u, 3u), 0.0f);
    EXPECT_FLOAT_EQ(at(16u, 12u), 0.0f);

    // Check the continuous bilinear field, not only its stored texels. The authored half-open
    // rectangle remains exactly zero at points arbitrarily near all four boundaries.
    constexpr float epsilon = 1.0e-6f;
    for (const float u : {0.375f + epsilon, 0.5f, 0.625f - epsilon}) {
      for (const float v : {0.25f + epsilon, 0.5f, 0.75f - epsilon}) {
        EXPECT_FLOAT_EQ(sample_linear_clamp(output, width, height, u, v), 0.0f)
          << "bilinear u=" << u << ", v=" << v;
      }
    }

    // The horizontal transition begins outside the guard and reaches the untouched field at the
    // minimum distance allowed by the 0.5 source-U slope.
    EXPECT_NEAR(at(10u, 8u), sign * 0.0078125f, 2.0e-7f);
    EXPECT_NEAR(at(9u, 8u), sign * 0.0234375f, 2.0e-7f);
    EXPECT_NEAR(at(8u, 8u), sign * 0.0390625f, 2.0e-7f);
    EXPECT_FLOAT_EQ(at(7u, 8u), sign * magnitude);
    EXPECT_FLOAT_EQ(at(24u, 8u), 0.0f);
    EXPECT_FLOAT_EQ(at(28u, 8u), sign * magnitude);

    // The disjoint upper rectangle is conditioned independently rather than collapsed into one
    // broad bounding box.
    EXPECT_FLOAT_EQ(at(5u, 1u), 0.0f);
    EXPECT_FLOAT_EQ(at(12u, 1u), sign * magnitude);

    const float horizontal_step = v2::max_horizontal_slope /
                                  static_cast<float>(width);
    const float vertical_step = v2::max_vertical_shear /
                                static_cast<float>(width);
    for (std::uint32_t y = 0u; y < height; ++y) {
      for (std::uint32_t x = 0u; x < width; ++x) {
        const float value = at(x, y);
        EXPECT_LE(std::abs(value), magnitude + 1.0e-7f);
        if (x > 0u) {
          EXPECT_LE(
            std::abs(value - at(x - 1u, y)),
            horizontal_step + 2.0e-7f
          ) << "horizontal x=" << x << ", y=" << y;
        }
        if (y > 0u) {
          EXPECT_LE(
            std::abs(value - at(x, y - 1u)),
            vertical_step + 2.0e-7f
          ) << "vertical x=" << x << ", y=" << y;
        }
      }
    }

    // Rectangles use the local full-resolution analysis domain. Scaling the source-pixel
    // dimensions and every rectangle together (as for an ROI-local crop) must produce the same
    // final model-grid field without any full-desktop width factor.
    const std::array roi_local_rectangles {
      models::depth_source_rect_t {600u, 200u, 1000u, 600u},
      models::depth_source_rect_t {200u, 50u, 400u, 150u},
      models::depth_source_rect_t {900u, 400u, 1200u, 550u},
    };
    std::vector<float> roi_local_output;
    ASSERT_TRUE(fixture.dispatch(
      width,
      height,
      analysis_width / 2u,
      analysis_height / 2u,
      input,
      roi_local_rectangles,
      roi_local_output,
      error
    )) << error;
    EXPECT_EQ(roi_local_output, output);
  }

  // The collar clamps only where required. If the original field is already nearer zero than the
  // distance budget, even a texel beside the guard remains byte-for-byte unchanged.
  constexpr float near_zero = 0.005f;
  const std::vector<float> near_input(
    static_cast<std::size_t>(width) * height,
    near_zero
  );
  std::vector<float> near_output;
  ASSERT_TRUE(fixture.dispatch(
    width,
    height,
    analysis_width,
    analysis_height,
    near_input,
    rectangles,
    near_output,
    error
  )) << error;
  EXPECT_FLOAT_EQ(near_output[8u * width + 10u], near_zero);
  EXPECT_FLOAT_EQ(near_output[8u * width + 11u], 0.0f);
  EXPECT_FLOAT_EQ(near_output[8u * width + 7u], near_zero);
}

TEST(HostSbsOverlayZeroPlaneGpuTest, ShaderCapacityMatchesFixedCpuPlanCapacity) {
  constexpr std::uint32_t width = 16u;
  constexpr std::uint32_t height = 8u;
  const std::vector<float> input(
    static_cast<std::size_t>(width) * height,
    0.02f
  );
  std::array<models::depth_source_rect_t, models::host_sbs_overlay_max_loose_rects>
    full_capacity;
  std::fill(
    full_capacity.begin(),
    full_capacity.end(),
    models::depth_source_rect_t {400u, 200u, 1200u, 600u}
  );

  overlay_zero_plane_fixture_t fixture;
  std::string error;
  ASSERT_TRUE(fixture.initialize(error)) << error;
  std::vector<float> output;
  ASSERT_TRUE(fixture.dispatch(
    width,
    height,
    1600u,
    800u,
    input,
    full_capacity,
    output,
    error
  )) << error;
  EXPECT_FLOAT_EQ(output[4u * width + 8u], 0.0f);

  std::array<
    models::depth_source_rect_t,
    models::host_sbs_overlay_max_loose_rects + 1u
  > over_capacity;
  std::fill(
    over_capacity.begin(),
    over_capacity.end(),
    models::depth_source_rect_t {400u, 200u, 1200u, 600u}
  );
  ASSERT_TRUE(fixture.dispatch(
    width,
    height,
    1600u,
    800u,
    input,
    over_capacity,
    output,
    error
  )) << error;
  EXPECT_EQ(output, input);
}

#endif  // _WIN32
