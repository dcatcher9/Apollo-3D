/**
 * @file tests/unit/test_prod_zipdepth_convex2x_shader.cpp
 * @brief Standalone compile and D3D11 WARP contract tests for ZipDepth convex 2x.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include <algorithm>
  #include <array>
  #include <cmath>
  #include <cstddef>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <d3d11shader.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <limits>
  #include <string>
  #include <string_view>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;

  constexpr std::string_view shader_filename = "prod_zipdepth_convex2x_cs.hlsl";
  constexpr std::string_view shader_entrypoint = "main";
  constexpr std::string_view shader_target = "cs_5_0";
  constexpr UINT shader_compile_flags =
    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;

  ComPtr<ID3DBlob> compile_shader(std::string &diagnostics) {
    const auto path =
      std::filesystem::path(SUNSHINE_SHADERS_DIR) / shader_filename;
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> errors;
    const auto status = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      shader_entrypoint.data(),
      shader_target.data(),
      shader_compile_flags,
      0u,
      &bytecode,
      &errors
    );
    if (errors) {
      diagnostics.assign(
        static_cast<const char *>(errors->GetBufferPointer()),
        errors->GetBufferSize()
      );
    }
    return SUCCEEDED(status) ? bytecode : nullptr;
  }

  struct convex2x_constants_t {
    std::uint32_t coarse_width;
    std::uint32_t coarse_height;
    std::array<std::uint32_t, 2u> reserved;
  };

  static_assert(sizeof(convex2x_constants_t) == 16u);

  bool create_structured_srv(
    ID3D11Device *device,
    const std::vector<float> &values,
    ComPtr<ID3D11Buffer> &buffer,
    ComPtr<ID3D11ShaderResourceView> &srv
  ) {
    if (values.empty() || values.size() >
                            std::numeric_limits<UINT>::max() / sizeof(float)) {
      return false;
    }
    D3D11_BUFFER_DESC description {};
    description.ByteWidth = static_cast<UINT>(values.size() * sizeof(float));
    description.Usage = D3D11_USAGE_IMMUTABLE;
    description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    description.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    description.StructureByteStride = sizeof(float);
    D3D11_SUBRESOURCE_DATA initial {values.data(), 0u, 0u};
    return SUCCEEDED(device->CreateBuffer(&description, &initial, &buffer)) &&
           SUCCEEDED(device->CreateShaderResourceView(buffer.Get(), nullptr, &srv));
  }

  bool read_texture(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    ID3D11Texture2D *source,
    std::vector<float> &output
  ) {
    D3D11_TEXTURE2D_DESC source_description {};
    source->GetDesc(&source_description);
    auto staging_description = source_description;
    staging_description.Usage = D3D11_USAGE_STAGING;
    staging_description.BindFlags = 0u;
    staging_description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&staging_description, nullptr, &staging))) {
      return false;
    }
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return false;
    }
    output.resize(
      static_cast<std::size_t>(source_description.Width) * source_description.Height
    );
    for (std::uint32_t row = 0u; row < source_description.Height; ++row) {
      const auto *source_row = reinterpret_cast<const float *>(
        static_cast<const std::byte *>(mapped.pData) +
        static_cast<std::size_t>(row) * mapped.RowPitch
      );
      std::copy_n(
        source_row,
        source_description.Width,
        output.begin() + static_cast<std::size_t>(row) * source_description.Width
      );
    }
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  std::vector<float> convex2x_reference(
    const std::vector<float> &coarse,
    const std::vector<float> &logits,
    const std::uint32_t width,
    const std::uint32_t height
  ) {
    constexpr std::array<std::array<int, 2u>, 9u> offsets {{
      {-1, -1}, {0, -1}, {1, -1},
      {-1, 0}, {0, 0}, {1, 0},
      {-1, 1}, {0, 1}, {1, 1},
    }};
    const auto texels = static_cast<std::size_t>(width) * height;
    const auto refined_width = width * 2u;
    std::vector<float> output(
      static_cast<std::size_t>(refined_width) * height * 2u
    );
    for (std::uint32_t y = 0u; y < height; ++y) {
      for (std::uint32_t x = 0u; x < width; ++x) {
        const auto cell = static_cast<std::size_t>(y) * width + x;
        for (std::uint32_t subpixel = 0u; subpixel < 4u; ++subpixel) {
          const auto logit = [&](const std::uint32_t neighbor) {
            return logits[(neighbor * 4u + subpixel) * texels + cell];
          };
          float maximum = logit(0u);
          for (std::uint32_t neighbor = 1u; neighbor < 9u; ++neighbor) {
            maximum = std::max(maximum, logit(neighbor));
          }
          float weighted_depth = 0.0f;
          float weight_sum = 0.0f;
          for (std::uint32_t neighbor = 0u; neighbor < 9u; ++neighbor) {
            const float weight = std::exp(logit(neighbor) - maximum);
            const auto sample_x = static_cast<std::uint32_t>(std::clamp(
              static_cast<int>(x) + offsets[neighbor][0],
              0,
              static_cast<int>(width) - 1
            ));
            const auto sample_y = static_cast<std::uint32_t>(std::clamp(
              static_cast<int>(y) + offsets[neighbor][1],
              0,
              static_cast<int>(height) - 1
            ));
            weighted_depth += weight * coarse[static_cast<std::size_t>(sample_y) * width + sample_x];
            weight_sum += weight;
          }
          const auto output_x = x * 2u + (subpixel & 1u);
          const auto output_y = y * 2u + (subpixel >> 1u);
          output[static_cast<std::size_t>(output_y) * refined_width + output_x] =
            std::max(weighted_depth / weight_sum, 0.0f);
        }
      }
    }
    return output;
  }

  TEST(ProdZipDepthConvex2xShaderTest, CompilesWithStandaloneResourceContract) {
    std::string diagnostics;
    const auto bytecode = compile_shader(diagnostics);
    ASSERT_TRUE(bytecode) << diagnostics;

    // MinGW declares this D3DCompiler IID as an external symbol. Keep the D3DCompiler 47 value
    // local, matching the production shader cache, so the test does not need dxguid.lib.
    constexpr IID shader_reflection_iid {
      0x8d536ca1u, 0x0ccau, 0x4956u,
      {0xa8u, 0x37u, 0x78u, 0x69u, 0x63u, 0x75u, 0x55u, 0x84u}};
    ComPtr<ID3D11ShaderReflection> reflection;
    ASSERT_TRUE(SUCCEEDED(D3DReflect(
      bytecode->GetBufferPointer(),
      bytecode->GetBufferSize(),
      shader_reflection_iid,
      reinterpret_cast<void **>(reflection.GetAddressOf())
    )));
    ASSERT_TRUE(reflection);

    D3D11_SHADER_DESC shader_description {};
    ASSERT_TRUE(SUCCEEDED(reflection->GetDesc(&shader_description)));
    ASSERT_EQ(shader_description.BoundResources, 4u);
    struct expected_binding_t {
      std::string_view name;
      D3D_SHADER_INPUT_TYPE type;
      UINT bind_point;
    };
    constexpr std::array expected_bindings {
      expected_binding_t {"Convex2xConstants", D3D_SIT_CBUFFER, 0u},
      expected_binding_t {"CoarseDepth", D3D_SIT_STRUCTURED, 0u},
      expected_binding_t {"ConvexLogits", D3D_SIT_STRUCTURED, 1u},
      expected_binding_t {"RefinedDepth", D3D_SIT_UAV_RWTYPED, 0u},
    };
    for (const auto &expected : expected_bindings) {
      D3D11_SHADER_INPUT_BIND_DESC binding {};
      ASSERT_TRUE(SUCCEEDED(reflection->GetResourceBindingDescByName(
        expected.name.data(), &binding
      ))) << expected.name;
      EXPECT_EQ(binding.Type, expected.type) << expected.name;
      EXPECT_EQ(binding.BindPoint, expected.bind_point) << expected.name;
      EXPECT_EQ(binding.BindCount, 1u) << expected.name;
    }
    UINT threads_x = 0u;
    UINT threads_y = 0u;
    UINT threads_z = 0u;
    EXPECT_EQ(reflection->GetThreadGroupSize(&threads_x, &threads_y, &threads_z), 64u);
    EXPECT_EQ(threads_x, 8u);
    EXPECT_EQ(threads_y, 8u);
    EXPECT_EQ(threads_z, 1u);

    auto *constants = reflection->GetConstantBufferByName("Convex2xConstants");
    ASSERT_NE(constants, nullptr);
    D3D11_SHADER_BUFFER_DESC constants_description {};
    ASSERT_TRUE(SUCCEEDED(constants->GetDesc(&constants_description)));
    EXPECT_EQ(constants_description.Size, sizeof(convex2x_constants_t));
  }

  TEST(ProdZipDepthConvex2xShaderTest, CompilesAndMatchesStableReferenceOnWarp) {
    constexpr std::uint32_t width = 3u;
    constexpr std::uint32_t height = 2u;
    constexpr std::size_t texels = width * height;
    const std::vector<float> coarse {-8.0f, -6.0f, -4.0f, 2.0f, 4.0f, 6.0f};
    std::vector<float> logits(36u * texels);
    for (std::uint32_t y = 0u; y < height; ++y) {
      for (std::uint32_t x = 0u; x < width; ++x) {
        const auto cell = static_cast<std::size_t>(y) * width + x;
        for (std::uint32_t subpixel = 0u; subpixel < 4u; ++subpixel) {
          const auto preferred = static_cast<std::uint32_t>(
            (cell + subpixel * 2u) % 9u
          );
          for (std::uint32_t neighbor = 0u; neighbor < 9u; ++neighbor) {
            const float relative = neighbor == preferred ?
                                     5.0f :
                                     -0.37f * static_cast<float>(neighbor) -
                                       0.11f * static_cast<float>(cell);
            logits[(neighbor * 4u + subpixel) * texels + cell] =
              10000.0f + 1000.0f * static_cast<float>(subpixel) + relative;
          }
        }
      }
    }
    const auto expected = convex2x_reference(coarse, logits, width, height);
    EXPECT_TRUE(std::any_of(expected.begin(), expected.end(), [](const float value) {
      return value == 0.0f;
    }));
    EXPECT_TRUE(std::any_of(expected.begin(), expected.end(), [](const float value) {
      return value > 0.0f;
    }));

    constexpr D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL actual {};
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
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
    )));
    ASSERT_GE(actual, D3D_FEATURE_LEVEL_11_0);

    std::string diagnostics;
    const auto bytecode = compile_shader(diagnostics);
    ASSERT_TRUE(bytecode) << diagnostics;
    ComPtr<ID3D11ComputeShader> shader;
    ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
      bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shader
    )));

    ComPtr<ID3D11Buffer> coarse_buffer;
    ComPtr<ID3D11ShaderResourceView> coarse_srv;
    ASSERT_TRUE(create_structured_srv(device.Get(), coarse, coarse_buffer, coarse_srv));
    ComPtr<ID3D11Buffer> logits_buffer;
    ComPtr<ID3D11ShaderResourceView> logits_srv;
    ASSERT_TRUE(create_structured_srv(device.Get(), logits, logits_buffer, logits_srv));

    const convex2x_constants_t constants {width, height, {0u, 0u}};
    D3D11_BUFFER_DESC constant_description {};
    constant_description.ByteWidth = sizeof(constants);
    constant_description.Usage = D3D11_USAGE_IMMUTABLE;
    constant_description.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA constant_initial {&constants, 0u, 0u};
    ComPtr<ID3D11Buffer> constant_buffer;
    ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(
      &constant_description, &constant_initial, &constant_buffer
    )));

    D3D11_TEXTURE2D_DESC output_description {};
    output_description.Width = width * 2u;
    output_description.Height = height * 2u;
    output_description.MipLevels = 1u;
    output_description.ArraySize = 1u;
    output_description.Format = DXGI_FORMAT_R32_FLOAT;
    output_description.SampleDesc.Count = 1u;
    output_description.Usage = D3D11_USAGE_DEFAULT;
    output_description.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    ComPtr<ID3D11Texture2D> output_texture;
    ComPtr<ID3D11UnorderedAccessView> output_uav;
    ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &output_description, nullptr, &output_texture
    )));
    ASSERT_TRUE(SUCCEEDED(device->CreateUnorderedAccessView(
      output_texture.Get(), nullptr, &output_uav
    )));

    ID3D11ShaderResourceView *srvs[] {coarse_srv.Get(), logits_srv.Get()};
    ID3D11UnorderedAccessView *uavs[] {output_uav.Get()};
    ID3D11Buffer *constant_buffers[] {constant_buffer.Get()};
    context->CSSetShader(shader.Get(), nullptr, 0u);
    context->CSSetShaderResources(0u, static_cast<UINT>(std::size(srvs)), srvs);
    context->CSSetUnorderedAccessViews(0u, static_cast<UINT>(std::size(uavs)), uavs, nullptr);
    context->CSSetConstantBuffers(
      0u,
      static_cast<UINT>(std::size(constant_buffers)),
      constant_buffers
    );
    context->Dispatch(1u, 1u, 1u);

    ID3D11ShaderResourceView *null_srvs[] {nullptr, nullptr};
    ID3D11UnorderedAccessView *null_uavs[] {nullptr};
    context->CSSetShaderResources(
      0u,
      static_cast<UINT>(std::size(null_srvs)),
      null_srvs
    );
    context->CSSetUnorderedAccessViews(
      0u,
      static_cast<UINT>(std::size(null_uavs)),
      null_uavs,
      nullptr
    );
    context->CSSetShader(nullptr, nullptr, 0u);

    std::vector<float> actual_output;
    ASSERT_TRUE(read_texture(
      device.Get(), context.Get(), output_texture.Get(), actual_output
    ));
    ASSERT_EQ(actual_output.size(), expected.size());
    for (std::size_t index = 0u; index < expected.size(); ++index) {
      EXPECT_TRUE(std::isfinite(actual_output[index])) << "output index " << index;
      EXPECT_NEAR(actual_output[index], expected[index], 2.0e-5f)
        << "output index " << index;
    }
  }
}  // namespace

#endif  // _WIN32
