/**
 * @file tests/unit/test_stream_gamma_shader.cpp
 * @brief WARP checks of the production stream gamma shader against pinned Gloam output.
 */
#include "../tests_common.h"

#ifdef _WIN32

  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <limits>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  using pixel_t = std::array<float, 4>;

  struct gamma_constants_t {
    std::uint32_t mode;
    float white_scrgb;
    std::array<float, 2> padding {};
  };

  static_assert(sizeof(gamma_constants_t) == 16);

  struct sdr_constants_t {
    std::uint32_t target_bt2020 = 0;
    std::uint32_t source_is_hdr = 0;
    float source_white_scrgb = 2.5f;
    std::uint32_t target_is_hdr = 0;
  };

  static_assert(sizeof(sdr_constants_t) == 16);

  class StreamGammaShaderTest: public testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL requested = D3D_FEATURE_LEVEL_11_0;
      D3D_FEATURE_LEVEL actual {};
      ASSERT_EQ(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, &requested, 1, D3D11_SDK_VERSION, &device, &actual, &context), S_OK);
      ASSERT_EQ(actual, requested);
    }

    template<class T>
    ComPtr<ID3D11Buffer> constant_buffer(const T &value) {
      D3D11_BUFFER_DESC description {};
      description.ByteWidth = sizeof(T);
      description.Usage = D3D11_USAGE_IMMUTABLE;
      description.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA data {&value, 0, 0};
      ComPtr<ID3D11Buffer> buffer;
      EXPECT_EQ(device->CreateBuffer(&description, &data, &buffer), S_OK);
      return buffer;
    }

    // This compiles the real production include. No gamma math is reproduced in the fixture.
    // Every pair contains the new conversion and the existing uncorrected conversion evaluated
    // in the same dispatch, so the default check compares actual GPU float bits.
    std::vector<pixel_t> evaluate(
      const std::vector<pixel_t> &pixels,
      std::uint32_t mode,
      float white_scrgb = 2.5f,
      const std::string &include = "include/stream_gamma.hlsl",
      const std::string &conversion = "StreamGammaScRGBToPQ(input)",
      const std::string &legacy = "scRGBTo2100PQ(input)",
      const sdr_constants_t &sdr = {}
    ) {
      const std::string common_include = include == "include/stream_gamma.hlsl" ? "#include \"include/common.hlsl\"\n" : "";
      const std::string source = common_include +
                                 "#include \"" + include +
                                 "\"\n"
                                 "StructuredBuffer<float4> source_pixels : register(t0);\n"
                                 "RWStructuredBuffer<float4> result_pixels : register(u0);\n"
                                 "[numthreads(64, 1, 1)] void main_cs(uint3 thread : SV_DispatchThreadID) {\n"
                                 "  uint count, stride; source_pixels.GetDimensions(count, stride);\n"
                                 "  if (thread.x >= count) return;\n"
                                 "  float3 input = source_pixels[thread.x].rgb;\n"
                                 "  result_pixels[2 * thread.x] = float4(" +
                                 conversion +
                                 ", 1);\n"
                                 "  result_pixels[2 * thread.x + 1] = float4(" +
                                 legacy +
                                 ", 1);\n"
                                 "}\n";
      const auto source_path = (std::filesystem::path(SUNSHINE_SHADERS_DIR) /
                                "stream_gamma_test_inline.hlsl")
                                 .string();
      ComPtr<ID3DBlob> bytecode, errors;
      const auto compiled = D3DCompile(
        source.data(),
        source.size(),
        source_path.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main_cs",
        "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &bytecode,
        &errors
      );
      EXPECT_EQ(compiled, S_OK) << (errors ? std::string(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize()) : "");
      if (!bytecode) {
        return {};
      }
      ComPtr<ID3D11ComputeShader> shader;
      EXPECT_EQ(device->CreateComputeShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shader), S_OK);
      if (!shader) {
        return {};
      }

      D3D11_BUFFER_DESC description {};
      description.ByteWidth = static_cast<UINT>(pixels.size() * sizeof(pixel_t));
      description.Usage = D3D11_USAGE_IMMUTABLE;
      description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      description.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      description.StructureByteStride = sizeof(pixel_t);
      D3D11_SUBRESOURCE_DATA data {pixels.data(), 0, 0};
      ComPtr<ID3D11Buffer> input_buffer;
      EXPECT_EQ(device->CreateBuffer(&description, &data, &input_buffer), S_OK);
      ComPtr<ID3D11ShaderResourceView> input_view;
      EXPECT_EQ(device->CreateShaderResourceView(input_buffer.Get(), nullptr, &input_view), S_OK);

      description.ByteWidth *= 2;
      description.Usage = D3D11_USAGE_DEFAULT;
      description.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      ComPtr<ID3D11Buffer> output_buffer;
      EXPECT_EQ(device->CreateBuffer(&description, nullptr, &output_buffer), S_OK);
      ComPtr<ID3D11UnorderedAccessView> output_view;
      EXPECT_EQ(device->CreateUnorderedAccessView(output_buffer.Get(), nullptr, &output_view), S_OK);
      if (!input_view || !output_view) {
        return {};
      }

      auto gamma_buffer = constant_buffer(gamma_constants_t {mode, white_scrgb});
      auto sdr_buffer = constant_buffer(sdr);
      context->CSSetConstantBuffers(4, 1, gamma_buffer.GetAddressOf());
      context->CSSetConstantBuffers(1, 1, sdr_buffer.GetAddressOf());
      context->CSSetShaderResources(0, 1, input_view.GetAddressOf());
      context->CSSetUnorderedAccessViews(0, 1, output_view.GetAddressOf(), nullptr);
      context->CSSetShader(shader.Get(), nullptr, 0);
      context->Dispatch(static_cast<UINT>((pixels.size() + 63) / 64), 1, 1);
      ID3D11ShaderResourceView *no_input = nullptr;
      ID3D11UnorderedAccessView *no_output = nullptr;
      context->CSSetShaderResources(0, 1, &no_input);
      context->CSSetUnorderedAccessViews(0, 1, &no_output, nullptr);

      description.BindFlags = 0;
      description.MiscFlags = 0;
      description.StructureByteStride = 0;
      description.Usage = D3D11_USAGE_STAGING;
      description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Buffer> staging;
      EXPECT_EQ(device->CreateBuffer(&description, nullptr, &staging), S_OK);
      if (!staging) {
        return {};
      }
      context->CopyResource(staging.Get(), output_buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const auto map_result = context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
      EXPECT_EQ(map_result, S_OK);
      if (FAILED(map_result)) {
        return {};
      }
      std::vector<pixel_t> result(pixels.size() * 2);
      std::memcpy(result.data(), mapped.pData, result.size() * sizeof(pixel_t));
      context->Unmap(staging.Get(), 0);
      return result;
    }

    static void expect_equal_bits(const pixel_t &actual, const pixel_t &expected) {
      for (std::size_t channel = 0; channel < actual.size(); ++channel) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[channel]), std::bit_cast<std::uint32_t>(expected[channel])) << "channel " << channel;
      }
    }

    float draw_luma(
      const char *pixel_shader_file,
      DXGI_FORMAT source_format,
      const void *source_pixel,
      UINT pixel_bytes,
      std::uint32_t mode,
      const sdr_constants_t &sdr
    ) {
      const auto failed = std::numeric_limits<float>::quiet_NaN();
      const auto compile = [](const char *file, const char *entry, const char *profile) {
        const auto path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / file;
        ComPtr<ID3DBlob> code, errors;
        const auto status = D3DCompileFromFile(
          path.c_str(),
          nullptr,
          D3D_COMPILE_STANDARD_FILE_INCLUDE,
          entry,
          profile,
          D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
          0,
          &code,
          &errors
        );
        EXPECT_EQ(status, S_OK) << file << ": " << (errors ? std::string(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize()) : "");
        return code;
      };
      auto vs_code = compile("convert_yuv420_planar_y_vs.hlsl", "main_vs", "vs_5_0");
      auto ps_code = compile(pixel_shader_file, "main_ps", "ps_5_0");
      if (!vs_code || !ps_code) {
        return failed;
      }
      ComPtr<ID3D11VertexShader> vs;
      ComPtr<ID3D11PixelShader> ps;
      EXPECT_EQ(device->CreateVertexShader(vs_code->GetBufferPointer(), vs_code->GetBufferSize(), nullptr, &vs), S_OK);
      EXPECT_EQ(device->CreatePixelShader(ps_code->GetBufferPointer(), ps_code->GetBufferSize(), nullptr, &ps), S_OK);
      if (!vs || !ps) {
        return failed;
      }

      D3D11_TEXTURE2D_DESC description {};
      description.Width = description.Height = description.MipLevels = description.ArraySize = description.SampleDesc.Count = 1;
      description.Format = source_format;
      description.Usage = D3D11_USAGE_IMMUTABLE;
      description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA data {source_pixel, pixel_bytes, 0};
      ComPtr<ID3D11Texture2D> input;
      EXPECT_EQ(device->CreateTexture2D(&description, &data, &input), S_OK);
      ComPtr<ID3D11ShaderResourceView> srv;
      EXPECT_EQ(device->CreateShaderResourceView(input.Get(), nullptr, &srv), S_OK);
      description.Format = DXGI_FORMAT_R32_FLOAT;
      description.Usage = D3D11_USAGE_DEFAULT;
      description.BindFlags = D3D11_BIND_RENDER_TARGET;
      ComPtr<ID3D11Texture2D> output;
      EXPECT_EQ(device->CreateTexture2D(&description, nullptr, &output), S_OK);
      ComPtr<ID3D11RenderTargetView> rtv;
      EXPECT_EQ(device->CreateRenderTargetView(output.Get(), nullptr, &rtv), S_OK);
      if (!srv || !rtv) {
        return failed;
      }

      // A neutral full-range BT.2020 Y matrix preserves grayscale PQ code values.
      auto color = constant_buffer(std::array<float, 16> {
        0.2627f,
        0.6780f,
        0.0593f,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
      });
      auto rotation = constant_buffer(std::array<std::uint32_t, 4> {});
      auto gamma = constant_buffer(gamma_constants_t {mode, 2.5f});
      auto source_state = constant_buffer(sdr);
      context->VSSetConstantBuffers(1, 1, rotation.GetAddressOf());
      context->PSSetConstantBuffers(0, 1, color.GetAddressOf());
      context->PSSetConstantBuffers(1, 1, source_state.GetAddressOf());
      context->PSSetConstantBuffers(4, 1, gamma.GetAddressOf());
      D3D11_SAMPLER_DESC sampler_description {};
      sampler_description.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
      sampler_description.AddressU = sampler_description.AddressV = sampler_description.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_description.MaxLOD = D3D11_FLOAT32_MAX;
      ComPtr<ID3D11SamplerState> sampler;
      EXPECT_EQ(device->CreateSamplerState(&sampler_description, &sampler), S_OK);
      context->PSSetSamplers(0, 1, sampler.GetAddressOf());
      context->PSSetShaderResources(0, 1, srv.GetAddressOf());
      context->OMSetRenderTargets(1, rtv.GetAddressOf(), nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      context->VSSetShader(vs.Get(), nullptr, 0);
      context->PSSetShader(ps.Get(), nullptr, 0);
      constexpr D3D11_VIEWPORT viewport {0, 0, 1, 1, 0, 1};
      context->RSSetViewports(1, &viewport);
      context->Draw(3, 0);
      context->OMSetRenderTargets(0, nullptr, nullptr);
      ID3D11ShaderResourceView *no_source = nullptr;
      context->PSSetShaderResources(0, 1, &no_source);

      description.BindFlags = 0;
      description.Usage = D3D11_USAGE_STAGING;
      description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      EXPECT_EQ(device->CreateTexture2D(&description, nullptr, &staging), S_OK);
      if (!staging) {
        return failed;
      }
      context->CopyResource(staging.Get(), output.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const auto status = context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
      EXPECT_EQ(status, S_OK);
      if (FAILED(status)) {
        return failed;
      }
      const auto result = *static_cast<const float *>(mapped.pData);
      context->Unmap(staging.Get(), 0);
      return result;
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
  };
}  // namespace

TEST_F(StreamGammaShaderTest, WindowsDefaultPreservesOriginalGpuBits) {
  const std::vector<pixel_t> pixels {
    {0, 0, 0, 1},
    {0.0001f, 0.01f, 0.5f, 1},
    {2.5f, 2.5f, 2.5f, 1},
    {7.5f, 12.5f, 25, 1},
    {125, 125, 125, 1},
    {0, 2.5f, 0, 1},
    {50, -4, -0.5f, 1},
    {-0.5f, 0.5f, 4, 1},
  };
  for (const float white : {1.0f, 2.5f, 6.0f, 12.5f}) {
    const auto result = evaluate(pixels, 0, white);
    ASSERT_EQ(result.size(), pixels.size() * 2);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
      SCOPED_TRACE(i);
      expect_equal_bits(result[2 * i], result[2 * i + 1]);
    }
  }
}

TEST_F(StreamGammaShaderTest, GrayscaleMatchesPinnedGloamTransferFunctions) {
  // Reference values were generated by the ORIGINAL Release Gloam.Core.dll, commit
  // 79d4c471bc5e8802fd4288712014776af023b4d2, TransferFunctions.cs and LutGenerator.cs.
  // Input is Windows' SDR-in-HDR mapping at W=200 nits (scRGB 1 = 80 nits).
  // The golden columns are PQ(200 * v^2.2) and PQ(200 * v^2.4), obtained with that DLL.
  // These fixed references intentionally do not call a C++ reimplementation of gamma.
  struct reference_t {
    float scrgb;
    float pq22;
    float pq24;
  };

  constexpr std::array reference {
    reference_t {0, 0, 0},
    reference_t {0.0019349845201238392f, 0.019154070073912767f, 0.011854362920266608f},
    reference_t {0.009839848760222418f, 0.09380005751721572f, 0.0739420232512888f},
    reference_t {0.025057063937172597f, 0.1621324593538086f, 0.1386397108312097f},
    reference_t {0.12719022042889197f, 0.2954490402670287f, 0.27420468441736606f},
    reference_t {0.5351028512055814f, 0.4271855303108265f, 0.41425832744138136f},
    reference_t {1.3063038849209803f, 0.514137206642564f, 0.5083504518255105f},
    reference_t {2.5f, 0.5791332452435196f, 0.5791332452435196f},
  };
  std::vector<pixel_t> pixels;
  for (const auto &sample : reference) {
    pixels.push_back({sample.scrgb, sample.scrgb, sample.scrgb, 1});
  }
  for (const std::uint32_t mode : {1u, 2u}) {
    const auto result = evaluate(pixels, mode);
    ASSERT_EQ(result.size(), pixels.size() * 2);
    for (std::size_t i = 0; i < reference.size(); ++i) {
      SCOPED_TRACE(i);
      const auto expected = mode == 1 ? reference[i].pq22 : reference[i].pq24;
      for (std::size_t channel = 0; channel < 3; ++channel) {
        EXPECT_NEAR(result[2 * i][channel], expected, 2e-5f);
      }
    }
    // Gloam 2.2 darkens the toe but slightly raises the 50% SDR signal. A blanket
    // pow(linear, gamma) or a monotone "always darker" correction fails this check.
    EXPECT_LT(result[2 * 3][0], result[2 * 3 + 1][0]);
    if (mode == 1) {
      EXPECT_GT(result[2 * 5][0], result[2 * 5 + 1][0]);
    }
  }
}

TEST_F(StreamGammaShaderTest, UsesCapturedDisplayReferenceWhite) {
  struct reference_t {
    float white_nits;
    float scrgb;
    float pq22;
    float pq24;
  };

  constexpr std::array reference {
    reference_t {80, 0.010022825574869039f, 0.11786645207248375f, 0.09928300368231763f},
    reference_t {203, 0.025432919896230182f, 0.16293480956399897f, 0.1393594361390526f},
    reference_t {480, 0.060136953449214235f, 0.2139384230686108f, 0.18551084769889167f},
    reference_t {1000, 0.12528531968586298f, 0.2646169315754133f, 0.23205364467108966f},
  };
  for (const auto &sample : reference) {
    for (const std::uint32_t mode : {1u, 2u}) {
      const auto result = evaluate({{sample.scrgb, sample.scrgb, sample.scrgb, 1}}, mode, sample.white_nits / 80);
      ASSERT_EQ(result.size(), 2);
      for (std::size_t channel = 0; channel < 3; ++channel) {
        EXPECT_NEAR(result[0][channel], mode == 1 ? sample.pq22 : sample.pq24, 2e-5f);
      }
    }
  }
}

TEST_F(StreamGammaShaderTest, CorrectsChannelsInRec2020WireBasis) {
  // Each RGB channel was transformed to Rec.2020 using the production primary matrix,
  // then evaluated by the pinned original Gloam 1024-entry LUT with linear interpolation.
  // The tolerance covers its finite LUT interpolation and the GPU's float PQ evaluation.
  // The mixed highlight has a red channel above SDR white and two below it. Correction
  // before gamut conversion or one luminance-derived gain would produce different colors.
  struct reference_t {
    pixel_t scrgb;
    std::array<float, 3> pq22;
    std::array<float, 3> pq24;
  };

  constexpr std::array reference {
    reference_t {{2.5f, 0, 0, 1}, {0.532328013448387f, 0.32251895837938366f, 0.20059880775473599f}, {0.5281322816279889f, 0.30263357598046703f, 0.17679210992723016f}},
    reference_t {{0, 2.5f, 0, 1}, {0.4687476661886612f, 0.570669173373865f, 0.34433902253313386f}, {0.4591226525274182f, 0.5699013949799259f, 0.32570148304484514f}},
    reference_t {{0, 0, 2.5f, 1}, {0.28143772685544627f, 0.17176056178556764f, 0.5680078625658003f}, {0.2595826036239447f, 0.1480947743797236f, 0.5669993970142224f}},
    reference_t {{12.5f, 1.25f, 0.25f, 1}, {0.7068219970409981f, 0.5574532267350282f, 0.4278529113246433f}, {0.7068219970409981f, 0.5554939053167449f, 0.414976808106558f}},
    reference_t {{50, -4, -0.5f, 1}, {0.8473791561224612f, 0, 0.144325573861256f}, {0.8473791561224612f, 0, 0.12134375469021048f}},
  };
  std::vector<pixel_t> pixels;
  for (const auto &sample : reference) {
    pixels.push_back(sample.scrgb);
  }
  for (const std::uint32_t mode : {1u, 2u}) {
    const auto result = evaluate(pixels, mode);
    ASSERT_EQ(result.size(), pixels.size() * 2);
    for (std::size_t i = 0; i < reference.size(); ++i) {
      SCOPED_TRACE(i);
      const auto &expected = mode == 1 ? reference[i].pq22 : reference[i].pq24;
      for (std::size_t channel = 0; channel < 3; ++channel) {
        EXPECT_NEAR(result[2 * i][channel], expected[channel], 2e-5f);
        EXPECT_TRUE(std::isfinite(result[2 * i][channel]));
      }
    }
  }
}

TEST_F(StreamGammaShaderTest, PreservesHighlightRangeAndMonotonicity) {
  std::vector<pixel_t> pixels;
  for (std::size_t i = 0; i <= 1024; ++i) {
    const auto scrgb = static_cast<float>(i) / 1024 * 2.5f;
    pixels.push_back({scrgb, scrgb, scrgb, 1});
  }
  for (const float nits : {201, 400, 600, 1000, 2000, 10000}) {
    pixels.push_back({nits / 80, nits / 80, nits / 80, 1});
  }
  for (const std::uint32_t mode : {1u, 2u}) {
    const auto result = evaluate(pixels, mode);
    ASSERT_EQ(result.size(), pixels.size() * 2);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
      for (std::size_t channel = 0; channel < 3; ++channel) {
        SCOPED_TRACE(i);
        const auto actual = result[2 * i][channel];
        EXPECT_TRUE(std::isfinite(actual));
        EXPECT_GE(actual, 0);
        EXPECT_LE(actual, 1);
        if (i > 0) {
          EXPECT_GE(actual, result[2 * (i - 1)][channel]);
        }
        if (i >= 1024) {
          EXPECT_NEAR(actual, result[2 * i + 1][channel], 2e-6f);
        }
      }
    }
    EXPECT_LE(result[0][0], 1e-6f);  // PQ's encoded zero rounds to black in P010.
  }
}

TEST_F(StreamGammaShaderTest, BgraHdrConversionUsesTheSameCorrection) {
  const std::vector<pixel_t> pixels {{0.1f, 0.1f, 0.1f, 1}, {0.5f, 0.5f, 0.5f, 1}, {1, 1, 1, 1}};
  sdr_constants_t hdr;
  hdr.target_is_hdr = 1;
  constexpr std::array<std::array<float, 3>, 2> expected {{
    {0.1621324593538086f, 0.4271855303108265f, 0.5791332452435196f},
    {0.1386397108312097f, 0.41425832744138136f, 0.5791332452435196f},
  }};
  for (const std::uint32_t mode : {0u, 1u, 2u}) {
    const auto result = evaluate(pixels, mode, 2.5f, "include/convert_base.hlsl", "CONVERT_FUNCTION(input)", "scRGBTo2100PQ(RemoveSRGBCurve(input) * source_sdr_white_scrgb)", hdr);
    ASSERT_EQ(result.size(), pixels.size() * 2);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
      if (mode == 0) {
        expect_equal_bits(result[2 * i], result[2 * i + 1]);
      } else {
        for (std::size_t channel = 0; channel < 3; ++channel) {
          EXPECT_NEAR(result[2 * i][channel], expected[mode - 1][i], 2e-5f);
        }
      }
    }
  }
}

TEST_F(StreamGammaShaderTest, SdrConverterOutputIsIndependentOfHdrGammaSetting) {
  const std::vector<pixel_t> pixels {{0, 0, 0, 1}, {0.01f, 0.05f, 0.1f, 1}, {0.25f, 0.5f, 0.75f, 1}, {1, 1, 1, 1}, {4, 2, 0.5f, 1}};
  for (const std::string include : {"include/convert_base.hlsl", "include/convert_linear_base.hlsl"}) {
    for (const std::uint32_t bt2020 : {0u, 1u}) {
      sdr_constants_t sdr;
      sdr.target_bt2020 = bt2020;
      sdr.source_is_hdr = 1;
      const auto original = evaluate(pixels, 0, 2.5f, include, "CONVERT_FUNCTION(input)", "CONVERT_FUNCTION(input)", sdr);
      ASSERT_EQ(original.size(), pixels.size() * 2);
      for (const std::uint32_t mode : {1u, 2u}) {
        const auto adjusted = evaluate(pixels, mode, 2.5f, include, "CONVERT_FUNCTION(input)", "CONVERT_FUNCTION(input)", sdr);
        ASSERT_EQ(adjusted.size(), original.size());
        for (std::size_t i = 0; i < adjusted.size(); ++i) {
          expect_equal_bits(adjusted[i], original[i]);
        }
      }
    }
  }
}

TEST_F(StreamGammaShaderTest, ProductionLumaDrawsMatchForBgraAndFp16Hdr) {
  const std::array<std::uint8_t, 4> bgra {128, 128, 128, 255};
  // Half-float representation of 2.5 * original Gloam SrgbEotf(128 / 255).
  const std::array<std::uint16_t, 4> scrgb {0x3851, 0x3851, 0x3851, 0x3c00};
  sdr_constants_t hdr;
  hdr.source_is_hdr = hdr.target_is_hdr = 1;
  constexpr std::array expected {0.427994097552649f, 0.415128806444964f};
  for (const std::uint32_t mode : {1u, 2u}) {
    const auto bgra_y = draw_luma("convert_yuv420_planar_y_ps.hlsl", DXGI_FORMAT_B8G8R8A8_UNORM, bgra.data(), sizeof(bgra), mode, hdr);
    const auto fp16_y = draw_luma("convert_yuv420_planar_y_ps_perceptual_quantizer.hlsl", DXGI_FORMAT_R16G16B16A16_FLOAT, scrgb.data(), sizeof(scrgb), mode, hdr);
    EXPECT_NEAR(bgra_y, expected[mode - 1], 2e-5f);
    EXPECT_NEAR(fp16_y, bgra_y, 2e-4f);  // FP16 capture quantization is the only input difference.
  }
}

TEST_F(StreamGammaShaderTest, ProductionSdrLumaDrawsPreserveGpuBits) {
  const std::array<std::uint8_t, 4> bgra {24, 93, 192, 255};
  const std::array<std::uint16_t, 4> scrgb {0x3800, 0x3400, 0x3000, 0x3c00};
  for (const std::uint32_t bt2020 : {0u, 1u}) {
    sdr_constants_t sdr;
    sdr.target_bt2020 = bt2020;
    sdr.source_is_hdr = 1;
    const auto original_bgra = draw_luma("convert_yuv420_planar_y_ps.hlsl", DXGI_FORMAT_B8G8R8A8_UNORM, bgra.data(), sizeof(bgra), 0, sdr);
    const auto original_fp16 = draw_luma("convert_yuv420_planar_y_ps_linear.hlsl", DXGI_FORMAT_R16G16B16A16_FLOAT, scrgb.data(), sizeof(scrgb), 0, sdr);
    ASSERT_TRUE(std::isfinite(original_bgra));
    ASSERT_TRUE(std::isfinite(original_fp16));
    for (const std::uint32_t mode : {1u, 2u}) {
      const auto bgra_y = draw_luma("convert_yuv420_planar_y_ps.hlsl", DXGI_FORMAT_B8G8R8A8_UNORM, bgra.data(), sizeof(bgra), mode, sdr);
      const auto fp16_y = draw_luma("convert_yuv420_planar_y_ps_linear.hlsl", DXGI_FORMAT_R16G16B16A16_FLOAT, scrgb.data(), sizeof(scrgb), mode, sdr);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(bgra_y), std::bit_cast<std::uint32_t>(original_bgra));
      EXPECT_EQ(std::bit_cast<std::uint32_t>(fp16_y), std::bit_cast<std::uint32_t>(original_fp16));
    }
  }
}

TEST(StreamGammaShaderCompileTest, AllProductionConverterEntryPointsCompile) {
  constexpr std::array files {
    "convert_yuv420_planar_y_ps.hlsl",
    "convert_yuv420_planar_y_ps_linear.hlsl",
    "convert_yuv420_planar_y_ps_perceptual_quantizer.hlsl",
    "convert_yuv420_packed_uv_type0_ps.hlsl",
    "convert_yuv420_packed_uv_type0_ps_linear.hlsl",
    "convert_yuv420_packed_uv_type0_ps_perceptual_quantizer.hlsl",
    "convert_yuv420_packed_uv_type0s_ps.hlsl",
    "convert_yuv420_packed_uv_type0s_ps_linear.hlsl",
    "convert_yuv420_packed_uv_type0s_ps_perceptual_quantizer.hlsl",
    "sbs_reprojection_v2_p010_y_ps.hlsl",
  };
  for (const auto *file : files) {
    const auto path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / file;
    ComPtr<ID3DBlob> bytecode, errors;
    const char *entry = std::strcmp(file, "sbs_reprojection_v2_p010_y_ps.hlsl") == 0 ? "main_p010_y_ps" : "main_ps";
    const auto status = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      entry,
      "ps_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0,
      &bytecode,
      &errors
    );
    EXPECT_EQ(status, S_OK) << file << ": " << (errors ? std::string(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize()) : "");
    EXPECT_TRUE(bytecode) << file;
  }
}

#endif
