// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tests/unit/platform/test_game3d_host_conversion.cpp
 * @brief WARP checks of the Game 3D host conversion: the PQ wire transfer against the legacy
 *        FP16 scRGB path, and the cursor patch against full-frame cursor composition.
 */
#ifdef _WIN32
  #include "src/platform/windows/packed_output_fit.h"
  #include "src/platform/windows/sbs_cursor.h"

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstdint>
  #include <cstring>
  #include <d3d11_1.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <functional>
  #include <gtest/gtest.h>
  #include <map>
  #include <optional>
  #include <string>
  #include <tuple>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace cursor = platf::sbs_cursor;

  struct color_matrix_t {
    std::array<float, 4> y, u, v;
    std::array<float, 2> range_y, range_uv;
  };

  struct sdr_transform_t {
    std::uint32_t target_bt2020 = 0;
    std::uint32_t source_is_hdr = 0;
    float source_sdr_white_scrgb = 2.5f;
    std::uint32_t target_is_hdr = 0;
  };

  struct chroma_t {
    std::array<float, 2> subsample_offset {};
    std::uint32_t source_width = 0, source_height = 0, packed_eye_width = 0;
    std::array<std::uint32_t, 3> padding {};
  };

  struct gamma_t {
    std::uint32_t mode = 0;
    float white_scrgb = 2.5f;
    std::array<float, 2> padding {};
  };

  static_assert(sizeof(color_matrix_t) == 64 && sizeof(sdr_transform_t) == 16 && sizeof(chroma_t) == 32 && sizeof(gamma_t) == 16);

  // Full-range BT.2020 non-constant-luminance coefficients keep code values comparable.
  constexpr color_matrix_t bt2020_full {
    {0.2627f, 0.6780f, 0.0593f, 0.0f},
    {-0.2627f / 1.8814f, -0.6780f / 1.8814f, 0.5f, 0.5f},
    {0.5f, -0.6780f / 1.4746f, -0.0593f / 1.4746f, 0.5f},
    {1.0f, 0.0f},
    {1.0f, 0.0f},
  };

  std::uint16_t exact_half(float value) {
    // Test inputs are chosen exactly representable as normal halves (or zero).
    if (value == 0.0f) {
      return std::signbit(value) ? 0x8000 : 0;
    }
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000;
    const int exponent = int((bits >> 23) & 0xff) - 127 + 15;
    const std::uint32_t mantissa = bits & 0x7fffff;
    EXPECT_TRUE(exponent > 0 && exponent < 31 && (mantissa & 0x1fff) == 0) << value;
    return std::uint16_t(sign | (std::uint32_t(exponent) << 10) | (mantissa >> 13));
  }

  // The host's legacy scRGBTo2100PQ, in double precision, as the add-on encodes PQ exports.
  std::array<double, 3> host_pq(const std::array<float, 3> &rgb) {
    constexpr double matrix[3][3] {
      {0.627402, 0.329292, 0.043306},
      {0.069095, 0.919544, 0.011360},
      {0.016394, 0.088028, 0.895578},
    };
    constexpr double m1 = 2610.0 / 4096.0 / 4, m2 = 2523.0 / 4096.0 * 128;
    constexpr double c1 = 3424.0 / 4096.0, c2 = 2413.0 / 4096.0 * 32, c3 = 2392.0 / 4096.0 * 32;
    std::array<double, 3> code {};
    for (int row = 0; row < 3; ++row) {
      const double nits = 80.0 * (matrix[row][0] * rgb[0] + matrix[row][1] * rgb[1] + matrix[row][2] * rgb[2]);
      const double lp = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), m1);
      code[row] = std::pow((c1 + c2 * lp) / (1.0 + c3 * lp), m2);
    }
    return code;
  }

  float half_value(std::uint16_t raw) {
    const int exponent = (raw >> 10) & 31;
    const float magnitude = exponent ? std::ldexp(1.0f + (raw & 1023) / 1024.0f, exponent - 15) : std::ldexp(float(raw & 1023), -24);
    return (raw & 0x8000) ? -magnitude : magnitude;
  }

  std::uint32_t pack_r10g10b10a2(const std::array<double, 3> &code) {
    std::uint32_t packed = 3u << 30;
    for (int c = 0; c < 3; ++c) {
      packed |= std::uint32_t(std::lround(std::clamp(code[c], 0.0, 1.0) * 1023.0)) << (10 * c);
    }
    return packed;
  }

  class Game3DHostConversion: public testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL requested = D3D_FEATURE_LEVEL_11_0;
      ASSERT_EQ(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, &requested, 1, D3D11_SDK_VERSION, &device, nullptr, &context), S_OK);
      D3D11_SAMPLER_DESC sampler_desc {};
      sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
      ASSERT_EQ(device->CreateSamplerState(&sampler_desc, &sampler), S_OK);
      D3D11_RASTERIZER_DESC raster_desc {};
      raster_desc.FillMode = D3D11_FILL_SOLID;
      raster_desc.CullMode = D3D11_CULL_BACK;
      raster_desc.DepthClipEnable = TRUE;
      raster_desc.ScissorEnable = TRUE;
      ASSERT_EQ(device->CreateRasterizerState(&raster_desc, &scissor_raster), S_OK);
    }

    ComPtr<ID3DBlob> compile(const char *file, const char *entry = "main_ps", const char *profile = "ps_5_0") {
      auto &cached = blobs[std::string(file) + entry];
      if (cached) {
        return cached;
      }
      const auto path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / file;
      ComPtr<ID3DBlob> errors;
      const auto status = D3DCompileFromFile(path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, profile, D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &cached, &errors);
      EXPECT_EQ(status, S_OK) << file << ": " << (errors ? std::string(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize()) : "");
      return cached;
    }

    ComPtr<ID3D11PixelShader> pixel_shader(const char *file) {
      auto blob = compile(file);
      ComPtr<ID3D11PixelShader> shader;
      if (blob) {
        EXPECT_EQ(device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &shader), S_OK);
      }
      return shader;
    }

    ComPtr<ID3D11VertexShader> vertex_shader(const char *file) {
      auto blob = compile(file, "main_vs", "vs_5_0");
      ComPtr<ID3D11VertexShader> shader;
      if (blob) {
        EXPECT_EQ(device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &shader), S_OK);
      }
      return shader;
    }

    template<class T>
    ComPtr<ID3D11Buffer> constants(const T &value) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(T);
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA data {&value, 0, 0};
      ComPtr<ID3D11Buffer> buffer;
      EXPECT_EQ(device->CreateBuffer(&desc, &data, &buffer), S_OK);
      return buffer;
    }

    struct texture_t {
      ComPtr<ID3D11Texture2D> texture;
      ComPtr<ID3D11ShaderResourceView> view;
      ComPtr<ID3D11RenderTargetView> target;
    };

    texture_t make_texture(UINT width, UINT height, DXGI_FORMAT format, const void *pixels = nullptr, UINT pitch = 0, bool render_target = false) {
      texture_t result;
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
      desc.Format = format;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | (render_target ? D3D11_BIND_RENDER_TARGET : 0);
      D3D11_SUBRESOURCE_DATA data {pixels, pitch, 0};
      EXPECT_EQ(device->CreateTexture2D(&desc, pixels ? &data : nullptr, &result.texture), S_OK);
      if (result.texture) {
        EXPECT_EQ(device->CreateShaderResourceView(result.texture.Get(), nullptr, &result.view), S_OK);
        if (render_target) {
          EXPECT_EQ(device->CreateRenderTargetView(result.texture.Get(), nullptr, &result.target), S_OK);
        }
      }
      return result;
    }

    std::vector<std::uint8_t> read(ID3D11Texture2D *texture) {
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      UINT pixel_bytes = 4;
      switch (desc.Format) {
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
          pixel_bytes = 16;
          break;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R32G32_FLOAT:
          pixel_bytes = 8;
          break;
        case DXGI_FORMAT_R16_UNORM:
          pixel_bytes = 2;
          break;
        case DXGI_FORMAT_R8_UNORM:
          pixel_bytes = 1;
          break;
        default:
          break;
      }
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = 0;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      EXPECT_EQ(device->CreateTexture2D(&desc, nullptr, &staging), S_OK);
      std::vector<std::uint8_t> bytes(std::size_t(desc.Width) * desc.Height * pixel_bytes);
      if (!staging) {
        return bytes;
      }
      context->CopyResource(staging.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      EXPECT_EQ(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), S_OK);
      if (!mapped.pData) {
        return bytes;
      }
      for (UINT row = 0; row < desc.Height; ++row) {
        std::memcpy(bytes.data() + std::size_t(row) * desc.Width * pixel_bytes, static_cast<const std::uint8_t *>(mapped.pData) + std::size_t(row) * mapped.RowPitch, desc.Width * pixel_bytes);
      }
      context->Unmap(staging.Get(), 0);
      return bytes;
    }

    // One production pass: fullscreen triangle into `target` through `viewport`.
    struct pass_t {
      ComPtr<ID3D11VertexShader> vs;
      ComPtr<ID3D11PixelShader> ps;
      D3D11_VIEWPORT viewport {};
      std::vector<ComPtr<ID3D11Buffer>> vs_constants, ps_constants;  // indexed by slot
    };

    void draw(const pass_t &pass, ID3D11RenderTargetView *target, ID3D11ShaderResourceView *input, const D3D11_RECT *scissor = nullptr) {
      context->ClearState();
      context->OMSetRenderTargets(1, &target, nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      context->VSSetShader(pass.vs.Get(), nullptr, 0);
      context->PSSetShader(pass.ps.Get(), nullptr, 0);
      for (UINT slot = 0; slot < pass.vs_constants.size(); ++slot) {
        context->VSSetConstantBuffers(slot, 1, pass.vs_constants[slot].GetAddressOf());
      }
      for (UINT slot = 0; slot < pass.ps_constants.size(); ++slot) {
        context->PSSetConstantBuffers(slot, 1, pass.ps_constants[slot].GetAddressOf());
      }
      context->PSSetSamplers(0, 1, sampler.GetAddressOf());
      context->PSSetShaderResources(0, 1, &input);
      context->RSSetViewports(1, &pass.viewport);
      if (scissor) {
        context->RSSetState(scissor_raster.Get());
        context->RSSetScissorRects(1, scissor);
      }
      context->Draw(3, 0);
      context->ClearState();
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID3D11RasterizerState> scissor_raster;
    std::map<std::string, ComPtr<ID3DBlob>> blobs;
  };

  // Every value is exact in FP16. The sweep spans black, SDR, bright HDR, above 10000 nits
  // (125 = 10000 nits) and negative Rec.709 components inside and outside Rec.2020.
  const std::vector<std::array<float, 3>> pq_sweep {
    {0.0f, 0.0f, 0.0f},
    {0.0009765625f, 0.001953125f, 0.00390625f},
    {0.015625f, 0.03125f, 0.0625f},
    {0.25f, 0.5f, 0.125f},
    {1.0f, 1.0f, 1.0f},
    {2.5f, 2.5f, 2.5f},
    {2.5f, 0.0f, 0.0f},
    {0.0f, 2.5f, 0.0f},
    {0.0f, 0.0f, 2.5f},
    {12.5f, 1.25f, 0.25f},
    {50.0f, 25.0f, 6.25f},
    {125.0f, 125.0f, 125.0f},
    {200.0f, 150.0f, 100.0f},
    {1024.0f, 0.5f, 2.0f},
    {-0.25f, 1.0f, 0.5f},
    {50.0f, -4.0f, -0.5f},
    {-0.5f, 0.5f, 4.0f},
    {6.0f, -0.125f, 3.0f},
  };
}  // namespace

TEST_F(Game3DHostConversion, PqDecodeInvertsTheHostEncode) {
  // Compute PQInputToScRGB(scRGBTo2100PQ(x)) with both production includes.
  const std::string source =
    "#include \"include/common.hlsl\"\n"
    "#include \"include/pq_input.hlsl\"\n"
    "StructuredBuffer<float4> input_pixels : register(t0);\n"
    "RWStructuredBuffer<float4> output_pixels : register(u0);\n"
    "[numthreads(64, 1, 1)] void main_cs(uint3 thread : SV_DispatchThreadID) {\n"
    "  uint count, stride; input_pixels.GetDimensions(count, stride);\n"
    "  if (thread.x >= count) return;\n"
    "  output_pixels[thread.x] = float4(PQInputToScRGB(scRGBTo2100PQ(input_pixels[thread.x].rgb)), 1);\n"
    "}\n";
  const auto path = (std::filesystem::path(SUNSHINE_SHADERS_DIR) / "pq_input_test_inline.hlsl").string();
  ComPtr<ID3DBlob> code, errors;
  ASSERT_EQ(D3DCompile(source.data(), source.size(), path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main_cs", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors), S_OK)
    << (errors ? static_cast<const char *>(errors->GetBufferPointer()) : "");
  ComPtr<ID3D11ComputeShader> shader;
  ASSERT_EQ(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader), S_OK);
  // In Rec.709 gamut and below 10000 nits, PQ represents scRGB exactly up to float error.
  std::vector<std::array<float, 4>> pixels;
  for (const float a : {0.0f, 0.001f, 0.02f, 0.3f, 1.0f, 2.5f, 12.5f, 60.0f, 120.0f}) {
    pixels.push_back({a, a, a, 1});
    pixels.push_back({a, a * 0.5f, a * 0.125f, 1});
    pixels.push_back({a * 0.1f, a * 0.8f, a, 1});
  }
  D3D11_BUFFER_DESC desc {};
  desc.ByteWidth = UINT(pixels.size() * sizeof(pixels[0]));
  desc.Usage = D3D11_USAGE_IMMUTABLE;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
  desc.StructureByteStride = sizeof(pixels[0]);
  D3D11_SUBRESOURCE_DATA data {pixels.data(), 0, 0};
  ComPtr<ID3D11Buffer> input, output, staging;
  ASSERT_EQ(device->CreateBuffer(&desc, &data, &input), S_OK);
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
  ASSERT_EQ(device->CreateBuffer(&desc, nullptr, &output), S_OK);
  ComPtr<ID3D11ShaderResourceView> input_view;
  ComPtr<ID3D11UnorderedAccessView> output_view;
  ASSERT_EQ(device->CreateShaderResourceView(input.Get(), nullptr, &input_view), S_OK);
  ASSERT_EQ(device->CreateUnorderedAccessView(output.Get(), nullptr, &output_view), S_OK);
  context->CSSetShader(shader.Get(), nullptr, 0);
  context->CSSetShaderResources(0, 1, input_view.GetAddressOf());
  context->CSSetUnorderedAccessViews(0, 1, output_view.GetAddressOf(), nullptr);
  context->Dispatch(1, 1, 1);
  context->ClearState();
  desc.Usage = D3D11_USAGE_STAGING;
  desc.BindFlags = desc.MiscFlags = desc.StructureByteStride = 0;
  desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  ASSERT_EQ(device->CreateBuffer(&desc, nullptr, &staging), S_OK);
  context->CopyResource(staging.Get(), output.Get());
  D3D11_MAPPED_SUBRESOURCE mapped {};
  ASSERT_EQ(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), S_OK);
  std::vector<std::array<float, 4>> result(pixels.size());
  std::memcpy(result.data(), mapped.pData, result.size() * sizeof(result[0]));
  context->Unmap(staging.Get(), 0);
  for (std::size_t i = 0; i < pixels.size(); ++i) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(result[i][c], pixels[i][c], 2e-3f * pixels[i][c] + 2e-5f) << i << ' ' << c;
    }
  }
}

TEST_F(Game3DHostConversion, PqExportMatchesTheLegacyFp16PathWithinOneCode) {
  // Each sample covers four columns of two rows. Luma reads its first column; the odd chroma
  // pixel's [1,2,1]x[1,1] taps stay inside the sample, so both paths average one color.
  const auto count = UINT(pq_sweep.size());
  const UINT width = count * 4, height = 2;
  std::vector<std::array<std::uint16_t, 4>> half_pixels(std::size_t(width) * height);
  std::vector<std::uint32_t> pq_pixels(half_pixels.size());
  for (UINT y = 0; y < height; ++y) {
    for (UINT x = 0; x < width; ++x) {
      const auto &rgb = pq_sweep[x / 4];
      half_pixels[y * width + x] = {exact_half(rgb[0]), exact_half(rgb[1]), exact_half(rgb[2]), 0x3c00};
      pq_pixels[y * width + x] = pack_r10g10b10a2(host_pq(rgb));
    }
  }
  const auto legacy = make_texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, half_pixels.data(), width * 8);
  const auto pq = make_texture(width, height, DXGI_FORMAT_R10G10B10A2_UNORM, pq_pixels.data(), width * 4);
  auto y_target = make_texture(width, height, DXGI_FORMAT_R32_FLOAT, nullptr, 0, true);
  auto uv_target = make_texture(width / 2, height / 2, DXGI_FORMAT_R32G32_FLOAT, nullptr, 0, true);

  struct variant_t {
    const char *legacy_y, *pq_y, *legacy_uv, *pq_uv;
    bool hdr;
  };

  for (const auto &variant : {
         variant_t {"convert_yuv420_planar_y_ps_perceptual_quantizer.hlsl", "convert_yuv420_planar_y_ps_pq_input_perceptual_quantizer.hlsl", "convert_yuv420_packed_uv_type0_ps_perceptual_quantizer.hlsl", "convert_yuv420_packed_uv_type0_ps_pq_input_perceptual_quantizer.hlsl", true},
         variant_t {"convert_yuv420_planar_y_ps_linear.hlsl", "convert_yuv420_planar_y_ps_pq_input_linear.hlsl", "convert_yuv420_packed_uv_type0_ps_linear.hlsl", "convert_yuv420_packed_uv_type0_ps_pq_input_linear.hlsl", false},
       }) {
    for (const std::uint32_t mode : variant.hdr ? std::vector<std::uint32_t> {0, 1, 2} : std::vector<std::uint32_t> {0}) {
      SCOPED_TRACE(std::string(variant.pq_y) + " gamma " + std::to_string(mode));
      sdr_transform_t transform;
      transform.source_is_hdr = !variant.hdr;  // external_color_transform: HDR export into SDR.
      transform.target_is_hdr = variant.hdr;
      const auto matrix = constants(bt2020_full);
      const auto sdr = constants(transform);
      const auto chroma = constants(chroma_t {{1.0f / width, 1.0f / height}, width, height, 0});
      const auto gamma = constants(gamma_t {mode, 2.5f});
      const auto rotation = constants(std::array<std::int32_t, 4> {});
      pass_t y_pass {vertex_shader("convert_yuv420_planar_y_vs.hlsl"), nullptr, {0, 0, float(width), float(height), 0, 1}, {nullptr, rotation}, {matrix, sdr, chroma, nullptr, gamma}};
      pass_t uv_pass {vertex_shader("convert_yuv420_packed_uv_type0_vs.hlsl"), nullptr, {0, 0, float(width) / 2, float(height) / 2, 0, 1}, {chroma, rotation}, {matrix, sdr, chroma, nullptr, gamma}};
      const auto run = [&](const char *y_shader, const char *uv_shader, ID3D11ShaderResourceView *input) {
        y_pass.ps = pixel_shader(y_shader);
        uv_pass.ps = pixel_shader(uv_shader);
        draw(y_pass, y_target.target.Get(), input);
        draw(uv_pass, uv_target.target.Get(), input);
        std::vector<float> values(count * 3);
        const auto y_bytes = read(y_target.texture.Get());
        const auto uv_bytes = read(uv_target.texture.Get());
        for (UINT i = 0; i < count; ++i) {
          std::memcpy(&values[i * 3], y_bytes.data() + std::size_t(i) * 4 * 4, 4);
          std::memcpy(&values[i * 3 + 1], uv_bytes.data() + (std::size_t(i) * 2 + 1) * 8, 8);
        }
        return values;
      };
      const auto expected = run(variant.legacy_y, variant.legacy_uv, legacy.view.Get());
      const auto actual = run(variant.pq_y, variant.pq_uv, pq.view.Get());
      // HDR: one 10-bit code. SDR: one 8-bit code, for colors PQ represents (inside Rec.2020
      // and at most 10000 nits); PQ clips the others before the SDR tone map does.
      const double scale = variant.hdr ? 1023.0 : 255.0;
      for (UINT i = 0; i < count; ++i) {
        const auto code = host_pq(pq_sweep[i]);
        const bool representable = std::all_of(code.begin(), code.end(), [](double c) {
                                     return c > 0.0 && c < 1.0;
                                   }) ||
                                   pq_sweep[i] == std::array<float, 3> {0, 0, 0};
        if (!variant.hdr && !representable) {
          continue;
        }
        for (int c = 0; c < 3; ++c) {
          EXPECT_TRUE(std::isfinite(actual[i * 3 + c]));
          EXPECT_LE(std::abs(double(actual[i * 3 + c]) - expected[i * 3 + c]) * scale, 1.0 + 1e-3) << "sample " << i << " plane value " << c << ": " << actual[i * 3 + c] << " vs " << expected[i * 3 + c];
        }
      }
    }
  }
}

TEST_F(Game3DHostConversion, AllPqInputEntryPointsCompile) {
  for (const auto *file : {
         "convert_yuv420_planar_y_ps_pq_input_perceptual_quantizer.hlsl",
         "convert_yuv420_planar_y_ps_pq_input_linear.hlsl",
         "convert_yuv420_packed_uv_type0_ps_pq_input_perceptual_quantizer.hlsl",
         "convert_yuv420_packed_uv_type0_ps_pq_input_linear.hlsl",
         "convert_yuv420_packed_uv_type0s_ps_pq_input_perceptual_quantizer.hlsl",
         "convert_yuv420_packed_uv_type0s_ps_pq_input_linear.hlsl",
         "rgb_present_pq_input_to_linear_ps.hlsl",
         "rgb_present_pq_input_to_srgb_ps.hlsl",
         "sbs_packed_resample_pq_input_ps.hlsl",
         "pq_input_to_scrgb_ps.hlsl",
       }) {
    ComPtr<ID3DBlob> errors, code;
    const auto path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / file;
    const auto status = D3DCompileFromFile(path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main_ps", "ps_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    EXPECT_EQ(status, S_OK) << file << ": " << (errors ? static_cast<const char *>(errors->GetBufferPointer()) : "");
    // Production logs every compiler message at startup; these entry points must be silent.
    EXPECT_FALSE(errors) << file << ": " << (errors ? static_cast<const char *>(errors->GetBufferPointer()) : "");
  }
}

TEST_F(Game3DHostConversion, PqResampleAndLocalPresentationFilterInLinearLight) {
  // Two texels, 0 and 10000 nits gray, per eye. Halfway between them the FP16 path averages
  // linear light (62.5 scRGB); filtering the codes would give about 92 nits instead.
  const std::array<std::uint32_t, 4> codes {
    pack_r10g10b10a2(host_pq({0, 0, 0})),
    pack_r10g10b10a2(host_pq({125, 125, 125})),
    pack_r10g10b10a2(host_pq({0, 0, 0})),
    pack_r10g10b10a2(host_pq({125, 125, 125})),
  };
  const auto source = make_texture(4, 1, DXGI_FORMAT_R10G10B10A2_UNORM, codes.data(), 16);
  auto target = make_texture(1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, nullptr, 0, true);
  // A 1x1 viewport over texture coordinate 0.25 samples halfway between the left eye's texels.
  pass_t pass {vertex_shader("sbs_reprojection_vs.hlsl"), pixel_shader("rgb_present_pq_input_to_linear_ps.hlsl"), {0, 0, 2, 1, 0, 1}, {}, {}};
  draw(pass, target.target.Get(), source.view.Get());
  std::array<float, 4> value {};
  std::memcpy(value.data(), read(target.texture.Get()).data(), sizeof(value));
  EXPECT_NEAR(value[0], 62.5f, 0.5f);
  EXPECT_NEAR(value[1], 62.5f, 0.5f);
  EXPECT_NEAR(value[3], 1.0f, 1e-6f);
  pass.ps = pixel_shader("sbs_packed_resample_pq_input_ps.hlsl");
  draw(pass, target.target.Get(), source.view.Get());
  std::memcpy(value.data(), read(target.texture.Get()).data(), sizeof(value));
  EXPECT_NEAR(value[0], 62.5f, 0.5f);
}

TEST_F(Game3DHostConversion, FittedEyesResampleIntoBlackBarsWithoutSeamBleed) {
  struct fit_case_t {
    UINT source_width, source_height, eye_width, eye_height;
    ::reshade_bridge::eye_rect_t expected;
  };

  struct transfer_case_t {
    DXGI_FORMAT format;
    const char *resample, *luma;
    bool pq;
  };

  // Per-eye texel colors in scRGB (multiples of 1/64, exact in FP16). The eyes differ strongly,
  // so any texel of one eye in the other's rectangle exceeds the comparison tolerance.
  const auto eye_color = [](UINT eye, UINT x, UINT y, UINT width, UINT height) {
    const auto q = [](float v) {
      return std::round(v * 64.0f) / 64.0f;
    };
    return eye == 0 ? std::array<float, 3> {q(0.25f + 0.5f * x / width), q(0.75f - 0.5f * y / height), 0.5f} :
                      std::array<float, 3> {0.015625f, q(0.05f * (x % 3)), q(0.875f - 0.25f * y / height)};
  };
  const auto make_source = [&](const transfer_case_t &transfer, UINT width, UINT height, const std::array<int, 2> &eyes) {
    // eyes[side]: which eye's colors fill that half (a duplicate of one eye is the reference).
    if (transfer.format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
      std::vector<std::array<std::uint16_t, 4>> texels(std::size_t(width) * 2 * height);
      for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < 2 * width; ++x) {
          const auto rgb = eye_color(eyes[x / width], x % width, y, width, height);
          texels[std::size_t(y) * 2 * width + x] = {exact_half(rgb[0]), exact_half(rgb[1]), exact_half(rgb[2]), 0x3c00};
        }
      }
      return make_texture(2 * width, height, transfer.format, texels.data(), 2 * width * 8);
    }
    std::vector<std::uint32_t> texels(std::size_t(width) * 2 * height);
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < 2 * width; ++x) {
        const auto rgb = eye_color(eyes[x / width], x % width, y, width, height);
        auto &texel = texels[std::size_t(y) * 2 * width + x];
        if (transfer.pq) {
          texel = pack_r10g10b10a2(host_pq(rgb));
        } else {
          texel = 0xff000000u;
          for (int c = 0; c < 3; ++c) {
            texel |= std::uint32_t(std::lround(std::clamp(rgb[c], 0.0f, 1.0f) * 255.0f)) << (8 * c);
          }
        }
      }
    }
    return make_texture(2 * width, height, transfer.format, texels.data(), 2 * width * 4);
  };
  const auto read_half = [&](ID3D11Texture2D *texture) {
    const auto bytes = read(texture);
    std::vector<float> values(bytes.size() / 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::uint16_t raw;
      std::memcpy(&raw, bytes.data() + i * 2, 2);
      values[i] = half_value(raw);
    }
    return values;
  };

  const auto color_matrix = constants(bt2020_full);
  const auto gamma = constants(gamma_t {});
  const auto rotation = constants(std::array<std::int32_t, 4> {});
  for (const auto &fit : {
         fit_case_t {24, 16, 28, 12, {4, 0, 18, 12}},  // Pillarboxed at 0.75.
         fit_case_t {32, 16, 24, 16, {0, 2, 24, 12}},  // Letterboxed at 0.75.
       }) {
    const UINT width = 2 * fit.eye_width, height = fit.eye_height;
    const auto placement = ::reshade_bridge::place_output(2 * fit.source_width, fit.source_height, width, height);
    ASSERT_EQ(placement.fit, ::reshade_bridge::output_fit::fitted);
    ASSERT_EQ(std::tie(placement.eye.x, placement.eye.y, placement.eye.width, placement.eye.height), std::tie(fit.expected.x, fit.expected.y, fit.expected.width, fit.expected.height));
    const auto passes = platf::packed_output::resample_passes(placement, width, height);
    ASSERT_EQ(passes.count, 2u);
    ASSERT_TRUE(passes.clear);
    const auto &eye = placement.eye;
    for (const auto &transfer : {
           transfer_case_t {DXGI_FORMAT_R8G8B8A8_UNORM, "sbs_packed_resample_ps.hlsl", "convert_yuv420_planar_y_ps.hlsl", false},
           transfer_case_t {DXGI_FORMAT_R16G16B16A16_FLOAT, "sbs_packed_resample_ps.hlsl", "convert_yuv420_planar_y_ps_perceptual_quantizer.hlsl", false},
           transfer_case_t {DXGI_FORMAT_R10G10B10A2_UNORM, "sbs_packed_resample_pq_input_ps.hlsl", "convert_yuv420_planar_y_ps_perceptual_quantizer.hlsl", true},
         }) {
      SCOPED_TRACE(std::to_string(fit.source_width) + "x" + std::to_string(fit.source_height) + " format " + std::to_string(int(transfer.format)));
      const auto source = make_source(transfer, fit.source_width, fit.source_height, {0, 1});
      // The production draws: black, then each eye's pass inside its scissor.
      auto output = make_texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, nullptr, 0, true);
      constexpr float black[4] {0.0f, 0.0f, 0.0f, 1.0f};
      context->ClearRenderTargetView(output.target.Get(), black);
      pass_t pass {vertex_shader("sbs_reprojection_vs.hlsl"), pixel_shader(transfer.resample), {}, {}, {}};
      for (std::uint32_t i = 0; i < passes.count; ++i) {
        pass.viewport = passes.passes[i].viewport;
        draw(pass, output.target.Get(), source.view.Get(), &*passes.passes[i].scissor);
      }
      const auto actual = read_half(output.texture.Get());
      // Each eye alone, scaled to the fitted size from a source holding only that eye.
      std::array<std::vector<float>, 2> expected;
      for (int side = 0; side < 2; ++side) {
        const auto alone = make_source(transfer, fit.source_width, fit.source_height, {side, side});
        auto reference = make_texture(2 * eye.width, eye.height, DXGI_FORMAT_R16G16B16A16_FLOAT, nullptr, 0, true);
        pass.viewport = {0, 0, float(2 * eye.width), float(eye.height), 0, 1};
        draw(pass, reference.target.Get(), alone.view.Get());
        expected[side] = read_half(reference.texture.Get());
      }
      std::size_t inside = 0;
      for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
          const UINT side = x / fit.eye_width, local = x % fit.eye_width;
          const float *value = &actual[(std::size_t(y) * width + x) * 4];
          if (local >= eye.x && local < eye.x + eye.width && y >= eye.y && y < eye.y + eye.height) {
            ++inside;
            const float *want = &expected[side][(std::size_t(y - eye.y) * 2 * eye.width + (local - eye.x)) * 4];
            // Two steps of 8-bit subtexel filtering precision; a texel of the other eye at a
            // weight above 1/32 differs by more.
            for (int c = 0; c < 3; ++c) {
              ASSERT_NEAR(value[c], want[c], 8e-3f * std::max(1.0f, std::abs(want[c]))) << "pixel " << x << ',' << y << " channel " << c;
            }
          } else {
            // Black in every transfer, opaque.
            ASSERT_EQ(value[0], 0.0f) << "pixel " << x << ',' << y;
            ASSERT_EQ(value[1], 0.0f) << "pixel " << x << ',' << y;
            ASSERT_EQ(value[2], 0.0f) << "pixel " << x << ',' << y;
            ASSERT_EQ(value[3], 1.0f) << "pixel " << x << ',' << y;
          }
        }
      }
      EXPECT_EQ(inside, std::size_t(2) * eye.width * eye.height);

      // The bars encode as black: luma code 0 in full range, for sRGB input and for linear input
      // encoded as PQ.
      sdr_transform_t transform;
      transform.target_is_hdr = transfer.format != DXGI_FORMAT_R8G8B8A8_UNORM;
      const auto sdr = constants(transform);
      const auto chroma = constants(chroma_t {{1.0f / width, 1.0f / height}, width, height, fit.eye_width});
      auto luma = make_texture(width, height, DXGI_FORMAT_R16_UNORM, nullptr, 0, true);
      const pass_t y_pass {vertex_shader("convert_yuv420_planar_y_vs.hlsl"), pixel_shader(transfer.luma), {0, 0, float(width), float(height), 0, 1}, {nullptr, rotation}, {color_matrix, sdr, chroma, nullptr, gamma}};
      draw(y_pass, luma.target.Get(), output.view.Get());
      const auto codes = read(luma.texture.Get());
      for (UINT y = 0; y < height; ++y) {
        for (const UINT x : {0u, eye.x + eye.width, fit.eye_width + eye.x - 1u, width - 1u}) {
          const UINT local = x % fit.eye_width;
          if (local >= eye.x && local < eye.x + eye.width && y >= eye.y && y < eye.y + eye.height) {
            continue;
          }
          std::uint16_t code;
          std::memcpy(&code, codes.data() + (std::size_t(y) * width + x) * 2, 2);
          EXPECT_EQ(code, 0u) << "luma at " << x << ',' << y;
        }
      }

      // Local AR presents the fitted raster at its own size: the bars stay black and the eyes
      // keep their rectangles.
      auto presented = make_texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, nullptr, 0, true);
      const pass_t present {vertex_shader("sbs_reprojection_vs.hlsl"), pixel_shader("cursor_ps.hlsl"), {0, 0, float(width), float(height), 0, 1}, {}, {}};
      draw(present, presented.target.Get(), output.view.Get());
      EXPECT_EQ(read(presented.texture.Get()), read(output.texture.Get()));
    }
  }
}

namespace {
  // The full production conversion shapes that read an export: Y and 4:2:0 UV planes at the
  // packed size, the same-aspect resample, and a scaled Local AR presentation.
  struct cursor_case_t {
    DXGI_FORMAT format;
    bool linear;
  };
}  // namespace

TEST_F(Game3DHostConversion, CursorPatchConversionIsBitIdenticalToFullFrameComposition) {
  constexpr UINT width = 96, height = 24, eye = width / 2;
  ComPtr<ID3DBlob> cursor_vs = compile("cursor_vs.hlsl", "main_vs", "vs_5_0");
  ComPtr<ID3DBlob> cursor_ps = compile("cursor_ps.hlsl");
  ComPtr<ID3DBlob> cursor_hdr = compile("cursor_ps_normalize_white.hlsl");
  ComPtr<ID3DBlob> pq_decode = compile("pq_input_to_scrgb_ps.hlsl");
  ASSERT_TRUE(cursor_vs && cursor_ps && cursor_hdr && pq_decode);
  cursor::compositor_t patch_compositor(device.Get(), context.Get(), cursor_vs.Get(), cursor_ps.Get(), cursor_hdr.Get(), pq_decode.Get());
  cursor::compositor_t frame_compositor(device.Get(), context.Get(), cursor_vs.Get(), cursor_ps.Get(), cursor_hdr.Get(), pq_decode.Get());

  // A translucent and an inverting cursor, placed at fractional capture coordinates.
  auto shape = std::make_shared<cursor::shape_t>();
  shape->width = shape->height = 4;
  shape->alpha_bgra.assign(64, 0);
  shape->xor_bgra.assign(64, 0);
  for (UINT i = 0; i < 16; ++i) {
    auto *alpha = &shape->alpha_bgra[i * 4];
    alpha[0] = std::uint8_t(30 + i * 13);
    alpha[1] = std::uint8_t(200 - i * 7);
    alpha[2] = std::uint8_t(90 + i * 5);
    alpha[3] = std::uint8_t(i % 3 == 0 ? 0 : 64 + i * 12);
    if (i % 5 == 0) {
      std::memset(&shape->xor_bgra[i * 4], 255, 4);
    }
  }

  const auto color_matrix = constants(bt2020_full);
  sdr_transform_t transform;
  transform.source_is_hdr = 1;
  const auto sdr = constants(transform);
  const auto chroma = constants(chroma_t {{1.0f / width, 1.0f / height}, width, height, eye});
  const auto gamma = constants(gamma_t {});
  const auto rotation = constants(std::array<std::int32_t, 4> {});

  for (const auto &test : {cursor_case_t {DXGI_FORMAT_R8G8B8A8_UNORM, false}, cursor_case_t {DXGI_FORMAT_R10G10B10A2_UNORM, false}, cursor_case_t {DXGI_FORMAT_R16G16B16A16_FLOAT, true}}) {
    SCOPED_TRACE(int(test.format));
    // Deterministic, spatially varying content so any footprint mistake changes the output.
    std::vector<std::uint8_t> bytes;
    UINT pitch = 0;
    if (test.linear) {
      std::vector<std::array<std::uint16_t, 4>> texels(width * height);
      for (UINT i = 0; i < texels.size(); ++i) {
        const float values[] {0.0f, 0.125f, 0.5f, 1.0f, 2.5f, 6.0f, 12.5f, -0.25f};
        texels[i] = {exact_half(values[i % 8]), exact_half(values[(i / 3) % 8]), exact_half(values[(i * 7 / 5) % 8]), 0x3c00};
      }
      bytes.resize(texels.size() * 8);
      std::memcpy(bytes.data(), texels.data(), bytes.size());
      pitch = width * 8;
    } else {
      bytes.resize(std::size_t(width) * height * 4);
      for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = std::uint8_t((i * 37 + (i / 384) * 11) & 0xff);
      }
      pitch = width * 4;
    }
    const auto source = make_texture(width, height, test.format, bytes.data(), pitch);

    struct output_pass_t {
      pass_t pass;
      UINT target_width, target_height;
      DXGI_FORMAT target_format;
      // A fitted resample: these viewports and scissors replace pass.viewport, over black.
      platf::packed_output::passes_t fit {};
    };

    const auto fitted = [&](UINT eye_width, UINT eye_height) {
      const auto placement = ::reshade_bridge::place_output(width, height, 2 * eye_width, eye_height);
      EXPECT_EQ(placement.fit, ::reshade_bridge::output_fit::fitted);
      return output_pass_t {{vertex_shader("sbs_reprojection_vs.hlsl"), pixel_shader("sbs_packed_resample_ps.hlsl"), {}, {}, {}}, 2 * eye_width, eye_height, DXGI_FORMAT_R16G16B16A16_FLOAT, platf::packed_output::resample_passes(placement, 2 * eye_width, eye_height)};
    };
    std::vector<output_pass_t> passes {
      {{vertex_shader("convert_yuv420_planar_y_vs.hlsl"), pixel_shader(test.linear ? "convert_yuv420_planar_y_ps_perceptual_quantizer.hlsl" : "convert_yuv420_planar_y_ps.hlsl"), {0, 0, float(width), float(height), 0, 1}, {nullptr, rotation}, {color_matrix, sdr, chroma, nullptr, gamma}}, width, height, DXGI_FORMAT_R16_UNORM},
      {{vertex_shader("convert_yuv420_packed_uv_type0_vs.hlsl"), pixel_shader(test.linear ? "convert_yuv420_packed_uv_type0_ps_perceptual_quantizer.hlsl" : "convert_yuv420_packed_uv_type0_ps.hlsl"), {0, 0, float(width) / 2, float(height) / 2, 0, 1}, {chroma, rotation}, {color_matrix, sdr, chroma, nullptr, gamma}}, width / 2, height / 2, DXGI_FORMAT_R16G16_UNORM},
      // Same-aspect resample to a smaller packed stream (each eye 36x18).
      {{vertex_shader("sbs_reprojection_vs.hlsl"), pixel_shader("sbs_packed_resample_ps.hlsl"), {0, 0, 72, 18, 0, 1}, {}, {}}, 72, 18, DXGI_FORMAT_R16G16B16A16_FLOAT},
      // Local AR presentation into a smaller swapchain (cursor_ps is rgb_present_ps).
      {{vertex_shader("sbs_reprojection_vs.hlsl"), pixel_shader("cursor_ps.hlsl"), {0, 0, 60, 15, 0, 1}, {}, {}}, 60, 15, test.linear ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM},
      // Other aspects fitted into each eye through offset viewports: pillarboxed (each 2:1 eye
      // 32x16 at x = 4 of 40x16) and letterboxed (36x18 at y = 4 of 36x26).
      fitted(40, 16),
      fitted(36, 26),
    };
    ASSERT_EQ(passes[4].fit.count, 2u);
    EXPECT_EQ(passes[4].fit.passes[1].scissor->left, 44);
    EXPECT_EQ(passes[5].fit.passes[0].scissor->top, 4);
    float minimum_scale = 1.0f;
    for (const auto &output : passes) {
      minimum_scale = std::min(minimum_scale, output.fit.count ? platf::packed_output::minimum_scale(output.fit, width, height) : std::min(output.pass.viewport.Width / width, output.pass.viewport.Height / height));
    }
    const auto margin = cursor::patch_margin(minimum_scale);
    ASSERT_LT(margin, 64u);

    for (const auto &[x, y, parallax] : std::array<std::tuple<float, float, float>, 5> {{
           {10.25f, 4.5f, 0.0f},
           {3.75f, 7.25f, 0.02f},
           {45.5f, 0.0f, -0.03f},  // Clipped by the left eye's seam and the right eye's edge.
           {-1.5f, 21.0f, 0.01f},  // Clipped by the frame edges.
           {22.0f, 10.0f, 0.04f},
         }}) {
      SCOPED_TRACE(x);
      cursor::snapshot_t snapshot {shape, {x, y, 4.0f, 4.0f, 0, 1}, eye, height, DXGI_MODE_ROTATION_IDENTITY, true};
      const auto transfer = test.linear ? cursor::source_transfer_e::scrgb : cursor::source_transfer_e::srgb;
      const auto full = frame_compositor.compose(source.texture.Get(), source.view.Get(), snapshot, transfer, 2.5f, parallax, cursor::whole_frame_margin);
      const auto patch = patch_compositor.compose(source.texture.Get(), source.view.Get(), snapshot, transfer, 2.5f, parallax, margin);
      ASSERT_TRUE(full && patch);
      ASSERT_GT(patch->regions, 0u);
      EXPECT_EQ(patch->regions, full->regions);
      for (std::uint32_t i = 0; i < patch->regions; ++i) {
        const auto &valid = patch->valid[i];
        EXPECT_LT((valid.right - valid.left) * (valid.bottom - valid.top), LONG(width * height));
      }
      for (const auto &output : passes) {
        auto reference = make_texture(output.target_width, output.target_height, output.target_format, nullptr, 0, true);
        auto actual = make_texture(output.target_width, output.target_height, output.target_format, nullptr, 0, true);
        if (!output.fit.count) {
          draw(output.pass, reference.target.Get(), full->view);
          draw(output.pass, actual.target.Get(), source.view.Get());
          for (std::uint32_t i = 0; i < patch->regions; ++i) {
            if (const auto region = cursor::output_region(patch->changed[i], width, height, output.pass.viewport)) {
              draw(output.pass, actual.target.Get(), patch->view, &*region);
            }
          }
        } else {
          // The converter's fitted resample: black, then per eye the export inside its scissor
          // and the patch over that eye's regions (pass_region keeps them inside the scissor).
          constexpr float black[4] {0.0f, 0.0f, 0.0f, 1.0f};
          context->ClearRenderTargetView(reference.target.Get(), black);
          context->ClearRenderTargetView(actual.target.Get(), black);
          std::uint32_t patched = 0;
          for (std::uint32_t p = 0; p < output.fit.count; ++p) {
            auto pass = output.pass;
            pass.viewport = output.fit.passes[p].viewport;
            const auto &scissor = *output.fit.passes[p].scissor;
            draw(pass, reference.target.Get(), full->view, &scissor);
            draw(pass, actual.target.Get(), source.view.Get(), &scissor);
            for (std::uint32_t i = 0; i < patch->regions; ++i) {
              if (const auto region = platf::packed_output::pass_region(output.fit.passes[p], patch->changed[i], width, height)) {
                EXPECT_GE(region->left, scissor.left);
                EXPECT_LE(region->right, scissor.right);
                EXPECT_GE(region->top, scissor.top);
                EXPECT_LE(region->bottom, scissor.bottom);
                draw(pass, actual.target.Get(), patch->view, &*region);
                ++patched;
              }
            }
          }
          EXPECT_GT(patched, 0u);
        }
        EXPECT_EQ(read(actual.texture.Get()), read(reference.texture.Get())) << "target " << output.target_width << 'x' << output.target_height;
      }
    }
  }
}

TEST_F(Game3DHostConversion, PqCursorPatchBlendsInLinearLight) {
  constexpr UINT width = 32, height = 8;
  ComPtr<ID3DBlob> cursor_vs = compile("cursor_vs.hlsl", "main_vs", "vs_5_0");
  ComPtr<ID3DBlob> cursor_ps = compile("cursor_ps.hlsl");
  ComPtr<ID3DBlob> cursor_hdr = compile("cursor_ps_normalize_white.hlsl");
  ComPtr<ID3DBlob> pq_decode = compile("pq_input_to_scrgb_ps.hlsl");
  cursor::compositor_t compositor(device.Get(), context.Get(), cursor_vs.Get(), cursor_ps.Get(), cursor_hdr.Get(), pq_decode.Get());
  cursor::compositor_t without_decoder(device.Get(), context.Get(), cursor_vs.Get(), cursor_ps.Get(), cursor_hdr.Get());
  std::vector<std::uint32_t> codes(width * height, pack_r10g10b10a2(host_pq({2.5f, 1.25f, 0.5f})));
  const auto source = make_texture(width, height, DXGI_FORMAT_R10G10B10A2_UNORM, codes.data(), width * 4);
  auto shape = std::make_shared<cursor::shape_t>();
  shape->width = shape->height = 2;
  shape->alpha_bgra.assign(16, 255);
  for (std::size_t i = 3; i < 16; i += 4) {
    shape->alpha_bgra[i] = 128;
  }
  cursor::snapshot_t snapshot {shape, {4, 2, 2, 2, 0, 1}, width / 2, height, DXGI_MODE_ROTATION_IDENTITY, true};
  EXPECT_FALSE(without_decoder.compose(source.texture.Get(), source.view.Get(), snapshot, cursor::source_transfer_e::pq, 2.5f, 0.0f, cursor::patch_margin(0.5f)));
  const auto result = compositor.compose(source.texture.Get(), source.view.Get(), snapshot, cursor::source_transfer_e::pq, 2.5f, 0.0f, cursor::patch_margin(0.5f));
  ASSERT_TRUE(result);
  ASSERT_EQ(result->regions, 2u);
  EXPECT_TRUE(result->linear);
  const auto patch = read(result->texture);
  const auto texel = [&](UINT x, UINT y) {
    std::array<std::uint16_t, 4> raw {};
    std::memcpy(raw.data(), patch.data() + (std::size_t(y) * width + x) * 8, 8);
    std::array<float, 4> value {};
    for (int c = 0; c < 4; ++c) {
      const int exponent = (raw[c] >> 10) & 31;
      const float magnitude = exponent ? std::ldexp(1.0f + (raw[c] & 1023) / 1024.0f, exponent - 15) : std::ldexp(float(raw[c] & 1023), -24);
      value[c] = (raw[c] & 0x8000) ? -magnitude : magnitude;
    }
    return value;
  };
  // Outside the cursor the patch holds the decoded export; under it, white (2.5 scRGB) is
  // blended with alpha 128/255 in linear light.
  const float alpha = 128.0f / 255.0f;
  const std::array<float, 3> background {2.5f, 1.25f, 0.5f};
  for (const UINT offset : {0u, width / 2}) {
    const auto plain = texel(offset + 2, 2);
    const auto blended = texel(offset + 4, 2);
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(plain[c], background[c], 0.02f * background[c]);
      EXPECT_NEAR(blended[c], 2.5f * alpha + plain[c] * (1.0f - alpha), 0.01f);
    }
  }
}
#endif
