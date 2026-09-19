/**
 * @file tests/unit/test_host_sbs_chroma.cpp
 * @brief WARP checks of packed 4:2:0 conversion using the production shaders and sampler.
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
  #include <src/video_colorspace.h>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  using uv_pixel_t = std::array<std::uint16_t, 2>;
  using rgba_pixel_t = std::array<std::uint16_t, 4>;

  using chroma_constants_t = video::chroma_sampling_t;

  class HostSbsChromaGpuTest : public testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
      D3D_FEATURE_LEVEL actual {};
      ASSERT_EQ(D3D11CreateDevice(
                  nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, &level, 1,
                  D3D11_SDK_VERSION, &device, &actual, &context), S_OK);
      ASSERT_EQ(actual, level);
      D3D11_SAMPLER_DESC sampler_desc {};
      sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW =
        D3D11_TEXTURE_ADDRESS_CLAMP;
      sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
      ASSERT_EQ(device->CreateSamplerState(&sampler_desc, &sampler), S_OK);
      context->PSSetSamplers(0, 1, sampler.GetAddressOf());
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      std::array<std::uint32_t, 4> zero {};
      auto rotation = constants(zero);
      context->VSSetConstantBuffers(1, 1, rotation.GetAddressOf());
      auto sdr = constants(std::array<float, 4> {0, 0, 1, 0});
      context->PSSetConstantBuffers(1, 1, sdr.GetAddressOf());
    }

    template<class T>
    ComPtr<ID3D11Buffer> constants(const T &value) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(T);
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA initial {&value, 0, 0};
      ComPtr<ID3D11Buffer> buffer;
      EXPECT_EQ(device->CreateBuffer(&desc, &initial, &buffer), S_OK);
      return buffer;
    }

    ComPtr<ID3DBlob> compile(const std::string &file, const char *entry, const char *target) {
      const auto path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / file;
      ComPtr<ID3DBlob> result, errors;
      const auto status = D3DCompileFromFile(
        path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, target,
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &result, &errors);
      EXPECT_EQ(status, S_OK) << file << ": " << (errors ?
        std::string(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize()) : "");
      return result;
    }

    ComPtr<ID3D11PixelShader> inline_pixel_shader(const std::string &source) {
      // Give the standard include handler the production shader directory. All color math and
      // the reference chroma body below are compiled from production includes, not a CPU replica.
      const auto path = (std::filesystem::path(SUNSHINE_SHADERS_DIR) /
                         "host_sbs_chroma_test_inline.hlsl").string();
      ComPtr<ID3DBlob> code, errors;
      const auto status = D3DCompile(
        source.data(), source.size(), path.c_str(), nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE, "main_ps", "ps_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
      EXPECT_EQ(status, S_OK) << (errors ?
        std::string(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize()) : "");
      ComPtr<ID3D11PixelShader> result;
      if (code) {
        EXPECT_EQ(device->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &result), S_OK);
      }
      return result;
    }

    ComPtr<ID3D11Texture2D> texture(
      UINT width, UINT height, DXGI_FORMAT format,
      const void *pixels = nullptr, UINT row_bytes = 0
    ) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
      desc.Format = format;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
      D3D11_SUBRESOURCE_DATA initial {pixels, row_bytes, 0};
      ComPtr<ID3D11Texture2D> result;
      EXPECT_EQ(device->CreateTexture2D(&desc, pixels ? &initial : nullptr, &result), S_OK);
      return result;
    }

    void render(
      ID3D11Texture2D *source, ID3D11Texture2D *destination,
      ID3D11VertexShader *vs, ID3D11PixelShader *ps,
      UINT width, UINT height
    ) {
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11RenderTargetView> rtv;
      ASSERT_EQ(device->CreateShaderResourceView(source, nullptr, &srv), S_OK);
      ASSERT_EQ(device->CreateRenderTargetView(destination, nullptr, &rtv), S_OK);
      context->PSSetShaderResources(0, 1, srv.GetAddressOf());
      context->OMSetRenderTargets(1, rtv.GetAddressOf(), nullptr);
      context->VSSetShader(vs, nullptr, 0);
      context->PSSetShader(ps, nullptr, 0);
      const D3D11_VIEWPORT viewport {0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
      context->RSSetViewports(1, &viewport);
      context->Draw(3, 0);
      context->OMSetRenderTargets(0, nullptr, nullptr);
      ID3D11ShaderResourceView *null_srv = nullptr;
      context->PSSetShaderResources(0, 1, &null_srv);
    }

    std::vector<uv_pixel_t> read_uv(ID3D11Texture2D *source) {
      D3D11_TEXTURE2D_DESC desc {};
      source->GetDesc(&desc);
      desc.BindFlags = 0;
      desc.Usage = D3D11_USAGE_STAGING;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      EXPECT_EQ(device->CreateTexture2D(&desc, nullptr, &staging), S_OK);
      context->CopyResource(staging.Get(), source);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const auto status = context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
      EXPECT_EQ(status, S_OK);
      if (FAILED(status)) {
        return {};
      }
      std::vector<uv_pixel_t> result(desc.Width * desc.Height);
      for (UINT y = 0; y < desc.Height; ++y) {
        std::memcpy(result.data() + y * desc.Width,
                    static_cast<const std::byte *>(mapped.pData) + y * mapped.RowPitch,
                    desc.Width * sizeof(uv_pixel_t));
      }
      context->Unmap(staging.Get(), 0);
      return result;
    }

    std::vector<std::array<float, 4>> render_float_pixels(
      ID3D11Texture2D *source,
      const std::string &shader,
      UINT width,
      UINT height
    ) {
      auto vs_code = compile("convert_yuv420_planar_y_vs.hlsl", "main_vs", "vs_5_0");
      auto ps_code = compile(shader, "main_ps", "ps_5_0");
      if (!vs_code || !ps_code) {
        return {};
      }
      ComPtr<ID3D11VertexShader> vs;
      ComPtr<ID3D11PixelShader> ps;
      EXPECT_EQ(device->CreateVertexShader(vs_code->GetBufferPointer(), vs_code->GetBufferSize(), nullptr, &vs), S_OK);
      EXPECT_EQ(device->CreatePixelShader(ps_code->GetBufferPointer(), ps_code->GetBufferSize(), nullptr, &ps), S_OK);
      auto output = texture(width, height, DXGI_FORMAT_R32G32B32A32_FLOAT);
      render(source, output.Get(), vs.Get(), ps.Get(), width, height);
      D3D11_TEXTURE2D_DESC desc {};
      output->GetDesc(&desc);
      desc.BindFlags = 0;
      desc.Usage = D3D11_USAGE_STAGING;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      EXPECT_EQ(device->CreateTexture2D(&desc, nullptr, &staging), S_OK);
      context->CopyResource(staging.Get(), output.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const auto mapped_status = context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
      EXPECT_EQ(mapped_status, S_OK);
      if (FAILED(mapped_status)) {
        return {};
      }
      std::vector<std::array<float, 4>> result(width * height);
      for (UINT y = 0; y < height; ++y) {
        std::memcpy(result.data() + y * width, static_cast<const std::byte *>(mapped.pData) + y * mapped.RowPitch, width * sizeof(result[0]));
      }
      context->Unmap(staging.Get(), 0);
      return result;
    }

    void compare_to_independent_eyes(
      const std::string &converter,
      const std::string &suffix,
      const std::vector<rgba_pixel_t> &pixels,
      bool encoded_sdr = false,
      bool source_is_hdr = false,
      bool packed = true,
      bool target_is_hdr = false
    ) {
      constexpr UINT width = 16, height = 6, eye_width = width / 2;
      ASSERT_EQ(pixels.size(), width * height);
      auto sdr = constants(std::array<std::uint32_t, 4> {
        0u,
        source_is_hdr ? 1u : 0u,
        std::bit_cast<std::uint32_t>(203.0f / 80.0f),
        target_is_hdr ? 1u : 0u,
      });
      context->PSSetConstantBuffers(1, 1, sdr.GetAddressOf());
      ComPtr<ID3D11Texture2D> source;
      if (encoded_sdr) {
        std::vector<std::array<std::uint8_t, 4>> unorm(pixels.size());
        for (std::size_t i = 0; i < pixels.size(); ++i) {
          for (std::size_t c = 0; c < 4; ++c) {
            const auto source_channel = c == 0 ? 2 : (c == 2 ? 0 : c);
            unorm[i][c] = pixels[i][source_channel] == 0 ? 0 : 255;
          }
        }
        source = texture(width, height, DXGI_FORMAT_B8G8R8A8_UNORM, unorm.data(), width * 4);
      } else {
        source = texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels.data(), width * sizeof(rgba_pixel_t));
      }
      ASSERT_TRUE(source);

      auto y_vs_code = compile("convert_yuv420_planar_y_vs.hlsl", "main_vs", "vs_5_0");
      ASSERT_TRUE(y_vs_code);
      ComPtr<ID3D11VertexShader> y_vs;
      ASSERT_EQ(device->CreateVertexShader(y_vs_code->GetBufferPointer(), y_vs_code->GetBufferSize(), nullptr, &y_vs), S_OK);
      auto point_transfer = inline_pixel_shader(
        "#include \"include/" + converter + "\"\n"
        "Texture2D image : register(t0);\n"
        "float4 main_ps(float4 position : SV_Position) : SV_Target {\n"
        "return float4(CONVERT_FUNCTION(image.Load(int3(int2(position.xy), 0)).rgb), 1); }\n");
      ASSERT_TRUE(point_transfer);
      auto transferred = texture(width, height, DXGI_FORMAT_R32G32B32A32_FLOAT);
      render(source.Get(), transferred.Get(), y_vs.Get(), point_transfer.Get(), width, height);

      const video::sunshine_colorspace_t colorspace {
        suffix == "_perceptual_quantizer" || target_is_hdr ? video::colorspace_e::bt2020 : video::colorspace_e::rec709,
        false,
        10,
      };
      auto matrix = constants(*video::color_vectors_from_colorspace(colorspace, true));
      context->PSSetConstantBuffers(0, 1, matrix.GetAddressOf());

      // The oracle transforms each real source texel first, then filters each eye as a separate
      // texture with the actual production type0 vertex shader, linear sampler and chroma body.
      // Texture clamping makes it impossible for this oracle to sample the other eye. Ordinary
      // 2D uses a single full-width reference texture, with no artificial boundary in the center.
      auto reference_ps = inline_pixel_shader(
        "#define CONVERT_FUNCTION(value) (value)\n#define LEFT_SUBSAMPLING\n"
        "#include \"include/convert_yuv420_packed_uv_ps_base.hlsl\"\n");
      ASSERT_TRUE(reference_ps);
      auto uv_vs_code = compile("convert_yuv420_packed_uv_type0_vs.hlsl", "main_vs", "vs_5_0");
      ASSERT_TRUE(uv_vs_code);
      ComPtr<ID3D11VertexShader> uv_vs;
      ASSERT_EQ(device->CreateVertexShader(uv_vs_code->GetBufferPointer(), uv_vs_code->GetBufferSize(), nullptr, &uv_vs), S_OK);
      std::array<std::vector<uv_pixel_t>, 2> expected;
      const UINT reference_width = packed ? eye_width : width;
      for (UINT eye = 0; eye < (packed ? 2u : 1u); ++eye) {
        auto eye_source = texture(reference_width, height, DXGI_FORMAT_R32G32B32A32_FLOAT);
        const D3D11_BOX box {eye * reference_width, 0, 0, (eye + 1) * reference_width, height, 1};
        context->CopySubresourceRegion(eye_source.Get(), 0, 0, 0, 0, transferred.Get(), 0, &box);
        auto offsets = constants(chroma_constants_t {1.0f / reference_width, 1.0f / height, 0, 0, 0});
        context->VSSetConstantBuffers(0, 1, offsets.GetAddressOf());
        context->PSSetConstantBuffers(2, 1, offsets.GetAddressOf());
        auto uv = texture(reference_width / 2, height / 2, DXGI_FORMAT_R16G16_UNORM);
        render(eye_source.Get(), uv.Get(), uv_vs.Get(), reference_ps.Get(), reference_width / 2, height / 2);
        expected[eye] = read_uv(uv.Get());
        ASSERT_EQ(expected[eye].size(), reference_width / 2 * height / 2);
      }

      for (const std::string layout : {"type0", "type0s"}) {
        SCOPED_TRACE(layout + suffix);
        auto vs_code = compile("convert_yuv420_packed_uv_" + layout + "_vs.hlsl", "main_vs", "vs_5_0");
        auto ps_code = compile("convert_yuv420_packed_uv_" + layout + "_ps" + suffix + ".hlsl", "main_ps", "ps_5_0");
        ASSERT_TRUE(vs_code);
        ASSERT_TRUE(ps_code);
        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        ASSERT_EQ(device->CreateVertexShader(vs_code->GetBufferPointer(), vs_code->GetBufferSize(), nullptr, &vs), S_OK);
        ASSERT_EQ(device->CreatePixelShader(ps_code->GetBufferPointer(), ps_code->GetBufferSize(), nullptr, &ps), S_OK);
        auto offsets = constants(chroma_constants_t {1.0f / width, 1.0f / height, width, height, packed ? eye_width : 0u});
        context->VSSetConstantBuffers(0, 1, offsets.GetAddressOf());
        context->PSSetConstantBuffers(2, 1, offsets.GetAddressOf());
        auto uv = texture(width / 2, height / 2, DXGI_FORMAT_R16G16_UNORM);
        render(source.Get(), uv.Get(), vs.Get(), ps.Get(), width / 2, height / 2);
        const auto actual = read_uv(uv.Get());
        ASSERT_EQ(actual.size(), width / 2 * height / 2);
        for (UINT y = 0; y < height / 2; ++y) {
          for (UINT x = 0; x < width / 2; ++x) {
            const auto &reference = expected[x / (reference_width / 2)][y * (reference_width / 2) + x % (reference_width / 2)];
            for (UINT channel = 0; channel < 2; ++channel) {
              // One UNORM16 storage unit accommodates floating-point filter reassociation.
              EXPECT_NEAR(actual[y * (width / 2) + x][channel], reference[channel], 1)
                << "UV(" << x << ',' << y << ") channel " << channel;
            }
          }
        }
      }
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11SamplerState> sampler;
  };

  TEST_F(HostSbsChromaGpuTest, HdrRowsConvertBeforeTheSamplerCanMixLight) {
    std::vector<rgba_pixel_t> pixels(16 * 6);
    for (UINT y = 0; y < 6; ++y) {
      for (UINT x = 0; x < 16; ++x) {
        pixels[y * 16 + x] = {static_cast<std::uint16_t>(y % 2 ? 0x4800 : 0), 0, 0, 0x3c00};
      }
    }
    compare_to_independent_eyes("convert_perceptual_quantizer_base.hlsl", "_perceptual_quantizer", pixels);
  }

  TEST_F(HostSbsChromaGpuTest, LinearSdrEdgesConvertBeforeChromaFiltering) {
    std::vector<rgba_pixel_t> pixels(16 * 6);
    for (UINT y = 0; y < 6; ++y) {
      for (UINT x = 0; x < 16; ++x) {
        pixels[y * 16 + x] = {static_cast<std::uint16_t>((x + y) % 2 ? 0x3c00 : 0), 0, 0, 0x3c00};
      }
    }
    compare_to_independent_eyes("convert_linear_base.hlsl", "_linear", pixels);
  }

  TEST_F(HostSbsChromaGpuTest, HdrToSdrToneMapAlsoPrecedesChromaFiltering) {
    std::vector<rgba_pixel_t> pixels(16 * 6);
    for (UINT y = 0; y < 6; ++y) {
      for (UINT x = 0; x < 16; ++x) {
        pixels[y * 16 + x] = {static_cast<std::uint16_t>(y % 2 ? 0x4800 : 0), 0, 0, 0x3c00};
      }
    }
    compare_to_independent_eyes("convert_linear_base.hlsl", "_linear", pixels, false, true);
  }

  TEST_F(HostSbsChromaGpuTest, OrdinaryOneToOneHdrUsesWholeTextureFiltering) {
    std::vector<rgba_pixel_t> pixels(16 * 6);
    for (UINT y = 0; y < 6; ++y) {
      for (UINT x = 0; x < 16; ++x) {
        pixels[y * 16 + x] = {static_cast<std::uint16_t>(y % 2 ? 0x4800 : 0), 0,
                              static_cast<std::uint16_t>(x < 8 ? 0 : 0x3c00), 0x3c00};
      }
    }
    compare_to_independent_eyes("convert_perceptual_quantizer_base.hlsl", "_perceptual_quantizer", pixels, false, false, false);
  }

  TEST_F(HostSbsChromaGpuTest, BothEyesKeepTheirOwnBoundaryChroma) {
    std::vector<rgba_pixel_t> pixels(16 * 6);
    for (UINT y = 0; y < 6; ++y) {
      for (UINT x = 0; x < 16; ++x) {
        pixels[y * 16 + x] = x < 8 ? rgba_pixel_t {0x3c00, 0, 0, 0x3c00} :
                                           rgba_pixel_t {0, 0, 0x3c00, 0x3c00};
      }
    }
    compare_to_independent_eyes("convert_base.hlsl", "", pixels, true);
    compare_to_independent_eyes("convert_perceptual_quantizer_base.hlsl", "_perceptual_quantizer", pixels);
  }

  TEST_F(HostSbsChromaGpuTest, EncodedSdrToHdrConvertsBeforeFilteringEachEye) {
    std::vector<rgba_pixel_t> pixels(16 * 6);
    for (UINT y = 0; y < 6; ++y) {
      for (UINT x = 0; x < 16; ++x) {
        const auto intensity = static_cast<std::uint16_t>((x + y) % 2 ? 0x3c00 : 0);
        pixels[y * 16 + x] = x < 8 ? rgba_pixel_t {intensity, 0, 0, 0x3c00} :
                                     rgba_pixel_t {0, 0, intensity, 0x3c00};
      }
    }
    compare_to_independent_eyes("convert_base.hlsl", "", pixels, true, false, true, true);
  }

  TEST_F(HostSbsChromaGpuTest, EncodedSdrToHdrYuvUsesDeclaredPqAndReferenceWhite) {
    const std::array<std::array<std::uint8_t, 4>, 4> pixels {{
      {0, 0, 0, 255},
      {64, 64, 64, 255},
      {128, 128, 128, 255},
      {255, 255, 255, 255},
    }};
    auto source = texture(4, 1, DXGI_FORMAT_R8G8B8A8_UNORM, pixels.data(), sizeof(pixels));
    const auto *vectors = video::color_vectors_from_colorspace({video::colorspace_e::bt2020, false, 10}, true);
    auto matrix = constants(*vectors);
    context->PSSetConstantBuffers(0, 1, matrix.GetAddressOf());
    for (const float white_nits : {203.0f, 320.0f}) {
      SCOPED_TRACE(white_nits);
      // A display-derived HDR flag must not tone-map the actual encoded SDR input.
      auto transform = constants(std::array<std::uint32_t, 4> {0, 1, std::bit_cast<std::uint32_t>(white_nits / 80.0f), 1});
      context->PSSetConstantBuffers(1, 1, transform.GetAddressOf());
      const auto output = render_float_pixels(source.Get(), "convert_yuv420_planar_y_ps.hlsl", 4, 1);
      ASSERT_EQ(output.size(), pixels.size());
      for (std::size_t i = 0; i < pixels.size(); ++i) {
        const double encoded = pixels[i][0] / 255.0;
        const double linear = encoded <= 0.04045 ? encoded / 12.92 : std::pow((encoded + 0.055) / 1.055, 2.4);
        // ST2084 absolute luminance reference; this is independent of the production HLSL.
        const double powered = std::pow(linear * white_nits / 10000.0, 2610.0 / 16384.0);
        const double pq = std::pow((3424.0 / 4096.0 + (2413.0 / 128.0) * powered) / (1.0 + (2392.0 / 128.0) * powered), 2523.0 / 32.0);
        const double y = (pq * (vectors->color_vec_y[0] + vectors->color_vec_y[1] + vectors->color_vec_y[2]) + vectors->color_vec_y[3]) *
                           vectors->range_y[0] +
                         vectors->range_y[1];
        EXPECT_NEAR(output[i][0], y, 0.0001) << "pixel " << i;
      }
    }
  }

  TEST_F(HostSbsChromaGpuTest, ExplicitHdrToSdrRetainsHighlightsInLocalRgbAndYuv) {
    const std::array<rgba_pixel_t, 4> pixels {{
      {0x3800, 0x3800, 0x3800, 0x3c00},
      {0x4000, 0x4000, 0x4000, 0x3c00},
      {0x4400, 0x4400, 0x4400, 0x3c00},
      {0x4800, 0x4800, 0x4800, 0x3c00},
    }};
    auto source = texture(4, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels.data(), sizeof(pixels));
    auto white = constants(std::array<float, 4> {2, 0, 0, 0});
    context->PSSetConstantBuffers(1, 1, white.GetAddressOf());
    const auto rgb = render_float_pixels(source.Get(), "rgb_present_hdr_to_srgb_ps.hlsl", 4, 1);
    ASSERT_EQ(rgb.size(), pixels.size());
    const auto *vectors = video::color_vectors_from_colorspace({video::colorspace_e::rec709, false, 8}, true);
    auto matrix = constants(*vectors);
    auto transform = constants(std::array<std::uint32_t, 4> {0, 1, std::bit_cast<std::uint32_t>(2.0f), 0});
    context->PSSetConstantBuffers(0, 1, matrix.GetAddressOf());
    context->PSSetConstantBuffers(1, 1, transform.GetAddressOf());
    const auto yuv = render_float_pixels(source.Get(), "convert_yuv420_planar_y_ps_linear.hlsl", 4, 1);
    ASSERT_EQ(yuv.size(), pixels.size());
    const std::array<double, 4> input {0.5, 2, 4, 8};
    std::array<std::array<float, 4>, 4> tone_mapped {};
    for (std::size_t i = 0; i < input.size(); ++i) {
      const double normalized = input[i] / 2;
      const float mapped = normalized <= 0.5 ? normalized : 1 - 0.5 * std::exp(-2 * (normalized - 0.5));
      tone_mapped[i] = {mapped, mapped, mapped, 1};
    }
    // Keep the established local sRGB approximation. Isolate tone mapping by applying the
    // production SDR transfer to independently calculated scene-linear reference values.
    auto reference_source = texture(4, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, tone_mapped.data(), sizeof(tone_mapped));
    const auto rgb_reference = render_float_pixels(reference_source.Get(), "rgb_present_linear_to_srgb_ps.hlsl", 4, 1);
    ASSERT_EQ(rgb_reference.size(), pixels.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
      const double normalized = input[i] / 2;
      const double mapped = normalized <= 0.5 ? normalized : 1 - 0.5 * std::exp(-2 * (normalized - 0.5));
      const double bt709 = mapped < 0.018 ? 4.5 * mapped : 1.099 * std::pow(mapped, 0.45) - 0.099;
      EXPECT_NEAR(rgb[i][0], rgb_reference[i][0], 0.0001);
      const double expected_y = (bt709 * (vectors->color_vec_y[0] + vectors->color_vec_y[1] + vectors->color_vec_y[2]) + vectors->color_vec_y[3]) *
                                  vectors->range_y[0] +
                                vectors->range_y[1];
      EXPECT_NEAR(yuv[i][0], expected_y, 0.0001);
      if (i) {
        EXPECT_GT(rgb[i][0], rgb[i - 1][0]);
      }
      EXPECT_LT(rgb[i][0], 1.0f);
    }
  }

  TEST_F(HostSbsChromaGpuTest, LocalSdrToHdrRestoresConfiguredScRgbWhite) {
    const std::array<std::uint8_t, 4> pixel {255, 255, 255, 255};
    auto source = texture(1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, pixel.data(), sizeof(pixel));
    auto white = constants(std::array<float, 4> {203.0f / 80.0f, 0, 0, 0});
    context->PSSetConstantBuffers(1, 1, white.GetAddressOf());
    const auto rgb = render_float_pixels(source.Get(), "rgb_present_srgb_to_linear_ps.hlsl", 1, 1);
    ASSERT_EQ(rgb.size(), 1u);
    for (int channel = 0; channel < 3; ++channel) {
      EXPECT_NEAR(rgb[0][channel], 203.0f / 80.0f, 0.00001);
    }
  }
}  // namespace

#endif
