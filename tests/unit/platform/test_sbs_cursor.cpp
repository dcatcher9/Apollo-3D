// SPDX-License-Identifier: GPL-3.0-only
#ifdef _WIN32
  #include "src/platform/windows/sbs_cursor.h"

  #include <algorithm>
  #include <array>
  #include <cmath>
  #include <cstring>
  #include <d3d11_1.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <fstream>
  #include <gtest/gtest.h>
  #include <iterator>
  #include <limits>
  #include <string>
  #include <wrl/client.h>

namespace {
  namespace cursor = platf::sbs_cursor;
  using Microsoft::WRL::ComPtr;

  cursor::snapshot_t snapshot(float x = 1, float y = 2, float width = 2, float height = 2) {
    auto shape = std::make_shared<cursor::shape_t>();
    shape->width = 2;
    shape->height = 2;
    shape->alpha_bgra.assign(16, 255);
    return {shape, {x, y, width, height, 0, 1}, 8, 8, DXGI_MODE_ROTATION_IDENTITY, true};
  }
}  // namespace

TEST(SbsCursorShape, ColorRowsDiscardPaddingAndPreserveAlpha) {
  const DXGI_OUTDUPL_POINTER_SHAPE_INFO info {DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR, 2, 2, 12, {}};
  const std::array<std::uint8_t, 24> bytes {
    1,
    2,
    3,
    255,
    4,
    5,
    6,
    128,
    90,
    91,
    92,
    93,
    7,
    8,
    9,
    0,
    10,
    11,
    12,
    64,
    94,
    95,
    96,
    97
  };
  const auto decoded = cursor::decode_shape(info, bytes.data(), bytes.size());
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->width, 2u);
  EXPECT_EQ(decoded->height, 2u);
  EXPECT_EQ(decoded->alpha_bgra, (std::vector<std::uint8_t> {1, 2, 3, 255, 4, 5, 6, 128, 7, 8, 9, 0, 10, 11, 12, 64}));
  EXPECT_TRUE(decoded->xor_bgra.empty());
}

TEST(SbsCursorShape, MaskedRowsSplitReplaceAndXorPixelsWithoutReadingPadding) {
  const DXGI_OUTDUPL_POINTER_SHAPE_INFO info {DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR, 2, 2, 12, {}};
  const std::array<std::uint8_t, 24> bytes {
    1,
    2,
    3,
    0,
    4,
    5,
    6,
    255,
    90,
    91,
    92,
    93,
    7,
    8,
    9,
    255,
    10,
    11,
    12,
    0,
    94,
    95,
    96,
    97
  };
  const auto decoded = cursor::decode_shape(info, bytes.data(), bytes.size());
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->alpha_bgra, (std::vector<std::uint8_t> {1, 2, 3, 255, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 255}));
  EXPECT_EQ(decoded->xor_bgra, (std::vector<std::uint8_t> {0, 0, 0, 0, 4, 5, 6, 255, 7, 8, 9, 255, 0, 0, 0, 0}));
}

TEST(SbsCursorShape, MonochromeMasksRespectRowPitchAndPartialBytes) {
  const DXGI_OUTDUPL_POINTER_SHAPE_INFO info {DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME, 3, 4, 2, {}};
  // Two AND rows followed by two XOR rows. Set padding bits must never become cursor pixels.
  const std::array<std::uint8_t, 8> bytes {0x7f, 0xff, 0x9f, 0xff, 0xbf, 0xff, 0x5f, 0xff};
  const auto decoded = cursor::decode_shape(info, bytes.data(), bytes.size());
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->height, 2u);
  EXPECT_EQ(decoded->alpha_bgra, (std::vector<std::uint8_t> {255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255}));
  EXPECT_EQ(decoded->xor_bgra, (std::vector<std::uint8_t> {0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(SbsCursorShape, RejectsInvalidDimensionsPitchMaskAndTruncatedInput) {
  DXGI_OUTDUPL_POINTER_SHAPE_INFO info {DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR, 2, 2, 12, {}};
  const std::array<std::uint8_t, 24> bytes {};
  EXPECT_FALSE(cursor::decode_shape(info, bytes.data(), bytes.size() - 1));
  EXPECT_FALSE(cursor::decode_shape(info, nullptr, bytes.size()));
  info.Pitch = 7;
  EXPECT_FALSE(cursor::decode_shape(info, bytes.data(), bytes.size()));
  info.Pitch = 0;
  EXPECT_FALSE(cursor::decode_shape(info, bytes.data(), bytes.size()));
  info.Pitch = std::numeric_limits<UINT>::max();
  EXPECT_FALSE(cursor::decode_shape(info, bytes.data(), bytes.size()));
  info.Pitch = 12;
  info.Width = 0;
  EXPECT_FALSE(cursor::decode_shape(info, bytes.data(), bytes.size()));
  info.Width = D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION + 1;
  EXPECT_FALSE(cursor::decode_shape(info, bytes.data(), bytes.size()));
  info.Width = 2;
  info.Type = DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME;
  info.Height = 3;
  EXPECT_FALSE(cursor::decode_shape(info, bytes.data(), bytes.size()));
  info.Height = 2;
  info.Type = 0;
  EXPECT_FALSE(cursor::decode_shape(info, bytes.data(), bytes.size()));
  info.Type = DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR;
  auto invalid_mask = bytes;
  invalid_mask[3] = 128;
  EXPECT_FALSE(cursor::decode_shape(info, invalid_mask.data(), invalid_mask.size()));
}

TEST(SbsCursorPlacement, ScalesBothEyesAndKeepsIndependentClipRectangles) {
  const auto p = cursor::place_cursor(snapshot(-1, 2, 2, 3), 32, 24);
  ASSERT_TRUE(p);
  EXPECT_FLOAT_EQ(p->viewports[0].TopLeftX, -2);
  EXPECT_FLOAT_EQ(p->viewports[1].TopLeftX, 14);
  EXPECT_FLOAT_EQ(p->viewports[0].TopLeftY, 6);
  EXPECT_FLOAT_EQ(p->viewports[0].Width, 4);
  EXPECT_FLOAT_EQ(p->viewports[0].Height, 9);
  EXPECT_EQ(p->scissors[0].right, 16);
  EXPECT_EQ(p->scissors[1].left, 16);
  EXPECT_EQ(p->scissors[1].right, 32);
}

TEST(SbsCursorPlacement, RejectsUnsupportedRotationAndInvalidCoordinates) {
  auto s = snapshot();
  for (const auto rotation : {DXGI_MODE_ROTATION_ROTATE90, DXGI_MODE_ROTATION_ROTATE180, DXGI_MODE_ROTATION_ROTATE270}) {
    s.rotation = rotation;
    EXPECT_FALSE(cursor::place_cursor(s, 16, 8));
  }
  s.rotation = DXGI_MODE_ROTATION_UNSPECIFIED;
  EXPECT_TRUE(cursor::place_cursor(s, 16, 8));
  s.viewport.Width = -1;
  EXPECT_FALSE(cursor::place_cursor(s, 16, 8));
  s = snapshot();
  s.viewport.TopLeftX = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(cursor::place_cursor(s, 16, 8));
  s = snapshot();
  EXPECT_FALSE(cursor::place_cursor(s, 15, 8));
  s.capture_width = 0;
  EXPECT_FALSE(cursor::place_cursor(s, 16, 8));
}

namespace {
  // Compile the same files/blobs supplied by production, including their shared color helpers.
  struct shader_includes_t: ID3DInclude {
    std::filesystem::path root;

    explicit shader_includes_t(std::filesystem::path value):
        root(std::move(value)) {}

    HRESULT STDMETHODCALLTYPE Open(D3D_INCLUDE_TYPE, LPCSTR name, LPCVOID, LPCVOID *data, UINT *bytes) override {
      std::ifstream input(root / name, std::ios::binary);
      if (!input) {
        return E_FAIL;
      }
      const std::string text((std::istreambuf_iterator<char>(input)), {});
      auto *copy = new char[text.size()];
      std::memcpy(copy, text.data(), text.size());
      *data = copy;
      *bytes = static_cast<UINT>(text.size());
      return S_OK;
    }

    HRESULT STDMETHODCALLTYPE Close(LPCVOID data) override {
      delete[] static_cast<const char *>(data);
      return S_OK;
    }
  };

  std::filesystem::path shader_directory() {
    for (auto base : {std::filesystem::absolute(__FILE__).parent_path(), std::filesystem::current_path()}) {
      for (int i = 0; i < 8; ++i) {
        auto path = base / "src_assets/windows/assets/shaders/directx";
        if (std::filesystem::exists(path / "cursor_vs.hlsl")) {
          return path;
        }
        base = base.parent_path();
      }
    }
    return {};
  }

  float half_float(std::uint16_t value) {
    const auto sign = (value & 0x8000) ? -1.0f : 1.0f;
    const int exponent = (value >> 10) & 31;
    const auto fraction = value & 1023;
    if (exponent == 31) {
      return fraction ? std::numeric_limits<float>::quiet_NaN() : sign * std::numeric_limits<float>::infinity();
    }
    return sign * (exponent ? std::ldexp(1.0f + fraction / 1024.0f, exponent - 15) : std::ldexp(float(fraction), -24));
  }

  struct pixels_t {
    UINT width = 0, height = 0;
    DXGI_FORMAT format {};
    std::vector<std::uint8_t> bytes;

    std::array<float, 4> at(UINT x, UINT y) const {
      const auto offset = std::size_t(y) * width + x;
      std::array<float, 4> rgba {};
      if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
        std::array<std::uint16_t, 4> raw {};
        std::memcpy(raw.data(), bytes.data() + offset * 8, 8);
        for (int c = 0; c < 4; ++c) {
          rgba[c] = half_float(raw[c]);
        }
      } else if (format == DXGI_FORMAT_R10G10B10A2_UNORM) {
        std::uint32_t raw;
        std::memcpy(&raw, bytes.data() + offset * 4, 4);
        rgba = {float(raw & 1023) / 1023, float((raw >> 10) & 1023) / 1023, float((raw >> 20) & 1023) / 1023, float(raw >> 30) / 3};
      } else {
        const auto *raw = bytes.data() + offset * 4;
        const bool bgra = format == DXGI_FORMAT_B8G8R8A8_UNORM;
        rgba = {raw[bgra ? 2 : 0] / 255.0f, raw[1] / 255.0f, raw[bgra ? 0 : 2] / 255.0f, raw[3] / 255.0f};
      }
      return rgba;
    }
  };

  class SbsCursorGpu: public testing::Test {
  protected:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3DBlob> vs, ps, hdr;
    ComPtr<ID3D11Texture2D> source;
    ComPtr<ID3D11ShaderResourceView> source_view;
    std::unique_ptr<cursor::compositor_t> compositor;
    const std::array<float, 4> background {0.125f, 0.25f, 0.5f, 0.75f};

    void SetUp() override {
      const D3D_FEATURE_LEVEL requested = D3D_FEATURE_LEVEL_11_0;
      const auto status = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, &requested, 1, D3D11_SDK_VERSION, &device, nullptr, &context);
      if (FAILED(status)) {
        GTEST_SKIP() << "Hardware D3D11 unavailable: " << std::hex << status;
      }
      ComPtr<ID3D11Device1> device1;
      ASSERT_EQ(device.As(&device1), S_OK);
      const auto root = shader_directory();
      ASSERT_FALSE(root.empty()) << "Production cursor shader sources must be available";
      shader_includes_t includes(root);
      const auto compile = [&](const wchar_t *name, const char *entry, const char *profile, ComPtr<ID3DBlob> &output) {
        ComPtr<ID3DBlob> errors;
        const auto result = D3DCompileFromFile((root / name).c_str(), nullptr, &includes, entry, profile, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &output, &errors);
        EXPECT_EQ(result, S_OK) << (errors ? static_cast<const char *>(errors->GetBufferPointer()) : "No compiler diagnostic");
        return SUCCEEDED(result);
      };
      ASSERT_TRUE(compile(L"cursor_vs.hlsl", "main_vs", "vs_5_0", vs));
      ASSERT_TRUE(compile(L"cursor_ps.hlsl", "main_ps", "ps_5_0", ps));
      ASSERT_TRUE(compile(L"cursor_ps_normalize_white.hlsl", "main_ps", "ps_5_0", hdr));
      compositor = std::make_unique<cursor::compositor_t>(device.Get(), context.Get(), vs.Get(), ps.Get(), hdr.Get());
      make_source(DXGI_FORMAT_R8G8B8A8_UNORM);
    }

    void make_source(DXGI_FORMAT format, UINT width = 16, UINT height = 8) {
      source.Reset();
      source_view.Reset();
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.ArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Format = format;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
      ASSERT_EQ(device->CreateTexture2D(&desc, nullptr, &source), S_OK);
      ASSERT_EQ(device->CreateShaderResourceView(source.Get(), nullptr, &source_view), S_OK);
      ComPtr<ID3D11RenderTargetView> target;
      ASSERT_EQ(device->CreateRenderTargetView(source.Get(), nullptr, &target), S_OK);
      context->ClearRenderTargetView(target.Get(), background.data());
    }

    pixels_t read(ID3D11Texture2D *texture) {
      pixels_t result;
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      result.width = desc.Width;
      result.height = desc.Height;
      result.format = desc.Format;
      const UINT pixel_bytes = desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 4;
      result.bytes.resize(std::size_t(desc.Width) * desc.Height * pixel_bytes);
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = 0;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      EXPECT_EQ(device->CreateTexture2D(&desc, nullptr, &staging), S_OK);
      if (!staging) {
        return result;
      }
      context->CopyResource(staging.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE map {};
      EXPECT_EQ(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map), S_OK);
      if (!map.pData) {
        return result;
      }
      for (UINT y = 0; y < desc.Height; ++y) {
        std::memcpy(result.bytes.data() + std::size_t(y) * desc.Width * pixel_bytes, static_cast<const std::uint8_t *>(map.pData) + std::size_t(y) * map.RowPitch, desc.Width * pixel_bytes);
      }
      context->Unmap(staging.Get(), 0);
      return result;
    }

    void expect_rgb(const pixels_t &pixels, UINT x, UINT y, const std::array<float, 4> &expected, float tolerance = 0.004f) {
      const auto actual = pixels.at(x, y);
      for (int c = 0; c < 4; ++c) {
        EXPECT_NEAR(actual[c], expected[c], tolerance) << "x=" << x << " y=" << y << " channel=" << c;
      }
    }
  };
}  // namespace

TEST_F(SbsCursorGpu, PaddedDxgiRowsRenderBothEyesWithoutChangingTheSource) {
  const auto original = read(source.Get());
  for (UINT type : {UINT(DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR), UINT(DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR)}) {
    SCOPED_TRACE(type);
    const std::uint8_t alpha = type == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR ? 255 : 0;
    const DXGI_OUTDUPL_POINTER_SHAPE_INFO info {type, 2, 2, 12, {}};
    const std::array<std::uint8_t, 24> bytes {
      0,
      0,
      255,
      alpha,
      0,
      255,
      0,
      alpha,
      11,
      22,
      33,
      44,
      255,
      0,
      0,
      alpha,
      255,
      255,
      255,
      alpha,
      55,
      66,
      77,
      88
    };
    auto decoded = cursor::decode_shape(info, bytes.data(), bytes.size());
    ASSERT_TRUE(decoded);
    auto s = snapshot();
    s.shape = std::make_shared<cursor::shape_t>(std::move(*decoded));
    const auto result = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
    ASSERT_TRUE(result);
    const auto output = read(result->texture);
    const auto frame_alpha = original.at(1, 2)[3];
    for (UINT offset : {0u, 8u}) {
      expect_rgb(output, 1 + offset, 2, {1, 0, 0, frame_alpha});
      expect_rgb(output, 2 + offset, 2, {0, 1, 0, frame_alpha});
      expect_rgb(output, 1 + offset, 3, {0, 0, 1, frame_alpha});
      expect_rgb(output, 2 + offset, 3, {1, 1, 1, frame_alpha});
      expect_rgb(output, 3 + offset, 3, original.at(3 + offset, 3));
    }
    EXPECT_EQ(read(source.Get()).bytes, original.bytes);
  }
}

TEST_F(SbsCursorGpu, DrawsBothEyesWithoutMutatingSourceAndMoveDoesNotLeaveTrails) {
  const auto original = read(source.Get());
  auto s = snapshot();
  auto result = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(result);
  ASSERT_NE(result->texture, source.Get());
  const auto output = read(result->texture);
  for (UINT y = 0; y < 8; ++y) {
    for (UINT x = 0; x < 16; ++x) {
      const bool covered = (x % 8 == 1 || x % 8 == 2) && (y == 2 || y == 3);
      auto expected = original.at(x, y);
      if (covered) {
        expected[0] = expected[1] = expected[2] = 1;
      }
      expect_rgb(output, x, y, expected);
    }
  }
  EXPECT_EQ(read(source.Get()).bytes, original.bytes);
  s.viewport.TopLeftX = 4;
  const auto moved = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(moved);
  const auto moved_pixels = read(moved->texture);
  expect_rgb(moved_pixels, 1, 2, original.at(1, 2));
  expect_rgb(moved_pixels, 9, 2, original.at(9, 2));
  EXPECT_NEAR(moved_pixels.at(4, 2)[0], 1, 0.004);
  s.visible = false;
  const auto hidden = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(hidden);
  EXPECT_EQ(hidden->texture, source.Get());
  EXPECT_EQ(hidden->view, source_view.Get());
  EXPECT_EQ(read(hidden->texture).bytes, original.bytes);
  s.visible = true;
  s.viewport.TopLeftX = 2;
  const auto reentered = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(reentered);
  const auto reentered_pixels = read(reentered->texture);
  EXPECT_NEAR(reentered_pixels.at(2, 2)[0], 1, 0.004);
  expect_rgb(reentered_pixels, 4, 2, original.at(4, 2));
  EXPECT_EQ(read(source.Get()).bytes, original.bytes);
}

TEST_F(SbsCursorGpu, HiddenPathRequiresNoShadersOrGpuInitialization) {
  cursor::compositor_t uninitialized(nullptr, nullptr, nullptr, nullptr, nullptr);
  auto s = snapshot();
  s.visible = false;
  const auto result = uninitialized.compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->texture, source.Get());
  s.visible = true;
  EXPECT_FALSE(uninitialized.compose(source.Get(), source_view.Get(), s, false, 1));
}

TEST_F(SbsCursorGpu, EyeBoundaryClipsWithoutLeakingIntoOtherEyeAndScalesCaptureCoordinates) {
  auto s = snapshot(-1, 2, 2, 2);
  const auto original = read(source.Get());
  auto result = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(result);
  auto output = read(result->texture);
  EXPECT_NEAR(output.at(0, 2)[0], 1, 0.004);
  EXPECT_NEAR(output.at(8, 2)[0], 1, 0.004);
  expect_rgb(output, 7, 2, original.at(7, 2));
  s.viewport.TopLeftX = 7;
  result = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(result);
  output = read(result->texture);
  EXPECT_NEAR(output.at(7, 2)[0], 1, 0.004);
  expect_rgb(output, 8, 2, original.at(8, 2));
  make_source(DXGI_FORMAT_R8G8B8A8_UNORM, 32, 16);
  s = snapshot(1, 2, 2, 2);
  result = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(result);
  output = read(result->texture);
  EXPECT_NEAR(output.at(2, 4)[0], 1, 0.004);
  EXPECT_NEAR(output.at(18, 4)[0], 1, 0.004);
  EXPECT_NEAR(output.at(5, 7)[0], 1, 0.004);
  expect_rgb(output, 6, 7, read(source.Get()).at(6, 7));
}

TEST_F(SbsCursorGpu, SupportsTenBitSdrAndBgraAndReuploadsChangedShape) {
  auto s = snapshot();
  auto shape = std::make_shared<cursor::shape_t>(*s.shape);
  for (std::size_t i = 0; i < shape->alpha_bgra.size(); i += 4) {
    shape->alpha_bgra[i] = 32;
    shape->alpha_bgra[i + 1] = 96;
    shape->alpha_bgra[i + 2] = 192;
  }
  s.shape = shape;
  for (const auto format : {DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM}) {
    make_source(format);
    const auto original = read(source.Get());
    const auto result = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
    ASSERT_TRUE(result);
    const auto output = read(result->texture);
    expect_rgb(output, 1, 2, {192.0f / 255, 96.0f / 255, 32.0f / 255, original.at(1, 2)[3]});
    expect_rgb(output, 9, 2, output.at(1, 2));
    EXPECT_EQ(read(source.Get()).bytes, original.bytes);
  }
  s = snapshot();
  const auto updated = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(updated);
  EXPECT_NEAR(read(updated->texture).at(1, 2)[0], 1, 0.004);
}

TEST_F(SbsCursorGpu, FourKPerEyeTenBitSourceMapsDesktopCursorAndClipsAtEachEye) {
  make_source(DXGI_FORMAT_R10G10B10A2_UNORM, 7680, 2160);
  const auto unchanged = read(source.Get()).at(0, 0);
  auto s = snapshot(1919, 20, 2, 2);
  s.capture_width = 1920;
  s.capture_height = 1080;
  const auto result = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(result);
  const auto output = read(result->texture);
  auto white = unchanged;
  white[0] = white[1] = white[2] = 1;
  expect_rgb(output, 3838, 40, white, 0.001f);
  expect_rgb(output, 3839, 43, white, 0.001f);
  expect_rgb(output, 7678, 40, white, 0.001f);
  expect_rgb(output, 7679, 43, white, 0.001f);
  expect_rgb(output, 3837, 40, unchanged, 0.001f);
  expect_rgb(output, 3840, 40, unchanged, 0.001f);
  expect_rgb(output, 3841, 43, unchanged, 0.001f);
  expect_rgb(output, 3839, 44, unchanged, 0.001f);
}

TEST_F(SbsCursorGpu, HdrCursorDecodesSrgbBeforeWhiteScalingAndAlphaBlend) {
  make_source(DXGI_FORMAT_R16G16B16A16_FLOAT);
  const auto original = read(source.Get());
  auto s = snapshot();
  auto shape = std::make_shared<cursor::shape_t>(*s.shape);
  for (std::size_t i = 0; i < shape->alpha_bgra.size(); i += 4) {
    shape->alpha_bgra[i] = shape->alpha_bgra[i + 1] = shape->alpha_bgra[i + 2] = 128;
    shape->alpha_bgra[i + 3] = 128;
  }
  s.shape = shape;
  constexpr float white = 203.0f / 80;
  const auto result = compositor->compose(source.Get(), source_view.Get(), s, true, white);
  ASSERT_TRUE(result);
  const auto output = read(result->texture);
  const float alpha = 128.0f / 255;
  const float decoded = std::pow((alpha + 0.055f) / 1.055f, 2.4f);
  auto expected = original.at(1, 2);
  for (int c = 0; c < 3; ++c) {
    expected[c] = decoded * white * alpha + expected[c] * (1 - alpha);
  }
  expect_rgb(output, 1, 2, expected, 0.001f);
  expect_rgb(output, 9, 2, expected, 0.001f);
  EXPECT_EQ(read(source.Get()).bytes, original.bytes);
  s = snapshot();
  const auto white_result = compositor->compose(source.Get(), source_view.Get(), s, true, white);
  ASSERT_TRUE(white_result);
  EXPECT_NEAR(read(white_result->texture).at(1, 2)[0], white, 0.003);
}

TEST_F(SbsCursorGpu, MonochromeXorInvertsRgbAndPreservesAlpha) {
  auto s = snapshot();
  auto shape = std::make_shared<cursor::shape_t>(*s.shape);
  shape->xor_bgra = std::move(shape->alpha_bgra);
  shape->alpha_bgra.clear();
  s.shape = shape;
  const auto original = read(source.Get());
  const auto result = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(result);
  auto expected = original.at(1, 2);
  for (int c = 0; c < 3; ++c) {
    expected[c] = 1 - expected[c];
  }
  expect_rgb(read(result->texture), 1, 2, expected);
}

TEST_F(SbsCursorGpu, RestoresCallerPipelineStateAndRejectsMalformedVisibleShape) {
  D3D11_RASTERIZER_DESC rd {};
  rd.FillMode = D3D11_FILL_SOLID;
  rd.CullMode = D3D11_CULL_FRONT;
  rd.ScissorEnable = FALSE;
  ComPtr<ID3D11RasterizerState> raster;
  ASSERT_EQ(device->CreateRasterizerState(&rd, &raster), S_OK);
  context->RSSetState(raster.Get());
  const D3D11_VIEWPORT viewport {0, 0, 16, 8, 0.2f, 0.8f};
  context->RSSetViewports(1, &viewport);
  D3D11_SAMPLER_DESC sd {};
  sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
  sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
  ComPtr<ID3D11SamplerState> sampler;
  ASSERT_EQ(device->CreateSamplerState(&sd, &sampler), S_OK);
  context->PSSetSamplers(0, 1, sampler.GetAddressOf());
  D3D11_BUFFER_DESC bd {};
  bd.ByteWidth = 16;
  bd.Usage = D3D11_USAGE_DEFAULT;
  bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  ComPtr<ID3D11Buffer> buffer;
  ASSERT_EQ(device->CreateBuffer(&bd, nullptr, &buffer), S_OK);
  context->VSSetConstantBuffers(2, 1, buffer.GetAddressOf());
  context->PSSetConstantBuffers(1, 1, buffer.GetAddressOf());
  context->PSSetShaderResources(0, 1, source_view.GetAddressOf());
  context->PSSetShader(nullptr, nullptr, 0);
  const float blend_factor[4] {0.1f, 0.2f, 0.3f, 0.4f};
  context->OMSetBlendState(nullptr, blend_factor, 0x12345678);
  auto s = snapshot();
  const auto result = compositor->compose(source.Get(), source_view.Get(), s, false, 1);
  ASSERT_TRUE(result);
  ComPtr<ID3D11RasterizerState> restored_raster;
  context->RSGetState(&restored_raster);
  EXPECT_EQ(restored_raster.Get(), raster.Get());
  D3D11_VIEWPORT restored_viewport {};
  UINT count = 1;
  context->RSGetViewports(&count, &restored_viewport);
  EXPECT_EQ(std::memcmp(&viewport, &restored_viewport, sizeof(viewport)), 0);
  ComPtr<ID3D11SamplerState> restored_sampler;
  context->PSGetSamplers(0, 1, &restored_sampler);
  EXPECT_EQ(restored_sampler.Get(), sampler.Get());
  ComPtr<ID3D11Buffer> restored_buffer;
  context->VSGetConstantBuffers(2, 1, &restored_buffer);
  EXPECT_EQ(restored_buffer.Get(), buffer.Get());
  restored_buffer.Reset();
  context->PSGetConstantBuffers(1, 1, &restored_buffer);
  EXPECT_EQ(restored_buffer.Get(), buffer.Get());
  ComPtr<ID3D11ShaderResourceView> restored_view;
  context->PSGetShaderResources(0, 1, &restored_view);
  EXPECT_EQ(restored_view.Get(), source_view.Get());
  ComPtr<ID3D11PixelShader> restored_shader;
  context->PSGetShader(&restored_shader, nullptr, nullptr);
  EXPECT_EQ(restored_shader.Get(), nullptr);
  float restored_factor[4] {};
  UINT restored_mask = 0;
  ComPtr<ID3D11BlendState> restored_blend;
  context->OMGetBlendState(&restored_blend, restored_factor, &restored_mask);
  EXPECT_EQ(restored_blend.Get(), nullptr);
  EXPECT_EQ(restored_mask, 0x12345678u);
  EXPECT_EQ(std::memcmp(restored_factor, blend_factor, sizeof(blend_factor)), 0);
  auto malformed = std::make_shared<cursor::shape_t>(*s.shape);
  malformed->alpha_bgra.pop_back();
  s.shape = malformed;
  EXPECT_FALSE(compositor->compose(source.Get(), source_view.Get(), s, false, 1));
}
#endif
