#include "src/platform/windows/shutter_eye_pass.h"

#include <array>
#include <cmath>
#include <d3dcompiler.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {
  using Microsoft::WRL::ComPtr;
  using platf::dxgi::detail::parse_shutter_ghost_settings;
  using platf::dxgi::detail::shutter_eye_constants;
  using platf::dxgi::detail::shutter_eye_pass_t;
  using platf::dxgi::detail::shutter_ghost_settings_t;

  using colour_t = std::array<float, 3>;
  using picture_t = std::vector<colour_t>;

  constexpr unsigned eye_width = 16;
  constexpr unsigned eye_height = 64;

  // The meaning of the settings: the share of the lens window in which a line shows the other
  // eye's picture.
  float share_at(const shutter_ghost_settings_t &settings, unsigned row) {
    const float position = (static_cast<float>(row) + 0.5f) / eye_height;
    const float top = settings.top_extent > 0.0f ? settings.top_peak * std::clamp(1.0f - position / settings.top_extent, 0.0f, 1.0f) : 0.0f;
    const float bottom = settings.bottom_extent > 0.0f ? settings.bottom_peak * std::clamp(1.0f - (1.0f - position) / settings.bottom_extent, 0.0f, 1.0f) : 0.0f;
    return top + bottom;
  }

  float srgb_to_linear(float coded) {
    return coded <= 0.04045f ? coded / 12.92f : std::pow((coded + 0.055f) / 1.055f, 2.4f);
  }

  float linear_to_srgb(float linear) {
    return linear <= 0.0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
  }

  picture_t make_picture(colour_t (*colour)(unsigned column, unsigned row)) {
    picture_t picture(eye_width * eye_height);
    for (unsigned row = 0; row < eye_height; ++row) {
      for (unsigned column = 0; column < eye_width; ++column) {
        picture[row * eye_width + column] = colour(column, row);
      }
    }
    return picture;
  }

  class ShutterEyePassGpu: public ::testing::Test {
  protected:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    shutter_eye_pass_t pass;

    void SetUp() override {
      const D3D_FEATURE_LEVEL requested = D3D_FEATURE_LEVEL_11_0;
      const auto status = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, &requested, 1, D3D11_SDK_VERSION, &device, nullptr, &context);
      if (FAILED(status)) {
        GTEST_SKIP() << "Hardware D3D11 unavailable: " << std::hex << status;
      }
      const auto compile = [](const wchar_t *name, const char *entry, const char *profile, ComPtr<ID3DBlob> &code) {
        ComPtr<ID3DBlob> errors;
        const auto path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / name;
        const auto result = D3DCompileFromFile(path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, profile, D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
        EXPECT_EQ(result, S_OK) << (errors ? static_cast<const char *>(errors->GetBufferPointer()) : "No compiler diagnostic");
        return SUCCEEDED(result);
      };
      ComPtr<ID3DBlob> vertex_code;
      ComPtr<ID3DBlob> pixel_code;
      ASSERT_TRUE(compile(L"local_shutter_eye_vs.hlsl", "main_vs", "vs_5_0", vertex_code));
      ASSERT_TRUE(compile(L"local_shutter_eye_ps.hlsl", "main_ps", "ps_5_0", pixel_code));
      ASSERT_TRUE(pass.init(device.Get(), vertex_code->GetBufferPointer(), vertex_code->GetBufferSize(), pixel_code->GetBufferPointer(), pixel_code->GetBufferSize()));
    }

    // Draws both eyes of the pair and returns what each put on the screen, in linear light.
    // An HDR pair is linear; an SDR pair carries the sRGB curve in eight bits.
    std::array<picture_t, 2> show(const picture_t &left, const picture_t &right, bool hdr, const shutter_ghost_settings_t &settings, bool with_pair = true) {
      std::array<picture_t, 2> shown;
      const auto format = hdr ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
      ComPtr<ID3D11Texture2D> pair;
      if (with_pair) {
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = eye_width * 2;
        desc.Height = eye_height;
        desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
        desc.Format = format;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA data {};
        std::vector<float> linear_texels;
        std::vector<std::uint8_t> coded_texels;
        const auto at = [&](unsigned column, unsigned row) -> const colour_t & {
          return column < eye_width ? left[row * eye_width + column] : right[row * eye_width + column - eye_width];
        };
        for (unsigned row = 0; row < eye_height; ++row) {
          for (unsigned column = 0; column < eye_width * 2; ++column) {
            const auto &colour = at(column, row);
            if (hdr) {
              linear_texels.insert(linear_texels.end(), {colour[0], colour[1], colour[2], 1.0f});
            } else {
              const auto code = [](float linear) {
                return static_cast<std::uint8_t>(std::lround(std::clamp(linear_to_srgb(linear), 0.0f, 1.0f) * 255.0f));
              };
              coded_texels.insert(coded_texels.end(), {code(colour[2]), code(colour[1]), code(colour[0]), 255});
            }
          }
        }
        data.pSysMem = hdr ? static_cast<const void *>(linear_texels.data()) : static_cast<const void *>(coded_texels.data());
        data.SysMemPitch = eye_width * 2 * (hdr ? 16 : 4);
        EXPECT_EQ(device->CreateTexture2D(&desc, &data, &pair), S_OK);
      }
      for (const bool left_eye : {true, false}) {
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = eye_width;
        desc.Height = eye_height;
        desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
        desc.Format = format;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        ComPtr<ID3D11Texture2D> target;
        ComPtr<ID3D11RenderTargetView> target_view;
        EXPECT_EQ(device->CreateTexture2D(&desc, nullptr, &target), S_OK);
        EXPECT_EQ(device->CreateRenderTargetView(target.Get(), nullptr, &target_view), S_OK);
        EXPECT_TRUE(pass.draw(context.Get(), target_view.Get(), pair.Get(), left_eye, eye_width, eye_height, hdr, settings));

        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> staging;
        EXPECT_EQ(device->CreateTexture2D(&desc, nullptr, &staging), S_OK);
        context->CopyResource(staging.Get(), target.Get());
        D3D11_MAPPED_SUBRESOURCE mapped {};
        EXPECT_EQ(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), S_OK);
        auto &picture = shown[left_eye ? 0 : 1];
        picture.resize(eye_width * eye_height);
        for (unsigned row = 0; mapped.pData && row < eye_height; ++row) {
          const auto *line = static_cast<const std::uint8_t *>(mapped.pData) + row * mapped.RowPitch;
          for (unsigned column = 0; column < eye_width; ++column) {
            auto &colour = picture[row * eye_width + column];
            if (hdr) {
              const auto *texel = reinterpret_cast<const float *>(line) + column * 4;
              colour = {texel[0], texel[1], texel[2]};
            } else {
              const auto *texel = line + column * 4;
              colour = {srgb_to_linear(texel[2] / 255.0f), srgb_to_linear(texel[1] / 255.0f), srgb_to_linear(texel[0] / 255.0f)};
            }
          }
        }
        context->Unmap(staging.Get(), 0);
      }
      return shown;
    }

    // What an eye receives through a lens window that also takes in the other eye's picture.
    static colour_t received(const shutter_ghost_settings_t &settings, const picture_t &own, const picture_t &other, unsigned column, unsigned row) {
      const float share = share_at(settings, row);
      colour_t colour {};
      for (unsigned channel = 0; channel < 3; ++channel) {
        colour[channel] = (1.0f - share) * own[row * eye_width + column][channel] + share * other[row * eye_width + column][channel];
      }
      return colour;
    }
  };

  constexpr shutter_ghost_settings_t leaking_window {
    .top_peak = 0.2f,
    .top_extent = 0.25f,
    .bottom_peak = 0.25f,
    .bottom_extent = 0.3f,
  };

  colour_t grey_left(unsigned, unsigned) {
    return {0.4f, 0.45f, 0.5f};
  }

  colour_t striped_right(unsigned column, unsigned row) {
    return (column / 2 + row / 8) % 2 ? colour_t {1.0f, 0.9f, 0.8f} : colour_t {0.2f, 0.25f, 0.3f};
  }
}  // namespace

TEST(ShutterGhostSettings, ReadsNamesCommentsAndKeepsValuesInRange) {
  std::vector<std::string> problems;
  const auto settings = parse_shutter_ghost_settings(
    "# balance run\r\n"
    "pattern = 1   # balance\n"
    "view=left\n"
    "  top_peak = 0.15\n"
    "top_extent = 0.09\n"
    "bottom_peak = 0.9\n"
    "bottom_extent = 2\n"
    "black_raise = -1\n"
    "white_nits = 5000\n",
    &problems
  );
  EXPECT_TRUE(problems.empty());
  EXPECT_EQ(settings.pattern, 1);
  EXPECT_TRUE(settings.view_left);
  EXPECT_FLOAT_EQ(settings.top_peak, 0.15f);
  EXPECT_FLOAT_EQ(settings.top_extent, 0.09f);
  EXPECT_FLOAT_EQ(settings.bottom_peak, 0.45f);
  EXPECT_FLOAT_EQ(settings.bottom_extent, 0.5f);
  EXPECT_FLOAT_EQ(settings.black_raise, 0.0f);
  EXPECT_FLOAT_EQ(settings.white_nits, 1000.0f);
}

TEST(ShutterGhostSettings, ReportsLinesItCannotUseAndKeepsTheRest) {
  std::vector<std::string> problems;
  const auto settings = parse_shutter_ghost_settings("top_peak 0.2\ntop_peak = much\nview = both\nwidth = 3\ntop_extent = 0.1\n", &problems);
  EXPECT_EQ(problems.size(), 4u);
  EXPECT_FLOAT_EQ(settings.top_peak, 0.0f);
  EXPECT_FLOAT_EQ(settings.top_extent, 0.1f);
  EXPECT_FALSE(settings.view_left);
}

TEST(ShutterGhostSettings, AnEmptyFileLeavesThePlainCopy) {
  const auto settings = parse_shutter_ghost_settings("");
  EXPECT_EQ(settings, shutter_ghost_settings_t {});
  EXPECT_FALSE(settings.needs_pass());
  EXPECT_FALSE((shutter_ghost_settings_t {.top_peak = 0.2f}).needs_pass());  // A peak with no extent corrects nothing.
  EXPECT_TRUE((shutter_ghost_settings_t {.top_peak = 0.2f, .top_extent = 0.1f}).needs_pass());
  EXPECT_TRUE((shutter_ghost_settings_t {.pattern = 2}).needs_pass());
}

TEST(ShutterGhostSettings, ConstantsNameTheEyesHalvesAndWhite) {
  const shutter_ghost_settings_t settings {.white_nits = 240.0f, .view_left = true};
  const auto left = shutter_eye_constants(settings, true, 5120, 2160, true);
  EXPECT_EQ(left.own_left, 0);
  EXPECT_EQ(left.other_left, 5120);
  EXPECT_EQ(left.srgb_coded, 0);
  EXPECT_FLOAT_EQ(left.white_level, 3.0f);
  const auto right = shutter_eye_constants(settings, false, 5120, 2160, false);
  EXPECT_EQ(right.own_left, 5120);
  EXPECT_EQ(right.other_left, 0);
  EXPECT_EQ(right.srgb_coded, 1);
  EXPECT_FLOAT_EQ(right.white_level, 1.0f);
  EXPECT_EQ(right.drawing_left, 0);
  EXPECT_EQ(right.view_left, 1);
}

TEST_F(ShutterEyePassGpu, BothEyesReceiveThePictureMeantForThemThroughALeakingWindow) {
  const auto left = make_picture(grey_left);
  const auto right = make_picture(striped_right);
  const auto shown = show(left, right, true, leaking_window);
  for (unsigned row = 0; row < eye_height; ++row) {
    for (unsigned column = 0; column < eye_width; ++column) {
      const auto to_left = received(leaking_window, shown[0], shown[1], column, row);
      const auto to_right = received(leaking_window, shown[1], shown[0], column, row);
      for (unsigned channel = 0; channel < 3; ++channel) {
        ASSERT_NEAR(to_left[channel], left[row * eye_width + column][channel], 2e-4f) << "left eye, row " << row << ", column " << column;
        ASSERT_NEAR(to_right[channel], right[row * eye_width + column][channel], 2e-4f) << "right eye, row " << row << ", column " << column;
      }
    }
  }
  // The middle of the screen is outside both edges and is shown as it came.
  EXPECT_FLOAT_EQ(shown[0][eye_height / 2 * eye_width][0], 0.4f);
}

TEST_F(ShutterEyePassGpu, AnEightBitPictureIsCorrectedInLinearLight) {
  // Eight bits have nothing above white, so these pictures stay far enough below it for the
  // brighter one to make up what the darker one takes away.
  const auto left = make_picture([](unsigned, unsigned) {
    return colour_t {0.4f, 0.4f, 0.4f};
  });
  const auto right = make_picture([](unsigned column, unsigned row) {
    return (column / 2 + row / 8) % 2 ? colour_t {0.7f, 0.7f, 0.7f} : colour_t {0.2f, 0.2f, 0.2f};
  });
  const auto shown = show(left, right, false, leaking_window);
  for (unsigned row = 0; row < eye_height; ++row) {
    for (unsigned column = 0; column < eye_width; ++column) {
      const auto to_left = received(leaking_window, shown[0], shown[1], column, row);
      const auto to_right = received(leaking_window, shown[1], shown[0], column, row);
      for (unsigned channel = 0; channel < 3; ++channel) {
        ASSERT_NEAR(to_left[channel], left[row * eye_width + column][channel], 0.02f) << "left eye, row " << row << ", column " << column;
        ASSERT_NEAR(to_right[channel], right[row * eye_width + column][channel], 0.02f) << "right eye, row " << row << ", column " << column;
      }
    }
  }
}

TEST_F(ShutterEyePassGpu, BlackBesideWhiteStaysBlackUnlessBlackIsRaised) {
  const auto black = make_picture([](unsigned, unsigned) {
    return colour_t {0.0f, 0.0f, 0.0f};
  });
  const auto white = make_picture([](unsigned, unsigned) {
    return colour_t {2.5f, 2.5f, 2.5f};
  });
  auto settings = leaking_window;
  settings.white_nits = 200.0f;
  const auto clipped = show(black, white, true, settings);
  for (const auto &colour : clipped[0]) {
    ASSERT_EQ(colour[0], 0.0f);
  }
  // Without a raise the dark eye still receives the share of white it cannot take out.
  EXPECT_GT(received(settings, clipped[0], clipped[1], 0, 0)[0], 0.4f);

  settings.black_raise = 1.0f;
  const auto raised = show(black, white, true, settings);
  for (unsigned row = 0; row < eye_height; ++row) {
    const float share = share_at(settings, row);
    const float raise = 2.5f * share / (1.0f - 2.0f * share);
    // Both eyes receive their picture on top of the same raised black.
    ASSERT_NEAR(received(settings, raised[0], raised[1], 3, row)[0], raise, 1e-3f) << "row " << row;
    ASSERT_NEAR(received(settings, raised[1], raised[0], 3, row)[0], 2.5f + raise, 1e-3f) << "row " << row;
  }
}

TEST_F(ShutterEyePassGpu, TheBalancePatternIsAnEvenGreyForTheJudgedEyeWhenTheSettingsMatchTheLeak) {
  for (const bool view_left : {false, true}) {
    auto settings = leaking_window;
    settings.pattern = 1;
    settings.view_left = view_left;
    const auto shown = show({}, {}, true, settings, false);
    const auto &judged = shown[view_left ? 0 : 1];
    const auto &other = shown[view_left ? 1 : 0];
    for (unsigned row = 0; row < eye_height; ++row) {
      if (row == eye_height / 2 - 1 || row == eye_height / 2) {
        continue;  // The column marks sit at half height.
      }
      for (unsigned column = 0; column < eye_width; ++column) {
        ASSERT_NEAR(received(settings, judged, other, column, row)[1], 0.4f * 2.5f, 3e-4f) << "row " << row << ", column " << column;
      }
    }
    // Settings that take out too little leave the other eye's stripes in the grey.
    auto too_little = settings;
    too_little.top_peak = 0.1f;
    const auto under = show({}, {}, true, too_little, false);
    const auto &under_judged = under[view_left ? 0 : 1];
    const auto &under_other = under[view_left ? 1 : 0];
    const float dark_stripe = received(settings, under_judged, under_other, 0, 0)[1];
    const float light_stripe = received(settings, under_judged, under_other, 1, 0)[1];
    EXPECT_GT(light_stripe - dark_stripe, 0.1f);
  }
}
