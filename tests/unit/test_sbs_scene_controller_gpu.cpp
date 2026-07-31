/**
 * @file tests/unit/test_sbs_scene_controller_gpu.cpp
 * @brief WARP execution checks for the GPU-only Host SBS rules backend.
 */

#ifdef _WIN32

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <chrono>
  #include <cmath>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <fstream>
  #include <gtest/gtest.h>
  #include <iostream>
  #include <iterator>
  #include <limits>
  #include <memory>
  #include <nlohmann/json.hpp>
  #include <src/generated/sbs_adaptive_state_contract.h>
  #include <src/generated/sbs_scene_controller_contract.h>
  #include <src/sbs_frame_roi_transform.h>
  #include <src/sbs_perf.h>
  #include <src/sbs_scene_controller_gpu.h>
  #include <src_assets/windows/assets/shaders/directx/include/sbs_scene_rules_summary_layout.shared.h>
  #include <string>
  #include <string_view>
  #include <thread>
  #include <utility>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;

  struct structured_buffer_t {
    ComPtr<ID3D11Buffer> buffer;
    ComPtr<ID3D11ShaderResourceView> srv;
  };

  struct rw_structured_buffer_t {
    ComPtr<ID3D11Buffer> buffer;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };

  class scoped_temp_directory_t {
  public:
    explicit scoped_temp_directory_t(const std::string_view prefix) {
      const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
      for (std::uint32_t attempt = 0; attempt < 64u; ++attempt) {
        auto candidate =
          std::filesystem::temp_directory_path() /
          (std::string(prefix) + "-" + std::to_string(nonce) + "-" +
           std::to_string(attempt));
        std::error_code create_error;
        if (std::filesystem::create_directory(candidate, create_error)) {
          path_ = std::move(candidate);
          return;
        }
        if (create_error) {
          return;
        }
      }
    }

    ~scoped_temp_directory_t() {
      if (path_.empty()) {
        return;
      }
      std::error_code remove_error;
      std::filesystem::remove_all(path_, remove_error);
    }

    scoped_temp_directory_t(const scoped_temp_directory_t &) = delete;
    scoped_temp_directory_t &operator=(
      const scoped_temp_directory_t &
    ) = delete;

    [[nodiscard]] const std::filesystem::path &path() const {
      return path_;
    }

  private:
    std::filesystem::path path_;
  };

  structured_buffer_t make_structured_buffer(
    ID3D11Device *device,
    const void *values,
    UINT byte_width,
    UINT stride
  ) {
    structured_buffer_t result;
    D3D11_BUFFER_DESC descriptor {};
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.ByteWidth = byte_width;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    descriptor.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    descriptor.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA initial {values, 0, 0};
    EXPECT_TRUE(SUCCEEDED(device->CreateBuffer(
      &descriptor,
      values ? &initial : nullptr,
      &result.buffer
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      result.buffer.Get(),
      nullptr,
      &result.srv
    )));
    return result;
  }

  rw_structured_buffer_t make_rw_structured_buffer(
    ID3D11Device *device,
    const void *values,
    UINT byte_width,
    UINT stride
  ) {
    rw_structured_buffer_t result;
    D3D11_BUFFER_DESC descriptor {};
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.ByteWidth = byte_width;
    descriptor.BindFlags =
      D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    descriptor.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    descriptor.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA initial {values, 0, 0};
    EXPECT_TRUE(SUCCEEDED(device->CreateBuffer(
      &descriptor,
      values ? &initial : nullptr,
      &result.buffer
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      result.buffer.Get(),
      nullptr,
      &result.srv
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateUnorderedAccessView(
      result.buffer.Get(),
      nullptr,
      &result.uav
    )));
    return result;
  }

  using frame_roi_transform_words_t = std::array<
    std::array<std::uint32_t, 4>,
    models::frame_roi_transform_vector_count
  >;

  frame_roi_transform_words_t make_active_frame_roi_transform(
    const std::uint64_t source_frame_id,
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const std::uint32_t model_width,
    const std::uint32_t model_height,
    const std::array<float, 4> &focus,
    const std::array<float, 4> &crop
  ) {
    constexpr std::uint32_t valid_flag = 1u << 0u;
    constexpr std::uint32_t active_roi_flag = 1u << 2u;
    frame_roi_transform_words_t words {};
    words[0] = {
      models::frame_roi_transform_contract_version,
      valid_flag | active_roi_flag,
      static_cast<std::uint32_t>(source_frame_id),
      static_cast<std::uint32_t>(source_frame_id >> 32u),
    };
    words[1] = {
      1u,
      1u,
      source_width,
      source_height,
    };
    words[2] = {
      model_width,
      model_height,
      model_width * model_height,
      1u,
    };
    for (std::size_t component = 0; component < 4u; ++component) {
      words[3][component] = std::bit_cast<std::uint32_t>(
        focus[component]
      );
      words[4][component] = std::bit_cast<std::uint32_t>(
        crop[component]
      );
    }
    words[5] = {
      0u,
      0u,
      model_width,
      model_height,
    };
    words[7] = {
      1u,
      0u,
      0u,
      0u,
    };
    return words;
  }

  ComPtr<ID3D11Buffer> make_constant_buffer(
    ID3D11Device *device,
    const void *values,
    const UINT byte_width
  ) {
    ComPtr<ID3D11Buffer> result;
    D3D11_BUFFER_DESC descriptor {};
    descriptor.Usage = D3D11_USAGE_IMMUTABLE;
    descriptor.ByteWidth = byte_width;
    descriptor.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial {values, 0, 0};
    EXPECT_TRUE(SUCCEEDED(device->CreateBuffer(
      &descriptor,
      &initial,
      &result
    )));
    return result;
  }

  ComPtr<ID3D11ComputeShader> compile_compute_shader(
    ID3D11Device *device,
    const std::string_view filename
  ) {
    const auto path =
      std::filesystem::path(SUNSHINE_SHADERS_DIR) / filename;
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    const auto compile_status = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      "main",
      "cs_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0,
      &bytecode,
      &diagnostics
    );
    EXPECT_TRUE(SUCCEEDED(compile_status))
      << filename << ": "
      << (diagnostics ?
            static_cast<const char *>(diagnostics->GetBufferPointer()) :
            "no compiler diagnostics");
    ComPtr<ID3D11ComputeShader> shader;
    if (SUCCEEDED(compile_status)) {
      EXPECT_TRUE(SUCCEEDED(device->CreateComputeShader(
        bytecode->GetBufferPointer(),
        bytecode->GetBufferSize(),
        nullptr,
        &shader
      )));
    }
    return shader;
  }

  structured_buffer_t make_adaptive_state_buffer(
    ID3D11Device *device,
    const std::uint32_t external_cut_count = 0u,
    const float valid_depth_fraction = 0.0f,
    const std::uint32_t detector_cut_count = 0u,
    const bool hard_cut_pulse = false,
    const std::uint32_t analysis_flags = 0u
  ) {
    sbs_adaptive_state::words_t words {};
    for (std::size_t index = 0; index < words.size(); ++index) {
      words[index] =
        std::bit_cast<std::uint32_t>(
          sbs_adaptive_state::initial_values[index]
        );
    }
    words[sbs_adaptive_state::index(
      sbs_adaptive_state::word_e::external_cut_count
    )] = external_cut_count;
    words[sbs_adaptive_state::index(
      sbs_adaptive_state::word_e::valid_depth_fraction
    )] = std::bit_cast<std::uint32_t>(valid_depth_fraction);
    words[sbs_adaptive_state::index(
      sbs_adaptive_state::word_e::hard_cut_count
    )] = detector_cut_count;
    words[sbs_adaptive_state::index(
      sbs_adaptive_state::word_e::hard_cut_pulse
    )] = std::bit_cast<std::uint32_t>(
      hard_cut_pulse ? 1.0f : 0.0f
    );
    words[sbs_adaptive_state::index(
      sbs_adaptive_state::word_e::analysis_flags
    )] = std::bit_cast<std::uint32_t>(
      static_cast<float>(analysis_flags)
    );
    return make_structured_buffer(
      device,
      words.data(),
      static_cast<UINT>(sizeof(words)),
      sizeof(float) * 4u
    );
  }

  ComPtr<ID3D11ShaderResourceView> make_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height,
    const std::uint32_t pattern_phase = 0
  ) {
    std::vector<std::uint32_t> source(width * height);
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        const std::uint8_t checker =
          ((x / 12u + y / 12u + pattern_phase) & 1u) != 0u ?
            224u :
            24u;
        source[y * width + x] =
          0xff000000u |
          static_cast<std::uint32_t>(checker) |
          (static_cast<std::uint32_t>(checker) << 8u) |
          (static_cast<std::uint32_t>(checker) << 16u);
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(width * sizeof(std::uint32_t)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView> make_content_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height
  ) {
    std::vector<std::uint32_t> source(width * height);
    const UINT left = width / 5u;
    const UINT right = 4u * width / 5u;
    const UINT top = height / 5u;
    const UINT bottom = 4u * height / 5u;
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        const bool in_content =
          x >= left && x < right && y >= top && y < bottom;
        if (!in_content) {
          source[y * width + x] = 0xff181818u;
          continue;
        }
        source[y * width + x] =
          ((x / 4u + y / 4u) & 1u) != 0u ?
            0xff3cdcf0u :
            0xffd04028u;
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(width * sizeof(std::uint32_t)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView> make_player_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height,
    const std::uint32_t pattern_phase
  ) {
    std::vector<std::uint32_t> source(width * height, 0xff202020u);
    const UINT left = width / 5u;
    const UINT right = 7u * width / 10u;
    const UINT top = height / 5u;
    const UINT bottom = 4u * height / 5u;
    for (UINT y = top; y < bottom; ++y) {
      for (UINT x = left; x < right; ++x) {
        std::uint32_t hash =
          (x + 1u) * 0x9e3779b9u ^
          (y + 1u) * 0x85ebca6bu ^
          (pattern_phase + 1u) * 0xc2b2ae35u;
        hash ^= hash >> 16u;
        hash *= 0x7feb352du;
        hash ^= hash >> 15u;
        const std::uint8_t value = static_cast<std::uint8_t>(
          24u + hash % 209u
        );
        source[y * width + x] =
          0xff000000u |
          static_cast<std::uint32_t>(value) |
          (static_cast<std::uint32_t>(value) << 8u) |
          (static_cast<std::uint32_t>(value) << 16u);
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(width * sizeof(std::uint32_t)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  struct adversarial_source_region_t {
    float x0;
    float y0;
    float x1;
    float y1;
    std::uint32_t pattern_phase;
    bool colorful;
  };

  template<std::size_t RegionCount>
  ComPtr<ID3D11ShaderResourceView> make_adversarial_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height,
    const std::array<adversarial_source_region_t, RegionCount> &regions
  ) {
    std::vector<std::uint32_t> source(width * height, 0xff202020u);
    for (const auto &region : regions) {
      const auto left = std::min(
        width,
        static_cast<UINT>(
          std::floor(std::clamp(region.x0, 0.0f, 1.0f) * width)
        )
      );
      const auto right = std::min(
        width,
        static_cast<UINT>(
          std::ceil(std::clamp(region.x1, 0.0f, 1.0f) * width)
        )
      );
      const auto top = std::min(
        height,
        static_cast<UINT>(
          std::floor(std::clamp(region.y0, 0.0f, 1.0f) * height)
        )
      );
      const auto bottom = std::min(
        height,
        static_cast<UINT>(
          std::ceil(std::clamp(region.y1, 0.0f, 1.0f) * height)
        )
      );
      for (UINT y = top; y < bottom; ++y) {
        for (UINT x = left; x < right; ++x) {
          if (region.colorful) {
            source[y * width + x] =
              ((x / 4u + y / 4u + region.pattern_phase) & 1u) != 0u ?
                0xff3cdcf0u :
                0xffd04028u;
            continue;
          }
          std::uint32_t hash =
            (x + 1u) * 0x9e3779b9u ^
            (y + 1u) * 0x85ebca6bu ^
            (region.pattern_phase + 1u) * 0xc2b2ae35u;
          hash ^= hash >> 16u;
          hash *= 0x7feb352du;
          hash ^= hash >> 15u;
          const std::uint8_t value = static_cast<std::uint8_t>(
            24u + hash % 209u
          );
          source[y * width + x] =
            0xff000000u |
            static_cast<std::uint32_t>(value) |
            (static_cast<std::uint32_t>(value) << 8u) |
            (static_cast<std::uint32_t>(value) << 16u);
        }
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(width * sizeof(std::uint32_t)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView>
  make_content_with_embedded_player_source_view(
    ID3D11Device *device,
    const std::uint32_t player_phase
  ) {
    return make_adversarial_source_view(
      device,
      320,
      180,
      std::array {
        adversarial_source_region_t {
          0.18f,
          0.12f,
          0.88f,
          0.90f,
          0u,
          true,
        },
        adversarial_source_region_t {
          0.08f,
          0.14f,
          0.70f,
          0.64f,
          player_phase,
          false,
        },
      }
    );
  }

  ComPtr<ID3D11ShaderResourceView> make_split_threshold_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height,
    const UINT gutter_width,
    const std::uint8_t contrast,
    const std::int8_t right_bias
  ) {
    std::vector<std::uint32_t> source(width * height, 0xff181818u);
    const UINT gutter_begin =
      width / 2u - std::min(gutter_width, width) / 2u;
    const UINT gutter_end =
      std::min(width, gutter_begin + gutter_width);
    const UINT top = height / 8u;
    const UINT bottom = height - top;
    for (UINT y = top; y < bottom; ++y) {
      for (UINT x = width / 16u; x < width - width / 16u; ++x) {
        if (x >= gutter_begin && x < gutter_end) {
          continue;
        }
        const int bias = x >= gutter_end ? right_bias : 0;
        const int direction = ((x / 5u + y / 5u) & 1u) != 0u ? 1 : -1;
        const auto value = static_cast<std::uint8_t>(std::clamp(
          128 + bias + direction * static_cast<int>(contrast),
          0,
          255
        ));
        source[y * width + x] =
          0xff000000u |
          static_cast<std::uint32_t>(value) |
          (static_cast<std::uint32_t>(value) << 8u) |
          (static_cast<std::uint32_t>(value) << 16u);
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(width * sizeof(std::uint32_t)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView> make_scroll_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height,
    const UINT vertical_shift,
    const int horizontal_shift = 0,
    const int exposure_offset = 0,
    const bool opposing_horizontal_panels = false
  ) {
    std::vector<std::uint32_t> source(width * height);
    const auto wrap =
      [](const int coordinate, const UINT extent) {
        const int modulus = static_cast<int>(std::max(extent, 1u));
        const int remainder = coordinate % modulus;
        return static_cast<UINT>(
          remainder < 0 ? remainder + modulus : remainder
        );
      };
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        const UINT shifted_y = (y + vertical_shift) % height;
        const int signed_horizontal_shift =
          opposing_horizontal_panels && y >= height / 2u ?
            -horizontal_shift :
            horizontal_shift;
        const UINT shifted_x = wrap(
          static_cast<int>(x) + signed_horizontal_shift,
          width
        );
        std::uint32_t hash =
          (shifted_x + 1u) * 0x9e3779b9u ^
          (shifted_y + 1u) * 0x85ebca6bu;
        hash ^= hash >> 16u;
        hash *= 0x7feb352du;
        hash ^= hash >> 15u;
        const std::uint8_t value = static_cast<std::uint8_t>(
          std::clamp(
            32 + static_cast<int>(hash % 192u) + exposure_offset,
            0,
            255
          )
        );
        source[y * width + x] =
          0xff000000u |
          static_cast<std::uint32_t>(value) |
          (static_cast<std::uint32_t>(value) << 8u) |
          (static_cast<std::uint32_t>(value) << 16u);
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(width * sizeof(std::uint32_t)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView>
  make_repetitive_page_scroll_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height,
    const UINT page_vertical_shift,
    const bool draw_widget = false,
    const UINT widget_top = 0u
  ) {
    std::vector<std::uint32_t> source(width * height);
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        const UINT period = ((x / 8u) & 1u) != 0u ? 2u : 3u;
        const UINT shifted_y = (y + page_vertical_shift) % height;
        const UINT phase = shifted_y % period;
        const std::uint8_t value =
          period == 2u ?
            (phase == 0u ? 36u : 220u) :
          phase == 0u ? 28u :
          phase == 1u ? 126u :
                        232u;
        source[y * width + x] =
          0xff000000u |
          static_cast<std::uint32_t>(value) |
          (static_cast<std::uint32_t>(value) << 8u) |
          (static_cast<std::uint32_t>(value) << 16u);
      }
    }

    if (draw_widget) {
      constexpr UINT widget_size = 12u;
      const UINT left =
        width > widget_size ? (width - widget_size) / 2u : 0u;
      const UINT right = std::min(width, left + widget_size);
      const UINT top = std::min(height, widget_top);
      const UINT bottom = std::min(height, top + widget_size);
      for (UINT y = top; y < bottom; ++y) {
        for (UINT x = left; x < right; ++x) {
          const bool checker =
            (((x - left) / 2u + (y - top) / 2u) & 1u) != 0u;
          source[y * width + x] =
            checker ? 0xff28d8f0u : 0xffe05028u;
        }
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(width * sizeof(std::uint32_t)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView>
  make_orthogonal_scroll_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height,
    const int horizontal_shift,
    const int vertical_shift
  ) {
    std::vector<std::uint32_t> source(width * height);
    const auto wrap =
      [](const int coordinate, const UINT extent) {
        const int modulus = static_cast<int>(std::max(extent, 1u));
        const int remainder = coordinate % modulus;
        return static_cast<UINT>(
          remainder < 0 ? remainder + modulus : remainder
        );
      };
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        const bool horizontal_region = y < height / 2u;
        const UINT coordinate =
          horizontal_region ?
            wrap(static_cast<int>(x) + horizontal_shift, width) :
            wrap(static_cast<int>(y) + vertical_shift, height);
        std::uint32_t hash =
          (coordinate + 1u) *
          (horizontal_region ? 0x9e3779b9u : 0x85ebca6bu);
        hash ^= hash >> 16u;
        hash *= 0x7feb352du;
        hash ^= hash >> 15u;
        const std::uint8_t value = static_cast<std::uint8_t>(
          32u + hash % 192u
        );
        source[y * width + x] =
          0xff000000u |
          static_cast<std::uint32_t>(value) |
          (static_cast<std::uint32_t>(value) << 8u) |
          (static_cast<std::uint32_t>(value) << 16u);
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(width * sizeof(std::uint32_t)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView> make_scrolling_content_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height,
    const UINT vertical_shift
  ) {
    std::vector<std::uint32_t> source(width * height, 0xff202020u);
    const UINT left = 3u * width / 20u;
    const UINT right = 17u * width / 20u;
    const UINT top = 3u * height / 20u;
    const UINT bottom = 17u * height / 20u;
    const UINT content_height = bottom - top;
    for (UINT y = top; y < bottom; ++y) {
      for (UINT x = left; x < right; ++x) {
        const UINT shifted_y =
          (y - top + vertical_shift) % content_height;
        std::uint32_t hash =
          (x + 1u) * 0x9e3779b9u ^
          (shifted_y + 1u) * 0x85ebca6bu;
        hash ^= hash >> 16u;
        hash *= 0x7feb352du;
        hash ^= hash >> 15u;
        const std::uint8_t red = static_cast<std::uint8_t>(
          32u + (hash & 0xbfu)
        );
        const std::uint8_t green = static_cast<std::uint8_t>(
          32u + ((hash >> 8u) & 0xbfu)
        );
        const std::uint8_t blue = static_cast<std::uint8_t>(
          32u + ((hash >> 16u) & 0xbfu)
        );
        source[y * width + x] =
          0xff000000u |
          static_cast<std::uint32_t>(blue) |
          (static_cast<std::uint32_t>(green) << 8u) |
          (static_cast<std::uint32_t>(red) << 16u);
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(width * sizeof(std::uint32_t)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView>
  make_large_player_with_scrolling_exterior_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height,
    const std::uint32_t player_phase,
    const UINT exterior_vertical_shift,
    const UINT player_vertical_shift = 0u,
    const UINT player_horizontal_inset = 0u,
    const int exterior_horizontal_shift = 0,
    const int player_horizontal_shift = 0,
    const UINT player_vertical_inset = 0u
  ) {
    std::vector<std::uint32_t> source(width * height);
    const UINT player_left =
      player_horizontal_inset > 0u ?
        std::min(player_horizontal_inset, width / 2u) :
        width / 10u;
    const UINT player_right = width - player_left;
    const UINT player_top = std::min(
      player_vertical_inset,
      height / 2u
    );
    const UINT player_bottom = height - player_top;
    const auto wrap =
      [](const int coordinate, const UINT extent) {
        const int modulus = static_cast<int>(std::max(extent, 1u));
        const int remainder = coordinate % modulus;
        return static_cast<UINT>(
          remainder < 0 ? remainder + modulus : remainder
        );
      };
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        const bool in_player =
          x >= player_left && x < player_right &&
          y >= player_top && y < player_bottom;
        const UINT shifted_y =
          (y + exterior_vertical_shift) % height;
        const UINT shifted_player_y =
          (y + player_vertical_shift) % height;
        const UINT shifted_x = wrap(
          static_cast<int>(x) +
            (in_player ?
               player_horizontal_shift :
               exterior_horizontal_shift),
          width
        );
        std::uint32_t hash =
          (shifted_x + 1u) * 0x9e3779b9u ^
          ((in_player ? shifted_player_y : shifted_y) + 1u) *
            0x85ebca6bu ^
          (in_player ? player_phase + 1u : 1u) * 0xc2b2ae35u;
        hash ^= hash >> 16u;
        hash *= 0x7feb352du;
        hash ^= hash >> 15u;
        const std::uint8_t value = static_cast<std::uint8_t>(
          in_player ? 24u + hash % 209u : 32u + hash % 192u
        );
        source[y * width + x] =
          0xff000000u |
          static_cast<std::uint32_t>(value) |
          (static_cast<std::uint32_t>(value) << 8u) |
          (static_cast<std::uint32_t>(value) << 16u);
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(width * sizeof(std::uint32_t)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView> make_nan_source_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height
  ) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 4> invalid_pixel {nan, nan, nan, 1.0f};
    std::vector<std::array<float, 4>> source(
      width * height,
      invalid_pixel
    );

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      source.data(),
      static_cast<UINT>(
        width * sizeof(std::array<float, 4>)
      ),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView> make_depth_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height
  ) {
    std::vector<float> depth(width * height);
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        depth[y * width + x] =
          0.2f + 0.6f * static_cast<float>(x) /
                   static_cast<float>(width - 1u);
      }
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_R32_FLOAT;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      depth.data(),
      static_cast<UINT>(width * sizeof(float)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  ComPtr<ID3D11ShaderResourceView> make_depth_view(
    ID3D11Device *device,
    const UINT width,
    const UINT height,
    const std::vector<float> &depth
  ) {
    EXPECT_EQ(depth.size(), static_cast<std::size_t>(width) * height);
    if (depth.size() != static_cast<std::size_t>(width) * height) {
      return {};
    }

    D3D11_TEXTURE2D_DESC descriptor {};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.MipLevels = 1;
    descriptor.ArraySize = 1;
    descriptor.Format = DXGI_FORMAT_R32_FLOAT;
    descriptor.SampleDesc.Count = 1;
    descriptor.Usage = D3D11_USAGE_DEFAULT;
    descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial {
      depth.data(),
      static_cast<UINT>(width * sizeof(float)),
      0,
    };
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> view;
    EXPECT_TRUE(SUCCEEDED(device->CreateTexture2D(
      &descriptor,
      &initial,
      &texture
    )));
    EXPECT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      texture.Get(),
      nullptr,
      &view
    )));
    return view;
  }

  template<class T>
  std::vector<T> read_buffer(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    ID3D11ShaderResourceView *view,
    std::size_t count
  ) {
    ComPtr<ID3D11Resource> resource;
    view->GetResource(&resource);
    ComPtr<ID3D11Buffer> source;
    EXPECT_TRUE(SUCCEEDED(resource.As(&source)));
    D3D11_BUFFER_DESC descriptor {};
    source->GetDesc(&descriptor);
    descriptor.Usage = D3D11_USAGE_STAGING;
    descriptor.BindFlags = 0;
    descriptor.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    descriptor.MiscFlags = 0;
    ComPtr<ID3D11Buffer> staging;
    EXPECT_TRUE(SUCCEEDED(device->CreateBuffer(
      &descriptor,
      nullptr,
      &staging
    )));
    context->CopyResource(staging.Get(), source.Get());
    D3D11_MAPPED_SUBRESOURCE mapped {};
    EXPECT_TRUE(SUCCEEDED(context->Map(
      staging.Get(),
      0,
      D3D11_MAP_READ,
      0,
      &mapped
    )));
    std::vector<T> values(count);
    if (mapped.pData) {
      std::memcpy(values.data(), mapped.pData, count * sizeof(T));
      context->Unmap(staging.Get(), 0);
    }
    return values;
  }

  float word_as_float(
    const std::vector<std::uint32_t> &words,
    const sbs_scene_controller::rule_state_word_e word
  ) {
    return std::bit_cast<float>(
      words[sbs_scene_controller::index(word)]
    );
  }

  void expect_all_finite(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    ID3D11ShaderResourceView *view,
    const std::size_t count,
    const std::string_view label
  ) {
    SCOPED_TRACE(label);
    ASSERT_NE(view, nullptr);
    const auto values = read_buffer<float>(
      device,
      context,
      view,
      count
    );
    const auto invalid_count = std::count_if(
      values.begin(),
      values.end(),
      [](const float value) {
        return !std::isfinite(value);
      }
    );
    EXPECT_EQ(invalid_count, 0);
  }

  void expect_canonical_viewport(
    const std::vector<float> &analysis,
    const std::size_t expected_x,
    const std::size_t expected_y,
    const std::size_t expected_width,
    const std::size_t expected_height
  ) {
    constexpr std::size_t canvas =
      sbs_scene_controller::analysis_canvas_size;
    constexpr std::size_t plane = canvas * canvas;
    const auto analysis_at =
      [&](const std::size_t channel, const std::size_t x, const std::size_t y) {
        return analysis[channel * plane + y * canvas + x];
      };
    const auto valid_channel = static_cast<std::size_t>(
      sbs_scene_controller::analysis_grid_channel_e::viewport_valid
    );
    const auto viewport_x_channel = static_cast<std::size_t>(
      sbs_scene_controller::analysis_grid_channel_e::viewport_x
    );
    const auto viewport_y_channel = static_cast<std::size_t>(
      sbs_scene_controller::analysis_grid_channel_e::viewport_y
    );

    std::size_t valid_count = 0;
    std::size_t validity_errors = 0;
    std::size_t nonzero_padding_values = 0;
    std::size_t invalid_viewport_coordinates = 0;
    for (std::size_t y = 0; y < canvas; ++y) {
      for (std::size_t x = 0; x < canvas; ++x) {
        const bool expected_valid =
          x >= expected_x && x < expected_x + expected_width &&
          y >= expected_y && y < expected_y + expected_height;
        const float actual_valid = analysis_at(valid_channel, x, y);
        valid_count += actual_valid == 1.0f ? 1u : 0u;
        validity_errors +=
          actual_valid != (expected_valid ? 1.0f : 0.0f) ? 1u : 0u;
        if (!expected_valid) {
          for (std::size_t channel = 0;
               channel <
               sbs_scene_controller::analysis_grid_channel_count;
               ++channel) {
            nonzero_padding_values +=
              analysis_at(channel, x, y) != 0.0f ? 1u : 0u;
          }
        } else {
          const float viewport_x =
            analysis_at(viewport_x_channel, x, y);
          const float viewport_y =
            analysis_at(viewport_y_channel, x, y);
          invalid_viewport_coordinates +=
            !std::isfinite(viewport_x) ||
                !std::isfinite(viewport_y) ||
                viewport_x < 0.0f || viewport_x > 1.0f ||
                viewport_y < 0.0f || viewport_y > 1.0f ?
              1u :
              0u;
        }
      }
    }

    EXPECT_EQ(valid_count, expected_width * expected_height);
    EXPECT_EQ(validity_errors, 0u);
    EXPECT_EQ(nonzero_padding_values, 0u);
    EXPECT_EQ(invalid_viewport_coordinates, 0u);
  }

  void expect_roi_coverage(
    const std::vector<float> &analysis,
    const bool roi_locked,
    const std::array<float, 4> &roi
  ) {
    constexpr std::size_t canvas =
      sbs_scene_controller::analysis_canvas_size;
    constexpr std::size_t plane = canvas * canvas;
    ASSERT_EQ(
      analysis.size(),
      sbs_scene_controller::analysis_grid_channel_count * plane
    );
    const auto channel_offset = [&](const auto channel) {
      return static_cast<std::size_t>(channel) * plane;
    };
    const auto validity_offset = channel_offset(
      sbs_scene_controller::analysis_grid_channel_e::viewport_valid
    );
    const auto coverage_offset = channel_offset(
      sbs_scene_controller::analysis_grid_channel_e::current_roi_coverage
    );
    const auto viewport_x_offset = channel_offset(
      sbs_scene_controller::analysis_grid_channel_e::viewport_x
    );
    const auto viewport_y_offset = channel_offset(
      sbs_scene_controller::analysis_grid_channel_e::viewport_y
    );

    std::size_t valid_cells = 0;
    std::size_t covered_cells = 0;
    for (std::size_t cell = 0; cell < plane; ++cell) {
      const bool viewport_valid = analysis[validity_offset + cell] > 0.5f;
      const float viewport_x = analysis[viewport_x_offset + cell];
      const float viewport_y = analysis[viewport_y_offset + cell];
      const bool expected_covered =
        viewport_valid &&
        (
          !roi_locked ||
          (viewport_x >= roi[0] && viewport_y >= roi[1] &&
           viewport_x < roi[2] && viewport_y < roi[3])
        );
      SCOPED_TRACE(cell);
      EXPECT_FLOAT_EQ(
        analysis[coverage_offset + cell],
        expected_covered ? 1.0f : 0.0f
      );
      valid_cells += viewport_valid ? 1u : 0u;
      covered_cells += expected_covered ? 1u : 0u;
    }
    EXPECT_GT(valid_cells, 0u);
    if (roi_locked) {
      EXPECT_GT(covered_cells, 0u);
      EXPECT_LT(covered_cells, valid_cells);
    } else {
      EXPECT_EQ(covered_cells, valid_cells);
    }
  }

  void expect_snapshot_abi_invariants(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    const models::scene_controller_gpu_snapshot &snapshot
  ) {
    constexpr std::size_t appearance_pixels =
      sbs_scene_controller::appearance_canvas_size *
      sbs_scene_controller::appearance_canvas_size;
    constexpr std::size_t analysis_pixels =
      sbs_scene_controller::analysis_canvas_size *
      sbs_scene_controller::analysis_canvas_size;
    constexpr std::size_t recurrent_pixels =
      sbs_scene_controller::recurrent_canvas_size *
      sbs_scene_controller::recurrent_canvas_size;

    ASSERT_TRUE(snapshot.snapshot_available);
    EXPECT_TRUE(snapshot.shadow);
    ASSERT_TRUE(snapshot.scene_rgb);
    ASSERT_TRUE(snapshot.analysis_grid);
    ASSERT_TRUE(snapshot.dense_output);
    ASSERT_TRUE(snapshot.global_output);
    ASSERT_TRUE(snapshot.layout_history);
    ASSERT_TRUE(snapshot.depth_history);
    ASSERT_TRUE(snapshot.hidden_output);
    ASSERT_TRUE(snapshot.meta);
    ASSERT_TRUE(snapshot.rule_state);

    expect_all_finite(
      device,
      context,
      snapshot.scene_rgb.Get(),
      3u * appearance_pixels,
      "scene_rgb"
    );
    expect_all_finite(
      device,
      context,
      snapshot.analysis_grid.Get(),
      sbs_scene_controller::analysis_grid_channel_count * analysis_pixels,
      "analysis_grid"
    );
    expect_all_finite(
      device,
      context,
      snapshot.dense_output.Get(),
      sbs_scene_controller::dense_out_channel_count * analysis_pixels,
      "dense_output"
    );
    expect_all_finite(
      device,
      context,
      snapshot.global_output.Get(),
      sbs_scene_controller::global_out_word_count,
      "global_output"
    );
    expect_all_finite(
      device,
      context,
      snapshot.layout_history.Get(),
      sbs_scene_controller::layout_history_channel_count * analysis_pixels,
      "layout_history"
    );
    expect_all_finite(
      device,
      context,
      snapshot.depth_history.Get(),
      sbs_scene_controller::depth_history_channel_count * analysis_pixels,
      "depth_history"
    );
    expect_all_finite(
      device,
      context,
      snapshot.hidden_output.Get(),
      sbs_scene_controller::hidden_channel_count * recurrent_pixels,
      "hidden_output"
    );
    expect_all_finite(
      device,
      context,
      snapshot.meta.Get(),
      sbs_scene_controller::meta_word_count,
      "meta"
    );

    const auto global = read_buffer<float>(
      device,
      context,
      snapshot.global_output.Get(),
      sbs_scene_controller::global_out_word_count
    );
    EXPECT_FLOAT_EQ(
      global[static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::backend_output_valid
      )],
      1.0f
    );
    for (std::size_t word = static_cast<std::size_t>(
           sbs_scene_controller::global_out_word_e::reserved_35
         );
         word < global.size();
         ++word) {
      EXPECT_FLOAT_EQ(global[word], 0.0f) << "global word " << word;
    }

    const auto meta = read_buffer<float>(
      device,
      context,
      snapshot.meta.Get(),
      sbs_scene_controller::meta_word_count
    );
    for (std::size_t word = static_cast<std::size_t>(
           sbs_scene_controller::meta_word_e::reserved_28
         );
         word < meta.size();
         ++word) {
      EXPECT_FLOAT_EQ(meta[word], 0.0f) << "meta word " << word;
    }

    const auto hidden = read_buffer<float>(
      device,
      context,
      snapshot.hidden_output.Get(),
      sbs_scene_controller::hidden_channel_count * recurrent_pixels
    );
    EXPECT_EQ(
      std::count_if(
        hidden.begin(),
        hidden.end(),
        [](const float value) {
          return value != 0.0f;
        }
      ),
      0
    );

    const auto rule_words = read_buffer<std::uint32_t>(
      device,
      context,
      snapshot.rule_state.Get(),
      sbs_scene_controller::rule_state_word_count
    );
    D3D11_SHADER_RESOURCE_VIEW_DESC state_view_descriptor {};
    snapshot.rule_state->GetDesc(&state_view_descriptor);
    EXPECT_EQ(
      state_view_descriptor.Buffer.NumElements,
      sbs_scene_controller::rule_state_vector_count
    );
    ComPtr<ID3D11Resource> state_resource;
    snapshot.rule_state->GetResource(&state_resource);
    ComPtr<ID3D11Buffer> state_buffer;
    ASSERT_TRUE(SUCCEEDED(state_resource.As(&state_buffer)));
    D3D11_BUFFER_DESC state_buffer_descriptor {};
    state_buffer->GetDesc(&state_buffer_descriptor);
    EXPECT_EQ(
      state_buffer_descriptor.StructureByteStride,
      sizeof(float) * 4u
    );
    EXPECT_EQ(
      state_buffer_descriptor.ByteWidth,
      sbs_scene_controller::rule_state_word_count * sizeof(float)
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        rule_words,
        sbs_scene_controller::rule_state_word_e::schema_version
      ),
      static_cast<float>(sbs_scene_controller::schema_version)
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        rule_words,
        sbs_scene_controller::rule_state_word_e::output_valid
      ),
      1.0f
    );
    EXPECT_EQ(
      rule_words[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::backend_generation
      )],
      snapshot.backend_generation
    );
    for (const auto &field : sbs_scene_controller::rule_state_fields) {
      const auto index = sbs_scene_controller::index(field.word);
      if (
        field.gpu_encoding !=
        sbs_scene_controller::gpu_encoding_e::uint_bits
      ) {
        EXPECT_TRUE(std::isfinite(std::bit_cast<float>(rule_words[index])))
          << field.name;
      }
      if (field.required_zero) {
        EXPECT_EQ(rule_words[index], 0u) << field.name;
      }
    }
  }

  class SbsSceneControllerGpu: public testing::Test {
  protected:
    static constexpr UINT depth_width = 64;
    static constexpr UINT depth_height = 36;

    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_0};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr,
        0,
        levels,
        static_cast<UINT>(std::size(levels)),
        D3D11_SDK_VERSION,
        &device,
        &feature_level,
        &context
      )));

      depth_view = make_depth_view(
        device.Get(),
        depth_width,
        depth_height
      );
      ASSERT_TRUE(depth_view);
      std::vector<float> raw_depth(
        depth_width * depth_height,
        0.5f
      );
      raw_depth_buffer = make_structured_buffer(
        device.Get(),
        raw_depth.data(),
        static_cast<UINT>(raw_depth.size() * sizeof(float)),
        sizeof(float)
      );
      ASSERT_TRUE(raw_depth_buffer.srv);

      const std::array<float, 4> frame_state {
        0.0f,
        1.0f,
        1.0f,
        1.0f,
      };
      frame_state_buffer = make_structured_buffer(
        device.Get(),
        frame_state.data(),
        sizeof(frame_state),
        sizeof(frame_state)
      );
      adaptive_buffer = make_adaptive_state_buffer(
        device.Get(),
        0u,
        0.0f
      );
      ASSERT_TRUE(frame_state_buffer.srv);
      ASSERT_TRUE(adaptive_buffer.srv);
      const std::array<
        std::array<std::uint32_t, 4>,
        models::frame_roi_transform_vector_count
      > zero_transform {};
      frame_roi_transform_buffer = make_structured_buffer(
        device.Get(),
        zero_transform.data(),
        sizeof(zero_transform),
        sizeof(zero_transform[0])
      );
      ASSERT_TRUE(frame_roi_transform_buffer.srv);

      config::video_t::sbs_t sbs;
      sbs.scene_controller =
        config::sbs_scene_controller_e::shadow_rules;
      const std::filesystem::path assets_dir =
        std::filesystem::path(SUNSHINE_SHADERS_DIR)
          .parent_path()
          .parent_path();
      controller = std::make_unique<models::sbs_scene_controller_gpu>(
        device,
        context,
        assets_dir,
        sbs.scene_controller,
        sbs
      );
      ASSERT_TRUE(controller->enabled());
      ASSERT_TRUE(controller->valid());
    }

    bool run_frame(
      const UINT source_width,
      const UINT source_height,
      const std::uint64_t frame_id,
      const std::uint32_t pattern_phase = 0
    ) {
      const auto source_view = make_source_view(
        device.Get(),
        source_width,
        source_height,
        pattern_phase
      );
      if (!source_view) {
        return false;
      }
      return run_source(source_view.Get(), frame_id);
    }

    bool run_source(
      ID3D11ShaderResourceView *source_view,
      const std::uint64_t frame_id
    ) {
      if (!controller->prepare_scene(
            source_view,
            models::input_color_space::srgb,
            frame_id
          )) {
        return false;
      }
      controller->mark_enqueued(frame_id);
      return resolve_pending(frame_id);
    }

    bool resolve_pending(const std::uint64_t frame_id) {
      return controller->resolve_completed(
        frame_id,
        raw_depth_buffer.srv.Get(),
        depth_view.Get(),
        frame_state_buffer.srv.Get(),
        adaptive_buffer.srv.Get(),
        depth_width,
        depth_height,
        frame_roi_transform_buffer.srv.Get()
      );
    }

    std::vector<float> read_analysis(
      const models::scene_controller_gpu_snapshot &snapshot
    ) const {
      constexpr std::size_t analysis_pixels =
        sbs_scene_controller::analysis_canvas_size *
        sbs_scene_controller::analysis_canvas_size;
      return read_buffer<float>(
        device.Get(),
        context.Get(),
        snapshot.analysis_grid.Get(),
        sbs_scene_controller::analysis_grid_channel_count *
          analysis_pixels
      );
    }

    std::vector<std::uint32_t> read_rule_state(
      const models::scene_controller_gpu_snapshot &snapshot
    ) const {
      return read_buffer<std::uint32_t>(
        device.Get(),
        context.Get(),
        snapshot.rule_state.Get(),
        sbs_scene_controller::rule_state_word_count
      );
    }

    void write_rule_state(
      const models::scene_controller_gpu_snapshot &snapshot,
      const std::vector<std::uint32_t> &words
    ) const {
      ASSERT_EQ(
        words.size(),
        sbs_scene_controller::rule_state_word_count
      );
      ASSERT_TRUE(snapshot.rule_state);
      ComPtr<ID3D11Resource> resource;
      snapshot.rule_state->GetResource(&resource);
      ComPtr<ID3D11Buffer> buffer;
      ASSERT_TRUE(SUCCEEDED(resource.As(&buffer)));
      context->UpdateSubresource(
        buffer.Get(),
        0u,
        nullptr,
        words.data(),
        0u,
        0u
      );
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL feature_level {};
    ComPtr<ID3D11ShaderResourceView> depth_view;
    structured_buffer_t raw_depth_buffer;
    structured_buffer_t frame_state_buffer;
    structured_buffer_t adaptive_buffer;
    structured_buffer_t frame_roi_transform_buffer;
    std::unique_ptr<models::sbs_scene_controller_gpu> controller;
  };
}  // namespace

TEST_F(
  SbsSceneControllerGpu,
  RejectsMismatchedIdentityWithoutConsumingMatchedWork
) {
  constexpr std::uint64_t frame_id = 41;
  const auto source_view = make_source_view(device.Get(), 320, 180);
  ASSERT_TRUE(source_view);
  ASSERT_TRUE(controller->prepare_scene(
    source_view.Get(),
    models::input_color_space::srgb,
    frame_id
  ));

  controller->mark_enqueued(frame_id + 1u);
  EXPECT_FALSE(controller->snapshot().snapshot_available);
  EXPECT_FALSE(controller->prepare_scene(
    source_view.Get(),
    models::input_color_space::srgb,
    frame_id + 1u
  ));

  controller->mark_enqueued(frame_id);
  ASSERT_FALSE(controller->resolve_completed(
    frame_id + 1u,
    raw_depth_buffer.srv.Get(),
    depth_view.Get(),
    frame_state_buffer.srv.Get(),
    adaptive_buffer.srv.Get(),
    depth_width,
    depth_height,
    frame_roi_transform_buffer.srv.Get()
  ));
  EXPECT_FALSE(controller->snapshot().snapshot_available);

  ASSERT_TRUE(controller->resolve_completed(
    frame_id,
    raw_depth_buffer.srv.Get(),
    depth_view.Get(),
    frame_state_buffer.srv.Get(),
    adaptive_buffer.srv.Get(),
    depth_width,
    depth_height,
    frame_roi_transform_buffer.srv.Get()
  ));
  const auto snapshot = controller->snapshot();
  EXPECT_EQ(snapshot.source_frame_id, frame_id);
  expect_canonical_viewport(read_analysis(snapshot), 0, 28, 128, 72);
  expect_snapshot_abi_invariants(device.Get(), context.Get(), snapshot);
}

TEST_F(
  SbsSceneControllerGpu,
  ShaderCacheInvalidatesWhenIncludedSourceChanges
) {
  scoped_temp_directory_t temp_directory(
    "apollo-scene-shader-cache-test"
  );
  ASSERT_FALSE(temp_directory.path().empty());

  const auto source_shader_root =
    std::filesystem::path(SUNSHINE_SHADERS_DIR);
  const auto copied_assets_root =
    temp_directory.path() / "assets";
  const auto copied_shader_root =
    copied_assets_root / "shaders" / "directx";
  std::error_code filesystem_error;
  ASSERT_TRUE(std::filesystem::create_directories(
    copied_shader_root,
    filesystem_error
  ));
  ASSERT_FALSE(filesystem_error) << filesystem_error.message();
  std::filesystem::copy(
    source_shader_root,
    copied_shader_root,
    std::filesystem::copy_options::recursive |
      std::filesystem::copy_options::overwrite_existing,
    filesystem_error
  );
  ASSERT_FALSE(filesystem_error) << filesystem_error.message();

  config::video_t::sbs_t sbs;
  sbs.scene_controller =
    config::sbs_scene_controller_e::shadow_rules;
  {
    models::sbs_scene_controller_gpu first(
      device,
      context,
      copied_assets_root,
      sbs.scene_controller,
      sbs
    );
    ASSERT_TRUE(first.enabled());
    ASSERT_TRUE(first.valid());
  }

  const auto included_source =
    copied_shader_root /
    "include" /
    "sbs_scene_controller_constants.hlsl";
  {
    std::ofstream invalid_include(
      included_source,
      std::ios::binary | std::ios::app
    );
    ASSERT_TRUE(invalid_include.good());
    invalid_include
      << "\nfloat4 apollo_shader_cache_regression_invalid( = ;\n";
    ASSERT_TRUE(invalid_include.good());
  }

  models::sbs_scene_controller_gpu second(
    device,
    context,
    copied_assets_root,
    sbs.scene_controller,
    sbs
  );
  EXPECT_TRUE(second.enabled());
  EXPECT_FALSE(second.valid());
}

TEST_F(
  SbsSceneControllerGpu,
  OffsetOwnedRoiMapsFullFrameEvidenceIntoTheMatchedModelLocation
) {
  constexpr UINT source_width = 280u;
  constexpr UINT source_height = 280u;
  constexpr UINT model_width = 28u;
  constexpr UINT model_height = 28u;
  constexpr std::uint64_t frame_id = 0x12345678000001F5ull;
  constexpr std::array<float, 4> roi {
    0.125f,
    0.25f,
    0.625f,
    0.75f,
  };

  const auto source = make_source_view(
    device.Get(),
    source_width,
    source_height
  );
  ASSERT_TRUE(source);
  std::vector<float> normalized_depth(
    model_width * model_height,
    0.1f
  );
  for (UINT y = 0u; y < model_height; ++y) {
    normalized_depth[y * model_width + 21u] = 0.9f;
  }
  const auto normalized = make_depth_view(
    device.Get(),
    model_width,
    model_height,
    normalized_depth
  );
  std::vector<float> raw_depth(
    model_width * model_height,
    0.5f
  );
  const auto raw = make_structured_buffer(
    device.Get(),
    raw_depth.data(),
    static_cast<UINT>(raw_depth.size() * sizeof(float)),
    sizeof(float)
  );
  const auto adaptive = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f
  );
  const auto transform_words = make_active_frame_roi_transform(
    frame_id,
    source_width,
    source_height,
    model_width,
    model_height,
    roi,
    roi
  );
  const auto transform = make_structured_buffer(
    device.Get(),
    transform_words.data(),
    sizeof(transform_words),
    sizeof(transform_words[0])
  );
  ASSERT_TRUE(normalized);
  ASSERT_TRUE(raw.srv);
  ASSERT_TRUE(adaptive.srv);
  ASSERT_TRUE(transform.srv);

  ASSERT_TRUE(controller->prepare_scene(
    source.Get(),
    models::input_color_space::srgb,
    frame_id
  ));
  controller->mark_enqueued(frame_id);
  ASSERT_TRUE(controller->resolve_completed(
    frame_id,
    raw.srv.Get(),
    normalized.Get(),
    frame_state_buffer.srv.Get(),
    adaptive.srv.Get(),
    model_width,
    model_height,
    transform.srv.Get()
  ));

  constexpr std::size_t canvas =
    sbs_scene_controller::analysis_canvas_size;
  constexpr std::size_t plane = canvas * canvas;
  const auto depth_history = read_buffer<float>(
    device.Get(),
    context.Get(),
    controller->snapshot().depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count * plane
  );
  const auto history_index =
    [](const sbs_scene_controller::depth_history_channel_e channel,
       const std::size_t x,
       const std::size_t y) {
      return static_cast<std::size_t>(channel) * plane +
             y * canvas + x;
    };
  const auto last =
    sbs_scene_controller::depth_history_channel_e::last_normalized_depth;

  // Source cell (64,64) maps through the offset crop to model column 21. A legacy direct
  // full-frame lookup would sample model column 14 and return 0.1 instead.
  EXPECT_FLOAT_EQ(depth_history[history_index(last, 64u, 64u)], 0.9f);
  // The model hotspot's old direct full-frame source position is outside the accepted ROI.
  EXPECT_FLOAT_EQ(depth_history[history_index(last, 96u, 64u)], 0.0f);
  EXPECT_FLOAT_EQ(depth_history[history_index(last, 112u, 64u)], 0.0f);
}

TEST_F(
  SbsSceneControllerGpu,
  MetaRoiGenerationChangedRequiresValidatedOwnedResetDebt
) {
  constexpr UINT source_width = 280u;
  constexpr UINT source_height = 280u;
  constexpr UINT model_width = 28u;
  constexpr UINT model_height = 28u;
  constexpr std::array<float, 4> full_frame {
    0.0f,
    0.0f,
    1.0f,
    1.0f,
  };
  const auto source = make_source_view(
    device.Get(),
    source_width,
    source_height
  );
  const auto normalized_depth = make_depth_view(
    device.Get(),
    model_width,
    model_height,
    std::vector<float>(model_width * model_height, 0.5f)
  );
  const std::vector<float> raw_depth(
    model_width * model_height,
    0.5f
  );
  const auto raw_depth_for_model = make_structured_buffer(
    device.Get(),
    raw_depth.data(),
    static_cast<UINT>(raw_depth.size() * sizeof(float)),
    sizeof(float)
  );
  const auto adaptive = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f
  );
  ASSERT_TRUE(source);
  ASSERT_TRUE(normalized_depth);
  ASSERT_TRUE(raw_depth_for_model.srv);
  ASSERT_TRUE(adaptive.srv);

  const auto run_with_transform =
    [&](const std::uint64_t frame_id,
        ID3D11ShaderResourceView *transform) {
      EXPECT_TRUE(controller->prepare_scene(
        source.Get(),
        models::input_color_space::srgb,
        frame_id
      ));
      controller->mark_enqueued(frame_id);
      EXPECT_TRUE(controller->resolve_completed(
        frame_id,
        raw_depth_for_model.srv.Get(),
        normalized_depth.Get(),
        frame_state_buffer.srv.Get(),
        adaptive.srv.Get(),
        model_width,
        model_height,
        transform
      ));
      const auto meta = read_buffer<float>(
        device.Get(),
        context.Get(),
        controller->snapshot().meta.Get(),
        sbs_scene_controller::meta_word_count
      );
      return meta[static_cast<std::size_t>(
        sbs_scene_controller::meta_word_e::roi_generation_changed
      )];
    };
  const auto make_owned_transform =
    [&](const std::uint64_t frame_id, const bool reset_debt) {
      auto words = make_active_frame_roi_transform(
        frame_id,
        source_width,
        source_height,
        model_width,
        model_height,
        full_frame,
        full_frame
      );
      if (reset_debt) {
        words[0][1] |= 1u << 3u;
      }
      return make_structured_buffer(
        device.Get(),
        words.data(),
        sizeof(words),
        sizeof(words[0])
      );
    };

  const frame_roi_transform_words_t zero_words {};
  const auto legacy_transform = make_structured_buffer(
    device.Get(),
    zero_words.data(),
    sizeof(zero_words),
    sizeof(zero_words[0])
  );
  ASSERT_TRUE(legacy_transform.srv);
  EXPECT_FLOAT_EQ(
    run_with_transform(500u, legacy_transform.srv.Get()),
    0.0f
  );

  const auto debt_transform = make_owned_transform(501u, true);
  ASSERT_TRUE(debt_transform.srv);
  EXPECT_FLOAT_EQ(
    run_with_transform(501u, debt_transform.srv.Get()),
    1.0f
  );

  const auto steady_transform = make_owned_transform(502u, false);
  ASSERT_TRUE(steady_transform.srv);
  EXPECT_FLOAT_EQ(
    run_with_transform(502u, steady_transform.srv.Get()),
    0.0f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  EvaluationElapsedOverridePreservesLongFiniteSourceTime
) {
  constexpr std::uint64_t frame_id = 73u;
  constexpr float expected_elapsed = 2.375f;
  const auto source_view = make_source_view(device.Get(), 320, 180);
  ASSERT_TRUE(source_view);

  EXPECT_FALSE(
    controller->set_next_elapsed_seconds_for_evaluation(-0.01f)
  );
  EXPECT_FALSE(
    controller->set_next_elapsed_seconds_for_evaluation(
      std::numeric_limits<float>::quiet_NaN()
    )
  );
  ASSERT_TRUE(
    controller->set_next_elapsed_seconds_for_evaluation(expected_elapsed)
  );
  ASSERT_TRUE(controller->prepare_scene(
    source_view.Get(),
    models::input_color_space::srgb,
    frame_id
  ));
  controller->mark_enqueued(frame_id);
  ASSERT_TRUE(controller->resolve_completed(
    frame_id,
    raw_depth_buffer.srv.Get(),
    depth_view.Get(),
    frame_state_buffer.srv.Get(),
    adaptive_buffer.srv.Get(),
    depth_width,
    depth_height,
    frame_roi_transform_buffer.srv.Get()
  ));
  const auto snapshot = controller->snapshot();
  const auto meta = read_buffer<float>(
    device.Get(),
    context.Get(),
    snapshot.meta.Get(),
    sbs_scene_controller::meta_word_count
  );
  ASSERT_EQ(meta.size(), sbs_scene_controller::meta_word_count);
  EXPECT_FLOAT_EQ(
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::elapsed_seconds
    )],
    expected_elapsed
  );
}

TEST_F(
  SbsSceneControllerGpu,
  GeometryReseedRebasesDepthHistoryWithoutEmittingStructuralCutEvidence
) {
  constexpr UINT source_width = 320u;
  constexpr UINT source_height = 180u;
  constexpr std::size_t canvas =
    sbs_scene_controller::analysis_canvas_size;
  constexpr std::size_t plane = canvas * canvas;
  constexpr std::array<float, 4> full_frame {
    0.0f,
    0.0f,
    1.0f,
    1.0f,
  };

  const auto set_depth = [&](const float value) {
    depth_view = make_depth_view(
      device.Get(),
      depth_width,
      depth_height,
      std::vector<float>(depth_width * depth_height, value)
    );
    ASSERT_TRUE(depth_view);
  };
  const auto set_frame_state = [&](const float state) {
    const std::array<float, 4> values {
      0.0f,
      1.0f,
      1.0f,
      state,
    };
    frame_state_buffer = make_structured_buffer(
      device.Get(),
      values.data(),
      sizeof(values),
      sizeof(values)
    );
    ASSERT_TRUE(frame_state_buffer.srv);
  };
  const auto set_active_transform =
    [&](const std::uint64_t frame_id, const bool reset_debt) {
      auto words = make_active_frame_roi_transform(
        frame_id,
        source_width,
        source_height,
        depth_width,
        depth_height,
        full_frame,
        full_frame
      );
      if (reset_debt) {
        words[0][1] |= 1u << 3u;
      }
      frame_roi_transform_buffer = make_structured_buffer(
        device.Get(),
        words.data(),
        sizeof(words),
        sizeof(words[0])
      );
      ASSERT_TRUE(frame_roi_transform_buffer.srv);
    };
  const auto set_legacy_transform = [&]() {
    const frame_roi_transform_words_t words {};
    frame_roi_transform_buffer = make_structured_buffer(
      device.Get(),
      words.data(),
      sizeof(words),
      sizeof(words[0])
    );
    ASSERT_TRUE(frame_roi_transform_buffer.srv);
  };
  const auto channel_max =
    [&](const models::scene_controller_gpu_snapshot &snapshot,
        const sbs_scene_controller::dense_out_channel_e channel) {
      const auto dense = read_buffer<float>(
        device.Get(),
        context.Get(),
        snapshot.dense_output.Get(),
        sbs_scene_controller::dense_out_channel_count * plane
      );
      const auto first =
        dense.begin() +
        static_cast<std::size_t>(channel) * plane;
      return *std::max_element(first, first + plane);
    };
  const auto activity_max =
    [&](const models::scene_controller_gpu_snapshot &snapshot) {
      const auto analysis = read_analysis(snapshot);
      const auto first =
        analysis.begin() +
        static_cast<std::size_t>(
          sbs_scene_controller::analysis_grid_channel_e::
            temporal_activity_occupancy
        ) * plane;
      return *std::max_element(first, first + plane);
    };
  set_frame_state(1.0f);
  set_depth(0.1f);
  set_active_transform(200u, false);
  ASSERT_TRUE(run_frame(source_width, source_height, 200u, 0u));

  // A transform reset-debt completion may have a completely different normalized range. It must
  // seed the new history rather than compare 0.9 against the previous geometry's 0.1.
  set_depth(0.9f);
  set_active_transform(201u, true);
  ASSERT_TRUE(run_frame(source_width, source_height, 201u, 1u));
  const auto debt_snapshot = controller->snapshot();
  EXPECT_GT(activity_max(debt_snapshot), 0.0f);
  EXPECT_FLOAT_EQ(
    channel_max(
      debt_snapshot,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support
    ),
    0.0f
  );

  // The frame-state reseed marker is the equivalent contract for legacy/full-frame geometry.
  set_frame_state(2.0f);
  set_depth(0.2f);
  set_legacy_transform();
  ASSERT_TRUE(run_frame(source_width, source_height, 202u, 0u));
  const auto state_snapshot = controller->snapshot();
  EXPECT_GT(activity_max(state_snapshot), 0.0f);
  EXPECT_FLOAT_EQ(
    channel_max(
      state_snapshot,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support
    ),
    0.0f
  );

  // Abstention is one frame only. The reseeded 0.2 history becomes comparable again once
  // frame_state returns to the ordinary valid value.
  set_frame_state(1.0f);
  set_depth(0.8f);
  ASSERT_TRUE(run_frame(source_width, source_height, 203u, 1u));
  const auto recovered_snapshot = controller->snapshot();
  EXPECT_GT(
    channel_max(
      recovered_snapshot,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support
    ),
    0.0f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  DepthInvalidCellsDoNotEmitStructuralOrExposureEvidence
) {
  constexpr UINT source_width = 320u;
  constexpr UINT source_height = 180u;
  constexpr std::size_t canvas =
    sbs_scene_controller::analysis_canvas_size;
  constexpr std::size_t plane = canvas * canvas;

  ASSERT_TRUE(run_frame(source_width, source_height, 300u, 0u));
  const std::vector<float> invalid_raw_depth(
    depth_width * depth_height,
    -1.0f
  );
  raw_depth_buffer = make_structured_buffer(
    device.Get(),
    invalid_raw_depth.data(),
    static_cast<UINT>(invalid_raw_depth.size() * sizeof(float)),
    sizeof(float)
  );
  ASSERT_TRUE(raw_depth_buffer.srv);
  ASSERT_TRUE(run_frame(source_width, source_height, 301u, 1u));

  const auto snapshot = controller->snapshot();
  const auto analysis = read_analysis(snapshot);
  const auto dense = read_buffer<float>(
    device.Get(),
    context.Get(),
    snapshot.dense_output.Get(),
    sbs_scene_controller::dense_out_channel_count * plane
  );
  const auto channel_begin =
    [&](const auto channel) {
      return dense.begin() +
             static_cast<std::size_t>(channel) * plane;
    };
  const auto activity_begin =
    analysis.begin() +
    static_cast<std::size_t>(
      sbs_scene_controller::analysis_grid_channel_e::
        temporal_activity_occupancy
    ) *
      plane;
  EXPECT_GT(
    *std::max_element(activity_begin, activity_begin + plane),
    0.0f
  );
  for (const auto channel : {
         sbs_scene_controller::dense_out_channel_e::
           structural_content_cut_support,
         sbs_scene_controller::dense_out_channel_e::
           exposure_only_change_support,
       }) {
    const auto first = channel_begin(channel);
    EXPECT_FLOAT_EQ(
      *std::max_element(first, first + plane),
      0.0f
    ) << sbs_scene_controller::dense_out_names[
      static_cast<std::size_t>(channel)
    ];
  }
}

TEST_F(
  SbsSceneControllerGpu,
  StaleAndMalformedOwnedTransformsAbstainAndRetainDepthHistory
) {
  constexpr UINT source_width = 280u;
  constexpr UINT source_height = 280u;
  constexpr UINT model_width = 28u;
  constexpr UINT model_height = 28u;
  constexpr std::uint64_t baseline_frame_id = 0x8765432100000259ull;
  constexpr std::array<float, 4> roi {
    0.125f,
    0.25f,
    0.625f,
    0.75f,
  };
  constexpr std::size_t canvas =
    sbs_scene_controller::analysis_canvas_size;
  constexpr std::size_t plane = canvas * canvas;
  constexpr std::size_t depth_value_channels =
    static_cast<std::size_t>(
      sbs_scene_controller::depth_history_channel_e::
        valid_depth_confidence
    );

  const auto source = make_source_view(
    device.Get(),
    source_width,
    source_height
  );
  ASSERT_TRUE(source);
  std::vector<float> baseline_values(
    model_width * model_height,
    0.37f
  );
  std::vector<float> replacement_values(
    model_width * model_height,
    0.91f
  );
  const auto baseline_depth = make_depth_view(
    device.Get(),
    model_width,
    model_height,
    baseline_values
  );
  const auto replacement_depth = make_depth_view(
    device.Get(),
    model_width,
    model_height,
    replacement_values
  );
  std::vector<float> raw_depth(
    model_width * model_height,
    0.5f
  );
  const auto raw = make_structured_buffer(
    device.Get(),
    raw_depth.data(),
    static_cast<UINT>(raw_depth.size() * sizeof(float)),
    sizeof(float)
  );
  const auto adaptive = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f
  );
  ASSERT_TRUE(baseline_depth);
  ASSERT_TRUE(replacement_depth);
  ASSERT_TRUE(raw.srv);
  ASSERT_TRUE(adaptive.srv);

  const auto resolve =
    [&](const std::uint64_t frame_id,
        ID3D11ShaderResourceView *depth,
        ID3D11ShaderResourceView *transform) {
      if (!controller->prepare_scene(
            source.Get(),
            models::input_color_space::srgb,
            frame_id
          )) {
        return false;
      }
      controller->mark_enqueued(frame_id);
      return controller->resolve_completed(
        frame_id,
        raw.srv.Get(),
        depth,
        frame_state_buffer.srv.Get(),
        adaptive.srv.Get(),
        model_width,
        model_height,
        transform
      );
    };

  ASSERT_TRUE(resolve(
    baseline_frame_id,
    baseline_depth.Get(),
    frame_roi_transform_buffer.srv.Get()
  ));
  const auto baseline = read_buffer<float>(
    device.Get(),
    context.Get(),
    controller->snapshot().depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count * plane
  );

  auto stale_words = make_active_frame_roi_transform(
    baseline_frame_id,
    source_width,
    source_height,
    model_width,
    model_height,
    roi,
    roi
  );
  const auto stale = make_structured_buffer(
    device.Get(),
    stale_words.data(),
    sizeof(stale_words),
    sizeof(stale_words[0])
  );
  ASSERT_TRUE(stale.srv);
  ASSERT_TRUE(resolve(
    baseline_frame_id + 1u,
    replacement_depth.Get(),
    stale.srv.Get()
  ));
  const auto stale_result = read_buffer<float>(
    device.Get(),
    context.Get(),
    controller->snapshot().depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count * plane
  );

  auto malformed_words = make_active_frame_roi_transform(
    baseline_frame_id + 2u,
    source_width,
    source_height,
    model_width,
    model_height,
    roi,
    roi
  );
  malformed_words[0][0] =
    models::frame_roi_transform_contract_version + 1u;
  const auto malformed = make_structured_buffer(
    device.Get(),
    malformed_words.data(),
    sizeof(malformed_words),
    sizeof(malformed_words[0])
  );
  ASSERT_TRUE(malformed.srv);
  ASSERT_TRUE(resolve(
    baseline_frame_id + 2u,
    replacement_depth.Get(),
    malformed.srv.Get()
  ));
  const auto malformed_result = read_buffer<float>(
    device.Get(),
    context.Get(),
    controller->snapshot().depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count * plane
  );

  auto wrong_source_words = make_active_frame_roi_transform(
    baseline_frame_id + 3u,
    source_width * 2u,
    source_height * 2u,
    model_width,
    model_height,
    roi,
    roi
  );
  const auto wrong_source = make_structured_buffer(
    device.Get(),
    wrong_source_words.data(),
    sizeof(wrong_source_words),
    sizeof(wrong_source_words[0])
  );
  ASSERT_TRUE(wrong_source.srv);
  ASSERT_TRUE(resolve(
    baseline_frame_id + 3u,
    replacement_depth.Get(),
    wrong_source.srv.Get()
  ));
  const auto wrong_source_result = read_buffer<float>(
    device.Get(),
    context.Get(),
    controller->snapshot().depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count * plane
  );

  constexpr std::array<float, 4> wide_roi {
    0.125f,
    0.25f,
    0.625f,
    0.50f,
  };
  auto wrong_model_words = make_active_frame_roi_transform(
    baseline_frame_id + 4u,
    source_width,
    source_height,
    model_width * 2u,
    model_height,
    wide_roi,
    wide_roi
  );
  const auto wrong_model = make_structured_buffer(
    device.Get(),
    wrong_model_words.data(),
    sizeof(wrong_model_words),
    sizeof(wrong_model_words[0])
  );
  ASSERT_TRUE(wrong_model.srv);
  ASSERT_TRUE(resolve(
    baseline_frame_id + 4u,
    replacement_depth.Get(),
    wrong_model.srv.Get()
  ));
  const auto wrong_model_result = read_buffer<float>(
    device.Get(),
    context.Get(),
    controller->snapshot().depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count * plane
  );

  // Geometry-bearing history remains exact. Validity confidence/age are intentionally the only
  // channels allowed to degrade when current depth abstains.
  for (std::size_t channel = 0u;
       channel < depth_value_channels;
       ++channel) {
    const auto first = baseline.begin() + channel * plane;
    const auto last = first + plane;
    EXPECT_TRUE(std::equal(
      first,
      last,
      stale_result.begin() + channel * plane
    )) << "stale transform changed depth channel " << channel;
    EXPECT_TRUE(std::equal(
      first,
      last,
      malformed_result.begin() + channel * plane
    )) << "malformed transform changed depth channel " << channel;
    EXPECT_TRUE(std::equal(
      first,
      last,
      wrong_source_result.begin() + channel * plane
    )) << "wrong-source transform changed depth channel " << channel;
    EXPECT_TRUE(std::equal(
      first,
      last,
      wrong_model_result.begin() + channel * plane
    )) << "wrong-model transform changed depth channel " << channel;
  }

  const auto meta = read_buffer<float>(
    device.Get(),
    context.Get(),
    controller->snapshot().meta.Get(),
    sbs_scene_controller::meta_word_count
  );
  EXPECT_FLOAT_EQ(
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::depth_input_valid
    )],
    0.0f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  HighBitExternalCutCounterUsesExactUintTransport
) {
  constexpr std::uint32_t subnormal_counter = 0x80000001u;
  constexpr std::uint32_t nan_payload_counter = 0xffc12345u;
  constexpr std::uint32_t reset_counter = 0xffd12345u;
  constexpr std::uint32_t post_reset_counter = 0xffe12345u;
  const auto external_count_index = sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::last_external_cut_count
  );
  const auto request_index = static_cast<std::size_t>(
    sbs_scene_controller::meta_word_e::external_cut_request
  );
  const auto check_counter =
    [&](const UINT source_width,
        const UINT source_height,
        const std::uint64_t frame_id,
        const std::uint32_t counter,
        const float expected_request) {
      adaptive_buffer = make_adaptive_state_buffer(
        device.Get(),
        counter,
        1.0f
      );
      ASSERT_TRUE(adaptive_buffer.srv);
      ASSERT_TRUE(run_frame(source_width, source_height, frame_id));
      const auto snapshot = controller->snapshot();
      const auto state = read_rule_state(snapshot);
      EXPECT_EQ(state[external_count_index], counter);
      const auto meta = read_buffer<float>(
        device.Get(),
        context.Get(),
        snapshot.meta.Get(),
        sbs_scene_controller::meta_word_count
      );
      EXPECT_FLOAT_EQ(meta[request_index], expected_request);
      expect_snapshot_abi_invariants(
        device.Get(),
        context.Get(),
        snapshot
      );
    };

  ASSERT_NO_FATAL_FAILURE(
    check_counter(320, 180, 160, subnormal_counter, 0.0f)
  );
  ASSERT_NO_FATAL_FAILURE(
    check_counter(320, 180, 161, nan_payload_counter, 1.0f)
  );
  ASSERT_NO_FATAL_FAILURE(
    check_counter(320, 180, 162, nan_payload_counter, 0.0f)
  );
  ASSERT_NO_FATAL_FAILURE(
    check_counter(640, 180, 163, reset_counter, 0.0f)
  );
  ASSERT_NO_FATAL_FAILURE(
    check_counter(640, 180, 164, post_reset_counter, 1.0f)
  );
}

TEST_F(
  SbsSceneControllerGpu,
  ExternalCutIncrementDuringInvalidFramePulsesOnceOnRecovery
) {
  constexpr std::size_t analysis_pixels =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  constexpr std::uint32_t baseline_counter = 41u;
  constexpr std::uint32_t incremented_counter = 42u;
  const auto external_count_index = sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::last_external_cut_count
  );
  const auto request_index = static_cast<std::size_t>(
    sbs_scene_controller::meta_word_e::external_cut_request
  );
  const auto request_value =
    [&](const models::scene_controller_gpu_snapshot &snapshot) {
      const auto meta = read_buffer<float>(
        device.Get(),
        context.Get(),
        snapshot.meta.Get(),
        sbs_scene_controller::meta_word_count
      );
      return meta[request_index];
    };
  const auto global_output =
    [&](const models::scene_controller_gpu_snapshot &snapshot) {
      return read_buffer<float>(
        device.Get(),
        context.Get(),
        snapshot.global_output.Get(),
        sbs_scene_controller::global_out_word_count
      );
    };
  const auto event_argmax =
    [](const std::vector<float> &global) {
      constexpr auto first = static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::event_same_shot
      );
      constexpr auto end =
        static_cast<std::size_t>(
          sbs_scene_controller::global_out_word_e::
            event_geometry_reset
        ) +
        1u;
      return static_cast<std::size_t>(std::distance(
        global.begin(),
        std::max_element(
          global.begin() + first,
          global.begin() + end
        )
      ));
    };

  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    baseline_counter,
    1.0f
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  ASSERT_TRUE(run_frame(320, 180, 170));
  const auto baseline_snapshot = controller->snapshot();
  EXPECT_EQ(
    read_rule_state(baseline_snapshot)[external_count_index],
    baseline_counter
  );
  const auto baseline_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    baseline_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  const auto baseline_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    baseline_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );

  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    incremented_counter,
    1.0f
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  const auto invalid_source = make_nan_source_view(
    device.Get(),
    320,
    180
  );
  ASSERT_TRUE(invalid_source);
  ASSERT_TRUE(run_source(invalid_source.Get(), 171));
  const auto invalid_snapshot = controller->snapshot();
  EXPECT_FLOAT_EQ(request_value(invalid_snapshot), 1.0f);
  const auto invalid_state = read_rule_state(invalid_snapshot);
  EXPECT_EQ(
    invalid_state[external_count_index],
    baseline_counter
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      invalid_state,
      sbs_scene_controller::rule_state_word_e::output_valid
    ),
    0.0f
  );
  EXPECT_EQ(
    read_buffer<float>(
      device.Get(),
      context.Get(),
      invalid_snapshot.layout_history.Get(),
      sbs_scene_controller::layout_history_channel_count *
        analysis_pixels
    ),
    baseline_layout
  );
  EXPECT_EQ(
    read_buffer<float>(
      device.Get(),
      context.Get(),
      invalid_snapshot.depth_history.Get(),
      sbs_scene_controller::depth_history_channel_count *
        analysis_pixels
    ),
    baseline_depth
  );

  ASSERT_TRUE(run_frame(320, 180, 172));
  const auto recovered_snapshot = controller->snapshot();
  const auto recovered_state = read_rule_state(recovered_snapshot);
  const auto recovered_global = global_output(recovered_snapshot);
  EXPECT_FLOAT_EQ(request_value(recovered_snapshot), 1.0f);
  EXPECT_EQ(
    recovered_state[external_count_index],
    incremented_counter
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      recovered_state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      recovered_state,
      sbs_scene_controller::rule_state_word_e::event_decision
    ),
    static_cast<float>(
      sbs_scene_controller::event_decision_e::hard_cut
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      recovered_state,
      sbs_scene_controller::rule_state_word_e::event_confidence
    ),
    1.0f
  );
  EXPECT_EQ(
    event_argmax(recovered_global),
    static_cast<std::size_t>(
      sbs_scene_controller::global_out_word_e::event_hard_cut
    )
  );
  EXPECT_FLOAT_EQ(
    recovered_global[static_cast<std::size_t>(
      sbs_scene_controller::global_out_word_e::confidence_event
    )],
    1.0f
  );
  EXPECT_NE(
    recovered_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::promotion_flags
    )] &
      sbs_scene_controller::promotion_flags_shot,
    0u
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      recovered_state,
      sbs_scene_controller::rule_state_word_e::scene_age_s
    ),
    0.0f
  );

  const auto recovered_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    recovered_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  const auto recovered_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    recovered_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );
  const auto layout_offset =
    [](const sbs_scene_controller::layout_history_channel_e channel) {
      return static_cast<std::size_t>(channel) * analysis_pixels;
    };
  const auto depth_offset =
    [](const sbs_scene_controller::depth_history_channel_e channel) {
      return static_cast<std::size_t>(channel) * analysis_pixels;
    };
  const auto fast_activity = layout_offset(
    sbs_scene_controller::layout_history_channel_e::fast_activity_ema
  );
  EXPECT_TRUE(std::all_of(
    recovered_layout.begin() + fast_activity,
    recovered_layout.begin() + fast_activity + analysis_pixels,
    [](const float value) {
      return value == 0.0f;
    }
  ));
  const auto last_depth = depth_offset(
    sbs_scene_controller::depth_history_channel_e::
      last_normalized_depth
  );
  for (const auto channel : {
         sbs_scene_controller::depth_history_channel_e::fast_depth_ema,
         sbs_scene_controller::depth_history_channel_e::slow_depth_ema,
         sbs_scene_controller::depth_history_channel_e::
           settled_shot_reference_depth,
       }) {
    const auto offset = depth_offset(channel);
    SCOPED_TRACE(static_cast<std::size_t>(channel));
    EXPECT_TRUE(std::equal(
      recovered_depth.begin() + offset,
      recovered_depth.begin() + offset + analysis_pixels,
      recovered_depth.begin() + last_depth
    ));
  }
  for (const auto channel : {
         sbs_scene_controller::depth_history_channel_e::
           depth_variance_ema,
         sbs_scene_controller::depth_history_channel_e::
           absolute_depth_change_ema,
       }) {
    const auto offset = depth_offset(channel);
    SCOPED_TRACE(static_cast<std::size_t>(channel));
    EXPECT_TRUE(std::all_of(
      recovered_depth.begin() + offset,
      recovered_depth.begin() + offset + analysis_pixels,
      [](const float value) {
        return value == 0.0f;
      }
    ));
  }

  ASSERT_TRUE(run_frame(320, 180, 173));
  const auto steady_snapshot = controller->snapshot();
  const auto steady_state = read_rule_state(steady_snapshot);
  const auto steady_global = global_output(steady_snapshot);
  EXPECT_FLOAT_EQ(request_value(steady_snapshot), 0.0f);
  EXPECT_EQ(
    steady_state[external_count_index],
    incremented_counter
  );
  EXPECT_NE(
    word_as_float(
      steady_state,
      sbs_scene_controller::rule_state_word_e::event_decision
    ),
    static_cast<float>(
      sbs_scene_controller::event_decision_e::hard_cut
    )
  );
  EXPECT_NE(
    event_argmax(steady_global),
    static_cast<std::size_t>(
      sbs_scene_controller::global_out_word_e::event_hard_cut
    )
  );
  EXPECT_EQ(
    steady_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::promotion_flags
    )] &
      sbs_scene_controller::promotion_flags_shot,
    0u
  );
  EXPECT_GT(
    word_as_float(
      steady_state,
      sbs_scene_controller::rule_state_word_e::scene_age_s
    ),
    0.0f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  DurableDetectorCutCounterIsConsumedOrRetainedExactlyOnce
) {
  struct alignas(16) scene_constants_t {
    std::uint32_t source_width = 320u;
    std::uint32_t source_height = 180u;
    std::uint32_t depth_width = 64u;
    std::uint32_t depth_height = 36u;
    std::uint32_t color_mode = 0u;
    std::uint32_t backend_generation = 1u;
    std::uint32_t history_valid = 1u;
    std::uint32_t reset_flags = 0u;
    float elapsed_seconds = 1.0f / 60.0f;
    float pop_floor = 1.2f;
    float pop_ceiling = 2.0f;
    float zero_plane_mode = 2.0f;
    float acquire_seconds = 0.12f;
    float challenger_seconds = 0.30f;
    float release_seconds = 60.0f;
    float scroll_enter_seconds = 0.05f;
    std::uint32_t source_frame_id_low = 0u;
    std::uint32_t source_frame_id_high = 0u;
    std::uint32_t identity_reserved_0 = 0u;
    std::uint32_t identity_reserved_1 = 0u;
    std::array<std::uint32_t, 8> ordered_abi_hash_words =
      sbs_scene_controller::ordered_abi_hash_words;
  };
  static_assert(sizeof(scene_constants_t) == 112u);

  struct step_result_t {
    std::vector<std::uint32_t> state;
    std::vector<float> global;
  };

  // Keep the direct resolver fixture exact with SBS_RULE_SUMMARY_FLOAT_COUNT.
  constexpr std::size_t rule_summary_float_count =
    SBS_RULE_SUMMARY_FLOAT_COUNT;
  constexpr std::uint32_t baseline_detector_count = 10u;
  constexpr std::uint32_t baseline_external_count = 20u;
  const auto resolver = compile_compute_shader(
    device.Get(),
    "sbs_scene_rules_resolve_cs.hlsl"
  );
  ASSERT_TRUE(resolver);
  bool reliable_roi_event_for_step = false;
  bool insufficient_roi_event_for_step = false;

  const auto resolve_step =
    [&](const std::vector<std::uint32_t> &previous_state,
        const std::uint32_t detector_cut_count,
        const std::uint32_t external_cut_count,
        const sbs_scene_controller::event_decision_e evidence_event,
        const bool evidence_valid,
        const bool scrolling,
        const bool external_cut_request,
        const std::uint32_t backend_generation,
        const std::uint32_t reset_flags = 0u,
        const bool competing_scroll_event = false) {
      std::array<float, sbs_scene_controller::global_out_word_count>
        evidence {};
      evidence[static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::layout_no_target
      )] = 1.0f;
      const auto event_index =
        static_cast<std::size_t>(
          sbs_scene_controller::global_out_word_e::event_same_shot
        ) +
        static_cast<std::size_t>(evidence_event);
      evidence[event_index] = 1.0f;
      if (competing_scroll_event) {
        evidence[static_cast<std::size_t>(
          sbs_scene_controller::global_out_word_e::event_scroll
        )] = 2.0f;
      }
      evidence[static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::
          global_scroll_confidence
      )] = scrolling ? 1.0f : 0.0f;
      evidence[static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::global_scroll_y
      )] = scrolling ? 1.0f : 0.0f;
      evidence[static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::backend_output_valid
      )] = evidence_valid ? 1.0f : 0.0f;
      evidence[static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::reserved_40
      )] = 1.0f;

      std::array<float, sbs_scene_controller::meta_word_count> meta {};
      meta[static_cast<std::size_t>(
        sbs_scene_controller::meta_word_e::external_cut_request
      )] = external_cut_request ? 1.0f : 0.0f;
      meta[static_cast<std::size_t>(
        sbs_scene_controller::meta_word_e::depth_input_valid
      )] = 1.0f;
      meta[static_cast<std::size_t>(
        sbs_scene_controller::meta_word_e::pop_floor
      )] = 1.2f;
      meta[static_cast<std::size_t>(
        sbs_scene_controller::meta_word_e::pop_ceiling
      )] = 2.0f;

      sbs_adaptive_state::words_t adaptive_words {};
      for (std::size_t word = 0u;
           word < adaptive_words.size();
           ++word) {
        adaptive_words[word] = std::bit_cast<std::uint32_t>(
          sbs_adaptive_state::initial_values[word]
        );
      }
      adaptive_words[sbs_adaptive_state::index(
        sbs_adaptive_state::word_e::hard_cut_count
      )] = detector_cut_count;
      adaptive_words[sbs_adaptive_state::index(
        sbs_adaptive_state::word_e::external_cut_count
      )] = external_cut_count;
      adaptive_words[sbs_adaptive_state::index(
        sbs_adaptive_state::word_e::valid_depth_fraction
      )] = std::bit_cast<std::uint32_t>(1.0f);

      std::vector<float> rule_summary(
        rule_summary_float_count,
        0.0f
      );
      if (reliable_roi_event_for_step) {
        rule_summary[
          SBS_RULE_SUMMARY_PLAN_BASE +
          SBS_RULE_SUMMARY_PLAN_DEPTH_COVERAGE
        ] = 4096.0f;
        rule_summary[
          SBS_RULE_SUMMARY_PLAN_BASE +
          SBS_RULE_SUMMARY_PLAN_DEPTH_CONFIDENCE
        ] = 4096.0f;
      } else if (insufficient_roi_event_for_step) {
        rule_summary[
          SBS_RULE_SUMMARY_PLAN_BASE +
          SBS_RULE_SUMMARY_PLAN_DEPTH_COVERAGE
        ] = 4096.0f;
        rule_summary[
          SBS_RULE_SUMMARY_PLAN_BASE +
          SBS_RULE_SUMMARY_PLAN_DEPTH_CONFIDENCE
        ] = 40.0f;
      }
      const std::array<float, sbs_scene_controller::global_out_word_count>
        zero_global {};
      const std::array<std::uint32_t,
                       sbs_scene_controller::rule_state_word_count>
        zero_state {};
      const auto evidence_buffer = make_structured_buffer(
        device.Get(),
        evidence.data(),
        sizeof(evidence),
        sizeof(float)
      );
      const auto previous_state_buffer = make_structured_buffer(
        device.Get(),
        previous_state.data(),
        static_cast<UINT>(
          previous_state.size() * sizeof(std::uint32_t)
        ),
        sizeof(std::uint32_t) * 4u
      );
      const auto meta_buffer = make_structured_buffer(
        device.Get(),
        meta.data(),
        sizeof(meta),
        sizeof(float)
      );
      const auto adaptive_buffer_for_step = make_structured_buffer(
        device.Get(),
        adaptive_words.data(),
        sizeof(adaptive_words),
        sizeof(float) * 4u
      );
      const auto summary_buffer = make_structured_buffer(
        device.Get(),
        rule_summary.data(),
        static_cast<UINT>(rule_summary.size() * sizeof(float)),
        sizeof(float)
      );
      const auto global_output = make_rw_structured_buffer(
        device.Get(),
        zero_global.data(),
        sizeof(zero_global),
        sizeof(float)
      );
      const auto next_state = make_rw_structured_buffer(
        device.Get(),
        zero_state.data(),
        sizeof(zero_state),
        sizeof(std::uint32_t) * 4u
      );
      scene_constants_t constants;
      constants.backend_generation = backend_generation;
      constants.reset_flags = reset_flags;
      const auto constant_buffer = make_constant_buffer(
        device.Get(),
        &constants,
        sizeof(constants)
      );
      const std::array<ID3D11ShaderResourceView *, 5> inputs {
        evidence_buffer.srv.Get(),
        previous_state_buffer.srv.Get(),
        meta_buffer.srv.Get(),
        adaptive_buffer_for_step.srv.Get(),
        summary_buffer.srv.Get(),
      };
      const std::array<ID3D11UnorderedAccessView *, 2> outputs {
        global_output.uav.Get(),
        next_state.uav.Get(),
      };
      EXPECT_TRUE(std::all_of(
        inputs.begin(),
        inputs.end(),
        [](const auto *view) {
          return view != nullptr;
        }
      ));
      EXPECT_TRUE(std::all_of(
        outputs.begin(),
        outputs.end(),
        [](const auto *view) {
          return view != nullptr;
        }
      ));
      EXPECT_TRUE(constant_buffer);

      context->CSSetShader(resolver.Get(), nullptr, 0u);
      context->CSSetShaderResources(
        0u,
        static_cast<UINT>(inputs.size()),
        inputs.data()
      );
      context->CSSetConstantBuffers(
        0u,
        1u,
        constant_buffer.GetAddressOf()
      );
      context->CSSetUnorderedAccessViews(
        0u,
        static_cast<UINT>(outputs.size()),
        outputs.data(),
        nullptr
      );
      context->Dispatch(1u, 1u, 1u);

      const std::array<ID3D11ShaderResourceView *, 5> null_inputs {};
      const std::array<ID3D11UnorderedAccessView *, 2> null_outputs {};
      ID3D11Buffer *null_constant = nullptr;
      context->CSSetUnorderedAccessViews(
        0u,
        static_cast<UINT>(null_outputs.size()),
        null_outputs.data(),
        nullptr
      );
      context->CSSetShaderResources(
        0u,
        static_cast<UINT>(null_inputs.size()),
        null_inputs.data()
      );
      context->CSSetConstantBuffers(0u, 1u, &null_constant);
      context->CSSetShader(nullptr, nullptr, 0u);

      return step_result_t {
        read_buffer<std::uint32_t>(
          device.Get(),
          context.Get(),
          next_state.srv.Get(),
          sbs_scene_controller::rule_state_word_count
        ),
        read_buffer<float>(
          device.Get(),
          context.Get(),
          global_output.srv.Get(),
          sbs_scene_controller::global_out_word_count
        ),
      };
    };
  const auto baseline =
    [&]() {
      const std::vector<std::uint32_t> initial_state(
        sbs_scene_controller::initial_word_bits.begin(),
        sbs_scene_controller::initial_word_bits.end()
      );
      auto result = resolve_step(
        initial_state,
        baseline_detector_count,
        baseline_external_count,
        sbs_scene_controller::event_decision_e::same_shot,
        true,
        false,
        false,
        1u
      );
      EXPECT_EQ(
        result.state[sbs_scene_controller::index(
          sbs_scene_controller::rule_state_word_e::
            last_detector_cut_count
        )],
        baseline_detector_count
      );
      EXPECT_EQ(
        result.state[sbs_scene_controller::index(
          sbs_scene_controller::rule_state_word_e::
            last_external_cut_count
        )],
        baseline_external_count
      );
      return result;
    };
  const auto expect_shot =
    [&](const step_result_t &result, const bool expected) {
      EXPECT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::event_decision
        ) ==
          static_cast<float>(
            sbs_scene_controller::event_decision_e::hard_cut
          ),
        expected
      );
      EXPECT_EQ(
        (result.state[sbs_scene_controller::index(
           sbs_scene_controller::rule_state_word_e::promotion_flags
         )] &
         sbs_scene_controller::promotion_flags_shot) != 0u,
        expected
      );
    };
  const auto pending =
    [](const step_result_t &result) {
      return (
        result.state[sbs_scene_controller::index(
          sbs_scene_controller::rule_state_word_e::state_flags
        )] &
        sbs_scene_controller::state_flags_detector_cut_pending
      ) != 0u;
    };
  const auto retain_through_quiet_scroll_hold =
    [&](step_result_t result,
        const std::uint32_t detector_cut_count,
        const std::uint32_t external_cut_count,
        const std::uint32_t expected_last_detector_cut_count,
        const bool expected_pending) {
      // The scroll overlay deliberately remains frozen for 120 ms after
      // motion stops. At 60 Hz the first seven quiet updates are still
      // inside that hold; the eighth update is the release-resync boundary,
      // and only the following update may resolve a retained detector edge.
      for (std::uint32_t update = 0u; update < 7u; ++update) {
        result = resolve_step(
          result.state,
          detector_cut_count,
          external_cut_count,
          sbs_scene_controller::event_decision_e::same_shot,
          true,
          false,
          false,
          1u
        );
        expect_shot(result, false);
        EXPECT_EQ(pending(result), expected_pending);
        EXPECT_EQ(
          result.state[sbs_scene_controller::index(
            sbs_scene_controller::rule_state_word_e::
              last_detector_cut_count
          )],
          expected_last_detector_cut_count
        );
      }
      return result;
    };
  const auto force_locked_content_roi =
    [](step_result_t &result) {
      result.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::state_kind
      )] = std::bit_cast<std::uint32_t>(
        static_cast<float>(sbs_scene_controller::state_kind_e::content)
      );
      result.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::state_flags
      )] |= sbs_scene_controller::state_flags_roi_locked;
      for (const auto &[word, value] : {
             std::pair {
               sbs_scene_controller::rule_state_word_e::committed_roi_x0,
               0.25f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::committed_roi_y0,
               0.25f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::committed_roi_x1,
               0.75f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::committed_roi_y1,
               0.75f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::
                 committed_roi_confidence,
               1.0f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::committed_layout,
               static_cast<float>(
                 sbs_scene_controller::layout_decision_e::content_collage
               ),
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::
                 committed_mask_confidence,
               0.0f,
             },
           }) {
        result.state[sbs_scene_controller::index(word)] =
          std::bit_cast<std::uint32_t>(value);
      }
    };
  const auto set_state_word =
    [](step_result_t &result,
       const sbs_scene_controller::rule_state_word_e word,
       const float value) {
      result.state[sbs_scene_controller::index(word)] =
        std::bit_cast<std::uint32_t>(value);
    };
  const auto force_active_scroll_hold =
    [&](step_result_t &result) {
      result.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::state_flags
      )] |= sbs_scene_controller::state_flags_scroll_hold_active;
      set_state_word(
        result,
        sbs_scene_controller::rule_state_word_e::scroll_confidence,
        1.0f
      );
      set_state_word(
        result,
        sbs_scene_controller::rule_state_word_e::scroll_y,
        1.0f
      );
      set_state_word(
        result,
        sbs_scene_controller::rule_state_word_e::scroll_dwell_s,
        0.05f
      );
      set_state_word(
        result,
        sbs_scene_controller::rule_state_word_e::scroll_hold_s,
        0.0f
      );
    };
  const auto expect_full_frame =
    [](const step_result_t &result) {
      EXPECT_FLOAT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::state_kind
        ),
        static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
      );
      EXPECT_EQ(
        result.state[sbs_scene_controller::index(
          sbs_scene_controller::rule_state_word_e::state_flags
        )] & sbs_scene_controller::state_flags_roi_locked,
        0u
      );
      EXPECT_FLOAT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::
            committed_roi_confidence
        ),
        0.0f
      );
    };
  const auto expect_acquisition_canonical =
    [](const step_result_t &result) {
      for (const auto word : {
             sbs_scene_controller::rule_state_word_e::acquisition_score,
             sbs_scene_controller::rule_state_word_e::acquisition_dwell_s,
             sbs_scene_controller::rule_state_word_e::acquisition_layout,
             sbs_scene_controller::rule_state_word_e::acquisition_valid,
           }) {
        EXPECT_FLOAT_EQ(word_as_float(result.state, word), 0.0f);
      }
      EXPECT_FLOAT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::acquisition_roi_x0
        ),
        0.0f
      );
      EXPECT_FLOAT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::acquisition_roi_y0
        ),
        0.0f
      );
      EXPECT_FLOAT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::acquisition_roi_x1
        ),
        1.0f
      );
      EXPECT_FLOAT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::acquisition_roi_y1
        ),
        1.0f
      );
    };
  const auto expect_challenger_canonical =
    [](const step_result_t &result) {
      for (const auto word : {
             sbs_scene_controller::rule_state_word_e::challenger_score,
             sbs_scene_controller::rule_state_word_e::challenger_dwell_s,
             sbs_scene_controller::rule_state_word_e::challenger_layout,
             sbs_scene_controller::rule_state_word_e::challenger_valid,
           }) {
        EXPECT_FLOAT_EQ(word_as_float(result.state, word), 0.0f);
      }
      EXPECT_FLOAT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::challenger_roi_x0
        ),
        0.0f
      );
      EXPECT_FLOAT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::challenger_roi_y0
        ),
        0.0f
      );
      EXPECT_FLOAT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::challenger_roi_x1
        ),
        1.0f
      );
      EXPECT_FLOAT_EQ(
        word_as_float(
          result.state,
          sbs_scene_controller::rule_state_word_e::challenger_roi_y1
        ),
        1.0f
      );
    };

  {
    SCOPED_TRACE("a state without the initialized identity bit cannot relock an ROI");
    auto state = baseline();
    force_locked_content_roi(state);
    auto &flags = state.state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )];
    flags &= ~sbs_scene_controller::state_flags_initialized;

    state = resolve_step(
      state.state,
      baseline_detector_count,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    const auto repaired_flags = state.state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )];
    EXPECT_NE(
      repaired_flags & sbs_scene_controller::state_flags_initialized,
      0u
    );
    EXPECT_EQ(
      repaired_flags & sbs_scene_controller::state_flags_roi_locked,
      0u
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        state.state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ),
      static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
    );
  }

  {
    SCOPED_TRACE("non-finite output validity retains reset debt fail closed");
    auto state = baseline();
    force_locked_content_roi(state);
    state.state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::output_valid
    )] = std::bit_cast<std::uint32_t>(
      std::numeric_limits<float>::quiet_NaN()
    );
    state.state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::reset_flags
    )] = sbs_scene_controller::reset_flags_geometry;

    state = resolve_step(
      state.state,
      baseline_detector_count,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    const auto repaired_flags = state.state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )];
    EXPECT_EQ(
      repaired_flags & sbs_scene_controller::state_flags_roi_locked,
      0u
    );
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::reset_flags
      )] & sbs_scene_controller::reset_flags_geometry,
      sbs_scene_controller::reset_flags_geometry
    );
  }

  {
    SCOPED_TRACE("an exact held-output zero preserves a validated committed role");
    auto state = baseline();
    force_locked_content_roi(state);
    set_state_word(
      state,
      sbs_scene_controller::rule_state_word_e::output_valid,
      0.0f
    );
    state = resolve_step(
      state.state,
      baseline_detector_count,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      false,
      false,
      false,
      1u
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        state.state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ),
      static_cast<float>(sbs_scene_controller::state_kind_e::content)
    );
    EXPECT_NE(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::state_flags
      )] & sbs_scene_controller::state_flags_roi_locked,
      0u
    );
  }

  for (const float malformed_output_valid : {
         0.01f,
         std::numeric_limits<float>::quiet_NaN(),
         1.01f,
       }) {
    SCOPED_TRACE("malformed held-output flag cannot preserve an ROI role");
    auto state = baseline();
    force_locked_content_roi(state);
    state.state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::output_valid
    )] = std::bit_cast<std::uint32_t>(malformed_output_valid);

    state = resolve_step(
      state.state,
      baseline_detector_count,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        state.state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ),
      static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
    );
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::state_flags
      )] & sbs_scene_controller::state_flags_roi_locked,
      0u
    );
  }

  for (const auto &[word, malformed] : {
         std::pair {
           sbs_scene_controller::rule_state_word_e::state_kind,
           2.5f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::state_kind,
           3.0f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::committed_layout,
           1.5f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::committed_layout,
           static_cast<float>(
             sbs_scene_controller::layout_decision_e::primary_video
           ),
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             committed_roi_confidence,
           0.0f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             committed_roi_confidence,
           1.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             committed_roi_confidence,
           std::numeric_limits<float>::quiet_NaN(),
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             committed_mask_confidence,
           -0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             committed_mask_confidence,
           1.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             committed_mask_confidence,
           std::numeric_limits<float>::quiet_NaN(),
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::committed_roi_x0,
           -0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::committed_roi_x0,
           0.75f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::committed_roi_x1,
           1.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::committed_roi_y0,
           std::numeric_limits<float>::quiet_NaN(),
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::committed_roi_age_s,
           -0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::committed_roi_age_s,
           std::numeric_limits<float>::quiet_NaN(),
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             seconds_since_layout_evidence,
           -0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             seconds_since_layout_evidence,
           std::numeric_limits<float>::quiet_NaN(),
         },
       }) {
    SCOPED_TRACE(
      "malformed committed role word " +
      std::to_string(sbs_scene_controller::index(word))
    );
    auto state = baseline();
    force_locked_content_roi(state);
    set_state_word(state, word, malformed);
    state = resolve_step(
      state.state,
      baseline_detector_count,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      false,
      false,
      false,
      1u
    );
    expect_full_frame(state);
  }

  const auto force_video_acquisition =
    [&](step_result_t &state) {
      for (const auto &[word, value] : {
             std::pair {
               sbs_scene_controller::rule_state_word_e::acquisition_roi_x0,
               0.20f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::acquisition_roi_y0,
               0.20f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::acquisition_roi_x1,
               0.80f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::acquisition_roi_y1,
               0.80f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::acquisition_score,
               0.0f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::
                 acquisition_dwell_s,
               0.20f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::acquisition_layout,
               static_cast<float>(
                 sbs_scene_controller::layout_decision_e::primary_video
               ),
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::acquisition_valid,
               1.0f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::
                 seconds_since_layout_evidence,
               0.0f,
             },
           }) {
        set_state_word(state, word, value);
      }
    };
  for (const auto &[word, malformed] : {
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_layout,
           1.5f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_valid,
           0.25f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_valid,
           2.0f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_score,
           0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_score,
           -0.0f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_score,
           1.0f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_score,
           std::numeric_limits<float>::quiet_NaN(),
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_dwell_s,
           -0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_dwell_s,
           std::numeric_limits<float>::quiet_NaN(),
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_roi_x0,
           -0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::acquisition_roi_x1,
           0.20f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
              seconds_since_layout_evidence,
            0.201f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             seconds_since_layout_evidence,
           -0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             seconds_since_layout_evidence,
           std::numeric_limits<float>::quiet_NaN(),
         },
       }) {
    SCOPED_TRACE(
      "malformed acquisition word " +
      std::to_string(sbs_scene_controller::index(word))
    );
    auto state = baseline();
    force_video_acquisition(state);
    set_state_word(state, word, malformed);
    state = resolve_step(
      state.state,
      baseline_detector_count,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      false,
      false,
      false,
      1u
    );
    expect_full_frame(state);
    expect_acquisition_canonical(state);
  }

  const auto force_video_challenger =
    [&](step_result_t &state) {
      force_locked_content_roi(state);
      for (const auto &[word, value] : {
             std::pair {
               sbs_scene_controller::rule_state_word_e::challenger_roi_x0,
               0.50f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::challenger_roi_y0,
               0.20f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::challenger_roi_x1,
               0.90f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::challenger_roi_y1,
               0.80f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::challenger_score,
               0.0f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::challenger_dwell_s,
               0.20f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::challenger_layout,
               static_cast<float>(
                 sbs_scene_controller::layout_decision_e::primary_video
               ),
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::challenger_valid,
               1.0f,
             },
             std::pair {
               sbs_scene_controller::rule_state_word_e::
                 seconds_since_layout_evidence,
               0.0f,
             },
           }) {
        set_state_word(state, word, value);
      }
    };
  for (const auto &[word, malformed] : {
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_layout,
           1.5f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_valid,
           0.25f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_valid,
           2.0f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_score,
           0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_score,
           -0.0f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_score,
           1.0f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_score,
           std::numeric_limits<float>::quiet_NaN(),
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_dwell_s,
           -0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_dwell_s,
           std::numeric_limits<float>::quiet_NaN(),
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_roi_x0,
           -0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::challenger_roi_x1,
           0.50f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
              seconds_since_layout_evidence,
            0.201f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             seconds_since_layout_evidence,
           -0.01f,
         },
         std::pair {
           sbs_scene_controller::rule_state_word_e::
             seconds_since_layout_evidence,
           std::numeric_limits<float>::quiet_NaN(),
         },
       }) {
    SCOPED_TRACE(
      "malformed challenger word " +
      std::to_string(sbs_scene_controller::index(word))
    );
    auto state = baseline();
    force_video_challenger(state);
    set_state_word(state, word, malformed);
    state = resolve_step(
      state.state,
      baseline_detector_count,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      false,
      false,
      false,
      1u
    );
    EXPECT_NE(
      word_as_float(
        state.state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ),
      static_cast<float>(sbs_scene_controller::state_kind_e::video)
    );
    expect_challenger_canonical(state);
  }

  {
    SCOPED_TRACE("scroll interruption clears a non-temporal challenger");
    auto state = baseline();
    force_locked_content_roi(state);
    for (const auto &[word, value] : {
           std::pair {
             sbs_scene_controller::rule_state_word_e::challenger_valid,
             1.0f,
           },
           std::pair {
             sbs_scene_controller::rule_state_word_e::challenger_dwell_s,
             0.29f,
           },
           std::pair {
             sbs_scene_controller::rule_state_word_e::challenger_score,
             0.0f,
           },
           std::pair {
             sbs_scene_controller::rule_state_word_e::challenger_layout,
             static_cast<float>(
               sbs_scene_controller::layout_decision_e::identity_fullscreen
             ),
           },
         }) {
      state.state[sbs_scene_controller::index(word)] =
        std::bit_cast<std::uint32_t>(value);
    }
    force_active_scroll_hold(state);

    state = resolve_step(
      state.state,
      baseline_detector_count,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      true,
      false,
      1u
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        state.state,
        sbs_scene_controller::rule_state_word_e::challenger_valid
      ),
      0.0f
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        state.state,
        sbs_scene_controller::rule_state_word_e::challenger_dwell_s
      ),
      0.0f
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        state.state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ),
      static_cast<float>(sbs_scene_controller::state_kind_e::content)
    );
    const auto flags = state.state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )];
    EXPECT_NE(
      flags & sbs_scene_controller::state_flags_scroll_hold_active,
      0u
    );
    EXPECT_NE(flags & sbs_scene_controller::state_flags_roi_locked, 0u);
  }

  {
    SCOPED_TRACE("valid local rejection consumes the durable edge");
    auto state = baseline();
    force_locked_content_roi(state);
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count + 1u
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count + 1u
    );
  }

  {
    SCOPED_TRACE(
      "full-frame scroll cuts are quarantined and consumed as ambiguous"
    );
    auto state = baseline();
    force_active_scroll_hold(state);
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::hard_cut,
      true,
      true,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 3u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::hard_cut,
      true,
      true,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count
    );

    state = retain_through_quiet_scroll_hold(
      std::move(state),
      baseline_detector_count + 3u,
      baseline_external_count,
      baseline_detector_count,
      false
    );
    // The first update after the quiet tail is a history-resync boundary.
    state = resolve_step(
      state.state,
      baseline_detector_count + 3u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count
    );

    // With no locked ROI attribution, the following quarantine update
    // consumes the coalesced count as ambiguous rather than pulsing late.
    state = resolve_step(
      state.state,
      baseline_detector_count + 3u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count + 3u
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 3u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
  }

  {
    SCOPED_TRACE(
      "locked-ROI corroboration remains durable under stronger scroll"
    );
    auto state = baseline();
    force_locked_content_roi(state);
    force_active_scroll_hold(state);
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::hard_cut,
      true,
      true,
      false,
      1u,
      0u,
      true
    );
    expect_shot(state, false);
    EXPECT_TRUE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count
    );

    state = retain_through_quiet_scroll_hold(
      std::move(state),
      baseline_detector_count + 1u,
      baseline_external_count,
      baseline_detector_count,
      true
    );
    // Release resynchronizes histories before a durable ROI-attributed edge
    // is allowed to act.
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_TRUE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, true);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count + 1u
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
  }

  {
    SCOPED_TRACE(
      "scroll-frozen reliable local rejection consumes immediately"
    );
    auto state = baseline();
    force_locked_content_roi(state);
    force_active_scroll_hold(state);
    reliable_roi_event_for_step = true;
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      true,
      false,
      1u
    );
    reliable_roi_event_for_step = false;
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count + 1u
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
  }

  {
    SCOPED_TRACE(
      "scroll-frozen insufficient ROI evidence retains the count"
    );
    auto state = baseline();
    force_locked_content_roi(state);
    force_active_scroll_hold(state);
    insufficient_roi_event_for_step = true;
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      true,
      false,
      1u
    );
    insufficient_roi_event_for_step = false;
    expect_shot(state, false);
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count
    );

    // Insufficient evidence cannot consume the count during the hold. The
    // release update first resynchronizes history; a following locked-ROI
    // corroboration may then fire exactly once.
    state = retain_through_quiet_scroll_hold(
      std::move(state),
      baseline_detector_count + 1u,
      baseline_external_count,
      baseline_detector_count,
      false
    );
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::hard_cut,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::hard_cut,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, true);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count + 1u
    );
  }

  {
    SCOPED_TRACE("invalid hold retains the latest edge");
    auto state = baseline();
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::hard_cut,
      false,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 3u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::hard_cut,
      false,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 3u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::hard_cut,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, true);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count + 3u
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 3u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
  }

  {
    SCOPED_TRACE("detector and external requests coalesce");
    auto state = baseline();
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count + 1u,
      sbs_scene_controller::event_decision_e::hard_cut,
      true,
      false,
      true,
      1u
    );
    expect_shot(state, true);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count + 1u
    );
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_external_cut_count
      )],
      baseline_external_count + 1u
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count + 1u,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
  }

  for (const auto reset_flags : {
         sbs_scene_controller::reset_flags_depth_shot,
         sbs_scene_controller::reset_flags_geometry,
       }) {
    SCOPED_TRACE(
      testing::Message()
        << "reset consumes detector and external deltas: "
        << reset_flags
    );
    auto state = baseline();
    force_locked_content_roi(state);
    force_active_scroll_hold(state);
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::hard_cut,
      true,
      true,
      false,
      1u
    );
    expect_shot(state, false);
    ASSERT_TRUE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 2u,
      baseline_external_count + 1u,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      true,
      true,
      1u,
      reset_flags
    );
    // A depth-only reset preserves the active scroll hold and consumes both
    // deltas without another pulse. A geometry reset necessarily clears that
    // hold, so the fresh external request remains authoritative.
    expect_shot(
      state,
      reset_flags == sbs_scene_controller::reset_flags_geometry
    );
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      baseline_detector_count + 2u
    );
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_external_cut_count
      )],
      baseline_external_count + 1u
    );

    state = resolve_step(
      state.state,
      baseline_detector_count + 2u,
      baseline_external_count + 1u,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      true,
      false,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
  }

  {
    SCOPED_TRACE("counter rollback starts a fresh detector/external epoch");
    constexpr std::uint32_t old_detector_count = 0xf0000010u;
    constexpr std::uint32_t old_external_count = 0xf0000020u;
    const std::vector<std::uint32_t> initial_state(
      sbs_scene_controller::initial_word_bits.begin(),
      sbs_scene_controller::initial_word_bits.end()
    );
    auto state = resolve_step(
      initial_state,
      old_detector_count,
      old_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      1u
    );
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      old_detector_count
    );
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_external_cut_count
      )],
      old_external_count
    );

    // A stale external request bit may overlap the adaptive-state rebuild.
    // The lower counters are baselines, not new edges.
    state = resolve_step(
      state.state,
      0u,
      0u,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      true,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      0u
    );
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_external_cut_count
      )],
      0u
    );

    state = resolve_step(
      state.state,
      1u,
      1u,
      sbs_scene_controller::event_decision_e::hard_cut,
      true,
      false,
      true,
      1u
    );
    expect_shot(state, true);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      1u
    );
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_external_cut_count
      )],
      1u
    );

    state = resolve_step(
      state.state,
      1u,
      1u,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      true,
      1u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
  }

  {
    SCOPED_TRACE("backend incompatibility baselines and clears pending");
    auto state = baseline();
    force_locked_content_roi(state);
    force_active_scroll_hold(state);
    state = resolve_step(
      state.state,
      baseline_detector_count + 1u,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::hard_cut,
      true,
      true,
      false,
      1u
    );
    ASSERT_TRUE(pending(state));

    constexpr std::uint32_t replacement_detector_count = 97u;
    state = resolve_step(
      state.state,
      replacement_detector_count,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      2u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          backend_generation
      )],
      2u
    );
    EXPECT_EQ(
      state.state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::
          last_detector_cut_count
      )],
      replacement_detector_count
    );

    state = resolve_step(
      state.state,
      replacement_detector_count,
      baseline_external_count,
      sbs_scene_controller::event_decision_e::same_shot,
      true,
      false,
      false,
      2u
    );
    expect_shot(state, false);
    EXPECT_FALSE(pending(state));
  }
}

TEST_F(
  SbsSceneControllerGpu,
  ConsumedDetectorCountIgnoresRepeatedStalePulse
) {
  constexpr std::uint32_t consumed_detector_count = 1u;
  const auto last_count_index = sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::last_detector_cut_count
  );
  const auto shot_was_promoted =
    [](const std::vector<std::uint32_t> &state) {
      return (
        state[sbs_scene_controller::index(
          sbs_scene_controller::rule_state_word_e::promotion_flags
        )] &
        sbs_scene_controller::promotion_flags_shot
      ) != 0u;
    };
  const auto event_is_hard_cut =
    [](const std::vector<std::uint32_t> &state) {
      return word_as_float(
               state,
               sbs_scene_controller::rule_state_word_e::event_decision
             ) ==
             static_cast<float>(
               sbs_scene_controller::event_decision_e::hard_cut
             );
    };

  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f,
    0u,
    false
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  ASSERT_TRUE(run_frame(320u, 180u, 174u, 0u));
  auto state = read_rule_state(controller->snapshot());
  EXPECT_EQ(state[last_count_index], 0u);
  EXPECT_FALSE(event_is_hard_cut(state));
  EXPECT_FALSE(shot_was_promoted(state));

  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f,
    consumed_detector_count,
    true
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  ASSERT_TRUE(run_frame(320u, 180u, 175u, 1u));
  state = read_rule_state(controller->snapshot());
  EXPECT_EQ(state[last_count_index], consumed_detector_count);
  EXPECT_TRUE(event_is_hard_cut(state));
  EXPECT_TRUE(shot_was_promoted(state));

  // Keep both the one-frame pulse and a large appearance change present. Once exact counter
  // state is initialized, the unchanged durable count is authoritative and the stale pulse
  // cannot create a second shot.
  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f,
    consumed_detector_count,
    true
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  ASSERT_TRUE(run_frame(320u, 180u, 176u, 0u));
  state = read_rule_state(controller->snapshot());
  EXPECT_EQ(state[last_count_index], consumed_detector_count);
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::event_decision
    ),
    static_cast<float>(
      sbs_scene_controller::event_decision_e::same_shot
    )
  );
  EXPECT_FALSE(shot_was_promoted(state));
}

TEST_F(
  SbsSceneControllerGpu,
  PortraitAndUltrawideUseAspectPreservingCanonicalPadding
) {
  ASSERT_TRUE(run_frame(288, 512, 100));
  const auto portrait_snapshot = controller->snapshot();
  EXPECT_EQ(portrait_snapshot.source_frame_id, 100u);
  expect_canonical_viewport(
    read_analysis(portrait_snapshot),
    28,
    0,
    72,
    128
  );

  ASSERT_TRUE(run_frame(640, 180, 101));
  const auto ultrawide_snapshot = controller->snapshot();
  EXPECT_EQ(ultrawide_snapshot.source_frame_id, 101u);
  expect_canonical_viewport(
    read_analysis(ultrawide_snapshot),
    0,
    46,
    128,
    36
  );
  expect_snapshot_abi_invariants(
    device.Get(),
    context.Get(),
    ultrawide_snapshot
  );
}

TEST_F(
  SbsSceneControllerGpu,
  CompletedSnapshotSurvivesPreparingAndEnqueuingTheNextFrame
) {
  constexpr std::size_t appearance_value_count =
    3u *
    sbs_scene_controller::appearance_canvas_size *
    sbs_scene_controller::appearance_canvas_size;
  constexpr std::size_t analysis_value_count =
    sbs_scene_controller::analysis_grid_channel_count *
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  constexpr std::uint64_t completed_frame_id = 150;
  constexpr std::uint64_t pending_frame_id = 151;

  ASSERT_TRUE(run_frame(320, 180, completed_frame_id, 0));
  const auto completed_snapshot = controller->snapshot();
  ASSERT_EQ(completed_snapshot.source_frame_id, completed_frame_id);
  const auto completed_scene = read_buffer<float>(
    device.Get(),
    context.Get(),
    completed_snapshot.scene_rgb.Get(),
    appearance_value_count
  );
  const auto completed_analysis = read_buffer<float>(
    device.Get(),
    context.Get(),
    completed_snapshot.analysis_grid.Get(),
    analysis_value_count
  );

  const auto next_source = make_source_view(device.Get(), 320, 180, 1);
  ASSERT_TRUE(next_source);
  ASSERT_TRUE(controller->prepare_scene(
    next_source.Get(),
    models::input_color_space::srgb,
    pending_frame_id
  ));
  controller->mark_enqueued(pending_frame_id);

  const auto while_pending_snapshot = controller->snapshot();
  EXPECT_TRUE(while_pending_snapshot.snapshot_available);
  EXPECT_EQ(
    while_pending_snapshot.source_frame_id,
    completed_frame_id
  );
  const auto scene_while_pending = read_buffer<float>(
    device.Get(),
    context.Get(),
    while_pending_snapshot.scene_rgb.Get(),
    appearance_value_count
  );
  const auto analysis_while_pending = read_buffer<float>(
    device.Get(),
    context.Get(),
    while_pending_snapshot.analysis_grid.Get(),
    analysis_value_count
  );
  EXPECT_TRUE(
    std::equal(
      completed_scene.begin(),
      completed_scene.end(),
      scene_while_pending.begin()
    )
  );
  EXPECT_TRUE(
    std::equal(
      completed_analysis.begin(),
      completed_analysis.end(),
      analysis_while_pending.begin()
    )
  );

  ASSERT_TRUE(controller->resolve_completed(
    pending_frame_id,
    raw_depth_buffer.srv.Get(),
    depth_view.Get(),
    frame_state_buffer.srv.Get(),
    adaptive_buffer.srv.Get(),
    depth_width,
    depth_height,
    frame_roi_transform_buffer.srv.Get()
  ));
  const auto next_snapshot = controller->snapshot();
  EXPECT_EQ(next_snapshot.source_frame_id, pending_frame_id);
  const auto next_scene = read_buffer<float>(
    device.Get(),
    context.Get(),
    next_snapshot.scene_rgb.Get(),
    appearance_value_count
  );
  EXPECT_FALSE(
    std::equal(
      completed_scene.begin(),
      completed_scene.end(),
      next_scene.begin()
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  RoiAcquisitionPreservesEveryUintBitStateField
) {
  const auto first_source = make_content_source_view(
    device.Get(),
    320,
    180
  );
  ASSERT_TRUE(first_source);
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(first_source.Get(), 180));

  const auto first_snapshot = controller->snapshot();
  const auto first_state = read_rule_state(first_snapshot);
  expect_roi_coverage(
    read_analysis(first_snapshot),
    false,
    {0.0f, 0.0f, 1.0f, 1.0f}
  );
  EXPECT_EQ(
    first_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::backend_generation
    )],
    1u
  );
  EXPECT_EQ(
    first_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    0u
  );
  EXPECT_EQ(
    first_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::update_count
    )],
    1u
  );
  EXPECT_EQ(
    first_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )],
    sbs_scene_controller::state_flags_initialized |
      sbs_scene_controller::state_flags_layout_history_valid |
      sbs_scene_controller::state_flags_depth_history_valid
  );
  EXPECT_EQ(
    first_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::reset_flags
    )],
    sbs_scene_controller::reset_flags_layout |
      sbs_scene_controller::reset_flags_depth_shot |
      sbs_scene_controller::reset_flags_backend
  );
  EXPECT_EQ(
    first_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::promotion_flags
    )],
    sbs_scene_controller::promotion_flags_layout_history |
      sbs_scene_controller::promotion_flags_depth_history
  );
  EXPECT_EQ(
    first_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::history_flags
    )],
    sbs_scene_controller::history_flags_layout_read_bank |
      sbs_scene_controller::history_flags_layout_write_bank |
      sbs_scene_controller::history_flags_depth_read_bank |
      sbs_scene_controller::history_flags_depth_write_bank
  );
  EXPECT_EQ(
    first_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::diagnostic_flags
    )],
    0u
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      first_state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      first_state,
      sbs_scene_controller::rule_state_word_e::acquisition_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::content_collage
    )
  );

  const auto second_source = make_content_source_view(
    device.Get(),
    320,
    180
  );
  ASSERT_TRUE(second_source);
  controller->set_next_elapsed_seconds_for_testing(0.71f);
  ASSERT_TRUE(run_source(second_source.Get(), 181));

  const auto acquired_state = read_rule_state(controller->snapshot());
  EXPECT_EQ(
    acquired_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::backend_generation
    )],
    1u
  );
  EXPECT_EQ(
    acquired_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    1u
  );
  EXPECT_EQ(
    acquired_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::update_count
    )],
    2u
  );
  EXPECT_EQ(
    acquired_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )],
    sbs_scene_controller::state_flags_initialized |
      sbs_scene_controller::state_flags_roi_locked |
      sbs_scene_controller::state_flags_layout_history_valid |
      sbs_scene_controller::state_flags_depth_history_valid
  );
  EXPECT_EQ(
    acquired_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::reset_flags
    )],
    0u
  );
  EXPECT_EQ(
    acquired_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::promotion_flags
    )],
    sbs_scene_controller::promotion_flags_layout_history |
      sbs_scene_controller::promotion_flags_depth_history |
      sbs_scene_controller::promotion_flags_roi
  );
  EXPECT_EQ(
    acquired_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::history_flags
    )],
    sbs_scene_controller::history_flags_layout_read_bank |
      sbs_scene_controller::history_flags_layout_write_bank |
      sbs_scene_controller::history_flags_depth_read_bank |
      sbs_scene_controller::history_flags_depth_write_bank
  );
  EXPECT_EQ(
    acquired_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::diagnostic_flags
    )],
    0u
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      acquired_state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::content
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      acquired_state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::content_collage
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      acquired_state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );

  const std::array<float, 4> committed_roi {
    word_as_float(
      acquired_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    ),
    word_as_float(
      acquired_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    ),
    word_as_float(
      acquired_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ),
    word_as_float(
      acquired_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    ),
  };
  const auto third_source = make_content_source_view(
    device.Get(),
    320,
    180
  );
  ASSERT_TRUE(third_source);
  controller->set_next_elapsed_seconds_for_testing(1.0f / 60.0f);
  ASSERT_TRUE(run_source(third_source.Get(), 182));
  expect_roi_coverage(
    read_analysis(controller->snapshot()),
    true,
    committed_roi
  );
}

TEST_F(
  SbsSceneControllerGpu,
  DepthConfidenceAggregateUsesCommittedRoiCoverage
) {
  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f
  );
  ASSERT_TRUE(adaptive_buffer.srv);

  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> partial_raw_depth(
    depth_width * depth_height,
    nan
  );
  for (UINT y = 0; y < depth_height; ++y) {
    for (UINT x = 0; x < depth_width; ++x) {
      const float viewport_x =
        (static_cast<float>(x) + 0.5f) /
        static_cast<float>(depth_width);
      const float viewport_y =
        (static_cast<float>(y) + 0.5f) /
        static_cast<float>(depth_height);
      if (
        viewport_x >= 0.20f && viewport_x < 0.80f &&
        viewport_y >= 0.20f && viewport_y < 0.50f
      ) {
        partial_raw_depth[y * depth_width + x] = 0.5f;
      }
    }
  }
  raw_depth_buffer = make_structured_buffer(
    device.Get(),
    partial_raw_depth.data(),
    static_cast<UINT>(partial_raw_depth.size() * sizeof(float)),
    sizeof(float)
  );
  ASSERT_TRUE(raw_depth_buffer.srv);

  const auto source = make_content_source_view(
    device.Get(),
    320,
    180
  );
  ASSERT_TRUE(source);
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(source.Get(), 210));
  controller->set_next_elapsed_seconds_for_testing(0.71f);
  ASSERT_TRUE(run_source(source.Get(), 211));
  const auto acquired_snapshot = controller->snapshot();
  const auto acquired_state = read_rule_state(acquired_snapshot);
  ASSERT_NE(
    acquired_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] &
      sbs_scene_controller::state_flags_roi_locked,
    0u
  );
  const auto acquired_global = read_buffer<float>(
    device.Get(),
    context.Get(),
    acquired_snapshot.global_output.Get(),
    sbs_scene_controller::global_out_word_count
  );
  const auto confidence_index = static_cast<std::size_t>(
    sbs_scene_controller::global_out_word_e::confidence_pop
  );

  controller->set_next_elapsed_seconds_for_testing(1.0f / 60.0f);
  ASSERT_TRUE(run_source(source.Get(), 212));
  const auto locked_snapshot = controller->snapshot();
  const auto locked_global = read_buffer<float>(
    device.Get(),
    context.Get(),
    locked_snapshot.global_output.Get(),
    sbs_scene_controller::global_out_word_count
  );
  EXPECT_LT(acquired_global[confidence_index], 0.80f);
  EXPECT_GT(locked_global[confidence_index], 0.88f);
  EXPECT_GT(
    locked_global[confidence_index] -
      acquired_global[confidence_index],
    0.20f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  RoiCoverageRetainsDepthShotButDropsLayoutInvalidatingReset
) {
  const auto source = make_content_source_view(
    device.Get(),
    320,
    180
  );
  ASSERT_TRUE(source);
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(source.Get(), 220));
  controller->set_next_elapsed_seconds_for_testing(0.71f);
  ASSERT_TRUE(run_source(source.Get(), 221));
  const auto locked_snapshot = controller->snapshot();
  const auto locked_state = read_rule_state(locked_snapshot);
  ASSERT_NE(
    locked_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_roi_locked,
    0u
  );
  const std::array<float, 4> committed_roi {
    word_as_float(
      locked_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    ),
    word_as_float(
      locked_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    ),
    word_as_float(
      locked_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ),
    word_as_float(
      locked_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    ),
  };

  controller->set_next_reset_flags_for_testing(
    sbs_scene_controller::reset_flags_depth_shot
  );
  controller->set_next_elapsed_seconds_for_testing(1.0f / 60.0f);
  ASSERT_TRUE(run_source(source.Get(), 222));
  expect_roi_coverage(
    read_analysis(controller->snapshot()),
    true,
    committed_roi
  );

  const auto geometry_source = make_content_source_view(
    device.Get(),
    640,
    180
  );
  ASSERT_TRUE(geometry_source);
  controller->set_next_elapsed_seconds_for_testing(1.0f / 60.0f);
  ASSERT_TRUE(run_source(geometry_source.Get(), 223));
  expect_roi_coverage(
    read_analysis(controller->snapshot()),
    false,
    {}
  );
}

TEST_F(
  SbsSceneControllerGpu,
  GeometryResetIsDeferredUntilTheMatchingCompletion
) {
  constexpr std::uint64_t baseline_frame_id = 190;
  constexpr std::uint64_t geometry_frame_id = 191;
  ASSERT_TRUE(run_frame(320, 180, baseline_frame_id));
  const auto baseline_snapshot = controller->snapshot();
  const auto baseline_state = read_rule_state(baseline_snapshot);

  const auto geometry_source = make_source_view(
    device.Get(),
    640,
    180
  );
  ASSERT_TRUE(geometry_source);
  ASSERT_TRUE(controller->prepare_scene(
    geometry_source.Get(),
    models::input_color_space::srgb,
    geometry_frame_id
  ));

  auto deferred_snapshot = controller->snapshot();
  EXPECT_EQ(
    deferred_snapshot.source_frame_id,
    baseline_frame_id
  );
  EXPECT_EQ(read_rule_state(deferred_snapshot), baseline_state);

  controller->mark_enqueued(geometry_frame_id);
  deferred_snapshot = controller->snapshot();
  EXPECT_EQ(
    deferred_snapshot.source_frame_id,
    baseline_frame_id
  );
  EXPECT_EQ(read_rule_state(deferred_snapshot), baseline_state);

  ASSERT_TRUE(resolve_pending(geometry_frame_id));
  const auto resolved_snapshot = controller->snapshot();
  EXPECT_EQ(
    resolved_snapshot.source_frame_id,
    geometry_frame_id
  );
  const auto resolved_state = read_rule_state(resolved_snapshot);
  EXPECT_EQ(
    resolved_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::backend_generation
    )],
    1u
  );
  EXPECT_EQ(
    resolved_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    0u
  );
  EXPECT_EQ(
    resolved_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::update_count
    )],
    1u
  );
  EXPECT_EQ(
    resolved_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::reset_flags
    )],
    sbs_scene_controller::reset_flags_layout |
      sbs_scene_controller::reset_flags_depth_shot |
      sbs_scene_controller::reset_flags_geometry |
      sbs_scene_controller::reset_flags_display_or_hdr
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      resolved_state,
      sbs_scene_controller::rule_state_word_e::event_decision
    ),
    static_cast<float>(
      sbs_scene_controller::event_decision_e::geometry_reset
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  NonFiniteComputedSceneFailsClosedAndFreezesPromotedHistories
) {
  constexpr std::size_t analysis_pixels =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  ASSERT_TRUE(run_frame(320, 180, 195));
  const auto baseline_snapshot = controller->snapshot();
  const auto baseline_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    baseline_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  const auto baseline_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    baseline_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );

  const auto nan_source = make_nan_source_view(
    device.Get(),
    320,
    180
  );
  ASSERT_TRUE(nan_source);
  ASSERT_TRUE(run_source(nan_source.Get(), 196));
  const auto invalid_snapshot = controller->snapshot();
  const auto invalid_state = read_rule_state(invalid_snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      invalid_state,
      sbs_scene_controller::rule_state_word_e::output_valid
    ),
    0.0f
  );
  EXPECT_EQ(
    invalid_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::update_count
    )],
    2u
  );
  EXPECT_EQ(
    invalid_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )],
    sbs_scene_controller::state_flags_initialized |
      sbs_scene_controller::state_flags_layout_history_valid |
      sbs_scene_controller::state_flags_depth_history_valid |
      sbs_scene_controller::state_flags_fallback_active
  );
  EXPECT_EQ(
    invalid_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::promotion_flags
    )],
    0u
  );
  EXPECT_EQ(
    invalid_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::history_flags
    )],
    sbs_scene_controller::history_flags_layout_read_bank |
      sbs_scene_controller::history_flags_depth_read_bank
  );
  EXPECT_EQ(
    invalid_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::diagnostic_flags
    )],
    sbs_scene_controller::diagnostic_flags_ood |
      sbs_scene_controller::diagnostic_flags_depth_invalid
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      invalid_state,
      sbs_scene_controller::rule_state_word_e::rejection_reason
    ),
    static_cast<float>(
      sbs_scene_controller::rejection_reason_e::invalid_input
    )
  );

  const auto invalid_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    invalid_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  const auto invalid_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    invalid_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );
  EXPECT_EQ(invalid_layout, baseline_layout);
  EXPECT_EQ(invalid_depth, baseline_depth);

  expect_all_finite(
    device.Get(),
    context.Get(),
    invalid_snapshot.global_output.Get(),
    sbs_scene_controller::global_out_word_count,
    "invalid global output"
  );
  const auto global = read_buffer<float>(
    device.Get(),
    context.Get(),
    invalid_snapshot.global_output.Get(),
    sbs_scene_controller::global_out_word_count
  );
  EXPECT_FLOAT_EQ(
    global[static_cast<std::size_t>(
      sbs_scene_controller::global_out_word_e::backend_output_valid
    )],
    0.0f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  NonFiniteRetainedHistoryTripsEvidenceSentinelAndFailsClosed
) {
  constexpr std::size_t analysis_pixels =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  ASSERT_TRUE(run_frame(320, 180, 199));
  const auto baseline = controller->snapshot();
  ASSERT_TRUE(baseline.layout_history);

  auto retained_history = read_buffer<float>(
    device.Get(),
    context.Get(),
    baseline.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  const std::size_t corrupted_cell = analysis_pixels / 2u;
  const std::size_t corrupted_index =
    static_cast<std::size_t>(
      sbs_scene_controller::layout_history_channel_e::slow_activity_ema
    ) *
      analysis_pixels +
    corrupted_cell;
  retained_history[corrupted_index] =
    std::numeric_limits<float>::quiet_NaN();

  ComPtr<ID3D11Resource> history_resource;
  baseline.layout_history->GetResource(&history_resource);
  ASSERT_TRUE(history_resource);
  context->UpdateSubresource(
    history_resource.Get(),
    0u,
    nullptr,
    retained_history.data(),
    0u,
    0u
  );

  ASSERT_TRUE(run_frame(320, 180, 200));
  const auto invalid = controller->snapshot();
  const auto state = read_rule_state(invalid);
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::output_valid
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::promotion_flags
    )],
    0u
  );

  const auto dense = read_buffer<float>(
    device.Get(),
    context.Get(),
    invalid.dense_output.Get(),
    sbs_scene_controller::dense_out_channel_count *
      analysis_pixels
  );
  const std::size_t sentinel_index =
    static_cast<std::size_t>(
      sbs_scene_controller::dense_out_channel_e::unknown_background
    ) *
      analysis_pixels +
    corrupted_cell;
  EXPECT_FLOAT_EQ(dense[sentinel_index], -1.0f);

  const auto global = read_buffer<float>(
    device.Get(),
    context.Get(),
    invalid.global_output.Get(),
    sbs_scene_controller::global_out_word_count
  );
  EXPECT_FLOAT_EQ(
    global[static_cast<std::size_t>(
      sbs_scene_controller::global_out_word_e::backend_output_valid
    )],
    0.0f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  NonFiniteDepthIsPartialAndDegradesOnlyDepthConfidence
) {
  constexpr std::size_t analysis_pixels =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  ASSERT_TRUE(run_frame(320, 180, 197));
  const auto baseline_snapshot = controller->snapshot();
  const auto baseline_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    baseline_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );

  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  std::vector<float> nan_depth(
    depth_width * depth_height,
    std::numeric_limits<float>::quiet_NaN()
  );
  raw_depth_buffer = make_structured_buffer(
    device.Get(),
    nan_depth.data(),
    static_cast<UINT>(nan_depth.size() * sizeof(float)),
    sizeof(float)
  );
  ASSERT_TRUE(raw_depth_buffer.srv);
  ASSERT_TRUE(run_frame(320, 180, 198));

  const auto partial_snapshot = controller->snapshot();
  const auto partial_state = read_rule_state(partial_snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      partial_state,
      sbs_scene_controller::rule_state_word_e::output_valid
    ),
    1.0f
  );
  EXPECT_EQ(
    partial_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )],
    sbs_scene_controller::state_flags_initialized |
      sbs_scene_controller::state_flags_layout_history_valid |
      sbs_scene_controller::state_flags_depth_history_valid
  );
  EXPECT_EQ(
    partial_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::promotion_flags
    )],
    sbs_scene_controller::promotion_flags_layout_history |
      sbs_scene_controller::promotion_flags_depth_history
  );
  EXPECT_EQ(
    partial_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::history_flags
    )],
    sbs_scene_controller::history_flags_layout_read_bank |
      sbs_scene_controller::history_flags_layout_write_bank |
      sbs_scene_controller::history_flags_depth_read_bank |
      sbs_scene_controller::history_flags_depth_write_bank
  );

  const auto partial_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    partial_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );
  const auto last_depth_offset = static_cast<std::size_t>(
                                   sbs_scene_controller::depth_history_channel_e::
                                     last_normalized_depth
                                 ) *
                                 analysis_pixels;
  EXPECT_TRUE(std::equal(partial_depth.begin() + last_depth_offset, partial_depth.begin() + last_depth_offset + analysis_pixels, baseline_depth.begin() + last_depth_offset));

  const auto confidence_offset = static_cast<std::size_t>(
                                   sbs_scene_controller::depth_history_channel_e::
                                     valid_depth_confidence
                                 ) *
                                 analysis_pixels;
  EXPECT_GT(
    std::count_if(
      partial_depth.begin() + confidence_offset,
      partial_depth.begin() + confidence_offset + analysis_pixels,
      [&, index = confidence_offset](const float value) mutable {
        return value < baseline_depth[index++];
      }
    ),
    0
  );
  const auto age_offset = static_cast<std::size_t>(
                            sbs_scene_controller::depth_history_channel_e::
                              seconds_since_valid_depth
                          ) *
                          analysis_pixels;
  EXPECT_GT(
    std::count_if(
      partial_depth.begin() + age_offset,
      partial_depth.begin() + age_offset + analysis_pixels,
      [&, index = age_offset](const float value) mutable {
        return value > baseline_depth[index++];
      }
    ),
    0
  );
  EXPECT_EQ(
    std::count_if(
      partial_depth.begin(),
      partial_depth.end(),
      [](const float value) {
        return !std::isfinite(value);
      }
    ),
    0
  );

  const auto global = read_buffer<float>(
    device.Get(),
    context.Get(),
    partial_snapshot.global_output.Get(),
    sbs_scene_controller::global_out_word_count
  );
  EXPECT_FLOAT_EQ(
    global[static_cast<std::size_t>(
      sbs_scene_controller::global_out_word_e::backend_output_valid
    )],
    1.0f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  GlobalScrollConsensusFindsOneSharedShiftAcrossLocallyAmbiguousBands
) {
  constexpr UINT width = 128u;
  constexpr UINT height = 128u;
  const auto baseline = make_repetitive_page_scroll_source_view(
    device.Get(),
    width,
    height,
    0u
  );
  const auto translated = make_repetitive_page_scroll_source_view(
    device.Get(),
    width,
    height,
    1u
  );
  ASSERT_TRUE(baseline);
  ASSERT_TRUE(translated);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(baseline.Get(), 780u));
  controller->set_next_elapsed_seconds_for_testing(0.08f);
  ASSERT_TRUE(run_source(translated.Get(), 781u));

  const auto snapshot = controller->snapshot();
  const auto state = read_rule_state(snapshot);

  // Two-row bands tie at several odd shifts and three-row bands tie at
  // different shifts, while +1 is the only displacement shared by the whole
  // page. The simplified rule exposes only the resulting global consensus.
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.82f
  );
  EXPECT_GT(
    std::abs(word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_y
    )),
    0.5f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  TinyMovingWidgetCannotTurnStaticRepetitivePageIntoGlobalScroll
) {
  constexpr UINT width = 128u;
  constexpr UINT height = 128u;
  const auto baseline = make_repetitive_page_scroll_source_view(
    device.Get(),
    width,
    height,
    0u,
    true,
    44u
  );
  const auto moved_widget = make_repetitive_page_scroll_source_view(
    device.Get(),
    width,
    height,
    0u,
    true,
    46u
  );
  ASSERT_TRUE(baseline);
  ASSERT_TRUE(moved_widget);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(baseline.Get(), 782u));
  controller->set_next_elapsed_seconds_for_testing(0.08f);
  ASSERT_TRUE(run_source(moved_widget.Get(), 783u));

  const auto snapshot = controller->snapshot();
  const auto state = read_rule_state(snapshot);

  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] &
      sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
}

TEST_F(
  SbsSceneControllerGpu,
  ScrollHoldEntryUsesCapturedStreamTimeAt72And90Hz
) {
  struct cadence_case_t {
    std::uint32_t frames_per_second;
    std::uint32_t consistent_updates_to_enter;
  };
  constexpr std::array cadence_cases {
    cadence_case_t {72u, 4u},
    cadence_case_t {90u, 5u},
  };
  std::array<float, cadence_cases.size()> entry_seconds {};
  std::uint64_t frame_id = 784u;

  for (std::size_t case_index = 0u;
       case_index < cadence_cases.size();
       ++case_index) {
    const auto &cadence = cadence_cases[case_index];
    SCOPED_TRACE(cadence.frames_per_second);
    const float frame_seconds =
      1.0f / static_cast<float>(cadence.frames_per_second);
    const auto baseline =
      make_scroll_source_view(device.Get(), 128u, 128u, 0u);
    const auto first_observation =
      make_scroll_source_view(device.Get(), 128u, 128u, 1u);
    ASSERT_TRUE(baseline);
    ASSERT_TRUE(first_observation);

    controller->set_next_reset_flags_for_testing(
      sbs_scene_controller::reset_flags_geometry
    );
    controller->set_next_elapsed_seconds_for_testing(0.0f);
    ASSERT_TRUE(run_source(baseline.Get(), frame_id++));
    controller->set_next_elapsed_seconds_for_testing(frame_seconds);
    ASSERT_TRUE(run_source(first_observation.Get(), frame_id++));

    auto state = read_rule_state(controller->snapshot());
    ASSERT_GT(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::scroll_confidence
      ),
      0.82f
    );
    ASSERT_GT(
      std::abs(word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::scroll_y
      )),
      0.5f
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::scroll_dwell_s
      ),
      0.0f
    );
    EXPECT_EQ(
      state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::state_flags
      )] & sbs_scene_controller::state_flags_scroll_hold_active,
      0u
    );

    for (std::uint32_t consistent_update = 1u;
         consistent_update <= cadence.consistent_updates_to_enter;
         ++consistent_update) {
      const auto translated = make_scroll_source_view(
        device.Get(),
        128u,
        128u,
        1u + consistent_update
      );
      ASSERT_TRUE(translated);
      controller->set_next_elapsed_seconds_for_testing(frame_seconds);
      ASSERT_TRUE(run_source(translated.Get(), frame_id++));
      state = read_rule_state(controller->snapshot());

      const float accumulated_seconds =
        static_cast<float>(consistent_update) * frame_seconds;
      EXPECT_NEAR(
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::scroll_dwell_s
        ),
        accumulated_seconds,
        1e-5f
      );
      const bool hold_active =
        (
          state[sbs_scene_controller::index(
            sbs_scene_controller::rule_state_word_e::state_flags
          )] &
          sbs_scene_controller::state_flags_scroll_hold_active
        ) != 0u;
      EXPECT_EQ(
        hold_active,
        consistent_update ==
          cadence.consistent_updates_to_enter
      );
    }

    entry_seconds[case_index] =
      cadence.consistent_updates_to_enter * frame_seconds;
    EXPECT_GE(entry_seconds[case_index], 0.05f);
    EXPECT_LT(entry_seconds[case_index], 0.05f + frame_seconds);
  }

  EXPECT_NEAR(entry_seconds[0], entry_seconds[1], 1e-5f);
}

TEST_F(
  SbsSceneControllerGpu,
  CommittedContentRoiDoesNotSuppressCoherentPageScroll
) {
  const auto baseline = make_scrolling_content_source_view(
    device.Get(),
    128,
    128,
    0u
  );
  const auto translated = make_scrolling_content_source_view(
    device.Get(),
    128,
    128,
    1u
  );
  const auto translated_again = make_scrolling_content_source_view(
    device.Get(),
    128,
    128,
    2u
  );
  ASSERT_TRUE(baseline);
  ASSERT_TRUE(translated);
  ASSERT_TRUE(translated_again);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(baseline.Get(), 790u));
  controller->set_next_elapsed_seconds_for_testing(0.71f);
  ASSERT_TRUE(run_source(baseline.Get(), 791u));
  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::content_collage
    )
  );

  // A coherent translation inside committed content is page scroll evidence.
  // Only a committed primary-video ROI may exclude its local motion vote.
  controller->set_next_elapsed_seconds_for_testing(0.08f);
  ASSERT_TRUE(run_source(translated.Get(), 792u));
  state = read_rule_state(controller->snapshot());
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::content
    )
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] &
      sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );

  // The first observation establishes direction; only another consistent
  // 50 ms of captured-stream evidence may latch the safety hold.
  controller->set_next_elapsed_seconds_for_testing(0.05f);
  ASSERT_TRUE(run_source(translated_again.Get(), 793u));
  state = read_rule_state(controller->snapshot());
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_NE(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] &
      sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
}

TEST_F(
  SbsSceneControllerGpu,
  LargeEmbeddedVerticalPanAbstainsAfterConsistentScrollEvidence
) {
  constexpr UINT width = 320u;
  constexpr UINT height = 180u;
  constexpr UINT player_inset = width / 5u;
  std::uint64_t frame_id = 980u;
  const auto run_pan =
    [&](const UINT player_shift, const float elapsed_seconds) {
      const auto source =
        make_large_player_with_scrolling_exterior_source_view(
          device.Get(),
          width,
          height,
          0u,
          0u,
          player_shift,
          player_inset
        );
      EXPECT_TRUE(source);
      if (!source) {
        return false;
      }
      controller->set_next_elapsed_seconds_for_testing(elapsed_seconds);
      return run_source(source.Get(), frame_id++);
    };

  ASSERT_TRUE(run_pan(0u, 0.0f));
  // Five input pixels are exactly two rows on the 128x72 analysis viewport. The fixture's
  // per-pixel random texture has no spatial correlation, so non-integral grid shifts would
  // measure resampling alias rather than the coherent-motion policy under test.
  ASSERT_TRUE(run_pan(5u, 0.10f));
  auto state = read_rule_state(controller->snapshot());
  ASSERT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  ) << "the current moving-video mask must remain eligible for raw scroll voting";
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
  );
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] &
      sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );

  ASSERT_TRUE(run_pan(10u, 0.05f));
  state = read_rule_state(controller->snapshot());
  ASSERT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_NE(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] &
      sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );

  ASSERT_TRUE(run_pan(15u, 0.25f));
  ASSERT_TRUE(run_pan(20u, 0.25f));
  ASSERT_TRUE(run_pan(25u, 0.25f));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
  );
}

TEST_F(
  SbsSceneControllerGpu,
  ExteriorPageScrollCancelsProvisionalPlayerWithoutCreatingRoiOwnership
) {
  constexpr UINT width = 320u;
  constexpr UINT height = 180u;
  // Keep the exterior below the full-viewport scroll threshold. The transition is observable
  // only when the numerator and eligible-area denominator exclude the same provisional player.
  // Keep the player safely outside the 8%-edge identity-fullscreen band after analysis-grid
  // quantization; this fixture is testing provisional ROI ownership, not identity suppression.
  constexpr UINT player_inset = width / 10u;
  std::uint64_t frame_id = 990u;
  const auto run_frame_with_motion =
    [&](const std::uint32_t player_phase,
        const UINT player_shift,
        const UINT exterior_shift,
        const float elapsed_seconds) {
      const auto source =
        make_large_player_with_scrolling_exterior_source_view(
          device.Get(),
          width,
          height,
          player_phase,
          exterior_shift,
          player_shift,
          player_inset,
          0,
          0,
          height / 10u
        );
      EXPECT_TRUE(source);
      if (!source) {
        return false;
      }
      controller->set_next_elapsed_seconds_for_testing(elapsed_seconds);
      return run_source(source.Get(), frame_id++);
    };

  ASSERT_TRUE(run_frame_with_motion(0u, 0u, 0u, 0.0f));
  ASSERT_TRUE(run_frame_with_motion(1u, 0u, 0u, 0.10f));
  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );

  ASSERT_TRUE(run_frame_with_motion(1u, 5u, 5u, 0.08f));
  state = read_rule_state(controller->snapshot());
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] &
      sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );

  ASSERT_TRUE(run_frame_with_motion(1u, 10u, 10u, 0.05f));
  state = read_rule_state(controller->snapshot());
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
  );
  const auto flags = state[sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::state_flags
  )];
  EXPECT_NE(
    flags & sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
  EXPECT_EQ(flags & sbs_scene_controller::state_flags_roi_locked, 0u);
}

TEST_F(
  SbsSceneControllerGpu,
  ScrollHoldFreezesHistoriesUntilTerminationEvidenceMatures
) {
  constexpr std::size_t analysis_pixels =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  const auto baseline_source = make_scroll_source_view(
    device.Get(),
    128,
    128,
    0
  );
  ASSERT_TRUE(baseline_source);
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(baseline_source.Get(), 200));
  const auto baseline_snapshot = controller->snapshot();
  const auto baseline_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    baseline_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  const auto first_scrolled_source = make_scroll_source_view(
    device.Get(),
    128,
    128,
    1
  );
  const auto held_source = make_scroll_source_view(
    device.Get(),
    128,
    128,
    2
  );
  ASSERT_TRUE(first_scrolled_source);
  ASSERT_TRUE(held_source);
  controller->set_next_elapsed_seconds_for_testing(0.08f);
  ASSERT_TRUE(run_source(first_scrolled_source.Get(), 201));
  const auto first_evidence_snapshot = controller->snapshot();
  const auto first_evidence_state =
    read_rule_state(first_evidence_snapshot);
  ASSERT_GT(
    word_as_float(
      first_evidence_state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.45f
  );
  EXPECT_EQ(
    first_evidence_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
  const auto pre_hold_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    first_evidence_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  const auto pre_hold_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    first_evidence_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );

  controller->set_next_elapsed_seconds_for_testing(0.05f);
  ASSERT_TRUE(run_source(held_source.Get(), 202));
  const auto held_snapshot = controller->snapshot();
  const auto held_state = read_rule_state(held_snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      held_state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::full_frame
    )
  );
  EXPECT_GT(
    word_as_float(
      held_state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.45f
  );
  EXPECT_EQ(
    held_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )],
    sbs_scene_controller::state_flags_initialized |
      sbs_scene_controller::state_flags_scroll_hold_active |
      sbs_scene_controller::state_flags_layout_history_valid |
      sbs_scene_controller::state_flags_depth_history_valid
  );
  EXPECT_EQ(
    held_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::promotion_flags
    )],
    sbs_scene_controller::promotion_flags_layout_history
  );
  EXPECT_EQ(
    held_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::history_flags
    )],
    sbs_scene_controller::history_flags_layout_read_bank |
      sbs_scene_controller::history_flags_layout_write_bank |
      sbs_scene_controller::history_flags_depth_read_bank
  );

  const auto held_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    held_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  const auto held_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    held_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );
  const auto layout_plane_diff_count =
    [&](const sbs_scene_controller::layout_history_channel_e channel) {
      const auto offset =
        static_cast<std::size_t>(channel) * analysis_pixels;
      return std::count_if(
        held_layout.begin() + offset,
        held_layout.begin() + offset + analysis_pixels,
        [&, index = offset](const float value) mutable {
           return value != pre_hold_layout[index++];
        }
      );
    };
  EXPECT_GT(
    layout_plane_diff_count(
      sbs_scene_controller::layout_history_channel_e::
        previous_luminance_ordinal
    ),
    0
  );
  // Motion endpoint lanes remain writable during a hold, but a steady
  // translation may reproduce their existing values exactly.
  for (std::size_t channel = 0;
       channel <
       sbs_scene_controller::layout_history_channel_count;
       ++channel) {
    const auto typed_channel =
      static_cast<sbs_scene_controller::layout_history_channel_e>(
        channel
      );
    if (
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::
          previous_luminance_ordinal ||
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::
          horizontal_motion ||
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::
          vertical_motion ||
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::
          motion_confidence
    ) {
      continue;
    }
    SCOPED_TRACE(channel);
    const auto offset = channel * analysis_pixels;
    EXPECT_TRUE(std::equal(held_layout.begin() + offset, held_layout.begin() + offset + analysis_pixels, pre_hold_layout.begin() + offset));
  }
  EXPECT_EQ(held_depth, pre_hold_depth);

  // Motion evidence can disappear before the 120 ms termination dwell has
  // matured. The explicit hold flag, not instantaneous confidence, must keep
  // semantic histories frozen throughout that quiet interval.
  controller->set_next_elapsed_seconds_for_testing(0.06f);
  ASSERT_TRUE(run_source(held_source.Get(), 203));
  const auto quiet_snapshot = controller->snapshot();
  const auto quiet_state = read_rule_state(quiet_snapshot);
  EXPECT_NE(
    quiet_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      quiet_state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  const auto quiet_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    quiet_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count * analysis_pixels
  );
  const auto quiet_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    quiet_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count * analysis_pixels
  );
  for (std::size_t channel = 0;
       channel < sbs_scene_controller::layout_history_channel_count;
       ++channel) {
    const auto typed_channel =
      static_cast<sbs_scene_controller::layout_history_channel_e>(channel);
    if (
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::
          previous_luminance_ordinal ||
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::horizontal_motion ||
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::vertical_motion ||
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::motion_confidence
    ) {
      continue;
    }
    SCOPED_TRACE(channel);
    const auto offset = channel * analysis_pixels;
    EXPECT_TRUE(std::equal(
      quiet_layout.begin() + offset,
      quiet_layout.begin() + offset + analysis_pixels,
      held_layout.begin() + offset
    ));
  }
  EXPECT_EQ(quiet_depth, held_depth);

  controller->set_next_elapsed_seconds_for_testing(0.15f);
  const auto settled_source = make_scroll_source_view(
    device.Get(),
    128,
    128,
    2
  );
  ASSERT_TRUE(settled_source);
  ASSERT_TRUE(run_source(settled_source.Get(), 204));
  const auto released_snapshot = controller->snapshot();
  const auto released_state = read_rule_state(released_snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      released_state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::full_frame
    )
  );
  // Release retains the completed quiet duration for exactly this one
  // resync update. The following update consumes it as the bounded
  // post-scroll detector quarantine and returns the timer to zero.
  EXPECT_NEAR(
    word_as_float(
      released_state,
      sbs_scene_controller::rule_state_word_e::scroll_hold_s
    ),
    0.21f,
    1e-6f
  );
  EXPECT_EQ(
    released_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::promotion_flags
    )],
    sbs_scene_controller::promotion_flags_layout_history |
      sbs_scene_controller::promotion_flags_depth_history
  );
  EXPECT_EQ(
    released_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::history_flags
    )],
    sbs_scene_controller::history_flags_layout_read_bank |
      sbs_scene_controller::history_flags_layout_write_bank |
      sbs_scene_controller::history_flags_depth_read_bank |
      sbs_scene_controller::history_flags_depth_write_bank
  );
  const auto released_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    released_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  EXPECT_FALSE(
    std::equal(
      released_layout.begin(),
      released_layout.end(),
      baseline_layout.begin()
    )
  );

  controller->set_next_elapsed_seconds_for_testing(1.0f / 60.0f);
  ASSERT_TRUE(run_source(settled_source.Get(), 205));
  const auto post_release_state =
    read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      post_release_state,
      sbs_scene_controller::rule_state_word_e::scroll_hold_s
    ),
    0.0f
  );
  EXPECT_EQ(
    post_release_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
}

TEST_F(
  SbsSceneControllerGpu,
  ScrollReleaseReseedsStaleDepthWithoutRetriggeringHardCut
) {
  constexpr std::size_t analysis_pixels =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  constexpr auto event_word =
    sbs_scene_controller::rule_state_word_e::event_decision;
  constexpr auto flags_word =
    sbs_scene_controller::rule_state_word_e::state_flags;
  constexpr auto promotion_word =
    sbs_scene_controller::rule_state_word_e::promotion_flags;
  constexpr auto last_detector_count_word =
    sbs_scene_controller::rule_state_word_e::
      last_detector_cut_count;
  constexpr std::uint32_t detector_cut_count = 1u;

  const auto baseline_source = make_scrolling_content_source_view(
    device.Get(),
    128u,
    128u,
    0u
  );
  const auto first_scroll_source =
    make_scrolling_content_source_view(
      device.Get(),
      128u,
      128u,
      1u
    );
  const auto held_source = make_scrolling_content_source_view(
    device.Get(),
    128u,
    128u,
    2u
  );
  const auto released_source = make_adversarial_source_view(
    device.Get(),
    128u,
    128u,
    std::array {
      adversarial_source_region_t {
        0.15f,
        0.15f,
        0.85f,
        0.85f,
        47u,
        false,
      },
    }
  );
  ASSERT_TRUE(baseline_source);
  ASSERT_TRUE(first_scroll_source);
  ASSERT_TRUE(held_source);
  ASSERT_TRUE(released_source);

  const std::vector<float> baseline_depth_values(
    depth_width * depth_height,
    0.10f
  );
  const std::vector<float> translated_depth_values(
    depth_width * depth_height,
    0.90f
  );
  const auto baseline_depth = make_depth_view(
    device.Get(),
    depth_width,
    depth_height,
    baseline_depth_values
  );
  const auto translated_depth = make_depth_view(
    device.Get(),
    depth_width,
    depth_height,
    translated_depth_values
  );
  ASSERT_TRUE(baseline_depth);
  ASSERT_TRUE(translated_depth);

  depth_view = baseline_depth;
  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f,
    0u,
    false
  );
  ASSERT_TRUE(adaptive_buffer.srv);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(baseline_source.Get(), 205u));
  controller->set_next_elapsed_seconds_for_testing(0.71f);
  ASSERT_TRUE(run_source(baseline_source.Get(), 206u));
  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );
  ASSERT_NE(
    state[sbs_scene_controller::index(flags_word)] &
      sbs_scene_controller::state_flags_roi_locked,
    0u
  );

  // The direction-establishing observation still promotes the old depth.
  controller->set_next_elapsed_seconds_for_testing(0.08f);
  ASSERT_TRUE(run_source(first_scroll_source.Get(), 207u));
  state = read_rule_state(controller->snapshot());
  ASSERT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  ASSERT_EQ(
    state[sbs_scene_controller::index(flags_word)] &
      sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
  const auto pre_hold_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    controller->snapshot().depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );

  // Change the inferred depth only after coherent scrolling enters its hold.
  // The current depth therefore remains intentionally absent from history.
  depth_view = translated_depth;
  controller->set_next_elapsed_seconds_for_testing(0.05f);
  ASSERT_TRUE(run_source(held_source.Get(), 208u));
  state = read_rule_state(controller->snapshot());
  ASSERT_NE(
    state[sbs_scene_controller::index(flags_word)] &
      sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
  ASSERT_EQ(
    state[sbs_scene_controller::index(promotion_word)] &
      sbs_scene_controller::promotion_flags_depth_history,
    0u
  );
  const auto held_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    controller->snapshot().depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );
  ASSERT_EQ(held_depth, pre_hold_depth);

  controller->set_next_elapsed_seconds_for_testing(0.06f);
  ASSERT_TRUE(run_source(held_source.Get(), 209u));
  state = read_rule_state(controller->snapshot());
  ASSERT_NE(
    state[sbs_scene_controller::index(flags_word)] &
      sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );

  // A real detector edge now arrives with broad appearance activity. Against
  // the frozen pre-scroll depth bank it has enough structural corroboration
  // to look like a cut, but this is the first update beyond the 120 ms quiet
  // tail and must be used only to resynchronize history.
  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f,
    detector_cut_count,
    true
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  controller->set_next_elapsed_seconds_for_testing(0.07f);
  ASSERT_TRUE(run_source(released_source.Get(), 210u));
  const auto released_snapshot = controller->snapshot();
  const auto released_state = read_rule_state(released_snapshot);
  EXPECT_EQ(
    released_state[sbs_scene_controller::index(flags_word)] &
      sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
  EXPECT_FLOAT_EQ(
    word_as_float(released_state, event_word),
    static_cast<float>(
      sbs_scene_controller::event_decision_e::same_shot
    )
  );
  EXPECT_EQ(
    released_state[sbs_scene_controller::index(promotion_word)] &
      sbs_scene_controller::promotion_flags_shot,
    0u
  );
  EXPECT_NE(
    released_state[sbs_scene_controller::index(promotion_word)] &
      sbs_scene_controller::promotion_flags_depth_history,
    0u
  );
  EXPECT_NEAR(
    word_as_float(
      released_state,
      sbs_scene_controller::rule_state_word_e::
        roi_structural_cut_support
    ),
    0.0f,
    1e-6f
  );

  const auto released_depth = read_buffer<float>(
    device.Get(),
    context.Get(),
    released_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count *
      analysis_pixels
  );
  const auto last_depth_offset =
    static_cast<std::size_t>(
      sbs_scene_controller::depth_history_channel_e::
        last_normalized_depth
    ) *
    analysis_pixels;
  EXPECT_GT(
    std::count_if(
      released_depth.begin() + last_depth_offset,
      released_depth.begin() + last_depth_offset + analysis_pixels,
      [&, index = last_depth_offset](const float value) mutable {
        return std::abs(value - held_depth[index++]) > 0.50f;
      }
    ),
    analysis_pixels / 4u
  );

  // The resynchronized frame is now the comparison baseline. The retained
  // detector count must be rejected/consumed rather than firing one update
  // late after the scroll overlay has disappeared.
  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f,
    detector_cut_count,
    false
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  controller->set_next_elapsed_seconds_for_testing(1.0f / 60.0f);
  ASSERT_TRUE(run_source(released_source.Get(), 211u));
  const auto stable_state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(stable_state, event_word),
    static_cast<float>(
      sbs_scene_controller::event_decision_e::same_shot
    )
  );
  EXPECT_EQ(
    stable_state[sbs_scene_controller::index(promotion_word)] &
      sbs_scene_controller::promotion_flags_shot,
    0u
  );
  EXPECT_EQ(
    stable_state[sbs_scene_controller::index(last_detector_count_word)],
    detector_cut_count
  );
  EXPECT_EQ(
    stable_state[sbs_scene_controller::index(flags_word)] &
      sbs_scene_controller::state_flags_detector_cut_pending,
    0u
  );
}

TEST_F(
  SbsSceneControllerGpu,
  LargeCommittedPlayerLeavesExteriorScrollEligibleAndFreezesAfterDwell
) {
  constexpr UINT width = 320u;
  constexpr UINT height = 180u;
  std::uint64_t frame_id = 970u;
  const auto run_player =
    [&](const std::uint32_t player_phase,
        const UINT exterior_vertical_shift,
        const float elapsed_seconds,
        const int exterior_horizontal_shift = 0,
        const int player_horizontal_shift = 0) {
      const auto source =
        make_large_player_with_scrolling_exterior_source_view(
          device.Get(),
          width,
          height,
          player_phase,
          exterior_vertical_shift,
          0u,
          0u,
          exterior_horizontal_shift,
          player_horizontal_shift,
          height / 10u
        );
      EXPECT_TRUE(source);
      if (!source) {
        return false;
      }
      controller->set_next_elapsed_seconds_for_testing(elapsed_seconds);
      return run_source(source.Get(), frame_id++);
    };

  ASSERT_TRUE(run_player(0u, 0u, 0.0f));
  ASSERT_TRUE(run_player(1u, 0u, 0.10f));
  ASSERT_TRUE(run_player(2u, 0u, 0.25f));
  ASSERT_TRUE(run_player(3u, 0u, 0.25f));
  ASSERT_TRUE(run_player(4u, 0u, 0.25f));

  const auto committed_snapshot = controller->snapshot();
  const auto committed_state = read_rule_state(committed_snapshot);
  constexpr std::size_t analysis_pixels =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  ASSERT_FLOAT_EQ(
    word_as_float(
      committed_state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  ASSERT_FLOAT_EQ(
    word_as_float(
      committed_state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  const float committed_width =
    word_as_float(
      committed_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ) -
    word_as_float(
      committed_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    );
  const float committed_height =
    word_as_float(
      committed_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    ) -
    word_as_float(
      committed_state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    );
  ASSERT_GT(committed_width * committed_height, 0.60f);
  const auto committed_generation =
    committed_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )];

  // Global scroll remains observable outside this large player. The first
  // observation establishes direction but cannot latch
  // the hold before the required captured-stream dwell.
  ASSERT_TRUE(run_player(4u, 5u, 0.01f));
  const auto first_scroll_snapshot = controller->snapshot();
  const auto first_scroll_state = read_rule_state(first_scroll_snapshot);
  EXPECT_GT(
    word_as_float(
      first_scroll_state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  ) << "committed_area=" << committed_width * committed_height;
  EXPECT_FLOAT_EQ(
    word_as_float(
      first_scroll_state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_LT(
    word_as_float(
      first_scroll_state,
      sbs_scene_controller::rule_state_word_e::scroll_dwell_s
    ),
    0.05f
  );
  EXPECT_EQ(
    first_scroll_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    committed_generation
  );
  EXPECT_EQ(
    first_scroll_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
  const auto pre_hold_layout_history = read_buffer<float>(
    device.Get(),
    context.Get(),
    first_scroll_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count * analysis_pixels
  );
  const auto pre_hold_depth_history = read_buffer<float>(
    device.Get(),
    context.Get(),
    first_scroll_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count * analysis_pixels
  );

  ASSERT_TRUE(run_player(4u, 10u, 0.05f));
  const auto held_scroll_snapshot = controller->snapshot();
  auto state = read_rule_state(held_scroll_snapshot);
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::video
    )
  );
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_NE(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    committed_generation
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::promotion_flags
    )],
    sbs_scene_controller::promotion_flags_layout_history
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::history_flags
    )],
    sbs_scene_controller::history_flags_layout_read_bank |
      sbs_scene_controller::history_flags_layout_write_bank |
      sbs_scene_controller::history_flags_depth_read_bank
  );

  const auto held_scroll_layout_history = read_buffer<float>(
    device.Get(),
    context.Get(),
    held_scroll_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count * analysis_pixels
  );
  const auto held_scroll_depth_history = read_buffer<float>(
    device.Get(),
    context.Get(),
    held_scroll_snapshot.depth_history.Get(),
    sbs_scene_controller::depth_history_channel_count * analysis_pixels
  );
  EXPECT_EQ(held_scroll_depth_history, pre_hold_depth_history);
  for (std::size_t channel = 0u;
       channel < sbs_scene_controller::layout_history_channel_count;
       ++channel) {
    const auto typed_channel =
      static_cast<sbs_scene_controller::layout_history_channel_e>(
        channel
      );
    const bool motion_endpoint =
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::
          previous_luminance_ordinal ||
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::
          horizontal_motion ||
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::
          vertical_motion ||
      typed_channel ==
        sbs_scene_controller::layout_history_channel_e::
          motion_confidence;
    if (motion_endpoint) {
      continue;
    }
    const auto offset = channel * analysis_pixels;
    SCOPED_TRACE(channel);
    EXPECT_TRUE(std::equal(
      held_scroll_layout_history.begin() + offset,
      held_scroll_layout_history.begin() + offset + analysis_pixels,
      pre_hold_layout_history.begin() + offset
    ));
  }

  // The orthogonal scroll hold preserves committed-layout ownership, so
  // player-local motion remains excluded throughout the 120-ms quiet interval.
  ASSERT_TRUE(run_player(5u, 10u, 0.06f));
  auto held_player_snapshot = controller->snapshot();
  state = read_rule_state(held_player_snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::video
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    committed_generation
  );
  EXPECT_NE(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );

  // A non-finite frame makes this update unavailable, but it must not erase
  // committed-player ownership. The first valid recovery frame still uses
  // the retained ROI for exclusion and containment.
  const auto invalid = make_nan_source_view(device.Get(), width, height);
  ASSERT_TRUE(invalid);
  controller->set_next_elapsed_seconds_for_testing(0.01f);
  ASSERT_TRUE(run_source(invalid.Get(), frame_id++));
  state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::output_valid
    ),
    0.0f
  );
  ASSERT_TRUE(run_player(6u, 10u, 0.02f));
  const auto recovery_snapshot = controller->snapshot();
  state = read_rule_state(recovery_snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::output_valid
    ),
    1.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::video
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    committed_generation
  );

  // The same retained player must also leave its left/right exterior eligible for horizontal
  // page motion. The resulting hold freezes geometry without changing ROI identity.
  ASSERT_TRUE(run_player(6u, 10u, 0.08f, 5, 0));
  state = read_rule_state(controller->snapshot());
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_GT(
    std::abs(word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_x
    )),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    committed_generation
  );

  // Once the exterior endpoint settles, motion confined to the established player is an
  // internal pan, not page scroll. Raw motion remains observable, but its page vote is masked.
  ASSERT_TRUE(run_player(6u, 10u, 0.13f, 5, 0));
  ASSERT_TRUE(run_player(6u, 10u, 0.08f, 5, 5));
  const auto internal_pan_snapshot = controller->snapshot();
  state = read_rule_state(internal_pan_snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    committed_generation
  );
}

TEST_F(
  SbsSceneControllerGpu,
  HorizontalPageScrollIsSignedExposureInvariantAndAspectSymmetric
) {
  struct case_t {
    UINT width;
    UINT height;
    int shift;
    int exposure_offset;
  };
  constexpr std::array cases {
    case_t {128u, 72u, 1, 24},
    case_t {128u, 72u, -3, 0},
    case_t {72u, 128u, 3, 24},
    case_t {72u, 128u, -1, 0},
  };
  std::uint64_t frame_id = 1200u;
  for (const auto &test_case : cases) {
    SCOPED_TRACE(testing::Message()
                 << test_case.width << "x" << test_case.height
                 << " shift=" << test_case.shift
                 << " exposure=" << test_case.exposure_offset);
    const auto baseline = make_scroll_source_view(
      device.Get(),
      test_case.width,
      test_case.height,
      0u
    );
    const auto shifted = make_scroll_source_view(
      device.Get(),
      test_case.width,
      test_case.height,
      0u,
      test_case.shift,
      test_case.exposure_offset
    );
    const auto shifted_again = make_scroll_source_view(
      device.Get(),
      test_case.width,
      test_case.height,
      0u,
      2 * test_case.shift,
      test_case.exposure_offset
    );
    ASSERT_TRUE(baseline);
    ASSERT_TRUE(shifted);
    ASSERT_TRUE(shifted_again);

    controller->set_next_reset_flags_for_testing(
      sbs_scene_controller::reset_flags_geometry
    );
    controller->set_next_elapsed_seconds_for_testing(0.0f);
    ASSERT_TRUE(run_source(baseline.Get(), frame_id++));
    controller->set_next_elapsed_seconds_for_testing(0.08f);
    ASSERT_TRUE(run_source(shifted.Get(), frame_id++));

    const auto first_state = read_rule_state(controller->snapshot());
    EXPECT_GT(
      word_as_float(
        first_state,
        sbs_scene_controller::rule_state_word_e::scroll_confidence
      ),
      0.82f
    );
    EXPECT_GT(
      word_as_float(
        first_state,
        sbs_scene_controller::rule_state_word_e::scroll_x
      ) * static_cast<float>(test_case.shift),
      0.0f
    );
    EXPECT_EQ(
      first_state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::state_flags
      )] & sbs_scene_controller::state_flags_scroll_hold_active,
      0u
    );

    controller->set_next_elapsed_seconds_for_testing(0.05f);
    ASSERT_TRUE(run_source(shifted_again.Get(), frame_id++));
    const auto state = read_rule_state(controller->snapshot());
    const float scroll_x = word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_x
    );
    const float scroll_y = word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_y
    );
    EXPECT_GT(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::scroll_confidence
      ),
      0.82f
    );
    EXPECT_GT(
      scroll_x * static_cast<float>(test_case.shift),
      0.0f
    );
    EXPECT_LT(std::abs(scroll_y), 0.10f);
    EXPECT_FLOAT_EQ(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ),
      static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_valid
      ),
      0.0f
    );
    const auto flags = state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )];
    EXPECT_NE(
      flags & sbs_scene_controller::state_flags_scroll_hold_active,
      0u
    );
    EXPECT_EQ(flags & sbs_scene_controller::state_flags_roi_locked, 0u);
  }
}

TEST_F(
  SbsSceneControllerGpu,
  OppositeHorizontalPanelsAbstainInsteadOfInventingPageScroll
) {
  const auto baseline = make_scroll_source_view(
    device.Get(),
    128u,
    128u,
    0u,
    0,
    0,
    true
  );
  const auto opposed = make_scroll_source_view(
    device.Get(),
    128u,
    128u,
    0u,
    1,
    0,
    true
  );
  ASSERT_TRUE(baseline);
  ASSERT_TRUE(opposed);
  controller->set_next_reset_flags_for_testing(
    sbs_scene_controller::reset_flags_geometry
  );
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(baseline.Get(), 1210u));
  controller->set_next_elapsed_seconds_for_testing(0.08f);
  ASSERT_TRUE(run_source(opposed.Get(), 1211u));

  const auto state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_x
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
}

TEST_F(
  SbsSceneControllerGpu,
  OrthogonalScrollVotesProduceOneBoundedTwoAxisVector
) {
  const auto baseline = make_orthogonal_scroll_source_view(
    device.Get(),
    128u,
    128u,
    0,
    0
  );
  const auto shifted = make_orthogonal_scroll_source_view(
    device.Get(),
    128u,
    128u,
    1,
    -1
  );
  ASSERT_TRUE(baseline);
  ASSERT_TRUE(shifted);
  controller->set_next_reset_flags_for_testing(
    sbs_scene_controller::reset_flags_geometry
  );
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(baseline.Get(), 1220u));
  controller->set_next_elapsed_seconds_for_testing(0.08f);
  ASSERT_TRUE(run_source(shifted.Get(), 1221u));

  const auto state = read_rule_state(controller->snapshot());
  const float scroll_x = word_as_float(
    state,
    sbs_scene_controller::rule_state_word_e::scroll_x
  );
  const float scroll_y = word_as_float(
    state,
    sbs_scene_controller::rule_state_word_e::scroll_y
  );
  const float vector_norm = std::hypot(scroll_x, scroll_y);
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.82f
  );
  EXPECT_GT(scroll_x, 0.40f);
  EXPECT_LT(scroll_y, -0.40f);
  EXPECT_GT(vector_norm, 0.90f);
  EXPECT_LE(vector_norm, 1.0f + 1e-6f);
}

TEST_F(
  SbsSceneControllerGpu,
  RepeatedFramesAdvanceTemporalStateAndPreserveAbiInvariants
) {
  constexpr std::size_t analysis_pixels =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  const auto temporal_activity_offset =
    static_cast<std::size_t>(
      sbs_scene_controller::analysis_grid_channel_e::
        temporal_activity_occupancy
    ) *
    analysis_pixels;
  const auto update_count_index = sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::update_count
  );
  const auto promotion_flags_index = sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::promotion_flags
  );
  const auto history_flags_index = sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::history_flags
  );
  const auto output_valid_index = sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::output_valid
  );
  const auto state_kind_index = sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::state_kind
  );

  ASSERT_TRUE(run_frame(320, 180, 200, 0));
  const auto first_snapshot = controller->snapshot();
  const auto first_analysis = read_analysis(first_snapshot);
  const auto first_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    first_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  const auto first_state = read_rule_state(first_snapshot);
  EXPECT_EQ(first_state[update_count_index], 1u);
  EXPECT_FLOAT_EQ(
    std::bit_cast<float>(first_state[output_valid_index]),
    1.0f
  );
  EXPECT_FLOAT_EQ(
    std::bit_cast<float>(first_state[state_kind_index]),
    0.0f
  );
  EXPECT_EQ(
    first_state[promotion_flags_index],
    sbs_scene_controller::promotion_flags_layout_history |
      sbs_scene_controller::promotion_flags_depth_history
  );
  EXPECT_EQ(
    first_state[history_flags_index],
    sbs_scene_controller::history_flags_layout_read_bank |
      sbs_scene_controller::history_flags_layout_write_bank |
      sbs_scene_controller::history_flags_depth_read_bank |
      sbs_scene_controller::history_flags_depth_write_bank
  );
  EXPECT_EQ(
    std::count_if(
      first_analysis.begin() + temporal_activity_offset,
      first_analysis.begin() + temporal_activity_offset + analysis_pixels,
      [](const float value) {
        return value != 0.0f;
      }
    ),
    0
  );
  EXPECT_EQ(
    std::count_if(
      first_analysis.begin() + analysis_pixels,
      first_analysis.begin() + 2u * analysis_pixels,
      [&, index = std::size_t {0}](const float value) mutable {
        const bool different =
          value != first_layout[index++];
        return different;
      }
    ),
    0
  );

  ASSERT_TRUE(run_frame(320, 180, 201, 0));
  const auto steady_snapshot = controller->snapshot();
  const auto steady_analysis = read_analysis(steady_snapshot);
  const auto steady_state = read_rule_state(steady_snapshot);
  EXPECT_EQ(steady_snapshot.source_frame_id, 201u);
  EXPECT_EQ(steady_state[update_count_index], 2u);
  EXPECT_EQ(
    std::count_if(
      steady_analysis.begin() + temporal_activity_offset,
      steady_analysis.begin() + temporal_activity_offset + analysis_pixels,
      [](const float value) {
        return value != 0.0f;
      }
    ),
    0
  );

  ASSERT_TRUE(run_frame(320, 180, 202, 1));
  const auto changed_snapshot = controller->snapshot();
  const auto changed_analysis = read_analysis(changed_snapshot);
  const auto changed_state = read_rule_state(changed_snapshot);
  EXPECT_EQ(changed_snapshot.source_frame_id, 202u);
  EXPECT_EQ(changed_state[update_count_index], 3u);
  EXPECT_GT(
    std::count_if(
      changed_analysis.begin() + temporal_activity_offset,
      changed_analysis.begin() + temporal_activity_offset + analysis_pixels,
      [](const float value) {
        return value > 0.0f;
      }
    ),
    0
  );
  expect_snapshot_abi_invariants(
    device.Get(),
    context.Get(),
    changed_snapshot
  );
}

TEST_F(
  SbsSceneControllerGpu,
  MissingTemporalEvidenceBeyondCadenceGapClearsProvisionalPlayer
) {
  const auto seed = make_player_source_view(
    device.Get(),
    320,
    180,
    0u
  );
  const auto changed = make_player_source_view(
    device.Get(),
    320,
    180,
    1u
  );
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 800u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 801u));

  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  // One half-second quiet interval is far beyond the bounded cadence gap used
  // for duplicated source frames, so the provisional player must be cleared.
  controller->set_next_elapsed_seconds_for_testing(0.50f);
  ASSERT_TRUE(run_source(changed.Get(), 802u));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_score
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_dwell_s
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::full_frame
    )
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_temporal_lockout_active,
    0u
  );
}

TEST_F(
  SbsSceneControllerGpu,
  LowerCadencePlaybackAcquiresUsingNinetyHertzStreamTime
) {
  // The 90-Hz stream timeline is the sole clock authority. One visually fresh
  // update followed by two duplicates is only an adversarial input pattern;
  // the controller neither receives nor infers an embedded-video frame rate.
  constexpr float update_seconds = 1.0f / 90.0f;
  const auto seed = make_player_source_view(
    device.Get(),
    320,
    180,
    0u
  );
  ASSERT_TRUE(seed);

  std::uint64_t frame_id = 8200u;
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), frame_id++));

  bool saw_temporal_probe = false;
  bool committed_video = false;
  for (std::uint32_t phase = 1u;
       phase <= 30u && !committed_video;
       ++phase) {
    const auto changed = make_player_source_view(
      device.Get(),
      320,
      180,
      phase
    );
    ASSERT_TRUE(changed);
    controller->set_next_elapsed_seconds_for_testing(update_seconds);
    ASSERT_TRUE(run_source(changed.Get(), frame_id++));
    auto state = read_rule_state(controller->snapshot());
    committed_video =
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ) ==
      static_cast<float>(sbs_scene_controller::state_kind_e::video);
    if (committed_video) {
      break;
    }

    const bool probe_active =
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_valid
      ) > 0.5f;
    saw_temporal_probe = saw_temporal_probe || probe_active;
    const float dwell_before_duplicates = word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_dwell_s
    );

    // Duplicate stream submissions retain the temporal probe for the bounded
    // stream-time gap, but cannot add dwell by themselves.
    for (std::uint32_t duplicate = 0u; duplicate < 2u; ++duplicate) {
      controller->set_next_elapsed_seconds_for_testing(update_seconds);
      ASSERT_TRUE(run_source(changed.Get(), frame_id++));
      state = read_rule_state(controller->snapshot());
      if (probe_active) {
        ASSERT_FLOAT_EQ(
          word_as_float(
            state,
            sbs_scene_controller::rule_state_word_e::acquisition_valid
          ),
          1.0f
        ) << "phase=" << phase
          << " duplicate=" << duplicate
          << " state_kind="
          << word_as_float(
               state,
               sbs_scene_controller::rule_state_word_e::state_kind
             )
          << " layout="
          << word_as_float(
               state,
               sbs_scene_controller::rule_state_word_e::layout_decision
             )
          << " acquisition_layout="
          << word_as_float(
               state,
               sbs_scene_controller::rule_state_word_e::acquisition_layout
             )
          << " scroll="
          << word_as_float(
               state,
               sbs_scene_controller::rule_state_word_e::scroll_confidence
             )
          << " missing="
          << word_as_float(
               state,
               sbs_scene_controller::rule_state_word_e::
                 seconds_since_layout_evidence
             );
        EXPECT_FLOAT_EQ(
          word_as_float(
            state,
            sbs_scene_controller::rule_state_word_e::acquisition_dwell_s
          ),
          dwell_before_duplicates
        );
        EXPECT_LE(
          word_as_float(
            state,
            sbs_scene_controller::rule_state_word_e::
              seconds_since_layout_evidence
          ),
          2.0f * update_seconds + 1e-6f
        );
      }
    }
  }

  EXPECT_TRUE(saw_temporal_probe);
  ASSERT_TRUE(committed_video);
  const auto state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  AcquisitionWallTimeIsInvariantAcrossStreamingRates
) {
  struct acquisition_result_t {
    float stream_elapsed_seconds;
    std::uint32_t stream_updates;
  };

  config::video_t::sbs_t sbs;
  sbs.scene_controller =
    config::sbs_scene_controller_e::shadow_rules;
  const std::filesystem::path assets_dir =
    std::filesystem::path(SUNSHINE_SHADERS_DIR)
      .parent_path()
      .parent_path();
  const auto acquire_at_stream_rate =
    [&](const float stream_fps,
        const std::uint64_t first_frame_id) {
      models::sbs_scene_controller_gpu target(
        device,
        context,
        assets_dir,
        sbs.scene_controller,
        sbs,
        false
      );
      EXPECT_TRUE(target.valid());
      const auto run =
        [&](ID3D11ShaderResourceView *source,
            const std::uint64_t frame_id,
            const float elapsed_seconds) {
          target.set_next_elapsed_seconds_for_testing(elapsed_seconds);
          if (!target.prepare_scene(
                source,
                models::input_color_space::srgb,
                frame_id
              )) {
            return false;
          }
          target.mark_enqueued(frame_id);
          return target.resolve_completed(
            frame_id,
            raw_depth_buffer.srv.Get(),
            depth_view.Get(),
            frame_state_buffer.srv.Get(),
            adaptive_buffer.srv.Get(),
            depth_width,
            depth_height,
            frame_roi_transform_buffer.srv.Get()
          );
        };

      const auto seed = make_player_source_view(
        device.Get(),
        320,
        180,
        0u
      );
      EXPECT_TRUE(seed);
      EXPECT_TRUE(run(seed.Get(), first_frame_id, 0.0f));

      const float stream_step_seconds = 1.0f / stream_fps;
      float stream_elapsed_seconds = 0.0f;
      for (std::uint32_t update = 1u; update <= 120u; ++update) {
        const auto changed = make_player_source_view(
          device.Get(),
          320,
          180,
          update
        );
        EXPECT_TRUE(changed);
        stream_elapsed_seconds += stream_step_seconds;
        EXPECT_TRUE(run(
          changed.Get(),
          first_frame_id + update,
          stream_step_seconds
        ));
        const auto state = read_rule_state(target.snapshot());
        if (
          word_as_float(
            state,
            sbs_scene_controller::rule_state_word_e::state_kind
          ) ==
          static_cast<float>(sbs_scene_controller::state_kind_e::video)
        ) {
          return acquisition_result_t {
            stream_elapsed_seconds,
            update,
          };
        }
      }
      ADD_FAILURE()
        << "video did not acquire at stream rate " << stream_fps;
      return acquisition_result_t {
        stream_elapsed_seconds,
        120u,
      };
    };

  const auto at_72_hz = acquire_at_stream_rate(72.0f, 8300u);
  const auto at_90_hz = acquire_at_stream_rate(90.0f, 8500u);
  EXPECT_NE(at_72_hz.stream_updates, at_90_hz.stream_updates);
  EXPECT_NEAR(
    at_72_hz.stream_elapsed_seconds,
    at_90_hz.stream_elapsed_seconds,
    1.0f / 72.0f + 1e-5f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  AmbiguityCannotBridgeProvisionalOrChallengerVideoDwell
) {
  const auto make_frame =
    [&](const std::uint32_t left_phase,
        const std::uint32_t right_phase,
        const std::uint32_t top_phase,
        const std::uint32_t bottom_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          adversarial_source_region_t {
            0.05f, 0.25f, 0.45f, 0.75f, left_phase, false,
          },
          adversarial_source_region_t {
            0.55f, 0.25f, 0.95f, 0.75f, right_phase, false,
          },
          adversarial_source_region_t {
            0.15f, 0.02f, 0.45f, 0.18f, top_phase, false,
          },
          adversarial_source_region_t {
            0.15f, 0.82f, 0.45f, 0.98f, bottom_phase, false,
          },
        }
      );
    };
  const auto update =
    [&](const std::uint64_t frame_id,
        const float elapsed_seconds,
        const std::uint32_t left_phase,
        const std::uint32_t right_phase,
        const std::uint32_t top_phase,
        const std::uint32_t bottom_phase) {
      const auto source = make_frame(
        left_phase,
        right_phase,
        top_phase,
        bottom_phase
      );
      EXPECT_TRUE(source);
      controller->set_next_elapsed_seconds_for_testing(elapsed_seconds);
      return run_source(source.Get(), frame_id);
    };
  const auto state_float =
    [](const std::vector<std::uint32_t> &state,
       const sbs_scene_controller::rule_state_word_e word) {
      return word_as_float(state, word);
    };
  constexpr float full_frame = static_cast<float>(
    sbs_scene_controller::state_kind_e::full_frame
  );
  constexpr float video = static_cast<float>(
    sbs_scene_controller::state_kind_e::video
  );
  constexpr float primary_video = static_cast<float>(
    sbs_scene_controller::layout_decision_e::primary_video
  );
  constexpr float ambiguous = static_cast<float>(
    sbs_scene_controller::layout_decision_e::ambiguous
  );

  // A valid left-player probe followed by two equal moving rectangles is
  // explicit ambiguity, not a low-cadence hole in the left player's stream.
  // If the 100-ms ambiguous interval survives, the returning 100-ms sample
  // can incorrectly satisfy the 200-ms initial VIDEO dwell.
  ASSERT_TRUE(update(8600u, 0.0f, 0u, 0u, 0u, 0u));
  ASSERT_TRUE(update(8601u, 0.10f, 1u, 0u, 0u, 0u));
  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  ASSERT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_layout
    ),
    primary_video
  );

  ASSERT_TRUE(update(8602u, 0.10f, 2u, 1u, 0u, 0u));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    ambiguous
  );
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    full_frame
  );

  ASSERT_TRUE(update(8603u, 0.10f, 3u, 1u, 0u, 0u));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    full_frame
  );
  if (
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ) > 0.5f
  ) {
    EXPECT_FLOAT_EQ(
      state_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_dwell_s
      ),
      0.0f
    );
  }

  // Establish the same left rectangle as a committed VIDEO, then hold a
  // right-side relocation challenger just below its 750-ms dwell.
  controller->set_next_reset_flags_for_testing(
    sbs_scene_controller::reset_flags_layout |
      sbs_scene_controller::reset_flags_depth_shot |
      sbs_scene_controller::reset_flags_geometry
  );
  ASSERT_TRUE(update(8610u, 0.0f, 10u, 10u, 10u, 10u));
  ASSERT_TRUE(update(8611u, 0.10f, 11u, 10u, 10u, 10u));
  ASSERT_TRUE(update(8612u, 0.10f, 12u, 10u, 10u, 10u));
  ASSERT_TRUE(update(8613u, 0.10f, 13u, 10u, 10u, 10u));
  state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    video
  );
  const auto incumbent_generation =
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )];

  const auto run_explicit_source =
    [&](ID3D11ShaderResourceView *source,
        const std::uint64_t frame_id,
        const float elapsed_seconds) {
      controller->set_next_elapsed_seconds_for_testing(elapsed_seconds);
      return run_source(source, frame_id);
    };
  const auto make_relocated =
    [&](const std::uint32_t phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          adversarial_source_region_t {
            0.55f, 0.25f, 0.95f, 0.75f, phase, false,
          },
        }
      );
    };
  const auto make_ambiguous_competitors =
    [&](const std::uint32_t phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          adversarial_source_region_t {
            0.15f, 0.02f, 0.45f, 0.18f, phase, false,
          },
          adversarial_source_region_t {
            0.15f, 0.82f, 0.45f, 0.98f, phase, false,
          },
        }
      );
    };

  std::uint64_t challenger_frame_id = 8614u;
  std::uint32_t relocated_phase = 0u;
  bool challenger_primed = false;
  for (; relocated_phase < 20u; ++relocated_phase) {
    const auto relocated = make_relocated(relocated_phase);
    ASSERT_TRUE(relocated);
    ASSERT_TRUE(run_explicit_source(
      relocated.Get(),
      challenger_frame_id++,
      0.10f
    ));
    state = read_rule_state(controller->snapshot());
    ASSERT_EQ(
      state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::roi_generation
      )],
      incumbent_generation
    );
    challenger_primed =
      state_float(
        state,
        sbs_scene_controller::rule_state_word_e::challenger_valid
      ) > 0.5f &&
      state_float(
        state,
        sbs_scene_controller::rule_state_word_e::challenger_layout
      ) == primary_video &&
      state_float(
        state,
        sbs_scene_controller::rule_state_word_e::challenger_dwell_s
      ) >= 0.50f;
    if (challenger_primed) {
      ++relocated_phase;
      break;
    }
  }
  ASSERT_TRUE(challenger_primed);

  // Neither ambiguous alternative overlaps the right-side challenger. The
  // first frame changes layouts; the second makes the two equal alternatives
  // the only fresh temporal evidence.
  const auto competitors_0 = make_ambiguous_competitors(0u);
  const auto competitors_1 = make_ambiguous_competitors(1u);
  ASSERT_TRUE(competitors_0);
  ASSERT_TRUE(competitors_1);
  ASSERT_TRUE(run_explicit_source(
    competitors_0.Get(),
    challenger_frame_id++,
    0.05f
  ));
  ASSERT_TRUE(run_explicit_source(
    competitors_1.Get(),
    challenger_frame_id++,
    0.10f
  ));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    ambiguous
  );
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    incumbent_generation
  );

  // Without the ambiguity clear, the retained 0.50-second dwell plus both
  // gaps and these returning samples crosses the 0.75-second relocation
  // threshold.
  const auto returned_0 = make_relocated(relocated_phase++);
  const auto returned_1 = make_relocated(relocated_phase++);
  ASSERT_TRUE(returned_0);
  ASSERT_TRUE(returned_1);
  ASSERT_TRUE(run_explicit_source(
    returned_0.Get(),
    challenger_frame_id++,
    0.15f
  ));
  ASSERT_TRUE(run_explicit_source(
    returned_1.Get(),
    challenger_frame_id++,
    0.15f
  ));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    video
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    incumbent_generation
  );
  if (
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ) > 0.5f
  ) {
    EXPECT_LE(
      state_float(
        state,
        sbs_scene_controller::rule_state_word_e::challenger_dwell_s
      ),
      0.15f
    );
  }
}

TEST_F(
  SbsSceneControllerGpu,
  ExposureFreezesAPlayingVideoProbeWithoutDiscardingItsGeometry
) {
  const auto seed = make_player_source_view(
    device.Get(),
    320,
    180,
    0u
  );
  const auto moving_1 = make_player_source_view(
    device.Get(),
    320,
    180,
    1u
  );
  const auto moving_2 = make_player_source_view(
    device.Get(),
    320,
    180,
    2u
  );
  const auto moving_3 = make_player_source_view(
    device.Get(),
    320,
    180,
    3u
  );
  ASSERT_TRUE(seed);
  ASSERT_TRUE(moving_1);
  ASSERT_TRUE(moving_2);
  ASSERT_TRUE(moving_3);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 806u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(moving_1.Get(), 807u));

  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  const auto probe_bounds = std::array {
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_roi_x0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_roi_y0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_roi_x1
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_roi_y1
    ),
  };
  const float probe_dwell = word_as_float(
    state,
    sbs_scene_controller::rule_state_word_e::acquisition_dwell_s
  );
  const float probe_age = word_as_float(
    state,
    sbs_scene_controller::rule_state_word_e::seconds_since_layout_evidence
  );

  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f,
    0u,
    false,
    sbs_adaptive_state::analysis_flag_exposure_like
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  controller->set_next_elapsed_seconds_for_testing(0.40f);
  ASSERT_TRUE(run_source(moving_2.Get(), 808u));
  state = read_rule_state(controller->snapshot());

  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::event_decision
    ),
    static_cast<float>(
      sbs_scene_controller::event_decision_e::flash_or_exposure
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_dwell_s
    ),
    probe_dwell
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::seconds_since_layout_evidence
    ),
    probe_age
  );
  EXPECT_EQ(
    (std::array {
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_roi_x0
      ),
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_roi_y0
      ),
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_roi_x1
      ),
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_roi_y1
      ),
    }),
    probe_bounds
  );

  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(moving_3.Get(), 809u));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  EXPECT_NEAR(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_dwell_s
    ),
    probe_dwell + 0.10f,
    1e-6f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  CommittedContentTemporalChallengerRequiresConsecutiveEvidence
) {
  const auto state_float =
    [&](const std::vector<std::uint32_t> &state,
        const sbs_scene_controller::rule_state_word_e word) {
      return word_as_float(state, word);
    };
  const auto challenger_bounds =
    [&](const std::vector<std::uint32_t> &state) {
      return std::array {
        state_float(
          state,
          sbs_scene_controller::rule_state_word_e::challenger_roi_x0
        ),
        state_float(
          state,
          sbs_scene_controller::rule_state_word_e::challenger_roi_y0
        ),
        state_float(
          state,
          sbs_scene_controller::rule_state_word_e::challenger_roi_x1
        ),
        state_float(
          state,
          sbs_scene_controller::rule_state_word_e::challenger_roi_y1
        ),
      };
    };

  const auto seed = make_content_with_embedded_player_source_view(
    device.Get(),
    0u
  );
  const auto event = make_content_with_embedded_player_source_view(
    device.Get(),
    1u
  );
  ASSERT_TRUE(seed);
  ASSERT_TRUE(event);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 900u));
  controller->set_next_elapsed_seconds_for_testing(0.71f);
  ASSERT_TRUE(run_source(seed.Get(), 901u));
  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );

  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(event.Get(), 902u));
  state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    1.0f
  );
  ASSERT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  // Let current-frame activity settle. Zero elapsed time cannot advance dwell.
  std::uint64_t frame_id = 903u;
  for (std::size_t index = 0; index < 8u; ++index) {
    controller->set_next_elapsed_seconds_for_testing(0.0f);
    ASSERT_TRUE(run_source(event.Get(), frame_id++));
  }

  // Missing current evidence clears the pending replacement immediately.
  controller->set_next_elapsed_seconds_for_testing(0.50f);
  ASSERT_TRUE(run_source(event.Get(), frame_id++));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_dwell_s
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_score
    ),
    0.0f
  );

  // A fresh replacement must provide a new continuous 750-ms evidence run.
  std::array<float, 4> replacement_seeded_bounds {};
  bool replacement_seeded = false;
  for (std::uint32_t phase = 2u; phase <= 6u; ++phase) {
    const auto contained_motion =
      make_content_with_embedded_player_source_view(
        device.Get(),
        phase
      );
    ASSERT_TRUE(contained_motion);
    controller->set_next_elapsed_seconds_for_testing(0.25f);
    ASSERT_TRUE(run_source(contained_motion.Get(), frame_id++));
    if (!replacement_seeded) {
      const auto replacement_state =
        read_rule_state(controller->snapshot());
      ASSERT_FLOAT_EQ(
        state_float(
          replacement_state,
          sbs_scene_controller::rule_state_word_e::challenger_valid
        ),
        1.0f
      );
      replacement_seeded_bounds = challenger_bounds(replacement_state);
      replacement_seeded = true;
    }
  }
  ASSERT_TRUE(replacement_seeded);
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_FLOAT_EQ(
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  const std::array committed_bounds {
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    ),
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    ),
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ),
    state_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    ),
  };
  EXPECT_EQ(committed_bounds, replacement_seeded_bounds);
}

TEST_F(
  SbsSceneControllerGpu,
  QuietCommittedContentTemporalChallengerClearsWithoutLockout
) {
  const auto seed = make_content_with_embedded_player_source_view(
    device.Get(),
    0u
  );
  const auto event = make_content_with_embedded_player_source_view(
    device.Get(),
    1u
  );
  ASSERT_TRUE(seed);
  ASSERT_TRUE(event);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 940u));
  controller->set_next_elapsed_seconds_for_testing(0.71f);
  ASSERT_TRUE(run_source(seed.Get(), 941u));
  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );

  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(event.Get(), 942u));
  state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    1.0f
  );
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );

  std::uint64_t frame_id = 943u;
  for (std::size_t index = 0; index < 8u; ++index) {
    controller->set_next_elapsed_seconds_for_testing(0.0f);
    ASSERT_TRUE(run_source(event.Get(), frame_id++));
  }
  for (std::size_t index = 0; index < 3u; ++index) {
    controller->set_next_elapsed_seconds_for_testing(1.0f);
    ASSERT_TRUE(run_source(event.Get(), frame_id++));
  }
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::
        seconds_since_layout_evidence
    ),
    0.0f
  );
  EXPECT_GE(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_score
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] &
      sbs_scene_controller::state_flags_temporal_lockout_active,
    0u
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );

  // Static content is immediately visible again; no quiet-rearm sub-state is
  // required before later evidence may create another replacement.
  controller->set_next_elapsed_seconds_for_testing(0.16f);
  ASSERT_TRUE(run_source(event.Get(), frame_id));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::content_collage
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );
}

TEST_F(
  SbsSceneControllerGpu,
  PersistentIdentityEvidencePromptlyRetiresACommittedRoi
) {
  const auto content = make_content_source_view(device.Get(), 320, 180);
  const auto near_full = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array {
      adversarial_source_region_t {
        0.02f,
        0.02f,
        0.98f,
        0.98f,
        0u,
        true,
      },
    }
  );
  ASSERT_TRUE(content);
  ASSERT_TRUE(near_full);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(content.Get(), 820u));
  controller->set_next_elapsed_seconds_for_testing(0.71f);
  ASSERT_TRUE(run_source(content.Get(), 821u));
  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );
  const auto committed_generation =
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )];

  // The first changed frame may be temporal event evidence. Two following
  // steady frames provide the persistent near-full media/identity evidence.
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(near_full.Get(), 822u));
  controller->set_next_elapsed_seconds_for_testing(0.50f);
  ASSERT_TRUE(run_source(near_full.Get(), 823u));
  state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::identity_fullscreen
    )
  );

  // Every committed-geometry replacement uses the same conservative 750-ms
  // consecutive-evidence floor, including a return to canonical full frame.
  controller->set_next_elapsed_seconds_for_testing(0.76f);
  ASSERT_TRUE(run_source(near_full.Get(), 824u));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    committed_generation + 1u
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::identity_fullscreen
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  MissingEvidenceClearsStaticChallengerDwell
) {
  const auto content = make_content_source_view(device.Get(), 320, 180);
  const auto near_full = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array {
      adversarial_source_region_t {
        0.02f,
        0.02f,
        0.98f,
        0.98f,
        0u,
        true,
      },
    }
  );
  const auto no_target = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array<adversarial_source_region_t, 0u> {}
  );
  ASSERT_TRUE(content);
  ASSERT_TRUE(near_full);
  ASSERT_TRUE(no_target);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(content.Get(), 825u));
  controller->set_next_elapsed_seconds_for_testing(0.71f);
  ASSERT_TRUE(run_source(content.Get(), 826u));
  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );

  // The first changed frame can be temporal evidence. The following steady
  // frames establish a static challenger without reaching the shared 750-ms dwell.
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(near_full.Get(), 827u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(near_full.Get(), 828u));
  controller->set_next_elapsed_seconds_for_testing(0.20f);
  ASSERT_TRUE(run_source(near_full.Get(), 829u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(near_full.Get(), 830u));
  state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    1.0f
  );
  ASSERT_NE(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::no_target
    )
  );
  // Bounds/layout projection can legitimately replace the challenger on this update,
  // restarting its dwell at zero. The score lane is retired and canonical zero; validity,
  // layout, bounds and dwell are the complete pending-transition record.
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_score
    ),
    0.0f
  );
  ASSERT_LT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_dwell_s
    ),
    0.75f
  );

  controller->set_next_elapsed_seconds_for_testing(0.01f);
  ASSERT_TRUE(run_source(no_target.Get(), 831u));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_dwell_s
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_score
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );
}

TEST_F(
  SbsSceneControllerGpu,
  AlternatingTemporalRectanglesCannotAccumulateStableDwellOrLockout
) {
  const auto make_frame =
    [&](const std::uint32_t phase_a,
        const std::uint32_t phase_b) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          adversarial_source_region_t {
            0.30f,
            0.76f,
            0.70f,
            0.96f,
            0u,
            true,
          },
          adversarial_source_region_t {
            0.08f,
            0.14f,
            0.38f,
            0.64f,
            phase_a,
            false,
          },
          adversarial_source_region_t {
            0.62f,
            0.14f,
            0.92f,
            0.64f,
            phase_b,
            false,
          },
        }
      );
    };
  const auto seed = make_frame(0u, 0u);
  const auto event_a_1 = make_frame(1u, 0u);
  const auto event_b_1 = make_frame(1u, 1u);
  const auto event_a_2 = make_frame(2u, 1u);
  const auto event_b_2 = make_frame(2u, 2u);
  const auto event_a_3 = make_frame(3u, 2u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(event_a_1);
  ASSERT_TRUE(event_b_1);
  ASSERT_TRUE(event_a_2);
  ASSERT_TRUE(event_b_2);
  ASSERT_TRUE(event_a_3);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 840u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(event_a_1.Get(), 841u));
  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );

  const std::array alternating {
    event_b_1,
    event_a_2,
    event_b_2,
    event_a_3,
  };
  std::uint64_t frame_id = 842u;
  for (const auto &frame : alternating) {
    controller->set_next_elapsed_seconds_for_testing(0.80f);
    ASSERT_TRUE(run_source(frame.Get(), frame_id++));
  }

  state = read_rule_state(controller->snapshot());
  EXPECT_NE(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_temporal_lockout_active,
    0u
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_score
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_score
    ),
    0.0f
  );

  // Invalid evidence is a strict hold, but the retired lockout flag remains
  // canonical zero rather than preserving an obsolete sub-state.
  const auto invalid = make_nan_source_view(device.Get(), 320, 180);
  ASSERT_TRUE(invalid);
  controller->set_next_elapsed_seconds_for_testing(0.02f);
  ASSERT_TRUE(run_source(invalid.Get(), frame_id++));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::output_valid
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_temporal_lockout_active,
    0u
  );
}

TEST_F(
  SbsSceneControllerGpu,
  TemporalMediaExpansionCannotClaimAnUnobservedViewportEdge
) {
  const auto make_frame =
    [&](const std::uint32_t player_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          // Photographic page content is connected to the viewport's left
          // edge but ends well before the right edge. It is not a full-frame
          // identity surface and cannot prove that an embedded player owns
          // the edge.
          adversarial_source_region_t {
            0.0f,
            0.16f,
            0.68f,
            0.86f,
            0u,
            true,
          },
          // Only this interior inset changes. The temporal seed therefore
          // has no direct observation at any viewport edge.
          adversarial_source_region_t {
            0.20f,
            0.32f,
            0.50f,
            0.70f,
            player_phase,
            false,
          },
        }
      );
    };
  const auto seed = make_frame(0u);
  const auto changed = make_frame(1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 868u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 869u));

  const auto snapshot = controller->snapshot();
  ASSERT_TRUE(snapshot.rule_summary);
  const auto summary = read_buffer<float>(
    device.Get(),
    context.Get(),
    snapshot.rule_summary.Get(),
    SBS_RULE_SUMMARY_FLOAT_COUNT
  );
  ASSERT_EQ(summary.size(), SBS_RULE_SUMMARY_FLOAT_COUNT);
  constexpr std::size_t global_x_first =
    SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE + 0u;
  constexpr std::size_t global_x_end =
    SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE + 1u;
  constexpr std::size_t global_x_valid =
    SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE + 2u;
  constexpr std::size_t global_x_ambiguous =
    SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE + 3u;
  constexpr std::size_t global_y_valid =
    SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE + 6u;
  constexpr std::size_t global_y_ambiguous =
    SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE + 7u;
  constexpr std::size_t seed_x_first =
    SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE + 8u;
  constexpr std::size_t seed_valid =
    SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE + 12u;
  constexpr std::size_t rectangle_coherence =
    SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE + 13u;
  constexpr std::size_t event_valid =
    SBS_RULE_SUMMARY_TEMPORAL_PROBE_BASE + 0u;
  constexpr std::size_t event_fill =
    SBS_RULE_SUMMARY_TEMPORAL_PROBE_BASE + 5u;
  constexpr std::size_t event_containment =
    SBS_RULE_SUMMARY_TEMPORAL_PROBE_BASE + 6u;
  constexpr std::size_t event_unresolved =
    SBS_RULE_SUMMARY_TEMPORAL_PROBE_BASE + 10u;
  constexpr std::size_t event_ambiguous =
    SBS_RULE_SUMMARY_TEMPORAL_PROBE_BASE + 11u;

  ASSERT_FLOAT_EQ(summary[global_x_valid], 1.0f);
  ASSERT_FLOAT_EQ(summary[global_y_valid], 1.0f);
  ASSERT_FLOAT_EQ(summary[global_x_ambiguous], 0.0f);
  ASSERT_FLOAT_EQ(summary[global_y_ambiguous], 0.0f);
  ASSERT_FLOAT_EQ(summary[seed_valid], 1.0f);
  EXPECT_FLOAT_EQ(summary[global_x_first], 0.0f);
  EXPECT_GT(summary[seed_x_first], summary[global_x_first]);
  EXPECT_LT(
    summary[global_x_end],
    static_cast<float>(SBS_RULE_SUMMARY_COLUMN_COUNT)
  ) << "the expanded envelope must remain non-identity";
  ASSERT_GE(summary[event_fill], 0.32f);
  ASSERT_GE(summary[event_containment], 0.72f);
  ASSERT_GE(summary[rectangle_coherence], 0.85f);
  EXPECT_FLOAT_EQ(summary[event_valid], 0.0f);
  EXPECT_FLOAT_EQ(summary[event_unresolved], 1.0f);
  EXPECT_FLOAT_EQ(summary[event_ambiguous], 1.0f);
}

TEST_F(
  SbsSceneControllerGpu,
  TemporalSeedUsesContainingContentHullWithoutSidebarAttraction
) {
  const auto make_frame =
    [&](const std::uint32_t player_phase,
        const std::uint32_t sidebar_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          // Static photographic player surface. Only the inset temporal seed
          // moves, so raw activity cannot reveal this outer rectangle.
          adversarial_source_region_t {
            0.12f,
            0.18f,
            0.68f,
            0.78f,
            0u,
            true,
          },
          adversarial_source_region_t {
            0.30f,
            0.32f,
            0.50f,
            0.62f,
            player_phase,
            false,
          },
          // Independent exterior animation remains outside the strong
          // whitespace boundaries and must not enlarge the envelope.
          adversarial_source_region_t {
            0.84f,
            0.36f,
            0.90f,
            0.48f,
            sidebar_phase,
            false,
          },
        }
      );
    };
  const auto seed = make_frame(0u, 0u);
  const auto changed = make_frame(1u, 1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 870u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 871u));

  const auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::content_collage
    )
  );
  const auto bound =
    [&](const sbs_scene_controller::rule_state_word_e word) {
      return word_as_float(state, word);
    };
  const float x0 = bound(
    sbs_scene_controller::rule_state_word_e::acquisition_roi_x0
  );
  const float y0 = bound(
    sbs_scene_controller::rule_state_word_e::acquisition_roi_y0
  );
  const float x1 = bound(
    sbs_scene_controller::rule_state_word_e::acquisition_roi_x1
  );
  const float y1 = bound(
    sbs_scene_controller::rule_state_word_e::acquisition_roi_y1
  );
  EXPECT_LT(x0, 0.25f);
  EXPECT_LT(y0, 0.27f);
  EXPECT_GT(x1, 0.55f);
  EXPECT_GT(y1, 0.67f);
  EXPECT_NEAR(x0, 0.12f, 0.04f);
  EXPECT_NEAR(y0, 0.18f, 0.04f);
  EXPECT_NEAR(x1, 0.68f, 0.04f);
  EXPECT_NEAR(y1, 0.78f, 0.04f);
  EXPECT_LT(x1, 0.80f);
}

TEST_F(
  SbsSceneControllerGpu,
  RepeatedBrowserStripesDoNotAttractTheTemporalEnvelope
) {
  const auto make_frame =
    [&](const std::uint32_t player_phase,
        const std::uint32_t sidebar_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          // Browser-like static text/accent rows sit above the player with
          // about one analysis cell between neighboring bands.
          adversarial_source_region_t {
            0.08f, 0.02f, 0.70f, 0.055f, 0u, true,
          },
          adversarial_source_region_t {
            0.08f, 0.075f, 0.70f, 0.11f, 0u, true,
          },
          adversarial_source_region_t {
            0.08f, 0.13f, 0.70f, 0.165f, 0u, true,
          },
          // The stable player owns a continuous photographic surface around
          // a smaller moving inset.
          adversarial_source_region_t {
            0.08f, 0.18f, 0.70f, 0.82f, 0u, true,
          },
          adversarial_source_region_t {
            0.16f, 0.25f, 0.62f, 0.75f, player_phase, false,
          },
          adversarial_source_region_t {
            0.84f, 0.36f, 0.90f, 0.48f, sidebar_phase, false,
          },
        }
      );
    };
  const auto seed = make_frame(0u, 0u);
  const auto changed = make_frame(1u, 1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 874u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 875u));

  const auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  const float y0 = word_as_float(
    state,
    sbs_scene_controller::rule_state_word_e::acquisition_roi_y0
  );
  EXPECT_GE(y0, 0.12f)
    << "repeated browser stripes must not attract the temporal envelope "
       "toward chrome";
  EXPECT_LE(y0, 0.27f)
    << "the conservative envelope must still contain the directly observed "
       "moving inset; recovering an unproven static border is optional";
}

TEST_F(
  SbsSceneControllerGpu,
  ContainingContentHullKeepsTopSpanningSidebarOutOfPlayerBounds
) {
  const auto make_frame =
    [&](const std::uint32_t player_phase,
        const std::uint32_t sidebar_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          // The stable player is larger than its moving inset. Local gutter
          // expansion should recover this exact envelope.
          adversarial_source_region_t {
            0.10f, 0.22f, 0.70f, 0.82f, 0u, true,
          },
          adversarial_source_region_t {
            0.24f, 0.34f, 0.58f, 0.72f, player_phase, false,
          },
          // A narrow independently moving ad begins in top chrome and
          // overlaps the player's rows. A global row projection joins their
          // Y support; conditioning rows on the dominant player X span must
          // remove it before the envelope is expanded.
          adversarial_source_region_t {
            0.86f, 0.02f, 0.92f, 0.58f, sidebar_phase, false,
          },
        }
      );
    };
  const auto seed = make_frame(0u, 0u);
  const auto changed = make_frame(1u, 1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 872u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 873u));

  const auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::content_collage
    )
  );
  const auto bound =
    [&](const sbs_scene_controller::rule_state_word_e word) {
      return word_as_float(state, word);
    };
  const float x0 = bound(
    sbs_scene_controller::rule_state_word_e::acquisition_roi_x0
  );
  const float y0 = bound(
    sbs_scene_controller::rule_state_word_e::acquisition_roi_y0
  );
  const float x1 = bound(
    sbs_scene_controller::rule_state_word_e::acquisition_roi_x1
  );
  const float y1 = bound(
    sbs_scene_controller::rule_state_word_e::acquisition_roi_y1
  );
  EXPECT_NEAR(x0, 0.10f, 0.04f);
  EXPECT_NEAR(y0, 0.22f, 0.04f);
  EXPECT_NEAR(x1, 0.70f, 0.04f);
  EXPECT_NEAR(y1, 0.82f, 0.04f);
  EXPECT_GT(y0, 0.14f)
    << "top-spanning sidebar activity must not become player chrome";
  EXPECT_LT(x1, 0.78f)
    << "the independent sidebar must remain outside the player ROI";
}

TEST_F(
  SbsSceneControllerGpu,
  ConnectedTemporalRunCrossesOnlyOneUnsupportedAnalysisCell
) {
  struct seam_case_t {
    UINT width;
    UINT height;
    bool split_vertically;
    UINT axis_cells;
  };
  constexpr std::array cases {
    // Square exercises exact 128-cell Y seams. Portrait uses the 72-cell
    // canonical X extent and proves the rule is orientation-neutral.
    seam_case_t {256u, 256u, true, 128u},
    seam_case_t {180u, 320u, false, 72u},
  };

  std::uint64_t frame_id = 874u;
  for (const auto &scene_case : cases) {
    for (const UINT gap_cells : {1u, 2u}) {
      SCOPED_TRACE(testing::Message()
                   << scene_case.width << 'x' << scene_case.height
                   << " split="
                   << (scene_case.split_vertically ? "Y" : "X")
                   << " gap=" << gap_cells);
      const float half_gap =
        0.5f * static_cast<float>(gap_cells) /
        static_cast<float>(scene_case.axis_cells);
      const auto make_frame =
        [&](const std::uint32_t phase) {
          if (scene_case.split_vertically) {
            return make_adversarial_source_view(
              device.Get(),
              scene_case.width,
              scene_case.height,
              std::array {
                adversarial_source_region_t {
                  0.15f, 0.20f, 0.85f, 0.50f - half_gap,
                  phase, false,
                },
                adversarial_source_region_t {
                  0.15f, 0.50f + half_gap, 0.85f, 0.80f,
                  phase, false,
                },
              }
            );
          }
          return make_adversarial_source_view(
            device.Get(),
            scene_case.width,
            scene_case.height,
            std::array {
              adversarial_source_region_t {
                0.15f, 0.20f, 0.50f - half_gap, 0.80f,
                phase, false,
              },
              adversarial_source_region_t {
                0.50f + half_gap, 0.20f, 0.85f, 0.80f,
                phase, false,
              },
            }
          );
        };
      const auto seed = make_frame(0u);
      const auto changed = make_frame(1u);
      ASSERT_TRUE(seed);
      ASSERT_TRUE(changed);

      controller->set_next_reset_flags_for_testing(
        sbs_scene_controller::reset_flags_layout |
          sbs_scene_controller::reset_flags_depth_shot |
          sbs_scene_controller::reset_flags_geometry
      );
      controller->set_next_elapsed_seconds_for_testing(0.0f);
      ASSERT_TRUE(run_source(seed.Get(), frame_id++));
      controller->set_next_elapsed_seconds_for_testing(0.10f);
      ASSERT_TRUE(run_source(changed.Get(), frame_id++));

      const auto state = read_rule_state(controller->snapshot());
      const float acquisition_valid = word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_valid
      );
      if (gap_cells == 1u) {
        EXPECT_FLOAT_EQ(acquisition_valid, 1.0f);
        EXPECT_FLOAT_EQ(
          word_as_float(
            state,
            sbs_scene_controller::rule_state_word_e::acquisition_layout
          ),
          static_cast<float>(
            sbs_scene_controller::layout_decision_e::primary_video
          )
        );
      } else {
        EXPECT_FLOAT_EQ(acquisition_valid, 0.0f)
          << "two unsupported cells must terminate the first component";
        EXPECT_NE(
          word_as_float(
            state,
            sbs_scene_controller::rule_state_word_e::layout_decision
          ),
          static_cast<float>(
            sbs_scene_controller::layout_decision_e::primary_video
          )
        );
      }
    }
  }
}

TEST_F(
  SbsSceneControllerGpu,
  LargerAnimatedCompetitorDoesNotStealAPlayerEnvelope
) {
  std::uint64_t frame_id = 882u;
  for (const bool mirror : {false, true}) {
    const auto transform_x =
      [&](const float x) {
        return mirror ? 1.0f - x : x;
      };
    const auto region =
      [&](float x0, float y0, float x1, float y1,
          const std::uint32_t phase) {
        const float transformed_x0 = transform_x(x0);
        const float transformed_x1 = transform_x(x1);
        return adversarial_source_region_t {
          std::min(transformed_x0, transformed_x1),
          y0,
          std::max(transformed_x0, transformed_x1),
          y1,
          phase,
          false,
        };
      };
    const auto make_frame =
      [&](const std::uint32_t phase) {
        return make_adversarial_source_view(
          device.Get(),
          320,
          180,
          std::array {
            // Intended player.
            region(0.08f, 0.25f, 0.48f, 0.75f, phase),
            // A larger animated exterior region. Its area is deliberately
            // inside the geometric dominance margin, so intensity or
            // left/right placement may not select either rectangle.
            region(0.55f, 0.12f, 0.95f, 0.88f, phase + 7u),
          }
        );
      };
    const auto seed = make_frame(0u);
    const auto changed = make_frame(1u);
    ASSERT_TRUE(seed);
    ASSERT_TRUE(changed);

    controller->set_next_reset_flags_for_testing(
      sbs_scene_controller::reset_flags_layout |
        sbs_scene_controller::reset_flags_depth_shot |
        sbs_scene_controller::reset_flags_geometry
    );
    controller->set_next_elapsed_seconds_for_testing(0.0f);
    ASSERT_TRUE(run_source(seed.Get(), frame_id++));
    controller->set_next_elapsed_seconds_for_testing(0.10f);
    ASSERT_TRUE(run_source(changed.Get(), frame_id++));

    const auto state = read_rule_state(controller->snapshot());
    EXPECT_FLOAT_EQ(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_valid
      ),
      0.0f
    );
    EXPECT_NE(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::layout_decision
      ),
      static_cast<float>(
        sbs_scene_controller::layout_decision_e::primary_video
      )
    );
  }
}

TEST_F(
  SbsSceneControllerGpu,
  UnequalTemporalPanelsUseContainingContentHullNotIntensityWinner
) {
  const auto make_frame =
    [&](const std::uint32_t phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          // Equal outer geometry with deliberately unequal activity
          // intensity. The sparse left bands still span the same supported
          // width as the dense right animation.
          adversarial_source_region_t {
            0.08f, 0.20f, 0.44f, 0.80f, 0u, true,
          },
          adversarial_source_region_t {
            0.56f, 0.20f, 0.92f, 0.80f, 0u, true,
          },
          adversarial_source_region_t {
            0.10f, 0.28f, 0.14f, 0.72f, phase, false,
          },
          adversarial_source_region_t {
            0.38f, 0.28f, 0.42f, 0.72f, phase, false,
          },
          adversarial_source_region_t {
            0.58f, 0.28f, 0.90f, 0.72f, phase, true,
          },
        }
      );
    };
  const auto seed = make_frame(0u);
  const auto changed = make_frame(1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 872u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 873u));

  constexpr std::size_t plane =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  const auto snapshot = controller->snapshot();
  const auto analysis = read_analysis(snapshot);
  const auto dense = read_buffer<float>(
    device.Get(),
    context.Get(),
    snapshot.dense_output.Get(),
    sbs_scene_controller::dense_out_channel_count * plane
  );
  const auto channel_offset =
    [](const auto channel) {
      return static_cast<std::size_t>(channel) * plane;
    };
  const std::size_t valid_offset = channel_offset(
    sbs_scene_controller::analysis_grid_channel_e::viewport_valid
  );
  const std::size_t x_offset = channel_offset(
    sbs_scene_controller::analysis_grid_channel_e::viewport_x
  );
  const std::size_t y_offset = channel_offset(
    sbs_scene_controller::analysis_grid_channel_e::viewport_y
  );
  const std::size_t temporal_offset = channel_offset(
    sbs_scene_controller::dense_out_channel_e::primary_playing_video
  );
  float left_temporal_mass = 0.0f;
  float right_temporal_mass = 0.0f;
  for (std::size_t cell = 0u; cell < plane; ++cell) {
    if (analysis[valid_offset + cell] <= 0.5f) {
      continue;
    }
    const float x = analysis[x_offset + cell];
    const float y = analysis[y_offset + cell];
    const float mass = std::clamp(
      (dense[temporal_offset + cell] - 1.0f / 8.0f) /
        (2.0f / 8.0f),
      0.0f,
      1.0f
    );
    if (y >= 0.20f && y < 0.80f) {
      left_temporal_mass +=
        x >= 0.08f && x < 0.44f ? mass : 0.0f;
      right_temporal_mass +=
        x >= 0.56f && x < 0.92f ? mass : 0.0f;
    }
  }
  ASSERT_GT(left_temporal_mass, 0.0f);
  ASSERT_GT(right_temporal_mass, 0.0f);
  EXPECT_LT(
    std::min(left_temporal_mass, right_temporal_mass) /
      std::max(left_temporal_mass, right_temporal_mass),
    0.60f
  ) << "fixture must prove intensity cannot break equal-span ambiguity";

  const auto state = read_rule_state(snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    1.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
  );
  EXPECT_NE(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::content_collage
    )
  );
  EXPECT_LE(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_roi_x0
    ),
    0.12f
  );
  EXPECT_GE(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_roi_x1
    ),
    0.88f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  EqualStackedTemporalPanelsAbstainSymmetrically
) {
  const auto make_frame =
    [&](const std::uint32_t phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          adversarial_source_region_t {
            0.20f, 0.08f, 0.80f, 0.44f, 0u, true,
          },
          adversarial_source_region_t {
            0.20f, 0.56f, 0.80f, 0.92f, 0u, true,
          },
          adversarial_source_region_t {
            0.28f, 0.10f, 0.72f, 0.42f, phase, false,
          },
          adversarial_source_region_t {
            0.28f, 0.58f, 0.72f, 0.90f, phase, false,
          },
        }
      );
    };
  const auto seed = make_frame(0u);
  const auto changed = make_frame(1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 874u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 875u));

  const auto state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::ambiguous
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  StrongerNarrowCompetitorPreventsSparseWideEnvelopeAcquisition
) {
  const auto make_frame =
    [&](const std::uint32_t player_phase,
        const std::uint32_t competitor_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          // A stable photographic player surface contains several sparse
          // moving bands. Their non-gutter gaps belong to one supported
          // spatial run even though their summed activity is modest.
          adversarial_source_region_t {
            0.12f, 0.18f, 0.70f, 0.78f, 0u, true,
          },
          adversarial_source_region_t {
            0.20f, 0.30f, 0.24f, 0.66f, player_phase, false,
          },
          adversarial_source_region_t {
            0.33f, 0.30f, 0.37f, 0.66f, player_phase, false,
          },
          adversarial_source_region_t {
            0.46f, 0.30f, 0.50f, 0.66f, player_phase, false,
          },
          adversarial_source_region_t {
            0.61f, 0.30f, 0.65f, 0.66f, player_phase, false,
          },
          // This isolated animation changes every covered pixel and carries
          // more aggregate temporal mass, but its supported span is much
          // narrower than the player.
          adversarial_source_region_t {
            0.79f, 0.30f, 0.96f, 0.66f, competitor_phase, true,
          },
        }
      );
    };
  const auto seed = make_frame(0u, 0u);
  const auto changed = make_frame(1u, 1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 874u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 875u));

  constexpr std::size_t plane =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  const auto snapshot = controller->snapshot();
  const auto analysis = read_analysis(snapshot);
  const auto dense = read_buffer<float>(
    device.Get(),
    context.Get(),
    snapshot.dense_output.Get(),
    sbs_scene_controller::dense_out_channel_count * plane
  );
  const auto channel_offset =
    [](const auto channel) {
      return static_cast<std::size_t>(channel) * plane;
    };
  const std::size_t valid_offset = channel_offset(
    sbs_scene_controller::analysis_grid_channel_e::viewport_valid
  );
  const std::size_t x_offset = channel_offset(
    sbs_scene_controller::analysis_grid_channel_e::viewport_x
  );
  const std::size_t y_offset = channel_offset(
    sbs_scene_controller::analysis_grid_channel_e::viewport_y
  );
  const std::size_t temporal_offset = channel_offset(
    sbs_scene_controller::dense_out_channel_e::primary_playing_video
  );
  float player_temporal_mass = 0.0f;
  float competitor_temporal_mass = 0.0f;
  for (std::size_t cell = 0u; cell < plane; ++cell) {
    if (analysis[valid_offset + cell] <= 0.5f) {
      continue;
    }
    const float x = analysis[x_offset + cell];
    const float y = analysis[y_offset + cell];
    if (y < 0.30f || y >= 0.66f) {
      continue;
    }
    const float mass = std::clamp(
      (dense[temporal_offset + cell] - 1.0f / 8.0f) /
        (2.0f / 8.0f),
      0.0f,
      1.0f
    );
    const bool in_player_band =
      (x >= 0.20f && x < 0.24f) ||
      (x >= 0.33f && x < 0.37f) ||
      (x >= 0.46f && x < 0.50f) ||
      (x >= 0.61f && x < 0.65f);
    player_temporal_mass += in_player_band ? mass : 0.0f;
    competitor_temporal_mass +=
      x >= 0.79f && x < 0.96f ? mass : 0.0f;
  }
  ASSERT_GT(player_temporal_mass, 0.0f);
  ASSERT_GT(competitor_temporal_mass, player_temporal_mass)
    << "fixture must make activity-mass containment favor the competitor";

  // Span-first discovery still finds the wide retained envelope, but current activity does not
  // corroborate it: the independent competitor owns most of the fresh mass. Conservative
  // fallback is required instead of turning stale slow occupancy into a player acquisition.
  const auto state = read_rule_state(snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::full_frame
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::ambiguous
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  ThreeStableSidesCannotRescueSparsePlayerWithTexturedOuterMargin
) {
  const auto make_frame =
    [&](const std::uint32_t player_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          // This full-height photographic margin deliberately prevents any
          // strong leading-X gutter. Its small width leaves both horizontal
          // gutter lanes observable around the player.
          adversarial_source_region_t {
            0.00f, 0.00f, 0.12f, 1.00f, 0u, true,
          },
          adversarial_source_region_t {
            0.12f, 0.18f, 0.70f, 0.78f, 0u, true,
          },
          // Sparse motion cannot prove one filled rectangle. Three quiet
          // sides are deliberately not a rescue route in the simplified
          // policy.
          adversarial_source_region_t {
            0.14f, 0.28f, 0.195f, 0.72f, player_phase, false,
          },
          adversarial_source_region_t {
            0.37f, 0.28f, 0.425f, 0.72f, player_phase, false,
          },
          adversarial_source_region_t {
            0.595f, 0.28f, 0.65f, 0.72f, player_phase, false,
          },
        }
      );
    };
  const auto seed = make_frame(0u);
  const auto changed = make_frame(1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 876u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 877u));

  const auto state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::full_frame
    )
  );
  EXPECT_NE(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  TwoStableSidesCannotCorroborateSparsePlayer
) {
  const auto make_frame =
    [&](const std::uint32_t player_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          // Full-height left and full-width top texture deliberately remove
          // the leading boundary on each axis. Quiet space remains only
          // beyond the trailing X and trailing Y sides.
          adversarial_source_region_t {
            0.00f, 0.00f, 0.12f, 1.00f, 0u, true,
          },
          adversarial_source_region_t {
            0.00f, 0.00f, 1.00f, 0.18f, 0u, true,
          },
          adversarial_source_region_t {
            0.12f, 0.18f, 0.70f, 0.78f, 0u, true,
          },
          adversarial_source_region_t {
            0.14f, 0.28f, 0.195f, 0.72f, player_phase, false,
          },
          adversarial_source_region_t {
            0.37f, 0.28f, 0.425f, 0.72f, player_phase, false,
          },
          adversarial_source_region_t {
            0.595f, 0.28f, 0.65f, 0.72f, player_phase, false,
          },
        }
      );
    };
  const auto seed = make_frame(0u);
  const auto changed = make_frame(1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 878u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 879u));

  const auto state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
  );
  EXPECT_NE(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  ColdStaticMediaCommitsOnlyGenericContentWithoutVideoHistory
) {
  const auto source = make_content_source_view(
    device.Get(),
    320,
    180
  );
  ASSERT_TRUE(source);

  const float primary_video = static_cast<float>(
    sbs_scene_controller::layout_decision_e::primary_video
  );
  const auto expect_no_video_authority =
    [&](const std::vector<std::uint32_t> &state) {
      const bool primary_probe =
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::acquisition_valid
        ) > 0.5f &&
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::acquisition_layout
        ) == primary_video;
      const bool primary_challenger =
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::challenger_valid
        ) > 0.5f &&
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::challenger_layout
        ) == primary_video;
      EXPECT_FALSE(primary_probe);
      EXPECT_FALSE(primary_challenger);
      EXPECT_NE(
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::state_kind
        ),
        static_cast<float>(sbs_scene_controller::state_kind_e::video)
      );
      EXPECT_NE(
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::committed_layout
        ),
        primary_video
      );
    };

  // Hold the exact same photographic rectangle for well beyond both the
  // generic-content and video dwell windows. A static first observation has
  // no playback history from which "paused video" could be inferred.
  std::vector<std::uint32_t> state;
  for (std::uint64_t update = 0u; update < 18u; ++update) {
    controller->set_next_elapsed_seconds_for_testing(
      update == 0u ? 0.0f : 0.25f
    );
    ASSERT_TRUE(run_source(source.Get(), 890u + update));
    state = read_rule_state(controller->snapshot());
    expect_no_video_authority(state);
  }

  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::content_collage
    )
  );
  EXPECT_NE(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_roi_locked,
    0u
  );
}

TEST_F(
  SbsSceneControllerGpu,
  DisjointWidgetsAbstainWhileContainedMixedMediaUsesGlobalHull
) {
  enum class scene_e {
    sidebar_right,
    widget_below,
    equal_image_and_video,
  };
  constexpr std::array scenes {
    scene_e::sidebar_right,
    scene_e::widget_below,
    scene_e::equal_image_and_video,
  };
  config::video_t::sbs_t sbs;
  sbs.scene_controller =
    config::sbs_scene_controller_e::shadow_rules;
  const auto assets_dir =
    std::filesystem::path(SUNSHINE_SHADERS_DIR)
      .parent_path()
      .parent_path();
  std::uint64_t frame_id = 9400u;

  for (const auto scene : scenes) {
    SCOPED_TRACE(static_cast<int>(scene));
    models::sbs_scene_controller_gpu target(
      device,
      context,
      assets_dir,
      sbs.scene_controller,
      sbs,
      false
    );
    ASSERT_TRUE(target.valid());
    const auto make_frame =
      [&](const std::uint32_t phase) {
        if (scene == scene_e::sidebar_right) {
          return make_adversarial_source_view(
            device.Get(),
            320,
            180,
            std::array {
              adversarial_source_region_t {
                0.08f, 0.14f, 0.72f, 0.86f, 0u, true,
              },
              adversarial_source_region_t {
                0.86f, 0.32f, 0.94f, 0.60f, phase, false,
              },
            }
          );
        }
        if (scene == scene_e::widget_below) {
          return make_adversarial_source_view(
            device.Get(),
            180,
            320,
            std::array {
              adversarial_source_region_t {
                0.14f, 0.08f, 0.86f, 0.72f, 0u, true,
              },
              adversarial_source_region_t {
                0.32f, 0.86f, 0.60f, 0.94f, phase, false,
              },
            }
          );
        }
        return make_adversarial_source_view(
          device.Get(),
          320,
          180,
          std::array {
            // Equal-sized vertically stacked media is intentionally
            // irreducible without semantics. The safe result is their union.
            adversarial_source_region_t {
              0.14f, 0.10f, 0.68f, 0.48f, 0u, true,
            },
            adversarial_source_region_t {
              0.14f, 0.58f, 0.68f, 0.96f, 0u, true,
            },
            adversarial_source_region_t {
              0.24f, 0.66f, 0.58f, 0.90f, phase, false,
            },
          }
        );
      };
    const auto run =
      [&](ID3D11ShaderResourceView *source, const float elapsed) {
        target.set_next_elapsed_seconds_for_testing(elapsed);
        if (!target.prepare_scene(
              source,
              models::input_color_space::srgb,
              frame_id
            )) {
          return false;
        }
        target.mark_enqueued(frame_id);
        const bool resolved = target.resolve_completed(
          frame_id,
          raw_depth_buffer.srv.Get(),
          depth_view.Get(),
          frame_state_buffer.srv.Get(),
          adaptive_buffer.srv.Get(),
          depth_width,
          depth_height,
          frame_roi_transform_buffer.srv.Get()
        );
        ++frame_id;
        return resolved;
      };

    // The default generic-content acquisition dwell can be longer than one
    // second. Keep the surrounding motion running beyond that deadline so
    // this exercises classification, not an under-filled dwell timer.
    for (std::uint32_t phase = 0u; phase < 12u; ++phase) {
      const auto source = make_frame(phase);
      ASSERT_TRUE(source);
      ASSERT_TRUE(run(source.Get(), phase == 0u ? 0.0f : 0.20f));
    }

    const auto state = read_rule_state(target.snapshot());
    if (scene != scene_e::equal_image_and_video) {
      EXPECT_FLOAT_EQ(
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::state_kind
        ),
        static_cast<float>(
          sbs_scene_controller::state_kind_e::full_frame
        )
      );
      EXPECT_EQ(
        state[sbs_scene_controller::index(
          sbs_scene_controller::rule_state_word_e::state_flags
        )] & sbs_scene_controller::state_flags_roi_locked,
        0u
      );
      continue;
    }
    ASSERT_FLOAT_EQ(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ),
      static_cast<float>(sbs_scene_controller::state_kind_e::content)
    );
    ASSERT_FLOAT_EQ(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_layout
      ),
      static_cast<float>(
        sbs_scene_controller::layout_decision_e::content_collage
      )
    );
    const float x0 = word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    );
    const float y0 = word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    );
    const float x1 = word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    );
    const float y1 = word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    );
    EXPECT_LT(y0, 0.25f);
    EXPECT_GT(y1, 0.88f);
    EXPECT_LT(x0, 0.24f);
    EXPECT_GT(x1, 0.60f);
  }
}

TEST_F(
  SbsSceneControllerGpu,
  CommittedVideoRetainsThroughPauseResumesAndResetMakesStaticCold
) {
  const auto make_frame =
    [&](const std::uint32_t player_phase,
        const std::uint32_t sidebar_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          adversarial_source_region_t {
            0.12f, 0.18f, 0.68f, 0.78f, player_phase, false,
          },
          adversarial_source_region_t {
            0.82f, 0.30f, 0.94f, 0.56f, sidebar_phase, true,
          },
        }
      );
    };
  const auto seed = make_frame(0u, 0u);
  // Establish the player with motion across its complete envelope before
  // introducing the independent sidebar.
  // Simultaneously animating both regions is deliberately contradictory acquisition evidence,
  // not a valid setup for the paused-player retention behavior exercised below.
  const auto moving_1 = make_frame(1u, 0u);
  const auto moving_2 = make_frame(2u, 0u);
  const auto moving_3 = make_frame(3u, 0u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(moving_1);
  ASSERT_TRUE(moving_2);
  ASSERT_TRUE(moving_3);

  std::uint64_t frame_id = 874u;
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), frame_id++));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(moving_1.Get(), frame_id++));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(moving_2.Get(), frame_id++));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(moving_3.Get(), frame_id++));

  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  ) << "layout="
    << word_as_float(
         state,
         sbs_scene_controller::rule_state_word_e::layout_decision
       )
    << " acquisition_valid="
    << word_as_float(
         state,
         sbs_scene_controller::rule_state_word_e::acquisition_valid
       )
    << " acquisition_layout="
    << word_as_float(
         state,
         sbs_scene_controller::rule_state_word_e::acquisition_layout
       )
    << " acquisition_dwell="
    << word_as_float(
         state,
         sbs_scene_controller::rule_state_word_e::acquisition_dwell_s
       )
    << " scroll="
    << word_as_float(
         state,
         sbs_scene_controller::rule_state_word_e::scroll_confidence
       );
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  const std::array committed {
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    ),
  };
  const auto committed_generation =
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )];

  // Freeze the player while a colorful exterior block continues changing beyond the ordinary
  // content-challenger dwell. It must not relabel the learned player as a static collage.
  for (std::uint32_t sidebar_phase = 4u;
       sidebar_phase < 14u;
       ++sidebar_phase) {
    const auto paused = make_frame(3u, sidebar_phase);
    ASSERT_TRUE(paused);
    controller->set_next_elapsed_seconds_for_testing(0.12f);
    ASSERT_TRUE(run_source(paused.Get(), frame_id++));
  }

  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    0.0f
  );
  const std::array retained {
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    ),
  };
  EXPECT_EQ(retained, committed);
  EXPECT_LT(retained[2], 0.80f);

  // Make the entire frame identical for 68 seconds: beyond the generic 60-second release
  // ceiling and long enough to drain all activity history, so retention here is
  // committed history—not residual motion evidence.
  const auto fully_paused = make_frame(3u, 7u);
  ASSERT_TRUE(fully_paused);
  for (const float elapsed : {30.0f, 30.0f, 8.0f}) {
    controller->set_next_elapsed_seconds_for_testing(elapsed);
    ASSERT_TRUE(run_source(fully_paused.Get(), frame_id++));
  }
  auto paused_snapshot = controller->snapshot();
  state = read_rule_state(paused_snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    committed_generation
  );
  EXPECT_NE(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_roi_locked,
    0u
  );
  const std::array long_pause_bounds {
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    ),
  };
  EXPECT_EQ(long_pause_bounds, committed);

  constexpr std::size_t analysis_pixels =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  const auto paused_layout = read_buffer<float>(
    device.Get(),
    context.Get(),
    paused_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count *
      analysis_pixels
  );
  const auto slow_activity_offset =
    static_cast<std::size_t>(
      sbs_scene_controller::layout_history_channel_e::slow_activity_ema
    ) * analysis_pixels;
  EXPECT_LT(
    *std::max_element(
      paused_layout.begin() +
        static_cast<std::ptrdiff_t>(slow_activity_offset),
      paused_layout.begin() +
        static_cast<std::ptrdiff_t>(
          slow_activity_offset + analysis_pixels
        )
    ),
    0.001f
  );
  const auto paused_analysis = read_analysis(paused_snapshot);
  const auto current_activity_offset =
    static_cast<std::size_t>(
      sbs_scene_controller::analysis_grid_channel_e::
        temporal_activity_occupancy
    ) * analysis_pixels;
  EXPECT_LE(
    *std::max_element(
      paused_analysis.begin() +
        static_cast<std::ptrdiff_t>(current_activity_offset),
      paused_analysis.begin() +
        static_cast<std::ptrdiff_t>(
          current_activity_offset + analysis_pixels
        )
    ),
    1e-6f
  );

  // Fresh contained playback after the history-only pause refreshes the incumbent instead of
  // creating a challenger or moving its convergence envelope.
  const auto resumed = make_frame(4u, 7u);
  ASSERT_TRUE(resumed);
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(resumed.Get(), frame_id++));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::
        seconds_since_layout_evidence
    ),
    0.0f
  );
  const std::array resumed_bounds {
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    ),
  };
  EXPECT_EQ(resumed_bounds, committed);

  // A proven player may move. Similar-or-larger motion at low overlap must first challenge the
  // retained rectangle and then replace it; otherwise the indefinite pause contract would make
  // a moved player permanently unreachable.
  const auto make_relocated =
    [&](const std::uint32_t relocated_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          adversarial_source_region_t {
            0.40f, 0.18f, 0.96f, 0.78f, relocated_phase, false,
          },
        }
      );
    };
  bool saw_relocation_challenger = false;
  bool relocation_committed = false;
  for (std::uint32_t phase = 0u; phase < 12u; ++phase) {
    const auto relocated = make_relocated(phase);
    ASSERT_TRUE(relocated);
    controller->set_next_elapsed_seconds_for_testing(
      phase == 0u ? 0.10f : 0.25f
    );
    ASSERT_TRUE(run_source(relocated.Get(), frame_id++));
    state = read_rule_state(controller->snapshot());
    const bool challenger_active =
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::challenger_valid
      ) > 0.5f;
    if (challenger_active) {
      saw_relocation_challenger = true;
      EXPECT_EQ(
        state[sbs_scene_controller::index(
          sbs_scene_controller::rule_state_word_e::roi_generation
        )],
        committed_generation
      );
    }
    relocation_committed =
      state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::roi_generation
      )] == committed_generation + 1u;
    if (relocation_committed) {
      break;
    }
  }
  EXPECT_TRUE(saw_relocation_challenger);
  ASSERT_TRUE(relocation_committed);
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::challenger_valid
    ),
    0.0f
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    committed_generation + 1u
  );
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    ),
    0.34f
  );
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ),
    0.90f
  );

  const auto expect_no_video_authority =
    [&](const std::vector<std::uint32_t> &cold_state) {
      const float primary_video = static_cast<float>(
        sbs_scene_controller::layout_decision_e::primary_video
      );
      EXPECT_NE(
        word_as_float(
          cold_state,
          sbs_scene_controller::rule_state_word_e::state_kind
        ),
        static_cast<float>(sbs_scene_controller::state_kind_e::video)
      );
      EXPECT_FALSE(
        word_as_float(
          cold_state,
          sbs_scene_controller::rule_state_word_e::acquisition_valid
        ) > 0.5f &&
        word_as_float(
          cold_state,
          sbs_scene_controller::rule_state_word_e::acquisition_layout
        ) == primary_video
      );
      EXPECT_FALSE(
        word_as_float(
          cold_state,
          sbs_scene_controller::rule_state_word_e::challenger_valid
        ) > 0.5f &&
        word_as_float(
          cold_state,
          sbs_scene_controller::rule_state_word_e::challenger_layout
        ) == primary_video
      );
      EXPECT_NE(
        word_as_float(
          cold_state,
          sbs_scene_controller::rule_state_word_e::committed_layout
        ),
        primary_video
      );
    };

  // Layout/geometry invalidation removes the historical right to call this rectangle video.
  // The exact same pixels are cold static media again and may become generic content only.
  controller->set_next_reset_flags_for_testing(
    sbs_scene_controller::reset_flags_layout |
      sbs_scene_controller::reset_flags_depth_shot |
      sbs_scene_controller::reset_flags_geometry
  );
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(resumed.Get(), frame_id++));
  state = read_rule_state(controller->snapshot());
  expect_no_video_authority(state);
  for (std::size_t index = 0u; index < 8u; ++index) {
    controller->set_next_elapsed_seconds_for_testing(0.25f);
    ASSERT_TRUE(run_source(resumed.Get(), frame_id++));
    state = read_rule_state(controller->snapshot());
    expect_no_video_authority(state);
  }
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::content_collage
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  ProvenVideoRemainsAuthoritativeBeyondReleaseDuringPauseExposureAndScroll
) {
  const auto make_frame =
    [&](const std::uint32_t player_phase,
        const std::uint32_t sidebar_phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          adversarial_source_region_t {
            0.12f, 0.18f, 0.68f, 0.78f, 0u, false,
          },
          adversarial_source_region_t {
            0.30f, 0.32f, 0.50f, 0.62f, player_phase, false,
          },
          adversarial_source_region_t {
            0.82f, 0.30f, 0.94f, 0.56f, sidebar_phase, true,
          },
        }
      );
    };
  const auto seed = make_frame(0u, 0u);
  const auto moving_1 = make_frame(1u, 0u);
  const auto moving_2 = make_frame(2u, 0u);
  const auto moving_3 = make_frame(3u, 0u);
  const auto outside_only = make_frame(3u, 8u);
  const auto resumed_1 = make_frame(4u, 8u);
  const auto resumed_2 = make_frame(5u, 8u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(moving_1);
  ASSERT_TRUE(moving_2);
  ASSERT_TRUE(moving_3);
  ASSERT_TRUE(outside_only);
  ASSERT_TRUE(resumed_1);
  ASSERT_TRUE(resumed_2);

  std::uint64_t frame_id = 9100u;
  const auto update =
    [&](ID3D11ShaderResourceView *source, const float elapsed) {
      controller->set_next_elapsed_seconds_for_testing(elapsed);
      return run_source(source, frame_id++);
    };
  ASSERT_TRUE(update(seed.Get(), 0.0f));
  ASSERT_TRUE(update(moving_1.Get(), 0.10f));
  ASSERT_TRUE(update(moving_2.Get(), 0.40f));
  ASSERT_TRUE(update(moving_3.Get(), 0.40f));

  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  ) << "layout="
    << word_as_float(
         state,
         sbs_scene_controller::rule_state_word_e::layout_decision
       )
    << " acquisition_valid="
    << word_as_float(
         state,
         sbs_scene_controller::rule_state_word_e::acquisition_valid
       )
    << " acquisition_layout="
    << word_as_float(
         state,
         sbs_scene_controller::rule_state_word_e::acquisition_layout
       )
    << " acquisition_dwell="
    << word_as_float(
         state,
         sbs_scene_controller::rule_state_word_e::acquisition_dwell_s
       )
    << " scroll="
    << word_as_float(
         state,
         sbs_scene_controller::rule_state_word_e::scroll_confidence
       );
  const std::array learned_bounds {
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    ),
  };
  const auto active_generation =
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )];
  ASSERT_GT(active_generation, 0u);
  ASSERT_LT(learned_bounds[2], 0.80f);

  // Quiet playback history remains useful indefinitely. The generic-content release ceiling
  // must not turn a proven paused player back into full-frame inference.
  ASSERT_TRUE(update(moving_3.Get(), 30.0f));
  ASSERT_TRUE(update(moving_3.Get(), 30.0f));
  ASSERT_TRUE(update(moving_3.Get(), 8.0f));
  state = read_rule_state(controller->snapshot());
  ASSERT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    active_generation
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::video
    )
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  EXPECT_NE(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_roi_locked,
    0u
  );
  EXPECT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::
        committed_roi_confidence
    ),
    0.0f
  );
  const auto expect_active =
    [&](const std::vector<std::uint32_t> &retained) {
      EXPECT_FLOAT_EQ(
        word_as_float(
          retained,
          sbs_scene_controller::rule_state_word_e::state_kind
        ),
        static_cast<float>(
          sbs_scene_controller::state_kind_e::video
        )
      );
      EXPECT_FLOAT_EQ(
        word_as_float(
          retained,
          sbs_scene_controller::rule_state_word_e::committed_layout
        ),
        static_cast<float>(
          sbs_scene_controller::layout_decision_e::primary_video
        )
      );
      EXPECT_EQ(
        retained[sbs_scene_controller::index(
          sbs_scene_controller::rule_state_word_e::roi_generation
        )],
        active_generation
      );
      EXPECT_NE(
        retained[sbs_scene_controller::index(
          sbs_scene_controller::rule_state_word_e::state_flags
        )] & sbs_scene_controller::state_flags_roi_locked,
        0u
      );
      EXPECT_FLOAT_EQ(
        word_as_float(
          retained,
          sbs_scene_controller::rule_state_word_e::acquisition_valid
        ),
        0.0f
      );
      EXPECT_EQ(
        (std::array {
          word_as_float(
            retained,
            sbs_scene_controller::rule_state_word_e::committed_roi_x0
          ),
          word_as_float(
            retained,
            sbs_scene_controller::rule_state_word_e::committed_roi_y0
          ),
          word_as_float(
            retained,
            sbs_scene_controller::rule_state_word_e::committed_roi_x1
          ),
          word_as_float(
            retained,
            sbs_scene_controller::rule_state_word_e::committed_roi_y1
          ),
        }),
        learned_bounds
      );
    };
  expect_active(state);

  // Activity wholly outside the learned player cannot relabel it as static content.
  ASSERT_TRUE(update(outside_only.Get(), 0.10f));
  expect_active(read_rule_state(controller->snapshot()));

  // A depth-shot reset does not change target coordinates and therefore preserves learned video.
  controller->set_next_reset_flags_for_testing(
    sbs_scene_controller::reset_flags_depth_shot
  );
  ASSERT_TRUE(update(outside_only.Get(), 0.10f));
  expect_active(read_rule_state(controller->snapshot()));

  // An exposure frame may replace the entire appearance, but cannot authorize geometry.
  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f,
    0u,
    false,
    sbs_adaptive_state::analysis_flag_exposure_like
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  const auto scroll_0 = make_scroll_source_view(
    device.Get(),
    320,
    180,
    0u
  );
  const auto scroll_5 = make_scroll_source_view(
    device.Get(),
    320,
    180,
    5u
  );
  const auto scroll_10 = make_scroll_source_view(
    device.Get(),
    320,
    180,
    10u
  );
  ASSERT_TRUE(scroll_0);
  ASSERT_TRUE(scroll_5);
  ASSERT_TRUE(scroll_10);
  ASSERT_TRUE(update(scroll_0.Get(), 0.10f));
  expect_active(read_rule_state(controller->snapshot()));

  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  ASSERT_TRUE(update(scroll_5.Get(), 0.08f));
  state = read_rule_state(controller->snapshot());
  ASSERT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  ASSERT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
  expect_active(state);

  ASSERT_TRUE(update(scroll_10.Get(), 0.05f));
  state = read_rule_state(controller->snapshot());
  ASSERT_GT(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::scroll_confidence
    ),
    0.0f
  );
  ASSERT_NE(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_scroll_hold_active,
    0u
  );
  expect_active(state);

  // The bounded scroll hold freezes decisions without revoking the learned player.
  ASSERT_TRUE(update(resumed_1.Get(), 0.05f));
  expect_active(read_rule_state(controller->snapshot()));
  ASSERT_TRUE(update(resumed_2.Get(), 0.08f));
  state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_NE(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] & sbs_scene_controller::state_flags_roi_locked,
    0u
  );
  EXPECT_EQ(
    state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )],
    active_generation
  );
  EXPECT_EQ(
    (std::array {
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_roi_x0
      ),
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_roi_y0
      ),
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_roi_x1
      ),
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_roi_y1
      ),
    }),
    learned_bounds
  );
}

TEST_F(
  SbsSceneControllerGpu,
  TargetChangingResetsDiscardOrdinalAndLayoutActivityHistory
) {
  const auto seed = make_player_source_view(
    device.Get(),
    320,
    180,
    0u
  );
  const auto moving_1 = make_player_source_view(
    device.Get(),
    320,
    180,
    1u
  );
  const auto moving_2 = make_player_source_view(
    device.Get(),
    320,
    180,
    2u
  );
  const auto moving_3 = make_player_source_view(
    device.Get(),
    320,
    180,
    3u
  );
  ASSERT_TRUE(seed);
  ASSERT_TRUE(moving_1);
  ASSERT_TRUE(moving_2);
  ASSERT_TRUE(moving_3);

  struct reset_case_t {
    const char *label;
    std::uint32_t flag;
  };
  constexpr std::array reset_cases {
    reset_case_t {
      "layout",
      sbs_scene_controller::reset_flags_layout,
    },
    reset_case_t {
      "geometry",
      sbs_scene_controller::reset_flags_geometry,
    },
    reset_case_t {
      "backend",
      sbs_scene_controller::reset_flags_backend,
    },
    reset_case_t {
      "display_or_hdr",
      sbs_scene_controller::reset_flags_display_or_hdr,
    },
  };
  constexpr std::size_t plane =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  const auto channel_max =
    [](const std::vector<float> &values,
       const std::size_t channel,
       const std::size_t channel_plane) {
      const auto first =
        values.begin() +
        static_cast<std::ptrdiff_t>(channel * channel_plane);
      return *std::max_element(
        first,
        first + static_cast<std::ptrdiff_t>(channel_plane)
      );
    };
  const std::filesystem::path assets_dir =
    std::filesystem::path(SUNSHINE_SHADERS_DIR)
      .parent_path()
      .parent_path();
  std::uint64_t frame_id = 8800u;

  for (const auto &reset_case : reset_cases) {
    SCOPED_TRACE(reset_case.label);
    config::video_t::sbs_t sbs;
    sbs.scene_controller =
      config::sbs_scene_controller_e::shadow_rules;
    controller = std::make_unique<models::sbs_scene_controller_gpu>(
      device,
      context,
      assets_dir,
      sbs.scene_controller,
      sbs
    );
    ASSERT_TRUE(controller->enabled());
    ASSERT_TRUE(controller->valid());

    controller->set_next_elapsed_seconds_for_testing(0.0f);
    ASSERT_TRUE(run_source(seed.Get(), frame_id++));
    controller->set_next_elapsed_seconds_for_testing(0.10f);
    ASSERT_TRUE(run_source(moving_1.Get(), frame_id++));
    controller->set_next_elapsed_seconds_for_testing(0.40f);
    ASSERT_TRUE(run_source(moving_2.Get(), frame_id++));
    controller->set_next_elapsed_seconds_for_testing(0.40f);
    ASSERT_TRUE(run_source(moving_3.Get(), frame_id++));

    auto state = read_rule_state(controller->snapshot());
    ASSERT_FLOAT_EQ(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ),
      static_cast<float>(sbs_scene_controller::state_kind_e::video)
    );
    ASSERT_FLOAT_EQ(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_layout
      ),
      static_cast<float>(
        sbs_scene_controller::layout_decision_e::primary_video
      )
    );

    // The reset frame is byte-for-byte identical to the previous frame. None of its temporal
    // authority may survive a reset that changes the target coordinate system or backend.
    controller->set_next_reset_flags_for_testing(reset_case.flag);
    controller->set_next_elapsed_seconds_for_testing(0.10f);
    ASSERT_TRUE(run_source(moving_3.Get(), frame_id++));
    const auto reset_snapshot = controller->snapshot();
    state = read_rule_state(reset_snapshot);
    const float primary_video = static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    );
    EXPECT_NE(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ),
      static_cast<float>(sbs_scene_controller::state_kind_e::video)
    );
    EXPECT_FALSE(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_valid
      ) > 0.5f &&
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_layout
      ) == primary_video
    );
    EXPECT_FALSE(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::challenger_valid
      ) > 0.5f &&
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::challenger_layout
      ) == primary_video
    );
    EXPECT_NE(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_layout
      ),
      primary_video
    );

    const auto analysis = read_analysis(reset_snapshot);
    EXPECT_FLOAT_EQ(
      channel_max(
        analysis,
        static_cast<std::size_t>(
          sbs_scene_controller::analysis_grid_channel_e::
            temporal_activity_occupancy
        ),
        plane
      ),
      0.0f
    );
    const auto layout_history = read_buffer<float>(
      device.Get(),
      context.Get(),
      reset_snapshot.layout_history.Get(),
      sbs_scene_controller::layout_history_channel_count * plane
    );
    EXPECT_FLOAT_EQ(
      channel_max(
        layout_history,
        static_cast<std::size_t>(
          sbs_scene_controller::layout_history_channel_e::
            fast_activity_ema
        ),
        plane
      ),
      0.0f
    );
    EXPECT_FLOAT_EQ(
      channel_max(
        layout_history,
        static_cast<std::size_t>(
          sbs_scene_controller::layout_history_channel_e::
            slow_activity_ema
        ),
        plane
      ),
      0.0f
    );
    EXPECT_FLOAT_EQ(
      channel_max(
        layout_history,
        static_cast<std::size_t>(
          sbs_scene_controller::layout_history_channel_e::
            seconds_since_valid_layout
        ),
        plane
      ),
      0.0f
    );
    const auto dense = read_buffer<float>(
      device.Get(),
      context.Get(),
      reset_snapshot.dense_output.Get(),
      sbs_scene_controller::dense_out_channel_count * plane
    );
    EXPECT_FLOAT_EQ(
      channel_max(
        dense,
        static_cast<std::size_t>(
          sbs_scene_controller::dense_out_channel_e::
            primary_playing_video
        ),
        plane
      ),
      0.0f
    );
  }
}

TEST_F(
  SbsSceneControllerGpu,
  DepthShotResetPreservesVideoGeometryAndOrdinalLayoutHistory
) {
  const auto seed = make_player_source_view(
    device.Get(),
    320,
    180,
    0u
  );
  const auto moving_1 = make_player_source_view(
    device.Get(),
    320,
    180,
    1u
  );
  const auto moving_2 = make_player_source_view(
    device.Get(),
    320,
    180,
    2u
  );
  const auto moving_3 = make_player_source_view(
    device.Get(),
    320,
    180,
    3u
  );
  const auto moving_4 = make_player_source_view(
    device.Get(),
    320,
    180,
    4u
  );
  ASSERT_TRUE(seed);
  ASSERT_TRUE(moving_1);
  ASSERT_TRUE(moving_2);
  ASSERT_TRUE(moving_3);
  ASSERT_TRUE(moving_4);

  std::uint64_t frame_id = 8850u;
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), frame_id++));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(moving_1.Get(), frame_id++));
  controller->set_next_elapsed_seconds_for_testing(0.40f);
  ASSERT_TRUE(run_source(moving_2.Get(), frame_id++));
  controller->set_next_elapsed_seconds_for_testing(0.40f);
  ASSERT_TRUE(run_source(moving_3.Get(), frame_id++));

  auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  const std::array committed_bounds {
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    ),
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    ),
  };

  controller->set_next_reset_flags_for_testing(
    sbs_scene_controller::reset_flags_depth_shot
  );
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(moving_4.Get(), frame_id++));
  const auto depth_shot_snapshot = controller->snapshot();
  state = read_rule_state(depth_shot_snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::video)
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::committed_layout
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
  EXPECT_EQ(
    (std::array {
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_roi_x0
      ),
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_roi_y0
      ),
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_roi_x1
      ),
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::committed_roi_y1
      ),
    }),
    committed_bounds
  );

  constexpr std::size_t plane =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  const auto analysis = read_analysis(depth_shot_snapshot);
  const auto activity_first =
    analysis.begin() +
    static_cast<std::ptrdiff_t>(
      static_cast<std::size_t>(
        sbs_scene_controller::analysis_grid_channel_e::
          temporal_activity_occupancy
      ) * plane
    );
  EXPECT_GT(
    *std::max_element(
      activity_first,
      activity_first + static_cast<std::ptrdiff_t>(plane)
    ),
    0.0f
  );
  const auto layout_history = read_buffer<float>(
    device.Get(),
    context.Get(),
    depth_shot_snapshot.layout_history.Get(),
    sbs_scene_controller::layout_history_channel_count * plane
  );
  const auto slow_first =
    layout_history.begin() +
    static_cast<std::ptrdiff_t>(
      static_cast<std::size_t>(
        sbs_scene_controller::layout_history_channel_e::
          slow_activity_ema
      ) * plane
    );
  EXPECT_GT(
    *std::max_element(
      slow_first,
      slow_first + static_cast<std::ptrdiff_t>(plane)
    ),
    0.0f
  );
  const auto invalid_layout_age_first =
    layout_history.begin() +
    static_cast<std::ptrdiff_t>(
      static_cast<std::size_t>(
        sbs_scene_controller::layout_history_channel_e::
          seconds_since_valid_layout
      ) * plane
    );
  EXPECT_GT(
    *std::max_element(
      invalid_layout_age_first,
      invalid_layout_age_first + static_cast<std::ptrdiff_t>(plane)
    ),
    0.9f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  IncoherentTemporalAxesAbstainAsAmbiguousWithoutAcquiring
) {
  const auto make_frame =
    [&](const std::uint32_t phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          // This strip can cross the row threshold but is too short for a column run.
          adversarial_source_region_t {
            0.10f, 0.15f, 0.58f, 0.22f, phase, false,
          },
          // This strip can cross the column threshold but is too narrow for a row run.
          adversarial_source_region_t {
            0.78f, 0.48f, 0.85f, 0.90f, phase, false,
          },
        }
      );
    };
  const auto seed = make_frame(0u);
  const auto changed = make_frame(1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 878u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 879u));

  const auto state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::ambiguous
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  DenseAchromaticAnimatedTerminalNeverAcquiresVideoRoi
) {
  // These wide, rapidly changing grayscale bands occupy one large
  // terminal-like rectangle. They provide abundant ordinal activity, but
  // their separated text rows provide neither chroma nor sustained static
  // media support and must not be promoted into an achromatic "video."
  const auto make_frame =
    [&](const std::uint32_t phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          adversarial_source_region_t {
            0.12f, 0.10f, 0.86f, 0.13f, phase, false,
          },
          adversarial_source_region_t {
            0.14f, 0.16f, 0.80f, 0.19f, phase, false,
          },
          adversarial_source_region_t {
            0.12f, 0.22f, 0.88f, 0.25f, phase, false,
          },
          adversarial_source_region_t {
            0.14f, 0.28f, 0.83f, 0.31f, phase, false,
          },
          adversarial_source_region_t {
            0.12f, 0.34f, 0.86f, 0.37f, phase, false,
          },
          adversarial_source_region_t {
            0.14f, 0.40f, 0.79f, 0.43f, phase, false,
          },
          adversarial_source_region_t {
            0.12f, 0.46f, 0.88f, 0.49f, phase, false,
          },
          adversarial_source_region_t {
            0.14f, 0.52f, 0.82f, 0.55f, phase, false,
          },
          adversarial_source_region_t {
            0.12f, 0.58f, 0.86f, 0.61f, phase, false,
          },
          adversarial_source_region_t {
            0.14f, 0.64f, 0.78f, 0.67f, phase, false,
          },
          adversarial_source_region_t {
            0.12f, 0.70f, 0.88f, 0.73f, phase, false,
          },
          adversarial_source_region_t {
            0.14f, 0.76f, 0.82f, 0.79f, phase, false,
          },
          adversarial_source_region_t {
            0.12f, 0.82f, 0.86f, 0.85f, phase, false,
          },
        }
      );
    };

  std::uint64_t frame_id = 9200u;
  const auto seed = make_frame(0u);
  ASSERT_TRUE(seed);
  controller->set_next_reset_flags_for_testing(
    sbs_scene_controller::reset_flags_geometry
  );
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), frame_id++));

  for (std::uint32_t phase = 1u; phase <= 16u; ++phase) {
    SCOPED_TRACE(phase);
    const auto frame = make_frame(phase);
    ASSERT_TRUE(frame);
    controller->set_next_elapsed_seconds_for_testing(0.10f);
    ASSERT_TRUE(run_source(frame.Get(), frame_id++));
    const auto state = read_rule_state(controller->snapshot());
    EXPECT_FLOAT_EQ(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::state_kind
      ),
      static_cast<float>(
        sbs_scene_controller::state_kind_e::full_frame
      )
    );
    EXPECT_FLOAT_EQ(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::acquisition_valid
      ),
      0.0f
    );
    EXPECT_NE(
      word_as_float(
        state,
        sbs_scene_controller::rule_state_word_e::layout_decision
      ),
      static_cast<float>(
        sbs_scene_controller::layout_decision_e::primary_video
      )
    );
    EXPECT_EQ(
      state[sbs_scene_controller::index(
        sbs_scene_controller::rule_state_word_e::state_flags
      )] & sbs_scene_controller::state_flags_roi_locked,
      0u
    );
  }

  constexpr std::size_t plane =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  const auto snapshot = controller->snapshot();
  const auto analysis = read_analysis(snapshot);
  const auto dense = read_buffer<float>(
    device.Get(),
    context.Get(),
    snapshot.dense_output.Get(),
    sbs_scene_controller::dense_out_channel_count * plane
  );
  const auto channel_offset =
    [](const auto channel) {
      return static_cast<std::size_t>(channel) * plane;
    };
  const auto activity_offset = channel_offset(
    sbs_scene_controller::analysis_grid_channel_e::
      temporal_activity_occupancy
  );
  const auto chroma_offset = channel_offset(
    sbs_scene_controller::analysis_grid_channel_e::chroma_saturation
  );
  const auto static_media_offset = channel_offset(
    sbs_scene_controller::dense_out_channel_e::
      photographic_content_collage
  );
  EXPECT_GT(
    std::count_if(
      analysis.begin() + activity_offset,
      analysis.begin() + activity_offset + plane,
      [](const float activity) {
        return activity > 0.0f;
      }
    ),
    512
  ) << "the fixture must retain dense raw ordinal activity";
  EXPECT_FLOAT_EQ(
    *std::max_element(
      analysis.begin() + chroma_offset,
      analysis.begin() + chroma_offset + plane
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    *std::max_element(
      dense.begin() + static_media_offset,
      dense.begin() + static_media_offset + plane
    ),
    0.0f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  DisjointTemporalIslandsCannotCrossProjectIntoAPlayerRectangle
) {
  const auto make_frame =
    [&](const std::uint32_t phase) {
      return make_adversarial_source_view(
        device.Get(),
        320,
        180,
        std::array {
          adversarial_source_region_t {
            0.10f,
            0.15f,
            0.38f,
            0.42f,
            phase,
            false,
          },
          adversarial_source_region_t {
            0.62f,
            0.58f,
            0.90f,
            0.85f,
            phase,
            false,
          },
        }
      );
    };
  const auto seed = make_frame(0u);
  const auto changed = make_frame(1u);
  ASSERT_TRUE(seed);
  ASSERT_TRUE(changed);

  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(seed.Get(), 860u));
  controller->set_next_elapsed_seconds_for_testing(0.10f);
  ASSERT_TRUE(run_source(changed.Get(), 861u));

  constexpr std::size_t plane =
    sbs_scene_controller::analysis_canvas_size *
    sbs_scene_controller::analysis_canvas_size;
  const auto analysis = read_analysis(controller->snapshot());
  const auto activity_first =
    analysis.begin() +
    static_cast<std::size_t>(
      sbs_scene_controller::analysis_grid_channel_e::
        temporal_activity_occupancy
    ) *
      plane;
  EXPECT_GT(
    std::count_if(
      activity_first,
      activity_first + plane,
      [](const float value) {
        return value >= 3.0f / 8.0f;
      }
    ),
    256
  );

  const auto state = read_rule_state(controller->snapshot());
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::acquisition_valid
    ),
    0.0f
  );
  EXPECT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::full_frame)
  );
  EXPECT_NE(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::layout_decision
    ),
    static_cast<float>(
      sbs_scene_controller::layout_decision_e::primary_video
    )
  );
}

TEST_F(
  SbsSceneControllerGpu,
  RobustMediaHullRejectsDetachedOutlierAndPreservesSupportedEdges
) {
  const auto committed_bounds =
    [&](const std::vector<std::uint32_t> &state) {
      return std::array {
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::committed_roi_x0
        ),
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::committed_roi_y0
        ),
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::committed_roi_x1
        ),
        word_as_float(
          state,
          sbs_scene_controller::rule_state_word_e::committed_roi_y1
        ),
      };
    };
  const auto acquire_content =
    [&](ID3D11ShaderResourceView *source, std::uint64_t frame_id) {
      controller->set_next_elapsed_seconds_for_testing(0.0f);
      EXPECT_TRUE(run_source(source, frame_id));
      controller->set_next_elapsed_seconds_for_testing(0.71f);
      EXPECT_TRUE(run_source(source, frame_id + 1u));
      return read_rule_state(controller->snapshot());
    };

  const auto separated_media_with_outlier =
    make_adversarial_source_view(
      device.Get(),
      320,
      180,
      std::array {
        adversarial_source_region_t {
          0.32f,
          0.22f,
          0.50f,
          0.72f,
          0u,
          true,
        },
        adversarial_source_region_t {
          0.58f,
          0.22f,
          0.76f,
          0.72f,
          0u,
          true,
        },
        adversarial_source_region_t {
          0.18f,
          0.40f,
          0.22f,
          0.48f,
          0u,
          true,
        },
      }
    );
  ASSERT_TRUE(separated_media_with_outlier);
  auto state = acquire_content(
    separated_media_with_outlier.Get(),
    880u
  );
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );
  auto bounds = committed_bounds(state);
  // Both substantial blocks own the common hull.
  EXPECT_LE(bounds[0], 0.32f);
  EXPECT_LE(bounds[1], 0.22f);
  EXPECT_GE(bounds[2], 0.76f);
  EXPECT_GE(bounds[3], 0.72f);
  // The detached low-mass cluster must not pull the left edge across the
  // empty gap.
  EXPECT_GT(bounds[0], 0.22f);
  EXPECT_LT(bounds[2], 0.80f);

  const auto uniform_media = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array {
      adversarial_source_region_t {
        0.23f,
        0.19f,
        0.71f,
        0.77f,
        0u,
        true,
      },
    }
  );
  ASSERT_TRUE(uniform_media);
  controller->set_next_reset_flags_for_testing(
    sbs_scene_controller::reset_flags_layout |
      sbs_scene_controller::reset_flags_depth_shot |
      sbs_scene_controller::reset_flags_geometry
  );
  state = acquire_content(uniform_media.Get(), 882u);
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );
  bounds = committed_bounds(state);
  // The 10% quantile is only a robust core. Continuous support must snap
  // back to the real media edge (plus the normal one-cell halo).
  EXPECT_LE(bounds[0], 0.23f);
  EXPECT_LE(bounds[1], 0.19f);
  EXPECT_GE(bounds[2], 0.71f);
  EXPECT_GE(bounds[3], 0.77f);
  EXPECT_GE(bounds[0], 0.19f);
  EXPECT_GE(bounds[1], 0.15f);
  EXPECT_LE(bounds[2], 0.75f);
  EXPECT_LE(bounds[3], 0.81f);
}

TEST_F(
  SbsSceneControllerGpu,
  ThinColoredTextCannotBridgeIntoTheStaticMediaHull
) {
  const auto source = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array {
      adversarial_source_region_t {
        0.02f, 0.10f, 0.80f, 0.12f, 0u, true,
      },
      adversarial_source_region_t {
        0.02f, 0.18f, 0.80f, 0.20f, 0u, true,
      },
      adversarial_source_region_t {
        0.02f, 0.34f, 0.80f, 0.36f, 0u, true,
      },
      adversarial_source_region_t {
        0.02f, 0.50f, 0.80f, 0.52f, 0u, true,
      },
      adversarial_source_region_t {
        0.02f, 0.66f, 0.80f, 0.68f, 0u, true,
      },
      adversarial_source_region_t {
        0.02f, 0.82f, 0.80f, 0.84f, 0u, true,
      },
      // A genuinely spatially sustained colorful image block. It is written
      // last so the thin line fixture cannot punch neutral gaps through it.
      adversarial_source_region_t {
        0.35f, 0.20f, 0.80f, 0.80f, 0u, true,
      },
    }
  );
  ASSERT_TRUE(source);

  controller->set_next_reset_flags_for_testing(
    sbs_scene_controller::reset_flags_layout |
      sbs_scene_controller::reset_flags_depth_shot |
      sbs_scene_controller::reset_flags_geometry
  );
  controller->set_next_elapsed_seconds_for_testing(0.0f);
  ASSERT_TRUE(run_source(source.Get(), 895u));
  controller->set_next_elapsed_seconds_for_testing(0.71f);
  ASSERT_TRUE(run_source(source.Get(), 896u));

  const auto state = read_rule_state(controller->snapshot());
  ASSERT_FLOAT_EQ(
    word_as_float(
      state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(sbs_scene_controller::state_kind_e::content)
  );
  const float x0 = word_as_float(
    state,
    sbs_scene_controller::rule_state_word_e::committed_roi_x0
  );
  const float y0 = word_as_float(
    state,
    sbs_scene_controller::rule_state_word_e::committed_roi_y0
  );
  const float x1 = word_as_float(
    state,
    sbs_scene_controller::rule_state_word_e::committed_roi_x1
  );
  const float y1 = word_as_float(
    state,
    sbs_scene_controller::rule_state_word_e::committed_roi_y1
  );
  EXPECT_GT(x0, 0.25f)
    << "thin text-like strokes must not bridge the media hull to the page edge";
  EXPECT_LE(x0, 0.35f);
  EXPECT_LE(y0, 0.20f);
  // Published bounds use integer analysis-cell edges. The last occupied cell center is below
  // the source block's 0.80 edge, so 102/128 is exact coverage; the production crop later adds
  // its separate two-cell inference halo.
  constexpr float expected_quantized_x1 =
    std::floor(
      0.80f *
      static_cast<float>(sbs_scene_controller::analysis_canvas_size)
    ) /
    static_cast<float>(sbs_scene_controller::analysis_canvas_size);
  EXPECT_GE(x1, expected_quantized_x1);
  EXPECT_GE(y1, 0.80f);
}

TEST_F(
  SbsSceneControllerGpu,
  StaticMediaEvidenceIsOrientationNeutralAtEveryViewportEdge
) {
  struct media_sample_t {
    float media_mean = 0.0f;
    float ui_mean = 0.0f;
    float transient_mean = 0.0f;
    std::size_t cells = 0u;
  };

  constexpr std::size_t canvas =
    sbs_scene_controller::analysis_canvas_size;
  constexpr std::size_t plane = canvas * canvas;
  const auto channel_offset =
    [](const auto channel) {
      return static_cast<std::size_t>(channel) * plane;
    };
  std::uint64_t frame_id = 900u;
  const auto sample_region =
    [&](const UINT width,
        const UINT height,
        const adversarial_source_region_t &region) {
      const auto source = make_adversarial_source_view(
        device.Get(),
        width,
        height,
        std::array {region}
      );
      EXPECT_TRUE(source);
      if (!source) {
        return media_sample_t {};
      }

      controller->set_next_reset_flags_for_testing(
        sbs_scene_controller::reset_flags_layout |
          sbs_scene_controller::reset_flags_depth_shot |
          sbs_scene_controller::reset_flags_geometry
      );
      controller->set_next_elapsed_seconds_for_testing(0.0f);
      EXPECT_TRUE(run_source(source.Get(), frame_id++));
      controller->set_next_elapsed_seconds_for_testing(0.10f);
      EXPECT_TRUE(run_source(source.Get(), frame_id++));

      const auto snapshot = controller->snapshot();
      const auto analysis = read_analysis(snapshot);
      const auto dense = read_buffer<float>(
        device.Get(),
        context.Get(),
        snapshot.dense_output.Get(),
        sbs_scene_controller::dense_out_channel_count * plane
      );
      const std::size_t valid_offset = channel_offset(
        sbs_scene_controller::analysis_grid_channel_e::viewport_valid
      );
      const std::size_t x_offset = channel_offset(
        sbs_scene_controller::analysis_grid_channel_e::viewport_x
      );
      const std::size_t y_offset = channel_offset(
        sbs_scene_controller::analysis_grid_channel_e::viewport_y
      );
      const std::size_t media_offset = channel_offset(
        sbs_scene_controller::dense_out_channel_e::
          photographic_content_collage
      );
      const std::size_t ui_offset = channel_offset(
        sbs_scene_controller::dense_out_channel_e::browser_system_chrome
      );
      const std::size_t transient_offset = channel_offset(
        sbs_scene_controller::dense_out_channel_e::
          advertisement_competitor_unsafe
      );

      media_sample_t sample;
      for (std::size_t cell = 0u; cell < plane; ++cell) {
        if (analysis[valid_offset + cell] <= 0.5f) {
          continue;
        }
        const float x = analysis[x_offset + cell];
        const float y = analysis[y_offset + cell];
        if (
          x < region.x0 || x >= region.x1 ||
          y < region.y0 || y >= region.y1
        ) {
          continue;
        }
        EXPECT_TRUE(std::isfinite(dense[media_offset + cell]));
        EXPECT_GE(dense[media_offset + cell], 0.0f);
        EXPECT_LE(dense[media_offset + cell], 1.0f);
        sample.media_mean += dense[media_offset + cell];
        sample.ui_mean += dense[ui_offset + cell];
        sample.transient_mean += dense[transient_offset + cell];
        ++sample.cells;
      }
      EXPECT_GT(sample.cells, 0u);
      if (sample.cells != 0u) {
        const float reciprocal =
          1.0f / static_cast<float>(sample.cells);
        sample.media_mean *= reciprocal;
        sample.ui_mean *= reciprocal;
        sample.transient_mean *= reciprocal;
      }
      return sample;
    };

  // Equal-area textured media touches each viewport edge. Left/right are a
  // mirror pair and top/bottom are a vertical-flip pair; no edge is special.
  const std::array edge_regions {
    adversarial_source_region_t {0.00f, 0.25f, 0.40f, 0.75f, 0u, true},
    adversarial_source_region_t {0.60f, 0.25f, 1.00f, 0.75f, 0u, true},
    adversarial_source_region_t {0.30f, 0.00f, 0.70f, 0.50f, 0u, true},
    adversarial_source_region_t {0.30f, 0.50f, 0.70f, 1.00f, 0u, true},
  };
  std::array<media_sample_t, edge_regions.size()> edge_samples {};
  for (std::size_t edge = 0u; edge < edge_regions.size(); ++edge) {
    edge_samples[edge] = sample_region(256u, 256u, edge_regions[edge]);
    EXPECT_GT(edge_samples[edge].media_mean, 0.05f);
  }
  EXPECT_NEAR(
    edge_samples[0].media_mean,
    edge_samples[1].media_mean,
    0.03f
  );
  EXPECT_NEAR(
    edge_samples[2].media_mean,
    edge_samples[3].media_mean,
    0.03f
  );
  const auto edge_media_extrema = std::minmax_element(
    edge_samples.begin(),
    edge_samples.end(),
    [](const auto &left, const auto &right) {
      return left.media_mean < right.media_mean;
    }
  );
  EXPECT_LE(
    edge_media_extrema.second->media_mean -
      edge_media_extrema.first->media_mean,
    0.04f
  );

  // The same normalized content in landscape and portrait exercises different
  // canonical viewport shapes without changing the semantic weight.
  const adversarial_source_region_t centered {
    0.20f,
    0.25f,
    0.80f,
    0.75f,
    0u,
    true,
  };
  const auto landscape = sample_region(320u, 180u, centered);
  const auto portrait = sample_region(180u, 320u, centered);
  EXPECT_NEAR(landscape.media_mean, portrait.media_mean, 0.04f);
  EXPECT_NEAR(landscape.ui_mean, portrait.ui_mean, 0.04f);
  EXPECT_NEAR(
    landscape.transient_mean,
    portrait.transient_mean,
    0.04f
  );
}

TEST_F(
  SbsSceneControllerGpu,
  ParallelReducerMatchesSerialReferenceAcrossThresholdAdjacentScenes
) {
  struct scene_case_t {
    UINT gutter_width;
    std::uint8_t contrast;
    std::int8_t right_bias;
  };
  constexpr std::array cases {
    scene_case_t {0u, 2u, 0},
    scene_case_t {4u, 7u, 0},
    scene_case_t {6u, 8u, 0},
    scene_case_t {7u, 9u, 1},
    scene_case_t {8u, 10u, -1},
    scene_case_t {10u, 16u, 4},
    scene_case_t {12u, 32u, -4},
    scene_case_t {16u, 64u, 12},
  };

  config::video_t::sbs_t sbs;
  sbs.scene_controller =
    config::sbs_scene_controller_e::shadow_rules;
  const std::filesystem::path assets_dir =
    std::filesystem::path(SUNSHINE_SHADERS_DIR)
      .parent_path()
      .parent_path();
  const auto argmax =
    [](const std::vector<float> &values, std::size_t first, std::size_t end) {
      return static_cast<std::size_t>(std::distance(
        values.begin(),
        std::max_element(
          values.begin() + static_cast<std::ptrdiff_t>(first),
          values.begin() + static_cast<std::ptrdiff_t>(end)
        )
      ));
    };
  constexpr std::array categorical_state_words {
    sbs_scene_controller::rule_state_word_e::state_flags,
    sbs_scene_controller::rule_state_word_e::reset_flags,
    sbs_scene_controller::rule_state_word_e::roi_generation,
    sbs_scene_controller::rule_state_word_e::layout_decision,
    sbs_scene_controller::rule_state_word_e::event_decision,
    sbs_scene_controller::rule_state_word_e::pop_action,
    sbs_scene_controller::rule_state_word_e::zero_plane_decision,
    sbs_scene_controller::rule_state_word_e::rejection_reason,
    sbs_scene_controller::rule_state_word_e::promotion_flags,
    sbs_scene_controller::rule_state_word_e::history_flags,
    sbs_scene_controller::rule_state_word_e::diagnostic_flags,
    sbs_scene_controller::rule_state_word_e::last_external_cut_count,
    sbs_scene_controller::rule_state_word_e::output_valid,
  };
  constexpr std::array diagnostic_state_words {
    sbs_scene_controller::rule_state_word_e::roi_structural_cut_support,
    sbs_scene_controller::rule_state_word_e::roi_exposure_only_support,
    sbs_scene_controller::rule_state_word_e::roi_event_depth_coverage,
  };

  std::uint64_t frame_id = 1000u;
  for (const auto &scene_case : cases) {
    SCOPED_TRACE(testing::Message()
                 << "gutter=" << scene_case.gutter_width
                 << " contrast=" << static_cast<int>(scene_case.contrast)
                 << " bias=" << static_cast<int>(scene_case.right_bias));
    const auto source = make_split_threshold_source_view(
      device.Get(),
      320,
      180,
      scene_case.gutter_width,
      scene_case.contrast,
      scene_case.right_bias
    );
    ASSERT_TRUE(source);

    models::sbs_scene_controller_gpu parallel(
      device,
      context,
      assets_dir,
      sbs.scene_controller,
      sbs,
      false
    );
    models::sbs_scene_controller_gpu reference(
      device,
      context,
      assets_dir,
      sbs.scene_controller,
      sbs,
      true
    );
    ASSERT_TRUE(parallel.valid());
    ASSERT_TRUE(reference.valid());
    const auto run =
      [&](models::sbs_scene_controller_gpu &target) {
        if (!target.prepare_scene(
              source.Get(),
              models::input_color_space::srgb,
              frame_id
            )) {
          return false;
        }
        target.mark_enqueued(frame_id);
        return target.resolve_completed(
          frame_id,
          raw_depth_buffer.srv.Get(),
          depth_view.Get(),
          frame_state_buffer.srv.Get(),
          adaptive_buffer.srv.Get(),
          depth_width,
          depth_height,
          frame_roi_transform_buffer.srv.Get()
        );
      };
    ASSERT_TRUE(run(parallel));
    ASSERT_TRUE(run(reference));

    const auto parallel_snapshot = parallel.snapshot();
    const auto reference_snapshot = reference.snapshot();
    const auto parallel_evidence = read_buffer<float>(
      device.Get(),
      context.Get(),
      parallel_snapshot.rule_evidence_for_testing.Get(),
      sbs_scene_controller::global_out_word_count
    );
    const auto reference_evidence = read_buffer<float>(
      device.Get(),
      context.Get(),
      reference_snapshot.rule_evidence_for_testing.Get(),
      sbs_scene_controller::global_out_word_count
    );
    ASSERT_EQ(parallel_evidence.size(), reference_evidence.size());
    EXPECT_EQ(argmax(parallel_evidence, 0u, 6u),
              argmax(reference_evidence, 0u, 6u));
    EXPECT_EQ(argmax(parallel_evidence, 6u, 12u),
              argmax(reference_evidence, 6u, 12u));
    EXPECT_EQ(argmax(parallel_evidence, 12u, 21u),
              argmax(reference_evidence, 12u, 21u));
    EXPECT_EQ(argmax(parallel_evidence, 21u, 25u),
              argmax(reference_evidence, 21u, 25u));
    for (std::size_t word = 0; word < parallel_evidence.size(); ++word) {
      EXPECT_NEAR(
        parallel_evidence[word],
        reference_evidence[word],
        3e-4f
      ) << sbs_scene_controller::global_out_names[word];
    }
    for (std::size_t word = 35u; word <= 38u; ++word) {
      EXPECT_FLOAT_EQ(parallel_evidence[word], reference_evidence[word])
        << sbs_scene_controller::global_out_names[word];
    }

    const auto parallel_state = read_rule_state(parallel_snapshot);
    const auto reference_state = read_rule_state(reference_snapshot);
    for (const auto word : categorical_state_words) {
      const auto index = sbs_scene_controller::index(word);
      EXPECT_EQ(parallel_state[index], reference_state[index])
        << "rule_state[" << index << "]";
    }
    for (const auto word : diagnostic_state_words) {
      const auto parallel_value = word_as_float(parallel_state, word);
      const auto reference_value = word_as_float(reference_state, word);
      EXPECT_NEAR(parallel_value, reference_value, 3e-4f)
        << sbs_scene_controller::rule_state_fields[
             sbs_scene_controller::index(word)
           ].name;
      EXPECT_GE(parallel_value, 0.0f);
      EXPECT_LE(parallel_value, 1.0f);
    }
    EXPECT_GT(
      word_as_float(
        parallel_state,
        sbs_scene_controller::rule_state_word_e::
          roi_event_depth_coverage
      ),
      0.0f
    );
    ++frame_id;
  }

  const auto content = make_content_source_view(device.Get(), 320, 180);
  const auto gutter_seed = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array {
      adversarial_source_region_t {
        0.12f, 0.18f, 0.68f, 0.78f, 0u, true,
      },
      adversarial_source_region_t {
        0.30f, 0.32f, 0.50f, 0.62f, 0u, false,
      },
      adversarial_source_region_t {
        0.84f, 0.36f, 0.90f, 0.48f, 0u, false,
      },
    }
  );
  const auto gutter_changed = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array {
      adversarial_source_region_t {
        0.12f, 0.18f, 0.68f, 0.78f, 0u, true,
      },
      adversarial_source_region_t {
        0.30f, 0.32f, 0.50f, 0.62f, 1u, false,
      },
      adversarial_source_region_t {
        0.84f, 0.36f, 0.90f, 0.48f, 1u, false,
      },
    }
  );
  const auto sidebar_seed = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array {
      adversarial_source_region_t {
        0.10f, 0.22f, 0.70f, 0.82f, 0u, true,
      },
      adversarial_source_region_t {
        0.24f, 0.34f, 0.58f, 0.72f, 0u, false,
      },
      adversarial_source_region_t {
        0.86f, 0.02f, 0.92f, 0.58f, 0u, false,
      },
    }
  );
  const auto sidebar_changed = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array {
      adversarial_source_region_t {
        0.10f, 0.22f, 0.70f, 0.82f, 0u, true,
      },
      adversarial_source_region_t {
        0.24f, 0.34f, 0.58f, 0.72f, 1u, false,
      },
      adversarial_source_region_t {
        0.86f, 0.02f, 0.92f, 0.58f, 1u, false,
      },
    }
  );
  const auto striped_page_seed = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array {
      adversarial_source_region_t {
        0.08f, 0.02f, 0.70f, 0.055f, 0u, true,
      },
      adversarial_source_region_t {
        0.08f, 0.075f, 0.70f, 0.11f, 0u, true,
      },
      adversarial_source_region_t {
        0.08f, 0.13f, 0.70f, 0.165f, 0u, true,
      },
      adversarial_source_region_t {
        0.08f, 0.18f, 0.70f, 0.82f, 0u, true,
      },
      adversarial_source_region_t {
        0.16f, 0.25f, 0.62f, 0.75f, 0u, false,
      },
      adversarial_source_region_t {
        0.84f, 0.36f, 0.90f, 0.48f, 0u, false,
      },
    }
  );
  const auto striped_page_changed = make_adversarial_source_view(
    device.Get(),
    320,
    180,
    std::array {
      adversarial_source_region_t {
        0.08f, 0.02f, 0.70f, 0.055f, 0u, true,
      },
      adversarial_source_region_t {
        0.08f, 0.075f, 0.70f, 0.11f, 0u, true,
      },
      adversarial_source_region_t {
        0.08f, 0.13f, 0.70f, 0.165f, 0u, true,
      },
      adversarial_source_region_t {
        0.08f, 0.18f, 0.70f, 0.82f, 0u, true,
      },
      adversarial_source_region_t {
        0.16f, 0.25f, 0.62f, 0.75f, 1u, false,
      },
      adversarial_source_region_t {
        0.84f, 0.36f, 0.90f, 0.48f, 1u, false,
      },
    }
  );
  const auto scroll_0 =
    make_scroll_source_view(device.Get(), 320, 180, 0);
  const auto scroll_6 =
    make_scroll_source_view(device.Get(), 320, 180, 6);
  const auto invalid = make_nan_source_view(device.Get(), 320, 180);
  ASSERT_TRUE(content);
  ASSERT_TRUE(gutter_seed);
  ASSERT_TRUE(gutter_changed);
  ASSERT_TRUE(sidebar_seed);
  ASSERT_TRUE(sidebar_changed);
  ASSERT_TRUE(striped_page_seed);
  ASSERT_TRUE(striped_page_changed);
  ASSERT_TRUE(scroll_0);
  ASSERT_TRUE(scroll_6);
  ASSERT_TRUE(invalid);
  struct sequence_frame_t {
    std::string_view label;
    ComPtr<ID3D11ShaderResourceView> source;
    float elapsed_seconds;
  };
  const std::array sequence {
    sequence_frame_t {"sidebar_seed", sidebar_seed, 0.0f},
    sequence_frame_t {"sidebar_changed", sidebar_changed, 0.10f},
    sequence_frame_t {"gutter_seed", gutter_seed, 0.0f},
    sequence_frame_t {"gutter_changed", gutter_changed, 0.10f},
    sequence_frame_t {"striped_page_seed", striped_page_seed, 0.0f},
    sequence_frame_t {
      "striped_page_changed",
      striped_page_changed,
      0.10f,
    },
    sequence_frame_t {"content_seed", content, 0.0f},
    sequence_frame_t {"content_acquire", content, 0.16f},
    sequence_frame_t {"scroll_seed", scroll_0, 0.016f},
    sequence_frame_t {"scroll_transition", scroll_6, 0.08f},
    sequence_frame_t {"invalid_hold", invalid, 0.016f},
    sequence_frame_t {"content_recovery", content, 0.016f},
  };
  models::sbs_scene_controller_gpu parallel(
    device,
    context,
    assets_dir,
    sbs.scene_controller,
    sbs,
    false
  );
  models::sbs_scene_controller_gpu reference(
    device,
    context,
    assets_dir,
    sbs.scene_controller,
    sbs,
    true
  );
  ASSERT_TRUE(parallel.valid());
  ASSERT_TRUE(reference.valid());
  for (const auto &sequence_frame : sequence) {
    SCOPED_TRACE(sequence_frame.label);
    const auto run =
      [&](models::sbs_scene_controller_gpu &target) {
        target.set_next_elapsed_seconds_for_testing(
          sequence_frame.elapsed_seconds
        );
        if (!target.prepare_scene(
              sequence_frame.source.Get(),
              models::input_color_space::srgb,
              frame_id
            )) {
          return false;
        }
        target.mark_enqueued(frame_id);
        return target.resolve_completed(
          frame_id,
          raw_depth_buffer.srv.Get(),
          depth_view.Get(),
          frame_state_buffer.srv.Get(),
          adaptive_buffer.srv.Get(),
          depth_width,
          depth_height,
          frame_roi_transform_buffer.srv.Get()
        );
      };
    ASSERT_TRUE(run(parallel));
    ASSERT_TRUE(run(reference));
    const auto parallel_snapshot = parallel.snapshot();
    const auto reference_snapshot = reference.snapshot();
    const auto parallel_evidence = read_buffer<float>(
      device.Get(),
      context.Get(),
      parallel_snapshot.rule_evidence_for_testing.Get(),
      sbs_scene_controller::global_out_word_count
    );
    const auto reference_evidence = read_buffer<float>(
      device.Get(),
      context.Get(),
      reference_snapshot.rule_evidence_for_testing.Get(),
      sbs_scene_controller::global_out_word_count
    );
    ASSERT_EQ(parallel_evidence.size(), reference_evidence.size());
    EXPECT_EQ(argmax(parallel_evidence, 0u, 6u),
              argmax(reference_evidence, 0u, 6u));
    EXPECT_EQ(argmax(parallel_evidence, 6u, 12u),
              argmax(reference_evidence, 6u, 12u));
    EXPECT_EQ(argmax(parallel_evidence, 12u, 21u),
              argmax(reference_evidence, 12u, 21u));
    EXPECT_EQ(argmax(parallel_evidence, 21u, 25u),
              argmax(reference_evidence, 21u, 25u));
    for (std::size_t word = 0; word < parallel_evidence.size(); ++word) {
      EXPECT_NEAR(
        parallel_evidence[word],
        reference_evidence[word],
        3e-4f
      ) << sbs_scene_controller::global_out_names[word];
    }
    const auto parallel_state = read_rule_state(parallel_snapshot);
    const auto reference_state = read_rule_state(reference_snapshot);
    for (const auto word : categorical_state_words) {
      const auto index = sbs_scene_controller::index(word);
      EXPECT_EQ(parallel_state[index], reference_state[index])
        << "rule_state[" << index << "]";
    }
    for (const auto word : diagnostic_state_words) {
      const auto parallel_value = word_as_float(parallel_state, word);
      const auto reference_value = word_as_float(reference_state, word);
      EXPECT_NEAR(parallel_value, reference_value, 3e-4f)
        << sbs_scene_controller::rule_state_fields[
             sbs_scene_controller::index(word)
           ].name;
      if (sequence_frame.label == "invalid_hold") {
        EXPECT_FLOAT_EQ(parallel_value, 0.0f);
        EXPECT_FLOAT_EQ(reference_value, 0.0f);
      }
    }
    if (sequence_frame.label == "sidebar_changed") {
      ASSERT_FLOAT_EQ(
        word_as_float(
          parallel_state,
          sbs_scene_controller::rule_state_word_e::acquisition_valid
        ),
        1.0f
      );
      EXPECT_GT(
        word_as_float(
          parallel_state,
          sbs_scene_controller::rule_state_word_e::acquisition_roi_y0
        ),
        0.14f
      );
      EXPECT_LT(
        word_as_float(
          parallel_state,
          sbs_scene_controller::rule_state_word_e::acquisition_roi_x1
        ),
        0.78f
      );
    }
    if (sequence_frame.label == "invalid_hold") {
      EXPECT_FLOAT_EQ(
        word_as_float(
          parallel_state,
          sbs_scene_controller::rule_state_word_e::output_valid
        ),
        0.0f
      );
      EXPECT_FLOAT_EQ(
        word_as_float(
          reference_state,
          sbs_scene_controller::rule_state_word_e::output_valid
        ),
        0.0f
      );
    }
    ++frame_id;
  }
}

TEST_F(
  SbsSceneControllerGpu,
  DisjointCrossAxisEvidenceAlwaysAbstains
) {
  constexpr std::uint32_t canvas =
    sbs_scene_controller::analysis_canvas_size;
  constexpr std::size_t summary_float_count =
    SBS_RULE_SUMMARY_FLOAT_COUNT;
  struct alignas(16) scene_constants_t {
    std::uint32_t source_width = canvas;
    std::uint32_t source_height = canvas;
    std::uint32_t depth_width = canvas;
    std::uint32_t depth_height = canvas;
    std::uint32_t color_mode = 0u;
    std::uint32_t backend_generation = 1u;
    std::uint32_t history_valid = 1u;
    std::uint32_t reset_flags = 0u;
    float elapsed_seconds = 0.10f;
    float pop_floor = 1.2f;
    float pop_ceiling = 2.0f;
    float zero_plane_mode = 2.0f;
    float acquire_seconds = 0.12f;
    float challenger_seconds = 0.30f;
    float release_seconds = 60.0f;
    float scroll_enter_seconds = 0.05f;
    std::uint32_t source_frame_id_low = 1u;
    std::uint32_t source_frame_id_high = 0u;
    std::uint32_t identity_reserved_0 = 0u;
    std::uint32_t identity_reserved_1 = 0u;
    std::array<std::uint32_t, 8> ordered_abi_hash_words =
      sbs_scene_controller::ordered_abi_hash_words;
  };
  static_assert(sizeof(scene_constants_t) == 112u);

  struct route_rectangle_t {
    std::uint32_t x0;
    std::uint32_t y0;
    std::uint32_t x1;
    std::uint32_t y1;
  };
  struct cross_case_t {
    std::string_view label;
    route_rectangle_t route_a;
    route_rectangle_t route_b;
  };
  constexpr std::array cases {
    cross_case_t {
      "route_a_wide_route_b_tall",
      {20u, 35u, 100u, 85u},
      {40u, 15u, 80u, 110u},
    },
    cross_case_t {
      "route_b_wide_route_a_tall",
      {40u, 15u, 80u, 110u},
      {20u, 35u, 100u, 85u},
    },
    cross_case_t {
      "corner_partial_overlap",
      {20u, 20u, 80u, 70u},
      {60u, 50u, 110u, 105u},
    },
  };

  const auto reduce_shader = compile_compute_shader(
    device.Get(),
    "sbs_scene_rules_reduce_cs.hlsl"
  );
  const auto serial_shader = compile_compute_shader(
    device.Get(),
    "sbs_scene_rules_reduce_serial_reference_cs.hlsl"
  );
  ASSERT_TRUE(reduce_shader);
  ASSERT_TRUE(serial_shader);

  sbs_adaptive_state::words_t adaptive_words {};
  for (std::size_t word = 0u; word < adaptive_words.size(); ++word) {
    adaptive_words[word] = std::bit_cast<std::uint32_t>(
      sbs_adaptive_state::initial_values[word]
    );
  }
  adaptive_words[sbs_adaptive_state::index(
    sbs_adaptive_state::word_e::valid_depth_fraction
  )] = std::bit_cast<std::uint32_t>(1.0f);
  const auto adaptive_buffer = make_structured_buffer(
    device.Get(),
    adaptive_words.data(),
    sizeof(adaptive_words),
    sizeof(float) * 4u
  );
  std::array<float, sbs_scene_controller::meta_word_count> meta {};
  meta[static_cast<std::size_t>(
    sbs_scene_controller::meta_word_e::elapsed_seconds
  )] = 0.10f;
  meta[static_cast<std::size_t>(
    sbs_scene_controller::meta_word_e::source_aspect_ratio
  )] = 1.0f;
  meta[static_cast<std::size_t>(
    sbs_scene_controller::meta_word_e::depth_input_valid
  )] = 1.0f;
  meta[static_cast<std::size_t>(
    sbs_scene_controller::meta_word_e::pop_floor
  )] = 1.2f;
  meta[static_cast<std::size_t>(
    sbs_scene_controller::meta_word_e::pop_ceiling
  )] = 2.0f;
  const auto meta_buffer = make_structured_buffer(
    device.Get(),
    meta.data(),
    sizeof(meta),
    sizeof(float)
  );
  auto previous_state = sbs_scene_controller::initial_word_bits;
  previous_state[sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::output_valid
  )] = std::bit_cast<std::uint32_t>(1.0f);
  previous_state[sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::backend_generation
  )] = 1u;
  previous_state[sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::state_flags
  )] =
    sbs_scene_controller::state_flags_initialized |
    sbs_scene_controller::state_flags_layout_history_valid;
  previous_state[sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::reset_flags
  )] = 0u;
  const auto previous_state_buffer = make_structured_buffer(
    device.Get(),
    previous_state.data(),
    sizeof(previous_state),
    sizeof(std::uint32_t) * 4u
  );
  const scene_constants_t constants;
  const auto constant_buffer = make_constant_buffer(
    device.Get(),
    &constants,
    sizeof(constants)
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  ASSERT_TRUE(meta_buffer.srv);
  ASSERT_TRUE(previous_state_buffer.srv);
  ASSERT_TRUE(constant_buffer);

  constexpr std::size_t temporal_row_active_mass = 0u;
  constexpr std::size_t event_fresh_mass =
    SBS_RULE_SUMMARY_EVENT_GROUP_FRESH_ACTIVITY_MASS;
  constexpr std::size_t event_fresh_inside_previous =
    SBS_RULE_SUMMARY_EVENT_GROUP_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET;
  constexpr std::size_t global_x_first = 0u;
  constexpr std::size_t global_y_first = 4u;
  constexpr std::size_t seed_x_first = 8u;
  constexpr std::size_t seed_x_end = 9u;
  constexpr std::size_t seed_y_first = 10u;
  constexpr std::size_t seed_y_end = 11u;
  constexpr std::size_t seed_valid = 12u;
  constexpr std::size_t global_rect_coherence = 13u;
  constexpr std::size_t global_rect_mass = 22u;
  constexpr std::size_t global_rect_fresh = 23u;
  constexpr std::size_t layout_primary_video = 1u;
  constexpr std::size_t layout_ambiguous = 4u;

  const auto in_rectangle =
    [](const route_rectangle_t &rectangle,
       const std::uint32_t x,
       const std::uint32_t y) {
      return
        x >= rectangle.x0 && x < rectangle.x1 &&
        y >= rectangle.y0 && y < rectangle.y1;
    };
  const auto attr_meta_index =
    [](const std::size_t field) {
      return
        static_cast<std::size_t>(
          SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE
        ) + field;
    };
  for (const auto &scene : cases) {
    SCOPED_TRACE(scene.label);
    std::vector<float> summary(summary_float_count, 0.0f);
    summary[
      SBS_RULE_SUMMARY_REDUCE_AGGREGATE_BASE
    ] = static_cast<float>(canvas * canvas);
    summary[
      SBS_RULE_SUMMARY_REDUCE_AGGREGATE_BASE + 1u
    ] = 0.0f;
    float total_mass = 0.0f;
    float global_rectangle_mass = 0.0f;
    std::array<float, 16> coherence_histogram {};
    for (std::uint32_t y = 0u; y < canvas; ++y) {
      float active_in_row = 0.0f;
      for (std::uint32_t x = 0u; x < canvas; ++x) {
        const bool active =
          in_rectangle(scene.route_a, x, y) ||
          in_rectangle(scene.route_b, x, y);
        active_in_row += active ? 1.0f : 0.0f;
        const bool in_global_rectangle =
          x >= scene.route_a.x0 && x < scene.route_a.x1 &&
          y >= scene.route_b.y0 && y < scene.route_b.y1;
        global_rectangle_mass +=
          active && in_global_rectangle ? 1.0f : 0.0f;
        if (active && in_global_rectangle) {
          const auto x_bin = std::min<std::uint32_t>(
            3u,
            ((x - scene.route_a.x0) * 4u) /
              (scene.route_a.x1 - scene.route_a.x0)
          );
          const auto y_bin = std::min<std::uint32_t>(
            3u,
            ((y - scene.route_b.y0) * 4u) /
              (scene.route_b.y1 - scene.route_b.y0)
          );
          coherence_histogram[x_bin * 4u + y_bin] += 1.0f;
        }
      }
      summary[
        SBS_RULE_SUMMARY_TEMPORAL_ROW_BASE +
        y * SBS_RULE_SUMMARY_TEMPORAL_ROW_STRIDE +
        temporal_row_active_mass
      ] = active_in_row;
      total_mass += active_in_row;
    }
    for (std::uint32_t x = 0u; x < canvas; ++x) {
      float active_in_column = 0.0f;
      for (std::uint32_t y = 0u; y < canvas; ++y) {
        active_in_column +=
          in_rectangle(scene.route_a, x, y) ||
              in_rectangle(scene.route_b, x, y) ?
            1.0f :
            0.0f;
      }
      summary[
        SBS_RULE_SUMMARY_COLUMN_BASE +
        static_cast<std::size_t>(x) *
          SBS_RULE_SUMMARY_COLUMN_STRIDE +
        2u
      ] = active_in_column;
    }

    const auto store_run =
      [&](const std::size_t first_field,
          const std::uint32_t first,
          const std::uint32_t end) {
        summary[attr_meta_index(first_field + 0u)] =
          static_cast<float>(first);
        summary[attr_meta_index(first_field + 1u)] =
          static_cast<float>(end);
        summary[attr_meta_index(first_field + 2u)] = 1.0f;
        summary[attr_meta_index(first_field + 3u)] = 0.0f;
      };
    store_run(
      global_x_first,
      scene.route_a.x0,
      scene.route_a.x1
    );
    store_run(
      global_y_first,
      scene.route_b.y0,
      scene.route_b.y1
    );
    summary[attr_meta_index(seed_x_first)] =
      static_cast<float>(scene.route_a.x0);
    summary[attr_meta_index(seed_x_end)] =
      static_cast<float>(scene.route_a.x1);
    summary[attr_meta_index(seed_y_first)] =
      static_cast<float>(scene.route_b.y0);
    summary[attr_meta_index(seed_y_end)] =
      static_cast<float>(scene.route_b.y1);
    summary[attr_meta_index(seed_valid)] = 1.0f;
    std::array<float, 4> x_marginal {};
    std::array<float, 4> y_marginal {};
    for (std::size_t x_bin = 0u; x_bin < 4u; ++x_bin) {
      for (std::size_t y_bin = 0u; y_bin < 4u; ++y_bin) {
        const auto mass = coherence_histogram[x_bin * 4u + y_bin];
        x_marginal[x_bin] += mass;
        y_marginal[y_bin] += mass;
      }
    }
    float total_variation = 1.0f;
    if (global_rectangle_mass > 0.0f) {
      total_variation = 0.0f;
      const auto inverse_total = 1.0f / global_rectangle_mass;
      for (std::size_t x_bin = 0u; x_bin < 4u; ++x_bin) {
        for (std::size_t y_bin = 0u; y_bin < 4u; ++y_bin) {
          const auto observed =
            coherence_histogram[x_bin * 4u + y_bin] *
            inverse_total;
          const auto separable =
            x_marginal[x_bin] *
            y_marginal[y_bin] *
            inverse_total *
            inverse_total;
          total_variation +=
            0.5f * std::abs(observed - separable);
        }
      }
    }
    const auto rectangle_coherence =
      std::clamp(1.0f - total_variation, 0.0f, 1.0f);
    const auto event_containment =
      total_mass > 0.0f ? global_rectangle_mass / total_mass : 0.0f;
    ASSERT_TRUE(
      rectangle_coherence < 0.85f ||
      event_containment < 0.72f
    ) << "the independent coherence and containment gates must jointly "
         "reject every cross-axis ordering";
    summary[attr_meta_index(global_rect_coherence)] =
      rectangle_coherence;
    summary[attr_meta_index(global_rect_mass)] =
      global_rectangle_mass;
    summary[attr_meta_index(global_rect_fresh)] =
      global_rectangle_mass;
    summary[SBS_RULE_SUMMARY_EVENT_SOURCE_MARKER] = 1.0f;
    summary[
      SBS_RULE_SUMMARY_EVENT_GROUP_BASE + event_fresh_mass
    ] = total_mass;
    summary[
      SBS_RULE_SUMMARY_EVENT_GROUP_BASE +
      event_fresh_inside_previous
    ] = 0.0f;

    constexpr std::size_t plane =
      static_cast<std::size_t>(canvas) * canvas;
    const auto plane_offset =
      [](const auto channel) {
        return static_cast<std::size_t>(channel) * plane;
      };
    std::vector<float> serial_analysis(
      sbs_scene_controller::analysis_grid_channel_count * plane,
      0.0f
    );
    std::vector<float> serial_dense(
      sbs_scene_controller::dense_out_channel_count * plane,
      0.0f
    );
    std::vector<float> serial_layout(
      sbs_scene_controller::layout_history_channel_count * plane,
      0.0f
    );
    std::vector<float> serial_depth(
      sbs_scene_controller::depth_history_channel_count * plane,
      0.0f
    );
    for (std::uint32_t y = 0u; y < canvas; ++y) {
      for (std::uint32_t x = 0u; x < canvas; ++x) {
        const auto cell =
          static_cast<std::size_t>(y) * canvas + x;
        const bool active =
          in_rectangle(scene.route_a, x, y) ||
          in_rectangle(scene.route_b, x, y);
        serial_analysis[plane_offset(
          sbs_scene_controller::analysis_grid_channel_e::viewport_valid
        ) + cell] = 1.0f;
        serial_analysis[plane_offset(
          sbs_scene_controller::analysis_grid_channel_e::viewport_x
        ) + cell] =
          (static_cast<float>(x) + 0.5f) /
          static_cast<float>(canvas);
        serial_analysis[plane_offset(
          sbs_scene_controller::analysis_grid_channel_e::viewport_y
        ) + cell] =
          (static_cast<float>(y) + 0.5f) /
          static_cast<float>(canvas);
        serial_analysis[plane_offset(
          sbs_scene_controller::analysis_grid_channel_e::
            temporal_activity_occupancy
        ) + cell] = active ? 1.0f : 0.0f;
        serial_dense[plane_offset(
          sbs_scene_controller::dense_out_channel_e::unknown_background
        ) + cell] = 1.0f;
        serial_dense[plane_offset(
          sbs_scene_controller::dense_out_channel_e::
            primary_playing_video
        ) + cell] = active ? 3.0f / 8.0f : 0.0f;
        serial_depth[plane_offset(
          sbs_scene_controller::depth_history_channel_e::
            valid_depth_confidence
        ) + cell] = 1.0f;
      }
    }
    const auto serial_analysis_buffer = make_structured_buffer(
      device.Get(),
      serial_analysis.data(),
      static_cast<UINT>(serial_analysis.size() * sizeof(float)),
      sizeof(float)
    );
    const auto serial_dense_buffer = make_structured_buffer(
      device.Get(),
      serial_dense.data(),
      static_cast<UINT>(serial_dense.size() * sizeof(float)),
      sizeof(float)
    );
    const auto serial_layout_buffer = make_structured_buffer(
      device.Get(),
      serial_layout.data(),
      static_cast<UINT>(serial_layout.size() * sizeof(float)),
      sizeof(float)
    );
    const auto serial_depth_buffer = make_structured_buffer(
      device.Get(),
      serial_depth.data(),
      static_cast<UINT>(serial_depth.size() * sizeof(float)),
      sizeof(float)
    );
    const auto summary_buffer = make_rw_structured_buffer(
      device.Get(),
      summary.data(),
      static_cast<UINT>(summary.size() * sizeof(float)),
      sizeof(float)
    );
    const std::array<float, sbs_scene_controller::global_out_word_count>
      zero_evidence {};
    const auto evidence_buffer = make_rw_structured_buffer(
      device.Get(),
      zero_evidence.data(),
      sizeof(zero_evidence),
      sizeof(float)
    );
    const std::vector<float> zero_summary(summary_float_count, 0.0f);
    const auto serial_summary_buffer = make_rw_structured_buffer(
      device.Get(),
      zero_summary.data(),
      static_cast<UINT>(zero_summary.size() * sizeof(float)),
      sizeof(float)
    );
    const auto serial_evidence_buffer = make_rw_structured_buffer(
      device.Get(),
      zero_evidence.data(),
      sizeof(zero_evidence),
      sizeof(float)
    );
    ASSERT_TRUE(serial_analysis_buffer.srv);
    ASSERT_TRUE(serial_dense_buffer.srv);
    ASSERT_TRUE(serial_layout_buffer.srv);
    ASSERT_TRUE(serial_depth_buffer.srv);
    ASSERT_TRUE(summary_buffer.uav);
    ASSERT_TRUE(evidence_buffer.uav);
    ASSERT_TRUE(serial_summary_buffer.uav);
    ASSERT_TRUE(serial_evidence_buffer.uav);

    const std::array<ID3D11ShaderResourceView *, 3> inputs {
      adaptive_buffer.srv.Get(),
      meta_buffer.srv.Get(),
      previous_state_buffer.srv.Get(),
    };
    context->CSSetShader(reduce_shader.Get(), nullptr, 0);
    context->CSSetShaderResources(
      4u,
      static_cast<UINT>(inputs.size()),
      inputs.data()
    );
    context->CSSetConstantBuffers(
      0u,
      1u,
      constant_buffer.GetAddressOf()
    );
    ID3D11UnorderedAccessView *outputs[] {
      summary_buffer.uav.Get(),
      evidence_buffer.uav.Get(),
    };
    context->CSSetUnorderedAccessViews(0u, 2u, outputs, nullptr);
    context->Dispatch(1u, 1u, 1u);
    std::array<ID3D11UnorderedAccessView *, 2> null_outputs {};
    context->CSSetUnorderedAccessViews(
      0u,
      static_cast<UINT>(null_outputs.size()),
      null_outputs.data(),
      nullptr
    );
    std::array<ID3D11ShaderResourceView *, 3> null_inputs {};
    context->CSSetShaderResources(
      4u,
      static_cast<UINT>(null_inputs.size()),
      null_inputs.data()
    );

    const std::array<ID3D11ShaderResourceView *, 7> serial_inputs {
      serial_analysis_buffer.srv.Get(),
      serial_dense_buffer.srv.Get(),
      serial_layout_buffer.srv.Get(),
      serial_depth_buffer.srv.Get(),
      adaptive_buffer.srv.Get(),
      meta_buffer.srv.Get(),
      previous_state_buffer.srv.Get(),
    };
    context->CSSetShader(serial_shader.Get(), nullptr, 0);
    context->CSSetShaderResources(
      0u,
      static_cast<UINT>(serial_inputs.size()),
      serial_inputs.data()
    );
    ID3D11UnorderedAccessView *serial_outputs[] {
      serial_evidence_buffer.uav.Get(),
      serial_summary_buffer.uav.Get(),
    };
    context->CSSetUnorderedAccessViews(
      0u,
      static_cast<UINT>(std::size(serial_outputs)),
      serial_outputs,
      nullptr
    );
    context->Dispatch(1u, 1u, 1u);
    std::array<ID3D11UnorderedAccessView *, 2> null_serial_outputs {};
    context->CSSetUnorderedAccessViews(
      0u,
      static_cast<UINT>(null_serial_outputs.size()),
      null_serial_outputs.data(),
      nullptr
    );
    std::array<ID3D11ShaderResourceView *, 7> null_serial_inputs {};
    context->CSSetShaderResources(
      0u,
      static_cast<UINT>(null_serial_inputs.size()),
      null_serial_inputs.data()
    );
    ID3D11Buffer *null_constant = nullptr;
    context->CSSetConstantBuffers(0u, 1u, &null_constant);
    context->CSSetShader(nullptr, nullptr, 0);

    const auto evidence = read_buffer<float>(
      device.Get(),
      context.Get(),
      evidence_buffer.srv.Get(),
      sbs_scene_controller::global_out_word_count
    );
    const auto serial_evidence = read_buffer<float>(
      device.Get(),
      context.Get(),
      serial_evidence_buffer.srv.Get(),
      sbs_scene_controller::global_out_word_count
    );
    ASSERT_EQ(
      evidence[static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::backend_output_valid
      )],
      1.0f
    );
    ASSERT_EQ(
      serial_evidence[static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::backend_output_valid
      )],
      1.0f
    );
    const auto layout_winner = static_cast<std::size_t>(
      std::distance(
        evidence.begin(),
        std::max_element(
          evidence.begin(),
          evidence.begin() + 6
        )
      )
    );
    const auto serial_layout_winner = static_cast<std::size_t>(
      std::distance(
        serial_evidence.begin(),
        std::max_element(
          serial_evidence.begin(),
          serial_evidence.begin() + 6
        )
      )
    );
    EXPECT_EQ(layout_winner, layout_ambiguous);
    EXPECT_EQ(serial_layout_winner, layout_ambiguous);
    EXPECT_NE(layout_winner, layout_primary_video);
    for (std::size_t word = 0u; word < 6u; ++word) {
      EXPECT_NEAR(evidence[word], serial_evidence[word], 3e-4f)
        << sbs_scene_controller::global_out_names[word];
    }
  }
}

TEST_F(
  SbsSceneControllerGpu,
  LockedRoiEventReducerUsesLocalCutAndGlobalExposureEvidenceAndMatchesSerialReference
) {
  constexpr std::size_t canvas =
    sbs_scene_controller::analysis_canvas_size;
  constexpr std::size_t plane = canvas * canvas;
  // Keep the low-level fixture exact with SBS_RULE_SUMMARY_FLOAT_COUNT.
  constexpr std::size_t summary_float_count =
    SBS_RULE_SUMMARY_FLOAT_COUNT;
  constexpr std::size_t plan_depth_coverage =
    SBS_RULE_SUMMARY_PLAN_BASE +
    SBS_RULE_SUMMARY_PLAN_DEPTH_COVERAGE;
  constexpr std::size_t plan_depth_confidence =
    SBS_RULE_SUMMARY_PLAN_BASE +
    SBS_RULE_SUMMARY_PLAN_DEPTH_CONFIDENCE;
  constexpr std::size_t plan_roi_structural_weight =
    SBS_RULE_SUMMARY_PLAN_BASE +
    SBS_RULE_SUMMARY_PLAN_ROI_STRUCTURAL_WEIGHT;
  constexpr std::size_t plan_roi_exposure_weight =
    SBS_RULE_SUMMARY_PLAN_BASE +
    SBS_RULE_SUMMARY_PLAN_ROI_EXPOSURE_WEIGHT;
  constexpr std::uint32_t roi_first = 32u;
  constexpr std::uint32_t roi_end = 96u;
  constexpr std::uint32_t inside_first = 48u;
  constexpr std::uint32_t inside_end = 80u;
  constexpr std::uint32_t outside_first = 0u;
  constexpr std::uint32_t outside_end = 32u;
  constexpr float expected_inside_support = 0.25f;

  struct alignas(16) scene_constants_t {
    std::uint32_t source_width = canvas;
    std::uint32_t source_height = canvas;
    std::uint32_t depth_width = canvas;
    std::uint32_t depth_height = canvas;
    std::uint32_t color_mode = 0u;
    std::uint32_t backend_generation = 1u;
    std::uint32_t history_valid = 1u;
    std::uint32_t reset_flags = 0u;
    float elapsed_seconds = 1.0f / 60.0f;
    float pop_floor = 1.2f;
    float pop_ceiling = 2.0f;
    float zero_plane_mode = 2.0f;
    float acquire_seconds = 0.12f;
    float challenger_seconds = 0.30f;
    float release_seconds = 60.0f;
    float scroll_enter_seconds = 0.05f;
    std::uint32_t source_frame_id_low = 0u;
    std::uint32_t source_frame_id_high = 0u;
    std::uint32_t identity_reserved_0 = 0u;
    std::uint32_t identity_reserved_1 = 0u;
    std::array<std::uint32_t, 8> ordered_abi_hash_words =
      sbs_scene_controller::ordered_abi_hash_words;
  };
  static_assert(sizeof(scene_constants_t) == 112u);

  enum class evidence_location_e {
    outside_roi,
    inside_roi,
  };
  struct event_case_t {
    std::string_view label;
    bool hard_cut_pulse;
    std::uint32_t analysis_flags;
    std::uint32_t reset_flags;
    sbs_scene_controller::dense_out_channel_e support_channel;
    evidence_location_e evidence_location;
    sbs_scene_controller::global_out_word_e expected_event;
    float expected_structural_support;
    float expected_exposure_support;
    float support_value = 1.0f;
    sbs_scene_controller::dense_out_channel_e secondary_support_channel =
      sbs_scene_controller::dense_out_channel_e::unknown_background;
    float secondary_support_value = 0.0f;
  };
  constexpr std::array cases {
    event_case_t {
      "hard_cut_evidence_outside_roi",
      true,
      0u,
      0u,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::outside_roi,
      sbs_scene_controller::global_out_word_e::event_same_shot,
      0.0f,
      0.0f,
    },
    event_case_t {
      "hard_cut_evidence_inside_roi",
      true,
      0u,
      0u,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::inside_roi,
      sbs_scene_controller::global_out_word_e::event_hard_cut,
      expected_inside_support,
      0.0f,
    },
    event_case_t {
      "exposure_evidence_outside_roi",
      false,
      sbs_adaptive_state::analysis_flag_exposure_like,
      0u,
      sbs_scene_controller::dense_out_channel_e::
        exposure_only_change_support,
      evidence_location_e::outside_roi,
      sbs_scene_controller::global_out_word_e::
        event_flash_or_exposure,
      0.0f,
      0.0f,
    },
    event_case_t {
      "exposure_evidence_inside_roi",
      false,
      sbs_adaptive_state::analysis_flag_exposure_like,
      0u,
      sbs_scene_controller::dense_out_channel_e::
        exposure_only_change_support,
      evidence_location_e::inside_roi,
      sbs_scene_controller::global_out_word_e::
        event_flash_or_exposure,
      0.0f,
      expected_inside_support,
    },
    event_case_t {
      "layout_reset_bypasses_stale_roi_gate",
      true,
      0u,
      sbs_scene_controller::reset_flags_layout,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::outside_roi,
      sbs_scene_controller::global_out_word_e::event_hard_cut,
      0.0f,
      0.0f,
    },
    event_case_t {
      "backend_reset_supersedes_detector_cut",
      true,
      0u,
      sbs_scene_controller::reset_flags_backend,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::outside_roi,
      sbs_scene_controller::global_out_word_e::event_same_shot,
      0.0f,
      0.0f,
    },
    event_case_t {
      "display_reset_supersedes_detector_cut",
      true,
      0u,
      sbs_scene_controller::reset_flags_display_or_hdr,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::outside_roi,
      sbs_scene_controller::global_out_word_e::event_same_shot,
      0.0f,
      0.0f,
    },
    event_case_t {
      "geometry_reset_supersedes_stale_roi_gate",
      true,
      0u,
      sbs_scene_controller::reset_flags_geometry,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::outside_roi,
      sbs_scene_controller::global_out_word_e::event_geometry_reset,
      0.0f,
      0.0f,
    },
    event_case_t {
      "depth_shot_reset_keeps_locked_roi_gate",
      true,
      0u,
      sbs_scene_controller::reset_flags_depth_shot,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::outside_roi,
      sbs_scene_controller::global_out_word_e::event_same_shot,
      0.0f,
      0.0f,
    },
    event_case_t {
      "global_exposure_does_not_require_locked_roi_participation",
      false,
      sbs_adaptive_state::analysis_flag_exposure_like,
      0u,
      sbs_scene_controller::dense_out_channel_e::
        exposure_only_change_support,
      evidence_location_e::outside_roi,
      sbs_scene_controller::global_out_word_e::
        event_flash_or_exposure,
      0.0f,
      0.0f,
      0.0625f,
    },
    event_case_t {
      "depth_structural_cut_is_not_vetoed_by_larger_exposure_lane",
      true,
      0u,
      0u,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::inside_roi,
      sbs_scene_controller::global_out_word_e::event_hard_cut,
      0.03125f,
      0.25f,
      0.125f,
      sbs_scene_controller::dense_out_channel_e::
        exposure_only_change_support,
      1.0f,
    },
    event_case_t {
      "malformed_temporal_lane_fails_closed",
      false,
      0u,
      0u,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::inside_roi,
      sbs_scene_controller::global_out_word_e::event_same_shot,
      expected_inside_support,
      0.0f,
    },
    event_case_t {
      "malformed_output_sentinel_fails_closed",
      false,
      0u,
      0u,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::inside_roi,
      sbs_scene_controller::global_out_word_e::event_same_shot,
      expected_inside_support,
      0.0f,
    },
    event_case_t {
      "malformed_viewport_coordinate_and_sentinel_fail_closed",
      false,
      0u,
      0u,
      sbs_scene_controller::dense_out_channel_e::
        structural_content_cut_support,
      evidence_location_e::inside_roi,
      sbs_scene_controller::global_out_word_e::event_same_shot,
      expected_inside_support,
      0.0f,
    },
  };

  const auto columns_shader = compile_compute_shader(
    device.Get(),
    "sbs_scene_rules_columns_cs.hlsl"
  );
  const auto reduce_shader = compile_compute_shader(
    device.Get(),
    "sbs_scene_rules_reduce_cs.hlsl"
  );
  const auto serial_shader = compile_compute_shader(
    device.Get(),
    "sbs_scene_rules_reduce_serial_reference_cs.hlsl"
  );
  ASSERT_TRUE(columns_shader);
  ASSERT_TRUE(reduce_shader);
  ASSERT_TRUE(serial_shader);

  const auto channel_offset =
    [](const auto channel) {
      return static_cast<std::size_t>(channel) * plane;
    };
  const auto event_argmax =
    [](const std::vector<float> &evidence) {
      constexpr std::size_t event_first =
        static_cast<std::size_t>(
          sbs_scene_controller::global_out_word_e::event_same_shot
        );
      constexpr std::size_t event_end =
        static_cast<std::size_t>(
          sbs_scene_controller::global_out_word_e::event_geometry_reset
        ) +
        1u;
      return static_cast<std::size_t>(std::distance(
        evidence.begin(),
        std::max_element(
          evidence.begin() +
            static_cast<std::ptrdiff_t>(event_first),
          evidence.begin() +
            static_cast<std::ptrdiff_t>(event_end)
        )
      ));
    };

  for (const auto &scene : cases) {
    SCOPED_TRACE(scene.label);
    std::vector<float> analysis(
      sbs_scene_controller::analysis_grid_channel_count * plane,
      0.0f
    );
    std::vector<float> dense(
      sbs_scene_controller::dense_out_channel_count * plane,
      0.0f
    );
    std::vector<float> layout(
      sbs_scene_controller::layout_history_channel_count * plane,
      0.0f
    );
    std::vector<float> depth_history(
      sbs_scene_controller::depth_history_channel_count * plane,
      0.0f
    );
    const std::uint32_t evidence_first =
      scene.evidence_location == evidence_location_e::inside_roi ?
        inside_first :
        outside_first;
    const std::uint32_t evidence_end =
      scene.evidence_location == evidence_location_e::inside_roi ?
        inside_end :
        outside_end;
    for (std::uint32_t y = 0u; y < canvas; ++y) {
      for (std::uint32_t x = 0u; x < canvas; ++x) {
        const std::size_t cell =
          static_cast<std::size_t>(y) * canvas + x;
        const bool in_roi =
          x >= roi_first && x < roi_end &&
          y >= roi_first && y < roi_end;
        analysis[channel_offset(
          sbs_scene_controller::analysis_grid_channel_e::viewport_valid
        ) + cell] = 1.0f;
        analysis[channel_offset(
          sbs_scene_controller::analysis_grid_channel_e::
            current_roi_coverage
        ) + cell] = in_roi ? 1.0f : 0.0f;
        analysis[channel_offset(
          sbs_scene_controller::analysis_grid_channel_e::viewport_x
        ) + cell] =
          (static_cast<float>(x) + 0.5f) /
          static_cast<float>(canvas);
        analysis[channel_offset(
          sbs_scene_controller::analysis_grid_channel_e::viewport_y
        ) + cell] =
          (static_cast<float>(y) + 0.5f) /
          static_cast<float>(canvas);
        dense[channel_offset(
          sbs_scene_controller::dense_out_channel_e::unknown_background
        ) + cell] = 1.0f;
        // Lane 2 is the authoritative static-media weight produced by the
        // evidence pass. Give the projection a nonuniform, already-classified
        // mass so this fixture catches any attempt to recompute it.
        dense[channel_offset(
          sbs_scene_controller::dense_out_channel_e::
            photographic_content_collage
        ) + cell] =
          static_cast<float>((3u * x + 5u * y) % 17u) / 32.0f;
        dense[channel_offset(
          sbs_scene_controller::dense_out_channel_e::stable_gutter
        ) + cell] =
          static_cast<float>((7u * x + 11u * y) % 19u) / 18.0f;
        depth_history[channel_offset(
          sbs_scene_controller::depth_history_channel_e::
            valid_depth_confidence
        ) + cell] = 1.0f;
        if (
          x >= evidence_first && x < evidence_end &&
          y >= evidence_first && y < evidence_end
        ) {
          dense[channel_offset(scene.support_channel) + cell] =
            scene.support_value;
          if (scene.secondary_support_value > 0.0f) {
            dense[
              channel_offset(scene.secondary_support_channel) + cell
            ] = scene.secondary_support_value;
          }
        }
      }
    }
    const bool malformed_temporal_lane =
      scene.label == "malformed_temporal_lane_fails_closed";
    const bool malformed_output_sentinel =
      scene.label == "malformed_output_sentinel_fails_closed";
    const bool malformed_viewport_coordinate =
      scene.label ==
        "malformed_viewport_coordinate_and_sentinel_fail_closed";
    const bool malformed_input =
      malformed_temporal_lane ||
      malformed_output_sentinel ||
      malformed_viewport_coordinate;
    if (malformed_temporal_lane) {
      dense[channel_offset(
        sbs_scene_controller::dense_out_channel_e::
          primary_playing_video
      ) + plane / 2u] =
        std::numeric_limits<float>::quiet_NaN();
    }
    if (malformed_output_sentinel || malformed_viewport_coordinate) {
      // Evidence owns this sentinel. The viewport-coordinate case mirrors the
      // exact pair emitted when evidence detects a malformed coordinate;
      // reducers must consume it identically without duplicating the scan.
      dense[channel_offset(
        sbs_scene_controller::dense_out_channel_e::unknown_background
      ) + plane / 2u] = -1.0f;
    }
    if (malformed_viewport_coordinate) {
      analysis[channel_offset(
        sbs_scene_controller::analysis_grid_channel_e::viewport_x
      ) + plane / 2u] =
        std::numeric_limits<float>::quiet_NaN();
    }

    sbs_adaptive_state::words_t adaptive_words {};
    for (std::size_t word = 0u;
         word < adaptive_words.size();
         ++word) {
      adaptive_words[word] = std::bit_cast<std::uint32_t>(
        sbs_adaptive_state::initial_values[word]
      );
    }
    adaptive_words[sbs_adaptive_state::index(
      sbs_adaptive_state::word_e::valid_depth_fraction
    )] = std::bit_cast<std::uint32_t>(1.0f);
    adaptive_words[sbs_adaptive_state::index(
      sbs_adaptive_state::word_e::hard_cut_pulse
    )] = std::bit_cast<std::uint32_t>(
      scene.hard_cut_pulse ? 1.0f : 0.0f
    );
    adaptive_words[sbs_adaptive_state::index(
      sbs_adaptive_state::word_e::hard_cut_count
    )] = scene.hard_cut_pulse ? 1u : 0u;
    adaptive_words[sbs_adaptive_state::index(
      sbs_adaptive_state::word_e::analysis_flags
    )] = std::bit_cast<std::uint32_t>(
      static_cast<float>(scene.analysis_flags)
    );

    std::array<float, sbs_scene_controller::meta_word_count> meta {};
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::elapsed_seconds
    )] = 1.0f / 60.0f;
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::source_aspect_ratio
    )] = 1.0f;
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::roi_x0
    )] = 0.25f;
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::roi_y0
    )] = 0.25f;
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::roi_x1
    )] = 0.75f;
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::roi_y1
    )] = 0.75f;
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::state_content
    )] = 1.0f;
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::depth_input_valid
    )] = 1.0f;
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::pop_floor
    )] = 1.2f;
    meta[static_cast<std::size_t>(
      sbs_scene_controller::meta_word_e::pop_ceiling
    )] = 2.0f;

    auto previous_state = sbs_scene_controller::initial_word_bits;
    const auto set_state_float =
      [&](const sbs_scene_controller::rule_state_word_e word,
          const float value) {
        previous_state[sbs_scene_controller::index(word)] =
          std::bit_cast<std::uint32_t>(value);
      };
    set_state_float(
      sbs_scene_controller::rule_state_word_e::state_kind,
      static_cast<float>(sbs_scene_controller::state_kind_e::content)
    );
    set_state_float(
      sbs_scene_controller::rule_state_word_e::output_valid,
      1.0f
    );
    previous_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] =
      sbs_scene_controller::state_flags_initialized |
      sbs_scene_controller::state_flags_roi_locked;
    previous_state[sbs_scene_controller::index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )] = 1u;
    set_state_float(
      sbs_scene_controller::rule_state_word_e::committed_roi_x0,
      0.25f
    );
    set_state_float(
      sbs_scene_controller::rule_state_word_e::committed_roi_y0,
      0.25f
    );
    set_state_float(
      sbs_scene_controller::rule_state_word_e::committed_roi_x1,
      0.75f
    );
    set_state_float(
      sbs_scene_controller::rule_state_word_e::committed_roi_y1,
      0.75f
    );
    set_state_float(
      sbs_scene_controller::rule_state_word_e::
        committed_roi_confidence,
      1.0f
    );

    const auto analysis_buffer = make_structured_buffer(
      device.Get(),
      analysis.data(),
      static_cast<UINT>(analysis.size() * sizeof(float)),
      sizeof(float)
    );
    const auto dense_buffer = make_structured_buffer(
      device.Get(),
      dense.data(),
      static_cast<UINT>(dense.size() * sizeof(float)),
      sizeof(float)
    );
    const auto layout_buffer = make_structured_buffer(
      device.Get(),
      layout.data(),
      static_cast<UINT>(layout.size() * sizeof(float)),
      sizeof(float)
    );
    const auto depth_buffer = make_structured_buffer(
      device.Get(),
      depth_history.data(),
      static_cast<UINT>(depth_history.size() * sizeof(float)),
      sizeof(float)
    );
    const auto adaptive_buffer = make_structured_buffer(
      device.Get(),
      adaptive_words.data(),
      sizeof(adaptive_words),
      sizeof(float) * 4u
    );
    const auto meta_buffer = make_structured_buffer(
      device.Get(),
      meta.data(),
      sizeof(meta),
      sizeof(float)
    );
    const auto previous_state_buffer = make_structured_buffer(
      device.Get(),
      previous_state.data(),
      sizeof(previous_state),
      sizeof(std::uint32_t) * 4u
    );
    std::vector<float> initial_summary(
      summary_float_count,
      0.0f
    );
    // The production evidence pass preaggregates ROI-local event support while
    // these values are in registers. This low-level reducer fixture bypasses
    // evidence, so reproduce its 16x16 groups explicitly rather than restoring a
    // test-only fallback scan to production.
    constexpr std::uint32_t event_group_width = 16u;
    static_assert(
      SBS_RULE_SUMMARY_EVENT_GROUP_COUNT ==
        (canvas / event_group_width) * (canvas / event_group_width)
    );
    for (std::uint32_t group = 0u;
         group < SBS_RULE_SUMMARY_EVENT_GROUP_COUNT;
         ++group) {
      const std::uint32_t group_x =
        (group % (canvas / event_group_width)) * event_group_width;
      const std::uint32_t group_y =
        (group / (canvas / event_group_width)) * event_group_width;
      std::array<float, SBS_RULE_SUMMARY_EVENT_GROUP_STRIDE> sums {};
      for (std::uint32_t y = group_y;
           y < group_y + event_group_width;
           ++y) {
        for (std::uint32_t x = group_x;
             x < group_x + event_group_width;
             ++x) {
          const std::size_t cell =
            static_cast<std::size_t>(y) * canvas + x;
          if (analysis[channel_offset(
                sbs_scene_controller::analysis_grid_channel_e::
                  current_roi_coverage
              ) + cell] <= 0.5f) {
            continue;
          }
          const float depth_confidence = std::clamp(
            depth_history[channel_offset(
              sbs_scene_controller::depth_history_channel_e::
                valid_depth_confidence
            ) + cell],
            0.0f,
            1.0f
          );
          const float structural_support = std::clamp(
            dense[channel_offset(
              sbs_scene_controller::dense_out_channel_e::
                structural_content_cut_support
            ) + cell],
            0.0f,
            1.0f
          );
          const float exposure_support = std::clamp(
            dense[channel_offset(
              sbs_scene_controller::dense_out_channel_e::
                exposure_only_change_support
            ) + cell],
            0.0f,
            1.0f
          );
          sums[0] += 1.0f;
          sums[1] += depth_confidence;
          sums[2] += structural_support * depth_confidence;
          sums[3] += exposure_support * depth_confidence;
        }
      }
      const std::size_t group_base =
        SBS_RULE_SUMMARY_EVENT_GROUP_BASE +
        static_cast<std::size_t>(group) *
          SBS_RULE_SUMMARY_EVENT_GROUP_STRIDE;
      std::copy(
        sums.begin(),
        sums.end(),
        initial_summary.begin() +
          static_cast<std::ptrdiff_t>(group_base)
      );
    }
    initial_summary[SBS_RULE_SUMMARY_EVENT_SOURCE_MARKER] = 1.0f;
    const auto parallel_summary = make_rw_structured_buffer(
      device.Get(),
      initial_summary.data(),
      static_cast<UINT>(initial_summary.size() * sizeof(float)),
      sizeof(float)
    );
    const auto serial_summary = make_rw_structured_buffer(
      device.Get(),
      initial_summary.data(),
      static_cast<UINT>(initial_summary.size() * sizeof(float)),
      sizeof(float)
    );
    const std::array<float, sbs_scene_controller::global_out_word_count>
      zero_evidence {};
    const auto parallel_evidence = make_rw_structured_buffer(
      device.Get(),
      zero_evidence.data(),
      sizeof(zero_evidence),
      sizeof(float)
    );
    const auto serial_evidence = make_rw_structured_buffer(
      device.Get(),
      zero_evidence.data(),
      sizeof(zero_evidence),
      sizeof(float)
    );
    scene_constants_t constants;
    constants.reset_flags = scene.reset_flags;
    const auto constant_buffer = make_constant_buffer(
      device.Get(),
      &constants,
      sizeof(constants)
    );
    const std::array<ID3D11ShaderResourceView *, 7> inputs {
      analysis_buffer.srv.Get(),
      dense_buffer.srv.Get(),
      layout_buffer.srv.Get(),
      depth_buffer.srv.Get(),
      adaptive_buffer.srv.Get(),
      meta_buffer.srv.Get(),
      previous_state_buffer.srv.Get(),
    };
    ASSERT_TRUE(std::all_of(
      inputs.begin(),
      inputs.end(),
      [](const auto *view) {
        return view != nullptr;
      }
    ));
    ASSERT_TRUE(parallel_summary.uav);
    ASSERT_TRUE(serial_summary.uav);
    ASSERT_TRUE(parallel_evidence.uav);
    ASSERT_TRUE(serial_evidence.uav);
    ASSERT_TRUE(constant_buffer);

    context->CSSetShaderResources(
      0,
      static_cast<UINT>(inputs.size()),
      inputs.data()
    );
    context->CSSetConstantBuffers(
      0,
      1,
      constant_buffer.GetAddressOf()
    );
    const auto dispatch =
      [&](ID3D11ComputeShader *shader,
          ID3D11UnorderedAccessView *first,
          ID3D11UnorderedAccessView *second = nullptr) {
        context->CSSetShader(shader, nullptr, 0);
        ID3D11UnorderedAccessView *outputs[2] {
          first,
          second,
        };
        const UINT output_count = second ? 2u : 1u;
        context->CSSetUnorderedAccessViews(
          0,
          output_count,
          outputs,
          nullptr
        );
        context->Dispatch(1u, 1u, 1u);
        ID3D11UnorderedAccessView *null_outputs[2] {
          nullptr,
          nullptr,
        };
        context->CSSetUnorderedAccessViews(
          0,
          output_count,
          null_outputs,
          nullptr
        );
    };
    dispatch(columns_shader.Get(), parallel_summary.uav.Get());
    dispatch(
      reduce_shader.Get(),
      parallel_summary.uav.Get(),
      parallel_evidence.uav.Get()
    );
    dispatch(
      serial_shader.Get(),
      serial_evidence.uav.Get(),
      serial_summary.uav.Get()
    );
    std::array<ID3D11ShaderResourceView *, 7> null_inputs {};
    context->CSSetShaderResources(
      0,
      static_cast<UINT>(null_inputs.size()),
      null_inputs.data()
    );
    ID3D11Buffer *null_constant = nullptr;
    context->CSSetConstantBuffers(0, 1, &null_constant);
    context->CSSetShader(nullptr, nullptr, 0);

    const auto parallel_result = read_buffer<float>(
      device.Get(),
      context.Get(),
      parallel_evidence.srv.Get(),
      sbs_scene_controller::global_out_word_count
    );
    const auto serial_result = read_buffer<float>(
      device.Get(),
      context.Get(),
      serial_evidence.srv.Get(),
      sbs_scene_controller::global_out_word_count
    );
    const auto parallel_summary_result = read_buffer<float>(
      device.Get(),
      context.Get(),
      parallel_summary.srv.Get(),
      summary_float_count
    );
    const auto serial_summary_result = read_buffer<float>(
      device.Get(),
      context.Get(),
      serial_summary.srv.Get(),
      summary_float_count
    );

    if (scene.label == cases.front().label) {
      constexpr std::size_t column_stride =
        SBS_RULE_SUMMARY_COLUMN_STRIDE;
      constexpr std::size_t column_media_mass = 0u;
      constexpr std::size_t column_media_invalid = 1u;
      constexpr std::size_t column_gutter_mass = 3u;
      const std::size_t media_offset = channel_offset(
        sbs_scene_controller::dense_out_channel_e::
          photographic_content_collage
      );
      const std::size_t gutter_offset = channel_offset(
        sbs_scene_controller::dense_out_channel_e::stable_gutter
      );
      for (std::size_t x = 0u; x < canvas; ++x) {
        float expected_mass = 0.0f;
        float expected_gutter_mass = 0.0f;
        for (std::size_t y = 0u; y < canvas; ++y) {
          expected_mass += dense[media_offset + y * canvas + x];
          expected_gutter_mass +=
            dense[gutter_offset + y * canvas + x];
        }
        EXPECT_FLOAT_EQ(
          parallel_summary_result[
            x * column_stride + column_media_mass
          ],
          expected_mass
        ) << "column " << x;
        EXPECT_FLOAT_EQ(
          parallel_summary_result[
            x * column_stride + column_media_invalid
          ],
          0.0f
        ) << "column " << x;
        EXPECT_FLOAT_EQ(
          parallel_summary_result[
            x * column_stride + column_gutter_mass
          ],
          expected_gutter_mass
        ) << "column " << x;
      }
      for (std::size_t y = 0u; y < canvas; ++y) {
        float expected_gutter_mass = 0.0f;
        for (std::size_t x = 0u; x < canvas; ++x) {
          expected_gutter_mass +=
            dense[gutter_offset + y * canvas + x];
        }
        EXPECT_FLOAT_EQ(
          parallel_summary_result[
            SBS_RULE_SUMMARY_TEMPORAL_ROW_BASE +
            y * SBS_RULE_SUMMARY_TEMPORAL_ROW_STRIDE +
            2u
          ],
          expected_gutter_mass
        ) << "row " << y;
      }
    }

    constexpr std::size_t event_first =
      static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::event_same_shot
      );
    constexpr std::size_t event_end =
      static_cast<std::size_t>(
        sbs_scene_controller::global_out_word_e::event_geometry_reset
      ) +
      1u;
    for (std::size_t word = event_first; word < event_end; ++word) {
      EXPECT_NEAR(
        parallel_result[word],
        serial_result[word],
        3e-4f
      ) << sbs_scene_controller::global_out_names[word];
    }
    const auto backend_output_valid = static_cast<std::size_t>(
      sbs_scene_controller::global_out_word_e::backend_output_valid
    );
    EXPECT_FLOAT_EQ(
      parallel_result[backend_output_valid],
      malformed_input ? 0.0f : 1.0f
    );
    EXPECT_FLOAT_EQ(
      serial_result[backend_output_valid],
      malformed_input ? 0.0f : 1.0f
    );
    EXPECT_EQ(
      event_argmax(parallel_result),
      static_cast<std::size_t>(scene.expected_event)
    );
    EXPECT_EQ(
      event_argmax(serial_result),
      static_cast<std::size_t>(scene.expected_event)
    );
    for (const auto word : {
           plan_depth_coverage,
           plan_depth_confidence,
           plan_roi_structural_weight,
           plan_roi_exposure_weight,
         }) {
      EXPECT_NEAR(
        parallel_summary_result[word],
        serial_summary_result[word],
        3e-4f
      ) << "summary[" << word << "]";
    }
    ASSERT_GT(parallel_summary_result[plan_depth_confidence], 0.0f);
    EXPECT_FLOAT_EQ(
      parallel_summary_result[plan_depth_coverage],
      4096.0f
    );
    EXPECT_FLOAT_EQ(
      parallel_summary_result[plan_depth_confidence],
      4096.0f
    );
    EXPECT_NEAR(
      parallel_summary_result[plan_roi_structural_weight] /
        parallel_summary_result[plan_depth_confidence],
      scene.expected_structural_support,
      1e-6f
    );
    EXPECT_NEAR(
      parallel_summary_result[plan_roi_exposure_weight] /
        parallel_summary_result[plan_depth_confidence],
      scene.expected_exposure_support,
      1e-6f
    );
  }
}


TEST(
  SbsSceneControllerGpuHardwarePerf,
  DISABLED_ReportsIsolatedFourKRuleTimings
) {
  struct setting_restore_t {
    bool diagnostics_enabled;
    bool perf_enabled;

    ~setting_restore_t() {
      sbs_perf::reset();
      config::sunshine.diagnostics_enabled = diagnostics_enabled;
      sbs_perf::set_enabled(perf_enabled);
    }
  } restore {
    config::sunshine.diagnostics_enabled,
    sbs_perf::enabled(),
  };
  config::sunshine.diagnostics_enabled = true;
  sbs_perf::set_enabled(true);
  sbs_perf::reset();

  constexpr D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_0};
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  D3D_FEATURE_LEVEL feature_level {};
  const auto create_status = D3D11CreateDevice(
    nullptr,
    D3D_DRIVER_TYPE_HARDWARE,
    nullptr,
    D3D11_CREATE_DEVICE_SINGLETHREADED,
    levels,
    static_cast<UINT>(std::size(levels)),
    D3D11_SDK_VERSION,
    &device,
    &feature_level,
    &context
  );
  if (FAILED(create_status)) {
    GTEST_SKIP() << "D3D11 hardware device is unavailable";
  }

  constexpr UINT depth_width = 770;
  constexpr UINT depth_height = 434;
  const auto checker_0 =
    make_source_view(device.Get(), 3840, 2160, 0);
  const auto checker_1 =
    make_source_view(device.Get(), 3840, 2160, 1);
  const auto content =
    make_content_source_view(device.Get(), 3840, 2160);
  const auto scroll_0 =
    make_scroll_source_view(device.Get(), 3840, 2160, 0);
  const auto scroll_24 =
    make_scroll_source_view(device.Get(), 3840, 2160, 24);
  const auto split = make_split_threshold_source_view(
    device.Get(),
    3840,
    2160,
    96,
    48,
    8
  );
  const auto depth = make_depth_view(
    device.Get(),
    depth_width,
    depth_height
  );
  ASSERT_TRUE(checker_0);
  ASSERT_TRUE(checker_1);
  ASSERT_TRUE(content);
  ASSERT_TRUE(scroll_0);
  ASSERT_TRUE(scroll_24);
  ASSERT_TRUE(split);
  ASSERT_TRUE(depth);
  const std::array<ID3D11ShaderResourceView *, 16> workload {
    checker_0.Get(),
    checker_0.Get(),
    checker_1.Get(),
    content.Get(),
    content.Get(),
    scroll_0.Get(),
    scroll_24.Get(),
    scroll_24.Get(),
    split.Get(),
    split.Get(),
    checker_1.Get(),
    content.Get(),
    scroll_0.Get(),
    scroll_24.Get(),
    split.Get(),
    checker_0.Get(),
  };

  std::vector<float> raw_depth(
    static_cast<std::size_t>(depth_width) * depth_height,
    0.5f
  );
  const auto raw_depth_buffer = make_structured_buffer(
    device.Get(),
    raw_depth.data(),
    static_cast<UINT>(raw_depth.size() * sizeof(float)),
    sizeof(float)
  );
  const std::array<float, 4> frame_state {
    0.0f,
    1.0f,
    1.0f,
    1.0f,
  };
  const auto frame_state_buffer = make_structured_buffer(
    device.Get(),
    frame_state.data(),
    sizeof(frame_state),
    sizeof(frame_state)
  );
  const auto adaptive_state = make_adaptive_state_buffer(
    device.Get(),
    0u,
    1.0f
  );
  const std::array<
    std::array<std::uint32_t, 4>,
    models::frame_roi_transform_vector_count
  > zero_transform {};
  const auto frame_roi_transform_buffer = make_structured_buffer(
    device.Get(),
    zero_transform.data(),
    sizeof(zero_transform),
    sizeof(zero_transform[0])
  );
  ASSERT_TRUE(raw_depth_buffer.srv);
  ASSERT_TRUE(frame_state_buffer.srv);
  ASSERT_TRUE(adaptive_state.srv);
  ASSERT_TRUE(frame_roi_transform_buffer.srv);

  config::video_t::sbs_t sbs;
  sbs.scene_controller =
    config::sbs_scene_controller_e::shadow_rules;
  const std::filesystem::path assets_dir =
    std::filesystem::path(SUNSHINE_SHADERS_DIR)
      .parent_path()
      .parent_path();
  models::sbs_scene_controller_gpu controller(
    device,
    context,
    assets_dir,
    sbs.scene_controller,
    sbs,
    false,
    true
  );
  ASSERT_TRUE(controller.valid());

  constexpr std::uint64_t serialization_batch = 8;
  const auto run_updates =
    [&](const std::uint64_t first_frame_id,
        const std::uint64_t frame_count,
        const bool record_ticks) {
      for (
        std::uint64_t offset = 0;
        offset < frame_count;
        ++offset
      ) {
        const auto frame_id = first_frame_id + offset;
        auto *source = workload[
          static_cast<std::size_t>(frame_id - 1u) % workload.size()
        ];
        ASSERT_TRUE(controller.prepare_scene(
          source,
          models::input_color_space::srgb,
          frame_id
        ));
        controller.mark_enqueued(frame_id);
        ASSERT_TRUE(controller.resolve_completed(
          frame_id,
          raw_depth_buffer.srv.Get(),
          depth.Get(),
          frame_state_buffer.srv.Get(),
          adaptive_state.srv.Get(),
          depth_width,
          depth_height,
          frame_roi_transform_buffer.srv.Get()
        ));

        // Drain at the timing-ring bound instead of after every tiny update. A per-frame
        // blocking Map lets the GPU repeatedly enter its low-power state and makes this a
        // clock-wakeup benchmark, while an unbounded queue would not represent the live frame
        // pipeline either. Eight updates fill (but cannot overrun) both timestamp rings and keep
        // the queue below two milliseconds on the target hardware. The readback then establishes
        // a hard batch boundary.
        if (
          (offset + 1u) % serialization_batch == 0u ||
          offset + 1u == frame_count
        ) {
          const auto snapshot = controller.snapshot();
          ASSERT_TRUE(snapshot.snapshot_available);
          const auto output = read_buffer<float>(
            device.Get(),
            context.Get(),
            snapshot.global_output.Get(),
            sbs_scene_controller::global_out_word_count
          );
          ASSERT_EQ(
            output.size(),
            sbs_scene_controller::global_out_word_count
          );
        }
        if (record_ticks) {
          sbs_perf::tick();
        }
      }
    };

  // A cold WDDM device can remain at its idle clock for this intentionally tiny workload, then
  // boost only after the first bounded run completes. Warm it with the exact workload before
  // resetting the performance generation so the gate measures steady-state streaming cost.
  constexpr std::uint64_t warmup_frames = 640;
  constexpr std::uint64_t measured_frames = 640;
  run_updates(1u, warmup_frames, false);
  ASSERT_FALSE(HasFatalFailure());
  sbs_perf::reset();
  run_updates(warmup_frames + 1u, measured_frames, true);
  ASSERT_FALSE(HasFatalFailure());

  const auto report_path =
    std::filesystem::temp_directory_path() /
    "apollo-scene-controller-hardware-perf.json";
  ASSERT_TRUE(sbs_perf::dump_json(report_path.string()));
  std::ifstream report(report_path);
  ASSERT_TRUE(report.good());
  const auto report_json = nlohmann::json::parse(report);
  ASSERT_TRUE(report_json.contains("stages"));
  const auto &stages = report_json.at("stages");
  constexpr std::array expected_stages {
    "scene_prepare_gpu",
    "scene_rules_evidence_gpu",
    "scene_rules_reduce_gpu",
    "scene_rules_resolve_history_gpu",
    "scene_rules_gpu",
  };
  for (const auto *stage : expected_stages) {
    ASSERT_TRUE(stages.contains(stage)) << stage;
    EXPECT_GE(
      stages.at(stage).at("total").get<std::uint64_t>(),
      measured_frames - 16u
    ) << stage;
    EXPECT_GT(stages.at(stage).at("n").get<std::uint64_t>(), 0u)
      << stage;
  }
  EXPECT_LT(
    stages.at("scene_rules_reduce_gpu").at("p95_ms").get<double>(),
    0.20
  );
  EXPECT_LE(
    stages.at("scene_rules_gpu").at("p95_ms").get<double>(),
    0.25
  );
  const auto stage_value =
    [&](const char *stage, const char *field) {
      return stages.at(stage).at(field).get<double>();
    };
  const double rules_p95 = stage_value("scene_rules_gpu", "p95_ms");
  EXPECT_GE(
    rules_p95,
    stage_value("scene_rules_evidence_gpu", "p95_ms")
  );
  EXPECT_GE(
    rules_p95,
    stage_value("scene_rules_reduce_gpu", "p95_ms")
  );
  EXPECT_GE(
    rules_p95,
    stage_value("scene_rules_resolve_history_gpu", "p95_ms")
  );
  EXPECT_NEAR(
    stage_value("scene_rules_gpu", "mean_ms"),
    stage_value("scene_rules_evidence_gpu", "mean_ms") +
      stage_value("scene_rules_reduce_gpu", "mean_ms") +
      stage_value("scene_rules_resolve_history_gpu", "mean_ms"),
    0.05
  );
  report.clear();
  report.seekg(0);
  std::cout << report.rdbuf() << std::endl;
  std::error_code remove_error;
  std::filesystem::remove(report_path, remove_error);
}

#endif  // _WIN32
