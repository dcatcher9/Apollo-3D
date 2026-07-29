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
  #include <limits>
  #include <memory>
  #include <nlohmann/json.hpp>
  #include <src/generated/sbs_adaptive_state_contract.h>
  #include <src/generated/sbs_scene_controller_contract.h>
  #include <src/sbs_frame_roi_transform.h>
  #include <src/sbs_perf.h>
  #include <src/sbs_scene_controller_gpu.h>
  #include <string_view>
  #include <thread>
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
    const float valid_depth_fraction = 0.0f
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
    const UINT vertical_shift
  ) {
    std::vector<std::uint32_t> source(width * height);
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        const UINT shifted_y = (y + vertical_shift) % height;
        std::uint32_t hash =
          (x + 1u) * 0x9e3779b9u ^
          (shifted_y + 1u) * 0x85ebca6bu;
        hash ^= hash >> 16u;
        hash *= 0x7feb352du;
        hash ^= hash >> 15u;
        const std::uint8_t value = static_cast<std::uint8_t>(
          126u + hash % 5u
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

  adaptive_buffer = make_adaptive_state_buffer(
    device.Get(),
    baseline_counter,
    1.0f
  );
  ASSERT_TRUE(adaptive_buffer.srv);
  ASSERT_TRUE(run_frame(320, 180, 170));
  EXPECT_EQ(
    read_rule_state(controller->snapshot())[external_count_index],
    baseline_counter
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
  EXPECT_EQ(
    read_rule_state(invalid_snapshot)[external_count_index],
    baseline_counter
  );

  ASSERT_TRUE(run_frame(320, 180, 172));
  const auto recovered_snapshot = controller->snapshot();
  EXPECT_FLOAT_EQ(request_value(recovered_snapshot), 1.0f);
  EXPECT_EQ(
    read_rule_state(recovered_snapshot)[external_count_index],
    incremented_counter
  );

  ASSERT_TRUE(run_frame(320, 180, 173));
  const auto steady_snapshot = controller->snapshot();
  EXPECT_FLOAT_EQ(request_value(steady_snapshot), 0.0f);
  EXPECT_EQ(
    read_rule_state(steady_snapshot)[external_count_index],
    incremented_counter
  );
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

  std::this_thread::sleep_for(std::chrono::milliseconds(160));
  const auto second_source = make_content_source_view(
    device.Get(),
    320,
    180
  );
  ASSERT_TRUE(second_source);
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
  ASSERT_TRUE(run_source(source.Get(), 210));
  std::this_thread::sleep_for(std::chrono::milliseconds(160));
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
  ASSERT_TRUE(run_source(source.Get(), 220));
  std::this_thread::sleep_for(std::chrono::milliseconds(160));
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
  ASSERT_TRUE(run_source(baseline_source.Get(), 200));
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

  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  const auto scrolled_source = make_scroll_source_view(
    device.Get(),
    128,
    128,
    1
  );
  ASSERT_TRUE(scrolled_source);
  ASSERT_TRUE(run_source(scrolled_source.Get(), 201));
  const auto held_snapshot = controller->snapshot();
  const auto held_state = read_rule_state(held_snapshot);
  EXPECT_FLOAT_EQ(
    word_as_float(
      held_state,
      sbs_scene_controller::rule_state_word_e::state_kind
    ),
    static_cast<float>(
      sbs_scene_controller::state_kind_e::scroll_hold
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
      sbs_scene_controller::state_flags_roi_locked |
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
          return value != baseline_layout[index++];
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
  EXPECT_GT(
    layout_plane_diff_count(
      sbs_scene_controller::layout_history_channel_e::vertical_motion
    ),
    0
  );
  EXPECT_GT(
    layout_plane_diff_count(
      sbs_scene_controller::layout_history_channel_e::motion_confidence
    ),
    0
  );
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
    EXPECT_TRUE(std::equal(held_layout.begin() + offset, held_layout.begin() + offset + analysis_pixels, baseline_layout.begin() + offset));
  }
  EXPECT_EQ(held_depth, baseline_depth);

  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  const auto settled_source = make_scroll_source_view(
    device.Get(),
    128,
    128,
    1
  );
  ASSERT_TRUE(settled_source);
  ASSERT_TRUE(run_source(settled_source.Get(), 202));
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
  EXPECT_FLOAT_EQ(
    word_as_float(
      released_state,
      sbs_scene_controller::rule_state_word_e::scroll_hold_s
    ),
    0.0f
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
    ++frame_id;
  }

  const auto content = make_content_source_view(device.Get(), 320, 180);
  const auto scroll_0 =
    make_scroll_source_view(device.Get(), 320, 180, 0);
  const auto scroll_6 =
    make_scroll_source_view(device.Get(), 320, 180, 6);
  const auto invalid = make_nan_source_view(device.Get(), 320, 180);
  ASSERT_TRUE(content);
  ASSERT_TRUE(scroll_0);
  ASSERT_TRUE(scroll_6);
  ASSERT_TRUE(invalid);
  struct sequence_frame_t {
    std::string_view label;
    ComPtr<ID3D11ShaderResourceView> source;
    float elapsed_seconds;
  };
  const std::array sequence {
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
    ++frame_id;
  }
}

TEST_F(
  SbsSceneControllerGpu,
  DenseReducerFixtureCoversCandidateCapacityAndStaleClearing
) {
  constexpr std::size_t canvas =
    sbs_scene_controller::analysis_canvas_size;
  constexpr std::size_t plane = canvas * canvas;
  constexpr std::size_t summary_float_count = 6144u;
  constexpr std::size_t plan_base = 512u;
  constexpr std::size_t plan_split = plan_base + 11u;
  constexpr std::size_t plan_region_count = plan_base + 12u;
  constexpr std::size_t plan_video_ambiguous = plan_base + 13u;
  constexpr std::size_t region_base = plan_base + 32u;
  constexpr std::size_t region_stride = 32u;
  constexpr std::size_t region_active = 0u;
  constexpr std::size_t region_candidate_first = 30u;
  constexpr std::size_t region_candidate_count = 31u;
  constexpr std::size_t candidate_desc_base = 4320u;
  constexpr std::size_t candidate_desc_stride = 8u;
  constexpr std::size_t candidate_active = 0u;
  constexpr std::size_t candidate_first_x = 1u;
  constexpr std::size_t candidate_end_x = 2u;
  constexpr std::size_t candidate_first_y = 3u;
  constexpr std::size_t candidate_end_y = 4u;
  constexpr std::size_t candidate_result_base = 4576u;
  constexpr std::size_t candidate_result_stride = 8u;
  constexpr std::size_t candidate_coverage = 5u;

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

  struct axis_run_t {
    std::uint32_t first;
    std::uint32_t end;
  };
  struct dense_case_t {
    std::string_view label;
    std::vector<axis_run_t> x_runs;
    std::vector<axis_run_t> y_runs;
    axis_run_t gutter {0u, 0u};
    std::uint32_t expected_candidates;
    std::uint32_t expected_regions;
    bool expected_split;
    bool expected_ambiguous;
  };
  struct run_result_t {
    std::vector<float> summary;
    std::vector<float> parallel_evidence;
    std::vector<float> serial_evidence;
  };

  constexpr std::array parallel_shader_names {
    "sbs_scene_rules_columns_cs.hlsl",
    "sbs_scene_rules_plan_columns_cs.hlsl",
    "sbs_scene_rules_rows_cs.hlsl",
    "sbs_scene_rules_plan_rows_cs.hlsl",
    "sbs_scene_rules_candidates_cs.hlsl",
    "sbs_scene_rules_reduce_cs.hlsl",
  };
  std::array<ComPtr<ID3D11ComputeShader>, parallel_shader_names.size()>
    parallel_shaders;
  for (std::size_t index = 0; index < parallel_shaders.size(); ++index) {
    parallel_shaders[index] = compile_compute_shader(
      device.Get(),
      parallel_shader_names[index]
    );
    ASSERT_TRUE(parallel_shaders[index]);
  }
  const auto serial_shader = compile_compute_shader(
    device.Get(),
    "sbs_scene_rules_reduce_serial_reference_cs.hlsl"
  );
  ASSERT_TRUE(serial_shader);

  const std::vector<float> zero_summary(summary_float_count, 0.0f);
  const auto summary = make_rw_structured_buffer(
    device.Get(),
    zero_summary.data(),
    static_cast<UINT>(zero_summary.size() * sizeof(float)),
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
  ASSERT_TRUE(summary.srv);
  ASSERT_TRUE(summary.uav);
  ASSERT_TRUE(parallel_evidence.srv);
  ASSERT_TRUE(parallel_evidence.uav);
  ASSERT_TRUE(serial_evidence.srv);
  ASSERT_TRUE(serial_evidence.uav);

  const auto contains =
    [](const std::vector<axis_run_t> &runs, const std::uint32_t value) {
      return std::any_of(
        runs.begin(),
        runs.end(),
        [value](const axis_run_t run) {
          return value >= run.first && value < run.end;
        }
      );
    };
  const auto run_case =
    [&](const dense_case_t &scene, run_result_t &result) {
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
      const auto channel_offset =
        [](const auto channel) {
          return static_cast<std::size_t>(channel) * plane;
        };
      for (std::uint32_t y = 0u; y < canvas; ++y) {
        for (std::uint32_t x = 0u; x < canvas; ++x) {
          const std::size_t cell =
            static_cast<std::size_t>(y) * canvas + x;
          analysis[channel_offset(
            sbs_scene_controller::analysis_grid_channel_e::viewport_valid
          ) + cell] = 1.0f;
          analysis[channel_offset(
            sbs_scene_controller::analysis_grid_channel_e::
              current_roi_coverage
          ) + cell] = 1.0f;
          analysis[channel_offset(
            sbs_scene_controller::analysis_grid_channel_e::viewport_x
          ) + cell] = (static_cast<float>(x) + 0.5f) /
                      static_cast<float>(canvas);
          analysis[channel_offset(
            sbs_scene_controller::analysis_grid_channel_e::viewport_y
          ) + cell] = (static_cast<float>(y) + 0.5f) /
                      static_cast<float>(canvas);
          dense[channel_offset(
            sbs_scene_controller::dense_out_channel_e::unknown_background
          ) + cell] = 1.0f;
          depth_history[channel_offset(
            sbs_scene_controller::depth_history_channel_e::
              valid_depth_confidence
          ) + cell] = 1.0f;
          if (x >= scene.gutter.first && x < scene.gutter.end) {
            dense[channel_offset(
              sbs_scene_controller::dense_out_channel_e::stable_gutter
            ) + cell] = 1.0f;
          }
          if (
            contains(scene.x_runs, x) &&
            contains(scene.y_runs, y)
          ) {
            dense[channel_offset(
              sbs_scene_controller::dense_out_channel_e::
                primary_playing_video
            ) + cell] = 0.9f;
            layout[channel_offset(
              sbs_scene_controller::layout_history_channel_e::
                photographic_density_ema
            ) + cell] = 0.5f;
          }
        }
      }

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
      const auto adaptive_buffer =
        make_adaptive_state_buffer(device.Get(), 0u, 1.0f);
      std::array<float, sbs_scene_controller::meta_word_count> meta {};
      meta[static_cast<std::size_t>(
        sbs_scene_controller::meta_word_e::elapsed_seconds
      )] = 1.0f / 60.0f;
      meta[static_cast<std::size_t>(
        sbs_scene_controller::meta_word_e::source_aspect_ratio
      )] = 1.0f;
      meta[static_cast<std::size_t>(
        sbs_scene_controller::meta_word_e::roi_x1
      )] = 1.0f;
      meta[static_cast<std::size_t>(
        sbs_scene_controller::meta_word_e::roi_y1
      )] = 1.0f;
      meta[static_cast<std::size_t>(
        sbs_scene_controller::meta_word_e::state_full_frame
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
      const auto previous_state_buffer = make_structured_buffer(
        device.Get(),
        sbs_scene_controller::initial_word_bits.data(),
        static_cast<UINT>(
          sbs_scene_controller::initial_word_bits.size() *
          sizeof(std::uint32_t)
        ),
        sizeof(std::uint32_t) * 4u
      );
      const scene_constants_t constants;
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
      if (
        std::any_of(
          inputs.begin(),
          inputs.end(),
          [](const auto *view) {
            return view == nullptr;
          }
        ) ||
        !constant_buffer
      ) {
        return false;
      }

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
            const UINT groups_x,
            ID3D11UnorderedAccessView *second_uav = nullptr) {
          context->CSSetShader(shader, nullptr, 0);
          ID3D11UnorderedAccessView *uavs[2] {
            summary.uav.Get(),
            second_uav,
          };
          const UINT uav_count = second_uav ? 2u : 1u;
          context->CSSetUnorderedAccessViews(
            0,
            uav_count,
            uavs,
            nullptr
          );
          context->Dispatch(groups_x, 1u, 1u);
          ID3D11UnorderedAccessView *null_uavs[2] {
            nullptr,
            nullptr,
          };
          context->CSSetUnorderedAccessViews(
            0,
            uav_count,
            null_uavs,
            nullptr
          );
        };
      dispatch(parallel_shaders[0].Get(), 1u);
      dispatch(parallel_shaders[1].Get(), 1u);
      dispatch(parallel_shaders[2].Get(), 2u);
      dispatch(parallel_shaders[3].Get(), 1u);
      dispatch(parallel_shaders[4].Get(), 32u);
      dispatch(
        parallel_shaders[5].Get(),
        1u,
        parallel_evidence.uav.Get()
      );

      context->CSSetShader(serial_shader.Get(), nullptr, 0);
      ID3D11UnorderedAccessView *serial_uav =
        serial_evidence.uav.Get();
      context->CSSetUnorderedAccessViews(
        0,
        1,
        &serial_uav,
        nullptr
      );
      context->Dispatch(1u, 1u, 1u);
      ID3D11UnorderedAccessView *null_uav = nullptr;
      context->CSSetUnorderedAccessViews(
        0,
        1,
        &null_uav,
        nullptr
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

      result.summary = read_buffer<float>(
        device.Get(),
        context.Get(),
        summary.srv.Get(),
        summary_float_count
      );
      result.parallel_evidence = read_buffer<float>(
        device.Get(),
        context.Get(),
        parallel_evidence.srv.Get(),
        sbs_scene_controller::global_out_word_count
      );
      result.serial_evidence = read_buffer<float>(
        device.Get(),
        context.Get(),
        serial_evidence.srv.Get(),
        sbs_scene_controller::global_out_word_count
      );
      return true;
    };

  const std::vector<axis_run_t> four_runs {
    {3u, 32u},
    {34u, 63u},
    {65u, 94u},
    {96u, 125u},
  };
  const dense_case_t one_candidate {
    "one_candidate",
    {{16u, 112u}},
    {{16u, 112u}},
    {},
    1u,
    1u,
    false,
    false,
  };
  const dense_case_t sixteen_candidates {
    "sixteen_candidates",
    four_runs,
    four_runs,
    {},
    16u,
    1u,
    false,
    false,
  };
  const dense_case_t thirty_two_candidates {
    "thirty_two_candidates",
    {
      {1u, 30u},
      {32u, 39u},
      {41u, 48u},
      {50u, 57u},
      {67u, 96u},
      {98u, 105u},
      {107u, 114u},
      {116u, 123u},
    },
    four_runs,
    {62u, 66u},
    32u,
    2u,
    true,
    false,
  };
  const dense_case_t overflow_ambiguous {
    "overflow_ambiguous",
    {
      {3u, 12u},
      {14u, 23u},
      {25u, 34u},
      {36u, 45u},
      {47u, 56u},
    },
    {
      {3u, 12u},
      {14u, 23u},
      {25u, 34u},
      {36u, 45u},
      {47u, 56u},
    },
    {},
    16u,
    1u,
    false,
    true,
  };

  const auto numeric =
    [](const std::vector<float> &values, const std::size_t index) {
      return static_cast<std::uint32_t>(
        std::lround(values[index])
      );
    };
  const auto argmax =
    [](const std::vector<float> &values,
       const std::size_t first,
       const std::size_t end) {
      return static_cast<std::size_t>(std::distance(
        values.begin(),
        std::max_element(
          values.begin() + static_cast<std::ptrdiff_t>(first),
          values.begin() + static_cast<std::ptrdiff_t>(end)
        )
      ));
    };
  const auto verify =
    [&](const dense_case_t &scene, const run_result_t &result) {
      SCOPED_TRACE(scene.label);
      ASSERT_EQ(result.summary.size(), summary_float_count);
      ASSERT_EQ(
        result.parallel_evidence.size(),
        sbs_scene_controller::global_out_word_count
      );
      ASSERT_EQ(
        result.serial_evidence.size(),
        sbs_scene_controller::global_out_word_count
      );
      EXPECT_EQ(
        numeric(result.summary, plan_split),
        scene.expected_split ? 1u : 0u
      );
      EXPECT_EQ(
        numeric(result.summary, plan_region_count),
        scene.expected_regions
      );
      EXPECT_EQ(
        numeric(result.summary, plan_video_ambiguous),
        scene.expected_ambiguous ? 1u : 0u
      );
      std::uint32_t candidate_total = 0u;
      for (std::uint32_t region = 0u;
           region < scene.expected_regions;
           ++region) {
        const std::size_t base =
          region_base + region * region_stride;
        EXPECT_EQ(
          numeric(result.summary, base + region_active),
          1u
        );
        EXPECT_EQ(
          numeric(result.summary, base + region_candidate_first),
          candidate_total
        );
        candidate_total += numeric(
          result.summary,
          base + region_candidate_count
        );
      }
      EXPECT_EQ(candidate_total, scene.expected_candidates);
      for (std::uint32_t candidate = 0u;
           candidate < scene.expected_candidates;
           ++candidate) {
        const std::size_t descriptor =
          candidate_desc_base + candidate * candidate_desc_stride;
        const std::size_t candidate_result =
          candidate_result_base + candidate * candidate_result_stride;
        EXPECT_EQ(
          numeric(result.summary, descriptor + candidate_active),
          1u
        ) << "candidate " << candidate;
        EXPECT_LT(
          numeric(result.summary, descriptor + candidate_first_x),
          numeric(result.summary, descriptor + candidate_end_x)
        ) << "candidate " << candidate;
        EXPECT_LT(
          numeric(result.summary, descriptor + candidate_first_y),
          numeric(result.summary, descriptor + candidate_end_y)
        ) << "candidate " << candidate;
        EXPECT_GT(
          result.summary[candidate_result + candidate_coverage],
          0.0f
        ) << "candidate " << candidate;
      }
      EXPECT_EQ(
        argmax(result.parallel_evidence, 0u, 6u),
        argmax(result.serial_evidence, 0u, 6u)
      );
      EXPECT_EQ(
        argmax(result.parallel_evidence, 6u, 12u),
        argmax(result.serial_evidence, 6u, 12u)
      );
      EXPECT_EQ(
        argmax(result.parallel_evidence, 12u, 21u),
        argmax(result.serial_evidence, 12u, 21u)
      );
      EXPECT_EQ(
        argmax(result.parallel_evidence, 21u, 25u),
        argmax(result.serial_evidence, 21u, 25u)
      );
      for (std::size_t word = 0u;
           word < result.parallel_evidence.size();
           ++word) {
        EXPECT_NEAR(
          result.parallel_evidence[word],
          result.serial_evidence[word],
          3e-4f
        ) << sbs_scene_controller::global_out_names[word];
      }
    };

  run_result_t result;
  ASSERT_TRUE(run_case(one_candidate, result));
  verify(one_candidate, result);
  ASSERT_TRUE(run_case(sixteen_candidates, result));
  verify(sixteen_candidates, result);
  ASSERT_TRUE(run_case(thirty_two_candidates, result));
  verify(thirty_two_candidates, result);

  // Reuse the uncleared scratch buffer across a 32 -> 1 transition. Planning must deactivate the
  // second split region and every stale candidate descriptor, while the candidate pass must zero
  // every inactive result lane before the final reduction reads it.
  ASSERT_TRUE(run_case(one_candidate, result));
  verify(one_candidate, result);
  EXPECT_EQ(
    numeric(result.summary, region_base + region_stride + region_active),
    0u
  );
  for (std::uint32_t candidate = 1u; candidate < 32u; ++candidate) {
    const std::size_t descriptor =
      candidate_desc_base + candidate * candidate_desc_stride;
    const std::size_t candidate_result =
      candidate_result_base + candidate * candidate_result_stride;
    EXPECT_EQ(
      numeric(result.summary, descriptor + candidate_active),
      0u
    ) << "candidate " << candidate;
    for (std::size_t field = 0u;
         field < candidate_result_stride;
         ++field) {
      EXPECT_FLOAT_EQ(
        result.summary[candidate_result + field],
        0.0f
      ) << "candidate " << candidate << ", field " << field;
    }
  }

  ASSERT_TRUE(run_case(overflow_ambiguous, result));
  verify(overflow_ambiguous, result);
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

  constexpr std::uint64_t measured_frames = 640;
  for (
    std::uint64_t frame_id = 1;
    frame_id <= measured_frames;
    ++frame_id
  ) {
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

    // Explicitly serialize this opt-in benchmark so each timestamp interval measures one
    // bounded controller update rather than an artificially deep command queue. Production
    // remains nonblocking and never performs this readback.
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
    sbs_perf::tick();
  }

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
    0.50
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
