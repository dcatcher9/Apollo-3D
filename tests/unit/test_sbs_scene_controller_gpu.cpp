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
  #include <filesystem>
  #include <gtest/gtest.h>
  #include <limits>
  #include <memory>
  #include <src/generated/sbs_adaptive_state_contract.h>
  #include <src/generated/sbs_scene_controller_contract.h>
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
      std::vector<float> roi_rgb(
        3u * depth_width * depth_height,
        0.25f
      );
      roi_rgb_buffer = make_structured_buffer(
        device.Get(),
        roi_rgb.data(),
        static_cast<UINT>(roi_rgb.size() * sizeof(float)),
        sizeof(float)
      );
      ASSERT_TRUE(raw_depth_buffer.srv);
      ASSERT_TRUE(roi_rgb_buffer.srv);

      const std::array<float, 4> frame_state {
        0.0f,
        1.0f,
        1.0f,
        2.0f,
      };
      frame_state_buffer = make_structured_buffer(
        device.Get(),
        frame_state.data(),
        sizeof(frame_state),
        sizeof(frame_state)
      );
      adaptive_buffer = make_structured_buffer(
        device.Get(),
        sbs_adaptive_state::initial_values.data(),
        static_cast<UINT>(sizeof(sbs_adaptive_state::initial_values)),
        sizeof(float) * 4u
      );
      ASSERT_TRUE(frame_state_buffer.srv);
      ASSERT_TRUE(adaptive_buffer.srv);

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
        roi_rgb_buffer.srv.Get(),
        raw_depth_buffer.srv.Get(),
        depth_view.Get(),
        frame_state_buffer.srv.Get(),
        adaptive_buffer.srv.Get(),
        depth_width,
        depth_height
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
    structured_buffer_t roi_rgb_buffer;
    structured_buffer_t frame_state_buffer;
    structured_buffer_t adaptive_buffer;
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
    roi_rgb_buffer.srv.Get(),
    raw_depth_buffer.srv.Get(),
    depth_view.Get(),
    frame_state_buffer.srv.Get(),
    adaptive_buffer.srv.Get(),
    depth_width,
    depth_height
  ));
  EXPECT_FALSE(controller->snapshot().snapshot_available);

  ASSERT_TRUE(controller->resolve_completed(
    frame_id,
    roi_rgb_buffer.srv.Get(),
    raw_depth_buffer.srv.Get(),
    depth_view.Get(),
    frame_state_buffer.srv.Get(),
    adaptive_buffer.srv.Get(),
    depth_width,
    depth_height
  ));
  const auto snapshot = controller->snapshot();
  EXPECT_EQ(snapshot.source_frame_id, frame_id);
  expect_canonical_viewport(read_analysis(snapshot), 0, 28, 128, 72);
  expect_snapshot_abi_invariants(device.Get(), context.Get(), snapshot);
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
    roi_rgb_buffer.srv.Get(),
    raw_depth_buffer.srv.Get(),
    depth_view.Get(),
    frame_state_buffer.srv.Get(),
    adaptive_buffer.srv.Get(),
    depth_width,
    depth_height
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

  const auto first_state = read_rule_state(controller->snapshot());
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

#endif  // _WIN32
