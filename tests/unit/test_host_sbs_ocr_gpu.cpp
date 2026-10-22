/**
 * @file tests/unit/test_host_sbs_ocr_gpu.cpp
 * @brief End-to-end D3D11 WARP coverage for the production OCR preprocess/box producer.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/depth_coordinate_v2.h"
  #include "src/host_sbs_resolution.h"

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstddef>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <limits>
  #include <optional>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace v2 = models::depth_coordinate_v2;

  constexpr std::uint32_t ocr_width = v2::subtitle_ocr_output_width;
  constexpr std::uint32_t ocr_height = v2::subtitle_ocr_output_height;
  constexpr std::uint32_t ocr_pixels = ocr_width * ocr_height;
  constexpr std::uint32_t ocr_cell_width = 8u;
  constexpr std::uint32_t ocr_grid_width = ocr_width / ocr_cell_width;
  constexpr std::uint32_t ocr_cells_per_group = 32u;
  constexpr std::uint32_t ocr_cell_rows_per_group = 4u;
  constexpr std::uint32_t ocr_cell_group_count =
    (ocr_grid_width + ocr_cells_per_group - 1u) / ocr_cells_per_group;
  constexpr std::uint32_t ocr_cell_group_row_count =
    (ocr_height + ocr_cell_rows_per_group - 1u) / ocr_cell_rows_per_group;
  constexpr std::uint32_t ocr_cell_words = 1u;
  constexpr std::uint32_t ocr_cell_stat_words =
    ocr_grid_width * ocr_height * ocr_cell_words;

  constexpr std::uint32_t ceil_crop_height(
    const std::uint32_t width,
    const std::uint32_t height
  ) {
    const auto crop = (
      static_cast<std::uint64_t>(width) * v2::subtitle_ocr_crop_aspect_height +
      v2::subtitle_ocr_crop_aspect_width - 1u
    ) / v2::subtitle_ocr_crop_aspect_width;
    return std::min(height, static_cast<std::uint32_t>(crop));
  }

  constexpr std::uint32_t subtitle_search_top(const std::uint32_t height) {
    const auto search_height = (
      static_cast<std::uint64_t>(height) * v2::subtitle_ocr_line_search_bottom_height_numerator +
      v2::subtitle_ocr_line_search_bottom_height_denominator - 1u
    ) / v2::subtitle_ocr_line_search_bottom_height_denominator;
    return height - static_cast<std::uint32_t>(search_height);
  }

  constexpr std::uint32_t subtitle_roi_edge(
    const std::uint32_t source_w,
    const std::uint32_t source_h,
    const std::uint32_t content_top,
    const std::uint32_t content_height,
    const std::uint32_t safe_row
  ) {
    const auto crop_height = ceil_crop_height(source_w, source_h);
    const auto crop_top = source_h - crop_height;
    const std::uint64_t numerator =
      (static_cast<std::uint64_t>(crop_top) * ocr_height +
       static_cast<std::uint64_t>(safe_row) * crop_height) * content_height;
    const std::uint64_t denominator = static_cast<std::uint64_t>(ocr_height) * source_h;
    return content_top +
           static_cast<std::uint32_t>((numerator + denominator - 1u) / denominator);
  }

  constexpr std::uint32_t subtitle_roi_edge(
    const std::uint32_t source_w,
    const std::uint32_t source_h,
    const std::uint32_t field_h,
    const std::uint32_t safe_row
  ) {
    return subtitle_roi_edge(
      source_w,
      source_h,
      0u,
      field_h,
      safe_row
    );
  }

  constexpr std::uint32_t map_ocr_y_floor(
    const std::uint32_t source_h,
    const std::uint32_t field_h,
    const std::uint32_t crop_top,
    const std::uint32_t crop_height,
    const std::uint32_t y
  ) {
    const std::uint64_t numerator =
      (static_cast<std::uint64_t>(crop_top) * ocr_height +
       static_cast<std::uint64_t>(y) * crop_height) * field_h;
    const std::uint64_t denominator = static_cast<std::uint64_t>(ocr_height) * source_h;
    return static_cast<std::uint32_t>(numerator / denominator);
  }

  constexpr std::uint32_t map_ocr_y_ceil(
    const std::uint32_t source_h,
    const std::uint32_t field_h,
    const std::uint32_t crop_top,
    const std::uint32_t crop_height,
    const std::uint32_t y
  ) {
    const std::uint64_t numerator =
      (static_cast<std::uint64_t>(crop_top) * ocr_height +
       static_cast<std::uint64_t>(y) * crop_height) * field_h;
    const std::uint64_t denominator = static_cast<std::uint64_t>(ocr_height) * source_h;
    return static_cast<std::uint32_t>((numerator + denominator - 1u) / denominator);
  }

  constexpr std::uint32_t source_width = 1920u;
  constexpr std::uint32_t source_height = 1080u;
  constexpr std::uint32_t source_crop_height =
    ceil_crop_height(source_width, source_height);
  constexpr std::uint32_t source_crop_top = source_height - source_crop_height;
  constexpr std::uint32_t field_width = v2::model_calibrated_shapes[0u].width;
  constexpr std::uint32_t field_height = v2::model_calibrated_shapes[0u].height;
  constexpr std::uint32_t roi_top = subtitle_roi_edge(
    source_width, source_height, field_height, v2::subtitle_ocr_safe_row_top);
  constexpr std::uint32_t roi_bottom = subtitle_roi_edge(
    source_width, source_height, field_height, v2::subtitle_ocr_safe_row_bottom);
  constexpr std::uint32_t cut_tag = 0x28632D48u;

  constexpr std::array<std::uint32_t, 4u> default_ocr_box(
    const std::uint32_t left,
    const std::uint32_t top,
    const std::uint32_t right,
    const std::uint32_t bottom
  ) {
    return {
      static_cast<std::uint32_t>(static_cast<std::uint64_t>(left) * field_width / ocr_width),
      map_ocr_y_floor(source_height, field_height, source_crop_top, source_crop_height, top),
      static_cast<std::uint32_t>((static_cast<std::uint64_t>(right) * field_width + ocr_width - 1u) / ocr_width),
      map_ocr_y_ceil(source_height, field_height, source_crop_top, source_crop_height, bottom),
    };
  }

  // Inputs are the independently authored DB-expanded bounds used by each fixture below.
  // The new ordinary line-end allowance is eight detector pixels on X only; this helper does
  // not recalculate production's DB expansion or apply the ordinary allowance to ribbons.
  constexpr std::array<std::uint32_t, 4u> default_ordinary_cover(
    const std::uint32_t left,
    const std::uint32_t top,
    const std::uint32_t right,
    const std::uint32_t bottom
  ) {
    constexpr std::uint32_t horizontal_guard = 8u;
    return default_ocr_box(left > horizontal_guard ? left - horizontal_guard : 0u,
      top, std::min(right + horizontal_guard, ocr_width), bottom);
  }

  struct rgba_t {
    float r;
    float g;
    float b;
    float a;
  };

  struct preprocess_constants_t {
    std::uint32_t source_w;
    std::uint32_t source_h;
    std::uint32_t crop_top;
    std::uint32_t crop_height;
    std::uint32_t color_mode;
    std::uint32_t source_left;
    std::uint32_t source_top;
    std::uint32_t reserved;
  };

  struct ocr_resolve_constants_t {
    std::uint32_t frame_lo;
    std::uint32_t frame_hi;
    std::uint32_t generation_lo;
    std::uint32_t generation_hi;
    std::uint32_t source_w;
    std::uint32_t source_h;
    std::uint32_t field_w;
    std::uint32_t field_h;
    std::uint32_t crop_top;
    std::uint32_t crop_height;
    std::uint32_t roi_top;
    std::uint32_t roi_bottom;
    std::array<std::uint32_t, 4u> tensor_content;
    std::int32_t line_center_twice_source;
    std::uint32_t line_reference_width;
    std::int32_t line_search_top_source;
    std::uint32_t reserved = 0u;
  };

  struct depth_constants_t {
    std::uint32_t target_w;
    std::uint32_t target_h;
    std::uint32_t color_mode;
    float ema_alpha;
    float minmax_alpha;
    std::uint32_t reduce_threads;
    float ema_edge_change;
    float ema_edge_gradient;
    float ema_edge_strength;
    std::array<std::uint32_t, 4u> tensor_content;
    std::array<float, 3u> reserved;
  };

  using v2_constants_t = v2::constants_t;

  struct subtitle_constants_t {
    std::array<std::uint32_t, 4u> field;
    std::array<std::uint32_t, 4u> source;
    std::array<std::uint32_t, 4u> frame;
    std::array<std::uint32_t, 4u> tensor_content;
    std::array<std::uint32_t, 4u> observation {};
  };

  static_assert(sizeof(rgba_t) == 4u * sizeof(float));
  static_assert(sizeof(preprocess_constants_t) == 32u);
  static_assert(sizeof(ocr_resolve_constants_t) == 80u);
  static_assert(sizeof(depth_constants_t) == 64u);
  static_assert(sizeof(v2_constants_t) == 64u);
  static_assert(sizeof(subtitle_constants_t) == 80u);
  static_assert(ocr_width == v2::subtitle_ocr_input_width);
  static_assert(ocr_height == v2::subtitle_ocr_input_height);

  bool compile_compute_shader(
    ID3D11Device *device,
    const std::filesystem::path &path,
    const char *entrypoint,
    ComPtr<ID3D11ComputeShader> &shader,
    std::string &error
  ) {
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    const auto status = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      entrypoint,
      "cs_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0u,
      &bytecode,
      &diagnostics
    );
    if (FAILED(status)) {
      error = diagnostics ?
                std::string(
                  static_cast<const char *>(diagnostics->GetBufferPointer()),
                  diagnostics->GetBufferSize()
                ) :
                std::string("D3DCompileFromFile failed without diagnostics");
      return false;
    }
    if (FAILED(device->CreateComputeShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shader))) {
      error = std::string("CreateComputeShader failed for ") + entrypoint;
      return false;
    }
    return true;
  }

  bool create_structured_buffer(
    ID3D11Device *device,
    const void *initial,
    const std::size_t byte_count,
    const UINT stride,
    const UINT bind_flags,
    ComPtr<ID3D11Buffer> &buffer,
    ComPtr<ID3D11ShaderResourceView> *srv,
    ComPtr<ID3D11UnorderedAccessView> *uav
  ) {
    if (byte_count == 0u || byte_count > std::numeric_limits<UINT>::max()) {
      return false;
    }
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = static_cast<UINT>(byte_count);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = bind_flags;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA data {initial, 0u, 0u};
    if (FAILED(device->CreateBuffer(&desc, initial ? &data : nullptr, &buffer))) {
      return false;
    }
    if (srv && FAILED(device->CreateShaderResourceView(buffer.Get(), nullptr, srv->ReleaseAndGetAddressOf()))) {
      return false;
    }
    return !uav || SUCCEEDED(device->CreateUnorderedAccessView(
                     buffer.Get(),
                     nullptr,
                     uav->ReleaseAndGetAddressOf()
                   ));
  }

  bool create_constant_buffer(
    ID3D11Device *device,
    const void *initial,
    const UINT byte_count,
    ComPtr<ID3D11Buffer> &buffer
  ) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = byte_count;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data {initial, 0u, 0u};
    return SUCCEEDED(device->CreateBuffer(&desc, initial ? &data : nullptr, &buffer));
  }

  template<typename T>
  bool read_buffer(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    ID3D11Buffer *source,
    std::vector<T> &output
  ) {
    D3D11_BUFFER_DESC source_desc {};
    source->GetDesc(&source_desc);
    if (source_desc.ByteWidth % sizeof(T) != 0u) {
      return false;
    }
    auto staging_desc = source_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0u;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0u;
    staging_desc.StructureByteStride = 0u;
    ComPtr<ID3D11Buffer> staging;
    if (FAILED(device->CreateBuffer(&staging_desc, nullptr, &staging))) {
      return false;
    }
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return false;
    }
    output.resize(source_desc.ByteWidth / sizeof(T));
    std::memcpy(output.data(), mapped.pData, source_desc.ByteWidth);
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  bool read_texture(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    ID3D11Texture2D *source,
    std::vector<float> &output
  ) {
    D3D11_TEXTURE2D_DESC source_desc {};
    source->GetDesc(&source_desc);
    auto staging_desc = source_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0u;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging))) {
      return false;
    }
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return false;
    }
    output.resize(static_cast<std::size_t>(source_desc.Width) * source_desc.Height);
    for (std::uint32_t y = 0u; y < source_desc.Height; ++y) {
      const auto *row = reinterpret_cast<const float *>(
        static_cast<const std::byte *>(mapped.pData) +
        static_cast<std::size_t>(y) * mapped.RowPitch
      );
      std::copy_n(
        row,
        source_desc.Width,
        output.begin() + static_cast<std::size_t>(y) * source_desc.Width
      );
    }
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  class ocr_warp_fixture_t {
  public:
    bool initialize(
      std::string &error,
      const bool need_preprocess,
      const bool need_boxes,
      const bool need_locator
    ) {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, requested, static_cast<UINT>(std::size(requested)), D3D11_SDK_VERSION, &device_, &actual, &context_)) || actual < D3D_FEATURE_LEVEL_11_0) {
        return false;
      }

      const auto root = std::filesystem::path(SUNSHINE_SHADERS_DIR);
      if (need_preprocess && !compile_compute_shader(device_.Get(), root / "host_sbs_ocr_preprocess_cs.hlsl", "main", preprocess_, error)) {
        return false;
      }
      if (need_boxes && (!compile_compute_shader(device_.Get(), root / "host_sbs_ocr_boxes_cs.hlsl", "cells_main", cells_, error) || !compile_compute_shader(device_.Get(), root / "host_sbs_ocr_boxes_cs.hlsl", "resolve_main", boxes_, error))) {
        return false;
      }
      if (need_locator && (!compile_compute_shader(device_.Get(), root / "host_sbs_subtitle_locator_cs.hlsl", "resolve_main", locator_resolve_, error) || !compile_compute_shader(device_.Get(), root / "host_sbs_subtitle_locator_cs.hlsl", "condition_main", condition_, error))) {
        return false;
      }

      if (need_preprocess) {
        const preprocess_constants_t zero {};
        if (!create_constant_buffer(
              device_.Get(),
              &zero,
              sizeof(zero),
              preprocess_cb_
            )) {
          return false;
        }
      }
      if (need_boxes && !initialize_box_resources()) {
        return false;
      }
      return !need_locator || initialize_locator_resources();
    }

    bool run_preprocess(
      const std::vector<rgba_t> &source,
      const std::uint32_t width,
      const std::uint32_t height,
      const std::uint32_t crop_top,
      const std::uint32_t crop_height,
      std::vector<float> &output,
      const std::uint32_t analysis_width = 0u,
      const std::uint32_t analysis_height = 0u,
      const std::uint32_t source_left = 0u,
      const std::uint32_t source_top = 0u
    ) {
      if (!preprocess_ || source.size() != static_cast<std::size_t>(width) * height) {
        return false;
      }
      D3D11_TEXTURE2D_DESC texture_desc {};
      texture_desc.Width = width;
      texture_desc.Height = height;
      texture_desc.MipLevels = 1u;
      texture_desc.ArraySize = 1u;
      texture_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
      texture_desc.SampleDesc.Count = 1u;
      texture_desc.Usage = D3D11_USAGE_DEFAULT;
      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA texture_data {
        source.data(),
        width * static_cast<UINT>(sizeof(rgba_t)),
        0u
      };
      ComPtr<ID3D11Texture2D> texture;
      ComPtr<ID3D11ShaderResourceView> texture_srv;
      if (FAILED(device_->CreateTexture2D(&texture_desc, &texture_data, &texture)) || FAILED(device_->CreateShaderResourceView(texture.Get(), nullptr, &texture_srv))) {
        return false;
      }

      ComPtr<ID3D11Buffer> output_buffer;
      ComPtr<ID3D11UnorderedAccessView> output_uav;
      if (!create_structured_buffer(
            device_.Get(),
            nullptr,
            3u * static_cast<std::size_t>(ocr_pixels) * sizeof(float),
            sizeof(float),
            D3D11_BIND_UNORDERED_ACCESS,
            output_buffer,
            nullptr,
            &output_uav
          )) {
        return false;
      }

      const preprocess_constants_t constants {
        analysis_width != 0u ? analysis_width : width,
        analysis_height != 0u ? analysis_height : height,
        crop_top,
        crop_height,
        0u,
        source_left,
        source_top,
        0u,
      };
      context_->UpdateSubresource(
        preprocess_cb_.Get(),
        0u,
        nullptr,
        &constants,
        0u,
        0u
      );
      context_->CSSetShader(preprocess_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 1u, preprocess_cb_.GetAddressOf());
      context_->CSSetShaderResources(0u, 1u, texture_srv.GetAddressOf());
      context_->CSSetUnorderedAccessViews(0u, 1u, output_uav.GetAddressOf(), nullptr);
      context_->Dispatch((ocr_width + 15u) / 16u, (ocr_height + 15u) / 16u, 1u);
      unbind();
      return read_buffer(device_.Get(), context_.Get(), output_buffer.Get(), output);
    }

    bool run_boxes(
      const std::vector<float> &probability,
      const std::uint64_t frame,
      const std::uint64_t generation,
      const std::uint32_t source_w = source_width,
      const std::uint32_t source_h = source_height,
      const std::uint32_t field_w = field_width,
      const std::uint32_t field_h = field_height,
      const std::uint32_t crop_top = source_crop_top,
      const std::uint32_t crop_height = source_crop_height,
      const std::uint32_t field_roi_top = roi_top,
      const std::uint32_t field_roi_bottom = roi_bottom,
      std::array<std::uint32_t, 4u> tensor_content = {},
      const std::optional<std::int32_t> line_center_twice_source = std::nullopt,
      const std::optional<std::uint32_t> line_reference_width = std::nullopt,
      const std::optional<std::int32_t> line_search_top_source = std::nullopt
    ) {
      if (!cells_ || !boxes_ || probability.size() != ocr_pixels) {
        return false;
      }
      context_->UpdateSubresource(
        probability_buffer_.Get(),
        0u,
        nullptr,
        probability.data(),
        0u,
        0u
      );
      if (tensor_content == std::array<std::uint32_t, 4u> {}) {
        tensor_content = {0u, 0u, field_w, field_h};
      }
      const ocr_resolve_constants_t constants {
        static_cast<std::uint32_t>(frame),
        static_cast<std::uint32_t>(frame >> 32u),
        static_cast<std::uint32_t>(generation),
        static_cast<std::uint32_t>(generation >> 32u),
        source_w,
        source_h,
        field_w,
        field_h,
        crop_top,
        crop_height,
        field_roi_top,
        field_roi_bottom,
        tensor_content,
        line_center_twice_source.value_or(static_cast<std::int32_t>(source_w)),
        line_reference_width.value_or(source_w),
        line_search_top_source.value_or(static_cast<std::int32_t>(subtitle_search_top(source_h))),
        0u,
      };
      context_->UpdateSubresource(
        boxes_cb_.Get(),
        0u,
        nullptr,
        &constants,
        0u,
        0u
      );

      context_->CSSetShader(cells_.Get(), nullptr, 0u);
      context_->CSSetShaderResources(0u, 1u, probability_srv_.GetAddressOf());
      context_->CSSetUnorderedAccessViews(0u, 1u, cell_stats_uav_.GetAddressOf(), nullptr);
      context_->Dispatch(ocr_cell_group_count, ocr_cell_group_row_count, 1u);
      unbind();

      context_->CSSetShader(boxes_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 1u, boxes_cb_.GetAddressOf());
      context_->CSSetShaderResources(1u, 1u, cell_stats_srv_.GetAddressOf());
      context_->CSSetUnorderedAccessViews(1u, 1u, record_uav_.GetAddressOf(), nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind();
      return read_buffer(device_.Get(), context_.Get(), record_buffer_.Get(), record_);
    }

    bool run_boxes_in_screen(
      const std::vector<float> &probability,
      const std::uint64_t frame,
      const std::uint64_t generation,
      const std::uint32_t screen_width,
      const std::uint32_t analysis_left,
      const std::uint32_t screen_height = 2160u,
      const std::uint32_t analysis_top = 1080u
    ) {
      return run_boxes(
        probability, frame, generation, source_width, source_height,
        field_width, field_height, source_crop_top, source_crop_height,
        roi_top, roi_bottom, {},
        static_cast<std::int32_t>(screen_width) - 2 * static_cast<std::int32_t>(analysis_left),
        screen_width,
        static_cast<std::int32_t>(subtitle_search_top(screen_height)) -
          static_cast<std::int32_t>(analysis_top));
    }

    bool dispatch_locator(
      const std::uint64_t frame,
      const std::uint64_t generation,
      const bool reset = false,
      const bool enabled = true
    ) {
      if (!locator_resolve_ || !condition_) {
        return false;
      }
      const subtitle_constants_t constants {
        {field_width, field_height, roi_top, roi_bottom},
        {source_width, source_height, enabled ? 1u : 0u, reset ? 1u : 0u},
        {
          static_cast<std::uint32_t>(frame),
          static_cast<std::uint32_t>(frame >> 32u),
          static_cast<std::uint32_t>(generation),
          static_cast<std::uint32_t>(generation >> 32u),
        },
        {0u, 0u, field_width, field_height},
        {static_cast<std::uint32_t>(frame), static_cast<std::uint32_t>(frame >> 32u), 0u, 0u},
      };
      // This current-publication fixture uses an authored microsecond cadence, shared by OCR and the
      // sealed Base publication. It exercises no artificial UI dwell or model inference.
      coordinate_constants_.joint_observation_timestamp_low = static_cast<std::uint32_t>(frame);
      coordinate_constants_.joint_observation_timestamp_high = static_cast<std::uint32_t>(frame >> 32u);
      context_->UpdateSubresource(v2_cb_.Get(), 0u, nullptr, &coordinate_constants_, 0u, 0u);
      upload_geometry_publication(frame);
      context_->UpdateSubresource(
        subtitle_cb_.Get(),
        0u,
        nullptr,
        &constants,
        0u,
        0u
      );
      std::array<ID3D11Buffer *, 3u> constant_buffers {
        depth_cb_.Get(),
        v2_cb_.Get(),
        subtitle_cb_.Get()
      };
      context_->CSSetConstantBuffers(
        0u,
        static_cast<UINT>(constant_buffers.size()),
        constant_buffers.data()
      );

      context_->CSSetShader(locator_resolve_.Get(), nullptr, 0u);
      std::array<ID3D11ShaderResourceView *, 8u> resolve_srvs {
        geometry_state_srv_.Get(),
        cut_srv_.Get(),
        base_srv_.Get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        record_srv_.Get(),
      };
      context_->CSSetShaderResources(
        0u,
        static_cast<UINT>(resolve_srvs.size()),
        resolve_srvs.data()
      );
      std::array<ID3D11UnorderedAccessView *, 3u> resolve_uavs {
        state_uav_.Get(), nullptr, condition_params_uav_.Get()
      };
      context_->CSSetUnorderedAccessViews(
        2u,
        static_cast<UINT>(resolve_uavs.size()),
        resolve_uavs.data(),
        nullptr
      );
      context_->Dispatch(1u, 1u, 1u);
      unbind();

      if (!author_current_contour(constants)) return false;
      context_->CSSetShader(condition_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(
        0u,
        static_cast<UINT>(constant_buffers.size()),
        constant_buffers.data()
      );
      std::array<ID3D11ShaderResourceView *, 8u> condition_srvs {
        geometry_state_srv_.Get(),
        cut_srv_.Get(),
        base_srv_.Get(),
        state_srv_.Get(),
        condition_params_srv_.Get(),
        contour_distance_srv_.Get(),
        contour_identity_srv_.Get(),
        record_srv_.Get()
      };
      context_->CSSetShaderResources(
        0u,
        static_cast<UINT>(condition_srvs.size()),
        condition_srvs.data()
      );
      std::array<ID3D11UnorderedAccessView *, 3u> condition_uavs {
        output_uav_.Get(), gaussian_control_uav_.Get(), gaussian_weight_uav_.Get()
      };
      context_->CSSetUnorderedAccessViews(
        3u, static_cast<UINT>(condition_uavs.size()), condition_uavs.data(), nullptr);
      context_->Dispatch((field_width + 15u) / 16u, (field_height + 15u) / 16u, 1u);
      unbind();

      return read_buffer(device_.Get(), context_.Get(), state_buffer_.Get(), state_) &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_) &&
             read_buffer(device_.Get(), context_.Get(), gaussian_control_buffer_.Get(), gaussian_control_) &&
             read_texture(device_.Get(), context_.Get(), gaussian_weight_texture_.Get(), gaussian_weight_);
    }

    void reset_locator() {
      const UINT zero[4] = {};
      context_->ClearUnorderedAccessViewUint(state_uav_.Get(), zero);
      state_.clear();
      output_.clear();
      gaussian_control_.clear();
      gaussian_weight_.clear();
    }

    const std::vector<std::uint32_t> &record() const {
      return record_;
    }

    const std::vector<std::uint32_t> &state() const {
      return state_;
    }

    bool output_is_exact_base() const {
      if (output_.size() != base_.size()) {
        return false;
      }
      for (std::size_t index = 0u; index < base_.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(output_[index]) != std::bit_cast<std::uint32_t>(base_[index])) {
          return false;
        }
      }
      return true;
    }

    float output_at(const std::uint32_t x, const std::uint32_t y) const {
      return output_.at(static_cast<std::size_t>(y) * field_width + x);
    }

    std::uint32_t smoothing_enabled() const {
      return gaussian_control_.at(0u);
    }

    float weight_at(const std::uint32_t x, const std::uint32_t y) const {
      return gaussian_weight_.at(static_cast<std::size_t>(y) * field_width + x);
    }

  private:
    bool author_current_contour(const subtitle_constants_t &constants) {
      std::vector<std::uint32_t> params;
      if (!read_buffer(device_.Get(), context_.Get(), state_buffer_.Get(), state_) ||
          !read_buffer(device_.Get(), context_.Get(), condition_params_buffer_.Get(), params)) return false;
      const auto count = record_[4u];
      const bool current_valid = count != 0u && count <= v2::subtitle_plane_cover_capacity &&
        params[0u] == v2::subtitle_condition_param_schema && params[1u] == v2::subtitle_condition_param_tag &&
        params[v2::subtitle_condition_param_current_count_word] == count &&
        params[v2::subtitle_condition_param_active_word] == v2::subtitle_condition_param_fully_active_value &&
        state_[v2::subtitle_plane_flags_word] == v2::subtitle_plane_current_valid_flag &&
        state_[v2::subtitle_plane_current_slot_mask_word] == (1u << count) - 1u &&
        params[v2::subtitle_condition_param_target_word] == state_[v2::subtitle_plane_adaptive_offset + 3u];
      const float infinity = std::bit_cast<float>(v2::subtitle_contour_distance_empty_bits);
      std::vector<float> distance(static_cast<std::size_t>(field_width) * field_height, infinity);
      // These OCR topology/current-publication tests author a squared-source-pixel distance to
      // every exact current OCR8 cover's field-node centers. This is an analytic conditioner input, not
      // the native five-center distance producer or a claim about glyph-mask quality.
      const double source_x_step = static_cast<double>(source_width) / field_width;
      const double source_y_step = static_cast<double>(source_height) / field_height;
      for (std::uint32_t y = 0u; y < field_height; ++y) {
        for (std::uint32_t x = 0u; x < field_width; ++x) {
          const auto index = static_cast<std::size_t>(y) * field_width + x;
          for (std::uint32_t slot = 0u; current_valid && slot < count; ++slot) {
            const auto offset = v2::subtitle_ocr_final_box_offset + slot * v2::subtitle_ocr_box_word_count;
            const auto nearest_x = std::clamp(x, record_[offset], record_[offset + 2u] - 1u);
            const auto nearest_y = std::clamp(y, record_[offset + 1u], record_[offset + 3u] - 1u);
            const double dx = (static_cast<double>(x) - nearest_x) * source_x_step;
            const double dy = (static_cast<double>(y) - nearest_y) * source_y_step;
            distance[index] = std::min(distance[index], static_cast<float>(dx * dx + dy * dy));
          }
        }
      }
      context_->UpdateSubresource(contour_distance_texture_.Get(), 0u, nullptr, distance.data(),
        field_width * static_cast<UINT>(sizeof(float)), 0u);
      std::array<std::uint32_t, 32u> identity {};
      identity[0u] = v2::subtitle_contour_schema;
      identity[1u] = v2::subtitle_contour_history_tag;
      identity[2u] = current_valid ? 1u : 0u;
      identity[3u] = current_valid ? state_[v2::subtitle_plane_cut_epoch_word] : 0u;
      std::copy(constants.frame.begin(), constants.frame.end(), identity.begin() + 4u);
      identity[8u] = source_width;
      identity[9u] = source_height;
      identity[10u] = source_crop_top;
      identity[11u] = source_crop_height;
      std::copy(constants.field.begin(), constants.field.end(), identity.begin() + 12u);
      std::copy(constants.tensor_content.begin(), constants.tensor_content.end(), identity.begin() + 16u);
      identity[20u] = constants.observation[0u];
      identity[21u] = constants.observation[1u];
      identity[22u] = v2::subtitle_contour_recipe;
      identity[v2::subtitle_contour_history_snapshot_valid_word] = current_valid ? 1u : 0u;
      context_->UpdateSubresource(contour_identity_buffer_.Get(), 0u, nullptr, identity.data(), 0u, 0u);
      return true;
    }

    bool initialize_box_resources() {
      std::vector<float> probability(ocr_pixels, 0.0f);
      if (!create_structured_buffer(
            device_.Get(),
            probability.data(),
            probability.size() * sizeof(float),
            sizeof(float),
            D3D11_BIND_SHADER_RESOURCE,
            probability_buffer_,
            &probability_srv_,
            nullptr
          )) {
        return false;
      }

      std::vector<std::uint32_t> zeros(ocr_cell_stat_words, 0u);
      if (!create_structured_buffer(
            device_.Get(),
            zeros.data(),
            zeros.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            cell_stats_buffer_,
            &cell_stats_srv_,
            &cell_stats_uav_
          )) {
        return false;
      }

      zeros.assign(v2::subtitle_ocr_record_word_count, 0u);
      if (!create_structured_buffer(
            device_.Get(),
            zeros.data(),
            zeros.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            record_buffer_,
            &record_srv_,
            &record_uav_
          )) {
        return false;
      }

      const ocr_resolve_constants_t zero {};
      return create_constant_buffer(device_.Get(), &zero, sizeof(zero), boxes_cb_);
    }

    bool initialize_locator_resources() {
      std::vector<std::uint32_t> zeros(v2::subtitle_plane_state_word_count, 0u);
      if (!create_structured_buffer(
            device_.Get(),
            zeros.data(),
            zeros.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            state_buffer_,
            &state_srv_,
            &state_uav_
          )) {
        return false;
      }

      zeros.assign(v2::subtitle_condition_param_word_count, 0u);
      if (!create_structured_buffer(
            device_.Get(),
            zeros.data(),
            zeros.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            condition_params_buffer_,
            &condition_params_srv_,
            &condition_params_uav_
          )) {
        return false;
      }

      const std::uint32_t smoothing_disabled = 0u;
      if (!create_structured_buffer(
            device_.Get(), &smoothing_disabled, sizeof(smoothing_disabled), sizeof(smoothing_disabled),
            D3D11_BIND_UNORDERED_ACCESS, gaussian_control_buffer_, nullptr, &gaussian_control_uav_)) {
        return false;
      }

      std::array<std::uint32_t, 32u> cut_words {};
      cut_words[0u] = cut_tag;
      if (!create_structured_buffer(
            device_.Get(),
            cut_words.data(),
            sizeof(cut_words),
            4u * sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE,
            cut_buffer_,
            &cut_srv_,
            nullptr
          )) {
        return false;
      }

      const auto geometry_initial = v2::state_initial_words;
      if (!create_structured_buffer(
            device_.Get(), geometry_initial.data(), sizeof(geometry_initial),
            4u * sizeof(std::uint32_t), D3D11_BIND_SHADER_RESOURCE,
            geometry_state_buffer_, &geometry_state_srv_, nullptr)) {
        return false;
      }

      base_.assign(static_cast<std::size_t>(field_width) * field_height, 0.03f);
      for (std::uint32_t y = 380u; y < 398u; ++y) {
        std::fill_n(
          base_.begin() + static_cast<std::size_t>(y) * field_width,
          field_width,
          0.01f
        );
      }
      D3D11_TEXTURE2D_DESC texture_desc {};
      texture_desc.Width = field_width;
      texture_desc.Height = field_height;
      texture_desc.MipLevels = 1u;
      texture_desc.ArraySize = 1u;
      texture_desc.Format = DXGI_FORMAT_R32_FLOAT;
      texture_desc.SampleDesc.Count = 1u;
      texture_desc.Usage = D3D11_USAGE_DEFAULT;
      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA base_data {
        base_.data(),
        field_width * static_cast<UINT>(sizeof(float)),
        0u
      };
      if (FAILED(device_->CreateTexture2D(&texture_desc, &base_data, &base_texture_)) || FAILED(device_->CreateShaderResourceView(base_texture_.Get(), nullptr, &base_srv_))) {
        return false;
      }
      texture_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      if (FAILED(device_->CreateTexture2D(&texture_desc, nullptr, &output_texture_)) || FAILED(device_->CreateUnorderedAccessView(output_texture_.Get(), nullptr, &output_uav_))) {
        return false;
      }

      if (FAILED(device_->CreateTexture2D(&texture_desc, nullptr, &gaussian_weight_texture_)) ||
          FAILED(device_->CreateUnorderedAccessView(gaussian_weight_texture_.Get(), nullptr, &gaussian_weight_uav_))) {
        return false;
      }

      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      if (FAILED(device_->CreateTexture2D(&texture_desc, nullptr, &contour_distance_texture_)) ||
          FAILED(device_->CreateShaderResourceView(contour_distance_texture_.Get(), nullptr, &contour_distance_srv_))) return false;
      const std::array<std::uint32_t, 32u> contour_identity {};
      if (!create_structured_buffer(
            device_.Get(), contour_identity.data(), sizeof(contour_identity), sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE, contour_identity_buffer_, &contour_identity_srv_, nullptr)) return false;

      const depth_constants_t depth_constants {
        field_width,
        field_height,
        0u,
        0.0f,
        0.0f,
        0u,
        0.0f,
        0.0f,
        0.0f,
        {0u, 0u, field_width, field_height},
        {}
      };
      coordinate_constants_ = v2_constants_t {
        0.04f, 0.0001f, 0.0f, 0.0f, 1.0f, 0.5f, 0.04f, 0.0f,
        v2::adaptive_policy_id, {}, 0u, 0u, {}
      };
      const subtitle_constants_t subtitle_constants {
        {field_width, field_height, roi_top, roi_bottom},
        {source_width, source_height, 1u, 0u},
        {},
        {0u, 0u, field_width, field_height},
      };
      return create_constant_buffer(
               device_.Get(),
               &depth_constants,
               sizeof(depth_constants),
               depth_cb_
             ) &&
             create_constant_buffer(
               device_.Get(),
               &coordinate_constants_,
               sizeof(coordinate_constants_),
               v2_cb_
             ) &&
             create_constant_buffer(
               device_.Get(),
               &subtitle_constants,
               sizeof(subtitle_constants),
               subtitle_cb_
             );
    }

    void upload_geometry_publication(const std::uint64_t timestamp) {
      auto words = v2::state_initial_words;
      const auto put_float = [&words](const std::size_t index, const float value) {
        words[index] = std::bit_cast<std::uint32_t>(value);
      };
      const auto put_clock = [&words](const std::size_t index, const std::uint64_t value) {
        words[index] = static_cast<std::uint32_t>(value);
        words[index + 1u] = static_cast<std::uint32_t>(value >> 32u);
      };
      const auto source_time = std::max(timestamp, std::uint64_t {1u});
      put_float(v2::center, 0.0f);
      put_float(v2::inverse_scale, 1.0f / coordinate_constants_.raw_coordinate_scale);
      put_float(v2::convergence_curve, 0.0f);
      put_float(v2::container_scale, 1.0f);
      words[v2::calibration_revision] = 1u;
      put_float(v2::frame_valid, timestamp != 0u ? 1.0f : 0.0f);
      words[v2::contract_tag_bits] = v2::contract_tag;
      words[v2::renderer_authorization_bits] = timestamp != 0u ? v2::contract_tag : 0u;
      words[v2::joint_plane_mode_bits] = v2::adaptive_policy_id;
      put_clock(v2::gain_last_observation_low, source_time);
      words[v2::gain_clock_armed] = timestamp != 0u ? 1u : 0u;
      words[v2::gain_seed_count] = 1u;
      put_float(v2::gain_target_zero, 0.0f);
      put_float(v2::gain_target_inverse_scale, 1.0f / coordinate_constants_.raw_coordinate_scale);
      put_float(v2::gain_target_nearest, coordinate_constants_.raw_coordinate_scale);
      put_float(v2::gain_display_limit, v2::direct_container_limit);
      put_clock(v2::gain_seed_first_low, source_time);
      put_clock(v2::gain_seed_last_low, source_time);
      put_float(v2::gain_seed_mean_nearest, coordinate_constants_.raw_coordinate_scale);
      put_float(v2::gain_seed_mean_zero, 0.0f);
      words[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(words);
      context_->UpdateSubresource(geometry_state_buffer_.Get(), 0u, nullptr, words.data(), 0u, 0u);
    }

    void unbind() {
      std::array<ID3D11ShaderResourceView *, 8u> null_srvs {};
      std::array<ID3D11UnorderedAccessView *, 6u> null_uavs {};
      std::array<ID3D11Buffer *, 3u> null_cbuffers {};
      context_->CSSetShaderResources(
        0u,
        static_cast<UINT>(null_srvs.size()),
        null_srvs.data()
      );
      context_->CSSetUnorderedAccessViews(
        0u,
        static_cast<UINT>(null_uavs.size()),
        null_uavs.data(),
        nullptr
      );
      context_->CSSetConstantBuffers(
        0u,
        static_cast<UINT>(null_cbuffers.size()),
        null_cbuffers.data()
      );
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> preprocess_;
    ComPtr<ID3D11ComputeShader> cells_;
    ComPtr<ID3D11ComputeShader> boxes_;
    ComPtr<ID3D11ComputeShader> locator_resolve_;
    ComPtr<ID3D11ComputeShader> condition_;
    ComPtr<ID3D11Buffer> preprocess_cb_;
    ComPtr<ID3D11Buffer> boxes_cb_;
    ComPtr<ID3D11Buffer> probability_buffer_;
    ComPtr<ID3D11ShaderResourceView> probability_srv_;
    ComPtr<ID3D11Buffer> cell_stats_buffer_;
    ComPtr<ID3D11ShaderResourceView> cell_stats_srv_;
    ComPtr<ID3D11UnorderedAccessView> cell_stats_uav_;
    ComPtr<ID3D11Buffer> record_buffer_;
    ComPtr<ID3D11ShaderResourceView> record_srv_;
    ComPtr<ID3D11UnorderedAccessView> record_uav_;
    ComPtr<ID3D11Buffer> state_buffer_;
    ComPtr<ID3D11ShaderResourceView> state_srv_;
    ComPtr<ID3D11UnorderedAccessView> state_uav_;
    ComPtr<ID3D11Buffer> condition_params_buffer_;
    ComPtr<ID3D11ShaderResourceView> condition_params_srv_;
    ComPtr<ID3D11UnorderedAccessView> condition_params_uav_;
    ComPtr<ID3D11Buffer> gaussian_control_buffer_;
    ComPtr<ID3D11UnorderedAccessView> gaussian_control_uav_;
    ComPtr<ID3D11Texture2D> gaussian_weight_texture_;
    ComPtr<ID3D11UnorderedAccessView> gaussian_weight_uav_;
    ComPtr<ID3D11Buffer> cut_buffer_;
    ComPtr<ID3D11ShaderResourceView> cut_srv_;
    ComPtr<ID3D11Buffer> geometry_state_buffer_;
    ComPtr<ID3D11ShaderResourceView> geometry_state_srv_;
    ComPtr<ID3D11Texture2D> base_texture_;
    ComPtr<ID3D11ShaderResourceView> base_srv_;
    ComPtr<ID3D11Texture2D> contour_distance_texture_;
    ComPtr<ID3D11ShaderResourceView> contour_distance_srv_;
    ComPtr<ID3D11Buffer> contour_identity_buffer_;
    ComPtr<ID3D11ShaderResourceView> contour_identity_srv_;
    ComPtr<ID3D11Texture2D> output_texture_;
    ComPtr<ID3D11UnorderedAccessView> output_uav_;
    ComPtr<ID3D11Buffer> depth_cb_;
    ComPtr<ID3D11Buffer> v2_cb_;
    ComPtr<ID3D11Buffer> subtitle_cb_;
    v2_constants_t coordinate_constants_ {};
    std::vector<std::uint32_t> record_;
    std::vector<std::uint32_t> state_;
    std::vector<float> base_;
    std::vector<float> output_;
    std::vector<std::uint32_t> gaussian_control_;
    std::vector<float> gaussian_weight_;
  };

  void paint_rectangle(
    std::vector<float> &probability,
    const std::uint32_t left,
    const std::uint32_t top,
    const std::uint32_t right,
    const std::uint32_t bottom,
    const float value = 0.9f
  ) {
    for (std::uint32_t y = top; y < bottom; ++y) {
      std::fill(
        probability.begin() + static_cast<std::size_t>(y) * ocr_width + left,
        probability.begin() + static_cast<std::size_t>(y) * ocr_width + right,
        value
      );
    }
  }

  std::array<std::uint32_t, 4u> read_box(
    const std::vector<std::uint32_t> &record,
    const std::uint32_t offset
  ) {
    return {
      record.at(offset + 0u),
      record.at(offset + 1u),
      record.at(offset + 2u),
      record.at(offset + 3u),
    };
  }

  TEST(HostSbsOcrGpuTest, PreprocessUsesBottomSixToOneCropAndBgrImageNetNormalization) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, true, false, false)) << error;

    constexpr std::uint32_t width = ocr_width;
    constexpr std::uint32_t height = 200u;
    constexpr std::uint32_t crop_top = height - ocr_height;
    std::vector<rgba_t> source(
      static_cast<std::size_t>(width) * height,
      rgba_t {0.1f, 0.2f, 0.3f, 1.0f}
    );
    // This row is immediately above the crop and must not leak into output row zero.
    std::fill_n(
      source.begin() + static_cast<std::size_t>(crop_top - 1u) * width,
      width,
      rgba_t {1.0f, 0.0f, 1.0f, 1.0f}
    );
    source[static_cast<std::size_t>(height - 1u) * width + 20u] =
      rgba_t {0.4f, 0.5f, 0.6f, 1.0f};

    std::vector<float> output;
    ASSERT_TRUE(fixture.run_preprocess(
      source,
      width,
      height,
      crop_top,
      ocr_height,
      output
    ));
    ASSERT_EQ(output.size(), 3u * static_cast<std::size_t>(ocr_pixels));
    const auto sample = [&](const std::uint32_t channel, const std::uint32_t x, const std::uint32_t y) {
      return output.at(
        static_cast<std::size_t>(channel) * ocr_pixels +
        static_cast<std::size_t>(y) * ocr_width + x
      );
    };

    EXPECT_NEAR(sample(0u, 10u, 0u), (0.3f - 0.485f) / 0.229f, 1.0e-5f);
    EXPECT_NEAR(sample(1u, 10u, 0u), (0.2f - 0.456f) / 0.224f, 1.0e-5f);
    EXPECT_NEAR(sample(2u, 10u, 0u), (0.1f - 0.406f) / 0.225f, 1.0e-5f);
    EXPECT_NEAR(sample(0u, 20u, 159u), (0.6f - 0.485f) / 0.229f, 1.0e-5f);
    EXPECT_NEAR(sample(1u, 20u, 159u), (0.5f - 0.456f) / 0.224f, 1.0e-5f);
    EXPECT_NEAR(sample(2u, 20u, 159u), (0.4f - 0.406f) / 0.225f, 1.0e-5f);
  }

  TEST(HostSbsOcrGpuTest, DirectRoiSamplingMatchesThePriorCropTextureBitwise) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, true, false, false)) << error;

    constexpr std::uint32_t texture_width = 160u;
    constexpr std::uint32_t texture_height = 130u;
    constexpr std::uint32_t source_left = 23u;
    constexpr std::uint32_t source_top = 17u;
    constexpr std::uint32_t analysis_width = 120u;
    constexpr std::uint32_t analysis_height = 90u;
    constexpr std::uint32_t crop_height = 20u;
    constexpr std::uint32_t crop_top = analysis_height - crop_height;
    std::vector<rgba_t> full_source(
      static_cast<std::size_t>(texture_width) * texture_height,
      rgba_t {31.0f, -17.0f, 43.0f, 1.0f}
    );
    std::vector<rgba_t> cropped_source;
    cropped_source.reserve(
      static_cast<std::size_t>(analysis_width) * analysis_height
    );
    for (std::uint32_t y = 0u; y < analysis_height; ++y) {
      for (std::uint32_t x = 0u; x < analysis_width; ++x) {
        const rgba_t pixel {
          static_cast<float>(x + 3u * y) / 389.0f,
          static_cast<float>(5u * x + y) / 691.0f,
          static_cast<float>(2u * x + 7u * y) / 887.0f,
          1.0f,
        };
        full_source[
          static_cast<std::size_t>(source_top + y) * texture_width +
          source_left + x
        ] = pixel;
        cropped_source.push_back(pixel);
      }
    }

    std::vector<float> crop_output;
    std::vector<float> direct_output;
    ASSERT_TRUE(fixture.run_preprocess(
      cropped_source,
      analysis_width,
      analysis_height,
      crop_top,
      crop_height,
      crop_output
    ));
    ASSERT_TRUE(fixture.run_preprocess(
      full_source,
      texture_width,
      texture_height,
      crop_top,
      crop_height,
      direct_output,
      analysis_width,
      analysis_height,
      source_left,
      source_top
    ));
    ASSERT_EQ(direct_output.size(), crop_output.size());
    for (std::size_t index = 0u; index < crop_output.size(); ++index) {
      EXPECT_EQ(
        std::bit_cast<std::uint32_t>(direct_output[index]),
        std::bit_cast<std::uint32_t>(crop_output[index])
      ) << "OCR input index=" << index;
    }
  }

  TEST(HostSbsOcrGpuTest, ProbabilityMapProducesCenteredOneAndTwoLineOcr8Records) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;

    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 240u, 100u, 720u, 108u);
    ASSERT_TRUE(fixture.run_boxes(probability, 1u, 77u));
    const auto &one = fixture.record();
    ASSERT_EQ(one.size(), v2::subtitle_ocr_record_word_count);
    EXPECT_EQ(one[0u], v2::subtitle_ocr_record_schema);
    EXPECT_EQ(one[1u], v2::subtitle_ocr_record_tag);
    EXPECT_EQ(one[2u], 1u);
    EXPECT_EQ(one[3u], 1u);
    EXPECT_EQ(one[4u], 1u);
    EXPECT_EQ(one[5u], 1u);
    EXPECT_EQ(one[7u], 77u);
    EXPECT_EQ(one[9u], source_width);
    EXPECT_EQ(one[10u], source_height);
    EXPECT_EQ(one[11u], field_width);
    EXPECT_EQ(one[12u], field_height);
    EXPECT_EQ(one[13u], roi_top);
    EXPECT_EQ(one[14u], roi_bottom);
    EXPECT_EQ(
      read_box(one, v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(240u, 100u, 720u, 108u)
    );
    EXPECT_EQ(
      read_box(one, v2::subtitle_ocr_final_box_offset),
      default_ordinary_cover(234u, 94u, 726u, 114u)
    );
    EXPECT_NEAR(
      std::bit_cast<float>(one[v2::subtitle_ocr_raw_box_offset + 4u]),
      0.9f,
      0.001f
    );
    EXPECT_EQ(one[v2::subtitle_ocr_raw_box_offset + 5u], 0u);
    EXPECT_EQ(one[v2::subtitle_ocr_raw_box_offset + 6u], 1u);
    EXPECT_EQ(one[v2::subtitle_ocr_raw_box_offset + 7u], 0u);
    EXPECT_EQ(one[v2::subtitle_ocr_final_box_offset + 5u], 0u);
    EXPECT_EQ(one[v2::subtitle_ocr_final_box_offset + 6u], 1u);
    EXPECT_EQ(one[v2::subtitle_ocr_final_box_offset + 7u], 0u);

    paint_rectangle(probability, 300u, 120u, 660u, 128u);
    ASSERT_TRUE(fixture.run_boxes(probability, 2u, 77u));
    const auto &two = fixture.record();
    EXPECT_EQ(two[2u], 1u);
    EXPECT_EQ(two[3u], 2u);
    EXPECT_EQ(two[4u], 2u);
    EXPECT_EQ(
      read_box(two, v2::subtitle_ocr_final_box_offset),
      default_ordinary_cover(234u, 94u, 726u, 114u)
    );
    EXPECT_EQ(
      read_box(
        two,
        v2::subtitle_ocr_final_box_offset + v2::subtitle_ocr_box_word_count
      ),
      default_ordinary_cover(294u, 114u, 666u, 134u)
    );
  }

  TEST(HostSbsOcrGpuTest, OrdinaryCoverAddsOneHorizontalCellAndKeepsVerticalDbBounds) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    struct cover_case_t {
      std::array<std::uint32_t, 4u> core;
      std::array<std::uint32_t, 4u> db_cover;
    };
    const std::array<cover_case_t, 3u> cases {{
      {{240u, 100u, 720u, 108u}, {234u, 94u, 726u, 114u}},
      // Stay inside center eligibility.
      {{0u, 100u, 776u, 108u}, {0u, 94u, 782u, 114u}},
      {{192u, 100u, 960u, 108u}, {186u, 94u, 960u, 114u}},
    }};
    for (const auto &test_case : cases) {
      SCOPED_TRACE(std::to_string(test_case.core[0u]));
      const auto &core = test_case.core;
      const auto &db = test_case.db_cover;
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, core[0u], core[1u], core[2u], core[3u]);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
      ASSERT_EQ(fixture.record()[2u], 1u);
      ASSERT_EQ(fixture.record()[3u], 1u);
      ASSERT_EQ(fixture.record()[4u], 1u);
      EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
        default_ocr_box(core[0u], core[1u], core[2u], core[3u]));
      const auto cover = read_box(fixture.record(), v2::subtitle_ocr_final_box_offset);
      EXPECT_EQ(cover, default_ordinary_cover(db[0u], db[1u], db[2u], db[3u]));
      const auto previous_db_cover = default_ocr_box(db[0u], db[1u], db[2u], db[3u]);
      EXPECT_EQ(cover[1u], previous_db_cover[1u]);
      EXPECT_EQ(cover[3u], previous_db_cover[3u]);
      if (db[0u] == 0u) {
        EXPECT_EQ(cover[0u], 0u);
      } else {
        EXPECT_LT(cover[0u], previous_db_cover[0u]);
      }
      if (db[2u] == ocr_width) {
        EXPECT_EQ(cover[2u], field_width);
      } else {
        EXPECT_GT(cover[2u], previous_db_cover[2u]);
      }
      EXPECT_EQ(fixture.record()[v2::subtitle_ocr_raw_box_offset + 5u], 0u);
    }
  }

  TEST(HostSbsOcrGpuTest, ColumnEdgesOnExactFieldMultiplesUseTheExactCeil) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    // On the 770-column field, 576 * 770 / 960 = 462 and 768 * 770 / 960 = 616 are exact. A float
    // multiply by the rounded-up reciprocal of 960 lifted both just above the integer, so ceil
    // returned one extra field column. The first case puts that edge on the core, the second on
    // its expanded cover (DB pad 6 plus the eight-pixel guard).
    struct edge_case_t {
      std::array<std::uint32_t, 4u> core;
      std::array<std::uint32_t, 4u> db_cover;
    };
    const std::array<edge_case_t, 2u> cases {{
      {{384u, 100u, 576u, 108u}, {378u, 94u, 582u, 114u}},
      {{206u, 100u, 754u, 108u}, {200u, 94u, 760u, 114u}},
    }};
    static_assert(576u * field_width % ocr_width == 0u);
    static_assert((760u + 8u) * field_width % ocr_width == 0u);
    for (const auto &test_case : cases) {
      SCOPED_TRACE(std::to_string(test_case.core[2u]));
      const auto &core = test_case.core;
      const auto &db = test_case.db_cover;
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, core[0u], core[1u], core[2u], core[3u]);
      ASSERT_TRUE(fixture.run_boxes(probability, 41u, 77u));
      ASSERT_EQ(fixture.record()[2u], 1u);
      ASSERT_EQ(fixture.record()[3u], 1u);
      ASSERT_EQ(fixture.record()[4u], 1u);
      EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
        default_ocr_box(core[0u], core[1u], core[2u], core[3u]));
      EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_final_box_offset),
        default_ordinary_cover(db[0u], db[1u], db[2u], db[3u]));
    }
  }

  TEST(HostSbsOcrGpuTest, OrdinaryHorizontalGuardIncludesModelMissedLeadingGlyph) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    constexpr std::uint32_t native_w = 1806u;
    constexpr std::uint32_t native_h = 1016u;
    constexpr std::uint32_t native_field_w = 1540u;
    constexpr std::uint32_t native_field_h = 868u;
    constexpr auto crop_h = ceil_crop_height(native_w, native_h);
    constexpr auto crop_top = native_h - crop_h;
    constexpr auto native_roi_top = subtitle_roi_edge(native_w, native_h,
      native_field_h, v2::subtitle_ocr_safe_row_top);
    constexpr auto native_roi_bottom = subtitle_roi_edge(native_w, native_h,
      native_field_h, v2::subtitle_ocr_safe_row_bottom);
    std::vector<float> probability(ocr_pixels, 0.0f);
    // Authored main-line geometry matches the real crop diagnostic's first English core.
    // The leading glyph's actual P response is negligible; do not fabricate a text island.
    paint_rectangle(probability, 331u, 82u, 645u, 94u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, native_w, native_h,
      native_field_w, native_field_h, crop_top, crop_h, native_roi_top, native_roi_bottom));
    const auto without_weak_glyph = fixture.record();
    paint_rectangle(probability, 314u, 77u, 319u, 99u, 0.0005f);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, native_w, native_h,
      native_field_w, native_field_h, crop_top, crop_h, native_roi_top, native_roi_bottom));
    EXPECT_EQ(fixture.record(), without_weak_glyph);
    ASSERT_EQ(fixture.record()[2u], 1u);
    ASSERT_EQ(fixture.record()[3u], 1u);
    ASSERT_EQ(fixture.record()[4u], 1u);
    const auto core = read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset);
    const auto cover = read_box(fixture.record(), v2::subtitle_ocr_final_box_offset);
    EXPECT_EQ(core, (std::array<std::uint32_t, 4u> {530u, 742u, 1035u, 762u}));
    EXPECT_EQ(cover, (std::array<std::uint32_t, 4u> {503u, 728u, 1062u, 777u}));
    const auto contains_source_center = [&](const std::array<std::uint32_t, 4u> &box,
                                           const std::uint32_t x, const std::uint32_t y) {
      const auto scaled_x = (2ull * x + 1u) * native_field_w;
      const auto scaled_y = (2ull * y + 1u) * native_field_h;
      return scaled_x >= 2ull * box[0u] * native_w &&
             scaled_x < 2ull * box[2u] * native_w &&
             scaled_y >= 2ull * box[1u] * native_h &&
             scaled_y < 2ull * box[3u] * native_h;
    };
    // This is the real initial-I white-body witness rectangle, not a production color rule.
    // All 234 centers gain current search authority; temporal confirmation is tested separately.
    std::uint32_t admitted_centers = 0u;
    for (std::uint32_t y = 860u; y < 899u; ++y) {
      for (std::uint32_t x = 592u; x < 598u; ++x) {
        EXPECT_FALSE(contains_source_center(core, x, y));
        EXPECT_TRUE(contains_source_center(cover, x, y));
        admitted_centers += contains_source_center(cover, x, y) ? 1u : 0u;
      }
    }
    EXPECT_EQ(admitted_centers, 234u);
  }

  struct ocr_candidate_patch_t {
    std::array<std::uint32_t, 4u> rectangle;
    float probability = 0.9f;
  };

  using ocr_native_pair_t = std::array<std::uint32_t, 16u>;

  ocr_native_pair_t read_native_pair(const std::vector<std::uint32_t> &record,
                                    const std::uint32_t slot) {
    ocr_native_pair_t pair {};
    for (std::uint32_t word = 0u; word < v2::subtitle_ocr_box_word_count; ++word) {
      pair[word] = record.at(v2::subtitle_ocr_raw_box_offset +
                             slot * v2::subtitle_ocr_box_word_count + word);
      pair[v2::subtitle_ocr_box_word_count + word] =
        record.at(v2::subtitle_ocr_final_box_offset +
                  slot * v2::subtitle_ocr_box_word_count + word);
    }
    return pair;
  }

  // Expected pair payloads come from isolated production GPU resolves, not a CPU DB classifier.
  bool capture_native_pairs(ocr_warp_fixture_t &fixture,
                            const std::vector<ocr_candidate_patch_t> &patches,
                            std::vector<ocr_native_pair_t> &pairs) {
    pairs.clear();
    for (const auto &patch : patches) {
      std::vector<float> probability(ocr_pixels, 0.0f);
      const auto &r = patch.rectangle;
      paint_rectangle(probability, r[0u], r[1u], r[2u], r[3u], patch.probability);
      // A valid 1920-wide analysis crop can lie at different offsets in a 3840-wide screen.
      // Align that screen's center with this isolated patch without changing its OCR geometry;
      // these captures provide the original pair payloads for capacity tests, not line filtering.
      const auto analysis_left = 1920u - r[0u] - r[2u];
      if (!fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, analysis_left) || fixture.record()[2u] != 1u ||
          fixture.record()[3u] != 1u || fixture.record()[4u] != 1u) return false;
      pairs.push_back(read_native_pair(fixture.record(), 0u));
    }
    return true;
  }

  void paint_candidate_patches(std::vector<float> &probability,
                               const std::vector<ocr_candidate_patch_t> &patches) {
    for (const auto &patch : patches) {
      const auto &r = patch.rectangle;
      paint_rectangle(probability, r[0u], r[1u], r[2u], r[3u], patch.probability);
    }
  }

  void expect_retained_native_pairs(const std::vector<std::uint32_t> &record,
                                    const std::vector<ocr_native_pair_t> &pairs) {
    ASSERT_EQ(record.size(), v2::subtitle_ocr_record_word_count);
    ASSERT_LE(pairs.size(), v2::subtitle_ocr_final_box_capacity);
    EXPECT_EQ(record[0u], v2::subtitle_ocr_record_schema);
    EXPECT_EQ(record[1u], v2::subtitle_ocr_record_tag);
    EXPECT_EQ(record[2u], 1u);
    EXPECT_EQ(record[3u], pairs.size());
    EXPECT_EQ(record[4u], pairs.size());
    EXPECT_EQ(record[5u], 40u);
    EXPECT_EQ(record[6u], 0u);
    EXPECT_EQ(record[7u], 77u);
    EXPECT_EQ(record[8u], 0u);
    EXPECT_EQ(record[9u], source_width);
    EXPECT_EQ(record[10u], source_height);
    EXPECT_EQ(record[11u], field_width);
    EXPECT_EQ(record[12u], field_height);
    EXPECT_EQ(record[13u], roi_top);
    EXPECT_EQ(record[14u], roi_bottom);
    EXPECT_EQ(record[15u], 0u);
    for (std::uint32_t slot = 0u; slot < v2::subtitle_ocr_raw_box_capacity; ++slot) {
      for (std::uint32_t word = 0u; word < v2::subtitle_ocr_box_word_count; ++word) {
        EXPECT_EQ(record[v2::subtitle_ocr_raw_box_offset + slot * v2::subtitle_ocr_box_word_count + word],
                  slot < pairs.size() ? pairs[slot][word] : 0u) << "raw slot=" << slot << " word=" << word;
      }
    }
    for (std::uint32_t slot = 0u; slot < v2::subtitle_ocr_final_box_capacity; ++slot) {
      for (std::uint32_t word = 0u; word < v2::subtitle_ocr_box_word_count; ++word) {
        EXPECT_EQ(record[v2::subtitle_ocr_final_box_offset + slot * v2::subtitle_ocr_box_word_count + word],
                  slot < pairs.size() ? pairs[slot][v2::subtitle_ocr_box_word_count + word] : 0u)
          << "cover slot=" << slot << " word=" << word;
      }
    }
  }

  std::vector<ocr_candidate_patch_t> equal_area_ocr_patches(const std::uint32_t count,
                                                         const std::uint32_t top = 70u) {
    std::vector<ocr_candidate_patch_t> patches;
    for (std::uint32_t index = 0u; index < count; ++index) {
      // A 96-pixel step projects to an integer field-X step; every tight core has equal area.
      // Height 20 also keeps the narrowed first-minimum fixture's 80-pixel neighbor gap within
      // the production line-locality limit, so capacity tests cannot pass by dropping that core.
      const auto left = index * 96u;
      patches.push_back({{left, top, left + 32u, top + 20u}, 0.8f + index * 0.01f});
    }
    return patches;
  }

  TEST(HostSbsOcrGpuTest, LowerQuarterEligibilityDoesNotChangeTheNativeOcrCrop) {
    EXPECT_EQ(ceil_crop_height(1920u, 1080u), 320u);
    EXPECT_EQ(ceil_crop_height(3840u, 1080u), 640u);
    EXPECT_EQ(ceil_crop_height(1080u, 1920u), 180u);
    EXPECT_EQ(source_crop_top, 760u);
    EXPECT_EQ(subtitle_search_top(1080u), 810u);
    EXPECT_EQ(subtitle_search_top(1920u), 1440u);
    EXPECT_EQ(subtitle_search_top(1357u), 1017u);
    EXPECT_EQ(subtitle_search_top(1u), 0u);

    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 400u, 18u, 560u, 25u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    expect_retained_native_pairs(fixture.record(), {});
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 400u, 25u, 560u, 32u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    ASSERT_EQ(fixture.record()[3u], 1u);
    EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(400u, 25u, 560u, 32u));
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 400u, 20u, 560u, 28u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    ASSERT_EQ(fixture.record()[3u], 1u);
    EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(400u, 25u, 560u, 28u));

    // A very wide source keeps the same width-derived model crop, but the search starts much
    // farther down that crop. Native lower-quarter eligibility must not mean detector row 25.
    constexpr std::uint32_t wide_source_w = 3840u;
    constexpr auto wide_crop_h = ceil_crop_height(wide_source_w, source_height);
    constexpr auto wide_crop_top = source_height - wide_crop_h;
    constexpr auto wide_roi_top = subtitle_roi_edge(wide_source_w, source_height,
      field_height, v2::subtitle_ocr_safe_row_top);
    constexpr auto wide_roi_bottom = subtitle_roi_edge(wide_source_w, source_height,
      field_height, v2::subtitle_ocr_safe_row_bottom);
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 400u, 70u, 560u, 80u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, wide_source_w, source_height,
      field_width, field_height, wide_crop_top, wide_crop_h, wide_roi_top, wide_roi_bottom));
    EXPECT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 400u, 100u, 560u, 110u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, wide_source_w, source_height,
      field_width, field_height, wide_crop_top, wide_crop_h, wide_roi_top, wide_roi_bottom));
    EXPECT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[3u], 1u);
  }

  TEST(HostSbsOcrGpuTest, LineCenterHasAnInclusiveOneTenthWidthTolerance) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    for (const auto left : {336u, 528u, 335u, 529u}) {
      SCOPED_TRACE(std::to_string(left));
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, left, 85u, left + 96u, 100u);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
      EXPECT_EQ(fixture.record()[2u], 1u);
      const auto expected_count = left == 336u || left == 528u ? 1u : 0u;
      EXPECT_EQ(fixture.record()[3u], expected_count);
      EXPECT_EQ(fixture.record()[4u], expected_count);
      if (expected_count == 0u) expect_retained_native_pairs(fixture.record(), {});
    }
  }

  TEST(HostSbsOcrGpuTest, NondivisibleVideoWidthKeepsBothExactCenterBoundaries) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    constexpr std::uint32_t video_w = 2538u;
    constexpr std::uint32_t video_h = 1427u;
    const auto plan = models::plan_host_sbs_v2_video_region(
      {277u, 415u, 277u + video_w, 415u + video_h}, 3840u, 2160u,
      {static_cast<int>(field_width), static_cast<int>(field_height)});
    ASSERT_TRUE(plan.has_value());
    const auto geometry = models::fit_subtitle_analysis_geometry(video_w, video_h,
      plan->tensor_shape, plan->tensor_content);
    ASSERT_TRUE(geometry.valid());
    const std::array<std::uint32_t, 4u> content {
      plan->tensor_content.left, plan->tensor_content.top,
      plan->tensor_content.right, plan->tensor_content.bottom,
    };
    constexpr auto crop_h = ceil_crop_height(video_w, video_h);
    constexpr auto crop_top = video_h - crop_h;
    for (const auto left : {336u, 528u, 335u, 529u}) {
      SCOPED_TRACE(std::to_string(left));
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, left, 85u, left + 96u, 100u);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, video_w, video_h,
        field_width, field_height, crop_top, crop_h,
        geometry.roi_top, geometry.roi_bottom, content));
      EXPECT_EQ(fixture.record()[2u], 1u);
      const auto expected_count = left == 336u || left == 528u ? 1u : 0u;
      EXPECT_EQ(fixture.record()[3u], expected_count);
      EXPECT_EQ(fixture.record()[4u], expected_count);
      // In particular, right-edge tightL+tightR=1152 is exactly 1/10 of native video width
      // from its center. Converting both sides independently to FP32 once rejected this case.
      if (expected_count == 0u) {
        for (std::size_t word = v2::subtitle_ocr_raw_box_offset;
             word < fixture.record().size(); ++word)
          EXPECT_EQ(fixture.record()[word], 0u);
      }
    }
  }

  TEST(HostSbsOcrGpuTest, InvalidReferenceExtremesAbstainWithoutPublishingAnyPair) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    struct reference_case_t {
      std::int32_t center_twice;
      std::uint32_t width;
      std::int32_t search_top;
    };
    const std::array<reference_case_t, 3u> cases {{
      {std::numeric_limits<std::int32_t>::min(), source_width,
       static_cast<std::int32_t>(subtitle_search_top(source_height))},
      {static_cast<std::int32_t>(source_width), std::numeric_limits<std::uint32_t>::max(),
       static_cast<std::int32_t>(subtitle_search_top(source_height))},
      {static_cast<std::int32_t>(source_width), source_width,
       std::numeric_limits<std::int32_t>::min()},
    }};
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 400u, 85u, 560u, 100u);
    for (const auto &reference : cases) {
      SCOPED_TRACE(std::to_string(reference.center_twice) + "/" +
        std::to_string(reference.width) + "/" + std::to_string(reference.search_top));
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, source_width, source_height,
        field_width, field_height, source_crop_top, source_crop_height,
        roi_top, roi_bottom, {}, reference.center_twice, reference.width, reference.search_top));
      EXPECT_EQ(fixture.record()[2u], 0u);
      EXPECT_EQ(fixture.record()[3u], 0u);
      EXPECT_EQ(fixture.record()[4u], 0u);
      for (std::size_t word = v2::subtitle_ocr_raw_box_offset;
           word < fixture.record().size(); ++word)
        EXPECT_EQ(fixture.record()[word], 0u);
    }
  }

  TEST(HostSbsOcrGpuTest, SameLineFragmentsUseTheirTightUnionButKeepSeparateCovers) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    const std::vector<ocr_candidate_patch_t> patches {
      {{300u, 85u, 440u, 100u}},
      {{500u, 85u, 660u, 100u}},
    };
    std::vector<ocr_native_pair_t> pairs;
    ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
    for (const auto &patch : patches) {
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_candidate_patches(probability, {patch});
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
      expect_retained_native_pairs(fixture.record(), {});
    }
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_candidate_patches(probability, patches);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    expect_retained_native_pairs(fixture.record(), pairs);
    const auto left_cover = read_box(fixture.record(), v2::subtitle_ocr_final_box_offset);
    const auto right_cover = read_box(fixture.record(), v2::subtitle_ocr_final_box_offset +
      v2::subtitle_ocr_box_word_count);
    EXPECT_LT(left_cover[2u], right_cover[0u]);
    EXPECT_LT(left_cover[2u], field_width / 2u);
    EXPECT_GT(right_cover[0u], field_width / 2u);
  }

  TEST(HostSbsOcrGpuTest, SameLineHorizontalGapHasAnInclusiveFourHeightLimit) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    for (const auto gap : {59u, 60u, 61u}) {
      SCOPED_TRACE(std::to_string(gap));
      const std::vector<ocr_candidate_patch_t> patches {
        {{300u, 85u, 440u, 100u}},
        {{440u + gap, 85u, 600u + gap, 100u}},
      };
      std::vector<ocr_native_pair_t> pairs;
      ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_candidate_patches(probability, patches);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
      // Each fragment alone is outside 480+/-96, while its tight union is centered.
      // The gap exceeds the ordinary 32-pixel join, so this exercises distinct line members.
      expect_retained_native_pairs(fixture.record(), gap <= 60u ? pairs :
        std::vector<ocr_native_pair_t> {});
    }
  }

  TEST(HostSbsOcrGpuTest, SameLineLocalFragmentsGrowFromTheCurrentHorizontalEdge) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    const std::vector<ocr_candidate_patch_t> patches {
      {{160u, 85u, 320u, 105u}},
      {{400u, 85u, 560u, 105u}},
      {{640u, 85u, 800u, 105u}},
    };
    std::vector<ocr_native_pair_t> pairs;
    ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_candidate_patches(probability, patches);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    // Both adjacent gaps equal 4*20. The third fragment is 320 pixels from the first member,
    // so using the fixed first member's X extent would lose this genuine local chain.
    expect_retained_native_pairs(fixture.record(), pairs);
  }

  TEST(HostSbsOcrGpuTest, DistantSameHeightColumnsCannotBorrowEachOthersCenter) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 80u, 85u, 240u, 100u);
    paint_rectangle(probability, 720u, 85u, 880u, 100u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    // The centered union crosses a 480-pixel empty column, 32 times the text height. The two
    // independent off-center columns must not become an eligible centered subtitle line.
    expect_retained_native_pairs(fixture.record(), {});
  }

  TEST(HostSbsOcrGpuTest, BrowserCaptionAndSidebarCannotCreateABoundaryCenteredLine) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    constexpr std::uint32_t browser_w = 2560u;
    constexpr std::uint32_t browser_h = 1440u;
    constexpr std::uint32_t browser_field_w = 1540u;
    constexpr std::uint32_t browser_field_h = 868u;
    constexpr auto crop_h = ceil_crop_height(browser_w, browser_h);
    constexpr auto crop_top = browser_h - crop_h;
    constexpr auto field_roi_top = subtitle_roi_edge(browser_w, browser_h,
      browser_field_h, v2::subtitle_ocr_safe_row_top);
    constexpr auto field_roi_bottom = subtitle_roi_edge(browser_w, browser_h,
      browser_field_h, v2::subtitle_ocr_safe_row_bottom);
    std::vector<float> probability(ocr_pixels, 0.0f);
    // These authored rectangles map to browser150 frame1's native core geometry:
    // caption [447,691,734,708], sidebar [1317,692,1401,703]. Their vertical IoU is 0.6
    // and their union center lands exactly at the inclusive right center boundary (P x576).
    paint_rectangle(probability, 279u, 50u, 457u, 60u);
    paint_rectangle(probability, 821u, 51u, 873u, 57u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, browser_w, browser_h,
      browser_field_w, browser_field_h, crop_top, crop_h,
      field_roi_top, field_roi_bottom));
    EXPECT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);
    for (std::size_t word = v2::subtitle_ocr_raw_box_offset;
         word < fixture.record().size(); ++word)
      EXPECT_EQ(fixture.record()[word], 0u);
  }

  TEST(HostSbsOcrGpuTest, SameLineGapUsesRoundedNativeCropAndMaximumSafeTextHeight) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    constexpr std::uint32_t video_w = 1000u;
    constexpr std::uint32_t video_h = 1600u;
    const auto plan = models::plan_host_sbs_v2_video_region(
      {1100u, 200u, 1100u + video_w, 200u + video_h}, 3840u, 2160u,
      {static_cast<int>(field_width), static_cast<int>(field_height)});
    ASSERT_TRUE(plan.has_value());
    const auto geometry = models::fit_subtitle_analysis_geometry(video_w, video_h,
      plan->tensor_shape, plan->tensor_content);
    ASSERT_TRUE(geometry.valid());
    const std::array<std::uint32_t, 4u> content {
      plan->tensor_content.left, plan->tensor_content.top,
      plan->tensor_content.right, plan->tensor_content.bottom,
    };
    constexpr auto crop_h = ceil_crop_height(video_w, video_h);
    constexpr auto crop_top = video_h - crop_h;
    static_assert(crop_h == 167u);
    static_assert(v2::subtitle_ocr_safe_row_bottom - v2::subtitle_ocr_safe_row_top == 131u);
    for (const auto gap : {525u, 526u}) {
      SCOPED_TRACE(std::to_string(gap));
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, 100u, v2::subtitle_ocr_safe_row_top,
        200u, v2::subtitle_ocr_safe_row_bottom);
      paint_rectangle(probability, 200u + gap, v2::subtitle_ocr_safe_row_top,
        335u + gap, v2::subtitle_ocr_safe_row_bottom);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, video_w, video_h,
        field_width, field_height, crop_top, crop_h,
        geometry.roi_top, geometry.roi_bottom, content));
      EXPECT_EQ(fixture.record()[2u], 1u);
      // Native lengths yield 525.048 detector pixels. Comparing detector coordinates directly
      // would incorrectly reject 525 because 4*131=524; 526 is outside even in native pixels.
      EXPECT_EQ(fixture.record()[3u], gap == 525u ? 2u : 0u);
      EXPECT_EQ(fixture.record()[4u], gap == 525u ? 2u : 0u);
      if (gap == 526u) {
        for (std::size_t word = v2::subtitle_ocr_raw_box_offset;
             word < fixture.record().size(); ++word)
          EXPECT_EQ(fixture.record()[word], 0u);
      }
    }
  }

  TEST(HostSbsOcrGpuTest, SameLineGapUsesClippedNativeCropAtMaximumSourceWidth) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    constexpr std::uint32_t video_w = 5120u;
    constexpr std::uint32_t video_h = 640u;
    const auto plan = models::plan_host_sbs_v2_video_region(
      {0u, 100u, video_w, 100u + video_h}, video_w, 1080u,
      {static_cast<int>(field_width), static_cast<int>(field_height)});
    ASSERT_TRUE(plan.has_value());
    const auto geometry = models::fit_subtitle_analysis_geometry(video_w, video_h,
      plan->tensor_shape, plan->tensor_content);
    ASSERT_TRUE(geometry.valid());
    const std::array<std::uint32_t, 4u> content {
      plan->tensor_content.left, plan->tensor_content.top,
      plan->tensor_content.right, plan->tensor_content.bottom,
    };
    constexpr auto crop_h = ceil_crop_height(video_w, video_h);
    constexpr auto crop_top = video_h - crop_h;
    static_assert(crop_h == video_h && crop_top == 0u);
    for (const auto gap : {60u, 61u}) {
      SCOPED_TRACE(std::to_string(gap));
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, 300u, 130u, 440u, 150u);
      paint_rectangle(probability, 440u + gap, 130u, 600u + gap, 150u);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, video_w, video_h,
        field_width, field_height, crop_top, crop_h,
        geometry.roi_top, geometry.roi_bottom, content));
      EXPECT_EQ(fixture.record()[2u], 1u);
      // The 8:1 source clips the requested 6:1 crop. Its true-native four-height limit is 60
      // detector pixels, whereas an uncorrected detector-space check would allow 80.
      EXPECT_EQ(fixture.record()[3u], gap == 60u ? 2u : 0u);
      EXPECT_EQ(fixture.record()[4u], gap == 60u ? 2u : 0u);
      if (gap == 61u) {
        for (std::size_t word = v2::subtitle_ocr_raw_box_offset;
             word < fixture.record().size(); ++word)
          EXPECT_EQ(fixture.record()[word], 0u);
      }
    }
  }

  TEST(HostSbsOcrGpuTest, DistinctRowsCannotBorrowEachOthersCenter) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 300u, 85u, 440u, 100u);
    paint_rectangle(probability, 500u, 110u, 660u, 125u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    expect_retained_native_pairs(fixture.record(), {});

    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 240u, 85u, 720u, 100u);
    paint_rectangle(probability, 720u, 110u, 880u, 125u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    ASSERT_EQ(fixture.record()[3u], 1u);
    EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(240u, 85u, 720u, 100u));
  }

  TEST(HostSbsOcrGpuTest, SameLineVerticalOverlapIsMeasuredAgainstAFixedAnchor) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    for (const auto right_top : {86u, 87u}) {
      SCOPED_TRACE(std::to_string(right_top));
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, 300u, 80u, 440u, 100u);
      paint_rectangle(probability, 500u, right_top, 660u, right_top + 20u);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
      EXPECT_EQ(fixture.record()[2u], 1u);
      EXPECT_EQ(fixture.record()[3u], right_top == 86u ? 2u : 0u);
      EXPECT_EQ(fixture.record()[4u], right_top == 86u ? 2u : 0u);
    }

    // The middle fragment overlaps each neighbor sufficiently, but it cannot expand the first
    // group's 80..100 anchor and thereby let a 92..112 fragment borrow that group's center.
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 180u, 80u, 300u, 100u);
    paint_rectangle(probability, 360u, 86u, 480u, 106u);
    paint_rectangle(probability, 540u, 92u, 660u, 112u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    expect_retained_native_pairs(fixture.record(), {});
  }

  TEST(HostSbsOcrGpuTest, TallTextureDoesNotBridgeSmallClothingAndLogoCandidates) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 0u, 70u, 320u, 140u);
    paint_rectangle(probability, 580u, 103u, 600u, 108u);
    paint_rectangle(probability, 896u, 105u, 928u, 150u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    expect_retained_native_pairs(fixture.record(), {});
  }

  TEST(HostSbsOcrGpuTest, LineCenterUsesAllCandidatesBeforeTheEightPairCapacity) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    const auto patches = equal_area_ocr_patches(9u);
    std::vector<ocr_native_pair_t> pairs;
    ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
    pairs.resize(8u);
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_candidate_patches(probability, patches);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    // All nine cores span 0..800 (center 400 within 480+/-96). The earlier retained eight span
    // only 0..704 (center 352 outside the tolerance), so selection cannot precede line filtering.
    expect_retained_native_pairs(fixture.record(), pairs);
  }

  TEST(HostSbsOcrGpuTest, ScreenFallbackUsesTheScreenCenterAndAnalysisOffset) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> screen_centered(ocr_pixels, 0.0f);
    paint_rectangle(screen_centered, 80u, 85u, 240u, 100u);
    ASSERT_TRUE(fixture.run_boxes(screen_centered, 40u, 77u));
    expect_retained_native_pairs(fixture.record(), {});
    // A 1920-wide foreground-client analysis at x1600 in a 3840-wide screen sees the true
    // screen center at local native x 320, or detector x 160, rather than its own local center.
    ASSERT_TRUE(fixture.run_boxes_in_screen(screen_centered, 40u, 77u, 3840u, 1600u));
    ASSERT_EQ(fixture.record()[3u], 1u);
    EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(80u, 85u, 240u, 100u));

    std::vector<float> roi_centered(ocr_pixels, 0.0f);
    paint_rectangle(roi_centered, 400u, 85u, 560u, 100u);
    ASSERT_TRUE(fixture.run_boxes(roi_centered, 40u, 77u));
    ASSERT_EQ(fixture.record()[3u], 1u);
    ASSERT_TRUE(fixture.run_boxes_in_screen(roi_centered, 40u, 77u, 3840u, 1600u));
    expect_retained_native_pairs(fixture.record(), {});
  }

  TEST(HostSbsOcrGpuTest, LineCenterIsIndependentOfTensorPaddingAndValidatesReferenceWidth) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    constexpr std::uint32_t video_w = 1000u;
    constexpr std::uint32_t video_h = 1600u;
    const auto plan = models::plan_host_sbs_v2_video_region(
      {1100u, 200u, 1100u + video_w, 200u + video_h}, 3840u, 2160u,
      {static_cast<int>(field_width), static_cast<int>(field_height)});
    ASSERT_TRUE(plan.has_value());
    const auto geometry = models::fit_subtitle_analysis_geometry(video_w, video_h,
      plan->tensor_shape, plan->tensor_content);
    ASSERT_TRUE(geometry.valid());
    const std::array<std::uint32_t, 4u> content {
      plan->tensor_content.left, plan->tensor_content.top,
      plan->tensor_content.right, plan->tensor_content.bottom,
    };
    constexpr auto crop_h = ceil_crop_height(video_w, video_h);
    constexpr auto crop_top = video_h - crop_h;
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 400u, 85u, 560u, 100u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, video_w, video_h,
      field_width, field_height, crop_top, crop_h,
      geometry.roi_top, geometry.roi_bottom, content));
    EXPECT_EQ(fixture.record()[2u], 1u);
    ASSERT_EQ(fixture.record()[3u], 1u);
    const auto box = read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset);
    EXPECT_EQ(box[0u], 361u);
    EXPECT_EQ(box[2u], 408u);

    // This native off-center line maps near the global field center because the production fit
    // has large horizontal padding. A field-space +/-10% gate would accept it incorrectly.
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 200u, 85u, 360u, 100u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, video_w, video_h,
      field_width, field_height, crop_top, crop_h,
      geometry.roi_top, geometry.roi_bottom, content));
    EXPECT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);

    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, source_width, source_height,
      field_width, field_height, source_crop_top, source_crop_height,
      roi_top, roi_bottom, {}, static_cast<std::int32_t>(source_width), 0u));
    EXPECT_EQ(fixture.record()[2u], 0u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);
  }

  TEST(HostSbsOcrGpuTest, ScreenReferencesRemainSignedOutsideTheAnalysisCrop) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    constexpr std::uint32_t cropped_w = 1500u;
    constexpr std::uint32_t cropped_h = 400u;
    constexpr auto crop_h = ceil_crop_height(cropped_w, cropped_h);
    constexpr auto crop_top = cropped_h - crop_h;
    constexpr std::array<std::uint32_t, 4u> content {0u, 114u, 770u, 319u};
    constexpr auto cropped_roi_top = subtitle_roi_edge(cropped_w, cropped_h,
      content[1u], content[3u] - content[1u], v2::subtitle_ocr_safe_row_top);
    constexpr auto cropped_roi_bottom = subtitle_roi_edge(cropped_w, cropped_h,
      content[1u], content[3u] - content[1u], v2::subtitle_ocr_safe_row_bottom);
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 400u, 70u, 560u, 80u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, cropped_w, cropped_h,
      field_width, field_height, crop_top, crop_h, cropped_roi_top, cropped_roi_bottom,
      content));
    EXPECT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    // A foreground client at (210,1700), size 1500x400, fits a 1920x2160 screen. The screen's
    // bottom-quarter start is 80 pixels above this client, so signed searchTop=-80 is valid.
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, cropped_w, cropped_h,
      field_width, field_height, crop_top, crop_h, cropped_roi_top, cropped_roi_bottom,
      content, 1500, 1920u, -80));
    EXPECT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[3u], 1u);

    constexpr std::uint32_t right_w = 1000u;
    constexpr std::uint32_t right_h = 600u;
    constexpr auto right_crop_h = ceil_crop_height(right_w, right_h);
    constexpr auto right_crop_top = right_h - right_crop_h;
    constexpr auto right_roi_top = subtitle_roi_edge(right_w, right_h,
      field_height, v2::subtitle_ocr_safe_row_top);
    constexpr auto right_roi_bottom = subtitle_roi_edge(right_w, right_h,
      field_height, v2::subtitle_ocr_safe_row_bottom);
    // A crop at (2400,1500), size 1000x600, fits a 3840x2160 screen but lies entirely right
    // of its center. Screen center2=-960 must not wrap into an unsigned positive center.
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, right_w, right_h,
      field_width, field_height, right_crop_top, right_crop_h, right_roi_top,
      right_roi_bottom, {}, -960, 3840u, 120));
    EXPECT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);
  }

  TEST(HostSbsOcrGpuTest, BoundedCandidateEmptyPublishesAuthoritativeZeroTails) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    ASSERT_TRUE(fixture.run_boxes(std::vector<float>(ocr_pixels, 0.0f), 40u, 77u));
    expect_retained_native_pairs(fixture.record(), {});
  }

  TEST(HostSbsOcrGpuTest, BoundedCandidateEightPreserveEveryNativePairAndDiscoveryOrder) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    const auto patches = equal_area_ocr_patches(8u);
    std::vector<ocr_native_pair_t> pairs;
    ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_candidate_patches(probability, patches);
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, 1216u));
    expect_retained_native_pairs(fixture.record(), pairs);
  }

  TEST(HostSbsOcrGpuTest, BoundedCandidateNinthEqualAreaRetainsEarlierDiscovery) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    const auto patches = equal_area_ocr_patches(9u);
    std::vector<ocr_native_pair_t> pairs;
    ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
    const auto area = [](const ocr_native_pair_t &pair) {
      return (pair[2u] - pair[0u]) * (pair[3u] - pair[1u]);
    };
    for (const auto &pair : pairs) ASSERT_EQ(area(pair), area(pairs.front()));
    pairs.resize(8u);
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_candidate_patches(probability, patches);
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, 1216u));
    expect_retained_native_pairs(fixture.record(), pairs);
  }

  TEST(HostSbsOcrGpuTest, BoundedCandidateLateLargerAreaEvictsLastEqualMinimum) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    auto patches = equal_area_ocr_patches(8u);
    // Lower probability still wins by tight-core area alone, without confidence retuning.
    patches.push_back({{768u, 70u, 896u, 80u}, 0.6f});
    std::vector<ocr_native_pair_t> pairs;
    ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
    pairs.erase(pairs.begin() + 7u);
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_candidate_patches(probability, patches);
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, 1216u));
    expect_retained_native_pairs(fixture.record(), pairs);
  }

  TEST(HostSbsOcrGpuTest, BoundedCandidateLateLargerAreaShiftsIntactPairsFromFirstMinimum) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    auto patches = equal_area_ocr_patches(8u);
    patches.front().rectangle[2u] = 16u;
    patches.push_back({{768u, 70u, 896u, 80u}, 0.6f});
    std::vector<ocr_native_pair_t> pairs;
    ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
    pairs.erase(pairs.begin());
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_candidate_patches(probability, patches);
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, 1216u));
    expect_retained_native_pairs(fixture.record(), pairs);
  }

  TEST(HostSbsOcrGpuTest, BoundedCandidateTwentyContinueScanningAndKeepLargestEight) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    auto patches = equal_area_ocr_patches(10u, 50u);
    auto later = equal_area_ocr_patches(10u, 100u);
    for (auto &patch : later) patch.rectangle[2u] = patch.rectangle[0u] + 48u;
    patches.insert(patches.end(), later.begin(), later.end());
    std::vector<ocr_native_pair_t> pairs;
    ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
    for (std::size_t index = 10u; index < pairs.size(); ++index) {
      const auto &early = pairs.front();
      const auto &late = pairs[index];
      ASSERT_GT((late[2u] - late[0u]) * (late[3u] - late[1u]),
                (early[2u] - early[0u]) * (early[3u] - early[1u]));
      ASSERT_EQ((late[2u] - late[0u]) * (late[3u] - late[1u]),
                (pairs[10u][2u] - pairs[10u][0u]) * (pairs[10u][3u] - pairs[10u][1u]));
    }
    const std::vector<ocr_native_pair_t> expected(pairs.begin() + 10u, pairs.begin() + 18u);
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_candidate_patches(probability, patches);
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, 1216u));
    expect_retained_native_pairs(fixture.record(), expected);
  }

  TEST(HostSbsOcrGpuTest, BoundedCandidateRibbonAndOrdinaryCoresKeepTheirPairedCovers) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    const auto patches = equal_area_ocr_patches(8u);
    std::vector<ocr_native_pair_t> pairs;
    ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
    constexpr std::array<std::array<std::uint32_t, 2u>, 7u> islands {{
      {{8u, 72u}}, {{86u, 155u}}, {{174u, 220u}}, {{267u, 455u}},
      {{520u, 558u}}, {{594u, 636u}}, {{742u, 863u}},
    }};
    std::vector<float> probability(ocr_pixels, 0.0f);
    for (const auto &island : islands)
      paint_rectangle(probability, island[0u], 120u, island[1u], 156u);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    ASSERT_EQ(fixture.record()[3u], 1u);
    const auto ribbon = read_native_pair(fixture.record(), 0u);
    ASSERT_EQ(ribbon[5u], v2::subtitle_ocr_box_flag_ribbon);
    ASSERT_EQ(ribbon[v2::subtitle_ocr_box_word_count + 5u], v2::subtitle_ocr_box_flag_ribbon);
    ASSERT_EQ(ribbon[v2::subtitle_ocr_box_word_count + 0u], 0u);
    ASSERT_EQ(ribbon[v2::subtitle_ocr_box_word_count + 2u], field_width);
    ASSERT_EQ(ribbon[v2::subtitle_ocr_box_word_count + 3u], field_height);
    pairs.pop_back();
    pairs.push_back(ribbon);
    paint_candidate_patches(probability, patches);
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, 1216u));
    expect_retained_native_pairs(fixture.record(), pairs);
  }

  TEST(HostSbsOcrGpuTest, BoundedCandidateCrowdingStillRejectsNonfiniteAndInvalidGeometry) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_candidate_patches(probability, equal_area_ocr_patches(9u));
    for (const auto value : {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}) {
      probability[0u] = value;
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
      EXPECT_EQ(fixture.record()[2u], 0u);
      EXPECT_EQ(fixture.record()[3u], 0u);
      EXPECT_EQ(fixture.record()[4u], 0u);
      for (std::size_t word = v2::subtitle_ocr_raw_box_offset; word < fixture.record().size(); ++word)
        EXPECT_EQ(fixture.record()[word], 0u);
    }
    probability[0u] = 0.0f;
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, source_width, source_height,
                                  field_width, field_height, source_crop_top + 1u));
    EXPECT_EQ(fixture.record()[2u], 0u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);
    for (std::size_t word = v2::subtitle_ocr_raw_box_offset; word < fixture.record().size(); ++word)
      EXPECT_EQ(fixture.record()[word], 0u);
  }

  TEST(HostSbsOcrGpuTest, OrdinaryTextUsesNarrowJoinAndIndependentIslandEvidence) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;

    std::vector<float> probability(ocr_pixels, 0.0f);
    // This reproduces the legitimate two-island subtitle topology: one inactive 8-pixel cell
    // separates two independently strong glyph groups, so the ordinary text join retains both.
    paint_rectangle(probability, 290u, 84u, 499u, 98u);
    paint_rectangle(probability, 518u, 84u, 671u, 98u);
    ASSERT_TRUE(fixture.run_boxes(probability, 10u, 77u));
    const auto &joined = fixture.record();
    ASSERT_EQ(joined[2u], 1u);
    ASSERT_EQ(joined[3u], 1u);
    ASSERT_EQ(joined[4u], 1u);
    EXPECT_EQ(
      read_box(joined, v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(290u, 84u, 671u, 98u)
    );
    EXPECT_EQ(
      read_box(joined, v2::subtitle_ocr_final_box_offset),
      default_ordinary_cover(280u, 74u, 681u, 108u)
    );
    EXPECT_EQ(joined[v2::subtitle_ocr_raw_box_offset + 6u], 2u);
    EXPECT_EQ(joined[v2::subtitle_ocr_raw_box_offset + 7u], 0u);

    // The bad live topology was a strong subtitle followed by exactly twelve inactive cells and
    // one remote active model pixel on a watch. Twelve cells remain eligible only for broad ribbon
    // classification. The non-ribbon rescan emits the subtitle alone; the 1x1 speck cannot borrow
    // its score or enlarge either tight/core or cover geometry.
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 320u, 85u, 635u, 100u);
    paint_rectangle(probability, 743u, 93u, 744u, 94u);
    ASSERT_TRUE(fixture.run_boxes(probability, 11u, 77u));
    const auto &remote_speck = fixture.record();
    ASSERT_EQ(remote_speck[2u], 1u);
    ASSERT_EQ(remote_speck[3u], 1u);
    ASSERT_EQ(remote_speck[4u], 1u);
    EXPECT_EQ(
      read_box(remote_speck, v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(320u, 85u, 635u, 100u)
    );
    EXPECT_EQ(
      read_box(remote_speck, v2::subtitle_ocr_final_box_offset),
      default_ordinary_cover(309u, 74u, 646u, 111u)
    );
    EXPECT_EQ(remote_speck[v2::subtitle_ocr_raw_box_offset + 6u], 1u);
    EXPECT_EQ(remote_speck[v2::subtitle_ocr_raw_box_offset + 7u], 0u);

    // The independent island gate also applies inside the four-cell ordinary join. A nearby 1x1
    // response is ignored, while a high-confidence 3x3 punctuation island at the same location is
    // retained. This is the exact minimum already required of every complete OCR candidate.
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 320u, 85u, 635u, 100u);
    paint_rectangle(probability, 674u, 92u, 675u, 93u);
    ASSERT_TRUE(fixture.run_boxes(probability, 12u, 77u));
    EXPECT_EQ(
      read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(320u, 85u, 635u, 100u)
    );
    EXPECT_EQ(fixture.record()[v2::subtitle_ocr_raw_box_offset + 6u], 1u);

    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 320u, 85u, 635u, 100u);
    paint_rectangle(probability, 672u, 90u, 675u, 93u);
    ASSERT_TRUE(fixture.run_boxes(probability, 13u, 77u));
    const auto &punctuation = fixture.record();
    ASSERT_EQ(punctuation[3u], 1u);
    ASSERT_EQ(punctuation[4u], 1u);
    EXPECT_EQ(
      read_box(punctuation, v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(320u, 85u, 675u, 100u)
    );
    EXPECT_EQ(
      read_box(punctuation, v2::subtitle_ocr_final_box_offset),
      default_ordinary_cover(309u, 74u, 686u, 111u)
    );
    EXPECT_EQ(punctuation[v2::subtitle_ocr_raw_box_offset + 6u], 2u);
    EXPECT_EQ(punctuation[v2::subtitle_ocr_raw_box_offset + 7u], 1u);

    // An invalid micro-island cannot bridge two valid text islands. Each raw gap is four cells, so
    // the unfiltered topology initially forms one run; after the 1x1 middle island is rejected, the
    // effective nine-cell retained gap must publish two independent OCR8 pairs. Their height18
    // keeps that72-pixel retained gap within line eligibility without weakening island evidence.
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 280u, 85u, 440u, 103u);
    paint_rectangle(probability, 479u, 92u, 480u, 93u);
    paint_rectangle(probability, 512u, 85u, 672u, 103u);
    ASSERT_TRUE(fixture.run_boxes(probability, 14u, 77u));
    const auto &micro_bridge = fixture.record();
    ASSERT_EQ(micro_bridge[2u], 1u);
    ASSERT_EQ(micro_bridge[3u], 2u);
    ASSERT_EQ(micro_bridge[4u], 2u);
    EXPECT_EQ(
      read_box(micro_bridge, v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(280u, 85u, 440u, 103u)
    );
    EXPECT_EQ(
      read_box(
        micro_bridge,
        v2::subtitle_ocr_raw_box_offset + v2::subtitle_ocr_box_word_count),
      default_ocr_box(512u, 85u, 672u, 103u)
    );
    EXPECT_EQ(micro_bridge[v2::subtitle_ocr_raw_box_offset + 6u], 1u);
    EXPECT_EQ(
      micro_bridge[
        v2::subtitle_ocr_raw_box_offset + v2::subtitle_ocr_box_word_count + 6u],
      1u
    );
  }

  TEST(HostSbsOcrGpuTest, BottomRibbonTopologyIsPublishedWithPairedCappedCover) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;

    std::vector<float> probability(ocr_pixels, 0.0f);
    // Seven bottom-index islands. Four internal gaps are at least three 8-pixel cells, while
    // every gap stays within the 12-cell line-join limit used by production resolve.
    constexpr std::array<std::array<std::uint32_t, 2u>, 7u> islands {{
      {{8u, 72u}},
      {{86u, 155u}},
      {{174u, 220u}},
      {{267u, 455u}},
      {{520u, 558u}},
      {{594u, 636u}},
      {{742u, 863u}},
    }};
    for (const auto &island : islands) {
      paint_rectangle(probability, island[0u], 120u, island[1u], 156u);
    }

    ASSERT_TRUE(fixture.run_boxes(probability, 30u, 77u));
    const auto &record = fixture.record();
    ASSERT_EQ(record[2u], 1u);
    ASSERT_EQ(record[3u], 1u);
    ASSERT_EQ(record[4u], 1u);
    EXPECT_EQ(
      read_box(record, v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(8u, 120u, 863u, v2::subtitle_ocr_safe_row_bottom)
    );
    EXPECT_EQ(
      read_box(record, v2::subtitle_ocr_final_box_offset),
      (std::array<std::uint32_t, 4u> {0u, default_ocr_box(0u, 112u, 960u, 160u)[1u], field_width, field_height})
    );
    EXPECT_EQ(
      record[v2::subtitle_ocr_raw_box_offset + 5u],
      v2::subtitle_ocr_box_flag_ribbon
    );
    EXPECT_EQ(
      record[v2::subtitle_ocr_raw_box_offset + 6u],
      static_cast<std::uint32_t>(islands.size())
    );
    EXPECT_EQ(record[v2::subtitle_ocr_raw_box_offset + 7u], 4u);
    EXPECT_EQ(
      record[v2::subtitle_ocr_final_box_offset + 5u],
      v2::subtitle_ocr_box_flag_ribbon
    );
    EXPECT_EQ(
      record[v2::subtitle_ocr_final_box_offset + 6u],
      static_cast<std::uint32_t>(islands.size())
    );
    EXPECT_EQ(record[v2::subtitle_ocr_final_box_offset + 7u], 4u);

    // The exact classifier boundary is SAFE_ROW_BOTTOM(155) - tolerance(2). An otherwise
    // identical segmented/wide run ending at detector row 153 is a ribbon; row 152 is ordinary.
    std::fill(probability.begin(), probability.end(), 0.0f);
    for (const auto &island : islands) {
      paint_rectangle(probability, island[0u], 120u, island[1u], 153u);
    }
    ASSERT_TRUE(fixture.run_boxes(probability, 31u, 77u));
    EXPECT_EQ(
      fixture.record()[v2::subtitle_ocr_raw_box_offset + 5u],
      v2::subtitle_ocr_box_flag_ribbon
    );
    std::fill(probability.begin(), probability.end(), 0.0f);
    for (const auto &island : islands) {
      paint_rectangle(probability, island[0u], 120u, island[1u], 152u);
    }
    ASSERT_TRUE(fixture.run_boxes(probability, 32u, 77u));
    EXPECT_EQ(fixture.record()[v2::subtitle_ocr_raw_box_offset + 5u], 0u);

    // The same segmented topology away from the bottom is a normal subtitle candidate. Position
    // is part of the ribbon conjunction; segmentation alone must not change subtitle semantics.
    std::fill(probability.begin(), probability.end(), 0.0f);
    for (const auto &island : islands) {
      paint_rectangle(probability, island[0u], 90u, island[1u], 100u);
    }
    ASSERT_TRUE(fixture.run_boxes(probability, 33u, 77u));
    EXPECT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[v2::subtitle_ocr_raw_box_offset + 5u], 0u);
    EXPECT_EQ(fixture.record()[v2::subtitle_ocr_final_box_offset + 5u], 0u);
  }

  TEST(HostSbsOcrGpuTest, Ocr8PublishesEveryCalibratedFieldAndRejectsWrongGeometry) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;

    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 240u, 80u, 720u, 90u);
    std::uint64_t frame = 100u;
    for (const auto &shape : v2::model_calibrated_shapes) {
      SCOPED_TRACE(
        std::to_string(shape.width) + "x" + std::to_string(shape.height)
      );
      // Use representative same-aspect source dimensions so the source->field mapping remains
      // the production fit rather than treating the tensor itself as a captured display.
      std::uint32_t source_w = shape.width;
      std::uint32_t source_h = shape.height;
      if (shape.width == 770u && shape.height == 434u) {
        source_w = 1920u;
        source_h = 1080u;
      } else if (shape.width == 1022u && shape.height == 434u) {
        source_w = 2560u;
        source_h = 1080u;
      } else if (shape.width == 1036u && shape.height == 434u) {
        source_w = 3840u;
        source_h = 1600u;
      } else if (shape.width == 434u && shape.height == 770u) {
        source_w = 1080u;
        source_h = 1920u;
      } else if (shape.width == 434u && shape.height == 1022u) {
        source_w = 1080u;
        source_h = 2560u;
      } else if (shape.width == 434u && shape.height == 1036u) {
        source_w = 1600u;
        source_h = 3840u;
      }
      const auto crop_height = ceil_crop_height(source_w, source_h);
      const auto crop_top = source_h - crop_height;
      const auto field_roi_top = subtitle_roi_edge(
        source_w,
        source_h,
        shape.height,
        v2::subtitle_ocr_safe_row_top
      );
      const auto field_roi_bottom = subtitle_roi_edge(
        source_w,
        source_h,
        shape.height,
        v2::subtitle_ocr_safe_row_bottom
      );
      ASSERT_TRUE(fixture.run_boxes(
        probability,
        frame++,
        91u,
        source_w,
        source_h,
        shape.width,
        shape.height,
        crop_top,
        crop_height,
        field_roi_top,
        field_roi_bottom
      ));
      const auto &record = fixture.record();
      ASSERT_EQ(record[2u], 1u);
      EXPECT_EQ(record[11u], shape.width);
      EXPECT_EQ(record[12u], shape.height);
      EXPECT_EQ(record[13u], field_roi_top);
      EXPECT_EQ(record[14u], field_roi_bottom);
      const auto box = read_box(record, v2::subtitle_ocr_final_box_offset);
      EXPECT_GE(box[1u], field_roi_top);
      EXPECT_LE(box[3u], field_roi_bottom);
    }

    // These custom source dimensions exercise exact rational rows where the former FP32 mapping
    // rounded across an integer boundary. The field remains one of the calibrated DAV2 shapes.
    constexpr std::uint32_t custom_landscape_w = 2416u;
    constexpr std::uint32_t custom_landscape_h = 1357u;
    constexpr std::uint32_t custom_landscape_field_w = 770u;
    constexpr std::uint32_t custom_landscape_field_h = 434u;
    constexpr auto custom_landscape_crop_h =
      ceil_crop_height(custom_landscape_w, custom_landscape_h);
    constexpr auto custom_landscape_crop_top =
      custom_landscape_h - custom_landscape_crop_h;
    constexpr auto custom_landscape_roi_top = subtitle_roi_edge(
      custom_landscape_w,
      custom_landscape_h,
      custom_landscape_field_h,
      v2::subtitle_ocr_safe_row_top
    );
    constexpr auto custom_landscape_roi_bottom = subtitle_roi_edge(
      custom_landscape_w,
      custom_landscape_h,
      custom_landscape_field_h,
      v2::subtitle_ocr_safe_row_bottom
    );
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 240u, 116u, 720u, 124u);
    ASSERT_TRUE(fixture.run_boxes(
      probability,
      frame++,
      91u,
      custom_landscape_w,
      custom_landscape_h,
      custom_landscape_field_w,
      custom_landscape_field_h,
      custom_landscape_crop_top,
      custom_landscape_crop_h,
      custom_landscape_roi_top,
      custom_landscape_roi_bottom
    ));
    ASSERT_EQ(fixture.record()[2u], 1u);
    const auto custom_landscape_box =
      read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset);
    EXPECT_EQ(
      custom_landscape_box[1u],
      map_ocr_y_floor(
        custom_landscape_h,
        custom_landscape_field_h,
        custom_landscape_crop_top,
        custom_landscape_crop_h,
        116u
      )
    );
    EXPECT_EQ(
      custom_landscape_box[3u],
      map_ocr_y_ceil(
        custom_landscape_h,
        custom_landscape_field_h,
        custom_landscape_crop_top,
        custom_landscape_crop_h,
        124u
      )
    );
    EXPECT_EQ(custom_landscape_box[3u], 406u);

    constexpr std::uint32_t custom_portrait_w = 1862u;
    constexpr std::uint32_t custom_portrait_h = 3274u;
    constexpr std::uint32_t custom_portrait_field_w = 434u;
    constexpr std::uint32_t custom_portrait_field_h = 770u;
    constexpr auto custom_portrait_crop_h =
      ceil_crop_height(custom_portrait_w, custom_portrait_h);
    constexpr auto custom_portrait_crop_top = custom_portrait_h - custom_portrait_crop_h;
    constexpr auto custom_portrait_roi_top = subtitle_roi_edge(
      custom_portrait_w,
      custom_portrait_h,
      custom_portrait_field_h,
      v2::subtitle_ocr_safe_row_top
    );
    constexpr auto custom_portrait_roi_bottom = subtitle_roi_edge(
      custom_portrait_w,
      custom_portrait_h,
      custom_portrait_field_h,
      v2::subtitle_ocr_safe_row_bottom
    );
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 240u, 90u, 720u, 98u);
    ASSERT_TRUE(fixture.run_boxes(
      probability,
      frame++,
      91u,
      custom_portrait_w,
      custom_portrait_h,
      custom_portrait_field_w,
      custom_portrait_field_h,
      custom_portrait_crop_top,
      custom_portrait_crop_h,
      custom_portrait_roi_top,
      custom_portrait_roi_bottom
    ));
    ASSERT_EQ(fixture.record()[2u], 1u);
    const auto custom_portrait_box =
      read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset);
    EXPECT_EQ(
      custom_portrait_box[1u],
      map_ocr_y_floor(
        custom_portrait_h,
        custom_portrait_field_h,
        custom_portrait_crop_top,
        custom_portrait_crop_h,
        90u
      )
    );
    EXPECT_EQ(custom_portrait_box[1u], 737u);

    // An arbitrary-aspect active window keeps the calibrated DAV2 tensor shape and fits the
    // source into an exact integer content rectangle. OCR geometry must map into that rectangle,
    // never into the edge-replicated tensor padding above or below it.
    constexpr std::uint32_t padded_source_w = 1500u;
    constexpr std::uint32_t padded_source_h = 500u;
    constexpr std::array<std::uint32_t, 4u> padded_content {
      0u, 89u, field_width, 345u
    };
    constexpr auto padded_crop_h =
      ceil_crop_height(padded_source_w, padded_source_h);
    constexpr auto padded_crop_top = padded_source_h - padded_crop_h;
    constexpr auto padded_roi_top = subtitle_roi_edge(
      padded_source_w,
      padded_source_h,
      padded_content[1u],
      padded_content[3u] - padded_content[1u],
      v2::subtitle_ocr_safe_row_top
    );
    constexpr auto padded_roi_bottom = subtitle_roi_edge(
      padded_source_w,
      padded_source_h,
      padded_content[1u],
      padded_content[3u] - padded_content[1u],
      v2::subtitle_ocr_safe_row_bottom
    );
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 240u, 100u, 720u, 108u);
    ASSERT_TRUE(fixture.run_boxes(
      probability,
      frame++,
      91u,
      padded_source_w,
      padded_source_h,
      field_width,
      field_height,
      padded_crop_top,
      padded_crop_h,
      padded_roi_top,
      padded_roi_bottom,
      padded_content
    ));
    ASSERT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[13u], padded_roi_top);
    EXPECT_EQ(fixture.record()[14u], padded_roi_bottom);
    EXPECT_EQ(
      read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
      (std::array<std::uint32_t, 4u> {
        192u,
        padded_content[1u] + map_ocr_y_floor(padded_source_h,
          padded_content[3u] - padded_content[1u], padded_crop_top, padded_crop_h, 100u),
        578u,
        padded_content[1u] + map_ocr_y_ceil(padded_source_h,
          padded_content[3u] - padded_content[1u], padded_crop_top, padded_crop_h, 108u),
      })
    );
    const auto padded_cover =
      read_box(fixture.record(), v2::subtitle_ocr_final_box_offset);
    EXPECT_GE(padded_cover[0u], padded_content[0u]);
    EXPECT_GE(padded_cover[1u], padded_content[1u]);
    EXPECT_LE(padded_cover[2u], padded_content[2u]);
    EXPECT_LE(padded_cover[3u], padded_content[3u]);

    const auto wrong_roi_top = roi_top + 1u;
    ASSERT_TRUE(fixture.run_boxes(
      probability,
      frame++,
      91u,
      source_width,
      source_height,
      field_width,
      field_height,
      source_crop_top,
      source_crop_height,
      wrong_roi_top,
      roi_bottom
    ));
    EXPECT_EQ(fixture.record()[2u], 0u);

    constexpr std::uint32_t unsupported_width = 800u;
    constexpr std::uint32_t unsupported_height = 450u;
    constexpr auto unsupported_crop_height =
      ceil_crop_height(unsupported_width, unsupported_height);
    constexpr auto unsupported_crop_top = unsupported_height - unsupported_crop_height;
    constexpr auto unsupported_roi_top = subtitle_roi_edge(
      unsupported_width,
      unsupported_height,
      unsupported_height,
      v2::subtitle_ocr_safe_row_top
    );
    constexpr auto unsupported_roi_bottom = subtitle_roi_edge(
      unsupported_width,
      unsupported_height,
      unsupported_height,
      v2::subtitle_ocr_safe_row_bottom
    );
    ASSERT_TRUE(fixture.run_boxes(
      probability,
      frame,
      91u,
      unsupported_width,
      unsupported_height,
      unsupported_width,
      unsupported_height,
      unsupported_crop_top,
      unsupported_crop_height,
      unsupported_roi_top,
      unsupported_roi_bottom
    ));
    EXPECT_EQ(fixture.record()[2u], 0u);
  }

  TEST(HostSbsOcrGpuTest, EveryAdmittedCoverProtectsWhileEmptyNaNAndMismatchedIdentityAbstain) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, true)) << error;
    constexpr std::uint64_t generation = 77u;

    std::vector<float> line_and_logo(ocr_pixels, 0.0f);
    paint_rectangle(line_and_logo, 240u, 100u, 720u, 108u);
    // A nearby text fragment shares this eligible line: its 40-pixel gap is four times its
    // ten-pixel height, and its vertical span overlaps the first member sufficiently.
    // A tall off-center logo belongs to a different group, covered by the authorization tests.
    paint_rectangle(line_and_logo, 760u, 99u, 840u, 109u);

    ASSERT_TRUE(fixture.run_boxes(line_and_logo, 10u, generation));
    const auto &logo_record = fixture.record();
    ASSERT_EQ(logo_record[2u], 1u);
    ASSERT_EQ(logo_record[3u], 2u);
    ASSERT_EQ(logo_record[4u], 2u);
    EXPECT_EQ(
      read_box(
        logo_record,
        v2::subtitle_ocr_raw_box_offset + v2::subtitle_ocr_box_word_count
      ),
      default_ocr_box(760u, 99u, 840u, 109u)
    );
    EXPECT_EQ(
      read_box(
        logo_record,
        v2::subtitle_ocr_final_box_offset + v2::subtitle_ocr_box_word_count
      ),
      default_ordinary_cover(753u, 92u, 847u, 116u)
    );

    // Every producer-admitted current pair participates immediately. There is no second shape
    // filter or pending owner. These authored covers exercise conditioning, not real glyph recall.
    fixture.reset_locator();
    ASSERT_TRUE(fixture.dispatch_locator(10u, generation, true));
    ASSERT_EQ(fixture.state().size(), v2::subtitle_plane_state_word_count);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_flags_word], v2::subtitle_plane_current_valid_flag);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 3u);
    EXPECT_FALSE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.smoothing_enabled(), 1u);
    EXPECT_EQ(fixture.weight_at(300u, 395u), 1.0f);
    EXPECT_EQ(fixture.weight_at(650u, 395u), 1.0f);
    EXPECT_LT(fixture.output_at(650u, 395u), 0.03f);

    ASSERT_TRUE(fixture.run_boxes(line_and_logo, 11u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(11u, generation));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_flags_word], v2::subtitle_plane_current_valid_flag);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 3u);
    EXPECT_EQ(
      (std::array<std::uint32_t, 4u> {
        fixture.record()[v2::subtitle_ocr_final_box_offset],
        fixture.record()[v2::subtitle_ocr_final_box_offset + 1u],
        fixture.record()[v2::subtitle_ocr_final_box_offset + 2u],
        fixture.record()[v2::subtitle_ocr_final_box_offset + 3u]
      }),
      default_ordinary_cover(234u, 94u, 726u, 114u)
    );
    EXPECT_LT(fixture.output_at(300u, 395u), 0.03f);
    EXPECT_EQ(fixture.smoothing_enabled(), 1u);
    EXPECT_EQ(fixture.weight_at(300u, 395u), 1.0f);
    EXPECT_GT(fixture.weight_at(180u, 395u), 0.0f);
    EXPECT_LT(fixture.weight_at(180u, 395u), 1.0f);
    // The literal Gaussian has fractional support immediately outside the authored rectangle;
    // far away its tail underflows to exact M=0 and the proposal preserves Base bits.
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.weight_at(0u, 0u)), 0u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(0u, 0u)),
      std::bit_cast<std::uint32_t>(0.03f));

    // A fully empty probability map is an authoritative same-frame absence. It removes the
    // current cover union immediately and the conditioned field becomes bit-exact Base.
    std::vector<float> empty(ocr_pixels, 0.0f);
    ASSERT_TRUE(fixture.run_boxes(empty, 12u, generation));
    EXPECT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);
    ASSERT_TRUE(fixture.dispatch_locator(12u, generation));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_flags_word], v2::subtitle_plane_current_valid_flag);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.smoothing_enabled(), 0u);

    // Restore a current line, then inject a non-finite probability. OCR8 abstains (flags=0),
    // UIP1 grants no current authority, and no stale target can alter Base.
    std::vector<float> line_only(ocr_pixels, 0.0f);
    paint_rectangle(line_only, 240u, 100u, 720u, 108u);
    ASSERT_TRUE(fixture.run_boxes(line_only, 13u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(13u, generation));
    ASSERT_TRUE(fixture.run_boxes(line_only, 14u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(14u, generation));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 1u);

    auto nonfinite = line_only;
    nonfinite[0u] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(fixture.run_boxes(nonfinite, 15u, generation));
    EXPECT_EQ(fixture.record()[2u], 0u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);
    ASSERT_TRUE(fixture.dispatch_locator(15u, generation));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.smoothing_enabled(), 0u);

    // A valid OCR8 record whose frame identity does not match b2 is equally non-authoritative.
    ASSERT_TRUE(fixture.run_boxes(line_only, 16u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(16u, generation));
    ASSERT_TRUE(fixture.run_boxes(line_only, 17u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(17u, generation));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 1u);
    ASSERT_TRUE(fixture.run_boxes(line_only, 18u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(19u, generation));
    EXPECT_EQ(fixture.state()[0u], v2::subtitle_plane_state_schema);
    EXPECT_EQ(fixture.state()[1u], v2::subtitle_plane_state_tag);
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.smoothing_enabled(), 0u);

    // Analysis generation and explicit current-record admission are independent authority gates.
    ASSERT_TRUE(fixture.run_boxes(line_only, 20u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(20u, generation + 1u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_flags_word], 0u);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.smoothing_enabled(), 0u);
    ASSERT_TRUE(fixture.run_boxes(line_only, 21u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(21u, generation, false, false));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_flags_word], 0u);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.smoothing_enabled(), 0u);
  }
}  // namespace

#else

TEST(HostSbsOcrGpuTest, WindowsOnly) {
  GTEST_SKIP() << "D3D11 WARP is Windows-only";
}

#endif
