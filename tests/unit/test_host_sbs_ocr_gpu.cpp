/**
 * @file tests/unit/test_host_sbs_ocr_gpu.cpp
 * @brief End-to-end D3D11 WARP coverage for the production OCR preprocess/box producer.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/crypto.h"
  #include "src/depth_coordinate_v2.h"
  #include "src/host_sbs_resolution.h"
  #include "src/host_sbs_shader_cache.h"
  #include "src/platform/windows/sbs_debug_dump.h"

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstddef>
  #include <cstdint>
  #include <cstdlib>
  #include <cstring>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <fstream>
  #include <iterator>
  #include <limits>
  #include <map>
  #include <mutex>
  #include <numeric>
  #include <nlohmann/json.hpp>
  #include <optional>
  #include <stdexcept>
  #include <string>
  #include <tuple>
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
  constexpr std::uint32_t ocr_record_prefix_words = v2::subtitle_ocr_final_box_offset +
    v2::subtitle_ocr_final_box_capacity * v2::subtitle_ocr_box_word_count;

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

  struct configured_center_boundary_t {
    std::uint32_t reference_width;
    std::uint32_t detector_offset;
  };

  constexpr configured_center_boundary_t configured_center_boundary(const std::uint32_t native_width) {
    constexpr auto numerator = static_cast<std::uint64_t>(v2::subtitle_ocr_line_center_tolerance_numerator);
    constexpr auto denominator = static_cast<std::uint64_t>(v2::subtitle_ocr_line_center_tolerance_denominator);
    static_assert(numerator > 0u && denominator > 0u);
    const auto scaled_width = static_cast<std::uint64_t>(native_width) * denominator;
    // The reference may be wider than the analysis crop. Choose its smallest
    // valid width that puts the configured inclusive edge on an integer detector
    // column; this keeps exact-boundary coverage meaningful at both 10% and 2%.
    const auto quantum = scaled_width / std::gcd(scaled_width, ocr_width * numerator);
    const auto reference_width = (native_width + quantum - 1u) / quantum * quantum;
    return {static_cast<std::uint32_t>(reference_width),
      static_cast<std::uint32_t>(ocr_width * reference_width * numerator / scaled_width)};
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

  // Inputs are independently authored DB-expanded bounds and current tight-core heights.
  // Pin the intended search-only policy independently: four eight-pixel cells on X and ceil
  // half the current tight height on Y. Do not recalculate DB unclip or apply this to ribbons.
  constexpr std::array<std::uint32_t, 4u> default_ordinary_cover(
    const std::uint32_t left,
    const std::uint32_t top,
    const std::uint32_t right,
    const std::uint32_t bottom,
    const std::uint32_t core_height = 8u
  ) {
    constexpr std::uint32_t horizontal_guard = 32u;
    const auto vertical_guard = (core_height + 1u) / 2u;
    const auto mapped = default_ocr_box(left > horizontal_guard ? left - horizontal_guard : 0u,
      top > vertical_guard ? top - vertical_guard : 0u,
      std::min(right + horizontal_guard, ocr_width),
      std::min(bottom + vertical_guard, ocr_height));
    return {mapped[0u], std::clamp(mapped[1u], roi_top, roi_bottom),
      mapped[2u], std::clamp(mapped[3u], roi_top, roi_bottom)};
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
    std::array<std::uint32_t, 4u> observation {};
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
  static_assert(sizeof(ocr_resolve_constants_t) == 96u);
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
    constexpr UINT compile_flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
    using cache_key_t = std::tuple<std::filesystem::path, std::string, UINT,
      std::string, std::uint32_t, crypto::sha256_t>;
    static std::map<cache_key_t, ComPtr<ID3DBlob>> compiled_bytecode;
    static std::mutex compiled_bytecode_mutex;
    const auto normalized_path = std::filesystem::weakly_canonical(path).lexically_normal();
    const auto sources = models::host_sbs_shader_cache::snapshot_sources(
      SUNSHINE_SHADERS_DIR, models::host_sbs_shader_cache::parallax_v2_producer_specs);
    const auto closure = models::host_sbs_shader_cache::source_closure_sha256(sources);
    if (closure != models::host_sbs_shader_cache::parallax_v2_producer_source_closure_sha256) {
      error = "OCR fixture sources changed outside their authenticated producer closure";
      return false;
    }
    std::ifstream source(normalized_path, std::ios::binary);
    if (!source) {
      error = "Cannot read OCR fixture shader: " + normalized_path.string();
      return false;
    }
    const std::string source_text((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
    const cache_key_t key {normalized_path, entrypoint, compile_flags, closure, v2::contract_tag,
      crypto::hash(source_text)};
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    {
      // Each process uses one frozen authenticated source closure. Reuse portable DXBC only;
      // every fixture still creates its compute shader on its own D3D11 device.
      const std::lock_guard lock(compiled_bytecode_mutex);
      const auto cached = compiled_bytecode.find(key);
      if (cached != compiled_bytecode.end()) {
        bytecode = cached->second;
      } else {
        const auto status = D3DCompileFromFile(
          normalized_path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
          entrypoint, "cs_5_0", compile_flags, 0u, &bytecode, &diagnostics);
        if (FAILED(status)) {
          error = diagnostics ?
                    std::string(static_cast<const char *>(diagnostics->GetBufferPointer()), diagnostics->GetBufferSize()) :
                    std::string("D3DCompileFromFile failed without diagnostics");
          return false;
        }
        compiled_bytecode.emplace(key, bytecode);
      }
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
      const bool need_locator,
      const D3D_DRIVER_TYPE driver_type = D3D_DRIVER_TYPE_WARP
    ) {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      if (FAILED(D3D11CreateDevice(nullptr, driver_type, nullptr, 0u, requested, static_cast<UINT>(std::size(requested)), D3D11_SDK_VERSION, &device_, &actual, &context_)) || actual < D3D_FEATURE_LEVEL_11_0) {
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
      const std::optional<std::int32_t> line_search_top_source = std::nullopt,
      const std::optional<std::uint64_t> observation_timestamp_us = std::nullopt,
      const bool input_domain_reset = false
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
      const auto source_clock = observation_timestamp_us.value_or(frame * 33333u);
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
        {static_cast<std::uint32_t>(source_clock), static_cast<std::uint32_t>(source_clock >> 32u),
          input_domain_reset ? 1u : 0u, 0u},
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
      std::array<ID3D11UnorderedAccessView *, 2u> resolve_uavs {record_uav_.Get(), candidate_scratch_uav_.Get()};
      context_->CSSetUnorderedAccessViews(1u, static_cast<UINT>(resolve_uavs.size()), resolve_uavs.data(), nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind();
      return read_buffer(device_.Get(), context_.Get(), record_buffer_.Get(), record_);
    }

    // Opt-in diagnostic import only. The replay caller first validates the complete serialized
    // record with the maintained dump reader; round-trip every uploaded word before dispatch.
    bool seed_validated_record_for_replay(const std::vector<std::uint32_t> &prior) {
      if (!record_buffer_ || prior.size() != v2::subtitle_ocr_record_word_count) {
        return false;
      }
      unbind();
      context_->UpdateSubresource(record_buffer_.Get(), 0u, nullptr, prior.data(), 0u, 0u);
      return read_buffer(device_.Get(), context_.Get(), record_buffer_.Get(), record_) &&
             record_ == prior;
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
      const auto source_clock = frame * 33333u;
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
        {static_cast<std::uint32_t>(source_clock), static_cast<std::uint32_t>(source_clock >> 32u), 0u, 0u},
      };
      // This current-publication fixture uses an authored microsecond cadence, shared by OCR and the
      // sealed Base publication. It exercises no artificial UI dwell or model inference.
      coordinate_constants_.joint_observation_timestamp_low = static_cast<std::uint32_t>(source_clock);
      coordinate_constants_.joint_observation_timestamp_high = static_cast<std::uint32_t>(source_clock >> 32u);
      context_->UpdateSubresource(v2_cb_.Get(), 0u, nullptr, &coordinate_constants_, 0u, 0u);
      upload_geometry_publication(source_clock);
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

    bool adapter_description(DXGI_ADAPTER_DESC &description) const {
      ComPtr<IDXGIDevice> dxgi_device;
      ComPtr<IDXGIAdapter> adapter;
      return SUCCEEDED(device_.As(&dxgi_device)) &&
        SUCCEEDED(dxgi_device->GetAdapter(&adapter)) &&
        SUCCEEDED(adapter->GetDesc(&description));
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
      // These OCR topology/current-publication tests author a nonnegative conditioner-budget
      // fixture: zero inside exact current OCR8 tight cores, increasing quadratically outside.
      // Search-cover padding grants no synthetic zero budget. This isolates publication and
      // current pair authority; it executes neither native RGB/palette classification nor the
      // production native-support budget producer and makes no glyph-mask quality claim.
      const double source_x_step = static_cast<double>(source_width) / field_width;
      const double source_y_step = static_cast<double>(source_height) / field_height;
      for (std::uint32_t y = 0u; y < field_height; ++y) {
        for (std::uint32_t x = 0u; x < field_width; ++x) {
          const auto index = static_cast<std::size_t>(y) * field_width + x;
          for (std::uint32_t slot = 0u; current_valid && slot < count; ++slot) {
            const auto offset = v2::subtitle_ocr_raw_box_offset + slot * v2::subtitle_ocr_box_word_count;
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

      zeros.assign(static_cast<std::size_t>(v2::subtitle_ocr_local_candidate_capacity) *
        v2::subtitle_ocr_local_candidate_word_count, 0u);
      if (!create_structured_buffer(device_.Get(), zeros.data(), zeros.size() * sizeof(std::uint32_t),
            4u * sizeof(std::uint32_t), D3D11_BIND_UNORDERED_ACCESS,
            candidate_scratch_buffer_, nullptr, &candidate_scratch_uav_)) return false;

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
    ComPtr<ID3D11Buffer> candidate_scratch_buffer_;
    ComPtr<ID3D11UnorderedAccessView> candidate_scratch_uav_;
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

  // Opt-in diagnostic: replay sealed detector bytes through the production box shaders.
  // The caller supplies paths/constants; no machine-local corpus or image reconstruction is
  // part of this test. It intentionally does not dispatch the fixture's analytic conditioner.
  TEST(HostSbsOcrGpuTest, SavedProbabilityNativeRecordReplay) {
    const auto *cases_environment = std::getenv("SUNSHINE_TEST_OCR_REPLAY_CASES");
    const auto *output_environment = std::getenv("SUNSHINE_TEST_OCR_REPLAY_OUTPUT");
    if (!cases_environment && !output_environment) {
      GTEST_SKIP() << "Set SUNSHINE_TEST_OCR_REPLAY_CASES and SUNSHINE_TEST_OCR_REPLAY_OUTPUT for sealed-P diagnostics";
    }
    ASSERT_NE(cases_environment, nullptr);
    ASSERT_NE(output_environment, nullptr);
    const std::filesystem::path cases_path(cases_environment);
    const std::filesystem::path output_root(output_environment);
    ASSERT_FALSE(output_root.empty());
    ASSERT_TRUE(!std::filesystem::exists(output_root) ||
                (std::filesystem::is_directory(output_root) && std::filesystem::is_empty(output_root)))
      << "Diagnostic output must be a new or empty directory";

    const auto read_bytes = [](const std::filesystem::path &path) {
      std::ifstream input(path, std::ios::binary | std::ios::ate);
      if (!input || input.tellg() < 0) throw std::runtime_error("Cannot read diagnostic input: " + path.string());
      std::vector<char> bytes(static_cast<std::size_t>(input.tellg()));
      input.seekg(0);
      if (!bytes.empty()) input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      if (!input) throw std::runtime_error("Incomplete diagnostic input: " + path.string());
      return bytes;
    };
    const auto sha256 = [](const void *data, const std::size_t size) {
      const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), size));
      constexpr char digits[] = "0123456789abcdef";
      std::string hex;
      hex.reserve(2u * digest.size());
      for (const auto byte : digest) {
        hex.push_back(digits[byte >> 4u]);
        hex.push_back(digits[byte & 15u]);
      }
      return hex;
    };
    const auto write_bytes = [](const std::filesystem::path &path, const void *data, const std::size_t size) {
      std::ofstream output(path, std::ios::binary);
      output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
      if (!output) throw std::runtime_error("Cannot write diagnostic output: " + path.string());
    };
    const auto specification_bytes = read_bytes(cases_path);
    const auto specification = nlohmann::json::parse(specification_bytes.begin(), specification_bytes.end());
    ASSERT_EQ(specification.at("schema"), 1);
    const auto &cases = specification.at("cases");
    ASSERT_TRUE(cases.is_array());
    ASSERT_GT(cases.size(), 0u);
    ASSERT_LE(cases.size(), 8u);
    const auto driver = specification.value("driver", std::string("warp"));
    ASSERT_TRUE(driver == "warp" || driver == "hardware");
    const auto shader_path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / "host_sbs_ocr_boxes_cs.hlsl";
    const auto shader_bytes = read_bytes(shader_path);
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false,
      driver == "hardware" ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP)) << error;
    DXGI_ADAPTER_DESC adapter {};
    ASSERT_TRUE(fixture.adapter_description(adapter));
    std::filesystem::create_directories(output_root);

    for (std::size_t index = 0u; index < cases.size(); ++index) {
      const auto &entry = cases[index];
      SCOPED_TRACE(index);
      const auto probability_path = cases_path.parent_path() / std::filesystem::path(entry.at("probability").get<std::string>());
      const auto probability_bytes = read_bytes(probability_path);
      ASSERT_EQ(probability_bytes.size(), static_cast<std::size_t>(ocr_pixels) * sizeof(float));
      std::vector<float> probability(ocr_pixels);
      std::memcpy(probability.data(), probability_bytes.data(), probability_bytes.size());
      ASSERT_TRUE(std::all_of(probability.begin(), probability.end(), [](const float value) {
        return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
      }));
      const auto source = entry.at("source").get<std::array<std::uint32_t, 2u>>();
      const auto crop = entry.at("crop").get<std::array<std::uint32_t, 2u>>();
      const auto field = entry.at("field").get<std::array<std::uint32_t, 2u>>();
      const auto roi = entry.at("roi").get<std::array<std::uint32_t, 2u>>();
      const auto content = entry.at("content").get<std::array<std::uint32_t, 4u>>();
      const auto reference = entry.at("line_reference").get<std::array<std::int32_t, 3u>>();
      ASSERT_GT(reference[1u], 0);
      const auto frame = entry.at("frame").get<std::uint64_t>();
      const auto generation = entry.at("generation").get<std::uint64_t>();
      const auto source_clock = entry.value("observation_timestamp_us", frame * 33333u);
      const auto domain_reset = entry.value("input_domain_reset", false);
      std::vector<char> prior_bytes;
      nlohmann::json prior_import = nullptr;
      if (entry.contains("prior_record")) {
        // A sealed current/older observation can supply its real SCL1 state. Do not manufacture
        // votes, replace a source clock, or accept a prefix/tail from a different analysis domain.
        const auto prior_path = cases_path.parent_path() /
          std::filesystem::path(entry.at("prior_record").get<std::string>());
        prior_bytes = read_bytes(prior_path);
        ASSERT_EQ(prior_bytes.size(), v2::subtitle_ocr_record_word_count * sizeof(std::uint32_t));
        const auto prior_sha = sha256(prior_bytes.data(), prior_bytes.size());
        ASSERT_EQ(prior_sha, entry.at("prior_record_sha256").get<std::string>());
        std::vector<std::uint32_t> prior(v2::subtitle_ocr_record_word_count);
        std::memcpy(prior.data(), prior_bytes.data(), prior_bytes.size());
        const auto join = [&prior](const std::size_t offset) {
          return prior.at(offset) | (static_cast<std::uint64_t>(prior.at(offset + 1u)) << 32u);
        };
        const auto prior_frame = join(5u);
        const auto prior_clock = join(v2::subtitle_ocr_classifier_history_offset + 4u);
        ASSERT_GT(prior_clock, 0u);
        ASSERT_LE(prior_frame, frame);
        ASSERT_LE(prior_clock, source_clock);
        if (prior_frame == frame) {
          ASSERT_EQ(prior_clock, source_clock);
        }

        const auto &prior_context = entry.at("prior_context");
        const auto capture = prior_context.at("source_extent").get<std::array<std::uint32_t, 2u>>();
        const auto rectangle = prior_context.at("analysis_rect").get<std::array<std::uint32_t, 4u>>();
        const auto authority = prior_context.at("authority").get<std::string>();
        ASSERT_TRUE(authority == "chromium-video" || authority == "foreground-client" ||
                    authority == "full-source");
        ASSERT_LT(rectangle[0u], rectangle[2u]);
        ASSERT_LT(rectangle[1u], rectangle[3u]);
        ASSERT_LE(rectangle[2u], capture[0u]);
        ASSERT_LE(rectangle[3u], capture[1u]);
        ASSERT_EQ(rectangle[2u] - rectangle[0u], source[0u]);
        ASSERT_EQ(rectangle[3u] - rectangle[1u], source[1u]);
        const auto expected_reference = models::subtitle_ocr_line_reference(
          source[0u], source[1u], capture[0u], capture[1u], rectangle[0u], rectangle[1u],
          authority != "full-source");
        ASSERT_TRUE(expected_reference.has_value());
        ASSERT_EQ(expected_reference->center_twice_source, reference[0u]);
        ASSERT_EQ(expected_reference->width, static_cast<std::uint32_t>(reference[1u]));
        ASSERT_EQ(expected_reference->search_top_source, reference[2u]);
        platf::sbs_debug::frame prior_completed;
        prior_completed.field_width = static_cast<int>(field[0u]);
        prior_completed.field_height = static_cast<int>(field[1u]);
        prior_completed.field_content = {content[0u], content[1u], content[2u], content[3u]};
        prior_completed.matched_frame_id = prior_frame;
        prior_completed.matched_observation_timestamp_us = prior_clock;
        prior_completed.depth_input_region = {
          .source_width = capture[0u],
          .source_height = capture[1u],
          .left = rectangle[0u],
          .top = rectangle[1u],
          .right = rectangle[2u],
          .bottom = rectangle[3u],
          .tensor_content = prior_completed.field_content,
          .analysis_generation = generation,
          .authority = authority == "chromium-video" ? models::depth_analysis_authority_e::chromium_video :
            (authority == "foreground-client" ? models::depth_analysis_authority_e::foreground_client :
              models::depth_analysis_authority_e::full_source),
        };
        ASSERT_TRUE(prior_completed.depth_input_region.valid());
        std::vector<std::uint8_t> canonical_bytes(prior_bytes.size());
        std::memcpy(canonical_bytes.data(), prior_bytes.data(), prior_bytes.size());
        ASSERT_TRUE(platf::sbs_debug::detail::subtitle_ocr_record_is_canonical_for_frame(
          canonical_bytes, prior_completed));
        ASSERT_TRUE(fixture.seed_validated_record_for_replay(prior));
        prior_import = {
          {"path", prior_path.string()}, {"sha256", prior_sha},
          {"canonical_validation", "maintained-subtitle_ocr_record_is_canonical_for_frame"},
          {"gpu_upload_roundtrip_bit_exact", true},
          {"matched_frame_id", prior_frame}, {"observation_timestamp_us", prior_clock},
          {"context", prior_context},
        };
      }
      ASSERT_TRUE(fixture.run_boxes(probability, frame, generation,
        source[0u], source[1u], field[0u], field[1u], crop[0u], crop[1u],
        roi[0u], roi[1u], content, reference[0u], static_cast<std::uint32_t>(reference[1u]), reference[2u],
        source_clock, domain_reset));
      const auto &record = fixture.record();
      ASSERT_EQ(record.size(), v2::subtitle_ocr_record_word_count);
      if (entry.contains("expected_final_count")) {
        EXPECT_EQ(record[4u], entry.at("expected_final_count").get<std::uint32_t>());
      }

      const auto case_root = output_root / ("case_" + std::to_string(index));
      ASSERT_TRUE(std::filesystem::create_directory(case_root));
      write_bytes(case_root / "uploaded_probability.f32", probability.data(), probability_bytes.size());
      if (!prior_bytes.empty()) {
        write_bytes(case_root / "uploaded_prior_record.u32", prior_bytes.data(), prior_bytes.size());
      }
      write_bytes(case_root / "ocr_record.u32", record.data(), record.size() * sizeof(record[0u]));
      nlohmann::json identity {
        {"schema", 1}, {"kind", "native-D3D11-production-cells-and-resolve-sealed-P-diagnostic"},
        {"driver", driver}, {"adapter", {{"vendor_id", adapter.VendorId}, {"device_id", adapter.DeviceId},
          {"subsystem_id", adapter.SubSysId}, {"revision", adapter.Revision},
          {"luid_low", adapter.AdapterLuid.LowPart}, {"luid_high", adapter.AdapterLuid.HighPart}}},
        {"case", entry}, {"input_specification", cases_path.string()},
        {"observation_timestamp_us", source_clock}, {"input_domain_reset", domain_reset},
        {"input_specification_sha256", sha256(specification_bytes.data(), specification_bytes.size())},
        {"probability_sha256", sha256(probability.data(), probability_bytes.size())},
        {"record_sha256", sha256(record.data(), record.size() * sizeof(record[0u]))},
        {"record_word_count", record.size()}, {"record_flags", record[2u]},
        {"record_raw_count", record[3u]}, {"record_final_count", record[4u]},
        {"depth_coordinate_v2_schema", v2::contract_schema}, {"depth_coordinate_v2_tag", v2::contract_tag},
        {"depth_coordinate_v2_canonical_sha256", v2::contract_canonical_sha256},
        {"producer_source_closure_sha256", models::host_sbs_shader_cache::parallax_v2_producer_source_closure_sha256},
        {"ocr_shader_path", shader_path.string()}, {"ocr_shader_sha256", sha256(shader_bytes.data(), shader_bytes.size())},
        {"shader_compile_flags", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3},
        {"expected_record_bit_exact", nullptr}, {"renderer_or_conditioner_executed", false},
        {"prior_record_input", prior_import},
        {"expected_final_count", entry.value("expected_final_count", nlohmann::json(nullptr))},
      };
      if (entry.contains("expected_record")) {
        const auto expected_path = cases_path.parent_path() / std::filesystem::path(entry.at("expected_record").get<std::string>());
        const auto expected_bytes = read_bytes(expected_path);
        ASSERT_EQ(expected_bytes.size(), record.size() * sizeof(record[0u]));
        const bool equal = std::memcmp(expected_bytes.data(), record.data(), expected_bytes.size()) == 0;
        identity["expected_record_sha256"] = sha256(expected_bytes.data(), expected_bytes.size());
        identity["expected_record_bit_exact"] = equal;
        EXPECT_TRUE(equal) << "Native OCR record differs from the sealed expected record";
      }
      const auto identity_text = identity.dump(2);
      write_bytes(case_root / "identity.json", identity_text.data(), identity_text.size());
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

  TEST(HostSbsOcrGpuTest, OrdinaryCoverAddsFourHorizontalCellsAndHalfCoreHeightSearchOnly) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    struct cover_case_t {
      std::array<std::uint32_t, 4u> core;
      std::array<std::uint32_t, 4u> db_cover;
    };
    const std::array<cover_case_t, 3u> cases {{
      {{240u, 100u, 720u, 108u}, {234u, 94u, 726u, 114u}},
      // Keep the left/right clipping inputs; reference placement is independent below.
      {{0u, 100u, 776u, 108u}, {0u, 94u, 782u, 114u}},
      {{192u, 100u, 960u, 108u}, {186u, 94u, 960u, 114u}},
    }};
    for (const auto &test_case : cases) {
      SCOPED_TRACE(std::to_string(test_case.core[0u]));
      const auto &core = test_case.core;
      const auto &db = test_case.db_cover;
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, core[0u], core[1u], core[2u], core[3u]);
      // Isolate cover geometry using a legal matched screen reference centered on
      // each unchanged synthetic line, just as capture_native_pairs does below.
      const auto analysis_left = source_width - core[0u] - core[2u];
      ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u,
        2u * source_width, analysis_left));
      ASSERT_EQ(fixture.record()[2u], 1u);
      ASSERT_EQ(fixture.record()[3u], 1u);
      ASSERT_EQ(fixture.record()[4u], 1u);
      EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
        default_ocr_box(core[0u], core[1u], core[2u], core[3u]));
      const auto cover = read_box(fixture.record(), v2::subtitle_ocr_final_box_offset);
      EXPECT_EQ(cover, default_ordinary_cover(db[0u], db[1u], db[2u], db[3u], core[3u] - core[1u]));
      const auto previous_db_cover = default_ocr_box(db[0u], db[1u], db[2u], db[3u]);
      EXPECT_LT(cover[1u], previous_db_cover[1u]);
      EXPECT_GT(cover[3u], previous_db_cover[3u]);
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
    // its expanded cover (DB pad 6 plus the 32-pixel search guard).
    struct edge_case_t {
      std::array<std::uint32_t, 4u> core;
      std::array<std::uint32_t, 4u> db_cover;
    };
    const std::array<edge_case_t, 2u> cases {{
      {{384u, 100u, 576u, 108u}, {378u, 94u, 582u, 114u}},
      {{230u, 100u, 730u, 108u}, {224u, 94u, 736u, 114u}},
    }};
    static_assert(576u * field_width % ocr_width == 0u);
    static_assert((736u + 32u) * field_width % ocr_width == 0u);
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
        default_ordinary_cover(db[0u], db[1u], db[2u], db[3u], core[3u] - core[1u]));
    }
  }

  // MapXFloor and MapXCeil clamp their own column to OCR_WIDTH, so a caller passing a column
  // beyond it maps to the content's right edge instead of a uint32 product that wrapped to a
  // narrow column (5577880 * 770 wraps to 304, which used to map to field columns 0 and 1).
  TEST(HostSbsOcrGpuTest, ColumnMappingClampsAnyColumnToTheOcrWidth) {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL actual {};
    ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, requested,
      static_cast<UINT>(std::size(requested)), D3D11_SDK_VERSION, &device, &actual, &context)));
    static constexpr char probe[] =
      "#include \"host_sbs_ocr_boxes_cs.hlsl\"\n"
      "StructuredBuffer<uint> MapXProbeColumns : register(t2);\n"
      "RWStructuredBuffer<uint> MapXProbeMapped : register(u2);\n"
      "[numthreads(1, 1, 1)]\n"
      "void map_x_probe_main(uint3 id : SV_DispatchThreadID) {\n"
      "    MapXProbeMapped[2u * id.x] = MapXFloor(MapXProbeColumns[id.x]);\n"
      "    MapXProbeMapped[2u * id.x + 1u] = MapXCeil(MapXProbeColumns[id.x]);\n"
      "}\n";
    // The probe compiles as if it sat beside the shader, so its includes resolve there.
    const auto probe_path = (std::filesystem::path(SUNSHINE_SHADERS_DIR) / "host_sbs_ocr_map_x_probe.hlsl").string();
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    const auto compiled = D3DCompile(probe, sizeof(probe) - 1u, probe_path.c_str(), nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE, "map_x_probe_main", "cs_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0u, &bytecode, &diagnostics);
    ASSERT_TRUE(SUCCEEDED(compiled)) << (diagnostics ?
      std::string(static_cast<const char *>(diagnostics->GetBufferPointer()), diagnostics->GetBufferSize()) : std::string());
    ComPtr<ID3D11ComputeShader> shader;
    ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shader)));

    // Content columns 3..773 of the field (770 wide); only tensor_content is read.
    constexpr std::array<std::uint32_t, 4u> content {3u, 0u, 773u, field_height};
    constexpr std::uint32_t content_width = content[2u] - content[0u];
    const ocr_resolve_constants_t constants {0u, 0u, 0u, 0u, source_width, source_height, field_width, field_height,
      0u, 1u, 0u, 1u, content, 0, source_width, 0, 0u};
    ComPtr<ID3D11Buffer> constant_buffer;
    ASSERT_TRUE(create_constant_buffer(device.Get(), &constants, sizeof(constants), constant_buffer));
    const std::vector<std::uint32_t> columns {0u, 1u, 576u, ocr_width - 1u, ocr_width, ocr_width + 1u, 5577880u,
      0x80000000u, 0xffffffffu};
    ComPtr<ID3D11Buffer> column_buffer, mapped_buffer;
    ComPtr<ID3D11ShaderResourceView> column_srv;
    ComPtr<ID3D11UnorderedAccessView> mapped_uav;
    ASSERT_TRUE(create_structured_buffer(device.Get(), columns.data(), columns.size() * sizeof(std::uint32_t),
      sizeof(std::uint32_t), D3D11_BIND_SHADER_RESOURCE, column_buffer, &column_srv, nullptr));
    const std::vector<std::uint32_t> zeros(2u * columns.size(), 0u);
    ASSERT_TRUE(create_structured_buffer(device.Get(), zeros.data(), zeros.size() * sizeof(std::uint32_t),
      sizeof(std::uint32_t), D3D11_BIND_UNORDERED_ACCESS, mapped_buffer, nullptr, &mapped_uav));
    context->CSSetShader(shader.Get(), nullptr, 0u);
    context->CSSetConstantBuffers(0u, 1u, constant_buffer.GetAddressOf());
    context->CSSetShaderResources(2u, 1u, column_srv.GetAddressOf());
    context->CSSetUnorderedAccessViews(2u, 1u, mapped_uav.GetAddressOf(), nullptr);
    context->Dispatch(static_cast<UINT>(columns.size()), 1u, 1u);
    std::vector<std::uint32_t> mapped;
    ASSERT_TRUE(read_buffer(device.Get(), context.Get(), mapped_buffer.Get(), mapped));
    ASSERT_EQ(mapped.size(), 2u * columns.size());
    for (std::size_t i = 0; i != columns.size(); ++i) {
      SCOPED_TRACE(std::to_string(columns[i]));
      const std::uint64_t x = std::min(columns[i], ocr_width);
      EXPECT_EQ(mapped[2u * i], content[0u] + x * content_width / ocr_width);
      EXPECT_EQ(mapped[2u * i + 1u], content[0u] + (x * content_width + ocr_width - 1u) / ocr_width);
      EXPECT_LE(mapped[2u * i + 1u], content[2u]);
    }
    EXPECT_EQ(mapped[2u * 6u], content[2u]);
    EXPECT_EQ(mapped[2u * 6u + 1u], content[2u]);
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
    EXPECT_EQ(cover, (std::array<std::uint32_t, 4u> {465u, 718u, 1101u, 787u}));
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


  TEST(HostSbsOcrGpuTest, CurrentCaptionSearchIncludesDetectorMissingLeadingAndLowerStrokeWitnesses) {
    // Actual live119 P component bounds, source2536x1427/crop1004+423 and content1540x866.
    // Rectangles author only the retained horizontal component geometry. The missing stroke
    // gets no synthetic P island, classifier relaxation, prior vote, or blanket-cover support.
    // These three source coordinates are partial visible-stroke witnesses, not all-glyph GT.
    constexpr std::uint32_t native_w = 2536u, native_h = 1427u;
    constexpr std::uint32_t native_field_w = 1540u, native_field_h = 868u;
    constexpr std::uint32_t content_h = 866u;
    constexpr auto crop_h = ceil_crop_height(native_w, native_h);
    constexpr auto crop_top = native_h - crop_h;
    constexpr auto native_roi_top = subtitle_roi_edge(native_w, native_h, 0u,
      content_h, v2::subtitle_ocr_safe_row_top);
    constexpr auto native_roi_bottom = subtitle_roi_edge(native_w, native_h, 0u,
      content_h, v2::subtitle_ocr_safe_row_bottom);
    struct live_geometry_case_t {
      std::array<std::uint32_t, 4u> detector_core;
      std::array<std::uint32_t, 4u> expected_core;
      std::array<std::uint32_t, 4u> expected_cover;
      std::array<std::uint32_t, 2u> lower_witness;
    };
    constexpr std::array<live_geometry_case_t, 3u> cases {{
      {{408u, 103u, 587u, 120u}, {654u, 774u, 942u, 802u}, {585u, 742u, 1011u, 834u}, {1428u, 1353u}},
      {{379u, 109u, 596u, 122u}, {607u, 784u, 957u, 806u}, {542u, 758u, 1022u, 831u}, {1092u, 1361u}},
      {{393u, 110u, 577u, 122u}, {630u, 785u, 926u, 806u}, {566u, 763u, 990u, 828u}, {1533u, 1360u}},
    }};
    const auto contains_source_center = [](const std::array<std::uint32_t, 4u> &box,
                                           const std::uint32_t x, const std::uint32_t y) {
      return (2ull * x + 1u) * native_field_w >= 2ull * box[0u] * native_w &&
             (2ull * x + 1u) * native_field_w < 2ull * box[2u] * native_w &&
             (2ull * y + 1u) * content_h >= 2ull * box[1u] * native_h &&
             (2ull * y + 1u) * content_h < 2ull * box[3u] * native_h;
    };
    for (std::size_t index = 0u; index != cases.size(); ++index) {
      SCOPED_TRACE(index);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      const auto &test_case = cases[index];
      const auto &p = test_case.detector_core;
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, p[0u], p[1u], p[2u], p[3u]);
      ASSERT_TRUE(fixture.run_boxes(probability, 7756u + index, 77u,
        native_w, native_h, native_field_w, native_field_h, crop_top, crop_h,
        native_roi_top, native_roi_bottom, {0u, 0u, native_field_w, content_h}));
      ASSERT_EQ(fixture.record()[2u], 1u);
      ASSERT_EQ(fixture.record()[3u], 1u);
      ASSERT_EQ(fixture.record()[4u], 1u);
      const auto core = read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset);
      const auto cover = read_box(fixture.record(), v2::subtitle_ocr_final_box_offset);
      EXPECT_EQ(core, test_case.expected_core);
      EXPECT_EQ(cover, test_case.expected_cover);
      EXPECT_EQ(fixture.record()[v2::subtitle_ocr_raw_box_offset + 5u], 0u);
      EXPECT_FALSE(contains_source_center(core, test_case.lower_witness[0u], test_case.lower_witness[1u]));
      EXPECT_TRUE(contains_source_center(cover, test_case.lower_witness[0u], test_case.lower_witness[1u]));
      if (index == 0u) {
        EXPECT_FALSE(contains_source_center(core, 966u, 1302u));
        EXPECT_TRUE(contains_source_center(cover, 966u, 1302u));
      }
    }
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
                                    const std::vector<ocr_native_pair_t> &pairs,
                                    const std::uint64_t frame_id = 40u) {
    ASSERT_EQ(record.size(), v2::subtitle_ocr_record_word_count);
    ASSERT_LE(pairs.size(), v2::subtitle_ocr_final_box_capacity);
    EXPECT_EQ(record[0u], v2::subtitle_ocr_record_schema);
    EXPECT_EQ(record[1u], v2::subtitle_ocr_record_tag);
    EXPECT_EQ(record[2u], 1u);
    EXPECT_EQ(record[3u], pairs.size());
    EXPECT_EQ(record[4u], pairs.size());
    EXPECT_EQ(record[5u], static_cast<std::uint32_t>(frame_id));
    EXPECT_EQ(record[6u], static_cast<std::uint32_t>(frame_id >> 32u));
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

  std::array<std::uint32_t, 48u> classifier_history(const std::vector<std::uint32_t> &record) {
    std::array<std::uint32_t, 48u> history {};
    for (std::size_t word = 0u; word < history.size(); ++word)
      history[word] = record.at(ocr_record_prefix_words + word);
    return history;
  }

  bool observe_classifier(ocr_warp_fixture_t &fixture, const std::vector<float> &probability,
                          const std::uint64_t frame, const std::uint64_t source_clock,
                          const bool reset = false, const std::uint64_t generation = 77u) {
    return fixture.run_boxes(probability, frame, generation, source_width, source_height,
      field_width, field_height, source_crop_top, source_crop_height, roi_top, roi_bottom,
      {}, std::nullopt, std::nullopt, std::nullopt, source_clock, reset);
  }

  void paint_tilted_line(std::vector<float> &probability, const std::uint32_t left,
                         const std::uint32_t top, const std::uint32_t width,
                         const std::uint32_t height, const std::uint32_t rise) {
    for (std::uint32_t x = 0u; x < width; ++x) {
      const auto y = top + x * rise / width;
      paint_rectangle(probability, left + x, y, left + x + 1u, y + height);
    }
  }

  void paint_envelope_line(std::vector<float> &probability, const std::uint32_t left,
                            const std::int32_t top, const std::uint32_t width,
                            const std::int32_t height, const std::int32_t top_rise,
                            const std::int32_t bottom_rise) {
    ASSERT_GT(width, 1u);
    for (std::uint32_t x = 0u; x < width; ++x) {
      const auto first = top + static_cast<std::int32_t>(x) * top_rise /
        static_cast<std::int32_t>(width - 1u);
      const auto last = top + height + static_cast<std::int32_t>(x) * bottom_rise /
        static_cast<std::int32_t>(width - 1u);
      ASSERT_GE(first, 0);
      ASSERT_GT(last, first);
      paint_rectangle(probability, left + x, static_cast<std::uint32_t>(first),
        left + x + 1u, static_cast<std::uint32_t>(last));
    }
  }

  bool learn_three_layout_entries(ocr_warp_fixture_t &fixture,
                                   const std::vector<float> &strong) {
    const std::vector<float> empty(ocr_pixels, 0.0f);
    std::uint64_t frame = 1u;
    std::uint64_t clock = 1'000'000u;
    for (std::uint32_t episode = 0u; episode < 3u; ++episode) {
      if (episode != 0u) {
        if (!observe_classifier(fixture, empty, frame++, clock)) return false;
        clock += 100'000u;
      }
      if (!observe_classifier(fixture, strong, frame++, clock) || fixture.record()[4u] == 0u)
        return false;
      clock += 100'000u;
    }
    return classifier_history(fixture.record())[3u] == 3u;
  }

  TEST(HostSbsOcrGpuTest, LeftAlignedMultilineRequiresEachCurrentLineToMeetExistingCenterPolicy) {
    // All three independent horizontal lines span the reference center at detector X480.
    // Their common left edge alone cannot authorize a multiline union: the shorter two
    // midpoints, X440 and X460, exceed the current two-percent midpoint tolerance.
    const std::vector<ocr_candidate_patch_t> left_aligned {
      {{300u, 70u, 660u, 85u}},
      {{300u, 100u, 580u, 115u}},
      {{300u, 130u, 620u, 145u}},
    };
    const std::vector<ocr_candidate_patch_t> centered {
      left_aligned.front(),
      {{340u, 100u, 620u, 115u}},
      {{320u, 130u, 640u, 145u}},
    };
    ocr_warp_fixture_t single_fixture, left_fixture, centered_fixture;
    std::string error;
    ASSERT_TRUE(single_fixture.initialize(error, false, true, false)) << error;
    ASSERT_TRUE(left_fixture.initialize(error, false, true, false)) << error;
    ASSERT_TRUE(centered_fixture.initialize(error, false, true, false)) << error;
    std::vector<float> single(ocr_pixels, 0.0f), left(ocr_pixels, 0.0f), aligned(ocr_pixels, 0.0f);
    paint_candidate_patches(single, {left_aligned.front()});
    paint_candidate_patches(left, left_aligned);
    paint_candidate_patches(aligned, centered);
    ASSERT_TRUE(observe_classifier(single_fixture, single, 40u, 1'000'000u));
    ASSERT_EQ(single_fixture.record()[4u], 1u);
    const auto single_pair = read_native_pair(single_fixture.record(), 0u);
    const auto single_history = classifier_history(single_fixture.record());
    ASSERT_TRUE(observe_classifier(left_fixture, left, 40u, 1'000'000u));
    expect_retained_native_pairs(left_fixture.record(), {single_pair});
    // Rejected short rows neither borrow a centered stack nor teach layout locations.
    EXPECT_EQ(classifier_history(left_fixture.record()), single_history);

    std::vector<ocr_native_pair_t> expected_centered;
    ASSERT_TRUE(capture_native_pairs(single_fixture, centered, expected_centered));
    ASSERT_EQ(expected_centered.size(), 3u);
    ASSERT_TRUE(observe_classifier(centered_fixture, aligned, 40u, 1'000'000u));
    expect_retained_native_pairs(centered_fixture.record(), expected_centered);
    for (const auto &pair : expected_centered) {
      EXPECT_EQ(pair[5u], 0u);
    }
    const auto centered_history = classifier_history(centered_fixture.record());

    // This is current geometric admission, with no retained cover or death grace.
    // An authoritative blank clears both layouts immediately and retains only layout memory.
    const std::vector<float> empty(ocr_pixels, 0.0f);
    for (auto *fixture : {&left_fixture, &centered_fixture}) {
      const auto before = fixture == &left_fixture ? single_history : centered_history;
      ASSERT_TRUE(observe_classifier(*fixture, empty, 41u, 1'033'333u));
      expect_retained_native_pairs(fixture->record(), {}, 41u);
      const auto after = classifier_history(fixture->record());
      EXPECT_EQ(after[3u], before[3u]);
      for (std::size_t word = 27u; word < before.size(); ++word) {
        EXPECT_EQ(after[word], before[word]);
      }
    }
  }

  TEST(HostSbsOcrGpuTest, CurrentBracketedStackRecoversAsymmetricMiddleWithoutTeaching) {
    // One edge varies with the current glyph shape while the other stays horizontal. These
    // authored fields exercise both edge choices and both signs without a fitted CPU classifier.
    for (const auto shape : {std::array<std::int32_t, 7u> {95, 8, 0, 95, 115, 125, 13},
                             std::array<std::int32_t, 7u> {95, 0, -8, 95, 115, 125, 13},
                             std::array<std::int32_t, 7u> {103, -8, 0, 95, 123, 135, 17}}) {
      SCOPED_TRACE(shape[1u]);
      SCOPED_TRACE(shape[2u]);
      ocr_warp_fixture_t anchors_fixture, stack_fixture;
      std::string error;
      ASSERT_TRUE(anchors_fixture.initialize(error, false, true, false)) << error;
      ASSERT_TRUE(stack_fixture.initialize(error, false, true, false)) << error;
      std::vector<float> anchors(ocr_pixels, 0.0f);
      paint_rectangle(anchors, 300u, 70u, 660u, 85u);
      const auto lower_top = static_cast<std::uint32_t>(shape[5u]);
      paint_rectangle(anchors, 300u, lower_top, 660u, lower_top + 15u);
      ASSERT_TRUE(observe_classifier(anchors_fixture, anchors, 1u, 1000000u));
      ASSERT_EQ(anchors_fixture.record()[4u], 2u);
      const auto upper_pair = read_native_pair(anchors_fixture.record(), 0u);
      const auto lower_pair = read_native_pair(anchors_fixture.record(), 1u);
      const auto anchors_history = classifier_history(anchors_fixture.record());
      auto stack = anchors;
      paint_envelope_line(stack, 390u, shape[0u], 180u, 20, shape[1u], shape[2u]);
      ASSERT_TRUE(observe_classifier(stack_fixture, stack, 1u, 1000000u));
      ASSERT_EQ(stack_fixture.record()[3u], 3u);
      ASSERT_EQ(stack_fixture.record()[4u], 3u);
      EXPECT_EQ(read_native_pair(stack_fixture.record(), 0u), upper_pair);
      EXPECT_EQ(read_native_pair(stack_fixture.record(), 2u), lower_pair);
      EXPECT_EQ(read_box(stack_fixture.record(), v2::subtitle_ocr_raw_box_offset +
        v2::subtitle_ocr_box_word_count), default_ocr_box(390u,
          static_cast<std::uint32_t>(shape[3u]), 570u, static_cast<std::uint32_t>(shape[4u])));
      const auto pad = static_cast<std::uint32_t>(shape[6u]);
      EXPECT_EQ(read_box(stack_fixture.record(), v2::subtitle_ocr_final_box_offset +
        v2::subtitle_ocr_box_word_count), default_ordinary_cover(390u - pad,
          static_cast<std::uint32_t>(shape[3u]) - pad, 570u + pad,
          static_cast<std::uint32_t>(shape[4u]) + pad,
          static_cast<std::uint32_t>(shape[4u] - shape[3u])));
      // Both observations are cold and have identical strong anchors. The assisted member
      // must not change any learned vote, trusted height/span, episode, or clock identity.
      EXPECT_EQ(classifier_history(stack_fixture.record()), anchors_history);
      std::vector<float> empty(ocr_pixels, 0.0f);
      ASSERT_TRUE(observe_classifier(stack_fixture, empty, 2u, 1033333u));
      EXPECT_EQ(stack_fixture.record()[4u], 0u);
    }
  }

  TEST(HostSbsOcrGpuTest, CurrentBracketedStackRequiresBothIndependentAnchorsAndOwnCenter) {
    // The off-center word still overlaps the horizontal anchors on X. A centered union with
    // those anchors must not lend the word its own missing center or opposite-side evidence.
    for (const auto scenario : {0u, 1u, 2u, 3u}) {
      SCOPED_TRACE(scenario);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> anchors(ocr_pixels, 0.0f);
      if (scenario != 0u) {
        // The last case keeps X overlap but moves only the upper anchor off center.
        paint_rectangle(anchors, scenario == 3u ? 470u : 300u, 70u,
          scenario == 3u ? 830u : 660u, 85u);
      }
      if (scenario != 1u) paint_rectangle(anchors, 300u, 125u, 660u, 140u);
      ASSERT_TRUE(observe_classifier(fixture, anchors, 40u, 1000000u));
      const auto count = fixture.record()[4u];
      ASSERT_EQ(count, scenario == 2u ? 2u : 1u);
      std::vector<ocr_native_pair_t> expected;
      for (std::uint32_t slot = 0u; slot < count; ++slot)
        expected.push_back(read_native_pair(fixture.record(), slot));
      auto combined = anchors;
      paint_envelope_line(combined, scenario == 2u ? 650u : 390u,
        95, 180u, 20, 8, 0);
      ASSERT_TRUE(observe_classifier(fixture, combined, 40u, 1000000u, true));
      expect_retained_native_pairs(fixture.record(), expected);
    }
  }

  TEST(HostSbsOcrGpuTest, CurrentBracketedStackRejectsSteepAndCoherentlyTiltedMembers) {
    for (const auto steep_one_edge : {false, true}) {
      SCOPED_TRACE(steep_one_edge);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> anchors(ocr_pixels, 0.0f);
      paint_rectangle(anchors, 300u, 70u, 660u, 85u);
      paint_rectangle(anchors, 300u, 135u, 660u, 150u);
      ASSERT_TRUE(observe_classifier(fixture, anchors, 40u, 1000000u));
      ASSERT_EQ(fixture.record()[4u], 2u);
      const std::vector<ocr_native_pair_t> expected {
        read_native_pair(fixture.record(), 0u), read_native_pair(fixture.record(), 1u)};
      auto combined = anchors;
      if (steep_one_edge) {
        // The bottom is flat, but the top rises by more than ten degrees. Its thirty-row
        // height is still within twice either anchor, so angle is the independent veto.
        paint_envelope_line(combined, 440u, 95, 80u, 30, 20, 0);
      } else {
        // Both boundaries follow the same moderate tilt; current stack evidence must not
        // launder a coherent oblique line merely because it lies between horizontal text.
        paint_envelope_line(combined, 420u, 95, 120u, 15, 8, 8);
      }
      ASSERT_TRUE(observe_classifier(fixture, combined, 40u, 1000000u, true));
      expect_retained_native_pairs(fixture.record(), expected);
    }
  }

  TEST(HostSbsOcrGpuTest, CurrentBracketedStackCannotChainAmbiguousNeighbors) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> anchors(ocr_pixels, 0.0f);
    paint_rectangle(anchors, 300u, 40u, 660u, 55u);
    paint_rectangle(anchors, 300u, 125u, 660u, 140u);
    ASSERT_TRUE(observe_classifier(fixture, anchors, 40u, 1000000u));
    ASSERT_EQ(fixture.record()[4u], 2u);
    const std::vector<ocr_native_pair_t> expected {
      read_native_pair(fixture.record(), 0u), read_native_pair(fixture.record(), 1u)};
    auto combined = anchors;
    paint_envelope_line(combined, 390u, 65, 180u, 20, 8, 0);
    paint_envelope_line(combined, 390u, 95, 180u, 20, 8, 0);
    // Each member has one nearby strong anchor and one nearby ambiguous member. Neither
    // has two independent strong anchors within a line height, so neither may bootstrap it.
    ASSERT_TRUE(observe_classifier(fixture, combined, 40u, 1000000u, true));
    expect_retained_native_pairs(fixture.record(), expected);
  }

  TEST(HostSbsOcrGpuTest, DistantRowBridgeDoesNotMergeIndependentCenteredLines) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> upper(ocr_pixels, 0.0f), lower(ocr_pixels, 0.0f);
    paint_rectangle(upper, 300u, 80u, 660u, 90u);
    paint_rectangle(lower, 300u, 110u, 660u, 120u);
    ASSERT_TRUE(fixture.run_boxes(upper, 1u, 77u));
    ASSERT_EQ(fixture.record()[4u], 1u);
    const auto upper_pair = read_native_pair(fixture.record(), 0u);
    ASSERT_TRUE(fixture.run_boxes(lower, 2u, 77u));
    ASSERT_EQ(fixture.record()[4u], 1u);
    const auto lower_pair = read_native_pair(fixture.record(), 0u);
    std::vector<float> combined = upper;
    paint_rectangle(combined, 300u, 110u, 660u, 120u);
    // This distant vertical component fills the global inactive-row gap, without touching
    // either centered text component. It must not change their cores or DB covers.
    paint_rectangle(combined, 850u, 85u, 866u, 115u);
    ASSERT_TRUE(fixture.run_boxes(combined, 40u, 77u));
    expect_retained_native_pairs(fixture.record(), {upper_pair, lower_pair});
  }

  TEST(HostSbsOcrGpuTest, CurrentHorizontalDialogueRejectsSeparateTiltedSceneText) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> dialogue(ocr_pixels, 0.0f);
    paint_rectangle(dialogue, 240u, 125u, 720u, 140u);
    ASSERT_TRUE(fixture.run_boxes(dialogue, 1u, 77u));
    ASSERT_EQ(fixture.record()[4u], 1u);
    const auto expected = read_native_pair(fixture.record(), 0u);
    auto combined = dialogue;
    paint_tilted_line(combined, 400u, 55u, 160u, 8u, 40u);
    ASSERT_TRUE(fixture.run_boxes(combined, 40u, 77u));
    expect_retained_native_pairs(fixture.record(), {expected});
  }

  TEST(HostSbsOcrGpuTest, FirstStrongCueAndSingleCjkSquareEnterImmediately) {
    for (const auto square : {false, true}) {
      SCOPED_TRACE(square);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> probability(ocr_pixels, 0.0f);
      const std::array<std::uint32_t, 4u> rectangle = square ?
        std::array<std::uint32_t, 4u> {468u, 100u, 492u, 124u} :
        std::array<std::uint32_t, 4u> {240u, 100u, 720u, 115u};
      paint_rectangle(probability, rectangle[0u], rectangle[1u], rectangle[2u], rectangle[3u]);
      ASSERT_TRUE(observe_classifier(fixture, probability, 1u, 1000000u));
      ASSERT_EQ(fixture.record()[4u], 1u);
      EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
        default_ocr_box(rectangle[0u], rectangle[1u], rectangle[2u], rectangle[3u]));
      const auto history = classifier_history(fixture.record());
      EXPECT_EQ(history[0u], 0x314C4353u);
      EXPECT_EQ(history[1u], 1u);
      EXPECT_EQ(history[3u], 1u);
      EXPECT_EQ(history[23u], 1u);
    }
  }

  TEST(HostSbsOcrGpuTest, EmptyObservationClearsCurrentProtectionButRetainsLayoutPrior) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> cue(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
    paint_rectangle(cue, 240u, 100u, 720u, 115u);
    ASSERT_TRUE(observe_classifier(fixture, cue, 1u, 1000000u));
    const auto learned = classifier_history(fixture.record());
    ASSERT_EQ(learned[3u], 1u);
    ASSERT_TRUE(observe_classifier(fixture, empty, 2u, 1033333u));
    ASSERT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[4u], 0u);
    for (std::size_t word = v2::subtitle_ocr_raw_box_offset; word < ocr_record_prefix_words; ++word)
      EXPECT_EQ(fixture.record()[word], 0u);
    const auto blank = classifier_history(fixture.record());
    EXPECT_EQ(blank[3u], learned[3u]);
    EXPECT_EQ(blank[24u], 1u);
    EXPECT_EQ(blank[25u], 1033333u);
    EXPECT_EQ(blank[26u], 0u);
    for (std::size_t word = 32u; word < blank.size(); ++word)
      EXPECT_EQ(blank[word], learned[word]);
  }

  TEST(HostSbsOcrGpuTest, LayoutLearnsIndependentEntriesRatherThanRepeatedStaticFrames) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> cue(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
    paint_rectangle(cue, 240u, 100u, 720u, 115u);
    ASSERT_TRUE(observe_classifier(fixture, cue, 1u, 1000000u));
    const auto first = classifier_history(fixture.record());
    for (std::uint64_t frame = 2u; frame <= 20u; ++frame)
      ASSERT_TRUE(observe_classifier(fixture, cue, frame, 1000000u + frame * 33333u));
    const auto static_tail = classifier_history(fixture.record());
    EXPECT_EQ(static_tail[3u], 1u);
    for (std::size_t word = 32u; word < first.size(); ++word)
      EXPECT_EQ(static_tail[word], first[word]);
    ASSERT_TRUE(observe_classifier(fixture, empty, 21u, 2000000u));
    ASSERT_TRUE(observe_classifier(fixture, cue, 22u, 2100000u));
    EXPECT_EQ(classifier_history(fixture.record())[3u], 2u);
    ASSERT_TRUE(observe_classifier(fixture, empty, 23u, 2200000u));
    ASSERT_TRUE(observe_classifier(fixture, cue, 24u, 2300000u));
    const auto third = classifier_history(fixture.record());
    EXPECT_EQ(third[3u], 3u);
    EXPECT_GE(*std::max_element(third.begin() + 32u, third.end()), 8u);
  }

  TEST(HostSbsOcrGpuTest, DuplicateFrameClockAndZeroClockCannotAccumulateLayoutEvidence) {
    for (const auto pair : {std::array<std::uint64_t, 2u> {1u, 1000000u}, {2u, 1000000u}, {1u, 1100000u}, {3u, 0u}}) {
      SCOPED_TRACE(pair[0u]);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> cue(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
      paint_rectangle(cue, 240u, 100u, 720u, 115u);
      ASSERT_TRUE(observe_classifier(fixture, cue, 1u, 1000000u));
      const auto before = classifier_history(fixture.record());
      ASSERT_TRUE(observe_classifier(fixture, empty, pair[0u], pair[1u]));
      const auto after = classifier_history(fixture.record());
      EXPECT_LE(after[3u], before[3u]);
      for (std::size_t word = 32u; word < before.size(); ++word) {
        EXPECT_LE(after[word], before[word]);
        if (pair[1u] == 1000000u) {
          EXPECT_EQ(after[word], before[word]);
        }
      }
      // An unaccepted blank cannot manufacture an independent entry when current text returns.
      ASSERT_TRUE(observe_classifier(fixture, cue, 4u, 1300000u));
      const auto reentered = classifier_history(fixture.record());
      EXPECT_LE(reentered[3u], before[3u]);
      for (std::size_t word = 32u; word < before.size(); ++word)
        EXPECT_LE(reentered[word], before[word]);
    }
  }

  TEST(HostSbsOcrGpuTest, LearnedLayoutAssistsAmbiguousCurrentTextWithoutTeachingOrHoldingIt) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> strong(ocr_pixels, 0.0f), ambiguous(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
    paint_rectangle(strong, 240u, 100u, 720u, 115u);
    paint_tilted_line(ambiguous, 400u, 100u, 160u, 10u, 10u);
    // The authored four-degree envelope is deliberately between clearly horizontal and
    // clearly oblique geometry. It has current P evidence, but no initial layout authority.
    ASSERT_TRUE(observe_classifier(fixture, ambiguous, 1u, 1000000u));
    EXPECT_EQ(fixture.record()[4u], 0u);
    ASSERT_TRUE(observe_classifier(fixture, strong, 2u, 1100000u));
    ASSERT_TRUE(observe_classifier(fixture, empty, 3u, 1200000u));
    ASSERT_TRUE(observe_classifier(fixture, strong, 4u, 1300000u));
    ASSERT_TRUE(observe_classifier(fixture, empty, 5u, 1400000u));
    ASSERT_TRUE(observe_classifier(fixture, strong, 6u, 1500000u));
    const auto trusted = classifier_history(fixture.record());
    ASSERT_EQ(trusted[3u], 3u);
    ASSERT_TRUE(observe_classifier(fixture, ambiguous, 7u, 1600000u));
    EXPECT_EQ(fixture.record()[4u], 1u);
    const auto assisted = classifier_history(fixture.record());
    EXPECT_EQ(assisted[3u], trusted[3u]);
    for (std::size_t word = 32u; word < trusted.size(); ++word)
      EXPECT_EQ(assisted[word], trusted[word]);
    ASSERT_TRUE(observe_classifier(fixture, empty, 8u, 1700000u));
    EXPECT_EQ(fixture.record()[4u], 0u);
    EXPECT_EQ(classifier_history(fixture.record())[3u], trusted[3u]);
  }

  TEST(HostSbsOcrGpuTest, QualifiedPriorCannotAuthorizeDivergentOrExcessiveObjectEnvelopes) {
    // Both shapes fit a genuinely learned height/location. The first has opposed moderate
    // edges; the second has one excessively tilted edge. Position alone must not admit either.
    for (const auto shape : {std::array<std::int32_t, 5u> {90, 17, -30, 15, 122},
                             std::array<std::int32_t, 5u> {90, 17, -40, 0, 107}}) {
      SCOPED_TRACE(shape[2u]);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> strong(ocr_pixels, 0.0f), object(ocr_pixels, 0.0f);
      paint_rectangle(strong, 240u, static_cast<std::uint32_t>(shape[4u] - 50),
        720u, static_cast<std::uint32_t>(shape[4u]));
      ASSERT_TRUE(learn_three_layout_entries(fixture, strong));
      const auto trusted = classifier_history(fixture.record());
      ASSERT_EQ(std::bit_cast<float>(trusted[27u]), 50.0f);
      ASSERT_GE(*std::max_element(trusted.begin() + 32u, trusted.end()), 8u);
      paint_envelope_line(object, 400u, shape[0u], 160u, shape[1u], shape[2u], shape[3u]);
      ASSERT_TRUE(observe_classifier(fixture, object, 6u, 1'500'000u));
      ASSERT_EQ(fixture.record()[2u], 1u);
      EXPECT_EQ(fixture.record()[3u], 0u);
      EXPECT_EQ(fixture.record()[4u], 0u);
      for (std::size_t word = v2::subtitle_ocr_raw_box_offset; word < ocr_record_prefix_words; ++word)
        EXPECT_EQ(fixture.record()[word], 0u);
      const auto rejected = classifier_history(fixture.record());
      EXPECT_EQ(rejected[3u], trusted[3u]);
      for (std::size_t word = 27u; word < trusted.size(); ++word)
        EXPECT_EQ(rejected[word], trusted[word]);
    }
  }

  TEST(HostSbsOcrGpuTest, QualifiedPriorPreservesAsymmetricFlatEnvelopeGlyphsWithoutTeaching) {
    for (const auto shape : {std::array<std::int32_t, 5u> {95, 8, 0, 95, 115},
                             std::array<std::int32_t, 5u> {95, 0, -8, 95, 115},
                             std::array<std::int32_t, 5u> {103, -8, 0, 95, 123}}) {
      SCOPED_TRACE(shape[1u]);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> strong(ocr_pixels, 0.0f), glyph(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
      paint_rectangle(strong, 240u, 100u, 720u, 115u);
      ASSERT_TRUE(learn_three_layout_entries(fixture, strong));
      const auto trusted = classifier_history(fixture.record());
      // A real blank makes the next entry eligible to teach if it were independently strong.
      // The glyph must be assisted instead, leaving the qualified location votes untouched.
      ASSERT_TRUE(observe_classifier(fixture, empty, 6u, 1'500'000u));
      paint_envelope_line(glyph, 390u, shape[0u], 180u, 20, shape[1u], shape[2u]);
      ASSERT_TRUE(observe_classifier(fixture, glyph, 7u, 1'600'000u));
      ASSERT_EQ(fixture.record()[4u], 1u);
      EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
        default_ocr_box(390u, static_cast<std::uint32_t>(shape[3u]), 570u,
          static_cast<std::uint32_t>(shape[4u])));
      const auto assisted = classifier_history(fixture.record());
      EXPECT_EQ(assisted[3u], trusted[3u]);
      for (std::size_t word = 27u; word < trusted.size(); ++word)
        EXPECT_EQ(assisted[word], trusted[word]);
      ASSERT_TRUE(observe_classifier(fixture, empty, 8u, 1'700'000u));
      EXPECT_EQ(fixture.record()[4u], 0u);
    }
  }

  TEST(HostSbsOcrGpuTest, AssistedEntryKeepsItsEpisodeForTheFirstTrustedObservation) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> strong(ocr_pixels, 0.0f), glyph(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
    paint_rectangle(strong, 240u, 100u, 720u, 115u);
    paint_envelope_line(glyph, 390u, 95, 180u, 20, 8, 0);
    ASSERT_TRUE(learn_three_layout_entries(fixture, strong));
    const auto trusted = classifier_history(fixture.record());
    ASSERT_TRUE(observe_classifier(fixture, empty, 6u, 1'500'000u));
    // The cue enters through the prior alone. It cannot teach, and it must not spend the
    // independent entry that its own later trusted observation is entitled to.
    ASSERT_TRUE(observe_classifier(fixture, glyph, 7u, 1'600'000u));
    ASSERT_EQ(fixture.record()[4u], 1u);
    const auto assisted = classifier_history(fixture.record());
    EXPECT_EQ(assisted[3u], trusted[3u]);
    EXPECT_EQ(assisted[23u], 0u);
    EXPECT_EQ(assisted[24u], 1u);
    EXPECT_EQ(assisted[25u], 1'500'000u);
    EXPECT_EQ(assisted[26u], 0u);
    for (std::size_t word = 27u; word < trusted.size(); ++word)
      EXPECT_EQ(assisted[word], trusted[word]);
    ASSERT_TRUE(observe_classifier(fixture, strong, 8u, 1'700'000u));
    const auto taught = classifier_history(fixture.record());
    EXPECT_EQ(taught[3u], trusted[3u] + 1u);
    EXPECT_EQ(taught[23u], 1u);
    EXPECT_EQ(taught[24u], 0u);
    EXPECT_EQ(taught[25u], 0u);
    // The same cue remains one episode: later static frames add nothing.
    ASSERT_TRUE(observe_classifier(fixture, strong, 9u, 1'800'000u));
    const auto held = classifier_history(fixture.record());
    EXPECT_EQ(held[3u], taught[3u]);
    for (std::size_t word = 32u; word < taught.size(); ++word)
      EXPECT_EQ(held[word], taught[word]);
  }

  TEST(HostSbsOcrGpuTest, QualifiedPriorKeepsUnknownNarrowGlyphsSeparateFromOrientationVeto) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> strong(ocr_pixels, 0.0f), narrow(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
    paint_rectangle(strong, 240u, 90u, 720u, 120u);
    ASSERT_TRUE(learn_three_layout_entries(fixture, strong));
    const auto trusted = classifier_history(fixture.record());
    ASSERT_TRUE(observe_classifier(fixture, empty, 6u, 1'500'000u));
    paint_rectangle(narrow, 478u, 90u, 482u, 120u);
    ASSERT_TRUE(observe_classifier(fixture, narrow, 7u, 1'600'000u));
    ASSERT_EQ(fixture.record()[4u], 1u);
    EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(478u, 90u, 482u, 120u));
    const auto assisted = classifier_history(fixture.record());
    EXPECT_EQ(assisted[3u], trusted[3u]);
    for (std::size_t word = 27u; word < trusted.size(); ++word)
      EXPECT_EQ(assisted[word], trusted[word]);
  }

  TEST(HostSbsOcrGpuTest, QualifiedPriorRequiresCurrentGroupToSpanReferenceCenter) {
    // Every four-column piece is orientation-unknown, height/location compatible with a
    // genuinely learned prior, and within the existing broad midpoint tolerance. Only the
    // current group containing the center may use history without a current line witness.
    // Detector spans are half-open: contact at the left is included; the right is excluded.
    for (const auto scenario : {std::array<std::uint32_t, 3u> {430u, 434u, 0u},
                               std::array<std::uint32_t, 3u> {526u, 530u, 0u},
                               std::array<std::uint32_t, 3u> {478u, 482u, 1u},
                               std::array<std::uint32_t, 3u> {480u, 484u, 1u},
                               std::array<std::uint32_t, 3u> {476u, 480u, 0u}}) {
      SCOPED_TRACE(scenario[0u]);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> strong(ocr_pixels, 0.0f), narrow(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
      paint_rectangle(strong, 240u, 90u, 720u, 120u);
      ASSERT_TRUE(learn_three_layout_entries(fixture, strong));
      const auto trusted = classifier_history(fixture.record());
      ASSERT_GE(*std::max_element(trusted.begin() + 32u, trusted.end()), 8u);
      paint_rectangle(narrow, scenario[0u], 90u, scenario[1u], 120u);
      ASSERT_TRUE(observe_classifier(fixture, narrow, 6u, 1'500'000u));
      ASSERT_EQ(fixture.record()[2u], 1u);
      ASSERT_EQ(fixture.record()[4u], scenario[2u]);
      if (scenario[2u] != 0u) {
        EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
          default_ocr_box(scenario[0u], 90u, scenario[1u], 120u));
      }
      const auto observed = classifier_history(fixture.record());
      EXPECT_EQ(observed[3u], trusted[3u]);
      for (std::size_t word = 27u; word < trusted.size(); ++word)
        EXPECT_EQ(observed[word], trusted[word]);
      ASSERT_TRUE(observe_classifier(fixture, empty, 7u, 1'600'000u));
      EXPECT_EQ(fixture.record()[4u], 0u);
    }
  }

  TEST(HostSbsOcrGpuTest, QualifiedPriorUsesSignedCurrentReferenceAndHalfOpenEdges) {
    // References can be offset within or outside the current analysis crop. Learn within
    // that same domain, then distinguish actual reference containment from midpoint slack.
    for (const auto scenario : {std::array<std::int32_t, 6u> {960, 238, 242, 100, 380, 1},
                               std::array<std::int32_t, 6u> {960, 270, 274, 100, 380, 0},
                               std::array<std::int32_t, 6u> {960, 240, 244, 100, 380, 1},
                               std::array<std::int32_t, 6u> {960, 236, 240, 100, 380, 0},
                               std::array<std::int32_t, 6u> {0, 0, 4, 0, 12, 1},
                               std::array<std::int32_t, 6u> {-4, 0, 4, 0, 12, 0}}) {
      SCOPED_TRACE(scenario[0u]);
      SCOPED_TRACE(scenario[1u]);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      const auto observe = [&](const std::vector<float> &probability,
                               const std::uint64_t frame, const std::uint64_t clock) {
        return fixture.run_boxes(probability, frame, 77u, source_width, source_height,
          field_width, field_height, source_crop_top, source_crop_height, roi_top, roi_bottom,
          {}, scenario[0u], source_width, std::nullopt, clock);
      };
      std::vector<float> strong(ocr_pixels, 0.0f), narrow(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
      paint_rectangle(strong, static_cast<std::uint32_t>(scenario[3u]), 90u,
        static_cast<std::uint32_t>(scenario[4u]), 120u);
      std::uint64_t frame = 1u;
      std::uint64_t clock = 1'000'000u;
      if (scenario[0u] < 0) {
        // A reference left of the source is valid geometry, but no ordinary
        // nonnegative tight group can contain it or establish a layout prior.
        const auto expect_empty_history = [&] {
          const auto history = classifier_history(fixture.record());
          EXPECT_EQ(history[3u], 0u);
          EXPECT_EQ(*std::max_element(history.begin() + 32u, history.end()), 0u);
        };
        for (std::uint32_t episode = 0u; episode < 3u; ++episode) {
          if (episode != 0u) {
            ASSERT_TRUE(observe(empty, frame++, clock));
            clock += 100'000u;
          }
          ASSERT_TRUE(observe(strong, frame++, clock));
          ASSERT_EQ(fixture.record()[2u], 1u);
          ASSERT_EQ(fixture.record()[3u], 0u);
          ASSERT_EQ(fixture.record()[4u], 0u);
          expect_empty_history();
          clock += 100'000u;
        }
        paint_rectangle(narrow, static_cast<std::uint32_t>(scenario[1u]), 90u,
          static_cast<std::uint32_t>(scenario[2u]), 120u);
        ASSERT_TRUE(observe(narrow, frame++, clock));
        ASSERT_EQ(fixture.record()[2u], 1u);
        ASSERT_EQ(fixture.record()[3u], 0u);
        ASSERT_EQ(fixture.record()[4u], 0u);
        expect_empty_history();
        continue;
      }
      for (std::uint32_t episode = 0u; episode < 3u; ++episode) {
        if (episode != 0u) {
          ASSERT_TRUE(observe(empty, frame++, clock));
          clock += 100'000u;
        }
        ASSERT_TRUE(observe(strong, frame++, clock));
        ASSERT_EQ(fixture.record()[4u], 1u);
        clock += 100'000u;
      }
      const auto trusted = classifier_history(fixture.record());
      ASSERT_EQ(trusted[3u], 3u);
      ASSERT_GE(*std::max_element(trusted.begin() + 32u, trusted.end()), 8u);
      paint_rectangle(narrow, static_cast<std::uint32_t>(scenario[1u]), 90u,
        static_cast<std::uint32_t>(scenario[2u]), 120u);
      ASSERT_TRUE(observe(narrow, frame++, clock));
      ASSERT_EQ(fixture.record()[2u], 1u);
      ASSERT_EQ(fixture.record()[4u], static_cast<std::uint32_t>(scenario[5u]));
      const auto observed = classifier_history(fixture.record());
      EXPECT_EQ(observed[3u], trusted[3u]);
      for (std::size_t word = 27u; word < trusted.size(); ++word)
        EXPECT_EQ(observed[word], trusted[word]);
    }
  }

  TEST(HostSbsOcrGpuTest, SmallLearnedHeightCannotAdmitOffAxisFragmentOrDelayCenteredEntry) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> strong(ocr_pixels, 0.0f), fragment(ocr_pixels, 0.0f), centered(ocr_pixels, 0.0f);
    paint_rectangle(strong, 240u, 112u, 720u, 118u);
    ASSERT_TRUE(learn_three_layout_entries(fixture, strong));
    const auto trusted = classifier_history(fixture.record());
    ASSERT_EQ(std::bit_cast<float>(trusted[27u]), 6.0f);
    // This three-row unknown fits the learned six-row height and baseline, but its current
    // group is entirely left of the center. Rejection must clear it in this observation.
    paint_rectangle(fragment, 428u, 115u, 432u, 118u);
    ASSERT_TRUE(observe_classifier(fixture, fragment, 6u, 1'500'000u));
    EXPECT_EQ(fixture.record()[4u], 0u);
    paint_rectangle(centered, 478u, 115u, 482u, 118u);
    ASSERT_TRUE(observe_classifier(fixture, centered, 7u, 1'533'333u));
    ASSERT_EQ(fixture.record()[4u], 1u);
    EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(478u, 115u, 482u, 118u));
    const auto observed = classifier_history(fixture.record());
    EXPECT_EQ(observed[3u], trusted[3u]);
    for (std::size_t word = 27u; word < trusted.size(); ++word)
      EXPECT_EQ(observed[word], trusted[word]);
  }

  TEST(HostSbsOcrGpuTest, UnknownPieceOnCurrentHorizontalLineDoesNotNeedItsOwnCenterCrossing) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> cue(ocr_pixels, 0.0f);
    paint_rectangle(cue, 300u, 90u, 600u, 120u);
    // More than four inactive cells prevent an ordinary merge, while the fixed-height
    // current line group still includes this unknown word. Equal height prevents the
    // short-contained-punctuation exception from hiding a per-piece center regression.
    paint_rectangle(cue, 640u, 90u, 644u, 120u);
    ASSERT_TRUE(observe_classifier(fixture, cue, 1u, 1'000'000u));
    ASSERT_EQ(fixture.record()[4u], 2u);
    EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
      default_ocr_box(300u, 90u, 600u, 120u));
    EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset + v2::subtitle_ocr_box_word_count),
      default_ocr_box(640u, 90u, 644u, 120u));
    // There is only one independent entry, below the prior-assistance threshold.
    const auto history = classifier_history(fixture.record());
    EXPECT_EQ(history[3u], 1u);
    EXPECT_LT(*std::max_element(history.begin() + 32u, history.end()), 8u);
  }

  TEST(HostSbsOcrGpuTest, UnknownPieceCannotPushACenteredHorizontalLineOffCenter) {
    // The caption is exactly centered. A same-height unknown blob inside the unchanged
    // four-height line gap makes the group union about 3.9% off-center, beyond the 2% rule.
    // The caption's own horizontal union still qualifies; the blob is not published with it.
    for (const auto with_blob : {false, true}) {
      SCOPED_TRACE(with_blob);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> cue(ocr_pixels, 0.0f);
      paint_rectangle(cue, 330u, 90u, 630u, 120u);
      if (with_blob) paint_rectangle(cue, 700u, 90u, 704u, 120u);
      ASSERT_TRUE(observe_classifier(fixture, cue, 1u, 1'000'000u));
      ASSERT_EQ(fixture.record()[4u], 1u);
      EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
        default_ocr_box(330u, 90u, 630u, 120u));
      EXPECT_EQ(classifier_history(fixture.record())[3u], 1u);
    }
  }

  TEST(HostSbsOcrGpuTest, ShortLineWithOneRowEnvelopeStepIsHorizontal) {
    // A 30-pixel centered line whose right half sits one detector row lower. The one-row step
    // is pure quantization, yet over this width its fitted slope exceeds the two-degree bound.
    // A two-row step on the same short line is real tilt and stays outside independent text.
    for (const auto step : {1u, 2u}) {
      SCOPED_TRACE(step);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> cue(ocr_pixels, 0.0f);
      paint_rectangle(cue, 465u, 100u, 480u, 115u);
      paint_rectangle(cue, 480u, 100u + step, 495u, 115u + step);
      ASSERT_TRUE(observe_classifier(fixture, cue, 1u, 1'000'000u));
      EXPECT_EQ(fixture.record()[4u], step == 1u ? 1u : 0u);
      EXPECT_EQ(classifier_history(fixture.record())[3u], step == 1u ? 1u : 0u);
    }
  }

  TEST(HostSbsOcrGpuTest, ExplicitDomainAndAnalysisGenerationChangesClearTheLayoutPrior) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> cue(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
    paint_rectangle(cue, 240u, 100u, 720u, 115u);
    ASSERT_TRUE(observe_classifier(fixture, cue, 1u, 1000000u));
    ASSERT_EQ(classifier_history(fixture.record())[3u], 1u);
    ASSERT_TRUE(observe_classifier(fixture, empty, 2u, 1100000u, true));
    auto cleared = classifier_history(fixture.record());
    EXPECT_EQ(cleared[3u], 0u);
    for (std::size_t word = 32u; word < cleared.size(); ++word) EXPECT_EQ(cleared[word], 0u);
    ASSERT_TRUE(observe_classifier(fixture, cue, 3u, 1200000u));
    ASSERT_EQ(classifier_history(fixture.record())[3u], 1u);
    ASSERT_TRUE(observe_classifier(fixture, empty, 4u, 1300000u, false, 78u));
    cleared = classifier_history(fixture.record());
    EXPECT_EQ(cleared[3u], 0u);
    EXPECT_EQ(cleared[8u], 78u);
    for (std::size_t word = 32u; word < cleared.size(); ++word) EXPECT_EQ(cleared[word], 0u);
  }

  TEST(HostSbsOcrGpuTest, ChangingVideoToScreenLineReferenceClearsTheLayoutPrior) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> cue(ocr_pixels, 0.0f), empty(ocr_pixels, 0.0f);
    paint_rectangle(cue, 240u, 100u, 720u, 115u);
    ASSERT_TRUE(observe_classifier(fixture, cue, 1u, 1000000u));
    ASSERT_EQ(classifier_history(fixture.record())[3u], 1u);
    // Same analysis pixels and generation, but an explicit capture reference uses the full
    // screen's center/lower quarter. A learned video-local location cannot cross that change.
    ASSERT_TRUE(fixture.run_boxes(empty, 2u, 77u, source_width, source_height,
      field_width, field_height, source_crop_top, source_crop_height, roi_top, roi_bottom,
      {}, 3840, 3840u, 540, 1100000u));
    const auto changed = classifier_history(fixture.record());
    EXPECT_EQ(changed[3u], 0u);
    EXPECT_EQ(changed[20u], 3840u);
    EXPECT_EQ(changed[21u], 3840u);
    EXPECT_EQ(changed[22u], 540u);
    for (std::size_t word = 32u; word < changed.size(); ++word) EXPECT_EQ(changed[word], 0u);
  }

  TEST(HostSbsOcrGpuTest, LeadingAndTrailingShortPunctuationPreserveTheCurrentLine) {
    for (const auto leading : {true, false}) {
      SCOPED_TRACE(leading);
      ocr_warp_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> cue(ocr_pixels, 0.0f);
      paint_rectangle(cue, 300u, 85u, 660u, 100u);
      paint_rectangle(cue, leading ? 280u : 672u, 94u, leading ? 283u : 675u, 97u);
      ASSERT_TRUE(fixture.run_boxes(cue, 1u, 77u));
      ASSERT_EQ(fixture.record()[4u], 1u);
      EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
        default_ocr_box(leading ? 280u : 300u, 85u, leading ? 660u : 675u, 100u));
      EXPECT_EQ(fixture.record()[v2::subtitle_ocr_raw_box_offset + 6u], 2u);
    }
  }

  TEST(HostSbsOcrGpuTest, SteepSceneMemberCannotBorrowThePunctuationAttachmentRule) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> dialogue(ocr_pixels, 0.0f);
    paint_rectangle(dialogue, 300u, 100u, 660u, 115u);
    ASSERT_TRUE(fixture.run_boxes(dialogue, 1u, 77u));
    ASSERT_EQ(fixture.record()[4u], 1u);
    const auto expected = read_native_pair(fixture.record(), 0u);
    auto nearby_scene = dialogue;
    // This fully observed steep piece is within the dialogue's vertical span and only twelve
    // pixels beyond its right edge. Its geometry is scene-like, rather than unknown short
    // punctuation; adjacency must neither attach it nor change the credible receiver.
    paint_tilted_line(nearby_scene, 672u, 102u, 16u, 3u, 10u);
    ASSERT_TRUE(fixture.run_boxes(nearby_scene, 40u, 77u));
    expect_retained_native_pairs(fixture.record(), {expected});
  }

  TEST(HostSbsOcrGpuTest, AmbiguousSceneMemberNeedsItsOwnQualifiedPriorBeforeAttachment) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> dialogue(ocr_pixels, 0.0f);
    paint_rectangle(dialogue, 300u, 90u, 600u, 110u);
    // Center the unchanged receiver in a legal 5120-wide screen reference.
    // Its 51.2-detector-pixel tolerance also contains an incorrectly merged
    // receiver+scene union, so centering cannot hide the attachment regression.
    ASSERT_TRUE(fixture.run_boxes_in_screen(dialogue, 1u, 77u, 5120u, 1660u));
    ASSERT_EQ(fixture.record()[4u], 1u);
    const auto expected = read_native_pair(fixture.record(), 0u);
    auto nearby_scene = dialogue;
    // A complete four-degree line is not unknown punctuation, even when contained in the
    // current strong line's vertical span. One newly learned entry cannot qualify it.
    paint_tilted_line(nearby_scene, 612u, 95u, 80u, 5u, 6u);
    ASSERT_TRUE(fixture.run_boxes_in_screen(nearby_scene, 40u, 77u, 5120u, 1660u));
    expect_retained_native_pairs(fixture.record(), {expected});
  }

  TEST(HostSbsOcrGpuTest, MoreThanOneHundredTwentyLocalCandidatesDoNotAbstainGlobally) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> crowded(ocr_pixels, 0.0f);
    // Twenty columns times eight independent local rows, joined into one global row band by
    // a remote strip. Even if that strip absorbs its closest column, 152 other local pieces
    // exceed the retired 120-slot queue. A credible dialogue below must survive the pressure.
    for (std::uint32_t row = 0u; row < 8u; ++row)
      for (std::uint32_t column = 0u; column < 20u; ++column)
        paint_rectangle(crowded, 8u + column * 48u, 35u + row * 10u,
          16u + column * 48u, 40u + row * 10u);
    paint_rectangle(crowded, 0u, 35u, 3u, 115u);
    paint_rectangle(crowded, 240u, 130u, 720u, 145u);
    ASSERT_TRUE(fixture.run_boxes(crowded, 1u, 77u));
    EXPECT_EQ(fixture.record()[2u], 1u);
    ASSERT_GT(fixture.record()[4u], 0u);
    bool found_dialogue = false;
    for (std::uint32_t slot = 0u; slot < fixture.record()[4u]; ++slot)
      found_dialogue |= read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset +
        slot * v2::subtitle_ocr_box_word_count) == default_ocr_box(240u, 130u, 720u, 145u);
    EXPECT_TRUE(found_dialogue);
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

  TEST(HostSbsOcrGpuTest, LineCenterHasAnInclusiveConfiguredWidthTolerance) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    constexpr auto boundary = configured_center_boundary(source_width);
    constexpr std::uint32_t patch_width = 256u;
    constexpr auto centered_left = (ocr_width - patch_width) / 2u;
    static_assert(boundary.reference_width <= v2::subtitle_contour_source_max_extent);
    static_assert(boundary.detector_offset > 0u && boundary.detector_offset < centered_left);
    const auto delta = static_cast<std::int32_t>(boundary.detector_offset);
    for (const auto offset : {-delta, delta, -delta - 1, delta + 1}) {
      SCOPED_TRACE(offset);
      const auto left = static_cast<std::uint32_t>(static_cast<std::int32_t>(centered_left) + offset);
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, left, 85u, left + patch_width, 100u);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, source_width, source_height,
        field_width, field_height, source_crop_top, source_crop_height, roi_top, roi_bottom,
        {}, static_cast<std::int32_t>(source_width), boundary.reference_width));
      EXPECT_EQ(fixture.record()[2u], 1u);
      const auto expected_count = std::abs(offset) == delta ? 1u : 0u;
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
    constexpr auto boundary = configured_center_boundary(video_w);
    constexpr std::uint32_t patch_width = 256u;
    constexpr auto centered_left = (ocr_width - patch_width) / 2u;
    static_assert(boundary.reference_width <= v2::subtitle_contour_source_max_extent);
    static_assert(boundary.detector_offset > 0u && boundary.detector_offset < centered_left);
    const auto delta = static_cast<std::int32_t>(boundary.detector_offset);
    for (const auto offset : {-delta, delta, -delta - 1, delta + 1}) {
      SCOPED_TRACE(offset);
      const auto left = static_cast<std::uint32_t>(static_cast<std::int32_t>(centered_left) + offset);
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_rectangle(probability, left, 85u, left + patch_width, 100u);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, video_w, video_h,
        field_width, field_height, crop_top, crop_h,
        geometry.roi_top, geometry.roi_bottom, content,
        static_cast<std::int32_t>(video_w), boundary.reference_width));
      EXPECT_EQ(fixture.record()[2u], 1u);
      const auto expected_count = std::abs(offset) == delta ? 1u : 0u;
      EXPECT_EQ(fixture.record()[3u], expected_count);
      EXPECT_EQ(fixture.record()[4u], expected_count);
      // Both inclusive configured edges are exact integer-rational equalities,
      // even though video_w is not divisible by the detector width. The next
      // detector column on either side is outside; do not replace this with FP32.
      if (expected_count == 0u) {
        for (std::size_t word = v2::subtitle_ocr_raw_box_offset;
             word < ocr_record_prefix_words; ++word)
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
           word < ocr_record_prefix_words; ++word)
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
    const auto left_core = read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset);
    const auto right_core = read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset +
      v2::subtitle_ocr_box_word_count);
    EXPECT_EQ(left_core, default_ocr_box(300u, 85u, 440u, 100u));
    EXPECT_EQ(right_core, default_ocr_box(500u, 85u, 660u, 100u));
    EXPECT_LT(left_core[2u], right_core[0u]);
    EXPECT_LT(left_core[2u], field_width / 2u);
    EXPECT_GT(right_core[0u], field_width / 2u);
    // Separate producer pairs keep their own DB unclip and ordinary search guards. The
    // four-cell search guards may overlap without merging the disjoint tight fragments.
    EXPECT_EQ(left_cover, default_ordinary_cover(290u, 75u, 450u, 110u, 15u));
    EXPECT_EQ(right_cover, default_ordinary_cover(490u, 75u, 670u, 110u, 15u));
    EXPECT_NE(left_cover, right_cover);
    EXPECT_GT(left_cover[2u], right_cover[0u]);
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

  TEST(HostSbsOcrGpuTest, SeparatedStrongFragmentsRequireTheirGroupToBeCentered) {
    // Relative geometry mirrors a pair of scene detections: both are individually
    // off-center, share a horizontal band, and are close enough for the unchanged
    // 4H line rule. Their union spans the reference but is about 2.92% off-center.
    // A common translation centers the twin without changing its split support.
    const std::vector<ocr_candidate_patch_t> off_center {
      {{268u, 45u, 449u, 104u}},
      {{548u, 45u, 748u, 92u}},
    };
    auto centered = off_center;
    for (auto &patch : centered) {
      patch.rectangle[0u] -= 28u;
      patch.rectangle[2u] -= 28u;
    }
    ocr_warp_fixture_t expected_fixture;
    std::string error;
    ASSERT_TRUE(expected_fixture.initialize(error, false, true, false)) << error;
    std::vector<ocr_native_pair_t> expected_pairs;
    ASSERT_TRUE(capture_native_pairs(expected_fixture, centered, expected_pairs));
    ASSERT_EQ(expected_pairs.size(), 2u);
    for (const auto &pair : expected_pairs) EXPECT_EQ(pair[5u], 0u); // Ordinary, not a ribbon.
    for (const auto exact_center : {false, true}) {
      SCOPED_TRACE(exact_center);
      ocr_warp_fixture_t fixture; // A fresh owner: neither case borrows learned layout.
      ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
      std::vector<float> probability(ocr_pixels, 0.0f);
      paint_candidate_patches(probability, exact_center ? centered : off_center);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
      expect_retained_native_pairs(fixture.record(), exact_center ? expected_pairs :
        std::vector<ocr_native_pair_t> {});
      EXPECT_EQ(classifier_history(fixture.record())[3u], exact_center ? 1u : 0u);
    }
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
         word < ocr_record_prefix_words; ++word)
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
             word < ocr_record_prefix_words; ++word)
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
             word < ocr_record_prefix_words; ++word)
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
    auto patches = equal_area_ocr_patches(9u);
    // Recenter the complete nine-member line, preserving every core's dimensions,
    // probability, discovery order and equal-area capacity tie.
    for (auto &patch : patches) {
      patch.rectangle[0u] += 80u;
      patch.rectangle[2u] += 80u;
    }
    std::vector<ocr_native_pair_t> pairs;
    ASSERT_TRUE(capture_native_pairs(fixture, patches, pairs));
    pairs.resize(8u);
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_candidate_patches(probability, patches);
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u));
    // All nine cores span 80..880 and center exactly at 480. The earlier retained
    // eight span only 80..784, whose center432 is outside the configured 2% margin.
    // Capacity selection before line filtering would therefore lose this line.
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
    // A diagnostic 1920-wide analysis at x1600 in a 3840-wide screen sees the true
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

  TEST(HostSbsOcrGpuTest, NativePlayerRoiKeepsCenteredCaptionsAwayFromCaptureCenter) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, false)) << error;
    std::vector<float> probability(ocr_pixels, 0.0f);
    paint_rectangle(probability, 400u, 85u, 560u, 100u);
    constexpr std::array<std::array<std::uint32_t, 2u>, 3u> positions {{
      {100u, 100u}, {1800u, 800u}, {1920u, 1080u},
    }};
    for (const auto position : positions) {
      SCOPED_TRACE(std::to_string(position[0u]) + "/" + std::to_string(position[1u]));
      const models::depth_input_region_t region {
        .source_width = 3840u,
        .source_height = 2160u,
        .left = position[0u],
        .top = position[1u],
        .right = position[0u] + source_width,
        .bottom = position[1u] + source_height,
        .tensor_content = {0u, 0u, field_width, field_height},
        .analysis_generation = 77u,
        .authority = models::depth_analysis_authority_e::foreground_client,
      };
      ASSERT_TRUE(region.valid());
      const auto reference = models::subtitle_ocr_line_reference(
        region.width(), region.height(), region.source_width, region.source_height,
        region.left, region.top, region.is_video_region());
      ASSERT_TRUE(reference);
      ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, source_width, source_height,
        field_width, field_height, source_crop_top, source_crop_height,
        roi_top, roi_bottom, {}, reference->center_twice_source, reference->width,
        reference->search_top_source));
      EXPECT_EQ(fixture.record()[2u], 1u);
      ASSERT_EQ(fixture.record()[3u], 1u);
      EXPECT_EQ(read_box(fixture.record(), v2::subtitle_ocr_raw_box_offset),
        default_ocr_box(400u, 85u, 560u, 100u));
    }
    // Under the former whole-capture reference, this same native caption was discarded.
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 41u, 77u, 3840u, 100u, 2160u, 100u));
    expect_retained_native_pairs(fixture.record(), {}, 41u);
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
    // A diagnostic crop at (210,1700), size 1500x400, fits a 1920x2160 screen. The screen's
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
    // Match the reference to the complete nine-core union center400; capacity remains the test.
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, 1120u));
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
    // Match the reference to the complete union center448; capacity remains the test.
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, 1024u));
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
    // Match the reference to the complete union center448; capacity remains the test.
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, 1024u));
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
    // Match the reference to the larger row center456; earlier row center448 remains eligible; capacity remains the test.
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 3840u, 1008u));
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
    // One legal screen reference keeps BOTH original groups eligible: ordinary
    // center352 and ribbon center435.5 are within 51.2 pixels of reference393.5.
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 5120u, 1773u));
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
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 40u, 77u, 5120u, 1773u));
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
      for (std::size_t word = v2::subtitle_ocr_raw_box_offset; word < ocr_record_prefix_words; ++word)
        EXPECT_EQ(fixture.record()[word], 0u);
    }
    probability[0u] = 0.0f;
    ASSERT_TRUE(fixture.run_boxes(probability, 40u, 77u, source_width, source_height,
                                  field_width, field_height, source_crop_top + 1u));
    EXPECT_EQ(fixture.record()[2u], 0u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);
    for (std::size_t word = v2::subtitle_ocr_raw_box_offset; word < ocr_record_prefix_words; ++word)
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
      default_ordinary_cover(280u, 74u, 681u, 108u, 14u)
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
      default_ordinary_cover(309u, 74u, 646u, 111u, 15u)
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
      default_ordinary_cover(309u, 74u, 686u, 111u, 15u)
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

    // Keep the original asymmetric seven-island topology and exact edge data;
    // a matched reference at its center435.5 isolates ribbon/bottom classification.
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 30u, 77u, 3840u, 1049u));
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
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 31u, 77u, 3840u, 1049u));
    EXPECT_EQ(
      fixture.record()[v2::subtitle_ocr_raw_box_offset + 5u],
      v2::subtitle_ocr_box_flag_ribbon
    );
    std::fill(probability.begin(), probability.end(), 0.0f);
    for (const auto &island : islands) {
      paint_rectangle(probability, island[0u], 120u, island[1u], 152u);
    }
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 32u, 77u, 3840u, 1049u));
    EXPECT_EQ(fixture.record()[v2::subtitle_ocr_raw_box_offset + 5u], 0u);

    // The same segmented topology away from the bottom is a normal subtitle candidate. Position
    // is part of the ribbon conjunction; segmentation alone must not change subtitle semantics.
    std::fill(probability.begin(), probability.end(), 0.0f);
    for (const auto &island : islands) {
      paint_rectangle(probability, island[0u], 90u, island[1u], 100u);
    }
    ASSERT_TRUE(fixture.run_boxes_in_screen(probability, 33u, 77u, 3840u, 1049u));
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

    // Keep one legal matched reference throughout the complete lifecycle. Group
    // center540 and later single-line center480 both lie within38.4 detector pixels
    // of reference510; EMPTY/fault/restore must not borrow a reference-context reset.
    ASSERT_TRUE(fixture.run_boxes_in_screen(line_and_logo, 10u, generation, 3840u, 900u));
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
      default_ordinary_cover(753u, 92u, 847u, 116u, 10u)
    );

    // Every producer-admitted current pair participates immediately. The fixture authors zero
    // budgets only in its tight cores; wider covers authorize search, not blanket protection.
    // These authored budgets exercise conditioning/publication, not native RGB glyph recall.
    fixture.reset_locator();
    ASSERT_TRUE(fixture.dispatch_locator(10u, generation, true));
    ASSERT_EQ(fixture.state().size(), v2::subtitle_plane_state_word_count);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_flags_word], v2::subtitle_plane_current_valid_flag);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 3u);
    EXPECT_FALSE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.smoothing_enabled(), 1u);
    EXPECT_EQ(fixture.weight_at(300u, 390u), 1.0f);
    EXPECT_EQ(fixture.weight_at(650u, 390u), 1.0f);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(650u, 390u)),
      fixture.state()[v2::subtitle_plane_adaptive_offset + 3u]);

    ASSERT_TRUE(fixture.run_boxes_in_screen(line_and_logo, 11u, generation, 3840u, 900u));
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
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(300u, 390u)),
      fixture.state()[v2::subtitle_plane_adaptive_offset + 3u]);
    EXPECT_EQ(fixture.smoothing_enabled(), 1u);
    EXPECT_EQ(fixture.weight_at(300u, 390u), 1.0f);
    // This point's authored completion budget already exceeds its immutable Base/UI gap;
    // authored core cells above remain exact M=1, while this outside-core background node
    // needs no UI influence. The proposal preserves Base bits at the distant corner as well.
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.weight_at(180u, 395u)), 0u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.weight_at(0u, 0u)), 0u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(0u, 0u)),
      std::bit_cast<std::uint32_t>(0.03f));

    // A fully empty probability map is an authoritative same-frame absence. It removes the
    // current cover union immediately and the conditioned field becomes bit-exact Base.
    std::vector<float> empty(ocr_pixels, 0.0f);
    ASSERT_TRUE(fixture.run_boxes_in_screen(empty, 12u, generation, 3840u, 900u));
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
    ASSERT_TRUE(fixture.run_boxes_in_screen(line_only, 13u, generation, 3840u, 900u));
    ASSERT_TRUE(fixture.dispatch_locator(13u, generation));
    ASSERT_TRUE(fixture.run_boxes_in_screen(line_only, 14u, generation, 3840u, 900u));
    ASSERT_TRUE(fixture.dispatch_locator(14u, generation));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 1u);

    auto nonfinite = line_only;
    nonfinite[0u] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(fixture.run_boxes_in_screen(nonfinite, 15u, generation, 3840u, 900u));
    EXPECT_EQ(fixture.record()[2u], 0u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);
    ASSERT_TRUE(fixture.dispatch_locator(15u, generation));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.smoothing_enabled(), 0u);

    // A valid OCR8 record whose frame identity does not match b2 is equally non-authoritative.
    ASSERT_TRUE(fixture.run_boxes_in_screen(line_only, 16u, generation, 3840u, 900u));
    ASSERT_TRUE(fixture.dispatch_locator(16u, generation));
    ASSERT_TRUE(fixture.run_boxes_in_screen(line_only, 17u, generation, 3840u, 900u));
    ASSERT_TRUE(fixture.dispatch_locator(17u, generation));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 1u);
    ASSERT_TRUE(fixture.run_boxes_in_screen(line_only, 18u, generation, 3840u, 900u));
    ASSERT_TRUE(fixture.dispatch_locator(19u, generation));
    EXPECT_EQ(fixture.state()[0u], v2::subtitle_plane_state_schema);
    EXPECT_EQ(fixture.state()[1u], v2::subtitle_plane_state_tag);
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.smoothing_enabled(), 0u);

    // Analysis generation and explicit current-record admission are independent authority gates.
    ASSERT_TRUE(fixture.run_boxes_in_screen(line_only, 20u, generation, 3840u, 900u));
    ASSERT_TRUE(fixture.dispatch_locator(20u, generation + 1u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_flags_word], 0u);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_current_slot_mask_word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.smoothing_enabled(), 0u);
    ASSERT_TRUE(fixture.run_boxes_in_screen(line_only, 21u, generation, 3840u, 900u));
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
