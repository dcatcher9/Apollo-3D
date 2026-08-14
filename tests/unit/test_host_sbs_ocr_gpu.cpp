/**
 * @file tests/unit/test_host_sbs_ocr_gpu.cpp
 * @brief End-to-end D3D11 WARP coverage for the production OCR preprocess/box producer.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/generated/depth_coordinate_v2_contract.h"

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
  constexpr std::uint32_t ocr_cell_words = 8u;
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
    std::array<std::uint32_t, 3u> reserved;
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

  struct v2_constants_t {
    float raw_coordinate_scale;
    float collapse_abs_epsilon;
    float far_tau;
    float near_log_tau;
    float requested_gain;
    float max_horizontal_slope;
    float direct_container_limit;
    float convergence_curve_default;
  };

  struct subtitle_constants_t {
    std::array<std::uint32_t, 4u> field;
    std::array<std::uint32_t, 4u> source;
    std::array<std::uint32_t, 4u> frame;
    std::array<std::uint32_t, 4u> tensor_content;
  };

  static_assert(sizeof(rgba_t) == 4u * sizeof(float));
  static_assert(sizeof(preprocess_constants_t) == 32u);
  static_assert(sizeof(ocr_resolve_constants_t) == 64u);
  static_assert(sizeof(depth_constants_t) == 64u);
  static_assert(sizeof(v2_constants_t) == 32u);
  static_assert(sizeof(subtitle_constants_t) == 64u);
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
      std::vector<float> &output
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
        width,
        height,
        crop_top,
        crop_height,
        0u,
        {}
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
      std::array<std::uint32_t, 4u> tensor_content = {}
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
      context_->Dispatch(ocr_grid_width, ocr_height, 1u);
      unbind();

      context_->CSSetShader(boxes_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 1u, boxes_cb_.GetAddressOf());
      context_->CSSetShaderResources(1u, 1u, cell_stats_srv_.GetAddressOf());
      context_->CSSetUnorderedAccessViews(1u, 1u, record_uav_.GetAddressOf(), nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind();
      return read_buffer(device_.Get(), context_.Get(), record_buffer_.Get(), record_);
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
      };
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
        nullptr,
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
      context_->CSSetUnorderedAccessViews(2u, 1u, state_uav_.GetAddressOf(), nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind();

      // unbind() deliberately clears all compute-stage bindings. Rebind the shared b0/b1/b2
      // contract before condition_main; otherwise target_w is read as zero and every thread
      // returns without writing, leaving the output texture's previous contents in place.
      context_->CSSetShader(condition_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(
        0u,
        static_cast<UINT>(constant_buffers.size()),
        constant_buffers.data()
      );
      std::array<ID3D11ShaderResourceView *, 4u> condition_srvs {
        nullptr,
        nullptr,
        base_srv_.Get(),
        state_srv_.Get()
      };
      context_->CSSetShaderResources(
        0u,
        static_cast<UINT>(condition_srvs.size()),
        condition_srvs.data()
      );
      context_->CSSetUnorderedAccessViews(3u, 1u, output_uav_.GetAddressOf(), nullptr);
      context_->Dispatch((field_width + 15u) / 16u, (field_height + 15u) / 16u, 1u);
      unbind();

      return read_buffer(device_.Get(), context_.Get(), state_buffer_.Get(), state_) &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    void reset_locator() {
      const UINT zero[4] = {};
      context_->ClearUnorderedAccessViewUint(state_uav_.Get(), zero);
      state_.clear();
      output_.clear();
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

  private:
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
      std::vector<std::uint32_t> zeros(v2::subtitle_locator_state_word_count, 0u);
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
      const v2_constants_t coordinate_constants {
        0.04f,
        0.0001f,
        0.0f,
        0.0f,
        1.0f,
        0.5f,
        0.04f,
        0.0f
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
               &coordinate_constants,
               sizeof(coordinate_constants),
               v2_cb_
             ) &&
             create_constant_buffer(
               device_.Get(),
               &subtitle_constants,
               sizeof(subtitle_constants),
               subtitle_cb_
             );
    }

    void unbind() {
      std::array<ID3D11ShaderResourceView *, 8u> null_srvs {};
      std::array<ID3D11UnorderedAccessView *, 4u> null_uavs {};
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
    ComPtr<ID3D11Buffer> cut_buffer_;
    ComPtr<ID3D11ShaderResourceView> cut_srv_;
    ComPtr<ID3D11Texture2D> base_texture_;
    ComPtr<ID3D11ShaderResourceView> base_srv_;
    ComPtr<ID3D11Texture2D> output_texture_;
    ComPtr<ID3D11UnorderedAccessView> output_uav_;
    ComPtr<ID3D11Buffer> depth_cb_;
    ComPtr<ID3D11Buffer> v2_cb_;
    ComPtr<ID3D11Buffer> subtitle_cb_;
    std::vector<std::uint32_t> record_;
    std::vector<std::uint32_t> state_;
    std::vector<float> base_;
    std::vector<float> output_;
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
      (std::array<std::uint32_t, 4u> {192u, 385u, 578u, 393u})
    );
    EXPECT_EQ(
      read_box(one, v2::subtitle_ocr_final_box_offset),
      (std::array<std::uint32_t, 4u> {187u, 380u, 583u, 398u})
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
      (std::array<std::uint32_t, 4u> {187u, 380u, 583u, 398u})
    );
    EXPECT_EQ(
      read_box(
        two,
        v2::subtitle_ocr_final_box_offset + v2::subtitle_ocr_box_word_count
      ),
      (std::array<std::uint32_t, 4u> {235u, 397u, 535u, 414u})
    );
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
      (std::array<std::uint32_t, 4u> {232u, 372u, 539u, 385u})
    );
    EXPECT_EQ(
      read_box(joined, v2::subtitle_ocr_final_box_offset),
      (std::array<std::uint32_t, 4u> {224u, 364u, 547u, 393u})
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
      (std::array<std::uint32_t, 4u> {256u, 373u, 510u, 386u})
    );
    EXPECT_EQ(
      read_box(remote_speck, v2::subtitle_ocr_final_box_offset),
      (std::array<std::uint32_t, 4u> {247u, 364u, 519u, 395u})
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
      (std::array<std::uint32_t, 4u> {256u, 373u, 510u, 386u})
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
      (std::array<std::uint32_t, 4u> {256u, 373u, 542u, 386u})
    );
    EXPECT_EQ(
      read_box(punctuation, v2::subtitle_ocr_final_box_offset),
      (std::array<std::uint32_t, 4u> {247u, 364u, 551u, 395u})
    );
    EXPECT_EQ(punctuation[v2::subtitle_ocr_raw_box_offset + 6u], 2u);
    EXPECT_EQ(punctuation[v2::subtitle_ocr_raw_box_offset + 7u], 1u);

    // An invalid micro-island cannot bridge two valid text islands. Each raw gap is four cells, so
    // the unfiltered topology initially forms one run; after the 1x1 middle island is rejected, the
    // effective nine-cell retained gap must publish two independent OCR8 pairs.
    std::fill(probability.begin(), probability.end(), 0.0f);
    paint_rectangle(probability, 80u, 85u, 240u, 100u);
    paint_rectangle(probability, 279u, 92u, 280u, 93u);
    paint_rectangle(probability, 312u, 85u, 472u, 100u);
    ASSERT_TRUE(fixture.run_boxes(probability, 14u, 77u));
    const auto &micro_bridge = fixture.record();
    ASSERT_EQ(micro_bridge[2u], 1u);
    ASSERT_EQ(micro_bridge[3u], 2u);
    ASSERT_EQ(micro_bridge[4u], 2u);
    EXPECT_EQ(
      read_box(micro_bridge, v2::subtitle_ocr_raw_box_offset),
      (std::array<std::uint32_t, 4u> {64u, 373u, 193u, 386u})
    );
    EXPECT_EQ(
      read_box(
        micro_bridge,
        v2::subtitle_ocr_raw_box_offset + v2::subtitle_ocr_box_word_count),
      (std::array<std::uint32_t, 4u> {250u, 373u, 379u, 386u})
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
      (std::array<std::uint32_t, 4u> {6u, 401u, 693u, 430u})
    );
    EXPECT_EQ(
      read_box(record, v2::subtitle_ocr_final_box_offset),
      (std::array<std::uint32_t, 4u> {0u, 395u, field_width, field_height})
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
      (std::array<std::uint32_t, 4u> {192u, 297u, 578u, 304u})
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

  TEST(HostSbsOcrGpuTest, LogoEmptyNaNAndMismatchedIdentityNeverGainConditioningAuthority) {
    ocr_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, false, true, true)) << error;
    constexpr std::uint64_t generation = 77u;

    std::vector<float> line_and_logo(ocr_pixels, 0.0f);
    paint_rectangle(line_and_logo, 240u, 100u, 720u, 108u);
    paint_rectangle(line_and_logo, 880u, 100u, 920u, 140u);

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
      (std::array<std::uint32_t, 4u> {705u, 385u, 738u, 418u})
    );
    EXPECT_EQ(
      read_box(
        logo_record,
        v2::subtitle_ocr_final_box_offset + v2::subtitle_ocr_box_word_count
      ),
      (std::array<std::uint32_t, 4u> {694u, 374u, 750u, 430u})
    );

    // OCR deliberately preserves the separated square candidate. SLR12 owns the generic shape
    // rejection and admits only the wide subtitle line into pending/current authority.
    fixture.reset_locator();
    ASSERT_TRUE(fixture.dispatch_locator(10u, generation, true));
    ASSERT_EQ(fixture.state().size(), v2::subtitle_locator_state_word_count);
    EXPECT_EQ(fixture.state()[2u], 2u);
    EXPECT_EQ(fixture.state()[12u], 1u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    ASSERT_TRUE(fixture.run_boxes(line_and_logo, 11u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(11u, generation));
    EXPECT_EQ(fixture.state()[2u], 1u | 4u);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(
      (std::array<std::uint32_t, 4u> {
        fixture.state()[64u],
        fixture.state()[65u],
        fixture.state()[66u],
        fixture.state()[67u]
      }),
      (std::array<std::uint32_t, 4u> {187u, 380u, 583u, 398u})
    );
    EXPECT_LT(fixture.output_at(300u, 389u), 0.03f);

    // A fully empty probability map is an authoritative same-frame absence. It removes the
    // established owner immediately and the conditioned field becomes bit-exact Base.
    std::vector<float> empty(ocr_pixels, 0.0f);
    ASSERT_TRUE(fixture.run_boxes(empty, 12u, generation));
    EXPECT_EQ(fixture.record()[2u], 1u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);
    ASSERT_TRUE(fixture.dispatch_locator(12u, generation));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // Re-establish a line owner, then inject a non-finite probability. OCR8 abstains (flags=0),
    // SLR12 discards all authority, and no stale target can alter Base.
    std::vector<float> line_only(ocr_pixels, 0.0f);
    paint_rectangle(line_only, 240u, 100u, 720u, 108u);
    ASSERT_TRUE(fixture.run_boxes(line_only, 13u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(13u, generation));
    ASSERT_TRUE(fixture.run_boxes(line_only, 14u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(14u, generation));
    ASSERT_EQ(fixture.state()[20u], 1u);

    auto nonfinite = line_only;
    nonfinite[0u] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(fixture.run_boxes(nonfinite, 15u, generation));
    EXPECT_EQ(fixture.record()[2u], 0u);
    EXPECT_EQ(fixture.record()[3u], 0u);
    EXPECT_EQ(fixture.record()[4u], 0u);
    ASSERT_TRUE(fixture.dispatch_locator(15u, generation));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // A valid OCR8 record whose frame identity does not match b2 is equally non-authoritative.
    ASSERT_TRUE(fixture.run_boxes(line_only, 16u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(16u, generation));
    ASSERT_TRUE(fixture.run_boxes(line_only, 17u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(17u, generation));
    ASSERT_EQ(fixture.state()[20u], 1u);
    ASSERT_TRUE(fixture.run_boxes(line_only, 18u, generation));
    ASSERT_TRUE(fixture.dispatch_locator(19u, generation));
    EXPECT_EQ(fixture.state()[0u], v2::subtitle_locator_state_schema);
    EXPECT_EQ(fixture.state()[1u], v2::subtitle_locator_state_tag);
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }
}  // namespace

#else

TEST(HostSbsOcrGpuTest, WindowsOnly) {
  GTEST_SKIP() << "D3D11 WARP is Windows-only";
}

#endif
