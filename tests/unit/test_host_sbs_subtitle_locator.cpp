/**
 * @file tests/unit/test_host_sbs_subtitle_locator.cpp
 * @brief Deterministic WARP coverage for the compact OCR8/SLR12 lower-text authority.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/generated/depth_coordinate_v2_contract.h"
  #include "src/generated/sbs_adaptive_state_contract.h"

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
  #include <numeric>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace v2 = models::depth_coordinate_v2;

  using tensor_content_t = std::array<std::uint32_t, 4u>;

  constexpr std::uint32_t ceil_ocr_crop_height(
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
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const tensor_content_t &content,
    const std::uint32_t detector_row
  ) {
    const auto crop_height = ceil_ocr_crop_height(source_width, source_height);
    const auto crop_top = source_height - crop_height;
    const auto content_height = content[3u] - content[1u];
    const std::uint64_t numerator =
      (static_cast<std::uint64_t>(crop_top) * v2::subtitle_ocr_output_height +
       static_cast<std::uint64_t>(detector_row) * crop_height) * content_height;
    const std::uint64_t denominator =
      static_cast<std::uint64_t>(v2::subtitle_ocr_output_height) * source_height;
    return content[1u] +
           static_cast<std::uint32_t>((numerator + denominator - 1u) / denominator);
  }

  constexpr std::uint32_t field_width = v2::model_calibrated_shapes[0u].width;
  constexpr std::uint32_t field_height = v2::model_calibrated_shapes[0u].height;
  constexpr auto default_roi =
    v2::subtitle_ocr_dynamic_roi(1920u, 1080u, field_width, field_height);
  constexpr std::uint32_t roi_top = default_roi.top;
  constexpr std::uint32_t roi_bottom = default_roi.bottom;
  constexpr std::uint32_t ocr_words = v2::subtitle_ocr_record_word_count;
  constexpr std::uint32_t state_words = v2::subtitle_locator_state_word_count;
  constexpr std::uint32_t ocr_schema = v2::subtitle_ocr_record_schema;
  constexpr std::uint32_t ocr_tag = v2::subtitle_ocr_record_tag;
  constexpr std::uint32_t slr_schema = v2::subtitle_locator_state_schema;
  constexpr std::uint32_t slr_tag = v2::subtitle_locator_state_tag;
  constexpr std::uint32_t cut_tag = sbs_adaptive_state::cut_contract_tag;
  constexpr std::uint32_t final_box_offset = v2::subtitle_ocr_final_box_offset;

  constexpr std::uint32_t flag_owner = 1u;
  constexpr std::uint32_t flag_pending = 2u;
  constexpr std::uint32_t flag_target_valid = 4u;
  constexpr std::uint32_t flag_target_reset = 8u;
  constexpr std::uint32_t box_flag_ribbon = v2::subtitle_ocr_box_flag_ribbon;
  constexpr std::uint32_t owner_kind_shift = v2::subtitle_locator_owner_kind_shift;
  constexpr std::uint32_t pending_kind_shift = v2::subtitle_locator_pending_kind_shift;
  constexpr std::uint32_t current_kind_shift = v2::subtitle_locator_current_kind_shift;

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
    tensor_content_t tensor_content;
    std::array<std::uint32_t, 3u> reserved;
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
    tensor_content_t tensor_content;
  };

  struct line_box_t {
    std::uint32_t left;
    std::uint32_t top;
    std::uint32_t right;
    std::uint32_t bottom;
  };

  struct ocr_box_t {
    line_box_t core;
    line_box_t cover;
    std::uint32_t kind_flags;
    std::uint32_t island_count;
    std::uint32_t structural_gap_count;

    constexpr ocr_box_t(const line_box_t box) :
        core(box), cover(box), kind_flags(0u), island_count(1u), structural_gap_count(0u) {}

    constexpr ocr_box_t(
      const line_box_t core_box,
      const line_box_t cover_box,
      const std::uint32_t flags,
      const std::uint32_t islands,
      const std::uint32_t structural_gaps
    ) :
        core(core_box),
        cover(cover_box),
        kind_flags(flags),
        island_count(islands),
        structural_gap_count(structural_gaps) {}
  };

  struct condition_dispatch_expectation_t {
    std::array<std::uint32_t, 2u> origin;
    std::array<std::uint32_t, 3u> groups;
  };

  std::uint32_t conservative_condition_pad(
    const float max_delta,
    const float core_range,
    const float slope,
    const std::uint32_t content_width,
    const std::uint32_t axis_extent
  ) {
    if (axis_extent == 0u || max_delta <= core_range) return 0u;
    const float step = slope / static_cast<float>(content_width);
    if (!std::isfinite(step) || step <= 0.0f) return axis_extent;
    const float raw_pad = (max_delta - core_range) / step;
    if (!std::isfinite(raw_pad) || raw_pad >= static_cast<float>(axis_extent)) {
      return axis_extent;
    }
    const auto pad = std::min(
      axis_extent,
      static_cast<std::uint32_t>(std::ceil(raw_pad)) + 1u
    );
    const float first_excluded_budget =
      core_range + static_cast<float>(pad + 1u) * step;
    return std::isfinite(first_excluded_budget) && first_excluded_budget >= max_delta ?
             pad : axis_extent;
  }

  condition_dispatch_expectation_t expected_ordinary_condition_dispatch(
    const std::vector<line_box_t> &covers,
    const float target,
    const std::uint32_t source_width,
    const std::uint32_t field_width,
    const std::uint32_t field_height
  ) {
    const float max_delta = v2::direct_container_limit + std::abs(target);
    const float core_range = 0.5f / static_cast<float>(source_width);
    const auto horizontal_pad = conservative_condition_pad(
      max_delta,
      core_range,
      v2::max_horizontal_slope,
      field_width,
      field_width
    );
    const auto vertical_pad = conservative_condition_pad(
      max_delta,
      core_range,
      v2::max_vertical_shear,
      field_width,
      field_height
    );
    std::uint32_t left = field_width;
    std::uint32_t top = field_height;
    std::uint32_t right = 0u;
    std::uint32_t bottom = 0u;
    for (const auto cover : covers) {
      left = std::min(left, cover.left - std::min(horizontal_pad, cover.left));
      top = std::min(top, cover.top - std::min(vertical_pad, cover.top));
      right = std::max(right, cover.right + std::min(
        horizontal_pad, field_width - cover.right));
      bottom = std::max(bottom, cover.bottom + std::min(
        vertical_pad, field_height - cover.bottom));
    }
    const std::uint32_t origin_x = (left / 16u) * 16u;
    const std::uint32_t origin_y = (top / 16u) * 16u;
    return {
      {origin_x, origin_y},
      {
        (right - origin_x + 15u) / 16u,
        (bottom - origin_y + 15u) / 16u,
        1u,
      },
    };
  }

  static_assert(sizeof(depth_constants_t) == 64u);
  static_assert(sizeof(v2_constants_t) == 32u);
  static_assert(sizeof(subtitle_constants_t) == 64u);

  bool compile_compute_shader(
    ID3D11Device *device,
    const std::filesystem::path &path,
    const char *entry,
    ComPtr<ID3D11ComputeShader> &shader,
    std::string &error
  ) {
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    const auto status = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      entry,
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
    const auto create_status = device->CreateComputeShader(
      bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shader
    );
    if (FAILED(create_status)) {
      error = std::string("CreateComputeShader failed for ") + entry;
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
    if (byte_count == 0u || byte_count > std::numeric_limits<UINT>::max()) return false;
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = static_cast<UINT>(byte_count);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = bind_flags;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA data {initial, 0u, 0u};
    if (FAILED(device->CreateBuffer(&desc, initial ? &data : nullptr, &buffer))) return false;
    if (srv && FAILED(device->CreateShaderResourceView(
                 buffer.Get(), nullptr, srv->ReleaseAndGetAddressOf()))) return false;
    return !uav || SUCCEEDED(device->CreateUnorderedAccessView(
                     buffer.Get(), nullptr, uav->ReleaseAndGetAddressOf()));
  }

  bool create_indirect_dispatch_buffer(
    ID3D11Device *device,
    ComPtr<ID3D11Buffer> &buffer,
    ComPtr<ID3D11UnorderedAccessView> &uav
  ) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = v2::subtitle_condition_dispatch_arg_word_count *
                     sizeof(std::uint32_t);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
    if (FAILED(device->CreateBuffer(&desc, nullptr, &buffer))) return false;
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc {};
    uav_desc.Format = DXGI_FORMAT_R32_UINT;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.NumElements = v2::subtitle_condition_dispatch_arg_word_count;
    return SUCCEEDED(device->CreateUnorderedAccessView(
      buffer.Get(), &uav_desc, &uav));
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
    return SUCCEEDED(device->CreateBuffer(&desc, &data, &buffer));
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
    D3D11_BUFFER_DESC staging_desc = source_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0u;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0u;
    staging_desc.StructureByteStride = 0u;
    ComPtr<ID3D11Buffer> staging;
    if (FAILED(device->CreateBuffer(&staging_desc, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
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
    if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
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

  class slr12_warp_fixture_t {
   public:
    explicit slr12_warp_fixture_t(
      const std::uint32_t field_w = field_width,
      const std::uint32_t field_h = field_height,
      const std::uint32_t source_w = 1920u,
      const std::uint32_t source_h = 1080u,
      const std::uint32_t roi_top_override = std::numeric_limits<std::uint32_t>::max(),
      const std::uint32_t roi_bottom_override = std::numeric_limits<std::uint32_t>::max(),
      tensor_content_t tensor_content = {}
    ) :
        field_width_(field_w),
        field_height_(field_h),
        source_width_(source_w),
        source_height_(source_h),
        tensor_content_(tensor_content) {
      if (tensor_content_ == tensor_content_t {}) {
        tensor_content_ = {0u, 0u, field_w, field_h};
      }
      const v2::subtitle_ocr_roi_t dynamic_roi {
        subtitle_roi_edge(
          source_w, source_h, tensor_content_, v2::subtitle_ocr_safe_row_top),
        subtitle_roi_edge(
          source_w, source_h, tensor_content_, v2::subtitle_ocr_safe_row_bottom),
      };
      roi_top_ = roi_top_override == std::numeric_limits<std::uint32_t>::max() ?
                   dynamic_roi.top :
                   roi_top_override;
      roi_bottom_ = roi_bottom_override == std::numeric_limits<std::uint32_t>::max() ?
                      dynamic_roi.bottom :
                      roi_bottom_override;
    }

    bool initialize(std::string &error) {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      if (FAILED(D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_WARP,
            nullptr,
            0u,
            requested,
            static_cast<UINT>(std::size(requested)),
            D3D11_SDK_VERSION,
            &device_,
            &actual,
            &context_
          )) || actual < D3D_FEATURE_LEVEL_11_0) return false;

      const auto shader_path = std::filesystem::path(SUNSHINE_SHADERS_DIR) /
                               "host_sbs_subtitle_locator_cs.hlsl";
      if (!compile_compute_shader(device_.Get(), shader_path, "resolve_main", resolve_, error) ||
          !compile_compute_shader(
            device_.Get(), shader_path, "condition_prepare_main", condition_prepare_, error) ||
          !compile_compute_shader(
            device_.Get(), shader_path, "condition_in_place_main", condition_in_place_, error) ||
          !compile_compute_shader(device_.Get(), shader_path, "condition_main", condition_, error)) {
        return false;
      }

      std::vector<std::uint32_t> zeros(state_words, 0u);
      if (!create_structured_buffer(
            device_.Get(),
            zeros.data(),
            zeros.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            state_buffer_,
            &state_srv_,
            &state_uav_
          )) return false;

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
          ) || !create_indirect_dispatch_buffer(
            device_.Get(), condition_args_buffer_, condition_args_uav_)) return false;

      zeros.assign(ocr_words, 0u);
      if (!create_structured_buffer(
            device_.Get(),
            zeros.data(),
            zeros.size() * sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE,
            ocr_buffer_,
            &ocr_srv_,
            nullptr
          )) return false;

      std::array<std::uint32_t, 32u> cut_words {};
      cut_words[0u] = cut_tag;
      if (!create_structured_buffer(
            device_.Get(),
            cut_words.data(),
            cut_words.size() * sizeof(std::uint32_t),
            4u * sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE,
            cut_buffer_,
            &cut_srv_,
            nullptr
          )) return false;

      base_.assign(static_cast<std::size_t>(field_width_) * field_height_, 0.03f);
      const auto sample_top = std::min(roi_top_ + 5u, field_height_);
      const auto sample_bottom = std::min(roi_top_ + 32u, field_height_);
      for (std::uint32_t y = sample_top; y < sample_bottom; ++y) {
        std::fill_n(base_.begin() + static_cast<std::size_t>(y) * field_width_,
                    field_width_, 0.01f);
      }
      D3D11_TEXTURE2D_DESC texture_desc {};
      texture_desc.Width = field_width_;
      texture_desc.Height = field_height_;
      texture_desc.MipLevels = 1u;
      texture_desc.ArraySize = 1u;
      texture_desc.Format = DXGI_FORMAT_R32_FLOAT;
      texture_desc.SampleDesc.Count = 1u;
      texture_desc.Usage = D3D11_USAGE_DEFAULT;
      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA base_data {
        base_.data(), static_cast<UINT>(field_width_ * sizeof(float)), 0u
      };
      if (FAILED(device_->CreateTexture2D(&texture_desc, &base_data, &base_texture_)) ||
          FAILED(device_->CreateShaderResourceView(base_texture_.Get(), nullptr, &base_srv_))) {
        return false;
      }
      texture_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      if (FAILED(device_->CreateTexture2D(&texture_desc, nullptr, &output_texture_)) ||
          FAILED(device_->CreateUnorderedAccessView(output_texture_.Get(), nullptr, &output_uav_))) {
        return false;
      }

      const depth_constants_t depth_constants {
        field_width_, field_height_, 0u, 0.0f, 0.0f, 0u, 0.0f, 0.0f, 0.0f,
        tensor_content_, {}
      };
      const v2_constants_t v2_constants {
        0.04f, 0.0001f, 0.0f, 0.0f, 1.0f, 0.5f, 0.04f, 0.0f
      };
      const subtitle_constants_t subtitle_constants {
        {field_width_, field_height_, roi_top_, roi_bottom_},
        {source_width_, source_height_, 1u, 0u},
        {0u, 0u, 0u, 0u},
        tensor_content_,
      };
      return create_constant_buffer(
               device_.Get(), &depth_constants, sizeof(depth_constants), depth_cb_) &&
             create_constant_buffer(
               device_.Get(), &v2_constants, sizeof(v2_constants), v2_cb_) &&
             create_constant_buffer(
               device_.Get(), &subtitle_constants, sizeof(subtitle_constants), subtitle_cb_);
    }

    bool observe(
      const std::uint64_t identity,
      const std::vector<ocr_box_t> &boxes,
      const bool reset,
      const bool authoritative = true,
      const bool corrupt_frame_identity = false,
      const bool corrupt_first_score = false,
      const bool submitted = true
    ) {
      std::array<std::uint32_t, 32u> cut_words {};
      cut_words[0u] = cut_tag;
      cut_words[16u] = scene_epoch_;
      cut_words[22u] = std::bit_cast<std::uint32_t>(cut_pulse_ ? 1.0f : 0.0f);
      context_->UpdateSubresource(
        cut_buffer_.Get(), 0u, nullptr, cut_words.data(), 0u, 0u
      );
      cut_pulse_ = false;

      std::vector<std::uint32_t> record(ocr_words, 0u);
      record[0u] = ocr_schema;
      record[1u] = ocr_tag;
      record[2u] = authoritative ? 1u : 0u;
      record[3u] = authoritative ? static_cast<std::uint32_t>(boxes.size()) : 0u;
      record[4u] = authoritative ? static_cast<std::uint32_t>(boxes.size()) : 0u;
      record[5u] = static_cast<std::uint32_t>(identity) + (corrupt_frame_identity ? 1u : 0u);
      record[6u] = static_cast<std::uint32_t>(identity >> 32u);
      record[7u] = static_cast<std::uint32_t>(identity);
      record[8u] = static_cast<std::uint32_t>(identity >> 32u);
      record[9u] = source_width_;
      record[10u] = source_height_;
      record[11u] = field_width_;
      record[12u] = field_height_;
      record[13u] = roi_top_;
      record[14u] = roi_bottom_;
      if (authoritative) {
        for (std::size_t slot = 0u; slot < boxes.size(); ++slot) {
          write_box(
            record,
            16u + static_cast<std::uint32_t>(slot) * 8u,
            boxes[slot].core,
            boxes[slot]
          );
          write_box(
            record,
            final_box_offset + static_cast<std::uint32_t>(slot) * 8u,
            boxes[slot].cover,
            boxes[slot]
          );
        }
        if (corrupt_first_score && !boxes.empty()) {
          record[16u + 4u] = std::bit_cast<std::uint32_t>(
            std::numeric_limits<float>::quiet_NaN()
          );
        }
      }
      context_->UpdateSubresource(ocr_buffer_.Get(), 0u, nullptr, record.data(), 0u, 0u);

      const subtitle_constants_t subtitle_constants {
        {field_width_, field_height_, roi_top_, roi_bottom_},
        {source_width_, source_height_, submitted ? 1u : 0u, reset ? 1u : 0u},
        {
          static_cast<std::uint32_t>(identity),
          static_cast<std::uint32_t>(identity >> 32u),
          static_cast<std::uint32_t>(identity),
          static_cast<std::uint32_t>(identity >> 32u),
        },
        tensor_content_,
      };
      context_->UpdateSubresource(
        subtitle_cb_.Get(), 0u, nullptr, &subtitle_constants, 0u, 0u
      );
      ID3D11Buffer *constant_buffers[] = {depth_cb_.Get(), v2_cb_.Get(), subtitle_cb_.Get()};
      context_->CSSetConstantBuffers(0u, 3u, constant_buffers);

      context_->CSSetShader(resolve_.Get(), nullptr, 0u);
      std::array<ID3D11ShaderResourceView *, 8u> resolve_srvs {
        nullptr, cut_srv_.Get(), base_srv_.Get(), nullptr, nullptr, nullptr, nullptr, ocr_srv_.Get()
      };
      context_->CSSetShaderResources(0u, resolve_srvs.size(), resolve_srvs.data());
      context_->CSSetUnorderedAccessViews(2u, 1u, state_uav_.GetAddressOf(), nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind();

      if (!dispatch_conditioner()) return false;

      return read_buffer(device_.Get(), context_.Get(), state_buffer_.Get(), state_) &&
             read_buffer(
               device_.Get(), context_.Get(), condition_params_buffer_.Get(), condition_params_) &&
             read_buffer(
               device_.Get(), context_.Get(), condition_args_buffer_.Get(), condition_args_) &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    bool condition_only() {
      if (!dispatch_conditioner()) return false;
      return read_buffer(
               device_.Get(), context_.Get(), condition_params_buffer_.Get(), condition_params_) &&
             read_buffer(
               device_.Get(), context_.Get(), condition_args_buffer_.Get(), condition_args_) &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    bool condition_only_in_place() {
      if (!dispatch_conditioner(true)) return false;
      return read_buffer(
               device_.Get(), context_.Get(), condition_params_buffer_.Get(), condition_params_) &&
             read_buffer(
               device_.Get(), context_.Get(), condition_args_buffer_.Get(), condition_args_) &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    void set_cut(const std::uint32_t scene_epoch, const bool pulse) {
      scene_epoch_ = scene_epoch;
      cut_pulse_ = pulse;
    }

    const std::vector<std::uint32_t> &state() const { return state_; }
    const std::vector<std::uint32_t> &condition_params() const { return condition_params_; }
    const std::vector<std::uint32_t> &condition_args() const { return condition_args_; }
    const std::vector<float> &base() const { return base_; }
    const std::vector<float> &output() const { return output_; }

    bool overwrite_state_word(const std::size_t index, const std::uint32_t value) {
      if (index >= state_.size()) return false;
      state_[index] = value;
      context_->UpdateSubresource(
        state_buffer_.Get(), 0u, nullptr, state_.data(), 0u, 0u
      );
      return true;
    }

    bool swap_state_rectangles(const std::size_t first, const std::size_t second) {
      if (first + 4u > state_.size() || second + 4u > state_.size()) return false;
      for (std::size_t word = 0u; word < 4u; ++word) {
        std::swap(state_[first + word], state_[second + word]);
      }
      context_->UpdateSubresource(
        state_buffer_.Get(), 0u, nullptr, state_.data(), 0u, 0u
      );
      return true;
    }

    float output_at(const std::uint32_t x, const std::uint32_t y) const {
      return output_.at(static_cast<std::size_t>(y) * field_width_ + x);
    }

    void set_base(const float value) {
      std::fill(base_.begin(), base_.end(), value);
      const auto sample_top = std::min(roi_top_ + 5u, field_height_);
      const auto sample_bottom = std::min(roi_top_ + 32u, field_height_);
      for (std::uint32_t y = sample_top; y < sample_bottom; ++y) {
        std::fill_n(base_.begin() + static_cast<std::size_t>(y) * field_width_,
                    field_width_, value);
      }
      context_->UpdateSubresource(
        base_texture_.Get(), 0u, nullptr, base_.data(),
        static_cast<UINT>(field_width_ * sizeof(float)), 0u
      );
    }

    void set_background_sample_rows(
      const line_box_t line,
      const float outer_value,
      const float inner_value
    ) {
      const auto outer_y = std::clamp(
        line.top >= 10u ? line.top - 10u : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto inner_y = std::clamp(
        line.top >= 4u ? line.top - 4u : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto center = 0.5f * static_cast<float>(line.left + line.right - 1u);
      for (std::uint32_t sample = 0u; sample < 16u; ++sample) {
        const auto x = static_cast<std::uint32_t>(std::clamp(
          std::floor(center - 30.0f + 4.0f * static_cast<float>(sample) + 0.5f),
          static_cast<float>(tensor_content_[0u]),
          static_cast<float>(tensor_content_[2u] - 1u)
        ));
        base_[static_cast<std::size_t>(outer_y) * field_width_ + x] = outer_value;
        base_[static_cast<std::size_t>(inner_y) * field_width_ + x] = inner_value;
      }
      context_->UpdateSubresource(
        base_texture_.Get(), 0u, nullptr, base_.data(),
        static_cast<UINT>(field_width_ * sizeof(float)), 0u
      );
    }

    void set_background_sample_row_values(
      const line_box_t line,
      const std::array<float, 16u> &outer_values,
      const std::array<float, 16u> &inner_values
    ) {
      const auto outer_y = std::clamp(
        line.top >= 10u ? line.top - 10u : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto inner_y = std::clamp(
        line.top >= 4u ? line.top - 4u : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto center = 0.5f * static_cast<float>(line.left + line.right - 1u);
      for (std::uint32_t sample = 0u; sample < 16u; ++sample) {
        const auto x = static_cast<std::uint32_t>(std::clamp(
          std::floor(center - 30.0f + 4.0f * static_cast<float>(sample) + 0.5f),
          static_cast<float>(tensor_content_[0u]),
          static_cast<float>(tensor_content_[2u] - 1u)
        ));
        base_[static_cast<std::size_t>(outer_y) * field_width_ + x] = outer_values[sample];
        base_[static_cast<std::size_t>(inner_y) * field_width_ + x] = inner_values[sample];
      }
      context_->UpdateSubresource(
        base_texture_.Get(), 0u, nullptr, base_.data(),
        static_cast<UINT>(field_width_ * sizeof(float)), 0u
      );
    }

    void set_background_sample_alternating(
      const line_box_t line,
      const float first_value,
      const float second_value
    ) {
      const auto outer_y = std::clamp(
        line.top >= 10u ? line.top - 10u : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto inner_y = std::clamp(
        line.top >= 4u ? line.top - 4u : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto center = 0.5f * static_cast<float>(line.left + line.right - 1u);
      for (std::uint32_t sample = 0u; sample < 16u; ++sample) {
        const auto x = static_cast<std::uint32_t>(std::clamp(
          std::floor(center - 30.0f + 4.0f * static_cast<float>(sample) + 0.5f),
          static_cast<float>(tensor_content_[0u]),
          static_cast<float>(tensor_content_[2u] - 1u)
        ));
        const auto value = (sample & 1u) == 0u ? first_value : second_value;
        base_[static_cast<std::size_t>(outer_y) * field_width_ + x] = value;
        base_[static_cast<std::size_t>(inner_y) * field_width_ + x] = value;
      }
      context_->UpdateSubresource(
        base_texture_.Get(), 0u, nullptr, base_.data(),
        static_cast<UINT>(field_width_ * sizeof(float)), 0u
      );
    }

    void set_base_columns(
      const std::uint32_t left,
      const std::uint32_t right,
      const float value
    ) {
      const auto clamped_left = std::min(left, field_width_);
      const auto clamped_right = std::min(std::max(right, clamped_left), field_width_);
      for (std::uint32_t y = 0u; y < field_height_; ++y) {
        std::fill(
          base_.begin() + static_cast<std::size_t>(y) * field_width_ + clamped_left,
          base_.begin() + static_cast<std::size_t>(y) * field_width_ + clamped_right,
          value
        );
      }
      context_->UpdateSubresource(
        base_texture_.Get(), 0u, nullptr, base_.data(),
        static_cast<UINT>(field_width_ * sizeof(float)), 0u
      );
    }

    void set_base_at(
      const std::uint32_t x,
      const std::uint32_t y,
      const float value
    ) {
      base_.at(static_cast<std::size_t>(y) * field_width_ + x) = value;
      context_->UpdateSubresource(
        base_texture_.Get(), 0u, nullptr, base_.data(),
        static_cast<UINT>(field_width_ * sizeof(float)), 0u
      );
    }

    bool output_is_exact_base() const {
      if (output_.size() != base_.size()) return false;
      for (std::size_t index = 0u; index < base_.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(output_[index]) !=
            std::bit_cast<std::uint32_t>(base_[index])) return false;
      }
      return true;
    }

    bool in_place_matches_current_output() {
      const auto expected = output_;
      if (!condition_only_in_place() || output_.size() != expected.size()) return false;
      for (std::size_t index = 0u; index < expected.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(output_[index]) !=
            std::bit_cast<std::uint32_t>(expected[index])) return false;
      }
      return true;
    }

   private:
    bool dispatch_conditioner(const bool in_place = false) {
      ID3D11Buffer *constant_buffers[] = {depth_cb_.Get(), v2_cb_.Get(), subtitle_cb_.Get()};
      context_->CSSetConstantBuffers(0u, 3u, constant_buffers);

      context_->CSSetShader(condition_prepare_.Get(), nullptr, 0u);
      std::array<ID3D11ShaderResourceView *, 5u> prepare_srvs {
        nullptr, cut_srv_.Get(), nullptr, state_srv_.Get(), nullptr
      };
      context_->CSSetShaderResources(0u, prepare_srvs.size(), prepare_srvs.data());
      std::array<ID3D11UnorderedAccessView *, 2u> prepare_uavs {
        condition_params_uav_.Get(), condition_args_uav_.Get()
      };
      context_->CSSetUnorderedAccessViews(4u, prepare_uavs.size(), prepare_uavs.data(), nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind();

      if (in_place) {
        // Production already has Base in the final UAV. Seed the standalone fixture the same way,
        // then exercise the no-SRV in-place entrypoint and its zero-group gate.
        context_->CopyResource(output_texture_.Get(), base_texture_.Get());
      }
      context_->CSSetShader(in_place ? condition_in_place_.Get() : condition_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 3u, constant_buffers);
      std::array<ID3D11ShaderResourceView *, 5u> condition_srvs {
        nullptr, nullptr, in_place ? nullptr : base_srv_.Get(),
        state_srv_.Get(), condition_params_srv_.Get()
      };
      context_->CSSetShaderResources(0u, condition_srvs.size(), condition_srvs.data());
      context_->CSSetUnorderedAccessViews(3u, 1u, output_uav_.GetAddressOf(), nullptr);
      if (in_place) {
        context_->DispatchIndirect(condition_args_buffer_.Get(), 0u);
      } else {
        context_->Dispatch(
          (field_width_ + 15u) / 16u,
          (field_height_ + 15u) / 16u,
          1u
        );
      }
      unbind();
      return true;
    }

    static void write_box(
      std::vector<std::uint32_t> &record,
      const std::uint32_t offset,
      const line_box_t box,
      const ocr_box_t &metadata
    ) {
      record[offset + 0u] = box.left;
      record[offset + 1u] = box.top;
      record[offset + 2u] = box.right;
      record[offset + 3u] = box.bottom;
      record[offset + 4u] = std::bit_cast<std::uint32_t>(0.9f);
      record[offset + 5u] = metadata.kind_flags;
      record[offset + 6u] = metadata.island_count;
      record[offset + 7u] = metadata.structural_gap_count;
    }

    void unbind() {
      std::array<ID3D11ShaderResourceView *, 8u> null_srvs {};
      std::array<ID3D11UnorderedAccessView *, 6u> null_uavs {};
      context_->CSSetShaderResources(0u, null_srvs.size(), null_srvs.data());
      context_->CSSetUnorderedAccessViews(0u, null_uavs.size(), null_uavs.data(), nullptr);
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> resolve_;
    ComPtr<ID3D11ComputeShader> condition_prepare_;
    ComPtr<ID3D11ComputeShader> condition_in_place_;
    ComPtr<ID3D11ComputeShader> condition_;
    ComPtr<ID3D11Buffer> state_buffer_;
    ComPtr<ID3D11ShaderResourceView> state_srv_;
    ComPtr<ID3D11UnorderedAccessView> state_uav_;
    ComPtr<ID3D11Buffer> condition_params_buffer_;
    ComPtr<ID3D11ShaderResourceView> condition_params_srv_;
    ComPtr<ID3D11UnorderedAccessView> condition_params_uav_;
    ComPtr<ID3D11Buffer> condition_args_buffer_;
    ComPtr<ID3D11UnorderedAccessView> condition_args_uav_;
    ComPtr<ID3D11Buffer> ocr_buffer_;
    ComPtr<ID3D11ShaderResourceView> ocr_srv_;
    ComPtr<ID3D11Buffer> cut_buffer_;
    ComPtr<ID3D11ShaderResourceView> cut_srv_;
    ComPtr<ID3D11Texture2D> base_texture_;
    ComPtr<ID3D11ShaderResourceView> base_srv_;
    ComPtr<ID3D11Texture2D> output_texture_;
    ComPtr<ID3D11UnorderedAccessView> output_uav_;
    ComPtr<ID3D11Buffer> depth_cb_;
    ComPtr<ID3D11Buffer> v2_cb_;
    ComPtr<ID3D11Buffer> subtitle_cb_;
    std::vector<std::uint32_t> state_;
    std::vector<std::uint32_t> condition_params_;
    std::vector<std::uint32_t> condition_args_;
    std::vector<float> base_;
    std::vector<float> output_;
    std::uint32_t field_width_;
    std::uint32_t field_height_;
    std::uint32_t source_width_;
    std::uint32_t source_height_;
    tensor_content_t tensor_content_ {};
    std::uint32_t roi_top_;
    std::uint32_t roi_bottom_;
    std::uint32_t scene_epoch_ = 0u;
    bool cut_pulse_ = false;
  };

  TEST(HostSbsSubtitleSlr12GpuTest, PartialOcr8RestampChangesOnlyFrameIdentity) {
    constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL actual {};
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
      nullptr,
      D3D_DRIVER_TYPE_WARP,
      nullptr,
      0u,
      requested,
      static_cast<UINT>(std::size(requested)),
      D3D11_SDK_VERSION,
      &device,
      &actual,
      &context
    )));

    std::vector<std::uint32_t> original(ocr_words);
    std::iota(original.begin(), original.end(), 0x1000u);
    ComPtr<ID3D11Buffer> record_buffer;
    ASSERT_TRUE(create_structured_buffer(
      device.Get(),
      original.data(),
      original.size() * sizeof(std::uint32_t),
      sizeof(std::uint32_t),
      D3D11_BIND_SHADER_RESOURCE,
      record_buffer,
      nullptr,
      nullptr
    ));

    constexpr std::array<std::uint32_t, 2> identity {0x89abcdefu, 0x01234567u};
    const D3D11_BOX identity_box {
      static_cast<UINT>(5u * sizeof(std::uint32_t)), 0u, 0u,
      static_cast<UINT>(7u * sizeof(std::uint32_t)), 1u, 1u,
    };
    context->UpdateSubresource(
      record_buffer.Get(), 0u, &identity_box, identity.data(), 0u, 0u
    );

    std::vector<std::uint32_t> restamped;
    ASSERT_TRUE(read_buffer(
      device.Get(), context.Get(), record_buffer.Get(), restamped
    ));
    ASSERT_EQ(restamped.size(), original.size());
    EXPECT_EQ(restamped[5u], identity[0u]);
    EXPECT_EQ(restamped[6u], identity[1u]);
    for (std::size_t word = 0u; word < restamped.size(); ++word) {
      if (word == 5u || word == 6u) continue;
      EXPECT_EQ(restamped[word], original[word]) << "word " << word;
    }
  }

  TEST(HostSbsSubtitleSlr12GpuTest, ConfirmsExactFrameStacksAndConditionsOnlyCurrentLines) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t first {180u, 360u, 590u, 370u};
    // Deliberately shorter and left-aligned: live captions commonly append a translated line
    // without recentering the first line.
    const line_box_t second {180u, 374u, 500u, 384u};

    // A single observation is pending and has no conditioning authority.
    ASSERT_TRUE(fixture.observe(1u, {first}, false));
    ASSERT_EQ(fixture.state().size(), state_words);
    EXPECT_EQ(fixture.state()[0u], slr_schema);
    EXPECT_EQ(fixture.state()[1u], slr_tag);
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 1u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(
      fixture.condition_args(),
      (std::vector<std::uint32_t> {0u, 0u, 0u})
    );
    ASSERT_EQ(
      fixture.condition_params().size(),
      v2::subtitle_condition_param_word_count
    );
    EXPECT_TRUE(std::all_of(
      fixture.condition_params().begin(),
      fixture.condition_params().end(),
      [](const std::uint32_t word) { return word == 0u; }
    ));
    EXPECT_TRUE(fixture.output_is_exact_base());

    // Redispatching the same exact frame/domain identity cannot self-confirm.
    ASSERT_TRUE(fixture.observe(1u, {first}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // A distinct compatible observation births the owner at half fade.
    ASSERT_TRUE(fixture.observe(2u, {first}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 1u);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 1u);
    EXPECT_EQ(fixture.state()[24u], 1u);
    const auto expected_dispatch = expected_ordinary_condition_dispatch(
      {first},
      std::bit_cast<float>(fixture.state()[18u]),
      1920u,
      field_width,
      field_height
    );
    EXPECT_EQ(
      fixture.condition_args(),
      (std::vector<std::uint32_t> {
        expected_dispatch.groups[0u],
        expected_dispatch.groups[1u],
        expected_dispatch.groups[2u],
      })
    );
    EXPECT_LT(
      expected_dispatch.groups[0u] * expected_dispatch.groups[1u],
      ((field_width + 15u) / 16u) * ((field_height + 15u) / 16u)
    );
    ASSERT_EQ(
      fixture.condition_params().size(),
      v2::subtitle_condition_param_word_count
    );
    EXPECT_EQ(fixture.condition_params()[0u], v2::subtitle_condition_param_schema);
    EXPECT_EQ(fixture.condition_params()[1u], v2::subtitle_condition_param_tag);
    EXPECT_EQ(fixture.condition_params()[2u], fixture.state()[20u]);
    EXPECT_EQ(
      fixture.condition_params()[3u],
      (fixture.state()[v2::subtitle_locator_kind_word] >> current_kind_shift) &
        v2::subtitle_locator_kind_mask
    );
    EXPECT_EQ(fixture.condition_params()[4u], fixture.state()[24u]);
    EXPECT_EQ(fixture.condition_params()[5u], fixture.state()[18u]);
    EXPECT_EQ(fixture.condition_params()[6u], expected_dispatch.origin[0u]);
    EXPECT_EQ(fixture.condition_params()[7u], expected_dispatch.origin[1u]);
    EXPECT_LT(fixture.output_at(300u, 364u), 0.025f);
    EXPECT_EQ(
      std::bit_cast<std::uint32_t>(fixture.output_at(10u, 100u)),
      std::bit_cast<std::uint32_t>(0.03f)
    );

    // The live entrypoint reads/writes only the final UAV. It must be bit-identical to the
    // complete out-of-place writer for a valid full-content field.
    const auto out_of_place = fixture.output();
    ASSERT_TRUE(fixture.condition_only_in_place());
    ASSERT_EQ(fixture.output().size(), out_of_place.size());
    for (std::size_t index = 0u; index < out_of_place.size(); ++index) {
      ASSERT_EQ(
        std::bit_cast<std::uint32_t>(fixture.output()[index]),
        std::bit_cast<std::uint32_t>(out_of_place[index])
      ) << "cell " << index;
    }

    // The next observation reaches full strength.
    ASSERT_TRUE(fixture.observe(3u, {first}, false));
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_NEAR(fixture.output_at(300u, 364u), 0.0102604f, 0.00002f);

    // A newly appended translation line is pending.  Only the matched old line is current;
    // the new line receives at most the old line's analytic collar, never a fabricated core.
    ASSERT_TRUE(fixture.observe(4u, {first, second}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_pending | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 2u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_GT(fixture.output_at(300u, 377u), 0.020f);

    // The second distinct stack observation confirms a two-line handoff.  Both dense line cores
    // are active, while their four-row gap remains a distance collar rather than a filled box.
    ASSERT_TRUE(fixture.observe(5u, {first, second}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 2u);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(fixture.state()[21u], 3u);
    EXPECT_EQ(fixture.state()[24u], 1u);
    const float first_core = fixture.output_at(300u, 364u);
    const float gap = fixture.output_at(300u, 372u);
    const float second_core = fixture.output_at(300u, 377u);
    EXPECT_LT(first_core, gap);
    EXPECT_LT(second_core, gap);
    EXPECT_LT(gap, 0.03f);
    const auto expected_multi_dispatch = expected_ordinary_condition_dispatch(
      {first, second},
      std::bit_cast<float>(fixture.state()[18u]),
      1920u,
      field_width,
      field_height
    );
    EXPECT_EQ(
      fixture.condition_args(),
      (std::vector<std::uint32_t> {
        expected_multi_dispatch.groups[0u],
        expected_multi_dispatch.groups[1u],
        expected_multi_dispatch.groups[2u],
      })
    );
    EXPECT_EQ(fixture.condition_params()[6u], expected_multi_dispatch.origin[0u]);
    EXPECT_EQ(fixture.condition_params()[7u], expected_multi_dispatch.origin[1u]);
    EXPECT_TRUE(fixture.in_place_matches_current_output());

    // Stress the strict collar boundary with the accepted Base value farthest from this target.
    // Any underestimated origin or extent would now diverge from the complete writer bit-for-bit.
    const float active_target = std::bit_cast<float>(fixture.state()[18u]);
    fixture.set_base(
      active_target >= 0.0f ? -v2::direct_container_limit : v2::direct_container_limit
    );
    ASSERT_TRUE(fixture.condition_only());
    EXPECT_TRUE(fixture.in_place_matches_current_output());

    // Authoritative empty OCR removes geometry immediately.  The cached target/grace is not
    // geometry authority, so every texel is exact Base on the death frame and thereafter.
    ASSERT_TRUE(fixture.observe(6u, {}, false));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 2u);
    EXPECT_EQ(fixture.state()[25u], 6u);
    EXPECT_EQ(
      fixture.condition_args(),
      (std::vector<std::uint32_t> {0u, 0u, 0u})
    );
    EXPECT_TRUE(std::all_of(
      fixture.condition_params().begin(),
      fixture.condition_params().end(),
      [](const std::uint32_t word) { return word == 0u; }
    ));
    EXPECT_TRUE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.observe(7u, {}, false));
    EXPECT_EQ(fixture.state()[25u], 5u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr12GpuTest, SameBaselineSegmentsShareOwnerWithoutUnioningCovers) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    // A detector gap just beyond the four-cell ordinary join becomes two OCR8 pairs. Their cores
    // remain strongly baseline-aligned and near enough to form one SLR12 owner, but each paired
    // cover stays independent and the horizontal gap is conditioned only by analytic collars.
    const ocr_box_t left {
      {180u, 360u, 380u, 372u},
      {176u, 356u, 384u, 376u},
      0u,
      1u,
      0u,
    };
    const ocr_box_t right {
      {420u, 360u, 600u, 372u},
      {416u, 356u, 604u, 376u},
      0u,
      1u,
      0u,
    };
    ASSERT_TRUE(fixture.observe(8u, {left, right}, false));
    ASSERT_EQ(fixture.state()[2u], flag_pending);
    ASSERT_EQ(fixture.state()[12u], 2u);
    ASSERT_TRUE(fixture.observe(9u, {left, right}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    ASSERT_EQ(fixture.state()[4u], 2u);
    ASSERT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(fixture.state()[32u], left.core.left);
    EXPECT_EQ(fixture.state()[36u], right.core.left);
    EXPECT_EQ(fixture.state()[64u], left.cover.left);
    EXPECT_EQ(fixture.state()[66u], left.cover.right);
    EXPECT_EQ(fixture.state()[68u], right.cover.left);
    EXPECT_EQ(fixture.state()[70u], right.cover.right);
    const float left_core = fixture.output_at(250u, 365u);
    const float cover_gap = fixture.output_at(400u, 365u);
    const float right_core = fixture.output_at(500u, 365u);
    EXPECT_LT(left_core, cover_gap);
    EXPECT_LT(right_core, cover_gap);
    EXPECT_LT(cover_gap, 0.03f);

    // A nearby scene-text line with no strong vertical overlap cannot bridge into this baseline
    // component. The larger two-segment subtitle wins deterministic area selection by itself.
    const line_box_t vertically_offset_scene_text {200u, 330u, 500u, 340u};
    slr12_warp_fixture_t guarded;
    ASSERT_TRUE(guarded.initialize(error)) << error;
    ASSERT_TRUE(guarded.observe(
      10u, {left, right, vertically_offset_scene_text}, false));
    EXPECT_EQ(guarded.state()[2u], flag_pending);
    EXPECT_EQ(guarded.state()[12u], 2u);
    EXPECT_EQ(guarded.state()[48u], left.core.left);
    EXPECT_EQ(guarded.state()[52u], right.core.left);

    // Component closure remains fail-flat at compact-state capacity. It must not retain an
    // arbitrary four of five same-baseline segments.
    const line_box_t one {50u, 360u, 150u, 370u};
    const line_box_t two {170u, 360u, 270u, 370u};
    const line_box_t three {290u, 360u, 390u, 370u};
    const line_box_t four {410u, 360u, 510u, 370u};
    const line_box_t five {530u, 360u, 630u, 370u};
    slr12_warp_fixture_t overflow;
    ASSERT_TRUE(overflow.initialize(error)) << error;
    ASSERT_TRUE(overflow.observe(11u, {one, two, three, four, five}, false));
    EXPECT_EQ(overflow.state()[2u], 0u);
    EXPECT_EQ(overflow.state()[4u], 0u);
    EXPECT_EQ(overflow.state()[12u], 0u);
    EXPECT_EQ(overflow.state()[20u], 0u);
    EXPECT_TRUE(overflow.output_is_exact_base());

    // Pairwise-valid edges must not let transitive closure exceed the ordinary 0.9-field span.
    // A-B and B-C satisfy SameBaselineSegments, but A-C does not and the whole component is wider
    // than the authenticated ordinary-line maximum; reject it rather than selecting a bridge chain.
    const line_box_t chain_a {10u, 380u, 210u, 390u};
    const line_box_t chain_b {250u, 380u, 450u, 390u};
    const line_box_t chain_c {490u, 380u, 730u, 390u};
    slr12_warp_fixture_t bridge;
    ASSERT_TRUE(bridge.initialize(error)) << error;
    ASSERT_TRUE(bridge.observe(12u, {chain_a, chain_b, chain_c}, false));
    EXPECT_EQ(bridge.state()[2u], 0u);
    EXPECT_EQ(bridge.state()[12u], 0u);
    EXPECT_TRUE(bridge.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr12GpuTest, StackConfirmationRequiresEveryMemberToOverlap) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    // The first two members are identical, but the third moves far enough to have zero IoU while
    // remaining a coherent three-line stack. Aggregate IoU would be 2/3 and used to confirm this
    // one-observation replacement immediately; every member must now satisfy the 0.6 threshold.
    const line_box_t first {100u, 350u, 600u, 360u};
    const line_box_t second {100u, 362u, 600u, 372u};
    const line_box_t old_third {100u, 374u, 350u, 384u};
    const line_box_t replacement_third {350u, 374u, 600u, 384u};

    ASSERT_TRUE(fixture.observe(80u, {first, second, old_third}, false));
    ASSERT_EQ(fixture.state()[2u], flag_pending);
    ASSERT_EQ(fixture.state()[12u], 3u);

    ASSERT_TRUE(fixture.observe(81u, {first, second, replacement_third}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 3u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    ASSERT_TRUE(fixture.observe(82u, {first, second, replacement_third}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 3u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 3u);
    EXPECT_EQ(fixture.state()[21u], 1u);
  }

  TEST(HostSbsSubtitleSlr12GpuTest, PartialAuthoritySamplesOnlyMatchedCurrentGeometry) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t old_line {180u, 360u, 590u, 370u};
    // A 100-cell shift retains just over 0.6 IoU with old_line. Its target-sampling strip is
    // disjoint from old_line's strip, so this test distinguishes exact current evidence from the
    // stale owner geometry retained while the appended stack is pending.
    const line_box_t shifted_line {280u, 360u, 690u, 370u};
    const line_box_t appended_line {280u, 374u, 600u, 384u};
    fixture.set_base(0.0f);
    ASSERT_TRUE(fixture.observe(90u, {old_line}, false));
    ASSERT_TRUE(fixture.observe(91u, {old_line}, false));
    ASSERT_EQ(fixture.state()[18u], std::bit_cast<std::uint32_t>(0.0f));

    fixture.set_base(0.0f);
    fixture.set_base_columns(450u, 520u, 4.0f / (2.0f * 1920.0f));
    ASSERT_TRUE(fixture.observe(92u, {shifted_line, appended_line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_pending | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 2u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    const auto one_slew =
      v2::subtitle_target_max_slew_binocular_source_pixels / (2.0f * 1920.0f);
    EXPECT_NEAR(std::bit_cast<float>(fixture.state()[18u]), one_slew, 1.0e-8f);
  }

  TEST(HostSbsSubtitleSlr12GpuTest, NoncanonicalOwnerAndPendingCoreOrderFailFlat) {
    const line_box_t first {180u, 350u, 590u, 360u};
    const line_box_t second {180u, 362u, 500u, 372u};
    std::string error;

    slr12_warp_fixture_t owner_fixture;
    ASSERT_TRUE(owner_fixture.initialize(error)) << error;
    ASSERT_TRUE(owner_fixture.observe(93u, {first, second}, false));
    ASSERT_TRUE(owner_fixture.observe(94u, {first, second}, false));
    ASSERT_EQ(owner_fixture.state()[4u], 2u);
    ASSERT_TRUE(owner_fixture.swap_state_rectangles(
      v2::subtitle_locator_owner_offset,
      v2::subtitle_locator_owner_offset + 4u
    ));
    ASSERT_TRUE(owner_fixture.condition_only());
    EXPECT_TRUE(owner_fixture.output_is_exact_base());
    ASSERT_TRUE(owner_fixture.observe(95u, {first, second}, false));
    EXPECT_EQ(owner_fixture.state()[2u], flag_pending);
    EXPECT_EQ(owner_fixture.state()[4u], 0u);
    EXPECT_EQ(owner_fixture.state()[12u], 2u);
    EXPECT_TRUE(owner_fixture.output_is_exact_base());

    slr12_warp_fixture_t pending_fixture;
    ASSERT_TRUE(pending_fixture.initialize(error)) << error;
    ASSERT_TRUE(pending_fixture.observe(96u, {first}, false));
    ASSERT_TRUE(pending_fixture.observe(97u, {first}, false));
    ASSERT_TRUE(pending_fixture.observe(98u, {first, second}, false));
    ASSERT_EQ(
      pending_fixture.state()[2u], flag_owner | flag_pending | flag_target_valid
    );
    ASSERT_EQ(pending_fixture.state()[12u], 2u);
    ASSERT_TRUE(pending_fixture.swap_state_rectangles(
      v2::subtitle_locator_pending_offset,
      v2::subtitle_locator_pending_offset + 4u
    ));
    ASSERT_TRUE(pending_fixture.condition_only());
    EXPECT_TRUE(pending_fixture.output_is_exact_base());
    ASSERT_TRUE(pending_fixture.observe(99u, {first, second}, false));
    EXPECT_EQ(pending_fixture.state()[2u], flag_pending);
    EXPECT_EQ(pending_fixture.state()[4u], 0u);
    EXPECT_EQ(pending_fixture.state()[12u], 2u);
    EXPECT_TRUE(pending_fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr12GpuTest, OwnerTargetTracksReliableLocalPlaneWithoutPumping) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t first {180u, 360u, 590u, 370u};
    const line_box_t jittered {182u, 361u, 592u, 371u};
    const line_box_t handoff {80u, 360u, 380u, 370u};
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    fixture.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(fixture.observe(100u, {first}, false));
    ASSERT_TRUE(fixture.observe(101u, {first}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    const std::uint32_t birth_target = fixture.state()[18u];
    // A reliable local supporting plane is used as-is instead of being pulled to an absolute
    // near-screen band.
    EXPECT_EQ(birth_target, std::bit_cast<std::uint32_t>(target_for_pixels(2.0f)));
    EXPECT_EQ(fixture.state()[24u], 1u);

    // A distinct observation within the one-pixel deadband preserves the exact target bits even
    // when compatible OCR geometry jitters. This is the no-pumping path.
    fixture.set_background_sample_rows(
      jittered, target_for_pixels(2.75f), target_for_pixels(2.75f)
    );
    ASSERT_TRUE(fixture.observe(102u, {jittered}, false));
    EXPECT_EQ(fixture.state()[18u], birth_target);
    EXPECT_EQ(fixture.state()[24u], 2u);

    // Outside the deadband, EMA asks for a larger move but the target advances by at most 0.25
    // binocular source-equivalent pixels per distinct observation. A duplicate identity is an
    // exact target/fade hold and cannot consume another slew step.
    fixture.set_background_sample_rows(
      jittered, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(fixture.observe(103u, {jittered}, false));
    const auto first_slew = std::bit_cast<float>(fixture.state()[18u]);
    EXPECT_NEAR(first_slew, target_for_pixels(2.25f), 1.0e-8f);
    const auto first_slew_bits = fixture.state()[18u];
    ASSERT_TRUE(fixture.observe(103u, {jittered}, false));
    EXPECT_EQ(fixture.state()[18u], first_slew_bits);
    EXPECT_EQ(fixture.state()[24u], 2u);
    ASSERT_TRUE(fixture.observe(104u, {jittered}, false));
    EXPECT_NEAR(
      std::bit_cast<float>(fixture.state()[18u]), target_for_pixels(2.5f), 1.0e-8f
    );

    // A disjoint material handoff has no same-frame current authority on its first observation.
    // It outputs exact Base and cannot use stale owner geometry to move the cached target. Once
    // confirmed, its current geometry samples the plane and takes one bounded step.
    const auto pre_handoff = std::bit_cast<float>(fixture.state()[18u]);
    ASSERT_TRUE(fixture.observe(105u, {handoff}, false));
    EXPECT_EQ(std::bit_cast<float>(fixture.state()[18u]), pre_handoff);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    const auto pending_target = std::bit_cast<float>(fixture.state()[18u]);
    fixture.set_background_sample_rows(
      handoff, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(fixture.observe(106u, {handoff}, false));
    ASSERT_EQ(fixture.state()[21u], 3u);
    EXPECT_NEAR(
      std::bit_cast<float>(fixture.state()[18u]) - pending_target,
      target_for_pixels(0.25f),
      1.0e-8f
    );
    EXPECT_EQ(fixture.state()[24u], 1u);

    // Missing current evidence clears geometry immediately but caches the reliable target.
    // Reacquisition during grace resumes through the same bounded update instead of reviving a
    // stale scene plane unchanged.
    ASSERT_TRUE(fixture.observe(107u, {}, false));
    ASSERT_EQ(fixture.state()[25u], 6u);
    const auto cached_target = std::bit_cast<float>(fixture.state()[18u]);
    fixture.set_background_sample_rows(
      handoff, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(fixture.observe(108u, {handoff}, false));
    EXPECT_EQ(std::bit_cast<float>(fixture.state()[18u]), cached_target);
    ASSERT_TRUE(fixture.observe(109u, {handoff}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_NEAR(
      std::bit_cast<float>(fixture.state()[18u]) - cached_target,
      target_for_pixels(0.25f),
      1.0e-8f
    );

    // A hard-cut survivor discards the old scene's plane and restarts from same-frame evidence.
    const auto pre_cut_generation = fixture.state()[3u];
    fixture.set_background_sample_rows(handoff, 0.03f, 0.03f);
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(110u, {handoff}, false));
    EXPECT_EQ(fixture.state()[24u], 1u);
    EXPECT_EQ(fixture.state()[18u], std::bit_cast<std::uint32_t>(0.03f));
    EXPECT_EQ(fixture.state()[3u], pre_cut_generation + 1u);

    // A fresh birth retains a reliable plane well beyond the retired 0..8-pixel band.
    slr12_warp_fixture_t local_plane;
    ASSERT_TRUE(local_plane.initialize(error)) << error;
    local_plane.set_base(0.03f);
    ASSERT_TRUE(local_plane.observe(200u, {first}, false));
    ASSERT_TRUE(local_plane.observe(201u, {first}, false));
    EXPECT_EQ(local_plane.state()[18u], std::bit_cast<std::uint32_t>(0.03f));

    // Event is part of the authenticated current-state envelope too; a foreign value must make
    // the conditioner copy exact Base rather than accepting otherwise plausible geometry.
    const auto birth_event = local_plane.state()[21u];
    ASSERT_TRUE(local_plane.overwrite_state_word(21u, 4u));
    ASSERT_TRUE(local_plane.condition_only());
    EXPECT_TRUE(local_plane.output_is_exact_base());
    ASSERT_TRUE(local_plane.overwrite_state_word(21u, birth_event));

    // Corrupting an otherwise well-formed target outside the direct container invalidates the
    // whole previous state. The next box is pending and conditioner output is exact Base.
    ASSERT_TRUE(local_plane.overwrite_state_word(
      18u,
      std::bit_cast<std::uint32_t>(v2::direct_container_limit + 0.001f)
    ));
    ASSERT_TRUE(local_plane.condition_only());
    EXPECT_TRUE(local_plane.output_is_exact_base());
    ASSERT_TRUE(local_plane.observe(202u, {first}, false));
    EXPECT_EQ(local_plane.state()[2u], flag_pending);
    EXPECT_EQ(local_plane.state()[4u], 0u);
    EXPECT_EQ(local_plane.state()[20u], 0u);
    EXPECT_TRUE(local_plane.output_is_exact_base());

  }

  TEST(HostSbsSubtitleSlr12GpuTest, UnreliablePlaneHoldsTwiceThenFailsBaseAndRecovers) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    fixture.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(fixture.observe(210u, {line}, false));
    ASSERT_TRUE(fixture.observe(211u, {line}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);

    // Both rows are bimodal with a ten-pixel Tukey IQR. Preserve the last reliable plane for two
    // distinct observations so one noisy estimate cannot expose warped glyph edges immediately.
    fixture.set_background_sample_alternating(
      line, target_for_pixels(2.0f), target_for_pixels(12.0f)
    );
    ASSERT_TRUE(fixture.observe(212u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(2.0f)));
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[25u], 1u);

    // Redispatching the same failed identity cannot consume another hold.
    ASSERT_TRUE(fixture.observe(212u, {line}, false));
    EXPECT_EQ(fixture.state()[25u], 1u);
    ASSERT_TRUE(fixture.observe(213u, {line}, false));
    EXPECT_EQ(fixture.state()[25u], 2u);
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);

    // The third distinct unreliable observation exhausts the bounded hold and copies exact Base.
    ASSERT_TRUE(fixture.observe(214u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_reset);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[25u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // A later reliable observation reacquires directly at fade 1 and clears the hold counter.
    fixture.set_background_sample_rows(
      line, target_for_pixels(12.0f), target_for_pixels(12.0f)
    );
    ASSERT_TRUE(fixture.observe(215u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(12.0f)));
    EXPECT_EQ(fixture.state()[24u], 1u);
    EXPECT_EQ(fixture.state()[25u], 0u);
  }

  TEST(HostSbsSubtitleSlr12GpuTest, RowSelectionUsesStableEvidenceAndNearerPlane) {
    std::string error;
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };

    // Two coherent nearby rows share their medians.
    slr12_warp_fixture_t close;
    ASSERT_TRUE(close.initialize(error)) << error;
    close.set_background_sample_rows(
      line, target_for_pixels(2.0f), target_for_pixels(5.0f)
    );
    ASSERT_TRUE(close.observe(220u, {line}, false));
    ASSERT_TRUE(close.observe(221u, {line}, false));
    const auto source_space_candidate = target_for_pixels(3.5f);
    const auto sm5_operation_order_candidate =
      0.5f * (target_for_pixels(2.0f) + target_for_pixels(5.0f));
    EXPECT_TRUE(
      close.state()[18u] == std::bit_cast<std::uint32_t>(source_space_candidate) ||
      close.state()[18u] == std::bit_cast<std::uint32_t>(sm5_operation_order_candidate)
    );

    // Two coherent split rows select the numerically larger/nearer supporting plane.
    slr12_warp_fixture_t split;
    ASSERT_TRUE(split.initialize(error)) << error;
    split.set_background_sample_rows(
      line, target_for_pixels(2.0f), target_for_pixels(7.0f)
    );
    ASSERT_TRUE(split.observe(222u, {line}, false));
    ASSERT_TRUE(split.observe(223u, {line}, false));
    EXPECT_EQ(split.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(7.0f)));

    // Dump-1-like evidence has a coherent far row near 41.89 px and a heterogeneous nearer row
    // whose median is 54.88 px. One coherent row establishes support; selecting the larger finite
    // median avoids flattening almost the whole subtitle cover seventeen pixels away from Base.
    std::array<float, 16u> coherent {};
    coherent.fill(target_for_pixels(41.89f));
    std::array<float, 16u> heterogeneous {
      target_for_pixels(44.0f), target_for_pixels(44.0f), target_for_pixels(44.0f),
      target_for_pixels(44.0f), target_for_pixels(44.0f), target_for_pixels(48.0f),
      target_for_pixels(53.0f), target_for_pixels(54.0f), target_for_pixels(55.76f),
      target_for_pixels(56.0f), target_for_pixels(58.0f), target_for_pixels(59.0f),
      target_for_pixels(59.0f), target_for_pixels(59.0f), target_for_pixels(59.0f),
      target_for_pixels(59.0f),
    };
    slr12_warp_fixture_t dump_like;
    ASSERT_TRUE(dump_like.initialize(error)) << error;
    dump_like.set_background_sample_row_values(line, coherent, heterogeneous);
    ASSERT_TRUE(dump_like.observe(224u, {line}, false));
    ASSERT_TRUE(dump_like.observe(225u, {line}, false));
    EXPECT_NEAR(
      std::bit_cast<float>(dump_like.state()[18u]), target_for_pixels(54.88f), 1.0e-8f
    );

    // If the heterogeneous row remains close to coherent support, their medians are averaged.
    std::array<float, 16u> close_heterogeneous {
      target_for_pixels(36.0f), target_for_pixels(36.0f), target_for_pixels(36.0f),
      target_for_pixels(36.0f), target_for_pixels(36.0f), target_for_pixels(40.0f),
      target_for_pixels(41.0f), target_for_pixels(42.0f), target_for_pixels(42.0f),
      target_for_pixels(43.0f), target_for_pixels(46.0f), target_for_pixels(46.0f),
      target_for_pixels(46.0f), target_for_pixels(46.0f), target_for_pixels(46.0f),
      target_for_pixels(46.0f),
    };
    slr12_warp_fixture_t close_mixed;
    ASSERT_TRUE(close_mixed.initialize(error)) << error;
    close_mixed.set_background_sample_row_values(line, coherent, close_heterogeneous);
    ASSERT_TRUE(close_mixed.observe(2250u, {line}, false));
    ASSERT_TRUE(close_mixed.observe(2251u, {line}, false));
    EXPECT_NEAR(
      std::bit_cast<float>(close_mixed.state()[18u]),
      target_for_pixels((41.89f + 42.0f) * 0.5f),
      1.0e-8f
    );

    // A malformed second row does not revoke an independently coherent first row.
    std::array<float, 16u> invalid {};
    for (std::size_t index = 0u; index < invalid.size(); ++index) {
      invalid[index] = (index & 1u) == 0u ?
                         std::numeric_limits<float>::quiet_NaN() :
                         v2::direct_container_limit + 0.001f;
    }
    slr12_warp_fixture_t one_valid;
    ASSERT_TRUE(one_valid.initialize(error)) << error;
    one_valid.set_background_sample_row_values(line, coherent, invalid);
    ASSERT_TRUE(one_valid.observe(226u, {line}, false));
    ASSERT_TRUE(one_valid.observe(227u, {line}, false));
    EXPECT_EQ(
      one_valid.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(41.89f))
    );

    // The exact eight-pixel row-IQR boundary remains coherent.
    std::array<float, 16u> boundary {
      target_for_pixels(0.0f), target_for_pixels(0.0f), target_for_pixels(0.0f),
      target_for_pixels(0.0f), target_for_pixels(0.0f), target_for_pixels(4.0f),
      target_for_pixels(4.0f), target_for_pixels(4.0f), target_for_pixels(4.0f),
      target_for_pixels(4.0f), target_for_pixels(8.0f), target_for_pixels(8.0f),
      target_for_pixels(8.0f), target_for_pixels(8.0f), target_for_pixels(8.0f),
      target_for_pixels(8.0f),
    };
    slr12_warp_fixture_t exact_boundary;
    ASSERT_TRUE(exact_boundary.initialize(error)) << error;
    exact_boundary.set_background_sample_row_values(line, boundary, invalid);
    ASSERT_TRUE(exact_boundary.observe(228u, {line}, false));
    ASSERT_TRUE(exact_boundary.observe(229u, {line}, false));
    EXPECT_EQ(
      exact_boundary.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(4.0f))
    );

    // With neither row coherent a fresh owner has no plane to hold and must copy exact Base.
    slr12_warp_fixture_t neither;
    ASSERT_TRUE(neither.initialize(error)) << error;
    neither.set_background_sample_alternating(
      line, target_for_pixels(2.0f), target_for_pixels(12.0f)
    );
    ASSERT_TRUE(neither.observe(230u, {line}, false));
    ASSERT_TRUE(neither.observe(231u, {line}, false));
    EXPECT_EQ(neither.state()[2u], flag_owner | flag_target_reset);
    EXPECT_EQ(neither.state()[25u], 0u);
    EXPECT_TRUE(neither.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr12GpuTest, LargeResidualAndHardCutRestartButCutNeverHolds) {
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    std::string error;

    slr12_warp_fixture_t tracking;
    ASSERT_TRUE(tracking.initialize(error)) << error;
    tracking.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(tracking.observe(230u, {line}, false));
    ASSERT_TRUE(tracking.observe(231u, {line}, false));
    tracking.set_background_sample_rows(
      line, target_for_pixels(12.0f), target_for_pixels(12.0f)
    );
    ASSERT_TRUE(tracking.observe(232u, {line}, false));
    EXPECT_EQ(tracking.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(
      tracking.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(12.0f))
    );
    EXPECT_EQ(tracking.state()[24u], 1u);

    // Across a confirmed scene boundary, old and new plane values are not temporally comparable.
    // Reliable same-frame evidence restarts immediately at fade 1 instead of rendering the old
    // plane at full strength and slewing through unrelated depths.
    slr12_warp_fixture_t cut;
    ASSERT_TRUE(cut.initialize(error)) << error;
    cut.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(cut.observe(240u, {line}, false));
    ASSERT_TRUE(cut.observe(241u, {line}, false));
    cut.set_background_sample_rows(
      line, target_for_pixels(12.0f), target_for_pixels(12.0f)
    );
    cut.set_cut(1u, true);
    ASSERT_TRUE(cut.observe(242u, {line}, false));
    EXPECT_EQ(cut.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(cut.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(12.0f)));
    EXPECT_EQ(cut.state()[24u], 1u);

    // Unreliable evidence on a hard cut cannot hold the old scene's target.
    cut.set_background_sample_alternating(
      line, target_for_pixels(2.0f), target_for_pixels(12.0f)
    );
    cut.set_cut(2u, true);
    ASSERT_TRUE(cut.observe(243u, {line}, false));
    EXPECT_EQ(cut.state()[2u], flag_owner | flag_target_reset);
    EXPECT_EQ(cut.state()[25u], 0u);
    EXPECT_TRUE(cut.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr12GpuTest, DuplicateCutPulseIsBitExactIdempotent) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t line {180u, 360u, 590u, 370u};
    fixture.set_base(0.01f);
    ASSERT_TRUE(fixture.observe(250u, {line}, false));
    ASSERT_TRUE(fixture.observe(251u, {line}, false));
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(252u, {line}, false));
    const auto state_after_cut = fixture.state();
    const auto output_after_cut = fixture.output();

    // A still-latched pulse on the exact same observation cannot start another owner generation,
    // reacquire the plane, advance fade, or reconstruct any R32_FLOAT output bit.
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(252u, {line}, false));
    EXPECT_EQ(fixture.state(), state_after_cut);
    ASSERT_EQ(fixture.output().size(), output_after_cut.size());
    for (std::size_t index = 0u; index < output_after_cut.size(); ++index) {
      ASSERT_EQ(
        std::bit_cast<std::uint32_t>(fixture.output()[index]),
        std::bit_cast<std::uint32_t>(output_after_cut[index])
      ) << "output index " << index;
    }

    // The conditioner independently binds the compact state to the same-frame authenticated
    // CutBridge epoch. A foreign but otherwise canonical state must copy exact Base.
    ASSERT_TRUE(fixture.overwrite_state_word(26u, 2u));
    ASSERT_TRUE(fixture.condition_only());
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr12GpuTest, MissedCutPulseEpochMismatchRestartsLocalPlane) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    fixture.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(fixture.observe(260u, {line}, false));
    ASSERT_TRUE(fixture.observe(261u, {line}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    ASSERT_EQ(
      fixture.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(2.0f))
    );
    ASSERT_EQ(fixture.state()[26u], 0u);
    const auto previous_generation = fixture.state()[3u];

    // Even if the one-frame pulse was missed, a newer authenticated CutBridge epoch is a hard
    // scene boundary. It discards the old target and directly reacquires reliable current-plane
    // evidence at half strength instead of taking a continuing-owner 0.25-pixel slew step.
    fixture.set_background_sample_rows(
      line, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    fixture.set_cut(1u, false);
    ASSERT_TRUE(fixture.observe(262u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], previous_generation + 1u);
    EXPECT_EQ(
      fixture.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(6.0f))
    );
    EXPECT_NE(
      fixture.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(2.25f))
    );
    EXPECT_EQ(fixture.state()[24u], 1u);
    EXPECT_EQ(fixture.state()[26u], 1u);
  }

  TEST(HostSbsSubtitleSlr12GpuTest, InvalidOcrClearsCurrentButPreservesAndAgesTargetGrace) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t line {180u, 360u, 590u, 370u};
    const line_box_t disjoint {80u, 395u, 380u, 405u};
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    fixture.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(fixture.observe(300u, {line}, false));
    ASSERT_TRUE(fixture.observe(301u, {line}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);

    // Start a disjoint handoff, then make its next observation malformed. Invalid OCR is a miss,
    // never current authority: it clears both current output and pending confirmation, starts
    // grace, and cannot advance the target from the last valid observation.
    fixture.set_base(target_for_pixels(6.0f));
    ASSERT_TRUE(fixture.observe(302u, {disjoint}, false));
    ASSERT_EQ(
      fixture.state()[2u],
      flag_owner | flag_pending | flag_target_valid
    );
    ASSERT_EQ(fixture.state()[12u], 1u);
    const std::uint32_t tracked_target = fixture.state()[18u];
    EXPECT_NEAR(
      std::bit_cast<float>(tracked_target), target_for_pixels(2.0f), 1.0e-8f
    );
    ASSERT_TRUE(fixture.observe(303u, {disjoint}, false, true, false, true));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 2u);
    EXPECT_EQ(fixture.state()[25u], 6u);
    EXPECT_EQ(fixture.state()[18u], tracked_target);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // A same-identity no-submit redispatch cannot consume two grace observations. A later distinct
    // stale/mismatched record consumes exactly one while still publishing no current geometry.
    ASSERT_TRUE(fixture.observe(303u, {}, false, false, false, false, false));
    EXPECT_EQ(fixture.state()[25u], 6u);
    EXPECT_EQ(fixture.state()[18u], tracked_target);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.observe(304u, {disjoint}, false, true, true));
    EXPECT_EQ(fixture.state()[25u], 5u);
    EXPECT_EQ(fixture.state()[18u], tracked_target);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // The first valid rebirth observation is still pending and retains the cached bits. The second
    // confirms an owner and is the next distinct authoritative opportunity for one bounded step.
    ASSERT_TRUE(fixture.observe(305u, {disjoint}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[18u], tracked_target);
    ASSERT_TRUE(fixture.observe(306u, {disjoint}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_NEAR(
      std::bit_cast<float>(fixture.state()[18u]), target_for_pixels(2.25f), 1.0e-8f
    );

    // A hard cut is an explicit lifetime boundary. Invalid same-frame OCR cannot carry grace or
    // target across it and, because it has no geometry, cannot sample a replacement target.
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(307u, {}, false, false));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[25u], 0u);
    EXPECT_EQ(fixture.state()[18u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr12GpuTest, OversizedDeathGraceCannotExtendTargetLifetime) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t line {180u, 360u, 590u, 370u};
    fixture.set_base(-0.011f);
    ASSERT_TRUE(fixture.observe(320u, {line}, false));
    ASSERT_TRUE(fixture.observe(321u, {line}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    const std::uint32_t expired_target = fixture.state()[18u];

    ASSERT_TRUE(fixture.observe(322u, {}, false));
    ASSERT_EQ(
      fixture.state()[25u], v2::subtitle_locator_death_grace_observations
    );
    ASSERT_EQ(fixture.state()[18u], expired_target);

    // Word 25 is authenticated previous-state lifetime, not an open-ended counter. Corrupting it
    // above the generated observation limit must invalidate the whole previous state: the next
    // box is a fresh first observation with no cached target, and only its successor may sample a
    // new one.
    ASSERT_TRUE(fixture.overwrite_state_word(
      25u, v2::subtitle_locator_death_grace_observations + 1u
    ));
    fixture.set_base(0.029f);
    ASSERT_TRUE(fixture.observe(323u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 1u);
    EXPECT_EQ(fixture.state()[18u], 0u);
    EXPECT_EQ(fixture.state()[25u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    ASSERT_TRUE(fixture.observe(324u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[18u], std::bit_cast<std::uint32_t>(0.029f));
    EXPECT_NE(fixture.state()[18u], expired_target);
  }

  TEST(HostSbsSubtitleSlr12GpuTest, TracksSubtitleAndBottomRibbonOnOnePlane) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t subtitle_core {180u, 360u, 590u, 370u};
    const line_box_t subtitle_cover {176u, 356u, 594u, 374u};
    const line_box_t changed_core {80u, 360u, 380u, 370u};
    const line_box_t changed_cover {76u, 356u, 384u, 374u};
    const line_box_t ribbon_core {6u, 401u, 693u, roi_bottom};
    const line_box_t ribbon_cover {0u, 397u, field_width, field_height};
    const ocr_box_t subtitle {subtitle_core, subtitle_cover, 0u, 1u, 0u};
    const ocr_box_t changed {changed_core, changed_cover, 0u, 1u, 0u};
    const ocr_box_t ribbon {ribbon_core, ribbon_cover, box_flag_ribbon, 7u, 4u};

    // Tight/core geometry owns selection and temporal confirmation. The paired full-field ribbon
    // cover is not allowed to merge with the subtitle merely because its conditioning strip meets
    // the bottom of the field.
    ASSERT_TRUE(fixture.observe(20u, {subtitle, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[12u], 2u);
    EXPECT_EQ(
      fixture.state()[31u],
      2u << pending_kind_shift
    );
    EXPECT_TRUE(fixture.output_is_exact_base());

    ASSERT_TRUE(fixture.observe(21u, {subtitle, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(
      fixture.state()[31u],
      (2u << owner_kind_shift) | (2u << current_kind_shift)
    );
    // Owner/pending rectangles are cores; only current same-frame rectangles are covers.
    EXPECT_EQ(fixture.state()[32u], subtitle_core.left);
    EXPECT_EQ(fixture.state()[33u], subtitle_core.top);
    EXPECT_EQ(fixture.state()[36u], ribbon_core.left);
    EXPECT_EQ(fixture.state()[37u], ribbon_core.top);
    EXPECT_EQ(fixture.state()[64u], subtitle_cover.left);
    EXPECT_EQ(fixture.state()[65u], subtitle_cover.top);
    EXPECT_EQ(fixture.state()[68u], ribbon_cover.left);
    EXPECT_EQ(fixture.state()[69u], ribbon_cover.top);

    // Both regions use exactly one sampled scalar plane. The ribbon's full cover includes its
    // gaps and reaches the field bottom; its only analytic collar is above the corrected top.
    EXPECT_NEAR(fixture.output_at(300u, 364u), fixture.output_at(300u, 410u), 0.000001f);
    EXPECT_LT(fixture.output_at(0u, 395u), 0.03f);
    EXPECT_LT(fixture.output_at(300u, field_height - 1u), 0.03f);
    EXPECT_LT(fixture.output_at(174u, 364u), 0.03f);  // Ordinary left collar remains active.
    // The analytic collar is slope-bounded rather than finite-radius, so it decays continuously
    // above the strip instead of creating a hard edge. A farther row must be closer to Base than
    // a row immediately above the corrected top.
    EXPECT_GT(fixture.output_at(0u, 380u), fixture.output_at(0u, 395u));
    EXPECT_LE(fixture.output_at(0u, 380u), 0.03f);
    ASSERT_EQ(
      fixture.condition_params().size(),
      v2::subtitle_condition_param_word_count
    );
    ASSERT_EQ(
      fixture.condition_args().size(),
      v2::subtitle_condition_dispatch_arg_word_count
    );
    EXPECT_EQ(fixture.condition_params()[6u], 0u);
    EXPECT_GT(fixture.condition_params()[7u], 0u);
    EXPECT_EQ(fixture.condition_args()[0u], (field_width + 15u) / 16u);
    EXPECT_LT(fixture.condition_args()[1u], (field_height + 15u) / 16u);
    EXPECT_EQ(
      fixture.condition_args()[1u],
      (field_height - fixture.condition_params()[7u] + 15u) / 16u
    );
    EXPECT_TRUE(fixture.in_place_matches_current_output());

    // A material subtitle handoff leaves the constant ribbon authoritative while the new subtitle
    // waits for its second distinct observation.
    ASSERT_TRUE(fixture.observe(22u, {changed, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_pending | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[12u], 2u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(
      fixture.state()[31u],
      (2u << owner_kind_shift) | (2u << pending_kind_shift) |
        (1u << current_kind_shift)
    );
    EXPECT_EQ(
      std::bit_cast<std::uint32_t>(fixture.output_at(200u, 364u)),
      std::bit_cast<std::uint32_t>(0.03f)
    );

    ASSERT_TRUE(fixture.observe(23u, {changed, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[21u], 3u);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(
      fixture.state()[31u],
      (2u << owner_kind_shift) | (2u << current_kind_shift)
    );

    // A same-frame mixed owner survives a hard cut as two independently matched core rectangles,
    // restarts its shared local plane at half strength, and keeps current cover kinds.
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(24u, {changed, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(fixture.state()[24u], 1u);
    EXPECT_EQ(fixture.state()[26u], 1u);
    EXPECT_EQ(
      fixture.state()[31u],
      (2u << owner_kind_shift) | (2u << current_kind_shift)
    );
  }

  TEST(HostSbsSubtitleSlr12GpuTest, ResetLogoAndInvalidRecordFailToExactBase) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    // A square bottom-corner badge is rejected by generic line geometry; reset state publishes
    // no pending stack and the conditioner is bit-exact Base.
    const line_box_t square_badge {700u, 350u, 740u, 390u};
    ASSERT_TRUE(fixture.observe(10u, {square_badge}, true));
    EXPECT_EQ(fixture.state()[0u], slr_schema);
    EXPECT_EQ(fixture.state()[1u], slr_tag);
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // Exact-frame identity is mandatory.  A structurally valid but stale/mismatched record resets
    // to a canonical current state envelope and cannot retain pending or owner authority.
    const line_box_t line {180u, 360u, 590u, 370u};
    ASSERT_TRUE(fixture.observe(11u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    ASSERT_TRUE(fixture.observe(12u, {line}, false, true, true));
    EXPECT_EQ(fixture.state()[0u], slr_schema);
    EXPECT_EQ(fixture.state()[1u], slr_tag);
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr12GpuTest, MixedSelectionOverCompactCapacityFailsFlat) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const ocr_box_t ribbon {
      {6u, 401u, 693u, roi_bottom},
      {0u, 397u, field_width, field_height},
      box_flag_ribbon,
      7u,
      4u,
    };
    // Four coherent ordinary lines still fit the established compact owner by themselves. Adding
    // an independently authoritative ribbon would require five entities, so the observation must
    // abstain as a whole instead of silently dropping either kind.
    const line_box_t first {180u, 330u, 500u, 338u};
    const line_box_t second {180u, 340u, 500u, 348u};
    const line_box_t third {180u, 350u, 500u, 358u};
    const line_box_t fourth {180u, 360u, 500u, 368u};
    ASSERT_TRUE(fixture.observe(25u, {first, second, third, fourth, ribbon}, false));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[31u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr12GpuTest, HardCutSurvivorRestartsLocalPlaneButDisjointStackIsPending) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t survivor {180u, 360u, 590u, 370u};
    const line_box_t disjoint {80u, 405u, 380u, 417u};
    ASSERT_TRUE(fixture.observe(30u, {survivor}, false));
    ASSERT_TRUE(fixture.observe(31u, {survivor}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);

    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(32u, {survivor}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 2u);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 0u);
    EXPECT_EQ(fixture.state()[24u], 1u);
    EXPECT_EQ(fixture.state()[26u], 1u);

    fixture.set_cut(2u, true);
    ASSERT_TRUE(fixture.observe(33u, {disjoint}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[3u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 1u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 2u);
    EXPECT_EQ(fixture.state()[26u], 2u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // The cut pulse is one observation only; the retained epoch lets the next distinct matching
    // observation confirm a new birth instead of being mistaken for another cut.
    ASSERT_TRUE(fixture.observe(34u, {disjoint}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 1u);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 1u);
    EXPECT_EQ(fixture.state()[26u], 2u);
  }

  TEST(HostSbsSubtitleSlr12GpuTest, PartialHardCutSurvivorHandsOffExpandedStackOnSecondObservation) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t first {180u, 350u, 590u, 358u};
    const line_box_t second {180u, 362u, 590u, 370u};
    const line_box_t added {180u, 374u, 590u, 382u};
    ASSERT_TRUE(fixture.observe(50u, {first, second}, false));
    ASSERT_TRUE(fixture.observe(51u, {first, second}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    ASSERT_EQ(fixture.state()[3u], 1u);

    // The cut observation retains only the two old-owner matches as current authority. The full
    // three-line stack is deliberately pending, while the survivor restarts the new scene's
    // supporting plane at half strength.
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(52u, {first, second, added}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_pending | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 2u);
    EXPECT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(fixture.state()[12u], 3u);
    EXPECT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(fixture.state()[21u], 0u);
    EXPECT_EQ(fixture.state()[24u], 1u);

    // The next distinct matching observation confirms the material three-line handoff. Its
    // generation bump and half-strength first fade are the specified second transaction, not a
    // same-frame promotion of the newly added line.
    ASSERT_TRUE(fixture.observe(53u, {first, second, added}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 3u);
    EXPECT_EQ(fixture.state()[4u], 3u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 3u);
    EXPECT_EQ(fixture.state()[21u], 3u);
    EXPECT_EQ(fixture.state()[24u], 1u);

    ASSERT_TRUE(fixture.observe(54u, {first, second, added}, false));
    EXPECT_EQ(fixture.state()[3u], 3u);
    EXPECT_EQ(fixture.state()[20u], 3u);
    EXPECT_EQ(fixture.state()[24u], 2u);
  }

  TEST(HostSbsSubtitleSlr12GpuTest, DomainResetRequiresPendingThenNextDistinctFrameBirth) {
    slr12_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t line {180u, 360u, 590u, 370u};
    ASSERT_TRUE(fixture.observe(40u, {line}, false));
    ASSERT_TRUE(fixture.observe(41u, {line}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);

    ASSERT_TRUE(fixture.observe(42u, {line}, true));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 1u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // The same exact record cannot satisfy the second-observation transaction after reset.
    ASSERT_TRUE(fixture.observe(42u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    ASSERT_TRUE(fixture.observe(43u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], 1u);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 1u);
    EXPECT_EQ(fixture.state()[24u], 1u);
  }

  TEST(HostSbsSubtitleSlr12GpuTest, SupportsAuthenticatedWideAndPortraitFields) {
    struct field_case_t {
      std::uint32_t field_width;
      std::uint32_t field_height;
      std::uint32_t source_width;
      std::uint32_t source_height;
      std::uint32_t line_left;
      std::uint32_t line_right;
    };
    constexpr std::array cases {
      field_case_t {1022u, 434u, 2560u, 1080u, 100u, 850u},
      field_case_t {434u, 770u, 1080u, 1920u, 40u, 360u},
    };

    std::uint64_t identity = 100u;
    for (const auto &field_case : cases) {
      SCOPED_TRACE(
        std::to_string(field_case.field_width) + "x" +
        std::to_string(field_case.field_height)
      );
      const auto dynamic_roi = v2::subtitle_ocr_dynamic_roi(
        field_case.source_width,
        field_case.source_height,
        field_case.field_width,
        field_case.field_height
      );
      ASSERT_TRUE(static_cast<bool>(dynamic_roi));
      const line_box_t line {
        field_case.line_left,
        dynamic_roi.top + 35u,
        field_case.line_right,
        dynamic_roi.top + 45u,
      };
      ASSERT_LE(line.bottom, dynamic_roi.bottom);

      slr12_warp_fixture_t fixture(
        field_case.field_width,
        field_case.field_height,
        field_case.source_width,
        field_case.source_height
      );
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      ASSERT_TRUE(fixture.observe(identity++, {line}, false));
      EXPECT_EQ(fixture.state()[2u], flag_pending);
      EXPECT_TRUE(fixture.output_is_exact_base());
      ASSERT_TRUE(fixture.observe(identity++, {line}, false));
      EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
      EXPECT_EQ(fixture.state()[4u], 1u);
      EXPECT_EQ(fixture.state()[20u], 1u);
      EXPECT_EQ(fixture.state()[27u], field_case.field_width);
      EXPECT_EQ(fixture.state()[28u], field_case.field_height);
      EXPECT_LT(
        fixture.output_at((line.left + line.right) / 2u, line.top + 4u),
        0.025f
      );
    }
  }

  TEST(HostSbsSubtitleSlr12GpuTest, AuthenticatesPaddedContentAndExtendsItsBoundary) {
    constexpr std::uint32_t source_width = 1500u;
    constexpr std::uint32_t source_height = 500u;
    constexpr tensor_content_t content {0u, 89u, field_width, 345u};
    constexpr auto padded_roi_top = subtitle_roi_edge(
      source_width, source_height, content, v2::subtitle_ocr_safe_row_top
    );
    constexpr auto padded_roi_bottom = subtitle_roi_edge(
      source_width, source_height, content, v2::subtitle_ocr_safe_row_bottom
    );
    static_assert(padded_roi_top == 237u);
    static_assert(padded_roi_bottom == 341u);

    slr12_warp_fixture_t fixture(
      field_width,
      field_height,
      source_width,
      source_height,
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<std::uint32_t>::max(),
      content
    );
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    // Even before an owner is confirmed, condition_main must publish synthetic tensor padding
    // as an exact extension of the nearest real-content cell rather than unrelated BaseField.
    constexpr std::uint32_t sample_x = 300u;
    constexpr float boundary_value = 0.012f;
    fixture.set_base_at(sample_x, content[1u], boundary_value);
    const line_box_t line {
      180u, padded_roi_top + 5u, 590u, padded_roi_top + 15u
    };
    ASSERT_TRUE(fixture.observe(500u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(
      fixture.condition_args(),
      (std::vector<std::uint32_t> {
        (field_width + 15u) / 16u,
        (field_height + 15u) / 16u,
        1u,
      })
    );
    EXPECT_TRUE(std::all_of(
      fixture.condition_params().begin(),
      fixture.condition_params().end(),
      [](const std::uint32_t word) { return word == 0u; }
    ));
    EXPECT_EQ(
      std::bit_cast<std::uint32_t>(fixture.output_at(sample_x, 0u)),
      std::bit_cast<std::uint32_t>(boundary_value)
    );
    EXPECT_EQ(
      std::bit_cast<std::uint32_t>(fixture.output_at(sample_x, content[1u])),
      std::bit_cast<std::uint32_t>(boundary_value)
    );

    ASSERT_TRUE(fixture.observe(501u, {line}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(
      fixture.condition_args(),
      (std::vector<std::uint32_t> {
        (field_width + 15u) / 16u,
        (field_height + 15u) / 16u,
        1u,
      })
    );
    EXPECT_EQ(fixture.condition_params()[6u], 0u);
    EXPECT_EQ(fixture.condition_params()[7u], 0u);
    constexpr auto current = v2::subtitle_locator_current_offset;
    EXPECT_EQ(fixture.state()[current + 0u], line.left);
    EXPECT_EQ(fixture.state()[current + 1u], line.top);
    EXPECT_EQ(fixture.state()[current + 2u], line.right);
    EXPECT_EQ(fixture.state()[current + 3u], line.bottom);
    EXPECT_GE(fixture.state()[current + 0u], content[0u]);
    EXPECT_GE(fixture.state()[current + 1u], content[1u]);
    EXPECT_LE(fixture.state()[current + 2u], content[2u]);
    EXPECT_LE(fixture.state()[current + 3u], content[3u]);
  }

  TEST(HostSbsSubtitleSlr12GpuTest, RibbonBottomToleranceProjectsExactlyAcrossFields) {
    struct field_case_t {
      std::uint32_t field_width;
      std::uint32_t field_height;
      std::uint32_t source_width;
      std::uint32_t source_height;
    };
    constexpr std::array cases {
      field_case_t {770u, 434u, 1920u, 1080u},
      field_case_t {1022u, 434u, 2560u, 1080u},
      field_case_t {1036u, 434u, 3840u, 1600u},
      field_case_t {434u, 770u, 1080u, 1920u},
      field_case_t {434u, 1022u, 1080u, 2560u},
      field_case_t {434u, 1036u, 1600u, 3840u},
    };

    std::uint64_t identity = 300u;
    for (const auto &field_case : cases) {
      SCOPED_TRACE(
        std::to_string(field_case.field_width) + "x" +
        std::to_string(field_case.field_height)
      );
      const auto dynamic_roi = v2::subtitle_ocr_dynamic_roi(
        field_case.source_width,
        field_case.source_height,
        field_case.field_width,
        field_case.field_height
      );
      ASSERT_TRUE(static_cast<bool>(dynamic_roi));
      const auto minimum_bottom_result = v2::subtitle_ocr_ribbon_min_bottom(
        field_case.source_width,
        field_case.source_height,
        field_case.field_width,
        field_case.field_height
      );
      ASSERT_TRUE(static_cast<bool>(minimum_bottom_result));
      const auto minimum_bottom = minimum_bottom_result.value;
      ASSERT_GE(minimum_bottom, dynamic_roi.top + 7u);
      ASSERT_LE(minimum_bottom, dynamic_roi.bottom);
      const auto left = field_case.field_width / 20u;
      const auto right = field_case.field_width - left;
      const auto core_top = minimum_bottom - 6u;
      const auto cover_top = core_top > 4u ? core_top - 4u : 0u;
      const ocr_box_t boundary_ribbon {
        {left, core_top, right, minimum_bottom},
        {0u, cover_top, field_case.field_width, field_case.field_height},
        box_flag_ribbon,
        7u,
        4u,
      };

      slr12_warp_fixture_t accepted(
        field_case.field_width,
        field_case.field_height,
        field_case.source_width,
        field_case.source_height
      );
      std::string error;
      ASSERT_TRUE(accepted.initialize(error)) << error;
      ASSERT_TRUE(accepted.observe(identity++, {boundary_ribbon}, false));
      EXPECT_EQ(accepted.state()[2u], flag_pending);
      ASSERT_TRUE(accepted.observe(identity++, {boundary_ribbon}, false));
      EXPECT_EQ(accepted.state()[2u], flag_owner | flag_target_valid);
      EXPECT_EQ(accepted.state()[4u], 1u);
      EXPECT_EQ(accepted.state()[20u], 1u);
      EXPECT_EQ(
        accepted.state()[31u],
        (1u << owner_kind_shift) | (1u << current_kind_shift)
      );

    }

    // One field row below the exact projected detector tolerance has no OCR8 ribbon authority,
    // even if all other topology metadata and the canonical cover are well formed. One rejection
    // fixture is sufficient because the generated helper's six-field values were exercised above.
    constexpr auto rejection_case = cases[1u];
    const auto rejection_roi = v2::subtitle_ocr_dynamic_roi(
      rejection_case.source_width,
      rejection_case.source_height,
      rejection_case.field_width,
      rejection_case.field_height
    );
    const auto rejection_minimum = v2::subtitle_ocr_ribbon_min_bottom(
      rejection_case.source_width,
      rejection_case.source_height,
      rejection_case.field_width,
      rejection_case.field_height
    );
    ASSERT_TRUE(static_cast<bool>(rejection_roi));
    ASSERT_TRUE(static_cast<bool>(rejection_minimum));
    ASSERT_GT(rejection_minimum.value, rejection_roi.top + 6u);
    const auto rejection_left = rejection_case.field_width / 20u;
    const auto rejection_right = rejection_case.field_width - rejection_left;
    const ocr_box_t below_boundary {
      {
        rejection_left,
        rejection_minimum.value - 7u,
        rejection_right,
        rejection_minimum.value - 1u,
      },
      {
        0u,
        rejection_minimum.value - 11u,
        rejection_case.field_width,
        rejection_case.field_height,
      },
      box_flag_ribbon,
      7u,
      4u,
    };
    slr12_warp_fixture_t rejected(
      rejection_case.field_width,
      rejection_case.field_height,
      rejection_case.source_width,
      rejection_case.source_height
    );
    std::string error;
    ASSERT_TRUE(rejected.initialize(error)) << error;
    ASSERT_TRUE(rejected.observe(identity++, {below_boundary}, false));
    EXPECT_EQ(rejected.state()[2u], 0u);
    EXPECT_EQ(rejected.state()[4u], 0u);
    EXPECT_EQ(rejected.state()[12u], 0u);
    EXPECT_EQ(rejected.state()[20u], 0u);
    EXPECT_TRUE(rejected.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr12GpuTest, DynamicMaximumWidthAndMalformedRoiFailFlat) {
    constexpr std::uint32_t portrait_field_width = 434u;
    constexpr std::uint32_t portrait_field_height = 770u;
    constexpr std::uint32_t portrait_source_width = 1080u;
    constexpr std::uint32_t portrait_source_height = 1920u;
    constexpr auto dynamic_roi = v2::subtitle_ocr_dynamic_roi(
      portrait_source_width,
      portrait_source_height,
      portrait_field_width,
      portrait_field_height
    );
    static_assert(static_cast<bool>(dynamic_roi));

    // floor(0.9 * 434) == 390; this otherwise valid 400-cell horizontal line must not acquire.
    slr12_warp_fixture_t overwide_fixture(
      portrait_field_width,
      portrait_field_height,
      portrait_source_width,
      portrait_source_height
    );
    std::string error;
    ASSERT_TRUE(overwide_fixture.initialize(error)) << error;
    const line_box_t overwide {
      17u, dynamic_roi.top + 35u, 417u, dynamic_roi.top + 45u
    };
    ASSERT_TRUE(overwide_fixture.observe(200u, {overwide}, false));
    ASSERT_TRUE(overwide_fixture.observe(201u, {overwide}, false));
    EXPECT_EQ(overwide_fixture.state()[2u], 0u);
    EXPECT_TRUE(overwide_fixture.output_is_exact_base());

    // Even when OCR8 repeats the same malformed bounds, SLR12 independently derives the safe ROI
    // from source/field geometry and rejects the record instead of trusting its header.
    const auto wrong_top = dynamic_roi.top + 1u;
    slr12_warp_fixture_t malformed_fixture(
      portrait_field_width,
      portrait_field_height,
      portrait_source_width,
      portrait_source_height,
      wrong_top,
      dynamic_roi.bottom
    );
    error.clear();
    ASSERT_TRUE(malformed_fixture.initialize(error)) << error;
    const line_box_t line {40u, wrong_top + 35u, 360u, wrong_top + 45u};
    ASSERT_TRUE(malformed_fixture.observe(210u, {line}, false));
    EXPECT_EQ(malformed_fixture.state()[0u], slr_schema);
    EXPECT_EQ(malformed_fixture.state()[1u], slr_tag);
    EXPECT_EQ(malformed_fixture.state()[2u], 0u);
    EXPECT_EQ(malformed_fixture.state()[4u], 0u);
    EXPECT_EQ(malformed_fixture.state()[20u], 0u);
    EXPECT_EQ(
      malformed_fixture.condition_args(),
      (std::vector<std::uint32_t> {0u, 0u, 0u})
    );
    EXPECT_TRUE(malformed_fixture.output_is_exact_base());
  }
}  // namespace

#else

TEST(HostSbsSubtitleSlr12GpuTest, WindowsOnly) {
  GTEST_SKIP() << "D3D11 WARP is Windows-only";
}

#endif
