/**
 * @file tests/unit/test_host_sbs_subtitle_locator.cpp
 * @brief Deterministic D3D11 coverage for direct current OCR8 and the adaptive shared UI plane.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/depth_coordinate_v2.h"
  #include "src/generated/sbs_adaptive_state_contract.h"
  #include "src/host_sbs_shader_cache.h"
  #include "src/platform/windows/sbs_debug_dump.h"

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstddef>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <limits>
  #include <iostream>
  #include <numeric>
  #include <optional>
  #include <string>
  #include <utility>
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
  constexpr std::uint32_t state_words = v2::subtitle_plane_state_word_count;
  constexpr std::uint32_t ocr_schema = v2::subtitle_ocr_record_schema;
  constexpr std::uint32_t ocr_tag = v2::subtitle_ocr_record_tag;
  constexpr std::uint32_t plane_schema = v2::subtitle_plane_state_schema;
  constexpr std::uint32_t plane_tag = v2::subtitle_plane_state_tag;
  constexpr std::uint32_t cut_tag = sbs_adaptive_state::cut_contract_tag;
  constexpr std::uint32_t final_box_offset = v2::subtitle_ocr_final_box_offset;

  static_assert([] {
    for (const auto &spec :
         models::host_sbs_shader_cache::parallax_v2_producer_specs) {
      if (spec.filename.find("condition_validate_test") != std::string_view::npos) {
        return false;
      }
    }
    return true;
  }(), "the adversarial condition validator must remain outside the production closure");

  constexpr std::uint32_t box_flag_ribbon = v2::subtitle_ocr_box_flag_ribbon;
  constexpr float contour_distance_infinity = std::numeric_limits<float>::infinity();

  // Independent authored input to the proposal writer, not a CPU renderer or a claim about
  // RGB classification. A field seed is placed at its source-coordinate node; the actual
  // five-native-center distance producer has a separate exhaustive native test suite.
  std::vector<float> authored_mask_distances(
    const std::vector<std::uint32_t> &mask,
    const std::uint32_t width,
    const std::uint32_t height,
    const tensor_content_t &content,
    const std::uint32_t source_width,
    const std::uint32_t source_height
  ) {
    std::vector<float> distance(mask.size(), contour_distance_infinity);
    if (content[0u] >= content[2u] || content[1u] >= content[3u] ||
        content[2u] > width || content[3u] > height) return distance;
    const double sx = static_cast<double>(source_width) / (content[2u] - content[0u]);
    const double sy = static_cast<double>(source_height) / (content[3u] - content[1u]);
    const auto transform = [](const std::vector<double> &input, const double scale_squared) {
      const auto n = input.size();
      std::vector<double> output(n, std::numeric_limits<double>::infinity()), starts(n + 1u);
      std::vector<std::size_t> rows(n);
      std::size_t count = 0u;
      for (std::size_t row = 0u; row < n; ++row) {
        if (!std::isfinite(input[row])) continue;
        double start = -std::numeric_limits<double>::infinity();
        while (count != 0u) {
          const auto prior = rows[count - 1u];
          start = 0.5 * static_cast<double>(row + prior) +
            (input[row] - input[prior]) / (2.0 * scale_squared * static_cast<double>(row - prior));
          if (start > starts[count - 1u]) break;
          --count;
        }
        rows[count] = row;
        starts[count] = count == 0u ? -std::numeric_limits<double>::infinity() : start;
        ++count;
      }
      if (count == 0u) return output;
      starts[count] = std::numeric_limits<double>::infinity();
      std::size_t selected = 0u;
      for (std::size_t query = 0u; query < n; ++query) {
        while (selected + 1u < count && starts[selected + 1u] < static_cast<double>(query)) ++selected;
        const auto delta = static_cast<double>(query) - rows[selected];
        output[query] = input[rows[selected]] + scale_squared * delta * delta;
      }
      return output;
    };
    const auto cw = content[2u] - content[0u], ch = content[3u] - content[1u];
    std::vector<double> horizontal(static_cast<std::size_t>(cw) * ch);
    for (std::uint32_t y = 0u; y < ch; ++y) {
      std::vector<double> row(cw, std::numeric_limits<double>::infinity());
      for (std::uint32_t x = 0u; x < cw; ++x) {
        if (mask[static_cast<std::size_t>(y + content[1u]) * width + x + content[0u]] != 0u) row[x] = 0.0;
      }
      const auto result = transform(row, sx * sx);
      std::copy(result.begin(), result.end(), horizontal.begin() + static_cast<std::size_t>(y) * cw);
    }
    for (std::uint32_t x = 0u; x < cw; ++x) {
      std::vector<double> column(ch);
      for (std::uint32_t y = 0u; y < ch; ++y) column[y] = horizontal[static_cast<std::size_t>(y) * cw + x];
      const auto result = transform(column, sy * sy);
      for (std::uint32_t y = 0u; y < ch; ++y) {
        distance[static_cast<std::size_t>(y + content[1u]) * width + x + content[0u]] = static_cast<float>(result[y]);
      }
    }
    return distance;
  }

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
    float coordinate_reserved0;
    float coordinate_reserved1;
    float requested_gain;
    float max_horizontal_slope;
    float direct_container_limit;
    float convergence_curve_default;
    std::uint32_t joint_plane_mode;
    std::array<std::uint32_t, 3u> reserved;
    std::uint32_t joint_observation_timestamp_low;
    std::uint32_t joint_observation_timestamp_high;
    std::array<std::uint32_t, 2u> observation_reserved;
  };

  struct subtitle_constants_t {
    std::array<std::uint32_t, 4u> field;
    std::array<std::uint32_t, 4u> source;
    std::array<std::uint32_t, 4u> frame;
    tensor_content_t tensor_content;
    std::array<std::uint32_t, 4u> observation;
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
    std::uint32_t score_bits = std::bit_cast<std::uint32_t>(0.9f);

    constexpr ocr_box_t(const line_box_t box) :
        core(box), cover(box), kind_flags(0u), island_count(1u), structural_gap_count(0u) {}

    constexpr ocr_box_t(
      const line_box_t core_box,
      const line_box_t cover_box,
      const std::uint32_t flags,
      const std::uint32_t islands,
      const std::uint32_t structural_gaps,
      const std::uint32_t exact_score_bits = std::bit_cast<std::uint32_t>(0.9f)
    ) :
        core(core_box),
        cover(cover_box),
        kind_flags(flags),
        island_count(islands),
        structural_gap_count(structural_gaps),
        score_bits(exact_score_bits) {}
  };

  static_assert(sizeof(depth_constants_t) == 64u);
  static_assert(sizeof(v2_constants_t) == 64u);
  static_assert(sizeof(v2_constants_t) == sizeof(v2::constants_t));
  static_assert(sizeof(subtitle_constants_t) == 80u);

  bool create_compute_shader(
    ID3D11Device *device,
    const models::host_sbs_shader_cache::source_snapshot_t &sources,
    const models::host_sbs_shader_cache::shader_spec &spec,
    ComPtr<ID3D11ComputeShader> &shader,
    std::string &error
  ) {
    const auto bytecode = models::host_sbs_shader_cache::get(sources, spec);
    if (!bytecode || bytecode->empty()) {
      error = std::string("authenticated shader compilation failed for ") +
              std::string(spec.filename) + ":" + std::string(spec.entrypoint);
      return false;
    }
    const auto create_status = device->CreateComputeShader(
      bytecode->data(), bytecode->size(), nullptr, &shader
    );
    if (FAILED(create_status)) {
      error = std::string("CreateComputeShader failed for ") +
              std::string(spec.entrypoint);
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

  class subtitle_plane_fixture_t {
   public:
    explicit subtitle_plane_fixture_t(
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

    bool initialize(std::string &error, const D3D_DRIVER_TYPE driver = D3D_DRIVER_TYPE_WARP) {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      if (FAILED(D3D11CreateDevice(
            nullptr,
            driver,
            nullptr,
            0u,
            requested,
            static_cast<UINT>(std::size(requested)),
            D3D11_SDK_VERSION,
            &device_,
            &actual,
            &context_
          )) || actual < D3D_FEATURE_LEVEL_11_0) return false;
      if (driver == D3D_DRIVER_TYPE_HARDWARE) {
        ComPtr<IDXGIDevice> dxgi_device;
        ComPtr<IDXGIAdapter> adapter;
        DXGI_ADAPTER_DESC description {};
        if (FAILED(device_.As(&dxgi_device)) ||
            FAILED(dxgi_device->GetAdapter(&adapter)) ||
            FAILED(adapter->GetDesc(&description))) return false;
        std::wcout << L"Native dump reproduction adapter: " << description.Description << L'\n';
      }

      namespace shader_cache = models::host_sbs_shader_cache;
      const auto production_sources = shader_cache::snapshot_sources(
        SUNSHINE_SHADERS_DIR, shader_cache::parallax_v2_producer_specs
      );
      static constexpr shader_cache::shader_spec condition_validator_spec {
        "test/host_sbs_subtitle_locator_condition_validate_test_cs.hlsl",
        "condition_validate_test_main",
        "cs_5_0",
      };
      const std::array validator_specs {condition_validator_spec};
      const auto validator_sources = shader_cache::snapshot_sources(
        SUNSHINE_SHADERS_DIR, validator_specs
      );
      if (!create_compute_shader(
            device_.Get(), production_sources,
            shader_cache::host_sbs_subtitle_locator_resolve, resolve_, error
          ) ||
          !create_compute_shader(
            device_.Get(),
            validator_sources,
            condition_validator_spec,
            condition_validate_test_,
            error
          ) ||
          !create_compute_shader(
            device_.Get(), production_sources,
            shader_cache::host_sbs_subtitle_condition, condition_, error
          )) {
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
          )) return false;

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

      const auto geometry_initial = v2::state_initial_words;
      if (!create_structured_buffer(
            device_.Get(), geometry_initial.data(), sizeof(geometry_initial),
            4u * sizeof(std::uint32_t), D3D11_BIND_SHADER_RESOURCE,
            geometry_state_buffer_, &geometry_state_srv_, nullptr)) return false;

      base_.assign(static_cast<std::size_t>(field_width_) * field_height_, 0.03f);
      const auto cell_scale = fixture_field_cell_scale();
      const auto sample_top = std::min(roi_top_ + 5u * cell_scale, field_height_);
      const auto sample_bottom = std::min(roi_top_ + 32u * cell_scale, field_height_);
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

      texture_desc.Format = DXGI_FORMAT_R32_FLOAT;
      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      if (FAILED(device_->CreateTexture2D(&texture_desc, nullptr, &contour_distance_texture_)) ||
          FAILED(device_->CreateShaderResourceView(contour_distance_texture_.Get(), nullptr, &contour_distance_srv_))) return false;
      if (!create_structured_buffer(
            device_.Get(), contour_identity_.data(), sizeof(contour_identity_), sizeof(std::uint32_t),
            D3D11_BIND_SHADER_RESOURCE, contour_identity_buffer_, &contour_identity_srv_, nullptr)) return false;

      const std::vector<float> zero_probability(v2::subtitle_ocr_output_width * v2::subtitle_ocr_output_height, 0.f);
      if (!create_structured_buffer(device_.Get(), zero_probability.data(), zero_probability.size() * sizeof(float),
            sizeof(float), D3D11_BIND_SHADER_RESOURCE, probability_buffer_, &probability_srv_, nullptr) ||
          !create_structured_buffer(device_.Get(), zero_probability.data(), zero_probability.size() * sizeof(float),
            sizeof(float), D3D11_BIND_SHADER_RESOURCE, previous_probability_buffer_, &previous_probability_srv_, nullptr) ||
          !create_structured_buffer(device_.Get(), probability_history_identity_.data(), sizeof(probability_history_identity_),
            sizeof(std::uint32_t), D3D11_BIND_SHADER_RESOURCE, probability_history_buffer_, &probability_history_srv_, nullptr)) return false;

      texture_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      if (FAILED(device_->CreateTexture2D(&texture_desc, nullptr, &gaussian_weight_texture_)) ||
          FAILED(device_->CreateUnorderedAccessView(gaussian_weight_texture_.Get(), nullptr, &gaussian_weight_uav_))) return false;
      const std::uint32_t disabled = 0u;
      if (!create_structured_buffer(device_.Get(), &disabled, sizeof(disabled), sizeof(disabled),
            D3D11_BIND_UNORDERED_ACCESS, gaussian_control_buffer_, nullptr, &gaussian_control_uav_)) return false;

      const depth_constants_t depth_constants {
        field_width_, field_height_, 0u, 0.0f, 0.0f, 0u, 0.0f, 0.0f, 0.0f,
        tensor_content_, {}
      };
      v2_constants_ = v2_constants_t {
        0.04f, 0.0001f, 0.0f, 0.0f, 1.0f, 0.5f, 0.04f, 0.0f,
        v2::adaptive_policy_id, {}, 0u, 0u, {}
      };
      const subtitle_constants_t subtitle_constants {
        {field_width_, field_height_, roi_top_, roi_bottom_},
        {source_width_, source_height_, 1u, 0u},
        {0u, 0u, 0u, 0u},
        tensor_content_,
        {},
      };
      return create_constant_buffer(
               device_.Get(), &depth_constants, sizeof(depth_constants), depth_cb_) &&
             create_constant_buffer(
               device_.Get(), &v2_constants_, sizeof(v2_constants_), v2_cb_) &&
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
      const bool submitted = true,
      const bool = true,
      std::uint64_t observation_timestamp_us = std::numeric_limits<std::uint64_t>::max()
    ) {
      // Current-authority tests use an explicit authored microsecond cadence; rate/dwell tests pass
      // their source timestamps through observe_at. Duplicate identity retains the same clock.
      if (observation_timestamp_us == std::numeric_limits<std::uint64_t>::max()) {
        observation_timestamp_us = identity;
      }
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
      record[7u] = static_cast<std::uint32_t>(analysis_generation_);
      record[8u] = static_cast<std::uint32_t>(analysis_generation_ >> 32u);
      record[9u] = source_width_;
      record[10u] = source_height_;
      record[11u] = field_width_;
      record[12u] = field_height_;
      record[13u] = roi_top_;
      record[14u] = roi_bottom_;
      if (authoritative) {
        for (std::size_t slot = 0u; slot < std::min<std::size_t>(boxes.size(), v2::subtitle_ocr_final_box_capacity); ++slot) {
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
      ocr_record_ = record;

      const subtitle_constants_t subtitle_constants {
        {field_width_, field_height_, roi_top_, roi_bottom_},
        {
          source_width_,
          source_height_,
          submitted ? 1u : 0u,
          reset ? 1u : 0u,
        },
        {
          static_cast<std::uint32_t>(identity),
          static_cast<std::uint32_t>(identity >> 32u),
          static_cast<std::uint32_t>(analysis_generation_),
          static_cast<std::uint32_t>(analysis_generation_ >> 32u),
        },
        tensor_content_,
        {static_cast<std::uint32_t>(observation_timestamp_us),
         static_cast<std::uint32_t>(observation_timestamp_us >> 32u), 0u, 0u},
      };
      context_->UpdateSubresource(
        subtitle_cb_.Get(), 0u, nullptr, &subtitle_constants, 0u, 0u
      );
      if (!override_joint_observation_) {
        v2_constants_.joint_observation_timestamp_low =
          static_cast<std::uint32_t>(observation_timestamp_us);
        v2_constants_.joint_observation_timestamp_high =
          static_cast<std::uint32_t>(observation_timestamp_us >> 32u);
        context_->UpdateSubresource(v2_cb_.Get(), 0u, nullptr, &v2_constants_, 0u, 0u);
      }
      upload_geometry_publication(observation_timestamp_us);
      ID3D11Buffer *constant_buffers[] = {depth_cb_.Get(), v2_cb_.Get(), subtitle_cb_.Get()};
      context_->CSSetConstantBuffers(0u, 3u, constant_buffers);

      context_->CSSetShader(resolve_.Get(), nullptr, 0u);
      if (probability_witness_enabled_) {
        context_->UpdateSubresource(probability_buffer_.Get(), 0u, nullptr, probability_values_.data(), 0u, 0u);
        context_->UpdateSubresource(previous_probability_buffer_.Get(), 0u, nullptr, previous_probability_values_.data(), 0u, 0u);
        context_->UpdateSubresource(probability_history_buffer_.Get(), 0u, nullptr, probability_history_identity_.data(), 0u, 0u);
      }
      std::array<ID3D11ShaderResourceView *, 10u> resolve_srvs {
        geometry_available_ ? geometry_state_srv_.Get() : nullptr,
        cut_srv_.Get(), base_srv_.Get(), nullptr, nullptr, nullptr,
        probability_witness_enabled_ ? probability_history_srv_.Get() : nullptr, ocr_srv_.Get(),
        probability_witness_enabled_ && current_probability_available_ ? probability_srv_.Get() : nullptr,
        probability_witness_enabled_ && previous_probability_available_ ? previous_probability_srv_.Get() : nullptr,
      };
      context_->CSSetShaderResources(0u, resolve_srvs.size(), resolve_srvs.data());
      std::array<ID3D11UnorderedAccessView *, 3u> resolve_uavs {
        state_uav_.Get(), nullptr, condition_params_uav_.Get()
      };
      context_->CSSetUnorderedAccessViews(2u, resolve_uavs.size(), resolve_uavs.data(), nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind();

      if (!author_current_contour(subtitle_constants)) return false;
      if (!dispatch_conditioner()) return false;

      const bool complete = read_buffer(device_.Get(), context_.Get(), state_buffer_.Get(), state_) &&
             read_buffer(
               device_.Get(), context_.Get(), condition_params_buffer_.Get(), condition_params_) &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
      if (complete && probability_witness_enabled_) {
        // Authored completed detector evidence isolates the real GPU resolver. Temporal pixel
        // advancement and native inference remain separate tests/evidence; no count is fabricated.
        previous_probability_values_ = probability_values_;
        probability_history_identity_ = contour_identity_;
      }
      return complete;
    }

    bool condition_only() {
      if (!dispatch_condition_validator_for_test()) return false;
      if (!dispatch_conditioner()) return false;
      return read_buffer(
               device_.Get(), context_.Get(), condition_params_buffer_.Get(), condition_params_) &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    void set_analysis_generation(const std::uint64_t generation) { analysis_generation_ = generation; }

    void enable_probability_witness(const float value) {
      probability_witness_enabled_ = true;
      probability_values_.assign(v2::subtitle_ocr_output_width * v2::subtitle_ocr_output_height, value);
      previous_probability_values_ = probability_values_;
      probability_history_identity_.fill(0u);
    }
    void set_probability(const float value) {
      std::fill(probability_values_.begin(), probability_values_.end(), value);
    }
    void set_previous_probability(const float value) {
      std::fill(previous_probability_values_.begin(), previous_probability_values_.end(), value);
    }
    void set_probability_word(const std::size_t word, const float value, const bool previous) {
      auto &values = previous ? previous_probability_values_ : probability_values_;
      ASSERT_LT(word, values.size());
      values[word] = value;
    }
    void set_probability_availability(const bool current, const bool previous) {
      current_probability_available_ = current;
      previous_probability_available_ = previous;
    }
    void overwrite_probability_history_word(const std::size_t word, const std::uint32_t value) {
      ASSERT_LT(word, probability_history_identity_.size());
      probability_history_identity_[word] = value;
    }
    // Authored detector patterns only; native sampling and reset decisions run in the shader.
    void set_probability_halves(const float left, const float right) {
      for (std::uint32_t y = 0u; y < v2::subtitle_ocr_output_height; ++y) {
        for (std::uint32_t x = 0u; x < v2::subtitle_ocr_output_width; ++x) {
          probability_values_[std::size_t(y) * v2::subtitle_ocr_output_width + x] =
            x < v2::subtitle_ocr_output_width / 2u ? left : right;
        }
      }
    }
    void set_probability_quarters(
      const float first, const float second, const float third, const float fourth
    ) {
      const std::array values {first, second, third, fourth};
      for (std::uint32_t y = 0u; y < v2::subtitle_ocr_output_height; ++y) {
        for (std::uint32_t x = 0u; x < v2::subtitle_ocr_output_width; ++x) {
          probability_values_[std::size_t(y) * v2::subtitle_ocr_output_width + x] =
            values[x * values.size() / v2::subtitle_ocr_output_width];
        }
      }
    }

    bool overwrite_ocr_word(const std::size_t index, const std::uint32_t value) {
      if (index >= ocr_record_.size()) return false;
      ocr_record_[index] = value;
      context_->UpdateSubresource(ocr_buffer_.Get(), 0u, nullptr, ocr_record_.data(), 0u, 0u);
      return dispatch_conditioner() && read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    void set_cut(const std::uint32_t scene_epoch, const bool pulse) {
      scene_epoch_ = scene_epoch;
      cut_pulse_ = pulse;
    }

    void set_adaptive_policy_words(
      const std::uint32_t mode,
      const std::uint32_t padding = 0u,
      const float requested_gain = 1.0f
    ) {
      v2_constants_ = v2_constants_t {
        0.04f, 0.0001f, 0.0f, 0.0f, requested_gain, 0.5f, 0.04f, 0.0f,
        mode, {padding, 0u, 0u}, 0u, 0u, {}
      };
      override_joint_observation_ = false;
      context_->UpdateSubresource(v2_cb_.Get(), 0u, nullptr, &v2_constants_, 0u, 0u);
    }

    void set_geometry_ready(const bool ready) { geometry_ready_ = ready; }
    void set_geometry_available(const bool available) { geometry_available_ = available; }
    void override_geometry_clock(const std::optional<std::uint64_t> timestamp) {
      geometry_clock_override_ = timestamp;
    }

    void set_display_guard(const float limit) {
      v2_constants_.direct_container_limit = limit;
      context_->UpdateSubresource(v2_cb_.Get(), 0u, nullptr, &v2_constants_, 0u, 0u);
    }

    void override_joint_observation(
      const std::uint64_t timestamp_us,
      const std::uint32_t reserved = 0u
    ) {
      override_joint_observation_ = true;
      v2_constants_.joint_observation_timestamp_low = static_cast<std::uint32_t>(timestamp_us);
      v2_constants_.joint_observation_timestamp_high = static_cast<std::uint32_t>(timestamp_us >> 32u);
      v2_constants_.observation_reserved = {reserved, 0u};
      context_->UpdateSubresource(v2_cb_.Get(), 0u, nullptr, &v2_constants_, 0u, 0u);
    }

    bool condition_with_parameter_word(const std::size_t index, const std::uint32_t value) {
      if (index >= condition_params_.size()) return false;
      condition_params_[index] = value;
      context_->UpdateSubresource(
        condition_params_buffer_.Get(), 0u, nullptr, condition_params_.data(), 0u, 0u);
      return dispatch_conditioner() &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    bool observe_at(
      const std::uint64_t identity,
      const std::vector<ocr_box_t> &boxes,
      const std::uint64_t timestamp_us,
      const bool reset = false,
      const bool authoritative = true,
      const bool submitted = true
    ) {
      return observe(identity, boxes, reset, authoritative, false, false,
                     submitted, true, timestamp_us);
    }

    const std::vector<std::uint32_t> &state() const { return state_; }
    const std::vector<std::uint32_t> &ocr_record() const { return ocr_record_; }
    const std::vector<std::uint32_t> &condition_params() const { return condition_params_; }
    const std::vector<float> &base() const { return base_; }
    const std::vector<float> &output() const { return output_; }
    const std::vector<float> &contour_distances() const { return contour_distances_; }
    const std::vector<float> &gaussian_weights() const { return gaussian_weight_; }
    std::uint32_t gaussian_control() const { return gaussian_control_.empty() ? 2u : gaussian_control_[0u]; }
    bool condition_with_distance_field(const std::vector<float> &q) {
      if (q.size() != contour_distances_.size()) return false;
      contour_distances_ = q;
      context_->UpdateSubresource(contour_distance_texture_.Get(), 0u, nullptr, q.data(),
        field_width_ * static_cast<UINT>(sizeof(float)), 0u);
      return dispatch_conditioner() && read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }
    const std::array<std::uint32_t, 32u> &contour_identity() const { return contour_identity_; }

    bool set_authored_contour_mask(const std::vector<std::uint32_t> &mask) {
      if (mask.size() != static_cast<std::size_t>(field_width_) * field_height_) return false;
      authored_contour_mask_ = mask;
      upload_authored_distances(mask);
      return true;
    }

    bool condition_with_contour_identity_word(const std::size_t index, const std::uint32_t value) {
      if (index >= contour_identity_.size()) return false;
      contour_identity_[index] = value;
      context_->UpdateSubresource(contour_identity_buffer_.Get(), 0u, nullptr, contour_identity_.data(), 0u, 0u);
      return dispatch_conditioner() && read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    bool set_base_field(const std::vector<float> &values) {
      if (values.size() != base_.size()) return false;
      base_ = values;
      context_->UpdateSubresource(base_texture_.Get(), 0u, nullptr, base_.data(),
        field_width_ * static_cast<UINT>(sizeof(float)), 0u);
      return true;
    }

    bool overwrite_state_word(const std::size_t index, const std::uint32_t value) {
      if (index >= state_.size()) return false;
      state_[index] = value;
      context_->UpdateSubresource(
        state_buffer_.Get(), 0u, nullptr, state_.data(), 0u, 0u
      );
      return true;
    }

    void poison_condition_params(const std::uint32_t value) {
      std::vector<std::uint32_t> poisoned(v2::subtitle_condition_param_word_count, value);
      context_->UpdateSubresource(
        condition_params_buffer_.Get(), 0u, nullptr, poisoned.data(), 0u, 0u
      );
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
      const auto cell_scale = fixture_field_cell_scale();
      const auto sample_top = std::min(roi_top_ + 5u * cell_scale, field_height_);
      const auto sample_bottom = std::min(roi_top_ + 32u * cell_scale, field_height_);
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
      const auto center = 0.5f * static_cast<float>(line.left + line.right - 1u);
      set_background_probe_rows(center, line.top, outer_value, inner_value);
    }

    void set_background_probe_rows(
      const float center,
      const std::uint32_t cover_top,
      const float outer_value,
      const float inner_value
    ) {
      const auto cell_scale = fixture_field_cell_scale();
      const auto outer_y = std::clamp(
        cover_top >= 10u * cell_scale ? cover_top - 10u * cell_scale : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto inner_y = std::clamp(
        cover_top >= 4u * cell_scale ? cover_top - 4u * cell_scale : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      for (std::uint32_t sample = 0u; sample < 16u; ++sample) {
        const auto x = static_cast<std::uint32_t>(std::clamp(
          std::floor(
            center - 30.0f * static_cast<float>(cell_scale) +
            4.0f * static_cast<float>(cell_scale) * static_cast<float>(sample) + 0.5f
          ),
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
      const auto cell_scale = fixture_field_cell_scale();
      const auto outer_y = std::clamp(
        line.top >= 10u * cell_scale ? line.top - 10u * cell_scale : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto inner_y = std::clamp(
        line.top >= 4u * cell_scale ? line.top - 4u * cell_scale : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto center = 0.5f * static_cast<float>(line.left + line.right - 1u);
      for (std::uint32_t sample = 0u; sample < 16u; ++sample) {
        const auto x = static_cast<std::uint32_t>(std::clamp(
          std::floor(
            center - 30.0f * static_cast<float>(cell_scale) +
            4.0f * static_cast<float>(cell_scale) * static_cast<float>(sample) + 0.5f
          ),
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
      const auto center = 0.5f * static_cast<float>(line.left + line.right - 1u);
      set_background_probe_alternating(center, line.top, first_value, second_value);
    }

    void set_background_probe_alternating(
      const float center,
      const std::uint32_t cover_top,
      const float first_value,
      const float second_value
    ) {
      const auto cell_scale = fixture_field_cell_scale();
      const auto outer_y = std::clamp(
        cover_top >= 10u * cell_scale ? cover_top - 10u * cell_scale : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto inner_y = std::clamp(
        cover_top >= 4u * cell_scale ? cover_top - 4u * cell_scale : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      for (std::uint32_t sample = 0u; sample < 16u; ++sample) {
        const auto x = static_cast<std::uint32_t>(std::clamp(
          std::floor(
            center - 30.0f * static_cast<float>(cell_scale) +
            4.0f * static_cast<float>(cell_scale) * static_cast<float>(sample) + 0.5f
          ),
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

    std::uint32_t fixture_field_cell_scale() const {
      return std::max(
        1u,
        v2::subtitle_ocr_field_cell_scale(field_width_, field_height_)
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

  private:
    void upload_authored_distances(const std::vector<std::uint32_t> &mask) {
      contour_distances_ = authored_mask_distances(mask, field_width_, field_height_, tensor_content_, source_width_, source_height_);
      context_->UpdateSubresource(contour_distance_texture_.Get(), 0u, nullptr, contour_distances_.data(),
        field_width_ * static_cast<UINT>(sizeof(float)), 0u);
    }

    bool author_current_contour(const subtitle_constants_t &constants) {
      if (!read_buffer(device_.Get(), context_.Get(), state_buffer_.Get(), state_) ||
          !read_buffer(device_.Get(), context_.Get(), condition_params_buffer_.Get(), condition_params_)) return false;
      if (!probability_witness_enabled_) {
        EXPECT_EQ(state_[v2::subtitle_plane_history_reset_mask_word], 0u)
          << "null current P/history fixtures supply no probability-change witness";
      }
      const bool current_valid = state_[v2::subtitle_plane_flags_word] ==
        v2::subtitle_plane_current_valid_flag && condition_params_[2u] != 0u &&
        condition_params_[0u] == v2::subtitle_condition_param_schema &&
        condition_params_[1u] == v2::subtitle_condition_param_tag &&
        state_[v2::subtitle_plane_current_slot_mask_word] ==
          ((1u << condition_params_[2u]) - 1u);
      std::vector<std::uint32_t> mask(static_cast<std::size_t>(field_width_) * field_height_, 0u);
      if (current_valid && authored_contour_mask_) {
        mask = *authored_contour_mask_;
      } else if (current_valid) {
        // Authored dense seeds isolate current OCR/plane and proposal contracts in this fixture.
        // They do not reproduce or qualify the temporal RGB classifier or real glyph detection.
        for (std::uint32_t slot = 0u; slot < condition_params_[2u]; ++slot) {
          const auto offset = final_box_offset + slot * v2::subtitle_ocr_box_word_count;
          for (std::uint32_t y = ocr_record_[offset + 1u]; y < ocr_record_[offset + 3u]; ++y) {
            std::fill(mask.begin() + static_cast<std::size_t>(y) * field_width_ + ocr_record_[offset],
              mask.begin() + static_cast<std::size_t>(y) * field_width_ + ocr_record_[offset + 2u], 1u);
          }
        }
      }
      upload_authored_distances(mask);
      contour_identity_.fill(0u);
      contour_identity_[0u] = v2::subtitle_contour_schema;
      contour_identity_[1u] = v2::subtitle_contour_history_tag;
      contour_identity_[2u] = current_valid ? 1u : 0u;
      contour_identity_[3u] = current_valid ? scene_epoch_ : 0u;
      std::copy(constants.frame.begin(), constants.frame.end(), contour_identity_.begin() + 4u);
      const auto crop_height = ceil_ocr_crop_height(source_width_, source_height_);
      contour_identity_[8u] = source_width_;
      contour_identity_[9u] = source_height_;
      contour_identity_[10u] = source_height_ - crop_height;
      contour_identity_[11u] = crop_height;
      std::copy(constants.field.begin(), constants.field.end(), contour_identity_.begin() + 12u);
      std::copy(tensor_content_.begin(), tensor_content_.end(), contour_identity_.begin() + 16u);
      contour_identity_[20u] = constants.observation[0u];
      contour_identity_[21u] = constants.observation[1u];
      contour_identity_[22u] = v2::subtitle_contour_recipe;
      contour_identity_[v2::subtitle_contour_history_snapshot_valid_word] = current_valid ? 1u : 0u;
      context_->UpdateSubresource(contour_identity_buffer_.Get(), 0u, nullptr, contour_identity_.data(), 0u, 0u);
      return true;
    }

    void upload_geometry_publication(const std::uint64_t timestamp) {
      // The fixture supplies a complete authenticated current Base publication without running
      // a model. Production binds its already-published raw-coordinate state at the same slot.
      auto words = v2::state_initial_words;
      const auto now = geometry_clock_override_.value_or(timestamp);
      const auto seed = now == 0u ? 1u : now;
      const auto put_float = [&words](const std::size_t index, const float value) {
        words[index] = std::bit_cast<std::uint32_t>(value);
      };
      const auto put_clock = [&words](const std::size_t index, const std::uint64_t value) {
        words[index] = static_cast<std::uint32_t>(value);
        words[index + 1u] = static_cast<std::uint32_t>(value >> 32u);
      };
      put_float(v2::center, 0.0f);
      put_float(v2::inverse_scale, 1.0f / v2_constants_.raw_coordinate_scale);
      put_float(v2::convergence_curve, 0.0f);
      put_float(v2::container_scale, 1.0f);
      words[v2::calibration_revision] = 1u;
      const bool ready = geometry_ready_ && now != 0u;
      put_float(v2::frame_valid, ready ? 1.0f : 0.0f);
      words[v2::confirmed_cut_count] = scene_epoch_;
      words[v2::contract_tag_bits] = v2::contract_tag;
      words[v2::renderer_authorization_bits] = ready ? v2::contract_tag : 0u;
      words[v2::joint_plane_mode_bits] = v2::adaptive_policy_id;
      put_clock(v2::gain_last_observation_low, seed);
      words[v2::gain_clock_armed] = ready ? 1u : 0u;
      words[v2::gain_seed_count] = 1u;
      put_float(v2::gain_target_zero, 0.0f);
      put_float(v2::gain_target_inverse_scale,
        ready ? 1.0f / v2_constants_.raw_coordinate_scale : 0.0f);
      put_float(v2::gain_target_nearest, ready ? v2_constants_.raw_coordinate_scale : 0.0f);
      put_float(v2::gain_display_limit, v2::direct_container_limit);
      put_clock(v2::gain_seed_first_low, seed);
      put_clock(v2::gain_seed_last_low, seed);
      put_float(v2::gain_seed_mean_nearest, v2_constants_.raw_coordinate_scale);
      put_float(v2::gain_seed_mean_zero, 0.0f);
      words[v2::camera_center_integrity_bits] =
        v2::camera_center_integrity_for_state_words(words);
      context_->UpdateSubresource(
        geometry_state_buffer_.Get(), 0u, nullptr, words.data(), 0u, 0u);
    }

    bool dispatch_condition_validator_for_test() {
      ID3D11Buffer *constant_buffers[] = {depth_cb_.Get(), v2_cb_.Get(), subtitle_cb_.Get()};
      context_->CSSetConstantBuffers(0u, 3u, constant_buffers);

      context_->CSSetShader(condition_validate_test_.Get(), nullptr, 0u);
      std::array<ID3D11ShaderResourceView *, 8u> validator_srvs {
        geometry_available_ ? geometry_state_srv_.Get() : nullptr,
        cut_srv_.Get(), nullptr, state_srv_.Get(), nullptr, nullptr, nullptr, ocr_srv_.Get()
      };
      context_->CSSetShaderResources(0u, validator_srvs.size(), validator_srvs.data());
      context_->CSSetUnorderedAccessViews(
        4u, 1u, condition_params_uav_.GetAddressOf(), nullptr
      );
      context_->Dispatch(1u, 1u, 1u);
      unbind();
      return true;
    }

    bool dispatch_conditioner() {
      ID3D11Buffer *constant_buffers[] = {depth_cb_.Get(), v2_cb_.Get(), subtitle_cb_.Get()};
      context_->CSSetShader(condition_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 3u, constant_buffers);
      std::array<ID3D11ShaderResourceView *, 8u> condition_srvs {
        geometry_available_ ? geometry_state_srv_.Get() : nullptr,
        cut_srv_.Get(), base_srv_.Get(), state_srv_.Get(), condition_params_srv_.Get(),
        contour_distance_srv_.Get(), contour_identity_srv_.Get(), ocr_srv_.Get()
      };
      context_->CSSetShaderResources(0u, condition_srvs.size(), condition_srvs.data());
      std::array<ID3D11UnorderedAccessView *, 3u> outputs {
        output_uav_.Get(), gaussian_control_uav_.Get(), gaussian_weight_uav_.Get()
      };
      context_->CSSetUnorderedAccessViews(3u, outputs.size(), outputs.data(), nullptr);
      context_->Dispatch(
        (field_width_ + 15u) / 16u,
        (field_height_ + 15u) / 16u,
        1u
      );
      unbind();
      return read_texture(device_.Get(), context_.Get(), gaussian_weight_texture_.Get(), gaussian_weight_) &&
        read_buffer(device_.Get(), context_.Get(), gaussian_control_buffer_.Get(), gaussian_control_);
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
      record[offset + 4u] = metadata.score_bits;
      record[offset + 5u] = metadata.kind_flags;
      record[offset + 6u] = metadata.island_count;
      record[offset + 7u] = metadata.structural_gap_count;
    }

    void unbind() {
      std::array<ID3D11ShaderResourceView *, 10u> null_srvs {};
      std::array<ID3D11UnorderedAccessView *, 6u> null_uavs {};
      context_->CSSetShaderResources(0u, null_srvs.size(), null_srvs.data());
      context_->CSSetUnorderedAccessViews(0u, null_uavs.size(), null_uavs.data(), nullptr);
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> resolve_;
    ComPtr<ID3D11ComputeShader> condition_validate_test_;
    ComPtr<ID3D11ComputeShader> condition_;
    ComPtr<ID3D11Buffer> state_buffer_;
    ComPtr<ID3D11ShaderResourceView> state_srv_;
    ComPtr<ID3D11UnorderedAccessView> state_uav_;
    ComPtr<ID3D11Buffer> condition_params_buffer_;
    ComPtr<ID3D11ShaderResourceView> condition_params_srv_;
    ComPtr<ID3D11UnorderedAccessView> condition_params_uav_;
    ComPtr<ID3D11Buffer> ocr_buffer_;
    ComPtr<ID3D11ShaderResourceView> ocr_srv_;
    ComPtr<ID3D11Buffer> geometry_state_buffer_;
    ComPtr<ID3D11ShaderResourceView> geometry_state_srv_;
    bool geometry_ready_ = true;
    bool geometry_available_ = true;
    std::optional<std::uint64_t> geometry_clock_override_;
    ComPtr<ID3D11Buffer> cut_buffer_;
    ComPtr<ID3D11ShaderResourceView> cut_srv_;
    ComPtr<ID3D11Texture2D> base_texture_;
    ComPtr<ID3D11ShaderResourceView> base_srv_;
    ComPtr<ID3D11Texture2D> contour_distance_texture_;
    ComPtr<ID3D11ShaderResourceView> contour_distance_srv_;
    ComPtr<ID3D11Buffer> contour_identity_buffer_;
    ComPtr<ID3D11ShaderResourceView> contour_identity_srv_;
    std::array<std::uint32_t, 32u> contour_identity_ {};
    ComPtr<ID3D11Buffer> probability_buffer_, previous_probability_buffer_, probability_history_buffer_;
    ComPtr<ID3D11ShaderResourceView> probability_srv_, previous_probability_srv_, probability_history_srv_;
    std::array<std::uint32_t, 32u> probability_history_identity_ {};
    std::vector<float> probability_values_, previous_probability_values_;
    bool probability_witness_enabled_ = false;
    bool current_probability_available_ = true;
    bool previous_probability_available_ = true;
    std::vector<float> contour_distances_;
    std::optional<std::vector<std::uint32_t>> authored_contour_mask_;
    ComPtr<ID3D11Texture2D> output_texture_;
    ComPtr<ID3D11UnorderedAccessView> output_uav_;
    ComPtr<ID3D11Buffer> depth_cb_;
    ComPtr<ID3D11Buffer> v2_cb_;
    ComPtr<ID3D11Buffer> subtitle_cb_;
    v2_constants_t v2_constants_ {};
    bool override_joint_observation_ = false;
    std::vector<std::uint32_t> state_;
    std::vector<std::uint32_t> ocr_record_;
    std::vector<std::uint32_t> condition_params_;
    std::vector<float> base_;
    std::vector<float> output_;
    ComPtr<ID3D11Texture2D> gaussian_weight_texture_;
    ComPtr<ID3D11UnorderedAccessView> gaussian_weight_uav_;
    ComPtr<ID3D11Buffer> gaussian_control_buffer_;
    ComPtr<ID3D11UnorderedAccessView> gaussian_control_uav_;
    std::vector<float> gaussian_weight_;
    std::vector<std::uint32_t> gaussian_control_;
    std::uint32_t field_width_;
    std::uint32_t field_height_;
    std::uint32_t source_width_;
    std::uint32_t source_height_;
    tensor_content_t tensor_content_ {};
    std::uint32_t roi_top_;
    std::uint32_t roi_bottom_;
    std::uint64_t analysis_generation_ = 1u;
    std::uint32_t scene_epoch_ = 0u;
    bool cut_pulse_ = false;
  };

  std::vector<std::uint32_t> authored_hollow_and_disjoint_contours() {
    std::vector<std::uint32_t> mask(static_cast<std::size_t>(field_width) * field_height, 0u);
    for (std::uint32_t y = 362u; y <= 382u; ++y) {
      for (std::uint32_t x = 270u; x <= 294u; ++x) {
        if (x == 270u || x == 294u || y == 362u || y == 382u) mask[y * field_width + x] = 1u;
      }
    }
    for (std::uint32_t y = 369u; y <= 376u; ++y) mask[y * field_width + 500u] = 1u;
    return mask;
  }


  TEST(HostSbsSubtitleSlr13GpuTest, WarmupProtectsCurrentCoversForFourFreshObservationsThenHandsOverToPixels) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    const line_box_t line {180u, 360u, 590u, 370u};
    auto mask = std::vector<std::uint32_t>(fixture.base().size(), 0u);
    ASSERT_TRUE(fixture.set_authored_contour_mask(mask));
    for (std::uint64_t frame = 1u; frame <= v2::subtitle_plane_warmup_observations; ++frame) {
      ASSERT_TRUE(fixture.observe_at(frame, {line}, frame));
      EXPECT_EQ(fixture.state()[v2::subtitle_plane_warmup_word], v2::subtitle_plane_warmup_observations + 1u - frame);
      EXPECT_EQ(fixture.gaussian_control(), 1u);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.gaussian_weights()[364u * field_width + 400u]),
        std::bit_cast<std::uint32_t>(1.0f));
      EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(400u, 364u)), fixture.condition_params()[5u]);
      EXPECT_TRUE(std::all_of(fixture.contour_distances().begin(), fixture.contour_distances().end(),
        [](float q) { return q == contour_distance_infinity; }))
        << "temporary presentation must not manufacture native pixel-distance evidence";
    }
    // This fixture authors distance seeds, not RGB counts. The temporal classifier has its
    // own GPU suite; here the fifth observation must use only the supplied confirmed pixels.
    mask[364u * field_width + 200u] = 1u;
    ASSERT_TRUE(fixture.set_authored_contour_mask(mask));
    ASSERT_TRUE(fixture.observe_at(5u, {line}, 5u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_warmup_word], 0u);
    EXPECT_EQ(fixture.gaussian_control(), 1u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(200u, 364u)), fixture.condition_params()[5u]);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.gaussian_weights()[364u * field_width + 400u]), 0u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(400u, 364u)),
      std::bit_cast<std::uint32_t>(fixture.base()[364u * field_width + 400u]));
  }

  TEST(HostSbsSubtitleSlr13GpuTest, WarmupEveryGenuineCueRearmsAcrossDifferentCurrentCoresAndDuplicatesHold) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.enable_probability_witness(1.f);
    const line_box_t left {180u, 360u, 300u, 370u}, right {500u, 382u, 650u, 392u};
    constexpr auto word = v2::subtitle_plane_warmup_word;
    constexpr auto cue = v2::subtitle_plane_warmup_cue_bit;
    ASSERT_TRUE(fixture.observe_at(1u, {left, right}, 100001u));
    EXPECT_EQ(fixture.state()[word], 4u);
    fixture.set_probability_quarters(1.f, 0.f, 1.f, 1.f);
    ASSERT_TRUE(fixture.observe_at(2u, {left, right}, 200001u));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 1u);
    EXPECT_EQ(fixture.state()[word], 4u | cue);
    ASSERT_TRUE(fixture.observe_at(2u, {left, right}, 200001u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u);
    EXPECT_EQ(fixture.state()[word], 4u | cue) << "duplicate must retain the last genuine cue";
    // Another current core changes while the preceding genuine any-cue latch remains high.
    fixture.set_probability_quarters(1.f, 0.f, 0.f, 1.f);
    ASSERT_TRUE(fixture.observe_at(3u, {left, right}, 300001u));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 2u);
    EXPECT_EQ(fixture.state()[word], 4u | cue);
    ASSERT_TRUE(fixture.observe_at(3u, {left, right}, 300001u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u);
    EXPECT_EQ(fixture.state()[word], 4u | cue);
    const std::array changes {
      std::array {0.f, 1.f, 0.f, 1.f},
      std::array {0.f, 1.f, 1.f, 0.f},
      std::array {1.f, 0.f, 1.f, 0.f}
    };
    for (std::size_t index = 0u; index < changes.size(); ++index) {
      const auto &values = changes[index];
      fixture.set_probability_quarters(values[0u], values[1u], values[2u], values[3u]);
      const auto frame = 4u + index;
      ASSERT_TRUE(fixture.observe_at(frame, {left, right}, frame * 100000u + 1u));
      ASSERT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], index == 1u ? 2u : 1u);
      EXPECT_EQ(fixture.state()[word], 4u | cue) << "continuous genuine cues may prolong protection";
    }
    for (std::uint64_t frame = 7u; frame <= 10u; ++frame) {
      ASSERT_TRUE(fixture.observe_at(frame, {left, right}, frame * 100000u + 1u));
      ASSERT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u);
      EXPECT_EQ(fixture.state()[word], 10u - frame);
      if (frame == 7u) {
        ASSERT_TRUE(fixture.observe_at(frame, {left, right}, frame * 100000u + 1u));
        EXPECT_EQ(fixture.state()[word], 3u) << "a quiet duplicate cannot spend an observation";
      }
    }
    fixture.set_probability_quarters(0.f, 1.f, 1.f, 0.f);
    ASSERT_TRUE(fixture.observe_at(11u, {left, right}, 1100001u));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 1u);
    EXPECT_EQ(fixture.state()[word], 4u | cue);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, WarmupRequiresBothStrictlyNewerSourceClocks) {
    const line_box_t line {180u, 360u, 590u, 370u};
    for (const auto fault : {0u, 1u, 2u, 3u, 4u}) {
      SCOPED_TRACE(fault);
      subtitle_plane_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      ASSERT_TRUE(fixture.set_authored_contour_mask(std::vector<std::uint32_t>(fixture.base().size(), 0u)));
      ASSERT_TRUE(fixture.observe_at(10u, {line}, 100001u));
      ASSERT_EQ(fixture.state()[v2::subtitle_plane_warmup_word], 4u);
      if (fault == 0u) ASSERT_TRUE(fixture.observe_at(10u, {line}, 100001u)); // Exact duplicate holds.
      if (fault == 1u) ASSERT_TRUE(fixture.observe_at(11u, {line}, 100001u)); // Equal time.
      if (fault == 2u) ASSERT_TRUE(fixture.observe_at(10u, {line}, 200001u)); // Equal frame.
      if (fault == 3u) ASSERT_TRUE(fixture.observe_at(9u, {line}, 200001u)); // Older frame.
      if (fault == 4u) ASSERT_TRUE(fixture.observe_at(11u, {line}, 100000u)); // Older time.
      EXPECT_EQ(fixture.state()[v2::subtitle_plane_warmup_word], fault == 0u ? 4u : 0u);
      EXPECT_EQ(fixture.gaussian_control(), fault == 0u ? 1u : 0u);
      if (fault != 0u) EXPECT_TRUE(fixture.output_is_exact_base());
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, WarmupClearsOnEmptyOrInvalidAndRearmsOnlyEligibleCurrentEvidence) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    ASSERT_TRUE(fixture.set_authored_contour_mask(std::vector<std::uint32_t>(fixture.base().size(), 0u)));
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto word = v2::subtitle_plane_warmup_word;
    ASSERT_TRUE(fixture.observe_at(1u, {line}, 1u));
    ASSERT_EQ(fixture.state()[word], 4u);
    ASSERT_TRUE(fixture.observe_at(2u, {}, 2u));
    EXPECT_EQ(fixture.state()[word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.observe_at(2u, {line}, 2u));
    EXPECT_EQ(fixture.state()[word], 0u) << "an exact duplicate after empty cannot acquire warmup";
    EXPECT_EQ(fixture.gaussian_control(), 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.observe_at(3u, {}, 3u));
    ASSERT_TRUE(fixture.observe_at(4u, {line}, 4u));
    EXPECT_EQ(fixture.state()[word], 4u);
    ASSERT_TRUE(fixture.observe_at(5u, {line}, 5u, false, false));
    EXPECT_EQ(fixture.state()[word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.observe_at(6u, {line}, 6u));
    EXPECT_EQ(fixture.state()[word], 4u);
    fixture.set_geometry_ready(false);
    ASSERT_TRUE(fixture.observe_at(7u, {line}, 7u));
    EXPECT_EQ(fixture.state()[word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    fixture.set_geometry_ready(true);
    ASSERT_TRUE(fixture.observe_at(8u, {line}, 8u));
    EXPECT_EQ(fixture.state()[word], 4u);
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe_at(9u, {line}, 9u));
    EXPECT_EQ(fixture.state()[word], 4u);
    ASSERT_TRUE(fixture.observe_at(10u, {line}, 10u, true));
    EXPECT_EQ(fixture.state()[word], 4u);
    fixture.set_base(std::numeric_limits<float>::quiet_NaN());
    ASSERT_TRUE(fixture.observe_at(11u, {line}, 11u));
    EXPECT_EQ(fixture.state()[word], 0u);
    EXPECT_EQ(fixture.condition_params()[4u], 0u);
    fixture.set_base(0.03f);
    ASSERT_TRUE(fixture.observe_at(12u, {line}, 12u));
    EXPECT_EQ(fixture.state()[word], 4u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, WarmupUsesMaximumGaussianUnionOfOnlyCurrentCoverRectangles) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    ASSERT_TRUE(fixture.set_authored_contour_mask(std::vector<std::uint32_t>(fixture.base().size(), 0u)));
    const ocr_box_t left {line_box_t {200u, 360u, 210u, 370u}, line_box_t {180u, 350u, 300u, 392u}, 0u, 1u, 0u};
    const ocr_box_t right {line_box_t {520u, 365u, 530u, 375u}, line_box_t {500u, 355u, 620u, 395u}, 0u, 1u, 0u};
    ASSERT_TRUE(fixture.observe_at(1u, {left, right}, 1u));
    const auto expected_weight = [](std::uint32_t x, std::uint32_t y, const std::vector<ocr_box_t> &boxes) {
      double q = std::numeric_limits<double>::infinity();
      for (const auto &box : boxes) {
        const auto &r = box.cover;
        const double dx = std::max({double(r.left) - (x + .5), (x + .5) - double(r.right), 0.0}) * 1920.0 / field_width;
        const double dy = std::max({double(r.top) - (y + .5), (y + .5) - double(r.bottom), 0.0}) * 1080.0 / field_height;
        q = std::min(q, dx * dx + dy * dy);
      }
      const double sigma = 1080.0 / v2::subtitle_contour_gaussian_sigma_short_side_divisor;
      return std::exp(-q / (2.0 * sigma * sigma));
    };
    for (const auto [x, y] : std::array<std::pair<std::uint32_t, std::uint32_t>, 7u> {
           {{180u, 350u}, {299u, 391u}, {300u, 392u}, {320u, 394u}, {400u, 365u}, {499u, 354u}, {619u, 394u}}}) {
      EXPECT_NEAR(fixture.gaussian_weights()[y * field_width + x], expected_weight(x, y, {left, right}),
        v2::subtitle_contour_gaussian_weight_absolute_error_limit);
    }
    const auto union_weights = fixture.gaussian_weights();
    ASSERT_TRUE(fixture.observe_at(2u, {right, left, left}, 2u));
    EXPECT_EQ(fixture.gaussian_weights(), union_weights) << "reordering and overlap do not sum intensity";
    ASSERT_TRUE(fixture.observe_at(3u, {right}, 3u));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.gaussian_weights()[360u * field_width + 200u]), 0u)
      << "the absent left current cover cannot retain temporary presentation authority";
    EXPECT_TRUE(std::all_of(fixture.contour_distances().begin(), fixture.contour_distances().end(),
      [](float q) { return q == contour_distance_infinity; }));
  }

  TEST(HostSbsSubtitleSlr13GpuTest, WarmupRectangleGaussianUsesFractionalSourceUnitsAndNearestContentPadding) {
    const tensor_content_t content {0u, 89u, 770u, 345u};
    subtitle_plane_fixture_t fixture(770u, 434u, 1500u, 500u,
      std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max(), content);
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    ASSERT_TRUE(fixture.set_authored_contour_mask(std::vector<std::uint32_t>(fixture.base().size(), 0u)));
    const auto bottom = subtitle_roi_edge(1500u, 500u, content, v2::subtitle_ocr_safe_row_bottom);
    const line_box_t line {180u, bottom - 20u, 590u, bottom};
    ASSERT_TRUE(fixture.observe_at(1u, {line}, 1u));
    ASSERT_EQ(fixture.gaussian_control(), 1u); // Finite rectangle q suffices even when the far M underflows.
    const auto x = line.left - 3u, y = line.top - 2u;
    const double dx = 2.5 * 1500.0 / (content[2u] - content[0u]);
    const double dy = 1.5 * 500.0 / (content[3u] - content[1u]);
    const double sigma = 500.0 / v2::subtitle_contour_gaussian_sigma_short_side_divisor;
    EXPECT_NEAR(fixture.gaussian_weights()[y * field_width + x],
      std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)),
      v2::subtitle_contour_gaussian_weight_absolute_error_limit);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.gaussian_weights()[(line.top + 1u) * field_width + line.left]),
      std::bit_cast<std::uint32_t>(1.0f));
    for (const auto pad_y : {0u, field_height - 1u}) {
      const auto cy = std::clamp(pad_y, content[1u], content[3u] - 1u);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(300u, pad_y)),
        std::bit_cast<std::uint32_t>(fixture.output_at(300u, cy)));
      EXPECT_EQ(fixture.contour_distances()[pad_y * field_width + 300u], contour_distance_infinity);
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, WarmupCannotWaiveMalformedPackedStateOrNativeDistanceEvidence) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    ASSERT_TRUE(fixture.set_authored_contour_mask(std::vector<std::uint32_t>(fixture.base().size(), 0u)));
    const line_box_t line {180u, 360u, 590u, 370u};
    ASSERT_TRUE(fixture.observe_at(1u, {line}, 1u));
    ASSERT_EQ(fixture.gaussian_control(), 1u);
    const auto identity = fixture.contour_identity();
    const auto warm_output = fixture.output(), warm_weights = fixture.gaussian_weights();
    for (const auto packed : {1u, 2u, 3u, 4u, 9u, 10u, 11u, 12u}) {
      SCOPED_TRACE(packed);
      ASSERT_TRUE(fixture.overwrite_state_word(v2::subtitle_plane_warmup_word, packed));
      ASSERT_TRUE(fixture.condition_with_contour_identity_word(0u, identity[0u]));
      EXPECT_EQ(fixture.gaussian_control(), 1u);
      EXPECT_EQ(fixture.gaussian_weights(), warm_weights);
      EXPECT_EQ(fixture.output(), warm_output);
    }
    for (const auto malformed : {5u, 6u, 7u, 13u, 14u, 15u, 16u, 0xffffffffu}) {
      ASSERT_TRUE(fixture.overwrite_state_word(v2::subtitle_plane_warmup_word, malformed));
      ASSERT_TRUE(fixture.condition_with_contour_identity_word(0u, identity[0u]));
      EXPECT_EQ(fixture.gaussian_control(), 0u);
      EXPECT_TRUE(fixture.output_is_exact_base());
    }
    ASSERT_TRUE(fixture.overwrite_state_word(v2::subtitle_plane_warmup_word, v2::subtitle_plane_warmup_observations));
    for (const auto bad : {std::numeric_limits<float>::quiet_NaN(), -1.0f,
           -std::numeric_limits<float>::infinity(), std::bit_cast<float>(0x80000000u)}) {
      auto q = std::vector<float>(fixture.base().size(), contour_distance_infinity);
      q[0u] = bad;
      ASSERT_TRUE(fixture.condition_with_distance_field(q));
      EXPECT_EQ(fixture.gaussian_control(), 0u);
      EXPECT_TRUE(fixture.output_is_exact_base());
    }
    ASSERT_TRUE(fixture.condition_with_distance_field(std::vector<float>(fixture.base().size(), contour_distance_infinity)));
    EXPECT_EQ(fixture.gaussian_control(), 1u);
    auto q = std::vector<float>(fixture.base().size(), contour_distance_infinity);
    constexpr auto malformed_pixel = 364u * field_width + 300u;
    q[malformed_pixel] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(fixture.condition_with_distance_field(q));
    EXPECT_EQ(fixture.gaussian_control(), 1u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.gaussian_weights()[malformed_pixel]), 0u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(300u, 364u)),
      std::bit_cast<std::uint32_t>(fixture.base()[malformed_pixel]))
      << "a finite rectangle distance must not hide a malformed native q at this node";
    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(301u, 364u)), fixture.condition_params()[5u]);
    for (const auto packed : {0u, v2::subtitle_plane_warmup_cue_bit}) {
      SCOPED_TRACE(packed);
      ASSERT_TRUE(fixture.overwrite_state_word(v2::subtitle_plane_warmup_word, packed));
      // Cue-only8 is a valid settled state: actual pixel q still authorizes conditioning.
      ASSERT_TRUE(fixture.condition_with_distance_field(std::vector<float>(fixture.base().size(), 0.f)));
      EXPECT_EQ(fixture.gaussian_control(), 1u);
      EXPECT_TRUE(std::all_of(fixture.gaussian_weights().begin(), fixture.gaussian_weights().end(),
        [](float weight) { return weight == 1.f; }));
      ASSERT_TRUE(fixture.condition_with_distance_field(std::vector<float>(fixture.base().size(), contour_distance_infinity)));
      EXPECT_EQ(fixture.gaussian_control(), 0u);
      EXPECT_TRUE(fixture.output_is_exact_base()) << "a cue latch alone cannot revive provisional cover protection";
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, DirectCurrentCoversNeedNoGeometryOwnerConfirmation) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    const std::vector<ocr_box_t> boxes {
      line_box_t {1u, 350u, 4u, 352u}, line_box_t {430u, 395u, 600u, 401u},
      line_box_t {180u, 360u, 590u, 370u}
    };
    ASSERT_TRUE(fixture.observe_at(1u, boxes, 6009000u));
    EXPECT_EQ(fixture.state().size(), 40u);
    EXPECT_EQ(fixture.state()[0u], v2::subtitle_plane_state_schema);
    EXPECT_EQ(fixture.state()[1u], v2::subtitle_plane_state_tag);
    EXPECT_EQ(fixture.state()[2u], v2::subtitle_plane_current_valid_flag);
    EXPECT_EQ(fixture.state()[3u], 7u);
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u)
      << "existing fixtures without probability/history inputs cannot manufacture a reset witness";
    EXPECT_EQ(fixture.condition_params()[2u], boxes.size());
    EXPECT_EQ(fixture.condition_params()[3u], 0u);
    EXPECT_EQ(fixture.condition_params()[4u], 2u);
    EXPECT_EQ(fixture.condition_params()[5u], fixture.state()[v2::subtitle_plane_adaptive_offset + 3u]);
    EXPECT_EQ(fixture.state()[8u], 6009000u);
    EXPECT_EQ(fixture.state()[6u], 1u);
    EXPECT_EQ(fixture.gaussian_control(), 1u); // Authored support; not a temporal confirmation claim.
  }

  TEST(HostSbsSubtitleSlr13GpuTest, CurrentReorderSplitMergeAndUnrelatedTextKeepIndependentUiPlane) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.012f);
    const line_box_t left {180u, 360u, 300u, 370u}, right {300u, 360u, 590u, 370u};
    const line_box_t merged {180u, 360u, 590u, 370u}, footer {4u, 413u, 40u, 419u};
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(1u, {merged}, 1u));
    ASSERT_TRUE(fixture.observe_at(2u, {merged}, 100001u));
    ASSERT_TRUE(fixture.observe_at(3u, {merged}, 400001u));
    const auto tail = std::vector<std::uint32_t>(fixture.state().begin() + a, fixture.state().end());
    for (const auto &boxes : {std::vector<ocr_box_t>{left, right},
           std::vector<ocr_box_t>{right, left}, std::vector<ocr_box_t>{merged, footer}}) {
      ASSERT_TRUE(fixture.observe_at(3u, boxes, 400001u));
      EXPECT_EQ(fixture.condition_params()[2u], boxes.size());
      EXPECT_EQ(fixture.state()[3u], 3u);
      EXPECT_TRUE(std::equal(tail.begin(), tail.end(), fixture.state().begin() + a));
      EXPECT_EQ(fixture.condition_params()[5u], tail[3u]);
    }
    // Pixel history compatibility across these changes is exercised by native contour tests.
  }

  TEST(HostSbsSubtitleSlr13GpuTest, UniformNonzeroProbabilityScalingNeverResetsTheSpatialDistribution) {
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto reset_word = v2::subtitle_plane_history_reset_mask_word;
    for (const auto current : {1.f, .9375f, .875f, .75f, .015625f,
           std::numeric_limits<float>::min()}) {
      SCOPED_TRACE(current);
      subtitle_plane_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      fixture.enable_probability_witness(1.f);
      ASSERT_TRUE(fixture.observe_at(1u, {line}, 1u));
      EXPECT_EQ(fixture.state()[reset_word], 0u); // No prior completed snapshot.
      fixture.set_probability(current);
      ASSERT_TRUE(fixture.observe_at(2u, {line}, 2u));
      EXPECT_EQ(fixture.state()[reset_word], 0u)
        << "finite nonzero magnitude changes preserve a uniform normalized distribution";
      EXPECT_EQ(fixture.condition_params()[4u], v2::subtitle_condition_param_fully_active_value);
      EXPECT_EQ(fixture.state()[v2::subtitle_plane_adaptive_offset], 1u);
      EXPECT_EQ(fixture.state()[v2::subtitle_plane_adaptive_offset + 4u], 2u)
        << "probability invalidation does not restart the independent controller clock";
      ASSERT_TRUE(fixture.observe_at(3u, {line}, 3u));
      EXPECT_EQ(fixture.state()[reset_word], 0u) << "a witness is current, never a retained change gate";
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, NonuniformProbabilityScalingPreservesShapeAtDifferentTotalMasses) {
    const line_box_t core {185u, 360u, 585u, 370u};
    for (const auto scale : {.125f, .5f, 1.f}) {
      SCOPED_TRACE(scale);
      subtitle_plane_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      fixture.enable_probability_witness(1.f);
      fixture.set_probability_halves(.25f, 1.f);
      ASSERT_TRUE(fixture.observe_at(1u, {core}, 1u));
      fixture.set_probability_halves(.25f * scale, scale);
      ASSERT_TRUE(fixture.observe_at(2u, {core}, 2u));
      EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u);
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, SpatialProbabilityChangeUsesStrictExactTotalVariationBoundary) {
    // The calibrated field contains a 1:1 native source rectangle. The source crop and detector
    // are both 960x160, so every sampled center is a detector center without interpolation.
    // This core is native [352,608)x[220,228): 2048 samples, exactly 1024 in each half.
    // Prior mass is 1024. Current .375/.625 also has mass 1024 and each absolute normalized
    // difference is 1/8192, so the production reduction gives T exactly 1/8 in FP32.
    const tensor_content_t content {3u, 57u, 963u, 377u};
    const line_box_t core {355u, 277u, 611u, 285u};
    for (const auto amplitude : {.0625f, .125f, .1875f}) {
      for (const auto scale : {.5f, 1.f}) {
        SCOPED_TRACE(std::to_string(amplitude) + ':' + std::to_string(scale));
        subtitle_plane_fixture_t fixture(966u, 434u, 960u, 320u,
          std::numeric_limits<std::uint32_t>::max(),
          std::numeric_limits<std::uint32_t>::max(), content);
        std::string error;
        ASSERT_TRUE(fixture.initialize(error)) << error;
        fixture.enable_probability_witness(.5f);
        ASSERT_TRUE(fixture.observe_at(1u, {core}, 1u));
        fixture.set_probability_halves((.5f - amplitude) * scale, (.5f + amplitude) * scale);
        ASSERT_TRUE(fixture.observe_at(2u, {core}, 2u));
        EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word],
          amplitude > .125f ? 1u : 0u)
          << "the strict boundary also holds when current and prior total masses differ";
        EXPECT_EQ(fixture.state()[v2::subtitle_plane_adaptive_offset + 4u], 2u);
        ASSERT_TRUE(fixture.observe_at(3u, {core}, 3u));
        EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u);
      }
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, ZeroProbabilityMassResetsOnlyWhenExactlyOneObservationHasMass) {
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto reset_word = v2::subtitle_plane_history_reset_mask_word;
    subtitle_plane_fixture_t zero;
    std::string error;
    ASSERT_TRUE(zero.initialize(error)) << error;
    zero.enable_probability_witness(0.f);
    ASSERT_TRUE(zero.observe_at(1u, {line}, 1u));
    ASSERT_TRUE(zero.observe_at(2u, {line}, 2u));
    EXPECT_EQ(zero.state()[reset_word], 0u);
    zero.set_probability(.5f);
    ASSERT_TRUE(zero.observe_at(3u, {line}, 3u));
    EXPECT_EQ(zero.state()[reset_word], 1u);
    zero.set_probability(0.f);
    ASSERT_TRUE(zero.observe_at(4u, {line}, 4u));
    EXPECT_EQ(zero.state()[reset_word], 1u);
    ASSERT_TRUE(zero.observe_at(5u, {line}, 5u));
    EXPECT_EQ(zero.state()[reset_word], 0u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, ProbabilityWitnessUsesIndependentCurrentRawCoresNotExpandedCoverUnion) {
    const line_box_t left {180u, 360u, 300u, 370u}, right {500u, 382u, 650u, 392u};
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.enable_probability_witness(1.f);
    ASSERT_TRUE(fixture.observe_at(1u, {left, right}, 1u));
    fixture.set_probability_quarters(1.f, 0.f, .984375f, .984375f);
    ASSERT_TRUE(fixture.observe_at(2u, {right, left}, 2u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 2u)
      << "bits follow independent current producer slots, not old order or a multiline union";
    const ocr_box_t expanded {left, line_box_t {180u, 355u, 650u, 397u}, 0u, 1u, 0u};
    // Change only probabilities outside the current raw/core but inside its expanded cover.
    fixture.set_probability_quarters(1.f, 0.f, 0.f, .125f);
    ASSERT_TRUE(fixture.observe_at(3u, {expanded}, 3u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u);
    fixture.set_probability_quarters(0.f, 1.f, 0.f, .125f);
    ASSERT_TRUE(fixture.observe_at(4u, {expanded}, 4u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 1u);
    const line_box_t left_a {180u, 360u, 240u, 370u}, left_b {240u, 360u, 300u, 370u};
    ASSERT_TRUE(fixture.observe_at(5u, {left_b, right, left_a}, 5u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u)
      << "stable P with reorder, expansion, split or merge does not reset by geometry alone";
  }

  TEST(HostSbsSubtitleSlr13GpuTest, SpatialProbabilityChangeReducesNormalizedMassRatherThanVotingPixels) {
    const line_box_t core {185u, 360u, 585u, 370u}; // Symmetric across the source midpoint.
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.enable_probability_witness(1.f);
    fixture.set_probability_halves(.0625f, 1.f);
    ASSERT_TRUE(fixture.observe_at(1u, {core}, 1u));
    fixture.set_probability_halves(0.f, 1.f);
    ASSERT_TRUE(fixture.observe_at(2u, {core}, 2u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u)
      << "about half the pixels change completely, but their low mass gives T about 1/17";
    fixture.set_probability_halves(.25f, 1.f);
    ASSERT_TRUE(fixture.observe_at(3u, {core}, 3u));
    EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 1u)
      << "the same spatial subset now redistributes enough mass: T about 1/5";
    for (std::uint64_t frame = 4u; frame <= 7u; ++frame) {
      fixture.set_probability_halves(frame % 2u ? .25f : .265625f, frame % 2u ? 1.f : .984375f);
      ASSERT_TRUE(fixture.observe_at(frame, {core}, frame));
      EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u)
        << "finite changing P below the ratio does not impose perpetual reacquisition";
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, EveryCurrentAndPriorBilinearContributorMustBeValidForCoreReset) {
    const line_box_t core {180u, 360u, 300u, 370u};
    // First native center in this half-open core. Only the sampling-address oracle is authored;
    // the normalized spatial reduction and reset flag are produced by the production shader.
    const auto px = std::uint32_t(std::ceil(double(core.left) * 1920.0 / field_width - .5));
    const auto py = std::uint32_t(std::ceil(double(core.top) * 1080.0 / field_height - .5)) - 760u;
    const auto dx = int(std::floor((double(px) + .5) * v2::subtitle_ocr_output_width / 1920.0 - .5));
    const auto dy = int(std::floor((double(py) + .5) * v2::subtitle_ocr_output_height / 320.0 - .5));
    const auto x0 = std::clamp(dx, 0, int(v2::subtitle_ocr_output_width) - 1);
    const auto x1 = std::clamp(dx + 1, 0, int(v2::subtitle_ocr_output_width) - 1);
    const auto y0 = std::clamp(dy, 0, int(v2::subtitle_ocr_output_height) - 1);
    const auto y1 = std::clamp(dy + 1, 0, int(v2::subtitle_ocr_output_height) - 1);
    const std::array contributors {
      std::size_t(y0) * v2::subtitle_ocr_output_width + x0,
      std::size_t(y0) * v2::subtitle_ocr_output_width + x1,
      std::size_t(y1) * v2::subtitle_ocr_output_width + x0,
      std::size_t(y1) * v2::subtitle_ocr_output_width + x1,
    };
    subtitle_plane_fixture_t valid;
    std::string error;
    ASSERT_TRUE(valid.initialize(error)) << error;
    valid.enable_probability_witness(1.f);
    ASSERT_TRUE(valid.observe_at(1u, {core}, 1u));
    valid.set_probability_quarters(1.f, 0.f, 1.f, 1.f);
    ASSERT_TRUE(valid.observe_at(2u, {core}, 2u));
    EXPECT_EQ(valid.state()[v2::subtitle_plane_history_reset_mask_word], 1u)
      << "the same spatial pattern must reset when every contributor is valid";
    for (const auto previous : {false, true}) {
      for (const auto contributor : contributors) {
        for (const auto invalid : {std::numeric_limits<float>::quiet_NaN(), -.25f, 1.25f}) {
          SCOPED_TRACE(std::to_string(previous) + ':' + std::to_string(contributor) + ':' +
            std::to_string(invalid));
          subtitle_plane_fixture_t fixture;
          ASSERT_TRUE(fixture.initialize(error)) << error;
          fixture.enable_probability_witness(1.f);
          ASSERT_TRUE(fixture.observe_at(1u, {core}, 1u));
          fixture.set_probability_quarters(1.f, 0.f, 1.f, 1.f);
          fixture.set_probability_word(contributor, invalid, previous);
          ASSERT_TRUE(fixture.observe_at(2u, {core}, 2u));
          EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u)
            << "an invalid sampled contributor cannot be omitted to manufacture a change flag";
        }
      }
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, ProbabilityWitnessRequiresBothNewSourceClocksAndCurrentActiveEvidence) {
    const line_box_t line {180u, 360u, 590u, 370u};
    for (const auto fault : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
      SCOPED_TRACE(fault);
      subtitle_plane_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      fixture.enable_probability_witness(1.f);
      ASSERT_TRUE(fixture.observe_at(1u, {line}, 100001u));
      fixture.set_probability_halves(0.f, 1.f);
      if (fault == 0u) ASSERT_TRUE(fixture.observe_at(1u, {line}, 100001u)); // Duplicate.
      else if (fault == 1u) ASSERT_TRUE(fixture.observe_at(2u, {line}, 100001u)); // Equal clock.
      else if (fault == 2u) ASSERT_TRUE(fixture.observe_at(1u, {line}, 200001u)); // Equal frame.
      else if (fault == 3u) ASSERT_TRUE(fixture.observe_at(2u, {line}, 200001u, true)); // Reset.
      else if (fault == 4u) {
        fixture.set_geometry_ready(false);
        ASSERT_TRUE(fixture.observe_at(2u, {line}, 200001u));
      } else if (fault == 5u) {
        fixture.set_base(std::numeric_limits<float>::quiet_NaN());
        ASSERT_TRUE(fixture.observe_at(2u, {line}, 200001u));
        EXPECT_EQ(fixture.condition_params()[4u], 0u); // Valid OCR, inactive Base probe.
      } else ASSERT_TRUE(fixture.observe_at(2u, {line}, 200001u)); // Valid positive control.
      EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], fault == 6u ? 1u : 0u);
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, InvalidPriorProbabilityOrSnapshotCannotManufactureResetBits) {
    const line_box_t line {180u, 360u, 590u, 370u};
    for (std::uint32_t fault = 0u; fault < 20u; ++fault) {
      SCOPED_TRACE(fault);
      subtitle_plane_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      fixture.enable_probability_witness(1.f);
      ASSERT_TRUE(fixture.observe_at(1u, {line}, 100001u));
      fixture.set_probability_halves(0.f, 1.f);
      switch (fault) {
        case 0u: fixture.overwrite_probability_history_word(0u, 0u); break;
        case 1u: fixture.overwrite_probability_history_word(1u, 0u); break;
        case 2u: fixture.overwrite_probability_history_word(2u, 0u); break; // Prior pin inactive.
        case 3u: fixture.overwrite_probability_history_word(3u, 1u); break; // Cut mismatch.
        case 4u: fixture.overwrite_probability_history_word(4u, 3u); break; // Future prior frame.
        case 5u: fixture.overwrite_probability_history_word(6u, 2u); break; // Generation.
        case 6u: fixture.overwrite_probability_history_word(8u, 1919u); break;
        case 7u: fixture.overwrite_probability_history_word(10u, 761u); break;
        case 8u: fixture.overwrite_probability_history_word(12u, field_width - 1u); break;
        case 9u: fixture.overwrite_probability_history_word(14u, roi_top + 1u); break;
        case 10u: fixture.overwrite_probability_history_word(16u, 1u); break;
        case 11u: fixture.overwrite_probability_history_word(20u, 200002u); break;
        case 12u: fixture.overwrite_probability_history_word(22u, v2::subtitle_contour_recipe - 1u); break;
        case 13u: fixture.overwrite_probability_history_word(v2::subtitle_contour_history_snapshot_valid_word, 0u); break;
        case 14u: fixture.overwrite_probability_history_word(v2::subtitle_contour_history_reserved_offset, 1u); break;
        case 15u: fixture.set_previous_probability(std::numeric_limits<float>::quiet_NaN()); break;
        case 16u: fixture.set_previous_probability(-.25f); break;
        case 17u: fixture.set_probability_availability(true, false); break;
        case 18u: fixture.set_probability_availability(false, true); break;
        case 19u: break; // Valid positive control for the same spatial redistribution.
      }
      ASSERT_TRUE(fixture.observe_at(2u, {line}, 200001u));
      EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], fault == 19u ? 1u : 0u);
    }
    for (const auto invalid : {std::numeric_limits<float>::quiet_NaN(), -1.f, 2.f}) {
      subtitle_plane_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      fixture.enable_probability_witness(1.f);
      fixture.set_probability_halves(0.f, 1.f);
      ASSERT_TRUE(fixture.observe_at(1u, {line}, 1u));
      fixture.set_probability(invalid);
      ASSERT_TRUE(fixture.observe_at(2u, {line}, 2u));
      EXPECT_EQ(fixture.state()[v2::subtitle_plane_history_reset_mask_word], 0u);
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, EmptyAndInvalidRecordsCannotKeepHistoricalProtection) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    const line_box_t line {180u, 360u, 590u, 370u};
    ASSERT_TRUE(fixture.observe_at(1u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(2u, {}, 100001u));
    EXPECT_EQ(fixture.state()[2u], v2::subtitle_plane_current_valid_flag);
    EXPECT_EQ(fixture.state()[3u], 0u);
    EXPECT_EQ(fixture.condition_params()[0u], v2::subtitle_condition_param_schema);
    EXPECT_EQ(fixture.condition_params()[1u], v2::subtitle_condition_param_tag);
    for (std::size_t word = 2u; word < 6u; ++word) EXPECT_EQ(fixture.condition_params()[word], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.gaussian_control(), 0u);
    ASSERT_TRUE(fixture.observe_at(3u, {line}, 200001u, false, false));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[3u], 0u);
    EXPECT_EQ(fixture.condition_params()[2u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, InvalidOcrDisarmsTimingWithoutAnEmptyReleaseProbe) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.012f);
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(1u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(2u, {line}, 100001u));
    ASSERT_TRUE(fixture.observe_at(3u, {line}, 400001u));
    const auto applied = fixture.state()[a + 3u], covered = fixture.state()[a + 13u];
    ASSERT_TRUE(fixture.observe_at(4u, {}, 2500000u, false, false));
    EXPECT_EQ(fixture.state()[a + 3u], applied);
    EXPECT_EQ(fixture.state()[a + 13u], covered);
    EXPECT_EQ(fixture.state()[a + 1u] & 3u, 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    // A zero source frame is invalid evidence, not a genuine empty observation.
    ASSERT_TRUE(fixture.observe_at(0u, {}, 3000000u));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[a + 3u], applied);
    EXPECT_EQ(fixture.state()[a + 13u], covered);
    EXPECT_EQ(fixture.state()[a + 1u] & 3u, 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, RejectsLateOcrMutationsInsteadOfBorrowingConditionAuthority) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    const line_box_t line {180u, 360u, 590u, 370u};
    ASSERT_TRUE(fixture.observe_at(10u, {line}, 1u));
    const auto record = fixture.ocr_record();
    const auto output = fixture.output();
    const std::vector<std::pair<std::size_t, std::uint32_t>> mutations {
      {0u, record[0u] ^ 1u}, {1u, record[1u] ^ 1u}, {2u, 0u}, {3u, 9u}, {4u, 9u},
      {5u, 11u}, {7u, 2u}, {9u, 1921u}, {11u, field_width + 1u}, {13u, roi_top + 1u},
      {15u, 1u}, {20u, std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN())},
      {final_box_offset, line.left + 1u}, {final_box_offset + 4u, record[final_box_offset + 4u] ^ 1u},
      {24u, 1u}, {final_box_offset + 8u, 1u}
    };
    for (const auto &[word, value] : mutations) {
      SCOPED_TRACE(word);
      ASSERT_TRUE(fixture.overwrite_ocr_word(word, value));
      EXPECT_TRUE(fixture.output_is_exact_base());
      EXPECT_EQ(fixture.gaussian_control(), 0u);
      ASSERT_TRUE(fixture.overwrite_ocr_word(word, record[word]));
      EXPECT_EQ(fixture.output(), output);
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, MalformedCurrentPublicationAndConditionPayloadFailClosed) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    const line_box_t line {180u, 360u, 590u, 370u};
    ASSERT_TRUE(fixture.observe_at(17u, {line}, 6676000u));
    const auto state = fixture.state(), params = fixture.condition_params();
    const auto identity = fixture.contour_identity();
    const auto output = fixture.output();
    // Consume the saved SCP/SCT tuple directly; do not first repair it with the resolver.
    ASSERT_TRUE(fixture.overwrite_state_word(v2::subtitle_plane_history_reset_mask_word, 1u));
    ASSERT_TRUE(fixture.condition_with_contour_identity_word(0u, identity[0u]));
    EXPECT_EQ(fixture.output(), output) << "A valid reset mask does not revoke the saved current tuple";
    ASSERT_TRUE(fixture.overwrite_state_word(v2::subtitle_plane_history_reset_mask_word, 0u));
    for (std::size_t word = 0u; word < v2::subtitle_plane_header_word_count; ++word) {
      SCOPED_TRACE(word);
      // The single current slot admits reset bit 0; bit 1 is outside its authority.
      const auto malformed = word == v2::subtitle_plane_history_reset_mask_word ? 2u :
        word == v2::subtitle_plane_warmup_word ? 16u : state[word] ^ 1u;
      ASSERT_TRUE(fixture.overwrite_state_word(word, malformed));
      ASSERT_TRUE(fixture.condition_with_contour_identity_word(0u, identity[0u]));
      EXPECT_TRUE(fixture.output_is_exact_base());
      EXPECT_EQ(fixture.gaussian_control(), 0u);
      ASSERT_TRUE(fixture.overwrite_state_word(word, state[word]));
      ASSERT_TRUE(fixture.condition_with_contour_identity_word(0u, identity[0u]));
      EXPECT_EQ(fixture.output(), output);
    }
    for (std::size_t word = 0u; word < params.size(); ++word) {
      ASSERT_TRUE(fixture.condition_with_parameter_word(word, params[word] ^ 1u));
      EXPECT_TRUE(fixture.output_is_exact_base());
      ASSERT_TRUE(fixture.condition_with_parameter_word(word, params[word]));
      EXPECT_EQ(fixture.output(), output);
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, CurrentEightPairCapacityIsNotAnotherSelectionLimit) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    std::vector<ocr_box_t> boxes;
    for (std::uint32_t slot = 0u; slot < 8u; ++slot)
      boxes.emplace_back(line_box_t{4u + slot * 80u, 350u, 20u + slot * 80u, 355u});
    ASSERT_TRUE(fixture.observe_at(1u, boxes, 1u));
    EXPECT_EQ(fixture.state()[3u], 255u);
    EXPECT_EQ(fixture.condition_params()[2u], 8u);
    boxes.emplace_back(line_box_t{20u, 380u, 40u, 390u});
    ASSERT_TRUE(fixture.observe_at(2u, boxes, 100001u));
    EXPECT_EQ(fixture.state()[3u], 0u);
    EXPECT_EQ(fixture.condition_params()[2u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, SourceTimeDwellStartsWithFirstCurrentCover) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(5.0f / 1920.0f);
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(1u, {line}, 1u));
    EXPECT_EQ(fixture.state()[a + 6u], 1u);
    EXPECT_EQ(fixture.state()[a + 2u], 0u);
    ASSERT_TRUE(fixture.observe_at(2u, {line}, 100000u));
    EXPECT_EQ(fixture.state()[a + 2u], 0u);
    ASSERT_TRUE(fixture.observe_at(3u, {line}, 100001u));
    EXPECT_EQ(fixture.state()[a + 2u], 2u);
    EXPECT_GT(std::bit_cast<float>(fixture.state()[a + 3u]), 0.0f);
    ASSERT_TRUE(fixture.observe_at(4u, {line}, 400001u));
    EXPECT_FLOAT_EQ(std::bit_cast<float>(fixture.state()[a + 3u]), 8.0f / 1920.0f);
    fixture.set_base(-0.01f);
    ASSERT_TRUE(fixture.observe_at(5u, {}, 500001u));
    ASSERT_TRUE(fixture.observe_at(6u, {}, 2000000u));
    EXPECT_EQ(fixture.state()[a + 2u], 2u);
    ASSERT_TRUE(fixture.observe_at(7u, {}, 2000001u));
    EXPECT_EQ(fixture.state()[a + 2u], 0u);
    EXPECT_GT(std::bit_cast<float>(fixture.state()[a + 3u]), 0.0f);
    EXPECT_EQ(fixture.state()[3u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, ClockFaultDisarmsAndDuplicateDoesNotSpendUiTime) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.012f);
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(1u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(2u, {line}, 100001u));
    ASSERT_TRUE(fixture.observe_at(3u, {line}, 400001u));
    const auto tail = fixture.state();
    ASSERT_TRUE(fixture.observe_at(3u, {line}, 800001u));
    EXPECT_TRUE(std::equal(tail.begin() + a, tail.end(), fixture.state().begin() + a));
    for (const auto now : {399999ull, 400001ull}) {
      ASSERT_TRUE(fixture.observe_at(4u + now % 2u, {line}, now));
      EXPECT_EQ(fixture.state()[a + 3u], tail[a + 3u]);
      EXPECT_EQ(fixture.state()[a + 1u] & 3u, 0u);
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, MissingGeometryKeepsAppliedPlaneAndRearmsFromReadySourceClock) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.012f);
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(1u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(2u, {line}, 100001u));
    ASSERT_TRUE(fixture.observe_at(3u, {line}, 400001u));
    const auto applied = fixture.state()[a + 3u], goal = fixture.state()[a + 2u];
    fixture.set_geometry_available(false);
    ASSERT_TRUE(fixture.observe_at(4u, {line}, 1900001u));
    EXPECT_EQ(fixture.state()[3u], 0u);
    EXPECT_EQ(fixture.state()[a + 3u], applied);
    EXPECT_EQ(fixture.state()[a + 2u], goal);
    EXPECT_EQ(fixture.state()[a + 4u], 0u);
    EXPECT_EQ(fixture.state()[a + 1u] & 11u, 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    fixture.set_geometry_available(true);
    fixture.override_geometry_clock(1900001u);
    ASSERT_TRUE(fixture.observe_at(5u, {line}, 2900001u));
    EXPECT_EQ(fixture.condition_params()[2u], 0u);
    EXPECT_EQ(fixture.state()[a + 3u], applied);
    fixture.override_geometry_clock(std::nullopt);
    ASSERT_TRUE(fixture.observe_at(6u, {line}, 3900001u));
    EXPECT_EQ(fixture.state()[a + 4u], 3900001u);
    EXPECT_EQ(fixture.state()[a + 3u], applied);
    EXPECT_EQ(fixture.condition_params()[2u], 1u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, CutAndDomainResetClearPlaneWithoutRejectingCurrentText) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.018f);
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    for (std::uint64_t frame = 1u; frame <= 5u; ++frame)
      ASSERT_TRUE(fixture.observe_at(frame, {line}, 1u + frame * 200000u));
    ASSERT_GT(std::bit_cast<float>(fixture.state()[a + 3u]), 0.01f);
    fixture.set_cut(2u, true);
    ASSERT_TRUE(fixture.observe_at(6u, {line}, 1400001u));
    EXPECT_EQ(fixture.state()[20u], 2u);
    EXPECT_EQ(fixture.state()[3u], 1u);
    EXPECT_EQ(fixture.state()[a + 3u], 0u);
    const auto state = fixture.state();
    fixture.set_cut(2u, true);
    ASSERT_TRUE(fixture.observe_at(6u, {line}, 1400001u));
    EXPECT_EQ(fixture.state(), state);
    fixture.set_analysis_generation(9u);
    ASSERT_TRUE(fixture.observe_at(7u, {line}, 1600001u));
    EXPECT_EQ(fixture.state()[6u], 9u);
    EXPECT_EQ(fixture.state()[a + 3u], 0u);
    ASSERT_TRUE(fixture.observe_at(8u, {line}, 1800001u, true));
    EXPECT_EQ(fixture.state()[a + 3u], 0u);
    EXPECT_EQ(fixture.condition_params()[2u], 1u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, RibbonAndShortOrdinaryTextShareExactlyCurrentKindMask) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    const ocr_box_t ribbon {{1u, 350u, 689u, roi_bottom},
      {0u, 346u, field_width, field_height}, box_flag_ribbon, 7u, 4u};
    const line_box_t short_text {8u, 352u, 11u, 354u};
    ASSERT_TRUE(fixture.observe_at(1u, {short_text, ribbon}, 1u));
    EXPECT_EQ(fixture.condition_params()[2u], 2u);
    EXPECT_EQ(fixture.condition_params()[3u], 2u);
    ASSERT_TRUE(fixture.observe_at(2u, {ribbon, short_text}, 100001u));
    EXPECT_EQ(fixture.condition_params()[3u], 1u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, AdaptiveUiFullBottomUnionCountsPeripheralGlobalConflict) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(-0.01f);
    fixture.set_base_columns(0u, 125u, 5.0f / 1920.0f);
    const ocr_box_t ribbon {
      {1u, 350u, 689u, roi_bottom}, {0u, 346u, field_width, field_height},
      box_flag_ribbon, 7u, 4u
    };
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(60u, {ribbon}, 1u));
    ASSERT_TRUE(fixture.observe_at(61u, {ribbon}, 100001u));
    ASSERT_EQ(fixture.state()[a + 13u], field_width * (field_height - 346u));
    // Peripheral conflicts occupy less than 20% of this broad cover, but exceed 2% of content.
    ASSERT_LT(125u * 100u, field_width * 20u);
    ASSERT_GT(125u * (field_height - 346u) * 100u, field_width * field_height * 2u);
    ASSERT_TRUE(fixture.observe_at(62u, {ribbon}, 200001u));
    EXPECT_EQ(fixture.state()[a + 2u], 2u);
    EXPECT_EQ(fixture.state()[a + 15u], 0u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, AdaptiveUiProbeVisitsAwkwardCoverWidthsAndPartialRows) {
    // Strides span several rows below 256 cells and carry within a row above it. None of these
    // rectangles has an area divisible by the group size, so the final partial visits matter.
    for (const std::uint32_t width : {93u, 255u, 257u, 513u}) {
      SCOPED_TRACE(width);
      subtitle_plane_fixture_t fixture;
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      fixture.set_base(5.0f / 1920.0f);
      const line_box_t line {31u, 360u, 31u + width, 371u};
      constexpr auto a = v2::subtitle_plane_adaptive_offset;
      ASSERT_TRUE(fixture.observe_at(70u, {line}, 1u));
      ASSERT_TRUE(fixture.observe_at(71u, {line}, 100001u));
      ASSERT_EQ(fixture.condition_params()[2u], 1u);
      EXPECT_EQ(fixture.state()[a + 13u], width * 11u);
      EXPECT_EQ(fixture.state()[a + 14u], 0u);
      EXPECT_EQ(fixture.state()[a + 15u], 0u);
      ASSERT_TRUE(fixture.observe_at(72u, {line}, 200001u));
      EXPECT_EQ(fixture.state()[a + 2u], 2u);
      const auto applied = fixture.state()[a + 3u];

      fixture.set_base_at(line.left, line.top, std::numeric_limits<float>::quiet_NaN());
      fixture.set_base_at(line.right - 1u, line.bottom - 1u,
        std::numeric_limits<float>::infinity());
      fixture.set_base_at(line.left + width / 2u, line.top + 5u, 0.041f);
      ASSERT_TRUE(fixture.observe_at(73u, {line}, 300001u));
      EXPECT_EQ(fixture.state()[a + 13u], width * 11u);
      EXPECT_EQ(fixture.state()[a + 14u], 3u);
      EXPECT_EQ(fixture.state()[a + 15u], 3u);
      EXPECT_EQ(fixture.state()[a + 3u], applied);
      EXPECT_EQ(fixture.condition_params()[2u], 0u);
      EXPECT_TRUE(fixture.output_is_exact_base());
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, AdaptiveUiProbeDeduplicatesOverlappingCurrentCovers) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.0395f);
    const ocr_box_t first {
      {180u, 360u, 410u, 370u}, {170u, 356u, 427u, 374u}, 0u, 1u, 0u
    };
    const ocr_box_t second {
      {180u, 374u, 410u, 384u}, {175u, 368u, 432u, 388u}, 0u, 1u, 0u
    };
    // The tight cores are distinct lines. Their coarse covers overlap in 252*6 cells, which
    // belong to the UI mask once regardless of the OCR input order or the probing lane.
    constexpr std::uint32_t covered = 257u * 18u + 257u * 20u - 252u * 6u;
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(80u, {first, second}, 1u));
    ASSERT_TRUE(fixture.observe_at(81u, {second, first}, 100001u));
    ASSERT_EQ(fixture.condition_params()[2u], 2u);
    EXPECT_EQ(fixture.state()[a + 13u], covered);
    EXPECT_EQ(fixture.state()[a + 14u], covered);
    EXPECT_EQ(fixture.state()[a + 15u], 0u);
    EXPECT_NE(fixture.state()[a + 1u] & 4u, 0u);
    ASSERT_TRUE(fixture.observe_at(82u, {first, second}, 200001u));
    const auto applied = fixture.state()[a + 3u];

    fixture.set_base_at(170u, 356u, std::numeric_limits<float>::quiet_NaN());
    fixture.set_base_at(431u, 387u, std::numeric_limits<float>::infinity());
    fixture.set_base_at(180u, 370u, 0.041f); // In both covers: still one invalid sample.
    ASSERT_TRUE(fixture.observe_at(83u, {second, first}, 300001u));
    EXPECT_EQ(fixture.state()[a + 13u], covered);
    EXPECT_EQ(fixture.state()[a + 14u], covered);
    EXPECT_EQ(fixture.state()[a + 15u], 3u);
    EXPECT_EQ(fixture.state()[a + 3u], applied);
    EXPECT_EQ(fixture.condition_params()[2u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, AdaptiveUiRejectsActiveTargetPolicyAndPaddingTampering) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    const line_box_t line {180u, 360u, 590u, 370u};
    ASSERT_TRUE(fixture.observe_at(70u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(71u, {line}, 100001u));
    ASSERT_EQ(fixture.condition_params()[4u], 2u);
    ASSERT_FALSE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.condition_with_parameter_word(4u, 1u));
    EXPECT_TRUE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.observe_at(72u, {line}, 200001u));
    ASSERT_TRUE(fixture.overwrite_state_word(
      v2::subtitle_plane_adaptive_offset + 3u, std::bit_cast<std::uint32_t>(0.001f)));
    ASSERT_TRUE(fixture.condition_with_parameter_word(5u, fixture.condition_params()[5u]));
    EXPECT_TRUE(fixture.output_is_exact_base());

    for (const auto retired_or_foreign_policy : {0u, 1u, 2u, 4u, 0xffffffffu}) {
      fixture.set_adaptive_policy_words(retired_or_foreign_policy);
      ASSERT_TRUE(fixture.observe_at(73u, {line}, 300001u));
      EXPECT_TRUE(fixture.output_is_exact_base());
      EXPECT_EQ(fixture.condition_params()[2u], 0u);
    }
    fixture.set_adaptive_policy_words(v2::adaptive_policy_id, 1u);
    ASSERT_TRUE(fixture.observe_at(74u, {line}, 400001u));
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.condition_params()[2u], 0u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, AdaptiveUiRejectsClockPaddingAndBudgetTampering) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    constexpr float budget = 0.006f;
    fixture.set_adaptive_policy_words(v2::adaptive_policy_id, 0u, budget);
    fixture.set_base(0.004f);
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(220u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(221u, {line}, 100001u));
    ASSERT_EQ(fixture.condition_params()[4u], 2u);
    ASSERT_FALSE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.condition_with_parameter_word(4u, 1u));
    EXPECT_TRUE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.overwrite_state_word(a + 12u,
      std::bit_cast<std::uint32_t>(16.0f / 1920.0f)));
    ASSERT_TRUE(fixture.condition_only());
    EXPECT_TRUE(fixture.output_is_exact_base());

    fixture.override_joint_observation(200000u);
    ASSERT_TRUE(fixture.observe_at(222u, {line}, 200001u));
    EXPECT_EQ(fixture.condition_params()[2u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    fixture.set_adaptive_policy_words(v2::adaptive_policy_id, 0u, budget);
    fixture.override_joint_observation(300001u, 1u);
    ASSERT_TRUE(fixture.observe_at(223u, {line}, 300001u));
    EXPECT_EQ(fixture.condition_params()[2u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, AdaptiveUiClockCarriesAcrossUint32Boundary) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(5.0f / 1920.0f);
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    constexpr std::uint64_t before_wrap = 0xffffffffull - 200000ull;
    ASSERT_TRUE(fixture.observe_at(80u, {line}, before_wrap));
    ASSERT_TRUE(fixture.observe_at(81u, {line}, before_wrap + 100000u));
    ASSERT_TRUE(fixture.observe_at(82u, {line}, before_wrap + 300000u));
    EXPECT_EQ(fixture.state()[a + 5u], 1u);
    EXPECT_EQ(fixture.state()[a + 2u], 2u);
    EXPECT_GT(std::bit_cast<float>(fixture.state()[a + 3u]), 0.0f);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, HostModelUiProposalUsesExactCentersAndActualGaussianForNonuniformBase) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_adaptive_policy_words(v2::adaptive_policy_id, 0u, 0.0065625f);
    fixture.set_base(-0.006f);
    // A nonuniform Base verifies that actual M, rather than a reconstructed ideal alpha,
    // drives each proposal cell. This fixture deliberately stops before shared post-limits.
    for (std::uint32_t column = 0u; column < 32u; ++column) {
      fixture.set_base_columns(column * field_width / 32u,
        (column + 1u) * field_width / 32u, -0.006f + 0.0004f * column);
    }
    const line_box_t line {180u, 360u, 590u, 370u};
    ASSERT_TRUE(fixture.observe_at(320u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(321u, {line}, 100001u));
    const auto target_bits = fixture.condition_params()[5u];
    float maximum_horizontal_step = 0.0f;
    float maximum_vertical_step = 0.0f;
    for (std::uint32_t y = 0u; y < field_height; ++y) {
      for (std::uint32_t x = 0u; x < field_width; ++x) {
        const float value = fixture.output_at(x, y);
        if (x >= line.left && x < line.right && y >= line.top && y < line.bottom) {
          EXPECT_EQ(std::bit_cast<std::uint32_t>(value), target_bits);
        }
        if (x != 0u) maximum_horizontal_step = std::max(maximum_horizontal_step,
          std::abs(value - fixture.output_at(x - 1u, y)));
        if (y != 0u) maximum_vertical_step = std::max(maximum_vertical_step,
          std::abs(value - fixture.output_at(x, y - 1u)));
        if (y < 100u) {
          EXPECT_EQ(std::bit_cast<std::uint32_t>(value),
            std::bit_cast<std::uint32_t>(fixture.base()[static_cast<std::size_t>(y) * field_width + x]));
        }
      }
    }
    EXPECT_GT(maximum_horizontal_step, 0.0f);
    EXPECT_GT(maximum_vertical_step, 0.0f);
    ASSERT_EQ(fixture.gaussian_control(), 1u);
    const auto sample = 354u * field_width + 300u;
    const auto m = fixture.gaussian_weights()[sample];
    ASSERT_GT(m, 0.0f);
    ASSERT_LT(m, 1.0f);
    volatile float base_term = (1.0f - m) * fixture.base()[sample];
    volatile float ui_term = m * std::bit_cast<float>(target_bits);
    EXPECT_NEAR(fixture.output()[sample], base_term + ui_term,
      4.0f * std::numeric_limits<float>::epsilon() * v2::direct_container_limit);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, HostModelUiReportsUnclearableConflictAtExactRepresentationCap) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_adaptive_policy_words(v2::adaptive_policy_id, 0u, 0.0065625f);
    fixture.set_base(0.0395f);
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(330u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(331u, {line}, 100001u));
    ASSERT_TRUE(fixture.observe_at(332u, {line}, 200001u));
    EXPECT_EQ(fixture.state()[a + 2u], 20u);
    EXPECT_NE(fixture.state()[a + 1u] & 4u, 0u);
    EXPECT_EQ(fixture.state()[a + 14u], fixture.state()[a + 13u]);
    for (std::uint64_t update = 1u; update <= 8u; ++update) {
      ASSERT_TRUE(fixture.observe_at(332u + update, {line}, 200001u + update * 250000u));
    }
    EXPECT_EQ(fixture.state()[a + 3u], std::bit_cast<std::uint32_t>(0.04f));
    EXPECT_FLOAT_EQ(fixture.output_at(300u, 365u), 0.04f);
    EXPECT_FLOAT_EQ(fixture.output_at(0u, 0u), 0.0395f);
    ASSERT_TRUE(fixture.overwrite_state_word(a + 1u, fixture.state()[a + 1u] & ~4u));
    ASSERT_TRUE(fixture.condition_only());
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, HostModelUiPeripheralUnionAndRetreatWaitForAppliedTarget) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_adaptive_policy_words(v2::adaptive_policy_id, 0u, 0.0065625f);
    fixture.set_base(-0.01f);
    fixture.set_base_columns(0u, 125u, 0.018f);
    const ocr_box_t ribbon {
      {1u, 350u, 689u, roi_bottom}, {0u, 346u, field_width, field_height},
      box_flag_ribbon, 7u, 4u
    };
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(350u, {ribbon}, 1u));
    ASSERT_TRUE(fixture.observe_at(351u, {ribbon}, 100001u));
    EXPECT_EQ(fixture.state()[a + 13u], field_width * (field_height - 346u));
    ASSERT_TRUE(fixture.observe_at(352u, {ribbon}, 200001u));
    EXPECT_EQ(fixture.state()[a + 2u], 10u);
    EXPECT_EQ(fixture.state()[a + 1u] & 4u, 0u);
    fixture.set_base(-0.01f);
    ASSERT_TRUE(fixture.observe_at(353u, {ribbon}, 300001u));
    EXPECT_EQ(fixture.state()[a + 2u], 10u);
    EXPECT_EQ(fixture.state()[a + 1u] & 2u, 0u);
    EXPECT_EQ(fixture.state()[a + 8u], 0u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, GaussianProposalUsesLiteralWeightsWithoutHardMarginOrTruncation) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    const line_box_t line {180u, 360u, 590u, 390u};
    const auto mask = authored_hollow_and_disjoint_contours();
    ASSERT_TRUE(fixture.set_authored_contour_mask(mask));
    ASSERT_TRUE(fixture.observe_at(400u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(401u, {line}, 100001u));
    ASSERT_TRUE(fixture.observe_at(402u, {line}, 100002u));
    ASSERT_TRUE(fixture.observe_at(403u, {line}, 100003u));
    ASSERT_TRUE(fixture.observe_at(404u, {line}, 100004u));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_warmup_word], 0u);
    ASSERT_EQ(fixture.condition_params()[2u], 1u);
    ASSERT_EQ(fixture.gaussian_control(), 1u);
    const auto target_bits = fixture.condition_params()[5u];
    const float target = std::bit_cast<float>(target_bits);
    const float sigma = std::min(fixture.contour_identity()[8u], fixture.contour_identity()[9u]) / v2::subtitle_contour_gaussian_sigma_short_side_divisor;
    for (std::size_t index = 0u; index < mask.size(); ++index) {
      const float q = fixture.contour_distances()[index], m = fixture.gaussian_weights()[index];
      ASSERT_TRUE(std::isfinite(q));
      EXPECT_NEAR(m, std::exp(-static_cast<double>(q) / (2.0 * sigma * sigma)),
        v2::subtitle_contour_gaussian_weight_absolute_error_limit);
      if (q == 0.0f) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(m), std::bit_cast<std::uint32_t>(1.0f));
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output()[index]), target_bits);
      } else if (m == 0.0f) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output()[index]), std::bit_cast<std::uint32_t>(fixture.base()[index]));
      } else {
        volatile float bt = (1.0f - m) * fixture.base()[index];
        volatile float ut = m * target;
        EXPECT_NEAR(fixture.output()[index], bt + ut, 4.0f * std::numeric_limits<float>::epsilon() * v2::direct_container_limit);
      }
    }
    // q>0 one node from support is a tail, not a presentation-only exact target margin.
    const auto adjacent = 372u * field_width + 269u;
    EXPECT_GT(fixture.contour_distances()[adjacent], 0.0f);
    EXPECT_GT(fixture.gaussian_weights()[adjacent], 0.0f);
    EXPECT_LT(fixture.gaussian_weights()[adjacent], 1.0f);
    EXPECT_GT(fixture.output()[adjacent], target);
    // Forty field columns from the ring is about 100 source pixels, beyond five current sigmas.
    const auto untruncated = 372u * field_width + 230u;
    EXPECT_GT(fixture.gaussian_weights()[untruncated], 0.0f);
    EXPECT_LT(fixture.gaussian_weights()[untruncated], 1.0e-6f);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, GaussianProposalPreservesSignedZeroAndExactZeroWeightBaseEndpoints) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    const line_box_t line {180u, 360u, 590u, 390u};
    ASSERT_TRUE(fixture.observe_at(430u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(431u, {line}, 100001u));
    ASSERT_TRUE(fixture.observe_at(432u, {line}, 100002u));
    ASSERT_TRUE(fixture.observe_at(433u, {line}, 100003u));
    ASSERT_TRUE(fixture.observe_at(434u, {line}, 100004u));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_warmup_word], 0u);
    std::vector<float> q(fixture.base().size(), std::numeric_limits<float>::infinity());
    q[364u * field_width + 300u] = 0.0f;
    q[0u] = 1.0e6f;  // Legal finite q enables smoothing; far cells underflow in FP32.
    for (const auto target_bits : {0u, 0x80000000u}) {
      for (const float base : {-0.03f, 0.03f, std::bit_cast<float>(0x33000001u)}) {
        fixture.set_base(base);
        ASSERT_TRUE(fixture.overwrite_state_word(v2::subtitle_plane_adaptive_offset + 3u, target_bits));
        ASSERT_TRUE(fixture.condition_with_parameter_word(5u, target_bits));
        ASSERT_TRUE(fixture.condition_with_distance_field(q));
        ASSERT_EQ(fixture.gaussian_control(), 1u);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(300u, 364u)), target_bits);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.gaussian_weights()[364u * field_width + 300u]), std::bit_cast<std::uint32_t>(1.0f));
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output_at(10u, 100u)), std::bit_cast<std::uint32_t>(base));
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.gaussian_weights()[100u * field_width + 10u]), 0u);
      }
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, GaussianActualExpSweepBoundsFractionalCentersAndFiniteUnderflow) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.0395f);
    const line_box_t line {180u, 360u, 590u, 390u};
    ASSERT_TRUE(fixture.observe_at(435u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(436u, {line}, 100001u));
    ASSERT_TRUE(fixture.observe_at(437u, {line}, 100002u));
    ASSERT_TRUE(fixture.observe_at(438u, {line}, 100003u));
    ASSERT_TRUE(fixture.observe_at(439u, {line}, 100004u));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_warmup_word], 0u);
    const std::array<float, 14u> samples {
      0.0f, 0.0001f, 0.01f, 0.125f, 0.5f, 1.0f, 2.25f, 10.5f, 25.0f, 100.0f,
      1000.25f, 10000.0f, 1000000.0f, std::numeric_limits<float>::infinity()
    };
    std::vector<float> q(fixture.base().size());
    for (std::size_t i = 0u; i < q.size(); ++i) q[i] = samples[i % samples.size()];
    ASSERT_TRUE(fixture.condition_with_distance_field(q));
    ASSERT_EQ(fixture.gaussian_control(), 1u);
    const float sigma = 1080.0f / v2::subtitle_contour_gaussian_sigma_short_side_divisor;
    volatile float sigma_squared = sigma * sigma;
    volatile float denominator = 2.0f * sigma_squared;
    double maximum_error = 0.0;
    for (std::size_t i = 0u; i < q.size(); ++i) {
      const auto actual = fixture.gaussian_weights()[i];
      ASSERT_TRUE(std::isfinite(actual));
      const double expected = std::isfinite(q[i]) ? std::exp(-static_cast<double>(q[i]) / denominator) : 0.0;
      maximum_error = std::max(maximum_error, std::abs(static_cast<double>(actual) - expected));
      if (q[i] == 0.0f) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual), std::bit_cast<std::uint32_t>(1.0f));
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output()[i]), fixture.condition_params()[5u]);
      } else if (!std::isfinite(q[i]) || q[i] == 1000000.0f) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual), 0u);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output()[i]), std::bit_cast<std::uint32_t>(fixture.base()[i]));
      }
    }
    RecordProperty("gaussian_weight_max_absolute_error", maximum_error);
    EXPECT_LE(maximum_error, v2::subtitle_contour_gaussian_weight_absolute_error_limit);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, GaussianEmptyAndInvalidFirstContentNodeDisableExactBase) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    const line_box_t line {180u, 360u, 590u, 390u};
    ASSERT_TRUE(fixture.observe_at(440u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(441u, {line}, 100001u));
    ASSERT_TRUE(fixture.observe_at(442u, {line}, 100002u));
    ASSERT_TRUE(fixture.observe_at(443u, {line}, 100003u));
    ASSERT_TRUE(fixture.observe_at(444u, {line}, 100004u));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_warmup_word], 0u);
    for (const auto bad_q : {std::numeric_limits<float>::infinity(), -1.0f,
           std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::infinity()}) {
      std::vector<float> q(fixture.base().size(), 0.0f);
      q[0u] = bad_q;
      ASSERT_TRUE(fixture.condition_with_distance_field(q));
      EXPECT_EQ(fixture.gaussian_control(), 0u);
      EXPECT_TRUE(fixture.output_is_exact_base());
      EXPECT_TRUE(std::all_of(fixture.gaussian_weights().begin(), fixture.gaussian_weights().end(),
        [](const float value) { return std::bit_cast<std::uint32_t>(value) == 0u; }));
    }
    ASSERT_TRUE(fixture.set_authored_contour_mask(std::vector<std::uint32_t>(fixture.base().size(), 0u)));
    ASSERT_TRUE(fixture.condition_only());
    EXPECT_EQ(fixture.gaussian_control(), 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    EXPECT_EQ(fixture.condition_params()[2u], 1u);  // No box fallback when temporal support is empty.
  }

  TEST(HostSbsSubtitleSlr13GpuTest, GaussianProposalExtendsNearestContentAndDuplicateIsBitExact) {
    const tensor_content_t content {0u, 89u, 770u, 345u};
    subtitle_plane_fixture_t fixture(770u, 434u, 1500u, 500u,
      std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max(), content);
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    const auto bottom = subtitle_roi_edge(1500u, 500u, content, v2::subtitle_ocr_safe_row_bottom);
    const line_box_t line {180u, bottom - 20u, 590u, bottom};
    ASSERT_TRUE(fixture.observe_at(450u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(451u, {line}, 100001u));
    ASSERT_TRUE(fixture.observe_at(452u, {line}, 100002u));
    ASSERT_TRUE(fixture.observe_at(453u, {line}, 100003u));
    ASSERT_TRUE(fixture.observe_at(454u, {line}, 100004u));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_warmup_word], 0u);
    ASSERT_EQ(fixture.gaussian_control(), 1u);
    for (std::uint32_t y = 0u; y < field_height; ++y) {
      for (std::uint32_t x = 0u; x < field_width; ++x) {
        const auto index = static_cast<std::size_t>(y) * field_width + x;
        const auto cy = std::clamp(y, content[1u], content[3u] - 1u);
        const auto ci = static_cast<std::size_t>(cy) * field_width + x;
        if (y != cy) {
          EXPECT_EQ(fixture.contour_distances()[index], contour_distance_infinity);
        }
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output()[index]), std::bit_cast<std::uint32_t>(fixture.output()[ci]));
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.gaussian_weights()[index]), std::bit_cast<std::uint32_t>(fixture.gaussian_weights()[ci]));
      }
    }
    const auto output = fixture.output(), weight = fixture.gaussian_weights(), q = fixture.contour_distances();
    const auto params = fixture.condition_params();
    const auto identity = fixture.contour_identity();
    ASSERT_TRUE(fixture.observe_at(454u, {line}, 100004u));
    EXPECT_EQ(fixture.condition_params(), params);
    EXPECT_EQ(fixture.contour_identity(), identity);
    EXPECT_EQ(fixture.contour_distances(), q);
    EXPECT_EQ(fixture.gaussian_weights(), weight);
    EXPECT_EQ(fixture.output(), output);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, GaussianProposalRejectsEveryStaleOrMalformedIdentityBinding) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(0.03f);
    fixture.set_cut(17u, false);
    const line_box_t line {180u, 360u, 590u, 390u};
    ASSERT_TRUE(fixture.set_authored_contour_mask(authored_hollow_and_disjoint_contours()));
    constexpr std::uint64_t frame = 0x100000010ull, source_time = 0x200000064ull;
    ASSERT_TRUE(fixture.observe_at(frame, {line}, source_time));
    ASSERT_TRUE(fixture.observe_at(frame + 1u, {line}, source_time + 100000u));
    ASSERT_EQ(fixture.gaussian_control(), 1u);
    const auto identity = fixture.contour_identity();
    const auto output = fixture.output();
    for (std::size_t word = 0u; word < identity.size(); ++word) {
      SCOPED_TRACE("SCT5 word " + std::to_string(word));
      ASSERT_TRUE(fixture.condition_with_contour_identity_word(word, identity[word] ^ 1u));
      EXPECT_TRUE(fixture.output_is_exact_base());
      EXPECT_EQ(fixture.gaussian_control(), 0u);
      ASSERT_TRUE(fixture.condition_with_contour_identity_word(word, identity[word]));
      EXPECT_EQ(fixture.gaussian_control(), 1u);
      EXPECT_EQ(fixture.output(), output);
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, LiteralGaussianProposalCanExceedSlopeAndRequiresSharedPostConditioning) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    // The valid representation endpoint preserves an exposed-arc slope witness at the wider sigma.
    fixture.set_base(v2::direct_container_limit);
    const line_box_t line {180u, 360u, 590u, 390u};
    ASSERT_TRUE(fixture.set_authored_contour_mask(authored_hollow_and_disjoint_contours()));
    ASSERT_TRUE(fixture.observe_at(460u, {line}, 1u));
    ASSERT_TRUE(fixture.observe_at(461u, {line}, 100001u));
    ASSERT_TRUE(fixture.observe_at(462u, {line}, 100002u));
    ASSERT_TRUE(fixture.observe_at(463u, {line}, 100003u));
    ASSERT_TRUE(fixture.observe_at(464u, {line}, 100004u));
    ASSERT_EQ(fixture.state()[v2::subtitle_plane_warmup_word], 0u);
    float maximum_horizontal = 0.0f, maximum_vertical = 0.0f;
    for (std::uint32_t y = 0u; y < field_height; ++y) {
      for (std::uint32_t x = 0u; x < field_width; ++x) {
        const auto i = static_cast<std::size_t>(y) * field_width + x;
        ASSERT_TRUE(std::isfinite(fixture.output()[i]));
        if (x != 0u) maximum_horizontal = std::max(maximum_horizontal, std::abs(fixture.output()[i] - fixture.output()[i - 1u]));
        if (y != 0u) maximum_vertical = std::max(maximum_vertical, std::abs(fixture.output()[i] - fixture.output()[i - field_width]));
      }
    }
    // This fixture executes the literal proposal writer only. Shared subtitle_main tests
    // independently qualify the final C75/Q30 field; no exact re-pin is asserted afterward.
    EXPECT_GT(maximum_horizontal, v2::max_horizontal_slope / field_width);
    EXPECT_GT(maximum_vertical, v2::max_vertical_shear / field_width);
  }


  TEST(HostSbsSubtitleSlr13GpuTest, ExactSavedWmfAndBrowserEightPairsAllAuthorizeCurrentCoverage) {
    // Literal paired vectors from authenticated native WMF17 and browser66 records. This
    // geometry regression has authored support; it makes no claim about real glyph recall.
    const std::vector<ocr_box_t> wmf {
      {{211u,650u,305u,697u},{181u,650u,336u,727u},0u,1u,0u,0x3f6aaff2u},
      {{402u,650u,1146u,700u},{349u,650u,1199u,753u},0u,1u,0u,0x3f7139d8u},
      {{214u,708u,312u,722u},{206u,700u,320u,730u},0u,1u,0u,0x3f755849u},
      {{425u,700u,1054u,733u},{402u,678u,1077u,756u},0u,2u,0u,0x3f708706u},
      {{214u,734u,1212u,767u},{192u,712u,1234u,790u},0u,2u,0u,0x3f76626au},
      {{198u,773u,262u,785u},{190u,765u,270u,793u},0u,1u,0u,0x3f73a341u},
      {{686u,776u,751u,785u},{680u,769u,758u,791u},0u,1u,0u,0x3f244ea2u},
      {{712u,851u,742u,860u},{705u,845u,748u,860u},0u,1u,0u,0x3f523770u}
    };
    const std::vector<ocr_box_t> browser {
      {{447u,691u,737u,708u},{436u,679u,748u,719u},0u,1u,0u,0x3f78d35bu},
      {{1317u,692u,1401u,703u},{1310u,686u,1407u,709u},0u,1u,0u,0x3f738c7au},
      {{405u,731u,767u,748u},{394u,720u,779u,759u},0u,1u,0u,0x3f7b441bu},
      {{1275u,774u,1354u,785u},{1268u,768u,1361u,791u},0u,2u,0u,0x3f6bafa5u},
      {{818u,797u,1106u,807u},{810u,789u,1114u,815u},0u,1u,0u,0x3f6536ceu},
      {{1251u,794u,1404u,823u},{1233u,776u,1422u,841u},0u,2u,1u,0x3f6aebc8u},
      {{49u,839u,268u,851u},{41u,831u,276u,859u},0u,3u,0u,0x3f7279b4u},
      {{999u,839u,1101u,851u},{991u,831u,1109u,859u},0u,2u,0u,0x3f70388bu}
    };
    for (const bool is_wmf : {true, false}) {
      subtitle_plane_fixture_t fixture(1540u,868u,is_wmf ? 1920u : 2560u,is_wmf ? 1080u : 1440u);
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      fixture.set_base(0.03f);
      const auto &boxes = is_wmf ? wmf : browser;
      ASSERT_TRUE(fixture.observe_at(is_wmf ? 17u : 66u, boxes, is_wmf ? 6676000u : 2166667u));
      EXPECT_EQ(fixture.state()[3u], 255u);
      EXPECT_EQ(fixture.condition_params()[2u], 8u);
      for (std::uint32_t slot = 0u; slot < boxes.size(); ++slot) {
        const auto offset = final_box_offset + slot * v2::subtitle_ocr_box_word_count;
        EXPECT_EQ(fixture.ocr_record()[offset], boxes[slot].cover.left);
        EXPECT_EQ(fixture.ocr_record()[offset + 3u], boxes[slot].cover.bottom);
        EXPECT_EQ(fixture.ocr_record()[offset + 4u], boxes[slot].score_bits);
        const auto x = (boxes[slot].cover.left + boxes[slot].cover.right) / 2u;
        const auto y = boxes[slot].cover.bottom - 1u;
        EXPECT_FLOAT_EQ(fixture.output_at(x,y), std::bit_cast<float>(fixture.condition_params()[5u]));
      }
      std::vector<std::uint8_t> union_cells(1540u * 868u, 0u);
      for (const auto &box : boxes)
        for (std::uint32_t y = box.cover.top; y < box.cover.bottom; ++y)
          std::fill(union_cells.begin() + y * 1540u + box.cover.left,
            union_cells.begin() + y * 1540u + box.cover.right, 1u);
      EXPECT_EQ(fixture.state()[v2::subtitle_plane_adaptive_offset + 13u],
        static_cast<std::uint32_t>(std::count(union_cells.begin(), union_cells.end(), 1u)));
    }
  }

  void verify_current_ui_source_domains(const D3D_DRIVER_TYPE driver) {
    struct domain_t { std::uint32_t w,h,sw,sh; tensor_content_t content; };
    const std::array cases {
      domain_t{770u,434u,1920u,1080u,{0u,0u,770u,434u}},
      domain_t{1540u,868u,1320u,720u,{0u,14u,1540u,854u}},
      domain_t{1540u,868u,2536u,1427u,{0u,1u,1540u,867u}},
      domain_t{1540u,868u,3840u,2088u,{0u,15u,1540u,853u}},
      domain_t{770u,434u,1500u,500u,{0u,89u,770u,345u}},
      domain_t{868u,1540u,1080u,1920u,{0u,0u,868u,1540u}}
    };
    const auto bytes = [](const auto &words) {
      std::vector<std::uint8_t> result(words.size() * sizeof(std::uint32_t));
      std::memcpy(result.data(),words.data(),result.size());
      return result;
    };
    for (const auto &domain : cases) {
      SCOPED_TRACE(std::to_string(domain.sw) + "x" + std::to_string(domain.sh));
      subtitle_plane_fixture_t fixture(domain.w,domain.h,domain.sw,domain.sh,
        std::numeric_limits<std::uint32_t>::max(),std::numeric_limits<std::uint32_t>::max(),domain.content);
      std::string error;
      ASSERT_TRUE(fixture.initialize(error,driver)) << error;
      fixture.set_cut(3u,false);
      const auto top = subtitle_roi_edge(domain.sw,domain.sh,domain.content,v2::subtitle_ocr_safe_row_top);
      const line_box_t text {20u,top,40u,top + 3u};
      constexpr std::uint64_t now = 0x200000001ull;
      ASSERT_TRUE(fixture.observe_at(1u,{text},now));
      EXPECT_EQ(fixture.condition_params()[2u],1u);
      EXPECT_EQ(fixture.state()[10u],domain.sw);
      EXPECT_EQ(fixture.state()[11u],domain.sh);
      EXPECT_EQ(fixture.state()[12u],domain.w);
      EXPECT_EQ(fixture.state()[13u],domain.h);
      EXPECT_EQ(fixture.state()[9u],2u);
      platf::sbs_debug::frame completed;
      completed.model_width = completed.raw_width = completed.field_width = domain.w;
      completed.model_height = completed.raw_height = completed.field_height = domain.h;
      completed.field_content = {domain.content[0u],domain.content[1u],domain.content[2u],domain.content[3u]};
      completed.matched_frame_id = 1u;
      completed.matched_observation_timestamp_us = now;
      completed.parallax_v2_joint_plane_mode = v2::adaptive_policy_id;
      completed.parallax_v2_requested_gain = 1.0f;
      completed.depth_input_region = {.source_width=domain.sw,.source_height=domain.sh,
        .left=0u,.top=0u,.right=domain.sw,.bottom=domain.sh,.tensor_content=completed.field_content,
        .analysis_generation=1u,.authority=models::depth_analysis_authority_e::foreground_client};
      EXPECT_TRUE(platf::sbs_debug::detail::subtitle_records_match_frame(
        bytes(fixture.ocr_record()),bytes(fixture.state()),completed,3u));
      auto changed = fixture.state();
      changed[v2::subtitle_plane_observation_timestamp_offset] ^= 1u;
      EXPECT_FALSE(platf::sbs_debug::detail::subtitle_records_match_frame(
        bytes(fixture.ocr_record()),bytes(changed),completed,3u));
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, DirectCurrentUiNativeDumpAuthenticatesSourceDomains) {
    verify_current_ui_source_domains(D3D_DRIVER_TYPE_WARP);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, DirectCurrentUiHardwareDumpAuthenticatesSourceDomains) {
    verify_current_ui_source_domains(D3D_DRIVER_TYPE_HARDWARE);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, UnprojectableRoiAndMalformedBoxCannotAuthorizeCurrentProtection) {
    for (const bool bad_roi : {true,false}) {
      subtitle_plane_fixture_t fixture(field_width,field_height,1920u,1080u,
        bad_roi ? roi_top + 1u : roi_top,roi_bottom);
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      const line_box_t line {180u,360u,590u,370u};
      ASSERT_TRUE(fixture.observe(1u,{line},false,true,false,!bad_roi));
      EXPECT_EQ(fixture.state()[3u],0u);
      EXPECT_EQ(fixture.condition_params()[2u],0u);
      EXPECT_EQ(fixture.gaussian_control(),0u);
      EXPECT_TRUE(fixture.output_is_exact_base());
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, FullUiRepresentationRangeStaysIndependentFromArtisticGain) {
    subtitle_plane_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_adaptive_policy_words(v2::adaptive_policy_id,0u,0.0065625f);
    fixture.set_base(0.012f);
    const line_box_t line {180u,360u,590u,370u};
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    ASSERT_TRUE(fixture.observe_at(1u,{line},1u));
    ASSERT_TRUE(fixture.observe_at(2u,{line},100001u));
    EXPECT_EQ(fixture.state()[a + 2u],7u);
    EXPECT_EQ(fixture.state()[a + 12u],std::bit_cast<std::uint32_t>(v2::direct_container_limit));
    for (std::uint64_t step=1u;step<=4u;++step)
      ASSERT_TRUE(fixture.observe_at(2u+step,{line},100001u+step*250000u));
    const auto applied=std::bit_cast<float>(fixture.state()[a + 3u]);
    EXPECT_FLOAT_EQ(applied,28.0f/1920.0f);
    EXPECT_GE(applied,0.012f+2.0f/1920.0f);
    EXPECT_GT(applied,0.0065625f);
    EXPECT_EQ(fixture.condition_params()[5u],fixture.state()[a + 3u]);
  }

}  // namespace

#else

TEST(HostSbsSubtitleSlr13GpuTest, WindowsOnly) {
  GTEST_SKIP() << "Direct OCR/UI plane GPU tests require Windows D3D11";
}

#endif
