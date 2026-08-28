/**
 * @file tests/unit/test_host_sbs_subtitle_locator.cpp
 * @brief Deterministic WARP coverage for the compact OCR8/SLR13 lower-text authority.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/generated/depth_coordinate_v2_contract.h"
  #include "src/generated/sbs_adaptive_state_contract.h"
  #include "src/host_sbs_shader_cache.h"

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstddef>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
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

  static_assert([] {
    for (const auto &spec :
         models::host_sbs_shader_cache::parallax_v2_producer_specs) {
      if (spec.filename.find("condition_validate_test") != std::string_view::npos) {
        return false;
      }
    }
    return true;
  }(), "the adversarial condition validator must remain outside the production closure");

  constexpr std::uint32_t flag_owner = 1u;
  constexpr std::uint32_t flag_pending = 2u;
  constexpr std::uint32_t flag_target_valid = 4u;
  constexpr std::uint32_t flag_target_reset = 8u;
  constexpr std::uint32_t flag_provisional_current =
    v2::subtitle_locator_provisional_current_flag;
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

  static_assert(sizeof(depth_constants_t) == 64u);
  static_assert(sizeof(v2_constants_t) == 32u);
  static_assert(sizeof(subtitle_constants_t) == 64u);

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

  class slr13_warp_fixture_t {
   public:
    explicit slr13_warp_fixture_t(
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
      const bool submitted = true,
      const bool = true
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
        {
          source_width_,
          source_height_,
          submitted ? 1u : 0u,
          reset ? 1u : 0u,
        },
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
      std::array<ID3D11UnorderedAccessView *, 3u> resolve_uavs {
        state_uav_.Get(), nullptr, condition_params_uav_.Get()
      };
      context_->CSSetUnorderedAccessViews(2u, resolve_uavs.size(), resolve_uavs.data(), nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind();

      if (!dispatch_conditioner()) return false;

      return read_buffer(device_.Get(), context_.Get(), state_buffer_.Get(), state_) &&
             read_buffer(
               device_.Get(), context_.Get(), condition_params_buffer_.Get(), condition_params_) &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    bool condition_only() {
      if (!dispatch_condition_validator_for_test()) return false;
      if (!dispatch_conditioner()) return false;
      return read_buffer(
               device_.Get(), context_.Get(), condition_params_buffer_.Get(), condition_params_) &&
             read_texture(device_.Get(), context_.Get(), output_texture_.Get(), output_);
    }

    void set_cut(const std::uint32_t scene_epoch, const bool pulse) {
      scene_epoch_ = scene_epoch;
      cut_pulse_ = pulse;
    }

    const std::vector<std::uint32_t> &state() const { return state_; }
    const std::vector<std::uint32_t> &condition_params() const { return condition_params_; }
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
      const std::uint32_t owner_top,
      const float outer_value,
      const float inner_value
    ) {
      const auto cell_scale = fixture_field_cell_scale();
      const auto outer_y = std::clamp(
        owner_top >= 10u * cell_scale ? owner_top - 10u * cell_scale : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto inner_y = std::clamp(
        owner_top >= 4u * cell_scale ? owner_top - 4u * cell_scale : 0u,
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
      const std::uint32_t owner_top,
      const float first_value,
      const float second_value
    ) {
      const auto cell_scale = fixture_field_cell_scale();
      const auto outer_y = std::clamp(
        owner_top >= 10u * cell_scale ? owner_top - 10u * cell_scale : 0u,
        tensor_content_[1u],
        tensor_content_[3u] - 1u
      );
      const auto inner_y = std::clamp(
        owner_top >= 4u * cell_scale ? owner_top - 4u * cell_scale : 0u,
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

    void set_target_probe_ring_alternating(
      const float primary_center,
      const std::uint32_t owner_top,
      const std::uint32_t horizontal_span,
      const float first_value,
      const float second_value
    ) {
      set_background_probe_alternating(
        primary_center, owner_top, first_value, second_value
      );
      const float step = static_cast<float>(horizontal_span) /
                         static_cast<float>(
                           v2::subtitle_target_horizontal_step_denominator
                         );
      for (std::uint32_t radius = 1u;
           radius <= v2::subtitle_target_horizontal_fallback_max_radius_steps;
           ++radius) {
        const float offset = step * static_cast<float>(radius);
        set_background_probe_alternating(
          primary_center - offset, owner_top, first_value, second_value
        );
        set_background_probe_alternating(
          primary_center + offset, owner_top, first_value, second_value
        );
      }
    }

    void set_target_probe_ring_alternating(
      const line_box_t line,
      const float first_value,
      const float second_value
    ) {
      set_target_probe_ring_alternating(
        0.5f * static_cast<float>(line.left + line.right - 1u),
        line.top,
        line.right - line.left,
        first_value,
        second_value
      );
    }

    std::uint32_t fixture_field_cell_scale() const {
      return std::max(
        1u,
        v2::subtitle_locator_field_cell_scale(field_width_, field_height_)
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
    bool dispatch_condition_validator_for_test() {
      ID3D11Buffer *constant_buffers[] = {depth_cb_.Get(), v2_cb_.Get(), subtitle_cb_.Get()};
      context_->CSSetConstantBuffers(0u, 3u, constant_buffers);

      context_->CSSetShader(condition_validate_test_.Get(), nullptr, 0u);
      std::array<ID3D11ShaderResourceView *, 8u> validator_srvs {
        nullptr, cut_srv_.Get(), nullptr, state_srv_.Get(), nullptr, nullptr, nullptr, ocr_srv_.Get()
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
      std::array<ID3D11ShaderResourceView *, 5u> condition_srvs {
        nullptr, nullptr, base_srv_.Get(), state_srv_.Get(), condition_params_srv_.Get()
      };
      context_->CSSetShaderResources(0u, condition_srvs.size(), condition_srvs.data());
      context_->CSSetUnorderedAccessViews(3u, 1u, output_uav_.GetAddressOf(), nullptr);
      context_->Dispatch(
        (field_width_ + 15u) / 16u,
        (field_height_ + 15u) / 16u,
        1u
      );
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

  TEST(HostSbsSubtitleSlr13GpuTest, ConfirmsExactFrameStacksAndConditionsOnlyCurrentLines) {
    slr13_warp_fixture_t fixture;
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
    EXPECT_LT(fixture.output_at(300u, 364u), 0.025f);
    EXPECT_EQ(
      std::bit_cast<std::uint32_t>(fixture.output_at(10u, 100u)),
      std::bit_cast<std::uint32_t>(0.03f)
    );

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
    EXPECT_EQ(fixture.state()[24u], 2u);
    const float first_core = fixture.output_at(300u, 364u);
    const float gap = fixture.output_at(300u, 372u);
    const float second_core = fixture.output_at(300u, 377u);
    EXPECT_LT(first_core, gap);
    EXPECT_LT(second_core, gap);
    EXPECT_LT(gap, 0.03f);

    // Authoritative empty OCR removes geometry immediately.  The cached target/grace is not
    // geometry authority, so every texel is exact Base on the death frame and thereafter.
    ASSERT_TRUE(fixture.observe(6u, {}, false));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 2u);
    EXPECT_EQ(fixture.state()[25u], 6u);
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

  TEST(HostSbsSubtitleSlr13GpuTest, SameBaselineSegmentsShareOwnerWithoutUnioningCovers) {
    slr13_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    // A detector gap just beyond the four-cell ordinary join becomes two OCR8 pairs. Their cores
    // remain strongly baseline-aligned and near enough to form one SLR13 owner, but each paired
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
    slr13_warp_fixture_t guarded;
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
    slr13_warp_fixture_t overflow;
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
    slr13_warp_fixture_t bridge;
    ASSERT_TRUE(bridge.initialize(error)) << error;
    ASSERT_TRUE(bridge.observe(12u, {chain_a, chain_b, chain_c}, false));
    EXPECT_EQ(bridge.state()[2u], 0u);
    EXPECT_EQ(bridge.state()[12u], 0u);
    EXPECT_TRUE(bridge.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, CurrentOwnerMatchPreemptsCompatiblePendingJitter) {
    slr13_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const ocr_box_t owner {
      {100u, 360u, 500u, 370u},
      {96u, 356u, 504u, 374u},
      0u,
      1u,
      0u,
    };
    const ocr_box_t pending {
      {300u, 360u, 700u, 370u},
      {296u, 356u, 704u, 374u},
      0u,
      1u,
      0u,
    };
    const ocr_box_t recovered {
      {200u, 360u, 600u, 370u},
      {196u, 356u, 604u, 374u},
      0u,
      1u,
      0u,
    };

    ASSERT_TRUE(fixture.observe(60u, {owner}, false));
    ASSERT_TRUE(fixture.observe(61u, {owner}, false));
    ASSERT_TRUE(fixture.observe(62u, {owner}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    ASSERT_EQ(fixture.state()[24u], 2u);
    const auto generation = fixture.state()[3u];

    // The material shift starts a pending handoff and has no current owner match.
    ASSERT_TRUE(fixture.observe(63u, {pending}, false));
    ASSERT_EQ(
      fixture.state()[2u], flag_owner | flag_pending | flag_target_valid
    );
    ASSERT_EQ(fixture.state()[12u], 1u);
    ASSERT_EQ(fixture.state()[20u], 0u);

    // This midpoint has exactly 0.6 IoU with both the established owner and pending geometry.
    // Owner continuity must win: confirming pending here would bump generation and restart fade.
    ASSERT_TRUE(fixture.observe(64u, {recovered}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], generation);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 0u);
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_EQ(fixture.state()[32u], recovered.core.left);
    EXPECT_EQ(fixture.state()[34u], recovered.core.right);
    EXPECT_EQ(fixture.state()[64u], recovered.cover.left);
    EXPECT_EQ(fixture.state()[66u], recovered.cover.right);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, CountChangingResegmentationUsesPendingConfirmation) {
    slr13_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const ocr_box_t merged {
      {180u, 360u, 590u, 372u},
      {170u, 350u, 600u, 382u},
      0u,
      3u,
      0u,
    };
    const ocr_box_t first {
      {180u, 360u, 290u, 372u},
      {178u, 358u, 292u, 374u},
      0u,
      1u,
      0u,
    };
    const ocr_box_t second {
      {300u, 360u, 410u, 372u},
      {298u, 358u, 412u, 374u},
      0u,
      1u,
      0u,
    };
    const ocr_box_t third {
      {420u, 360u, 590u, 372u},
      {418u, 358u, 592u, 374u},
      0u,
      1u,
      0u,
    };
    const ocr_box_t current_merge {
      merged.core,
      {174u, 354u, 596u, 378u},
      0u,
      3u,
      0u,
    };
    const std::vector<ocr_box_t> split {first, second, third};

    ASSERT_TRUE(fixture.observe(100u, {merged}, false));
    ASSERT_TRUE(fixture.observe(101u, {merged}, false));
    ASSERT_TRUE(fixture.observe(102u, {merged}, false));
    ASSERT_EQ(fixture.state()[24u], 2u);
    const auto generation = fixture.state()[3u];

    // One detector core becoming three aligned cores is still a material count change. Aggregate
    // overlap alone grants no authority: the first observation is pending and renders exact Base.
    ASSERT_TRUE(fixture.observe(103u, split, false));
    EXPECT_EQ(
      fixture.state()[2u], flag_owner | flag_pending | flag_target_valid
    );
    EXPECT_EQ(fixture.state()[3u], generation);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 3u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 0u);
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_TRUE(fixture.output_is_exact_base());

    // A second distinct compatible observation confirms the ordinary handoff. Only then do its
    // exact paired covers become current; mature target/fade preservation remains unchanged.
    ASSERT_TRUE(fixture.observe(104u, split, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_NE(fixture.state()[3u], generation);
    EXPECT_EQ(fixture.state()[4u], 3u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 3u);
    EXPECT_EQ(fixture.state()[21u], 3u);
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_EQ(fixture.state()[64u], first.cover.left);
    EXPECT_EQ(fixture.state()[66u], first.cover.right);
    EXPECT_EQ(fixture.state()[68u], second.cover.left);
    EXPECT_EQ(fixture.state()[70u], second.cover.right);
    EXPECT_EQ(fixture.state()[72u], third.cover.left);
    EXPECT_EQ(fixture.state()[74u], third.cover.right);

    // The reverse three-to-one merge follows the same two-observation transaction.
    const auto split_generation = fixture.state()[3u];
    ASSERT_TRUE(fixture.observe(105u, {current_merge}, false));
    EXPECT_EQ(
      fixture.state()[2u], flag_owner | flag_pending | flag_target_valid
    );
    EXPECT_EQ(fixture.state()[3u], split_generation);
    EXPECT_EQ(fixture.state()[4u], 3u);
    EXPECT_EQ(fixture.state()[12u], 1u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.observe(106u, {current_merge}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_NE(fixture.state()[3u], split_generation);
    EXPECT_EQ(fixture.state()[4u], 1u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 3u);
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_EQ(fixture.state()[64u], current_merge.cover.left);
    EXPECT_EQ(fixture.state()[65u], current_merge.cover.top);
    EXPECT_EQ(fixture.state()[66u], current_merge.cover.right);
    EXPECT_EQ(fixture.state()[67u], current_merge.cover.bottom);

    // A cut still cannot carry the old target/geometry into a count-changing stack.
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(107u, split, false));
    EXPECT_EQ(fixture.state()[2u], flag_pending);
    EXPECT_EQ(fixture.state()[3u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 3u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 2u);
    EXPECT_EQ(fixture.state()[24u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, CountChangingStacksDoNotBorrowAggregateOverlap) {
    std::string error;

    // Aggregate bbox IoU is above 0.6, but the added same-baseline core overlaps no old member.
    // It remains a normal pending addition and only the exactly matched owner cover is current.
    slr13_warp_fixture_t disjoint;
    ASSERT_TRUE(disjoint.initialize(error)) << error;
    const ocr_box_t old_line {
      {180u, 360u, 500u, 372u}, {176u, 356u, 504u, 376u}, 0u, 1u, 0u,
    };
    const ocr_box_t added {
      {510u, 360u, 600u, 372u}, {508u, 358u, 602u, 374u}, 0u, 1u, 0u,
    };
    ASSERT_TRUE(disjoint.observe(120u, {old_line}, false));
    ASSERT_TRUE(disjoint.observe(121u, {old_line}, false));
    ASSERT_TRUE(disjoint.observe(122u, {old_line}, false));
    const auto disjoint_generation = disjoint.state()[3u];
    ASSERT_TRUE(disjoint.observe(123u, {old_line, added}, false));
    EXPECT_EQ(
      disjoint.state()[2u], flag_owner | flag_pending | flag_target_valid
    );
    EXPECT_EQ(disjoint.state()[3u], disjoint_generation);
    EXPECT_EQ(disjoint.state()[4u], 1u);
    EXPECT_EQ(disjoint.state()[12u], 2u);
    EXPECT_EQ(disjoint.state()[20u], 1u);
    EXPECT_EQ(disjoint.state()[64u], old_line.cover.left);
    EXPECT_EQ(disjoint.state()[66u], old_line.cover.right);

    // A one-cell overlap cannot borrow the old core's high aggregate IoU to make mostly-new
    // geometry immediately current. Ordinary per-member matching leaves this stack pending.
    slr13_warp_fixture_t grazing_split;
    error.clear();
    ASSERT_TRUE(grazing_split.initialize(error)) << error;
    const ocr_box_t wide_owner {
      {100u, 360u, 600u, 372u}, {96u, 356u, 604u, 376u}, 0u, 1u, 0u,
    };
    const ocr_box_t retained_segment {
      {100u, 360u, 520u, 372u}, {98u, 358u, 522u, 374u}, 0u, 1u, 0u,
    };
    const ocr_box_t grazing_segment {
      {599u, 360u, 690u, 372u}, {597u, 358u, 692u, 374u}, 0u, 1u, 0u,
    };
    ASSERT_TRUE(grazing_split.observe(124u, {wide_owner}, false));
    ASSERT_TRUE(grazing_split.observe(125u, {wide_owner}, false));
    ASSERT_TRUE(grazing_split.observe(
      126u, {retained_segment, grazing_segment}, false));
    EXPECT_EQ(
      grazing_split.state()[2u], flag_owner | flag_pending | flag_target_valid
    );
    EXPECT_EQ(grazing_split.state()[4u], 1u);
    EXPECT_EQ(grazing_split.state()[12u], 2u);
    EXPECT_EQ(grazing_split.state()[20u], 1u);
    EXPECT_EQ(grazing_split.state()[64u], retained_segment.cover.left);
    EXPECT_EQ(grazing_split.state()[66u], retained_segment.cover.right);

    // The reverse change is also a pending merge: no old member has 0.6 IoU, so aggregate contact
    // cannot create current authority.
    slr13_warp_fixture_t grazing_merge;
    error.clear();
    ASSERT_TRUE(grazing_merge.initialize(error)) << error;
    const line_box_t old_left {100u, 360u, 450u, 392u};
    const line_box_t old_grazing {699u, 360u, 764u, 392u};
    const line_box_t merged_current {100u, 360u, 700u, 392u};
    ASSERT_TRUE(grazing_merge.observe(127u, {old_left, old_grazing}, false));
    ASSERT_TRUE(grazing_merge.observe(128u, {old_left, old_grazing}, false));
    ASSERT_TRUE(grazing_merge.observe(129u, {merged_current}, false));
    EXPECT_EQ(
      grazing_merge.state()[2u], flag_owner | flag_pending | flag_target_valid
    );
    EXPECT_EQ(grazing_merge.state()[4u], 2u);
    EXPECT_EQ(grazing_merge.state()[12u], 1u);
    EXPECT_EQ(grazing_merge.state()[20u], 0u);

    // Two vertically separate lines fill the old tall bbox and overlap it individually, but still
    // require ordinary pending confirmation.
    slr13_warp_fixture_t multiline;
    error.clear();
    ASSERT_TRUE(multiline.initialize(error)) << error;
    const line_box_t tall {180u, 350u, 590u, 390u};
    const line_box_t upper {180u, 350u, 590u, 368u};
    const line_box_t lower {180u, 372u, 590u, 390u};
    ASSERT_TRUE(multiline.observe(130u, {tall}, false));
    ASSERT_TRUE(multiline.observe(131u, {tall}, false));
    ASSERT_TRUE(multiline.observe(132u, {upper, lower}, false));
    EXPECT_EQ(
      multiline.state()[2u], flag_owner | flag_pending | flag_target_valid
    );
    EXPECT_EQ(multiline.state()[4u], 1u);
    EXPECT_EQ(multiline.state()[12u], 2u);
    EXPECT_EQ(multiline.state()[20u], 0u);

    // Splitting text beside a ribbon leaves only the exactly matched ribbon current while the
    // complete mixed stack remains pending.
    slr13_warp_fixture_t mixed;
    error.clear();
    ASSERT_TRUE(mixed.initialize(error)) << error;
    const ocr_box_t ribbon {
      {6u, 401u, 693u, roi_bottom},
      {0u, 397u, field_width, field_height},
      box_flag_ribbon,
      7u,
      4u,
    };
    const ocr_box_t merged {
      {180u, 360u, 590u, 372u}, {176u, 356u, 594u, 376u}, 0u, 3u, 0u,
    };
    const ocr_box_t first {
      {180u, 360u, 290u, 372u}, {178u, 358u, 292u, 374u}, 0u, 1u, 0u,
    };
    const ocr_box_t second {
      {300u, 360u, 410u, 372u}, {298u, 358u, 412u, 374u}, 0u, 1u, 0u,
    };
    const ocr_box_t third {
      {420u, 360u, 590u, 372u}, {418u, 358u, 592u, 374u}, 0u, 1u, 0u,
    };
    ASSERT_TRUE(mixed.observe(140u, {merged, ribbon}, false));
    ASSERT_TRUE(mixed.observe(141u, {merged, ribbon}, false));
    ASSERT_TRUE(mixed.observe(142u, {first, second, third, ribbon}, false));
    EXPECT_EQ(mixed.state()[2u], flag_owner | flag_pending | flag_target_valid);
    EXPECT_EQ(mixed.state()[4u], 2u);
    EXPECT_EQ(mixed.state()[12u], 4u);
    EXPECT_EQ(mixed.state()[20u], 1u);
    EXPECT_EQ(
      (mixed.state()[31u] >> current_kind_shift) &
        v2::subtitle_locator_kind_mask,
      1u
    );
    EXPECT_EQ(mixed.state()[64u], ribbon.cover.left);
    EXPECT_EQ(mixed.state()[65u], ribbon.cover.top);
    EXPECT_EQ(mixed.state()[66u], ribbon.cover.right);
    EXPECT_EQ(mixed.state()[67u], ribbon.cover.bottom);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, StackConfirmationRequiresEveryMemberToOverlap) {
    slr13_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, PartialAuthoritySamplesOnlyMatchedCurrentGeometry) {
    slr13_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, NoncanonicalOwnerAndPendingCoreOrderFailFlat) {
    const line_box_t first {180u, 350u, 590u, 360u};
    const line_box_t second {180u, 362u, 500u, 372u};
    std::string error;

    slr13_warp_fixture_t owner_fixture;
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

    slr13_warp_fixture_t pending_fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, OwnerTargetTracksReliableLocalPlaneWithoutPumping) {
    slr13_warp_fixture_t fixture;
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
    // confirmed, its current geometry samples the compatible plane, takes one bounded step, and
    // preserves the mature fade instead of inserting another half-strength frame.
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
    EXPECT_EQ(fixture.state()[24u], 2u);

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
    EXPECT_EQ(fixture.state()[21u], 1u);
    EXPECT_EQ(fixture.state()[24u], 1u);

    // A hard-cut survivor discards the old scene's plane and restarts from same-frame evidence.
    const auto pre_cut_generation = fixture.state()[3u];
    fixture.set_background_sample_rows(handoff, 0.03f, 0.03f);
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(110u, {handoff}, false));
    EXPECT_EQ(fixture.state()[24u], 1u);
    EXPECT_EQ(fixture.state()[18u], std::bit_cast<std::uint32_t>(0.03f));
    EXPECT_EQ(fixture.state()[3u], pre_cut_generation + 1u);

    // A fresh birth retains a reliable plane well beyond the retired 0..8-pixel band.
    slr13_warp_fixture_t local_plane;
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

  TEST(HostSbsSubtitleSlr13GpuTest, SameSceneHandoffPreservesOnlyCompatibleInheritedFade) {
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    const line_box_t owner {180u, 360u, 590u, 370u};
    const line_box_t handoff {80u, 360u, 380u, 370u};
    std::string error;

    // A mature owner and a close, reliable replacement retain full fade after the required
    // pending observation confirms the handoff. The pending observation itself still has no
    // current geometry authority and therefore copies exact Base.
    slr13_warp_fixture_t close;
    ASSERT_TRUE(close.initialize(error)) << error;
    close.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(close.observe(300u, {owner}, false));
    ASSERT_TRUE(close.observe(301u, {owner}, false));
    ASSERT_TRUE(close.observe(302u, {owner}, false));
    ASSERT_EQ(close.state()[24u], 2u);
    close.set_background_sample_rows(
      handoff, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(close.observe(303u, {handoff}, false));
    EXPECT_EQ(close.state()[20u], 0u);
    EXPECT_TRUE(close.output_is_exact_base());
    ASSERT_TRUE(close.observe(304u, {handoff}, false));
    EXPECT_EQ(close.state()[21u], 3u);
    EXPECT_EQ(close.state()[24u], 2u);
    EXPECT_NEAR(
      std::bit_cast<float>(close.state()[18u]),
      target_for_pixels(2.25f),
      1.0e-8f
    );

    // Equality belongs to the compatible side of the existing eight-pixel residual boundary.
    slr13_warp_fixture_t boundary;
    ASSERT_TRUE(boundary.initialize(error)) << error;
    boundary.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(boundary.observe(310u, {owner}, false));
    ASSERT_TRUE(boundary.observe(311u, {owner}, false));
    ASSERT_TRUE(boundary.observe(312u, {owner}, false));
    boundary.set_background_sample_rows(
      handoff, target_for_pixels(10.0f), target_for_pixels(10.0f)
    );
    ASSERT_TRUE(boundary.observe(313u, {handoff}, false));
    ASSERT_TRUE(boundary.observe(314u, {handoff}, false));
    EXPECT_EQ(boundary.state()[21u], 3u);
    EXPECT_EQ(boundary.state()[24u], 2u);
    EXPECT_NEAR(
      std::bit_cast<float>(boundary.state()[18u]),
      target_for_pixels(2.25f),
      1.0e-8f
    );

    // A replacement just beyond the residual boundary starts from its current plane at half
    // strength, exactly as before; it must not inherit either the target or mature fade.
    slr13_warp_fixture_t beyond;
    ASSERT_TRUE(beyond.initialize(error)) << error;
    beyond.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(beyond.observe(320u, {owner}, false));
    ASSERT_TRUE(beyond.observe(321u, {owner}, false));
    ASSERT_TRUE(beyond.observe(322u, {owner}, false));
    beyond.set_background_sample_rows(
      handoff, target_for_pixels(10.25f), target_for_pixels(10.25f)
    );
    ASSERT_TRUE(beyond.observe(323u, {handoff}, false));
    ASSERT_TRUE(beyond.observe(324u, {handoff}, false));
    EXPECT_EQ(beyond.state()[21u], 3u);
    EXPECT_EQ(beyond.state()[24u], 1u);
    EXPECT_EQ(
      beyond.state()[18u],
      std::bit_cast<std::uint32_t>(target_for_pixels(10.25f))
    );

    // Preservation never promotes a half-faded prior owner to full strength.
    slr13_warp_fixture_t half;
    ASSERT_TRUE(half.initialize(error)) << error;
    half.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(half.observe(330u, {owner}, false));
    ASSERT_TRUE(half.observe(331u, {owner}, false));
    ASSERT_EQ(half.state()[24u], 1u);
    half.set_background_sample_rows(
      handoff, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(half.observe(332u, {handoff}, false));
    EXPECT_EQ(half.state()[20u], 0u);
    EXPECT_TRUE(half.output_is_exact_base());
    ASSERT_TRUE(half.observe(333u, {handoff}, false));
    EXPECT_EQ(half.state()[21u], 3u);
    EXPECT_EQ(half.state()[24u], 1u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, ProvisionalPendingBridgeUsesExactCurrentPairOnly) {
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    const ocr_box_t owner {
      {203u, 369u, 566u, 384u}, {198u, 365u, 571u, 389u}, 0u, 1u, 0u
    };
    const ocr_box_t replacement {
      {342u, 368u, 415u, 386u}, {337u, 364u, 420u, 390u}, 0u, 1u, 0u
    };
    slr13_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(fixture.observe(400u, {owner}, false));
    ASSERT_TRUE(fixture.observe(401u, {owner}, false));
    ASSERT_TRUE(fixture.observe(402u, {owner}, false));
    ASSERT_EQ(fixture.state()[24u], 2u);
    const auto durable_generation = fixture.state()[3u];
    const auto durable_target_bits = fixture.state()[18u];
    const auto durable_fade = fixture.state()[24u];

    fixture.set_background_sample_rows(
      replacement.core, target_for_pixels(12.0f), target_for_pixels(12.0f)
    );
    ASSERT_TRUE(fixture.observe(403u, {replacement}, false));
    EXPECT_EQ(
      fixture.state()[2u],
      flag_owner | flag_pending | flag_target_valid | flag_provisional_current
    );
    EXPECT_EQ(fixture.state()[3u], durable_generation);
    EXPECT_EQ(fixture.state()[18u], durable_target_bits);
    EXPECT_EQ(fixture.state()[24u], durable_fade);
    EXPECT_EQ(fixture.state()[12u], 1u);
    EXPECT_EQ(fixture.state()[20u], 1u);
    EXPECT_EQ(fixture.state()[21u], 0u);
    EXPECT_EQ(fixture.state()[25u], 0u);
    EXPECT_EQ(fixture.state()[48u], replacement.core.left);
    EXPECT_EQ(fixture.state()[49u], replacement.core.top);
    EXPECT_EQ(fixture.state()[50u], replacement.core.right);
    EXPECT_EQ(fixture.state()[51u], replacement.core.bottom);
    EXPECT_EQ(fixture.state()[64u], replacement.cover.left);
    EXPECT_EQ(fixture.state()[65u], replacement.cover.top);
    EXPECT_EQ(fixture.state()[66u], replacement.cover.right);
    EXPECT_EQ(fixture.state()[67u], replacement.cover.bottom);
    EXPECT_EQ(
      fixture.condition_params()[4u],
      fixture.state()[v2::subtitle_locator_provisional_fade_word]
    );
    EXPECT_EQ(fixture.state()[v2::subtitle_locator_provisional_fade_word], 1u);
    EXPECT_EQ(
      fixture.condition_params()[5u],
      fixture.state()[v2::subtitle_locator_provisional_target_word]
    );
    EXPECT_FALSE(fixture.output_is_exact_base());

    const auto provisional_state = fixture.state();
    const auto provisional_target_bits =
      fixture.state()[v2::subtitle_locator_provisional_target_word];
    ASSERT_TRUE(fixture.observe(403u, {replacement}, false));
    EXPECT_EQ(fixture.state(), provisional_state);
    EXPECT_EQ(
      fixture.condition_params()[5u],
      fixture.state()[v2::subtitle_locator_provisional_target_word]
    );

    ASSERT_TRUE(fixture.observe(404u, {replacement}, false));
    EXPECT_EQ(
      fixture.state()[2u], flag_owner | flag_target_valid
    );
    EXPECT_EQ(fixture.state()[3u], durable_generation + 1u);
    EXPECT_EQ(fixture.state()[21u], 3u);
    EXPECT_EQ(fixture.state()[18u], provisional_target_bits);
    EXPECT_EQ(fixture.state()[24u], 1u);
    EXPECT_EQ(fixture.state()[v2::subtitle_locator_provisional_target_word], 0u);
    EXPECT_EQ(fixture.state()[v2::subtitle_locator_provisional_fade_word], 0u);
    ASSERT_TRUE(fixture.observe(405u, {replacement}, false));
    EXPECT_EQ(fixture.state()[24u], 2u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, ProvisionalPendingBridgeKeepsFullFadeForSecondTracePair) {
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    const ocr_box_t owner {
      {340u, 368u, 416u, 387u}, {335u, 364u, 421u, 391u}, 0u, 1u, 0u
    };
    const ocr_box_t replacement {
      {290u, 369u, 481u, 385u}, {285u, 365u, 486u, 389u}, 0u, 1u, 0u
    };
    slr13_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(fixture.observe(405u, {owner}, false));
    ASSERT_TRUE(fixture.observe(406u, {owner}, false));
    ASSERT_TRUE(fixture.observe(407u, {owner}, false));
    ASSERT_EQ(fixture.state()[24u], 2u);
    const auto generation = fixture.state()[3u];
    const auto durable_target = fixture.state()[18u];
    fixture.set_background_sample_rows(
      replacement.core, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(fixture.observe(408u, {replacement}, false));
    EXPECT_NE(fixture.state()[2u] & flag_provisional_current, 0u);
    EXPECT_EQ(fixture.state()[3u], generation);
    EXPECT_EQ(fixture.state()[18u], durable_target);
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_EQ(fixture.state()[v2::subtitle_locator_provisional_fade_word], 2u);
    EXPECT_EQ(fixture.condition_params()[4u], 2u);
    EXPECT_EQ(
      fixture.condition_params()[5u],
      fixture.state()[v2::subtitle_locator_provisional_target_word]
    );
    EXPECT_FALSE(fixture.output_is_exact_base());

    ASSERT_TRUE(fixture.observe(409u, {replacement}, false));
    EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(fixture.state()[3u], generation + 1u);
    EXPECT_EQ(fixture.state()[21u], 3u);
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_EQ(fixture.state()[v2::subtitle_locator_provisional_target_word], 0u);
    EXPECT_EQ(fixture.state()[v2::subtitle_locator_provisional_fade_word], 0u);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, ProvisionalPendingBridgeFailsClosedAtSafetyBounds) {
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    const ocr_box_t owner {
      {203u, 368u, 566u, 384u}, {198u, 364u, 571u, 389u}, 0u, 1u, 0u
    };
    const ocr_box_t overlap_boundary {
      {342u, 372u, 415u, 388u}, {337u, 368u, 420u, 392u}, 0u, 1u, 0u
    };
    const ocr_box_t below_overlap_boundary {
      {342u, 373u, 415u, 389u}, {337u, 369u, 420u, 393u}, 0u, 1u, 0u
    };
    std::string error;

    slr13_warp_fixture_t boundary;
    ASSERT_TRUE(boundary.initialize(error)) << error;
    boundary.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(boundary.observe(410u, {owner}, false));
    ASSERT_TRUE(boundary.observe(411u, {owner}, false));
    ASSERT_TRUE(boundary.observe(412u, {owner}, false));
    boundary.set_background_sample_rows(
      overlap_boundary.core, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(boundary.observe(413u, {overlap_boundary}, false));
    EXPECT_NE(boundary.state()[2u] & flag_provisional_current, 0u);
    EXPECT_FALSE(boundary.output_is_exact_base());

    slr13_warp_fixture_t outside;
    ASSERT_TRUE(outside.initialize(error)) << error;
    outside.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(outside.observe(420u, {owner}, false));
    ASSERT_TRUE(outside.observe(421u, {owner}, false));
    ASSERT_TRUE(outside.observe(422u, {owner}, false));
    outside.set_background_sample_rows(
      below_overlap_boundary.core, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(outside.observe(423u, {below_overlap_boundary}, false));
    EXPECT_EQ(outside.state()[2u] & flag_provisional_current, 0u);
    EXPECT_EQ(outside.state()[20u], 0u);
    EXPECT_TRUE(outside.output_is_exact_base());

    const ocr_box_t iou_owner {
      {100u, 360u, 200u, 370u}, {96u, 356u, 204u, 374u}, 0u, 1u, 0u
    };
    const ocr_box_t iou_equality {
      {125u, 360u, 225u, 370u}, {121u, 356u, 229u, 374u}, 0u, 1u, 0u
    };
    slr13_warp_fixture_t match_boundary;
    ASSERT_TRUE(match_boundary.initialize(error)) << error;
    match_boundary.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(match_boundary.observe(424u, {iou_owner}, false));
    ASSERT_TRUE(match_boundary.observe(425u, {iou_owner}, false));
    ASSERT_TRUE(match_boundary.observe(426u, {iou_owner}, false));
    const auto match_generation = match_boundary.state()[3u];
    ASSERT_TRUE(match_boundary.observe(427u, {iou_equality}, false));
    EXPECT_EQ(match_boundary.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(match_boundary.state()[3u], match_generation);
    EXPECT_EQ(match_boundary.state()[12u], 0u);
    EXPECT_EQ(match_boundary.state()[20u], 1u);
    EXPECT_EQ(match_boundary.state()[2u] & flag_provisional_current, 0u);

    slr13_warp_fixture_t unreliable;
    ASSERT_TRUE(unreliable.initialize(error)) << error;
    unreliable.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(unreliable.observe(428u, {owner}, false));
    ASSERT_TRUE(unreliable.observe(429u, {owner}, false));
    ASSERT_TRUE(unreliable.observe(430u, {owner}, false));
    unreliable.set_base(std::numeric_limits<float>::quiet_NaN());
    ASSERT_TRUE(unreliable.observe(431u, {overlap_boundary}, false));
    EXPECT_EQ(unreliable.state()[2u] & flag_provisional_current, 0u);
    EXPECT_EQ(unreliable.state()[20u], 0u);
    EXPECT_TRUE(unreliable.output_is_exact_base());

    slr13_warp_fixture_t half_fade;
    ASSERT_TRUE(half_fade.initialize(error)) << error;
    half_fade.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(half_fade.observe(430u, {owner}, false));
    ASSERT_TRUE(half_fade.observe(431u, {owner}, false));
    ASSERT_EQ(half_fade.state()[24u], 1u);
    ASSERT_TRUE(half_fade.observe(432u, {overlap_boundary}, false));
    EXPECT_EQ(half_fade.state()[2u] & flag_provisional_current, 0u);
    EXPECT_TRUE(half_fade.output_is_exact_base());

    slr13_warp_fixture_t cut;
    ASSERT_TRUE(cut.initialize(error)) << error;
    cut.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(cut.observe(440u, {owner}, false));
    ASSERT_TRUE(cut.observe(441u, {owner}, false));
    ASSERT_TRUE(cut.observe(442u, {owner}, false));
    cut.set_cut(1u, true);
    ASSERT_TRUE(cut.observe(443u, {overlap_boundary}, false));
    EXPECT_EQ(cut.state()[2u] & flag_provisional_current, 0u);
    EXPECT_EQ(cut.state()[20u], 0u);
    EXPECT_TRUE(cut.output_is_exact_base());

    slr13_warp_fixture_t large_residual;
    ASSERT_TRUE(large_residual.initialize(error)) << error;
    large_residual.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(large_residual.observe(470u, {owner}, false));
    ASSERT_TRUE(large_residual.observe(471u, {owner}, false));
    ASSERT_TRUE(large_residual.observe(472u, {owner}, false));
    large_residual.set_background_sample_rows(
      overlap_boundary.core, target_for_pixels(12.0f), target_for_pixels(12.0f)
    );
    ASSERT_TRUE(large_residual.observe(473u, {overlap_boundary}, false));
    EXPECT_NE(large_residual.state()[2u] & flag_provisional_current, 0u);
    EXPECT_EQ(large_residual.state()[24u], 2u);
    EXPECT_EQ(
      large_residual.state()[v2::subtitle_locator_provisional_fade_word], 1u
    );
    EXPECT_EQ(large_residual.condition_params()[4u], 1u);
    EXPECT_FALSE(large_residual.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, ProvisionalDuplicateAndCorruptStateCannotBorrowAuthority) {
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    const ocr_box_t owner {
      {203u, 369u, 566u, 384u}, {198u, 365u, 571u, 389u}, 0u, 1u, 0u
    };
    const ocr_box_t replacement {
      {342u, 368u, 415u, 386u}, {337u, 364u, 420u, 390u}, 0u, 1u, 0u
    };
    const ocr_box_t changed_cover {
      replacement.core, {336u, 363u, 421u, 391u}, 0u, 1u, 0u
    };
    std::string error;

    slr13_warp_fixture_t duplicate;
    ASSERT_TRUE(duplicate.initialize(error)) << error;
    duplicate.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(duplicate.observe(450u, {owner}, false));
    ASSERT_TRUE(duplicate.observe(451u, {owner}, false));
    ASSERT_TRUE(duplicate.observe(452u, {owner}, false));
    duplicate.set_background_sample_rows(
      replacement.core, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(duplicate.observe(453u, {replacement}, false));
    ASSERT_NE(duplicate.state()[2u] & flag_provisional_current, 0u);
    ASSERT_TRUE(duplicate.observe(453u, {changed_cover}, false));
    EXPECT_EQ(duplicate.state()[2u] & flag_provisional_current, 0u);
    EXPECT_EQ(duplicate.state()[20u], 0u);
    EXPECT_TRUE(duplicate.output_is_exact_base());
    ASSERT_TRUE(duplicate.observe(454u, {owner}, false));
    EXPECT_EQ(duplicate.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(duplicate.state()[12u], 0u);
    EXPECT_EQ(duplicate.state()[20u], 1u);

    slr13_warp_fixture_t corrupt;
    ASSERT_TRUE(corrupt.initialize(error)) << error;
    corrupt.set_base(target_for_pixels(2.0f));
    ASSERT_TRUE(corrupt.observe(460u, {owner}, false));
    ASSERT_TRUE(corrupt.observe(461u, {owner}, false));
    ASSERT_TRUE(corrupt.observe(462u, {owner}, false));
    corrupt.set_background_sample_rows(
      replacement.core, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(corrupt.observe(463u, {replacement}, false));
    ASSERT_NE(corrupt.state()[2u] & flag_provisional_current, 0u);
    const auto valid_provisional_target =
      corrupt.state()[v2::subtitle_locator_provisional_target_word];
    const auto valid_provisional_fade =
      corrupt.state()[v2::subtitle_locator_provisional_fade_word];
    const auto expect_condition_rejected = [&corrupt]() {
      ASSERT_TRUE(corrupt.condition_only());
      EXPECT_TRUE(std::all_of(
        corrupt.condition_params().begin(), corrupt.condition_params().end(),
        [](const auto word) { return word == 0u; }
      ));
      EXPECT_TRUE(corrupt.output_is_exact_base());
    };
    ASSERT_TRUE(corrupt.overwrite_state_word(
      v2::subtitle_locator_provisional_target_word,
      std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN())
    ));
    expect_condition_rejected();
    ASSERT_TRUE(corrupt.overwrite_state_word(
      v2::subtitle_locator_provisional_target_word, valid_provisional_target
    ));
    ASSERT_TRUE(corrupt.overwrite_state_word(
      v2::subtitle_locator_provisional_fade_word, 0u
    ));
    expect_condition_rejected();
    ASSERT_TRUE(corrupt.overwrite_state_word(
      v2::subtitle_locator_provisional_fade_word, valid_provisional_fade
    ));
    ASSERT_TRUE(corrupt.overwrite_state_word(
      v2::subtitle_locator_current_offset, replacement.cover.left - 1u
    ));
    expect_condition_rejected();
    ASSERT_TRUE(corrupt.overwrite_state_word(
      v2::subtitle_locator_current_offset, replacement.cover.left
    ));
    ASSERT_TRUE(corrupt.condition_only());
    ASSERT_FALSE(corrupt.output_is_exact_base());

    const line_box_t unrelated_owner {100u, 330u, 500u, 340u};
    ASSERT_TRUE(corrupt.overwrite_state_word(5u, unrelated_owner.left));
    ASSERT_TRUE(corrupt.overwrite_state_word(6u, unrelated_owner.top));
    ASSERT_TRUE(corrupt.overwrite_state_word(7u, unrelated_owner.right));
    ASSERT_TRUE(corrupt.overwrite_state_word(8u, unrelated_owner.bottom));
    ASSERT_TRUE(corrupt.overwrite_state_word(
      9u,
      (unrelated_owner.right - unrelated_owner.left) *
        (unrelated_owner.bottom - unrelated_owner.top)
    ));
    ASSERT_TRUE(corrupt.overwrite_state_word(32u, unrelated_owner.left));
    ASSERT_TRUE(corrupt.overwrite_state_word(33u, unrelated_owner.top));
    ASSERT_TRUE(corrupt.overwrite_state_word(34u, unrelated_owner.right));
    ASSERT_TRUE(corrupt.overwrite_state_word(35u, unrelated_owner.bottom));
    expect_condition_rejected();
  }

  TEST(HostSbsSubtitleSlr13GpuTest, HandoffFadePolicyDoesNotChangeBirthGraceOrCutFade) {
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    const line_box_t first {180u, 360u, 590u, 370u};
    const line_box_t reborn {80u, 360u, 380u, 370u};
    slr13_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;
    fixture.set_base(target_for_pixels(2.0f));

    // A fresh owner still starts at half strength.
    ASSERT_TRUE(fixture.observe(340u, {first}, false));
    ASSERT_TRUE(fixture.observe(341u, {first}, false));
    EXPECT_EQ(fixture.state()[21u], 1u);
    EXPECT_EQ(fixture.state()[24u], 1u);
    ASSERT_TRUE(fixture.observe(342u, {first}, false));
    ASSERT_EQ(fixture.state()[24u], 2u);

    // Death grace can seed a replacement target, but its confirmation remains a birth rather
    // than a same-scene owner handoff and therefore restarts at half strength.
    ASSERT_TRUE(fixture.observe(343u, {}, false));
    ASSERT_EQ(fixture.state()[25u], 6u);
    fixture.set_background_sample_rows(
      reborn, target_for_pixels(6.0f), target_for_pixels(6.0f)
    );
    ASSERT_TRUE(fixture.observe(344u, {reborn}, false));
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_TRUE(fixture.output_is_exact_base());
    ASSERT_TRUE(fixture.observe(345u, {reborn}, false));
    EXPECT_EQ(fixture.state()[21u], 1u);
    EXPECT_EQ(fixture.state()[24u], 1u);

    // A scene-epoch change discards inheritance and always restarts the surviving geometry from
    // same-frame evidence at half strength.
    fixture.set_background_sample_rows(
      reborn, target_for_pixels(7.0f), target_for_pixels(7.0f)
    );
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(346u, {reborn}, false));
    EXPECT_EQ(fixture.state()[24u], 1u);
    EXPECT_EQ(
      fixture.state()[18u],
      std::bit_cast<std::uint32_t>(target_for_pixels(7.0f))
    );
  }

  TEST(HostSbsSubtitleSlr13GpuTest, UnreliablePlaneHoldsTwiceThenFailsBaseAndRecovers) {
    slr13_warp_fixture_t fixture;
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

    // A non-finite/out-of-container sample pair invalidates every primary and fallback row.
    // Preserve the last reliable plane for two distinct observations so one failed estimate
    // cannot expose warped glyph edges immediately.
    fixture.set_target_probe_ring_alternating(
      line,
      std::numeric_limits<float>::quiet_NaN(),
      v2::direct_container_limit + 0.001f
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

  TEST(HostSbsSubtitleSlr13GpuTest, RowSelectionUsesStableEvidenceAndNearerPlane) {
    std::string error;
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };

    // Two coherent nearby rows share their medians.
    slr13_warp_fixture_t close;
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
    slr13_warp_fixture_t split;
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
    slr13_warp_fixture_t dump_like;
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
    slr13_warp_fixture_t close_mixed;
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
    slr13_warp_fixture_t one_valid;
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
    slr13_warp_fixture_t exact_boundary;
    ASSERT_TRUE(exact_boundary.initialize(error)) << error;
    exact_boundary.set_background_sample_row_values(line, boundary, invalid);
    ASSERT_TRUE(exact_boundary.observe(228u, {line}, false));
    ASSERT_TRUE(exact_boundary.observe(229u, {line}, false));
    EXPECT_EQ(
      exact_boundary.state()[18u], std::bit_cast<std::uint32_t>(target_for_pixels(4.0f))
    );

    // With neither row valid a fresh owner has no plane to hold and must copy exact Base.
    slr13_warp_fixture_t neither;
    ASSERT_TRUE(neither.initialize(error)) << error;
    neither.set_target_probe_ring_alternating(
      line,
      std::numeric_limits<float>::quiet_NaN(),
      v2::direct_container_limit + 0.001f
    );
    ASSERT_TRUE(neither.observe(230u, {line}, false));
    ASSERT_TRUE(neither.observe(231u, {line}, false));
    EXPECT_EQ(neither.state()[2u], flag_owner | flag_target_reset);
    EXPECT_EQ(neither.state()[25u], 0u);
    EXPECT_TRUE(neither.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, DumpCompleteDispersedPrimaryRowsRemainCurrentAndConditioned) {
    const auto decode_row = [](const std::array<std::uint32_t, 16u> &bits) {
      std::array<float, 16u> row {};
      std::transform(
        bits.begin(), bits.end(), row.begin(),
        [](const std::uint32_t value) { return std::bit_cast<float>(value); }
      );
      return row;
    };
    const auto verify = [&decode_row](
      const line_box_t line,
      const std::array<std::uint32_t, 16u> &outer_bits,
      const std::array<std::uint32_t, 16u> &inner_bits,
      const std::uint32_t expected_target_bits,
      const std::uint64_t first_identity
    ) {
      slr13_warp_fixture_t fixture(field_width, field_height, 3840u, 2160u);
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      fixture.set_background_sample_row_values(
        line, decode_row(outer_bits), decode_row(inner_bits)
      );

      // The first exact observation is pending and therefore still publishes immutable Base.
      ASSERT_TRUE(fixture.observe(first_identity, {line}, false));
      EXPECT_EQ(fixture.state()[2u], flag_pending);
      EXPECT_TRUE(fixture.output_is_exact_base());

      // A distinct confirmation must accept the two complete primary rows despite both IQRs
      // exceeding eight binocular source pixels. It must not enter the unreliable-target hold or
      // TARGET_RESET path seen in the captured fullscreen transition.
      ASSERT_TRUE(fixture.observe(first_identity + 1u, {line}, false));
      EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
      EXPECT_EQ(fixture.state()[18u], expected_target_bits);
      EXPECT_EQ(fixture.state()[20u], 1u);
      EXPECT_EQ(fixture.state()[24u], 1u);
      EXPECT_EQ(fixture.state()[25u], 0u);
      EXPECT_FALSE(fixture.output_is_exact_base());

      // Continuing exact-frame OCR keeps the same target/current cover, reaches full fade, and
      // remains conditioned without consuming a hold observation.
      ASSERT_TRUE(fixture.observe(first_identity + 2u, {line}, false));
      EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
      EXPECT_EQ(fixture.state()[18u], expected_target_bits);
      EXPECT_EQ(fixture.state()[20u], 1u);
      EXPECT_EQ(fixture.state()[24u], 2u);
      EXPECT_EQ(fixture.state()[25u], 0u);
      EXPECT_FALSE(fixture.output_is_exact_base());

      // The same geometry after a durable scene-epoch change must reacquire from these robust
      // rows immediately. It cannot inherit the old scene target or misclassify the observation
      // as an unreliable hold merely because both row IQRs are dispersed.
      fixture.set_cut(1u, true);
      ASSERT_TRUE(fixture.observe(first_identity + 3u, {line}, false));
      EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
      EXPECT_EQ(fixture.state()[18u], expected_target_bits);
      EXPECT_EQ(fixture.state()[20u], 1u);
      EXPECT_EQ(fixture.state()[24u], 1u);
      EXPECT_EQ(fixture.state()[25u], 0u);
      EXPECT_FALSE(fixture.output_is_exact_base());
    };

    // Dump _2: row IQRs are 16.359/11.373 px and the medians differ by 0.481 px, so the
    // authoritative target is their mean (0x3b896b40).
    verify(
      {306u, 369u, 463u, 385u},
      {
        0x3b2d1cb4u, 0x3b3de960u, 0x3b4a3b2cu, 0x3b544f88u,
        0x3b5ec36cu, 0x3b78e500u, 0x3b81ac60u, 0x3b84a8deu,
        0x3b8c200eu, 0x3b8fdba0u, 0x3ba3a3ceu, 0x3bb0b3aeu,
        0x3bb46f00u, 0x3bb64ca6u, 0x3bb7cac2u, 0x3bba6770u,
      },
      {
        0x3b3fc738u, 0x3b4d974cu, 0x3b54af18u, 0x3b5bc6e0u,
        0x3b657ba0u, 0x3b7b2264u, 0x3b85086eu, 0x3b88c408u,
        0x3b8c200eu, 0x3b8e5d66u, 0x3b909abcu, 0x3b9811d0u,
        0x3ba99c7au, 0x3bac98c8u, 0x3baf9514u, 0x3bb2f0e0u,
      },
      0x3b896b40u,
      2350u
    );

    // Dump _3: row IQRs are 58.372/35.314 px and the medians differ by 9.026 px, so the
    // authoritative target is the numerically larger outer-row median (0x3b8c1878).
    verify(
      {324u, 367u, 444u, 385u},
      {
        0xbb2dd154u, 0xbb2dd154u, 0xbb2f4618u, 0xbb3060acu,
        0xbb2fc8c4u, 0xba9979e0u, 0x3abaf860u, 0x3b83daa8u,
        0x3b945648u, 0x3b99ef90u, 0x3b9d4b82u, 0x3ba047e4u,
        0x3ba40358u, 0x3ba5e110u, 0x3ba87ddau, 0x3baa5b8eu,
      },
      {
        0xbb284b7cu, 0xba9ff8f0u, 0x3ab47950u, 0x3b0e2040u,
        0x399beb60u, 0xbb16bbb4u, 0xb7fb1400u, 0x3b2842f8u,
        0x3b6e11a0u, 0x3b8567feu, 0x3b95d47eu, 0x3b993076u,
        0x3b9cebf6u, 0x3ba106fcu, 0x3ba462e4u, 0x3ba87ddau,
      },
      0x3b8c1878u,
      2360u
    );
  }

  TEST(HostSbsSubtitleSlr13GpuTest, PrimaryFailureUsesStrictNearestOrdinaryProbeBesideRibbon) {
    slr13_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t subtitle_core {40u, 360u, 728u, 370u};
    const line_box_t subtitle_cover {36u, 356u, 732u, 374u};
    const line_box_t ribbon_core {1u, 401u, 689u, roi_bottom};
    const line_box_t ribbon_cover {0u, 397u, field_width, field_height};
    const ocr_box_t subtitle {subtitle_core, subtitle_cover, 0u, 1u, 0u};
    const ocr_box_t ribbon {ribbon_core, ribbon_cover, box_flag_ribbon, 7u, 4u};
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };

    // The unchanged aggregate primary center is the mean of the ordinary and ribbon member
    // centers: (383.5 + 344.5) / 2 = 364. Its rows and every fallback ring start invalid.
    // W/16 is 43 cells, so the five 4-cell sample lattices are disjoint.
    constexpr float primary_center = 364.0f;
    fixture.set_base(target_for_pixels(2.0f));
    fixture.set_target_probe_ring_alternating(
      primary_center,
      subtitle_core.top,
      subtitle_core.right - subtitle_core.left,
      std::numeric_limits<float>::quiet_NaN(),
      v2::direct_container_limit + 0.001f
    );
    constexpr float first_fallback_center = primary_center - 688.0f / 16.0f;
    fixture.set_background_probe_rows(
      first_fallback_center,
      subtitle_core.top,
      target_for_pixels(24.0f),
      target_for_pixels(24.5f)
    );

    ASSERT_TRUE(fixture.observe(2320u, {subtitle, ribbon}, false));
    ASSERT_TRUE(fixture.observe(2321u, {subtitle, ribbon}, false));
    ASSERT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
    ASSERT_EQ(fixture.state()[4u], 2u);
    EXPECT_EQ(
      fixture.state()[31u],
      (2u << owner_kind_shift) | (2u << current_kind_shift)
    );
    EXPECT_NEAR(
      std::bit_cast<float>(fixture.state()[18u]),
      0.5f * (target_for_pixels(24.0f) + target_for_pixels(24.5f)),
      1.0e-8f
    );
    EXPECT_FALSE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, FallbackConflictAndClampedStripsFailExactBase) {
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    std::string error;

    // Two strict radius-one probes on planes more than four pixels apart are an explicit
    // conflict. Radius two is coherent, but must not hide that nearer ambiguity.
    slr13_warp_fixture_t conflict;
    ASSERT_TRUE(conflict.initialize(error)) << error;
    const line_box_t line {40u, 360u, 728u, 370u};
    constexpr float line_center = 383.5f;
    constexpr float line_step = 688.0f / 16.0f;
    conflict.set_base(target_for_pixels(2.0f));
    conflict.set_target_probe_ring_alternating(
      line,
      std::numeric_limits<float>::quiet_NaN(),
      v2::direct_container_limit + 0.001f
    );
    conflict.set_background_probe_rows(
      line_center - line_step, line.top,
      target_for_pixels(10.0f), target_for_pixels(10.5f)
    );
    conflict.set_background_probe_rows(
      line_center + line_step, line.top,
      target_for_pixels(20.0f), target_for_pixels(20.5f)
    );
    conflict.set_background_probe_rows(
      line_center - 2.0f * line_step, line.top,
      target_for_pixels(30.0f), target_for_pixels(30.5f)
    );
    conflict.set_background_probe_rows(
      line_center + 2.0f * line_step, line.top,
      target_for_pixels(31.0f), target_for_pixels(31.5f)
    );
    ASSERT_TRUE(conflict.observe(2330u, {line}, false));
    ASSERT_TRUE(conflict.observe(2331u, {line}, false));
    EXPECT_EQ(conflict.state()[2u], flag_owner | flag_target_reset);
    EXPECT_TRUE(conflict.output_is_exact_base());

    // Three members retain an aggregate center at x=31.5 while the earlier ribbon top keeps the
    // primary rows independent from the ordinary fallback rows. The negative shifted rows
    // look perfectly coherent only because their raw strip falls left of content and repeats x=0.
    // The positive probes contain an invalid x=62 sample. Rejecting both clamped negative strips
    // is therefore required for exact-Base output.
    slr13_warp_fixture_t edge;
    ASSERT_TRUE(edge.initialize(error)) << error;
    const line_box_t upper {0u, 380u, 64u, 386u};
    const line_box_t lower {0u, 389u, 64u, 395u};
    const line_box_t edge_ribbon_core {1u, 350u, 689u, roi_bottom};
    const line_box_t edge_ribbon_cover {0u, 346u, field_width, field_height};
    const ocr_box_t edge_ribbon {
      edge_ribbon_core, edge_ribbon_cover, box_flag_ribbon, 7u, 4u
    };
    edge.set_base(target_for_pixels(2.0f));
    edge.set_background_probe_alternating(
      31.5f, edge_ribbon_core.top,
      std::numeric_limits<float>::quiet_NaN(),
      v2::direct_container_limit + 0.001f
    );
    edge.set_background_probe_rows(
      27.5f, upper.top, target_for_pixels(50.0f), target_for_pixels(50.0f)
    );
    const auto outer_y = upper.top - 10u;
    const auto inner_y = upper.top - 4u;
    edge.set_base_at(62u, outer_y, v2::direct_container_limit + 0.001f);
    edge.set_base_at(62u, inner_y, v2::direct_container_limit + 0.001f);
    ASSERT_TRUE(edge.observe(2340u, {upper, lower, edge_ribbon}, false));
    ASSERT_TRUE(edge.observe(2341u, {upper, lower, edge_ribbon}, false));
    EXPECT_EQ(edge.state()[2u], flag_owner | flag_target_reset);
    EXPECT_TRUE(edge.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, LargeResidualAndHardCutRestartButCutNeverHolds) {
    const line_box_t line {180u, 360u, 590u, 370u};
    constexpr auto target_for_pixels = [](const float binocular_source_pixels) {
      return binocular_source_pixels / (2.0f * static_cast<float>(1920u));
    };
    std::string error;

    slr13_warp_fixture_t tracking;
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
    slr13_warp_fixture_t cut;
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
    cut.set_target_probe_ring_alternating(
      line,
      std::numeric_limits<float>::quiet_NaN(),
      v2::direct_container_limit + 0.001f
    );
    cut.set_cut(2u, true);
    ASSERT_TRUE(cut.observe(243u, {line}, false));
    EXPECT_EQ(cut.state()[2u], flag_owner | flag_target_reset);
    EXPECT_EQ(cut.state()[25u], 0u);
    EXPECT_TRUE(cut.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, DuplicateCutPulseIsBitExactIdempotent) {
    slr13_warp_fixture_t fixture;
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
  }

  TEST(HostSbsSubtitleSlr13GpuTest, FrozenCutBridgePulseDoesNotRetriggerDistinctReuseObservation) {
    slr13_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error)) << error;

    const line_box_t line {180u, 360u, 590u, 370u};
    fixture.set_base(0.01f);
    ASSERT_TRUE(fixture.observe(250u, {line}, false));
    ASSERT_TRUE(fixture.observe(251u, {line}, false));
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(252u, {line}, false));
    ASSERT_EQ(fixture.state()[24u], 1u);
    const auto generation_after_cut = fixture.state()[3u];

    // The next infer-authorized observation can still arrive while CutBridge exposes the prior
    // delivery's pulse. A distinct OCR identity in the same authenticated epoch is an ordinary
    // continuation: the owner generation stays fixed and fade may mature normally.
    fixture.set_cut(1u, true);
    ASSERT_TRUE(fixture.observe(253u, {line}, false));
    EXPECT_EQ(fixture.state()[3u], generation_after_cut);
    EXPECT_EQ(fixture.state()[24u], 2u);
    EXPECT_EQ(fixture.state()[26u], 1u);

    // The conditioner independently binds the compact state to the authenticated current scene
    // epoch. A foreign but otherwise canonical state must copy exact Base.
    ASSERT_TRUE(fixture.overwrite_state_word(26u, 2u));
    ASSERT_TRUE(fixture.condition_only());
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, MissedCutPulseEpochMismatchRestartsLocalPlane) {
    slr13_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, InvalidOcrClearsCurrentButPreservesAndAgesTargetGrace) {
    slr13_warp_fixture_t fixture;
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
    fixture.poison_condition_params(0xd1ced1ceu);
    ASSERT_TRUE(fixture.observe(303u, {disjoint}, false, true, false, true));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[21u], 2u);
    EXPECT_EQ(fixture.state()[25u], 6u);
    EXPECT_EQ(fixture.state()[18u], tracked_target);
    EXPECT_TRUE(std::all_of(
      fixture.condition_params().begin(), fixture.condition_params().end(),
      [](const auto word) { return word == 0u; }
    ));
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
    fixture.poison_condition_params(0xc01dc01du);
    ASSERT_TRUE(fixture.observe(307u, {}, false, false));
    EXPECT_EQ(fixture.state()[2u], 0u);
    EXPECT_EQ(fixture.state()[4u], 0u);
    EXPECT_EQ(fixture.state()[12u], 0u);
    EXPECT_EQ(fixture.state()[20u], 0u);
    EXPECT_EQ(fixture.state()[25u], 0u);
    EXPECT_EQ(fixture.state()[18u], 0u);
    EXPECT_TRUE(std::all_of(
      fixture.condition_params().begin(), fixture.condition_params().end(),
      [](const auto word) { return word == 0u; }
    ));
    EXPECT_TRUE(fixture.output_is_exact_base());
  }

  TEST(HostSbsSubtitleSlr13GpuTest, OversizedDeathGraceCannotExtendTargetLifetime) {
    slr13_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, TracksSubtitleAndBottomRibbonOnOnePlane) {
    slr13_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, ResetLogoAndInvalidRecordFailToExactBase) {
    slr13_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, RejectsOnlyStrictBottomCornerOrdinaryCores) {
    constexpr auto edge_threshold =
      field_width / v2::subtitle_locator_corner_edge_divisor;
    constexpr auto bottom_threshold =
      roi_bottom - v2::subtitle_locator_corner_bottom_rows;
    static_assert(edge_threshold == 24u);

    const auto observe_twice = [](slr13_warp_fixture_t &fixture, const ocr_box_t &box) {
      EXPECT_TRUE(fixture.observe(500u, {box}, false));
      EXPECT_TRUE(fixture.observe(501u, {box}, false));
    };

    // Strictly inside the symmetric edge threshold and exactly on the bottom threshold rejects.
    slr13_warp_fixture_t left_corner;
    std::string error;
    ASSERT_TRUE(left_corner.initialize(error)) << error;
    const line_box_t left_rejected {
      edge_threshold - 1u, bottom_threshold - 10u,
      edge_threshold + 115u, bottom_threshold,
    };
    observe_twice(left_corner, left_rejected);
    EXPECT_EQ(left_corner.state()[2u], 0u);
    EXPECT_TRUE(left_corner.output_is_exact_base());

    // Equality at the edge threshold is intentionally eligible.
    slr13_warp_fixture_t edge_equal;
    error.clear();
    ASSERT_TRUE(edge_equal.initialize(error)) << error;
    const line_box_t equality_passes {
      edge_threshold, bottom_threshold - 10u,
      edge_threshold + 116u, bottom_threshold,
    };
    observe_twice(edge_equal, equality_passes);
    EXPECT_EQ(edge_equal.state()[2u], flag_owner | flag_target_valid);

    // The same strict rejection applies at the right corner.
    slr13_warp_fixture_t right_corner;
    error.clear();
    ASSERT_TRUE(right_corner.initialize(error)) << error;
    const line_box_t right_rejected {
      field_width - edge_threshold - 115u,
      bottom_threshold - 10u,
      field_width - (edge_threshold - 1u),
      bottom_threshold,
    };
    observe_twice(right_corner, right_rejected);
    EXPECT_EQ(right_corner.state()[2u], 0u);
    EXPECT_TRUE(right_corner.output_is_exact_base());

    // One row above the bottom proximity band remains ordinary eligible.
    slr13_warp_fixture_t above_band;
    error.clear();
    ASSERT_TRUE(above_band.initialize(error)) << error;
    const line_box_t above_passes {
      edge_threshold - 1u, bottom_threshold - 11u,
      edge_threshold + 115u, bottom_threshold - 1u,
    };
    observe_twice(above_band, above_passes);
    EXPECT_EQ(above_band.state()[2u], flag_owner | flag_target_valid);

    // Detector-authenticated ribbons are exempt even when their core touches a bottom corner.
    slr13_warp_fixture_t ribbon_fixture;
    error.clear();
    ASSERT_TRUE(ribbon_fixture.initialize(error)) << error;
    const ocr_box_t ribbon {
      {0u, roi_bottom - 29u, field_width - 77u, roi_bottom},
      {0u, roi_bottom - 33u, field_width, field_height},
      box_flag_ribbon,
      7u,
      4u,
    };
    observe_twice(ribbon_fixture, ribbon);
    EXPECT_EQ(ribbon_fixture.state()[2u], flag_owner | flag_target_valid);
    EXPECT_EQ(ribbon_fixture.state()[31u],
              (1u << owner_kind_shift) | (1u << current_kind_shift));
  }

  TEST(HostSbsSubtitleSlr13GpuTest, MixedSelectionOverCompactCapacityFailsFlat) {
    slr13_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, HardCutSurvivorRestartsLocalPlaneButDisjointStackIsPending) {
    slr13_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, PartialHardCutSurvivorHandsOffExpandedStackOnSecondObservation) {
    slr13_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, DomainResetRequiresPendingThenNextDistinctFrameBirth) {
    slr13_warp_fixture_t fixture;
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

  TEST(HostSbsSubtitleSlr13GpuTest, SupportsAuthenticatedWideAndPortraitFields) {
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
      field_case_t {1540u, 868u, 1920u, 1080u, 200u, 1300u},
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
      const auto cell_scale = v2::subtitle_locator_field_cell_scale(
        field_case.field_width,
        field_case.field_height
      );
      ASSERT_NE(cell_scale, 0u);
      const line_box_t line {
        field_case.line_left,
        dynamic_roi.top + 35u * cell_scale,
        field_case.line_right,
        dynamic_roi.top + 45u * cell_scale,
      };
      ASSERT_LE(line.bottom, dynamic_roi.bottom);

      slr13_warp_fixture_t fixture(
        field_case.field_width,
        field_case.field_height,
        field_case.source_width,
        field_case.source_height
      );
      std::string error;
      ASSERT_TRUE(fixture.initialize(error)) << error;
      fixture.set_base(0.03f);
      fixture.set_background_sample_rows(line, 0.01f, 0.01f);
      ASSERT_TRUE(fixture.observe(identity++, {line}, false));
      EXPECT_EQ(fixture.state()[2u], flag_pending);
      EXPECT_TRUE(fixture.output_is_exact_base());
      ASSERT_TRUE(fixture.observe(identity++, {line}, false));
      EXPECT_EQ(fixture.state()[2u], flag_owner | flag_target_valid);
      EXPECT_EQ(fixture.state()[4u], 1u);
      EXPECT_EQ(fixture.state()[20u], 1u);
      EXPECT_EQ(fixture.state()[27u], field_case.field_width);
      EXPECT_EQ(fixture.state()[28u], field_case.field_height);
      EXPECT_EQ(
        fixture.state()[18u],
        std::bit_cast<std::uint32_t>(0.01f)
      );
      EXPECT_LT(
        fixture.output_at(
          (line.left + line.right) / 2u,
          line.top + 4u * cell_scale
        ),
        0.025f
      );
    }
  }

  TEST(HostSbsSubtitleSlr13GpuTest, ScalesFixedLineThresholdsOnConvex2xFields) {
    constexpr std::uint32_t live_width = 1540u;
    constexpr std::uint32_t live_height = 868u;
    constexpr std::uint32_t source_width = 1920u;
    constexpr std::uint32_t source_height = 1080u;
    constexpr auto cell_scale =
      v2::subtitle_locator_field_cell_scale(live_width, live_height);
    static_assert(cell_scale == 2u);
    constexpr auto dynamic_roi = v2::subtitle_ocr_dynamic_roi(
      source_width, source_height, live_width, live_height
    );
    static_assert(static_cast<bool>(dynamic_roi));

    const line_box_t below_threshold {
      600u,
      dynamic_roi.top + 50u,
      600u + v2::subtitle_locator_min_width_cells * cell_scale - 1u,
      dynamic_roi.top + 50u +
        v2::subtitle_locator_min_height_cells * cell_scale - 1u,
    };
    slr13_warp_fixture_t rejected(
      live_width, live_height, source_width, source_height
    );
    std::string error;
    ASSERT_TRUE(rejected.initialize(error)) << error;
    ASSERT_TRUE(rejected.observe(600u, {below_threshold}, false));
    ASSERT_TRUE(rejected.observe(601u, {below_threshold}, false));
    EXPECT_EQ(rejected.state()[2u], 0u);
    EXPECT_TRUE(rejected.output_is_exact_base());

    const line_box_t exact_threshold {
      600u,
      dynamic_roi.top + 50u,
      600u + v2::subtitle_locator_min_width_cells * cell_scale,
      dynamic_roi.top + 50u +
        v2::subtitle_locator_min_height_cells * cell_scale,
    };
    slr13_warp_fixture_t accepted(
      live_width, live_height, source_width, source_height
    );
    error.clear();
    ASSERT_TRUE(accepted.initialize(error)) << error;
    ASSERT_TRUE(accepted.observe(602u, {exact_threshold}, false));
    EXPECT_EQ(accepted.state()[2u], flag_pending);
    ASSERT_TRUE(accepted.observe(603u, {exact_threshold}, false));
    EXPECT_EQ(accepted.state()[2u], flag_owner | flag_target_valid);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, ScalesBottomCornerBandOnConvex2xFields) {
    constexpr std::uint32_t live_width = 1540u;
    constexpr std::uint32_t live_height = 868u;
    constexpr std::uint32_t source_width = 1920u;
    constexpr std::uint32_t source_height = 1080u;
    constexpr auto cell_scale =
      v2::subtitle_locator_field_cell_scale(live_width, live_height);
    static_assert(cell_scale == 2u);
    constexpr auto dynamic_roi = v2::subtitle_ocr_dynamic_roi(
      source_width, source_height, live_width, live_height
    );
    static_assert(static_cast<bool>(dynamic_roi));
    constexpr auto edge_threshold =
      live_width / v2::subtitle_locator_corner_edge_divisor;
    constexpr auto bottom_threshold =
      dynamic_roi.bottom - v2::subtitle_locator_corner_bottom_rows * cell_scale;
    static_assert(edge_threshold == 48u);

    const auto observe_twice = [](slr13_warp_fixture_t &fixture, const ocr_box_t &box) {
      EXPECT_TRUE(fixture.observe(610u, {box}, false));
      EXPECT_TRUE(fixture.observe(611u, {box}, false));
    };

    slr13_warp_fixture_t rejected(
      live_width, live_height, source_width, source_height
    );
    std::string error;
    ASSERT_TRUE(rejected.initialize(error)) << error;
    const line_box_t inside_band {
      edge_threshold - 1u,
      bottom_threshold - 20u,
      edge_threshold + 229u,
      bottom_threshold,
    };
    observe_twice(rejected, inside_band);
    EXPECT_EQ(rejected.state()[2u], 0u);
    EXPECT_TRUE(rejected.output_is_exact_base());

    slr13_warp_fixture_t accepted(
      live_width, live_height, source_width, source_height
    );
    error.clear();
    ASSERT_TRUE(accepted.initialize(error)) << error;
    const line_box_t above_band {
      edge_threshold - 1u,
      bottom_threshold - 21u,
      edge_threshold + 229u,
      bottom_threshold - 1u,
    };
    observe_twice(accepted, above_band);
    EXPECT_EQ(accepted.state()[2u], flag_owner | flag_target_valid);
  }

  TEST(HostSbsSubtitleSlr13GpuTest, AuthenticatesPaddedContentAndExtendsItsBoundary) {
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

    slr13_warp_fixture_t fixture(
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

  TEST(HostSbsSubtitleSlr13GpuTest, RibbonBottomToleranceProjectsExactlyAcrossFields) {
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
      field_case_t {1540u, 868u, 1920u, 1080u},
      field_case_t {2044u, 868u, 2560u, 1080u},
      field_case_t {2072u, 868u, 3440u, 1440u},
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
      const auto cell_scale = v2::subtitle_locator_field_cell_scale(
        field_case.field_width,
        field_case.field_height
      );
      ASSERT_NE(cell_scale, 0u);
      ASSERT_GE(minimum_bottom, dynamic_roi.top + 7u * cell_scale);
      ASSERT_LE(minimum_bottom, dynamic_roi.bottom);
      const auto left = field_case.field_width / 20u;
      const auto right = field_case.field_width - left;
      const auto core_top = minimum_bottom - 6u * cell_scale;
      const auto cover_pad = 4u * cell_scale;
      const auto cover_top = core_top > cover_pad ? core_top - cover_pad : 0u;
      const ocr_box_t boundary_ribbon {
        {left, core_top, right, minimum_bottom},
        {0u, cover_top, field_case.field_width, field_case.field_height},
        box_flag_ribbon,
        7u,
        4u,
      };

      slr13_warp_fixture_t accepted(
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
    slr13_warp_fixture_t rejected(
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

  TEST(HostSbsSubtitleSlr13GpuTest, DynamicMaximumWidthAndMalformedRoiFailFlat) {
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
    slr13_warp_fixture_t overwide_fixture(
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

    // Even when OCR8 repeats the same malformed bounds, SLR13 independently derives the safe ROI
    // from source/field geometry and rejects the record instead of trusting its header.
    const auto wrong_top = dynamic_roi.top + 1u;
    slr13_warp_fixture_t malformed_fixture(
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
    EXPECT_TRUE(malformed_fixture.output_is_exact_base());
  }
}  // namespace

#else

TEST(HostSbsSubtitleSlr13GpuTest, WindowsOnly) {
  GTEST_SKIP() << "D3D11 WARP is Windows-only";
}

#endif
