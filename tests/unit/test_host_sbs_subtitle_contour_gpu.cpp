/**
 * @file tests/unit/test_host_sbs_subtitle_contour_gpu.cpp
 * @brief Actual D3D11 consecutive-RGB mask entries and an independent projection-area oracle.
 *
 * These tests isolate contour classification from model detection: the probability/accepted
 * records are authored native inputs. They do not replicate a stereo renderer on the CPU.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/depth_coordinate_v2.h"
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
  #include <string>
  #include <utility>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace v2 = models::depth_coordinate_v2;
  using uint4_t = std::array<std::uint32_t, 4u>;
  using pixel_t = std::array<float, 4u>;
  using half_pixel_t = std::array<std::uint16_t, 4u>;

  struct contour_constants_t {
    uint4_t source, field, content, frame, observation, options;
  };

  static_assert(sizeof(contour_constants_t) == 96u);
  static_assert(sizeof(v2::constants_t) == 64u);
  static_assert(v2::state_float_count % 4u == 0u);

  // Native input is the production FP16 display-band format, not an easier FP32 substitute.
  std::uint16_t float_to_half(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16u) & 0x8000u);
    const auto exponent = (bits >> 23u) & 0xffu;
    auto mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
      return sign | (mantissa != 0u ? 0x7e00u : 0x7c00u);
    }
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
      return sign | 0x7c00u;
    }
    if (half_exponent <= 0) {
      if (half_exponent < -10) {
        return sign;
      }
      mantissa |= 0x800000u;
      const auto shift = static_cast<unsigned>(14 - half_exponent);
      const auto rounded = (mantissa + (1u << (shift - 1u)) - 1u + ((mantissa >> shift) & 1u)) >> shift;
      return sign | static_cast<std::uint16_t>(rounded);
    }
    const auto rounded = mantissa + 0xfffu + ((mantissa >> 13u) & 1u);
    return sign | static_cast<std::uint16_t>((static_cast<std::uint32_t>(half_exponent) << 10u) + (rounded >> 13u));
  }

  struct texture_t {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };

  struct buffer_t {
    ComPtr<ID3D11Buffer> buffer;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };

  bool create_texture(ID3D11Device *device, std::uint32_t width, std::uint32_t height, DXGI_FORMAT format, UINT binds, texture_t &out) {
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1u;
    desc.ArraySize = 1u;
    desc.Format = format;
    desc.SampleDesc.Count = 1u;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = binds;
    return SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &out.texture)) &&
           ((binds & D3D11_BIND_SHADER_RESOURCE) == 0u ||
            SUCCEEDED(device->CreateShaderResourceView(out.texture.Get(), nullptr, &out.srv))) &&
           ((binds & D3D11_BIND_UNORDERED_ACCESS) == 0u ||
            SUCCEEDED(device->CreateUnorderedAccessView(out.texture.Get(), nullptr, &out.uav)));
  }

  bool create_buffer(ID3D11Device *device, std::size_t bytes, UINT stride, UINT binds, buffer_t &out) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = static_cast<UINT>(bytes);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = binds;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = stride;
    return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &out.buffer)) &&
           ((binds & D3D11_BIND_SHADER_RESOURCE) == 0u ||
            SUCCEEDED(device->CreateShaderResourceView(out.buffer.Get(), nullptr, &out.srv))) &&
           ((binds & D3D11_BIND_UNORDERED_ACCESS) == 0u ||
            SUCCEEDED(device->CreateUnorderedAccessView(out.buffer.Get(), nullptr, &out.uav)));
  }

  template<class T>
  bool read_texture(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Texture2D *source, std::vector<T> &out) {
    D3D11_TEXTURE2D_DESC desc {};
    source->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0u;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0u;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) {
      return false;
    }
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return false;
    }
    out.resize(static_cast<std::size_t>(desc.Width) * desc.Height);
    for (UINT y = 0u; y < desc.Height; ++y) {
      std::memcpy(out.data() + static_cast<std::size_t>(y) * desc.Width, static_cast<const std::uint8_t *>(mapped.pData) + static_cast<std::size_t>(y) * mapped.RowPitch, static_cast<std::size_t>(desc.Width) * sizeof(T));
    }
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  bool read_buffer(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Buffer *source, std::vector<std::uint32_t> &out) {
    D3D11_BUFFER_DESC desc {};
    source->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0u;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0u;
    desc.StructureByteStride = 0u;
    ComPtr<ID3D11Buffer> staging;
    if (FAILED(device->CreateBuffer(&desc, nullptr, &staging))) {
      return false;
    }
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return false;
    }
    out.resize(desc.ByteWidth / sizeof(std::uint32_t));
    std::memcpy(out.data(), mapped.pData, desc.ByteWidth);
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  // Feature.w contains valid RGB, a 0..4 confirmation count, and a mask bit only at count4.
  constexpr std::uint32_t snapshot_valid_word = v2::subtitle_contour_history_snapshot_valid_word;
  static_assert(snapshot_valid_word == 23u);
  constexpr std::uint32_t reserved_word = v2::subtitle_contour_history_reserved_offset;
  static_assert(reserved_word == 24u && v2::subtitle_contour_history_reserved_word_count == 8u);
  static_assert(v2::subtitle_contour_history_feature_valid_bit == 2u);
  static_assert(v2::subtitle_contour_history_feature_mask_bit == 1u);
  constexpr float valid_rgb = float(v2::subtitle_contour_history_feature_valid_bit);
  constexpr std::uint32_t confirmation_count = v2::subtitle_contour_history_confirmation_observations;
  static_assert(confirmation_count == 4u);
  static_assert(v2::subtitle_contour_history_feature_count_shift == 2u);

  constexpr float feature_code(std::uint32_t count) {
    return float(v2::subtitle_contour_history_feature_valid_bit | (count << v2::subtitle_contour_history_feature_count_shift) | (count == confirmation_count ? v2::subtitle_contour_history_feature_mask_bit : 0u));
  }

  struct snapshot_t {
    std::vector<std::uint32_t> support, field, identity, probability_bits;
    std::vector<pixel_t> features, next_features;
    std::vector<half_pixel_t> source;
  };

  class HostSbsSubtitleContourGpuTest: public ::testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &device_, &actual, &context_)));
      ASSERT_EQ(actual, D3D_FEATURE_LEVEL_11_0);
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {
        cache::shader_spec {"host_sbs_subtitle_contour_cs.hlsl", "temporal_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_contour_cs.hlsl", "project_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_contour_cs.hlsl", "history_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_contour_cs.hlsl", "snapshot_main", "cs_5_0"}
      };
      const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, specs);
      for (std::size_t pass = 0u; pass < specs.size(); ++pass) {
        const auto bytecode = cache::get(sources, specs[pass]);
        ASSERT_TRUE(bytecode && !bytecode->empty()) << specs[pass].entrypoint;
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &shaders_[pass])));
      }
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(contour_constants_t);
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &contour_cb_)));
      desc.ByteWidth = sizeof(v2::constants_t);
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &geometry_cb_)));
      desc.ByteWidth = 3u * sizeof(std::uint32_t);
      desc.BindFlags = 0u;
      desc.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &indirect_)));
      ASSERT_TRUE(resize(160u, 90u));
    }

    bool resize(std::uint32_t width, std::uint32_t height) {
      unbind();
      source_w_ = width;
      source_h_ = height;
      crop_h_ = std::min(height, (width + 5u) / 6u);
      crop_top_ = height - crop_h_;
      field_w_ = v2::model_calibrated_shapes[0u].width;
      field_h_ = v2::model_calibrated_shapes[0u].height;
      content_ = {0u, 0u, field_w_, field_h_};
      const UINT both = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      display_ = {};
      support_ = {};
      field_ = {};
      features_ = {};
      next_features_ = {};
      completed_display_ = {};
      probability_ = {};
      plane_ = {};
      params_ = {};
      ocr_ = {};
      cut_ = {};
      geometry_ = {};
      identity_ = {};
      completed_probability_ = {};
      if (!create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_SHADER_RESOURCE, display_) || !create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R32_UINT, both, support_) || !create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R16G16B16A16_FLOAT, both, completed_display_) || !create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_UINT, both, field_) || !create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R32G32B32A32_FLOAT, both, features_) || !create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R32G32B32A32_FLOAT, both, next_features_) || !create_buffer(device_.Get(), v2::subtitle_ocr_output_width * v2::subtitle_ocr_output_height * 4u, 4u, D3D11_BIND_SHADER_RESOURCE, probability_) || !create_buffer(device_.Get(), v2::subtitle_ocr_output_width * v2::subtitle_ocr_output_height * 4u, 4u, both, completed_probability_) || !create_buffer(device_.Get(), v2::subtitle_plane_state_word_count * 4u, 4u, D3D11_BIND_SHADER_RESOURCE, plane_) || !create_buffer(device_.Get(), v2::subtitle_condition_param_word_count * 4u, 4u, D3D11_BIND_SHADER_RESOURCE, params_) || !create_buffer(device_.Get(), v2::subtitle_ocr_record_word_count * 4u, 4u, D3D11_BIND_SHADER_RESOURCE, ocr_) || !create_buffer(device_.Get(), sbs_adaptive_state::word_count * 4u, 16u, D3D11_BIND_SHADER_RESOURCE, cut_) || !create_buffer(device_.Get(), sizeof(v2::state_words_t), 16u, D3D11_BIND_SHADER_RESOURCE, geometry_) || !create_buffer(device_.Get(), v2::subtitle_contour_history_word_count * 4u, 4u, both, identity_)) {
        return false;
      }
      const UINT zero[4] {};
      const float zeros[4] {};
      context_->ClearUnorderedAccessViewUint(identity_.uav.Get(), zero);
      context_->ClearUnorderedAccessViewFloat(features_.uav.Get(), zeros);
      clear_scene();
      make_authority(1u, 1000001u);
      return true;
    }

    std::uint32_t body_y() const {
      return (v2::subtitle_ocr_safe_row_top * crop_h_ + v2::subtitle_ocr_output_height - 1u) / v2::subtitle_ocr_output_height + 3u;
    }

    void clear_scene(float gray = .15f) {
      pixels_.assign(std::size_t(source_w_) * crop_h_, pixel_t {gray, gray, gray, 1.f});
    }

    void paint(uint4_t rect, pixel_t color) {
      for (auto y = rect[1u]; y < std::min(rect[3u], crop_h_); ++y) {
        for (auto x = rect[0u]; x < std::min(rect[2u], source_w_); ++x) {
          pixels_[native_index(x, y)] = color;
        }
      }
    }

    void without_current_authority(bool empty_ocr = false) {
      plane_words_[v2::subtitle_plane_flags_word] = empty_ocr ? v2::subtitle_plane_current_valid_flag : 0u;
      plane_words_[v2::subtitle_plane_current_slot_mask_word] = 0u;
      std::fill(params_words_.begin() + 2u, params_words_.end(), 0u);
      if (empty_ocr) {
        ocr_words_[3u] = ocr_words_[4u] = 0u;
        std::fill(ocr_words_.begin() + v2::subtitle_ocr_raw_box_offset, ocr_words_.end(), 0u);
        plane_words_[v2::subtitle_plane_adaptive_offset + 13u] = 0u;
      }
    }

    // Direct current OCR8 fixture: no component selector, owner, pending geometry or box lifetime.
    // Raw/cover payloads remain independently paired in producer order, including small boxes.
    void current_covers(const std::vector<uint4_t> &covers, bool ribbon = false) {
      ASSERT_LE(covers.size(), v2::subtitle_plane_cover_capacity);
      ASSERT_TRUE(!ribbon || covers.size() == 1u);
      const auto count = static_cast<std::uint32_t>(covers.size());
      plane_words_[v2::subtitle_plane_flags_word] = v2::subtitle_plane_current_valid_flag;
      plane_words_[v2::subtitle_plane_current_slot_mask_word] = (1u << count) - 1u;
      params_words_[2u] = count;
      params_words_[3u] = ribbon ? 1u : 0u;
      params_words_[4u] = count != 0u ? v2::subtitle_condition_param_fully_active_value : 0u;
      params_words_[5u] = count != 0u ? plane_words_[v2::subtitle_plane_adaptive_offset + 3u] : 0u;
      ocr_words_[3u] = ocr_words_[4u] = count;
      std::fill(ocr_words_.begin() + v2::subtitle_ocr_raw_box_offset, ocr_words_.end(), 0u);
      for (std::uint32_t slot = 0u; slot < count; ++slot) {
        const auto core = ribbon ? uint4_t {content_[0u], constants_.field[2u], content_[2u], constants_.field[3u]} : covers[slot];
        for (auto offset : {v2::subtitle_ocr_raw_box_offset, v2::subtitle_ocr_final_box_offset}) {
          const auto base = offset + slot * v2::subtitle_ocr_box_word_count;
          const auto &rect = offset == v2::subtitle_ocr_raw_box_offset ? core : covers[slot];
          std::copy(rect.begin(), rect.end(), ocr_words_.begin() + base);
          ocr_words_[base + 4u] = std::bit_cast<std::uint32_t>(.9f);
          ocr_words_[base + 5u] = ribbon ? v2::subtitle_ocr_box_flag_ribbon : 0u;
          ocr_words_[base + 6u] = ribbon ? v2::subtitle_ocr_ribbon_min_structural_gaps + 1u : 1u;
          ocr_words_[base + 7u] = ribbon ? v2::subtitle_ocr_ribbon_min_structural_gaps : 0u;
        }
      }
      // Author a finite controller cover-union statistic without summing overlapping rectangles.
      std::uint32_t covered = 0u;
      for (auto y = content_[1u]; y < content_[3u]; ++y) {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> spans;
        for (const auto &rect : covers) if (y >= rect[1u] && y < rect[3u]) spans.emplace_back(rect[0u], rect[2u]);
        std::sort(spans.begin(), spans.end());
        std::uint32_t end = content_[0u];
        for (const auto &[left, right] : spans) {
          if (right > end) covered += right - std::max(end, left);
          end = std::max(end, right);
        }
      }
      plane_words_[v2::subtitle_plane_adaptive_offset + 13u] = covered;
    }

    void current_cores(const std::vector<uint4_t> &cores) {
      ASSERT_EQ(cores.size(), params_words_[2u]);
      for (std::size_t slot = 0u; slot < cores.size(); ++slot) {
        const auto base = v2::subtitle_ocr_raw_box_offset + slot * v2::subtitle_ocr_box_word_count;
        std::copy(cores[slot].begin(), cores[slot].end(), ocr_words_.begin() + base);
      }
    }

    std::uint32_t source_x_edge(std::uint32_t x) const {
      return content_[0u] + static_cast<std::uint32_t>(std::uint64_t(x) * (content_[2u] - content_[0u]) / source_w_);
    }

    std::uint32_t roi_edge(std::uint32_t row) const {
      const std::uint64_t numerator =
        (std::uint64_t(crop_top_) * v2::subtitle_ocr_output_height + std::uint64_t(row) * crop_h_) *
        (content_[3u] - content_[1u]);
      const std::uint64_t denominator = std::uint64_t(v2::subtitle_ocr_output_height) * source_h_;
      return content_[1u] + static_cast<std::uint32_t>((numerator + denominator - 1u) / denominator);
    }

    void make_authority(std::uint64_t frame, std::uint64_t timestamp, bool reset = false, std::uint64_t generation = 7u, std::uint32_t epoch = 3u) {
      const auto roi_top = roi_edge(v2::subtitle_ocr_safe_row_top);
      const auto roi_bottom = roi_edge(v2::subtitle_ocr_safe_row_bottom);
      native_cover_ = {20u, (v2::subtitle_ocr_safe_row_top * crop_h_ + v2::subtitle_ocr_output_height - 1u) / v2::subtitle_ocr_output_height, source_w_ - 20u, v2::subtitle_ocr_safe_row_bottom * crop_h_ / v2::subtitle_ocr_output_height};
      const auto map_floor = [](std::uint32_t value, std::uint32_t extent, std::uint32_t source) {
        return static_cast<std::uint32_t>(std::uint64_t(value) * extent / source);
      };
      const auto map_ceil = [](std::uint32_t value, std::uint32_t extent, std::uint32_t source) {
        return static_cast<std::uint32_t>((std::uint64_t(value) * extent + source - 1u) / source);
      };
      const auto cw = content_[2u] - content_[0u], ch = content_[3u] - content_[1u];
      field_cover_ = {content_[0u] + map_floor(native_cover_[0u], cw, source_w_), std::max(roi_top, content_[1u] + map_floor(crop_top_ + native_cover_[1u], ch, source_h_)), content_[0u] + map_ceil(native_cover_[2u], cw, source_w_), std::min(roi_bottom, content_[1u] + map_ceil(crop_top_ + native_cover_[3u], ch, source_h_))};
      constants_ = {{source_w_, source_h_, crop_top_, crop_h_}, {field_w_, field_h_, roi_top, roi_bottom}, content_, {std::uint32_t(frame), std::uint32_t(frame >> 32u), std::uint32_t(generation), std::uint32_t(generation >> 32u)}, {std::uint32_t(timestamp), std::uint32_t(timestamp >> 32u), reset ? 1u : 0u, 0u}, {1u, v2::subtitle_contour_recipe, 0u, 0u}};
      geometry_constants_ = {};
      geometry_constants_.raw_coordinate_scale = 2.25f;
      geometry_constants_.collapse_abs_epsilon = v2::collapse_abs_epsilon;
      geometry_constants_.requested_gain = v2::requested_gain_for_config(1.0f);
      geometry_constants_.max_horizontal_slope = v2::max_horizontal_slope;
      geometry_constants_.direct_container_limit = v2::direct_container_limit;
      geometry_constants_.convergence_curve_default = v2::convergence_curve_default;
      geometry_constants_.joint_plane_mode = v2::adaptive_policy_id;
      geometry_constants_.joint_observation_timestamp_low = std::uint32_t(timestamp);
      geometry_constants_.joint_observation_timestamp_high = std::uint32_t(timestamp >> 32u);
      geometry_words_ = v2::state_initial_words;
      const auto f = [](float value) {
        return std::bit_cast<std::uint32_t>(value);
      };
      geometry_words_[v2::center] = f(1.0f);
      geometry_words_[v2::inverse_scale] = f(1.0f / 2.25f);
      geometry_words_[v2::container_scale] = f(1.0f);
      geometry_words_[v2::calibration_revision] = 1u;
      geometry_words_[v2::frame_valid] = f(1.0f);
      geometry_words_[v2::confirmed_cut_count] = epoch;
      geometry_words_[v2::contract_tag_bits] = v2::contract_tag;
      geometry_words_[v2::renderer_authorization_bits] = v2::contract_tag;
      geometry_words_[v2::gain_last_observation_low] = std::uint32_t(timestamp);
      geometry_words_[v2::gain_last_observation_high] = std::uint32_t(timestamp >> 32u);
      geometry_words_[v2::gain_clock_armed] = 1u;
      geometry_words_[v2::gain_seed_count] = 1u;
      geometry_words_[v2::gain_target_zero] = f(1.0f);
      geometry_words_[v2::gain_target_inverse_scale] = f(1.0f / 2.25f);
      geometry_words_[v2::gain_target_nearest] = f(2.25f);
      geometry_words_[v2::gain_display_limit] = f(v2::direct_container_limit);
      geometry_words_[v2::gain_seed_first_low] = 1u;
      geometry_words_[v2::gain_seed_last_low] = 1u;
      geometry_words_[v2::gain_seed_mean_nearest] = f(2.25f);
      geometry_words_[v2::gain_seed_mean_zero] = f(1.0f);
      reseal_geometry();
      plane_words_.assign(v2::subtitle_plane_state_word_count, 0u);
      plane_words_[0u] = v2::subtitle_plane_state_schema;
      plane_words_[1u] = v2::subtitle_plane_state_tag;
      plane_words_[v2::subtitle_plane_flags_word] = v2::subtitle_plane_current_valid_flag;
      plane_words_[v2::subtitle_plane_current_slot_mask_word] = 1u;
      for (std::uint32_t lane = 0u; lane < 2u; ++lane) {
        plane_words_[v2::subtitle_plane_frame_offset + lane] = constants_.frame[lane];
        plane_words_[v2::subtitle_plane_analysis_generation_offset + lane] = constants_.frame[2u + lane];
        plane_words_[v2::subtitle_plane_observation_timestamp_offset + lane] = constants_.observation[lane];
        plane_words_[v2::subtitle_plane_source_extent_offset + lane] = constants_.source[lane];
        plane_words_[v2::subtitle_plane_field_extent_offset + lane] = constants_.field[lane];
        plane_words_[v2::subtitle_plane_roi_offset + lane] = constants_.field[2u + lane];
      }
      std::copy(content_.begin(), content_.end(), plane_words_.begin() + v2::subtitle_plane_content_offset);
      plane_words_[v2::subtitle_plane_cut_epoch_word] = epoch;
      const auto adaptive = v2::subtitle_plane_adaptive_offset;
      plane_words_[adaptive] = 1u;
      plane_words_[adaptive + 1u] = 8u;
      plane_words_[adaptive + 4u] = constants_.observation[0u];
      plane_words_[adaptive + 5u] = constants_.observation[1u];
      plane_words_[adaptive + 12u] = f(v2::direct_container_limit);
      plane_words_[adaptive + 13u] = (field_cover_[2u] - field_cover_[0u]) * (field_cover_[3u] - field_cover_[1u]);
      params_words_ = {v2::subtitle_condition_param_schema, v2::subtitle_condition_param_tag, 1u, 0u, 2u, 0u};
      ocr_words_.assign(v2::subtitle_ocr_record_word_count, 0u);
      ocr_words_[0u] = v2::subtitle_ocr_record_schema;
      ocr_words_[1u] = v2::subtitle_ocr_record_tag;
      ocr_words_[2u] = ocr_words_[3u] = ocr_words_[4u] = 1u;
      for (std::size_t i = 0u; i < 4u; ++i) {
        ocr_words_[5u + i] = constants_.frame[i];
      }
      ocr_words_[9u] = source_w_;
      ocr_words_[10u] = source_h_;
      ocr_words_[11u] = field_w_;
      ocr_words_[12u] = field_h_;
      ocr_words_[13u] = roi_top;
      ocr_words_[14u] = roi_bottom;
      for (auto offset : {v2::subtitle_ocr_raw_box_offset, v2::subtitle_ocr_final_box_offset}) {
        std::copy(field_cover_.begin(), field_cover_.end(), ocr_words_.begin() + offset);
        ocr_words_[offset + 4u] = f(.9f);
        ocr_words_[offset + 6u] = 1u;
      }
      cut_words_.assign(sbs_adaptive_state::word_count, 0u);
      cut_words_[sbs_adaptive_state::index(sbs_adaptive_state::word_e::cut_contract_tag_bits)] = sbs_adaptive_state::cut_contract_tag;
      cut_words_[sbs_adaptive_state::index(sbs_adaptive_state::word_e::initialized)] = f(1.0f);
      cut_words_[sbs_adaptive_state::index(sbs_adaptive_state::word_e::hard_cut_count)] = epoch;
      probability_values_.assign(v2::subtitle_ocr_output_width * v2::subtitle_ocr_output_height, 0.0f);
      for (std::uint32_t y = 0u; y < v2::subtitle_ocr_output_height; ++y) {
        for (std::uint32_t x = 0u; x < v2::subtitle_ocr_output_width; ++x) {
          const double px = (double(x) + .5) * source_w_ / v2::subtitle_ocr_output_width;
          const double py = (double(y) + .5) * crop_h_ / v2::subtitle_ocr_output_height;
          if (px >= native_cover_[0u] && px < native_cover_[2u] && py >= native_cover_[1u] && py < native_cover_[3u]) {
            probability_values_[std::size_t(y) * v2::subtitle_ocr_output_width + x] = .9f;
          }
        }
      }
    }

    void reseal_geometry() {
      geometry_words_[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(geometry_words_);
    }

    bool upload() {
      std::vector<half_pixel_t> half_pixels(pixels_.size());
      for (std::size_t i = 0u; i < pixels_.size(); ++i) {
        for (std::size_t lane = 0u; lane < 4u; ++lane) {
          half_pixels[i][lane] = float_to_half(pixels_[i][lane]);
        }
      }
      context_->UpdateSubresource(display_.texture.Get(), 0u, nullptr, half_pixels.data(), source_w_ * sizeof(half_pixel_t), 0u);
      context_->UpdateSubresource(probability_.buffer.Get(), 0u, nullptr, probability_values_.data(), 0u, 0u);
      context_->UpdateSubresource(plane_.buffer.Get(), 0u, nullptr, plane_words_.data(), 0u, 0u);
      context_->UpdateSubresource(params_.buffer.Get(), 0u, nullptr, params_words_.data(), 0u, 0u);
      context_->UpdateSubresource(ocr_.buffer.Get(), 0u, nullptr, ocr_words_.data(), 0u, 0u);
      context_->UpdateSubresource(cut_.buffer.Get(), 0u, nullptr, cut_words_.data(), 0u, 0u);
      context_->UpdateSubresource(geometry_.buffer.Get(), 0u, nullptr, geometry_words_.data(), 0u, 0u);
      context_->UpdateSubresource(geometry_cb_.Get(), 0u, nullptr, &geometry_constants_, 0u, 0u);
      context_->UpdateSubresource(contour_cb_.Get(), 0u, nullptr, &constants_, 0u, 0u);
      return true;
    }

    void unbind() {
      if (!context_) {
        return;
      }
      std::array<ID3D11ShaderResourceView *, 12u> srvs {};
      std::array<ID3D11UnorderedAccessView *, 6u> uavs {};
      context_->CSSetShaderResources(0u, srvs.size(), srvs.data());
      context_->CSSetUnorderedAccessViews(0u, uavs.size(), uavs.data(), nullptr);
    }

    void bind_inputs() {
      const std::array<ID3D11ShaderResourceView *, 12u> srvs {
        display_.srv.Get(),
        probability_.srv.Get(),
        plane_.srv.Get(),
        params_.srv.Get(),
        nullptr,
        nullptr,
        features_.srv.Get(),
        identity_.srv.Get(),
        cut_.srv.Get(),
        ocr_.srv.Get(),
        geometry_.srv.Get(),
        nullptr
      };
      context_->CSSetShaderResources(0u, srvs.size(), srvs.data());
      const std::array<ID3D11Buffer *, 2u> cbuffers {contour_cb_.Get(), geometry_cb_.Get()};
      context_->CSSetConstantBuffers(0u, cbuffers.size(), cbuffers.data());
    }

    bool run(bool zero_groups = false) {
      unbind();
      if (!upload()) {
        return false;
      }
      if (!zero_groups) {
        const UINT poison[4] {0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
        context_->ClearUnorderedAccessViewUint(support_.uav.Get(), poison);
        context_->ClearUnorderedAccessViewUint(field_.uav.Get(), poison);
        context_->ClearUnorderedAccessViewUint(completed_probability_.uav.Get(), poison);
        const float poison_color[4] {.75f, .75f, .75f, .75f};
        context_->ClearUnorderedAccessViewFloat(completed_display_.uav.Get(), poison_color);
        context_->ClearUnorderedAccessViewFloat(next_features_.uav.Get(), poison_color);
      }
      const std::array<std::uint32_t, 3u> arguments {
        zero_groups ? 0u : (field_w_ + 15u) / 16u,
        zero_groups ? 0u : (field_h_ + 15u) / 16u,
        zero_groups ? 0u : 1u
      };
      context_->UpdateSubresource(indirect_.Get(), 0u, nullptr, arguments.data(), 0u, 0u);
      bind_inputs();
      ID3D11UnorderedAccessView *null_uav = nullptr;
      context_->CSSetShader(shaders_[0u].Get(), nullptr, 0u);
      context_->CSSetUnorderedAccessViews(0u, 1u, support_.uav.GetAddressOf(), nullptr);
      context_->CSSetUnorderedAccessViews(2u, 1u, next_features_.uav.GetAddressOf(), nullptr);
      context_->DispatchIndirect(indirect_.Get(), 0u);
      context_->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);
      context_->CSSetUnorderedAccessViews(2u, 1u, &null_uav, nullptr);
      context_->CSSetShaderResources(5u, 1u, support_.srv.GetAddressOf());
      context_->CSSetShader(shaders_[1u].Get(), nullptr, 0u);
      context_->CSSetUnorderedAccessViews(1u, 1u, field_.uav.GetAddressOf(), nullptr);
      context_->DispatchIndirect(indirect_.Get(), 0u);
      context_->CSSetUnorderedAccessViews(1u, 1u, &null_uav, nullptr);
      // Consume previous RGB/context before current history becomes UAV output.
      const std::array<ID3D11ShaderResourceView *, 2u> null_previous {};
      context_->CSSetShaderResources(6u, null_previous.size(), null_previous.data());
      context_->CSSetShaderResources(11u, 1u, next_features_.srv.GetAddressOf());
      const std::array<ID3D11UnorderedAccessView *, 2u> history_outputs {features_.uav.Get(), identity_.uav.Get()};
      context_->CSSetShader(shaders_[2u].Get(), nullptr, 0u);
      context_->CSSetUnorderedAccessViews(2u, history_outputs.size(), history_outputs.data(), nullptr);
      context_->DispatchIndirect(indirect_.Get(), 0u);
      const std::array<ID3D11UnorderedAccessView *, 2u> null_outputs {};
      context_->CSSetUnorderedAccessViews(2u, null_outputs.size(), null_outputs.data(), nullptr);
      ID3D11ShaderResourceView *null_next = nullptr;
      context_->CSSetShaderResources(11u, 1u, &null_next);
      const std::array<ID3D11UnorderedAccessView *, 2u> evidence_outputs {completed_display_.uav.Get(), completed_probability_.uav.Get()};
      context_->CSSetShader(shaders_[3u].Get(), nullptr, 0u);
      context_->CSSetUnorderedAccessViews(4u, evidence_outputs.size(), evidence_outputs.data(), nullptr);
      context_->DispatchIndirect(indirect_.Get(), 0u);
      unbind();
      return read_texture(device_.Get(), context_.Get(), support_.texture.Get(), result_.support) &&
             read_texture(device_.Get(), context_.Get(), field_.texture.Get(), result_.field) &&
             read_texture(device_.Get(), context_.Get(), features_.texture.Get(), result_.features) &&
             read_texture(device_.Get(), context_.Get(), next_features_.texture.Get(), result_.next_features) &&
             read_texture(device_.Get(), context_.Get(), completed_display_.texture.Get(), result_.source) &&
             read_buffer(device_.Get(), context_.Get(), identity_.buffer.Get(), result_.identity) &&
             read_buffer(device_.Get(), context_.Get(), completed_probability_.buffer.Get(), result_.probability_bits);
    }

    // Real source-frame/time progression is authored on every dispatch. Pixel evidence survives;
    // authority and probability records are freshly paired, rather than faking a count header.
    bool observe(std::uint64_t frame) {
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      return run();
    }

    bool mature_observations(std::uint32_t first = 1u, std::uint32_t count = confirmation_count) {
      for (auto frame = first; frame < first + count; ++frame) {
        if (!observe(frame)) {
          return false;
        }
      }
      return true;
    }

    void paint_point(std::uint32_t x = 60u, float gray = .75f) {
      pixels_[native_index(x, body_y() + 5u)] = {gray, gray, gray, 1.f};
    }

    std::size_t point_index(std::uint32_t x = 60u) const {
      return native_index(x, body_y() + 5u);
    }

    void expect_count(std::size_t point, std::uint32_t count) {
      EXPECT_EQ(result_.features[point][3u], feature_code(count));
      EXPECT_EQ(result_.support[point], count == confirmation_count ? 1u : 0u);
      EXPECT_EQ(result_.next_features, result_.features)
        << "persistent history is an exact copy of staged next features, never an aliased neighbor update";
    }

    void expect_flat_patch(uint4_t rect, std::uint32_t count, pixel_t anchor) {
      ASSERT_LT(rect[0u], rect[2u]);
      ASSERT_LT(rect[1u], rect[3u]);
      ASSERT_LE(rect[2u], source_w_);
      ASSERT_LE(rect[3u], crop_h_);
      anchor[3u] = feature_code(count);
      for (auto y = rect[1u]; y < rect[3u]; ++y) {
        for (auto x = rect[0u]; x < rect[2u]; ++x) {
          const auto point = native_index(x, y);
          ASSERT_EQ(result_.features[point], anchor) << "patch pixel " << x << ',' << y;
          ASSERT_EQ(result_.support[point], count == confirmation_count ? 1u : 0u)
            << "patch pixel " << x << ',' << y;
        }
      }
      EXPECT_EQ(result_.next_features, result_.features);
    }

    void replace_prior_features(const std::vector<pixel_t> &values) {
      unbind();
      context_->UpdateSubresource(features_.texture.Get(), 0u, nullptr, values.data(), source_w_ * sizeof(pixel_t), 0u);
    }

    void replace_prior_identity(const std::vector<std::uint32_t> &values) {
      unbind();
      context_->UpdateSubresource(identity_.buffer.Get(), 0u, nullptr, values.data(), 0u, 0u);
    }

    std::size_t native_index(std::uint32_t x, std::uint32_t y) const {
      return std::size_t(y) * source_w_ + x;
    }

    std::size_t pin_count(const std::vector<std::uint32_t> &values) const {
      return std::count_if(values.begin(), values.end(), [](auto value) {
        return value != 0u;
      });
    }

    void expect_no_mask() {
      EXPECT_EQ(pin_count(result_.support), 0u);
      EXPECT_EQ(pin_count(result_.field), 0u);
      for (auto value : result_.support) {
        EXPECT_LE(value, 1u);
      }
    }

    void expect_cleared_snapshot() {
      expect_no_mask();
      ASSERT_EQ(result_.identity.size(), v2::subtitle_contour_history_word_count);
      EXPECT_EQ(result_.identity[2u], 0u);
      EXPECT_EQ(result_.identity[snapshot_valid_word], 0u);
      EXPECT_EQ(result_.features, std::vector<pixel_t>(pixels_.size(), pixel_t {}));
      EXPECT_EQ(result_.source, std::vector<half_pixel_t>(pixels_.size(), half_pixel_t {}));
      EXPECT_EQ(pin_count(result_.probability_bits), 0u);
    }

    void expect_evidence_exact_copy() {
      std::vector<half_pixel_t> source(pixels_.size());
      for (std::size_t i = 0u; i < pixels_.size(); ++i) {
        for (std::size_t lane = 0u; lane < 4u; ++lane) {
          source[i][lane] = float_to_half(pixels_[i][lane]);
        }
      }
      std::vector<std::uint32_t> probability(probability_values_.size());
      std::transform(probability_values_.begin(), probability_values_.end(), probability.begin(), [](float value) {
        return std::bit_cast<std::uint32_t>(value);
      });
      EXPECT_EQ(result_.source, source);
      EXPECT_EQ(result_.probability_bits, probability);
    }

    void expect_tuple_equal(const snapshot_t &old) {
      EXPECT_EQ(result_.support, old.support);
      EXPECT_EQ(result_.field, old.field);
      EXPECT_EQ(result_.identity, old.identity);
      EXPECT_EQ(result_.features, old.features);
      EXPECT_EQ(result_.source, old.source);
      EXPECT_EQ(result_.probability_bits, old.probability_bits);
    }

    void expect_projection_oracle() {
      std::vector<std::uint32_t> expected(std::size_t(field_w_) * field_h_, 0u);
      const std::int64_t cw = content_[2u] - content_[0u], ch = content_[3u] - content_[1u];
      // Scatter source-pixel rectangles against open node support, using exact rational
      // intersections independent of the shader's gather recurrence. Endpoint equality must
      // not turn into an accidental pin through floating-point oracle rounding.
      for (std::uint32_t y = 0u; y < crop_h_; ++y) {
        for (std::uint32_t x = 0u; x < source_w_; ++x) {
          if (result_.support[native_index(x, y)] == 0u) {
            continue;
          }
          const auto absolute_y = std::int64_t(crop_top_) + y;
          const int x_begin = int(content_[0u] + x * cw / source_w_) - 2;
          const int x_end = int(content_[0u] + ((x + 1u) * cw + source_w_ - 1u) / source_w_) + 2;
          const int y_begin = int(content_[1u] + absolute_y * ch / source_h_) - 2;
          const int y_end = int(content_[1u] + ((absolute_y + 1) * ch + source_h_ - 1u) / source_h_) + 2;
          for (int fy = std::max(int(content_[1u]), y_begin); fy < std::min(int(content_[3u]), y_end); ++fy) {
            for (int fx = std::max(int(content_[0u]), x_begin); fx < std::min(int(content_[2u]), x_end); ++fx) {
              const auto local_x = std::int64_t(fx) - content_[0u];
              const auto local_y = std::int64_t(fy) - content_[1u];
              if ((2 * local_x + 3) * source_w_ > 2 * x * cw && (2 * local_x - 1) * source_w_ < 2 * (x + 1u) * cw && (2 * local_y + 3) * source_h_ > 2 * absolute_y * ch && (2 * local_y - 1) * source_h_ < 2 * (absolute_y + 1) * ch) {
                expected[std::size_t(fy) * field_w_ + fx] = 1u;
              }
            }
          }
        }
      }
      EXPECT_EQ(result_.field, expected);
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    std::array<ComPtr<ID3D11ComputeShader>, 4u> shaders_;
    ComPtr<ID3D11Buffer> contour_cb_, geometry_cb_, indirect_;
    texture_t display_, support_, field_, features_, next_features_, completed_display_;
    buffer_t probability_, plane_, params_, ocr_, cut_, geometry_, identity_, completed_probability_;
    std::uint32_t source_w_ {}, source_h_ {}, crop_h_ {}, crop_top_ {}, field_w_ {}, field_h_ {};
    uint4_t content_ {}, native_cover_ {}, field_cover_ {};
    contour_constants_t constants_ {};
    v2::constants_t geometry_constants_ {};
    v2::state_words_t geometry_words_ {};
    std::vector<std::uint32_t> plane_words_, params_words_, ocr_words_, cut_words_;
    std::vector<float> probability_values_;
    std::vector<pixel_t> pixels_;
    snapshot_t result_;
  };

  TEST_F(HostSbsSubtitleContourGpuTest, FirstCurrentObservationBootstrapsOneAndCopiesActualRgb) {
    paint_point();
    ASSERT_TRUE(observe(1u));
    expect_no_mask();
    expect_count(point_index(), 1u);
    EXPECT_EQ(result_.identity[2u], 1u);
    EXPECT_EQ(result_.identity[snapshot_valid_word], 1u);
    EXPECT_EQ(result_.features[point_index()], (pixel_t {.75f, .75f, .75f, feature_code(1u)}));
    expect_count(point_index(100u), 1u);
    expect_evidence_exact_copy();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, FourCurrentObservationsConfirmCompleteFlatGlyphPatchesWithZeroProbability) {
    const std::array colors {pixel_t {.9375f, .9375f, .9375f, 1.f}, pixel_t {0.f, 0.f, 0.f, 1.f}, pixel_t {.125f, .75f, .5f, 1.f}, pixel_t {.25f, .3125f, .375f, 1.f}};
    const std::array backgrounds {.0625f, .25f, .5f, .8125f};
    const auto y = body_y() + 5u;
    for (std::uint32_t frame = 1u; frame <= confirmation_count; ++frame) {
      const auto background = backgrounds[frame - 1u];
      clear_scene(background);
      for (std::size_t k = 0u; k < colors.size(); ++k) {
        const auto x = 35u + 25u * std::uint32_t(k);
        paint({x - 3u, y - 3u, x + 4u, y + 4u}, colors[k]);
      }
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      std::fill(probability_values_.begin(), probability_values_.end(), 0.f);  // P is validity evidence, not alpha.
      ASSERT_TRUE(run());
      for (std::size_t k = 0u; k < colors.size(); ++k) {
        const auto x = 35u + 25u * std::uint32_t(k);
        expect_flat_patch({x - 3u, y - 3u, x + 4u, y + 4u}, frame, colors[k]);
      }
      // Each background change exceeds .08; subsequent RGB is separated from every patch anchor.
      expect_count(point_index(100u), 1u);
      EXPECT_EQ(result_.features[point_index(100u)], (pixel_t {background, background, background, feature_code(1u)}));
      if (frame < confirmation_count) {
        expect_no_mask();
      }
      expect_evidence_exact_copy();
    }
    EXPECT_EQ(pin_count(result_.support), colors.size() * 49u)
      << "all authored flat patch pixels, including interiors, and no changing background";
    for (auto word : result_.support) {
      ASSERT_LE(word, 1u);
    }
    expect_projection_oracle();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, FixedAnchorRejectsChainedAdjacentColorDrift) {
    clear_scene(0.f);
    const std::array colors {.5f, .5625f, .625f, .6875f};  // Every adjacent difference .0625 < .08.
    const std::array counts {1u, 2u, 1u, 2u};
    for (std::uint32_t frame = 1u; frame <= 4u; ++frame) {
      paint_point(60u, colors[frame - 1u]);
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), counts[frame - 1u]);
      const float anchor = frame <= 2u ? .5f : .625f;
      EXPECT_EQ(result_.features[point_index()][0u], anchor);
      EXPECT_EQ(result_.support[point_index()], 0u);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, EncodingNoiseWithinFixedAnchorConfirmsOnFourthObservation) {
    clear_scene(0.f);
    const std::array colors {.5f, .53125f, .46875f, .5625f};
    for (std::uint32_t frame = 1u; frame <= 4u; ++frame) {
      paint_point(60u, colors[frame - 1u]);
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame);
      EXPECT_EQ(result_.features[point_index()][0u], .5f) << "reference is first RGB, not the most recent input";
      expect_evidence_exact_copy();
    }
    EXPECT_NE(result_.source[point_index()][0u], float_to_half(result_.features[point_index()][0u]))
      << "published source is current RGB while history carries the fixed anchor";
  }

  TEST_F(HostSbsSubtitleContourGpuTest, TwoPixelJitterCarriesOneReferenceAcrossFourGenuineFrames) {
    const std::array colors {pixel_t {.875f, .625f, .3125f, 1.f}, pixel_t {.84375f, .59375f, .28125f, 1.f}, pixel_t {.90625f, .65625f, .34375f, 1.f}, pixel_t {.8125f, .5625f, .25f, 1.f}};
    for (std::uint32_t frame = 1u; frame <= 4u; ++frame) {
      clear_scene(.125f * float(frame - 1u));
      const auto x = 50u + 2u * (frame - 1u);
      pixels_[point_index(x)] = colors[frame - 1u];
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(x), frame);
      for (std::size_t lane = 0u; lane < 3u; ++lane) {
        EXPECT_EQ(result_.features[point_index(x)][lane], colors[0u][lane]);
      }
    }
    EXPECT_EQ(result_.support[point_index(50u)], 0u) << "changed old position is not retained";
    EXPECT_EQ(result_.features[point_index(50u)], (pixel_t {.375f, .375f, .375f, feature_code(1u)}));
    // This is an explicit <=2 per-observation correspondence chain, not four observations
    // of one absolute source coordinate or semantic proof that the tracked color is text.
  }

  TEST_F(HostSbsSubtitleContourGpuTest, MovementBeyondJitterRestartsFourObservationConfirmation) {
    clear_scene(0.f);
    paint_point(50u);
    ASSERT_TRUE(observe(1u));
    for (std::uint32_t frame = 2u; frame <= 5u; ++frame) {
      clear_scene(.125f * float(frame - 1u));
      paint_point(60u);
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame - 1u);
    }
    EXPECT_EQ(result_.support[point_index(50u)], 0u);
    EXPECT_EQ(result_.features[point_index(50u)], (pixel_t {.5f, .5f, .5f, feature_code(1u)}));
  }

  TEST_F(HostSbsSubtitleContourGpuTest, PriorChoiceUsesColorThenSpatialDistanceRatherThanGreatestCount) {
    // The neighbor has three real observations; the same-pixel anchor is born on frame3.
    paint_point(61u, .625f);
    ASSERT_TRUE(observe(1u));
    ASSERT_TRUE(observe(2u));
    paint_point(60u, .5f);
    ASSERT_TRUE(observe(3u));
    expect_count(point_index(61u), 3u);
    expect_count(point_index(), 1u);
    paint_point(60u, .5625f);  // Both anchors differ exactly .0625; same pixel wins the tie.
    ASSERT_TRUE(observe(4u));
    expect_count(point_index(), 2u);
    EXPECT_EQ(result_.features[point_index()][0u], .5f);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, SamePixelReferenceConfirmsDespiteCloserNewNeighborColor) {
    paint_point(60u, .5f);
    ASSERT_TRUE(observe(1u));
    ASSERT_TRUE(observe(2u));
    paint_point(61u, .625f);  // Too far from the old .5 anchor: a genuine new streak.
    ASSERT_TRUE(observe(3u));
    expect_count(point_index(60u), 3u);
    expect_count(point_index(61u), 1u);

    paint_point(60u, .578125f);  // .078125 from own anchor, .046875 from new neighbour.
    ASSERT_TRUE(observe(4u));
    expect_count(point_index(60u), 4u);
    EXPECT_EQ(result_.features[point_index(60u)][0u], .5f);
    EXPECT_EQ(result_.support[point_index(60u)], 1u);
    expect_count(point_index(61u), 2u);
    EXPECT_EQ(result_.support[point_index(61u)], 0u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, EqualColorAndSpatialTiesUseScanOrderNotGreatestCount) {
    paint_point(62u, .5f);
    ASSERT_TRUE(observe(1u));
    ASSERT_TRUE(observe(2u));
    paint_point(58u, .5f);  // Four pixels from older anchor, so this is a genuine new streak1.
    ASSERT_TRUE(observe(3u));
    expect_count(point_index(58u), 1u);
    expect_count(point_index(62u), 3u);
    paint_point(60u, .5f);
    ASSERT_TRUE(observe(4u));
    expect_count(point_index(), 2u);  // Equal RGB/dist2 selects left anchor before count3 at right.
  }

  TEST_F(HostSbsSubtitleContourGpuTest, StableUniformBackgroundWithinCurrentCoverIsAnExplicitAmbiguity) {
    clear_scene(.5f);
    for (std::uint32_t frame = 1u; frame <= confirmation_count; ++frame) {
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame);
      expect_count(point_index(5u), 0u);
      if (frame < confirmation_count) {
        expect_no_mask();
      }
      expect_evidence_exact_copy();
    }
    ASSERT_GT(pin_count(result_.support), 0u)
      << "stable uniform background satisfies RGB duration; this is not semantic glyph truth";
    expect_projection_oracle();
    paint_point(60u, .5625f);
    ASSERT_TRUE(observe(5u));
    expect_count(point_index(), confirmation_count);
    EXPECT_EQ(result_.features[point_index()], (pixel_t {.5f, .5f, .5f, feature_code(confirmation_count)}))
      << "similar new RGB can inherit confirmed background; semantic cue birth is not recognized";
    expect_evidence_exact_copy();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, ValidCurrentPixelConfirmsWithoutNeighborAppearanceEvidence) {
    clear_scene(.5f);
    paint_point(60u, .625f);
    ASSERT_TRUE(mature_observations());
    expect_count(point_index(), 4u);
    ASSERT_TRUE(resize(160u, 90u));
    clear_scene(.5f);
    const auto x = 60u, y = body_y() + 5u;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx != 0 || dy != 0) {
          pixels_[native_index(std::uint32_t(int(x) + dx), std::uint32_t(int(y) + dy))] = {};
        }
      }
    }
    for (std::uint32_t frame = 1u; frame <= confirmation_count; ++frame) {
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame);
    }
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx != 0 || dy != 0) {
          const auto point = native_index(std::uint32_t(int(x) + dx), std::uint32_t(int(y) + dy));
          EXPECT_EQ(result_.features[point], (pixel_t {}));
          EXPECT_EQ(result_.support[point], 0u);
        }
      }
    }
    expect_evidence_exact_copy();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, InvalidPlaneAndLegalEmptyCoversKeepRgbButDoNotPrecount) {
    for (bool empty : {false, true}) {
      SCOPED_TRACE(empty);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      without_current_authority(empty);
      ASSERT_TRUE(run());
      expect_no_mask();
      expect_count(point_index(), 0u);
      EXPECT_EQ(result_.identity[2u], 0u);
      EXPECT_EQ(result_.identity[snapshot_valid_word], 1u);
      EXPECT_EQ(result_.identity[3u], 3u);
      expect_evidence_exact_copy();
      for (std::uint32_t frame = 2u; frame <= 5u; ++frame) {
        ASSERT_TRUE(observe(frame));
        expect_count(point_index(), frame - 1u);
      }
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, InvalidPlaneAndOffCoverObservationsClearPositiveCount) {
    paint_point();
    ASSERT_TRUE(mature_observations(1u, 3u));
    make_authority(4u, 1300001u);
    without_current_authority();
    ASSERT_TRUE(run());
    expect_count(point_index(), 0u);
    EXPECT_EQ(result_.identity[snapshot_valid_word], 1u);
    for (std::uint32_t frame = 5u; frame <= 8u; ++frame) {
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame - 4u);
    }
    make_authority(9u, 1800001u);
    auto removed = field_cover_;
    removed[0u] = source_x_edge(90u);
    current_covers({removed});
    ASSERT_TRUE(run());
    expect_count(point_index(), 0u);
    ASSERT_TRUE(observe(10u));
    expect_count(point_index(), 1u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, ExactDuplicatesDoNotAdvanceOrBorrowNeighborCounts) {
    paint_point();
    for (std::uint32_t frame = 1u; frame <= 4u; ++frame) {
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame);
      const auto held = result_;
      ASSERT_TRUE(run());
      expect_tuple_equal(held);
      expect_count(point_index(), frame);
    }
    paint_point(62u);  // Same color lies within jitter but source frame/time did not advance.
    ASSERT_TRUE(run());
    expect_count(point_index(62u), 0u);
    expect_count(point_index(), 4u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, DuplicateAfterLegalEmptyCannotDiscoverCurrentCoverPixels) {
    paint_point();
    current_covers({});
    ASSERT_TRUE(run());
    expect_count(point_index(), 0u);
    EXPECT_EQ(result_.identity[snapshot_valid_word], 1u);
    make_authority(1u, 1000001u); // Same genuine source observation, now a new current cover.
    ASSERT_TRUE(run());
    expect_count(point_index(), 0u);
    for (std::uint32_t frame = 2u; frame <= 5u; ++frame) {
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame - 1u);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, MalformedControllerCannotAuthorizeFourthConfirmation) {
    for (std::uint32_t fault = 0u; fault < 9u; ++fault) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      make_authority(4u, 1300001u);
      const auto a = v2::subtitle_plane_adaptive_offset;
      switch (fault) {
        case 0u: plane_words_[a + 2u] = 3u; break; // Above ceil(160/100).
        case 1u: plane_words_[a + 10u] = 3u; break;
        case 2u: plane_words_[a + 11u] = 3u; break;
        case 3u: plane_words_[a + 1u] |= 3u; break; // Opposing dwell flags.
        case 4u: plane_words_[a + 1u] &= ~8u; break; // Nonzero clock without clock flag.
        case 5u: plane_words_[a + 6u] = 1u; break; // Dwell start without approach flag.
        case 6u: plane_words_[a + 14u] = plane_words_[a + 13u] + 1u; break;
        case 7u: plane_words_[a + 15u] = plane_words_[a + 13u] + 1u; break;
        case 8u: plane_words_[a + 15u] = 1u; break; // Canonical invalid probe cannot accept forged active SCP4.
      }
      ASSERT_TRUE(run());
      expect_count(point_index(), 0u);
      EXPECT_EQ(result_.identity[2u], 0u);
      EXPECT_EQ(result_.identity[snapshot_valid_word], 1u);
      expect_evidence_exact_copy();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, AddedRemovedReorderedSplitAndMergedCoversKeepStablePixels) {
    const pixel_t text {.3125f, .75f, .5625f, 1.f};
    auto left = field_cover_, right = field_cover_;
    left[2u] = source_x_edge(90u);
    right[0u] = source_x_edge(100u);
    for (std::uint32_t frame = 1u; frame <= 8u; ++frame) {
      clear_scene(.125f * float(frame - 1u)); // Mature text survives a genuinely changed background.
      pixels_[point_index(60u)] = pixels_[point_index(110u)] = text;
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      if (frame <= 3u || frame == 8u) current_covers({left});
      else if (frame == 4u) current_covers({left, right});
      else if (frame == 5u) current_covers({right, left});
      else if (frame == 6u) {
        auto a = left, b = left;
        a[2u] = b[0u] = source_x_edge(55u);
        current_covers({right, b, a});
      } else current_covers({field_cover_});
      ASSERT_TRUE(run());
      expect_count(point_index(60u), std::min(frame, confirmation_count));
      expect_count(point_index(110u), frame <= 3u || frame == 8u ? 0u : frame - 3u);
      expect_count(point_index(75u), 1u);
      for (auto word = reserved_word; word < result_.identity.size(); ++word) EXPECT_EQ(result_.identity[word], 0u);
      expect_evidence_exact_copy();
    }
  }

  // This fixture authors a valid UIP reset bit; the resolver tests exercise its probability
  // witness. Neither fixture replaces native clip evidence or supplies semantic glyph truth.
  TEST_F(HostSbsSubtitleContourGpuTest, CurrentProbabilityFlagsPreserveQualifyingSamePixelRgbInsideFlaggedCores) {
    const std::array colors {pixel_t {.9375f, .9375f, .9375f, 1.f}, pixel_t {0.f, 0.f, 0.f, 1.f},
      pixel_t {.125f, .75f, .5f, 1.f}, pixel_t {.25f, .3125f, .375f, 1.f}};
    const auto y = body_y() + 5u;
    clear_scene(.0625f);
    for (std::size_t k = 0u; k < colors.size(); ++k) {
      const auto x = 35u + 25u * std::uint32_t(k);
      paint({x - 3u, y - 3u, x + 4u, y + 4u}, colors[k]);
    }
    ASSERT_TRUE(mature_observations());
    auto core = field_cover_;
    core[0u] = source_x_edge(50u);
    core[2u] = source_x_edge(100u);
    for (std::uint32_t frame = 5u; frame <= 8u; ++frame) {
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      current_cores({core});
      if (frame == 5u) plane_words_[v2::subtitle_plane_history_reset_mask_word] = 1u;
      // Finite detector noise is not alpha. A flag does not discard a qualifying same-pixel
      // fixed RGB reference; zero probability remains eligible evidence.
      std::fill(probability_values_.begin(), probability_values_.end(), frame % 2u ? 0.f : .015625f);
      ASSERT_TRUE(run());
      for (std::size_t k = 0u; k < colors.size(); ++k) {
        const auto x = 35u + 25u * std::uint32_t(k);
        expect_flat_patch({x - 3u, y - 3u, x + 4u, y + 4u}, confirmation_count, colors[k]);
      }
      EXPECT_GT(pin_count(result_.support), 0u) << "mature expanded-cover margins stay protected";
      expect_evidence_exact_copy();
    }
    expect_projection_oracle();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, RepeatedCurrentFlagsAdvanceSamePixelCountsAndPreserveTheFixedRgbAnchor) {
    clear_scene(0.f);
    // Every input stays within .08 of the first fixed anchor, including .078125 drift.
    // The valid current flag cannot reset counts1/2/3 or an already mature count4.
    const std::array colors {.5f, .5625f, .578125f, .46875f, .53125f};
    for (std::uint32_t frame = 1u; frame <= colors.size(); ++frame) {
      paint_point(60u, colors[frame - 1u]);
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      plane_words_[v2::subtitle_plane_history_reset_mask_word] = 1u;
      ASSERT_TRUE(run());
      expect_count(point_index(), std::min(frame, confirmation_count));
      for (std::size_t lane = 0u; lane < 3u; ++lane) EXPECT_EQ(result_.features[point_index()][lane], .5f);
      EXPECT_EQ(result_.source[point_index()][0u], float_to_half(colors[frame - 1u]));
      expect_evidence_exact_copy();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, FlaggedRgbMismatchBlocksAvailableMatureNeighborOnlyOnThatGenuineObservation) {
    for (const bool flagged : {false, true}) {
      SCOPED_TRACE(flagged);
      ASSERT_TRUE(resize(160u, 90u));
      clear_scene(.125f);
      paint_point(60u, .75f);
      paint_point(62u, .5f); // A genuinely acquired mature reference at the2px search boundary.
      paint_point(85u, .9375f); // A stable same-pixel reference inside the same flagged core.
      ASSERT_TRUE(mature_observations());
      expect_count(point_index(62u), confirmation_count);
      auto core = field_cover_;
      core[0u] = source_x_edge(50u);
      core[2u] = source_x_edge(90u);
      paint_point(60u, .5625f); // Own anchor mismatch>.08, neighbor distance .0625<=.08.
      for (std::uint32_t frame = 5u; frame <= 8u; ++frame) {
        make_authority(frame, 1000001u + (frame - 1u) * 100000u);
        current_cores({core});
        plane_words_[v2::subtitle_plane_history_reset_mask_word] = flagged ? 1u : 0u;
        ASSERT_TRUE(run());
        expect_count(point_index(), flagged ? frame - 4u : confirmation_count);
        EXPECT_EQ(result_.features[point_index()][0u], flagged ? .5625f : .5f);
        expect_count(point_index(62u), confirmation_count);
        expect_count(point_index(85u), confirmation_count);
        EXPECT_EQ(result_.features[point_index(85u)][0u], .9375f);
        expect_evidence_exact_copy();
      }
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, CurrentResetIsLocalToFlaggedCoreInsideOverlappingExpandedCovers) {
    const pixel_t left_rgb {.125f, .75f, .5f, 1.f}, right_rgb {.3125f, .25f, .9375f, 1.f};
    auto left = field_cover_, right = field_cover_;
    left[2u] = source_x_edge(105u);
    right[0u] = source_x_edge(70u); // Expanded covers overlap well beyond the changed core.
    auto left_core = left, right_core = right;
    left_core[0u] = source_x_edge(40u);
    left_core[2u] = source_x_edge(90u);
    right_core[0u] = source_x_edge(100u);
    right_core[2u] = source_x_edge(130u);
    clear_scene(.5f); // Nearby background references mature alongside each tested coordinate.
    pixels_[point_index(60u)] = pixels_[point_index(85u)] = pixels_[point_index(95u)] = left_rgb;
    pixels_[point_index(110u)] = right_rgb;
    for (std::uint32_t frame = 1u; frame <= 8u; ++frame) {
      if (frame == 5u) {
        // All three own anchors mismatch. Matching mature neighbors exist within2px, so the
        // unflagged shared margin can borrow while the two flagged core coordinates cannot.
        for (const auto x : {60u, 85u, 95u}) paint_point(x, .5f);
      }
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      current_covers(frame == 6u ? std::vector<uint4_t> {right, left} : std::vector<uint4_t> {left, right});
      current_cores(frame == 6u ? std::vector<uint4_t> {right_core, left_core} : std::vector<uint4_t> {left_core, right_core});
      if (frame == 5u) plane_words_[v2::subtitle_plane_history_reset_mask_word] = 1u;
      ASSERT_TRUE(run());
      const auto left_count = frame <= 4u ? frame : frame - 4u;
      expect_count(point_index(60u), left_count);
      expect_count(point_index(85u), left_count);
      // Shared expanded-cover margins retain ordinary fallback despite an RGB mismatch.
      expect_count(point_index(95u), std::min(frame, confirmation_count));
      expect_count(point_index(110u), std::min(frame, confirmation_count));
      EXPECT_EQ(result_.features[point_index(110u)][0u], right_rgb[0u]);
      expect_evidence_exact_copy();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, FlaggedCurrentCoreUnionResetsSharedPixelsWithoutExtendingExpandedCovers) {
    for (const auto reset_mask : {1u, 2u, 3u}) {
      SCOPED_TRACE(reset_mask);
      ASSERT_TRUE(resize(160u, 90u));
      clear_scene(.5f);
      for (const auto x : {60u, 85u, 110u, 130u}) paint_point(x, .9375f);
      ASSERT_TRUE(mature_observations());
      auto left = field_cover_, right = field_cover_;
      left[0u] = source_x_edge(40u);
      left[2u] = source_x_edge(90u);
      right[0u] = source_x_edge(70u);
      right[2u] = source_x_edge(120u);
      for (std::uint32_t frame = 5u; frame <= 8u; ++frame) {
        // Own RGB mismatches; unchanged adjacent .5 references can preconfirm an unflagged
        // coordinate. This makes the raw-core union assertions exercise the fallback veto.
        if (frame == 5u) for (const auto x : {60u, 85u, 110u, 130u}) paint_point(x, .5f);
        make_authority(frame, 1000001u + (frame - 1u) * 100000u);
        current_covers({field_cover_, field_cover_});
        current_cores(frame == 6u ? std::vector<uint4_t> {right, left} : std::vector<uint4_t> {left, right});
        if (frame == 5u) plane_words_[v2::subtitle_plane_history_reset_mask_word] = reset_mask;
        ASSERT_TRUE(run());
        expect_count(point_index(60u), (reset_mask & 1u) != 0u ? frame - 4u : confirmation_count);
        expect_count(point_index(85u), frame - 4u); // Inside both raw/cores, so either bit resets.
        expect_count(point_index(110u), (reset_mask & 2u) != 0u ? frame - 4u : confirmation_count);
        expect_count(point_index(130u), confirmation_count); // Expanded covers only: fallback allowed.
        expect_evidence_exact_copy();
      }
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, CurrentCoreResetUsesHalfOpenNativeCentersAcrossPaddedAwkwardCrops) {
    // 161x91 gives the rounded 27-row bottom crop. The first padded content transform has
    // exact center/edge equalities; the second has fractional centers in both dimensions.
    const std::array contents {uint4_t {5u, 9u, 649u, 373u}, uint4_t {3u, 7u, 766u, 426u}};
    const std::array cores {uint4_t {247u, 307u, 407u, 335u}, uint4_t {290u, 350u, 479u, 382u}};
    for (std::size_t domain = 0u; domain < contents.size(); ++domain) {
      SCOPED_TRACE(domain);
      ASSERT_TRUE(resize(161u, 91u));
      content_ = contents[domain];
      clear_scene(.5f);
      const std::array points {std::array {59u, 10u}, std::array {60u, 10u}, std::array {61u, 10u},
        std::array {99u, 16u}, std::array {100u, 16u}, std::array {61u, 9u}, std::array {99u, 17u}};
      for (const auto &p : points) pixels_[native_index(p[0u], p[1u])] = {.9375f, .9375f, .9375f, 1.f};
      ASSERT_TRUE(mature_observations());
      for (std::uint32_t frame = 5u; frame <= 8u; ++frame) {
        // Each boundary coordinate mismatches its own anchor but has mature .5 neighbors.
        // Outside the flagged half-open core the otherwise unchanged fallback must succeed.
        if (frame == 5u) for (const auto &p : points) pixels_[native_index(p[0u], p[1u])] = {.5f, .5f, .5f, 1.f};
        make_authority(frame, 1000001u + (frame - 1u) * 100000u);
        current_cores({cores[domain]});
        if (frame == 5u) plane_words_[v2::subtitle_plane_history_reset_mask_word] = 1u;
        ASSERT_TRUE(run());
        // In domain0, native center(60,10) equals the left/top raw edge and is included.
        // Centers x100 and y17 equal the right/bottom edges and are excluded. Domain1 moves
        // the first included x to61 through the fractional transform without rounding authority.
        expect_count(native_index(59u, 10u), confirmation_count);
        expect_count(native_index(60u, 10u), domain == 0u ? frame - 4u : confirmation_count);
        expect_count(native_index(61u, 10u), frame - 4u);
        expect_count(native_index(99u, 16u), frame - 4u);
        expect_count(native_index(100u, 16u), confirmation_count);
        expect_count(native_index(61u, 9u), confirmation_count);
        expect_count(native_index(99u, 17u), confirmation_count);
        expect_evidence_exact_copy();
      }
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, MalformedFlaggedCurrentCoreCannotUseMatureExpandedCoverHistory) {
    for (const auto fault : {0u, 1u, 2u, 3u, 4u, 5u}) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      ASSERT_TRUE(mature_observations());
      auto core = field_cover_;
      core[0u] = source_x_edge(50u);
      core[2u] = source_x_edge(70u);
      make_authority(5u, 1400001u);
      current_cores({core});
      plane_words_[v2::subtitle_plane_history_reset_mask_word] = 1u;
      const auto base = v2::subtitle_ocr_raw_box_offset;
      switch (fault) {
        case 0u: ocr_words_[base] = 0u; break; // Raw rectangle extends outside its paired cover.
        case 1u: ocr_words_[base + 2u] = core[0u]; break; // Empty current raw/core.
        case 2u: ocr_words_[base + 1u] = constants_.field[2u] - 1u; break;
        case 3u: ocr_words_[base + 3u] = constants_.field[3u] + 1u; break;
        case 4u: ocr_words_[base + 4u] ^= 1u; break; // Score differs from current paired cover.
        case 5u: ocr_words_[5u] ^= 1u; break; // Stale current OCR identity.
      }
      ASSERT_TRUE(run());
      expect_cleared_snapshot();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, ChangedCoreBoundaryRgbRetainsExistingMatureMarginJitterCorrespondence) {
    clear_scene(.125f);
    paint_point(60u, .9375f);
    ASSERT_TRUE(mature_observations());
    auto core = field_cover_;
    core[0u] = source_x_edge(60u);
    core[2u] = source_x_edge(61u);
    const auto y = body_y() + 5u;
    const auto content_height = content_[3u] - content_[1u];
    core[1u] = content_[1u] + std::uint32_t(std::uint64_t(crop_top_ + y) * content_height / source_h_);
    core[3u] = content_[1u] + std::uint32_t(std::uint64_t(crop_top_ + y + 1u) * content_height / source_h_);
    paint_point(60u, .75f); // A true same-pixel mismatch from the mature .9375 fixed anchor.
    make_authority(5u, 1400001u);
    current_cores({core});
    plane_words_[v2::subtitle_plane_history_reset_mask_word] = 1u;
    ASSERT_TRUE(run());
    expect_count(point_index(60u), 1u);
    expect_count(point_index(59u), confirmation_count);
    // This next genuine observation changes RGB beyond the reset pixel's fixed-anchor bound.
    // The unchanged2px fallback can choose an unreset mature margin anchor. Four observations
    // after a flagged mismatch are not guaranteed for arbitrary later matched references.
    paint_point(60u, .125f);
    make_authority(6u, 1500001u);
    current_cores({core});
    ASSERT_TRUE(run());
    expect_count(point_index(60u), confirmation_count);
    EXPECT_EQ(result_.features[point_index(60u)][0u], .125f);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, ResetUsesOnlyCurrentCoreAcrossExpansionShrinkAndReplacement) {
    const auto left = source_x_edge(40u), middle = source_x_edge(85u), right = source_x_edge(130u);
    for (const auto shape : {0u, 1u, 2u}) {
      SCOPED_TRACE(shape);
      ASSERT_TRUE(resize(160u, 90u));
      clear_scene(.5f);
      paint_point(60u, .5625f); // Compatible with an already mature background color.
      auto previous = field_cover_;
      previous[0u] = left;
      previous[2u] = shape == 1u ? right : middle;
      for (std::uint32_t frame = 1u; frame <= 4u; ++frame) {
        make_authority(frame, 1000001u + (frame - 1u) * 100000u);
        current_covers({previous});
        ASSERT_TRUE(run());
      }
      auto current = field_cover_;
      current[0u] = shape == 2u ? middle : left;
      current[2u] = shape == 0u ? right : (shape == 1u ? middle : right);
      paint_point(60u, .75f); // Genuine RGB mismatch, rather than flag-only invalidation.
      for (std::uint32_t frame = 5u; frame <= 8u; ++frame) {
        make_authority(frame, 1000001u + (frame - 1u) * 100000u);
        current_covers({current});
        if (frame == 5u) plane_words_[v2::subtitle_plane_history_reset_mask_word] = 1u;
        ASSERT_TRUE(run());
        expect_count(point_index(60u), shape == 2u ? 0u : frame - 4u);
        expect_count(point_index(110u), shape == 1u ? 0u : frame - 4u);
      }
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, DuplicateResetCannotSpendAConfirmationAndZeroDispatchHoldsTuple) {
    paint_point();
    paint_point(110u, .5f);
    ASSERT_TRUE(mature_observations());
    auto core = field_cover_;
    core[0u] = source_x_edge(50u);
    core[2u] = source_x_edge(70u);
    paint_point(60u, .5625f); // Genuine mismatch starts count1 before the duplicate/zero hold.
    make_authority(5u, 1400001u);
    current_cores({core});
    plane_words_[v2::subtitle_plane_history_reset_mask_word] = 1u;
    ASSERT_TRUE(run());
    expect_count(point_index(), 1u);
    expect_count(point_index(110u), confirmation_count);
    const auto reset_tuple = result_;
    ASSERT_TRUE(run()); // Same reset publication, same genuine source observation.
    expect_tuple_equal(reset_tuple);
    make_authority(99u, 9900001u);
    plane_words_[v2::subtitle_plane_history_reset_mask_word] = 1u;
    ASSERT_TRUE(run(true));
    expect_tuple_equal(reset_tuple);
    for (std::uint32_t frame = 6u; frame <= 8u; ++frame) {
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame - 4u);
      expect_count(point_index(110u), confirmation_count);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, ForgedResetBitsOutsideCurrentSlotsCannotAuthorizeSupport) {
    for (const auto forged : {2u, 256u, 0xffffffffu}) {
      SCOPED_TRACE(forged);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      make_authority(4u, 1300001u);
      plane_words_[v2::subtitle_plane_history_reset_mask_word] = forged;
      ASSERT_TRUE(run());
      expect_no_mask();
      expect_count(point_index(), 0u);
      EXPECT_EQ(result_.identity[2u], 0u);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, NewCoverPixelCannotBorrowAdjacentConfirmedReference) {
    auto small = field_cover_;
    small[2u] = source_x_edge(62u); // Pixel62 is outside; pixel60 is mature only two pixels away.
    for (std::uint32_t frame = 1u; frame <= 8u; ++frame) {
      clear_scene(.125f * float(frame - 1u));
      paint_point(60u, .9375f);
      paint_point(62u, .9375f);
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      current_covers({frame <= 4u ? small : field_cover_});
      ASSERT_TRUE(run());
      expect_count(point_index(60u), std::min(frame, confirmation_count));
      expect_count(point_index(62u), frame <= 4u ? 0u : frame - 4u);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, AllEightCurrentOcrCoversAuthorizeSmallIndependentText) {
    std::vector<uint4_t> covers;
    for (std::uint32_t slot = 0u; slot < v2::subtitle_plane_cover_capacity; ++slot) {
      auto rect = field_cover_;
      rect[0u] = source_x_edge(20u + 15u * slot);
      rect[2u] = source_x_edge(35u + 15u * slot);
      covers.push_back(rect);
    }
    for (std::uint32_t frame = 1u; frame <= confirmation_count; ++frame) {
      clear_scene(.125f * float(frame - 1u));
      for (std::uint32_t slot = 0u; slot < covers.size(); ++slot) paint_point(27u + 15u * slot, .9375f);
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      current_covers(covers);
      ASSERT_TRUE(run());
      for (std::uint32_t slot = 0u; slot < covers.size(); ++slot) expect_count(point_index(27u + 15u * slot), frame);
      std::reverse(covers.begin(), covers.end());
    }
    EXPECT_EQ(pin_count(result_.support), 8u);
    expect_projection_oracle();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, CurrentRibbonKindsDoNotResetStillCoveredFixedRgbReference) {
    paint_point();
    ASSERT_TRUE(mature_observations(1u, 3u));
    make_authority(4u, 1300001u);
    const uint4_t ribbon {content_[0u], constants_.field[2u], content_[2u], content_[3u]};
    current_covers({ribbon}, true);
    ASSERT_TRUE(run());
    expect_count(point_index(), 4u);
    ASSERT_TRUE(observe(5u));
    expect_count(point_index(), 4u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, CurrentRectangleJitterKeepsFixedRgbStreak) {
    paint_point();
    for (std::uint32_t frame = 1u; frame <= confirmation_count; ++frame) {
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      auto rect = field_cover_;
      rect[0u] += frame & 1u;
      rect[2u] += frame & 1u;
      current_covers({rect});
      ASSERT_TRUE(run());
      expect_count(point_index(), frame);
      const auto held = result_;
      ASSERT_TRUE(run());
      expect_tuple_equal(held);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, BothSourceFrameAndClockMustAdvanceBeforeCountFourAcrossCutsAndDomains) {
    for (std::uint32_t fault = 0u; fault < 10u; ++fault) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      switch (fault) {
        case 0u:
          make_authority(3u, 1200001u);
          break;  // Exact duplicate holds count3.
        case 1u:
          make_authority(4u, 1200001u);
          break;
        case 2u:
          make_authority(3u, 1300001u);
          break;
        case 3u:
          make_authority(4u, 1100001u);
          break;
        case 4u:
          make_authority(4u, 1300001u, false, 7u, 4u);
          break;
        case 5u:
          make_authority(4u, 1300001u, false, 8u, 3u);
          break;
        case 6u:
          make_authority(4u, 1300001u, true);
          break;
        default:
          {
            make_authority(4u, 1300001u);
            auto prior = result_.identity;
            if (fault == 7u) {
              prior[4u] = 5u;
            }
            if (fault == 8u) {
              prior[8u] += 1u;
            }
            if (fault == 9u) {
              prior[snapshot_valid_word] = 0u;
            }
            replace_prior_identity(prior);
            break;
          }
      }
      ASSERT_TRUE(run());
      expect_count(point_index(), fault == 0u ? 3u : 1u);
      expect_no_mask();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, FourObservationSourceFrameAndTimestampCarryUseRealUint64Identity) {
    const std::array<std::uint64_t, 4u> frames {0xfffffffeull, 0xffffffffull, 0x100000000ull, 0x100000001ull};
    const std::array<std::uint64_t, 4u> times {0xffffffe0ull, 0xfffffff0ull, 0x100000000ull, 0x100000010ull};
    paint_point();
    for (std::size_t frame = 0u; frame < frames.size(); ++frame) {
      make_authority(frames[frame], times[frame]);
      ASSERT_TRUE(run());
      expect_count(point_index(), std::uint32_t(frame + 1u));
    }
    EXPECT_EQ(result_.identity[4u], 1u);
    EXPECT_EQ(result_.identity[5u], 1u);
    EXPECT_EQ(result_.identity[20u], 0x10u);
    EXPECT_EQ(result_.identity[21u], 1u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, InvalidPriorBlackCannotContributeToFourRealBlackObservations) {
    clear_scene(.5f);
    paint_point(60u, 0.f);
    pixels_[point_index()][3u] = 0.f;
    ASSERT_TRUE(observe(1u));
    EXPECT_EQ(result_.features[point_index()], (pixel_t {}));
    for (std::uint32_t frame = 2u; frame <= 5u; ++frame) {
      paint_point(60u, 0.f);
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame - 1u);
      EXPECT_EQ(result_.features[point_index()][0u], 0.f);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, InvalidCurrentAlphaRgbOrRangeBreaksAThreeObservationStreak) {
    for (std::uint32_t fault = 0u; fault < 7u; ++fault) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      make_authority(4u, 1300001u);
      auto &pixel = pixels_[point_index()];
      switch (fault) {
        case 0u:
          pixel[3u] = 0.f;
          break;
        case 1u:
          pixel[3u] = .5f;
          break;
        case 2u:
          pixel[0u] = std::numeric_limits<float>::quiet_NaN();
          break;
        case 3u:
          pixel[1u] = std::numeric_limits<float>::infinity();
          break;
        case 4u:
          pixel[2u] = -.125f;
          break;
        case 5u:
          pixel[0u] = 1.125f;
          break;
        case 6u:
          pixel[3u] = std::numeric_limits<float>::quiet_NaN();
          break;
      }
      ASSERT_TRUE(run());
      EXPECT_EQ(result_.support[point_index()], 0u);
      EXPECT_EQ(result_.features[point_index()], (pixel_t {}));
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, MalformedPriorRgbAndCountNamespaceCannotForgeFourthObservation) {
    for (std::uint32_t fault = 0u; fault < 15u; ++fault) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      auto prior = result_.features;
      pixel_t invalid {.75f, .75f, .75f, feature_code(3u)};
      switch (fault) {
        case 0u:
          invalid[0u] = std::numeric_limits<float>::quiet_NaN();
          break;
        case 1u:
          invalid[1u] = std::numeric_limits<float>::infinity();
          break;
        case 2u:
          invalid[2u] = -.125f;
          break;
        case 3u:
          invalid[0u] = 1.125f;
          break;
        case 4u:
          invalid[3u] = 0.f;
          break;
        case 5u:
          invalid[3u] = 1.f;
          break;
        case 6u:
          invalid[3u] = 3.f;
          break;  // Old SCT2 mask has no confirmation count.
        case 7u:
          invalid[3u] = 7.f;
          break;  // Mask bit at count1.
        case 8u:
          invalid[3u] = 18.f;
          break;  // Count4 without required mask bit.
        case 9u:
          invalid[3u] = 22.f;
          break;  // Count5 is outside bounded namespace.
        case 10u:
          invalid[3u] = 14.5f;
          break;
        case 11u:
          invalid[3u] = -1.f;
          break;
        case 12u:
          invalid[3u] = std::numeric_limits<float>::quiet_NaN();
          break;
        case 13u:
          invalid[3u] = std::numeric_limits<float>::infinity();
          break;
        case 14u:
          invalid[3u] = valid_rgb;
          break;  // Valid RGB but zero count cannot be borrowed.
      }
      for (auto y = body_y() + 3u; y <= body_y() + 7u; ++y) {
        for (auto x = 58u; x <= 62u; ++x) {
          prior[native_index(x, y)] = invalid;
        }
      }
      replace_prior_features(prior);
      make_authority(4u, 1300001u);
      ASSERT_TRUE(run());
      expect_count(point_index(), 1u);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, AllFourProbabilityContributorsMustBeValidAtTheFourthObservation) {
    const auto x = 60u, y = body_y() + 5u;
    const double px = (double(x) + .5) * v2::subtitle_ocr_output_width / source_w_ - .5;
    const double py = (double(y) + .5) * v2::subtitle_ocr_output_height / crop_h_ - .5;
    const auto x0 = std::uint32_t(std::floor(px)), y0 = std::uint32_t(std::floor(py));
    const std::array contributors {std::size_t(y0) * v2::subtitle_ocr_output_width + x0, std::size_t(y0) * v2::subtitle_ocr_output_width + x0 + 1u, std::size_t(y0 + 1u) * v2::subtitle_ocr_output_width + x0, std::size_t(y0 + 1u) * v2::subtitle_ocr_output_width + x0 + 1u};
    const std::array faults {std::bit_cast<float>(0x7fc12345u), std::numeric_limits<float>::infinity(), -.125f, 1.125f};
    for (std::size_t contributor = 0u; contributor < contributors.size(); ++contributor) {
      for (auto invalid : faults) {
        SCOPED_TRACE(contributor);
        ASSERT_TRUE(resize(160u, 90u));
        paint_point();
        ASSERT_TRUE(mature_observations(1u, 3u));
        make_authority(4u, 1300001u);
        std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
        probability_values_[contributors[contributor]] = invalid;
        ASSERT_TRUE(run());
        expect_count(point_index(), 0u);
        expect_evidence_exact_copy();  // Raw invalid bits remain diagnostic evidence, not alpha.
      }
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, MalformedHeadersStopConfirmationWhilePlaneFaultKeepsRgbSnapshot) {
    for (const auto packed : {0u, 1u, 2u, 3u, 4u, 8u, 9u, 10u, 11u, 12u}) {
      SCOPED_TRACE(packed);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      for (std::uint32_t frame = 1u; frame <= confirmation_count; ++frame) {
        make_authority(frame, 1000001u + (frame - 1u) * 100000u);
        plane_words_[v2::subtitle_plane_warmup_word] = packed;
        ASSERT_TRUE(run());
        EXPECT_EQ(result_.identity[2u], 1u);
        EXPECT_EQ(result_.identity[snapshot_valid_word], 1u);
        expect_count(point_index(), frame);
      }
    }
    for (std::uint32_t fault = 0u; fault < 18u; ++fault) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      make_authority(4u, 1300001u);
      switch (fault) {
        case 0u:
          ocr_words_[5u] ^= 1u;
          break;
        case 1u:
          plane_words_[v2::subtitle_plane_frame_offset] ^= 1u;
          break;
        case 2u:
          ocr_words_[9u] += 1u;
          break;
        case 3u:
          plane_words_[v2::subtitle_plane_field_extent_offset] += 1u;
          break;
        case 4u:
          params_words_[4u] = 1u;
          break;
        case 5u:
          cut_words_[sbs_adaptive_state::index(sbs_adaptive_state::word_e::cut_contract_tag_bits)] ^= 1u;
          break;
        case 6u:
          plane_words_[v2::subtitle_plane_cut_epoch_word] += 1u;
          break;
        case 7u:
          constants_.options[3u] = 1u;
          break;
        case 8u:
          ocr_words_[v2::subtitle_ocr_final_box_offset] += 1u;
          break;
        case 9u:
          ocr_words_[2u] = 0u;
          break;
        case 10u:
          plane_words_[v2::subtitle_plane_reserved_offset] = 1u; // Reserved header cannot hide behind current-valid.
          break;
        case 11u:
          plane_words_[v2::subtitle_plane_warmup_word] = 16u; // Unknown packed countdown bits grant no authority.
          break;
        default:
          // Bits are known, but remaining5/6/7 are outside the fixed four-observation window.
          plane_words_[v2::subtitle_plane_warmup_word] = (5u + (fault - 12u) % 3u) |
            (fault >= 15u ? v2::subtitle_plane_warmup_cue_bit : 0u);
          break;
      }
      ASSERT_TRUE(run());
      expect_no_mask();
      EXPECT_EQ(result_.identity[2u], 0u);
      if (fault == 1u || fault == 3u || fault == 4u || fault == 6u || fault == 10u || fault >= 11u) {
        EXPECT_EQ(result_.identity[snapshot_valid_word], 1u);
        expect_count(point_index(), 0u);
      } else {
        expect_cleared_snapshot();
      }
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, HardCutThenReplacementCueRearmsThroughFourthQuietObservationHandover) {
    // One finite source pixel isolates actual RGB confirmation from stable background seeds.
    // These are authored source/OCR inputs; all countdown, count, q and M outputs below come
    // from the production locator, temporal, history, distance and condition shader entries.
    ASSERT_TRUE(resize(160u, 90u));
    content_ = {0u, 0u, source_w_, source_h_}; // 1:1 content nodes expose exact q=0 at the source pixel.
    for (auto &pixel : pixels_) pixel[3u] = 0.f;
    paint_point(60u, .75f);
    constexpr UINT both = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    ASSERT_TRUE(create_buffer(device_.Get(), v2::subtitle_plane_state_word_count * 4u, 4u, both, plane_));
    ASSERT_TRUE(create_buffer(device_.Get(), v2::subtitle_condition_param_word_count * 4u, 4u, both, params_));
    texture_t base, horizontal_q, native_q, proposal, weight;
    ASSERT_TRUE(create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_SHADER_RESOURCE, base));
    ASSERT_TRUE(create_texture(device_.Get(), field_w_, crop_h_ + 4u, DXGI_FORMAT_R32_FLOAT, both, horizontal_q));
    ASSERT_TRUE(create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, both, native_q));
    ASSERT_TRUE(create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, both, proposal));
    ASSERT_TRUE(create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, both, weight));
    const std::vector<float> base_values(std::size_t(field_w_) * field_h_, .03f);
    context_->UpdateSubresource(base.texture.Get(), 0u, nullptr, base_values.data(), field_w_ * sizeof(float), 0u);
    buffer_t control;
    ASSERT_TRUE(create_buffer(device_.Get(), 4u, 4u, both, control));
    const auto make_cb = [&](UINT bytes, ComPtr<ID3D11Buffer> &buffer) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = bytes;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      return SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &buffer));
    };
    ComPtr<ID3D11Buffer> depth_cb, locator_cb, distance_cb;
    ASSERT_TRUE(make_cb(64u, depth_cb));
    ASSERT_TRUE(make_cb(80u, locator_cb));
    ASSERT_TRUE(make_cb(48u, distance_cb));
    namespace cache = models::host_sbs_shader_cache;
    const std::array specs {
      cache::shader_spec {"host_sbs_subtitle_locator_cs.hlsl", "resolve_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_subtitle_contour_distance_cs.hlsl", "horizontal_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_subtitle_contour_distance_cs.hlsl", "vertical_main", "cs_5_0"},
      cache::shader_spec {"host_sbs_subtitle_locator_cs.hlsl", "condition_main", "cs_5_0"}
    };
    const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, specs);
    std::array<ComPtr<ID3D11ComputeShader>, 4u> pipeline;
    for (std::size_t pass = 0u; pass < specs.size(); ++pass) {
      const auto bytecode = cache::get(sources, specs[pass]);
      ASSERT_TRUE(bytecode && !bytecode->empty()) << specs[pass].entrypoint;
      ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &pipeline[pass])));
    }
    std::vector<std::uint32_t> prior_plane(v2::subtitle_plane_state_word_count, 0u);
    std::vector<float> weights, distances, proposals;
    const auto point_y = crop_top_ + body_y() + 5u;
    const auto point_node = std::size_t(point_y) * field_w_ + 60u;
    const auto distant_node = std::size_t(point_y) * field_w_ + 100u;
    for (std::uint32_t frame = 1u; frame <= 10u; ++frame) {
      SCOPED_TRACE(frame);
      // Epoch2 old text is mature before a valid epoch3 cut. No input-domain reset is used.
      const auto epoch = frame < 5u ? 2u : 3u;
      if (frame == 6u) paint_point(60u, .5f);
      make_authority(frame, 1000001u + (frame - 1u) * 100000u, false, 7u, epoch);
      std::fill(probability_values_.begin(), probability_values_.end(), 1.f);
      if (frame >= 6u) {
        for (std::uint32_t y = 0u; y < v2::subtitle_ocr_output_height; ++y)
          for (std::uint32_t x = 0u; x < v2::subtitle_ocr_output_width / 2u; ++x)
            probability_values_[std::size_t(y) * v2::subtitle_ocr_output_width + x] = 0.f;
      }
      ASSERT_TRUE(upload());
      context_->UpdateSubresource(plane_.buffer.Get(), 0u, nullptr, prior_plane.data(), 0u, 0u);
      std::array<std::uint32_t, 16u> depth_words {};
      depth_words[0u] = field_w_; depth_words[1u] = field_h_;
      std::copy(content_.begin(), content_.end(), depth_words.begin() + 9u);
      const std::array locator_words {constants_.field, uint4_t {source_w_, source_h_, 1u, 0u},
        constants_.frame, content_, uint4_t {constants_.observation[0u], constants_.observation[1u], 0u, 0u}};
      context_->UpdateSubresource(depth_cb.Get(), 0u, nullptr, depth_words.data(), 0u, 0u);
      context_->UpdateSubresource(locator_cb.Get(), 0u, nullptr, locator_words.data(), 0u, 0u);
      const std::array<ID3D11Buffer *, 3u> locator_cbs {depth_cb.Get(), geometry_cb_.Get(), locator_cb.Get()};
      context_->CSSetConstantBuffers(0u, locator_cbs.size(), locator_cbs.data());
      const std::array<ID3D11ShaderResourceView *, 10u> resolve_inputs {
        geometry_.srv.Get(), cut_.srv.Get(), base.srv.Get(), nullptr, nullptr, nullptr,
        identity_.srv.Get(), ocr_.srv.Get(), probability_.srv.Get(), completed_probability_.srv.Get()
      };
      const std::array<ID3D11UnorderedAccessView *, 3u> resolve_outputs {plane_.uav.Get(), nullptr, params_.uav.Get()};
      context_->CSSetShaderResources(0u, resolve_inputs.size(), resolve_inputs.data());
      context_->CSSetUnorderedAccessViews(2u, resolve_outputs.size(), resolve_outputs.data(), nullptr);
      context_->CSSetShader(pipeline[0u].Get(), nullptr, 0u);
      context_->Dispatch(1u, 1u, 1u);
      unbind();
      ASSERT_TRUE(read_buffer(device_.Get(), context_.Get(), plane_.buffer.Get(), plane_words_));
      ASSERT_TRUE(read_buffer(device_.Get(), context_.Get(), params_.buffer.Get(), params_words_));
      prior_plane = plane_words_;
      ASSERT_EQ(params_words_[4u], v2::subtitle_condition_param_fully_active_value);
      ASSERT_EQ(plane_words_[v2::subtitle_plane_current_slot_mask_word], 1u);
      ASSERT_TRUE(run()); // The actual temporal/history/snapshot entries consume the resolved plane.
      const auto expected_count = frame < 5u ? std::min(frame, confirmation_count) :
        frame == 5u ? 1u : std::min(frame - 5u, confirmation_count);
      expect_count(point_index(), expected_count);
      const auto remaining = plane_words_[v2::subtitle_plane_warmup_word] & v2::subtitle_plane_warmup_remaining_mask;
      if (frame >= 5u) {
        EXPECT_EQ(remaining, frame == 5u ? 4u : 10u - frame);
        EXPECT_EQ(plane_words_[v2::subtitle_plane_history_reset_mask_word], frame == 6u ? 1u : 0u);
      }
      const std::array distance_words {constants_.source, uint4_t {field_w_, field_h_, 0u, 0u}, content_};
      context_->UpdateSubresource(distance_cb.Get(), 0u, nullptr, distance_words.data(), 0u, 0u);
      context_->CSSetConstantBuffers(0u, 1u, distance_cb.GetAddressOf());
      const auto grid_width = (field_w_ + 15u) / 16u;
      context_->CSSetShaderResources(0u, 1u, support_.srv.GetAddressOf());
      context_->CSSetUnorderedAccessViews(0u, 1u, horizontal_q.uav.GetAddressOf(), nullptr);
      context_->CSSetShader(pipeline[1u].Get(), nullptr, 0u);
      context_->Dispatch(grid_width, (crop_h_ + 4u + grid_width - 1u) / grid_width, 1u);
      unbind();
      context_->CSSetShaderResources(1u, 1u, horizontal_q.srv.GetAddressOf());
      context_->CSSetUnorderedAccessViews(0u, 1u, native_q.uav.GetAddressOf(), nullptr);
      context_->CSSetShader(pipeline[2u].Get(), nullptr, 0u);
      context_->Dispatch(grid_width, (field_w_ + grid_width - 1u) / grid_width, 1u);
      unbind();
      context_->CSSetConstantBuffers(0u, locator_cbs.size(), locator_cbs.data());
      const std::array<ID3D11ShaderResourceView *, 8u> condition_inputs {
        geometry_.srv.Get(), cut_.srv.Get(), base.srv.Get(), plane_.srv.Get(), params_.srv.Get(),
        native_q.srv.Get(), identity_.srv.Get(), ocr_.srv.Get()
      };
      const std::array<ID3D11UnorderedAccessView *, 3u> condition_outputs {proposal.uav.Get(), control.uav.Get(), weight.uav.Get()};
      context_->CSSetShaderResources(0u, condition_inputs.size(), condition_inputs.data());
      context_->CSSetUnorderedAccessViews(3u, condition_outputs.size(), condition_outputs.data(), nullptr);
      context_->CSSetShader(pipeline[3u].Get(), nullptr, 0u);
      context_->Dispatch(grid_width, (field_h_ + 15u) / 16u, 1u);
      unbind();
      ASSERT_TRUE(read_texture(device_.Get(), context_.Get(), weight.texture.Get(), weights));
      ASSERT_TRUE(read_texture(device_.Get(), context_.Get(), native_q.texture.Get(), distances));
      ASSERT_TRUE(read_texture(device_.Get(), context_.Get(), proposal.texture.Get(), proposals));
      if (frame >= 5u) {
        if (frame < 9u) {
          EXPECT_TRUE(std::isinf(distances[point_node]));
        } else {
          EXPECT_EQ(distances[point_node], 0.f);
          EXPECT_EQ(result_.support[point_index()], 1u);
        }
        EXPECT_EQ(weights[point_node], 1.f);
        if (frame < 10u) {
          EXPECT_EQ(weights[distant_node], 1.f);
          EXPECT_EQ(std::bit_cast<std::uint32_t>(proposals[distant_node]), params_words_[5u]);
        } else {
          EXPECT_EQ(weights[distant_node], 0.f);
          EXPECT_EQ(proposals[distant_node], base_values[distant_node]);
        }
      }
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, InvalidSealedGeometryCannotPublishPriorRgbOrMask) {
    for (std::uint32_t fault = 0u; fault < 6u; ++fault) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      make_authority(4u, 1300001u);
      switch (fault) {
        case 0u:
          geometry_words_[v2::camera_center_integrity_bits] ^= 1u;
          break;
        case 1u:
          geometry_words_[v2::gain_last_observation_low] -= 1u;
          reseal_geometry();
          break;
        case 2u:
          geometry_constants_.joint_observation_timestamp_low -= 1u;
          break;
        case 3u:
          geometry_words_[v2::confirmed_cut_count] += 1u;
          break;
        case 4u:
          geometry_words_[v2::frame_valid] = 0u;
          geometry_words_[v2::renderer_authorization_bits] = 0u;
          reseal_geometry();
          break;
        case 5u:
          geometry_constants_.joint_observation_reserved0 = 1u;
          break;
      }
      ASSERT_TRUE(run());
      expect_cleared_snapshot();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, CountOneDuplicateNeedsBitExactRgbButLaterCountsCarryAnchor) {
    clear_scene(.5f);
    paint_point(60u, 0.f);
    ASSERT_TRUE(observe(1u));
    expect_count(point_index(), 1u);
    paint_point(60u, -0.f);  // Same numerical color, different source bits is not a fresh frame.
    ASSERT_TRUE(run());
    expect_count(point_index(), 0u);
    EXPECT_EQ(result_.source[point_index()][0u], float_to_half(-0.f));
    paint_point(60u, .03125f);
    ASSERT_TRUE(observe(2u));
    expect_count(point_index(), 1u);
    paint_point(60u, .0625f);  // Within tolerance, but count1 duplicate cannot replace its reference.
    ASSERT_TRUE(run());
    expect_count(point_index(), 0u);
    ASSERT_TRUE(observe(3u));
    expect_count(point_index(), 1u);
    paint_point(60u, .03125f);
    ASSERT_TRUE(observe(4u));
    expect_count(point_index(), 2u);
    const float anchor = result_.features[point_index()][0u];
    paint_point(60u, .0625f);
    ASSERT_TRUE(run());
    expect_count(point_index(), 2u);
    EXPECT_EQ(result_.features[point_index()][0u], anchor);
    expect_evidence_exact_copy();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, OffCoverRgbSnapshotsCannotSupplyPreconfirmedStreaks) {
    paint_point(5u);
    ASSERT_TRUE(mature_observations());
    expect_count(point_index(5u), 0u);
    for (std::uint32_t frame = 5u; frame <= 8u; ++frame) {
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      const uint4_t wider {content_[0u], field_cover_[1u], content_[2u], field_cover_[3u]};
      ASSERT_GE(wider[2u] - wider[0u], 2u * (wider[3u] - wider[1u]));
      current_covers({wider});
      ASSERT_TRUE(run());
      expect_count(point_index(5u), frame - 4u);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, ZeroDispatchReuseHoldsPublishedTupleAndStagedAnchorBits) {
    paint_point();
    ASSERT_TRUE(mature_observations());
    const auto held = result_;
    ASSERT_GT(pin_count(held.support), 0u);
    make_authority(99u, 9900001u, true, 123u, 55u);
    clear_scene(.95f);
    geometry_words_[v2::camera_center_integrity_bits] ^= 1u;
    ASSERT_TRUE(run(true));
    expect_tuple_equal(held);
    EXPECT_EQ(result_.next_features, held.next_features);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, LargeNativeFlatGlyphBodyConfirmsCompletelyAcrossTwoPixelJitter) {
    ASSERT_TRUE(resize(2536u, 1427u));
    ASSERT_GE(field_cover_[2u] - field_cover_[0u], 2u * (field_cover_[3u] - field_cover_[1u]));
    const auto top = native_cover_[1u] + 50u;
    const pixel_t original {.3125f, .75f, .5625f, 1.f};
    for (std::uint32_t frame = 1u; frame <= 3u; ++frame) {
      clear_scene(.125f * float(frame - 1u));
      paint({1000u, top, 1100u, top + 60u}, original);
      ASSERT_TRUE(observe(frame));
      expect_flat_patch({1000u, top, 1100u, top + 60u}, frame, original);
      expect_no_mask();
    }
    clear_scene(.375f);
    paint({1002u, top + 2u, 1102u, top + 62u}, {.28125f, .71875f, .53125f, 1.f});
    ASSERT_TRUE(observe(4u));
    expect_flat_patch({1002u, top + 2u, 1102u, top + 62u}, 4u, original);
    EXPECT_EQ(pin_count(result_.support), 6000u) << "complete current flat body without changing-background support";
    const auto old_left = native_index(1000u, top + 32u);
    expect_count(old_left, 1u);
    EXPECT_EQ(result_.features[old_left], (pixel_t {.375f, .375f, .375f, feature_code(1u)}));
    for (auto value : result_.support) {
      ASSERT_LE(value, 1u) << "every grid-stride tail cell overwrites poison";
    }
    EXPECT_GT(pin_count(result_.support), 0u);
    expect_projection_oracle();
    expect_evidence_exact_copy();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, PaddedFieldProjectionPreservesBinaryPinsAndWritesZeroPadding) {
    content_ = {7u, 9u, field_w_ - 11u, field_h_ - 13u};
    paint_point();
    ASSERT_TRUE(mature_observations());
    expect_count(point_index(), 4u);
    expect_projection_oracle();
    for (std::uint32_t y = 0u; y < field_h_; ++y) {
      for (std::uint32_t x = 0u; x < field_w_; ++x) {
        if (x < content_[0u] || x >= content_[2u] || y < content_[1u] || y >= content_[3u]) {
          EXPECT_EQ(result_.field[std::size_t(y) * field_w_ + x], 0u);
        }
      }
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, MalformedPriorIdentityCannotCompleteAThreeObservationStreak) {
    for (std::uint32_t fault = 0u; fault < 16u; ++fault) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      paint_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      auto prior = result_.identity;
      switch (fault) {
        case 0u:
          prior[0u] ^= 1u;
          break;
        case 1u:
          prior[1u] ^= 1u;
          break;
        case 2u:
          prior[2u] = 2u;
          break;
        case 3u:
          prior[3u] += 1u;
          break;
        case 4u:
          prior[6u] += 1u;
          break;
        case 5u:
          prior[12u] += 1u;
          break;
        case 6u:
          prior[14u] += 1u;
          break;
        case 7u:
          prior[16u] += 1u;
          break;
        case 8u:
          prior[20u] = 1400001u;
          break;
        case 9u:
          prior[22u] -= 1u;
          break;
        case 10u:
          prior[snapshot_valid_word] = 2u;
          break;
        case 11u:
          prior[27u] = 1u;
          break;
        case 12u:
          prior[reserved_word] = 1u;
          break;
        case 13u:
          prior[reserved_word + 1u] = 1u;
          break;
        case 14u:
          prior[reserved_word + 2u] = 1u;
          break;
        case 15u:
          prior[2u] = 0u;
          break;
      }
      replace_prior_identity(prior);
      make_authority(4u, 1300001u);
      ASSERT_TRUE(run());
      expect_count(point_index(), 1u);
      expect_no_mask();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, InvalidNarrowRibbonAtEpochZeroCannotBorrowThreeObservationHistory) {
    paint_point();
    for (std::uint32_t frame = 1u; frame <= 3u; ++frame) {
      make_authority(frame, 1000001u + (frame - 1u) * 100000u, false, 7u, 0u);
      ASSERT_TRUE(run());
      expect_count(point_index(), frame);
    }
    ASSERT_EQ(result_.identity[3u], 0u);
    ASSERT_EQ(result_.identity[snapshot_valid_word], 1u);
    make_authority(4u, 1300001u, false, 7u, 0u);
    const auto cw = content_[2u] - content_[0u];
    const uint4_t core {content_[0u] + cw / 3u, constants_.field[2u], content_[0u] + cw / 2u, constants_.field[3u]};
    const uint4_t cover {content_[0u], constants_.field[2u], content_[2u], content_[3u]};
    ASSERT_LT((core[2u] - core[0u]) * v2::subtitle_ocr_ribbon_min_width_denominator, cw * v2::subtitle_ocr_ribbon_min_width_numerator);
    params_words_[3u] = 1u;
    plane_words_[v2::subtitle_plane_adaptive_offset + 13u] = (cover[2u] - cover[0u]) * (cover[3u] - cover[1u]);
    for (auto offset : {v2::subtitle_ocr_raw_box_offset, v2::subtitle_ocr_final_box_offset}) {
      const auto &rect = offset == v2::subtitle_ocr_raw_box_offset ? core : cover;
      std::copy(rect.begin(), rect.end(), ocr_words_.begin() + offset);
      ocr_words_[offset + 5u] = v2::subtitle_ocr_box_flag_ribbon;
      ocr_words_[offset + 6u] = v2::subtitle_ocr_ribbon_min_structural_gaps + 1u;
      ocr_words_[offset + 7u] = v2::subtitle_ocr_ribbon_min_structural_gaps;
    }
    ASSERT_TRUE(run());
    expect_cleared_snapshot();
    EXPECT_EQ(result_.identity[2u], 0u);
    EXPECT_EQ(result_.identity[snapshot_valid_word], 0u);
    EXPECT_EQ(result_.identity[3u], 0u);
  }
}  // namespace
#endif
