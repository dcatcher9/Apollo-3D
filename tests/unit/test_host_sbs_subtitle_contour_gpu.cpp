/**
 * @file tests/unit/test_host_sbs_subtitle_contour_gpu.cpp
 * @brief Actual D3D11 current-evidence masks, diagnostic RGB history, and projection-area oracle.
 *
 * These tests isolate contour classification from model detection: the probability/accepted
 * records are authored native inputs. They do not replicate a stereo renderer on the CPU.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/depth_coordinate_v2.h"
  #include "src/crypto.h"
  #include "src/generated/sbs_adaptive_state_contract.h"
  #include "src/host_sbs_shader_cache.h"
  #include "src/host_sbs_subtitle_glyph.h"
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
  #include <limits>
  #include <nlohmann/json.hpp>
  #include <stdexcept>
  #include <string>
  #include <string_view>
  #include <utility>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace v2 = models::depth_coordinate_v2;
  using uint4_t = std::array<std::uint32_t, 4u>;
  using pixel_t = std::array<float, 4u>;
  using half_pixel_t = std::array<std::uint16_t, 4u>;

  std::vector<char> replay_bytes(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() < 0) throw std::runtime_error("Cannot read native fixture input: " + path.string());
    std::vector<char> bytes(static_cast<std::size_t>(input.tellg()));
    input.seekg(0);
    if (!bytes.empty()) input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("Incomplete native fixture input: " + path.string());
    return bytes;
  }

  std::string replay_sha256(const void *data, std::size_t size) {
    const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), size));
    constexpr char digits[] = "0123456789abcdef";
    std::string hex;
    for (const auto byte : digest) {
      hex.push_back(digits[byte >> 4u]);
      hex.push_back(digits[byte & 15u]);
    }
    return hex;
  }

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

  float half_to_float(std::uint16_t value) {
    const auto exponent = (value >> 10u) & 31u, mantissa = value & 1023u;
    float decoded = exponent == 0u ? std::ldexp(static_cast<float>(mantissa), -24) :
      exponent == 31u ? (mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity()) :
      std::ldexp(1.f + static_cast<float>(mantissa) / 1024.f, static_cast<int>(exponent) - 15);
    return (value & 0x8000u) != 0u ? -decoded : decoded;
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

  // Feature.w retains diagnostic RGB/count encoding; its legacy count4 bit does not own support.
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
    std::vector<std::uint32_t> support, matte, field, identity, probability_bits;
    std::vector<pixel_t> features, next_features;
    std::vector<half_pixel_t> source;
  };

  class HostSbsSubtitleContourGpuTest: public ::testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      D3D_DRIVER_TYPE driver = D3D_DRIVER_TYPE_WARP;
      const auto test_name = std::string(::testing::UnitTest::GetInstance()->current_test_info()->name());
      const auto *replay_environment = test_name == "SavedNativeContourReplay" ? "SUNSHINE_TEST_CONTOUR_REPLAY_CASES" :
        test_name == "ExperimentalBirthPaletteReplay" ? "SUNSHINE_TEST_BIRTH_PALETTE_CASES" : nullptr;
      if (replay_environment) {
        if (const auto *path = std::getenv(replay_environment)) {
          const auto bytes = replay_bytes(path);
          const auto specification = nlohmann::json::parse(bytes.begin(), bytes.end());
          const auto selected = specification.value("driver", std::string("warp"));
          ASSERT_TRUE(selected == "warp" || selected == "hardware");
          driver = selected == "hardware" ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP;
        }
      }
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, driver, nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &device_, &actual, &context_)));
      ASSERT_EQ(actual, D3D_FEATURE_LEVEL_11_0);
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {
        cache::shader_spec {"host_sbs_subtitle_contour_cs.hlsl", "temporal_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_contour_cs.hlsl", "dense_main", "cs_5_0"},
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

    bool resize(std::uint32_t width, std::uint32_t height,
                std::uint32_t field_width = 0u, std::uint32_t field_height = 0u,
                uint4_t content = {}) {
      unbind();
      source_w_ = width;
      source_h_ = height;
      crop_h_ = std::min(height, (width + 5u) / 6u);
      crop_top_ = height - crop_h_;
      field_w_ = field_width != 0u ? field_width : v2::model_calibrated_shapes[0u].width;
      field_h_ = field_height != 0u ? field_height : v2::model_calibrated_shapes[0u].height;
      content_ = content != uint4_t {} ? content : uint4_t {0u, 0u, field_w_, field_h_};
      imported_source_.clear();
      const UINT both = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      display_ = {};
      support_ = {};
      matte_ = {};
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
      if (!create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_SHADER_RESOURCE, display_) || !create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R32_UINT, both, support_) || !create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R32_UINT, both, matte_) || !create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R16G16B16A16_FLOAT, both, completed_display_) || !create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_UINT, both, field_) || !create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R32G32B32A32_FLOAT, both, features_) || !create_texture(device_.Get(), width, crop_h_, DXGI_FORMAT_R32G32B32A32_FLOAT, both, next_features_) || !create_buffer(device_.Get(), v2::subtitle_ocr_output_width * v2::subtitle_ocr_output_height * 4u, 4u, D3D11_BIND_SHADER_RESOURCE, probability_) || !create_buffer(device_.Get(), v2::subtitle_ocr_output_width * v2::subtitle_ocr_output_height * 4u, 4u, both, completed_probability_) || !create_buffer(device_.Get(), v2::subtitle_plane_state_word_count * 4u, 4u, D3D11_BIND_SHADER_RESOURCE, plane_) || !create_buffer(device_.Get(), v2::subtitle_condition_param_word_count * 4u, 4u, D3D11_BIND_SHADER_RESOURCE, params_) || !create_buffer(device_.Get(), v2::subtitle_ocr_record_word_count * 4u, 4u, D3D11_BIND_SHADER_RESOURCE, ocr_) || !create_buffer(device_.Get(), sbs_adaptive_state::word_count * 4u, 16u, D3D11_BIND_SHADER_RESOURCE, cut_) || !create_buffer(device_.Get(), sizeof(v2::state_words_t), 16u, D3D11_BIND_SHADER_RESOURCE, geometry_) || !create_buffer(device_.Get(), v2::subtitle_contour_history_word_count * 4u, 4u, both, identity_)) {
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
      std::vector<half_pixel_t> half_pixels = imported_source_;
      if (half_pixels.empty()) {
        half_pixels.resize(pixels_.size());
        for (std::size_t i = 0u; i < pixels_.size(); ++i) {
          for (std::size_t lane = 0u; lane < 4u; ++lane) {
            half_pixels[i][lane] = float_to_half(pixels_[i][lane]);
          }
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
      std::array<ID3D11ShaderResourceView *, 13u> srvs {};
      std::array<ID3D11UnorderedAccessView *, 7u> uavs {};
      context_->CSSetShaderResources(0u, srvs.size(), srvs.data());
      context_->CSSetUnorderedAccessViews(0u, uavs.size(), uavs.data(), nullptr);
    }

    void bind_inputs(ID3D11ShaderResourceView *baseline_support = nullptr,
                     ID3D11ShaderResourceView *eligibility = nullptr) {
      const std::array<ID3D11ShaderResourceView *, 13u> srvs {
        display_.srv.Get(),
        probability_.srv.Get(),
        plane_.srv.Get(),
        params_.srv.Get(),
        nullptr,
        baseline_support,
        features_.srv.Get(),
        identity_.srv.Get(),
        cut_.srv.Get(),
        ocr_.srv.Get(),
        geometry_.srv.Get(),
        nullptr,
        eligibility
      };
      context_->CSSetShaderResources(0u, srvs.size(), srvs.data());
      const std::array<ID3D11Buffer *, 2u> cbuffers {contour_cb_.Get(), geometry_cb_.Get()};
      context_->CSSetConstantBuffers(0u, cbuffers.size(), cbuffers.data());
    }

    bool run(bool zero_groups = false,
             ID3D11ShaderResourceView *baseline_support = nullptr,
             ID3D11ShaderResourceView *eligibility = nullptr) {
      unbind();
      if (!upload()) {
        return false;
      }
      if (!zero_groups) {
        const UINT poison[4] {0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
        context_->ClearUnorderedAccessViewUint(support_.uav.Get(), poison);
        context_->ClearUnorderedAccessViewUint(matte_.uav.Get(), poison);
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
      bind_inputs(baseline_support, eligibility);
      ID3D11UnorderedAccessView *null_uav = nullptr;
      context_->CSSetShader(shaders_[0u].Get(), nullptr, 0u);
      context_->CSSetUnorderedAccessViews(0u, 1u, support_.uav.GetAddressOf(), nullptr);
      context_->CSSetUnorderedAccessViews(2u, 1u, next_features_.uav.GetAddressOf(), nullptr);
      context_->DispatchIndirect(indirect_.Get(), 0u);
      context_->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);
      context_->CSSetUnorderedAccessViews(2u, 1u, &null_uav, nullptr);
      // Experimental inputs exist only during temporal_main. Dense/history/snapshot retain
      // their production bindings and never read the external eligibility bitmap.
      ID3D11ShaderResourceView *null_eligibility = nullptr;
      context_->CSSetShaderResources(12u, 1u, &null_eligibility);
      context_->CSSetShaderResources(5u, 1u, support_.srv.GetAddressOf());
      context_->CSSetShader(shaders_[1u].Get(), nullptr, 0u);
      context_->CSSetUnorderedAccessViews(6u, 1u, matte_.uav.GetAddressOf(), nullptr);
      context_->DispatchIndirect(indirect_.Get(), 0u);
      context_->CSSetUnorderedAccessViews(6u, 1u, &null_uav, nullptr);
      context_->CSSetShader(shaders_[2u].Get(), nullptr, 0u);
      context_->CSSetUnorderedAccessViews(1u, 1u, field_.uav.GetAddressOf(), nullptr);
      context_->DispatchIndirect(indirect_.Get(), 0u);
      context_->CSSetUnorderedAccessViews(1u, 1u, &null_uav, nullptr);
      // Consume previous RGB/context before current history becomes UAV output.
      const std::array<ID3D11ShaderResourceView *, 2u> null_previous {};
      context_->CSSetShaderResources(6u, null_previous.size(), null_previous.data());
      context_->CSSetShaderResources(11u, 1u, next_features_.srv.GetAddressOf());
      const std::array<ID3D11UnorderedAccessView *, 2u> history_outputs {features_.uav.Get(), identity_.uav.Get()};
      context_->CSSetShader(shaders_[3u].Get(), nullptr, 0u);
      context_->CSSetUnorderedAccessViews(2u, history_outputs.size(), history_outputs.data(), nullptr);
      context_->DispatchIndirect(indirect_.Get(), 0u);
      const std::array<ID3D11UnorderedAccessView *, 2u> null_outputs {};
      context_->CSSetUnorderedAccessViews(2u, null_outputs.size(), null_outputs.data(), nullptr);
      ID3D11ShaderResourceView *null_next = nullptr;
      context_->CSSetShaderResources(11u, 1u, &null_next);
      const std::array<ID3D11UnorderedAccessView *, 2u> evidence_outputs {completed_display_.uav.Get(), completed_probability_.uav.Get()};
      context_->CSSetShader(shaders_[4u].Get(), nullptr, 0u);
      context_->CSSetUnorderedAccessViews(4u, evidence_outputs.size(), evidence_outputs.data(), nullptr);
      context_->DispatchIndirect(indirect_.Get(), 0u);
      unbind();
      return read_texture(device_.Get(), context_.Get(), support_.texture.Get(), result_.support) &&
             read_texture(device_.Get(), context_.Get(), matte_.texture.Get(), result_.matte) &&
             read_texture(device_.Get(), context_.Get(), field_.texture.Get(), result_.field) &&
             read_texture(device_.Get(), context_.Get(), features_.texture.Get(), result_.features) &&
             read_texture(device_.Get(), context_.Get(), next_features_.texture.Get(), result_.next_features) &&
             read_texture(device_.Get(), context_.Get(), completed_display_.texture.Get(), result_.source) &&
             read_buffer(device_.Get(), context_.Get(), identity_.buffer.Get(), result_.identity) &&
             read_buffer(device_.Get(), context_.Get(), completed_probability_.buffer.Get(), result_.probability_bits);
    }


    // The opt-in replay pins its temporal variant independently. An additional explicitly
    // named dense cover-intersection diagnostic is pinned separately below.
    bool set_variant_temporal(const std::filesystem::path &root,
                              const std::string &expected_closure,
                              std::string &closure) {
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {
        cache::shader_spec {"host_sbs_subtitle_contour_cs.hlsl", "temporal_main", "cs_5_0"}
      };
      const auto sources = cache::snapshot_sources(root, specs);
      closure = cache::source_closure_sha256(sources);
      if (closure.empty() || closure != expected_closure) return false;
      const auto bytecode = cache::get(sources, specs[0u]);
      ComPtr<ID3D11ComputeShader> shader;
      if (!bytecode || bytecode->empty() ||
          FAILED(device_->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &shader))) return false;
      unbind();
      shaders_[0u] = std::move(shader);
      return true;
    }

    // An explicitly pinned dense-only diagnostic cannot replace the authenticated baseline.
    bool set_variant_dense(const std::filesystem::path &root,
                           const std::string &expected_closure,
                           std::string &closure) {
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {
        cache::shader_spec {"host_sbs_subtitle_contour_cs.hlsl", "dense_main", "cs_5_0"}
      };
      const auto sources = cache::snapshot_sources(root, specs);
      closure = cache::source_closure_sha256(sources);
      if (closure.empty() || closure != expected_closure) return false;
      const auto bytecode = cache::get(sources, specs[0u]);
      ComPtr<ID3D11ComputeShader> shader;
      if (!bytecode || bytecode->empty() ||
          FAILED(device_->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &shader))) return false;
      unbind();
      shaders_[1u] = std::move(shader);
      return true;
    }

    // Isolate actual dense compute from sparse classification with explicit current seeds.
    // Authority, FP16 source upload, field-indirect dispatch, native boundaries and poisoned
    // output clearing remain the production mechanisms.
    bool run_dense_only(const std::vector<std::uint32_t> &seeds) {
      if (seeds.size() != std::size_t(source_w_) * crop_h_) return false;
      unbind();
      if (!upload()) return false;
      context_->UpdateSubresource(support_.texture.Get(), 0u, nullptr, seeds.data(), source_w_ * sizeof(std::uint32_t), 0u);
      const UINT poison[4] {0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
      context_->ClearUnorderedAccessViewUint(matte_.uav.Get(), poison);
      const std::array<std::uint32_t, 3u> arguments {(field_w_ + 15u) / 16u, (field_h_ + 15u) / 16u, 1u};
      context_->UpdateSubresource(indirect_.Get(), 0u, nullptr, arguments.data(), 0u, 0u);
      bind_inputs();
      context_->CSSetShaderResources(5u, 1u, support_.srv.GetAddressOf());
      context_->CSSetShader(shaders_[1u].Get(), nullptr, 0u);
      context_->CSSetUnorderedAccessViews(6u, 1u, matte_.uav.GetAddressOf(), nullptr);
      context_->DispatchIndirect(indirect_.Get(), 0u);
      unbind();
      result_.support = seeds;
      return read_texture(device_.Get(), context_.Get(), matte_.texture.Get(), result_.matte) &&
        read_texture(device_.Get(), context_.Get(), display_.texture.Get(), result_.source);
    }

    void expect_dense_reference(bool intersect_current_cover = false) {
      namespace glyph = models::host_sbs_subtitle_glyph;
      ASSERT_EQ(result_.source.size(), std::size_t(source_w_) * crop_h_);
      auto expected = glyph::dense_matte(source_w_, crop_h_, result_.support, [&](std::size_t p) {
        glyph::sample_t sample;
        sample.valid = half_to_float(result_.source[p][3u]) == 1.f;
        for (std::size_t lane = 0u; lane < 3u; ++lane) sample.rgb[lane] = half_to_float(result_.source[p][lane]);
        return sample;
      });
      if (intersect_current_cover) {
        ASSERT_LE(ocr_words_[4u], v2::subtitle_plane_cover_capacity);
        for (std::size_t p = 0u; p < expected.size(); ++p) {
          const auto x = p % source_w_, y = p / source_w_ + crop_top_;
          const auto sx = 2ull * content_[0u] * source_w_ +
            (2ull * x + 1u) * (content_[2u] - content_[0u]);
          const auto sy = 2ull * content_[1u] * source_h_ +
            (2ull * y + 1u) * (content_[3u] - content_[1u]);
          bool covered = false;
          for (std::uint32_t slot = 0u; slot < ocr_words_[4u]; ++slot) {
            const auto b = v2::subtitle_ocr_final_box_offset + slot * v2::subtitle_ocr_box_word_count;
            covered |= sx >= 2ull * ocr_words_[b] * source_w_ &&
              sx < 2ull * ocr_words_[b + 2u] * source_w_ &&
              sy >= 2ull * ocr_words_[b + 1u] * source_h_ &&
              sy < 2ull * ocr_words_[b + 3u] * source_h_;
          }
          if (!covered) expected[p] = 0u;
        }
      }
      ASSERT_EQ(result_.matte.size(), expected.size());
      for (std::size_t p = 0u; p < expected.size(); ++p)
        ASSERT_EQ(result_.matte[p], expected[p]) << "native dense matte at " << p % source_w_ << ',' << p / source_w_;
    }

    struct replay_fields_t {
      std::vector<float> q, weight, blend, vertical, final, uploaded_base;
      std::vector<std::uint32_t> control;
    };

    bool run_replay_conditioner(const std::vector<float> &base_values, replay_fields_t &out) {
      if (base_values.size() != std::size_t(field_w_) * field_h_) return false;
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {
        cache::shader_spec {"host_sbs_subtitle_contour_distance_cs.hlsl", "horizontal_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_contour_distance_cs.hlsl", "vertical_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_locator_cs.hlsl", "condition_main", "cs_5_0"},
        cache::shader_spec {"depth_coordinate_v2_vertical_limit_cs.hlsl", "subtitle_main", "cs_5_0"},
        cache::shader_spec {"depth_coordinate_v2_limit_cs.hlsl", "subtitle_main", "cs_5_0"},
      };
      const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, specs);
      std::array<ComPtr<ID3D11ComputeShader>, 5u> passes;
      for (std::size_t pass = 0u; pass < passes.size(); ++pass) {
        const auto bytes = cache::get(sources, specs[pass]);
        if (!bytes || bytes->empty() || FAILED(device_->CreateComputeShader(
              bytes->data(), bytes->size(), nullptr, &passes[pass]))) return false;
      }
      const UINT both = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      texture_t base, horizontal, q, blend, weight, majorant, vertical, final;
      if (!create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_SHADER_RESOURCE, base) ||
          !create_texture(device_.Get(), field_w_, crop_h_ + 4u, DXGI_FORMAT_R32_FLOAT, both, horizontal) ||
          !create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, both, q) ||
          !create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, both, blend) ||
          !create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, both, weight) ||
          !create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, both, majorant) ||
          !create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, both, vertical) ||
          !create_texture(device_.Get(), field_w_, field_h_, DXGI_FORMAT_R32_FLOAT, both, final)) return false;
      buffer_t control;
      if (!create_buffer(device_.Get(), sizeof(std::uint32_t), sizeof(std::uint32_t), both, control)) return false;
      const auto create_cb = [&](const void *data, UINT bytes, ComPtr<ID3D11Buffer> &buffer) {
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = bytes;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial {data, 0u, 0u};
        return SUCCEEDED(device_->CreateBuffer(&desc, &initial, &buffer));
      };
      const std::array<uint4_t, 3u> distance_words {
        constants_.source, uint4_t {field_w_, field_h_, 0u, 0u}, content_,
      };
      std::array<std::uint32_t, 16u> depth_words {};
      depth_words[0u] = field_w_;
      depth_words[1u] = field_h_;
      std::copy(content_.begin(), content_.end(), depth_words.begin() + 9u);
      const std::array<uint4_t, 5u> locator_words {
        constants_.field, uint4_t {source_w_, source_h_, 1u, 0u},
        constants_.frame, content_, uint4_t {constants_.observation[0u], constants_.observation[1u], 0u, 0u},
      };
      ComPtr<ID3D11Buffer> distance_cb, depth_cb, locator_cb;
      if (!create_cb(distance_words.data(), sizeof(distance_words), distance_cb) ||
          !create_cb(depth_words.data(), sizeof(depth_words), depth_cb) ||
          !create_cb(locator_words.data(), sizeof(locator_words), locator_cb)) return false;
      context_->UpdateSubresource(base.texture.Get(), 0u, nullptr, base_values.data(), field_w_ * sizeof(float), 0u);
      unbind();
      context_->CSSetConstantBuffers(0u, 1u, distance_cb.GetAddressOf());
      context_->CSSetShader(passes[0u].Get(), nullptr, 0u);
      context_->CSSetShaderResources(0u, 1u, support_.srv.GetAddressOf());
      context_->CSSetUnorderedAccessViews(0u, 1u, horizontal.uav.GetAddressOf(), nullptr);
      context_->Dispatch((field_w_ + 15u) / 16u, (field_h_ + 15u) / 16u, 1u);
      unbind();
      context_->CSSetShader(passes[1u].Get(), nullptr, 0u);
      context_->CSSetShaderResources(1u, 1u, horizontal.srv.GetAddressOf());
      context_->CSSetUnorderedAccessViews(0u, 1u, q.uav.GetAddressOf(), nullptr);
      context_->Dispatch((field_w_ + 15u) / 16u, (field_h_ + 15u) / 16u, 1u);
      unbind();
      ID3D11Buffer *cbuffers[3u] {depth_cb.Get(), geometry_cb_.Get(), locator_cb.Get()};
      context_->CSSetConstantBuffers(0u, 3u, cbuffers);
      ID3D11ShaderResourceView *condition_srvs[8u] {
        geometry_.srv.Get(), cut_.srv.Get(), base.srv.Get(), plane_.srv.Get(),
        params_.srv.Get(), q.srv.Get(), identity_.srv.Get(), ocr_.srv.Get(),
      };
      ID3D11UnorderedAccessView *condition_uavs[3u] {blend.uav.Get(), control.uav.Get(), weight.uav.Get()};
      context_->CSSetShader(passes[2u].Get(), nullptr, 0u);
      context_->CSSetShaderResources(0u, 8u, condition_srvs);
      context_->CSSetUnorderedAccessViews(3u, 3u, condition_uavs, nullptr);
      context_->Dispatch((field_w_ + 15u) / 16u, (field_h_ + 15u) / 16u, 1u);
      unbind();
      ID3D11ShaderResourceView *limiter_srvs[4u] {blend.srv.Get(), nullptr, control.srv.Get(), base.srv.Get()};
      ID3D11UnorderedAccessView *vertical_uavs[2u] {majorant.uav.Get(), vertical.uav.Get()};
      context_->CSSetShader(passes[3u].Get(), nullptr, 0u);
      context_->CSSetShaderResources(0u, 4u, limiter_srvs);
      context_->CSSetUnorderedAccessViews(0u, 2u, vertical_uavs, nullptr);
      context_->Dispatch((field_w_ + v2::limiter_group_lines - 1u) / v2::limiter_group_lines, 1u, 1u);
      unbind();
      limiter_srvs[0u] = vertical.srv.Get();
      context_->CSSetShader(passes[4u].Get(), nullptr, 0u);
      context_->CSSetShaderResources(0u, 4u, limiter_srvs);
      context_->CSSetUnorderedAccessViews(0u, 1u, final.uav.GetAddressOf(), nullptr);
      context_->Dispatch((field_h_ + v2::limiter_group_lines - 1u) / v2::limiter_group_lines, 1u, 1u);
      unbind();
      return read_texture(device_.Get(), context_.Get(), q.texture.Get(), out.q) &&
        read_texture(device_.Get(), context_.Get(), weight.texture.Get(), out.weight) &&
        read_texture(device_.Get(), context_.Get(), blend.texture.Get(), out.blend) &&
        read_texture(device_.Get(), context_.Get(), vertical.texture.Get(), out.vertical) &&
        read_texture(device_.Get(), context_.Get(), final.texture.Get(), out.final) &&
        read_texture(device_.Get(), context_.Get(), base.texture.Get(), out.uploaded_base) &&
        read_buffer(device_.Get(), context_.Get(), control.buffer.Get(), out.control);
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

    bool sample_probability_native(std::uint32_t x, std::uint32_t y, float &value) {
      if (!probability_probe_) {
        const std::string source =
          "#include \"host_sbs_subtitle_contour_cs.hlsl\"\n"
          "StructuredBuffer<uint> ProbePoint : register(t12);\n"
          "RWStructuredBuffer<uint> ProbeValue : register(u6);\n"
          "[numthreads(1,1,1)] void probe_main() { float p; bool ok = CurrentProbabilityEvidence("
          "int2(ProbePoint[0],ProbePoint[1]),p); ProbeValue[0] = ok ? 1u : 0u; ProbeValue[1] = asuint(p); }\n";
        ComPtr<ID3DBlob> bytecode, diagnostics;
        const auto path = (std::filesystem::path(SUNSHINE_SHADERS_DIR) / "current_probability_probe.hlsl").string();
        const auto compiled = D3DCompile(source.data(), source.size(), path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "probe_main", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0u, &bytecode, &diagnostics);
        if (FAILED(compiled)) {
          ADD_FAILURE() << (diagnostics ? std::string(static_cast<const char *>(diagnostics->GetBufferPointer()), diagnostics->GetBufferSize()) : "Probability probe compilation failed");
          return false;
        }
        if (FAILED(device_->CreateComputeShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &probability_probe_))) {
          return false;
        }
      }
      buffer_t point, output;
      if (!create_buffer(device_.Get(), 8u, 4u, D3D11_BIND_SHADER_RESOURCE, point) || !create_buffer(device_.Get(), 8u, 4u, D3D11_BIND_UNORDERED_ACCESS, output)) {
        return false;
      }
      const std::array<std::uint32_t, 2u> xy {x, y};
      unbind();
      context_->UpdateSubresource(point.buffer.Get(), 0u, nullptr, xy.data(), 0u, 0u);
      context_->CSSetConstantBuffers(0u, 1u, contour_cb_.GetAddressOf());
      context_->CSSetShaderResources(1u, 1u, probability_.srv.GetAddressOf());
      context_->CSSetShaderResources(12u, 1u, point.srv.GetAddressOf());
      context_->CSSetUnorderedAccessViews(6u, 1u, output.uav.GetAddressOf(), nullptr);
      context_->CSSetShader(probability_probe_.Get(), nullptr, 0u);
      context_->Dispatch(1u, 1u, 1u);
      ID3D11ShaderResourceView *empty_srv = nullptr;
      ID3D11UnorderedAccessView *empty_uav = nullptr;
      context_->CSSetShaderResources(12u, 1u, &empty_srv);
      context_->CSSetUnorderedAccessViews(6u, 1u, &empty_uav, nullptr);
      unbind();
      std::vector<std::uint32_t> words;
      if (!read_buffer(device_.Get(), context_.Get(), output.buffer.Get(), words) || words.size() != 2u || words[0u] != 1u) {
        return false;
      }
      value = std::bit_cast<float>(words[1u]);
      return true;
    }

    void paint_point(std::uint32_t x = 60u, float gray = .75f) {
      pixels_[native_index(x, body_y() + 5u)] = {gray, gray, gray, 1.f};
    }

    // Explicit authored current font evidence, separate from the pixel whose diagnostic
    // history is under test. It is never added implicitly by run(), observe(), or upload().
    std::array<std::uint32_t, 2u> palette_anchor_point(std::uint32_t slot = 0u, std::uint32_t grid_x = 2u, std::uint32_t grid_y = 2u) const {
      const auto offset = v2::subtitle_ocr_raw_box_offset + slot * v2::subtitle_ocr_box_word_count;
      const auto project = [](std::uint32_t low, std::uint32_t high, std::uint32_t origin, std::uint32_t sample, std::uint32_t grid, std::uint32_t source, std::uint32_t extent) {
        return static_cast<std::uint32_t>(((std::uint64_t(low - origin) * (2u * grid) + (2u * sample + 1u) * (high - low)) * source) / (2ull * grid * extent));
      };
      return {std::min(project(ocr_words_[offset], ocr_words_[offset + 2u], content_[0u], grid_x, v2::subtitle_contour_palette_grid_columns, source_w_, content_[2u] - content_[0u]), source_w_ - 1u), std::min(project(ocr_words_[offset + 1u], ocr_words_[offset + 3u], content_[1u], grid_y, v2::subtitle_contour_palette_grid_rows, source_h_, content_[3u] - content_[1u]) - crop_top_, crop_h_ - 1u)};
    }

    void paint_palette_anchor(pixel_t color, std::uint32_t slot = 0u) {
      const auto p = palette_anchor_point(slot);
      pixels_[native_index(p[0u], p[1u])] = color;
    }

    void paint_glyph_point(std::uint32_t x = 60u, float gray = .75f) {
      paint_point(x, gray);
      paint_palette_anchor({gray, gray, gray, 1.f});
    }

    void probability_at_native(std::uint32_t x, std::uint32_t y, float value) {
      const auto dx = (static_cast<float>(x) + .5f) *
                        (static_cast<float>(v2::subtitle_ocr_output_width) / source_w_) -
                      .5f;
      const auto dy = (static_cast<float>(y) + .5f) *
                        (static_cast<float>(v2::subtitle_ocr_output_height) / crop_h_) -
                      .5f;
      const auto left = static_cast<std::int32_t>(std::floor(dx));
      const auto top = static_cast<std::int32_t>(std::floor(dy));
      for (auto oy = 0; oy != 2; ++oy) {
        for (auto ox = 0; ox != 2; ++ox) {
          const auto px = std::clamp(left + ox, 0, int(v2::subtitle_ocr_output_width) - 1);
          const auto py = std::clamp(top + oy, 0, int(v2::subtitle_ocr_output_height) - 1);
          probability_values_[std::size_t(py) * v2::subtitle_ocr_output_width + px] = value;
        }
      }
    }

    // Author only a canonical current SCL1 context. No learning episode, trusted span or vote
    // supplies pixel authority; the signed current line reference owns geometry admission.
    void current_line_reference(std::int32_t center_twice_source, std::uint32_t reference_width) {
      const auto h = v2::subtitle_ocr_classifier_history_offset;
      std::fill(ocr_words_.begin() + h, ocr_words_.end(), 0u);
      ocr_words_[h] = v2::subtitle_ocr_classifier_history_tag;
      ocr_words_[h + 1u] = v2::subtitle_ocr_classifier_history_schema;
      ocr_words_[h + 2u] = v2::subtitle_ocr_classifier_history_known_flags;
      std::copy_n(constants_.observation.begin(), 2u, ocr_words_.begin() + h + 4u);
      std::copy_n(constants_.frame.begin(), 2u, ocr_words_.begin() + h + 6u);
      std::copy_n(constants_.frame.begin() + 2u, 2u, ocr_words_.begin() + h + 8u);
      std::copy(constants_.source.begin(), constants_.source.end(), ocr_words_.begin() + h + 10u);
      std::copy_n(constants_.field.begin(), 2u, ocr_words_.begin() + h + 14u);
      std::copy(content_.begin(), content_.end(), ocr_words_.begin() + h + 16u);
      ocr_words_[h + 20u] = std::bit_cast<std::uint32_t>(center_twice_source);
      ocr_words_[h + 21u] = reference_width;
      const auto lower = (source_h_ * v2::subtitle_ocr_line_search_bottom_height_numerator +
                          v2::subtitle_ocr_line_search_bottom_height_denominator - 1u) /
                         v2::subtitle_ocr_line_search_bottom_height_denominator;
      ocr_words_[h + 22u] = source_h_ - lower;
    }

    std::size_t point_index(std::uint32_t x = 60u) const {
      return native_index(x, body_y() + 5u);
    }

    void expect_count(std::size_t point, std::uint32_t count) {
      EXPECT_EQ(result_.features[point][3u], feature_code(count));
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
      EXPECT_EQ(pin_count(result_.matte), 0u);
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
    std::array<ComPtr<ID3D11ComputeShader>, 5u> shaders_;
    ComPtr<ID3D11ComputeShader> probability_probe_;
    ComPtr<ID3D11Buffer> contour_cb_, geometry_cb_, indirect_;
    texture_t display_, support_, matte_, field_, features_, next_features_, completed_display_;
    buffer_t probability_, plane_, params_, ocr_, cut_, geometry_, identity_, completed_probability_;
    std::uint32_t source_w_ {}, source_h_ {}, crop_h_ {}, crop_top_ {}, field_w_ {}, field_h_ {};
    uint4_t content_ {}, native_cover_ {}, field_cover_ {};
    contour_constants_t constants_ {};
    v2::constants_t geometry_constants_ {};
    v2::state_words_t geometry_words_ {};
    std::vector<std::uint32_t> plane_words_, params_words_, ocr_words_, cut_words_;
    std::vector<float> probability_values_;
    std::vector<pixel_t> pixels_;
    std::vector<half_pixel_t> imported_source_;
    snapshot_t result_;
  };

  // Explicit opt-in import of an authenticated older capture's exact native RGB/P/Base bytes
  // and a newly produced current-contract OCR record. This is a native fixture experiment, not
  // a live Dump, camera replay, source-image preprocessing result, or headset acceptance test.
  TEST_F(HostSbsSubtitleContourGpuTest, DenseMatteHasExactBoundedNativeFootprintAcrossTileAndImageEdges) {
    clear_scene(.5f);
    std::vector<std::uint32_t> seeds(pixels_.size());
    seeds[native_index(64u, 12u)] = 1u;
    seeds[native_index(0u, 0u)] = 1u;
    seeds[native_index(source_w_ - 1u, crop_h_ - 1u)] = 1u;
    ASSERT_TRUE(run_dense_only(seeds));
    expect_dense_reference();
    EXPECT_EQ(result_.matte[native_index(70u, 16u)], 1u)
      << "three same-frame growth steps plus the (3,1) collar member";
    EXPECT_EQ(result_.matte[native_index(70u, 17u)], 0u)
      << "the excluded (3,2) collar corner cannot acquire a broad rectangular plateau";
    EXPECT_EQ(result_.matte[native_index(71u, 12u)], 0u);
    EXPECT_EQ(result_.matte[native_index(6u, 4u)], 1u);
    EXPECT_EQ(result_.matte[native_index(6u, 5u)], 0u);
    for (const auto value : result_.matte) EXPECT_LE(value, 1u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, DenseMatteFillsOutlinedBodiesPreservesAaAndLeavesDistantGapOpenOnFirstFrame) {
    clear_scene(.5f);
    paint({53u, 6u, 66u, 22u}, {0.f, 0.f, 0.f, 1.f});
    paint({56u, 9u, 63u, 19u}, {1.f, 1.f, 1.f, 1.f});
    paint({50u, 6u, 51u, 7u}, {.499f, .499f, .499f, 1.f});
    std::vector<std::uint32_t> seeds(pixels_.size());
    for (std::uint32_t y = 6u; y < 22u; ++y) {
      seeds[native_index(53u, y)] = seeds[native_index(65u, y)] = 1u;
    }
    for (std::uint32_t y = 9u; y < 19u; ++y) {
      seeds[native_index(56u, y)] = seeds[native_index(62u, y)] = 1u;
    }
    ASSERT_TRUE(run_dense_only(seeds));
    expect_dense_reference();
    for (std::uint32_t y = 9u; y < 19u; ++y) for (std::uint32_t x = 56u; x < 63u; ++x)
      EXPECT_EQ(result_.matte[native_index(x, y)], 1u) << "opaque body cannot remain an edge-only mask";
    EXPECT_EQ(result_.matte[native_index(50u, 6u)], 1u)
      << "low-amplitude AA remains protected without a color-visibility threshold";
    EXPECT_EQ(result_.matte[native_index(90u, 12u)], 0u);
    EXPECT_EQ(result_.matte[native_index(120u, 12u)], 0u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, DenseMatteBrightnessBarrierDoesNotGrowAcrossAnOtherwiseCoherentEdge) {
    clear_scene(.375f);
    paint({65u, 0u, source_w_, crop_h_}, {.4375f, .4375f, .4375f, 1.f});
    std::vector<std::uint32_t> seeds(pixels_.size());
    seeds[native_index(64u, 12u)] = 1u;
    ASSERT_TRUE(run_dense_only(seeds));
    expect_dense_reference();
    EXPECT_EQ(result_.matte[native_index(67u, 12u)], 1u) << "the explicit AA collar crosses the class boundary";
    EXPECT_EQ(result_.matte[native_index(68u, 12u)], 0u) << "geodesic growth itself cannot cross the dark/bright boundary";
  }

  TEST_F(HostSbsSubtitleContourGpuTest, DenseMatteKeepsLargeGlyphCounterAndSceneGapOpenWithInvisibleAaCollar) {
    clear_scene(.5f);
    paint({48u, 1u, 92u, 26u}, {0.f, 0.f, 0.f, 1.f});
    paint({51u, 4u, 89u, 23u}, {1.f, 1.f, 1.f, 1.f});
    paint({57u, 6u, 83u, 21u}, {0.f, 0.f, 0.f, 1.f});
    paint({60u, 9u, 80u, 18u}, {.5f, .5f, .5f, 1.f});
    std::vector<std::uint32_t> seeds(pixels_.size());
    for (std::size_t p = 0u; p < pixels_.size(); ++p) seeds[p] = pixels_[p][0u] != .5f ? 1u : 0u;
    ASSERT_TRUE(run_dense_only(seeds));
    expect_dense_reference();
    EXPECT_EQ(result_.matte[native_index(70u, 13u)], 0u) << "the glyph counter remains scene membership";
    EXPECT_EQ(result_.matte[native_index(98u, 13u)], 0u) << "a separate scene gap remains scene membership";
    EXPECT_EQ(result_.matte[native_index(45u, 12u)], 1u)
      << "the native collar protects RGB-rounded-to-background AA without visibility heuristics";
  }

  TEST_F(HostSbsSubtitleContourGpuTest, DenseMatteCurrentEmptyInvalidAndUnsupportedClearPoisonDespiteStaleSeeds) {
    clear_scene(.5f);
    std::vector<std::uint32_t> seeds(pixels_.size(), 1u);
    ASSERT_TRUE(run_dense_only(seeds));
    ASSERT_EQ(pin_count(result_.matte), seeds.size());
    without_current_authority(true);
    ASSERT_TRUE(run_dense_only(seeds));
    EXPECT_EQ(pin_count(result_.matte), 0u);
    make_authority(2u, 1100001u);
    without_current_authority();
    ASSERT_TRUE(run_dense_only(seeds));
    EXPECT_EQ(pin_count(result_.matte), 0u);
    make_authority(3u, 1200001u);
    constants_.source[0u] = v2::subtitle_contour_source_max_extent + 1u;
    matte_ = {};
    ASSERT_TRUE(create_texture(device_.Get(), 1u, 1u, DXGI_FORMAT_R32_UINT,
      D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, matte_));
    ASSERT_TRUE(run_dense_only(seeds));
    EXPECT_EQ(result_.matte, std::vector<std::uint32_t> {0u});
  }

  TEST_F(HostSbsSubtitleContourGpuTest, DenseMatteInvalidNativeColorsAndNonbinarySeedsCannotAuthorizePixels) {
    clear_scene(.5f);
    std::vector<std::uint32_t> seeds(pixels_.size());
    seeds[native_index(64u, 12u)] = 2u;
    seeds[native_index(90u, 12u)] = 1u;
    pixels_[native_index(90u, 12u)][3u] = 0.f;
    seeds[native_index(110u, 12u)] = 1u;
    pixels_[native_index(110u, 12u)][0u] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(run_dense_only(seeds));
    expect_dense_reference();
    EXPECT_EQ(pin_count(result_.matte), 0u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, DenseMatteFieldIndirectGridStridesOverLargerNativeBandWithoutSeams) {
    ASSERT_TRUE(resize(2053u, 1155u, 770u, 434u));
    clear_scene(.5f);
    std::vector<std::uint32_t> seeds(pixels_.size());
    for (const auto x : {15u, 16u, 767u, 768u, 1535u, 1536u, 2052u})
      seeds[native_index(x, crop_h_ / 2u)] = 1u;
    seeds[native_index(1024u, 0u)] = seeds[native_index(1024u, crop_h_ - 1u)] = 1u;
    ASSERT_TRUE(run_dense_only(seeds));
    expect_dense_reference();
    EXPECT_EQ(result_.matte[native_index(774u, crop_h_ / 2u)], 1u);
    EXPECT_EQ(result_.matte[native_index(1542u, crop_h_ / 2u)], 1u);
    EXPECT_EQ(result_.matte[native_index(1000u, crop_h_ / 2u)], 0u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, SavedNativeContourReplay) {
    const auto *cases_environment = std::getenv("SUNSHINE_TEST_CONTOUR_REPLAY_CASES");
    const auto *output_environment = std::getenv("SUNSHINE_TEST_CONTOUR_REPLAY_OUTPUT");
    if (!cases_environment && !output_environment) {
      GTEST_SKIP() << "Set SUNSHINE_TEST_CONTOUR_REPLAY_CASES and SUNSHINE_TEST_CONTOUR_REPLAY_OUTPUT for native fixture imports";
    }
    ASSERT_NE(cases_environment, nullptr);
    ASSERT_NE(output_environment, nullptr);
    const std::filesystem::path cases_path(cases_environment), output_root(output_environment);
    ASSERT_FALSE(output_root.empty());
    ASSERT_TRUE(!std::filesystem::exists(output_root) ||
                (std::filesystem::is_directory(output_root) && std::filesystem::is_empty(output_root)))
      << "Native fixture output must be a new or empty directory";
    const auto specification_bytes = replay_bytes(cases_path);
    const auto specification = nlohmann::json::parse(specification_bytes.begin(), specification_bytes.end());
    ASSERT_EQ(specification.at("schema"), 1);
    const auto &cases = specification.at("cases");
    ASSERT_TRUE(cases.is_array());
    ASSERT_GT(cases.size(), 0u);
    ASSERT_LE(cases.size(), 8u);
    const auto producer_sources = models::host_sbs_shader_cache::snapshot_sources(
      SUNSHINE_SHADERS_DIR, models::host_sbs_shader_cache::parallax_v2_producer_specs);
    const auto producer_closure = models::host_sbs_shader_cache::source_closure_sha256(producer_sources);
    ASSERT_EQ(producer_closure, models::host_sbs_shader_cache::parallax_v2_producer_source_closure_sha256)
      << "The fixture must use the generated current production closure";
    ComPtr<IDXGIDevice> dxgi_device;
    ComPtr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC adapter_description {};
    ASSERT_TRUE(SUCCEEDED(device_.As(&dxgi_device)));
    ASSERT_TRUE(SUCCEEDED(dxgi_device->GetAdapter(&adapter)));
    ASSERT_TRUE(SUCCEEDED(adapter->GetDesc(&adapter_description)));
    std::filesystem::create_directories(output_root);
    const auto write_bytes = [](const std::filesystem::path &path, const void *data, std::size_t size) {
      std::ofstream output(path, std::ios::binary);
      output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
      if (!output) throw std::runtime_error("Cannot write native fixture output: " + path.string());
    };
    for (std::size_t index = 0u; index < cases.size(); ++index) {
      const auto &entry = cases[index];
      SCOPED_TRACE(index);
      ASSERT_TRUE(entry.at("provenance").is_object());
      const auto source = entry.at("source").get<std::array<std::uint32_t, 2u>>();
      const auto field = entry.at("field").get<std::array<std::uint32_t, 2u>>();
      const auto content = entry.at("content").get<uint4_t>();
      const auto frame = entry.at("frame").get<std::uint64_t>();
      const auto generation = entry.at("generation").get<std::uint64_t>();
      const auto timestamp = entry.at("observation_timestamp_us").get<std::uint64_t>();
      const auto epoch = entry.at("cut_epoch").get<std::uint32_t>();
      const auto target_bits = entry.at("ui_target_bits").get<std::uint32_t>();
      const auto target = std::bit_cast<float>(target_bits);
      ASSERT_TRUE(std::isfinite(target) && target >= 0.0f && target <= v2::direct_container_limit);
      ASSERT_GT(frame, 0u);
      ASSERT_GT(timestamp, 0u);
      ASSERT_GT(source[0u], 40u);
      ASSERT_LE(source[0u], v2::subtitle_contour_source_max_extent);
      ASSERT_GT(source[1u], 0u);
      ASSERT_LE(source[1u], v2::subtitle_contour_source_max_extent);
      ASSERT_TRUE(v2::subtitle_ocr_field_is_calibrated(field[0u], field[1u]));
      ASSERT_LT(content[0u], content[2u]);
      ASSERT_LT(content[1u], content[3u]);
      ASSERT_LE(content[2u], field[0u]);
      ASSERT_LE(content[3u], field[1u]);
      const auto load = [&](const char *key) {
        const auto &artifact = entry.at("artifacts").at(key);
        const auto path = cases_path.parent_path() / std::filesystem::path(artifact.at("path").get<std::string>());
        auto bytes = replay_bytes(path);
        if (replay_sha256(bytes.data(), bytes.size()) != artifact.at("sha256").get<std::string>()) {
          throw std::runtime_error("Native fixture input digest mismatch: " + path.string());
        }
        return bytes;
      };
      const auto source_bytes = load("source_band");
      const auto probability_bytes = load("probability");
      const auto base_bytes = load("base");
      const auto record_bytes = load("ocr_record");
      ASSERT_EQ(probability_bytes.size(), std::size_t(v2::subtitle_ocr_output_width) * v2::subtitle_ocr_output_height * sizeof(float));
      ASSERT_EQ(base_bytes.size(), std::size_t(field[0u]) * field[1u] * sizeof(float));
      ASSERT_EQ(record_bytes.size(), std::size_t(v2::subtitle_ocr_record_word_count) * sizeof(std::uint32_t));
      ASSERT_TRUE(resize(source[0u], source[1u], field[0u], field[1u], content));
      ASSERT_EQ(source_bytes.size(), std::size_t(source_w_) * crop_h_ * sizeof(half_pixel_t));
      make_authority(frame, timestamp, true, generation, epoch);
      imported_source_.resize(std::size_t(source_w_) * crop_h_);
      std::memcpy(imported_source_.data(), source_bytes.data(), source_bytes.size());
      std::memcpy(probability_values_.data(), probability_bytes.data(), probability_bytes.size());
      ASSERT_TRUE(std::all_of(probability_values_.begin(), probability_values_.end(), [](float value) {
        return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
      }));
      std::vector<float> base_values(std::size_t(field_w_) * field_h_);
      std::memcpy(base_values.data(), base_bytes.data(), base_bytes.size());
      ASSERT_TRUE(std::all_of(base_values.begin(), base_values.end(), [](float value) { return std::isfinite(value); }));
      std::vector<std::uint32_t> record(v2::subtitle_ocr_record_word_count);
      std::memcpy(record.data(), record_bytes.data(), record_bytes.size());
      ASSERT_EQ(record[0u], v2::subtitle_ocr_record_schema);
      ASSERT_EQ(record[1u], v2::subtitle_ocr_record_tag);
      ASSERT_EQ(record[2u], 1u);
      ASSERT_EQ(record[3u], record[4u]);
      ASSERT_LE(record[4u], v2::subtitle_plane_cover_capacity);
      for (std::size_t word = 5u; word < 16u; ++word) ASSERT_EQ(record[word], ocr_words_[word]) << word;
      std::vector<uint4_t> covers;
      std::uint32_t kinds = 0u;
      for (std::uint32_t slot = 0u; slot < record[4u]; ++slot) {
        const auto offset = v2::subtitle_ocr_final_box_offset + slot * v2::subtitle_ocr_box_word_count;
        covers.push_back({record[offset], record[offset + 1u], record[offset + 2u], record[offset + 3u]});
        if ((record[offset + 5u] & v2::subtitle_ocr_box_flag_ribbon) != 0u) kinds |= 1u << slot;
      }
      current_covers(covers);
      ocr_words_ = record;
      plane_words_[v2::subtitle_plane_adaptive_offset + 3u] = target_bits;
      params_words_[v2::subtitle_condition_param_current_kind_mask_word] = kinds;
      params_words_[v2::subtitle_condition_param_target_word] = covers.empty() ? 0u : target_bits;
      ASSERT_TRUE(run());
      bool frozen_support_replayed = false;
      if (entry.at("artifacts").contains("native_support")) {
        const auto bytes = load("native_support");
        ASSERT_EQ(bytes.size(), result_.support.size() * sizeof(std::uint32_t));
        std::vector<std::uint32_t> frozen(result_.support.size());
        std::memcpy(frozen.data(), bytes.data(), bytes.size());
        ASSERT_EQ(result_.support, frozen) << "Current sparse producer must preserve the supplied frozen native support";
        ASSERT_TRUE(run_dense_only(frozen));
        frozen_support_replayed = true;
      }
      ASSERT_NO_FATAL_FAILURE(expect_dense_reference());
      // Stored display-band values are uploaded verbatim as RGBA16F, and the independent
      // production snapshot must return the exact bytes. No source PNG or transfer conversion.
      ASSERT_EQ(result_.source, imported_source_);
      ASSERT_EQ(replay_sha256(result_.probability_bits.data(), probability_bytes.size()),
                replay_sha256(probability_bytes.data(), probability_bytes.size()));
      replay_fields_t fields;
      ASSERT_TRUE(run_replay_conditioner(base_values, fields));
      ASSERT_EQ(replay_sha256(fields.uploaded_base.data(), base_bytes.size()), replay_sha256(base_bytes.data(), base_bytes.size()));
      ASSERT_EQ(fields.control.size(), 1u);
      ASSERT_TRUE(std::all_of(fields.weight.begin(), fields.weight.end(), [](float value) {
        return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
      }));
      for (const auto *values : {&fields.blend, &fields.vertical, &fields.final}) {
        ASSERT_TRUE(std::all_of(values->begin(), values->end(), [](float value) { return std::isfinite(value); }));
      }
      if (covers.empty()) {
        expect_no_mask();
        EXPECT_EQ(fields.control[0u], 0u);
        EXPECT_TRUE(std::all_of(fields.weight.begin(), fields.weight.end(), [](float value) { return std::bit_cast<std::uint32_t>(value) == 0u; }));
        EXPECT_EQ(replay_sha256(fields.final.data(), base_bytes.size()), replay_sha256(base_bytes.data(), base_bytes.size()));
      } else {
        ASSERT_GT(pin_count(result_.support), 0u);
        ASSERT_EQ(fields.control[0u], 1u) << "The production conditioner must authenticate this newly authored fixture identity";
      }
      // Authenticate the actual GPU leaves through the same independent C++ checks used by
      // Dump 3D. A native shader fixture must not pass while its saved live tuple is rejected.
      const auto vector_bytes = [](const auto &values) {
        std::vector<std::uint8_t> bytes(values.size() * sizeof(values[0u]));
        if (!bytes.empty()) {
          std::memcpy(bytes.data(), values.data(), bytes.size());
        }
        return bytes;
      };
      std::vector<float> geometry_state;
      geometry_state.reserve(geometry_words_.size());
      for (const auto word : geometry_words_) {
        geometry_state.push_back(std::bit_cast<float>(word));
      }
      const bool cpp_contour_authenticated =
        platf::sbs_debug::detail::subtitle_contour_clear_policy_is_authenticated(
          vector_bytes(result_.identity),
          vector_bytes(ocr_words_),
          geometry_state,
          vector_bytes(result_.support),
          vector_bytes(result_.field),
          vector_bytes(fields.q),
          vector_bytes(result_.features),
          vector_bytes(result_.source),
          vector_bytes(result_.probability_bits)
        );
      ASSERT_TRUE(cpp_contour_authenticated) << "C++ current-glyph/source/history/distance authentication must accept the real GPU tuple";
      const bool cpp_presentation_authenticated =
        platf::sbs_debug::detail::subtitle_gaussian_fields_are_authenticated(
          vector_bytes(result_.identity),
          vector_bytes(plane_words_),
          vector_bytes(ocr_words_),
          vector_bytes(fields.q),
          vector_bytes(fields.weight),
          vector_bytes(fields.blend),
          vector_bytes(fields.vertical),
          vector_bytes(fields.control),
          vector_bytes(fields.uploaded_base),
          vector_bytes(fields.final)
        );
      ASSERT_TRUE(cpp_presentation_authenticated) << "C++ M/proposal/shared-conditioner authentication must accept the real GPU leaves";
      const auto case_root = output_root / ("case_" + std::to_string(index));
      ASSERT_TRUE(std::filesystem::create_directory(case_root));
      nlohmann::json artifacts = nlohmann::json::object();
      const auto save = [&](const char *name, const auto &values) {
        const auto size = values.size() * sizeof(values[0u]);
        write_bytes(case_root / name, values.data(), size);
        artifacts[name] = {{"sha256", replay_sha256(values.data(), size)}, {"byte_count", size}};
      };
      save("uploaded_source.rgba16f", result_.source);
      save("uploaded_probability.f32", result_.probability_bits);
      save("uploaded_base.f32", fields.uploaded_base);
      save("ocr_record.u32", ocr_words_);
      save("locator_state.u32", plane_words_);
      save("contour_identity.u32", result_.identity);
      save("contour_history.rgba32f", result_.features);
      save("contour_support.u32", result_.support);
      save("subtitle_glyph_matte.u32", result_.matte);
      save("contour_field_pin.u32", result_.field);
      save("contour_distance.f32", fields.q);
      save("gaussian_weight.f32", fields.weight);
      save("gaussian_blend.f32", fields.blend);
      save("gaussian_control.u32", fields.control);
      save("gaussian_vertical_conditioned.f32", fields.vertical);
      save("final.f32", fields.final);
      nlohmann::json identity {
        {"schema", 1}, {"kind", "new-current-contract-native-contour-fixture-import"},
        {"case", entry}, {"input_specification", cases_path.string()},
        {"input_specification_sha256", replay_sha256(specification_bytes.data(), specification_bytes.size())},
        {"driver", specification.value("driver", std::string("warp"))},
        {"adapter", {{"vendor_id", adapter_description.VendorId}, {"device_id", adapter_description.DeviceId},
          {"luid_low", adapter_description.AdapterLuid.LowPart}, {"luid_high", adapter_description.AdapterLuid.HighPart}}},
        {"depth_coordinate_v2_schema", v2::contract_schema}, {"depth_coordinate_v2_tag", v2::contract_tag},
        {"depth_coordinate_v2_canonical_sha256", v2::contract_canonical_sha256},
        {"producer_source_closure_sha256", producer_closure},
        {"artifacts", artifacts}, {"ui_target_bits", target_bits},
        {"native_support_count", pin_count(result_.support)},
        {"native_glyph_matte_count", pin_count(result_.matte)},
        {"cpp_dense_glyph_matte_authenticated", true},
        {"frozen_native_support_replayed", frozen_support_replayed},
        {"cpp_contour_clear_policy_authenticated", cpp_contour_authenticated},
        {"cpp_gaussian_fields_authenticated", cpp_presentation_authenticated},
        {"geometry_authority", "new fixture-authored current-contract state sealed for the provided source frame/time/domain; not an imported live state"},
        {"upstream_dump_authentication_performed_by_this_test", false},
        {"source_preprocessing_or_camera_replay_executed", false},
        {"production_native_contour_distance_gaussian_and_shared_limiters_executed", true},
        {"renderer_or_headset_executed", false},
      };
      const auto serialized = identity.dump(2) + '\n';
      write_bytes(case_root / "native_fixture_identity.json", serialized.data(), serialized.size());
    }
  }


  // Deliberately separate opt-in experiment. An external source bitmap and temporal-only
  // shader can reduce the production current support; neither is production mask authority.
  TEST_F(HostSbsSubtitleContourGpuTest, ExperimentalBirthPaletteReplay) {
    const auto *cases_environment = std::getenv("SUNSHINE_TEST_BIRTH_PALETTE_CASES");
    const auto *output_environment = std::getenv("SUNSHINE_TEST_BIRTH_PALETTE_OUTPUT");
    if (!cases_environment && !output_environment) {
      GTEST_SKIP() << "Set SUNSHINE_TEST_BIRTH_PALETTE_CASES and SUNSHINE_TEST_BIRTH_PALETTE_OUTPUT for the source-bitmap diagnostic";
    }
    ASSERT_NE(cases_environment, nullptr);
    ASSERT_NE(output_environment, nullptr);
    const std::filesystem::path cases_path(cases_environment), output_root(output_environment);
    ASSERT_FALSE(output_root.empty());
    ASSERT_TRUE(!std::filesystem::exists(output_root) ||
                (std::filesystem::is_directory(output_root) && std::filesystem::is_empty(output_root)))
      << "Experimental output must be a new or empty directory";
    const auto specification_bytes = replay_bytes(cases_path);
    const auto specification = nlohmann::json::parse(specification_bytes.begin(), specification_bytes.end());
    ASSERT_EQ(specification.at("schema"), 1);
    const auto &cases = specification.at("cases");
    ASSERT_TRUE(cases.is_array());
    ASSERT_GT(cases.size(), 0u);
    ASSERT_LE(cases.size(), 32u);
    const auto resolve = [&](const std::string &path) {
      return cases_path.parent_path() / std::filesystem::path(path);
    };
    const auto baseline_root = resolve(specification.at("baseline_shader_root").get<std::string>());
    const auto variant_root = resolve(specification.at("variant_shader_root").get<std::string>());
    namespace cache = models::host_sbs_shader_cache;
    const auto baseline_sources = cache::snapshot_sources(baseline_root, cache::parallax_v2_producer_specs);
    const auto baseline_closure = cache::source_closure_sha256(baseline_sources);
    ASSERT_EQ(baseline_closure, specification.at("baseline_producer_closure_sha256").get<std::string>());
    ASSERT_EQ(baseline_closure, cache::parallax_v2_producer_source_closure_sha256)
      << "The experimental baseline must still authenticate the current production producer";
    const auto fixture_sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, cache::parallax_v2_producer_specs);
    ASSERT_EQ(cache::source_closure_sha256(fixture_sources), baseline_closure)
      << "Normal fixture shaders and the supplied production baseline must be identical";
    const auto expected_variant_closure = specification.at("variant_temporal_closure_sha256").get<std::string>();
    const auto production_temporal = shaders_[0u];
    const auto production_dense = shaders_[1u];
    const bool clip_dense = specification.contains("variant_dense_shader_root");
    ASSERT_EQ(clip_dense, specification.contains("variant_dense_closure_sha256"));
    ASSERT_EQ(clip_dense, specification.contains("variant_dense_policy"));
    if (clip_dense) {
      ASSERT_EQ(specification.at("variant_dense_policy"), "dense_m1_current_cover_intersection");
    }
    const auto dense_root = clip_dense ? resolve(specification.at("variant_dense_shader_root").get<std::string>()) : std::filesystem::path {};
    const auto expected_dense_closure = clip_dense ? specification.at("variant_dense_closure_sha256").get<std::string>() : std::string {};
    ComPtr<IDXGIDevice> dxgi_device;
    ComPtr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC adapter_description {};
    ASSERT_TRUE(SUCCEEDED(device_.As(&dxgi_device)));
    ASSERT_TRUE(SUCCEEDED(dxgi_device->GetAdapter(&adapter)));
    ASSERT_TRUE(SUCCEEDED(adapter->GetDesc(&adapter_description)));
    std::filesystem::create_directories(output_root);
    const auto write_bytes = [](const std::filesystem::path &path, const void *data, std::size_t size) {
      std::ofstream output(path, std::ios::binary);
      output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
      if (!output) throw std::runtime_error("Cannot write experimental fixture output: " + path.string());
    };
    for (std::size_t index = 0u; index < cases.size(); ++index) {
      const auto &entry = cases[index];
      SCOPED_TRACE(index);
      ASSERT_TRUE(entry.at("provenance").is_object());
      const auto source = entry.at("source").get<std::array<std::uint32_t, 2u>>();
      const auto field = entry.at("field").get<std::array<std::uint32_t, 2u>>();
      const auto content = entry.at("content").get<uint4_t>();
      const auto frame = entry.at("frame").get<std::uint64_t>();
      const auto generation = entry.at("generation").get<std::uint64_t>();
      const auto timestamp = entry.at("observation_timestamp_us").get<std::uint64_t>();
      const auto epoch = entry.at("cut_epoch").get<std::uint32_t>();
      const auto target_bits = entry.at("ui_target_bits").get<std::uint32_t>();
      const auto target = std::bit_cast<float>(target_bits);
      ASSERT_TRUE(std::isfinite(target) && target >= 0.0f && target <= v2::direct_container_limit);
      ASSERT_GT(frame, 0u);
      ASSERT_GT(timestamp, 0u);
      ASSERT_GT(source[0u], 40u);
      ASSERT_LE(source[0u], v2::subtitle_contour_source_max_extent);
      ASSERT_GT(source[1u], 0u);
      ASSERT_LE(source[1u], v2::subtitle_contour_source_max_extent);
      ASSERT_TRUE(v2::subtitle_ocr_field_is_calibrated(field[0u], field[1u]));
      ASSERT_LT(content[0u], content[2u]);
      ASSERT_LT(content[1u], content[3u]);
      ASSERT_LE(content[2u], field[0u]);
      ASSERT_LE(content[3u], field[1u]);
      const auto load = [&](const char *key) {
        const auto &artifact = entry.at("artifacts").at(key);
        const auto path = resolve(artifact.at("path").get<std::string>());
        auto bytes = replay_bytes(path);
        if (replay_sha256(bytes.data(), bytes.size()) != artifact.at("sha256").get<std::string>()) {
          throw std::runtime_error("Experimental fixture input digest mismatch: " + path.string());
        }
        if (artifact.contains("bytes") && bytes.size() != artifact.at("bytes").get<std::size_t>()) {
          throw std::runtime_error("Experimental fixture input byte count mismatch: " + path.string());
        }
        return bytes;
      };
      const auto source_bytes = load("source_band");
      const auto probability_bytes = load("probability");
      const auto record_bytes = load("ocr_record");
      const auto eligibility_bytes = load("eligibility");
      ASSERT_EQ(probability_bytes.size(), std::size_t(v2::subtitle_ocr_output_width) * v2::subtitle_ocr_output_height * sizeof(float));
      ASSERT_EQ(record_bytes.size(), std::size_t(v2::subtitle_ocr_record_word_count) * sizeof(std::uint32_t));
      ASSERT_TRUE(resize(source[0u], source[1u], field[0u], field[1u], content));
      ASSERT_EQ(source_bytes.size(), std::size_t(source_w_) * crop_h_ * sizeof(half_pixel_t));
      ASSERT_EQ(eligibility_bytes.size(), std::size_t(source_w_) * crop_h_ * sizeof(std::uint32_t));
      make_authority(frame, timestamp, true, generation, epoch);
      imported_source_.resize(std::size_t(source_w_) * crop_h_);
      std::memcpy(imported_source_.data(), source_bytes.data(), source_bytes.size());
      std::memcpy(probability_values_.data(), probability_bytes.data(), probability_bytes.size());
      ASSERT_TRUE(std::all_of(probability_values_.begin(), probability_values_.end(), [](float value) {
        return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
      }));
      std::vector<std::uint32_t> eligibility(std::size_t(source_w_) * crop_h_);
      std::memcpy(eligibility.data(), eligibility_bytes.data(), eligibility_bytes.size());
      ASSERT_TRUE(std::all_of(eligibility.begin(), eligibility.end(), [](auto value) { return value <= 1u; }));
      std::vector<std::uint32_t> record(v2::subtitle_ocr_record_word_count);
      std::memcpy(record.data(), record_bytes.data(), record_bytes.size());
      ASSERT_EQ(record[0u], v2::subtitle_ocr_record_schema);
      ASSERT_EQ(record[1u], v2::subtitle_ocr_record_tag);
      ASSERT_EQ(record[2u], 1u);
      ASSERT_EQ(record[3u], record[4u]);
      ASSERT_LE(record[4u], v2::subtitle_plane_cover_capacity);
      for (std::size_t word = 5u; word < 16u; ++word) ASSERT_EQ(record[word], ocr_words_[word]) << word;
      std::vector<uint4_t> covers;
      std::uint32_t kinds = 0u;
      for (std::uint32_t slot = 0u; slot < record[4u]; ++slot) {
        const auto offset = v2::subtitle_ocr_final_box_offset + slot * v2::subtitle_ocr_box_word_count;
        covers.push_back({record[offset], record[offset + 1u], record[offset + 2u], record[offset + 3u]});
        if ((record[offset + 5u] & v2::subtitle_ocr_box_flag_ribbon) != 0u) kinds |= 1u << slot;
      }
      current_covers(covers);
      ocr_words_ = record;
      plane_words_[v2::subtitle_plane_adaptive_offset + 3u] = target_bits;
      params_words_[v2::subtitle_condition_param_current_kind_mask_word] = kinds;
      params_words_[v2::subtitle_condition_param_target_word] = covers.empty() ? 0u : target_bits;
      shaders_[0u] = production_temporal;
      shaders_[1u] = production_dense;
      ASSERT_TRUE(run());
      ASSERT_NO_FATAL_FAILURE(expect_dense_reference());
      const auto baseline_support = result_.support;
      const auto baseline_matte = result_.matte;
      ASSERT_TRUE(std::all_of(baseline_support.begin(), baseline_support.end(), [](auto value) { return value <= 1u; }));
      ASSERT_TRUE(std::all_of(baseline_matte.begin(), baseline_matte.end(), [](auto value) { return value <= 1u; }));
      ASSERT_EQ(result_.source, imported_source_);
      ASSERT_EQ(replay_sha256(result_.probability_bits.data(), probability_bytes.size()),
                replay_sha256(probability_bytes.data(), probability_bytes.size()));
      if (entry.at("artifacts").contains("expected_baseline_support")) {
        const auto bytes = load("expected_baseline_support");
        ASSERT_EQ(bytes.size(), baseline_support.size() * sizeof(std::uint32_t));
        ASSERT_EQ(replay_sha256(baseline_support.data(), bytes.size()), replay_sha256(bytes.data(), bytes.size()))
          << "The untouched production baseline must match supplied expected S exactly";
      }
      if (entry.at("artifacts").contains("expected_baseline_matte")) {
        const auto bytes = load("expected_baseline_matte");
        ASSERT_EQ(bytes.size(), baseline_matte.size() * sizeof(std::uint32_t));
        ASSERT_EQ(replay_sha256(baseline_matte.data(), bytes.size()), replay_sha256(bytes.data(), bytes.size()))
          << "The untouched production baseline must match supplied expected M exactly";
      }
      texture_t baseline_read, eligibility_read;
      ASSERT_TRUE(create_texture(device_.Get(), source_w_, crop_h_, DXGI_FORMAT_R32_UINT, D3D11_BIND_SHADER_RESOURCE, baseline_read));
      ASSERT_TRUE(create_texture(device_.Get(), source_w_, crop_h_, DXGI_FORMAT_R32_UINT, D3D11_BIND_SHADER_RESOURCE, eligibility_read));
      ASSERT_NE(baseline_read.texture.Get(), support_.texture.Get());
      ASSERT_NE(eligibility_read.texture.Get(), support_.texture.Get());
      ASSERT_NE(baseline_read.texture.Get(), eligibility_read.texture.Get());
      unbind();
      context_->CopyResource(baseline_read.texture.Get(), support_.texture.Get());
      context_->UpdateSubresource(eligibility_read.texture.Get(), 0u, nullptr, eligibility.data(), source_w_ * sizeof(std::uint32_t), 0u);
      // Both passes start with identical empty diagnostic history. The baseline is not an
      // extra genuine observation, and its same-frame history must not become variant input.
      const UINT zero[4] {};
      const float zeros[4] {};
      context_->ClearUnorderedAccessViewUint(identity_.uav.Get(), zero);
      context_->ClearUnorderedAccessViewFloat(features_.uav.Get(), zeros);
      std::string variant_closure;
      ASSERT_TRUE(set_variant_temporal(variant_root, expected_variant_closure, variant_closure));
      std::string variant_dense_closure;
      if (clip_dense) {
        ASSERT_TRUE(set_variant_dense(dense_root, expected_dense_closure, variant_dense_closure));
      }
      ASSERT_TRUE(run(false, baseline_read.srv.Get(), eligibility_read.srv.Get()));
      ASSERT_NO_FATAL_FAILURE(expect_dense_reference(clip_dense));
      ASSERT_EQ(result_.source, imported_source_);
      ASSERT_EQ(replay_sha256(result_.probability_bits.data(), probability_bytes.size()),
                replay_sha256(probability_bytes.data(), probability_bytes.size()));
      ASSERT_EQ(result_.support.size(), baseline_support.size());
      ASSERT_EQ(result_.matte.size(), baseline_matte.size());
      for (std::size_t p = 0u; p < baseline_support.size(); ++p) {
        ASSERT_LE(result_.support[p], 1u);
        ASSERT_LE(result_.matte[p], 1u);
        ASSERT_LE(result_.support[p], baseline_support[p]) << "Variant S must be a subset at native pixel " << p;
        ASSERT_LE(result_.matte[p], baseline_matte[p]) << "Variant completion must remain a subset at native pixel " << p;
      }
      std::vector<std::uint32_t> preserved_support, preserved_eligibility;
      ASSERT_TRUE(read_texture(device_.Get(), context_.Get(), baseline_read.texture.Get(), preserved_support));
      ASSERT_TRUE(read_texture(device_.Get(), context_.Get(), eligibility_read.texture.Get(), preserved_eligibility));
      ASSERT_EQ(preserved_support, baseline_support);
      ASSERT_EQ(preserved_eligibility, eligibility);
      const auto case_root = output_root / ("case_" + std::to_string(index));
      ASSERT_TRUE(std::filesystem::create_directory(case_root));
      nlohmann::json artifacts = nlohmann::json::object();
      const auto save = [&](const char *name, const auto &values) {
        const auto size = values.size() * sizeof(values[0u]);
        write_bytes(case_root / name, values.data(), size);
        artifacts[name] = {{"sha256", replay_sha256(values.data(), size)}, {"byte_count", size}};
      };
      save("baseline_support.u32", baseline_support);
      save("baseline_matte.u32", baseline_matte);
      save("variant_support.u32", result_.support);
      save("variant_matte.u32", result_.matte);
      save("eligibility.u32", eligibility);
      save("uploaded_source.rgba16f", result_.source);
      save("uploaded_probability.f32", result_.probability_bits);
      save("ocr_record.u32", ocr_words_);
      nlohmann::json identity {
        {"schema", 1}, {"kind", clip_dense ? "opt-in-experimental-temporal-and-dense-clip-replay" : "opt-in-experimental-source-bitmap-temporal-replay"},
        {"scope", "Diagnostic current support and optional dense cover intersection; not production/upstream qualification or a GT/quality assertion"},
        {"case", entry}, {"input_specification", cases_path.string()},
        {"input_specification_sha256", replay_sha256(specification_bytes.data(), specification_bytes.size())},
        {"driver", specification.value("driver", std::string("warp"))},
        {"adapter", {{"vendor_id", adapter_description.VendorId}, {"device_id", adapter_description.DeviceId},
          {"luid_low", adapter_description.AdapterLuid.LowPart}, {"luid_high", adapter_description.AdapterLuid.HighPart}}},
        {"baseline_shader_root", baseline_root.string()}, {"fixture_shader_root", SUNSHINE_SHADERS_DIR},
        {"variant_shader_root", variant_root.string()},
        {"baseline_producer_closure_sha256", baseline_closure},
        {"variant_temporal_closure_sha256", variant_closure},
        {"variant_dense_closure_sha256", variant_dense_closure},
        {"variant_dense_policy", clip_dense ? "dense_m1_current_cover_intersection" : "unchanged_production_dense_m1"},
        {"variant_pass", clip_dense ? "pinned temporal_main and dense_main current-cover intersection; project/history/snapshot unchanged" : "temporal_main only; dense/project/history/snapshot use unchanged production shaders"},
        {"depth_coordinate_v2_schema", v2::contract_schema}, {"depth_coordinate_v2_tag", v2::contract_tag},
        {"depth_coordinate_v2_canonical_sha256", v2::contract_canonical_sha256},
        {"artifacts", artifacts}, {"baseline_support_count", pin_count(baseline_support)},
        {"baseline_matte_count", pin_count(baseline_matte)},
        {"variant_support_count", pin_count(result_.support)}, {"variant_matte_count", pin_count(result_.matte)},
        {"baseline_expected_support_checked", entry.at("artifacts").contains("expected_baseline_support")},
        {"baseline_expected_matte_checked", entry.at("artifacts").contains("expected_baseline_matte")},
        {"baseline_S_and_eligibility_read_only_preserved", true},
        {"variant_S_subset_baseline_S", true}, {"variant_M_subset_baseline_M", true},
        {"source_and_probability_snapshot_bytes_unchanged", true},
        {"history_reset_between_comparisons", true},
        {"production_manifest_assertion_preserved", true},
        {"production_qualification_claimed", false}, {"GT_or_quality_asserted", false},
        {"conditioner_renderer_or_headset_executed", false}
      };
      const auto serialized = identity.dump(2) + '\n';
      write_bytes(case_root / "experimental_fixture_identity.json", serialized.data(), serialized.size());
    }
    unbind();
    shaders_[0u] = production_temporal;
    shaders_[1u] = production_dense;
  }

  TEST_F(HostSbsSubtitleContourGpuTest, EmptyCurrentAuthorityCopiesFullBaseBitsIncludingDistinctPaddingAndSignedZero) {
    constexpr std::uint32_t width = 770u, height = 434u, content_height = 433u;
    ASSERT_TRUE(resize(160u, 90u, width, height, {0u, 0u, width, content_height}));
    current_covers({});
    ASSERT_TRUE(run());
    expect_no_mask();
    std::vector<float> base(std::size_t(width) * height);
    for (std::uint32_t y = 0u; y < height; ++y) {
      for (std::uint32_t x = 0u; x < width; ++x) {
        base[std::size_t(y) * width + x] = .002f + .00001f * float((x + 3u * y) % 101u);
      }
    }
    const auto padding = std::size_t(content_height) * width;
    for (std::uint32_t x = 0u; x < width; ++x) {
      base[padding + x] = x % 3u == 0u ? .0125f :
        std::bit_cast<float>(x % 3u == 1u ? 0x80000000u : 0u);
      ASSERT_NE(std::bit_cast<std::uint32_t>(base[padding + x]),
                std::bit_cast<std::uint32_t>(base[padding - width + x]));
    }
    // Include signed zero in real content as well as padding. Value equality alone cannot
    // establish that the disabled production branches copied the published Base bits.
    base[7u] = std::bit_cast<float>(0x80000000u);
    replay_fields_t fields;
    ASSERT_TRUE(run_replay_conditioner(base, fields));
    ASSERT_EQ(fields.control, std::vector<std::uint32_t> {0u});
    ASSERT_TRUE(std::all_of(fields.weight.begin(), fields.weight.end(), [](float value) {
      return std::bit_cast<std::uint32_t>(value) == 0u;
    }));
    const auto expected = replay_sha256(base.data(), base.size() * sizeof(float));
    for (const auto *leaf : {&fields.uploaded_base, &fields.blend, &fields.vertical, &fields.final}) {
      ASSERT_EQ(leaf->size(), base.size());
      EXPECT_EQ(replay_sha256(leaf->data(), leaf->size() * sizeof(float)), expected);
      EXPECT_EQ(std::bit_cast<std::uint32_t>((*leaf)[7u]), 0x80000000u);
      EXPECT_EQ(std::bit_cast<std::uint32_t>((*leaf)[padding + 1u]), 0x80000000u);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, NativeHorizontalExtensionChangesOnlyInfeasibleBackgroundInsideFootprintRows) {
    ASSERT_TRUE(resize(160u, 90u));
    // One current glyph seed plus its valid two-pixel contrast neighbor. The seed is
    // explicitly aligned to a current core palette sample; invalid surroundings cannot
    // become an accidental outline. Distance/completion remains the real native producer.
    for (auto &pixel : pixels_) {
      pixel[3u] = 0.f;
    }
    const auto anchor = palette_anchor_point();
    pixels_[native_index(anchor[0u], anchor[1u])] = {.75f, .75f, .75f, 1.f};
    pixels_[native_index(anchor[0u] + 2u, anchor[1u])] = {0.f, 0.f, 0.f, 1.f};
    ASSERT_TRUE(run());
    ASSERT_EQ(std::count(result_.support.begin(), result_.support.end(), 1u), 1);
    ASSERT_EQ(params_words_[v2::subtitle_condition_param_target_word], 0u);
    for (const auto base_value : {-0.03f, 0.03f}) {
      SCOPED_TRACE(base_value);
      const std::vector<float> base(std::size_t(field_w_) * field_h_, base_value);
      replay_fields_t fields;
      ASSERT_TRUE(run_replay_conditioner(base, fields));
      ASSERT_EQ(fields.control, std::vector<std::uint32_t> {1u});
      const volatile float gap = std::abs(base_value) * static_cast<float>(source_w_);
      std::size_t pins = 0u, partial = 0u, already_feasible = 0u, infinite_rows = 0u;
      double maximum_guard_change = 0.0;
      for (std::uint32_t y = 0u; y < field_h_; ++y) {
        const bool inactive_row = std::isinf(fields.q[std::size_t(y) * field_w_ + content_[0u]]);
        infinite_rows += inactive_row ? 1u : 0u;
        for (std::uint32_t x = 0u; x < field_w_; ++x) {
          const auto i = std::size_t(y) * field_w_ + x;
          const float distance = fields.q[i];
          ASSERT_TRUE((std::isfinite(distance) && distance >= 0.f) || (std::isinf(distance) && distance > 0.f));
          float expected_weight = 0.f, expected_proposal = base_value;
          if (distance == 0.f) {
            ++pins;
            expected_weight = 1.f;
            expected_proposal = 0.f;
            EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.blend[i]), 0u);
          } else if (distance < gap) {
            ++partial;
            const volatile float ratio = distance / gap;
            expected_weight = 1.f - ratio;
            const volatile float normalized = distance / static_cast<float>(source_w_);
            expected_proposal = expected_weight == 1.f ? 0.f :
                                expected_weight == 0.f ? base_value :
                                base_value < 0.f       ? -normalized :
                                                         normalized;
            expected_proposal = std::clamp(expected_proposal, std::min(base_value, 0.f), std::max(base_value, 0.f));
          } else {
            ++already_feasible;
            EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.blend[i]), std::bit_cast<std::uint32_t>(base_value));
          }
          EXPECT_NEAR(fields.weight[i], expected_weight, v2::subtitle_contour_gaussian_weight_absolute_error_limit);
          EXPECT_NEAR(fields.blend[i], expected_proposal, 2.0e-8f);
          EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.vertical[i]), std::bit_cast<std::uint32_t>(fields.blend[i]));
          maximum_guard_change = std::max(maximum_guard_change, std::abs(static_cast<double>(fields.final[i]) - fields.blend[i]));
          EXPECT_NEAR(fields.final[i], fields.blend[i], 1.0e-7f);
          if (x != 0u) {
            EXPECT_LE(std::abs(fields.final[i] - fields.final[i - 1u]), v2::max_horizontal_slope / (content_[2u] - content_[0u]) + 1.0e-7f);
          }
          if (inactive_row) {
            EXPECT_TRUE(std::isinf(distance) && distance > 0.f);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.weight[i]), 0u);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.blend[i]), std::bit_cast<std::uint32_t>(base_value));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.final[i]), std::bit_cast<std::uint32_t>(base_value));
          }
        }
      }
      EXPECT_GT(pins, 0u);
      EXPECT_GT(partial, 0u);
      EXPECT_GT(already_feasible, 0u);
      EXPECT_GT(infinite_rows, 0u);
      EXPECT_LT(infinite_rows, field_h_);
      RecordProperty(base_value < 0.f ? "minimum_extension_negative_Base_guard_drift" : "minimum_extension_positive_Base_guard_drift", maximum_guard_change);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, GlobalCurrentAuthorityLeavesOffFootprintRowsAndPaddingProposalExactBase) {
    ASSERT_TRUE(resize(160u, 90u, 770u, 434u, {7u, 9u, 759u, 421u}));
    for (auto &pixel : pixels_) {
      pixel[3u] = 0.f;
    }
    const auto anchor = palette_anchor_point();
    pixels_[native_index(anchor[0u], anchor[1u])] = {.75f, .75f, .75f, 1.f};
    pixels_[native_index(anchor[0u] + 2u, anchor[1u])] = {0.f, 0.f, 0.f, 1.f};
    ASSERT_TRUE(run());
    ASSERT_EQ(std::count(result_.support.begin(), result_.support.end(), 1u), 1);
    std::vector<float> base(std::size_t(field_w_) * field_h_);
    for (std::uint32_t y = 0u; y < field_h_; ++y) {
      for (std::uint32_t x = 0u; x < field_w_; ++x) {
        const auto i = std::size_t(y) * field_w_ + x;
        base[i] = (x < content_[0u] || x >= content_[2u] || y < content_[1u] || y >= content_[3u]) ?
                    ((x + y) % 2u ? -0.f : .01234567f) :
                    .003f + 1e-6f * float(x % 13u) + 1e-7f * float(y % 7u);
      }
    }
    replay_fields_t fields;
    ASSERT_TRUE(run_replay_conditioner(base, fields));
    ASSERT_EQ(fields.control, std::vector<std::uint32_t> {1u})
      << "finite footprint rows below an infinite first content row still enable current conditioning";
    std::uint32_t first_active = field_h_, last_active = 0u, active_rows = 0u;
    std::size_t zero_budget_nodes = 0u, finite_padding_extensions = 0u;
    for (std::uint32_t y = 0u; y < field_h_; ++y) {
      const auto sy = std::clamp(y, content_[1u], content_[3u] - 1u);
      const bool active = std::isfinite(fields.q[std::size_t(sy) * field_w_ + content_[0u]]);
      if (active) {
        first_active = std::min(first_active, y);
        last_active = std::max(last_active, y);
        ++active_rows;
      }
      for (std::uint32_t x = 0u; x < field_w_; ++x) {
        const auto i = std::size_t(y) * field_w_ + x;
        const auto sx = std::clamp(x, content_[0u], content_[2u] - 1u);
        const auto local = std::size_t(sy) * field_w_ + sx;
        const auto local_budget = fields.q[local];
        const bool local_finite = std::isfinite(local_budget) && local_budget >= 0.f &&
                                  std::bit_cast<std::uint32_t>(local_budget) != 0x80000000u;
        const bool padding = x < content_[0u] || x >= content_[2u] || y < content_[1u] || y >= content_[3u];
        EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.vertical[i]), std::bit_cast<std::uint32_t>(fields.blend[i]));
        if (padding || !active) {
          EXPECT_TRUE(std::isinf(fields.q[i]) && fields.q[i] > 0.f);
        }
        if (!local_finite) {
          EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.weight[i]), 0u);
          EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.blend[i]), std::bit_cast<std::uint32_t>(base[i]));
        } else if (padding) {
          // Only a finite clamped content budget permits padding to extend its nearest
          // content proposal. Raw padding D remains infinity; no invalid-row fallback exists.
          EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.weight[i]), std::bit_cast<std::uint32_t>(fields.weight[local]));
          EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.blend[i]), std::bit_cast<std::uint32_t>(fields.blend[local]));
          finite_padding_extensions += std::bit_cast<std::uint32_t>(fields.blend[i]) !=
                                           std::bit_cast<std::uint32_t>(base[i]) ?
                                         1u :
                                         0u;
        }
        if (!active) {
          EXPECT_EQ(std::bit_cast<std::uint32_t>(fields.final[i]), std::bit_cast<std::uint32_t>(base[i]));
        }
        if (local_budget == 0.f) {
          ++zero_budget_nodes;
          EXPECT_EQ(fields.weight[i], 1.f);
        }
        if (active && x > content_[0u] && x < content_[2u]) {
          EXPECT_LE(std::abs(fields.final[i] - fields.final[i - 1u]), v2::max_horizontal_slope / (content_[2u] - content_[0u]) + 1e-7f);
        }
      }
    }
    EXPECT_GT(active_rows, 0u);
    EXPECT_GT(zero_budget_nodes, 0u);
    EXPECT_GT(finite_padding_extensions, 0u);
    EXPECT_GT(first_active, content_[1u]);
    EXPECT_LT(last_active + 1u, content_[3u]);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, OrdinarySearchKeepsOldTopGuardAndFullPublishedBottomWithoutBlanketProtection) {
    ASSERT_TRUE(resize(160u, 39u));
    clear_scene(.125f);
    auto cover = field_cover_;
    cover[0u] = 0u;
    cover[2u] = field_w_;
    const uint4_t core {0u, 223u, field_w_, 235u};
    ASSERT_EQ((2ull * (crop_top_ + 7u) + 1u) * field_h_,
              2ull * source_h_ * 217u);  // Exact old-top boundary, not a rounded neighboring row.
    current_covers({cover});
    current_cores({core});
    // Native y6 is above the retained old top guard; y7 equals its exact field boundary217.
    // Native y25 is below the old core-plus-half-height bottom, inside the published cover.
    // An independent current core sample supplies the palette. All three probes have P0
    // and the same own contrast, so only the current search bounds distinguish them.
    paint_palette_anchor({.75f, .75f, .75f, 1.f});
    const auto anchor = palette_anchor_point();
    pixels_[native_index(110u, 6u)] = {.75f, .75f, .75f, 1.f};
    pixels_[native_index(90u, 7u)] = {.75f, .75f, .75f, 1.f};
    pixels_[native_index(110u, 25u)] = {.75f, .75f, .75f, 1.f};
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(anchor[0u], anchor[1u], .9f);
    ASSERT_TRUE(run());
    expect_count(native_index(110u, 6u), 1u);
    EXPECT_EQ(result_.support[native_index(110u, 6u)], 0u);
    EXPECT_EQ(result_.support[native_index(90u, 7u)], 1u);
    EXPECT_EQ(result_.support[native_index(110u, 25u)], 1u);
    EXPECT_EQ(result_.support[native_index(70u, 6u)], 0u);
    expect_projection_oracle();
    // Same current geometry and strong detector response do not protect a uniform scene.
    clear_scene(.5f);
    make_authority(2u, 1100001u);
    current_covers({cover});
    current_cores({core});
    std::fill(probability_values_.begin(), probability_values_.end(), 1.f);
    ASSERT_TRUE(run());
    expect_count(native_index(110u, 6u), 1u);
    expect_no_mask();
    expect_evidence_exact_copy();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, CurrentLineCenterChoosesCorrectiveSideAndKeepsSignedOutsideReferences) {
    struct case_t {
      std::uint32_t left, right;
      std::int32_t reference;
      std::uint32_t width;
      bool nonzero;
      bool left_mask, right_mask;
    };

    const std::array cases {
      case_t {90u, 120u, 160, 160u, false, true, false},
      case_t {40u, 70u, 160, 160u, false, false, true},
      case_t {60u, 100u, 160, 160u, false, false, false},  // all-zero SCL1 uses the source center.
      case_t {60u, 100u, -160, 480u, true, true, false},  // a valid reference center outside the left edge.
      case_t {60u, 100u, 480, 480u, true, false, true},  // a valid reference center outside the right edge.
    };
    for (const auto &test_case : cases) {
      SCOPED_TRACE(::testing::Message() << test_case.left << ',' << test_case.right << " ref=" << test_case.reference);
      ASSERT_TRUE(resize(160u, 90u));
      clear_scene(.125f);
      auto core = field_cover_;
      core[0u] = source_x_edge(test_case.left);
      core[2u] = source_x_edge(test_case.right);
      current_cores({core});
      if (test_case.nonzero) {
        current_line_reference(test_case.reference, test_case.width);
      }
      paint_palette_anchor({.75f, .75f, .75f, 1.f});
      const auto anchor = palette_anchor_point();
      for (auto x : {22u, 80u, 138u}) {
        paint_point(x, .75f);
      }
      std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
      probability_at_native(anchor[0u], anchor[1u], .9f);
      ASSERT_TRUE(run());
      expect_count(point_index(22u), 1u);
      expect_count(point_index(138u), 1u);
      EXPECT_EQ(result_.support[point_index(22u)], test_case.left_mask ? 1u : 0u);
      EXPECT_EQ(result_.support[point_index(138u)], test_case.right_mask ? 1u : 0u);
      EXPECT_EQ(result_.support[point_index(80u)], 1u);
      expect_projection_oracle();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, OrdinaryInsetPreservesTinyCurrentCoreEqualToCover) {
    clear_scene(.125f);
    auto core = field_cover_;
    core[0u] = source_x_edge(55u);
    core[2u] = source_x_edge(57u);  // Narrower than two ordinary insets.
    current_covers({core});
    const auto anchor = palette_anchor_point(0u, 40u, 2u);
    pixels_[native_index(anchor[0u], anchor[1u])] = {.75f, .75f, .75f, 1.f};
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(anchor[0u], anchor[1u], .9f);
    ASSERT_TRUE(run());
    expect_count(native_index(anchor[0u], anchor[1u]), 1u);
    EXPECT_EQ(result_.support[native_index(anchor[0u], anchor[1u])], 1u);
    expect_projection_oracle();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, OrdinaryInsetKeepsAlreadyClampedContentEdges) {
    clear_scene(.125f);
    auto cover = field_cover_;
    cover[0u] = content_[0u];
    cover[2u] = content_[2u];
    auto core = field_cover_;
    core[0u] = source_x_edge(60u);
    core[2u] = source_x_edge(100u);
    current_covers({cover});
    current_cores({core});
    const auto anchor = palette_anchor_point();
    pixels_[native_index(anchor[0u], anchor[1u])] = {.75f, .75f, .75f, 1.f};
    paint_point(1u, .75f);
    paint_point(158u, .75f);
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(anchor[0u], anchor[1u], .9f);
    ASSERT_TRUE(run());
    EXPECT_EQ(result_.support[point_index(1u)], 1u);
    EXPECT_EQ(result_.support[point_index(158u)], 1u);
    expect_projection_oracle();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, OrdinaryLineUnionUsesFixedSlotVerticalIouWithoutTransitiveGrowth) {
    clear_scene(.125f);
    const uint4_t a {source_x_edge(40u), 340u, source_x_edge(60u), 360u};
    const uint4_t b {source_x_edge(80u), 346u, source_x_edge(100u), 366u};
    const uint4_t c {source_x_edge(110u), 352u, source_x_edge(130u), 372u};
    // A/B and B/C have IoU14/26>=1/2, A/C only8/32. SlotA may union B, never C via B.
    // Only A's cover contains either margin probe, preventing another slot from masking it.
    current_covers({field_cover_, b, c});
    current_cores({a, b, c});
    const auto anchor = palette_anchor_point();
    pixels_[native_index(anchor[0u], anchor[1u])] = {.75f, .75f, .75f, 1.f};
    for (auto x : {22u, 138u}) {
      pixels_[native_index(x, 9u)] = {.75f, .75f, .75f, 1.f};
    }
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(anchor[0u], anchor[1u], .9f);
    ASSERT_TRUE(run());
    EXPECT_EQ(result_.support[native_index(22u, 9u)], 0u);
    EXPECT_EQ(result_.support[native_index(138u, 9u)], 1u);
    expect_projection_oracle();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, RibbonKeepsCompletePublishedSearchDespiteOutsideReferenceAndBottomCore) {
    clear_scene(.125f);
    current_covers({uint4_t {content_[0u], constants_.field[2u], content_[2u], content_[3u]}}, true);
    current_line_reference(-160, 480u);
    const auto anchor = palette_anchor_point();
    pixels_[native_index(anchor[0u], anchor[1u])] = {.75f, .75f, .75f, 1.f};
    for (auto x : {1u, 158u}) {
      pixels_[native_index(x, 26u)] = {.75f, .75f, .75f, 1.f};
    }
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(anchor[0u], anchor[1u], .9f);
    ASSERT_TRUE(run());
    EXPECT_EQ(result_.support[native_index(1u, 26u)], 1u);
    EXPECT_EQ(result_.support[native_index(158u, 26u)], 1u);
    expect_projection_oracle();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, NonzeroMalformedCurrentReferenceContextCannotBorrowPositiveGlyphOrHistory) {
    for (auto fault = 0u; fault < 20u; ++fault) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      clear_scene(.125f);
      paint_glyph_point();
      current_line_reference(160, 160u);
      ASSERT_TRUE(run());
      ASSERT_EQ(result_.support[point_index()], 1u);
      expect_count(point_index(), 1u);
      const auto h = v2::subtitle_ocr_classifier_history_offset;
      switch (fault) {
        case 0u:
          ocr_words_[h] ^= 1u;
          break;
        case 1u:
          ocr_words_[h + 1u] ^= 1u;
          break;
        case 2u:
          ocr_words_[h + 2u] |= 2u;
          break;
        case 3u:
          ocr_words_[h + 4u] ^= 1u;
          break;
        case 4u:
          ocr_words_[h + 6u] ^= 1u;
          break;
        case 5u:
          ocr_words_[h + 8u] ^= 1u;
          break;
        case 6u:
          ocr_words_[h + 10u] += 1u;
          break;
        case 7u:
          ocr_words_[h + 12u] += 1u;
          break;
        case 8u:
          ocr_words_[h + 14u] += 1u;
          break;
        case 9u:
          ocr_words_[h + 16u] += 1u;
          break;
        case 10u:
          ocr_words_[h + 21u] = 0u;
          break;
        case 11u:
          ocr_words_[h + 21u] = v2::subtitle_contour_source_max_extent + 1u;
          break;
        case 12u:
          ocr_words_[h + 20u] = 161u;
          break;
        case 13u:
          ocr_words_[h + 20u] = std::bit_cast<std::uint32_t>(std::int32_t(-161));
          break;
        case 14u:
          ocr_words_[h + 22u] = v2::subtitle_contour_source_max_extent + 1u;
          break;
        case 15u:
          ocr_words_[h + 27u] = 0x7fc12345u;
          break;
        case 16u:
          ocr_words_[h + 30u] = 1u;
          break;
        case 17u:
          ocr_words_[h + 32u] = 1u;
          break;  // Votes without a trusted height/episode.
        case 18u:
          ocr_words_[h + 25u] = 1u;
          break;  // Gap clock without the armed flag.
        default:
          std::fill(ocr_words_.begin() + h, ocr_words_.end(), 0u);
          ocr_words_[h + 30u] = 1u;  // A partially zero noncanonical block is not zero fallback.
          break;
      }
      ASSERT_TRUE(run());
      expect_cleared_snapshot();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, PaletteReadsDenseSecondHalfAndUsesActualRepresentativeWithStrictRgb) {
    clear_scene(.125f);
    const auto late = palette_anchor_point(0u, 40u, 6u);  // 6*64+40 exceeds the256thread first batch.
    pixels_[native_index(late[0u], late[1u])] = {.75f, .75f, .75f, 1.f};
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(late[0u], late[1u], .9f);
    ASSERT_TRUE(run());
    ASSERT_EQ(result_.support[native_index(late[0u], late[1u])], 1u);

    ASSERT_TRUE(resize(160u, 90u));
    clear_scene(.125f);
    const pixel_t red {.9375f, .75f, .75f, 1.f}, green {.75f, .9375f, .75f, 1.f};
    const auto first = palette_anchor_point(0u, 4u, 2u), second = palette_anchor_point(0u, 20u, 2u);
    pixels_[native_index(first[0u], first[1u])] = red;
    pixels_[native_index(second[0u], second[1u])] = green;
    pixels_[point_index(80u)] = red;
    pixels_[point_index(100u)] = green;
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(first[0u], first[1u], .9f);
    probability_at_native(second[0u], second[1u], .9f);
    ASSERT_TRUE(run());
    // Both colors occupy RGBbin63 and tie in peak. The earlier real anchor represents it;
    // a manufactured mean would admit both colors. Strict .15 rejects the green target.
    EXPECT_EQ(result_.support[point_index(80u)], 1u);
    EXPECT_EQ(result_.support[point_index(100u)], 0u);

    ASSERT_TRUE(resize(160u, 90u));
    clear_scene(.125f);
    paint_palette_anchor({.75f, .75f, .75f, 1.f});
    const auto anchor = palette_anchor_point();
    paint_point(60u, .9375f);
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(anchor[0u], anchor[1u], .9f);
    ASSERT_TRUE(run());
    EXPECT_EQ(result_.support[point_index()], 0u) << "common-channel brightening beyond strict .15 has no foreground bypass";
  }

  TEST_F(HostSbsSubtitleContourGpuTest, PalettePrunesDimmerSameHueEvenOutsideTheOldRgbMergeDistance) {
    clear_scene(.125f);
    const auto bright = palette_anchor_point(0u, 10u, 2u), dim = palette_anchor_point(0u, 20u, 2u);
    pixels_[native_index(bright[0u], bright[1u])] = {.875f, .875f, .875f, 1.f};
    pixels_[native_index(dim[0u], dim[1u])] = {.5f, .5f, .5f, 1.f};
    paint_point(80u, .5f);
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(bright[0u], bright[1u], .9f);
    probability_at_native(dim[0u], dim[1u], .9f);
    ASSERT_TRUE(run());
    ASSERT_EQ(result_.support[native_index(bright[0u], bright[1u])], 1u);
    EXPECT_EQ(result_.support[point_index(80u)], 0u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, FirstCurrentObservationBootstrapsOneAndCopiesActualRgb) {
    paint_glyph_point();
    ASSERT_TRUE(observe(1u));
    EXPECT_EQ(result_.support[point_index()], 1u) << "current text evidence needs no RGB acquisition delay";
    expect_count(point_index(), 1u);
    EXPECT_EQ(result_.identity[2u], 1u);
    EXPECT_EQ(result_.identity[snapshot_valid_word], 1u);
    EXPECT_EQ(result_.features[point_index()], (pixel_t {.75f, .75f, .75f, feature_code(1u)}));
    expect_count(point_index(100u), 1u);
    expect_evidence_exact_copy();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, CurrentPaletteAnchorProbabilityThresholdIsStrictAndIndependentOfRgbHistory) {
    const auto threshold = v2::subtitle_ocr_active_probability_threshold;
    const std::array probabilities {0.f, threshold, std::nextafter(threshold, 1.f), 1.f};
    for (const auto probability : probabilities) {
      SCOPED_TRACE(probability);
      ASSERT_TRUE(resize(160u, 90u));
      clear_scene(.125f);
      paint_palette_anchor({.75f, .75f, .75f, 1.f});
      const auto anchor = palette_anchor_point();
      std::fill(probability_values_.begin(), probability_values_.end(), probability);
      ASSERT_TRUE(run());
      expect_count(native_index(anchor[0u], anchor[1u]), 1u);
      EXPECT_EQ(result_.support[native_index(anchor[0u], anchor[1u])], probability > threshold ? 1u : 0u);
      if (probability <= threshold) expect_no_mask();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, VaryingProbabilityContributorsUseSeparateFp32RoundingAtTheStrictThreshold) {
    // The oracle is only the contracted four-texel probability reduction, not geometry.
    // Double operations followed by explicit float conversion expose each binary32 round.
    const auto add = [](float a, float b) { return static_cast<float>(static_cast<double>(a) + b); };
    const auto subtract = [](float a, float b) { return static_cast<float>(static_cast<double>(a) - b); };
    const auto multiply = [](float a, float b) { return static_cast<float>(static_cast<double>(a) * b); };
    const auto divide = [](float a, float b) { return static_cast<float>(static_cast<double>(a) / b); };
    const auto threshold = v2::subtitle_ocr_active_probability_threshold;
    bool saw_below = false, saw_equal = false, saw_above = false;
    for (const auto &[width, height] : std::array {std::pair {162u, 91u}, std::pair {166u, 93u}}) {
      ASSERT_TRUE(resize(width, height));
      clear_scene(.5f); // No native contrast edge can hide an interpolation discrepancy.
      const auto px = 60u, py = body_y() + 5u;
      const auto detector_x = subtract(multiply(add(static_cast<float>(px), .5f),
        divide(static_cast<float>(v2::subtitle_ocr_output_width), static_cast<float>(source_w_))), .5f);
      const auto detector_y = subtract(multiply(add(static_cast<float>(py), .5f),
        divide(static_cast<float>(v2::subtitle_ocr_output_height), static_cast<float>(crop_h_))), .5f);
      const auto ax = static_cast<std::uint32_t>(std::floor(detector_x));
      const auto ay = static_cast<std::uint32_t>(std::floor(detector_y));
      const auto fx = subtract(detector_x, static_cast<float>(ax));
      const auto fy = subtract(detector_y, static_cast<float>(ay));
      ASSERT_LT(ax + 1u, v2::subtitle_ocr_output_width);
      ASSERT_LT(ay + 1u, v2::subtitle_ocr_output_height);
      const std::array locations {
        std::size_t(ay) * v2::subtitle_ocr_output_width + ax,
        std::size_t(ay) * v2::subtitle_ocr_output_width + ax + 1u,
        std::size_t(ay + 1u) * v2::subtitle_ocr_output_width + ax,
        std::size_t(ay + 1u) * v2::subtitle_ocr_output_width + ax + 1u
      };
      std::uint32_t frame = 0u;
      for (const auto magnitude : {1u, 4u}) {
        float low = threshold, high = threshold;
        for (auto step = 0u; step < magnitude; ++step) {
          low = std::nextafter(low, 0.f);
          high = std::nextafter(high, 1.f);
        }
        for (auto pattern = 0u; pattern < 16u; ++pattern) {
          SCOPED_TRACE(::testing::Message() << width << 'x' << height << " ULP=" << magnitude << " pattern=" << pattern);
          ++frame;
          make_authority(frame, 1000001u + (frame - 1u) * 100000u);
          std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
          std::array<float, 4u> values {};
          for (auto corner = 0u; corner < values.size(); ++corner) {
            values[corner] = (pattern & (1u << corner)) != 0u ? high : low;
            probability_values_[locations[corner]] = values[corner];
          }
          const auto upper = add(values[0u], multiply(subtract(values[1u], values[0u]), fx));
          const auto lower = add(values[2u], multiply(subtract(values[3u], values[2u]), fx));
          const auto expected = add(upper, multiply(subtract(lower, upper), fy));
          saw_below |= expected < threshold;
          saw_equal |= expected == threshold;
          saw_above |= expected > threshold;
          ASSERT_TRUE(run());
          float sampled {};
          ASSERT_TRUE(sample_probability_native(px, py, sampled));
          EXPECT_EQ(std::bit_cast<std::uint32_t>(sampled), std::bit_cast<std::uint32_t>(expected));
          expect_no_mask();
          expect_count(native_index(px, py), std::min(frame, confirmation_count));
        }
      }
    }
    EXPECT_TRUE(saw_below);
    EXPECT_TRUE(saw_equal);
    EXPECT_TRUE(saw_above);

    // This nonuniform pair crosses the boundary only with the contracted separate rounds:
    // fused multiply/add produces threshold exactly; separate operations produce its next float.
    ASSERT_TRUE(resize(162u, 91u));
    clear_scene(.5f);
    const auto px = 31u, py = body_y() + 5u;
    const auto dx = subtract(multiply(add(static_cast<float>(px), .5f), divide(960.f, 162.f)), .5f);
    const auto dy = subtract(multiply(add(static_cast<float>(py), .5f), divide(160.f, static_cast<float>(crop_h_))), .5f);
    const auto ax = static_cast<std::uint32_t>(std::floor(dx)), ay = static_cast<std::uint32_t>(std::floor(dy));
    constexpr float low = .12226862460374832f, high = .5886853933334351f;
    const auto expected = add(low, multiply(subtract(high, low), subtract(dx, static_cast<float>(ax))));
    ASSERT_EQ(std::bit_cast<std::uint32_t>(expected), std::bit_cast<std::uint32_t>(std::nextafter(threshold, 1.f)));
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    for (const auto y : {ay, ay + 1u}) {
      probability_values_[std::size_t(y) * v2::subtitle_ocr_output_width + ax] = low;
      probability_values_[std::size_t(y) * v2::subtitle_ocr_output_width + ax + 1u] = high;
    }
    ASSERT_TRUE(run());
    expect_count(native_index(px, py), 1u);
    float sampled {};
    ASSERT_TRUE(sample_probability_native(px, py, sampled));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(sampled), std::bit_cast<std::uint32_t>(expected));
    expect_no_mask(); // A strict palette-anchor sample still cannot protect uniform scenery.
  }

  TEST_F(HostSbsSubtitleContourGpuTest, ProbabilityWithdrawalClearsMatureRgbSupportWithCurrentCoversRetained) {
    clear_scene(.125f);
    paint_glyph_point(); // Strong current palette evidence is acquired before P is withdrawn.
    ASSERT_TRUE(mature_observations());
    expect_count(point_index(), confirmation_count);
    ASSERT_EQ(result_.support[point_index()], 1u);
    const auto covers = ocr_words_;
    for (std::uint32_t frame = 5u; frame <= 8u; ++frame) {
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
      ASSERT_TRUE(run());
      expect_count(point_index(), confirmation_count);
      EXPECT_EQ(params_words_[2u], 1u);
      EXPECT_TRUE(std::equal(ocr_words_.begin() + v2::subtitle_ocr_raw_box_offset,
        ocr_words_.end(), covers.begin() + v2::subtitle_ocr_raw_box_offset));
      expect_no_mask();
      expect_evidence_exact_copy();
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, SameShapeReplacementUsesCurrentTextEvidenceOnFirstObservationWithoutCue) {
    paint_glyph_point(60u, .75f);
    ASSERT_TRUE(mature_observations());
    const auto prior_probability = result_.probability_bits;
    // Differ from both the old glyph and its .15 background, so diagnostic neighbor
    // correspondence cannot legitimately carry a mature background reference.
    paint_glyph_point(60u, .5f);
    ASSERT_TRUE(observe(5u));
    EXPECT_EQ(result_.probability_bits, prior_probability);
    EXPECT_EQ(plane_words_[v2::subtitle_plane_history_reset_mask_word], 0u);
    expect_count(point_index(), 1u);
    EXPECT_EQ(result_.support[point_index()], 1u);
    expect_projection_oracle();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, NativeEdgeWithdrawalClearsProtectionOnTheSameObservationWithCurrentCoverRetained) {
    clear_scene(.125f);
    paint_point(60u, .75f);
    paint_palette_anchor({.75f, .75f, .75f, 1.f});
    const auto anchor = palette_anchor_point();
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(anchor[0u], anchor[1u], .9f);
    ASSERT_TRUE(run());
    expect_count(point_index(), 1u);
    EXPECT_EQ(result_.support[point_index()], 1u);
    const auto radius = v2::subtitle_contour_spatial_edge_radius_source_pixels;
    EXPECT_EQ(result_.support[point_index(60u + radius)], 0u) << "a bare contrast neighbor is not a current glyph";
    clear_scene(.125f);
    make_authority(2u, 1100001u);
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    ASSERT_TRUE(run());
    EXPECT_EQ(params_words_[2u], 1u);
    EXPECT_GT(result_.features[point_index()][3u], valid_rgb);
    expect_no_mask();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, NativeEdgeMayUseValidCurrentNeighborOutsideTheSearchCover) {
    clear_scene(.125f);
    const auto radius = v2::subtitle_contour_spatial_edge_radius_source_pixels;
    const auto edge_x = native_cover_[0u];
    paint_glyph_point(edge_x, .75f);
    const auto y = body_y() + 5u;
    // Suppress all interior cardinal contrasts; only the valid outside-left sample differs.
    for (const auto &[dx, dy] : std::array {std::pair {2,0}, std::pair {0,2}, std::pair {0,-2}})
      pixels_[native_index(edge_x + dx, y + dy)] = {.75f,.75f,.75f,1.f};
    const auto anchor = palette_anchor_point();
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    probability_at_native(anchor[0u], anchor[1u], .9f);
    ASSERT_TRUE(run());
    EXPECT_EQ(result_.support[point_index(edge_x)], 1u);
    EXPECT_EQ(result_.support[point_index(edge_x - radius)], 0u) << "the outside neighbor grants no search authority to itself";
    expect_count(point_index(edge_x - radius), 0u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, NativeDarkOutlineAtSixPixelOffsetNeedsCurrentPaletteAndOwnContrast) {
    ASSERT_TRUE(resize(160u, 90u));
    clear_scene(.5f);
    const auto anchor = palette_anchor_point();
    const std::array<std::uint32_t, 2u> target {anchor[0u] + 6u, anchor[1u]};
    const auto point = native_index(target[0u], target[1u]);
    paint_palette_anchor({.75f, .75f, .75f, 1.f});
    pixels_[point] = {.125f, .125f, .125f, 1.f};
    const auto current_palette_probability = [&]() {
      std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
      probability_at_native(anchor[0u], anchor[1u], .9f);
    };
    current_palette_probability();
    ASSERT_TRUE(run());
    expect_count(point, 1u);
    ASSERT_EQ(result_.support[native_index(anchor[0u], anchor[1u])], 1u);
    // Dark .125 is far outside the .75 foreground palette. Its only admitted route is the
    // six-pixel outline probe; its own +/-2 contrast sees .5, which is not a palette match.
    ASSERT_EQ(result_.support[point], 1u) << "the dark-outline branch must execute on the first current observation";
    expect_projection_oracle();

    make_authority(2u, 1100001u);
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    ASSERT_TRUE(run());
    expect_count(point, 2u);
    EXPECT_EQ(result_.support[point], 0u) << "unchanged RGB/history cannot substitute for the current font palette";

    make_authority(3u, 1200001u);
    current_palette_probability();
    ASSERT_TRUE(run());
    expect_count(point, 3u);
    ASSERT_EQ(result_.support[point], 1u) << "re-establish the same positive input before withdrawing its own edge";

    make_authority(4u, 1300001u);
    current_palette_probability();
    for (const auto &[dx, dy] : std::array {std::pair {-2, 0}, std::pair {2, 0}, std::pair {0, -2}, std::pair {0, 2}}) {
      pixels_[native_index(static_cast<std::uint32_t>(int(target[0u]) + dx), static_cast<std::uint32_t>(int(target[1u]) + dy))] = {.125f, .125f, .125f, 1.f};
    }
    ASSERT_TRUE(run());
    expect_count(point, confirmation_count);
    ASSERT_EQ(result_.support[native_index(anchor[0u], anchor[1u])], 1u);
    EXPECT_EQ(result_.support[point], 0u) << "a current palette and six-pixel foreground probe still require the target's own contrast";
    expect_projection_oracle();
    expect_evidence_exact_copy();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, ZeroProbabilityPatchesKeepDiagnosticHistoryButCannotInventCurrentPalette) {
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
      std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
      ASSERT_TRUE(run());
      for (std::size_t k = 0u; k < colors.size(); ++k) {
        const auto x = 35u + 25u * std::uint32_t(k);
        expect_flat_patch({x - 3u, y - 3u, x + 4u, y + 4u}, frame, colors[k]);
      }
      // Each background change exceeds .08; subsequent RGB is separated from every patch anchor.
      expect_count(point_index(100u), 1u);
      EXPECT_EQ(result_.features[point_index(100u)], (pixel_t {background, background, background, feature_code(1u)}));
      expect_no_mask(); // No strong current core sample can teach a font palette.
      expect_evidence_exact_copy();
    }
    expect_no_mask();
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
      paint_glyph_point(60u, colors[frame - 1u]);
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), counts[frame - 1u]);
      const float anchor = frame <= 2u ? .5f : .625f;
      EXPECT_EQ(result_.features[point_index()][0u], anchor);
      EXPECT_EQ(result_.support[point_index()], 1u) << "RGB drift does not delay current text evidence";
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
      paint_palette_anchor(colors[frame - 1u]);
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(x), frame);
      for (std::size_t lane = 0u; lane < 3u; ++lane) {
        EXPECT_EQ(result_.features[point_index(x)][lane], colors[0u][lane]);
      }
    }
    EXPECT_EQ(result_.support[point_index(50u)], 0u) << "uniform former glyph background has no current stroke";
    EXPECT_EQ(result_.support[point_index(56u)], 1u);
    EXPECT_EQ(result_.features[point_index(50u)], (pixel_t {.375f, .375f, .375f, feature_code(1u)}));
    // This is an explicit <=2 per-observation correspondence chain, not four observations
    // of one absolute source coordinate or semantic proof that the tracked color is text.
  }

  TEST_F(HostSbsSubtitleContourGpuTest, MovementBeyondJitterRestartsFourObservationConfirmation) {
    clear_scene(0.f);
    paint_glyph_point(50u);
    ASSERT_TRUE(observe(1u));
    for (std::uint32_t frame = 2u; frame <= 5u; ++frame) {
      clear_scene(.125f * float(frame - 1u));
      paint_glyph_point(60u);
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame - 1u);
    }
    EXPECT_EQ(result_.support[point_index(50u)], 0u) << "old P cannot protect the now-uniform background";
    EXPECT_EQ(result_.support[point_index(60u)], 1u);
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
    paint_glyph_point(60u, .5f);
    ASSERT_TRUE(observe(1u));
    ASSERT_TRUE(observe(2u));
    paint_glyph_point(61u, .625f);  // Too far from the old .5 anchor: a genuine new streak.
    ASSERT_TRUE(observe(3u));
    expect_count(point_index(60u), 3u);
    expect_count(point_index(61u), 1u);

    paint_glyph_point(60u, .578125f);  // .078125 from own anchor, .046875 from new neighbour.
    ASSERT_TRUE(observe(4u));
    expect_count(point_index(60u), 4u);
    EXPECT_EQ(result_.features[point_index(60u)][0u], .5f);
    EXPECT_EQ(result_.support[point_index(60u)], 1u);
    expect_count(point_index(61u), 2u);
    EXPECT_EQ(result_.support[point_index(61u)], 1u);
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

  TEST_F(HostSbsSubtitleContourGpuTest, StableUniformZeroProbabilityBackgroundNeverAcquiresProtection) {
    clear_scene(.5f);
    for (std::uint32_t frame = 1u; frame <= 8u; ++frame) {
      make_authority(frame, 1000001u + (frame - 1u) * 100000u);
      std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
      ASSERT_TRUE(run());
      expect_count(point_index(), std::min(frame, confirmation_count));
      expect_count(point_index(5u), 0u);
      expect_no_mask();
      expect_evidence_exact_copy();
    }
    expect_projection_oracle();
    paint_point(60u, .5625f);
    make_authority(9u, 1800001u);
    std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
    ASSERT_TRUE(run());
    expect_count(point_index(), confirmation_count);
    EXPECT_EQ(result_.features[point_index()], (pixel_t {.5f, .5f, .5f, feature_code(confirmation_count)}))
      << "diagnostic RGB correspondence can persist without granting protection";
    expect_no_mask();
    expect_evidence_exact_copy();
  }

  TEST_F(HostSbsSubtitleContourGpuTest, DiagnosticCountsNeedNoNativeContrastButCurrentSupportDoes) {
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
      EXPECT_EQ(result_.support[point_index()], 0u);
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
      paint_glyph_point();
      ASSERT_TRUE(run());
      ASSERT_EQ(result_.support[point_index()], 1u);
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
    paint_glyph_point();
    ASSERT_TRUE(mature_observations(1u, 3u));
    ASSERT_EQ(result_.support[point_index()], 1u) << "the exact valid pre-fault input must protect a current glyph";
    make_authority(4u, 1300001u);
    without_current_authority();
    ASSERT_TRUE(run());
    expect_count(point_index(), 0u);
    expect_no_mask();
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
    EXPECT_EQ(result_.support[point_index()], 0u);
    ASSERT_TRUE(observe(10u));
    expect_count(point_index(), 1u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, ExactDuplicatesDoNotAdvanceOrBorrowNeighborCounts) {
    paint_glyph_point();
    for (std::uint32_t frame = 1u; frame <= 4u; ++frame) {
      ASSERT_TRUE(observe(frame));
      ASSERT_EQ(result_.support[point_index()], 1u);
      expect_count(point_index(), frame);
      const auto held = result_;
      ASSERT_TRUE(run());
      expect_tuple_equal(held);
      expect_count(point_index(), frame);
    }
    paint_glyph_point(62u);  // Same color lies within jitter but source frame/time did not advance.
    ASSERT_TRUE(run());
    expect_count(point_index(62u), 0u);
    EXPECT_EQ(result_.support[point_index(62u)], 0u); // Fresh RGB/palette cannot bypass duplicate count0.
    expect_count(point_index(), 4u);
  }

  TEST_F(HostSbsSubtitleContourGpuTest, DuplicateAfterLegalEmptyCannotDiscoverCurrentCoverPixels) {
    paint_glyph_point();
    current_covers({});
    ASSERT_TRUE(run());
    expect_count(point_index(), 0u);
    expect_no_mask();
    EXPECT_EQ(result_.identity[snapshot_valid_word], 1u);
    make_authority(1u, 1000001u); // Same genuine source observation, now a new current cover.
    ASSERT_TRUE(run());
    expect_count(point_index(), 0u);
    expect_no_mask();
    for (std::uint32_t frame = 2u; frame <= 5u; ++frame) {
      ASSERT_TRUE(observe(frame));
      expect_count(point_index(), frame - 1u);
      EXPECT_EQ(result_.support[point_index()], 1u);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, MalformedControllerCannotAuthorizeFourthConfirmation) {
    for (std::uint32_t fault = 0u; fault < 9u; ++fault) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      paint_glyph_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      ASSERT_EQ(result_.support[point_index()], 1u) << "the exact valid pre-fault input must protect a current glyph";
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
      expect_no_mask();
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
  TEST_F(HostSbsSubtitleContourGpuTest, CurrentFlagsKeepRgbCountsButLowProbabilityCannotLearnCurrentPalette) {
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
      // A flag preserves diagnostic same-pixel history. With all current P low,
      // contrast alone cannot recreate the current font palette.
      std::fill(probability_values_.begin(), probability_values_.end(), frame % 2u ? 0.f : .015625f);
      ASSERT_TRUE(run());
      for (std::size_t k = 0u; k < colors.size(); ++k) {
        const auto x = 35u + 25u * std::uint32_t(k);
        expect_flat_patch({x - 3u, y - 3u, x + 4u, y + 4u}, confirmation_count, colors[k]);
      }
      expect_no_mask();
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
      paint_glyph_point();
      ASSERT_TRUE(mature_observations());
      ASSERT_EQ(result_.support[point_index()], 1u) << "the exact valid pre-fault input must protect a current glyph";
      auto core = field_cover_;
      core[0u] = source_x_edge(50u);
      core[2u] = source_x_edge(70u);
      make_authority(5u, 1400001u);
      current_cores({core});
      paint_palette_anchor({.75f,.75f,.75f,1.f}); // Re-anchor inside the actual current narrow core.
      ASSERT_TRUE(run());
      ASSERT_EQ(result_.support[point_index()], 1u);
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
      paint_glyph_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      ASSERT_EQ(result_.support[point_index()], 1u) << "the exact valid pre-fault input must protect a current glyph";
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
      for (std::uint32_t slot = 0u; slot < covers.size(); ++slot)
        paint_palette_anchor({.9375f,.9375f,.9375f,1.f}, slot);
      ASSERT_TRUE(run());
      for (std::uint32_t slot = 0u; slot < covers.size(); ++slot) expect_count(point_index(27u + 15u * slot), frame);
      std::reverse(covers.begin(), covers.end());
    }
    for (std::uint32_t slot = 0u; slot < covers.size(); ++slot) {
      EXPECT_EQ(result_.support[point_index(27u + 15u * slot)], 1u);
    }
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
      paint_glyph_point();
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
      EXPECT_EQ(result_.support[point_index()], 1u) << "current evidence is independent of diagnostic clock resets";
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
      paint_glyph_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      ASSERT_EQ(result_.support[point_index()], 1u) << "the exact valid pre-fault input must protect a current glyph";
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
        paint_glyph_point();
        ASSERT_TRUE(mature_observations(1u, 3u));
        ASSERT_EQ(result_.support[point_index()], 1u) << "the exact valid pre-fault input must protect a current glyph";
        make_authority(4u, 1300001u);
        std::fill(probability_values_.begin(), probability_values_.end(), 0.f);
        probability_values_[contributors[contributor]] = invalid;
        ASSERT_TRUE(run());
        expect_count(point_index(), 0u);
        EXPECT_EQ(result_.support[point_index()], 0u);
        expect_evidence_exact_copy();  // Raw invalid bits remain diagnostic evidence, not alpha.
      }
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, MalformedHeadersStopConfirmationWhilePlaneFaultKeepsRgbSnapshot) {
    for (const auto packed : {0u}) {
      SCOPED_TRACE(packed);
      ASSERT_TRUE(resize(160u, 90u));
      paint_glyph_point();
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
      paint_glyph_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      ASSERT_EQ(result_.support[point_index()], 1u) << "the exact valid pre-fault input must protect a current glyph";
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
          plane_words_[v2::subtitle_plane_warmup_word] = 16u;
          break;
        default:
          // Formerly legal countdown/cue values cannot revive retired whole-cover protection.
          plane_words_[v2::subtitle_plane_warmup_word] = (1u + (fault - 12u) % 3u) |
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

  TEST_F(HostSbsSubtitleContourGpuTest, CurrentProbabilityWithdrawalClearsGaussianDespiteMatureRgbAndRetainedCover) {
    // One finite source pixel isolates current text evidence from diagnostic RGB history.
    // These are authored source/OCR inputs; all count, q and M outputs below come
    // from the production locator, temporal, history, distance and condition shader entries.
    ASSERT_TRUE(resize(160u, 90u));
    content_ = {0u, 0u, source_w_, source_h_}; // 1:1 content nodes expose exact q=0 at the source pixel.
    for (auto &pixel : pixels_) pixel[3u] = 0.f;
    make_authority(1u, 1000001u);
    const auto anchor = palette_anchor_point();
    const auto ink_index = native_index(anchor[0u], anchor[1u]);
    pixels_[ink_index] = {.75f,.75f,.75f,1.f};
    pixels_[native_index(anchor[0u] + 2u, anchor[1u])] = {0.f,0.f,0.f,1.f};
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
    const auto point_y = crop_top_ + anchor[1u];
    const auto point_node = std::size_t(point_y) * field_w_ + anchor[0u];
    const auto distant_node = std::size_t(point_y) * field_w_ + 100u;
    for (std::uint32_t frame = 1u; frame <= 10u; ++frame) {
      SCOPED_TRACE(frame);
      // Epoch2 old text is mature before a valid epoch3 cut. No input-domain reset is used.
      const auto epoch = frame < 5u ? 2u : 3u;
      if (frame == 6u) pixels_[ink_index] = {.5f,.5f,.5f,1.f};
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
      expect_count(ink_index, expected_count);
      const auto remaining = plane_words_[v2::subtitle_plane_warmup_word] & v2::subtitle_plane_warmup_remaining_mask;
      EXPECT_EQ(plane_words_[v2::subtitle_plane_warmup_word], 0u);
      if (frame >= 5u) {
        EXPECT_EQ(remaining, 0u);
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
      if (frame <= 5u) {
        EXPECT_EQ(distances[point_node], 0.f);
        EXPECT_EQ(result_.support[ink_index], 1u);
        EXPECT_EQ(weights[point_node], 1.f);
      } else {
        EXPECT_TRUE(std::isinf(distances[point_node]));
        EXPECT_EQ(result_.support[ink_index], 0u);
        EXPECT_EQ(weights[point_node], 0.f);
        EXPECT_EQ(proposals[point_node], base_values[point_node]);
      }
      EXPECT_EQ(weights[distant_node], 0.f);
      EXPECT_EQ(proposals[distant_node], base_values[distant_node]);
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, InvalidSealedGeometryCannotPublishPriorRgbOrMask) {
    for (std::uint32_t fault = 0u; fault < 6u; ++fault) {
      SCOPED_TRACE(fault);
      ASSERT_TRUE(resize(160u, 90u));
      paint_glyph_point();
      ASSERT_TRUE(mature_observations(1u, 3u));
      ASSERT_EQ(result_.support[point_index()], 1u) << "the exact valid pre-fault input must protect a current glyph";
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
    paint_glyph_point();
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

  TEST_F(HostSbsSubtitleContourGpuTest, LargeNativeFlatPatchKeepsHistoryWhileOnlyCurrentEdgesSeedSupport) {
    ASSERT_TRUE(resize(2536u, 1427u));
    ASSERT_GE(field_cover_[2u] - field_cover_[0u], 2u * (field_cover_[3u] - field_cover_[1u]));
    const auto top = native_cover_[1u] + 50u;
    const pixel_t original {.3125f, .75f, .5625f, 1.f};
    for (std::uint32_t frame = 1u; frame <= 3u; ++frame) {
      clear_scene(.125f * float(frame - 1u));
      paint({1000u, top, 1100u, top + 60u}, original);
      paint_palette_anchor(original);
      ASSERT_TRUE(observe(frame));
      expect_flat_patch({1000u, top, 1100u, top + 60u}, frame, original);
      EXPECT_EQ(result_.support[native_index(1050u, top + 30u)], 0u);
      EXPECT_EQ(result_.support[native_index(1000u, top + 30u)], 1u);
    }
    clear_scene(.375f);
    paint({1002u, top + 2u, 1102u, top + 62u}, {.28125f, .71875f, .53125f, 1.f});
    paint_palette_anchor({.28125f,.71875f,.53125f,1.f});
    ASSERT_TRUE(observe(4u));
    expect_flat_patch({1002u, top + 2u, 1102u, top + 62u}, 4u, original);
    EXPECT_LT(pin_count(result_.support), 6000u) << "a flat interior cannot become a binary glyph seed from P alone";
    EXPECT_EQ(result_.support[native_index(1050u, top + 30u)], 0u);
    EXPECT_EQ(result_.support[native_index(1002u, top + 30u)], 1u);
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
    make_authority(1u, 1000001u);
    paint_glyph_point();
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
      paint_glyph_point();
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
      EXPECT_EQ(result_.support[point_index()], 1u) << "malformed diagnostic history cannot delay valid current text evidence";
    }
  }

  TEST_F(HostSbsSubtitleContourGpuTest, InvalidNarrowRibbonAtEpochZeroCannotBorrowThreeObservationHistory) {
    paint_glyph_point();
    for (std::uint32_t frame = 1u; frame <= 3u; ++frame) {
      make_authority(frame, 1000001u + (frame - 1u) * 100000u, false, 7u, 0u);
      ASSERT_TRUE(run());
      ASSERT_EQ(result_.support[point_index()], 1u);
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
    paint_palette_anchor({.75f,.75f,.75f,1.f});
    ASSERT_TRUE(run());
    expect_cleared_snapshot();
    EXPECT_EQ(result_.identity[2u], 0u);
    EXPECT_EQ(result_.identity[snapshot_valid_word], 0u);
    EXPECT_EQ(result_.identity[3u], 0u);
  }
}  // namespace
#endif
