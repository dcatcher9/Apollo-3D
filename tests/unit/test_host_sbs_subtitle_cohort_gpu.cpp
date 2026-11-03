/**
 * @file tests/unit/test_host_sbs_subtitle_cohort_gpu.cpp
 * @brief Opt-in native GPU cue-lifetime prototype, not production mask authority.
 *
 * Authored tests exercise state ownership. Saved replay consumes current native evidence,
 * never a CPU eligibility bitmap. Readback/export is diagnostic and does not drive a pass.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/crypto.h"
  #include "src/depth_coordinate_v2.h"
  #include "src/host_sbs_shader_cache.h"
  #include "src/host_sbs_subtitle_cohort.h"

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstddef>
  #include <cstdint>
  #include <cstdlib>
  #include <cstring>
  #include <d3d11.h>
  #include <filesystem>
  #include <fstream>
  #include <limits>
  #include <nlohmann/json.hpp>
  #include <stdexcept>
  #include <string>
  #include <string_view>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace cv2 = models::depth_coordinate_v2;
  namespace cohort = host_sbs_subtitle_cohort;
  using cohort_rect_t = std::array<std::uint32_t, 4u>;
  using cohort_half_t = std::array<std::uint16_t, 4u>;

  std::vector<char> cohort_read_bytes(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0) throw std::runtime_error("Cannot read cohort input: " + path.string());
    std::vector<char> result(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    if (!result.empty()) file.read(result.data(), static_cast<std::streamsize>(result.size()));
    if (!file) throw std::runtime_error("Incomplete cohort input: " + path.string());
    return result;
  }

  std::string cohort_sha256(const void *data, std::size_t bytes) {
    const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), bytes));
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (auto byte : digest) {
      result.push_back(digits[byte >> 4u]);
      result.push_back(digits[byte & 15u]);
    }
    return result;
  }

  std::uint16_t cohort_half(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16u) & 0x8000u);
    const auto exponent = (bits >> 23u) & 255u;
    auto fraction = bits & 0x7fffffu;
    if (exponent == 255u) return sign | (fraction ? 0x7e00u : 0x7c00u);
    const int e = static_cast<int>(exponent) - 127 + 15;
    if (e >= 31) return sign | 0x7c00u;
    if (e <= 0) {
      if (e < -10) return sign;
      fraction |= 0x800000u;
      const auto shift = static_cast<unsigned>(14 - e);
      return sign | static_cast<std::uint16_t>((fraction + (1u << (shift - 1u)) - 1u + ((fraction >> shift) & 1u)) >> shift);
    }
    fraction += 0xfffu + ((fraction >> 13u) & 1u);
    return sign | static_cast<std::uint16_t>((std::uint32_t(e) << 10u) + (fraction >> 13u));
  }

  float cohort_float(std::uint16_t value) {
    const auto exponent = (value >> 10u) & 31u, mantissa = value & 1023u;
    const float magnitude = exponent == 0u ? std::ldexp(float(mantissa), -24) :
      exponent == 31u ? (mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity()) :
      std::ldexp(1.f + float(mantissa) / 1024.f, int(exponent) - 15);
    return (value & 0x8000u) != 0u ? -magnitude : magnitude;
  }

  struct cohort_texture_t {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };
  struct cohort_buffer_t {
    ComPtr<ID3D11Buffer> buffer;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };

  bool cohort_texture(ID3D11Device *device, UINT width, UINT height, DXGI_FORMAT format, UINT flags, cohort_texture_t &out) {
    out = {};
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width; desc.Height = height; desc.MipLevels = desc.ArraySize = 1u;
    desc.Format = format; desc.SampleDesc.Count = 1u;
    desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = flags;
    return SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &out.texture)) &&
      (!(flags & D3D11_BIND_SHADER_RESOURCE) || SUCCEEDED(device->CreateShaderResourceView(out.texture.Get(), nullptr, &out.srv))) &&
      (!(flags & D3D11_BIND_UNORDERED_ACCESS) || SUCCEEDED(device->CreateUnorderedAccessView(out.texture.Get(), nullptr, &out.uav)));
  }

  bool cohort_buffer(ID3D11Device *device, UINT words, UINT flags, cohort_buffer_t &out) {
    out = {};
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = words * 4u; desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = flags;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride = 4u;
    return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &out.buffer)) &&
      (!(flags & D3D11_BIND_SHADER_RESOURCE) || SUCCEEDED(device->CreateShaderResourceView(out.buffer.Get(), nullptr, &out.srv))) &&
      (!(flags & D3D11_BIND_UNORDERED_ACCESS) || SUCCEEDED(device->CreateUnorderedAccessView(out.buffer.Get(), nullptr, &out.uav)));
  }

  template<class T>
  bool cohort_read_texture(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Texture2D *input, std::vector<T> &out) {
    D3D11_TEXTURE2D_DESC desc {}; input->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING; desc.BindFlags = 0u;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; desc.MiscFlags = 0u;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), input);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
    out.resize(std::size_t(desc.Width) * desc.Height);
    for (UINT y = 0; y < desc.Height; ++y) {
      std::memcpy(out.data() + std::size_t(y) * desc.Width,
                  static_cast<const char *>(mapped.pData) + std::size_t(y) * mapped.RowPitch,
                  std::size_t(desc.Width) * sizeof(T));
    }
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  bool cohort_read_buffer(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Buffer *input, std::vector<std::uint32_t> &out) {
    D3D11_BUFFER_DESC desc {}; input->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING; desc.BindFlags = 0u;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; desc.MiscFlags = desc.StructureByteStride = 0u;
    ComPtr<ID3D11Buffer> staging;
    if (FAILED(device->CreateBuffer(&desc, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), input);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
    out.resize(desc.ByteWidth / 4u);
    std::memcpy(out.data(), mapped.pData, desc.ByteWidth);
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  struct cohort_result_t {
    std::vector<std::uint32_t> state, owner, birth_rgb, next_state, next_owner, next_rgb, eligibility, filtered;
    std::vector<std::uint32_t> previous_p, previous_identity;
    std::vector<std::uint32_t> previous_support;
    std::vector<cohort_half_t> previous_source;
    bool operator==(const cohort_result_t &) const = default;
  };

  class HostSbsSubtitleCohortGpuTest: public ::testing::Test {
  protected:
    void SetUp() override {
      shader_root_ = SUNSHINE_SHADERS_DIR;
      std::string expected;
      if (std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) == "SavedNativeSequenceReplay") {
        if (const auto *path = std::getenv("SUNSHINE_TEST_COHORT_CASES")) {
          const auto bytes = cohort_read_bytes(path);
          const auto spec = nlohmann::json::parse(bytes.begin(), bytes.end());
          driver_name_ = spec.value("driver", std::string("warp"));
          ASSERT_TRUE(driver_name_ == "warp" || driver_name_ == "hardware");
          shader_root_ = std::filesystem::path(path).parent_path() / spec.at("shader_root").get<std::string>();
          expected = spec.at("prototype_closure_sha256").get<std::string>();
          ASSERT_EQ(expected.size(), 64u);
        }
      }
      const D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr,
        driver_name_ == "hardware" ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP,
        nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &device_, &actual, &context_)));
      ASSERT_EQ(actual, D3D_FEATURE_LEVEL_11_0);
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {
        cache::shader_spec {"host_sbs_subtitle_cohort_cs.hlsl", "prepare_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_cohort_cs.hlsl", "members_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_cohort_cs.hlsl", "filter_main", "cs_5_0"}
      };
      const auto sources = cache::snapshot_sources(shader_root_, specs);
      closure_ = cache::source_closure_sha256(sources);
      if (!expected.empty()) {
        ASSERT_EQ(closure_, expected) << "Prototype closure pin mismatch";
      }
      for (std::size_t i = 0; i < specs.size(); ++i) {
        const auto code = cache::get(sources, specs[i]);
        ASSERT_TRUE(code && !code->empty()) << specs[i].entrypoint;
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &shaders_[i])));
      }
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(cohort::constants_t); desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &constants_buffer_)));
      desc.ByteWidth = 12u; desc.BindFlags = 0u; desc.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &prepare_arguments_)));
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &pixel_arguments_)));
      ASSERT_TRUE(resize(160u, 90u));
    }

    void TearDown() override { unbind(); }

    void unbind() {
      if (!context_) return;
      const std::array<ID3D11ShaderResourceView *, 15u> reads {};
      const std::array<ID3D11UnorderedAccessView *, 8u> writes {};
      context_->CSSetShaderResources(0u, static_cast<UINT>(reads.size()), reads.data());
      context_->CSSetUnorderedAccessViews(0u, static_cast<UINT>(writes.size()), writes.data(), nullptr);
    }

    bool resize(UINT width, UINT height, UINT field_width = 0u, UINT field_height = 0u, cohort_rect_t content = {}) {
      unbind();
      width_ = width; height_ = height;
      crop_height_ = std::min(height, (width + 5u) / 6u); crop_top_ = height - crop_height_;
      field_width_ = field_width ? field_width : cv2::model_calibrated_shapes[0u].width;
      field_height_ = field_height ? field_height : cv2::model_calibrated_shapes[0u].height;
      content_ = content == cohort_rect_t {} ? cohort_rect_t {0u, 0u, field_width_, field_height_} : content;
      const UINT both = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      const UINT read = D3D11_BIND_SHADER_RESOURCE;
      if (!cohort_texture(device_.Get(), width_, crop_height_, DXGI_FORMAT_R16G16B16A16_FLOAT, read, current_source_) ||
          !cohort_texture(device_.Get(), width_, crop_height_, DXGI_FORMAT_R16G16B16A16_FLOAT, both, previous_source_) ||
          !cohort_texture(device_.Get(), width_, crop_height_, DXGI_FORMAT_R32_UINT, read, baseline_) ||
          !cohort_texture(device_.Get(), width_, crop_height_, DXGI_FORMAT_R32_UINT, both, previous_support_) ||
          !cohort_texture(device_.Get(), width_, crop_height_, DXGI_FORMAT_R32_UINT, both, old_owner_) ||
          !cohort_texture(device_.Get(), width_, crop_height_, DXGI_FORMAT_R32_UINT, both, old_rgb_) ||
          !cohort_texture(device_.Get(), width_, crop_height_, DXGI_FORMAT_R32_UINT, both, next_owner_) ||
          !cohort_texture(device_.Get(), width_, crop_height_, DXGI_FORMAT_R32_UINT, both, next_rgb_) ||
          !cohort_texture(device_.Get(), width_, crop_height_, DXGI_FORMAT_R32_UINT, both, eligibility_) ||
          !cohort_texture(device_.Get(), width_, crop_height_, DXGI_FORMAT_R32_UINT, both, filtered_) ||
          !cohort_buffer(device_.Get(), cv2::subtitle_ocr_output_width * cv2::subtitle_ocr_output_height, read, current_p_) ||
          !cohort_buffer(device_.Get(), cv2::subtitle_ocr_output_width * cv2::subtitle_ocr_output_height, both, previous_p_) ||
          !cohort_buffer(device_.Get(), cv2::subtitle_ocr_record_word_count, read, ocr_) ||
          !cohort_buffer(device_.Get(), cv2::subtitle_plane_state_word_count, read, plane_) ||
          !cohort_buffer(device_.Get(), cv2::subtitle_contour_history_word_count, read, current_identity_) ||
          !cohort_buffer(device_.Get(), cv2::subtitle_contour_history_word_count, both, previous_identity_) ||
          !cohort_buffer(device_.Get(), SBC_STATE_WORDS, both, old_state_) ||
          !cohort_buffer(device_.Get(), SBC_STATE_WORDS, both, next_state_)) return false;
      reset_state();
      author(1u, {});
      return true;
    }

    void reset_state() {
      unbind();
      const auto initial = cohort::initial_state();
      context_->UpdateSubresource(old_state_.buffer.Get(), 0u, nullptr, initial.data(), 0u, 0u);
      context_->UpdateSubresource(next_state_.buffer.Get(), 0u, nullptr, initial.data(), 0u, 0u);
      const UINT zero[4] {}, one[4] {1u, 1u, 1u, 1u};
      for (auto *view : {old_owner_.uav.Get(), old_rgb_.uav.Get(), next_owner_.uav.Get(), next_rgb_.uav.Get(), filtered_.uav.Get()})
        context_->ClearUnorderedAccessViewUint(view, zero);
      context_->ClearUnorderedAccessViewUint(eligibility_.uav.Get(), one);
      std::vector<std::uint32_t> zero_p(cv2::subtitle_ocr_output_width * cv2::subtitle_ocr_output_height);
      std::vector<std::uint32_t> zero_identity(cv2::subtitle_contour_history_word_count);
      std::vector<cohort_half_t> zero_source(std::size_t(width_) * crop_height_);
      std::vector<UINT> zero_support(std::size_t(width_) * crop_height_);
      context_->UpdateSubresource(previous_p_.buffer.Get(), 0u, nullptr, zero_p.data(), 0u, 0u);
      context_->UpdateSubresource(previous_identity_.buffer.Get(), 0u, nullptr, zero_identity.data(), 0u, 0u);
      context_->UpdateSubresource(previous_source_.texture.Get(), 0u, nullptr, zero_source.data(), width_ * sizeof(cohort_half_t), 0u);
      context_->UpdateSubresource(previous_support_.texture.Get(), 0u, nullptr, zero_support.data(), width_ * 4u, 0u);
      snapshot_frame_ = snapshot_time_ = 0u;
      result_ = {};
    }

    UINT row_edge(UINT detector_y) const {
      const auto numerator = std::uint64_t(crop_top_ * cv2::subtitle_ocr_output_height + detector_y * crop_height_) * (content_[3u] - content_[1u]);
      const auto denominator = std::uint64_t(height_) * cv2::subtitle_ocr_output_height;
      return content_[1u] + static_cast<UINT>((numerator + denominator - 1u) / denominator);
    }

    cohort_rect_t field_rect(cohort_rect_t native) const {
      const auto cw = content_[2u] - content_[0u], ch = content_[3u] - content_[1u];
      return {content_[0u] + static_cast<UINT>(std::uint64_t(native[0u]) * cw / width_),
        content_[1u] + static_cast<UINT>(std::uint64_t(crop_top_ + native[1u]) * ch / height_),
        content_[0u] + static_cast<UINT>((std::uint64_t(native[2u]) * cw + width_ - 1u) / width_),
        content_[1u] + static_cast<UINT>((std::uint64_t(crop_top_ + native[3u]) * ch + height_ - 1u) / height_)};
    }

    // Construct only typed current evidence; no shader decision is replicated here.
    void author(std::uint64_t frame, const std::vector<cohort_rect_t> &native_covers,
                bool reset = false, std::uint64_t generation = 7u, UINT cut = 3u) {
      const auto time = 1000000u + frame * 33333u;
      constants_ = {{width_, height_, crop_top_, crop_height_},
        {field_width_, field_height_, row_edge(cv2::subtitle_ocr_safe_row_top), row_edge(cv2::subtitle_ocr_safe_row_bottom)}, content_,
        {UINT(frame), UINT(frame >> 32u), UINT(generation), UINT(generation >> 32u)},
        {UINT(time), UINT(time >> 32u), reset ? 1u : 0u, 0u}, {1u, SBC_SCHEMA, 0u, 0u}};
      source_values_.assign(std::size_t(width_) * crop_height_, {cohort_half(.15f), cohort_half(.15f), cohort_half(.15f), cohort_half(1.f)});
      baseline_values_.assign(source_values_.size(), 0u);
      probability_values_.assign(cv2::subtitle_ocr_output_width * cv2::subtitle_ocr_output_height, 0.f);
      ocr_words_.assign(cv2::subtitle_ocr_record_word_count, 0u);
      plane_words_.assign(cv2::subtitle_plane_state_word_count, 0u);
      identity_words_.assign(cv2::subtitle_contour_history_word_count, 0u);
      const auto count = static_cast<UINT>(native_covers.size());
      ocr_words_[0u] = cv2::subtitle_ocr_record_schema; ocr_words_[1u] = cv2::subtitle_ocr_record_tag;
      ocr_words_[2u] = 1u; ocr_words_[3u] = ocr_words_[4u] = count;
      std::copy(constants_.frame.begin(), constants_.frame.end(), ocr_words_.begin() + 5u);
      ocr_words_[9u] = width_; ocr_words_[10u] = height_; ocr_words_[11u] = field_width_; ocr_words_[12u] = field_height_;
      ocr_words_[13u] = constants_.field[2u]; ocr_words_[14u] = constants_.field[3u];
      for (UINT slot = 0u; slot < count; ++slot) {
        auto box = field_rect(native_covers[slot]);
        box[1u] = std::max(box[1u], constants_.field[2u]); box[3u] = std::min(box[3u], constants_.field[3u]);
        for (auto offset : {cv2::subtitle_ocr_raw_box_offset, cv2::subtitle_ocr_final_box_offset}) {
          const auto base = offset + slot * cv2::subtitle_ocr_box_word_count;
          std::copy(box.begin(), box.end(), ocr_words_.begin() + base);
          ocr_words_[base + 4u] = std::bit_cast<UINT>(.9f); ocr_words_[base + 6u] = 1u;
        }
        for (UINT y = 0u; y < cv2::subtitle_ocr_output_height; ++y)
          for (UINT x = 0u; x < cv2::subtitle_ocr_output_width; ++x) {
            const double nx = (x + .5) * width_ / cv2::subtitle_ocr_output_width;
            const double ny = (y + .5) * crop_height_ / cv2::subtitle_ocr_output_height;
            if (nx >= native_covers[slot][0u] && nx < native_covers[slot][2u] &&
                ny >= native_covers[slot][1u] && ny < native_covers[slot][3u])
              probability_values_[std::size_t(y) * cv2::subtitle_ocr_output_width + x] = .9f;
          }
      }
      plane_words_[0u] = cv2::subtitle_plane_state_schema; plane_words_[1u] = cv2::subtitle_plane_state_tag;
      plane_words_[2u] = cv2::subtitle_plane_current_valid_flag; plane_words_[3u] = (1u << count) - 1u;
      for (UINT lane = 0u; lane < 2u; ++lane) {
        plane_words_[4u + lane] = constants_.frame[lane]; plane_words_[6u + lane] = constants_.frame[2u + lane];
        plane_words_[8u + lane] = constants_.observation[lane]; plane_words_[10u + lane] = constants_.source[lane];
        plane_words_[12u + lane] = constants_.field[lane]; plane_words_[14u + lane] = constants_.field[2u + lane];
      }
      std::copy(content_.begin(), content_.end(), plane_words_.begin() + 16u);
      plane_words_[20u] = cut;
      identity_words_[0u] = cv2::subtitle_contour_schema; identity_words_[1u] = cv2::subtitle_contour_history_tag;
      identity_words_[2u] = count ? 1u : 0u; identity_words_[3u] = cut;
      identity_words_[4u] = constants_.frame[0u]; identity_words_[5u] = constants_.frame[1u];
      identity_words_[6u] = constants_.frame[2u]; identity_words_[7u] = constants_.frame[3u];
      std::copy(constants_.source.begin(), constants_.source.end(), identity_words_.begin() + 8u);
      std::copy(constants_.field.begin(), constants_.field.end(), identity_words_.begin() + 12u);
      std::copy(content_.begin(), content_.end(), identity_words_.begin() + 16u);
      identity_words_[20u] = constants_.observation[0u]; identity_words_[21u] = constants_.observation[1u];
      identity_words_[22u] = cv2::subtitle_contour_recipe; identity_words_[23u] = 1u;
    }

    void paint(cohort_rect_t rect, float value = .95f, bool support = true) {
      for (UINT y = rect[1u]; y < std::min(rect[3u], crop_height_); ++y)
        for (UINT x = rect[0u]; x < std::min(rect[2u], width_); ++x) {
          source_values_[pixel(x, y)] = {cohort_half(value), cohort_half(value), cohort_half(value), cohort_half(1.f)};
          baseline_values_[pixel(x, y)] = support ? 1u : 0u;
        }
    }

    void glyph() {
      paint({50u, 8u, 54u, 21u}); paint({60u, 8u, 64u, 21u});
      paint({74u, 8u, 78u, 21u}); paint({84u, 8u, 88u, 21u});
    }

    bool run(bool zero_dispatch = false) {
      unbind();
      // Diagnostic input snapshots only. Their values never choose a dispatch, role,
      // eligibility, or state transition; they authenticate what the GPU consumed.
      if (!cohort_read_texture(device_.Get(), context_.Get(), previous_support_.texture.Get(), consumed_previous_support_) ||
          !cohort_read_texture(device_.Get(), context_.Get(), previous_source_.texture.Get(), consumed_previous_source_)) return false;
      context_->UpdateSubresource(current_source_.texture.Get(), 0u, nullptr, source_values_.data(), width_ * sizeof(cohort_half_t), 0u);
      context_->UpdateSubresource(baseline_.texture.Get(), 0u, nullptr, baseline_values_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(current_p_.buffer.Get(), 0u, nullptr, probability_values_.data(), 0u, 0u);
      context_->UpdateSubresource(ocr_.buffer.Get(), 0u, nullptr, ocr_words_.data(), 0u, 0u);
      context_->UpdateSubresource(plane_.buffer.Get(), 0u, nullptr, plane_words_.data(), 0u, 0u);
      context_->UpdateSubresource(current_identity_.buffer.Get(), 0u, nullptr, identity_words_.data(), 0u, 0u);
      context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, &constants_, 0u, 0u);
      const std::array<UINT, 3u> prep {zero_dispatch ? 0u : 1u, zero_dispatch ? 0u : 1u, zero_dispatch ? 0u : 1u};
      const std::array<UINT, 3u> grid {zero_dispatch ? 0u : (field_width_ + 15u) / 16u,
        zero_dispatch ? 0u : (field_height_ + 15u) / 16u, zero_dispatch ? 0u : 1u};
      context_->UpdateSubresource(prepare_arguments_.Get(), 0u, nullptr, prep.data(), 0u, 0u);
      context_->UpdateSubresource(pixel_arguments_.Get(), 0u, nullptr, grid.data(), 0u, 0u);
      const std::array<ID3D11ShaderResourceView *, 15u> inputs {
        current_source_.srv.Get(), current_p_.srv.Get(), ocr_.srv.Get(), plane_.srv.Get(),
        previous_source_.srv.Get(), previous_p_.srv.Get(), previous_identity_.srv.Get(),
        old_state_.srv.Get(), old_owner_.srv.Get(), old_rgb_.srv.Get(), baseline_.srv.Get(), nullptr, nullptr, nullptr, previous_support_.srv.Get()};
      context_->CSSetShaderResources(0u, static_cast<UINT>(inputs.size()), inputs.data());
      context_->CSSetConstantBuffers(0u, 1u, constants_buffer_.GetAddressOf());
      const std::array<ID3D11UnorderedAccessView *, 3u> member_outputs {next_state_.uav.Get(), next_owner_.uav.Get(), next_rgb_.uav.Get()};
      // prepare validates NEXT texture dimensions as well as its state buffer.
      context_->CSSetUnorderedAccessViews(0u, static_cast<UINT>(member_outputs.size()), member_outputs.data(), nullptr);
      context_->CSSetShader(shaders_[0u].Get(), nullptr, 0u);
      context_->DispatchIndirect(prepare_arguments_.Get(), 0u);
      context_->CSSetShader(shaders_[1u].Get(), nullptr, 0u);
      context_->DispatchIndirect(pixel_arguments_.Get(), 0u);
      // Publish from NEXT only after every old SRV and every NEXT UAV is unbound.
      const std::array<ID3D11ShaderResourceView *, 3u> null_reads {};
      const std::array<ID3D11UnorderedAccessView *, 3u> null_writes {};
      context_->CSSetShaderResources(7u, 3u, null_reads.data());
      context_->CSSetUnorderedAccessViews(0u, 3u, null_writes.data(), nullptr);
      const std::array<ID3D11ShaderResourceView *, 3u> next_reads {next_state_.srv.Get(), next_owner_.srv.Get(), next_rgb_.srv.Get()};
      context_->CSSetShaderResources(11u, 3u, next_reads.data());
      const std::array<ID3D11UnorderedAccessView *, 5u> filter_outputs {
        eligibility_.uav.Get(), filtered_.uav.Get(), old_state_.uav.Get(), old_owner_.uav.Get(), old_rgb_.uav.Get()};
      context_->CSSetUnorderedAccessViews(3u, 5u, filter_outputs.data(), nullptr);
      context_->CSSetShader(shaders_[2u].Get(), nullptr, 0u);
      context_->DispatchIndirect(pixel_arguments_.Get(), 0u);
      unbind();
      // Fixture snapshot advancement depends solely on the authored submission clock/zero
      // branch, never on readback, phase, qualification, eligibility, or a CPU color verdict.
      const auto frame = std::uint64_t(constants_.frame[0u]) | (std::uint64_t(constants_.frame[1u]) << 32u);
      const auto time = std::uint64_t(constants_.observation[0u]) | (std::uint64_t(constants_.observation[1u]) << 32u);
      const bool genuine_snapshot = !zero_dispatch && (snapshot_frame_ == 0u || (frame > snapshot_frame_ && time > snapshot_time_));
      if (genuine_snapshot) {
        context_->CopyResource(previous_source_.texture.Get(), current_source_.texture.Get());
        context_->CopyResource(previous_p_.buffer.Get(), current_p_.buffer.Get());
        context_->CopyResource(previous_identity_.buffer.Get(), current_identity_.buffer.Get());
        context_->CopyResource(previous_support_.texture.Get(), baseline_.texture.Get());
        snapshot_frame_ = frame; snapshot_time_ = time;
      }
      return capture() && result_.previous_support == (genuine_snapshot ? baseline_values_ : consumed_previous_support_) &&
        result_.previous_source == (genuine_snapshot ? source_values_ : consumed_previous_source_);
    }

    // Independent fourth research entry. Existing three-pass tests and their closure
    // remain unchanged. Call only after every old RGB/P/SCT/raw-S reader is unbound.
    // The fixture controls the authored indirect work/reuse branch, never a GPU verdict.
    bool snapshot_current_tuple(bool zero_dispatch = false) {
      unbind();
      if (!snapshot_shader_) {
        namespace cache = models::host_sbs_shader_cache;
        const std::array specs {cache::shader_spec {"host_sbs_subtitle_cohort_snapshot_cs.hlsl", "snapshot_main", "cs_5_0"}};
        const auto sources = cache::snapshot_sources(shader_root_, specs);
        snapshot_closure_ = cache::source_closure_sha256(sources);
        const auto code = cache::get(sources, specs[0u]);
        if (!code || code->empty() || FAILED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &snapshot_shader_))) return false;
      }
      context_->UpdateSubresource(current_source_.texture.Get(), 0u, nullptr, source_values_.data(), width_ * sizeof(cohort_half_t), 0u);
      context_->UpdateSubresource(baseline_.texture.Get(), 0u, nullptr, baseline_values_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(current_p_.buffer.Get(), 0u, nullptr, probability_values_.data(), 0u, 0u);
      context_->UpdateSubresource(current_identity_.buffer.Get(), 0u, nullptr, identity_words_.data(), 0u, 0u);
      context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, &constants_, 0u, 0u);
      const std::array<UINT, 3u> grid {zero_dispatch ? 0u : (field_width_ + 15u) / 16u,
        zero_dispatch ? 0u : (field_height_ + 15u) / 16u, zero_dispatch ? 0u : 1u};
      context_->UpdateSubresource(pixel_arguments_.Get(), 0u, nullptr, grid.data(), 0u, 0u);
      const std::array<ID3D11ShaderResourceView *, 4u> reads {
        current_source_.srv.Get(), current_p_.srv.Get(), current_identity_.srv.Get(), baseline_.srv.Get()};
      const std::array<ID3D11UnorderedAccessView *, 4u> writes {
        previous_source_.uav.Get(), previous_p_.uav.Get(), previous_identity_.uav.Get(), previous_support_.uav.Get()};
      context_->CSSetShaderResources(0u, static_cast<UINT>(reads.size()), reads.data());
      context_->CSSetUnorderedAccessViews(0u, static_cast<UINT>(writes.size()), writes.data(), nullptr);
      context_->CSSetConstantBuffers(0u, 1u, constants_buffer_.GetAddressOf());
      context_->CSSetShader(snapshot_shader_.Get(), nullptr, 0u);
      context_->DispatchIndirect(pixel_arguments_.Get(), 0u);
      unbind();
      return capture();
    }

    bool capture() {
      return cohort_read_buffer(device_.Get(), context_.Get(), old_state_.buffer.Get(), result_.state) &&
        cohort_read_buffer(device_.Get(), context_.Get(), next_state_.buffer.Get(), result_.next_state) &&
        cohort_read_texture(device_.Get(), context_.Get(), old_owner_.texture.Get(), result_.owner) &&
        cohort_read_texture(device_.Get(), context_.Get(), old_rgb_.texture.Get(), result_.birth_rgb) &&
        cohort_read_texture(device_.Get(), context_.Get(), next_owner_.texture.Get(), result_.next_owner) &&
        cohort_read_texture(device_.Get(), context_.Get(), next_rgb_.texture.Get(), result_.next_rgb) &&
        cohort_read_texture(device_.Get(), context_.Get(), eligibility_.texture.Get(), result_.eligibility) &&
        cohort_read_texture(device_.Get(), context_.Get(), filtered_.texture.Get(), result_.filtered) &&
        cohort_read_texture(device_.Get(), context_.Get(), previous_source_.texture.Get(), result_.previous_source) &&
        cohort_read_buffer(device_.Get(), context_.Get(), previous_p_.buffer.Get(), result_.previous_p) &&
        cohort_read_buffer(device_.Get(), context_.Get(), previous_identity_.buffer.Get(), result_.previous_identity) &&
        cohort_read_texture(device_.Get(), context_.Get(), previous_support_.texture.Get(), result_.previous_support);
    }

    std::size_t pixel(UINT x, UINT y) const { return std::size_t(y) * width_ + x; }
    static bool is_known(UINT owner) { return owner != 0u && (owner & SBC_OWNER_UNKNOWN_BIT) == 0u; }
    static bool is_unknown(UINT owner) { return (owner & SBC_OWNER_UNKNOWN_BIT) != 0u; }
    static bool in_unknown_dependency(int dx, int dy) {
      const int x = std::max(std::abs(dx) - 3, 0), y = std::max(std::abs(dy) - 3, 0);
      return x * x + y * y <= 10;
    }
    bool pixel_in_field_rect(std::size_t p, const UINT *rect) const {
      const auto x = p % width_, y = p / width_;
      const auto a = 2ull * content_[0u] * width_ + (2ull * x + 1u) * (content_[2u] - content_[0u]);
      const auto b = 2ull * content_[1u] * height_ + (2ull * (y + crop_top_) + 1u) * (content_[3u] - content_[1u]);
      return a >= 2ull * rect[0u] * width_ && a < 2ull * rect[2u] * width_ &&
        b >= 2ull * rect[1u] * height_ && b < 2ull * rect[3u] * height_;
    }
    static UINT packed_color(const cohort_half_t &color) {
      UINT value = 0u;
      for (UINT lane = 0u; lane < 3u; ++lane) {
        value |= UINT(std::floor(cohort_float(color[lane]) * 255.f + .5f)) << (lane * 8u);
      }
      return value;
    }
    bool on_current_palette_grid(UINT x, UINT y) const {
      const auto b = cv2::subtitle_ocr_raw_box_offset;
      for (UINT gy = 0u; gy < 8u; ++gy) {
        for (UINT gx = 0u; gx < 64u; ++gx) {
          const auto sx = ((std::uint64_t(ocr_words_[b] - content_[0u]) * 128u +
            (2u * gx + 1u) * (ocr_words_[b + 2u] - ocr_words_[b])) * width_) / (128u * (content_[2u] - content_[0u]));
          const auto sy = ((std::uint64_t(ocr_words_[b + 1u] - content_[1u]) * 16u +
            (2u * gy + 1u) * (ocr_words_[b + 3u] - ocr_words_[b + 1u])) * height_) / (16u * (content_[3u] - content_[1u]));
          if (std::min<std::uint64_t>(sx, width_ - 1u) == x &&
              std::clamp<std::int64_t>(std::int64_t(sy) - crop_top_, 0, crop_height_ - 1u) == y) return true;
        }
      }
      return false;
    }
    UINT phase(UINT slot = 0u) const { return result_.state.at(cohort::entry_offset(slot) + SBC_E_PHASE); }
    UINT id(UINT slot = 0u) const { return result_.state.at(cohort::entry_offset(slot) + SBC_E_ID); }
    std::array<UINT, 4u> birth_words(UINT slot = 0u) const {
      std::array<UINT, 4u> words {};
      std::copy_n(result_.state.begin() + cohort::entry_offset(slot) + SBC_E_BIRTH_FRAME, 4u, words.begin());
      return words;
    }
    void expect_known_members_unchanged(const std::vector<UINT> &owners, const std::vector<UINT> &rgb) const {
      ASSERT_EQ(owners.size(), result_.owner.size()); ASSERT_EQ(rgb.size(), result_.birth_rgb.size());
      for (std::size_t p = 0u; p < owners.size(); ++p) {
        if (is_known(owners[p]) || is_known(result_.owner[p])) {
          ASSERT_EQ(result_.owner[p], owners[p]) << p;
          ASSERT_EQ(result_.birth_rgb[p], rgb[p]) << p;
        } else if (is_unknown(result_.owner[p])) {
          ASSERT_NE(result_.owner[p] & SBC_OWNER_ID_MASK, 0u) << p;
          ASSERT_EQ(result_.birth_rgb[p] & ~SBC_UNKNOWN_RGB_MASK, 0u) << p;
        }
      }
    }
    void expect_fallback() const {
      EXPECT_TRUE(std::all_of(result_.eligibility.begin(), result_.eligibility.end(), [](UINT x) { return x == 1u; }));
      EXPECT_EQ(result_.filtered, baseline_values_);
    }
    void expect_binary_subset() const {
      ASSERT_EQ(result_.filtered.size(), baseline_values_.size());
      for (std::size_t p = 0; p < result_.filtered.size(); ++p) {
        ASSERT_LE(result_.eligibility[p], 1u); ASSERT_LE(result_.filtered[p], baseline_values_[p]);
      }
      if (result_.state.at(SBC_H_VALID) == 1u) {
        for (std::size_t p = 0u; p < result_.owner.size(); ++p) {
          if (is_unknown(result_.owner[p])) {
            ASSERT_NE(result_.owner[p] & SBC_OWNER_ID_MASK, 0u) << p;
            ASSERT_EQ(result_.birth_rgb[p] & ~SBC_UNKNOWN_RGB_MASK, 0u) << p;
          } else if (result_.owner[p] == 0u) {
            ASSERT_EQ(result_.birth_rgb[p], 0u) << p;
          }
          if (result_.owner[p] == result_.next_owner[p]) {
            ASSERT_EQ(result_.birth_rgb[p], result_.next_rgb[p]) << p;
            continue;
          }
          if (is_unknown(result_.next_owner[p]) && result_.owner[p] == 0u) {
            // Final per-line invalidity/SUS can retire UNKNOWN role
            // without erasing KNOWN provenance or lending it to another line.
            const auto line = result_.next_owner[p] & SBC_OWNER_ID_MASK;
            bool valid_context = false;
            for (UINT slot = 0u; slot < SBC_CAPACITY; ++slot) {
              const auto b = cohort::entry_offset(slot);
              valid_context |= result_.next_state[b + SBC_E_ID] == line &&
                (result_.next_state[b + SBC_E_PHASE] == SBC_PHASE_PENDING || result_.next_state[b + SBC_E_PHASE] == SBC_PHASE_ACTIVE) &&
                (result_.next_state[b + SBC_E_FLAGS] & SBC_ENTRY_BLOCKING_FLAGS) == 0u;
            }
            ASSERT_FALSE(valid_context) << p;
            ASSERT_EQ(result_.birth_rgb[p], 0u) << p;
            continue;
          }
          // The sole valid publication exception: FILTER may add a latent UNKNOWN
          // next to immutable NEW_BIRTH KNOWN support while NEXT owner remains0.
          ASSERT_EQ(result_.next_owner[p], 0u) << p;
          ASSERT_EQ(result_.next_rgb[p], 0u) << p;
          ASSERT_TRUE(is_unknown(result_.owner[p])) << p;
          ASSERT_EQ(result_.birth_rgb[p] & ~SBC_UNKNOWN_RGB_MASK, 0u) << p;
          ASSERT_EQ(result_.birth_rgb[p], packed_color(source_values_[p])) << p;
          const auto line = result_.owner[p] & SBC_OWNER_ID_MASK;
          ASSERT_NE(line, 0u);
          bool pending_birth = false;
          cohort_rect_t pending_cover {};
          for (UINT slot = 0u; slot < SBC_CAPACITY; ++slot) {
            const auto b = cohort::entry_offset(slot);
            const bool match = result_.next_state[b + SBC_E_ID] == line &&
              result_.next_state[b + SBC_E_PHASE] == SBC_PHASE_PENDING &&
              (result_.next_state[b + SBC_E_FLAGS] & SBC_ENTRY_NEW_BIRTH) != 0u;
            if (match) {
              pending_birth = true;
              std::copy_n(result_.next_state.begin() + b + SBC_E_LAST_COVER, 4u, pending_cover.begin());
            }
          }
          ASSERT_TRUE(pending_birth) << p;
          ASSERT_TRUE(pixel_in_field_rect(p, pending_cover.data())) << p;
          ASSERT_EQ(baseline_values_[p], 1u); ASSERT_EQ(consumed_previous_support_[p], 0u);
          for (UINT lane = 0u; lane < 3u; ++lane) {
            ASSERT_EQ(cohort_float(source_values_[p][lane]), cohort_float(consumed_previous_source_[p][lane])) << p << ':' << lane;
          }
          const int x = int(p % width_), y = int(p / width_);
          bool near_new_known = false;
          for (int dy = -6; dy <= 6 && !near_new_known; ++dy) {
            for (int dx = -6; dx <= 6 && !near_new_known; ++dx) {
              const int qx = x + dx, qy = y + dy;
              if (qx < 0 || qy < 0 || qx >= int(width_) || qy >= int(crop_height_) || !in_unknown_dependency(dx, dy)) continue;
              const auto q = pixel(UINT(qx), UINT(qy));
              near_new_known = result_.next_owner[q] == line && is_known(result_.next_owner[q]) &&
                baseline_values_[q] == 1u && pixel_in_field_rect(q, pending_cover.data());
            }
          }
          ASSERT_TRUE(near_new_known) << p;
        }
      } else {
        // Whole-plane invalidity is finalized after MEMBERS. Raw NEXT remains a
        // diagnostic candidate, while invalid publication must expose no owners.
        EXPECT_TRUE(std::all_of(result_.owner.begin(), result_.owner.end(), [](UINT value) { return value == 0u; }));
        EXPECT_TRUE(std::all_of(result_.birth_rgb.begin(), result_.birth_rgb.end(), [](UINT value) { return value == 0u; }));
      }
    }
    void acquire() {
      author(1u, {}); ASSERT_TRUE(run());
      author(2u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
      ASSERT_EQ(phase(), SBC_PHASE_PENDING);
      author(3u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
      ASSERT_EQ(phase(), SBC_PHASE_ACTIVE);
    }

    // Explicit private-state fault injection for FILTER in isolation. This does not
    // claim that the forged tuple is a valid source observation or a drawing mask.
    bool filter_forged_next_for_test(const std::vector<UINT> &state,
                                     const std::vector<UINT> &owners,
                                     const std::vector<UINT> &birth_rgb) {
      unbind();
      if (state.size() != SBC_STATE_WORDS || owners.size() != std::size_t(width_) * crop_height_ ||
          birth_rgb.size() != owners.size()) return false;
      context_->UpdateSubresource(next_state_.buffer.Get(), 0u, nullptr, state.data(), 0u, 0u);
      context_->UpdateSubresource(next_owner_.texture.Get(), 0u, nullptr, owners.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(next_rgb_.texture.Get(), 0u, nullptr, birth_rgb.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(current_source_.texture.Get(), 0u, nullptr, source_values_.data(), width_ * sizeof(cohort_half_t), 0u);
      context_->UpdateSubresource(baseline_.texture.Get(), 0u, nullptr, baseline_values_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, &constants_, 0u, 0u);
      const std::array<ID3D11ShaderResourceView *, 15u> inputs {
        current_source_.srv.Get(), nullptr, nullptr, nullptr, previous_source_.srv.Get(), nullptr, nullptr,
        nullptr, nullptr, nullptr, baseline_.srv.Get(), next_state_.srv.Get(), next_owner_.srv.Get(), next_rgb_.srv.Get(), previous_support_.srv.Get()};
      context_->CSSetShaderResources(0u, static_cast<UINT>(inputs.size()), inputs.data());
      context_->CSSetConstantBuffers(0u, 1u, constants_buffer_.GetAddressOf());
      const std::array<ID3D11UnorderedAccessView *, 5u> outputs {
        eligibility_.uav.Get(), filtered_.uav.Get(), old_state_.uav.Get(), old_owner_.uav.Get(), old_rgb_.uav.Get()};
      context_->CSSetUnorderedAccessViews(3u, static_cast<UINT>(outputs.size()), outputs.data(), nullptr);
      context_->CSSetShader(shaders_[2u].Get(), nullptr, 0u);
      context_->Dispatch((field_width_ + 15u) / 16u, (field_height_ + 15u) / 16u, 1u);
      unbind();
      return capture();
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    std::array<ComPtr<ID3D11ComputeShader>, 3u> shaders_;
    ComPtr<ID3D11ComputeShader> snapshot_shader_;
    ComPtr<ID3D11Buffer> constants_buffer_, prepare_arguments_, pixel_arguments_;
    cohort_texture_t current_source_, previous_source_, baseline_, previous_support_, old_owner_, old_rgb_, next_owner_, next_rgb_, eligibility_, filtered_;
    cohort_buffer_t current_p_, previous_p_, ocr_, plane_, current_identity_, previous_identity_, old_state_, next_state_;
    cohort::constants_t constants_ {};
    std::vector<cohort_half_t> source_values_;
    std::vector<float> probability_values_;
    std::vector<UINT> baseline_values_, consumed_previous_support_, ocr_words_, plane_words_, identity_words_;
    std::vector<cohort_half_t> consumed_previous_source_;
    cohort_result_t result_;
    UINT width_ {}, height_ {}, crop_top_ {}, crop_height_ {}, field_width_ {}, field_height_ {};
    cohort_rect_t content_ {};
    std::uint64_t snapshot_frame_ {}, snapshot_time_ {};
    std::filesystem::path shader_root_;
    std::string driver_name_ {"warp"}, closure_, snapshot_closure_;
  };

  TEST_F(HostSbsSubtitleCohortGpuTest, PositiveStartupIsUnknownAndBitExactFallback) {
    author(1u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    EXPECT_NE(phase(), SBC_PHASE_ACTIVE); expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, QuietReferenceThenTwoGenuineObservationsOwnWholeBirth) {
    author(1u, {}); ASSERT_TRUE(run());
    author(2u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    ASSERT_EQ(phase(), SBC_PHASE_PENDING); EXPECT_GT(id(), 0u); expect_fallback();
    const auto born = result_.owner;
    author(3u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    EXPECT_EQ(phase(), SBC_PHASE_ACTIVE); EXPECT_EQ(result_.owner, born);
    EXPECT_EQ(result_.owner[pixel(51u, 12u)], id()); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, DuplicateDoesNotConfirmAndZeroDispatchHoldsEntireTuple) {
    author(1u, {}); ASSERT_TRUE(run());
    author(2u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    const auto born = result_.owner;
    ASSERT_TRUE(run()); EXPECT_EQ(phase(), SBC_PHASE_PENDING); EXPECT_EQ(result_.owner, born); expect_fallback();
    const auto saved = result_;
    author(3u, {}); ASSERT_TRUE(run(true)); EXPECT_EQ(result_, saved);
    author(3u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run()); EXPECT_EQ(phase(), SBC_PHASE_ACTIVE);
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, TemporaryRgbDriftAndReferenceEqualityKeepLifetimeOwner) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto owner = id(), fixed = result_.birth_rgb[pixel(51u, 12u)];
    author(4u, {{30u, 6u, 130u, 24u}}); glyph(); paint({51u, 12u, 52u, 13u}, .15f, false);
    ASSERT_TRUE(run()); EXPECT_EQ(phase(), SBC_PHASE_ACTIVE);
    EXPECT_EQ(result_.owner[pixel(51u, 12u)], owner); EXPECT_EQ(result_.birth_rgb[pixel(51u, 12u)], fixed);
    EXPECT_EQ(result_.eligibility[pixel(51u, 12u)], 1u);
    author(5u, {{30u, 6u, 130u, 24u}}); glyph(); paint({51u, 12u, 52u, 13u}, .55f);
    ASSERT_TRUE(run()); EXPECT_EQ(result_.owner[pixel(51u, 12u)], owner); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, PendingTwoLevelRgbDriftDoesNotRelaxStrictAcquisition) {
    author(1u, {}); ASSERT_TRUE(run());
    author(2u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    ASSERT_EQ(phase(), SBC_PHASE_PENDING);
    author(3u, {{30u, 6u, 130u, 24u}});
    for (UINT x : {50u, 60u, 74u, 84u}) {
      paint({x, 8u, x + 4u, 21u}, .94f);
    }
    ASSERT_TRUE(run());
    // .95 birth rounds to RGB8 242; .94 rounds to 240. Both are far inside
    // the retirement tolerance, but neither is a <=1-level acquisition witness.
    EXPECT_NE(phase(), SBC_PHASE_ACTIVE); expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, MajorityBoundedRgbDriftRetainsActiveLifetimeWithStrictMinority) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto original_id = id();
    const auto original_owner = result_.owner, original_rgb = result_.birth_rgb;
    const auto offset = cohort::entry_offset(0u);
    const std::vector<UINT> birth(result_.state.begin() + offset + SBC_E_BIRTH_FRAME,
                                  result_.state.begin() + offset + SBC_E_BIRTH_FRAME + 4u);
    author(4u, {{30u, 6u, 130u, 24u}}); glyph();
    for (UINT x : {50u, 60u, 74u}) {
      paint({x, 8u, x + 4u, 21u}, .90f);
    }
    ASSERT_TRUE(run());
    // Three of four equal strokes move >1 RGB8 level but <the existing .08
    // retirement distance. The fourth supplies genuine strict stable witnesses.
    EXPECT_EQ(phase(), SBC_PHASE_ACTIVE); EXPECT_EQ(id(), original_id);
    const auto tested = result_.state[offset + SBC_E_TESTED_ANCHORS];
    const auto stable = result_.state[offset + SBC_E_STABLE_ANCHORS];
    ASSERT_GT(stable, 0u); EXPECT_GT(tested, 2u * stable);
    EXPECT_EQ(result_.owner, original_owner); EXPECT_EQ(result_.birth_rgb, original_rgb);
    const std::vector<UINT> current_birth(result_.state.begin() + offset + SBC_E_BIRTH_FRAME,
                                          result_.state.begin() + offset + SBC_E_BIRTH_FRAME + 4u);
    EXPECT_EQ(current_birth, birth); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, MajorityAboveExistingRgbToleranceStillSuspendsReplacement) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto original_id = id();
    const auto original_owner = result_.owner, original_rgb = result_.birth_rgb;
    author(4u, {{30u, 6u, 130u, 24u}}); glyph();
    for (UINT x : {50u, 60u, 74u}) {
      paint({x, 8u, x + 4u, 21u}, .80f);
    }
    ASSERT_TRUE(run());
    EXPECT_EQ(phase(), SBC_PHASE_SUSPENDED); EXPECT_EQ(id(), original_id);
    EXPECT_NE(result_.state[cohort::entry_offset(0u) + SBC_E_FLAGS] & SBC_ENTRY_UNCERTAIN_REPLACEMENT, 0u);
    EXPECT_EQ(result_.owner, original_owner); EXPECT_EQ(result_.birth_rgb, original_rgb);
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, NonzeroReservedCutConstantCannotSupplyEpochAuthority) {
    SCOPED_TRACE("Explicit reserved-constant fault; fresh cut authority belongs to GPU UIP");
    ASSERT_NO_FATAL_FAILURE(acquire());
    author(4u, {{30u, 6u, 130u, 24u}}); glyph();
    ASSERT_EQ(constants_.observation[3u], 0u);
    constants_.observation[3u] = 3u;
    ASSERT_TRUE(run());
    EXPECT_EQ(result_.state[SBC_H_VALID], 0u); EXPECT_EQ(result_.state[SBC_H_CUT], 0u);
    EXPECT_NE(result_.state[SBC_H_FLAGS] & SBC_HEADER_CURRENT_INVALID, 0u);
    EXPECT_TRUE(std::all_of(result_.owner.begin(), result_.owner.end(), [](UINT value) { return value == 0u; }));
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, FreshGpuUipEpochChangeInvalidatesOldCueWithReservedConstantZero) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    ASSERT_EQ(result_.state[SBC_H_CUT], 3u);
    author(4u, {{30u, 6u, 130u, 24u}}, false, 7u, 4u); glyph();
    ASSERT_EQ(constants_.observation[3u], 0u);
    ASSERT_EQ(plane_words_[cv2::subtitle_plane_cut_epoch_word], 4u);
    ASSERT_TRUE(run());
    EXPECT_EQ(result_.state[SBC_H_VALID], 1u); EXPECT_EQ(result_.state[SBC_H_CUT], 4u);
    EXPECT_EQ(id(), 0u); EXPECT_EQ(phase(), SBC_PHASE_UNKNOWN);
    EXPECT_TRUE(std::all_of(result_.owner.begin(), result_.owner.end(), [](UINT value) { return value == 0u; }));
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, StaleGpuUipCannotLendItsCutEpochToCurrentFrame) {
    SCOPED_TRACE("Explicit stale-UIP frame fault; OCR and source retain the actual current frame");
    ASSERT_NO_FATAL_FAILURE(acquire());
    author(4u, {{30u, 6u, 130u, 24u}}, false, 7u, 4u); glyph();
    plane_words_[cv2::subtitle_plane_frame_offset] = 3u;
    ASSERT_EQ(constants_.observation[3u], 0u);
    ASSERT_TRUE(run());
    EXPECT_EQ(result_.state[SBC_H_VALID], 0u); EXPECT_EQ(result_.state[SBC_H_CUT], 0u);
    EXPECT_NE(result_.state[SBC_H_FLAGS] & SBC_HEADER_CURRENT_INVALID, 0u);
    EXPECT_TRUE(std::all_of(result_.owner.begin(), result_.owner.end(), [](UINT value) { return value == 0u; }));
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, FreshEmptyImmediatelyReleasesAndCutCannotBorrowOldOwner) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    author(4u, {}); ASSERT_TRUE(run()); EXPECT_EQ(result_.state[SBC_H_COUNT], 0u);
    EXPECT_TRUE(std::all_of(result_.owner.begin(), result_.owner.end(), [](UINT x) { return x == 0u; }));
    expect_fallback();
    author(5u, {{30u, 6u, 130u, 24u}}, true, 7u, 4u); glyph(); ASSERT_TRUE(run());
    EXPECT_NE(phase(), SBC_PHASE_ACTIVE); expect_fallback();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, DomainGenerationChangeInvalidatesCue) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    author(4u, {{30u, 6u, 130u, 24u}}, false, 8u); glyph(); ASSERT_TRUE(run());
    EXPECT_NE(phase(), SBC_PHASE_ACTIVE); expect_fallback();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, LatePatchIsLocalUnknownWithoutJoiningKnownBirth) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto existing_id = id();
    const auto known = result_.owner, fixed = result_.birth_rgb;
    author(4u, {{30u, 6u, 130u, 24u}}); glyph(); paint({100u, 10u, 105u, 18u}); ASSERT_TRUE(run());
    EXPECT_EQ(id(), existing_id); EXPECT_EQ(phase(), SBC_PHASE_ACTIVE);
    EXPECT_EQ(result_.owner[pixel(102u, 12u)], SBC_OWNER_UNKNOWN_BIT | existing_id);
    EXPECT_EQ(result_.filtered[pixel(102u, 12u)], 1u);
    expect_known_members_unchanged(known, fixed);
    expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, NewDarkSupportIsUnknownWithoutSuspendingKnownCue) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto original_id = id();
    const auto original_owner = result_.owner, original_rgb = result_.birth_rgb;
    author(4u, {{30u, 6u, 130u, 24u}}); glyph();
    paint({92u, 10u, 94u, 18u}, .05f);
    ASSERT_EQ(baseline_values_[pixel(92u, 12u)], 1u);
    ASSERT_TRUE(run());
    EXPECT_EQ(phase(), SBC_PHASE_ACTIVE); EXPECT_EQ(id(), original_id);
    expect_known_members_unchanged(original_owner, original_rgb);
    EXPECT_EQ(result_.owner[pixel(92u, 12u)], SBC_OWNER_UNKNOWN_BIT | original_id);
    EXPECT_EQ(result_.filtered[pixel(92u, 12u)], 1u);
    expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, NonbinaryPreviousRawSupportOutsideEveryCoverInvalidatesPublication) {
    SCOPED_TRACE("Explicit compatible previous-S fault outside all current covers");
    ASSERT_NO_FATAL_FAILURE(acquire());
    auto forged = result_.previous_support;
    forged[pixel(5u, 2u)] = 2u;
    unbind();
    context_->UpdateSubresource(previous_support_.texture.Get(), 0u, nullptr, forged.data(), width_ * 4u, 0u);
    author(4u, {{30u, 6u, 130u, 24u}}); glyph();
    ASSERT_TRUE(run());
    EXPECT_EQ(result_.state[SBC_H_VALID], 0u); EXPECT_EQ(result_.state[SBC_H_COUNT], 0u);
    EXPECT_NE(result_.state[SBC_H_FLAGS] & SBC_HEADER_CURRENT_INVALID, 0u);
    EXPECT_TRUE(std::all_of(result_.state.begin() + SBC_HEADER_WORDS, result_.state.end(), [](UINT value) { return value == 0u; }));
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, InvalidPreviousSnapshotCannotRetainOrConfirmKnownAuthority) {
    SCOPED_TRACE("Explicit stale previous-SCT context with an otherwise binary previous native S");
    ASSERT_NO_FATAL_FAILURE(acquire());
    auto forged = result_.previous_identity;
    forged[cv2::subtitle_contour_history_snapshot_valid_word] = 0u;
    unbind();
    context_->UpdateSubresource(previous_identity_.buffer.Get(), 0u, nullptr, forged.data(), 0u, 0u);
    author(4u, {{30u, 6u, 130u, 24u}}); glyph();
    ASSERT_TRUE(run());
    EXPECT_NE(phase(), SBC_PHASE_ACTIVE); EXPECT_EQ(id(), 0u);
    EXPECT_TRUE(std::all_of(result_.owner.begin(), result_.owner.end(), [](UINT value) { return value == 0u; }));
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, StableUnknownPatchNeverBorrowsKnownBirthAge) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto original_id = id();
    const auto original_owner = result_.owner, original_rgb = result_.birth_rgb;
    const auto birth_offset = cohort::entry_offset(0u) + SBC_E_BIRTH_FRAME;
    const std::vector<UINT> original_birth(result_.state.begin() + birth_offset, result_.state.begin() + birth_offset + 4u);
    for (std::uint64_t frame = 4u; frame <= 6u; ++frame) {
      SCOPED_TRACE(frame);
      author(frame, {{30u, 6u, 130u, 24u}}); glyph(); paint({100u, 10u, 105u, 18u});
      ASSERT_TRUE(run());
      EXPECT_EQ(phase(), SBC_PHASE_ACTIVE); EXPECT_EQ(id(), original_id);
      expect_known_members_unchanged(original_owner, original_rgb);
      EXPECT_EQ(result_.owner[pixel(102u, 12u)], SBC_OWNER_UNKNOWN_BIT | original_id);
      EXPECT_EQ(result_.filtered[pixel(102u, 12u)], 1u);
      const std::vector<UINT> current_birth(result_.state.begin() + birth_offset, result_.state.begin() + birth_offset + 4u);
      EXPECT_EQ(current_birth, original_birth);
      expect_binary_subset();
    }
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, KnownAndUnknownOwnersClearOnEmptyCutOrDomainChange) {
    for (UINT boundary = 0u; boundary < 3u; ++boundary) {
      SCOPED_TRACE(boundary);
      reset_state(); ASSERT_NO_FATAL_FAILURE(acquire());
      author(4u, {{30u, 6u, 130u, 24u}}); glyph(); paint({100u, 10u, 105u, 18u});
      ASSERT_TRUE(run()); ASSERT_EQ(phase(), SBC_PHASE_ACTIVE);
      ASSERT_TRUE(is_unknown(result_.owner[pixel(102u, 12u)]));
      if (boundary == 0u) {
        author(5u, {});
      } else {
        author(5u, {{30u, 6u, 130u, 24u}}, false, boundary == 2u ? 8u : 7u, boundary == 1u ? 4u : 3u);
        glyph();
      }
      ASSERT_TRUE(run());
      EXPECT_TRUE(std::all_of(result_.owner.begin(), result_.owner.end(), [](UINT value) { return value == 0u; }));
      EXPECT_TRUE(std::all_of(result_.birth_rgb.begin(), result_.birth_rgb.end(), [](UINT value) { return value == 0u; }));
      if (boundary == 0u) {
        EXPECT_EQ(result_.state[SBC_H_COUNT], 0u);
      } else {
        EXPECT_EQ(id(), 0u); EXPECT_EQ(phase(), SBC_PHASE_UNKNOWN);
      }
      expect_fallback(); expect_binary_subset();
    }
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ForgedOwnedBirthHighBitsSuspendWholeUnitWithoutAddingMembers) {
    SCOPED_TRACE("Explicit private RGB-state fault injection; not source evidence or GT");
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto original_id = id();
    const auto target = pixel(51u, 12u);
    const auto original_owner = result_.owner;
    ASSERT_EQ(original_owner[target], original_id);
    auto forged = result_.birth_rgb;
    forged[target] |= 0x80000000u;
    unbind();
    context_->UpdateSubresource(old_rgb_.texture.Get(), 0u, nullptr, forged.data(), width_ * 4u, 0u);
    author(4u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    EXPECT_EQ(phase(), SBC_PHASE_SUSPENDED); EXPECT_EQ(id(), original_id);
    EXPECT_NE(result_.state[cohort::entry_offset(0u) + SBC_E_FLAGS] & SBC_ENTRY_INVALID_REFERENCE, 0u);
    for (std::size_t p = 0u; p < result_.owner.size(); ++p) {
      if (result_.owner[p] != 0u) {
        ASSERT_EQ(result_.owner[p], original_owner[p]) << p;
      }
    }
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ForgedNextKnownOrUnknownCannotLendOutsideItsCurrentLine) {
    SCOPED_TRACE("Explicit private NEXT-state fault injection isolates FILTER owner-cover admission");
    const cohort_rect_t left {28u, 6u, 68u, 24u}, right {92u, 6u, 132u, 24u};
    author(1u, {}); ASSERT_TRUE(run());
    for (std::uint64_t frame = 2u; frame <= 3u; ++frame) {
      author(frame, {left, right}); paint({40u, 8u, 44u, 21u}); paint({106u, 8u, 110u, 21u}); ASSERT_TRUE(run());
    }
    ASSERT_EQ(phase(0u), SBC_PHASE_ACTIVE); ASSERT_EQ(phase(1u), SBC_PHASE_ACTIVE);
    const auto left_id = id(0u), right_id = id(1u);
    const auto target = pixel(107u, 12u);
    ASSERT_NE(left_id, right_id); ASSERT_EQ(result_.owner[target], right_id);
    auto forged_state = result_.state, forged_owner = result_.owner, forged_rgb = result_.birth_rgb;
    const auto left_rgb = result_.birth_rgb[pixel(41u, 12u)];
    author(4u, {left, right}); paint({40u, 8u, 44u, 21u}); paint({106u, 8u, 110u, 21u});
    forged_state[SBC_H_FRAME] = constants_.frame[0u]; forged_state[SBC_H_FRAME + 1u] = constants_.frame[1u];
    forged_state[SBC_H_CLOCK] = constants_.observation[0u]; forged_state[SBC_H_CLOCK + 1u] = constants_.observation[1u];
    for (const UINT role : {left_id, SBC_OWNER_UNKNOWN_BIT | left_id}) {
      SCOPED_TRACE(role);
      forged_owner[target] = role;
      forged_rgb[target] = is_unknown(role) ? left_rgb & SBC_UNKNOWN_RGB_MASK : left_rgb;
      ASSERT_TRUE(filter_forged_next_for_test(forged_state, forged_owner, forged_rgb));
      ASSERT_EQ(baseline_values_[target], 1u);
      EXPECT_EQ(result_.eligibility[target], 0u); EXPECT_EQ(result_.filtered[target], 0u);
      expect_binary_subset();
    }
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ForgedCrossLineOwnerConflictFallsBackForBothUnits) {
    SCOPED_TRACE("Explicit old-owner fault injection; the two current line covers are disjoint");
    const cohort_rect_t left {28u, 6u, 68u, 24u}, right {92u, 6u, 132u, 24u};
    author(1u, {}); ASSERT_TRUE(run());
    for (std::uint64_t frame = 2u; frame <= 3u; ++frame) {
      author(frame, {left, right}); paint({40u, 8u, 44u, 21u}); paint({106u, 8u, 110u, 21u}); ASSERT_TRUE(run());
    }
    ASSERT_EQ(phase(0u), SBC_PHASE_ACTIVE); ASSERT_EQ(phase(1u), SBC_PHASE_ACTIVE);
    const auto left_id = id(0u), right_id = id(1u);
    const auto target = pixel(107u, 12u);
    auto forged_owner = result_.owner, forged_rgb = result_.birth_rgb;
    forged_owner[target] = left_id; forged_rgb[target] = result_.birth_rgb[pixel(41u, 12u)];
    unbind();
    context_->UpdateSubresource(old_owner_.texture.Get(), 0u, nullptr, forged_owner.data(), width_ * 4u, 0u);
    context_->UpdateSubresource(old_rgb_.texture.Get(), 0u, nullptr, forged_rgb.data(), width_ * 4u, 0u);
    author(4u, {left, right}); paint({40u, 8u, 44u, 21u}); paint({106u, 8u, 110u, 21u}); ASSERT_TRUE(run());
    EXPECT_EQ(id(0u), left_id); EXPECT_EQ(id(1u), right_id);
    EXPECT_EQ(phase(0u), SBC_PHASE_SUSPENDED); EXPECT_EQ(phase(1u), SBC_PHASE_SUSPENDED);
    for (UINT slot = 0u; slot < 2u; ++slot) {
      EXPECT_NE(result_.state[cohort::entry_offset(slot) + SBC_E_FLAGS] & SBC_ENTRY_AMBIGUOUS_PIXELS, 0u);
    }
    EXPECT_EQ(result_.owner[target], 0u);
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, InvalidOldPinAuthorityIsNotAnOcrBlankWitness) {
    author(1u, {{30u, 6u, 130u, 24u}}); glyph();
    plane_words_[cv2::subtitle_plane_flags_word] = 0u;
    plane_words_[cv2::subtitle_plane_current_slot_mask_word] = 0u;
    identity_words_[2u] = 0u; ASSERT_TRUE(run());
    author(2u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    EXPECT_NE(phase(), SBC_PHASE_ACTIVE); expect_fallback();
    author(3u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    EXPECT_NE(phase(), SBC_PHASE_ACTIVE); expect_fallback();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, UniqueLineSlotReorderPreservesIdsAndAmbiguousMergeFallsBack) {
    const cohort_rect_t left {28u, 6u, 68u, 24u}, right {92u, 6u, 132u, 24u};
    author(1u, {}); ASSERT_TRUE(run());
    author(2u, {left, right}); paint({40u, 8u, 44u, 21u}); paint({106u, 8u, 110u, 21u}); ASSERT_TRUE(run());
    author(3u, {left, right}); paint({40u, 8u, 44u, 21u}); paint({106u, 8u, 110u, 21u}); ASSERT_TRUE(run());
    ASSERT_EQ(phase(0u), SBC_PHASE_ACTIVE); ASSERT_EQ(phase(1u), SBC_PHASE_ACTIVE);
    const auto left_id = result_.owner[pixel(41u, 12u)], right_id = result_.owner[pixel(107u, 12u)];
    ASSERT_NE(left_id, right_id); ASSERT_NE(left_id, 0u); ASSERT_NE(right_id, 0u);
    author(4u, {right, left}); paint({40u, 8u, 44u, 21u}); paint({106u, 8u, 110u, 21u}); ASSERT_TRUE(run());
    EXPECT_EQ(result_.owner[pixel(41u, 12u)], left_id); EXPECT_EQ(result_.owner[pixel(107u, 12u)], right_id);
    author(5u, {{28u, 6u, 132u, 24u}}); paint({40u, 8u, 44u, 21u}); paint({106u, 8u, 110u, 21u}); ASSERT_TRUE(run());
    expect_fallback();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, AmbiguousLineSplitCannotBorrowOneOldCueTwice) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    author(4u, {{30u, 6u, 80u, 24u}, {80u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, PartialSameBoxReplacementCannotClipNewStroke) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto old_id = id();
    const auto known = result_.owner, fixed = result_.birth_rgb;
    author(4u, {{30u, 6u, 130u, 24u}}); glyph();
    paint({84u, 8u, 88u, 21u}, .15f, false); paint({100u, 8u, 104u, 21u}); ASSERT_TRUE(run());
    EXPECT_EQ(result_.filtered[pixel(101u, 12u)], baseline_values_[pixel(101u, 12u)]);
    EXPECT_EQ(phase(), SBC_PHASE_ACTIVE); EXPECT_EQ(id(), old_id);
    EXPECT_EQ(result_.owner[pixel(101u, 12u)], SBC_OWNER_UNKNOWN_BIT | old_id);
    expect_known_members_unchanged(known, fixed);
    expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, OtherColourPartialReplacementGetsLocalUnknownWithoutKnownAge) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto original_id = id();
    const auto original_owner = result_.owner, original_rgb = result_.birth_rgb;
    const auto offset = cohort::entry_offset(0u) + SBC_E_BIRTH_FRAME;
    const std::vector<UINT> birth(result_.state.begin() + offset, result_.state.begin() + offset + 4u);
    author(4u, {{30u, 6u, 130u, 24u}}); glyph();
    // Three white strokes remain strict stable anchors. One old stroke disappears;
    // a different-color new stroke has current strong P, edge and authored baseline S=1.
    paint({84u, 8u, 88u, 21u}, .15f, false);
    for (UINT y = 8u; y < 21u; ++y) {
      for (UINT x = 100u; x < 104u; ++x) {
        source_values_[pixel(x, y)] = {cohort_half(.10f), cohort_half(.85f), cohort_half(.95f), cohort_half(1.f)};
        baseline_values_[pixel(x, y)] = 1u;
      }
    }
    ASSERT_TRUE(run());
    EXPECT_EQ(phase(), SBC_PHASE_ACTIVE); EXPECT_EQ(id(), original_id);
    expect_known_members_unchanged(original_owner, original_rgb);
    EXPECT_EQ(result_.owner[pixel(101u, 12u)], SBC_OWNER_UNKNOWN_BIT | original_id);
    EXPECT_EQ(result_.filtered[pixel(101u, 12u)], baseline_values_[pixel(101u, 12u)]);
    const std::vector<UINT> current_birth(result_.state.begin() + offset, result_.state.begin() + offset + 4u);
    EXPECT_EQ(current_birth, birth); expect_binary_subset();
  }

  // Enabled source-only counterexample for an outline-only visible onset. The
  // future glyph's white fill is already present as background before cue birth.
  TEST_F(HostSbsSubtitleCohortGpuTest, NewBlackOutlineOnUnchangedWhiteFillGetsImmediateUnknownProtection) {
    const cohort_rect_t cover {30u, 6u, 130u, 24u};
    const cohort_rect_t white_background {100u, 8u, 112u, 21u};
    author(1u, {}); paint(white_background, .95f, false); ASSERT_TRUE(run());
    for (std::uint64_t frame = 2u; frame <= 3u; ++frame) {
      author(frame, {cover}); glyph(); paint(white_background, .95f, false);
      ASSERT_TRUE(run());
    }
    ASSERT_EQ(phase(), SBC_PHASE_ACTIVE);
    const auto original_id = id();
    const auto original_owner = result_.owner, original_rgb = result_.birth_rgb;
    const auto previous_source = result_.previous_source;
    const auto offset = cohort::entry_offset(0u) + SBC_E_BIRTH_FRAME;
    const std::vector<UINT> birth(result_.state.begin() + offset, result_.state.begin() + offset + 4u);
    ASSERT_EQ(result_.owner[pixel(104u, 8u)], 0u);

    author(4u, {cover}); glyph(); paint(white_background, .95f, false);
    const std::array<cohort_rect_t, 4u> outline {
      cohort_rect_t {99u, 7u, 113u, 8u}, cohort_rect_t {99u, 21u, 113u, 22u},
      cohort_rect_t {99u, 8u, 100u, 21u}, cohort_rect_t {112u, 8u, 113u, 21u}};
    for (const auto &rect : outline) {
      paint(rect, .05f);
    }
    // White body support can become current without an RGB change at that pixel.
    for (const auto &rect : {cohort_rect_t {100u, 8u, 112u, 10u}, cohort_rect_t {100u, 19u, 112u, 21u},
                            cohort_rect_t {100u, 10u, 102u, 19u}, cohort_rect_t {110u, 10u, 112u, 19u}}) {
      paint(rect, .95f);
    }
    ASSERT_EQ(source_values_[pixel(104u, 8u)], previous_source[pixel(104u, 8u)]);
    ASSERT_EQ(baseline_values_[pixel(99u, 12u)], 1u);
    // Nearest old born bar ends at x87; the new outline begins at x99. The
    // >=12-pixel gap exceeds M1's three growth steps plus three-pixel collar.
    ASSERT_TRUE(run());
    EXPECT_EQ(id(), original_id); EXPECT_EQ(phase(), SBC_PHASE_ACTIVE);
    expect_known_members_unchanged(original_owner, original_rgb);
    EXPECT_EQ(result_.owner[pixel(99u, 12u)], SBC_OWNER_UNKNOWN_BIT | original_id);
    EXPECT_EQ(result_.owner[pixel(104u, 8u)], SBC_OWNER_UNKNOWN_BIT | original_id);
    bool every_new_black_support_preserved = true;
    for (const auto &rect : outline) {
      for (UINT y = rect[1u]; y < rect[3u]; ++y) {
        for (UINT x = rect[0u]; x < rect[2u]; ++x) {
          every_new_black_support_preserved &= result_.filtered[pixel(x, y)] == baseline_values_[pixel(x, y)];
          EXPECT_EQ(result_.filtered[pixel(x, y)], baseline_values_[pixel(x, y)]) << x << ',' << y;
        }
      }
    }
    EXPECT_TRUE(every_new_black_support_preserved);
    const std::vector<UINT> current_birth(result_.state.begin() + offset, result_.state.begin() + offset + 4u);
    EXPECT_EQ(current_birth, birth); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, PreviousSupportIsRawAndStaticBackgroundDoesNotCreep) {
    const cohort_rect_t point {107u, 12u, 108u, 13u};
    author(1u, {}); paint(point, .95f, false); ASSERT_TRUE(run());
    for (std::uint64_t frame = 2u; frame <= 7u; ++frame) {
      author(frame, {{30u, 6u, 130u, 24u}}); glyph(); paint(point, .95f); ASSERT_TRUE(run());
      ASSERT_EQ(result_.previous_support, baseline_values_);
      if (frame >= 3u) {
        EXPECT_EQ(phase(), SBC_PHASE_ACTIVE);
        EXPECT_EQ(result_.previous_support[pixel(107u, 12u)], 1u);
        EXPECT_EQ(result_.filtered[pixel(107u, 12u)], 0u);
        EXPECT_EQ(result_.owner[pixel(107u, 12u)], 0u);
        EXPECT_FALSE(std::any_of(result_.owner.begin(), result_.owner.end(), [](UINT owner) { return is_unknown(owner); }));
      }
      expect_binary_subset();
    }
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, WeakHighProbabilityOffGridStrokeGetsImmediateUnknown) {
    const cohort_rect_t point {107u, 12u, 108u, 13u};
    author(1u, {}); paint(point, .90f, false); ASSERT_TRUE(run());
    for (std::uint64_t frame = 2u; frame <= 3u; ++frame) {
      author(frame, {{30u, 6u, 130u, 24u}}); glyph(); paint(point, .90f); ASSERT_TRUE(run());
    }
    ASSERT_EQ(phase(), SBC_PHASE_ACTIVE); ASSERT_EQ(result_.owner[pixel(107u, 12u)], 0u);
    const auto line = id(); const auto known = result_.owner, rgb = result_.birth_rgb;
    const auto previous = result_.previous_source[pixel(107u, 12u)];
    ASSERT_EQ(result_.previous_support[pixel(107u, 12u)], 1u);
    author(4u, {{30u, 6u, 130u, 24u}}); glyph(); paint(point, .95f);
    ASSERT_FALSE(on_current_palette_grid(107u, 12u));
    ASSERT_GT(cohort_float(source_values_[pixel(107u, 12u)][0u]) - cohort_float(previous[0u]), 0.f);
    ASSERT_LT(cohort_float(source_values_[pixel(107u, 12u)][0u]) - cohort_float(previous[0u]), cv2::subtitle_contour_history_rgb_distance);
    ASSERT_TRUE(run());
    EXPECT_EQ(phase(), SBC_PHASE_ACTIVE); EXPECT_EQ(result_.owner[pixel(107u, 12u)], SBC_OWNER_UNKNOWN_BIT | line);
    EXPECT_EQ(result_.filtered[pixel(107u, 12u)], 1u); expect_known_members_unchanged(known, rgb); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, UnknownRefreshFadeAndSupportFlickerNeverBorrowKnownAge) {
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto line = id(); const auto known = result_.owner, rgb = result_.birth_rgb;
    const auto known_birth = birth_words();
    const cohort_rect_t point {107u, 12u, 108u, 13u};
    author(4u, {{30u, 6u, 130u, 24u}}); glyph(); paint(point); ASSERT_TRUE(run());
    const auto first_reference = result_.birth_rgb[pixel(107u, 12u)];
    ASSERT_EQ(result_.owner[pixel(107u, 12u)], SBC_OWNER_UNKNOWN_BIT | line);
    for (std::uint64_t frame = 5u; frame <= 7u; ++frame) {
      author(frame, {{30u, 6u, 130u, 24u}}); glyph(); paint(point, .90f, frame != 6u); ASSERT_TRUE(run());
      EXPECT_EQ(phase(), SBC_PHASE_ACTIVE);
      EXPECT_EQ(birth_words(), known_birth);
      EXPECT_EQ(result_.owner[pixel(107u, 12u)], SBC_OWNER_UNKNOWN_BIT | line);
      EXPECT_EQ(result_.birth_rgb[pixel(107u, 12u)], first_reference);
      EXPECT_EQ(result_.filtered[pixel(107u, 12u)], frame == 6u ? 0u : 1u);
      expect_known_members_unchanged(known, rgb); expect_binary_subset();
    }
    author(8u, {{30u, 6u, 130u, 24u}}); glyph(); paint(point, .85f); ASSERT_TRUE(run());
    EXPECT_EQ(result_.owner[pixel(107u, 12u)], SBC_OWNER_UNKNOWN_BIT | line);
    EXPECT_NE(result_.birth_rgb[pixel(107u, 12u)], first_reference);
    EXPECT_EQ(result_.birth_rgb[pixel(107u, 12u)], packed_color(source_values_[pixel(107u, 12u)]));
    EXPECT_EQ(result_.filtered[pixel(107u, 12u)], 1u); expect_known_members_unchanged(known, rgb); expect_binary_subset();
    EXPECT_EQ(birth_words(), known_birth);
    author(9u, {{30u, 6u, 130u, 24u}}); glyph(); paint(point, .70f, false); ASSERT_TRUE(run());
    EXPECT_EQ(result_.owner[pixel(107u, 12u)], 0u); EXPECT_EQ(result_.birth_rgb[pixel(107u, 12u)], 0u);
    EXPECT_EQ(result_.filtered[pixel(107u, 12u)], 0u); expect_known_members_unchanged(known, rgb); expect_binary_subset();
    EXPECT_EQ(birth_words(), known_birth);
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ExactUnknownDependencyDoesNotPromotePermittedNeighborsToSeeds) {
    const cohort_rect_t inside {106u, 16u, 107u, 17u}, outside {106u, 18u, 107u, 19u}, seed {100u, 12u, 101u, 13u};
    author(1u, {}); paint(inside, .95f, false); paint(outside, .95f, false); ASSERT_TRUE(run());
    for (std::uint64_t frame = 2u; frame <= 3u; ++frame) {
      author(frame, {{30u, 6u, 130u, 24u}}); glyph(); paint(inside); paint(outside); ASSERT_TRUE(run());
    }
    ASSERT_EQ(phase(), SBC_PHASE_ACTIVE); ASSERT_EQ(result_.owner[pixel(106u, 16u)], 0u); ASSERT_EQ(result_.owner[pixel(106u, 18u)], 0u);
    const auto line = id(); const auto known = result_.owner, rgb = result_.birth_rgb;
    const auto known_birth = birth_words();
    ASSERT_TRUE(in_unknown_dependency(6, 4)); ASSERT_FALSE(in_unknown_dependency(6, 6));
    for (std::uint64_t frame = 4u; frame <= 6u; ++frame) {
      author(frame, {{30u, 6u, 130u, 24u}}); glyph(); paint(inside); paint(outside); paint(seed, .95f, frame != 5u); ASSERT_TRUE(run());
      EXPECT_EQ(result_.owner[pixel(100u, 12u)], SBC_OWNER_UNKNOWN_BIT | line);
      EXPECT_EQ(result_.filtered[pixel(100u, 12u)], frame == 5u ? 0u : 1u);
      EXPECT_EQ(result_.filtered[pixel(106u, 16u)], 1u); EXPECT_EQ(result_.filtered[pixel(106u, 18u)], 0u);
      EXPECT_EQ(result_.owner[pixel(106u, 16u)], 0u); EXPECT_EQ(result_.owner[pixel(106u, 18u)], 0u);
      EXPECT_EQ(birth_words(), known_birth);
      expect_known_members_unchanged(known, rgb); expect_binary_subset();
    }
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, PendingRawHalfChangeWithSameRgb8RetainsUnknownOnActivation) {
    const cohort_rect_t point {107u, 12u, 108u, 13u};
    author(1u, {}); paint(point, .85f, false); ASSERT_TRUE(run());
    const auto previous = result_.previous_source[pixel(107u, 12u)];
    author(2u, {{30u, 6u, 130u, 24u}}); glyph(); paint(point, .8505859375f);
    ASSERT_NE(source_values_[pixel(107u, 12u)][0u], previous[0u]);
    ASSERT_EQ(packed_color(source_values_[pixel(107u, 12u)]), packed_color(previous));
    ASSERT_TRUE(run()); ASSERT_EQ(phase(), SBC_PHASE_PENDING);
    const auto line = id();
    EXPECT_EQ(result_.owner[pixel(107u, 12u)], SBC_OWNER_UNKNOWN_BIT | line); expect_binary_subset();
    author(3u, {{30u, 6u, 130u, 24u}}); glyph(); paint(point, .8505859375f); ASSERT_TRUE(run());
    EXPECT_EQ(phase(), SBC_PHASE_ACTIVE); EXPECT_EQ(result_.owner[pixel(107u, 12u)], SBC_OWNER_UNKNOWN_BIT | line);
    EXPECT_EQ(birth_words()[0u], 2u);
    EXPECT_EQ(result_.filtered[pixel(107u, 12u)], 1u); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, PendingLatentWhiteBodyUsesOnlyNewKnownDependencyNotFarBackground) {
    const cohort_rect_t white {100u, 8u, 112u, 21u}, far_body {125u, 12u, 126u, 13u};
    author(1u, {}); paint(white, .95f, false); paint(far_body, .95f, false); ASSERT_TRUE(run());
    UINT line = 0u;
    for (std::uint64_t frame = 2u; frame <= 3u; ++frame) {
      author(frame, {{30u, 6u, 130u, 24u}}); glyph(); paint(white, .95f, false); paint(far_body);
      for (const auto &r : {cohort_rect_t {99u, 7u, 113u, 8u}, cohort_rect_t {99u, 21u, 113u, 22u},
                            cohort_rect_t {99u, 8u, 100u, 21u}, cohort_rect_t {112u, 8u, 113u, 21u}}) paint(r, .05f);
      for (const auto &r : {cohort_rect_t {100u, 8u, 112u, 10u}, cohort_rect_t {100u, 19u, 112u, 21u},
                            cohort_rect_t {100u, 10u, 102u, 19u}, cohort_rect_t {110u, 10u, 112u, 19u}}) paint(r);
      ASSERT_TRUE(run()); line = id();
      EXPECT_EQ(result_.owner[pixel(104u, 8u)], SBC_OWNER_UNKNOWN_BIT | line);
      EXPECT_EQ(result_.owner[pixel(125u, 12u)], 0u);
      if (frame == 2u) {
        EXPECT_EQ(phase(), SBC_PHASE_PENDING); EXPECT_EQ(result_.next_owner[pixel(104u, 8u)], 0u);
      } else {
        EXPECT_EQ(phase(), SBC_PHASE_ACTIVE); EXPECT_EQ(result_.filtered[pixel(104u, 8u)], 1u);
        EXPECT_EQ(result_.filtered[pixel(125u, 12u)], 0u);
      }
      expect_binary_subset();
    }
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, Last31BitKnownIdAllocatesOnceAndExhaustionNeverWraps) {
    author(1u, {}); ASSERT_TRUE(run());
    auto forged = result_.state; forged[SBC_H_NEXT_ID] = SBC_OWNER_ID_MASK;
    unbind(); context_->UpdateSubresource(old_state_.buffer.Get(), 0u, nullptr, forged.data(), 0u, 0u);
    for (std::uint64_t frame = 2u; frame <= 3u; ++frame) {
      author(frame, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
      EXPECT_EQ(id(), SBC_OWNER_ID_MASK); EXPECT_EQ(result_.state[SBC_H_NEXT_ID], SBC_OWNER_ID_LIMIT);
      EXPECT_TRUE(is_known(result_.owner[pixel(51u, 12u)])); expect_binary_subset();
    }
    author(4u, {}); ASSERT_TRUE(run());
    author(5u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    EXPECT_EQ(result_.state[SBC_H_NEXT_ID], SBC_OWNER_ID_LIMIT); EXPECT_EQ(id(), 0u);
    EXPECT_NE(result_.state[SBC_H_FLAGS] & SBC_HEADER_ID_EXHAUSTED, 0u);
    EXPECT_TRUE(std::all_of(result_.owner.begin(), result_.owner.end(), [](UINT owner) { return owner == 0u; }));
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, DuplicateRawSupportMutationInvalidatesWithoutSnapshotAdvance) {
    SCOPED_TRACE("Explicit duplicate-current-S mutation; not a genuine observation");
    ASSERT_NO_FATAL_FAILURE(acquire());
    const auto previous_raw = result_.previous_support;
    baseline_values_[pixel(107u, 12u)] = 1u;
    ASSERT_TRUE(run());
    EXPECT_EQ(result_.state[SBC_H_VALID], 0u); EXPECT_EQ(result_.state[SBC_H_COUNT], 0u);
    EXPECT_EQ(result_.previous_support, previous_raw);
    EXPECT_TRUE(std::all_of(result_.owner.begin(), result_.owner.end(), [](UINT owner) { return owner == 0u; }));
    expect_fallback(); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ResearchSnapshotCopiesCoherentNativeTupleWithoutChangingCohort) {
    author(0x100000004ull, {{30u, 6u, 130u, 24u}}, false, 0x200000007ull); glyph();
    source_values_.front() = {cohort_half(.3f), cohort_half(.4f), cohort_half(.5f), cohort_half(1.f)};
    source_values_.back() = {cohort_half(.7f), cohort_half(.6f), cohort_half(.2f), cohort_half(1.f)};
    baseline_values_.front() = baseline_values_.back() = 1u;
    probability_values_.front() = .1234f; probability_values_.back() = .8765f;
    const auto initial = cohort::initial_state();
    ASSERT_TRUE(snapshot_current_tuple());
    EXPECT_EQ(result_.previous_source, source_values_);
    EXPECT_EQ(result_.previous_support, baseline_values_);
    EXPECT_EQ(result_.previous_identity, identity_words_);
    ASSERT_EQ(result_.previous_p.size(), probability_values_.size());
    for (std::size_t i = 0u; i < probability_values_.size(); ++i) {
      ASSERT_EQ(result_.previous_p[i], std::bit_cast<UINT>(probability_values_[i])) << i;
    }
    EXPECT_TRUE(std::equal(initial.begin(), initial.end(), result_.state.begin(), result_.state.end()));
    EXPECT_TRUE(std::all_of(result_.owner.begin(), result_.owner.end(), [](UINT value) { return value == 0u; }));
    EXPECT_EQ(closure_,
      models::host_sbs_shader_cache::subtitle_cohort_prototype_research_source_closure_sha256);
    EXPECT_EQ(snapshot_closure_.size(), 64u); EXPECT_NE(snapshot_closure_, closure_);
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ResearchSnapshotReady23AllowsEmptyPinWithoutInventingProbabilityBlank) {
    author(1u, {});
    ASSERT_EQ(identity_words_[2u], 0u);
    ASSERT_EQ(identity_words_[cv2::subtitle_contour_history_snapshot_valid_word], 1u);
    probability_values_.back() = .9f;
    ASSERT_TRUE(snapshot_current_tuple());
    EXPECT_EQ(result_.previous_identity, identity_words_);
    EXPECT_EQ(result_.previous_identity[2u], 0u);
    EXPECT_EQ(result_.previous_identity[cv2::subtitle_contour_history_snapshot_valid_word], 1u);
    EXPECT_EQ(result_.previous_p.back(), std::bit_cast<UINT>(.9f));
    EXPECT_EQ(result_.previous_source, source_values_);
    author(2u, {}, true, 8u, 17u);
    ASSERT_TRUE(snapshot_current_tuple());
    EXPECT_EQ(result_.previous_identity, identity_words_);
    EXPECT_EQ(result_.previous_identity[3u], 17u);
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ResearchSnapshotRejectsStaleMalformedAndReservedIdentityAsAWholeTuple) {
    const std::array<UINT, 14u> fault_words {0u, 1u, 2u, 4u, 6u, 8u, 10u, 12u, 14u, 16u, 20u, 22u, 23u, 24u};
    for (const auto word : fault_words) {
      SCOPED_TRACE("Explicit current SCT fault word " + std::to_string(word));
      author(4u, {{30u, 6u, 130u, 24u}}); glyph();
      ++identity_words_[word];
      ASSERT_TRUE(snapshot_current_tuple());
      EXPECT_TRUE(std::all_of(result_.previous_identity.begin(), result_.previous_identity.end(), [](UINT value) { return value == 0u; }));
      EXPECT_EQ(result_.previous_source, source_values_);
      EXPECT_EQ(result_.previous_support, baseline_values_);
    }
    for (UINT fault = 0u; fault < 12u; ++fault) {
      SCOPED_TRACE("Explicit current snapshot CB fault " + std::to_string(fault));
      author(4u, {{30u, 6u, 130u, 24u}}); glyph();
      switch (fault) {
        case 0u: constants_.observation[3u] = 1u; break;
        case 1u: constants_.observation[2u] = 2u; break;
        case 2u: constants_.control[0u] = 0u; break;
        case 3u: ++constants_.control[1u]; break;
        case 4u: constants_.control[2u] = 1u; break;
        case 5u: constants_.control[3u] = 1u; break;
        case 6u: constants_.frame[0u] = constants_.frame[1u] = 0u; break;
        case 7u: constants_.observation[0u] = constants_.observation[1u] = 0u; break;
        case 8u: constants_.field[0u] = 0u; break;
        case 9u: constants_.field[1u] = std::numeric_limits<UINT>::max(); break;
        case 10u: constants_.content[2u] = constants_.content[0u]; break;
        case 11u: ++constants_.field[2u]; break;
      }
      ASSERT_TRUE(snapshot_current_tuple());
      EXPECT_TRUE(std::all_of(result_.previous_identity.begin(), result_.previous_identity.end(), [](UINT value) { return value == 0u; }));
      EXPECT_EQ(result_.previous_source, source_values_);
      EXPECT_EQ(result_.previous_support, baseline_values_);
    }
    // An invalid old reference cannot become a positive inverse-onset witness.
    author(1u, {}); identity_words_[cv2::subtitle_contour_history_snapshot_valid_word] = 0u;
    ASSERT_TRUE(snapshot_current_tuple());
    author(2u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    EXPECT_NE(phase(), SBC_PHASE_ACTIVE); EXPECT_NE(phase(), SBC_PHASE_PENDING); expect_fallback();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ResearchSnapshotDimensionMismatchInvalidatesAllPrivateDestinations) {
    author(4u, {{30u, 6u, 130u, 24u}}); glyph();
    ++constants_.source[0u];
    ASSERT_TRUE(snapshot_current_tuple());
    EXPECT_TRUE(std::all_of(result_.previous_identity.begin(), result_.previous_identity.end(), [](UINT value) { return value == 0u; }));
    EXPECT_TRUE(std::all_of(result_.previous_p.begin(), result_.previous_p.end(), [](UINT value) { return value == 0u; }));
    EXPECT_TRUE(std::all_of(result_.previous_support.begin(), result_.previous_support.end(), [](UINT value) { return value == 0u; }));
    EXPECT_TRUE(std::all_of(result_.previous_source.begin(), result_.previous_source.end(), [](const cohort_half_t &value) { return value == cohort_half_t {}; }));
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ResearchSnapshotPreservesRawFaultsForFollowingPrepareValidation) {
    author(1u, {});
    probability_values_.back() = std::bit_cast<float>(UINT(0x7fc12345u));
    baseline_values_.front() = 2u;
    ASSERT_TRUE(snapshot_current_tuple());
    EXPECT_EQ(result_.previous_p.back(), 0x7fc12345u);
    EXPECT_EQ(result_.previous_support.front(), 2u);
    EXPECT_EQ(result_.previous_identity, identity_words_);
    author(2u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    EXPECT_EQ(result_.state[SBC_H_VALID], 0u); expect_fallback();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ResearchSnapshotZeroDispatchHoldsEveryPrivatePlane) {
    author(1u, {}); probability_values_.back() = .1234f;
    ASSERT_TRUE(snapshot_current_tuple()); const auto saved = result_;
    author(2u, {{30u, 6u, 130u, 24u}}); glyph();
    ASSERT_TRUE(snapshot_current_tuple(true));
    EXPECT_EQ(result_, saved);
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, ResearchSnapshotAfterReadersFeedsOnlyTheFollowingObservation) {
    author(1u, {}); ASSERT_TRUE(snapshot_current_tuple());
    const auto quiet_source = result_.previous_source;
    author(2u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    ASSERT_EQ(phase(), SBC_PHASE_PENDING); EXPECT_EQ(consumed_previous_source_, quiet_source);
    const auto pending_state = result_.state, pending_owner = result_.owner, pending_rgb = result_.birth_rgb;
    ASSERT_TRUE(snapshot_current_tuple());
    EXPECT_EQ(result_.state, pending_state); EXPECT_EQ(result_.owner, pending_owner); EXPECT_EQ(result_.birth_rgb, pending_rgb);
    EXPECT_EQ(result_.previous_identity, identity_words_);
    author(3u, {{30u, 6u, 130u, 24u}}); glyph(); ASSERT_TRUE(run());
    EXPECT_EQ(phase(), SBC_PHASE_ACTIVE); expect_binary_subset();
  }

  TEST_F(HostSbsSubtitleCohortGpuTest, SavedNativeSequenceReplay) {
    const auto *cases_environment = std::getenv("SUNSHINE_TEST_COHORT_CASES");
    const auto *output_environment = std::getenv("SUNSHINE_TEST_COHORT_OUTPUT");
    if (!cases_environment && !output_environment) GTEST_SKIP() << "Set SUNSHINE_TEST_COHORT_CASES/OUTPUT for the opt-in GPU state diagnostic";
    ASSERT_NE(cases_environment, nullptr); ASSERT_NE(output_environment, nullptr);
    const std::filesystem::path path(cases_environment), output_root(output_environment);
    ASSERT_FALSE(output_root.empty());
    ASSERT_TRUE(!std::filesystem::exists(output_root) || (std::filesystem::is_directory(output_root) && std::filesystem::is_empty(output_root)));
    const auto bytes = cohort_read_bytes(path);
    const auto spec = nlohmann::json::parse(bytes.begin(), bytes.end());
    ASSERT_EQ(spec.at("schema"), 1);
    const auto &cases = spec.at("cases"); ASSERT_TRUE(cases.is_array()); ASSERT_GT(cases.size(), 0u); ASSERT_LE(cases.size(), 32u);
    ASSERT_EQ(spec.at("prototype_closure_sha256").get<std::string>(), closure_);
    ComPtr<IDXGIDevice> dxgi; ComPtr<IDXGIAdapter> adapter; DXGI_ADAPTER_DESC adapter_description {};
    ASSERT_TRUE(SUCCEEDED(device_.As(&dxgi))); ASSERT_TRUE(SUCCEEDED(dxgi->GetAdapter(&adapter)));
    ASSERT_TRUE(SUCCEEDED(adapter->GetDesc(&adapter_description)));
    std::filesystem::create_directories(output_root);
    nlohmann::json summary { {"schema", 1}, {"kind", "native-gpu-cohort-state-prototype-replay"},
      {"scope", "Diagnostic only; no production/renderer/GT/quality authority"},
      {"input_specification", path.string()}, {"input_specification_sha256", cohort_sha256(bytes.data(), bytes.size())},
      {"driver", driver_name_}, {"shader_root", shader_root_.string()}, {"prototype_closure_sha256", closure_},
      {"generated_production_manifest_claimed", false}, {"CPU_eligibility_input", false},
      {"previous_unfiltered_baseline_S_is_GPU_snapshot", true}, {"CPU_mask_derivation", false},
      {"readback_drives_algorithm", false}, {"cases", nlohmann::json::array()} };
    const auto write = [](const std::filesystem::path &destination, const void *data, std::size_t size) {
      std::ofstream file(destination, std::ios::binary); file.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
      if (!file) throw std::runtime_error("Cannot write cohort output: " + destination.string());
    };
    std::string previous_sequence;
    for (std::size_t index = 0u; index < cases.size(); ++index) {
      SCOPED_TRACE(index); const auto &entry = cases[index];
      const auto sequence = entry.at("sequence").get<std::string>(); ASSERT_FALSE(sequence.empty());
      const auto source = entry.at("source").get<std::array<UINT, 2u>>();
      const auto field = entry.at("field").get<std::array<UINT, 2u>>();
      const auto content = entry.at("content").get<cohort_rect_t>();
      ASSERT_GT(source[0u], 40u); ASSERT_LE(source[0u], cv2::subtitle_contour_source_max_extent);
      ASSERT_GT(source[1u], 0u); ASSERT_LE(source[1u], cv2::subtitle_contour_source_max_extent);
      ASSERT_TRUE(cv2::subtitle_ocr_field_is_calibrated(field[0u], field[1u]));
      ASSERT_LT(content[0u], content[2u]); ASSERT_LT(content[1u], content[3u]);
      ASSERT_LE(content[2u], field[0u]); ASSERT_LE(content[3u], field[1u]);
      const bool sequence_changed = sequence != previous_sequence;
      if (sequence_changed) {
        ASSERT_TRUE(resize(source[0u], source[1u], field[0u], field[1u], content));
        previous_sequence = sequence;
      } else {
        ASSERT_EQ(source[0u], width_); ASSERT_EQ(source[1u], height_);
        // Same-sized domain changes are shader evidence, not a fixture state reset.
        field_width_ = field[0u]; field_height_ = field[1u]; content_ = content;
      }
      const auto frame = entry.at("frame").get<std::uint64_t>();
      const auto time = entry.at("observation_timestamp_us").get<std::uint64_t>();
      const auto generation = entry.at("generation").get<std::uint64_t>();
      ASSERT_GT(frame, 0u); ASSERT_GT(time, 0u);
      author(frame, {}, entry.value("reset_before", false), generation, entry.at("cut_epoch").get<UINT>());
      constants_.observation[0u] = UINT(time); constants_.observation[1u] = UINT(time >> 32u);
      const auto load = [&](const char *key) {
        const auto &binding = entry.at("artifacts").at(key);
        const auto input_path = path.parent_path() / binding.at("path").get<std::string>();
        auto values = cohort_read_bytes(input_path);
        if (cohort_sha256(values.data(), values.size()) != binding.at("sha256").get<std::string>())
          throw std::runtime_error("Cohort input digest mismatch: " + input_path.string());
        if (binding.contains("bytes") && values.size() != binding.at("bytes").get<std::size_t>())
          throw std::runtime_error("Cohort input size mismatch: " + input_path.string());
        return values;
      };
      const auto source_bytes = load("source_band"), p_bytes = load("probability"), ocr_bytes = load("ocr_record");
      const auto plane_bytes = load("current_plane"), identity_bytes = load("current_identity"), support_bytes = load("baseline_support");
      ASSERT_EQ(source_bytes.size(), source_values_.size() * sizeof(cohort_half_t));
      ASSERT_EQ(p_bytes.size(), probability_values_.size() * sizeof(float)); ASSERT_EQ(ocr_bytes.size(), ocr_words_.size() * 4u);
      ASSERT_EQ(plane_bytes.size(), plane_words_.size() * 4u); ASSERT_EQ(identity_bytes.size(), identity_words_.size() * 4u);
      ASSERT_EQ(support_bytes.size(), baseline_values_.size() * 4u);
      std::memcpy(source_values_.data(), source_bytes.data(), source_bytes.size());
      std::memcpy(probability_values_.data(), p_bytes.data(), p_bytes.size());
      std::memcpy(ocr_words_.data(), ocr_bytes.data(), ocr_bytes.size());
      std::memcpy(plane_words_.data(), plane_bytes.data(), plane_bytes.size());
      std::memcpy(identity_words_.data(), identity_bytes.data(), identity_bytes.size());
      std::memcpy(baseline_values_.data(), support_bytes.data(), support_bytes.size());
      ASSERT_TRUE(std::all_of(baseline_values_.begin(), baseline_values_.end(), [](UINT value) { return value <= 1u; }));
      const bool zero = entry.value("dispatch_zero", false);
      const auto before = result_;
      ASSERT_TRUE(run(zero));
      if (zero && !before.state.empty()) {
        EXPECT_EQ(result_, before);
      }
      if (!zero) {
        ASSERT_NO_FATAL_FAILURE(expect_binary_subset());
      }
      const bool fallback = std::all_of(result_.eligibility.begin(), result_.eligibility.end(), [](UINT value) { return value == 1u; });
      if (!zero && fallback) {
        EXPECT_EQ(result_.filtered, baseline_values_);
      }
      const auto case_root = output_root / ("case_" + std::to_string(index));
      ASSERT_TRUE(std::filesystem::create_directory(case_root));
      nlohmann::json artifacts = nlohmann::json::object();
      const auto save = [&](const char *name, const auto &values) {
        const auto size = values.size() * sizeof(values[0u]); write(case_root / name, values.data(), size);
        artifacts[name] = {{"path", (case_root / name).string()}, {"sha256", cohort_sha256(values.data(), size)}, {"bytes", size}};
      };
      save("eligibility.u32", result_.eligibility); save("filtered_support.u32", result_.filtered);
      save("cohort_state.u32", result_.state); save("cohort_owner.u32", result_.owner); save("cohort_birth_rgb8.u32", result_.birth_rgb);
      save("next_state.u32", result_.next_state); save("next_owner.u32", result_.next_owner); save("next_birth_rgb8.u32", result_.next_rgb);
      save("previous_source.rgba16f", result_.previous_source); save("previous_probability.f32", result_.previous_p);
      save("previous_identity.u32", result_.previous_identity);
      save("uploaded_source.rgba16f", source_values_); save("uploaded_probability.f32", probability_values_);
      save("ocr_record.u32", ocr_words_); save("current_plane.u32", plane_words_); save("current_identity.u32", identity_words_);
      save("baseline_support.u32", baseline_values_);
      save("consumed_previous_support.u32", consumed_previous_support_);
      save("consumed_previous_source.rgba16f", consumed_previous_source_);
      save("previous_support.u32", result_.previous_support);
      std::vector<UINT> constant_words(sizeof(constants_) / 4u);
      std::memcpy(constant_words.data(), &constants_, sizeof(constants_)); save("constants.u32", constant_words);
      UINT active = 0u, pending = 0u;
      for (UINT slot = 0u; slot < SBC_CAPACITY; ++slot) {
        const auto phase_value = result_.state.at(cohort::entry_offset(slot) + SBC_E_PHASE);
        active += phase_value == SBC_PHASE_ACTIVE; pending += phase_value == SBC_PHASE_PENDING;
      }
      nlohmann::json record { {"schema", 1}, {"kind", "native-gpu-generated-cohort-state-diagnostic"},
        {"case", entry}, {"sequence_changed", sequence_changed}, {"dispatch_zero", zero},
        {"prototype_closure_sha256", closure_}, {"driver", driver_name_},
        {"adapter", {{"vendor_id", adapter_description.VendorId}, {"device_id", adapter_description.DeviceId},
          {"luid_low", adapter_description.AdapterLuid.LowPart}, {"luid_high", adapter_description.AdapterLuid.HighPart}}},
        {"artifacts", artifacts}, {"active_cues", active}, {"pending_cues", pending}, {"fallback_all_ones", fallback},
        {"known_owner_pixels", std::count_if(result_.owner.begin(), result_.owner.end(), [](UINT owner) { return is_known(owner); })},
        {"unknown_owner_pixels", std::count_if(result_.owner.begin(), result_.owner.end(), [](UINT owner) { return is_unknown(owner); })},
        {"filtered_support_count", std::count(result_.filtered.begin(), result_.filtered.end(), 1u)},
        {"production_qualification_claimed", false}, {"GT_or_quality_asserted", false},
        {"previous_unfiltered_baseline_S_is_GPU_snapshot", true},
        {"previous_S_ever_derived_from_filtered_support", false},
        {"CPU_mask_derivation", false},
        {"scope", "Source/P/OCR/UIP/SCT/current baseline-S inputs; GPU-generated KNOWN/UNKNOWN roles and eligibility; previous raw S snapshots advance with RGB/P/SCT only on genuine observations; no CPU bitmap or production drawing claim"} };
      const auto text = record.dump(2) + '\n'; write(case_root / "cohort_fixture_identity.json", text.data(), text.size());
      summary["cases"].push_back({{"index", index}, {"sequence", sequence}, {"frame", frame}, {"output", case_root.string()},
        {"identity_sha256", cohort_sha256(text.data(), text.size())}, {"active_cues", active}, {"pending_cues", pending}});
      const auto summary_text = summary.dump(2) + '\n'; write(output_root / "cohort_replay_identity.json", summary_text.data(), summary_text.size());
    }
  }
}  // namespace

#endif
