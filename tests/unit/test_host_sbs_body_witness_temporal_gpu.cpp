/**
 * @file tests/unit/test_host_sbs_subtitle_inner_colour_gpu.cpp
 * @brief Opt-in complete current-input colour extraction plus frozen component operator.
 * Readbacks select diagnostic exports only; GPU-written mode arguments govern all work.
 */
#include <gtest/gtest.h>

#ifdef _WIN32
  #include "src/crypto.h"
  #include "src/depth_coordinate_v2.h"
  #include "src/host_sbs_body_witness_temporal_gpu.h"
  #include "src/host_sbs_body_witness_temporal_recorded.h"
  #include "src/host_sbs_temporal_frame_context_gpu.h"
  #include "src/host_sbs_shader_cache.h"
  #include "src/host_sbs_subtitle_cohort.h"

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <chrono>
  #include <cmath>
  #include <cstdint>
  #include <cstdlib>
  #include <cstring>
  #include <d3d11.h>
  #include <dxgi.h>
  #include <filesystem>
  #include <fstream>
  #include <map>
  #include <nlohmann/json.hpp>
  #include <stdexcept>
  #include <string>
  #include <string_view>
  #include <thread>
  #include <utility>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace colour_cv2 = models::depth_coordinate_v2;
  namespace temporal = models::host_sbs_body_witness_temporal_gpu;
  using colour_half_t = std::array<std::uint16_t, 4u>;
  using colour_rect_t = std::array<UINT, 4u>;

  struct alignas(16) colour_constants_t {
    std::array<UINT, 24u> words;
  };

  struct alignas(16) colour_request_t {
    std::array<UINT, 4u> words;
  };

  static_assert(sizeof(colour_constants_t) == 96u && sizeof(colour_request_t) == 16u);

  std::vector<char> colour_bytes(const std::filesystem::path &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0) {
      throw std::runtime_error("Cannot read inner-colour input: " + path.string());
    }
    std::vector<char> out(static_cast<std::size_t>(f.tellg()));
    f.seekg(0);
    if (!out.empty()) {
      f.read(out.data(), static_cast<std::streamsize>(out.size()));
    }
    if (!f) {
      throw std::runtime_error("Incomplete inner-colour input");
    }
    return out;
  }

  std::string colour_sha(const void *data, std::size_t size) {
    const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), size));
    constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (auto byte : digest) {
      out.push_back(digits[byte >> 4u]);
      out.push_back(digits[byte & 15u]);
    }
    return out;
  }

  std::vector<char> colour_bound(const nlohmann::json &b, std::size_t size) {
    auto out = colour_bytes(b.at("path").get<std::string>());
    if (out.size() != size || out.size() != b.at("bytes").get<std::size_t>() || colour_sha(out.data(), out.size()) != b.at("sha256").get<std::string>()) {
      throw std::runtime_error("Inner-colour binding mismatch");
    }
    return out;
  }

  template<class T>
  std::vector<char> colour_raw(const std::vector<T> &v) {
    std::vector<char> out(v.size() * sizeof(T));
    if (!out.empty()) {
      std::memcpy(out.data(), v.data(), out.size());
    }
    return out;
  }

  template<std::size_t N>
  std::vector<char> colour_raw(const std::array<UINT, N> &v) {
    std::vector<char> out(N * 4u);
    std::memcpy(out.data(), v.data(), out.size());
    return out;
  }

  std::vector<UINT> colour_words(const std::vector<char> &v) {
    if (v.size() % 4u != 0u) {
      throw std::runtime_error("Inner-colour word size mismatch");
    }
    std::vector<UINT> out(v.size() / 4u);
    if (!v.empty()) {
      std::memcpy(out.data(), v.data(), v.size());
    }
    return out;
  }

  std::uint16_t colour_half(float value) {
    const auto bits = std::bit_cast<UINT>(value);
    const auto sign = std::uint16_t((bits >> 16u) & 0x8000u);
    const auto exponent = (bits >> 23u) & 255u;
    auto fraction = bits & 0x7fffffu;
    if (exponent == 255u) {
      return sign | (fraction ? 0x7e00u : 0x7c00u);
    }
    const int e = int(exponent) - 127 + 15;
    if (e >= 31) {
      return sign | 0x7c00u;
    }
    if (e <= 0) {
      if (e < -10) {
        return sign;
      }
      fraction |= 0x800000u;
      const auto shift = unsigned(14 - e);
      return sign | std::uint16_t((fraction + (1u << (shift - 1u)) - 1u + ((fraction >> shift) & 1u)) >> shift);
    }
    fraction += 0xfffu + ((fraction >> 13u) & 1u);
    return sign | std::uint16_t((UINT(e) << 10u) + (fraction >> 13u));
  }

  float colour_float(std::uint16_t value) {
    const auto exponent = (value >> 10u) & 31u, mantissa = value & 1023u;
    const float magnitude = exponent == 0u  ? std::ldexp(float(mantissa), -24) :
                            exponent == 31u ? (mantissa ? NAN : INFINITY) :
                                              std::ldexp(1.f + float(mantissa) / 1024.f, int(exponent) - 15);
    return (value & 0x8000u) ? -magnitude : magnitude;
  }

  struct colour_texture_t {
    ComPtr<ID3D11Texture2D> value;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };

  struct colour_buffer_t {
    ComPtr<ID3D11Buffer> value;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };

  bool colour_texture(ID3D11Device *d, UINT w, UINT h, DXGI_FORMAT format, bool output, colour_texture_t &out) {
    out = {};
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = w;
    desc.Height = h;
    desc.MipLevels = desc.ArraySize = 1u;
    desc.Format = format;
    desc.SampleDesc.Count = 1u;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | (output ? D3D11_BIND_UNORDERED_ACCESS : 0u);
    return SUCCEEDED(d->CreateTexture2D(&desc, nullptr, &out.value)) && SUCCEEDED(d->CreateShaderResourceView(out.value.Get(), nullptr, &out.srv)) &&
           (!output || SUCCEEDED(d->CreateUnorderedAccessView(out.value.Get(), nullptr, &out.uav)));
  }

  bool colour_buffer(ID3D11Device *d, UINT words, bool output, colour_buffer_t &out) {
    out = {};
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = words * 4u;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | (output ? D3D11_BIND_UNORDERED_ACCESS : 0u);
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = 4u;
    return SUCCEEDED(d->CreateBuffer(&desc, nullptr, &out.value)) && SUCCEEDED(d->CreateShaderResourceView(out.value.Get(), nullptr, &out.srv)) &&
           (!output || SUCCEEDED(d->CreateUnorderedAccessView(out.value.Get(), nullptr, &out.uav)));
  }

  struct colour_archive_t {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11Buffer> buffer;
    UINT row_bytes {}, height {};
  };

  struct colour_mode_t {
    std::map<std::string, colour_archive_t> pending;
    std::map<std::string, std::vector<char>> planes;
    bool dispatched = false;
  };

  struct colour_unit_t {
    colour_archive_t palette_pending, args_pending;
    std::vector<UINT> palette, args;
  };

  class HostSbsBodyWitnessTemporalGpuTest: public ::testing::Test {
  protected:
    void SetUp() override {
      inner_root_ = component_root_ = temporal_root_ = SUNSHINE_SHADERS_DIR;
      std::string inner_pin, component_pin, temporal_pin;
      if (const char *p = std::getenv("SUNSHINE_TEST_BODY_WITNESS_TEMPORAL_CASES")) {
        spec_path_ = p;
        auto b = colour_bytes(p);
        input_spec_ = nlohmann::json::parse(b.begin(), b.end());
        const auto parent = spec_path_.parent_path();
        driver_ = input_spec_.at("driver").get<std::string>();
        inner_root_ = parent / input_spec_.at("inner_shader_root").get<std::string>();
        component_root_ = parent / input_spec_.at("component_shader_root").get<std::string>();
        temporal_root_ = parent / input_spec_.at("temporal_shader_root").get<std::string>();
        inner_pin = input_spec_.at("inner_source_closure_sha256").get<std::string>();
        component_pin = input_spec_.at("component_source_closure_sha256").get<std::string>();
        temporal_pin = input_spec_.at("temporal_source_closure_sha256").get<std::string>();
        if (input_spec_.contains("capacities")) {
          const auto &cap = input_spec_.at("capacities");
          capacities_ = {cap.at("components").get<UINT>(), cap.at("members").get<UINT>(), cap.at("bit_words").get<UINT>(), cap.at("hash_words").get<UINT>()};
        }
      } else {
        const char *ip = std::getenv("SUNSHINE_TEST_BODY_WITNESS_COLOUR_PIN");
        const char *cp = std::getenv("SUNSHINE_TEST_BODY_WITNESS_COMPONENT_PIN");
        const char *tp = std::getenv("SUNSHINE_TEST_BODY_WITNESS_TEMPORAL_PIN");
        if (!ip || !cp || !tp) {
          GTEST_SKIP() << "Explicit three shader closures required";
        }
        inner_pin = ip;
        component_pin = cp;
        temporal_pin = tp;
        if (const char *p = std::getenv("SUNSHINE_TEST_BODY_WITNESS_COLOUR_ROOT")) {
          inner_root_ = p;
        }
        if (const char *p = std::getenv("SUNSHINE_TEST_BODY_WITNESS_COMPONENT_ROOT")) {
          component_root_ = p;
        }
        if (const char *p = std::getenv("SUNSHINE_TEST_BODY_WITNESS_TEMPORAL_ROOT")) {
          temporal_root_ = p;
        }
        if (const char *p = std::getenv("SUNSHINE_TEST_BODY_WITNESS_DRIVER")) {
          driver_ = p;
        }
      }
      ASSERT_EQ(temporal_pin.size(), 64u);
      ASSERT_EQ(inner_pin.size(), 64u);
      ASSERT_EQ(component_pin.size(), 64u);
      ASSERT_TRUE(driver_ == "warp" || driver_ == "hardware");
      D3D_FEATURE_LEVEL actual {};
      const D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, driver_ == "hardware" ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP, nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &device_, &actual, &context_)));
      ASSERT_EQ(actual, D3D_FEATURE_LEVEL_11_0);
      namespace cache = models::host_sbs_shader_cache;
      const std::array inner_specs {
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "palette_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "emit_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "clear_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "accumulate_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "validate_main", "cs_5_0"}
      };
      const std::array component_specs {
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "init_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "union_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "reduce_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "resolve_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "dilate_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "dilate_box_main", "cs_5_0"}
      };
      auto inner = cache::snapshot_sources(inner_root_, inner_specs), component = cache::snapshot_sources(component_root_, component_specs);
      inner_closure_ = cache::source_closure_sha256(inner);
      component_closure_ = cache::source_closure_sha256(component);
      ASSERT_EQ(inner_closure_, inner_pin);
      ASSERT_EQ(component_closure_, component_pin);
      for (std::size_t i = 0; i < inner_specs.size(); ++i) {
        auto code = cache::get(inner, inner_specs[i]);
        ASSERT_TRUE(code && !code->empty());
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &inner_shaders_[i])));
      }
      for (std::size_t i = 0; i < component_specs.size(); ++i) {
        auto code = cache::get(component, component_specs[i]);
        ASSERT_TRUE(code && !code->empty());
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &component_shaders_[i])));
      }

      std::array<cache::shader_spec, 19u> temporal_specs;
      for (UINT i = 0u; i < temporal_specs.size(); ++i) {
        temporal_specs[i] = {"host_sbs_body_witness_temporal_cs.hlsl", temporal::entry_names[i], "cs_5_0"};
      }
      const auto temporal_source = cache::snapshot_sources(temporal_root_, temporal_specs);
      temporal_closure_ = cache::source_closure_sha256(temporal_source);
      ASSERT_EQ(temporal_closure_, temporal_pin);
      for (UINT i = 0u; i < temporal_specs.size(); ++i) {
        auto code = cache::get(temporal_source, temporal_specs[i]);
        ASSERT_TRUE(code && !code->empty());
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &temporal_shaders_[i])));
      }
      D3D11_BUFFER_DESC d {};
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      d.ByteWidth = 96u;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &constants_buffer_)));
      d.ByteWidth = 16u;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &request_buffer_)));
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &component_constants_)));
      d.ByteWidth = 96u;
      d.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &arguments_)));
      D3D11_UNORDERED_ACCESS_VIEW_DESC u {};
      u.Format = DXGI_FORMAT_R32_TYPELESS;
      u.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      u.Buffer.NumElements = 24u;
      u.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      ASSERT_TRUE(SUCCEEDED(device_->CreateUnorderedAccessView(arguments_.Get(), &u, &arguments_uav_)));
      ASSERT_TRUE(resize(160u, 90u));
    }

    void TearDown() override {
      unbind();
    }

    void unbind() {
      if (!context_) {
        return;
      }
      const std::array<ID3D11ShaderResourceView *, 9u> r {};
      const std::array<ID3D11UnorderedAccessView *, 8u> w {};
      context_->CSSetShaderResources(0u, UINT(r.size()), r.data());
      context_->CSSetUnorderedAccessViews(0u, 8u, w.data(), nullptr);
    }

    bool resize(UINT width, UINT height, UINT top = UINT(-1), UINT crop_height = 0u) {
      unbind();
      width_ = width;
      height_ = height;
      crop_height_ = crop_height ? crop_height : std::min(height, (width + 5u) / 6u);
      crop_top_ = top == UINT(-1) ? height - crop_height_ : top;
      if (width == 0u || height == 0u || width > 5120u || height > 5120u || crop_height_ == 0u || crop_top_ > height || crop_height_ != height - crop_top_) {
        return false;
      }
      const auto n = width * crop_height_;
      requested_ = {};
      requested_[0u].fill(true);
      const bool ok = colour_texture(device_.Get(), width, crop_height_, DXGI_FORMAT_R16G16B16A16_FLOAT, false, source_texture_) &&
                      colour_texture(device_.Get(), width, crop_height_, DXGI_FORMAT_R32_UINT, false, support_texture_) &&
                      colour_buffer(device_.Get(), 32u, true, validation_buffer_) && colour_buffer(device_.Get(), 153600u, false, p_buffer_) && colour_buffer(device_.Get(), 256u, false, ocr_buffer_) &&
                      colour_buffer(device_.Get(), 128u, true, palette_buffer_) && colour_buffer(device_.Get(), n + 1u, true, parent_buffer_) && colour_buffer(device_.Get(), n + 1u, true, flags_buffer_);
      if (!ok) {
        return false;
      }
      for (auto *t : {&eligible_texture_, &seed_texture_, &domain_texture_, &cover_texture_, &body_texture_, &final_texture_, &accum_body_, &accum_final_}) {
        if (!colour_texture(device_.Get(), width, crop_height_, DXGI_FORMAT_R32_UINT, true, *t)) {
          return false;
        }
      }

      if (!backend_.ensure(device_.Get(), width_, crop_height_, capacities_, error_)) {
        return false;
      }
      const std::array<UINT, 3u> root {1u, 1u, 1u};
      D3D11_BUFFER_DESC rd {};
      rd.ByteWidth = 12u;
      rd.Usage = D3D11_USAGE_DEFAULT;
      rd.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_SUBRESOURCE_DATA initial {};
      initial.pSysMem = root.data();
      if (FAILED(device_->CreateBuffer(&rd, &initial, &root_arguments_)) || FAILED(device_->CreateBuffer(&rd, &initial, &poison_arguments_))) {
        return false;
      }
      author(1u, false);
      frame_for_source(1u);
      return true;
    }

    UINT row_edge(UINT detector_y) const {
      const auto &c = constants_.words;
      const auto numerator = std::uint64_t(crop_top_ * 160u + detector_y * crop_height_) * (c[11u] - c[9u]);
      return c[9u] + UINT((numerator + std::uint64_t(height_) * 160u - 1u) / (std::uint64_t(height_) * 160u));
    }

    colour_rect_t field_rect(colour_rect_t native) const {
      const auto &c = constants_.words;
      const UINT cw = c[10u] - c[8u], ch = c[11u] - c[9u];
      return {c[8u] + UINT(std::uint64_t(native[0u]) * cw / width_), c[9u] + UINT(std::uint64_t(native[1u] + crop_top_) * ch / height_), c[8u] + UINT((std::uint64_t(native[2u]) * cw + width_ - 1u) / width_), c[9u] + UINT((std::uint64_t(native[3u] + crop_top_) * ch + height_ - 1u) / height_)};
    }

    void author(UINT frame, bool caption, float background = .15f) {
      const UINT fw = colour_cv2::model_calibrated_shapes[0u].width, fh = colour_cv2::model_calibrated_shapes[0u].height;
      constants_.words = {width_, height_, crop_top_, crop_height_, fw, fh, 0u, 0u, 0u, 0u, fw, fh, frame, 0u, 7u, 0u, 1000000u + frame * 33333u, 0u, 0u, 0u, 1u, SBC_SCHEMA, 0u, 0u};
      constants_.words[6u] = row_edge(colour_cv2::subtitle_ocr_safe_row_top);
      constants_.words[7u] = row_edge(colour_cv2::subtitle_ocr_safe_row_bottom);
      source_.assign(std::size_t(width_) * crop_height_, {colour_half(background), colour_half(background), colour_half(background), colour_half(1.f)});
      support_.assign(source_.size(), 0u);
      probability_.assign(153600u, caption ? .9f : 0.f);
      ocr_.assign(256u, 0u);
      request_reserved_ = 0u;
      ocr_[0u] = 4u;
      ocr_[1u] = 0x3852434fu;
      ocr_[2u] = 1u;
      ocr_[3u] = ocr_[4u] = caption ? 1u : 0u;
      std::copy(constants_.words.begin() + 12u, constants_.words.begin() + 16u, ocr_.begin() + 5u);
      ocr_[9u] = width_;
      ocr_[10u] = height_;
      std::copy(constants_.words.begin() + 4u, constants_.words.begin() + 8u, ocr_.begin() + 11u);
      if (caption) {
        auto core = field_rect({25u, 5u, 135u, 24u}), cover = field_rect({20u, 3u, 140u, 26u});
        core[1u] = std::max(core[1u], constants_.words[6u]);
        core[3u] = std::min(core[3u], constants_.words[7u]);
        cover[1u] = std::max(cover[1u], constants_.words[6u]);
        cover[3u] = std::min(cover[3u], constants_.words[7u]);
        for (const auto &[offset, r] : {std::pair {16u, core}, std::pair {144u, cover}}) {
          std::copy(r.begin(), r.end(), ocr_.begin() + offset);
          ocr_[offset + 4u] = std::bit_cast<UINT>(.9f);
          ocr_[offset + 6u] = 1u;
        }
      }
    }

    std::size_t pixel(UINT x, UINT y) const {
      return std::size_t(y) * width_ + x;
    }

    void paint(colour_rect_t r, std::array<float, 3u> rgb, bool support) {
      for (UINT y = r[1u]; y < std::min(r[3u], crop_height_); ++y) {
        for (UINT x = r[0u]; x < std::min(r[2u], width_); ++x) {
          source_[pixel(x, y)] = {colour_half(rgb[0u]), colour_half(rgb[1u]), colour_half(rgb[2u]), colour_half(1.f)};
          support_[pixel(x, y)] = support ? 1u : 0u;
        }
      }
    }

    void bar(UINT x, std::array<float, 3u> rgb = {.95f, .95f, .95f}, bool support = true, std::array<float, 3u> border = {0.f, 0.f, 0.f}) {
      paint({x - 2u, 6u, x + 6u, 22u}, border, false);
      paint({x, 8u, x + 4u, 20u}, rgb, support);
    }

    void native_probability(colour_rect_t r, float value) {
      for (UINT y = 0u; y < 160u; ++y) {
        for (UINT x = 0u; x < 960u; ++x) {
          const double nx = (x + .5) * width_ / 960u, ny = (y + .5) * crop_height_ / 160u;
          if (nx >= r[0u] && nx < r[2u] && ny >= r[1u] && ny < r[3u]) {
            probability_[std::size_t(y) * 960u + x] = value;
          }
        }
      }
    }

    colour_archive_t archive(ID3D11Texture2D *source) {
      D3D11_TEXTURE2D_DESC d {};
      source->GetDesc(&d);
      colour_archive_t a;
      a.row_bytes = d.Width * (d.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8u : 4u);
      a.height = d.Height;
      archive_bytes_ += std::uint64_t(a.row_bytes) * a.height;
      if (archive_bytes_ > 2ull * 1024u * 1024u * 1024u) {
        throw std::runtime_error("Functional diagnostic archive allocation cap exceeded");
      }
      d.Usage = D3D11_USAGE_STAGING;
      d.BindFlags = d.MiscFlags = 0u;
      d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      if (FAILED(device_->CreateTexture2D(&d, nullptr, &a.texture))) {
        throw std::runtime_error("Cannot allocate inner-colour texture archive");
      }
      context_->CopyResource(a.texture.Get(), source);
      return a;
    }

    colour_archive_t archive(ID3D11Buffer *source) {
      D3D11_BUFFER_DESC d {};
      source->GetDesc(&d);
      colour_archive_t a;
      a.row_bytes = d.ByteWidth;
      a.height = 1u;
      archive_bytes_ += d.ByteWidth;
      if (archive_bytes_ > 2ull * 1024u * 1024u * 1024u) {
        throw std::runtime_error("Functional archive allocation cap exceeded");
      }
      d.Usage = D3D11_USAGE_STAGING;
      d.BindFlags = d.MiscFlags = d.StructureByteStride = 0u;
      d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      if (FAILED(device_->CreateBuffer(&d, nullptr, &a.buffer))) {
        throw std::runtime_error("Cannot allocate inner-colour buffer archive");
      }
      context_->CopyResource(a.buffer.Get(), source);
      return a;
    }

    std::vector<char> read_archive(const colour_archive_t &a) {
      ID3D11Resource *r = a.texture ? static_cast<ID3D11Resource *>(a.texture.Get()) : static_cast<ID3D11Resource *>(a.buffer.Get());
      D3D11_MAPPED_SUBRESOURCE m {};
      if (!r || FAILED(context_->Map(r, 0u, D3D11_MAP_READ, 0u, &m))) {
        throw std::runtime_error("Cannot map inner-colour diagnostic archive");
      }
      std::vector<char> out(std::size_t(a.row_bytes) * a.height);
      for (UINT y = 0u; y < a.height; ++y) {
        std::memcpy(out.data() + std::size_t(y) * a.row_bytes, static_cast<const char *>(m.pData) + std::size_t(y) * (a.texture ? m.RowPitch : a.row_bytes), a.row_bytes);
      }
      context_->Unmap(r, 0u);
      return out;
    }

    std::array<ID3D11ShaderResourceView *, 8u> inner_reads() {
      return {source_texture_.srv.Get(), p_buffer_.srv.Get(), support_texture_.srv.Get(), ocr_buffer_.srv.Get(), palette_buffer_.srv.Get(), nullptr, nullptr, nullptr};
    }

    void bind_inner(const std::array<ID3D11ShaderResourceView *, 8u> &r, const std::array<ID3D11UnorderedAccessView *, 8u> &w) {
      context_->CSSetShaderResources(0u, UINT(r.size()), r.data());
      context_->CSSetUnorderedAccessViews(0u, 8u, w.data(), nullptr);
      ID3D11ShaderResourceView *validation = validation_buffer_.srv.Get();
      context_->CSSetShaderResources(8u, 1u, &validation);
      ID3D11Buffer *cb[] {constants_buffer_.Get(), request_buffer_.Get()};
      context_->CSSetConstantBuffers(0u, 2u, cb);
    }

    bool run() {
      unbind();
      units_ = {};
      modes_.clear();
      input_pending_.clear();
      uploaded_.clear();
      archive_bytes_ = 0u;
      context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, &constants_, 0u, 0u);
      context_->UpdateSubresource(source_texture_.value.Get(), 0u, nullptr, source_.data(), width_ * 8u, 0u);
      context_->UpdateSubresource(support_texture_.value.Get(), 0u, nullptr, support_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(p_buffer_.value.Get(), 0u, nullptr, probability_.data(), 0u, 0u);
      context_->UpdateSubresource(ocr_buffer_.value.Get(), 0u, nullptr, ocr_.data(), 0u, 0u);
      const std::array<UINT, 4u> math {width_, crop_height_, width_ * crop_height_, 0u};
      context_->UpdateSubresource(component_constants_.Get(), 0u, nullptr, math.data(), 0u, 0u);
      input_pending_["uploaded_source.rgba16f"] = archive(source_texture_.value.Get());
      input_pending_["uploaded_probability.f32"] = archive(p_buffer_.value.Get());
      input_pending_["uploaded_current_support.u32"] = archive(support_texture_.value.Get());
      input_pending_["uploaded_ocr_record.u32"] = archive(ocr_buffer_.value.Get());
      input_pending_["uploaded_constants.u32"] = archive(constants_buffer_.Get());

      constexpr std::array<UINT, 4u> zero {};
      context_->ClearUnorderedAccessViewUint(validation_buffer_.uav.Get(), zero.data());
      colour_request_t validation_request {{width_, crop_height_, 0u, 0u}};
      context_->UpdateSubresource(request_buffer_.Get(), 0u, nullptr, &validation_request, 0u, 0u);
      ID3D11Buffer *vconstants[] {constants_buffer_.Get(), request_buffer_.Get()};
      context_->CSSetConstantBuffers(0u, 2u, vconstants);
      const auto vreads = inner_reads();
      context_->CSSetShaderResources(0u, 4u, vreads.data());
      ID3D11UnorderedAccessView *validation_output = validation_buffer_.uav.Get();
      context_->CSSetUnorderedAccessViews(0u, 1u, &validation_output, nullptr);
      context_->CSSetShader(inner_shaders_[4u].Get(), nullptr, 0u);
      context_->Dispatch((width_ + 15u) / 16u, (crop_height_ + 15u) / 16u, 1u);
      unbind();
      std::array<ID3D11ComputeShader *, 19u> tshaders {};
      for (UINT i = 0u; i < tshaders.size(); ++i) {
        tshaders[i] = temporal_shaders_[i].Get();
      }
      const std::array<UINT, 3u> root {root_gate_, 1u, 1u};
      context_->UpdateSubresource(root_arguments_.Get(), 0u, nullptr, root.data(), 0u, 0u);
      if (!backend_.begin(context_.Get(), tshaders, frame_, constants_buffer_.Get(), source_texture_.srv.Get(), validation_buffer_.srv.Get(), root_arguments_.Get(), 0u, error_)) {
        return false;
      }
      colour_request_t request {{0u, 0u, request_reserved_, 0u}};
      context_->UpdateSubresource(request_buffer_.Get(), 0u, nullptr, &request, 0u, 0u);
      std::array<ID3D11ShaderResourceView *, 8u> reads {};
      std::array<ID3D11UnorderedAccessView *, 8u> writes {};
      writes[5u] = accum_body_.uav.Get();
      writes[6u] = accum_final_.uav.Get();
      bind_inner(reads, writes);
      context_->CSSetShader(inner_shaders_[2u].Get(), nullptr, 0u);
      context_->Dispatch((width_ + 15u) / 16u, (crop_height_ + 15u) / 16u, 1u);
      unbind();
      // All8x8 command iterations are fixed. No palette/argument readback is mapped
      // until every command below completes. GPU zero arguments alone skip work.
      for (UINT slot = 0u; slot < 8u; ++slot) {
        request.words = {slot, 0u, request_reserved_, 0u};
        context_->UpdateSubresource(request_buffer_.Get(), 0u, nullptr, &request, 0u, 0u);
        reads = inner_reads();
        reads[4u] = nullptr;
        writes = {};
        writes[0u] = palette_buffer_.uav.Get();
        writes[7u] = arguments_uav_.Get();
        bind_inner(reads, writes);
        context_->CSSetShader(inner_shaders_[0u].Get(), nullptr, 0u);
        context_->Dispatch(1u, 1u, 1u);
        unbind();
        if (argument_fault_ != 0u) {
          // Authored transport-fault test only: no CPU palette/readback controls this override.
          // Complete source replay always leaves it0 and uses literal producer arguments.
          std::array<UINT, 24u> malformed {};
          if (argument_fault_ == 2u) {
            malformed[0] = malformed[1] = malformed[2] = 1u;
          }
          context_->UpdateSubresource(arguments_.Get(), 0u, nullptr, malformed.data(), 0u, 0u);
        }
        if (!backend_.unit_palette(context_.Get(), slot, palette_buffer_.srv.Get(), error_)) {
          return false;
        }
        units_[slot].palette_pending = archive(palette_buffer_.value.Get());
        units_[slot].args_pending = archive(arguments_.Get());
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const UINT offset = mode * 12u;
          request.words = {slot, mode, request_reserved_, 0u};
          context_->UpdateSubresource(request_buffer_.Get(), 0u, nullptr, &request, 0u, 0u);
          reads = inner_reads();
          writes = {};
          writes[1u] = eligible_texture_.uav.Get();
          writes[2u] = seed_texture_.uav.Get();
          writes[3u] = domain_texture_.uav.Get();
          writes[4u] = cover_texture_.uav.Get();
          bind_inner(reads, writes);
          context_->CSSetShader(inner_shaders_[1u].Get(), nullptr, 0u);
          context_->DispatchIndirect(arguments_.Get(), offset);
          unbind();
          reads = {eligible_texture_.srv.Get(), seed_texture_.srv.Get(), domain_texture_.srv.Get(), cover_texture_.srv.Get(), nullptr, nullptr, nullptr, nullptr};
          writes = {parent_buffer_.uav.Get(), flags_buffer_.uav.Get(), body_texture_.uav.Get(), final_texture_.uav.Get(), nullptr, nullptr, nullptr, nullptr};
          context_->CSSetShaderResources(0u, 8u, reads.data());
          context_->CSSetUnorderedAccessViews(0u, 8u, writes.data(), nullptr);
          context_->CSSetConstantBuffers(0u, 1u, component_constants_.GetAddressOf());
          for (UINT phase = 0u; phase < 4u; ++phase) {
            context_->CSSetShader(component_shaders_[phase].Get(), nullptr, 0u);
            context_->DispatchIndirect(arguments_.Get(), offset);
          }

          unbind();
          const std::array<UINT, 3u> poisoned {(width_ + 15u) / 16u, (crop_height_ + 15u) / 16u, 1u};
          if (poison_mode_arguments_) {
            context_->UpdateSubresource(poison_arguments_.Get(), 0u, nullptr, poisoned.data(), 0u, 0u);
          }
          if (!backend_.mode_after_ccresolve(context_.Get(), slot, mode, seed_texture_.srv.Get(), body_texture_.srv.Get(), parent_buffer_.srv.Get(), flags_buffer_.srv.Get(), palette_buffer_.srv.Get(), cover_texture_.srv.Get(), poison_mode_arguments_ ? poison_arguments_.Get() : arguments_.Get(), poison_mode_arguments_ ? 0u : offset, error_)) {
            return false;
          }
          // Independent frozen V5 control: SAME emitted mode/4CC, no second extraction/CC.
          context_->CSSetShaderResources(0u, 8u, reads.data());
          context_->CSSetUnorderedAccessViews(0u, 8u, writes.data(), nullptr);
          context_->CSSetConstantBuffers(0u, 1u, component_constants_.GetAddressOf());
          context_->CSSetShader(component_shaders_[5u].Get(), nullptr, 0u);
          context_->DispatchIndirect(arguments_.Get(), offset);
          unbind();
          reads = inner_reads();
          reads[5u] = body_texture_.srv.Get();
          reads[6u] = final_texture_.srv.Get();
          reads[7u] = parent_buffer_.srv.Get();
          writes = {};
          writes[5u] = accum_body_.uav.Get();
          writes[6u] = accum_final_.uav.Get();
          bind_inner(reads, writes);
          context_->CSSetShader(inner_shaders_[3u].Get(), nullptr, 0u);
          context_->DispatchIndirect(arguments_.Get(), offset);
          unbind();
          if (requested_[slot][mode]) {
            auto &m = modes_[{slot, mode}];
            m.pending["eligible.u32"] = archive(eligible_texture_.value.Get());
            m.pending["seeds.u32"] = archive(seed_texture_.value.Get());
            m.pending["domain_tag.u32"] = archive(domain_texture_.value.Get());
            m.pending["cover_clip.u32"] = archive(cover_texture_.value.Get());
            m.pending["parent_roots.u32"] = archive(parent_buffer_.value.Get());
            m.pending["root_flags.u32"] = archive(flags_buffer_.value.Get());
            m.pending["body.u32"] = archive(body_texture_.value.Get());
            m.pending["final.u32"] = archive(final_texture_.value.Get());
            m.pending["uploaded_request.u32"] = archive(request_buffer_.Get());
          }
        }
      }

      if (!backend_.end(context_.Get(), error_)) {
        return false;
      }
      const auto tv = backend_.mutable_views();
      temporal_pending_.clear();
      const auto archive_srv = [&](const char *name, ID3D11ShaderResourceView *view) {
        ComPtr<ID3D11Resource> r;
        view->GetResource(&r);
        ComPtr<ID3D11Buffer> b;
        ComPtr<ID3D11Texture2D> t;
        if (SUCCEEDED(r.As(&b))) {
          temporal_pending_[name] = archive(b.Get());
        } else if (SUCCEEDED(r.As(&t))) {
          temporal_pending_[name] = archive(t.Get());
        } else {
          throw std::runtime_error("Unknown private output resource");
        }
      };
      archive_srv("body.u32", tv.body.Get());
      archive_srv("final.u32", tv.matte.Get());
      archive_srv("receipt.u32", tv.receipt.Get());
      archive_srv("state.u32", tv.state.Get());
      archive_srv("records.u32", tv.records.Get());
      archive_srv("members.u32x2", tv.members.Get());
      archive_srv("bits.u32", tv.bits.Get());
      archive_srv("root-rgb.u32", tv.root_rgb.Get());
      // Export every array layer, never a CPU-selected history bank or only subresource0.
      ComPtr<ID3D11Resource> image_resource;
      ComPtr<ID3D11Texture2D> image_array;
      tv.images->GetResource(&image_resource);
      if (FAILED(image_resource.As(&image_array))) {
        return false;
      }
      D3D11_TEXTURE2D_DESC image_desc {};
      image_array->GetDesc(&image_desc);
      if (image_desc.ArraySize != 9u || image_desc.MipLevels != 1u) {
        return false;
      }
      for (UINT layer = 0u; layer < 9u; ++layer) {
        auto d = image_desc;
        d.ArraySize = 1u;
        d.Usage = D3D11_USAGE_STAGING;
        d.BindFlags = d.MiscFlags = 0u;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        colour_archive_t capture;
        capture.row_bytes = width_ * 4u;
        capture.height = crop_height_;
        archive_bytes_ += std::uint64_t(capture.row_bytes) * capture.height;
        if (archive_bytes_ > 2ull * 1024u * 1024u * 1024u || FAILED(device_->CreateTexture2D(&d, nullptr, &capture.texture))) {
          throw std::runtime_error("Cannot allocate temporal image-layer archive");
        }
        context_->CopySubresourceRegion(capture.texture.Get(), 0u, 0u, 0u, 0u, image_array.Get(), layer, nullptr);
        temporal_pending_["images-layer-" + std::to_string(layer) + ".u32"] = std::move(capture);
      }
      temporal_pending_["uploaded_frame_constants.u32"] = archive(tv.frame_constants.Get());
      temporal_pending_["uploaded_root_arguments.u32"] = archive(root_arguments_.Get());
      const auto body_pending = archive(accum_body_.value.Get()), final_pending = archive(accum_final_.value.Get());
      // This is the first CPU mapping after the entire fixed-command GPU chain.
      for (auto &[name, a] : input_pending_) {
        uploaded_[name] = read_archive(a);
      }
      body_ = colour_words(read_archive(body_pending));
      final_ = colour_words(read_archive(final_pending));
      capture_complete_ = true;
      for (UINT slot = 0u; slot < 8u; ++slot) {
        units_[slot].palette = colour_words(read_archive(units_[slot].palette_pending));
        units_[slot].args = colour_words(read_archive(units_[slot].args_pending));
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const auto &a = units_[slot].args;
          const bool active = a[3u * mode] != 0u || a[3u * mode + 1u] != 0u || a[3u * mode + 2u] != 0u;
          if (active && !requested_[slot][mode]) {
            capture_complete_ = false;
          }
          if (!requested_[slot][mode]) {
            continue;
          }
          auto &m = modes_.at({slot, mode});
          m.dispatched = active;
          for (auto &[name, pending] : m.pending) {
            m.planes[name] = read_archive(pending);
          }
        }
      }

      temporal_uploaded_.clear();
      for (const auto &[name, pending] : temporal_pending_) {
        temporal_uploaded_[name] = read_archive(pending);
      }
      temporal_body_ = colour_words(temporal_uploaded_.at("body.u32"));
      temporal_final_ = colour_words(temporal_uploaded_.at("final.u32"));
      temporal_receipt_ = colour_words(temporal_uploaded_.at("receipt.u32"));
      return true;
    }

    bool run_composed(bool recorded, ID3D11Buffer *gpu_frame256 = nullptr) {
      unbind();
      units_ = {};
      modes_.clear();
      input_pending_.clear();
      uploaded_.clear();
      archive_bytes_ = 0u;
      context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, &constants_, 0u, 0u);
      context_->UpdateSubresource(source_texture_.value.Get(), 0u, nullptr, source_.data(), width_ * 8u, 0u);
      context_->UpdateSubresource(support_texture_.value.Get(), 0u, nullptr, support_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(p_buffer_.value.Get(), 0u, nullptr, probability_.data(), 0u, 0u);
      context_->UpdateSubresource(ocr_buffer_.value.Get(), 0u, nullptr, ocr_.data(), 0u, 0u);
      const std::array<UINT, 4u> math {width_, crop_height_, width_ * crop_height_, 0u};
      context_->UpdateSubresource(component_constants_.Get(), 0u, nullptr, math.data(), 0u, 0u);
      input_pending_["uploaded_source.rgba16f"] = archive(source_texture_.value.Get());
      input_pending_["uploaded_probability.f32"] = archive(p_buffer_.value.Get());
      input_pending_["uploaded_current_support.u32"] = archive(support_texture_.value.Get());
      input_pending_["uploaded_ocr_record.u32"] = archive(ocr_buffer_.value.Get());
      input_pending_["uploaded_constants.u32"] = archive(constants_buffer_.Get());

      if (!backend_.ensure(device_.Get(), width_, crop_height_, capacities_, error_) || !coordinator_.ensure(device_.Get(), width_, crop_height_, error_)) {
        return false;
      }
      const std::array<UINT, 3u> root {root_gate_, 1u, 1u};
      context_->UpdateSubresource(root_arguments_.Get(), 0u, nullptr, root.data(), 0u, 0u);
      std::array<ID3D11ComputeShader *, 5u> colour {};
      std::array<ID3D11ComputeShader *, 6u> graph {};
      for (UINT i = 0u; i < colour.size(); ++i) {
        colour[i] = inner_shaders_[i].Get();
      }
      for (UINT i = 0u; i < graph.size(); ++i) {
        graph[i] = component_shaders_[i].Get();
      }
      models::host_sbs_body_witness_temporal_recorded::program_t program;
      program.resources = &backend_;
      program.frame = frame_;
      if (gpu_frame256) {
        // Shape/caps are the ONLY CPU precondition. No CPU source/cut/model word is uploaded.
        program.frame = {};
        std::copy_n(frame_.words.begin(), 4u, program.frame.words.begin());
        std::copy_n(frame_.words.begin() + 12u, 4u, program.frame.words.begin() + 12u);
        program.gpu_frame256 = gpu_frame256;
      }
      program.include_baseline_control = true;
      for (UINT i = 0u; i < program.shaders.size(); ++i) {
        program.shaders[i] = temporal_shaders_[i].Get();
      }
      coordinator_.set_recorded_execution(recorded);
      if (!coordinator_.dispatch(context_.Get(), colour, graph, constants_buffer_.Get(), source_texture_.srv.Get(), p_buffer_.srv.Get(), support_texture_.srv.Get(), ocr_buffer_.srv.Get(), root_arguments_.Get(), 0u, error_, &program)) {
        return false;
      }
      const auto tv = backend_.mutable_views();
      temporal_pending_.clear();
      const auto archive_srv = [&](const char *name, ID3D11ShaderResourceView *view) {
        ComPtr<ID3D11Resource> r;
        view->GetResource(&r);
        ComPtr<ID3D11Buffer> b;
        ComPtr<ID3D11Texture2D> t;
        if (SUCCEEDED(r.As(&b))) {
          temporal_pending_[name] = archive(b.Get());
        } else if (SUCCEEDED(r.As(&t))) {
          temporal_pending_[name] = archive(t.Get());
        } else {
          throw std::runtime_error("Unknown private output resource");
        }
      };
      archive_srv("body.u32", tv.body.Get());
      archive_srv("final.u32", tv.matte.Get());
      archive_srv("receipt.u32", tv.receipt.Get());
      archive_srv("state.u32", tv.state.Get());
      archive_srv("records.u32", tv.records.Get());
      archive_srv("members.u32x2", tv.members.Get());
      archive_srv("bits.u32", tv.bits.Get());
      archive_srv("root-rgb.u32", tv.root_rgb.Get());
      // Export every array layer, never a CPU-selected history bank or only subresource0.
      ComPtr<ID3D11Resource> image_resource;
      ComPtr<ID3D11Texture2D> image_array;
      tv.images->GetResource(&image_resource);
      if (FAILED(image_resource.As(&image_array))) {
        return false;
      }
      D3D11_TEXTURE2D_DESC image_desc {};
      image_array->GetDesc(&image_desc);
      if (image_desc.ArraySize != 9u || image_desc.MipLevels != 1u) {
        return false;
      }
      for (UINT layer = 0u; layer < 9u; ++layer) {
        auto d = image_desc;
        d.ArraySize = 1u;
        d.Usage = D3D11_USAGE_STAGING;
        d.BindFlags = d.MiscFlags = 0u;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        colour_archive_t capture;
        capture.row_bytes = width_ * 4u;
        capture.height = crop_height_;
        archive_bytes_ += std::uint64_t(capture.row_bytes) * capture.height;
        if (archive_bytes_ > 2ull * 1024u * 1024u * 1024u || FAILED(device_->CreateTexture2D(&d, nullptr, &capture.texture))) {
          throw std::runtime_error("Cannot allocate temporal image-layer archive");
        }
        context_->CopySubresourceRegion(capture.texture.Get(), 0u, 0u, 0u, 0u, image_array.Get(), layer, nullptr);
        temporal_pending_["images-layer-" + std::to_string(layer) + ".u32"] = std::move(capture);
      }
      temporal_pending_["uploaded_frame_constants.u32"] = archive(tv.frame_constants.Get());
      temporal_pending_["uploaded_root_arguments.u32"] = archive(root_arguments_.Get());

      const auto baseline = coordinator_.baseline_views();
      const auto archive_native = [&](ID3D11ShaderResourceView *view) {
        ComPtr<ID3D11Resource> resource;
        ComPtr<ID3D11Texture2D> texture;
        view->GetResource(&resource);
        if (FAILED(resource.As(&texture))) {
          throw std::runtime_error("Expected native baseline texture");
        }
        return archive(texture.Get());
      };
      const auto bp = archive_native(baseline.body.Get()), mp = archive_native(baseline.matte.Get());
      for (auto &[name, pending] : input_pending_) {
        uploaded_[name] = read_archive(pending);
      }
      body_ = colour_words(read_archive(bp));
      final_ = colour_words(read_archive(mp));
      temporal_uploaded_.clear();
      for (auto &[name, pending] : temporal_pending_) {
        temporal_uploaded_[name] = read_archive(pending);
      }
      temporal_body_ = colour_words(temporal_uploaded_.at("body.u32"));
      temporal_final_ = colour_words(temporal_uploaded_.at("final.u32"));
      temporal_receipt_ = colour_words(temporal_uploaded_.at("receipt.u32"));
      return true;
    }

    void frame_for_source(UINT serial, UINT reset = 0u) {
      const auto &c = constants_.words;
      frame_ = {};
      frame_.words[0] = width_;
      frame_.words[1] = crop_height_;
      frame_.words[2] = width_ * crop_height_;
      frame_.words[4] = serial;
      frame_.words[6] = 2000000u + serial * 33333u;
      frame_.words[8] = reset;
      frame_.words[9] = frame_.words[10] = 1u;
      frame_.words[12] = capacities_.components;
      frame_.words[13] = capacities_.members;
      frame_.words[14] = capacities_.bit_words;
      frame_.words[15] = capacities_.hash_words;
      frame_.words[16] = 0x41555448u;
      frame_.words[18] = 1u;  // AUTHORED mechanics namespace/epoch.
      frame_.words[20] = c[14];
      frame_.words[21] = c[15];
      frame_.words[23] = 1u;
      std::copy_n(c.begin(), 12u, frame_.words.begin() + 24u);
      frame_.words[48] = c[12];
      frame_.words[49] = c[13];
      frame_.words[50] = c[16];
      frame_.words[51] = c[17];
      frame_.words[52] = 1u;  // Authored mechanical sources are known-adjacent material observations.
    }

    void expect_available() {
      ASSERT_EQ(temporal_receipt_.size(), 256u);
      EXPECT_EQ(temporal_receipt_[0], 0x31545742u);
      EXPECT_EQ(temporal_receipt_[1], 1u);
      EXPECT_EQ(temporal_receipt_[2], 1u);
      EXPECT_EQ(temporal_receipt_[3], 0u);
      EXPECT_EQ(temporal_receipt_[17], 1u);
      EXPECT_EQ(temporal_receipt_[18], 1u);
      EXPECT_TRUE(std::all_of(temporal_body_.begin(), temporal_body_.end(), [](UINT v) {
        return v <= 1u;
      }));
      EXPECT_TRUE(std::all_of(temporal_final_.begin(), temporal_final_.end(), [](UINT v) {
        return v <= 1u;
      }));
      ASSERT_EQ(temporal_body_.size(), body_.size());
      ASSERT_EQ(temporal_final_.size(), final_.size());
      bool subset = true;
      for (std::size_t i = 0u; i < body_.size(); ++i) {
        subset &= temporal_body_[i] <= body_[i] && temporal_final_[i] <= final_[i];
      }
      EXPECT_TRUE(subset);
    }

    void expect_uploaded_exact() {
      EXPECT_EQ(uploaded_.at("uploaded_source.rgba16f"), colour_raw(source_));
      EXPECT_EQ(uploaded_.at("uploaded_probability.f32"), colour_raw(probability_));
      EXPECT_EQ(uploaded_.at("uploaded_current_support.u32"), colour_raw(support_));
      EXPECT_EQ(uploaded_.at("uploaded_ocr_record.u32"), colour_raw(ocr_));
      EXPECT_EQ(uploaded_.at("uploaded_constants.u32"), colour_raw(constants_.words));
    }

    bool uploaded_byte_exact() const {
      return uploaded_.at("uploaded_source.rgba16f") == colour_raw(source_) && uploaded_.at("uploaded_probability.f32") == colour_raw(probability_) &&
             uploaded_.at("uploaded_current_support.u32") == colour_raw(support_) && uploaded_.at("uploaded_ocr_record.u32") == colour_raw(ocr_) && uploaded_.at("uploaded_constants.u32") == colour_raw(constants_.words);
    }

    bool arguments_byte_exact() const {
      for (const auto &u : units_) {
        if (u.palette.size() != 128u || u.args.size() != 24u) {
          return false;
        }
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const bool active = u.palette[2u] == 1u && u.palette[22u] == 2u && mode < u.palette[4u] && u.palette[64u + 8u * mode + 3u] != 0u;
          if (u.args[3u * mode] != (active ? (width_ + 15u) / 16u : 0u) || u.args[3u * mode + 1u] != (active ? (crop_height_ + 15u) / 16u : 0u) || u.args[3u * mode + 2u] != (active ? 1u : 0u)) {
            return false;
          }
        }
      }
      return true;
    }

    bool gpu_or_byte_exact() const {
      if (!capture_complete_) {
        return false;
      }
      std::vector<UINT> b(body_.size(), 0u), f(final_.size(), 0u);
      for (const auto &[key, m] : modes_) {
        if (!m.dispatched) {
          continue;
        }
        const auto mb = colour_words(m.planes.at("body.u32")), mf = colour_words(m.planes.at("final.u32"));
        if (mb.size() != b.size() || mf.size() != f.size()) {
          return false;
        }
        for (std::size_t p = 0u; p < b.size(); ++p) {
          if (mb[p] > 1u || mf[p] > 1u) {
            return false;
          }
          b[p] |= mb[p];
          f[p] |= mf[p];
        }
      }
      return b == body_ && f == final_;
    }

    void expect_args_exact() {
      for (UINT slot = 0u; slot < 8u; ++slot) {
        const auto &u = units_[slot];
        ASSERT_EQ(u.palette.size(), 128u);
        ASSERT_EQ(u.args.size(), 24u);
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const bool active = u.palette[2u] == 1u && u.palette[22u] == 2u && mode < u.palette[4u] && u.palette[64u + 8u * mode + 3u] != 0u;
          EXPECT_EQ(u.args[3u * mode], active ? (width_ + 15u) / 16u : 0u);
          EXPECT_EQ(u.args[3u * mode + 1u], active ? (crop_height_ + 15u) / 16u : 0u);
          EXPECT_EQ(u.args[3u * mode + 2u], active ? 1u : 0u);
        }
      }
    }

    void expect_gpu_or_exact() {
      ASSERT_TRUE(capture_complete_) << "Diagnostics must capture every dispatched mode; they never control its execution";
      std::vector<UINT> reference_body(body_.size(), 0u), reference_final(final_.size(), 0u);
      for (const auto &[key, m] : modes_) {
        if (!m.dispatched) {
          continue;
        }
        const auto body = colour_words(m.planes.at("body.u32")), final = colour_words(m.planes.at("final.u32"));
        ASSERT_EQ(body.size(), reference_body.size());
        ASSERT_EQ(final.size(), reference_final.size());
        for (std::size_t p = 0u; p < body.size(); ++p) {
          ASSERT_LE(body[p], 1u);
          ASSERT_LE(final[p], 1u);
          reference_body[p] |= body[p];
          reference_final[p] |= final[p];
        }
      }
      EXPECT_EQ(body_, reference_body);
      EXPECT_EQ(final_, reference_final);
    }

    std::pair<UINT, UINT> expected_peak_tie(UINT target_peak, UINT target_bin) const {
      // Independent authored-grid expectation. These tests use globally strong P
      // and four-pixel-wide outlined bars: every matching colour has a +/-2 edge.
      // This does not run the CPU candidate or supply a GPU palette verdict.
      const auto &c = constants_.words;
      const UINT cw = c[10u] - c[8u], ch = c[11u] - c[9u];
      UINT first = UINT(-1), count = 0u;
      for (UINT grid = 0u; grid < 512u; ++grid) {
        const UINT gx = grid % 64u, gy = grid / 64u;
        const UINT x = UINT(((std::uint64_t(ocr_[16u] - c[8u]) * 128u + (2u * gx + 1u) * (ocr_[18u] - ocr_[16u])) * width_) / (128u * cw));
        const int absolute_y = int(((std::uint64_t(ocr_[17u] - c[9u]) * 16u + (2u * gy + 1u) * (ocr_[19u] - ocr_[17u])) * height_) / (16u * ch));
        const UINT y = UINT(std::clamp(absolute_y - int(crop_top_), 0, int(crop_height_) - 1));
        const auto &v = source_[pixel(std::min(x, width_ - 1u), y)];
        std::array<UINT, 3u> q {}, bin {};
        for (UINT lane = 0u; lane < 3u; ++lane) {
          const float value = colour_float(v[lane]);
          const float scaled = value * 65535.f;
          q[lane] = UINT(std::floor(scaled + .5f));
          bin[lane] = std::min(UINT(value * 4.f), 3u);
        }
        if (*std::max_element(q.begin(), q.end()) == target_peak && bin[0u] * 16u + bin[1u] * 4u + bin[2u] == target_bin) {
          first = std::min(first, grid);
          ++count;
        }
      }
      return {first, count};
    }

    nlohmann::json adapter_identity() {
      ComPtr<IDXGIDevice> dxgi;
      ComPtr<IDXGIAdapter> adapter;
      DXGI_ADAPTER_DESC d {};
      if (FAILED(device_.As(&dxgi)) || FAILED(dxgi->GetAdapter(&adapter)) || FAILED(adapter->GetDesc(&d))) {
        return nullptr;
      }
      std::wstring wide(d.Description);
      return {{"description", std::string(wide.begin(), wide.end())}, {"vendor_id", d.VendorId}, {"device_id", d.DeviceId}, {"luid_low", d.AdapterLuid.LowPart}, {"luid_high", d.AdapterLuid.HighPart}};
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    std::array<ComPtr<ID3D11ComputeShader>, 5u> inner_shaders_;
    std::array<ComPtr<ID3D11ComputeShader>, 6u> component_shaders_;
    ComPtr<ID3D11Buffer> constants_buffer_, request_buffer_, component_constants_, arguments_;
    ComPtr<ID3D11UnorderedAccessView> arguments_uav_;
    colour_texture_t source_texture_, support_texture_, eligible_texture_, seed_texture_, domain_texture_, cover_texture_, body_texture_, final_texture_, accum_body_, accum_final_;
    colour_buffer_t validation_buffer_, p_buffer_, ocr_buffer_, palette_buffer_, parent_buffer_, flags_buffer_;
    colour_constants_t constants_ {};
    UINT width_ {}, height_ {}, crop_top_ {}, crop_height_ {}, request_reserved_ {};
    std::vector<colour_half_t> source_;
    std::vector<float> probability_;
    std::vector<UINT> support_, ocr_, body_, final_;
    std::array<std::array<bool, 8u>, 8u> requested_ {};
    std::array<colour_unit_t, 8u> units_ {};
    std::map<std::pair<UINT, UINT>, colour_mode_t> modes_;
    std::map<std::string, colour_archive_t> input_pending_;
    std::map<std::string, std::vector<char>> uploaded_;
    std::uint64_t archive_bytes_ {};
    bool capture_complete_ {};

    temporal::resources_t backend_;
    models::host_sbs_body_witness_temporal_recorded::resources_t coordinator_;
    temporal::caps_t capacities_;
    temporal::frame_constants_t frame_;
    std::array<ComPtr<ID3D11ComputeShader>, 19u> temporal_shaders_;
    ComPtr<ID3D11Buffer> root_arguments_, poison_arguments_;
    UINT root_gate_ = 1u;
    UINT argument_fault_ = 0u;
    bool poison_mode_arguments_ = false;
    std::vector<UINT> temporal_body_, temporal_final_, temporal_receipt_;
    std::map<std::string, colour_archive_t> temporal_pending_;
    std::map<std::string, std::vector<char>> temporal_uploaded_;
    std::filesystem::path temporal_root_, spec_path_;
    std::string temporal_closure_, error_;
    nlohmann::json input_spec_;
    std::filesystem::path inner_root_, component_root_;
    std::string inner_closure_, component_closure_, driver_ {"warp"};
  };

  TEST_F(HostSbsBodyWitnessTemporalGpuTest, FourComponentSameCurrentBaselineAndStableOwnedRemain) {
    author(1u, true);
    for (UINT x : {35u, 60u, 90u, 120u}) {
      bar(x);
    }
    frame_for_source(1u);
    ASSERT_TRUE(run()) << error_;
    expect_uploaded_exact();
    expect_available();
    EXPECT_EQ(temporal_body_, body_);
    EXPECT_EQ(temporal_final_, final_);
    EXPECT_EQ(temporal_receipt_[13], 0u);
    const auto first = temporal_final_;
    author(3u, true);
    for (UINT x : {35u, 60u, 90u, 120u}) {
      bar(x);
    }
    frame_for_source(2u);
    ASSERT_TRUE(run()) << error_;
    expect_available();
    EXPECT_EQ(temporal_receipt_[8], 1u);
    EXPECT_EQ(temporal_final_, first);
    EXPECT_EQ(temporal_body_, body_);
  }

  TEST_F(HostSbsBodyWitnessTemporalGpuTest, UnownedStableComponentVetoCarryDuplicateAndReset) {
    author(1u, true);
    bar(40u);
    bar(110u, {.95f, .95f, .95f}, false);
    native_probability({95u, 3u, 130u, 26u}, 0.f);
    frame_for_source(1u);
    ASSERT_TRUE(run()) << error_;
    expect_available();
    EXPECT_EQ(body_[pixel(111u, 12u)], 0u);
    author(3u, true);
    bar(70u);
    bar(110u);
    frame_for_source(2u);
    ASSERT_TRUE(run()) << error_;
    expect_available();
    EXPECT_EQ(body_[pixel(111u, 12u)], 1u);
    EXPECT_EQ(temporal_body_[pixel(111u, 12u)], 0u);
    EXPECT_EQ(temporal_body_[pixel(71u, 12u)], 1u);
    EXPECT_EQ(temporal_receipt_[4], 2u);  // SOURCE identity was not compressed from MODEL3.
    EXPECT_EQ(temporal_receipt_[76], 3u);
    ASSERT_GT(temporal_receipt_[13], 0u);
    const auto veto = temporal_final_;
    ASSERT_TRUE(run()) << error_;
    expect_available();
    EXPECT_EQ(temporal_receipt_[9], 1u);
    EXPECT_EQ(temporal_final_, veto);
    author(6u, true);
    bar(70u);
    bar(110u);
    frame_for_source(3u);
    ASSERT_TRUE(run()) << error_;
    expect_available();
    EXPECT_EQ(temporal_final_, veto);
    EXPECT_EQ(temporal_receipt_[4], 3u);
    EXPECT_EQ(temporal_receipt_[76], 6u);
    frame_.words[52] = 0u;  // Same-source permission conflict must revoke a cached negative.
    ASSERT_TRUE(run()) << error_;
    expect_available();
    EXPECT_EQ(temporal_receipt_[9], 0u);
    EXPECT_EQ(temporal_body_, body_);
    EXPECT_EQ(temporal_final_, final_);
    frame_for_source(3u, 1u);
    ASSERT_TRUE(run()) << error_;
    expect_available();
    EXPECT_EQ(temporal_body_, body_);
    EXPECT_EQ(temporal_final_, final_);
  }

  TEST_F(HostSbsBodyWitnessTemporalGpuTest, RootZeroPoisonedModesClearCurrentOutputAndHoldWatermark) {
    author(1u, true);
    bar(50u);
    frame_for_source(1u);
    ASSERT_TRUE(run()) << error_;
    expect_available();
    const auto previous = colour_words(temporal_uploaded_.at("state.u32"));
    author(3u, true);
    bar(100u);
    frame_for_source(2u);
    root_gate_ = 0u;
    poison_mode_arguments_ = true;
    ASSERT_TRUE(run()) << error_;
    EXPECT_TRUE(std::all_of(temporal_body_.begin(), temporal_body_.end(), [](UINT v) {
      return v == 0u;
    }));
    EXPECT_TRUE(std::all_of(temporal_final_.begin(), temporal_final_.end(), [](UINT v) {
      return v == 0u;
    }));
    EXPECT_TRUE(std::all_of(temporal_receipt_.begin(), temporal_receipt_.end(), [](UINT v) {
      return v == 0u;
    }));
    EXPECT_EQ(colour_words(temporal_uploaded_.at("state.u32")), previous);
    root_gate_ = 1u;
    poison_mode_arguments_ = false;
    ASSERT_TRUE(run()) << error_;
    expect_available();
    EXPECT_EQ(temporal_receipt_[8], 1u);
  }

  TEST_F(HostSbsBodyWitnessTemporalGpuTest, GenuineModelEmptyZeroIsStillSemanticUnknown) {
    author(1u, true);
    bar(50u);
    frame_for_source(1u);
    ASSERT_TRUE(run()) << error_;
    expect_available();
    author(3u, false);
    frame_for_source(2u);
    ASSERT_TRUE(run()) << error_;
    expect_available();
    EXPECT_EQ(temporal_body_, body_);
    EXPECT_EQ(temporal_final_, final_);
    EXPECT_TRUE(std::all_of(temporal_final_.begin(), temporal_final_.end(), [](UINT v) {
      return v == 0u;
    }));
    EXPECT_EQ(temporal_receipt_[17], 1u);
  }

  TEST_F(HostSbsBodyWitnessTemporalGpuTest, MissingOrPartialActiveModeCoverageIsUnavailable) {
    for (UINT fault : {1u, 2u}) {
      backend_ = temporal::resources_t {};
      ASSERT_TRUE(backend_.ensure(device_.Get(), width_, crop_height_, capacities_, error_));
      author(1u, true);
      bar(50u);
      frame_for_source(1u);
      argument_fault_ = fault;
      ASSERT_TRUE(run()) << error_;
      ASSERT_EQ(temporal_receipt_.size(), 256u);
      EXPECT_EQ(temporal_receipt_[2], 0u);
      EXPECT_NE(temporal_receipt_[3], 0u);
      EXPECT_TRUE(std::all_of(temporal_body_.begin(), temporal_body_.end(), [](UINT v) {
        return v == 0u;
      }));
      EXPECT_TRUE(std::all_of(temporal_final_.begin(), temporal_final_.end(), [](UINT v) {
        return v == 0u;
      }));
    }
  }

  TEST_F(HostSbsBodyWitnessTemporalGpuTest, ResourceBoundsDeferredContextAndPredicationRejectBeforeCommands) {
    temporal::resources_t invalid;
    auto bad = capacities_;
    bad.components = 65536u;
    EXPECT_FALSE(invalid.ensure(device_.Get(), width_, crop_height_, bad, error_));
    author(1u, true);
    bar(50u);
    frame_for_source(1u);
    ASSERT_TRUE(run()) << error_;
    std::array<ID3D11ComputeShader *, 19u> shaders {};
    for (UINT i = 0u; i < shaders.size(); ++i) {
      shaders[i] = temporal_shaders_[i].Get();
    }
    ComPtr<ID3D11DeviceContext> deferred;
    ASSERT_TRUE(SUCCEEDED(device_->CreateDeferredContext(0u, &deferred)));
    EXPECT_FALSE(backend_.begin(deferred.Get(), shaders, frame_, constants_buffer_.Get(), source_texture_.srv.Get(), validation_buffer_.srv.Get(), root_arguments_.Get(), 0u, error_));
    D3D11_QUERY_DESC pd {};
    pd.Query = D3D11_QUERY_OCCLUSION_PREDICATE;
    ComPtr<ID3D11Predicate> predicate;
    ASSERT_TRUE(SUCCEEDED(device_->CreatePredicate(&pd, &predicate)));
    context_->SetPredication(predicate.Get(), FALSE);
    EXPECT_FALSE(backend_.begin(context_.Get(), shaders, frame_, constants_buffer_.Get(), source_texture_.srv.Get(), validation_buffer_.srv.Get(), root_arguments_.Get(), 0u, error_));
    ComPtr<ID3D11Predicate> retained;
    BOOL value = TRUE;
    context_->GetPredication(&retained, &value);
    EXPECT_EQ(retained.Get(), predicate.Get());
    EXPECT_EQ(value, FALSE);
    context_->SetPredication(nullptr, FALSE);
  }

  TEST_F(HostSbsBodyWitnessTemporalGpuTest, RestoresNineSrvEightUavThreeCbAndShaderAcrossRootZeroBegin) {
    author(1u, true);
    bar(50u);
    frame_for_source(1u);
    ASSERT_TRUE(run()) << error_;
    std::array<ID3D11ComputeShader *, 19u> shaders {};
    for (UINT i = 0u; i < shaders.size(); ++i) {
      shaders[i] = temporal_shaders_[i].Get();
    }
    const std::array<UINT, 3u> zero {0u, 1u, 1u};
    context_->UpdateSubresource(root_arguments_.Get(), 0u, nullptr, zero.data(), 0u, 0u);
    std::array<ID3D11ShaderResourceView *, 9u> reads;
    reads.fill(source_texture_.srv.Get());
    std::array<ID3D11UnorderedAccessView *, 8u> writes {palette_buffer_.uav.Get(), parent_buffer_.uav.Get(), flags_buffer_.uav.Get(), eligible_texture_.uav.Get(), seed_texture_.uav.Get(), domain_texture_.uav.Get(), cover_texture_.uav.Get(), body_texture_.uav.Get()};
    std::array<ID3D11Buffer *, 3u> constants;
    constants.fill(constants_buffer_.Get());
    context_->CSSetShaderResources(0u, 9u, reads.data());
    context_->CSSetUnorderedAccessViews(0u, 8u, writes.data(), nullptr);
    context_->CSSetConstantBuffers(0u, 3u, constants.data());
    context_->CSSetShader(inner_shaders_[0].Get(), nullptr, 0u);
    ASSERT_TRUE(backend_.begin(context_.Get(), shaders, frame_, constants_buffer_.Get(), source_texture_.srv.Get(), validation_buffer_.srv.Get(), root_arguments_.Get(), 0u, error_)) << error_;
    std::array<ID3D11ShaderResourceView *, 9u> got_reads {};
    std::array<ID3D11UnorderedAccessView *, 8u> got_writes {};
    std::array<ID3D11Buffer *, 3u> got_constants {};
    context_->CSGetShaderResources(0u, 9u, got_reads.data());
    context_->CSGetUnorderedAccessViews(0u, 8u, got_writes.data());
    context_->CSGetConstantBuffers(0u, 3u, got_constants.data());
    EXPECT_EQ(got_reads, reads);
    EXPECT_EQ(got_writes, writes);
    EXPECT_EQ(got_constants, constants);
    for (auto *v : got_reads) {
      if (v) {
        v->Release();
      }
    }
    for (auto *v : got_writes) {
      if (v) {
        v->Release();
      }
    }
    for (auto *v : got_constants) {
      if (v) {
        v->Release();
      }
    }
    ComPtr<ID3D11ComputeShader> got_shader;
    context_->CSGetShader(&got_shader, nullptr, nullptr);
    EXPECT_EQ(got_shader.Get(), inner_shaders_[0].Get());
    ASSERT_TRUE(backend_.abandon(context_.Get(), error_));
    unbind();
    EXPECT_FALSE(backend_.begin(context_.Get(), shaders, frame_, constants_buffer_.Get(), source_texture_.srv.Get(), validation_buffer_.srv.Get(), root_arguments_.Get(), 0u, error_));
  }

  TEST_F(HostSbsBodyWitnessTemporalGpuTest, OwnedRecordedMatchesDirectLiveIdentityRootZeroAndSingleRecording) {
    author(1u, true);
    for (UINT x : {35u, 60u, 90u, 120u}) {
      bar(x);
    }
    frame_for_source(1u);
    ASSERT_TRUE(run_composed(false)) << error_;
    expect_uploaded_exact();
    expect_available();
    const auto reference_body = temporal_body_, reference_final = temporal_final_, reference_receipt = temporal_receipt_;
    EXPECT_EQ(coordinator_.diagnostics().direct_emissions, 1u);
    backend_ = temporal::resources_t {};
    coordinator_ = models::host_sbs_body_witness_temporal_recorded::resources_t {};
    ASSERT_TRUE(run_composed(true)) << error_;
    expect_uploaded_exact();
    expect_available();
    EXPECT_EQ(temporal_body_, reference_body);
    EXPECT_EQ(temporal_final_, reference_final);
    EXPECT_EQ(temporal_receipt_, reference_receipt);
    EXPECT_EQ(coordinator_.diagnostics().recordings, 1u);
    EXPECT_EQ(coordinator_.diagnostics().replays, 1u);
    author(3u, true);
    for (UINT x : {35u, 60u, 90u, 120u}) {
      bar(x);
    }
    frame_for_source(2u);
    ASSERT_TRUE(run_composed(true)) << error_;
    expect_available();
    EXPECT_EQ(temporal_receipt_[4], 2u);
    EXPECT_EQ(temporal_receipt_[76], 3u);
    EXPECT_EQ(temporal_receipt_[8], 1u);
    EXPECT_EQ(temporal_body_, reference_body);
    EXPECT_EQ(coordinator_.diagnostics().recordings, 1u);
    EXPECT_EQ(coordinator_.diagnostics().replays, 2u);
    ASSERT_TRUE(run_composed(true)) << error_;
    expect_available();
    EXPECT_EQ(temporal_receipt_[9], 1u);
    EXPECT_EQ(coordinator_.diagnostics().recordings, 1u);
    EXPECT_EQ(coordinator_.diagnostics().replays, 3u);
    const auto before_zero = colour_words(temporal_uploaded_.at("state.u32"));
    author(6u, true);
    for (UINT x : {35u, 60u, 90u, 120u}) {
      bar(x);
    }
    frame_for_source(3u);
    root_gate_ = 0u;
    ASSERT_TRUE(run_composed(true)) << error_;
    EXPECT_TRUE(std::all_of(temporal_body_.begin(), temporal_body_.end(), [](UINT v) {
      return v == 0u;
    }));
    EXPECT_TRUE(std::all_of(temporal_receipt_.begin(), temporal_receipt_.end(), [](UINT v) {
      return v == 0u;
    }));
    EXPECT_EQ(colour_words(temporal_uploaded_.at("state.u32")), before_zero);
    root_gate_ = 1u;
    ASSERT_TRUE(run_composed(true)) << error_;
    expect_available();
    EXPECT_EQ(temporal_receipt_[4], 3u);
    EXPECT_EQ(temporal_receipt_[76], 6u);
    EXPECT_EQ(temporal_receipt_[8], 1u);
    EXPECT_EQ(coordinator_.diagnostics().recordings, 1u);
    EXPECT_EQ(coordinator_.diagnostics().replays, 5u);
    frame_for_source(3u, 1u);
    ASSERT_TRUE(run_composed(true)) << error_;
    expect_available();
    EXPECT_EQ(temporal_receipt_[9], 0u);
    EXPECT_EQ(temporal_body_, body_);
    EXPECT_EQ(temporal_final_, final_);
    EXPECT_EQ(coordinator_.diagnostics().recordings, 1u);
    EXPECT_EQ(coordinator_.diagnostics().replays, 6u);
  }

  TEST_F(HostSbsBodyWitnessTemporalGpuTest, NullTemporalProgramDelegatesFrozenBaselineAndLeavesHistoryLiteral) {
    author(1u, true);
    bar(50u);
    frame_for_source(1u);
    ASSERT_TRUE(run_composed(true)) << error_;
    expect_available();
    const auto expected_body = body_, expected_final = final_, old_state = colour_words(temporal_uploaded_.at("state.u32"));
    std::array<ID3D11ComputeShader *, 5u> colour {};
    std::array<ID3D11ComputeShader *, 6u> graph {};
    for (UINT i = 0u; i < colour.size(); ++i) {
      colour[i] = inner_shaders_[i].Get();
    }
    for (UINT i = 0u; i < graph.size(); ++i) {
      graph[i] = component_shaders_[i].Get();
    }
    ASSERT_TRUE(coordinator_.dispatch(context_.Get(), colour, graph, constants_buffer_.Get(), source_texture_.srv.Get(), p_buffer_.srv.Get(), support_texture_.srv.Get(), ocr_buffer_.srv.Get(), root_arguments_.Get(), 0u, error_, nullptr)) << error_;
    const auto views = coordinator_.baseline_views();
    const auto read_texture = [&](ID3D11ShaderResourceView *view) {
      ComPtr<ID3D11Resource> r;
      ComPtr<ID3D11Texture2D> t;
      view->GetResource(&r);
      if (FAILED(r.As(&t))) {
        throw std::runtime_error("Expected baseline2D");
      }
      return colour_words(read_archive(archive(t.Get())));
    };
    EXPECT_EQ(read_texture(views.body.Get()), expected_body);
    EXPECT_EQ(read_texture(views.matte.Get()), expected_final);
    const auto temporal_views = backend_.mutable_views();
    ComPtr<ID3D11Resource> r;
    ComPtr<ID3D11Buffer> state;
    temporal_views.state->GetResource(&r);
    ASSERT_TRUE(SUCCEEDED(r.As(&state)));
    EXPECT_EQ(colour_words(read_archive(archive(state.Get()))), old_state);
  }

  TEST_F(HostSbsBodyWitnessTemporalGpuTest, SavedFullSourceTemporalReplay) {
    if (spec_path_.empty()) {
      GTEST_SKIP() << "Explicit saved sequence fixture required";
    }
    ASSERT_EQ(input_spec_.at("schema").get<UINT>(), 1u);
    const char *out = std::getenv("SUNSHINE_TEST_BODY_WITNESS_TEMPORAL_OUTPUT");
    ASSERT_NE(out, nullptr);
    const std::filesystem::path output = out;
    ASSERT_FALSE(std::filesystem::exists(output));
    std::filesystem::create_directories(output);
    nlohmann::json aggregate {{"schema", 1}, {"cases", nlohmann::json::array()}};
    const auto save = [&](const std::filesystem::path &p, const std::vector<char> &bytes) {
      std::filesystem::create_directories(p.parent_path());
      std::ofstream stream(p, std::ios::binary);
      stream.write(bytes.data(), std::streamsize(bytes.size()));
      if (!stream) {
        throw std::runtime_error("Cannot save temporal readback");
      }
      return nlohmann::json {{"path", p.string()}, {"sha256", colour_sha(bytes.data(), bytes.size())}, {"bytes", bytes.size()}};
    };
    UINT observation = 0u;
    for (const auto &sequence : input_spec_.at("sequences")) {
      backend_ = temporal::resources_t {};  // Explicit independent authenticated sequence scope, no serial compression.
      for (const auto &item : sequence.at("cases")) {
        const auto &a = item.at("artifacts");
        auto cbytes = colour_bound(a.at("constants"), 96u);
        std::memcpy(constants_.words.data(), cbytes.data(), 96u);
        const auto c = constants_.words;
        if (width_ != c[0] || height_ != c[1] || crop_top_ != c[2] || crop_height_ != c[3]) {
          ASSERT_TRUE(resize(c[0], c[1], c[2], c[3]));
        }
        std::memcpy(constants_.words.data(), cbytes.data(), 96u);
        source_.resize(std::size_t(width_) * crop_height_);
        auto raw = colour_bound(a.at("source_band"), source_.size() * 8u);
        std::memcpy(source_.data(), raw.data(), raw.size());
        probability_.resize(153600u);
        raw = colour_bound(a.at("probability"), 153600u * 4u);
        std::memcpy(probability_.data(), raw.data(), raw.size());
        support_ = colour_words(colour_bound(a.at("current_support"), source_.size() * 4u));
        ocr_ = colour_words(colour_bound(a.at("ocr_record"), 1024u));
        raw = colour_bound(a.at("frame_constants"), 256u);
        std::memcpy(frame_.words.data(), raw.data(), 256u);
        const auto root = colour_words(colour_bound(a.at("root_arguments"), 12u));
        ASSERT_EQ(root[1], 1u);
        ASSERT_EQ(root[2], 1u);
        ASSERT_LE(root[0], 1u);
        root_gate_ = root[0];
        ASSERT_EQ(frame_.words[48], c[12]);
        ASSERT_EQ(frame_.words[49], c[13]);
        ASSERT_EQ(frame_.words[50], c[16]);
        ASSERT_EQ(frame_.words[51], c[17]);
        ASSERT_TRUE(backend_.ensure(device_.Get(), width_, crop_height_, capacities_, error_));
        const auto execution = input_spec_.value("execution", std::string("recorded"));
        ASSERT_TRUE(execution == "recorded" || execution == "direct");
        ASSERT_TRUE(run_composed(execution == "recorded")) << error_;
        expect_uploaded_exact();
        EXPECT_EQ(temporal_uploaded_.at("uploaded_frame_constants.u32"), colour_raw(frame_.words));
        EXPECT_EQ(temporal_uploaded_.at("uploaded_root_arguments.u32"), colour_bound(a.at("root_arguments"), 12u));
        if (root_gate_ == 1u) {
          expect_available();
        } else {
          EXPECT_TRUE(std::all_of(temporal_receipt_.begin(), temporal_receipt_.end(), [](UINT v) {
            return v == 0u;
          }));
          EXPECT_TRUE(std::all_of(temporal_body_.begin(), temporal_body_.end(), [](UINT v) {
            return v == 0u;
          }));
          EXPECT_TRUE(std::all_of(temporal_final_.begin(), temporal_final_.end(), [](UINT v) {
            return v == 0u;
          }));
        }
        const auto folder = output / ("case_" + std::to_string(++observation));
        nlohmann::json artifacts;
        for (const auto &[name, bytes] : uploaded_) {
          artifacts[name] = save(folder / name, bytes);
        }
        for (const auto &[name, bytes] : temporal_uploaded_) {
          artifacts[name] = save(folder / name, bytes);
        }
        artifacts["baseline_body.u32"] = save(folder / "baseline_body.u32", colour_raw(body_));
        artifacts["baseline_final.u32"] = save(folder / "baseline_final.u32", colour_raw(final_));
        bool exact = true;
        for (const auto &[name, bytes] : std::array<std::pair<const char *, std::vector<char>>, 4u> {{{"body", colour_raw(temporal_body_)}, {"final", colour_raw(temporal_final_)}, {"baseline_body", colour_raw(body_)}, {"baseline_final", colour_raw(final_)}}}) {
          exact &= bytes == colour_bound(item.at("expected").at(name), source_.size() * 4u);
        }
        const nlohmann::json identity {{"schema", 1}, {"case", item}, {"sequence", sequence.at("name")}, {"driver", driver_}, {"adapter", adapter_identity()}, {"temporal_source_closure_sha256", temporal_closure_}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", component_closure_}, {"artifacts", artifacts}, {"expected_whole_baseline_and_candidate_byte_exact", exact}, {"all_semantic_status_UNKNOWN", true}, {"dispatch_policy", "One owned source pipeline: globalvalidate/palette/emission/4CC once; temporalhooks; originalBOX3control; globaltemporalend. Frame256 liveupdate outsidecachedlist; oneExecuteCommandList forrecorded. No mode replay or CPU verdict."}, {"execution", execution}, {"recordings", coordinator_.diagnostics().recordings}, {"replays", coordinator_.diagnostics().replays}, {"direct_emissions", coordinator_.diagnostics().direct_emissions}, {"source_identity_scope", item.at("provenance")}};
        std::filesystem::create_directories(folder);
        std::ofstream file(folder / "body_witness_temporal_identity.json");
        file << identity.dump(2) << '\n';
        ASSERT_TRUE(file.good());
        aggregate["cases"].push_back(identity);
        EXPECT_TRUE(exact) << "Saved output preserved; no metric or rule waiver";
      }
    }
    std::ofstream file(output / "body_witness_temporal_replay_identity.json");
    file << aggregate.dump(2) << '\n';
    ASSERT_TRUE(file.good());
  }

  #include "body_witness_temporal_math_cases.inc"
  #include "body_witness_temporal_cost_cases.inc"
  #include "body_witness_temporal_gpu_frame_replay_cases.inc"
}  // namespace
#endif
