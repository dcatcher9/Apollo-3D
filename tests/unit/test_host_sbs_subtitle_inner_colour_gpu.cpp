/**
 * @file tests/unit/test_host_sbs_subtitle_inner_colour_gpu.cpp
 * @brief Opt-in complete current-input colour extraction plus frozen component operator.
 * Readbacks select diagnostic exports only; GPU-written mode arguments govern all work.
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
  #include <utility>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace colour_cv2 = models::depth_coordinate_v2;
  using colour_half_t = std::array<std::uint16_t, 4u>;
  using colour_rect_t = std::array<UINT, 4u>;
  struct alignas(16) colour_constants_t { std::array<UINT, 24u> words; };
  struct alignas(16) colour_request_t { std::array<UINT, 4u> words; };
  static_assert(sizeof(colour_constants_t) == 96u && sizeof(colour_request_t) == 16u);

  std::vector<char> colour_bytes(const std::filesystem::path &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0) throw std::runtime_error("Cannot read inner-colour input: " + path.string());
    std::vector<char> out(static_cast<std::size_t>(f.tellg())); f.seekg(0);
    if (!out.empty()) { f.read(out.data(), static_cast<std::streamsize>(out.size())); }
    if (!f) throw std::runtime_error("Incomplete inner-colour input"); return out;
  }
  std::string colour_sha(const void *data, std::size_t size) {
    const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), size));
    constexpr char digits[] = "0123456789abcdef"; std::string out;
    for (auto byte : digest) { out.push_back(digits[byte >> 4u]); out.push_back(digits[byte & 15u]); } return out;
  }
  std::vector<char> colour_bound(const nlohmann::json &b, std::size_t size) {
    auto out = colour_bytes(b.at("path").get<std::string>());
    if (out.size() != size || out.size() != b.at("bytes").get<std::size_t>() || colour_sha(out.data(), out.size()) != b.at("sha256").get<std::string>()) {
      throw std::runtime_error("Inner-colour binding mismatch");
    }
    return out;
  }
  template<class T> std::vector<char> colour_raw(const std::vector<T> &v) {
    std::vector<char> out(v.size() * sizeof(T)); if (!out.empty()) std::memcpy(out.data(), v.data(), out.size()); return out;
  }
  template<std::size_t N> std::vector<char> colour_raw(const std::array<UINT, N> &v) {
    std::vector<char> out(N * 4u); std::memcpy(out.data(), v.data(), out.size()); return out;
  }
  std::vector<UINT> colour_words(const std::vector<char> &v) {
    if (v.size() % 4u != 0u) throw std::runtime_error("Inner-colour word size mismatch");
    std::vector<UINT> out(v.size() / 4u); if (!v.empty()) std::memcpy(out.data(), v.data(), v.size()); return out;
  }
  std::uint16_t colour_half(float value) {
    const auto bits = std::bit_cast<UINT>(value); const auto sign = std::uint16_t((bits >> 16u) & 0x8000u);
    const auto exponent = (bits >> 23u) & 255u; auto fraction = bits & 0x7fffffu;
    if (exponent == 255u) return sign | (fraction ? 0x7e00u : 0x7c00u);
    const int e = int(exponent) - 127 + 15;
    if (e >= 31) return sign | 0x7c00u;
    if (e <= 0) {
      if (e < -10) return sign;
      fraction |= 0x800000u; const auto shift = unsigned(14 - e);
      return sign | std::uint16_t((fraction + (1u << (shift - 1u)) - 1u + ((fraction >> shift) & 1u)) >> shift);
    }
    fraction += 0xfffu + ((fraction >> 13u) & 1u); return sign | std::uint16_t((UINT(e) << 10u) + (fraction >> 13u));
  }
  float colour_float(std::uint16_t value) {
    const auto exponent = (value >> 10u) & 31u, mantissa = value & 1023u;
    const float magnitude = exponent == 0u ? std::ldexp(float(mantissa), -24) :
      exponent == 31u ? (mantissa ? NAN : INFINITY) : std::ldexp(1.f + float(mantissa) / 1024.f, int(exponent) - 15);
    return (value & 0x8000u) ? -magnitude : magnitude;
  }
  struct colour_texture_t { ComPtr<ID3D11Texture2D> value; ComPtr<ID3D11ShaderResourceView> srv; ComPtr<ID3D11UnorderedAccessView> uav; };
  struct colour_buffer_t { ComPtr<ID3D11Buffer> value; ComPtr<ID3D11ShaderResourceView> srv; ComPtr<ID3D11UnorderedAccessView> uav; };
  bool colour_texture(ID3D11Device *d, UINT w, UINT h, DXGI_FORMAT format, bool output, colour_texture_t &out) {
    out = {}; D3D11_TEXTURE2D_DESC desc {}; desc.Width = w; desc.Height = h; desc.MipLevels = desc.ArraySize = 1u;
    desc.Format = format; desc.SampleDesc.Count = 1u; desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | (output ? D3D11_BIND_UNORDERED_ACCESS : 0u);
    return SUCCEEDED(d->CreateTexture2D(&desc, nullptr, &out.value)) && SUCCEEDED(d->CreateShaderResourceView(out.value.Get(), nullptr, &out.srv)) &&
      (!output || SUCCEEDED(d->CreateUnorderedAccessView(out.value.Get(), nullptr, &out.uav)));
  }
  bool colour_buffer(ID3D11Device *d, UINT words, bool output, colour_buffer_t &out) {
    out = {}; D3D11_BUFFER_DESC desc {}; desc.ByteWidth = words * 4u; desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | (output ? D3D11_BIND_UNORDERED_ACCESS : 0u);
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride = 4u;
    return SUCCEEDED(d->CreateBuffer(&desc, nullptr, &out.value)) && SUCCEEDED(d->CreateShaderResourceView(out.value.Get(), nullptr, &out.srv)) &&
      (!output || SUCCEEDED(d->CreateUnorderedAccessView(out.value.Get(), nullptr, &out.uav)));
  }
  struct colour_archive_t { ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11Buffer> buffer; UINT row_bytes {}, height {}; };
  struct colour_mode_t {
    std::map<std::string, colour_archive_t> pending;
    std::map<std::string, std::vector<char>> planes;
    bool dispatched = false;
  };
  struct colour_unit_t { colour_archive_t palette_pending, args_pending; std::vector<UINT> palette, args; };

  class HostSbsSubtitleInnerColourGpuTest: public ::testing::Test {
  protected:
    void SetUp() override {
      inner_root_ = component_root_ = SUNSHINE_SHADERS_DIR; std::string inner_pin, component_pin;
      if (std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) == "SavedNativeInnerColourReplay") {
        const auto *path = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_CASES");
        if (!path) { GTEST_SKIP() << "Explicit full inner-colour replay input required"; }
        auto bytes = colour_bytes(path); auto spec = nlohmann::json::parse(bytes.begin(), bytes.end());
        driver_ = spec.value("driver", std::string("warp")); const auto parent = std::filesystem::path(path).parent_path();
        inner_root_ = parent / spec.at("inner_shader_root").get<std::string>(); component_root_ = parent / spec.at("component_shader_root").get<std::string>();
        inner_pin = spec.at("inner_source_closure_sha256").get<std::string>(); component_pin = spec.at("component_source_closure_sha256").get<std::string>();
      } else {
        const auto *ip = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_SHADER_PIN"), *cp = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_COMPONENT_PIN");
        if (!ip || !cp) { GTEST_SKIP() << "Explicit two-group research source pins required"; }
        inner_pin = ip; component_pin = cp;
        if (const auto *p = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_SHADER_ROOT")) inner_root_ = p;
        if (const auto *p = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_COMPONENT_ROOT")) component_root_ = p;
        if (const auto *p = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_DRIVER")) driver_ = p;
      }
      ASSERT_EQ(inner_pin.size(), 64u); ASSERT_EQ(component_pin.size(), 64u); ASSERT_TRUE(driver_ == "warp" || driver_ == "hardware");
      D3D_FEATURE_LEVEL actual {}; const D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, driver_ == "hardware" ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP,
        nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &device_, &actual, &context_))); ASSERT_EQ(actual, D3D_FEATURE_LEVEL_11_0);
      namespace cache = models::host_sbs_shader_cache;
      const std::array inner_specs {
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "palette_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "emit_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "clear_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_inner_colour_cs.hlsl", "accumulate_main", "cs_5_0"}
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
      inner_closure_ = cache::source_closure_sha256(inner); component_closure_ = cache::source_closure_sha256(component);
      ASSERT_EQ(inner_closure_, inner_pin); ASSERT_EQ(component_closure_, component_pin);
      for (std::size_t i = 0; i < inner_specs.size(); ++i) {
        auto code = cache::get(inner, inner_specs[i]); ASSERT_TRUE(code && !code->empty());
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &inner_shaders_[i])));
      }
      for (std::size_t i = 0; i < component_specs.size(); ++i) {
        auto code = cache::get(component, component_specs[i]); ASSERT_TRUE(code && !code->empty());
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &component_shaders_[i])));
      }
      D3D11_BUFFER_DESC d {}; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_CONSTANT_BUFFER; d.ByteWidth = 96u;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &constants_buffer_))); d.ByteWidth = 16u;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &request_buffer_))); ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &component_constants_)));
      d.ByteWidth = 96u; d.BindFlags = D3D11_BIND_UNORDERED_ACCESS; d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &arguments_)));
      D3D11_UNORDERED_ACCESS_VIEW_DESC u {}; u.Format = DXGI_FORMAT_R32_TYPELESS; u.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      u.Buffer.NumElements = 24u; u.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      ASSERT_TRUE(SUCCEEDED(device_->CreateUnorderedAccessView(arguments_.Get(), &u, &arguments_uav_)));
      ASSERT_TRUE(resize(160u, 90u));
    }
    void TearDown() override { unbind(); }
    void unbind() {
      if (!context_) return;
      const std::array<ID3D11ShaderResourceView *, 8u> r {}; const std::array<ID3D11UnorderedAccessView *, 8u> w {};
      context_->CSSetShaderResources(0u, 8u, r.data()); context_->CSSetUnorderedAccessViews(0u, 8u, w.data(), nullptr);
    }
    bool resize(UINT width, UINT height, UINT top = UINT(-1), UINT crop_height = 0u) {
      unbind(); width_ = width; height_ = height; crop_height_ = crop_height ? crop_height : std::min(height, (width + 5u) / 6u);
      crop_top_ = top == UINT(-1) ? height - crop_height_ : top;
      if (width == 0u || height == 0u || width > 5120u || height > 5120u || crop_height_ == 0u || crop_top_ > height || crop_height_ != height - crop_top_) return false;
      const auto n = width * crop_height_; requested_ = {}; requested_[0u].fill(true);
      const bool ok = colour_texture(device_.Get(), width, crop_height_, DXGI_FORMAT_R16G16B16A16_FLOAT, false, source_texture_) &&
        colour_texture(device_.Get(), width, crop_height_, DXGI_FORMAT_R32_UINT, false, support_texture_) &&
        colour_buffer(device_.Get(), 153600u, false, p_buffer_) && colour_buffer(device_.Get(), 256u, false, ocr_buffer_) &&
        colour_buffer(device_.Get(), 128u, true, palette_buffer_) && colour_buffer(device_.Get(), n + 1u, true, parent_buffer_) && colour_buffer(device_.Get(), n + 1u, true, flags_buffer_);
      if (!ok) return false;
      for (auto *t : {&eligible_texture_, &seed_texture_, &domain_texture_, &cover_texture_, &body_texture_, &final_texture_, &accum_body_, &accum_final_}) {
        if (!colour_texture(device_.Get(), width, crop_height_, DXGI_FORMAT_R32_UINT, true, *t)) return false;
      }
      author(1u, false); return true;
    }
    UINT row_edge(UINT detector_y) const {
      const auto &c = constants_.words; const auto numerator = std::uint64_t(crop_top_ * 160u + detector_y * crop_height_) * (c[11u] - c[9u]);
      return c[9u] + UINT((numerator + std::uint64_t(height_) * 160u - 1u) / (std::uint64_t(height_) * 160u));
    }
    colour_rect_t field_rect(colour_rect_t native) const {
      const auto &c = constants_.words; const UINT cw = c[10u] - c[8u], ch = c[11u] - c[9u];
      return {c[8u] + UINT(std::uint64_t(native[0u]) * cw / width_), c[9u] + UINT(std::uint64_t(native[1u] + crop_top_) * ch / height_),
        c[8u] + UINT((std::uint64_t(native[2u]) * cw + width_ - 1u) / width_), c[9u] + UINT((std::uint64_t(native[3u] + crop_top_) * ch + height_ - 1u) / height_)};
    }
    void author(UINT frame, bool caption, float background = .15f) {
      const UINT fw = colour_cv2::model_calibrated_shapes[0u].width, fh = colour_cv2::model_calibrated_shapes[0u].height;
      constants_.words = {width_, height_, crop_top_, crop_height_, fw, fh, 0u, 0u, 0u, 0u, fw, fh,
        frame, 0u, 7u, 0u, 1000000u + frame * 33333u, 0u, 0u, 0u, 1u, SBC_SCHEMA, 0u, 0u};
      constants_.words[6u] = row_edge(colour_cv2::subtitle_ocr_safe_row_top); constants_.words[7u] = row_edge(colour_cv2::subtitle_ocr_safe_row_bottom);
      source_.assign(std::size_t(width_) * crop_height_, {colour_half(background), colour_half(background), colour_half(background), colour_half(1.f)});
      support_.assign(source_.size(), 0u); probability_.assign(153600u, caption ? .9f : 0.f); ocr_.assign(256u, 0u); request_reserved_ = 0u;
      ocr_[0u] = 4u; ocr_[1u] = 0x3852434fu; ocr_[2u] = 1u; ocr_[3u] = ocr_[4u] = caption ? 1u : 0u;
      std::copy(constants_.words.begin() + 12u, constants_.words.begin() + 16u, ocr_.begin() + 5u);
      ocr_[9u] = width_; ocr_[10u] = height_; std::copy(constants_.words.begin() + 4u, constants_.words.begin() + 8u, ocr_.begin() + 11u);
      if (caption) {
        auto core = field_rect({25u, 5u, 135u, 24u}), cover = field_rect({20u, 3u, 140u, 26u});
        core[1u] = std::max(core[1u], constants_.words[6u]); core[3u] = std::min(core[3u], constants_.words[7u]);
        cover[1u] = std::max(cover[1u], constants_.words[6u]); cover[3u] = std::min(cover[3u], constants_.words[7u]);
        for (const auto &[offset, r] : {std::pair {16u, core}, std::pair {144u, cover}}) {
          std::copy(r.begin(), r.end(), ocr_.begin() + offset); ocr_[offset + 4u] = std::bit_cast<UINT>(.9f); ocr_[offset + 6u] = 1u;
        }
      }
    }
    std::size_t pixel(UINT x, UINT y) const { return std::size_t(y) * width_ + x; }
    void paint(colour_rect_t r, std::array<float, 3u> rgb, bool support) {
      for (UINT y = r[1u]; y < std::min(r[3u], crop_height_); ++y) {
        for (UINT x = r[0u]; x < std::min(r[2u], width_); ++x) {
          source_[pixel(x, y)] = {colour_half(rgb[0u]), colour_half(rgb[1u]), colour_half(rgb[2u]), colour_half(1.f)}; support_[pixel(x, y)] = support ? 1u : 0u;
        }
      }
    }
    void bar(UINT x, std::array<float, 3u> rgb = {.95f, .95f, .95f}, bool support = true, std::array<float, 3u> border = {0.f, 0.f, 0.f}) {
      paint({x - 2u, 6u, x + 6u, 22u}, border, false); paint({x, 8u, x + 4u, 20u}, rgb, support);
    }
    void native_probability(colour_rect_t r, float value) {
      for (UINT y = 0u; y < 160u; ++y) { for (UINT x = 0u; x < 960u; ++x) {
        const double nx = (x + .5) * width_ / 960u, ny = (y + .5) * crop_height_ / 160u;
        if (nx >= r[0u] && nx < r[2u] && ny >= r[1u] && ny < r[3u]) probability_[std::size_t(y) * 960u + x] = value;
      } }
    }
    colour_archive_t archive(ID3D11Texture2D *source) {
      D3D11_TEXTURE2D_DESC d {}; source->GetDesc(&d); colour_archive_t a;
      a.row_bytes = d.Width * (d.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8u : 4u); a.height = d.Height;
      archive_bytes_ += std::uint64_t(a.row_bytes) * a.height;
      if (archive_bytes_ > 2ull * 1024u * 1024u * 1024u) throw std::runtime_error("Functional diagnostic archive allocation cap exceeded");
      d.Usage = D3D11_USAGE_STAGING; d.BindFlags = d.MiscFlags = 0u; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      if (FAILED(device_->CreateTexture2D(&d, nullptr, &a.texture))) throw std::runtime_error("Cannot allocate inner-colour texture archive");
      context_->CopyResource(a.texture.Get(), source); return a;
    }
    colour_archive_t archive(ID3D11Buffer *source) {
      D3D11_BUFFER_DESC d {}; source->GetDesc(&d); colour_archive_t a; a.row_bytes = d.ByteWidth; a.height = 1u;
      archive_bytes_ += d.ByteWidth; if (archive_bytes_ > 2ull * 1024u * 1024u * 1024u) throw std::runtime_error("Functional archive allocation cap exceeded");
      d.Usage = D3D11_USAGE_STAGING; d.BindFlags = d.MiscFlags = d.StructureByteStride = 0u; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      if (FAILED(device_->CreateBuffer(&d, nullptr, &a.buffer))) throw std::runtime_error("Cannot allocate inner-colour buffer archive");
      context_->CopyResource(a.buffer.Get(), source); return a;
    }
    std::vector<char> read_archive(const colour_archive_t &a) {
      ID3D11Resource *r = a.texture ? static_cast<ID3D11Resource *>(a.texture.Get()) : static_cast<ID3D11Resource *>(a.buffer.Get());
      D3D11_MAPPED_SUBRESOURCE m {}; if (!r || FAILED(context_->Map(r, 0u, D3D11_MAP_READ, 0u, &m))) throw std::runtime_error("Cannot map inner-colour diagnostic archive");
      std::vector<char> out(std::size_t(a.row_bytes) * a.height);
      for (UINT y = 0u; y < a.height; ++y) { std::memcpy(out.data() + std::size_t(y) * a.row_bytes, static_cast<const char *>(m.pData) + std::size_t(y) * (a.texture ? m.RowPitch : a.row_bytes), a.row_bytes); }
      context_->Unmap(r, 0u); return out;
    }
    std::array<ID3D11ShaderResourceView *, 8u> inner_reads() {
      return {source_texture_.srv.Get(), p_buffer_.srv.Get(), support_texture_.srv.Get(), ocr_buffer_.srv.Get(), palette_buffer_.srv.Get(), nullptr, nullptr, nullptr};
    }
    void bind_inner(const std::array<ID3D11ShaderResourceView *, 8u> &r, const std::array<ID3D11UnorderedAccessView *, 8u> &w) {
      context_->CSSetShaderResources(0u, 8u, r.data()); context_->CSSetUnorderedAccessViews(0u, 8u, w.data(), nullptr);
      ID3D11Buffer *cb[] {constants_buffer_.Get(), request_buffer_.Get()}; context_->CSSetConstantBuffers(0u, 2u, cb);
    }
    bool run() {
      unbind(); units_ = {}; modes_.clear(); input_pending_.clear(); uploaded_.clear(); archive_bytes_ = 0u;
      context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, &constants_, 0u, 0u);
      context_->UpdateSubresource(source_texture_.value.Get(), 0u, nullptr, source_.data(), width_ * 8u, 0u);
      context_->UpdateSubresource(support_texture_.value.Get(), 0u, nullptr, support_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(p_buffer_.value.Get(), 0u, nullptr, probability_.data(), 0u, 0u); context_->UpdateSubresource(ocr_buffer_.value.Get(), 0u, nullptr, ocr_.data(), 0u, 0u);
      const std::array<UINT, 4u> math {width_, crop_height_, width_ * crop_height_, 0u}; context_->UpdateSubresource(component_constants_.Get(), 0u, nullptr, math.data(), 0u, 0u);
      input_pending_["uploaded_source.rgba16f"] = archive(source_texture_.value.Get()); input_pending_["uploaded_probability.f32"] = archive(p_buffer_.value.Get());
      input_pending_["uploaded_current_support.u32"] = archive(support_texture_.value.Get()); input_pending_["uploaded_ocr_record.u32"] = archive(ocr_buffer_.value.Get()); input_pending_["uploaded_constants.u32"] = archive(constants_buffer_.Get());
      colour_request_t request {{0u, 0u, request_reserved_, 0u}}; context_->UpdateSubresource(request_buffer_.Get(), 0u, nullptr, &request, 0u, 0u);
      std::array<ID3D11ShaderResourceView *, 8u> reads {}; std::array<ID3D11UnorderedAccessView *, 8u> writes {};
      writes[5u] = accum_body_.uav.Get(); writes[6u] = accum_final_.uav.Get(); bind_inner(reads, writes);
      context_->CSSetShader(inner_shaders_[2u].Get(), nullptr, 0u); context_->Dispatch((width_ + 15u) / 16u, (crop_height_ + 15u) / 16u, 1u); unbind();
      // All8x8 command iterations are fixed. No palette/argument readback is mapped
      // until every command below completes. GPU zero arguments alone skip work.
      for (UINT slot = 0u; slot < 8u; ++slot) {
        request.words = {slot, 0u, request_reserved_, 0u}; context_->UpdateSubresource(request_buffer_.Get(), 0u, nullptr, &request, 0u, 0u);
        reads = inner_reads(); reads[4u] = nullptr; writes = {}; writes[0u] = palette_buffer_.uav.Get(); writes[7u] = arguments_uav_.Get(); bind_inner(reads, writes);
        context_->CSSetShader(inner_shaders_[0u].Get(), nullptr, 0u); context_->Dispatch(1u, 1u, 1u); unbind();
        units_[slot].palette_pending = archive(palette_buffer_.value.Get()); units_[slot].args_pending = archive(arguments_.Get());
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const UINT offset = mode * 12u; request.words = {slot, mode, request_reserved_, 0u}; context_->UpdateSubresource(request_buffer_.Get(), 0u, nullptr, &request, 0u, 0u);
          reads = inner_reads(); writes = {}; writes[1u] = eligible_texture_.uav.Get(); writes[2u] = seed_texture_.uav.Get(); writes[3u] = domain_texture_.uav.Get(); writes[4u] = cover_texture_.uav.Get(); bind_inner(reads, writes);
          context_->CSSetShader(inner_shaders_[1u].Get(), nullptr, 0u); context_->DispatchIndirect(arguments_.Get(), offset); unbind();
          reads = {eligible_texture_.srv.Get(), seed_texture_.srv.Get(), domain_texture_.srv.Get(), cover_texture_.srv.Get(), nullptr, nullptr, nullptr, nullptr};
          writes = {parent_buffer_.uav.Get(), flags_buffer_.uav.Get(), body_texture_.uav.Get(), final_texture_.uav.Get(), nullptr, nullptr, nullptr, nullptr};
          context_->CSSetShaderResources(0u, 8u, reads.data()); context_->CSSetUnorderedAccessViews(0u, 8u, writes.data(), nullptr); context_->CSSetConstantBuffers(0u, 1u, component_constants_.GetAddressOf());
          for (UINT phase = 0u; phase < 4u; ++phase) { context_->CSSetShader(component_shaders_[phase].Get(), nullptr, 0u); context_->DispatchIndirect(arguments_.Get(), offset); }
          context_->CSSetShader(component_shaders_[5u].Get(), nullptr, 0u); context_->DispatchIndirect(arguments_.Get(), offset); unbind();
          reads = inner_reads(); reads[5u] = body_texture_.srv.Get(); reads[6u] = final_texture_.srv.Get(); reads[7u] = parent_buffer_.srv.Get();
          writes = {}; writes[5u] = accum_body_.uav.Get(); writes[6u] = accum_final_.uav.Get(); bind_inner(reads, writes);
          context_->CSSetShader(inner_shaders_[3u].Get(), nullptr, 0u); context_->DispatchIndirect(arguments_.Get(), offset); unbind();
          if (requested_[slot][mode]) {
            auto &m = modes_[{slot, mode}]; m.pending["eligible.u32"] = archive(eligible_texture_.value.Get()); m.pending["seeds.u32"] = archive(seed_texture_.value.Get());
            m.pending["domain_tag.u32"] = archive(domain_texture_.value.Get()); m.pending["cover_clip.u32"] = archive(cover_texture_.value.Get());
            m.pending["parent_roots.u32"] = archive(parent_buffer_.value.Get()); m.pending["root_flags.u32"] = archive(flags_buffer_.value.Get());
            m.pending["body.u32"] = archive(body_texture_.value.Get()); m.pending["final.u32"] = archive(final_texture_.value.Get()); m.pending["uploaded_request.u32"] = archive(request_buffer_.Get());
          }
        }
      }
      const auto body_pending = archive(accum_body_.value.Get()), final_pending = archive(accum_final_.value.Get());
      // This is the first CPU mapping after the entire fixed-command GPU chain.
      for (auto &[name, a] : input_pending_) uploaded_[name] = read_archive(a);
      body_ = colour_words(read_archive(body_pending)); final_ = colour_words(read_archive(final_pending)); capture_complete_ = true;
      for (UINT slot = 0u; slot < 8u; ++slot) {
        units_[slot].palette = colour_words(read_archive(units_[slot].palette_pending)); units_[slot].args = colour_words(read_archive(units_[slot].args_pending));
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const auto &a = units_[slot].args; const bool active = a[3u * mode] != 0u || a[3u * mode + 1u] != 0u || a[3u * mode + 2u] != 0u;
          if (active && !requested_[slot][mode]) capture_complete_ = false;
          if (!requested_[slot][mode]) continue;
          auto &m = modes_.at({slot, mode}); m.dispatched = active;
          for (auto &[name, pending] : m.pending) m.planes[name] = read_archive(pending);
        }
      }
      return true;
    }
    void expect_uploaded_exact() {
      EXPECT_EQ(uploaded_.at("uploaded_source.rgba16f"), colour_raw(source_)); EXPECT_EQ(uploaded_.at("uploaded_probability.f32"), colour_raw(probability_));
      EXPECT_EQ(uploaded_.at("uploaded_current_support.u32"), colour_raw(support_)); EXPECT_EQ(uploaded_.at("uploaded_ocr_record.u32"), colour_raw(ocr_)); EXPECT_EQ(uploaded_.at("uploaded_constants.u32"), colour_raw(constants_.words));
    }
    bool uploaded_byte_exact() const {
      return uploaded_.at("uploaded_source.rgba16f") == colour_raw(source_) && uploaded_.at("uploaded_probability.f32") == colour_raw(probability_) &&
        uploaded_.at("uploaded_current_support.u32") == colour_raw(support_) && uploaded_.at("uploaded_ocr_record.u32") == colour_raw(ocr_) && uploaded_.at("uploaded_constants.u32") == colour_raw(constants_.words);
    }
    bool arguments_byte_exact() const {
      for (const auto &u : units_) {
        if (u.palette.size() != 128u || u.args.size() != 24u) return false;
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const bool active = u.palette[2u] == 1u && u.palette[22u] == 2u && mode < u.palette[4u] && u.palette[64u + 8u * mode + 3u] != 0u;
          if (u.args[3u * mode] != (active ? (width_ + 15u) / 16u : 0u) || u.args[3u * mode + 1u] != (active ? (crop_height_ + 15u) / 16u : 0u) || u.args[3u * mode + 2u] != (active ? 1u : 0u)) return false;
        }
      }
      return true;
    }
    bool gpu_or_byte_exact() const {
      if (!capture_complete_) return false;
      std::vector<UINT> b(body_.size(), 0u), f(final_.size(), 0u);
      for (const auto &[key, m] : modes_) {
        if (!m.dispatched) continue;
        const auto mb = colour_words(m.planes.at("body.u32")), mf = colour_words(m.planes.at("final.u32"));
        if (mb.size() != b.size() || mf.size() != f.size()) return false;
        for (std::size_t p = 0u; p < b.size(); ++p) { if (mb[p] > 1u || mf[p] > 1u) return false; b[p] |= mb[p]; f[p] |= mf[p]; }
      }
      return b == body_ && f == final_;
    }
    void expect_args_exact() {
      for (UINT slot = 0u; slot < 8u; ++slot) { const auto &u = units_[slot]; ASSERT_EQ(u.palette.size(), 128u); ASSERT_EQ(u.args.size(), 24u);
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const bool active = u.palette[2u] == 1u && u.palette[22u] == 2u && mode < u.palette[4u] && u.palette[64u + 8u * mode + 3u] != 0u;
          EXPECT_EQ(u.args[3u * mode], active ? (width_ + 15u) / 16u : 0u); EXPECT_EQ(u.args[3u * mode + 1u], active ? (crop_height_ + 15u) / 16u : 0u); EXPECT_EQ(u.args[3u * mode + 2u], active ? 1u : 0u);
        }
      }
    }
    void expect_gpu_or_exact() {
      ASSERT_TRUE(capture_complete_) << "Diagnostics must capture every dispatched mode; they never control its execution";
      std::vector<UINT> reference_body(body_.size(), 0u), reference_final(final_.size(), 0u);
      for (const auto &[key, m] : modes_) {
        if (!m.dispatched) continue;
        const auto body = colour_words(m.planes.at("body.u32")), final = colour_words(m.planes.at("final.u32"));
        ASSERT_EQ(body.size(), reference_body.size()); ASSERT_EQ(final.size(), reference_final.size());
        for (std::size_t p = 0u; p < body.size(); ++p) { ASSERT_LE(body[p], 1u); ASSERT_LE(final[p], 1u); reference_body[p] |= body[p]; reference_final[p] |= final[p]; }
      }
      EXPECT_EQ(body_, reference_body); EXPECT_EQ(final_, reference_final);
    }
    std::pair<UINT, UINT> expected_peak_tie(UINT target_peak, UINT target_bin) const {
      // Independent authored-grid expectation. These tests use globally strong P
      // and four-pixel-wide outlined bars: every matching colour has a +/-2 edge.
      // This does not run the CPU candidate or supply a GPU palette verdict.
      const auto &c = constants_.words; const UINT cw = c[10u] - c[8u], ch = c[11u] - c[9u];
      UINT first = UINT(-1), count = 0u;
      for (UINT grid = 0u; grid < 512u; ++grid) {
        const UINT gx = grid % 64u, gy = grid / 64u;
        const UINT x = UINT(((std::uint64_t(ocr_[16u] - c[8u]) * 128u + (2u * gx + 1u) * (ocr_[18u] - ocr_[16u])) * width_) / (128u * cw));
        const int absolute_y = int(((std::uint64_t(ocr_[17u] - c[9u]) * 16u + (2u * gy + 1u) * (ocr_[19u] - ocr_[17u])) * height_) / (16u * ch));
        const UINT y = UINT(std::clamp(absolute_y - int(crop_top_), 0, int(crop_height_) - 1));
        const auto &v = source_[pixel(std::min(x, width_ - 1u), y)]; std::array<UINT, 3u> q {}, bin {};
        for (UINT lane = 0u; lane < 3u; ++lane) {
          const float value = colour_float(v[lane]); const float scaled = value * 65535.f; q[lane] = UINT(std::floor(scaled + .5f)); bin[lane] = std::min(UINT(value * 4.f), 3u);
        }
        if (*std::max_element(q.begin(), q.end()) == target_peak && bin[0u] * 16u + bin[1u] * 4u + bin[2u] == target_bin) { first = std::min(first, grid); ++count; }
      }
      return {first, count};
    }
    nlohmann::json adapter_identity() {
      ComPtr<IDXGIDevice> dxgi; ComPtr<IDXGIAdapter> adapter; DXGI_ADAPTER_DESC d {};
      if (FAILED(device_.As(&dxgi)) || FAILED(dxgi->GetAdapter(&adapter)) || FAILED(adapter->GetDesc(&d))) return nullptr;
      std::wstring wide(d.Description); return {{"description", std::string(wide.begin(), wide.end())}, {"vendor_id", d.VendorId}, {"device_id", d.DeviceId}, {"luid_low", d.AdapterLuid.LowPart}, {"luid_high", d.AdapterLuid.HighPart}};
    }
    ComPtr<ID3D11Device> device_; ComPtr<ID3D11DeviceContext> context_;
    std::array<ComPtr<ID3D11ComputeShader>, 4u> inner_shaders_; std::array<ComPtr<ID3D11ComputeShader>, 6u> component_shaders_;
    ComPtr<ID3D11Buffer> constants_buffer_, request_buffer_, component_constants_, arguments_; ComPtr<ID3D11UnorderedAccessView> arguments_uav_;
    colour_texture_t source_texture_, support_texture_, eligible_texture_, seed_texture_, domain_texture_, cover_texture_, body_texture_, final_texture_, accum_body_, accum_final_;
    colour_buffer_t p_buffer_, ocr_buffer_, palette_buffer_, parent_buffer_, flags_buffer_;
    colour_constants_t constants_ {}; UINT width_ {}, height_ {}, crop_top_ {}, crop_height_ {}, request_reserved_ {};
    std::vector<colour_half_t> source_; std::vector<float> probability_; std::vector<UINT> support_, ocr_, body_, final_;
    std::array<std::array<bool, 8u>, 8u> requested_ {}; std::array<colour_unit_t, 8u> units_ {};
    std::map<std::pair<UINT, UINT>, colour_mode_t> modes_; std::map<std::string, colour_archive_t> input_pending_;
    std::map<std::string, std::vector<char>> uploaded_; std::uint64_t archive_bytes_ {}; bool capture_complete_ {};
    std::filesystem::path inner_root_, component_root_; std::string inner_closure_, component_closure_, driver_ {"warp"};
  };

  TEST_F(HostSbsSubtitleInnerColourGpuTest, AuthenticatedEmptyClearsAccumulationWhileInactiveScratchHolds) {
    author(1u, true); bar(50u); ASSERT_TRUE(run()); expect_uploaded_exact(); expect_args_exact(); expect_gpu_or_exact();
    ASSERT_GT(std::count(body_.begin(), body_.end(), 1u), 0); const auto prior = modes_.at({0u, 0u}).planes.at("eligible.u32");
    author(2u, false); ASSERT_TRUE(run()); expect_uploaded_exact(); expect_args_exact(); expect_gpu_or_exact();
    EXPECT_EQ(units_[0u].palette[2u], 1u); EXPECT_EQ(units_[0u].palette[22u], 1u); EXPECT_EQ(units_[0u].palette[4u], 0u);
    EXPECT_EQ(std::count(body_.begin(), body_.end(), 1u), 0); EXPECT_EQ(std::count(final_.begin(), final_.end(), 1u), 0);
    EXPECT_FALSE(modes_.at({0u, 0u}).dispatched); EXPECT_EQ(modes_.at({0u, 0u}).planes.at("eligible.u32"), prior);
  }
  TEST_F(HostSbsSubtitleInnerColourGpuTest, UniformColourHasNoPaletteEdgesOrRectangleFallback) {
    author(1u, true, .95f); std::fill(support_.begin(), support_.end(), 1u); ASSERT_TRUE(run()); expect_uploaded_exact(); expect_args_exact(); expect_gpu_or_exact();
    EXPECT_EQ(units_[0u].palette[2u], 1u); EXPECT_EQ(units_[0u].palette[5u], 0u); EXPECT_EQ(units_[0u].palette[4u], 0u); EXPECT_EQ(std::count(final_.begin(), final_.end(), 1u), 0);
  }
  TEST_F(HostSbsSubtitleInnerColourGpuTest, Q16PeakTieReloadsTheEarliestActualGridColour) {
    author(1u, true); bar(50u, {.95f, .90f, .95f}); bar(100u, {.95f, .95f, .90f}); ASSERT_TRUE(run()); expect_uploaded_exact(); expect_args_exact(); expect_gpu_or_exact();
    const auto &p = units_[0u].palette; ASSERT_EQ(p[4u], 1u); EXPECT_EQ(p[68u], 63u);
    const float scaled = colour_float(colour_half(.95f)) * 65535.f; const UINT peak = UINT(std::floor(scaled + .5f));
    const auto [first, votes] = expected_peak_tie(peak, 63u); ASSERT_LT(first, 512u);
    EXPECT_EQ(p[67u], peak); EXPECT_EQ(p[69u], votes); EXPECT_EQ(p[70u], first);
    const float secondary = colour_float(colour_half(.90f)) * 65535.f;
    EXPECT_EQ(p[64u], peak); EXPECT_EQ(p[65u], UINT(std::floor(secondary + .5f))); EXPECT_EQ(p[66u], peak);
  }
  TEST_F(HostSbsSubtitleInnerColourGpuTest, HuePrunesDimmerGreyButRetainsDistinctCyan) {
    author(1u, true); bar(40u); bar(80u, {.55f, .55f, .55f}); bar(120u, {.05f, .80f, .90f}); ASSERT_TRUE(run()); expect_uploaded_exact(); expect_args_exact(); expect_gpu_or_exact();
    const auto &p = units_[0u].palette; ASSERT_EQ(p[4u], 2u); EXPECT_EQ(p[68u], 63u); EXPECT_EQ(p[76u], 15u);
    EXPECT_EQ(body_[pixel(81u, 12u)], 1u); EXPECT_EQ(body_[pixel(121u, 12u)], 1u);
  }
  TEST_F(HostSbsSubtitleInnerColourGpuTest, CurrentSupportAndStrongProbabilityAreAlternativeEligibleSeedWitnesses) {
    author(1u, true); bar(50u, {.95f, .95f, .95f}, false); ASSERT_TRUE(run()); expect_args_exact(); expect_gpu_or_exact(); EXPECT_EQ(body_[pixel(51u, 12u)], 1u);
    author(2u, true); bar(50u, {.95f, .95f, .95f}, false); bar(100u); native_probability({90u, 0u, 140u, crop_height_}, .01f);
    ASSERT_TRUE(run()); expect_uploaded_exact(); expect_args_exact(); expect_gpu_or_exact(); EXPECT_EQ(body_[pixel(101u, 12u)], 1u);
    author(3u, true); bar(50u, {.95f, .95f, .95f}, false); bar(100u, {.95f, .95f, .95f}, false); native_probability({90u, 0u, 140u, crop_height_}, .01f);
    ASSERT_TRUE(run()); expect_args_exact(); expect_gpu_or_exact(); EXPECT_EQ(body_[pixel(101u, 12u)], 0u); EXPECT_EQ(final_[pixel(101u, 12u)], 0u);
  }
  TEST_F(HostSbsSubtitleInnerColourGpuTest, EligibleDimmerFillDoesNotRestoreAbsoluteRgbMatching) {
    author(1u, true); bar(50u); bar(100u, {.55f, .55f, .55f}, false); ASSERT_TRUE(run()); expect_args_exact(); expect_gpu_or_exact();
    ASSERT_EQ(units_[0u].palette[4u], 1u); const auto e = colour_words(modes_.at({0u, 0u}).planes.at("eligible.u32"));
    EXPECT_EQ(e[pixel(101u, 12u)], 1u); EXPECT_EQ(body_[pixel(101u, 12u)], 1u);
  }
  TEST_F(HostSbsSubtitleInnerColourGpuTest, ChromaticOutsideContrastCannotSupplyBlackWitness) {
    author(1u, true, .5f); bar(50u, {.95f, .95f, .95f}, true, {.35f, 0.f, 0.f}); ASSERT_TRUE(run()); expect_uploaded_exact(); expect_args_exact(); expect_gpu_or_exact();
    ASSERT_GT(units_[0u].palette[4u], 0u); const auto seeds = colour_words(modes_.at({0u, 0u}).planes.at("seeds.u32"));
    EXPECT_EQ(seeds[pixel(51u, 12u)], 0u); EXPECT_EQ(body_[pixel(51u, 12u)], 0u); EXPECT_EQ(std::count(body_.begin(), body_.end(), 1u), 0);
  }
  TEST_F(HostSbsSubtitleInnerColourGpuTest, SearchBoundaryComponentPublishesNeitherInteriorNorItsSeeds) {
    author(1u, true); bar(50u); paint({0u, 8u, 53u, 10u}, {.95f, .95f, .95f}, true); ASSERT_TRUE(run()); expect_args_exact(); expect_gpu_or_exact();
    ASSERT_GT(units_[0u].palette[4u], 0u); const auto seeds = colour_words(modes_.at({0u, 0u}).planes.at("seeds.u32"));
    EXPECT_GT(std::count(seeds.begin(), seeds.end(), 1u), 0); EXPECT_EQ(std::count(body_.begin(), body_.end(), 1u), 0);
  }
  TEST_F(HostSbsSubtitleInnerColourGpuTest, StalePrefixReservedOrAnyGlobalNumericFaultFailsClosed) {
    for (UINT fault = 0u; fault < 10u; ++fault) {
      SCOPED_TRACE(fault); author(1u, true); bar(50u);
      switch (fault) {
        case 0u: ocr_[1u] ^= 1u; break;
        case 1u: ++ocr_[5u]; break;
        case 2u: constants_.words[19u] = 1u; break;
        case 3u: request_reserved_ = 1u; break;
        case 4u: support_.back() = 2u; break;
        case 5u: probability_.back() = NAN; break;
        case 6u: source_.front()[3u] = colour_half(0.f); break;
        case 7u: ocr_[3u] = ocr_[4u] = 9u; break;
        case 8u: constants_.words[0u] = 0u; break;
        case 9u: constants_.words[6u] = constants_.words[7u]; break;
      }
      ASSERT_TRUE(run()); expect_uploaded_exact(); expect_args_exact(); expect_gpu_or_exact();
      EXPECT_EQ(units_[0u].palette[2u], 0u); EXPECT_EQ(std::count(body_.begin(), body_.end(), 1u), 0); EXPECT_EQ(std::count(final_.begin(), final_.end(), 1u), 0);
    }
  }

  TEST_F(HostSbsSubtitleInnerColourGpuTest, SavedNativeInnerColourReplay) {
    const auto *input_env = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_CASES"), *output_env = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_OUTPUT");
    if (!input_env || !output_env) { GTEST_SKIP() << "Explicit full inner-colour source replay paths required"; }
    const auto input_bytes = colour_bytes(input_env); const auto spec = nlohmann::json::parse(input_bytes.begin(), input_bytes.end());
    ASSERT_EQ(spec.at("schema"), 1u); const auto &cases = spec.at("cases"); ASSERT_GT(cases.size(), 0u); ASSERT_LE(cases.size(), 64u);
    const std::filesystem::path destination(output_env); ASSERT_FALSE(std::filesystem::exists(destination)); ASSERT_TRUE(std::filesystem::create_directories(destination));
    const auto input_hash = colour_sha(input_bytes.data(), input_bytes.size());
    std::array<wchar_t, 32768u> exe_path {}; const auto exe_length = GetModuleFileNameW(nullptr, exe_path.data(), DWORD(exe_path.size())); nlohmann::json executable = nullptr;
    if (exe_length > 0u && exe_length < exe_path.size()) {
      const std::filesystem::path p(std::wstring(exe_path.data(), exe_length)); const auto b = colour_bytes(p); executable = {{"path", p.string()}, {"sha256", colour_sha(b.data(), b.size())}, {"bytes", b.size()}};
    }
    nlohmann::json report {{"schema", 1u}, {"driver", driver_}, {"adapter", adapter_identity()}, {"inner_source_closure_sha256", inner_closure_},
      {"component_source_closure_sha256", component_closure_}, {"input_specification_sha256", input_hash}, {"test_executable", executable}, {"cases", nlohmann::json::array()}};
    for (std::size_t index = 0u; index < cases.size(); ++index) {
      const auto &item = cases.at(index); SCOPED_TRACE(item.at("name").get<std::string>()); const auto &a = item.at("artifacts");
      const auto cb = colour_words(colour_bound(a.at("constants"), 96u)); ASSERT_TRUE(resize(cb[0u], cb[1u], cb[2u], cb[3u]));
      std::copy(cb.begin(), cb.end(), constants_.words.begin()); const auto n = std::size_t(width_) * crop_height_;
      auto raw = colour_bound(a.at("source_band"), n * 8u); std::memcpy(source_.data(), raw.data(), raw.size());
      raw = colour_bound(a.at("probability"), 153600u * 4u); std::memcpy(probability_.data(), raw.data(), raw.size());
      raw = colour_bound(a.at("current_support"), n * 4u); std::memcpy(support_.data(), raw.data(), raw.size());
      ocr_ = colour_words(colour_bound(a.at("ocr_record"), 1024u)); requested_ = {};
      const bool has_expected = item.contains("expected");
      if (has_expected) {
        for (const auto &u : item.at("expected").at("units")) {
          const auto slot = u.at("slot").get<UINT>(); ASSERT_LT(slot, 8u);
          for (const auto &m : u.at("modes")) { const auto mode = m.at("mode").get<UINT>(); ASSERT_LT(mode, 8u); requested_[slot][mode] = true; }
        }
      } else { requested_[0u].fill(true); }
      ASSERT_TRUE(run());
      const bool uploaded_exact = uploaded_byte_exact(), arguments_exact = arguments_byte_exact(), gpu_union_exact = gpu_or_byte_exact();
      EXPECT_TRUE(uploaded_exact); EXPECT_TRUE(arguments_exact); EXPECT_TRUE(gpu_union_exact);
      bool expected_exact = true; std::array<bool, 8u> expected_palette_supplied {}, expected_args_supplied {}, expected_palette_exact {}, expected_args_exact {};
      if (has_expected) {
        const auto &e = item.at("expected");
        expected_exact &= colour_raw(body_) == colour_bound(e.at("body"), n * 4u); expected_exact &= colour_raw(final_) == colour_bound(e.at("final"), n * 4u);
        for (const auto &u : e.at("units")) {
          const UINT slot = u.at("slot").get<UINT>(); expected_palette_supplied[slot] = true;
          expected_palette_exact[slot] = colour_raw(units_[slot].palette) == colour_bound(u.at("palette"), 512u); expected_exact &= expected_palette_exact[slot];
          if (u.contains("arguments")) { expected_args_supplied[slot] = true; expected_args_exact[slot] = colour_raw(units_[slot].args) == colour_bound(u.at("arguments"), 96u); expected_exact &= expected_args_exact[slot]; }
          for (const auto &m : u.at("modes")) {
            const UINT mode = m.at("mode").get<UINT>(); const auto &native = modes_.at({slot, mode});
            if (!native.dispatched) continue; // zero args prove no contribution; scratch is held, not newly zero.
            for (const auto &[key, file] : std::array<std::pair<const char *, const char *>, 6u> {{{"eligible", "eligible.u32"}, {"seeds", "seeds.u32"}, {"domain_tag", "domain_tag.u32"}, {"cover_clip", "cover_clip.u32"}, {"body", "body.u32"}, {"final", "final.u32"}}}) {
              expected_exact &= native.planes.at(file) == colour_bound(m.at(key), n * 4u);
            }
          }
        }
      }
      const auto folder = destination / ("case_" + std::to_string(index)); ASSERT_TRUE(std::filesystem::create_directory(folder));
      const auto save = [&](const std::filesystem::path &p, const std::vector<char> &bytes) {
        std::ofstream f(p, std::ios::binary); f.write(bytes.data(), std::streamsize(bytes.size())); if (!f) throw std::runtime_error("Inner-colour export failed");
        return nlohmann::json {{"byte_count", bytes.size()}, {"sha256", colour_sha(bytes.data(), bytes.size())}};
      };
      nlohmann::json saved = nlohmann::json::object(); for (const auto &[name, bytes] : uploaded_) saved[name] = save(folder / name, bytes);
      saved["body.u32"] = save(folder / "body.u32", colour_raw(body_)); saved["final.u32"] = save(folder / "final.u32", colour_raw(final_));
      nlohmann::json unit_rows = nlohmann::json::array();
      for (UINT slot = 0u; slot < 8u; ++slot) {
        const auto unit_folder = folder / ("unit_" + std::to_string(slot)); ASSERT_TRUE(std::filesystem::create_directory(unit_folder));
        nlohmann::json u {{"slot", slot}, {"palette", save(unit_folder / "palette.u32", colour_raw(units_[slot].palette))},
          {"arguments", save(unit_folder / "mode_arguments.u32", colour_raw(units_[slot].args))}, {"mode_count", units_[slot].palette[4u]},
          {"status", units_[slot].palette[22u]}, {"current_valid", units_[slot].palette[2u]}, {"modes", nlohmann::json::array()},
          {"expected_palette_byte_exact", expected_palette_supplied[slot] ? nlohmann::json(expected_palette_exact[slot]) : nlohmann::json(nullptr)},
          {"expected_arguments_byte_exact", expected_args_supplied[slot] ? nlohmann::json(expected_args_exact[slot]) : nlohmann::json(nullptr)}};
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const auto &args = units_[slot].args; const bool active = args[3u * mode] != 0u || args[3u * mode + 1u] != 0u || args[3u * mode + 2u] != 0u;
          nlohmann::json m {{"mode", mode}, {"dispatched", active}, {"arguments", {args[3u * mode], args[3u * mode + 1u], args[3u * mode + 2u]}},
            {"diagnostics_requested", requested_[slot][mode]}, {"inactive_role", active ? "current mode publication" : "zero indirect groups: raw scratch held, no accumulation or new intermediate publication"}};
          if (active && requested_[slot][mode]) {
            const auto mode_folder = unit_folder / ("mode_" + std::to_string(mode)); ASSERT_TRUE(std::filesystem::create_directory(mode_folder));
            m["artifacts"] = nlohmann::json::object(); for (const auto &[name, bytes] : modes_.at({slot, mode}).planes) m["artifacts"][name] = save(mode_folder / name, bytes);
          }
          u["modes"].push_back(m);
        }
        unit_rows.push_back(u);
      }
      nlohmann::json identity {{"schema", 1u}, {"case", item}, {"driver", driver_}, {"adapter", adapter_identity()}, {"inner_shader_root", inner_root_.string()},
        {"component_shader_root", component_root_.string()}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", component_closure_},
        {"input_specification_sha256", input_hash}, {"test_executable", executable}, {"artifacts", saved}, {"units", unit_rows},
        {"expected_source_pipeline_supplied", has_expected}, {"expected_source_pipeline_byte_exact", has_expected ? nlohmann::json(expected_exact) : nlohmann::json(nullptr)},
        {"uploaded_five_native_operands_byte_exact", uploaded_exact}, {"all_mode_dispatch_arguments_exact", arguments_exact}, {"GPU_OR_matches_literal_component_readbacks", gpu_union_exact}, {"diagnostic_capture_complete", capture_complete_},
        {"dispatch_policy", "GPU clear once; fixed8unit palette groups and fixed8mode commandloops/unit. Emit+all5CC+accumulate use identicalGPU-written triple offset12*mode. Palette/args map only after all source commands. Expected/capture plan controls copies/export only, never shader acceptance or commands."},
        {"scope", "Complete frozenV5 current-input colour/palette/seed/component/BOX3/OR candidate, with exact CPU-sidecar comparisons. No temporal age, historical draw authority, full producer authentication, source classification model execution, runtime draw integration or speed proof. Source tuple provenance remains external."}};
      const auto text = identity.dump(2) + "\n"; std::ofstream f(folder / "inner_colour_fixture_identity.json", std::ios::binary); f << text; ASSERT_TRUE(f.good()); f.close();
      report["cases"].push_back({{"index", index}, {"name", item.at("name")}, {"identity_sha256", colour_sha(text.data(), text.size())}, {"expected_source_pipeline_byte_exact", identity["expected_source_pipeline_byte_exact"]}});
      std::ofstream aggregate(destination / "inner_colour_replay_identity.json", std::ios::binary | std::ios::trunc); aggregate << report.dump(2) << '\n'; ASSERT_TRUE(aggregate.good());
      EXPECT_TRUE(expected_exact) << "Exact frozenV5 palette/emission/body/final sidecar mismatch; outputs preserved for diagnosis";
    }
  }
}
#endif
