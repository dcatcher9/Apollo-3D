/**
 * @file tests/unit/test_host_sbs_subtitle_matte_guard_gpu.cpp
 * @brief Private final-matte guard contract/replay; no production drawing authority.
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
  #include <chrono>
  #include <cstdint>
  #include <cstdlib>
  #include <cstring>
  #include <d3d11.h>
  #include <dxgi.h>
  #include <filesystem>
  #include <fstream>
  #include <limits>
  #include <nlohmann/json.hpp>
  #include <stdexcept>
  #include <string>
  #include <string_view>
  #include <thread>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace cv2 = models::depth_coordinate_v2;
  namespace cohort = host_sbs_subtitle_cohort;
  using guard_rect_t = std::array<UINT, 4u>;

  std::vector<char> guard_bytes(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0) throw std::runtime_error("Cannot read guard operand: " + path.string());
    std::vector<char> bytes(static_cast<std::size_t>(file.tellg()));
    file.seekg(0); if (!bytes.empty()) file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("Incomplete guard operand: " + path.string());
    return bytes;
  }

  std::string guard_sha(const void *data, std::size_t size) {
    const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), size));
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (auto byte : digest) { result.push_back(digits[byte >> 4u]); result.push_back(digits[byte & 15u]); }
    return result;
  }

  std::vector<UINT> guard_bound_words(const nlohmann::json &binding, std::size_t words) {
    const auto bytes = guard_bytes(binding.at("path").get<std::string>());
    if (bytes.size() != words * sizeof(UINT) || bytes.size() != binding.at("bytes").get<std::size_t>() ||
        guard_sha(bytes.data(), bytes.size()) != binding.at("sha256").get<std::string>())
      throw std::runtime_error("Guard operand binding mismatch");
    std::vector<UINT> result(words); std::memcpy(result.data(), bytes.data(), bytes.size()); return result;
  }

  struct guard_texture_t {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };
  struct guard_buffer_t {
    ComPtr<ID3D11Buffer> buffer;
    ComPtr<ID3D11ShaderResourceView> srv;
  };

  bool guard_texture(ID3D11Device *device, UINT width, UINT height, bool output, guard_texture_t &out) {
    out = {}; D3D11_TEXTURE2D_DESC d {};
    d.Width = width; d.Height = height; d.MipLevels = d.ArraySize = 1u;
    d.Format = DXGI_FORMAT_R32_UINT; d.SampleDesc.Count = 1u; d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | (output ? D3D11_BIND_UNORDERED_ACCESS : 0u);
    return SUCCEEDED(device->CreateTexture2D(&d, nullptr, &out.texture)) &&
      SUCCEEDED(device->CreateShaderResourceView(out.texture.Get(), nullptr, &out.srv)) &&
      (!output || SUCCEEDED(device->CreateUnorderedAccessView(out.texture.Get(), nullptr, &out.uav)));
  }

  bool guard_buffer(ID3D11Device *device, UINT words, guard_buffer_t &out) {
    out = {}; D3D11_BUFFER_DESC d {};
    d.ByteWidth = words * 4u; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; d.StructureByteStride = 4u;
    return SUCCEEDED(device->CreateBuffer(&d, nullptr, &out.buffer)) &&
      SUCCEEDED(device->CreateShaderResourceView(out.buffer.Get(), nullptr, &out.srv));
  }

  bool guard_readback(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Texture2D *input, std::vector<UINT> &out) {
    D3D11_TEXTURE2D_DESC d {}; input->GetDesc(&d);
    const auto width = d.Width, height = d.Height;
    d.Usage = D3D11_USAGE_STAGING; d.BindFlags = 0u; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ; d.MiscFlags = 0u;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&d, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), input);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
    out.resize(std::size_t(width) * height);
    for (UINT y = 0u; y < height; ++y)
      std::memcpy(out.data() + std::size_t(y) * width, static_cast<const char *>(mapped.pData) + std::size_t(y) * mapped.RowPitch, width * 4u);
    context->Unmap(staging.Get(), 0u); return true;
  }

  class HostSbsSubtitleMatteGuardGpuTest: public ::testing::Test {
  protected:
    void SetUp() override {
      shader_root_ = SUNSHINE_SHADERS_DIR;
      std::string requested_pin;
      if (std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) == "SavedNativeMatteGuardReplay") {
        if (const auto *path = std::getenv("SUNSHINE_TEST_MATTE_GUARD_CASES")) {
          const auto bytes = guard_bytes(path); const auto spec = nlohmann::json::parse(bytes.begin(), bytes.end());
          driver_ = spec.value("driver", std::string("warp"));
          shader_root_ = std::filesystem::path(path).parent_path() / spec.at("shader_root").get<std::string>();
          requested_pin = spec.at("source_closure_sha256").get<std::string>();
        }
      }
      ASSERT_TRUE(driver_ == "warp" || driver_ == "hardware");
      const D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0}; D3D_FEATURE_LEVEL actual {};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, driver_ == "hardware" ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP,
        nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &device_, &actual, &context_)));
      ASSERT_EQ(actual, D3D_FEATURE_LEVEL_11_0);
      namespace cache = models::host_sbs_shader_cache;
      const auto sources = cache::snapshot_sources(shader_root_, cache::subtitle_matte_guard_research_specs);
      closure_ = cache::source_closure_sha256(sources);
      ASSERT_EQ(cache::subtitle_matte_guard_research_specs.size(), 1u);
      ASSERT_EQ(closure_, cache::subtitle_matte_guard_research_source_closure_sha256);
      if (!requested_pin.empty()) { ASSERT_EQ(closure_, requested_pin); }
      const auto code = cache::get(sources, cache::subtitle_matte_guard_research_specs[0u]);
      ASSERT_TRUE(code && !code->empty());
      ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &shader_)));
      D3D11_BUFFER_DESC cb {}; cb.ByteWidth = sizeof(cohort::constants_t); cb.Usage = D3D11_USAGE_DEFAULT; cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&cb, nullptr, &constant_buffer_)));
      cb.ByteWidth = 12u; cb.BindFlags = 0u; cb.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&cb, nullptr, &arguments_)));
      ASSERT_TRUE(allocate(160u, 90u));
    }

    void TearDown() override { unbind(); }
    void unbind() {
      if (!context_) return;
      const std::array<ID3D11ShaderResourceView *, 7u> reads {}; ID3D11UnorderedAccessView *write = nullptr;
      context_->CSSetShaderResources(0u, static_cast<UINT>(reads.size()), reads.data());
      context_->CSSetUnorderedAccessViews(0u, 1u, &write, nullptr);
    }

    bool allocate(UINT width, UINT height, guard_rect_t content = {}, UINT field_width = 0u, UINT field_height = 0u) {
      unbind(); width_ = width; height_ = height; crop_height_ = std::min(height, (width + 5u) / 6u); crop_top_ = height - crop_height_;
      field_width_ = field_width ? field_width : cv2::model_calibrated_shapes[0u].width;
      field_height_ = field_height ? field_height : cv2::model_calibrated_shapes[0u].height;
      content_ = content == guard_rect_t {} ? guard_rect_t {0u, 0u, field_width_, field_height_} : content;
      return guard_texture(device_.Get(), width_, crop_height_, false, raw_) &&
        guard_texture(device_.Get(), width_, crop_height_, false, support_) &&
        guard_texture(device_.Get(), width_, crop_height_, false, owner_) &&
        guard_texture(device_.Get(), width_, crop_height_, true, output_) &&
        guard_buffer(device_.Get(), SBC_STATE_WORDS, state_) && guard_buffer(device_.Get(), cv2::subtitle_ocr_record_word_count, ocr_) &&
        guard_buffer(device_.Get(), cv2::subtitle_plane_state_word_count, uip_) && guard_buffer(device_.Get(), cv2::subtitle_contour_history_word_count, sct_);
    }

    UINT row_edge(UINT detector_y) const {
      const auto n = std::uint64_t(crop_top_ * cv2::subtitle_ocr_output_height + detector_y * crop_height_) * (content_[3u] - content_[1u]);
      const auto d = std::uint64_t(height_) * cv2::subtitle_ocr_output_height;
      return content_[1u] + static_cast<UINT>((n + d - 1u) / d);
    }
    guard_rect_t field_rect(guard_rect_t native) const {
      const auto cw = content_[2u] - content_[0u], ch = content_[3u] - content_[1u];
      return {content_[0u] + UINT(std::uint64_t(native[0u]) * cw / width_), content_[1u] + UINT(std::uint64_t(crop_top_ + native[1u]) * ch / height_),
        content_[0u] + UINT((std::uint64_t(native[2u]) * cw + width_ - 1u) / width_), content_[1u] + UINT((std::uint64_t(crop_top_ + native[3u]) * ch + height_ - 1u) / height_)};
    }
    std::size_t pixel(UINT x, UINT y) const { return std::size_t(y) * width_ + x; }
    bool in_cover(UINT x, UINT y, const guard_rect_t &r) const {
      const auto nx = 2ull * content_[0u] * width_ + (2ull * x + 1u) * (content_[2u] - content_[0u]);
      const auto ny = 2ull * content_[1u] * height_ + (2ull * (y + crop_top_) + 1u) * (content_[3u] - content_[1u]);
      return nx >= 2ull * r[0u] * width_ && nx < 2ull * r[2u] * width_ && ny >= 2ull * r[1u] * height_ && ny < 2ull * r[3u] * height_;
    }

    void set_rect(UINT slot, guard_rect_t r) {
      for (auto offset : {cv2::subtitle_ocr_raw_box_offset, cv2::subtitle_ocr_final_box_offset}) {
        const auto a = offset + slot * cv2::subtitle_ocr_box_word_count;
        std::copy(r.begin(), r.end(), ocr_words_.begin() + a);
        ocr_words_[a + 4u] = std::bit_cast<UINT>(.9f); ocr_words_[a + 6u] = 1u;
      }
      const auto b = cohort::entry_offset(slot);
      for (auto offset : {SBC_E_BIRTH_CORE, SBC_E_BIRTH_COVER, SBC_E_LAST_CORE, SBC_E_LAST_COVER})
        std::copy(r.begin(), r.end(), state_words_.begin() + b + offset);
    }

    void author(const std::vector<guard_rect_t> &native_covers = {{20u, 6u, 140u, 26u}}, UINT phase = SBC_PHASE_ACTIVE) {
      constants_ = {{width_, height_, crop_top_, crop_height_}, {field_width_, field_height_, row_edge(cv2::subtitle_ocr_safe_row_top), row_edge(cv2::subtitle_ocr_safe_row_bottom)},
        content_, {5u, 0u, 7u, 0u}, {1166665u, 0u, 0u, 0u}, {1u, SBC_SCHEMA, 0u, 0u}};
      raw_values_.assign(std::size_t(width_) * crop_height_, 1u); support_values_.assign(raw_values_.size(), 0u); owner_values_.assign(raw_values_.size(), 0u);
      state_words_.assign(SBC_STATE_WORDS, 0u); ocr_words_.assign(cv2::subtitle_ocr_record_word_count, 0u);
      uip_words_.assign(cv2::subtitle_plane_state_word_count, 0u); sct_words_.assign(cv2::subtitle_contour_history_word_count, 0u);
      const UINT count = static_cast<UINT>(native_covers.size());
      ocr_words_[0u] = cv2::subtitle_ocr_record_schema; ocr_words_[1u] = cv2::subtitle_ocr_record_tag; ocr_words_[2u] = 1u; ocr_words_[3u] = ocr_words_[4u] = count;
      std::copy(constants_.frame.begin(), constants_.frame.end(), ocr_words_.begin() + 5u);
      ocr_words_[9u] = width_; ocr_words_[10u] = height_; ocr_words_[11u] = field_width_; ocr_words_[12u] = field_height_;
      ocr_words_[13u] = constants_.field[2u]; ocr_words_[14u] = constants_.field[3u];
      uip_words_[0u] = cv2::subtitle_plane_state_schema; uip_words_[1u] = cv2::subtitle_plane_state_tag;
      uip_words_[2u] = cv2::subtitle_plane_current_valid_flag; uip_words_[3u] = (1u << count) - 1u; uip_words_[20u] = 3u;
      for (UINT k = 0u; k < 2u; ++k) {
        uip_words_[4u + k] = constants_.frame[k]; uip_words_[6u + k] = constants_.frame[2u + k]; uip_words_[8u + k] = constants_.observation[k];
        uip_words_[10u + k] = constants_.source[k]; uip_words_[12u + k] = constants_.field[k]; uip_words_[14u + k] = constants_.field[2u + k];
      }
      std::copy(content_.begin(), content_.end(), uip_words_.begin() + 16u);
      sct_words_[0u] = cv2::subtitle_contour_schema; sct_words_[1u] = cv2::subtitle_contour_history_tag; sct_words_[2u] = count ? 1u : 0u; sct_words_[3u] = 3u;
      std::copy(constants_.frame.begin(), constants_.frame.end(), sct_words_.begin() + 4u); std::copy(constants_.source.begin(), constants_.source.end(), sct_words_.begin() + 8u);
      std::copy(constants_.field.begin(), constants_.field.end(), sct_words_.begin() + 12u); std::copy(content_.begin(), content_.end(), sct_words_.begin() + 16u);
      sct_words_[20u] = constants_.observation[0u]; sct_words_[21u] = constants_.observation[1u]; sct_words_[22u] = cv2::subtitle_contour_recipe; sct_words_[23u] = 1u;
      state_words_[SBC_H_TAG] = SBC_TAG; state_words_[SBC_H_SCHEMA] = SBC_SCHEMA; state_words_[SBC_H_VALID] = 1u; state_words_[SBC_H_CUT] = 3u;
      for (UINT k = 0u; k < 2u; ++k) {
        state_words_[SBC_H_FRAME + k] = constants_.frame[k]; state_words_[SBC_H_GENERATION + k] = constants_.frame[2u + k];
        state_words_[SBC_H_CLOCK + k] = constants_.observation[k]; state_words_[SBC_H_SOURCE + k] = constants_.source[k]; state_words_[SBC_H_CROP + k] = constants_.source[2u + k];
        state_words_[SBC_H_FIELD + k] = constants_.field[k];
      }
      std::copy(content_.begin(), content_.end(), state_words_.begin() + SBC_H_CONTENT);
      state_words_[SBC_H_REFERENCE] = width_; state_words_[SBC_H_REFERENCE + 1u] = width_; state_words_[SBC_H_NEXT_ID] = 7u + count;
      state_words_[SBC_H_EPOCH] = 1u; state_words_[SBC_H_COUNT] = count; state_words_[SBC_H_FLAGS] = SBC_HEADER_P_MAP_VALID;
      for (UINT slot = 0u; slot < count; ++slot) {
        auto r = field_rect(native_covers[slot]); r[1u] = std::max(r[1u], constants_.field[2u]); r[3u] = std::min(r[3u], constants_.field[3u]);
        set_rect(slot, r); set_phase(slot, phase);
      }
    }

    void set_phase(UINT slot, UINT phase) {
      const auto b = cohort::entry_offset(slot);
      state_words_[b + SBC_E_ID] = phase == SBC_PHASE_UNKNOWN ? 0u : 7u + slot; state_words_[b + SBC_E_PHASE] = phase;
      state_words_[b + SBC_E_FLAGS] = phase == SBC_PHASE_SUSPENDED ? SBC_ENTRY_UNCERTAIN_REPLACEMENT : phase == SBC_PHASE_PENDING ? SBC_ENTRY_NEW_BIRTH : 0u;
      state_words_[b + SBC_E_PREVIOUS_SLOT] = SBC_INVALID_SLOT;
      state_words_[b + SBC_E_BIRTH_FRAME] = 2u; state_words_[b + SBC_E_BIRTH_CLOCK] = 1066666u;
      state_words_[b + SBC_E_REFERENCE_FRAME] = 1u; state_words_[b + SBC_E_REFERENCE_FLAGS] = SBC_REFERENCE_KNOWN_FLAGS;
      if (phase == SBC_PHASE_UNKNOWN) {
        for (UINT k = SBC_E_BIRTH_CORE; k < SBC_E_LAST_CORE; ++k) state_words_[b + k] = 0u;
        state_words_[b + SBC_E_BIRTH_FRAME] = state_words_[b + SBC_E_BIRTH_CLOCK] = 0u;
        state_words_[b + SBC_E_REFERENCE_FRAME] = state_words_[b + SBC_E_REFERENCE_FLAGS] = 0u;
      }
    }

    void bind_dispatch() {
      const std::array<ID3D11ShaderResourceView *, 7u> reads {raw_.srv.Get(), support_.srv.Get(), owner_.srv.Get(), state_.srv.Get(), ocr_.srv.Get(), uip_.srv.Get(), sct_.srv.Get()};
      context_->CSSetShaderResources(0u, static_cast<UINT>(reads.size()), reads.data());
      context_->CSSetUnorderedAccessViews(0u, 1u, output_.uav.GetAddressOf(), nullptr);
      context_->CSSetConstantBuffers(0u, 1u, constant_buffer_.GetAddressOf()); context_->CSSetShader(shader_.Get(), nullptr, 0u);
      context_->DispatchIndirect(arguments_.Get(), 0u);
    }
    bool run() {
      unbind(); context_->UpdateSubresource(raw_.texture.Get(), 0u, nullptr, raw_values_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(support_.texture.Get(), 0u, nullptr, support_values_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(owner_.texture.Get(), 0u, nullptr, owner_values_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(state_.buffer.Get(), 0u, nullptr, state_words_.data(), 0u, 0u);
      context_->UpdateSubresource(ocr_.buffer.Get(), 0u, nullptr, ocr_words_.data(), 0u, 0u);
      context_->UpdateSubresource(uip_.buffer.Get(), 0u, nullptr, uip_words_.data(), 0u, 0u);
      context_->UpdateSubresource(sct_.buffer.Get(), 0u, nullptr, sct_words_.data(), 0u, 0u);
      context_->UpdateSubresource(constant_buffer_.Get(), 0u, nullptr, &constants_, 0u, 0u);
      const std::array<UINT, 3u> grid {(field_width_ + 15u) / 16u, (field_height_ + 15u) / 16u, 1u};
      context_->UpdateSubresource(arguments_.Get(), 0u, nullptr, grid.data(), 0u, 0u); bind_dispatch(); unbind();
      return guard_readback(device_.Get(), context_.Get(), output_.texture.Get(), result_);
    }
    void expect_subset() {
      ASSERT_EQ(result_.size(), raw_values_.size());
      for (std::size_t p = 0u; p < result_.size(); ++p) { ASSERT_LE(result_[p], 1u) << p; ASSERT_LE(result_[p], raw_values_[p]) << p; }
    }
    void expect_cover_clip() {
      const auto count = ocr_words_[4u];
      for (UINT y = 0u; y < crop_height_; ++y) for (UINT x = 0u; x < width_; ++x) {
        bool inside = false;
        for (UINT slot = 0u; slot < count; ++slot) {
          guard_rect_t r {}; std::copy_n(ocr_words_.begin() + cv2::subtitle_ocr_final_box_offset + slot * cv2::subtitle_ocr_box_word_count, 4u, r.begin());
          inside = inside || in_cover(x, y, r);
        }
        ASSERT_EQ(result_[pixel(x, y)], inside ? raw_values_[pixel(x, y)] : 0u) << x << ',' << y;
      }
    }

    nlohmann::json gpu_timestamps() {
      // Explicit replay-only opt-in. Uploads/readbacks are outside these intervals.
      constexpr UINT warmup = 3u, repeats = 20u;
      unbind(); for (UINT i = 0u; i < warmup; ++i) bind_dispatch();
      D3D11_QUERY_DESC d {D3D11_QUERY_TIMESTAMP_DISJOINT, 0u}; ComPtr<ID3D11Query> disjoint;
      if (FAILED(device_->CreateQuery(&d, &disjoint))) throw std::runtime_error("Cannot create guard timestamp disjoint query");
      std::array<ComPtr<ID3D11Query>, repeats * 2u> queries;
      d.Query = D3D11_QUERY_TIMESTAMP;
      for (auto &q : queries) if (FAILED(device_->CreateQuery(&d, &q))) throw std::runtime_error("Cannot create guard timestamp query");
      context_->Begin(disjoint.Get());
      for (UINT i = 0u; i < repeats; ++i) { context_->End(queries[2u * i].Get()); bind_dispatch(); context_->End(queries[2u * i + 1u].Get()); }
      context_->End(disjoint.Get()); unbind(); context_->Flush();
      const auto wait = [&](ID3D11Query *q, void *value, UINT bytes) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        HRESULT status;
        while ((status = context_->GetData(q, value, bytes, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE) {
          if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Guard timestamp query timed out");
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (FAILED(status)) throw std::runtime_error("Guard timestamp query failed");
      };
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT evidence {}; wait(disjoint.Get(), &evidence, sizeof(evidence));
      nlohmann::json result { {"enabled", true}, {"warmup", warmup}, {"repeats", repeats}, {"disjoint", evidence.Disjoint != FALSE},
        {"frequency_hz", evidence.Frequency}, {"scope", "Guard dispatch GPU timestamp interval only; no upload/readback, pipeline throughput, or CPU walltime claim"} };
      nlohmann::json ticks = nlohmann::json::array(), ns = nlohmann::json::array();
      for (UINT i = 0u; i < repeats; ++i) {
        UINT64 start {}, end {}; wait(queries[2u * i].Get(), &start, sizeof(start)); wait(queries[2u * i + 1u].Get(), &end, sizeof(end));
        ticks.push_back({start, end});
        if (!evidence.Disjoint && evidence.Frequency != 0u && end >= start) ns.push_back(double(end - start) * 1e9 / double(evidence.Frequency));
      }
      result["timestamp_ticks"] = ticks; result["gpu_ns"] = ns;
      if (ns.size() == repeats) {
        auto values = ns.get<std::vector<double>>(); std::sort(values.begin(), values.end());
        double sum = 0.0; for (auto value : values) sum += value;
        result["gpu_ns_summary"] = {{"minimum", values.front()}, {"median", .5 * (values[9u] + values[10u])}, {"mean", sum / repeats}, {"maximum", values.back()}};
      }
      return result;
    }

    nlohmann::json adapter_identity() const {
      ComPtr<IDXGIDevice> dxgi; ComPtr<IDXGIAdapter> adapter; DXGI_ADAPTER_DESC d {};
      if (FAILED(device_.As(&dxgi)) || FAILED(dxgi->GetAdapter(&adapter)) || FAILED(adapter->GetDesc(&d))) return {{"available", false}};
      std::string name; for (const auto ch : d.Description) { if (ch == 0) break; name.push_back(ch <= 127 ? char(ch) : '?'); }
      return {{"description", name}, {"vendor_id", d.VendorId}, {"device_id", d.DeviceId}, {"subsystem_id", d.SubSysId},
        {"revision", d.Revision}, {"adapter_luid_low", d.AdapterLuid.LowPart}, {"adapter_luid_high", d.AdapterLuid.HighPart}};
    }

    ComPtr<ID3D11Device> device_; ComPtr<ID3D11DeviceContext> context_; ComPtr<ID3D11ComputeShader> shader_;
    ComPtr<ID3D11Buffer> constant_buffer_, arguments_;
    guard_texture_t raw_, support_, owner_, output_; guard_buffer_t state_, ocr_, uip_, sct_;
    cohort::constants_t constants_ {};
    std::vector<UINT> raw_values_, support_values_, owner_values_, state_words_, ocr_words_, uip_words_, sct_words_, result_;
    UINT width_ {}, height_ {}, crop_top_ {}, crop_height_ {}, field_width_ {}, field_height_ {};
    guard_rect_t content_ {}; std::filesystem::path shader_root_; std::string closure_, driver_ {"warp"};
  };

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, KnownLifetimeWithCurrentSupportZeroGetsExactD1) {
    author(); owner_values_[pixel(80u, 16u)] = 7u; ASSERT_TRUE(run()); expect_subset();
    EXPECT_EQ(std::count(result_.begin(), result_.end(), 1u), 9);
    EXPECT_EQ(result_[pixel(81u, 17u)], 1u); EXPECT_EQ(result_[pixel(82u, 16u)], 0u);
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, UnknownCurrentSupportGetsExactK157NotSquareRadiusSix) {
    author(); owner_values_[pixel(80u, 16u)] = SBC_OWNER_UNKNOWN_BIT | 7u; support_values_[pixel(80u, 16u)] = 1u;
    ASSERT_TRUE(run()); expect_subset(); EXPECT_EQ(std::count(result_.begin(), result_.end(), 1u), 157);
    EXPECT_EQ(result_[pixel(86u, 20u)], 1u); EXPECT_EQ(result_[pixel(86u, 22u)], 0u);
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, UnknownMemoryWithCurrentSupportZeroCannotDraw) {
    author(); owner_values_[pixel(80u, 16u)] = SBC_OWNER_UNKNOWN_BIT | 7u;
    ASSERT_TRUE(run()); expect_subset(); EXPECT_EQ(std::count(result_.begin(), result_.end(), 1u), 0);
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, CurrentFilteredSupportWithoutLiteralKnownOwnerUsesUnknownAllowance) {
    for (UINT owner : {0u, 8u, SBC_OWNER_UNKNOWN_BIT | 7u}) {
      SCOPED_TRACE(owner); author(); owner_values_[pixel(80u, 16u)] = owner; support_values_[pixel(80u, 16u)] = 1u;
      ASSERT_TRUE(run()); EXPECT_EQ(std::count(result_.begin(), result_.end(), 1u), 157); expect_subset();
    }
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, ValidUnknownPendingAndSuspendedUseCurrentCoverFallback) {
    for (UINT phase : {SBC_PHASE_UNKNOWN, SBC_PHASE_PENDING, SBC_PHASE_SUSPENDED}) {
      SCOPED_TRACE(phase); author({{20u, 6u, 140u, 26u}}, phase); ASSERT_TRUE(run()); expect_cover_clip(); expect_subset();
    }
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, MixedActiveAndFallbackRetainsFallbackDependencyAcrossCoverBoundary) {
    author({{20u, 6u, 80u, 26u}, {80u, 6u, 140u, 26u}}); set_phase(1u, SBC_PHASE_PENDING);
    owner_values_[pixel(45u, 16u)] = 7u; ASSERT_TRUE(run()); expect_subset();
    EXPECT_EQ(result_[pixel(76u, 16u)], 1u); EXPECT_EQ(result_[pixel(70u, 16u)], 0u);
    EXPECT_EQ(result_[pixel(100u, 16u)], 1u); EXPECT_EQ(result_[pixel(45u, 16u)], 1u); EXPECT_EQ(result_[pixel(15u, 16u)], 0u);
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, FreshCurrentEmptyClearsEvenWhenPrivateStateIsUnavailable) {
    author({}); std::fill(state_words_.begin(), state_words_.end(), 0u); ASSERT_TRUE(run());
    EXPECT_EQ(std::count(result_.begin(), result_.end(), 1u), 0); expect_subset();
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, StaleOrMalformedTypedPacketsFailOpenToExactRawM1) {
    const std::array<UINT, 11u> state_faults {SBC_H_TAG, SBC_H_SCHEMA, SBC_H_FRAME, SBC_H_GENERATION, SBC_H_CLOCK, SBC_H_CUT,
      SBC_H_SOURCE, SBC_H_FIELD, SBC_H_CONTENT, SBC_H_COUNT, SBC_H_RESERVED};
    for (UINT word : state_faults) {
      SCOPED_TRACE("Private state fault " + std::to_string(word)); author(); ++state_words_[word];
      for (std::size_t p = 0u; p < raw_values_.size(); ++p) raw_values_[p] = UINT(p % 3u == 0u);
      ASSERT_TRUE(run()); EXPECT_EQ(result_, raw_values_); expect_subset();
    }
    for (UINT packet = 0u; packet < 3u; ++packet) {
      const std::vector<UINT> faults = packet == 0u ? std::vector<UINT> {0u, 1u, 3u, 5u, 7u, 9u, 11u, 13u} :
        packet == 1u ? std::vector<UINT> {0u, 1u, 4u, 6u, 8u, 20u, 23u} : std::vector<UINT> {0u, 1u, 4u, 6u, 20u, 22u, 23u, 24u};
      for (UINT word : faults) {
        SCOPED_TRACE("Current packet " + std::to_string(packet) + " word " + std::to_string(word)); author();
        auto &operand = packet == 0u ? ocr_words_ : packet == 1u ? uip_words_ : sct_words_; ++operand[word];
        for (std::size_t p = 0u; p < raw_values_.size(); ++p) raw_values_[p] = UINT(p % 3u == 0u);
        ASSERT_TRUE(run()); EXPECT_EQ(result_, raw_values_); expect_subset();
      }
    }
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, WrongSupportOrOwnerDimensionsCannotWithdrawRawM1) {
    for (UINT operand = 0u; operand < 2u; ++operand) {
      ASSERT_TRUE(allocate(160u, 90u)); author();
      ASSERT_TRUE(guard_texture(device_.Get(), width_ - 1u, crop_height_, false, operand == 0u ? support_ : owner_));
      ASSERT_TRUE(run()); EXPECT_EQ(result_, raw_values_); expect_subset();
    }
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, HalfOpenNativeCentersRespectOffsetContentAndBottomCrop) {
    ASSERT_TRUE(allocate(160u, 90u, {31u, 17u, 671u, 377u})); author({{40u, 10u, 111u, 21u}}, SBC_PHASE_PENDING);
    const guard_rect_t exact {31u + 4u * 40u + 2u, 17u + 4u * (crop_top_ + 10u) + 2u,
      31u + 4u * 110u + 2u, 17u + 4u * (crop_top_ + 20u) + 2u};
    set_rect(0u, exact); ASSERT_TRUE(run()); expect_cover_clip(); expect_subset();
    EXPECT_EQ(std::count(result_.begin(), result_.end(), 1u), 700);
    EXPECT_EQ(result_[pixel(40u, 10u)], 1u); EXPECT_EQ(result_[pixel(110u, 10u)], 0u);
    EXPECT_EQ(result_[pixel(40u, 20u)], 0u); EXPECT_EQ(result_[pixel(39u, 10u)], 0u);
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, AllowanceNeverAddsPixelsOutsideRawMatte) {
    author(); owner_values_[pixel(80u, 16u)] = 7u;
    for (std::size_t p = 0u; p < raw_values_.size(); ++p) raw_values_[p] = UINT(p % 2u == 0u);
    raw_values_[pixel(80u, 16u)] = 0u; ASSERT_TRUE(run()); expect_subset(); EXPECT_EQ(result_[pixel(80u, 16u)], 0u);
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, CurrentCoverMismatchUsesPerUnitFallbackInsteadOfOldGeometry) {
    author(); ++state_words_[cohort::entry_offset(0u) + SBC_E_LAST_COVER]; ASSERT_TRUE(run()); expect_cover_clip(); expect_subset();
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, LegalInvalidOrAmbiguousPrivateFlagsRetainCurrentCoverFallback) {
    for (UINT flag : {SBC_HEADER_CURRENT_INVALID, SBC_HEADER_AMBIGUOUS_ASSOCIATION}) {
      author(); state_words_[SBC_H_FLAGS] |= flag; ASSERT_TRUE(run()); expect_cover_clip(); expect_subset();
    }
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, ALineCannotBorrowKnownLifetimeFromAnotherCurrentCover) {
    author({{20u, 6u, 80u, 26u}, {80u, 6u, 140u, 26u}}); owner_values_[pixel(100u, 16u)] = 7u;
    ASSERT_TRUE(run()); EXPECT_EQ(std::count(result_.begin(), result_.end(), 1u), 0); expect_subset();
  }

  TEST_F(HostSbsSubtitleMatteGuardGpuTest, SavedNativeMatteGuardReplay) {
    const auto *input_env = std::getenv("SUNSHINE_TEST_MATTE_GUARD_CASES");
    const auto *output_env = std::getenv("SUNSHINE_TEST_MATTE_GUARD_OUTPUT");
    if (!input_env || !output_env) { GTEST_SKIP() << "Explicit saved-native guard replay paths required"; }
    const auto input_bytes = guard_bytes(input_env); const auto input = nlohmann::json::parse(input_bytes.begin(), input_bytes.end());
    ASSERT_EQ(input.at("schema"), 1u); const auto &cases = input.at("cases"); ASSERT_GT(cases.size(), 0u); ASSERT_LE(cases.size(), 32u);
    const std::filesystem::path destination(output_env); ASSERT_FALSE(std::filesystem::exists(destination)); ASSERT_TRUE(std::filesystem::create_directories(destination));
    const auto input_sha = guard_sha(input_bytes.data(), input_bytes.size());
    const auto *perf_env = std::getenv("SUNSHINE_TEST_MATTE_GUARD_PERF"); const bool performance = perf_env && std::string_view(perf_env) == "1";
    nlohmann::json report {{"schema", 1u}, {"driver", driver_}, {"adapter", adapter_identity()}, {"source_closure_sha256", closure_},
      {"input_specification_sha256", input_sha}, {"production_qualification_claimed", false}, {"cases", nlohmann::json::array()}};
    for (std::size_t i = 0u; i < cases.size(); ++i) {
      const auto &item = cases.at(i); SCOPED_TRACE(item.at("name").get<std::string>());
      const auto source = item.at("source").get<std::array<UINT, 2u>>(), field = item.at("field").get<std::array<UINT, 2u>>();
      const auto content = item.at("content").get<guard_rect_t>();
      ASSERT_GT(source[0u], 0u); ASSERT_GT(source[1u], 0u); ASSERT_LE(source[0u], cv2::subtitle_contour_source_max_extent); ASSERT_LE(source[1u], cv2::subtitle_contour_source_max_extent);
      ASSERT_TRUE(allocate(source[0u], source[1u], content, field[0u], field[1u]));
      const auto &artifacts = item.at("artifacts"); const auto cb = guard_bound_words(artifacts.at("constants"), SBC_CONSTANT_WORDS);
      std::memcpy(&constants_, cb.data(), sizeof(constants_));
      ASSERT_EQ(constants_.source, (std::array<UINT, 4u> {width_, height_, crop_top_, crop_height_}));
      ASSERT_EQ(constants_.field[0u], field_width_); ASSERT_EQ(constants_.field[1u], field_height_); ASSERT_EQ(constants_.content, content_);
      const auto frame = item.at("frame").get<std::uint64_t>(), gen = item.at("generation").get<std::uint64_t>(), time = item.at("observation_timestamp_us").get<std::uint64_t>();
      ASSERT_EQ(constants_.frame, (std::array<UINT, 4u> {UINT(frame), UINT(frame >> 32u), UINT(gen), UINT(gen >> 32u)}));
      ASSERT_EQ(constants_.observation, (std::array<UINT, 4u> {UINT(time), UINT(time >> 32u), item.value("input_domain_reset", false) ? 1u : 0u, 0u}));
      if (item.contains("roi")) {
        const auto roi = item.at("roi").get<std::array<UINT, 2u>>();
        ASSERT_EQ((std::array<UINT, 2u> {constants_.field[2u], constants_.field[3u]}), roi);
      }
      ASSERT_EQ(constants_.control, (std::array<UINT, 4u> {1u, SBC_SCHEMA, 0u, 0u}));
      const auto pixels = std::size_t(width_) * crop_height_;
      raw_values_ = guard_bound_words(artifacts.at("raw_matte"), pixels); support_values_ = guard_bound_words(artifacts.at("filtered_support"), pixels);
      owner_values_ = guard_bound_words(artifacts.at("owner"), pixels); state_words_ = guard_bound_words(artifacts.at("state"), SBC_STATE_WORDS);
      ocr_words_ = guard_bound_words(artifacts.at("ocr_record"), cv2::subtitle_ocr_record_word_count); uip_words_ = guard_bound_words(artifacts.at("current_plane"), cv2::subtitle_plane_state_word_count);
      sct_words_ = guard_bound_words(artifacts.at("current_identity"), cv2::subtitle_contour_history_word_count);
      ASSERT_TRUE(std::all_of(raw_values_.begin(), raw_values_.end(), [](UINT value) { return value <= 1u; }));
      ASSERT_TRUE(std::all_of(support_values_.begin(), support_values_.end(), [](UINT value) { return value <= 1u; }));
      ASSERT_TRUE(run()); expect_subset();
      const bool expected_supplied = artifacts.contains("expected_guarded_matte");
      if (expected_supplied) { ASSERT_EQ(result_, guard_bound_words(artifacts.at("expected_guarded_matte"), pixels)); }
      const auto folder = destination / ("case_" + std::to_string(i)); ASSERT_TRUE(std::filesystem::create_directory(folder));
      nlohmann::json saved = nlohmann::json::object();
      const auto save = [&](const std::string &name, const std::vector<UINT> &words) {
        const auto path = folder / name; std::ofstream file(path, std::ios::binary); file.write(reinterpret_cast<const char *>(words.data()), std::streamsize(words.size() * 4u));
        if (!file) throw std::runtime_error("Guard export write failed");
        saved[name] = {{"byte_count", words.size() * 4u}, {"sha256", guard_sha(words.data(), words.size() * 4u)}};
      };
      save("uploaded_raw_matte.u32", raw_values_); save("uploaded_filtered_support.u32", support_values_); save("uploaded_owner.u32", owner_values_);
      save("uploaded_state.u32", state_words_); save("uploaded_ocr_record.u32", ocr_words_); save("uploaded_current_plane.u32", uip_words_);
      save("uploaded_current_identity.u32", sct_words_); save("uploaded_constants.u32", cb); save("guarded_matte.u32", result_);
      nlohmann::json identity {{"schema", 1u}, {"case", item}, {"driver", driver_}, {"adapter", adapter_identity()}, {"shader_root", shader_root_.string()},
        {"source_closure_sha256", closure_}, {"input_specification_sha256", input_sha}, {"artifacts", saved}, {"expected_guarded_matte_supplied", expected_supplied},
        {"expected_guarded_byte_exact", expected_supplied ? nlohmann::json(true) : nlohmann::json(nullptr)}, {"subset_raw_M1", true},
        {"scope", "Private saved-native guard function only; exact CPUguard comparison is not GT quality, runtime integration, or production qualification"}, {"production_qualification_claimed", false}};
      if (performance) identity["performance"] = gpu_timestamps();
      const auto identity_text = identity.dump(2) + "\n"; std::ofstream id_file(folder / "matte_guard_fixture_identity.json", std::ios::binary); id_file << identity_text; ASSERT_TRUE(id_file.good()); id_file.close();
      report["cases"].push_back({{"index", i}, {"name", item.at("name")}, {"identity_sha256", guard_sha(identity_text.data(), identity_text.size())},
        {"expected_guarded_byte_exact", identity["expected_guarded_byte_exact"]}, {"guarded_pixels", std::count(result_.begin(), result_.end(), 1u)}});
      std::ofstream aggregate(destination / "matte_guard_replay_identity.json", std::ios::binary | std::ios::trunc); aggregate << report.dump(2) << '\n'; ASSERT_TRUE(aggregate.good());
    }
  }
}

#endif
