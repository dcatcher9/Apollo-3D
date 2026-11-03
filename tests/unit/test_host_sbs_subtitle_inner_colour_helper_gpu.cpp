/** Private helper GPU mechanics. Authored source/root controls, no model/warp/GT oracle. */
#include <gtest/gtest.h>
#ifdef _WIN32
#include <src/host_sbs_shader_cache.h>
#include <src/crypto.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string_view>
#include <thread>
#include <src/host_sbs_subtitle_inner_colour_gpu.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <d3d11.h>
#include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  using helper_t = models::host_sbs_subtitle_inner_colour_gpu::resources_t;
  struct helper_texture_t {
    ComPtr<ID3D11Texture2D> value;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };
  struct helper_buffer_t {
    ComPtr<ID3D11Buffer> value;
    ComPtr<ID3D11ShaderResourceView> srv;
  };
  bool helper_texture(ID3D11Device *d, UINT w, UINT h, DXGI_FORMAT format, helper_texture_t &out, bool writable = false) {
    D3D11_TEXTURE2D_DESC x {};
    x.Width = w;x.Height = h;x.MipLevels = x.ArraySize = 1u;x.Format = format;x.SampleDesc.Count = 1u;
    x.Usage = D3D11_USAGE_DEFAULT;x.BindFlags = D3D11_BIND_SHADER_RESOURCE | (writable ? D3D11_BIND_UNORDERED_ACCESS : 0u);
    return SUCCEEDED(d->CreateTexture2D(&x, nullptr, &out.value)) &&
      SUCCEEDED(d->CreateShaderResourceView(out.value.Get(), nullptr, &out.srv)) &&
      (!writable || SUCCEEDED(d->CreateUnorderedAccessView(out.value.Get(), nullptr, &out.uav)));
  }
  bool helper_buffer(ID3D11Device *d, UINT words, helper_buffer_t &out) {
    D3D11_BUFFER_DESC x {};
    x.ByteWidth = words * 4u;x.Usage = D3D11_USAGE_DEFAULT;x.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    x.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;x.StructureByteStride = 4u;
    return SUCCEEDED(d->CreateBuffer(&x, nullptr, &out.value)) &&
      SUCCEEDED(d->CreateShaderResourceView(out.value.Get(), nullptr, &out.srv));
  }
  bool helper_cb(ID3D11Device *d, UINT bytes, ComPtr<ID3D11Buffer> &out) {
    D3D11_BUFFER_DESC x {};x.ByteWidth = bytes;x.Usage = D3D11_USAGE_DEFAULT;x.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    return SUCCEEDED(d->CreateBuffer(&x, nullptr, &out));
  }

  std::vector<char> whole_perf_bytes(const std::filesystem::path &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0) throw std::runtime_error("Cannot read whole-helper input");
    std::vector<char> out(static_cast<std::size_t>(f.tellg())); f.seekg(0);
    if (!out.empty()) { f.read(out.data(), static_cast<std::streamsize>(out.size())); }
    if (!f) throw std::runtime_error("Incomplete whole-helper input");
    return out;
  }
  std::string whole_perf_sha(const void *data, std::size_t bytes) {
    const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), bytes));
    constexpr char digits[] = "0123456789abcdef"; std::string out;
    for (auto byte : digest) { out.push_back(digits[byte >> 4u]); out.push_back(digits[byte & 15u]); }
    return out;
  }
  std::vector<char> whole_perf_bound(const nlohmann::json &b, std::size_t size) {
    auto out = whole_perf_bytes(b.at("path").get<std::string>());
    const auto declared = b.contains("bytes") ? b.at("bytes").get<std::size_t>() : b.at("byte_count").get<std::size_t>();
    if (b.contains("bytes") && b.contains("byte_count") && b.at("bytes") != b.at("byte_count")) throw std::runtime_error("Conflicting byte-count aliases");
    if (out.size() != size || out.size() != declared || whole_perf_sha(out.data(), out.size()) != b.at("sha256").get<std::string>()) throw std::runtime_error("Whole-helper binding mismatch");
    return out;
  }
  template<class T> std::vector<char> whole_perf_raw(const std::vector<T> &v) {
    std::vector<char> out(v.size() * sizeof(T)); if (!out.empty()) std::memcpy(out.data(), v.data(), out.size()); return out;
  }
  class HostSbsInnerColourHelperGpuTest : public ::testing::Test {
  protected:
    static constexpr UINT width = 32u, height = 16u, root_offset = 240u;
    using half_pixel = std::array<std::uint16_t, 4u>;
    void SetUp() override {
      if (std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) == "SavedFullInputPerformance") {
        const auto *flag = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_HELPER_PERF");
        if (!flag || std::string_view(flag) != "1" || !std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_HELPER_PERF_CASES") || !std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_HELPER_PERF_OUTPUT")) {
          GTEST_SKIP() << "Explicit whole-helper performance opt-in and input/output required";
        }
      }
      const auto *ip = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_SHADER_PIN");
      const auto *cp = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_COMPONENT_PIN");
      if (!ip || !cp) GTEST_SKIP() << "Explicit private colour/component source pins required.";
      auto inner_root = std::filesystem::path {SUNSHINE_SHADERS_DIR};
      auto component_root = inner_root;
      if (const auto *p = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_SHADER_ROOT")) inner_root = p;
      if (const auto *p = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_COMPONENT_ROOT")) component_root = p;
      const auto *driver = std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_DRIVER");
      ASSERT_TRUE(!driver || std::string {driver} == "warp" || std::string {driver} == "hardware");
      hardware_ = driver && std::string {driver} == "hardware";
      D3D_FEATURE_LEVEL actual {}, requested[] {D3D_FEATURE_LEVEL_11_0};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, hardware_ ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP,
        nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &device_, &actual, &context_)));
      ASSERT_EQ(actual, D3D_FEATURE_LEVEL_11_0);
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
      const auto inner = cache::snapshot_sources(inner_root, inner_specs);
      const auto components = cache::snapshot_sources(component_root, component_specs);
      ASSERT_EQ(cache::source_closure_sha256(inner), ip);
      ASSERT_EQ(cache::source_closure_sha256(components), cp);
      inner_closure_ = cache::source_closure_sha256(inner); component_closure_ = cache::source_closure_sha256(components);
      inner_root_path_ = inner_root; component_root_path_ = component_root;
      RecordProperty("inner_closure", ip);RecordProperty("component_closure", cp);
      for (std::size_t i = 0u; i < inner_specs.size(); ++i) {
        const auto code = cache::get(inner, inner_specs[i]);ASSERT_TRUE(code && !code->empty());
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &colour_[i])));
      }
      for (std::size_t i = 0u; i < component_specs.size(); ++i) {
        const auto code = cache::get(components, component_specs[i]);ASSERT_TRUE(code && !code->empty());
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &components_[i])));
      }
      ASSERT_TRUE(helper_.ensure(device_.Get(), width, height, error_)) << error_;
      ASSERT_TRUE(helper_texture(device_.Get(), width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, rgb_));
      ASSERT_TRUE(helper_texture(device_.Get(), width, height, DXGI_FORMAT_R32_UINT, support_));
      ASSERT_TRUE(helper_buffer(device_.Get(), 153600u, probability_));
      ASSERT_TRUE(helper_buffer(device_.Get(), 256u, ocr_));
      ASSERT_TRUE(helper_cb(device_.Get(), 96u, current_cb_));
      D3D11_BUFFER_DESC arg {};arg.ByteWidth = 256u;arg.Usage = D3D11_USAGE_DEFAULT;
      arg.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&arg, nullptr, &root_args_)));
      ASSERT_NO_FATAL_FAILURE(author(true, 1u));
    }
    void TearDown() override {
      if (!context_) return;
      std::array<ID3D11ShaderResourceView *, 8u> s {};
      std::array<ID3D11UnorderedAccessView *, 8u> u {};
      std::array<ID3D11Buffer *, 2u> c {};
      context_->CSSetShaderResources(0u, 8u, s.data());
      context_->CSSetUnorderedAccessViews(0u, 8u, u.data(), nullptr);
      context_->CSSetConstantBuffers(0u, 2u, c.data());context_->CSSetShader(nullptr, nullptr, 0u);
    }
    void author(bool caption, UINT frame) {
      std::vector<half_pixel> pixels(width * height, {0x3000u, 0x3000u, 0x3000u, 0x3c00u});
      if (caption) {
        // Exact closed12x5 white body, explicit two-pixel black surround. Native S stays0.
        for (UINT y = 4u; y < 13u; ++y) for (UINT x = 8u; x < 24u; ++x) pixels[y * width + x] = {0u, 0u, 0u, 0x3c00u};
        for (UINT y = 6u; y < 11u; ++y) for (UINT x = 10u; x < 22u; ++x) pixels[y * width + x] = {0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u};
      }
      const std::vector<UINT> support(width * height, 0u);
      const std::vector<float> probability(153600u, 1.0f);
      const std::array<UINT, 24u> cb {width, height, 0u, height, 16u, 16u, 0u, 16u, 0u, 0u, 16u, 16u,
        frame, 0u, 7u, 0u, 1000000u + frame, 0u, 0u, 0u, 1u, 5u, 0u, 0u};
      std::array<UINT, 256u> ocr {};
      ocr[0] = 4u;ocr[1] = 0x3852434fu;ocr[2] = 1u;ocr[3] = ocr[4] = caption ? 1u : 0u;
      std::copy(cb.begin() + 12u, cb.begin() + 16u, ocr.begin() + 5u);
      ocr[9] = width;ocr[10] = height;std::copy(cb.begin() + 4u, cb.begin() + 8u, ocr.begin() + 11u);
      if (caption) {
        const std::array<UINT, 4u> core {4u, 5u, 12u, 12u}, cover {2u, 3u, 14u, 14u};
        std::copy(core.begin(), core.end(), ocr.begin() + 16u);
        std::copy(cover.begin(), cover.end(), ocr.begin() + 144u);
      }
      context_->UpdateSubresource(rgb_.value.Get(), 0u, nullptr, pixels.data(), width * 8u, 0u);
      context_->UpdateSubresource(support_.value.Get(), 0u, nullptr, support.data(), width * 4u, 0u);
      context_->UpdateSubresource(probability_.value.Get(), 0u, nullptr, probability.data(), 0u, 0u);
      context_->UpdateSubresource(ocr_.value.Get(), 0u, nullptr, ocr.data(), 0u, 0u);
      context_->UpdateSubresource(current_cb_.Get(), 0u, nullptr, cb.data(), 0u, 0u);
    }
    bool run(UINT gate) {
      std::array<UINT, 64u> args {};args[60] = gate;args[61] = args[62] = 1u;
      context_->UpdateSubresource(root_args_.Get(), 0u, nullptr, args.data(), 0u, 0u);
      std::array<ID3D11ComputeShader *, 4u> c {};
      std::array<ID3D11ComputeShader *, 6u> m {};
      for (std::size_t i = 0u; i < c.size(); ++i) c[i] = colour_[i].Get();
      for (std::size_t i = 0u; i < m.size(); ++i) m[i] = components_[i].Get();
      return helper_.dispatch(context_.Get(), c, m, current_cb_.Get(), rgb_.srv.Get(), probability_.srv.Get(),
        support_.srv.Get(), ocr_.srv.Get(), root_args_.Get(), root_offset, error_);
    }
    bool dispatch_preuploaded() {
      // ROOT1 and all full source inputs are uploaded outside the timed range.
      std::array<ID3D11ComputeShader *, 4u> c {}; std::array<ID3D11ComputeShader *, 6u> m {};
      for (std::size_t i = 0u; i < c.size(); ++i) c[i] = colour_[i].Get();
      for (std::size_t i = 0u; i < m.size(); ++i) m[i] = components_[i].Get();
      return helper_.dispatch(context_.Get(), c, m, current_cb_.Get(), rgb_.srv.Get(), probability_.srv.Get(),
        support_.srv.Get(), ocr_.srv.Get(), root_args_.Get(), root_offset, error_);
    }
    std::vector<char> perf_read_texture_bytes(ID3D11Texture2D *input) {
      D3D11_TEXTURE2D_DESC d {}; input->GetDesc(&d);
      const UINT pixel_bytes = d.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8u : d.Format == DXGI_FORMAT_R32_UINT ? 4u : 0u;
      if (!pixel_bytes) throw std::runtime_error("Unsupported whole-helper input texture");
      d.Usage = D3D11_USAGE_STAGING; d.BindFlags = d.MiscFlags = 0u; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> stage; require_hresult(device_->CreateTexture2D(&d, nullptr, &stage));
      context_->CopyResource(stage.Get(), input); D3D11_MAPPED_SUBRESOURCE mapped {};
      require_hresult(context_->Map(stage.Get(), 0u, D3D11_MAP_READ, 0u, &mapped));
      std::vector<char> out(std::size_t(d.Width) * d.Height * pixel_bytes);
      for (UINT y = 0u; y < d.Height; ++y) { std::memcpy(out.data() + std::size_t(y) * d.Width * pixel_bytes,
        static_cast<const char *>(mapped.pData) + std::size_t(y) * mapped.RowPitch, d.Width * pixel_bytes); }
      context_->Unmap(stage.Get(), 0u); return out;
    }
    nlohmann::json whole_helper_performance(UINT source_width, UINT crop_height) {
      constexpr UINT warmup = 3u, repeats = 20u;
      for (UINT i = 0u; i < warmup; ++i) { if (!dispatch_preuploaded()) throw std::runtime_error(error_); }
      D3D11_QUERY_DESC d {D3D11_QUERY_TIMESTAMP_DISJOINT, 0u}; ComPtr<ID3D11Query> disjoint;
      require_hresult(device_->CreateQuery(&d, &disjoint)); d.Query = D3D11_QUERY_TIMESTAMP;
      std::array<ComPtr<ID3D11Query>, repeats * 2u> queries;
      for (auto &q : queries) { require_hresult(device_->CreateQuery(&d, &q)); }
      context_->Begin(disjoint.Get());
      for (UINT i = 0u; i < repeats; ++i) {
        context_->End(queries[2u * i].Get());
        if (!dispatch_preuploaded()) throw std::runtime_error(error_);
        context_->End(queries[2u * i + 1u].Get());
      }
      context_->End(disjoint.Get()); context_->Flush();
      const auto wait = [&](ID3D11Query *q, void *value, UINT bytes) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20); HRESULT hr;
        while ((hr = context_->GetData(q, value, bytes, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE) {
          if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Whole-helper timestamp timeout");
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require_hresult(hr);
      };
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT evidence {}; wait(disjoint.Get(), &evidence, sizeof(evidence));
      nlohmann::json ticks = nlohmann::json::array(), ns = nlohmann::json::array();
      for (UINT i = 0u; i < repeats; ++i) {
        UINT64 begin {}, end {}; wait(queries[2u * i].Get(), &begin, sizeof(begin)); wait(queries[2u * i + 1u].Get(), &end, sizeof(end));
        ticks.push_back({begin, end});
        if (!evidence.Disjoint && evidence.Frequency != 0u && end >= begin) ns.push_back(double(end - begin) * 1e9 / double(evidence.Frequency));
      }
      nlohmann::json result {{"warmup", warmup}, {"repeats", repeats}, {"timestamp_ticks", ticks}, {"frequency_hz", evidence.Frequency},
        {"disjoint", evidence.Disjoint != FALSE}, {"gpu_ns", ns}, {"valid_intervals", ns.size()}, {"source_width", source_width}, {"crop_height", crop_height},
        {"native_clear_groups", {(source_width + 15u) / 16u, (crop_height + 15u) / 16u, 1u}}, {"forced_root_one", true},
        {"scope", "GPU timestamp range enclosing complete resources_t::dispatch call: nativeclear GPUgatecopy, perunitmodeargsclear, fullRGB/P/S validation andpalette, activeemit+CC5+BOX3+GPUOR, commandbindings/requestCBupdates/CSsave-restore. Fixed8unit/8mode CPUsubmission loop withGPU inactive gating. Not a sum of isolated kernels."},
        {"excluded", "Inputupload, allocation, shadercompilation, functional/diagnosticCopyResource andreadback, expecteddata/CPUprep, baselineSgeneration, depth/OCR/modelinference, renderer andcompletepresentation pipeline."},
        {"diagnostic_only_until_external_idle_GPU_proof", true}, {"qualified_performance_comparison", false}, {"frequency_role", "GPUtimestamp clockHz, notGPUcorefrequency"}};
      if (ns.size() == repeats) {
        auto values = ns.get<std::vector<double>>(); std::sort(values.begin(), values.end()); double sum = 0.; for (auto value : values) { sum += value; }
        result["gpu_ns_summary"] = {{"minimum", values.front()}, {"median", .5 * (values[9u] + values[10u])}, {"mean", sum / repeats}, {"maximum", values.back()}};
      }
      return result;
    }
    std::vector<UINT> read(ID3D11Resource *resource) {
      if (!resource) throw std::runtime_error("Missing helper GPU resource");
      D3D11_RESOURCE_DIMENSION dimension {};resource->GetType(&dimension);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (dimension == D3D11_RESOURCE_DIMENSION_BUFFER) {
        ComPtr<ID3D11Buffer> source, staging;require_hresult(resource->QueryInterface(IID_PPV_ARGS(&source)));
        D3D11_BUFFER_DESC d {};source->GetDesc(&d);d.Usage = D3D11_USAGE_STAGING;d.BindFlags = 0u;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;d.MiscFlags = d.StructureByteStride = 0u;
        require_hresult(device_->CreateBuffer(&d, nullptr, &staging));context_->CopyResource(staging.Get(), source.Get());
        require_hresult(context_->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped));
        std::vector<UINT> result(d.ByteWidth / 4u);std::memcpy(result.data(), mapped.pData, d.ByteWidth);
        context_->Unmap(staging.Get(), 0u);return result;
      }
      ComPtr<ID3D11Texture2D> source, staging;require_hresult(resource->QueryInterface(IID_PPV_ARGS(&source)));
      D3D11_TEXTURE2D_DESC d {};source->GetDesc(&d);
      if (d.Format != DXGI_FORMAT_R32_UINT) throw std::runtime_error("Helper output must be R32uint");
      d.Usage = D3D11_USAGE_STAGING;d.BindFlags = 0u;d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;d.MiscFlags = 0u;
      require_hresult(device_->CreateTexture2D(&d, nullptr, &staging));context_->CopyResource(staging.Get(), source.Get());
      require_hresult(context_->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped));
      std::vector<UINT> result(std::size_t(d.Width) * d.Height);
      for (UINT y = 0u; y < d.Height; ++y) std::memcpy(result.data() + std::size_t(y) * d.Width,
        static_cast<const char *>(mapped.pData) + std::size_t(y) * mapped.RowPitch, d.Width * 4u);
      context_->Unmap(staging.Get(), 0u);return result;
    }
    std::vector<UINT> read(ID3D11ShaderResourceView *view) {
      ComPtr<ID3D11Resource> resource;view->GetResource(&resource);return read(resource.Get());
    }
    static void require_hresult(HRESULT result) {
      if (FAILED(result)) throw std::runtime_error("Helper GPU staging operation failed");
    }
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    std::array<ComPtr<ID3D11ComputeShader>, 4u> colour_;
    std::array<ComPtr<ID3D11ComputeShader>, 6u> components_;
    helper_texture_t rgb_, support_;
    helper_buffer_t probability_, ocr_;
    ComPtr<ID3D11Buffer> current_cb_, root_args_;
    helper_t helper_;
    std::string error_;
    bool hardware_ = false;
    std::string inner_closure_, component_closure_;
    std::filesystem::path inner_root_path_, component_root_path_;
  };

  TEST_F(HostSbsInnerColourHelperGpuTest, FreshThenGpuRootZeroHoldsAndFreshEmptyClearsWholeNativeWidth) {
    ASSERT_TRUE(run(1u)) << error_;
    const auto views = helper_.mutable_views();
    const auto body = read(views.body.Get()), matte = read(views.matte.Get());
    std::vector<UINT> expected_body(width * height), expected_matte(width * height);
    for (UINT y = 6u; y < 11u; ++y) for (UINT x = 10u; x < 22u; ++x) expected_body[y * width + x] = 1u;
    for (UINT y = 3u; y < 14u; ++y) for (UINT x = 7u; x < 25u; ++x) expected_matte[y * width + x] = 1u;
    ASSERT_EQ(body, expected_body);ASSERT_EQ(matte, expected_matte);
    EXPECT_EQ(read(views.native_clear_args.Get()), (std::vector<UINT> {2u, 1u, 1u}));
    // Poison prior indirect arguments. Root0 must prevent BOTH a clear and any stale-mode work.
    std::array<UINT, 24u> poison;poison.fill(1u);
    context_->UpdateSubresource(views.current_mode_args.Get(), 0u, nullptr, poison.data(), 0u, 0u);
    ASSERT_NO_FATAL_FAILURE(author(false, 2u));
    ASSERT_TRUE(run(0u)) << error_;
    EXPECT_EQ(read(views.body.Get()), body);EXPECT_EQ(read(views.matte.Get()), matte);
    EXPECT_EQ(read(views.native_clear_args.Get()), (std::vector<UINT> {2u, 1u, 0u}));
    EXPECT_EQ(read(views.current_mode_args.Get()), std::vector<UINT>(24u, 0u));
    ASSERT_TRUE(run(1u)) << error_;
    EXPECT_EQ(read(views.body.Get()), std::vector<UINT>(width * height, 0u));
    EXPECT_EQ(read(views.matte.Get()), std::vector<UINT>(width * height, 0u));
    EXPECT_EQ(read(views.native_clear_args.Get()), (std::vector<UINT> {2u, 1u, 1u}));
  }

  TEST_F(HostSbsInnerColourHelperGpuTest, RestoresAllTouchedComputeBindings) {
    const auto views = helper_.mutable_views();
    std::array<ID3D11ShaderResourceView *, 8u> srvs {rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(),
      views.body.Get(), views.matte.Get(), views.palette.Get(), views.parent.Get()};
    std::array<helper_texture_t, 8u> dummy;
    std::array<ID3D11UnorderedAccessView *, 8u> uavs {};
    for (std::size_t i = 0u; i < dummy.size(); ++i) {
      ASSERT_TRUE(helper_texture(device_.Get(), width, height, DXGI_FORMAT_R32_UINT, dummy[i], true));uavs[i] = dummy[i].uav.Get();
    }
    ComPtr<ID3D11Buffer> cb0, cb1;
    ASSERT_TRUE(helper_cb(device_.Get(), 96u, cb0));ASSERT_TRUE(helper_cb(device_.Get(), 16u, cb1));
    ID3D11Buffer *cbs[] {cb0.Get(), cb1.Get()};
    context_->CSSetShaderResources(0u, 8u, srvs.data());context_->CSSetUnorderedAccessViews(0u, 8u, uavs.data(), nullptr);
    context_->CSSetConstantBuffers(0u, 2u, cbs);context_->CSSetShader(colour_[2u].Get(), nullptr, 0u);
    ASSERT_TRUE(run(1u)) << error_;
    std::array<ID3D11ShaderResourceView *, 8u> actual_srvs {};
    std::array<ID3D11UnorderedAccessView *, 8u> actual_uavs {};
    std::array<ID3D11Buffer *, 2u> actual_cbs {};
    ID3D11ComputeShader *actual_shader = nullptr;
    context_->CSGetShaderResources(0u, 8u, actual_srvs.data());context_->CSGetUnorderedAccessViews(0u, 8u, actual_uavs.data());
    context_->CSGetConstantBuffers(0u, 2u, actual_cbs.data());context_->CSGetShader(&actual_shader, nullptr, nullptr);
    for (std::size_t i = 0u; i < srvs.size(); ++i) {EXPECT_EQ(actual_srvs[i], srvs[i]);if (actual_srvs[i]) actual_srvs[i]->Release();}
    for (std::size_t i = 0u; i < uavs.size(); ++i) {EXPECT_EQ(actual_uavs[i], uavs[i]);if (actual_uavs[i]) actual_uavs[i]->Release();}
    for (std::size_t i = 0u; i < actual_cbs.size(); ++i) {EXPECT_EQ(actual_cbs[i], cbs[i]);if (actual_cbs[i]) actual_cbs[i]->Release();}
    EXPECT_EQ(actual_shader, colour_[2u].Get());if (actual_shader) actual_shader->Release();
  }

  TEST_F(HostSbsInnerColourHelperGpuTest, EnsureReusesGeometryAndReplacesResizeOrDeviceWithDefinedZero) {
    auto views = helper_.mutable_views();ComPtr<ID3D11Resource> first;
    views.matte->GetResource(&first);
    ASSERT_TRUE(helper_.ensure(device_.Get(), width, height, error_));
    ComPtr<ID3D11Resource> same;helper_.mutable_views().matte->GetResource(&same);EXPECT_EQ(first.Get(), same.Get());
    ASSERT_TRUE(helper_.ensure(device_.Get(), width + 16u, height, error_));
    auto resized = helper_.mutable_views();ComPtr<ID3D11Resource> changed;resized.matte->GetResource(&changed);
    EXPECT_NE(first.Get(), changed.Get());EXPECT_EQ(resized.width, width + 16u);
    EXPECT_EQ(read(resized.matte.Get()), std::vector<UINT>((width + 16u) * height, 0u));
    EXPECT_FALSE(helper_.ensure(device_.Get(), 0u, height, error_));EXPECT_EQ(helper_.mutable_views().matte.Get(), resized.matte.Get());
    ASSERT_TRUE(helper_.ensure(device_.Get(), 1u, 1u, error_));
    auto tiny = helper_.mutable_views();EXPECT_EQ(read(tiny.matte.Get()), std::vector<UINT>(1u, 0u));
    EXPECT_EQ(read(tiny.palette.Get()), std::vector<UINT>(128u, 0u));
    ComPtr<ID3D11Device> other;ComPtr<ID3D11DeviceContext> other_context;
    D3D_FEATURE_LEVEL actual {}, requested[] {D3D_FEATURE_LEVEL_11_0};
    ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, hardware_ ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP,
      nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &other, &actual, &other_context)));
    ASSERT_TRUE(helper_.ensure(other.Get(), width, height, error_));
    auto switched = helper_.mutable_views();ComPtr<ID3D11Device> owner;switched.matte->GetDevice(&owner);
    EXPECT_EQ(owner.Get(), other.Get());EXPECT_NE(switched.matte.Get(), views.matte.Get());
  }
  TEST_F(HostSbsInnerColourHelperGpuTest, SavedFullInputPerformance) {
    const std::filesystem::path input(std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_HELPER_PERF_CASES"));
    const std::filesystem::path output(std::getenv("SUNSHINE_TEST_SUBTITLE_INNER_HELPER_PERF_OUTPUT"));
    const auto input_bytes = whole_perf_bytes(input); const auto spec = nlohmann::json::parse(input_bytes.begin(), input_bytes.end());
    ASSERT_EQ(spec.at("schema"), 1u); ASSERT_EQ(spec.at("inner_source_closure_sha256"), inner_closure_); ASSERT_EQ(spec.at("component_source_closure_sha256"), component_closure_);
    ASSERT_EQ(spec.value("driver", std::string("warp")), hardware_ ? "hardware" : "warp");
    const auto &cases = spec.at("cases"); ASSERT_GT(cases.size(), 0u); ASSERT_LE(cases.size(), 64u); ASSERT_FALSE(std::filesystem::exists(output)); ASSERT_TRUE(std::filesystem::create_directories(output));
    ComPtr<IDXGIDevice> dxgi; ComPtr<IDXGIAdapter> gpu; DXGI_ADAPTER_DESC desc {};
    require_hresult(device_.As(&dxgi)); require_hresult(dxgi->GetAdapter(&gpu)); require_hresult(gpu->GetDesc(&desc));
    const std::wstring wide(desc.Description); const nlohmann::json adapter {{"description", std::string(wide.begin(), wide.end())}, {"vendor_id", desc.VendorId}, {"device_id", desc.DeviceId}, {"luid_low", desc.AdapterLuid.LowPart}, {"luid_high", desc.AdapterLuid.HighPart}};
    if (hardware_) { ASSERT_EQ(desc.VendorId, 4318u) << "Expected native NVIDIA hardware benchmark device"; }
    std::array<wchar_t, 32768u> executable_path {}; const auto length = GetModuleFileNameW(nullptr, executable_path.data(), DWORD(executable_path.size()));
    ASSERT_GT(length, 0u); ASSERT_LT(length, executable_path.size()); const std::filesystem::path executable(std::wstring(executable_path.data(), length)); const auto exe = whole_perf_bytes(executable);
    const nlohmann::json exe_binding {{"path", executable.string()}, {"bytes", exe.size()}, {"sha256", whole_perf_sha(exe.data(), exe.size())}};
    const auto input_hash = whole_perf_sha(input_bytes.data(), input_bytes.size());
    nlohmann::json aggregate {{"schema", 1u}, {"input_specification_sha256", input_hash}, {"adapter", adapter}, {"test_executable", exe_binding},
      {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", component_closure_}, {"helper_source_pin", "d003fa5f185c606ff6d748697dd70405e881cc1dc7ab18d8d66e487ea5794f3a"},
      {"qualified_performance_comparison", false}, {"cases", nlohmann::json::array()}};
    for (std::size_t index = 0u; index < cases.size(); ++index) {
      const auto &item = cases.at(index); SCOPED_TRACE(item.at("name").get<std::string>()); const auto &a = item.at("artifacts");
      const auto cb_bytes = whole_perf_bound(a.at("constants"), 96u); std::array<UINT, 24u> cb {}; std::memcpy(cb.data(), cb_bytes.data(), 96u);
      const UINT source_width = cb[0u], source_height = cb[1u], top = cb[2u], crop_height = cb[3u];
      ASSERT_GT(source_width, 0u); ASSERT_LE(source_width, 5120u); ASSERT_GT(source_height, 0u); ASSERT_LE(source_height, 5120u); ASSERT_LE(top, source_height); ASSERT_EQ(crop_height, source_height - top); ASSERT_GT(crop_height, 0u);
      const auto n = std::size_t(source_width) * crop_height;
      const auto source = whole_perf_bound(a.at("source_band"), n * 8u), probability = whole_perf_bound(a.at("probability"), 614400u);
      const auto support = whole_perf_bound(a.at("current_support"), n * 4u), ocr = whole_perf_bound(a.at("ocr_record"), 1024u);
      ASSERT_TRUE(helper_texture(device_.Get(), source_width, crop_height, DXGI_FORMAT_R16G16B16A16_FLOAT, rgb_)); ASSERT_TRUE(helper_texture(device_.Get(), source_width, crop_height, DXGI_FORMAT_R32_UINT, support_));
      ASSERT_TRUE(helper_.ensure(device_.Get(), source_width, crop_height, error_)) << error_;
      context_->UpdateSubresource(rgb_.value.Get(), 0u, nullptr, source.data(), source_width * 8u, 0u); context_->UpdateSubresource(support_.value.Get(), 0u, nullptr, support.data(), source_width * 4u, 0u);
      context_->UpdateSubresource(probability_.value.Get(), 0u, nullptr, probability.data(), 0u, 0u); context_->UpdateSubresource(ocr_.value.Get(), 0u, nullptr, ocr.data(), 0u, 0u); context_->UpdateSubresource(current_cb_.Get(), 0u, nullptr, cb_bytes.data(), 0u, 0u);
      std::array<UINT, 64u> root {}; root[60u] = root[61u] = root[62u] = 1u; context_->UpdateSubresource(root_args_.Get(), 0u, nullptr, root.data(), 0u, 0u);
      ASSERT_TRUE(dispatch_preuploaded()) << error_;
      const auto views = helper_.mutable_views(); const auto body = read(views.body.Get()), matte = read(views.matte.Get());
      const auto expected_body = whole_perf_bound(item.at("expected").at("body"), n * 4u), expected_final = whole_perf_bound(item.at("expected").at("final"), n * 4u);
      ASSERT_EQ(whole_perf_raw(body), expected_body); ASSERT_EQ(whole_perf_raw(matte), expected_final);
      const auto uploaded_source = perf_read_texture_bytes(rgb_.value.Get()), uploaded_support = perf_read_texture_bytes(support_.value.Get());
      const auto uploaded_probability = whole_perf_raw(read(probability_.value.Get())), uploaded_ocr = whole_perf_raw(read(ocr_.value.Get())), uploaded_cb = whole_perf_raw(read(current_cb_.Get()));
      ASSERT_EQ(uploaded_source, source); ASSERT_EQ(uploaded_support, support); ASSERT_EQ(uploaded_probability, probability); ASSERT_EQ(uploaded_ocr, ocr); ASSERT_EQ(uploaded_cb, cb_bytes);
      ASSERT_EQ(read(views.native_clear_args.Get()), (std::vector<UINT> {(source_width + 15u) / 16u, (crop_height + 15u) / 16u, 1u}));
      const auto folder = output / ("case_" + std::to_string(index)); ASSERT_TRUE(std::filesystem::create_directory(folder));
      const auto save = [&](const std::string &name, const std::vector<char> &bytes) {
        std::ofstream f(folder / name, std::ios::binary); f.write(bytes.data(), std::streamsize(bytes.size())); if (!f) throw std::runtime_error("Whole-helper export failed");
        return nlohmann::json {{"byte_count", bytes.size()}, {"sha256", whole_perf_sha(bytes.data(), bytes.size())}};
      };
      nlohmann::json artifacts {{"uploaded_source.rgba16f", save("uploaded_source.rgba16f", uploaded_source)}, {"uploaded_probability.f32", save("uploaded_probability.f32", uploaded_probability)},
        {"uploaded_current_support.u32", save("uploaded_current_support.u32", uploaded_support)}, {"uploaded_ocr_record.u32", save("uploaded_ocr_record.u32", uploaded_ocr)}, {"uploaded_constants.u32", save("uploaded_constants.u32", uploaded_cb)},
        {"body.u32", save("body.u32", whole_perf_raw(body))}, {"final.u32", save("final.u32", whole_perf_raw(matte))}, {"root_arguments.u32", save("root_arguments.u32", whole_perf_raw(read(root_args_.Get())))},
        {"native_clear_arguments.u32", save("native_clear_arguments.u32", whole_perf_raw(read(views.native_clear_args.Get())))}};
      nlohmann::json identity {{"schema", 1u}, {"case", item}, {"driver", hardware_ ? "hardware" : "warp"}, {"adapter", adapter}, {"input_specification_sha256", input_hash}, {"test_executable", exe_binding},
        {"inner_shader_root", inner_root_path_.string()}, {"component_shader_root", component_root_path_.string()}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", component_closure_},
        {"helper_source_pin", "d003fa5f185c606ff6d748697dd70405e881cc1dc7ab18d8d66e487ea5794f3a"}, {"artifacts", artifacts}, {"expected_whole_body_and_final_byte_exact", true}, {"uploaded_five_native_inputs_byte_exact", true},
        {"benchmark_root_control", "Preuploaded canonical ROOT1 atoffset240; forced recompute for benchmark, not an observed new model/GPU-adaptive verdict"},
        {"scope", "Actual whole helper dispatch on authentic full source inputs. Not preuploaded eligibility, not model/OCR/baselineS generation, not renderer or end-to-end speed."}, {"qualified_performance_comparison", false}};
      const auto proof_text = identity.dump(2) + "\n"; std::ofstream proof(folder / "functional_before_perf.json", std::ios::binary); proof << proof_text; ASSERT_TRUE(proof.good()); proof.close();
      // No diagnosticCopyResource/readback/sidecar access occurs inside this call.
      identity["performance"] = whole_helper_performance(source_width, crop_height);
      const bool post_exact = read(views.body.Get()) == body && read(views.matte.Get()) == matte;
      identity["post_performance_outputs_byte_exact"] = post_exact;
      identity["functional_proof_before_performance"] = {{"filename", "functional_before_perf.json"}, {"byte_count", proof_text.size()}, {"sha256", whole_perf_sha(proof_text.data(), proof_text.size())}};
      const auto text = identity.dump(2) + "\n"; std::ofstream f(folder / "whole_helper_perf_fixture_identity.json", std::ios::binary); f << text; ASSERT_TRUE(f.good()); f.close();
      aggregate["cases"].push_back({{"index", index}, {"name", item.at("name")}, {"identity_sha256", whole_perf_sha(text.data(), text.size())}, {"post_performance_outputs_byte_exact", post_exact}});
      std::ofstream report(output / "whole_helper_perf_replay_identity.json", std::ios::binary | std::ios::trunc); report << aggregate.dump(2) << '\n'; ASSERT_TRUE(report.good()); EXPECT_TRUE(post_exact);
    }
  }
} // namespace
#endif
