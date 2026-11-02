/**
 * @file tests/unit/test_host_sbs_subtitle_seeded_components_gpu.cpp
 * @brief Research-only exact seeded components; no palette, OCR or drawing authority.
 */
#include <gtest/gtest.h>

#ifdef _WIN32
  #include "src/crypto.h"
  #include "src/host_sbs_shader_cache.h"
  #include <algorithm>
  #include <array>
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
  constexpr UINT component_invalid = std::numeric_limits<UINT>::max();
  struct alignas(16) component_constants_t { std::array<UINT, 4u> shape; };
  static_assert(sizeof(component_constants_t) == 16u);

  std::vector<char> component_bytes(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0) throw std::runtime_error("Cannot read component input: " + path.string());
    std::vector<char> result(static_cast<std::size_t>(file.tellg())); file.seekg(0);
    if (!result.empty()) { file.read(result.data(), static_cast<std::streamsize>(result.size())); }
    if (!file) throw std::runtime_error("Incomplete component input");
    return result;
  }
  std::string component_sha(const void *data, std::size_t bytes) {
    const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), bytes));
    constexpr char digits[] = "0123456789abcdef"; std::string result;
    for (const auto byte : digest) { result.push_back(digits[byte >> 4u]); result.push_back(digits[byte & 15u]); }
    return result;
  }
  std::vector<UINT> component_bound(const nlohmann::json &binding, std::size_t words) {
    const auto raw = component_bytes(binding.at("path").get<std::string>());
    if (raw.size() != words * 4u || raw.size() != binding.at("bytes").get<std::size_t>() ||
        component_sha(raw.data(), raw.size()) != binding.at("sha256").get<std::string>()) throw std::runtime_error("Component binding mismatch");
    std::vector<UINT> result(words); std::memcpy(result.data(), raw.data(), raw.size()); return result;
  }
  struct component_texture_t {
    ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11ShaderResourceView> srv; ComPtr<ID3D11UnorderedAccessView> uav;
  };
  bool component_texture(ID3D11Device *device, UINT width, UINT height, bool output, component_texture_t &out) {
    out = {}; D3D11_TEXTURE2D_DESC d {}; d.Width = width; d.Height = height; d.MipLevels = d.ArraySize = 1u;
    d.Format = DXGI_FORMAT_R32_UINT; d.SampleDesc.Count = 1u; d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | (output ? D3D11_BIND_UNORDERED_ACCESS : 0u);
    return SUCCEEDED(device->CreateTexture2D(&d, nullptr, &out.texture)) &&
      SUCCEEDED(device->CreateShaderResourceView(out.texture.Get(), nullptr, &out.srv)) &&
      (!output || SUCCEEDED(device->CreateUnorderedAccessView(out.texture.Get(), nullptr, &out.uav)));
  }
  bool component_buffer(ID3D11Device *device, UINT words, ComPtr<ID3D11Buffer> &buffer, ComPtr<ID3D11UnorderedAccessView> &uav) {
    buffer.Reset(); uav.Reset(); D3D11_BUFFER_DESC d {};
    d.ByteWidth = words * 4u; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; d.StructureByteStride = 4u;
    return SUCCEEDED(device->CreateBuffer(&d, nullptr, &buffer)) && SUCCEEDED(device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
  }
  bool component_read_texture(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Texture2D *input, std::vector<UINT> &out) {
    D3D11_TEXTURE2D_DESC d {}; input->GetDesc(&d); const auto width = d.Width, height = d.Height;
    d.Usage = D3D11_USAGE_STAGING; d.BindFlags = d.MiscFlags = 0u; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging; if (FAILED(device->CreateTexture2D(&d, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), input); D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
    out.resize(std::size_t(width) * height);
    for (UINT y = 0u; y < height; ++y) { std::memcpy(out.data() + std::size_t(y) * width,
      static_cast<const char *>(mapped.pData) + std::size_t(y) * mapped.RowPitch, width * 4u); }
    context->Unmap(staging.Get(), 0u); return true;
  }
  bool component_read_buffer(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Buffer *input, std::vector<UINT> &out) {
    D3D11_BUFFER_DESC d {}; input->GetDesc(&d); const auto bytes = d.ByteWidth;
    d.Usage = D3D11_USAGE_STAGING; d.BindFlags = d.MiscFlags = d.StructureByteStride = 0u; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> staging; if (FAILED(device->CreateBuffer(&d, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), input); D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
    out.resize(bytes / 4u); std::memcpy(out.data(), mapped.pData, bytes); context->Unmap(staging.Get(), 0u); return true;
  }

  struct component_reference_t {
    std::vector<UINT> body, final_mask, parent, flags;
    std::size_t longest_bfs_distance = 0u;
  };
  component_reference_t component_bfs(UINT width, UINT height, const std::vector<UINT> &eligible,
      const std::vector<UINT> &seeds, const std::vector<UINT> &domain, const std::vector<UINT> &cover, bool box_geometry = false) {
    // Independent queue traversal, not union-find or an imported candidate helper.
    const auto n = std::size_t(width) * height; component_reference_t result;
    result.body.assign(n, 0u); result.final_mask.assign(n, 0u);
    result.parent.assign(n + 1u, component_invalid); result.parent[0u] = 0u; result.flags.assign(n + 1u, 0u);
    std::vector<unsigned char> visited(n); std::vector<std::size_t> queue, distance(n);
    constexpr std::array<std::array<int, 2u>, 4u> directions {{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
    for (std::size_t start = 0u; start < n; ++start) {
      if (domain[start] == 0u || eligible[start] != 1u || visited[start]) continue;
      queue.clear(); queue.push_back(start); visited[start] = 1u; distance[start] = 0u;
      UINT flags = 0u, root = static_cast<UINT>(start + 1u);
      for (std::size_t head = 0u; head < queue.size(); ++head) {
        const auto current = queue[head]; const int x = int(current % width), y = int(current / width);
        root = std::min(root, static_cast<UINT>(current + 1u)); if (seeds[current] == 1u) flags |= 1u;
        result.longest_bfs_distance = std::max(result.longest_bfs_distance, distance[current]);
        for (const auto &d : directions) {
          const int nx = x + d[0u], ny = y + d[1u];
          if (nx < 0 || ny < 0 || nx >= int(width) || ny >= int(height)) { flags |= 2u; continue; }
          const auto next = std::size_t(ny) * width + UINT(nx);
          if (domain[next] != domain[current]) { flags |= 2u; continue; }
          if (eligible[next] == 1u && !visited[next]) { visited[next] = 1u; distance[next] = distance[current] + 1u; queue.push_back(next); }
        }
      }
      result.flags[root] = flags;
      for (const auto current : queue) { result.parent[current + 1u] = root; result.body[current] = flags == 1u ? 1u : 0u; }
    }
    for (UINT y = 0u; y < height; ++y) {
      for (UINT x = 0u; x < width; ++x) {
        const auto p = std::size_t(y) * width + x; if (cover[p] != 1u) continue;
        for (int dy = -3; dy <= 3; ++dy) {
          for (int dx = -3; dx <= 3; ++dx) {
            const int nx = int(x) + dx, ny = int(y) + dy;
            if ((box_geometry || dx * dx + dy * dy <= 9) && nx >= 0 && ny >= 0 && nx < int(width) && ny < int(height) &&
                result.body[std::size_t(ny) * width + UINT(nx)] == 1u) result.final_mask[p] = 1u;
          }
        }
      }
    }
    return result;
  }

  class HostSbsSubtitleSeededComponentsGpuTest: public ::testing::Test {
  protected:
    void SetUp() override {
      shader_root_ = SUNSHINE_SHADERS_DIR; std::string expected;
      const auto name = std::string(::testing::UnitTest::GetInstance()->current_test_info()->name());
      if (name == "SavedNativeSeededComponentsReplay") {
        const auto *path = std::getenv("SUNSHINE_TEST_SUBTITLE_COMPONENTS_CASES");
        if (!path) { GTEST_SKIP() << "Explicit component replay input required"; }
        const auto bytes = component_bytes(path); const auto spec = nlohmann::json::parse(bytes.begin(), bytes.end());
        driver_ = spec.value("driver", std::string("warp")); expected = spec.at("source_closure_sha256").get<std::string>();
        geometry_ = spec.value("geometry", std::string("circle3"));
        shader_root_ = std::filesystem::path(path).parent_path() / spec.at("shader_root").get<std::string>();
      } else {
        if (const auto *root = std::getenv("SUNSHINE_TEST_SUBTITLE_COMPONENTS_SHADER_ROOT")) shader_root_ = root;
        if (const auto *driver = std::getenv("SUNSHINE_TEST_SUBTITLE_COMPONENTS_DRIVER")) driver_ = driver;
        if (const auto *geometry = std::getenv("SUNSHINE_TEST_SUBTITLE_COMPONENTS_GEOMETRY")) geometry_ = geometry;
        const auto *pin = std::getenv("SUNSHINE_TEST_SUBTITLE_COMPONENTS_SHADER_PIN");
        if (!pin) { GTEST_SKIP() << "Explicit external source-closure pin required for research component tests"; }
        expected = pin;
      }
      ASSERT_EQ(expected.size(), 64u); ASSERT_TRUE(driver_ == "warp" || driver_ == "hardware");
      ASSERT_TRUE(geometry_ == "circle3" || geometry_ == "box3");
      const D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0}; D3D_FEATURE_LEVEL actual {};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, driver_ == "hardware" ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP,
        nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &device_, &actual, &context_))); ASSERT_EQ(actual, D3D_FEATURE_LEVEL_11_0);
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "init_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "union_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "reduce_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "resolve_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "dilate_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_seeded_components_cs.hlsl", "dilate_box_main", "cs_5_0"}
      };
      const auto sources = cache::snapshot_sources(shader_root_, specs); closure_ = cache::source_closure_sha256(sources); ASSERT_EQ(closure_, expected);
      for (std::size_t i = 0u; i < specs.size(); ++i) {
        const auto code = cache::get(sources, specs[i]); ASSERT_TRUE(code && !code->empty());
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &shaders_[i])));
      }
      D3D11_BUFFER_DESC d {}; d.ByteWidth = sizeof(component_constants_t); d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &constants_buffer_)));
      d.ByteWidth = 12u; d.BindFlags = 0u; d.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &arguments_))); ASSERT_TRUE(allocate(21u, 21u));
    }
    void TearDown() override { unbind(); }
    void unbind() {
      if (!context_) return;
      const std::array<ID3D11ShaderResourceView *, 4u> reads {}; const std::array<ID3D11UnorderedAccessView *, 4u> writes {};
      context_->CSSetShaderResources(0u, 4u, reads.data()); context_->CSSetUnorderedAccessViews(0u, 4u, writes.data(), nullptr);
    }
    bool allocate(UINT width, UINT height) {
      unbind(); width_ = width; height_ = height;
      if (width == 0u || height == 0u || std::uint64_t(width) * height > 32u * 1024u * 1024u) return false;
      const auto n = width * height; constants_ = {{width, height, n, 0u}};
      eligible_.assign(n, 0u); seeds_.assign(n, 0u); domain_.assign(n, 7u); cover_.assign(n, 1u);
      return component_texture(device_.Get(), width, height, false, eligible_texture_) && component_texture(device_.Get(), width, height, false, seeds_texture_) &&
        component_texture(device_.Get(), width, height, false, domain_texture_) && component_texture(device_.Get(), width, height, false, cover_texture_) &&
        component_texture(device_.Get(), width, height, true, body_texture_) && component_texture(device_.Get(), width, height, true, final_texture_) &&
        component_buffer(device_.Get(), n + 1u, parent_, parent_uav_) && component_buffer(device_.Get(), n + 1u, flags_, flags_uav_);
    }
    std::size_t pixel(UINT x, UINT y) const { return std::size_t(y) * width_ + x; }
    void box(UINT l, UINT t, UINT r, UINT b, std::vector<UINT> &plane, UINT value = 1u) {
      for (UINT y = t; y < b; ++y) { for (UINT x = l; x < r; ++x) { plane[pixel(x, y)] = value; } }
    }
    void dispatch_five() {
      const std::array<ID3D11ShaderResourceView *, 4u> reads {eligible_texture_.srv.Get(), seeds_texture_.srv.Get(), domain_texture_.srv.Get(), cover_texture_.srv.Get()};
      const std::array<ID3D11UnorderedAccessView *, 4u> writes {parent_uav_.Get(), flags_uav_.Get(), body_texture_.uav.Get(), final_texture_.uav.Get()};
      context_->CSSetShaderResources(0u, 4u, reads.data()); context_->CSSetUnorderedAccessViews(0u, 4u, writes.data(), nullptr);
      context_->CSSetConstantBuffers(0u, 1u, constants_buffer_.GetAddressOf());
      // UAV bindings remain distinct and ordered dispatches finish each whole-grid
      // phase before the next phase reads it. Body is read through its UAV only in
      // DILATE, never simultaneously rebound as an SRV or written by that phase.
      for (std::size_t phase = 0u; phase < 4u; ++phase) {
        context_->CSSetShader(shaders_[phase].Get(), nullptr, 0u); context_->DispatchIndirect(arguments_.Get(), 0u);
      }
      context_->CSSetShader(shaders_[geometry_ == "box3" ? 5u : 4u].Get(), nullptr, 0u); context_->DispatchIndirect(arguments_.Get(), 0u);
      unbind();
    }
    bool run(bool reuse = false) {
      unbind(); context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, &constants_, 0u, 0u);
      context_->UpdateSubresource(eligible_texture_.texture.Get(), 0u, nullptr, eligible_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(seeds_texture_.texture.Get(), 0u, nullptr, seeds_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(domain_texture_.texture.Get(), 0u, nullptr, domain_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(cover_texture_.texture.Get(), 0u, nullptr, cover_.data(), width_ * 4u, 0u);
      const std::array<UINT, 3u> args {reuse ? 0u : (width_ + 15u) / 16u, reuse ? 0u : (height_ + 15u) / 16u, reuse ? 0u : 1u};
      context_->UpdateSubresource(arguments_.Get(), 0u, nullptr, args.data(), 0u, 0u); dispatch_five();
      return component_read_texture(device_.Get(), context_.Get(), body_texture_.texture.Get(), body_) &&
        component_read_texture(device_.Get(), context_.Get(), final_texture_.texture.Get(), final_) &&
        component_read_buffer(device_.Get(), context_.Get(), parent_.Get(), parents_) && component_read_buffer(device_.Get(), context_.Get(), flags_.Get(), root_flags_);
    }
    void expect_exact() {
      const auto ref = component_bfs(width_, height_, eligible_, seeds_, domain_, cover_, geometry_ == "box3");
      EXPECT_EQ(body_, ref.body); EXPECT_EQ(final_, ref.final_mask); EXPECT_EQ(parents_, ref.parent); EXPECT_EQ(root_flags_, ref.flags);
    }
    nlohmann::json adapter_identity() {
      ComPtr<IDXGIDevice> dxgi; ComPtr<IDXGIAdapter> adapter; DXGI_ADAPTER_DESC d {};
      if (FAILED(device_.As(&dxgi)) || FAILED(dxgi->GetAdapter(&adapter)) || FAILED(adapter->GetDesc(&d))) return nullptr;
      std::wstring wide(d.Description); std::string description(wide.begin(), wide.end());
      return {{"description", description}, {"vendor_id", d.VendorId}, {"device_id", d.DeviceId},
        {"luid_low", d.AdapterLuid.LowPart}, {"luid_high", d.AdapterLuid.HighPart}, {"feature_level", "11_0"}};
    }
    nlohmann::json performance() {
      constexpr UINT warmup = 3u, repeats = 20u;
      // The current case already uploaded its exact E/Seeds/Domain/Cover/CB/args.
      // Every repetition includes INIT resetting all Parent/RootFlags/Body/Final;
      // no previous component membership or warm-start result skips any phase.
      unbind();
      for (UINT i = 0u; i < warmup; ++i) { dispatch_five(); }
      D3D11_QUERY_DESC d {D3D11_QUERY_TIMESTAMP_DISJOINT, 0u}; ComPtr<ID3D11Query> disjoint;
      if (FAILED(device_->CreateQuery(&d, &disjoint))) throw std::runtime_error("Component disjoint query creation failed");
      d.Query = D3D11_QUERY_TIMESTAMP; std::array<ComPtr<ID3D11Query>, repeats * 2u> queries;
      for (auto &q : queries) {
        if (FAILED(device_->CreateQuery(&d, &q))) throw std::runtime_error("Component timestamp creation failed");
      }
      context_->Begin(disjoint.Get());
      for (UINT i = 0u; i < repeats; ++i) {
        context_->End(queries[2u * i].Get());
        dispatch_five();
        context_->End(queries[2u * i + 1u].Get());
      }
      context_->End(disjoint.Get()); unbind(); context_->Flush();
      const auto wait = [&](ID3D11Query *q, void *value, UINT bytes) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20); HRESULT hr;
        while ((hr = context_->GetData(q, value, bytes, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE) {
          if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Component timestamp timeout");
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (FAILED(hr)) throw std::runtime_error("Component timestamp query failed");
      };
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT evidence {}; wait(disjoint.Get(), &evidence, sizeof(evidence));
      nlohmann::json ticks = nlohmann::json::array(), ns = nlohmann::json::array();
      for (UINT i = 0u; i < repeats; ++i) {
        UINT64 begin {}, end {};
        wait(queries[2u * i].Get(), &begin, sizeof(begin)); wait(queries[2u * i + 1u].Get(), &end, sizeof(end));
        // Serialize original UINT64 integers. Subtract before converting the
        // elapsed interval to double; absolute timestamps may exceed 2^53.
        ticks.push_back({begin, end});
        if (!evidence.Disjoint && evidence.Frequency != 0u && end >= begin) {
          ns.push_back(double(end - begin) * 1e9 / double(evidence.Frequency));
        }
      }
      nlohmann::json result {{"enabled", true}, {"warmup", warmup}, {"repeats", repeats},
        {"disjoint", evidence.Disjoint != FALSE}, {"frequency_hz", evidence.Frequency},
        {"timestamp_ticks", ticks}, {"gpu_ns", ns}, {"valid_intervals", ns.size()},
        {"geometry", geometry_}, {"width", width_}, {"height", height_},
        {"dispatch_groups", {(width_ + 15u) / 16u, (height_ + 15u) / 16u, 1u}},
        {"ordered_passes", {"init_main", "union_main", "reduce_main", "resolve_main", geometry_ == "box3" ? "dilate_box_main" : "dilate_main"}},
        {"scratch_reset_inside_every_timed_interval", true},
        {"frequency_role", "GPU timestamp clock Hz, not GPU core frequency"},
        {"scope", "Cache-warm complete five-dispatch seeded-component operator for one preuploaded binary mode/unit, including GPU command bindings. Uploads/readbacks/query polling, CPU BFS/composition, colour/palette/P/S/seed extraction, OCR/cohort/renderer/full pipeline excluded. No wall-clock throughput or complete native colour-port claim"}};
      if (ns.size() == repeats) {
        auto values = ns.get<std::vector<double>>(); std::sort(values.begin(), values.end());
        double sum = 0.0; for (auto value : values) { sum += value; }
        result["gpu_ns_summary"] = {{"minimum", values.front()}, {"median", .5 * (values[9u] + values[10u])}, {"mean", sum / repeats}, {"maximum", values.back()}};
      }
      return result;
    }
    ComPtr<ID3D11Device> device_; ComPtr<ID3D11DeviceContext> context_;
    std::array<ComPtr<ID3D11ComputeShader>, 6u> shaders_; ComPtr<ID3D11Buffer> constants_buffer_, arguments_, parent_, flags_;
    ComPtr<ID3D11UnorderedAccessView> parent_uav_, flags_uav_;
    component_texture_t eligible_texture_, seeds_texture_, domain_texture_, cover_texture_, body_texture_, final_texture_;
    component_constants_t constants_ {}; UINT width_ {}, height_ {};
    std::vector<UINT> eligible_, seeds_, domain_, cover_, body_, final_, parents_, root_flags_;
    std::filesystem::path shader_root_; std::string closure_, driver_ {"warp"}, geometry_ {"circle3"};
  };

  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, EmptyAndOutsideDomainInputsPublishZero) {
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(std::count(body_.begin(), body_.end(), 1u), 0);
    std::fill(eligible_.begin(), eligible_.end(), 1u); std::fill(seeds_.begin(), seeds_.end(), 1u); std::fill(domain_.begin(), domain_.end(), 0u);
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(std::count(final_.begin(), final_.end(), 1u), 0);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, ASelectedOnePixelSeedHasNoMinimumAreaAndExactDisk29) {
    geometry_ = "circle3";
    eligible_[pixel(10u, 10u)] = seeds_[pixel(10u, 10u)] = 1u; ASSERT_TRUE(run()); expect_exact();
    EXPECT_EQ(std::count(body_.begin(), body_.end(), 1u), 1); EXPECT_EQ(std::count(final_.begin(), final_.end(), 1u), 29);
    EXPECT_EQ(final_[pixel(12u, 12u)], 1u); EXPECT_EQ(final_[pixel(13u, 11u)], 0u); EXPECT_EQ(final_[pixel(13u, 10u)], 1u);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, SeedlessComponentsCannotBorrowFromAnotherIsland) {
    box(3u, 3u, 6u, 6u, eligible_); box(12u, 12u, 16u, 16u, eligible_); seeds_[pixel(4u, 4u)] = 1u;
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(body_[pixel(4u, 4u)], 1u); EXPECT_EQ(body_[pixel(13u, 13u)], 0u);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, BoundaryComponentsKeepDistinctRootsAndDiscardTheirSeeds) {
    box(0u, 2u, 4u, 5u, eligible_); box(0u, 14u, 4u, 18u, eligible_); box(9u, 8u, 12u, 11u, eligible_);
    seeds_[pixel(2u, 3u)] = seeds_[pixel(10u, 9u)] = 1u;
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(body_[pixel(2u, 3u)], 0u); EXPECT_EQ(body_[pixel(10u, 9u)], 1u);
    EXPECT_NE(parents_[pixel(2u, 3u) + 1u], parents_[pixel(2u, 15u) + 1u]);
    EXPECT_NE(parents_[pixel(2u, 3u) + 1u], 0u); EXPECT_EQ(root_flags_[parents_[pixel(2u, 3u) + 1u]], 3u);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, OneOrthogonalConnectionToDomainBorderCensorsTheWholeComponent) {
    box(5u, 7u, 11u, 13u, eligible_); seeds_[pixel(8u, 9u)] = 1u; ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(body_[pixel(8u, 9u)], 1u);
    for (UINT x = 0u; x < 5u; ++x) { eligible_[pixel(x, 9u)] = 1u; }
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(std::count(body_.begin(), body_.end(), 1u), 0);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, DiagonalContactDoesNotShareSeedOrBoundaryFlags) {
    for (UINT x = 0u; x <= 7u; ++x) { eligible_[pixel(x, 7u)] = 1u; }
    eligible_[pixel(8u, 8u)] = seeds_[pixel(8u, 8u)] = 1u;
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(body_[pixel(8u, 8u)], 1u); EXPECT_NE(parents_[pixel(7u, 7u) + 1u], parents_[pixel(8u, 8u) + 1u]);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, LongThinSerpentineHasNoSixtyFourStepLimit) {
    ASSERT_TRUE(allocate(129u, 129u));
    for (UINT y = 2u; y <= 126u; y += 2u) {
      for (UINT x = 2u; x <= 126u; ++x) { eligible_[pixel(x, y)] = 1u; }
      if (y < 126u) { eligible_[pixel(((y / 2u) % 2u) ? 126u : 2u, y + 1u)] = 1u; }
    }
    seeds_[pixel(2u, 2u)] = 1u; const auto ref = component_bfs(width_, height_, eligible_, seeds_, domain_, cover_);
    ASSERT_GT(ref.longest_bfs_distance, 64u); ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(body_[pixel(64u, 126u)], 1u);
    eligible_[pixel(1u, 2u)] = eligible_[pixel(0u, 2u)] = 1u; ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(body_[pixel(64u, 126u)], 0u);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, AdjacentDomainIdsDoNotJoinAndTheirInterfaceIsBoundary) {
    ASSERT_TRUE(allocate(32u, 21u)); box(16u, 0u, 32u, 21u, domain_, 8u); box(13u, 8u, 19u, 13u, eligible_);
    seeds_[pixel(14u, 10u)] = 1u; ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(body_[pixel(14u, 10u)], 0u);
    EXPECT_NE(parents_[pixel(15u, 10u) + 1u], parents_[pixel(16u, 10u) + 1u]);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, MultipleSeededIslandsRetainTheirWholeInteriors) {
    ASSERT_TRUE(allocate(41u, 31u)); box(3u, 3u, 12u, 11u, eligible_); box(25u, 18u, 36u, 27u, eligible_);
    seeds_[pixel(3u, 3u)] = seeds_[pixel(35u, 26u)] = 1u; ASSERT_TRUE(run()); expect_exact();
    EXPECT_EQ(std::count(body_.begin(), body_.end(), 1u), 9 * 8 + 11 * 9);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, SeedsOutsideEligibilityCannotCreateAuthority) {
    seeds_[pixel(10u, 10u)] = 1u; ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(std::count(final_.begin(), final_.end(), 1u), 0);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, DomainIdsAreOpaqueAndCoverClipsOnlyTheFixedMargin) {
    std::fill(domain_.begin(), domain_.end(), component_invalid); eligible_[pixel(10u, 10u)] = seeds_[pixel(10u, 10u)] = 1u;
    std::fill(cover_.begin(), cover_.end(), 0u); cover_[pixel(11u, 10u)] = 1u; ASSERT_TRUE(run()); expect_exact();
    EXPECT_EQ(body_[pixel(10u, 10u)], 1u); EXPECT_EQ(final_[pixel(10u, 10u)], 0u); EXPECT_EQ(std::count(final_.begin(), final_.end(), 1u), 1);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, NonbinaryBooleanAnywhereInvalidatesTheWholePublication) {
    for (UINT fault = 0u; fault < 3u; ++fault) {
      SCOPED_TRACE(fault); ASSERT_TRUE(allocate(21u, 21u)); std::fill(domain_.begin(), domain_.end(), 0u);
      if (fault == 0u) eligible_.back() = 2u;
      if (fault == 1u) seeds_.back() = 2u;
      if (fault == 2u) cover_.back() = 2u;
      ASSERT_TRUE(run()); EXPECT_EQ(parents_[0u], component_invalid);
      EXPECT_TRUE(std::all_of(body_.begin(), body_.end(), [](UINT value) { return value == 1u; }));
      EXPECT_TRUE(std::all_of(final_.begin(), final_.end(), [](UINT value) { return value == 1u; }));
    }
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, MalformedGeometryConservativelyReturnsRepresentableOnes) {
    for (UINT fault = 0u; fault < 10u; ++fault) {
      SCOPED_TRACE(fault); ASSERT_TRUE(allocate(21u, 21u));
      switch (fault) {
        case 0u: ASSERT_TRUE(component_texture(device_.Get(), width_ - 1u, height_, false, eligible_texture_)); break;
        case 1u: ASSERT_TRUE(component_texture(device_.Get(), width_, height_ - 1u, false, seeds_texture_)); break;
        case 2u: ASSERT_TRUE(component_texture(device_.Get(), width_, height_ - 1u, false, domain_texture_)); break;
        case 3u: ASSERT_TRUE(component_texture(device_.Get(), width_, height_ - 1u, false, cover_texture_)); break;
        case 4u: ASSERT_TRUE(component_buffer(device_.Get(), width_ * height_, parent_, parent_uav_)); break;
        case 5u: ASSERT_TRUE(component_buffer(device_.Get(), width_ * height_, flags_, flags_uav_)); break;
        case 6u: --constants_.shape[2u]; break;
        case 7u: constants_.shape[3u] = 1u; break;
        case 8u: constants_.shape[0u] = 0u; break;
        case 9u: ASSERT_TRUE(component_texture(device_.Get(), width_ - 1u, height_, true, final_texture_)); break;
      }
      ASSERT_TRUE(run()); EXPECT_TRUE(std::all_of(body_.begin(), body_.end(), [](UINT value) { return value == 1u; }));
      EXPECT_TRUE(std::all_of(final_.begin(), final_.end(), [](UINT value) { return value == 1u; }));
    }
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, FiveZeroDispatchesHoldBodyFinalRootsAndFlags) {
    box(5u, 5u, 12u, 12u, eligible_); seeds_[pixel(7u, 7u)] = 1u; ASSERT_TRUE(run());
    const auto previous_body = body_, previous_final = final_, previous_parents = parents_, previous_flags = root_flags_;
    std::fill(eligible_.begin(), eligible_.end(), 0u); std::fill(seeds_.begin(), seeds_.end(), 0u); std::fill(domain_.begin(), domain_.end(), 0u);
    ASSERT_TRUE(run(true)); EXPECT_EQ(body_, previous_body); EXPECT_EQ(final_, previous_final); EXPECT_EQ(parents_, previous_parents); EXPECT_EQ(root_flags_, previous_flags);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, FixedBox3IncludesDiagonalSupportWithoutRecursiveGrowth) {
    geometry_ = "box3"; eligible_[pixel(10u, 10u)] = seeds_[pixel(10u, 10u)] = 1u;
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(std::count(body_.begin(), body_.end(), 1u), 1);
    EXPECT_EQ(std::count(final_.begin(), final_.end(), 1u), 49);
    EXPECT_EQ(final_[pixel(13u, 12u)], 1u); EXPECT_EQ(final_[pixel(13u, 13u)], 1u); EXPECT_EQ(final_[pixel(14u, 10u)], 0u);
    std::fill(cover_.begin(), cover_.end(), 0u); cover_[pixel(13u, 12u)] = cover_[pixel(14u, 10u)] = 1u;
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(std::count(final_.begin(), final_.end(), 1u), 1); EXPECT_EQ(body_[pixel(10u, 10u)], 1u);
  }
  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, BoxAndCircleShareIdenticalBodyRootsAndFlags) {
    box(4u, 4u, 9u, 10u, eligible_); seeds_[pixel(6u, 6u)] = 1u;
    geometry_ = "circle3"; ASSERT_TRUE(run()); expect_exact();
    const auto circle_body = body_, circle_parents = parents_, circle_flags = root_flags_, circle_final = final_;
    geometry_ = "box3"; ASSERT_TRUE(run()); expect_exact();
    EXPECT_EQ(body_, circle_body); EXPECT_EQ(parents_, circle_parents); EXPECT_EQ(root_flags_, circle_flags);
    for (std::size_t p = 0u; p < final_.size(); ++p) { EXPECT_GE(final_[p], circle_final[p]); }
    EXPECT_GT(std::count(final_.begin(), final_.end(), 1u), std::count(circle_final.begin(), circle_final.end(), 1u));
  }

  TEST_F(HostSbsSubtitleSeededComponentsGpuTest, SavedNativeSeededComponentsReplay) {
    const auto *input_env = std::getenv("SUNSHINE_TEST_SUBTITLE_COMPONENTS_CASES"), *output_env = std::getenv("SUNSHINE_TEST_SUBTITLE_COMPONENTS_OUTPUT");
    if (!input_env || !output_env) { GTEST_SKIP() << "Explicit component replay paths required"; }
    const auto input_bytes = component_bytes(input_env); const auto spec = nlohmann::json::parse(input_bytes.begin(), input_bytes.end()); ASSERT_EQ(spec.at("schema"), 1u);
    const auto &cases = spec.at("cases"); ASSERT_GT(cases.size(), 0u); ASSERT_LE(cases.size(), 256u);
    const std::filesystem::path destination(output_env); ASSERT_FALSE(std::filesystem::exists(destination)); ASSERT_TRUE(std::filesystem::create_directories(destination));
    const auto input_sha = component_sha(input_bytes.data(), input_bytes.size());
    const auto *perf_env = std::getenv("SUNSHINE_TEST_SUBTITLE_COMPONENTS_PERF"); const bool perf = perf_env && std::string_view(perf_env) == "1";
    std::array<wchar_t, 32768u> executable_path {}; const auto executable_length = GetModuleFileNameW(nullptr, executable_path.data(), static_cast<DWORD>(executable_path.size()));
    nlohmann::json executable = nullptr;
    if (executable_length > 0u && executable_length < executable_path.size()) {
      const std::filesystem::path path(std::wstring(executable_path.data(), executable_length)); const auto bytes = component_bytes(path);
      executable = {{"path", path.string()}, {"sha256", component_sha(bytes.data(), bytes.size())}, {"bytes", bytes.size()}};
    }
    nlohmann::json report {{"schema", 1u}, {"driver", driver_}, {"geometry", geometry_}, {"adapter", adapter_identity()}, {"source_closure_sha256", closure_},
      {"input_specification_sha256", input_sha}, {"test_executable", executable}, {"cases", nlohmann::json::array()}, {"drawing_or_inference_authority", false}};
    for (std::size_t i = 0u; i < cases.size(); ++i) {
      const auto &item = cases.at(i); SCOPED_TRACE(item.at("name").get<std::string>());
      ASSERT_TRUE(allocate(item.at("width").get<UINT>(), item.at("height").get<UINT>()));
      const auto &artifacts = item.at("artifacts"); const auto n = std::size_t(width_) * height_;
      eligible_ = component_bound(artifacts.at("eligible"), n); seeds_ = component_bound(artifacts.at("seeds"), n);
      domain_ = component_bound(artifacts.at("domain_tag"), n); cover_ = component_bound(artifacts.at("cover_clip"), n);
      for (const auto *plane : {&eligible_, &seeds_, &cover_}) { ASSERT_TRUE(std::all_of(plane->begin(), plane->end(), [](UINT v) { return v <= 1u; })); }
      if (artifacts.contains("constants")) {
        const auto words = component_bound(artifacts.at("constants"), 4u); ASSERT_TRUE(std::equal(words.begin(), words.end(), constants_.shape.begin(), constants_.shape.end()));
      }
      ASSERT_TRUE(run()); ASSERT_NO_FATAL_FAILURE(expect_exact());
      const bool expected_body = artifacts.contains("expected_body"), expected_final = artifacts.contains("expected_final");
      if (expected_body) { ASSERT_EQ(body_, component_bound(artifacts.at("expected_body"), n)); }
      if (expected_final) { ASSERT_EQ(final_, component_bound(artifacts.at("expected_final"), n)); }
      const auto folder = destination / ("case_" + std::to_string(i)); ASSERT_TRUE(std::filesystem::create_directory(folder)); nlohmann::json saved = nlohmann::json::object();
      const auto save = [&](const std::string &name, const std::vector<UINT> &words) {
        std::ofstream f(folder / name, std::ios::binary); f.write(reinterpret_cast<const char *>(words.data()), std::streamsize(words.size() * 4u));
        if (!f) throw std::runtime_error("Component export failed");
        saved[name] = {{"byte_count", words.size() * 4u}, {"sha256", component_sha(words.data(), words.size() * 4u)}};
      };
      save("uploaded_eligible.u32", eligible_); save("uploaded_seeds.u32", seeds_); save("uploaded_domain_tag.u32", domain_); save("uploaded_cover_clip.u32", cover_);
      save("uploaded_constants.u32", std::vector<UINT>(constants_.shape.begin(), constants_.shape.end()));
      save("parent_roots.u32", parents_); save("root_flags.u32", root_flags_); save("body.u32", body_); save("final.u32", final_);
      nlohmann::json identity {{"schema", 1u}, {"case", item}, {"driver", driver_}, {"geometry", geometry_}, {"adapter", adapter_identity()}, {"shader_root", shader_root_.string()},
        {"source_closure_sha256", closure_}, {"input_specification_sha256", input_sha}, {"test_executable", executable}, {"artifacts", saved},
        {"independent_CPU_BFS_byte_exact", true}, {"expected_body_byte_exact", expected_body ? nlohmann::json(true) : nlohmann::json(nullptr)},
        {"expected_final_byte_exact", expected_final ? nlohmann::json(true) : nlohmann::json(nullptr)},
        {"parent_and_flags_are_literal_GPU_readback", true},
        {"parent_role", "N+1 uints: parent0 fault sentinel only; ineligible/domain0 MAX; eligible fully compressed minimum positive component ID"},
        {"root_flags_role", "At final positive roots: bit1 seed, bit2 domain-boundary; no virtual exterior merge or temporal age"},
        {"scope", "Exact seeded4CC and selected circle3disk29 or box3square49 primitive for one binary mode/current unit. No native colour extraction, palette, OCR, black witness, whole pipeline speed, drawing authority or production integration"}};
      if (perf) {
        // Preserve a byte-bound functional proof and all uploaded/readback planes
        // BEFORE timing mutates scratch again. Optional query failures cannot
        // rewrite these already verified functional artifacts.
        const auto proof_text = identity.dump(2) + "\n";
        std::ofstream proof_file(folder / "seeded_components_functional_identity.json", std::ios::binary);
        proof_file << proof_text; ASSERT_TRUE(proof_file.good()); proof_file.close();
        identity["functional_proof_before_performance"] = {{"filename", "seeded_components_functional_identity.json"},
          {"sha256", component_sha(proof_text.data(), proof_text.size())}, {"byte_count", proof_text.size()}};
        identity["performance"] = performance();
      }
      const auto text = identity.dump(2) + "\n"; std::ofstream f(folder / "seeded_components_fixture_identity.json", std::ios::binary); f << text; ASSERT_TRUE(f.good()); f.close();
      report["cases"].push_back({{"index", i}, {"name", item.at("name")}, {"identity_sha256", component_sha(text.data(), text.size())},
        {"independent_CPU_BFS_byte_exact", true}, {"expected_body_byte_exact", identity["expected_body_byte_exact"]}, {"expected_final_byte_exact", identity["expected_final_byte_exact"]},
        {"body_pixels", std::count(body_.begin(), body_.end(), 1u)}, {"final_pixels", std::count(final_.begin(), final_.end(), 1u)}});
      std::ofstream aggregate(destination / "seeded_components_replay_identity.json", std::ios::binary | std::ios::trunc); aggregate << report.dump(2) << '\n'; ASSERT_TRUE(aggregate.good());
    }
  }
}
#endif
