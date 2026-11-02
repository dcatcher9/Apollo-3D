/**
 * @file tests/unit/test_host_sbs_subtitle_topology_gpu.cpp
 * @brief Exact private four-connected per-domain fill; no drawing or inference authority.
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
  constexpr UINT topology_barrier = std::numeric_limits<UINT>::max();
  struct alignas(16) topology_constants_t { std::array<UINT, 4u> shape; };
  static_assert(sizeof(topology_constants_t) == 16u);

  std::vector<char> topology_bytes(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0) throw std::runtime_error("Cannot read topology input: " + path.string());
    std::vector<char> result(static_cast<std::size_t>(file.tellg())); file.seekg(0);
    if (!result.empty()) file.read(result.data(), static_cast<std::streamsize>(result.size()));
    if (!file) throw std::runtime_error("Incomplete topology input"); return result;
  }
  std::string topology_sha(const void *data, std::size_t bytes) {
    const auto digest = crypto::hash(std::string_view(static_cast<const char *>(data), bytes));
    constexpr char digits[] = "0123456789abcdef"; std::string result;
    for (const auto byte : digest) { result.push_back(digits[byte >> 4u]); result.push_back(digits[byte & 15u]); } return result;
  }
  std::vector<UINT> topology_bound(const nlohmann::json &binding, std::size_t words) {
    const auto raw = topology_bytes(binding.at("path").get<std::string>());
    if (raw.size() != words * 4u || raw.size() != binding.at("bytes").get<std::size_t>() ||
        topology_sha(raw.data(), raw.size()) != binding.at("sha256").get<std::string>()) throw std::runtime_error("Topology binding mismatch");
    std::vector<UINT> result(words); std::memcpy(result.data(), raw.data(), raw.size()); return result;
  }
  struct topology_texture_t {
    ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11ShaderResourceView> srv; ComPtr<ID3D11UnorderedAccessView> uav;
  };
  bool topology_texture(ID3D11Device *device, UINT width, UINT height, bool output, topology_texture_t &out) {
    out = {}; D3D11_TEXTURE2D_DESC d {}; d.Width = width; d.Height = height; d.MipLevels = d.ArraySize = 1u;
    d.Format = DXGI_FORMAT_R32_UINT; d.SampleDesc.Count = 1u; d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | (output ? D3D11_BIND_UNORDERED_ACCESS : 0u);
    return SUCCEEDED(device->CreateTexture2D(&d, nullptr, &out.texture)) &&
      SUCCEEDED(device->CreateShaderResourceView(out.texture.Get(), nullptr, &out.srv)) &&
      (!output || SUCCEEDED(device->CreateUnorderedAccessView(out.texture.Get(), nullptr, &out.uav)));
  }
  bool topology_parent(ID3D11Device *device, UINT words, ComPtr<ID3D11Buffer> &buffer, ComPtr<ID3D11UnorderedAccessView> &uav) {
    buffer.Reset(); uav.Reset(); D3D11_BUFFER_DESC d {};
    d.ByteWidth = words * 4u; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; d.StructureByteStride = 4u;
    return SUCCEEDED(device->CreateBuffer(&d, nullptr, &buffer)) && SUCCEEDED(device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
  }
  bool topology_read_texture(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Texture2D *input, std::vector<UINT> &out) {
    D3D11_TEXTURE2D_DESC d {}; input->GetDesc(&d); const auto width = d.Width, height = d.Height;
    d.Usage = D3D11_USAGE_STAGING; d.BindFlags = d.MiscFlags = 0u; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging; if (FAILED(device->CreateTexture2D(&d, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), input); D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
    out.resize(std::size_t(width) * height);
    for (UINT y = 0u; y < height; ++y) std::memcpy(out.data() + std::size_t(y) * width,
      static_cast<const char *>(mapped.pData) + std::size_t(y) * mapped.RowPitch, width * 4u);
    context->Unmap(staging.Get(), 0u); return true;
  }
  bool topology_read_parent(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Buffer *input, std::vector<UINT> &out) {
    D3D11_BUFFER_DESC d {}; input->GetDesc(&d); const auto bytes = d.ByteWidth;
    d.Usage = D3D11_USAGE_STAGING; d.BindFlags = d.MiscFlags = d.StructureByteStride = 0u; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> staging; if (FAILED(device->CreateBuffer(&d, nullptr, &staging))) return false;
    context->CopyResource(staging.Get(), input); D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
    out.resize(bytes / 4u); std::memcpy(out.data(), mapped.pData, bytes); context->Unmap(staging.Get(), 0u); return true;
  }

  struct topology_reference_t { std::vector<UINT> enclosed, parent; std::size_t longest_bfs_distance = 0u; };
  topology_reference_t topology_bfs(UINT width, UINT height, const std::vector<UINT> &known, const std::vector<UINT> &domain) {
    // Independent queue flood fill. No union-find, shader helper, GPU output, or fixed round cap.
    const auto n = std::size_t(width) * height; topology_reference_t result;
    result.enclosed.assign(n, 0u); result.parent.assign(n + 1u, 0u);
    std::vector<unsigned char> visited(n); std::vector<std::size_t> queue, distance(n);
    constexpr std::array<std::array<int, 2u>, 4u> directions {{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
    for (std::size_t p = 0u; p < n; ++p) {
      if (domain[p] == 0u) continue;
      if (known[p] == 1u) { result.enclosed[p] = 1u; result.parent[p + 1u] = topology_barrier; continue; }
      if (visited[p]) continue;
      queue.clear(); queue.push_back(p); visited[p] = 1u; distance[p] = 0u;
      bool exterior = false; UINT root = static_cast<UINT>(p + 1u);
      for (std::size_t head = 0u; head < queue.size(); ++head) {
        const auto current = queue[head]; const int x = int(current % width), y = int(current / width);
        root = std::min(root, static_cast<UINT>(current + 1u)); result.longest_bfs_distance = std::max(result.longest_bfs_distance, distance[current]);
        for (const auto &d : directions) {
          const int nx = x + d[0u], ny = y + d[1u];
          if (nx < 0 || ny < 0 || nx >= int(width) || ny >= int(height)) { exterior = true; continue; }
          const auto next = std::size_t(ny) * width + UINT(nx);
          if (domain[next] != domain[current]) { exterior = true; continue; }
          if (known[next] == 0u && !visited[next]) { visited[next] = 1u; distance[next] = distance[current] + 1u; queue.push_back(next); }
        }
      }
      for (const auto current : queue) { result.enclosed[current] = exterior ? 0u : 1u; result.parent[current + 1u] = exterior ? 0u : root; }
    }
    return result;
  }

  class HostSbsSubtitleTopologyGpuTest: public ::testing::Test {
  protected:
    void SetUp() override {
      shader_root_ = SUNSHINE_SHADERS_DIR; std::string expected;
      const auto name = std::string(::testing::UnitTest::GetInstance()->current_test_info()->name());
      if (name == "SavedNativeTopologyReplay") {
        const auto *path = std::getenv("SUNSHINE_TEST_SUBTITLE_TOPOLOGY_CASES");
        if (!path) { GTEST_SKIP() << "Explicit topology replay input required"; }
        const auto bytes = topology_bytes(path); const auto spec = nlohmann::json::parse(bytes.begin(), bytes.end());
        driver_ = spec.value("driver", std::string("warp")); expected = spec.at("source_closure_sha256").get<std::string>();
        shader_root_ = std::filesystem::path(path).parent_path() / spec.at("shader_root").get<std::string>();
      } else {
        if (const auto *root = std::getenv("SUNSHINE_TEST_SUBTITLE_TOPOLOGY_SHADER_ROOT")) shader_root_ = root;
        if (const auto *driver = std::getenv("SUNSHINE_TEST_SUBTITLE_TOPOLOGY_DRIVER")) driver_ = driver;
        const auto *pin = std::getenv("SUNSHINE_TEST_SUBTITLE_TOPOLOGY_SHADER_PIN");
        if (!pin) { GTEST_SKIP() << "Explicit external source-closure pin required for research topology tests"; }
        expected = pin;
      }
      ASSERT_EQ(expected.size(), 64u); ASSERT_TRUE(driver_ == "warp" || driver_ == "hardware");
      const D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0}; D3D_FEATURE_LEVEL actual {};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, driver_ == "hardware" ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP,
        nullptr, 0u, requested, 1u, D3D11_SDK_VERSION, &device_, &actual, &context_))); ASSERT_EQ(actual, D3D_FEATURE_LEVEL_11_0);
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {
        cache::shader_spec {"host_sbs_subtitle_topology_cs.hlsl", "init_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_topology_cs.hlsl", "union_main", "cs_5_0"},
        cache::shader_spec {"host_sbs_subtitle_topology_cs.hlsl", "resolve_main", "cs_5_0"}
      };
      const auto sources = cache::snapshot_sources(shader_root_, specs); closure_ = cache::source_closure_sha256(sources); ASSERT_EQ(closure_, expected);
      for (std::size_t i = 0u; i < specs.size(); ++i) {
        const auto code = cache::get(sources, specs[i]); ASSERT_TRUE(code && !code->empty());
        ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &shaders_[i])));
      }
      D3D11_BUFFER_DESC d {}; d.ByteWidth = sizeof(topology_constants_t); d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &constants_buffer_)));
      d.ByteWidth = 12u; d.BindFlags = 0u; d.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &arguments_))); ASSERT_TRUE(allocate(21u, 21u));
    }
    void TearDown() override { unbind(); }
    void unbind() {
      if (!context_) return;
      const std::array<ID3D11ShaderResourceView *, 2u> reads {}; const std::array<ID3D11UnorderedAccessView *, 2u> writes {};
      context_->CSSetShaderResources(0u, 2u, reads.data()); context_->CSSetUnorderedAccessViews(0u, 2u, writes.data(), nullptr);
    }
    bool allocate(UINT width, UINT height) {
      unbind(); width_ = width; height_ = height;
      if (width == 0u || height == 0u || std::uint64_t(width) * height > 32u * 1024u * 1024u) return false;
      const auto n = width * height; constants_ = {{width, height, n, 0u}};
      known_.assign(n, 0u); domain_.assign(n, 7u);
      return topology_texture(device_.Get(), width, height, false, known_texture_) && topology_texture(device_.Get(), width, height, false, domain_texture_) &&
        topology_texture(device_.Get(), width, height, true, enclosed_texture_) && topology_parent(device_.Get(), n + 1u, parent_, parent_uav_);
    }
    std::size_t pixel(UINT x, UINT y) const { return std::size_t(y) * width_ + x; }
    void ring(UINT l, UINT t, UINT r, UINT b) {
      for (UINT x = l; x <= r; ++x) known_[pixel(x, t)] = known_[pixel(x, b)] = 1u;
      for (UINT y = t; y <= b; ++y) known_[pixel(l, y)] = known_[pixel(r, y)] = 1u;
    }
    void dispatch_three() {
      const std::array<ID3D11ShaderResourceView *, 2u> reads {known_texture_.srv.Get(), domain_texture_.srv.Get()};
      const std::array<ID3D11UnorderedAccessView *, 2u> writes {parent_uav_.Get(), enclosed_texture_.uav.Get()};
      context_->CSSetShaderResources(0u, 2u, reads.data()); context_->CSSetUnorderedAccessViews(0u, 2u, writes.data(), nullptr);
      context_->CSSetConstantBuffers(0u, 1u, constants_buffer_.GetAddressOf());
      // Parent remains a UAV throughout. Ordered dispatch boundaries publish the
      // complete init and exact atomic-union results before their next readers.
      for (auto &shader : shaders_) { context_->CSSetShader(shader.Get(), nullptr, 0u); context_->DispatchIndirect(arguments_.Get(), 0u); }
    }
    bool run(bool zero_dispatch = false) {
      unbind(); context_->UpdateSubresource(known_texture_.texture.Get(), 0u, nullptr, known_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(domain_texture_.texture.Get(), 0u, nullptr, domain_.data(), width_ * 4u, 0u);
      context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, &constants_, 0u, 0u);
      const std::array<UINT, 3u> grid {zero_dispatch ? 0u : (width_ + 15u) / 16u, zero_dispatch ? 0u : (height_ + 15u) / 16u, zero_dispatch ? 0u : 1u};
      context_->UpdateSubresource(arguments_.Get(), 0u, nullptr, grid.data(), 0u, 0u); dispatch_three(); unbind();
      return topology_read_texture(device_.Get(), context_.Get(), enclosed_texture_.texture.Get(), enclosed_) &&
        topology_read_parent(device_.Get(), context_.Get(), parent_.Get(), parents_);
    }
    void expect_exact() {
      const auto reference = topology_bfs(width_, height_, known_, domain_);
      ASSERT_EQ(enclosed_, reference.enclosed) << "Independent per-domain four-neighbor flood fill differs";
      ASSERT_EQ(parents_, reference.parent) << "Exact exterior/minimum-component roots or barrier roles differ";
      ASSERT_EQ(parents_[0u], 0u);
      for (std::size_t p = 0u; p < known_.size(); ++p) {
        const auto root = parents_[p + 1u];
        if (root != 0u && root != topology_barrier) {
          ASSERT_LE(root, p + 1u); ASSERT_LT(root, parents_.size()); ASSERT_EQ(parents_[root], root);
          ASSERT_EQ(domain_[root - 1u], domain_[p]); ASSERT_EQ(known_[root - 1u], 0u);
        }
      }
    }
    nlohmann::json adapter_identity() const {
      ComPtr<IDXGIDevice> dxgi; ComPtr<IDXGIAdapter> adapter; DXGI_ADAPTER_DESC d {};
      if (FAILED(device_.As(&dxgi)) || FAILED(dxgi->GetAdapter(&adapter)) || FAILED(adapter->GetDesc(&d))) return {{"available", false}};
      std::string name; for (auto ch : d.Description) { if (ch == 0) break; name.push_back(ch <= 127 ? char(ch) : '?'); }
      return {{"description", name}, {"vendor_id", d.VendorId}, {"device_id", d.DeviceId}, {"adapter_luid_low", d.AdapterLuid.LowPart}, {"adapter_luid_high", d.AdapterLuid.HighPart}};
    }
    nlohmann::json performance() {
      constexpr UINT warmup = 3u, repeats = 20u;
      unbind(); for (UINT i = 0u; i < warmup; ++i) dispatch_three();
      D3D11_QUERY_DESC d {D3D11_QUERY_TIMESTAMP_DISJOINT, 0u}; ComPtr<ID3D11Query> disjoint;
      if (FAILED(device_->CreateQuery(&d, &disjoint))) throw std::runtime_error("Topology disjoint query creation failed");
      d.Query = D3D11_QUERY_TIMESTAMP; std::array<ComPtr<ID3D11Query>, repeats * 2u> queries;
      for (auto &q : queries) if (FAILED(device_->CreateQuery(&d, &q))) throw std::runtime_error("Topology timestamp creation failed");
      context_->Begin(disjoint.Get());
      for (UINT i = 0u; i < repeats; ++i) { context_->End(queries[2u * i].Get()); dispatch_three(); context_->End(queries[2u * i + 1u].Get()); }
      context_->End(disjoint.Get()); unbind(); context_->Flush();
      const auto wait = [&](ID3D11Query *q, void *value, UINT bytes) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20); HRESULT hr;
        while ((hr = context_->GetData(q, value, bytes, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE) {
          if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Topology timestamp timeout");
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (FAILED(hr)) throw std::runtime_error("Topology timestamp query failed");
      };
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT evidence {}; wait(disjoint.Get(), &evidence, sizeof(evidence));
      nlohmann::json ticks = nlohmann::json::array(), ns = nlohmann::json::array();
      for (UINT i = 0u; i < repeats; ++i) {
        UINT64 begin {}, end {}; wait(queries[2u * i].Get(), &begin, sizeof(begin)); wait(queries[2u * i + 1u].Get(), &end, sizeof(end)); ticks.push_back({begin, end});
        if (!evidence.Disjoint && evidence.Frequency != 0u && end >= begin) ns.push_back(double(end - begin) * 1e9 / double(evidence.Frequency));
      }
      nlohmann::json result {{"enabled", true}, {"warmup", warmup}, {"repeats", repeats}, {"disjoint", evidence.Disjoint != FALSE},
        {"frequency_hz", evidence.Frequency}, {"timestamp_ticks", ticks}, {"gpu_ns", ns},
        {"scope", "Cache-warm complete init/union/resolve GPU primitive including dispatch bindings; uploads/readbacks and CPU BFS excluded; no cohort/OCR/pipeline throughput claim"}};
      if (ns.size() == repeats) {
        auto values = ns.get<std::vector<double>>(); std::sort(values.begin(), values.end()); double sum = 0.0; for (auto value : values) sum += value;
        result["gpu_ns_summary"] = {{"minimum", values.front()}, {"median", .5 * (values[9u] + values[10u])}, {"mean", sum / repeats}, {"maximum", values.back()}};
      }
      return result;
    }
    ComPtr<ID3D11Device> device_; ComPtr<ID3D11DeviceContext> context_;
    std::array<ComPtr<ID3D11ComputeShader>, 3u> shaders_; ComPtr<ID3D11Buffer> constants_buffer_, arguments_, parent_;
    ComPtr<ID3D11UnorderedAccessView> parent_uav_; topology_texture_t known_texture_, domain_texture_, enclosed_texture_;
    topology_constants_t constants_ {}; UINT width_ {}, height_ {};
    std::vector<UINT> known_, domain_, enclosed_, parents_; std::filesystem::path shader_root_; std::string closure_, driver_ {"warp"};
  };

  TEST_F(HostSbsSubtitleTopologyGpuTest, ZeroOneAndOutsideDomainsHaveExactDistinctRoles) {
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(std::count(enclosed_.begin(), enclosed_.end(), 1u), 0);
    std::fill(known_.begin(), known_.end(), 1u); ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(std::count(enclosed_.begin(), enclosed_.end(), 1u), width_ * height_);
    std::fill(domain_.begin(), domain_.end(), 0u); ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(std::count(enclosed_.begin(), enclosed_.end(), 1u), 0);
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, ClosedOutlineEnclosesWhiteInteriorWithoutColorInput) {
    ring(4u, 4u, 12u, 12u); ASSERT_TRUE(run()); expect_exact();
    EXPECT_EQ(known_[pixel(8u, 8u)], 0u); EXPECT_EQ(enclosed_[pixel(8u, 8u)], 1u); EXPECT_EQ(std::count(enclosed_.begin(), enclosed_.end(), 1u), 81);
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, OneOrthogonalBreakOpensTheWholeRing) {
    ring(4u, 4u, 12u, 12u); known_[pixel(8u, 4u)] = 0u; ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(enclosed_[pixel(8u, 8u)], 0u);
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, ACornerGapIsNotAnEightConnectedEscape) {
    ring(4u, 4u, 6u, 6u); known_[pixel(4u, 4u)] = 0u; ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(enclosed_[pixel(5u, 5u)], 1u);
    known_[pixel(5u, 4u)] = 0u; ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(enclosed_[pixel(5u, 5u)], 0u);
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, LongThinSerpentineHasNoSixtyFourStepPropagationLimit) {
    ASSERT_TRUE(allocate(129u, 129u)); std::fill(known_.begin(), known_.end(), 1u);
    for (UINT y = 2u; y <= 126u; y += 2u) {
      for (UINT x = 1u; x <= 126u; ++x) known_[pixel(x, y)] = 0u;
      if (y < 126u) known_[pixel(((y / 2u) % 2u) ? 126u : 1u, y + 1u)] = 0u;
    }
    known_[pixel(0u, 2u)] = 0u;
    const auto reference = topology_bfs(width_, height_, known_, domain_); ASSERT_GT(reference.longest_bfs_distance, 64u);
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(enclosed_[pixel(64u, 126u)], 0u);
    known_[pixel(0u, 2u)] = 1u; ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(enclosed_[pixel(64u, 126u)], 1u);
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, AdjacentDomainsCannotLendClosureAcrossTheirBoundary) {
    ASSERT_TRUE(allocate(32u, 17u));
    for (UINT y = 0u; y < height_; ++y) for (UINT x = 16u; x < width_; ++x) domain_[pixel(x, y)] = 8u;
    ring(12u, 4u, 20u, 12u); ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(enclosed_[pixel(14u, 8u)], 0u); EXPECT_EQ(enclosed_[pixel(18u, 8u)], 0u);
    for (UINT y = 4u; y <= 12u; ++y) known_[pixel(15u, y)] = known_[pixel(16u, y)] = 1u;
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(enclosed_[pixel(14u, 8u)], 1u); EXPECT_EQ(enclosed_[pixel(18u, 8u)], 1u);
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, MultipleIslandsAndNestedHolesRemainIndependent) {
    ASSERT_TRUE(allocate(41u, 31u)); ring(2u, 2u, 14u, 14u); ring(5u, 5u, 10u, 10u); ring(24u, 17u, 35u, 27u); known_[pixel(19u, 15u)] = 1u;
    ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(enclosed_[pixel(8u, 8u)], 1u); EXPECT_EQ(enclosed_[pixel(20u, 15u)], 0u);
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, KnownBoundaryComponentDoesNotBecomeAnOutsideRoot) {
    ring(0u, 0u, 8u, 8u); ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(enclosed_[pixel(0u, 0u)], 1u); EXPECT_EQ(enclosed_[pixel(4u, 4u)], 1u);
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, DomainIdsAreOpaqueAndDoNotBorrowOwnerRoleBits) {
    std::fill(domain_.begin(), domain_.end(), 0x80000000u); ring(4u, 4u, 12u, 12u); ASSERT_TRUE(run()); expect_exact();
    std::fill(domain_.begin(), domain_.end(), 0xffffffffu); ASSERT_TRUE(run()); expect_exact(); EXPECT_EQ(enclosed_[pixel(8u, 8u)], 1u);
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, NonbinaryKnownAnywhereFaultsTheWholePublication) {
    std::fill(domain_.begin(), domain_.end(), 0u); known_.back() = 2u; ASSERT_TRUE(run());
    EXPECT_TRUE(std::all_of(enclosed_.begin(), enclosed_.end(), [](UINT value) { return value == 1u; })); ASSERT_FALSE(parents_.empty()); EXPECT_EQ(parents_[0u], topology_barrier);
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, MalformedDimensionsOrConstantsConservativelyReturnOnes) {
    for (UINT fault = 0u; fault < 7u; ++fault) {
      SCOPED_TRACE(fault); ASSERT_TRUE(allocate(21u, 21u));
      switch (fault) {
        case 0u: ASSERT_TRUE(topology_texture(device_.Get(), width_ - 1u, height_, false, known_texture_)); break;
        case 1u: ASSERT_TRUE(topology_texture(device_.Get(), width_, height_ - 1u, false, domain_texture_)); break;
        case 2u: ASSERT_TRUE(topology_parent(device_.Get(), width_ * height_, parent_, parent_uav_)); break;
        case 3u: --constants_.shape[2u]; break;
        case 4u: constants_.shape[3u] = 1u; break;
        case 5u: constants_.shape[0u] = 0u; break;
        case 6u: ASSERT_TRUE(topology_texture(device_.Get(), width_ - 1u, height_, true, enclosed_texture_)); break;
      }
      ASSERT_TRUE(run()); EXPECT_TRUE(std::all_of(enclosed_.begin(), enclosed_.end(), [](UINT value) { return value == 1u; }));
    }
  }
  TEST_F(HostSbsSubtitleTopologyGpuTest, ThreeZeroDispatchesHoldBothOutputAndParent) {
    ring(4u, 4u, 12u, 12u); ASSERT_TRUE(run()); const auto saved_output = enclosed_, saved_parent = parents_;
    std::fill(known_.begin(), known_.end(), 0u); std::fill(domain_.begin(), domain_.end(), 0u); ASSERT_TRUE(run(true));
    EXPECT_EQ(enclosed_, saved_output); EXPECT_EQ(parents_, saved_parent);
  }

  TEST_F(HostSbsSubtitleTopologyGpuTest, SavedNativeTopologyReplay) {
    const auto *input_env = std::getenv("SUNSHINE_TEST_SUBTITLE_TOPOLOGY_CASES"), *output_env = std::getenv("SUNSHINE_TEST_SUBTITLE_TOPOLOGY_OUTPUT");
    if (!input_env || !output_env) { GTEST_SKIP() << "Explicit saved topology replay paths required"; }
    const auto input_bytes = topology_bytes(input_env); const auto spec = nlohmann::json::parse(input_bytes.begin(), input_bytes.end()); ASSERT_EQ(spec.at("schema"), 1u);
    const auto &cases = spec.at("cases"); ASSERT_GT(cases.size(), 0u); ASSERT_LE(cases.size(), 32u);
    const std::filesystem::path destination(output_env); ASSERT_FALSE(std::filesystem::exists(destination)); ASSERT_TRUE(std::filesystem::create_directories(destination));
    const auto input_sha = topology_sha(input_bytes.data(), input_bytes.size());
    const auto *perf_env = std::getenv("SUNSHINE_TEST_SUBTITLE_TOPOLOGY_PERF"); const bool perf = perf_env && std::string_view(perf_env) == "1";
    std::array<wchar_t, 32768u> executable_path {}; const auto executable_length = GetModuleFileNameW(nullptr, executable_path.data(), static_cast<DWORD>(executable_path.size()));
    nlohmann::json executable = nullptr;
    if (executable_length > 0u && executable_length < executable_path.size()) {
      const std::filesystem::path path(std::wstring(executable_path.data(), executable_length)); const auto bytes = topology_bytes(path);
      executable = {{"path", path.string()}, {"sha256", topology_sha(bytes.data(), bytes.size())}, {"bytes", bytes.size()}};
    }
    nlohmann::json report {{"schema", 1u}, {"driver", driver_}, {"adapter", adapter_identity()}, {"source_closure_sha256", closure_},
      {"input_specification_sha256", input_sha}, {"test_executable", executable}, {"cases", nlohmann::json::array()}, {"drawing_or_inference_authority", false}};
    for (std::size_t i = 0u; i < cases.size(); ++i) {
      const auto &item = cases.at(i); SCOPED_TRACE(item.at("name").get<std::string>());
      ASSERT_TRUE(allocate(item.at("width").get<UINT>(), item.at("height").get<UINT>()));
      const auto &artifacts = item.at("artifacts"); const auto n = std::size_t(width_) * height_;
      known_ = topology_bound(artifacts.at("known_barrier"), n); domain_ = topology_bound(artifacts.at("domain_tag"), n);
      ASSERT_TRUE(std::all_of(known_.begin(), known_.end(), [](UINT value) { return value <= 1u; }));
      if (artifacts.contains("constants")) {
        const auto words = topology_bound(artifacts.at("constants"), 4u); ASSERT_TRUE(std::equal(words.begin(), words.end(), constants_.shape.begin(), constants_.shape.end()));
      }
      ASSERT_TRUE(run()); ASSERT_NO_FATAL_FAILURE(expect_exact()); const bool expected_supplied = artifacts.contains("expected_enclosed");
      if (expected_supplied) { ASSERT_EQ(enclosed_, topology_bound(artifacts.at("expected_enclosed"), n)); }
      const auto folder = destination / ("case_" + std::to_string(i)); ASSERT_TRUE(std::filesystem::create_directory(folder)); nlohmann::json saved = nlohmann::json::object();
      const auto save = [&](const std::string &name, const std::vector<UINT> &words) {
        std::ofstream f(folder / name, std::ios::binary); f.write(reinterpret_cast<const char *>(words.data()), std::streamsize(words.size() * 4u));
        if (!f) throw std::runtime_error("Topology export failed"); saved[name] = {{"byte_count", words.size() * 4u}, {"sha256", topology_sha(words.data(), words.size() * 4u)}};
      };
      save("uploaded_known_barrier.u32", known_); save("uploaded_domain_tag.u32", domain_);
      save("uploaded_constants.u32", std::vector<UINT>(constants_.shape.begin(), constants_.shape.end())); save("parent_roots.u32", parents_); save("enclosed.u32", enclosed_);
      nlohmann::json identity {{"schema", 1u}, {"case", item}, {"driver", driver_}, {"adapter", adapter_identity()}, {"shader_root", shader_root_.string()},
        {"source_closure_sha256", closure_}, {"input_specification_sha256", input_sha}, {"test_executable", executable}, {"artifacts", saved},
        {"independent_CPU_BFS_byte_exact", true}, {"expected_enclosed_supplied", expected_supplied},
        {"expected_enclosed_byte_exact", expected_supplied ? nlohmann::json(true) : nlohmann::json(nullptr)},
        {"parent_roots_are_literal_GPU_readback", true}, {"parent_role", "N+1 uints: outside root0, positive-domain knownMAX, free final minimum component root; no KNOWN/UNKNOWN age role"},
        {"scope", "Pure exact4-connected per-domain fill primitive. No source detection, glyph inference, draw mask, production integration or quality claim"}};
      if (perf) identity["performance"] = performance();
      const auto text = identity.dump(2) + "\n"; std::ofstream f(folder / "topology_fixture_identity.json", std::ios::binary); f << text; ASSERT_TRUE(f.good()); f.close();
      report["cases"].push_back({{"index", i}, {"name", item.at("name")}, {"identity_sha256", topology_sha(text.data(), text.size())}, {"independent_CPU_BFS_byte_exact", true},
        {"expected_enclosed_byte_exact", identity["expected_enclosed_byte_exact"]}, {"enclosed_pixels", std::count(enclosed_.begin(), enclosed_.end(), 1u)}});
      std::ofstream aggregate(destination / "topology_replay_identity.json", std::ios::binary | std::ios::trunc); aggregate << report.dump(2) << '\n'; ASSERT_TRUE(aggregate.good());
    }
  }
}
#endif
