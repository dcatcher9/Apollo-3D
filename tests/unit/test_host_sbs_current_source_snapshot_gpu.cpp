// Private source transport tests: authentic GPU preprocessing parity, no glyph/draw oracle.
#include <gtest/gtest.h>
#ifdef _WIN32
  #include <algorithm>
  #include <bit>
  #include <cstdlib>
  #include <cstring>
  #include <filesystem>
  #include <fstream>
  #include <limits>
  #include <nlohmann/json.hpp>
  #include <src/crypto.h>
  #include <src/host_sbs_current_source_snapshot_gpu.h>
  #include <src/host_sbs_shader_cache.h>
  #include <stdexcept>
  #include <vector>

namespace {
  namespace source = models::research_current_source_snapshot_v1;
  using Microsoft::WRL::ComPtr;
  constexpr const char *closure_pin = "7d028552417c2003af85f90661b8958ac6f371cb65847a2ed2ecfd495e529f5e";

  struct pixel_t {
    float r, g, b, a;
  };

  struct result_t {
    std::vector<std::byte> image;
    source::receipt_t receipt {};
  };

  class HostSbsCurrentSourceSnapshotGpuTest: public ::testing::Test {
  protected:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    std::array<ComPtr<ID3D11ComputeShader>, 4> shaders;
    source::resources_t helper;

    static void hr(HRESULT h) {
      if (FAILED(h)) {
        throw std::runtime_error("Current-source fixture D3D failure");
      }
    }

    void SetUp() override {
      const auto *root = std::getenv("SUNSHINE_TEST_CURRENT_SOURCE_SHADER_ROOT");
      if (!root || !*root) {
        GTEST_SKIP() << "Explicit frozen external source root required";
      }
      const auto *driver = std::getenv("SUNSHINE_TEST_CURRENT_SOURCE_DRIVER");
      const bool hardware = driver && std::string(driver) == "hardware";
      ASSERT_TRUE(!driver || std::string(driver) == "warp" || std::string(driver) == "hardware");
      D3D_FEATURE_LEVEL got {}, wanted[] {D3D_FEATURE_LEVEL_11_0};
      hr(D3D11CreateDevice(nullptr, hardware ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP, nullptr, 0, wanted, 1, D3D11_SDK_VERSION, &device, &got, &context));
      ASSERT_EQ(got, D3D_FEATURE_LEVEL_11_0);
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {cache::shader_spec {"host_sbs_current_source_snapshot_cs.hlsl", "clear_main", "cs_5_0"}, cache::shader_spec {"host_sbs_current_source_snapshot_cs.hlsl", "snapshot_main", "cs_5_0"}, cache::shader_spec {"host_sbs_current_source_snapshot_cs.hlsl", "resolve_main", "cs_5_0"}, cache::shader_spec {"host_sbs_ocr_preprocess_cs.hlsl", "main", "cs_5_0"}};
      auto sources = cache::snapshot_sources(std::filesystem::path(root), specs);
      ASSERT_EQ(cache::source_closure_sha256(sources), closure_pin);
      for (std::size_t i = 0; i < specs.size(); ++i) {
        auto code = cache::get(sources, specs[i]);
        ASSERT_TRUE(code && !code->empty());
        hr(device->CreateComputeShader(code->data(), code->size(), nullptr, &shaders[i]));
      }
      RecordProperty("source_closure", closure_pin);
      RecordProperty("source_transport_only_not_current_ui_authority", true);
      RecordProperty("performance_qualified", false);
    }

    template<class T>
    ComPtr<ID3D11Buffer> cb(const T &v) {
      static_assert(sizeof(T) % 16 == 0);
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = sizeof(T);
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA init {&v, 0, 0};
      ComPtr<ID3D11Buffer> b;
      hr(device->CreateBuffer(&d, &init, &b));
      return b;
    }

    ComPtr<ID3D11ShaderResourceView> input(UINT w, UINT h, DXGI_FORMAT format, const void *data, UINT pitch) {
      D3D11_TEXTURE2D_DESC d {};
      d.Width = w;
      d.Height = h;
      d.MipLevels = d.ArraySize = d.SampleDesc.Count = 1;
      d.Format = format;
      d.Usage = D3D11_USAGE_IMMUTABLE;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA init {data, pitch, 0};
      ComPtr<ID3D11Texture2D> t;
      ComPtr<ID3D11ShaderResourceView> srv;
      hr(device->CreateTexture2D(&d, &init, &t));
      hr(device->CreateShaderResourceView(t.Get(), nullptr, &srv));
      return srv;
    }

    static source::constants_t constants(UINT w = 64, UINT h = 48, UINT top = 16) {
      source::constants_t c;
      c.control = {1, source::tag, 1, 0};
      c.source_rect = {0, 0, w, h};
      c.band_field = {top, h - top, w, h};
      c.content = {0, 0, w, h};
      c.current = {1, 0, 33334, 0};
      c.scope = {91, 0, 5, 0};
      c.domain = {7, 0, 31, 0};
      return c;
    }

    std::vector<std::byte> readTexture(ID3D11Texture2D *t, UINT pixel_bytes) {
      D3D11_TEXTURE2D_DESC d {};
      t->GetDesc(&d);
      d.Usage = D3D11_USAGE_STAGING;
      d.BindFlags = d.MiscFlags = 0;
      d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> stage;
      hr(device->CreateTexture2D(&d, nullptr, &stage));
      context->CopyResource(stage.Get(), t);
      D3D11_MAPPED_SUBRESOURCE map {};
      hr(context->Map(stage.Get(), 0, D3D11_MAP_READ, 0, &map));
      std::vector<std::byte> out(std::size_t(d.Width) * d.Height * pixel_bytes);
      for (UINT y = 0; y < d.Height; ++y) {
        std::memcpy(out.data() + std::size_t(y) * d.Width * pixel_bytes, static_cast<const char *>(map.pData) + std::size_t(y) * map.RowPitch, std::size_t(d.Width) * pixel_bytes);
      }
      context->Unmap(stage.Get(), 0);
      return out;
    }

    result_t read() {
      const auto v = helper.working_views();
      result_t r;
      r.image = readTexture(v.texture.Get(), 8);
      D3D11_BUFFER_DESC d {};
      v.receipt_buffer->GetDesc(&d);
      d.Usage = D3D11_USAGE_STAGING;
      d.BindFlags = d.MiscFlags = d.StructureByteStride = 0;
      d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Buffer> stage;
      hr(device->CreateBuffer(&d, nullptr, &stage));
      context->CopyResource(stage.Get(), v.receipt_buffer.Get());
      D3D11_MAPPED_SUBRESOURCE map {};
      hr(context->Map(stage.Get(), 0, D3D11_MAP_READ, 0, &map));
      std::memcpy(r.receipt.data(), map.pData, sizeof(r.receipt));
      context->Unmap(stage.Get(), 0);
      return r;
    }

    result_t run(ID3D11ShaderResourceView *capture, const source::constants_t &c) {
      std::string error;
      EXPECT_TRUE(helper.ensure(device.Get(), c.source_rect[2], c.band_field[1], error)) << error;
      EXPECT_TRUE(helper.dispatch(context.Get(), {shaders[0].Get(), shaders[1].Get(), shaders[2].Get()}, capture, c, error)) << error;
      return read();
    }

    void expectReady(const result_t &r, const source::constants_t &c) {
      EXPECT_EQ(r.receipt[0], source::tag);
      EXPECT_EQ(r.receipt[1], 1u);
      EXPECT_EQ(r.receipt[2], 1u);
      EXPECT_EQ(r.receipt[3], 0u);
      EXPECT_EQ(r.receipt[6], c.source_rect[2] * c.band_field[1]);
      EXPECT_EQ(r.receipt[44], 0u);
      EXPECT_EQ(r.receipt[45], 1u);
      EXPECT_EQ(std::memcmp(r.receipt.data() + 8, &c, sizeof(c)), 0);
      EXPECT_TRUE(std::all_of(r.receipt.begin() + 46, r.receipt.end(), [](auto v) {
        return v == 0;
      }));
    }

    std::vector<std::byte> productionBand(ID3D11ShaderResourceView *capture, const source::constants_t &c) {
      D3D11_TEXTURE2D_DESC td {};
      td.Width = c.source_rect[2];
      td.Height = c.band_field[1];
      td.MipLevels = td.ArraySize = td.SampleDesc.Count = 1;
      td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      td.Usage = D3D11_USAGE_DEFAULT;
      td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      ComPtr<ID3D11Texture2D> band;
      ComPtr<ID3D11UnorderedAccessView> band_uav;
      hr(device->CreateTexture2D(&td, nullptr, &band));
      hr(device->CreateUnorderedAccessView(band.Get(), nullptr, &band_uav));
      D3D11_BUFFER_DESC bd {};
      bd.ByteWidth = 960u * 160u * 3u * 4u;
      bd.Usage = D3D11_USAGE_DEFAULT;
      bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      bd.StructureByteStride = 4;
      ComPtr<ID3D11Buffer> buffer;
      ComPtr<ID3D11UnorderedAccessView> ocr;
      hr(device->CreateBuffer(&bd, nullptr, &buffer));
      hr(device->CreateUnorderedAccessView(buffer.Get(), nullptr, &ocr));
      std::array<UINT, 8> raw {c.source_rect[2], c.source_rect[3], c.band_field[0], c.band_field[1], c.control[3], c.source_rect[0], c.source_rect[1], 0};
      auto b = cb(raw);
      auto *p = b.Get();
      context->CSSetConstantBuffers(0, 1, &p);
      context->CSSetShaderResources(0, 1, &capture);
      ID3D11UnorderedAccessView *outputs[2] {ocr.Get(), band_uav.Get()};
      context->CSSetUnorderedAccessViews(0, 2, outputs, nullptr);
      context->CSSetShader(shaders[3].Get(), nullptr, 0);
      context->Dispatch(60, 10, 1);
      ID3D11ShaderResourceView *s = nullptr;
      ID3D11UnorderedAccessView *u[2] {};
      context->CSSetShaderResources(0, 1, &s);
      context->CSSetUnorderedAccessViews(0, 2, u, nullptr);
      return readTexture(band.Get(), 8);
    }

    static std::vector<std::byte> bound(const nlohmann::json &v, std::size_t expected) {
      std::ifstream f(v.at("path").get<std::string>(), std::ios::binary | std::ios::ate);
      if (!f || std::size_t(f.tellg()) != expected) {
        throw std::runtime_error("Current-source input size mismatch");
      }
      std::vector<std::byte> b(expected);
      f.seekg(0);
      f.read(reinterpret_cast<char *>(b.data()), b.size());
      if (!f) {
        throw std::runtime_error("Current-source input read failed");
      }
      auto digest = crypto::hash(std::string_view(reinterpret_cast<const char *>(b.data()), b.size()));
      constexpr char hex[] = "0123456789abcdef";
      std::string actual;
      for (auto x : digest) {
        actual += hex[x >> 4];
        actual += hex[x & 15];
      }
      if (actual != v.at("sha256").get<std::string>()) {
        throw std::runtime_error("Current-source input hash mismatch");
      }
      return b;
    }
  };

  TEST_F(HostSbsCurrentSourceSnapshotGpuTest, ByteExactProductionColorAndNativeCropForEveryColorMode) {
    std::vector<pixel_t> pixels(80 * 72);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
      pixels[i] = {float(i % 29) / 17.f, float(i % 13) / 9.f, float(i % 7) / 5.f, 1};
    }
    auto capture = input(80, 72, DXGI_FORMAT_R32G32B32A32_FLOAT, pixels.data(), 80 * sizeof(pixel_t));
    for (UINT mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(mode);
      auto c = constants();
      c.source_rect[0] = 7;
      c.source_rect[1] = 9;
      c.control[3] = mode;
      auto actual = run(capture.Get(), c);
      expectReady(actual, c);
      EXPECT_EQ(actual.image, productionBand(capture.Get(), c));
    }
  }

  TEST_F(HostSbsCurrentSourceSnapshotGpuTest, InvalidGeometryAndIdentityOverwriteEveryPixelWithZero) {
    std::vector<pixel_t> pixels(64 * 48, {1, .5f, .25f, 1});
    auto capture = input(64, 48, DXGI_FORMAT_R32G32B32A32_FLOAT, pixels.data(), 64 * sizeof(pixel_t));
    auto c = constants();
    expectReady(run(capture.Get(), c), c);
    for (UINT kind = 0; kind < 7; ++kind) {
      SCOPED_TRACE(kind);
      auto bad = c;
      if (kind == 0) {
        bad.source_rect[0] = 1;
      }
      if (kind == 1) {
        bad.control[1] ^= 1;
      }
      if (kind == 2) {
        bad.control[3] = 3;
      }
      if (kind == 3) {
        bad.current[2] = 0;
      }
      if (kind == 4) {
        bad.reserved[0] = 1;
      }
      if (kind == 5) {
        bad.band_field[0] = 15;
      }
      if (kind == 6) {
        bad.content[2] = 65;
      }
      auto actual = run(capture.Get(), bad);
      EXPECT_EQ(actual.receipt[2], 0u);
      EXPECT_NE(actual.receipt[3], 0u);
      EXPECT_TRUE(std::all_of(actual.image.begin(), actual.image.end(), [](auto x) {
        return x == std::byte {0};
      }));
    }
  }

  TEST_F(HostSbsCurrentSourceSnapshotGpuTest, NonfinitePixelRevokesWholeReceiptAndNextCaptureRecovers) {
    std::vector<pixel_t> pixels(64 * 48, {1, .5f, .25f, 1});
    auto c = constants();
    pixels[20 * 64 + 5].r = std::numeric_limits<float>::quiet_NaN();
    auto capture = input(64, 48, DXGI_FORMAT_R32G32B32A32_FLOAT, pixels.data(), 64 * sizeof(pixel_t));
    auto actual = run(capture.Get(), c);
    EXPECT_EQ(actual.receipt[2], 0u);
    EXPECT_EQ(actual.receipt[3], 2u);
    EXPECT_EQ(actual.receipt[44], 1u);
    EXPECT_EQ(actual.receipt[6], 64u * 32u - 1u);
    for (UINT i = 0; i < 8; ++i) {
      EXPECT_EQ(actual.image[(4 * 64 + 5) * 8 + i], std::byte {0});
    }
    pixels[20 * 64 + 5].r = 1;
    capture = input(64, 48, DXGI_FORMAT_R32G32B32A32_FLOAT, pixels.data(), 64 * sizeof(pixel_t));
    c.current = {2, 0, 66667, 0};
    expectReady(run(capture.Get(), c), c);
  }

  TEST_F(HostSbsCurrentSourceSnapshotGpuTest, OnlyCurrentBandPixelsAffectNumericReadiness) {
    std::vector<pixel_t> pixels(64 * 48, {.5f, .25f, .125f, 1});
    pixels[0].r = std::numeric_limits<float>::infinity();
    auto capture = input(64, 48, DXGI_FORMAT_R32G32B32A32_FLOAT, pixels.data(), 64 * sizeof(pixel_t));
    auto c = constants();
    expectReady(run(capture.Get(), c), c);
  }

  TEST_F(HostSbsCurrentSourceSnapshotGpuTest, EveryNewCaptureReplacesSourceAndOwnIdentityWithoutOldMask) {
    std::vector<pixel_t> pixels(64 * 48, {1, 0, 0, 1});
    auto c = constants();
    std::vector<std::byte> previous;
    for (UINT n = 1; n <= 3; ++n) {
      SCOPED_TRACE(n);
      if (n == 2) {
        for (auto &p : pixels) {
          p = {0, 1, 0, 1};
        }
      }
      if (n == 3) {
        for (auto &p : pixels) {
          p = {0, 0, 0, 1};
        }
      }
      auto capture = input(64, 48, DXGI_FORMAT_R32G32B32A32_FLOAT, pixels.data(), 64 * sizeof(pixel_t));
      c.current = {n, 0, n * 33334u, 0};
      auto actual = run(capture.Get(), c);
      expectReady(actual, c);
      if (n > 1) {
        EXPECT_NE(actual.image, previous);
      }
      previous = actual.image;
    }
  }

  TEST_F(HostSbsCurrentSourceSnapshotGpuTest, StableAllocationAndComputeSlotsArePreserved) {
    std::vector<pixel_t> pixels(64 * 48, {1, 0, 0, 1});
    auto capture = input(64, 48, DXGI_FORMAT_R32G32B32A32_FLOAT, pixels.data(), 64 * sizeof(pixel_t));
    auto c = constants();
    std::string error;
    ASSERT_TRUE(helper.ensure(device.Get(), 64, 32, error));
    auto first = helper.working_views();
    ASSERT_TRUE(helper.ensure(device.Get(), 64, 32, error));
    EXPECT_EQ(first.texture.Get(), helper.working_views().texture.Get());
    auto b = cb(c);
    auto *constant = b.Get();
    auto *srv = capture.Get();
    context->CSSetConstantBuffers(0, 1, &constant);
    context->CSSetShaderResources(0, 1, &srv);
    context->CSSetShader(shaders[3].Get(), nullptr, 0);
    ASSERT_TRUE(helper.dispatch(context.Get(), {shaders[0].Get(), shaders[1].Get(), shaders[2].Get()}, capture.Get(), c, error));
    ComPtr<ID3D11Buffer> gotb;
    ComPtr<ID3D11ShaderResourceView> gots;
    ComPtr<ID3D11ComputeShader> gotshader;
    context->CSGetConstantBuffers(0, 1, &gotb);
    context->CSGetShaderResources(0, 1, &gots);
    UINT count = 0;
    context->CSGetShader(&gotshader, nullptr, &count);
    EXPECT_EQ(gotb.Get(), b.Get());
    EXPECT_EQ(gots.Get(), capture.Get());
    EXPECT_EQ(gotshader.Get(), shaders[3].Get());
    expectReady(read(), c);
  }

  TEST_F(HostSbsCurrentSourceSnapshotGpuTest, SavedActualAThenBThenEmptyMatchesOwnedNativeHalfReferences) {
    const auto *path = std::getenv("SUNSHINE_TEST_CURRENT_SOURCE_CASES");
    ASSERT_TRUE(path && *path);
    std::ifstream file(path);
    nlohmann::json config;
    file >> config;
    ASSERT_EQ(config.at("schema"), 1);
    ASSERT_EQ(config.at("cases").size(), 3u);
    const auto *output = std::getenv("SUNSHINE_TEST_CURRENT_SOURCE_OUTPUT");
    ASSERT_TRUE(output && *output);
    const std::filesystem::path directory(output);
    ASSERT_FALSE(std::filesystem::exists(directory));
    ASSERT_TRUE(std::filesystem::create_directories(directory));
    auto write = [](const std::filesystem::path &p, const void *bytes, std::size_t size) {
      std::ofstream f(p, std::ios::binary);
      f.write(static_cast<const char *>(bytes), size);
      if (!f) {
        throw std::runtime_error("Current-source output write failed");
      }
    };
    UINT index = 0;
    for (const auto &row : config.at("cases")) {
      SCOPED_TRACE(row.at("name").get<std::string>());
      source::constants_t c;
      auto raw = row.at("constants").get<std::array<UINT, 32>>();
      std::memcpy(&c, raw.data(), sizeof(c));
      const UINT w = row.at("input_width"), h = row.at("input_height");
      auto bytes = bound(row.at("input"), std::size_t(w) * h * 4);
      auto expected = bound(row.at("expected"), std::size_t(c.source_rect[2]) * c.band_field[1] * 8);
      auto capture = input(w, h, DXGI_FORMAT_B8G8R8A8_UNORM, bytes.data(), w * 4u);
      auto actual = run(capture.Get(), c);
      expectReady(actual, c);
      auto reference = productionBand(capture.Get(), c);
      EXPECT_EQ(actual.image, expected);
      EXPECT_EQ(actual.image, reference);
      auto d = directory / ("case_" + std::to_string(++index));
      std::filesystem::create_directory(d);
      write(d / "uploaded-current.bgra8", bytes.data(), bytes.size());
      write(d / "uploaded-constants.u32", &c, sizeof(c));
      write(d / "actual-source.rgba16f", actual.image.data(), actual.image.size());
      write(d / "actual-receipt.u32", actual.receipt.data(), sizeof(actual.receipt));
      write(d / "production-preprocess-source.rgba16f", reference.data(), reference.size());
      std::ofstream identity(d / "identity.json");
      identity << nlohmann::json {{"input_case", row}, {"source_closure", closure_pin}, {"source_transport_only", true}, {"no_detector_or_empty_authority", true}}.dump(2) << '\n';
    }
  }

  TEST_F(HostSbsCurrentSourceSnapshotGpuTest, ActivePredicationIsRejectedWithoutRecordingOrClearing) {
    std::vector<pixel_t> pixels(64 * 48, {1, 0, 0, 1});
    auto capture = input(64, 48, DXGI_FORMAT_R32G32B32A32_FLOAT, pixels.data(), 64 * sizeof(pixel_t));
    auto c = constants();
    auto before = run(capture.Get(), c);
    D3D11_QUERY_DESC desc {D3D11_QUERY_OCCLUSION_PREDICATE, 0};
    ComPtr<ID3D11Predicate> predicate;
    hr(device->CreatePredicate(&desc, &predicate));
    context->SetPredication(predicate.Get(), FALSE);
    std::string error;
    EXPECT_FALSE(helper.dispatch(context.Get(), {shaders[0].Get(), shaders[1].Get(), shaders[2].Get()}, capture.Get(), c, error));
    context->SetPredication(nullptr, FALSE);
    auto after = read();
    EXPECT_EQ(before.image, after.image);
    EXPECT_EQ(before.receipt, after.receipt);
  }

  TEST_F(HostSbsCurrentSourceSnapshotGpuTest, AliasedMutableOutputCannotBeConsumedAsCurrentCapture) {
    std::vector<pixel_t> pixels(64 * 48, {1, 0, 0, 1});
    auto capture = input(64, 48, DXGI_FORMAT_R32G32B32A32_FLOAT, pixels.data(), 64 * sizeof(pixel_t));
    auto c = constants();
    auto before = run(capture.Get(), c);
    auto alias = helper.working_views().source;
    std::string error;
    EXPECT_FALSE(helper.dispatch(context.Get(), {shaders[0].Get(), shaders[1].Get(), shaders[2].Get()}, alias.Get(), c, error));
    auto after = read();
    EXPECT_EQ(before.image, after.image);
    EXPECT_EQ(before.receipt, after.receipt);
  }
}  // namespace
#endif
