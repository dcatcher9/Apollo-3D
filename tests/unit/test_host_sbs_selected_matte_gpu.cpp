#include <gtest/gtest.h>
#ifdef _WIN32
  #include "src/host_sbs_selected_matte_gpu.h"
  #include "src/host_sbs_shader_cache.h"

  #include <algorithm>
  #include <array>
  #include <cstdlib>
  #include <cstring>
  #include <d3d11.h>
  #include <stdexcept>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace selected = models::research_selected_matte_v1;

  class HostSbsSelectedMatteGpuTest: public testing::Test {
  protected:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11ComputeShader> shader;

    struct buffer_t {
      ComPtr<ID3D11Buffer> b;
      ComPtr<ID3D11ShaderResourceView> s;
    };

    struct texture_t {
      ComPtr<ID3D11Texture2D> t;
      ComPtr<ID3D11ShaderResourceView> s;
    };

    selected::resources_t output;
    buffer_t cb, root, args, sct, proof, receipt, frame;
    texture_t legacy, baseline, temporal;
    std::array<UINT, 24> constants {};
    std::array<UINT, 64> root_words {}, frame_words {};
    std::array<UINT, 32> identity {}, proof_words {};
    std::array<UINT, 256> temporal_words {};
    static constexpr UINT W = 8, H = 4;
    std::vector<UINT> lm, bm, tm;

    void require(bool v) {
      if (!v) {
        throw std::runtime_error("Selected-matte fixture API failed");
      }
    }

    buffer_t buffer(const void *p, UINT bytes, bool raw = false, bool indirect = false, bool constant = false) {
      buffer_t o;
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = bytes;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = constant ? D3D11_BIND_CONSTANT_BUFFER : D3D11_BIND_SHADER_RESOURCE;
      d.MiscFlags = constant ? 0u : (raw ? D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS : D3D11_RESOURCE_MISC_BUFFER_STRUCTURED);
      d.StructureByteStride = constant || raw ? 0u : 4u;
      if (indirect) {
        d.MiscFlags |= D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      }
      D3D11_SUBRESOURCE_DATA init {p, 0, 0};
      require(SUCCEEDED(device->CreateBuffer(&d, &init, &o.b)));
      if (!constant) {
        D3D11_SHADER_RESOURCE_VIEW_DESC s {};
        s.ViewDimension = raw ? D3D11_SRV_DIMENSION_BUFFEREX : D3D11_SRV_DIMENSION_BUFFER;
        s.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
        if (raw) {
          s.BufferEx.NumElements = bytes / 4u;
          s.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        } else {
          s.Buffer.NumElements = bytes / 4u;
        }
        require(SUCCEEDED(device->CreateShaderResourceView(o.b.Get(), &s, &o.s)));
      }
      return o;
    }

    texture_t texture(const std::vector<UINT> &values) {
      texture_t o;
      D3D11_TEXTURE2D_DESC d {};
      d.Width = W;
      d.Height = H;
      d.MipLevels = d.ArraySize = 1;
      d.Format = DXGI_FORMAT_R32_UINT;
      d.SampleDesc.Count = 1;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA init {values.data(), W * 4u, 0};
      require(SUCCEEDED(device->CreateTexture2D(&d, &init, &o.t)));
      require(SUCCEEDED(device->CreateShaderResourceView(o.t.Get(), nullptr, &o.s)));
      return o;
    }

    std::vector<UINT> read(ID3D11ShaderResourceView *s) {
      ComPtr<ID3D11Resource> r;
      s->GetResource(&r);
      D3D11_RESOURCE_DIMENSION dim;
      r->GetType(&dim);
      ComPtr<ID3D11Resource> stage;
      UINT row_bytes = 0, rows = 1;
      if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
        ComPtr<ID3D11Texture2D> t;
        require(SUCCEEDED(r.As(&t)));
        D3D11_TEXTURE2D_DESC d {};
        t->GetDesc(&d);
        row_bytes = d.Width * 4u;
        rows = d.Height;
        d.Usage = D3D11_USAGE_STAGING;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        d.BindFlags = d.MiscFlags = 0;
        ComPtr<ID3D11Texture2D> x;
        require(SUCCEEDED(device->CreateTexture2D(&d, nullptr, &x)));
        stage = x;
      } else {
        ComPtr<ID3D11Buffer> b;
        require(SUCCEEDED(r.As(&b)));
        D3D11_BUFFER_DESC d {};
        b->GetDesc(&d);
        row_bytes = d.ByteWidth;
        d.Usage = D3D11_USAGE_STAGING;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        d.BindFlags = d.MiscFlags = d.StructureByteStride = 0;
        ComPtr<ID3D11Buffer> x;
        require(SUCCEEDED(device->CreateBuffer(&d, nullptr, &x)));
        stage = x;
      }
      context->CopyResource(stage.Get(), r.Get());
      D3D11_MAPPED_SUBRESOURCE m {};
      require(SUCCEEDED(context->Map(stage.Get(), 0, D3D11_MAP_READ, 0, &m)));
      std::vector<UINT> v(row_bytes * rows / 4u);
      for (UINT y = 0; y < rows; ++y) {
        std::memcpy(reinterpret_cast<char *>(v.data()) + y * row_bytes, reinterpret_cast<const char *>(m.pData) + y * (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D ? m.RowPitch : row_bytes), row_bytes);
      }
      context->Unmap(stage.Get(), 0);
      return v;
    }

    void upload() {
      context->UpdateSubresource(cb.b.Get(), 0, nullptr, constants.data(), 0, 0);
      context->UpdateSubresource(root.b.Get(), 0, nullptr, root_words.data(), 0, 0);
      context->UpdateSubresource(args.b.Get(), 0, nullptr, root_words.data(), 0, 0);
      context->UpdateSubresource(sct.b.Get(), 0, nullptr, identity.data(), 0, 0);
      context->UpdateSubresource(proof.b.Get(), 0, nullptr, proof_words.data(), 0, 0);
      context->UpdateSubresource(receipt.b.Get(), 0, nullptr, temporal_words.data(), 0, 0);
      context->UpdateSubresource(frame.b.Get(), 0, nullptr, frame_words.data(), 0, 0);
    }

    void root_for(UINT mode) {
      root_words = {};
      root_words[0] = 1;
      root_words[1] = 1u ^ 0xd1ec15a5u ^ 0x52434f4fu;
      root_words[2] = 71;
      root_words[3] = 9;
      root_words[4] = 71u ^ 0xa3756c91u;
      root_words[5] = 9u ^ 0x5c8a936eu;
      root_words[6] = 0x47524243u;
      root_words[7] = 0x52434f4fu;
      std::copy_n(root_words.begin() + 2, 4, root_words.begin() + 8);
      root_words[12] = 0x54535152u;
      root_words[13] = 1;
      root_words[14] = 1u ^ 0x6f435257u;
      if (mode != 2u) {
        root_words[20] = root_words[21] = root_words[22] = 1u;
      }
      root_words[52] = root_words[53] = root_words[54] = 1u;
      root_words[60] = root_words[61] = root_words[62] = 1u;
    }

    selected::inputs_t inputs(UINT mode = 2u, bool with_temporal = false) {
      selected::inputs_t i;
      i.current_cb96 = cb.b.Get();
      i.actual_root_args = args.b.Get();
      i.legacy_matte = legacy.s.Get();
      i.baseline_matte = baseline.s.Get();
      i.temporal_matte = temporal.s.Get();
      i.numeric_proof = proof.s.Get();
      i.temporal_receipt = receipt.s.Get();
      i.sct = sct.s.Get();
      i.root = root.s.Get();
      i.frame256 = frame.s.Get();
      i.mode = static_cast<selected::postprocess_mode_e>(mode);
      i.expected_work = 1;
      i.transaction_token = (std::uint64_t(9) << 32u) | 71u;
      i.temporal_supplied = with_temporal;
      return i;
    }

    void queue(UINT mode = 2u, bool t = false) {
      std::string error;
      require(output.queue(context.Get(), shader.Get(), inputs(mode, t), error));
    }

    void expect(UINT kind, const std::vector<UINT> &mask, UINT fault = 0u) {
      auto v = output.views();
      EXPECT_EQ(read(v.matte.Get()), mask);
      auto p = read(v.receipt.Get());
      ASSERT_EQ(p.size(), 64u);
      EXPECT_EQ(p[0], selected::tag);
      EXPECT_EQ(p[1], 1u);
      EXPECT_EQ(p[2], kind);
      EXPECT_EQ(p[3], fault);
      EXPECT_TRUE(std::equal(constants.begin(), constants.end(), p.begin() + 4));
      if (!fault) {
        EXPECT_TRUE(std::equal(identity.begin(), identity.end(), p.begin() + 28));
      }
      EXPECT_EQ(p[60], 71u);
      EXPECT_EQ(p[61], 9u);
      EXPECT_EQ(p[62], 1u);
      EXPECT_EQ(p[63], 0u);
    }

    void SetUp() override {
      const char *enabled = std::getenv("SUNSHINE_SELECTED_MATTE_GPU_TEST");
      if (!enabled || std::string(enabled) != "1") {
        GTEST_SKIP() << "Explicit private selected-matte opt-in required";
      }
      const char *root_path = std::getenv("SUNSHINE_SELECTED_MATTE_SHADER_ROOT");
      const char *closure = std::getenv("SUNSHINE_SELECTED_MATTE_SOURCE_CLOSURE");
      ASSERT_TRUE(root_path && closure);
      const char *driver = std::getenv("SUNSHINE_SELECTED_MATTE_GPU_DRIVER");
      const bool hardware = driver && std::string(driver) == "hardware";
      if (driver) {
        ASSERT_TRUE(hardware || std::string(driver) == "warp");
      }
      D3D_FEATURE_LEVEL level {};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, hardware ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, &level, &context)));
      ASSERT_GE(level, D3D_FEATURE_LEVEL_11_0);
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {cache::shader_spec {"host_sbs_selected_matte_cs.hlsl", "select_main", "cs_5_0"}};
      const auto sources = cache::snapshot_sources(root_path, specs);
      ASSERT_TRUE(sources);
      ASSERT_EQ(cache::source_closure_sha256(sources), closure);
      const auto code = cache::get(sources, specs[0]);
      ASSERT_TRUE(code);
      ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(code->data(), code->size(), nullptr, &shader)));
      constants = {W, H, 0, H, 4, 2, 0, 2, 0, 0, 4, 2, 3, 0, 7, 0, 66667, 0, 0, 0, 1, 27, 0, 0};
      identity = {5, 0x35544353u, 1, 11, 3, 0, 7, 0, W, H, 0, H, 4, 2, 0, 2, 0, 0, 4, 2, 66667, 0, 27, 1, 0, 0, 0, 0, 0, 0, 0, 0};
      proof_words[0] = 0x31564c49u;
      proof_words[1] = proof_words[2] = 1;
      std::copy(constants.begin(), constants.end(), proof_words.begin() + 4);
      frame_words[0] = W;
      frame_words[1] = H;
      frame_words[2] = W * H;
      frame_words[4] = 2;
      frame_words[6] = 100000;
      frame_words[9] = frame_words[10] = 1;
      frame_words[16] = 101;
      frame_words[18] = 102;
      frame_words[20] = 7;
      frame_words[22] = 11;
      frame_words[23] = 1;
      std::copy_n(constants.begin(), 12, frame_words.begin() + 24);
      frame_words[48] = 3;
      frame_words[50] = 66667;
      frame_words[52] = 1;
      temporal_words[0] = 0x31545742u;
      temporal_words[1] = temporal_words[2] = 1;
      std::copy_n(frame_words.begin() + 4, 4, temporal_words.begin() + 4);
      std::copy_n(frame_words.begin() + 16, 32, temporal_words.begin() + 20);
      std::copy(constants.begin(), constants.end(), temporal_words.begin() + 52);
      std::copy_n(frame_words.begin() + 48, 4, temporal_words.begin() + 76);
      std::copy_n(frame_words.begin() + 52, 4, temporal_words.begin() + 80);
      root_for(2u);
      cb = buffer(constants.data(), 96, false, false, true);
      root = buffer(root_words.data(), 256, true);
      args = buffer(root_words.data(), 256, true, true);
      sct = buffer(identity.data(), 128);
      proof = buffer(proof_words.data(), 128);
      receipt = buffer(temporal_words.data(), 1024, true);
      frame = buffer(frame_words.data(), 256);
      lm.assign(W * H, 0);
      bm = tm = lm;
      lm[1] = 1;
      lm[2] = 1;
      bm[2] = 1;
      tm[3] = 1;
      legacy = texture(lm);
      baseline = texture(bm);
      temporal = texture(tm);
      std::string error;
      ASSERT_TRUE(output.ensure(device.Get(), W, H, error));
    }
  };

  TEST_F(HostSbsSelectedMatteGpuTest, DirectForceTwoUsesCanonicalZeroInferRecord) {
    queue();
    expect(2, bm);
  }
  TEST_F(HostSbsSelectedMatteGpuTest, AuthorizedOneAcceptsItsOwnInferReceipt) {
    root_for(1);
    upload();
    queue(1, true);
    expect(3, tm);
  }

  TEST_F(HostSbsSelectedMatteGpuTest, QueuedForceFourSelectsExactTemporalAndKeepsLiteralReceipt) {
    root_for(4);
    upload();
    queue(4, true);
    expect(3, tm);
  }

  TEST_F(HostSbsSelectedMatteGpuTest, TemporalFaultFallsBackToCompleteCurrentBaseline) {
    temporal_words[3] = 4;
    upload();
    queue(2, true);
    expect(2, bm);
  }

  TEST_F(HostSbsSelectedMatteGpuTest, InvalidNumericProofPreservesCurrentProductionMatte) {
    proof_words[3] = 1;
    upload();
    queue(2, true);
    expect(1, lm);
  }

  TEST_F(HostSbsSelectedMatteGpuTest, RootZeroHoldsWholeSelectedTupleAcrossNewTemporalBacking) {
    queue();
    auto v = output.views();
    const auto before_m = read(v.matte.Get()), before_r = read(v.receipt.Get());
    root_words[60] = 0;
    upload();
    temporal = texture(lm);
    proof = buffer(proof_words.data(), 128);
    queue(2, true);
    EXPECT_EQ(read(v.matte.Get()), before_m);
    EXPECT_EQ(read(v.receipt.Get()), before_r);
  }

  TEST_F(HostSbsSelectedMatteGpuTest, SnapshotInactiveCopiesCurrentZeroWithoutPresenceType) {
    lm.assign(W * H, 0);
    bm = lm;
    legacy = texture(lm);
    baseline = texture(bm);
    identity[2] = 0;
    upload();
    queue();
    expect(2, bm);
  }

  TEST_F(HostSbsSelectedMatteGpuTest, ClassSpecificInferShapesCannotCrossAuthenticate) {
    root_for(4);
    upload();
    queue(2);
    expect(1, std::vector<UINT>(W * H), 1);
    root_for(2);
    upload();
    queue(4);
    expect(1, std::vector<UINT>(W * H), 1);
  }

  TEST_F(HostSbsSelectedMatteGpuTest, ModelClockAndCutMismatchesFailClosed) {
    queue();
    identity[20]++;
    upload();
    queue();
    expect(1, std::vector<UINT>(W * H), 1);
    identity[20]--;
    frame_words[22]++;
    upload();
    queue(2, true);
    expect(2, bm);
  }

  TEST_F(HostSbsSelectedMatteGpuTest, EnabledScopeRejectsPredicationAndRestoresReceiptSlot) {
    queue();
    auto v = output.views();
    auto before = read(v.receipt.Get());
    ID3D11ShaderResourceView *sentinel = proof.s.Get();
    context->PSSetShaderResources(7, 1, &sentinel);
    {
      selected::scoped_receipt_t s;
      ASSERT_TRUE(s.bind(context.Get(), v.receipt.Get()));
    }
    ComPtr<ID3D11ShaderResourceView> restored;
    context->PSGetShaderResources(7, 1, &restored);
    EXPECT_EQ(restored.Get(), sentinel);
    ID3D11ShaderResourceView *null = nullptr;
    context->PSSetShaderResources(7, 1, &null);
    D3D11_QUERY_DESC d {D3D11_QUERY_OCCLUSION_PREDICATE, 0};
    ComPtr<ID3D11Predicate> p;
    ASSERT_TRUE(SUCCEEDED(device->CreatePredicate(&d, &p)));
    context->SetPredication(p.Get(), FALSE);
    std::string error;
    EXPECT_FALSE(output.queue(context.Get(), shader.Get(), inputs(), error));
    context->SetPredication(nullptr, FALSE);
    EXPECT_EQ(read(v.receipt.Get()), before);
  }
  TEST_F(HostSbsSelectedMatteGpuTest, MalformedZeroAndSmallerSourceClearsWholeOutputButRootZeroStillHolds) {
    // Put a valid positive at the last actual output row: the old CB-based bounds would
    // skip it after a smaller crop and skip EVERY lane for zero width/height.
    bm.back() = 1u;
    baseline = texture(bm);
    queue();
    expect(2u, bm);
    const auto valid_constants = constants;
    for (UINT field : {0u, 3u}) {
      constants = valid_constants;
      constants[field] = 0u;
      upload();
      queue();
      expect(1u, std::vector<UINT>(W * H), 1u);
      constants = valid_constants;
      upload();
      queue();
      expect(2u, bm);
    }
    // A positive but smaller source/crop must clear the actual bottom row too.
    constants = valid_constants;
    constants[3] = H - 1u;
    upload();
    queue();
    expect(1u, std::vector<UINT>(W * H), 1u);
    constants = valid_constants;
    upload();
    queue();
    const auto v = output.views();
    const auto before_m = read(v.matte.Get());
    const auto before_r = read(v.receipt.Get());
    constants[0] = 0u;
    root_words[60] = 0u;
    upload();
    queue();
    EXPECT_EQ(read(v.matte.Get()), before_m);
    EXPECT_EQ(read(v.receipt.Get()), before_r);
  }
}  // namespace
#endif
