#pragma once
#ifdef _WIN32
  #include <array>
  #include <cstdint>
  #include <d3d11.h>
  #include <string>
  #include <utility>
  #include <wrl/client.h>

// PRIVATE opt-in version1. Queue success is not GPU readiness, UI presence or EMPTY.
// Caller owns the exact source/SCT/root publication and keeps all resources unbound outside
// these CS slots. AddRef retains storage, not immutable contents; normal publication leases apply.
namespace models::research_selected_matte_v1 {
  using Microsoft::WRL::ComPtr;
  inline constexpr std::uint32_t tag = 0x31534d53u, schema = 1u, algorithm = 1u;
  enum class postprocess_mode_e : std::uint32_t {
    authorized = 1u,
    direct_force = 2u,
    queued_force = 4u
  };

  struct alignas(16) request_t {
    std::array<std::uint32_t, 4> control {}, token {};
  };

  static_assert(sizeof(request_t) == 32u);

  struct views_t {
    ComPtr<ID3D11ShaderResourceView> matte, receipt;
  };

  struct inputs_t {
    ID3D11Buffer *current_cb96 = nullptr, *actual_root_args = nullptr;
    UINT observation_one_offset = 240u;
    ID3D11ShaderResourceView *legacy_matte = nullptr, *baseline_matte = nullptr, *temporal_matte = nullptr;
    ID3D11ShaderResourceView *numeric_proof = nullptr, *temporal_receipt = nullptr, *sct = nullptr, *root = nullptr, *frame256 = nullptr;
    postprocess_mode_e mode = postprocess_mode_e::direct_force;
    std::uint32_t expected_work = 0u;
    std::uint64_t transaction_token = 0u;
    bool temporal_supplied = false;
  };

  class resources_t {
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11Texture2D> matte_;
    ComPtr<ID3D11ShaderResourceView> matte_srv_, receipt_srv_;
    ComPtr<ID3D11UnorderedAccessView> matte_uav_, receipt_uav_;
    ComPtr<ID3D11Buffer> receipt_, request_, native_args_;
    UINT width_ = 0u, height_ = 0u;
    bool initialize_ = true;

    bool child(ID3D11DeviceChild *x) const {
      if (!x) {
        return false;
      }
      ComPtr<ID3D11Device> d;
      x->GetDevice(&d);
      return d.Get() == device_.Get();
    }

    struct saved_t {
      ID3D11DeviceContext *c;
      std::array<ComPtr<ID3D11ShaderResourceView>, 8> s;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 2> u;
      std::array<ComPtr<ID3D11Buffer>, 2> b;
      ComPtr<ID3D11ComputeShader> shader;
      std::array<ComPtr<ID3D11ClassInstance>, 256> classes;
      UINT count = 256u;

      explicit saved_t(ID3D11DeviceContext *x):
          c(x) {
        ID3D11ShaderResourceView *sv[8] {};
        ID3D11UnorderedAccessView *uv[2] {};
        ID3D11Buffer *cb[2] {};
        ID3D11ComputeShader *cs = nullptr;
        ID3D11ClassInstance *ci[256] {};
        c->CSGetShaderResources(0, 8, sv);
        c->CSGetUnorderedAccessViews(0, 2, uv);
        c->CSGetConstantBuffers(0, 2, cb);
        c->CSGetShader(&cs, ci, &count);
        for (UINT i = 0; i < 8; ++i) {
          s[i].Attach(sv[i]);
        }
        for (UINT i = 0; i < 2; ++i) {
          u[i].Attach(uv[i]);
          b[i].Attach(cb[i]);
        }
        shader.Attach(cs);
        for (UINT i = 0; i < count; ++i) {
          classes[i].Attach(ci[i]);
        }
      }

      static void unbind(ID3D11DeviceContext *c) {
        ID3D11ShaderResourceView *s[8] {};
        ID3D11UnorderedAccessView *u[2] {};
        c->CSSetShaderResources(0, 8, s);
        c->CSSetUnorderedAccessViews(0, 2, u, nullptr);
      }

      ~saved_t() {
        unbind(c);
        ID3D11ShaderResourceView *sv[8] {};
        ID3D11UnorderedAccessView *uv[2] {};
        ID3D11Buffer *cb[2] {};
        ID3D11ClassInstance *ci[256] {};
        for (UINT i = 0; i < 8; ++i) {
          sv[i] = s[i].Get();
        }
        for (UINT i = 0; i < 2; ++i) {
          uv[i] = u[i].Get();
          cb[i] = b[i].Get();
        }
        for (UINT i = 0; i < count; ++i) {
          ci[i] = classes[i].Get();
        }
        c->CSSetUnorderedAccessViews(0, 2, uv, nullptr);
        c->CSSetShaderResources(0, 8, sv);
        c->CSSetConstantBuffers(0, 2, cb);
        c->CSSetShader(shader.Get(), count ? ci : nullptr, count);
      }
    };

  public:
    // Lifetime intentionally independent of temporal backend/source epoch. Shape replacement is an
    // existing tuple-domain revocation; retaining an old ComPtr never authenticates a new shape.
    bool ensure(ID3D11Device *d, UINT w, UINT h, std::string &error) {
      if (!d || !w || !h || w > 5120u || h > 5120u) {
        error = "Selected matte shape/device";
        return false;
      }
      if (device_.Get() == d && width_ == w && height_ == h && matte_) {
        return true;
      }
      resources_t n;
      n.device_ = d;
      n.width_ = w;
      n.height_ = h;
      D3D11_TEXTURE2D_DESC td {};
      td.Width = w;
      td.Height = h;
      td.MipLevels = 1;
      td.ArraySize = 1;
      td.Format = DXGI_FORMAT_R32_UINT;
      td.SampleDesc.Count = 1;
      td.Usage = D3D11_USAGE_DEFAULT;
      td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      if (FAILED(d->CreateTexture2D(&td, nullptr, &n.matte_)) || FAILED(d->CreateShaderResourceView(n.matte_.Get(), nullptr, &n.matte_srv_)) || FAILED(d->CreateUnorderedAccessView(n.matte_.Get(), nullptr, &n.matte_uav_))) {
        error = "Selected matte texture allocation";
        return false;
      }
      D3D11_BUFFER_DESC bd {};
      bd.ByteWidth = 256;
      bd.Usage = D3D11_USAGE_DEFAULT;
      bd.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      bd.StructureByteStride = 4;
      if (FAILED(d->CreateBuffer(&bd, nullptr, &n.receipt_)) || FAILED(d->CreateShaderResourceView(n.receipt_.Get(), nullptr, &n.receipt_srv_)) || FAILED(d->CreateUnorderedAccessView(n.receipt_.Get(), nullptr, &n.receipt_uav_))) {
        error = "Selected matte receipt allocation";
        return false;
      }
      bd = {};
      bd.ByteWidth = 32;
      bd.Usage = D3D11_USAGE_DEFAULT;
      bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      request_t zero {};
      D3D11_SUBRESOURCE_DATA init {&zero, 0, 0};
      if (FAILED(d->CreateBuffer(&bd, &init, &n.request_))) {
        error = "Selected matte request allocation";
        return false;
      }
      const std::array<UINT, 3> args {(w + 15u) / 16u, (h + 15u) / 16u, 0u};
      bd = {};
      bd.ByteWidth = 12;
      bd.Usage = D3D11_USAGE_DEFAULT;
      bd.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      init = {args.data(), 0, 0};
      if (FAILED(d->CreateBuffer(&bd, &init, &n.native_args_))) {
        error = "Selected matte native arguments allocation";
        return false;
      }
      *this = std::move(n);
      return true;
    }

    bool queue(ID3D11DeviceContext *c, ID3D11ComputeShader *shader, const inputs_t &i, std::string &error) {
      error.clear();
      if (!c || c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !child(shader) || !child(i.current_cb96) || !child(i.actual_root_args) || !child(i.legacy_matte) || !child(i.sct) || !child(i.root) || !matte_) {
        error = "Selected matte required operands/context";
        return false;
      }
      ComPtr<ID3D11Device> d;
      c->GetDevice(&d);
      if (d.Get() != device_.Get()) {
        error = "Selected matte context device";
        return false;
      }
      ComPtr<ID3D11Predicate> p;
      BOOL pv = FALSE;
      c->GetPredication(&p, &pv);
      if (p) {
        error = "Selected matte predication unavailable";
        return false;
      }
      for (auto *x : {i.baseline_matte, i.temporal_matte, i.numeric_proof, i.temporal_receipt, i.frame256}) {
        if (x && !child(x)) {
          error = "Selected matte optional operand device";
          return false;
        }
      }
      D3D11_BUFFER_DESC bd {};
      i.current_cb96->GetDesc(&bd);
      if (bd.ByteWidth != 96u || (bd.BindFlags & D3D11_BIND_CONSTANT_BUFFER) == 0u) {
        error = "Selected matte CB96";
        return false;
      }
      i.actual_root_args->GetDesc(&bd);
      if (bd.ByteWidth != 256u || (bd.MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) == 0u || i.observation_one_offset != 240u) {
        error = "Selected matte actual observationOne arguments";
        return false;
      }
      auto mode = static_cast<UINT>(i.mode);
      if ((mode != 1u && mode != 2u && mode != 4u) || i.expected_work > 2u || !i.transaction_token) {
        error = "Selected matte actual root class/work/token";
        return false;
      }
      if (i.temporal_supplied && (!i.temporal_matte || !i.temporal_receipt || !i.frame256)) {
        error = "Selected matte incomplete temporal bundle";
        return false;
      }
      const saved_t saved(c);
      saved_t::unbind(c);
      if (initialize_) {
        const UINT z[4] {};
        c->ClearUnorderedAccessViewUint(matte_uav_.Get(), z);
        c->ClearUnorderedAccessViewUint(receipt_uav_.Get(), z);
        initialize_ = false;
      }
      request_t r;
      r.control = {mode, i.expected_work, i.temporal_supplied ? 1u : 0u, 1u};
      r.token = {static_cast<UINT>(i.transaction_token), static_cast<UINT>(i.transaction_token >> 32u), 0u, 0u};
      c->UpdateSubresource(request_.Get(), 0, nullptr, &r, 0, 0);
      // Separate argument storage: neither receipt nor another writable UAV aliases DispatchIndirect.
      const D3D11_BOX box {i.observation_one_offset, 0, 0, i.observation_one_offset + 4u, 1, 1};
      c->CopySubresourceRegion(native_args_.Get(), 0, 8, 0, 0, i.actual_root_args, 0, &box);
      ID3D11ShaderResourceView *s[8] {i.legacy_matte, i.baseline_matte, i.temporal_matte, i.numeric_proof, i.temporal_receipt, i.sct, i.root, i.frame256};
      ID3D11UnorderedAccessView *u[2] {matte_uav_.Get(), receipt_uav_.Get()};
      ID3D11Buffer *b[2] {i.current_cb96, request_.Get()};
      c->CSSetConstantBuffers(0, 2, b);
      c->CSSetShaderResources(0, 8, s);
      c->CSSetUnorderedAccessViews(0, 2, u, nullptr);
      c->CSSetShader(shader, nullptr, 0);
      c->DispatchIndirect(native_args_.Get(), 0);
      return true;
    }

    views_t views() const {
      return {matte_srv_, receipt_srv_};
    }
  };

  // One private PS slot only; normal recorder still owns literal t0..6/b2/b3. Scope ends
  // before any next estimator mutation. No readback, predicate change or completed-authority grant.
  class scoped_receipt_t {
    ID3D11DeviceContext *c_ = nullptr;
    ComPtr<ID3D11ShaderResourceView> previous_;

  public:
    scoped_receipt_t() = default;
    scoped_receipt_t(const scoped_receipt_t &) = delete;
    scoped_receipt_t &operator=(const scoped_receipt_t &) = delete;
    bool bind(ID3D11DeviceContext *c, ID3D11ShaderResourceView *receipt) {
      if (!c || !receipt || c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
        return false;
      }
      ComPtr<ID3D11Device> a, b;
      c->GetDevice(&a);
      receipt->GetDevice(&b);
      if (a.Get() != b.Get()) {
        return false;
      }
      ComPtr<ID3D11Predicate> p;
      BOOL v = FALSE;
      c->GetPredication(&p, &v);
      if (p) {
        return false;
      }
      ID3D11ShaderResourceView *old = nullptr;
      c->PSGetShaderResources(7, 1, &old);
      previous_.Attach(old);
      c_ = c;
      c_->PSSetShaderResources(7, 1, &receipt);
      return true;
    }

    ~scoped_receipt_t() {
      if (c_) {
        auto *x = previous_.Get();
        c_->PSSetShaderResources(7, 1, &x);
      }
    }
  };
}  // namespace models::research_selected_matte_v1
#endif
