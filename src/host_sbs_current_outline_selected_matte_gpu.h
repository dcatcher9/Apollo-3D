#pragma once
#ifdef _WIN32
  #include "host_sbs_selected_matte_receipt.h"
  #include "host_sbs_compute_program_lease.h"
  #include <array>
  #include <cstdint>
  #include <d3d11.h>
  #include <string>
  #include <type_traits>
  #include <utility>
  #include <wrl/client.h>

// Production current-outline selector and explicitly requested algorithm1 experiment support.
// Queue success is not GPU readiness, UI presence or EMPTY.
// Caller owns the exact source/SCT/root publication and keeps all resources unbound outside
// these CS slots. AddRef retains storage, not immutable contents; normal publication leases apply.
namespace models::research_current_outline_selected_matte_v2 {
  using Microsoft::WRL::ComPtr;
  namespace receipt_abi = selected_matte_receipt_v1;
  inline constexpr std::uint32_t tag = receipt_abi::tag, schema = receipt_abi::schema,
                                 algorithm = receipt_abi::current_outline_algorithm;
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

  struct program_t { ID3D11ComputeShader *prepare = nullptr, *select = nullptr; };

  struct inputs_t {
    ID3D11Buffer *current_cb96 = nullptr, *actual_root_args = nullptr;
    UINT observation_one_offset = 240u;
    ID3D11ShaderResourceView *legacy_matte = nullptr, *baseline_matte = nullptr, *temporal_matte = nullptr;
    ID3D11ShaderResourceView *numeric_proof = nullptr, *temporal_receipt = nullptr, *sct = nullptr, *root = nullptr, *frame256 = nullptr;
    postprocess_mode_e mode = postprocess_mode_e::direct_force;
    std::uint32_t expected_work = 0u;
    std::uint64_t transaction_token = 0u;
    bool temporal_supplied = false;
    bool current_outline_requested = false;
    ID3D11ShaderResourceView *source_envelope104 = nullptr, *cbc_meta512 = nullptr;
    ID3D11ShaderResourceView *raw_meta512 = nullptr, *f2_data = nullptr, *f2_matte = nullptr;
  };

  class resources_t {
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11Texture2D> matte_;
    ComPtr<ID3D11ShaderResourceView> matte_srv_, receipt_srv_;
    ComPtr<ID3D11UnorderedAccessView> matte_uav_, receipt_uav_;
    ComPtr<ID3D11Buffer> receipt_, request_, native_args_, root_mirror_;
    ComPtr<ID3D11ShaderResourceView> root_mirror_srv_;
    ComPtr<ID3D11UnorderedAccessView> native_args_uav_;
    UINT width_ = 0u, height_ = 0u;
    bool initialize_ = true;
    host_sbs_compute_program_lease::group_t<1u> select_program_lease_, prepare_program_lease_;

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
      std::array<ComPtr<ID3D11ShaderResourceView>, 16> s;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 3> u;
      std::array<ComPtr<ID3D11Buffer>, 2> b;
      ComPtr<ID3D11ComputeShader> shader;
      std::array<ComPtr<ID3D11ClassInstance>, 256> classes;
      UINT count = 256u;

      explicit saved_t(ID3D11DeviceContext *x):
          c(x) {
        ID3D11ShaderResourceView *sv[16] {};
        ID3D11UnorderedAccessView *uv[3] {};
        ID3D11Buffer *cb[2] {};
        ID3D11ComputeShader *cs = nullptr;
        ID3D11ClassInstance *ci[256] {};
        c->CSGetShaderResources(0, 16, sv);
        c->CSGetUnorderedAccessViews(0, 3, uv);
        c->CSGetConstantBuffers(0, 2, cb);
        c->CSGetShader(&cs, ci, &count);
        for (UINT i = 0; i < 16; ++i) {
          s[i].Attach(sv[i]);
        }
        for (UINT i = 0; i < 3; ++i) u[i].Attach(uv[i]);
        for (UINT i = 0; i < 2; ++i) b[i].Attach(cb[i]);
        shader.Attach(cs);
        for (UINT i = 0; i < count; ++i) {
          classes[i].Attach(ci[i]);
        }
      }

      static void unbind(ID3D11DeviceContext *c) {
        ID3D11ShaderResourceView *s[16] {};
        ID3D11UnorderedAccessView *u[3] {};
        c->CSSetShaderResources(0, 16, s);
        c->CSSetUnorderedAccessViews(0, 3, u, nullptr);
      }

      ~saved_t() {
        unbind(c);
        ID3D11ShaderResourceView *sv[16] {};
        ID3D11UnorderedAccessView *uv[3] {};
        ID3D11Buffer *cb[2] {};
        ID3D11ClassInstance *ci[256] {};
        for (UINT i = 0; i < 16; ++i) {
          sv[i] = s[i].Get();
        }
        for (UINT i = 0; i < 3; ++i) uv[i] = u[i].Get();
        for (UINT i = 0; i < 2; ++i) cb[i] = b[i].Get();
        for (UINT i = 0; i < count; ++i) {
          ci[i] = classes[i].Get();
        }
        c->CSSetUnorderedAccessViews(0, 3, uv, nullptr);
        c->CSSetShaderResources(0, 16, sv);
        c->CSSetConstantBuffers(0, 2, cb);
        c->CSSetShader(shader.Get(), count ? ci : nullptr, count);
      }
    };

  public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;
    resources_t(resources_t &&other) noexcept { *this=std::move(other); }
    resources_t &operator=(resources_t &&other) noexcept {
      if(this!=&other) {
        // MinGW WRL's ComPtr rvalue assignment can retain the source reference.
        // Detach/Attach makes this resource owner an actual transfer on both SDKs.
        device_.Attach(other.device_.Detach());matte_.Attach(other.matte_.Detach());
        matte_srv_.Attach(other.matte_srv_.Detach());receipt_srv_.Attach(other.receipt_srv_.Detach());
        matte_uav_.Attach(other.matte_uav_.Detach());receipt_uav_.Attach(other.receipt_uav_.Detach());
        receipt_.Attach(other.receipt_.Detach());request_.Attach(other.request_.Detach());
        native_args_.Attach(other.native_args_.Detach());root_mirror_.Attach(other.root_mirror_.Detach());
        native_args_uav_.Attach(other.native_args_uav_.Detach());root_mirror_srv_.Attach(other.root_mirror_srv_.Detach());
        width_=std::exchange(other.width_,0u);height_=std::exchange(other.height_,0u);
        initialize_=std::exchange(other.initialize_,true);
      }
      return *this;
    }

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
      bd.ByteWidth = receipt_abi::byte_count;
      bd.Usage = D3D11_USAGE_DEFAULT;
      bd.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      bd.StructureByteStride = receipt_abi::word_bytes;
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
      const std::array<UINT, 4> args {(w + 15u) / 16u, (h + 15u) / 16u, 0u, 0u};
      bd = {};
      bd.ByteWidth = 16;
      bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      bd.Usage = D3D11_USAGE_DEFAULT;
      bd.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS | D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
      init = {args.data(), 0, 0};
      if (FAILED(d->CreateBuffer(&bd, &init, &n.native_args_))) {
        error = "Selected matte native arguments allocation";
        return false;
      }
      D3D11_UNORDERED_ACCESS_VIEW_DESC au {};
      au.Format = DXGI_FORMAT_R32_TYPELESS; au.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      au.Buffer.NumElements = 4u; au.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      if (FAILED(d->CreateUnorderedAccessView(n.native_args_.Get(), &au, &n.native_args_uav_))) { error="Selected native arguments UAV"; return false; }
      bd={}; bd.ByteWidth=256u; bd.Usage=D3D11_USAGE_DEFAULT; bd.BindFlags=D3D11_BIND_SHADER_RESOURCE;
      bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
      D3D11_SHADER_RESOURCE_VIEW_DESC rs {}; rs.Format=DXGI_FORMAT_R32_TYPELESS;
      rs.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX; rs.BufferEx.NumElements=64u; rs.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;
      if (FAILED(d->CreateBuffer(&bd,nullptr,&n.root_mirror_)) || FAILED(d->CreateShaderResourceView(n.root_mirror_.Get(),&rs,&n.root_mirror_srv_))) { error="Selected actual-root mirror"; return false; }
      *this = std::move(n);
      return true;
    }

    bool queue(ID3D11DeviceContext *c, const program_t &program, const inputs_t &i, std::string &error) {
      error.clear();
      if (!c || c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !select_program_lease_.acquire(device_.Get(), {program.select}) || (i.current_outline_requested && !prepare_program_lease_.acquire(device_.Get(), {program.prepare})) || !child(i.current_cb96) || !child(i.actual_root_args) || !child(i.legacy_matte) || !child(i.sct) || (!i.current_outline_requested && !child(i.root)) || !matte_) {
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
      for (auto *x : {i.baseline_matte, i.temporal_matte, i.numeric_proof, i.temporal_receipt, i.frame256, i.source_envelope104, i.cbc_meta512, i.raw_meta512, i.f2_data, i.f2_matte}) {
        if (x && !child(x)) {
          error = "Selected matte optional operand device";
          return false;
        }
      }
      D3D11_BUFFER_DESC bd {};
      i.current_cb96->GetDesc(&bd);
      if (bd.ByteWidth != receipt_abi::contour_byte_count || (bd.BindFlags & D3D11_BIND_CONSTANT_BUFFER) == 0u) {
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
      if (i.current_outline_requested) {
        if (!i.source_envelope104 || !i.cbc_meta512 || !i.raw_meta512 || !i.f2_data || !i.f2_matte) {error="Selected incomplete current-outline bundle";return false;}
        for (auto *view : {i.legacy_matte,i.baseline_matte,i.temporal_matte,i.numeric_proof,i.temporal_receipt,i.sct,i.root,i.frame256,i.source_envelope104,i.cbc_meta512,i.raw_meta512,i.f2_data,i.f2_matte}) {
          if (!view) continue;
          ComPtr<ID3D11Resource> r; view->GetResource(&r);
          if (r.Get()==matte_.Get() || r.Get()==receipt_.Get() || r.Get()==native_args_.Get() || r.Get()==root_mirror_.Get()) {error="Selected writable operand alias";return false;}
        }
        D3D11_SHADER_RESOURCE_VIEW_DESC vd {};i.legacy_matte->GetDesc(&vd);
        ComPtr<ID3D11Resource> r;ComPtr<ID3D11Texture2D> t;i.legacy_matte->GetResource(&r);
        if (FAILED(r.As(&t))) {error="Selected original legacy M texture";return false;}
        D3D11_TEXTURE2D_DESC td {};t->GetDesc(&td);
        if (vd.Format!=DXGI_FORMAT_R32_UINT || vd.ViewDimension!=D3D11_SRV_DIMENSION_TEXTURE2D || vd.Texture2D.MostDetailedMip!=0u ||
            td.Format!=DXGI_FORMAT_R32_UINT || td.Width!=width_ || td.Height!=height_ || td.MipLevels!=1u || td.ArraySize!=1u || td.SampleDesc.Count!=1u) {error="Selected literal legacy M descriptor";return false;}
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
      r.control = {mode, i.expected_work, i.current_outline_requested ? 0u : (i.temporal_supplied ? 1u : 0u), i.current_outline_requested ? 2u : 1u};
      r.token = {static_cast<UINT>(i.transaction_token), static_cast<UINT>(i.transaction_token >> 32u), 0u, 0u};
      c->UpdateSubresource(request_.Get(), 0, nullptr, &r, 0, 0);
      ID3D11ShaderResourceView *root = i.root;
      if (i.current_outline_requested) {c->CopyResource(root_mirror_.Get(),i.actual_root_args);root=root_mirror_srv_.Get();}
      ID3D11ShaderResourceView *s[16] {i.legacy_matte,i.baseline_matte,i.temporal_matte,i.numeric_proof,i.temporal_receipt,i.sct,root,i.frame256,i.source_envelope104,i.cbc_meta512,i.raw_meta512,i.f2_data,i.f2_matte,nullptr,nullptr,nullptr};
      ID3D11UnorderedAccessView *u[3] {matte_uav_.Get(),receipt_uav_.Get(),nullptr};
      ID3D11Buffer *b[2] {i.current_cb96,request_.Get()};
      c->CSSetConstantBuffers(0,2,b);c->CSSetShaderResources(0,16,s);c->CSSetUnorderedAccessViews(0,3,u,nullptr);
      if (i.current_outline_requested) {
        auto *a=native_args_uav_.Get();c->CSSetUnorderedAccessViews(2,1,&a,nullptr);
        c->CSSetShader(program.prepare,nullptr,0);c->Dispatch(1,1,1);
        ID3D11UnorderedAccessView *z=nullptr;c->CSSetUnorderedAccessViews(2,1,&z,nullptr);
      } else {
        // Original algorithm1 copies observationOne.X unchanged.
        const D3D11_BOX box {i.observation_one_offset,0,0,i.observation_one_offset+4u,1,1};
        c->CopySubresourceRegion(native_args_.Get(),0,8,0,0,i.actual_root_args,0,&box);
      }
      c->CSSetShader(program.select,nullptr,0);c->DispatchIndirect(native_args_.Get(),0);
      return true;
    }

    views_t views() const {
      return {matte_srv_, receipt_srv_};
    }
  };

  static_assert(!std::is_copy_constructible_v<resources_t> &&
                !std::is_copy_assignable_v<resources_t> &&
                std::is_nothrow_move_constructible_v<resources_t> &&
                std::is_nothrow_move_assignable_v<resources_t>);

  // Two private readonly PS slots7/8; normal recorder owns literal t0..6/b2/b3. Scope ends
  // before any next estimator mutation. No readback, predicate change or completed-authority grant.
  class scoped_receipt_t {
    ID3D11DeviceContext *context_=nullptr;
    std::array<ComPtr<ID3D11ShaderResourceView>,2> previous_;
  public:
    scoped_receipt_t()=default;scoped_receipt_t(const scoped_receipt_t&)=delete;
    scoped_receipt_t&operator=(const scoped_receipt_t&)=delete;
    bool bind(ID3D11DeviceContext*c,ID3D11ShaderResourceView*receipt,ID3D11ShaderResourceView*original_legacy) {
      if(!c||!receipt||!original_legacy||c->GetType()!=D3D11_DEVICE_CONTEXT_IMMEDIATE)return false;
      ComPtr<ID3D11Device>a,b,d;c->GetDevice(&a);receipt->GetDevice(&b);original_legacy->GetDevice(&d);
      if(a.Get()!=b.Get()||a.Get()!=d.Get())return false;
      ComPtr<ID3D11Predicate>p;BOOL v=FALSE;c->GetPredication(&p,&v);if(p)return false;
      ID3D11ShaderResourceView*old[2]{};c->PSGetShaderResources(7,2,old);
      for(UINT i=0;i<2;++i)previous_[i].Attach(old[i]);
      context_=c;ID3D11ShaderResourceView*next[2]{receipt,original_legacy};c->PSSetShaderResources(7,2,next);return true;
    }
    ~scoped_receipt_t(){if(context_){ID3D11ShaderResourceView*p[2]{previous_[0].Get(),previous_[1].Get()};context_->PSSetShaderResources(7,2,p);}}
  };
} // namespace models::research_current_outline_selected_matte_v2
#endif
