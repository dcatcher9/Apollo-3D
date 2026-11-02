#pragma once
// DEFAULT-OFF private retirement helper. No model work and no positive/EMPTY authority.
// Capture/observe execute OUTSIDE scoped_ps_negative_t lifetime; t10 receipt must
// not remain bound as PS SRV while this helper writes its receipt UAV.
#ifdef _WIN32
  #include "host_sbs_source_pattern_retirement_render.h"
  #include "video_depth_estimator.h"

  #include <algorithm>
  #include <array>
  #include <cstring>
  #include <string>
  #include <vector>

namespace models::research_source_pattern_retirement_v1 {
  using Microsoft::WRL::ComPtr;
  inline constexpr std::array<const char *, 8> entries {{"owner_init_main", "owner_validate_main", "owner_palette_main", "owner_anchors_main", "observe_init_main", "observe_validate_main", "compare_main", "retire_main"}};

  struct shader_set_t {
    std::array<ID3D11ComputeShader *, 8> shaders {};
  };

  class resources_t {
    struct words_t {
      ComPtr<ID3D11Buffer> buffer;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    struct copy_t {
      ComPtr<ID3D11Resource> resource;
      ComPtr<ID3D11ShaderResourceView> srv;
    };

    struct bank_t {
      // source,P,OCR,UIP,SCT,CB96,transaction,args. Independently allocated and initially zero.
      std::array<copy_t, 8> old;
      words_t owner, anchors, scratch, scores, receipt;
      ComPtr<ID3D11Buffer> expected;
      host_sbs_gpu_completion_receipt::expected_t publication {};
      bool captured = false;
    };

    ComPtr<ID3D11Device> device_;
    std::array<bank_t, 2> banks_;
    words_t global_;
    shader_set_t shaders_;
    depth_input_region_t region_ {};
    depth_tensor_content_rect_t content_ {};
    UINT width_ = 0, height_ = 0, fieldW_ = 0, fieldH_ = 0;
    std::uint64_t domain_ = 0;
    bool prepared_ = false;

    class saved_cs_t {
      ID3D11DeviceContext *c_;
      std::array<ComPtr<ID3D11ShaderResourceView>, 7> srvs_;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 6> uavs_;
      std::array<ComPtr<ID3D11Buffer>, 3> cbs_;
      ComPtr<ID3D11ComputeShader> shader_;
      std::array<ComPtr<ID3D11ClassInstance>, 256> instances_;
      UINT count_ = 256;

    public:
      explicit saved_cs_t(ID3D11DeviceContext *c):
          c_(c) {
        ID3D11ShaderResourceView *s[7] {};
        c->CSGetShaderResources(0, 7, s);
        for (UINT i = 0; i < 7; ++i) {
          srvs_[i].Attach(s[i]);
        }
        ID3D11UnorderedAccessView *u[6] {};
        c->CSGetUnorderedAccessViews(0, 6, u);
        for (UINT i = 0; i < 6; ++i) {
          uavs_[i].Attach(u[i]);
        }
        ID3D11Buffer *b[3] {};
        c->CSGetConstantBuffers(0, 3, b);
        for (UINT i = 0; i < 3; ++i) {
          cbs_[i].Attach(b[i]);
        }
        ID3D11ClassInstance *cl[256] {};
        c->CSGetShader(shader_.GetAddressOf(), cl, &count_);
        for (UINT i = 0; i < count_; ++i) {
          instances_[i].Attach(cl[i]);
        }
      }

      static void unbind(ID3D11DeviceContext *c) {
        ID3D11ShaderResourceView *s[7] {};
        ID3D11UnorderedAccessView *u[6] {};
        ID3D11Buffer *b[3] {};
        c->CSSetShaderResources(0, 7, s);
        c->CSSetUnorderedAccessViews(0, 6, u, nullptr);
        c->CSSetConstantBuffers(0, 3, b);
      }

      ~saved_cs_t() {
        unbind(c_);
        ID3D11ShaderResourceView *s[7] {};
        ID3D11UnorderedAccessView *u[6] {};
        ID3D11Buffer *b[3] {};
        for (UINT i = 0; i < 7; ++i) {
          s[i] = srvs_[i].Get();
        }
        for (UINT i = 0; i < 6; ++i) {
          u[i] = uavs_[i].Get();
        }
        for (UINT i = 0; i < 3; ++i) {
          b[i] = cbs_[i].Get();
        }
        c_->CSSetShaderResources(0, 7, s);
        c_->CSSetUnorderedAccessViews(0, 6, u, nullptr);
        c_->CSSetConstantBuffers(0, 3, b);
        ID3D11ClassInstance *cl[256] {};
        for (UINT i = 0; i < count_; ++i) {
          cl[i] = instances_[i].Get();
        }
        c_->CSSetShader(shader_.Get(), count_ ? cl : nullptr, count_);
      }

      saved_cs_t(const saved_cs_t &) = delete;
    };

    static ComPtr<ID3D11Resource> resource(ID3D11ShaderResourceView *v) {
      ComPtr<ID3D11Resource> r;
      if (v) {
        v->GetResource(&r);
      }
      return r;
    }

    static std::array<ComPtr<ID3D11Resource>, 8> operands(const completed_ocr_conditioner_research_views_t &v) {
      return {resource(v.source_band.Get()), resource(v.probability_snapshot.Get()), resource(v.ocr_record.Get()), resource(v.uip.Get()), resource(v.sct.Get()), v.constants, v.transaction, v.dispatch_arguments};
    }

    bool own_context(ID3D11DeviceContext *c, std::string &error) const {
      if (!c || c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
        error = "Retirement requires immediate context";
        return false;
      }
      ComPtr<ID3D11Device> d;
      c->GetDevice(&d);
      if (d.Get() != device_.Get()) {
        error = "Retirement device mismatch";
        return false;
      }
      ComPtr<ID3D11Predicate> p;
      BOOL value = FALSE;
      c->GetPredication(&p, &value);
      if (p) {
        error = "Retirement rejects predication before commands/copies";
        return false;
      }
      return true;
    }

    static bool buffer_shape(ID3D11Resource *r, UINT bytes) {
      ComPtr<ID3D11Buffer> b;
      if (!r || FAILED(r->QueryInterface(IID_PPV_ARGS(&b)))) {
        return false;
      }
      D3D11_BUFFER_DESC d {};
      b->GetDesc(&d);
      return d.ByteWidth == bytes;
    }

    bool allocate_words(words_t &w, UINT count) {
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = count * 4;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      d.StructureByteStride = 4;
      std::vector<std::uint32_t> zero(count);
      D3D11_SUBRESOURCE_DATA data {zero.data(), 0, 0};
      return SUCCEEDED(device_->CreateBuffer(&d, &data, &w.buffer)) && SUCCEEDED(device_->CreateShaderResourceView(w.buffer.Get(), nullptr, &w.srv)) && SUCCEEDED(device_->CreateUnorderedAccessView(w.buffer.Get(), nullptr, &w.uav));
    }

    bool allocate_copy(copy_t &out, ID3D11Resource *source, bool view) {
      if (!source) {
        return false;
      }
      D3D11_RESOURCE_DIMENSION kind {};
      source->GetType(&kind);
      if (kind == D3D11_RESOURCE_DIMENSION_BUFFER) {
        ComPtr<ID3D11Buffer> b;
        source->QueryInterface(IID_PPV_ARGS(&b));
        D3D11_BUFFER_DESC d {};
        b->GetDesc(&d);
        d.Usage = D3D11_USAGE_DEFAULT;
        d.CPUAccessFlags = 0;
        std::vector<std::uint8_t> zero(d.ByteWidth);
        D3D11_SUBRESOURCE_DATA init {zero.data(), 0, 0};
        ComPtr<ID3D11Buffer> target;
        if (FAILED(device_->CreateBuffer(&d, &init, &target))) {
          return false;
        }
        out.resource = target;
        if (view && FAILED(device_->CreateShaderResourceView(target.Get(), nullptr, &out.srv))) {
          return false;
        }
        return true;
      }
      if (kind == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
        ComPtr<ID3D11Texture2D> t;
        source->QueryInterface(IID_PPV_ARGS(&t));
        D3D11_TEXTURE2D_DESC d {};
        t->GetDesc(&d);
        if (d.MipLevels != 1 || d.ArraySize != 1 || d.SampleDesc.Count != 1 || d.Format != DXGI_FORMAT_R16G16B16A16_FLOAT) {
          return false;
        }
        d.Usage = D3D11_USAGE_DEFAULT;
        d.CPUAccessFlags = 0;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        std::vector<std::uint8_t> zero(std::size_t(d.Width) * d.Height * 8);
        D3D11_SUBRESOURCE_DATA init {zero.data(), d.Width * 8, 0};
        ComPtr<ID3D11Texture2D> target;
        if (FAILED(device_->CreateTexture2D(&d, &init, &target))) {
          return false;
        }
        out.resource = target;
        return !view || SUCCEEDED(device_->CreateShaderResourceView(target.Get(), nullptr, &out.srv));
      }
      return false;
    }

    static bool same_shape(ID3D11Resource *a, ID3D11Resource *b) {
      if (!a || !b) {
        return false;
      }
      D3D11_RESOURCE_DIMENSION x {}, y {};
      a->GetType(&x);
      b->GetType(&y);
      if (x != y) {
        return false;
      }
      if (x == D3D11_RESOURCE_DIMENSION_BUFFER) {
        ComPtr<ID3D11Buffer> u, v;
        a->QueryInterface(IID_PPV_ARGS(&u));
        b->QueryInterface(IID_PPV_ARGS(&v));
        D3D11_BUFFER_DESC du {}, dv {};
        u->GetDesc(&du);
        v->GetDesc(&dv);
        return du.ByteWidth == dv.ByteWidth && du.StructureByteStride == dv.StructureByteStride;
      }
      if (x == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
        ComPtr<ID3D11Texture2D> u, v;
        a->QueryInterface(IID_PPV_ARGS(&u));
        b->QueryInterface(IID_PPV_ARGS(&v));
        D3D11_TEXTURE2D_DESC du {}, dv {};
        u->GetDesc(&du);
        v->GetDesc(&dv);
        return du.Width == dv.Width && du.Height == dv.Height && du.Format == dv.Format && du.MipLevels == dv.MipLevels && du.ArraySize == dv.ArraySize && du.SampleDesc.Count == dv.SampleDesc.Count;
      }
      return false;
    }

    void bind(ID3D11DeviceContext *c, bank_t &b, ID3D11ShaderResourceView *current, ID3D11ShaderResourceView *sourceReceipt, ID3D11Buffer *currentCB, bool compare) {
      auto oldCB = static_cast<ID3D11Buffer *>(b.old[5].resource.Get());
      ID3D11Buffer *cb[] {oldCB, currentCB, b.expected.Get()};
      c->CSSetConstantBuffers(0, 3, cb);
      ID3D11ShaderResourceView *s[] {b.old[0].srv.Get(), b.old[1].srv.Get(), b.old[2].srv.Get(), b.old[4].srv.Get(), current, sourceReceipt, compare ? b.anchors.srv.Get() : nullptr};
      c->CSSetShaderResources(0, 7, s);
      ID3D11UnorderedAccessView *u[] {b.owner.uav.Get(), compare ? nullptr : b.anchors.uav.Get(), b.scratch.uav.Get(), b.scores.uav.Get(), b.receipt.uav.Get(), global_.uav.Get()};
      c->CSSetUnorderedAccessViews(0, 6, u, nullptr);
    }

    bool expected_owner(const expected_t &e, const bank_t &b) const {
      const auto v = make_expected(research_current_source_snapshot_v1::constants_t {}, b.publication);
      return e.owner0 == v.owner0 && e.owner1 == v.owner1 && e.owner2 == v.owner2 && e.control == std::array<std::uint32_t, 4> {1, tag, schema, 0};
    }

  public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;

    bool prepare(ID3D11Device *d, ID3D11DeviceContext *c, const completed_ocr_conditioner_research_views_t &warm, const shader_set_t &shaders, std::string &error) {
      error.clear();
      if (prepared_ || !d) {
        error = "Retirement prepare called twice/no device";
        return false;
      }
      device_ = d;
      if (!own_context(c, error)) {
        return false;
      }
      for (auto *s : shaders.shaders) {
        if (!s) {
          error = "Retirement shader missing";
          return false;
        }
      }
      const auto a = operands(warm);
      for (const auto &r : a) {
        if (!r) {
          error = "Warmup old snapshot operand missing";
          return false;
        }
      }
      if (!buffer_shape(a[1].Get(), 960 * 160 * 4) || !buffer_shape(a[2].Get(), 256 * 4) || !buffer_shape(a[3].Get(), 40 * 4) || !buffer_shape(a[4].Get(), 32 * 4) || !buffer_shape(a[5].Get(), 96) || !buffer_shape(a[6].Get(), 256) || !buffer_shape(a[7].Get(), 256)) {
        error = "Warmup old operand byte sizes invalid";
        return false;
      }
      ComPtr<ID3D11Texture2D> t;
      if (FAILED(a[0].As(&t))) {
        error = "Warmup source not texture";
        return false;
      }
      D3D11_TEXTURE2D_DESC td {};
      t->GetDesc(&td);
      width_ = td.Width;
      height_ = td.Height;
      const auto N = std::uint64_t(width_) * height_;
      if (!N || width_ > 5120u || height_ > 5120u) {
        error = "Retirement unsupported native validation dispatch";
        return false;
      }
      region_ = warm.input_region;
      content_ = warm.field_content;
      fieldW_ = warm.field_width;
      fieldH_ = warm.field_height;
      domain_ = warm.publication.domain_tag;
      shaders_ = shaders;
      for (auto &b : banks_) {
        for (std::size_t i = 0; i < a.size(); ++i) {
          if (!allocate_copy(b.old[i], a[i].Get(), i < 5)) {
            error = "Retirement immutable owner allocation failed";
            return false;
          }
        }
        if (!allocate_words(b.owner, 256) || !allocate_words(b.anchors, 4096) || !allocate_words(b.scratch, 64) || !allocate_words(b.scores, 32) || !allocate_words(b.receipt, 64)) {
          error = "Retirement buffers allocation failed";
          return false;
        }
        D3D11_BUFFER_DESC bd {};
        bd.ByteWidth = sizeof(expected_t);
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        expected_t zero {};
        D3D11_SUBRESOURCE_DATA data {&zero, 0, 0};
        if (FAILED(device_->CreateBuffer(&bd, &data, &b.expected))) {
          error = "Retirement expectedCB allocation failed";
          return false;
        }
      }
      if (!allocate_words(global_, 32)) {
        error = "Retirement global source watermark allocation failed";
        return false;
      }
      const UINT clear[4] {};
      c->ClearUnorderedAccessViewUint(global_.uav.Get(), clear);  // Explicit pipeline reset ONCE; never owner adoption.
      prepared_ = true;
      return true;
    }

    bool capture_owner(UINT index, ID3D11DeviceContext *c, const completed_ocr_conditioner_research_views_t &actual, const host_sbs_gpu_completion_receipt::expected_t &publication, const expected_t &expected, std::string &error) {
      error.clear();
      if (!prepared_ || index >= 2 || banks_[index].captured || actual.publication != publication || actual.input_region != region_ || actual.field_content != content_ || actual.field_width != fieldW_ || actual.field_height != fieldH_ || publication.domain_tag != domain_) {
        error = "Retirement owner copy identity/layout unavailable";
        return false;
      }
      if (!own_context(c, error)) {
        return false;
      }
      auto &b = banks_[index];
      const auto a = operands(actual);
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (!same_shape(a[i].Get(), b.old[i].resource.Get())) {
          error = "Retirement owner operand shape changed";
          return false;
        }
      }
      b.publication = publication;
      if (!expected_owner(expected, b)) {
        error = "Retirement expected owner mismatch";
        return false;
      }
      saved_cs_t state(c);
      saved_cs_t::unbind(c);
      for (std::size_t i = 0; i < a.size(); ++i) {
        c->CopyResource(b.old[i].resource.Get(), a[i].Get());
      }
      c->UpdateSubresource(b.expected.Get(), 0, nullptr, &expected, 0, 0);
      bind(c, b, nullptr, nullptr, nullptr, false);
      c->CSSetShader(shaders_.shaders[0], nullptr, 0);
      c->Dispatch(1, 1, 1);
      c->CSSetShader(shaders_.shaders[1], nullptr, 0);
      c->Dispatch(std::min((std::max(width_ * height_, 960u * 160u) + 255u) / 256u, 65535u), 1, 1);
      c->CSSetShader(shaders_.shaders[2], nullptr, 0);
      c->Dispatch(1, 1, 1);
      c->CSSetShader(shaders_.shaders[3], nullptr, 0);
      c->Dispatch(1, 1, 1);
      b.captured = true;
      return true;
    }

    bool observe(UINT index, ID3D11DeviceContext *c, ID3D11ShaderResourceView *current, ID3D11ShaderResourceView *sourceReceipt, ID3D11Buffer *currentCB, const expected_t &expected, std::string &error) {
      error.clear();
      if (!prepared_ || index >= 2 || !banks_[index].captured || !current || !sourceReceipt || !currentCB) {
        error = "Retirement observation operands unavailable";
        return false;
      }
      if (!own_context(c, error)) {
        return false;
      }
      auto &b = banks_[index];
      if (!expected_owner(expected, b)) {
        error = "Retirement expected publication mismatch";
        return false;
      }
      if (!same_shape(resource(current).Get(), b.old[0].resource.Get()) || !buffer_shape(resource(sourceReceipt).Get(), 256) || !buffer_shape(currentCB, 128)) {
        error = "Retirement CURRENT band/packet shape changed";
        return false;
      }
      saved_cs_t state(c);
      saved_cs_t::unbind(c);
      c->UpdateSubresource(b.expected.Get(), 0, nullptr, &expected, 0, 0);
      bind(c, b, current, sourceReceipt, currentCB, true);
      c->CSSetShader(shaders_.shaders[4], nullptr, 0);
      c->Dispatch(1, 1, 1);
      c->CSSetShader(shaders_.shaders[5], nullptr, 0);
      c->Dispatch(std::min((width_ * height_ + 255u) / 256u, 65535u), 1, 1);
      saved_cs_t::unbind(c);
      bind(c, b, current, sourceReceipt, currentCB, true);
      c->CSSetShader(shaders_.shaders[6], nullptr, 0);
      c->Dispatch(1, 1, 1);
      c->CSSetShader(shaders_.shaders[7], nullptr, 0);
      c->Dispatch(1, 1, 1);
      return true;
    }

    ID3D11Buffer *expected_buffer(UINT i) const noexcept {
      return i < 2 ? banks_[i].expected.Get() : nullptr;
    }

    ID3D11ShaderResourceView *negative_receipt(UINT i) const noexcept {
      return i < 2 ? banks_[i].receipt.srv.Get() : nullptr;
    }

    bool captured(UINT i) const noexcept {
      return i < 2 && banks_[i].captured;
    }

    const host_sbs_gpu_completion_receipt::expected_t &publication(UINT i) const {
      return banks_.at(i).publication;
    }

    std::array<ComPtr<ID3D11Resource>, 15> export_resources(UINT index) const {
      const auto &b = banks_.at(index);
      return {b.old[0].resource, b.old[1].resource, b.old[2].resource, b.old[3].resource, b.old[4].resource, b.old[5].resource, b.old[6].resource, b.old[7].resource, b.owner.buffer, b.anchors.buffer, b.scratch.buffer, b.scores.buffer, b.receipt.buffer, b.expected, global_.buffer};
    }
  };
}  // namespace models::research_source_pattern_retirement_v1
#endif
