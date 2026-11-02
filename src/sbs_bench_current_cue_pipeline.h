#pragma once
// EXTERNAL V6 transport orchestration only. All cue outputs remain semantic UNKNOWN.
// No target B/C P/S/OCR, no current-A OCR conditioning, no renderer/current-UI authority.
#ifdef _WIN32
  #include "host_sbs_current_rgb_cue_gpu.h"
  #include "video_depth_estimator.h"

  #include <array>
  #include <cstdint>
  #include <d3d11.h>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace models::research_current_cue_pipeline_v6 {
  using Microsoft::WRL::ComPtr;

  struct copy_t {
    ComPtr<ID3D11Resource> resource;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11Buffer> buffer;
  };

  class pipeline_t {
    class saved_cs_t {
      ID3D11DeviceContext *c_;
      std::array<ComPtr<ID3D11ShaderResourceView>, 2> r_;
      ComPtr<ID3D11UnorderedAccessView> u_;
      ComPtr<ID3D11Buffer> b_;
      ComPtr<ID3D11ComputeShader> s_;
      std::array<ComPtr<ID3D11ClassInstance>, 256> i_;
      UINT n_ = 256u;

    public:
      explicit saved_cs_t(ID3D11DeviceContext *c):
          c_(c) {
        ID3D11ShaderResourceView *r[2] {};
        c->CSGetShaderResources(0, 2, r);
        for (UINT k = 0; k < 2; ++k) {
          r_[k].Attach(r[k]);
        }
        c->CSGetUnorderedAccessViews(0, 1, u_.GetAddressOf());
        c->CSGetConstantBuffers(0, 1, b_.GetAddressOf());
        ID3D11ClassInstance *i[256] {};
        c->CSGetShader(s_.GetAddressOf(), i, &n_);
        for (UINT k = 0; k < n_; ++k) {
          i_[k].Attach(i[k]);
        }
      }

      ~saved_cs_t() {
        unbind(c_);
        auto *u = u_.Get();
        c_->CSSetUnorderedAccessViews(0, 1, &u, nullptr);
        ID3D11ShaderResourceView *r[2] {r_[0].Get(), r_[1].Get()};
        c_->CSSetShaderResources(0, 2, r);
        auto *b = b_.Get();
        c_->CSSetConstantBuffers(0, 1, &b);
        ID3D11ClassInstance *i[256] {};
        for (UINT k = 0; k < n_; ++k) {
          i[k] = i_[k].Get();
        }
        c_->CSSetShader(s_.Get(), n_ ? i : nullptr, n_);
      }

      static void unbind(ID3D11DeviceContext *c) {
        ID3D11ShaderResourceView *r[2] {};
        ID3D11UnorderedAccessView *u = nullptr;
        ID3D11Buffer *b = nullptr;
        c->CSSetShaderResources(0, 2, r);
        c->CSSetUnorderedAccessViews(0, 1, &u, nullptr);
        c->CSSetConstantBuffers(0, 1, &b);
        c->CSSetShader(nullptr, nullptr, 0);
      }
    };

    static ComPtr<ID3D11Resource> resource(ID3D11ShaderResourceView *s) {
      ComPtr<ID3D11Resource> r;
      if (s) {
        s->GetResource(&r);
      }
      return r;
    }

    bool own_device(ID3D11DeviceChild *child) const {
      if (!child) {
        return false;
      }
      ComPtr<ID3D11Device> d;
      child->GetDevice(&d);
      return d.Get() == device_.Get();
    }

    static bool same_layout(ID3D11Resource *a, ID3D11Resource *b) {
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
        ComPtr<ID3D11Buffer> aa, bb;
        if (FAILED(a->QueryInterface(IID_PPV_ARGS(&aa))) || FAILED(b->QueryInterface(IID_PPV_ARGS(&bb)))) {
          return false;
        }
        D3D11_BUFFER_DESC da {}, db {};
        aa->GetDesc(&da);
        bb->GetDesc(&db);
        return da.ByteWidth == db.ByteWidth && da.StructureByteStride == db.StructureByteStride;
      }
      if (x == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
        ComPtr<ID3D11Texture2D> aa, bb;
        if (FAILED(a->QueryInterface(IID_PPV_ARGS(&aa))) || FAILED(b->QueryInterface(IID_PPV_ARGS(&bb)))) {
          return false;
        }
        D3D11_TEXTURE2D_DESC da {}, db {};
        aa->GetDesc(&da);
        bb->GetDesc(&db);
        return da.Width == db.Width && da.Height == db.Height && da.Format == db.Format && da.MipLevels == db.MipLevels && da.ArraySize == db.ArraySize && da.SampleDesc.Count == db.SampleDesc.Count && da.SampleDesc.Quality == db.SampleDesc.Quality;
      }
      return false;
    }

    bool clone(ID3D11Resource *source, bool make_srv, copy_t &out) {
      if (!source) {
        return false;
      }
      D3D11_RESOURCE_DIMENSION type {};
      source->GetType(&type);
      if (type == D3D11_RESOURCE_DIMENSION_BUFFER) {
        ComPtr<ID3D11Buffer> b;
        if (FAILED(source->QueryInterface(IID_PPV_ARGS(&b)))) {
          return false;
        }
        D3D11_BUFFER_DESC d {};
        b->GetDesc(&d);
        d.Usage = D3D11_USAGE_DEFAULT;
        d.CPUAccessFlags = 0;
        d.MiscFlags &= D3D11_RESOURCE_MISC_BUFFER_STRUCTURED | D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
        if (FAILED(device_->CreateBuffer(&d, nullptr, &out.buffer)) || FAILED(out.buffer.As(&out.resource))) {
          return false;
        }
      } else if (type == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
        ComPtr<ID3D11Texture2D> t, copy;
        if (FAILED(source->QueryInterface(IID_PPV_ARGS(&t)))) {
          return false;
        }
        D3D11_TEXTURE2D_DESC d {};
        t->GetDesc(&d);
        if (d.MipLevels != 1 || d.ArraySize != 1 || d.SampleDesc.Count != 1) {
          return false;
        }
        d.Usage = D3D11_USAGE_DEFAULT;
        d.CPUAccessFlags = d.MiscFlags = 0;
        if (FAILED(device_->CreateTexture2D(&d, nullptr, &copy)) || FAILED(copy.As(&out.resource))) {
          return false;
        }
      } else {
        return false;
      }
      return !make_srv || SUCCEEDED(device_->CreateShaderResourceView(out.resource.Get(), nullptr, &out.srv));
    }

    bool words(UINT count, bool uav, bool initialized_zero, copy_t &out, ComPtr<ID3D11UnorderedAccessView> *write = nullptr) {
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = count * 4u;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE | (uav ? D3D11_BIND_UNORDERED_ACCESS : 0u);
      d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      d.StructureByteStride = 4u;
      const std::vector<UINT> zeros(count, 0u);
      const D3D11_SUBRESOURCE_DATA data {zeros.data(), 0u, 0u};
      if (FAILED(device_->CreateBuffer(&d, initialized_zero ? &data : nullptr, &out.buffer)) || FAILED(out.buffer.As(&out.resource)) || FAILED(device_->CreateShaderResourceView(out.buffer.Get(), nullptr, &out.srv))) {
        return false;
      }
      return !uav || (write && SUCCEEDED(device_->CreateUnorderedAccessView(out.buffer.Get(), nullptr, write->GetAddressOf())));
    }

    bool constant(UINT bytes, bool initialized_zero, copy_t &out) {
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = bytes;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const std::vector<UINT> zeros(bytes / 4u, 0u);
      const D3D11_SUBRESOURCE_DATA data {zeros.data(), 0u, 0u};
      return SUCCEEDED(device_->CreateBuffer(&d, initialized_zero ? &data : nullptr, &out.buffer)) && SUCCEEDED(out.buffer.As(&out.resource));
    }

    static std::array<ComPtr<ID3D11Resource>, 7> operands(const completed_ocr_conditioner_research_views_t &v) {
      return {resource(v.source_band.Get()), resource(v.ocr_record.Get()), ComPtr<ID3D11Resource>(v.constants.Get()), resource(v.uip.Get()), resource(v.sct.Get()), ComPtr<ID3D11Resource>(v.transaction.Get()), ComPtr<ID3D11Resource>(v.dispatch_arguments.Get())};
    }

  public:
    // ONLY before authored epoch, after genuine empty warmup. No CPU resource readback.
    bool prepare(ID3D11Device *d, ID3D11DeviceContext *c, const completed_ocr_conditioner_research_views_t &warm, const std::array<ID3D11ComputeShader *, 20> &cue, const std::array<ID3D11ComputeShader *, 3> &graph, ID3D11ComputeShader *wire, std::string &error) {
      error.clear();
      if (prepared_ || !d || !c || !wire || c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
        error = "Invalid/duplicate V6 preparation";
        return false;
      }
      ComPtr<ID3D11Predicate> predicate;
      BOOL predicate_value = FALSE;
      c->GetPredication(&predicate, &predicate_value);
      if (predicate) {
        error = "V6 preparation rejects predication before copies";
        return false;
      }
      device_ = d;
      if (!own_device(c) || !own_device(wire)) {
        error = "V6 preparation device mismatch";
        return false;
      }
      const auto src = resource(warm.source_band.Get());
      ComPtr<ID3D11Texture2D> texture;
      if (!src || FAILED(src.As(&texture))) {
        error = "V6 warmup completed native source missing";
        return false;
      }
      D3D11_TEXTURE2D_DESC td {};
      texture->GetDesc(&td);
      width_ = td.Width;
      height_ = td.Height;
      if (td.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || td.MipLevels != 1 || td.ArraySize != 1 || td.SampleDesc.Count != 1 || width_ != warm.input_region.width() || !height_ || height_ > warm.input_region.height()) {
        error = "V6 warmup source layout invalid";
        return false;
      }
      for (std::size_t i = 0; i < cue.size(); ++i) {
        if (!own_device(cue[i])) {
          error = "Missing/mismatched V6 cue entry";
          return false;
        }
        cue_[i] = cue[i];
      }
      for (std::size_t i = 0; i < graph.size(); ++i) {
        if (!own_device(graph[i])) {
          error = "Missing/mismatched V6 graph entry";
          return false;
        }
        graph_[i] = graph[i];
      }
      wire_ = wire;
      if (!helper_.ensure(d, width_, height_, error)) {
        return false;
      }
      const auto input = operands(warm);
      for (std::size_t i = 0; i < input.size(); ++i) {
        if (!own_device(input[i].Get()) || !clone(input[i].Get(), i == 1u, frozen_A_[i])) {
          error = "V6 immutable-A resource preallocation failed";
          return false;
        }
      }
      D3D11_BUFFER_DESC cbdesc {};
      warm.constants->GetDesc(&cbdesc);
      D3D11_SHADER_RESOURCE_VIEW_DESC od {};
      warm.ocr_record->GetDesc(&od);
      if (cbdesc.ByteWidth != 96u || !(cbdesc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) || od.ViewDimension != D3D11_SRV_DIMENSION_BUFFER || od.Buffer.FirstElement != 0u || od.Buffer.NumElements != 256u) {
        error = "V6 old CB/OCR physical shape invalid";
        return false;
      }
      if (!words(24u, false, false, geometry_) || !words(24u, true, false, current_words_, &current_words_uav_) || !constant(96u, false, current_cb_) || !constant(96u, true, zero_cb_) || !words(256u, false, true, zero_ocr_)) {
        error = "V6 current/zero context preallocation failed";
        return false;
      }
      c->CopyResource(geometry_.resource.Get(), warm.constants.Get());  // immutable geometry only; no old clock used
      warm_region_ = warm.input_region;
      warm_field_width_ = warm.field_width;
      warm_field_height_ = warm.field_height;
      warm_content_ = warm.field_content;
      warm_domain_ = warm.publication.domain_tag;
      prepared_ = true;
      return true;
    }

    // Copies literal actual A resources ONCE, before B submission. True means copy commands,
    // not GPU completion/fresh OCR verdict; caller must Map snapshots and authenticate real receipts.
    bool capture_A(ID3D11DeviceContext *c, const completed_ocr_conditioner_research_views_t &actual, const host_sbs_gpu_completion_receipt::expected_t &expected, std::string &error) {
      error.clear();
      if (!prepared_ || captured_A_ || !c || c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !own_device(c) || actual.publication != expected || expected.frame_id != 2u || expected.observation_timestamp_us != 33334u || expected.domain_tag != warm_domain_ || actual.input_region != warm_region_ || actual.field_width != warm_field_width_ || actual.field_height != warm_field_height_ || actual.field_content != warm_content_) {
        error = "V6 actual A identity/profile unavailable";
        return false;
      }
      ComPtr<ID3D11Predicate> predicate;
      BOOL predicate_value = FALSE;
      c->GetPredication(&predicate, &predicate_value);
      if (predicate) {
        error = "V6 A capture rejects predication before copies";
        return false;
      }
      const auto input = operands(actual);
      for (std::size_t i = 0; i < input.size(); ++i) {
        if (!own_device(input[i].Get()) || !same_layout(frozen_A_[i].resource.Get(), input[i].Get())) {
          error = "V6 A layout changed after warmup";
          return false;
        }
      }
      for (std::size_t i = 0; i < input.size(); ++i) {
        c->CopyResource(frozen_A_[i].resource.Get(), input[i].Get());
      }
      A_publication_ = expected;
      captured_A_ = true;
      return true;
    }

    // Fully GPU currentCSP->currentCB->frozen cue. All allocations were made in prepare().
    // A uses canonicalzero prior; B/C use the private immutable actual-A OCR/CB only.
    bool dispatch(ID3D11DeviceContext *c, ID3D11ShaderResourceView *current_source, ID3D11ShaderResourceView *source_receipt, ID3D11Buffer *source_constants128, bool apply_A, std::string &error) {
      error.clear();
      if (!prepared_ || !captured_A_ || !c || !current_source || !source_receipt || !source_constants128 || c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
        error = "V6 current source/A context unavailable";
        return false;
      }
      ComPtr<ID3D11Device> d;
      c->GetDevice(&d);
      if (d.Get() != device_.Get()) {
        error = "V6 context device mismatch";
        return false;
      }
      if (!own_device(current_source) || !own_device(source_receipt) || !own_device(source_constants128)) {
        error = "V6 source input device mismatch";
        return false;
      }
      ComPtr<ID3D11Predicate> predicate;
      BOOL value = FALSE;
      c->GetPredication(&predicate, &value);
      if (predicate) {
        error = "V6 pipeline rejects predication before commands";
        return false;
      }
      D3D11_BUFFER_DESC cbdesc {};
      source_constants128->GetDesc(&cbdesc);
      if (cbdesc.ByteWidth != 128u || !(cbdesc.BindFlags & D3D11_BIND_CONSTANT_BUFFER)) {
        error = "V6 source CB must be literal128B";
        return false;
      }
      const saved_cs_t save {c};
      saved_cs_t::unbind(c);
      ID3D11Buffer *cb = source_constants128;
      c->CSSetConstantBuffers(0, 1, &cb);
      ID3D11ShaderResourceView *reads[2] {geometry_.srv.Get(), source_receipt};
      auto *write = current_words_uav_.Get();
      c->CSSetShaderResources(0, 2, reads);
      c->CSSetUnorderedAccessViews(0, 1, &write, nullptr);
      c->CSSetShader(wire_.Get(), nullptr, 0);
      c->Dispatch(1, 1, 1);
      saved_cs_t::unbind(c);
      c->CopyResource(current_cb_.resource.Get(), current_words_.resource.Get());
      std::array<ID3D11ComputeShader *, 20> cue {};
      std::array<ID3D11ComputeShader *, 3> graph {};
      for (std::size_t i = 0; i < cue.size(); ++i) {
        cue[i] = cue_[i].Get();
      }
      for (std::size_t i = 0; i < graph.size(); ++i) {
        graph[i] = graph_[i].Get();
      }
      const auto ocr = apply_A ? frozen_A_[1].srv.Get() : zero_ocr_.srv.Get();
      const auto oldcb = apply_A ? frozen_A_[2].buffer.Get() : zero_cb_.buffer.Get();
      if (!helper_.dispatch(c, cue, graph, current_cb_.buffer.Get(), oldcb, current_source, ocr, apply_A, error)) {
        return false;
      }
      last_prior_A_ = apply_A;
      return true;
    }

    // Mutable current outputs + immutable-A copies, in exact harness slot15..28 order.
    // Copy ALL into independently owned per-phase CPU vectors before next CSP/helper dispatch.
    std::array<ComPtr<ID3D11Resource>, 14> export_resources(ID3D11ShaderResourceView *current_source) const {
      const auto v = helper_.mutable_views();
      return {resource(v.body.Get()), resource(v.matte.Get()), resource(v.receipt.Get()), resource(current_source), current_cb_.resource, last_prior_A_ ? frozen_A_[1].resource : zero_ocr_.resource, last_prior_A_ ? frozen_A_[2].resource : zero_cb_.resource, frozen_A_[0].resource, frozen_A_[1].resource, frozen_A_[2].resource, frozen_A_[3].resource, frozen_A_[4].resource, frozen_A_[5].resource, frozen_A_[6].resource};
    }

    const host_sbs_gpu_completion_receipt::expected_t &A_publication() const {
      return A_publication_;
    }

    bool captured_A() const {
      return captured_A_;
    }

  private:
    ComPtr<ID3D11Device> device_;
    UINT width_ = 0, height_ = 0;
    bool prepared_ = false, captured_A_ = false, last_prior_A_ = false;
    depth_input_region_t warm_region_ {};
    depth_tensor_content_rect_t warm_content_ {};
    UINT warm_field_width_ = 0, warm_field_height_ = 0;
    std::uint64_t warm_domain_ = 0;
    host_sbs_gpu_completion_receipt::expected_t A_publication_ {};
    host_sbs_current_rgb_cue_gpu_v1::resources_t helper_;
    std::array<ComPtr<ID3D11ComputeShader>, 20> cue_;
    std::array<ComPtr<ID3D11ComputeShader>, 3> graph_;
    ComPtr<ID3D11ComputeShader> wire_;
    std::array<copy_t, 7> frozen_A_;
    copy_t geometry_, current_words_, current_cb_, zero_cb_, zero_ocr_;
    ComPtr<ID3D11UnorderedAccessView> current_words_uav_;
  };
}  // namespace models::research_current_cue_pipeline_v6
#endif
