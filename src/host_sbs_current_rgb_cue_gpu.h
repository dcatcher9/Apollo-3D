#pragma once

// EXTERNAL research-only orchestration. No renderer/current-UI authority or producer wire.
// CPU inputs: current RGBA16F, CURRENT CB96, historical OCR256 and its OLD CB96 only.
// Never target P/S/OCR. GPU Body/M remain provisional; semantic status is always UNKNOWN.
#ifdef _WIN32
#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <d3d11.h>
#include <wrl/client.h>

namespace models::host_sbs_current_rgb_cue_gpu_v1 {
  using Microsoft::WRL::ComPtr;
  inline constexpr std::array<const char *, 20u> cue_entries {
    "clear_frame_main", "validate_main", "prepare_unit_main", "palette_main",
    "clear_black_main", "black_learn_main", "black_select_main", "clear_work_main",
    "domain_main", "roots_main", "hash_main", "modal_count_main", "modal_first_main",
    "modal_pair_main", "augment_partial_main", "augment_select_main", "seed_main",
    "unit_body_main", "center_main", "final_main"
  };
  inline constexpr std::array<const char *, 3u> graph_entries {"init_main", "union_main", "reduce_main"};
  // AddRef owns STORAGE only. Contents are mutable until caller copies on its serialized owner.
  struct mutable_views_t {
    UINT width {}, height {}, hash_capacity {}, work_words {};
    ComPtr<ID3D11ShaderResourceView> body, matte, receipt, unit, work, face_data;
    ComPtr<ID3D11ShaderResourceView> parent, root_flags, root_map, free_eligible, zero_seeds;
    ComPtr<ID3D11ShaderResourceView> domain, eligible_mode_bits, seed_mode_bits, body_unit_bits, cover_unit_bits;
    ComPtr<ID3D11Buffer> unit_arguments;
  };

  class resources_t {
    struct texture_t {ComPtr<ID3D11Texture2D> resource; ComPtr<ID3D11ShaderResourceView> srv; ComPtr<ID3D11UnorderedAccessView> uav;};
    struct buffer_t {ComPtr<ID3D11Buffer> resource; ComPtr<ID3D11ShaderResourceView> srv; ComPtr<ID3D11UnorderedAccessView> uav;};
    class saved_cs_t {
    public:
      explicit saved_cs_t(ID3D11DeviceContext *c): context_(c) {
        std::array<ID3D11ShaderResourceView *, 9u> r {};
        std::array<ID3D11UnorderedAccessView *, 8u> w {};
        std::array<ID3D11Buffer *, 3u> b {};
        c->CSGetShaderResources(0u, UINT(r.size()), r.data());
        c->CSGetUnorderedAccessViews(0u, UINT(w.size()), w.data());
        c->CSGetConstantBuffers(0u, UINT(b.size()), b.data());
        for (std::size_t i=0u;i<r.size();++i) srvs_[i].Attach(r[i]);
        for (std::size_t i=0u;i<w.size();++i) uavs_[i].Attach(w[i]);
        for (std::size_t i=0u;i<b.size();++i) constants_[i].Attach(b[i]);
        ID3D11ComputeShader *s=nullptr; std::array<ID3D11ClassInstance *,256u> ci {}; count_=UINT(ci.size());
        c->CSGetShader(&s,ci.data(),&count_); shader_.Attach(s);
        for(UINT i=0u;i<count_;++i) instances_[i].Attach(ci[i]);
      }
      saved_cs_t(const saved_cs_t &)=delete;
      saved_cs_t &operator=(const saved_cs_t &)=delete;
      ~saved_cs_t() {
        unbind(context_);
        std::array<ID3D11ShaderResourceView *,9u> r {}; std::array<ID3D11UnorderedAccessView *,8u> w {};
        std::array<ID3D11Buffer *,3u> b {}; std::array<ID3D11ClassInstance *,256u> ci {};
        for(std::size_t i=0u;i<r.size();++i) r[i]=srvs_[i].Get();
        for(std::size_t i=0u;i<w.size();++i) w[i]=uavs_[i].Get();
        for(std::size_t i=0u;i<b.size();++i) b[i]=constants_[i].Get();
        for(UINT i=0u;i<count_;++i) ci[i]=instances_[i].Get();
        context_->CSSetUnorderedAccessViews(0u,UINT(w.size()),w.data(),nullptr);
        context_->CSSetShaderResources(0u,UINT(r.size()),r.data());
        context_->CSSetConstantBuffers(0u,UINT(b.size()),b.data());
        context_->CSSetShader(shader_.Get(),count_?ci.data():nullptr,count_);
      }
      static void unbind(ID3D11DeviceContext *c) noexcept {
        const std::array<ID3D11ShaderResourceView *,9u> r {}; const std::array<ID3D11UnorderedAccessView *,8u> w {};
        c->CSSetShaderResources(0u,UINT(r.size()),r.data());
        c->CSSetUnorderedAccessViews(0u,UINT(w.size()),w.data(),nullptr);
      }
    private:
      ID3D11DeviceContext *context_;
      std::array<ComPtr<ID3D11ShaderResourceView>,9u> srvs_;
      std::array<ComPtr<ID3D11UnorderedAccessView>,8u> uavs_;
      std::array<ComPtr<ID3D11Buffer>,3u> constants_;
      ComPtr<ID3D11ComputeShader> shader_; std::array<ComPtr<ID3D11ClassInstance>,256u> instances_; UINT count_ {};
    };
  public:
    resources_t()=default;
    resources_t(const resources_t &)=delete;
    resources_t &operator=(const resources_t &)=delete;
    resources_t(resources_t &&)=default;
    resources_t &operator=(resources_t &&)=default;

    // Atomic replacement: failure preserves the previous allocation, but grants no current result.
    // No automatic trimming/alternate mask if required table memory is unavailable.
    bool ensure(ID3D11Device *device,UINT width,UINT height,std::string &error) {
      error.clear(); const std::uint64_t n=std::uint64_t(width)*height;
      if(!device || !width || !height || width>5120u || height>5120u || n>UINT_MAX/16u-1u) {
        error="Invalid current-cue physical native extent/device."; return false;
      }
      if(device_.Get()==device && width_==width && height_==height) return true;
      std::uint64_t t=16u; while(t<2u*n) t*=2u;
      const auto work=std::max({3u*t,std::uint64_t(1458u),4u*((n+255u)/256u)});
      // Both one-dimensional indirect grids must fit the D3D11 X dispatch limit.
      // This is availability rejection only; never truncate work or change pixel semantics.
      if((t+255u)/256u>65535u || (n+255u)/256u>65535u) {
        error="Current-cue one-dimensional indirect grid exceeds D3D11 dispatch capacity.";return false;
      }
      if(t>UINT_MAX || work>UINT_MAX/4u) {error="Current-cue working buffer byte extent overflow.";return false;}
      resources_t next; next.device_=device; next.width_=width;next.height_=height;
      next.pixels_=UINT(n);next.table_=UINT(t);next.work_words_=UINT(work);
      if(!next.allocate(error)) return false;
      *this=std::move(next);return true;
    }

    // Serialized immediate-context owner only. Inputs and both CBs stay immutable throughout.
    // Working resources must be unbound from other pipeline stages. No concurrent ensure/dispatch.
    // True means commands were submitted, NOT completion/current authority/semantic validity.
    // Fresh calls always clear total Body/M. No ROOT-reuse/old-mask hold in this helper.
    bool dispatch(ID3D11DeviceContext *context,
      const std::array<ID3D11ComputeShader *,20u> &cue,
      const std::array<ID3D11ComputeShader *,3u> &graph,
      ID3D11Buffer *current_cb96,ID3D11Buffer *old_cb96,
      ID3D11ShaderResourceView *current_rgba16f,ID3D11ShaderResourceView *old_ocr256,
      bool prior_present,std::string &error) {
      error.clear();
      if(!context || !device_ || context->GetType()!=D3D11_DEVICE_CONTEXT_IMMEDIATE || !current_cb96 || !old_cb96 || !current_rgba16f || !old_ocr256) {
        error="Current-cue dispatch requires its immediate device/current+historical operands.";return false;
      }
      for(auto *s:cue) if(!s) {error="Incomplete20-entry cue shader set.";return false;}
      for(auto *s:graph) if(!s) {error="Incomplete3-entry frozen graph shader set.";return false;}
      const auto same_device=[&](ID3D11DeviceChild *child) {ComPtr<ID3D11Device> d;child->GetDevice(&d);return d.Get()==device_.Get();};
      if(!same_device(context) || !same_device(current_cb96) || !same_device(old_cb96) || !same_device(current_rgba16f) || !same_device(old_ocr256)) {
        error="Current-cue context/input device mismatch.";return false;
      }
      for(auto *s:cue) if(!same_device(s)) {error="Current-cue shader device mismatch.";return false;}
      for(auto *s:graph) if(!same_device(s)) {error="Current-cue graph device mismatch.";return false;}
      ComPtr<ID3D11Predicate> predicate;BOOL predicate_value=FALSE;context->GetPredication(&predicate,&predicate_value);
      if(predicate) {error="Current-cue research dispatch requires no active predication.";return false;}
      for(auto *b:{current_cb96,old_cb96}) {D3D11_BUFFER_DESC d {};b->GetDesc(&d);if(d.ByteWidth!=96u || !(d.BindFlags&D3D11_BIND_CONSTANT_BUFFER)) {error="Both source contexts must be exact CB96.";return false;}}
      D3D11_SHADER_RESOURCE_VIEW_DESC sd {};current_rgba16f->GetDesc(&sd);
      ComPtr<ID3D11Resource> source_resource;current_rgba16f->GetResource(&source_resource);
      ComPtr<ID3D11Texture2D> source;D3D11_TEXTURE2D_DESC td {};
      if(FAILED(source_resource.As(&source))) {error="Current RGB must be a physical texture.";return false;}
      source->GetDesc(&td);
      if(sd.ViewDimension!=D3D11_SRV_DIMENSION_TEXTURE2D || sd.Texture2D.MostDetailedMip!=0u || sd.Texture2D.MipLevels!=1u ||
        sd.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT || td.Width!=width_ || td.Height!=height_ || td.MipLevels!=1u || td.ArraySize!=1u || td.SampleDesc.Count!=1u) {
        error="Current RGB must be the exact allocated native single-mip RGBA16F band.";return false;
      }
      old_ocr256->GetDesc(&sd);ComPtr<ID3D11Resource> ocr_resource;old_ocr256->GetResource(&ocr_resource);
      ComPtr<ID3D11Buffer> ocr;D3D11_BUFFER_DESC od {};
      if(FAILED(ocr_resource.As(&ocr))) {error="Historical OCR must be structured256uint.";return false;}
      ocr->GetDesc(&od);
      if(sd.ViewDimension!=D3D11_SRV_DIMENSION_BUFFER || sd.Buffer.FirstElement!=0u || sd.Buffer.NumElements!=256u ||
        od.ByteWidth!=1024u || od.StructureByteStride!=4u || !(od.MiscFlags&D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)) {
        error="Historical OCR view must expose exact256uint.";return false;
      }
      const saved_cs_t saved {context}; emit(context,cue,graph,current_cb96,old_cb96,current_rgba16f,old_ocr256,prior_present);
      return true;
    }
    [[nodiscard]] mutable_views_t mutable_views() const {
      return {width_,height_,table_,work_words_,out_body_.srv,out_matte_.srv,frame_.srv,unit_.srv,work_.srv,face_.srv,
        parent_.srv,flags_.srv,root_map_.srv,free_.srv,zero_.srv,domain_.srv,eligible_.srv,seeds_.srv,body_bits_.srv,cover_bits_.srv,args_.resource};
    }
  private:
    void emit(ID3D11DeviceContext *c,const std::array<ID3D11ComputeShader *,20u> &cue,
      const std::array<ID3D11ComputeShader *,3u> &graph,ID3D11Buffer *current,ID3D11Buffer *old,
      ID3D11ShaderResourceView *rgb,ID3D11ShaderResourceView *ocr,bool prior) {
      saved_cs_t::unbind(c);const std::array<UINT,4u> zeros {};
      for(auto *t:{&out_body_,&out_matte_,&body_bits_,&cover_bits_}) c->ClearUnorderedAccessViewUint(t->uav.Get(),zeros.data());
      const auto stage=[&](UINT shader,texture_t *a,texture_t *b,texture_t *d,bool indirect,UINT offset,bool write_args) {
        saved_cs_t::unbind(c);
        ID3D11Buffer *cb[3] {current,old,request_.Get()};c->CSSetConstantBuffers(0u,3u,cb);
        std::array<ID3D11ShaderResourceView *,9u> r {rgb,ocr,parent_.srv.Get(),flags_.srv.Get(),root_map_.srv.Get(),free_.srv.Get(),domain_.srv.Get(),body_bits_.srv.Get(),cover_bits_.srv.Get()};
        // Stage-local t5 alias: unit-body resolves the eligible MODE BITS emitted by seed_main,
        // rather than the earlier binary nonblack graph input.
        if(shader==17u) r[5u]=eligible_.srv.Get();
        const std::array<texture_t *,5u> readable {&root_map_,&free_,&domain_,&body_bits_,&cover_bits_};
        for(std::size_t i=0u;i<readable.size();++i) if(readable[i]==a || readable[i]==b || readable[i]==d) r[4u+i]=nullptr;
        c->CSSetShaderResources(0u,UINT(r.size()),r.data());
        const std::array<ID3D11UnorderedAccessView *,8u> u {unit_.uav.Get(),frame_.uav.Get(),work_.uav.Get(),face_.uav.Get(),
          a?a->uav.Get():nullptr,b?b->uav.Get():nullptr,d?d->uav.Get():nullptr,write_args?args_.uav.Get():nullptr};
        c->CSSetUnorderedAccessViews(0u,UINT(u.size()),u.data(),nullptr);c->CSSetShader(cue[shader],nullptr,0u);
        if(indirect) c->DispatchIndirect(args_.resource.Get(),offset);
        else if(shader==2u || shader==6u || shader==15u || shader==18u) c->Dispatch(1u,1u,1u);
        else c->Dispatch((width_+15u)/16u,(height_+15u)/16u,1u);
        saved_cs_t::unbind(c);
      };
      const std::array<UINT,4u> initial {0u,table_,prior?1u:0u,0u};c->UpdateSubresource(request_.Get(),0u,nullptr,initial.data(),0u,0u);
      stage(0u,&body_bits_,&cover_bits_,&out_body_,false,0u,false);
      stage(1u,nullptr,nullptr,nullptr,false,0u,false);
      for(UINT slot=0u;slot<8u;++slot) {
        const std::array<UINT,4u> req {slot,table_,prior?1u:0u,0u};c->UpdateSubresource(request_.Get(),0u,nullptr,req.data(),0u,0u);
        stage(2u,nullptr,nullptr,nullptr,false,0u,true);
        stage(3u,nullptr,nullptr,nullptr,true,0u,false);
        stage(4u,nullptr,nullptr,nullptr,true,0u,false);
        stage(5u,nullptr,nullptr,nullptr,true,12u,false);
        stage(6u,nullptr,nullptr,nullptr,false,0u,true);
        stage(7u,nullptr,nullptr,nullptr,true,24u,false);
        stage(8u,&free_,&zero_,&domain_,true,12u,false);
        saved_cs_t::unbind(c);
        ID3D11Buffer *gcb=graph_cb_.Get();c->CSSetConstantBuffers(0u,1u,&gcb);
        const std::array<ID3D11ShaderResourceView *,4u> gr {free_.srv.Get(),zero_.srv.Get(),domain_.srv.Get(),domain_.srv.Get()};
        const std::array<ID3D11UnorderedAccessView *,8u> gu {parent_.uav.Get(),flags_.uav.Get(),out_body_.uav.Get(),out_matte_.uav.Get(),nullptr,nullptr,nullptr,nullptr};
        c->CSSetShaderResources(0u,4u,gr.data());c->CSSetUnorderedAccessViews(0u,8u,gu.data(),nullptr);
        for(auto *s:graph) {c->CSSetShader(s,nullptr,0u);c->DispatchIndirect(args_.resource.Get(),12u);}
        saved_cs_t::unbind(c);
        stage(9u,&root_map_,nullptr,nullptr,true,12u,false);
        stage(10u,nullptr,nullptr,nullptr,true,12u,false);
        stage(11u,nullptr,nullptr,nullptr,true,24u,false);
        stage(12u,nullptr,nullptr,nullptr,true,24u,false);
        stage(13u,nullptr,nullptr,nullptr,true,12u,false);
        for(UINT step=0u;step<9u;++step) {stage(14u,nullptr,nullptr,nullptr,true,36u,false);stage(15u,nullptr,nullptr,nullptr,false,0u,true);}
        stage(16u,&eligible_,&seeds_,nullptr,true,12u,false);
        stage(17u,&body_bits_,&cover_bits_,nullptr,true,12u,false);
      }
      stage(18u,nullptr,nullptr,nullptr,false,0u,false);
      stage(19u,&out_body_,&out_matte_,nullptr,false,0u,false);
    }
    bool allocate(std::string &error) {
      const auto fail=[&](const char *s){error=s;return false;};
      const auto texture=[&](texture_t &t) {D3D11_TEXTURE2D_DESC d {};d.Width=width_;d.Height=height_;d.MipLevels=d.ArraySize=1u;d.Format=DXGI_FORMAT_R32_UINT;d.SampleDesc.Count=1u;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        return SUCCEEDED(device_->CreateTexture2D(&d,nullptr,&t.resource)) && SUCCEEDED(device_->CreateShaderResourceView(t.resource.Get(),nullptr,&t.srv)) && SUCCEEDED(device_->CreateUnorderedAccessView(t.resource.Get(),nullptr,&t.uav));};
      for(auto *t:{&free_,&zero_,&domain_,&root_map_,&eligible_,&seeds_,&body_bits_,&cover_bits_,&out_body_,&out_matte_}) if(!texture(*t)) return fail("Could not allocate current-cue native texture.");
      const auto structured=[&](UINT words,buffer_t &b) {D3D11_BUFFER_DESC d {};d.ByteWidth=words*4u;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;d.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;d.StructureByteStride=4u;
        return SUCCEEDED(device_->CreateBuffer(&d,nullptr,&b.resource)) && SUCCEEDED(device_->CreateShaderResourceView(b.resource.Get(),nullptr,&b.srv)) && SUCCEEDED(device_->CreateUnorderedAccessView(b.resource.Get(),nullptr,&b.uav));};
      if(!structured(256u,unit_) || !structured(512u,frame_) || !structured(work_words_,work_) || !structured(4u*(pixels_+1u),face_) || !structured(pixels_+1u,parent_) || !structured(pixels_+1u,flags_)) return fail("Could not allocate required current-cue structured memory.");
      D3D11_BUFFER_DESC a {};a.ByteWidth=60u;a.Usage=D3D11_USAGE_DEFAULT;a.BindFlags=D3D11_BIND_UNORDERED_ACCESS;a.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS|D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_UNORDERED_ACCESS_VIEW_DESC u {};u.Format=DXGI_FORMAT_R32_TYPELESS;u.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;u.Buffer.NumElements=15u;u.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
      if(FAILED(device_->CreateBuffer(&a,nullptr,&args_.resource)) || FAILED(device_->CreateUnorderedAccessView(args_.resource.Get(),&u,&args_.uav))) return fail("Could not allocate current-cue writable arguments.");
      const auto cb=[&](const std::array<UINT,4u> &v,ComPtr<ID3D11Buffer> &b) {D3D11_BUFFER_DESC d {};d.ByteWidth=16u;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;const D3D11_SUBRESOURCE_DATA i {v.data(),0u,0u};return SUCCEEDED(device_->CreateBuffer(&d,&i,&b));};
      if(!cb({0u,table_,0u,0u},request_) || !cb({width_,height_,pixels_,0u},graph_cb_)) return fail("Could not allocate current-cue graph/request constants.");
      return true;
    }
    ComPtr<ID3D11Device> device_;UINT width_ {},height_ {},pixels_ {},table_ {},work_words_ {};
    texture_t free_,zero_,domain_,root_map_,eligible_,seeds_,body_bits_,cover_bits_,out_body_,out_matte_;
    buffer_t unit_,frame_,work_,face_,parent_,flags_,args_;ComPtr<ID3D11Buffer> request_,graph_cb_;
  };
}
#endif
