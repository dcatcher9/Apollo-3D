#pragma once
// DEFAULT-OFF private command lane. Source transport + negative permission ONLY.
#ifdef _WIN32
#include "host_sbs_pending_source_pattern_retirement_gpu.h"
#include "host_sbs_material_capture_identity.h"
#include "host_sbs_shader_cache.h"
#include <filesystem>
#include <optional>

namespace models::research_pending_current_colour_v1 {
  using Microsoft::WRL::ComPtr;
  using material_identity_t=research_material_capture_v1::identity_t;
  using snapshot_t=research_current_source_snapshot_v1::constants_t;
  namespace retirement=research_pending_source_pattern_retirement_v1;
  struct configuration_t {
    bool enabled=false;
    std::filesystem::path shader_root;
    std::string exact_source_closure;
  };
  struct scoped_negative_t {
    ID3D11DeviceContext *context=nullptr;
    std::array<ComPtr<ID3D11Buffer>,2> oldCB;
    std::array<ComPtr<ID3D11ShaderResourceView>,2> oldSRV;
    bool bind(ID3D11DeviceContext *c,ID3D11Buffer *cb,ID3D11ShaderResourceView *receipt,ID3D11Buffer *currentCB,ID3D11ShaderResourceView *currentReceipt) noexcept {
      if(context || !c || !cb || !receipt || !currentCB || !currentReceipt || c->GetType()!=D3D11_DEVICE_CONTEXT_IMMEDIATE)return false;
      ComPtr<ID3D11Predicate> p;BOOL value=FALSE;c->GetPredication(&p,&value);if(p)return false;
      ID3D11Buffer *b[2]{};ID3D11ShaderResourceView *s[2]{};
      c->PSGetConstantBuffers(6u,2u,b);c->PSGetShaderResources(11u,2u,s);
      for(UINT i=0;i<2u;++i){oldCB[i].Attach(b[i]);oldSRV[i].Attach(s[i]);}
      context=c;ID3D11Buffer *bindCB[2]{cb,currentCB};ID3D11ShaderResourceView *bindSRV[2]{receipt,currentReceipt};
      c->PSSetConstantBuffers(6u,2u,bindCB);c->PSSetShaderResources(11u,2u,bindSRV);return true;
    }
    ~scoped_negative_t(){if(context){ID3D11ShaderResourceView *s[2]{oldSRV[0].Get(),oldSRV[1].Get()};ID3D11Buffer *b[2]{oldCB[0].Get(),oldCB[1].Get()};context->PSSetShaderResources(11u,2u,s);context->PSSetConstantBuffers(6u,2u,b);}}
    scoped_negative_t()=default;scoped_negative_t(const scoped_negative_t &)=delete;
  };
  class resources_t {
    std::array<ComPtr<ID3D11ComputeShader>,3> snapshot_shaders_;
    std::array<ComPtr<ID3D11ComputeShader>,8> retirement_shaders_;
    ComPtr<ID3D11PixelShader> combined_ps_;
    research_current_source_snapshot_v1::resources_t snapshot_;
    retirement::resources_t retirement_;
    depth_input_region_t region_{};depth_tensor_content_rect_t content_{};
    std::uint32_t band_height_=0,field_width_=0,field_height_=0;
    std::uint64_t domain_=0,epoch_=0,permission_=0;
    bool prepared_=false;
    std::optional<UINT> active_bank_;
    std::optional<queued_same_source_gpu_continuation_research_views_t> ticket_;
    snapshot_t last_current_{};
    static std::array<std::uint32_t,2> split(std::uint64_t v) noexcept {return {std::uint32_t(v),std::uint32_t(v>>32u)};}
  public:
    inline static constexpr std::array<host_sbs_shader_cache::shader_spec,12> specs{{
      {"host_sbs_current_source_snapshot_cs.hlsl","clear_main","cs_5_0"},
      {"host_sbs_current_source_snapshot_cs.hlsl","snapshot_main","cs_5_0"},
      {"host_sbs_current_source_snapshot_cs.hlsl","resolve_main","cs_5_0"},
      {"host_sbs_pending_source_pattern_retirement_cs.hlsl","owner_init_main","cs_5_0"},
      {"host_sbs_pending_source_pattern_retirement_cs.hlsl","owner_validate_main","cs_5_0"},
      {"host_sbs_pending_source_pattern_retirement_cs.hlsl","owner_palette_main","cs_5_0"},
      {"host_sbs_pending_source_pattern_retirement_cs.hlsl","owner_anchors_main","cs_5_0"},
      {"host_sbs_pending_source_pattern_retirement_cs.hlsl","observe_init_main","cs_5_0"},
      {"host_sbs_pending_source_pattern_retirement_cs.hlsl","observe_validate_main","cs_5_0"},
      {"host_sbs_pending_source_pattern_retirement_cs.hlsl","compare_main","cs_5_0"},
      {"host_sbs_pending_source_pattern_retirement_cs.hlsl","retire_main","cs_5_0"},
      {"sbs_reprojection_v2_pending_current_colour_ps.hlsl","main_ps","ps_5_0"}
    }};
    // The genuine completed warm owner supplies descriptor shapes ONLY. All compilation and
    // allocation happen here before the held B interval. No per-frame get/ensure/prepare.
    bool prepare(ID3D11Device *device,ID3D11DeviceContext *context,
      const completed_ocr_conditioner_research_views_t &warm,const configuration_t &cfg,
      std::uint64_t capture_epoch,std::uint64_t permission,std::string &error) {
      error.clear();if(prepared_ || !cfg.enabled || !device || !context || !capture_epoch || !permission || cfg.exact_source_closure.empty())return false;
      const auto source=host_sbs_shader_cache::snapshot_sources(cfg.shader_root,specs);
      if(!source || host_sbs_shader_cache::source_closure_sha256(source)!=cfg.exact_source_closure) {error="Pending colour source closure mismatch";return false;}
      for(std::size_t i=0;i<specs.size();++i){const auto code=host_sbs_shader_cache::get(source,specs[i]);if(!code){error="Pending colour bytecode unavailable";return false;}
        HRESULT hr=E_FAIL;
        if(i<3u)hr=device->CreateComputeShader(code->data(),code->size(),nullptr,&snapshot_shaders_[i]);
        else if(i<11u)hr=device->CreateComputeShader(code->data(),code->size(),nullptr,&retirement_shaders_[i-3u]);
        else hr=device->CreatePixelShader(code->data(),code->size(),nullptr,&combined_ps_);
        if(FAILED(hr)){error="Pending colour shader creation failed";return false;}
      }
      ComPtr<ID3D11Resource> r;if(warm.source_band)warm.source_band->GetResource(&r);
      ComPtr<ID3D11Texture2D> t;if(!r || FAILED(r.As(&t))){error="Warm source band unavailable";return false;}
      D3D11_TEXTURE2D_DESC desc{};t->GetDesc(&desc);band_height_=desc.Height;
      if(desc.Width!=warm.input_region.width() || !band_height_ || band_height_>warm.input_region.height()) {error="Warm source crop mismatch";return false;}
      retirement::shader_set_t set;for(UINT i=0;i<8u;++i)set.shaders[i]=retirement_shaders_[i].Get();
      if(!snapshot_.ensure(device,desc.Width,desc.Height,error) || !retirement_.prepare(device,context,warm,set,error))return false;
      region_=warm.input_region;content_=warm.field_content;field_width_=warm.field_width;field_height_=warm.field_height;
      // Defined UNREADY storage before first C; archive0 source was initially zeroed.
      context->CopyResource(snapshot_.working_views().texture.Get(),retirement_.export_resources(0u)[0].Get());
      snapshot_t zero_current{};context->UpdateSubresource(snapshot_.working_constants_buffer().Get(),0,nullptr,&zero_current,0,0);
      domain_=warm.publication.domain_tag;epoch_=capture_epoch;permission_=permission;prepared_=true;return true;
    }
    snapshot_t context(const material_identity_t &id,input_color_space transfer) const noexcept {
      snapshot_t c;if(!prepared_ || !id.usable() || id.epoch!=epoch_)return c;
      c.control={1u,research_current_source_snapshot_v1::tag,1u,static_cast<std::uint32_t>(transfer)};
      c.source_rect={region_.left,region_.top,region_.width(),region_.height()};
      c.band_field={region_.height()-band_height_,band_height_,field_width_,field_height_};
      c.content={content_.left,content_.top,content_.right,content_.bottom};
      const auto serial=split(id.material_serial),clock=split(id.material_capture_clock_us),epoch=split(epoch_),gen=split(region_.analysis_generation),permission=split(permission_),domain=split(domain_);
      c.current={serial[0],serial[1],clock[0],clock[1]};c.scope={epoch[0],epoch[1],gen[0],gen[1]};c.domain={permission[0],permission[1],domain[0],domain[1]};return c;
    }
    bool capture_pending(ID3D11DeviceContext *c,const pending_ocr_conditioner_research_views_t &owner,
      const queued_same_source_gpu_continuation_research_views_t &ticket,const material_identity_t &id,std::string &error) {
      if(!prepared_ || !id.usable() || owner.ticket_sequence!=ticket.ticket_sequence ||
         owner.input_region!=region_ || owner.field_content!=content_ || owner.field_width!=field_width_ || owner.field_height!=field_height_ ||
         owner.submitted_frame_id!=ticket.submitted_frame_id || owner.transaction_token!=ticket.transaction_token ||
         owner.planned_owner!=ticket.planned_publication || owner.color_space!=ticket.color_space) {error="Pending colour owner lease mismatch";return false;}
      const auto e=research_source_pattern_retirement_v1::make_expected(context(id,owner.color_space),owner.planned_owner);
      const auto bank=retirement_.capture_pending_owner(c,owner,e,error);if(!bank)return false;
      active_bank_=bank;ticket_=ticket;return true;
    }
    // Caller proves the exact SAME pending owner/slot still leased and the actual CURRENT
    // capture is keyed-locked through snapshot and draw submission. No current model wait.
    bool observe_current(ID3D11DeviceContext *c,ID3D11ShaderResourceView *current,
      const material_identity_t &id,std::string &error) {
      if(!active_bank_ || !ticket_ || !id.usable()){error="No pending colour owner/current identity";return false;}
      last_current_=context(id,ticket_->color_space);
      std::array<ID3D11ComputeShader *,3> shaders{};for(UINT i=0;i<3u;++i)shaders[i]=snapshot_shaders_[i].Get();
      if(!snapshot_.dispatch(c,shaders,current,last_current_,error))return false;
      const auto views=snapshot_.working_views();
      const auto e=research_source_pattern_retirement_v1::make_expected(last_current_,ticket_->planned_publication);
      return retirement_.observe(*active_bank_,c,views.source.Get(),views.receipt.Get(),snapshot_.working_constants_buffer().Get(),e,error);
    }
    // Release geometry alias use on ACK/new admission/error. This never clears either bank,
    // retirement latch, ticket high-watermark, or global CURRENT source watermark.
    void release_geometry_lease() noexcept {ticket_.reset();active_bank_.reset();}
    const auto &ticket() const noexcept {return ticket_;}
    bool prepared() const noexcept {return prepared_;}
    ID3D11PixelShader *pixel_shader() const noexcept {return combined_ps_.Get();}
    ID3D11Buffer *negative_expected() const noexcept {return active_bank_?retirement_.expected_buffer(*active_bank_):nullptr;}
    ID3D11ShaderResourceView *negative_receipt() const noexcept {return active_bank_?retirement_.negative_receipt(*active_bank_):nullptr;}
    auto current_snapshot() const {return snapshot_.working_views();}
    auto current_constants() const {return snapshot_.working_constants_buffer();}
    const snapshot_t &last_current_context() const noexcept {return last_current_;}
    auto owner_resources(UINT bank) const {return retirement_.export_resources(bank);}
    std::optional<UINT> active_bank() const noexcept {return active_bank_;}
  };
} // namespace models::research_pending_current_colour_v1
#endif
