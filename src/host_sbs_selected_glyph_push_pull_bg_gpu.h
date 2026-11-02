#pragma once
// PRIVATE ONE push/pull+erase1 prototype. No runtime activation or new protection mask.
#ifdef _WIN32
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <d3d11.h>
#include <wrl/client.h>
namespace models::selected_glyph_push_pull_bg_gpu_v1 {
using Microsoft::WRL::ComPtr;
inline constexpr UINT proof_words=96u,expected_words=16u,maximum_levels=16u;
struct program_t { ID3D11ComputeShader *prepare=nullptr,*init=nullptr,*down=nullptr,*coarse=nullptr,*pull=nullptr,*seal=nullptr; };
struct request_t {
 std::array<ID3D11ShaderResourceView*,8u> draw_srvs{}; // literal PS t0..7
 ID3D11Buffer *actual_contour_constants96=nullptr;
 ID3D11Buffer *actual_subtitle_render_constants32=nullptr;
 std::array<UINT,4u> analysis_shape{}; // W,fullH,cropTop,bandH; independently GPU joined
};
struct views_t {
 ComPtr<ID3D11ShaderResourceView> background,proof;
 ComPtr<ID3D11Buffer> expected_constants;
 std::uint64_t draw_serial=0;
};
class resources_t {
 ComPtr<ID3D11Device> device_;
 ComPtr<ID3D11Texture2D> background_;
 ComPtr<ID3D11ShaderResourceView> background_srv_,proof_srv_;
 ComPtr<ID3D11UnorderedAccessView> background_uav_,pool_uav_,proof_uav_,args_uav_;
 ComPtr<ID3D11Buffer> pool_,proof_,args_,constants_,expected_;
 std::array<ComPtr<ID3D11ShaderResourceView>,8u> lease_inputs_;
 ComPtr<ID3D11Buffer> lease_cb_,lease_b3_;
 std::array<std::array<UINT,4u>,16u> mips_{};
 UINT width_=0,band_height_=0,levels_=0,pool_cells_=0;
 std::uint64_t serial_=0;
 bool issued_=false;
 struct constants_t { std::array<UINT,16u> lease{}; std::array<UINT,4u> shape{},pool{},level{}; std::array<std::array<UINT,4u>,16u> mips{}; };
 static_assert(sizeof(constants_t)==368u);
    struct saved_t {
      ID3D11DeviceContext *context;
      std::array<ComPtr<ID3D11ShaderResourceView>, 16u> srvs;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 8u> uavs;
      std::array<ComPtr<ID3D11Buffer>, 3u> cbs;
      ComPtr<ID3D11ComputeShader> shader;
      std::array<ComPtr<ID3D11ShaderResourceView>, 3u> ps_background;
      std::array<ComPtr<ID3D11ClassInstance>, 256u> instances;
      UINT instance_count = 256u;

      explicit saved_t(ID3D11DeviceContext *c):
          context(c) {
        ID3D11ShaderResourceView *p[3] {};
        c->PSGetShaderResources(9u, 3u, p);
        for (UINT i=0;i<3u;++i) ps_background[i].Attach(p[i]);
        ID3D11ShaderResourceView *s[16] {};
        ID3D11UnorderedAccessView *u[8] {};
        ID3D11Buffer *b[3] {};
        ID3D11ComputeShader *sh = nullptr;
        ID3D11ClassInstance *ci[256] {};
        c->CSGetShaderResources(0u, 16u, s);
        c->CSGetUnorderedAccessViews(0u, 8u, u);
        c->CSGetConstantBuffers(0u, 3u, b);
        c->CSGetShader(&sh, ci, &instance_count);
        for (UINT i = 0u; i < 16u; ++i) {
          srvs[i].Attach(s[i]);
        }
        for (UINT i = 0u; i < 8u; ++i) {
          uavs[i].Attach(u[i]);
        }
        for (UINT i = 0u; i < 3u; ++i) {
          cbs[i].Attach(b[i]);
        }
        for (UINT i = 0u; i < instance_count; ++i) {
          instances[i].Attach(ci[i]);
        }
        shader.Attach(sh);
      }

      static void unbind(ID3D11DeviceContext *c) {
        ID3D11ShaderResourceView *p[3] {};
        c->PSSetShaderResources(9u,3u,p);
        ID3D11ShaderResourceView *s[16] {};
        ID3D11UnorderedAccessView *u[8] {};
        ID3D11Buffer *b[3] {};
        c->CSSetShaderResources(0u, 16u, s);
        c->CSSetUnorderedAccessViews(0u, 8u, u, nullptr);
        c->CSSetConstantBuffers(0u, 3u, b);
      }

      ~saved_t() {
        unbind(context);
        ID3D11ShaderResourceView *s[16] {};
        ID3D11UnorderedAccessView *u[8] {};
        ID3D11Buffer *b[3] {};
        ID3D11ClassInstance *ci[256] {};
        for (UINT i = 0u; i < 16u; ++i) {
          s[i] = srvs[i].Get();
        }
        for (UINT i = 0u; i < 8u; ++i) {
          u[i] = uavs[i].Get();
        }
        for (UINT i = 0u; i < 3u; ++i) {
          b[i] = cbs[i].Get();
        }
        for (UINT i = 0u; i < instance_count; ++i) {
          ci[i] = instances[i].Get();
        }
        context->CSSetShaderResources(0u, 16u, s);
        context->CSSetUnorderedAccessViews(0u, 8u, u, nullptr);
        context->CSSetConstantBuffers(0u, 3u, b);
        context->CSSetShader(shader.Get(), ci, instance_count);
        ID3D11ShaderResourceView *p[3] {};
        for(UINT i=0;i<3u;++i)p[i]=ps_background[i].Get();
        context->PSSetShaderResources(9u,3u,p);
      }
    };


 bool same_device(ID3D11DeviceChild *c)const { if(!c||!device_)return false;ComPtr<ID3D11Device>d;c->GetDevice(&d);return d.Get()==device_.Get(); }
 bool context_ok(ID3D11DeviceContext*c)const { if(!same_device(c)||c->GetType()!=D3D11_DEVICE_CONTEXT_IMMEDIATE)return false;ComPtr<ID3D11Predicate>p;BOOL v=FALSE;c->GetPredication(&p,&v);return !p; }
 bool constant(ID3D11Buffer*b,UINT bytes)const {if(!same_device(b))return false;D3D11_BUFFER_DESC d{};b->GetDesc(&d);return d.ByteWidth==bytes&&(d.BindFlags&D3D11_BIND_CONSTANT_BUFFER)!=0u;}
 bool structured(ID3D11ShaderResourceView*v,UINT elements,UINT stride,bool at_least=false)const {
  if(!same_device(v))return false;D3D11_SHADER_RESOURCE_VIEW_DESC s{};v->GetDesc(&s);ComPtr<ID3D11Resource>r;v->GetResource(&r);ComPtr<ID3D11Buffer>b;if(FAILED(r.As(&b)))return false;D3D11_BUFFER_DESC d{};b->GetDesc(&d);
  return s.ViewDimension==D3D11_SRV_DIMENSION_BUFFER&&s.Format==DXGI_FORMAT_UNKNOWN&&s.Buffer.FirstElement==0u&&(at_least?s.Buffer.NumElements>=elements:s.Buffer.NumElements==elements)&&d.StructureByteStride==stride&&d.ByteWidth==s.Buffer.NumElements*stride&&(d.MiscFlags&D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)!=0u;
 }
 bool texture(ID3D11ShaderResourceView*v,D3D11_TEXTURE2D_DESC&d,D3D11_SHADER_RESOURCE_VIEW_DESC&s,ComPtr<ID3D11Resource>&r)const {
  if(!same_device(v))return false;v->GetDesc(&s);v->GetResource(&r);ComPtr<ID3D11Texture2D>t;if(FAILED(r.As(&t)))return false;t->GetDesc(&d);
  return s.ViewDimension==D3D11_SRV_DIMENSION_TEXTURE2D&&s.Texture2D.MostDetailedMip==0u&&(s.Texture2D.MipLevels==1u||s.Texture2D.MipLevels==~0u)&&d.MipLevels==1u&&d.ArraySize==1u&&d.SampleDesc.Count==1u&&d.SampleDesc.Quality==0u&&(d.BindFlags&D3D11_BIND_SHADER_RESOURCE)!=0u;
 }
 bool make_buffer(UINT bytes,UINT stride,UINT bind,UINT misc,ComPtr<ID3D11Buffer>&b) {
  D3D11_BUFFER_DESC d{};d.ByteWidth=bytes;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=bind;d.MiscFlags=misc;d.StructureByteStride=stride;return SUCCEEDED(device_->CreateBuffer(&d,nullptr,b.GetAddressOf()));
 }
 bool check(const request_t&r,std::array<UINT,16u>&lease,ID3D11DeviceContext*c,std::string&error) {
  if(!constant(r.actual_contour_constants96,96u)||!constant(r.actual_subtitle_render_constants32,32u)||r.analysis_shape[0]!=width_||r.analysis_shape[3]!=band_height_||r.analysis_shape[1]==0u||r.analysis_shape[1]>5120u||r.analysis_shape[2]>=r.analysis_shape[1]||r.analysis_shape[3]!=r.analysis_shape[1]-r.analysis_shape[2]) {error="BG actual CB or shape unavailable";return false;}
  std::array<ComPtr<ID3D11Resource>,8u> actual;
  for(UINT i=0;i<8u;++i) {if(!same_device(r.draw_srvs[i])){error="BG draw input absent/foreign";return false;}r.draw_srvs[i]->GetResource(&actual[i]);if(actual[i].Get()==background_.Get()||actual[i].Get()==pool_.Get()||actual[i].Get()==proof_.Get()||actual[i].Get()==args_.Get()||actual[i].Get()==constants_.Get()||actual[i].Get()==expected_.Get()){error="BG writable input alias";return false;}}
  if(!structured(r.draw_srvs[2],4u,16u,true)||!structured(r.draw_srvs[5],6u,4u)||!structured(r.draw_srvs[6],32u,4u)||!structured(r.draw_srvs[7],64u,4u)){error="BG typed structured inputs unavailable";return false;}
  for(UINT i:{0u,1u,3u,4u}) {D3D11_TEXTURE2D_DESC d{};D3D11_SHADER_RESOURCE_VIEW_DESC s{};ComPtr<ID3D11Resource>a;if(!texture(r.draw_srvs[i],d,s,a)){error="BG exact texture input unavailable";return false;}
   if(i==0u) {if(s.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT&&s.Format!=DXGI_FORMAT_R8G8B8A8_UNORM&&s.Format!=DXGI_FORMAT_B8G8R8A8_UNORM&&s.Format!=DXGI_FORMAT_B8G8R8X8_UNORM){error="BG native working source format unsupported";return false;}}
   if(i==1u&&s.Format!=DXGI_FORMAT_R32_FLOAT){error="BG Base format unavailable";return false;}
   if(i==3u&&(s.Format!=DXGI_FORMAT_R32_UINT||d.Width!=width_||d.Height!=band_height_)){error="BG selectedM shape/format unavailable";return false;}
   if(i==4u&&(s.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT||d.Width!=width_||d.Height!=band_height_)){error="BG actual display band unavailable";return false;}
  }
  if(serial_==std::numeric_limits<std::uint64_t>::max()){error="BG draw serial exhausted";return false;}++serial_;
  auto put=[&](UINT at,std::uint64_t v){lease[at]=static_cast<UINT>(v);lease[at+1u]=static_cast<UINT>(v>>32u);};
  put(0u,serial_);put(2u,reinterpret_cast<std::uintptr_t>(r.draw_srvs[0]));put(4u,reinterpret_cast<std::uintptr_t>(actual[0].Get()));put(6u,reinterpret_cast<std::uintptr_t>(r.draw_srvs[3]));put(8u,reinterpret_cast<std::uintptr_t>(r.draw_srvs[7]));put(10u,reinterpret_cast<std::uintptr_t>(r.draw_srvs[4]));put(12u,reinterpret_cast<std::uintptr_t>(r.draw_srvs[6]));put(14u,reinterpret_cast<std::uintptr_t>(c));return true;
 }
public:
 bool ensure(ID3D11Device*device,UINT width,UINT band_height,std::string&error) {
  error.clear();if(!device||width==0u||band_height==0u||width>5120u||band_height>5120u){error="BG native domain unavailable";return false;}
  if(device_.Get()==device&&width_==width&&band_height_==band_height&&pool_)return true;
  resources_t n;n.device_=device;n.width_=width;n.band_height_=band_height;n.serial_=serial_;
  UINT w=1u,h=1u;while(w<width)w*=2u;while(h<band_height)h*=2u;
  do {if(n.levels_>=16u){error="BG mip count unsupported";return false;}n.mips_[n.levels_++]={w,h,n.pool_cells_,0u};n.pool_cells_+=w*h;if(w==1u&&h==1u)break;w=w>1u?w/2u:1u;h=h>1u?h/2u:1u;}while(true);
  if(static_cast<std::uint64_t>(n.pool_cells_)*16u>std::numeric_limits<UINT>::max()){error="BG pool allocation size unavailable";return false;}
  if(!n.make_buffer(n.pool_cells_*16u,16u,D3D11_BIND_UNORDERED_ACCESS,D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,n.pool_)||FAILED(device->CreateUnorderedAccessView(n.pool_.Get(),nullptr,n.pool_uav_.GetAddressOf()))||!n.make_buffer(96u*4u,4u,D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS,D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,n.proof_)||FAILED(device->CreateShaderResourceView(n.proof_.Get(),nullptr,n.proof_srv_.GetAddressOf()))||FAILED(device->CreateUnorderedAccessView(n.proof_.Get(),nullptr,n.proof_uav_.GetAddressOf()))||!n.make_buffer(32u*12u,0u,D3D11_BIND_UNORDERED_ACCESS,D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS|D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS,n.args_)||!n.make_buffer(sizeof(constants_t),0u,D3D11_BIND_CONSTANT_BUFFER,0u,n.constants_)||!n.make_buffer(64u,0u,D3D11_BIND_CONSTANT_BUFFER,0u,n.expected_)){error="BG buffer allocation failed";return false;}
  D3D11_UNORDERED_ACCESS_VIEW_DESC a{};a.Format=DXGI_FORMAT_R32_TYPELESS;a.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;a.Buffer.NumElements=96u;a.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
  if(FAILED(device->CreateUnorderedAccessView(n.args_.Get(),&a,n.args_uav_.GetAddressOf()))){error="BG indirect arguments unavailable";return false;}
  D3D11_TEXTURE2D_DESC t{};t.Width=width;t.Height=band_height;t.MipLevels=t.ArraySize=t.SampleDesc.Count=1u;t.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;t.Usage=D3D11_USAGE_DEFAULT;t.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
  if(FAILED(device->CreateTexture2D(&t,nullptr,n.background_.GetAddressOf()))||FAILED(device->CreateShaderResourceView(n.background_.Get(),nullptr,n.background_srv_.GetAddressOf()))||FAILED(device->CreateUnorderedAccessView(n.background_.Get(),nullptr,n.background_uav_.GetAddressOf()))){error="BG texture allocation failed";return false;}
  *this=std::move(n);return true;
 }
 // Serialized immediate-context owner; true means commands submitted, never GPU readiness.
 // Keep this owner and all actual inputs immutable until its draw is issued before next queue.
 // BG/proof may be bound only at reserved PS9/11 or saved CS slots, never elsewhere.
 bool queue(ID3D11DeviceContext*c,const program_t&p,const request_t&r,std::string&error) {
  issued_=false;
  error.clear();if(!pool_||!context_ok(c)||!same_device(p.prepare)||!same_device(p.init)||!same_device(p.down)||!same_device(p.coarse)||!same_device(p.pull)||!same_device(p.seal)){error="BG program/context unavailable";return false;}
  constants_t cb{};if(!check(r,cb.lease,c,error))return false;cb.shape=r.analysis_shape;cb.pool={mips_[0][0],mips_[0][1],pool_cells_,levels_};cb.mips=mips_;
  saved_t saved(c);saved_t::unbind(c);
  c->UpdateSubresource(expected_.Get(),0u,nullptr,cb.lease.data(),0u,0u);
  auto bind_cb=[&](){c->UpdateSubresource(constants_.Get(),0u,nullptr,&cb,0u,0u);ID3D11Buffer*b[3]={r.actual_contour_constants96,constants_.Get(),r.actual_subtitle_render_constants32};c->CSSetConstantBuffers(0u,3u,b);};
  bind_cb();c->CSSetShaderResources(0u,8u,r.draw_srvs.data());ID3D11UnorderedAccessView*u[4]={pool_uav_.Get(),background_uav_.Get(),proof_uav_.Get(),args_uav_.Get()};c->CSSetUnorderedAccessViews(0u,4u,u,nullptr);
  c->CSSetShader(p.prepare,nullptr,0u);c->Dispatch(1u,1u,1u);
  ID3D11UnorderedAccessView*nullu=nullptr;c->CSSetUnorderedAccessViews(3u,1u,&nullu,nullptr);
  c->CSSetShader(p.init,nullptr,0u);c->DispatchIndirect(args_.Get(),0u);
  for(UINT level=1u;level<levels_;++level){cb.level={level-1u,level,0u,0u};bind_cb();c->CSSetShader(p.down,nullptr,0u);c->DispatchIndirect(args_.Get(),level*12u);}
  c->CSSetShader(p.coarse,nullptr,0u);c->DispatchIndirect(args_.Get(),levels_*12u);
  for(UINT step=0u;step+1u<levels_;++step){const UINT fine=levels_-2u-step;cb.level={fine+1u,fine,0u,0u};bind_cb();c->CSSetShader(p.pull,nullptr,0u);c->DispatchIndirect(args_.Get(),(levels_+1u+step)*12u);}
  c->CSSetShader(p.seal,nullptr,0u);c->Dispatch(1u,1u,1u);
  for(UINT i=0;i<8u;++i)lease_inputs_[i]=r.draw_srvs[i];lease_cb_=r.actual_contour_constants96;lease_b3_=r.actual_subtitle_render_constants32;issued_=true;return true;
 }
 views_t views()const{if(!issued_)return {};return {background_srv_,proof_srv_,expected_,serial_};}
};
} // namespace models::selected_glyph_push_pull_bg_gpu_v1
#endif
