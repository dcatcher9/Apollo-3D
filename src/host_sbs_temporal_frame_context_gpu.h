#pragma once
#ifdef _WIN32
#include "host_sbs_body_witness_temporal_gpu.h"
#include "host_sbs_temporal_source_ledger.h"
#include <array>
#include <cstdint>
#include <d3d11.h>
#include <string>
#include <wrl/client.h>
namespace models::host_sbs_temporal_frame_context_gpu {
using Microsoft::WRL::ComPtr;
using frame_t=host_sbs_body_witness_temporal_gpu::frame_constants_t;
// Exact existing postprocess request classes, sealed by the caller from actual pending state.
// This is NOT inferred from argument bytes or a CPU completion fabricated by the serializer.
enum class postprocess_mode_e : std::uint32_t { authorized=1u,cpu_completed_force=2u,queued_force=4u };
// Literal upload seed, distinct from the GPU output Frame256 ABI. Cut/model context is GPU-owned.
inline frame_t seed(const host_sbs_temporal_source_bridge::certificate_t &c,
 UINT width,UINT crop_height,host_sbs_body_witness_temporal_gpu::caps_t caps,
 std::uint64_t model_frame,std::uint64_t model_observation,std::uint64_t transaction_token,postprocess_mode_e mode) {
 frame_t x;auto &w=x.words;
 w[0]=width;w[1]=crop_height;w[2]=width*crop_height;
 auto put64=[&](UINT i,std::uint64_t v){w[i]=static_cast<UINT>(v);w[i+1]=static_cast<UINT>(v>>32u);};
 put64(4,c.material_serial);put64(6,c.capture_clock_us);
 w[8]=c.valid()?1u:0u;w[9]=c.predecessor_contiguous?1u:0u;w[10]=1u;w[11]=static_cast<std::uint32_t>(mode);
 w[12]=caps.components;w[13]=caps.members;w[14]=caps.bit_words;w[15]=caps.hash_words;
 put64(16,c.source_epoch);put64(18,c.domain.context_generation);put64(20,c.domain.analysis_generation);
 w[22]=c.domain.analysis_ltrb[2]-c.domain.analysis_ltrb[0];w[23]=c.domain.analysis_ltrb[3]-c.domain.analysis_ltrb[1];
 put64(24,transaction_token);w[26]=1u;w[27]=1u^0x6f435257u;put64(28,model_frame);put64(30,model_observation);
 return x;
}
class resources_t {
 ComPtr<ID3D11Device> device_;
 ComPtr<ID3D11Buffer> seed_,output_;
 ComPtr<ID3D11UnorderedAccessView> output_uav_;
 ComPtr<ID3D11ShaderResourceView> output_srv_;
 bool child(ID3D11DeviceChild *x) const {if(!x)return false;ComPtr<ID3D11Device>d;x->GetDevice(&d);return d.Get()==device_.Get();}
 struct saved_t {
  ID3D11DeviceContext *c;
  std::array<ComPtr<ID3D11ShaderResourceView>,2> s;
  ComPtr<ID3D11UnorderedAccessView> u;
  std::array<ComPtr<ID3D11Buffer>,2> b;
  ComPtr<ID3D11ComputeShader> shader;
  std::array<ComPtr<ID3D11ClassInstance>,256> classes;UINT count=256;
  explicit saved_t(ID3D11DeviceContext *x):c(x){ID3D11ShaderResourceView*sv[2]{};ID3D11Buffer*cb[2]{};ID3D11UnorderedAccessView*uv=nullptr;ID3D11ComputeShader*cs=nullptr;ID3D11ClassInstance*ci[256]{};
   c->CSGetShaderResources(0,2,sv);c->CSGetUnorderedAccessViews(0,1,&uv);c->CSGetConstantBuffers(0,2,cb);c->CSGetShader(&cs,ci,&count);
   for(UINT i=0;i<2;++i){s[i].Attach(sv[i]);b[i].Attach(cb[i]);}u.Attach(uv);shader.Attach(cs);for(UINT i=0;i<count;++i)classes[i].Attach(ci[i]);}
  static void unbind(ID3D11DeviceContext*c){ID3D11ShaderResourceView*s[2]{};ID3D11UnorderedAccessView*u=nullptr;c->CSSetShaderResources(0,2,s);c->CSSetUnorderedAccessViews(0,1,&u,nullptr);}
  ~saved_t(){unbind(c);ID3D11ShaderResourceView*sv[2]{s[0].Get(),s[1].Get()};ID3D11Buffer*cb[2]{b[0].Get(),b[1].Get()};auto uv=u.Get();ID3D11ClassInstance*ci[256]{};for(UINT i=0;i<count;++i)ci[i]=classes[i].Get();c->CSSetUnorderedAccessViews(0,1,&uv,nullptr);c->CSSetShaderResources(0,2,sv);c->CSSetConstantBuffers(0,2,cb);c->CSSetShader(shader.Get(),count?ci:nullptr,count);}
 };
public:
 bool ensure(ID3D11Device *d,std::string &error){
  if(!d){error="Missing Frame256 device";return false;}if(device_.Get()==d&&seed_&&output_)return true;
  resources_t next;next.device_=d;D3D11_BUFFER_DESC desc{};desc.ByteWidth=256;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
  if(FAILED(d->CreateBuffer(&desc,nullptr,&next.seed_))){error="Frame256 seed allocation";return false;}
  desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS|D3D11_BIND_SHADER_RESOURCE;desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;desc.StructureByteStride=4;
  D3D11_UNORDERED_ACCESS_VIEW_DESC u{};u.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;u.Buffer.NumElements=64;
  D3D11_SHADER_RESOURCE_VIEW_DESC s{};s.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;s.Buffer.NumElements=64;
  if(FAILED(d->CreateBuffer(&desc,nullptr,&next.output_))||FAILED(d->CreateUnorderedAccessView(next.output_.Get(),&u,&next.output_uav_))||FAILED(d->CreateShaderResourceView(next.output_.Get(),&s,&next.output_srv_))){error="Frame256 GPU output allocation";return false;}
  *this=std::move(next);return true;
 }
 // True is queue success only. The shader may deliberately publish all-zero unavailable Frame256.
 bool queue(ID3D11DeviceContext*c,ID3D11ComputeShader*shader,ID3D11Buffer*actual_cb96,
  ID3D11ShaderResourceView*uip,ID3D11ShaderResourceView*root,const frame_t&seed,std::string&error){
  error.clear();if(!c||c->GetType()!=D3D11_DEVICE_CONTEXT_IMMEDIATE||!child(shader)||!child(actual_cb96)||!child(uip)||!child(root)||!output_){error="Frame256 operands/context unavailable";return false;}
  ComPtr<ID3D11Device>d;c->GetDevice(&d);if(d.Get()!=device_.Get()){error="Frame256 context device mismatch";return false;}
  ComPtr<ID3D11Predicate>pred;BOOL predval=FALSE;c->GetPredication(&pred,&predval);if(pred){error="Frame256 active predication unavailable";return false;}
  D3D11_BUFFER_DESC desc{};actual_cb96->GetDesc(&desc);if(desc.ByteWidth!=96||(desc.BindFlags&D3D11_BIND_CONSTANT_BUFFER)==0){error="Frame256 actual CB descriptor";return false;}
  const saved_t saved(c);saved_t::unbind(c);c->UpdateSubresource(seed_.Get(),0,nullptr,&seed,0,0);
  ID3D11Buffer*b[2]{seed_.Get(),actual_cb96};ID3D11ShaderResourceView*s[2]{uip,root};auto u=output_uav_.Get();
  c->CSSetConstantBuffers(0,2,b);c->CSSetShaderResources(0,2,s);c->CSSetUnorderedAccessViews(0,1,&u,nullptr);c->CSSetShader(shader,nullptr,0);c->Dispatch(1,1,1);return true;
 }
 ComPtr<ID3D11Buffer> output() const{return output_;}
 ComPtr<ID3D11ShaderResourceView> output_view()const{return output_srv_;}
};
}
#endif
