#pragma once
#ifdef _WIN32
#include "host_sbs_selected_glyph_push_pull_bg_gpu.h"
#include "host_sbs_selected_glyph_receipt_c24_transport.h"
#include "host_sbs_current_draw_colour_lease.h"
#include "host_sbs_shader_cache.h"
#include <array>
#include <filesystem>
#include <exception>
#include <string>
#include <string_view>
namespace models::selected_glyph_push_pull_runtime_v1 {
namespace bg=selected_glyph_push_pull_bg_gpu_v1;
namespace cache=host_sbs_shader_cache;
inline constexpr std::array<cache::shader_spec,6u> specs{{
 {"host_sbs_selected_glyph_push_pull_bg_cs.hlsl","prepare_main","cs_5_0"},
 {"host_sbs_selected_glyph_push_pull_bg_cs.hlsl","init_main","cs_5_0"},
 {"host_sbs_selected_glyph_push_pull_bg_cs.hlsl","down_main","cs_5_0"},
 {"host_sbs_selected_glyph_push_pull_bg_cs.hlsl","coarse_main","cs_5_0"},
 {"host_sbs_selected_glyph_push_pull_bg_cs.hlsl","pull_main","cs_5_0"},
 {"host_sbs_selected_glyph_push_pull_bg_cs.hlsl","seal_main","cs_5_0"},
}};
inline constexpr std::string_view closure="80a521fb1e77c373d1f297bbfa4957984d8db5529923589a6573e628daeac073";
class programs_t {
 std::array<Microsoft::WRL::ComPtr<ID3D11ComputeShader>,6u> shaders_;
public:
 bool load(ID3D11Device*d,const std::filesystem::path&root,std::string&error){
  error.clear();try { const auto source=cache::snapshot_sources(root,specs);
  if(!d||!source||cache::source_closure_sha256(source)!=closure){error="Current background shader source mismatch";return false;}
  programs_t next;
  for(std::size_t i=0;i<specs.size();++i){const auto code=cache::get(source,specs[i]);
   if(!code||code->empty()||FAILED(d->CreateComputeShader(code->data(),code->size(),nullptr,next.shaders_[i].GetAddressOf()))){error="Current background shader creation failed";return false;}}
  *this=std::move(next);return true;
  }catch(const std::exception&e){error=e.what();return false;}
 }
 bool ready()const noexcept {for(const auto&s:shaders_)if(!s)return false;return true;}
 bg::program_t program()const noexcept{return {shaders_[0].Get(),shaders_[1].Get(),shaders_[2].Get(),shaders_[3].Get(),shaders_[4].Get(),shaders_[5].Get()};}
};
// Optional BG scope only. Original selected t7/t8 and queued t10/b5 remain caller-owned.
class scoped_background_t {
 ID3D11DeviceContext*context_=nullptr;
 Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> old9_,old11_;
 Microsoft::WRL::ComPtr<ID3D11Buffer> old4_;
public:
 bool bind(ID3D11DeviceContext*c,const bg::views_t*views) {
  if(!c||context_)return false;
  c->PSGetShaderResources(9u,1u,old9_.GetAddressOf());c->PSGetShaderResources(11u,1u,old11_.GetAddressOf());c->PSGetConstantBuffers(4u,1u,old4_.GetAddressOf());context_=c;
  auto*b=views?views->background.Get():nullptr;auto*p=views?views->proof.Get():nullptr;auto*k=views?views->expected_constants.Get():nullptr;
  c->PSSetShaderResources(9u,1u,&b);c->PSSetShaderResources(11u,1u,&p);c->PSSetConstantBuffers(4u,1u,&k);return true;
 }
 ~scoped_background_t(){if(context_){auto*b=old9_.Get();auto*p=old11_.Get();auto*k=old4_.Get();context_->PSSetShaderResources(9u,1u,&b);context_->PSSetShaderResources(11u,1u,&p);context_->PSSetConstantBuffers(4u,1u,&k);}}
};
} //namespace
#endif
