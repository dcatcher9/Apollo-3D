#pragma once
// PRIVATE CPU role check only. It creates no pixel/clock proof and grants no subtitle presence.
#ifdef _WIN32
#include <array>
#include <cstdint>
#include <d3d11.h>
#include <wrl/client.h>
namespace models::push_pull_current_draw_lease_v1 {
struct lease_t {
 using key_t=std::array<std::uint64_t,4u>;
 Microsoft::WRL::ComPtr<ID3D11DeviceContext>context;
 std::array<Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>,8u>srvs;
 Microsoft::WRL::ComPtr<ID3D11Buffer>contour,subtitle;
 std::array<UINT,4u>shape{};key_t colour_key{};std::uint64_t serial=0;
 template<class Request>void capture(ID3D11DeviceContext*c,const Request&r,key_t key,std::uint64_t value){
  context=c;for(UINT i=0;i<8u;++i)srvs[i]=r.draw_srvs[i];contour=r.actual_contour_constants96;subtitle=r.actual_subtitle_render_constants32;shape=r.analysis_shape;colour_key=key;serial=value;
 }
 template<class Request>bool matches(ID3D11DeviceContext*c,const Request&r,key_t key,std::uint64_t current_serial)const noexcept{
  if(!serial||current_serial!=serial||context.Get()!=c||colour_key!=key||shape!=r.analysis_shape||contour.Get()!=r.actual_contour_constants96||subtitle.Get()!=r.actual_subtitle_render_constants32)return false;
  for(UINT i=0;i<8u;++i)if(srvs[i].Get()!=r.draw_srvs[i])return false;
  return true;
 }
 // Caller serializes queue -> matches/bind/draw before next queue, and keeps the raw source
 // contents immutable for that colour_key. Reusing a texture for a new frame requires a new key.
};
}
#endif
