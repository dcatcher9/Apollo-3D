#pragma once
// GPU byte transport only. This does not prove UI presence, freshness, completion, or opacity.
// Receipt C24 remains joined independently by the unchanged BG owner to SCT/state/b3/RGB.
#ifdef _WIN32
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <d3d11.h>
#include <wrl/client.h>
namespace models::selected_glyph_receipt_c24_transport_v1 {
class resources_t {
 using MicrosoftComDevice=Microsoft::WRL::ComPtr<ID3D11Device>;
 MicrosoftComDevice device_;
 Microsoft::WRL::ComPtr<ID3D11Buffer>constants_;
 Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>source_;
 Microsoft::WRL::ComPtr<ID3D11Buffer>source_buffer_;
 Microsoft::WRL::ComPtr<ID3D11DeviceContext>context_;
 std::uint64_t serial_=0;bool issued_=false;
 bool same_device(ID3D11DeviceChild*p)const noexcept{
  if(!p||!device_)return false;
  MicrosoftComDevice d;
  p->GetDevice(&d);
  return d.Get()==device_.Get();
 }
public:
 bool ensure(ID3D11Device*device,std::string&error){
  issued_=false;error.clear();if(!device){error="Receipt C24 device absent";return false;}
  if(device_.Get()==device&&constants_)return true;
  D3D11_BUFFER_DESC d{};d.ByteWidth=96u;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
  Microsoft::WRL::ComPtr<ID3D11Buffer>next;
  if(FAILED(device->CreateBuffer(&d,nullptr,&next))){error="Receipt C24 owned constant allocation failed";return false;}
  device_=device;constants_=std::move(next);source_.Reset();source_buffer_.Reset();context_.Reset();return true;
 }
 bool queue(ID3D11DeviceContext*context,ID3D11ShaderResourceView*selected_receipt,std::string&error){
  issued_=false;error.clear();
  if(!constants_||!same_device(context)||context->GetType()!=D3D11_DEVICE_CONTEXT_IMMEDIATE||!same_device(selected_receipt)){
   error="Receipt C24 same-device immediate source unavailable";return false;
  }
  Microsoft::WRL::ComPtr<ID3D11Predicate>predicate;BOOL value=FALSE;context->GetPredication(&predicate,&value);
  if(predicate){error="Receipt C24 predicated context unavailable";return false;}
  D3D11_SHADER_RESOURCE_VIEW_DESC s{};selected_receipt->GetDesc(&s);Microsoft::WRL::ComPtr<ID3D11Resource>resource;selected_receipt->GetResource(&resource);
  Microsoft::WRL::ComPtr<ID3D11Buffer>buffer;if(FAILED(resource.As(&buffer))){error="Receipt C24 source is not buffer";return false;}
  D3D11_BUFFER_DESC b{};buffer->GetDesc(&b);
  if(s.ViewDimension!=D3D11_SRV_DIMENSION_BUFFER||s.Format!=DXGI_FORMAT_UNKNOWN||s.Buffer.FirstElement!=0u||s.Buffer.NumElements!=64u||
     b.ByteWidth!=256u||b.StructureByteStride!=4u||(b.BindFlags&D3D11_BIND_SHADER_RESOURCE)==0u||
     (b.MiscFlags&D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)==0u||b.CPUAccessFlags!=0u||
     (b.Usage!=D3D11_USAGE_DEFAULT&&b.Usage!=D3D11_USAGE_IMMUTABLE)||serial_==std::numeric_limits<std::uint64_t>::max()){
   error="Receipt C24 exact structured64 source unavailable";return false;
  }
  // All coordinates are bytes. Source words4..27 map to the WHOLE 96-byte destination.
  // Copy ordering and object lifetime remain on this same serialized immediate draw context.
  const D3D11_BOX box{16u,0u,0u,112u,1u,1u};
  context->CopySubresourceRegion(constants_.Get(),0u,0u,0u,0u,buffer.Get(),0u,&box);
  ++serial_;source_=selected_receipt;source_buffer_=buffer;context_=context;issued_=true;return true;
 }
 [[nodiscard]] ID3D11Buffer*constants()const noexcept{return issued_?constants_.Get():nullptr;}
 [[nodiscard]] ID3D11ShaderResourceView*source_receipt()const noexcept{return issued_?source_.Get():nullptr;}
 [[nodiscard]] std::uint64_t copy_serial()const noexcept{return issued_?serial_:0u;}
 // True means COPY commands issued only. Caller queues the sealed BG owner and its draw before
 // the next queue; never reinterpret this as GPU completion or independently certified C24.
};
}
#endif
