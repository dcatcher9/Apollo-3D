// PRIVATE RECIPE5 SOURCE-AVAILABILITY FIXTURE. External draft; no live/capture authority.
// Synthetic typed final/SCT/CB/UIP inputs are fixture stimuli, never model-completion certificates.
#include <gtest/gtest.h>
#ifdef _WIN32
#include "subtitle_helper_source_pins.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <dxgi1_2.h>
#include <src/crypto.h>
#include <src/host_sbs_direct_glyph_tiles.h>
#include <src/host_sbs_display_snapshot_rgb8_analysis_gpu.h>
#include <src/host_sbs_resolution.h>
#include <src/host_sbs_shader_cache.h>
#include <src/host_sbs_direct_glyph_publication_gpu.h>
namespace {
namespace cg=models::display_snapshot_rgb8_analysis_gpu_v1;
namespace cache=models::host_sbs_shader_cache;
using Microsoft::WRL::ComPtr;using json=nlohmann::json;using bytes=std::vector<std::uint8_t>;
constexpr const char *helper_pin=subtitle_test_source_pins::display_snapshot_rgb8;
struct tex_t {ComPtr<ID3D11Texture2D> texture;ComPtr<ID3D11ShaderResourceView> srv;ComPtr<ID3D11UnorderedAccessView> uav;};
struct buf_t {ComPtr<ID3D11Buffer> buffer;ComPtr<ID3D11ShaderResourceView> srv;ComPtr<ID3D11UnorderedAccessView> uav;};
class HostSbsDisplaySnapshotRgb8GpuTest:public ::testing::Test {
protected:
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
 std::array<ComPtr<ID3D11ComputeShader>,4> shaders;cg::resources_t owner;cg::program_t program;cg::request_t request;
 tex_t raw,display;buf_t root,sct,uip;ComPtr<ID3D11Buffer> contour;
 std::array<UINT,64> roots{};std::array<UINT,32> scts{};std::array<UINT,40> uips{};std::array<UINT,24> cb{};
 std::vector<std::uint16_t> pixels;
 std::filesystem::path directory;json report;std::string error;
 UINT W=257,H=43,crop=24,B=19,rawW=299,rawH=90,left=17,top=11;DXGI_FORMAT rawFormat=DXGI_FORMAT_R8G8B8A8_UNORM;
 static void need(bool x,const char *why){if(!x)throw std::runtime_error(why);}
 static void hr(HRESULT h){need(SUCCEEDED(h),"Recipe5 fixture D3D failure");}
 static std::string hash(const void *p,std::size_t n){auto h=crypto::hash(std::string_view(static_cast<const char*>(p),n));const char *hex="0123456789abcdef";std::string out;for(auto b:h){out+=hex[b>>4];out+=hex[b&15];}return out;}
 static void unbind(ID3D11DeviceContext *c){ID3D11ShaderResourceView *s[16]{};ID3D11UnorderedAccessView *u[8]{};ID3D11Buffer *b[3]{};c->CSSetShaderResources(0,16,s);c->CSSetUnorderedAccessViews(0,8,u,nullptr);c->CSSetConstantBuffers(0,3,b);}
 static double half_value(std::uint16_t h){UINT e=(h>>10)&31,m=h&1023;double v=e?std::ldexp(double(1024+m),int(e)-25):std::ldexp(double(m),-24);return h&0x8000?-v:v;}
 static std::uint16_t nearest_half(double v){UINT lo=0,hi=0x3c00;while(lo<hi){UINT mid=(lo+hi)/2;if(half_value(std::uint16_t(mid))<v)lo=mid+1;else hi=mid;}if(!lo)return 0;double a=v-half_value(std::uint16_t(lo-1)),b=half_value(std::uint16_t(lo))-v;return std::uint16_t(a<b||(a==b&&((lo-1)&1)==0)?lo-1:lo);}
 void SetUp()override {
  const char *sr=std::getenv("SUNSHINE_TEST_DISPLAY_RGB8_SHADER_ROOT"),*out=std::getenv("SUNSHINE_TEST_DISPLAY_RGB8_OUTPUT"),*dr=std::getenv("SUNSHINE_TEST_DISPLAY_RGB8_DRIVER");
  if(!sr||!out||!dr)GTEST_SKIP()<<"Explicit private recipe5 source/output/driver required";
  need(std::string(dr)=="warp"||std::string(dr)=="hardware","Unknown driver");
  D3D_FEATURE_LEVEL level{},wanted[]{D3D_FEATURE_LEVEL_11_0};
  hr(D3D11CreateDevice(nullptr,std::string(dr)=="hardware"?D3D_DRIVER_TYPE_HARDWARE:D3D_DRIVER_TYPE_WARP,nullptr,0,wanted,1,D3D11_SDK_VERSION,&device,&level,&context));need(level==D3D_FEATURE_LEVEL_11_0,"Feature level");
  ComPtr<IDXGIDevice> xd;ComPtr<IDXGIAdapter> adapter;hr(device.As(&xd));hr(xd->GetAdapter(&adapter));DXGI_ADAPTER_DESC desc{};hr(adapter->GetDesc(&desc));
  constexpr auto &specs=cache::direct_glyph_source_audit_specs;
  auto cs=cache::snapshot_sources(sr,specs);need(cache::source_closure_sha256(cs)==cache::direct_glyph_source_audit_source_closure_sha256,"Producer source drift");
  for(std::size_t i=0;i<4;++i){auto b=cache::get(cs,specs[i]);need(b&&!b->empty(),"No source bytecode");hr(device->CreateComputeShader(b->data(),b->size(),nullptr,&shaders[i]));}
  program={shaders[0].Get(),shaders[1].Get(),shaders[2].Get(),shaders[3].Get()};
  directory=std::filesystem::path(out)/::testing::UnitTest::GetInstance()->current_test_info()->name();
  need(!std::filesystem::exists(directory),"Refuse output overwrite");std::filesystem::create_directories(directory);
  report={{"schema",1},{"driver",dr},{"helper_source_sha256",helper_pin},{"producer_source_closure_sha256",cache::direct_glyph_source_audit_source_closure_sha256},{"compile_flags",34816},{"scope","Synthetic source availability only; no live/capture/model/UI/EMPTY or HDR-quality proof"},{"adapter",{{"vendor_id",desc.VendorId},{"device_id",desc.DeviceId},{"luid_low",desc.AdapterLuid.LowPart},{"luid_high",desc.AdapterLuid.HighPart}}},{"artifacts",json::object()},{"observations",json::array()}};
 }
 void TearDown()override {if(context){context->SetPredication(nullptr,FALSE);unbind(context.Get());}if(!directory.empty()){report["gtest_failed"]=HasFailure();std::ofstream f(directory/"report.json");f<<report.dump(2)<<'\n';}}
 json save(const std::string &name,const bytes &b){auto p=directory/name;need(!std::filesystem::exists(p),"Refuse artifact overwrite");std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.data()),b.size());f.close();json v{{"path",std::filesystem::absolute(p).string()},{"bytes",b.size()},{"sha256",hash(b.data(),b.size())}};report["artifacts"][name]=v;return v;}
 template<std::size_t N>static bytes pack(const std::array<UINT,N>&v){bytes b(sizeof(v));std::memcpy(b.data(),v.data(),b.size());return b;}
 static std::vector<UINT> words(const bytes &b){need(b.size()%4==0,"Word bytes");std::vector<UINT> out(b.size()/4);std::memcpy(out.data(),b.data(),b.size());return out;}
 template<class T>ComPtr<ID3D11Buffer> constants(const T&v){static_assert(sizeof(T)%16==0);D3D11_BUFFER_DESC d{};d.ByteWidth=sizeof(T);d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA init{&v,0,0};ComPtr<ID3D11Buffer> b;hr(device->CreateBuffer(&d,&init,&b));return b;}
 buf_t buffer(UINT count,bool raw=false,bool indirect=false,bool writable=false,const void *initial=nullptr){
  buf_t b;D3D11_BUFFER_DESC d{};d.ByteWidth=count*4;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE|(writable?D3D11_BIND_UNORDERED_ACCESS:0);
  d.MiscFlags=(raw?D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS:D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)|(indirect?D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS:0);d.StructureByteStride=raw?0:4;
  D3D11_SUBRESOURCE_DATA init{initial,0,0};hr(device->CreateBuffer(&d,initial?&init:nullptr,&b.buffer));
  D3D11_SHADER_RESOURCE_VIEW_DESC s{};s.Format=raw?DXGI_FORMAT_R32_TYPELESS:DXGI_FORMAT_UNKNOWN;s.ViewDimension=raw?D3D11_SRV_DIMENSION_BUFFEREX:D3D11_SRV_DIMENSION_BUFFER;
  if(raw){s.BufferEx.NumElements=count;s.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;}else s.Buffer.NumElements=count;
  hr(device->CreateShaderResourceView(b.buffer.Get(),&s,&b.srv));
  if(writable){D3D11_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=s.Format;u.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;u.Buffer.NumElements=count;u.Buffer.Flags=raw?D3D11_BUFFER_UAV_FLAG_RAW:0;hr(device->CreateUnorderedAccessView(b.buffer.Get(),&u,&b.uav));}
  return b;
 }
 tex_t texture(UINT w,UINT h,DXGI_FORMAT f,bool writable,const void *initial=nullptr,UINT pitch=0){
  tex_t t;D3D11_TEXTURE2D_DESC d{};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;d.Format=f;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE|(writable?D3D11_BIND_UNORDERED_ACCESS:0);
  D3D11_SUBRESOURCE_DATA init{initial,pitch,0};hr(device->CreateTexture2D(&d,initial?&init:nullptr,&t.texture));hr(device->CreateShaderResourceView(t.texture.Get(),nullptr,&t.srv));if(writable)hr(device->CreateUnorderedAccessView(t.texture.Get(),nullptr,&t.uav));return t;
 }
 bytes read(ID3D11Resource *r,UINT pixelbytes=4){
  D3D11_RESOURCE_DIMENSION kind{};r->GetType(&kind);ComPtr<ID3D11Resource> staging;UINT row=0,rows=1;
  if(kind==D3D11_RESOURCE_DIMENSION_BUFFER){ComPtr<ID3D11Buffer>b;hr(r->QueryInterface(IID_PPV_ARGS(&b)));D3D11_BUFFER_DESC d{};b->GetDesc(&d);row=d.ByteWidth;d.Usage=D3D11_USAGE_STAGING;d.BindFlags=d.MiscFlags=d.StructureByteStride=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ComPtr<ID3D11Buffer>x;hr(device->CreateBuffer(&d,nullptr,&x));staging=x;}
  else{ComPtr<ID3D11Texture2D>t;hr(r->QueryInterface(IID_PPV_ARGS(&t)));D3D11_TEXTURE2D_DESC d{};t->GetDesc(&d);row=d.Width*pixelbytes;rows=d.Height;d.Usage=D3D11_USAGE_STAGING;d.BindFlags=d.MiscFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ComPtr<ID3D11Texture2D>x;hr(device->CreateTexture2D(&d,nullptr,&x));staging=x;}
  context->CopyResource(staging.Get(),r);D3D11_MAPPED_SUBRESOURCE m{};hr(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&m));bytes b(std::size_t(row)*rows);
  for(UINT y=0;y<rows;++y)std::memcpy(b.data()+std::size_t(y)*row,static_cast<const char*>(m.pData)+std::size_t(y)*(kind==D3D11_RESOURCE_DIMENSION_BUFFER?row:m.RowPitch),row);
  context->Unmap(staging.Get(),0);return b;
 }
 bytes readView(ID3D11ShaderResourceView *v,UINT pixelbytes=4){ComPtr<ID3D11Resource>r;v->GetResource(&r);return read(r.Get(),pixelbytes);}

 void upload(){context->UpdateSubresource(contour.Get(),0,nullptr,cb.data(),0,0);context->UpdateSubresource(root.buffer.Get(),0,nullptr,roots.data(),0,0);context->UpdateSubresource(sct.buffer.Get(),0,nullptr,scts.data(),0,0);context->UpdateSubresource(uip.buffer.Get(),0,nullptr,uips.data(),0,0);}
 void author(UINT transfer=0,bool direct=false,bool empty=false,std::uint64_t frame=0x100000007ull,std::uint64_t token=0xf000000000001007ull){
  rawFormat=transfer==0?DXGI_FORMAT_R8G8B8A8_UNORM:DXGI_FORMAT_R16G16B16A16_FLOAT;
  bytes rb(std::size_t(rawW)*rawH*(transfer==0?4:8),0);raw=texture(rawW,rawH,rawFormat,false,rb.data(),rawW*(transfer==0?4:8));
  std::array<std::uint16_t,256> lut{};for(UINT c=0;c<256;++c)lut[c]=nearest_half(double(c)/255.0);
  pixels.resize(std::size_t(W)*B*4);for(UINT y=0;y<B;++y)for(UINT x=0;x<W;++x){auto i=(std::size_t(y)*W+x)*4;pixels[i]=lut[(x+UINT(frame))%256];pixels[i+1]=lut[(73*x+19*y+UINT(frame))%256];pixels[i+2]=lut[(255-(x%256)+17*y+UINT(frame))%256];pixels[i+3]=0x3c00;}
  display=texture(W,B,DXGI_FORMAT_R16G16B16A16_FLOAT,false,pixels.data(),W*8);
  cb={W,H,crop,B,65,35,20,35,0,0,65,35,UINT(frame),UINT(frame>>32),0x11u,2u,0x60000007u,3u,0,0,1,27,0,0};
  std::uint64_t obs=(std::uint64_t(cb[17])<<32)|cb[16],gen=(std::uint64_t(cb[15])<<32)|cb[14];
  roots={};UINT lo=UINT(token),hi=UINT(token>>32);roots[0]=1;roots[1]=1^0xd1ec15a5u^0x52434f4fu;roots[2]=lo;roots[3]=hi;roots[4]=lo^0xa3756c91u;roots[5]=hi^0x5c8a936eu;roots[6]=0x47524243u;roots[7]=0x52434f4fu;
  std::copy_n(roots.begin()+2,4,roots.begin()+8);roots[12]=0x54535152u;roots[13]=1;roots[14]=1^0x6f435257u;
  if(!direct){roots[20]=roots[21]=roots[22]=1;}roots[44]=(cb[4]+15)/16;roots[45]=(cb[5]+15)/16;roots[46]=1;roots[52]=roots[53]=roots[54]=1;roots[57]=roots[58]=1;roots[60]=roots[61]=roots[62]=1;
  scts={};scts[0]=5;scts[1]=0x35544353u;scts[2]=empty?0:1;scts[3]=13;std::copy_n(cb.begin()+12,4,scts.begin()+4);std::copy_n(cb.begin(),12,scts.begin()+8);scts[20]=cb[16];scts[21]=cb[17];scts[22]=27;scts[23]=1;
  for(UINT i=0;i<40;++i)uips[i]=0x70000000u+i;uips[0]=4;uips[1]=0x34504955u;uips[2]=1;uips[3]=0;std::copy_n(cb.begin()+12,4,uips.begin()+4);uips[8]=cb[16];uips[9]=cb[17];uips[10]=W;uips[11]=H;std::copy_n(cb.begin()+4,8,uips.begin()+12);uips[20]=scts[3];uips[21]=uips[22]=uips[23]=0;
  contour=constants(cb);root=buffer(64,true,true,false,roots.data());sct=buffer(32,false,false,false,scts.data());uip=buffer(40,false,false,false,uips.data());
  need(owner.ensure_direct_audit(device.Get(),W,H,crop,B,error),"Ensure recipe5");
  request={display.srv.Get(),raw.srv.Get(),contour.Get(),sct.srv.Get(),root.buffer.Get(),{left,top,left+W,top+H},frame,obs,gen,token,transfer};
 }
 // One audit captures the proof and then the envelope from the same queued tuple.
 void queue(ID3D11Buffer *work=nullptr){need(owner.queue_current_audit(context.Get(),program,request,uip.srv.Get(),error,work).has_value(),error.c_str());}
 std::array<UINT,64> expected_proof()const {
  std::array<UINT,64> p{};p[0]=cg::direct_audit_tag;p[1]=1;p[2]=5;p[3]=1;std::copy_n(cb.begin()+12,2,p.begin()+4);std::copy_n(cb.begin()+16,2,p.begin()+6);std::copy_n(cb.begin()+14,2,p.begin()+8);p[10]=UINT(request.token);p[11]=UINT(request.token>>32);std::copy_n(cb.begin(),4,p.begin()+12);std::copy(request.roi_ltrb.begin(),request.roi_ltrb.end(),p.begin()+16);
  p[20]=rawW;p[21]=rawH;p[22]=p[23]=UINT(rawFormat);p[24]=8;p[25]=4;p[26]=p[27]=p[28]=1;p[30]=request.transfer_mode;p[31]=cg::direct_audit_features;p[32]=p[33]=p[35]=W*B;p[36]=p[37]=((W+15)/16)*((B+15)/16);p[39]=scts[3];std::copy_n(cb.begin()+4,4,p.begin()+40);std::copy_n(cb.begin()+8,4,p.begin()+44);std::copy_n(cb.begin()+20,4,p.begin()+48);std::copy_n(cb.begin()+16,4,p.begin()+52);std::copy_n(roots.begin()+60,4,p.begin()+56);p[60]=10;p[61]=0;p[62]=0;p[63]=1;return p;
 }
 void check_outputs(){EXPECT_EQ(readView(owner.views().source_proof.Get()),pack(expected_proof()));}
 void prefix(){auto e=words(readView(owner.views().uip_envelope.Get()));need(e.size()==104,"Envelope104");EXPECT_TRUE(std::equal(uips.begin(),uips.end(),e.begin()));}
 json snapshot(const std::string&name,const std::string&kind){
  json row{{"name",name},{"kind",kind},{"roi_wh",{W,H}},{"crop_top",crop},{"band_h",B},{"raw_wh",{rawW,rawH}},{"roi_ltrb",request.roi_ltrb},{"transfer_mode",request.transfer_mode},{"raw_format",UINT(rawFormat)},{"frame",request.frame},{"observation_us",request.observation_us},{"generation",request.analysis_generation},{"token",request.token},{"synthetic_tuple",true},{"source_availability_only",true}};
  for(auto [key,value]:std::initializer_list<std::pair<std::string,bytes>>{{"display",read(display.texture.Get(),8)},{"proof",readView(owner.views().source_proof.Get())},{"cb",pack(cb)},{"sct",pack(scts)},{"root",pack(roots)},{"uip",pack(uips)}})row[key]=save(name+"-"+key+".bin",value);
  report["observations"].push_back(row);return row;
 }
 void positive(const std::string&name){auto before=read(display.texture.Get(),8);queue();check_outputs();prefix();EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43],1u);EXPECT_EQ(read(display.texture.Get(),8),before);auto row=snapshot(name,"positive");row["envelope"]=save(name+"-envelope.bin",readView(owner.views().uip_envelope.Get()));report["observations"].back()=row;}
};
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, AllCodesBothRootModesAllTransfersAndEmptySource) {
 for(UINT transfer=0;transfer<3;++transfer)for(bool direct:{false,true})for(bool empty:{false,true}){author(transfer,direct,empty);positive("m"+std::to_string(transfer)+(direct?"-direct":"-queued")+(empty?"-empty":"-active"));}
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, DirectAndQueuedSourceOutputsIdentical) {
 author();queue();auto p=readView(owner.views().source_proof.Get()),e=readView(owner.views().uip_envelope.Get());roots[20]=roots[21]=roots[22]=0;upload();queue();EXPECT_EQ(readView(owner.views().source_proof.Get()),p);EXPECT_EQ(readView(owner.views().uip_envelope.Get()),e);auto row=snapshot("direct-identical","positive");row["compared_queued_proof"]=save("queued-proof.bin",p);row["compared_queued_envelope"]=save("queued-envelope.bin",e);row["envelope"]=save("direct-envelope.bin",readView(owner.views().uip_envelope.Get()));report["observations"].back()=row;
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, MalformedRoot80ShapesRevokeSourceOnly) {
 author();queue();auto good=roots;UINT n=0;for(auto g:{std::array<UINT,4>{1,0,1,0},std::array<UINT,4>{0,1,1,0},std::array<UINT,4>{1,1,0,0},std::array<UINT,4>{1,1,1,1},std::array<UINT,4>{2,1,1,0},std::array<UINT,4>{0,0,0,1}}){roots=good;std::copy(g.begin(),g.end(),roots.begin()+20);upload();queue();auto p=words(readView(owner.views().source_proof.Get()));EXPECT_EQ(p[0],0u);EXPECT_EQ(p[3],0u);EXPECT_EQ(p[33],0u);EXPECT_EQ(p[34],0u);EXPECT_EQ(p[35],0u);EXPECT_EQ(p[38],3u);snapshot("bad80-"+std::to_string(n++),"join_reject");}
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, InvalidCellsRevokeFreshReadiness) {
 const std::array<std::array<std::uint16_t,4>,8> bad{{{0,0,0,0},{0,0,0,0x3800},{0x7e00,0,0,0x3c00},{0x7c00,0,0,0x3c00},{0xfc00,0,0,0x3c00},{0x8001,0,0,0x3c00},{0x3c01,0,0,0x3c00},{0,0,0,0xbc00}}};
 for(UINT k=0;k<bad.size();++k){author(2);queue();auto at=(std::size_t(2)*W+3)*4;std::copy(bad[k].begin(),bad[k].end(),pixels.begin()+at);context->UpdateSubresource(display.texture.Get(),0,nullptr,pixels.data(),W*8,0);queue();auto p=words(readView(owner.views().source_proof.Get()));EXPECT_EQ(p[0],0u);EXPECT_EQ(p[3],0u);EXPECT_EQ(p[33],W*B);EXPECT_EQ(p[34],1u);EXPECT_EQ(p[35],W*B-1);EXPECT_EQ(p[38],4u);prefix();EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43],0u);auto row=snapshot("invalid-"+std::to_string(k),"invalid");row["invalid_flat_pixel"]=2*W+3;report["observations"].back()=row;}
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, RootZeroHoldsPublishedTupleThenFreshTokenWrites) {
 author();queue();auto p=readView(owner.views().source_proof.Get()),e=readView(owner.views().uip_envelope.Get());auto good=roots;UINT n=0;
 for(auto g:{std::array<UINT,4>{0,1,1,0},std::array<UINT,4>{0,0,0,0},std::array<UINT,4>{1,0,1,0},std::array<UINT,4>{1,1,0,0}}){author(0,false,false,0x100000020ull+n,0xe000000000002000ull+n);std::copy(g.begin(),g.end(),roots.begin()+60);upload();queue();EXPECT_EQ(readView(owner.views().source_proof.Get()),p);EXPECT_EQ(readView(owner.views().uip_envelope.Get()),e);auto row=snapshot("root0-"+std::to_string(n++),"root0_hold");row["envelope"]=save(row["name"].get<std::string>()+"-envelope.bin",readView(owner.views().uip_envelope.Get()));row["held_proof"]=save(row["name"].get<std::string>()+"-held-proof.bin",p);row["held_envelope"]=save(row["name"].get<std::string>()+"-held-envelope.bin",e);report["observations"].back()=row;}
 author(1,false,false,0x100000099ull,0xe000000000009999ull);positive("fresh-token");EXPECT_NE(readView(owner.views().source_proof.Get()),p);
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, EmptyGatePreservesSourceAndPublishesCurrentZero) {
  namespace model = models::host_sbs_direct_glyph_publication;
  namespace v2 = models::depth_coordinate_v2;
  model::resources_t matte;
  ASSERT_TRUE(matte.load_direct(device.Get(), SUNSHINE_SHADERS_DIR, error, false)) << error;
  H = 65u;
  B = models::subtitle_ocr_source_crop_height(W, H);
  crop = H - B;
  constexpr UINT field_w = 770u, field_h = 434u;
  const auto start = v2::subtitle_ocr_project_row_ceil(W, H, field_w, field_h, v2::subtitle_ocr_safe_row_top);
  const auto end = v2::subtitle_ocr_project_row_ceil(W, H, field_w, field_h, v2::subtitle_ocr_safe_row_bottom);
  ASSERT_TRUE(start && end && start.value < end.value);
  buf_t ocr, condition;
  model::request_t input;
  const auto author_current = [&](bool empty, std::uint64_t frame) {
    author(0u, true, empty, frame, 0xf000000000001000ull + frame);
    const std::array<UINT, 4> field {field_w, field_h, start.value, end.value};
    const std::array<UINT, 4> content {0u, 0u, field_w, field_h};
    std::copy(field.begin(), field.end(), cb.begin() + 4u);
    std::copy(content.begin(), content.end(), cb.begin() + 8u);
    std::copy_n(cb.begin(), 12u, scts.begin() + 8u);
    scts[3u] = 0u;
    std::copy_n(cb.begin() + 4u, 8u, uips.begin() + 12u);
    uips[3u] = empty ? 0u : 1u;
    uips[20u] = 0u;
    roots[44u] = (field_w + 15u) / 16u;
    roots[45u] = (field_h + 15u) / 16u;
    upload();

    std::array<UINT, v2::subtitle_ocr_record_word_count> detector {
      v2::subtitle_ocr_record_schema, v2::subtitle_ocr_record_tag, 1u
    };
    detector[3u] = detector[4u] = empty ? 0u : 1u;
    std::copy_n(cb.begin() + 12u, 4u, detector.begin() + 5u);
    std::copy_n(cb.begin(), 2u, detector.begin() + 9u);
    std::copy(field.begin(), field.end(), detector.begin() + 11u);
    if (!empty) {
      const std::array<UINT, 8> core {380u, start.value + 8u, 390u, start.value + 12u, std::bit_cast<UINT>(1.0f), 0u, 1u, 0u};
      const std::array<UINT, 8> cover {370u, start.value + 4u, 400u, start.value + 16u, std::bit_cast<UINT>(1.0f), 0u, 1u, 0u};
      std::copy(core.begin(), core.end(), detector.begin() + v2::subtitle_ocr_raw_box_offset);
      std::copy(cover.begin(), cover.end(), detector.begin() + v2::subtitle_ocr_final_box_offset);
    }
    const std::array<UINT, v2::subtitle_condition_param_word_count> scp {
      v2::subtitle_condition_param_schema, v2::subtitle_condition_param_tag, empty ? 0u : 1u
    };
    ocr = buffer(detector.size(), false, false, false, detector.data());
    condition = buffer(scp.size(), false, false, false, scp.data());
    input = {.width = W, .height = B, .frame = request.frame, .observation_us = request.observation_us,
      .analysis_generation = request.analysis_generation, .token = request.token, .current_constants = contour.Get(),
      .ocr = ocr.srv.Get(), .sct = sct.srv.Get(), .condition = condition.srv.Get(), .root = root.srv.Get(),
      .postprocess_mode = 2u, .expected_work = 1u, .uip = uip.srv.Get(), .use_empty_gate = true};
  };
  const auto prepare = [&] { return matte.prepare_empty_gate(context.Get(), input, error); };
  const auto publish = [&] {
    return matte.plan_direct(context.Get(), input, nullptr, false, error) &&
           matte.publish_direct(context.Get(), input, nullptr, nullptr);
  };
  const auto source_contents = [&] {
    const auto views = owner.views();
    return std::array<bytes, 2> {readView(views.source_proof.Get()), readView(views.uip_envelope.Get())};
  };
  const auto capture = [&](ID3D11Buffer *work) {
    return owner.queue_current_audit(context.Get(), program, request, uip.srv.Get(), error, work).has_value();
  };

  author_current(false, 0x100000007ull);
  auto *work = prepare();
  ASSERT_NE(work, nullptr) << error;
  EXPECT_EQ(read(work), pack(roots));
  ASSERT_TRUE(capture(work)) << error;
  check_outputs();
  const auto positive = source_contents();

  author_current(true, 0x100000008ull);
  ++input.observation_us;
  ASSERT_NE(prepare(), nullptr) << error;
  EXPECT_EQ(read(work), pack(roots));
  input.observation_us = request.observation_us;
  const std::array<UINT, v2::subtitle_condition_param_word_count> invalid_scp {};
  auto invalid_condition = buffer(invalid_scp.size(), false, false, false, invalid_scp.data());
  input.condition = invalid_condition.srv.Get();
  ASSERT_NE(prepare(), nullptr) << error;
  EXPECT_EQ(read(work), pack(roots));
  input.condition = condition.srv.Get();

  ASSERT_EQ(prepare(), work) << error;
  auto parked = roots;
  parked[60u] = 0u;
  EXPECT_EQ(read(work), pack(parked));
  EXPECT_EQ(read(root.buffer.Get()), pack(roots));
  ASSERT_TRUE(capture(work)) << error;
  EXPECT_EQ(source_contents(), positive);
  EXPECT_EQ(read(owner.views().final_root_mirror.Get()), pack(roots));
  EXPECT_EQ(words(read(owner.views().dispatch_arguments.Get())), (std::vector<UINT> {1u, 1u, 0u, 0u}));
  // The parked producer has no current model inputs. The typed empty witness alone publishes zero.
  ASSERT_TRUE(publish()) << error;
  const auto current = matte.views();
  ASSERT_TRUE(current.matte && current.alpha8 && current.receipt && current.work);
  const auto receipt = words(readView(current.receipt.Get()));
  ASSERT_EQ(receipt.size(), 64u);
  EXPECT_EQ(receipt[2u], 4u);
  EXPECT_EQ(receipt[3u], 0u);
  EXPECT_EQ(receipt[60u], static_cast<UINT>(request.token));
  EXPECT_EQ(receipt[61u], static_cast<UINT>(request.token >> 32u));
  EXPECT_EQ(receipt[62u], 4u);
  EXPECT_TRUE(std::equal(cb.begin(), cb.end(), receipt.begin() + 4u));
  EXPECT_TRUE(std::equal(scts.begin(), scts.end(), receipt.begin() + 28u));
  EXPECT_EQ(readView(current.matte.Get()), bytes(std::size_t(W) * B * 4u, 0u));
  EXPECT_EQ(readView(current.alpha8.Get()), bytes(std::size_t(W) * B * 4u, 0u));
  EXPECT_EQ(read(current.work.Get()).size(), std::size_t(model::direct_work_word_count) * 4u);

  author_current(false, 0x100000009ull);
  roots[60u] = 0u;
  upload();
  ASSERT_EQ(prepare(), work) << error;
  EXPECT_EQ(read(work), pack(roots));
  ASSERT_TRUE(capture(work)) << error;
  EXPECT_EQ(source_contents(), positive);
  ASSERT_TRUE(publish()) << error;
  EXPECT_EQ(words(readView(matte.views().receipt.Get())), receipt);

  author_current(false, 0x10000000aull);
  ASSERT_EQ(prepare(), work) << error;
  EXPECT_EQ(read(work), pack(roots));
  ASSERT_TRUE(capture(work)) << error;
  check_outputs();
  EXPECT_NE(source_contents(), positive);
  EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43u], 1u);
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, CurrentCbSctRootFaultsAndNoOcrAbstentionReject) {
 author();queue();auto C=cb;auto S=scts;auto R=roots;UINT n=0;
 for(UINT k=0;k<24;++k){if(k==18)continue;cb=C;cb[k]^=1u;upload();queue();auto p=words(readView(owner.views().source_proof.Get()));EXPECT_EQ(p[3],0u)<<k;EXPECT_EQ(p[38],3u)<<k;auto row=snapshot("cb-"+std::to_string(k),"join_reject");row["fault_family"]="cb";row["fault_word"]=k;report["observations"].back()=row;++n;}cb=C;
 for(UINT k=0;k<32;++k){if(k==2||k==3)continue;scts=S;scts[k]^=1u;upload();queue();EXPECT_EQ(words(readView(owner.views().source_proof.Get()))[3],0u)<<k;auto row=snapshot("sct-"+std::to_string(k),"join_reject");row["fault_family"]="sct";row["fault_word"]=k;report["observations"].back()=row;++n;}scts=S;
 std::vector<UINT> rw;for(UINT k=0;k<16;++k)rw.push_back(k);for(UINT a:{44u,52u,56u})for(UINT k=a;k<a+4;++k)rw.push_back(k);rw.push_back(63);
 for(UINT k:rw){roots=R;roots[k]^=1u;upload();queue();EXPECT_EQ(words(readView(owner.views().source_proof.Get()))[3],0u)<<k;auto row=snapshot("root-"+std::to_string(k),"join_reject");row["fault_family"]="root";row["fault_word"]=k;report["observations"].back()=row;++n;}roots=R;
 // Legitimate source-independent alternatives must NOT be incorrectly rejected.
 scts=S;scts[2]=0;cb=C;cb[18]=1;roots=R;upload();queue();check_outputs();snapshot("legal-empty-reset","positive");
 // Real no-OCR finalization produces SCT23=0; work2 must never gain a fresh source certificate.
 cb=C;scts=S;roots=R;roots[13]=2;roots[14]=2^0x6f435257u;roots[1]=1^0xd1ec15a5u;roots[7]=0;roots[52]=0;roots[56]=1;scts[2]=0;scts[3]=0;scts[23]=0;upload();queue();EXPECT_EQ(words(readView(owner.views().source_proof.Get()))[3],0u);snapshot("no-ocr-abstention","join_reject");
 report["fault_count"]=n;
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, LiteralUipPrefixRejectsEveryUipFault) {
 author();queue();auto stable=readView(owner.views().uip_envelope.Get());auto v=owner.views();
 auto base=uips;for(UINT k:std::vector<UINT>{0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23}){uips=base;uips[k]^=1;context->UpdateSubresource(uip.buffer.Get(),0,nullptr,uips.data(),0,0);queue();prefix();EXPECT_EQ(words(readView(v.uip_envelope.Get()))[43],0u)<<k;}uips=base;upload();queue();EXPECT_EQ(readView(v.uip_envelope.Get()),stable);
 report["GPU_uip_faults"]=22;
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, HostDescriptorsForeignAliasesPredicationAndStateRestore) {
 author();queue();auto p=readView(owner.views().source_proof.Get()),e=readView(owner.views().uip_envelope.Get());auto original=request;
 const auto rejects=[&](ID3D11DeviceContext *c,ID3D11ShaderResourceView *u){return !owner.queue_current_audit(c,program,request,u,error).has_value();};
 for(auto roi:{std::array<UINT,4>{0,0,0,H},std::array<UINT,4>{left,top,0xffffffffu,top+H},std::array<UINT,4>{left,top,left+W,rawH+1}}){request=original;request.roi_ltrb=roi;EXPECT_TRUE(rejects(context.Get(),uip.srv.Get()));}
 request=original;request.transfer_mode=3;EXPECT_TRUE(rejects(context.Get(),uip.srv.Get()));request=original;
 auto wrong=texture(W,B,DXGI_FORMAT_R8G8B8A8_UNORM,false);request.display_snapshot_band=wrong.srv.Get();EXPECT_TRUE(rejects(context.Get(),uip.srv.Get()));request=original;
 // A missing or short UIP is refused before anything is captured.
 auto short_uip=buffer(39u);EXPECT_TRUE(rejects(context.Get(),nullptr));EXPECT_TRUE(rejects(context.Get(),short_uip.srv.Get()));
 ComPtr<ID3D11DeviceContext>deferred;hr(device->CreateDeferredContext(0,&deferred));EXPECT_TRUE(rejects(deferred.Get(),uip.srv.Get()));
 ComPtr<ID3D11Device>fd;ComPtr<ID3D11DeviceContext>fc;D3D_FEATURE_LEVEL level{},fl[]{D3D_FEATURE_LEVEL_11_0};hr(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,fl,1,D3D11_SDK_VERSION,&fd,&level,&fc));EXPECT_TRUE(rejects(fc.Get(),uip.srv.Get()));
 D3D11_TEXTURE2D_DESC td{};td.Width=rawW;td.Height=rawH;td.MipLevels=td.ArraySize=td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R8G8B8A8_UNORM;td.Usage=D3D11_USAGE_DEFAULT;td.BindFlags=D3D11_BIND_SHADER_RESOURCE;ComPtr<ID3D11Texture2D>ft;ComPtr<ID3D11ShaderResourceView>fs;hr(fd->CreateTexture2D(&td,nullptr,&ft));hr(fd->CreateShaderResourceView(ft.Get(),nullptr,&fs));request.parent_raw_source=fs.Get();EXPECT_TRUE(rejects(context.Get(),uip.srv.Get()));request=original;
 auto shortRoot=buffer(63,true,true,false);request.actual_final256=shortRoot.buffer.Get();EXPECT_TRUE(rejects(context.Get(),uip.srv.Get()));request=original;
 std::array<UINT,20> shortC{};auto smallCB=constants(shortC);request.actual_contour_constants96=smallCB.Get();EXPECT_TRUE(rejects(context.Get(),uip.srv.Get()));request=original;
 auto shortS=buffer(31);request.actual_sct32=shortS.srv.Get();EXPECT_TRUE(rejects(context.Get(),uip.srv.Get()));request=original;
 D3D11_QUERY_DESC qd{};qd.Query=D3D11_QUERY_OCCLUSION_PREDICATE;ComPtr<ID3D11Predicate>pred;hr(device->CreatePredicate(&qd,&pred));context->SetPredication(pred.Get(),FALSE);EXPECT_TRUE(rejects(context.Get(),uip.srv.Get()));
 ComPtr<ID3D11Predicate>actual_predicate;BOOL predicate_value=TRUE;context->GetPredication(&actual_predicate,&predicate_value);EXPECT_EQ(actual_predicate.Get(),pred.Get());EXPECT_FALSE(predicate_value);context->SetPredication(nullptr,FALSE);
 EXPECT_EQ(readView(owner.views().source_proof.Get()),p);EXPECT_EQ(readView(owner.views().uip_envelope.Get()),e);
 std::array<buf_t,16>ss;std::array<buf_t,8>uu;std::array<ComPtr<ID3D11Buffer>,3>bb;std::array<ID3D11ShaderResourceView*,16>s{};std::array<ID3D11UnorderedAccessView*,8>u{};std::array<ID3D11Buffer*,3>b{};
 for(UINT i=0;i<16;++i){ss[i]=buffer(4);s[i]=ss[i].srv.Get();}for(UINT i=0;i<8;++i){uu[i]=buffer(4,false,false,true);u[i]=uu[i].uav.Get();}for(UINT i=0;i<3;++i){std::array<UINT,4>x{i,0,0,0};bb[i]=constants(x);b[i]=bb[i].Get();}
 context->CSSetShaderResources(0,16,s.data());context->CSSetUnorderedAccessViews(0,8,u.data(),nullptr);context->CSSetConstantBuffers(0,3,b.data());context->CSSetShader(shaders[1].Get(),nullptr,0);
 {queue();std::array<ID3D11ShaderResourceView*,16>as{};std::array<ID3D11UnorderedAccessView*,8>au{};std::array<ID3D11Buffer*,3>ab{};context->CSGetShaderResources(0,16,as.data());context->CSGetUnorderedAccessViews(0,8,au.data());context->CSGetConstantBuffers(0,3,ab.data());for(UINT i=0;i<16;++i){EXPECT_EQ(as[i],s[i]);if(as[i])as[i]->Release();}for(UINT i=0;i<8;++i){EXPECT_EQ(au[i],u[i]);if(au[i])au[i]->Release();}for(UINT i=0;i<3;++i){EXPECT_EQ(ab[i],b[i]);if(ab[i])ab[i]->Release();}ComPtr<ID3D11ComputeShader>cs;UINT n=0;context->CSGetShader(&cs,nullptr,&n);EXPECT_EQ(cs.Get(),shaders[1].Get());EXPECT_EQ(n,0u);}
 report["host_rejections"]=15;report["CS_state_restored"]={{"SRV",16},{"UAV",8},{"CB",3},{"shader",true},{"queue",true}};
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, ActualMalformedCopyGridSeparatesCoverageFromValidity) {
 W=32;H=80;crop=16;B=64;rawW=64;rawH=96;left=5;top=7;author();queue();auto v=owner.views();
 auto uav=[&](ID3D11ShaderResourceView*s){ComPtr<ID3D11Resource>r;s->GetResource(&r);ComPtr<ID3D11UnorderedAccessView>u;hr(device->CreateUnorderedAccessView(r.Get(),nullptr,&u));return u;};
 auto p=uav(v.source_proof.Get());D3D11_UNORDERED_ACCESS_VIEW_DESC ad{};ad.Format=DXGI_FORMAT_R32_TYPELESS;ad.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ad.Buffer.NumElements=4;ad.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;ComPtr<ID3D11UnorderedAccessView>a;hr(device->CreateUnorderedAccessView(v.dispatch_arguments.Get(),&ad,&a));
 D3D11_SHADER_RESOURCE_VIEW_DESC rd{};rd.Format=DXGI_FORMAT_R32_TYPELESS;rd.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;rd.BufferEx.NumElements=64;rd.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;ComPtr<ID3D11ShaderResourceView>rs;hr(device->CreateShaderResourceView(v.final_root_mirror.Get(),&rd,&rs));UINT n=0;
 for(auto grid:{std::array<UINT,4>{4,2,1,0},std::array<UINT,4>{2,4,2,0},std::array<UINT,4>{3,4,1,0},std::array<UINT,4>{2,3,1,0}}){unbind(context.Get());ID3D11Buffer*c[2]{contour.Get(),v.request_constants.Get()};ID3D11ShaderResourceView*s[6]{display.srv.Get(),rs.Get(),sct.srv.Get(),nullptr,nullptr,raw.srv.Get()};ID3D11UnorderedAccessView*u[4]{nullptr,nullptr,p.Get(),a.Get()};context->CSSetConstantBuffers(0,2,c);context->CSSetShaderResources(0,6,s);context->CSSetUnorderedAccessViews(0,4,u,nullptr);context->CSSetShader(shaders[0].Get(),nullptr,0);context->Dispatch(1,1,1);ID3D11UnorderedAccessView*nullu=nullptr;context->CSSetUnorderedAccessViews(3,1,&nullu,nullptr);context->UpdateSubresource(v.dispatch_arguments.Get(),0,nullptr,grid.data(),0,0);context->CSSetShader(shaders[1].Get(),nullptr,0);context->DispatchIndirect(v.dispatch_arguments.Get(),0);context->CSSetShader(shaders[2].Get(),nullptr,0);context->DispatchIndirect(root.buffer.Get(),240);unbind(context.Get());auto got=words(readView(v.source_proof.Get()));EXPECT_EQ(got[0],0u);EXPECT_EQ(got[3],0u);EXPECT_EQ(got[34],0u);EXPECT_EQ(got[35],got[33]);EXPECT_EQ(got[38],2u);auto row=snapshot("grid-"+std::to_string(n++),"coverage_reject");row["actual_copy_grid"]=grid;report["observations"].back()=row;}
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, ShapeTransitionRetainsOldStorageAndWritesNewProof) {
 author();positive("first");auto held=owner.views();auto p=readView(held.source_proof.Get());auto oldpointer=held.source_proof.Get();need(owner.ensure_direct_audit(device.Get(),W,H,crop,B,error),"Stable ensure");EXPECT_EQ(owner.views().source_proof.Get(),oldpointer);W=243;H=37;crop=20;B=17;author(2,true,true,0x1000000aaull,0xe00000000000aaa1ull);positive("second");EXPECT_NE(owner.views().source_proof.Get(),oldpointer);EXPECT_EQ(readView(held.source_proof.Get()),p);report["retired_storage_held"]=true;
}

TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, DirectAuditUsesOriginalSnapshotAndPreservesSourceValidity) {
  author(0u, true);
  const auto capture = [&](ID3D11Buffer *work_root = nullptr) {
    const auto audit = owner.queue_current_audit(context.Get(), program, request, uip.srv.Get(), error, work_root);
    need(audit.has_value(), error.c_str());
    EXPECT_EQ(audit->display_snapshot_band(), display.srv.Get());
    EXPECT_EQ(audit->uip_envelope(), owner.views().uip_envelope.Get());
    return readView(audit->uip_envelope());
  };
  const auto expected_envelope = [&](bool invalid = false) {
    auto proof = expected_proof();
    if (invalid) {
      proof[0u] = proof[3u] = 0u;
      proof[34u] = 1u;
      --proof[35u];
      proof[38u] = 4u;
    }
    auto envelope = pack(uips);
    const auto proof_bytes = pack(proof);
    envelope.insert(envelope.end(), proof_bytes.begin(), proof_bytes.end());
    EXPECT_EQ(readView(owner.views().source_proof.Get()), proof_bytes);
    return envelope;
  };

  const auto positive = capture();
  EXPECT_EQ(positive, expected_envelope());
  author(0u, true, false, request.frame + 1u, request.token + 1u);
  roots[60u] = 0u;
  upload();
  EXPECT_EQ(capture(), positive);

  roots[60u] = 1u;
  upload();
  auto parked = roots;
  parked[60u] = 0u;
  auto work_root = buffer(64u, true, true, false, parked.data());
  EXPECT_EQ(capture(work_root.buffer.Get()), positive);

  pixels.back() = 0u;
  context->UpdateSubresource(display.texture.Get(), 0u, nullptr, pixels.data(), W * 8u, 0u);
  const auto invalid = capture();
  EXPECT_EQ(invalid, expected_envelope(true));
  EXPECT_NE(invalid, positive);

  author(0u, true, false, request.frame + 1u, request.token + 1u);
  const auto recovered = capture();
  EXPECT_EQ(recovered, expected_envelope());
  EXPECT_NE(recovered, positive);
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, DirectPublishesScaledScoresOnlyInCurrentNativeWork) {
  const auto &specs = cache::direct_glyph_plan_specs;
  const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, specs);
  ASSERT_TRUE(sources);
  ASSERT_EQ(cache::source_closure_sha256(sources), cache::direct_glyph_plan_source_closure_sha256);
  const auto bytecode = cache::get(sources, specs[1u]);
  ASSERT_TRUE(bytecode);
  ComPtr<ID3D11ComputeShader> shader;
  hr(device->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &shader));

  constexpr UINT width = 7u, height = 5u;
  const auto model_request = constants(std::array<UINT, 12u> {width, height, 38u});
  const auto contour_constants = constants(std::array<UINT, 24u> {width, height, 0u, height});
  const auto current_sct = buffer(32u);
  const auto receipt = buffer(64u, false, false, true);
  constexpr std::array<UINT, 4u> parked_dispatch {};
  const auto legacy_dispatch = buffer(parked_dispatch.size(), true, true, true, parked_dispatch.data());
  const auto matte = texture(width, height, DXGI_FORMAT_R32_UINT, true);
  const auto output = texture(width, height, DXGI_FORMAT_R32_UINT, true);

  for (const UINT scale : {1u, 2u, 4u, 3u}) {
    SCOPED_TRACE(scale);
    const UINT storage_scale = scale == 3u ? 2u : scale;
    const UINT scaled_width = (width + storage_scale - 1u) / storage_scale;
    const UINT scaled_height = (height + storage_scale - 1u) / storage_scale;
    const auto probability = [](auto x, auto y) {
      return float(1 + x + 2 * y) / 16.0f;
    };
    // Only the upper-left score lattice is current. The remaining texels carry poison.
    std::vector<UINT> codes(width * height, 0xffffffffu);
    for (UINT y = 0u; y < scaled_height; ++y) {
      for (UINT x = 0u; x < scaled_width; ++x) {
        codes[y * width + x] = scale == 1u ? 30u + x + y : std::bit_cast<UINT>(probability(x, y));
      }
    }
    const auto candidate = texture(width, height, DXGI_FORMAT_R32_UINT, false, codes.data(), width * 4u);
    const std::array<UINT, 36u> work_words {2u, 1u, 0u, scale, 1u, 0u, width, height - 1u};
    const auto work = buffer(work_words.size(), true, false, false, work_words.data());
    ID3D11Buffer *const cbuffers[] {model_request.Get(), contour_constants.Get()};
    std::array<ID3D11ShaderResourceView *, 16u> in {};
    in[7u] = work.srv.Get();
    in[13u] = current_sct.srv.Get();
    in[15u] = candidate.srv.Get();
    ID3D11UnorderedAccessView *const out[] {matte.uav.Get(), legacy_dispatch.uav.Get(), nullptr,
                                          output.uav.Get(), nullptr, receipt.uav.Get()};
    context->CSSetShader(shader.Get(), nullptr, 0u);
    context->CSSetConstantBuffers(0u, 2u, cbuffers);
    context->CSSetShaderResources(0u, in.size(), in.data());
    context->CSSetUnorderedAccessViews(0u, std::size(out), out, nullptr);
    context->Dispatch(1u, 1u, 1u);
    unbind(context.Get());

    const auto alpha = words(read(output.texture.Get()));
    const auto selected = words(read(matte.texture.Get()));
    const auto receipt_words = words(read(receipt.buffer.Get()));
    EXPECT_EQ(receipt_words[2u], scale == 3u ? 1u : 4u);
    EXPECT_EQ(receipt_words[3u], scale == 3u ? 1u : 0u);
    EXPECT_EQ(receipt_words[62u], 4u);
    EXPECT_EQ(words(read(legacy_dispatch.buffer.Get())), (std::vector<UINT> {0u, 0u, 0u, 0u}));
    for (UINT y = 0u; y < height; ++y) {
      for (UINT x = 0u; x < width; ++x) {
        UINT expected = 0u;
        if (scale != 3u && x >= 1u && y < height - 1u) {
          if (scale == 1u) {
            expected = codes[y * width + x];
          } else {
            const auto mx = std::clamp((x + 0.5f) / scale - 0.5f, 0.0f, float(scaled_width - 1u));
            const auto my = std::clamp((y + 0.5f) / scale - 0.5f, 0.0f, float(scaled_height - 1u));
            expected = UINT(std::nearbyint(255.0f * probability(mx, my)));
          }
        }
        EXPECT_EQ(alpha[y * width + x], expected) << x << "," << y;
        EXPECT_EQ(selected[y * width + x], expected >= 38u ? 1u : 0u) << x << "," << y;
      }
    }
  }
}

class HostSbsDirectGlyphPublishGpuTest: public HostSbsDisplaySnapshotRgb8GpuTest {
protected:
  buf_t structured(UINT count, UINT stride, const void *initial) {
    buf_t b;
    D3D11_BUFFER_DESC d {count * stride, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0u, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, stride};
    D3D11_SUBRESOURCE_DATA data {initial, 0, 0};
    hr(device->CreateBuffer(&d, initial ? &data : nullptr, &b.buffer));
    hr(device->CreateShaderResourceView(b.buffer.Get(), nullptr, &b.srv));
    hr(device->CreateUnorderedAccessView(b.buffer.Get(), nullptr, &b.uav));
    return b;
  }

  void SetUp() override {
    constexpr auto level = D3D_FEATURE_LEVEL_11_0;
    ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, &level, 1u,
      D3D11_SDK_VERSION, &device, nullptr, &context)));
  }
};

TEST_F(HostSbsDirectGlyphPublishGpuTest, BatchedRolesKeepTheOwnedColumnsAndPartialLastTile) {
  const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, cache::direct_glyph_palette_specs);
  ASSERT_TRUE(sources);
  const auto code = cache::get(sources, cache::host_sbs_direct_glyph_palette_export_roles);
  ASSERT_TRUE(code);
  ComPtr<ID3D11ComputeShader> shader;
  hr(device->CreateComputeShader(code->data(), code->size(), nullptr, &shader));

  constexpr auto tile = models::host_sbs_direct_glyph_gpu::palette_tile;
  constexpr UINT width = 881, cells = tile.height * tile.width;
  for (const UINT rows : {28u, 37u, tile.height}) {
    SCOPED_TRACE(rows);
    const UINT top = (tile.height - rows) / 2u;
    std::vector<float> logits(2 * 4 * cells, -12.0f);
    for (UINT y = 0; y < rows; ++y) {
      for (UINT x = 0; x < 440; ++x) {
        const UINT index = (top + y) * tile.width + 36 + x;
        logits[cells + index] = 12.0f;
        logits[4 * cells + 2 * cells + index] = 12.0f;
      }
    }
    const auto input = buffer(UINT(logits.size()), false, false, false, logits.data());
    using roles_t = std::array<float, 4>;
    const std::vector<roles_t> zero(width * rows + 1);
    D3D11_BUFFER_DESC desc {UINT(zero.size() * sizeof(roles_t)), D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS,
      0, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, sizeof(roles_t)};
    D3D11_SUBRESOURCE_DATA data {zero.data(), 0, 0};
    ComPtr<ID3D11Buffer> output;
    ComPtr<ID3D11UnorderedAccessView> target;
    hr(device->CreateBuffer(&desc, &data, &output));
    hr(device->CreateUnorderedAccessView(output.Get(), nullptr, &target));
    const auto cb = constants(std::array<UINT, 8> {0, width, 0, 440, 0, 0, 0, rows});
    context->CSSetShader(shader.Get(), nullptr, 0);
    context->CSSetConstantBuffers(0, 1, cb.GetAddressOf());
    context->CSSetShaderResources(12, 1, input.srv.GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, target.GetAddressOf(), nullptr);
    context->Dispatch(28, (rows + 7u) / 8u, 2);
    context->ClearState();
    const auto values = words(read(output.Get()));
    for (UINT y : {0u, rows - 1u}) {
      EXPECT_EQ(values[(y * width + 439) * 4 + 1], 0u);
      for (UINT x : {440u, 879u}) EXPECT_GT(std::bit_cast<float>(values[(y * width + x) * 4 + 1]), .99f);
      EXPECT_GT(std::bit_cast<float>(values[(y * width + 880) * 4 + 2]), .99f);
    }
    EXPECT_EQ(values[width * rows * 4], 0u);
  }
}

TEST_F(HostSbsDirectGlyphPublishGpuTest, CommonBackdropDoesNotDisplaceAnOutlineColor) {
  const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, cache::direct_glyph_palette_specs);
  ASSERT_TRUE(sources);
  ASSERT_EQ(cache::source_closure_sha256(sources), cache::direct_glyph_palette_source_closure_sha256);
  const auto code = cache::get(sources, cache::host_sbs_direct_glyph_palette_choose_colors);
  ASSERT_TRUE(code);
  ComPtr<ID3D11ComputeShader> shader;
  hr(device->CreateComputeShader(code->data(), code->size(), nullptr, &shader));

  using RoleVotes = std::array<float, 4>;
  const auto view = [&](const std::vector<RoleVotes> &values) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = static_cast<UINT>(values.size() * sizeof(RoleVotes));
    desc.StructureByteStride = sizeof(RoleVotes);
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    const D3D11_SUBRESOURCE_DATA data {values.data(), 0, 0};
    ComPtr<ID3D11Buffer> resource;
    hr(device->CreateBuffer(&desc, &data, &resource));
    ComPtr<ID3D11ShaderResourceView> result;
    hr(device->CreateShaderResourceView(resource.Get(), nullptr, &result));
    return result;
  };

  const std::array<UINT, 16> line {};
  const auto plan = structured(1, sizeof(line), line.data());
  enum class PaintCase { common_backdrop, equal_role_fractions, bright_backdrop, broad_outline };
  const auto check_backdrop = [&](UINT glyph, PaintCase paint_case) {
    SCOPED_TRACE(glyph);
    SCOPED_TRACE(static_cast<int>(paint_case));
    const UINT backdrop = 4912u - glyph;
    std::vector<RoleVotes> votes(2 * 4913), posterior(4913);
    if (paint_case == PaintCase::equal_role_fractions || paint_case == PaintCase::bright_backdrop) {
      // Bright background can collect stronger fill votes than the thin ink itself.
      votes[backdrop] = paint_case == PaintCase::bright_backdrop ? RoleVotes {400, 500, 100, 0} : RoleVotes {800, 100, 100, 0};
      votes[glyph] = {320, 40, 40, 0};
      votes[4913 + backdrop] = {390, 610, 0, 0};
      votes[4913 + glyph] = {0, 400, 0, 0};
    } else if (paint_case == PaintCase::broad_outline) {
      // The more frequent outline receives more total fill votes, even though their fraction is low.
      votes[backdrop] = {612, 48, 130, 10};
      votes[glyph] = {50, 28, 1, 1};
    } else {
      votes[backdrop] = {9950, 10, 40, 0};
      votes[glyph] = {984, 1000, 16, 0};
      for (UINT bin = 1; bin <= 200; ++bin)
        votes[bin] = {9.87f, 0, .13f, 0};
    }
    for (size_t bin = 0; bin < posterior.size(); ++bin) {
      const auto &v = votes[bin];
      const auto total = v[0] + v[1] + v[2] + v[3];
      for (size_t role = 0; role < v.size(); ++role)
        posterior[bin][role] = (v[role] + .12f) / (total + .48f);
    }
    ASSERT_GT(votes[backdrop][2] * posterior[backdrop][2], votes[glyph][2] * posterior[glyph][2]);
    const auto histogram = view(votes);
    const auto palette = view(posterior);
    const auto winners = buffer(2, false, false, true);
    const auto setup = constants(std::array<UINT, 8> {0, 0, 0, 0, 1, 390, 1010, 0});
    context->CSSetShader(shader.Get(), nullptr, 0);
    context->CSSetConstantBuffers(0, 1, setup.GetAddressOf());
    context->CSSetShaderResources(4, 1, palette.GetAddressOf());
    context->CSSetShaderResources(10, 1, histogram.GetAddressOf());
    context->CSSetShaderResources(11, 1, plan.srv.GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, winners.uav.GetAddressOf(), nullptr);
    context->Dispatch(1, 1, 1);
    context->ClearState();
    const UINT outline = paint_case == PaintCase::broad_outline ? backdrop : glyph;
    EXPECT_EQ(words(read(winners.buffer.Get())), (std::vector<UINT> {glyph, outline}));
  };
  for (const UINT glyph : {0u, 4912u}) {
    for (const auto paint_case : {PaintCase::common_backdrop, PaintCase::equal_role_fractions, PaintCase::bright_backdrop, PaintCase::broad_outline}) {
      check_backdrop(glyph, paint_case);
    }
  }
}

TEST_F(HostSbsDirectGlyphPublishGpuTest, WeakLineCanBeConfirmedByInkAnywhereInItsOwnedEncoderCells) {
  const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, cache::direct_glyph_palette_specs);
  ASSERT_TRUE(sources);
  const auto code = cache::get(sources, cache::host_sbs_direct_glyph_palette_choose_colors);
  ASSERT_TRUE(code);
  ComPtr<ID3D11ComputeShader> shader;
  hr(device->CreateComputeShader(code->data(), code->size(), nullptr, &shader));

  using Roles = std::array<float, 4>; // background, fill, outline, shadow
  constexpr UINT count = 512;
  const std::vector<Roles> bins(2 * 4913);
  const auto histogram = structured(static_cast<UINT>(bins.size()), sizeof(Roles), bins.data());
  const auto classify = [&](bool weak, float last_outline) {
    std::array<UINT, 16> line {};
    line[15] = weak;
    const auto plan = structured(1, sizeof(line), line.data());
    std::vector<Roles> predictions(count + 1, {0.05f, 0.10f, 0.10f, 0.75f});
    predictions[count - 1] = {0.5f - last_outline, 0.2f, last_outline, 0.3f};
    predictions[count] = {0, 1, 0, 0}; // Stale capacity past the end of this line.
    const auto roles = structured(count + 1, sizeof(Roles), predictions.data());
    const auto winners = buffer(2, false, false, true);
    const auto setup = constants(std::array<UINT, 8> {0, count / 2, 0, 0, 1, 1, 1, 2});
    context->CSSetShader(shader.Get(), nullptr, 0);
    context->CSSetConstantBuffers(0, 1, setup.GetAddressOf());
    context->CSSetShaderResources(1, 1, roles.srv.GetAddressOf());
    context->CSSetShaderResources(4, 1, histogram.srv.GetAddressOf());
    context->CSSetShaderResources(10, 1, histogram.srv.GetAddressOf());
    context->CSSetShaderResources(11, 1, plan.srv.GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, winners.uav.GetAddressOf(), nullptr);
    context->Dispatch(1, 1, 1);
    context->ClearState();
    return words(read(winners.buffer.Get()))[0];
  };
  EXPECT_EQ(classify(true, 0.19f), 0xffffffffu);
  EXPECT_EQ(classify(true, 0.20f), 0u);
  EXPECT_EQ(classify(false, 0.19f), 0u);
}

TEST_F(HostSbsDirectGlyphPublishGpuTest, ResolvedFringeRequiresCurrentCompletedTapsAndQ64) {
  const auto &specs = cache::direct_glyph_plan_specs;
  const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, specs);
  ASSERT_TRUE(sources);
  ASSERT_EQ(cache::source_closure_sha256(sources), cache::direct_glyph_plan_source_closure_sha256);
  const auto bytecode = cache::get(sources, specs[1u]);
  ASSERT_TRUE(bytecode);
  ComPtr<ID3D11ComputeShader> shader;
  hr(device->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &shader));

  constexpr UINT width = 32u, height = 16u;
  constexpr UINT fringe_flag = 32u, token_low = 13u, token_high = 29u;
  const auto contour_constants = constants(std::array<UINT, 24u> {width, height, 0u, height});
  constexpr std::array<UINT, 32u> sct_words {};
  const auto current_sct = buffer(sct_words.size(), false, false, false, sct_words.data());
  const auto receipt = buffer(64u, false, false, true);
  const auto matte = texture(width, height, DXGI_FORMAT_R32_UINT, true);
  const auto alpha = texture(width, height, DXGI_FORMAT_R32_UINT, true);

  for (const UINT scale : {2u, 4u}) {
    SCOPED_TRACE(scale);
    const UINT cells_w = width / scale, cells_h = height / scale;
    // One coarse column is left unowned but retains a high, finite scratch value.
    const std::array<UINT, 52u> placement_words {
      2u, 3u, 0u, 0u,
      0u, 0u, 0u, 0u, 5u, cells_h, scale, 144u,
      0u, 0u, 6u, 0u, cells_w, cells_h, 0u, 144u
    };
    auto partial_words = placement_words;
    partial_words[1u] = 1u;
    const auto placement = buffer(placement_words.size(), false, false, false, placement_words.data());
    const auto partial = buffer(partial_words.size(), false, false, false, partial_words.data());

    const std::array<UINT, 36u> work_words {2u, 1u, 0u, scale, 2u * scale, scale, 3u * scale, 2u * scale};
    const std::array<UINT, 36u> empty_words {1u};
    const std::array<UINT, 36u> fringe_words {1u, 1u, token_low, token_high, 2u * scale, 0u, 6u * scale, 3u * scale};
    auto stale_words = fringe_words;
    --stale_words[3u];
    const auto fringe = buffer(fringe_words.size(), true, false, false, fringe_words.data());
    const auto stale = buffer(stale_words.size(), true, false, false, stale_words.data());

    std::vector<UINT> score_words(width * height, 0xffffffffu);
    for (UINT y = 0u; y < cells_h; ++y) {
      for (UINT x = 0u; x < cells_w; ++x) {
        const UINT q = x == 1u || x == 2u ? 63u : x == 3u || x == 4u ? 64u : 128u;
        score_words[y * width + x] = std::bit_cast<UINT>(float(q) / 255.0f);
      }
    }
    const auto candidate = texture(width, height, DXGI_FORMAT_R32_UINT, false, score_words.data(), width * 4u);

    const auto publish = [&](UINT flags, const auto &work_record,
                             ID3D11ShaderResourceView *fringe_bounds, ID3D11ShaderResourceView *completed) {
      const auto work = buffer(work_record.size(), true, false, false, work_record.data());
      const auto request = constants(std::array<UINT, 12u> {width, height, 38u, flags, 0u, 0u, 0u, 0u,
                                                           0u, 0u, token_low, token_high});
      ID3D11Buffer *const cbuffers[] {request.Get(), contour_constants.Get()};
      std::array<ID3D11ShaderResourceView *, 24u> in {};
      in[7u] = work.srv.Get();
      in[13u] = current_sct.srv.Get();
      in[15u] = candidate.srv.Get();
      in[22u] = fringe_bounds;
      in[23u] = completed;
      ID3D11UnorderedAccessView *const out[] {matte.uav.Get(), nullptr, nullptr, alpha.uav.Get(), nullptr, receipt.uav.Get()};
      context->CSSetShader(shader.Get(), nullptr, 0u);
      context->CSSetConstantBuffers(0u, 2u, cbuffers);
      context->CSSetShaderResources(0u, in.size(), in.data());
      context->CSSetUnorderedAccessViews(0u, std::size(out), out, nullptr);
      context->Dispatch(width / 16u, height / 8u, 1u);
      context->ClearState();
      return std::array {words(read(alpha.texture.Get())), words(read(matte.texture.Get())), words(read(receipt.buffer.Get()))};
    };

    const auto base = publish(0u, work_words, fringe.srv.Get(), placement.srv.Get());
    const auto accepted = publish(fringe_flag, work_words, fringe.srv.Get(), placement.srv.Get());
    ASSERT_EQ(base[0].size(), width * height);
    ASSERT_EQ(accepted[0].size(), width * height);
    ASSERT_EQ(accepted[1].size(), width * height);
    ASSERT_EQ(accepted[2].size(), 64u);
    EXPECT_EQ(accepted[2][2u], 4u);
    EXPECT_EQ(accepted[2][3u], 0u);
    EXPECT_EQ(accepted[2][60u], token_low);
    EXPECT_EQ(accepted[2][61u], token_high);
    const auto pixel = [](UINT x, UINT y) { return y * width + x; };
    EXPECT_EQ(base[0][pixel(2u * scale, scale)], 63u);
    EXPECT_EQ(accepted[0][pixel(2u * scale, 0u)], 0u);
    EXPECT_EQ(accepted[0][pixel(4u * scale - 1u, 0u)], 64u);
    EXPECT_EQ(accepted[0][pixel(5u * scale, 0u)], 0u) << "Unowned bilinear tap";
    EXPECT_EQ(accepted[0][pixel(7u * scale, 0u)], 0u) << "Owned high score outside authorized bounds";
    EXPECT_EQ(accepted[0][pixel(4u * scale - 1u, 3u * scale)], 0u) << "Half-open bottom edge";
    EXPECT_EQ(accepted[2], base[2]);
    for (UINT y = 0u; y < height; ++y) {
      for (UINT x = 0u; x < width; ++x) {
        const auto i = pixel(x, y);
        EXPECT_EQ(accepted[1][i], accepted[0][i] >= 38u ? 1u : 0u) << x << ',' << y;
        if (x < 2u * scale || x >= 6u * scale || y >= 3u * scale ||
            (x < 3u * scale && y >= scale && y < 2u * scale)) {
          EXPECT_EQ(accepted[0][i], base[0][i]) << x << ',' << y;
        }
      }
    }

    EXPECT_EQ(publish(fringe_flag, work_words, stale.srv.Get(), placement.srv.Get()), base);
    EXPECT_EQ(publish(fringe_flag, work_words, fringe.srv.Get(), partial.srv.Get()), base);
    EXPECT_EQ(publish(fringe_flag, work_words, fringe.srv.Get(), nullptr), base);
    const auto empty = publish(fringe_flag, empty_words, fringe.srv.Get(), placement.srv.Get());
    const std::vector<UINT> zeros(width * height, 0u);
    EXPECT_EQ(empty[0], zeros);
    EXPECT_EQ(empty[1], zeros);
    EXPECT_EQ(empty[2], base[2]);
  }
}

TEST_F(HostSbsDirectGlyphPublishGpuTest, NativeHistoryRequiresOneMatchingCueInsideTheCurrentDetectorCap) {
  constexpr UINT width = 96, height = 48;
  const auto &specs = cache::direct_glyph_palette_specs;
  const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, specs);
  ASSERT_TRUE(sources);
  ASSERT_EQ(cache::source_closure_sha256(sources), cache::direct_glyph_palette_source_closure_sha256);
  const auto load = [&](const cache::shader_spec &spec) {
    const auto code = cache::get(sources, spec);
    need(bool(code), "Missing glyph history shader");
    ComPtr<ID3D11ComputeShader> shader;
    hr(device->CreateComputeShader(code->data(), code->size(), nullptr, &shader));
    return shader;
  };
  const auto begin = load(cache::host_sbs_direct_glyph_begin_history);
  const auto confirm = load(cache::host_sbs_direct_glyph_confirm_history);
  const auto finish = load(cache::host_sbs_direct_glyph_finish_windows);
  const auto confirm_mask = load(cache::host_sbs_direct_glyph_confirm_mask_history);
  const auto scatter = load(cache::host_sbs_direct_glyph_publish_line);
  const std::array<UINT, 4> old_box {10, 8, 82, 40}, current_box {16, 10, 80, 38}, cap_box {12, 6, 84, 42};
  std::array<UINT, 8 * 16> old {};
  const std::array<UINT, 2> ink_bins {0u, 4912u}; // Black fill and white outline.
  old[15] = ink_bins[1] << 13;
  const auto palette = structured(2, sizeof(UINT), ink_bins.data());
  for (UINT at : {0u, 4u, 8u}) {
    std::copy(old_box.begin(), old_box.end(), old.begin() + at);
  }
  const auto prior_lines = structured(8, 64, old.data());
  const auto current = structured(8, 64, nullptr);
  const auto sizes = structured(17, 2 * sizeof(UINT), nullptr);
  using opacity_t = std::array<float, 2>; // Raw model alpha, then published alpha.
  const std::vector<opacity_t> past(width * height, {1.0f, 1.0f});
  const auto prior_alpha = structured(width * height, sizeof(opacity_t), past.data());
  auto left_ink = past;
  const UINT old_width = old_box[2] - old_box[0];
  for (UINT i = 0; i < left_ink.size(); ++i) {
    if (old_box[0] + i % old_width >= width / 2) left_ink[i] = {0.0f, 0.0f};
  }
  const auto prior_left = structured(width * height, sizeof(opacity_t), left_ink.data());
  const std::vector<float> clear(width * height, 0.0f), solid(width * height, 1.0f);
  const auto new_alpha = structured(width * height, sizeof(float), clear.data());
  const auto solid_alpha = structured(width * height, sizeof(float), solid.data());
  const auto remembered = structured(width * height, sizeof(opacity_t), past.data());
  const auto output = texture(width, height, DXGI_FORMAT_R32_UINT, true);
  std::vector<std::uint16_t> a(width * height * 4u), b(a.size());
  for (UINT i = 0; i < width * height; ++i) {
    const bool white = ((i % width) / 4u) % 2u == 0u;
    for (UINT c = 0; c < 3; ++c) {
      a[4 * i + c] = white ? 0x3c00u : 0u;
      b[4 * i + c] = white ? 0u : 0x3c00u;
    }
  }
  const auto previous_rgb = texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, false, a.data(), width * 8);
  const auto swapped_rgb = texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, false, b.data(), width * 8);
  for (UINT y = 0; y < height; ++y) {
    std::copy_n(a.begin() + 4u * y * width, 2u * width, b.begin() + 4u * y * width);
  }
  const auto appended_rgb = texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, false, b.data(), width * 8);
  const auto exercise = [&](UINT count, UINT cap_token, bool swap, UINT center_twice = 0, bool append = false, UINT has_previous = 1, bool held = false) {
    const auto cb = constants(std::array<UINT, 8> {width, height, 17, 0, 1, has_previous, 0, center_twice});
    std::array<UINT, 36> work {count ? 2u : 1u, count}, caps {1u, count, cap_token, 0u};
    if (held) work[0] = 3u;
    for (UINT slot = 0; slot < count; ++slot) {
      std::copy(current_box.begin(), current_box.end(), work.begin() + 4u + 4u * slot);
      std::copy(cap_box.begin(), cap_box.end(), caps.begin() + 4u + 4u * slot);
    }
    const auto current_work = buffer(work.size(), true, false, true, work.data());
    const auto detector = buffer(caps.size(), true, false, false, caps.data());
    auto *rgb = append ? appended_rgb.srv.Get() : (swap ? swapped_rgb.srv.Get() : previous_rgb.srv.Get());
    auto *old_mask = append ? prior_left.srv.Get() : prior_alpha.srv.Get();
    auto *current_mask = append ? solid_alpha.srv.Get() : new_alpha.srv.Get();
    auto pass = [&](ID3D11ComputeShader *shader, UINT gx, UINT gy, std::array<ID3D11ShaderResourceView *, 8> in, std::array<ID3D11UnorderedAccessView *, 5> out) {
      context->CSSetConstantBuffers(0, 1, cb.GetAddressOf());
      context->CSSetShader(shader, nullptr, 0);
      context->CSSetShaderResources(0, in.size(), in.data());
      context->CSSetUnorderedAccessViews(0, out.size(), out.data(), nullptr);
      context->Dispatch(gx, gy, 1);
      unbind(context.Get());
    };
    const std::array<UINT, 4> zero {};
    context->ClearUnorderedAccessViewUint(output.uav.Get(), zero.data());
    pass(begin.Get(), 1, 1, {detector.srv.Get(), prior_lines.srv.Get()}, {current_work.uav.Get(), current.uav.Get(), sizes.uav.Get()});
    pass(confirm.Get(), 8, 1, {nullptr, prior_lines.srv.Get(), previous_rgb.srv.Get(), rgb, old_mask}, {nullptr, current.uav.Get()});
    pass(finish.Get(), 1, 1, {}, {current_work.uav.Get(), current.uav.Get(), sizes.uav.Get()});
    pass(confirm_mask.Get(), 1, 1, {nullptr, prior_lines.srv.Get(), previous_rgb.srv.Get(), rgb, old_mask, current_mask, nullptr, palette.srv.Get()}, {nullptr, current.uav.Get()});
    pass(scatter.Get(), 6, 6, {nullptr, prior_lines.srv.Get(), previous_rgb.srv.Get(), rgb, old_mask, current_mask, current.srv.Get()}, {current_work.uav.Get(), nullptr, nullptr, remembered.uav.Get(), output.uav.Get()});
    return std::array {words(read(current_work.buffer.Get())), words(read(current.buffer.Get())), words(read(output.texture.Get()))};
  };
  const auto same = exercise(1, 17, false);
  EXPECT_EQ((std::vector<UINT>(same[0].begin() + 4, same[0].begin() + 8)), (std::vector<UINT> {12, 6, 82, 42}));
  EXPECT_EQ(same[1][14], 1u);
  EXPECT_EQ(same[2][24u * width + 35u], 242u);
  EXPECT_EQ(same[2][24u * width + 11u], 0u);
  const auto stored = words(read(remembered.buffer.Get()));
  const UINT pixel = (24u - cap_box[1]) * 70u + (35u - 12u);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(stored[2u * pixel]), 0.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(stored[2u * pixel + 1u]), .95f);
  const auto onset = exercise(1, 17, false, 0, false, 0);
  EXPECT_EQ(onset[0][5], cap_box[1]);
  EXPECT_EQ(onset[0][7], cap_box[3]);
  const auto schedule = words(read(sizes.buffer.Get()));
  EXPECT_EQ(schedule[1], cap_box[3] - cap_box[1]);
  EXPECT_EQ(schedule[18], 32u); // The expanded OCR work height still sets the sampling scale.
  EXPECT_EQ(onset[1][14], 0u);
  EXPECT_EQ(onset[1][15], ink_bins[1] << 13);
  const auto swap = exercise(1, 17, true);
  EXPECT_EQ((std::vector<UINT>(swap[0].begin() + 4, swap[0].begin() + 8)), (std::vector<UINT> {14, 6, 82, 42}));
  EXPECT_EQ(swap[1][14], 0u);
  EXPECT_EQ(swap[2][24u * width + 35u], 0u);
  // The previous strokes and RGB agree on the left. New strokes on the right must veto the blend.
  const auto append = exercise(1, 17, false, 0, true);
  EXPECT_EQ(append[0], same[0]);
  EXPECT_EQ(append[1][14], 0u);
  for (UINT y : {cap_box[1], 24u, cap_box[3] - 1u}) {
    EXPECT_EQ(append[2][y * width + 65u], 255u);
  }
  // Swapped RGB prevents history from extending the current box.
  const auto left = exercise(1, 17, true, 90);
  EXPECT_EQ((std::vector<UINT>(left[0].begin() + 4, left[0].begin() + 8)), (std::vector<UINT> {12, 6, 82, 42}));
  const auto right = exercise(1, 17, true, 102);
  EXPECT_EQ((std::vector<UINT>(right[0].begin() + 4, right[0].begin() + 8)), (std::vector<UINT> {14, 6, 84, 42}));
  EXPECT_EQ(exercise(1, 17, true, 20)[0], swap[0]);
  const auto ambiguous = exercise(2, 17, false);
  EXPECT_EQ(ambiguous[1][12], 0u);
  EXPECT_EQ(ambiguous[1][16 + 12], 0u);
  EXPECT_EQ(exercise(0, 17, false)[0][0], 1u);
  EXPECT_EQ(words(read(sizes.buffer.Get()))[16], 0u);
  EXPECT_EQ(exercise(0, 17, false, 0, false, 1, true)[0][0], 3u);
  EXPECT_EQ(words(read(sizes.buffer.Get()))[16], 1u);
  const auto stale = exercise(1, 16, false);
  EXPECT_EQ(stale[0][0], 0u);
  EXPECT_EQ(words(read(sizes.buffer.Get()))[16], 0u);
  EXPECT_NE(stale[0][2], 0u);
  EXPECT_EQ(stale[1][14], 0u);
}

TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, EpochAndEmptyAreSourceFactsNotUiAuthority) {
 author();queue();scts[2]=2;upload();queue();EXPECT_EQ(words(readView(owner.views().source_proof.Get()))[3],0u);snapshot("bad-SCT2","join_reject");
 scts[2]=0;scts[3]+=1;upload();queue();check_outputs();prefix();EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43],0u);auto row=snapshot("epoch-source-only","source_ready_envelope_reject");row["envelope"]=save("epoch-source-only-envelope.bin",readView(owner.views().uip_envelope.Get()));report["observations"].back()=row;
 uips[20]=scts[3];upload();queue();prefix();EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43],1u);
 uips[3]=256;upload();queue();prefix();EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43],0u);
}
} // namespace
#endif
