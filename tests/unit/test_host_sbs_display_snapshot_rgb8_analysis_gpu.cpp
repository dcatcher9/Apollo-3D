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
#include <src/host_sbs_display_snapshot_rgb8_analysis_gpu.h>
#include <src/host_sbs_resolution.h>
#include <src/host_sbs_shader_cache.h>
#include <src/sbs_bench_raw_matte_gpu.h>
namespace {
namespace cg=models::display_snapshot_rgb8_analysis_gpu_v1;
namespace cache=models::host_sbs_shader_cache;
using Microsoft::WRL::ComPtr;using json=nlohmann::json;using bytes=std::vector<std::uint8_t>;
constexpr const char *helper_pin=subtitle_test_source_pins::display_snapshot_rgb8;
constexpr const char *producer_pin="59ac71657fa3dbe9372da310ab1aaed3f3ff124145192ad8d7d682b162356de6";
constexpr const char *producer_freeze_pin="a00dae08ecb4f54a22f88edbb62f0dc686c037a9e8777aa801d66529625408a0";
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
 static std::uint16_t floor_half_code(UINT code){UINT lo=0,hi=0x3c01;while(lo+1<hi){UINT mid=(lo+hi)/2;if(half_value(std::uint16_t(mid))*255.0<=double(code))lo=mid;else hi=mid;}return std::uint16_t(lo);}
 static UINT code_of(std::uint16_t h){return UINT(std::floor(half_value(h)*255.0+0.5));}
 void SetUp()override {
  const char *sr=std::getenv("SUNSHINE_TEST_DISPLAY_RGB8_SHADER_ROOT"),*out=std::getenv("SUNSHINE_TEST_DISPLAY_RGB8_OUTPUT"),*dr=std::getenv("SUNSHINE_TEST_DISPLAY_RGB8_DRIVER");
  if(!sr||!out||!dr)GTEST_SKIP()<<"Explicit private recipe5 source/output/driver required";
  need(std::string(dr)=="warp"||std::string(dr)=="hardware","Unknown driver");
  D3D_FEATURE_LEVEL level{},wanted[]{D3D_FEATURE_LEVEL_11_0};
  hr(D3D11CreateDevice(nullptr,std::string(dr)=="hardware"?D3D_DRIVER_TYPE_HARDWARE:D3D_DRIVER_TYPE_WARP,nullptr,0,wanted,1,D3D11_SDK_VERSION,&device,&level,&context));need(level==D3D_FEATURE_LEVEL_11_0,"Feature level");
  ComPtr<IDXGIDevice> xd;ComPtr<IDXGIAdapter> adapter;hr(device.As(&xd));hr(xd->GetAdapter(&adapter));DXGI_ADAPTER_DESC desc{};hr(adapter->GetDesc(&desc));
  const std::array specs{
   cache::shader_spec{"host_sbs_display_snapshot_rgb8_analysis_cs.hlsl","prepare_main","cs_5_0"},
   cache::shader_spec{"host_sbs_display_snapshot_rgb8_analysis_cs.hlsl","copy_main","cs_5_0"},
   cache::shader_spec{"host_sbs_display_snapshot_rgb8_analysis_cs.hlsl","finalize_main","cs_5_0"},
   cache::shader_spec{"host_sbs_display_snapshot_rgb8_analysis_cs.hlsl","envelope_main","cs_5_0"}};
  auto cs=cache::snapshot_sources(sr,specs);need(cache::source_closure_sha256(cs)==producer_pin,"Producer source drift");
  for(std::size_t i=0;i<4;++i){auto b=cache::get(cs,specs[i]);need(b&&!b->empty(),"No source bytecode");hr(device->CreateComputeShader(b->data(),b->size(),nullptr,&shaders[i]));}
  program={shaders[0].Get(),shaders[1].Get(),shaders[2].Get(),shaders[3].Get()};
  directory=std::filesystem::path(out)/::testing::UnitTest::GetInstance()->current_test_info()->name();
  need(!std::filesystem::exists(directory),"Refuse output overwrite");std::filesystem::create_directories(directory);
  report={{"schema",1},{"driver",dr},{"helper_source_sha256",helper_pin},{"producer_source_closure_sha256",producer_pin},{"producer_freeze_sha256",producer_freeze_pin},{"compile_flags",34816},{"scope","Synthetic source availability only; no live/capture/model/UI/EMPTY or HDR-quality proof"},{"adapter",{{"vendor_id",desc.VendorId},{"device_id",desc.DeviceId},{"luid_low",desc.AdapterLuid.LowPart},{"luid_high",desc.AdapterLuid.HighPart}}},{"artifacts",json::object()},{"observations",json::array()}};
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

 static bytes half_bytes(const std::vector<std::uint16_t>&v){bytes b(v.size()*2);std::memcpy(b.data(),v.data(),b.size());return b;}
 static std::vector<std::uint16_t> half_words(const bytes&b){need(b.size()%2==0,"Half bytes");std::vector<std::uint16_t> v(b.size()/2);std::memcpy(v.data(),b.data(),b.size());return v;}
 bytes reference_band(){auto v=owner.views();ComPtr<ID3D11Resource>r;v.reference_rgb8->GetResource(&r);auto t=texture(W,B,DXGI_FORMAT_R8G8B8A8_UNORM,false);D3D11_BOX box{0,crop,0,W,H,1};context->CopySubresourceRegion(t.texture.Get(),0,0,0,0,r.Get(),0,&box);return read(t.texture.Get());}
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
  need(owner.ensure(device.Get(),W,H,crop,B,error),"Ensure recipe5");
  request={display.srv.Get(),raw.srv.Get(),contour.Get(),sct.srv.Get(),root.buffer.Get(),{left,top,left+W,top+H},frame,obs,gen,token,transfer};
 }
 void queue(){need(owner.queue(context.Get(),program,request,error),error.c_str());}
 void envelope(){need(owner.queue_uip_envelope(context.Get(),program,uip.srv.Get(),request,error),error.c_str());}
 std::array<UINT,64> expected_proof()const {
  std::array<UINT,64> p{};p[0]=cg::tag;p[1]=1;p[2]=5;p[3]=1;std::copy_n(cb.begin()+12,2,p.begin()+4);std::copy_n(cb.begin()+16,2,p.begin()+6);std::copy_n(cb.begin()+14,2,p.begin()+8);p[10]=UINT(request.token);p[11]=UINT(request.token>>32);std::copy_n(cb.begin(),4,p.begin()+12);std::copy(request.roi_ltrb.begin(),request.roi_ltrb.end(),p.begin()+16);
  p[20]=rawW;p[21]=rawH;p[22]=p[23]=UINT(rawFormat);p[24]=8;p[25]=4;p[26]=p[27]=p[28]=1;p[30]=request.transfer_mode;p[31]=63;p[32]=p[33]=p[35]=W*B;p[36]=p[37]=((W+15)/16)*((B+15)/16);p[39]=scts[3];std::copy_n(cb.begin()+4,4,p.begin()+40);std::copy_n(cb.begin()+8,4,p.begin()+44);std::copy_n(cb.begin()+20,4,p.begin()+48);std::copy_n(cb.begin()+16,4,p.begin()+52);std::copy_n(roots.begin()+60,4,p.begin()+56);p[60]=10;p[61]=28;p[62]=10;p[63]=1;return p;
 }
 void check_outputs(){auto got=reference_band();auto b=half_words(readView(owner.views().analysis_band.Get(),8));need(got.size()==std::size_t(W)*B*4&&b.size()==pixels.size(),"Band sizes");for(std::size_t i=0;i<pixels.size();i+=4){for(UINT c=0;c<3;++c){auto k=code_of(pixels[i+c]);EXPECT_EQ(got[i+c],k)<<i<<":"<<c;EXPECT_EQ(b[i+c],floor_half_code(k))<<i<<":"<<c;}EXPECT_EQ(got[i+3],255);EXPECT_EQ(b[i+3],0x3c00);}EXPECT_EQ(readView(owner.views().source_proof.Get()),pack(expected_proof()));}
 void prefix(){auto e=words(readView(owner.views().uip_envelope.Get()));need(e.size()==104,"Envelope104");EXPECT_TRUE(std::equal(uips.begin(),uips.end(),e.begin()));}
 json snapshot(const std::string&name,const std::string&kind){
  json row{{"name",name},{"kind",kind},{"roi_wh",{W,H}},{"crop_top",crop},{"band_h",B},{"raw_wh",{rawW,rawH}},{"roi_ltrb",request.roi_ltrb},{"transfer_mode",request.transfer_mode},{"raw_format",UINT(rawFormat)},{"frame",request.frame},{"observation_us",request.observation_us},{"generation",request.analysis_generation},{"token",request.token},{"synthetic_tuple",true},{"source_availability_only",true},{"upper_carrier_read",false}};
  for(auto [key,value]:std::initializer_list<std::pair<std::string,bytes>>{{"display",read(display.texture.Get(),8)},{"reference_band",reference_band()},{"analysis_band",readView(owner.views().analysis_band.Get(),8)},{"proof",readView(owner.views().source_proof.Get())},{"cb",pack(cb)},{"sct",pack(scts)},{"root",pack(roots)},{"uip",pack(uips)}})row[key]=save(name+"-"+key+".bin",value);
  report["observations"].push_back(row);return row;
 }
 void positive(const std::string&name){auto before=read(display.texture.Get(),8);queue();check_outputs();envelope();prefix();EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43],1u);EXPECT_EQ(read(display.texture.Get(),8),before);auto row=snapshot(name,"positive");row["envelope"]=save(name+"-envelope.bin",readView(owner.views().uip_envelope.Get()));report["observations"].back()=row;}
};
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, AllCodesBothRootModesAllTransfersAndEmptySource) {
 for(UINT transfer=0;transfer<3;++transfer)for(bool direct:{false,true})for(bool empty:{false,true}){author(transfer,direct,empty);positive("m"+std::to_string(transfer)+(direct?"-direct":"-queued")+(empty?"-empty":"-active"));}
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, DirectAndQueuedSourceOutputsIdentical) {
 author();queue();envelope();auto r=reference_band(),b=readView(owner.views().analysis_band.Get(),8),p=readView(owner.views().source_proof.Get()),e=readView(owner.views().uip_envelope.Get());roots[20]=roots[21]=roots[22]=0;upload();queue();envelope();EXPECT_EQ(reference_band(),r);EXPECT_EQ(readView(owner.views().analysis_band.Get(),8),b);EXPECT_EQ(readView(owner.views().source_proof.Get()),p);EXPECT_EQ(readView(owner.views().uip_envelope.Get()),e);auto row=snapshot("direct-identical","positive");row["compared_queued_reference"]=save("queued-reference.bin",r);row["compared_queued_analysis"]=save("queued-analysis.bin",b);row["compared_queued_proof"]=save("queued-proof.bin",p);row["compared_queued_envelope"]=save("queued-envelope.bin",e);row["envelope"]=save("direct-envelope.bin",readView(owner.views().uip_envelope.Get()));report["observations"].back()=row;
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, CombinedCurrentQueueMatchesSeparateCaptureAndEnvelope) {
 author();
 cg::resources_t combined;
 ASSERT_TRUE(combined.ensure(device.Get(),W,H,crop,B,error))<<error;
 // Source5's upper carrier rows are intentionally unwritten. Compare only the
 // actual native band, plus the whole proof/envelope, on independently owned outputs.
 const auto capture=[&](cg::resources_t &source){
  const auto v=source.views();ComPtr<ID3D11Resource>r;v.reference_rgb8->GetResource(&r);
  auto band=texture(W,B,DXGI_FORMAT_R8G8B8A8_UNORM,false);
  const D3D11_BOX box{0u,crop,0u,W,H,1u};
  context->CopySubresourceRegion(band.texture.Get(),0u,0u,0u,0u,r.Get(),0u,&box);
  return std::array<bytes,4>{read(band.texture.Get()),readView(v.analysis_band.Get(),8),
    readView(v.source_proof.Get()),readView(v.uip_envelope.Get())};
 };
 const std::array<const char *,4> names{"reference-band","analysis-band","proof","envelope"};
 report["combined_queue_comparisons"]=json::array();
 const auto compare=[&](const std::string &label,const std::array<bytes,4> &expected,const std::array<bytes,4> &actual){
  json row{{"name",label},{"expected",json::object()},{"actual",json::object()}};
  for(UINT i=0u;i<names.size();++i){
   row["expected"][names[i]]=save(label+"-separate-"+names[i]+".bin",expected[i]);
   row["actual"][names[i]]=save(label+"-combined-"+names[i]+".bin",actual[i]);
   EXPECT_EQ(actual[i],expected[i])<<label<<":"<<names[i];
  }
  report["combined_queue_comparisons"].push_back(row);
 };
 // Unrelated caller state must survive both successful and rejected atomic calls.
 std::array<buf_t,16> state_srvs;std::array<buf_t,8> state_uavs;
 std::array<ComPtr<ID3D11Buffer>,3> state_cbs;
 std::array<ID3D11ShaderResourceView*,16> ss{};
 std::array<ID3D11UnorderedAccessView*,8> uu{};
 std::array<ID3D11Buffer*,3> bb{};
 for(UINT i=0u;i<ss.size();++i){state_srvs[i]=buffer(4);ss[i]=state_srvs[i].srv.Get();}
 for(UINT i=0u;i<uu.size();++i){state_uavs[i]=buffer(4,false,false,true);uu[i]=state_uavs[i].uav.Get();}
 for(UINT i=0u;i<bb.size();++i){state_cbs[i]=constants(std::array<UINT,4>{0x61700000u+i,0u,0u,0u});bb[i]=state_cbs[i].Get();}
 const auto bind_state=[&](){
  context->CSSetShaderResources(0u,16u,ss.data());context->CSSetUnorderedAccessViews(0u,8u,uu.data(),nullptr);
  context->CSSetConstantBuffers(0u,3u,bb.data());context->CSSetShader(shaders[1].Get(),nullptr,0u);
 };
 const auto check_state=[&](){
  std::array<ID3D11ShaderResourceView*,16> s{};std::array<ID3D11UnorderedAccessView*,8> u{};std::array<ID3D11Buffer*,3> b{};
  context->CSGetShaderResources(0u,16u,s.data());context->CSGetUnorderedAccessViews(0u,8u,u.data());context->CSGetConstantBuffers(0u,3u,b.data());
  for(UINT i=0u;i<s.size();++i){EXPECT_EQ(s[i],ss[i]);if(s[i])s[i]->Release();}
  for(UINT i=0u;i<u.size();++i){EXPECT_EQ(u[i],uu[i]);if(u[i])u[i]->Release();}
  for(UINT i=0u;i<b.size();++i){EXPECT_EQ(b[i],bb[i]);if(b[i])b[i]->Release();}
  ComPtr<ID3D11ComputeShader> shader;UINT instances=0u;context->CSGetShader(&shader,nullptr,&instances);
  EXPECT_EQ(shader.Get(),shaders[1].Get());EXPECT_EQ(instances,0u);
 };
 queue();envelope();check_outputs();prefix();
 const auto first=capture(owner);
 bind_state();ASSERT_TRUE(combined.queue_current(context.Get(),program,request,uip.srv.Get(),error))<<error;check_state();
 compare("sdr-queued-active",first,capture(combined));
 EXPECT_EQ(read(combined.views().final_root_mirror.Get()),pack(roots));
 EXPECT_EQ(read(combined.views().request_constants.Get()),read(owner.views().request_constants.Get()));

 // A different real input tuple makes accidental capture-before-UIP-validation
 // observable in both pixels and proof. An invalid descriptor must submit nothing.
 author(2u,true,true,0x1000000a9ull,0xe00000000000a901ull);
 auto short_uip=buffer(39u);
 for(auto *bad_uip:std::array<ID3D11ShaderResourceView*,2>{nullptr,short_uip.srv.Get()}){
  bind_state();EXPECT_FALSE(combined.queue_current(context.Get(),program,request,bad_uip,error));check_state();
  const auto held=capture(combined);
  for(UINT i=0u;i<first.size();++i)EXPECT_EQ(held[i],first[i])<<"early UIP rejection:"<<names[i];
 }
 queue();envelope();check_outputs();prefix();
 const auto second=capture(owner);
 ASSERT_NE(second[0u],first[0u]);ASSERT_NE(second[2u],first[2u]);
 bind_state();ASSERT_TRUE(combined.queue_current(context.Get(),program,request,uip.srv.Get(),error))<<error;check_state();
 compare("hdr-metadata-direct-empty",second,capture(combined));

 D3D11_QUERY_DESC desc{};desc.Query=D3D11_QUERY_OCCLUSION_PREDICATE;
 ComPtr<ID3D11Predicate> predicate;hr(device->CreatePredicate(&desc,&predicate));
 bind_state();context->SetPredication(predicate.Get(),FALSE);
 EXPECT_FALSE(combined.queue_current(context.Get(),program,request,uip.srv.Get(),error));check_state();
 ComPtr<ID3D11Predicate> actual_predicate;BOOL predicate_value=TRUE;
 context->GetPredication(&actual_predicate,&predicate_value);
 EXPECT_EQ(actual_predicate.Get(),predicate.Get());EXPECT_FALSE(predicate_value);
 context->SetPredication(nullptr,FALSE);
 const auto after_rejection=capture(combined);
 for(UINT i=0u;i<second.size();++i)EXPECT_EQ(after_rejection[i],second[i]);

 roots[60u]=0u;upload();
 bind_state();ASSERT_TRUE(combined.queue_current(context.Get(),program,request,uip.srv.Get(),error))<<error;check_state();
 compare("root0-held",second,capture(combined));
 report["combined_CS_state_restored"]={{"SRV",16},{"UAV",8},{"CB",3},{"shader",true},{"predication",true}};
 report["combined_early_UIP_rejections"]=2u;
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, MalformedRoot80ShapesRevokeSourceOnly) {
 author();queue();auto good=roots;UINT n=0;for(auto g:{std::array<UINT,4>{1,0,1,0},std::array<UINT,4>{0,1,1,0},std::array<UINT,4>{1,1,0,0},std::array<UINT,4>{1,1,1,1},std::array<UINT,4>{2,1,1,0},std::array<UINT,4>{0,0,0,1}}){roots=good;std::copy(g.begin(),g.end(),roots.begin()+20);upload();queue();auto p=words(readView(owner.views().source_proof.Get()));EXPECT_EQ(p[0],0u);EXPECT_EQ(p[3],0u);EXPECT_EQ(p[33],0u);EXPECT_EQ(p[34],0u);EXPECT_EQ(p[35],0u);EXPECT_EQ(p[38],3u);snapshot("bad80-"+std::to_string(n++),"join_reject");}
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, InvalidCellsWriteZeroBothAndRevokeFreshReadiness) {
 const std::array<std::array<std::uint16_t,4>,8> bad{{{0,0,0,0},{0,0,0,0x3800},{0x7e00,0,0,0x3c00},{0x7c00,0,0,0x3c00},{0xfc00,0,0,0x3c00},{0x8001,0,0,0x3c00},{0x3c01,0,0,0x3c00},{0,0,0,0xbc00}}};
 for(UINT k=0;k<bad.size();++k){author(2);queue();auto at=(std::size_t(2)*W+3)*4;std::copy(bad[k].begin(),bad[k].end(),pixels.begin()+at);context->UpdateSubresource(display.texture.Get(),0,nullptr,pixels.data(),W*8,0);queue();auto p=words(readView(owner.views().source_proof.Get()));EXPECT_EQ(p[0],0u);EXPECT_EQ(p[3],0u);EXPECT_EQ(p[33],W*B);EXPECT_EQ(p[34],1u);EXPECT_EQ(p[35],W*B-1);EXPECT_EQ(p[38],4u);auto r=reference_band();auto b=half_words(readView(owner.views().analysis_band.Get(),8));for(UINT c=0;c<4;++c){EXPECT_EQ(r[at+c],0);EXPECT_EQ(b[at+c],0);}envelope();prefix();EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43],0u);auto row=snapshot("invalid-"+std::to_string(k),"invalid");row["invalid_flat_pixel"]=2*W+3;report["observations"].back()=row;}
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, RootZeroHoldsPublishedTupleThenFreshTokenWrites) {
 author();queue();envelope();auto r=reference_band(),b=readView(owner.views().analysis_band.Get(),8),p=readView(owner.views().source_proof.Get()),e=readView(owner.views().uip_envelope.Get());auto good=roots;UINT n=0;
 for(auto g:{std::array<UINT,4>{0,1,1,0},std::array<UINT,4>{0,0,0,0},std::array<UINT,4>{1,0,1,0},std::array<UINT,4>{1,1,0,0}}){author(0,false,false,0x100000020ull+n,0xe000000000002000ull+n);std::copy(g.begin(),g.end(),roots.begin()+60);upload();queue();envelope();EXPECT_EQ(reference_band(),r);EXPECT_EQ(readView(owner.views().analysis_band.Get(),8),b);EXPECT_EQ(readView(owner.views().source_proof.Get()),p);EXPECT_EQ(readView(owner.views().uip_envelope.Get()),e);auto row=snapshot("root0-"+std::to_string(n++),"root0_hold");row["envelope"]=save(row["name"].get<std::string>()+"-envelope.bin",readView(owner.views().uip_envelope.Get()));row["held_reference"]=save(row["name"].get<std::string>()+"-held-reference.bin",r);row["held_analysis"]=save(row["name"].get<std::string>()+"-held-analysis.bin",b);row["held_proof"]=save(row["name"].get<std::string>()+"-held-proof.bin",p);row["held_envelope"]=save(row["name"].get<std::string>()+"-held-envelope.bin",e);report["observations"].back()=row;}
 author(1,false,false,0x100000099ull,0xe000000000009999ull);positive("fresh-token");EXPECT_NE(readView(owner.views().source_proof.Get()),p);EXPECT_NE(reference_band(),r);
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, RawMatteEmptyGatePreservesSourceAndPublishesCurrentZero) {
  namespace model = sbs_bench::raw_matte_gpu;
  namespace v2 = models::depth_coordinate_v2;
  model::resources_t matte;
  ASSERT_TRUE(matte.load(device.Get(), SUNSHINE_SHADERS_DIR, error, true)) << error;
  H = 65u;
  B = models::subtitle_ocr_source_crop_height(W, H);
  crop = H - B;
  constexpr UINT field_w = 770u, field_h = 434u;
  const auto start = v2::subtitle_ocr_project_row_ceil(W, H, field_w, field_h, v2::subtitle_ocr_safe_row_top);
  const auto end = v2::subtitle_ocr_project_row_ceil(W, H, field_w, field_h, v2::subtitle_ocr_safe_row_bottom);
  ASSERT_TRUE(start && end && start.value < end.value);
  buf_t ocr, condition;
  model::early_request_t input;
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
  const auto source_contents = [&] {
    const auto views = owner.views();
    return std::array<bytes, 4> {reference_band(), readView(views.analysis_band.Get(), 8u),
      readView(views.source_proof.Get()), readView(views.uip_envelope.Get())};
  };
  const auto capture = [&](ID3D11Buffer *work) { return owner.queue_current(context.Get(), program, request, uip.srv.Get(), error, work); };

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
  ASSERT_TRUE(matte.queue_early(context.Get(), input, error)) << error;
  const auto current = matte.views();
  ASSERT_TRUE(current.matte && current.alpha8 && current.receipt && current.work);
  const auto receipt = words(readView(current.receipt.Get()));
  ASSERT_EQ(receipt.size(), 64u);
  EXPECT_EQ(receipt[2u], 4u);
  EXPECT_EQ(receipt[3u], 0u);
  EXPECT_EQ(receipt[60u], static_cast<UINT>(request.token));
  EXPECT_EQ(receipt[61u], static_cast<UINT>(request.token >> 32u));
  EXPECT_EQ(receipt[62u], 3u);
  EXPECT_TRUE(std::equal(cb.begin(), cb.end(), receipt.begin() + 4u));
  EXPECT_TRUE(std::equal(scts.begin(), scts.end(), receipt.begin() + 28u));
  EXPECT_EQ(readView(current.matte.Get()), bytes(std::size_t(W) * B * 4u, 0u));
  EXPECT_EQ(readView(current.alpha8.Get()), bytes(std::size_t(W) * B * 4u, 0u));
  const auto empty_work = words(read(current.work.Get()));
  ASSERT_EQ(empty_work.size(), 60u);
  EXPECT_EQ(empty_work[4u], 1u);
  EXPECT_EQ(empty_work[16u], 1u);
  EXPECT_EQ(empty_work[54u], 0u);
  EXPECT_EQ(empty_work[55u], 0u);
  EXPECT_EQ(empty_work[57u], 0u);
  EXPECT_EQ(empty_work[58u], 0u);
  EXPECT_EQ(empty_work[59u], 0u);

  author_current(false, 0x100000009ull);
  roots[60u] = 0u;
  upload();
  ASSERT_EQ(prepare(), work) << error;
  EXPECT_EQ(read(work), pack(roots));
  ASSERT_TRUE(capture(work)) << error;
  EXPECT_EQ(source_contents(), positive);
  ASSERT_TRUE(matte.queue_early(context.Get(), input, error)) << error;
  EXPECT_EQ(words(readView(matte.views().receipt.Get())), receipt);
  const auto held_work = words(read(matte.views().work.Get()));
  ASSERT_EQ(held_work.size(), 60u);
  EXPECT_EQ(held_work[16u], 3u);

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
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, LiteralUipPrefixProofEveryWordAndPendingLease) {
 author();queue();envelope();auto stable=readView(owner.views().uip_envelope.Get());auto good=words(readView(owner.views().source_proof.Get()));auto v=owner.views();ComPtr<ID3D11Resource>proof;v.source_proof->GetResource(&proof);
 for(UINT k=0;k<64;++k){auto bad=good;bad[k]^=1;context->UpdateSubresource(proof.Get(),0,nullptr,bad.data(),0,0);envelope();prefix();auto e=words(readView(v.uip_envelope.Get()));EXPECT_EQ(e[43],0u)<<k;json row{{"name","proof-"+std::to_string(k)},{"kind","proof_reject"},{"word",k}};row["supplied_proof"]=save(row["name"].get<std::string>()+"-input.bin",bytes(reinterpret_cast<const std::uint8_t*>(bad.data()),reinterpret_cast<const std::uint8_t*>(bad.data()+bad.size())));row["uip"]=save(row["name"].get<std::string>()+"-uip.bin",pack(uips));row["envelope"]=save(row["name"].get<std::string>()+"-envelope.bin",readView(v.uip_envelope.Get()));report["observations"].push_back(row);}
 context->UpdateSubresource(proof.Get(),0,nullptr,good.data(),0,0);envelope();EXPECT_EQ(readView(v.uip_envelope.Get()),stable);
 auto original=request;for(UINT k=0;k<6;++k){auto bad=original;if(k==0)++bad.frame;if(k==1)++bad.observation_us;if(k==2)++bad.analysis_generation;if(k==3)++bad.token;if(k==4){++bad.roi_ltrb[0];++bad.roi_ltrb[2];}if(k==5)bad.transfer_mode=1;EXPECT_FALSE(owner.queue_uip_envelope(context.Get(),program,uip.srv.Get(),bad,error));EXPECT_EQ(readView(v.uip_envelope.Get()),stable);}
 ComPtr<ID3D11ShaderResourceView>dc,rc;hr(device->CreateShaderResourceView(display.texture.Get(),nullptr,&dc));hr(device->CreateShaderResourceView(raw.texture.Get(),nullptr,&rc));auto cc=constants(cb);auto ss2=buffer(32,false,false,false,scts.data());auto rr2=buffer(64,true,true,false,roots.data());for(UINT k=0;k<5;++k){auto bad=original;if(k==0)bad.display_snapshot_band=dc.Get();if(k==1)bad.parent_raw_source=rc.Get();if(k==2)bad.actual_contour_constants96=cc.Get();if(k==3)bad.actual_sct32=ss2.srv.Get();if(k==4)bad.actual_final256=rr2.buffer.Get();EXPECT_FALSE(owner.queue_uip_envelope(context.Get(),program,uip.srv.Get(),bad,error));EXPECT_EQ(readView(v.uip_envelope.Get()),stable);}
 auto base=uips;for(UINT k:std::vector<UINT>{0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23}){uips=base;uips[k]^=1;context->UpdateSubresource(uip.buffer.Get(),0,nullptr,uips.data(),0,0);envelope();prefix();EXPECT_EQ(words(readView(v.uip_envelope.Get()))[43],0u)<<k;}uips=base;upload();envelope();EXPECT_EQ(readView(v.uip_envelope.Get()),stable);
 report["host_pending_faults"]=6;report["host_pending_pointer_faults"]=5;report["GPU_uip_faults"]=22;report["proof_faults"]=64;
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, HostDescriptorsForeignAliasesPredicationAndStateRestore) {
 author();queue();envelope();auto p=readView(owner.views().source_proof.Get()),e=readView(owner.views().uip_envelope.Get());auto original=request;
 for(auto roi:{std::array<UINT,4>{0,0,0,H},std::array<UINT,4>{left,top,0xffffffffu,top+H},std::array<UINT,4>{left,top,left+W,rawH+1}}){request=original;request.roi_ltrb=roi;EXPECT_FALSE(owner.queue(context.Get(),program,request,error));}
 request=original;request.transfer_mode=3;EXPECT_FALSE(owner.queue(context.Get(),program,request,error));request=original;
 auto wrong=texture(W,B,DXGI_FORMAT_R8G8B8A8_UNORM,false);request.display_snapshot_band=wrong.srv.Get();EXPECT_FALSE(owner.queue(context.Get(),program,request,error));request=original;
 request.display_snapshot_band=owner.views().analysis_band.Get();EXPECT_FALSE(owner.queue(context.Get(),program,request,error));request=original;
 request.parent_raw_source=owner.views().reference_rgb8.Get();request.roi_ltrb={0,0,W,H};EXPECT_FALSE(owner.queue(context.Get(),program,request,error));request=original;
 ComPtr<ID3D11DeviceContext>deferred;hr(device->CreateDeferredContext(0,&deferred));EXPECT_FALSE(owner.queue(deferred.Get(),program,request,error));
 ComPtr<ID3D11Device>fd;ComPtr<ID3D11DeviceContext>fc;D3D_FEATURE_LEVEL level{},fl[]{D3D_FEATURE_LEVEL_11_0};hr(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,fl,1,D3D11_SDK_VERSION,&fd,&level,&fc));EXPECT_FALSE(owner.queue(fc.Get(),program,request,error));
 D3D11_TEXTURE2D_DESC td{};td.Width=rawW;td.Height=rawH;td.MipLevels=td.ArraySize=td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R8G8B8A8_UNORM;td.Usage=D3D11_USAGE_DEFAULT;td.BindFlags=D3D11_BIND_SHADER_RESOURCE;ComPtr<ID3D11Texture2D>ft;ComPtr<ID3D11ShaderResourceView>fs;hr(fd->CreateTexture2D(&td,nullptr,&ft));hr(fd->CreateShaderResourceView(ft.Get(),nullptr,&fs));request.parent_raw_source=fs.Get();EXPECT_FALSE(owner.queue(context.Get(),program,request,error));request=original;
 auto shortRoot=buffer(63,true,true,false);request.actual_final256=shortRoot.buffer.Get();EXPECT_FALSE(owner.queue(context.Get(),program,request,error));request=original;
 std::array<UINT,20> shortC{};auto smallCB=constants(shortC);request.actual_contour_constants96=smallCB.Get();EXPECT_FALSE(owner.queue(context.Get(),program,request,error));request=original;
 auto shortS=buffer(31);request.actual_sct32=shortS.srv.Get();EXPECT_FALSE(owner.queue(context.Get(),program,request,error));request=original;
 D3D11_QUERY_DESC qd{};qd.Query=D3D11_QUERY_OCCLUSION_PREDICATE;ComPtr<ID3D11Predicate>pred;hr(device->CreatePredicate(&qd,&pred));context->SetPredication(pred.Get(),FALSE);EXPECT_FALSE(owner.queue(context.Get(),program,request,error));EXPECT_FALSE(owner.queue_uip_envelope(context.Get(),program,uip.srv.Get(),request,error));context->SetPredication(nullptr,FALSE);
 EXPECT_EQ(readView(owner.views().source_proof.Get()),p);EXPECT_EQ(readView(owner.views().uip_envelope.Get()),e);
 std::array<buf_t,16>ss;std::array<buf_t,8>uu;std::array<ComPtr<ID3D11Buffer>,3>bb;std::array<ID3D11ShaderResourceView*,16>s{};std::array<ID3D11UnorderedAccessView*,8>u{};std::array<ID3D11Buffer*,3>b{};
 for(UINT i=0;i<16;++i){ss[i]=buffer(4);s[i]=ss[i].srv.Get();}for(UINT i=0;i<8;++i){uu[i]=buffer(4,false,false,true);u[i]=uu[i].uav.Get();}for(UINT i=0;i<3;++i){std::array<UINT,4>x{i,0,0,0};bb[i]=constants(x);b[i]=bb[i].Get();}
 context->CSSetShaderResources(0,16,s.data());context->CSSetUnorderedAccessViews(0,8,u.data(),nullptr);context->CSSetConstantBuffers(0,3,b.data());context->CSSetShader(shaders[1].Get(),nullptr,0);
 for(UINT call=0;call<2;++call){if(call==0)queue();else envelope();std::array<ID3D11ShaderResourceView*,16>as{};std::array<ID3D11UnorderedAccessView*,8>au{};std::array<ID3D11Buffer*,3>ab{};context->CSGetShaderResources(0,16,as.data());context->CSGetUnorderedAccessViews(0,8,au.data());context->CSGetConstantBuffers(0,3,ab.data());for(UINT i=0;i<16;++i){EXPECT_EQ(as[i],s[i]);if(as[i])as[i]->Release();}for(UINT i=0;i<8;++i){EXPECT_EQ(au[i],u[i]);if(au[i])au[i]->Release();}for(UINT i=0;i<3;++i){EXPECT_EQ(ab[i],b[i]);if(ab[i])ab[i]->Release();}ComPtr<ID3D11ComputeShader>cs;UINT n=0;context->CSGetShader(&cs,nullptr,&n);EXPECT_EQ(cs.Get(),shaders[1].Get());EXPECT_EQ(n,0u);}
 report["host_rejections"]=15;report["CS_state_restored"]={{"SRV",16},{"UAV",8},{"CB",3},{"shader",true},{"queue",true},{"envelope",true}};
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, ActualMalformedCopyGridSeparatesCoverageFromValidity) {
 W=32;H=80;crop=16;B=64;rawW=64;rawH=96;left=5;top=7;author();queue();auto v=owner.views();
 auto uav=[&](ID3D11ShaderResourceView*s){ComPtr<ID3D11Resource>r;s->GetResource(&r);ComPtr<ID3D11UnorderedAccessView>u;hr(device->CreateUnorderedAccessView(r.Get(),nullptr,&u));return u;};
 auto r=uav(v.reference_rgb8.Get()),b=uav(v.analysis_band.Get()),p=uav(v.source_proof.Get());D3D11_UNORDERED_ACCESS_VIEW_DESC ad{};ad.Format=DXGI_FORMAT_R32_TYPELESS;ad.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ad.Buffer.NumElements=4;ad.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;ComPtr<ID3D11UnorderedAccessView>a;hr(device->CreateUnorderedAccessView(v.dispatch_arguments.Get(),&ad,&a));
 D3D11_SHADER_RESOURCE_VIEW_DESC rd{};rd.Format=DXGI_FORMAT_R32_TYPELESS;rd.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;rd.BufferEx.NumElements=64;rd.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;ComPtr<ID3D11ShaderResourceView>rs;hr(device->CreateShaderResourceView(v.final_root_mirror.Get(),&rd,&rs));UINT n=0;
 for(auto grid:{std::array<UINT,4>{4,2,1,0},std::array<UINT,4>{2,4,2,0},std::array<UINT,4>{3,4,1,0},std::array<UINT,4>{2,3,1,0}}){unbind(context.Get());ID3D11Buffer*c[2]{contour.Get(),v.request_constants.Get()};ID3D11ShaderResourceView*s[6]{display.srv.Get(),rs.Get(),sct.srv.Get(),nullptr,nullptr,raw.srv.Get()};ID3D11UnorderedAccessView*u[4]{r.Get(),b.Get(),p.Get(),a.Get()};context->CSSetConstantBuffers(0,2,c);context->CSSetShaderResources(0,6,s);context->CSSetUnorderedAccessViews(0,4,u,nullptr);context->CSSetShader(shaders[0].Get(),nullptr,0);context->Dispatch(1,1,1);ID3D11UnorderedAccessView*nullu=nullptr;context->CSSetUnorderedAccessViews(3,1,&nullu,nullptr);context->UpdateSubresource(v.dispatch_arguments.Get(),0,nullptr,grid.data(),0,0);context->CSSetShader(shaders[1].Get(),nullptr,0);context->DispatchIndirect(v.dispatch_arguments.Get(),0);context->CSSetShader(shaders[2].Get(),nullptr,0);context->DispatchIndirect(root.buffer.Get(),240);unbind(context.Get());auto got=words(readView(v.source_proof.Get()));EXPECT_EQ(got[0],0u);EXPECT_EQ(got[3],0u);EXPECT_EQ(got[34],0u);EXPECT_EQ(got[35],got[33]);EXPECT_EQ(got[38],2u);auto row=snapshot("grid-"+std::to_string(n++),"coverage_reject");row["actual_copy_grid"]=grid;report["observations"].back()=row;}
}
TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, ShapeTransitionRetainsOldStorageAndWritesNewBand) {
 author();positive("first");auto held=owner.views();auto old=reference_band(),p=readView(held.source_proof.Get());auto oldpointer=held.reference_rgb8.Get();need(owner.ensure(device.Get(),W,H,crop,B,error),"Stable ensure");EXPECT_EQ(owner.views().reference_rgb8.Get(),oldpointer);W=243;H=37;crop=20;B=17;author(2,true,true,0x1000000aaull,0xe00000000000aaa1ull);positive("second");EXPECT_NE(owner.views().reference_rgb8.Get(),oldpointer);EXPECT_EQ(readView(held.source_proof.Get()),p);report["retired_storage_held"]=true;
}

TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, DirectAuditUsesOriginalSnapshotAndPreservesSourceValidity) {
  constexpr std::array audit_specs {
    cache::shader_spec {"direct_source_audit.hlsl", "prepare_main", "cs_5_0"},
    cache::shader_spec {"direct_source_audit.hlsl", "copy_main", "cs_5_0"},
    cache::shader_spec {"direct_source_audit.hlsl", "finalize_main", "cs_5_0"},
    cache::shader_spec {"direct_source_audit.hlsl", "envelope_main", "cs_5_0"}
  };
  const auto sources = cache::snapshot_sources(std::getenv("SUNSHINE_TEST_DISPLAY_RGB8_SHADER_ROOT"), audit_specs);
  ASSERT_EQ(cache::source_closure_sha256(sources), "a5145b7de00d113a135b1c852e4a244735d08b9211bafad5bc08f4bc70e47261");
  std::array<ComPtr<ID3D11ComputeShader>, audit_specs.size()> audit_shaders;
  for (std::size_t i = 0u; i < audit_specs.size(); ++i) {
    const auto bytecode = cache::get(sources, audit_specs[i]);
    ASSERT_TRUE(bytecode && !bytecode->empty());
    hr(device->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &audit_shaders[i]));
  }
  const cg::program_t audit_program {
    audit_shaders[0].Get(), audit_shaders[1].Get(), audit_shaders[2].Get(), audit_shaders[3].Get()
  };
  cg::resources_t audit_owner;
  author(0u, true);
  ASSERT_TRUE(audit_owner.ensure_direct_audit(device.Get(), W, H, crop, B, error)) << error;
  EXPECT_FALSE(audit_owner.views().reference_rgb8);
  EXPECT_FALSE(audit_owner.views().analysis_band);
  EXPECT_FALSE(audit_owner.queue_current(context.Get(), audit_program, request, uip.srv.Get(), error));
  EXPECT_FALSE(owner.queue_current_audit(context.Get(), program, request, uip.srv.Get(), error));
  const auto capture = [&](ID3D11Buffer *work_root = nullptr) {
    const auto audit = audit_owner.queue_current_audit(context.Get(), audit_program, request, uip.srv.Get(), error, work_root);
    need(audit.has_value(), error.c_str());
    EXPECT_EQ(audit->display_snapshot_band(), display.srv.Get());
    EXPECT_EQ(audit->uip_envelope(), audit_owner.views().uip_envelope.Get());
    return readView(audit->uip_envelope());
  };
  const auto expected_envelope = [&](bool invalid = false) {
    auto proof = expected_proof();
    proof[0u] = invalid ? 0u : cg::direct_audit_tag;
    proof[3u] = invalid ? 0u : 1u;
    proof[31u] = cg::direct_audit_features;
    if (invalid) {
      proof[34u] = 1u;
      --proof[35u];
      proof[38u] = 4u;
    }
    proof[61u] = proof[62u] = 0u;
    auto envelope = pack(uips);
    const auto proof_bytes = pack(proof);
    envelope.insert(envelope.end(), proof_bytes.begin(), proof_bytes.end());
    EXPECT_EQ(readView(audit_owner.views().source_proof.Get()), proof_bytes);
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
  void SetUp() override {
    constexpr auto level = D3D_FEATURE_LEVEL_11_0;
    ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u, &level, 1u,
      D3D11_SDK_VERSION, &device, nullptr, &context)));
  }
};

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
  const auto scatter = load(cache::host_sbs_direct_glyph_publish_line);
  const auto structured = [&](UINT count, UINT stride, const void *initial) {
    buf_t b;
    D3D11_BUFFER_DESC d {count * stride, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0u, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, stride};
    D3D11_SUBRESOURCE_DATA data {initial, 0, 0};
    hr(device->CreateBuffer(&d, initial ? &data : nullptr, &b.buffer));
    hr(device->CreateShaderResourceView(b.buffer.Get(), nullptr, &b.srv));
    hr(device->CreateUnorderedAccessView(b.buffer.Get(), nullptr, &b.uav));
    return b;
  };
  const std::array<UINT, 4> old_box {10, 8, 82, 40}, current_box {16, 10, 80, 38}, cap_box {12, 6, 84, 42};
  std::array<UINT, 8 * 16> old {};
  for (UINT at : {0u, 4u, 8u}) {
    std::copy(old_box.begin(), old_box.end(), old.begin() + at);
  }
  const auto prior_lines = structured(8, 64, old.data());
  const auto current = structured(8, 64, nullptr);
  const auto sizes = structured(8, 8, nullptr);
  std::vector<float> past(width * height, 1.0f), raw_now(width * height, 0.0f);
  const auto prior_alpha = structured(width * height, 4, past.data());
  const auto new_alpha = structured(width * height, 4, raw_now.data());
  const auto remembered = structured(width * height, 4, past.data());
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
  const auto exercise = [&](UINT count, UINT cap_token, bool swap, UINT center_twice = 0) {
    const auto cb = constants(std::array<UINT, 8> {width, height, 17, 0, 1, 1, 0, center_twice});
    std::array<UINT, 36> work {count ? 2u : 1u, count}, caps {1u, count, cap_token, 0u};
    for (UINT slot = 0; slot < count; ++slot) {
      std::copy(current_box.begin(), current_box.end(), work.begin() + 4u + 4u * slot);
      std::copy(cap_box.begin(), cap_box.end(), caps.begin() + 4u + 4u * slot);
    }
    const auto current_work = buffer(work.size(), true, false, true, work.data());
    const auto detector = buffer(caps.size(), true, false, false, caps.data());
    auto *rgb = swap ? swapped_rgb.srv.Get() : previous_rgb.srv.Get();
    auto pass = [&](ID3D11ComputeShader *shader, UINT gx, UINT gy, std::array<ID3D11ShaderResourceView *, 7> in, std::array<ID3D11UnorderedAccessView *, 5> out) {
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
    pass(confirm.Get(), 8, 1, {nullptr, prior_lines.srv.Get(), previous_rgb.srv.Get(), rgb, prior_alpha.srv.Get()}, {nullptr, current.uav.Get()});
    pass(finish.Get(), 1, 1, {}, {current_work.uav.Get(), current.uav.Get(), sizes.uav.Get()});
    pass(scatter.Get(), 6, 6, {nullptr, prior_lines.srv.Get(), previous_rgb.srv.Get(), rgb, prior_alpha.srv.Get(), new_alpha.srv.Get(), current.srv.Get()}, {current_work.uav.Get(), nullptr, nullptr, remembered.uav.Get(), output.uav.Get()});
    return std::array {words(read(current_work.buffer.Get())), words(read(current.buffer.Get())), words(read(output.texture.Get()))};
  };
  const auto same = exercise(1, 17, false);
  EXPECT_EQ((std::vector<UINT>(same[0].begin() + 4, same[0].begin() + 8)), (std::vector<UINT> {12, 8, 82, 40}));
  EXPECT_EQ(same[1][14], 1u);
  EXPECT_EQ(same[2][24u * width + 35u], 242u);
  EXPECT_EQ(same[2][24u * width + 11u], 0u);
  EXPECT_EQ(words(read(remembered.buffer.Get()))[16u * 70u + 23u], 0u);
  const auto swap = exercise(1, 17, true);
  EXPECT_EQ((std::vector<UINT>(swap[0].begin() + 4, swap[0].begin() + 8)), (std::vector<UINT> {14, 8, 82, 40}));
  EXPECT_EQ(swap[1][14], 0u);
  EXPECT_EQ(swap[2][24u * width + 35u], 0u);
  // Swapped RGB prevents history from extending the current box.
  const auto left = exercise(1, 17, true, 90);
  EXPECT_EQ((std::vector<UINT>(left[0].begin() + 4, left[0].begin() + 8)), (std::vector<UINT> {12, 8, 82, 40}));
  const auto right = exercise(1, 17, true, 102);
  EXPECT_EQ((std::vector<UINT>(right[0].begin() + 4, right[0].begin() + 8)), (std::vector<UINT> {14, 8, 84, 40}));
  EXPECT_EQ(exercise(1, 17, true, 20)[0], swap[0]);
  const auto ambiguous = exercise(2, 17, false);
  EXPECT_EQ(ambiguous[1][12], 0u);
  EXPECT_EQ(ambiguous[1][16 + 12], 0u);
  EXPECT_EQ(exercise(0, 17, false)[0][0], 1u);
  const auto stale = exercise(1, 16, false);
  EXPECT_EQ(stale[0][0], 0u);
  EXPECT_NE(stale[0][2], 0u);
  EXPECT_EQ(stale[1][14], 0u);
}

TEST_F(HostSbsDisplaySnapshotRgb8GpuTest, EpochAndEmptyAreSourceFactsNotUiAuthority) {
 author();queue();scts[2]=2;upload();queue();EXPECT_EQ(words(readView(owner.views().source_proof.Get()))[3],0u);snapshot("bad-SCT2","join_reject");
 scts[2]=0;scts[3]+=1;upload();queue();check_outputs();envelope();prefix();EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43],0u);auto row=snapshot("epoch-source-only","source_ready_envelope_reject");row["envelope"]=save("epoch-source-only-envelope.bin",readView(owner.views().uip_envelope.Get()));report["observations"].back()=row;
 uips[20]=scts[3];upload();envelope();prefix();EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43],1u);
 uips[3]=256;upload();envelope();prefix();EXPECT_EQ(words(readView(owner.views().uip_envelope.Get()))[43],0u);
}
} // namespace
#endif
