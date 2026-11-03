#ifdef _WIN32
#include "src/host_sbs_shader_cache.h"
#include "src/host_sbs_temporal_frame_context_gpu.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include "src/crypto.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>
using Microsoft::WRL::ComPtr;
namespace {
namespace gpu=models::host_sbs_temporal_frame_context_gpu;
namespace temporal=models::host_sbs_body_witness_temporal_gpu;
namespace source=models::host_sbs_temporal_source_bridge;
class TemporalFrameContextGpu:public ::testing::Test {
protected:
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;ComPtr<ID3D11ComputeShader> shader;
 ComPtr<ID3D11Buffer> cb,uip,root;ComPtr<ID3D11ShaderResourceView> uip_srv,root_srv;
 gpu::resources_t transport;gpu::frame_t seed;std::array<UINT,24> c{};std::array<UINT,40> u{};std::array<UINT,64> r{};
 std::string error;
 void SetUp() override {
  const auto path=std::getenv("SUNSHINE_TEMPORAL_FRAME_CONTEXT_ROOT"),pin=std::getenv("SUNSHINE_TEMPORAL_FRAME_CONTEXT_CLOSURE");
  if(!path||!pin)GTEST_SKIP()<<"Private Frame256 source pins not configured";const std::array specs{models::host_sbs_shader_cache::shader_spec{"host_sbs_temporal_frame_context_cs.hlsl","main","cs_5_0"}};
  const auto sources=models::host_sbs_shader_cache::snapshot_sources(path,specs);ASSERT_TRUE(sources);ASSERT_EQ(models::host_sbs_shader_cache::source_closure_sha256(sources),pin);
  const auto code=models::host_sbs_shader_cache::get(sources,specs[0]);ASSERT_TRUE(code);
  const bool hardware=std::getenv("SUNSHINE_TEMPORAL_FRAME_CONTEXT_HARDWARE")&&std::string(std::getenv("SUNSHINE_TEMPORAL_FRAME_CONTEXT_HARDWARE"))=="1";
  const D3D_FEATURE_LEVEL levels[]{D3D_FEATURE_LEVEL_11_0};D3D_FEATURE_LEVEL actual;
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr,hardware?D3D_DRIVER_TYPE_HARDWARE:D3D_DRIVER_TYPE_WARP,nullptr,0,levels,1,D3D11_SDK_VERSION,&device,&actual,&context)));
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(code->data(),code->size(),nullptr,&shader)));
  RecordProperty("driver",hardware?"hardware":"warp");RecordProperty("source_closure",pin);
  c={32,64,48,16,16,16,12,16,0,0,16,16,3,0,2,0,1001,0,1,0,1,27,0,0};
  u[0]=4;u[1]=0x34504955u;u[2]=1;u[3]=1;u[4]=3;u[6]=2;u[8]=1001;u[10]=32;u[11]=64;u[12]=16;u[13]=16;u[14]=12;u[15]=16;u[18]=16;u[19]=16;u[20]=23;
  const UINT cookie0=99u^0xa3756c91u,cookie1=0u^0x5c8a936eu,marker=0x52434f4fu;
  r[0]=1;r[1]=1u^0xd1ec15a5u^marker;r[2]=99;r[4]=cookie0;r[5]=cookie1;r[6]=0x47524243u;r[7]=marker;
  r[8]=99;r[10]=cookie0;r[11]=cookie1;r[12]=0x54535152u;r[13]=1;r[14]=1u^0x6f435257u;
  for(UINT i:{20u,52u,60u})r[i]=r[i+1]=r[i+2]=1;
  source::certificate_t cert;cert.domain.source_width=32;cert.domain.source_height=64;cert.domain.analysis_ltrb={0,0,32,64};cert.domain.authority=1;cert.domain.analysis_generation=2;cert.domain.context_generation=6;
  cert.source_epoch=7;cert.material_serial=2;cert.capture_clock_us=501;cert.predecessor_contiguous=true;cert.genuine_capture=true;
  seed=gpu::seed(cert,32,16,temporal::caps_t{},3,1001,99,gpu::postprocess_mode_e::queued_force);
  ASSERT_NO_FATAL_FAILURE(make_buffer(c.data(),96,D3D11_BIND_CONSTANT_BUFFER,0,0,cb,nullptr));
  ASSERT_NO_FATAL_FAILURE(make_buffer(u.data(),160,D3D11_BIND_SHADER_RESOURCE,D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,4,uip,&uip_srv));
  ASSERT_NO_FATAL_FAILURE(make_buffer(r.data(),256,D3D11_BIND_SHADER_RESOURCE,D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS,0,root,&root_srv));
  ASSERT_TRUE(transport.ensure(device.Get(),error))<<error;
 }
 void make_buffer(const void*data,UINT bytes,UINT bind,UINT misc,UINT stride,ComPtr<ID3D11Buffer>&b,ComPtr<ID3D11ShaderResourceView>*s){D3D11_BUFFER_DESC d{};d.ByteWidth=bytes;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=bind;d.MiscFlags=misc;d.StructureByteStride=stride;D3D11_SUBRESOURCE_DATA initial{};initial.pSysMem=data;ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(&d,&initial,&b)));if(s){D3D11_SHADER_RESOURCE_VIEW_DESC v{};if(misc&D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS){v.Format=DXGI_FORMAT_R32_TYPELESS;v.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;v.BufferEx.NumElements=bytes/4;v.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;}else{v.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;v.Buffer.NumElements=bytes/4;}ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(b.Get(),&v,s->ReleaseAndGetAddressOf())));}}
 std::array<UINT,64> expected(){std::array<UINT,64>x{};std::copy_n(seed.words.begin(),8,x.begin());x[8]=c[18];x[9]=x[10]=1;std::copy_n(seed.words.begin()+12,8,x.begin()+12);x[20]=c[14];x[21]=c[15];x[22]=u[20];x[23]=1;std::copy_n(c.begin(),12,x.begin()+24);x[48]=c[12];x[49]=c[13];x[50]=c[16];x[51]=c[17];x[52]=seed.words[9];return x;}
 void upload(){context->UpdateSubresource(cb.Get(),0,nullptr,c.data(),0,0);context->UpdateSubresource(uip.Get(),0,nullptr,u.data(),0,0);context->UpdateSubresource(root.Get(),0,nullptr,r.data(),0,0);}
 std::array<UINT,64> read(){std::array<UINT,64>x{};D3D11_BUFFER_DESC d{};transport.output()->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;d.BindFlags=0;d.MiscFlags=0;d.StructureByteStride=0;ComPtr<ID3D11Buffer>s;EXPECT_TRUE(SUCCEEDED(device->CreateBuffer(&d,nullptr,&s)));if(!s)return x;context->CopyResource(s.Get(),transport.output().Get());D3D11_MAPPED_SUBRESOURCE m{};const auto h=context->Map(s.Get(),0,D3D11_MAP_READ,0,&m);EXPECT_TRUE(SUCCEEDED(h));if(SUCCEEDED(h)){std::memcpy(x.data(),m.pData,256);context->Unmap(s.Get(),0);}return x;}
 void queue(){upload();ASSERT_TRUE(transport.queue(context.Get(),shader.Get(),cb.Get(),uip_srv.Get(),root_srv.Get(),seed,error))<<error;}
};
TEST_F(TemporalFrameContextGpu, Exact256IncludesOwnCutAndSeparateSourceModel){queue();EXPECT_EQ(read(),expected());EXPECT_NE(seed.words[4],c[12]);EXPECT_NE(seed.words[6],c[16]);}
TEST_F(TemporalFrameContextGpu, SameResourcesMutateCutOnlyFromGpuUIP){queue();u[20]=24;queue();EXPECT_EQ(read(),expected());}
TEST_F(TemporalFrameContextGpu, SourceGapRemainsZeroContinuity){seed.words[9]=0;queue();EXPECT_EQ(read(),expected());EXPECT_EQ(read()[52],0u);}
TEST_F(TemporalFrameContextGpu, StaleUIPClockWritesLiteralZero){u[8]=1000;queue();EXPECT_EQ(read(),(std::array<UINT,64>{}));}
TEST_F(TemporalFrameContextGpu, MissingPostChildTagWritesLiteralZero){r[6]=0;queue();EXPECT_EQ(read(),(std::array<UINT,64>{}));}
TEST_F(TemporalFrameContextGpu, MissingOptionalMarkerWritesLiteralZero){r[7]=0;r[1]=1u^0xd1ec15a5u;queue();EXPECT_EQ(read(),(std::array<UINT,64>{}));}
TEST_F(TemporalFrameContextGpu, WrongSourceOrModelNamespaceWritesLiteralZero){seed.words[28]=4;queue();EXPECT_EQ(read(),(std::array<UINT,64>{}));}
TEST_F(TemporalFrameContextGpu, RootZeroNeverBorrowsOldFrame){queue();r[60]=0;queue();EXPECT_EQ(read(),(std::array<UINT,64>{}));}
TEST_F(TemporalFrameContextGpu, CsSlotsShaderAndConstantsRestore){ID3D11ShaderResourceView*sentinel[2]{uip_srv.Get(),root_srv.Get()};ID3D11Buffer*constants[2]{cb.Get(),cb.Get()};context->CSSetShaderResources(0,2,sentinel);context->CSSetConstantBuffers(0,2,constants);context->CSSetShader(shader.Get(),nullptr,0);queue();ID3D11ShaderResourceView*after[2]{};context->CSGetShaderResources(0,2,after);for(UINT i=0;i<2;++i){EXPECT_EQ(after[i],sentinel[i]);if(after[i])after[i]->Release();}ID3D11Buffer*b[2]{};context->CSGetConstantBuffers(0,2,b);for(UINT i=0;i<2;++i){EXPECT_EQ(b[i],constants[i]);if(b[i])b[i]->Release();}ComPtr<ID3D11ComputeShader>cs;context->CSGetShader(&cs,nullptr,nullptr);EXPECT_EQ(cs.Get(),shader.Get());}
TEST_F(TemporalFrameContextGpu, Saved28ActualContextParity){
 const char*path=std::getenv("SUNSHINE_TEMPORAL_FRAME_CONTEXT_CASES");if(!path)GTEST_SKIP()<<"Saved28 transport cards not configured";
 const auto read_file=[](const std::filesystem::path&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||f.tellg()<0)throw std::runtime_error("Cannot read context fixture");std::vector<char>b(static_cast<std::size_t>(f.tellg()));f.seekg(0);f.read(b.data(),static_cast<std::streamsize>(b.size()));if(!f)throw std::runtime_error("Context fixture truncated");return b;};
 const auto bound=[&](const nlohmann::json&x,std::size_t n){auto b=read_file(x.at("path").get<std::string>());const auto d=crypto::hash(std::string_view(b.data(),b.size()));constexpr char digits[]="0123456789abcdef";std::string h;for(auto v:d){h+=digits[v>>4u];h+=digits[v&15u];}if(b.size()!=n||b.size()!=x.at("bytes").get<std::size_t>()||h!=x.at("sha256").get<std::string>())throw std::runtime_error("Context fixture binding mismatch");return b;};
 const auto text=read_file(path);const auto spec=nlohmann::json::parse(text.begin(),text.end());ASSERT_EQ(spec.at("schema"),1u);ASSERT_EQ(spec.at("cases").size(),28u);
 for(const auto &item:spec.at("cases")){SCOPED_TRACE(item.at("name").get<std::string>());const auto&a=item.at("artifacts");auto x=bound(a.at("seed"),256);std::memcpy(seed.words.data(),x.data(),256);x=bound(a.at("constants"),96);std::memcpy(c.data(),x.data(),96);x=bound(a.at("uip"),160);std::memcpy(u.data(),x.data(),160);x=bound(a.at("transaction"),256);std::memcpy(r.data(),x.data(),256);const auto e=bound(item.at("expected"),256);std::array<UINT,64>expected_frame{};std::memcpy(expected_frame.data(),e.data(),256);queue();EXPECT_EQ(read(),expected_frame);}
}

TEST_F(TemporalFrameContextGpu, ActivePredicationRejectsWithoutChangingIt){queue();D3D11_QUERY_DESC q{D3D11_QUERY_OCCLUSION_PREDICATE,0};ComPtr<ID3D11Predicate>predicate;ASSERT_TRUE(SUCCEEDED(device->CreatePredicate(&q,&predicate)));context->SetPredication(predicate.Get(),TRUE);EXPECT_FALSE(transport.queue(context.Get(),shader.Get(),cb.Get(),uip_srv.Get(),root_srv.Get(),seed,error));ComPtr<ID3D11Predicate>after;BOOL value=FALSE;context->GetPredication(&after,&value);EXPECT_EQ(after.Get(),predicate.Get());EXPECT_EQ(value,TRUE);context->SetPredication(nullptr,FALSE);EXPECT_EQ(read(),expected());}
TEST_F(TemporalFrameContextGpu, FullGpuBufferCopiesToExistingConstantBuffer){queue();temporal::resources_t backend;const temporal::caps_t caps{16,4096,4096,32};seed.words[12]=caps.components;seed.words[13]=caps.members;seed.words[14]=caps.bit_words;seed.words[15]=caps.hash_words;queue();ASSERT_TRUE(backend.ensure(device.Get(),32,16,caps,error))<<error;ASSERT_TRUE(backend.prepare_live_frame_owned(context.Get(),seed,error))<<error;const auto frame=backend.mutable_views().frame_constants;context->CopyResource(frame.Get(),transport.output().Get());D3D11_BUFFER_DESC d{};frame->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ComPtr<ID3D11Buffer>staging;ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(&d,nullptr,&staging)));context->CopyResource(staging.Get(),frame.Get());D3D11_MAPPED_SUBRESOURCE map{};ASSERT_TRUE(SUCCEEDED(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&map)));std::array<UINT,64>actual{};std::memcpy(actual.data(),map.pData,256);context->Unmap(staging.Get(),0);EXPECT_EQ(actual,expected());}

TEST_F(TemporalFrameContextGpu, CpuCompletedForceAcceptsLiteralInactiveInferRecord){seed.words[11]=2u;for(UINT i=20u;i<24u;++i)r[i]=0u;queue();EXPECT_EQ(read(),expected());}
TEST_F(TemporalFrameContextGpu, QueuedForceAcceptsFinalizedInferRecord){seed.words[11]=4u;queue();EXPECT_EQ(read(),expected());}
TEST_F(TemporalFrameContextGpu, QueuedForceRejectsInactiveInferRecord){seed.words[11]=4u;for(UINT i=20u;i<24u;++i)r[i]=0u;queue();EXPECT_EQ(read(),(std::array<UINT,64>{}));}
TEST_F(TemporalFrameContextGpu, CpuCompletedForceRejectsActiveInferRecord){seed.words[11]=2u;queue();EXPECT_EQ(read(),(std::array<UINT,64>{}));}
TEST_F(TemporalFrameContextGpu, UnknownOrMixedModeCannotAuthorize){for(UINT mode:{0u,3u,5u,6u,7u,0xffffffffu}){seed.words[11]=mode;queue();EXPECT_EQ(read(),(std::array<UINT,64>{}))<<mode;}}
TEST_F(TemporalFrameContextGpu, AuthorizedOpaqueInferRequiresFinalizedRecord){seed.words[11]=1u;queue();EXPECT_EQ(read(),expected());r[20]=0u;queue();EXPECT_EQ(read(),(std::array<UINT,64>{}));}

}
#endif
