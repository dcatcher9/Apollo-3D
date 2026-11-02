/** EXTERNAL contract-only GPU fixture. Preuploaded M/target, no glyph-detection proof. */
#include <gtest/gtest.h>
#ifdef _WIN32
#include <src/host_sbs_current_ui_capability.h>
#include <src/depth_coordinate_v2.h>
#include <src/host_sbs_shader_cache.h>
#include <src/host_sbs_subtitle_render.h>
#include <src/host_sbs_v2_geometry.h>
#include <src/platform/windows/host_sbs_v2_renderer.h>
#include <tests/fixtures/host_sbs_adaptive_camera_state.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <d3d11.h>
#include <wrl/client.h>

namespace {
using Microsoft::WRL::ComPtr;
namespace cui=models::research_current_ui_v1;
namespace v2=models::depth_coordinate_v2;
struct pixel {float r,g,b,a;};
struct half_pixel {std::uint16_t r,g,b,a;};
static_assert(sizeof(pixel)==16u&&sizeof(half_pixel)==8u);
constexpr UINT W=128u,H=4u,Y=2u,X=64u;
constexpr const char *source_pin="66f1fcc804ce9e955ac8f2a1b967a8fc4b4d6a1eabe214720580dfd5e4710e07";
constexpr const char *header_pin="3fdf09beb4a6fc8c37617cd8c67d4629b111853686e0476477ebb94a01114090";
struct operands {
  std::vector<pixel> source=std::vector<pixel>(W*H,{0,0,0,1});
  std::vector<float> base=std::vector<float>(W*H,6.0f/W);
  v2::state_words_t state=v2::state_initial_words;
  std::vector<UINT> oldM=std::vector<UINT>(W*H,0u),currentM=std::vector<UINT>(W*H,0u);
  std::vector<half_pixel> oldRGB=std::vector<half_pixel>(W*H,{0,0,0,0x3c00u}),currentRGB=oldRGB;
  std::array<UINT,6> scp{};std::array<UINT,32> sct{};
  models::host_sbs_subtitle_render_constants_t oldCB{1u,0u,7u,0u,0u,0u,W,H};
  cui::constants_t currentCB;
  cui::packet_t packet;
  operands() {
    state[v2::center]=std::bit_cast<UINT>(.25f);state[v2::inverse_scale]=std::bit_cast<UINT>(1.0f/v2::model_calibrations.front().raw_coordinate_scale);
    state[v2::convergence_curve]=std::bit_cast<UINT>(v2::convergence_curve_default);state[v2::container_scale]=std::bit_cast<UINT>(1.0f);
    state[v2::calibration_revision]=1u;state[v2::frame_valid]=std::bit_cast<UINT>(1.0f);state[v2::contract_tag_bits]=v2::contract_tag;
    state[v2::renderer_authorization_bits]=v2::contract_tag;state[v2::confirmed_cut_count]=3u;host_sbs_test::seal_adaptive_camera(state);
    state[v2::gain_last_observation_low]=1000u;state[v2::gain_last_observation_high]=0u;state[v2::camera_center_integrity_bits]=v2::camera_center_integrity_for_state_words(state);
    source[Y*W+X]={1,1,1,1};oldM[Y*W+X]=currentM[Y*W+X]=1u;oldRGB[Y*W+X]=currentRGB[Y*W+X]={0x3c00u,0x3c00u,0x3c00u,0x3c00u};
    scp={v2::subtitle_condition_param_schema,v2::subtitle_condition_param_tag,1u,0u,v2::subtitle_condition_param_fully_active_value,std::bit_cast<UINT>(2.0f/W)};
    sct[0]=v2::subtitle_contour_schema;sct[1]=v2::subtitle_contour_history_tag;sct[2]=sct[23]=1u;sct[3]=3u;sct[4]=2u;sct[6]=7u;
    sct[8]=W;sct[9]=H;sct[11]=H;sct[12]=W;sct[13]=H;sct[15]=H;sct[18]=W;sct[19]=H;sct[20]=1000u;sct[22]=v2::subtitle_contour_recipe;
    const cui::current_capture_t capture{11u,22u,2000u,7u,13u,0xabcu};const cui::retained_base_t owner{17u,19u,1000u,3u};
    const cui::layout_t layout{{0u,0u,W,H},{0u,H,W,H},{0u,0u,W,H}};
    currentCB=cui::make_constants(capture,owner,layout);packet=cui::make_fixture_packet(currentCB,cui::decision_e::positive,2.0f/W);
  }
  void decision(cui::decision_e value,float target) {packet=cui::make_fixture_packet(currentCB,value,target);}
};
struct result {std::vector<std::byte> pixels;std::vector<float> actualBase;v2::state_words_t actualState{};};
class HostSbsCurrentUiCapabilityGpuTest:public ::testing::Test {
protected:
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;ComPtr<ID3D11VertexShader> vs;std::array<ComPtr<ID3D11PixelShader>,3> ps;ComPtr<ID3D11SamplerState> sampler;
 static void hr(HRESULT value){if(FAILED(value))throw std::runtime_error("CurrentUI contract fixture D3D failure");}
 void SetUp()override {
  const auto *path=std::getenv("SUNSHINE_TEST_CURRENT_UI_SHADER_ROOT");if(!path||!*path)GTEST_SKIP()<<"Explicit external shader root required";
  const auto *driver=std::getenv("SUNSHINE_TEST_CURRENT_UI_DRIVER");const bool hardware=driver&&std::string(driver)=="hardware";
  ASSERT_TRUE(!driver||std::string(driver)=="warp"||std::string(driver)=="hardware");D3D_FEATURE_LEVEL level{},requested[]{D3D_FEATURE_LEVEL_11_0};
  hr(D3D11CreateDevice(nullptr,hardware?D3D_DRIVER_TYPE_HARDWARE:D3D_DRIVER_TYPE_WARP,nullptr,0u,requested,1u,D3D11_SDK_VERSION,&device,&level,&context));ASSERT_EQ(level,D3D_FEATURE_LEVEL_11_0);
  namespace cache=models::host_sbs_shader_cache;const std::array specs{
    cache::shader_spec{"sbs_reprojection_vs.hlsl","main_vs","vs_5_0"},cache::shader_spec{"legacy_live_ps.hlsl","main_ps","ps_5_0"},
    cache::shader_spec{"current_ui_research_ps.hlsl","main_ps","ps_5_0"},cache::shader_spec{"current_ui_research_ps.hlsl","current_ui_contract_probe_ps","ps_5_0"}};
  const auto sources=cache::snapshot_sources(std::filesystem::path(path),specs);ASSERT_EQ(cache::source_closure_sha256(sources),source_pin);
  for(std::size_t i=0;i<specs.size();++i){const auto code=cache::get(sources,specs[i]);ASSERT_TRUE(code&&!code->empty());if(i==0)hr(device->CreateVertexShader(code->data(),code->size(),nullptr,&vs));else hr(device->CreatePixelShader(code->data(),code->size(),nullptr,&ps[i-1]));}
  D3D11_SAMPLER_DESC desc{};desc.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;desc.AddressU=desc.AddressV=desc.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;desc.MaxLOD=D3D11_FLOAT32_MAX;hr(device->CreateSamplerState(&desc,&sampler));
  RecordProperty("external_source_closure",source_pin);RecordProperty("current_ui_header",header_pin);RecordProperty("preuploaded_contract_inputs_not_glyph_detection",true);RecordProperty("performance_qualified",false);
 }
 ComPtr<ID3D11ShaderResourceView> texture(DXGI_FORMAT format,UINT w,UINT h,const void *data,UINT pitch,ComPtr<ID3D11Texture2D> *keep=nullptr) {
  D3D11_TEXTURE2D_DESC d{};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=1u;d.Format=format;d.SampleDesc.Count=1u;d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
  D3D11_SUBRESOURCE_DATA initial{data,pitch,0u};ComPtr<ID3D11Texture2D> t;ComPtr<ID3D11ShaderResourceView> v;hr(device->CreateTexture2D(&d,&initial,&t));hr(device->CreateShaderResourceView(t.Get(),nullptr,&v));if(keep)*keep=t;return v;
 }
 template<class T>ComPtr<ID3D11Buffer> cb(const T &data) {
  static_assert(sizeof(T)%16u==0u);D3D11_BUFFER_DESC d{};d.ByteWidth=sizeof(T);d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA initial{&data,0,0};ComPtr<ID3D11Buffer> b;hr(device->CreateBuffer(&d,&initial,&b));return b;
 }
 template<class T>ComPtr<ID3D11ShaderResourceView> buffer(const T &data,UINT stride,ComPtr<ID3D11Buffer> *keep=nullptr) {
  D3D11_BUFFER_DESC d{};d.ByteWidth=sizeof(T);d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;d.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;d.StructureByteStride=stride;
  D3D11_SUBRESOURCE_DATA initial{&data,0,0};ComPtr<ID3D11Buffer> b;ComPtr<ID3D11ShaderResourceView> v;hr(device->CreateBuffer(&d,&initial,&b));hr(device->CreateShaderResourceView(b.Get(),nullptr,&v));if(keep)*keep=b;return v;
 }
 std::vector<std::byte> readTexture(ID3D11Texture2D *source,UINT pixelBytes) {
  D3D11_TEXTURE2D_DESC d{};source->GetDesc(&d);auto sd=d;sd.Usage=D3D11_USAGE_STAGING;sd.BindFlags=sd.MiscFlags=0;sd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ComPtr<ID3D11Texture2D> stage;hr(device->CreateTexture2D(&sd,nullptr,&stage));context->CopyResource(stage.Get(),source);D3D11_MAPPED_SUBRESOURCE m{};hr(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&m));std::vector<std::byte> out(std::size_t(d.Width)*d.Height*pixelBytes);
  for(UINT y=0;y<d.Height;++y)std::memcpy(out.data()+std::size_t(y)*d.Width*pixelBytes,static_cast<const char*>(m.pData)+std::size_t(y)*m.RowPitch,std::size_t(d.Width)*pixelBytes);context->Unmap(stage.Get(),0);return out;
 }
 result draw(const operands &in,UINT shader,bool optional_resources=true,bool bind_expected=true) {
  auto color=texture(DXGI_FORMAT_R32G32B32A32_FLOAT,W,H,in.source.data(),W*sizeof(pixel));ComPtr<ID3D11Texture2D> baseTexture;
  auto base=texture(DXGI_FORMAT_R32_FLOAT,W,H,in.base.data(),W*4u,&baseTexture);ComPtr<ID3D11Buffer> stateBuffer;auto state=buffer(in.state,16u,&stateBuffer);
  auto oldM=texture(DXGI_FORMAT_R32_UINT,W,H,in.oldM.data(),W*4u);auto oldRGB=texture(DXGI_FORMAT_R16G16B16A16_FLOAT,W,H,in.oldRGB.data(),W*8u);auto scp=buffer(in.scp,4u),sct=buffer(in.sct,4u);
  auto currentM=texture(DXGI_FORMAT_R32_UINT,W,H,in.currentM.data(),W*4u);auto currentRGB=texture(DXGI_FORMAT_R16G16B16A16_FLOAT,W,H,in.currentRGB.data(),W*8u);auto packet=buffer(in.packet,4u);
  auto geometry=cb(models::make_host_sbs_v2_full_frame_geometry(1,1)),oldCB=cb(in.oldCB),currentCB=cb(in.currentCB);
  D3D11_TEXTURE2D_DESC d{};d.Width=W*2u;d.Height=H;d.MipLevels=d.ArraySize=1u;d.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;d.SampleDesc.Count=1;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_RENDER_TARGET;ComPtr<ID3D11Texture2D> output;ComPtr<ID3D11RenderTargetView> rtv;hr(device->CreateTexture2D(&d,nullptr,&output));hr(device->CreateRenderTargetView(output.Get(),nullptr,&rtv));
  ID3D11ShaderResourceView *extra[]{optional_resources?currentM.Get():nullptr,optional_resources?currentRGB.Get():nullptr,optional_resources?packet.Get():nullptr};context->PSSetShaderResources(7,3,extra);ID3D11Buffer *expected=bind_expected?currentCB.Get():nullptr;context->PSSetConstantBuffers(4,1,&expected);
  const platf::dxgi::host_sbs_v2_draw_command_t command{.render_targets={rtv.Get(),nullptr},.render_target_count=1,.vertex_shader=vs.Get(),.pixel_shader=ps.at(shader).Get(),.viewport={0,0,float(W*2u),float(H),0,1},.sampler=sampler.Get(),.shader_resources={color.Get(),base.Get(),state.Get(),oldM.Get(),oldRGB.Get(),scp.Get(),sct.Get()},.geometry_constants=geometry.Get(),.subtitle_constants=oldCB.Get()};
  if(!platf::dxgi::record_host_sbs_v2_draw(context.Get(),command))throw std::runtime_error("Shared production draw recorder rejected fixture");ID3D11ShaderResourceView *nulls[3]{};ID3D11Buffer *nullcb=nullptr;context->PSSetShaderResources(7,3,nulls);context->PSSetConstantBuffers(4,1,&nullcb);
  result out;out.pixels=readTexture(output.Get(),16u);auto actual=readTexture(baseTexture.Get(),4u);out.actualBase.resize(W*H);std::memcpy(out.actualBase.data(),actual.data(),actual.size());
  D3D11_BUFFER_DESC bd{};stateBuffer->GetDesc(&bd);bd.Usage=D3D11_USAGE_STAGING;bd.BindFlags=bd.MiscFlags=bd.StructureByteStride=0;bd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ComPtr<ID3D11Buffer> stage;hr(device->CreateBuffer(&bd,nullptr,&stage));context->CopyResource(stage.Get(),stateBuffer.Get());D3D11_MAPPED_SUBRESOURCE m{};hr(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&m));std::memcpy(out.actualState.data(),m.pData,sizeof(out.actualState));context->Unmap(stage.Get(),0);return out;
 }
 static pixel at(const result &r,UINT x,UINT y){pixel p{};std::memcpy(&p,r.pixels.data()+(std::size_t(y)*W*2u+x)*sizeof(pixel),sizeof(p));return p;}
 void expectBase(const operands &in){auto baseline=in;baseline.oldCB.enabled=0u;const auto b=draw(baseline,0),tested=draw(in,1);EXPECT_EQ(tested.pixels,b.pixels);EXPECT_EQ(tested.actualBase,in.base);EXPECT_EQ(tested.actualState,in.state);}
};
TEST_F(HostSbsCurrentUiCapabilityGpuTest,DisabledAndUnboundOptionalSlotAreByteExactLegacy) {
 operands in;in.currentCB.control[0]=0;const auto old=draw(in,0),disabled=draw(in,1,false,true),unbound=draw(in,1,false,false);EXPECT_EQ(old.pixels,disabled.pixels);EXPECT_EQ(old.pixels,unbound.pixels);
}
TEST_F(HostSbsCurrentUiCapabilityGpuTest,CurrentUiClockCanDifferFromActualRetainedBaseClock) {
 operands in;in.oldCB.enabled=0u;ASSERT_NE(in.currentCB.current[2],in.currentCB.base[0]);const auto probe=draw(in,2);EXPECT_FLOAT_EQ(at(probe,0,0).r,1.0f);EXPECT_FLOAT_EQ(at(probe,0,0).g,0.0f);
 const auto out=draw(in,1);EXPECT_NEAR(at(out,X+2,Y).r,1.0f,2e-6f);EXPECT_NEAR(at(out,W+X-2,Y).r,1.0f,2e-6f);EXPECT_NEAR(at(out,X+6,Y).r,0.0f,2e-6f);EXPECT_NEAR(at(out,W+X-6,Y).r,0.0f,2e-6f);EXPECT_EQ(out.actualBase,in.base);EXPECT_EQ(out.actualState,in.state);
}
TEST_F(HostSbsCurrentUiCapabilityGpuTest,CurrentEmptyOverridesPositiveLegacyMatteWithoutRestampingIt) {
 operands in;in.decision(cui::decision_e::current_empty,0.0f);const auto probe=draw(in,2);EXPECT_FLOAT_EQ(at(probe,0,0).g,1.0f);EXPECT_FLOAT_EQ(at(probe,0,0).r,0.0f);expectBase(in);
}
TEST_F(HostSbsCurrentUiCapabilityGpuTest,UnknownIsNotAuthenticatedEmpty) {
 operands in;in.decision(cui::decision_e::unknown,0.0f);const auto probe=draw(in,2);EXPECT_FLOAT_EQ(at(probe,0,0).r,0.0f);EXPECT_FLOAT_EQ(at(probe,0,0).g,0.0f);expectBase(in);
}
TEST_F(HostSbsCurrentUiCapabilityGpuTest,StaleOwnUiObservationClockRejectedEvenWhenColorsMatch) {
 operands in;in.packet[4]=1000u;const auto probe=draw(in,2);EXPECT_FLOAT_EQ(at(probe,0,0).r,0.0f);expectBase(in);
}
TEST_F(HostSbsCurrentUiCapabilityGpuTest,EveryExpectedIdentityWordMustMatchTheCurrentPacket) {
 for(UINT word=0;word<36u;++word){SCOPED_TRACE(word);operands in;in.packet[8u+word]^=1u;expectBase(in);}
}
TEST_F(HostSbsCurrentUiCapabilityGpuTest,RetainedBaseClockCutAndFieldAreCheckedAgainstActualResources) {
 for(UINT kind=0;kind<4u;++kind){SCOPED_TRACE(kind);operands in;if(kind==0)in.currentCB.base[0]^=1u;if(kind==1)in.currentCB.base[2]^=1u;if(kind==2)in.currentCB.native_field[2]-=1u;if(kind==3)in.currentCB.base[3]=1u;in.decision(cui::decision_e::positive,2.0f/W);expectBase(in);}
}
TEST_F(HostSbsCurrentUiCapabilityGpuTest,MalformedPacketAndUnboundCurrentInputsCannotAdmitUi) {
 for(UINT kind=0;kind<5u;++kind){SCOPED_TRACE(kind);operands in;if(kind==0)in.packet[0]^=1u;if(kind==1)in.packet[3]=0u;if(kind==2)in.packet[44]=1u;if(kind==3)in.packet[6]=0x7fc00000u;if(kind==4)in.packet[7]=1u;expectBase(in);}
 operands in;auto baseline=in;baseline.oldCB.enabled=0;EXPECT_EQ(draw(in,1,false).pixels,draw(baseline,0).pixels);
}
TEST_F(HostSbsCurrentUiCapabilityGpuTest,LiteralOneAndExactCurrentColorAreStillRequiredPerPixel) {
 operands noOne;noOne.currentM[Y*W+X]=2u;expectBase(noOne);operands changed;changed.currentRGB[Y*W+X]={0,0,0,0x3c00u};expectBase(changed);
}
TEST_F(HostSbsCurrentUiCapabilityGpuTest,InvalidBaseAuthorizationKeepsExistingFlatFallback) {
 operands in;in.state[v2::renderer_authorization_bits]=0u;expectBase(in);
}
} //namespace
#endif
