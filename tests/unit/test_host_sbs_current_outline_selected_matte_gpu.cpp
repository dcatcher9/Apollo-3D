#include <gtest/gtest.h>
#ifdef _WIN32
  #include "src/host_sbs_current_outline_runtime_programs.h"
  #include <algorithm>
  #include <array>
  #include <cstdlib>
  #include <cstring>
  #include <filesystem>
  #include <stdexcept>
  #include <vector>

namespace {
  namespace selected = models::research_current_outline_selected_matte_v2;
  namespace runtime = models::current_outline_runtime;
  namespace cache = models::host_sbs_shader_cache;
  using Microsoft::WRL::ComPtr;

  class HostSbsCurrentOutlineSelectedMatteGpuTest : public testing::Test {
  protected:
    static constexpr UINT W=8, FullH=6, Top=2, H=4, N=W*H;
    static constexpr std::uint64_t Token=(std::uint64_t(9)<<32u)|71u;
    struct buffer_t {ComPtr<ID3D11Buffer> b;ComPtr<ID3D11ShaderResourceView> s;};
    struct image_t {ComPtr<ID3D11Texture2D> t;ComPtr<ID3D11ShaderResourceView> s;};
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
    std::array<ComPtr<ID3D11ComputeShader>,2> shaders;
    selected::resources_t output;
    buffer_t cb,root,sct,numeric,envelope,cbc,raw,f2;
    image_t legacy,baseline,current;
    std::array<UINT,24> c{};
    std::array<UINT,64> root_words{};
    std::array<UINT,32> identity{},numeric_words{};
    std::array<UINT,104> source{};
    std::array<UINT,512> enclosure{},ring{};
    std::vector<UINT> band=std::vector<UINT>(68608u),lm=std::vector<UINT>(N),bm=std::vector<UINT>(N),cm=std::vector<UINT>(N);

    static void require(bool ok) {if(!ok)throw std::runtime_error("Current-outline selector fixture API failed");}
    buffer_t buffer(const void *bytes,UINT size,bool raw_view=false,bool indirect=false,bool constant=false) {
      buffer_t result;D3D11_BUFFER_DESC d{};d.ByteWidth=size;d.Usage=D3D11_USAGE_DEFAULT;
      d.BindFlags=constant?D3D11_BIND_CONSTANT_BUFFER:D3D11_BIND_SHADER_RESOURCE;
      d.MiscFlags=constant?0u:raw_view?D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS:D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      d.StructureByteStride=constant||raw_view?0u:4u;if(indirect)d.MiscFlags|=D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_SUBRESOURCE_DATA init{bytes,0,0};require(SUCCEEDED(device->CreateBuffer(&d,&init,&result.b)));
      if(!constant){D3D11_SHADER_RESOURCE_VIEW_DESC v{};v.ViewDimension=raw_view?D3D11_SRV_DIMENSION_BUFFEREX:D3D11_SRV_DIMENSION_BUFFER;v.Format=raw_view?DXGI_FORMAT_R32_TYPELESS:DXGI_FORMAT_UNKNOWN;
        if(raw_view){v.BufferEx.NumElements=size/4u;v.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;}else v.Buffer.NumElements=size/4u;
        require(SUCCEEDED(device->CreateShaderResourceView(result.b.Get(),&v,&result.s)));}
      return result;
    }
    image_t image(const std::vector<UINT>&pixels) {
      image_t result;D3D11_TEXTURE2D_DESC d{};d.Width=W;d.Height=H;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
      d.Format=DXGI_FORMAT_R32_UINT;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA init{pixels.data(),W*4u,0};require(SUCCEEDED(device->CreateTexture2D(&d,&init,&result.t)));
      require(SUCCEEDED(device->CreateShaderResourceView(result.t.Get(),nullptr,&result.s)));return result;
    }
    std::vector<UINT> read(ID3D11ShaderResourceView*view) {
      ComPtr<ID3D11Resource> r,stage;view->GetResource(&r);D3D11_RESOURCE_DIMENSION kind;r->GetType(&kind);UINT bytes=0,rows=1;
      if(kind==D3D11_RESOURCE_DIMENSION_BUFFER){ComPtr<ID3D11Buffer>b;r.As(&b);D3D11_BUFFER_DESC d{};b->GetDesc(&d);bytes=d.ByteWidth;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;d.BindFlags=d.MiscFlags=d.StructureByteStride=0;ComPtr<ID3D11Buffer>x;require(SUCCEEDED(device->CreateBuffer(&d,nullptr,&x)));stage=x;}
      else{ComPtr<ID3D11Texture2D>t;r.As(&t);D3D11_TEXTURE2D_DESC d{};t->GetDesc(&d);bytes=d.Width*4u;rows=d.Height;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;d.BindFlags=d.MiscFlags=0;ComPtr<ID3D11Texture2D>x;require(SUCCEEDED(device->CreateTexture2D(&d,nullptr,&x)));stage=x;}
      context->CopyResource(stage.Get(),r.Get());D3D11_MAPPED_SUBRESOURCE m{};require(SUCCEEDED(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&m)));
      std::vector<UINT> result(bytes*rows/4u);for(UINT y=0;y<rows;++y)std::memcpy(reinterpret_cast<char*>(result.data())+y*bytes,static_cast<const char*>(m.pData)+y*(kind==D3D11_RESOURCE_DIMENSION_BUFFER?bytes:m.RowPitch),bytes);context->Unmap(stage.Get(),0);return result;
    }
    void set_root(UINT mode) {
      root_words={};root_words[0]=1;root_words[1]=1u^0xd1ec15a5u^0x52434f4fu;root_words[2]=71;root_words[3]=9;
      root_words[4]=71u^0xa3756c91u;root_words[5]=9u^0x5c8a936eu;root_words[6]=0x47524243u;root_words[7]=0x52434f4fu;
      std::copy_n(root_words.begin()+2,4,root_words.begin()+8);root_words[12]=0x54535152u;root_words[13]=1;root_words[14]=1u^0x6f435257u;
      if(mode!=2){root_words[20]=root_words[21]=root_words[22]=1;}
      root_words[44]=root_words[45]=root_words[46]=1;
      root_words[52]=root_words[53]=root_words[54]=1;root_words[57]=root_words[58]=1;
      root_words[60]=root_words[61]=root_words[62]=1;
    }
    void tuple(UINT frame=3,UINT observation=66667,bool empty=false,UINT mode=2) {
      c={W,FullH,Top,H,4,2,0,2,0,0,4,2,frame,0,7,0,observation,0,0,0,1,27,0,0};set_root(mode);
      identity={5,0x35544353u,empty?0u:1u,2};std::copy_n(c.begin()+12,4,identity.begin()+4);std::copy_n(c.begin(),12,identity.begin()+8);identity[20]=observation;identity[22]=27;identity[23]=1;
      numeric_words={0x31564c49u,1,1,0};std::copy(c.begin(),c.end(),numeric_words.begin()+4);
      enclosure={0x31434e45u,5,1,0};enclosure[4]=enclosure[5]=enclosure[9]=1;std::copy(c.begin(),c.end(),enclosure.begin()+16);
      const std::array<UINT,32> enc{0x31434e45u,1,1,mode,W,H,28,1,W,FullH,28,1,71,9,1,1u^0x6f435257u,frame,0,observation,0,7,0,0,0,0,0,W,FullH,5,0,4,136};std::copy(enc.begin(),enc.end(),enclosure.begin()+40);enclosure[80]=identity[3];
      ring={0x31474e52u,1,1,0};std::copy(c.begin(),c.end(),ring.begin()+12);std::copy_n(enclosure.begin(),16,ring.begin()+36);
      band.assign(68608u,0);band[0]=0x37424643u;band[1]=7;band[2]=1;std::copy(c.begin(),c.end(),band.begin()+8);std::copy_n(ring.begin(),16,band.begin()+32);
      const std::array<UINT,19> layout{W,H,N,68608,64,128,16,0,0,0,0,N,0x3fffffffu,0x80000000u,0x40000000u,255,256,512,1024};std::copy(layout.begin(),layout.end(),band.begin()+48);band[71]=2048;band[74]=4096;band[77]=0x31444e49u;band[78]=1;band[83]=1;
      source={4,0x34504955u,1,empty?0u:1u};std::copy_n(c.begin()+12,4,source.begin()+4);source[8]=observation;source[10]=W;source[11]=FullH;std::copy_n(c.begin()+4,8,source.begin()+12);source[20]=identity[3];
      const std::array<UINT,40> env{0x35534743u,1,5,1,frame,0,observation,0,7,0,71,9,W,FullH,Top,H,0,0,W,FullH,W,FullH,W,FullH,8,4,1,1,1,0,0,63,N,N,0,N,1,1,0,identity[3]};std::copy(env.begin(),env.end(),source.begin()+40);
      std::copy_n(c.begin()+4,8,source.begin()+80);std::copy_n(c.begin()+20,4,source.begin()+88);std::copy_n(c.begin()+16,4,source.begin()+92);std::copy_n(root_words.begin()+60,4,source.begin()+96);source[100]=10;source[101]=28;source[102]=10;source[103]=1;
    }
    void upload() {
      for(auto pair:{std::pair{&cb,c.data()},std::pair{&root,root_words.data()},std::pair{&sct,identity.data()},std::pair{&numeric,numeric_words.data()},std::pair{&envelope,source.data()},std::pair{&cbc,enclosure.data()},std::pair{&raw,ring.data()},std::pair{&f2,band.data()}})context->UpdateSubresource(pair.first->b.Get(),0,nullptr,pair.second,0,0);
      context->UpdateSubresource(legacy.t.Get(),0,nullptr,lm.data(),W*4u,0);context->UpdateSubresource(current.t.Get(),0,nullptr,cm.data(),W*4u,0);
    }
    void queue(UINT mode=2) {
      selected::inputs_t i;i.current_cb96=cb.b.Get();i.actual_root_args=root.b.Get();i.legacy_matte=legacy.s.Get();i.baseline_matte=baseline.s.Get();i.numeric_proof=numeric.s.Get();i.sct=sct.s.Get();i.mode=static_cast<selected::postprocess_mode_e>(mode);i.expected_work=1;i.transaction_token=Token;i.current_outline_requested=true;i.source_envelope104=envelope.s.Get();i.cbc_meta512=cbc.s.Get();i.raw_meta512=raw.s.Get();i.f2_data=f2.s.Get();i.f2_matte=current.s.Get();
      std::string error;ASSERT_TRUE(output.queue(context.Get(),{shaders[0].Get(),shaders[1].Get()},i,error))<<error;
    }
    void expect(UINT kind,const std::vector<UINT>&mask,UINT fault=0) {
      auto v=output.views();EXPECT_EQ(read(v.matte.Get()),mask);auto receipt=read(v.receipt.Get());ASSERT_EQ(receipt.size(),64u);EXPECT_EQ(receipt[0],selected::tag);EXPECT_EQ(receipt[1],1u);EXPECT_EQ(receipt[2],kind);EXPECT_EQ(receipt[3],fault);EXPECT_TRUE(std::equal(c.begin(),c.end(),receipt.begin()+4));
      if(!fault)EXPECT_TRUE(std::equal(identity.begin(),identity.end(),receipt.begin()+28));else EXPECT_TRUE(std::all_of(receipt.begin()+28,receipt.begin()+60,[](UINT v){return v==0;}));EXPECT_EQ(receipt[60],71u);EXPECT_EQ(receipt[61],9u);EXPECT_EQ(receipt[62],2u);EXPECT_EQ(receipt[63],0u);
    }
    void SetUp() override {
      const auto*enabled=std::getenv("SUNSHINE_CURRENT_OUTLINE_SELECTOR_GPU_TEST");if(!enabled||std::string(enabled)!="1")GTEST_SKIP()<<"Explicit current-outline selector GPU opt-in required";
      const auto*driver=std::getenv("SUNSHINE_CURRENT_OUTLINE_SELECTOR_GPU_DRIVER");ASSERT_TRUE(!driver||std::string(driver)=="hardware"||std::string(driver)=="warp");D3D_FEATURE_LEVEL level{};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr,driver&&std::string(driver)=="hardware"?D3D_DRIVER_TYPE_HARDWARE:D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context)));ASSERT_GE(level,D3D_FEATURE_LEVEL_11_0);
      const auto*path=std::getenv("SUNSHINE_CURRENT_OUTLINE_SELECTOR_SHADER_ROOT");auto shader_root=path?std::filesystem::path(path):std::filesystem::path(SUNSHINE_SHADERS_DIR)/"current_outline"/"selector";const auto snapshot=cache::snapshot_sources(shader_root,runtime::selector_specs);ASSERT_TRUE(snapshot);ASSERT_EQ(cache::source_closure_sha256(snapshot),runtime::selector_closure);
      for(UINT k=0;k<2;++k){const auto code=cache::get(snapshot,runtime::selector_specs[k]);ASSERT_TRUE(code);ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(code->data(),code->size(),nullptr,&shaders[k])));}
      tuple();lm[0]=1;bm[1]=1;cm[2]=cm[N-1]=1;legacy=image(lm);baseline=image(bm);current=image(cm);
      cb=buffer(c.data(),sizeof(c),false,false,true);root=buffer(root_words.data(),sizeof(root_words),true,true);sct=buffer(identity.data(),sizeof(identity));numeric=buffer(numeric_words.data(),sizeof(numeric_words));envelope=buffer(source.data(),sizeof(source));cbc=buffer(enclosure.data(),sizeof(enclosure));raw=buffer(ring.data(),sizeof(ring));f2=buffer(band.data(),band.size()*4u,true);
      std::string error;ASSERT_TRUE(output.ensure(device.Get(),W,H,error))<<error;
    }
  };

  TEST_F(HostSbsCurrentOutlineSelectedMatteGpuTest, SelectsCurrentCbc5Cfb7ForDirectAndQueuedOwners) {
    for(UINT mode:{2u,4u}){tuple(3,66667,false,mode);upload();queue(mode);expect(3,cm);}
  }
  TEST_F(HostSbsCurrentOutlineSelectedMatteGpuTest, PrepareRootZeroHoldsWholeOutputAcrossFreshBacking) {
    queue();const auto beforeM=read(output.views().matte.Get()),beforeReceipt=read(output.views().receipt.Get());tuple(4,99999);cm.assign(N,0);root_words[60]=root_words[61]=root_words[62]=0;upload();queue();EXPECT_EQ(read(output.views().matte.Get()),beforeM);EXPECT_EQ(read(output.views().receipt.Get()),beforeReceipt);
  }
  TEST_F(HostSbsCurrentOutlineSelectedMatteGpuTest, FreshEmptyClearsEarlierMaskAndAdvancesIdentity) {
    queue();tuple(4,99999,true);lm.assign(N,0);cm.assign(N,0);upload();queue();expect(3,cm);
  }
  TEST_F(HostSbsCurrentOutlineSelectedMatteGpuTest, StaleAndFaultedCurrentStagesPreserveLiteralLegacy) {
    for(UINT fault=0;fault<5;++fault){tuple();if(fault==0)enclosure[16]^=1u;if(fault==1)ring[0]^=1u;if(fault==2)band[3]=1u;if(fault==3)source[50]^=1u;if(fault==4)band[1]=6u;upload();queue();expect(1,lm);}
  }
  TEST_F(HostSbsCurrentOutlineSelectedMatteGpuTest, InvalidRootCannotGrantCurrentKindButKeepsLegacyContext) {
    root_words[2]^=1u;upload();queue();expect(1,lm);
  }
  TEST_F(HostSbsCurrentOutlineSelectedMatteGpuTest, StaleSctMarksReceiptUnavailableWithoutRelabelingLegacy) {
    identity[20]^=1u;upload();queue();expect(1,lm,1u);
  }
  TEST_F(HostSbsCurrentOutlineSelectedMatteGpuTest, MoveTransfersOwnerEmptiesSourceAndPreservesSelfMove) {
    queue();const auto before=read(output.views().matte.Get());selected::resources_t moved=std::move(output);EXPECT_FALSE(output.views().matte);EXPECT_FALSE(output.views().receipt);EXPECT_EQ(read(moved.views().matte.Get()),before);output=std::move(moved);EXPECT_FALSE(moved.views().matte);EXPECT_FALSE(moved.views().receipt);output=std::move(output);expect(3,cm);queue();expect(3,cm);
  }
}
#endif
