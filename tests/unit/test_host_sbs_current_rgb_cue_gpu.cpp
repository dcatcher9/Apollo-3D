/** EXTERNAL current-only cue parity fixture. No live wire, model execution or drawing authority. */
#include <gtest/gtest.h>
#ifdef _WIN32
#include "src/crypto.h"
#include "src/host_sbs_shader_cache.h"
#include "src/host_sbs_current_rgb_cue_gpu.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dxgi.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace cue_gpu = models::host_sbs_current_rgb_cue_gpu_v1;
  using words24 = std::array<UINT,24u>;
  std::vector<char> cue_bytes(const std::filesystem::path &p) {
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()<0) throw std::runtime_error("Cannot read cue binding: "+p.string());
    std::vector<char> v(static_cast<std::size_t>(f.tellg()));f.seekg(0);
    if(!v.empty()) f.read(v.data(),std::streamsize(v.size()));
    if(!f) throw std::runtime_error("Incomplete cue binding");return v;
  }
  std::string cue_sha(const void *data,std::size_t size) {
    const auto digest=crypto::hash(std::string_view(static_cast<const char *>(data),size));
    constexpr char digits[]="0123456789abcdef";std::string s;
    for(auto b:digest) {s.push_back(digits[b>>4u]);s.push_back(digits[b&15u]);}return s;
  }
  template<class T> std::vector<char> cue_raw(const std::vector<T> &v) {
    std::vector<char> b(v.size()*sizeof(T));if(!b.empty()) std::memcpy(b.data(),v.data(),b.size());return b;
  }
  template<std::size_t N> std::vector<char> cue_raw(const std::array<UINT,N> &v) {
    std::vector<char> b(N*4u);std::memcpy(b.data(),v.data(),b.size());return b;
  }
  std::vector<UINT> cue_words(const std::vector<char> &b) {
    if(b.size()%4u) throw std::runtime_error("Cue word extent mismatch");
    std::vector<UINT> w(b.size()/4u);if(!b.empty()) std::memcpy(w.data(),b.data(),b.size());return w;
  }
  std::vector<char> cue_bound(const nlohmann::json &b,std::size_t count) {
    auto v=cue_bytes(b.at("path").get<std::string>());
    if(v.size()!=count || b.at("bytes").get<std::size_t>()!=count || cue_sha(v.data(),v.size())!=b.at("sha256").get<std::string>()) throw std::runtime_error("Cue bound bytes/hash mismatch");return v;
  }
  nlohmann::json cue_save(const std::filesystem::path &p,const std::vector<char> &b) {
    if(std::filesystem::exists(p)) throw std::runtime_error("Refusing cue artifact overwrite");
    std::ofstream f(p,std::ios::binary);if(!b.empty()) f.write(b.data(),std::streamsize(b.size()));
    if(!f) throw std::runtime_error("Cue artifact write failed");
    return {{"path",p.string()},{"bytes",b.size()},{"sha256",cue_sha(b.data(),b.size())}};
  }

  class HostSbsCurrentRgbCueGpuTest:public ::testing::Test {
  protected:
    void SetUp() override {
      const bool saved=std::string(::testing::UnitTest::GetInstance()->current_test_info()->name())=="SavedCurrentRgbCueParity";
      std::string cue_pin,graph_pin;
      if(saved) {
        const auto *p=std::getenv("SUNSHINE_TEST_CURRENT_RGB_CUE_CASES");
        if(!p) {GTEST_SKIP()<<"Explicit current-only cue source cards required";}
        spec_path_=p;spec_bytes_=cue_bytes(p);spec_=nlohmann::json::parse(spec_bytes_.begin(),spec_bytes_.end());
        ASSERT_EQ(spec_.at("schema"),1u);const auto base=spec_path_.parent_path();
        cue_root_=base/spec_.at("cue_shader_root").get<std::string>();graph_root_=base/spec_.at("component_shader_root").get<std::string>();
        cue_pin=spec_.at("cue_source_closure_sha256").get<std::string>();graph_pin=spec_.at("component_source_closure_sha256").get<std::string>();driver_=spec_.value("driver",std::string("warp"));
      } else {
        const auto *cp=std::getenv("SUNSHINE_TEST_CURRENT_RGB_CUE_PIN"),*gp=std::getenv("SUNSHINE_TEST_CURRENT_RGB_CUE_COMPONENT_PIN");
        const auto *cr=std::getenv("SUNSHINE_TEST_CURRENT_RGB_CUE_ROOT"),*gr=std::getenv("SUNSHINE_TEST_CURRENT_RGB_CUE_COMPONENT_ROOT");
        if(!cp || !gp || !cr || !gr) {GTEST_SKIP()<<"Explicit frozen cue/component roots+pins required";}
        cue_pin=cp;graph_pin=gp;cue_root_=cr;graph_root_=gr;
        if(const auto *d=std::getenv("SUNSHINE_TEST_CURRENT_RGB_CUE_DRIVER")) driver_=d;
      }
      ASSERT_EQ(cue_pin.size(),64u);ASSERT_EQ(graph_pin,"7e2f44690d64cebe3261133a3642fa558c8c25c7086bf19551f284a4cb13ebe0");ASSERT_TRUE(driver_=="warp" || driver_=="hardware");
      const D3D_FEATURE_LEVEL wanted[]{D3D_FEATURE_LEVEL_11_0};D3D_FEATURE_LEVEL got {};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr,driver_=="hardware"?D3D_DRIVER_TYPE_HARDWARE:D3D_DRIVER_TYPE_WARP,nullptr,0u,wanted,1u,D3D11_SDK_VERSION,&device_,&got,&context_)));ASSERT_EQ(got,D3D_FEATURE_LEVEL_11_0);
      namespace cache=models::host_sbs_shader_cache;
      std::vector<cache::shader_spec> cs;for(const auto *entry:cue_gpu::cue_entries) cs.push_back({"host_sbs_current_rgb_cue_cs.hlsl",entry,"cs_5_0"});
      const std::array gs {
        cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","init_main","cs_5_0"},
        cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","union_main","cs_5_0"},
        cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","reduce_main","cs_5_0"},
        cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","resolve_main","cs_5_0"},
        cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","dilate_main","cs_5_0"},
        cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","dilate_box_main","cs_5_0"}};
      const auto csrc=cache::snapshot_sources(cue_root_,cs),gsrc=cache::snapshot_sources(graph_root_,gs);
      cue_closure_=cache::source_closure_sha256(csrc);graph_closure_=cache::source_closure_sha256(gsrc);
      ASSERT_EQ(cue_closure_,cue_pin);ASSERT_EQ(graph_closure_,graph_pin);
      for(std::size_t i=0u;i<cs.size();++i) {const auto b=cache::get(csrc,cs[i]);ASSERT_TRUE(b && !b->empty());ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(b->data(),b->size(),nullptr,&cue_[i])));}
      for(std::size_t i=0u;i<graph_.size();++i) {const auto b=cache::get(gsrc,gs[i]);ASSERT_TRUE(b && !b->empty());ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(b->data(),b->size(),nullptr,&graph_[i])));}
      ASSERT_TRUE(resize(160u,30u));author(false);
    }
    void TearDown() override {if(context_) context_->ClearState();}
    bool resize(UINT w,UINT h) {
      if(context_) context_->ClearState();width_=w;height_=h;
      std::string error;if(!resources_.ensure(device_.Get(),w,h,error)) {ADD_FAILURE()<<error;return false;}
      D3D11_TEXTURE2D_DESC t {};t.Width=w;t.Height=h;t.MipLevels=t.ArraySize=1u;t.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;t.SampleDesc.Count=1u;t.Usage=D3D11_USAGE_DEFAULT;t.BindFlags=D3D11_BIND_SHADER_RESOURCE;
      rgb_.Reset();rgb_srv_.Reset();if(FAILED(device_->CreateTexture2D(&t,nullptr,&rgb_)) || FAILED(device_->CreateShaderResourceView(rgb_.Get(),nullptr,&rgb_srv_))) return false;
      D3D11_BUFFER_DESC b {};b.ByteWidth=96u;b.Usage=D3D11_USAGE_DEFAULT;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
      current_cb_.Reset();old_cb_.Reset();if(FAILED(device_->CreateBuffer(&b,nullptr,&current_cb_)) || FAILED(device_->CreateBuffer(&b,nullptr,&old_cb_))) return false;
      b.ByteWidth=1024u;b.BindFlags=D3D11_BIND_SHADER_RESOURCE;b.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;b.StructureByteStride=4u;
      ocr_.Reset();ocr_srv_.Reset();return SUCCEEDED(device_->CreateBuffer(&b,nullptr,&ocr_)) && SUCCEEDED(device_->CreateShaderResourceView(ocr_.Get(),nullptr,&ocr_srv_));
    }
    void author(bool prior) {
      current_={width_,180u,180u-height_,height_,960u,544u,400u,540u,0u,0u,960u,544u,44u,0u,9u,0u,900000u,0u,0u,0u,1u,27u,0u,0u};
      old_={};old_ocr_.assign(256u,0u);prior_=prior;
      if(prior) {
        old_=current_;old_[12u]=13u;old_[16u]=33334u;
        old_ocr_[0u]=4u;old_ocr_[1u]=0x3852434fu;old_ocr_[2u]=1u;old_ocr_[3u]=old_ocr_[4u]=1u;
        std::copy(old_.begin()+12u,old_.begin()+16u,old_ocr_.begin()+5u);old_ocr_[9u]=old_[0u];old_ocr_[10u]=old_[1u];std::copy(old_.begin()+4u,old_.begin()+8u,old_ocr_.begin()+11u);
        const std::array<UINT,4u> core {120u,460u,840u,530u},cover {60u,420u,900u,544u};
        std::copy(core.begin(),core.end(),old_ocr_.begin()+16u);std::copy(cover.begin(),cover.end(),old_ocr_.begin()+144u);
      }
      source_.resize(std::size_t(width_)*height_*8u);
      // Exact binary half constants (0.25,alpha1), authored fixture pixels; no UNORM conversion.
      const std::array<std::uint16_t,4u> p {0x3400u,0x3400u,0x3400u,0x3c00u};
      for(std::size_t i=0u;i<source_.size();i+=8u) std::memcpy(source_.data()+i,p.data(),8u);
    }
    std::vector<char> read(ID3D11Resource *resource) {
      ComPtr<ID3D11Texture2D> t;
      if(SUCCEEDED(resource->QueryInterface(IID_PPV_ARGS(&t)))) {
        D3D11_TEXTURE2D_DESC d {};t->GetDesc(&d);const UINT pixel=d.Format==DXGI_FORMAT_R16G16B16A16_FLOAT?8u:4u;
        d.Usage=D3D11_USAGE_STAGING;d.BindFlags=d.MiscFlags=0u;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> s;if(FAILED(device_->CreateTexture2D(&d,nullptr,&s))) throw std::runtime_error("Cue texture staging failed");context_->CopyResource(s.Get(),resource);
        D3D11_MAPPED_SUBRESOURCE m {};if(FAILED(context_->Map(s.Get(),0u,D3D11_MAP_READ,0u,&m))) throw std::runtime_error("Cue texture Map failed");
        std::vector<char> out(std::size_t(d.Width)*d.Height*pixel);for(UINT y=0u;y<d.Height;++y) std::memcpy(out.data()+std::size_t(y)*d.Width*pixel,static_cast<const char *>(m.pData)+std::size_t(y)*m.RowPitch,std::size_t(d.Width)*pixel);
        context_->Unmap(s.Get(),0u);return out;
      }
      ComPtr<ID3D11Buffer> b;if(FAILED(resource->QueryInterface(IID_PPV_ARGS(&b)))) throw std::runtime_error("Cue readback not texture/buffer");
      D3D11_BUFFER_DESC d {};b->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;d.BindFlags=d.MiscFlags=d.StructureByteStride=0u;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Buffer> s;if(FAILED(device_->CreateBuffer(&d,nullptr,&s))) throw std::runtime_error("Cue buffer staging failed");context_->CopyResource(s.Get(),resource);
      D3D11_MAPPED_SUBRESOURCE m {};if(FAILED(context_->Map(s.Get(),0u,D3D11_MAP_READ,0u,&m))) throw std::runtime_error("Cue buffer Map failed");
      std::vector<char> out(d.ByteWidth);std::memcpy(out.data(),m.pData,out.size());context_->Unmap(s.Get(),0u);return out;
    }
    std::vector<char> read(ID3D11ShaderResourceView *v) {ComPtr<ID3D11Resource> r;v->GetResource(&r);return read(r.Get());}
    bool run() {
      context_->UpdateSubresource(rgb_.Get(),0u,nullptr,source_.data(),width_*8u,0u);
      context_->UpdateSubresource(current_cb_.Get(),0u,nullptr,current_.data(),0u,0u);context_->UpdateSubresource(old_cb_.Get(),0u,nullptr,old_.data(),0u,0u);context_->UpdateSubresource(ocr_.Get(),0u,nullptr,old_ocr_.data(),0u,0u);
      std::array<ID3D11ComputeShader *,20u> c {};std::array<ID3D11ComputeShader *,3u> g {};for(std::size_t i=0u;i<c.size();++i)c[i]=cue_[i].Get();for(std::size_t i=0u;i<g.size();++i)g[i]=graph_[i].Get();
      std::string error;if(!resources_.dispatch(context_.Get(),c,g,current_cb_.Get(),old_cb_.Get(),rgb_srv_.Get(),ocr_srv_.Get(),prior_,error)) {ADD_FAILURE()<<error;return false;}
      const auto v=resources_.mutable_views();saved_.clear();
      saved_["uploaded_current_rgb.rgba16f"]=read(rgb_.Get());saved_["uploaded_current_constants.u32"]=read(current_cb_.Get());saved_["uploaded_prior_constants.u32"]=read(old_cb_.Get());saved_["uploaded_prior_ocr.u32"]=read(ocr_.Get());
      saved_["body.u32"]=read(v.body.Get());saved_["matte.u32"]=read(v.matte.Get());saved_["receipt.u32"]=read(v.receipt.Get());saved_["last_unit.u32"]=read(v.unit.Get());saved_["last_arguments.u32"]=read(v.unit_arguments.Get());
      saved_["last_parent.u32"]=read(v.parent.Get());saved_["last_root_flags.u32"]=read(v.root_flags.Get());
      saved_["last_body_unit_bits.u32"]=read(v.body_unit_bits.Get());saved_["last_cover_unit_bits.u32"]=read(v.cover_unit_bits.Get());
      return true;
    }
    void expect_invariants() {
      EXPECT_EQ(saved_.at("uploaded_current_rgb.rgba16f"),source_);EXPECT_EQ(saved_.at("uploaded_current_constants.u32"),cue_raw(current_));EXPECT_EQ(saved_.at("uploaded_prior_constants.u32"),cue_raw(old_));EXPECT_EQ(saved_.at("uploaded_prior_ocr.u32"),cue_raw(old_ocr_));
      const auto receipt=cue_words(saved_.at("receipt.u32"));ASSERT_EQ(receipt.size(),512u);EXPECT_EQ(receipt[0u],0x31475243u);EXPECT_EQ(receipt[1u],1u);EXPECT_EQ(receipt[2u],0u);
      EXPECT_TRUE(std::equal(current_.begin(),current_.end(),receipt.begin()+16u));EXPECT_TRUE(std::equal(old_.begin(),old_.end(),receipt.begin()+40u));
      const auto b=cue_words(saved_.at("body.u32")),m=cue_words(saved_.at("matte.u32"));ASSERT_EQ(b.size(),std::size_t(width_)*height_);ASSERT_EQ(m.size(),b.size());
      for(std::size_t i=0u;i<b.size();++i) {ASSERT_LE(b[i],1u);ASSERT_LE(m[i],1u);ASSERT_LE(b[i],m[i]);if(receipt[3u]) {ASSERT_EQ(b[i],0u);ASSERT_EQ(m[i],0u);}}
      EXPECT_EQ(receipt[8u],UINT(std::count(b.begin(),b.end(),1u)));EXPECT_EQ(receipt[9u],UINT(std::count(m.begin(),m.end(),1u)));
    }
    void expect_zero() {for(const auto *key:{"body.u32","matte.u32"}) {const auto w=cue_words(saved_.at(key));EXPECT_EQ(std::count(w.begin(),w.end(),1u),0);}}
    nlohmann::json adapter() {
      ComPtr<IDXGIDevice> d;ComPtr<IDXGIAdapter> a;DXGI_ADAPTER_DESC x {};if(FAILED(device_.As(&d)) || FAILED(d->GetAdapter(&a)) || FAILED(a->GetDesc(&x))) return nullptr;
      const std::wstring s(x.Description);return {{"description",std::string(s.begin(),s.end())},{"vendor_id",x.VendorId},{"device_id",x.DeviceId},{"luid_low",x.AdapterLuid.LowPart},{"luid_high",x.AdapterLuid.HighPart}};
    }
    cue_gpu::resources_t resources_;ComPtr<ID3D11Device> device_;ComPtr<ID3D11DeviceContext> context_;
    std::array<ComPtr<ID3D11ComputeShader>,20u> cue_;std::array<ComPtr<ID3D11ComputeShader>,3u> graph_;
    ComPtr<ID3D11Texture2D> rgb_;ComPtr<ID3D11ShaderResourceView> rgb_srv_,ocr_srv_;ComPtr<ID3D11Buffer> ocr_,current_cb_,old_cb_;
    words24 current_ {},old_ {};std::vector<UINT> old_ocr_;std::vector<char> source_;bool prior_ {};UINT width_ {},height_ {};
    std::map<std::string,std::vector<char>> saved_;nlohmann::json spec_;std::vector<char> spec_bytes_;
    std::filesystem::path spec_path_,cue_root_,graph_root_;std::string cue_closure_,graph_closure_,driver_ {"warp"};
  };

  TEST_F(HostSbsCurrentRgbCueGpuTest, NoPriorIsFreshUnknownNotEmpty) {
    author(false);ASSERT_TRUE(run());expect_invariants();expect_zero();
    EXPECT_EQ(cue_words(saved_.at("receipt.u32"))[2u],0u);
  }
  TEST_F(HostSbsCurrentRgbCueGpuTest, UniformCurrentColourDoesNotFillHistoricalRectangle) {
    author(true);ASSERT_TRUE(run());expect_invariants();expect_zero();
  }
  TEST_F(HostSbsCurrentRgbCueGpuTest, PriorPrefixAuthenticatesItsOwnClockNotCurrentClock) {
    author(true);ASSERT_NE(old_[12u],current_[12u]);ASSERT_NE(old_[16u],current_[16u]);ASSERT_TRUE(run());expect_invariants();
    auto r=cue_words(saved_.at("receipt.u32"));EXPECT_EQ(r[3u],0u);
    ++old_ocr_[5u];ASSERT_TRUE(run());expect_invariants();expect_zero();EXPECT_NE(cue_words(saved_.at("receipt.u32"))[3u],0u);
  }
  TEST_F(HostSbsCurrentRgbCueGpuTest, FarPixelNumericFaultCannotPublishAnyUnit) {
    author(true);const std::uint16_t nan=0x7e00u;std::memcpy(source_.data()+source_.size()-8u,&nan,2u);
    ASSERT_TRUE(run());expect_invariants();expect_zero();EXPECT_NE(cue_words(saved_.at("receipt.u32"))[3u],0u);
  }
  TEST_F(HostSbsCurrentRgbCueGpuTest, CanonicalNoPriorRejectsNonzeroOldPacket) {
    author(false);old_ocr_[255u]=1u;ASSERT_TRUE(run());expect_invariants();expect_zero();EXPECT_NE(cue_words(saved_.at("receipt.u32"))[3u],0u);
  }
  TEST_F(HostSbsCurrentRgbCueGpuTest, FreshUnknownClearsPoisonedOldBodyAndMatte) {
    author(false);const auto v=resources_.mutable_views();std::vector<UINT> poison(std::size_t(width_)*height_,1u);
    for(auto *s:{v.body.Get(),v.matte.Get()}) {ComPtr<ID3D11Resource> r;s->GetResource(&r);context_->UpdateSubresource(r.Get(),0u,nullptr,poison.data(),width_*4u,0u);}
    ASSERT_TRUE(run());expect_invariants();expect_zero();
  }
  TEST_F(HostSbsCurrentRgbCueGpuTest, StableAllocationAndTouchedCsBindingsAreRestored) {
    const auto before=resources_.mutable_views();std::string e;ASSERT_TRUE(resources_.ensure(device_.Get(),width_,height_,e));EXPECT_EQ(before.body.Get(),resources_.mutable_views().body.Get());
    ID3D11Buffer *cb[3] {current_cb_.Get(),old_cb_.Get(),current_cb_.Get()};context_->CSSetConstantBuffers(0u,3u,cb);context_->CSSetShader(cue_[0u].Get(),nullptr,0u);
    ASSERT_TRUE(run());std::array<ID3D11Buffer *,3u> after {};context_->CSGetConstantBuffers(0u,3u,after.data());
    for(std::size_t i=0u;i<after.size();++i) {EXPECT_EQ(after[i],cb[i]);if(after[i]) after[i]->Release();}
    ID3D11ComputeShader *shader=nullptr;context_->CSGetShader(&shader,nullptr,nullptr);EXPECT_EQ(shader,cue_[0u].Get());if(shader)shader->Release();
  }
  TEST_F(HostSbsCurrentRgbCueGpuTest, ActivePredicationRejectsBeforePrivateClearOrCommands) {
    const auto v=resources_.mutable_views();std::vector<UINT> poison(std::size_t(width_)*height_,1u);
    ComPtr<ID3D11Resource> out;v.matte->GetResource(&out);context_->UpdateSubresource(out.Get(),0u,nullptr,poison.data(),width_*4u,0u);
    D3D11_QUERY_DESC d {};d.Query=D3D11_QUERY_OCCLUSION_PREDICATE;ComPtr<ID3D11Predicate> p;ASSERT_TRUE(SUCCEEDED(device_->CreatePredicate(&d,&p)));
    context_->SetPredication(p.Get(),FALSE);
    std::array<ID3D11ComputeShader *,20u> c {};std::array<ID3D11ComputeShader *,3u> g {};
    for(std::size_t i=0u;i<c.size();++i)c[i]=cue_[i].Get();for(std::size_t i=0u;i<g.size();++i)g[i]=graph_[i].Get();
    std::string error;EXPECT_FALSE(resources_.dispatch(context_.Get(),c,g,current_cb_.Get(),old_cb_.Get(),rgb_srv_.Get(),ocr_srv_.Get(),false,error));EXPECT_FALSE(error.empty());
    context_->SetPredication(nullptr,FALSE);EXPECT_EQ(read(out.Get()),cue_raw(poison));
  }
  TEST_F(HostSbsCurrentRgbCueGpuTest, FinalGpuFaultVetoClearsEarlierUnitBitsAndOldOutput) {
    // Mechanical finalizer input only, not authored glyph detection evidence.
    const auto v=resources_.mutable_views();std::vector<UINT> ones(std::size_t(width_)*height_,1u);
    for(auto *s:{v.body.Get(),v.matte.Get(),v.body_unit_bits.Get(),v.cover_unit_bits.Get()}) {
      ComPtr<ID3D11Resource> r;s->GetResource(&r);context_->UpdateSubresource(r.Get(),0u,nullptr,ones.data(),width_*4u,0u);
    }
    std::array<UINT,512u> state {};state[0u]=0x31475243u;state[1u]=1u;state[3u]=4u;state[4u]=state[5u]=state[7u]=1u;
    ComPtr<ID3D11Resource> receipt;v.receipt->GetResource(&receipt);context_->UpdateSubresource(receipt.Get(),0u,nullptr,state.data(),0u,0u);
    context_->UpdateSubresource(current_cb_.Get(),0u,nullptr,current_.data(),0u,0u);ID3D11Buffer *cb=current_cb_.Get();context_->CSSetConstantBuffers(0u,1u,&cb);
    const auto uav=[&](ID3D11ShaderResourceView *srv) {ComPtr<ID3D11Resource> r;srv->GetResource(&r);ComPtr<ID3D11UnorderedAccessView> u;if(FAILED(device_->CreateUnorderedAccessView(r.Get(),nullptr,&u))) throw std::runtime_error("Finalizer fixture UAV creation failed");return u;};
    const auto frame=uav(v.receipt.Get()),body=uav(v.body.Get()),matte=uav(v.matte.Get());
    const std::array<ID3D11UnorderedAccessView *,8u> w {nullptr,frame.Get(),nullptr,nullptr,body.Get(),matte.Get(),nullptr,nullptr};
    const std::array<ID3D11ShaderResourceView *,9u> r {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,v.body_unit_bits.Get(),v.cover_unit_bits.Get()};
    context_->CSSetShaderResources(0u,UINT(r.size()),r.data());context_->CSSetUnorderedAccessViews(0u,UINT(w.size()),w.data(),nullptr);context_->CSSetShader(cue_[19u].Get(),nullptr,0u);context_->Dispatch((width_+15u)/16u,(height_+15u)/16u,1u);
    context_->ClearState();const auto b=cue_words(read(v.body.Get())),m=cue_words(read(v.matte.Get()));EXPECT_EQ(std::count(b.begin(),b.end(),1u),0);EXPECT_EQ(std::count(m.begin(),m.end(),1u),0);
  }

  TEST_F(HostSbsCurrentRgbCueGpuTest, SavedCurrentRgbCueParity) {
    const auto *out=std::getenv("SUNSHINE_TEST_CURRENT_RGB_CUE_OUTPUT");if(!out) {GTEST_SKIP()<<"Explicit new output directory required";}
    const auto &cases=spec_.at("cases");ASSERT_GT(cases.size(),0u);ASSERT_LE(cases.size(),64u);
    const std::filesystem::path destination(out);ASSERT_FALSE(std::filesystem::exists(destination));ASSERT_TRUE(std::filesystem::create_directories(destination));
    const auto input_sha=cue_sha(spec_bytes_.data(),spec_bytes_.size());nlohmann::json executable=nullptr;
    std::array<wchar_t,32768u> path {};const auto len=GetModuleFileNameW(nullptr,path.data(),DWORD(path.size()));
    if(len>0u && len<path.size()) {const std::filesystem::path p(std::wstring(path.data(),len));const auto b=cue_bytes(p);executable={{"path",p.string()},{"sha256",cue_sha(b.data(),b.size())},{"bytes",b.size()}};}
    nlohmann::json aggregate {{"schema",1u},{"driver",driver_},{"adapter",adapter()},{"input_specification_sha256",input_sha},{"cue_source_closure_sha256",cue_closure_},{"component_source_closure_sha256",graph_closure_},{"test_executable",executable},{"cases",nlohmann::json::array()}};
    for(std::size_t i=0u;i<cases.size();++i) {
      const auto &item=cases.at(i);SCOPED_TRACE(item.at("name").get<std::string>());
      const UINT w=item.at("width"),h=item.at("height");ASSERT_TRUE(resize(w,h));const auto &a=item.at("artifacts");
      source_=cue_bound(a.at("current_rgb"),std::size_t(w)*h*8u);
      const auto cc=cue_words(cue_bound(a.at("current_constants"),96u)),oc=cue_words(cue_bound(a.at("prior_constants"),96u));
      std::copy(cc.begin(),cc.end(),current_.begin());std::copy(oc.begin(),oc.end(),old_.begin());old_ocr_=cue_words(cue_bound(a.at("prior_ocr"),1024u));
      prior_=item.at("prior_present").get<bool>();ASSERT_TRUE(run());expect_invariants();
      bool expected=true;const bool supplied=item.contains("expected");
      if(supplied) {
        const auto &e=item.at("expected");expected &= saved_.at("body.u32")==cue_bound(e.at("body"),std::size_t(w)*h*4u);
        expected &= saved_.at("matte.u32")==cue_bound(e.at("matte"),std::size_t(w)*h*4u);
        if(e.contains("receipt")) expected &= saved_.at("receipt.u32")==cue_bound(e.at("receipt"),2048u);
      }
      const auto folder=destination/("case_"+std::to_string(i));ASSERT_TRUE(std::filesystem::create_directory(folder));nlohmann::json artifacts=nlohmann::json::object();
      for(const auto &[name,bytes]:saved_) artifacts[name]=cue_save(folder/name,bytes);
      nlohmann::json identity {{"schema",1u},{"case",item},{"driver",driver_},{"adapter",adapter()},{"test_executable",executable},{"input_specification_sha256",input_sha},
        {"cue_shader_root",cue_root_.string()},{"component_shader_root",graph_root_.string()},{"cue_source_closure_sha256",cue_closure_},{"component_source_closure_sha256",graph_closure_},{"artifacts",artifacts},
        {"uploaded_four_operands_byte_exact",saved_.at("uploaded_current_rgb.rgba16f")==source_ && saved_.at("uploaded_current_constants.u32")==cue_raw(current_) && saved_.at("uploaded_prior_constants.u32")==cue_raw(old_) && saved_.at("uploaded_prior_ocr.u32")==cue_raw(old_ocr_)},
        {"expected_spatial_supplied",supplied},{"expected_spatial_byte_exact",supplied?nlohmann::json(expected):nlohmann::json(nullptr)},
        {"semantic_status","UNKNOWN"},{"publication_scope","Provisional spatial Body/M only; no CurrentUI kind2, positive/EMPTY semantic certificate, live estimator path, targetP/S/OCR, CPU modal decisions or speed claim"},
        {"diagnostic_scope","last_* graph/unit buffers may hold inactive last-slot scratch; they are diagnostic, not fresh per-unit publication or authority"}};
      const auto text=identity.dump(2)+"\n";std::vector<char> bytes(text.begin(),text.end());const auto bound=cue_save(folder/"current_rgb_cue_fixture_identity.json",bytes);
      aggregate["cases"].push_back({{"index",i},{"name",item.at("name")},{"identity",bound},{"expected_spatial_byte_exact",identity["expected_spatial_byte_exact"]}});
      const auto report=aggregate.dump(2)+"\n";std::ofstream f(destination/"current_rgb_cue_replay_identity.json",std::ios::binary|std::ios::trunc);f<<report;ASSERT_TRUE(f.good());f.close();
      EXPECT_TRUE(expected)<<"Frozen CPU V4 spatial mismatch; actual native bytes preserved, no acceptance waiver";
    }
  }
}
#endif
