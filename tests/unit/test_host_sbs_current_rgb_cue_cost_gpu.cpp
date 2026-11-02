/** EXTERNAL immediate whole-current-RGB helper cost fixture; not live or pure-kernel performance. */
#include <gtest/gtest.h>
#ifdef _WIN32
#include "src/crypto.h"
#include "src/host_sbs_shader_cache.h"
#include "src/host_sbs_current_rgb_cue_gpu.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <thread>
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

  class HostSbsCurrentRgbCueCostGpuTest:public ::testing::Test {
  protected:
    void SetUp() override {
      const bool saved=std::string(::testing::UnitTest::GetInstance()->current_test_info()->name())=="SavedCurrentRgbCueCost";
      std::string cue_pin,graph_pin;
      if(saved) {
        const auto *p=std::getenv("SUNSHINE_TEST_CURRENT_RGB_CUE_COST_CASES");
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
      ASSERT_EQ(cue_pin,"51c945dddd62e2ad558e2037cab4cb3b2b2b541ca9f1f78bc82f6921585872e1");
      ASSERT_EQ(spec_.at("helper_source_sha256"),"2cbdd4545c1d2c3829e8e3dbc5cb0d82cb4021be161b972d6e1e1aa8862fc9c8");
      ASSERT_EQ(spec_.at("execution"),"IMMEDIATE_DIRECT");
      ASSERT_EQ(spec_.at("warmups"),3u);
      ASSERT_EQ(spec_.at("blocks"),3u);
      ASSERT_EQ(spec_.at("repeats"),20u);
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

    // Inputs are uploaded once outside every timed range; no run()/read()/Map()/save() here.
    nlohmann::json measure_block() {
      constexpr UINT repeats=20u;
      std::array<ID3D11ComputeShader *,20u> c {};
      std::array<ID3D11ComputeShader *,3u> g {};
      for(std::size_t i=0u;i<c.size();++i)c[i]=cue_[i].Get();
      for(std::size_t i=0u;i<g.size();++i)g[i]=graph_[i].Get();
      D3D11_QUERY_DESC d {};d.Query=D3D11_QUERY_TIMESTAMP_DISJOINT;
      ComPtr<ID3D11Query> disjoint;
      if(FAILED(device_->CreateQuery(&d,&disjoint)))throw std::runtime_error("Cost disjoint query allocation failed");
      d.Query=D3D11_QUERY_TIMESTAMP;
      std::array<ComPtr<ID3D11Query>,2u*repeats> queries;
      for(auto &query:queries)if(FAILED(device_->CreateQuery(&d,&query)))throw std::runtime_error("Cost timestamp query allocation failed");
      std::array<std::uint64_t,repeats> cpu_ns {};
      std::string error;
      context_->Begin(disjoint.Get());
      for(UINT i=0u;i<repeats;++i) {
        context_->End(queries[2u*i].Get());
        const auto begin=std::chrono::steady_clock::now();
        const bool ok=resources_.dispatch(context_.Get(),c,g,current_cb_.Get(),old_cb_.Get(),rgb_srv_.Get(),ocr_srv_.Get(),prior_,error);
        const auto end=std::chrono::steady_clock::now();
        context_->End(queries[2u*i+1u].Get());
        cpu_ns[i]=std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        if(!ok)throw std::runtime_error("Timed full helper dispatch rejected: "+error);
      }
      context_->End(disjoint.Get());
      // Submission and GPU query wait are outside all measured CPU-dispatch ranges.
      context_->Flush();
      const auto wait=[&](ID3D11Query *query,void *result,UINT bytes) {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(60);
        HRESULT hr=S_FALSE;
        while((hr=context_->GetData(query,result,bytes,D3D11_ASYNC_GETDATA_DONOTFLUSH))==S_FALSE) {
          if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("GPU cost query timeout");
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if(FAILED(hr))throw std::runtime_error("GPU cost query failed");
      };
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT clock {};
      wait(disjoint.Get(),&clock,sizeof(clock));
      nlohmann::json ticks=nlohmann::json::array(),gpu_ns=nlohmann::json::array();
      for(UINT i=0u;i<repeats;++i) {
        UINT64 begin {},end {};
        wait(queries[2u*i].Get(),&begin,sizeof(begin));
        wait(queries[2u*i+1u].Get(),&end,sizeof(end));
        ticks.push_back({begin,end});
        if(!clock.Disjoint&&clock.Frequency!=0u&&end>=begin)gpu_ns.push_back(double(end-begin)*1e9/double(clock.Frequency));
      }
      const auto summary=[](std::vector<double> values) {
        std::sort(values.begin(),values.end());double sum=0.;for(const auto v:values)sum+=v;
        return nlohmann::json {{"minimum",values.front()},{"median",.5*(values[9u]+values[10u])},{"mean",sum/values.size()},{"maximum",values.back()}};
      };
      nlohmann::json result {{"repeats",repeats},{"timestamp_ticks",ticks},{"frequency_hz",clock.Frequency},{"disjoint",clock.Disjoint!=FALSE},
        {"valid_intervals",gpu_ns.size()},{"gpu_timeline_ns",gpu_ns},{"cpu_submit_ns",cpu_ns},{"cpu_submit_ns_summary",summary(std::vector<double>(cpu_ns.begin(),cpu_ns.end()))}};
      if(gpu_ns.size()==repeats)result["gpu_timeline_ns_summary"]=summary(gpu_ns.get<std::vector<double>>());
      return result;
    }
    bool dispatch_preuploaded() {
      std::array<ID3D11ComputeShader *,20u> c {};
      std::array<ID3D11ComputeShader *,3u> g {};
      for(std::size_t i=0u;i<c.size();++i)c[i]=cue_[i].Get();
      for(std::size_t i=0u;i<g.size();++i)g[i]=graph_[i].Get();
      std::string error;
      if(!resources_.dispatch(context_.Get(),c,g,current_cb_.Get(),old_cb_.Get(),rgb_srv_.Get(),ocr_srv_.Get(),prior_,error)) {
        ADD_FAILURE()<<error;return false;
      }
      return true;
    }
    std::map<std::string,std::vector<char>> read_outputs() {
      const auto views=resources_.mutable_views();
      return {{"body.u32",read(views.body.Get())},{"matte.u32",read(views.matte.Get())},{"receipt.u32",read(views.receipt.Get())}};
    }

    cue_gpu::resources_t resources_;ComPtr<ID3D11Device> device_;ComPtr<ID3D11DeviceContext> context_;
    std::array<ComPtr<ID3D11ComputeShader>,20u> cue_;std::array<ComPtr<ID3D11ComputeShader>,3u> graph_;
    ComPtr<ID3D11Texture2D> rgb_;ComPtr<ID3D11ShaderResourceView> rgb_srv_,ocr_srv_;ComPtr<ID3D11Buffer> ocr_,current_cb_,old_cb_;
    words24 current_ {},old_ {};std::vector<UINT> old_ocr_;std::vector<char> source_;bool prior_ {};UINT width_ {},height_ {};
    std::map<std::string,std::vector<char>> saved_;nlohmann::json spec_;std::vector<char> spec_bytes_;
    std::filesystem::path spec_path_,cue_root_,graph_root_;std::string cue_closure_,graph_closure_,driver_ {"warp"};
  };


  TEST_F(HostSbsCurrentRgbCueCostGpuTest, SavedCurrentRgbCueCost) {
    const auto *out=std::getenv("SUNSHINE_TEST_CURRENT_RGB_CUE_COST_OUTPUT");
    if(!out) {GTEST_SKIP()<<"Explicit new whole-helper cost output directory required";}
    const auto &cases=spec_.at("cases");
    ASSERT_EQ(cases.size(),6u);
    ASSERT_EQ(spec_.at("case_roles").size(),cases.size());
    const std::filesystem::path destination(out);
    ASSERT_FALSE(std::filesystem::exists(destination));
    ASSERT_TRUE(std::filesystem::create_directories(destination));
    std::array<wchar_t,32768u> executable_name {};
    const auto len=GetModuleFileNameW(nullptr,executable_name.data(),DWORD(executable_name.size()));
    ASSERT_GT(len,0u);
    ASSERT_LT(len,executable_name.size());
    const std::filesystem::path executable_path(std::wstring(executable_name.data(),len));
    const auto executable_bytes=cue_bytes(executable_path);
    const nlohmann::json executable {{"path",executable_path.string()},{"bytes",executable_bytes.size()},{"sha256",cue_sha(executable_bytes.data(),executable_bytes.size())}};
    const auto input_sha=cue_sha(spec_bytes_.data(),spec_bytes_.size());
    nlohmann::json compiler=nlohmann::json::object();
#ifdef __VERSION__
    compiler["version"]=__VERSION__;
#endif
#ifdef __OPTIMIZE__
    compiler["optimize_macro"]=__OPTIMIZE__;
#else
    compiler["optimize_macro"]=nullptr;
#endif
#ifdef NDEBUG
    compiler["NDEBUG"]=true;
#else
    compiler["NDEBUG"]=false;
#endif
    const auto *flags=std::getenv("SUNSHINE_TEST_CURRENT_RGB_CUE_COST_BUILD_FLAGS");
    compiler["effective_build_flags_external_evidence"]=flags?nlohmann::json(flags):nlohmann::json(nullptr);
    nlohmann::json aggregate {{"schema",1u},{"driver",driver_},{"adapter",adapter()},{"input_specification_sha256",input_sha},{"test_executable",executable},
      {"cue_shader_root",cue_root_.string()},{"component_shader_root",graph_root_.string()},{"cue_source_closure_sha256",cue_closure_},{"component_source_closure_sha256",graph_closure_},
      {"helper_source_sha256","2cbdd4545c1d2c3829e8e3dbc5cb0d82cb4021be161b972d6e1e1aa8862fc9c8"},{"execution","IMMEDIATE_DIRECT"},{"compiler",compiler},
      {"warmups",3u},{"blocks",3u},{"repeats_per_block",20u},{"semantic_status","UNKNOWN"},{"qualified_performance",false},{"cases",nlohmann::json::array()},
      {"scope","GPU timestamps enclose complete resources_t::dispatch, including clears, validation, palette/black/modal/CC/augmentation/seed/BOX3, CPU-submitted binding/requestCB commands and CS state save/restore. CPU submit duration is measured separately around the same call. No CPU mode/palette decision or precomputed eligibility input."},
      {"excluded","Allocation, shader compilation, source/CB/OCR input upload, expected-data loading/hashing, diagnostic CopyResource/Map/readback, query creation/retrieval and disk I/O; depth/OCR/model execution, renderer and complete live path."},
      {"limitations","GPU timestamp intervals can include CPU submission gaps and driver scheduling. Instrumented or O0 test builds can inflate CPU submission and these gaps; inspect effective compile flags (last optimization flag wins). Frequency is GPU timestamp-clock Hz, not core frequency. GoogleTest wall time is not algorithm time. No pure-kernel, live FPS or end-to-end performance qualification."}};
    for(std::size_t i=0u;i<cases.size();++i) {
      const auto &item=cases.at(i);
      SCOPED_TRACE(item.at("name").get<std::string>());
      const UINT w=item.at("width"),h=item.at("height");
      ASSERT_TRUE(resize(w,h));
      const auto &a=item.at("artifacts");
      source_=cue_bound(a.at("current_rgb"),std::size_t(w)*h*8u);
      const auto cc=cue_words(cue_bound(a.at("current_constants"),96u));
      const auto oc=cue_words(cue_bound(a.at("prior_constants"),96u));
      std::copy(cc.begin(),cc.end(),current_.begin());
      std::copy(oc.begin(),oc.end(),old_.begin());
      old_ocr_=cue_words(cue_bound(a.at("prior_ocr"),1024u));
      prior_=item.at("prior_present").get<bool>();
      std::map<std::string,std::vector<char>> expected {
        {"body.u32",cue_bound(item.at("expected").at("body"),std::size_t(w)*h*4u)},
        {"matte.u32",cue_bound(item.at("expected").at("matte"),std::size_t(w)*h*4u)},
        {"receipt.u32",cue_bound(item.at("expected").at("receipt"),2048u)}};
      ASSERT_TRUE(run());
      ASSERT_NO_FATAL_FAILURE(expect_invariants());
      ASSERT_FALSE(::testing::Test::HasFailure()) << "Do not emit a successful proof or time failed operands";
      for(const auto &[name,bytes]:expected)ASSERT_EQ(saved_.at(name),bytes)<<name;
      const auto folder=destination/("case_"+std::to_string(i));
      ASSERT_TRUE(std::filesystem::create_directory(folder));
      nlohmann::json artifacts=nlohmann::json::object();
      for(const auto *name:{"uploaded_current_rgb.rgba16f","uploaded_current_constants.u32","uploaded_prior_constants.u32","uploaded_prior_ocr.u32","body.u32","matte.u32","receipt.u32"})artifacts[name]=cue_save(folder/name,saved_.at(name));
      nlohmann::json proof {{"schema",1u},{"case",item},{"case_role",spec_.at("case_roles").at(i)},{"driver",driver_},{"input_specification_sha256",input_sha},
        {"test_executable",executable},{"cue_source_closure_sha256",cue_closure_},{"component_source_closure_sha256",graph_closure_},
        {"helper_source_sha256",aggregate.at("helper_source_sha256")},{"uploaded_four_operands_byte_exact",true},{"original_body_matte_receipt_byte_exact",true},{"artifacts",artifacts},{"semantic_status","UNKNOWN"}};
      const auto proof_text=proof.dump(2)+"\n";
      const std::vector<char> proof_bytes(proof_text.begin(),proof_text.end());
      const auto proof_binding=cue_save(folder/"functional_before_cost.json",proof_bytes);
      for(UINT warm=0u;warm<3u;++warm)ASSERT_TRUE(dispatch_preuploaded());
      auto actual=read_outputs();
      ASSERT_EQ(actual,expected)<<"Warm outputs must still match full original references";
      nlohmann::json blocks=nlohmann::json::array();
      for(UINT block=0u;block<3u;++block) {
        auto measured=measure_block();
        // This is the first output read after all timed calls, before any new dispatch/upload.
        actual=read_outputs();
        bool exact=actual==expected;
        measured["block"]=block;
        measured["post_body_matte_receipt_byte_exact"]=exact;
        measured["post_artifacts"]=nlohmann::json::object();
        for(const auto &[name,bytes]:actual)measured["post_artifacts"][name]=cue_save(folder/("block_"+std::to_string(block)+"_"+name),bytes);
        blocks.push_back(measured);
        EXPECT_TRUE(exact)<<"Timed calls must not change any original mask or receipt";
      }
      const nlohmann::json identity {{"schema",1u},{"case",item},{"case_role",spec_.at("case_roles").at(i)},
        {"functional_before_cost",proof_binding},{"performance_blocks",blocks},{"execution","IMMEDIATE_DIRECT"},{"semantic_status","UNKNOWN"},{"qualified_performance",false}};
      const auto identity_text=identity.dump(2)+"\n";
      const std::vector<char> identity_bytes(identity_text.begin(),identity_text.end());
      const auto identity_binding=cue_save(folder/"whole_current_rgb_helper_cost.json",identity_bytes);
      aggregate["cases"].push_back({{"index",i},{"name",item.at("name")},{"role",spec_.at("case_roles").at(i)},{"identity",identity_binding}});
      const auto report=aggregate.dump(2)+"\n";
      std::ofstream stream(destination/"whole_current_rgb_helper_cost_identity.json",std::ios::binary|std::ios::trunc);
      stream<<report;
      ASSERT_TRUE(stream.good());
    }
  }
}
#endif
