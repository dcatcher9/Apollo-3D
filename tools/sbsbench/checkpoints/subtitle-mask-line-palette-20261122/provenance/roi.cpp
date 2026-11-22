// One offline GPU submission chain from the FP16 host snapshot through the mixed CNN to native alpha.
#define main sampler_main
#include "../rawmatte-spatial-guide-1122/sampling_probe.cpp"
#undef main
#include <cmath>
#include <map>
#include <chrono>
#include <iterator>
#include "encoder.h"

struct Buffer {
  ComPtr<ID3D11Buffer> resource;
  ComPtr<ID3D11ShaderResourceView> srv;
  ComPtr<ID3D11UnorderedAccessView> uav;
};
struct PaletteProbe: Probe {
  std::map<std::string,ComPtr<ID3D11ComputeShader>> stages;
  Buffer input,logits;
  std::unique_ptr<GlyphEncoder> encoder;
  PaletteProbe(const fs::path& root,const fs::path& engine):Probe(root.parent_path()/"rawmatte-spatial-guide-1122"){
    for(auto name:{"pack_rgb","pack_tile","export_roles","coarse","make_hist","smooth_hist","choose_colors","make_moments","smooth_moments","prototypes","decode_palette"}){
      ComPtr<ID3DBlob> code,errors;ComPtr<ID3D11ComputeShader> shader;
      const HRESULT hr=D3DCompileFromFile((fs::path("E:/Git/Repo/Apollo-3D/src_assets/windows/assets/shaders/directx/host_sbs_direct_glyph_palette_cs.hlsl")).c_str(),nullptr,D3D_COMPILE_STANDARD_FILE_INCLUDE,
          name,"cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
      if(FAILED(hr)&&errors)std::cerr<<name<<' '<<static_cast<char*>(errors->GetBufferPointer());
      checked(hr,"compile palette pipeline");
      checked(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&shader),"create pipeline program");
      stages.emplace(name,std::move(shader));
    }
    input=make(3*144*512,4);logits=make(4*144*512,4);
    encoder=std::make_unique<GlyphEncoder>(device.Get(),input.resource.Get(),logits.resource.Get(),engine);
  }
  Buffer make(u32 n,u32 stride){
    Buffer b;D3D11_BUFFER_DESC desc{n*stride,D3D11_USAGE_DEFAULT,D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS,
        0u,D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,stride};
    checked(device->CreateBuffer(&desc,nullptr,&b.resource),"allocate field");
    checked(device->CreateShaderResourceView(b.resource.Get(),nullptr,&b.srv),"create field input");
    checked(device->CreateUnorderedAccessView(b.resource.Get(),nullptr,&b.uav),"create field output");
    return b;
  }
  void runLine(const fs::path& sourceFile,const fs::path& dest,u32 w,u32 h,std::array<u32,8> geometry,bool timing){
    const auto div=[](u32 n,u32 by){return (n+by-1)/by;};
    const u32 nw=geometry[2]-geometry[0],nh=geometry[3]-geometry[1],w28=geometry[4],w84=geometry[5],n84=w84*84;
    const u32 step=std::max(1u,u32(std::nearbyint(double(nh)/96)));
    geometry[7]=std::min(65536u,0x7fffffffu/(div(nw,step)*div(nh,step)));
    std::vector<std::uint16_t> pixels(size_t(w)*h*4);
    std::ifstream file(sourceFile,std::ios::binary);
    if(!file.read(reinterpret_cast<char*>(pixels.data()),pixels.size()*2))throw std::runtime_error("bad FP16 source");
    D3D11_TEXTURE2D_DESC desc{w,h,1,1,DXGI_FORMAT_R16G16B16A16_FLOAT,{1,0},D3D11_USAGE_DEFAULT,D3D11_BIND_SHADER_RESOURCE,0,0};
    D3D11_SUBRESOURCE_DATA texels{pixels.data(),w*8,0};ComPtr<ID3D11Texture2D> texture;ComPtr<ID3D11ShaderResourceView> source;
    checked(device->CreateTexture2D(&desc,&texels,&texture),"create snapshot");
    checked(device->CreateShaderResourceView(texture.Get(),nullptr,&source),"view snapshot");
    auto constants=buffer(sizeof(geometry),D3D11_BIND_CONSTANT_BUFFER,false,geometry.data());
    context->CSSetConstantBuffers(0,1,constants.GetAddressOf());
    auto roles=make(w28*28,16),low=make(w28*28,4),fine=make(n84,4),peak=make(w28*28,16),mean=make(w28*28,16);
    auto hist=make(4913*4,4),lut=make(4913,16),temp=make(4913,16),palette=make(4913,16),winner=make(2,4);
    auto moment=make(n84,16),squared=make(n84,16),h1=make(n84,16),h2=make(n84,16),nearby=make(n84,16),distant=make(n84,16);
    auto parts=make(2*div(n84,128),16),proto=make(2,16),alpha=make(nw*nh,4);
    using Binding=std::pair<unsigned,ID3D11ShaderResourceView*>;
    auto stage=[&](const char* name,u32 arg,u32 gx,u32 gy,std::initializer_list<Binding> bindings,Buffer& output,Buffer *secondary=nullptr){
      std::array<ID3D11ShaderResourceView*,16> reads{};
      for(auto [at,srv]:bindings)reads.at(at)=srv;
      std::array<ID3D11UnorderedAccessView*,2> writes{output.uav.Get(),secondary?secondary->uav.Get():nullptr};
      geometry[6]=arg;context->UpdateSubresource(constants.Get(),0,nullptr,geometry.data(),0,0);
      context->CSSetShaderResources(0,UINT(reads.size()),reads.data());context->CSSetUnorderedAccessViews(0,2,writes.data(),nullptr);
      context->CSSetShader(stages.at(name).Get(),nullptr,0);context->Dispatch(gx,gy,1);
      reads.fill(nullptr);writes.fill(nullptr);
      context->CSSetShaderResources(0,UINT(reads.size()),reads.data());context->CSSetUnorderedAccessViews(0,2,writes.data(),nullptr);
    };
    auto submit=[&](){
      constexpr std::array<u32,4> zero{};context->ClearUnorderedAccessViewUint(hist.uav.Get(),zero.data());
      stage("pack_rgb",28,div(w28,16),4,{{0,source.Get()}},low);
      stage("pack_rgb",84,div(w84,16),11,{{0,source.Get()}},fine);
      for(u32 start=0;start<w28;start+=440){
        stage("pack_tile",start,32,18,{{7,low.srv.Get()}},input);
        encoder->run();
        stage("export_roles",start,div(std::min(440u,w28-start),16),4,{{12,logits.srv.Get()}},roles);
      }
      stage("coarse",0,div(w28,16),4,{{1,roles.srv.Get()}},peak,&mean);
      stage("make_hist",0,div(div(nw,step),16),div(div(nh,step),8),{{0,source.Get()},{1,roles.srv.Get()}},hist);
      stage("smooth_hist",0,39,1,{{9,hist.srv.Get()}},lut);
      stage("smooth_hist",1,39,1,{{10,lut.srv.Get()}},temp);
      stage("smooth_hist",2,39,1,{{10,temp.srv.Get()}},lut,&palette);
      stage("choose_colors",0,1,1,{{4,palette.srv.Get()},{10,lut.srv.Get()}},winner);
      stage("make_moments",0,div(w84,16),11,{{1,roles.srv.Get()},{8,fine.srv.Get()}},moment,&squared);
      for(u32 radius:{6u,17u}){
        stage("smooth_moments",radius,div(w84,16),11,{{12,moment.srv.Get()},{13,squared.srv.Get()}},h1,&h2);
        stage("smooth_moments",64+radius,div(w84,16),11,{{12,h1.srv.Get()},{13,h2.srv.Get()}},radius==6?nearby:distant);
      }
      stage("prototypes",0,div(n84,128),1,{{1,roles.srv.Get()},{8,fine.srv.Get()}},parts);
      stage("prototypes",1,1,1,{{15,parts.srv.Get()}},proto);
      stage("decode_palette",0,div(nw,16),div(nh,8),{{0,source.Get()},{1,roles.srv.Get()},{2,peak.srv.Get()},
        {3,mean.srv.Get()},{4,palette.srv.Get()},{5,winner.srv.Get()},{6,nearby.srv.Get()},{8,fine.srv.Get()},
        {14,distant.srv.Get()},{15,proto.srv.Get()}},alpha);
    };
    submit();
    write(alpha.resource.Get(),dest/"alpha.f32");
    
    if(timing){
      D3D11_QUERY_DESC event{D3D11_QUERY_EVENT,0};ComPtr<ID3D11Query> done;
      checked(device->CreateQuery(&event,&done),"create qualification completion fence");
      std::ofstream log(dest/"timing-ms.txt");
      for(int i=0;i<36;++i){
        auto before=std::chrono::steady_clock::now();submit();context->End(done.Get());context->Flush();
        HRESULT state;
        while((state=context->GetData(done.Get(),nullptr,0,0))==S_FALSE)SwitchToThread();
        checked(state,"read completion fence");
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
        if(i>=6)log<<ms<<'\n';
      }
    }
  }
};
int main(int argc,char **argv) try{
  if(argc<4||argc>5)throw std::runtime_error("supply root, native source directory and frozen engine");
  fs::path root=argv[1],sources=argv[2];PaletteProbe probe(root,fs::path(argv[3]));
  std::ifstream list(root/"cases.txt");std::string name;u32 w,h;std::array<u32,8> geom{};
  while(list>>name>>w>>h>>geom[0]>>geom[1]>>geom[2]>>geom[3]>>geom[4]>>geom[5])
    probe.runLine(sources/name/"source.bin",root/name,w,h,geom,argc==5&&(name=="real-667"||name=="landing-2160-color-shadow-onset"));
  if(!list.eof())throw std::runtime_error("invalid case list");
  probe.encoder->finish();
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
