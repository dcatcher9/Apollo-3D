#pragma once
#include "cuda_driver_api.h"
#include <NvInferRuntime.h>

struct GlyphEncoder {
  struct Log: nvinfer1::ILogger {
    void log(Severity severity,const char* message) noexcept override {
      if(severity<=Severity::kERROR)std::cerr<<message<<'\n';
    }
  } log;
  cuda_driver_api &cu=cuda_driver_api::get();
  CUcontext primary=nullptr;
  CUstream stream=nullptr;
  nvinfer1::IRuntime *runtime=nullptr;
  nvinfer1::ICudaEngine *engine=nullptr;
  nvinfer1::IExecutionContext *exec=nullptr;
  std::array<CUgraphicsResource,2> handles{};
  static void requireCuda(CUresult status,const char *action) {
    if(status!=CUDA_SUCCESS)throw std::runtime_error(std::string(action)+" CUDA "+std::to_string(status));
  }
  GlyphEncoder(ID3D11Device* device,ID3D11Buffer* in,ID3D11Buffer* out,const fs::path& plan) {
    ComPtr<IDXGIDevice> dx;ComPtr<IDXGIAdapter> adapter;
    checked(device->QueryInterface(IID_PPV_ARGS(&dx)),"D3D device adapter");
    checked(dx->GetAdapter(&adapter),"D3D adapter");
    if(!cu.is_valid()||!cu.cuD3D11GetDevice)throw std::runtime_error("missing CUDA driver entry points");
    CUdevice ordinal=0;
    requireCuda(cu.cuInit(0),"init");requireCuda(cu.cuD3D11GetDevice(&ordinal,adapter.Get()),"find D3D GPU");
    requireCuda(cu.cuDevicePrimaryCtxRetain(&primary,ordinal),"retain device");
    requireCuda(cu.cuCtxSetCurrent(primary),"select device");
    requireCuda(cu.cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"create stream");
    runtime=nvinfer1::createInferRuntime(log);
    std::ifstream file(plan,std::ios::binary);
    std::vector<char> blob((std::istreambuf_iterator<char>(file)),{});
    if(!runtime||blob.empty())throw std::runtime_error("missing runtime or checkpoint engine");
    engine=runtime->deserializeCudaEngine(blob.data(),blob.size());
    exec=engine?engine->createExecutionContext():nullptr;
    if(!exec||engine->getNbIOTensors()!=2||exec->getMaxOutputSize("logits")!=4*4*512*144)
      throw std::runtime_error("expected fixed 144x512 mixed CNN");
    requireCuda(cu.cuGraphicsD3D11RegisterResource(&handles[0],in,0),"register RGB binding");
    requireCuda(cu.cuGraphicsD3D11RegisterResource(&handles[1],out,0),"register logit binding");
  }
  void run() {
    requireCuda(cu.cuGraphicsMapResources(2,handles.data(),stream),"map encoder bindings");
    std::array<CUdeviceptr,2> ptr{};
    std::array<size_t,2> bytes{};
    for(size_t i=0;i<2;++i)
      requireCuda(cu.cuGraphicsResourceGetMappedPointer(&ptr[i],&bytes[i],handles[i]),"map binding address");
    if(bytes[0]!=3*4*144*512 || bytes[1]!=4*4*144*512 ||
       !exec->setTensorAddress("rgb",reinterpret_cast<void*>(ptr[0])) ||
       !exec->setTensorAddress("logits",reinterpret_cast<void*>(ptr[1]))||!exec->enqueueV3(stream))
      throw std::runtime_error("mixed CNN binding or enqueue failed");
    requireCuda(cu.cuGraphicsUnmapResources(2,handles.data(),stream),"unmap encoder bindings");
  }
  void finish() {
    requireCuda(cu.cuStreamSynchronize(stream),"finish qualification");
    for(auto handle:handles)requireCuda(cu.cuGraphicsUnregisterResource(handle),"release interop resource");
    requireCuda(cu.cuStreamDestroy(stream),"release stream");
  }
};
