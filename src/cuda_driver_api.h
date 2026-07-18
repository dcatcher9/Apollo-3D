#pragma once

#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <mutex>
#include <string>

// Essential CUDA Driver API types and functions
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef unsigned long long CUdeviceptr;
typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_NOT_READY = 600
} CUresult;
typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
} CUdevice_attribute;

typedef CUresult(__stdcall* PFN_cuInit)(unsigned int Flags);
typedef CUresult(__stdcall* PFN_cuDeviceGet)(CUdevice* device, int ordinal);
typedef CUresult(__stdcall* PFN_cuDeviceGetAttribute)(int* pi, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult(__stdcall* PFN_cuDeviceGetName)(char* name, int len, CUdevice dev);
typedef CUresult(__stdcall* PFN_cuDevicePrimaryCtxRetain)(CUcontext* pctx, CUdevice dev);
typedef CUresult(__stdcall* PFN_cuCtxCreate)(CUcontext* pctx, unsigned int flags, CUdevice dev);
typedef CUresult(__stdcall* PFN_cuCtxGetCurrent)(CUcontext* pctx);
typedef CUresult(__stdcall* PFN_cuCtxSetCurrent)(CUcontext ctx);
typedef CUresult(__stdcall* PFN_cuMemAlloc)(CUdeviceptr* dptr, size_t bytesize);
typedef CUresult(__stdcall* PFN_cuMemFree)(CUdeviceptr dptr);
typedef CUresult(__stdcall* PFN_cuMemcpyHtoD)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
typedef CUresult(__stdcall* PFN_cuMemcpyDtoH)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);

// D3D11 Interop
typedef struct CUstream_st* CUstream;
#define CU_STREAM_NON_BLOCKING 0x1
typedef CUresult(__stdcall* PFN_cuStreamCreate)(CUstream* phStream, unsigned int Flags);
typedef CUresult(__stdcall* PFN_cuStreamDestroy)(CUstream hStream);
typedef CUresult(__stdcall* PFN_cuStreamSynchronize)(CUstream hStream);
typedef CUresult(__stdcall* PFN_cuStreamQuery)(CUstream hStream);
typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphExec_st* CUgraphExec;
typedef enum CUstreamCaptureMode_enum {
    CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    CU_STREAM_CAPTURE_MODE_RELAXED = 2
} CUstreamCaptureMode;
typedef CUresult(__stdcall* PFN_cuStreamBeginCapture)(CUstream hStream, CUstreamCaptureMode mode);
typedef CUresult(__stdcall* PFN_cuStreamEndCapture)(CUstream hStream, CUgraph* phGraph);
typedef CUresult(__stdcall* PFN_cuGraphInstantiateWithFlags)(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags);
typedef CUresult(__stdcall* PFN_cuGraphLaunch)(CUgraphExec hGraphExec, CUstream hStream);
typedef CUresult(__stdcall* PFN_cuGraphDestroy)(CUgraph hGraph);
typedef CUresult(__stdcall* PFN_cuGraphExecDestroy)(CUgraphExec hGraphExec);
typedef struct CUgraphicsResource_st* CUgraphicsResource;
typedef CUresult(__stdcall* PFN_cuGraphicsD3D11RegisterResource)(CUgraphicsResource* pCudaResource, ID3D11Resource* pD3DResource, unsigned int Flags);
typedef CUresult(__stdcall* PFN_cuD3D11GetDevice)(CUdevice* pCudaDevice, IDXGIAdapter* pAdapter);

// Events (for GPU-stream timing of async TensorRT enqueues; see src/sbs_perf.*)
typedef struct CUevent_st* CUevent;
#define CU_EVENT_DEFAULT 0x0
typedef CUresult(__stdcall* PFN_cuEventCreate)(CUevent* phEvent, unsigned int Flags);
typedef CUresult(__stdcall* PFN_cuEventRecord)(CUevent hEvent, CUstream hStream);
typedef CUresult(__stdcall* PFN_cuEventQuery)(CUevent hEvent);
typedef CUresult(__stdcall* PFN_cuEventSynchronize)(CUevent hEvent);
typedef CUresult(__stdcall* PFN_cuEventElapsedTime)(float* pMilliseconds, CUevent hStart, CUevent hEnd);
typedef CUresult(__stdcall* PFN_cuEventDestroy)(CUevent hEvent);
typedef CUresult(__stdcall* PFN_cuGraphicsMapResources)(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
typedef CUresult(__stdcall* PFN_cuGraphicsUnmapResources)(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
typedef CUresult(__stdcall* PFN_cuGraphicsResourceGetMappedPointer)(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource);
typedef CUresult(__stdcall* PFN_cuGraphicsUnregisterResource)(CUgraphicsResource resource);

struct cuda_driver_api {
    HMODULE hMod = nullptr;
    PFN_cuInit cuInit = nullptr;
    PFN_cuDeviceGet cuDeviceGet = nullptr;
    PFN_cuDeviceGetAttribute cuDeviceGetAttribute = nullptr;
    PFN_cuDeviceGetName cuDeviceGetName = nullptr;
    PFN_cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain = nullptr;
    PFN_cuCtxCreate cuCtxCreate = nullptr;
    PFN_cuCtxGetCurrent cuCtxGetCurrent = nullptr;
    PFN_cuCtxSetCurrent cuCtxSetCurrent = nullptr;
    PFN_cuMemAlloc cuMemAlloc = nullptr;
    PFN_cuMemFree cuMemFree = nullptr;
    PFN_cuMemcpyHtoD cuMemcpyHtoD = nullptr;
    PFN_cuMemcpyDtoH cuMemcpyDtoH = nullptr;
    PFN_cuStreamCreate cuStreamCreate = nullptr;
    PFN_cuStreamDestroy cuStreamDestroy = nullptr;
    PFN_cuStreamSynchronize cuStreamSynchronize = nullptr;
    PFN_cuStreamQuery cuStreamQuery = nullptr;
    PFN_cuStreamBeginCapture cuStreamBeginCapture = nullptr;
    PFN_cuStreamEndCapture cuStreamEndCapture = nullptr;
    PFN_cuGraphInstantiateWithFlags cuGraphInstantiateWithFlags = nullptr;
    PFN_cuGraphLaunch cuGraphLaunch = nullptr;
    PFN_cuGraphDestroy cuGraphDestroy = nullptr;
    PFN_cuGraphExecDestroy cuGraphExecDestroy = nullptr;
    
    PFN_cuGraphicsD3D11RegisterResource cuGraphicsD3D11RegisterResource = nullptr;
    PFN_cuD3D11GetDevice cuD3D11GetDevice = nullptr;
    PFN_cuGraphicsMapResources cuGraphicsMapResources = nullptr;
    PFN_cuGraphicsUnmapResources cuGraphicsUnmapResources = nullptr;
    PFN_cuGraphicsResourceGetMappedPointer cuGraphicsResourceGetMappedPointer = nullptr;
    PFN_cuGraphicsUnregisterResource cuGraphicsUnregisterResource = nullptr;

    PFN_cuEventCreate cuEventCreate = nullptr;
    PFN_cuEventRecord cuEventRecord = nullptr;
    PFN_cuEventQuery cuEventQuery = nullptr;
    PFN_cuEventSynchronize cuEventSynchronize = nullptr;
    PFN_cuEventElapsedTime cuEventElapsedTime = nullptr;
    PFN_cuEventDestroy cuEventDestroy = nullptr;

    bool is_valid() const {
        return cuInit && cuMemAlloc && cuGraphicsD3D11RegisterResource;
    }

    static cuda_driver_api& get() {
        static cuda_driver_api api;
        static std::once_flag load_once;
        std::call_once(load_once, []() {
            api.hMod = LoadLibraryA("nvcuda.dll");
            if (api.hMod) {
                api.cuInit = (PFN_cuInit)GetProcAddress(api.hMod, "cuInit");
                api.cuDeviceGet = (PFN_cuDeviceGet)GetProcAddress(api.hMod, "cuDeviceGet");
                api.cuDeviceGetAttribute = (PFN_cuDeviceGetAttribute)GetProcAddress(api.hMod, "cuDeviceGetAttribute");
                api.cuDeviceGetName = (PFN_cuDeviceGetName)GetProcAddress(api.hMod, "cuDeviceGetName");
                api.cuDevicePrimaryCtxRetain = (PFN_cuDevicePrimaryCtxRetain)GetProcAddress(api.hMod, "cuDevicePrimaryCtxRetain");
                api.cuCtxCreate = (PFN_cuCtxCreate)GetProcAddress(api.hMod, "cuCtxCreate_v2");
                api.cuCtxGetCurrent = (PFN_cuCtxGetCurrent)GetProcAddress(api.hMod, "cuCtxGetCurrent");
                api.cuCtxSetCurrent = (PFN_cuCtxSetCurrent)GetProcAddress(api.hMod, "cuCtxSetCurrent");
                api.cuMemAlloc = (PFN_cuMemAlloc)GetProcAddress(api.hMod, "cuMemAlloc_v2");
                api.cuMemFree = (PFN_cuMemFree)GetProcAddress(api.hMod, "cuMemFree_v2");
                api.cuMemcpyHtoD = (PFN_cuMemcpyHtoD)GetProcAddress(api.hMod, "cuMemcpyHtoD_v2");
                api.cuMemcpyDtoH = (PFN_cuMemcpyDtoH)GetProcAddress(api.hMod, "cuMemcpyDtoH_v2");
                api.cuStreamCreate = (PFN_cuStreamCreate)GetProcAddress(api.hMod, "cuStreamCreate");
                api.cuStreamDestroy = (PFN_cuStreamDestroy)GetProcAddress(api.hMod, "cuStreamDestroy_v2");
                if (!api.cuStreamDestroy) api.cuStreamDestroy = (PFN_cuStreamDestroy)GetProcAddress(api.hMod, "cuStreamDestroy");
                api.cuStreamSynchronize = (PFN_cuStreamSynchronize)GetProcAddress(api.hMod, "cuStreamSynchronize");
                api.cuStreamQuery = (PFN_cuStreamQuery)GetProcAddress(api.hMod, "cuStreamQuery");
                api.cuStreamBeginCapture = (PFN_cuStreamBeginCapture)GetProcAddress(api.hMod, "cuStreamBeginCapture_v2");
                if (!api.cuStreamBeginCapture) api.cuStreamBeginCapture = (PFN_cuStreamBeginCapture)GetProcAddress(api.hMod, "cuStreamBeginCapture");
                api.cuStreamEndCapture = (PFN_cuStreamEndCapture)GetProcAddress(api.hMod, "cuStreamEndCapture");
                api.cuGraphInstantiateWithFlags = (PFN_cuGraphInstantiateWithFlags)GetProcAddress(api.hMod, "cuGraphInstantiateWithFlags");
                api.cuGraphLaunch = (PFN_cuGraphLaunch)GetProcAddress(api.hMod, "cuGraphLaunch");
                api.cuGraphDestroy = (PFN_cuGraphDestroy)GetProcAddress(api.hMod, "cuGraphDestroy");
                api.cuGraphExecDestroy = (PFN_cuGraphExecDestroy)GetProcAddress(api.hMod, "cuGraphExecDestroy");
                
                api.cuGraphicsD3D11RegisterResource = (PFN_cuGraphicsD3D11RegisterResource)GetProcAddress(api.hMod, "cuGraphicsD3D11RegisterResource");
                api.cuD3D11GetDevice = (PFN_cuD3D11GetDevice)GetProcAddress(api.hMod, "cuD3D11GetDevice");
                api.cuGraphicsMapResources = (PFN_cuGraphicsMapResources)GetProcAddress(api.hMod, "cuGraphicsMapResources");
                api.cuGraphicsUnmapResources = (PFN_cuGraphicsUnmapResources)GetProcAddress(api.hMod, "cuGraphicsUnmapResources");
                api.cuGraphicsResourceGetMappedPointer = (PFN_cuGraphicsResourceGetMappedPointer)GetProcAddress(api.hMod, "cuGraphicsResourceGetMappedPointer_v2");
                if (!api.cuGraphicsResourceGetMappedPointer) {
                    api.cuGraphicsResourceGetMappedPointer = (PFN_cuGraphicsResourceGetMappedPointer)GetProcAddress(api.hMod, "cuGraphicsResourceGetMappedPointer");
                }
                api.cuGraphicsUnregisterResource = (PFN_cuGraphicsUnregisterResource)GetProcAddress(api.hMod, "cuGraphicsUnregisterResource");

                api.cuEventCreate = (PFN_cuEventCreate)GetProcAddress(api.hMod, "cuEventCreate");
                api.cuEventRecord = (PFN_cuEventRecord)GetProcAddress(api.hMod, "cuEventRecord");
                api.cuEventQuery = (PFN_cuEventQuery)GetProcAddress(api.hMod, "cuEventQuery");
                api.cuEventSynchronize = (PFN_cuEventSynchronize)GetProcAddress(api.hMod, "cuEventSynchronize");
                api.cuEventElapsedTime = (PFN_cuEventElapsedTime)GetProcAddress(api.hMod, "cuEventElapsedTime");
                api.cuEventDestroy = (PFN_cuEventDestroy)GetProcAddress(api.hMod, "cuEventDestroy_v2");
                if (!api.cuEventDestroy) api.cuEventDestroy = (PFN_cuEventDestroy)GetProcAddress(api.hMod, "cuEventDestroy");
            }
        });
        return api;
    }
};
