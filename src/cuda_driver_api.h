#pragma once

#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

// Essential CUDA Driver API types and functions
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef unsigned long long CUdeviceptr;
typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_NOT_SUPPORTED = 801
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
typedef CUresult(__stdcall* PFN_cuDevicePrimaryCtxRelease)(CUdevice dev);
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
typedef struct CUgraphNode_st* CUgraphNode;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUarray_st* CUarray;
typedef unsigned long long CUgraphConditionalHandle;
#define CU_GRAPH_COND_ASSIGN_DEFAULT 0x1u
typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 0x01,
    CU_MEMORYTYPE_DEVICE = 0x02,
    CU_MEMORYTYPE_ARRAY = 0x03,
    CU_MEMORYTYPE_UNIFIED = 0x04
} CUmemorytype;
typedef enum CUgraphNodeType_enum {
    CU_GRAPH_NODE_TYPE_KERNEL = 0,
    CU_GRAPH_NODE_TYPE_MEMCPY = 1,
    CU_GRAPH_NODE_TYPE_MEMSET = 2,
    CU_GRAPH_NODE_TYPE_HOST = 3,
    CU_GRAPH_NODE_TYPE_GRAPH = 4,
    CU_GRAPH_NODE_TYPE_EMPTY = 5,
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
    CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12,
    CU_GRAPH_NODE_TYPE_CONDITIONAL = 13
} CUgraphNodeType;
typedef enum CUgraphConditionalNodeType_enum {
    CU_GRAPH_COND_TYPE_IF = 0,
    CU_GRAPH_COND_TYPE_WHILE = 1
} CUgraphConditionalNodeType;
typedef struct CUDA_KERNEL_NODE_PARAMS_st {
    CUfunction func;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    void** kernelParams;
    void** extra;
} CUDA_KERNEL_NODE_PARAMS;
typedef struct CUDA_MEMSET_NODE_PARAMS_st {
    CUdeviceptr dst;
    size_t pitch;
    unsigned int value;
    unsigned int elementSize;
    size_t width;
    size_t height;
} CUDA_MEMSET_NODE_PARAMS;
typedef struct CUDA_MEMCPY3D_st {
    size_t srcXInBytes;
    size_t srcY;
    size_t srcZ;
    size_t srcLOD;
    CUmemorytype srcMemoryType;
    const void* srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    void* reserved0;
    size_t srcPitch;
    size_t srcHeight;
    size_t dstXInBytes;
    size_t dstY;
    size_t dstZ;
    size_t dstLOD;
    CUmemorytype dstMemoryType;
    void* dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    void* reserved1;
    size_t dstPitch;
    size_t dstHeight;
    size_t WidthInBytes;
    size_t Height;
    size_t Depth;
} CUDA_MEMCPY3D;
typedef struct CUDA_CONDITIONAL_NODE_PARAMS_st {
    CUgraphConditionalHandle handle;
    CUgraphConditionalNodeType type;
    unsigned int size;
    CUgraph* phGraph_out;
    CUcontext ctx;
} CUDA_CONDITIONAL_NODE_PARAMS;
typedef struct CUgraphNodeParams_st {
    CUgraphNodeType type;
    int reserved0[3];
    union {
        CUDA_CONDITIONAL_NODE_PARAMS conditional;
        std::int64_t reserved1[29];
    } params;
    std::int64_t reserved2;
} CUgraphNodeParams;
static_assert(sizeof(CUDA_KERNEL_NODE_PARAMS) == 56, "CUDA Driver API x64 kernel-node ABI drift");
static_assert(sizeof(CUDA_MEMSET_NODE_PARAMS) == 40, "CUDA Driver API x64 memset-node ABI drift");
static_assert(sizeof(CUDA_MEMCPY3D) == 200, "CUDA Driver API x64 memcpy-node ABI drift");
static_assert(alignof(CUDA_MEMCPY3D) == 8, "CUDA Driver API x64 memcpy-node alignment drift");
static_assert(offsetof(CUDA_MEMCPY3D, srcMemoryType) == 32, "CUDA memcpy source-type ABI drift");
static_assert(offsetof(CUDA_MEMCPY3D, srcDevice) == 48, "CUDA memcpy source-pointer ABI drift");
static_assert(offsetof(CUDA_MEMCPY3D, dstMemoryType) == 120, "CUDA memcpy destination-type ABI drift");
static_assert(offsetof(CUDA_MEMCPY3D, dstDevice) == 136, "CUDA memcpy destination-pointer ABI drift");
static_assert(offsetof(CUDA_MEMCPY3D, WidthInBytes) == 176, "CUDA memcpy width ABI drift");
static_assert(offsetof(CUDA_MEMCPY3D, Height) == 184, "CUDA memcpy height ABI drift");
static_assert(offsetof(CUDA_MEMCPY3D, Depth) == 192, "CUDA memcpy depth ABI drift");
static_assert(sizeof(CUDA_CONDITIONAL_NODE_PARAMS) == 32, "CUDA Driver API x64 conditional-node ABI drift");
static_assert(offsetof(CUgraphNodeParams, params) == 16, "CUDA Driver API graph-node ABI drift");
static_assert(sizeof(CUgraphNodeParams) == 256, "CUDA Driver API x64 graph-node ABI drift");
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
typedef CUresult(__stdcall* PFN_cuModuleLoadDataEx)(CUmodule* module, const void* image, unsigned int numOptions, int* options, void** optionValues);
typedef CUresult(__stdcall* PFN_cuModuleGetFunction)(CUfunction* hfunc, CUmodule hmod, const char* name);
typedef CUresult(__stdcall* PFN_cuModuleUnload)(CUmodule hmod);
typedef CUresult(__stdcall* PFN_cuGraphCreate)(CUgraph* phGraph, unsigned int flags);
typedef CUresult(__stdcall* PFN_cuGraphConditionalHandleCreate)(CUgraphConditionalHandle* pHandle_out, CUgraph hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags);
typedef CUresult(__stdcall* PFN_cuGraphAddKernelNode)(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
typedef CUresult(__stdcall* PFN_cuGraphAddMemsetNode)(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx);
typedef CUresult(__stdcall* PFN_cuGraphAddMemcpyNode)(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx);
// CUDA exports both ABIs. The legacy symbol has five arguments; only _v2 accepts edge data.
typedef CUresult(__stdcall* PFN_cuGraphAddNode)(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraphNodeParams* nodeParams);
typedef CUresult(__stdcall* PFN_cuGraphAddNode_v2)(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, const void* dependencyData, size_t numDependencies, CUgraphNodeParams* nodeParams);
typedef CUresult(__stdcall* PFN_cuGraphAddChildGraphNode)(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph);
typedef CUresult(__stdcall* PFN_cuGraphGetNodes)(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes);
typedef CUresult(__stdcall* PFN_cuGraphGetRootNodes)(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes);
typedef CUresult(__stdcall* PFN_cuGraphNodeGetType)(CUgraphNode hNode, CUgraphNodeType* type);
typedef CUresult(__stdcall* PFN_cuGraphChildGraphNodeGetGraph)(CUgraphNode hNode, CUgraph* phGraph);
typedef CUresult(__stdcall* PFN_cuGraphMemcpyNodeGetParams)(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams);
typedef CUresult(__stdcall* PFN_cuGraphMemcpyNodeSetParams)(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams);
// CUDA 12.3+ exposes edge-data-aware _v2 forms beside the legacy three-argument ABI.
typedef CUresult(__stdcall* PFN_cuGraphNodeGetDependencies)(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies);
typedef CUresult(__stdcall* PFN_cuGraphNodeGetDependencies_v2)(CUgraphNode hNode, CUgraphNode* dependencies, void* edgeData, size_t* numDependencies);
typedef CUresult(__stdcall* PFN_cuGraphNodeGetDependentNodes)(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes);
typedef CUresult(__stdcall* PFN_cuGraphNodeGetDependentNodes_v2)(CUgraphNode hNode, CUgraphNode* dependentNodes, void* edgeData, size_t* numDependentNodes);
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
    PFN_cuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease = nullptr;
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

    // Optional CUDA conditional-graph infrastructure. Keep this separate from is_valid(): the
    // estimator's existing minimum Driver API contract must continue to work on older drivers.
    PFN_cuModuleLoadDataEx cuModuleLoadDataEx = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuModuleUnload cuModuleUnload = nullptr;
    PFN_cuGraphCreate cuGraphCreate = nullptr;
    PFN_cuGraphConditionalHandleCreate cuGraphConditionalHandleCreate = nullptr;
    PFN_cuGraphAddKernelNode cuGraphAddKernelNode = nullptr;
    PFN_cuGraphAddMemsetNode cuGraphAddMemsetNode = nullptr;
    PFN_cuGraphAddMemcpyNode cuGraphAddMemcpyNode = nullptr;
    PFN_cuGraphAddNode cuGraphAddNode = nullptr;
    PFN_cuGraphAddNode_v2 cuGraphAddNode_v2 = nullptr;
    PFN_cuGraphAddChildGraphNode cuGraphAddChildGraphNode = nullptr;
    PFN_cuGraphGetNodes cuGraphGetNodes = nullptr;
    PFN_cuGraphGetRootNodes cuGraphGetRootNodes = nullptr;
    PFN_cuGraphNodeGetType cuGraphNodeGetType = nullptr;
    PFN_cuGraphChildGraphNodeGetGraph cuGraphChildGraphNodeGetGraph = nullptr;
    PFN_cuGraphMemcpyNodeGetParams cuGraphMemcpyNodeGetParams = nullptr;
    PFN_cuGraphMemcpyNodeSetParams cuGraphMemcpyNodeSetParams = nullptr;
    PFN_cuGraphNodeGetDependencies cuGraphNodeGetDependencies = nullptr;
    PFN_cuGraphNodeGetDependencies_v2 cuGraphNodeGetDependencies_v2 = nullptr;
    PFN_cuGraphNodeGetDependentNodes cuGraphNodeGetDependentNodes = nullptr;
    PFN_cuGraphNodeGetDependentNodes_v2 cuGraphNodeGetDependentNodes_v2 = nullptr;
    
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
        // This is the minimum contract used by video_depth_estimator, not merely proof that
        // nvcuda.dll loaded. Keeping the check complete prevents a partially resolved driver API
        // from becoming a null function-pointer call in warmup, stream teardown, or D3D interop.
        return cuInit &&
               cuDeviceGet &&
               cuCtxSetCurrent &&
               cuMemAlloc &&
               cuMemFree &&
               cuStreamCreate &&
               cuStreamDestroy &&
               cuStreamSynchronize &&
               cuStreamQuery &&
               cuGraphicsD3D11RegisterResource &&
               cuGraphicsMapResources &&
               cuGraphicsUnmapResources &&
               cuGraphicsResourceGetMappedPointer &&
               cuGraphicsUnregisterResource;
    }

    bool has_conditional_graph_support() const {
        return cuCtxGetCurrent &&
               cuCtxSetCurrent &&
               cuModuleLoadDataEx &&
               cuModuleGetFunction &&
               cuModuleUnload &&
               cuGraphCreate &&
               cuGraphDestroy &&
               cuGraphExecDestroy &&
               cuGraphInstantiateWithFlags &&
               cuGraphLaunch &&
               cuGraphConditionalHandleCreate &&
               cuGraphAddKernelNode &&
               (cuGraphAddNode_v2 || cuGraphAddNode) &&
               cuGraphAddChildGraphNode &&
               cuGraphGetNodes &&
               cuGraphGetRootNodes &&
               cuGraphNodeGetType &&
               cuGraphChildGraphNodeGetGraph &&
               cuGraphMemcpyNodeGetParams &&
               (cuGraphNodeGetDependencies_v2 || cuGraphNodeGetDependencies) &&
               (cuGraphNodeGetDependentNodes_v2 || cuGraphNodeGetDependentNodes);
    }

    CUresult graph_add_node(CUgraphNode* node, CUgraph graph, const CUgraphNode* dependencies, size_t dependency_count, CUgraphNodeParams* params) const {
        if (cuGraphAddNode_v2) {
            return cuGraphAddNode_v2(node, graph, dependencies, nullptr, dependency_count, params);
        }
        return cuGraphAddNode ? cuGraphAddNode(node, graph, dependencies, dependency_count, params) : CUDA_ERROR_INVALID_HANDLE;
    }

    CUresult graph_node_get_dependencies(CUgraphNode node, CUgraphNode* dependencies, size_t* dependency_count) const {
        if (cuGraphNodeGetDependencies_v2) {
            return cuGraphNodeGetDependencies_v2(node, dependencies, nullptr, dependency_count);
        }
        return cuGraphNodeGetDependencies ? cuGraphNodeGetDependencies(node, dependencies, dependency_count) : CUDA_ERROR_INVALID_HANDLE;
    }

    CUresult graph_node_get_dependents(CUgraphNode node, CUgraphNode* dependents, size_t* dependent_count) const {
        if (cuGraphNodeGetDependentNodes_v2) {
            return cuGraphNodeGetDependentNodes_v2(node, dependents, nullptr, dependent_count);
        }
        return cuGraphNodeGetDependentNodes ? cuGraphNodeGetDependentNodes(node, dependents, dependent_count) : CUDA_ERROR_INVALID_HANDLE;
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
                api.cuDevicePrimaryCtxRelease = (PFN_cuDevicePrimaryCtxRelease)GetProcAddress(api.hMod, "cuDevicePrimaryCtxRelease_v2");
                if (!api.cuDevicePrimaryCtxRelease) api.cuDevicePrimaryCtxRelease = (PFN_cuDevicePrimaryCtxRelease)GetProcAddress(api.hMod, "cuDevicePrimaryCtxRelease");
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

                api.cuModuleLoadDataEx = (PFN_cuModuleLoadDataEx)GetProcAddress(api.hMod, "cuModuleLoadDataEx");
                api.cuModuleGetFunction = (PFN_cuModuleGetFunction)GetProcAddress(api.hMod, "cuModuleGetFunction");
                api.cuModuleUnload = (PFN_cuModuleUnload)GetProcAddress(api.hMod, "cuModuleUnload");
                api.cuGraphCreate = (PFN_cuGraphCreate)GetProcAddress(api.hMod, "cuGraphCreate");
                api.cuGraphConditionalHandleCreate = (PFN_cuGraphConditionalHandleCreate)GetProcAddress(api.hMod, "cuGraphConditionalHandleCreate");
                api.cuGraphAddKernelNode = (PFN_cuGraphAddKernelNode)GetProcAddress(api.hMod, "cuGraphAddKernelNode");
                api.cuGraphAddMemsetNode = (PFN_cuGraphAddMemsetNode)GetProcAddress(api.hMod, "cuGraphAddMemsetNode");
                api.cuGraphAddMemcpyNode = (PFN_cuGraphAddMemcpyNode)GetProcAddress(api.hMod, "cuGraphAddMemcpyNode");
                api.cuGraphAddNode = (PFN_cuGraphAddNode)GetProcAddress(api.hMod, "cuGraphAddNode");
                api.cuGraphAddNode_v2 = (PFN_cuGraphAddNode_v2)GetProcAddress(api.hMod, "cuGraphAddNode_v2");
                api.cuGraphAddChildGraphNode = (PFN_cuGraphAddChildGraphNode)GetProcAddress(api.hMod, "cuGraphAddChildGraphNode");
                api.cuGraphGetNodes = (PFN_cuGraphGetNodes)GetProcAddress(api.hMod, "cuGraphGetNodes");
                api.cuGraphGetRootNodes = (PFN_cuGraphGetRootNodes)GetProcAddress(api.hMod, "cuGraphGetRootNodes");
                api.cuGraphNodeGetType = (PFN_cuGraphNodeGetType)GetProcAddress(api.hMod, "cuGraphNodeGetType");
                api.cuGraphChildGraphNodeGetGraph = (PFN_cuGraphChildGraphNodeGetGraph)GetProcAddress(api.hMod, "cuGraphChildGraphNodeGetGraph");
                api.cuGraphMemcpyNodeGetParams = (PFN_cuGraphMemcpyNodeGetParams)GetProcAddress(api.hMod, "cuGraphMemcpyNodeGetParams");
                api.cuGraphMemcpyNodeSetParams = (PFN_cuGraphMemcpyNodeSetParams)GetProcAddress(api.hMod, "cuGraphMemcpyNodeSetParams");
                api.cuGraphNodeGetDependencies = (PFN_cuGraphNodeGetDependencies)GetProcAddress(api.hMod, "cuGraphNodeGetDependencies");
                api.cuGraphNodeGetDependencies_v2 = (PFN_cuGraphNodeGetDependencies_v2)GetProcAddress(api.hMod, "cuGraphNodeGetDependencies_v2");
                api.cuGraphNodeGetDependentNodes = (PFN_cuGraphNodeGetDependentNodes)GetProcAddress(api.hMod, "cuGraphNodeGetDependentNodes");
                api.cuGraphNodeGetDependentNodes_v2 = (PFN_cuGraphNodeGetDependentNodes_v2)GetProcAddress(api.hMod, "cuGraphNodeGetDependentNodes_v2");
                
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
