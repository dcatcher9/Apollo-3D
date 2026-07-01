#include "video_depth_estimator.h"
#include "model_manager.h"
#include "logging.h"
#include "platform/windows/utils.h"
#include "cuda_driver_api.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <cmath>
#include <windows.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

using namespace std::literals;

class Logger : public nvinfer1::ILogger {
public:
#ifdef __GNUC__
    void msvc_dummy_destructor(char flags) noexcept override {}
#endif
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            BOOST_LOG(warning) << "TensorRT: " << msg;
        } else {
            BOOST_LOG(info) << "TensorRT: " << msg;
        }
    }
};

static Logger gLogger;

// Shared TensorRT state. The runtime and engine are created once and shared by every
// encoder instance. Execution contexts are pooled and reused: creating one allocates
// ~1.3 GB of device scratch and takes several seconds, and it cannot be safely deleted
// across the MinGW/MSVC ABI boundary (see AGENTS.md rule #4). Creating a fresh context
// on every encoder recreation (which happens frequently during video playback via MPO
// flips / HDR / resolution changes) therefore leaked ~1.3 GB each time until the GPU ran
// out of memory and the device was removed. Pooling caps live contexts at peak concurrency.
static std::mutex g_trt_mutex;
static nvinfer1::IRuntime* g_runtime = nullptr;
static nvinfer1::ICudaEngine* g_engine = nullptr;
static std::vector<nvinfer1::IExecutionContext*> g_context_pool;

template <typename T>
struct TrtDeleter {
    void operator()(T* ptr) const {
        if (ptr) {
#ifdef __GNUC__
            ptr->msvc_dummy_destructor(1);
#else
            delete ptr;
#endif
        }
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

namespace models {

    void precompile_tensorrt_engine(const std::filesystem::path& assets_dir) {
        static std::mutex compile_mutex;
        std::lock_guard<std::mutex> lock(compile_mutex);

        auto model_path = ensure_model_available(assets_dir);
        if (model_path.empty()) {
            BOOST_LOG(warning) << "Model not found. Background precompilation aborted.";
            return;
        }
        if (model_path.extension() == ".engine") {
            BOOST_LOG(info) << "TensorRT engine already compiled and ready.";
            return;
        }

        BOOST_LOG(info) << "Building TensorRT engine from ONNX... This will take a few minutes.";
        
        auto& cuda = cuda_driver_api::get();
        if (cuda.is_valid()) {
            cuda.cuInit(0);
            CUdevice cu_dev;
            if (cuda.cuDeviceGet(&cu_dev, 0) == 0) {
                CUcontext ctx;
                cuda.cuDevicePrimaryCtxRetain(&ctx, cu_dev);
                cuda.cuCtxSetCurrent(ctx);
            }
        }

        initLibNvInferPlugins(&gLogger, "");
        auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
        auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        
        // Set memory limit to 4GB
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4ULL << 30);
        
        auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        if (!parser->parseFromFile(model_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            BOOST_LOG(error) << "Failed to parse ONNX file.";
            return;
        }
        
        // For Depth Anything V2 we assume input tensor "pixel_values" and output tensor "predicted_depth"
        auto profile = builder->createOptimizationProfile();
        if (network->getNbInputs() > 0) {
            auto input = network->getInput(0);
            // Dims4 is (batch, channels, height, width)
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 14, 14});
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 518, 518});
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 1008, 1008});
            config->addOptimizationProfile(profile);
        }
        
        auto serializedModel = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (serializedModel) {
            // Save to disk so we don't rebuild next time
            auto engine_path = assets_dir / "depth_anything_v2.engine";
            std::ofstream p(engine_path, std::ios::binary);
            if (p) {
                p.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());
                BOOST_LOG(info) << "Saved built engine to " << engine_path;
            }
        } else {
            BOOST_LOG(error) << "Engine build failed.";
        }
    }

    struct video_depth_estimator::impl {
        Microsoft::WRL::ComPtr<ID3D11Device> device;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
        
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* exec_context = nullptr;
        std::mutex* trt_mutex = nullptr;
        CUcontext cuda_ctx = nullptr;
        CUstream cu_stream = nullptr;
        
        int width;
        int height;
        float ema_alpha;
        int depth_area;  // target pixel-area budget for the model input (dims derived from this)
        int depth_short_side;  // if > 0, budget the short side instead of the area
        float max_aspect;  // aspect cap for short-side mode
        bool normalize;  // per-frame min/max normalization of raw disparity
        float depth_gamma;  // shaping exponent on normalized depth (normalize mode only)
        float minmax_alpha;  // temporal EMA blend for the normalized min/max
        float edge_dilation;  // foreground-biased edge smoothing strength (0 = off)

        // Caching
        int target_w = 0;
        int target_h = 0;
        UINT reduce_groups = 0;  // threadgroups for the min/max reduction (groups * 256 = total threads)

        Microsoft::WRL::ComPtr<ID3D11ComputeShader> rgb_to_nchw_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> buffer_to_tex_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_minmax_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_minmax_ema_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_edge_dilate_cs;
        Microsoft::WRL::ComPtr<ID3D11SamplerState> linear_sampler;
        Microsoft::WRL::ComPtr<ID3D11Buffer> cbuffer;

        Microsoft::WRL::ComPtr<ID3D11Buffer> tensor_in_buf;
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> tensor_in_uav;

        Microsoft::WRL::ComPtr<ID3D11Buffer> tensor_out_buf;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> tensor_out_srv;

        // GPU-resident min/max for per-frame disparity normalization (no CPU readback).
        Microsoft::WRL::ComPtr<ID3D11Buffer> minmax_raw_buf;  // 2 uints: reduction accumulator
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> minmax_raw_uav;
        Microsoft::WRL::ComPtr<ID3D11Buffer> minmax_ema_buf;  // float4 {min, max, initialized, pad}
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> minmax_ema_uav;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> minmax_ema_srv;
        
        Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_tex;
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_uav;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_srv;

        // Edge-dilated copy of depth_tex (reprojection samples this when edge_dilation > 0).
        // Kept separate so the temporal EMA in buffer_to_tex feeds back the un-dilated depth.
        Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_tex2;
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_tex2_uav;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_tex2_srv;
        
        CUgraphicsResource cuda_in_res = nullptr;
        CUgraphicsResource cuda_out_res = nullptr;
        bool has_previous_frame = false;
        
        bool compile_shader(const std::filesystem::path& path, Microsoft::WRL::ComPtr<ID3D11ComputeShader>& out_cs) {
            Microsoft::WRL::ComPtr<ID3DBlob> blob;
            Microsoft::WRL::ComPtr<ID3DBlob> err;
            DWORD flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
            if (FAILED(D3DCompileFromFile(path.wstring().c_str(), nullptr, nullptr, "main", "cs_5_0", flags, 0, &blob, &err))) {
                if (err) BOOST_LOG(error) << "Shader compile error (" << path << "): " << (char*)err->GetBufferPointer();
                return false;
            }
            return SUCCEEDED(device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &out_cs));
        }

        impl(Microsoft::WRL::ComPtr<ID3D11Device> d, Microsoft::WRL::ComPtr<ID3D11DeviceContext> c, const std::filesystem::path& assets_dir, int w, int h, const depth_estimator_config& cfg)
            : device(d), context(c), width(w), height(h), ema_alpha(cfg.ema_alpha), depth_area(std::max(196, cfg.depth_area)),
              depth_short_side(cfg.depth_short_side), max_aspect(std::max(1.0f, cfg.max_aspect)),
              normalize(cfg.normalize), depth_gamma(cfg.depth_gamma), minmax_alpha(cfg.minmax_alpha), edge_dilation(cfg.edge_dilation)
        {
            auto model_path = ensure_model_available(assets_dir);
            if (model_path.empty()) {
                BOOST_LOG(error) << "Depth estimator failed: No model available.";
                return;
            }
            if (model_path.extension() == ".onnx") {
                precompile_tensorrt_engine(assets_dir);
                model_path = ensure_model_available(assets_dir);
            }

            if (model_path.extension() != ".engine") {
                BOOST_LOG(error) << "Depth estimator failed: No engine file available after compilation phase.";
                return;
            }

            auto& cuda = cuda_driver_api::get();
            if (cuda.is_valid()) {
                static bool cuda_init = false;
                if (!cuda_init) {
                    cuda.cuInit(0);
                    cuda_init = true;
                }
                CUdevice cu_dev;
                if (cuda.cuDeviceGet(&cu_dev, 0) == 0) {
                    cuda.cuDevicePrimaryCtxRetain(&cuda_ctx, cu_dev);
                    if (cuda_ctx) {
                        cuda.cuCtxSetCurrent(cuda_ctx);
                        cuda.cuStreamCreate(&cu_stream, CU_STREAM_NON_BLOCKING);
                    }
                }
            }

            std::lock_guard<std::mutex> lock(g_trt_mutex);
            if (!g_runtime) {
                g_runtime = nvinfer1::createInferRuntime(gLogger);
                if (g_runtime) {
                    std::ifstream file(model_path, std::ios::binary);
                    std::vector<char> trtModelStream((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                    g_engine = g_runtime->deserializeCudaEngine(trtModelStream.data(), trtModelStream.size());
                }
            }

            runtime = g_runtime;
            engine = g_engine;
            if (engine) {
                // Reuse a pooled context if one is free; otherwise pay the one-time cost of
                // creating one. Contexts are returned to the pool (not destroyed) on teardown.
                if (!g_context_pool.empty()) {
                    exec_context = g_context_pool.back();
                    g_context_pool.pop_back();
                    BOOST_LOG(info) << "Reusing pooled TensorRT execution context.";
                } else {
                    BOOST_LOG(info) << "Creating TensorRT execution context (allocates device scratch; may take several seconds)...";
                    exec_context = engine->createExecutionContext();
                }
            }
            trt_mutex = &g_trt_mutex;
            
            // Compile Shaders
            compile_shader(assets_dir / "shaders" / "directx" / "rgb_to_nchw_cs.hlsl", rgb_to_nchw_cs);
            compile_shader(assets_dir / "shaders" / "directx" / "buffer_to_tex_cs.hlsl", buffer_to_tex_cs);
            compile_shader(assets_dir / "shaders" / "directx" / "depth_minmax_cs.hlsl", depth_minmax_cs);
            compile_shader(assets_dir / "shaders" / "directx" / "depth_minmax_ema_cs.hlsl", depth_minmax_ema_cs);
            compile_shader(assets_dir / "shaders" / "directx" / "depth_edge_dilate_cs.hlsl", depth_edge_dilate_cs);

            // Min/max reduction accumulator (2 uints), pre-seeded to the reduction identity
            // {min = 0xFFFFFFFF, max = 0}. depth_minmax_ema_cs resets it after each frame.
            {
                uint32_t init_raw[2] = {0xFFFFFFFFu, 0u};
                D3D11_BUFFER_DESC bd = {};
                bd.Usage = D3D11_USAGE_DEFAULT;
                bd.ByteWidth = sizeof(init_raw);
                bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
                bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
                D3D11_SUBRESOURCE_DATA sd = {init_raw, 0, 0};
                device->CreateBuffer(&bd, &sd, &minmax_raw_buf);

                D3D11_UNORDERED_ACCESS_VIEW_DESC uav = {};
                uav.Format = DXGI_FORMAT_R32_TYPELESS;
                uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
                uav.Buffer.FirstElement = 0;
                uav.Buffer.NumElements = 2;
                uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
                device->CreateUnorderedAccessView(minmax_raw_buf.Get(), &uav, &minmax_raw_uav);
            }

            // EMA'd min/max {min, max, initialized, pad}; initialized = 0 so the first
            // frame seeds directly instead of blending from zero.
            {
                float init_ema[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                D3D11_BUFFER_DESC bd = {};
                bd.Usage = D3D11_USAGE_DEFAULT;
                bd.ByteWidth = sizeof(init_ema);
                bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
                bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                bd.StructureByteStride = sizeof(float) * 4;
                D3D11_SUBRESOURCE_DATA sd = {init_ema, 0, 0};
                device->CreateBuffer(&bd, &sd, &minmax_ema_buf);
                device->CreateUnorderedAccessView(minmax_ema_buf.Get(), nullptr, &minmax_ema_uav);
                device->CreateShaderResourceView(minmax_ema_buf.Get(), nullptr, &minmax_ema_srv);
            }

            D3D11_SAMPLER_DESC samp_desc = {};
            samp_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            samp_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
            samp_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
            samp_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            device->CreateSamplerState(&samp_desc, &linear_sampler);
            
            D3D11_BUFFER_DESC cb_desc = {};
            cb_desc.Usage = D3D11_USAGE_DYNAMIC;
            cb_desc.ByteWidth = 48;
            cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            cb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            device->CreateBuffer(&cb_desc, nullptr, &cbuffer);
        }

        ~impl() {
            auto& cuda = cuda_driver_api::get();
            if (cuda.is_valid() && cuda_ctx) {
                cuda.cuCtxSetCurrent(cuda_ctx);
                if (cu_stream) {
                    if (cuda.cuStreamSynchronize) cuda.cuStreamSynchronize(cu_stream);
                    cuda.cuStreamDestroy(cu_stream);
                }
                if (cuda_in_res) cuda.cuGraphicsUnregisterResource(cuda_in_res);
                if (cuda_out_res) cuda.cuGraphicsUnregisterResource(cuda_out_res);
            }

            // Return the execution context to the shared pool for reuse instead of leaking
            // (or destroying, which faults across the DLL boundary). The stream was
            // synchronized above, so no inference is still in flight referencing this
            // instance's tensor bindings, making the context safe for another instance to reuse.
            if (exec_context) {
                std::lock_guard<std::mutex> lock(g_trt_mutex);
                g_context_pool.push_back(exec_context);
                exec_context = nullptr;
            }
            // TRT runtime/engine are cached globally, do not destroy them here.
        }

        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> estimate(Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> input_srv, bool is_hdr) {
            if (!exec_context || !rgb_to_nchw_cs || !buffer_to_tex_cs) return nullptr;
            
            auto& cuda = cuda_driver_api::get();
            if (!cuda.is_valid()) {
                BOOST_LOG(error) << "CUDA Driver API is not available.";
                return nullptr;
            }

            if (cuda_ctx) {
                cuda.cuCtxSetCurrent(cuda_ctx);
            }
            
            // Prevent GPU starvation: if the previous AI frame is still crunching, drop this frame.
            // This prevents an infinite queue of heavy TensorRT workloads from starving the DWM and Edge Browser.
            if (cu_stream && cuda.cuStreamQuery) {
                if (cuda.cuStreamQuery(cu_stream) == 600) { // CUDA_ERROR_NOT_READY
                    return (edge_dilation > 0.0f && depth_tex2_srv) ? depth_tex2_srv : depth_srv;
                }
            }

            D3D11_TEXTURE2D_DESC input_desc = {0};
            Microsoft::WRL::ComPtr<ID3D11Resource> input_res;
            input_srv->GetResource(&input_res);
            Microsoft::WRL::ComPtr<ID3D11Texture2D> input_tex;
            if (SUCCEEDED(input_res.As(&input_tex))) {
                input_tex->GetDesc(&input_desc);
            }

            if (target_w == 0 || target_h == 0) {
                // The capture surface can report a 0x0 descriptor mid HDR/mode transition or
                // before the first real frame. Deriving the model resolution from that yields a
                // garbage size (NaN aspect -> integer-overflow -> clamps to 1008x1008) that would
                // be cached for the whole session. Wait for a valid frame instead.
                if (input_desc.Width == 0 || input_desc.Height == 0) {
                    return nullptr;
                }
                float aspect_ratio = (float)input_desc.Width / (float)input_desc.Height;
                if (depth_short_side > 0) {
                    // Short-side budget (iw3-style): pin the short side to the target so
                    // vertical depth detail is constant regardless of aspect ratio; the
                    // long side grows with aspect, capped by max_aspect to bound cost.
                    int short_side = std::max(14, (int)std::round((float)depth_short_side / 14.0f) * 14);
                    if (aspect_ratio >= 1.0f) {
                        target_h = short_side;
                        target_w = (int)std::round(short_side * std::min(aspect_ratio, max_aspect));
                    } else {
                        target_w = short_side;
                        target_h = (int)std::round(short_side * std::min(1.0f / aspect_ratio, max_aspect));
                    }
                } else {
                    // Area budget (legacy): fixed total pixel count across aspect ratios.
                    target_h = std::round(std::sqrt((float)depth_area / aspect_ratio));
                    target_w = std::round(aspect_ratio * target_h);
                }

                target_h = std::max(14, (int)std::round((float)target_h / 14.0f) * 14);
                target_w = std::max(14, (int)std::round((float)target_w / 14.0f) * 14);

                // TensorRT engine's MAX profile is 1008x1008. Scale down aspect-preserving
                // (rather than clamping each axis independently, which would distort depth).
                int longest = std::max(target_w, target_h);
                if (longest > 1008) {
                    float s = 1008.0f / (float)longest;
                    target_w = std::max(14, (int)std::round((float)target_w * s / 14.0f) * 14);
                    target_h = std::max(14, (int)std::round((float)target_h * s / 14.0f) * 14);
                }

                // Threads for the min/max reduction; grid-stride handles any element count.
                int elems = target_w * target_h;
                reduce_groups = (UINT)std::min(64, std::max(1, (elems + 255) / 256));

                BOOST_LOG(info) << "Depth Estimator dynamic resolution set to " << target_w << "x" << target_h;

                if (cuda_in_res) cuda.cuGraphicsUnregisterResource(cuda_in_res);
                if (cuda_out_res) cuda.cuGraphicsUnregisterResource(cuda_out_res);
                
                D3D11_BUFFER_DESC buf_desc = {};
                buf_desc.Usage = D3D11_USAGE_DEFAULT;
                buf_desc.ByteWidth = target_w * target_h * 3 * sizeof(float);
                buf_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
                buf_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                buf_desc.StructureByteStride = sizeof(float);
                device->CreateBuffer(&buf_desc, nullptr, &tensor_in_buf);
                device->CreateUnorderedAccessView(tensor_in_buf.Get(), nullptr, &tensor_in_uav);
                
                buf_desc.ByteWidth = target_w * target_h * sizeof(float);
                buf_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                device->CreateBuffer(&buf_desc, nullptr, &tensor_out_buf);
                device->CreateShaderResourceView(tensor_out_buf.Get(), nullptr, &tensor_out_srv);
                
                D3D11_TEXTURE2D_DESC tex_desc = {};
                tex_desc.Width = target_w;
                tex_desc.Height = target_h;
                tex_desc.MipLevels = 1;
                tex_desc.ArraySize = 1;
                tex_desc.Format = DXGI_FORMAT_R32_FLOAT;
                tex_desc.SampleDesc.Count = 1;
                tex_desc.Usage = D3D11_USAGE_DEFAULT;
                tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
                device->CreateTexture2D(&tex_desc, nullptr, &depth_tex);
                device->CreateUnorderedAccessView(depth_tex.Get(), nullptr, &depth_uav);
                device->CreateShaderResourceView(depth_tex.Get(), nullptr, &depth_srv);

                // Second depth texture for the edge-dilated output.
                device->CreateTexture2D(&tex_desc, nullptr, &depth_tex2);
                device->CreateUnorderedAccessView(depth_tex2.Get(), nullptr, &depth_tex2_uav);
                device->CreateShaderResourceView(depth_tex2.Get(), nullptr, &depth_tex2_srv);

                // Clear the texture to exactly 0.0f so the EMA shader correctly initializes on the first frame
                // instead of blending with undefined GPU garbage memory.
                const float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                context->ClearUnorderedAccessViewFloat(depth_uav.Get(), clear_color);
                
                auto res1 = cuda.cuGraphicsD3D11RegisterResource(&cuda_in_res, tensor_in_buf.Get(), 0);
                auto res2 = cuda.cuGraphicsD3D11RegisterResource(&cuda_out_res, tensor_out_buf.Get(), 0);
                if (res1 != 0 || res2 != 0) {
                    BOOST_LOG(error) << "cuGraphicsD3D11RegisterResource failed: " << res1 << ", " << res2;
                }
            }


            // IF WE REACH HERE, CUDA HAS FINISHED!
            // This means tensor_out_buf contains the FINISHED depth map from the previous frame!
            // Let's run Step 3 first to copy it into depth_tex. Since tensor_out_buf is fully unmapped
            // from CUDA, D3D11 will execute this instantly without blocking the CPU thread!
            
            // Populate the shared constant buffer up front; buffer_to_tex_cs, the min/max
            // passes and rgb_to_nchw_cs all read it this frame.
            D3D11_MAPPED_SUBRESOURCE mapped;
            if (SUCCEEDED(context->Map(cbuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                uint32_t* u = (uint32_t*)mapped.pData;
                float* f = (float*)mapped.pData;
                u[0] = target_w;
                u[1] = target_h;
                u[2] = is_hdr ? 1 : 0;
                f[3] = ema_alpha;
                u[4] = normalize ? 1u : 0u;
                f[5] = depth_gamma;
                f[6] = minmax_alpha;
                u[7] = reduce_groups * 256u;  // total threads for the reduction grid-stride
                f[8] = edge_dilation;
                f[9] = 0.0f;
                f[10] = 0.0f;
                f[11] = 0.0f;
                context->Unmap(cbuffer.Get(), 0);
            }

            // IF WE REACH HERE, CUDA HAS FINISHED!
            // tensor_out_buf holds the FINISHED raw disparity from the PREVIOUS frame, and is
            // fully unmapped from CUDA, so these D3D11 passes run without blocking the CPU.
            if (has_previous_frame) {
                // 3a. Per-frame min/max normalization (GPU-resident; no CPU readback).
                if (normalize && depth_minmax_cs && depth_minmax_ema_cs && minmax_raw_uav && minmax_ema_uav) {
                    // Pass A: parallel reduction of the raw disparity -> min/max (uint bits).
                    context->CSSetShader(depth_minmax_cs.Get(), nullptr, 0);
                    context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
                    context->CSSetShaderResources(0, 1, tensor_out_srv.GetAddressOf());
                    context->CSSetUnorderedAccessViews(0, 1, minmax_raw_uav.GetAddressOf(), nullptr);
                    context->Dispatch(reduce_groups, 1, 1);

                    ID3D11UnorderedAccessView* null_uav1 = nullptr;
                    ID3D11ShaderResourceView* null_srv1 = nullptr;
                    context->CSSetUnorderedAccessViews(0, 1, &null_uav1, nullptr);
                    context->CSSetShaderResources(0, 1, &null_srv1);

                    // Pass B: fold into the EMA'd min/max and reset the accumulator (1 thread).
                    context->CSSetShader(depth_minmax_ema_cs.Get(), nullptr, 0);
                    ID3D11UnorderedAccessView* ema_uavs[2] = {minmax_ema_uav.Get(), minmax_raw_uav.Get()};
                    context->CSSetUnorderedAccessViews(0, 2, ema_uavs, nullptr);
                    context->Dispatch(1, 1, 1);

                    ID3D11UnorderedAccessView* null_uav2[2] = {nullptr, nullptr};
                    context->CSSetUnorderedAccessViews(0, 2, null_uav2, nullptr);
                }

                // 3b. Buffer to Texture: map/normalize the disparity into depth_tex + temporal EMA.
                context->CSSetShader(buffer_to_tex_cs.Get(), nullptr, 0);
                context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
                ID3D11ShaderResourceView* bt_srvs[2] = {tensor_out_srv.Get(), minmax_ema_srv.Get()};
                context->CSSetShaderResources(0, 2, bt_srvs);
                context->CSSetUnorderedAccessViews(0, 1, depth_uav.GetAddressOf(), nullptr);

                context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

                ID3D11UnorderedAccessView* null_uav = nullptr;
                ID3D11ShaderResourceView* null_srvs[2] = {nullptr, nullptr};
                context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
                context->CSSetShaderResources(0, 2, null_srvs);

                // 3c. Edge dilation: smooth the depth silhouette into depth_tex2 (reduces the
                // jaggy fringe at object edges). depth_tex keeps the un-dilated depth so the
                // temporal EMA above doesn't compound the smoothing frame over frame.
                if (edge_dilation > 0.0f && depth_edge_dilate_cs) {
                    context->CSSetShader(depth_edge_dilate_cs.Get(), nullptr, 0);
                    context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
                    context->CSSetShaderResources(0, 1, depth_srv.GetAddressOf());
                    context->CSSetUnorderedAccessViews(0, 1, depth_tex2_uav.GetAddressOf(), nullptr);

                    context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

                    ID3D11UnorderedAccessView* null_uav_d = nullptr;
                    ID3D11ShaderResourceView* null_srv_d = nullptr;
                    context->CSSetUnorderedAccessViews(0, 1, &null_uav_d, nullptr);
                    context->CSSetShaderResources(0, 1, &null_srv_d);
                }
            }

            // 1. D3D11 Compute Shader: Resize & Normalize to NCHW FP32 Buffer (for CURRENT frame)
            context->CSSetShader(rgb_to_nchw_cs.Get(), nullptr, 0);
            context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
            context->CSSetShaderResources(0, 1, input_srv.GetAddressOf());
            context->CSSetUnorderedAccessViews(0, 1, tensor_in_uav.GetAddressOf(), nullptr);
            context->CSSetSamplers(0, 1, linear_sampler.GetAddressOf());
            
            context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);
            
            ID3D11UnorderedAccessView* null_uav = nullptr;
            ID3D11ShaderResourceView* null_srv = nullptr;
            context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
            context->CSSetShaderResources(0, 1, &null_srv);
            // No explicit Flush: cuGraphicsMapResources() below already guarantees the
            // preceding D3D11 compute work completes before the CUDA stream reads the buffer.
            // Force-flushing every frame only prevents the driver from interleaving other GPU
            // consumers (DWM / Edge / the Widgets panel), which starves them and can trigger a TDR.

            // 2. CUDA Execution (for CURRENT frame)
            CUgraphicsResource resources[2] = {cuda_in_res, cuda_out_res};
            auto map_res = cuda.cuGraphicsMapResources(2, resources, cu_stream);
            if (map_res != 0) {
                BOOST_LOG(error) << "cuGraphicsMapResources failed: " << map_res;
            }

            void* d_in = nullptr;
            void* d_out = nullptr;
            cuda.cuGraphicsResourceGetMappedPointer((CUdeviceptr*)&d_in, nullptr, cuda_in_res);
            cuda.cuGraphicsResourceGetMappedPointer((CUdeviceptr*)&d_out, nullptr, cuda_out_res);

            if (!d_in || !d_out) {
                BOOST_LOG(error) << "Failed to get mapped pointer for TensorRT.";
            } else {
                exec_context->setInputShape("pixel_values", nvinfer1::Dims4{1, 3, target_h, target_w});
                exec_context->setTensorAddress("pixel_values", (void*)d_in);
                exec_context->setTensorAddress("predicted_depth", (void*)d_out);
                {
                    // Serialize TensorRT async enqueue to avoid driver-level concurrent execution faults
                    std::lock_guard<std::mutex> lock(*trt_mutex);
                    exec_context->enqueueV3(cu_stream);
                }
            }
            
            cuda.cuGraphicsUnmapResources(2, resources, cu_stream);
            
            has_previous_frame = true;

            return (edge_dilation > 0.0f && depth_tex2_srv) ? depth_tex2_srv : depth_srv;
        }
    };

    video_depth_estimator::video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device,
                          Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
                          const std::filesystem::path& assets_dir,
                          int input_width, int input_height, const depth_estimator_config& cfg)
        : pimpl(std::make_unique<impl>(device, context, assets_dir, input_width, input_height, cfg)) {}

    video_depth_estimator::~video_depth_estimator() = default;

    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> video_depth_estimator::estimate_depth(Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> input_srv, bool is_hdr) {
        return pimpl->estimate(input_srv, is_hdr);
    }
}
