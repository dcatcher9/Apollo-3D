#include "video_depth_estimator.h"
#include "model_manager.h"
#include "logging.h"
#include "platform/windows/utils.h"
#include "cuda_driver_api.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
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

// One resident engine per depth model (keyed by config::depth_model_info::name), so switching
// models mid-stream loads/keeps a distinct engine instead of being pinned to the first model
// (the old single-g_engine design required a restart). Engines are never evicted: an
// IExecutionContext holds ~1.3 GB scratch and cannot be safely destroyed across the MinGW/MSVC
// ABI boundary, so contexts are pooled per engine and reused (see the ctor/dtor). With
// sequential model testing this leaves 2-3 engines resident, which is acceptable.
struct engine_slot {
    nvinfer1::ICudaEngine* engine = nullptr;
    std::vector<nvinfer1::IExecutionContext*> context_pool;
    bool io_validated = false;
};
static std::map<std::string, engine_slot> g_engines;  // guarded by g_trt_mutex

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

    void precompile_tensorrt_engine(const std::filesystem::path& assets_dir, const config::depth_model_info& model) {
        static std::mutex compile_mutex;
        std::lock_guard<std::mutex> lock(compile_mutex);

        const std::string& model_name = model.name;
        const std::string& model_url = model.url;
        const std::string engine_name = engine_filename(model);
        auto model_path = ensure_model_available(assets_dir, model_name, model_url, engine_name);
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
        
        // Input tensor "pixel_values" (DA-V2: rank-4 [1,3,H,W]; DA-V3: rank-5 [1,1,3,H,W]),
        // output tensor "predicted_depth". Build the optimization profile at the model's input
        // rank. Passing Dims INTO TensorRT is ABI-safe (only returning Dims by value faults).
        auto profile = builder->createOptimizationProfile();
        // Fixed-shape models (static input dims baked into the ONNX) need no optimization profile;
        // adding one with a different range would violate the static shape (DA3MONO-LARGE's dynamic
        // export baked resolution-dependent shape math, so it is exported at a fixed resolution).
        if (!model.fixed_shape && network->getNbInputs() > 0) {
            auto input = network->getInput(0);
            auto dims_for = [&](int h, int w) -> nvinfer1::Dims {
                nvinfer1::Dims d {};
                if (model.input_rank == 5) {
                    d.nbDims = 5;
                    d.d[0] = 1; d.d[1] = 1; d.d[2] = 3; d.d[3] = h; d.d[4] = w;
                } else {
                    d.nbDims = 4;
                    d.d[0] = 1; d.d[1] = 3; d.d[2] = h; d.d[3] = w;
                }
                return d;
            };
            if (model.dynamic_width) {
                // Height is baked into the ONNX (fixed_h); only width is a real dynamic axis.
                // Pin H and range W from 1:1 up to ~4:1 so a single engine serves every landscape
                // aspect (4:3 -> 16:9 -> ultrawide/32:9). OPT is the common ultrawide (~2.37:1).
                const int p = std::max(1, model.patch);
                const int h = model.fixed_h;
                auto r14 = [&](double x) { return std::max(p, (int)std::round(x / p) * p); };
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims_for(h, h));
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims_for(h, r14(h * 2.37)));
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims_for(h, r14(h * 4.0)));
            } else {
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims_for(14, 14));
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims_for(518, 518));
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims_for(1008, 1008));
            }
            config->addOptimizationProfile(profile);
        }

        // Prune outputs the pipeline doesn't consume (DA-V3 emits confidence/extrinsics/intrinsics
        // alongside predicted_depth). unmarkOutput drops them from the engine I/O and lets the
        // builder dead-code-eliminate their exclusive branches, so enqueueV3 only needs the two
        // tensors the pipeline binds. Names via getName()/getNbOutputs() are ABI-safe (no Dims).
        {
            std::vector<nvinfer1::ITensor*> to_unmark;
            for (int i = 0; i < network->getNbOutputs(); i++) {
                auto* t = network->getOutput(i);
                std::string nm = t->getName();
                bool keep = (nm == model.output_tensor) || (model.keep_confidence && nm == "confidence");
                if (!keep) to_unmark.push_back(t);
            }
            for (auto* t : to_unmark) {
                BOOST_LOG(info) << "Depth engine: pruning unused output '" << t->getName() << "'.";
                network->unmarkOutput(*t);
            }
        }

        auto serializedModel = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (serializedModel) {
            // Save under the recipe-specific engine name so a later recipe change rebuilds
            // rather than silently reusing this engine's (now-wrong) I/O layout.
            auto engine_path = assets_dir / engine_name;
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
        
        float ema_alpha;
        int depth_short_side;  // depth map short-side resolution (clamped to native short side)
        float max_aspect;  // aspect cap for short-side mode
        float minmax_alpha;  // temporal EMA blend for the normalized min/max
        float depth_fps;  // target depth-update rate (interval auto-derived from measured video fps)
        bool guided_upsample;  // color-guided depth upsample (snaps silhouettes to color edges)
        float guided_sigma;  // color-distance sigma for the guided upsample
        std::string model_name;  // local file stem; engine cached as <model_name>.engine
        std::string model_url;  // where to download the onnx if absent
        int input_rank;  // model input rank: 4 = [1,3,H,W] (DA-V2), 5 = [1,1,3,H,W] (DA-V3)
        uint32_t output_transform;  // applied to raw model output before normalization: 0=identity, 1=shifted reciprocal 1/(depth+depth_shift) (DA-V3 depth->disparity)
        float depth_shift;  // shift in the DA-V3 disparity transform (bounds the near spike; foreground-scale knob)
        bool fixed_shape;  // ONNX has static input dims; skip runtime setInputShape (dims are baked)
        bool dynamic_width;  // ONNX height is baked (fixed_h) but width is dynamic; pin height, set width from aspect
        int fixed_h;         // baked model input height for dynamic_width (e.g. 336); 0 otherwise
        std::string output_tensor_name;  // depth output tensor bound for inference

        // Cadence: measure the video frame rate from the call period and derive how many
        // frames to skip between depth inferences so depth refreshes near depth_fps.
        unsigned frame_counter = 0;
        float measured_fps = 0.0f;
        int effective_interval = 1;
        int last_logged_interval = 0;
        std::chrono::steady_clock::time_point last_call_time {};

        // Caching
        int target_w = 0;
        int target_h = 0;
        UINT reduce_groups = 0;  // threadgroups for the min/max reduction (groups * 256 = total threads)
        int cb_is_hdr = -1;  // is_hdr baked into the constant buffers (-1 = not built yet)

        Microsoft::WRL::ComPtr<ID3D11ComputeShader> rgb_to_nchw_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> buffer_to_tex_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_minmax_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_minmax_ema_cs;
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

        // Color-guided (joint-bilateral) depth upsample: guide_tex holds the full-res color
        // downsampled to the depth grid; guided_depth_tex is the 2x-res, color-edge-snapped
        // depth the reprojection samples when guided_upsample is on. Refreshed EVERY frame
        // against the current frame's color, so silhouettes track between inference frames.
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_guide_downsample_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_guided_upsample_cs;
        Microsoft::WRL::ComPtr<ID3D11Buffer> guided_cbuffer;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> guide_tex;
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> guide_uav;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> guide_srv;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> guided_depth_tex;
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> guided_depth_uav;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> guided_depth_srv;
        
        CUgraphicsResource cuda_in_res = nullptr;
        CUgraphicsResource cuda_out_res = nullptr;
        bool has_previous_frame = false;
        bool stream_error_logged = false;
        
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

        impl(Microsoft::WRL::ComPtr<ID3D11Device> d, Microsoft::WRL::ComPtr<ID3D11DeviceContext> c, const std::filesystem::path& assets_dir, const config::video_t::sbs_t& cfg, const config::depth_model_info& model)
            : device(d), context(c), ema_alpha((float)cfg.ema),
              depth_short_side(std::max(196, cfg.depth_short_side)), max_aspect(std::max(1.0f, (float)cfg.depth_max_aspect)),
              minmax_alpha((float)cfg.minmax_ema),
              depth_fps((float)cfg.depth_fps),
              guided_upsample(cfg.guided_upsample), guided_sigma(std::max(0.01f, (float)cfg.guided_sigma)),
              model_name(model.name), model_url(model.url),
              input_rank(model.input_rank), output_transform((uint32_t)model.output_transform),
              depth_shift(std::max(0.001f, (float)cfg.depth_shift)),
              fixed_shape(model.fixed_shape),
              dynamic_width(model.dynamic_width), fixed_h(model.fixed_h),
              output_tensor_name(model.output_tensor)
        {
            // Per-model depth-rate override (heavier models that can't hold the global rate);
            // 0 = use the config depth_fps.
            if (model.depth_fps_override > 0.0) {
                depth_fps = (float) model.depth_fps_override;
            }
            const std::string engine_name = engine_filename(model);
            auto model_path = ensure_model_available(assets_dir, model_name, model_url, engine_name);
            if (model_path.empty()) {
                BOOST_LOG(error) << "Depth estimator failed: No model available.";
                return;
            }
            if (model_path.extension() == ".onnx") {
                precompile_tensorrt_engine(assets_dir, model);
                model_path = ensure_model_available(assets_dir, model_name, model_url, engine_name);
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

            {  // Scope this lock to the g_engines/g_runtime access only: it MUST be released before
               // warmup_inference() at the end of the ctor (which re-locks g_trt_mutex) -- a
               // non-recursive std::mutex would otherwise self-deadlock and hang construction.
            std::lock_guard<std::mutex> lock(g_trt_mutex);
            // Load (once) the engine for THIS model into its own slot. Different models coexist;
            // switching models never reuses a stale engine and never needs a restart.
            auto& slot = g_engines[model_name];
            if (!g_runtime) {
                g_runtime = nvinfer1::createInferRuntime(gLogger);
            }
            if (g_runtime && !slot.engine) {
                std::ifstream file(model_path, std::ios::binary);
                std::vector<char> trtModelStream((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                slot.engine = g_runtime->deserializeCudaEngine(trtModelStream.data(), trtModelStream.size());
            }

            runtime = g_runtime;
            engine = slot.engine;

            // Validate the engine's I/O once per model against what the D3D pipeline binds: FP32
            // tensors named "pixel_values" (input) and "predicted_depth" (output). The model is
            // user-selectable (sbs_3d_depth_model_url), so a model with FP16/other I/O dtypes or
            // different tensor names would otherwise bind mismatched buffers and silently produce
            // garbage depth. We log the actual bindings and warn loudly on any mismatch.
            if (engine && !slot.io_validated) {
                slot.io_validated = true;
                auto dtype_name = [](nvinfer1::DataType t) -> const char* {
                    switch (t) {
                        case nvinfer1::DataType::kFLOAT: return "FP32";
                        case nvinfer1::DataType::kHALF:  return "FP16";
                        case nvinfer1::DataType::kINT8:  return "INT8";
                        case nvinfer1::DataType::kINT32: return "INT32";
                        default: return "other";
                    }
                };
                bool have_in = false, have_out = false;
                for (int i = 0; i < engine->getNbIOTensors(); i++) {
                    const char* tname = engine->getIOTensorName(i);
                    auto dt = engine->getTensorDataType(tname);
                    bool is_input = engine->getTensorIOMode(tname) == nvinfer1::TensorIOMode::kINPUT;
                    BOOST_LOG(info) << "Depth engine tensor '" << tname << "' " << (is_input ? "(input)" : "(output)")
                                    << " dtype=" << dtype_name(dt);
                    if (std::string(tname) == "pixel_values") {
                        have_in = true;
                        if (dt != nvinfer1::DataType::kFLOAT)
                            BOOST_LOG(error) << "Depth model input 'pixel_values' is " << dtype_name(dt)
                                             << ", not FP32 -> the FP32 NCHW buffer will feed garbage. Use a keep_io_types (FP32 I/O) model.";
                    } else if (std::string(tname) == "predicted_depth") {
                        have_out = true;
                        if (dt != nvinfer1::DataType::kFLOAT)
                            BOOST_LOG(error) << "Depth model output 'predicted_depth' is " << dtype_name(dt)
                                             << ", not FP32 -> the FP32 depth/min-max passes will read garbage.";
                    }
                }
                if (!have_in || !have_out)
                    BOOST_LOG(error) << "Depth model is missing the expected tensor name(s) 'pixel_values'/'predicted_depth'; "
                                        "the pipeline binds those by name and will not work with this model.";
            }

            if (engine) {
                // Reuse a pooled context if one is free; otherwise pay the one-time cost of
                // creating one. Contexts are returned to this model's pool (not destroyed) on
                // teardown. Pool is per-engine: a context is bound to the engine that made it.
                if (!slot.context_pool.empty()) {
                    exec_context = slot.context_pool.back();
                    slot.context_pool.pop_back();
                    BOOST_LOG(info) << "Reusing pooled TensorRT execution context.";
                } else {
                    BOOST_LOG(info) << "Creating TensorRT execution context (allocates device scratch; may take several seconds)...";
                    exec_context = engine->createExecutionContext();
                }
            }
            trt_mutex = &g_trt_mutex;
            }  // release g_trt_mutex before the shader/buffer setup and warmup below

            // Compile Shaders
            compile_shader(assets_dir / "shaders" / "directx" / "rgb_to_nchw_cs.hlsl", rgb_to_nchw_cs);
            compile_shader(assets_dir / "shaders" / "directx" / "buffer_to_tex_cs.hlsl", buffer_to_tex_cs);
            compile_shader(assets_dir / "shaders" / "directx" / "depth_minmax_cs.hlsl", depth_minmax_cs);
            compile_shader(assets_dir / "shaders" / "directx" / "depth_minmax_ema_cs.hlsl", depth_minmax_ema_cs);
            if (guided_upsample) {
                bool ok = compile_shader(assets_dir / "shaders" / "directx" / "depth_guide_downsample_cs.hlsl", depth_guide_downsample_cs) &&
                          compile_shader(assets_dir / "shaders" / "directx" / "depth_guided_upsample_cs.hlsl", depth_guided_upsample_cs);
                if (ok) {
                    BOOST_LOG(info) << "Guided depth upsample enabled (sigma " << guided_sigma << ").";
                } else {
                    BOOST_LOG(warning) << "Guided depth upsample shaders failed to compile; falling back to plain depth.";
                    guided_upsample = false;
                }
            }

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

            // Constant buffers are created in ensure_cbuffers() once the model resolution is
            // known: every field is fixed for the session, so they are built once (immutable)
            // instead of being re-mapped on the encode thread every frame.

            // Warm up here so TensorRT's CUDA lazy kernel load / JIT (~20 s on the big models)
            // happens during construction -- which ensure_depth_estimator() runs on a background
            // thread -- rather than stalling the first real convert() on the encode thread and
            // freezing the stream right after a host-SBS / model switch.
            warmup_inference();
        }

        // Run one throwaway inference at the engine's optimization shape so TensorRT loads its
        // CUDA modules now. The bulk of the "first inference" cost is module loading, which is
        // shape-independent, so a warmup at the OPT shape spares the first real frame the stall
        // even if its resolution differs. Uses its own scratch device buffers because the per-
        // frame D3D-interop buffers aren't allocated until convert() knows the frame resolution.
        // Pure CUDA + TensorRT (no D3D immediate context), so it's safe on the construction thread.
        void warmup_inference() {
            if (!exec_context || !cu_stream) return;
            auto& cuda = cuda_driver_api::get();
            if (!cuda.is_valid()) return;
            if (cuda_ctx) cuda.cuCtxSetCurrent(cuda_ctx);

            int h, w;
            if (dynamic_width) {
                h = fixed_h;
                w = std::max(14, (int)std::round(fixed_h * 2.37 / 14.0) * 14);
            } else if (fixed_shape) {
                return;  // baked shape; querying engine dims here is ABI-unsafe and none ship today
            } else {
                h = w = 518;  // square OPT profile point shared by the DA-V2/V3 dynamic engines
            }

            const size_t in_elems = (size_t) 3 * h * w;  // batch/view dims are 1 for rank-4 and rank-5
            const size_t out_elems = (size_t) h * w;
            CUdeviceptr d_in = 0, d_out = 0;
            if (cuda.cuMemAlloc(&d_in, in_elems * sizeof(float)) != 0) return;
            if (cuda.cuMemAlloc(&d_out, out_elems * sizeof(float)) != 0) {
                cuda.cuMemFree(d_in);
                return;
            }

            nvinfer1::Dims in_dims {};
            if (input_rank == 5) {
                in_dims.nbDims = 5;
                in_dims.d[0] = 1; in_dims.d[1] = 1; in_dims.d[2] = 3; in_dims.d[3] = h; in_dims.d[4] = w;
            } else {
                in_dims.nbDims = 4;
                in_dims.d[0] = 1; in_dims.d[1] = 3; in_dims.d[2] = h; in_dims.d[3] = w;
            }
            exec_context->setInputShape("pixel_values", in_dims);
            exec_context->setTensorAddress("pixel_values", (void*) d_in);
            exec_context->setTensorAddress(output_tensor_name.c_str(), (void*) d_out);
            bool ok;
            {
                std::lock_guard<std::mutex> lock(*trt_mutex);
                ok = exec_context->enqueueV3(cu_stream);
            }
            if (ok && cuda.cuStreamSynchronize) cuda.cuStreamSynchronize(cu_stream);
            cuda.cuMemFree(d_in);
            cuda.cuMemFree(d_out);
            BOOST_LOG(info) << "Depth estimator warmup inference complete (" << w << 'x' << h
                            << (ok ? ")." : "); enqueue failed, first frame may stall.");
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
                g_engines[model_name].context_pool.push_back(exec_context);
                exec_context = nullptr;
            }
            // TRT runtime/engine are cached globally, do not destroy them here.
        }

        // The depth SRV handed back to the reprojection: the color-guided 2x upsample when
        // enabled, else the raw depth. Also the value returned on frames where we reuse the
        // last depth (stream busy or off-cadence).
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> output_srv() {
            return (guided_upsample && guided_depth_srv) ? guided_depth_srv : depth_srv;
        }

        // (Re)build the two constant buffers. All contents are session-constant once the model
        // resolution is fixed, so they are immutable buffers created once -- rebuilt only if
        // is_hdr ever flips (an HDR change normally recreates the whole encode device anyway).
        void ensure_cbuffers(bool is_hdr) {
            if (cb_is_hdr == (int)is_hdr && cbuffer) {
                return;
            }
            cb_is_hdr = (int)is_hdr;

            D3D11_BUFFER_DESC cb_desc = {};
            cb_desc.Usage = D3D11_USAGE_IMMUTABLE;
            cb_desc.ByteWidth = 32;
            cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

            // Shared depth-pass constants: {target_w, target_h, is_hdr, ema_alpha,
            // minmax_alpha, reduce_threads, output_transform, depth_shift} (see buffer_to_tex_cs.hlsl).
            uint32_t cb[8] = {};
            float* cbf = (float*)cb;
            cb[0] = (uint32_t)target_w;
            cb[1] = (uint32_t)target_h;
            cb[2] = is_hdr ? 1u : 0u;
            cbf[3] = ema_alpha;
            cbf[4] = minmax_alpha;
            cb[5] = reduce_groups * 256u;  // total threads for the reduction grid-stride
            cb[6] = output_transform;  // 0=identity (DA-V2 disparity), 1=shifted reciprocal (DA-V3 depth)
            cbf[7] = depth_shift;  // shift in 1/(depth + depth_shift) when output_transform==1
            D3D11_SUBRESOURCE_DATA sd = {cb, 0, 0};
            cbuffer.Reset();
            device->CreateBuffer(&cb_desc, &sd, &cbuffer);

            if (guided_upsample) {
                // Guided passes: {in_w, in_h, out_w, out_h, inv2sig_sp2, inv2sig_r2, is_hdr, radius}.
                const float kernel_radius = 5.0f;  // low-res texels; spans the model's ~8-9 texel edge ramp
                const float sigma_sp = kernel_radius * 0.5f;
                uint32_t gb[8] = {};
                float* gbf = (float*)gb;
                gb[0] = (uint32_t)target_w;
                gb[1] = (uint32_t)target_h;
                gb[2] = (uint32_t)target_w * 2;
                gb[3] = (uint32_t)target_h * 2;
                gbf[4] = 1.0f / (2.0f * sigma_sp * sigma_sp);
                gbf[5] = 1.0f / (2.0f * guided_sigma * guided_sigma);
                gb[6] = is_hdr ? 1u : 0u;
                gbf[7] = kernel_radius;
                D3D11_SUBRESOURCE_DATA gsd = {gb, 0, 0};
                guided_cbuffer.Reset();
                device->CreateBuffer(&cb_desc, &gsd, &guided_cbuffer);
            }
        }

        // Re-snap the (possibly stale) low-res depth to the CURRENT frame's color edges: pass 1
        // downsamples the color to the depth grid, pass 2 joint-bilaterally upsamples depth to
        // 2x res using the full-res color as the reference. Runs every frame (cheap: taps hit
        // two low-res textures), including cadence-skip frames -- that is what keeps silhouettes
        // tracking the image while the depth itself refreshes at depth_fps.
        void run_guided(ID3D11ShaderResourceView* color_srv, bool is_hdr) {
            if (!guided_upsample || !guided_depth_uav || !guide_uav || !color_srv ||
                !depth_guide_downsample_cs || !depth_guided_upsample_cs) {
                return;
            }
            ensure_cbuffers(is_hdr);
            if (!guided_cbuffer) {
                return;
            }

            const UINT out_w = (UINT) target_w * 2, out_h = (UINT) target_h * 2;

            ID3D11UnorderedAccessView* null_uav = nullptr;
            ID3D11ShaderResourceView* null_srvs[3] = {nullptr, nullptr, nullptr};

            // Pass 1: full-res color -> low-res guide.
            context->CSSetShader(depth_guide_downsample_cs.Get(), nullptr, 0);
            context->CSSetConstantBuffers(0, 1, guided_cbuffer.GetAddressOf());
            context->CSSetShaderResources(0, 1, &color_srv);
            context->CSSetUnorderedAccessViews(0, 1, guide_uav.GetAddressOf(), nullptr);
            context->CSSetSamplers(0, 1, linear_sampler.GetAddressOf());
            context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);
            context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
            context->CSSetShaderResources(0, 1, null_srvs);

            // Pass 2: joint-bilateral upsample of the UN-dilated depth, guided by color.
            context->CSSetShader(depth_guided_upsample_cs.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srvs[3] = {depth_srv.Get(), guide_srv.Get(), color_srv};
            context->CSSetShaderResources(0, 3, srvs);
            context->CSSetUnorderedAccessViews(0, 1, guided_depth_uav.GetAddressOf(), nullptr);
            context->Dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
            context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
            context->CSSetShaderResources(0, 3, null_srvs);
        }

        // Called once per video frame. Measures the video frame rate from the inter-call
        // period and derives effective_interval so depth refreshes near depth_fps. A
        // ±0.5-frame deadband keeps the interval from oscillating when the measured fps
        // sits near an integer boundary.
        void update_cadence() {
            auto now = std::chrono::steady_clock::now();
            if (last_call_time.time_since_epoch().count() != 0) {
                float dt = std::chrono::duration<float>(now - last_call_time).count();
                if (dt > 1e-4f && dt < 0.5f) {  // ignore first call and long stalls (paused/occluded)
                    float inst = 1.0f / dt;
                    measured_fps = (measured_fps <= 0.0f) ? inst : (measured_fps * 0.95f + inst * 0.05f);
                }
            }
            last_call_time = now;

            if (depth_fps > 0.0f && measured_fps > 1.0f) {
                float ideal = measured_fps / depth_fps;
                if (ideal > effective_interval + 0.5f) {
                    effective_interval += 1;
                } else if (effective_interval > 1 && ideal < effective_interval - 0.5f) {
                    effective_interval -= 1;
                }
            } else {
                effective_interval = 1;  // auto disabled, or still warming up
            }

            if (effective_interval != last_logged_interval) {
                BOOST_LOG(info) << "Depth cadence: video ~" << (int)(measured_fps + 0.5f) << "fps -> inference every "
                                << effective_interval << " frame(s) (~" << (int)(measured_fps / effective_interval + 0.5f) << "fps depth)";
                last_logged_interval = effective_interval;
            }
        }

        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> estimate(ID3D11ShaderResourceView* input_srv, bool is_hdr) {
            if (!exec_context || !rgb_to_nchw_cs || !buffer_to_tex_cs || !input_srv) return nullptr;

            auto& cuda = cuda_driver_api::get();
            if (!cuda.is_valid()) {
                BOOST_LOG(error) << "CUDA Driver API is not available.";
                return nullptr;
            }

            if (cuda_ctx) {
                cuda.cuCtxSetCurrent(cuda_ctx);
            }

            // Measure video fps and derive the depth-inference interval (called every frame).
            update_cadence();

            // Prevent GPU starvation: if the previous AI frame is still crunching, drop this frame.
            // This prevents an infinite queue of heavy TensorRT workloads from starving the DWM and Edge Browser.
            if (cu_stream && cuda.cuStreamQuery) {
                auto q = cuda.cuStreamQuery(cu_stream);
                if (q == CUDA_ERROR_NOT_READY) {
                    // Still re-snap the stale depth to THIS frame's color edges (D3D-only, cheap).
                    run_guided(input_srv, is_hdr);
                    return output_srv();
                }
                if (q != CUDA_SUCCESS && !stream_error_logged) {
                    // Anything other than success/not-ready means the stream (or context) is
                    // broken; keep going -- enqueueV3 below will report per-frame failures.
                    BOOST_LOG(error) << "cuStreamQuery failed: " << q;
                    stream_error_logged = true;
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
                if (dynamic_width) {
                    // Height is baked into the engine (fixed_h); only the width tracks the source
                    // aspect. The engine's width profile spans [fixed_h .. fixed_h*4] (1:1 .. 4:1),
                    // so this covers every landscape aspect (4:3 -> 16:9 -> ultrawide/32:9) with one
                    // engine. Height must NOT be rescaled (that would break the baked binding), so the
                    // generic short-side/native-downscale path below is deliberately bypassed.
                    // Portrait (aspect < 1) is not a streaming case; it clamps to 1:1 (square).
                    target_h = fixed_h;
                    float a = std::min(std::max(aspect_ratio, 1.0f), std::min(max_aspect, 4.0f));
                    target_w = std::max(14, (int)std::round(fixed_h * a / 14.0f) * 14);
                } else {
                    // Short-side budget (iw3-style): pin the short side so vertical depth detail
                    // is constant regardless of aspect ratio; the long side grows with aspect,
                    // capped by max_aspect to bound cost.
                    int short_side = std::max(14, (int)std::round((float)depth_short_side / 14.0f) * 14);
                    if (aspect_ratio >= 1.0f) {
                        target_h = short_side;
                        target_w = (int)std::round(short_side * std::min(aspect_ratio, max_aspect));
                    } else {
                        target_w = short_side;
                        target_h = (int)std::round(short_side * std::min(1.0f / aspect_ratio, max_aspect));
                    }

                    target_h = std::max(14, (int)std::round((float)target_h / 14.0f) * 14);
                    target_w = std::max(14, (int)std::round((float)target_w / 14.0f) * 14);

                    // Cap the model input aspect-preserving against two limits: the TensorRT
                    // engine's MAX profile (1008), and the frame's native resolution -- never
                    // upscale a small input up to the budget (matches iw3's limit_resolution).
                    // Scaling both axes by a single factor keeps the depth undistorted.
                    int max_w = std::min(1008, (int)input_desc.Width);
                    int max_h = std::min(1008, (int)input_desc.Height);
                    if (target_w > max_w || target_h > max_h) {
                        float s = std::min((float)max_w / (float)target_w, (float)max_h / (float)target_h);
                        target_w = std::max(14, (int)std::round((float)target_w * s / 14.0f) * 14);
                        target_h = std::max(14, (int)std::round((float)target_h * s / 14.0f) * 14);
                    }
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

                if (guided_upsample) {
                    // Low-res color guide (same grid as the depth map). RGBA16F: guaranteed
                    // UAV-store format, read back as an SRV in the upsample pass.
                    D3D11_TEXTURE2D_DESC guide_desc = tex_desc;
                    guide_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                    device->CreateTexture2D(&guide_desc, nullptr, &guide_tex);
                    device->CreateUnorderedAccessView(guide_tex.Get(), nullptr, &guide_uav);
                    device->CreateShaderResourceView(guide_tex.Get(), nullptr, &guide_srv);

                    // Color-edge-snapped depth at 2x the model resolution; what the
                    // reprojection actually samples when guided upsampling is on.
                    D3D11_TEXTURE2D_DESC gd_desc = tex_desc;
                    gd_desc.Width = (UINT) target_w * 2;
                    gd_desc.Height = (UINT) target_h * 2;
                    device->CreateTexture2D(&gd_desc, nullptr, &guided_depth_tex);
                    device->CreateUnorderedAccessView(guided_depth_tex.Get(), nullptr, &guided_depth_uav);
                    device->CreateShaderResourceView(guided_depth_tex.Get(), nullptr, &guided_depth_srv);
                }

                // Clear the depth textures to 0.0f: depth_tex so the EMA shader initializes
                // correctly on the first frame instead of blending with undefined memory, and
                // the guided output so output_srv() returns flat (not garbage) depth on the
                // frames before the first guided pass runs.
                const float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                context->ClearUnorderedAccessViewFloat(depth_uav.Get(), clear_color);
                if (guided_depth_uav) {
                    context->ClearUnorderedAccessViewFloat(guided_depth_uav.Get(), clear_color);
                }
                
                auto res1 = cuda.cuGraphicsD3D11RegisterResource(&cuda_in_res, tensor_in_buf.Get(), 0);
                auto res2 = cuda.cuGraphicsD3D11RegisterResource(&cuda_out_res, tensor_out_buf.Get(), 0);
                if (res1 != 0 || res2 != 0) {
                    BOOST_LOG(error) << "cuGraphicsD3D11RegisterResource failed: " << res1 << ", " << res2;
                }
            }


            // Depth-update cadence: at high video framerates the depth map does not need to
            // refresh every frame -- the temporal EMA hides a lower update rate, and skipping
            // inference frees GPU time for the encoder (and cuts inference/encode contention).
            // Reuse the last depth on skipped frames. Runs the first frame, then every
            // effective_interval-th frame thereafter.
            frame_counter++;
            if (effective_interval > 1 && (frame_counter % (unsigned)effective_interval) != 1u) {
                // Off-cadence frame: depth is reused, but re-snap it to this frame's color edges
                // so silhouettes keep tracking the image between inference frames.
                run_guided(input_srv, is_hdr);
                return output_srv();
            }

            // tensor_out_buf now holds the finished raw disparity from the previous inference
            // (fully unmapped from CUDA), so the passes below don't block the CPU thread.

            // Shared constants for buffer_to_tex_cs, the min/max passes and rgb_to_nchw_cs.
            // Session-constant, so the buffer is built once (immutable), not mapped per frame.
            ensure_cbuffers(is_hdr);
            if (!cbuffer) {
                return nullptr;
            }

            if (has_previous_frame) {
                // 3a. Per-frame min/max normalization (GPU-resident; no CPU readback).
                // Depth Anything V2's relative output is affine-invariant, so this is required
                // for a stable parallax scale.
                if (depth_minmax_cs && depth_minmax_ema_cs && minmax_raw_uav && minmax_ema_uav) {
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
            }

            // 3c. Guided upsample: snap the freshly-updated depth to this frame's color edges.
            run_guided(input_srv, is_hdr);

            // 1. D3D11 Compute Shader: Resize & Normalize to NCHW FP32 Buffer (for CURRENT frame)
            context->CSSetShader(rgb_to_nchw_cs.Get(), nullptr, 0);
            context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
            context->CSSetShaderResources(0, 1, &input_srv);
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
                // Fixed-shape engines have their input dims baked in; setInputShape is unnecessary
                // (and target_w/target_h MUST equal the export resolution or the bindings mismatch).
                if (!fixed_shape) {
                    nvinfer1::Dims in_dims {};
                    if (input_rank == 5) {
                        in_dims.nbDims = 5;
                        in_dims.d[0] = 1; in_dims.d[1] = 1; in_dims.d[2] = 3; in_dims.d[3] = target_h; in_dims.d[4] = target_w;
                    } else {
                        in_dims.nbDims = 4;
                        in_dims.d[0] = 1; in_dims.d[1] = 3; in_dims.d[2] = target_h; in_dims.d[3] = target_w;
                    }
                    if (!exec_context->setInputShape("pixel_values", in_dims)) {
                        BOOST_LOG(error) << "TensorRT setInputShape failed for " << target_w << "x" << target_h
                                         << " (outside the engine's optimization profile?)";
                    }
                }
                exec_context->setTensorAddress("pixel_values", (void*)d_in);
                exec_context->setTensorAddress(output_tensor_name.c_str(), (void*)d_out);
                {
                    // Serialize TensorRT async enqueue to avoid driver-level concurrent execution faults
                    std::lock_guard<std::mutex> lock(*trt_mutex);
                    if (!exec_context->enqueueV3(cu_stream) && !stream_error_logged) {
                        BOOST_LOG(error) << "TensorRT enqueueV3 failed; depth will stop updating.";
                        stream_error_logged = true;
                    }
                }
            }
            
            cuda.cuGraphicsUnmapResources(2, resources, cu_stream);
            
            has_previous_frame = true;

            return output_srv();
        }
    };

    video_depth_estimator::video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device,
                          Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
                          const std::filesystem::path& assets_dir,
                          const config::video_t::sbs_t& cfg,
                          const config::depth_model_info& model)
        : pimpl(std::make_unique<impl>(device, context, assets_dir, cfg, model)) {}

    video_depth_estimator::~video_depth_estimator() = default;

    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> video_depth_estimator::estimate_depth(ID3D11ShaderResourceView* input_srv, bool is_hdr) {
        return pimpl->estimate(input_srv, is_hdr);
    }
}
