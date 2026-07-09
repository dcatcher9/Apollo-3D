#include "video_depth_estimator.h"
#include "model_manager.h"
#include "logging.h"
#include "platform/windows/utils.h"
#include "cuda_driver_api.h"
#include "sbs_perf.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <string_view>
#include <cstring>
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

// Build the model input Dims for a given rank: rank-4 = [1,3,H,W] (DA-V2),
// rank-5 = [1,1,3,H,W] (DA-V3). Passing Dims INTO TensorRT is ABI-safe (only RETURNING a Dims
// by value across the MinGW/MSVC boundary faults), so this helper is safe to hand to the API.
static nvinfer1::Dims make_input_dims(int rank, int h, int w) {
    nvinfer1::Dims d {};
    if (rank == 5) {
        d.nbDims = 5;
        d.d[0] = 1; d.d[1] = 1; d.d[2] = 3; d.d[3] = h; d.d[4] = w;
    } else {
        d.nbDims = 4;
        d.d[0] = 1; d.d[1] = 3; d.d[2] = h; d.d[3] = w;
    }
    return d;
}

// Round x to the nearest positive multiple of `patch` (the model's spatial patch size; 14 for the
// Depth Anything family). Model input dims must be patch-aligned or TensorRT rejects the shape.
static int round_to_patch(float x, int patch = 14) {
    return std::max(patch, (int) std::round(x / patch) * patch);
}

// Ensure the shared runtime exists, deserialize `model_name`'s engine into its global slot if not
// already resident, and hand back a spare pooled execution context if one is available. The CALLER
// must hold g_trt_mutex. Context CREATION is deliberately left to the caller OUTSIDE the lock:
// createExecutionContext() allocates ~1.3 GB of scratch and takes seconds, and holding the lock
// across it would block concurrent teardowns' pool returns and other sessions' enqueues.
static nvinfer1::ICudaEngine* acquire_engine_locked(
        const std::string& model_name, const std::filesystem::path& engine_path,
        nvinfer1::IExecutionContext*& out_context, bool& out_pooled) {
    out_context = nullptr;
    out_pooled = false;
    auto& slot = g_engines[model_name];
    if (!g_runtime) {
        g_runtime = nvinfer1::createInferRuntime(gLogger);
    }
    if (g_runtime && !slot.engine) {
        std::ifstream file(engine_path, std::ios::binary);
        std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        slot.engine = g_runtime->deserializeCudaEngine(blob.data(), blob.size());
    }
    if (slot.engine && !slot.context_pool.empty()) {
        out_context = slot.context_pool.back();
        slot.context_pool.pop_back();
        out_pooled = true;
    }
    return slot.engine;
}

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
            auto dims_for = [&](int h, int w) { return make_input_dims(model.input_rank, h, w); };
            if (model.dynamic_width) {
                // Height is baked into the ONNX (fixed_h); only width is a real dynamic axis.
                // Pin H and range W from 1:1 up to ~4:1 so a single engine serves every landscape
                // aspect (4:3 -> 16:9 -> ultrawide/32:9). OPT is the common ultrawide (~2.37:1).
                const int p = std::max(1, model.patch);
                const int h = model.fixed_h;
                auto r14 = [&](double x) { return round_to_patch((float) x, p); };
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
        // An EMPTY output_tensor means "keep every output" -- used for non-depth engines whose
        // outputs are all consumed (e.g. the MLBW warp model: delta + layer_weight).
        if (!model.output_tensor.empty()) {
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
        float pct_lo;        // robust normalization: low percentile as a fraction (0 = raw min)
        float pct_hi;        // robust normalization: high percentile as a fraction (1 = raw max)
        float norm_lock_frames;  // scene lock: updates before the normalization bounds freeze (0 = off)
        bool subject_track;      // VD3D-style shaped disparity: run the subject-estimate passes
        float subject_recenter;  // recenter strength consumed by depth_subject_resolve_cs
        float subject_lock;      // anchor strength (probe: parallax subtraction; MLBW: conv blend)
        bool subject_stretch;    // apply the shape_depth_for_pop 5/95 disparity stretch
        float stretch_lo;        // low percentile for the stretch (fraction)
        float stretch_hi;        // high percentile for the stretch (fraction)
        bool use_percentile; // either percentile bound active -> histogram pass runs
        bool sync_depth;     // wait for THIS frame's inference so color and depth match (no async ghost)
        float minmax_snap;   // A1: raw-vs-EMA range ratio that snaps the scale on a scene cut (0 = off)
        float range_floor_frac;      // A3: current range < ref*frac -> compress parallax (0 = off)
        float range_floor_ref_alpha; // A3: slow-max reference-range decay
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

        // GPU-stream timing of the async TensorRT enqueues (perf benchmark; sbs_3d_perf_stats).
        // A small ring of CUDA event pairs per engine lets several inferences be in flight; the
        // elapsed time is resolved lazily once the stop event completes and pushed to sbs_perf.
        // All CUDA calls here run on the estimator thread with cuda_ctx current, like the rest
        // of estimate(); no-ops entirely when perf stats are off.
        struct perf_evt_ring {
            static constexpr int N = 4;
            CUevent start[N] {};
            CUevent stop[N] {};
            bool busy[N] {};
            int head = 0;
            const char* stage = nullptr;
        };
        perf_evt_ring perf_depth;  // "depth_infer": one DA-V2/DA3MONO inference
        perf_evt_ring perf_warp;   // "warp_infer": both eyes of the MLBW warp

        void perf_try_resolve(perf_evt_ring& r, int slot, cuda_driver_api& cuda) {
            if (!r.busy[slot] || !cuda.cuEventQuery) return;
            if (cuda.cuEventQuery(r.stop[slot]) != CUDA_SUCCESS) return;  // not finished yet
            float ms = 0.0f;
            if (cuda.cuEventElapsedTime && cuda.cuEventElapsedTime(&ms, r.start[slot], r.stop[slot]) == CUDA_SUCCESS) {
                sbs_perf::add_sample_ms(r.stage, ms);
            }
            r.busy[slot] = false;
        }
        void perf_drain(perf_evt_ring& r) {
            auto& cuda = cuda_driver_api::get();
            for (int i = 0; i < perf_evt_ring::N; i++) perf_try_resolve(r, i, cuda);
        }
        // Record a start event before an enqueue; returns the ring slot (or -1 to skip timing).
        int perf_begin(perf_evt_ring& r, CUstream stream) {
            if (!sbs_perf::enabled()) return -1;
            auto& cuda = cuda_driver_api::get();
            if (!cuda.cuEventCreate || !cuda.cuEventRecord) return -1;
            int slot = r.head;
            perf_try_resolve(r, slot, cuda);   // reclaim the slot if its prior sample is ready
            if (r.busy[slot]) return -1;       // still in flight -> drop this measurement
            if (!r.start[slot] && cuda.cuEventCreate(&r.start[slot], CU_EVENT_DEFAULT) != CUDA_SUCCESS) return -1;
            if (!r.stop[slot] && cuda.cuEventCreate(&r.stop[slot], CU_EVENT_DEFAULT) != CUDA_SUCCESS) return -1;
            if (cuda.cuEventRecord(r.start[slot], stream) != CUDA_SUCCESS) return -1;
            return slot;
        }
        // Record the stop event after the enqueue and mark the slot pending.
        void perf_end(perf_evt_ring& r, int slot, CUstream stream) {
            if (slot < 0) return;
            auto& cuda = cuda_driver_api::get();
            if (!cuda.cuEventRecord || cuda.cuEventRecord(r.stop[slot], stream) != CUDA_SUCCESS) return;
            r.busy[slot] = true;
            r.head = (r.head + 1) % perf_evt_ring::N;
        }
        void perf_destroy_events() {
            auto& cuda = cuda_driver_api::get();
            if (!cuda.cuEventDestroy) return;
            for (auto* r : {&perf_depth, &perf_warp}) {
                for (int i = 0; i < perf_evt_ring::N; i++) {
                    if (r->start[i]) cuda.cuEventDestroy(r->start[i]);
                    if (r->stop[i]) cuda.cuEventDestroy(r->stop[i]);
                    r->start[i] = r->stop[i] = nullptr;
                }
            }
        }

        // Caching
        int target_w = 0;
        int target_h = 0;
        UINT reduce_groups = 0;  // threadgroups for the min/max reduction (groups * 256 = total threads)
        int cb_is_hdr = -1;  // is_hdr baked into the constant buffers (-1 = not built yet)

        Microsoft::WRL::ComPtr<ID3D11ComputeShader> rgb_to_nchw_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> buffer_to_tex_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_minmax_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_minmax_ema_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_hist_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_subject_hist_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_subject_resolve_cs;
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
        Microsoft::WRL::ComPtr<ID3D11Buffer> minmax_raw_stage;  // CPU-readable copy for [NORMDBG]
        unsigned norm_log_counter = 0;  // per-frame raw-stats trajectory for the norm-window study
        Microsoft::WRL::ComPtr<ID3D11Buffer> hist_buf;  // 256 uint bins for percentile normalization
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> hist_uav;
        Microsoft::WRL::ComPtr<ID3D11Buffer> subject_hist_buf;  // 256 weighted bins for subject tracking
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subject_hist_uav;
        Microsoft::WRL::ComPtr<ID3D11Buffer> subject_plain_buf;  // 256 unweighted bins for the stretch 5/95
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subject_plain_uav;
        Microsoft::WRL::ComPtr<ID3D11Buffer> subject_buf;  // float4 {delta, scurve, subj_ema, init}
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subject_uav;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subject_srv;
        Microsoft::WRL::ComPtr<ID3D11Buffer> subject_stage;  // CPU-readable copy for the debug log
        unsigned subject_log_counter = 0;  // paces the [SUBJDBG] readback (every 24 depth updates)
        
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

        // Learned warp (iw3 MLBW): a second, tiny TRT engine that turns the depth map into
        // per-eye multi-layer warp fields (2 deltas + 2 softmax weights per texel), composited
        // by sbs_mlbw_composite_ps instead of the probe-search reprojection. Runs on its OWN
        // CUDA stream so its ~1 ms result can be consumed as soon as it lands, without waiting
        // behind the much longer depth inference on cu_stream.
        bool learned_warp;
        std::string warp_model;  // file stem; engine cached as <warp_model>.engine
        std::string warp_model_url;
        float warp_divergence;  // config divergence (parallax fraction of width)
        float warp_focal;  // config focal_plane
        float warp_floor;  // config depth_floor
        float warp_border;  // config border_fade (>0 = ramp features near L/R edges)
        nvinfer1::IExecutionContext* warp_context = nullptr;
        CUstream warp_stream = nullptr;
        int warp_w = 0, warp_h = 0;  // model grid (from the stem naming convention)
        int warp_layers = 0;  // model layer count (from the stem, e.g. mlbw_l2/l4; <= 4)
        bool warp_pending = false;  // inference enqueued; outputs not yet consumed
        bool depth_context_pooled = false;  // context reused from the pool (modules already loaded -> skip warmup)
        bool warp_context_pooled = false;  // ditto for the warp context
        bool warp_resources_registered = false;  // CUDA-D3D interop registered (lazily, on the encode thread)
        bool warp_fields_valid = false;  // field textures hold a usable result
        bool warp_error_logged = false;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> mlbw_input_cs;
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> mlbw_field_cs;
        Microsoft::WRL::ComPtr<ID3D11Buffer> mlbw_input_cbuffer;
        Microsoft::WRL::ComPtr<ID3D11Buffer> mlbw_field_cbuffer[2];  // per eye
        Microsoft::WRL::ComPtr<ID3D11Buffer> mlbw_in_buf[2];  // [3*H*W] fp32, left/right
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> mlbw_in_uav[2];
        Microsoft::WRL::ComPtr<ID3D11Buffer> mlbw_delta_buf[2];  // [L*H*W] fp32
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mlbw_delta_srv[2];
        Microsoft::WRL::ComPtr<ID3D11Buffer> mlbw_weight_buf[2];
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mlbw_weight_srv[2];
        Microsoft::WRL::ComPtr<ID3D11Texture2D> delta_tex[2];  // RGBA32F per-layer deltas
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> delta_tex_uav[2];
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> delta_tex_srv[2];
        Microsoft::WRL::ComPtr<ID3D11Texture2D> weight_tex[2];  // RGBA32F per-layer weights
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> weight_tex_uav[2];
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> weight_tex_srv[2];
        CUgraphicsResource cuda_warp_res[6] = {};  // in_l, in_r, delta_l, weight_l, delta_r, weight_r

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

        // Load the MLBW warp engine and create every GPU resource it needs. Called from the
        // ctor WITHOUT g_trt_mutex held (a first-ever engine build takes ~14 s and must not
        // block other sessions' enqueues); the engine-slot access below takes the lock itself.
        // On any failure, learned_warp is switched off and the pipeline falls back to the
        // probe-search reprojection.
        void setup_learned_warp(const std::filesystem::path& assets_dir) {
            auto fail = [&](const char* why) {
                BOOST_LOG(warning) << "Learned warp disabled: " << why << " -- falling back to the probe-search reprojection.";
                learned_warp = false;
            };

            // The warp model is not a depth model, but it reuses the same engine plumbing: a
            // synthesized registry entry with fixed_shape (no optimization profile) and an empty
            // output_tensor (keep ALL outputs: delta + layer_weight). Engine cached as
            // "<warp_model>.engine" (rank-4 default = no recipe tag).
            config::depth_model_info warp_info;
            warp_info.name = warp_model;
            warp_info.url = warp_model_url;
            warp_info.fixed_shape = true;
            warp_info.output_tensor = "";

            auto path = ensure_model_available(assets_dir, warp_model, warp_model_url);
            if (path.empty()) {
                return fail("warp model not found (place <warp_model>.onnx in the assets dir or set sbs_3d_warp_model_url)");
            }
            if (path.extension() == ".onnx") {
                precompile_tensorrt_engine(assets_dir, warp_info);
                path = ensure_model_available(assets_dir, warp_model, warp_model_url);
            }
            if (path.extension() != ".engine") {
                return fail("no engine after compilation");
            }

            nvinfer1::ICudaEngine* warp_engine = nullptr;
            {
                std::lock_guard<std::mutex> lock(g_trt_mutex);
                warp_engine = acquire_engine_locked(warp_model, path, warp_context, warp_context_pooled);
            }
            if (warp_context_pooled) {
                BOOST_LOG(info) << "Reusing pooled TensorRT execution context (" << warp_model << ").";
            }
            if (warp_engine && !warp_context) {
                // Outside g_trt_mutex (same rationale as the depth context): creation takes
                // seconds and must not block a concurrent teardown's pool return or enqueues.
                BOOST_LOG(info) << "Creating TensorRT execution context for '" << warp_model << "'...";
                warp_context = warp_engine->createExecutionContext();
            }
            if (!warp_context || !warp_engine) {
                return fail("engine load failed");
            }

            // The export is fixed-shape; the grid dims come from the model STEM (naming
            // convention "..._<W>x<H>...", e.g. mlbw_l2_798x336_fp16), and depth is resampled
            // onto that grid by mlbw_input_cs, so it need not match the DA resolution.
            //
            // ABI WARNING: the dims can NOT be read from the engine -- getTensorShape() returns
            // a Dims struct BY VALUE, which crashes across the MinGW/MSVC boundary (return-slot
            // pointer ordering differs; this SIGSEGV'd in nvinfer when tried). Only const char*
            // / enum / integer-returning TRT APIs are safe here.
            {
                std::smatch m;
                static const std::regex grid_re("_(\\d+)x(\\d+)");
                if (!std::regex_search(warp_model, m, grid_re)) {
                    return fail("cannot determine the warp grid: name the model '<stem>_<W>x<H>[_...]' (e.g. mlbw_l2_798x336_fp16)");
                }
                warp_w = std::stoi(m[1].str());
                warp_h = std::stoi(m[2].str());
                if (warp_w < 64 || warp_h < 64 || warp_w > 4096 || warp_h > 4096) {
                    return fail("implausible warp grid parsed from the model name");
                }
                std::smatch lm;
                static const std::regex layers_re("mlbw_l(\\d+)");
                if (!std::regex_search(warp_model, lm, layers_re)) {
                    return fail("cannot determine the layer count: name the model 'mlbw_l<N>_...' (e.g. mlbw_l2_798x336_fp16)");
                }
                warp_layers = std::stoi(lm[1].str());
                if (warp_layers < 1 || warp_layers > 4) {
                    return fail("unsupported layer count (the field textures pack at most 4 layers)");
                }
            }
            auto* eng = warp_engine;
            bool have_in = false, have_delta = false, have_weight = false;
            for (int i = 0; i < eng->getNbIOTensors(); i++) {
                std::string_view tname = eng->getIOTensorName(i);
                have_in = have_in || tname == "mlbw_input";
                have_delta = have_delta || tname == "delta";
                have_weight = have_weight || tname == "layer_weight";
            }
            if (!have_in || !have_delta || !have_weight) {
                return fail("engine is missing mlbw_input/delta/layer_weight tensors (is this an mlbw_l2 delta_output export?)");
            }
            for (const char* tname : {"mlbw_input", "delta", "layer_weight"}) {
                if (eng->getTensorDataType(tname) != nvinfer1::DataType::kFLOAT) {
                    return fail("warp engine I/O is not FP32 (use an fp16-weights/fp32-IO export)");
                }
            }
            BOOST_LOG(info) << "Learned warp (MLBW) engine loaded: grid " << warp_w << "x" << warp_h
                            << ", " << warp_layers << " layers";

            if (!compile_shader(assets_dir / "shaders" / "directx" / "mlbw_input_cs.hlsl", mlbw_input_cs) ||
                !compile_shader(assets_dir / "shaders" / "directx" / "mlbw_field_cs.hlsl", mlbw_field_cs)) {
                return fail("mlbw shaders failed to compile");
            }

            auto& cuda = cuda_driver_api::get();
            if (!cuda.is_valid() || cuda.cuStreamCreate(&warp_stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
                return fail("could not create the warp CUDA stream");
            }

            // Tensor buffers: input [3*H*W] and outputs [2*H*W] per eye, CUDA-mapped.
            const UINT plane = (UINT) warp_w * (UINT) warp_h;
            D3D11_BUFFER_DESC bd = {};
            bd.Usage = D3D11_USAGE_DEFAULT;
            bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            bd.StructureByteStride = sizeof(float);
            for (int e = 0; e < 2; e++) {
                bd.ByteWidth = plane * 3 * sizeof(float);
                bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
                device->CreateBuffer(&bd, nullptr, &mlbw_in_buf[e]);
                device->CreateUnorderedAccessView(mlbw_in_buf[e].Get(), nullptr, &mlbw_in_uav[e]);

                bd.ByteWidth = plane * (UINT) warp_layers * sizeof(float);
                bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                device->CreateBuffer(&bd, nullptr, &mlbw_delta_buf[e]);
                device->CreateShaderResourceView(mlbw_delta_buf[e].Get(), nullptr, &mlbw_delta_srv[e]);
                device->CreateBuffer(&bd, nullptr, &mlbw_weight_buf[e]);
                device->CreateShaderResourceView(mlbw_weight_buf[e].Get(), nullptr, &mlbw_weight_srv[e]);

                D3D11_TEXTURE2D_DESC td = {};
                td.Width = (UINT) warp_w;
                td.Height = (UINT) warp_h;
                td.MipLevels = 1;
                td.ArraySize = 1;
                td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_DEFAULT;
                td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
                device->CreateTexture2D(&td, nullptr, &delta_tex[e]);
                device->CreateUnorderedAccessView(delta_tex[e].Get(), nullptr, &delta_tex_uav[e]);
                device->CreateShaderResourceView(delta_tex[e].Get(), nullptr, &delta_tex_srv[e]);
                device->CreateTexture2D(&td, nullptr, &weight_tex[e]);
                device->CreateUnorderedAccessView(weight_tex[e].Get(), nullptr, &weight_tex_uav[e]);
                device->CreateShaderResourceView(weight_tex[e].Get(), nullptr, &weight_tex_srv[e]);
            }
            // NOTE: the CUDA-D3D registration of these buffers is deliberately DEFERRED to the
            // first enqueue_warp() on the ENCODE thread (see register_warp_resources). Doing
            // cuGraphicsD3D11RegisterResource here -- on the background build thread while the
            // encode thread is actively driving the same D3D device (flat SBS + NVENC) -- hung
            // the constructor twice (driver-level CUDA/D3D lock inversion; sessions became
            // unkillable). The depth pipeline's interop registration has always run on the
            // encode thread inside estimate(), which is the proven-safe pattern.

            // Feature-plane constants (iw3 make_divergence_feature_value): the config warp is
            // parallax = (floor + (1-floor)*d - focal) * divergence, which in iw3 terms is
            //   divergence% = (1-floor)*divergence / 0.005,  convergence = (focal-floor)/(1-floor)
            // so divergence_pix = iw3_div * 0.5 * 0.01 * max(W,H) = (1-floor)*divergence * max(W,H).
            const float slope = (1.0f - warp_floor) * warp_divergence;
            const float base_size = (float) std::max(warp_w, warp_h);
            const float div_pix = slope * base_size;
            const float conv = std::min(1.0f, std::max(0.0f, (warp_focal - warp_floor) / std::max(1e-6f, 1.0f - warp_floor)));
            // Border ramp width per iw3's preserve_screen_border, gated by border_fade > 0.
            const float border_texels = (warp_border > 0.0f) ? std::round(1.5f * slope * (float) warp_w) : 0.0f;

            D3D11_BUFFER_DESC cbd = {};
            cbd.Usage = D3D11_USAGE_IMMUTABLE;
            cbd.ByteWidth = 32;
            cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            struct { uint32_t fw, fh; float div_feat, conv_feat, border, subject_track_f, subject_lock_f, conv_cfg; } icb = {
                (uint32_t) warp_w, (uint32_t) warp_h,
                div_pix / 32.0f, -div_pix * conv / 32.0f, border_texels,
                subject_track ? 1.0f : 0.0f, subject_lock, conv
            };
            D3D11_SUBRESOURCE_DATA sd = {&icb, 0, 0};
            device->CreateBuffer(&cbd, &sd, &mlbw_input_cbuffer);
            for (uint32_t e = 0; e < 2; e++) {
                struct { uint32_t fw, fh, is_right, layers; float pad2[4]; } fcb = {(uint32_t) warp_w, (uint32_t) warp_h, e, (uint32_t) warp_layers, {}};
                D3D11_SUBRESOURCE_DATA fsd = {&fcb, 0, 0};
                device->CreateBuffer(&cbd, &fsd, &mlbw_field_cbuffer[e]);
            }
            if (!mlbw_input_cbuffer || !mlbw_field_cbuffer[0] || !mlbw_field_cbuffer[1]) {
                return fail("constant buffer creation failed");
            }

            // Warm up the warp engine here (background construction thread) so its one-time CUDA
            // module load doesn't stall the encode thread on the first real enqueue_warp() --
            // the same treatment warmup_inference() gives the depth engine. Pure CUDA/TRT into
            // throwaway scratch (no D3D immediate-context calls, which the encode thread owns).
            // POOLED contexts skip this: their engine's modules are already loaded process-wide,
            // so the warmup is pure waste -- and an immediate re-enqueue on a pooled context of a
            // MULTI-STREAM engine (l4 uses TRT aux streams) hung the constructor once (2026-07-06
            // log: build stuck after "engine loaded", session unkillable). Fresh contexts only.
            if (!warp_context_pooled) {
                const size_t in_elems = (size_t) 3 * warp_w * warp_h;
                const size_t out_elems = (size_t) warp_layers * warp_w * warp_h;
                CUdeviceptr d_in = 0, d_delta = 0, d_weight = 0;
                if (cuda.cuMemAlloc(&d_in, in_elems * sizeof(float)) == CUDA_SUCCESS &&
                    cuda.cuMemAlloc(&d_delta, out_elems * sizeof(float)) == CUDA_SUCCESS &&
                    cuda.cuMemAlloc(&d_weight, out_elems * sizeof(float)) == CUDA_SUCCESS) {
                    bool ok;
                    {
                        std::lock_guard<std::mutex> lock(g_trt_mutex);
                        warp_context->setTensorAddress("mlbw_input", (void*) d_in);
                        warp_context->setTensorAddress("delta", (void*) d_delta);
                        warp_context->setTensorAddress("layer_weight", (void*) d_weight);
                        ok = warp_context->enqueueV3(warp_stream);
                    }
                    if (ok && cuda.cuStreamSynchronize) cuda.cuStreamSynchronize(warp_stream);
                    BOOST_LOG(info) << "Learned warp warmup inference complete" << (ok ? "." : " (enqueue failed).");
                }
                if (d_in) cuda.cuMemFree(d_in);
                if (d_delta) cuda.cuMemFree(d_delta);
                if (d_weight) cuda.cuMemFree(d_weight);
            }
        }

        impl(Microsoft::WRL::ComPtr<ID3D11Device> d, Microsoft::WRL::ComPtr<ID3D11DeviceContext> c, const std::filesystem::path& assets_dir, const config::video_t::sbs_t& cfg, const config::depth_model_info& model)
            : device(d), context(c), ema_alpha((float)cfg.ema),
              depth_short_side(std::max(196, cfg.depth_short_side)), max_aspect(std::max(1.0f, (float)cfg.depth_max_aspect)),
              minmax_alpha((float)cfg.minmax_ema),
              pct_lo((float)(cfg.norm_pct_lo / 100.0)),
              pct_hi((float)(cfg.norm_pct_hi / 100.0)),
              norm_lock_frames((float)cfg.norm_lock_frames),
              subject_track(cfg.subject_track),
              subject_recenter((float)cfg.subject_recenter),
              subject_lock((float)cfg.subject_lock),
              subject_stretch(cfg.subject_stretch),
              stretch_lo((float)cfg.stretch_lo), stretch_hi((float)cfg.stretch_hi),
              use_percentile(cfg.norm_pct_lo > 0.0 || cfg.norm_pct_hi < 100.0),
              sync_depth(cfg.sync_depth),
              minmax_snap((float)cfg.minmax_snap),
              range_floor_frac((float)cfg.range_floor),
              range_floor_ref_alpha(0.004f),  // ~decays over a few hundred depth updates
              depth_fps((float)cfg.depth_fps),
              // The learned warp wants the model's raw SOFT depth: texel-sharp guided edges
              // are out of its training distribution (they render as a staircase fringe).
              guided_upsample(cfg.learned_warp ? false : cfg.guided_upsample),
              guided_sigma(std::max(0.01f, (float)cfg.guided_sigma)),
              model_name(model.name), model_url(model.url),
              input_rank(model.input_rank), output_transform((uint32_t)model.output_transform),
              depth_shift(std::max(0.001f, (float)cfg.depth_shift)),
              fixed_shape(model.fixed_shape),
              dynamic_width(model.dynamic_width), fixed_h(model.fixed_h),
              output_tensor_name(model.output_tensor),
              learned_warp(cfg.learned_warp), warp_model(cfg.warp_model), warp_model_url(cfg.warp_model_url),
              warp_divergence((float)cfg.divergence), warp_focal((float)cfg.focal_plane),
              warp_floor((float)cfg.depth_floor), warp_border((float)cfg.border_fade)
        {
            // Perf benchmark: enable per-stage timing for this run and reset the rolling window
            // so it reflects this model/mode rather than blending across a switch.
            perf_depth.stage = "depth_infer";
            perf_warp.stage = "warp_infer";
            sbs_perf::set_enabled(cfg.perf_stats);
            if (cfg.perf_stats) sbs_perf::reset();

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
            // Load (once) the engine for THIS model into its own slot and take a pooled execution
            // context if one is free. Different models coexist; switching models never reuses a
            // stale engine and never needs a restart.
            engine = acquire_engine_locked(model_name, model_path, exec_context, depth_context_pooled);
            runtime = g_runtime;
            if (depth_context_pooled) {
                BOOST_LOG(info) << "Reusing pooled TensorRT execution context.";
            }
            auto& slot = g_engines[model_name];

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

            trt_mutex = &g_trt_mutex;
            }  // release g_trt_mutex before the shader/buffer setup and warmup below

            if (engine && !exec_context) {
                // Pool empty. On a back-to-back session rebuild the previous estimator is often
                // still tearing down on the async-teardown thread and will return its context to
                // the pool momentarily -- wait briefly for that before paying seconds (and ~1.3 GB
                // of device scratch) for a fresh context.
                for (int i = 0; i < 10 && !exec_context; i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    std::lock_guard<std::mutex> lock(g_trt_mutex);
                    auto& pool = g_engines[model_name].context_pool;
                    if (!pool.empty()) {
                        exec_context = pool.back();
                        pool.pop_back();
                        depth_context_pooled = true;
                        BOOST_LOG(info) << "Reusing pooled TensorRT execution context (freed by a racing teardown).";
                    }
                }
                if (!exec_context) {
                    // Deliberately OUTSIDE g_trt_mutex: creation allocates device scratch and can
                    // take many seconds; holding the lock would block a concurrent estimator
                    // destructor from returning its context to the pool (observed 46 s teardown)
                    // and any concurrent enqueueV3. ICudaEngine is thread-safe for this call.
                    BOOST_LOG(info) << "Creating TensorRT execution context (allocates device scratch; may take several seconds)...";
                    exec_context = engine->createExecutionContext();
                }
            }

            // Learned warp engine + resources (takes g_trt_mutex itself, briefly; a first-ever
            // engine build ~14 s runs here on the background construction thread).
            if (learned_warp) {
                setup_learned_warp(assets_dir);
            }

            // Compile Shaders
            compile_shader(assets_dir / "shaders" / "directx" / "rgb_to_nchw_cs.hlsl", rgb_to_nchw_cs);
            compile_shader(assets_dir / "shaders" / "directx" / "buffer_to_tex_cs.hlsl", buffer_to_tex_cs);
            compile_shader(assets_dir / "shaders" / "directx" / "depth_minmax_cs.hlsl", depth_minmax_cs);
            compile_shader(assets_dir / "shaders" / "directx" / "depth_minmax_ema_cs.hlsl", depth_minmax_ema_cs);
            if (use_percentile) {
                if (compile_shader(assets_dir / "shaders" / "directx" / "depth_hist_cs.hlsl", depth_hist_cs)) {
                    BOOST_LOG(info) << "Percentile depth normalization enabled (p" << cfg.norm_pct_lo
                                    << " .. p" << cfg.norm_pct_hi << ").";
                } else {
                    BOOST_LOG(warning) << "depth_hist_cs failed to compile; falling back to min/max normalization.";
                    use_percentile = false;
                }
            }
            if (sync_depth) {
                BOOST_LOG(info) << "Synchronous depth enabled: inference waits on the encode thread (no async ghost).";
            }
            if (subject_track) {
                bool ok = compile_shader(assets_dir / "shaders" / "directx" / "depth_subject_hist_cs.hlsl", depth_subject_hist_cs) &&
                          compile_shader(assets_dir / "shaders" / "directx" / "depth_subject_resolve_cs.hlsl", depth_subject_resolve_cs);
                if (ok) {
                    BOOST_LOG(info) << "Subject tracking enabled (recenter " << subject_recenter << ").";
                } else {
                    BOOST_LOG(warning) << "Subject-tracking shaders failed to compile; falling back to the linear depth mapping.";
                    subject_track = false;
                }
            }
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

                // Staging copy for the [NORMDBG] raw-stats trajectory (the norm-window study).
                D3D11_BUFFER_DESC stg = {};
                stg.Usage = D3D11_USAGE_STAGING;
                stg.ByteWidth = sizeof(init_raw);
                stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                device->CreateBuffer(&stg, nullptr, &minmax_raw_stage);
            }

            // EMA'd min/max, 2 float4 elements: [0]={min, max, initialized, ref_range},
            // [1]={range_scale, pad...}. initialized = 0 so the first frame seeds directly;
            // range_scale = 1 so the range floor is a no-op until depth_minmax_ema_cs runs.
            {
                float init_ema[8] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
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

            // Percentile histogram: 256 uint bins, zero-init (depth_minmax_ema_cs resets them
            // after each frame's scan, so the steady state matches this initial state).
            if (use_percentile) {
                uint32_t init_hist[256] = {};
                D3D11_BUFFER_DESC bd = {};
                bd.Usage = D3D11_USAGE_DEFAULT;
                bd.ByteWidth = sizeof(init_hist);
                bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
                bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                bd.StructureByteStride = sizeof(uint32_t);
                D3D11_SUBRESOURCE_DATA sd = {init_hist, 0, 0};
                device->CreateBuffer(&bd, &sd, &hist_buf);
                if (hist_buf) {
                    device->CreateUnorderedAccessView(hist_buf.Get(), nullptr, &hist_uav);
                }
                if (!hist_uav) {
                    BOOST_LOG(warning) << "Percentile histogram buffer creation failed; falling back to min/max normalization.";
                    use_percentile = false;
                }
            }

            // Subject tracking: weighted histogram (256 uint bins) + the per-frame subject
            // state (one float4, zero-init so `initialized` starts false and the reprojection
            // uses the linear path until the first resolve).
            if (subject_track) {
                uint32_t init_hist[256] = {};
                D3D11_BUFFER_DESC bd = {};
                bd.Usage = D3D11_USAGE_DEFAULT;
                bd.ByteWidth = sizeof(init_hist);
                bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
                bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                bd.StructureByteStride = sizeof(uint32_t);
                D3D11_SUBRESOURCE_DATA sd = {init_hist, 0, 0};
                device->CreateBuffer(&bd, &sd, &subject_hist_buf);
                if (subject_hist_buf) {
                    device->CreateUnorderedAccessView(subject_hist_buf.Get(), nullptr, &subject_hist_uav);
                }
                device->CreateBuffer(&bd, &sd, &subject_plain_buf);  // same 256-uint layout
                if (subject_plain_buf) {
                    device->CreateUnorderedAccessView(subject_plain_buf.Get(), nullptr, &subject_plain_uav);
                }

                float init_state[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};  // 2 float4
                bd.ByteWidth = sizeof(init_state);
                bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
                bd.StructureByteStride = sizeof(float) * 4;
                D3D11_SUBRESOURCE_DATA sd2 = {init_state, 0, 0};
                device->CreateBuffer(&bd, &sd2, &subject_buf);
                if (subject_buf) {
                    device->CreateUnorderedAccessView(subject_buf.Get(), nullptr, &subject_uav);
                    device->CreateShaderResourceView(subject_buf.Get(), nullptr, &subject_srv);
                    // Staging copy so the debug log can read the resolved subject state back
                    // (a GPU->CPU sync; only mapped when perf stats are on, every 24 updates).
                    D3D11_BUFFER_DESC stg = {};
                    stg.Usage = D3D11_USAGE_STAGING;
                    stg.ByteWidth = sizeof(init_state);
                    stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                    stg.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                    stg.StructureByteStride = sizeof(float) * 4;
                    device->CreateBuffer(&stg, nullptr, &subject_stage);
                }
                if (!subject_hist_uav || !subject_uav || !subject_srv || !subject_plain_uav) {
                    BOOST_LOG(warning) << "Subject-tracking buffer creation failed; falling back to the linear depth mapping.";
                    subject_track = false;
                }
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
            if (depth_context_pooled) return;  // modules already loaded; warmup is pure waste on a pooled context
            auto& cuda = cuda_driver_api::get();
            if (!cuda.is_valid()) return;
            if (cuda_ctx) cuda.cuCtxSetCurrent(cuda_ctx);

            int h, w;
            if (dynamic_width) {
                h = fixed_h;
                w = round_to_patch(fixed_h * 2.37f);
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

            nvinfer1::Dims in_dims = make_input_dims(input_rank, h, w);
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
                if (warp_stream) {
                    if (cuda.cuStreamSynchronize) cuda.cuStreamSynchronize(warp_stream);
                    cuda.cuStreamDestroy(warp_stream);
                }
                if (cuda_in_res) cuda.cuGraphicsUnregisterResource(cuda_in_res);
                if (cuda_out_res) cuda.cuGraphicsUnregisterResource(cuda_out_res);
                for (auto& res : cuda_warp_res) {
                    if (res) cuda.cuGraphicsUnregisterResource(res);
                }
                perf_destroy_events();  // free the timing events while cuda_ctx is still current
            }

            // Return the execution contexts to their engine's pool for reuse instead of leaking
            // (or destroying, which faults across the DLL boundary). The streams were
            // synchronized above, so no inference is still in flight referencing this
            // instance's tensor bindings, making the contexts safe for another instance to reuse.
            std::lock_guard<std::mutex> lock(g_trt_mutex);
            if (exec_context) {
                g_engines[model_name].context_pool.push_back(exec_context);
                exec_context = nullptr;
            }
            if (warp_context) {
                g_engines[warp_model].context_pool.push_back(warp_context);
                warp_context = nullptr;
            }
            // TRT runtime/engines are cached globally, do not destroy them here.
        }

        // The depth SRV handed back to the reprojection: the color-guided 2x upsample when
        // enabled, else the raw depth. Also the value returned on frames where we reuse the
        // last depth (stream busy or off-cadence).
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> output_srv() {
            return (guided_upsample && guided_depth_srv) ? guided_depth_srv : depth_srv;
        }

        estimate_result make_result() {
            estimate_result r;
            r.depth = output_srv();
            if (subject_track) {
                r.subject = subject_srv;
            }
            if (learned_warp && warp_fields_valid) {
                r.delta_left = delta_tex_srv[0];
                r.weight_left = weight_tex_srv[0];
                r.delta_right = delta_tex_srv[1];
                r.weight_right = weight_tex_srv[1];
                r.field_w = warp_w;
                r.field_h = warp_h;
                r.layers = warp_layers;
            }
            return r;
        }

        // Consume a finished MLBW inference: pack the raw output buffers into the per-eye
        // field textures. Called every frame; cheap no-op unless an inference just landed.
        void poll_warp_fields(cuda_driver_api& cuda) {
            if (!warp_pending || !cuda.cuStreamQuery) {
                return;
            }
            auto q = cuda.cuStreamQuery(warp_stream);
            if (q == CUDA_ERROR_NOT_READY) {
                return;
            }
            warp_pending = false;
            if (q != CUDA_SUCCESS) {
                if (!warp_error_logged) {
                    BOOST_LOG(error) << "MLBW warp stream failed: " << q;
                    warp_error_logged = true;
                }
                return;  // keep the last good fields (if any)
            }

            context->CSSetShader(mlbw_field_cs.Get(), nullptr, 0);
            for (int e = 0; e < 2; e++) {
                context->CSSetConstantBuffers(0, 1, mlbw_field_cbuffer[e].GetAddressOf());
                ID3D11ShaderResourceView* srvs[2] = {mlbw_delta_srv[e].Get(), mlbw_weight_srv[e].Get()};
                context->CSSetShaderResources(0, 2, srvs);
                ID3D11UnorderedAccessView* uavs[2] = {delta_tex_uav[e].Get(), weight_tex_uav[e].Get()};
                context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
                context->Dispatch((warp_w + 15) / 16, (warp_h + 15) / 16, 1);
            }
            ID3D11UnorderedAccessView* null_uavs[2] = {nullptr, nullptr};
            ID3D11ShaderResourceView* null_srvs[2] = {nullptr, nullptr};
            context->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
            context->CSSetShaderResources(0, 2, null_srvs);
            warp_fields_valid = true;
        }

        // Kick off MLBW for both eyes from the freshly-updated depth_tex. Runs on its own
        // CUDA stream (~1 ms per eye) so the result can be consumed as soon as it lands,
        // instead of waiting behind the much longer depth inference on cu_stream.
        void enqueue_warp(cuda_driver_api& cuda) {
            if (!learned_warp || !warp_context || warp_pending) {
                return;
            }

            // Lazy CUDA-D3D registration on the ENCODE thread (this thread owns the device's
            // immediate work; registering from the background build thread deadlocked in the
            // driver -- see the note in setup_learned_warp). One-time, ~ms.
            if (!warp_resources_registered) {
                ID3D11Resource* to_register[6] = {
                    mlbw_in_buf[0].Get(), mlbw_in_buf[1].Get(),
                    mlbw_delta_buf[0].Get(), mlbw_weight_buf[0].Get(),
                    mlbw_delta_buf[1].Get(), mlbw_weight_buf[1].Get()
                };
                for (int i = 0; i < 6; i++) {
                    if (cuda.cuGraphicsD3D11RegisterResource(&cuda_warp_res[i], to_register[i], 0) != CUDA_SUCCESS) {
                        BOOST_LOG(error) << "cuGraphicsD3D11RegisterResource failed for a warp tensor; learned warp disabled.";
                        learned_warp = false;  // permanent fallback to the probe reprojection
                        return;
                    }
                }
                warp_resources_registered = true;
            }

            // Build both eyes' input tensors (right = horizontally flipped depth). t1 carries
            // the subject state for the subject-anchored convergence plane (null when off).
            context->CSSetShader(mlbw_input_cs.Get(), nullptr, 0);
            context->CSSetConstantBuffers(0, 1, mlbw_input_cbuffer.GetAddressOf());
            ID3D11ShaderResourceView* in_srvs[2] = {depth_srv.Get(), subject_srv.Get()};
            context->CSSetShaderResources(0, 2, in_srvs);
            context->CSSetSamplers(0, 1, linear_sampler.GetAddressOf());
            ID3D11UnorderedAccessView* uavs[2] = {mlbw_in_uav[0].Get(), mlbw_in_uav[1].Get()};
            context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
            context->Dispatch((warp_w + 15) / 16, (warp_h + 15) / 16, 1);
            ID3D11UnorderedAccessView* null_uavs[2] = {nullptr, nullptr};
            ID3D11ShaderResourceView* null_srvs2[2] = {nullptr, nullptr};
            context->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
            context->CSSetShaderResources(0, 2, null_srvs2);

            if (cuda.cuGraphicsMapResources(6, cuda_warp_res, warp_stream) != CUDA_SUCCESS) {
                if (!warp_error_logged) {
                    BOOST_LOG(error) << "cuGraphicsMapResources failed for the MLBW warp tensors.";
                    warp_error_logged = true;
                }
                return;
            }
            void* p[6] = {};
            bool ok = true;
            for (int i = 0; i < 6; i++) {
                cuda.cuGraphicsResourceGetMappedPointer((CUdeviceptr*) &p[i], nullptr, cuda_warp_res[i]);
                ok = ok && p[i];
            }
            if (ok) {
                // One context, sequential enqueues (left then right); addresses are captured
                // at enqueue time, so rebinding between enqueues on the same stream is safe.
                std::lock_guard<std::mutex> lock(*trt_mutex);
                int perf_slot = perf_begin(perf_warp, warp_stream);  // times both eyes together
                for (int e = 0; e < 2 && ok; e++) {
                    warp_context->setTensorAddress("mlbw_input", p[e]);
                    warp_context->setTensorAddress("delta", p[2 + 2 * e]);
                    warp_context->setTensorAddress("layer_weight", p[3 + 2 * e]);
                    if (!warp_context->enqueueV3(warp_stream)) {
                        if (!warp_error_logged) {
                            BOOST_LOG(error) << "MLBW enqueueV3 failed; learned warp will stop updating.";
                            warp_error_logged = true;
                        }
                        ok = false;
                    }
                }
                perf_end(perf_warp, perf_slot, warp_stream);
            }
            cuda.cuGraphicsUnmapResources(6, cuda_warp_res, warp_stream);
            warp_pending = ok;
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
            cb_desc.ByteWidth = 80;  // shared depth-pass cbuffer (20 floats/uints; see below)
            cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

            // Shared depth-pass constants: {target_w, target_h, is_hdr, ema_alpha, minmax_alpha,
            // reduce_threads, output_transform, depth_shift, snap_ratio, floor_frac,
            // floor_ref_alpha, pct_lo, pct_hi, pads} (see buffer_to_tex_cs.hlsl /
            // depth_minmax_ema_cs.hlsl; shaders declaring only the 12-slot prefix still bind fine).
            uint32_t cb[20] = {};
            float* cbf = (float*)cb;
            cb[0] = (uint32_t)target_w;
            cb[1] = (uint32_t)target_h;
            cb[2] = is_hdr ? 1u : 0u;
            cbf[3] = ema_alpha;
            cbf[4] = minmax_alpha;
            cb[5] = reduce_groups * 256u;  // total threads for the reduction grid-stride
            cb[6] = output_transform;  // 0=identity (DA-V2 disparity), 1=shifted reciprocal (DA-V3 depth)
            cbf[7] = depth_shift;  // shift in 1/(depth + depth_shift) when output_transform==1
            cbf[8] = minmax_snap;       // A1 scene-cut snap ratio (0 = off)
            cbf[9] = range_floor_frac;  // A3 range-floor fraction (0 = off)
            cbf[10] = range_floor_ref_alpha;  // A3 reference-range decay
            cbf[11] = use_percentile ? pct_lo : 0.0f;  // robust normalization, low bound (fraction)
            cbf[12] = use_percentile ? pct_hi : 1.0f;  // robust normalization, high bound (fraction)
            cbf[13] = norm_lock_frames;  // scene lock: updates before the bounds freeze (0 = off)
            cbf[14] = 0.005f;  // locked drift rate: ~4-7 s time constant, imperceptible per frame,
                               // but a long scene that slowly changes depth range can't diverge
            cbf[15] = subject_recenter;  // subject recenter strength (depth_subject_resolve_cs)
            cbf[16] = stretch_lo;                          // shape_depth_for_pop stretch bounds
            cbf[17] = stretch_hi;
            cbf[18] = subject_stretch ? 1.0f : 0.0f;
            cbf[19] = 0.0f;
            D3D11_SUBRESOURCE_DATA sd = {cb, 0, 0};
            cbuffer.Reset();
            device->CreateBuffer(&cb_desc, &sd, &cbuffer);

            cb_desc.ByteWidth = 32;  // guided cbuffer keeps the original 8-slot layout
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

        // Normalize the finished raw disparity in tensor_out_buf into depth_tex: the scale
        // passes (min/max reduction, optional percentile histogram, EMA fold) followed by the
        // mapping/temporal-EMA pass. GPU-resident throughout, no CPU readback. In async mode
        // this consumes the PREVIOUS frame's inference; in sync mode, the current frame's.
        void normalize_depth_output() {
            // 3a. Per-frame scale (GPU-resident; no CPU readback). Depth Anything V2's
            // relative output is affine-invariant, so this is required for a stable parallax scale.
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

                // Pass A2 (percentile mode): 256-bin histogram over the raw range, so pass B
                // can replace the outlier-sensitive min/max with robust percentile bounds.
                if (use_percentile && depth_hist_cs && hist_uav) {
                    context->CSSetShader(depth_hist_cs.Get(), nullptr, 0);
                    context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
                    context->CSSetShaderResources(0, 1, tensor_out_srv.GetAddressOf());
                    ID3D11UnorderedAccessView* hist_uavs[2] = {hist_uav.Get(), minmax_raw_uav.Get()};
                    context->CSSetUnorderedAccessViews(0, 2, hist_uavs, nullptr);
                    context->Dispatch(reduce_groups, 1, 1);

                    ID3D11UnorderedAccessView* null_uavs_h[2] = {nullptr, nullptr};
                    context->CSSetUnorderedAccessViews(0, 2, null_uavs_h, nullptr);
                    context->CSSetShaderResources(0, 1, &null_srv1);
                }

                // [NORMDBG] per-frame raw min/max trajectory for normalization studies. Opt-in
                // via APOLLO_NORMDBG (NOT perf-gated: the per-frame Map is a CPU sync that would
                // perturb perf wall-time). 2026-07-09 study: a centered/look-ahead window is
                // WORSE than the causal EMA (short window tracks range noise); a slower EMA
                // (minmax_ema 0.1->0.03) halves range swim for free, but it's already ~0.2
                // levels/frame = imperceptible. Kept as a tool for future normalization work.
                static const bool normdbg = std::getenv("APOLLO_NORMDBG") != nullptr;
                if (normdbg && minmax_raw_stage) {
                    context->CopyResource(minmax_raw_stage.Get(), minmax_raw_buf.Get());
                    D3D11_MAPPED_SUBRESOURCE ms {};
                    if (SUCCEEDED(context->Map(minmax_raw_stage.Get(), 0, D3D11_MAP_READ, 0, &ms))) {
                        const uint32_t* u = (const uint32_t*) ms.pData;
                        float rmin, rmax;
                        std::memcpy(&rmin, &u[0], 4);
                        std::memcpy(&rmax, &u[1], 4);
                        BOOST_LOG(info) << "[NORMDBG] f=" << norm_log_counter++
                                        << " raw_min=" << rmin << " raw_max=" << rmax;
                        context->Unmap(minmax_raw_stage.Get(), 0);
                    }
                }

                // Pass B: fold into the EMA'd bounds and reset the accumulators (1 thread).
                context->CSSetShader(depth_minmax_ema_cs.Get(), nullptr, 0);
                ID3D11UnorderedAccessView* ema_uavs[3] = {minmax_ema_uav.Get(), minmax_raw_uav.Get(), hist_uav.Get()};
                context->CSSetUnorderedAccessViews(0, 3, ema_uavs, nullptr);
                context->Dispatch(1, 1, 1);

                ID3D11UnorderedAccessView* null_uav2[3] = {nullptr, nullptr, nullptr};
                context->CSSetUnorderedAccessViews(0, 3, null_uav2, nullptr);
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

            // 3s. Subject tracking: weighted depth histogram over the freshly-normalized
            // depth, then a 1-thread resolve into the subject state the reprojection reads.
            if (subject_track && depth_subject_hist_cs && depth_subject_resolve_cs) {
                context->CSSetShader(depth_subject_hist_cs.Get(), nullptr, 0);
                context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
                context->CSSetShaderResources(0, 1, depth_srv.GetAddressOf());
                ID3D11UnorderedAccessView* hist_uavs[2] = {subject_hist_uav.Get(), subject_plain_uav.Get()};
                context->CSSetUnorderedAccessViews(0, 2, hist_uavs, nullptr);
                context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

                ID3D11UnorderedAccessView* null_uavs_h2[2] = {nullptr, nullptr};
                context->CSSetUnorderedAccessViews(0, 2, null_uavs_h2, nullptr);
                context->CSSetShaderResources(0, 1, null_srvs);

                context->CSSetShader(depth_subject_resolve_cs.Get(), nullptr, 0);
                ID3D11UnorderedAccessView* subj_uavs[3] = {subject_hist_uav.Get(), subject_uav.Get(), subject_plain_uav.Get()};
                context->CSSetUnorderedAccessViews(0, 3, subj_uavs, nullptr);
                context->Dispatch(1, 1, 1);

                ID3D11UnorderedAccessView* null_uavs2[3] = {nullptr, nullptr, nullptr};
                context->CSSetUnorderedAccessViews(0, 3, null_uavs2, nullptr);

                // Ground-truth log to number-match against VD3D's [3DDBG] subj=. VD3D's render
                // depth is LOW=near; Apollo is HIGH=near, so print both subj and 1-subj (the
                // VD3D-convention value to compare directly). Perf-gated + every 24 updates so
                // the GPU->CPU readback never touches the shipping path.
                if (sbs_perf::enabled() && subject_stage && (++subject_log_counter % 24u) == 1u) {
                    context->CopyResource(subject_stage.Get(), subject_buf.Get());
                    D3D11_MAPPED_SUBRESOURCE ms {};
                    if (SUCCEEDED(context->Map(subject_stage.Get(), 0, D3D11_MAP_READ, 0, &ms))) {
                        const float* s = (const float*) ms.pData;  // {delta, scurve, subj_ema, init}
                        BOOST_LOG(info) << "[SUBJDBG] u=" << subject_log_counter
                                        << " subj_hi_near=" << s[2]
                                        << " subj_vd3d=" << (1.0f - s[2])
                                        << " recenter_delta=" << s[0]
                                        << " subject_curve=" << s[1]
                                        << " init=" << s[3];
                        context->Unmap(subject_stage.Get(), 0);
                    }
                }
            }
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
                // ±0.65 hysteresis (not ±0.5): with a symmetric half-frame band, a video rate
                // sitting exactly on a boundary (e.g. ~67 fps / depth_fps 45 -> ideal ~1.49)
                // flips the interval every few seconds, oscillating the depth rate and GPU
                // load. The wider band holds the current interval anywhere in (i-0.65, i+0.65).
                float ideal = measured_fps / depth_fps;
                if (ideal > effective_interval + 0.65f) {
                    effective_interval += 1;
                } else if (effective_interval > 1 && ideal < effective_interval - 0.65f) {
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

        estimate_result estimate(ID3D11ShaderResourceView* input_srv, bool is_hdr) {
            if (!exec_context || !rgb_to_nchw_cs || !buffer_to_tex_cs || !input_srv) return {};

            auto& cuda = cuda_driver_api::get();
            if (!cuda.is_valid()) {
                BOOST_LOG(error) << "CUDA Driver API is not available.";
                return {};
            }

            if (cuda_ctx) {
                cuda.cuCtxSetCurrent(cuda_ctx);
            }

            // Measure video fps and derive the depth-inference interval (called every frame).
            update_cadence();

            // Consume a finished MLBW warp inference (if any) into the field textures. Done
            // every frame so the ~1 ms result lands one video frame after its depth update.
            if (learned_warp) {
                poll_warp_fields(cuda);
            }

            // Perf benchmark: resolve any completed inference-timing events into samples.
            if (sbs_perf::enabled()) {
                perf_drain(perf_depth);
                perf_drain(perf_warp);
            }

            // Prevent GPU starvation: if the previous AI frame is still crunching, drop this frame.
            // This prevents an infinite queue of heavy TensorRT workloads from starving the DWM and Edge Browser.
            // Async only: sync mode drains its own stream every frame, so it can never be busy here.
            if (!sync_depth && cu_stream && cuda.cuStreamQuery) {
                auto q = cuda.cuStreamQuery(cu_stream);
                if (q == CUDA_ERROR_NOT_READY) {
                    // Still re-snap the stale depth to THIS frame's color edges (D3D-only, cheap).
                    run_guided(input_srv, is_hdr);
                    return make_result();
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
                    return {};
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
                    target_w = round_to_patch(fixed_h * a);
                } else {
                    // Short-side budget (iw3-style): pin the short side so vertical depth detail
                    // is constant regardless of aspect ratio; the long side grows with aspect,
                    // capped by max_aspect to bound cost.
                    int short_side = round_to_patch((float)depth_short_side);
                    if (aspect_ratio >= 1.0f) {
                        target_h = short_side;
                        target_w = (int)std::round(short_side * std::min(aspect_ratio, max_aspect));
                    } else {
                        target_w = short_side;
                        target_h = (int)std::round(short_side * std::min(1.0f / aspect_ratio, max_aspect));
                    }

                    target_h = round_to_patch((float)target_h);
                    target_w = round_to_patch((float)target_w);

                    // Cap the model input aspect-preserving against two limits: the TensorRT
                    // engine's MAX profile (1008), and the frame's native resolution -- never
                    // upscale a small input up to the budget (matches iw3's limit_resolution).
                    // Scaling both axes by a single factor keeps the depth undistorted.
                    int max_w = std::min(1008, (int)input_desc.Width);
                    int max_h = std::min(1008, (int)input_desc.Height);
                    if (target_w > max_w || target_h > max_h) {
                        float s = std::min((float)max_w / (float)target_w, (float)max_h / (float)target_h);
                        target_w = round_to_patch((float)target_w * s);
                        target_h = round_to_patch((float)target_h * s);
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
                return make_result();
            }

            // Shared constants for buffer_to_tex_cs, the min/max passes and rgb_to_nchw_cs.
            // Session-constant, so the buffer is built once (immutable), not mapped per frame.
            ensure_cbuffers(is_hdr);
            if (!cbuffer) {
                return {};
            }

            if (!sync_depth) {
                // ASYNC: tensor_out_buf holds the finished raw disparity from the PREVIOUS
                // inference (fully unmapped from CUDA), so consuming it here never blocks the
                // encode thread -- at the cost of depth lagging color by the inference cadence.
                if (has_previous_frame) {
                    normalize_depth_output();

                    // 3c. Learned warp: turn the freshly-updated depth into per-eye warp fields
                    // (both eyes, ~1 ms each on the warp stream; consumed by poll_warp_fields).
                    enqueue_warp(cuda);
                }

                // 3d. Guided upsample: snap the freshly-updated depth to this frame's color edges.
                run_guided(input_srv, is_hdr);
            }

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
                    nvinfer1::Dims in_dims = make_input_dims(input_rank, target_h, target_w);
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
                    int perf_slot = perf_begin(perf_depth, cu_stream);
                    if (!exec_context->enqueueV3(cu_stream) && !stream_error_logged) {
                        BOOST_LOG(error) << "TensorRT enqueueV3 failed; depth will stop updating.";
                        stream_error_logged = true;
                    }
                    perf_end(perf_depth, perf_slot, cu_stream);
                }
            }
            
            cuda.cuGraphicsUnmapResources(2, resources, cu_stream);

            if (sync_depth) {
                // SYNC: wait for THIS frame's inference (~2-3 ms for DA-V2 small; measured via
                // the perf events above) so the depth warped this frame is the depth OF this
                // frame -- color and depth always match, eliminating the async motion ghost.
                // The unmap above is stream-ordered, so after the wait the D3D passes read a
                // complete, coherent tensor_out_buf.
                if (cuda.cuStreamSynchronize) {
                    cuda.cuStreamSynchronize(cu_stream);
                }
                normalize_depth_output();
                if (learned_warp) {
                    // The warp fields must also describe THIS frame: enqueue and wait (~1 ms
                    // both eyes), then pack the fields immediately instead of next frame.
                    enqueue_warp(cuda);
                    if (cuda.cuStreamSynchronize) {
                        cuda.cuStreamSynchronize(warp_stream);
                    }
                    poll_warp_fields(cuda);
                }
                run_guided(input_srv, is_hdr);
            }

            has_previous_frame = true;

            return make_result();
        }
    };

    video_depth_estimator::video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device,
                          Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
                          const std::filesystem::path& assets_dir,
                          const config::video_t::sbs_t& cfg,
                          const config::depth_model_info& model)
        : pimpl(std::make_unique<impl>(device, context, assets_dir, cfg, model)) {}

    video_depth_estimator::~video_depth_estimator() = default;

    estimate_result video_depth_estimator::estimate_depth(ID3D11ShaderResourceView* input_srv, bool is_hdr) {
        return pimpl->estimate(input_srv, is_hdr);
    }
}
