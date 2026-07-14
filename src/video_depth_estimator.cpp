#include "video_depth_estimator.h"

#include "cuda_driver_api.h"
#include "logging.h"
#include "model_manager.h"
#include "platform/windows/misc.h"
#include "platform/windows/utils.h"
#include "sbs_perf.h"
#include "utility.h"

#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <d3dcompiler.h>
#include <fstream>
#include <map>
#include <mutex>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <regex>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

#pragma comment(lib, "d3dcompiler.lib")

using namespace std::literals;

class Logger: public nvinfer1::ILogger {
public:
#ifdef __GNUC__
  void msvc_dummy_destructor(char flags) noexcept override {}
#endif
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      BOOST_LOG(warning) << "TensorRT: " << msg;
    } else {
      BOOST_LOG(info) << "TensorRT: " << msg;
    }
  }
};

static Logger gLogger;

static std::mutex g_engine_build_status_mutex;
static std::map<std::string, models::engine_build_status> g_engine_build_status;
static std::mutex g_model_prepare_status_mutex;
static std::map<std::string, models::engine_build_status> g_model_prepare_status;
static std::mutex g_depth_shader_cache_mutex;

struct depth_shader_cache_entry {
  std::filesystem::file_time_type modified;
  std::vector<std::uint8_t> bytecode;
};

static std::map<std::filesystem::path, depth_shader_cache_entry> g_depth_shader_cache;

static void set_engine_build_status(const std::string &engine_name, models::engine_build_status status) {
  std::lock_guard<std::mutex> lock(g_engine_build_status_mutex);
  g_engine_build_status[engine_name] = status;
}

static void set_model_prepare_status(const std::string &engine_name, models::engine_build_status status) {
  std::lock_guard<std::mutex> lock(g_model_prepare_status_mutex);
  g_model_prepare_status[engine_name] = status;
}

static bool depth_shader_bytecode(
  const std::filesystem::path &path,
  std::vector<std::uint8_t> &bytecode
) {
  // D3D shader bytecode is device-independent. Cache blobs across device recreation, but compare
  // source mtimes so a newly created estimator sees an edit without restarting. Never hold the
  // global map lock across D3DCompileFromFile: unrelated estimators may initialize concurrently.
  std::error_code ec;
  const auto modified = std::filesystem::last_write_time(path, ec);
  {
    std::lock_guard<std::mutex> lock(g_depth_shader_cache_mutex);
    if (auto it = g_depth_shader_cache.find(path); it != g_depth_shader_cache.end() && !ec && it->second.modified == modified) {
      bytecode = it->second.bytecode;
      return true;
    }
  }

  Microsoft::WRL::ComPtr<ID3DBlob> blob;
  Microsoft::WRL::ComPtr<ID3DBlob> err;
  constexpr DWORD flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
  if (FAILED(D3DCompileFromFile(path.wstring().c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "cs_5_0", flags, 0, &blob, &err))) {
    if (err) {
      BOOST_LOG(error) << "Shader compile error (" << path << "): " << (char *) err->GetBufferPointer();
    }
    return false;
  }
  auto *begin = static_cast<const std::uint8_t *>(blob->GetBufferPointer());
  bytecode.assign(begin, begin + blob->GetBufferSize());
  if (!ec) {
    std::lock_guard<std::mutex> lock(g_depth_shader_cache_mutex);
    g_depth_shader_cache.insert_or_assign(path, depth_shader_cache_entry {modified, bytecode});
  }
  return true;
}

// Shared TensorRT state. The runtime and engine are created once and shared by every
// encoder instance. Execution contexts are pooled and reused: creating one allocates
// ~1.3 GB of device scratch and takes several seconds, and it cannot be safely deleted
// across the MinGW/MSVC ABI boundary (see AGENTS.md rule #4). Creating a fresh context
// on every encoder recreation (which happens frequently during video playback via MPO
// flips / HDR / resolution changes) therefore leaked ~1.3 GB each time until the GPU ran
// out of memory and the device was removed. Pooling caps live contexts at peak concurrency.
static std::mutex g_trt_mutex;
static std::condition_variable g_trt_context_available;
// One active stream normally needs one context; four permits multi-client/transition overlap
// without allowing rapid asynchronous rebuilds to consume VRAM without bound.
static constexpr std::size_t kMaxContextsPerEngine = 4;
static nvinfer1::IRuntime *g_runtime = nullptr;
static std::once_flag g_cuda_init_once;
static CUresult g_cuda_init_result = CUDA_ERROR_NOT_READY;
static std::mutex g_cuda_context_mutex;
static std::map<CUdevice, CUcontext> g_cuda_primary_contexts;

static bool ensure_cuda_initialized(cuda_driver_api &cuda) {
  std::call_once(g_cuda_init_once, [&cuda]() {
    g_cuda_init_result = cuda.cuInit(0);
  });
  return g_cuda_init_result == CUDA_SUCCESS;
}

// Retain each primary context once for the process lifetime. TensorRT engines/contexts are also
// process-resident, so releasing it from an estimator destructor would invalidate pooled state.
static CUcontext primary_context(cuda_driver_api &cuda, CUdevice device) {
  std::lock_guard<std::mutex> lock(g_cuda_context_mutex);
  auto found = g_cuda_primary_contexts.find(device);
  if (found != g_cuda_primary_contexts.end()) {
    return found->second;
  }
  CUcontext context = nullptr;
  if (cuda.cuDevicePrimaryCtxRetain && cuda.cuDevicePrimaryCtxRetain(&context, device) == CUDA_SUCCESS && context) {
    g_cuda_primary_contexts.emplace(device, context);
  }
  return context;
}

static bool cuda_device_for_d3d(cuda_driver_api &cuda, ID3D11Device *d3d, CUdevice &out) {
  if (cuda.cuD3D11GetDevice && d3d) {
    IDXGIDevice *dxgi_device = nullptr;
    IDXGIAdapter *adapter = nullptr;
    if (SUCCEEDED(d3d->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void **>(&dxgi_device))) && SUCCEEDED(dxgi_device->GetAdapter(&adapter))) {
      const CUresult result = cuda.cuD3D11GetDevice(&out, adapter);
      adapter->Release();
      dxgi_device->Release();
      if (result == CUDA_SUCCESS) {
        return true;
      }
    } else if (dxgi_device) {
      dxgi_device->Release();
    }
  }
  BOOST_LOG(warning) << "Could not map the D3D11 adapter to CUDA; falling back to CUDA device 0.";
  return cuda.cuDeviceGet && cuda.cuDeviceGet(&out, 0) == CUDA_SUCCESS;
}

// Resolve the same explicitly configured DXGI adapter that the capture pipeline will use. When
// adapter_name is empty, CUDA device 0 remains Apollo's default. This keeps startup preparation
// from allocating a large TensorRT context on the wrong NVIDIA GPU in multi-adapter systems.
static bool cuda_device_for_configured_adapter(
  cuda_driver_api &cuda,
  const std::string &adapter_name,
  CUdevice &out
) {
  if (adapter_name.empty()) {
    return cuda.cuDeviceGet && cuda.cuDeviceGet(&out, 0) == CUDA_SUCCESS;
  }
  if (!cuda.cuD3D11GetDevice) {
    BOOST_LOG(error) << "Startup depth-model preparation cannot map configured adapter '"
                     << adapter_name << "' because CUDA/D3D11 interop is unavailable.";
    return false;
  }

  Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
    BOOST_LOG(error) << "Startup depth-model preparation failed to create a DXGI factory.";
    return false;
  }

  const auto wanted = platf::from_utf8(adapter_name);
  for (UINT index = 0;; ++index) {
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    const HRESULT enumerated = factory->EnumAdapters1(index, &adapter);
    if (enumerated == DXGI_ERROR_NOT_FOUND) {
      break;
    }
    if (FAILED(enumerated)) {
      BOOST_LOG(error) << "Startup depth-model preparation failed while enumerating DXGI adapters.";
      return false;
    }
    DXGI_ADAPTER_DESC1 desc {};
    if (FAILED(adapter->GetDesc1(&desc)) || wanted != desc.Description) {
      continue;
    }
    const CUresult mapped = cuda.cuD3D11GetDevice(&out, adapter.Get());
    if (mapped == CUDA_SUCCESS) {
      BOOST_LOG(info) << "Startup depth model mapped configured adapter '" << adapter_name
                      << "' to CUDA device " << out << '.';
      return true;
    }
    BOOST_LOG(error) << "Configured adapter '" << adapter_name
                     << "' is not available to CUDA/TensorRT.";
    return false;
  }

  BOOST_LOG(error) << "Configured adapter '" << adapter_name
                   << "' was not found during startup depth-model preparation.";
  return false;
}

// One resident engine per CUDA-device/model pair, so multi-adapter sessions never reuse a
// TensorRT engine or execution context deserialized under another CUDA primary context. Distinct
// startup model configurations remain isolated instead of being pinned to the first model.
// Engines are never evicted: an
// IExecutionContext holds ~1.3 GB scratch and cannot be safely destroyed across the MinGW/MSVC
// ABI boundary, so contexts are pooled per engine and reused (see the ctor/dtor). With
// sequential evaluator model testing this can leave 2-3 engines resident, which is acceptable.
struct engine_slot {
  nvinfer1::ICudaEngine *engine = nullptr;
  std::vector<nvinfer1::IExecutionContext *> context_pool;
  std::size_t context_count = 0;
  bool io_validated = false;
  bool io_compatible = false;
};

static std::map<std::string, engine_slot> g_engines;  // guarded by g_trt_mutex

template<typename T>
struct TrtDeleter {
  void operator()(T *ptr) const {
    if (ptr) {
#ifdef __GNUC__
      ptr->msvc_dummy_destructor(1);
#else
      delete ptr;
#endif
    }
  }
};

template<typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

// Build the DA-V2 input dimensions [1,3,H,W]. Passing Dims INTO TensorRT is ABI-safe (only
// RETURNING a Dims by value across the MinGW/MSVC boundary faults).
static nvinfer1::Dims make_input_dims(int h, int w) {
  nvinfer1::Dims d {};
  d.nbDims = 4;
  d.d[0] = 1;
  d.d[1] = 3;
  d.d[2] = h;
  d.d[3] = w;
  return d;
}

// Round x to the nearest positive multiple of `patch` (the model's spatial patch size; 14 for the
// Depth Anything family). Model input dims must be patch-aligned or TensorRT rejects the shape.
static int round_to_patch(float x, int patch = 14) {
  return std::max(patch, (int) std::round(x / patch) * patch);
}

// Pick the largest patch-aligned short side that fits both native/profile limits, deriving the
// long side from the source aspect instead of rounding/capping both axes independently. Independent
// rounding turned 5120x2160 into 1008x420 (2.400:1); this returns 994x420 (2.367:1), keeping the
// model grid much closer to the 2.370:1 source without wasting a meaningful amount of inference.
static std::pair<int, int> aspect_aligned_dims(float aspect, int short_side, int max_w, int max_h, int patch = 14) {
  aspect = std::max(aspect, 1e-6f);
  max_w = std::max(patch, (max_w / patch) * patch);
  max_h = std::max(patch, (max_h / patch) * patch);
  const int requested_short = round_to_patch((float) short_side, patch);
  if (aspect >= 1.0f) {
    for (int h = std::min(requested_short, max_h); h >= patch; h -= patch) {
      const int w = round_to_patch((float) h * aspect, patch);
      if (w <= max_w) {
        return {w, h};
      }
    }
  } else {
    for (int w = std::min(requested_short, max_w); w >= patch; w -= patch) {
      const int h = round_to_patch((float) w / aspect, patch);
      if (h <= max_h) {
        return {w, h};
      }
    }
  }
  return {patch, patch};
}

// Ensure the shared runtime exists, deserialize `model_name`'s engine into its global slot if not
// already resident, and hand back a spare pooled execution context if one is available. The CALLER
// must hold g_trt_mutex. Context CREATION is deliberately left to the caller OUTSIDE the lock:
// createExecutionContext() allocates ~1.3 GB of scratch and takes seconds, and holding the lock
// across it would block concurrent teardowns' pool returns and other sessions' enqueues.
static nvinfer1::ICudaEngine *acquire_engine_locked(
  const std::string &engine_key,
  const std::filesystem::path &engine_path,
  nvinfer1::IExecutionContext *&out_context,
  bool &out_pooled
) {
  out_context = nullptr;
  out_pooled = false;
  auto &slot = g_engines[engine_key];
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

static const char *tensor_dtype_name(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return "FP32";
    case nvinfer1::DataType::kHALF:
      return "FP16";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    default:
      return "other";
  }
}

// Validate once per resident engine against Apollo's fixed D3D/CUDA tensor contract.
// Caller holds g_trt_mutex.
static bool validate_engine_io_locked(nvinfer1::ICudaEngine *engine, engine_slot &slot) {
  if (!engine) {
    return false;
  }
  if (slot.io_validated) {
    return slot.io_compatible;
  }

  slot.io_validated = true;
  bool have_in = false;
  bool have_out = false;
  bool input_fp32 = false;
  bool output_fp32 = false;
  for (int i = 0; i < engine->getNbIOTensors(); i++) {
    const char *name = engine->getIOTensorName(i);
    const auto type = engine->getTensorDataType(name);
    const bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    BOOST_LOG(info) << "Depth engine tensor '" << name << "' " << (is_input ? "(input)" : "(output)")
                    << " dtype=" << tensor_dtype_name(type);
    if (std::string_view(name) == "pixel_values") {
      have_in = true;
      input_fp32 = type == nvinfer1::DataType::kFLOAT;
      if (!input_fp32) {
        BOOST_LOG(error) << "Depth model input 'pixel_values' is " << tensor_dtype_name(type)
                         << ", not FP32; rejecting the engine. Use a keep_io_types (FP32 I/O) model.";
      }
    } else if (std::string_view(name) == "predicted_depth") {
      have_out = true;
      output_fp32 = type == nvinfer1::DataType::kFLOAT;
      if (!output_fp32) {
        BOOST_LOG(error) << "Depth model output 'predicted_depth' is " << tensor_dtype_name(type)
                         << ", not FP32; rejecting the engine.";
      }
    }
  }
  if (!have_in || !have_out) {
    BOOST_LOG(error) << "Depth model is missing the expected tensor name(s) 'pixel_values'/'predicted_depth'; "
                        "rejecting the engine.";
  }
  slot.io_compatible = have_in && have_out && input_fp32 && output_fp32;
  return slot.io_compatible;
}

static bool warmup_execution_context(
  cuda_driver_api &cuda,
  CUcontext cuda_ctx,
  nvinfer1::IExecutionContext *exec_context
) {
  if (!exec_context || !cuda.is_valid()) {
    return false;
  }
  if (cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
    return false;
  }

  CUstream stream = nullptr;
  if (cuda.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS || !stream) {
    return false;
  }
  auto destroy_stream = util::fail_guard([&]() {
    cuda.cuStreamDestroy(stream);
  });

  constexpr int h = models::depth_engine_opt_height;
  constexpr int w = models::depth_engine_opt_width;
  const size_t in_elems = (size_t) 3 * h * w;
  const size_t out_elems = (size_t) h * w;
  CUdeviceptr d_in = 0;
  CUdeviceptr d_out = 0;
  if (cuda.cuMemAlloc(&d_in, in_elems * sizeof(float)) != CUDA_SUCCESS) {
    return false;
  }
  auto free_input = util::fail_guard([&]() {
    cuda.cuMemFree(d_in);
  });
  if (cuda.cuMemAlloc(&d_out, out_elems * sizeof(float)) != CUDA_SUCCESS) {
    return false;
  }
  auto free_output = util::fail_guard([&]() {
    cuda.cuMemFree(d_out);
  });

  const auto input_dims = make_input_dims(h, w);
  const bool bound = exec_context->setInputShape("pixel_values", input_dims) &&
                     exec_context->setTensorAddress("pixel_values", (void *) d_in) &&
                     exec_context->setTensorAddress("predicted_depth", (void *) d_out);
  bool enqueued = false;
  if (bound) {
    std::lock_guard<std::mutex> lock(g_trt_mutex);
    enqueued = exec_context->enqueueV3(stream);
  }
  const bool synchronized = enqueued && cuda.cuStreamSynchronize &&
                            cuda.cuStreamSynchronize(stream) == CUDA_SUCCESS;
  BOOST_LOG(info) << "Depth model startup warmup complete (" << w << 'x' << h
                  << (synchronized ? ")." : "); execution failed.");
  return synchronized;
}

namespace models {

  engine_build_status tensorrt_engine_build_status(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model
  ) {
    const auto engine_name = engine_filename(model);
    std::error_code ec;
    if (std::filesystem::is_regular_file(assets_dir / engine_name, ec)) {
      return engine_build_status::ready;
    }
    std::lock_guard<std::mutex> lock(g_engine_build_status_mutex);
    auto it = g_engine_build_status.find(engine_name);
    return it == g_engine_build_status.end() ? engine_build_status::unknown : it->second;
  }

  void precompile_tensorrt_engine(const std::filesystem::path &assets_dir, const config::depth_model_info &model) {
    const std::string engine_name = engine_filename(model);
    set_engine_build_status(engine_name, engine_build_status::building);
    auto failed = util::fail_guard([&]() {
      set_engine_build_status(engine_name, engine_build_status::failed);
    });
    static std::mutex compile_mutex;
    std::lock_guard<std::mutex> lock(compile_mutex);

    const std::string &model_name = model.name;
    const std::string &model_url = model.url;
    auto model_path = ensure_model_available(assets_dir, model_name, model_url, engine_name);
    if (model_path.empty()) {
      BOOST_LOG(warning) << "Model not found. Background precompilation aborted.";
      set_engine_build_status(engine_name, engine_build_status::failed);
      return;
    }
    if (model_path.extension() == ".engine") {
      BOOST_LOG(info) << "TensorRT engine already compiled and ready.";
      set_engine_build_status(engine_name, engine_build_status::ready);
      failed.disable();
      return;
    }

    BOOST_LOG(info) << "Building TensorRT engine from ONNX... This will take a few minutes.";

    auto &cuda = cuda_driver_api::get();
    if (cuda.is_valid() && ensure_cuda_initialized(cuda)) {
      CUdevice cu_dev;
      if (cuda.cuDeviceGet(&cu_dev, 0) == 0) {
        if (CUcontext ctx = primary_context(cuda, cu_dev)) {
          cuda.cuCtxSetCurrent(ctx);
        }
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
      set_engine_build_status(engine_name, engine_build_status::failed);
      return;
    }
    if (network->getNbInputs() != 1 || std::string_view(network->getInput(0)->getName()) != "pixel_values") {
      BOOST_LOG(error) << "Unsupported depth model input contract; expected one 'pixel_values' tensor.";
      set_engine_build_status(engine_name, engine_build_status::failed);
      return;
    }

    // DA-V2 contract: input "pixel_values" [1,3,H,W], output "predicted_depth".
    auto profile = builder->createOptimizationProfile();
    if (network->getNbInputs() > 0) {
      auto input = network->getInput(0);
      auto dims_for = [&](int h, int w) {
        return make_input_dims(h, w);
      };
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims_for(14, 14));
      profile->setDimensions(
        input->getName(),
        nvinfer1::OptProfileSelector::kOPT,
        dims_for(depth_engine_opt_height, depth_engine_opt_width)
      );
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims_for(1008, 1008));
      config->addOptimizationProfile(profile);
    }

    std::vector<nvinfer1::ITensor *> to_unmark;
    bool found_depth_output = false;
    for (int i = 0; i < network->getNbOutputs(); i++) {
      auto *tensor = network->getOutput(i);
      if (std::string_view(tensor->getName()) == "predicted_depth") {
        found_depth_output = true;
      } else {
        to_unmark.push_back(tensor);
      }
    }
    if (!found_depth_output) {
      BOOST_LOG(error) << "Unsupported depth model output contract; missing 'predicted_depth'.";
      set_engine_build_status(engine_name, engine_build_status::failed);
      return;
    }
    for (auto *tensor : to_unmark) {
      BOOST_LOG(info) << "Depth engine: pruning unsupported output '" << tensor->getName() << "'.";
      network->unmarkOutput(*tensor);
    }

    auto serializedModel = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (serializedModel) {
      // Save under the recipe-specific engine name so a later recipe change rebuilds
      // rather than silently reusing this engine's (now-wrong) I/O layout.
      auto engine_path = assets_dir / engine_name;
      std::ofstream p(engine_path, std::ios::binary);
      if (p) {
        p.write(static_cast<const char *>(serializedModel->data()), serializedModel->size());
        p.close();
        if (p) {
          BOOST_LOG(info) << "Saved built engine to " << engine_path;
          set_engine_build_status(engine_name, engine_build_status::ready);
          failed.disable();
          return;
        }
      }
      BOOST_LOG(error) << "Failed to save built engine to " << engine_path;
    } else {
      BOOST_LOG(error) << "Engine build failed.";
    }
    set_engine_build_status(engine_name, engine_build_status::failed);
  }

  engine_build_status tensorrt_model_prepare_status(const config::depth_model_info &model) {
    const auto engine_name = engine_filename(model);
    std::lock_guard<std::mutex> lock(g_model_prepare_status_mutex);
    auto it = g_model_prepare_status.find(engine_name);
    return it == g_model_prepare_status.end() ? engine_build_status::unknown : it->second;
  }

  bool prepare_tensorrt_model(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model,
    const std::string &adapter_name
  ) {
    const auto engine_name = engine_filename(model);
    set_model_prepare_status(engine_name, engine_build_status::building);
    auto failed = util::fail_guard([&]() {
      set_model_prepare_status(engine_name, engine_build_status::failed);
    });

    precompile_tensorrt_engine(assets_dir, model);
    if (tensorrt_engine_build_status(assets_dir, model) != engine_build_status::ready) {
      return false;
    }

    auto &cuda = cuda_driver_api::get();
    if (!cuda.is_valid() || !ensure_cuda_initialized(cuda)) {
      BOOST_LOG(error) << "Startup depth-model preparation failed: CUDA initialization failed.";
      return false;
    }
    CUdevice cuda_device = -1;
    if (!cuda_device_for_configured_adapter(cuda, adapter_name, cuda_device)) {
      BOOST_LOG(error) << "Startup depth-model preparation failed: the configured CUDA device is unavailable.";
      return false;
    }
    CUcontext cuda_ctx = primary_context(cuda, cuda_device);
    if (!cuda_ctx) {
      BOOST_LOG(error) << "Startup depth-model preparation failed: CUDA primary context is unavailable.";
      return false;
    }
    cuda.cuCtxSetCurrent(cuda_ctx);

    const auto engine_path = assets_dir / engine_name;
    const auto engine_key = std::to_string(cuda_device) + ":" + model.name;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *exec_context = nullptr;
    bool pooled = false;
    bool create_context = false;
    {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      engine = acquire_engine_locked(engine_key, engine_path, exec_context, pooled);
      auto &slot = g_engines[engine_key];
      if (!validate_engine_io_locked(engine, slot)) {
        if (exec_context) {
          slot.context_pool.push_back(exec_context);
        }
        return false;
      }
      if (!exec_context) {
        if (slot.context_count >= kMaxContextsPerEngine) {
          // A live session already populated the engine before startup preparation finished.
          // The resident contexts are already warmed by their constructors, so no extra VRAM is
          // needed merely to satisfy the startup-prepared state.
          set_model_prepare_status(engine_name, engine_build_status::ready);
          failed.disable();
          return true;
        }
        ++slot.context_count;
        create_context = true;
      }
    }

    if (create_context) {
      BOOST_LOG(info) << "Creating startup TensorRT execution context...";
      exec_context = engine->createExecutionContext();
      if (!exec_context) {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        --g_engines[engine_key].context_count;
        g_trt_context_available.notify_all();
        return false;
      }
      if (!warmup_execution_context(cuda, cuda_ctx, exec_context)) {
        // The context cannot be destroyed safely across the MinGW/MSVC ABI boundary, so retain it
        // in the bounded pool. Do not report preparation ready: a pooled context is otherwise
        // assumed warm and the first live frame would inherit the failed lazy-load operation.
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        g_engines[engine_key].context_pool.push_back(exec_context);
        g_trt_context_available.notify_all();
        BOOST_LOG(error) << "Startup depth-model context warmup failed.";
        return false;
      }
    }

    {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      g_engines[engine_key].context_pool.push_back(exec_context);
      g_trt_context_available.notify_all();
    }
    BOOST_LOG(info) << "Startup depth model '" << model.name << "' is resident and ready.";
    set_model_prepare_status(engine_name, engine_build_status::ready);
    failed.disable();
    return true;
  }

  struct video_depth_estimator::impl {
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *exec_context = nullptr;
    std::mutex *trt_mutex = nullptr;
    CUcontext cuda_ctx = nullptr;
    CUstream cu_stream = nullptr;
    CUdevice cuda_device = -1;
    std::string engine_key;

    float ema_alpha;
    float ema_edge_change;
    float ema_edge_gradient;
    int ema_edge_dilation;
    float ema_edge_strength;
    int depth_short_side;  // depth map short-side resolution (clamped to native short side)
    float max_aspect;  // aspect cap for short-side mode
    float minmax_alpha;  // temporal EMA blend for the normalized min/max
    bool cuda_graph_enabled;
    CUgraph inference_graph = nullptr;
    CUgraphExec inference_graph_exec = nullptr;
    CUdeviceptr graph_input = 0;
    CUdeviceptr graph_output = 0;
    int graph_width = 0;
    int graph_height = 0;
    bool graph_signature_warmed = false;
    bool graph_capture_failed = false;
    bool valid = false;  // all mandatory engine, shader, and session resources are ready
    float subject_recenter;  // recenter strength consumed by depth_subject_resolve_cs
    bool subject_stretch;  // apply the shape_depth_for_pop 5/95 disparity stretch
    bool exact_plane_lock;  // Bestv2's full silhouette morphology + weighted shift mean
    float plane_lock_strength;
    float plane_lock_width;
    std::string model_name;  // local file stem; engine cache is recipe-specific
    std::string model_url;  // where to download the onnx if absent

    // Throughput telemetry for the permanent stream-cadence matched-frame pipeline.
    float measured_fps = 0.0f;
    std::chrono::steady_clock::time_point last_call_time {};
    std::chrono::steady_clock::time_point throughput_stats_start {};
    unsigned throughput_stats_calls = 0;
    unsigned throughput_stats_busy_drops = 0;
    unsigned throughput_stats_enqueues = 0;
    unsigned throughput_stats_completions = 0;

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
      const char *stage = nullptr;
    };

    perf_evt_ring perf_depth;  // "depth_infer": one DA-V2 inference

    // D3D11 timing for the work around TensorRT. CUDA events above deliberately measure only
    // the inference enqueue; these timestamp queries expose the resize/normalization input pass
    // and the depth normalization/EMA/subject passes without ever synchronizing the CPU. A ring
    // is required because query results commonly become available several source frames later.
    struct d3d_perf_slot {
      Microsoft::WRL::ComPtr<ID3D11Query> disjoint;
      Microsoft::WRL::ComPtr<ID3D11Query> post_start;
      Microsoft::WRL::ComPtr<ID3D11Query> post_end;
      Microsoft::WRL::ComPtr<ID3D11Query> pre_start;
      Microsoft::WRL::ComPtr<ID3D11Query> pre_end;
      bool pending = false;
      bool has_post = false;
      bool has_pre = false;
      std::uint64_t perf_generation = 0;
    };

    static constexpr std::size_t d3d_perf_ring_size = 16;
    std::array<d3d_perf_slot, d3d_perf_ring_size> d3d_perf_slots;
    std::size_t d3d_perf_next = 0;
    bool d3d_perf_ready = false;

    void initialize_d3d_perf() {
      if (!sbs_perf::enabled()) {
        return;
      }
      for (auto &slot : d3d_perf_slots) {
        D3D11_QUERY_DESC desc {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
        if (FAILED(device->CreateQuery(&desc, &slot.disjoint))) {
          BOOST_LOG(warning) << "Depth D3D11 timing unavailable: could not create disjoint query.";
          return;
        }
        desc.Query = D3D11_QUERY_TIMESTAMP;
        if (FAILED(device->CreateQuery(&desc, &slot.post_start)) || FAILED(device->CreateQuery(&desc, &slot.post_end)) || FAILED(device->CreateQuery(&desc, &slot.pre_start)) || FAILED(device->CreateQuery(&desc, &slot.pre_end))) {
          BOOST_LOG(warning) << "Depth D3D11 timing unavailable: could not create timestamp queries.";
          return;
        }
      }
      d3d_perf_ready = true;
    }

    void resolve_d3d_perf() {
      if (!d3d_perf_ready) {
        return;
      }
      for (auto &slot : d3d_perf_slots) {
        if (!slot.pending) {
          continue;
        }
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timing {};
        const auto ready = context->GetData(
          slot.disjoint.Get(),
          &timing,
          sizeof(timing),
          D3D11_ASYNC_GETDATA_DONOTFLUSH
        );
        if (ready == S_FALSE) {
          continue;
        }
        if (FAILED(ready)) {
          slot.pending = false;
          continue;
        }

        UINT64 post_start = 0;
        UINT64 post_end = 0;
        UINT64 pre_start = 0;
        UINT64 pre_end = 0;
        const auto post_start_status = context->GetData(slot.post_start.Get(), &post_start, sizeof(post_start), 0);
        const auto post_end_status = context->GetData(slot.post_end.Get(), &post_end, sizeof(post_end), 0);
        const auto pre_start_status = context->GetData(slot.pre_start.Get(), &pre_start, sizeof(pre_start), 0);
        const auto pre_end_status = context->GetData(slot.pre_end.Get(), &pre_end, sizeof(pre_end), 0);
        if (SUCCEEDED(post_start_status) && SUCCEEDED(post_end_status) && SUCCEEDED(pre_start_status) && SUCCEEDED(pre_end_status) && !timing.Disjoint && timing.Frequency > 0 && post_end >= post_start && pre_start >= post_end && pre_end >= pre_start) {
          const double to_ms = 1000.0 / static_cast<double>(timing.Frequency);
          if (slot.has_post) {
            sbs_perf::add_sample_ms_if_current(
              "depth_postprocess_gpu",
              static_cast<double>(post_end - post_start) * to_ms,
              slot.perf_generation
            );
          }
          if (slot.has_pre) {
            sbs_perf::add_sample_ms_if_current(
              "depth_preprocess_gpu",
              static_cast<double>(pre_end - pre_start) * to_ms,
              slot.perf_generation
            );
          }
        }
        slot.pending = false;
      }
    }

    d3d_perf_slot *begin_d3d_perf(bool has_post, bool has_pre) {
      resolve_d3d_perf();
      if (!d3d_perf_ready) {
        return nullptr;
      }
      for (std::size_t i = 0; i < d3d_perf_slots.size(); ++i) {
        const std::size_t index = (d3d_perf_next + i) % d3d_perf_slots.size();
        auto &slot = d3d_perf_slots[index];
        if (slot.pending) {
          continue;
        }
        d3d_perf_next = (index + 1) % d3d_perf_slots.size();
        slot.has_post = has_post;
        slot.has_pre = has_pre;
        slot.perf_generation = sbs_perf::generation();
        context->Begin(slot.disjoint.Get());
        context->End(slot.post_start.Get());
        return &slot;
      }
      return nullptr;  // Never stall the encode thread merely to collect telemetry.
    }

    void mark_d3d_post_end(d3d_perf_slot *slot) {
      if (slot) {
        context->End(slot->post_end.Get());
      }
    }

    void mark_d3d_pre_start(d3d_perf_slot *slot) {
      if (slot) {
        context->End(slot->pre_start.Get());
      }
    }

    void end_d3d_perf(d3d_perf_slot *slot) {
      if (!slot) {
        return;
      }
      context->End(slot->pre_end.Get());
      context->End(slot->disjoint.Get());
      slot->pending = true;
    }

    void perf_try_resolve(perf_evt_ring &r, int slot, cuda_driver_api &cuda) {
      if (!r.busy[slot] || !cuda.cuEventQuery) {
        return;
      }
      if (cuda.cuEventQuery(r.stop[slot]) != CUDA_SUCCESS) {
        return;  // not finished yet
      }
      float ms = 0.0f;
      if (cuda.cuEventElapsedTime && cuda.cuEventElapsedTime(&ms, r.start[slot], r.stop[slot]) == CUDA_SUCCESS) {
        sbs_perf::add_sample_ms(r.stage, ms);
      }
      r.busy[slot] = false;
    }

    void perf_drain(perf_evt_ring &r) {
      auto &cuda = cuda_driver_api::get();
      for (int i = 0; i < perf_evt_ring::N; i++) {
        perf_try_resolve(r, i, cuda);
      }
    }

    // Record a start event before an enqueue; returns the ring slot (or -1 to skip timing).
    int perf_begin(perf_evt_ring &r, CUstream stream) {
      if (!sbs_perf::enabled()) {
        return -1;
      }
      auto &cuda = cuda_driver_api::get();
      if (!cuda.cuEventCreate || !cuda.cuEventRecord) {
        return -1;
      }
      int slot = r.head;
      perf_try_resolve(r, slot, cuda);  // reclaim the slot if its prior sample is ready
      if (r.busy[slot]) {
        return -1;  // still in flight -> drop this measurement
      }
      if (!r.start[slot] && cuda.cuEventCreate(&r.start[slot], CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
        return -1;
      }
      if (!r.stop[slot] && cuda.cuEventCreate(&r.stop[slot], CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
        return -1;
      }
      if (cuda.cuEventRecord(r.start[slot], stream) != CUDA_SUCCESS) {
        return -1;
      }
      return slot;
    }

    // Record the stop event after the enqueue and mark the slot pending.
    void perf_end(perf_evt_ring &r, int slot, CUstream stream) {
      if (slot < 0) {
        return;
      }
      auto &cuda = cuda_driver_api::get();
      if (!cuda.cuEventRecord || cuda.cuEventRecord(r.stop[slot], stream) != CUDA_SUCCESS) {
        return;
      }
      r.busy[slot] = true;
      r.head = (r.head + 1) % perf_evt_ring::N;
    }

    void perf_destroy_events() {
      auto &cuda = cuda_driver_api::get();
      if (!cuda.cuEventDestroy) {
        return;
      }
      for (auto *r : {&perf_depth}) {
        for (int i = 0; i < perf_evt_ring::N; i++) {
          if (r->start[i]) {
            cuda.cuEventDestroy(r->start[i]);
          }
          if (r->stop[i]) {
            cuda.cuEventDestroy(r->stop[i]);
          }
          r->start[i] = r->stop[i] = nullptr;
        }
      }
    }

    void destroy_inference_graph(cuda_driver_api &cuda) {
      if (inference_graph_exec && cuda.cuGraphExecDestroy) {
        cuda.cuGraphExecDestroy(inference_graph_exec);
      }
      if (inference_graph && cuda.cuGraphDestroy) {
        cuda.cuGraphDestroy(inference_graph);
      }
      inference_graph_exec = nullptr;
      inference_graph = nullptr;
      graph_signature_warmed = false;
    }

    bool enqueue_inference(CUdeviceptr input, CUdeviceptr output, cuda_driver_api &cuda) {
      const bool graph_api = cuda_graph_enabled && cuda.cuStreamBeginCapture &&
                             cuda.cuStreamEndCapture && cuda.cuGraphInstantiateWithFlags &&
                             cuda.cuGraphLaunch && cuda.cuGraphDestroy &&
                             cuda.cuGraphExecDestroy;
      if (!graph_api || graph_capture_failed) {
        return exec_context->enqueueV3(cu_stream);
      }
      auto launch_or_fallback = [&]() {
        const CUresult launch = cuda.cuGraphLaunch(inference_graph_exec, cu_stream);
        if (launch == CUDA_SUCCESS) {
          return true;
        }
        BOOST_LOG(warning) << "TensorRT CUDA graph launch failed (" << launch
                           << "); using ordinary enqueue.";
        destroy_inference_graph(cuda);
        graph_capture_failed = true;
        return exec_context->enqueueV3(cu_stream);
      };

      // CUDA explicitly permits an interop mapping to return a different address on each map.
      // A graph embeds TensorRT's tensor pointers, so never replay it across a changed mapping or
      // shape. The first enqueue after each signature change is deliberately ordinary: TensorRT
      // may perform deferred shape-dependent setup that cannot be captured.
      if (input != graph_input || output != graph_output || target_w != graph_width || target_h != graph_height) {
        destroy_inference_graph(cuda);
        graph_input = input;
        graph_output = output;
        graph_width = target_w;
        graph_height = target_h;
      }
      if (inference_graph_exec) {
        return launch_or_fallback();
      }
      if (!graph_signature_warmed) {
        graph_signature_warmed = true;
        return exec_context->enqueueV3(cu_stream);
      }

      CUgraph captured = nullptr;
      const CUresult begin = cuda.cuStreamBeginCapture(
        cu_stream,
        CU_STREAM_CAPTURE_MODE_RELAXED
      );
      const bool captured_enqueue = begin == CUDA_SUCCESS && exec_context->enqueueV3(cu_stream);
      const CUresult end = begin == CUDA_SUCCESS ?
                             cuda.cuStreamEndCapture(cu_stream, &captured) :
                             begin;
      if (captured_enqueue && end == CUDA_SUCCESS && captured && cuda.cuGraphInstantiateWithFlags(&inference_graph_exec, captured, 0) == CUDA_SUCCESS && inference_graph_exec) {
        inference_graph = captured;
        BOOST_LOG(info) << "TensorRT CUDA graph captured for " << target_w << 'x' << target_h << '.';
        return launch_or_fallback();
      }

      if (captured) {
        cuda.cuGraphDestroy(captured);
      }
      inference_graph_exec = nullptr;
      inference_graph = nullptr;
      graph_capture_failed = true;
      BOOST_LOG(warning) << "TensorRT CUDA graph capture failed (begin=" << begin
                         << ", enqueue=" << captured_enqueue << ", end=" << end
                         << "); using ordinary enqueue.";
      return exec_context->enqueueV3(cu_stream);
    }

    // Caching
    int target_w = 0;
    int target_h = 0;
    UINT reduce_groups = 0;  // threadgroups for the min/max reduction (groups * 256 = total threads)
    int cb_color_mode = -1;  // input_color_space baked into constant buffers

    Microsoft::WRL::ComPtr<ID3D11ComputeShader> rgb_to_nchw_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> buffer_to_tex_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_ema_motion_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_minmax_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_minmax_ema_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_hist_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_subject_hist_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_subject_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_plane_band_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_plane_filter_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_plane_combine_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_plane_reduce_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_plane_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11Buffer> plane_cbuffer;
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

    Microsoft::WRL::ComPtr<ID3D11Texture2D> plane_band_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> plane_band_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> plane_band_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> plane_work_a_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> plane_work_a_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> plane_work_a_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> plane_work_b_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> plane_work_b_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> plane_work_b_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> plane_group_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> plane_group_uav;
    UINT plane_group_count = 0;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_previous_tex;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_previous_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> ema_motion_mask_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> ema_motion_mask_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ema_motion_mask_srv;
    bool depth_history_valid = false;

    CUgraphicsResource cuda_in_res = nullptr;
    CUgraphicsResource cuda_out_res = nullptr;
    bool has_previous_frame = false;
    std::uint64_t pending_frame_id = 0;
    bool stream_error_logged = false;
    bool depth_context_pooled = false;  // context reused from the pool (modules already loaded -> skip warmup)

    bool compile_shader(const std::filesystem::path &path, Microsoft::WRL::ComPtr<ID3D11ComputeShader> &out_cs) {
      std::vector<std::uint8_t> bytecode;
      if (!depth_shader_bytecode(path, bytecode)) {
        return false;
      }
      return SUCCEEDED(device->CreateComputeShader(bytecode.data(), bytecode.size(), nullptr, &out_cs));
    }

    impl(Microsoft::WRL::ComPtr<ID3D11Device> d, Microsoft::WRL::ComPtr<ID3D11DeviceContext> c, const std::filesystem::path &assets_dir, const config::video_t::sbs_t &cfg, const config::depth_model_info &model):
        device(d),
        context(c),
        ema_alpha((float) cfg.ema),
        ema_edge_change((float) cfg.ema_edge_change),
        ema_edge_gradient((float) cfg.ema_edge_gradient),
        ema_edge_dilation(cfg.ema_edge_dilation),
        ema_edge_strength((float) cfg.ema_edge_strength),
        depth_short_side(std::max(196, cfg.depth_short_side)),
        max_aspect(std::max(1.0f, (float) cfg.depth_max_aspect)),
        minmax_alpha((float) cfg.minmax_ema),
        cuda_graph_enabled(cfg.cuda_graph),
        subject_recenter((float) cfg.subject_recenter),
        subject_stretch(cfg.subject_stretch),
        exact_plane_lock(cfg.subject_plane_lock > 0.0),
        plane_lock_strength((float) cfg.subject_plane_lock),
        plane_lock_width((float) cfg.subject_plane_width),
        model_name(model.name),
        model_url(model.url) {
      const auto init_started = std::chrono::steady_clock::now();
      // Perf benchmark: enable per-stage timing for this run and reset the rolling window
      // so it reflects this encode session rather than blending across a rebuild.
      perf_depth.stage = "depth_infer";
      sbs_perf::set_enabled(cfg.perf_stats);
      if (cfg.perf_stats) {
        sbs_perf::reset();
      }
      initialize_d3d_perf();

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

      auto &cuda = cuda_driver_api::get();
      if (cuda.is_valid() && ensure_cuda_initialized(cuda)) {
        if (cuda_device_for_d3d(cuda, device.Get(), cuda_device)) {
          cuda_ctx = primary_context(cuda, cuda_device);
          if (cuda_ctx) {
            cuda.cuCtxSetCurrent(cuda_ctx);
            cuda.cuStreamCreate(&cu_stream, CU_STREAM_NON_BLOCKING);
          }
        }
      }
      engine_key = std::to_string(cuda_device) + ":" + model_name;

      {  // Scope this lock to the g_engines/g_runtime access only: it MUST be released before
         // warmup_inference() at the end of the ctor (which re-locks g_trt_mutex) -- a
         // non-recursive std::mutex would otherwise self-deadlock and hang construction.
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        // Load (once) the engine for this configured model into its own slot and take a pooled
        // execution context if one is free. Different startup configurations remain isolated.
        engine = acquire_engine_locked(engine_key, model_path, exec_context, depth_context_pooled);
        if (depth_context_pooled) {
          BOOST_LOG(info) << "Reusing pooled TensorRT execution context.";
        }
        auto &slot = g_engines[engine_key];

        if (!validate_engine_io_locked(engine, slot)) {
          BOOST_LOG(error) << "Depth engine I/O contract is incompatible with Apollo; streaming flat SBS.";
          if (exec_context) {
            slot.context_pool.push_back(exec_context);
            exec_context = nullptr;
            g_trt_context_available.notify_all();
          }
          engine = nullptr;
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
          auto &pool = g_engines[engine_key].context_pool;
          if (!pool.empty()) {
            exec_context = pool.back();
            pool.pop_back();
            depth_context_pooled = true;
            BOOST_LOG(info) << "Reusing pooled TensorRT execution context (freed by a racing teardown).";
          }
        }
        bool create_context = false;
        if (!exec_context) {
          std::unique_lock<std::mutex> lock(g_trt_mutex);
          auto &slot = g_engines[engine_key];
          if (slot.context_pool.empty() && slot.context_count >= kMaxContextsPerEngine) {
            BOOST_LOG(warning) << "TensorRT context cap reached for this depth model; waiting for "
                                  "an asynchronous encoder teardown to return one.";
            const bool available = g_trt_context_available.wait_for(
              lock,
              std::chrono::seconds(5),
              [&slot]() {
                return !slot.context_pool.empty() || slot.context_count < kMaxContextsPerEngine;
              }
            );
            if (!available) {
              BOOST_LOG(error) << "TensorRT context cap remained saturated; leaving this encode "
                                  "session flat instead of allocating unbounded GPU memory.";
              engine = nullptr;
            }
          }
          if (engine && !slot.context_pool.empty()) {
            exec_context = slot.context_pool.back();
            slot.context_pool.pop_back();
            depth_context_pooled = true;
            BOOST_LOG(info) << "Reusing pooled TensorRT execution context after bounded wait.";
          } else if (engine) {
            ++slot.context_count;  // reserve atomically so concurrent constructors cannot exceed the cap
            create_context = true;
          }
        }
        if (create_context) {
          // Deliberately OUTSIDE g_trt_mutex: creation allocates device scratch and can
          // take many seconds; holding the lock would block a concurrent estimator
          // destructor from returning its context to the pool (observed 46 s teardown)
          // and any concurrent enqueueV3. ICudaEngine is thread-safe for this call.
          BOOST_LOG(info) << "Creating TensorRT execution context (allocates device scratch; may take several seconds)...";
          exec_context = engine->createExecutionContext();
          if (!exec_context) {
            std::lock_guard<std::mutex> lock(g_trt_mutex);
            auto &slot = g_engines[engine_key];
            --slot.context_count;
            g_trt_context_available.notify_all();
          }
        }
      }

      // Bestv2 normalization and subject shaping are one permanent pipeline. Never create a
      // partially usable estimator: without any one of these shaders the warp would either
      // consume invalid bounds or silently collapse to flat 2D.
      const bool core_shaders_ok =
        compile_shader(assets_dir / "shaders" / "directx" / "rgb_to_nchw_cs.hlsl", rgb_to_nchw_cs) &&
        compile_shader(assets_dir / "shaders" / "directx" / "buffer_to_tex_cs.hlsl", buffer_to_tex_cs) &&
        compile_shader(assets_dir / "shaders" / "directx" / "depth_ema_motion_cs.hlsl", depth_ema_motion_cs) &&
        compile_shader(assets_dir / "shaders" / "directx" / "depth_minmax_cs.hlsl", depth_minmax_cs) &&
        compile_shader(assets_dir / "shaders" / "directx" / "depth_minmax_ema_cs.hlsl", depth_minmax_ema_cs) &&
        compile_shader(assets_dir / "shaders" / "directx" / "depth_hist_cs.hlsl", depth_hist_cs) &&
        compile_shader(assets_dir / "shaders" / "directx" / "depth_subject_hist_cs.hlsl", depth_subject_hist_cs) &&
        compile_shader(assets_dir / "shaders" / "directx" / "depth_subject_resolve_cs.hlsl", depth_subject_resolve_cs);
      if (!core_shaders_ok) {
        BOOST_LOG(error) << "Depth estimator failed: required Bestv2 shader initialization failed.";
        return;
      }
      BOOST_LOG(info) << "Permanent Bestv2 subject shaping enabled (recenter " << subject_recenter << ").";
      if (exact_plane_lock) {
        bool ok = compile_shader(assets_dir / "shaders" / "directx" / "depth_plane_band_cs.hlsl", depth_plane_band_cs) &&
                  compile_shader(assets_dir / "shaders" / "directx" / "depth_plane_filter_cs.hlsl", depth_plane_filter_cs) &&
                  compile_shader(assets_dir / "shaders" / "directx" / "depth_plane_combine_cs.hlsl", depth_plane_combine_cs) &&
                  compile_shader(assets_dir / "shaders" / "directx" / "depth_plane_reduce_cs.hlsl", depth_plane_reduce_cs) &&
                  compile_shader(assets_dir / "shaders" / "directx" / "depth_plane_resolve_cs.hlsl", depth_plane_resolve_cs);
        if (ok) {
          D3D11_BUFFER_DESC bd = {};
          bd.Usage = D3D11_USAGE_DYNAMIC;
          bd.ByteWidth = 48;
          bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
          bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
          if (FAILED(device->CreateBuffer(&bd, nullptr, &plane_cbuffer))) {
            ok = false;
          }
        }
        if (ok) {
          BOOST_LOG(info) << "Bestv2 exact subject-plane lock enabled (strength "
                          << plane_lock_strength << ", width " << plane_lock_width << ").";
        } else {
          BOOST_LOG(warning) << "Bestv2 subject-plane-lock shaders failed; using the depth-band fallback.";
          exact_plane_lock = false;
        }
      } else {
        exact_plane_lock = false;
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

      // EMA'd P2/P98 bounds. initialized = 0 so the first frame seeds directly.
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

      // Permanent P2/P98 histogram: 256 uint bins, reset after every scan.
      {
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
          BOOST_LOG(error) << "Required P2/P98 histogram buffer creation failed.";
        }
      }

      // Subject tracking: weighted histogram (256 uint bins) + per-frame state. The third
      // float4 carries the exact Bestv2 plane-lock mean and its initialized flag.
      {
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

        float init_state[12] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // 3 float4
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
      }

      D3D11_SAMPLER_DESC samp_desc = {};
      samp_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      samp_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
      samp_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
      samp_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      device->CreateSamplerState(&samp_desc, &linear_sampler);

      valid = engine && exec_context && cu_stream && rgb_to_nchw_cs && buffer_to_tex_cs &&
              depth_minmax_cs && depth_minmax_ema_cs && depth_hist_cs &&
              depth_subject_hist_cs && depth_subject_resolve_cs &&
              minmax_raw_uav && minmax_ema_uav && minmax_ema_srv && hist_uav &&
              subject_hist_uav && subject_plain_uav && subject_uav && subject_srv &&
              linear_sampler;
      if (!valid) {
        BOOST_LOG(error) << "Depth estimator failed: required engine or Bestv2 GPU resource initialization failed.";
        return;
      }

      // Constant buffers are created in ensure_cbuffers() once the model resolution is
      // known: every field is fixed for the session, so they are built once (immutable)
      // instead of being re-mapped on the encode thread every frame.

      // Warm up here so TensorRT's CUDA lazy kernel load / JIT (~20 s on the big models)
      // happens during construction -- which ensure_depth_estimator() runs on a background
      // thread -- rather than stalling the first real convert() on the encode thread and
      // freezing the stream when Host SBS first becomes active.
      warmup_inference();
      BOOST_LOG(info) << "Depth estimator pipeline initialized in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - init_started
                         )
                           .count()
                      << " ms";
    }

    // Run one throwaway inference at the engine's optimization shape so TensorRT loads its
    // CUDA modules now. The bulk of the "first inference" cost is module loading, which is
    // shape-independent, so a warmup at the OPT shape spares the first real frame the stall
    // even if its resolution differs. Uses its own scratch device buffers because the per-
    // frame D3D-interop buffers aren't allocated until convert() knows the frame resolution.
    // Pure CUDA + TensorRT (no D3D immediate context), so it's safe on the construction thread.
    void warmup_inference() {
      if (!exec_context || !cu_stream) {
        return;
      }
      if (depth_context_pooled) {
        return;  // modules already loaded; warmup is pure waste on a pooled context
      }
      auto &cuda = cuda_driver_api::get();
      if (!cuda.is_valid()) {
        return;
      }
      if (cuda_ctx) {
        cuda.cuCtxSetCurrent(cuda_ctx);
      }

      const int h = depth_engine_opt_height;
      const int w = depth_engine_opt_width;

      const size_t in_elems = (size_t) 3 * h * w;  // batch/view dims are 1 for rank-4 and rank-5
      const size_t out_elems = (size_t) h * w;
      CUdeviceptr d_in = 0, d_out = 0;
      if (cuda.cuMemAlloc(&d_in, in_elems * sizeof(float)) != 0) {
        return;
      }
      if (cuda.cuMemAlloc(&d_out, out_elems * sizeof(float)) != 0) {
        cuda.cuMemFree(d_in);
        return;
      }

      nvinfer1::Dims in_dims = make_input_dims(h, w);
      exec_context->setInputShape("pixel_values", in_dims);
      exec_context->setTensorAddress("pixel_values", (void *) d_in);
      exec_context->setTensorAddress("predicted_depth", (void *) d_out);
      bool ok;
      {
        std::lock_guard<std::mutex> lock(*trt_mutex);
        ok = exec_context->enqueueV3(cu_stream);
      }
      if (ok && cuda.cuStreamSynchronize) {
        cuda.cuStreamSynchronize(cu_stream);
      }
      cuda.cuMemFree(d_in);
      cuda.cuMemFree(d_out);
      BOOST_LOG(info) << "Depth estimator warmup inference complete (" << w << 'x' << h
                      << (ok ? ")." : "); enqueue failed, first frame may stall.");
    }

    ~impl() {
      auto &cuda = cuda_driver_api::get();
      if (cuda.is_valid() && cuda_ctx) {
        cuda.cuCtxSetCurrent(cuda_ctx);
        if (cu_stream) {
          if (cuda.cuStreamSynchronize) {
            cuda.cuStreamSynchronize(cu_stream);
          }
          destroy_inference_graph(cuda);
          cuda.cuStreamDestroy(cu_stream);
        }
        if (cuda_in_res) {
          cuda.cuGraphicsUnregisterResource(cuda_in_res);
        }
        if (cuda_out_res) {
          cuda.cuGraphicsUnregisterResource(cuda_out_res);
        }
        perf_destroy_events();  // free the timing events while cuda_ctx is still current
      }

      // Return the execution contexts to their engine's pool for reuse instead of leaking
      // (or destroying, which faults across the DLL boundary). The streams were
      // synchronized above, so no inference is still in flight referencing this
      // instance's tensor bindings, making the contexts safe for another instance to reuse.
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      if (exec_context) {
        g_engines[engine_key].context_pool.push_back(exec_context);
        exec_context = nullptr;
        g_trt_context_available.notify_all();
      }
      // TRT runtime/engines are cached globally, do not destroy them here.
    }

    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> output_srv() {
      return depth_srv;
    }

    struct plane_constants_t {
      uint32_t w, h;
      float strength, width;
      uint32_t axis, radius, op, group_count;
      uint32_t subject_stretch, reserved[3];
    };

    static_assert(sizeof(plane_constants_t) == 48, "PlaneConstants must match depth_plane_constants.hlsl");

    bool set_plane_constants(uint32_t axis = 0, uint32_t radius = 0, uint32_t op = 0) {
      if (!plane_cbuffer) {
        return false;
      }
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(plane_cbuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        return false;
      }
      plane_constants_t pc {(uint32_t) target_w, (uint32_t) target_h, plane_lock_strength, plane_lock_width, axis, radius, op, plane_group_count, subject_stretch ? 1u : 0u, {0u, 0u, 0u}};
      std::memcpy(mapped.pData, &pc, sizeof(pc));
      context->Unmap(plane_cbuffer.Get(), 0);
      return true;
    }

    void run_exact_plane_lock() {
      if (!exact_plane_lock || !depth_srv || !subject_srv || !subject_uav || !plane_band_uav || !plane_work_a_uav || !plane_work_b_uav || !plane_group_uav || !set_plane_constants()) {
        return;
      }

      ID3D11Buffer *cb = plane_cbuffer.Get();
      context->CSSetConstantBuffers(1, 1, &cb);
      const UINT gx = ((UINT) target_w + 15u) / 16u;
      const UINT gy = ((UINT) target_h + 15u) / 16u;

      // Depth band + Bestv2 center weighting.
      context->CSSetShader(depth_plane_band_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *band_srvs[2] = {depth_srv.Get(), subject_srv.Get()};
      context->CSSetShaderResources(0, 2, band_srvs);
      context->CSSetUnorderedAccessViews(0, 1, plane_band_uav.GetAddressOf(), nullptr);
      context->Dispatch(gx, gy, 1);
      ID3D11ShaderResourceView *null_srvs3[3] = {nullptr, nullptr, nullptr};
      ID3D11UnorderedAccessView *null_uavs2[2] = {nullptr, nullptr};
      context->CSSetUnorderedAccessViews(0, 1, null_uavs2, nullptr);
      context->CSSetShaderResources(0, 2, null_srvs3);

      auto filter = [&](ID3D11ShaderResourceView *input, ID3D11UnorderedAccessView *output, uint32_t axis, uint32_t radius, uint32_t op) {
        set_plane_constants(axis, radius, op);
        context->CSSetShader(depth_plane_filter_cs.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, &input);
        context->CSSetUnorderedAccessViews(0, 1, &output, nullptr);
        context->Dispatch(gx, gy, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_uavs2, nullptr);
        context->CSSetShaderResources(0, 1, null_srvs3);
      };

      // Bestv2's 21/15/13 kernels were calibrated on a 336px-short-side depth grid. Scale
      // their radii with this model grid's short side so the silhouette support covers the
      // same fraction of the source at 336, 420, 434, or any future inference resolution.
      const float morphology_scale = (float) std::min(target_w, target_h) / 336.0f;
      const uint32_t dilate_radius = std::max(1u, (uint32_t) std::lround(10.0f * morphology_scale));
      const uint32_t close_radius = std::max(1u, (uint32_t) std::lround(7.0f * morphology_scale));
      const uint32_t smooth_radius = std::max(1u, (uint32_t) std::lround(6.0f * morphology_scale));
      filter(plane_band_srv.Get(), plane_work_a_uav.Get(), 0, dilate_radius, 0);
      filter(plane_work_a_srv.Get(), plane_work_b_uav.Get(), 1, dilate_radius, 0);
      filter(plane_work_b_srv.Get(), plane_work_a_uav.Get(), 0, close_radius, 1);
      filter(plane_work_a_srv.Get(), plane_work_b_uav.Get(), 1, close_radius, 1);

      // max(original, closed*.70), then exact 13x13 average smoothing.
      set_plane_constants();
      context->CSSetShader(depth_plane_combine_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *combine_srvs[2] = {plane_band_srv.Get(), plane_work_b_srv.Get()};
      context->CSSetShaderResources(0, 2, combine_srvs);
      context->CSSetUnorderedAccessViews(0, 1, plane_work_a_uav.GetAddressOf(), nullptr);
      context->Dispatch(gx, gy, 1);
      context->CSSetUnorderedAccessViews(0, 1, null_uavs2, nullptr);
      context->CSSetShaderResources(0, 2, null_srvs3);
      filter(plane_work_a_srv.Get(), plane_work_b_uav.Get(), 0, smooth_radius, 2);
      filter(plane_work_b_srv.Get(), plane_work_a_uav.Get(), 1, smooth_radius, 2);

      // Weighted mean of the actual Bestv2 raw shift field.
      set_plane_constants();
      context->CSSetShader(depth_plane_reduce_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *reduce_srvs[3] = {depth_srv.Get(), plane_work_a_srv.Get(), subject_srv.Get()};
      context->CSSetShaderResources(0, 3, reduce_srvs);
      context->CSSetUnorderedAccessViews(0, 1, plane_group_uav.GetAddressOf(), nullptr);
      context->Dispatch(gx, gy, 1);
      context->CSSetUnorderedAccessViews(0, 1, null_uavs2, nullptr);
      context->CSSetShaderResources(0, 3, null_srvs3);

      context->CSSetShader(depth_plane_resolve_cs.Get(), nullptr, 0);
      ID3D11UnorderedAccessView *resolve_uavs[2] = {plane_group_uav.Get(), subject_uav.Get()};
      context->CSSetUnorderedAccessViews(0, 2, resolve_uavs, nullptr);
      context->Dispatch(1, 1, 1);
      context->CSSetUnorderedAccessViews(0, 2, null_uavs2, nullptr);
    }

    estimate_result make_result(bool completed_frame_valid = false, std::uint64_t completed_frame_id = 0, bool inference_enqueued = false, std::uint64_t enqueued_frame_id = 0) {
      estimate_result r;
      r.depth = output_srv();
      r.subject = subject_srv;
      if (exact_plane_lock) {
        r.plane_lock = plane_work_a_srv;
      }
      r.ema_motion_mask = ema_motion_mask_srv;
      r.raw_model_depth = tensor_out_srv;
      r.raw_width = target_w;
      r.raw_height = target_h;
      r.completed_frame_valid = completed_frame_valid;
      r.completed_frame_id = completed_frame_id;
      r.inference_enqueued = inference_enqueued;
      r.enqueued_frame_id = enqueued_frame_id;
      r.cuda_graph_active = inference_graph_exec != nullptr && !graph_capture_failed;
      return r;
    }

    // estimate() has already submitted one inference. Wait for that exact inference, consume it
    // once, and deliberately do NOT enqueue a duplicate. This is the synchronous quality oracle.
    estimate_result finish_pending(input_color_space color_space) {
      auto &cuda = cuda_driver_api::get();
      if (!has_previous_frame || !cu_stream || !cuda.cuStreamSynchronize) {
        return make_result();
      }
      if (cuda_ctx) {
        cuda.cuCtxSetCurrent(cuda_ctx);
      }
      CUresult sync = cuda.cuStreamSynchronize(cu_stream);
      if (sync != CUDA_SUCCESS) {
        BOOST_LOG(error) << "Depth synchronization failed: " << sync;
        return make_result();
      }
      if (sbs_perf::enabled()) {
        perf_drain(perf_depth);
      }
      ensure_cbuffers(color_space);
      if (!cbuffer) {
        return {};
      }
      auto *d3d_timer = begin_d3d_perf(true, false);
      normalize_depth_output();
      mark_d3d_post_end(d3d_timer);
      mark_d3d_pre_start(d3d_timer);
      end_d3d_perf(d3d_timer);
      const auto completed_frame_id = pending_frame_id;
      has_previous_frame = false;  // the output buffer has been consumed; never fold it twice
      return make_result(true, completed_frame_id);
    }

    // (Re)build the two constant buffers. All contents are session-constant once the model
    // resolution is fixed, so they are immutable buffers created once -- rebuilt only if
    // Rebuild if capture color encoding changes during a display/mode transition.
    void ensure_cbuffers(input_color_space color_space) {
      const int color_mode = (int) color_space;
      if (cb_color_mode == color_mode && cbuffer) {
        return;
      }
      cb_color_mode = color_mode;

      D3D11_BUFFER_DESC cb_desc = {};
      cb_desc.Usage = D3D11_USAGE_IMMUTABLE;
      cb_desc.ByteWidth = 80;  // shared depth-pass cbuffer (20 floats/uints; see below)
      cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

      // Shared depth-pass constants, 20 scalars = 5 float4 registers. THIS fill is the
      // single source of truth for the canonical layout in
      // shaders/directx/include/depth_constants.hlsl -- every cbf[N] below must stay
      // slot-for-slot with the include (which every depth shader #includes). To add a
      // field: append it here AND to the include. Slots 13-14 are reserved so the subject
      // fields retain their established offsets.
      uint32_t cb[20] = {};
      float *cbf = (float *) cb;
      cb[0] = (uint32_t) target_w;
      cb[1] = (uint32_t) target_h;
      cb[2] = (uint32_t) color_mode;
      cbf[3] = ema_alpha;
      cbf[4] = minmax_alpha;
      cb[5] = reduce_groups * 256u;  // total threads for the reduction grid-stride
      cbf[6] = ema_edge_change;
      cbf[7] = ema_edge_gradient;
      cbf[8] = (float) ema_edge_dilation;
      cbf[9] = ema_edge_strength;
      cbf[15] = subject_recenter;  // subject recenter strength (depth_subject_resolve_cs)
      cbf[18] = subject_stretch ? 1.0f : 0.0f;
      D3D11_SUBRESOURCE_DATA sd = {cb, 0, 0};
      cbuffer.Reset();
      device->CreateBuffer(&cb_desc, &sd, &cbuffer);
    }

    // Normalize the finished raw disparity in tensor_out_buf into depth_tex: the scale
    // passes (min/max reduction, permanent percentile histogram, EMA fold) followed by the
    // mapping/temporal-EMA pass. GPU-resident throughout, no CPU readback.
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

        ID3D11UnorderedAccessView *null_uav1 = nullptr;
        ID3D11ShaderResourceView *null_srv1 = nullptr;
        context->CSSetUnorderedAccessViews(0, 1, &null_uav1, nullptr);
        context->CSSetShaderResources(0, 1, &null_srv1);

        // Pass A2 (percentile mode): 256-bin histogram over the raw range, so pass B
        // can replace the outlier-sensitive min/max with robust percentile bounds.
        if (depth_hist_cs && hist_uav) {
          context->CSSetShader(depth_hist_cs.Get(), nullptr, 0);
          context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
          context->CSSetShaderResources(0, 1, tensor_out_srv.GetAddressOf());
          ID3D11UnorderedAccessView *hist_uavs[2] = {hist_uav.Get(), minmax_raw_uav.Get()};
          context->CSSetUnorderedAccessViews(0, 2, hist_uavs, nullptr);
          context->Dispatch(reduce_groups, 1, 1);

          ID3D11UnorderedAccessView *null_uavs_h[2] = {nullptr, nullptr};
          context->CSSetUnorderedAccessViews(0, 2, null_uavs_h, nullptr);
          context->CSSetShaderResources(0, 1, &null_srv1);
        }

        // [NORMDBG] opt-in raw min/max trajectory for normalization diagnosis. This is
        // intentionally not perf-gated: its per-frame Map is a CPU sync and would distort
        // timing measurements. Enable only with APOLLO_NORMDBG.
        static const bool normdbg = std::getenv("APOLLO_NORMDBG") != nullptr;
        if (normdbg && minmax_raw_stage) {
          context->CopyResource(minmax_raw_stage.Get(), minmax_raw_buf.Get());
          D3D11_MAPPED_SUBRESOURCE ms {};
          if (SUCCEEDED(context->Map(minmax_raw_stage.Get(), 0, D3D11_MAP_READ, 0, &ms))) {
            const uint32_t *u = (const uint32_t *) ms.pData;
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
        ID3D11UnorderedAccessView *ema_uavs[3] = {minmax_ema_uav.Get(), minmax_raw_uav.Get(), hist_uav.Get()};
        context->CSSetUnorderedAccessViews(0, 3, ema_uavs, nullptr);
        context->Dispatch(1, 1, 1);

        ID3D11UnorderedAccessView *null_uav2[3] = {nullptr, nullptr, nullptr};
        context->CSSetUnorderedAccessViews(0, 3, null_uav2, nullptr);
      }

      // Snapshot the complete previous depth before any thread writes the new result. The
      // experimental mask reads neighborhoods, so an in-place read/write pass would be racy.
      context->CopyResource(depth_previous_tex.Get(), depth_tex.Get());

      const UINT clear_mask[4] = {0u, 0u, 0u, 0u};
      if (ema_edge_change > 0.0f && ema_edge_gradient > 0.0f && depth_history_valid) {
        context->CSSetShader(depth_ema_motion_cs.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
        ID3D11ShaderResourceView *mask_srvs[3] = {
          tensor_out_srv.Get(),
          minmax_ema_srv.Get(),
          depth_previous_srv.Get()
        };
        context->CSSetShaderResources(0, 3, mask_srvs);
        context->CSSetUnorderedAccessViews(0, 1, ema_motion_mask_uav.GetAddressOf(), nullptr);
        context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);
        ID3D11UnorderedAccessView *null_mask_uav = nullptr;
        ID3D11ShaderResourceView *null_mask_srvs[3] = {nullptr, nullptr, nullptr};
        context->CSSetUnorderedAccessViews(0, 1, &null_mask_uav, nullptr);
        context->CSSetShaderResources(0, 3, null_mask_srvs);
      } else {
        context->ClearUnorderedAccessViewUint(ema_motion_mask_uav.Get(), clear_mask);
      }

      // 3b. Buffer to Texture: normalize disparity and either apply temporal EMA or snap the
      // pixels selected by the deterministic moving-edge mask.
      context->CSSetShader(buffer_to_tex_cs.Get(), nullptr, 0);
      context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
      ID3D11ShaderResourceView *bt_srvs[4] = {
        tensor_out_srv.Get(),
        minmax_ema_srv.Get(),
        depth_previous_srv.Get(),
        ema_motion_mask_srv.Get()
      };
      context->CSSetShaderResources(0, 4, bt_srvs);
      context->CSSetUnorderedAccessViews(0, 1, depth_uav.GetAddressOf(), nullptr);

      context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

      ID3D11UnorderedAccessView *null_uav2[2] = {nullptr, nullptr};
      ID3D11ShaderResourceView *null_srvs[4] = {nullptr, nullptr, nullptr, nullptr};
      context->CSSetUnorderedAccessViews(0, 1, null_uav2, nullptr);
      context->CSSetShaderResources(0, 4, null_srvs);
      depth_history_valid = true;

      // 3s. Subject tracking: weighted depth histogram over the freshly-normalized
      // depth, then a 1-thread resolve into the subject state the reprojection reads.
      {
        context->CSSetShader(depth_subject_hist_cs.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
        context->CSSetShaderResources(0, 1, depth_srv.GetAddressOf());
        ID3D11UnorderedAccessView *hist_uavs[2] = {subject_hist_uav.Get(), subject_plain_uav.Get()};
        context->CSSetUnorderedAccessViews(0, 2, hist_uavs, nullptr);
        context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

        ID3D11UnorderedAccessView *null_uavs_h2[2] = {nullptr, nullptr};
        context->CSSetUnorderedAccessViews(0, 2, null_uavs_h2, nullptr);
        context->CSSetShaderResources(0, 1, null_srvs);

        context->CSSetShader(depth_subject_resolve_cs.Get(), nullptr, 0);
        ID3D11UnorderedAccessView *subj_uavs[3] = {subject_hist_uav.Get(), subject_uav.Get(), subject_plain_uav.Get()};
        context->CSSetUnorderedAccessViews(0, 3, subj_uavs, nullptr);
        context->Dispatch(1, 1, 1);

        ID3D11UnorderedAccessView *null_uavs2[3] = {nullptr, nullptr, nullptr};
        context->CSSetUnorderedAccessViews(0, 3, null_uavs2, nullptr);

        run_exact_plane_lock();

        // Ground-truth log for the original Bestv2 reference's LOW=near convention. Apollo is
        // HIGH=near, so print both subject values for direct comparison. Opt-in via APOLLO_SUBJDBG (NOT
        // perf-gated: CopyResource+Map is a CPU/GPU sync that would perturb the very
        // perf numbers a benchmark run measures), every 24 updates, off the ship path.
        static const bool subjdbg = std::getenv("APOLLO_SUBJDBG") != nullptr;
        if (subjdbg && subject_stage && (++subject_log_counter % 24u) == 1u) {
          context->CopyResource(subject_stage.Get(), subject_buf.Get());
          D3D11_MAPPED_SUBRESOURCE ms {};
          if (SUCCEEDED(context->Map(subject_stage.Get(), 0, D3D11_MAP_READ, 0, &ms))) {
            const float *s = (const float *) ms.pData;  // {delta, scurve, subj_ema, init}
            BOOST_LOG(info) << "[SUBJDBG] u=" << subject_log_counter
                            << " subj_hi_near=" << s[2]
                            << " subj_low_near=" << (1.0f - s[2])
                            << " recenter_delta=" << s[0]
                            << " reserved=" << s[1]
                            << " init=" << s[3];
            context->Unmap(subject_stage.Get(), 0);
          }
        }
      }
    }

    // Called once per submitted source frame. Reports achieved inference throughput and busy
    // drops without altering cadence; production always attempts the newest available frame.
    void update_throughput_stats() {
      auto now = std::chrono::steady_clock::now();
      if (last_call_time.time_since_epoch().count() != 0) {
        float dt = std::chrono::duration<float>(now - last_call_time).count();
        if (dt > 1e-4f && dt < 0.5f) {  // ignore first call and long stalls (paused/occluded)
          float inst = 1.0f / dt;
          measured_fps = (measured_fps <= 0.0f) ? inst : (measured_fps * 0.95f + inst * 0.05f);
        }
      }
      last_call_time = now;

      // A five-second window is responsive enough for headset tuning without flooding the log.
      if (throughput_stats_start.time_since_epoch().count() == 0) {
        throughput_stats_start = now;
      } else {
        float stats_seconds = std::chrono::duration<float>(now - throughput_stats_start).count();
        if (stats_seconds >= 5.0f) {
          float calls = (float) std::max(1u, throughput_stats_calls);
          if (sbs_perf::enabled()) {
            BOOST_LOG(info) << "Depth throughput: source ~" << (int) (measured_fps + 0.5f)
                            << "fps, completed ~" << (int) (throughput_stats_completions / stats_seconds + 0.5f)
                            << "fps, enqueued ~" << (int) (throughput_stats_enqueues / stats_seconds + 0.5f)
                            << "fps, busy drops " << (int) (100.0f * throughput_stats_busy_drops / calls + 0.5f)
                            << "% (" << throughput_stats_busy_drops << '/' << throughput_stats_calls << ')';
          }
          throughput_stats_start = now;
          throughput_stats_calls = 0;
          throughput_stats_busy_drops = 0;
          throughput_stats_enqueues = 0;
          throughput_stats_completions = 0;
        }
      }
      throughput_stats_calls++;
    }

    estimate_result estimate(ID3D11ShaderResourceView *input_srv, input_color_space color_space, std::uint64_t frame_id) {
      if (!valid || !input_srv) {
        return {};
      }
      bool completed_frame_valid = false;
      std::uint64_t completed_frame_id = 0;

      auto &cuda = cuda_driver_api::get();
      if (!cuda.is_valid()) {
        BOOST_LOG(error) << "CUDA Driver API is not available.";
        return {};
      }

      if (cuda_ctx) {
        cuda.cuCtxSetCurrent(cuda_ctx);
      }

      update_throughput_stats();

      // Perf benchmark: resolve any completed inference-timing events into samples.
      if (sbs_perf::enabled()) {
        perf_drain(perf_depth);
      }

      // Prevent GPU starvation: if the previous AI frame is still crunching, drop this frame.
      // This prevents an infinite queue of heavy TensorRT workloads from starving the DWM and Edge Browser.
      if (cu_stream && cuda.cuStreamQuery) {
        auto q = cuda.cuStreamQuery(cu_stream);
        if (q == CUDA_ERROR_NOT_READY) {
          // Reuse the last normalized depth and subject state while inference is busy.
          throughput_stats_busy_drops++;
          return make_result();
        }
        if (q != CUDA_SUCCESS && !stream_error_logged) {
          BOOST_LOG(error) << "cuStreamQuery failed: " << q;
          stream_error_logged = true;
        }
        if (q != CUDA_SUCCESS) {
          return make_result();
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
        float aspect_ratio = (float) input_desc.Width / (float) input_desc.Height;
        // Keep the patch-aligned tensor as close as possible to source aspect while respecting
        // the TensorRT profile, configured aspect cap, and native size.
        int max_w = std::min(1008, (int) input_desc.Width);
        int max_h = std::min(1008, (int) input_desc.Height);
        const float fitted_aspect = aspect_ratio >= 1.0f ? std::min(aspect_ratio, max_aspect) : 1.0f / std::min(1.0f / aspect_ratio, max_aspect);
        const auto fitted_dims = aspect_aligned_dims(
          fitted_aspect,
          depth_short_side,
          max_w,
          max_h
        );
        target_w = fitted_dims.first;
        target_h = fitted_dims.second;

        // Threads for the min/max reduction; grid-stride handles any element count.
        int elems = target_w * target_h;
        reduce_groups = (UINT) std::min(64, std::max(1, (elems + 255) / 256));

        BOOST_LOG(info) << "Depth Estimator dynamic resolution set to " << target_w << "x" << target_h;

        if (cuda_in_res) {
          cuda.cuGraphicsUnregisterResource(cuda_in_res);
        }
        if (cuda_out_res) {
          cuda.cuGraphicsUnregisterResource(cuda_out_res);
        }
        cuda_in_res = nullptr;
        cuda_out_res = nullptr;

        D3D11_BUFFER_DESC buf_desc = {};
        buf_desc.Usage = D3D11_USAGE_DEFAULT;
        buf_desc.ByteWidth = target_w * target_h * 3 * sizeof(float);
        buf_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        buf_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        buf_desc.StructureByteStride = sizeof(float);
        bool resources_ok = SUCCEEDED(device->CreateBuffer(&buf_desc, nullptr, &tensor_in_buf)) &&
                            SUCCEEDED(device->CreateUnorderedAccessView(
                              tensor_in_buf.Get(),
                              nullptr,
                              &tensor_in_uav
                            ));

        buf_desc.ByteWidth = target_w * target_h * sizeof(float);
        buf_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateBuffer(&buf_desc, nullptr, &tensor_out_buf)) &&
                       SUCCEEDED(device->CreateShaderResourceView(
                         tensor_out_buf.Get(),
                         nullptr,
                         &tensor_out_srv
                       ));

        D3D11_TEXTURE2D_DESC tex_desc = {};
        tex_desc.Width = target_w;
        tex_desc.Height = target_h;
        tex_desc.MipLevels = 1;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_R32_FLOAT;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;
        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateTexture2D(&tex_desc, nullptr, &depth_tex)) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(depth_tex.Get(), nullptr, &depth_uav)) &&
                       SUCCEEDED(device->CreateShaderResourceView(depth_tex.Get(), nullptr, &depth_srv));

        // Immutable previous-depth snapshot for deterministic neighborhood tests. Reading the
        // in-place EMA output while adjacent threads write it would make dilation race-dependent.
        auto previous_desc = tex_desc;
        previous_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateTexture2D(&previous_desc, nullptr, &depth_previous_tex)) &&
                       SUCCEEDED(device->CreateShaderResourceView(depth_previous_tex.Get(), nullptr, &depth_previous_srv));

        auto mask_desc = tex_desc;
        mask_desc.Format = DXGI_FORMAT_R32_UINT;
        mask_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateTexture2D(&mask_desc, nullptr, &ema_motion_mask_tex)) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(ema_motion_mask_tex.Get(), nullptr, &ema_motion_mask_uav)) &&
                       SUCCEEDED(device->CreateShaderResourceView(ema_motion_mask_tex.Get(), nullptr, &ema_motion_mask_srv));

        if (exact_plane_lock) {
          auto create_plane_texture = [&](Microsoft::WRL::ComPtr<ID3D11Texture2D> &tex, Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> &uav, Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> &srv) {
            return SUCCEEDED(device->CreateTexture2D(&tex_desc, nullptr, &tex)) &&
                   SUCCEEDED(device->CreateUnorderedAccessView(tex.Get(), nullptr, &uav)) &&
                   SUCCEEDED(device->CreateShaderResourceView(tex.Get(), nullptr, &srv));
          };
          bool ok = create_plane_texture(plane_band_tex, plane_band_uav, plane_band_srv) &&
                    create_plane_texture(plane_work_a_tex, plane_work_a_uav, plane_work_a_srv) &&
                    create_plane_texture(plane_work_b_tex, plane_work_b_uav, plane_work_b_srv);

          const UINT groups_x = ((UINT) target_w + 15u) / 16u;
          const UINT groups_y = ((UINT) target_h + 15u) / 16u;
          plane_group_count = groups_x * groups_y;
          D3D11_BUFFER_DESC pbd = {};
          pbd.Usage = D3D11_USAGE_DEFAULT;
          pbd.ByteWidth = plane_group_count * sizeof(float) * 2u;
          pbd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
          pbd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
          pbd.StructureByteStride = sizeof(float) * 2u;
          ok = ok && SUCCEEDED(device->CreateBuffer(&pbd, nullptr, &plane_group_buf)) &&
               SUCCEEDED(device->CreateUnorderedAccessView(plane_group_buf.Get(), nullptr, &plane_group_uav));
          if (!ok) {
            BOOST_LOG(warning) << "Bestv2 subject-plane-lock resource creation failed; using the depth-band fallback.";
            exact_plane_lock = false;
            plane_band_tex.Reset();
            plane_band_uav.Reset();
            plane_band_srv.Reset();
            plane_work_a_tex.Reset();
            plane_work_a_uav.Reset();
            plane_work_a_srv.Reset();
            plane_work_b_tex.Reset();
            plane_work_b_uav.Reset();
            plane_work_b_srv.Reset();
            plane_group_buf.Reset();
            plane_group_uav.Reset();
          }
        }

        if (!resources_ok) {
          BOOST_LOG(error) << "Depth estimator D3D11 resource creation failed; retrying on a later frame.";
          target_w = target_h = 0;
          return {};
        }

        // Clear depth so the range->pixel EMA initializes from a known value.
        const float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        context->ClearUnorderedAccessViewFloat(depth_uav.Get(), clear_color);
        const UINT clear_uint[4] = {0u, 0u, 0u, 0u};
        context->ClearUnorderedAccessViewUint(ema_motion_mask_uav.Get(), clear_uint);
        depth_history_valid = false;
        if (plane_band_uav) {
          context->ClearUnorderedAccessViewFloat(plane_band_uav.Get(), clear_color);
        }
        if (plane_work_a_uav) {
          context->ClearUnorderedAccessViewFloat(plane_work_a_uav.Get(), clear_color);
        }
        if (plane_work_b_uav) {
          context->ClearUnorderedAccessViewFloat(plane_work_b_uav.Get(), clear_color);
        }

        auto res1 = cuda.cuGraphicsD3D11RegisterResource(&cuda_in_res, tensor_in_buf.Get(), 0);
        auto res2 = cuda.cuGraphicsD3D11RegisterResource(&cuda_out_res, tensor_out_buf.Get(), 0);
        if (res1 != 0 || res2 != 0) {
          BOOST_LOG(error) << "cuGraphicsD3D11RegisterResource failed: " << res1 << ", " << res2;
          if (cuda_in_res) {
            cuda.cuGraphicsUnregisterResource(cuda_in_res);
          }
          if (cuda_out_res) {
            cuda.cuGraphicsUnregisterResource(cuda_out_res);
          }
          cuda_in_res = nullptr;
          cuda_out_res = nullptr;
          target_w = target_h = 0;
          return {};
        }
      }

      // Shared constants for buffer_to_tex_cs, the min/max passes and rgb_to_nchw_cs.
      // Session-constant, so the buffer is built once (immutable), not mapped per frame.
      ensure_cbuffers(color_space);
      if (!cbuffer) {
        return {};
      }

      auto *d3d_timer = begin_d3d_perf(has_previous_frame, true);

      // tensor_out_buf holds the finished raw disparity from the previous asynchronous submit
      // (fully unmapped from CUDA), so consuming it here never blocks the encode thread. The
      // caller uses completed_frame_id to select the color slot that produced this exact result.
      if (has_previous_frame) {
        normalize_depth_output();
        completed_frame_id = pending_frame_id;
        completed_frame_valid = true;
        has_previous_frame = false;
        throughput_stats_completions++;
      }
      mark_d3d_post_end(d3d_timer);

      // 1. D3D11 Compute Shader: Resize & Normalize to NCHW FP32 Buffer (for CURRENT frame)
      mark_d3d_pre_start(d3d_timer);
      context->CSSetShader(rgb_to_nchw_cs.Get(), nullptr, 0);
      context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
      context->CSSetShaderResources(0, 1, &input_srv);
      context->CSSetUnorderedAccessViews(0, 1, tensor_in_uav.GetAddressOf(), nullptr);
      context->CSSetSamplers(0, 1, linear_sampler.GetAddressOf());

      context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

      ID3D11UnorderedAccessView *null_uav = nullptr;
      ID3D11ShaderResourceView *null_srv = nullptr;
      context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
      context->CSSetShaderResources(0, 1, &null_srv);
      end_d3d_perf(d3d_timer);
      // No explicit Flush: cuGraphicsMapResources() below already guarantees the
      // preceding D3D11 compute work completes before the CUDA stream reads the buffer.
      // Force-flushing every frame only prevents the driver from interleaving other GPU
      // consumers (DWM / Edge / the Widgets panel), which starves them and can trigger a TDR.

      // 2. CUDA Execution (for CURRENT frame)
      CUgraphicsResource resources[2] = {cuda_in_res, cuda_out_res};
      auto map_res = cuda.cuGraphicsMapResources(2, resources, cu_stream);
      if (map_res != 0) {
        BOOST_LOG(error) << "cuGraphicsMapResources failed: " << map_res;
        return make_result(completed_frame_valid, completed_frame_id);
      }

      void *d_in = nullptr;
      void *d_out = nullptr;
      auto in_ptr_res = cuda.cuGraphicsResourceGetMappedPointer(
        (CUdeviceptr *) &d_in,
        nullptr,
        cuda_in_res
      );
      auto out_ptr_res = cuda.cuGraphicsResourceGetMappedPointer(
        (CUdeviceptr *) &d_out,
        nullptr,
        cuda_out_res
      );

      bool enqueued = false;
      if (in_ptr_res != CUDA_SUCCESS || out_ptr_res != CUDA_SUCCESS || !d_in || !d_out) {
        BOOST_LOG(error) << "Failed to get mapped pointer for TensorRT: "
                         << in_ptr_res << ", " << out_ptr_res;
      } else {
        nvinfer1::Dims in_dims = make_input_dims(target_h, target_w);
        bool bindings_ok = exec_context->setInputShape("pixel_values", in_dims);
        if (!bindings_ok) {
          BOOST_LOG(error) << "TensorRT setInputShape failed for " << target_w << "x" << target_h
                           << " (outside the engine's optimization profile?)";
        }
        bindings_ok = bindings_ok &&
                      exec_context->setTensorAddress("pixel_values", (void *) d_in) &&
                      exec_context->setTensorAddress("predicted_depth", (void *) d_out);
        if (bindings_ok) {
          // Serialize TensorRT async enqueue to avoid driver-level concurrent execution faults
          std::lock_guard<std::mutex> lock(*trt_mutex);
          int perf_slot = perf_begin(perf_depth, cu_stream);
          enqueued = enqueue_inference(
            (CUdeviceptr) d_in,
            (CUdeviceptr) d_out,
            cuda
          );
          if (!enqueued) {
            if (!stream_error_logged) {
              BOOST_LOG(error) << "TensorRT enqueueV3 failed; retaining the last valid depth.";
              stream_error_logged = true;
            }
          }
          perf_end(perf_depth, perf_slot, cu_stream);
        }
      }

      auto unmap_res = cuda.cuGraphicsUnmapResources(2, resources, cu_stream);
      if (unmap_res != CUDA_SUCCESS) {
        BOOST_LOG(error) << "cuGraphicsUnmapResources failed: " << unmap_res;
        enqueued = false;
      }

      has_previous_frame = enqueued;
      if (enqueued) {
        pending_frame_id = frame_id;
        throughput_stats_enqueues++;
      }

      return make_result(completed_frame_valid, completed_frame_id, enqueued, enqueued ? frame_id : 0);
    }
  };

  video_depth_estimator::video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device, Microsoft::WRL::ComPtr<ID3D11DeviceContext> context, const std::filesystem::path &assets_dir, const config::video_t::sbs_t &cfg, const config::depth_model_info &model):
      pimpl(std::make_unique<impl>(device, context, assets_dir, cfg, model)) {}

  video_depth_estimator::~video_depth_estimator() = default;

  bool video_depth_estimator::is_valid() const {
    return pimpl && pimpl->valid;
  }

  estimate_result video_depth_estimator::estimate_depth(
    ID3D11ShaderResourceView *input_srv,
    input_color_space color_space,
    std::uint64_t frame_id
  ) {
    return pimpl->estimate(input_srv, color_space, frame_id);
  }

  estimate_result video_depth_estimator::finish_pending_depth_for_evaluation(input_color_space color_space) {
    return pimpl->finish_pending(color_space);
  }
}  // namespace models
