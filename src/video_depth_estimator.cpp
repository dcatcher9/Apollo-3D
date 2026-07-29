#include "video_depth_estimator.h"

#include "cuda_driver_api.h"
#include "generated/sbs_adaptive_state_contract.h"
#include "generated/sbs_scene_controller_contract.h"
#include "logging.h"
#include "model_manager.h"
#include "platform/windows/misc.h"
#include "platform/windows/utils.h"
#include "sbs_perf.h"
#include "sbs_roi_shape_request_gpu.h"
#include "sbs_roi_shape_transition.h"
#include "sbs_scene_controller_gpu.h"
#include "utility.h"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <d3dcompiler.h>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
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
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        BOOST_LOG(error) << "TensorRT: " << msg;
        break;
      case Severity::kWARNING:
        BOOST_LOG(warning) << "TensorRT: " << msg;
        break;
      case Severity::kINFO:
        BOOST_LOG(debug) << "TensorRT: " << msg;
        break;
      case Severity::kVERBOSE:
        BOOST_LOG(verbose) << "TensorRT: " << msg;
        break;
    }
  }
};

static Logger gLogger;

static std::mutex g_model_prepare_status_mutex;
static std::map<std::string, models::engine_build_status> g_model_prepare_status;
static std::mutex g_depth_shader_cache_mutex;

struct depth_shader_cache_entry {
  std::filesystem::file_time_type modified;
  std::shared_ptr<const std::vector<std::uint8_t>> bytecode;
};

static std::map<std::filesystem::path, depth_shader_cache_entry> g_depth_shader_cache;

static void set_model_prepare_status(const std::string &engine_name, models::engine_build_status status) {
  std::lock_guard<std::mutex> lock(g_model_prepare_status_mutex);
  g_model_prepare_status[engine_name] = status;
}

static std::shared_ptr<const std::vector<std::uint8_t>> depth_shader_bytecode(
  const std::filesystem::path &path
) {
  // D3D shader bytecode is device-independent. Cache blobs across device recreation, but compare
  // source mtimes so a newly created estimator sees an edit without restarting. Never hold the
  // global map lock across D3DCompileFromFile: unrelated estimators may initialize concurrently.
  std::error_code ec;
  const auto modified = std::filesystem::last_write_time(path, ec);
  {
    std::lock_guard<std::mutex> lock(g_depth_shader_cache_mutex);
    if (auto it = g_depth_shader_cache.find(path); it != g_depth_shader_cache.end() && !ec && it->second.modified == modified) {
      return it->second.bytecode;
    }
  }

  Microsoft::WRL::ComPtr<ID3DBlob> blob;
  Microsoft::WRL::ComPtr<ID3DBlob> err;
  constexpr DWORD flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
  if (FAILED(D3DCompileFromFile(path.wstring().c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "cs_5_0", flags, 0, &blob, &err))) {
    if (err) {
      BOOST_LOG(error) << "Shader compile error (" << path << "): " << (char *) err->GetBufferPointer();
    }
    return {};
  }
  auto *begin = static_cast<const std::uint8_t *>(blob->GetBufferPointer());
  auto bytecode = std::make_shared<const std::vector<std::uint8_t>>(
    begin,
    begin + blob->GetBufferSize()
  );
  if (!ec) {
    std::lock_guard<std::mutex> lock(g_depth_shader_cache_mutex);
    g_depth_shader_cache.insert_or_assign(path, depth_shader_cache_entry {modified, bytecode});
  }
  return bytecode;
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
// One active stream normally needs one context; four permits bounded encoder-transition and
// failed-warmup recovery without letting repeated rebuilds consume VRAM without bound.
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

// TensorRT plans are tied to the TensorRT ABI and, unless hardware-compatibility mode is
// explicitly enabled, the GPU model on which tactics were selected. Keep those identities in the
// disk filename so another adapter (or a later TensorRT upgrade) never consumes an incompatible
// serialized plan. The stable name hash avoids filesystem-hostile adapter characters.
static std::string engine_compatibility_tag(cuda_driver_api &cuda, CUdevice device) {
  int sm_major = -1;
  int sm_minor = -1;
  if (cuda.cuDeviceGetAttribute) {
    cuda.cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuda.cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  }

  std::array<char, 256> device_name {};
  if (!cuda.cuDeviceGetName || cuda.cuDeviceGetName(device_name.data(), (int) device_name.size(), device) != CUDA_SUCCESS) {
    std::snprintf(device_name.data(), device_name.size(), "cuda-device-%d", (int) device);
  }
  std::uint64_t name_hash = 1469598103934665603ULL;
  for (const unsigned char ch : std::string_view(device_name.data())) {
    name_hash ^= ch;
    name_hash *= 1099511628211ULL;
  }

  std::ostringstream tag;
  tag << "trt" << NV_TENSORRT_MAJOR << '_' << NV_TENSORRT_MINOR << '_' << NV_TENSORRT_PATCH
      << "-sm" << sm_major << sm_minor << "-gpu" << std::hex << name_hash;
  return tag.str();
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
  // Usable contexts include both checked-out and pooled instances. Failed warmup contexts cannot
  // be destroyed across the MinGW/MSVC ABI boundary, so account for them separately: they must
  // never re-enter the pool, while the combined count still enforces the physical VRAM cap.
  std::size_t context_count = 0;
  std::size_t warmed_context_count = 0;
  std::size_t quarantined_context_count = 0;
  bool io_validated = false;
  bool io_compatible = false;
};

static std::map<std::string, engine_slot> g_engines;  // guarded by g_trt_mutex

static std::size_t allocated_context_count(const engine_slot &slot) {
  return slot.context_count + slot.quarantined_context_count;
}

// The object is deliberately leaked because destroying TensorRT interfaces across this compiler
// boundary corrupts the heap. Removing it from usable accounting prevents a later session from
// treating a failed lazy-load/binding operation as a warmed context; quarantined accounting keeps
// repeated failures bounded by kMaxContextsPerEngine.
static void quarantine_execution_context_locked(
  const std::string &engine_key,
  nvinfer1::IExecutionContext *&context,
  const bool was_warmed = false
) {
  if (!context) {
    return;
  }
  auto &slot = g_engines[engine_key];
  const auto accounting =
    models::sbs_trt_context_accounting_after_quarantine(
      {
        slot.context_count,
        slot.warmed_context_count,
        slot.quarantined_context_count,
      },
      was_warmed
    );
  slot.context_count = accounting.usable;
  slot.warmed_context_count = accounting.warmed;
  slot.quarantined_context_count = accounting.quarantined;
  context = nullptr;
  g_trt_context_available.notify_all();
}

static void mark_execution_context_warmed_locked(const std::string &engine_key) {
  auto &slot = g_engines[engine_key];
  ++slot.warmed_context_count;
}

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

// Ensure the shared runtime exists, deserialize the compatible engine into its global slot if not
// already resident, and hand back a spare pooled execution context if one is available. The CALLER
// must hold g_trt_mutex. Context CREATION is deliberately left to the caller OUTSIDE the lock:
// createExecutionContext() allocates ~1.3 GB of scratch and takes seconds, and holding the lock
// across it would delay pooled-context returns and subsequent pipeline acquisition.
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
    if (!file) {
      BOOST_LOG(error) << "Could not open TensorRT engine " << engine_path << '.';
      return nullptr;
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (blob.empty()) {
      BOOST_LOG(error) << "TensorRT engine cache is empty: " << engine_path << '.';
      return nullptr;
    }
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
    if (!name) {
      BOOST_LOG(error) << "TensorRT returned a null I/O tensor name; rejecting the engine.";
      continue;
    }
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
  if (!exec_context || !cuda_ctx || !cuda.is_valid()) {
    return false;
  }
  if (cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
    return false;
  }

  CUstream stream = nullptr;
  if (cuda.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS || !stream) {
    return false;
  }

  constexpr int h = models::depth_engine_opt_height;
  constexpr int w = models::depth_engine_opt_width;
  const size_t in_elems = (size_t) 3 * h * w;
  const size_t out_elems = (size_t) h * w;
  CUdeviceptr d_in = 0;
  CUdeviceptr d_out = 0;
  const bool input_allocated =
    cuda.cuMemAlloc(
      &d_in,
      in_elems * sizeof(float)
    ) == CUDA_SUCCESS;
  const bool output_allocated =
    input_allocated &&
    cuda.cuMemAlloc(
      &d_out,
      out_elems * sizeof(float)
    ) == CUDA_SUCCESS;
  const auto input_dims = make_input_dims(h, w);
  const bool bound =
    output_allocated &&
    exec_context->setInputShape("pixel_values", input_dims) &&
    exec_context->setTensorAddress("pixel_values", (void *) d_in) &&
    exec_context->setTensorAddress("predicted_depth", (void *) d_out);
  bool enqueued = false;
  if (bound) {
    std::lock_guard<std::mutex> lock(g_trt_mutex);
    enqueued = exec_context->enqueueV3(stream);
  }

  // Re-select the owning context and prove the stream idle before issuing any cleanup call. A
  // failed enqueue may still have submitted partial work. If selection or synchronization fails,
  // intentionally retain the bounded warmup allocations/stream until process exit; the caller
  // quarantines this TensorRT context, and speculative cleanup would be less safe than the leak.
  const bool context_selected =
    cuda.cuCtxSetCurrent(cuda_ctx) == CUDA_SUCCESS;
  const bool synchronized =
    context_selected &&
    cuda.cuStreamSynchronize(stream) == CUDA_SUCCESS;
  if (
    models::sbs_cuda_resource_cleanup_policy(
      context_selected,
      synchronized
    ) != models::sbs_cuda_resource_cleanup_disposition::release
  ) {
    BOOST_LOG(error)
      << "Depth model startup warmup could not prove its CUDA stream idle; "
         "retaining terminal warmup handles until process exit.";
    return false;
  }

  bool cleanup_succeeded = true;
  if (d_out != 0) {
    cleanup_succeeded =
      cuda.cuMemFree(d_out) == CUDA_SUCCESS &&
      cleanup_succeeded;
  }
  if (d_in != 0) {
    cleanup_succeeded =
      cuda.cuMemFree(d_in) == CUDA_SUCCESS &&
      cleanup_succeeded;
  }
  cleanup_succeeded =
    cuda.cuStreamDestroy(stream) == CUDA_SUCCESS &&
    cleanup_succeeded;
  const bool succeeded =
    input_allocated && output_allocated && bound &&
    enqueued && synchronized && cleanup_succeeded;
  BOOST_LOG(info) << "Depth model startup warmup complete (" << w << 'x' << h
                  << (succeeded ? ")." : "); execution or cleanup failed.");
  return succeeded;
}

namespace models {

  struct engine_artifact {
    std::string name;
    std::string source_sha256;
    std::filesystem::path source_path;
    std::filesystem::path engine_path;
  };

  static std::mutex g_active_engine_manifest_mutex;

  static bool publish_active_engine_manifest(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model,
    const engine_artifact &artifact
  ) {
    if (artifact.name.empty() || artifact.source_sha256.empty()) {
      return false;
    }

    const auto path = assets_dir / (model.name + ".active-engine.json");
    auto temporary_path = path;
    temporary_path += ".tmp";
    const nlohmann::json manifest {
      {"schema", 1},
      {"model", model.name},
      {"engine", artifact.name},
      {"onnx_sha256", artifact.source_sha256},
    };

    std::lock_guard lock(g_active_engine_manifest_mutex);
    {
      std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
      if (!output) {
        return false;
      }
      output << manifest.dump(2) << '\n';
      output.flush();
      if (!output) {
        output.close();
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        return false;
      }
    }
    if (!MoveFileExW(
          temporary_path.c_str(),
          path.c_str(),
          MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
        )) {
      std::error_code ignored;
      std::filesystem::remove(temporary_path, ignored);
      return false;
    }
    BOOST_LOG(info) << "Published active TensorRT engine manifest " << path.filename() << '.';
    return true;
  }

  // Resolve the exact ONNX+TensorRT+GPU identity and ensure the corresponding engine exists.
  // The full source hash is intentional: model names and URLs are user-overridable, so neither is
  // a safe cache identity when a local ONNX is replaced in-place.
  static bool ensure_tensorrt_engine_for_device(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model,
    cuda_driver_api &cuda,
    CUdevice cuda_device,
    engine_artifact &artifact
  ) {
    static std::mutex compile_mutex;
    std::lock_guard<std::mutex> lock(compile_mutex);

    artifact.source_path = ensure_onnx_available(assets_dir, model.name, model.url);
    if (artifact.source_path.empty()) {
      BOOST_LOG(warning) << "ONNX source not found. TensorRT compilation aborted.";
      return false;
    }
    artifact.source_sha256 = file_sha256_hex(artifact.source_path);
    if (artifact.source_sha256.empty()) {
      BOOST_LOG(error) << "Could not hash depth-model source " << artifact.source_path << '.';
      return false;
    }
    artifact.name = engine_filename(
      model,
      engine_compatibility_tag(cuda, cuda_device) + "-onnx" + artifact.source_sha256
    );
    artifact.engine_path = assets_dir / artifact.name;

    std::error_code existing_ec;
    if (std::filesystem::is_regular_file(artifact.engine_path, existing_ec)) {
      BOOST_LOG(info) << "TensorRT engine cache hit: " << artifact.engine_path.filename();
      return true;
    }

    BOOST_LOG(info) << "Building TensorRT engine from ONNX... This will take a few minutes.";

    if (CUcontext ctx = primary_context(cuda, cuda_device)) {
      if (cuda.cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
        BOOST_LOG(error) << "Failed to select the configured CUDA device for TensorRT engine compilation.";
        return false;
      }
    } else {
      BOOST_LOG(error) << "Failed to retain the configured CUDA device for TensorRT engine compilation.";
      return false;
    }

    initLibNvInferPlugins(&gLogger, "");
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
      BOOST_LOG(error) << "TensorRT failed to create an engine builder.";
      return false;
    }
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!network || !config) {
      BOOST_LOG(error) << "TensorRT failed to create the network or builder configuration.";
      return false;
    }

    // Set memory limit to 4GB
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4ULL << 30);
    // Level 5 makes TensorRT compare generated kernels against its static tactics. Keep this in
    // the recipe-specific engine contract: changing the level must never silently reuse a plan
    // selected under the default level 3 search.
    config->setBuilderOptimizationLevel(depth_engine_builder_level);
    BOOST_LOG(info) << "TensorRT builder optimization level " << depth_engine_builder_level << '.';

    auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
      BOOST_LOG(error) << "TensorRT failed to create the ONNX parser.";
      return false;
    }
    if (!parser->parseFromFile(artifact.source_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      BOOST_LOG(error) << "Failed to parse ONNX file.";
      return false;
    }
    auto *input = network->getNbInputs() == 1 ? network->getInput(0) : nullptr;
    if (!input || !input->getName() || std::string_view(input->getName()) != "pixel_values") {
      BOOST_LOG(error) << "Unsupported depth model input contract; expected one 'pixel_values' tensor.";
      return false;
    }

    // DA-V2 contract: input "pixel_values" [1,3,H,W], output "predicted_depth".
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
      BOOST_LOG(error) << "TensorRT failed to create the depth optimization profile.";
      return false;
    }
    auto dims_for = [&](int h, int w) {
      return make_input_dims(h, w);
    };
    const bool profile_ok =
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims_for(14, 14)) &&
      profile->setDimensions(
        input->getName(),
        nvinfer1::OptProfileSelector::kOPT,
        dims_for(depth_engine_opt_height, depth_engine_opt_width)
      ) &&
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims_for(depth_engine_max_dim, depth_engine_max_dim));
    if (!profile_ok || config->addOptimizationProfile(profile) < 0) {
      BOOST_LOG(error) << "TensorRT rejected the depth optimization profile.";
      return false;
    }

    std::vector<nvinfer1::ITensor *> to_unmark;
    bool found_depth_output = false;
    for (int i = 0; i < network->getNbOutputs(); i++) {
      auto *tensor = network->getOutput(i);
      if (!tensor || !tensor->getName()) {
        BOOST_LOG(error) << "TensorRT returned a null output tensor while building the depth engine.";
        return false;
      }
      if (std::string_view(tensor->getName()) == "predicted_depth") {
        found_depth_output = true;
      } else {
        to_unmark.push_back(tensor);
      }
    }
    if (!found_depth_output) {
      BOOST_LOG(error) << "Unsupported depth model output contract; missing 'predicted_depth'.";
      return false;
    }
    for (auto *tensor : to_unmark) {
      BOOST_LOG(info) << "Depth engine: pruning unsupported output '" << tensor->getName() << "'.";
      network->unmarkOutput(*tensor);
    }

    auto serializedModel = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (serializedModel) {
      // Save under the recipe-specific engine name so a later recipe change rebuilds
      // rather than silently reusing this engine's (now-wrong) I/O layout.
      auto part_path = artifact.engine_path;
      part_path += ".part";
      std::error_code ec;
      std::filesystem::remove(part_path, ec);
      std::ofstream p(part_path, std::ios::binary | std::ios::trunc);
      if (p) {
        p.write(static_cast<const char *>(serializedModel->data()), serializedModel->size());
        p.close();
        if (p) {
          std::filesystem::rename(part_path, artifact.engine_path, ec);
          if (!ec) {
            BOOST_LOG(info) << "Saved built engine atomically to " << artifact.engine_path;
            return true;
          }
          BOOST_LOG(error) << "Failed to publish built engine " << artifact.engine_path << ": " << ec.message();
        }
      }
      std::filesystem::remove(part_path, ec);
      BOOST_LOG(error) << "Failed to save built engine to " << artifact.engine_path;
    } else {
      BOOST_LOG(error) << "Engine build failed.";
    }
    return false;
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
    const auto status_key = engine_filename(model);
    set_model_prepare_status(status_key, engine_build_status::building);
    auto failed = util::fail_guard([&]() {
      set_model_prepare_status(status_key, engine_build_status::failed);
    });

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
    if (cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
      BOOST_LOG(error) << "Startup depth-model preparation failed: could not select the configured CUDA context.";
      return false;
    }

    engine_artifact artifact;
    if (!ensure_tensorrt_engine_for_device(assets_dir, model, cuda, cuda_device, artifact)) {
      return false;
    }
    auto engine_path = artifact.engine_path;
    auto engine_key = std::to_string(cuda_device) + ":" + artifact.name;

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *exec_context = nullptr;
    bool pooled = false;
    bool create_context = false;
    bool resident_warmed_context = false;
    {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      engine = acquire_engine_locked(engine_key, engine_path, exec_context, pooled);
    }

    // An existing file is not proof of a usable TensorRT plan: interrupted legacy writes, a
    // runtime upgrade, or copied assets can all leave a regular file that fails deserialization.
    // Remove only a slot with no resident engine/contexts, rebuild atomically from ONNX, and retry
    // once. This turns the former permanent flat-SBS state into a self-healing startup path.
    if (!engine) {
      {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        auto found = g_engines.find(engine_key);
        if (found != g_engines.end() && !found->second.engine && allocated_context_count(found->second) == 0 && found->second.context_pool.empty()) {
          g_engines.erase(found);
        }
      }
      std::error_code ec;
      std::filesystem::remove(engine_path, ec);
      BOOST_LOG(warning) << "Cached TensorRT plan could not be deserialized; rebuilding " << engine_path.filename() << '.';
      if (!ensure_tensorrt_engine_for_device(assets_dir, model, cuda, cuda_device, artifact)) {
        return false;
      }
      engine_path = artifact.engine_path;
      engine_key = std::to_string(cuda_device) + ":" + artifact.name;
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      engine = acquire_engine_locked(engine_key, engine_path, exec_context, pooled);
    }

    {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      auto &slot = g_engines[engine_key];
      if (!validate_engine_io_locked(engine, slot)) {
        if (exec_context) {
          slot.context_pool.push_back(exec_context);
          g_trt_context_available.notify_all();
        }
        return false;
      }
      if (!exec_context) {
        if (allocated_context_count(slot) >= kMaxContextsPerEngine) {
          // A live session may already have populated the engine before startup preparation
          // finished. Only a context that actually completed warmup can establish readiness;
          // quarantined or still-constructing contexts are not evidence that the plan is usable.
          if (slot.warmed_context_count == 0) {
            BOOST_LOG(error) << "TensorRT context capacity contains no successfully warmed context.";
            return false;
          }
          resident_warmed_context = true;
        } else {
          ++slot.context_count;
          create_context = true;
        }
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
        // This context cannot be destroyed across the MinGW/MSVC ABI boundary, but it must never
        // enter the reusable pool: pooled contexts are assumed warmed and skip this operation.
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        quarantine_execution_context_locked(engine_key, exec_context);
        BOOST_LOG(error) << "Startup depth-model context warmup failed.";
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        mark_execution_context_warmed_locked(engine_key);
      }
    }

    if (exec_context) {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      g_engines[engine_key].context_pool.push_back(exec_context);
      g_trt_context_available.notify_all();
    }
    if (!exec_context && !resident_warmed_context) {
      BOOST_LOG(error) << "Startup depth-model preparation produced no reusable warmed context.";
      return false;
    }
    if (!publish_active_engine_manifest(assets_dir, model, artifact)) {
      // Manifest publication is an evaluator/preflight contract, not a reason to discard a model
      // that is already resident and proven usable for production streaming.
      BOOST_LOG(error) << "Could not publish the active TensorRT engine manifest for model '"
                       << model.name << "'.";
    }
    BOOST_LOG(info) << "Startup depth model '" << model.name << "' is resident and ready.";
    set_model_prepare_status(status_key, engine_build_status::ready);
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
    float ema_edge_strength;
    int depth_short_side;  // depth map short-side resolution (clamped to native short side)
    float max_aspect;  // aspect cap for short-side mode
    float minmax_alpha;  // temporal EMA blend for the normalized min/max
    bool cuda_graph_enabled;
    const bool diagnostics_enabled;
    CUgraph inference_graph = nullptr;
    CUgraphExec inference_graph_exec = nullptr;
    CUdeviceptr graph_input = 0;
    CUdeviceptr graph_output = 0;
    int graph_width = 0;
    int graph_height = 0;
    int configured_input_width = 0;
    int configured_input_height = 0;
    bool graph_signature_warmed = false;
    bool graph_capture_failed = false;
    bool valid = false;  // all mandatory engine, shader, and session resources are ready
    float subject_recenter;  // recenter strength consumed by depth_subject_resolve_cs
    bool subject_stretch;  // apply the shape_depth_for_pop 5/95 disparity stretch
    bool adaptive_pop;
    float adaptive_pop_max_ratio;
    float zero_plane_mode;  // 1 subject, 2 median depth, 3 far/mid-background
    // Active rules remain deliberately unexposed. Keeping the complete owned-transform path behind
    // one runtime gate lets WARP/tests exercise it without allowing `shadow_rules` to affect output.
    bool active_roi_authority = false;
    std::unique_ptr<sbs_scene_controller_gpu> scene_controller;
    std::unique_ptr<sbs_roi_shape_request_gpu> roi_shape_request_gpu;
    std::optional<sbs_roi_shape_request> newest_roi_shape_request;

    // Subscription-gated, nonblocking telemetry readback. Resources are created lazily only after
    // a client enables the protocol, then a three-slot staging/query ring absorbs GPU latency
    // without ever flushing or waiting on the encode thread.
    static constexpr std::size_t telemetry_state_float_count =
      sbs_adaptive_state::word_count;

    struct telemetry_readback_slot {
      Microsoft::WRL::ComPtr<ID3D11Buffer> staging;
      Microsoft::WRL::ComPtr<ID3D11Query> completion;
      bool pending = false;
      std::uint64_t sampled_frame_id = 0;
    };

    std::array<telemetry_readback_slot, 3> telemetry_readback_slots;
    std::size_t telemetry_readback_next = 0;
    bool telemetry_readback_ready = false;
    bool telemetry_readback_init_failed = false;

    // Throughput telemetry for the permanent stream-cadence matched-frame pipeline.
    float measured_fps = 0.0f;
    std::chrono::steady_clock::time_point last_call_time {};
    std::chrono::steady_clock::time_point throughput_stats_start {};
    unsigned throughput_stats_calls = 0;
    unsigned throughput_stats_busy_drops = 0;
    unsigned throughput_stats_enqueues = 0;
    unsigned throughput_stats_completions = 0;

    // GPU-stream timing of the async TensorRT enqueues (diagnostics only).
    // A small ring of CUDA event pairs per engine lets several inferences be in flight; the
    // elapsed time is resolved lazily once the stop event completes and pushed to sbs_perf.
    // All CUDA calls here run on the estimator thread with cuda_ctx current, like the rest
    // of estimate(); no-ops entirely when diagnostics are off.
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
      Microsoft::WRL::ComPtr<ID3D11Query> scene_rules_start;
      Microsoft::WRL::ComPtr<ID3D11Query> scene_rules_end;
      Microsoft::WRL::ComPtr<ID3D11Query> scene_prepare_start;
      Microsoft::WRL::ComPtr<ID3D11Query> scene_prepare_end;
      bool pending = false;
      bool has_post = false;
      bool has_pre = false;
      bool has_scene_rules = false;
      bool has_scene_prepare = false;
      bool scene_rules_issued = false;
      bool scene_prepare_issued = false;
      bool pre_ended = false;
      std::uint64_t perf_generation = 0;
    };

    static constexpr std::size_t d3d_perf_ring_size = 16;
    std::array<d3d_perf_slot, d3d_perf_ring_size> d3d_perf_slots;
    std::size_t d3d_perf_next = 0;
    bool d3d_perf_ready = false;

    void initialize_d3d_perf() {
      if (!diagnostics_enabled) {
        return;
      }
      for (auto &slot : d3d_perf_slots) {
        D3D11_QUERY_DESC desc {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
        if (FAILED(device->CreateQuery(&desc, &slot.disjoint))) {
          BOOST_LOG(warning) << "Depth D3D11 timing unavailable: could not create disjoint query.";
          return;
        }
        desc.Query = D3D11_QUERY_TIMESTAMP;
        if (
          FAILED(device->CreateQuery(&desc, &slot.post_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.post_end)) ||
          FAILED(device->CreateQuery(&desc, &slot.pre_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.pre_end)) ||
          FAILED(device->CreateQuery(&desc, &slot.scene_rules_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.scene_rules_end)) ||
          FAILED(device->CreateQuery(&desc, &slot.scene_prepare_start)) ||
          FAILED(device->CreateQuery(&desc, &slot.scene_prepare_end))
        ) {
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
        UINT64 scene_rules_start = 0;
        UINT64 scene_rules_end = 0;
        UINT64 scene_prepare_start = 0;
        UINT64 scene_prepare_end = 0;
        const auto post_start_status = context->GetData(slot.post_start.Get(), &post_start, sizeof(post_start), 0);
        const auto post_end_status = context->GetData(slot.post_end.Get(), &post_end, sizeof(post_end), 0);
        const auto pre_start_status = context->GetData(slot.pre_start.Get(), &pre_start, sizeof(pre_start), 0);
        const auto pre_end_status = context->GetData(slot.pre_end.Get(), &pre_end, sizeof(pre_end), 0);
        const auto scene_rules_start_status = context->GetData(
          slot.scene_rules_start.Get(),
          &scene_rules_start,
          sizeof(scene_rules_start),
          0
        );
        const auto scene_rules_end_status = context->GetData(
          slot.scene_rules_end.Get(),
          &scene_rules_end,
          sizeof(scene_rules_end),
          0
        );
        const auto scene_prepare_start_status = context->GetData(
          slot.scene_prepare_start.Get(),
          &scene_prepare_start,
          sizeof(scene_prepare_start),
          0
        );
        const auto scene_prepare_end_status = context->GetData(
          slot.scene_prepare_end.Get(),
          &scene_prepare_end,
          sizeof(scene_prepare_end),
          0
        );
        if (
          SUCCEEDED(post_start_status) &&
          SUCCEEDED(post_end_status) &&
          SUCCEEDED(pre_start_status) &&
          SUCCEEDED(pre_end_status) &&
          SUCCEEDED(scene_rules_start_status) &&
          SUCCEEDED(scene_rules_end_status) &&
          SUCCEEDED(scene_prepare_start_status) &&
          SUCCEEDED(scene_prepare_end_status) &&
          !timing.Disjoint &&
          timing.Frequency > 0
        ) {
          const double to_ms = 1000.0 / static_cast<double>(timing.Frequency);
          const bool frame_intervals_ordered =
            post_end >= post_start &&
            pre_start >= post_end &&
            pre_end >= pre_start;
          if (slot.has_post && frame_intervals_ordered) {
            sbs_perf::add_sample_ms_if_current(
              "depth_postprocess_gpu",
              static_cast<double>(post_end - post_start) * to_ms,
              slot.perf_generation
            );
          }
          if (slot.has_pre && frame_intervals_ordered) {
            sbs_perf::add_sample_ms_if_current(
              "depth_preprocess_gpu",
              static_cast<double>(pre_end - pre_start) * to_ms,
              slot.perf_generation
            );
          }
          if (
            slot.has_scene_rules &&
            scene_rules_end >= scene_rules_start &&
            scene_rules_start >= post_start &&
            scene_rules_end <= post_end &&
            frame_intervals_ordered
          ) {
            sbs_perf::add_sample_ms_if_current(
              "scene_rules_gpu",
              static_cast<double>(
                scene_rules_end - scene_rules_start
              ) * to_ms,
              slot.perf_generation
            );
          }
          if (
            slot.has_scene_prepare &&
            scene_prepare_end >= scene_prepare_start &&
            scene_prepare_start >= pre_end &&
            frame_intervals_ordered
          ) {
            sbs_perf::add_sample_ms_if_current(
              "scene_prepare_gpu",
              static_cast<double>(
                scene_prepare_end - scene_prepare_start
              ) * to_ms,
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
        slot.has_scene_rules = false;
        slot.has_scene_prepare = false;
        slot.scene_rules_issued = false;
        slot.scene_prepare_issued = false;
        slot.pre_ended = false;
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

    void mark_d3d_pre_end(d3d_perf_slot *slot) {
      if (slot && !slot->pre_ended) {
        context->End(slot->pre_end.Get());
        slot->pre_ended = true;
      }
    }

    void mark_d3d_scene_rules_start(d3d_perf_slot *slot) {
      if (slot) {
        slot->scene_rules_issued = true;
        context->End(slot->scene_rules_start.Get());
      }
    }

    void mark_d3d_scene_rules_end(
      d3d_perf_slot *slot,
      const bool succeeded
    ) {
      if (slot) {
        context->End(slot->scene_rules_end.Get());
        slot->has_scene_rules = succeeded;
      }
    }

    void mark_d3d_scene_prepare_start(d3d_perf_slot *slot) {
      if (slot) {
        slot->scene_prepare_issued = true;
        context->End(slot->scene_prepare_start.Get());
      }
    }

    void mark_d3d_scene_prepare_end(
      d3d_perf_slot *slot,
      const bool succeeded
    ) {
      if (slot) {
        context->End(slot->scene_prepare_end.Get());
        slot->has_scene_prepare = succeeded;
      }
    }

    void end_d3d_perf(d3d_perf_slot *slot) {
      if (!slot) {
        return;
      }
      mark_d3d_pre_end(slot);
      if (!slot->scene_rules_issued) {
        context->End(slot->scene_rules_start.Get());
        context->End(slot->scene_rules_end.Get());
      }
      if (!slot->scene_prepare_issued) {
        context->End(slot->scene_prepare_start.Get());
        context->End(slot->scene_prepare_end.Get());
      }
      context->End(slot->disjoint.Get());
      slot->pending = true;
    }

    bool ensure_telemetry_readback() {
      if (telemetry_readback_ready) {
        return true;
      }
      if (telemetry_readback_init_failed || !subject_buf) {
        return false;
      }

      D3D11_BUFFER_DESC source_desc {};
      subject_buf->GetDesc(&source_desc);
      if (source_desc.ByteWidth < telemetry_state_float_count * sizeof(float)) {
        BOOST_LOG(error) << "Host SBS telemetry source is smaller than its append-only state contract.";
        telemetry_readback_init_failed = true;
        return false;
      }

      D3D11_BUFFER_DESC staging_desc = source_desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0;

      D3D11_QUERY_DESC query_desc {D3D11_QUERY_EVENT, 0};
      for (auto &slot : telemetry_readback_slots) {
        if (FAILED(device->CreateBuffer(&staging_desc, nullptr, &slot.staging)) || FAILED(device->CreateQuery(&query_desc, &slot.completion))) {
          for (auto &created : telemetry_readback_slots) {
            created.staging.Reset();
            created.completion.Reset();
            created.pending = false;
          }
          BOOST_LOG(error) << "Host SBS telemetry staging/query ring initialization failed.";
          // Device/resource pressure can be transient. The render caller rate-limits attempts to
          // the requested telemetry cadence, so leave this retryable without creating a hot loop.
          return false;
        }
      }

      telemetry_readback_ready = true;
      return true;
    }

    static bool decode_telemetry_words(
      const std::array<std::uint32_t, telemetry_state_float_count> &words,
      int depth_width,
      int depth_height,
      std::uint64_t sampled_frame_id,
      depth_telemetry_sample &sample
    ) {
      using sbs_adaptive_state::word_e;
      const auto scalar = [&](const word_e word) {
        return std::bit_cast<float>(words[sbs_adaptive_state::index(word)]);
      };
      for (const auto &field : sbs_adaptive_state::fields) {
        if (
          field.gpu_encoding != sbs_adaptive_state::gpu_encoding_e::uint_bits &&
          !std::isfinite(scalar(field.word))
        ) {
          return false;
        }
      }

      const float scene_age = scalar(word_e::scene_age);
      const float cut_flags = scalar(word_e::cut_flags);
      const float analysis_flags = scalar(word_e::analysis_flags);
      if (
        scene_age < 0.0f ||
        cut_flags < 0.0f ||
        cut_flags > static_cast<float>(sbs_adaptive_state::known_cut_flag_mask) ||
        std::trunc(cut_flags) != cut_flags ||
        analysis_flags < 0.0f ||
        analysis_flags >
          static_cast<float>(sbs_adaptive_state::known_analysis_flag_mask) ||
        std::trunc(analysis_flags) != analysis_flags
      ) {
        return false;
      }

      sample.depth_width = depth_width;
      sample.depth_height = depth_height;
      sample.adaptive_pop_ratio =
        std::max(scalar(word_e::adaptive_pop_ratio), 1.0f);
      sample.edge_fraction = scalar(word_e::latched_edge_fraction);
      sample.change_fraction = scalar(word_e::current_depth_change_fraction);
      sample.valid_depth_fraction = scalar(word_e::valid_depth_fraction);
      sample.effective_range_width = scalar(word_e::effective_raw_range_width);
      sample.current_edge_fraction = scalar(word_e::current_edge_fraction);
      sample.current_zero_anchor_candidate_shift_px =
        scalar(word_e::current_zero_anchor_candidate_shift_px);
      sample.structural_change_fraction =
        scalar(word_e::structural_change_fraction);
      sample.raw_rgb_change_fraction = scalar(word_e::raw_rgb_change_fraction);
      sample.zero_anchor_shift_px = scalar(word_e::zero_anchor_shift_px);
      sample.subject_depth = scalar(word_e::subject_depth_ema);
      sample.scene_age = static_cast<std::uint32_t>(std::min(
        scene_age,
        static_cast<float>(std::numeric_limits<std::uint32_t>::max())
      ));
      sample.cut_flags = static_cast<std::uint32_t>(std::min(
        cut_flags,
        static_cast<float>(std::numeric_limits<std::uint32_t>::max())
      ));
      // SubjectState[4] stores counters as uint bits so they remain exact past float's 24-bit
      // integer range. The shader saturates them one value below UINT_MAX.
      sample.hard_cut_count =
        words[sbs_adaptive_state::index(word_e::hard_cut_count)];
      sample.external_cut_count =
        words[sbs_adaptive_state::index(word_e::external_cut_count)];
      sample.empty_raw_count =
        words[sbs_adaptive_state::index(word_e::empty_raw_count)];
      sample.collapsed_raw_count =
        words[sbs_adaptive_state::index(word_e::collapsed_raw_count)];
      sample.sampled_frame_id = sampled_frame_id;
      sample.profile_initialized = scalar(word_e::initialized) > 0.5f;
      sample.anchor_valid = scalar(word_e::zero_anchor_valid) > 0.5f;
      sample.range_collapsed = scalar(word_e::range_collapsed) > 0.5f;
      sample.depth_ready = scalar(word_e::depth_ready) > 0.5f;
      sample.hard_cut_pulse = scalar(word_e::hard_cut_pulse) > 0.5f;
      return true;
    }

    depth_telemetry_poll_result poll_depth_telemetry(
      bool schedule_copy,
      std::uint64_t sampled_frame_id
    ) {
      depth_telemetry_poll_result result;
      if (!telemetry_readback_ready && schedule_copy && !ensure_telemetry_readback()) {
        result.failed = true;
        return result;
      }
      if (!telemetry_readback_ready) {
        return result;
      }

      std::uint64_t newest_frame_id = 0;
      for (auto &slot : telemetry_readback_slots) {
        if (!slot.pending) {
          continue;
        }

        BOOL complete = FALSE;
        const auto query_status = context->GetData(
          slot.completion.Get(),
          &complete,
          sizeof(complete),
          D3D11_ASYNC_GETDATA_DONOTFLUSH
        );
        if (query_status == S_FALSE || (SUCCEEDED(query_status) && !complete)) {
          continue;
        }
        if (FAILED(query_status)) {
          slot.pending = false;
          result.failed = true;
          continue;
        }

        D3D11_MAPPED_SUBRESOURCE mapped {};
        const auto map_status = context->Map(
          slot.staging.Get(),
          0,
          D3D11_MAP_READ,
          D3D11_MAP_FLAG_DO_NOT_WAIT,
          &mapped
        );
        if (map_status == DXGI_ERROR_WAS_STILL_DRAWING) {
          continue;
        }
        slot.pending = false;
        if (FAILED(map_status) || !mapped.pData) {
          result.failed = true;
          continue;
        }

        std::array<std::uint32_t, telemetry_state_float_count> words {};
        std::memcpy(words.data(), mapped.pData, sizeof(words));
        context->Unmap(slot.staging.Get(), 0);

        depth_telemetry_sample decoded;
        if (!decode_telemetry_words(
              words,
              target_w,
              target_h,
              slot.sampled_frame_id,
              decoded
            )) {
          result.failed = true;
          continue;
        }
        if (!result.sample || slot.sampled_frame_id >= newest_frame_id) {
          newest_frame_id = slot.sampled_frame_id;
          result.sample = decoded;
        }
      }

      if (schedule_copy) {
        for (std::size_t offset = 0; offset < telemetry_readback_slots.size(); ++offset) {
          const auto index =
            (telemetry_readback_next + offset) % telemetry_readback_slots.size();
          auto &slot = telemetry_readback_slots[index];
          if (slot.pending) {
            continue;
          }
          // The caller invokes this only after submitting the SBS warp and encoder/local-output
          // draw. D3D11 command ordering therefore puts this low-priority diagnostic copy behind
          // every frame-critical consumer of SubjectState.
          context->CopyResource(slot.staging.Get(), subject_buf.Get());
          context->End(slot.completion.Get());
          slot.pending = true;
          slot.sampled_frame_id = sampled_frame_id;
          telemetry_readback_next = (index + 1) % telemetry_readback_slots.size();
          result.copy_scheduled = true;
          break;
        }
      }

      return result;
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
      if (!diagnostics_enabled) {
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
    int canonical_target_w = 0;
    int canonical_target_h = 0;
    int requested_target_w = 0;
    int requested_target_h = 0;
    std::optional<sbs_roi_shape_request> pending_roi_shape_transition;
    // Active mode freezes controller advancement only after a delayed request proposes a
    // destructive tensor-shape change. Same-shape requests remain GPU-validated by ROI generation
    // and committed geometry without inserting a readback bubble into ordinary inference.
    sbs_roi_shape_confirmation_guard roi_shape_confirmation;
    bool roi_canonical_recovery_requested = false;
    std::uint32_t applied_roi_shape_request_id = 0;
    unsigned shape_resource_allocation_failures = 0;
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
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_valid_history_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> frame_roi_transform_cs;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> linear_sampler;
    Microsoft::WRL::ComPtr<ID3D11Buffer> cbuffer;

    struct gpu_uint4_buffer {
      Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
      Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> uav;
    };

    // The all-zero transform is the explicit legacy ABI. It is bound in off/shadow mode so every
    // added shader slot is deterministic and no stale D3D11 binding can accidentally activate an
    // ROI. Active rollout uses the frame-owned banks only after the shape/identity handshake.
    gpu_uint4_buffer zero_roi_transform;
    gpu_uint4_buffer zero_scene_rule_state;
    std::array<
      gpu_uint4_buffer,
      frame_roi_transform_bank_count
    > frame_roi_transform_gpu;
    std::array<
      gpu_uint4_buffer,
      frame_roi_transform_bank_count
    > depth_surface_transform_gpu;
    gpu_uint4_buffer reliable_roi_transform_gpu;
    std::uint32_t current_depth_surface_transform_bank = 0;
    std::array<
      Microsoft::WRL::ComPtr<ID3D11Buffer>,
      frame_roi_transform_bank_count
    > frame_roi_builder_cbuffers;

    Microsoft::WRL::ComPtr<ID3D11Buffer> tensor_in_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> tensor_in_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> tensor_in_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> tensor_previous_input_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> tensor_previous_input_srv;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> tensor_previous_input_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> appearance_ordinal_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> appearance_ordinal_srv;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> appearance_ordinal_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> previous_appearance_ordinal_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> previous_appearance_ordinal_srv;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> previous_appearance_ordinal_uav;

    Microsoft::WRL::ComPtr<ID3D11Buffer> tensor_out_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> tensor_out_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> raw_snapshot_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_snapshot_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> model_input_snapshot_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> model_input_snapshot_srv;
    bool raw_snapshot_error_logged = false;
    bool model_input_snapshot_error_logged = false;
    unsigned raw_snapshot_retry_frames = 0;
    unsigned model_input_snapshot_retry_frames = 0;

    // GPU-resident min/max for per-frame disparity normalization (no CPU readback).
    Microsoft::WRL::ComPtr<ID3D11Buffer> minmax_raw_buf;  // min bits, max bits, valid count, accepted-focus count
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> minmax_raw_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> minmax_ema_buf;  // float4 {min,max,initialized,frame_state}
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> minmax_ema_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> minmax_ema_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> hist_buf;  // 256 uint bins for percentile normalization
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> hist_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subject_hist_buf;  // 256 weighted bins for subject tracking
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subject_hist_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subject_plain_buf;  // 256 bins + seven evidence counters
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subject_plain_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subject_buf;  // seven float4 elements; first three are the warp contract
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subject_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subject_srv;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_previous_tex;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_previous_srv;
    // Cut detection keeps a separate reliable depth endpoint. The ordinary previous texture must
    // still advance every frame for temporal EMA, including through clipped/structureless frames.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_cut_history_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_cut_history_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_cut_history_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> ema_motion_mask_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> ema_motion_mask_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ema_motion_mask_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_reliable_validity_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_reliable_validity_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_reliable_validity_srv;

    CUgraphicsResource cuda_in_res = nullptr;
    CUgraphicsResource cuda_out_res = nullptr;
    bool has_previous_frame = false;
    std::uint64_t pending_frame_id = 0;
    // Off/shadow mode does not allocate, dispatch, map, or lifecycle-track the experimental GPU
    // transform banks. This CPU-only record preserves exact matched-frame provenance for the
    // scene controller while every depth shader binds the immutable all-zero legacy transform.
    std::optional<frame_roi_transform_identity>
      pending_legacy_frame_identity;
    bool disable_after_pending_drop = false;
    // The two identities keep the just-completed GPU bank owned while the next accepted
    // inference uses the other bank. Coordinates, ROI generation, flags, and reset debt never
    // become CPU authority.
    frame_roi_transform_buffer roi_transform_slots;
    bool stream_error_logged = false;
    bool roi_transform_error_logged = false;
    bool scene_controller_error_logged = false;
    bool roi_shape_request_error_logged = false;
    bool readiness_preflighted = false;  // can_accept_frame() already counted/queried this source opportunity
    bool depth_context_pooled = false;  // context reused from the pool (modules already loaded -> skip warmup)
    bool context_warmed = false;  // only warmed contexts may return to context_pool
    // Any terminal TensorRT/CUDA failure poisons this execution context. It remains allocated
    // because deleting the MSVC-owned object from MinGW corrupts the heap, but must never be
    // handed to a later stream as a known-good warmed context.
    bool context_reusable = true;

    void observe_context_event(
      const sbs_trt_context_event event
    ) {
      context_reusable =
        sbs_trt_context_reusable_after(
          context_reusable,
          event
        );
    }

    bool compile_shader(const std::filesystem::path &path, Microsoft::WRL::ComPtr<ID3D11ComputeShader> &out_cs) {
      auto bytecode = depth_shader_bytecode(path);
      if (!bytecode) {
        return false;
      }
      return SUCCEEDED(device->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &out_cs));
    }

    bool create_roi_transform_buffer(
      gpu_uint4_buffer &out,
      bool writable,
      std::size_t vector_count = frame_roi_transform_vector_count
    ) {
      if (
        vector_count == 0 ||
        vector_count >
          std::numeric_limits<UINT>::max() /
            (sizeof(std::uint32_t) * 4u)
      ) {
        return false;
      }
      std::vector<std::uint32_t> zeros(vector_count * 4u, 0u);
      D3D11_BUFFER_DESC desc {};
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.ByteWidth = static_cast<UINT>(
        zeros.size() * sizeof(zeros.front())
      );
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE |
                       (writable ? D3D11_BIND_UNORDERED_ACCESS : 0u);
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = sizeof(std::uint32_t) * 4u;
      D3D11_SUBRESOURCE_DATA initial {zeros.data(), 0, 0};
      if (
        FAILED(device->CreateBuffer(&desc, &initial, &out.buffer)) ||
        FAILED(device->CreateShaderResourceView(
          out.buffer.Get(),
          nullptr,
          &out.srv
        )) ||
        (writable &&
         FAILED(device->CreateUnorderedAccessView(
           out.buffer.Get(),
           nullptr,
           &out.uav
         )))
      ) {
        out = {};
        return false;
      }
      return true;
    }

    bool create_frame_roi_builder_cbuffer(
      Microsoft::WRL::ComPtr<ID3D11Buffer> &out
    ) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(frame_roi_builder_constants);
      desc.Usage = D3D11_USAGE_DYNAMIC;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
      return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &out));
    }

    [[nodiscard]] std::optional<sbs_roi_shape_request_gpu_submission>
    make_roi_shape_submission(
      const scene_controller_gpu_snapshot &controller,
      std::uint32_t source_width,
      std::uint32_t source_height
    ) const {
      if (
        !roi_shape_request_gpu ||
        canonical_target_w <= 0 ||
        canonical_target_h <= 0 ||
        source_width == 0u ||
        source_height == 0u
      ) {
        return std::nullopt;
      }

      sbs_roi_shape_request_gpu_submission submission;
      submission.rule_state =
        controller.snapshot_available ?
          controller.rule_state.Get() :
          nullptr;
      submission.source_frame_id =
        controller.source_frame_id;
      submission.source_width = source_width;
      submission.source_height = source_height;
      submission.canonical_model_width =
        static_cast<std::uint32_t>(canonical_target_w);
      submission.canonical_model_height =
        static_cast<std::uint32_t>(canonical_target_h);
      submission.target_pixel_budget =
        submission.canonical_model_width *
        submission.canonical_model_height;
      submission.profile_max_width = std::min(
        sbs_roi_shape_request_engine_max_dimension,
        source_width
      );
      submission.profile_max_height = std::min(
        sbs_roi_shape_request_engine_max_dimension,
        source_height
      );
      submission.expected_backend_generation =
        controller.backend_generation;
      submission.quiet_halo_cells = 2.0f;
      submission.analysis_canvas_size =
        static_cast<std::uint32_t>(
          sbs_scene_controller::analysis_canvas_size
        );
      submission.max_model_aspect = max_aspect;
      submission.active_rules =
        active_roi_authority ? 1u : 0u;
      return submission;
    }

    [[nodiscard]] bool roi_shape_request_matches_submission(
      const sbs_roi_shape_request &request,
      const sbs_roi_shape_request_gpu_submission &submission
    ) const {
      const sbs_roi_shape_request_limits limits {
        submission.canonical_model_width,
        submission.canonical_model_height,
        submission.profile_max_width,
        submission.profile_max_height,
        submission.max_model_aspect,
      };
      if (
        request.shape[0] != submission.source_width ||
        request.shape[1] != submission.source_height ||
        !sbs_roi_shape_request_valid(request, limits)
      ) {
        return false;
      }

      if (
        sbs_roi_shape_has_flag(
          request,
          sbs_roi_shape_request_flag::active_roi
        )
      ) {
        return submission.expected_backend_generation != 0u &&
               request.identity[0] ==
                 submission.expected_backend_generation;
      }

      // A helper-generated canonical fallback carries the exact staging-slot source/frame
      // provenance but may intentionally have backend zero when the controller SRV is absent or
      // malformed. It can only reduce authority, never authorize a crop.
      return sbs_roi_shape_has_flag(
               request,
               sbs_roi_shape_request_flag::full_frame
             ) &&
             sbs_roi_shape_has_flag(
               request,
               sbs_roi_shape_request_flag::fallback
             );
    }

    /**
     * Write the exact GPU transform paired with a reserved inference bank.
     *
     * Shadow mode deliberately emits a canonical full-frame record and continues binding the
     * explicit all-zero legacy ABI to preprocessing/postprocessing/rendering. This still exercises
     * the real frame/version/bank ownership path without allowing a controller decision to alter
     * output. Active rollout will populate the shape-request expectation and switch consumers only
     * after the dynamic-shape transition gate is complete.
     */
    bool dispatch_frame_roi_transform(
      const frame_roi_transform_identity &identity
    ) {
      if (
        !identity.is_committed() ||
        identity.gpu_bank_index >= frame_roi_transform_gpu.size() ||
        !frame_roi_transform_cs ||
        !frame_roi_transform_gpu[identity.gpu_bank_index].uav ||
        !frame_roi_builder_cbuffers[identity.gpu_bank_index] ||
        !zero_scene_rule_state.srv ||
        !zero_roi_transform.srv
      ) {
        return false;
      }

      frame_roi_builder_constants constants {};
      constants.source_width = identity.source_width;
      constants.source_height = identity.source_height;
      constants.model_width = identity.model_width;
      constants.model_height = identity.model_height;
      constants.source_frame_id_low =
        static_cast<std::uint32_t>(identity.source_frame_id);
      constants.source_frame_id_high =
        static_cast<std::uint32_t>(identity.source_frame_id >> 32u);
      constants.active_rules =
        active_roi_authority ? 1u : 0u;
      constants.expected_backend_generation = identity.backend_generation;
      constants.transform_version_low =
        static_cast<std::uint32_t>(identity.transform_version);
      constants.transform_version_high =
        static_cast<std::uint32_t>(identity.transform_version >> 32u);
      constants.gpu_bank_identity = identity.gpu_bank_index;
      constants.lifecycle_reserved = 1u;
      constants.quiet_halo_cells = 2.0f;
      constants.feather_cells = 2.0f;
      constants.min_focus_cells = 4.0f;
      constants.analysis_canvas_size =
        static_cast<std::uint32_t>(
          sbs_scene_controller::analysis_canvas_size
        );

      ID3D11ShaderResourceView *rule_state =
        zero_scene_rule_state.srv.Get();
      if (active_roi_authority && scene_controller) {
        const auto snapshot = scene_controller->snapshot();
        if (
          snapshot.snapshot_available &&
          snapshot.backend_generation ==
            identity.backend_generation &&
          snapshot.rule_state
        ) {
          rule_state = snapshot.rule_state.Get();
        }
      }

      if (
        active_roi_authority &&
        newest_roi_shape_request
      ) {
        const auto &request = *newest_roi_shape_request;
        const sbs_roi_shape_request_limits limits {
          static_cast<std::uint32_t>(canonical_target_w),
          static_cast<std::uint32_t>(canonical_target_h),
          std::min(
            sbs_roi_shape_request_engine_max_dimension,
            identity.source_width
          ),
          std::min(
            sbs_roi_shape_request_engine_max_dimension,
            identity.source_height
          ),
          max_aspect,
        };
        const bool exact_shape_owner =
          request.identity[0] ==
            identity.backend_generation &&
          request.shape[0] == identity.source_width &&
          request.shape[1] == identity.source_height &&
          request.shape[2] == identity.model_width &&
          request.shape[3] == identity.model_height;
        const bool authorizes_active_roi =
          sbs_roi_shape_has_flag(
            request,
            sbs_roi_shape_request_flag::active_roi
          ) &&
          !sbs_roi_shape_has_flag(
            request,
            sbs_roi_shape_request_flag::fallback
          ) &&
          request.header[2] ==
            static_cast<std::uint32_t>(
              sbs_roi_shape_request_reason::none
            );
        if (
          exact_shape_owner &&
          authorizes_active_roi &&
          sbs_roi_shape_request_valid(request, limits)
        ) {
          constants.expected_request_authority =
            (request.header[1] & 0xFFFFu) |
            ((request.header[2] & 0xFFFFu) << 16u);
          constants.expected_roi_generation =
            request.identity[1];
          constants.shape_request_id =
            request.header[3];
          constants.expected_rule_update_count =
            request.identity[2];
          constants.expected_committed_roi_bits =
            request.committed_roi_bits;
        }
      }

      auto *builder_cbuffer =
        frame_roi_builder_cbuffers[identity.gpu_bank_index].Get();
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const auto map_result = context->Map(
        builder_cbuffer,
        0,
        D3D11_MAP_WRITE_DISCARD,
        0,
        &mapped
      );
      if (FAILED(map_result)) {
        return false;
      }
      if (!mapped.pData) {
        context->Unmap(builder_cbuffer, 0);
        return false;
      }
      std::memcpy(
        mapped.pData,
        &constants,
        sizeof(constants)
      );
      context->Unmap(builder_cbuffer, 0);

      context->CSSetShader(frame_roi_transform_cs.Get(), nullptr, 0);
      context->CSSetConstantBuffers(0, 1, &builder_cbuffer);
      ID3D11ShaderResourceView *builder_srvs[2] = {
        rule_state,
        active_roi_authority ?
          depth_surface_transform_gpu[
            current_depth_surface_transform_bank
          ].srv.Get() :
          zero_roi_transform.srv.Get(),
      };
      context->CSSetShaderResources(0, 2, builder_srvs);
      auto *builder_uav =
        frame_roi_transform_gpu[identity.gpu_bank_index].uav.Get();
      context->CSSetUnorderedAccessViews(0, 1, &builder_uav, nullptr);
      context->Dispatch(1, 1, 1);

      ID3D11UnorderedAccessView *null_uav = nullptr;
      ID3D11ShaderResourceView *null_srvs[2] = {nullptr, nullptr};
      ID3D11Buffer *null_cbuffer = nullptr;
      context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
      context->CSSetShaderResources(0, 2, null_srvs);
      context->CSSetConstantBuffers(0, 1, &null_cbuffer);
      return true;
    }

    impl(Microsoft::WRL::ComPtr<ID3D11Device> d, Microsoft::WRL::ComPtr<ID3D11DeviceContext> c, const std::filesystem::path &assets_dir, const config::video_t::sbs_t &cfg, const config::depth_model_info &model):
        device(d),
        context(c),
        ema_alpha((float) cfg.ema),
        ema_edge_change((float) cfg.ema_edge_change),
        ema_edge_gradient((float) cfg.ema_edge_gradient),
        ema_edge_strength((float) cfg.ema_edge_strength),
        depth_short_side(std::max(196, cfg.depth_short_side)),
        max_aspect(std::max(1.0f, (float) cfg.depth_max_aspect)),
        minmax_alpha((float) cfg.minmax_ema),
        cuda_graph_enabled(cfg.cuda_graph),
        diagnostics_enabled(config::sunshine.diagnostics_enabled),
        subject_recenter((float) cfg.subject_recenter),
        subject_stretch(cfg.subject_stretch),
        adaptive_pop(cfg.adaptive_pop),
        adaptive_pop_max_ratio((float) (std::max(cfg.adaptive_pop_max, cfg.pop_strength) /
                                        std::max(cfg.pop_strength, 0.25))),
        // Unrecognised falls back to `median`, the default and validated plane. It must NOT fall
        // back to 0: that was `legacy`, which no longer exists, and the shader would read it as
        // `subject` (its selector is `< 1.5f`). config.cpp resets bad strings, but the offline
        // harness assigns sbs_cfg directly and bypasses that.
        zero_plane_mode(cfg.zero_plane == "subject" ? 1.0f : cfg.zero_plane == "background" ? 3.0f :
                                                                                              2.0f) {
      const auto init_started = std::chrono::steady_clock::now();
      // Enable the process-wide rolling collector for diagnostic runs. Do not reset it here:
      // Galaxy XR and local-AR estimators may coexist, and one session must not invalidate the
      // other session's pending D3D query generation. The offline harness resets explicitly.
      perf_depth.stage = "depth_infer";
      sbs_perf::set_enabled(diagnostics_enabled);
      initialize_d3d_perf();

      auto &cuda = cuda_driver_api::get();
      if (cuda.is_valid() && ensure_cuda_initialized(cuda)) {
        if (cuda_device_for_d3d(cuda, device.Get(), cuda_device)) {
          cuda_ctx = primary_context(cuda, cuda_device);
          if (
            cuda_ctx &&
            cuda.cuCtxSetCurrent(cuda_ctx) == CUDA_SUCCESS
          ) {
            if (
              cuda.cuStreamCreate(
                &cu_stream,
                CU_STREAM_NON_BLOCKING
              ) != CUDA_SUCCESS
            ) {
              cu_stream = nullptr;
            }
          }
        }
      }
      if (!cuda_ctx || !cu_stream) {
        BOOST_LOG(error) << "Depth estimator failed: CUDA context/stream initialization failed.";
        return;
      }

      engine_artifact artifact;
      if (!ensure_tensorrt_engine_for_device(assets_dir, model, cuda, cuda_device, artifact)) {
        BOOST_LOG(error) << "Depth estimator failed: TensorRT engine preparation failed.";
        return;
      }
      auto model_path = artifact.engine_path;
      engine_key = std::to_string(cuda_device) + ":" + artifact.name;

      {  // Scope this lock to the g_engines/g_runtime access only: it MUST be released before
         // warmup_inference() at the end of the ctor (which re-locks g_trt_mutex) -- a
         // non-recursive std::mutex would otherwise self-deadlock and hang construction.
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        // Load (once) the engine for this configured model into its own slot and take a pooled
        // execution context if one is free. Different startup configurations remain isolated.
        engine = acquire_engine_locked(engine_key, model_path, exec_context, depth_context_pooled);
        context_warmed = depth_context_pooled;
        if (depth_context_pooled) {
          BOOST_LOG(info) << "Reusing pooled TensorRT execution context.";
        }
        auto &slot = g_engines[engine_key];

        if (!validate_engine_io_locked(engine, slot)) {
          BOOST_LOG(error) << "Depth engine I/O contract is incompatible with Sunshine 3D; streaming flat SBS.";
          if (exec_context) {
            slot.context_pool.push_back(exec_context);
            exec_context = nullptr;
            g_trt_context_available.notify_all();
          }
          engine = nullptr;
        }

        trt_mutex = &g_trt_mutex;
      }  // release g_trt_mutex before the shader/buffer setup and warmup below

      if (!engine) {
        // The startup preparation normally repairs this already. The constructor also serves the
        // standalone evaluator, so retain the same one-shot self-heal when it is the first owner.
        {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          auto found = g_engines.find(engine_key);
          if (found != g_engines.end() && !found->second.engine && allocated_context_count(found->second) == 0 && found->second.context_pool.empty()) {
            g_engines.erase(found);
          }
        }
        std::error_code ec;
        std::filesystem::remove(model_path, ec);
        BOOST_LOG(warning) << "Depth estimator found an unreadable TensorRT plan; rebuilding " << model_path.filename() << '.';
        if (!ensure_tensorrt_engine_for_device(assets_dir, model, cuda, cuda_device, artifact)) {
          return;
        }
        model_path = artifact.engine_path;
        engine_key = std::to_string(cuda_device) + ":" + artifact.name;
        {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          engine = acquire_engine_locked(engine_key, model_path, exec_context, depth_context_pooled);
          context_warmed = depth_context_pooled;
          auto &slot = g_engines[engine_key];
          if (!validate_engine_io_locked(engine, slot)) {
            engine = nullptr;
          }
        }
        if (!engine) {
          BOOST_LOG(error) << "Depth estimator failed: rebuilt TensorRT plan could not be deserialized.";
          return;
        }
      }

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
            context_warmed = true;
            BOOST_LOG(info) << "Reusing pooled TensorRT execution context (freed by a racing teardown).";
          }
        }
        bool create_context = false;
        if (!exec_context) {
          std::unique_lock<std::mutex> lock(g_trt_mutex);
          auto &slot = g_engines[engine_key];
          if (slot.context_pool.empty() && allocated_context_count(slot) >= kMaxContextsPerEngine) {
            BOOST_LOG(warning) << "TensorRT context cap reached for this depth model; waiting for "
                                  "an asynchronous encoder teardown to return one.";
            const bool available = g_trt_context_available.wait_for(
              lock,
              std::chrono::seconds(5),
              [&slot]() {
                return !slot.context_pool.empty() || allocated_context_count(slot) < kMaxContextsPerEngine;
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
            context_warmed = true;
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
        compile_shader(assets_dir / "shaders" / "directx" / "depth_subject_resolve_cs.hlsl", depth_subject_resolve_cs) &&
        compile_shader(assets_dir / "shaders" / "directx" / "depth_valid_history_cs.hlsl", depth_valid_history_cs);
      if (!core_shaders_ok) {
        BOOST_LOG(error) << "Depth estimator failed: required Bestv2 shader initialization failed.";
        return;
      }
      bool roi_transform_resources_ok =
        create_roi_transform_buffer(zero_roi_transform, false);
      if (active_roi_authority) {
        roi_transform_resources_ok =
          compile_shader(
            assets_dir / "shaders" / "directx" /
              "sbs_frame_roi_transform_cs.hlsl",
            frame_roi_transform_cs
          ) &&
          create_roi_transform_buffer(
            zero_scene_rule_state,
            false,
            sbs_scene_controller::rule_state_vector_count
          ) &&
          create_roi_transform_buffer(
            reliable_roi_transform_gpu,
            true
          ) &&
          roi_transform_resources_ok;
        for (auto &bank : frame_roi_transform_gpu) {
          roi_transform_resources_ok =
            create_roi_transform_buffer(bank, true) &&
            roi_transform_resources_ok;
        }
        for (auto &bank : depth_surface_transform_gpu) {
          roi_transform_resources_ok =
            create_roi_transform_buffer(bank, true) &&
            roi_transform_resources_ok;
        }
        for (auto &cbuffer : frame_roi_builder_cbuffers) {
          roi_transform_resources_ok =
            create_frame_roi_builder_cbuffer(cbuffer) &&
            roi_transform_resources_ok;
        }
      }
      if (!roi_transform_resources_ok) {
        BOOST_LOG(error)
          << "Depth estimator failed: could not allocate the GPU ROI-transform contract.";
        return;
      }
      BOOST_LOG(info) << "Permanent Bestv2 subject shaping enabled (recenter " << subject_recenter << ").";
      BOOST_LOG(info) << "SBS zero-plane mode: " << cfg.zero_plane
                      << (zero_plane_mode > 0.5f ? " (shot-latched experimental anchor)." : ".");
      // Min/max reduction accumulator, pre-seeded to the reduction identity
      // {min = 0xFFFFFFFF, max = 0, valid = 0, accepted = 0}.
      // depth_minmax_ema_cs resets all four words after each frame.
      {
        uint32_t init_raw[4] = {0xFFFFFFFFu, 0u, 0u, 0u};
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
        uav.Buffer.NumElements = 4;
        uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(minmax_raw_buf.Get(), &uav, &minmax_raw_uav);
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

      // Subject tracking: weighted histogram (256 uint bins), plain histogram plus depth-edge,
      // depth-change, ordinal-structure, broad-RGB-change, and three structure-support counters
      // (263 uints), and
      // three-float4 state.
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
        uint32_t init_plain[263] = {};
        bd.ByteWidth = sizeof(init_plain);
        D3D11_SUBRESOURCE_DATA plain_sd = {init_plain, 0, 0};
        device->CreateBuffer(&bd, &plain_sd, &subject_plain_buf);
        if (subject_plain_buf) {
          device->CreateUnorderedAccessView(subject_plain_buf.Get(), nullptr, &subject_plain_uav);
        }

        // [0] subject/recenter, [1] stretch/depth-cut baseline/pop,
        // [2] explicit zero-plane anchor/cut flags. [3..7] are append-only diagnostics and are
        // never consumed by the production warp. Keep [3].x at the unclassified sentinel.
        const auto &init_state = sbs_adaptive_state::initial_values;
        bd.ByteWidth = sizeof(init_state);
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        bd.StructureByteStride = sizeof(float) * 4;
        D3D11_SUBRESOURCE_DATA sd2 = {init_state.data(), 0, 0};
        device->CreateBuffer(&bd, &sd2, &subject_buf);
        if (subject_buf) {
          device->CreateUnorderedAccessView(subject_buf.Get(), nullptr, &subject_uav);
          device->CreateShaderResourceView(subject_buf.Get(), nullptr, &subject_srv);
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
              depth_subject_hist_cs && depth_subject_resolve_cs && depth_valid_history_cs &&
              minmax_raw_uav && minmax_ema_uav && minmax_ema_srv && hist_uav &&
              subject_hist_uav && subject_plain_uav && subject_uav && subject_srv &&
              zero_roi_transform.srv && linear_sampler;
      if (!valid) {
        BOOST_LOG(error) << "Depth estimator failed: required engine or Bestv2 GPU resource initialization failed.";
        return;
      }

      if (cfg.scene_controller != config::sbs_scene_controller_e::off) {
        scene_controller = std::make_unique<sbs_scene_controller_gpu>(
          device,
          context,
          assets_dir,
          cfg.scene_controller,
          cfg
        );
        if (!scene_controller->valid()) {
          BOOST_LOG(warning)
            << "Host SBS scene controller is unavailable; continuing with the unchanged "
               "full-frame depth controller.";
          scene_controller.reset();
        }
        if (scene_controller) {
          roi_shape_request_gpu =
            std::make_unique<sbs_roi_shape_request_gpu>(
              device,
              context,
              assets_dir
            );
          if (!roi_shape_request_gpu->valid()) {
            BOOST_LOG(warning)
              << "Host SBS ROI shape-request readback is unavailable; "
                 "shadow rendering remains unchanged.";
            roi_shape_request_gpu.reset();
          }
        }
      }

      // Constant buffers are created in ensure_cbuffers() once the model resolution is
      // known: every field is fixed for the session, so they are built once (immutable)
      // instead of being re-mapped on the encode thread every frame.

      // Warm up here so TensorRT's CUDA lazy kernel load / JIT (~20 s on the big models)
      // happens during construction -- which ensure_depth_estimator() runs on a background
      // thread -- rather than stalling the first real convert() on the encode thread and
      // freezing the stream when Host SBS first becomes active.
      if (!warmup_inference()) {
        valid = false;
        BOOST_LOG(error) << "Depth estimator failed: TensorRT execution-context warmup failed.";
        return;
      }
      if (!publish_active_engine_manifest(assets_dir, model, artifact)) {
        BOOST_LOG(error) << "Could not publish the active TensorRT engine manifest for model '"
                         << model.name << "'.";
      }
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
    bool warmup_inference() {
      if (!exec_context || !cu_stream) {
        return false;
      }
      if (depth_context_pooled) {
        return true;  // only successfully warmed contexts are admitted to the pool
      }
      auto &cuda = cuda_driver_api::get();
      if (!cuda.is_valid()) {
        return false;
      }
      if (!warmup_execution_context(cuda, cuda_ctx, exec_context)) {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        quarantine_execution_context_locked(engine_key, exec_context);
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        mark_execution_context_warmed_locked(engine_key);
      }
      context_warmed = true;
      return true;
    }

    ~impl() {
      auto &cuda = cuda_driver_api::get();
      bool context_selected = false;
      bool stream_idle = cu_stream == nullptr;
      if (
        cuda.is_valid() && cuda_ctx && cuda.cuCtxSetCurrent &&
        cuda.cuCtxSetCurrent(cuda_ctx) == CUDA_SUCCESS
      ) {
        context_selected = true;
        if (cu_stream && cuda.cuStreamSynchronize) {
          stream_idle =
            cuda.cuStreamSynchronize(cu_stream) == CUDA_SUCCESS;
        }
      }

      if (!context_selected) {
        observe_context_event(
          sbs_trt_context_event::cuda_context_failure
        );
      } else if (!stream_idle) {
        // A missing synchronize entry point is equivalent to a failed synchronization: neither
        // case proves that TensorRT and D3D interop have stopped using their backing resources.
        observe_context_event(
          sbs_trt_context_event::cuda_stream_failure
        );
      }

      const auto cleanup_disposition =
        sbs_cuda_resource_cleanup_policy(
          context_selected,
          stream_idle
        );
      if (
        cleanup_disposition ==
        sbs_cuda_resource_cleanup_disposition::release
      ) {
        if (cu_stream) {
          destroy_inference_graph(cuda);
          if (
            !cuda.cuStreamDestroy ||
            cuda.cuStreamDestroy(cu_stream) != CUDA_SUCCESS
          ) {
            observe_context_event(
              sbs_trt_context_event::cuda_stream_failure
            );
          }
        }
        if (
          cuda_in_res &&
          (
            !cuda.cuGraphicsUnregisterResource ||
            cuda.cuGraphicsUnregisterResource(cuda_in_res) !=
              CUDA_SUCCESS
          )
        ) {
          // Do not release a D3D resource still owned by an untracked CUDA registration. This is
          // a terminal error path; intentionally retain one COM reference until process exit.
          tensor_in_buf.Detach();
          observe_context_event(
            sbs_trt_context_event::cuda_interop_failure
          );
        } else {
          cuda_in_res = nullptr;
        }
        if (
          cuda_out_res &&
          (
            !cuda.cuGraphicsUnregisterResource ||
            cuda.cuGraphicsUnregisterResource(cuda_out_res) !=
              CUDA_SUCCESS
          )
        ) {
          tensor_out_buf.Detach();
          observe_context_event(
            sbs_trt_context_event::cuda_interop_failure
          );
        } else {
          cuda_out_res = nullptr;
        }
        perf_destroy_events();  // free the timing events while cuda_ctx is still current
      } else {
        // Do not issue any context-bound CUDA call under an unknown current context or against a
        // stream whose idleness is unproven. The process owns these handles until exit. Retain the
        // backing COM references too, because pending/registered CUDA work may still reference
        // them after this C++ object is gone.
        if (cuda_in_res) {
          tensor_in_buf.Detach();
        }
        if (cuda_out_res) {
          tensor_out_buf.Detach();
        }
      }

      // Return only a successfully warmed context to the reusable pool. Construction can fail
      // after createExecutionContext() but before warmup (shader/resource allocation is one such
      // path); admitting that context would make the next instance skip the failed lazy load.
      // Contexts cannot be destroyed safely across the DLL boundary, so quarantine them instead.
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      if (exec_context) {
        if (
          sbs_trt_context_pool_disposition(
            context_warmed,
            context_reusable
          ) == sbs_trt_context_disposition::reuse
        ) {
          g_engines[engine_key].context_pool.push_back(exec_context);
          exec_context = nullptr;
          g_trt_context_available.notify_all();
        } else {
          quarantine_execution_context_locked(
            engine_key,
            exec_context,
            context_warmed
          );
        }
      }
      // TRT runtime/engines are cached globally, do not destroy them here.
    }

    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> output_srv() {
      return depth_srv;
    }

    estimate_result make_result(
      bool completed_frame_valid = false,
      std::uint64_t completed_frame_id = 0,
      bool inference_enqueued = false,
      bool raw_snapshot_valid = false,
      bool model_input_snapshot_valid = false,
      bool completion_dropped = false,
      std::uint64_t dropped_frame_id = 0
    ) {
      estimate_result r;
      r.depth = output_srv();
      r.depth_roi_transform =
        active_roi_authority ?
          depth_surface_transform_gpu[
            current_depth_surface_transform_bank
          ].srv :
          zero_roi_transform.srv;
      r.subject = subject_srv;
      r.depth_frame_state = minmax_ema_srv;
      r.ema_motion_mask = ema_motion_mask_srv;
      r.raw_model_depth = tensor_out_srv;
      if (raw_snapshot_valid) {
        r.raw_model_depth_snapshot = raw_snapshot_srv;
      }
      if (model_input_snapshot_valid) {
        r.model_input_snapshot = model_input_snapshot_srv;
      }
      r.raw_width = target_w;
      r.raw_height = target_h;
      r.completed_frame_valid = completed_frame_valid;
      r.completed_frame_id = completed_frame_id;
      r.inference_enqueued = inference_enqueued;
      r.cuda_graph_active = inference_graph_exec != nullptr && !graph_capture_failed;
      r.completion_dropped = completion_dropped;
      r.dropped_frame_id = dropped_frame_id;
      if (completed_frame_valid) {
        if (const auto *transform =
              roi_transform_slots.completed_for(completed_frame_id)) {
          r.completed_roi_transform_identity = *transform;
        }
      }
      if (inference_enqueued) {
        if (const auto *transform =
              roi_transform_slots.pending_for(pending_frame_id)) {
          r.enqueued_roi_transform_identity = *transform;
        }
      }
      if (scene_controller) {
        const auto controller = scene_controller->snapshot();
        r.scene_controller_scene_rgb = controller.scene_rgb;
        r.scene_controller_analysis_grid = controller.analysis_grid;
        r.scene_controller_dense_output = controller.dense_output;
        r.scene_controller_global_output = controller.global_output;
        r.scene_controller_layout_history = controller.layout_history;
        r.scene_controller_depth_history = controller.depth_history;
        r.scene_controller_hidden_output = controller.hidden_output;
        r.scene_controller_meta = controller.meta;
        r.scene_controller_rule_state = controller.rule_state;
        r.scene_controller_frame_id = controller.source_frame_id;
        r.scene_controller_backend_generation = controller.backend_generation;
        r.scene_controller_snapshot_available =
          controller.snapshot_available;
        r.scene_controller_shadow = controller.shadow;
      }
      return r;
    }

    /**
     * Promote the exact pending ROI ownership before any normalization/history shader can consume
     * the raw output. Returning nullopt is a hard fail-closed decision for that completion.
     */
    std::optional<frame_roi_transform_identity> claim_depth_completion(
      std::uint64_t source_frame_id
    ) {
      if (!active_roi_authority) {
        if (
          !pending_legacy_frame_identity ||
          pending_legacy_frame_identity->source_frame_id !=
            source_frame_id
        ) {
          return std::nullopt;
        }
        const auto identity = *pending_legacy_frame_identity;
        pending_legacy_frame_identity.reset();
        return identity;
      }

      const auto *pending = roi_transform_slots.pending_for(source_frame_id);
      if (!pending) {
        return std::nullopt;
      }
      const auto identity = *pending;
      if (!roi_transform_slots.complete(identity)) {
        return std::nullopt;
      }
      return identity;
    }

    void drop_depth_completion(
      std::uint64_t source_frame_id,
      std::string_view reason
    ) {
      if (active_roi_authority) {
        if (!roi_transform_slots.drop_in_flight(source_frame_id)) {
          // A corrupt/missing state must not retain a bank and starve every later submission.
          roi_transform_slots.abandon_in_flight();
        }
      } else {
        pending_legacy_frame_identity.reset();
      }
      if (!roi_transform_error_logged) {
        BOOST_LOG(error)
          << "Depth estimator dropped frame " << source_frame_id
          << " before normalization because " << reason
          << "; the last valid depth remains authoritative.";
        roi_transform_error_logged = true;
      }
    }

    std::optional<std::uint64_t> abandon_pending_after_cuda_error(
      std::string_view reason,
      const sbs_trt_context_event event =
        sbs_trt_context_event::cuda_stream_failure
    ) {
      std::optional<std::uint64_t> dropped;
      if (has_previous_frame) {
        dropped = pending_frame_id;
      }
      roi_transform_slots.abandon_in_flight();
      pending_legacy_frame_identity.reset();
      has_previous_frame = false;
      disable_after_pending_drop = false;
      valid = false;
      observe_context_event(event);
      if (!stream_error_logged) {
        BOOST_LOG(error)
          << "Depth estimator abandoned its pending inference after " << reason
          << "; the estimator is disabled for this session.";
        stream_error_logged = true;
      }
      return dropped;
    }

    bool snapshot_buffer(
      ID3D11Buffer *source,
      Microsoft::WRL::ComPtr<ID3D11Buffer> &snapshot,
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> &snapshot_srv,
      bool &error_logged,
      unsigned &retry_frames,
      std::string_view label
    ) {
      if (!source) {
        return false;
      }
      if (retry_frames > 0) {
        --retry_frames;
        return false;
      }

      D3D11_BUFFER_DESC source_desc {};
      source->GetDesc(&source_desc);
      bool recreate = !snapshot || !snapshot_srv;
      if (!recreate) {
        D3D11_BUFFER_DESC snapshot_desc {};
        snapshot->GetDesc(&snapshot_desc);
        recreate = snapshot_desc.ByteWidth != source_desc.ByteWidth ||
                   snapshot_desc.StructureByteStride != source_desc.StructureByteStride ||
                   snapshot_desc.MiscFlags != source_desc.MiscFlags;
      }
      if (recreate) {
        snapshot.Reset();
        snapshot_srv.Reset();
        if (FAILED(device->CreateBuffer(&source_desc, nullptr, &snapshot)) || FAILED(device->CreateShaderResourceView(snapshot.Get(), nullptr, &snapshot_srv))) {
          snapshot.Reset();
          snapshot_srv.Reset();
          if (!error_logged) {
            BOOST_LOG(warning) << "Depth estimator could not allocate the Dump 3D " << label
                               << " snapshot.";
            error_logged = true;
          }
          // Keep a retained request recoverable after transient device pressure without retrying
          // two allocations on every completed inference indefinitely.
          retry_frames = 120;
          return false;
        }
      }
      context->CopyResource(snapshot.Get(), source);
      retry_frames = 0;
      return true;
    }

    /**
     * Retire every shape-dependent resource while the TensorRT stream is known idle.
     *
     * The caller invokes this only on the capture opportunity after the old-shape completion was
     * rendered. The display therefore repeats its already materialized SBS texture while this
     * method tears down interop and the following allocation block builds the requested shape.
     */
    bool unregister_cuda_interop_resource(
      cuda_driver_api &cuda,
      CUgraphicsResource &resource,
      const char *label
    ) {
      if (!resource) {
        return true;
      }
      const auto status =
        cuda.cuGraphicsUnregisterResource(resource);
      if (status == CUDA_SUCCESS) {
        resource = nullptr;
        return true;
      }
      BOOST_LOG(error)
        << "Depth estimator could not unregister the " << label
        << " CUDA interop resource: " << status;
      // Preserve the handle and its backing COM resource. Replacing the D3D allocation while
      // CUDA still owns an untracked registration can fault the device; the caller must disable
      // this estimator instead.
      return false;
    }

    bool release_shape_resources_for_rebuild(
      cuda_driver_api &cuda
    ) {
      if (
        has_previous_frame ||
        roi_transform_slots.has_pending() ||
        roi_transform_slots.has_reserved() ||
        roi_transform_slots.has_orphaned()
      ) {
        return false;
      }

      destroy_inference_graph(cuda);
      graph_input = 0;
      graph_output = 0;
      graph_width = 0;
      graph_height = 0;
      configured_input_width = 0;
      configured_input_height = 0;

      const bool input_released =
        unregister_cuda_interop_resource(
          cuda,
          cuda_in_res,
          "old ROI input"
        );
      const bool output_released =
        unregister_cuda_interop_resource(
          cuda,
          cuda_out_res,
          "old ROI output"
        );
      const bool interop_released =
        input_released && output_released;
      if (!interop_released) {
        valid = false;
        observe_context_event(
          sbs_trt_context_event::cuda_interop_failure
        );
        return false;
      }

      tensor_in_uav.Reset();
      tensor_in_srv.Reset();
      tensor_in_buf.Reset();
      tensor_previous_input_uav.Reset();
      tensor_previous_input_srv.Reset();
      tensor_previous_input_buf.Reset();
      appearance_ordinal_uav.Reset();
      appearance_ordinal_srv.Reset();
      appearance_ordinal_buf.Reset();
      previous_appearance_ordinal_uav.Reset();
      previous_appearance_ordinal_srv.Reset();
      previous_appearance_ordinal_buf.Reset();
      tensor_out_srv.Reset();
      tensor_out_buf.Reset();
      raw_snapshot_srv.Reset();
      raw_snapshot_buf.Reset();
      model_input_snapshot_srv.Reset();
      model_input_snapshot_buf.Reset();
      depth_uav.Reset();
      depth_srv.Reset();
      depth_tex.Reset();
      depth_previous_srv.Reset();
      depth_previous_tex.Reset();
      depth_cut_history_uav.Reset();
      depth_cut_history_srv.Reset();
      depth_cut_history_tex.Reset();
      ema_motion_mask_uav.Reset();
      ema_motion_mask_srv.Reset();
      ema_motion_mask_tex.Reset();
      depth_reliable_validity_uav.Reset();
      depth_reliable_validity_srv.Reset();
      depth_reliable_validity_tex.Reset();

      cbuffer.Reset();
      cb_color_mode = -1;
      reduce_groups = 0u;
      target_w = 0;
      target_h = 0;
      raw_snapshot_retry_frames = 0u;
      model_input_snapshot_retry_frames = 0u;
      current_depth_surface_transform_bank = 0u;
      pending_legacy_frame_identity.reset();
      roi_transform_slots.reset_for_resource_rebuild();

      const UINT clear_uint[4] = {0u, 0u, 0u, 0u};
      for (auto &bank : frame_roi_transform_gpu) {
        context->ClearUnorderedAccessViewUint(
          bank.uav.Get(),
          clear_uint
        );
      }
      for (auto &bank : depth_surface_transform_gpu) {
        context->ClearUnorderedAccessViewUint(
          bank.uav.Get(),
          clear_uint
        );
      }
      context->ClearUnorderedAccessViewUint(
        reliable_roi_transform_gpu.uav.Get(),
        clear_uint
      );

      const std::array<std::uint32_t, 4> raw_identity {
        0xFFFFFFFFu,
        0u,
        0u,
        0u,
      };
      context->UpdateSubresource(
        minmax_raw_buf.Get(),
        0,
        nullptr,
        raw_identity.data(),
        0,
        0
      );
      const std::array<float, 4> zero_ema {};
      context->UpdateSubresource(
        minmax_ema_buf.Get(),
        0,
        nullptr,
        zero_ema.data(),
        0,
        0
      );
      context->ClearUnorderedAccessViewUint(
        hist_uav.Get(),
        clear_uint
      );
      context->ClearUnorderedAccessViewUint(
        subject_hist_uav.Get(),
        clear_uint
      );
      context->ClearUnorderedAccessViewUint(
        subject_plain_uav.Get(),
        clear_uint
      );
      context->UpdateSubresource(
        subject_buf.Get(),
        0,
        nullptr,
        sbs_adaptive_state::initial_values.data(),
        0,
        0
      );
      return true;
    }

    // estimate() has already submitted one inference. Wait for that exact inference, consume it
    // once, and deliberately do NOT enqueue a duplicate. This is the synchronous quality oracle.
    estimate_result finish_pending(input_color_space color_space) {
      auto &cuda = cuda_driver_api::get();
      if (!has_previous_frame || !cu_stream || !cuda.cuStreamSynchronize) {
        if (has_previous_frame && (!cu_stream || !cuda.cuStreamSynchronize)) {
          const auto dropped =
            abandon_pending_after_cuda_error(
              "the CUDA stream synchronization contract is unavailable",
              sbs_trt_context_event::cuda_stream_failure
            );
          return make_result(
            false,
            0,
            false,
            false,
            false,
            dropped.has_value(),
            dropped.value_or(0)
          );
        }
        return make_result();
      }
      if (cuda_ctx) {
        const auto set_current = cuda.cuCtxSetCurrent(cuda_ctx);
        if (set_current != CUDA_SUCCESS) {
          BOOST_LOG(error)
            << "cuCtxSetCurrent failed while finishing depth: "
            << set_current;
          const auto dropped =
            abandon_pending_after_cuda_error(
              "cuCtxSetCurrent failed while finishing depth",
              sbs_trt_context_event::cuda_context_failure
            );
          return make_result(
            false,
            0,
            false,
            false,
            false,
            dropped.has_value(),
            dropped.value_or(0)
          );
        }
      }
      CUresult sync = cuda.cuStreamSynchronize(cu_stream);
      if (sync != CUDA_SUCCESS) {
        BOOST_LOG(error) << "Depth synchronization failed: " << sync;
        const auto dropped =
          abandon_pending_after_cuda_error("cuStreamSynchronize failed");
        return make_result(
          false,
          0,
          false,
          false,
          false,
          dropped.has_value(),
          dropped.value_or(0)
        );
      }
      if (diagnostics_enabled) {
        perf_drain(perf_depth);
      }
      ensure_cbuffers(color_space);
      if (!cbuffer) {
        return {};
      }
      const auto completed_frame_id = pending_frame_id;
      const bool explicitly_orphaned =
        active_roi_authority &&
        roi_transform_slots.orphaned_for(completed_frame_id);
      const auto completed_roi_identity =
        claim_depth_completion(completed_frame_id);
      if (!completed_roi_identity) {
        drop_depth_completion(
          completed_frame_id,
          "its exact pending ROI-transform identity was absent or mismatched"
        );
        has_previous_frame = false;
        if (disable_after_pending_drop || !explicitly_orphaned) {
          disable_after_pending_drop = false;
          valid = false;
        }
        return make_result(
          false,
          0,
          false,
          false,
          false,
          true,
          completed_frame_id
        );
      }
      auto *d3d_timer = diagnostics_enabled ? begin_d3d_perf(true, false) : nullptr;
      normalize_depth_output(*completed_roi_identity, d3d_timer);
      mark_d3d_post_end(d3d_timer);
      mark_d3d_pre_start(d3d_timer);
      end_d3d_perf(d3d_timer);
      has_previous_frame = false;  // the output buffer has been consumed; never fold it twice
      return make_result(true, completed_frame_id);
    }

    // (Re)build the depth constant buffer. Its contents are session-constant once the model
    // resolution is fixed, so it is immutable and rebuilt only if capture color encoding changes
    // during a display/mode transition.
    void ensure_cbuffers(input_color_space color_space) {
      const int color_mode = (int) color_space;
      if (cb_color_mode == color_mode && cbuffer) {
        return;
      }
      cb_color_mode = color_mode;

      D3D11_BUFFER_DESC cb_desc = {};
      cb_desc.Usage = D3D11_USAGE_IMMUTABLE;
      cb_desc.ByteWidth = 64;  // shared depth-pass cbuffer (16 floats/uints; see below)
      cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

      // Shared depth-pass constants, 16 scalars = 4 float4 registers. THIS fill is the
      // single source of truth for the canonical layout in
      // shaders/directx/include/depth_constants.hlsl -- every cbf[N] below must stay
      // slot-for-slot with the include (which every depth shader #includes). To add a
      // field: append it here AND to the include.
      uint32_t cb[16] = {};
      float *cbf = (float *) cb;
      cb[0] = (uint32_t) target_w;
      cb[1] = (uint32_t) target_h;
      cb[2] = (uint32_t) color_mode;
      cbf[3] = ema_alpha;
      cbf[4] = minmax_alpha;
      cb[5] = reduce_groups * 256u;  // total threads for the reduction grid-stride
      cbf[6] = ema_edge_change;
      cbf[7] = ema_edge_gradient;
      cbf[8] = ema_edge_strength;
      cbf[9] = subject_recenter;  // subject recenter strength (depth_subject_resolve_cs)
      cbf[10] = subject_stretch ? 1.0f : 0.0f;
      cbf[11] = adaptive_pop ? 1.0f : 0.0f;
      cbf[12] = adaptive_pop_max_ratio;
      cbf[13] = zero_plane_mode;
      D3D11_SUBRESOURCE_DATA sd = {cb, 0, 0};
      cbuffer.Reset();
      device->CreateBuffer(&cb_desc, &sd, &cbuffer);
    }

    // Normalize the finished raw disparity in tensor_out_buf into depth_tex: the scale
    // passes (min/max reduction, permanent percentile histogram, EMA fold) followed by the
    // mapping/temporal-EMA pass. GPU-resident throughout, no CPU readback.
    void normalize_depth_output(
      const frame_roi_transform_identity &completed_roi_identity,
      d3d_perf_slot *d3d_timer = nullptr
    ) {
      const auto completed_bank =
        completed_roi_identity.gpu_bank_index;
      const bool owned_transform_ready =
        active_roi_authority &&
        completed_bank < frame_roi_transform_gpu.size() &&
        frame_roi_transform_gpu[completed_bank].srv &&
        depth_surface_transform_gpu[
          current_depth_surface_transform_bank
        ].srv &&
        reliable_roi_transform_gpu.srv;
      // `shadow_rules` binds the explicit all-zero legacy ABI everywhere. Active rollout switches
      // this tuple atomically: current accepted-frame transform, transform paired with the retained
      // depth surface, and transform paired with the structurally reliable cut-history endpoint.
      ID3D11ShaderResourceView *current_transform =
        owned_transform_ready ?
          frame_roi_transform_gpu[completed_bank].srv.Get() :
          zero_roi_transform.srv.Get();
      ID3D11ShaderResourceView *previous_surface_transform =
        owned_transform_ready ?
          depth_surface_transform_gpu[
            current_depth_surface_transform_bank
          ].srv.Get() :
          zero_roi_transform.srv.Get();
      ID3D11ShaderResourceView *reliable_transform =
        owned_transform_ready ?
          reliable_roi_transform_gpu.srv.Get() :
          zero_roi_transform.srv.Get();
      ID3D11ShaderResourceView *null_transform_srvs[2] = {
        nullptr,
        nullptr
      };

      // 3a. Per-frame scale (GPU-resident; no CPU readback). Depth Anything V2's
      // relative output is affine-invariant, so this is required for a stable parallax scale.
      if (depth_minmax_cs && depth_minmax_ema_cs && minmax_raw_uav && minmax_ema_uav) {
        // Pass A: parallel reduction of the raw disparity -> min/max (uint bits).
        context->CSSetShader(depth_minmax_cs.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
        ID3D11ShaderResourceView *minmax_srvs[2] = {
          tensor_out_srv.Get(),
          current_transform
        };
        context->CSSetShaderResources(0, 2, minmax_srvs);
        context->CSSetUnorderedAccessViews(0, 1, minmax_raw_uav.GetAddressOf(), nullptr);
        context->Dispatch(reduce_groups, 1, 1);

        ID3D11UnorderedAccessView *null_uav1 = nullptr;
        context->CSSetUnorderedAccessViews(0, 1, &null_uav1, nullptr);
        context->CSSetShaderResources(0, 2, null_transform_srvs);

        // Pass A2 (percentile mode): 256-bin histogram over the raw range, so pass B
        // can replace the outlier-sensitive min/max with robust percentile bounds.
        if (depth_hist_cs && hist_uav) {
          context->CSSetShader(depth_hist_cs.Get(), nullptr, 0);
          context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
          ID3D11ShaderResourceView *hist_srvs[2] = {
            tensor_out_srv.Get(),
            current_transform
          };
          context->CSSetShaderResources(0, 2, hist_srvs);
          ID3D11UnorderedAccessView *hist_uavs[2] = {hist_uav.Get(), minmax_raw_uav.Get()};
          context->CSSetUnorderedAccessViews(0, 2, hist_uavs, nullptr);
          context->Dispatch(reduce_groups, 1, 1);

          ID3D11UnorderedAccessView *null_uavs_h[2] = {nullptr, nullptr};
          context->CSSetUnorderedAccessViews(0, 2, null_uavs_h, nullptr);
          context->CSSetShaderResources(0, 2, null_transform_srvs);
        }

        // Pass B: fold into the EMA'd bounds and reset the accumulators (1 thread).
        context->CSSetShader(depth_minmax_ema_cs.Get(), nullptr, 0);
        ID3D11ShaderResourceView *ema_fold_srvs[2] = {
          current_transform,
          previous_surface_transform
        };
        context->CSSetShaderResources(0, 2, ema_fold_srvs);
        ID3D11UnorderedAccessView *ema_uavs[4] = {
          minmax_ema_uav.Get(),
          minmax_raw_uav.Get(),
          hist_uav.Get(),
          subject_uav.Get(),
        };
        context->CSSetUnorderedAccessViews(0, 4, ema_uavs, nullptr);
        context->Dispatch(1, 1, 1);

        ID3D11UnorderedAccessView *null_uav2[4] = {nullptr, nullptr, nullptr, nullptr};
        context->CSSetUnorderedAccessViews(0, 4, null_uav2, nullptr);
        context->CSSetShaderResources(0, 2, null_transform_srvs);
      }

      // Snapshot the complete previous depth before any thread writes the new result.
      context->CopyResource(depth_previous_tex.Get(), depth_tex.Get());

      const UINT clear_mask[4] = {0u, 0u, 0u, 0u};
      if (ema_edge_change > 0.0f && ema_edge_gradient > 0.0f) {
        context->CSSetShader(depth_ema_motion_cs.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
        ID3D11ShaderResourceView *mask_srvs[5] = {
          tensor_out_srv.Get(),
          minmax_ema_srv.Get(),
          depth_previous_srv.Get(),
          current_transform,
          previous_surface_transform
        };
        context->CSSetShaderResources(0, 5, mask_srvs);
        context->CSSetUnorderedAccessViews(0, 1, ema_motion_mask_uav.GetAddressOf(), nullptr);
        context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);
        ID3D11UnorderedAccessView *null_mask_uav = nullptr;
        ID3D11ShaderResourceView *null_mask_srvs[5] = {
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr
        };
        context->CSSetUnorderedAccessViews(0, 1, &null_mask_uav, nullptr);
        context->CSSetShaderResources(0, 5, null_mask_srvs);
      } else {
        context->ClearUnorderedAccessViewUint(ema_motion_mask_uav.Get(), clear_mask);
      }

      // 3b. Buffer to Texture: normalize disparity and either apply temporal EMA or snap the
      // pixels selected by the deterministic moving-edge mask. MinMaxEma.frame_state makes the
      // first valid frame snap and makes an all-invalid frame hold entirely on the GPU.
      context->CSSetShader(buffer_to_tex_cs.Get(), nullptr, 0);
      context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
      ID3D11ShaderResourceView *bt_srvs[6] = {
        tensor_out_srv.Get(),
        minmax_ema_srv.Get(),
        depth_previous_srv.Get(),
        ema_motion_mask_srv.Get(),
        current_transform,
        previous_surface_transform
      };
      context->CSSetShaderResources(0, 6, bt_srvs);
      ID3D11UnorderedAccessView *bt_uavs[2] = {
        depth_uav.Get(),
        owned_transform_ready ?
          depth_surface_transform_gpu[
            current_depth_surface_transform_bank ^ 1u
          ].uav.Get() :
          nullptr
      };
      context->CSSetUnorderedAccessViews(0, 2, bt_uavs, nullptr);

      context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

      ID3D11UnorderedAccessView *null_uav2[2] = {nullptr, nullptr};
      ID3D11ShaderResourceView *null_bt_srvs[6] = {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
      };
      context->CSSetUnorderedAccessViews(0, 2, null_uav2, nullptr);
      context->CSSetShaderResources(0, 6, null_bt_srvs);
      if (owned_transform_ready) {
        // buffer_to_tex copies either the valid current transform or the retained previous
        // transform into the next bank in the same dispatch that writes/holds depth_tex.
        current_depth_surface_transform_bank ^= 1u;
      }

      // 3s. Subject tracking: weighted depth histogram over the freshly-normalized
      // depth, then a 1-thread resolve into the subject state the reprojection reads.
      {
        context->CSSetShader(depth_subject_hist_cs.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
        ID3D11ShaderResourceView *subject_srvs[11] = {
          depth_srv.Get(),
          depth_cut_history_srv.Get(),
          tensor_in_srv.Get(),
          tensor_previous_input_srv.Get(),
          minmax_ema_srv.Get(),
          appearance_ordinal_srv.Get(),
          previous_appearance_ordinal_srv.Get(),
          current_transform,
          reliable_transform,
          tensor_out_srv.Get(),
          depth_reliable_validity_srv.Get()
        };
        context->CSSetShaderResources(0, 11, subject_srvs);
        ID3D11UnorderedAccessView *hist_uavs[2] = {subject_hist_uav.Get(), subject_plain_uav.Get()};
        context->CSSetUnorderedAccessViews(0, 2, hist_uavs, nullptr);
        context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

        ID3D11UnorderedAccessView *null_uavs_h2[2] = {nullptr, nullptr};
        context->CSSetUnorderedAccessViews(0, 2, null_uavs_h2, nullptr);
        ID3D11ShaderResourceView *null_subject_srvs[11] = {
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr
        };
        context->CSSetShaderResources(0, 11, null_subject_srvs);

        context->CSSetShader(depth_subject_resolve_cs.Get(), nullptr, 0);
        ID3D11ShaderResourceView *subject_resolve_srvs[2] = {
          current_transform,
          reliable_transform
        };
        context->CSSetShaderResources(0, 2, subject_resolve_srvs);
        ID3D11UnorderedAccessView *subj_uavs[3] = {subject_hist_uav.Get(), subject_uav.Get(), subject_plain_uav.Get()};
        context->CSSetUnorderedAccessViews(0, 3, subj_uavs, nullptr);
        context->Dispatch(1, 1, 1);

        ID3D11UnorderedAccessView *null_uavs2[3] = {nullptr, nullptr, nullptr};
        context->CSSetUnorderedAccessViews(0, 3, null_uavs2, nullptr);
        context->CSSetShaderResources(0, 2, null_transform_srvs);

        // rules_v1 must consume the completed frame HERE. Its retained scene/analysis buffers and
        // the depth resources still own pending_frame_id, and the reliable history below has not
        // yet advanced. Running after estimate_depth() returns would pair the rules with the newly
        // preprocessed source frame.
        if (scene_controller) {
          mark_d3d_scene_rules_start(d3d_timer);
          const bool controller_resolved =
            scene_controller->resolve_completed(
            pending_frame_id,
            tensor_out_srv.Get(),
            depth_srv.Get(),
            minmax_ema_srv.Get(),
            subject_srv.Get(),
            target_w,
            target_h,
            current_transform
          );
          if (controller_resolved && roi_shape_request_gpu) {
            const auto controller_snapshot =
              scene_controller->snapshot();
            const bool exact_controller_owner =
              controller_snapshot.snapshot_available &&
              controller_snapshot.source_frame_id ==
                completed_roi_identity.source_frame_id &&
              controller_snapshot.backend_generation ==
                completed_roi_identity.backend_generation;
            if (exact_controller_owner) {
              const auto submission =
                make_roi_shape_submission(
                  controller_snapshot,
                  completed_roi_identity.source_width,
                  completed_roi_identity.source_height
                );
              if (!submission) {
                if (active_roi_authority) {
                  roi_canonical_recovery_requested = true;
                }
              } else {
                const auto shape_result =
                  roi_shape_request_gpu->submit(*submission);
                bool confirm_destructive_shape_change = false;

                // Poll completion and current scheduling are orthogonal. A valid older readback
                // remains evidence even if this call's new dispatch/view/device operation failed.
                if (
                  shape_result.fresh_sample &&
                  shape_result.request &&
                  roi_shape_request_matches_submission(
                    *shape_result.request,
                    *submission
                  )
                ) {
                  if (!active_roi_authority) {
                    newest_roi_shape_request =
                      *shape_result.request;
                  } else {
                    const auto &request =
                      *shape_result.request;
                    newest_roi_shape_request = request;
                    switch (sbs_roi_shape_sample_transition(
                      shape_result.completed_source_frame_id,
                      controller_snapshot.source_frame_id,
                      request.shape[2],
                      request.shape[3],
                      static_cast<std::uint32_t>(target_w),
                      static_cast<std::uint32_t>(target_h)
                    )) {
                      case sbs_roi_shape_sample_action::
                          continue_current_shape:
                        // The builder still requires the same ROI generation/geometry before this
                        // delayed request can activate a crop, so no CPU confirmation is needed.
                        break;
                      case sbs_roi_shape_sample_action::
                          apply_exact_shape_change:
                        pending_roi_shape_transition = request;
                        break;
                      case sbs_roi_shape_sample_action::
                          confirm_shape_change:
                        confirm_destructive_shape_change = true;
                        break;
                    }
                  }
                }

                if (
                  active_roi_authority &&
                  confirm_destructive_shape_change
                ) {
                  // Freeze only when a delayed request proposes destructive tensor teardown. The
                  // exact current controller frame must independently confirm that shape.
                  if (!roi_shape_confirmation.begin(
                        controller_snapshot.source_frame_id,
                        completed_roi_identity.source_width,
                        completed_roi_identity.source_height,
                        shape_result.copy_scheduled
                      )) {
                    roi_canonical_recovery_requested = true;
                  }
                }

                if (
                  shape_result.failed &&
                  !roi_shape_request_error_logged
                ) {
                  BOOST_LOG(warning)
                    << "Host SBS ROI shape-request sampling encountered an infrastructure "
                       "failure; a valid completed sample, if present, was retained.";
                  roi_shape_request_error_logged = true;
                }
              }
            } else if (!roi_shape_request_error_logged) {
              BOOST_LOG(warning)
                << "Host SBS ROI shape-request sampling rejected a "
                   "controller/depth ownership mismatch; shadow rendering remains unchanged.";
              roi_shape_request_error_logged = true;
              if (active_roi_authority) {
                roi_canonical_recovery_requested = true;
              }
            }
          }
          if (
            controller_resolved &&
            active_roi_authority &&
            !roi_shape_request_gpu
          ) {
            roi_canonical_recovery_requested = true;
          }
          mark_d3d_scene_rules_end(
            d3d_timer,
            controller_resolved
          );
          if (!controller_resolved && !scene_controller_error_logged) {
            BOOST_LOG(warning)
              << "Host SBS scene-controller rejected a completed matched frame; "
                 "shadow output is invalid and the full-frame render remains authoritative.";
            scene_controller_error_logged = true;
          }
          if (!controller_resolved && active_roi_authority) {
            roi_canonical_recovery_requested = true;
          }
        }

        // tensor_in_buf, appearance_ordinal_buf and depth_tex still own the matched inputs/result
        // for this completed inference. Advance the complete appearance/depth tuple only when the
        // resolve pass selects state 1 or 3. State 2 retains the last structurally reliable tuple
        // for one black/clipped update, so an immediate supported return compares A against A or B
        // rather than the empty slate. State 3 advances an accepted persistent-low endpoint.
        context->CSSetShader(depth_valid_history_cs.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
        ID3D11ShaderResourceView *history_srvs[7] = {
          minmax_ema_srv.Get(),
          tensor_in_srv.Get(),
          appearance_ordinal_srv.Get(),
          subject_srv.Get(),
          depth_srv.Get(),
          tensor_out_srv.Get(),
          current_transform
        };
        ID3D11UnorderedAccessView *history_uavs[5] = {
          tensor_previous_input_uav.Get(),
          previous_appearance_ordinal_uav.Get(),
          depth_cut_history_uav.Get(),
          reliable_roi_transform_gpu.uav.Get(),
          depth_reliable_validity_uav.Get()
        };
        context->CSSetShaderResources(0, 7, history_srvs);
        context->CSSetUnorderedAccessViews(0, 5, history_uavs, nullptr);
        context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);
        ID3D11ShaderResourceView *null_history_srvs[7] = {
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr
        };
        ID3D11UnorderedAccessView *null_history_uavs[5] = {
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr
        };
        context->CSSetShaderResources(0, 7, null_history_srvs);
        context->CSSetUnorderedAccessViews(0, 5, null_history_uavs, nullptr);
      }
    }

    // Diagnostics-only accounting for achieved inference throughput and busy drops. Callers
    // bypass this function entirely when diagnostics are disabled, avoiding even a clock read.
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
          BOOST_LOG(info) << "Depth throughput: source ~" << (int) (measured_fps + 0.5f)
                          << "fps, completed ~" << (int) (throughput_stats_completions / stats_seconds + 0.5f)
                          << "fps, enqueued ~" << (int) (throughput_stats_enqueues / stats_seconds + 0.5f)
                          << "fps, busy drops " << (int) (100.0f * throughput_stats_busy_drops / calls + 0.5f)
                          << "% (" << throughput_stats_busy_drops << '/' << throughput_stats_calls << ')';
          throughput_stats_start = now;
          throughput_stats_calls = 0;
          throughput_stats_busy_drops = 0;
          throughput_stats_enqueues = 0;
          throughput_stats_completions = 0;
        }
      }
      throughput_stats_calls++;
    }

    // Query-only producer preflight. It deliberately leaves has_previous_frame and the finished
    // output buffer untouched; estimate() consumes that result after the caller has copied the
    // exact color frame that will own the next inference.
    bool can_accept() {
      if (!valid) {
        return false;
      }
      auto &cuda = cuda_driver_api::get();
      if (!cuda.is_valid() || !cu_stream || !cuda.cuStreamQuery) {
        valid = false;
        observe_context_event(
          !cuda.is_valid() ?
            sbs_trt_context_event::cuda_context_failure :
            sbs_trt_context_event::cuda_stream_failure
        );
        return false;
      }
      if (
        cuda_ctx &&
        (
          !cuda.cuCtxSetCurrent ||
          cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS
        )
      ) {
        abandon_pending_after_cuda_error(
          "cuCtxSetCurrent failed during depth readiness preflight",
          sbs_trt_context_event::cuda_context_failure
        );
        return false;
      }

      if (diagnostics_enabled) {
        update_throughput_stats();
      }
      const auto query = cuda.cuStreamQuery(cu_stream);
      if (query == CUDA_ERROR_NOT_READY) {
        if (diagnostics_enabled) {
          throughput_stats_busy_drops++;
        }
        readiness_preflighted = false;
        return false;
      }
      if (query != CUDA_SUCCESS) {
        BOOST_LOG(error)
          << "cuStreamQuery failed during depth readiness preflight: " << query;
        abandon_pending_after_cuda_error(
          "cuStreamQuery returned a terminal error during readiness preflight"
        );
        readiness_preflighted = false;
        return false;
      }
      readiness_preflighted = true;
      return true;
    }

    estimate_result estimate(
      ID3D11ShaderResourceView *input_srv,
      input_color_space color_space,
      std::uint64_t frame_id,
      bool snapshot_raw_model_depth
    ) {
      if (!valid || !input_srv) {
        return {};
      }
      bool completed_frame_valid = false;
      std::uint64_t completed_frame_id = 0;
      bool completion_dropped = false;
      std::uint64_t dropped_frame_id = 0;
      bool raw_snapshot_valid = false;
      bool model_input_snapshot_valid = false;
      bool scene_controller_prepared = false;

      auto &cuda = cuda_driver_api::get();
      if (!cuda.is_valid()) {
        BOOST_LOG(error) << "CUDA Driver API is not available.";
        valid = false;
        observe_context_event(
          sbs_trt_context_event::cuda_context_failure
        );
        return {};
      }

      if (cuda_ctx) {
        const auto set_current = cuda.cuCtxSetCurrent ?
                                   cuda.cuCtxSetCurrent(cuda_ctx) :
                                   static_cast<CUresult>(-1);
        if (set_current != CUDA_SUCCESS) {
          BOOST_LOG(error)
            << "cuCtxSetCurrent failed during depth estimation: "
            << set_current;
          const auto dropped =
            abandon_pending_after_cuda_error(
              "cuCtxSetCurrent failed during depth estimation",
              sbs_trt_context_event::cuda_context_failure
            );
          return make_result(
            false,
            0,
            false,
            false,
            false,
            dropped.has_value(),
            dropped.value_or(0)
          );
        }
      }

      // Production preflights before its expensive full-resolution color copy. The evaluator and
      // any direct callers do not, so retain the self-contained query/counting path here.
      const bool preflighted = std::exchange(readiness_preflighted, false);
      if (!preflighted && diagnostics_enabled) {
        update_throughput_stats();
      }

      // Resolve completed inference-timing events only for diagnostic runs.
      if (diagnostics_enabled) {
        perf_drain(perf_depth);
      }

      // Prevent GPU starvation: if the previous AI frame is still crunching, drop this frame.
      // This prevents an infinite queue of heavy TensorRT workloads from starving the DWM and Edge Browser.
      if (!preflighted && cu_stream && cuda.cuStreamQuery) {
        auto q = cuda.cuStreamQuery(cu_stream);
        if (q == CUDA_ERROR_NOT_READY) {
          // Reuse the last normalized depth and subject state while inference is busy.
          if (diagnostics_enabled) {
            throughput_stats_busy_drops++;
          }
          return make_result();
        }
        if (q != CUDA_SUCCESS && !stream_error_logged) {
          BOOST_LOG(error) << "cuStreamQuery failed: " << q;
          stream_error_logged = true;
        }
        if (q != CUDA_SUCCESS) {
          const auto dropped =
            abandon_pending_after_cuda_error(
              "cuStreamQuery returned a terminal error"
            );
          return make_result(
            false,
            0,
            false,
            false,
            false,
            dropped.has_value(),
            dropped.value_or(0)
          );
        }
      }

      D3D11_TEXTURE2D_DESC input_desc = {0};
      Microsoft::WRL::ComPtr<ID3D11Resource> input_res;
      input_srv->GetResource(&input_res);
      Microsoft::WRL::ComPtr<ID3D11Texture2D> input_tex;
      if (SUCCEEDED(input_res.As(&input_tex))) {
        input_tex->GetDesc(&input_desc);
      }

      if (
        active_roi_authority &&
        roi_shape_confirmation.awaiting() &&
        !has_previous_frame
      ) {
        const auto controller =
          scene_controller ?
            scene_controller->snapshot() :
            scene_controller_gpu_snapshot {};
        const bool frozen_owner_matches =
          controller.snapshot_available &&
          controller.source_frame_id ==
            roi_shape_confirmation.source_frame_id() &&
          input_desc.Width ==
            roi_shape_confirmation.source_width() &&
          input_desc.Height ==
            roi_shape_confirmation.source_height();
        const auto submission =
          frozen_owner_matches ?
            make_roi_shape_submission(
              controller,
              roi_shape_confirmation.source_width(),
              roi_shape_confirmation.source_height()
            ) :
            std::nullopt;

        if (!submission) {
          roi_canonical_recovery_requested = true;
        } else {
          const auto shape_result =
            roi_shape_confirmation.copy_scheduled() ?
              roi_shape_request_gpu->poll() :
              roi_shape_request_gpu->submit(*submission);
          const bool sample_valid =
            shape_result.fresh_sample &&
            shape_result.request &&
            roi_shape_request_matches_submission(
              *shape_result.request,
              *submission
            );
          const auto confirmation =
            roi_shape_confirmation.observe(
              shape_result.copy_scheduled,
              shape_result.fresh_sample,
              shape_result.completed_source_frame_id,
              sample_valid
            );
          if (
            confirmation ==
              sbs_roi_shape_confirmation_result::confirmed
          ) {
            const auto request = *shape_result.request;
            newest_roi_shape_request = request;
            if (
              request.shape[2] !=
                static_cast<std::uint32_t>(target_w) ||
              request.shape[3] !=
                static_cast<std::uint32_t>(target_h)
            ) {
              pending_roi_shape_transition = request;
            }
          } else if (
            confirmation ==
              sbs_roi_shape_confirmation_result::recover_canonical
          ) {
            BOOST_LOG(error)
              << "Host SBS ROI shape request did not confirm within "
              << sbs_roi_shape_confirmation_guard::
                   max_capture_opportunities
              << " capture opportunities; returning to canonical full-frame depth.";
            roi_canonical_recovery_requested = true;
          } else {
            // No inference is submitted while validation is outstanding. The controller
            // snapshot is frozen, so any eventual exact-frame completion proves the rule state
            // that authorizes the transition rather than a merely similar older request.
            return make_result();
          }
        }
      }

      if (
        sbs_roi_canonical_recovery_requires_rebuild(
          active_roi_authority,
          roi_canonical_recovery_requested
        ) &&
        !has_previous_frame
      ) {
        roi_shape_confirmation.reset();
        pending_roi_shape_transition.reset();
        newest_roi_shape_request.reset();
        // Crop-coordinate histories are shape-dependent even when the ROI happens to use the
        // canonical tensor dimensions. Always rebuild before withdrawing authority.
        if (!release_shape_resources_for_rebuild(cuda)) {
          valid = false;
          return {};
        }
        requested_target_w = canonical_target_w;
        requested_target_h = canonical_target_h;
        // Disable crop authority before the canonical enqueue. The explicit all-zero transform
        // then selects the validated legacy path, and future controller/helper failures cannot
        // strand an ROI-specific tensor shape.
        active_roi_authority = false;
        roi_canonical_recovery_requested = false;
        applied_roi_shape_request_id = 0u;
        BOOST_LOG(error)
          << "Host SBS active ROI authority was disabled for this stream; "
             "canonical full-frame depth remains available.";
      }

      if (
        active_roi_authority &&
        pending_roi_shape_transition &&
        !has_previous_frame
      ) {
        const auto request =
          *pending_roi_shape_transition;
        const sbs_roi_shape_request_limits limits {
          static_cast<std::uint32_t>(canonical_target_w),
          static_cast<std::uint32_t>(canonical_target_h),
          std::min(
            sbs_roi_shape_request_engine_max_dimension,
            input_desc.Width
          ),
          std::min(
            sbs_roi_shape_request_engine_max_dimension,
            input_desc.Height
          ),
          max_aspect,
        };
        const bool current_source_matches =
          request.shape[0] == input_desc.Width &&
          request.shape[1] == input_desc.Height;
        const bool request_valid =
          current_source_matches &&
          sbs_roi_shape_request_valid(request, limits);
        if (!request_valid) {
          pending_roi_shape_transition.reset();
          newest_roi_shape_request.reset();
          if (!roi_shape_request_error_logged) {
            BOOST_LOG(warning)
              << "Host SBS discarded an obsolete ROI shape request after "
                 "the source/session geometry changed.";
            roi_shape_request_error_logged = true;
          }
        } else if (
          request.shape[2] ==
            static_cast<std::uint32_t>(target_w) &&
          request.shape[3] ==
            static_cast<std::uint32_t>(target_h)
        ) {
          applied_roi_shape_request_id = request.header[3];
          pending_roi_shape_transition.reset();
        } else if (
          release_shape_resources_for_rebuild(cuda)
        ) {
          requested_target_w =
            static_cast<int>(request.shape[2]);
          requested_target_h =
            static_cast<int>(request.shape[3]);
          applied_roi_shape_request_id = request.header[3];
          pending_roi_shape_transition.reset();
          BOOST_LOG(info)
            << "Host SBS applying ROI model-shape transition "
            << requested_target_w << 'x' << requested_target_h
            << " for request " << request.header[3] << '.';
        } else {
          valid = false;
          return {};
        }
      }

      if (target_w == 0 || target_h == 0) {
        // The capture surface can report a 0x0 descriptor mid HDR/mode transition or
        // before the first real frame. Deriving the model resolution from that yields a
        // garbage size (NaN aspect -> integer-overflow -> clamps to the profile max) that would
        // be cached for the whole session. Wait for a valid frame instead.
        if (input_desc.Width == 0 || input_desc.Height == 0) {
          return {};
        }
        if (
          canonical_target_w == 0 ||
          canonical_target_h == 0
        ) {
          float aspect_ratio =
            (float) input_desc.Width /
            (float) input_desc.Height;
          // Keep the patch-aligned tensor as close as possible to source aspect while respecting
          // the TensorRT profile, configured inference-area budget, and native size.
          int max_w = std::min(
            models::depth_engine_max_dim,
            (int) input_desc.Width
          );
          int max_h = std::min(
            models::depth_engine_max_dim,
            (int) input_desc.Height
          );
          // `depth_max_aspect` is a pixel-budget envelope, not permission to squeeze the source
          // into a different tensor geometry. Wider/taller sources reduce the requested short
          // side by sqrt(cap/actual), preserving source aspect with no more inference pixels than
          // the former clamped-aspect shape.
          const auto fitted_dims =
            sbs_roi_full_frame_model_shape(
              aspect_ratio,
              static_cast<float>(depth_short_side),
              max_aspect,
              static_cast<std::uint32_t>(max_w),
              static_cast<std::uint32_t>(max_h)
            );
          if (fitted_dims[0] == 0u || fitted_dims[1] == 0u) {
            BOOST_LOG(error)
              << "Depth estimator cannot represent source aspect "
              << aspect_ratio
              << " with a patch-aligned full-frame tensor; streaming flat SBS.";
            valid = false;
            return {};
          }
          canonical_target_w = static_cast<int>(fitted_dims[0]);
          canonical_target_h = static_cast<int>(fitted_dims[1]);
        }
        target_w =
          requested_target_w > 0 ?
            requested_target_w :
            canonical_target_w;
        target_h =
          requested_target_h > 0 ?
            requested_target_h :
            canonical_target_h;

        // Threads for the min/max reduction; grid-stride handles any element count.
        int elems = target_w * target_h;
        reduce_groups = (UINT) std::min(64, std::max(1, (elems + 255) / 256));

        BOOST_LOG(info) << "Depth Estimator dynamic resolution set to " << target_w << "x" << target_h;

        const bool stale_input_released =
          unregister_cuda_interop_resource(
            cuda,
            cuda_in_res,
            "stale input"
          );
        const bool stale_output_released =
          unregister_cuda_interop_resource(
            cuda,
            cuda_out_res,
            "stale output"
          );
        if (!stale_input_released || !stale_output_released) {
          valid = false;
          observe_context_event(
            sbs_trt_context_event::cuda_interop_failure
          );
          return {};
        }

        D3D11_BUFFER_DESC buf_desc = {};
        buf_desc.Usage = D3D11_USAGE_DEFAULT;
        buf_desc.ByteWidth = target_w * target_h * 3 * sizeof(float);
        buf_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        buf_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        buf_desc.StructureByteStride = sizeof(float);
        bool resources_ok = SUCCEEDED(device->CreateBuffer(&buf_desc, nullptr, &tensor_in_buf)) &&
                            SUCCEEDED(device->CreateUnorderedAccessView(
                              tensor_in_buf.Get(),
                              nullptr,
                              &tensor_in_uav
                            )) &&
                            SUCCEEDED(device->CreateShaderResourceView(
                              tensor_in_buf.Get(),
                              nullptr,
                              &tensor_in_srv
                            ));

        auto previous_input_desc = buf_desc;
        previous_input_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateBuffer(&previous_input_desc, nullptr, &tensor_previous_input_buf)) &&
                       SUCCEEDED(device->CreateShaderResourceView(
                         tensor_previous_input_buf.Get(),
                         nullptr,
                         &tensor_previous_input_srv
                       )) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(
                         tensor_previous_input_buf.Get(),
                         nullptr,
                         &tensor_previous_input_uav
                       ));

        // One capture-domain point maxRGB scalar per model texel. Keeping it outside TensorRT's
        // three-plane input preserves a pre-tone-map exposure ordinal without changing the engine
        // binding size. Current/history are copied in lockstep with the valid NCHW/depth pair.
        auto appearance_desc = buf_desc;
        appearance_desc.ByteWidth = target_w * target_h * sizeof(float);
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateBuffer(
                         &appearance_desc,
                         nullptr,
                         &appearance_ordinal_buf
                       )) &&
                       SUCCEEDED(device->CreateShaderResourceView(
                         appearance_ordinal_buf.Get(),
                         nullptr,
                         &appearance_ordinal_srv
                       )) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(
                         appearance_ordinal_buf.Get(),
                         nullptr,
                         &appearance_ordinal_uav
                       )) &&
                       SUCCEEDED(device->CreateBuffer(
                         &appearance_desc,
                         nullptr,
                         &previous_appearance_ordinal_buf
                       )) &&
                       SUCCEEDED(device->CreateShaderResourceView(
                         previous_appearance_ordinal_buf.Get(),
                         nullptr,
                         &previous_appearance_ordinal_srv
                       )) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(
                         previous_appearance_ordinal_buf.Get(),
                         nullptr,
                         &previous_appearance_ordinal_uav
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

        // Immutable previous-depth snapshot for motion-edge classification and EMA input. Keep
        // the history SRV separate from the depth UAV that receives the current frame.
        auto previous_desc = tex_desc;
        previous_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateTexture2D(&previous_desc, nullptr, &depth_previous_tex)) &&
                       SUCCEEDED(device->CreateShaderResourceView(depth_previous_tex.Get(), nullptr, &depth_previous_srv));

        auto cut_history_desc = tex_desc;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateTexture2D(
                         &cut_history_desc,
                         nullptr,
                         &depth_cut_history_tex
                       )) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(
                         depth_cut_history_tex.Get(),
                         nullptr,
                         &depth_cut_history_uav
                       )) &&
                       SUCCEEDED(device->CreateShaderResourceView(
                         depth_cut_history_tex.Get(),
                         nullptr,
                         &depth_cut_history_srv
                       ));

        auto mask_desc = tex_desc;
        mask_desc.Format = DXGI_FORMAT_R32_UINT;
        mask_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateTexture2D(&mask_desc, nullptr, &ema_motion_mask_tex)) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(ema_motion_mask_tex.Get(), nullptr, &ema_motion_mask_uav)) &&
                       SUCCEEDED(device->CreateShaderResourceView(ema_motion_mask_tex.Get(), nullptr, &ema_motion_mask_srv)) &&
                       SUCCEEDED(device->CreateTexture2D(
                         &mask_desc,
                         nullptr,
                         &depth_reliable_validity_tex
                       )) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(
                         depth_reliable_validity_tex.Get(),
                         nullptr,
                         &depth_reliable_validity_uav
                       )) &&
                       SUCCEEDED(device->CreateShaderResourceView(
                         depth_reliable_validity_tex.Get(),
                         nullptr,
                         &depth_reliable_validity_srv
                       ));

        if (!resources_ok) {
          BOOST_LOG(error) << "Depth estimator D3D11 resource creation failed; retrying on a later frame.";
          target_w = target_h = 0;
          ++shape_resource_allocation_failures;
          if (shape_resource_allocation_failures >= 3u) {
            if (
              active_roi_authority &&
              (
                requested_target_w != canonical_target_w ||
                requested_target_h != canonical_target_h
              )
            ) {
              roi_canonical_recovery_requested = true;
            } else {
              valid = false;
            }
          }
          return {};
        }

        // Clear depth so the range->pixel EMA initializes from a known value.
        const float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        context->ClearUnorderedAccessViewFloat(
          tensor_previous_input_uav.Get(),
          clear_color
        );
        context->ClearUnorderedAccessViewFloat(
          previous_appearance_ordinal_uav.Get(),
          clear_color
        );
        context->ClearUnorderedAccessViewFloat(depth_uav.Get(), clear_color);
        context->ClearUnorderedAccessViewFloat(depth_cut_history_uav.Get(), clear_color);
        const UINT clear_uint[4] = {0u, 0u, 0u, 0u};
        context->ClearUnorderedAccessViewUint(ema_motion_mask_uav.Get(), clear_uint);
        context->ClearUnorderedAccessViewUint(
          depth_reliable_validity_uav.Get(),
          clear_uint
        );

        auto res1 = cuda.cuGraphicsD3D11RegisterResource(&cuda_in_res, tensor_in_buf.Get(), 0);
        auto res2 = cuda.cuGraphicsD3D11RegisterResource(&cuda_out_res, tensor_out_buf.Get(), 0);
        if (res1 != 0 || res2 != 0) {
          BOOST_LOG(error) << "cuGraphicsD3D11RegisterResource failed: " << res1 << ", " << res2;
          const bool input_cleanup =
            unregister_cuda_interop_resource(
              cuda,
              cuda_in_res,
              "partially registered input"
            );
          const bool output_cleanup =
            unregister_cuda_interop_resource(
              cuda,
              cuda_out_res,
              "partially registered output"
            );
          target_w = target_h = 0;
          if (!input_cleanup || !output_cleanup) {
            valid = false;
            observe_context_event(
              sbs_trt_context_event::cuda_interop_failure
            );
          } else {
            ++shape_resource_allocation_failures;
            if (shape_resource_allocation_failures >= 3u) {
              if (
                active_roi_authority &&
                (
                  requested_target_w != canonical_target_w ||
                  requested_target_h != canonical_target_h
                )
              ) {
                roi_canonical_recovery_requested = true;
              } else {
                valid = false;
              }
            }
          }
          return {};
        }
        shape_resource_allocation_failures = 0u;
        requested_target_w = 0;
        requested_target_h = 0;
      }

      // Shared constants for buffer_to_tex_cs, the min/max passes and rgb_to_nchw_cs.
      // Session-constant, so the buffer is built once (immutable), not mapped per frame.
      ensure_cbuffers(color_space);
      if (!cbuffer) {
        return {};
      }

      std::optional<frame_roi_transform_identity> completed_roi_identity;
      if (has_previous_frame) {
        completed_frame_id = pending_frame_id;
        const bool explicitly_orphaned =
          active_roi_authority &&
          roi_transform_slots.orphaned_for(completed_frame_id);
        completed_roi_identity =
          claim_depth_completion(completed_frame_id);
        if (!completed_roi_identity) {
          drop_depth_completion(
            completed_frame_id,
            "its exact pending ROI-transform identity was absent or mismatched"
          );
          completion_dropped = true;
          dropped_frame_id = completed_frame_id;
          // A known post-accept orphan can recover after being drained. Any other ownership
          // mismatch is internal corruption, so stop before reusing either bank.
          disable_after_pending_drop =
            disable_after_pending_drop || !explicitly_orphaned;
        }
        // The raw output is consumed exactly once whether it is normalized or dropped.
        has_previous_frame = false;
      }

      auto *d3d_timer = diagnostics_enabled ?
                          begin_d3d_perf(
                            completed_roi_identity.has_value(),
                            true
                          ) :
                          nullptr;

      // tensor_out_buf holds the finished raw disparity from the previous asynchronous submit
      // (fully unmapped from CUDA), so consuming it here never blocks the encode thread. The
      // caller uses completed_frame_id to select the color slot that produced this exact result.
      if (completed_roi_identity) {
        normalize_depth_output(*completed_roi_identity, d3d_timer);
        // Production post-process timing ends at the normalized depth result. The two stable
        // Dump 3D copies below are explicit diagnostic work and must not contaminate live
        // depth_postprocess_gpu samples.
        mark_d3d_post_end(d3d_timer);
        if (snapshot_raw_model_depth) {
          // tensor_in_buf and tensor_out_buf still own the exact completed frame here. Preserve
          // both before the current frame's preprocess and CUDA mapping reuse those allocations.
          raw_snapshot_valid = snapshot_buffer(
            tensor_out_buf.Get(),
            raw_snapshot_buf,
            raw_snapshot_srv,
            raw_snapshot_error_logged,
            raw_snapshot_retry_frames,
            "raw-depth"
          );
          model_input_snapshot_valid = snapshot_buffer(
            tensor_in_buf.Get(),
            model_input_snapshot_buf,
            model_input_snapshot_srv,
            model_input_snapshot_error_logged,
            model_input_snapshot_retry_frames,
            "model-input"
          );
        }
        completed_frame_valid = true;
        if (diagnostics_enabled) {
          throughput_stats_completions++;
        }
      } else {
        mark_d3d_post_end(d3d_timer);
      }

      if (completion_dropped && disable_after_pending_drop) {
        disable_after_pending_drop = false;
        valid = false;
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return make_result(
          false,
          0,
          false,
          false,
          false,
          true,
          dropped_frame_id
        );
      }

      if (
        active_roi_authority &&
        (
          pending_roi_shape_transition ||
          roi_shape_confirmation.awaiting() ||
          roi_canonical_recovery_requested
        )
      ) {
        // This call has just materialized the completion that owns the returned color slot.
        // Return it unchanged and skip the current enqueue while an exact-frame request is
        // confirmed, a shape transition is applied, or canonical recovery disables authority.
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return make_result(
          completed_frame_valid,
          completed_frame_id,
          false,
          raw_snapshot_valid,
          model_input_snapshot_valid,
          completion_dropped,
          dropped_frame_id
        );
      }

      // Active ROI geometry/generation is written into a selected GPU bank by the controller.
      // Off/shadow mode keeps only a CPU provenance record and binds the immutable zero transform;
      // it never depends on this experimental dispatch or its dynamic cbuffer Map.
      std::uint32_t roi_backend_generation = 0;
      if (scene_controller) {
        roi_backend_generation =
          scene_controller->snapshot().backend_generation;
      }
      std::optional<frame_roi_transform_identity> reserved_roi_transform;
      frame_roi_transform_identity legacy_frame_identity;
      legacy_frame_identity.source_frame_id = frame_id;
      legacy_frame_identity.backend_generation =
        roi_backend_generation;
      legacy_frame_identity.source_width = input_desc.Width;
      legacy_frame_identity.source_height = input_desc.Height;
      legacy_frame_identity.model_width =
        static_cast<std::uint32_t>(target_w);
      legacy_frame_identity.model_height =
        static_cast<std::uint32_t>(target_h);
      if (active_roi_authority) {
        if (const auto writable_bank =
              roi_transform_slots.writable_bank()) {
          reserved_roi_transform = roi_transform_slots.reserve(
            make_frame_roi_transform_identity(
              frame_id,
              input_desc.Width,
              input_desc.Height,
              static_cast<std::uint32_t>(target_w),
              static_cast<std::uint32_t>(target_h),
              roi_backend_generation,
              *writable_bank
            )
          );
        }
      }
      if (active_roi_authority && !reserved_roi_transform) {
        if (!roi_transform_error_logged) {
          BOOST_LOG(error)
            << "Depth estimator could not reserve a frame-owned ROI transform bank for frame "
            << frame_id << "; inference was not submitted.";
          roi_transform_error_logged = true;
        }
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return make_result(
          completed_frame_valid,
          completed_frame_id,
          false,
          raw_snapshot_valid,
          model_input_snapshot_valid,
          completion_dropped,
          dropped_frame_id
        );
      }
      const auto rollback_roi_reservation = [&]() {
        if (!active_roi_authority || !reserved_roi_transform) {
          return;
        }
        if (!roi_transform_slots.rollback_reserved(*reserved_roi_transform)) {
          roi_transform_slots.abandon_in_flight();
          if (!roi_transform_error_logged) {
            BOOST_LOG(error)
              << "Depth estimator could not roll back the unaccepted ROI transform for frame "
              << frame_id << "; all in-flight ROI ownership was discarded.";
            roi_transform_error_logged = true;
          }
        }
      };

      if (
        active_roi_authority &&
        !dispatch_frame_roi_transform(*reserved_roi_transform)
      ) {
        if (!roi_transform_error_logged) {
          BOOST_LOG(error)
            << "Depth estimator could not build the GPU ROI transform for frame "
            << frame_id << "; inference was not submitted.";
          roi_transform_error_logged = true;
        }
        rollback_roi_reservation();
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return make_result(
          completed_frame_valid,
          completed_frame_id,
          false,
          raw_snapshot_valid,
          model_input_snapshot_valid,
          completion_dropped,
          dropped_frame_id
        );
      }

      // 1. D3D11 Compute Shader: Resize & Normalize to NCHW FP32 Buffer (for CURRENT frame)
      mark_d3d_pre_start(d3d_timer);
      context->CSSetShader(rgb_to_nchw_cs.Get(), nullptr, 0);
      context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
      ID3D11ShaderResourceView *preprocess_srvs[2] = {
        input_srv,
        active_roi_authority ?
          frame_roi_transform_gpu[
            reserved_roi_transform->gpu_bank_index
          ].srv.Get() :
          zero_roi_transform.srv.Get()
      };
      context->CSSetShaderResources(0, 2, preprocess_srvs);
      ID3D11UnorderedAccessView *preprocess_uavs[2] = {
        tensor_in_uav.Get(),
        appearance_ordinal_uav.Get()
      };
      context->CSSetUnorderedAccessViews(0, 2, preprocess_uavs, nullptr);
      context->CSSetSamplers(0, 1, linear_sampler.GetAddressOf());

      context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

      ID3D11UnorderedAccessView *null_uavs[2] = {nullptr, nullptr};
      ID3D11ShaderResourceView *null_preprocess_srvs[2] = {
        nullptr,
        nullptr
      };
      context->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
      context->CSSetShaderResources(0, 2, null_preprocess_srvs);
      mark_d3d_pre_end(d3d_timer);
      if (scene_controller) {
        mark_d3d_scene_prepare_start(d3d_timer);
        scene_controller_prepared = scene_controller->prepare_scene(
          input_srv,
          color_space,
          frame_id
        );
        mark_d3d_scene_prepare_end(
          d3d_timer,
          scene_controller_prepared
        );
        if (!scene_controller_prepared && !scene_controller_error_logged) {
          BOOST_LOG(warning)
            << "Host SBS scene-controller could not retain the matched source frame; "
               "shadow state will hold and the full-frame render remains authoritative.";
          scene_controller_error_logged = true;
        }
      }
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
        valid = false;
        observe_context_event(
          sbs_trt_context_event::cuda_interop_failure
        );
        rollback_roi_reservation();
        if (scene_controller && scene_controller_prepared) {
          scene_controller->discard_prepared(frame_id);
        }
        return make_result(
          completed_frame_valid,
          completed_frame_id,
          false,
          raw_snapshot_valid,
          model_input_snapshot_valid,
          completion_dropped,
          dropped_frame_id
        );
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
      bool input_shape_rejected = false;
      bool tensor_address_rejected = false;
      const bool mapped_pointer_rejected =
        in_ptr_res != CUDA_SUCCESS ||
        out_ptr_res != CUDA_SUCCESS ||
        !d_in || !d_out;
      if (mapped_pointer_rejected) {
        BOOST_LOG(error) << "Failed to get mapped pointer for TensorRT: "
                         << in_ptr_res << ", " << out_ptr_res;
        valid = false;
        observe_context_event(
          sbs_trt_context_event::cuda_interop_failure
        );
      } else {
        bool bindings_ok = true;
        if (
          configured_input_width != target_w ||
          configured_input_height != target_h
        ) {
          nvinfer1::Dims in_dims = make_input_dims(target_h, target_w);
          bindings_ok = exec_context->setInputShape("pixel_values", in_dims);
          if (!bindings_ok) {
            input_shape_rejected = true;
            BOOST_LOG(error) << "TensorRT setInputShape failed for " << target_w << "x" << target_h
                             << " (outside the engine's optimization profile? request "
                             << applied_roi_shape_request_id << ')';
          } else {
            configured_input_width = target_w;
            configured_input_height = target_h;
          }
        }
        if (bindings_ok) {
          tensor_address_rejected =
            !exec_context->setTensorAddress("pixel_values", (void *) d_in) ||
            !exec_context->setTensorAddress("predicted_depth", (void *) d_out);
          bindings_ok = !tensor_address_rejected;
          if (tensor_address_rejected && !stream_error_logged) {
            BOOST_LOG(error)
              << "TensorRT rejected an input/output tensor address; "
                 "retiring this execution context.";
            stream_error_logged = true;
          }
        }
        // Active inference requires the exact versioned GPU-bank reservation. Off/shadow uses the
        // immutable legacy transform and therefore has no experimental ownership dependency.
        const bool roi_transform_ready =
          !active_roi_authority ||
          (
            reserved_roi_transform &&
            roi_transform_slots.is_reserved(
              *reserved_roi_transform
            )
          );
        if (!roi_transform_ready && !roi_transform_error_logged) {
          BOOST_LOG(error)
            << "Depth estimator has no valid frame-owned ROI transform slot for frame "
            << frame_id << "; inference was not submitted.";
          roi_transform_error_logged = true;
        }
        bindings_ok = bindings_ok && roi_transform_ready;
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
            observe_context_event(
              sbs_trt_context_event::enqueue_rejection
            );
            valid = false;
            if (!stream_error_logged) {
              BOOST_LOG(error)
                << "TensorRT enqueueV3 failed; retiring this execution context "
                   "and retaining the last valid depth.";
              stream_error_logged = true;
            }
          }
          perf_end(perf_depth, perf_slot, cu_stream);
        }
      }

      const bool inference_accepted = enqueued;
      const auto unmap_res =
        cuda.cuGraphicsUnmapResources(2, resources, cu_stream);
      const bool unmap_succeeded = unmap_res == CUDA_SUCCESS;
      if (!unmap_succeeded) {
        BOOST_LOG(error) << "cuGraphicsUnmapResources failed: " << unmap_res;
        observe_context_event(
          sbs_trt_context_event::cuda_interop_failure
        );
        valid = false;
      }

      if (input_shape_rejected) {
        const auto failure_action =
          sbs_roi_shape_binding_failure(
            active_roi_authority,
            static_cast<std::uint32_t>(
              std::max(target_w, 0)
            ),
            static_cast<std::uint32_t>(
              std::max(target_h, 0)
            ),
            static_cast<std::uint32_t>(
              std::max(canonical_target_w, 0)
            ),
            static_cast<std::uint32_t>(
              std::max(canonical_target_h, 0)
            )
          );
        if (
          failure_action ==
            sbs_roi_shape_binding_failure_action::
              recover_canonical
        ) {
          observe_context_event(
            sbs_trt_context_event::
              recoverable_dynamic_shape_rejection
          );
          // The current resources represent a shape TensorRT refused. The next idle call tears
          // them down, restores canonical dimensions, and disables ROI authority for this stream.
          roi_canonical_recovery_requested = true;
        } else {
          // Failure of the canonical shape means no safe depth configuration remains. The display
          // observes !is_valid(), retires this estimator, and continues advancing flat SBS color.
          valid = false;
          observe_context_event(
            sbs_trt_context_event::canonical_shape_rejection
          );
        }
      }

      if (tensor_address_rejected) {
        valid = false;
        observe_context_event(
          sbs_trt_context_event::tensor_address_rejection
        );
      }

      bool roi_ownership_pending = false;
      if (inference_accepted && unmap_succeeded) {
        if (active_roi_authority) {
          roi_ownership_pending =
            reserved_roi_transform &&
            roi_transform_slots.commit_reserved_enqueued(
              *reserved_roi_transform
            );
        } else {
          pending_legacy_frame_identity =
            legacy_frame_identity;
          roi_ownership_pending = true;
        }
        if (!roi_ownership_pending) {
          // TensorRT accepted the work, so it cannot be rolled back. Preserve only its frame
          // identity and drop the eventual output before normalization.
          if (active_roi_authority) {
            roi_transform_slots.orphan_reserved_enqueued(frame_id);
          } else {
            pending_legacy_frame_identity.reset();
          }
          if (!roi_transform_error_logged) {
            BOOST_LOG(error)
              << "Depth estimator could not transition the accepted ROI reservation for frame "
              << frame_id << "; its output will be consumed and dropped.";
            roi_transform_error_logged = true;
          }
        }
      } else if (inference_accepted) {
        // The CUDA work may still execute even though interop unmap failed. Do not claim a valid
        // GPU transform; consume/drop it if the stream reaches completion, then disable this
        // estimator before either interop resource can be reused.
        if (active_roi_authority) {
          roi_transform_slots.orphan_reserved_enqueued(frame_id);
        } else {
          pending_legacy_frame_identity.reset();
        }
        disable_after_pending_drop = true;
      } else {
        rollback_roi_reservation();
      }

      has_previous_frame = inference_accepted;
      if (inference_accepted) {
        pending_frame_id = frame_id;
        if (
          roi_ownership_pending &&
          scene_controller &&
          scene_controller_prepared
        ) {
          scene_controller->mark_enqueued(frame_id);
        } else if (scene_controller && scene_controller_prepared) {
          scene_controller->discard_prepared(frame_id);
        }
        if (diagnostics_enabled) {
          throughput_stats_enqueues++;
        }
      } else if (scene_controller && scene_controller_prepared) {
        scene_controller->discard_prepared(frame_id);
      }

      return make_result(
        completed_frame_valid,
        completed_frame_id,
        inference_accepted,
        raw_snapshot_valid,
        model_input_snapshot_valid,
        completion_dropped,
        dropped_frame_id
      );
    }
  };

  video_depth_estimator::video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device, Microsoft::WRL::ComPtr<ID3D11DeviceContext> context, const std::filesystem::path &assets_dir, const config::video_t::sbs_t &cfg, const config::depth_model_info &model):
      pimpl(std::make_unique<impl>(device, context, assets_dir, cfg, model)) {}

  video_depth_estimator::~video_depth_estimator() = default;

  bool video_depth_estimator::is_valid() const {
    return pimpl && pimpl->valid;
  }

  bool video_depth_estimator::can_accept_frame() {
    return pimpl && pimpl->can_accept();
  }

  estimate_result video_depth_estimator::estimate_depth(
    ID3D11ShaderResourceView *input_srv,
    input_color_space color_space,
    std::uint64_t frame_id,
    bool snapshot_debug_inputs
  ) {
    return pimpl->estimate(
      input_srv,
      color_space,
      frame_id,
      snapshot_debug_inputs
    );
  }

  estimate_result video_depth_estimator::finish_pending_depth_for_evaluation(input_color_space color_space) {
    return pimpl->finish_pending(color_space);
  }

  depth_telemetry_poll_result video_depth_estimator::poll_depth_telemetry(
    bool schedule_copy,
    std::uint64_t sampled_frame_id
  ) {
    if (!pimpl) {
      depth_telemetry_poll_result result;
      result.failed = schedule_copy;
      return result;
    }
    return pimpl->poll_depth_telemetry(schedule_copy, sampled_frame_id);
  }
}  // namespace models
