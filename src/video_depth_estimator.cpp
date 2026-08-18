#include "video_depth_estimator.h"

#include "cuda_conditional_graph.h"
#include "cuda_driver_api.h"
#include "depth_coordinate_v2.h"
#include "generated/sbs_adaptive_state_contract.h"
#include "host_sbs_gpu_trace.h"
#include "host_sbs_shader_cache.h"
#include "logging.h"
#include "model_manager.h"
#include "platform/windows/misc.h"
#include "platform/windows/utils.h"
#include "sbs_perf.h"
#include "utility.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <windows.h>

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

static void set_model_prepare_status(const std::string &engine_name, models::engine_build_status status) {
  std::lock_guard<std::mutex> lock(g_model_prepare_status_mutex);
  g_model_prepare_status[engine_name] = status;
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
static std::mutex g_tensorrt_compile_mutex;
// One active stream normally needs one context; four permits bounded encoder-transition and
// failed-warmup recovery without letting repeated rebuilds consume VRAM without bound.
static constexpr std::size_t kMaxContextsPerEngine = 4;
static nvinfer1::IRuntime *g_runtime = nullptr;
static std::once_flag g_cuda_init_once;
static CUresult g_cuda_init_result = CUDA_ERROR_NOT_READY;
static std::mutex g_cuda_context_mutex;
static std::map<CUdevice, CUcontext> g_cuda_primary_contexts;
// CUDA's CU_EVENT_DISABLE_TIMING flag. The local driver shim intentionally exposes only the
// small ABI surface this translation unit needs, so keep the readiness-only flag local too.
static constexpr unsigned int cuda_event_disable_timing = 0x2u;

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

static bool unregister_cuda_graphics_resource(
  cuda_driver_api &cuda,
  CUgraphicsResource &resource
) {
  if (!resource) {
    return true;
  }
  const bool unregistered = cuda.cuGraphicsUnregisterResource &&
                            cuda.cuGraphicsUnregisterResource(resource) == CUDA_SUCCESS;
  if (unregistered) {
    resource = nullptr;
  }
  return unregistered;
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
      << '_' << NV_TENSORRT_BUILD
      << "-sm" << sm_major << sm_minor << "-gpu" << std::hex << name_hash;
  return tag.str();
}

// One resident engine per CUDA-device/model pair, so multi-adapter sessions never reuse a
// TensorRT engine or execution context deserialized under another CUDA primary context. Distinct
// startup model configurations remain isolated instead of being pinned to the first model.
// TensorRT interfaces are never destroyed: an incompatible engine may be detached from its slot,
// but its allocation remains deliberately leaked across the MinGW/MSVC ABI boundary. A usable
// IExecutionContext holds ~1.3 GB scratch, so contexts are pooled per engine and reused (see the
// ctor/dtor). Sequential evaluator model testing can leave 2-3 engines resident, which is
// acceptable.
struct engine_slot {
  nvinfer1::ICudaEngine *engine = nullptr;
  std::vector<nvinfer1::IExecutionContext *> context_pool;
  // Usable contexts include both checked-out and pooled instances. Failed warmup contexts cannot
  // be destroyed across the MinGW/MSVC ABI boundary, so account for them separately: they must
  // never re-enter the pool, while the combined count still enforces the physical VRAM cap.
  models::detail::execution_context_accounting_t context_accounting;
  bool io_validated = false;
  bool io_compatible = false;
};

static std::map<std::string, engine_slot> g_engines;  // guarded by g_trt_mutex

static std::size_t allocated_context_count(const engine_slot &slot) {
  return slot.context_accounting.allocated();
}

// The object is deliberately leaked because destroying TensorRT interfaces across this compiler
// boundary corrupts the heap. Removing it from usable accounting prevents a later session from
// treating a context that never completed warmup, or later suffered an asynchronous execution
// failure, as reusable; quarantined accounting keeps repeated failures bounded by
// kMaxContextsPerEngine.
static void quarantine_execution_context_locked(
  const std::string &engine_key,
  nvinfer1::IExecutionContext *&context,
  const bool was_warmed
) {
  if (!context) {
    return;
  }
  auto &slot = g_engines[engine_key];
  slot.context_accounting.quarantine(was_warmed);
  context = nullptr;
  g_trt_context_available.notify_all();
}

static void mark_execution_context_warmed_locked(const std::string &engine_key) {
  g_engines[engine_key].context_accounting.mark_warmed();
}

static void return_execution_context_locked(
  const std::string &engine_key,
  nvinfer1::IExecutionContext *&context
) {
  if (!context) {
    return;
  }
  g_engines[engine_key].context_pool.push_back(context);
  context = nullptr;
  g_trt_context_available.notify_all();
}

static void release_context_reservation_locked(const std::string &engine_key) {
  g_engines[engine_key].context_accounting.release_reservation();
  g_trt_context_available.notify_all();
}

enum class startup_context_reservation_e {
  create,
  resident_warmed,
  unavailable,
};

static startup_context_reservation_e reserve_startup_context_locked(
  const std::string &engine_key
) {
  auto &slot = g_engines[engine_key];
  if (allocated_context_count(slot) >= kMaxContextsPerEngine) {
    return slot.context_accounting.warmed() != 0u ?
             startup_context_reservation_e::resident_warmed :
             startup_context_reservation_e::unavailable;
  }
  slot.context_accounting.reserve(kMaxContextsPerEngine);
  return startup_context_reservation_e::create;
}

static void erase_empty_engine_slot_locked(const std::string &engine_key) {
  const auto found = g_engines.find(engine_key);
  if (found != g_engines.end() && !found->second.engine &&
      allocated_context_count(found->second) == 0u &&
      found->second.context_pool.empty()) {
    g_engines.erase(found);
  }
}

static void recycle_or_quarantine_execution_context_locked(
  const std::string &engine_key,
  nvinfer1::IExecutionContext *&context,
  const bool warmed,
  const bool poisoned
) {
  if (warmed && !poisoned) {
    return_execution_context_locked(engine_key, context);
  } else {
    quarantine_execution_context_locked(engine_key, context, warmed);
  }
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
  bool *out_pooled = nullptr
) {
  out_context = nullptr;
  if (out_pooled) {
    *out_pooled = false;
  }
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
    if (out_pooled) {
      *out_pooled = true;
    }
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
  bool input_mode_ok = false;
  bool output_mode_ok = false;
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
      input_mode_ok = is_input;
      if (!input_mode_ok) {
        BOOST_LOG(error) << "Depth model tensor 'pixel_values' is not an input; rejecting the engine.";
      }
      if (!input_fp32) {
        BOOST_LOG(error) << "Depth model input 'pixel_values' is " << tensor_dtype_name(type)
                         << ", not FP32; rejecting the engine. Use a keep_io_types (FP32 I/O) model.";
      }
    } else if (std::string_view(name) == "predicted_depth") {
      have_out = true;
      output_fp32 = type == nvinfer1::DataType::kFLOAT;
      output_mode_ok = !is_input;
      if (!output_mode_ok) {
        BOOST_LOG(error) << "Depth model tensor 'predicted_depth' is not an output; rejecting the engine.";
      }
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
  slot.io_compatible = have_in && have_out && input_fp32 && output_fp32 &&
                       input_mode_ok && output_mode_ok;
  return slot.io_compatible;
}

// Validate the fixed PP-OCRv6 tiny boundary without calling any TensorRT method that returns
// Dims by value (unsafe across the MinGW/MSVC ABI boundary on Windows).
static bool validate_ocr_engine_io_locked(nvinfer1::ICudaEngine *engine, engine_slot &slot) {
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
  bool input_mode_ok = false;
  bool output_mode_ok = false;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char *name = engine->getIOTensorName(i);
    if (!name) {
      BOOST_LOG(error) << "OCR TensorRT engine returned a null I/O tensor name.";
      continue;
    }
    const auto type = engine->getTensorDataType(name);
    const bool is_input =
      engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    BOOST_LOG(info) << "OCR engine tensor '" << name << "' "
                    << (is_input ? "(input)" : "(output)")
                    << " dtype=" << tensor_dtype_name(type);
    if (std::string_view(name) == "x") {
      have_in = true;
      input_fp32 = type == nvinfer1::DataType::kFLOAT;
      input_mode_ok = is_input;
    } else if (std::string_view(name) == "fetch_name_0") {
      have_out = true;
      output_fp32 = type == nvinfer1::DataType::kFLOAT;
      output_mode_ok = !is_input;
    }
  }
  slot.io_compatible = engine->getNbIOTensors() == 2 && have_in && have_out &&
                        input_fp32 && output_fp32 &&
                        input_mode_ok && output_mode_ok;
  if (!slot.io_compatible) {
    BOOST_LOG(error)
      << "OCR engine must expose FP32 input 'x' and FP32 output 'fetch_name_0'; "
         "detector conditioning will remain flat.";
  }
  return slot.io_compatible;
}

// Detach a resident engine that deserialized successfully but violates the fixed model I/O
// contract, so rebuilding its on-disk plan can actually replace it. TensorRT objects cannot be
// destroyed safely across the MinGW/MSVC ABI boundary; pooled contexts and the engine are
// therefore deliberately leaked and quarantined. Never detach while another estimator owns a
// context from this slot: that session may still be executing against the resident engine.
// Caller holds g_trt_mutex.
static bool detach_incompatible_engine_locked(const std::string &engine_key) {
  auto found = g_engines.find(engine_key);
  if (found == g_engines.end()) {
    return true;
  }
  auto &slot = found->second;
  if (!slot.engine) {
    slot.io_validated = false;
    slot.io_compatible = false;
    return allocated_context_count(slot) == 0 && slot.context_pool.empty();
  }

  if (slot.context_pool.size() > slot.context_accounting.usable()) {
    BOOST_LOG(error) << "TensorRT engine slot accounting is corrupt; refusing unsafe replacement.";
    return false;
  }
  const std::size_t checked_out =
    slot.context_accounting.usable() - slot.context_pool.size();
  if (checked_out != 0) {
    BOOST_LOG(error) << "Cannot replace incompatible TensorRT engine while " << checked_out
                     << " execution context(s) are still owned by active sessions.";
    return false;
  }

  slot.context_accounting.detach_pooled(slot.context_pool.size());
  slot.context_pool.clear();
  slot.engine = nullptr;  // intentionally leaked; see the ABI note above
  slot.io_validated = false;
  slot.io_compatible = false;
  g_trt_context_available.notify_all();
  return true;
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
  const bool enqueue_attempted = bound;
  if (bound) {
    std::lock_guard<std::mutex> lock(g_trt_mutex);
    enqueued = exec_context->enqueueV3(stream);
  }
  CUresult synchronize_result = CUDA_SUCCESS;
  if (enqueue_attempted) {
    synchronize_result = cuda.cuStreamSynchronize ?
                           cuda.cuStreamSynchronize(stream) :
                           static_cast<CUresult>(-1);
  }
  const bool stream_quiesced = models::detail::asynchronous_operands_may_be_released(
    enqueue_attempted,
    cuda.cuStreamSynchronize != nullptr,
    synchronize_result == CUDA_SUCCESS ?
      models::detail::async_stream_readiness_e::ready :
      models::detail::async_stream_readiness_e::failed
  );
  if (!stream_quiesced) {
    // enqueueV3(false) can still mean partial asynchronous submission, and a failed synchronize
    // may itself be reporting that launch. Without a positive quiescence proof, freeing either
    // operand or destroying the stream can race live driver work. The caller quarantines the
    // TensorRT context; retain these small startup allocations with it deliberately.
    free_output.disable();
    free_input.disable();
    destroy_stream.disable();
    BOOST_LOG(warning)
      << "Depth startup warmup could not prove stream quiescence (enqueue=" << enqueued
      << ", synchronize=" << synchronize_result
      << "); retaining its CUDA operands with the quarantined context.";
  }
  const bool synchronized = enqueued && stream_quiesced;
  BOOST_LOG(info) << "Depth model startup warmup complete (" << w << 'x' << h
                  << (synchronized ? ")." : "); execution failed.");
  return synchronized;
}

static bool warmup_ocr_execution_context(
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

  constexpr int h = models::ocr_engine_height;
  constexpr int w = models::ocr_engine_width;
  constexpr std::size_t pixels = static_cast<std::size_t>(h) * w;
  CUdeviceptr d_in = 0;
  CUdeviceptr d_out = 0;
  if (cuda.cuMemAlloc(&d_in, 3u * pixels * sizeof(float)) != CUDA_SUCCESS) {
    return false;
  }
  auto free_input = util::fail_guard([&]() {
    cuda.cuMemFree(d_in);
  });
  if (cuda.cuMemAlloc(&d_out, pixels * sizeof(float)) != CUDA_SUCCESS) {
    return false;
  }
  auto free_output = util::fail_guard([&]() {
    cuda.cuMemFree(d_out);
  });

  const auto input_dims = make_input_dims(h, w);
  constexpr std::int64_t output_bytes =
    static_cast<std::int64_t>(pixels * sizeof(float));
  const bool shape_bound = exec_context->setInputShape("x", input_dims);
  const std::int64_t required_output_bytes =
    shape_bound ? exec_context->getMaxOutputSize("fetch_name_0") : -1;
  if (required_output_bytes <= 0 || required_output_bytes > output_bytes) {
    BOOST_LOG(error)
      << "PP-OCRv6 tiny resolved output requires " << required_output_bytes
      << " bytes; the authenticated fixed buffer provides " << output_bytes << '.';
    return false;
  }
  const bool bound = shape_bound &&
                     exec_context->setTensorAddress("x", reinterpret_cast<void *>(d_in)) &&
                     exec_context->setTensorAddress(
                       "fetch_name_0",
                       reinterpret_cast<void *>(d_out)
                     );
  bool enqueued = false;
  const bool enqueue_attempted = bound;
  if (bound) {
    std::lock_guard<std::mutex> lock(g_trt_mutex);
    enqueued = exec_context->enqueueV3(stream);
  }
  CUresult synchronize_result = CUDA_SUCCESS;
  if (enqueue_attempted) {
    synchronize_result = cuda.cuStreamSynchronize ?
                           cuda.cuStreamSynchronize(stream) :
                           static_cast<CUresult>(-1);
  }
  const bool stream_quiesced = models::detail::asynchronous_operands_may_be_released(
    enqueue_attempted,
    cuda.cuStreamSynchronize != nullptr,
    synchronize_result == CUDA_SUCCESS ?
      models::detail::async_stream_readiness_e::ready :
      models::detail::async_stream_readiness_e::failed
  );
  if (!stream_quiesced) {
    // Keep exactly the same partial-submit lifetime rule as DAV2. The owning cache quarantines
    // this OCR context after the failed warmup; its possibly-live stream and operands must follow
    // that context instead of being freed underneath it.
    free_output.disable();
    free_input.disable();
    destroy_stream.disable();
    BOOST_LOG(warning)
      << "PP-OCRv6 tiny startup warmup could not prove stream quiescence (enqueue="
      << enqueued << ", synchronize=" << synchronize_result
      << "); retaining its CUDA operands with the quarantined context.";
  }
  const bool synchronized = enqueued && stream_quiesced;
  BOOST_LOG(info) << "PP-OCRv6 tiny startup warmup "
                  << (synchronized ? "complete" : "failed") << " (" << w << 'x' << h
                  << ").";
  return synchronized;
}

namespace models {

  namespace {

    static_assert(near_identical_decision_cookie == cuda_conditional_graph::decision_cookie);
    static_assert(near_identical_token_low_cookie == cuda_conditional_graph::token_low_cookie);
    static_assert(near_identical_token_high_cookie == cuda_conditional_graph::token_high_cookie);
    static_assert(near_identical_proposal_magic == cuda_conditional_graph::proposal_magic);
    static_assert(near_identical_request_magic == cuda_conditional_graph::request_magic);
    static_assert(near_identical_receipt_magic == cuda_conditional_graph::receipt_magic);
    static_assert(near_identical_optional_receipt_magic ==
                  cuda_conditional_graph::optional_ocr_receipt_magic);
    static_assert(near_identical_work_flags_cookie ==
                  cuda_conditional_graph::work_flags_cookie);
    static_assert(near_identical_work_optional_ocr ==
                  cuda_conditional_graph::work_flags_value(
                    cuda_conditional_graph::work_flag_e::optional_ocr
                  ));
    static_assert(near_identical_work_subtitle_observation ==
                  cuda_conditional_graph::work_flags_value(
                    cuda_conditional_graph::work_flag_e::subtitle_observation
                  ));
    static_assert(near_identical_work_optional_ocr_due ==
                  cuda_conditional_graph::work_flags_value(
                    cuda_conditional_graph::work_flag_e::optional_ocr_due
                  ));
    static_assert(near_identical_work_subtitle_observation_due ==
                  cuda_conditional_graph::work_flags_value(
                    cuda_conditional_graph::work_flag_e::subtitle_observation_due
                  ));
    static_assert(near_identical_gpu_decision_record_byte_offset == 0u);
    static_assert(near_identical_gpu_request_record_byte_offset ==
                  sizeof(cuda_conditional_graph::decision_record_t));

    [[nodiscard]] constexpr cuda_conditional_graph::work_flag_e
    subtitle_transaction_work(
      const depth_optional_work_mode_e requested,
      const bool ocr_ready
    ) noexcept {
      switch (requested) {
        case depth_optional_work_mode_e::ordinary:
          return ocr_ready ?
                   cuda_conditional_graph::work_flag_e::optional_ocr :
                   cuda_conditional_graph::work_flag_e::subtitle_observation;
        case depth_optional_work_mode_e::ordinary_due:
          return ocr_ready ?
                   cuda_conditional_graph::work_flag_e::optional_ocr_due :
                   cuda_conditional_graph::work_flag_e::subtitle_observation_due;
        case depth_optional_work_mode_e::suppress_subtitle:
          return cuda_conditional_graph::work_flag_e::none;
      }
      return cuda_conditional_graph::work_flag_e::none;
    }

    static_assert(fit_subtitle_analysis_geometry(
                    1920u, 1080u, {770, 434}).roi_top == 325u);
    static_assert(fit_subtitle_analysis_geometry(
                    1920u, 1080u, {770, 434}).roi_bottom == 430u);
    static_assert(fit_subtitle_analysis_geometry(
                    2560u, 1080u, {1022, 434}).roi_top == 289u);
    static_assert(fit_subtitle_analysis_geometry(
                    3440u, 1440u, {1036, 434}).roi_top == 287u);
    static_assert(fit_subtitle_analysis_geometry(
                    1080u, 1920u, {434, 770}).roi_top == 709u);
    static_assert(fit_subtitle_analysis_geometry(
                    1080u, 1920u, {434, 770}).roi_bottom == 768u);
    static_assert(host_sbs_v2_depth_shape_is_authenticated({770, 434}));
    static_assert(host_sbs_v2_depth_shape_is_authenticated({1022, 434}));
    static_assert(host_sbs_v2_depth_shape_is_authenticated({1036, 434}));
    static_assert(host_sbs_v2_depth_shape_is_authenticated({434, 770}));
    static_assert(host_sbs_v2_depth_shape_is_authenticated({434, 1022}));
    static_assert(host_sbs_v2_depth_shape_is_authenticated({434, 1036}));
    static_assert(!host_sbs_v2_depth_shape_is_authenticated({1008, 434}));

  }  // namespace

  bool parallax_v2_result_is_authenticated(const estimate_result &result) {
    if (!result.completed_frame_valid || result.completed_frame_id == 0u ||
        !result.parallax_v2_producer_active || !result.shadow_candidate_parallax ||
        !result.shadow_ownership_refined_parallax ||
        !result.shadow_vertical_majorant || !result.shadow_vertical_conditioned ||
        !result.shadow_base_final_parallax || !result.shadow_final_parallax ||
        !result.shadow_state || !result.shadow_frame_stats ||
        !result.raw_model_provenance || !result.parallax_v2_shader_provenance ||
        result.raw_width <= 0 || result.raw_height <= 0) {
      return false;
    }
    Microsoft::WRL::ComPtr<ID3D11Resource> base_final_resource;
    Microsoft::WRL::ComPtr<ID3D11Resource> final_resource;
    result.shadow_base_final_parallax->GetResource(&base_final_resource);
    result.shadow_final_parallax->GetResource(&final_resource);
    if (!base_final_resource || !final_resource ||
        base_final_resource.Get() == final_resource.Get()) {
      return false;
    }

    const auto &input_region = result.input_region;
    const auto full_source_shape = fit_host_sbs_v2_depth_tensor_shape(
      input_region.source_width,
      input_region.source_height
    );
    const depth_tensor_shape_t result_shape {result.raw_width, result.raw_height};
    if (!input_region.valid() ||
        !host_sbs_v2_source_resolution_is_supported(
          input_region.source_width, input_region.source_height) ||
        full_source_shape != result_shape ||
        !input_region.tensor_content.valid(result_shape)) {
      return false;
    }
    if (input_region.video_region) {
      const auto expected = plan_host_sbs_v2_video_region(
        {input_region.left, input_region.top, input_region.right, input_region.bottom},
        input_region.source_width,
        input_region.source_height,
        result_shape
      );
      if (!expected || expected->source_rect != depth_source_rect_t {
            input_region.left, input_region.top, input_region.right, input_region.bottom
          } || expected->tensor_content != input_region.tensor_content) {
        return false;
      }
    } else if (!input_region.tensor_content.full(result_shape)) {
      return false;
    }
    if (!result.ocr_box_record || !result.subtitle_locator_state) {
      return false;
    }

    if (!parallax_v2_shader_provenance_matches_current_contract(
          *result.parallax_v2_shader_provenance
        )) {
      return false;
    }

    const auto &model = *result.raw_model_provenance;
    const auto *calibration = depth_coordinate_v2::find_capture_calibration(
      model.depth_model,
      model.depth_model_url,
      model.onnx_sha256,
      model.preprocess_profile,
      model.preprocess_source_closure_sha256,
      static_cast<std::uint32_t>(result.raw_width),
      static_cast<std::uint32_t>(result.raw_height)
    );
    if (!calibration ||
        std::abs(result.parallax_v2_raw_coordinate_scale -
                 calibration->raw_coordinate_scale) > 1.0e-6f ||
        !depth_coordinate_v2::parallax_runtime_constants_are_valid(
          result.parallax_v2_raw_coordinate_scale,
          result.parallax_v2_requested_pop_strength,
          result.parallax_v2_requested_gain
        )) {
      return false;
    }
    return true;
  }

  struct engine_artifact {
    std::string name;
    std::string source_sha256;
    std::filesystem::path source_path;
    std::filesystem::path engine_path;
  };

  static bool publish_serialized_engine(
    const std::filesystem::path &engine_path,
    const nvinfer1::IHostMemory &serialized,
    const std::string_view description
  ) {
    auto part_path = engine_path;
    part_path += ".part";
    std::error_code filesystem_error;
    std::filesystem::remove(part_path, filesystem_error);
    {
      std::ofstream output(part_path, std::ios::binary | std::ios::trunc);
      if (output) {
        output.write(
          static_cast<const char *>(serialized.data()),
          serialized.size()
        );
        output.close();
      }
      if (!output) {
        std::filesystem::remove(part_path, filesystem_error);
        BOOST_LOG(error) << "Failed to save built " << description << " to " << engine_path;
        return false;
      }
    }
    std::filesystem::rename(part_path, engine_path, filesystem_error);
    if (filesystem_error) {
      BOOST_LOG(error) << "Failed to publish built " << description << ' ' << engine_path
                       << ": " << filesystem_error.message();
      std::filesystem::remove(part_path, filesystem_error);
      return false;
    }
    BOOST_LOG(info) << "Saved built " << description << " atomically to " << engine_path;
    return true;
  }

  static std::mutex g_active_engine_manifest_mutex;

  static bool publish_active_engine_manifest(
    const std::filesystem::path &assets_dir,
    const std::string_view model_name,
    const engine_artifact &artifact
  ) {
    if (artifact.name.empty() || artifact.source_sha256.empty()) {
      return false;
    }

    const auto path = assets_dir / (std::string(model_name) + ".active-engine.json");
    auto temporary_path = path;
    temporary_path += ".tmp";
    const nlohmann::json manifest {
      {"schema", 1},
      {"model", std::string(model_name)},
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
    std::lock_guard<std::mutex> lock(g_tensorrt_compile_mutex);

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
    if (!serializedModel) {
      BOOST_LOG(error) << "Engine build failed.";
      return false;
    }
    // Save under the recipe-specific engine name so a later recipe change rebuilds rather than
    // silently reusing this engine's (now-wrong) I/O layout.
    return publish_serialized_engine(artifact.engine_path, *serializedModel, "depth engine");
  }

  static bool ensure_ocr_tensorrt_engine_for_device(
    const std::filesystem::path &assets_dir,
    cuda_driver_api &cuda,
    CUdevice cuda_device,
    engine_artifact &artifact
  ) {
    std::lock_guard<std::mutex> lock(g_tensorrt_compile_mutex);

    // The ModelOpt-derived graph is a required packaged asset, not a network-repairable cache.
    // The pinned upstream URL names different FP32 bytes, so a missing or mismatched bundle must
    // fail flat without deleting it or silently downloading the source model in its place.
    artifact.source_path = assets_dir / std::filesystem::path {ocr_model_asset_path};
    std::error_code source_error;
    if (!std::filesystem::is_regular_file(artifact.source_path, source_error) || source_error) {
      BOOST_LOG(warning)
        << "Bundled mixed-FP16 PP-OCRv6 tiny ONNX is unavailable at "
        << artifact.source_path << "; subtitle conditioning stays flat.";
      return false;
    }
    artifact.source_sha256 = file_sha256_hex(artifact.source_path);
    if (artifact.source_sha256.empty()) {
      BOOST_LOG(error)
        << "Bundled mixed-FP16 PP-OCRv6 tiny ONNX is unreadable; subtitle conditioning stays flat.";
      return false;
    }

    if (artifact.source_sha256 != ocr_model_artifact_onnx_sha256) {
      BOOST_LOG(error)
        << "Bundled mixed-FP16 PP-OCRv6 tiny ONNX SHA-256 mismatch (expected "
        << ocr_model_artifact_onnx_sha256 << ", got " << artifact.source_sha256
        << "); refusing the packaged artifact without deleting or replacing it.";
      return false;
    }
    artifact.name = ocr_engine_filename(
      engine_compatibility_tag(cuda, cuda_device) + "-onnx" + artifact.source_sha256
    );
    if (artifact.name.empty()) {
      BOOST_LOG(error)
        << "Could not derive the bounded PP-OCRv6 tiny TensorRT cache identity; "
           "subtitle conditioning stays flat.";
      return false;
    }
    artifact.engine_path = assets_dir / artifact.name;

    std::error_code existing_ec;
    if (std::filesystem::is_regular_file(artifact.engine_path, existing_ec)) {
      BOOST_LOG(info) << "PP-OCRv6 tiny TensorRT engine cache hit: "
                      << artifact.engine_path.filename();
      return true;
    }

    if (CUcontext ctx = primary_context(cuda, cuda_device)) {
      if (cuda.cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
        BOOST_LOG(error) << "Could not select the configured CUDA device for OCR compilation.";
        return false;
      }
    } else {
      BOOST_LOG(error) << "Could not retain the configured CUDA device for OCR compilation.";
      return false;
    }

    initLibNvInferPlugins(&gLogger, "");
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
      return false;
    }
#if NV_TENSORRT_MAJOR < 11
    // TensorRT 10 still permits weakly typed networks, so make the production type policy
    // explicit. TensorRT 11 makes every network strongly typed and deprecates/ignores this flag.
    constexpr std::uint32_t ocr_network_creation_flags =
      1u << static_cast<std::uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED
      );
    static_assert(ocr_network_creation_flags != 0u);
#else
    constexpr std::uint32_t ocr_network_creation_flags = 0u;
    static_assert(ocr_network_creation_flags == 0u);
#endif
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(ocr_network_creation_flags)
    );
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!network || !config) {
      return false;
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    config->setBuilderOptimizationLevel(ocr_engine_builder_level);
    // The graph is strongly typed FP16 with FP32 I/O. Keep TF32 explicitly permitted for any
    // authenticated FP32 island; the flag cannot change the FP16 tensors selected by ModelOpt.
    config->setFlag(nvinfer1::BuilderFlag::kTF32);

    auto parser = TrtUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, gLogger)
    );
    if (!parser || !parser->parseFromFile(artifact.source_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      BOOST_LOG(error) << "Failed to parse authenticated PP-OCRv6 tiny ONNX.";
      return false;
    }

    auto *input = network->getNbInputs() == 1 ? network->getInput(0) : nullptr;
    if (!input || !input->getName() || std::string_view(input->getName()) != "x" ||
        input->getType() != nvinfer1::DataType::kFLOAT) {
      BOOST_LOG(error)
        << "PP-OCRv6 tiny input contract must be exactly one FP32 tensor named 'x'.";
      return false;
    }
    auto profile = builder->createOptimizationProfile();
    const auto fixed_dims = make_input_dims(ocr_engine_height, ocr_engine_width);
    const bool profile_ok = profile &&
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, fixed_dims) &&
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, fixed_dims) &&
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, fixed_dims);
    if (!profile_ok || config->addOptimizationProfile(profile) < 0) {
      BOOST_LOG(error) << "TensorRT rejected the fixed 960x160 OCR profile.";
      return false;
    }

    nvinfer1::ITensor *output = nullptr;
    for (int i = 0; i < network->getNbOutputs(); ++i) {
      auto *candidate = network->getOutput(i);
      if (candidate && candidate->getName() &&
          std::string_view(candidate->getName()) == "fetch_name_0") {
        output = candidate;
      }
    }
    if (!output || network->getNbOutputs() != 1 ||
        output->getType() != nvinfer1::DataType::kFLOAT) {
      BOOST_LOG(error)
        << "PP-OCRv6 tiny output contract must be exactly one FP32 tensor named "
           "'fetch_name_0'.";
      return false;
    }

    BOOST_LOG(info)
      << "Building authenticated PP-OCRv6 tiny TensorRT engine (ModelOpt FP16 graph, FP32 I/O, "
         "fixed 960x160).";
    auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config)
    );
    if (!serialized) {
      BOOST_LOG(error) << "PP-OCRv6 tiny TensorRT engine build failed.";
      return false;
    }

    return publish_serialized_engine(
      artifact.engine_path,
      *serialized,
      "PP-OCRv6 tiny engine"
    );
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

    // Shader compilation is CPU-only, so do it before TensorRT engine/context preparation can
    // occupy this background worker for several seconds. A concurrent early stream joins the
    // same process-wide cache entry instead of compiling a duplicate on its encoder thread.
    // Keep model preparation independent: an installation can repair a missing shader while the
    // already validated TensorRT plan remains resident and reusable.
    if (!host_sbs_shader_cache::prewarm(assets_dir)) {
      BOOST_LOG(error)
        << "One or more fixed-shape Host SBS shaders could not be precompiled at startup.";
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
    bool create_context = false;
    bool resident_warmed_context = false;
    bool engine_repair_allowed = true;
    {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      engine = acquire_engine_locked(engine_key, engine_path, exec_context);
      auto &slot = g_engines[engine_key];
      if (engine && !validate_engine_io_locked(engine, slot)) {
        return_execution_context_locked(engine_key, exec_context);
        engine_repair_allowed = detach_incompatible_engine_locked(engine_key);
        engine = nullptr;
      }
    }

    // An existing file is not proof of a usable TensorRT plan: interrupted legacy writes, a
    // runtime upgrade, or copied assets can all leave a regular file that fails deserialization.
    // Remove only a slot with no resident engine/contexts, rebuild atomically from ONNX, and retry
    // once. This turns the former permanent flat-SBS state into a self-healing startup path.
    if (!engine) {
      if (!engine_repair_allowed) {
        BOOST_LOG(error) << "Cached TensorRT plan is incompatible but still owned by an active "
                            "session; refusing to replace it unsafely.";
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        erase_empty_engine_slot_locked(engine_key);
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
      engine = acquire_engine_locked(engine_key, engine_path, exec_context);
    }

    {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      auto &slot = g_engines[engine_key];
      if (!validate_engine_io_locked(engine, slot)) {
        return_execution_context_locked(engine_key, exec_context);
        detach_incompatible_engine_locked(engine_key);
        return false;
      }
      if (!exec_context) {
        switch (reserve_startup_context_locked(engine_key)) {
          case startup_context_reservation_e::create:
            create_context = true;
            break;
          case startup_context_reservation_e::resident_warmed:
            resident_warmed_context = true;
            break;
          case startup_context_reservation_e::unavailable:
            // A live session may already have populated the engine before startup preparation
            // finished. Only a context that actually completed warmup can establish readiness;
            // quarantined or still-constructing contexts are not evidence that the plan is usable.
            BOOST_LOG(error) << "TensorRT context capacity contains no successfully warmed context.";
            return false;
        }
      }
    }

    if (create_context) {
      BOOST_LOG(info) << "Creating startup TensorRT execution context...";
      exec_context = engine->createExecutionContext();
      if (!exec_context) {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        release_context_reservation_locked(engine_key);
        return false;
      }
      if (!warmup_execution_context(cuda, cuda_ctx, exec_context)) {
        // This context cannot be destroyed across the MinGW/MSVC ABI boundary, but it must never
        // enter the reusable pool: pooled contexts are assumed warmed and skip this operation.
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        quarantine_execution_context_locked(engine_key, exec_context, false);
        BOOST_LOG(error) << "Startup depth-model context warmup failed.";
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        mark_execution_context_warmed_locked(engine_key);
      }
    }

    const bool reusable_warmed_context = exec_context || resident_warmed_context;
    if (exec_context) {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      return_execution_context_locked(engine_key, exec_context);
    }
    if (!reusable_warmed_context) {
      BOOST_LOG(error) << "Startup depth-model preparation produced no reusable warmed context.";
      return false;
    }
    if (!publish_active_engine_manifest(assets_dir, model.name, artifact)) {
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

  bool prepare_ocr_tensorrt_model(
    const std::filesystem::path &assets_dir,
    const std::string &adapter_name
  ) {
    auto &cuda = cuda_driver_api::get();
    if (!cuda.is_valid() || !ensure_cuda_initialized(cuda)) {
      BOOST_LOG(warning) << "CUDA is unavailable; PP-OCRv6 tiny remains fail-flat.";
      return false;
    }
    CUdevice cuda_device = 0;
    if (!cuda_device_for_configured_adapter(cuda, adapter_name, cuda_device)) {
      BOOST_LOG(warning) << "Could not resolve the configured adapter for PP-OCRv6 tiny.";
      return false;
    }
    CUcontext cuda_ctx = primary_context(cuda, cuda_device);
    if (!cuda_ctx || cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
      return false;
    }

    engine_artifact artifact;
    if (!ensure_ocr_tensorrt_engine_for_device(
          assets_dir, cuda, cuda_device,
          artifact
        )) {
      return false;
    }
    auto engine_path = artifact.engine_path;
    auto engine_key = std::to_string(cuda_device) + ":" + artifact.name;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *exec_context = nullptr;
    bool create_context = false;
    bool resident_warmed_context = false;
    bool engine_repair_allowed = true;
    {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      engine = acquire_engine_locked(
        engine_key, engine_path, exec_context
      );
      auto &slot = g_engines[engine_key];
      if (!validate_ocr_engine_io_locked(engine, slot)) {
        return_execution_context_locked(engine_key, exec_context);
        engine_repair_allowed = detach_incompatible_engine_locked(engine_key);
        engine = nullptr;
      }
    }

    // A regular cache file may still be truncated, stale across a runtime upgrade, or expose a
    // foreign tensor contract. Replace it only when no live context owns its resident engine,
    // rebuild atomically from the authenticated ONNX, and retry exactly once.
    if (!engine) {
      if (!engine_repair_allowed) {
        BOOST_LOG(error)
          << "Cached PP-OCRv6 tiny plan is incompatible but still owned by an active session; "
             "refusing unsafe replacement.";
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        erase_empty_engine_slot_locked(engine_key);
      }
      std::error_code remove_error;
      std::filesystem::remove(engine_path, remove_error);
      BOOST_LOG(warning)
        << "Cached PP-OCRv6 tiny plan is unreadable or incompatible; rebuilding "
        << engine_path.filename() << '.';
      if (!ensure_ocr_tensorrt_engine_for_device(
            assets_dir, cuda, cuda_device,
            artifact
          )) {
        return false;
      }
      engine_path = artifact.engine_path;
      engine_key = std::to_string(cuda_device) + ":" + artifact.name;
      {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        engine = acquire_engine_locked(
          engine_key, engine_path, exec_context
        );
        auto &slot = g_engines[engine_key];
        if (!validate_ocr_engine_io_locked(engine, slot)) {
          return_execution_context_locked(engine_key, exec_context);
          detach_incompatible_engine_locked(engine_key);
          BOOST_LOG(error)
            << "Rebuilt PP-OCRv6 tiny engine still violates the fixed FP32 I/O contract.";
          return false;
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      if (!exec_context) {
        switch (reserve_startup_context_locked(engine_key)) {
          case startup_context_reservation_e::create:
            create_context = true;
            break;
          case startup_context_reservation_e::resident_warmed:
            resident_warmed_context = true;
            break;
          case startup_context_reservation_e::unavailable:
            return false;
        }
      }
    }

    if (create_context) {
      exec_context = engine->createExecutionContext();
      if (!exec_context) {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        release_context_reservation_locked(engine_key);
        return false;
      }
      if (!warmup_ocr_execution_context(cuda, cuda_ctx, exec_context)) {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        quarantine_execution_context_locked(engine_key, exec_context, false);
        return false;
      }
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      mark_execution_context_warmed_locked(engine_key);
    }
    const bool reusable_warmed_context = exec_context || resident_warmed_context;
    if (exec_context) {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      return_execution_context_locked(engine_key, exec_context);
    }
    if (!reusable_warmed_context) {
      return false;
    }
    BOOST_LOG(info)
      << "Authenticated PP-OCRv6 tiny is resident and ready for isolated live contexts.";
    if (!publish_active_engine_manifest(assets_dir, ocr_model_name, artifact)) {
      BOOST_LOG(warning) << "Could not publish the active PP-OCRv6 tiny engine manifest.";
    }
    return true;
  }

  struct video_depth_estimator::impl {
    static constexpr std::uint32_t ocr_grid_width = ocr_engine_width / 8;
    static constexpr std::uint32_t ocr_grid_height = ocr_engine_height;
    static constexpr std::uint32_t ocr_cells_per_group = 32;
    static constexpr std::uint32_t ocr_cell_rows_per_group = 4;
    static constexpr std::uint32_t ocr_cell_group_count =
      (ocr_grid_width + ocr_cells_per_group - 1u) / ocr_cells_per_group;
    static constexpr std::uint32_t ocr_cell_group_row_count =
      (ocr_grid_height + ocr_cell_rows_per_group - 1u) / ocr_cell_rows_per_group;
    static constexpr std::uint32_t ocr_cell_words = 1;
    static constexpr std::uint32_t ocr_cell_stats_word_count =
      ocr_grid_width * ocr_grid_height * ocr_cell_words;
    static constexpr std::uint32_t ocr_box_record_word_count =
      depth_coordinate_v2::subtitle_ocr_record_word_count;
    static constexpr std::uint32_t ocr_box_record_schema =
      depth_coordinate_v2::subtitle_ocr_record_schema;
    static constexpr std::uint32_t ocr_box_record_tag =
      depth_coordinate_v2::subtitle_ocr_record_tag;
    static constexpr std::uint32_t subtitle_locator_state_word_count =
      depth_coordinate_v2::subtitle_locator_state_word_count;
    static constexpr std::uint32_t subtitle_condition_param_word_count =
      depth_coordinate_v2::subtitle_condition_param_word_count;
    using near_identical_constants_t = std::array<std::uint32_t, 20u>;
    using ocr_resolve_constants_t = std::array<std::uint32_t, 16u>;
    static_assert(sizeof(near_identical_constants_t) == 80u);
    static_assert(sizeof(near_identical_constants_t) % 16u == 0u);
    static_assert(sizeof(ocr_resolve_constants_t) == 64u);
    static_assert(sizeof(ocr_resolve_constants_t) % 16u == 0u);
    static_assert(
      host_sbs_gpu_trace::transaction_word_count == near_identical_gpu_decision_word_count
    );
    static_assert(
      host_sbs_gpu_trace::subtitle_locator_word_count == subtitle_locator_state_word_count
    );
    static_assert(
      host_sbs_gpu_trace::subtitle_condition_word_count == subtitle_condition_param_word_count
    );
    static_assert(ocr_grid_width * 8u ==
                  depth_coordinate_v2::subtitle_ocr_output_width);
    static_assert(ocr_grid_height ==
                  depth_coordinate_v2::subtitle_ocr_output_height);

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
    const std::filesystem::path shader_root;

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *exec_context = nullptr;
    CUcontext cuda_ctx = nullptr;
    CUstream cu_stream = nullptr;
    CUdevice cuda_device = -1;
    std::string engine_key;
    // PP-OCR owns a distinct mutable TensorRT context. Its captured graph is a sibling IF node in
    // the one joined CUDA root, so DAV2 and OCR may overlap without a second host-managed stream.
    nvinfer1::ICudaEngine *ocr_engine = nullptr;
    nvinfer1::IExecutionContext *ocr_exec_context = nullptr;
    std::string ocr_engine_key;
    bool ocr_context_pooled = false;
    bool ocr_context_warmed = false;
    detail::warmed_execution_context_health_t ocr_context_health;
    bool ocr_available = false;

    float ema_alpha;
    float ema_edge_change;
    float ema_edge_gradient;
    float ema_edge_strength;
    int depth_short_side;  // depth map short-side resolution (clamped to native short side)
    float max_aspect;  // aspect cap for short-side mode
    float minmax_alpha;  // temporal EMA blend for the normalized min/max
    const bool diagnostics_enabled;
    struct tensorrt_cuda_graph_t {
      CUgraph graph = nullptr;
      detail::cuda_graph_replay_policy_t policy;
    };

    tensorrt_cuda_graph_t depth_inference_graph;
    cuda_conditional_graph::executable_t depth_conditional_graph;
    CUdeviceptr depth_conditional_decision_ptr = 0u;
    CUdeviceptr depth_conditional_request_ptr = 0u;
    CUgraph depth_conditional_child_graph = nullptr;
    CUgraph depth_conditional_optional_child_graph = nullptr;
    bool gpu_conditional_bridge_available = true;
    bool gpu_conditional_bridge_error_logged = false;
    bool gpu_conditional_bridge_context_failed = false;
    // OCR owns a distinct TensorRT context and graph. The graph captures only enqueueV3; D3D/CUDA
    // interop map/unmap and mutable tensor bindings always remain outside the capture boundary.
    // Depth retains only the captured source CUgraph. Every authoritative depth launch uses the
    // conditional wrapper; there is deliberately no standalone raw depth CUgraphExec.
    tensorrt_cuda_graph_t ocr_inference_graph;
    bool valid = false;  // all mandatory engine, shader, and session resources are ready
    float parallax_v2_raw_coordinate_scale = 0.0f;
    const float parallax_v2_requested_pop_strength;
    const float parallax_v2_requested_gain;
    std::shared_ptr<const raw_model_provenance_t> raw_model_provenance;
    std::shared_ptr<const parallax_v2_shader_provenance_t>
      parallax_v2_shader_provenance;
    bool parallax_v2_producer_shaders_ready = false;
    bool parallax_v2_producer_active = false;
    bool parallax_v2_producer_failed = false;

    // Demand-gated, nonblocking telemetry readback. Resources are created lazily only after an
    // external protocol subscriber requests evidence; a three-slot staging/query ring absorbs GPU
    // latency without flushing or waiting on the encode thread.
    static constexpr std::size_t telemetry_state_float_count =
      sbs_adaptive_state::word_count;

    struct telemetry_readback_slot {
      Microsoft::WRL::ComPtr<ID3D11Buffer> staging;
      Microsoft::WRL::ComPtr<ID3D11Query> completion;
      bool pending = false;
      std::uint64_t sampled_frame_id = 0;
      std::chrono::steady_clock::time_point sampled_at {};
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
    unsigned throughput_stats_force_infer_enqueues = 0;
    unsigned throughput_stats_gpu_undecided_enqueues = 0;
    unsigned throughput_stats_gpu_undecided_initial_enqueues = 0;
    unsigned throughput_stats_gpu_undecided_followup_enqueues = 0;
    unsigned throughput_stats_completions = 0;
    unsigned throughput_stats_subtitle_suppressed = 0;
    unsigned throughput_stats_ocr_armed = 0;

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

    perf_evt_ring perf_depth_conditional;  // GPU arbitration plus either infer or reuse

    // D3D11 timing for the work around TensorRT. CUDA events above deliberately measure only
    // the inference enqueue; these timestamp queries expose the resize/normalization input pass
    // and the depth normalization/EMA/cut-analysis passes without ever synchronizing the CPU. A ring
    // is required because query results commonly become available several source frames later.
    struct d3d_perf_slot {
      Microsoft::WRL::ComPtr<ID3D11Query> disjoint;
      Microsoft::WRL::ComPtr<ID3D11Query> post_start;
      Microsoft::WRL::ComPtr<ID3D11Query> post_end;
      Microsoft::WRL::ComPtr<ID3D11Query> parallax_frame_stats_start;
      Microsoft::WRL::ComPtr<ID3D11Query> parallax_frame_stats_end;
      Microsoft::WRL::ComPtr<ID3D11Query> parallax_start;
      Microsoft::WRL::ComPtr<ID3D11Query> parallax_end;
      Microsoft::WRL::ComPtr<ID3D11Query> parallax_map_start;
      Microsoft::WRL::ComPtr<ID3D11Query> parallax_subtitle_start;
      Microsoft::WRL::ComPtr<ID3D11Query> ownership_start;
      Microsoft::WRL::ComPtr<ID3D11Query> ownership_end;
      Microsoft::WRL::ComPtr<ID3D11Query> pre_start;
      Microsoft::WRL::ComPtr<ID3D11Query> pre_end;
      bool pending = false;
      bool has_post = false;
      bool has_parallax_frame_stats = false;
      bool has_parallax = false;
      bool has_parallax_map_start = false;
      bool has_parallax_subtitle_start = false;
      bool has_ownership = false;
      bool has_pre = false;
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
        if (FAILED(device->CreateQuery(&desc, &slot.post_start)) ||
            FAILED(device->CreateQuery(&desc, &slot.post_end)) ||
            FAILED(device->CreateQuery(&desc, &slot.parallax_frame_stats_start)) ||
            FAILED(device->CreateQuery(&desc, &slot.parallax_frame_stats_end)) ||
            FAILED(device->CreateQuery(&desc, &slot.parallax_start)) ||
            FAILED(device->CreateQuery(&desc, &slot.parallax_end)) ||
            FAILED(device->CreateQuery(&desc, &slot.parallax_map_start)) ||
            FAILED(device->CreateQuery(&desc, &slot.parallax_subtitle_start)) ||
            FAILED(device->CreateQuery(&desc, &slot.ownership_start)) ||
            FAILED(device->CreateQuery(&desc, &slot.ownership_end)) ||
            FAILED(device->CreateQuery(&desc, &slot.pre_start)) ||
            FAILED(device->CreateQuery(&desc, &slot.pre_end))) {
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
        UINT64 parallax_frame_stats_start = 0;
        UINT64 parallax_frame_stats_end = 0;
        UINT64 parallax_start = 0;
        UINT64 parallax_end = 0;
        UINT64 parallax_map_start = 0;
        UINT64 parallax_subtitle_start = 0;
        UINT64 ownership_start = 0;
        UINT64 ownership_end = 0;
        UINT64 pre_start = 0;
        UINT64 pre_end = 0;
        constexpr UINT nonblocking = D3D11_ASYNC_GETDATA_DONOTFLUSH;
        const auto post_start_status = context->GetData(
          slot.post_start.Get(), &post_start, sizeof(post_start),
          nonblocking
        );
        const auto post_end_status = context->GetData(
          slot.post_end.Get(), &post_end, sizeof(post_end),
          nonblocking
        );
        const auto pre_start_status = context->GetData(
          slot.pre_start.Get(), &pre_start, sizeof(pre_start),
          nonblocking
        );
        const auto pre_end_status = context->GetData(
          slot.pre_end.Get(), &pre_end, sizeof(pre_end),
          nonblocking
        );
        HRESULT parallax_start_status = S_OK;
        HRESULT parallax_end_status = S_OK;
        HRESULT parallax_frame_stats_start_status = S_OK;
        HRESULT parallax_frame_stats_end_status = S_OK;
        HRESULT parallax_map_start_status = S_OK;
        HRESULT parallax_subtitle_start_status = S_OK;
        HRESULT ownership_start_status = S_OK;
        HRESULT ownership_end_status = S_OK;
        if (slot.has_parallax_frame_stats) {
          parallax_frame_stats_start_status = context->GetData(
            slot.parallax_frame_stats_start.Get(),
            &parallax_frame_stats_start,
            sizeof(parallax_frame_stats_start),
            nonblocking
          );
          parallax_frame_stats_end_status = context->GetData(
            slot.parallax_frame_stats_end.Get(),
            &parallax_frame_stats_end,
            sizeof(parallax_frame_stats_end),
            nonblocking
          );
        }
        if (slot.has_parallax) {
          parallax_start_status = context->GetData(
            slot.parallax_start.Get(), &parallax_start, sizeof(parallax_start),
            nonblocking
          );
          parallax_end_status = context->GetData(
            slot.parallax_end.Get(), &parallax_end, sizeof(parallax_end),
            nonblocking
          );
        }
        if (slot.has_parallax_map_start) {
          parallax_map_start_status = context->GetData(
            slot.parallax_map_start.Get(),
            &parallax_map_start,
            sizeof(parallax_map_start),
            nonblocking
          );
        }
        if (slot.has_parallax_subtitle_start) {
          parallax_subtitle_start_status = context->GetData(
            slot.parallax_subtitle_start.Get(),
            &parallax_subtitle_start,
            sizeof(parallax_subtitle_start),
            nonblocking
          );
        }
        if (slot.has_ownership) {
          ownership_start_status = context->GetData(
            slot.ownership_start.Get(), &ownership_start, sizeof(ownership_start),
            nonblocking
          );
          ownership_end_status = context->GetData(
            slot.ownership_end.Get(), &ownership_end, sizeof(ownership_end),
            nonblocking
          );
        }
        const bool any_pending =
          post_start_status == S_FALSE || post_end_status == S_FALSE ||
          pre_start_status == S_FALSE || pre_end_status == S_FALSE ||
          parallax_frame_stats_start_status == S_FALSE ||
          parallax_frame_stats_end_status == S_FALSE ||
          parallax_start_status == S_FALSE || parallax_end_status == S_FALSE ||
          parallax_map_start_status == S_FALSE ||
          parallax_subtitle_start_status == S_FALSE ||
          ownership_start_status == S_FALSE || ownership_end_status == S_FALSE;
        if (any_pending) {
          continue;
        }
        if (post_start_status == S_OK && post_end_status == S_OK &&
            pre_start_status == S_OK && pre_end_status == S_OK &&
            parallax_frame_stats_start_status == S_OK &&
            parallax_frame_stats_end_status == S_OK &&
            parallax_start_status == S_OK && parallax_end_status == S_OK &&
            parallax_map_start_status == S_OK &&
            parallax_subtitle_start_status == S_OK &&
            ownership_start_status == S_OK && ownership_end_status == S_OK &&
            !timing.Disjoint && timing.Frequency > 0 && post_end >= post_start &&
            pre_start >= post_end && pre_end >= pre_start &&
            (!slot.has_parallax_frame_stats ||
              (slot.has_parallax && parallax_frame_stats_start >= post_start &&
               parallax_frame_stats_end >= parallax_frame_stats_start &&
               parallax_start >= parallax_frame_stats_end)) &&
            (!slot.has_parallax ||
              (parallax_start >= post_start && parallax_end >= parallax_start &&
               post_end >= parallax_end)) &&
            (!slot.has_parallax_map_start ||
             (slot.has_parallax && parallax_map_start >= parallax_start &&
              parallax_end >= parallax_map_start)) &&
            (!slot.has_parallax_subtitle_start ||
             (slot.has_parallax &&
              parallax_subtitle_start >= parallax_start &&
              parallax_end >= parallax_subtitle_start)) &&
            (!slot.has_ownership ||
             (slot.has_parallax && ownership_start >= parallax_start &&
              ownership_end >= ownership_start && parallax_end >= ownership_end)) &&
            (!slot.has_parallax_map_start || !slot.has_ownership ||
             ownership_start >= parallax_map_start) &&
            (!slot.has_parallax_subtitle_start || !slot.has_ownership ||
             parallax_subtitle_start >= ownership_end)) {
          const double to_ms = 1000.0 / static_cast<double>(timing.Frequency);
          if (slot.has_post) {
            sbs_perf::add_sample_ms_if_current(
              "depth_postprocess_gpu",
              static_cast<double>(post_end - post_start) * to_ms,
              slot.perf_generation
            );
          }
          if (slot.has_parallax) {
            const UINT64 frame_stats_ticks = slot.has_parallax_frame_stats ?
                                               parallax_frame_stats_end -
                                                 parallax_frame_stats_start :
                                               0u;
            sbs_perf::add_sample_ms_if_current(
              "depth_parallax_gpu",
              static_cast<double>(
                frame_stats_ticks + parallax_end - parallax_start
              ) * to_ms,
              slot.perf_generation
            );
          }
          if (slot.has_parallax_map_start) {
            const UINT64 frame_stats_ticks = slot.has_parallax_frame_stats ?
                                               parallax_frame_stats_end -
                                                 parallax_frame_stats_start :
                                               0u;
            sbs_perf::add_sample_ms_if_current(
              "depth_parallax_stats_gpu",
              static_cast<double>(
                frame_stats_ticks + parallax_map_start - parallax_start
              ) * to_ms,
              slot.perf_generation
            );
          }
          if (slot.has_parallax_map_start && slot.has_ownership) {
            sbs_perf::add_sample_ms_if_current(
              "depth_parallax_map_gpu",
              static_cast<double>(ownership_start - parallax_map_start) * to_ms,
              slot.perf_generation
            );
          }
          if (slot.has_parallax_subtitle_start && slot.has_ownership) {
            sbs_perf::add_sample_ms_if_current(
              "depth_parallax_limits_gpu",
              static_cast<double>(parallax_subtitle_start - ownership_end) * to_ms,
              slot.perf_generation
            );
          }
          if (slot.has_parallax_subtitle_start) {
            sbs_perf::add_sample_ms_if_current(
              "depth_parallax_subtitle_gpu",
              static_cast<double>(parallax_end - parallax_subtitle_start) * to_ms,
              slot.perf_generation
            );
          }
          if (slot.has_ownership) {
            sbs_perf::add_sample_ms_if_current(
              "depth_parallax_ownership_gpu",
              static_cast<double>(ownership_end - ownership_start) * to_ms,
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
        slot.has_parallax_frame_stats = false;
        slot.has_parallax = false;
        slot.has_parallax_map_start = false;
        slot.has_parallax_subtitle_start = false;
        slot.has_ownership = false;
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

    bool ensure_telemetry_readback() {
      if (telemetry_readback_ready) {
        return true;
      }
      if (telemetry_readback_init_failed || !cut_state_buf) {
        return false;
      }

      D3D11_BUFFER_DESC source_desc {};
      cut_state_buf->GetDesc(&source_desc);
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
      std::chrono::steady_clock::time_point sampled_at,
      depth_telemetry_sample &sample
    ) {
      using sbs_adaptive_state::word_e;
      const auto scalar = [&](const word_e word) {
        return std::bit_cast<float>(words[sbs_adaptive_state::index(word)]);
      };
      if (words[sbs_adaptive_state::index(word_e::cut_contract_tag_bits)] !=
          sbs_adaptive_state::cut_contract_tag) {
        return false;
      }
      for (const auto &field : sbs_adaptive_state::fields) {
        const auto word_index = sbs_adaptive_state::index(field.word);
        if (field.name.starts_with("reserved_") &&
            words[word_index] != sbs_adaptive_state::initial_words[word_index]) {
          return false;
        }
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
      const float model_input_history_state =
        scalar(word_e::model_input_history_state);
      const float hard_cut_pulse = scalar(word_e::hard_cut_pulse);
      if (
        scene_age < 0.0f ||
        cut_flags < 0.0f ||
        cut_flags > static_cast<float>(sbs_adaptive_state::known_cut_flag_mask) ||
        std::trunc(cut_flags) != cut_flags ||
        analysis_flags < 0.0f ||
        analysis_flags >
          static_cast<float>(sbs_adaptive_state::known_analysis_flag_mask) ||
        std::trunc(analysis_flags) != analysis_flags ||
        model_input_history_state < 0.0f ||
        model_input_history_state > 4.0f ||
        std::trunc(model_input_history_state) != model_input_history_state ||
        (hard_cut_pulse != 0.0f && hard_cut_pulse != 1.0f) ||
        words[sbs_adaptive_state::index(word_e::hard_cut_count)] >
          sbs_adaptive_state::counter_max ||
        words[sbs_adaptive_state::index(word_e::empty_raw_count)] >
          sbs_adaptive_state::counter_max ||
        words[sbs_adaptive_state::index(word_e::collapsed_raw_count)] >
          sbs_adaptive_state::counter_max
      ) {
        return false;
      }

      sample.depth_width = depth_width;
      sample.depth_height = depth_height;
      // Protocol-13 telemetry retains these pre-V2 columns for client wire compatibility. The
      // V2 cut bridge has no adaptive-pop, subject, or zero-anchor authority, so publish explicit
      // unavailable/default values instead of decoding reserved GPU words.
      sample.adaptive_pop_ratio = 1.0f;
      sample.edge_fraction = -1.0f;
      sample.change_fraction = scalar(word_e::current_depth_change_fraction);
      sample.valid_depth_fraction = scalar(word_e::valid_depth_fraction);
      sample.effective_range_width = scalar(word_e::effective_raw_range_width);
      sample.current_edge_fraction = -1.0f;
      sample.current_zero_anchor_candidate_shift_px = -1.0f;
      sample.structural_change_fraction =
        scalar(word_e::structural_change_fraction);
      sample.raw_rgb_change_fraction = scalar(word_e::raw_rgb_change_fraction);
      sample.analysis_flags = static_cast<std::uint32_t>(analysis_flags);
      sample.model_input_history_state =
        static_cast<std::uint32_t>(model_input_history_state);
      sample.zero_anchor_shift_px = 0.0f;
      sample.subject_depth = 0.0f;
      sample.scene_age = static_cast<std::uint32_t>(std::min(
        scene_age,
        static_cast<float>(std::numeric_limits<std::uint32_t>::max())
      ));
      sample.cut_flags = static_cast<std::uint32_t>(std::min(
        cut_flags,
        static_cast<float>(std::numeric_limits<std::uint32_t>::max())
      ));
      // The cut bridge stores counters as uint bits so they remain exact past float's 24-bit
      // integer range. The shader saturates them one value below UINT_MAX.
      sample.hard_cut_count =
        words[sbs_adaptive_state::index(word_e::hard_cut_count)];
      // Protocol 13 retains this wire slot for older clients. V2 has no external-reset
      // controller, so its compatibility value is permanently zero rather than a fake counter.
      sample.external_cut_count = 0u;
      sample.empty_raw_count =
        words[sbs_adaptive_state::index(word_e::empty_raw_count)];
      sample.collapsed_raw_count =
        words[sbs_adaptive_state::index(word_e::collapsed_raw_count)];
      sample.sampled_frame_id = sampled_frame_id;
      sample.sampled_at = sampled_at;
      sample.profile_initialized = scalar(word_e::initialized) > 0.5f;
      sample.anchor_valid = false;
      sample.range_collapsed = scalar(word_e::range_collapsed) > 0.5f;
      sample.depth_ready = scalar(word_e::depth_ready) > 0.5f;
      sample.hard_cut_pulse = hard_cut_pulse > 0.5f;
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
              slot.sampled_at,
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
          // every frame-critical consumer of the cut bridge.
          slot.sampled_at = std::chrono::steady_clock::now();
          context->CopyResource(slot.staging.Get(), cut_state_buf.Get());
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
      const CUresult query = cuda.cuEventQuery(r.stop[slot]);
      if (query == CUDA_ERROR_NOT_READY) {
        return;
      }
      if (query != CUDA_SUCCESS) {
        (void) observe_joined_cuda_failure(
          "diagnostic timing-event query", query, false
        );
        return;
      }
      float ms = 0.0f;
      const CUresult elapsed = cuda.cuEventElapsedTime ?
                                 cuda.cuEventElapsedTime(
                                   &ms, r.start[slot], r.stop[slot]
                                 ) :
                                 static_cast<CUresult>(-1);
      if (elapsed != CUDA_SUCCESS) {
        (void) observe_joined_cuda_failure(
          "diagnostic timing-event elapsed query", elapsed, false
        );
        return;
      }
      sbs_perf::add_sample_ms(r.stage, ms);
      r.busy[slot] = false;
    }

    void perf_drain(perf_evt_ring &r) {
      auto &cuda = cuda_driver_api::get();
      for (int i = 0; i < perf_evt_ring::N; i++) {
        perf_try_resolve(r, i, cuda);
        if (terminal_failure) {
          break;
        }
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
      if (terminal_failure || r.busy[slot]) {
        return -1;  // terminal, or still in flight: do not start another measurement
      }
      if (!r.start[slot]) {
        const CUresult created = cuda.cuEventCreate(
          &r.start[slot], CU_EVENT_DEFAULT
        );
        if (created != CUDA_SUCCESS) {
          (void) observe_joined_cuda_failure(
            "diagnostic timing-event creation", created, false
          );
          return -1;
        }
      }
      if (!r.stop[slot]) {
        const CUresult created = cuda.cuEventCreate(
          &r.stop[slot], CU_EVENT_DEFAULT
        );
        if (created != CUDA_SUCCESS) {
          (void) observe_joined_cuda_failure(
            "diagnostic timing-event creation", created, false
          );
          return -1;
        }
      }
      const CUresult recorded = cuda.cuEventRecord(r.start[slot], stream);
      if (recorded != CUDA_SUCCESS) {
        (void) observe_joined_cuda_failure(
          "diagnostic timing-event start record", recorded, false
        );
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
      const CUresult recorded = cuda.cuEventRecord ?
                                  cuda.cuEventRecord(r.stop[slot], stream) :
                                  static_cast<CUresult>(-1);
      if (recorded != CUDA_SUCCESS) {
        (void) observe_joined_cuda_failure(
          "diagnostic timing-event stop record", recorded, false
        );
        return;
      }
      r.busy[slot] = true;
      r.head = (r.head + 1) % perf_evt_ring::N;
    }

    [[nodiscard]] bool perf_destroy_events(cuda_driver_api &cuda) {
      for (auto *r : {&perf_depth_conditional}) {
        for (int i = 0; i < perf_evt_ring::N; i++) {
          for (auto event : {
                 std::pair<CUevent *, const char *> {&r->start[i], "start"},
                 std::pair<CUevent *, const char *> {&r->stop[i], "stop"},
               }) {
            if (!*event.first) {
              continue;
            }
            const bool destroy_api_available = cuda.cuEventDestroy != nullptr;
            const CUresult destroyed = destroy_api_available ?
                                         cuda.cuEventDestroy(*event.first) :
                                         static_cast<CUresult>(-1);
            if (detail::teardown_cuda_handle_may_be_forgotten(
                  true,
                  destroy_api_available,
                  destroyed == CUDA_SUCCESS
                )) {
              *event.first = nullptr;
              continue;
            }
            // Keep the failed raw handle non-null. CUevent has no value destructor, so estimator
            // destruction deliberately leaks this and every later event together with retained
            // operands. Do not issue another CUDA teardown call after a possibly deferred fault.
            BOOST_LOG(warning)
              << "Depth-estimator diagnostic " << event.second << " event " << i
              << " destroy failed: result=" << destroyed
              << "; retaining CUDA operands and quarantining participating contexts.";
            return false;
          }
        }
      }
      return true;
    }

    [[nodiscard]] bool reset_depth_conditional_graph() noexcept {
      const bool reset_ok = depth_conditional_graph.reset();
      const bool wrapper_released = depth_conditional_graph.empty();
      if (!reset_ok) {
        // A failed restore after successful destruction no longer owns dependencies, but it still
        // proves the caller's CUDA-context state is unsafe for reuse. A failure before destruction
        // additionally leaves wrapper_released false, which makes every caller retain backing.
        execution_context_poisoned = true;
        gpu_conditional_bridge_context_failed = true;
        if (depth_conditional_optional_child_graph) {
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::unsafe_teardown
          );
        }
        mark_terminal_failure(true);
      }
      if (!wrapper_released) {
        // The wrapper still owns nodes embedded from the TensorRT source graph and captures the
        // mapped decision pointer. Losing either dependency here would be a use-after-free in the
        // CUDA driver. Quarantine the context and retain every dependency until terminal teardown.
        execution_context_poisoned = true;
        if (depth_conditional_optional_child_graph) {
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::unsafe_teardown
          );
        }
        mark_terminal_failure(true);
        return false;
      }
      depth_conditional_decision_ptr = 0u;
      depth_conditional_request_ptr = 0u;
      depth_conditional_child_graph = nullptr;
      depth_conditional_optional_child_graph = nullptr;
      // Even when every wrapper handle was destroyed, a failed context restore is a CUDA teardown
      // failure. Callers must stop before unregistering or destroying the wrapper's dependencies.
      return reset_ok;
    }

    [[nodiscard]] bool destroy_inference_graph(
      cuda_driver_api &cuda,
      tensorrt_cuda_graph_t &state
    ) {
      // The wrapper owns node-cloned children embedded from both captured TensorRT source graphs.
      // It must die before either source is replaced after a signature transition. If wrapper
      // teardown cannot release those clones, retain the source graph/context/interops rather than
      // creating a dangling embedded child.
      const bool wrapper_owns_state =
        (&state == &depth_inference_graph &&
         depth_conditional_child_graph == state.graph) ||
        (&state == &ocr_inference_graph &&
         depth_conditional_optional_child_graph == state.graph);
      if (wrapper_owns_state) {
        if (!reset_depth_conditional_graph()) {
          return false;
        }
      }
      if (state.graph) {
        if (!cuda.cuGraphDestroy ||
            cuda.cuGraphDestroy(state.graph) != CUDA_SUCCESS) {
          mark_shared_execution_failure(ocr_exec_context != nullptr);
          return false;
        }
        state.graph = nullptr;
      }
      state.policy.signature_warmed = false;
      return true;
    }

    // D3D interop is allowed to return different device pointers after any map. TensorRT graph
    // nodes retain the context's addresses (and depth's dynamic shape), so a graph is reusable
    // only for one exact signature. OCR calls this before rebinding; the common enqueue path also
    // calls it defensively so the depth and OCR lifecycle cannot drift apart.
    [[nodiscard]] bool select_inference_graph_signature(
      cuda_driver_api &cuda,
      tensorrt_cuda_graph_t &state,
      const detail::cuda_graph_signature_t &signature
    ) {
      if (!detail::select_cuda_graph_signature(state.policy, signature)) {
        return true;
      }
      return destroy_inference_graph(cuda, state);
    }

    bool prepare_depth_inference_graph(
      cuda_driver_api &cuda,
      const bool allow_private_bootstrap
    ) {
      const bool graph_api = cuda.cuStreamBeginCapture &&
                             cuda.cuStreamEndCapture && cuda.cuGraphDestroy;
      if (!graph_api || depth_inference_graph.policy.capture_failed) {
        fail_gpu_conditional_bridge_once(
          !graph_api ?
            "required DAV2 graph-capture APIs are unavailable" :
            "DAV2 graph capture previously failed"
        );
        return false;
      }

      // The quiescent pre-binding path selected this exact signature before mutating TensorRT
      // addresses. This helper owns only the private warm/capture bootstrap for that selection.
      if (depth_inference_graph.graph) {
        return true;
      }
      if (!allow_private_bootstrap) {
        fail_gpu_conditional_bridge_once(
          "DAV2 graph signature changed during a GPU-undecided transaction"
        );
        return false;
      }

      // TensorRT requires one ordinary enqueue after a dynamic-shape/address signature change
      // before that exact signature may be captured. This is a private bootstrap only: the raw
      // result is immediately overwritten by the wrapper's force-infer launch on the same stream
      // and is never exposed as a pending or rendered completion.
      if (!depth_inference_graph.policy.signature_warmed) {
        if (!exec_context->enqueueV3(cu_stream)) {
          depth_inference_graph.policy.capture_failed = true;
          fail_gpu_conditional_bridge_once(
            "private DAV2 signature bootstrap enqueue failed",
            CUDA_SUCCESS,
            true
          );
          return false;
        }
        joined_stream_work_ever_submitted = true;
        depth_inference_graph.policy.signature_warmed = true;
      }

      CUgraph captured = nullptr;
      const CUresult begin = cuda.cuStreamBeginCapture(
        cu_stream,
        CU_STREAM_CAPTURE_MODE_RELAXED
      );
      const bool captured_enqueue =
        begin == CUDA_SUCCESS && exec_context->enqueueV3(cu_stream);
      const CUresult end = begin == CUDA_SUCCESS ?
                             cuda.cuStreamEndCapture(cu_stream, &captured) :
                             begin;
      if (captured_enqueue && end == CUDA_SUCCESS && captured) {
        depth_inference_graph.graph = captured;
        BOOST_LOG(info)
          << "DAV2 private signature bootstrap captured " << target_w << 'x' << target_h
          << "; authoritative launches now require the GPU conditional wrapper.";
        return true;
      }

      depth_inference_graph.graph = captured;
      (void) destroy_inference_graph(cuda, depth_inference_graph);
      depth_inference_graph.policy.capture_failed = true;
      fail_gpu_conditional_bridge_once(
        "private DAV2 graph capture failed (begin=" + std::to_string(begin) +
          ", enqueue=" + std::to_string(captured_enqueue) +
          ", end=" + std::to_string(end) + ')',
        end,
        true
      );
      return false;
    }

    // Capture only the OCR TensorRT source graph. It is never instantiated or launched raw:
    // build() embeds it under the authenticated optional-infer handle beside DAV2, and one root
    // event therefore joins both isolated execution contexts without a branch readback.
    bool prepare_ocr_inference_graph(
      cuda_driver_api &cuda,
      const bool allow_private_bootstrap
    ) {
      if (!ocr_available) {
        return true;
      }
      const bool graph_api = cuda.cuStreamBeginCapture &&
                             cuda.cuStreamEndCapture && cuda.cuGraphDestroy;
      if (!graph_api || ocr_inference_graph.policy.capture_failed) {
        ocr_available = false;
        if (!ocr_error_logged) {
          BOOST_LOG(warning)
            << "PP-OCRv6 tiny conditional graph capture is unavailable; depth remains active "
               "and subtitle conditioning stays flat.";
          ocr_error_logged = true;
        }
        return true;
      }
      if (ocr_inference_graph.graph) {
        return true;
      }
      if (!allow_private_bootstrap) {
        fail_gpu_conditional_bridge_once(
          "OCR graph signature changed during a GPU-undecided transaction",
          CUDA_SUCCESS,
          true
        );
        return false;
      }
      if (!ocr_inference_graph.policy.signature_warmed) {
        if (!ocr_exec_context->enqueueV3(cu_stream)) {
          ocr_inference_graph.policy.capture_failed = true;
          mark_ocr_enqueue_failure("private signature bootstrap enqueueV3", CUDA_SUCCESS);
          return false;
        }
        joined_stream_work_ever_submitted = true;
        joined_stream_ocr_ever_submitted_or_armed = true;
        ocr_inference_graph.policy.signature_warmed = true;
      }

      CUgraph captured = nullptr;
      const CUresult begin = cuda.cuStreamBeginCapture(
        cu_stream,
        CU_STREAM_CAPTURE_MODE_RELAXED
      );
      if (begin != CUDA_SUCCESS) {
        ocr_inference_graph.policy.capture_failed = true;
        // BeginCapture may surface the immediately preceding private bootstrap's asynchronous
        // failure. That enqueue can have partially submitted either shared-stream context, so an
        // ordinary depth-only continuation would hide a poisoned joined transaction.
        mark_ocr_enqueue_failure("conditional source graph capture begin", begin);
        return false;
      }
      const bool captured_enqueue = ocr_exec_context->enqueueV3(cu_stream);
      const CUresult end = cuda.cuStreamEndCapture(cu_stream, &captured);
      if (captured_enqueue && end == CUDA_SUCCESS && captured) {
        ocr_inference_graph.graph = captured;
        BOOST_LOG(info)
          << "PP-OCRv6 tiny private source graph captured for fixed "
          << ocr_engine_width << 'x' << ocr_engine_height
          << "; it now runs only under authenticated optional inference.";
        return true;
      }

      ocr_inference_graph.graph = captured;
      const bool graph_released = destroy_inference_graph(
        cuda, ocr_inference_graph
      );
      ocr_inference_graph.policy.capture_failed = true;
      mark_ocr_enqueue_failure(
        "conditional source graph capture",
        end == CUDA_SUCCESS ? static_cast<CUresult>(-1) : end
      );
      execution_context_poisoned = execution_context_poisoned || !graph_released;
      return false;
    }

    // Caching
    int target_w = 0;
    int target_h = 0;
    UINT reduce_groups = 0;  // threadgroups for the shared moments/range reduction
    int cb_color_mode = -1;  // input_color_space baked into constant buffers
    depth_tensor_content_rect_t cb_tensor_content {};
    depth_source_rect_t cb_source_region {};
    // TensorRT retains dynamic shape and tensor-address bindings on an execution context. The
    // interop pointer is allowed to change after any map, so cache each piece independently and
    // rebind only the values that actually changed.
    int trt_bound_width = 0;
    int trt_bound_height = 0;
    CUdeviceptr trt_bound_input = 0;
    CUdeviceptr trt_bound_output = 0;
    CUdeviceptr ocr_bound_input = 0;
    CUdeviceptr ocr_bound_output = 0;

    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_fused_preprocess_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> buffer_to_tex_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> buffer_to_tex_pad_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_minmax_ema_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_hist_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_scene_cut_evidence_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_scene_cut_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_valid_history_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_moments_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_frame_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_state_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_map_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_ownership_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_coordinate_diagnostic_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_vertical_limit_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_limit_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> ocr_preprocess_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> ocr_box_cells_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> ocr_box_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_locator_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_condition_prepare_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_condition_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_history_owner_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_scene_seed_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_finalize_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_reuse_depth_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> gpu_trace_cs;
    Microsoft::WRL::ComPtr<ID3D11Buffer> cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> source_region_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> fused_preprocess_force_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> fused_preprocess_compare_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> depth_coordinate_v2_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_preprocess_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_resolve_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subtitle_locator_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> near_identical_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> near_identical_tile_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> near_identical_tile_srv;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> near_identical_tile_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> near_identical_history_owner_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> near_identical_history_owner_srv;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> near_identical_history_owner_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> near_identical_gpu_decision_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> near_identical_gpu_decision_srv;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> near_identical_gpu_decision_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> near_identical_gpu_dispatch_buf;
    Microsoft::WRL::ComPtr<ID3D11Buffer> gpu_trace_transaction_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> gpu_trace_transaction_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> gpu_trace_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> gpu_trace_ring_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> gpu_trace_ring_srv;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> gpu_trace_ring_uav;
    std::shared_ptr<const host_sbs_gpu_trace_provenance_t> gpu_trace_provenance;
    bool gpu_trace_error_logged = false;
    struct pending_gpu_trace_append_t {
      host_sbs_gpu_trace::host_subtitle_outcome_e host_subtitle_outcome =
        host_sbs_gpu_trace::host_subtitle_outcome_e::suppressed;
      bool ocr_record_submitted = false;
      bool subtitle_work_suppressed = false;
      bool condition_executed = false;
      bool subtitle_branch_gated = false;
      bool input_domain_reset = false;
      bool valid = false;
    } pending_gpu_trace_append;

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
    Microsoft::WRL::ComPtr<ID3D11Buffer> minmax_raw_buf;  // min/max bits, valid and unmasked-eligible counts
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> minmax_raw_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> minmax_ema_buf;  // float4 {min,max,initialized,frame_state}
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> minmax_ema_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> minmax_ema_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> hist_buf;  // 256 uint bins for percentile normalization
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> hist_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> scene_cut_evidence_buf;  // nine counters + frame delta
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> scene_cut_evidence_uav;
    // Established 32-word analysis ABI. V2 writes only its cut/health fields; the retired
    // subject/adaptive/stretch/zero-plane words keep their initial values.
    Microsoft::WRL::ComPtr<ID3D11Buffer> cut_state_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> cut_state_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> cut_state_srv;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_previous_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_previous_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_previous_srv;
    // Cut detection keeps a separate reliable depth endpoint. The ordinary previous texture must
    // still advance every frame for temporal EMA, including through clipped/structureless frames.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_cut_history_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_cut_history_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_cut_history_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> ema_motion_mask_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> ema_motion_mask_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ema_motion_mask_srv;
    // Synthetic letterbox support is visible to DAV2 but excluded from every analysis statistic,
    // history comparison, ownership decision, and positioned renderer sample.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> tensor_exclusion_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> tensor_exclusion_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> tensor_exclusion_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> tensor_previous_exclusion_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> tensor_previous_exclusion_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> tensor_previous_exclusion_srv;

    // GPU-only raw-coordinate V2 producer. The canonical-coordinate texture is exceptional: its
    // shader, texture, and views are created only for an explicit Dump 3D snapshot. The
    // authenticated final field is the sole live position authority.
    Microsoft::WRL::ComPtr<ID3D11Buffer> depth_coordinate_v2_partials_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_partials_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_partials_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> depth_coordinate_v2_frame_stats_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_frame_stats_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_frame_stats_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> depth_coordinate_v2_state_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_state_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_state_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_coordinate_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_coordinate_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_coordinate_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_candidate_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_candidate_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_candidate_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_ownership_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_ownership_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_ownership_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_vertical_majorant_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_vertical_majorant_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_vertical_majorant_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_vertical_conditioned_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>
      depth_coordinate_v2_vertical_conditioned_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>
      depth_coordinate_v2_vertical_conditioned_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_final_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_final_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_final_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> subtitle_conditioned_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subtitle_conditioned_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subtitle_conditioned_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subtitle_locator_state_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subtitle_locator_state_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subtitle_locator_state_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subtitle_condition_params_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subtitle_condition_params_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subtitle_condition_params_srv;
    bool depth_coordinate_v2_coordinate_diagnostic_error_logged = false;

    CUgraphicsResource cuda_in_res = nullptr;
    CUgraphicsResource cuda_out_res = nullptr;
    CUgraphicsResource cuda_near_identical_decision_res = nullptr;
    CUgraphicsResource cuda_ocr_in_res = nullptr;
    CUgraphicsResource cuda_ocr_out_res = nullptr;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_input_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> ocr_input_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_output_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ocr_output_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_cell_stats_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> ocr_cell_stats_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ocr_cell_stats_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_box_record_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> ocr_box_record_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ocr_box_record_srv;
    // The joined completion event is recorded after the single root stream has issued both child
    // branches and every interop-unmap tail. A ready event therefore proves the same resource
    // release boundary as the conservative full-stream query without exposing either branch.
    CUevent depth_inference_done_event = nullptr;
    bool near_identical_detector_available = false;
    bool near_identical_detector_error_logged = false;
    bool pending_depth_inference_event_recorded = false;
    bool inference_event_poll_available = false;
    bool inference_event_ever_recorded = false;
    // CUDA may report an earlier asynchronous bootstrap/root error through a later map, pointer,
    // or timing-event call even after nominal completion. Retain stream-lifetime participation so
    // those late failures quarantine every TensorRT context that could have executed.
    bool joined_stream_work_ever_submitted = false;
    bool joined_stream_ocr_ever_submitted_or_armed = false;
    // The host knows which authenticated subtitle transaction it submitted without learning the
    // DAV2 branch. Receipt-gated postprocess uses this exact expected disposition.
    cuda_conditional_graph::work_flag_e pending_subtitle_work =
      cuda_conditional_graph::work_flag_e::none;
    bool pending_ocr_submitted = false;
    // This host bit means the optional OCR child was present and requested in the joined root, not
    // that the device executed it. The authenticated receipt remains OCR-output authority.
    depth_optional_work_mode_e pending_optional_work =
      depth_optional_work_mode_e::ordinary;
    enum class pending_submission_class_e : std::uint8_t {
      force_infer,
      gpu_undecided,
    };
    pending_submission_class_e pending_submission_class =
      pending_submission_class_e::force_infer;
    std::uint64_t pending_wrapper_transaction_token = 0u;
    std::uint64_t pending_gpu_transaction_token = 0u;
    std::uint64_t pending_gpu_transaction_current_frame_id = 0u;
    std::uint64_t pending_gpu_transaction_baseline_frame_id = 0u;
    std::uint64_t pending_observation_timestamp_us = 0u;
    bool pending_dump_forced = false;
    std::uint64_t force_infer_transaction_sequence = 0u;
    bool gpu_undecided_requires_force_infer_refresh = false;
    bool ocr_signature_refresh_required = false;
    bool ocr_shape_bound = false;
    bool ocr_output_size_validated = false;
    bool ocr_error_logged = false;
    // The accepted input is a private matched-frame texture. Retain its SRV until the exact
    // asynchronous raw-depth completion has run the full-resolution ownership pass; this adds no
    // color copy and keeps every RGB/depth lookup on the same D3D11 command stream.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> pending_source_srv;
    input_color_space pending_color_space = input_color_space::srgb;
    depth_input_region_t pending_input_region {};
    depth_input_domain_tracker_t processed_input_domain;
    bool has_previous_frame = false;
    std::uint64_t pending_frame_id = 0;
    std::uint64_t last_postprocessed_frame_id = 0;
    bool has_last_postprocessed_frame_id = false;
    // Host metadata may authorize only the immediately preceding opaque root. The device history
    // owner remains the branch authority: infer advances it while reuse/invalid leaves it older.
    // The detector compares cumulatively against that last actual infer for at most four frame
    // steps and strictly less than 100 ms of source-observation time, so drift cannot be hidden by
    // pairwise-near-identical opaque follow-ups.
    std::uint64_t last_gpu_opaque_transaction_frame_id = 0;
    bool stream_error_logged = false;
    bool terminal_failure = false;
    bool execution_context_poisoned = false;
    bool readiness_preflighted = false;  // can_accept_frame() already counted/queried this source opportunity
    bool depth_context_pooled = false;  // context reused from the pool (modules already loaded -> skip warmup)
    bool context_warmed = false;  // only warmed contexts may return to context_pool

    bool live_v2_producer_unavailable() const {
      return parallax_v2_producer_failed;
    }

    static depth_input_region_t resolved_input_region(
      depth_input_region_t requested,
      const D3D11_TEXTURE2D_DESC &input_desc
    ) noexcept {
      if (!requested.video_region) {
        const auto tensor_shape = fit_host_sbs_v2_depth_tensor_shape(
          input_desc.Width,
          input_desc.Height
        );
        requested.source_width = input_desc.Width;
        requested.source_height = input_desc.Height;
        requested.left = 0u;
        requested.top = 0u;
        requested.right = input_desc.Width;
        requested.bottom = input_desc.Height;
        requested.tensor_content = {
          0u,
          0u,
          static_cast<std::uint32_t>(std::max(tensor_shape.width, 0)),
          static_cast<std::uint32_t>(std::max(tensor_shape.height, 0)),
        };
        requested.analysis_generation = 0u;
        return requested;
      }
      // Live ROI analysis samples the exact rectangle directly from the retained full matched
      // frame. A crop-sized SRV is deliberately rejected: accepting both layouts would make the
      // same placement metadata select two different physical coordinate systems.
      if (!requested.valid() || requested.source_width != input_desc.Width ||
          requested.source_height != input_desc.Height) {
        return {};
      }
      return requested;
    }

    void clear_pending_inference_event_state() noexcept {
      pending_depth_inference_event_recorded = false;
    }

    // Host V2 fails flat on any producer error. Context quarantine is owned separately by failure
    // provenance: pre-enqueue validation may leave a context reusable, while a CUDA error after
    // asynchronous bootstrap/root work was submitted may be deferred and quarantines every
    // context that participated.
    void mark_terminal_failure(const bool poison_execution_context = false) {
      clear_pending_inference_event_state();
      execution_context_poisoned =
        execution_context_poisoned || poison_execution_context;
      terminal_failure = true;
    }

    void mark_ocr_context_failure(
      const detail::warmed_execution_context_failure_e failure
    ) noexcept {
      ocr_context_health.observe(failure);
    }

    /** Apply the canonical fail-flat response to a CUDA call around the joined root stream.
     *
     * Required map/pointer operations are terminal even before the first launch. Optional timing
     * calls remain optional only until asynchronous bootstrap or root work has been submitted;
     * afterward their error result may be a deferred report from DAV2 or OCR and therefore owns
     * the same quarantine response as a root query failure.
     */
    bool observe_joined_cuda_failure(
      const std::string_view operation,
      const CUresult result,
      const bool required_operation
    ) {
      const auto policy = detail::joined_cuda_failure_policy(
        required_operation,
        joined_stream_work_ever_submitted,
        joined_stream_ocr_ever_submitted_or_armed
      );
      if (!policy.terminal) {
        return false;
      }
      if (!stream_error_logged) {
        BOOST_LOG(error) << "DAV2/OCR joined CUDA " << operation
                         << " failed: result=" << result;
        stream_error_logged = true;
      }
      if (policy.quarantine_ocr) {
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::asynchronous_execution_or_query
        );
      }
      mark_terminal_failure(policy.quarantine_depth);
      return true;
    }

    void fail_gpu_conditional_bridge_once(
      const std::string_view reason,
      const CUresult result = CUDA_SUCCESS,
      const bool poison_execution_context = false,
      const bool retain_live_wrapper = false
    ) {
      gpu_conditional_bridge_available = false;
      if (detail::bridge_failure_quarantines_ocr(
            poison_execution_context,
            joined_stream_ocr_ever_submitted_or_armed
          )) {
        // A later depth-only rebuild/launch API can surface an asynchronous fault from an earlier
        // optional child on the same stream. Current suppression does not erase that ownership.
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::asynchronous_execution_or_query
        );
      }
      // A failed launch can leave work from this executable in flight. Never destroy its embedded
      // child or mirror operands here; terminal teardown first quiesces the CUDA stream, then
      // releases the wrapper and every captured dependency in order.
      const bool wrapper_released = retain_live_wrapper ?
                                      depth_conditional_graph.empty() :
                                      reset_depth_conditional_graph();
      // Active arbitration is a required execution contract, not an optional accelerator. Even a
      // clean capability/topology rejection is terminal. Launch or unsafe teardown failures also
      // quarantine the TensorRT context so none of its captured dependencies can be reused.
      mark_terminal_failure(
        poison_execution_context || (!wrapper_released && !retain_live_wrapper)
      );
      if (!gpu_conditional_bridge_error_logged) {
        BOOST_LOG(error)
          << "Host SBS GPU near-identical arbitration failed (" << reason
          << (result == CUDA_SUCCESS ? "" : ", CUDA result " + std::to_string(result))
          << (retain_live_wrapper && !wrapper_released ?
                "); the live wrapper is retained until quiescent terminal teardown." :
              wrapper_released ?
                "); this estimator is terminal and will render flat without raw-graph fallback." :
                "); teardown was unsafe; dependencies are retained and the estimator is quarantined.");
        gpu_conditional_bridge_error_logged = true;
      }
    }

    bool ensure_depth_conditional_graph(
      cuda_driver_api &cuda,
      const CUdeviceptr decision_record,
      const CUdeviceptr request_record,
      const CUgraph optional_infer_child,
      const conditional_optional_child_policy_e optional_child_policy
    ) {
      const char *precondition_failure = nullptr;
      if (!gpu_conditional_bridge_available) {
        precondition_failure = "the bridge was previously disabled";
      } else if (!cuda.has_conditional_graph_support()) {
        precondition_failure = "required CUDA conditional/topology APIs are unavailable";
      } else if (!cuda_ctx) {
        precondition_failure = "the CUDA context is unavailable";
      } else if (!depth_inference_graph.graph) {
        precondition_failure = "the captured DAV2 graph is unavailable";
      } else if (depth_inference_graph.policy.capture_failed) {
        precondition_failure = "DAV2 graph capture previously failed";
      } else if (
                 optional_child_policy ==
                   conditional_optional_child_policy_e::build_ready &&
                 !optional_infer_child) {
        precondition_failure = "the mapped OCR child requested for construction is unavailable";
      } else if (optional_infer_child &&
                 (optional_infer_child != ocr_inference_graph.graph ||
                  ocr_inference_graph.policy.capture_failed)) {
        precondition_failure = "the captured OCR graph is not a valid optional child";
      } else if (
                 optional_infer_child &&
                 optional_child_policy !=
                   conditional_optional_child_policy_e::build_ready) {
        precondition_failure = "an unmapped OCR child cannot be used for construction";
      } else if (!decision_record || !request_record ||
                 (decision_record & 15u) != 0u || (request_record & 15u) != 0u) {
        precondition_failure = "the device transaction records are missing or misaligned";
      }
      if (precondition_failure) {
        if (!terminal_failure) {
          fail_gpu_conditional_bridge_once(precondition_failure);
        }
        return false;
      }
      const bool wrapper_base_matches =
        depth_conditional_graph.ready() &&
        depth_conditional_decision_ptr == decision_record &&
        depth_conditional_request_ptr == request_record &&
        depth_conditional_child_graph == depth_inference_graph.graph;
      const auto optional_topology = select_conditional_optional_topology(
        wrapper_base_matches,
        depth_conditional_optional_child_graph != nullptr,
        depth_conditional_optional_child_graph == optional_infer_child,
        optional_child_policy
      );
      if (optional_topology.reuse_existing) {
        return true;
      }

      if (!reset_depth_conditional_graph() || terminal_failure) {
        return false;
      }
      const CUgraph optional_child_for_build =
        optional_topology.build_with_optional ? optional_infer_child : nullptr;
      auto candidate = cuda_conditional_graph::executable_t::build(
        cuda,
        {
          .context = cuda_ctx,
          .infer_child = depth_inference_graph.graph,
          .optional_infer_child = optional_child_for_build,
          .reuse_child = nullptr,
          .decision_record = decision_record,
          .request_record = request_record,
        }
      );
      if (!candidate.ready()) {
        const auto result = candidate.cuda_result();
        const auto build_failure = candidate.failure();
        const auto audit = candidate.audit_result();
        if (!candidate.empty()) {
          // A failed build normally self-cleans. If the CUDA driver could not destroy a partial
          // wrapper, take ownership of it so the child graph and mapped records cannot outlive
          // their backing. This estimator is terminal; teardown will retry under its context and
          // deliberately retain dependencies if the driver still refuses destruction.
          depth_conditional_graph = std::move(candidate);
          depth_conditional_decision_ptr = decision_record;
          depth_conditional_request_ptr = request_record;
          depth_conditional_child_graph = depth_inference_graph.graph;
          depth_conditional_optional_child_graph = optional_child_for_build;
          gpu_conditional_bridge_available = false;
          execution_context_poisoned = true;
          if (optional_child_for_build) {
            mark_ocr_context_failure(
              detail::warmed_execution_context_failure_e::
                asynchronous_execution_or_query
            );
          }
          mark_terminal_failure(true);
          if (!gpu_conditional_bridge_error_logged) {
            BOOST_LOG(warning)
              << "Host SBS GPU near-identical conditional graph construction left live CUDA "
                 "objects (CUDA result " << result
              << "); retaining all dependencies and quarantining this estimator.";
            gpu_conditional_bridge_error_logged = true;
          }
          return false;
        }
        BOOST_LOG(error)
          << "Conditional child topology audit: graphs=" << audit.visited_graphs
          << ", nodes=" << audit.visited_nodes
          << ", kernel=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_KERNEL]
          << ", memcpy=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_MEMCPY]
          << ", memset=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_MEMSET]
          << ", child=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_GRAPH]
          << ", empty=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_EMPTY]
          << ", wait-event=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_WAIT_EVENT]
          << ", record-event=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_EVENT_RECORD]
          << ", conditional=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_CONDITIONAL]
          << ".";
        if (optional_child_for_build) {
          // Any CUDA error while constructing the joined root may be a deferred report from one
          // of the private TensorRT bootstraps that preceded construction on this stream.
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::
              asynchronous_execution_or_query
          );
        }
        fail_gpu_conditional_bridge_once(
          "conditional graph construction failed at stage " +
            std::to_string(static_cast<unsigned>(build_failure)) +
            ", child audit " +
            std::to_string(static_cast<unsigned>(audit.failure)) +
            ", rejected node " +
            std::to_string(static_cast<int>(audit.rejected_type)),
          result,
          true
        );
        return false;
      }
      depth_conditional_graph = std::move(candidate);
      depth_conditional_decision_ptr = decision_record;
      depth_conditional_request_ptr = request_record;
      depth_conditional_child_graph = depth_inference_graph.graph;
      depth_conditional_optional_child_graph = optional_child_for_build;
      BOOST_LOG(info)
        << "Host SBS GPU near-identical arbitration is active with device-owned branch "
           "selection, optional OCR sibling=" << (optional_child_for_build ? "on" : "off")
        << ", and no decision readback.";
      return true;
    }

    // CUDA may report an earlier asynchronous launch error through a later root query,
    // synchronization, map, or unmap. Conservatively quarantine every TensorRT context embedded
    // in that root instead of assuming the failure belongs only to DAV2.
    void mark_shared_execution_failure(const bool ocr_may_have_participated) noexcept {
      mark_terminal_failure(true);
      if (ocr_may_have_participated || joined_stream_ocr_ever_submitted_or_armed) {
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::asynchronous_execution_or_query
        );
      }
    }

    void initialize_inference_done_events(cuda_driver_api &cuda) {
      clear_pending_inference_event_state();
      inference_event_poll_available = false;
      if (
        !cuda.cuEventCreate || !cuda.cuEventRecord || !cuda.cuEventQuery ||
        !cuda.cuEventDestroy
      ) {
        BOOST_LOG(warning)
          << "CUDA inference-event API is incomplete; same-frame completion uses one full-stream "
             "query without bounded waiting.";
        return;
      }

      const CUresult depth_create = cuda.cuEventCreate(
        &depth_inference_done_event,
        cuda_event_disable_timing
      );
      if (depth_create != CUDA_SUCCESS || !depth_inference_done_event) {
        BOOST_LOG(warning)
          << "Depth inference-event creation failed (" << depth_create
          << "); same-frame completion uses one full-stream query without bounded waiting.";
        (void) destroy_inference_done_events(cuda);
        return;
      }

      inference_event_poll_available = true;
    }

    bool destroy_inference_done_events(cuda_driver_api &cuda) noexcept {
      clear_pending_inference_event_state();
      inference_event_poll_available = false;
      if (!depth_inference_done_event) {
        return true;
      }
      const CUresult result = cuda.cuEventDestroy ?
                                cuda.cuEventDestroy(depth_inference_done_event) :
                                static_cast<CUresult>(-1);
      if (result != CUDA_SUCCESS) {
        BOOST_LOG(warning) << "CUDA root inference-event destruction failed: " << result;
        return false;
      }
      depth_inference_done_event = nullptr;
      return true;
    }

    bool record_inference_done_event(
      cuda_driver_api &cuda,
      CUevent event,
      CUstream stream,
      bool &recorded,
      const char *member,
      const bool ocr_was_submitted
    ) {
      recorded = false;
      const CUresult result = event && stream && cuda.cuEventRecord ?
                                cuda.cuEventRecord(event, stream) :
                                static_cast<CUresult>(-1);
      if (result != CUDA_SUCCESS) {
        BOOST_LOG(error) << member << " inference-event record failed: " << result;
        mark_shared_execution_failure(ocr_was_submitted);
        return false;
      }
      recorded = true;
      inference_event_ever_recorded = true;
      return true;
    }

    // Failing to select the shared CUDA context prevents proving the joined root's completion.
    void mark_cuda_context_failure() noexcept {
      mark_shared_execution_failure(pending_ocr_submitted);
    }

    void mark_ocr_enqueue_failure(
      const char *operation,
      const CUresult result
    ) noexcept {
      ocr_available = false;
      // enqueueV3(false) is not proof that no kernels were partially dispatched, and CUDA may
      // surface the failure through another context on the same driver context. Discard the depth
      // member too and quarantine both TensorRT contexts.
      mark_shared_execution_failure(true);
      if (!ocr_error_logged) {
        BOOST_LOG(warning) << "PP-OCRv6 tiny " << operation << " failed (" << result
                           << "); the exact-frame DAV2/OCR unit fails closed.";
        ocr_error_logged = true;
      }
    }

    static detail::async_stream_readiness_e normalized_cuda_readiness(
      const CUresult result
    ) noexcept {
      if (result == CUDA_SUCCESS) {
        return detail::async_stream_readiness_e::ready;
      }
      if (result == CUDA_ERROR_NOT_READY) {
        return detail::async_stream_readiness_e::busy;
      }
      return detail::async_stream_readiness_e::failed;
    }

    enum class pending_execution_readiness_e : std::uint8_t {
      ready,
      busy,
      failed,
    };

    pending_execution_readiness_e query_pending_execution(
      cuda_driver_api &cuda,
      const char *phase
    ) {
      const bool ocr_context_participated = pending_ocr_submitted;
      if (!cu_stream || !cuda.cuStreamQuery) {
        mark_shared_execution_failure(ocr_context_participated);
        return pending_execution_readiness_e::failed;
      }

      const CUresult root_query = cuda.cuStreamQuery(cu_stream);
      switch (normalized_cuda_readiness(root_query)) {
        case detail::async_stream_readiness_e::ready:
          return pending_execution_readiness_e::ready;
        case detail::async_stream_readiness_e::busy:
          return pending_execution_readiness_e::busy;
        case detail::async_stream_readiness_e::failed:
          if (!stream_error_logged) {
            BOOST_LOG(error) << "DAV2/OCR joined CUDA root query failed during " << phase
                             << ": result=" << root_query;
            stream_error_logged = true;
          }
          mark_shared_execution_failure(ocr_context_participated);
          return pending_execution_readiness_e::failed;
      }
      return pending_execution_readiness_e::failed;
    }

    pending_execution_readiness_e query_pending_inference_events(
      cuda_driver_api &cuda,
      const char *phase
    ) {
      const bool event_state_valid = cuda.cuEventQuery && depth_inference_done_event &&
                                     pending_depth_inference_event_recorded;
      if (!event_state_valid) {
        if (!stream_error_logged) {
          BOOST_LOG(error)
            << "DAV2/OCR root inference-event state was incomplete during " << phase << '.';
          stream_error_logged = true;
        }
        mark_shared_execution_failure(pending_ocr_submitted);
        return pending_execution_readiness_e::failed;
      }

      const CUresult root_query = cuda.cuEventQuery(depth_inference_done_event);
      switch (normalized_cuda_readiness(root_query)) {
        case detail::async_stream_readiness_e::ready:
          return pending_execution_readiness_e::ready;
        case detail::async_stream_readiness_e::busy:
          return pending_execution_readiness_e::busy;
        case detail::async_stream_readiness_e::failed:
          if (!stream_error_logged) {
            BOOST_LOG(error) << "DAV2/OCR root event query failed during " << phase
                             << ": result=" << root_query;
            stream_error_logged = true;
          }
          mark_shared_execution_failure(pending_ocr_submitted);
          return pending_execution_readiness_e::failed;
      }
      return pending_execution_readiness_e::failed;
    }

    bool synchronize_pending_execution(cuda_driver_api &cuda) {
      const bool ocr_participated = pending_ocr_submitted;
      if (!cu_stream || !cuda.cuStreamSynchronize) {
        mark_shared_execution_failure(ocr_participated);
        return false;
      }
      const CUresult root_sync = cuda.cuStreamSynchronize(cu_stream);
      if (root_sync != CUDA_SUCCESS) {
        BOOST_LOG(error) << "DAV2/OCR root synchronization failed: " << root_sync;
        mark_shared_execution_failure(ocr_participated);
        return false;
      }
      return true;
    }

    bool create_shader(
      const host_sbs_shader_cache::source_snapshot_t &sources,
      const host_sbs_shader_cache::shader_spec &spec,
      Microsoft::WRL::ComPtr<ID3D11ComputeShader> &out_cs
    ) {
      auto bytecode = host_sbs_shader_cache::get(sources, spec);
      if (!bytecode) {
        return false;
      }
      return SUCCEEDED(device->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &out_cs));
    }

    void log_near_identical_detector_failure_once(const std::string_view reason) {
      if (!near_identical_detector_error_logged) {
        BOOST_LOG(error) << "Host SBS GPU near-identical detector is unavailable ("
                         << reason << "); the mandatory DAV2 wrapper cannot run.";
        near_identical_detector_error_logged = true;
      }
    }

    void initialize_near_identical_detector() {
      const auto sources = host_sbs_shader_cache::snapshot_sources(
        shader_root,
        host_sbs_shader_cache::near_identical_detector_specs
      );
      if (!sources) {
        log_near_identical_detector_failure_once("shader source snapshot failed");
        return;
      }
      if (
        host_sbs_shader_cache::source_closure_sha256(sources) !=
          host_sbs_shader_cache::near_identical_detector_source_closure_sha256
      ) {
        log_near_identical_detector_failure_once(
          "shader source closure authentication failed"
        );
        return;
      }
      const bool shaders_ready =
        create_shader(
          sources,
          host_sbs_shader_cache::host_sbs_near_identical_resolve,
          near_identical_resolve_cs
        ) &&
        create_shader(
          sources,
          host_sbs_shader_cache::host_sbs_near_identical_history_owner,
          near_identical_history_owner_cs
        ) &&
        create_shader(
          sources,
          host_sbs_shader_cache::host_sbs_near_identical_scene_seed,
          near_identical_scene_seed_cs
        ) &&
        create_shader(
          sources,
          host_sbs_shader_cache::host_sbs_near_identical_finalize,
          near_identical_finalize_cs
        ) &&
        create_shader(
          sources,
          host_sbs_shader_cache::host_sbs_near_identical_reuse_depth,
          near_identical_reuse_depth_cs
        );
      D3D11_BUFFER_DESC constants_desc {};
      constants_desc.Usage = D3D11_USAGE_DEFAULT;
      constants_desc.ByteWidth = static_cast<UINT>(sizeof(near_identical_constants_t));
      constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const bool constants_ready = shaders_ready && SUCCEEDED(device->CreateBuffer(
        &constants_desc,
        nullptr,
        near_identical_cbuffer.ReleaseAndGetAddressOf()
      ));
      if (!constants_ready) {
        near_identical_resolve_cs.Reset();
        near_identical_history_owner_cs.Reset();
        near_identical_scene_seed_cs.Reset();
        near_identical_finalize_cs.Reset();
        near_identical_reuse_depth_cs.Reset();
        near_identical_cbuffer.Reset();
        log_near_identical_detector_failure_once(
          "shader or constant-buffer setup failed"
        );
        return;
      }
      near_identical_detector_available = true;
      BOOST_LOG(info)
        << "Host SBS GPU near-identical detector initialized from its independently "
           "authenticated mandatory closure.";
    }

    void log_gpu_trace_failure_once(const std::string_view reason) {
      if (!gpu_trace_error_logged) {
        BOOST_LOG(warning) << "Host SBS diagnostic GPU completion trace is unavailable ("
                           << reason << "); live rendering is unaffected.";
        gpu_trace_error_logged = true;
      }
    }

    void reset_gpu_trace_resources() noexcept {
      gpu_trace_cs.Reset();
      gpu_trace_transaction_buf.Reset();
      gpu_trace_transaction_srv.Reset();
      gpu_trace_cbuffer.Reset();
      gpu_trace_ring_buf.Reset();
      gpu_trace_ring_srv.Reset();
      gpu_trace_ring_uav.Reset();
      gpu_trace_provenance.reset();
      pending_gpu_trace_append = {};
    }

    void initialize_gpu_trace() {
      const auto sources = host_sbs_shader_cache::snapshot_sources(
        shader_root,
        host_sbs_shader_cache::gpu_trace_specs
      );
      const std::string closure_sha256 = sources ?
        host_sbs_shader_cache::source_closure_sha256(sources) : std::string {};
      if (!sources || closure_sha256 !=
            host_sbs_shader_cache::gpu_trace_source_closure_sha256 || !create_shader(
            sources,
            host_sbs_shader_cache::host_sbs_gpu_trace,
            gpu_trace_cs
          )) {
        reset_gpu_trace_resources();
        log_gpu_trace_failure_once(
          "shader source authentication or compilation failed"
        );
        return;
      }

      D3D11_BUFFER_DESC transaction_desc {};
      transaction_desc.Usage = D3D11_USAGE_DEFAULT;
      transaction_desc.ByteWidth = near_identical_gpu_decision_byte_count;
      transaction_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      transaction_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
      D3D11_SHADER_RESOURCE_VIEW_DESC transaction_srv_desc {};
      transaction_srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
      transaction_srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
      transaction_srv_desc.BufferEx.FirstElement = 0u;
      transaction_srv_desc.BufferEx.NumElements = near_identical_gpu_decision_word_count;
      transaction_srv_desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;

      std::vector<std::uint32_t> initial_ring(host_sbs_gpu_trace::ring_word_count, 0u);
      initial_ring[host_sbs_gpu_trace::word_index(
        host_sbs_gpu_trace::header_word_e::schema
      )] = host_sbs_gpu_trace::ring_schema;
      initial_ring[host_sbs_gpu_trace::word_index(
        host_sbs_gpu_trace::header_word_e::tag
      )] = host_sbs_gpu_trace::ring_tag;
      initial_ring[host_sbs_gpu_trace::word_index(
        host_sbs_gpu_trace::header_word_e::capacity
      )] = host_sbs_gpu_trace::capacity;
      initial_ring[host_sbs_gpu_trace::word_index(
        host_sbs_gpu_trace::header_word_e::record_words
      )] = host_sbs_gpu_trace::record_word_count;
      initial_ring[host_sbs_gpu_trace::word_index(
        host_sbs_gpu_trace::header_word_e::next_sequence_low
      )] = 1u;
      D3D11_BUFFER_DESC ring_desc {};
      ring_desc.Usage = D3D11_USAGE_DEFAULT;
      ring_desc.ByteWidth = host_sbs_gpu_trace::ring_byte_count;
      ring_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      ring_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      ring_desc.StructureByteStride = sizeof(std::uint32_t);
      D3D11_SUBRESOURCE_DATA ring_data {initial_ring.data(), 0u, 0u};

      D3D11_BUFFER_DESC constants_desc {};
      constants_desc.Usage = D3D11_USAGE_DEFAULT;
      constants_desc.ByteWidth =
        host_sbs_gpu_trace::constant_word_count * sizeof(std::uint32_t);
      constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

      const bool resources_ready =
        SUCCEEDED(device->CreateBuffer(
          &transaction_desc,
          nullptr,
          gpu_trace_transaction_buf.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateShaderResourceView(
          gpu_trace_transaction_buf.Get(),
          &transaction_srv_desc,
          gpu_trace_transaction_srv.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateBuffer(
          &ring_desc,
          &ring_data,
          gpu_trace_ring_buf.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateShaderResourceView(
          gpu_trace_ring_buf.Get(), nullptr, gpu_trace_ring_srv.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateUnorderedAccessView(
          gpu_trace_ring_buf.Get(), nullptr, gpu_trace_ring_uav.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateBuffer(
          &constants_desc, nullptr, gpu_trace_cbuffer.ReleaseAndGetAddressOf()
        ));
      if (!resources_ready) {
        reset_gpu_trace_resources();
        log_gpu_trace_failure_once("optional shader-resource setup failed");
        return;
      }

      gpu_trace_provenance =
        std::make_shared<const host_sbs_gpu_trace_provenance_t>(
          host_sbs_gpu_trace_provenance_t {
            .source_closure_schema = host_sbs_shader_cache::source_closure_schema,
            .source_compile_flags = host_sbs_shader_cache::shader_compile_flags,
            .source_macro_count = 0u,
            .source_closure_sha256 = closure_sha256,
          }
        );
      BOOST_LOG(info) << "Host SBS diagnostic GPU completion trace initialized ("
                      << host_sbs_gpu_trace::capacity << " records, "
                      << host_sbs_gpu_trace::record_word_count * sizeof(std::uint32_t)
                      << " bytes each).";
    }

    bool unregister_near_identical_decision_interop(cuda_driver_api &cuda) {
      if (!reset_depth_conditional_graph()) {
        fail_gpu_conditional_bridge_once(
          "conditional-wrapper teardown failed before decision-buffer unregister",
          CUDA_SUCCESS,
          true,
          true
        );
        return false;
      }
      if (unregister_cuda_graphics_resource(
            cuda, cuda_near_identical_decision_res
          )) {
        return true;
      }
      execution_context_poisoned = true;
      fail_gpu_conditional_bridge_once(
        "decision-buffer interop unregister failed",
        CUDA_SUCCESS,
        true,
        true
      );
      return false;
    }

    bool ensure_near_identical_decision_interop(cuda_driver_api &cuda) {
      if (cuda_near_identical_decision_res) {
        return true;
      }
      if (
        !gpu_conditional_bridge_available ||
        !cuda.has_conditional_graph_support() ||
        !near_identical_gpu_decision_buf ||
        !cuda.cuGraphicsD3D11RegisterResource
      ) {
        return false;
      }
      const CUresult registered = cuda.cuGraphicsD3D11RegisterResource(
        &cuda_near_identical_decision_res,
        near_identical_gpu_decision_buf.Get(),
        0u
      );
      if (
        registered == CUDA_SUCCESS && cuda_near_identical_decision_res
      ) {
        return true;
      }

      bool cleanup_ok = true;
      if (cuda_near_identical_decision_res) {
        cleanup_ok = unregister_cuda_graphics_resource(
          cuda, cuda_near_identical_decision_res
        );
      }
      if (!cleanup_ok) {
        execution_context_poisoned = true;
        mark_terminal_failure(true);
      }
      fail_gpu_conditional_bridge_once(
        cleanup_ok ?
          "decision-buffer CUDA registration failed" :
          "decision-buffer registration cleanup failed",
        registered
      );
      return false;
    }

    bool ensure_near_identical_detector_resources() {
      if (!near_identical_detector_available || target_w <= 0 || target_h <= 0) {
        return false;
      }
      if (
        near_identical_tile_buf && near_identical_tile_srv &&
        near_identical_tile_uav && near_identical_gpu_decision_buf &&
        near_identical_gpu_decision_srv && near_identical_gpu_decision_uav &&
        near_identical_gpu_dispatch_buf &&
        near_identical_history_owner_buf && near_identical_history_owner_srv &&
        near_identical_history_owner_uav
      ) {
        return true;
      }

      auto &cuda = cuda_driver_api::get();
      if (
        cuda_near_identical_decision_res &&
        !unregister_near_identical_decision_interop(cuda)
      ) {
        mark_terminal_failure(true);
        return false;
      }

      near_identical_tile_buf.Reset();
      near_identical_tile_srv.Reset();
      near_identical_tile_uav.Reset();
      near_identical_gpu_decision_buf.Reset();
      near_identical_gpu_decision_srv.Reset();
      near_identical_gpu_decision_uav.Reset();
      near_identical_gpu_dispatch_buf.Reset();
      near_identical_history_owner_buf.Reset();
      near_identical_history_owner_srv.Reset();
      near_identical_history_owner_uav.Reset();

      constexpr std::uint32_t tile_word_count = 4u;
      const std::uint32_t tile_group_width =
        (static_cast<std::uint32_t>(target_w) + 15u) / 16u;
      const std::uint32_t tile_group_height =
        (static_cast<std::uint32_t>(target_h) + 15u) / 16u;
      const std::uint32_t tile_group_count = tile_group_width * tile_group_height;
      const auto create_structured_uint_buffer = [&] (
                                                   const std::uint32_t word_count,
                                                   const std::uint32_t stride_words,
                                                   Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer,
                                                   Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> &srv,
                                                   Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> &uav
                                                 ) {
        if (word_count == 0u || stride_words == 0u ||
            word_count % stride_words != 0u) {
          return false;
        }
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = word_count * sizeof(std::uint32_t);
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = stride_words * sizeof(std::uint32_t);
        return SUCCEEDED(device->CreateBuffer(
                 &desc, nullptr, buffer.ReleaseAndGetAddressOf()
               )) &&
               SUCCEEDED(device->CreateShaderResourceView(
                 buffer.Get(), nullptr, srv.ReleaseAndGetAddressOf()
               )) &&
               SUCCEEDED(device->CreateUnorderedAccessView(
                 buffer.Get(), nullptr, uav.ReleaseAndGetAddressOf()
               ));
      };
      const auto create_decision_buffers = [&]() {
        std::array<std::uint32_t, near_identical_gpu_decision_word_count> initial {};
        initial[near_identical_gpu_decision_word_index(
          near_identical_gpu_decision_word_e::decision
        )] = static_cast<std::uint32_t>(near_identical_gpu_branch_e::infer);
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = static_cast<UINT>(sizeof(initial));
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        // CUDA rejects a D3D11 buffer carrying DRAWINDIRECT_ARGS. Keep the authenticated
        // proposal/receipt in this registerable raw buffer, let postprocess write every argument,
        // then copy the 256-byte transaction into a D3-only twin before indirect consumers.
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA initial_data {initial.data(), 0u, 0u};
        if (FAILED(device->CreateBuffer(
              &desc,
              &initial_data,
              near_identical_gpu_decision_buf.ReleaseAndGetAddressOf()
            ))) {
          return false;
        }
        D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc {};
        uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
        uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.FirstElement = 0u;
        uav_desc.Buffer.NumElements =
          static_cast<UINT>(near_identical_gpu_decision_word_count);
        uav_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        if (FAILED(device->CreateUnorderedAccessView(
              near_identical_gpu_decision_buf.Get(),
              &uav_desc,
              near_identical_gpu_decision_uav.ReleaseAndGetAddressOf()
            ))) {
          return false;
        }
        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc {};
        srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        srv_desc.BufferEx.FirstElement = 0u;
        srv_desc.BufferEx.NumElements =
          static_cast<UINT>(near_identical_gpu_decision_word_count);
        srv_desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        if (FAILED(device->CreateShaderResourceView(
              near_identical_gpu_decision_buf.Get(),
              &srv_desc,
              near_identical_gpu_decision_srv.ReleaseAndGetAddressOf()
            ))) {
          return false;
        }
        desc.BindFlags = 0u;
        desc.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
        return SUCCEEDED(device->CreateBuffer(
                 &desc,
                 &initial_data,
                 near_identical_gpu_dispatch_buf.ReleaseAndGetAddressOf()
               ));
      };
      const bool resources_ready =
        create_structured_uint_buffer(
          tile_group_count * tile_word_count,
          tile_word_count,
          near_identical_tile_buf,
          near_identical_tile_srv,
          near_identical_tile_uav
        ) &&
        create_structured_uint_buffer(
          static_cast<std::uint32_t>(near_identical_history_owner_word_count),
          1u,
          near_identical_history_owner_buf,
          near_identical_history_owner_srv,
          near_identical_history_owner_uav
        ) &&
        create_decision_buffers();
      if (!resources_ready) {
        near_identical_tile_buf.Reset();
        near_identical_tile_srv.Reset();
        near_identical_tile_uav.Reset();
        near_identical_gpu_decision_buf.Reset();
        near_identical_gpu_decision_srv.Reset();
        near_identical_gpu_decision_uav.Reset();
        near_identical_gpu_dispatch_buf.Reset();
        near_identical_history_owner_buf.Reset();
        near_identical_history_owner_srv.Reset();
        near_identical_history_owner_uav.Reset();
        near_identical_detector_available = false;
        log_near_identical_detector_failure_once("GPU transaction-buffer setup failed");
        return false;
      }
      const UINT zero[4] = {};
      context->ClearUnorderedAccessViewUint(
        near_identical_history_owner_uav.Get(), zero
      );
      return true;
    }

    void update_near_identical_constants(
      const std::uint32_t request_flags,
      const std::uint32_t tile_group_width,
      const std::uint32_t tile_group_height,
      const std::uint64_t current_frame_id,
      const std::uint64_t baseline_frame_id,
      const std::uint64_t domain_tag,
      const std::uint64_t request_token,
      const std::uint32_t stream_frame_delta = 0u,
      const cuda_conditional_graph::work_flag_e expected_work =
        cuda_conditional_graph::work_flag_e::none,
      const std::uint64_t observation_timestamp_us = 0u
    ) {
      const near_identical_constants_t constants {
        request_flags,
        tile_group_width,
        tile_group_height,
        tile_group_width * tile_group_height,
        static_cast<std::uint32_t>(current_frame_id),
        static_cast<std::uint32_t>(current_frame_id >> 32u),
        static_cast<std::uint32_t>(baseline_frame_id),
        static_cast<std::uint32_t>(baseline_frame_id >> 32u),
        static_cast<std::uint32_t>(domain_tag),
        static_cast<std::uint32_t>(domain_tag >> 32u),
        static_cast<std::uint32_t>(request_token),
        static_cast<std::uint32_t>(request_token >> 32u),
        reduce_groups,
        stream_frame_delta,
        cuda_conditional_graph::work_flags_value(expected_work),
        cuda_conditional_graph::work_flags_value(expected_work) == 0u ?
          0u : cuda_conditional_graph::work_flags_value(expected_work) ^
                 cuda_conditional_graph::work_flags_cookie,
        static_cast<std::uint32_t>(observation_timestamp_us),
        static_cast<std::uint32_t>(observation_timestamp_us >> 32u),
        0u,
        0u,
      };
      context->UpdateSubresource(
        near_identical_cbuffer.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
    }

    [[nodiscard]] std::uint64_t next_force_infer_transaction_token() noexcept {
      // Keep host-forced tokens in a visibly distinct nonzero namespace. The record is consumed
      // only by the GPU and is never read back; uniqueness across this estimator lifetime is
      // sufficient to reject stale records from an earlier mapped transaction.
      constexpr std::uint64_t force_namespace = 0xf000000000000000ull;
      constexpr std::uint64_t sequence_mask = 0x0fffffffffffffffull;
      force_infer_transaction_sequence =
        (force_infer_transaction_sequence + 1u) & sequence_mask;
      if (force_infer_transaction_sequence == 0u) {
        force_infer_transaction_sequence = 1u;
      }
      return force_namespace | force_infer_transaction_sequence;
    }

    bool publish_force_infer_transaction(
      const std::uint64_t token,
      const cuda_conditional_graph::work_flag_e subtitle_work
    ) {
      if (token == 0u || !near_identical_gpu_decision_buf) {
        return false;
      }
      std::array<std::uint32_t, near_identical_gpu_decision_word_count> transaction {};
      const auto proposal = cuda_conditional_graph::make_proposal(
        cuda_conditional_graph::branch_e::infer,
        token
      );
      const auto request = cuda_conditional_graph::make_request(
        token,
        subtitle_work
      );
      std::memcpy(
        reinterpret_cast<std::byte *>(transaction.data()) +
          near_identical_gpu_decision_record_byte_offset,
        &proposal,
        sizeof(proposal)
      );
      std::memcpy(
        reinterpret_cast<std::byte *>(transaction.data()) +
          near_identical_gpu_request_record_byte_offset,
        &request,
        sizeof(request)
      );
      if (subtitle_work == cuda_conditional_graph::work_flag_e::optional_ocr ||
          subtitle_work == cuda_conditional_graph::work_flag_e::optional_ocr_due) {
        transaction[near_identical_gpu_decision_word_index(
          near_identical_gpu_decision_word_e::optional_preprocess_x
        )] = (ocr_engine_width + 15u) / 16u;
        transaction[near_identical_gpu_decision_word_index(
          near_identical_gpu_decision_word_e::optional_preprocess_y
        )] = (ocr_engine_height + 15u) / 16u;
        transaction[near_identical_gpu_decision_word_index(
          near_identical_gpu_decision_word_e::optional_preprocess_z
        )] = 1u;
      }
      context->UpdateSubresource(
        near_identical_gpu_decision_buf.Get(),
        0u,
        nullptr,
        transaction.data(),
        0u,
        0u
      );
      return true;
    }

    void publish_near_identical_history_owner(
      const std::uint64_t frame_id,
      const depth_input_region_t &input_region,
      const input_color_space color_space,
      const std::uint64_t observation_timestamp_us
    ) {
      if (
        !gpu_conditional_bridge_available ||
        depth_inference_graph.policy.capture_failed ||
        !near_identical_detector_available ||
        !ensure_near_identical_detector_resources() || frame_id == 0u ||
        !input_region.tensor_content.valid({target_w, target_h}) ||
        !near_identical_history_owner_cs || !near_identical_cbuffer ||
        !near_identical_history_owner_uav || !cut_state_srv || !minmax_ema_srv || !cbuffer
      ) {
        return;
      }
      const auto domain_tag = near_identical_input_domain_tag(
        input_region,
        color_space,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      update_near_identical_constants(
        0u, 0u, 0u, frame_id, 0u, domain_tag, 0u, 0u,
        cuda_conditional_graph::work_flag_e::none,
        observation_timestamp_us
      );
      ID3D11Buffer *constant_buffers[2] = {
        cbuffer.Get(),
        near_identical_cbuffer.Get(),
      };
      ID3D11ShaderResourceView *inputs[7] = {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        cut_state_srv.Get(),
        nullptr,
        minmax_ema_srv.Get(),
      };
      ID3D11UnorderedAccessView *outputs[3] = {
        nullptr,
        nullptr,
        near_identical_history_owner_uav.Get(),
      };
      context->CSSetShader(near_identical_history_owner_cs.Get(), nullptr, 0u);
      context->CSSetConstantBuffers(0u, 2u, constant_buffers);
      context->CSSetShaderResources(0u, 7u, inputs);
      context->CSSetUnorderedAccessViews(0u, 3u, outputs, nullptr);
      dispatch_infer_postprocess(
        1u,
        1u,
        1u,
        near_identical_gpu_infer_one_byte_offset
      );
      ID3D11ShaderResourceView *null_inputs[7] = {};
      ID3D11UnorderedAccessView *null_outputs[3] = {};
      ID3D11Buffer *null_constants[2] = {};
      context->CSSetUnorderedAccessViews(0u, 3u, null_outputs, nullptr);
      context->CSSetShaderResources(0u, 7u, null_inputs);
      context->CSSetConstantBuffers(0u, 2u, null_constants);
    }

    bool dispatch_near_identical_scene_seed() {
      if (
        !near_identical_scene_seed_cs || !near_identical_cbuffer ||
        !near_identical_history_owner_srv || !scene_cut_evidence_uav || !cbuffer ||
        pending_frame_id == 0u ||
        !pending_input_region.tensor_content.valid({target_w, target_h})
      ) {
        return false;
      }
      const auto domain_tag = near_identical_input_domain_tag(
        pending_input_region,
        pending_color_space,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      update_near_identical_constants(
        0u, 0u, 0u, pending_frame_id, 0u, domain_tag, 0u, 0u,
        cuda_conditional_graph::work_flag_e::none,
        pending_observation_timestamp_us
      );
      ID3D11Buffer *constant_buffers[2] = {
        cbuffer.Get(),
        near_identical_cbuffer.Get(),
      };
      ID3D11ShaderResourceView *inputs[4] = {
        nullptr,
        nullptr,
        nullptr,
        near_identical_history_owner_srv.Get(),
      };
      ID3D11UnorderedAccessView *outputs[6] = {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        scene_cut_evidence_uav.Get(),
      };
      context->CSSetShader(near_identical_scene_seed_cs.Get(), nullptr, 0u);
      context->CSSetConstantBuffers(0u, 2u, constant_buffers);
      context->CSSetShaderResources(0u, 4u, inputs);
      context->CSSetUnorderedAccessViews(0u, 6u, outputs, nullptr);
      dispatch_infer_postprocess(
        1u, 1u, 1u, near_identical_gpu_infer_one_byte_offset
      );
      ID3D11ShaderResourceView *null_inputs[4] = {};
      ID3D11UnorderedAccessView *null_outputs[6] = {};
      ID3D11Buffer *null_constants[2] = {};
      context->CSSetUnorderedAccessViews(0u, 6u, null_outputs, nullptr);
      context->CSSetShaderResources(0u, 4u, null_inputs);
      context->CSSetConstantBuffers(0u, 2u, null_constants);
      return true;
    }

    [[nodiscard]] bool gpu_undecided_baseline_authorized(
      const gpu_adaptive_reuse_request &request
    ) const noexcept {
      return request.opaque_followup ?
               last_gpu_opaque_transaction_frame_id != 0u &&
                 request.baseline_frame_id == last_gpu_opaque_transaction_frame_id :
               last_gpu_opaque_transaction_frame_id == 0u &&
                 has_last_postprocessed_frame_id &&
                 request.baseline_frame_id == last_postprocessed_frame_id;
    }

    bool prepare_near_identical_detector(
      const gpu_adaptive_reuse_request &request,
      const std::uint64_t current_frame_id,
      const depth_input_region_t &input_region,
      const input_color_space color_space,
      const cuda_conditional_graph::work_flag_e subtitle_work
    ) {
      if (
        !request.authorize_gpu_undecided_reuse ||
        request.gpu_reuse_decision_token == 0u ||
        request.observation_timestamp_us == 0u ||
        !near_identical_detector_available ||
        !ensure_near_identical_detector_resources() ||
        current_frame_id == 0u || request.baseline_frame_id == 0u ||
        current_frame_id <= request.baseline_frame_id ||
        !gpu_undecided_baseline_authorized(request) ||
        !processed_input_domain.matches_analysis_domain(input_region, color_space) ||
        !input_region.tensor_content.valid({target_w, target_h}) ||
        !near_identical_fused_preprocess_cs || !near_identical_resolve_cs ||
        !near_identical_cbuffer || !near_identical_tile_uav ||
        !near_identical_tile_srv || !near_identical_history_owner_srv ||
        !near_identical_gpu_decision_buf || !near_identical_gpu_decision_uav ||
        !near_identical_gpu_dispatch_buf ||
        !tensor_in_srv || !tensor_previous_input_srv || !cbuffer
      ) {
        return false;
      }

      const std::uint32_t tile_group_width =
        (static_cast<std::uint32_t>(target_w) + 15u) / 16u;
      const std::uint32_t tile_group_height =
        (static_cast<std::uint32_t>(target_h) + 15u) / 16u;
      const auto domain_tag = near_identical_input_domain_tag(
        input_region,
        color_space,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      update_near_identical_constants(
        1u,
        tile_group_width,
        tile_group_height,
        current_frame_id,
        request.baseline_frame_id,
        domain_tag,
        request.gpu_reuse_decision_token,
        0u,
        subtitle_work,
        request.observation_timestamp_us
      );
      std::array<std::uint32_t, near_identical_gpu_decision_word_count>
        initialized_transaction {};
      const auto transaction_request = cuda_conditional_graph::make_request(
        request.gpu_reuse_decision_token,
        subtitle_work
      );
      std::memcpy(
        reinterpret_cast<std::byte *>(initialized_transaction.data()) +
          near_identical_gpu_request_record_byte_offset,
        &transaction_request,
        sizeof(transaction_request)
      );
      // One full-record write invalidates every prior proposal/receipt and every indirect arg
      // before this frame's fused evidence pass. Optional preprocess starts at zero and is enabled
      // only by a matching authenticated proposal; a missing/malformed proposal cannot waste OCR work or
      // authorize stale OCR output even though CUDA safely fails the mandatory depth branch open.
      context->UpdateSubresource(
        near_identical_gpu_decision_buf.Get(),
        0u,
        nullptr,
        initialized_transaction.data(),
        0u,
        0u
      );
      return true;
    }

    bool dispatch_near_identical_detector() {
      if (
        !near_identical_detector_available ||
        !near_identical_resolve_cs ||
        !near_identical_cbuffer || !near_identical_tile_srv ||
        !near_identical_history_owner_srv ||
        !near_identical_gpu_decision_uav || !cbuffer
      ) {
        return false;
      }

      ID3D11Buffer *constant_buffers[2] = {
        cbuffer.Get(),
        near_identical_cbuffer.Get(),
      };
      context->CSSetConstantBuffers(0u, 2u, constant_buffers);

      ID3D11ShaderResourceView *resolve_inputs[4] = {
        nullptr,
        nullptr,
        near_identical_tile_srv.Get(),
        near_identical_history_owner_srv.Get(),
      };
      ID3D11UnorderedAccessView *resolve_outputs[4] = {
        nullptr,
        nullptr,
        nullptr,
        near_identical_gpu_decision_uav.Get(),
      };
      context->CSSetShader(near_identical_resolve_cs.Get(), nullptr, 0u);
      context->CSSetShaderResources(0u, 4u, resolve_inputs);
      context->CSSetUnorderedAccessViews(0u, 4u, resolve_outputs, nullptr);
      context->Dispatch(1u, 1u, 1u);
      ID3D11ShaderResourceView *null_resolve_inputs[4] = {};
      ID3D11UnorderedAccessView *null_resolve_outputs[4] = {};
      ID3D11Buffer *null_constants[2] = {};
      context->CSSetUnorderedAccessViews(0u, 4u, null_resolve_outputs, nullptr);
      context->CSSetShaderResources(0u, 4u, null_resolve_inputs);
      context->CSSetConstantBuffers(0u, 2u, null_constants);

      return true;
    }

    [[nodiscard]] bool gpu_undecided_postprocess_pending() const noexcept {
      return pending_submission_class == pending_submission_class_e::gpu_undecided;
    }

    void dispatch_infer_postprocess(
      const UINT direct_x,
      const UINT direct_y,
      const UINT direct_z,
      const std::uint32_t indirect_byte_offset
    ) {
      if (!gpu_undecided_postprocess_pending()) {
        context->Dispatch(direct_x, direct_y, direct_z);
      } else if (near_identical_gpu_dispatch_buf) {
        context->DispatchIndirect(
          near_identical_gpu_dispatch_buf.Get(), indirect_byte_offset
        );
      }
    }

    void update_pending_near_identical_postprocess_constants() {
      const bool gpu_undecided = gpu_undecided_postprocess_pending();
      const auto domain_tag = near_identical_input_domain_tag(
        pending_input_region,
        pending_color_space,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      const std::uint32_t groups_x =
        (static_cast<std::uint32_t>(target_w) + 15u) / 16u;
      const std::uint32_t groups_y =
        (static_cast<std::uint32_t>(target_h) + 15u) / 16u;
      std::uint64_t stream_frame_delta = 1u;
      if (pending_gpu_transaction_baseline_frame_id != 0u &&
          pending_gpu_transaction_current_frame_id >
            pending_gpu_transaction_baseline_frame_id) {
        stream_frame_delta = pending_gpu_transaction_current_frame_id -
                             pending_gpu_transaction_baseline_frame_id;
      }
      update_near_identical_constants(
        gpu_undecided ? 1u : 2u,
        gpu_undecided ? groups_x : 0u,
        gpu_undecided ? groups_y : 0u,
        gpu_undecided ? pending_gpu_transaction_current_frame_id : pending_frame_id,
        gpu_undecided ? pending_gpu_transaction_baseline_frame_id : 0u,
        domain_tag,
        pending_wrapper_transaction_token,
        static_cast<std::uint32_t>(
          std::min<std::uint64_t>(stream_frame_delta, 65535u)
        ),
        pending_subtitle_work,
        pending_observation_timestamp_us
      );
    }

    bool dispatch_near_identical_finalizer() {
      const bool subtitle_publication_expected =
        cuda_conditional_graph::work_flags_value(pending_subtitle_work) != 0u;
      if (
        !near_identical_finalize_cs ||
        !near_identical_gpu_decision_uav ||
        !near_identical_gpu_decision_buf || !near_identical_gpu_dispatch_buf ||
        !near_identical_cbuffer || !cbuffer ||
        pending_wrapper_transaction_token == 0u ||
        (subtitle_publication_expected &&
         (!ocr_box_record_uav || !ocr_resolve_cbuffer ||
          !update_pending_ocr_constants()))
      ) {
        return false;
      }
      update_pending_near_identical_postprocess_constants();
      ID3D11Buffer *constant_buffers[3] = {
        cbuffer.Get(),
        near_identical_cbuffer.Get(),
        subtitle_publication_expected ? ocr_resolve_cbuffer.Get() : nullptr,
      };
      ID3D11UnorderedAccessView *outputs[4] = {
        nullptr,
        subtitle_publication_expected ? ocr_box_record_uav.Get() : nullptr,
        nullptr,
        near_identical_gpu_decision_uav.Get(),
      };
      context->CSSetConstantBuffers(0u, 3u, constant_buffers);
      context->CSSetUnorderedAccessViews(0u, 4u, outputs, nullptr);
      context->CSSetShader(near_identical_finalize_cs.Get(), nullptr, 0u);
      context->Dispatch(1u, 1u, 1u);
      ID3D11UnorderedAccessView *null_outputs[4] = {};
      ID3D11Buffer *null_constants[3] = {};
      context->CSSetUnorderedAccessViews(0u, 4u, null_outputs, nullptr);
      context->CSSetConstantBuffers(0u, 3u, null_constants);
      // CUDA cannot register a DRAWINDIRECT_ARGS buffer. Once the receipt validator has written
      // every args record and any deterministic abstention into their registerable resources,
      // copy the complete 256-byte transaction into the D3-only indirect twin consumed below.
      context->CopyResource(
        near_identical_gpu_dispatch_buf.Get(),
        near_identical_gpu_decision_buf.Get()
      );
      return true;
    }

    void dispatch_near_identical_reuse_depth() {
      if (
        !gpu_undecided_postprocess_pending() || !near_identical_reuse_depth_cs ||
        !near_identical_gpu_dispatch_buf || !depth_previous_srv || !depth_uav ||
        !cbuffer
      ) {
        return;
      }
      ID3D11ShaderResourceView *inputs[6] = {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        depth_previous_srv.Get(),
      };
      ID3D11UnorderedAccessView *outputs[5] = {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        depth_uav.Get(),
      };
      context->CSSetShader(near_identical_reuse_depth_cs.Get(), nullptr, 0u);
      context->CSSetConstantBuffers(0u, 1u, cbuffer.GetAddressOf());
      context->CSSetShaderResources(0u, 6u, inputs);
      context->CSSetUnorderedAccessViews(0u, 5u, outputs, nullptr);
      context->DispatchIndirect(
        near_identical_gpu_dispatch_buf.Get(),
        near_identical_gpu_reuse_grid16_byte_offset
      );
      ID3D11ShaderResourceView *null_inputs[6] = {};
      ID3D11UnorderedAccessView *null_outputs[5] = {};
      ID3D11Buffer *null_constant = nullptr;
      context->CSSetUnorderedAccessViews(0u, 5u, null_outputs, nullptr);
      context->CSSetShaderResources(0u, 6u, null_inputs);
      context->CSSetConstantBuffers(0u, 1u, &null_constant);
    }


    bool initialize_ocr_context(
      const std::filesystem::path &assets_dir,
      cuda_driver_api &cuda
    ) {
      engine_artifact artifact;
      if (!ensure_ocr_tensorrt_engine_for_device(
            assets_dir,
            cuda,
            cuda_device,
            artifact
          )) {
        return false;
      }
      auto engine_path = artifact.engine_path;
      ocr_engine_key = std::to_string(cuda_device) + ":" + artifact.name;
      bool engine_repair_allowed = true;
      {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        ocr_engine = acquire_engine_locked(
          ocr_engine_key,
          engine_path,
          ocr_exec_context,
          &ocr_context_pooled
        );
        auto &slot = g_engines[ocr_engine_key];
        if (!validate_ocr_engine_io_locked(ocr_engine, slot)) {
          return_execution_context_locked(ocr_engine_key, ocr_exec_context);
          ocr_context_pooled = false;
          ocr_context_warmed = false;
          engine_repair_allowed =
            detach_incompatible_engine_locked(ocr_engine_key);
          ocr_engine = nullptr;
        }
        ocr_context_warmed = ocr_context_pooled;
      }

      if (!ocr_engine) {
        // Startup normally repairs this cache first, but standalone evaluators can construct the
        // estimator directly. Keep the same one-shot, no-live-owner repair at this second seam.
        if (!engine_repair_allowed) {
          BOOST_LOG(error)
            << "PP-OCRv6 tiny cannot replace an incompatible TensorRT plan while another "
               "session owns a context.";
          return false;
        }
        {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          erase_empty_engine_slot_locked(ocr_engine_key);
        }
        std::error_code remove_error;
        std::filesystem::remove(engine_path, remove_error);
        BOOST_LOG(warning)
          << "Depth estimator found an unreadable or incompatible PP-OCRv6 tiny plan; "
             "rebuilding "
          << engine_path.filename() << '.';
        if (!ensure_ocr_tensorrt_engine_for_device(
              assets_dir, cuda, cuda_device,
              artifact
            )) {
          return false;
        }
        engine_path = artifact.engine_path;
        ocr_engine_key = std::to_string(cuda_device) + ":" + artifact.name;
        {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          ocr_engine = acquire_engine_locked(
            ocr_engine_key,
            engine_path,
            ocr_exec_context,
            &ocr_context_pooled
          );
          auto &slot = g_engines[ocr_engine_key];
          if (!validate_ocr_engine_io_locked(ocr_engine, slot)) {
            return_execution_context_locked(ocr_engine_key, ocr_exec_context);
            ocr_context_pooled = false;
            ocr_context_warmed = false;
            detach_incompatible_engine_locked(ocr_engine_key);
            ocr_engine = nullptr;
          } else {
            ocr_context_warmed = ocr_context_pooled;
          }
        }
        if (!ocr_engine) {
          BOOST_LOG(error)
            << "Rebuilt PP-OCRv6 tiny plan is unreadable or violates the FP32 I/O contract.";
          return false;
        }
      }

      bool create_context = false;
      if (!ocr_exec_context) {
        std::unique_lock<std::mutex> lock(g_trt_mutex);
        auto &slot = g_engines[ocr_engine_key];
        if (slot.context_pool.empty() &&
            allocated_context_count(slot) >= kMaxContextsPerEngine) {
          g_trt_context_available.wait_for(
            lock,
            std::chrono::seconds(2),
            [&slot]() {
              return !slot.context_pool.empty() ||
                     allocated_context_count(slot) < kMaxContextsPerEngine;
            }
          );
        }
        if (!slot.context_pool.empty()) {
          ocr_exec_context = slot.context_pool.back();
          slot.context_pool.pop_back();
          ocr_context_pooled = true;
          ocr_context_warmed = true;
        } else if (allocated_context_count(slot) < kMaxContextsPerEngine) {
          create_context = slot.context_accounting.reserve(kMaxContextsPerEngine);
        }
      }
      if (create_context) {
        ocr_exec_context = ocr_engine->createExecutionContext();
        if (!ocr_exec_context) {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          release_context_reservation_locked(ocr_engine_key);
          return false;
        }
        if (!warmup_ocr_execution_context(cuda, cuda_ctx, ocr_exec_context)) {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          quarantine_execution_context_locked(
            ocr_engine_key, ocr_exec_context,
            false
          );
          return false;
        }
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        mark_execution_context_warmed_locked(ocr_engine_key);
        ocr_context_warmed = true;
      }
      if (!ocr_exec_context || !ocr_context_warmed) {
        BOOST_LOG(warning)
          << "No isolated warmed PP-OCRv6 tiny context is available; conditioning stays flat.";
        return false;
      }
      // A pooled context was warmed before it entered the global cache. Revalidate its fixed
      // context-dependent output bound in this owning instance before any D3D interop binding.
      const auto fixed_dims = make_input_dims(ocr_engine_height, ocr_engine_width);
      constexpr std::int64_t fixed_output_bytes =
        static_cast<std::int64_t>(ocr_engine_width) * ocr_engine_height * sizeof(float);
      if (!ocr_exec_context->setInputShape("x", fixed_dims)) {
        BOOST_LOG(error)
          << "PP-OCRv6 tiny could not restore its authenticated fixed input shape.";
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::pre_enqueue_interop_or_binding
        );
        return false;
      }
      ocr_shape_bound = true;
      const std::int64_t required_output_bytes =
        ocr_exec_context->getMaxOutputSize("fetch_name_0");
      if (required_output_bytes <= 0 || required_output_bytes > fixed_output_bytes) {
        BOOST_LOG(error)
          << "PP-OCRv6 tiny resolved output requires " << required_output_bytes
          << " bytes; the authenticated fixed buffer provides " << fixed_output_bytes << '.';
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::pre_enqueue_interop_or_binding
        );
        return false;
      }
      ocr_output_size_validated = true;
      if (!publish_active_engine_manifest(assets_dir, ocr_model_name, artifact)) {
        BOOST_LOG(warning) << "Could not publish the active PP-OCRv6 tiny engine manifest.";
      }
      BOOST_LOG(info) << "PP-OCRv6 tiny live context acquired (fixed 960x160).";
      return true;
    }

    impl(
      Microsoft::WRL::ComPtr<ID3D11Device> d,
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> c,
      const std::filesystem::path &assets_dir,
      const config::video_t::sbs_t &cfg,
      const config::depth_model_info &model
    ):
        device(d),
        context(c),
        shader_root(assets_dir / "shaders" / "directx"),
        ema_alpha((float) config::host_sbs_v2_live_calibration::depth_ema),
        ema_edge_change((float) config::host_sbs_v2_live_calibration::edge_change),
        ema_edge_gradient((float) config::host_sbs_v2_live_calibration::edge_gradient),
        ema_edge_strength((float) config::host_sbs_v2_live_calibration::edge_strength),
        depth_short_side(config::host_sbs_v2_live_calibration::depth_short_side),
        max_aspect((float) config::host_sbs_v2_live_calibration::depth_max_aspect),
        minmax_alpha((float) config::host_sbs_v2_live_calibration::minmax_ema),
        diagnostics_enabled(config::sunshine.diagnostics_enabled),
        parallax_v2_requested_pop_strength(
          depth_coordinate_v2::requested_pop_strength(
            static_cast<float>(cfg.pop_strength)
          )
        ),
        parallax_v2_requested_gain(depth_coordinate_v2::requested_gain_for_config(static_cast<float>(cfg.pop_strength))) {
      const auto init_started = std::chrono::steady_clock::now();
      // Enable the process-wide rolling collector for diagnostic runs. Do not reset it here:
      // Galaxy XR and local-AR estimators may coexist, and one session must not invalidate the
      // other session's pending D3D query generation. The offline harness resets explicitly.
      perf_depth_conditional.stage = "depth_conditional_transaction";
      sbs_perf::set_enabled(diagnostics_enabled);
      initialize_d3d_perf();

      auto &cuda = cuda_driver_api::get();
      if (cuda.is_valid() && ensure_cuda_initialized(cuda)) {
        if (cuda_device_for_d3d(cuda, device.Get(), cuda_device)) {
          cuda_ctx = primary_context(cuda, cuda_device);
          if (cuda_ctx) {
            const auto context_result = cuda.cuCtxSetCurrent(cuda_ctx);
            if (context_result != CUDA_SUCCESS) {
              BOOST_LOG(error)
                << "Depth estimator failed: could not select the D3D adapter's CUDA context ("
                << context_result << ").";
              cuda_ctx = nullptr;
            } else {
              const auto stream_result =
                cuda.cuStreamCreate(&cu_stream, CU_STREAM_NON_BLOCKING);
              if (stream_result != CUDA_SUCCESS || !cu_stream) {
                BOOST_LOG(error)
                  << "Depth estimator failed: CUDA stream creation failed ("
                  << stream_result << ").";
                // CUDA documents a null stream on failure, but defensively destroy any handle it
                // returned before rejecting construction so no wrong/partial stream can survive.
                if (cu_stream && cuda.cuStreamDestroy) {
                  cuda.cuStreamDestroy(cu_stream);
                }
                cu_stream = nullptr;
              }
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
      bool engine_repair_allowed = true;

      {  // Scope this lock to the g_engines/g_runtime access only: it MUST be released before
         // warmup_inference() at the end of the ctor (which re-locks g_trt_mutex) -- a
         // non-recursive std::mutex would otherwise self-deadlock and hang construction.
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        // Load (once) the engine for this configured model into its own slot and take a pooled
        // execution context if one is free. Different startup configurations remain isolated.
        engine = acquire_engine_locked(
          engine_key,
          model_path,
          exec_context,
          &depth_context_pooled
        );
        context_warmed = depth_context_pooled;
        if (depth_context_pooled) {
          BOOST_LOG(info) << "Reusing pooled TensorRT execution context.";
        }
        auto &slot = g_engines[engine_key];

        if (!validate_engine_io_locked(engine, slot)) {
          BOOST_LOG(error) << "Depth engine I/O contract is incompatible with Sunshine 3D; streaming flat SBS.";
          return_execution_context_locked(engine_key, exec_context);
          depth_context_pooled = false;
          context_warmed = false;
          engine_repair_allowed = detach_incompatible_engine_locked(engine_key);
          engine = nullptr;
        }

      }  // release g_trt_mutex before the shader/buffer setup and warmup below

      if (!engine) {
        // The startup preparation normally repairs this already. The constructor also serves the
        // standalone evaluator, so retain the same one-shot self-heal when it is the first owner.
        if (!engine_repair_allowed) {
          BOOST_LOG(error) << "Depth estimator cannot replace an incompatible TensorRT plan "
                              "while another session still owns its execution context.";
          return;
        }
        {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          erase_empty_engine_slot_locked(engine_key);
        }
        std::error_code ec;
        std::filesystem::remove(model_path, ec);
        BOOST_LOG(warning) << "Depth estimator found an unreadable or incompatible TensorRT plan; rebuilding "
                           << model_path.filename() << '.';
        if (!ensure_tensorrt_engine_for_device(assets_dir, model, cuda, cuda_device, artifact)) {
          return;
        }
        model_path = artifact.engine_path;
        engine_key = std::to_string(cuda_device) + ":" + artifact.name;
        {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          engine = acquire_engine_locked(
            engine_key,
            model_path,
            exec_context,
            &depth_context_pooled
          );
          context_warmed = depth_context_pooled;
          auto &slot = g_engines[engine_key];
          if (!validate_engine_io_locked(engine, slot)) {
            return_execution_context_locked(engine_key, exec_context);
            depth_context_pooled = false;
            context_warmed = false;
            detach_incompatible_engine_locked(engine_key);
            engine = nullptr;
          }
        }
        if (!engine) {
          BOOST_LOG(error) << "Depth estimator failed: rebuilt TensorRT plan is unreadable or incompatible.";
          return;
        }
      }

      const auto *model_coordinate_calibration =
        depth_coordinate_v2::find_model_calibration(
          model.name,
          model.url,
          artifact.source_sha256
        );
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
            // Reserve atomically so concurrent constructors cannot exceed the cap.
            create_context = slot.context_accounting.reserve(kMaxContextsPerEngine);
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
            release_context_reservation_locked(engine_key);
          }
        }
      }

      // Preprocessing plus the private normalized depth consumed by cut evidence. The only
      // analysis roots are the compact scene-cut detector passes.
      const auto preprocess_sources = host_sbs_shader_cache::snapshot_sources(
        shader_root,
        host_sbs_shader_cache::preprocess_specs
      );
      if (!preprocess_sources) {
        BOOST_LOG(error)
          << "Depth estimator failed: the calibrated RGB-preprocess source closure is unavailable.";
        return;
      }
      const std::string preprocess_source_closure_sha256 =
        host_sbs_shader_cache::source_closure_sha256(preprocess_sources);
      const auto *coordinate_calibration =
        model_coordinate_calibration &&
        model_coordinate_calibration->preprocess.source_closure_schema ==
          host_sbs_shader_cache::source_closure_schema &&
        model_coordinate_calibration->preprocess.source_file ==
          host_sbs_shader_cache::host_sbs_near_identical_fused_preprocess.filename &&
        model_coordinate_calibration->preprocess.source_entrypoint ==
          host_sbs_shader_cache::host_sbs_near_identical_fused_preprocess.entrypoint &&
        model_coordinate_calibration->preprocess.source_target ==
          host_sbs_shader_cache::host_sbs_near_identical_fused_preprocess.target &&
        model_coordinate_calibration->preprocess.source_compile_flags ==
          host_sbs_shader_cache::shader_compile_flags &&
        model_coordinate_calibration->preprocess.source_macro_count == 0u &&
        model_coordinate_calibration->preprocess.source_closure_sha256 ==
          preprocess_source_closure_sha256 ?
          model_coordinate_calibration : nullptr;
      if (!coordinate_calibration) {
        parallax_v2_producer_failed = true;
        BOOST_LOG(error)
          << "Host SBS V2 cannot authenticate uncalibrated depth model '"
          << model.name << "' (ONNX SHA-256 " << artifact.source_sha256
          << ", preprocess source SHA-256 " << preprocess_source_closure_sha256
          << "); Host SBS will fail flat.";
        return;
      }
      parallax_v2_raw_coordinate_scale = coordinate_calibration->raw_coordinate_scale;
      raw_model_provenance = std::make_shared<const raw_model_provenance_t>(
        raw_model_provenance_t {
          .depth_model = model.name,
          .depth_model_url = model.url,
          .onnx_sha256 = artifact.source_sha256,
          .preprocess_profile = std::string {
            coordinate_calibration->preprocess.profile
          },
          .preprocess_source_closure_sha256 = preprocess_source_closure_sha256,
        }
      );

      const auto producer_sources = host_sbs_shader_cache::snapshot_sources(
        shader_root,
        host_sbs_shader_cache::parallax_v2_producer_specs
      );
      static_assert(
        depth_coordinate_v2::shader_source_closure_schema ==
        host_sbs_shader_cache::source_closure_schema
      );
      static_assert(
        depth_coordinate_v2::shader_source_compile_flags ==
        host_sbs_shader_cache::shader_compile_flags
      );
      static_assert(depth_coordinate_v2::shader_source_macro_count == 0u);
      static_assert(
        depth_coordinate_v2::shader_source_specs.size() ==
        host_sbs_shader_cache::parallax_v2_producer_specs.size()
      );
      bool shader_specs_match = true;
      for (std::size_t index = 0;
           index < depth_coordinate_v2::shader_source_specs.size();
           ++index) {
        const auto &contract_spec = depth_coordinate_v2::shader_source_specs[index];
        const auto &runtime_spec =
          host_sbs_shader_cache::parallax_v2_producer_specs[index];
        shader_specs_match =
          shader_specs_match &&
          contract_spec.source_file == runtime_spec.filename &&
          contract_spec.source_entrypoint == runtime_spec.entrypoint &&
          contract_spec.source_target == runtime_spec.target;
      }
      const std::string shader_source_closure_sha256 =
        producer_sources ?
          host_sbs_shader_cache::source_closure_sha256(producer_sources) :
          std::string {};
      const bool shader_identity_matches =
        shader_specs_match &&
        shader_source_closure_sha256 ==
          depth_coordinate_v2::shader_source_closure_sha256;

      // Create every compute object from bytecode keyed by the one authenticated producer
      // snapshot. Startup prewarm populates these exact cache entries, so the constructor neither
      // recompiles the fused preprocess from its identity-only calibration closure nor compiles the shared
      // analysis roots once as "core" and then again as V2.
      using producer_shader_e = host_sbs_shader_cache::producer_shader_e;
      const auto shader_output = [&](const producer_shader_e shader) ->
        Microsoft::WRL::ComPtr<ID3D11ComputeShader> * {
        switch (shader) {
          case producer_shader_e::host_sbs_near_identical_fused_preprocess:
            return std::addressof(near_identical_fused_preprocess_cs);
          case producer_shader_e::buffer_to_tex: return std::addressof(buffer_to_tex_cs);
          case producer_shader_e::buffer_to_tex_pad:
            return std::addressof(buffer_to_tex_pad_cs);
          case producer_shader_e::depth_minmax_ema: return std::addressof(depth_minmax_ema_cs);
          case producer_shader_e::depth_hist: return std::addressof(depth_hist_cs);
          case producer_shader_e::depth_scene_cut_evidence:
            return std::addressof(depth_scene_cut_evidence_cs);
          case producer_shader_e::depth_scene_cut_resolve:
            return std::addressof(depth_scene_cut_resolve_cs);
          case producer_shader_e::depth_valid_history:
            return std::addressof(depth_valid_history_cs);
          case producer_shader_e::depth_coordinate_v2_moments:
            return std::addressof(depth_coordinate_v2_moments_cs);
          case producer_shader_e::depth_coordinate_v2_frame_resolve:
            return std::addressof(depth_coordinate_v2_frame_resolve_cs);
          case producer_shader_e::depth_coordinate_v2_state_resolve:
            return std::addressof(depth_coordinate_v2_state_resolve_cs);
          case producer_shader_e::depth_coordinate_v2_map:
            return std::addressof(depth_coordinate_v2_map_cs);
          case producer_shader_e::depth_coordinate_v2_ownership:
            return std::addressof(depth_coordinate_v2_ownership_cs);
          case producer_shader_e::depth_coordinate_v2_vertical_limit:
            return std::addressof(depth_coordinate_v2_vertical_limit_cs);
          case producer_shader_e::depth_coordinate_v2_limit:
            return std::addressof(depth_coordinate_v2_limit_cs);
          case producer_shader_e::host_sbs_ocr_preprocess:
            return std::addressof(ocr_preprocess_cs);
          case producer_shader_e::host_sbs_ocr_cells:
            return std::addressof(ocr_box_cells_cs);
          case producer_shader_e::host_sbs_ocr_resolve:
            return std::addressof(ocr_box_resolve_cs);
          case producer_shader_e::host_sbs_subtitle_locator_resolve:
            return std::addressof(subtitle_locator_resolve_cs);
          case producer_shader_e::host_sbs_subtitle_condition_prepare:
            return std::addressof(subtitle_condition_prepare_cs);
          case producer_shader_e::host_sbs_subtitle_condition:
            return std::addressof(subtitle_condition_cs);
        }
        return nullptr;
      };
      bool producer_shaders_ok = producer_sources && shader_identity_matches;
      for (const auto &binding : host_sbs_shader_cache::parallax_v2_producer_bindings) {
        if (!producer_shaders_ok) {
          break;
        }
        auto *const output = shader_output(binding.id);
        producer_shaders_ok = output && create_shader(
          producer_sources, binding.spec, *output);
      }
      if (!producer_shaders_ok) {
        parallax_v2_producer_failed = true;
        BOOST_LOG(error)
          << "Host SBS V2 shader initialization or source-identity verification failed; "
             "Host SBS will fail flat.";
        return;
      }

      // The sole preprocess shader runs for every root. Two immutable mode buffers make the
      // comparison decision explicit at b1 without rebuilding per-frame state: force/bootstrap
      // exits after canonical NCHW production, while GPU-undecided continues into tile evidence.
      const auto create_fused_mode_cbuffer = [&] (
                                                const std::uint32_t enabled,
                                                Microsoft::WRL::ComPtr<ID3D11Buffer> &output
                                              ) {
        const std::array<std::uint32_t, 4u> constants {enabled, 0u, 0u, 0u};
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.ByteWidth = static_cast<UINT>(sizeof(constants));
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA data {constants.data(), 0u, 0u};
        return SUCCEEDED(device->CreateBuffer(
          &desc, &data, output.ReleaseAndGetAddressOf()
        ));
      };
      if (!create_fused_mode_cbuffer(0u, fused_preprocess_force_cbuffer) ||
          !create_fused_mode_cbuffer(1u, fused_preprocess_compare_cbuffer)) {
        parallax_v2_producer_failed = true;
        BOOST_LOG(error)
          << "Host SBS fused preprocess mode-buffer initialization failed; "
             "Host SBS will fail flat.";
        return;
      }

      parallax_v2_producer_shaders_ready = true;
      parallax_v2_shader_provenance =
        std::make_shared<const parallax_v2_shader_provenance_t>(
          parallax_v2_shader_provenance_t {
            .source_closure_schema =
              depth_coordinate_v2::shader_source_closure_schema,
            .source_compile_flags =
              depth_coordinate_v2::shader_source_compile_flags,
            .source_macro_count =
              depth_coordinate_v2::shader_source_macro_count,
            .source_closure_sha256 = shader_source_closure_sha256,
          }
        );
      BOOST_LOG(info)
        << "Host SBS parallax-v2 GPU producer loaded for model '" << model.name
        << "' (fixed raw coordinate scale " << parallax_v2_raw_coordinate_scale
        << ", requested pop " << parallax_v2_requested_pop_strength
        << ", requested gain " << parallax_v2_requested_gain
        << "); completed fields remain separately authenticated before rendering.";
      BOOST_LOG(info) << "Host SBS V2 cut-only GPU analysis enabled.";
      BOOST_LOG(info)
        << "Host SBS RGB preprocess + near-identical tile fusion is active for every frame.";

      if (!cuda.has_conditional_graph_support()) {
        fail_gpu_conditional_bridge_once(
          "the installed CUDA driver lacks conditional-graph support"
        );
        return;
      }

      initialize_near_identical_detector();
      if (!near_identical_detector_available) {
        fail_gpu_conditional_bridge_once(
          "mandatory GPU detector/postprocess shader initialization failed"
        );
        return;
      }
      if (diagnostics_enabled) {
        initialize_gpu_trace();
      }

      // OCR is optional. Its authenticated source, engine and mutable execution context are
      // isolated from depth; an unavailable detector leaves the compact OCR8 record invalid and
      // the SLR13 conditioner copies BaseField exactly.
      ocr_available = initialize_ocr_context(assets_dir, cuda);
      if (!ocr_available) {
        BOOST_LOG(warning)
          << "PP-OCRv6 tiny is unavailable for this estimator; subtitle conditioning is flat.";
      }
      initialize_inference_done_events(cuda);
      // Raw normalization record, pre-seeded to
      // {min = 0xFFFFFFFF, max = 0, valid = 0, eligible = 0}.
      // The fused V2 frame resolve overwrites all four words before the histogram consumes them;
      // depth_minmax_ema_cs resets the identity after each frame as an additional safe default.
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

      // The state buffer keeps the established 32-word ABI for offline traces, Dump 3D, and live
      // telemetry. V2 allocates nine cut counters plus the source-stream frame delta beside it.
      {
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(uint32_t);
        uint32_t initial_cut_evidence[10] = {};
        bd.ByteWidth = sizeof(initial_cut_evidence);
        D3D11_SUBRESOURCE_DATA cut_sd = {initial_cut_evidence, 0, 0};
        device->CreateBuffer(&bd, &cut_sd, &scene_cut_evidence_buf);
        if (scene_cut_evidence_buf) {
          device->CreateUnorderedAccessView(
            scene_cut_evidence_buf.Get(),
            nullptr,
            &scene_cut_evidence_uav
          );
        }

        const auto &init_state = sbs_adaptive_state::initial_words;
        bd.ByteWidth = sizeof(init_state);
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        bd.StructureByteStride = sizeof(float) * 4;
        D3D11_SUBRESOURCE_DATA sd2 = {init_state.data(), 0, 0};
        device->CreateBuffer(&bd, &sd2, &cut_state_buf);
        if (cut_state_buf) {
          device->CreateUnorderedAccessView(cut_state_buf.Get(), nullptr, &cut_state_uav);
          device->CreateShaderResourceView(cut_state_buf.Get(), nullptr, &cut_state_srv);
        }
      }

      const bool analysis_ready =
        depth_scene_cut_evidence_cs && depth_scene_cut_resolve_cs && scene_cut_evidence_uav;
      valid = engine && exec_context && cu_stream &&
              near_identical_fused_preprocess_cs &&
              fused_preprocess_force_cbuffer && fused_preprocess_compare_cbuffer &&
              buffer_to_tex_cs &&
              buffer_to_tex_pad_cs &&
              depth_minmax_ema_cs && depth_hist_cs && depth_valid_history_cs &&
              minmax_raw_uav && minmax_ema_uav && minmax_ema_srv && hist_uav &&
              analysis_ready && cut_state_uav && cut_state_srv;
      if (!valid) {
        BOOST_LOG(error) << "Depth estimator failed: required engine or Host SBS GPU resource initialization failed.";
        return;
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
      if (!publish_active_engine_manifest(assets_dir, model.name, artifact)) {
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
        quarantine_execution_context_locked(engine_key, exec_context, false);
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
      bool cleanup_execution_ok = cuda.is_valid() && cuda_ctx;
      const auto retain_all_cuda_backing = [&]() noexcept {
        // When CUDA teardown cannot positively release every handle, preserve every possible D3
        // operand rather than let ComPtr destruction create a driver-side use-after-free.
        (void) near_identical_gpu_decision_uav.Detach();
        (void) near_identical_gpu_decision_buf.Detach();
        (void) tensor_in_srv.Detach();
        (void) tensor_in_uav.Detach();
        (void) tensor_in_buf.Detach();
        (void) tensor_out_srv.Detach();
        (void) tensor_out_buf.Detach();
        (void) ocr_input_uav.Detach();
        (void) ocr_input_buf.Detach();
        (void) ocr_output_srv.Detach();
        (void) ocr_output_buf.Detach();
      };
      if (cuda.is_valid() && cuda_ctx) {
        const auto set_current = cuda.cuCtxSetCurrent(cuda_ctx);
        if (set_current != CUDA_SUCCESS) {
          cleanup_execution_ok = false;
          BOOST_LOG(warning)
            << "cuCtxSetCurrent failed during depth-estimator teardown: " << set_current;
          retain_all_cuda_backing();
          depth_conditional_graph.abandon_unsafe();
        } else {
          bool stream_quiesced = !cu_stream;
          if (cu_stream) {
            if (cuda.cuStreamQuery) {
              // Teardown runs on the lifecycle worker, not the presenter. Allow a weak GPU up to a
              // quarter second to finish one already-submitted root before choosing the deliberate
              // retain/quarantine path; 1 ms pacing bounds driver calls and avoids a busy spin.
              constexpr auto teardown_quiescence_budget =
                std::chrono::milliseconds {250};
              const auto deadline =
                std::chrono::steady_clock::now() + teardown_quiescence_budget;
              CUresult last_query = CUDA_ERROR_NOT_READY;
              unsigned int query_count = 0u;
              while (true) {
                last_query = cuda.cuStreamQuery(cu_stream);
                ++query_count;
                const auto readiness = normalized_cuda_readiness(last_query);
                const auto action = detail::teardown_quiescence_action(
                  readiness,
                  std::chrono::steady_clock::now() < deadline
                );
                if (action == detail::teardown_quiescence_action_e::release_operands) {
                  stream_quiesced = true;
                  break;
                }
                if (action == detail::teardown_quiescence_action_e::retain_operands) {
                  cleanup_execution_ok = false;
                  BOOST_LOG(warning)
                    << "Depth-estimator teardown could not prove stream quiescence after "
                    << query_count << " nonblocking query attempt(s): result=" << last_query
                    << (readiness == detail::async_stream_readiness_e::busy ?
                          " (250 ms deadline expired)." : " (CUDA query failed).");
                  break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds {1});
              }
            } else {
              cleanup_execution_ok = false;
              BOOST_LOG(warning)
                << "Depth-estimator teardown cannot query stream quiescence; retaining all CUDA "
                   "operands.";
            }
          }
          if (!stream_quiesced) {
            // A failed query may be the launch fault being reported; a timeout still leaves work
            // potentially live. Neither proves that graph operands are idle. Retain the wrapper,
            // source graphs, stream, registrations, and their D3 backing.
            retain_all_cuda_backing();
            depth_conditional_graph.abandon_unsafe();
          } else {
            // Every CUDA teardown call may be the first API to report a deferred root fault. The
            // chain is monotonic: after the first failure, retain every not-yet-proven handle and
            // D3 operand and issue no later CUDA destroy/unregister call.
            bool cuda_teardown_may_continue = true;
            std::string_view failed_teardown_operation;
            const auto observe_teardown = [&](const bool succeeded, const std::string_view name) {
              const bool next = detail::cuda_teardown_chain_may_continue(
                cuda_teardown_may_continue,
                succeeded
              );
              if (cuda_teardown_may_continue && !next) {
                failed_teardown_operation = name;
              }
              cuda_teardown_may_continue = next;
            };

            // A previously failed context restore is already a failed first teardown operation.
            // Do not use that context for even an otherwise independent event destroy.
            observe_teardown(
              !gpu_conditional_bridge_context_failed,
              "conditional wrapper context state"
            );
            if (cuda_teardown_may_continue) {
              observe_teardown(
                perf_destroy_events(cuda),
                "diagnostic event destruction"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                destroy_inference_done_events(cuda),
                inference_event_ever_recorded ?
                  "recorded inference event destruction" :
                  "unrecorded inference event destruction"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                unregister_near_identical_decision_interop(cuda),
                "conditional wrapper/decision interop release"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                destroy_inference_graph(cuda, depth_inference_graph),
                "DAV2 source graph destruction"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                destroy_inference_graph(cuda, ocr_inference_graph),
                "OCR source graph destruction"
              );
            }
            if (cuda_teardown_may_continue && cu_stream) {
              const CUresult stream_destroyed = cuda.cuStreamDestroy ?
                                                  cuda.cuStreamDestroy(cu_stream) :
                                                  static_cast<CUresult>(-1);
              observe_teardown(
                stream_destroyed == CUDA_SUCCESS,
                "joined stream destruction"
              );
              if (cuda_teardown_may_continue) {
                cu_stream = nullptr;
              }
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                unregister_cuda_graphics_resource(cuda, cuda_in_res),
                "DAV2 input interop unregister"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                unregister_cuda_graphics_resource(cuda, cuda_out_res),
                "DAV2 output interop unregister"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                unregister_cuda_graphics_resource(cuda, cuda_ocr_in_res),
                "OCR input interop unregister"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                unregister_cuda_graphics_resource(cuda, cuda_ocr_out_res),
                "OCR output interop unregister"
              );
            }

            if (!cuda_teardown_may_continue) {
              cleanup_execution_ok = false;
              retain_all_cuda_backing();
              depth_conditional_graph.abandon_unsafe();
              BOOST_LOG(warning)
                << "Depth-estimator CUDA teardown stopped after "
                << failed_teardown_operation
                << " failed; retaining all remaining CUDA/D3 operands.";
            }
          }
        }
      } else {
        retain_all_cuda_backing();
        depth_conditional_graph.abandon_unsafe();
      }
      if (!cleanup_execution_ok) {
        // Any teardown API can be the first call to surface a deferred launch fault.
        // CUDA can surface one stream's earlier launch fault through another stream's cleanup.
        // Never give either participating context to a later estimator in that case.
        execution_context_poisoned = true;
        if (joined_stream_ocr_ever_submitted_or_armed) {
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::unsafe_teardown
          );
        }
      }

      // Return only a successfully warmed context to the reusable pool. Construction can fail
      // after createExecutionContext() but before warmup (shader/resource allocation is one such
      // path); admitting that context would make the next instance skip the failed lazy load.
      // Contexts cannot be destroyed safely across the DLL boundary, so quarantine them instead.
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      if (exec_context) {
        if (execution_context_poisoned) {
          BOOST_LOG(warning)
            << "Quarantining a TensorRT execution context after CUDA/TensorRT execution or "
               "teardown failure.";
        }
        recycle_or_quarantine_execution_context_locked(
          engine_key,
          exec_context,
          context_warmed,
          execution_context_poisoned
        );
      }
      if (ocr_exec_context) {
        if (ocr_context_health.poisoned()) {
          BOOST_LOG(warning)
            << "Quarantining the PP-OCRv6 tiny execution context after asynchronous GPU "
               "execution or unsafe teardown.";
        }
        recycle_or_quarantine_execution_context_locked(
          ocr_engine_key,
          ocr_exec_context,
          ocr_context_warmed,
          ocr_context_health.poisoned()
        );
      }
      // TRT runtime/engines are cached globally, do not destroy them here.
    }

    void mark_d3d_parallax_start(d3d_perf_slot *slot) {
      if (slot) {
        slot->has_parallax = true;
        context->End(slot->parallax_start.Get());
      }
    }

    void mark_d3d_parallax_frame_stats_start(d3d_perf_slot *slot) {
      if (slot) {
        slot->has_parallax_frame_stats = true;
        context->End(slot->parallax_frame_stats_start.Get());
      }
    }

    void mark_d3d_parallax_frame_stats_end(d3d_perf_slot *slot) {
      if (slot && slot->has_parallax_frame_stats) {
        context->End(slot->parallax_frame_stats_end.Get());
      }
    }

    void mark_d3d_parallax_end(d3d_perf_slot *slot) {
      if (slot && slot->has_parallax) {
        context->End(slot->parallax_end.Get());
      }
    }

    void mark_d3d_parallax_map_start(d3d_perf_slot *slot) {
      if (slot && slot->has_parallax) {
        slot->has_parallax_map_start = true;
        context->End(slot->parallax_map_start.Get());
      }
    }

    void mark_d3d_parallax_subtitle_start(d3d_perf_slot *slot) {
      if (slot && slot->has_parallax) {
        slot->has_parallax_subtitle_start = true;
        context->End(slot->parallax_subtitle_start.Get());
      }
    }

    void release_parallax_v2_resources() {
      parallax_v2_producer_active = false;
      depth_coordinate_v2_cbuffer.Reset();
      depth_coordinate_v2_partials_buf.Reset();
      depth_coordinate_v2_partials_uav.Reset();
      depth_coordinate_v2_partials_srv.Reset();
      depth_coordinate_v2_frame_stats_buf.Reset();
      depth_coordinate_v2_frame_stats_uav.Reset();
      depth_coordinate_v2_frame_stats_srv.Reset();
      depth_coordinate_v2_state_buf.Reset();
      depth_coordinate_v2_state_uav.Reset();
      depth_coordinate_v2_state_srv.Reset();
      depth_coordinate_v2_coordinate_tex.Reset();
      depth_coordinate_v2_coordinate_uav.Reset();
      depth_coordinate_v2_coordinate_srv.Reset();
      depth_coordinate_v2_candidate_tex.Reset();
      depth_coordinate_v2_candidate_uav.Reset();
      depth_coordinate_v2_candidate_srv.Reset();
      depth_coordinate_v2_ownership_tex.Reset();
      depth_coordinate_v2_ownership_uav.Reset();
      depth_coordinate_v2_ownership_srv.Reset();
      depth_coordinate_v2_vertical_majorant_tex.Reset();
      depth_coordinate_v2_vertical_majorant_uav.Reset();
      depth_coordinate_v2_vertical_majorant_srv.Reset();
      depth_coordinate_v2_vertical_conditioned_tex.Reset();
      depth_coordinate_v2_vertical_conditioned_uav.Reset();
      depth_coordinate_v2_vertical_conditioned_srv.Reset();
      depth_coordinate_v2_final_tex.Reset();
      depth_coordinate_v2_final_uav.Reset();
      depth_coordinate_v2_final_srv.Reset();
      subtitle_conditioned_tex.Reset();
      subtitle_conditioned_uav.Reset();
      subtitle_conditioned_srv.Reset();
      subtitle_locator_state_buf.Reset();
      subtitle_locator_state_uav.Reset();
      subtitle_locator_state_srv.Reset();
      subtitle_condition_params_buf.Reset();
      subtitle_condition_params_uav.Reset();
      subtitle_condition_params_srv.Reset();
      subtitle_locator_cbuffer.Reset();
      pending_source_srv.Reset();
    }
    void fail_parallax_v2_producer(std::string_view reason) {
      if (!parallax_v2_producer_failed) {
        BOOST_LOG(error) << "Host SBS V2 producer failed: " << reason
                         << "; this stream will remain live flat identity.";
      }
      parallax_v2_producer_failed = true;
      release_parallax_v2_resources();
    }

    bool ensure_subtitle_resources() {
      auto create_uint_buffer = [&](
                                  const std::uint32_t word_count,
                                  Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer,
                                  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> &srv,
                                  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> &uav
                                ) {
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = word_count * sizeof(std::uint32_t);
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(std::uint32_t);
        return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer)) &&
               SUCCEEDED(device->CreateShaderResourceView(buffer.Get(), nullptr, &srv)) &&
               SUCCEEDED(device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
      };
      auto create_constant_buffer = [&](
                                      const UINT byte_width,
                                      Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer
                                    ) {
        if (buffer) {
          return true;
        }
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = byte_width;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer));
      };
      bool resources_ok = true;
      if (!ocr_box_record_buf || !ocr_box_record_srv || !ocr_box_record_uav) {
        ocr_box_record_buf.Reset();
        ocr_box_record_srv.Reset();
        ocr_box_record_uav.Reset();
        resources_ok = create_uint_buffer(
          ocr_box_record_word_count,
          ocr_box_record_buf,
          ocr_box_record_srv,
          ocr_box_record_uav
        );
      }
      if (!subtitle_locator_state_buf || !subtitle_locator_state_srv ||
          !subtitle_locator_state_uav) {
        subtitle_locator_state_buf.Reset();
        subtitle_locator_state_srv.Reset();
        subtitle_locator_state_uav.Reset();
        resources_ok = resources_ok && create_uint_buffer(
          subtitle_locator_state_word_count,
          subtitle_locator_state_buf,
          subtitle_locator_state_srv,
          subtitle_locator_state_uav
        );
      }
      if (!subtitle_condition_params_buf || !subtitle_condition_params_srv ||
          !subtitle_condition_params_uav) {
        subtitle_condition_params_buf.Reset();
        subtitle_condition_params_srv.Reset();
        subtitle_condition_params_uav.Reset();
        resources_ok = resources_ok && create_uint_buffer(
          subtitle_condition_param_word_count,
          subtitle_condition_params_buf,
          subtitle_condition_params_srv,
          subtitle_condition_params_uav
        );
      }
      if (!subtitle_conditioned_tex || !subtitle_conditioned_srv ||
          !subtitle_conditioned_uav) {
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = static_cast<UINT>(target_w);
        desc.Height = static_cast<UINT>(target_h);
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R32_FLOAT;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        resources_ok = resources_ok &&
          SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &subtitle_conditioned_tex)) &&
          SUCCEEDED(device->CreateShaderResourceView(
            subtitle_conditioned_tex.Get(), nullptr,
                         &subtitle_conditioned_srv
                       )) &&
          SUCCEEDED(device->CreateUnorderedAccessView(
            subtitle_conditioned_tex.Get(), nullptr,
                         &subtitle_conditioned_uav
                       ));
      }
      resources_ok = resources_ok &&
        create_constant_buffer(64u, subtitle_locator_cbuffer);
      if (!resources_ok) {
        return false;
      }

      const UINT zero[4] = {};
      context->ClearUnorderedAccessViewUint(ocr_box_record_uav.Get(), zero);
      context->ClearUnorderedAccessViewUint(subtitle_locator_state_uav.Get(), zero);
      context->ClearUnorderedAccessViewUint(subtitle_condition_params_uav.Get(), zero);
      const float zero_float[4] = {};
      context->ClearUnorderedAccessViewFloat(subtitle_conditioned_uav.Get(), zero_float);

      if (!ocr_available) {
        return true;
      }
      if (cuda_ocr_in_res && cuda_ocr_out_res && ocr_input_uav && ocr_output_srv &&
          ocr_cell_stats_srv && ocr_cell_stats_uav && ocr_preprocess_cbuffer &&
          ocr_resolve_cbuffer) {
        return true;
      }

      auto create_float_buffer = [&](
                                   const std::uint32_t float_count,
                                    const UINT bind_flags,
                                    Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer,
                                    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> *srv,
                                    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> *uav
                                 ) {
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = float_count * sizeof(float);
        desc.BindFlags = bind_flags;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
        return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer)) &&
               (!srv || SUCCEEDED(device->CreateShaderResourceView(
                          buffer.Get(), nullptr,
                          srv->ReleaseAndGetAddressOf()
                        ))) &&
               (!uav || SUCCEEDED(device->CreateUnorderedAccessView(
                          buffer.Get(), nullptr,
                          uav->ReleaseAndGetAddressOf()
                        )));
      };

      const std::uint32_t ocr_pixels = ocr_engine_width * ocr_engine_height;
      const bool ocr_input_resources_ok = create_float_buffer(
        3u * ocr_pixels,
        D3D11_BIND_UNORDERED_ACCESS,
        ocr_input_buf,
        nullptr,
        &ocr_input_uav
      );
      bool ocr_resources_ok = ocr_input_resources_ok &&
                              create_float_buffer(
                                ocr_pixels,
                                D3D11_BIND_SHADER_RESOURCE,
                                ocr_output_buf,
                                &ocr_output_srv,
                                nullptr
                              ) &&
                              create_uint_buffer(
                                ocr_cell_stats_word_count,
                                ocr_cell_stats_buf,
                                ocr_cell_stats_srv,
                                ocr_cell_stats_uav
                              ) &&
                              create_constant_buffer(32u, ocr_preprocess_cbuffer) &&
                              create_constant_buffer(64u, ocr_resolve_cbuffer);
      auto &cuda = cuda_driver_api::get();
      bool ocr_interop_teardown_failed = false;
      const auto unregister_ocr_interop = [&]() {
        // Do not short-circuit: every successfully registered member must be released before any
        // backing D3D resource can be replaced.
        const bool input_unregistered =
          unregister_cuda_graphics_resource(cuda, cuda_ocr_in_res);
        const bool output_unregistered =
          unregister_cuda_graphics_resource(cuda, cuda_ocr_out_res);
        return input_unregistered && output_unregistered;
      };
      if (ocr_resources_ok && cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) == CUDA_SUCCESS) {
        CUresult input_status = static_cast<CUresult>(-1);
        CUresult output_status = static_cast<CUresult>(-1);
        const auto register_ocr_interop = [&]() {
          input_status = cuda.cuGraphicsD3D11RegisterResource(
            &cuda_ocr_in_res, ocr_input_buf.Get(), 0
          );
          output_status = cuda.cuGraphicsD3D11RegisterResource(
            &cuda_ocr_out_res, ocr_output_buf.Get(), 0
          );
          return input_status == CUDA_SUCCESS && output_status == CUDA_SUCCESS &&
                 cuda_ocr_in_res && cuda_ocr_out_res;
        };
        ocr_resources_ok = register_ocr_interop();
        if (!ocr_resources_ok) {
          BOOST_LOG(warning)
            << "PP-OCRv6 tiny D3D11/CUDA registration failed (" << input_status << ", "
            << output_status << "); subtitle conditioning remains flat.";
        }
      } else {
        ocr_resources_ok = false;
      }
      if (!ocr_resources_ok) {
        const bool interop_released = unregister_ocr_interop();
        if (!interop_released && !ocr_interop_teardown_failed) {
          ocr_interop_teardown_failed = true;
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::unsafe_teardown
          );
        }
        if (interop_released) {
          ocr_input_buf.Reset();
          ocr_input_uav.Reset();
          ocr_output_buf.Reset();
          ocr_output_srv.Reset();
        } else {
          BOOST_LOG(error)
            << "PP-OCRv6 tiny interop cleanup did not release every CUDA resource; retaining "
               "the backing D3D resources and quarantining the OCR context.";
        }
        ocr_cell_stats_buf.Reset();
        ocr_cell_stats_uav.Reset();
        ocr_cell_stats_srv.Reset();
        ocr_available = false;
      }
      return true;
    }

    bool ensure_parallax_v2_resources() {
      using namespace depth_coordinate_v2;
      if (parallax_v2_producer_active) {
        return true;
      }
      if (!parallax_v2_producer_shaders_ready || parallax_v2_producer_failed ||
          target_w <= 0 || target_h <= 0 || reduce_groups == 0) {
        return false;
      }
      if (!raw_model_provenance || !depth_coordinate_v2::capture_identity_is_calibrated(raw_model_provenance->depth_model, raw_model_provenance->depth_model_url,
            raw_model_provenance->onnx_sha256,
            raw_model_provenance->preprocess_profile,
            raw_model_provenance->preprocess_source_closure_sha256,
            static_cast<std::uint32_t>(target_w), static_cast<std::uint32_t>(target_h))) {
        std::ostringstream reason;
        reason << "model identity, preprocess profile/source closure, or input shape "
               << target_w << 'x' << target_h
               << " is outside the calibrated allowlist";
        fail_parallax_v2_producer(reason.str());
        return false;
      }

      auto create_float4_buffer = [&](
                                    std::size_t vector_count,
                                    const void *initial_data,
                                    Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer,
                                    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> &srv,
                                    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> &uav
                                  ) {
        const std::size_t byte_count = vector_count * sizeof(float) * 4u;
        if (byte_count == 0 || byte_count > std::numeric_limits<UINT>::max()) {
          return false;
        }
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = static_cast<UINT>(byte_count);
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float) * 4u;
        D3D11_SUBRESOURCE_DATA init {initial_data, 0, 0};
        return SUCCEEDED(device->CreateBuffer(
                 &desc,
                 initial_data ? &init : nullptr,
                 &buffer
               )) &&
               SUCCEEDED(device->CreateShaderResourceView(buffer.Get(), nullptr, &srv)) &&
               SUCCEEDED(device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
      };

      bool resources_ok = create_float4_buffer(
                            static_cast<std::size_t>(reduce_groups) * 3u,
                            nullptr,
                            depth_coordinate_v2_partials_buf,
                            depth_coordinate_v2_partials_srv,
                            depth_coordinate_v2_partials_uav
                          ) &&
                          create_float4_buffer(
                            frame_stats_vector_count,
                            nullptr,
                            depth_coordinate_v2_frame_stats_buf,
                            depth_coordinate_v2_frame_stats_srv,
                            depth_coordinate_v2_frame_stats_uav
                          );

      resources_ok = resources_ok &&
                     create_float4_buffer(
                       state_vector_count,
                       state_initial_words.data(),
                       depth_coordinate_v2_state_buf,
                       depth_coordinate_v2_state_srv,
                       depth_coordinate_v2_state_uav
                     );

      D3D11_TEXTURE2D_DESC texture_desc {};
      texture_desc.Width = static_cast<UINT>(target_w);
      texture_desc.Height = static_cast<UINT>(target_h);
      texture_desc.MipLevels = 1;
      texture_desc.ArraySize = 1;
      texture_desc.Format = DXGI_FORMAT_R32_FLOAT;
      texture_desc.SampleDesc.Count = 1;
      texture_desc.Usage = D3D11_USAGE_DEFAULT;
      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      auto create_float_texture = [&](
                                    Microsoft::WRL::ComPtr<ID3D11Texture2D> &texture,
                                  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> *srv,
                                  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> &uav
                                  ) {
        return SUCCEEDED(device->CreateTexture2D(&texture_desc, nullptr, &texture)) &&
               (!srv || SUCCEEDED(device->CreateShaderResourceView(
                           texture.Get(),
                           nullptr,
                           srv->ReleaseAndGetAddressOf()
                         ))) &&
               SUCCEEDED(device->CreateUnorderedAccessView(texture.Get(), nullptr, &uav));
      };
      resources_ok = resources_ok &&
                     create_float_texture(
                        depth_coordinate_v2_candidate_tex,
                        &depth_coordinate_v2_candidate_srv,
                        depth_coordinate_v2_candidate_uav
                      ) &&
                     create_float_texture(
                       depth_coordinate_v2_ownership_tex,
                       &depth_coordinate_v2_ownership_srv,
                       depth_coordinate_v2_ownership_uav
                     ) &&
                     create_float_texture(
                       depth_coordinate_v2_vertical_majorant_tex,
                       &depth_coordinate_v2_vertical_majorant_srv,
                       depth_coordinate_v2_vertical_majorant_uav
                     ) &&
                     create_float_texture(
                       depth_coordinate_v2_vertical_conditioned_tex,
                       &depth_coordinate_v2_vertical_conditioned_srv,
                       depth_coordinate_v2_vertical_conditioned_uav
                     ) &&
                     create_float_texture(
                       depth_coordinate_v2_final_tex,
                       &depth_coordinate_v2_final_srv,
                       depth_coordinate_v2_final_uav
                     );

      const constants_t constants {
        .raw_coordinate_scale = parallax_v2_raw_coordinate_scale,
        .collapse_abs_epsilon = collapse_abs_epsilon,
        .far_tau = far_tau,
        .near_log_tau = near_log_tau,
        .requested_gain = parallax_v2_requested_gain,
        .max_horizontal_slope = max_horizontal_slope,
        .direct_container_limit = direct_container_limit,
        .convergence_curve_default = convergence_curve_default,
      };
      D3D11_BUFFER_DESC constants_desc {};
      constants_desc.Usage = D3D11_USAGE_IMMUTABLE;
      constants_desc.ByteWidth = sizeof(constants);
      constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA constants_data {&constants, 0, 0};
      resources_ok = resources_ok && SUCCEEDED(device->CreateBuffer(
        &constants_desc,
        &constants_data,
        &depth_coordinate_v2_cbuffer
      ));
      resources_ok = resources_ok && ensure_subtitle_resources();

      if (!resources_ok) {
        fail_parallax_v2_producer("GPU resource allocation failed"sv);
        return false;
      }

      const float clear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      context->ClearUnorderedAccessViewFloat(depth_coordinate_v2_candidate_uav.Get(), clear);
      context->ClearUnorderedAccessViewFloat(depth_coordinate_v2_ownership_uav.Get(), clear);
      context->ClearUnorderedAccessViewFloat(
        depth_coordinate_v2_vertical_majorant_uav.Get(),
        clear
      );
      context->ClearUnorderedAccessViewFloat(
        depth_coordinate_v2_vertical_conditioned_uav.Get(),
        clear
      );
      context->ClearUnorderedAccessViewFloat(depth_coordinate_v2_final_uav.Get(), clear);
      parallax_v2_producer_active = true;
      BOOST_LOG(info)
        << "Host SBS V2 GPU producer active at " << target_w << 'x' << target_h
        << "; full-resolution conservative foreground ownership precedes the 75% vertical "
           "upper and 25% vertical lower envelope "
           "followed by one horizontal majorant; the renderer will authenticate its first "
           "completed field before the one-time "
           "V2-or-flat latch.";
      return true;
    }

    bool dispatch_parallax_v2_frame_stats(d3d_perf_slot *perf_slot) {
      if (!parallax_v2_producer_active) {
        return false;
      }

      mark_d3d_parallax_frame_stats_start(perf_slot);
      ID3D11Buffer *constant_buffers[2] = {
        cbuffer.Get(),
        depth_coordinate_v2_cbuffer.Get(),
      };
      context->CSSetConstantBuffers(0, 2, constant_buffers);

      // Stable frame mean/std/min/max.
      context->CSSetShader(depth_coordinate_v2_moments_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *moments_srvs[2] = {
        tensor_out_srv.Get(),
        tensor_exclusion_srv.Get(),
      };
      context->CSSetShaderResources(0, 2, moments_srvs);
      context->CSSetUnorderedAccessViews(
        0,
        1,
        depth_coordinate_v2_partials_uav.GetAddressOf(),
        nullptr
      );
      dispatch_infer_postprocess(
        reduce_groups,
        1u,
        1u,
        near_identical_gpu_infer_reduce_byte_offset
      );
      ID3D11ShaderResourceView *null_srvs3[3] = {nullptr, nullptr, nullptr};
      ID3D11UnorderedAccessView *null_uavs2[2] = {nullptr, nullptr};
      context->CSSetShaderResources(0, 2, null_srvs3);
      context->CSSetUnorderedAccessViews(0, 1, null_uavs2, nullptr);

      context->CSSetShader(depth_coordinate_v2_frame_resolve_cs.Get(), nullptr, 0);
      context->CSSetShaderResources(0, 1, depth_coordinate_v2_partials_srv.GetAddressOf());
      ID3D11UnorderedAccessView *frame_uavs[2] = {
        depth_coordinate_v2_frame_stats_uav.Get(),
        minmax_raw_uav.Get(),
      };
      context->CSSetUnorderedAccessViews(0, 2, frame_uavs, nullptr);
      dispatch_infer_postprocess(
        1u, 1u, 1u, near_identical_gpu_infer_one_byte_offset
      );
      context->CSSetShaderResources(0, 1, null_srvs3);
      context->CSSetUnorderedAccessViews(0, 2, null_uavs2, nullptr);
      mark_d3d_parallax_frame_stats_end(perf_slot);
      return true;
    }

    bool dispatch_parallax_v2_producer(d3d_perf_slot *perf_slot) {
      if (!parallax_v2_producer_active) {
        return false;
      }

      ID3D11Buffer *constant_buffers[2] = {
        cbuffer.Get(),
        depth_coordinate_v2_cbuffer.Get(),
      };
      context->CSSetConstantBuffers(0, 2, constant_buffers);
      ID3D11ShaderResourceView *null_srvs3[3] = {nullptr, nullptr, nullptr};
      ID3D11UnorderedAccessView *null_uavs2[2] = {nullptr, nullptr};

      // Acquire the first usable camera on startup/cut, then hold it until the next authenticated
      // cut. The model scale is fixed. An unusable no-cut frame publishes flat without erasing
      // the retained camera.
      context->CSSetShader(depth_coordinate_v2_state_resolve_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *state_srvs[2] = {
        depth_coordinate_v2_frame_stats_srv.Get(),
        cut_state_srv.Get(),
      };
      context->CSSetShaderResources(0, 2, state_srvs);
      context->CSSetUnorderedAccessViews(
        0, 1, depth_coordinate_v2_state_uav.GetAddressOf(),
        nullptr
      );
      dispatch_infer_postprocess(
        1u, 1u, 1u, near_identical_gpu_infer_one_byte_offset
      );
      context->CSSetShaderResources(0, 2, null_srvs3);
      context->CSSetUnorderedAccessViews(0, 1, null_uavs2, nullptr);

      mark_d3d_parallax_map_start(perf_slot);

      // Raw depth -> immutable pre-limiter candidate. The full-size canonical-coordinate field
      // is deliberately absent from production; an explicit Dump 3D dispatches it separately.
      context->CSSetShader(depth_coordinate_v2_map_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *map_srvs[3] = {
        tensor_out_srv.Get(),
        depth_coordinate_v2_state_srv.Get(),
        tensor_exclusion_srv.Get(),
      };
      context->CSSetShaderResources(0, 3, map_srvs);
      context->CSSetUnorderedAccessViews(
        0,
        1,
        depth_coordinate_v2_candidate_uav.GetAddressOf(),
        nullptr
      );
      dispatch_infer_postprocess(
        (target_w + 15) / 16,
        (target_h + 15) / 16,
        1u,
        near_identical_gpu_infer_grid16_byte_offset
      );
      context->CSSetShaderResources(0, 3, null_srvs3);
      context->CSSetUnorderedAccessViews(0, 1, null_uavs2, nullptr);

      // Candidate -> conservative full-resolution RGB ownership refinement. The retained source
      // SRV belongs to this exact asynchronous raw-depth completion. Missing source evidence is a
      // safe identity copy; ordinary production pairing always supplies it.
      if (pending_source_srv) {
        context->CSSetShader(depth_coordinate_v2_ownership_cs.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0u, 1u, cbuffer.GetAddressOf());
        context->CSSetConstantBuffers(
          2u, 1u, source_region_cbuffer.GetAddressOf()
        );
        ID3D11ShaderResourceView *ownership_srvs[3] = {
          depth_coordinate_v2_candidate_srv.Get(),
          pending_source_srv.Get(),
          tensor_exclusion_srv.Get(),
        };
        context->CSSetShaderResources(0, 3, ownership_srvs);
        context->CSSetUnorderedAccessViews(
          0,
          1,
          depth_coordinate_v2_ownership_uav.GetAddressOf(),
          nullptr
        );
        if (perf_slot) {
          perf_slot->has_ownership = true;
          context->End(perf_slot->ownership_start.Get());
        }
        dispatch_infer_postprocess(
          (target_w + 7) / 8,
          (target_h + 7) / 8,
          1u,
          near_identical_gpu_infer_grid8_byte_offset
        );
        if (perf_slot) {
          context->End(perf_slot->ownership_end.Get());
        }
        context->CSSetShaderResources(0, 3, null_srvs3);
        context->CSSetUnorderedAccessViews(0, 1, null_uavs2, nullptr);
      } else if (!gpu_undecided_postprocess_pending()) {
        context->CopyResource(
          depth_coordinate_v2_ownership_tex.Get(),
          depth_coordinate_v2_candidate_tex.Get()
        );
      }

      // Produce the exact column upper envelope plus the fixed 75/25 vertical share. The upper
      // remains diagnostic evidence; only the neutral conditioned field feeds live geometry.
      context->CSSetShader(depth_coordinate_v2_vertical_limit_cs.Get(), nullptr, 0);
      context->CSSetShaderResources(0, 1, depth_coordinate_v2_ownership_srv.GetAddressOf());
      ID3D11UnorderedAccessView *vertical_envelope_uavs[2] = {
        depth_coordinate_v2_vertical_majorant_uav.Get(),
        depth_coordinate_v2_vertical_conditioned_uav.Get(),
      };
      context->CSSetUnorderedAccessViews(0, 2, vertical_envelope_uavs, nullptr);
      dispatch_infer_postprocess(
        target_w,
        1u,
        1u,
        near_identical_gpu_infer_columns_byte_offset
      );
      context->CSSetShaderResources(0, 1, null_srvs3);
      context->CSSetUnorderedAccessViews(0, 2, null_uavs2, nullptr);

      // Apply one pure horizontal majorant to the vertically conditioned field. There is no
      // horizontal minorant or completed-2D-envelope blend, so lateral lowering is not added.
      context->CSSetShader(depth_coordinate_v2_limit_cs.Get(), nullptr, 0);
      context->CSSetShaderResources(
        0,
        1,
        depth_coordinate_v2_vertical_conditioned_srv.GetAddressOf()
      );
      context->CSSetUnorderedAccessViews(
        0,
        1,
        depth_coordinate_v2_final_uav.GetAddressOf(),
        nullptr
      );
      dispatch_infer_postprocess(
        target_h,
        1u,
        1u,
        near_identical_gpu_infer_rows_byte_offset
      );
      context->CSSetShaderResources(0, 1, null_srvs3);
      context->CSSetUnorderedAccessViews(0, 1, null_uavs2, nullptr);

      mark_d3d_parallax_subtitle_start(perf_slot);

      ID3D11Buffer *null_constants[2] = {nullptr, nullptr};
      context->CSSetConstantBuffers(1, 2, null_constants);
      return true;
    }

    bool update_pending_ocr_constants() {
      if (!ocr_resolve_cbuffer || pending_frame_id == 0u || target_w <= 0 || target_h <= 0) {
        return false;
      }
      const std::uint32_t source_width = pending_input_region.width();
      const std::uint32_t source_height = pending_input_region.height();
      const auto roi = fit_subtitle_analysis_geometry(
        source_width,
        source_height,
        {target_w, target_h},
        pending_input_region.tensor_content
      );
      if (!roi.valid()) {
        return false;
      }
      const std::uint32_t crop_height = subtitle_ocr_source_crop_height(
        source_width,
        source_height
      );
      if (crop_height == 0u || crop_height > source_height) {
        return false;
      }
      const ocr_resolve_constants_t constants {
        static_cast<std::uint32_t>(pending_frame_id),
        static_cast<std::uint32_t>(pending_frame_id >> 32u),
        static_cast<std::uint32_t>(pending_input_region.analysis_generation),
        static_cast<std::uint32_t>(pending_input_region.analysis_generation >> 32u),
        source_width,
        source_height,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h),
        source_height - crop_height,
        crop_height,
        roi.roi_top,
        roi.roi_bottom,
        roi.tensor_content.left,
        roi.tensor_content.top,
        roi.tensor_content.right,
        roi.tensor_content.bottom,
      };
      context->UpdateSubresource(
        ocr_resolve_cbuffer.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
      return true;
    }

    bool dispatch_ocr_postprocess() {
      const bool optional_child_armed = std::exchange(pending_ocr_submitted, false);
      if (!near_identical_gpu_dispatch_buf || !ocr_box_record_uav ||
          !ocr_resolve_cbuffer) {
        return false;
      }
      const bool optional_postprocess_ready =
        ocr_box_cells_cs && ocr_box_resolve_cs && ocr_output_srv &&
        ocr_cell_stats_srv && ocr_cell_stats_uav;
      if (optional_child_armed && !optional_postprocess_ready) {
        // Once the root has joined an OCR child, losing its exact postprocess resources must fail
        // this unit closed. Publishing a miss would hide a completed-but-unconsumed OCR result.
        return false;
      }
      ID3D11ShaderResourceView *null_srv = nullptr;
      ID3D11UnorderedAccessView *null_uav = nullptr;
      if (optional_postprocess_ready) {
        context->CSSetShader(ocr_box_cells_cs.Get(), nullptr, 0u);
        context->CSSetShaderResources(0u, 1u, ocr_output_srv.GetAddressOf());
        context->CSSetUnorderedAccessViews(
          0u, 1u, ocr_cell_stats_uav.GetAddressOf(), nullptr
        );
        context->DispatchIndirect(
          near_identical_gpu_dispatch_buf.Get(),
          near_identical_gpu_optional_cells_byte_offset
        );
        context->CSSetShaderResources(0u, 1u, &null_srv);
        context->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);

        context->CSSetShader(ocr_box_resolve_cs.Get(), nullptr, 0u);
        context->CSSetConstantBuffers(0u, 1u, ocr_resolve_cbuffer.GetAddressOf());
        context->CSSetShaderResources(1u, 1u, ocr_cell_stats_srv.GetAddressOf());
        context->CSSetUnorderedAccessViews(
          1u, 1u, ocr_box_record_uav.GetAddressOf(), nullptr
        );
        context->DispatchIndirect(
          near_identical_gpu_dispatch_buf.Get(),
          near_identical_gpu_optional_one_byte_offset
        );
        context->CSSetShaderResources(1u, 1u, &null_srv);
        context->CSSetUnorderedAccessViews(1u, 1u, &null_uav, nullptr);
      }

      // The detector finalizer already published the exact-current abstention for every
      // authenticated observation without OOCR. Only a matching optional receipt reaches the
      // resolver above, so there is no second fallback dispatch here.
      ID3D11Buffer *null_buffer = nullptr;
      context->CSSetConstantBuffers(0u, 1u, &null_buffer);
      return true;
    }

    bool dispatch_subtitle_conditioner(
      const bool ocr_record_submitted,
      const bool input_domain_reset
    ) {
      if (!subtitle_locator_resolve_cs || !subtitle_condition_prepare_cs ||
          !subtitle_condition_cs ||
          !subtitle_locator_cbuffer || !subtitle_locator_state_srv ||
          !subtitle_locator_state_uav || !subtitle_conditioned_srv ||
          !subtitle_conditioned_uav || !depth_coordinate_v2_final_srv ||
          !depth_coordinate_v2_final_uav ||
          !depth_coordinate_v2_final_tex || !subtitle_conditioned_tex ||
          !subtitle_condition_params_srv || !subtitle_condition_params_uav ||
          !ocr_box_record_srv ||
          !cut_state_srv || !near_identical_gpu_dispatch_buf) {
        return false;
      }

      const std::uint32_t source_width = pending_input_region.width();
      const std::uint32_t source_height = pending_input_region.height();
      const std::uint32_t field_width = target_w > 0 ?
                                          static_cast<std::uint32_t>(target_w) :
                                          0u;
      const std::uint32_t field_height = target_h > 0 ?
                                           static_cast<std::uint32_t>(target_h) :
                                           0u;
      const auto roi = fit_subtitle_analysis_geometry(
        source_width,
        source_height,
        {target_w, target_h},
        pending_input_region.tensor_content
      );
      // OCR8/SLR follows the current authenticated analysis field. An unsupported tensor or an
      // unprojectable bottom crop has no subtitle authority; the complete condition writer still
      // publishes Base exactly into the distinct live output texture.
      const bool locator_geometry_valid = roi.valid();
      const auto tensor_content = pending_input_region.tensor_content;
      const std::array<std::uint32_t, 16> constants {
        field_width,
        field_height,
        roi.roi_top,
        roi.roi_bottom,
        source_width,
        source_height,
        ocr_record_submitted ? 1u : 0u,
        input_domain_reset ? 1u : 0u,
        static_cast<std::uint32_t>(pending_frame_id),
        static_cast<std::uint32_t>(pending_frame_id >> 32u),
        static_cast<std::uint32_t>(pending_input_region.analysis_generation),
        static_cast<std::uint32_t>(pending_input_region.analysis_generation >> 32u),
        tensor_content.left,
        tensor_content.top,
        tensor_content.right,
        tensor_content.bottom,
      };
      context->UpdateSubresource(
        subtitle_locator_cbuffer.Get(), 0, nullptr, constants.data(), 0,
        0
      );

      ID3D11Buffer *constant_buffers[3] = {
        cbuffer.Get(),
        depth_coordinate_v2_cbuffer.Get(),
        subtitle_locator_cbuffer.Get(),
      };
      ID3D11ShaderResourceView *null_srvs[9] = {};
      ID3D11UnorderedAccessView *null_uav = nullptr;
      context->CSSetConstantBuffers(0, 3, constant_buffers);
      // resolve_main already publishes an empty authenticated state for invalid geometry. Keep
      // that reset inside the same authenticated observation dispatch; only explicit subtitle
      // suppression freezes SLR byte-for-byte.
      (void) locator_geometry_valid;
      context->CSSetShader(subtitle_locator_resolve_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *resolve_srvs[8] = {
        nullptr,
        cut_state_srv.Get(),
        depth_coordinate_v2_final_srv.Get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        ocr_box_record_srv.Get(),
      };
      context->CSSetShaderResources(0, 8, resolve_srvs);
      context->CSSetUnorderedAccessViews(
        2, 1, subtitle_locator_state_uav.GetAddressOf(),
        nullptr
      );
      context->DispatchIndirect(
        near_identical_gpu_dispatch_buf.Get(),
        near_identical_gpu_observation_one_byte_offset
      );
      context->CSSetShaderResources(0, 8, null_srvs);
      context->CSSetUnorderedAccessViews(2, 1, &null_uav, nullptr);

      // Authenticate the complete SLR13 state once, not once per 16x16 field group, and publish
      // the immutable parameters consumed by the complete out-of-place writer.
      context->CSSetShader(subtitle_condition_prepare_cs.Get(), nullptr, 0);
      ID3D11ShaderResourceView *prepare_srvs[8] = {
        nullptr,
        cut_state_srv.Get(),
        nullptr,
        subtitle_locator_state_srv.Get(),
        nullptr,
        nullptr,
        nullptr,
        ocr_box_record_srv.Get(),
      };
      context->CSSetShaderResources(0, 8, prepare_srvs);
      context->CSSetUnorderedAccessViews(
        4, 1, subtitle_condition_params_uav.GetAddressOf(), nullptr
      );
      context->DispatchIndirect(
        near_identical_gpu_dispatch_buf.Get(),
        near_identical_gpu_observation_one_byte_offset
      );
      context->CSSetShaderResources(0, 8, null_srvs);
      context->CSSetUnorderedAccessViews(4, 1, &null_uav, nullptr);

      // Base remains immutable. Every authenticated subtitle observation writes the complete
      // out-of-place field, including authoritative empty/abstaining observations. Ordinary work
      // holds the prior conditioned target on reuse; cadence-due work may advance it independently.
      ID3D11ShaderResourceView *condition_srvs[5] = {
        nullptr,
        nullptr,
        depth_coordinate_v2_final_srv.Get(),
        subtitle_locator_state_srv.Get(),
        subtitle_condition_params_srv.Get(),
      };
      context->CSSetShaderResources(0, 5, condition_srvs);
      ID3D11UnorderedAccessView *condition_output = subtitle_conditioned_uav.Get();
      context->CSSetUnorderedAccessViews(3, 1, &condition_output, nullptr);
      context->CSSetShader(subtitle_condition_cs.Get(), nullptr, 0);
      context->DispatchIndirect(
        near_identical_gpu_dispatch_buf.Get(),
        near_identical_gpu_subtitle_condition_grid16_byte_offset
      );
      context->CSSetShaderResources(0, 5, null_srvs);
      context->CSSetUnorderedAccessViews(3, 1, &null_uav, nullptr);
      ID3D11Buffer *null_constants[3] = {};
      context->CSSetConstantBuffers(0, 3, null_constants);
      return true;
    }

    void dispatch_pending_gpu_completion_trace() {
      if (!diagnostics_enabled) {
        pending_gpu_trace_append = {};
        return;
      }
      const auto append = std::exchange(pending_gpu_trace_append, {});
      if (!append.valid || !gpu_trace_cs || !gpu_trace_transaction_buf ||
          !gpu_trace_transaction_srv || !near_identical_gpu_decision_buf ||
          !gpu_trace_cbuffer || !gpu_trace_ring_uav || !subtitle_locator_state_srv ||
          !subtitle_condition_params_srv || !gpu_trace_provenance) {
        return;
      }
      // Snapshot the complete postprocessed CBRG/RQST transaction only for this append, after the
      // production postprocess timer has ended and before the next enqueue can reuse the source.
      context->CopyResource(
        gpu_trace_transaction_buf.Get(), near_identical_gpu_decision_buf.Get()
      );

      const auto domain_tag = near_identical_input_domain_tag(
        pending_input_region,
        pending_color_space,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      std::uint32_t flags = 0u;
      if (append.input_domain_reset) {
        flags |= host_sbs_gpu_trace::record_flag_e::input_domain_reset;
      }
      if (pending_dump_forced) {
        flags |= host_sbs_gpu_trace::record_flag_e::dump_forced;
      }
      if (append.ocr_record_submitted) {
        flags |= host_sbs_gpu_trace::record_flag_e::ocr_record_submitted;
      }
      if (append.subtitle_work_suppressed) {
        flags |= host_sbs_gpu_trace::record_flag_e::subtitle_suppressed;
      }
      if (append.condition_executed) {
        flags |= host_sbs_gpu_trace::record_flag_e::condition_executed;
      }
      if (append.subtitle_branch_gated) {
        flags |= host_sbs_gpu_trace::record_flag_e::subtitle_branch_gated;
      }
      const auto submission_class =
        pending_submission_class == pending_submission_class_e::gpu_undecided ?
          host_sbs_gpu_trace::submission_class_e::gpu_undecided :
          host_sbs_gpu_trace::submission_class_e::force_infer;
      const std::array<std::uint32_t, host_sbs_gpu_trace::constant_word_count> constants {
        static_cast<std::uint32_t>(pending_frame_id),
        static_cast<std::uint32_t>(pending_frame_id >> 32u),
        static_cast<std::uint32_t>(pending_input_region.analysis_generation),
        static_cast<std::uint32_t>(pending_input_region.analysis_generation >> 32u),
        static_cast<std::uint32_t>(domain_tag),
        static_cast<std::uint32_t>(domain_tag >> 32u),
        static_cast<std::uint32_t>(pending_wrapper_transaction_token),
        static_cast<std::uint32_t>(pending_wrapper_transaction_token >> 32u),
        cuda_conditional_graph::work_flags_value(pending_subtitle_work),
        static_cast<std::uint32_t>(submission_class),
        flags,
        static_cast<std::uint32_t>(append.host_subtitle_outcome),
        pending_input_region.width(),
        pending_input_region.height(),
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h),
        static_cast<std::uint32_t>(pending_observation_timestamp_us),
        static_cast<std::uint32_t>(pending_observation_timestamp_us >> 32u),
        0u,
        0u,
      };
      context->UpdateSubresource(
        gpu_trace_cbuffer.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
      ID3D11ShaderResourceView *inputs[3] = {
        gpu_trace_transaction_srv.Get(),
        subtitle_locator_state_srv.Get(),
        subtitle_condition_params_srv.Get(),
      };
      context->CSSetShader(gpu_trace_cs.Get(), nullptr, 0u);
      context->CSSetConstantBuffers(0u, 1u, gpu_trace_cbuffer.GetAddressOf());
      context->CSSetShaderResources(0u, 3u, inputs);
      context->CSSetUnorderedAccessViews(
        0u, 1u, gpu_trace_ring_uav.GetAddressOf(), nullptr
      );
      context->Dispatch(1u, 1u, 1u);

      ID3D11ShaderResourceView *null_inputs[3] = {};
      ID3D11UnorderedAccessView *null_output = nullptr;
      ID3D11Buffer *null_constant = nullptr;
      context->CSSetShaderResources(0u, 3u, null_inputs);
      context->CSSetUnorderedAccessViews(0u, 1u, &null_output, nullptr);
      context->CSSetConstantBuffers(0u, 1u, &null_constant);
      context->CSSetShader(nullptr, nullptr, 0u);
    }

    bool ensure_parallax_v2_coordinate_diagnostic_resource() {
      if (depth_coordinate_v2_coordinate_tex && depth_coordinate_v2_coordinate_uav &&
          depth_coordinate_v2_coordinate_srv) {
        return true;
      }
      if (!parallax_v2_producer_active || target_w <= 0 || target_h <= 0) {
        return false;
      }
      if (!depth_coordinate_v2_coordinate_diagnostic_cs) {
        const auto diagnostic_sources = host_sbs_shader_cache::snapshot_sources(
          shader_root,
          host_sbs_shader_cache::parallax_v2_diagnostic_specs
        );
        if (!diagnostic_sources || !create_shader(diagnostic_sources, host_sbs_shader_cache::depth_coordinate_v2_coordinate_diagnostic, depth_coordinate_v2_coordinate_diagnostic_cs)) {
          if (!depth_coordinate_v2_coordinate_diagnostic_error_logged) {
            BOOST_LOG(warning)
              << "Host SBS V2 canonical-coordinate Dump 3D shader is unavailable; "
                 "live rendering remains active.";
            depth_coordinate_v2_coordinate_diagnostic_error_logged = true;
          }
          return false;
        }
      }

      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = static_cast<UINT>(target_w);
      desc.Height = static_cast<UINT>(target_h);
      desc.MipLevels = 1;
      desc.ArraySize = 1;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      desc.SampleDesc.Count = 1;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      const bool created =
        SUCCEEDED(device->CreateTexture2D(
          &desc,
          nullptr,
          depth_coordinate_v2_coordinate_tex.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateShaderResourceView(
          depth_coordinate_v2_coordinate_tex.Get(),
          nullptr,
          depth_coordinate_v2_coordinate_srv.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateUnorderedAccessView(
          depth_coordinate_v2_coordinate_tex.Get(),
          nullptr,
          depth_coordinate_v2_coordinate_uav.ReleaseAndGetAddressOf()
        ));
      if (!created) {
        depth_coordinate_v2_coordinate_tex.Reset();
        depth_coordinate_v2_coordinate_uav.Reset();
        depth_coordinate_v2_coordinate_srv.Reset();
        if (!depth_coordinate_v2_coordinate_diagnostic_error_logged) {
          BOOST_LOG(warning)
            << "Host SBS V2 could not allocate the Dump 3D canonical-coordinate texture; "
               "live rendering is unaffected.";
          depth_coordinate_v2_coordinate_diagnostic_error_logged = true;
        }
      }
      return created;
    }

    bool dispatch_parallax_v2_coordinate_diagnostic() {
      if (!ensure_parallax_v2_coordinate_diagnostic_resource()) {
        return false;
      }

      ID3D11Buffer *constant_buffers[2] = {
        cbuffer.Get(),
        depth_coordinate_v2_cbuffer.Get(),
      };
      ID3D11ShaderResourceView *inputs[3] = {
        tensor_out_srv.Get(),
        depth_coordinate_v2_state_srv.Get(),
        tensor_exclusion_srv.Get(),
      };
      context->CSSetShader(
        depth_coordinate_v2_coordinate_diagnostic_cs.Get(),
        nullptr,
        0
      );
      context->CSSetConstantBuffers(0, 2, constant_buffers);
      context->CSSetShaderResources(0, 3, inputs);
      context->CSSetUnorderedAccessViews(
        0,
        1,
        depth_coordinate_v2_coordinate_uav.GetAddressOf(),
        nullptr
      );
      context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

      ID3D11ShaderResourceView *null_srvs[3] = {nullptr, nullptr, nullptr};
      ID3D11UnorderedAccessView *null_uav = nullptr;
      ID3D11Buffer *null_constant = nullptr;
      context->CSSetShaderResources(0, 3, null_srvs);
      context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
      context->CSSetConstantBuffers(1, 1, &null_constant);
      return true;
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
      bool coordinate_snapshot_valid = false,
      depth_input_region_t completed_input_region = {},
      input_color_space completed_color_space = input_color_space::srgb,
      bool input_domain_reset = false,
      bool subtitle_work_suppressed = false,
      bool subtitle_ocr_inference_enqueued = false,
      bool gpu_undecided_transaction_enqueued = false,
      bool gpu_undecided_completion = false
    ) {
      estimate_result r;
      r.depth = output_srv();
      r.cut_state = cut_state_srv;
      r.depth_frame_state = minmax_ema_srv;
      r.ema_motion_mask = ema_motion_mask_srv;
      r.raw_model_depth = tensor_out_srv;
      if (raw_snapshot_valid) {
        r.raw_model_depth_snapshot = raw_snapshot_srv;
      }
      if (model_input_snapshot_valid) {
        r.model_input_snapshot = model_input_snapshot_srv;
      }
      r.raw_model_provenance = raw_model_provenance;
      if (parallax_v2_producer_active) {
        if (coordinate_snapshot_valid) {
          r.shadow_coordinate = depth_coordinate_v2_coordinate_srv;
        }
        r.shadow_candidate_parallax = depth_coordinate_v2_candidate_srv;
        r.shadow_ownership_refined_parallax = depth_coordinate_v2_ownership_srv;
        r.shadow_vertical_majorant = depth_coordinate_v2_vertical_majorant_srv;
        r.shadow_vertical_conditioned = depth_coordinate_v2_vertical_conditioned_srv;
        r.shadow_base_final_parallax = depth_coordinate_v2_final_srv;
        r.shadow_final_parallax = subtitle_conditioned_srv;
        r.shadow_state = depth_coordinate_v2_state_srv;
        r.shadow_frame_stats = depth_coordinate_v2_frame_stats_srv;
        if (host_sbs_v2_depth_shape_is_authenticated({target_w, target_h})) {
          r.ocr_box_record = ocr_box_record_srv;
          r.subtitle_locator_state = subtitle_locator_state_srv;
        }
        r.parallax_v2_shader_provenance = parallax_v2_shader_provenance;
        if (gpu_trace_ring_srv && gpu_trace_provenance) {
          r.gpu_trace_ring = gpu_trace_ring_srv;
          r.gpu_trace_provenance = gpu_trace_provenance;
        }
        r.parallax_v2_producer_active = true;
        r.parallax_v2_raw_coordinate_scale = parallax_v2_raw_coordinate_scale;
        r.parallax_v2_requested_pop_strength = parallax_v2_requested_pop_strength;
        r.parallax_v2_requested_gain = parallax_v2_requested_gain;
      }
      r.raw_width = target_w;
      r.raw_height = target_h;
      r.completed_frame_valid = completed_frame_valid;
      r.completed_frame_id = completed_frame_id;
      r.inference_enqueued = inference_enqueued;
      r.gpu_undecided_transaction_enqueued =
        gpu_undecided_transaction_enqueued;
      r.gpu_undecided_completion = gpu_undecided_completion;
      r.subtitle_ocr_inference_enqueued = subtitle_ocr_inference_enqueued;
      r.cuda_graph_active = depth_conditional_graph.ready() &&
                            !depth_inference_graph.policy.capture_failed;
      if (completed_frame_valid) {
        r.input_region = completed_input_region;
        r.color_space = completed_color_space;
        r.input_domain_reset = input_domain_reset;
        r.subtitle_work_suppressed = subtitle_work_suppressed;
      }
      return r;
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

    // estimate() has already submitted one inference. Wait for that exact inference, consume it
    // once, and deliberately do NOT enqueue a duplicate. This is the synchronous quality oracle.
    estimate_result finish_pending(
      input_color_space color_space,
      bool snapshot_debug_inputs = false,
      const bool root_completion_proven = false
    ) {
      // A caller may first prove nonblocking readiness with can_accept(). Consuming instead of
      // enqueueing must retire that admission token so a later estimate cannot skip its own query.
      readiness_preflighted = false;
      auto &cuda = cuda_driver_api::get();
      if (!has_previous_frame) {
        return make_result();
      }
      if (gpu_undecided_postprocess_pending()) {
        snapshot_debug_inputs = false;
      }
      if (cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
        BOOST_LOG(error) << "cuCtxSetCurrent failed while finishing pending depth.";
        mark_cuda_context_failure();
        return make_result();
      }
      // A nonblocking or bounded try-finish caller may prove the joined root complete through its
      // post-unmap event or, when events are unavailable, a successful root-stream query. D3D11
      // postprocess is then ordered behind the same release boundary without another synchronize.
      // Offline evaluation alone retains the conservative synchronization path.
      if (!root_completion_proven && !synchronize_pending_execution(cuda)) {
        return make_result();
      }
      if (diagnostics_enabled) {
        perf_drain(perf_depth_conditional);
        if (terminal_failure) {
          return make_result();
        }
      }
      (void) color_space;  // the pending frame owns its transfer mode
      ensure_cbuffers(pending_color_space, pending_input_region);
      ensure_source_region_cbuffer(pending_input_region);
      if (!cbuffer || !source_region_cbuffer) {
        // A persistent constant-buffer allocation failure must latch terminal like every other
        // resource failure; returning empty without latching leaves the caller retrying at the
        // minimum-FPS cadence forever.
        BOOST_LOG(error)
          << "Depth/source-region constant-buffer creation failed while finishing a pending frame";
        mark_terminal_failure();
        return make_result();
      }
      auto *d3d_timer = diagnostics_enabled ? begin_d3d_perf(true, false) : nullptr;
      const bool input_domain_reset = prepare_pending_input_domain();
      if (!normalize_depth_output(
            d3d_timer,
            input_domain_reset
          )) {
        mark_d3d_post_end(d3d_timer);
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return make_result();
      }
      mark_d3d_post_end(d3d_timer);
      dispatch_pending_gpu_completion_trace();
      bool raw_snapshot_valid = false;
      bool model_input_snapshot_valid = false;
      bool coordinate_snapshot_valid = false;
      // Dump 3D binds these immutable tensors to pending_input_region. For a video ROI they are
      // bound to the logical crop-local analysis domain and its exact tensor content rectangle;
      // the package separately records the full-source embedding.
      if (snapshot_debug_inputs) {
        coordinate_snapshot_valid = dispatch_parallax_v2_coordinate_diagnostic();
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
      mark_d3d_pre_start(d3d_timer);
      end_d3d_perf(d3d_timer);
      const auto completed_frame_id = pending_frame_id;
      const auto completed_input_region = pending_input_region;
      const auto completed_color_space = pending_color_space;
      const bool completed_gpu_undecided = gpu_undecided_postprocess_pending();
      const bool completed_subtitle_work_suppressed =
        pending_optional_work == depth_optional_work_mode_e::suppress_subtitle;
      if (!completed_gpu_undecided) {
        gpu_undecided_requires_force_infer_refresh = false;
      }
      has_previous_frame = false;  // the output buffer has been consumed; never fold it twice
      clear_pending_inference_event_state();
      pending_optional_work = depth_optional_work_mode_e::ordinary;
      pending_subtitle_work = cuda_conditional_graph::work_flag_e::none;
      pending_submission_class = pending_submission_class_e::force_infer;
      pending_wrapper_transaction_token = 0u;
      pending_dump_forced = false;
      pending_gpu_transaction_token = 0u;
      pending_gpu_transaction_current_frame_id = 0u;
      pending_gpu_transaction_baseline_frame_id = 0u;
      pending_observation_timestamp_us = 0u;
      // normalize_depth_output() has submitted and unbound every D3D11 read of this exact source.
      // Drop our retained reference only after the ownership pass has consumed it.
      pending_source_srv.Reset();
      if (diagnostics_enabled) {
        throughput_stats_completions++;
      }
      return make_result(
        true,
        completed_frame_id,
        false,
        raw_snapshot_valid,
        model_input_snapshot_valid,
        coordinate_snapshot_valid,
        completed_input_region,
        completed_color_space,
        input_domain_reset,
        completed_subtitle_work_suppressed,
        false,
        false,
        completed_gpu_undecided
      );
    }

    pending_depth_poll_result try_finish_pending(
      input_color_space color_space,
      bool snapshot_debug_inputs,
      const bool allow_wait,
      const std::chrono::steady_clock::time_point deadline,
      const std::uint32_t max_queries
    ) {
      readiness_preflighted = false;
      auto &cuda = cuda_driver_api::get();
      const auto started = std::chrono::steady_clock::now();
      const bool use_inference_events = inference_event_poll_available;
      const bool repeated_wait_allowed = allow_wait && use_inference_events;
      pending_depth_poll_result polled;
      if (!has_previous_frame) {
        polled.result = make_result();
        polled.ready = true;
        return polled;
      }
      if (cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
        mark_cuda_context_failure();
        polled.result = make_result();
        polled.ready = true;
        return polled;
      }

      const auto query_limit = std::max<std::uint32_t>(1u, max_queries);
      constexpr auto repeated_query_interval = std::chrono::microseconds {50};
      while (true) {
        // The first joined query is always immediate. Before every repeated query, preserve the
        // absolute deadline even if the thread was descheduled beyond its intended poll interval.
        if (
          polled.query_count > 0u &&
          std::chrono::steady_clock::now() >= deadline
        ) {
          polled.timed_out = repeated_wait_allowed;
          polled.wait_duration = std::chrono::steady_clock::now() - started;
          return polled;
        }
        ++polled.query_count;
        const auto readiness = use_inference_events ?
                                 query_pending_inference_events(
                                   cuda,
                                   allow_wait ? "bounded same-frame completion" :
                                                "nonblocking completion"
                                 ) :
                                 query_pending_execution(
                                   cuda,
                                   "same-frame completion fallback"
                                 );
        switch (readiness) {
          case pending_execution_readiness_e::busy: {
            const auto now = std::chrono::steady_clock::now();
            if (
              !repeated_wait_allowed || polled.query_count >= query_limit ||
              now >= deadline
            ) {
              polled.timed_out = repeated_wait_allowed;
              polled.wait_duration = now - started;
              return polled;
            }
            polled.wait_attempted = true;
            // Do not burn the query fuse faster than the GPU can make useful progress. Yield until
            // the next 50 us query point, capped by the absolute deadline; no sleeping primitive
            // can overshoot the budget unnoticed because steady_clock is checked every turn.
            const auto next_query_at = std::min(
              deadline,
              now + repeated_query_interval
            );
            while (std::chrono::steady_clock::now() < next_query_at) {
              std::this_thread::yield();
            }
            continue;
          }
          case pending_execution_readiness_e::failed:
            polled.result = make_result();
            polled.ready = true;
            polled.wait_duration = std::chrono::steady_clock::now() - started;
            return polled;
          case pending_execution_readiness_e::ready:
            polled.wait_duration = std::chrono::steady_clock::now() - started;
            break;
        }
        break;
      }
      polled.result = finish_pending(
        color_space,
        snapshot_debug_inputs,
        true
      );
      polled.ready = true;
      return polled;
    }

    pending_depth_poll_result try_finish_pending_nonblocking(
      input_color_space color_space,
      bool snapshot_debug_inputs = false
    ) {
      return try_finish_pending(
        color_space,
        snapshot_debug_inputs,
        false,
        std::chrono::steady_clock::time_point {},
        1u
      );
    }

    pending_depth_poll_result try_finish_pending_until(
      input_color_space color_space,
      const std::chrono::steady_clock::time_point deadline,
      const std::uint32_t max_queries,
      bool snapshot_debug_inputs = false
    ) {
      return try_finish_pending(
        color_space,
        snapshot_debug_inputs,
        true,
        deadline,
        max_queries
      );
    }

    // (Re)build the depth constant buffer. The fixed tensor dimensions stay session-constant,
    // while transfer mode and an ROI's centered tensor-content rectangle follow the exact pending
    // frame. Domain changes are rare, so an immutable replacement keeps all dispatches simple.
    void ensure_cbuffers(
      input_color_space color_space,
      const depth_input_region_t &input_region
    ) {
      const int color_mode = (int) color_space;
      const auto tensor_content = input_region.tensor_content.valid() ?
                                    input_region.tensor_content :
                                    depth_tensor_content_rect_t {
                                      0u, 0u,
                                      static_cast<std::uint32_t>(std::max(target_w, 0)),
                                      static_cast<std::uint32_t>(std::max(target_h, 0)),
                                    };
      if (cb_color_mode == color_mode && cb_tensor_content == tensor_content && cbuffer) {
        return;
      }
      cb_color_mode = color_mode;
      cb_tensor_content = tensor_content;

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
      cb[9] = tensor_content.left;
      cb[10] = tensor_content.top;
      cb[11] = tensor_content.right;
      cb[12] = tensor_content.bottom;
      D3D11_SUBRESOURCE_DATA sd = {cb, 0, 0};
      cbuffer.Reset();
      device->CreateBuffer(&cb_desc, &sd, &cbuffer);
    }

    // Bind the physical rectangle inside the retained full matched frame separately from the
    // analysis-domain constants. All shader resize/ownership math remains crop-local; only its
    // final integer Texture2D.Load address receives this offset. Translation changes this buffer
    // without changing temporal analysis-domain identity.
    void ensure_source_region_cbuffer(const depth_input_region_t &input_region) {
      const depth_source_rect_t source_region {
        input_region.left,
        input_region.top,
        input_region.right,
        input_region.bottom,
      };
      if (source_region_cbuffer && cb_source_region == source_region) {
        return;
      }
      if (!source_region_cbuffer) {
        D3D11_BUFFER_DESC cb_desc {};
        cb_desc.Usage = D3D11_USAGE_DEFAULT;
        cb_desc.ByteWidth = 16u;
        cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (FAILED(device->CreateBuffer(
              &cb_desc, nullptr, &source_region_cbuffer
            ))) {
          return;
        }
      }
      const std::array<std::uint32_t, 4> constants {
        source_region.left,
        source_region.top,
        source_region.right,
        source_region.bottom,
      };
      context->UpdateSubresource(
        source_region_cbuffer.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
      cb_source_region = source_region;
    }

    void reset_temporal_state_for_input_domain() {
      // Domain changes are rare (video selection/extent or full-frame fallback). Reset every
      // history that could otherwise compare browser pixels with video-local pixels. All writes
      // are ordered on the owning immediate context; there is no CPU readback or synchronization.
      const std::uint32_t raw_identity[4] = {0xFFFFFFFFu, 0u, 0u, 0u};
      const float zero4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const std::uint32_t zero_uint4[4] = {0u, 0u, 0u, 0u};
      const std::array<std::uint32_t, 256> zero_hist {};
      const std::array<std::uint32_t, 10> zero_cut_evidence {};
      if (minmax_raw_buf) {
        context->UpdateSubresource(minmax_raw_buf.Get(), 0, nullptr, raw_identity, 0, 0);
      }
      if (minmax_ema_buf) {
        context->UpdateSubresource(minmax_ema_buf.Get(), 0, nullptr, zero4, 0, 0);
      }
      if (hist_buf) {
        context->UpdateSubresource(hist_buf.Get(), 0, nullptr, zero_hist.data(), 0, 0);
      }
      if (scene_cut_evidence_buf) {
        context->UpdateSubresource(
          scene_cut_evidence_buf.Get(), 0, nullptr, zero_cut_evidence.data(), 0,
          0
        );
      }
      if (cut_state_buf) {
        context->UpdateSubresource(
          cut_state_buf.Get(),
          0,
          nullptr,
          sbs_adaptive_state::initial_words.data(),
          0,
          0
        );
      }
      if (tensor_previous_input_uav) {
        context->ClearUnorderedAccessViewFloat(tensor_previous_input_uav.Get(), zero4);
      }
      if (near_identical_history_owner_uav) {
        context->ClearUnorderedAccessViewUint(
          near_identical_history_owner_uav.Get(), zero_uint4
        );
      }
      if (previous_appearance_ordinal_uav) {
        context->ClearUnorderedAccessViewFloat(previous_appearance_ordinal_uav.Get(), zero4);
      }
      if (depth_uav) {
        context->ClearUnorderedAccessViewFloat(depth_uav.Get(), zero4);
      }
      if (depth_previous_uav) {
        context->ClearUnorderedAccessViewFloat(depth_previous_uav.Get(), zero4);
      }
      if (depth_cut_history_uav) {
        context->ClearUnorderedAccessViewFloat(depth_cut_history_uav.Get(), zero4);
      }
      if (ema_motion_mask_uav) {
        context->ClearUnorderedAccessViewUint(ema_motion_mask_uav.Get(), zero_uint4);
      }
      if (tensor_previous_exclusion_uav) {
        context->ClearUnorderedAccessViewUint(
          tensor_previous_exclusion_uav.Get(), zero_uint4);
      }
      if (subtitle_locator_state_uav) {
        context->ClearUnorderedAccessViewUint(
          subtitle_locator_state_uav.Get(),
          zero_uint4
        );
      }
      if (depth_coordinate_v2_state_buf) {
        context->UpdateSubresource(
          depth_coordinate_v2_state_buf.Get(),
          0,
          nullptr,
          depth_coordinate_v2::state_initial_words.data(),
          0,
          0
        );
      }
      for (auto *uav : {
             depth_coordinate_v2_partials_uav.Get(),
             depth_coordinate_v2_frame_stats_uav.Get(),
             depth_coordinate_v2_coordinate_uav.Get(),
             depth_coordinate_v2_candidate_uav.Get(),
             depth_coordinate_v2_ownership_uav.Get(),
             depth_coordinate_v2_vertical_majorant_uav.Get(),
             depth_coordinate_v2_vertical_conditioned_uav.Get(),
             depth_coordinate_v2_final_uav.Get(),
             subtitle_conditioned_uav.Get(),
           }) {
        if (uav) {
          context->ClearUnorderedAccessViewFloat(uav, zero4);
        }
      }
      has_last_postprocessed_frame_id = false;
      last_gpu_opaque_transaction_frame_id = 0u;
    }

    bool prepare_pending_input_domain() {
      const bool changed = processed_input_domain.update(
        pending_input_region,
        pending_color_space
      );
      if (changed) {
        reset_temporal_state_for_input_domain();
      }
      return changed;
    }

    // Normalize the finished raw disparity in tensor_out_buf into depth_tex: the scale
    // passes (min/max reduction, permanent percentile histogram, EMA fold) followed by the
    // mapping/temporal-EMA pass. GPU-resident throughout, no CPU readback.
    bool normalize_depth_output(
      d3d_perf_slot *perf_slot,
      const bool input_domain_reset
    ) {
      pending_gpu_trace_append = {};
      const bool subtitle_publication_branch_opaque =
        gpu_undecided_postprocess_pending();
      if (!dispatch_near_identical_finalizer()) {
        // A conditional completion without its single receipt/argument/abstention finalizer must
        // not consume possibly stale tensor_out or mutate history. This is a mandatory active-path
        // resource failure: retain prior GPU state but terminally fail flat.
        fail_gpu_conditional_bridge_once(
          "GPU-undecided postprocess finalization failed"
        );
        return false;
      }
      // DAV2 and optional OCR are sibling nodes under one conditional CUDA root. Its completion
      // event follows both children and every interop-unmap tail, so both outputs are safe before
      // any next-frame mapping reuses the fixed buffers.
      const bool subtitle_work_suppressed =
        pending_optional_work == depth_optional_work_mode_e::suppress_subtitle;
      // Native interactive move/resize keeps the depth/cut/camera pipeline live, but OCR evidence
      // would describe a rapidly moving, deliberately de-authorized window. Leave same-domain
      // locator bytes untouched and publish the freshly produced Base field for this completion;
      // an analysis-domain transition still performed its mandatory reset above.
      bool ocr_record_submitted = false;
      auto trace_host_subtitle_outcome =
        host_sbs_gpu_trace::host_subtitle_outcome_e::suppressed;
      if (subtitle_work_suppressed) {
        pending_ocr_submitted = false;
      } else {
        ocr_record_submitted = dispatch_ocr_postprocess();
        if (ocr_record_submitted) {
          trace_host_subtitle_outcome =
            host_sbs_gpu_trace::host_subtitle_outcome_e::ordinary_record;
        }
      }
      if (!subtitle_work_suppressed && !ocr_record_submitted) {
        // Never publish a nominally exact completion whose OCR8 identity may still describe an
        // older frame. The expected-work fallback makes OCR-unavailable ordinary frames a valid
        // current abstention; reaching this branch is therefore a mandatory producer failure.
        mark_terminal_failure();
        if (!ocr_error_logged) {
          BOOST_LOG(error)
            << "Host SBS could not publish the authenticated current-frame OCR record; "
               "the V2 producer is terminal.";
          ocr_error_logged = true;
        }
        return false;
      }
      // 3a. One shared traversal produces V2 finite-value moments and the older non-negative
      // normalization reduction. Frame resolve publishes both FrameStats and MinMaxRaw before the
      // histogram consumes the latter. This remains GPU-resident and introduces no readback.
      const bool frame_stats_ready = dispatch_parallax_v2_frame_stats(perf_slot);
      if (frame_stats_ready && depth_minmax_ema_cs && minmax_raw_uav && minmax_ema_uav) {
        ID3D11ShaderResourceView *reduction_srvs[2] = {
          tensor_out_srv.Get(),
          tensor_exclusion_srv.Get(),
        };
        ID3D11ShaderResourceView *null_reduction_srvs[2] = {nullptr, nullptr};

        // Pass A2 (percentile mode): 256-bin histogram over the raw range, so pass B
        // can replace the outlier-sensitive min/max with robust percentile bounds.
        if (depth_hist_cs && hist_uav) {
          context->CSSetShader(depth_hist_cs.Get(), nullptr, 0);
          context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
          context->CSSetShaderResources(0, 2, reduction_srvs);
          ID3D11UnorderedAccessView *hist_uavs[2] = {hist_uav.Get(), minmax_raw_uav.Get()};
          context->CSSetUnorderedAccessViews(0, 2, hist_uavs, nullptr);
          dispatch_infer_postprocess(
            reduce_groups,
            1u,
            1u,
            near_identical_gpu_infer_reduce_byte_offset
          );

          ID3D11UnorderedAccessView *null_uavs_h[2] = {nullptr, nullptr};
          context->CSSetUnorderedAccessViews(0, 2, null_uavs_h, nullptr);
          context->CSSetShaderResources(0, 2, null_reduction_srvs);

        }

        // Pass B: fold into the EMA'd bounds and reset the accumulators (1 thread).
        context->CSSetShader(depth_minmax_ema_cs.Get(), nullptr, 0);
        ID3D11UnorderedAccessView *ema_uavs[4] = {
          minmax_ema_uav.Get(),
          minmax_raw_uav.Get(),
          hist_uav.Get(),
          cut_state_uav.Get(),
        };
        context->CSSetUnorderedAccessViews(0, 4, ema_uavs, nullptr);
        dispatch_infer_postprocess(
          1u, 1u, 1u, near_identical_gpu_infer_one_byte_offset
        );

        ID3D11UnorderedAccessView *null_uav2[4] = {nullptr, nullptr, nullptr, nullptr};
        context->CSSetUnorderedAccessViews(0, 4, null_uav2, nullptr);
      }

      // Rotate the two identically provisioned depth textures. The old current side becomes the
      // immutable previous-frame SRV, while the older side becomes this completion's UAV target.
      // This preserves the exact EMA history without a full-field CopyResource every frame.
      std::swap(depth_tex, depth_previous_tex);
      std::swap(depth_uav, depth_previous_uav);
      std::swap(depth_srv, depth_previous_srv);

      // 3b. Buffer to Texture: normalize disparity, classify the deterministic moving-edge
      // stencil, apply temporal EMA, and publish the diagnostic mask in one content dispatch.
      // Consuming the mask decision in-register removes the R32 mask round trip while retaining
      // the exact independently observable mask. Contain-fit padding is filled afterward by a
      // lightweight UAV copy entry point, avoiding repeated boundary-stencil work. Full-content
      // fields remain one dispatch. MinMaxEma.frame_state makes the first valid frame snap and
      // makes an all-invalid frame hold entirely on the GPU.
      context->CSSetShader(buffer_to_tex_cs.Get(), nullptr, 0);
      context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
      ID3D11ShaderResourceView *bt_srvs[4] = {
        tensor_out_srv.Get(),
        minmax_ema_srv.Get(),
        depth_previous_srv.Get(),
        tensor_exclusion_srv.Get(),
      };
      ID3D11UnorderedAccessView *bt_uavs[2] = {
        depth_uav.Get(),
        ema_motion_mask_uav.Get(),
      };
      context->CSSetShaderResources(0, 4, bt_srvs);
      context->CSSetUnorderedAccessViews(0, 2, bt_uavs, nullptr);

      dispatch_infer_postprocess(
        (target_w + 15) / 16,
        (target_h + 15) / 16,
        1u,
        near_identical_gpu_infer_grid16_byte_offset
      );

      const auto &tensor_content = pending_input_region.tensor_content;
      const bool has_synthetic_padding =
        tensor_content.left != 0u || tensor_content.top != 0u ||
        tensor_content.right != target_w || tensor_content.bottom != target_h;
      if (has_synthetic_padding) {
        // The preceding dispatch has completed its UAV accesses before this dispatch begins.
        // pad_main reads only an already-written admitted boundary texel and writes one excluded
        // destination, so no cross-thread or cross-group dependency exists inside this pass.
        context->CSSetShader(buffer_to_tex_pad_cs.Get(), nullptr, 0);
        dispatch_infer_postprocess(
          (target_w + 15) / 16,
          (target_h + 15) / 16,
          1u,
          near_identical_gpu_infer_grid16_byte_offset
        );
      }

      ID3D11UnorderedAccessView *null_uav2[2] = {nullptr, nullptr};
      ID3D11ShaderResourceView *null_srvs[4] = {
        nullptr, nullptr, nullptr, nullptr
      };
      context->CSSetUnorderedAccessViews(0, 2, null_uav2, nullptr);
      context->CSSetShaderResources(0, 4, null_srvs);
      dispatch_near_identical_reuse_depth();
      if (!dispatch_near_identical_scene_seed()) {
        fail_gpu_conditional_bridge_once(
          "GPU inference-observation scene-age publication failed"
        );
        return false;
      }

      // 3s. Analyze the freshly normalized private cut field: compact evidence + cut resolve.
      {
        // scene_seed_main has just published the exact age from the last receipt-authorized infer
        // postprocess. Reuse dispatches zero groups and leaves all cut/history state untouched.
        if (!gpu_undecided_postprocess_pending()) {
          last_postprocessed_frame_id = pending_frame_id;
          has_last_postprocessed_frame_id = true;
        }

        context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
        ID3D11ShaderResourceView *analysis_srvs[9] = {
          depth_srv.Get(),
          depth_cut_history_srv.Get(),
          tensor_in_srv.Get(),
          tensor_previous_input_srv.Get(),
          minmax_ema_srv.Get(),
          appearance_ordinal_srv.Get(),
          previous_appearance_ordinal_srv.Get(),
          tensor_exclusion_srv.Get(),
          tensor_previous_exclusion_srv.Get(),
        };
        context->CSSetShaderResources(0, 9, analysis_srvs);
        context->CSSetShader(depth_scene_cut_evidence_cs.Get(), nullptr, 0);
        context->CSSetUnorderedAccessViews(
          0,
          1,
          scene_cut_evidence_uav.GetAddressOf(),
          nullptr
        );
        dispatch_infer_postprocess(
          (target_w + 15) / 16,
          (target_h + 15) / 16,
          1u,
          near_identical_gpu_infer_grid16_byte_offset
        );

        ID3D11UnorderedAccessView *null_uavs_h2[2] = {nullptr, nullptr};
        context->CSSetUnorderedAccessViews(0, 2, null_uavs_h2, nullptr);
        ID3D11ShaderResourceView *null_analysis_srvs[9] = {
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
        context->CSSetShaderResources(0, 9, null_analysis_srvs);

        context->CSSetShader(depth_scene_cut_resolve_cs.Get(), nullptr, 0);
        ID3D11UnorderedAccessView *cut_uavs[2] = {
          cut_state_uav.Get(),
          scene_cut_evidence_uav.Get()
        };
        context->CSSetUnorderedAccessViews(0, 2, cut_uavs, nullptr);
        dispatch_infer_postprocess(
          1u, 1u, 1u, near_identical_gpu_infer_one_byte_offset
        );
        context->CSSetUnorderedAccessViews(0, 2, null_uavs_h2, nullptr);

        // tensor_in_buf, appearance_ordinal_buf and depth_tex still own the matched inputs/result
        // for this completed inference. Advance the complete appearance/depth tuple only when the
        // resolve pass selects state 1 or 3. State 2 retains the last structurally reliable tuple
        // for one black/clipped update, so an immediate supported return compares A against A or B
        // rather than the empty slate. State 3 advances an accepted persistent-low endpoint.
        context->CSSetShader(depth_valid_history_cs.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
        ID3D11ShaderResourceView *history_srvs[6] = {
          minmax_ema_srv.Get(),
          tensor_in_srv.Get(),
          appearance_ordinal_srv.Get(),
          cut_state_srv.Get(),
          depth_srv.Get(),
          tensor_exclusion_srv.Get(),
        };
        ID3D11UnorderedAccessView *history_uavs[4] = {
          tensor_previous_input_uav.Get(),
          previous_appearance_ordinal_uav.Get(),
          depth_cut_history_uav.Get(),
          tensor_previous_exclusion_uav.Get(),
        };
        context->CSSetShaderResources(0, 6, history_srvs);
        context->CSSetUnorderedAccessViews(0, 4, history_uavs, nullptr);
        dispatch_infer_postprocess(
          (target_w + 15) / 16,
          (target_h + 15) / 16,
          1u,
          near_identical_gpu_infer_grid16_byte_offset
        );
        ID3D11ShaderResourceView *null_history_srvs[6] = {
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr
        };
        ID3D11UnorderedAccessView *null_history_uavs[4] = {
          nullptr, nullptr, nullptr,
          nullptr
        };
        context->CSSetShaderResources(0, 6, null_history_srvs);
        context->CSSetUnorderedAccessViews(0, 4, null_history_uavs, nullptr);
        publish_near_identical_history_owner(
          pending_frame_id,
          pending_input_region,
          pending_color_space,
          pending_observation_timestamp_us
        );
      }

      // Production V2 runs after the shared scene-cut bridge and consumes its confirmed-cut
      // generation, with the same-frame pulse retained for attribution.
      if (parallax_v2_producer_active) {
        // Together with the earlier fused frame-stats query pair, this interval covers the same
        // V2 stats/state/geometry work as before. Both remain inside depth_postprocess_gpu, and the
        // diagnostics resolver sums the discontiguous intervals so benchmark semantics do not move.
        mark_d3d_parallax_start(perf_slot);
      }
      const bool base_ready = frame_stats_ready && dispatch_parallax_v2_producer(perf_slot);
      bool trace_condition_executed = false;
      if (base_ready) {
        bool conditioner_ready = false;
        if (!subtitle_work_suppressed) {
          conditioner_ready = dispatch_subtitle_conditioner(
            ocr_record_submitted,
            input_domain_reset
          );
          if (!conditioner_ready) {
            mark_terminal_failure();
            return false;
          }
          // For an opaque root this call submits only receipt-gated indirect work. The CPU must
          // not claim that the device executed it; the trace shader derives that from CBRG.
          trace_condition_executed = !subtitle_publication_branch_opaque;
        }
        if (!conditioner_ready) {
          // Native suppression deliberately skips the conditioner and advances no OCR/SLR state.
          // Publish the fresh immutable Base through the same complete out-of-place texture used
          // by every transaction class. Non-suppressed conditioner failure returned terminal above.
          if (subtitle_conditioned_tex && depth_coordinate_v2_final_tex) {
            context->CopyResource(
              subtitle_conditioned_tex.Get(), depth_coordinate_v2_final_tex.Get()
            );
          } else {
            mark_terminal_failure();
            return false;
          }
        }
      } else {
        mark_terminal_failure();
        return false;
      }
      if (parallax_v2_producer_active) {
        mark_d3d_parallax_end(perf_slot);
      }

      if (diagnostics_enabled) {
        pending_gpu_trace_append = {
          .host_subtitle_outcome = trace_host_subtitle_outcome,
          .ocr_record_submitted =
            !subtitle_publication_branch_opaque && ocr_record_submitted,
          .subtitle_work_suppressed = subtitle_work_suppressed,
          .condition_executed = trace_condition_executed,
          .subtitle_branch_gated =
            subtitle_publication_branch_opaque && !subtitle_work_suppressed,
          .input_domain_reset = input_domain_reset,
          .valid = true,
        };
      }

      return true;
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
                          << "% (" << throughput_stats_busy_drops << '/' << throughput_stats_calls
                           << "), force-infer roots " << throughput_stats_force_infer_enqueues
                           << ", GPU-undecided roots " << throughput_stats_gpu_undecided_enqueues
                           << ", GPU-undecided initial roots "
                           << throughput_stats_gpu_undecided_initial_enqueues
                          << ", GPU-undecided follow-up roots "
                          << throughput_stats_gpu_undecided_followup_enqueues
                          << ", optional OCR armed " << throughput_stats_ocr_armed
                          << ", native subtitle suppressions " << throughput_stats_subtitle_suppressed;
          throughput_stats_start = now;
          throughput_stats_calls = 0;
          throughput_stats_busy_drops = 0;
          throughput_stats_enqueues = 0;
          throughput_stats_force_infer_enqueues = 0;
          throughput_stats_gpu_undecided_enqueues = 0;
          throughput_stats_gpu_undecided_initial_enqueues = 0;
          throughput_stats_gpu_undecided_followup_enqueues = 0;
          throughput_stats_completions = 0;
          throughput_stats_subtitle_suppressed = 0;
          throughput_stats_ocr_armed = 0;
        }
      }
      throughput_stats_calls++;
    }

    // Query-only producer preflight. It deliberately leaves has_previous_frame and the finished
    // output buffer untouched; estimate() consumes that result after the caller has copied the
    // exact color frame that will own the next inference.
    bool can_accept() {
      if (!valid || terminal_failure || live_v2_producer_unavailable()) {
        return false;
      }
      auto &cuda = cuda_driver_api::get();
      if (!cuda.is_valid() || !cu_stream || !cuda.cuStreamQuery) {
        mark_cuda_context_failure();
        return false;
      }
      if (cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
        if (!stream_error_logged) {
          BOOST_LOG(error) << "cuCtxSetCurrent failed during depth readiness preflight.";
          stream_error_logged = true;
        }
        mark_cuda_context_failure();
        return false;
      }

      if (diagnostics_enabled) {
        update_throughput_stats();
      }
      switch (query_pending_execution(cuda, "depth readiness preflight")) {
        case pending_execution_readiness_e::busy:
          if (diagnostics_enabled) {
            throughput_stats_busy_drops++;
          }
          readiness_preflighted = false;
          return false;
        case pending_execution_readiness_e::failed:
          readiness_preflighted = false;
          return false;
        case pending_execution_readiness_e::ready:
          break;
      }
      readiness_preflighted = true;
      return true;
    }

    estimate_result estimate(
      ID3D11ShaderResourceView *input_srv,
      input_color_space color_space,
      std::uint64_t frame_id,
      bool snapshot_raw_model_depth,
      depth_input_region_t input_region,
      depth_optional_work_mode_e optional_work,
      const gpu_adaptive_reuse_request &adaptive_reuse_request
    ) {
      if (!valid || terminal_failure || live_v2_producer_unavailable() || !input_srv) {
        return {};
      }
      bool completed_frame_valid = false;
      std::uint64_t completed_frame_id = 0;
      bool raw_snapshot_valid = false;
      bool model_input_snapshot_valid = false;
      bool coordinate_snapshot_valid = false;
      depth_input_region_t completed_input_region {};
      input_color_space completed_color_space = input_color_space::srgb;
      bool completed_input_domain_reset = false;
      bool completed_subtitle_work_suppressed = false;
      bool completed_gpu_undecided = false;
      // An explicitly armed Dump 3D frame always runs ordinary exact-frame OCR/locator work.
      auto accepted_optional_work = snapshot_raw_model_depth ?
                                      depth_optional_work_mode_e::ordinary :
                                      optional_work;

      auto &cuda = cuda_driver_api::get();
      if (!cuda.is_valid()) {
        BOOST_LOG(error) << "CUDA Driver API is not available.";
        mark_cuda_context_failure();
        return {};
      }

      if (cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
        if (!stream_error_logged) {
          BOOST_LOG(error) << "cuCtxSetCurrent failed during depth estimation.";
          stream_error_logged = true;
        }
        mark_cuda_context_failure();
        return {};
      }

      // Production preflights before its expensive full-resolution color copy. The evaluator and
      // any direct callers do not, so retain the self-contained query/counting path here.
      const bool preflighted = std::exchange(readiness_preflighted, false);
      if (!preflighted && diagnostics_enabled) {
        update_throughput_stats();
      }

      // Resolve completed inference-timing events only for diagnostic runs.
      if (diagnostics_enabled) {
        perf_drain(perf_depth_conditional);
        if (terminal_failure) {
          return {};
        }
      }

      // Prevent GPU starvation: if the previous AI frame is still crunching, drop this frame.
      // This prevents an infinite queue of heavy TensorRT workloads from starving the DWM and Edge Browser.
      if (!preflighted) {
        switch (query_pending_execution(cuda, "depth estimate admission")) {
          case pending_execution_readiness_e::busy:
            // Reuse the last normalized depth and cut state while the joined exact-frame root is
            // busy. This keeps the bounded one-observation contract without exposing its branch.
            if (diagnostics_enabled) {
              throughput_stats_busy_drops++;
            }
            return make_result();
          case pending_execution_readiness_e::failed:
            return make_result();
          case pending_execution_readiness_e::ready:
            break;
        }
      }

      D3D11_TEXTURE2D_DESC input_desc = {0};
      Microsoft::WRL::ComPtr<ID3D11Resource> input_res;
      input_srv->GetResource(&input_res);
      Microsoft::WRL::ComPtr<ID3D11Texture2D> input_tex;
      if (SUCCEEDED(input_res.As(&input_tex))) {
        input_tex->GetDesc(&input_desc);
      }
      if (input_desc.Width == 0u || input_desc.Height == 0u) {
        return make_result();
      }
      input_region = resolved_input_region(input_region, input_desc);
      if (!input_region.valid()) {
        return make_result();
      }
      const auto requested_shape = models::fit_depth_tensor_shape(
        input_region.source_width,
        input_region.source_height,
        depth_short_side,
        max_aspect
      );
      const bool current_shape_matches =
        target_w == 0 || target_h == 0 ||
        (target_w == requested_shape.width && target_h == requested_shape.height);

      if (target_w == 0 || target_h == 0) {
        // The capture surface can report a 0x0 descriptor mid HDR/mode transition or
        // before the first real frame. Deriving the model resolution from that yields a
        // garbage size (NaN aspect -> integer-overflow -> clamps to the profile max) that would
        // be cached for the whole session. Wait for a valid frame instead.
        if (input_desc.Width == 0 || input_desc.Height == 0) {
          return {};
        }
        // Keep the patch-aligned tensor as close as possible to source aspect while respecting
        // the TensorRT profile, configured aspect cap, and native size.
        target_w = requested_shape.width;
        target_h = requested_shape.height;

        // Threads for the shared moments/range reduction; grid-stride handles any element count.
        int elems = target_w * target_h;
        reduce_groups = (UINT) std::min(64, std::max(1, (elems + 255) / 256));

        // A failed CUDA registration resets target_w/target_h and retries shape setup on a later
        // frame. Never let optional textures/state from that abandoned shape survive into the
        // retry (whose source aspect may already have changed).
        release_parallax_v2_resources();
        if (!unregister_near_identical_decision_interop(cuda)) {
          mark_terminal_failure(true);
          return {};
        }
        near_identical_tile_buf.Reset();
        near_identical_tile_srv.Reset();
        near_identical_tile_uav.Reset();
        near_identical_gpu_decision_buf.Reset();
        near_identical_gpu_decision_srv.Reset();
        near_identical_gpu_decision_uav.Reset();
        near_identical_gpu_dispatch_buf.Reset();
        near_identical_history_owner_buf.Reset();
        near_identical_history_owner_srv.Reset();
        near_identical_history_owner_uav.Reset();

        BOOST_LOG(info) << "Depth Estimator dynamic resolution set to " << target_w << "x" << target_h;

        const bool previous_input_unregistered =
          unregister_cuda_graphics_resource(cuda, cuda_in_res);
        const bool previous_output_unregistered =
          unregister_cuda_graphics_resource(cuda, cuda_out_res);
        if (!previous_input_unregistered || !previous_output_unregistered) {
          // A retained registration still owns the old D3 backing. Shape setup must not release
          // or overwrite it; quarantine this context and let terminal teardown retry safely.
          execution_context_poisoned = true;
          mark_terminal_failure(true);
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

        // The two ordinary depth fields rotate roles after every completed inference. Give both
        // identical views so rotation replaces a full-field copy without changing shader bindings.
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateTexture2D(&tex_desc, nullptr, &depth_previous_tex)) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(
                         depth_previous_tex.Get(),
                         nullptr,
                         &depth_previous_uav
                       )) &&
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
                       SUCCEEDED(device->CreateTexture2D(&mask_desc, nullptr, &tensor_exclusion_tex)) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(tensor_exclusion_tex.Get(), nullptr, &tensor_exclusion_uav)) &&
                       SUCCEEDED(device->CreateShaderResourceView(tensor_exclusion_tex.Get(), nullptr, &tensor_exclusion_srv)) &&
                       SUCCEEDED(device->CreateTexture2D(&mask_desc, nullptr, &tensor_previous_exclusion_tex)) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(tensor_previous_exclusion_tex.Get(), nullptr, &tensor_previous_exclusion_uav)) &&
                       SUCCEEDED(device->CreateShaderResourceView(tensor_previous_exclusion_tex.Get(), nullptr, &tensor_previous_exclusion_srv));

        if (!resources_ok) {
          BOOST_LOG(error)
            << "Depth estimator D3D11 resource creation failed; Host SBS V2 will fail flat.";
          target_w = target_h = 0;
          mark_terminal_failure();
          return {};
        }

        if (!ensure_parallax_v2_resources()) {
          mark_terminal_failure();
          return {};
        }

        // Clear depth and model-input history so the sole fused preprocess always observes
        // initialized storage, including the first force/bootstrap frame. Force mode exits before
        // reading history; this clear remains defense in depth and defines first compare state.
        const float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        context->ClearUnorderedAccessViewFloat(
          tensor_previous_input_uav.Get(), clear_color
        );
        context->ClearUnorderedAccessViewFloat(depth_uav.Get(), clear_color);
        context->ClearUnorderedAccessViewFloat(depth_previous_uav.Get(), clear_color);
        context->ClearUnorderedAccessViewFloat(depth_cut_history_uav.Get(), clear_color);
        const UINT clear_uint[4] = {0u, 0u, 0u, 0u};
        context->ClearUnorderedAccessViewUint(ema_motion_mask_uav.Get(), clear_uint);
        context->ClearUnorderedAccessViewUint(tensor_exclusion_uav.Get(), clear_uint);
        context->ClearUnorderedAccessViewUint(tensor_previous_exclusion_uav.Get(), clear_uint);

        auto res1 = cuda.cuGraphicsD3D11RegisterResource(&cuda_in_res, tensor_in_buf.Get(), 0);
        auto res2 = cuda.cuGraphicsD3D11RegisterResource(&cuda_out_res, tensor_out_buf.Get(), 0);
        if (res1 != 0 || res2 != 0) {
          BOOST_LOG(error) << "cuGraphicsD3D11RegisterResource failed: " << res1 << ", " << res2;
          const bool input_cleanup_ok =
            unregister_cuda_graphics_resource(cuda, cuda_in_res);
          const bool output_cleanup_ok =
            unregister_cuda_graphics_resource(cuda, cuda_out_res);
          if (!input_cleanup_ok || !output_cleanup_ok) {
            execution_context_poisoned = true;
          }
          target_w = target_h = 0;
          mark_terminal_failure(!input_cleanup_ok || !output_cleanup_ok);
          return {};
        }
      }

      // Shared constants for buffer_to_tex_cs, the min/max passes and rgb_to_nchw_cs.
      // Session-constant, so the buffer is built once (immutable), not mapped per frame.
      // The caller's color_space can already describe a new mode while the pending raw field and
      // retained source still belong to the previous one. Always interpret that completed pair
      // with the mode accepted alongside it.
      ensure_cbuffers(
        has_previous_frame ? pending_color_space : color_space,
        has_previous_frame ? pending_input_region : input_region
      );
      ensure_source_region_cbuffer(
        has_previous_frame ? pending_input_region : input_region
      );
      if (!cbuffer || !source_region_cbuffer) {
        mark_terminal_failure();
        return {};
      }

      auto *d3d_timer = diagnostics_enabled ?
                          begin_d3d_perf(has_previous_frame, true) :
                          nullptr;

      // tensor_out_buf holds the finished raw disparity from the previous asynchronous submit
      // (fully unmapped from CUDA), so consuming it here never blocks the encode thread. The
      // caller uses completed_frame_id to select the color slot that produced this exact result.
      if (has_previous_frame) {
        completed_gpu_undecided = gpu_undecided_postprocess_pending();
        completed_subtitle_work_suppressed =
          pending_optional_work == depth_optional_work_mode_e::suppress_subtitle;
        completed_input_domain_reset = prepare_pending_input_domain();
        if (!normalize_depth_output(
              d3d_timer,
              completed_input_domain_reset
            )) {
          mark_d3d_post_end(d3d_timer);
          mark_d3d_pre_start(d3d_timer);
          end_d3d_perf(d3d_timer);
          return {};
        }
        // The ownership dispatch has now consumed and unbound the completed frame's source SRV.
        // It is safe to release before preprocessing the newly accepted source frame.
        pending_source_srv.Reset();
        // Restore the newly supplied frame's transfer mode before its full-resolution preprocess.
        ensure_cbuffers(color_space, input_region);
        ensure_source_region_cbuffer(input_region);
        if (!cbuffer || !source_region_cbuffer) {
          mark_terminal_failure();
          return {};
        }
        // Production post-process timing ends at the normalized depth result. The two stable
        // Dump 3D copies and canonical-coordinate pass below are explicit diagnostic work and
        // must not contaminate live
        // depth_postprocess_gpu samples.
        mark_d3d_post_end(d3d_timer);
        dispatch_pending_gpu_completion_trace();
        if (snapshot_raw_model_depth && !completed_gpu_undecided) {
          // The diagnostic map belongs to the completed raw tensor, including its exact content
          // rectangle. The current-frame preprocess has not run yet, so the exclusion texture is
          // still the same completed authority as well.
          ensure_cbuffers(pending_color_space, pending_input_region);
          coordinate_snapshot_valid =
            dispatch_parallax_v2_coordinate_diagnostic();
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
          ensure_cbuffers(color_space, input_region);
          ensure_source_region_cbuffer(input_region);
          if (!cbuffer || !source_region_cbuffer) {
            mark_terminal_failure();
            return {};
          }
        }
        // Every debug copy is now ordered after this completion's consumers and before estimator
        // working resources can be reused.
        completed_frame_id = pending_frame_id;
        completed_input_region = pending_input_region;
        completed_color_space = pending_color_space;
        completed_frame_valid = true;
        has_previous_frame = false;
        clear_pending_inference_event_state();
        pending_optional_work = depth_optional_work_mode_e::ordinary;
        pending_subtitle_work = cuda_conditional_graph::work_flag_e::none;
        pending_submission_class = pending_submission_class_e::force_infer;
        pending_wrapper_transaction_token = 0u;
        pending_dump_forced = false;
        pending_gpu_transaction_token = 0u;
        pending_gpu_transaction_current_frame_id = 0u;
        pending_gpu_transaction_baseline_frame_id = 0u;
        pending_observation_timestamp_us = 0u;
        if (!completed_gpu_undecided) {
          gpu_undecided_requires_force_infer_refresh = false;
        }
        if (diagnostics_enabled) {
          throughput_stats_completions++;
        }
      } else {
        mark_d3d_post_end(d3d_timer);
      }

      // The first ROI trial deliberately keeps one session tensor shape. A validated crop whose
      // fitted shape differs is not allowed to poison the authenticated producer; consume any old
      // completion above, skip this enqueue, and let the caller use its full-frame fallback.
      if (!current_shape_matches) {
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return make_result(
          completed_frame_valid,
          completed_frame_id,
          false,
          raw_snapshot_valid,
          model_input_snapshot_valid,
          coordinate_snapshot_valid,
          completed_input_region,
          completed_color_space,
          completed_input_domain_reset,
          completed_subtitle_work_suppressed,
          false,
          false,
          completed_gpu_undecided
        );
      }

      if (!ocr_available) {
        ocr_signature_refresh_required = false;
      }

      // Every DAV2 transaction, including a CPU-known force-infer, maps the same authenticated
      // request/receipt buffer and launches the conditional wrapper. These are mandatory pipeline
      // resources: losing one is terminal rather than a reason to bypass the wrapper.
      if (!ensure_near_identical_detector_resources() ||
          !ensure_near_identical_decision_interop(cuda) ||
          !near_identical_fused_preprocess_cs ||
          !fused_preprocess_force_cbuffer ||
          !fused_preprocess_compare_cbuffer ||
          !tensor_previous_input_srv || !near_identical_tile_uav) {
        if (!terminal_failure) {
          fail_gpu_conditional_bridge_once(
            "mandatory GPU transaction resources or interop are unavailable"
          );
        }
        return {};
      }

      // The CPU may only classify a frame as GPU-undecided from host-owned facts. The wrapper is
      // prepared after the exact D3D interop pointers are mapped; no pre-existing source graph is
      // an admission prerequisite and no admitted class may bypass the wrapper.
      const bool gpu_undecided_host_candidate =
        gpu_conditional_bridge_available &&
        cuda.has_conditional_graph_support() &&
        !depth_inference_graph.policy.capture_failed &&
        adaptive_reuse_request.authorize_gpu_undecided_reuse &&
        adaptive_reuse_request.observation_timestamp_us != 0u &&
        adaptive_reuse_request.gpu_reuse_decision_token != 0u &&
        adaptive_reuse_request.baseline_frame_id != 0u &&
        gpu_undecided_baseline_authorized(adaptive_reuse_request) &&
        frame_id > adaptive_reuse_request.baseline_frame_id &&
        !gpu_undecided_requires_force_infer_refresh &&
        !ocr_signature_refresh_required &&
        depth_optional_work_allows_gpu_undecided(accepted_optional_work) &&
        !snapshot_raw_model_depth && current_shape_matches &&
        processed_input_domain.matches_analysis_domain(input_region, color_space) &&
        input_region.tensor_content.valid({target_w, target_h});
      auto accepted_submission_class =
        gpu_undecided_host_candidate ?
          pending_submission_class_e::gpu_undecided :
          pending_submission_class_e::force_infer;
      const std::uint64_t accepted_transaction_token =
        gpu_undecided_host_candidate ?
          adaptive_reuse_request.gpu_reuse_decision_token :
          next_force_infer_transaction_token();

      // Decide whether this transaction has a complete optional OCR path using host-owned
      // resource/geometry state only. GPU-undecided must publish its RQST before the fused
      // preprocess can generate image-derived tile evidence.
      const auto current_ocr_roi = fit_subtitle_analysis_geometry(
        input_region.width(),
        input_region.height(),
        {target_w, target_h},
        input_region.tensor_content
      );
      const std::uint32_t ocr_crop_height = subtitle_ocr_source_crop_height(
        input_region.width(),
        input_region.height()
      );
      const bool ocr_interop_available =
        ocr_available && ocr_exec_context && ocr_input_uav && ocr_output_srv &&
        cuda_ocr_in_res && cuda_ocr_out_res;
      bool ocr_frame_eligible =
        (accepted_optional_work == depth_optional_work_mode_e::ordinary ||
         accepted_optional_work == depth_optional_work_mode_e::ordinary_due) &&
        ocr_interop_available && ocr_preprocess_cs && ocr_preprocess_cbuffer &&
        ocr_box_cells_cs && ocr_box_resolve_cs &&
        near_identical_finalize_cs && ocr_cell_stats_srv &&
        ocr_cell_stats_uav && ocr_box_record_uav && ocr_resolve_cbuffer &&
        current_ocr_roi.valid() && ocr_crop_height != 0u &&
        ocr_crop_height <= input_region.height();
      const auto subtitle_work = subtitle_transaction_work(
        accepted_optional_work,
        ocr_frame_eligible
      );
      if (gpu_undecided_host_candidate && !prepare_near_identical_detector(
            adaptive_reuse_request,
            frame_id,
            input_region,
            color_space,
            subtitle_work
          )) {
        fail_gpu_conditional_bridge_once(
          "GPU near-identical request preparation failed"
        );
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return {};
      }

      mark_d3d_pre_start(d3d_timer);
      ID3D11ShaderResourceView *analysis_input_srv = input_srv;

      // 1. The one authenticated fused entry point is the only RGB-to-NCHW runtime producer.
      // Force/bootstrap/debug selects a uniform early exit after canonical preprocessing;
      // GPU-undecided continues into comparison and tile reduction in the same dispatch.
      context->CSSetShader(near_identical_fused_preprocess_cs.Get(), nullptr, 0u);
      ID3D11Buffer *preprocess_cbuffers[3] = {
        cbuffer.Get(),
        gpu_undecided_host_candidate ?
          fused_preprocess_compare_cbuffer.Get() :
          fused_preprocess_force_cbuffer.Get(),
        source_region_cbuffer.Get(),
      };
      context->CSSetConstantBuffers(0u, 3u, preprocess_cbuffers);
      ID3D11ShaderResourceView *preprocess_srvs[2] = {
        analysis_input_srv,
        tensor_previous_input_srv.Get(),
      };
      context->CSSetShaderResources(0, 2, preprocess_srvs);
      ID3D11UnorderedAccessView *preprocess_uavs[4] = {
        tensor_in_uav.Get(),
        appearance_ordinal_uav.Get(),
        tensor_exclusion_uav.Get(),
        near_identical_tile_uav.Get(),
      };
      context->CSSetUnorderedAccessViews(0, 4, preprocess_uavs, nullptr);
      context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

      ID3D11UnorderedAccessView *null_uavs[4] = {};
      ID3D11ShaderResourceView *null_srvs[2] = {};
      ID3D11Buffer *null_preprocess_cbuffers[3] = {};
      context->CSSetUnorderedAccessViews(0, 4, null_uavs, nullptr);
      context->CSSetShaderResources(0, 2, null_srvs);
      context->CSSetConstantBuffers(0u, 3u, null_preprocess_cbuffers);
      // GPU-undecided publishes its dense proposal. CPU-known force-infer publishes a complete
      // authenticated infer PROP/RQST instead. Both records carry the exact subtitle disposition;
      // only prepared OCR carries the optional-child request, and every completion consumes CBRG.
      const bool transaction_published =
        accepted_submission_class == pending_submission_class_e::gpu_undecided ?
          dispatch_near_identical_detector() :
          publish_force_infer_transaction(
            accepted_transaction_token,
            subtitle_work
          );
      if (!transaction_published) {
        fail_gpu_conditional_bridge_once(
          accepted_submission_class == pending_submission_class_e::gpu_undecided ?
            "GPU near-identical proposal dispatch failed" :
            "force-infer transaction publication failed"
        );
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return {};
      }

      // CUDA cannot map the raw transaction buffer while D3D owns it. Copy the proposal-authored
      // preprocess args to the D3-only indirect twin, then run the crop for authenticated OCR:
      // ordinary work on infer, or cadence-due work on either resolved depth branch.
      // Malformed/missing proposals do not consume stale input.
      context->CopyResource(
        near_identical_gpu_dispatch_buf.Get(), near_identical_gpu_decision_buf.Get()
      );
      if (ocr_frame_eligible) {
        const std::array<std::uint32_t, 8> ocr_constants {
          input_region.width(),
          input_region.height(),
          input_region.height() - ocr_crop_height,
          ocr_crop_height,
          static_cast<std::uint32_t>(color_space),
          input_region.left,
          input_region.top,
          0u,
        };
        context->UpdateSubresource(
          ocr_preprocess_cbuffer.Get(), 0, nullptr, ocr_constants.data(), 0,
          0
        );
        context->CSSetShader(ocr_preprocess_cs.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, ocr_preprocess_cbuffer.GetAddressOf());
        context->CSSetShaderResources(0, 1, &analysis_input_srv);
        context->CSSetUnorderedAccessViews(
          0, 1, ocr_input_uav.GetAddressOf(),
          nullptr
        );
        context->DispatchIndirect(
          near_identical_gpu_dispatch_buf.Get(),
          near_identical_gpu_optional_preprocess_byte_offset
        );
        context->CSSetUnorderedAccessViews(0, 1, null_uavs, nullptr);
        context->CSSetShaderResources(0, 1, null_srvs);
        ID3D11Buffer *null_cbuffer = nullptr;
        context->CSSetConstantBuffers(0, 1, &null_cbuffer);
      }

      end_d3d_perf(d3d_timer);
      // No explicit Flush: cuGraphicsMapResources() below already guarantees the
      // preceding D3D11 compute work completes before the CUDA stream reads the buffer.
      // Force-flushing every frame only prevents the driver from interleaving other GPU
      // consumers (DWM / Edge / the Widgets panel), which starves them and can trigger a TDR.

      // 2. CUDA Execution (for CURRENT frame). Depth remains mandatory. Optional OCR interop maps
      // on the same root stream and its TensorRT graph is embedded as a sibling conditional child;
      // one completion event after both unmap tails therefore covers the exact transaction.
      std::array<CUgraphicsResource, 3> depth_resources {
        cuda_in_res,
        cuda_out_res,
        cuda_near_identical_decision_res,
      };
      constexpr unsigned int depth_resource_count = 3u;
      auto map_res = cuda.cuGraphicsMapResources(
        depth_resource_count, depth_resources.data(), cu_stream
      );
      if (map_res != CUDA_SUCCESS) {
        (void) observe_joined_cuda_failure(
          "depth interop map", map_res, true
        );
        return make_result(
          completed_frame_valid,
          completed_frame_id,
          false,
          raw_snapshot_valid,
          model_input_snapshot_valid,
          coordinate_snapshot_valid,
          completed_input_region,
          completed_color_space,
          completed_input_domain_reset,
          completed_subtitle_work_suppressed,
          false,
          false,
          completed_gpu_undecided
        );
      }
      CUgraphicsResource ocr_resources[2] = {cuda_ocr_in_res, cuda_ocr_out_res};
      bool ocr_mapped = false;
      if (terminal_failure) {
        ocr_frame_eligible = false;
      }
      if (ocr_frame_eligible) {
        const auto ocr_map = cuda.cuGraphicsMapResources(
          2,
          ocr_resources,
          cu_stream
        );
        if (ocr_map == CUDA_SUCCESS) {
          ocr_mapped = true;
        } else {
          ocr_available = false;
          ocr_frame_eligible = false;
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::pre_enqueue_interop_or_binding
          );
          (void) observe_joined_cuda_failure(
            "optional OCR interop map", ocr_map, true
          );
        }
      }

      CUdeviceptr d_in = 0;
      CUdeviceptr d_out = 0;
      auto in_ptr_res = cuda.cuGraphicsResourceGetMappedPointer(
        &d_in,
        nullptr,
        cuda_in_res
      );
      auto out_ptr_res = cuda.cuGraphicsResourceGetMappedPointer(
        &d_out,
        nullptr,
        cuda_out_res
      );
      CUdeviceptr d_near_identical_decision = 0u;
      std::size_t near_identical_decision_mapped_bytes = 0u;
      const CUresult near_identical_decision_ptr_res =
        cuda.cuGraphicsResourceGetMappedPointer(
          &d_near_identical_decision,
          &near_identical_decision_mapped_bytes,
          cuda_near_identical_decision_res
        );

      CUdeviceptr d_ocr_in = 0;
      CUdeviceptr d_ocr_out = 0;
      std::size_t ocr_input_mapped_bytes = 0u;
      std::size_t ocr_output_mapped_bytes = 0u;
      CUresult ocr_in_ptr_res = CUDA_ERROR_NOT_READY;
      CUresult ocr_out_ptr_res = CUDA_ERROR_NOT_READY;
      if (ocr_mapped) {
        ocr_in_ptr_res = cuda.cuGraphicsResourceGetMappedPointer(
          &d_ocr_in, &ocr_input_mapped_bytes,
          cuda_ocr_in_res
        );
        ocr_out_ptr_res = cuda.cuGraphicsResourceGetMappedPointer(
          &d_ocr_out, &ocr_output_mapped_bytes,
          cuda_ocr_out_res
        );
      }

      const bool ocr_pointer_contract_valid =
        !ocr_mapped ||
        (ocr_in_ptr_res == CUDA_SUCCESS && ocr_out_ptr_res == CUDA_SUCCESS &&
         d_ocr_in != 0u && d_ocr_out != 0u);
      if (ocr_mapped && !ocr_pointer_contract_valid) {
        ocr_available = false;
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::pre_enqueue_interop_or_binding
        );
        const CUresult pointer_failure =
          ocr_in_ptr_res != CUDA_SUCCESS ? ocr_in_ptr_res :
          ocr_out_ptr_res != CUDA_SUCCESS ? ocr_out_ptr_res :
          static_cast<CUresult>(-1);
        (void) observe_joined_cuda_failure(
          "optional OCR mapped-pointer contract", pointer_failure, true
        );
      }

      // Admission proved the preceding observation's full stream reusable. From here on these
      // flags describe only the candidate whose interop unmaps will be issued below.
      clear_pending_inference_event_state();
      bool enqueued = false;
      // This is request authority, not merely wrapper topology: a retained superset wrapper may
      // contain the OCR sibling while suppression leaves that child dormant.
      bool ocr_armed = false;
      if (in_ptr_res != CUDA_SUCCESS || out_ptr_res != CUDA_SUCCESS || !d_in || !d_out) {
        const CUresult pointer_failure =
          in_ptr_res != CUDA_SUCCESS ? in_ptr_res :
          out_ptr_res != CUDA_SUCCESS ? out_ptr_res :
          static_cast<CUresult>(-1);
        (void) observe_joined_cuda_failure(
          "depth mapped-pointer contract", pointer_failure, true
        );
      } else if (!terminal_failure) {
        constexpr std::size_t transaction_last_byte =
          near_identical_gpu_request_record_byte_offset +
          sizeof(cuda_conditional_graph::request_record_t) - 1u;
        const bool mapped_decision_valid =
          near_identical_decision_ptr_res == CUDA_SUCCESS &&
          d_near_identical_decision != 0u &&
          near_identical_decision_mapped_bytes >=
            near_identical_gpu_decision_byte_count &&
          d_near_identical_decision <=
            std::numeric_limits<CUdeviceptr>::max() -
              transaction_last_byte &&
          ((d_near_identical_decision +
            near_identical_gpu_decision_record_byte_offset) & 15u) == 0u &&
          ((d_near_identical_decision +
            near_identical_gpu_request_record_byte_offset) & 15u) == 0u;
        if (!mapped_decision_valid) {
          if (near_identical_decision_ptr_res != CUDA_SUCCESS) {
            (void) observe_joined_cuda_failure(
              "conditional transaction mapped-pointer query",
              near_identical_decision_ptr_res,
              true
            );
          }
          fail_gpu_conditional_bridge_once(
            "mapped decision-buffer contract is invalid",
            near_identical_decision_ptr_res,
            false,
            near_identical_decision_ptr_res != CUDA_SUCCESS
          );
        }

        // Graph signatures include D3D interop addresses, which CUDA permits to change on any
        // map. A force-infer transaction may privately recapture that legal new signature. A
        // GPU-undecided transaction cannot bootstrap raw TensorRT work, so retain the old wrapper,
        // drop this candidate after unmap, and require the next transaction to be force-infer.
        const detail::cuda_graph_signature_t depth_signature {
          d_in, d_out, target_w, target_h
        };
        bool dropped_for_signature_change =
          accepted_submission_class == pending_submission_class_e::gpu_undecided &&
          !detail::cuda_graph_signature_matches(
            depth_inference_graph.policy, depth_signature
          );
        if (dropped_for_signature_change) {
          gpu_undecided_requires_force_infer_refresh = true;
          BOOST_LOG(info)
            << "DAV2 interop address changed during a GPU-undecided candidate; "
               "dropping it and rebuilding on the next force-infer transaction.";
        }
        if (!terminal_failure && !dropped_for_signature_change &&
            !select_inference_graph_signature(
              cuda,
              depth_inference_graph,
              depth_signature
            )) {
          mark_terminal_failure(true);
        }
        bool bindings_ok = !terminal_failure && !dropped_for_signature_change;
        if (trt_bound_width != target_w || trt_bound_height != target_h) {
          nvinfer1::Dims in_dims = make_input_dims(target_h, target_w);
          bindings_ok = exec_context->setInputShape("pixel_values", in_dims);
          if (bindings_ok) {
            trt_bound_width = target_w;
            trt_bound_height = target_h;
          } else {
            BOOST_LOG(error) << "TensorRT setInputShape failed for " << target_w << "x" << target_h
                             << " (outside the engine's optimization profile?)";
          }
        }
        if (bindings_ok && trt_bound_input != d_in) {
          bindings_ok = exec_context->setTensorAddress(
            "pixel_values",
            reinterpret_cast<void *>(d_in)
          );
          if (bindings_ok) {
            trt_bound_input = d_in;
          }
        }
        if (bindings_ok && trt_bound_output != d_out) {
          bindings_ok = exec_context->setTensorAddress(
            "predicted_depth",
            reinterpret_cast<void *>(d_out)
          );
          if (bindings_ok) {
            trt_bound_output = d_out;
          }
        }
        if (!bindings_ok && !dropped_for_signature_change) {
          mark_terminal_failure();
          if (!stream_error_logged) {
            BOOST_LOG(error)
              << "TensorRT shape/tensor binding failed; Host SBS V2 will fail flat.";
            stream_error_logged = true;
          }
        }
        bool ocr_bindings_ok = !terminal_failure && ocr_mapped &&
                               ocr_pointer_contract_valid && ocr_exec_context;
        constexpr std::size_t expected_ocr_input_bytes =
          static_cast<std::size_t>(3u) * ocr_engine_width * ocr_engine_height *
          sizeof(float);
        constexpr std::size_t expected_ocr_output_bytes =
          static_cast<std::size_t>(ocr_engine_width) * ocr_engine_height *
          sizeof(float);
        if (ocr_bindings_ok &&
            (ocr_input_mapped_bytes < expected_ocr_input_bytes ||
             ocr_output_mapped_bytes < expected_ocr_output_bytes)) {
          BOOST_LOG(error)
            << "PP-OCRv6 tiny interop buffer is smaller than its authenticated FP32 boundary: "
            << "input " << ocr_input_mapped_bytes << '/' << expected_ocr_input_bytes
            << ", output " << ocr_output_mapped_bytes << '/' << expected_ocr_output_bytes
            << " bytes.";
          ocr_bindings_ok = false;
        }
        const detail::cuda_graph_signature_t ocr_signature {
          d_ocr_in, d_ocr_out, ocr_engine_width, ocr_engine_height
        };
        if (
          ocr_bindings_ok &&
          accepted_submission_class == pending_submission_class_e::gpu_undecided &&
          !detail::cuda_graph_signature_matches(
            ocr_inference_graph.policy, ocr_signature
          )
        ) {
          dropped_for_signature_change = true;
          ocr_signature_refresh_required = true;
          ocr_bindings_ok = false;
          BOOST_LOG(info)
            << "PP-OCR interop address changed during a GPU-undecided candidate; "
               "dropping it and rebuilding on the next force-infer transaction.";
        }
        if (ocr_bindings_ok && !dropped_for_signature_change) {
          // TensorRT graph nodes retain the execution context's bound addresses. Tear down a
          // graph for the previous D3D mapping before mutating those bindings.
          ocr_bindings_ok = select_inference_graph_signature(
            cuda,
            ocr_inference_graph,
            ocr_signature
          );
        }
        if (ocr_bindings_ok && !ocr_shape_bound) {
          ocr_bindings_ok = ocr_exec_context->setInputShape(
            "x", make_input_dims(ocr_engine_height, ocr_engine_width)
          );
          ocr_shape_bound = ocr_bindings_ok;
        }
        if (ocr_bindings_ok && !ocr_output_size_validated) {
          const std::int64_t required_output_bytes =
            ocr_exec_context->getMaxOutputSize("fetch_name_0");
          ocr_bindings_ok = required_output_bytes > 0 &&
                            static_cast<std::uint64_t>(required_output_bytes) <=
                              expected_ocr_output_bytes;
          ocr_output_size_validated = ocr_bindings_ok;
          if (!ocr_bindings_ok) {
            BOOST_LOG(error)
              << "PP-OCRv6 tiny resolved output requires " << required_output_bytes
              << " bytes; the authenticated interop buffer provides "
              << expected_ocr_output_bytes << '.';
          }
        }
        if (ocr_bindings_ok && ocr_bound_input != d_ocr_in) {
          ocr_bindings_ok = ocr_exec_context->setTensorAddress(
            "x", reinterpret_cast<void *>(d_ocr_in)
          );
          if (ocr_bindings_ok) {
            ocr_bound_input = d_ocr_in;
          }
        }
        if (ocr_bindings_ok && ocr_bound_output != d_ocr_out) {
          ocr_bindings_ok = ocr_exec_context->setTensorAddress(
            "fetch_name_0", reinterpret_cast<void *>(d_ocr_out)
          );
          if (ocr_bindings_ok) {
            ocr_bound_output = d_ocr_out;
          }
        }
        if (ocr_mapped && !ocr_bindings_ok && !dropped_for_signature_change &&
            !terminal_failure) {
          ocr_available = false;
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::pre_enqueue_interop_or_binding
          );
          mark_terminal_failure();
          if (!ocr_error_logged) {
            BOOST_LOG(error)
              << "PP-OCRv6 tiny fixed-shape interop/tensor binding failed; the exact-frame "
                 "DAV2/OCR unit fails flat.";
            ocr_error_logged = true;
          }
        }

        if (bindings_ok && !terminal_failure && !dropped_for_signature_change) {
          // TensorRT host-side binding/capture remains serialized across estimator instances. The
          // resulting depth and optional OCR children are siblings under one authenticated root;
          // no raw OCR enqueue or CPU-visible branch result exists.
          {
            std::lock_guard<std::mutex> lock(g_trt_mutex);
            const bool allow_private_bootstrap =
              accepted_submission_class == pending_submission_class_e::force_infer;
            bool wrapper_ready = prepare_depth_inference_graph(
              cuda, allow_private_bootstrap
            );
            // Once captured, retain the OCR child in a superset wrapper even on USER32-suppressed
            // transactions. The authenticated request flag keeps its handle at
            // zero when OCR is not armed, so rebuilding the entire DAV2 root on every optional
            // mode transition is unnecessary. This is the same lifetime the wrapper already has
            // between ordinary frames: the child keeps its captured addresses while interop is
            // unmapped and can execute only after a later map validates that signature.
            CUgraph buildable_optional_child = nullptr;
            auto optional_child_policy =
              ocr_available && ocr_inference_graph.graph &&
                  !ocr_inference_graph.policy.capture_failed ?
                conditional_optional_child_policy_e::retain_if_present :
                conditional_optional_child_policy_e::disabled;
            bool optional_child_ready = false;
            if (wrapper_ready && ocr_bindings_ok) {
              wrapper_ready = prepare_ocr_inference_graph(
                cuda, allow_private_bootstrap
              );
              if (wrapper_ready && ocr_available) {
                buildable_optional_child = ocr_inference_graph.graph;
                optional_child_ready = buildable_optional_child != nullptr;
                optional_child_policy = optional_child_ready ?
                  conditional_optional_child_policy_e::build_ready :
                  conditional_optional_child_policy_e::disabled;
              }
            }
            if (wrapper_ready) {
              wrapper_ready = ensure_depth_conditional_graph(
                cuda,
                d_near_identical_decision +
                  near_identical_gpu_decision_record_byte_offset,
                d_near_identical_decision +
                  near_identical_gpu_request_record_byte_offset,
                buildable_optional_child,
                optional_child_policy
              );
            }
            const int depth_perf_slot = wrapper_ready ?
                                          perf_begin(perf_depth_conditional, cu_stream) :
                                          -1;
            if (wrapper_ready && !terminal_failure) {
              const bool launch_ocr_may_participate =
                optional_child_ready &&
                (subtitle_work == cuda_conditional_graph::work_flag_e::optional_ocr ||
                 subtitle_work == cuda_conditional_graph::work_flag_e::optional_ocr_due);
              // cuGraphLaunch failure does not prove that neither child was partially submitted.
              // Establish attempt lineage before the call so a suppressed transaction also retains
              // OCR ownership from any earlier root in this shared stream lifetime.
              joined_stream_work_ever_submitted = true;
              joined_stream_ocr_ever_submitted_or_armed =
                joined_stream_ocr_ever_submitted_or_armed || launch_ocr_may_participate;
              const CUresult launched = cuda.cuGraphLaunch(
                depth_conditional_graph.get(), cu_stream
              );
              enqueued = launched == CUDA_SUCCESS;
              if (!enqueued) {
                fail_gpu_conditional_bridge_once(
                  "conditional graph launch failed",
                  launched,
                  true,
                  true
                );
              }
              ocr_armed = enqueued && launch_ocr_may_participate;
            }
            perf_end(perf_depth_conditional, depth_perf_slot, cu_stream);
          }
          if (!enqueued && !terminal_failure) {
            mark_terminal_failure(true);
            if (!stream_error_logged) {
              BOOST_LOG(error) << "DAV2 conditional-wrapper submission failed; the execution "
                                   "context is quarantined and this estimator cannot continue.";
              stream_error_logged = true;
            }
          }
        }
      }

      CUresult ocr_unmap_res = CUDA_SUCCESS;
      if (ocr_mapped) {
        ocr_unmap_res = cuda.cuGraphicsUnmapResources(
          2,
          ocr_resources,
          cu_stream
        );
      }
      auto unmap_res = cuda.cuGraphicsUnmapResources(
        depth_resource_count, depth_resources.data(), cu_stream
      );
      if (unmap_res != CUDA_SUCCESS) {
        (void) observe_joined_cuda_failure(
          "depth interop unmap", unmap_res, true
        );
        enqueued = false;
      }
      if (ocr_unmap_res != CUDA_SUCCESS) {
        (void) observe_joined_cuda_failure(
          "optional OCR interop unmap", ocr_unmap_res, true
        );
        enqueued = false;
        ocr_armed = false;
      }

      // Record completion after every interop unmap tail. Event polling and the stream-query
      // fallback now prove the same joined depth/OCR transaction and all of its resource release.
      if (
        enqueued && !terminal_failure && inference_event_poll_available &&
        !record_inference_done_event(
          cuda,
          depth_inference_done_event,
          cu_stream,
          pending_depth_inference_event_recorded,
          "DAV2/OCR root",
          ocr_armed
        )
      ) {
        enqueued = false;
        ocr_armed = false;
      }

      if (terminal_failure) {
        enqueued = false;
        ocr_armed = false;
      }

      if (ocr_signature_refresh_satisfied(
            ocr_signature_refresh_required,
            ocr_available,
            enqueued &&
              accepted_submission_class == pending_submission_class_e::force_infer,
            accepted_optional_work,
            ocr_armed
          )) {
        ocr_signature_refresh_required = false;
      }

      has_previous_frame = enqueued;
      pending_ocr_submitted = enqueued && ocr_armed;
      if (enqueued) {
        // Retain the exact private matched-frame color SRV across asynchronous TensorRT execution.
        // The next normalize_depth_output() uses it for local, full-resolution ownership evidence
        // and releases it immediately after submitting that D3D11 pass.
        pending_source_srv = analysis_input_srv;
        pending_color_space = color_space;
        pending_frame_id = frame_id;
        pending_input_region = input_region;
        pending_optional_work = accepted_optional_work;
        pending_subtitle_work = subtitle_work;
        pending_submission_class = accepted_submission_class;
        pending_wrapper_transaction_token = accepted_transaction_token;
        pending_dump_forced = snapshot_raw_model_depth;
        pending_gpu_transaction_token =
          accepted_submission_class == pending_submission_class_e::gpu_undecided ?
            accepted_transaction_token : 0u;
        pending_gpu_transaction_current_frame_id =
          accepted_submission_class == pending_submission_class_e::gpu_undecided ?
            frame_id : 0u;
        pending_gpu_transaction_baseline_frame_id =
          accepted_submission_class == pending_submission_class_e::gpu_undecided ?
            adaptive_reuse_request.baseline_frame_id : 0u;
        pending_observation_timestamp_us = adaptive_reuse_request.observation_timestamp_us;
        last_gpu_opaque_transaction_frame_id =
          accepted_submission_class == pending_submission_class_e::gpu_undecided ?
            frame_id : 0u;
        if (diagnostics_enabled) {
          throughput_stats_enqueues++;
          if (accepted_submission_class == pending_submission_class_e::force_infer) {
            throughput_stats_force_infer_enqueues++;
          } else if (
            accepted_submission_class == pending_submission_class_e::gpu_undecided
          ) {
            throughput_stats_gpu_undecided_enqueues++;
            if (adaptive_reuse_request.opaque_followup) {
              throughput_stats_gpu_undecided_followup_enqueues++;
            } else {
              throughput_stats_gpu_undecided_initial_enqueues++;
            }
          }
          if (accepted_optional_work == depth_optional_work_mode_e::suppress_subtitle) {
            throughput_stats_subtitle_suppressed++;
          }
          if (ocr_armed) {
            // GPU-undecided roots are branch-opaque: this counts requested/available optional
            // work, never a CPU claim that the OCR child actually ran.
            throughput_stats_ocr_armed++;
          }
        }
      } else {
        pending_source_srv.Reset();
      }

      return make_result(
        completed_frame_valid,
        completed_frame_id,
        enqueued && accepted_submission_class == pending_submission_class_e::force_infer,
        raw_snapshot_valid,
        model_input_snapshot_valid,
        coordinate_snapshot_valid,
        completed_input_region,
        completed_color_space,
        completed_input_domain_reset,
        completed_subtitle_work_suppressed,
        enqueued && ocr_armed &&
          (accepted_optional_work == depth_optional_work_mode_e::ordinary ||
           accepted_optional_work == depth_optional_work_mode_e::ordinary_due),
        enqueued &&
          accepted_submission_class == pending_submission_class_e::gpu_undecided,
        completed_gpu_undecided
      );
    }
  };

  video_depth_estimator::video_depth_estimator(
    Microsoft::WRL::ComPtr<ID3D11Device> device,
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
    const std::filesystem::path &assets_dir,
    const config::video_t::sbs_t &cfg,
    const config::depth_model_info &model
  ):
      pimpl(std::make_unique<impl>(
        device,
        context,
        assets_dir,
        cfg,
        model
      )) {}

  video_depth_estimator::~video_depth_estimator() = default;

  bool video_depth_estimator::is_valid() const {
    return pimpl && pimpl->valid;
  }

  bool video_depth_estimator::can_accept_frame() {
    return pimpl && pimpl->can_accept();
  }

  bool video_depth_estimator::has_terminal_failure() const {
    return !pimpl || !pimpl->valid || pimpl->terminal_failure ||
           pimpl->live_v2_producer_unavailable();
  }

  estimate_result video_depth_estimator::estimate_depth(
    ID3D11ShaderResourceView *input_srv,
    input_color_space color_space,
    std::uint64_t frame_id,
    bool snapshot_debug_inputs,
    depth_input_region_t input_region,
    depth_optional_work_mode_e optional_work,
    gpu_adaptive_reuse_request adaptive_reuse
  ) {
    return pimpl->estimate(
      input_srv,
      color_space,
      frame_id,
      snapshot_debug_inputs,
      input_region,
      optional_work,
      adaptive_reuse
    );
  }

  estimate_result video_depth_estimator::finish_pending_depth_for_evaluation(input_color_space color_space) {
    return pimpl->finish_pending(color_space);
  }

  pending_depth_poll_result video_depth_estimator::try_finish_pending_depth_nonblocking(
    input_color_space color_space,
    bool snapshot_debug_inputs
  ) {
    return pimpl ? pimpl->try_finish_pending_nonblocking(
                     color_space,
                     snapshot_debug_inputs
                   ) :
                   pending_depth_poll_result {.ready = true};
  }

  pending_depth_poll_result video_depth_estimator::try_finish_pending_depth_until(
    input_color_space color_space,
    const std::chrono::steady_clock::time_point deadline,
    const std::uint32_t max_queries,
    bool snapshot_debug_inputs
  ) {
    return pimpl ? pimpl->try_finish_pending_until(
                     color_space,
                     deadline,
                     max_queries,
                     snapshot_debug_inputs
                   ) :
                   pending_depth_poll_result {.ready = true};
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
