/**
 * @file src/tensorrt_engine_pool.cpp
 * @brief Process-wide TensorRT runtime, resident engines, pooled contexts and startup preparation.
 */
#include "tensorrt_engine_pool.h"

#include "cuda_engine_identity.h"
#include "depth_coordinate_v2.h"
#include "host_sbs_shader_cache.h"
#include "live_model_preparation.h"
#include "logging.h"
#include "model_manager.h"
#include "platform/windows/misc.h"
#include "utility.h"

#ifdef SUNSHINE_TESTS
  #include "live_model_preparation_test.h"
#endif

#include <algorithm>
#include <cstdint>
#include <dxgi.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <windows.h>
#include <wrl/client.h>

using namespace std::literals;

namespace models::tensorrt_engine_pool {
  void Logger::log(Severity severity, const char *msg) noexcept {
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

  Logger gLogger;

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
  std::mutex g_trt_mutex;
  std::condition_variable g_trt_context_available;
  std::mutex g_tensorrt_compile_mutex;
  nvinfer1::IRuntime *g_runtime = nullptr;
  static std::once_flag g_cuda_init_once;
  static CUresult g_cuda_init_result = CUDA_ERROR_NOT_READY;
  static std::mutex g_cuda_context_mutex;
  static std::map<CUdevice, CUcontext> g_cuda_primary_contexts;

  bool ensure_cuda_initialized(cuda_driver_api &cuda) {
    std::call_once(g_cuda_init_once, [&cuda]() {
      g_cuda_init_result = cuda.cuInit(0);
    });
    return g_cuda_init_result == CUDA_SUCCESS;
  }

  // Retain each primary context once for the process lifetime. TensorRT engines/contexts are also
  // process-resident, so releasing it from an estimator destructor would invalidate pooled state.
  CUcontext primary_context(cuda_driver_api &cuda, CUdevice device) {
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

  bool unregister_cuda_graphics_resource(
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

  bool cuda_device_for_d3d(cuda_driver_api &cuda, ID3D11Device *d3d, CUdevice &out) {
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
    // CUDA device 0 may be another card: textures of this adapter could not be shared with it.
    BOOST_LOG(error) << "Could not map the D3D11 adapter to a CUDA device; Host 3D needs the NVIDIA GPU that captures and encodes.";
    return false;
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
  // explicitly enabled, the reported device properties on which tactics were selected. The filename
  // hashes the complete identity. Never substitute a device ordinal or sentinel after a failed query.
  std::string engine_compatibility_tag(cuda_driver_api &cuda, CUdevice device) {
    // Match the process-lifetime CUDA primary context and engine slots. In particular a display
    // topology change must not trigger another engine build during an established host process.
    static std::mutex identity_mutex;
    static std::map<CUdevice, std::string> identities;
    std::lock_guard identity_lock(identity_mutex);
    if (const auto found = identities.find(device); found != identities.end()) {
      return found->second;
    }
    std::string identity_error;
    const auto device_identity = models::detail::cuda_engine_device_identity(cuda, device, identity_error);
    if (device_identity.empty()) {
      BOOST_LOG(error) << "Cannot select a TensorRT cache identity for CUDA device " << device
                       << ": " << identity_error << ". Refusing cache reuse and compilation.";
      return {};
    }
    const auto identity = "trt" + std::to_string(NV_TENSORRT_MAJOR) + '_' +
           std::to_string(NV_TENSORRT_MINOR) + '_' + std::to_string(NV_TENSORRT_PATCH) + '_' +
           std::to_string(NV_TENSORRT_BUILD) + '|' + device_identity;
    identities.emplace(device, identity);
    return identity;
  }

  std::map<std::string, engine_slot> g_engines;  // guarded by g_trt_mutex

  std::size_t allocated_context_count(const engine_slot &slot) {
    return slot.context_accounting.allocated();
  }

  // TensorRT 11.2 does not permit one optimization profile to be used concurrently by two
  // execution contexts. The fused engine's profiles are distinct shapes, so it has no duplicate
  // profile available for transition concurrency and must remain a one-physical-context lifetime.
  // Invalid is used by OCR slots and deliberately retains the established standard pool cap.
  std::size_t slot_context_limit(const engine_slot &slot) {
    const bool fused =
      slot.io_kind == models::prod_zipdepth_convex2x::engine_io_e::
                        production_dav2_zipdepth_convex2x;
    return models::detail::tensorrt_context_limit(fused);
  }

  // The object is deliberately leaked because destroying TensorRT interfaces across this compiler
  // boundary corrupts the heap. Removing it from usable accounting prevents a later session from
  // treating a context that never completed warmup, or later suffered an asynchronous execution
  // failure, as reusable; quarantined accounting keeps repeated failures bounded by
  // the slot-specific physical context limit.
  void quarantine_execution_context_locked(
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

  void mark_execution_context_warmed_locked(const std::string &engine_key) {
    g_engines[engine_key].context_accounting.mark_warmed();
  }

  void return_execution_context_locked(
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

  void release_context_reservation_locked(const std::string &engine_key) {
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
    const std::size_t context_limit = slot_context_limit(slot);
    if (allocated_context_count(slot) >= context_limit) {
      return slot.context_accounting.warmed() != 0u ?
               startup_context_reservation_e::resident_warmed :
               startup_context_reservation_e::unavailable;
    }
    slot.context_accounting.reserve(context_limit);
    return startup_context_reservation_e::create;
  }

  void erase_empty_engine_slot_locked(const std::string &engine_key) {
    const auto found = g_engines.find(engine_key);
    if (found != g_engines.end() && !found->second.engine &&
        allocated_context_count(found->second) == 0u &&
        found->second.context_pool.empty()) {
      g_engines.erase(found);
    }
  }

  void recycle_or_quarantine_execution_context_locked(
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

  // Build the DA-V2 input dimensions [1,3,H,W]. Passing Dims INTO TensorRT is ABI-safe (only
  // RETURNING a Dims by value across the MinGW/MSVC boundary faults).
  nvinfer1::Dims make_input_dims(int h, int w) {
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
  nvinfer1::ICudaEngine *acquire_engine_locked(
    const std::string &engine_key,
    const std::filesystem::path &engine_path,
    nvinfer1::IExecutionContext *&out_context,
    bool *out_pooled
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
  bool validate_engine_io_locked(nvinfer1::ICudaEngine *engine, engine_slot &slot) {
    if (!engine) {
      return false;
    }
    if (slot.io_validated) {
      return slot.io_compatible;
    }

    slot.io_validated = true;
    slot.io_kind = models::prod_zipdepth_convex2x::engine_io_e::invalid;
    struct observed_tensor_t {
      bool present = false;
      bool fp32 = false;
      bool mode_ok = false;
    };
    observed_tensor_t input;
    observed_tensor_t refined_output;
    const auto observe = [](
                           observed_tensor_t &tensor,
                           const nvinfer1::DataType type,
                           const nvinfer1::TensorIOMode mode,
                           const nvinfer1::TensorIOMode expected_mode
                         ) {
      // Tensor names are required to be unique. Treat a duplicate defensively as an invalid mode
      // observation even though TensorRT normally rejects it while parsing/deserializing.
      const bool first = !tensor.present;
      tensor.present = true;
      tensor.fp32 = first && type == nvinfer1::DataType::kFLOAT;
      tensor.mode_ok = first && mode == expected_mode;
    };
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
      const char *name = engine->getIOTensorName(i);
      if (!name) {
        BOOST_LOG(error) << "TensorRT returned a null I/O tensor name; rejecting the engine.";
        continue;
      }
      const auto type = engine->getTensorDataType(name);
      const auto mode = engine->getTensorIOMode(name);
      const bool is_input = mode == nvinfer1::TensorIOMode::kINPUT;
      BOOST_LOG(info) << "Depth engine tensor '" << name << "' " << (is_input ? "(input)" : "(output)")
                      << " dtype=" << tensor_dtype_name(type);
      const std::string_view tensor_name {name};
      if (tensor_name == models::prod_zipdepth_convex2x::input) {
        observe(input, type, mode, nvinfer1::TensorIOMode::kINPUT);
      } else if (tensor_name == models::prod_zipdepth_convex2x::refined_output) {
        observe(refined_output, type, mode, nvinfer1::TensorIOMode::kOUTPUT);
      }
    }
    slot.io_kind = models::prod_zipdepth_convex2x::classify_named_io(
      static_cast<std::uint32_t>(engine->getNbIOTensors()),
      input.present,
      refined_output.present
    );
    const bool required_properties_ok =
      input.fp32 && input.mode_ok && refined_output.fp32 && refined_output.mode_ok;
    const bool profiles_ok =
      engine->getNbOptimizationProfiles() ==
        static_cast<int>(models::prod_zipdepth_convex2x::fixed_profile_shapes.size());
    slot.io_compatible =
      slot.io_kind == models::prod_zipdepth_convex2x::engine_io_e::
                        production_dav2_zipdepth_convex2x &&
      required_properties_ok && profiles_ok;
    if (!slot.io_compatible) {
      BOOST_LOG(error)
        << "Depth engine must expose the exact fused single-high two-tensor FP32 contract with "
           "the authenticated fixed profiles; rejecting the engine.";
    }
    return slot.io_compatible;
  }

  // Validate the fixed PP-OCRv6 tiny boundary without calling any TensorRT method that returns
  // Dims by value (unsafe across the MinGW/MSVC ABI boundary on Windows).
  bool validate_ocr_engine_io_locked(nvinfer1::ICudaEngine *engine, engine_slot &slot) {
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
  bool detach_incompatible_engine_locked(const std::string &engine_key) {
    auto found = g_engines.find(engine_key);
    if (found == g_engines.end()) {
      return true;
    }
    auto &slot = found->second;
    if (!slot.engine) {
      slot.io_validated = false;
      slot.io_compatible = false;
      slot.io_kind = models::prod_zipdepth_convex2x::engine_io_e::invalid;
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
    slot.io_kind = models::prod_zipdepth_convex2x::engine_io_e::invalid;
    g_trt_context_available.notify_all();
    return true;
  }

  bool warmup_execution_context(
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

    const int h = static_cast<int>(
      models::prod_zipdepth_convex2x::fixed_profile_shapes[0].height
    );
    const int w = static_cast<int>(
      models::prod_zipdepth_convex2x::fixed_profile_shapes[0].width
    );
    const size_t in_elems = (size_t) 3 * h * w;
    const size_t out_elems = (size_t) h * w;
    const std::size_t output_logical_bytes = out_elems * sizeof(float);
    const auto aligned_output_bytes =
      models::detail::checked_tensorrt_output_allocation_bytes(output_logical_bytes);
    if (!aligned_output_bytes) {
      return false;
    }
    const std::size_t output_capacity = *aligned_output_bytes;
    CUdeviceptr d_in = 0;
    CUdeviceptr d_out = 0;
    if (cuda.cuMemAlloc(&d_in, in_elems * sizeof(float)) != CUDA_SUCCESS) {
      return false;
    }
    auto free_input = util::fail_guard([&]() {
      cuda.cuMemFree(d_in);
    });
    if (cuda.cuMemAlloc(&d_out, output_capacity) != CUDA_SUCCESS) {
      return false;
    }
    auto free_output = util::fail_guard([&]() {
      cuda.cuMemFree(d_out);
    });

    const auto input_dims = make_input_dims(h, w);
    const bool profile_selection_attempted = true;
    const bool profile_selected = exec_context->setOptimizationProfileAsync(0, stream);
    const bool shape_attempted = profile_selected;
    const bool shape_set = shape_attempted &&
      exec_context->setInputShape(
        models::prod_zipdepth_convex2x::input.data(), input_dims
      );
    const bool output_bounds_attempted = shape_set;
    std::int64_t output_required = -1;
    if (output_bounds_attempted) {
      output_required = exec_context->getMaxOutputSize(
        models::prod_zipdepth_convex2x::refined_output.data()
      );
    }
    const bool output_bounds_valid = output_bounds_attempted && output_required > 0 &&
      static_cast<std::uint64_t>(output_required) <= output_capacity;
    const bool address_binding_attempted = shape_set && output_bounds_valid;
    const bool input_address_bound = address_binding_attempted &&
      exec_context->setTensorAddress(
        models::prod_zipdepth_convex2x::input.data(),
        reinterpret_cast<void *>(d_in)
      );
    const bool output_address_bound = address_binding_attempted &&
      exec_context->setTensorAddress(
        models::prod_zipdepth_convex2x::refined_output.data(),
        reinterpret_cast<void *>(d_out)
      );
    const bool bound = address_binding_attempted &&
      input_address_bound && output_address_bound;
    bool enqueued = false;
    const bool enqueue_attempted = bound;
    if (bound) {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      enqueued = exec_context->enqueueV3(stream);
    }
    CUresult synchronize_result = CUDA_SUCCESS;
    const bool asynchronous_work_attempted =
      profile_selection_attempted || enqueue_attempted;
    if (asynchronous_work_attempted) {
      synchronize_result = cuda.cuStreamSynchronize ?
                             cuda.cuStreamSynchronize(stream) :
                             static_cast<CUresult>(-1);
    }
    const bool stream_quiesced = models::detail::asynchronous_operands_may_be_released(
      asynchronous_work_attempted,
      cuda.cuStreamSynchronize != nullptr,
      synchronize_result == CUDA_SUCCESS ?
        models::detail::async_stream_readiness_e::ready :
        models::detail::async_stream_readiness_e::failed
    );
    if (!stream_quiesced) {
      // enqueueV3(false) can still mean partial asynchronous submission, and a failed synchronize
      // may itself be reporting that launch. Without a positive quiescence proof, freeing any
      // startup operand or destroying the stream can race live driver work. The caller quarantines
      // TensorRT context; retain these small startup allocations with it deliberately.
      free_output.disable();
      free_input.disable();
      destroy_stream.disable();
      BOOST_LOG(warning)
        << "Depth startup warmup could not prove stream quiescence (enqueue=" << enqueued
        << ", profile=" << profile_selected
        << ", synchronize=" << synchronize_result
        << "); retaining its CUDA operands with the quarantined context.";
    }
    const bool synchronized = enqueued && stream_quiesced;
    if (!synchronized) {
      BOOST_LOG(error)
        << "Fused DAV2 + convex2x startup warmup diagnostics: profile_attempted="
        << profile_selection_attempted << " profile_selected=" << profile_selected
        << " shape_attempted=" << shape_attempted
        << " shape_set=" << shape_set
        << " output_bounds_attempted=" << output_bounds_attempted
        << " output_required/capacity=" << output_required << '/' << output_capacity
        << " output_bounds_valid=" << output_bounds_valid
        << " address_binding_attempted=" << address_binding_attempted
        << " input_address=" << input_address_bound
        << " output_address=" << output_address_bound
        << " enqueue_attempted=" << enqueue_attempted << " enqueued=" << enqueued
        << " synchronize=" << synchronize_result
        << " stream_quiesced=" << stream_quiesced << '.';
    }
    BOOST_LOG(info) << "Fused DAV2 + convex2x depth model startup warmup complete ("
                    << w << 'x' << h
                    << (synchronized ? ")." : "); execution failed.");
    return synchronized;
  }

  bool warmup_ocr_execution_context(
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

  /** Require the local fused graph without changing the public Host-SBS DAV2 model identity. */
  bool resolve_depth_runtime(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &raw_model,
    resolved_depth_runtime_t &resolved
  ) {
    static_assert(depth_coordinate_v2::model_calibrations.size() == 1u);
    const auto &embedded_dav2 = depth_coordinate_v2::model_calibrations.front();
    if (raw_model.name != embedded_dav2.depth_model ||
        raw_model.url != embedded_dav2.depth_model_url) {
      BOOST_LOG(error)
        << "The fused Host SBS runtime requires the canonical production DAV2 identity; got '"
        << raw_model.name << "'.";
      return false;
    }
    const auto fused_path = assets_dir /
      (std::string {prod_zipdepth_convex2x::logical_model} + ".onnx");
    std::error_code status_error;
    const auto status = std::filesystem::symlink_status(fused_path, status_error);
    if (status_error) {
      BOOST_LOG(error) << "Could not inspect required fused DAV2 + ZipDepth asset "
                       << fused_path << ": " << status_error.message() << '.';
      return false;
    }
    const bool present = std::filesystem::exists(status);
    const bool regular_file = present && std::filesystem::is_regular_file(status);
    const std::string sha256 = regular_file ? file_sha256_hex(fused_path) : std::string {};
    if (!detail::composite_depth_asset_is_authenticated(
      present,
      regular_file,
      sha256 == prod_zipdepth_convex2x::fused_onnx_sha256
    )) {
      BOOST_LOG(error)
        << "Required fused DAV2 + ZipDepth asset is missing or failed its frozen identity at "
        << fused_path
        << " (expected SHA-256 " << prod_zipdepth_convex2x::fused_onnx_sha256
        << ", got " << (sha256.empty() ? "unreadable/non-regular" : sha256)
        << "); Host SBS will fail flat.";
      return false;
    }
    resolved.model = {
      .name = std::string {prod_zipdepth_convex2x::logical_model},
      .url = {},
    };
    BOOST_LOG(info) << "Selected authenticated local fused DAV2 + ZipDepth runtime "
                    << fused_path.filename() << '.';
    return true;
  }

  bool publish_serialized_engine(
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

  bool publish_active_engine_manifest(
    const std::filesystem::path &assets_dir,
    const std::string_view model_name,
    const engine_artifact &artifact,
    const std::optional<prod_zipdepth_convex2x::engine_io_e> depth_io_kind
  ) {
    if (artifact.name.empty() || artifact.source_sha256.empty()) {
      return false;
    }

    const auto path = assets_dir / (std::string(model_name) + ".active-engine.json");
    auto temporary_path = path;
    temporary_path += ".tmp";
    nlohmann::json manifest {
      {"schema", 1},
      {"model", std::string(model_name)},
      {"engine", artifact.name},
      {"onnx_sha256", artifact.source_sha256},
    };
    if (depth_io_kind == prod_zipdepth_convex2x::engine_io_e::
                           production_dav2_zipdepth_convex2x) {
      const auto &raw = depth_coordinate_v2::model_calibrations.front();
      manifest = {
        {"schema", 2},
        {"runtime", "dav2_zipdepth_convex2x_composite"},
        {"model", std::string(model_name)},
        {"engine", artifact.name},
        {"engine_recipe", std::string {prod_zipdepth_convex2x::engine_recipe}},
        {"composite_onnx_sha256", artifact.source_sha256},
        {"embedded_dav2_onnx_sha256",
         std::string {prod_zipdepth_convex2x::dav2_onnx_sha256}},
        {"zipdepth_checkpoint_sha256",
         std::string {prod_zipdepth_convex2x::zipdepth_checkpoint_sha256}},
        {"guidance_preprocess_source_closure_sha256",
         std::string {raw.preprocess.source_closure_sha256}},
      };
    }

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
  bool ensure_tensorrt_engine_for_device(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model,
    cuda_driver_api &cuda,
    CUdevice cuda_device,
    engine_artifact &artifact
  ) {
    std::lock_guard<std::mutex> lock(g_tensorrt_compile_mutex);

    artifact.source_path = ensure_onnx_available(assets_dir, model.name);
    if (artifact.source_path.empty()) {
      BOOST_LOG(warning) << "ONNX source not found. TensorRT compilation aborted.";
      return false;
    }
    artifact.source_sha256 = file_sha256_hex(artifact.source_path);
    if (artifact.source_sha256.empty()) {
      BOOST_LOG(error) << "Could not hash depth-model source " << artifact.source_path << '.';
      return false;
    }
    if (CUcontext ctx = primary_context(cuda, cuda_device)) {
      if (cuda.cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
        BOOST_LOG(error) << "Failed to select the configured CUDA device for TensorRT engine compilation.";
        return false;
      }
    } else {
      BOOST_LOG(error) << "Failed to retain the configured CUDA device for TensorRT engine compilation.";
      return false;
    }

    // Resolve the cache entry before creating a builder. Its name is derived from the ONNX
    // digest and the device, so a hit needs no parse of the model; every estimator
    // construction reaches this under g_tensorrt_compile_mutex.
    if (model.name != models::prod_zipdepth_convex2x::logical_model ||
        artifact.source_sha256 != models::prod_zipdepth_convex2x::fused_onnx_sha256) {
      BOOST_LOG(error)
        << "Fused DAV2 + ZipDepth ONNX identity does not match the frozen production "
           "contract; refusing to build it.";
      return false;
    }

    const auto compatibility_tag = engine_compatibility_tag(cuda, cuda_device);
    if (compatibility_tag.empty()) {
      return false;
    }
    artifact.name = engine_filename(model, compatibility_tag + "-onnx" + artifact.source_sha256);
    if (artifact.name.empty()) {
      BOOST_LOG(error) << "No TensorRT cache recipe exists for the classified depth ONNX.";
      return false;
    }
    artifact.engine_path = assets_dir / artifact.name;

    if (!engine_cache_path_fits_native_limit(artifact.engine_path)) {
      BOOST_LOG(error) << "TensorRT cache path including .part exceeds the native file limit: "
                       << artifact.engine_path << ". Choose a shorter model assets directory.";
      return false;
    }

    std::error_code existing_ec;
    if (std::filesystem::is_regular_file(artifact.engine_path, existing_ec)) {
      BOOST_LOG(info) << "TensorRT engine cache hit: " << artifact.engine_path.filename();
      return true;
    }

    initLibNvInferPlugins(&gLogger, "");
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
      BOOST_LOG(error) << "TensorRT failed to create an engine builder.";
      return false;
    }
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) {
      BOOST_LOG(error) << "TensorRT failed to create the network.";
      return false;
    }

    auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
      BOOST_LOG(error) << "TensorRT failed to create the ONNX parser.";
      return false;
    }
    if (!parser->parseFromFile(artifact.source_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      BOOST_LOG(error) << "Failed to parse ONNX file.";
      return false;
    }

    nvinfer1::ITensor *input_tensor = nullptr;
    for (int index = 0; index < network->getNbInputs(); ++index) {
      auto *tensor = network->getInput(index);
      if (!tensor || !tensor->getName()) {
        BOOST_LOG(error)
          << "TensorRT returned a null input tensor while classifying the depth ONNX.";
        return false;
      }
      const std::string_view name {tensor->getName()};
      if (name == models::prod_zipdepth_convex2x::input && !input_tensor) {
        input_tensor = tensor;
      } else {
        BOOST_LOG(error) << "Unsupported depth-model input tensor '" << name
                         << "'; refusing an ambiguous engine contract.";
        return false;
      }
    }

    if (network->getNbInputs() != 1 || !input_tensor ||
        input_tensor->getType() != nvinfer1::DataType::kFLOAT) {
      BOOST_LOG(error)
        << "Depth ONNX must expose exactly one FP32 input 'pixel_values'.";
      return false;
    }

    auto *refined_output_tensor = network->getNbOutputs() == 1 ? network->getOutput(0) : nullptr;
    if (!refined_output_tensor || !refined_output_tensor->getName() ||
        std::string_view {refined_output_tensor->getName()} !=
          models::prod_zipdepth_convex2x::refined_output ||
        refined_output_tensor->getType() != nvinfer1::DataType::kFLOAT) {
      BOOST_LOG(error)
        << "Depth ONNX must expose exactly one FP32 output 'refined_depth'.";
      return false;
    }

    BOOST_LOG(info)
      << "Building fused DAV2 + convex2x TensorRT engine from ONNX... This will take a few minutes.";

    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
      BOOST_LOG(error) << "TensorRT failed to create the builder configuration.";
      return false;
    }
    // Set memory limit to 4GB.
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4ULL << 30);
    // Changing tactic search must produce a new cache recipe rather than silently consuming a
    // plan selected under another level.
    config->setBuilderOptimizationLevel(depth_engine_builder_level);
    BOOST_LOG(info) << "TensorRT builder optimization level "
                    << depth_engine_builder_level << '.';

    const auto set_point_dimensions = [](
                                        nvinfer1::IOptimizationProfile *profile,
                                        const char *name,
                                        const nvinfer1::Dims &dimensions
                                      ) {
      return profile->setDimensions(
               name, nvinfer1::OptProfileSelector::kMIN, dimensions
             ) &&
             profile->setDimensions(
               name, nvinfer1::OptProfileSelector::kOPT, dimensions
             ) &&
             profile->setDimensions(
               name, nvinfer1::OptProfileSelector::kMAX, dimensions
             );
    };
    for (const auto shape : models::prod_zipdepth_convex2x::fixed_profile_shapes) {
      auto *profile = builder->createOptimizationProfile();
      const auto high_dimensions = make_input_dims(
        static_cast<int>(shape.height), static_cast<int>(shape.width)
      );
      const bool profile_ok = profile &&
        set_point_dimensions(profile, input_tensor->getName(), high_dimensions);
      if (!profile_ok || config->addOptimizationProfile(profile) < 0) {
        BOOST_LOG(error)
          << "TensorRT rejected fused convex2x point profile "
          << shape.width << 'x' << shape.height << '.';
        return false;
      }
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

  bool ensure_ocr_tensorrt_engine_for_device(
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
    const auto compatibility_tag = engine_compatibility_tag(cuda, cuda_device);
    if (compatibility_tag.empty()) {
      return false;
    }
    artifact.name = ocr_engine_filename(compatibility_tag + "-onnx" + artifact.source_sha256);
    if (artifact.name.empty()) {
      BOOST_LOG(error)
        << "Could not derive the bounded PP-OCRv6 tiny TensorRT cache identity; "
           "subtitle conditioning stays flat.";
      return false;
    }
    artifact.engine_path = assets_dir / artifact.name;

    if (!engine_cache_path_fits_native_limit(artifact.engine_path)) {
      BOOST_LOG(error) << "PP-OCRv6 cache path including .part exceeds the native file limit: "
                       << artifact.engine_path << ". Choose a shorter model assets directory.";
      return false;
    }

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
}  // namespace models::tensorrt_engine_pool

namespace models {
  using namespace tensorrt_engine_pool;

  void detail::begin_live_model_preparation(const config::depth_model_info &model) {
    set_model_prepare_status(engine_filename(model), engine_build_status::building);
  }

  void detail::fail_live_model_preparation(const config::depth_model_info &model) {
    set_model_prepare_status(engine_filename(model), engine_build_status::failed);
  }

#ifdef SUNSHINE_TESTS
  void testing::publish_model_preparation_status_for_test(const config::depth_model_info &model, engine_build_status status) {
    set_model_prepare_status(engine_filename(model), status);
  }
#endif
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
#ifdef SUNSHINE_TESTS
    testing::observe_preparation_stage_for_test(testing::preparation_stage_e::depth_building);
#endif
    auto failed = util::fail_guard([&]() {
      set_model_prepare_status(status_key, engine_build_status::failed);
    });

    resolved_depth_runtime_t depth_runtime;
    if (!resolve_depth_runtime(assets_dir, model, depth_runtime)) {
      return false;
    }
    const auto &runtime_model = depth_runtime.model;

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
    if (!ensure_tensorrt_engine_for_device(
          assets_dir, runtime_model, cuda, cuda_device, artifact
        )) {
      return false;
    }
    auto engine_path = artifact.engine_path;
    auto engine_key = std::to_string(cuda_device) + ":" + artifact.name;

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *exec_context = nullptr;
    auto engine_io_kind =
      models::prod_zipdepth_convex2x::engine_io_e::invalid;
    bool create_context = false;
    bool resident_warmed_context = false;
    bool engine_repair_allowed = true;
    {
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      engine = acquire_engine_locked(engine_key, engine_path, exec_context);
      auto &slot = g_engines[engine_key];
      if (engine) {
        if (!validate_engine_io_locked(engine, slot)) {
          return_execution_context_locked(engine_key, exec_context);
          engine_repair_allowed = detach_incompatible_engine_locked(engine_key);
          engine = nullptr;
        } else {
          engine_io_kind = slot.io_kind;
          if (engine_io_kind != prod_zipdepth_convex2x::engine_io_e::
                                  production_dav2_zipdepth_convex2x) {
            return_execution_context_locked(engine_key, exec_context);
            engine_repair_allowed = detach_incompatible_engine_locked(engine_key);
            engine = nullptr;
          }
        }
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
      if (!ensure_tensorrt_engine_for_device(
            assets_dir, runtime_model, cuda, cuda_device, artifact
          )) {
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
      engine_io_kind = slot.io_kind;
      if (engine_io_kind != prod_zipdepth_convex2x::engine_io_e::
                              production_dav2_zipdepth_convex2x) {
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
            if (slot_context_limit(slot) ==
                  detail::fused_tensorrt_context_limit &&
                slot.context_accounting.quarantined() != 0u) {
              BOOST_LOG(error)
                << "The sole fused TensorRT execution context is quarantined; process restart "
                   "is required before its optimization profiles can be owned safely again.";
            } else {
              BOOST_LOG(error)
                << "TensorRT context capacity contains no successfully warmed context.";
            }
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
    if (!publish_active_engine_manifest(
          assets_dir, runtime_model.name, artifact, engine_io_kind
        )) {
      // Manifest publication is an evaluator/preflight contract, not a reason to discard a model
      // that is already resident and proven usable for production streaming.
      BOOST_LOG(error) << "Could not publish the active TensorRT engine manifest for model '"
                        << runtime_model.name << "'.";
    }
    BOOST_LOG(info) << "Startup depth runtime '" << runtime_model.name
                    << "' is resident and ready.";
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
}  // namespace models
