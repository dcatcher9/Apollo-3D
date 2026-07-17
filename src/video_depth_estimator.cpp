#include "video_depth_estimator.h"

#include "crypto.h"
#include "cuda_driver_api.h"
#include "logging.h"
#include "model_manager.h"
#include "platform/windows/misc.h"
#include "platform/windows/utils.h"
#include "sbs_perf.h"
#include "utility.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <d3dcompiler.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <sstream>
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
static std::mutex g_model_hash_cache_mutex;

struct depth_shader_cache_entry {
  std::filesystem::file_time_type modified;
  std::vector<std::uint8_t> bytecode;
};

static std::map<std::filesystem::path, depth_shader_cache_entry> g_depth_shader_cache;

struct model_hash_cache_entry {
  std::filesystem::file_time_type modified;
  std::uintmax_t size = 0;
  std::string sha256;
};

static std::map<std::filesystem::path, model_hash_cache_entry> g_model_hash_cache;

static std::string sha256_hex(const crypto::sha256_t &digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (const auto byte : digest) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 0x0f]);
  }
  return result;
}

static std::string sha256_bytes(std::string_view contents) {
  return sha256_hex(crypto::hash(contents));
}

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
  // Hash of the exact serialized bytes passed to deserializeCudaEngine(). Policy provenance
  // must bind this resident plan, not whichever file happens to occupy its path later.
  std::string engine_sha256;
  std::vector<nvinfer1::IExecutionContext *> context_pool;
  std::size_t context_count = 0;
  bool io_validated = false;
  bool io_compatible = false;
  bool has_artistic_outputs = false;
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

// Set the dynamic input shape and validate output byte counts without calling TensorRT APIs that
// return Dims by value (unsafe across the MinGW/MSVC ABI boundary). This also prevents a custom
// model from overflowing Apollo's fixed-size D3D/CUDA output allocations merely by reusing the
// expected tensor names with incompatible shapes.
struct depth_output_sizes {
  size_t depth = 0;
  size_t artistic_global = 0;
};

static bool set_depth_shape_and_validate_outputs(
  nvinfer1::IExecutionContext *exec_context,
  int h,
  int w,
  bool has_artistic_outputs,
  depth_output_sizes &sizes
) {
  if (!exec_context || !exec_context->setInputShape("pixel_values", make_input_dims(h, w))) {
    return false;
  }
  const int64_t expected_depth = (int64_t) h * w * sizeof(float);
  const int64_t depth_bytes = exec_context->getMaxOutputSize("predicted_depth");
  auto compatible_bound = [](int64_t upper_bound, int64_t expected) {
    // TensorRT pads getMaxOutputSize() by a small implementation-defined alignment margin.
    return upper_bound >= expected && upper_bound <= expected + 4096;
  };
  if (!compatible_bound(depth_bytes, expected_depth)) {
    BOOST_LOG(error) << "Depth model output 'predicted_depth' requires " << depth_bytes
                     << " bytes at " << w << 'x' << h << "; expected approximately "
                     << expected_depth << '.';
    return false;
  }
  sizes.depth = (size_t) depth_bytes;
  if (!has_artistic_outputs) {
    return true;
  }
  const int64_t expected_global = 2 * sizeof(float);
  const int64_t global_bytes = exec_context->getMaxOutputSize("artistic_global");
  if (!compatible_bound(global_bytes, expected_global)) {
    BOOST_LOG(error) << "Artistic model output has an incompatible byte count at " << w << 'x' << h
                     << ": global=" << global_bytes << " (expected " << expected_global << ").";
    return false;
  }
  sizes.artistic_global = (size_t) global_bytes;
  return true;
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
    slot.engine_sha256 = blob.empty() ? std::string() :
                                      sha256_bytes(std::string_view(blob.data(), blob.size()));
    slot.engine = g_runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (!slot.engine) {
      slot.engine_sha256.clear();
    }
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
  bool have_artistic_global = false;
  bool input_fp32 = false;
  bool output_fp32 = false;
  bool artistic_global_fp32 = false;
  bool have_unknown_io = false;
  for (int i = 0; i < engine->getNbIOTensors(); i++) {
    const char *name = engine->getIOTensorName(i);
    const auto type = engine->getTensorDataType(name);
    const bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    BOOST_LOG(info) << "Depth engine tensor '" << name << "' " << (is_input ? "(input)" : "(output)")
                    << " dtype=" << tensor_dtype_name(type);
    if (std::string_view(name) == "pixel_values" && is_input) {
      have_in = true;
      input_fp32 = type == nvinfer1::DataType::kFLOAT;
      if (!input_fp32) {
        BOOST_LOG(error) << "Depth model input 'pixel_values' is " << tensor_dtype_name(type)
                         << ", not FP32; rejecting the engine. Use a keep_io_types (FP32 I/O) model.";
      }
    } else if (std::string_view(name) == "predicted_depth" && !is_input) {
      have_out = true;
      output_fp32 = type == nvinfer1::DataType::kFLOAT;
      if (!output_fp32) {
        BOOST_LOG(error) << "Depth model output 'predicted_depth' is " << tensor_dtype_name(type)
                         << ", not FP32; rejecting the engine.";
      }
    } else if (std::string_view(name) == "artistic_global" && !is_input) {
      have_artistic_global = true;
      artistic_global_fp32 = type == nvinfer1::DataType::kFLOAT;
      if (!artistic_global_fp32) {
        BOOST_LOG(error) << "Optional model output 'artistic_global' is " << tensor_dtype_name(type)
                         << ", not FP32; rejecting the engine.";
      }
    } else {
      have_unknown_io = true;
      BOOST_LOG(error) << "Depth model has unsupported " << (is_input ? "input" : "output")
                       << " tensor '" << name << "'; rejecting the engine.";
    }
  }
  if (!have_in || !have_out) {
    BOOST_LOG(error) << "Depth model is missing the expected tensor name(s) 'pixel_values'/'predicted_depth'; "
                        "rejecting the engine.";
  }
  slot.has_artistic_outputs = have_artistic_global;
  slot.io_compatible = have_in && have_out && input_fp32 && output_fp32 &&
                       !have_unknown_io &&
                       (!slot.has_artistic_outputs || artistic_global_fp32);
  return slot.io_compatible;
}

static bool sha256_file(const std::filesystem::path &path, std::string &result) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }

  crypto::md_ctx_t digest_context {EVP_MD_CTX_create()};
  if (!digest_context ||
      !EVP_DigestInit_ex(digest_context.get(), EVP_sha256(), nullptr)) {
    return false;
  }

  std::array<char, 1024 * 1024> buffer {};
  while (file) {
    file.read(buffer.data(), buffer.size());
    const auto read = file.gcount();
    if (read > 0 &&
        !EVP_DigestUpdate(digest_context.get(), buffer.data(), (std::size_t) read)) {
      return false;
    }
  }
  if (file.bad()) {
    return false;
  }

  crypto::sha256_t digest {};
  unsigned int digest_size = 0;
  if (!EVP_DigestFinal_ex(digest_context.get(), digest.data(), &digest_size) ||
      digest_size != digest.size()) {
    return false;
  }
  result = sha256_hex(digest);
  return true;
}

static bool sha256_file_cached(
  const std::filesystem::path &path,
  std::string &result,
  bool reuse_existing = true
) {
  const auto cache_path = path.lexically_normal();
  std::error_code ec;
  const auto modified = std::filesystem::last_write_time(cache_path, ec);
  if (ec) {
    return false;
  }
  const auto size = std::filesystem::file_size(cache_path, ec);
  if (ec) {
    return false;
  }
  if (reuse_existing) {
    std::lock_guard<std::mutex> lock(g_model_hash_cache_mutex);
    const auto cached = g_model_hash_cache.find(cache_path);
    if (cached != g_model_hash_cache.end() &&
        cached->second.modified == modified && cached->second.size == size) {
      result = cached->second.sha256;
      return true;
    }
  }

  std::string calculated;
  if (!sha256_file(cache_path, calculated)) {
    return false;
  }

  // Do not cache a digest if the file changed while it was being read. A subsequent estimator
  // can retry once the copy/export has reached a stable state.
  ec.clear();
  const auto final_modified = std::filesystem::last_write_time(cache_path, ec);
  if (ec) {
    return false;
  }
  const auto final_size = std::filesystem::file_size(cache_path, ec);
  if (ec || final_modified != modified || final_size != size) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(g_model_hash_cache_mutex);
    g_model_hash_cache.insert_or_assign(
      cache_path,
      model_hash_cache_entry {modified, size, calculated}
    );
  }
  result = std::move(calculated);
  return true;
}

static bool read_json_object(
  const std::filesystem::path &path,
  nlohmann::json &result,
  std::string &reason
) {
  std::ifstream file(path);
  if (!file) {
    reason = "missing " + path.filename().string();
    return false;
  }
  try {
    file >> result;
  } catch (const std::exception &e) {
    reason = "invalid " + path.filename().string() + ": " + e.what();
    return false;
  }
  if (!result.is_object()) {
    reason = path.filename().string() + " is not a JSON object";
    return false;
  }
  return true;
}

static std::filesystem::path engine_source_marker_path(const std::filesystem::path &engine_path) {
  auto result = engine_path;
  result += ".source.json";
  return result;
}

static bool write_engine_source_marker(
  const std::filesystem::path &engine_path,
  const std::filesystem::path &onnx_path,
  const std::string &model_name,
  const std::string &build_source_onnx_sha256,
  const std::string &engine_sha256
) {
  std::string onnx_sha256;
  // The builder just consumed this ONNX. Bypass any older cached entry so the marker is always
  // derived from the exact current source, then refresh the process cache for live validation.
  if (!sha256_file_cached(onnx_path, onnx_sha256, false)) {
    BOOST_LOG(warning) << "Could not hash " << onnx_path
                       << "; learned artistic policy provenance will fail closed.";
    return false;
  }
  if (build_source_onnx_sha256.empty() || onnx_sha256 != build_source_onnx_sha256) {
    BOOST_LOG(warning) << "Source ONNX changed while TensorRT was building " << engine_path
                       << "; learned artistic policy provenance will fail closed.";
    return false;
  }
  if (engine_sha256.empty()) {
    BOOST_LOG(warning) << "Could not hash serialized TensorRT plan for " << engine_path
                       << "; learned artistic policy provenance will fail closed.";
    return false;
  }
  const nlohmann::json marker {
    {"schema", 1},
    {"model", model_name},
    {"engine_recipe", models::depth_engine_recipe},
    {"onnx_sha256", onnx_sha256},
    {"engine_sha256", engine_sha256},
  };
  const auto marker_path = engine_source_marker_path(engine_path);
  std::ofstream file(marker_path, std::ios::trunc);
  if (!file) {
    BOOST_LOG(warning) << "Could not write TensorRT source marker " << marker_path
                       << "; learned artistic policy provenance will fail closed.";
    return false;
  }
  file << marker.dump(2) << '\n';
  file.close();
  if (!file) {
    BOOST_LOG(warning) << "Could not finish TensorRT source marker " << marker_path
                       << "; learned artistic policy provenance will fail closed.";
    return false;
  }
  return true;
}

static bool json_number_is(const nlohmann::json &value, double expected) {
  return value.is_number() && std::abs(value.get<double>() - expected) <= 1e-9;
}

static constexpr std::string_view kArtisticDepthInputColorContractSha256 =
  "a18f5bd9829ce79aea9e6945fa829b414456ec700e2b7cbef2693e46d081d104";
static constexpr std::string_view kArtisticInputVariantManifestSha256 =
  "3fff2fb536bcedb805c59f82712ec8d7a8f59f3e09d2cd1676fba337f588d489";

static nlohmann::json artistic_hdr_input_variant(int raw_white) {
  return {
    {"schema", 1},
    {"contract", "apollo-depth-input-variant-v1"},
    {"kind", "simulated-sdr-in-windows-hdr"},
    {"color_mode", "hdr-scrgb-fp16"},
    {"source_encoding", "srgb-rec709-unorm8"},
    {"capture_encoding", "linear-scrgb-rec709-float16"},
    {"windows_sdr_white_level_raw", raw_white},
    {"windows_sdr_white_nits", (double) raw_white * 80.0 / 1000.0},
    {"scrgb_white_scale", (double) raw_white / 1000.0},
    {"color_contract_sha256", kArtisticDepthInputColorContractSha256},
  };
}

static bool validate_artistic_input_variant_manifest(
  const nlohmann::json &metadata,
  std::set<std::string> &declared_color_modes,
  std::string &reason
) {
  const auto manifest = metadata.find("input_variant_manifest");
  const auto manifest_hash = metadata.find("input_variant_manifest_sha256");
  const auto color_hash = metadata.find("depth_input_color_contract_sha256");
  if (manifest == metadata.end() || !manifest->is_object() ||
      manifest_hash == metadata.end() || !manifest_hash->is_string() ||
      color_hash == metadata.end() || !color_hash->is_string()) {
    reason = "missing artistic-policy input color provenance";
    return false;
  }

  const nlohmann::json sdr_variant {
    {"schema", 1},
    {"contract", "apollo-depth-input-variant-v1"},
    {"kind", "sdr-rgb8"},
    {"color_mode", "sdr-srgb-8bit"},
    {"source_encoding", "srgb-rec709-unorm8"},
    {"capture_encoding", "srgb-rec709-unorm8"},
    {"windows_sdr_white_level_raw", nullptr},
    {"windows_sdr_white_nits", nullptr},
    {"scrgb_white_scale", nullptr},
    {"color_contract_sha256", kArtisticDepthInputColorContractSha256},
  };
  // The manifest order is canonical: variants are sorted by their canonical SHA-256 identities,
  // not by white level. Keep this exact four-condition training domain fail closed.
  const nlohmann::json expected_manifest {
    {"schema", 1},
    {"contract", "exact-artistic-policy-input-variants-v1"},
    {"depth_input_color_contract_sha256", kArtisticDepthInputColorContractSha256},
    {"variants", nlohmann::json::array({
       artistic_hdr_input_variant(1000),
       artistic_hdr_input_variant(6000),
       sdr_variant,
       artistic_hdr_input_variant(2500),
     })},
  };
  if (*color_hash != kArtisticDepthInputColorContractSha256 ||
      *manifest_hash != kArtisticInputVariantManifestSha256 ||
      *manifest != expected_manifest || manifest->dump() != expected_manifest.dump() ||
      sha256_bytes(manifest->dump()) != kArtisticInputVariantManifestSha256) {
    reason = "artistic-policy input variant manifest is stale, noncanonical, or unsupported";
    return false;
  }

  declared_color_modes = {"sdr-srgb-8bit", "hdr-scrgb-fp16"};
  return true;
}

static std::string_view artistic_live_color_mode(
  DXGI_FORMAT format,
  models::input_color_space color_space
) {
  if (color_space == models::input_color_space::srgb &&
      (format == DXGI_FORMAT_B8G8R8A8_UNORM ||
       format == DXGI_FORMAT_B8G8R8X8_UNORM ||
       format == DXGI_FORMAT_R8G8B8A8_UNORM)) {
    return "sdr-srgb-8bit";
  }
  if (color_space == models::input_color_space::scrgb_hdr &&
      format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
    return "hdr-scrgb-fp16";
  }
  // In particular, linear_sdr and every format/enum mismatch remain out of domain.
  return {};
}

static bool valid_runtime_regime_acceptance(const nlohmann::json &approval) {
  const auto evidence = approval.find("runtime_regime_acceptance");
  if (evidence == approval.end() || !evidence->is_object() || evidence->size() != 6) {
    return false;
  }
  const nlohmann::json expected {
    {"required_regimes", nlohmann::json::array({"sdr", "hdr"})},
    {"expected_hdr_white_levels_raw", nlohmann::json::array({1000, 2500, 6000})},
    {"missing_regimes", nlohmann::json::array()},
    {"missing_hdr_white_levels_raw", nlohmann::json::array()},
    {"regime_pass", {{"sdr", true}, {"hdr", true}}},
    {"accepted", true},
  };
  return *evidence == expected;
}

static std::string artistic_geometry_key(const nlohmann::json &value) {
  // Canonicalize by declared type before serialization. In particular, JSON 4 and 4.0 are the
  // same numeric geometry even though nlohmann preserves their parser types in dump().
  const nlohmann::json canonical {
    {"source_width", value.at("source_width").get<int64_t>()},
    {"source_height", value.at("source_height").get<int64_t>()},
    {"model_input_width", value.at("model_input_width").get<int64_t>()},
    {"model_input_height", value.at("model_input_height").get<int64_t>()},
    {"depth_short_side", value.at("depth_short_side").get<int64_t>()},
    {"depth_max_aspect", value.at("depth_max_aspect").get<double>()},
    {"eye_width", value.at("eye_width").get<int64_t>()},
    {"eye_height", value.at("eye_height").get<int64_t>()},
    {"content_scale_x", value.at("content_scale_x").get<double>()},
    {"content_scale_y", value.at("content_scale_y").get<double>()},
    {"disparity_raster_width", value.at("disparity_raster_width").get<int64_t>()},
    {"disparity_raster_height", value.at("disparity_raster_height").get<int64_t>()},
    {"color_mode", value.at("color_mode").get<std::string>()},
  };
  return canonical.dump();
}

static bool json_lower_hex_hash_is(const nlohmann::json &value, size_t length) {
  if (!value.is_string()) {
    return false;
  }
  const auto &text = value.get_ref<const std::string &>();
  return text.size() == length && std::all_of(text.begin(), text.end(), [](char ch) {
           return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
         });
}

struct validated_artistic_policy_metadata_t {
  std::string onnx_sha256;
  std::string metadata_sha256;
  std::string deployment_geometry_allowlist_sha256;
  nlohmann::json deployment_geometry_allowlist;
};

static bool json_nonempty_unique_strings(const nlohmann::json &value) {
  if (!value.is_array() || value.empty()) {
    return false;
  }
  std::set<std::string> unique;
  std::string previous;
  for (const auto &item : value) {
    if (!item.is_string() || item.get_ref<const std::string &>().empty() ||
        (!previous.empty() && item.get_ref<const std::string &>() <= previous) ||
        !unique.insert(item.get<std::string>()).second) {
      return false;
    }
    previous = item.get<std::string>();
  }
  return true;
}

static bool valid_unsafe_ceiling_overshoot(const nlohmann::json &approval) {
  const auto evidence = approval.find("unsafe_ceiling_overshoot");
  if (evidence == approval.end() || !evidence->is_object()) {
    return false;
  }
  const auto maximum = evidence->find("maximum_scale");
  const auto maximum_limit = evidence->find("maximum_limit_scale");
  const auto balanced_mean = evidence->find("film_balanced_mean_scale");
  const auto balanced_limit = evidence->find("film_balanced_mean_limit_scale");
  const auto overshoot_rate = evidence->find("film_balanced_overshoot_rate_pct");
  if (maximum == evidence->end() || !maximum->is_number() ||
      maximum_limit == evidence->end() || !maximum_limit->is_number() ||
      balanced_mean == evidence->end() || !balanced_mean->is_number() ||
      balanced_limit == evidence->end() || !balanced_limit->is_number() ||
      overshoot_rate == evidence->end() || !overshoot_rate->is_number()) {
    return false;
  }
  const double maximum_value = maximum->get<double>();
  const double maximum_limit_value = maximum_limit->get<double>();
  const double balanced_mean_value = balanced_mean->get<double>();
  const double balanced_limit_value = balanced_limit->get<double>();
  const double overshoot_rate_value = overshoot_rate->get<double>();
  return std::isfinite(maximum_value) && maximum_value >= 0.0 &&
         maximum_value <= 0.05 + 1e-9 &&
         std::isfinite(maximum_limit_value) &&
         std::abs(maximum_limit_value - 0.05) <= 1e-12 &&
         std::isfinite(balanced_mean_value) && balanced_mean_value >= 0.0 &&
         balanced_mean_value <= 0.01 + 1e-9 &&
         std::isfinite(balanced_limit_value) &&
         std::abs(balanced_limit_value - 0.01) <= 1e-12 &&
         std::isfinite(overshoot_rate_value) && overshoot_rate_value >= 0.0 &&
         overshoot_rate_value <= 100.0 &&
         evidence->value("maximum_pass", false) &&
         evidence->value("film_balanced_mean_pass", false);
}

// The artistic head is optional. Its output is consumed only when the exported semantic
// sidecar, the exact ONNX bytes used to build the resident plan, and every policy-affecting
// production setting match the label frontier. A failure disables only the artistic action;
// predicted_depth remains a valid ordinary DA-V2 output.
static bool validate_artistic_policy_metadata_impl(
  const std::filesystem::path &assets_dir,
  const std::filesystem::path &engine_path,
  const config::depth_model_info &model,
  const config::video_t::sbs_t &cfg,
  const std::string &resident_engine_sha256,
  models::artistic_policy_authorization authorization,
  validated_artistic_policy_metadata_t &validated,
  std::string &reason
) {
  validated = {};
  const auto onnx_path = assets_dir / (model.name + ".onnx");
  const auto metadata_path = assets_dir / (model.name + ".json");
  std::error_code ec;
  if (!std::filesystem::is_regular_file(onnx_path, ec)) {
    reason = "missing source ONNX " + onnx_path.filename().string();
    return false;
  }
  ec.clear();
  if (!std::filesystem::is_regular_file(engine_path, ec)) {
    reason = "missing TensorRT engine " + engine_path.filename().string();
    return false;
  }
  ec.clear();
  const auto onnx_modified = std::filesystem::last_write_time(onnx_path, ec);
  if (ec) {
    reason = "cannot read source ONNX timestamp";
    return false;
  }
  const auto engine_modified = std::filesystem::last_write_time(engine_path, ec);
  if (ec || engine_modified < onnx_modified) {
    reason = ec ? "cannot read TensorRT engine timestamp" :
                  "TensorRT engine predates its source ONNX";
    return false;
  }

  std::string onnx_sha256;
  if (!sha256_file_cached(onnx_path, onnx_sha256)) {
    reason = "cannot hash source ONNX";
    return false;
  }
  if (resident_engine_sha256.empty()) {
    reason = "resident TensorRT engine has no deserialization hash";
    return false;
  }

  nlohmann::json metadata;
  if (!read_json_object(metadata_path, metadata, reason)) {
    return false;
  }
  std::string metadata_sha256;
  if (!sha256_file(metadata_path, metadata_sha256)) {
    reason = "cannot hash artistic-policy metadata sidecar";
    return false;
  }
  if (metadata.value("schema", 0) != 5 ||
      metadata.value("onnx_sha256", "") != onnx_sha256 ||
      metadata.value("deployed_model", "") != model.name ||
      metadata.value("policy_contract", "") != "safe-frontier-multistyle-apollo-v1" ||
      metadata.value("policy_feature_contract", "") != "multiscale-dino-depth-dpt-stats-v1") {
    reason = "export schema, deployed-model identity, ONNX hash, or policy/feature contract mismatch";
    return false;
  }

  const auto evaluation_sha = metadata.find("evaluation_sha256");
  const auto metric_sha = metadata.find("metric_sha256");
  const auto geometry_hash = metadata.find("deployment_geometry_allowlist_sha256");
  const auto geometry_allowlist = metadata.find("deployment_geometry_allowlist");
  const auto approval = metadata.find("approval_contract");
  if (evaluation_sha == metadata.end() || !json_lower_hex_hash_is(*evaluation_sha, 64) ||
      metric_sha == metadata.end() || !json_lower_hex_hash_is(*metric_sha, 16) ||
      geometry_hash == metadata.end() || !json_lower_hex_hash_is(*geometry_hash, 64) ||
      geometry_allowlist == metadata.end() || !geometry_allowlist->is_object() ||
      approval == metadata.end() || !approval->is_object()) {
    reason = "missing or malformed sealed-test artistic-policy approval";
    return false;
  }
  std::set<std::string> declared_color_modes;
  if (!validate_artistic_input_variant_manifest(
        metadata, declared_color_modes, reason
      )) {
    return false;
  }
  const auto decision_accepted = approval->find("decision_accepted");
  const auto productions = approval->find("sealed_test_productions");
  if (approval->value("contract", "") != "sealed-test-artistic-policy-v3" ||
      approval->value("evaluation_schema", 0) != 13 ||
      approval->value("split", "") != "test" ||
      approval->value("evaluation_sha256", "") != evaluation_sha->get<std::string>() ||
      approval->value("metric_sha256", "") != metric_sha->get<std::string>() ||
      approval->value("deployment_geometry_allowlist_sha256", "") !=
        geometry_hash->get<std::string>() ||
      metric_sha->get<std::string>() != APOLLO_ARTISTIC_METRIC_CONTRACT_SHA256 ||
      decision_accepted == approval->end() || !decision_accepted->is_boolean() ||
      !decision_accepted->get<bool>() ||
      !json_lower_hex_hash_is(approval->value("checkpoint_sha256", nlohmann::json()), 64) ||
      !json_lower_hex_hash_is(approval->value("active_split_sha256", nlohmann::json()), 64) ||
      !json_lower_hex_hash_is(approval->value("metric_sha256", nlohmann::json()), 16) ||
      !json_lower_hex_hash_is(
        approval->value("label_fitter_identity_sha256", nlohmann::json()),
        64
      ) ||
      !json_lower_hex_hash_is(approval->value("test_labels_sha256", nlohmann::json()), 64) ||
      !json_lower_hex_hash_is(
        approval->value("deployment_geometry_allowlist_sha256", nlohmann::json()),
        64
      ) ||
      productions == approval->end() || !productions->is_array() || productions->empty()) {
    reason = "sealed-test artistic-policy approval contract is incompatible or incomplete";
    return false;
  }
  if (!json_nonempty_unique_strings(*productions) ||
      !valid_unsafe_ceiling_overshoot(*approval) ||
      !valid_runtime_regime_acceptance(*approval)) {
    reason = "sealed-test production identities, safety evidence, or SDR/HDR acceptance are invalid";
    return false;
  }

  const auto outputs = metadata.find("outputs");
  if (outputs == metadata.end() || !outputs->is_object()) {
    reason = "missing outputs contract";
    return false;
  }
  const auto artistic = outputs->find("artistic_global");
  const nlohmann::json expected_shape = nlohmann::json::array({1, 2});
  const nlohmann::json expected_channels = nlohmann::json::array({
    "safe_scale_ceiling",
    "safe_ceiling_confidence",
  });
  if (artistic == outputs->end() || !artistic->is_object() ||
      artistic->value("dtype", "") != "float32" ||
      artistic->value("shape", nlohmann::json()) != expected_shape ||
      artistic->value("channels", nlohmann::json()) != expected_channels) {
    reason = "artistic_global must be FP32 [1,2] with safe-ceiling/confidence channels";
    return false;
  }

  const auto output_semantics = metadata.find("output_semantics");
  const nlohmann::json expected_output_rules {
    {"safe_cap", "safe_ceiling_confidence >= 0.5 ? clamp(safe_scale_ceiling, 1.0, 1.5) : 1.0"},
    {"clean", "1.0"},
    {"balanced", "1.0 + 0.5 * (safe_cap - 1.0)"},
    {"immersive", "safe_cap"},
  };
  if (output_semantics == metadata.end() || !output_semantics->is_object() ||
      output_semantics->value("artistic_global_0", "") != "safe_scale_ceiling" ||
      output_semantics->value("artistic_global_1", "") != "safe_ceiling_confidence" ||
      output_semantics->value("confidence_semantics", "") != "hard actionable probability" ||
      !json_number_is(output_semantics->value("action_threshold", nlohmann::json()), 0.5) ||
      output_semantics->value("preset_rules", nlohmann::json()) != expected_output_rules) {
    reason = "checkpoint output semantics mismatch";
    return false;
  }

  const auto bounds = metadata.find("bounds");
  const auto runtime = metadata.find("runtime");
  if (bounds == metadata.end() || !bounds->is_object() ||
      !json_number_is(bounds->value("scale_delta_max", nlohmann::json()), 0.5) ||
      runtime == metadata.end() || !runtime->is_object() ||
      runtime->value("confidence_semantics", "") != "hard actionable probability" ||
      !json_number_is(runtime->value("action_threshold", nlohmann::json()), 0.5) ||
      !json_number_is(runtime->value("inactive_ceiling", nlohmann::json()), 1.0) ||
      runtime->value("ceiling_bounds", nlohmann::json()) != nlohmann::json::array({1.0, 1.5})) {
    reason = "safe-ceiling runtime bounds or confidence semantics mismatch";
    return false;
  }
  const auto preset_rules = runtime->find("preset_rules");
  const nlohmann::json expected_rules {
    {"safe_cap", "confidence >= 0.5 ? clamp(ceiling, 1.0, 1.5) : 1.0"},
    {"clean", "1.0"},
    {"balanced", "1.0 + 0.5 * (safe_cap - 1.0)"},
    {"immersive", "safe_cap"},
  };
  if (preset_rules == runtime->end() || *preset_rules != expected_rules) {
    reason = "artistic preset semantics mismatch";
    return false;
  }

  nlohmann::json marker;
  const auto marker_path = engine_source_marker_path(engine_path);
  if (!read_json_object(marker_path, marker, reason)) {
    return false;
  }
  if (marker.value("schema", 0) != 1 ||
      marker.value("model", "") != model.name ||
      marker.value("engine_recipe", "") != models::depth_engine_recipe ||
      marker.value("onnx_sha256", "") != onnx_sha256 ||
      marker.value("engine_sha256", "") != resident_engine_sha256) {
    reason = "TensorRT engine source marker does not match the resident plan/ONNX/recipe";
    return false;
  }

  const nlohmann::json expected_baseline {
    {"profile", cfg.profile},
    {"pop_strength", cfg.pop_strength},
    {"adaptive_pop", cfg.adaptive_pop},
    {"adaptive_pop_max", cfg.adaptive_pop_max},
    {"ema", cfg.ema},
    {"ema_edge_change", cfg.ema_edge_change},
    {"ema_edge_gradient", cfg.ema_edge_gradient},
    {"ema_edge_strength", cfg.ema_edge_strength},
    {"minmax_ema", cfg.minmax_ema},
    {"subject_lock", cfg.subject_lock},
    {"subject_recenter", cfg.subject_recenter},
    {"subject_stretch", cfg.subject_stretch},
    {"depth_short_side", cfg.depth_short_side},
    {"depth_max_aspect", cfg.depth_max_aspect},
    {"zero_plane", cfg.zero_plane},
    {"depth_step", "current-once"},
    {"depth_compensation", "none"},
    {"literal_bestv2", false},
    {"harness_schema", 28},
    {"eval_schema", 31},
    {"warp_contract", "apollo-safe-frontier-v1"},
    {"policy_warp_source_sha256", APOLLO_ARTISTIC_WARP_CONTRACT_SHA256},
    {"metric_sha256", APOLLO_ARTISTIC_METRIC_CONTRACT_SHA256},
  };
  const auto baseline = metadata.find("policy_baseline");
  if (baseline == metadata.end() || !baseline->is_object()) {
    reason = "missing policy training baseline";
    return false;
  }
  const std::string base_depth_model = metadata.value("base_depth_model", "");
  if (base_depth_model.empty() || baseline->value("depth_model", "") != base_depth_model) {
    reason = "source depth-model identity differs from the policy training baseline";
    return false;
  }
  auto comparable_baseline = *baseline;
  comparable_baseline.erase("depth_model");
  if (comparable_baseline != expected_baseline) {
    reason = "current resolved SBS configuration differs from the policy training baseline";
    return false;
  }

  if (geometry_allowlist->size() != 3 ||
      geometry_allowlist->value("schema", 0) != 1 ||
      geometry_allowlist->value("contract", "") !=
        "exact-artistic-policy-render-tuples-v1") {
    reason = "deployment geometry allow-list contract is incompatible";
    return false;
  }
  const auto geometry_tuples = geometry_allowlist->find("tuples");
  if (geometry_tuples == geometry_allowlist->end() || !geometry_tuples->is_array() ||
      geometry_tuples->empty()) {
    reason = "deployment geometry allow-list has no exact tuples";
    return false;
  }
  std::string previous_geometry_key;
  std::set<std::string> allowed_geometry_keys;
  std::set<std::string> geometry_color_modes;
  for (const auto &tuple : *geometry_tuples) {
    if (!tuple.is_object() || tuple.size() != 13) {
      reason = "deployment geometry allow-list contains a non-object tuple";
      return false;
    }
    for (const auto *field : {
           "source_width", "source_height", "model_input_width", "model_input_height",
           "eye_width", "eye_height", "disparity_raster_width", "disparity_raster_height"
         }) {
      const auto value = tuple.find(field);
      if (value == tuple.end() || !value->is_number_integer() ||
          value->get<int64_t>() <= 0 ||
          value->get<int64_t>() > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) {
        reason = std::string("deployment geometry has invalid ") + field;
        return false;
      }
    }
    for (const auto *field : {"content_scale_x", "content_scale_y"}) {
      const auto value = tuple.find(field);
      if (value == tuple.end() || !value->is_number()) {
        reason = std::string("deployment geometry has invalid ") + field;
        return false;
      }
      const double numeric = value->get<double>();
      if (!std::isfinite(numeric) || numeric <= 0.0 || numeric > 1.0) {
        reason = std::string("deployment geometry has out-of-range ") + field;
        return false;
      }
    }
    const auto tuple_short_side = tuple.find("depth_short_side");
    const auto tuple_max_aspect = tuple.find("depth_max_aspect");
    if (tuple_short_side == tuple.end() || !tuple_short_side->is_number_integer() ||
        tuple_short_side->get<int>() != std::max(196, cfg.depth_short_side) ||
        tuple_max_aspect == tuple.end() || !tuple_max_aspect->is_number() ||
        !std::isfinite(tuple_max_aspect->get<double>()) ||
        tuple_max_aspect->get<double>() < 1.0 ||
        std::abs(tuple_max_aspect->get<double>() -
                 std::max(1.0, cfg.depth_max_aspect)) > 1e-9) {
      reason = "deployment geometry depth preprocessing differs from the resolved profile";
      return false;
    }
    const std::string tuple_color_mode = tuple.value("color_mode", "");
    if (tuple.value("disparity_raster_width", 0) != tuple.value("eye_width", -1) ||
        tuple.value("disparity_raster_height", 0) != tuple.value("eye_height", -1) ||
        tuple.value("eye_width", 0) > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION / 2 ||
        !declared_color_modes.contains(tuple_color_mode)) {
      reason = "deployment geometry must be a complete, declared input-color output-eye raster";
      return false;
    }
    geometry_color_modes.insert(tuple_color_mode);

    const int source_width = tuple.value("source_width", 0);
    const int source_height = tuple.value("source_height", 0);
    const float raw_aspect = (float) source_width / (float) source_height;
    const float fitted_aspect = raw_aspect >= 1.0f ?
                                  std::min(raw_aspect, (float) tuple_max_aspect->get<double>()) :
                                  1.0f / std::min(
                                             1.0f / raw_aspect,
                                             (float) tuple_max_aspect->get<double>()
                                           );
    const auto expected_model = aspect_aligned_dims(
      fitted_aspect,
      tuple_short_side->get<int>(),
      std::min(1008, source_width),
      std::min(1008, source_height)
    );
    if (tuple.value("model_input_width", 0) != expected_model.first ||
        tuple.value("model_input_height", 0) != expected_model.second) {
      reason = "deployment geometry has stale model-input dimensions";
      return false;
    }

    const int eye_width = tuple.value("eye_width", 0);
    const int eye_height = tuple.value("eye_height", 0);
    const float source_aspect = (float) source_width / (float) source_height;
    const float eye_aspect = (float) eye_width / (float) eye_height;
    const float expected_scale_x = source_aspect > eye_aspect ?
                                     1.0f : source_aspect / eye_aspect;
    const float expected_scale_y = source_aspect > eye_aspect ?
                                     eye_aspect / source_aspect : 1.0f;
    if (std::abs(tuple.value("content_scale_x", 0.0) - expected_scale_x) > 1e-7 ||
        std::abs(tuple.value("content_scale_y", 0.0) - expected_scale_y) > 1e-7) {
      reason = "deployment geometry has inconsistent source-to-eye content scales";
      return false;
    }

    const std::string geometry_key = artistic_geometry_key(tuple);
    if (tuple.dump() != geometry_key ||
        (!previous_geometry_key.empty() && geometry_key <= previous_geometry_key)) {
      reason = "deployment geometry allow-list is not canonical, sorted, and unique";
      return false;
    }
    previous_geometry_key = geometry_key;
    allowed_geometry_keys.insert(geometry_key);
  }
  if (geometry_color_modes != declared_color_modes) {
    reason = "deployment geometry allow-list does not cover every authenticated input color mode";
    return false;
  }
  if (sha256_bytes(geometry_allowlist->dump()) != geometry_hash->get<std::string>()) {
    reason = "deployment geometry allow-list hash does not match its canonical contents";
    return false;
  }

  if (authorization == models::artistic_policy_authorization::candidate_evaluation) {
    // The offline harness must render a fresh candidate before the final deployment manifest can
    // exist. It still requires the sealed test, exact ONNX/engine/config identities, and canonical
    // geometry allow-list above. Only the final live-only promotion checks below are deferred.
    validated.onnx_sha256 = onnx_sha256;
    validated.metadata_sha256 = metadata_sha256;
    validated.deployment_geometry_allowlist_sha256 = geometry_hash->get<std::string>();
    validated.deployment_geometry_allowlist = *geometry_allowlist;
    return true;
  }

  nlohmann::json deployment;
  const auto deployment_path = assets_dir / (model.name + ".deployment.json");
  if (!read_json_object(deployment_path, deployment, reason)) {
    return false;
  }
  const auto deployment_approved = deployment.find("approved");
  const auto deployment_model = deployment.find("model");
  if (deployment.value("schema", 0) != 1 ||
      deployment.value("contract", "") != "apollo-artistic-policy-deployment-v1" ||
      deployment.value("created_at", "").empty() ||
      deployment_approved == deployment.end() || !deployment_approved->is_boolean() ||
      deployment_model == deployment.end() || !deployment_model->is_object()) {
    reason = "artistic-policy deployment manifest is malformed";
    return false;
  }
  if (authorization == models::artistic_policy_authorization::headset_review) {
    if (deployment.value("stage", "") != "headset-review" ||
        deployment_approved->get<bool>() || deployment.contains("headset_review") ||
        deployment.size() != 9) {
      reason = "live headset review requires an unapproved headset-review stage without a prior headset decision";
      return false;
    }
  } else if (deployment.value("stage", "") != "production" ||
             !deployment_approved->get<bool>() || deployment.size() != 10) {
    reason = "live production requires an approved production deployment";
    return false;
  }
  const auto &promoted = *deployment_model;
  if (promoted.size() != 14 ||
      promoted.value("deployed_model", "") != model.name ||
      promoted.value("base_depth_model", "") != base_depth_model ||
      promoted.value("onnx_sha256", "") != onnx_sha256 ||
      promoted.value("metadata_sha256", "") != metadata_sha256 ||
      promoted.value("checkpoint_sha256", "") !=
        approval->value("checkpoint_sha256", "") ||
      promoted.value("evaluation_sha256", "") != evaluation_sha->get<std::string>() ||
      promoted.value("metric_sha256", "") != metric_sha->get<std::string>() ||
      promoted.value("policy_warp_source_sha256", "") !=
        std::string(APOLLO_ARTISTIC_WARP_CONTRACT_SHA256) ||
      promoted.value("active_split_sha256", "") !=
        approval->value("active_split_sha256", "") ||
      promoted.value("label_fitter_identity_sha256", "") !=
        approval->value("label_fitter_identity_sha256", "") ||
      promoted.value("test_labels_sha256", "") !=
        approval->value("test_labels_sha256", "") ||
      promoted.value("deployment_geometry_allowlist_sha256", "") !=
        geometry_hash->get<std::string>() ||
      promoted.value("deployment_geometry_allowlist", nlohmann::json()) !=
        *geometry_allowlist ||
      promoted.value("sealed_test_productions", nlohmann::json()) != *productions) {
    reason = "deployment promotion does not match the accepted model/evaluation contracts";
    return false;
  }
  for (const auto *field : {
         "checkpoint_sha256", "evaluation_sha256", "active_split_sha256",
         "label_fitter_identity_sha256", "test_labels_sha256",
         "deployment_geometry_allowlist_sha256"
       }) {
    if (!json_lower_hex_hash_is(promoted.value(field, nlohmann::json()), 64)) {
      reason = std::string("deployment promotion has invalid ") + field;
      return false;
    }
  }

  const auto neutrality = deployment.find("neutrality");
  if (neutrality == deployment.end() || !neutrality->is_object() ||
      neutrality->size() != 8 ||
      neutrality->value("candidate_onnx_sha256", "") != onnx_sha256 ||
      neutrality->value("reference_model", "") != base_depth_model ||
      neutrality->value("preprocessing_contract", "") !=
        "apollo-dav2-srgb-native-capped-v1" ||
      !json_lower_hex_hash_is(
        neutrality->value("report_sha256", nlohmann::json()),
        64
      ) ||
      !json_lower_hex_hash_is(
        neutrality->value("reference_onnx_sha256", nlohmann::json()),
        64
      )) {
    reason = "deployment promotion has invalid depth-neutrality evidence";
    return false;
  }
  const auto limits = neutrality->find("limits");
  if (limits == neutrality->end() || !limits->is_object() || limits->size() != 2 ||
      !limits->contains("production_normalized_mean_abs") ||
      !limits->contains("production_normalized_p99_abs") ||
      !limits->at("production_normalized_mean_abs").is_number() ||
      !limits->at("production_normalized_p99_abs").is_number() ||
      !std::isfinite(limits->at("production_normalized_mean_abs").get<double>()) ||
      !std::isfinite(limits->at("production_normalized_p99_abs").get<double>()) ||
      limits->at("production_normalized_mean_abs").get<double>() <= 0.0 ||
      limits->at("production_normalized_p99_abs").get<double>() <= 0.0 ||
      limits->at("production_normalized_mean_abs").get<double>() > 1.0 / 1024.0 ||
      limits->at("production_normalized_p99_abs").get<double>() > 2.0 / 1024.0) {
    reason = "deployment promotion relaxed the depth-neutrality limits";
    return false;
  }
  const auto canonical_frames = neutrality->find("canonical_core_first_frames");
  const auto evidence_count = neutrality->find("evidence_image_count");
  if (canonical_frames == neutrality->end() || !canonical_frames->is_object() ||
      canonical_frames->empty() || evidence_count == neutrality->end() ||
      !evidence_count->is_number_integer() || evidence_count->get<int64_t>() <= 0 ||
      (size_t) evidence_count->get<int64_t>() < canonical_frames->size()) {
    reason = "deployment promotion has incomplete canonical depth-neutrality coverage";
    return false;
  }
  for (auto it = canonical_frames->begin(); it != canonical_frames->end(); ++it) {
    if (it.key().empty() || !json_lower_hex_hash_is(it.value(), 64)) {
      reason = "deployment promotion has an invalid canonical depth-neutrality identity";
      return false;
    }
  }

  const auto render_gates = deployment.find("render_gates");
  if (render_gates == deployment.end() || !render_gates->is_object() ||
      render_gates->size() != 4) {
    reason = "deployment promotion lacks full render gates";
    return false;
  }
  std::set<std::string> observed_geometry_keys;
  struct render_gate_requirement_t {
    const char *key;
    const char *suite;
    const char *style;
  };
  constexpr std::array render_gate_requirements {
    render_gate_requirement_t {"core", "core", "immersive"},
    render_gate_requirement_t {"extended", "extended", "immersive"},
    render_gate_requirement_t {"balanced_core", "core", "balanced"},
    render_gate_requirement_t {"balanced_extended", "extended", "balanced"},
  };
  for (const auto &requirement : render_gate_requirements) {
    const auto gate = render_gates->find(requirement.key);
    if (gate == render_gates->end() || !gate->is_object() ||
        gate->size() != 17 ||
        gate->value("suite", "") != requirement.suite ||
        gate->value("artistic_style", "") != requirement.style ||
        gate->value("verdict", "") != "pass" ||
        gate->value("eval_schema", 0) != 31 || gate->value("harness_schema", 0) != 28 ||
        gate->value("metric_sha256", "") != metric_sha->get<std::string>() ||
        gate->value("policy_warp_source_sha256", "") !=
          std::string(APOLLO_ARTISTIC_WARP_CONTRACT_SHA256) ||
        gate->value("model_onnx_sha256", "") != onnx_sha256 ||
        gate->value("policy_metadata_sha256", "") != metadata_sha256 ||
        gate->value("deployment_geometry_allowlist_sha256", "") !=
          geometry_hash->get<std::string>() ||
        gate->value("artistic_policy_consumed", false) != true ||
        gate->value("artistic_policy_authorization", "") != "candidate-evaluation" ||
        gate->value("timestamp", "").empty() ||
        !json_lower_hex_hash_is(gate->value("results_sha256", nlohmann::json()), 64)) {
      reason = std::string("deployment promotion has invalid ") + requirement.key +
               " render gate";
      return false;
    }
    const auto clip_set = gate->find("clip_set_sha1");
    if (clip_set == gate->end() || !clip_set->is_object() || clip_set->empty()) {
      reason = std::string("deployment promotion has no ") + requirement.key +
               " clip identities";
      return false;
    }
    for (auto it = clip_set->begin(); it != clip_set->end(); ++it) {
      if (it.key().empty() || !json_lower_hex_hash_is(it.value(), 12)) {
        reason = std::string("deployment promotion has invalid ") + requirement.key +
                 " clip identity";
        return false;
      }
    }
    const auto baseline_identities = gate->find("baseline_identities");
    if (baseline_identities == gate->end() || !baseline_identities->is_object() ||
        baseline_identities->size() != clip_set->size()) {
      reason = std::string("deployment promotion has incomplete ") + requirement.key +
               " baseline identities";
      return false;
    }
    for (auto it = baseline_identities->begin(); it != baseline_identities->end(); ++it) {
      if (!clip_set->contains(it.key()) || !json_lower_hex_hash_is(it.value(), 64)) {
        reason = std::string("deployment promotion has invalid ") + requirement.key +
                 " baseline identity";
        return false;
      }
    }
    const auto observed = gate->find("observed_deployment_geometries");
    if (observed == gate->end() || !observed->is_array() || observed->empty()) {
      reason = std::string("deployment promotion has no ") + requirement.key +
               " observed deployment geometries";
      return false;
    }
    for (const auto &geometry : *observed) {
      if (!geometry.is_object() || geometry.size() != 13) {
        reason = std::string("deployment promotion has a malformed ") + requirement.key +
                 " geometry";
        return false;
      }
      const std::string key = artistic_geometry_key(geometry);
      if (!allowed_geometry_keys.contains(key)) {
        reason = std::string("deployment promotion has an unapproved ") + requirement.key +
                 " geometry";
        return false;
      }
      observed_geometry_keys.insert(key);
    }
  }

  const auto geometry_coverage = deployment.find("deployment_geometry_coverage");
  std::set<std::string> coverage_geometry_keys;
  if (geometry_coverage != deployment.end() && geometry_coverage->is_array()) {
    for (const auto &geometry : *geometry_coverage) {
      if (geometry.is_object() && geometry.size() == 13) {
        coverage_geometry_keys.insert(artistic_geometry_key(geometry));
      }
    }
  }
  if (geometry_coverage == deployment.end() || !geometry_coverage->is_array() ||
      geometry_coverage->size() != geometry_tuples->size() ||
      coverage_geometry_keys != allowed_geometry_keys ||
      observed_geometry_keys != allowed_geometry_keys) {
    reason = "deployment promotion does not prove fresh render coverage of every exact geometry";
    return false;
  }

  if (authorization == models::artistic_policy_authorization::headset_review) {
    validated.onnx_sha256 = onnx_sha256;
    validated.metadata_sha256 = metadata_sha256;
    validated.deployment_geometry_allowlist_sha256 = geometry_hash->get<std::string>();
    validated.deployment_geometry_allowlist = *geometry_allowlist;
    return true;
  }

  const auto headset = deployment.find("headset_review");
  if (headset == deployment.end() || !headset->is_object() || headset->size() != 12) {
    reason = "deployment promotion lacks headset review";
    return false;
  }
  const auto headset_approved = headset->find("approved");
  const auto reviewed_geometry = headset->find("deployment_geometry");
  const auto reviewed_geometry_index = headset->find("deployment_geometry_index");
  const auto refresh_hz = headset->find("refresh_hz");
  if (headset_approved == headset->end() || !headset_approved->is_boolean() ||
      !headset_approved->get<bool>() || headset->value("style", "") != "immersive" ||
      headset->value("reviewer", "").empty() || headset->value("device", "").empty() ||
      headset->value("notes", "").empty() || headset->value("reviewed_at", "").empty() ||
      headset->value("deployment_geometry_allowlist_sha256", "") !=
        geometry_hash->get<std::string>() ||
      reviewed_geometry == headset->end() || !reviewed_geometry->is_object() ||
      reviewed_geometry_index == headset->end() ||
      !reviewed_geometry_index->is_number_integer() ||
      reviewed_geometry_index->get<int64_t>() < 0 ||
      (size_t) reviewed_geometry_index->get<int64_t>() >= geometry_tuples->size() ||
      refresh_hz == headset->end() || !refresh_hz->is_number() ||
      !std::isfinite(refresh_hz->get<double>()) || refresh_hz->get<double>() <= 0.0) {
    reason = "deployment promotion headset review is incomplete";
    return false;
  }
  const auto &indexed_geometry = geometry_tuples->at(
    (size_t) reviewed_geometry_index->get<int64_t>()
  );
  const std::uint64_t full_sbs_width =
    2ull * indexed_geometry.value("eye_width", std::uint64_t {0});
  const std::string expected_resolution =
    std::to_string(full_sbs_width) + "x" +
    std::to_string(indexed_geometry.value("eye_height", std::uint64_t {0}));
  if (artistic_geometry_key(*reviewed_geometry) != artistic_geometry_key(indexed_geometry) ||
      headset->value("color_mode", "") != indexed_geometry.value("color_mode", "") ||
      headset->value("resolution", "") != expected_resolution) {
    reason = "deployment promotion headset review is not bound to its exact approved geometry";
    return false;
  }

  validated.onnx_sha256 = onnx_sha256;
  validated.metadata_sha256 = metadata_sha256;
  validated.deployment_geometry_allowlist_sha256 = geometry_hash->get<std::string>();
  validated.deployment_geometry_allowlist = *geometry_allowlist;
  return true;
}

static bool validate_artistic_policy_metadata(
  const std::filesystem::path &assets_dir,
  const std::filesystem::path &engine_path,
  const config::depth_model_info &model,
  const config::video_t::sbs_t &cfg,
  const std::string &resident_engine_sha256,
  models::artistic_policy_authorization authorization,
  validated_artistic_policy_metadata_t &validated,
  std::string &reason
) noexcept {
  try {
    return validate_artistic_policy_metadata_impl(
      assets_dir,
      engine_path,
      model,
      cfg,
      resident_engine_sha256,
      authorization,
      validated,
      reason
    );
  } catch (const std::exception &e) {
    validated = {};
    try {
      reason = std::string("artistic policy metadata validation threw: ") + e.what();
    } catch (...) {
      reason.clear();
    }
    return false;
  } catch (...) {
    validated = {};
    try {
      reason = "artistic policy metadata validation threw an unknown exception";
    } catch (...) {
      reason.clear();
    }
    return false;
  }
}

static bool warmup_execution_context(
  cuda_driver_api &cuda,
  CUcontext cuda_ctx,
  nvinfer1::IExecutionContext *exec_context,
  bool has_artistic_outputs
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
  depth_output_sizes output_sizes;
  if (!set_depth_shape_and_validate_outputs(
        exec_context,
        h,
        w,
        has_artistic_outputs,
        output_sizes
      )) {
    return false;
  }
  const size_t in_elems = (size_t) 3 * h * w;
  CUdeviceptr d_in = 0;
  CUdeviceptr d_out = 0;
  CUdeviceptr d_artistic_global = 0;
  if (cuda.cuMemAlloc(&d_in, in_elems * sizeof(float)) != CUDA_SUCCESS) {
    return false;
  }
  auto free_input = util::fail_guard([&]() {
    cuda.cuMemFree(d_in);
  });
  if (cuda.cuMemAlloc(&d_out, output_sizes.depth) != CUDA_SUCCESS) {
    return false;
  }
  auto free_output = util::fail_guard([&]() {
    cuda.cuMemFree(d_out);
  });
  if (has_artistic_outputs) {
    if (cuda.cuMemAlloc(&d_artistic_global, output_sizes.artistic_global) != CUDA_SUCCESS) {
      return false;
    }
  }
  auto free_artistic_outputs = util::fail_guard([&]() {
    if (d_artistic_global) {
      cuda.cuMemFree(d_artistic_global);
    }
  });

  bool bound = exec_context->setTensorAddress("pixel_values", (void *) d_in) &&
               exec_context->setTensorAddress("predicted_depth", (void *) d_out);
  if (has_artistic_outputs) {
    bound = bound &&
            exec_context->setTensorAddress("artistic_global", (void *) d_artistic_global);
  }
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

    // TensorRT parses by filename and may spend minutes building. Hash before parsing and require
    // the same bytes after serialization so a concurrent model replacement can never receive a
    // marker for a plan built from different ONNX contents. Marker failure does not discard the
    // usable depth engine; it only keeps the optional policy disabled.
    std::string build_source_onnx_sha256;
    if (!sha256_file_cached(model_path, build_source_onnx_sha256, false)) {
      BOOST_LOG(warning) << "Could not hash source ONNX before TensorRT build; "
                            "learned artistic policy provenance will fail closed.";
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
    // Level 5 makes TensorRT compare generated kernels against its static tactics. Keep this in
    // the recipe-specific engine contract: changing the level must never silently reuse a plan
    // selected under the default level 3 search.
    config->setBuilderOptimizationLevel(depth_engine_builder_level);
    BOOST_LOG(info) << "TensorRT builder optimization level " << depth_engine_builder_level << '.';

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

    // DA-V2 contract: input "pixel_values" [1,3,H,W], mandatory output
    // "predicted_depth", and an optional global artistic policy.
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
      } else if (std::string_view(tensor->getName()) == "artistic_global") {
        // Optional global policy output is retained and validated after deserialization.
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
      const auto *serialized_data = static_cast<const char *>(serializedModel->data());
      const std::string serialized_sha256 = serializedModel->size() > 0 ?
                                              sha256_bytes(std::string_view(
                                                serialized_data,
                                                serializedModel->size()
                                              )) :
                                              std::string();
      // Save under the recipe-specific engine name so a later recipe change rebuilds
      // rather than silently reusing this engine's (now-wrong) I/O layout.
      auto engine_path = assets_dir / engine_name;
      std::ofstream p(engine_path, std::ios::binary);
      if (p) {
        p.write(serialized_data, serializedModel->size());
        p.close();
        if (p) {
          BOOST_LOG(info) << "Saved built engine to " << engine_path;
          write_engine_source_marker(
            engine_path,
            model_path,
            model_name,
            build_source_onnx_sha256,
            serialized_sha256
          );
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
    bool has_artistic_outputs = false;
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
      has_artistic_outputs = slot.has_artistic_outputs;
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
      if (!warmup_execution_context(cuda, cuda_ctx, exec_context, has_artistic_outputs)) {
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
    float ema_edge_strength;
    int depth_short_side;  // depth map short-side resolution (clamped to native short side)
    float max_aspect;  // aspect cap for short-side mode
    float minmax_alpha;  // temporal EMA blend for the normalized min/max
    bool cuda_graph_enabled;
    bool has_artistic_outputs = false;
    bool consume_artistic_policy;
    artistic_policy_authorization artistic_authorization;
    float artistic_scale_override;
    std::string artistic_model_onnx_sha256;
    std::string artistic_policy_metadata_sha256;
    std::string artistic_geometry_allowlist_sha256;
    nlohmann::json artistic_geometry_allowlist;
    bool artistic_geometry_validated = false;
    bool artistic_geometry_matched = false;
    std::atomic_bool artistic_policy_consumed_once {false};
    std::uint32_t artistic_eye_width = 0;
    std::uint32_t artistic_eye_height = 0;
    float artistic_content_scale_x = 0.0f;
    float artistic_content_scale_y = 0.0f;
    float artistic_style_mix;  // 0 clean, 0.5 balanced, 1 immersive
    CUgraph inference_graph = nullptr;
    CUgraphExec inference_graph_exec = nullptr;
    CUdeviceptr graph_input = 0;
    CUdeviceptr graph_output = 0;
    CUdeviceptr graph_artistic_output = 0;
    int graph_width = 0;
    int graph_height = 0;
    bool graph_signature_warmed = false;
    bool graph_capture_failed = false;
    bool valid = false;  // all mandatory engine, shader, and session resources are ready
    float subject_recenter;  // recenter strength consumed by depth_subject_resolve_cs
    bool subject_stretch;  // apply the shape_depth_for_pop 5/95 disparity stretch
    bool adaptive_pop;
    float adaptive_pop_max_ratio;
    float zero_plane_mode;  // 0 legacy, 1 subject, 2 median depth, 3 far/mid-background
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

    bool enqueue_inference(
      CUdeviceptr input,
      CUdeviceptr output,
      CUdeviceptr artistic_output,
      cuda_driver_api &cuda
    ) {
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
      if (input != graph_input || output != graph_output ||
          artistic_output != graph_artistic_output ||
          target_w != graph_width || target_h != graph_height) {
        destroy_inference_graph(cuda);
        graph_input = input;
        graph_output = output;
        graph_artistic_output = artistic_output;
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
    depth_output_sizes output_sizes;
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
    Microsoft::WRL::ComPtr<ID3D11SamplerState> linear_sampler;
    Microsoft::WRL::ComPtr<ID3D11Buffer> cbuffer;

    Microsoft::WRL::ComPtr<ID3D11Buffer> tensor_in_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> tensor_in_uav;

    Microsoft::WRL::ComPtr<ID3D11Buffer> tensor_out_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> tensor_out_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> artistic_global_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> artistic_global_srv;

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
    Microsoft::WRL::ComPtr<ID3D11Buffer> subject_plain_buf;  // 258 uints: 256 bins + edge/change counters
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subject_plain_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subject_buf;  // three float4 elements; see depth_subject_resolve_cs
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subject_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subject_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subject_stage;  // CPU-readable copy for the debug log
    unsigned subject_log_counter = 0;  // paces the [SUBJDBG] readback (every 24 depth updates)
    runtime_scene_evidence evaluation_last_scene_evidence;

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
    CUgraphicsResource cuda_artistic_res = nullptr;
    bool has_previous_frame = false;
    std::uint64_t pending_frame_id = 0;
    input_color_space pending_input_color_space = input_color_space::srgb;
    DXGI_FORMAT pending_input_format = DXGI_FORMAT_UNKNOWN;
    bool pending_artistic_geometry_matched = false;
    bool stream_error_logged = false;
    bool depth_context_pooled = false;  // context reused from the pool (modules already loaded -> skip warmup)

    bool compile_shader(const std::filesystem::path &path, Microsoft::WRL::ComPtr<ID3D11ComputeShader> &out_cs) {
      std::vector<std::uint8_t> bytecode;
      if (!depth_shader_bytecode(path, bytecode)) {
        return false;
      }
      return SUCCEEDED(device->CreateComputeShader(bytecode.data(), bytecode.size(), nullptr, &out_cs));
    }

    impl(Microsoft::WRL::ComPtr<ID3D11Device> d, Microsoft::WRL::ComPtr<ID3D11DeviceContext> c, const std::filesystem::path &assets_dir, const config::video_t::sbs_t &cfg, const config::depth_model_info &model, bool consume_policy, float scale_override, artistic_policy_authorization authorization):
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
        consume_artistic_policy(consume_policy),
        artistic_authorization(authorization),
        artistic_scale_override(scale_override),
        artistic_style_mix(cfg.artistic_style == "clean" ? 0.0f :
                           cfg.artistic_style == "balanced" ? 0.5f : 1.0f),
        subject_recenter((float) cfg.subject_recenter),
        subject_stretch(cfg.subject_stretch),
        adaptive_pop(cfg.adaptive_pop),
        adaptive_pop_max_ratio((float) (std::max(cfg.adaptive_pop_max, cfg.pop_strength) /
                                        std::max(cfg.pop_strength, 0.25))),
        zero_plane_mode(cfg.zero_plane == "subject" ? 1.0f :
                        cfg.zero_plane == "median" ? 2.0f :
                        cfg.zero_plane == "background" ? 3.0f : 0.0f),
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
      std::string resident_engine_sha256;

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
        } else {
          has_artistic_outputs = slot.has_artistic_outputs;
          resident_engine_sha256 = slot.engine_sha256;
        }

        trt_mutex = &g_trt_mutex;
      }  // release g_trt_mutex before the shader/buffer setup and warmup below

      if (engine && has_artistic_outputs) {
        std::string policy_reason;
        validated_artistic_policy_metadata_t validated_policy;
        const bool policy_metadata_valid = validate_artistic_policy_metadata(
          assets_dir,
          model_path,
          model,
          cfg,
          resident_engine_sha256,
          artistic_authorization,
          validated_policy,
          policy_reason
        );
        if (!policy_metadata_valid) {
          consume_artistic_policy = false;
          if (consume_policy) {
            BOOST_LOG(warning) << "Learned artistic policy disabled (" << policy_reason
                               << "); normal depth inference remains enabled.";
          } else {
            BOOST_LOG(info) << "Ignoring untrusted artistic output (" << policy_reason
                            << "); this run requested depth-only inference.";
          }
        } else if (consume_policy) {
          artistic_model_onnx_sha256 = std::move(validated_policy.onnx_sha256);
          artistic_policy_metadata_sha256 = std::move(validated_policy.metadata_sha256);
          artistic_geometry_allowlist_sha256 =
            std::move(validated_policy.deployment_geometry_allowlist_sha256);
          artistic_geometry_allowlist =
            std::move(validated_policy.deployment_geometry_allowlist);
          BOOST_LOG(info) << "Learned artistic policy metadata, ONNX/engine provenance, and "
                             "resolved SBS baseline validated under "
                          << (artistic_authorization ==
                                  artistic_policy_authorization::candidate_evaluation ?
                                "offline candidate-evaluation authorization" :
                              artistic_authorization ==
                                  artistic_policy_authorization::headset_review ?
                                "explicit live headset-review authorization" :
                                "final live-deployment authorization")
                          << "; awaiting an exact live geometry match.";
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
      BOOST_LOG(info) << "SBS zero-plane mode: " << cfg.zero_plane
                      << (zero_plane_mode > 0.5f ? " (shot-latched experimental anchor)." : ".");
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

      // Subject tracking: weighted histogram (256 uint bins), plain histogram plus two scene-risk
      // counters (258 uints), and three-float4 per-frame state.
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
        uint32_t init_plain[258] = {};
        bd.ByteWidth = sizeof(init_plain);
        D3D11_SUBRESOURCE_DATA plain_sd = {init_plain, 0, 0};
        device->CreateBuffer(&bd, &plain_sd, &subject_plain_buf);
        if (subject_plain_buf) {
          device->CreateUnorderedAccessView(subject_plain_buf.Get(), nullptr, &subject_plain_uav);
        }

        // [0] subject/recenter, [1] stretch/convergence/pop, [2] explicit zero-plane anchor.
        float init_state[12] = {0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f};
        bd.ByteWidth = sizeof(init_state);
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        bd.StructureByteStride = sizeof(float) * 4;
        D3D11_SUBRESOURCE_DATA sd2 = {init_state, 0, 0};
        device->CreateBuffer(&bd, &sd2, &subject_buf);
        if (subject_buf) {
          device->CreateUnorderedAccessView(subject_buf.Get(), nullptr, &subject_uav);
          device->CreateShaderResourceView(subject_buf.Get(), nullptr, &subject_srv);
          // Staging copy for explicit evaluator scene evidence. The live
          // capture path never maps it because GPU->CPU readback synchronizes.
          D3D11_BUFFER_DESC stg = {};
          stg.Usage = D3D11_USAGE_STAGING;
          stg.ByteWidth = sizeof(init_state);
          stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
          // CPU-readable staging buffers cannot carry D3D11 misc flags. The
          // structure description belongs only to the GPU source buffer.
          const HRESULT stage_hr = device->CreateBuffer(
            &stg, nullptr, &subject_stage
          );
          if (FAILED(stage_hr)) {
            BOOST_LOG(error) << "Depth estimator failed to create subject-state "
                                "readback buffer (HRESULT 0x"
                             << std::hex
                             << static_cast<unsigned long>(stage_hr) << ").";
          }
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
      if (!valid) {
        return;
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
      depth_output_sizes warmup_sizes;
      if (!set_depth_shape_and_validate_outputs(
            exec_context,
            h,
            w,
            has_artistic_outputs,
            warmup_sizes
          )) {
        BOOST_LOG(error) << "Depth estimator warmup rejected incompatible output shapes.";
        valid = false;
        return;
      }

      const size_t in_elems = (size_t) 3 * h * w;  // batch/view dims are 1 for rank-4 and rank-5
      CUdeviceptr d_in = 0, d_out = 0;
      CUdeviceptr d_artistic_global = 0;
      if (cuda.cuMemAlloc(&d_in, in_elems * sizeof(float)) != 0) {
        return;
      }
      if (cuda.cuMemAlloc(&d_out, warmup_sizes.depth) != 0) {
        cuda.cuMemFree(d_in);
        return;
      }
      if (has_artistic_outputs && cuda.cuMemAlloc(&d_artistic_global, warmup_sizes.artistic_global) != CUDA_SUCCESS) {
        cuda.cuMemFree(d_in);
        cuda.cuMemFree(d_out);
        return;
      }

      bool bound = exec_context->setTensorAddress("pixel_values", (void *) d_in) &&
                   exec_context->setTensorAddress("predicted_depth", (void *) d_out);
      if (has_artistic_outputs) {
        bound = bound &&
                exec_context->setTensorAddress("artistic_global", (void *) d_artistic_global);
      }
      bool ok = false;
      if (bound) {
        std::lock_guard<std::mutex> lock(*trt_mutex);
        ok = exec_context->enqueueV3(cu_stream);
      }
      if (ok && cuda.cuStreamSynchronize) {
        cuda.cuStreamSynchronize(cu_stream);
      }
      cuda.cuMemFree(d_in);
      cuda.cuMemFree(d_out);
      if (d_artistic_global) {
        cuda.cuMemFree(d_artistic_global);
      }
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
        if (cuda_artistic_res) {
          cuda.cuGraphicsUnregisterResource(cuda_artistic_res);
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

    estimate_result make_result(bool completed_frame_valid = false, std::uint64_t completed_frame_id = 0, bool inference_enqueued = false, std::uint64_t enqueued_frame_id = 0) {
      estimate_result r;
      r.depth = output_srv();
      r.subject = subject_srv;
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

    runtime_scene_evidence read_runtime_scene_evidence(std::uint64_t completed_frame_id) {
      if (evaluation_last_scene_evidence.valid) {
        if (completed_frame_id == evaluation_last_scene_evidence.completed_frame_id) {
          return evaluation_last_scene_evidence;
        }
        if (completed_frame_id !=
            evaluation_last_scene_evidence.completed_frame_id + 1) {
          return {};
        }
      } else if (completed_frame_id != 0) {
        return {};
      }
      if (!subject_stage || !subject_buf || !context) {
        return {};
      }

      // This CopyResource+Map intentionally synchronizes the GPU. It is exposed only through the
      // explicitly named evaluation API below and is never called by the live capture path.
      context->CopyResource(subject_stage.Get(), subject_buf.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(subject_stage.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return {};
      }
      const float *subject = static_cast<const float *>(mapped.pData);
      const float scene_age = subject[1];
      const bool initialized = subject[3] > 0.5f;
      std::array<float, 12> subject_state {};
      std::copy_n(subject, subject_state.size(), subject_state.begin());
      context->Unmap(subject_stage.Get(), 0);
      if (!std::isfinite(scene_age) || scene_age < 0.0f ||
          !std::all_of(subject_state.begin(), subject_state.end(), [](float value) {
            return std::isfinite(value);
          })) {
        return {};
      }

      // The shader computes candidate_age = prior_age + 1 and accepts a hard cut only when that
      // candidate is >= 8, then writes age zero. Thus prior age >= 7 -> current age zero is the
      // exact observable reset transition when every completed depth frame is read.
      const bool hard_cut = evaluation_last_scene_evidence.valid &&
                            evaluation_last_scene_evidence.subject_initialized && initialized &&
                            evaluation_last_scene_evidence.scene_age >= 7.0f &&
                            scene_age < 0.5f;
      runtime_scene_evidence evidence;
      evidence.valid = true;
      evidence.completed_frame_id = completed_frame_id;
      evidence.runtime_scene_id = evaluation_last_scene_evidence.valid ?
                                    evaluation_last_scene_evidence.runtime_scene_id : 0;
      if (hard_cut) {
        ++evidence.runtime_scene_id;
      }
      evidence.scene_age = scene_age;
      evidence.subject_initialized = initialized;
      evidence.hard_cut = hard_cut;
      evidence.scene_start = !evaluation_last_scene_evidence.valid || hard_cut;
      evidence.subject_state = subject_state;
      evaluation_last_scene_evidence = evidence;
      return evidence;
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
      const bool pending_input_contract_exact = !artistic_live_color_mode(
        pending_input_format, pending_input_color_space
      ).empty();
      if (consume_artistic_policy &&
          (color_space != pending_input_color_space || !pending_artistic_geometry_matched ||
           !pending_input_contract_exact)) {
        consume_artistic_policy = false;
        artistic_geometry_validated = true;
        artistic_geometry_matched = false;
        artistic_policy_consumed_once.store(false, std::memory_order_release);
        cb_color_mode = -1;
        cbuffer.Reset();
        if (subject_buf) {
          const float reset_subject_state[12] = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
          };
          context->UpdateSubresource(subject_buf.Get(), 0, nullptr, reset_subject_state, 0, 0);
        }
        BOOST_LOG(warning) << "Learned artistic policy disabled because evaluation completion "
                              "did not match its pending geometry/color submission contract.";
      }
      // The pending inference was preprocessed under this exact color contract. Never let the
      // completion caller reinterpret it with an independent enum.
      color_space = pending_input_color_space;
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
      pending_input_format = DXGI_FORMAT_UNKNOWN;
      pending_artistic_geometry_matched = false;
      return make_result(true, completed_frame_id);
    }

    bool artistic_live_geometry_matches(
      const D3D11_TEXTURE2D_DESC &input_desc,
      input_color_space color_space
    ) const {
      const std::string_view live_color_mode = artistic_live_color_mode(
        input_desc.Format, color_space
      );
      if (live_color_mode.empty() || !artistic_geometry_allowlist.is_object()) {
        return false;
      }
      const auto tuples = artistic_geometry_allowlist.find("tuples");
      if (tuples == artistic_geometry_allowlist.end() || !tuples->is_array()) {
        return false;
      }
      for (const auto &tuple : *tuples) {
        if (tuple.value("source_width", 0u) != input_desc.Width ||
            tuple.value("source_height", 0u) != input_desc.Height ||
            tuple.value("model_input_width", 0) != target_w ||
            tuple.value("model_input_height", 0) != target_h ||
            tuple.value("eye_width", 0u) != artistic_eye_width ||
            tuple.value("eye_height", 0u) != artistic_eye_height ||
            tuple.value("disparity_raster_width", 0u) != artistic_eye_width ||
            tuple.value("disparity_raster_height", 0u) != artistic_eye_height ||
            tuple.value("color_mode", "") != live_color_mode) {
          continue;
        }
        const double scale_x = tuple.value("content_scale_x", 0.0);
        const double scale_y = tuple.value("content_scale_y", 0.0);
        if (std::abs(scale_x - artistic_content_scale_x) <= 1e-7 &&
            std::abs(scale_y - artistic_content_scale_y) <= 1e-7) {
          return true;
        }
      }
      return false;
    }

    void validate_artistic_live_geometry(
      const D3D11_TEXTURE2D_DESC &input_desc,
      input_color_space color_space
    ) {
      if (!has_artistic_outputs || !consume_artistic_policy ||
          artistic_scale_override > 0.0f) {
        return;
      }
      const bool matched = artistic_live_geometry_matches(input_desc, color_space);
      if (matched) {
        if (!artistic_geometry_validated) {
          BOOST_LOG(info) << "Learned artistic policy enabled for approved live geometry "
                          << input_desc.Width << 'x' << input_desc.Height << " -> "
                          << target_w << 'x' << target_h << " model -> "
                          << artistic_eye_width << 'x' << artistic_eye_height << " per eye.";
        }
        artistic_geometry_validated = true;
        artistic_geometry_matched = true;
        return;
      }

      consume_artistic_policy = false;
      artistic_geometry_validated = true;
      artistic_geometry_matched = false;
      artistic_policy_consumed_once.store(false, std::memory_order_release);
      cb_color_mode = -1;
      cbuffer.Reset();
      if (subject_buf) {
        // Remove the previously latched multiplier immediately. Waiting for another completed
        // inference would leak the old camera action through busy-drop frames after a mode change.
        const float reset_subject_state[12] = {
          0.0f, 0.0f, 0.0f, 0.0f,
          0.0f, 1.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f, 0.0f,
        };
        context->UpdateSubresource(subject_buf.Get(), 0, nullptr, reset_subject_state, 0, 0);
      }
      BOOST_LOG(warning) << "Learned artistic policy disabled because live geometry/color is not "
                            "in its exact deployment allow-list (source "
                         << input_desc.Width << 'x' << input_desc.Height << ", model "
                         << target_w << 'x' << target_h << ", eye " << artistic_eye_width << 'x'
                         << artistic_eye_height << ", content scale " << artistic_content_scale_x
                         << 'x' << artistic_content_scale_y << ", color mode "
                         << (int) color_space << ", DXGI format " << (int) input_desc.Format
                         << "); normal depth inference remains enabled.";
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
      // 0 = disabled, 1 = learned safe ceiling, 2 = harness exact-scale override.
      // Slot 15 is a style mix in learned mode and an absolute scale in override mode.
      cbf[14] = artistic_scale_override > 0.0f ? 2.0f :
                (has_artistic_outputs && consume_artistic_policy ? 1.0f : 0.0f);
      cbf[15] = artistic_scale_override > 0.0f ?
                  artistic_scale_override : artistic_style_mix;
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

      // Snapshot the complete previous depth before any thread writes the new result.
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
        ID3D11ShaderResourceView *subject_srvs[2] = {depth_srv.Get(), depth_previous_srv.Get()};
        context->CSSetShaderResources(0, 2, subject_srvs);
        ID3D11UnorderedAccessView *hist_uavs[2] = {subject_hist_uav.Get(), subject_plain_uav.Get()};
        context->CSSetUnorderedAccessViews(0, 2, hist_uavs, nullptr);
        context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

        ID3D11UnorderedAccessView *null_uavs_h2[2] = {nullptr, nullptr};
        context->CSSetUnorderedAccessViews(0, 2, null_uavs_h2, nullptr);
        ID3D11ShaderResourceView *null_subject_srvs[2] = {nullptr, nullptr};
        context->CSSetShaderResources(0, 2, null_subject_srvs);

        context->CSSetShader(depth_subject_resolve_cs.Get(), nullptr, 0);
        ID3D11ShaderResourceView *resolve_srvs[1] = {artistic_global_srv.Get()};
        context->CSSetShaderResources(0, 1, resolve_srvs);
        ID3D11UnorderedAccessView *subj_uavs[3] = {subject_hist_uav.Get(), subject_uav.Get(), subject_plain_uav.Get()};
        context->CSSetUnorderedAccessViews(0, 3, subj_uavs, nullptr);
        context->Dispatch(1, 1, 1);
        if (consume_artistic_policy && artistic_geometry_matched && artistic_global_srv) {
          // This is the first point at which the accepted artistic output has actually entered
          // the warp state. Metadata/config validation alone is not consumption evidence.
          artistic_policy_consumed_once.store(true, std::memory_order_release);
        }

        ID3D11UnorderedAccessView *null_uavs2[3] = {nullptr, nullptr, nullptr};
        ID3D11ShaderResourceView *null_resolve_srv = nullptr;
        context->CSSetUnorderedAccessViews(0, 3, null_uavs2, nullptr);
        context->CSSetShaderResources(0, 1, &null_resolve_srv);

        // Ground-truth log for the original Bestv2 reference's LOW=near convention. Apollo is
        // HIGH=near, so print both subject values for direct comparison. Opt-in via APOLLO_SUBJDBG (NOT
        // perf-gated: CopyResource+Map is a CPU/GPU sync that would perturb the very
        // perf numbers a benchmark run measures), every 24 updates, off the ship path.
        static const bool subjdbg = std::getenv("APOLLO_SUBJDBG") != nullptr;
        if (subjdbg && subject_stage && (++subject_log_counter % 24u) == 1u) {
          context->CopyResource(subject_stage.Get(), subject_buf.Get());
          D3D11_MAPPED_SUBRESOURCE ms {};
          if (SUCCEEDED(context->Map(subject_stage.Get(), 0, D3D11_MAP_READ, 0, &ms))) {
            const float *s = (const float *) ms.pData;
            BOOST_LOG(info) << "[SUBJDBG] u=" << subject_log_counter
                            << " subj_hi_near=" << s[2]
                            << " subj_low_near=" << (1.0f - s[2])
                            << " recenter_delta=" << s[0]
                            << " scene_age=" << s[1]
                            << " init=" << s[3]
                            << " zero_anchor_shift_px=" << s[8]
                            << " zero_anchor_valid=" << s[9];
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

      D3D11_TEXTURE2D_DESC input_desc = {0};
      Microsoft::WRL::ComPtr<ID3D11Resource> input_res;
      input_srv->GetResource(&input_res);
      Microsoft::WRL::ComPtr<ID3D11Texture2D> input_tex;
      if (SUCCEEDED(input_res.As(&input_tex))) {
        input_tex->GetDesc(&input_desc);
      }

      // A mode/color transition must disable an out-of-domain learned action even while the
      // prior inference is still busy. Otherwise the busy-drop path could keep reusing a stale
      // artistic multiplier from the previous geometry.
      if (target_w > 0 && target_h > 0) {
        validate_artistic_live_geometry(input_desc, color_space);
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

        if (!set_depth_shape_and_validate_outputs(
              exec_context,
              target_h,
              target_w,
              has_artistic_outputs,
              output_sizes
            )) {
          BOOST_LOG(error) << "Depth estimator rejected incompatible dynamic output shapes.";
          valid = false;
          target_w = target_h = 0;
          return {};
        }

        if (cuda_in_res) {
          cuda.cuGraphicsUnregisterResource(cuda_in_res);
        }
        if (cuda_out_res) {
          cuda.cuGraphicsUnregisterResource(cuda_out_res);
        }
        if (cuda_artistic_res) {
          cuda.cuGraphicsUnregisterResource(cuda_artistic_res);
        }
        cuda_in_res = nullptr;
        cuda_out_res = nullptr;
        cuda_artistic_res = nullptr;
        artistic_global_buf.Reset();
        artistic_global_srv.Reset();

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

        buf_desc.ByteWidth = (UINT) output_sizes.depth;
        buf_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateBuffer(&buf_desc, nullptr, &tensor_out_buf)) &&
                       SUCCEEDED(device->CreateShaderResourceView(
                         tensor_out_buf.Get(),
                         nullptr,
                         &tensor_out_srv
                       ));

        if (has_artistic_outputs) {
          buf_desc.ByteWidth = (UINT) output_sizes.artistic_global;
          resources_ok = resources_ok &&
                         SUCCEEDED(device->CreateBuffer(&buf_desc, nullptr, &artistic_global_buf)) &&
                         SUCCEEDED(device->CreateShaderResourceView(
                           artistic_global_buf.Get(),
                           nullptr,
                           &artistic_global_srv
                         ));
        }

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

        auto mask_desc = tex_desc;
        mask_desc.Format = DXGI_FORMAT_R32_UINT;
        mask_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateTexture2D(&mask_desc, nullptr, &ema_motion_mask_tex)) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(ema_motion_mask_tex.Get(), nullptr, &ema_motion_mask_uav)) &&
                       SUCCEEDED(device->CreateShaderResourceView(ema_motion_mask_tex.Get(), nullptr, &ema_motion_mask_srv));

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

        auto res1 = cuda.cuGraphicsD3D11RegisterResource(&cuda_in_res, tensor_in_buf.Get(), 0);
        auto res2 = cuda.cuGraphicsD3D11RegisterResource(&cuda_out_res, tensor_out_buf.Get(), 0);
        auto res3 = has_artistic_outputs ?
                      cuda.cuGraphicsD3D11RegisterResource(
                        &cuda_artistic_res, artistic_global_buf.Get(), 0
                      ) : CUDA_SUCCESS;
        if (res1 != 0 || res2 != 0 || res3 != 0) {
          BOOST_LOG(error) << "cuGraphicsD3D11RegisterResource failed: "
                           << res1 << ", " << res2 << ", " << res3;
          if (cuda_in_res) {
            cuda.cuGraphicsUnregisterResource(cuda_in_res);
          }
          if (cuda_out_res) {
            cuda.cuGraphicsUnregisterResource(cuda_out_res);
          }
          if (cuda_artistic_res) {
            cuda.cuGraphicsUnregisterResource(cuda_artistic_res);
          }
          cuda_in_res = nullptr;
          cuda_out_res = nullptr;
          cuda_artistic_res = nullptr;
          target_w = target_h = 0;
          return {};
        }
      }

      // The global policy head cannot observe source/output geometry. Prove the exact live tuple
      // before the immutable depth constants can enable it, and keep checking so HDR or mode
      // transitions fail closed instead of silently applying an out-of-domain camera action.
      validate_artistic_live_geometry(input_desc, color_space);

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
      CUgraphicsResource resources[3] = {
        cuda_in_res, cuda_out_res, cuda_artistic_res
      };
      const unsigned resource_count = has_artistic_outputs ? 3u : 2u;
      auto map_res = cuda.cuGraphicsMapResources(resource_count, resources, cu_stream);
      if (map_res != 0) {
        BOOST_LOG(error) << "cuGraphicsMapResources failed: " << map_res;
        return make_result(completed_frame_valid, completed_frame_id);
      }

      void *d_in = nullptr;
      void *d_out = nullptr;
      void *d_artistic_global = nullptr;
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
      auto artistic_ptr_res = has_artistic_outputs ?
                                cuda.cuGraphicsResourceGetMappedPointer(
                                  (CUdeviceptr *) &d_artistic_global,
                                  nullptr,
                                  cuda_artistic_res
                                ) : CUDA_SUCCESS;

      bool enqueued = false;
      if (in_ptr_res != CUDA_SUCCESS || out_ptr_res != CUDA_SUCCESS ||
          artistic_ptr_res != CUDA_SUCCESS || !d_in || !d_out ||
          (has_artistic_outputs && !d_artistic_global)) {
        BOOST_LOG(error) << "Failed to get mapped pointer for TensorRT: "
                         << in_ptr_res << ", " << out_ptr_res << ", "
                         << artistic_ptr_res;
      } else {
        bool bindings_ok = exec_context->setTensorAddress("pixel_values", (void *) d_in) &&
                           exec_context->setTensorAddress("predicted_depth", (void *) d_out);
        if (has_artistic_outputs) {
          bindings_ok = bindings_ok &&
                        exec_context->setTensorAddress("artistic_global", d_artistic_global);
        }
        if (bindings_ok) {
          // Serialize TensorRT async enqueue to avoid driver-level concurrent execution faults
          std::lock_guard<std::mutex> lock(*trt_mutex);
          int perf_slot = perf_begin(perf_depth, cu_stream);
          enqueued = enqueue_inference(
            (CUdeviceptr) d_in,
            (CUdeviceptr) d_out,
            (CUdeviceptr) d_artistic_global,
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

      auto unmap_res = cuda.cuGraphicsUnmapResources(resource_count, resources, cu_stream);
      if (unmap_res != CUDA_SUCCESS) {
        BOOST_LOG(error) << "cuGraphicsUnmapResources failed: " << unmap_res;
        enqueued = false;
      }

      has_previous_frame = enqueued;
      if (enqueued) {
        pending_frame_id = frame_id;
        pending_input_color_space = color_space;
        pending_input_format = input_desc.Format;
        pending_artistic_geometry_matched = artistic_geometry_matched;
        throughput_stats_enqueues++;
      }

      return make_result(completed_frame_valid, completed_frame_id, enqueued, enqueued ? frame_id : 0);
    }
  };

  video_depth_estimator::video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device, Microsoft::WRL::ComPtr<ID3D11DeviceContext> context, const std::filesystem::path &assets_dir, const config::video_t::sbs_t &cfg, const config::depth_model_info &model, bool consume_artistic_policy, float artistic_scale_override, artistic_policy_authorization authorization):
      pimpl(std::make_unique<impl>(device, context, assets_dir, cfg, model, consume_artistic_policy, artistic_scale_override, authorization)) {}

  video_depth_estimator::~video_depth_estimator() = default;

  bool video_depth_estimator::is_valid() const {
    return pimpl && pimpl->valid;
  }

  artistic_policy_provenance video_depth_estimator::artistic_policy_status() const {
    if (!pimpl || !pimpl->has_artistic_outputs || !pimpl->consume_artistic_policy ||
        !pimpl->artistic_geometry_validated || !pimpl->artistic_geometry_matched ||
        !pimpl->artistic_policy_consumed_once.load(std::memory_order_acquire) ||
        pimpl->artistic_scale_override > 0.0f) {
      return {};
    }
    return {
      true,
      pimpl->artistic_authorization == artistic_policy_authorization::candidate_evaluation ?
        "candidate-evaluation" :
      pimpl->artistic_authorization == artistic_policy_authorization::headset_review ?
        "headset-review" :
        "deployment",
      pimpl->artistic_model_onnx_sha256,
      pimpl->artistic_policy_metadata_sha256,
      pimpl->artistic_geometry_allowlist_sha256,
    };
  }

  void video_depth_estimator::set_artistic_output_geometry(
    std::uint32_t eye_width,
    std::uint32_t eye_height,
    float content_scale_x,
    float content_scale_y
  ) {
    if (!pimpl) {
      return;
    }
    const bool changed = pimpl->artistic_eye_width != eye_width ||
                         pimpl->artistic_eye_height != eye_height ||
                         std::abs(pimpl->artistic_content_scale_x - content_scale_x) > 1e-7f ||
                         std::abs(pimpl->artistic_content_scale_y - content_scale_y) > 1e-7f;
    if (changed && pimpl->target_w > 0) {
      pimpl->consume_artistic_policy = false;
      pimpl->artistic_geometry_validated = true;
      pimpl->artistic_geometry_matched = false;
      pimpl->artistic_policy_consumed_once.store(false, std::memory_order_release);
      pimpl->cb_color_mode = -1;
      pimpl->cbuffer.Reset();
      if (pimpl->subject_buf) {
        const float reset_subject_state[12] = {
          0.0f, 0.0f, 0.0f, 0.0f,
          0.0f, 1.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f, 0.0f,
        };
        pimpl->context->UpdateSubresource(
          pimpl->subject_buf.Get(), 0, nullptr, reset_subject_state, 0, 0
        );
      }
      BOOST_LOG(warning) << "Learned artistic policy disabled because output geometry changed "
                            "after inference initialization; rebuild the estimator to revalidate.";
    }
    pimpl->artistic_eye_width = eye_width;
    pimpl->artistic_eye_height = eye_height;
    pimpl->artistic_content_scale_x = content_scale_x;
    pimpl->artistic_content_scale_y = content_scale_y;
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

  runtime_scene_evidence video_depth_estimator::read_runtime_scene_evidence_for_evaluation(
    std::uint64_t completed_frame_id
  ) {
    return pimpl ? pimpl->read_runtime_scene_evidence(completed_frame_id) :
                   runtime_scene_evidence {};
  }
}  // namespace models
