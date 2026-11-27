/**
 * @file src/tensorrt_engine_pool.h
 * @brief Process-wide TensorRT runtime, resident engines and pooled execution contexts.
 *
 * Private to the Host SBS depth estimator and its startup preparation. Every caller that touches
 * an engine slot or a pooled context must hold g_trt_mutex, exactly as the `_locked` names say.
 */
#pragma once

#include "config.h"
#include "cuda_driver_api.h"
#include "prod_zipdepth_convex2x.h"
#include "video_depth_estimator.h"

#include <condition_variable>
#include <cstddef>
#include <d3d11.h>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <NvInfer.h>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace models::tensorrt_engine_pool {
  class Logger: public nvinfer1::ILogger {
  public:
#ifdef __GNUC__
    void msvc_dummy_destructor(char flags) noexcept override {}
#endif
    void log(Severity severity, const char *msg) noexcept override;
  };

  extern Logger gLogger;

  extern std::mutex g_trt_mutex;
  extern std::condition_variable g_trt_context_available;
  extern std::mutex g_tensorrt_compile_mutex;
  extern nvinfer1::IRuntime *g_runtime;

  bool ensure_cuda_initialized(cuda_driver_api &cuda);
  CUcontext primary_context(cuda_driver_api &cuda, CUdevice device);
  bool unregister_cuda_graphics_resource(
    cuda_driver_api &cuda,
    CUgraphicsResource &resource
  );
  bool cuda_device_for_d3d(cuda_driver_api &cuda, ID3D11Device *d3d, CUdevice &out);
  std::string engine_compatibility_tag(cuda_driver_api &cuda, CUdevice device);

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
    models::prod_zipdepth_convex2x::engine_io_e io_kind =
      models::prod_zipdepth_convex2x::engine_io_e::invalid;
  };

  extern std::map<std::string, engine_slot> g_engines;  // guarded by g_trt_mutex

  std::size_t allocated_context_count(const engine_slot &slot);
  std::size_t slot_context_limit(const engine_slot &slot);
  void quarantine_execution_context_locked(
    const std::string &engine_key,
    nvinfer1::IExecutionContext *&context,
    const bool was_warmed
  );
  void mark_execution_context_warmed_locked(const std::string &engine_key);
  void return_execution_context_locked(
    const std::string &engine_key,
    nvinfer1::IExecutionContext *&context
  );
  void release_context_reservation_locked(const std::string &engine_key);
  void erase_empty_engine_slot_locked(const std::string &engine_key);
  void recycle_or_quarantine_execution_context_locked(
    const std::string &engine_key,
    nvinfer1::IExecutionContext *&context,
    const bool warmed,
    const bool poisoned
  );

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

  nvinfer1::Dims make_input_dims(int h, int w);
  nvinfer1::ICudaEngine *acquire_engine_locked(
    const std::string &engine_key,
    const std::filesystem::path &engine_path,
    nvinfer1::IExecutionContext *&out_context,
    bool *out_pooled = nullptr
  );
  bool validate_engine_io_locked(nvinfer1::ICudaEngine *engine, engine_slot &slot);
  bool validate_ocr_engine_io_locked(nvinfer1::ICudaEngine *engine, engine_slot &slot);
  bool detach_incompatible_engine_locked(const std::string &engine_key);
  bool warmup_execution_context(
    cuda_driver_api &cuda,
    CUcontext cuda_ctx,
    nvinfer1::IExecutionContext *exec_context
  );
  bool warmup_ocr_execution_context(
    cuda_driver_api &cuda,
    CUcontext cuda_ctx,
    nvinfer1::IExecutionContext *exec_context
  );

  struct engine_artifact {
    std::string name;
    std::string source_sha256;
    std::filesystem::path source_path;
    std::filesystem::path engine_path;
  };

  struct resolved_depth_runtime_t {
    config::depth_model_info model;
  };

  bool resolve_depth_runtime(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &raw_model,
    resolved_depth_runtime_t &resolved
  );
  bool publish_serialized_engine(
    const std::filesystem::path &engine_path,
    const nvinfer1::IHostMemory &serialized,
    const std::string_view description
  );
  bool publish_active_engine_manifest(
    const std::filesystem::path &assets_dir,
    const std::string_view model_name,
    const engine_artifact &artifact,
    const std::optional<prod_zipdepth_convex2x::engine_io_e> depth_io_kind =
      std::nullopt
  );
  bool ensure_tensorrt_engine_for_device(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model,
    cuda_driver_api &cuda,
    CUdevice cuda_device,
    engine_artifact &artifact
  );
  bool ensure_ocr_tensorrt_engine_for_device(
    const std::filesystem::path &assets_dir,
    cuda_driver_api &cuda,
    CUdevice cuda_device,
    engine_artifact &artifact
  );
}  // namespace models::tensorrt_engine_pool
