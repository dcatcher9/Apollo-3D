/**
 * @file src/host_sbs_direct_glyph_engine.h
 * @brief Startup build, validation and warmup of the direct glyph TensorRT engine.
 */
#pragma once

#include "cuda_driver_api.h"

#include <filesystem>
#include <NvInfer.h>
#include <string>

namespace models {
  // Checks out one execution context of the authenticated direct glyph engine, building or
  // repairing its exact cache file when needed, and binds the fixed palette tile shape. A new
  // context is warmed once before it is returned. The caller owns the lease and returns it to
  // the shared pool only after its interop and graph work drains.
  bool acquire_direct_glyph_context(
    const std::filesystem::path &assets_dir,
    cuda_driver_api &cuda,
    CUdevice device,
    nvinfer1::ICudaEngine *&engine,
    nvinfer1::IExecutionContext *&context,
    std::string &engine_key
  );
}  // namespace models
