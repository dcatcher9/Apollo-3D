/** Checked CUDA device properties used only when selecting/building a TensorRT engine. */
#pragma once

#include "cuda_driver_api.h"

#include <algorithm>
#include <array>
#include <string>

namespace models::detail {

  struct cuda_engine_attribute_t {
    CUdevice_attribute attribute;
    const char *name;
    int minimum;
  };

  // TensorRT's documented compatibility checks, plus the opt-in shared-memory limit used by
  // modern kernels. Total/free memory is deliberately absent: OS reservation changes alone do
  // not establish engine incompatibility and must not cause automatic recompilation.
  inline constexpr std::array cuda_engine_attributes {
    cuda_engine_attribute_t {CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "sm-major", 1},
    cuda_engine_attribute_t {CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "sm-minor", 0},
    cuda_engine_attribute_t {CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, "memory-bus-bits", 1},
    cuda_engine_attribute_t {CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, "l2-bytes", 0},
    cuda_engine_attribute_t {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, "shared-block-bytes", 1},
    cuda_engine_attribute_t {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "shared-sm-bytes", 1},
    cuda_engine_attribute_t {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, "shared-optin-bytes", 1},
    cuda_engine_attribute_t {CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, "texture-alignment", 1},
    cuda_engine_attribute_t {CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "multiprocessors", 1},
    cuda_engine_attribute_t {CU_DEVICE_ATTRIBUTE_INTEGRATED, "integrated", 0},
  };

  // Public scalar/pointer CUDA calls avoid a CUDA Toolkit dependency and struct-return ABI issues.
  // No ordinal fallback, free-memory query, clock sampling, or per-frame work belongs in this key.
  inline std::string cuda_engine_device_identity(
    const cuda_driver_api &cuda, CUdevice device, std::string &error
  ) {
    error.clear();
    if (!cuda.cuDeviceGetName || !cuda.cuDriverGetVersion || !cuda.cuDeviceGetAttribute) {
      error = "required CUDA device-identity query is unavailable";
      return {};
    }
    const auto failed = [&](const char *query, CUresult status) {
      error = std::string {query} + " failed (CUDA " +
              std::to_string(static_cast<int>(status)) + ")";
      return std::string {};
    };
    std::array<char, 256> name {};
    auto status = cuda.cuDeviceGetName(name.data(), static_cast<int>(name.size()), device);
    if (status != CUDA_SUCCESS) {
      return failed("cuDeviceGetName", status);
    }
    const auto end = std::find(name.begin(), name.end(), '\0');
    if (end == name.begin() || end == name.end()) {
      error = "cuDeviceGetName returned an empty or unterminated name";
      return {};
    }
    int driver_api_version = 0;
    status = cuda.cuDriverGetVersion(&driver_api_version);
    if (status != CUDA_SUCCESS) {
      return failed("cuDriverGetVersion", status);
    }
    if (driver_api_version <= 0) {
      error = "CUDA returned invalid supported driver API version";
      return {};
    }
    // cuDriverGetVersion is the supported CUDA API major/minor, NOT the Windows package version.
    // Changes to driver-reported properties invalidate the key even if that API version is stable.
    std::string identity = "cuda-device-v1|api=" + std::to_string(driver_api_version);
    for (const auto &entry : cuda_engine_attributes) {
      int value = -1;
      status = cuda.cuDeviceGetAttribute(&value, entry.attribute, device);
      if (status != CUDA_SUCCESS) {
        return failed(entry.name, status);
      }
      if (value < entry.minimum ||
          (entry.attribute == CU_DEVICE_ATTRIBUTE_INTEGRATED && value > 1)) {
        error = std::string {"CUDA returned invalid "} + entry.name;
        return {};
      }
      identity += '|' + std::string {entry.name} + '=' + std::to_string(value);
    }
    const std::string device_name(name.begin(), end);
    return identity + "|name=" + std::to_string(device_name.size()) + ':' + device_name;
  }

}  // namespace models::detail
