/**
 * @file src/nvenc/nvenc_d3d11.cpp
 * @brief Definitions for abstract Direct3D11 NVENC encoder.
 */
// local includes
#include "src/logging.h"

#ifdef _WIN32
  #include "nvenc_d3d11.h"

namespace nvenc {

  nvenc_d3d11::nvenc_d3d11(NV_ENC_DEVICE_TYPE device_type):
      nvenc_base(device_type) {
    async_event_handle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  }

  nvenc_d3d11::~nvenc_d3d11() {
    if (dll) {
      FreeLibrary(dll);
      dll = nullptr;
    }
    if (async_event_handle) {
      CloseHandle(async_event_handle);
    }
  }

  bool nvenc_d3d11::init_library() {
    if (dll) {
      return true;
    }

  #ifdef _WIN64
    constexpr auto dll_name = "nvEncodeAPI64.dll";
  #else
    constexpr auto dll_name = "nvEncodeAPI.dll";
  #endif

    auto unload_library = [this] {
      if (dll) {
        FreeLibrary(dll);
        dll = nullptr;
      }
      return false;
    };

    if ((dll = LoadLibraryEx(dll_name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32))) {
      auto get_max_supported_version = (decltype(NvEncodeAPIGetMaxSupportedVersion) *) GetProcAddress(dll, "NvEncodeAPIGetMaxSupportedVersion");
      if (!get_max_supported_version) {
        BOOST_LOG(error) << "NvEnc: No NvEncodeAPIGetMaxSupportedVersion() in " << dll_name;
        return unload_library();
      }

      uint32_t max_supported_version = 0;
      if (nvenc_failed(get_max_supported_version(&max_supported_version))) {
        BOOST_LOG(error) << "NvEnc: NvEncodeAPIGetMaxSupportedVersion() failed: " << last_nvenc_error_string;
        return unload_library();
      }

      constexpr uint32_t required_version = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
      if (max_supported_version < required_version) {
        BOOST_LOG(error) << "NvEnc: NVIDIA driver supports API " << (max_supported_version >> 4) << '.' << (max_supported_version & 0xF)
                         << ", but Apollo requires API " << NVENCAPI_MAJOR_VERSION << '.' << NVENCAPI_MINOR_VERSION;
        return unload_library();
      }

      if (auto create_instance = (decltype(NvEncodeAPICreateInstance) *) GetProcAddress(dll, "NvEncodeAPICreateInstance")) {
        auto new_nvenc = std::make_unique<NV_ENCODE_API_FUNCTION_LIST>();
        new_nvenc->version = NV_ENCODE_API_FUNCTION_LIST_VER;
        if (nvenc_failed(create_instance(new_nvenc.get()))) {
          BOOST_LOG(error) << "NvEnc: NvEncodeAPICreateInstance() failed: " << last_nvenc_error_string;
        } else {
          nvenc = std::move(new_nvenc);
          return true;
        }
      } else {
        BOOST_LOG(error) << "NvEnc: No NvEncodeAPICreateInstance() in " << dll_name;
      }
    } else {
      BOOST_LOG(debug) << "NvEnc: Couldn't load NvEnc library " << dll_name;
    }

    return unload_library();
  }

  bool nvenc_d3d11::wait_for_async_event(uint32_t timeout_ms) {
    return WaitForSingleObject(async_event_handle, timeout_ms) == WAIT_OBJECT_0;
  }

}  // namespace nvenc
#endif
