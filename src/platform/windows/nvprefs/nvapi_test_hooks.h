/**
 * @file src/platform/windows/nvprefs/nvapi_test_hooks.h
 * @brief Private loader seam for tests of the real NVAPI wrapper.
 */
#pragma once

#ifdef SUNSHINE_TESTS
  #include <Windows.h>

namespace nvprefs::detail {
  struct nvapi_loader_api_t {
    decltype(&::LoadLibraryExA) load_library {};
    decltype(&::GetProcAddress) get_proc_address {};
    decltype(&::FreeLibrary) free_library {};
  };

  // Never replace the loader that owns a currently loaded module.
  bool set_nvapi_loader_for_test(const nvapi_loader_api_t &api);
  bool reset_nvapi_loader_for_test();
}  // namespace nvprefs::detail
#endif
