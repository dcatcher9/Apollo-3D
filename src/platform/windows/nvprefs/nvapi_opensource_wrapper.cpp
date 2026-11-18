/**
 * @file src/platform/windows/nvprefs/nvapi_opensource_wrapper.cpp
 * @brief Definitions for the NVAPI wrapper.
 */
// local includes
#include "driver_settings.h"
#include "nvapi_test_hooks.h"
#include "nvprefs_common.h"

#define SUNSHINE_NVAPI_BINDINGS(X) \
  X(NvAPI_GetErrorMessage, 0x6c2d048c) \
  X(NvAPI_DRS_CreateSession, 0x0694d52e) \
  X(NvAPI_DRS_DestroySession, 0xdad9cff8) \
  X(NvAPI_DRS_LoadSettings, 0x375dbd6b) \
  X(NvAPI_DRS_SaveSettings, 0xfcbc7e14) \
  X(NvAPI_DRS_CreateProfile, 0xcc176068) \
  X(NvAPI_DRS_FindProfileByName, 0x7e4a9a0b) \
  X(NvAPI_DRS_CreateApplication, 0x4347a9de) \
  X(NvAPI_DRS_GetApplicationInfo, 0xed1f8c69) \
  X(NvAPI_DRS_SetSetting, 0x577dd202) \
  X(NvAPI_DRS_GetSetting, 0x73bf8338) \
  X(NvAPI_DRS_DeleteProfileSetting, 0xe4a26362) \
  X(NvAPI_DRS_GetBaseProfile, 0xda8466a0)

namespace {

  struct nvapi_interfaces_t {
#define SUNSHINE_NVAPI_SLOT(name, id) decltype(&::name) name {};
    SUNSHINE_NVAPI_BINDINGS(SUNSHINE_NVAPI_SLOT)
#undef SUNSHINE_NVAPI_SLOT
  };

  nvapi_interfaces_t interfaces;
  HMODULE dll = nullptr;

#ifdef SUNSHINE_TESTS
  auto load_library_api = &::LoadLibraryExA;
  auto get_proc_address_api = &::GetProcAddress;
  auto free_library_api = &::FreeLibrary;
#else
  constexpr auto load_library_api = &::LoadLibraryExA;
  constexpr auto get_proc_address_api = &::GetProcAddress;
  constexpr auto free_library_api = &::FreeLibrary;
#endif

  template<typename Func, typename... Args>
  NvAPI_Status call_interface(Func func, Args... args) {
    if (!func) {
      // The old map lookup also returned NOT_SUPPORTED before initialization.
      return NVAPI_NOT_SUPPORTED;
    }

    return func(args...);
  }

}  // namespace

#ifdef SUNSHINE_TESTS
namespace nvprefs::detail {
  bool set_nvapi_loader_for_test(const nvapi_loader_api_t &api) {
    if (dll || !api.load_library || !api.get_proc_address || !api.free_library) {
      return false;
    }
    load_library_api = api.load_library;
    get_proc_address_api = api.get_proc_address;
    free_library_api = api.free_library;
    return true;
  }

  bool reset_nvapi_loader_for_test() {
    if (dll) {
      return false;
    }
    load_library_api = &::LoadLibraryExA;
    get_proc_address_api = &::GetProcAddress;
    free_library_api = &::FreeLibrary;
    return true;
  }
}  // namespace nvprefs::detail
#endif

#undef NVAPI_INTERFACE
#define NVAPI_INTERFACE NvAPI_Status __cdecl

extern void *__cdecl nvapi_QueryInterface(NvU32 id);

NVAPI_INTERFACE
NvAPI_Initialize() {
  if (dll) {
    return NVAPI_OK;
  }

#ifdef _WIN64
  auto dll_name = "nvapi64.dll";
#else
  auto dll_name = "nvapi.dll";
#endif

  if ((dll = load_library_api(dll_name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32))) {
    if (auto query_interface = (decltype(nvapi_QueryInterface) *) get_proc_address_api(dll, "nvapi_QueryInterface")) {
#define SUNSHINE_NVAPI_RESOLVE(name, id) interfaces.name = reinterpret_cast<decltype(&::name)>(query_interface(id));
      SUNSHINE_NVAPI_BINDINGS(SUNSHINE_NVAPI_RESOLVE)
#undef SUNSHINE_NVAPI_RESOLVE
      return NVAPI_OK;
    }
  }

  NvAPI_Unload();
  return NVAPI_LIBRARY_NOT_FOUND;
}

#undef SUNSHINE_NVAPI_BINDINGS

NVAPI_INTERFACE NvAPI_Unload() {
  if (dll) {
    interfaces = {};
    free_library_api(dll);
    dll = nullptr;
  }
  return NVAPI_OK;
}

NVAPI_INTERFACE NvAPI_GetErrorMessage(NvAPI_Status nr, NvAPI_ShortString szDesc) {
  return call_interface(interfaces.NvAPI_GetErrorMessage, nr, szDesc);
}

// This is only a subset of NvAPI_DRS_* functions, more can be added if needed

NVAPI_INTERFACE NvAPI_DRS_CreateSession(NvDRSSessionHandle *phSession) {
  return call_interface(interfaces.NvAPI_DRS_CreateSession, phSession);
}

NVAPI_INTERFACE NvAPI_DRS_DestroySession(NvDRSSessionHandle hSession) {
  return call_interface(interfaces.NvAPI_DRS_DestroySession, hSession);
}

NVAPI_INTERFACE NvAPI_DRS_LoadSettings(NvDRSSessionHandle hSession) {
  return call_interface(interfaces.NvAPI_DRS_LoadSettings, hSession);
}

NVAPI_INTERFACE NvAPI_DRS_SaveSettings(NvDRSSessionHandle hSession) {
  return call_interface(interfaces.NvAPI_DRS_SaveSettings, hSession);
}

NVAPI_INTERFACE NvAPI_DRS_CreateProfile(NvDRSSessionHandle hSession, NVDRS_PROFILE *pProfileInfo, NvDRSProfileHandle *phProfile) {
  return call_interface(interfaces.NvAPI_DRS_CreateProfile, hSession, pProfileInfo, phProfile);
}

NVAPI_INTERFACE NvAPI_DRS_FindProfileByName(NvDRSSessionHandle hSession, NvAPI_UnicodeString profileName, NvDRSProfileHandle *phProfile) {
  return call_interface(interfaces.NvAPI_DRS_FindProfileByName, hSession, profileName, phProfile);
}

NVAPI_INTERFACE NvAPI_DRS_CreateApplication(NvDRSSessionHandle hSession, NvDRSProfileHandle hProfile, NVDRS_APPLICATION *pApplication) {
  return call_interface(interfaces.NvAPI_DRS_CreateApplication, hSession, hProfile, pApplication);
}

NVAPI_INTERFACE NvAPI_DRS_GetApplicationInfo(NvDRSSessionHandle hSession, NvDRSProfileHandle hProfile, NvAPI_UnicodeString appName, NVDRS_APPLICATION *pApplication) {
  return call_interface(interfaces.NvAPI_DRS_GetApplicationInfo, hSession, hProfile, appName, pApplication);
}

NVAPI_INTERFACE NvAPI_DRS_SetSetting(NvDRSSessionHandle hSession, NvDRSProfileHandle hProfile, NVDRS_SETTING *pSetting) {
  return call_interface(interfaces.NvAPI_DRS_SetSetting, hSession, hProfile, pSetting);
}

NVAPI_INTERFACE NvAPI_DRS_GetSetting(NvDRSSessionHandle hSession, NvDRSProfileHandle hProfile, NvU32 settingId, NVDRS_SETTING *pSetting) {
  return call_interface(interfaces.NvAPI_DRS_GetSetting, hSession, hProfile, settingId, pSetting);
}

NVAPI_INTERFACE NvAPI_DRS_DeleteProfileSetting(NvDRSSessionHandle hSession, NvDRSProfileHandle hProfile, NvU32 settingId) {
  return call_interface(interfaces.NvAPI_DRS_DeleteProfileSetting, hSession, hProfile, settingId);
}

NVAPI_INTERFACE NvAPI_DRS_GetBaseProfile(NvDRSSessionHandle hSession, NvDRSProfileHandle *phProfile) {
  return call_interface(interfaces.NvAPI_DRS_GetBaseProfile, hSession, phProfile);
}
