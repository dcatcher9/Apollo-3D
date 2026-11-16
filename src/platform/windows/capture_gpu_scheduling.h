/**
 * @file src/platform/windows/capture_gpu_scheduling.h
 * @brief Internal GDI32 imports and capture process scheduling setup.
 */
#pragma once

#include <cstddef>
#include <dxgi.h>
#include <windows.h>

namespace platf::dxgi::detail {
  using status_t = LONG;

  typedef enum _D3DKMT_SCHEDULINGPRIORITYCLASS {
    D3DKMT_SCHEDULINGPRIORITYCLASS_IDLE,  ///< Idle priority class
    D3DKMT_SCHEDULINGPRIORITYCLASS_BELOW_NORMAL,  ///< Below normal priority class
    D3DKMT_SCHEDULINGPRIORITYCLASS_NORMAL,  ///< Normal priority class
    D3DKMT_SCHEDULINGPRIORITYCLASS_ABOVE_NORMAL,  ///< Above normal priority class
    D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH,  ///< High priority class
    D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME  ///< Realtime priority class
  } D3DKMT_SCHEDULINGPRIORITYCLASS;

  typedef UINT D3DKMT_HANDLE;

  typedef struct _D3DKMT_OPENADAPTERFROMLUID {
    LUID AdapterLuid;
    D3DKMT_HANDLE hAdapter;
  } D3DKMT_OPENADAPTERFROMLUID;

  typedef struct _D3DKMT_WDDM_2_7_CAPS {
    union {
      struct
      {
        UINT HwSchSupported : 1;
        UINT HwSchEnabled : 1;
        UINT HwSchEnabledByDefault : 1;
        UINT IndependentVidPnVSyncControl : 1;
        UINT Reserved : 28;
      };

      UINT Value;
    };
  } D3DKMT_WDDM_2_7_CAPS;

  typedef struct _D3DKMT_QUERYADAPTERINFO {
    D3DKMT_HANDLE hAdapter;
    UINT Type;
    VOID *pPrivateDriverData;
    UINT PrivateDriverDataSize;
  } D3DKMT_QUERYADAPTERINFO;

  inline constexpr UINT KMTQAITYPE_WDDM_2_7_CAPS = 70;

  typedef struct _D3DKMT_CLOSEADAPTER {
    D3DKMT_HANDLE hAdapter;
  } D3DKMT_CLOSEADAPTER;

  // UCRT does not provide d3dkmthk.h. Keep its documented imports typed against the
  // existing project ABI structs; these exports predate the Windows 11 baseline.
  // https://github.com/microsoft/win32metadata/blob/main/generation/WinSDK/RecompiledIdlHeaders/shared/d3dkmthk.h
  static_assert(sizeof(status_t) == 4 && sizeof(UINT) == 4);
  static_assert(sizeof(D3DKMT_SCHEDULINGPRIORITYCLASS) == 4);
  static_assert(sizeof(D3DKMT_OPENADAPTERFROMLUID) == 12 && offsetof(D3DKMT_OPENADAPTERFROMLUID, hAdapter) == 8);
  static_assert(sizeof(D3DKMT_WDDM_2_7_CAPS) == 4 && sizeof(D3DKMT_CLOSEADAPTER) == 4);
  static_assert(offsetof(D3DKMT_QUERYADAPTERINFO, Type) == 4 && offsetof(D3DKMT_QUERYADAPTERINFO, pPrivateDriverData) == 8);
  static_assert(offsetof(D3DKMT_QUERYADAPTERINFO, PrivateDriverDataSize) == 8 + sizeof(void *));
  static_assert(sizeof(D3DKMT_QUERYADAPTERINFO) == (sizeof(void *) == 8 ? 24 : 16));

  extern "C" {
    DECLSPEC_IMPORT status_t WINAPI D3DKMTOpenAdapterFromLuid(const D3DKMT_OPENADAPTERFROMLUID *);
    DECLSPEC_IMPORT status_t WINAPI D3DKMTQueryAdapterInfo(const D3DKMT_QUERYADAPTERINFO *);
    DECLSPEC_IMPORT status_t WINAPI D3DKMTCloseAdapter(const D3DKMT_CLOSEADAPTER *);
    DECLSPEC_IMPORT status_t WINAPI D3DKMTSetProcessSchedulingPriorityClass(HANDLE, D3DKMT_SCHEDULINGPRIORITYCLASS);
  }

  struct capture_gpu_scheduling_api_t {
    decltype(&D3DKMTOpenAdapterFromLuid) open_adapter = D3DKMTOpenAdapterFromLuid;
    decltype(&D3DKMTQueryAdapterInfo) query_adapter_info = D3DKMTQueryAdapterInfo;
    decltype(&D3DKMTCloseAdapter) close_adapter = D3DKMTCloseAdapter;
    decltype(&D3DKMTSetProcessSchedulingPriorityClass) set_process_priority = D3DKMTSetProcessSchedulingPriorityClass;
  };

  // The API table permits focused failure tests without changing real GPU scheduling.
  void configure_capture_process_gpu_priority(
    const DXGI_ADAPTER_DESC &adapter_desc,
    bool nv_realtime_hags,
    const capture_gpu_scheduling_api_t &api = {}
  );
}  // namespace platf::dxgi::detail
