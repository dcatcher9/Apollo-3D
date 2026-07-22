#include "virtual_display.h"

#include <algorithm>
#include <chrono>
#include <combaseapi.h>
#include <condition_variable>
#include <cwctype>
#include <dxgi.h>
#include <dxgi1_6.h>
#include <highlevelmonitorconfigurationapi.h>
#include <initguid.h>
#include <mutex>
#include <optional>
#include <physicalmonitorenumerationapi.h>
#include <setupapi.h>
#include <thread>
#include <vector>
#include <wrl/client.h>

using namespace SUDOVDA;

namespace VDISPLAY {
  // {dff7fd29-5b75-41d1-9731-b32a17a17104}
  // static const GUID DEFAULT_DISPLAY_GUID = { 0xdff7fd29, 0x5b75, 0x41d1, { 0x97, 0x31, 0xb3, 0x2a, 0x17, 0xa1, 0x71, 0x04 } };

  namespace {
    HANDLE sudovdaDriverHandle = INVALID_HANDLE_VALUE;
    std::mutex virtualDisplayMutationMutex;
    std::mutex driverLifecycleMutex;
    std::mutex watchdogOwnerMutex;
    std::mutex watchdogWaitMutex;
    std::condition_variable_any watchdogWake;
    std::jthread watchdogThread;

    bool containsCaseInsensitive(std::wstring_view value, std::wstring_view needle) {
      return std::search(
               value.begin(),
               value.end(),
               needle.begin(),
               needle.end(),
               [](wchar_t left, wchar_t right) {
                 return std::towlower(left) == std::towlower(right);
               }
             ) != value.end();
    }

    bool isSudoVirtualDisplayPath(std::wstring_view devicePath) {
      // Current SudoVDA/IddCx targets publish the SMKD1CE hardware ID. Keep the legacy driver-name
      // marker as well for already deployed variants and diagnostic/probe builds.
      return containsCaseInsensitive(devicePath, L"SMKD1CE") ||
             containsCaseInsensitive(devicePath, L"SUDOVDA");
    }

    bool sameLuid(const LUID &left, const LUID &right) {
      return left.HighPart == right.HighPart && left.LowPart == right.LowPart;
    }

    bool matchesVirtualDisplayIdentity(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &expectedIdentity,
      std::wstring_view learnedDevicePath,
      const LUID &candidateAdapter,
      uint32_t candidateTargetId,
      std::wstring_view candidateDevicePath
    ) {
      // A path learned from the exact AddVirtualDisplay result remains stable when Windows renumbers
      // DISPLAY names or target IDs. It is therefore the strongest available post-publication key.
      const bool learnedPath = !learnedDevicePath.empty() &&
                               learnedDevicePath == candidateDevicePath;
      const bool exactVirtualIdentity = sameLuid(expectedIdentity.AdapterLuid, candidateAdapter) &&
                                        expectedIdentity.TargetId == candidateTargetId &&
                                        isSudoVirtualDisplayPath(candidateDevicePath);
      // Do not fall back to a matching GDI name or to "the only SudoVDA output". DISPLAY numbers
      // are recyclable, and the driver intentionally supports multiple monitors and shared clients;
      // neither condition proves that a candidate belongs to this retirement record.
      return learnedPath || exactVirtualIdentity;
    }

    void stopWatchdogThread() {
      std::jthread retiring;
      {
        std::lock_guard lock(watchdogOwnerMutex);
        watchdogThread.request_stop();
        watchdogWake.notify_all();
        retiring = std::move(watchdogThread);
      }
      if (retiring.joinable()) {
        retiring.join();
      }
    }

    void closeDriverHandleLocked() {
      if (sudovdaDriverHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(sudovdaDriverHandle);
        sudovdaDriverHandle = INVALID_HANDLE_VALUE;
      }
    }

    std::optional<LUID> adapterLuidByName(const std::wstring &adapterName) {
      Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
      if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        return std::nullopt;
      }

      for (UINT index = 0;; ++index) {
        Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
        if (factory->EnumAdapters(index, &adapter) == DXGI_ERROR_NOT_FOUND) {
          break;
        }

        DXGI_ADAPTER_DESC desc {};
        if (SUCCEEDED(adapter->GetDesc(&desc)) && std::wstring_view(desc.Description) == adapterName) {
          return desc.AdapterLuid;
        }
      }
      return std::nullopt;
    }
  }  // namespace

  LONG getDeviceSettings(const wchar_t* deviceName, DEVMODEW& devMode) {
	devMode.dmSize = sizeof(DEVMODEW);
	return EnumDisplaySettingsW(deviceName, ENUM_CURRENT_SETTINGS, &devMode);
}

namespace {
LONG applyDisplaySettings(const wchar_t *deviceName, int width, int height, int refresh_rate) {
  std::vector<DISPLAYCONFIG_PATH_INFO> pathArray;
  std::vector<DISPLAYCONFIG_MODE_INFO> modeArray;

  if (!queryActiveDisplayConfig(pathArray, modeArray)) {
    wprintf(L"[SUDOVDA] Failed to query display configuration.\n");
    return ERROR_INVALID_PARAMETER;
  }
  const UINT32 pathCount = (UINT32) pathArray.size();
  const UINT32 modeCount = (UINT32) modeArray.size();
  for (UINT32 i = 0; i < pathCount; i++) {
    DISPLAYCONFIG_SOURCE_DEVICE_NAME sourceName = {};
    sourceName.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
    sourceName.header.size = sizeof(sourceName);
    sourceName.header.adapterId = pathArray[i].sourceInfo.adapterId;
    sourceName.header.id = pathArray[i].sourceInfo.id;

    if (DisplayConfigGetDeviceInfo(&sourceName.header) != ERROR_SUCCESS) {
      continue;
    }

    auto *sourceInfo = &pathArray[i].sourceInfo;
    auto *targetInfo = &pathArray[i].targetInfo;

    if (std::wstring_view(sourceName.viewGdiDeviceName) == std::wstring_view(deviceName)) {
      wprintf(L"[SUDOVDA] Display found: %ls\n", deviceName);
      for (UINT32 j = 0; j < modeCount; j++) {
        if (
          modeArray[j].infoType == DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE &&
          modeArray[j].adapterId.HighPart == sourceInfo->adapterId.HighPart &&
          modeArray[j].adapterId.LowPart == sourceInfo->adapterId.LowPart &&
          modeArray[j].id == sourceInfo->id
        ) {
          auto *sourceMode = &modeArray[j].sourceMode;

          wprintf(L"[SUDOVDA] Current mode found: [%dx%dx%d]\n", sourceMode->width, sourceMode->height, targetInfo->refreshRate);

          sourceMode->width = width;
          sourceMode->height = height;

          targetInfo->refreshRate = {(UINT32) refresh_rate, 1000};

          // Apply the changes
          LONG status = SetDisplayConfig(
            pathCount,
            pathArray.data(),
            modeCount,
            modeArray.data(),
            SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG | SDC_SAVE_TO_DATABASE
          );
          if (status != ERROR_SUCCESS) {
            wprintf(L"[SUDOVDA] Failed to apply display settings.\n");
          } else {
            wprintf(L"[SUDOVDA] Display settings updated successfully.\n");
          }

          return status;
        }
      }

      wprintf(L"[SUDOVDA] Mode [%dx%dx%d] not found for display: %ls\n", width, height, refresh_rate, deviceName);
      return ERROR_INVALID_PARAMETER;
    }
  }

  wprintf(L"[SUDOVDA] Display not found: %ls\n", deviceName);
  return ERROR_DEVICE_NOT_CONNECTED;
}
}  // namespace

LONG changeDisplaySettings(const wchar_t *deviceName, int width, int height, int refresh_rate) {
  DEVMODEW devMode = {};
  devMode.dmSize = sizeof(devMode);

  // Old method to set at least baseline refresh rate
  if (EnumDisplaySettingsW(deviceName, ENUM_CURRENT_SETTINGS, &devMode)) {
    DWORD targetRefreshRate = refresh_rate / 1000;
    DWORD altRefreshRate = targetRefreshRate;

    if (refresh_rate % 1000) {
      if (refresh_rate % 1000 >= 900) {
        targetRefreshRate += 1;
      } else {
        altRefreshRate += 1;
      }
    } else {
      altRefreshRate -= 1;
    }

    wprintf(L"[SUDOVDA] Applying baseline display mode [%dx%dx%d] for %ls.\n", width, height, targetRefreshRate, deviceName);

    devMode.dmPelsWidth = width;
    devMode.dmPelsHeight = height;
    devMode.dmDisplayFrequency = targetRefreshRate;
    devMode.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;

    auto res = ChangeDisplaySettingsExW(deviceName, &devMode, NULL, CDS_UPDATEREGISTRY, NULL);

    if (res != ERROR_SUCCESS) {
      wprintf(L"[SUDOVDA] Failed to apply baseline display mode, trying alt mode: [%dx%dx%d].\n", width, height, altRefreshRate);
      devMode.dmDisplayFrequency = altRefreshRate;
      res = ChangeDisplaySettingsExW(deviceName, &devMode, NULL, CDS_UPDATEREGISTRY, NULL);
      if (res != ERROR_SUCCESS) {
        wprintf(L"[SUDOVDA] Failed to apply alt baseline display mode.\n");
      }
    }

    if (res == ERROR_SUCCESS) {
      wprintf(L"[SUDOVDA] Baseline display mode applied successfully.");
    }
  }

  // Apply the exact fractional refresh rate through DisplayConfig.
  return applyDisplaySettings(deviceName, width, height, refresh_rate);
}

bool findDisplayIds(const wchar_t *displayName, LUID &adapterId, uint32_t &targetId) {
  std::vector<DISPLAYCONFIG_PATH_INFO> paths;
  std::vector<DISPLAYCONFIG_MODE_INFO> modes;
  if (!queryActiveDisplayConfig(paths, modes)) {
    return false;
  }

  auto path = std::find_if(paths.begin(), paths.end(), [&displayName](DISPLAYCONFIG_PATH_INFO _path) {
    DISPLAYCONFIG_PATH_SOURCE_INFO sourceInfo = _path.sourceInfo;

    DISPLAYCONFIG_SOURCE_DEVICE_NAME sourceName = {};
    sourceName.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
    sourceName.header.size = sizeof(sourceName);
    sourceName.header.adapterId = sourceInfo.adapterId;
    sourceName.header.id = sourceInfo.id;

    if (DisplayConfigGetDeviceInfo(&sourceName.header) != ERROR_SUCCESS) {
      return false;
    }

    return std::wstring_view(displayName) == sourceName.viewGdiDeviceName;
  });

  if (path == paths.end()) {
    return false;
  }

  adapterId = path->targetInfo.adapterId;
  targetId = path->targetInfo.id;

  return true;
}

std::optional<bool> queryDisplayHDR(const LUID &adapterLuid, uint32_t targetId) {
  // Query the display configuration state directly. A virtual HDR desktop is represented as
  // linear scRGB to desktop applications, so its DXGI output color space is not required to be
  // the physical-output PQ/Rec.2020 space and is not a reliable HDR-enabled test.
  DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO_2 info2 {};
  info2.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO_2;
  info2.header.size = sizeof(info2);
  info2.header.adapterId = adapterLuid;
  info2.header.id = targetId;
  if (DisplayConfigGetDeviceInfo(&info2.header) == ERROR_SUCCESS) {
    return info2.activeColorMode == DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  }

  DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO info {};
  info.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO;
  info.header.size = sizeof(info);
  info.header.adapterId = adapterLuid;
  info.header.id = targetId;
  if (DisplayConfigGetDeviceInfo(&info.header) == ERROR_SUCCESS) {
    return info.advancedColorEnabled != 0;
  }
  return std::nullopt;
}

bool setDisplayHDR(const LUID& adapterId, const uint32_t& targetId, bool enableAdvancedColor) {
  DISPLAYCONFIG_SET_HDR_STATE setHdrState = {};
  setHdrState.header.type = DISPLAYCONFIG_DEVICE_INFO_SET_HDR_STATE;
  setHdrState.header.size = sizeof(setHdrState);
  setHdrState.header.adapterId = adapterId;
  setHdrState.header.id = targetId;
  setHdrState.enableHdr = enableAdvancedColor;

  if (DisplayConfigSetDeviceInfo(&setHdrState.header) == ERROR_SUCCESS) {
    return true;
  }

  // Windows 10 exposes only the combined Advanced Color setter.
  DISPLAYCONFIG_SET_ADVANCED_COLOR_STATE setHdrInfo = {};
  setHdrInfo.header.type = DISPLAYCONFIG_DEVICE_INFO_SET_ADVANCED_COLOR_STATE;
  setHdrInfo.header.size = sizeof(setHdrInfo);
  setHdrInfo.header.adapterId = adapterId;
  setHdrInfo.header.id = targetId;
  setHdrInfo.enableAdvancedColor = enableAdvancedColor;

  return DisplayConfigSetDeviceInfo(&setHdrInfo.header) == ERROR_SUCCESS;
}

std::optional<bool> queryDisplayHDRByName(const wchar_t* displayName) {
	LUID adapterId;
	uint32_t targetId;

	if (!findDisplayIds(displayName, adapterId, targetId)) {
		wprintf(L"[SUDOVDA] Failed to find display IDs for %ls!\n", displayName);
		return std::nullopt;
	}

  return queryDisplayHDR(adapterId, targetId);
}

bool setDisplayHDRByName(const wchar_t* displayName, bool enableAdvancedColor) {
	LUID adapterId;
	uint32_t targetId;

	if (!findDisplayIds(displayName, adapterId, targetId)) {
		return false;
	}

	return setDisplayHDR(adapterId, targetId, enableAdvancedColor);
}

void closeVDisplayDevice() {
  std::lock_guard lifecycle_lock(driverLifecycleMutex);
  stopWatchdogThread();
  std::lock_guard device_lock(virtualDisplayMutationMutex);
  closeDriverHandleLocked();
}

DRIVER_STATUS openVDisplayDevice() {
  std::lock_guard lifecycle_lock(driverLifecycleMutex);
  stopWatchdogThread();
  std::lock_guard device_lock(virtualDisplayMutationMutex);
  closeDriverHandleLocked();

  uint32_t retryInterval = 20;
  while (true) {
    sudovdaDriverHandle = OpenDevice(&SUVDA_INTERFACE_GUID);
    if (sudovdaDriverHandle == INVALID_HANDLE_VALUE) {
      if (retryInterval > 320) {
        printf("[SUDOVDA] Open device failed!\n");
        return DRIVER_STATUS::FAILED;
      }
      retryInterval *= 2;
      Sleep(retryInterval);
      continue;
    }
    break;
  }

  if (!CheckProtocolCompatible(sudovdaDriverHandle)) {
    printf("[SUDOVDA] SUDOVDA protocol not compatible with driver!\n");
    closeDriverHandleLocked();
    return DRIVER_STATUS::VERSION_INCOMPATIBLE;
  }
  return DRIVER_STATUS::OK;
}

bool startPingThread(std::function<void()> failCb) {
  std::lock_guard lifecycle_lock(driverLifecycleMutex);
  stopWatchdogThread();

  VIRTUAL_DISPLAY_GET_WATCHDOG_OUT watchdogOut {};
  {
    std::lock_guard device_lock(virtualDisplayMutationMutex);
    if (sudovdaDriverHandle == INVALID_HANDLE_VALUE) {
      return false;
    }
    if (!GetWatchdogTimeout(sudovdaDriverHandle, watchdogOut)) {
      printf("[SUDOVDA] Watchdog fetch failed!\n");
      return false;
    }
  }
  printf("[SUDOVDA] Watchdog: Timeout %d, Countdown %d\n", watchdogOut.Timeout, watchdogOut.Countdown);
  if (!watchdogOut.Timeout) {
    return true;
  }

  const auto sleepInterval = std::chrono::milliseconds(watchdogOut.Timeout * 1000 / 3);
  std::lock_guard owner_lock(watchdogOwnerMutex);
  watchdogThread = std::jthread([sleepInterval, failCb = std::move(failCb)](std::stop_token stop_token) {
    uint8_t fail_count = 0;
    while (!stop_token.stop_requested()) {
      bool watchdog_failed = false;
      {
        std::lock_guard device_lock(virtualDisplayMutationMutex);
        if (sudovdaDriverHandle == INVALID_HANDLE_VALUE) {
          return;
        }
        if (PingDriver(sudovdaDriverHandle)) {
          fail_count = 0;
        } else if (++fail_count > 3) {
          // The watchdog owns handle invalidation. The callback only publishes status, avoiding a
          // self-join when failure is reported from this worker.
          closeDriverHandleLocked();
          watchdog_failed = true;
        }
      }
      if (watchdog_failed) {
        failCb();
        return;
      }
      std::unique_lock wait_lock(watchdogWaitMutex);
      watchdogWake.wait_for(wait_lock, stop_token, sleepInterval, []() {
        return false;
      });
    }
  });
  return true;
}

bool queryActiveDisplayConfig(
  std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
  std::vector<DISPLAYCONFIG_MODE_INFO> &modes
) {
  // The active-path count can change between the size and data calls while Windows is applying a
  // hotplug, HDR, or IddCx topology transition. Retry that documented race instead of presenting a
  // transient empty desktop to callers.
  for (int attempt = 0; attempt < 8; ++attempt) {
    UINT32 pathCount = 0;
    UINT32 modeCount = 0;
    const auto sizeStatus = GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &pathCount, &modeCount);
    if (sizeStatus != ERROR_SUCCESS) {
      if (sizeStatus != ERROR_INSUFFICIENT_BUFFER) {
        return false;
      }
      Sleep(1u << std::min(attempt, 5));
      continue;
    }

    paths.resize(pathCount);
    modes.resize(modeCount);
    const auto queryStatus = QueryDisplayConfig(
      QDC_ONLY_ACTIVE_PATHS,
      &pathCount,
      paths.data(),
      &modeCount,
      modes.data(),
      nullptr
    );
    if (queryStatus == ERROR_SUCCESS) {
      paths.resize(pathCount);
      modes.resize(modeCount);
      return true;
    }
    if (queryStatus != ERROR_INSUFFICIENT_BUFFER) {
      return false;
    }
    Sleep(1u << std::min(attempt, 5));
  }

  paths.clear();
  modes.clear();
  return false;
}

display_identity_query_t queryDisplayIdentity(const LUID &adapterLuid, uint32_t targetId) {
  std::vector<DISPLAYCONFIG_PATH_INFO> paths;
  std::vector<DISPLAYCONFIG_MODE_INFO> modes;
  if (!queryActiveDisplayConfig(paths, modes)) {
    return {};
  }

  for (const auto &path : paths) {
    if (path.targetInfo.id != targetId || path.targetInfo.adapterId.HighPart != adapterLuid.HighPart || path.targetInfo.adapterId.LowPart != adapterLuid.LowPart) {
      continue;
    }

    DISPLAYCONFIG_SOURCE_DEVICE_NAME sourceName = {};
    sourceName.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
    sourceName.header.size = sizeof(sourceName);
    sourceName.header.adapterId = path.sourceInfo.adapterId;
    sourceName.header.id = path.sourceInfo.id;
    DISPLAYCONFIG_TARGET_DEVICE_NAME targetName {};
    targetName.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
    targetName.header.size = sizeof(targetName);
    targetName.header.adapterId = path.targetInfo.adapterId;
    targetName.header.id = path.targetInfo.id;
    if (DisplayConfigGetDeviceInfo(&sourceName.header) == ERROR_SUCCESS &&
        DisplayConfigGetDeviceInfo(&targetName.header) == ERROR_SUCCESS) {
      std::wstring display_name = sourceName.viewGdiDeviceName;
      if (!display_name.empty()) {
        return {
          display_identity_state_e::present,
          std::move(display_name),
          targetName.monitorDevicePath,
          targetName.monitorFriendlyDeviceName,
        };
      }
    }

    // The exact stable adapter/target identity is still in the active topology, but Windows could
    // not publish its GDI source name. Do not let callers mistake that transient query failure for
    // completed removal.
    return {};
  }

  // Absence is authoritative only after a complete active-topology snapshot contains no exact
  // adapter/target match.
  return {display_identity_state_e::absent, {}, {}, {}};
}

display_identity_query_t queryVirtualDisplayIdentity(
  const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
  std::wstring_view devicePath,
  std::wstring_view /*displayName*/
) {
  std::vector<DISPLAYCONFIG_PATH_INFO> paths;
  std::vector<DISPLAYCONFIG_MODE_INFO> modes;
  if (!queryActiveDisplayConfig(paths, modes)) {
    return {};
  }

  for (const auto &path : paths) {
    DISPLAYCONFIG_SOURCE_DEVICE_NAME sourceName {};
    sourceName.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
    sourceName.header.size = sizeof(sourceName);
    sourceName.header.adapterId = path.sourceInfo.adapterId;
    sourceName.header.id = path.sourceInfo.id;

    DISPLAYCONFIG_TARGET_DEVICE_NAME targetName {};
    targetName.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
    targetName.header.size = sizeof(targetName);
    targetName.header.adapterId = path.targetInfo.adapterId;
    targetName.header.id = path.targetInfo.id;
    if (DisplayConfigGetDeviceInfo(&sourceName.header) != ERROR_SUCCESS ||
        DisplayConfigGetDeviceInfo(&targetName.header) != ERROR_SUCCESS) {
      return {};
    }

    const std::wstring_view candidatePath = targetName.monitorDevicePath;
    if (matchesVirtualDisplayIdentity(
          identity,
          devicePath,
          path.targetInfo.adapterId,
          path.targetInfo.id,
          candidatePath
        )) {
      return {
        display_identity_state_e::present,
        sourceName.viewGdiDeviceName,
        targetName.monitorDevicePath,
        targetName.monitorFriendlyDeviceName,
      };
    }
  }
  return {display_identity_state_e::absent, {}, {}, {}};
}

#ifdef SUNSHINE_TESTS
bool isSudoVirtualDisplayPathForTest(std::wstring_view devicePath) {
  return isSudoVirtualDisplayPath(devicePath);
}

bool virtualDisplayIdentityMatchesForTest(
  const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &expectedIdentity,
  std::wstring_view learnedDevicePath,
  const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &candidateIdentity,
  std::wstring_view candidateDevicePath
) {
  return matchesVirtualDisplayIdentity(
    expectedIdentity,
    learnedDevicePath,
    candidateIdentity.AdapterLuid,
    candidateIdentity.TargetId,
    candidateDevicePath
  );
}
#endif

namespace {
  std::optional<LUID> primaryDisplayAdapterLuid() {
    std::wstring primaryDisplay;
    for (DWORD index = 0;; ++index) {
      DISPLAY_DEVICEW device {};
      device.cb = sizeof(device);
      if (!EnumDisplayDevicesW(nullptr, index, &device, 0)) {
        break;
      }
      if ((device.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE) != 0) {
        primaryDisplay = device.DeviceName;
        break;
      }
    }
    if (primaryDisplay.empty()) {
      return std::nullopt;
    }

    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    std::vector<DISPLAYCONFIG_MODE_INFO> modes;
    if (!queryActiveDisplayConfig(paths, modes)) {
      return std::nullopt;
    }
    for (const auto &path : paths) {
      DISPLAYCONFIG_SOURCE_DEVICE_NAME sourceName {};
      sourceName.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
      sourceName.header.size = sizeof(sourceName);
      sourceName.header.adapterId = path.sourceInfo.adapterId;
      sourceName.header.id = path.sourceInfo.id;
      if (DisplayConfigGetDeviceInfo(&sourceName.header) == ERROR_SUCCESS && std::wstring_view(sourceName.viewGdiDeviceName) == primaryDisplay) {
        return path.sourceInfo.adapterId;
      }
    }
    return std::nullopt;
  }

  creation_result_t createVirtualDisplayImpl(
    const char *s_client_uid,
    const char *s_client_name,
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    const GUID &guid,
    const std::optional<LUID> &adapterLuid
  ) {
    creation_result_t result;
    VIRTUAL_DISPLAY_ADD_OUT output {};
    {
      // SudoVDA's render-adapter choice is process-global. Keep adapter selection and AddVirtualDisplay
      // in one critical section so local AR and remote streaming sessions cannot steal each other's GPU.
      std::lock_guard lock(virtualDisplayMutationMutex);
      if (sudovdaDriverHandle == INVALID_HANDLE_VALUE) {
        return {};
      }
      if (adapterLuid && !SetRenderAdapter(sudovdaDriverHandle, *adapterLuid)) {
        printf("[SUDOVDA] Failed to select render adapter for virtual display.\n");
        return {};
      }
      if (!AddVirtualDisplay(sudovdaDriverHandle, width, height, fps, guid, s_client_name, s_client_uid, output)) {
        printf("[SUDOVDA] Failed to add virtual display.\n");
        return {};
      }
    }
    // Driver creation and Windows display-name publication are separate operations. Record Add's
    // success immediately so callers can always retire the exact output even if name lookup times
    // out during topology churn.
    result.identity = output;

    uint32_t retryInterval = 20;
    display_identity_query_t published;
    while ((published = queryDisplayIdentity(output.AdapterLuid, output.TargetId)).state !=
           display_identity_state_e::present) {
      Sleep(retryInterval);
      if (retryInterval > 320) {
        printf("[SUDOVDA] Cannot get name for newly added virtual display!\n");
        return result;
      }
      retryInterval *= 2;
    }

    wprintf(L"[SUDOVDA] Virtual display added successfully: %ls\n", published.display_name.c_str());
    printf("[SUDOVDA] Configuration: W: %d, H: %d, FPS: %d\n", width, height, fps);

    result.display_name = std::move(published.display_name);
    result.device_path = std::move(published.device_path);
    result.friendly_name = std::move(published.friendly_name);
    return result;
  }
}  // namespace

creation_result_t createVirtualDisplay(
  const char *s_client_uid,
  const char *s_client_name,
  uint32_t width,
  uint32_t height,
  uint32_t fps,
  const GUID &guid
) {
  return createVirtualDisplayImpl(
    s_client_uid,
    s_client_name,
    width,
    height,
    fps,
    guid,
    primaryDisplayAdapterLuid()
  );
}

creation_result_t createVirtualDisplayOnAdapter(
  const char *s_client_uid,
  const char *s_client_name,
  uint32_t width,
  uint32_t height,
  uint32_t fps,
  const GUID &guid,
  const LUID &adapterLuid
) {
  return createVirtualDisplayImpl(
    s_client_uid,
    s_client_name,
    width,
    height,
    fps,
    guid,
    adapterLuid
  );
}

creation_result_t createVirtualDisplayOnAdapter(
  const char *s_client_uid,
  const char *s_client_name,
  uint32_t width,
  uint32_t height,
  uint32_t fps,
  const GUID &guid,
  const std::wstring &adapterName
) {
  const auto adapterLuid = adapterLuidByName(adapterName);
  if (!adapterLuid) {
    printf("[SUDOVDA] Cannot find requested render adapter.\n");
    return {};
  }
  return createVirtualDisplayImpl(
    s_client_uid,
    s_client_name,
    width,
    height,
    fps,
    guid,
    *adapterLuid
  );
}

bool removeVirtualDisplay(const GUID &guid) {
  std::lock_guard lock(virtualDisplayMutationMutex);
  if (sudovdaDriverHandle == INVALID_HANDLE_VALUE) {
    return false;
  }

  if (RemoveVirtualDisplay(sudovdaDriverHandle, guid)) {
    printf("[SUDOVDA] Virtual display removed successfully.\n");
    return true;
  } else {
    return false;
  }
}

// Utility function to match the DeviceString to the Display Names
// Typical DeviceStrings are the driver names
//
// Example: matchDisplay(L"SudoMaker Virtual Display Adapter")
// Result: L"\\\\.\\Display2"

std::vector <std::wstring> matchDisplay(std::wstring sMatch) {
	DISPLAY_DEVICEW displayDevice;
	displayDevice.cb = sizeof(DISPLAY_DEVICE);

	std::wstring matchDeviceName;

	std::vector <std::wstring>vMatches;

	int deviceIndex = 0;
	while (EnumDisplayDevicesW(NULL, deviceIndex, &displayDevice, 0)) {
		if (std::wstring(displayDevice.DeviceString) == sMatch &&
			displayDevice.StateFlags > 0) {
			matchDeviceName = displayDevice.DeviceName;
			vMatches.push_back(matchDeviceName);
		}
		deviceIndex++;
	}
	return vMatches;
}

}  // namespace VDISPLAY
