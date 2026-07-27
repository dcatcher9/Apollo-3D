#include "virtual_display.h"

#include <algorithm>
#include <chrono>
#include <combaseapi.h>
#include <condition_variable>
#include <cstdint>
#include <cwctype>
#include <dxgi.h>
#include <dxgi1_6.h>
#include <highlevelmonitorconfigurationapi.h>
#include <initguid.h>
#include <limits>
#include <mutex>
#include <optional>
#include <physicalmonitorenumerationapi.h>
#include <setupapi.h>
#include <thread>
#include <tuple>
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

    std::chrono::milliseconds watchdogPingInterval(uint32_t timeoutSeconds) {
      const auto thirdOfTimeout = std::chrono::milliseconds(
        static_cast<uint64_t>(timeoutSeconds) * 1000 / 3
      );
      // A longer driver lease protects the monitor from debugger and GPU-startup stalls, but the
      // heartbeat itself should remain frequent. This also detects a dead driver promptly.
      return std::min(thirdOfTimeout, std::chrono::milliseconds(1000));
    }

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

    enum class detach_plan_e {
      ready,
      already_inactive,
      skipped_only_active,
      ambiguous_identity,
    };

    struct detach_plan_t {
      detach_plan_e state {detach_plan_e::ambiguous_identity};
      std::optional<std::size_t> path_index;
    };

    detach_plan_t prepareVirtualDisplayDetachPaths(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view learnedDevicePath,
      const std::vector<std::wstring> &candidateDevicePaths,
      std::vector<DISPLAYCONFIG_PATH_INFO> &paths
    ) {
      if (candidateDevicePaths.size() != paths.size()) {
        return {};
      }

      std::vector<std::size_t> matches;
      for (std::size_t index = 0; index < paths.size(); ++index) {
        const auto &path = paths[index];
        if ((path.flags & DISPLAYCONFIG_PATH_ACTIVE) == 0) {
          continue;
        }

        const bool exact_driver_identity =
          sameLuid(identity.AdapterLuid, path.targetInfo.adapterId) &&
          identity.TargetId == path.targetInfo.id;
        if (candidateDevicePaths[index].empty() &&
            (exact_driver_identity || !learnedDevicePath.empty())) {
          // The device path is what distinguishes SudoVDA from a physical target reusing the same
          // adapter/target identity, and what follows a Sudo target after CCD renumbering.
          return {detach_plan_e::ambiguous_identity, std::nullopt};
        }

        if (matchesVirtualDisplayIdentity(
              identity,
              learnedDevicePath,
              path.targetInfo.adapterId,
              path.targetInfo.id,
              candidateDevicePaths[index]
            )) {
          matches.push_back(index);
        }
      }

      if (matches.empty()) {
        return {detach_plan_e::already_inactive, std::nullopt};
      }
      if (matches.size() != 1) {
        return {detach_plan_e::ambiguous_identity, std::nullopt};
      }
      bool has_available_survivor = false;
      for (std::size_t index = 0; index < paths.size(); ++index) {
        if (index != matches.front() &&
            (paths[index].flags & DISPLAYCONFIG_PATH_ACTIVE) != 0 &&
            paths[index].targetInfo.targetAvailable != FALSE) {
          has_available_survivor = true;
          break;
        }
      }
      if (!has_available_survivor) {
        return {detach_plan_e::skipped_only_active, std::nullopt};
      }

      auto &retiring_path = paths[matches.front()];
      retiring_path.flags &= ~DISPLAYCONFIG_PATH_ACTIVE;
      // An inactive supplied path must not reference modes. Set every virtual-mode-aware union
      // member explicitly; this is especially important when the retiring target shares a source
      // mode or clone group with a surviving path.
      retiring_path.sourceInfo.cloneGroupId = DISPLAYCONFIG_PATH_CLONE_GROUP_INVALID;
      retiring_path.sourceInfo.sourceModeInfoIdx = DISPLAYCONFIG_PATH_SOURCE_MODE_IDX_INVALID;
      retiring_path.targetInfo.desktopModeInfoIdx = DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID;
      retiring_path.targetInfo.targetModeInfoIdx = DISPLAYCONFIG_PATH_TARGET_MODE_IDX_INVALID;
      return {detach_plan_e::ready, matches.front()};
    }

    bool rebaseSurvivingSourceModes(
      const std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
      std::vector<DISPLAYCONFIG_MODE_INFO> &modes
    ) {
      std::vector<std::size_t> source_mode_indices;
      for (const auto &path : paths) {
        if ((path.flags & DISPLAYCONFIG_PATH_ACTIVE) == 0) {
          continue;
        }

        const bool uses_virtual_mode_indices =
          (path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE) != 0;
        const UINT32 mode_index = uses_virtual_mode_indices ?
                                    path.sourceInfo.sourceModeInfoIdx :
                                    path.sourceInfo.modeInfoIdx;
        const UINT32 invalid_mode_index = uses_virtual_mode_indices ?
                                            DISPLAYCONFIG_PATH_SOURCE_MODE_IDX_INVALID :
                                            DISPLAYCONFIG_PATH_MODE_IDX_INVALID;
        if (mode_index == invalid_mode_index ||
            mode_index >= modes.size()) {
          return false;
        }
        const auto &mode = modes[mode_index];
        if (mode.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE ||
            !sameLuid(mode.adapterId, path.sourceInfo.adapterId) ||
            mode.id != path.sourceInfo.id) {
          return false;
        }
        if (std::find(
              source_mode_indices.begin(),
              source_mode_indices.end(),
              mode_index
            ) == source_mode_indices.end()) {
          source_mode_indices.push_back(mode_index);
        }
      }
      if (source_mode_indices.empty()) {
        return false;
      }

      bool has_primary_source = false;
      for (const auto mode_index : source_mode_indices) {
        const auto &position = modes[mode_index].sourceMode.position;
        if (position.x == 0 && position.y == 0) {
          has_primary_source = true;
          break;
        }
      }

      // If the retiring source owned (0,0), promote the closest surviving source deterministically
      // and translate every survivor by the same amount. Relative layout and modes stay unchanged.
      const auto position_key = [&](std::size_t mode_index) {
        const auto &mode = modes[mode_index];
        const auto &position = mode.sourceMode.position;
        const auto magnitude = [](LONG coordinate) {
          const auto wide = static_cast<std::int64_t>(coordinate);
          return static_cast<std::uint64_t>(wide < 0 ? -wide : wide);
        };
        return std::tuple {
          magnitude(position.x) + magnitude(position.y),
          position.y,
          position.x,
          mode.adapterId.HighPart,
          mode.adapterId.LowPart,
          mode.id,
        };
      };
      POINTL anchor_position {};
      if (!has_primary_source) {
        const auto anchor = *std::min_element(
          source_mode_indices.begin(),
          source_mode_indices.end(),
          [&](std::size_t left, std::size_t right) {
            return position_key(left) < position_key(right);
          }
        );
        anchor_position = modes[anchor].sourceMode.position;
      }

      std::vector<POINTL> translated_positions;
      translated_positions.reserve(source_mode_indices.size());
      for (const auto mode_index : source_mode_indices) {
        const auto &position = modes[mode_index].sourceMode.position;
        const auto translated_x =
          static_cast<std::int64_t>(position.x) - anchor_position.x;
        const auto translated_y =
          static_cast<std::int64_t>(position.y) - anchor_position.y;
        if (translated_x < std::numeric_limits<LONG>::min() ||
            translated_x > std::numeric_limits<LONG>::max() ||
            translated_y < std::numeric_limits<LONG>::min() ||
            translated_y > std::numeric_limits<LONG>::max()) {
          return false;
        }
        translated_positions.push_back({
          static_cast<LONG>(translated_x),
          static_cast<LONG>(translated_y),
        });
      }

      struct source_rectangle_t {
        std::int64_t left;
        std::int64_t top;
        std::int64_t right;
        std::int64_t bottom;
      };
      std::vector<source_rectangle_t> rectangles;
      rectangles.reserve(source_mode_indices.size());
      for (std::size_t index = 0; index < source_mode_indices.size(); ++index) {
        const auto &mode = modes[source_mode_indices[index]].sourceMode;
        if (mode.width == 0 || mode.height == 0) {
          return false;
        }
        rectangles.push_back({
          translated_positions[index].x,
          translated_positions[index].y,
          static_cast<std::int64_t>(translated_positions[index].x) + mode.width,
          static_cast<std::int64_t>(translated_positions[index].y) + mode.height,
        });
      }

      // GDI may rearrange a supplied layout in an undefined way to repair gaps or overlaps. Refuse
      // such a detach before SetDisplayConfig, so surviving display positions are never silently
      // changed. Edge-connected rectangles with positive overlap on the shared edge form one
      // unambiguous desktop component.
      std::vector<std::vector<std::size_t>> adjacent(rectangles.size());
      for (std::size_t left = 0; left < rectangles.size(); ++left) {
        for (std::size_t right = left + 1; right < rectangles.size(); ++right) {
          const auto &a = rectangles[left];
          const auto &b = rectangles[right];
          const bool x_overlap =
            std::max(a.left, b.left) < std::min(a.right, b.right);
          const bool y_overlap =
            std::max(a.top, b.top) < std::min(a.bottom, b.bottom);
          const bool identical_clone_geometry =
            a.left == b.left && a.top == b.top &&
            a.right == b.right && a.bottom == b.bottom;
          if (identical_clone_geometry) {
            adjacent[left].push_back(right);
            adjacent[right].push_back(left);
            continue;
          }
          if (x_overlap && y_overlap) {
            return false;
          }
          const bool shares_vertical_edge =
            (a.right == b.left || b.right == a.left) && y_overlap;
          const bool shares_horizontal_edge =
            (a.bottom == b.top || b.bottom == a.top) && x_overlap;
          if (shares_vertical_edge || shares_horizontal_edge) {
            adjacent[left].push_back(right);
            adjacent[right].push_back(left);
          }
        }
      }

      std::vector<bool> visited(rectangles.size(), false);
      std::vector<std::size_t> pending {0};
      visited[0] = true;
      for (std::size_t cursor = 0; cursor < pending.size(); ++cursor) {
        for (const auto neighbor : adjacent[pending[cursor]]) {
          if (!visited[neighbor]) {
            visited[neighbor] = true;
            pending.push_back(neighbor);
          }
        }
      }
      if (std::find(visited.begin(), visited.end(), false) != visited.end()) {
        return false;
      }

      for (std::size_t index = 0; index < source_mode_indices.size(); ++index) {
        modes[source_mode_indices[index]].sourceMode.position = translated_positions[index];
      }
      return true;
    }

    struct retirement_candidate_t {
      SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT identity {};
      std::wstring device_path;
      bool target_available = false;
      bool device_info_available = true;
    };

    display_identity_state_e classifyVirtualDisplayRetirementState(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view learnedDevicePath,
      const std::vector<retirement_candidate_t> &candidates
    ) {
      bool indeterminate_candidate = false;
      for (const auto &candidate : candidates) {
        if (!candidate.target_available) {
          continue;
        }

        const bool exact_driver_identity =
          sameLuid(identity.AdapterLuid, candidate.identity.AdapterLuid) &&
          identity.TargetId == candidate.identity.TargetId;
        const bool has_usable_device_path =
          candidate.device_info_available && !candidate.device_path.empty();
        if (!has_usable_device_path) {
          // A failed name query for the exact target, or for any available target when the
          // renumbering-safe learned path is needed, means absence cannot be proved.
          if (exact_driver_identity || !learnedDevicePath.empty()) {
            indeterminate_candidate = true;
          }
          continue;
        }

        if (matchesVirtualDisplayIdentity(
              identity,
              learnedDevicePath,
              candidate.identity.AdapterLuid,
              candidate.identity.TargetId,
              candidate.device_path
            )) {
          return display_identity_state_e::present;
        }
      }
      return indeterminate_candidate ?
               display_identity_state_e::indeterminate :
               display_identity_state_e::absent;
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
struct baseline_refresh_rates_t {
  DWORD preferred;
  DWORD alternate;
};

baseline_refresh_rates_t baselineRefreshRates(int refresh_rate) {
  DWORD preferred = refresh_rate / 1000;
  DWORD alternate = preferred;

  if (refresh_rate % 1000) {
    if (refresh_rate % 1000 >= 900) {
      preferred += 1;
    } else {
      alternate += 1;
    }
  } else if (alternate > 0) {
    alternate -= 1;
  }

  return {preferred, alternate};
}

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

LONG testDisplaySettings(const wchar_t *deviceName, int width, int height, int refresh_rate) {
  DEVMODEW devMode = {};
  devMode.dmSize = sizeof(devMode);
  if (!EnumDisplaySettingsW(deviceName, ENUM_CURRENT_SETTINGS, &devMode)) {
    return DISP_CHANGE_FAILED;
  }

  const auto refreshRates = baselineRefreshRates(refresh_rate);
  devMode.dmPelsWidth = width;
  devMode.dmPelsHeight = height;
  devMode.dmDisplayFrequency = refreshRates.preferred;
  devMode.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;

  auto result = ChangeDisplaySettingsExW(deviceName, &devMode, nullptr, CDS_TEST, nullptr);
  if (result == DISP_CHANGE_SUCCESSFUL || refreshRates.alternate == refreshRates.preferred) {
    return result;
  }

  devMode.dmDisplayFrequency = refreshRates.alternate;
  return ChangeDisplaySettingsExW(deviceName, &devMode, nullptr, CDS_TEST, nullptr);
}

LONG changeDisplaySettings(const wchar_t *deviceName, int width, int height, int refresh_rate) {
  DEVMODEW devMode = {};
  devMode.dmSize = sizeof(devMode);

  // Old method to set at least baseline refresh rate
  if (EnumDisplaySettingsW(deviceName, ENUM_CURRENT_SETTINGS, &devMode)) {
    const auto refreshRates = baselineRefreshRates(refresh_rate);

    wprintf(L"[SUDOVDA] Applying baseline display mode [%dx%dx%d] for %ls.\n", width, height, refreshRates.preferred, deviceName);

    devMode.dmPelsWidth = width;
    devMode.dmPelsHeight = height;
    devMode.dmDisplayFrequency = refreshRates.preferred;
    devMode.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;

    auto res = ChangeDisplaySettingsExW(deviceName, &devMode, NULL, CDS_UPDATEREGISTRY, NULL);

    if (res != ERROR_SUCCESS && refreshRates.alternate != refreshRates.preferred) {
      wprintf(L"[SUDOVDA] Failed to apply baseline display mode, trying alt mode: [%dx%dx%d].\n", width, height, refreshRates.alternate);
      devMode.dmDisplayFrequency = refreshRates.alternate;
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

  // Keep the lifetime heartbeat independent of Add/RemoveVirtualDisplay. A separate shared handle
  // avoids delaying a ping behind the process-wide mutation lock during topology transitions.
  const auto watchdogHandle = OpenDevice(&SUVDA_INTERFACE_GUID);
  if (watchdogHandle == INVALID_HANDLE_VALUE) {
    printf("[SUDOVDA] Failed to open dedicated watchdog heartbeat handle.\n");
    return false;
  }

  VIRTUAL_DISPLAY_GET_WATCHDOG_OUT watchdogOut {};
  if (!GetWatchdogTimeout(watchdogHandle, watchdogOut)) {
    printf("[SUDOVDA] Watchdog fetch failed!\n");
    CloseHandle(watchdogHandle);
    return false;
  }
  printf("[SUDOVDA] Watchdog: Timeout %d, Countdown %d\n", watchdogOut.Timeout, watchdogOut.Countdown);
  if (!watchdogOut.Timeout) {
    CloseHandle(watchdogHandle);
    return true;
  }
  if (watchdogOut.Timeout < 10) {
    printf(
      "[SUDOVDA] Warning: watchdog timeout is only %d seconds. "
      "Install or configure Apollo XR's recommended 30-second timeout to prevent "
      "the retained virtual display from being removed during GPU initialization.\n",
      watchdogOut.Timeout
    );
  }

  const auto sleepInterval = watchdogPingInterval(watchdogOut.Timeout);
  std::lock_guard owner_lock(watchdogOwnerMutex);
  try {
    watchdogThread = std::jthread([watchdogHandle, sleepInterval, failCb = std::move(failCb)](std::stop_token stop_token) {
      // This heartbeat is the lifetime lease for every SudoVDA monitor. Favor it over ordinary
      // application work so a busy encoder startup cannot let the driver remove an active desktop.
      if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        printf("[SUDOVDA] Failed to raise watchdog heartbeat thread priority: %lu\n", GetLastError());
      }

      uint8_t fail_count = 0;
      while (!stop_token.stop_requested()) {
        bool watchdog_failed = false;
        if (PingDriver(watchdogHandle)) {
          fail_count = 0;
        } else if (++fail_count > 3) {
          {
            std::lock_guard device_lock(virtualDisplayMutationMutex);
            closeDriverHandleLocked();
          }
          watchdog_failed = true;
        }
        if (watchdog_failed) {
          failCb();
          break;
        }

        std::unique_lock wait_lock(watchdogWaitMutex);
        watchdogWake.wait_for(wait_lock, stop_token, sleepInterval, []() {
          return false;
        });
      }
      CloseHandle(watchdogHandle);
    });
  } catch (...) {
    CloseHandle(watchdogHandle);
    printf("[SUDOVDA] Failed to start watchdog heartbeat thread.\n");
    return false;
  }
  return true;
}

#ifdef SUNSHINE_TESTS
uint32_t watchdogPingIntervalMsForTest(uint32_t timeoutSeconds) {
  return static_cast<uint32_t>(watchdogPingInterval(timeoutSeconds).count());
}
#endif

namespace {
bool queryDisplayConfig(
  UINT32 flags,
  std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
  std::vector<DISPLAYCONFIG_MODE_INFO> &modes,
  LONG *failureStatus = nullptr
) {
  // The active-path count can change between the size and data calls while Windows is applying a
  // hotplug, HDR, or IddCx topology transition. Retry that documented race instead of presenting a
  // transient empty desktop to callers.
  for (int attempt = 0; attempt < 8; ++attempt) {
    UINT32 pathCount = 0;
    UINT32 modeCount = 0;
    const auto sizeStatus = GetDisplayConfigBufferSizes(flags, &pathCount, &modeCount);
    if (sizeStatus != ERROR_SUCCESS) {
      if (sizeStatus != ERROR_INSUFFICIENT_BUFFER) {
        if (failureStatus) {
          *failureStatus = sizeStatus;
        }
        return false;
      }
      Sleep(1u << std::min(attempt, 5));
      continue;
    }

    paths.resize(pathCount);
    modes.resize(modeCount);
    const auto queryStatus = QueryDisplayConfig(
      flags,
      &pathCount,
      paths.data(),
      &modeCount,
      modes.data(),
      nullptr
    );
    if (queryStatus == ERROR_SUCCESS) {
      paths.resize(pathCount);
      modes.resize(modeCount);
      if (failureStatus) {
        *failureStatus = ERROR_SUCCESS;
      }
      return true;
    }
    if (queryStatus != ERROR_INSUFFICIENT_BUFFER) {
      if (failureStatus) {
        *failureStatus = queryStatus;
      }
      return false;
    }
    Sleep(1u << std::min(attempt, 5));
  }

  paths.clear();
  modes.clear();
  if (failureStatus) {
    *failureStatus = ERROR_INSUFFICIENT_BUFFER;
  }
  return false;
}
}  // namespace

bool queryActiveDisplayConfig(
  std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
  std::vector<DISPLAYCONFIG_MODE_INFO> &modes
) {
  return queryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, paths, modes);
}

namespace {
  enum class color_contract_api_e {
    modern,
    legacy,
  };

  struct display_color_contract_t {
    color_contract_api_e api {color_contract_api_e::legacy};
    bool legacy_advanced_color_enabled = false;
    bool hdr_user_enabled = false;
    bool wcg_user_enabled = false;
    bool advanced_color_active = false;
    DISPLAYCONFIG_ADVANCED_COLOR_MODE active_mode {DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR};
  };

  std::optional<display_color_contract_t> queryDisplayColorContract(
    const LUID &adapter_id,
    UINT32 target_id
  ) {
    DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO_2 info2 {};
    info2.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO_2;
    info2.header.size = sizeof(info2);
    info2.header.adapterId = adapter_id;
    info2.header.id = target_id;
    const auto modern_status = DisplayConfigGetDeviceInfo(&info2.header);
    if (modern_status == ERROR_SUCCESS) {
      return display_color_contract_t {
        color_contract_api_e::modern,
        false,
        info2.highDynamicRangeUserEnabled != 0,
        info2.wideColorUserEnabled != 0,
        info2.advancedColorActive != 0,
        info2.activeColorMode,
      };
    }
    if (modern_status != ERROR_INVALID_PARAMETER &&
        modern_status != ERROR_NOT_SUPPORTED) {
      return std::nullopt;
    }

    DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO info {};
    info.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO;
    info.header.size = sizeof(info);
    info.header.adapterId = adapter_id;
    info.header.id = target_id;
    if (DisplayConfigGetDeviceInfo(&info.header) == ERROR_SUCCESS) {
      return display_color_contract_t {
        color_contract_api_e::legacy,
        info.advancedColorEnabled != 0,
      };
    }
    return std::nullopt;
  }

  bool setDisplayWcg(const LUID &adapter_id, UINT32 target_id, bool enabled) {
    DISPLAYCONFIG_SET_WCG_STATE state {};
    state.header.type = DISPLAYCONFIG_DEVICE_INFO_SET_WCG_STATE;
    state.header.size = sizeof(state);
    state.header.adapterId = adapter_id;
    state.header.id = target_id;
    state.enableWcg = enabled;
    return DisplayConfigSetDeviceInfo(&state.header) == ERROR_SUCCESS;
  }

  bool setDisplayHdrModern(const LUID &adapter_id, UINT32 target_id, bool enabled) {
    DISPLAYCONFIG_SET_HDR_STATE state {};
    state.header.type = DISPLAYCONFIG_DEVICE_INFO_SET_HDR_STATE;
    state.header.size = sizeof(state);
    state.header.adapterId = adapter_id;
    state.header.id = target_id;
    state.enableHdr = enabled;
    return DisplayConfigSetDeviceInfo(&state.header) == ERROR_SUCCESS;
  }

  bool reconcileDisplayColorContract(
    const LUID &adapter_id,
    UINT32 target_id,
    const display_color_contract_t &expected
  ) {
    const auto current = queryDisplayColorContract(adapter_id, target_id);
    if (!current || current->api != expected.api) {
      return false;
    }
    if (expected.api == color_contract_api_e::legacy) {
      if (current->legacy_advanced_color_enabled ==
          expected.legacy_advanced_color_enabled) {
        return true;
      }
      setDisplayHDR(adapter_id, target_id, expected.legacy_advanced_color_enabled);
      return false;
    }

    if (current->hdr_user_enabled != expected.hdr_user_enabled) {
      setDisplayHdrModern(adapter_id, target_id, expected.hdr_user_enabled);
      return false;
    }
    if (current->wcg_user_enabled != expected.wcg_user_enabled) {
      setDisplayWcg(adapter_id, target_id, expected.wcg_user_enabled);
      return false;
    }
    if (current->advanced_color_active == expected.advanced_color_active &&
        current->active_mode == expected.active_mode) {
      return true;
    }

    // User preferences match but the active mode has not settled. Reassert only the expected
    // non-SDR mode; policy-limited SDR is observed until the bounded caller deadline instead of
    // changing either saved user preference.
    if (expected.active_mode == DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR) {
      setDisplayHdrModern(adapter_id, target_id, true);
    } else if (expected.active_mode == DISPLAYCONFIG_ADVANCED_COLOR_MODE_WCG) {
      setDisplayWcg(adapter_id, target_id, true);
    }
    return false;
  }

  struct survivor_color_state_t {
    std::wstring device_path;
    display_color_contract_t contract;
  };

  struct detach_snapshot_t {
    detach_plan_e state {detach_plan_e::ambiguous_identity};
    UINT32 set_display_config_awareness_flags = 0;
    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    std::vector<DISPLAYCONFIG_MODE_INFO> modes;
    std::vector<survivor_color_state_t> survivor_color_states;
  };

  std::optional<detach_snapshot_t> buildVirtualDisplayDetachSnapshot(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
    std::wstring_view learnedDevicePath
  ) {
    detach_snapshot_t snapshot;
    constexpr UINT32 base_query_flags =
      QDC_ONLY_ACTIVE_PATHS | QDC_VIRTUAL_MODE_AWARE;
    LONG refresh_aware_status = ERROR_SUCCESS;
    if (queryDisplayConfig(
          base_query_flags | QDC_VIRTUAL_REFRESH_RATE_AWARE,
          snapshot.paths,
          snapshot.modes,
          &refresh_aware_status
        )) {
      snapshot.set_display_config_awareness_flags =
        SDC_VIRTUAL_MODE_AWARE | SDC_VIRTUAL_REFRESH_RATE_AWARE;
    } else if ((refresh_aware_status == ERROR_INVALID_PARAMETER ||
                refresh_aware_status == ERROR_NOT_SUPPORTED) &&
               queryDisplayConfig(base_query_flags, snapshot.paths, snapshot.modes)) {
      // Windows 10 understands virtual modes but not virtual refresh-rate awareness.
      snapshot.set_display_config_awareness_flags = SDC_VIRTUAL_MODE_AWARE;
    } else {
      return std::nullopt;
    }

    std::vector<std::wstring> candidate_device_paths;
    candidate_device_paths.reserve(snapshot.paths.size());
    for (const auto &path : snapshot.paths) {
      DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
      target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
      target_name.header.size = sizeof(target_name);
      target_name.header.adapterId = path.targetInfo.adapterId;
      target_name.header.id = path.targetInfo.id;
      if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS ||
          target_name.monitorDevicePath[0] == L'\0') {
        // A failed target query could be the retiring display after a topology renumber. Treat the
        // whole snapshot as indeterminate instead of detaching a merely similar monitor.
        return std::nullopt;
      }
      candidate_device_paths.emplace_back(target_name.monitorDevicePath);
    }

    const auto plan = prepareVirtualDisplayDetachPaths(
      identity,
      learnedDevicePath,
      candidate_device_paths,
      snapshot.paths
    );
    snapshot.state = plan.state;
    if (snapshot.state == detach_plan_e::ready &&
        !rebaseSurvivingSourceModes(snapshot.paths, snapshot.modes)) {
      return std::nullopt;
    }
    if (snapshot.state == detach_plan_e::ready) {
      for (std::size_t index = 0; index < snapshot.paths.size(); ++index) {
        const auto &path = snapshot.paths[index];
        if ((path.flags & DISPLAYCONFIG_PATH_ACTIVE) == 0) {
          continue;
        }
        const auto color_contract =
          queryDisplayColorContract(path.targetInfo.adapterId, path.targetInfo.id);
        if (!color_contract) {
          // SetDisplayConfig can change Advanced Color state. Do not mutate a desktop whose
          // surviving target color contract cannot first be captured.
          return std::nullopt;
        }
        snapshot.survivor_color_states.push_back({
          candidate_device_paths[index],
          *color_contract,
        });
      }
    }
    return snapshot;
  }

  bool restoreSurvivorColorStatesOnce(
    const std::vector<survivor_color_state_t> &expected_states
  ) {
    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    std::vector<DISPLAYCONFIG_MODE_INFO> modes;
    if (!queryActiveDisplayConfig(paths, modes)) {
      return false;
    }
    std::vector<bool> matched(expected_states.size(), false);

    for (const auto &path : paths) {
      DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
      target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
      target_name.header.size = sizeof(target_name);
      target_name.header.adapterId = path.targetInfo.adapterId;
      target_name.header.id = path.targetInfo.id;
      if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS ||
          target_name.monitorDevicePath[0] == L'\0') {
        return false;
      }

      for (std::size_t index = 0; index < expected_states.size(); ++index) {
        if (expected_states[index].device_path != target_name.monitorDevicePath) {
          continue;
        }
        if (matched[index]) {
          return false;
        }
        matched[index] = true;
        if (!reconcileDisplayColorContract(
              path.targetInfo.adapterId,
              path.targetInfo.id,
              expected_states[index].contract
            )) {
          // Advanced Color can renumber this target. Perform at most one setter per fresh CCD
          // snapshot, then resolve every survivor again before trusting the result.
          return false;
        }
      }
    }
    return std::find(matched.begin(), matched.end(), false) == matched.end();
  }

  desktop_detach_result_t detachPlanFailure(detach_plan_e state) {
    switch (state) {
      case detach_plan_e::already_inactive:
        return {desktop_detach_state_e::already_inactive, ERROR_SUCCESS};
      case detach_plan_e::skipped_only_active:
        return {desktop_detach_state_e::skipped_only_active, ERROR_SUCCESS};
      case detach_plan_e::ambiguous_identity:
        return {desktop_detach_state_e::ambiguous_identity, ERROR_SUCCESS};
      case detach_plan_e::ready:
        break;
    }
    return {desktop_detach_state_e::topology_query_failed, ERROR_INVALID_DATA};
  }
}  // namespace

desktop_detach_result_t deactivateVirtualDisplay(
  const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
  std::wstring_view devicePath,
  std::chrono::milliseconds timeout
) {
  const auto deadline =
    std::chrono::steady_clock::now() + std::max(timeout, std::chrono::milliseconds::zero());
  auto snapshot = buildVirtualDisplayDetachSnapshot(identity, devicePath);
  if (!snapshot) {
    return {desktop_detach_state_e::topology_query_failed, ERROR_GEN_FAILURE};
  }
  bool was_already_inactive = snapshot->state == detach_plan_e::already_inactive;
  if (!was_already_inactive && snapshot->state != detach_plan_e::ready) {
    return detachPlanFailure(snapshot->state);
  }

  constexpr UINT32 validate_flags = SDC_VALIDATE | SDC_USE_SUPPLIED_DISPLAY_CONFIG;
  constexpr UINT32 apply_flags = SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG;
  std::vector<survivor_color_state_t> expected_color_states;

  if (!was_already_inactive) {
    LONG validation_status = SetDisplayConfig(
      static_cast<UINT32>(snapshot->paths.size()),
      snapshot->paths.data(),
      static_cast<UINT32>(snapshot->modes.size()),
      snapshot->modes.data(),
      validate_flags | snapshot->set_display_config_awareness_flags
    );
    if (validation_status != ERROR_SUCCESS) {
      return {desktop_detach_state_e::validation_failed, validation_status};
    }

    // Validation does not reserve the desktop topology. Re-query immediately before applying so a
    // monitor/HDR change that completed during validation is preserved rather than overwritten.
    snapshot = buildVirtualDisplayDetachSnapshot(identity, devicePath);
    if (!snapshot) {
      return {desktop_detach_state_e::topology_query_failed, ERROR_GEN_FAILURE};
    }
    was_already_inactive = snapshot->state == detach_plan_e::already_inactive;
    if (!was_already_inactive && snapshot->state != detach_plan_e::ready) {
      return detachPlanFailure(snapshot->state);
    }

    if (!was_already_inactive) {
      validation_status = SetDisplayConfig(
        static_cast<UINT32>(snapshot->paths.size()),
        snapshot->paths.data(),
        static_cast<UINT32>(snapshot->modes.size()),
        snapshot->modes.data(),
        validate_flags | snapshot->set_display_config_awareness_flags
      );
      if (validation_status != ERROR_SUCCESS) {
        return {desktop_detach_state_e::validation_failed, validation_status};
      }

      expected_color_states = snapshot->survivor_color_states;
      const LONG apply_status = SetDisplayConfig(
        static_cast<UINT32>(snapshot->paths.size()),
        snapshot->paths.data(),
        static_cast<UINT32>(snapshot->modes.size()),
        snapshot->modes.data(),
        apply_flags | snapshot->set_display_config_awareness_flags
      );
      if (apply_status != ERROR_SUCCESS) {
        return {desktop_detach_state_e::apply_failed, apply_status};
      }
    }
  }

  // Removing the active desktop path and removing the IddCx monitor must be separate events.
  // Require stable active-path absence, then leave Explorer time to finish migrating windows and
  // rebuilding its multi-monitor taskbar before the driver monitor disappears.
  auto stable_since = std::chrono::steady_clock::now();
  constexpr auto shell_settle_time = std::chrono::milliseconds(500);
  constexpr unsigned int required_absent_observations = 3;
  unsigned int consecutive_absent_observations = 0;
  bool color_states_match = true;
  while (true) {
    const auto observation = buildVirtualDisplayDetachSnapshot(identity, devicePath);
    color_states_match =
      expected_color_states.empty() ||
      restoreSurvivorColorStatesOnce(expected_color_states);
    if (observation &&
        observation->state == detach_plan_e::already_inactive &&
        color_states_match) {
      ++consecutive_absent_observations;
    } else {
      consecutive_absent_observations = 0;
      stable_since = std::chrono::steady_clock::now();
    }

    const auto now = std::chrono::steady_clock::now();
    if (consecutive_absent_observations >= required_absent_observations &&
        now - stable_since >= shell_settle_time) {
      return {
        was_already_inactive ?
          desktop_detach_state_e::already_inactive :
          desktop_detach_state_e::detached,
        ERROR_SUCCESS,
      };
    }
    if (now >= deadline) {
      return {
        color_states_match ?
          desktop_detach_state_e::settle_timeout :
          desktop_detach_state_e::color_restore_timeout,
        ERROR_TIMEOUT,
      };
    }
    std::this_thread::sleep_for(std::min(
      std::chrono::milliseconds(50),
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
    ));
  }
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

display_identity_state_e queryVirtualDisplayRetirementState(
  const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
  std::wstring_view devicePath,
  std::wstring_view displayName
) {
  const auto active = queryVirtualDisplayIdentity(identity, devicePath, displayName);
  if (active.state != display_identity_state_e::absent) {
    return active.state;
  }

  // A desktop detach intentionally removes the target from QDC_ONLY_ACTIVE_PATHS while the IddCx
  // monitor object is still alive. Keep the retirement barrier until Windows stops advertising
  // that exact target as available in the complete path set.
  std::vector<DISPLAYCONFIG_PATH_INFO> paths;
  std::vector<DISPLAYCONFIG_MODE_INFO> modes;
  if (!queryDisplayConfig(QDC_ALL_PATHS, paths, modes)) {
    return display_identity_state_e::indeterminate;
  }

  std::vector<retirement_candidate_t> candidates;
  candidates.reserve(paths.size());
  for (const auto &path : paths) {
    retirement_candidate_t candidate {
      {path.targetInfo.adapterId, path.targetInfo.id},
      {},
      path.targetInfo.targetAvailable != FALSE,
      true,
    };
    if (candidate.target_available) {
      DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
      target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
      target_name.header.size = sizeof(target_name);
      target_name.header.adapterId = path.targetInfo.adapterId;
      target_name.header.id = path.targetInfo.id;
      if (DisplayConfigGetDeviceInfo(&target_name.header) == ERROR_SUCCESS) {
        candidate.device_path = target_name.monitorDevicePath;
      } else {
        candidate.device_info_available = false;
      }
    }
    candidates.push_back(std::move(candidate));
  }

  return classifyVirtualDisplayRetirementState(identity, devicePath, candidates);
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

bool rebaseVirtualDisplaySurvivorsForTest(
  const std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
  std::vector<DISPLAYCONFIG_MODE_INFO> &modes
) {
  return rebaseSurvivingSourceModes(paths, modes);
}

desktop_detach_plan_e prepareVirtualDisplayDetachPathsForTest(
  const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
  std::wstring_view learnedDevicePath,
  const std::vector<std::wstring> &candidateDevicePaths,
  std::vector<DISPLAYCONFIG_PATH_INFO> &paths
) {
  switch (prepareVirtualDisplayDetachPaths(
    identity,
    learnedDevicePath,
    candidateDevicePaths,
    paths
  ).state) {
    case detach_plan_e::ready:
      return desktop_detach_plan_e::ready;
    case detach_plan_e::already_inactive:
      return desktop_detach_plan_e::already_inactive;
    case detach_plan_e::skipped_only_active:
      return desktop_detach_plan_e::skipped_only_active;
    case detach_plan_e::ambiguous_identity:
      return desktop_detach_plan_e::ambiguous_identity;
  }
  return desktop_detach_plan_e::ambiguous_identity;
}

display_identity_state_e virtualDisplayRetirementStateForTest(
  const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
  std::wstring_view learnedDevicePath,
  const std::vector<VDISPLAY::retirement_path_candidate_t> &candidates
) {
  std::vector<retirement_candidate_t> internal_candidates;
  internal_candidates.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    internal_candidates.push_back({
      candidate.identity,
      candidate.device_path,
      candidate.target_available,
      candidate.device_info_available,
    });
  }
  return classifyVirtualDisplayRetirementState(
    identity,
    learnedDevicePath,
    internal_candidates
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
    result.render_adapter_luid = adapterLuid;
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
  return createVirtualDisplayWithRenderAdapter(
    s_client_uid,
    s_client_name,
    width,
    height,
    fps,
    guid,
    primaryDisplayAdapterLuid()
  );
}

creation_result_t createVirtualDisplayWithRenderAdapter(
  const char *s_client_uid,
  const char *s_client_name,
  uint32_t width,
  uint32_t height,
  uint32_t fps,
  const GUID &guid,
  const std::optional<LUID> &adapterLuid
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
