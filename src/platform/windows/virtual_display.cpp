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

    std::vector<retirement_candidate_t> coalesceRetirementCandidates(
      const std::vector<retirement_candidate_t> &candidates
    ) {
      std::vector<retirement_candidate_t> coalesced;
      coalesced.reserve(candidates.size());
      for (const auto &candidate : candidates) {
        const auto existing = std::find_if(
          coalesced.begin(),
          coalesced.end(),
          [&](const retirement_candidate_t &entry) {
            return sameLuid(entry.identity.AdapterLuid, candidate.identity.AdapterLuid) &&
                   entry.identity.TargetId == candidate.identity.TargetId;
          }
        );
        if (existing == coalesced.end()) {
          coalesced.push_back(candidate);
          continue;
        }

        existing->target_available =
          existing->target_available || candidate.target_available;
        if (existing->device_path.empty() && !candidate.device_path.empty()) {
          existing->device_path = candidate.device_path;
        }
        existing->device_info_available =
          existing->device_info_available && candidate.device_info_available;
      }
      return coalesced;
    }

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

    struct detach_settle_evidence_t {
      bool path_confirmed_inactive = false;
      bool shell_settled = false;
    };

    detach_settle_evidence_t desktopDetachEvidence(
      bool latest_path_inactive,
      unsigned int consecutive_inactive_observations,
      std::chrono::milliseconds continuously_inactive_for
    ) {
      constexpr unsigned int required_inactive_observations = 3;
      constexpr auto shell_settle_time = std::chrono::milliseconds(500);
      const bool path_confirmed_inactive =
        latest_path_inactive &&
        consecutive_inactive_observations >= required_inactive_observations;
      return {
        path_confirmed_inactive,
        path_confirmed_inactive &&
          continuously_inactive_for >= shell_settle_time,
      };
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
      "Install or configure Sunshine 3D's recommended 30-second timeout to prevent "
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

  enum class color_reconcile_action_internal_e {
    settled,
    wait_for_active_mode,
    set_legacy_advanced_color,
    set_hdr_user_state,
    set_wcg_user_state,
  };

  struct color_setter_attempts_t {
    bool legacy = false;
    bool hdr = false;
    bool wcg = false;
    std::chrono::steady_clock::time_point legacy_retry_not_before {};
    std::chrono::steady_clock::time_point hdr_retry_not_before {};
    std::chrono::steady_clock::time_point wcg_retry_not_before {};
  };

  struct active_mode_settle_state_t {
    unsigned int consecutive_mismatch_observations = 0;
    std::chrono::steady_clock::time_point mismatch_since {};
    bool grace_expired = false;
  };

  bool colorUserStatesMatch(
    const display_color_contract_t &current,
    const display_color_contract_t &expected
  ) {
    if (current.api != expected.api) {
      return false;
    }
    if (expected.api == color_contract_api_e::legacy) {
      return current.legacy_advanced_color_enabled ==
             expected.legacy_advanced_color_enabled;
    }
    return current.hdr_user_enabled == expected.hdr_user_enabled &&
           current.wcg_user_enabled == expected.wcg_user_enabled;
  }

  bool activeColorModeObservationSettled(
    bool user_states_match,
    bool latest_active_mode_mismatch,
    unsigned int consecutive_mismatch_observations,
    std::chrono::milliseconds continuously_mismatched_for
  ) {
    constexpr unsigned int required_mismatch_observations = 3;
    constexpr auto active_mode_grace = std::chrono::milliseconds(500);
    return user_states_match &&
           latest_active_mode_mismatch &&
           consecutive_mismatch_observations >= required_mismatch_observations &&
           continuously_mismatched_for >= active_mode_grace;
  }

  bool consumeColorSetterAttempt(
    color_reconcile_action_internal_e action,
    color_setter_attempts_t &attempts
  ) {
    bool *attempted = nullptr;
    switch (action) {
      case color_reconcile_action_internal_e::set_legacy_advanced_color:
        attempted = &attempts.legacy;
        break;
      case color_reconcile_action_internal_e::set_hdr_user_state:
        attempted = &attempts.hdr;
        break;
      case color_reconcile_action_internal_e::set_wcg_user_state:
        attempted = &attempts.wcg;
        break;
      case color_reconcile_action_internal_e::settled:
      case color_reconcile_action_internal_e::wait_for_active_mode:
        return false;
    }
    if (*attempted) {
      return false;
    }
    *attempted = true;
    return true;
  }

  bool colorSetterWasAccepted(
    color_reconcile_action_internal_e action,
    const color_setter_attempts_t &attempts
  ) {
    switch (action) {
      case color_reconcile_action_internal_e::set_legacy_advanced_color:
        return attempts.legacy;
      case color_reconcile_action_internal_e::set_hdr_user_state:
        return attempts.hdr;
      case color_reconcile_action_internal_e::set_wcg_user_state:
        return attempts.wcg;
      case color_reconcile_action_internal_e::settled:
      case color_reconcile_action_internal_e::wait_for_active_mode:
        return false;
    }
    return false;
  }

  std::chrono::steady_clock::time_point &colorSetterRetryNotBefore(
    color_reconcile_action_internal_e action,
    color_setter_attempts_t &attempts
  ) {
    switch (action) {
      case color_reconcile_action_internal_e::set_legacy_advanced_color:
        return attempts.legacy_retry_not_before;
      case color_reconcile_action_internal_e::set_hdr_user_state:
        return attempts.hdr_retry_not_before;
      case color_reconcile_action_internal_e::set_wcg_user_state:
        return attempts.wcg_retry_not_before;
      case color_reconcile_action_internal_e::settled:
      case color_reconcile_action_internal_e::wait_for_active_mode:
        break;
    }
    return attempts.hdr_retry_not_before;
  }

  bool colorSetterRetryAvailable(
    color_reconcile_action_internal_e action,
    color_setter_attempts_t &attempts,
    std::chrono::steady_clock::time_point now
  ) {
    return !colorSetterWasAccepted(action, attempts) &&
           now >= colorSetterRetryNotBefore(action, attempts);
  }

  void deferRejectedColorSetter(
    color_reconcile_action_internal_e action,
    color_setter_attempts_t &attempts,
    std::chrono::steady_clock::time_point now
  ) {
    constexpr auto rejected_setter_backoff = std::chrono::seconds(1);
    colorSetterRetryNotBefore(action, attempts) =
      now + rejected_setter_backoff;
  }

  color_reconcile_action_internal_e colorReconcileAction(
    const display_color_contract_t &current,
    const display_color_contract_t &expected,
    const color_setter_attempts_t *attempts = nullptr
  ) {
    if (current.api != expected.api) {
      return color_reconcile_action_internal_e::wait_for_active_mode;
    }
    if (expected.api == color_contract_api_e::legacy) {
      if (current.legacy_advanced_color_enabled !=
          expected.legacy_advanced_color_enabled) {
        return !attempts || !attempts->legacy ?
                 color_reconcile_action_internal_e::set_legacy_advanced_color :
                 color_reconcile_action_internal_e::wait_for_active_mode;
      }
      return color_reconcile_action_internal_e::settled;
    }
    const bool hdr_mismatch =
      current.hdr_user_enabled != expected.hdr_user_enabled;
    const bool wcg_mismatch =
      current.wcg_user_enabled != expected.wcg_user_enabled;
    if (hdr_mismatch && (!attempts || !attempts->hdr)) {
      return color_reconcile_action_internal_e::set_hdr_user_state;
    }
    if (wcg_mismatch && (!attempts || !attempts->wcg)) {
      return color_reconcile_action_internal_e::set_wcg_user_state;
    }
    if (hdr_mismatch || wcg_mismatch) {
      return color_reconcile_action_internal_e::wait_for_active_mode;
    }
    if (current.advanced_color_active == expected.advanced_color_active &&
        current.active_mode == expected.active_mode) {
      return color_reconcile_action_internal_e::settled;
    }
    // activeColorMode is an observed policy/driver result. Reasserting an already-matching user
    // preference can itself trigger another display transition and prevent this state from
    // settling, so wait without issuing a setter.
    return color_reconcile_action_internal_e::wait_for_active_mode;
  }

  color_reconcile_action_internal_e colorReconcileActionWithRetryCadence(
    const display_color_contract_t &current,
    const display_color_contract_t &expected,
    color_setter_attempts_t &attempts,
    std::chrono::steady_clock::time_point now
  ) {
    auto action = colorReconcileAction(current, expected, &attempts);
    if (action == color_reconcile_action_internal_e::set_legacy_advanced_color) {
      return colorSetterRetryAvailable(action, attempts, now) ?
               action :
               color_reconcile_action_internal_e::wait_for_active_mode;
    }
    if (expected.api != color_contract_api_e::modern ||
        current.api != color_contract_api_e::modern) {
      return action;
    }

    const bool hdr_mismatch =
      current.hdr_user_enabled != expected.hdr_user_enabled;
    const bool wcg_mismatch =
      current.wcg_user_enabled != expected.wcg_user_enabled;
    if (hdr_mismatch &&
        colorSetterRetryAvailable(
          color_reconcile_action_internal_e::set_hdr_user_state,
          attempts,
          now
        )) {
      return color_reconcile_action_internal_e::set_hdr_user_state;
    }
    if (wcg_mismatch &&
        colorSetterRetryAvailable(
          color_reconcile_action_internal_e::set_wcg_user_state,
          attempts,
          now
        )) {
      return color_reconcile_action_internal_e::set_wcg_user_state;
    }
    if (hdr_mismatch || wcg_mismatch) {
      return color_reconcile_action_internal_e::wait_for_active_mode;
    }
    return action;
  }

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
    const display_color_contract_t &expected,
    color_setter_attempts_t &attempts,
    active_mode_settle_state_t &active_mode_settle
  ) {
    const auto current = queryDisplayColorContract(adapter_id, target_id);
    if (!current) {
      return false;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto action =
      colorReconcileActionWithRetryCadence(*current, expected, attempts, now);
    switch (action) {
      case color_reconcile_action_internal_e::settled:
        active_mode_settle = {};
        return true;
      case color_reconcile_action_internal_e::set_legacy_advanced_color:
        active_mode_settle = {};
        if (colorSetterRetryAvailable(action, attempts, now)) {
          if (setDisplayHDR(
                adapter_id,
                target_id,
                expected.legacy_advanced_color_enabled
              )) {
            consumeColorSetterAttempt(action, attempts);
          } else {
            deferRejectedColorSetter(action, attempts, now);
          }
        }
        break;
      case color_reconcile_action_internal_e::set_hdr_user_state:
        active_mode_settle = {};
        if (colorSetterRetryAvailable(action, attempts, now)) {
          if (setDisplayHdrModern(adapter_id, target_id, expected.hdr_user_enabled)) {
            consumeColorSetterAttempt(action, attempts);
          } else {
            deferRejectedColorSetter(action, attempts, now);
          }
        }
        break;
      case color_reconcile_action_internal_e::set_wcg_user_state:
        active_mode_settle = {};
        if (colorSetterRetryAvailable(action, attempts, now)) {
          if (setDisplayWcg(adapter_id, target_id, expected.wcg_user_enabled)) {
            consumeColorSetterAttempt(action, attempts);
          } else {
            deferRejectedColorSetter(action, attempts, now);
          }
        }
        break;
      case color_reconcile_action_internal_e::wait_for_active_mode:
        if (!colorUserStatesMatch(*current, expected)) {
          active_mode_settle = {};
          break;
        }
        if (active_mode_settle.grace_expired) {
          return true;
        }
        {
          if (active_mode_settle.consecutive_mismatch_observations == 0) {
            active_mode_settle.mismatch_since = now;
          }
          ++active_mode_settle.consecutive_mismatch_observations;
          active_mode_settle.grace_expired = activeColorModeObservationSettled(
            true,
            true,
            active_mode_settle.consecutive_mismatch_observations,
            std::chrono::duration_cast<std::chrono::milliseconds>(
              now - active_mode_settle.mismatch_since
            )
          );
          return active_mode_settle.grace_expired;
        }
    }
    return false;
  }

  struct survivor_color_state_t {
    std::wstring device_path;
    display_color_contract_t contract;
    color_setter_attempts_t setter_attempts;
    active_mode_settle_state_t active_mode_settle;
    bool retired = false;
    unsigned int consecutive_missing_observations = 0;
    std::chrono::steady_clock::time_point missing_since {};
  };

  bool missingColorSurvivorRetired(
    bool latest_missing,
    unsigned int consecutive_missing_observations,
    std::chrono::milliseconds continuously_missing_for
  ) {
    constexpr unsigned int required_missing_observations = 3;
    constexpr auto missing_settle_time = std::chrono::milliseconds(500);
    return latest_missing &&
           consecutive_missing_observations >= required_missing_observations &&
           continuously_missing_for >= missing_settle_time;
  }

  void observeColorSurvivorPresence(
    survivor_color_state_t &survivor,
    bool present,
    std::chrono::steady_clock::time_point now
  ) {
    if (present) {
      survivor.retired = false;
      survivor.consecutive_missing_observations = 0;
      survivor.missing_since = {};
      return;
    }
    if (survivor.retired) {
      return;
    }
    if (survivor.consecutive_missing_observations == 0) {
      survivor.missing_since = now;
    }
    ++survivor.consecutive_missing_observations;
    survivor.retired = missingColorSurvivorRetired(
      true,
      survivor.consecutive_missing_observations,
      std::chrono::duration_cast<std::chrono::milliseconds>(
        now - survivor.missing_since
      )
    );
  }

  void mergeSurvivorColorStates(
    std::vector<survivor_color_state_t> &expected_states,
    const std::vector<survivor_color_state_t> &current_states
  ) {
    for (const auto &current : current_states) {
      const auto existing = std::find_if(
        expected_states.begin(),
        expected_states.end(),
        [&](const survivor_color_state_t &expected) {
          return expected.device_path == current.device_path;
        }
      );
      if (existing == expected_states.end()) {
        expected_states.push_back(current);
      }
    }
  }

  void resetSurvivorColorTransitionState(
    std::vector<survivor_color_state_t> &expected_states
  ) {
    for (auto &expected : expected_states) {
      expected.setter_attempts = {};
      expected.active_mode_settle = {};
    }
  }

  bool anyActiveModeGraceExpired(
    const std::vector<survivor_color_state_t> &expected_states
  ) {
    return std::any_of(
      expected_states.begin(),
      expected_states.end(),
      [](const survivor_color_state_t &expected) {
        return expected.active_mode_settle.grace_expired;
      }
    );
  }

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
    std::vector<survivor_color_state_t> &expected_states
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
        observeColorSurvivorPresence(
          expected_states[index],
          true,
          std::chrono::steady_clock::now()
        );
        if (!reconcileDisplayColorContract(
              path.targetInfo.adapterId,
              path.targetInfo.id,
              expected_states[index].contract,
              expected_states[index].setter_attempts,
              expected_states[index].active_mode_settle
            )) {
          // Advanced Color can renumber this target. Perform each required user-state setter at
          // most once for the entire detach transition, then resolve every survivor again before
          // trusting the result.
          return false;
        }
      }
    }

    const auto now = std::chrono::steady_clock::now();
    bool all_resolved = true;
    for (std::size_t index = 0; index < expected_states.size(); ++index) {
      if (matched[index]) {
        continue;
      }
      auto &expected = expected_states[index];
      if (expected.retired) {
        continue;
      }
      observeColorSurvivorPresence(expected, false, now);
      if (!expected.retired) {
        all_resolved = false;
      }
    }
    return all_resolved;
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

struct desktop_detach_context_t {
  bool was_already_inactive = false;
  bool has_captured_color_contract = false;
  std::vector<survivor_color_state_t> expected_color_states;
};

namespace {
  bool detachRetryMustCaptureColorContract(
    bool has_context,
    bool context_has_captured_color_contract,
    bool path_needs_detach
  ) {
    return path_needs_detach &&
           (!has_context || !context_has_captured_color_contract);
  }
}

desktop_detach_result_t deactivateVirtualDisplay(
  const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
  std::wstring_view devicePath,
  std::chrono::milliseconds timeout,
  std::shared_ptr<desktop_detach_context_t> &context
) {
  const auto deadline =
    std::chrono::steady_clock::now() + std::max(timeout, std::chrono::milliseconds::zero());
  constexpr UINT32 validate_flags = SDC_VALIDATE | SDC_USE_SUPPLIED_DISPLAY_CONFIG;
  constexpr UINT32 apply_flags = SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG;
  const bool continuing_transition = static_cast<bool>(context);
  auto snapshot = buildVirtualDisplayDetachSnapshot(identity, devicePath);
  if (!snapshot) {
    return {desktop_detach_state_e::topology_query_failed, ERROR_GEN_FAILURE};
  }
  bool was_already_inactive =
    continuing_transition ?
      context->was_already_inactive :
      snapshot->state == detach_plan_e::already_inactive;
  if (snapshot->state != detach_plan_e::ready &&
      snapshot->state != detach_plan_e::already_inactive) {
    if (snapshot->state == detach_plan_e::skipped_only_active) {
      context.reset();
    }
    return detachPlanFailure(snapshot->state);
  }

  if (snapshot->state == detach_plan_e::ready) {
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
    if (snapshot->state != detach_plan_e::ready &&
        snapshot->state != detach_plan_e::already_inactive) {
      if (snapshot->state == detach_plan_e::skipped_only_active) {
        context.reset();
      }
      return detachPlanFailure(snapshot->state);
    }

    if (snapshot->state == detach_plan_e::ready) {
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

      const bool capture_color_contract = detachRetryMustCaptureColorContract(
        static_cast<bool>(context),
        context && context->has_captured_color_contract,
        true
      );
      if (!context) {
        context = std::make_shared<desktop_detach_context_t>();
      }
      if (capture_color_contract) {
        context->expected_color_states = snapshot->survivor_color_states;
      } else {
        // A monitor may join while a prior detach is settling. Preserve every original contract
        // already owned by the context and add only newly active survivors before another apply.
        mergeSurvivorColorStates(
          context->expected_color_states,
          snapshot->survivor_color_states
        );
      }
      context->was_already_inactive = false;
      context->has_captured_color_contract = true;
      was_already_inactive = false;
      // Each SetDisplayConfig apply is a new color transition. Retain the original user contract,
      // but allow one accepted setter per preference again and require a fresh active-mode grace.
      resetSurvivorColorTransitionState(context->expected_color_states);
      const LONG apply_status = SetDisplayConfig(
        static_cast<UINT32>(snapshot->paths.size()),
        snapshot->paths.data(),
        static_cast<UINT32>(snapshot->modes.size()),
        snapshot->modes.data(),
        apply_flags | snapshot->set_display_config_awareness_flags
      );
      if (apply_status != ERROR_SUCCESS) {
        if (!continuing_transition) {
          context.reset();
        }
        return {desktop_detach_state_e::apply_failed, apply_status};
      }
    }
  }
  if (!context) {
    context = std::make_shared<desktop_detach_context_t>();
    context->was_already_inactive = true;
    context->has_captured_color_contract = false;
    was_already_inactive = true;
  }

  // Removing the active desktop path and removing the IddCx monitor must be separate events.
  // Require stable active-path absence, then leave Explorer time to finish migrating windows and
  // rebuilding its multi-monitor taskbar before the driver monitor disappears.
  auto inactive_since = std::chrono::steady_clock::now();
  unsigned int consecutive_inactive_observations = 0;
  bool latest_path_inactive = false;
  bool color_states_match = true;
  while (true) {
    const auto observation = buildVirtualDisplayDetachSnapshot(identity, devicePath);
    color_states_match =
      context->expected_color_states.empty() ||
      restoreSurvivorColorStatesOnce(context->expected_color_states);
    latest_path_inactive =
      observation && observation->state == detach_plan_e::already_inactive;
    if (latest_path_inactive) {
      ++consecutive_inactive_observations;
    } else {
      consecutive_inactive_observations = 0;
      inactive_since = std::chrono::steady_clock::now();
    }

    const auto now = std::chrono::steady_clock::now();
    const auto evidence = desktopDetachEvidence(
      latest_path_inactive,
      consecutive_inactive_observations,
      std::chrono::duration_cast<std::chrono::milliseconds>(now - inactive_since)
    );
    if (evidence.shell_settled && color_states_match) {
      const bool active_mode_grace_expired =
        anyActiveModeGraceExpired(context->expected_color_states);
      context.reset();
      return {
        was_already_inactive ?
          desktop_detach_state_e::already_inactive :
          desktop_detach_state_e::detached,
        ERROR_SUCCESS,
        true,
        true,
        active_mode_grace_expired,
      };
    }
    if (now >= deadline) {
      return {
        evidence.shell_settled && !color_states_match ?
          desktop_detach_state_e::color_restore_timeout :
          desktop_detach_state_e::settle_timeout,
        ERROR_TIMEOUT,
        evidence.path_confirmed_inactive,
        evidence.shell_settled,
        anyActiveModeGraceExpired(context->expected_color_states),
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

  std::vector<retirement_candidate_t> raw_candidates;
  raw_candidates.reserve(paths.size());
  for (const auto &path : paths) {
    raw_candidates.push_back({
      {path.targetInfo.adapterId, path.targetInfo.id},
      {},
      path.targetInfo.targetAvailable != FALSE,
      true,
    });
  }

  // QDC_ALL_PATHS returns possible source-to-target combinations, so one target can appear many
  // times. Collapse those paths before calling DisplayConfigGetDeviceInfo; device identity and
  // availability are target properties and only need to be queried once.
  auto candidates = coalesceRetirementCandidates(raw_candidates);
  for (auto &candidate : candidates) {
    if (candidate.target_available) {
      DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
      target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
      target_name.header.size = sizeof(target_name);
      target_name.header.adapterId = candidate.identity.AdapterLuid;
      target_name.header.id = candidate.identity.TargetId;
      if (DisplayConfigGetDeviceInfo(&target_name.header) == ERROR_SUCCESS) {
        candidate.device_path = target_name.monitorDevicePath;
      } else {
        candidate.device_info_available = false;
      }
    }
  }

  return classifyVirtualDisplayRetirementState(identity, devicePath, candidates);
}

bool isDriverRemovalSafeAfterDesktopDetach(
  bool requiresActivePathAbsence,
  display_identity_state_e activePathState
) {
  // The only intentional exception is a sole active virtual output: no surviving desktop or
  // multi-monitor shell state exists to preserve. Every normal detach requires an authoritative
  // exact active-topology absence observation immediately before driver removal.
  return !requiresActivePathAbsence ||
         activePathState == display_identity_state_e::absent;
}

unsigned int advanceStableAbsenceEvidence(
  unsigned int priorAbsentObservations,
  bool evidenceInvalidated,
  display_identity_state_e latestState
) {
  if (evidenceInvalidated ||
      latestState != display_identity_state_e::absent) {
    return 0;
  }
  return priorAbsentObservations + 1;
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

desktop_detach_evidence_t desktopDetachEvidenceForTest(
  bool latestPathInactive,
  unsigned int consecutiveInactiveObservations,
  std::chrono::milliseconds continuouslyInactiveFor
) {
  const auto evidence = desktopDetachEvidence(
    latestPathInactive,
    consecutiveInactiveObservations,
    continuouslyInactiveFor
  );
  return {
    evidence.path_confirmed_inactive,
    evidence.shell_settled,
  };
}

color_reconcile_action_e colorReconcileActionForTest(
  bool legacyApi,
  bool currentLegacyEnabled,
  bool expectedLegacyEnabled,
  bool currentHdrUserEnabled,
  bool expectedHdrUserEnabled,
  bool currentWcgUserEnabled,
  bool expectedWcgUserEnabled,
  bool currentAdvancedColorActive,
  bool expectedAdvancedColorActive,
  DISPLAYCONFIG_ADVANCED_COLOR_MODE currentActiveMode,
  DISPLAYCONFIG_ADVANCED_COLOR_MODE expectedActiveMode
) {
  const display_color_contract_t current {
    legacyApi ? color_contract_api_e::legacy : color_contract_api_e::modern,
    currentLegacyEnabled,
    currentHdrUserEnabled,
    currentWcgUserEnabled,
    currentAdvancedColorActive,
    currentActiveMode,
  };
  const display_color_contract_t expected {
    legacyApi ? color_contract_api_e::legacy : color_contract_api_e::modern,
    expectedLegacyEnabled,
    expectedHdrUserEnabled,
    expectedWcgUserEnabled,
    expectedAdvancedColorActive,
    expectedActiveMode,
  };
  switch (colorReconcileAction(current, expected)) {
    case color_reconcile_action_internal_e::settled:
      return color_reconcile_action_e::settled;
    case color_reconcile_action_internal_e::wait_for_active_mode:
      return color_reconcile_action_e::wait_for_active_mode;
    case color_reconcile_action_internal_e::set_legacy_advanced_color:
      return color_reconcile_action_e::set_legacy_advanced_color;
    case color_reconcile_action_internal_e::set_hdr_user_state:
      return color_reconcile_action_e::set_hdr_user_state;
    case color_reconcile_action_internal_e::set_wcg_user_state:
      return color_reconcile_action_e::set_wcg_user_state;
  }
  return color_reconcile_action_e::wait_for_active_mode;
}

color_reconcile_action_e colorReconcileActionAfterAcceptedSettersForTest(
  bool currentHdrUserEnabled,
  bool expectedHdrUserEnabled,
  bool currentWcgUserEnabled,
  bool expectedWcgUserEnabled,
  bool hdrSetterAccepted,
  bool wcgSetterAccepted
) {
  const display_color_contract_t current {
    color_contract_api_e::modern,
    false,
    currentHdrUserEnabled,
    currentWcgUserEnabled,
  };
  const display_color_contract_t expected {
    color_contract_api_e::modern,
    false,
    expectedHdrUserEnabled,
    expectedWcgUserEnabled,
  };
  const color_setter_attempts_t attempts {
    false,
    hdrSetterAccepted,
    wcgSetterAccepted,
  };
  switch (colorReconcileAction(current, expected, &attempts)) {
    case color_reconcile_action_internal_e::settled:
      return color_reconcile_action_e::settled;
    case color_reconcile_action_internal_e::wait_for_active_mode:
      return color_reconcile_action_e::wait_for_active_mode;
    case color_reconcile_action_internal_e::set_legacy_advanced_color:
      return color_reconcile_action_e::set_legacy_advanced_color;
    case color_reconcile_action_internal_e::set_hdr_user_state:
      return color_reconcile_action_e::set_hdr_user_state;
    case color_reconcile_action_internal_e::set_wcg_user_state:
      return color_reconcile_action_e::set_wcg_user_state;
  }
  return color_reconcile_action_e::wait_for_active_mode;
}

unsigned int repeatedColorSetterAttemptCountForTest(
  color_reconcile_action_e action,
  unsigned int repetitions
) {
  color_reconcile_action_internal_e internal_action;
  switch (action) {
    case color_reconcile_action_e::settled:
      internal_action = color_reconcile_action_internal_e::settled;
      break;
    case color_reconcile_action_e::wait_for_active_mode:
      internal_action = color_reconcile_action_internal_e::wait_for_active_mode;
      break;
    case color_reconcile_action_e::set_legacy_advanced_color:
      internal_action = color_reconcile_action_internal_e::set_legacy_advanced_color;
      break;
    case color_reconcile_action_e::set_hdr_user_state:
      internal_action = color_reconcile_action_internal_e::set_hdr_user_state;
      break;
    case color_reconcile_action_e::set_wcg_user_state:
      internal_action = color_reconcile_action_internal_e::set_wcg_user_state;
      break;
  }

  color_setter_attempts_t attempts;
  unsigned int consumed = 0;
  for (unsigned int repetition = 0; repetition < repetitions; ++repetition) {
    consumed += consumeColorSetterAttempt(internal_action, attempts) ? 1u : 0u;
  }
  return consumed;
}

bool missingColorSurvivorRetiredForTest(
  bool latestMissing,
  unsigned int consecutiveMissingObservations,
  std::chrono::milliseconds continuouslyMissingFor
) {
  return missingColorSurvivorRetired(
    latestMissing,
    consecutiveMissingObservations,
    continuouslyMissingFor
  );
}

bool retiredColorSurvivorReactivatesForTest() {
  survivor_color_state_t survivor;
  survivor.retired = true;
  survivor.consecutive_missing_observations = 3;
  survivor.missing_since =
    std::chrono::steady_clock::now() - std::chrono::milliseconds(500);
  observeColorSurvivorPresence(
    survivor,
    true,
    std::chrono::steady_clock::now()
  );
  return !survivor.retired &&
         survivor.consecutive_missing_observations == 0 &&
         survivor.missing_since == std::chrono::steady_clock::time_point {};
}

bool detachRetryCapturesColorContractForTest(
  bool hasContext,
  bool contextHasCapturedColorContract,
  bool pathNeedsDetach
) {
  return detachRetryMustCaptureColorContract(
    hasContext,
    contextHasCapturedColorContract,
    pathNeedsDetach
  );
}

bool activeColorModeObservationSettledForTest(
  bool userStatesMatch,
  bool latestActiveModeMismatch,
  unsigned int consecutiveMismatchObservations,
  std::chrono::milliseconds continuouslyMismatchedFor
) {
  return activeColorModeObservationSettled(
    userStatesMatch,
    latestActiveModeMismatch,
    consecutiveMismatchObservations,
    continuouslyMismatchedFor
  );
}

bool rejectedColorSetterRetryAllowedForTest(
  std::chrono::milliseconds elapsedSinceFailure
) {
  color_setter_attempts_t attempts;
  const auto failure_time =
    std::chrono::steady_clock::time_point {} + std::chrono::seconds(10);
  deferRejectedColorSetter(
    color_reconcile_action_internal_e::set_hdr_user_state,
    attempts,
    failure_time
  );
  return colorSetterRetryAvailable(
    color_reconcile_action_internal_e::set_hdr_user_state,
    attempts,
    failure_time + elapsedSinceFailure
  );
}

bool colorSurvivorContractMergePreservesExistingForTest() {
  survivor_color_state_t original;
  original.device_path = LR"(\\?\DISPLAY#PHYSICAL#original)";
  original.contract.api = color_contract_api_e::modern;
  original.contract.hdr_user_enabled = true;

  auto changed_original = original;
  changed_original.contract.hdr_user_enabled = false;
  survivor_color_state_t newly_connected;
  newly_connected.device_path = LR"(\\?\DISPLAY#PHYSICAL#new)";

  std::vector<survivor_color_state_t> expected {original};
  mergeSurvivorColorStates(expected, {changed_original, newly_connected});
  return expected.size() == 2 &&
         expected[0].contract.hdr_user_enabled &&
         expected[1].device_path == newly_connected.device_path;
}

bool wcgSetterSelectedDuringHdrBackoffForTest() {
  const display_color_contract_t current {
    color_contract_api_e::modern,
    false,
    false,
    false,
  };
  const display_color_contract_t expected {
    color_contract_api_e::modern,
    false,
    true,
    true,
  };
  color_setter_attempts_t attempts;
  const auto failure_time =
    std::chrono::steady_clock::time_point {} + std::chrono::seconds(10);
  deferRejectedColorSetter(
    color_reconcile_action_internal_e::set_hdr_user_state,
    attempts,
    failure_time
  );
  return colorReconcileActionWithRetryCadence(
           current,
           expected,
           attempts,
           failure_time
         ) == color_reconcile_action_internal_e::set_wcg_user_state;
}

bool colorSurvivorTransitionStateResetsForReapplyForTest() {
  survivor_color_state_t survivor;
  survivor.setter_attempts.hdr = true;
  survivor.setter_attempts.hdr_retry_not_before =
    std::chrono::steady_clock::time_point {} + std::chrono::seconds(20);
  survivor.active_mode_settle.consecutive_mismatch_observations = 4;
  survivor.active_mode_settle.grace_expired = true;
  std::vector<survivor_color_state_t> expected {survivor};
  resetSurvivorColorTransitionState(expected);
  return !expected[0].setter_attempts.hdr &&
         expected[0].setter_attempts.hdr_retry_not_before ==
           std::chrono::steady_clock::time_point {} &&
         expected[0].active_mode_settle.consecutive_mismatch_observations == 0 &&
         !expected[0].active_mode_settle.grace_expired;
}

std::size_t coalescedRetirementCandidateCountForTest(
  const std::vector<retirement_path_candidate_t> &candidates
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
  return coalesceRetirementCandidates(internal_candidates).size();
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
