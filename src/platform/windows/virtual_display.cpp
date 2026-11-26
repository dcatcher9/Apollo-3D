#include "virtual_display.h"

#include "display_config.h"
#include "misc.h"
#include "src/logging.h"
#include "src/platform/common.h"

#include <algorithm>
#include <chrono>
#include <combaseapi.h>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cwctype>
#include <dxgi.h>
#include <dxgi1_6.h>
#include <initguid.h>
#include <limits>
#include <mutex>
#include <optional>
#include <setupapi.h>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>
#include <wrl/client.h>

using namespace SUDOVDA;

namespace VDISPLAY {
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

    template<typename Query, typename Now, typename WaitUntil>
    display_identity_query_t waitForDisplayIdentity(
      const LUID &adapterLuid,
      uint32_t targetId,
      const Query &query,
      const Now &now,
      const WaitUntil &waitUntil
    ) {
      // Preserve the original 20+40+80+160+320+640 ms wait budget, but observe publication after
      // every wait, including the final one. The deadline bounds waits, not Windows API latency.
      const auto deadline = now() + std::chrono::milliseconds(1260);
      auto retryInterval = std::chrono::milliseconds(20);
      for (;;) {
        auto published = query(adapterLuid, targetId);
        if (published.state == display_identity_state_e::present) {
          return published;
        }
        const auto observedAt = now();
        if (observedAt >= deadline) {
          return published;
        }
        waitUntil(std::min(observedAt + retryInterval, deadline));
        retryInterval *= 2;
      }
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
      // marker as well for already deployed variants.
      return containsCaseInsensitive(devicePath, L"SMKD1CE") ||
             containsCaseInsensitive(devicePath, L"SUDOVDA");
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
      const bool learnedPath = platf::display_config::same_device_path(learnedDevicePath, candidateDevicePath);
      const bool exactVirtualIdentity = platf::display_config::same_luid(expectedIdentity.AdapterLuid, candidateAdapter) &&
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
          platf::display_config::same_luid(identity.AdapterLuid, path.targetInfo.adapterId) &&
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
            !platf::display_config::same_luid(mode.adapterId, path.sourceInfo.adapterId) ||
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
            return platf::display_config::same_luid(entry.identity.AdapterLuid, candidate.identity.AdapterLuid) &&
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
          platf::display_config::same_luid(identity.AdapterLuid, candidate.identity.AdapterLuid) &&
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

LONG changeBaselineDisplaySettings(
  const wchar_t *deviceName,
  const int width,
  const int height,
  const baseline_refresh_rates_t refreshRates,
  const DWORD flags
) {
  DEVMODEW devMode {};
  devMode.dmSize = sizeof(devMode);
  if (!EnumDisplaySettingsW(deviceName, ENUM_CURRENT_SETTINGS, &devMode)) {
    return DISP_CHANGE_FAILED;
  }

  devMode.dmPelsWidth = width;
  devMode.dmPelsHeight = height;
  devMode.dmDisplayFrequency = refreshRates.preferred;
  devMode.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;
  auto result = ChangeDisplaySettingsExW(deviceName, &devMode, nullptr, flags, nullptr);
  if (result == DISP_CHANGE_SUCCESSFUL ||
      refreshRates.alternate == refreshRates.preferred) {
    return result;
  }

  devMode.dmDisplayFrequency = refreshRates.alternate;
  return ChangeDisplaySettingsExW(deviceName, &devMode, nullptr, flags, nullptr);
}

LONG applyDisplaySettings(const wchar_t *deviceName, int width, int height, int refresh_rate, bool persist_settings) {
  std::vector<DISPLAYCONFIG_PATH_INFO> pathArray;
  std::vector<DISPLAYCONFIG_MODE_INFO> modeArray;

  if (!queryActiveDisplayConfig(pathArray, modeArray)) {
    BOOST_LOG(error) << "[SUDOVDA] Failed to query display configuration.";
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

    if (platf::display_config::same_device_path(sourceName.viewGdiDeviceName, deviceName)) {
      BOOST_LOG(debug) << "[SUDOVDA] Display found: " << platf::to_utf8(deviceName);
      for (UINT32 j = 0; j < modeCount; j++) {
        if (
          modeArray[j].infoType == DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE &&
          platf::display_config::same_luid(modeArray[j].adapterId, sourceInfo->adapterId) &&
          modeArray[j].id == sourceInfo->id
        ) {
          auto *sourceMode = &modeArray[j].sourceMode;

          BOOST_LOG(debug) << "[SUDOVDA] Current mode found: [" << sourceMode->width << 'x' << sourceMode->height << 'x'
                            << targetInfo->refreshRate.Numerator << '/' << targetInfo->refreshRate.Denominator << ']';

          sourceMode->width = width;
          sourceMode->height = height;

          targetInfo->refreshRate = {(UINT32) refresh_rate, 1000};

          // Apply the changes
          LONG status = SetDisplayConfig(
            pathCount,
            pathArray.data(),
            modeCount,
            modeArray.data(),
            SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG | (persist_settings ? SDC_SAVE_TO_DATABASE : 0)
          );
          if (status != ERROR_SUCCESS) {
            BOOST_LOG(error) << "[SUDOVDA] Failed to apply display settings.";
          } else {
            BOOST_LOG(info) << "[SUDOVDA] Display settings updated successfully.";
          }

          return status;
        }
      }

      BOOST_LOG(warning) << "[SUDOVDA] Mode [" << width << 'x' << height << 'x' << refresh_rate
                         << "] not found for display: " << platf::to_utf8(deviceName);
      return ERROR_INVALID_PARAMETER;
    }
  }

  BOOST_LOG(warning) << "[SUDOVDA] Display not found: " << platf::to_utf8(deviceName);
  return ERROR_DEVICE_NOT_CONNECTED;
}
}  // namespace

LONG testDisplaySettings(const wchar_t *deviceName, int width, int height, int refresh_rate) {
  return changeBaselineDisplaySettings(
    deviceName,
    width,
    height,
    baselineRefreshRates(refresh_rate),
    CDS_TEST
  );
}

LONG changeDisplaySettings(const wchar_t *deviceName, int width, int height, int refresh_rate, bool persist_settings) {
  // Capture attempts open across this call restart because the host changed the mode.
  platf::note_display_change();
  const auto refreshRates = baselineRefreshRates(refresh_rate);
  BOOST_LOG(info) << "[SUDOVDA] Applying baseline display mode [" << width << 'x' << height << 'x'
                   << refreshRates.preferred << "] for " << platf::to_utf8(deviceName) << '.';
  const auto baselineStatus = changeBaselineDisplaySettings(
    deviceName,
    width,
    height,
    refreshRates,
    persist_settings ? CDS_UPDATEREGISTRY : 0
  );
  if (baselineStatus == DISP_CHANGE_SUCCESSFUL) {
    BOOST_LOG(info) << "[SUDOVDA] Baseline display mode applied successfully.";
  } else {
    BOOST_LOG(error) << "[SUDOVDA] Failed to apply baseline display mode.";
  }

  // Apply the exact fractional refresh rate through DisplayConfig.
  const auto status = applyDisplaySettings(deviceName, width, height, refresh_rate, persist_settings);
  platf::note_display_change();
  return status;
}

std::optional<session_mode_record_t> sessionModeRecord(const DEVMODEW &active) {
  // DEVMODE reserves frequencies 0 and 1 for the hardware default; never save that as a session.
  if (active.dmPelsWidth == 0 || active.dmPelsHeight == 0 || active.dmDisplayFrequency <= 1) {
    return std::nullopt;
  }
  return session_mode_record_t {
    active.dmPelsWidth,
    active.dmPelsHeight,
    active.dmDisplayFrequency,
    DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY,
    // The host may run under another account than the application, so save for all users.
    CDS_UPDATEREGISTRY | CDS_GLOBAL | CDS_NORESET,
  };
}

LONG saveSessionModeRecord(const wchar_t *deviceName, const session_mode_record_t &record) {
  DEVMODEW devMode {};
  devMode.dmSize = sizeof(devMode);
  devMode.dmFields = record.fields;
  devMode.dmPelsWidth = record.width;
  devMode.dmPelsHeight = record.height;
  devMode.dmDisplayFrequency = record.refresh_hz;
  return ChangeDisplaySettingsExW(deviceName, &devMode, nullptr, record.flags, nullptr);
}

std::optional<bool> queryDisplayHDR(const LUID &adapterLuid, uint32_t targetId) {
  // Query the display configuration state directly. A virtual HDR desktop is represented as
  // linear scRGB to desktop applications, so its DXGI output color space is not required to be
  // the physical-output PQ/Rec.2020 space and is not a reliable HDR-enabled test.
  const auto state = platf::display_config::query_advanced_color(adapterLuid, targetId);
  if (!state) {
    return std::nullopt;
  }
  return state->active_mode == DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
}

bool setDisplayHDR(const LUID& adapterId, const uint32_t& targetId, bool enableAdvancedColor) {
  return platf::display_config::set_hdr_state(adapterId, targetId, enableAdvancedColor);
}

std::optional<bool> queryDisplayHDRByName(const wchar_t* displayName) {
  // One exact active target: a cloned or ambiguous name has no single HDR state to report.
  const auto target = platf::display_config::resolve_display_target({}, displayName);
  if (!target) {
    BOOST_LOG(warning) << "The named display is not one single active target; its HDR state is unknown.";
    return std::nullopt;
  }
  return queryDisplayHDR(target->adapter_id, target->target_id);
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
        BOOST_LOG(error) << "[SUDOVDA] Open device failed!";
        return DRIVER_STATUS::FAILED;
      }
      retryInterval *= 2;
      Sleep(retryInterval);
      continue;
    }
    break;
  }

  if (!CheckProtocolCompatible(sudovdaDriverHandle)) {
    BOOST_LOG(error) << "[SUDOVDA] SUDOVDA protocol not compatible with driver!";
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
    BOOST_LOG(error) << "[SUDOVDA] Failed to open dedicated watchdog heartbeat handle.";
    return false;
  }

  VIRTUAL_DISPLAY_GET_WATCHDOG_OUT watchdogOut {};
  if (!GetWatchdogTimeout(watchdogHandle, watchdogOut)) {
    BOOST_LOG(error) << "[SUDOVDA] Watchdog fetch failed!";
    CloseHandle(watchdogHandle);
    return false;
  }
  BOOST_LOG(info) << "[SUDOVDA] Watchdog: Timeout " << watchdogOut.Timeout << ", Countdown " << watchdogOut.Countdown;
  if (!watchdogOut.Timeout) {
    CloseHandle(watchdogHandle);
    return true;
  }
  if (watchdogOut.Timeout < 10) {
    BOOST_LOG(warning) << "[SUDOVDA] Watchdog timeout is only " << watchdogOut.Timeout
                       << " seconds. Install or configure Sunshine 3D's recommended 30-second timeout to prevent "
                          "the retained virtual display from being removed during GPU initialization.";
  }

  const auto sleepInterval = watchdogPingInterval(watchdogOut.Timeout);
  std::lock_guard owner_lock(watchdogOwnerMutex);
  try {
    watchdogThread = std::jthread([watchdogHandle, sleepInterval, failCb = std::move(failCb)](std::stop_token stop_token) {
      // This heartbeat is the lifetime lease for every SudoVDA monitor. Favor it over ordinary
      // application work so a busy encoder startup cannot let the driver remove an active desktop.
      if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        BOOST_LOG(warning) << "[SUDOVDA] Failed to raise watchdog heartbeat thread priority: " << GetLastError();
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
    BOOST_LOG(error) << "[SUDOVDA] Failed to start watchdog heartbeat thread.";
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
    std::vector<DISPLAYCONFIG_MODE_INFO> &modes
  ) {
    return static_cast<bool>(platf::display_config::query_display_config(
      flags,
      paths,
      modes,
      {
        8,
        platf::display_config::retry_delay_e::exponential_milliseconds,
      }
    ));
  }
}  // namespace

bool queryActiveDisplayConfig(
  std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
  std::vector<DISPLAYCONFIG_MODE_INFO> &modes
) {
  return queryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, paths, modes);
}

namespace {
  struct display_color_contract_t {
    bool hdr_user_enabled = false;
    bool wcg_user_enabled = false;
    bool advanced_color_active = false;
    DISPLAYCONFIG_ADVANCED_COLOR_MODE active_mode {DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR};
  };

  enum class color_reconcile_action_internal_e {
    settled,
    wait_for_active_mode,
    set_hdr_user_state,
    set_wcg_user_state,
  };

  struct color_setter_attempts_t {
    bool hdr = false;
    bool wcg = false;
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
    UINT32 target_id,
    const platf::display_config::device_info_api_t &api
  ) {
    const auto state = platf::display_config::query_advanced_color(adapter_id, target_id, api);
    if (!state) {
      return std::nullopt;
    }
    return display_color_contract_t {
      state->hdr_user_enabled,
      state->wcg_user_enabled,
      state->advanced_color_active,
      state->active_mode,
    };
  }

  bool reconcileDisplayColorContract(
    const LUID &adapter_id,
    UINT32 target_id,
    const display_color_contract_t &expected,
    color_setter_attempts_t &attempts,
    active_mode_settle_state_t &active_mode_settle,
    const platf::display_config::device_info_api_t &api
  ) {
    const auto current = queryDisplayColorContract(adapter_id, target_id, api);
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
      case color_reconcile_action_internal_e::set_hdr_user_state:
        active_mode_settle = {};
        if (colorSetterRetryAvailable(action, attempts, now)) {
          if (platf::display_config::set_hdr_state(adapter_id, target_id, expected.hdr_user_enabled, api)) {
            consumeColorSetterAttempt(action, attempts);
          } else {
            deferRejectedColorSetter(action, attempts, now);
          }
        }
        break;
      case color_reconcile_action_internal_e::set_wcg_user_state:
        active_mode_settle = {};
        if (colorSetterRetryAvailable(action, attempts, now)) {
          if (platf::display_config::set_wcg_state(adapter_id, target_id, expected.wcg_user_enabled, api)) {
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
          return platf::display_config::equal_device_paths(expected.device_path, current.device_path);
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
    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    std::vector<DISPLAYCONFIG_MODE_INFO> modes;
    std::vector<survivor_color_state_t> survivor_color_states;
  };

  std::string describeDetachSurvivorModes(const detach_snapshot_t &snapshot) {
    std::ostringstream description;
    for (std::size_t index = 0; index < snapshot.paths.size(); ++index) {
      const auto &path = snapshot.paths[index];
      if ((path.flags & DISPLAYCONFIG_PATH_ACTIVE) == 0) {
        continue;
      }
      const bool virtual_indices =
        (path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE) != 0;
      const UINT32 mode_index = virtual_indices ?
                                  path.sourceInfo.sourceModeInfoIdx :
                                  path.sourceInfo.modeInfoIdx;
      description << " [path=" << index
                  << " source=" << path.sourceInfo.adapterId.HighPart << ':'
                  << path.sourceInfo.adapterId.LowPart << '/' << path.sourceInfo.id
                  << " target=" << path.targetInfo.adapterId.HighPart << ':'
                  << path.targetInfo.adapterId.LowPart << '/' << path.targetInfo.id
                  << " mode=" << mode_index
                  << " virtual_indices=" << virtual_indices;
      if (mode_index < snapshot.modes.size()) {
        const auto &mode = snapshot.modes[mode_index];
        description << " mode_identity=" << mode.adapterId.HighPart << ':'
                    << mode.adapterId.LowPart << '/' << mode.id
                    << " type=" << mode.infoType;
        if (mode.infoType == DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          description << " rect=" << mode.sourceMode.position.x << ','
                      << mode.sourceMode.position.y << '+' << mode.sourceMode.width
                      << 'x' << mode.sourceMode.height;
        }
      }
      description << ']';
    }
    return description.str();
  }

  // One active-topology observation: a single CCD query and the monitor path of each target.
  struct detach_topology_t {
    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    std::vector<DISPLAYCONFIG_MODE_INFO> modes;
    std::vector<std::wstring> device_paths;  // In the same order as paths.
  };

  std::optional<detach_topology_t> queryDetachTopology(const desktop_detach_api_t &api) {
    detach_topology_t topology;
    constexpr UINT32 query_flags =
      QDC_ONLY_ACTIVE_PATHS | QDC_VIRTUAL_MODE_AWARE | QDC_VIRTUAL_REFRESH_RATE_AWARE;
    const auto query = platf::display_config::query_display_config(query_flags, topology.paths, topology.modes, api.ccd);
    if (!query) {
      BOOST_LOG(warning) << "Virtual-display detach snapshot: CCD query failed (flags "
                         << query_flags << ", status " << query.status << ").";
      return std::nullopt;
    }
    topology.device_paths.reserve(topology.paths.size());
    for (const auto &path : topology.paths) {
      DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
      target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
      target_name.header.size = sizeof(target_name);
      target_name.header.adapterId = path.targetInfo.adapterId;
      target_name.header.id = path.targetInfo.id;
      const auto target_name_status = api.device_info.get(&target_name.header);
      if (target_name_status != ERROR_SUCCESS || target_name.monitorDevicePath[0] == L'\0') {
        // A failed target query could be the retiring display after a topology renumber. Treat the
        // whole snapshot as indeterminate instead of detaching a merely similar monitor.
        BOOST_LOG(warning) << "Virtual-display detach snapshot: target identity lookup failed (adapter "
                           << path.targetInfo.adapterId.HighPart << ':'
                           << path.targetInfo.adapterId.LowPart << ", target "
                           << path.targetInfo.id << ", status " << target_name_status
                           << ", empty device path "
                           << (target_name.monitorDevicePath[0] == L'\0') << ").";
        return std::nullopt;
      }
      topology.device_paths.emplace_back(target_name.monitorDevicePath);
    }
    return topology;
  }

  detach_plan_e detachPlanState(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
    std::wstring_view learnedDevicePath,
    const detach_topology_t &topology
  ) {
    // The plan marks the retiring path inactive; this question needs only its verdict.
    auto paths = topology.paths;
    return prepareVirtualDisplayDetachPaths(identity, learnedDevicePath, topology.device_paths, paths).state;
  }

  std::optional<detach_snapshot_t> buildVirtualDisplayDetachSnapshot(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
    std::wstring_view learnedDevicePath,
    detach_topology_t topology,
    const desktop_detach_api_t &api
  ) {
    detach_snapshot_t snapshot;
    snapshot.paths = std::move(topology.paths);
    snapshot.modes = std::move(topology.modes);
    const auto &candidate_device_paths = topology.device_paths;

    const auto plan = prepareVirtualDisplayDetachPaths(
      identity,
      learnedDevicePath,
      candidate_device_paths,
      snapshot.paths
    );
    snapshot.state = plan.state;
    if (snapshot.state == detach_plan_e::ready &&
        !rebaseSurvivingSourceModes(snapshot.paths, snapshot.modes)) {
      BOOST_LOG(warning) << "Virtual-display detach snapshot: surviving source geometry is invalid or disconnected."
                         << describeDetachSurvivorModes(snapshot);
      return std::nullopt;
    }
    if (snapshot.state == detach_plan_e::ready) {
      for (std::size_t index = 0; index < snapshot.paths.size(); ++index) {
        const auto &path = snapshot.paths[index];
        if ((path.flags & DISPLAYCONFIG_PATH_ACTIVE) == 0) {
          continue;
        }
        const auto color_contract =
          queryDisplayColorContract(path.targetInfo.adapterId, path.targetInfo.id, api.device_info);
        if (!color_contract) {
          // SetDisplayConfig can change Advanced Color state. Do not mutate a desktop whose
          // surviving target color contract cannot first be captured.
          BOOST_LOG(warning) << "Virtual-display detach snapshot: could not capture the surviving target's Advanced Color state (adapter "
                             << path.targetInfo.adapterId.HighPart << ':'
                             << path.targetInfo.adapterId.LowPart << ", target "
                             << path.targetInfo.id << ").";
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

  // Whether SetDisplayConfig would be given the same configuration for both snapshots.
  bool sameDetachConfiguration(const detach_snapshot_t &left, const detach_snapshot_t &right) {
    return left.state == right.state &&
           left.paths.size() == right.paths.size() && left.modes.size() == right.modes.size() &&
           std::memcmp(left.paths.data(), right.paths.data(), left.paths.size() * sizeof(DISPLAYCONFIG_PATH_INFO)) == 0 &&
           std::memcmp(left.modes.data(), right.modes.data(), left.modes.size() * sizeof(DISPLAYCONFIG_MODE_INFO)) == 0;
  }

  bool restoreSurvivorColorStatesOnce(
    const detach_topology_t &topology,
    std::vector<survivor_color_state_t> &expected_states,
    const desktop_detach_api_t &api
  ) {
    std::vector<bool> matched(expected_states.size(), false);

    for (std::size_t path_index = 0; path_index < topology.paths.size(); ++path_index) {
      const auto &path = topology.paths[path_index];
      for (std::size_t index = 0; index < expected_states.size(); ++index) {
        if (!platf::display_config::same_device_path(expected_states[index].device_path, topology.device_paths[path_index])) {
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
              expected_states[index].active_mode_settle,
              api.device_info
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
    bool context_has_captured_color_contract
  ) {
    return !has_context || !context_has_captured_color_contract;
  }
}

desktop_detach_result_t deactivateVirtualDisplay(
  const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
  std::wstring_view devicePath,
  std::chrono::milliseconds timeout,
  std::shared_ptr<desktop_detach_context_t> &context,
  const desktop_detach_api_t &api
) {
  const auto deadline =
    std::chrono::steady_clock::now() + std::max(timeout, std::chrono::milliseconds::zero());
  // The detach snapshot is queried virtual-mode and virtual-refresh aware.
  constexpr UINT32 awareness_flags = SDC_VIRTUAL_MODE_AWARE | SDC_VIRTUAL_REFRESH_RATE_AWARE;
  constexpr UINT32 validate_flags = SDC_VALIDATE | SDC_USE_SUPPLIED_DISPLAY_CONFIG | awareness_flags;
  constexpr UINT32 apply_flags = SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG | awareness_flags;
  const bool continuing_transition = static_cast<bool>(context);
  const auto build_snapshot = [&]() -> std::optional<detach_snapshot_t> {
    auto topology = queryDetachTopology(api);
    return topology ? buildVirtualDisplayDetachSnapshot(identity, devicePath, std::move(*topology), api) : std::nullopt;
  };
  const auto set_display_config = [&](detach_snapshot_t &supplied, UINT32 flags) {
    return api.set_display_config(
      static_cast<UINT32>(supplied.paths.size()),
      supplied.paths.data(),
      static_cast<UINT32>(supplied.modes.size()),
      supplied.modes.data(),
      flags
    );
  };
  auto snapshot = build_snapshot();
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
    LONG validation_status = set_display_config(*snapshot, validate_flags);
    if (validation_status != ERROR_SUCCESS) {
      return {desktop_detach_state_e::validation_failed, validation_status};
    }

    // Validation does not reserve the desktop topology. Query again just before applying: a
    // monitor/HDR change that completed during the validation is kept, and a changed
    // configuration is validated again. One that lands after this query or during that second
    // validation can still be overwritten.
    auto latest = build_snapshot();
    if (!latest) {
      return {desktop_detach_state_e::topology_query_failed, ERROR_GEN_FAILURE};
    }
    if (latest->state != detach_plan_e::ready && latest->state != detach_plan_e::already_inactive) {
      if (latest->state == detach_plan_e::skipped_only_active) {
        context.reset();
      }
      return detachPlanFailure(latest->state);
    }
    const bool validated = sameDetachConfiguration(*latest, *snapshot);
    snapshot = std::move(latest);

    if (snapshot->state == detach_plan_e::ready) {
      if (!validated) {
        validation_status = set_display_config(*snapshot, validate_flags);
        if (validation_status != ERROR_SUCCESS) {
          return {desktop_detach_state_e::validation_failed, validation_status};
        }
      }

      const bool capture_color_contract = detachRetryMustCaptureColorContract(
        static_cast<bool>(context),
        context && context->has_captured_color_contract
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
      const LONG apply_status = set_display_config(*snapshot, apply_flags);
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
    // One topology observation answers both questions of this tick.
    const auto observation = queryDetachTopology(api);
    latest_path_inactive =
      observation && detachPlanState(identity, devicePath, *observation) == detach_plan_e::already_inactive;
    color_states_match =
      context->expected_color_states.empty() ||
      (observation && restoreSurvivorColorStatesOnce(*observation, context->expected_color_states, api));
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
    const auto delay = std::min(
      std::chrono::milliseconds(50),
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
    );
    api.sleep(static_cast<DWORD>(delay.count()));
  }
}

namespace {
display_identity_query_t queryDisplayIdentity(const LUID &adapterLuid, uint32_t targetId) {
  std::vector<DISPLAYCONFIG_PATH_INFO> paths;
  std::vector<DISPLAYCONFIG_MODE_INFO> modes;
  if (!queryActiveDisplayConfig(paths, modes)) {
    return {};
  }

  for (const auto &path : paths) {
    if (path.targetInfo.id != targetId || !platf::display_config::same_luid(path.targetInfo.adapterId, adapterLuid)) {
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
}  // namespace

display_identity_query_t queryVirtualDisplayIdentity(
  const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
  std::wstring_view devicePath
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
  std::wstring_view devicePath
) {
  const auto active = queryVirtualDisplayIdentity(identity, devicePath);
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

bool isDriverRemovalSafeAfterDesktopDetach(display_identity_state_e activePathState) {
  // Every detach requires an authoritative exact active-topology absence observation
  // immediately before driver removal. Callers skip this check only for a sole active virtual
  // output, which has no surviving desktop or multi-monitor shell state to preserve.
  return activePathState == display_identity_state_e::absent;
}

#ifdef SUNSHINE_TESTS
bool isSudoVirtualDisplayPathForTest(std::wstring_view devicePath) {
  return isSudoVirtualDisplayPath(devicePath);
}

display_identity_query_t waitForDisplayIdentityForTest(
  const LUID &adapterLuid,
  uint32_t targetId,
  const std::function<display_identity_query_t(const LUID &, uint32_t)> &query,
  const std::function<std::chrono::steady_clock::time_point()> &now,
  const std::function<void(std::chrono::steady_clock::time_point)> &waitUntil
) {
  return waitForDisplayIdentity(adapterLuid, targetId, query, now, waitUntil);
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
    currentHdrUserEnabled,
    currentWcgUserEnabled,
    currentAdvancedColorActive,
    currentActiveMode,
  };
  const display_color_contract_t expected {
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
    currentHdrUserEnabled,
    currentWcgUserEnabled,
  };
  const display_color_contract_t expected {
    expectedHdrUserEnabled,
    expectedWcgUserEnabled,
  };
  const color_setter_attempts_t attempts {
    hdrSetterAccepted,
    wcgSetterAccepted,
  };
  switch (colorReconcileAction(current, expected, &attempts)) {
    case color_reconcile_action_internal_e::settled:
      return color_reconcile_action_e::settled;
    case color_reconcile_action_internal_e::wait_for_active_mode:
      return color_reconcile_action_e::wait_for_active_mode;
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
  bool contextHasCapturedColorContract
) {
  return detachRetryMustCaptureColorContract(
    hasContext,
    contextHasCapturedColorContract
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
  original.contract.hdr_user_enabled = true;

  auto changed_original = original;
  changed_original.contract.hdr_user_enabled = false;
  survivor_color_state_t newly_connected;
  newly_connected.device_path = LR"(\\?\DISPLAY#PHYSICAL#new)";

  std::vector<survivor_color_state_t> expected {original};
  mergeSurvivorColorStates(expected, {changed_original, newly_connected});
  return expected.size() == 2 &&
         expected[0].contract.hdr_user_enabled &&
         platf::display_config::equal_device_paths(expected[1].device_path, newly_connected.device_path);
}

bool wcgSetterSelectedDuringHdrBackoffForTest() {
  const display_color_contract_t current {
    false,
    false,
  };
  const display_color_contract_t expected {
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
      if (DisplayConfigGetDeviceInfo(&sourceName.header) == ERROR_SUCCESS && platf::display_config::same_device_path(sourceName.viewGdiDeviceName, primaryDisplay)) {
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
        BOOST_LOG(error) << "[SUDOVDA] Failed to select render adapter for virtual display.";
        return {};
      }
      if (!AddVirtualDisplay(sudovdaDriverHandle, width, height, fps, guid, s_client_name, s_client_uid, output)) {
        BOOST_LOG(error) << "[SUDOVDA] Failed to add virtual display.";
        return {};
      }
    }
    // Driver creation and Windows display-name publication are separate operations. Record Add's
    // success immediately so callers can always retire the exact output even if name lookup times
    // out during topology churn.
    result.identity = output;

    auto published = waitForDisplayIdentity(
      output.AdapterLuid,
      output.TargetId,
      queryDisplayIdentity,
      std::chrono::steady_clock::now,
      [](std::chrono::steady_clock::time_point deadline) {
        std::this_thread::sleep_until(deadline);
      }
    );
    if (published.state != display_identity_state_e::present) {
      BOOST_LOG(error) << "[SUDOVDA] Cannot get name for newly added virtual display!";
      return result;
    }

    BOOST_LOG(info) << "[SUDOVDA] Virtual display added successfully: " << platf::to_utf8(published.display_name);
    BOOST_LOG(info) << "[SUDOVDA] Configuration: W: " << width << ", H: " << height << ", FPS: " << fps;

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
  const GUID &guid,
  const std::optional<LUID> &adapterLuid,
  const std::wstring &adapterName
) {
  std::optional<LUID> renderAdapter = adapterLuid;
  if (!renderAdapter && !adapterName.empty()) {
    renderAdapter = adapterLuidByName(adapterName);
    if (!renderAdapter) {
      BOOST_LOG(error) << "[SUDOVDA] Cannot find requested render adapter.";
      return {};
    }
  }
  return createVirtualDisplayImpl(
    s_client_uid,
    s_client_name,
    width,
    height,
    fps,
    guid,
    renderAdapter ? renderAdapter : primaryDisplayAdapterLuid()
  );
}

bool removeVirtualDisplay(const GUID &guid) {
  std::lock_guard lock(virtualDisplayMutationMutex);
  if (sudovdaDriverHandle == INVALID_HANDLE_VALUE) {
    return false;
  }

  if (RemoveVirtualDisplay(sudovdaDriverHandle, guid)) {
    BOOST_LOG(info) << "[SUDOVDA] Virtual display removed successfully.";
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
	// The size of the structure that is passed: this file is built without UNICODE, where
	// DISPLAY_DEVICE names the smaller narrow structure.
	DISPLAY_DEVICEW displayDevice {};
	displayDevice.cb = sizeof(displayDevice);

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
