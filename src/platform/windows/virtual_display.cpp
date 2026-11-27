#include "virtual_display.h"

#include "display_config.h"
#include "misc.h"
#include "src/logging.h"
#include "src/platform/common.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace VDISPLAY {
  LONG getDeviceSettings(const wchar_t* deviceName, DEVMODEW& devMode) {
	devMode.dmSize = sizeof(DEVMODEW);
	return EnumDisplaySettingsW(deviceName, ENUM_CURRENT_SETTINGS, &devMode);
}

namespace {
  // Local shutter sessions on a desktop refresh rate that are running now.
  std::atomic<int> desktopRefreshRateKeepers {0};

  bool keepingDesktopRefreshRates() {
    return desktopRefreshRateKeepers.load(std::memory_order_acquire) > 0;
  }

  bool pathUsesVirtualRefreshRate(const DISPLAYCONFIG_PATH_INFO &path, const std::vector<DISPLAYCONFIG_MODE_INFO> &modes) {
    if (!(path.flags & DISPLAYCONFIG_PATH_ACTIVE)) {
      return false;
    }
    if (path.flags & DISPLAYCONFIG_PATH_BOOST_REFRESH_RATE) {
      return true;
    }
    // An indirect display's signal description is the driver's own invention.
    const auto technology = static_cast<int>(path.targetInfo.outputTechnology);
    if (technology == static_cast<int>(DISPLAYCONFIG_OUTPUT_TECHNOLOGY_INDIRECT_WIRED) ||
        technology == static_cast<int>(DISPLAYCONFIG_OUTPUT_TECHNOLOGY_INDIRECT_VIRTUAL)) {
      return false;
    }
    const auto index = path.targetInfo.modeInfoIdx;
    if (index >= modes.size() || modes[index].infoType != DISPLAYCONFIG_MODE_INFO_TYPE_TARGET) {
      return false;
    }
    const auto &signal = modes[index].targetMode.targetVideoSignalInfo;
    const auto &desktop = path.targetInfo.refreshRate;
    if (signal.scanLineOrdering != DISPLAYCONFIG_SCANLINE_ORDERING_PROGRESSIVE || !desktop.Numerator || !desktop.Denominator ||
        !signal.pixelRate || !signal.totalSize.cx || !signal.totalSize.cy) {
      return false;
    }
    // The rate the signal's own clock and size give, against the rate the desktop runs at.
    const double signal_hz = static_cast<double>(signal.pixelRate) / (static_cast<double>(signal.totalSize.cx) * signal.totalSize.cy);
    const double desktop_hz = static_cast<double>(desktop.Numerator) / desktop.Denominator;
    return std::abs(signal_hz - desktop_hz) > signal_hz * 0.005;
  }

  bool sameRate(const DISPLAYCONFIG_RATIONAL &left, const DISPLAYCONFIG_RATIONAL &right) {
    return static_cast<std::uint64_t>(left.Numerator) * right.Denominator == static_cast<std::uint64_t>(right.Numerator) * left.Denominator;
  }

  // An active physical display on a desktop refresh rate, and that rate.
  struct desktop_refresh_rate_t {
    LUID adapterId;
    uint32_t targetId;
    DISPLAYCONFIG_RATIONAL rate;
  };

  std::vector<desktop_refresh_rate_t> desktopRefreshRatesInUse() {
    std::vector<desktop_refresh_rate_t> rates;
    platf::display_config::topology_t topology;
    if (!platf::display_config::query_topology(QDC_ONLY_ACTIVE_PATHS, topology)) {
      return rates;
    }
    for (const auto &path : topology.paths) {
      if (pathUsesVirtualRefreshRate(path, topology.modes)) {
        rates.push_back({path.targetInfo.adapterId, path.targetInfo.id, path.targetInfo.refreshRate});
      }
    }
    return rates;
  }

  // Asks Windows again for each of those rates that a display has since left. Displays that are
  // no longer active are left alone.
  bool restoreDesktopRefreshRates(const std::vector<desktop_refresh_rate_t> &rates) {
    if (rates.empty()) {
      return true;
    }
    platf::display_config::topology_t topology;
    if (!platf::display_config::query_topology(QDC_ONLY_ACTIVE_PATHS | QDC_VIRTUAL_REFRESH_RATE_AWARE, topology)) {
      return false;
    }
    bool changed = false;
    for (const auto &rate : rates) {
      const auto path = std::find_if(topology.paths.begin(), topology.paths.end(), [&](const DISPLAYCONFIG_PATH_INFO &candidate) {
        return platf::display_config::same_luid(candidate.targetInfo.adapterId, rate.adapterId) && candidate.targetInfo.id == rate.targetId;
      });
      if (path != topology.paths.end() && !sameRate(path->targetInfo.refreshRate, rate.rate)) {
        BOOST_LOG(info) << "A display left its selected desktop refresh rate "
                        << (static_cast<double>(rate.rate.Numerator) / rate.rate.Denominator) << " Hz for "
                        << (path->targetInfo.refreshRate.Denominator ? static_cast<double>(path->targetInfo.refreshRate.Numerator) / path->targetInfo.refreshRate.Denominator : 0.0)
                        << " Hz when another display's mode was set; asking Windows for it again.";
        path->targetInfo.refreshRate = rate.rate;
        changed = true;
      }
    }
    if (!changed) {
      return true;
    }
    platf::note_display_change();
    const LONG status = platf::display_config::set_display_config(
      topology,
      SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG | SDC_VIRTUAL_REFRESH_RATE_AWARE
    );
    platf::note_display_change();
    if (status != ERROR_SUCCESS) {
      BOOST_LOG(warning) << "Windows refused the selected desktop refresh rate (status " << status << ").";
    }
    return status == ERROR_SUCCESS;
  }

  // The active configuration for a caller that hands it back to SetDisplayConfig() with
  // SDC_USE_SUPPLIED_DISPLAY_CONFIG. While a local shutter session keeps desktop refresh rates
  // and a display is on one, it is read with virtual-refresh awareness and awarenessFlags names
  // the flag the apply needs with it; otherwise both are exactly as before.
  bool queryActiveTopologyForApply(platf::display_config::topology_t &topology, UINT32 &awarenessFlags) {
    awarenessFlags = 0;
    if (!platf::display_config::query_topology(QDC_ONLY_ACTIVE_PATHS, topology)) {
      return false;
    }
    if (!keepingDesktopRefreshRates() || !displayConfigUsesVirtualRefreshRate(topology.paths, topology.modes)) {
      return true;
    }
    platf::display_config::topology_t aware;
    if (platf::display_config::query_topology(QDC_ONLY_ACTIVE_PATHS | QDC_VIRTUAL_REFRESH_RATE_AWARE, aware)) {
      topology = std::move(aware);
      awarenessFlags = SDC_VIRTUAL_REFRESH_RATE_AWARE;
    }
    return true;
  }
}  // namespace

desktop_refresh_rate_keeper_t::desktop_refresh_rate_keeper_t() {
  desktopRefreshRateKeepers.fetch_add(1, std::memory_order_acq_rel);
}

desktop_refresh_rate_keeper_t::~desktop_refresh_rate_keeper_t() {
  desktopRefreshRateKeepers.fetch_sub(1, std::memory_order_acq_rel);
}

bool displayConfigUsesVirtualRefreshRate(
  const std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
  const std::vector<DISPLAYCONFIG_MODE_INFO> &modes
) {
  return std::any_of(paths.begin(), paths.end(), [&](const DISPLAYCONFIG_PATH_INFO &path) {
    return pathUsesVirtualRefreshRate(path, modes);
  });
}

bool targetUsesDesktopRefreshRate(const LUID &adapterId, uint32_t targetId) {
  const auto rates = desktopRefreshRatesInUse();
  return std::any_of(rates.begin(), rates.end(), [&](const desktop_refresh_rate_t &rate) {
    return platf::display_config::same_luid(rate.adapterId, adapterId) && rate.targetId == targetId;
  });
}

LONG setTargetDesktopRefreshRate(const LUID &adapterId, uint32_t targetId, int refreshMillihz) {
  if (refreshMillihz <= 0) {
    return ERROR_INVALID_PARAMETER;
  }
  platf::display_config::topology_t topology;
  if (const auto queried = platf::display_config::query_topology(QDC_ONLY_ACTIVE_PATHS | QDC_VIRTUAL_REFRESH_RATE_AWARE, topology); !queried) {
    return queried.status == ERROR_SUCCESS ? ERROR_GEN_FAILURE : queried.status;
  }
  const auto path = std::find_if(topology.paths.begin(), topology.paths.end(), [&](const DISPLAYCONFIG_PATH_INFO &candidate) {
    return platf::display_config::same_luid(candidate.targetInfo.adapterId, adapterId) && candidate.targetInfo.id == targetId;
  });
  if (path == topology.paths.end()) {
    return ERROR_NOT_FOUND;
  }
  // Capture attempts open across this call restart because the host changed the mode.
  platf::note_display_change();
  path->targetInfo.refreshRate = {static_cast<UINT32>(refreshMillihz), 1000};
  const LONG status = platf::display_config::set_display_config(
    topology,
    SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG | SDC_VIRTUAL_REFRESH_RATE_AWARE
  );
  platf::note_display_change();
  return status;
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
  platf::display_config::topology_t topology;
  // The whole configuration goes back to Windows, the other displays included. While a local
  // shutter session keeps desktop refresh rates, a monitor on one keeps it only if the
  // configuration is read and written with awareness of it.
  UINT32 awarenessFlags = 0;
  if (!queryActiveTopologyForApply(topology, awarenessFlags)) {
    BOOST_LOG(error) << "[SUDOVDA] Failed to query display configuration.";
    return ERROR_INVALID_PARAMETER;
  }
  auto &pathArray = topology.paths;
  auto &modeArray = topology.modes;
  const UINT32 pathCount = (UINT32) pathArray.size();
  const UINT32 modeCount = (UINT32) modeArray.size();
  for (UINT32 i = 0; i < pathCount; i++) {
    auto *sourceInfo = &pathArray[i].sourceInfo;
    auto *targetInfo = &pathArray[i].targetInfo;
    const auto sourceName = platf::display_config::query_source_name(sourceInfo->adapterId, sourceInfo->id);
    if (!sourceName) {
      continue;
    }

    if (platf::display_config::same_device_path(*sourceName, deviceName)) {
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
          LONG status = platf::display_config::set_display_config(
            topology,
            SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG | awarenessFlags | (persist_settings ? SDC_SAVE_TO_DATABASE : 0)
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
  // Setting one display's mode makes Windows lay out the whole configuration again, and a
  // monitor on a desktop refresh rate then falls back to its signal's own rate. A local shutter
  // session on such a rate asks for it again.
  const auto keptDesktopRates = keepingDesktopRefreshRates() ? desktopRefreshRatesInUse() : std::vector<desktop_refresh_rate_t> {};
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
  restoreDesktopRefreshRates(keptDesktopRates);
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
