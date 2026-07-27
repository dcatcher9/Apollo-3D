#pragma once

#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

#ifndef FILE_DEVICE_UNKNOWN
#define FILE_DEVICE_UNKNOWN 0x00000022
#endif

#include <ddk/d4iface.h>
#include <ddk/d4drvif.h>
#include <sudovda/sudovda.h>

namespace VDISPLAY {
	struct creation_result_t {
		std::wstring display_name;
		std::wstring device_path;
		std::wstring friendly_name;
		std::optional<SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT> identity;
		std::optional<LUID> render_adapter_luid;

		[[nodiscard]] bool added() const {
			return identity.has_value();
		}
	};

	enum class display_identity_state_e {
		indeterminate,
		absent,
		present,
	};

	struct display_identity_query_t {
		display_identity_state_e state {display_identity_state_e::indeterminate};
		std::wstring display_name;
		std::wstring device_path;
		std::wstring friendly_name;
	};

	enum class DRIVER_STATUS {
		UNKNOWN              = 1,
		OK                   = 0,
		FAILED               = -1,
		VERSION_INCOMPATIBLE = -2,
		WATCHDOG_FAILED      = -3
	};

  enum class desktop_detach_state_e {
    detached,
    already_inactive,
    skipped_only_active,
    ambiguous_identity,
    topology_query_failed,
    validation_failed,
    apply_failed,
    color_restore_timeout,
    settle_timeout,
  };

  struct desktop_detach_result_t {
    desktop_detach_state_e state {desktop_detach_state_e::topology_query_failed};
    LONG windows_status {ERROR_SUCCESS};
  };

	LONG getDeviceSettings(const wchar_t* deviceName, DEVMODEW& devMode);
	LONG testDisplaySettings(const wchar_t* deviceName, int width, int height, int refresh_rate);
	LONG changeDisplaySettings(const wchar_t* deviceName, int width, int height, int refresh_rate);
	std::optional<bool> queryDisplayHDRByName(const wchar_t* displayName);
	bool setDisplayHDRByName(const wchar_t* displayName, bool enableAdvancedColor);

	void closeVDisplayDevice();
	DRIVER_STATUS openVDisplayDevice();
	bool startPingThread(std::function<void()> failCb);
  bool queryActiveDisplayConfig(
    std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
    std::vector<DISPLAYCONFIG_MODE_INFO> &modes
  );
  display_identity_query_t queryDisplayIdentity(const LUID &adapterLuid, uint32_t targetId);
  display_identity_query_t queryVirtualDisplayIdentity(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
    std::wstring_view devicePath,
    std::wstring_view displayName
  );
  display_identity_state_e queryVirtualDisplayRetirementState(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
    std::wstring_view devicePath,
    std::wstring_view displayName
  );
  desktop_detach_result_t deactivateVirtualDisplay(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
    std::wstring_view devicePath,
    std::chrono::milliseconds timeout
  );
#ifdef SUNSHINE_TESTS
  enum class desktop_detach_plan_e {
    ready,
    already_inactive,
    skipped_only_active,
    ambiguous_identity,
  };

  struct retirement_path_candidate_t {
    SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT identity {};
    std::wstring device_path;
    bool target_available = false;
    bool device_info_available = true;
  };

  bool isSudoVirtualDisplayPathForTest(std::wstring_view devicePath);
  uint32_t watchdogPingIntervalMsForTest(uint32_t timeoutSeconds);
  bool virtualDisplayIdentityMatchesForTest(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &expectedIdentity,
    std::wstring_view learnedDevicePath,
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &candidateIdentity,
    std::wstring_view candidateDevicePath
  );
  bool rebaseVirtualDisplaySurvivorsForTest(
    const std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
    std::vector<DISPLAYCONFIG_MODE_INFO> &modes
  );
  desktop_detach_plan_e prepareVirtualDisplayDetachPathsForTest(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
    std::wstring_view learnedDevicePath,
    const std::vector<std::wstring> &candidateDevicePaths,
    std::vector<DISPLAYCONFIG_PATH_INFO> &paths
  );
  display_identity_state_e virtualDisplayRetirementStateForTest(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
    std::wstring_view learnedDevicePath,
    const std::vector<retirement_path_candidate_t> &candidates
  );
#endif
	creation_result_t createVirtualDisplay(
    const char *s_client_uid,
    const char *s_client_name,
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    const GUID &guid
  );
  creation_result_t createVirtualDisplayWithRenderAdapter(
    const char *s_client_uid,
    const char *s_client_name,
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    const GUID &guid,
    const std::optional<LUID> &adapterLuid
  );
  creation_result_t createVirtualDisplayOnAdapter(
    const char *s_client_uid,
    const char *s_client_name,
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    const GUID &guid,
    const LUID &adapterLuid
  );
  creation_result_t createVirtualDisplayOnAdapter(
    const char *s_client_uid,
    const char *s_client_name,
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    const GUID &guid,
    const std::wstring &adapterName
  );
  bool removeVirtualDisplay(const GUID &guid);

  std::vector<std::wstring> matchDisplay(std::wstring sMatch);
}
