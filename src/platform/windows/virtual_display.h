#pragma once

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
#ifdef SUNSHINE_TESTS
  bool isSudoVirtualDisplayPathForTest(std::wstring_view devicePath);
  uint32_t watchdogPingIntervalMsForTest(uint32_t timeoutSeconds);
  bool virtualDisplayIdentityMatchesForTest(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &expectedIdentity,
    std::wstring_view learnedDevicePath,
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &candidateIdentity,
    std::wstring_view candidateDevicePath
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
