#pragma once

#include <functional>
#include <optional>
#include <vector>
#include <windows.h>

#ifndef FILE_DEVICE_UNKNOWN
#define FILE_DEVICE_UNKNOWN 0x00000022
#endif

#include <ddk/d4iface.h>
#include <ddk/d4drvif.h>
#include <sudovda/sudovda.h>

namespace VDISPLAY {
	enum class DRIVER_STATUS {
		UNKNOWN              = 1,
		OK                   = 0,
		FAILED               = -1,
		VERSION_INCOMPATIBLE = -2,
		WATCHDOG_FAILED      = -3
	};

	LONG getDeviceSettings(const wchar_t* deviceName, DEVMODEW& devMode);
	LONG changeDisplaySettings(const wchar_t* deviceName, int width, int height, int refresh_rate);
	LONG changeDisplaySettings2(const wchar_t* deviceName, int width, int height, int refresh_rate, bool bApplyIsolated=false);	
	std::wstring getPrimaryDisplay();
	bool setPrimaryDisplay(const wchar_t* primaryDeviceName);
	bool getDisplayHDRByName(const wchar_t* displayName);
	bool setDisplayHDRByName(const wchar_t* displayName, bool enableAdvancedColor);

	void closeVDisplayDevice();
	DRIVER_STATUS openVDisplayDevice();
	bool startPingThread(std::function<void()> failCb);
  bool queryActiveDisplayConfig(
    std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
    std::vector<DISPLAYCONFIG_MODE_INFO> &modes
  );
  std::wstring getDisplayName(const LUID &adapterLuid, uint32_t targetId);
  std::wstring createVirtualDisplay(
    const char *s_client_uid,
    const char *s_client_name,
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    const GUID &guid,
    SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT *createdDisplay = nullptr
  );
  std::wstring createVirtualDisplayOnAdapter(
    const char *s_client_uid,
    const char *s_client_name,
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    const GUID &guid,
    const LUID &adapterLuid,
    SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT *createdDisplay = nullptr
  );
  std::wstring createVirtualDisplayOnAdapter(
    const char *s_client_uid,
    const char *s_client_name,
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    const GUID &guid,
    const std::wstring &adapterName,
    SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT *createdDisplay = nullptr
  );
  bool removeVirtualDisplay(const GUID &guid);

  std::vector<std::wstring> matchDisplay(std::wstring sMatch);
}
