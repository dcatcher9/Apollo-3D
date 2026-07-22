/**
 * @file src/display_device.cpp
 * @brief Read-only Windows display enumeration and name mapping.
 */
#include "display_device.h"

#include "logging.h"

#include <display_device/windows/win_api_layer.h>
#include <display_device/windows/win_display_device.h>
#include <memory>
#include <mutex>

namespace display_device {
  namespace {
    std::mutex display_api_mutex;

    WinDisplayDevice &display_api() {
      static auto windows_api = std::make_shared<WinApiLayer>();
      static WinDisplayDevice api {windows_api};
      return api;
    }
  }  // namespace

  std::string map_output_name(const std::string &output_name) {
    if (output_name.empty()) {
      return {};
    }

    try {
      std::lock_guard lock {display_api_mutex};
      return display_api().getDisplayName(output_name);
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Failed to resolve display device '" << output_name << "': " << exception.what();
      return {};
    }
  }

  std::string map_display_name(const std::string &display_name) {
    if (display_name.empty()) {
      return {};
    }

    try {
      std::lock_guard lock {display_api_mutex};
      for (const auto &device : display_api().enumAvailableDevices()) {
        if (device.m_display_name == display_name) {
          return device.m_device_id;
        }
      }
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Failed to resolve display name '" << display_name << "': " << exception.what();
    }
    return {};
  }

  EnumeratedDeviceList enumerate_devices() {
    try {
      std::lock_guard lock {display_api_mutex};
      return display_api().enumAvailableDevices();
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Failed to enumerate Windows displays: " << exception.what();
      return {};
    }
  }
}  // namespace display_device
