/**
 * @file src/display_device.h
 * @brief Read-only Windows display enumeration and name mapping.
 */
#pragma once

#include <display_device/types.h>
#include <string>

namespace display_device {
  /** Resolve a persistent Windows display-device ID to its active GDI display name. */
  [[nodiscard]] std::string map_output_name(const std::string &output_name);

  /** Resolve an active GDI display name to its persistent Windows display-device ID. */
  [[nodiscard]] std::string map_display_name(const std::string &display_name);

  /** Enumerate Windows displays for capture selection and the configuration UI. */
  [[nodiscard]] EnumeratedDeviceList enumerate_devices();
}  // namespace display_device
