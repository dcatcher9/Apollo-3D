/**
 * @file src/platform/windows/ar_glasses_devices.h
 * @brief The AR device registry: the saved decision for each monitor model, the physical outputs
 * Windows reports, and which approved output local presentation may use.
 */
#pragma once

#include "ar_glasses.h"
#include "ar_glasses_config_store.h"
#include "display_config.h"

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

namespace ar_glasses {
  namespace internal {
    using platf::display_config::equal_device_paths;
    using platf::display_config::same_device_path;
    using platf::display_config::same_luid;
    using platf::display_config::same_rect;
    using platf::display_config::valid_rect;

    inline constexpr int source_width = 1920;
    inline constexpr int source_height = 1080;

    bool is_apollo_ar_virtual_name(std::wstring_view value);

    struct hdr_state_t {
      bool known = false;
      bool supported = false;
      bool user_enabled = false;
      bool active = false;
      bool limited_by_policy = false;
      UINT32 bits_per_color = 0;
    };

    // Discovery observes every physical output; approval decides which need presentation contracts.
    struct discovered_target_t {
      std::string device_id;
      std::wstring device_path;
      std::string friendly_name;
      std::string gdi_name;
      LUID source_adapter_id {};
      UINT32 source_id = 0;
      LUID adapter_id {};
      UINT32 target_id = 0;
      RECT rect {};
      int refresh_millihz = 60000;
      presentation_mode_e mode = presentation_mode_e::unsupported;
      // Glasses only: whether their 3D comes from the game. A monitor carries it in its mode.
      bool game_source = false;
    };

    struct target_state_t: discovered_target_t {
      std::string desktop_topology;
      hdr_state_t hdr;
      bool is_primary = false;
      bool is_cloned = false;

      bool same_display_observation(const target_state_t &other) const {
        return device_id == other.device_id && equal_device_paths(device_path, other.device_path) &&
               gdi_name == other.gdi_name && desktop_topology == other.desktop_topology &&
               same_luid(adapter_id, other.adapter_id) &&
               rect.left == other.rect.left && rect.top == other.rect.top &&
               rect.right == other.rect.right && rect.bottom == other.rect.bottom &&
               refresh_millihz == other.refresh_millihz &&
               is_primary == other.is_primary && is_cloned == other.is_cloned &&
               hdr.known == other.hdr.known &&
               hdr.supported == other.hdr.supported &&
               hdr.active == other.hdr.active &&
               hdr.limited_by_policy == other.hdr.limited_by_policy;
      }

      bool operator==(const target_state_t &other) const {
        return mode == other.mode && game_source == other.game_source && same_display_observation(other);
      }
    };

    int target_source_width(const target_state_t &target);
    int target_source_height(const target_state_t &target);
    bool target_exclusive(const target_state_t &target);

    struct stored_device_t: device_info_t {
      bool notified = false;
    };

    /** Guarded by device_mutex. */
    extern std::vector<stored_device_t> known_devices;

    void load_devices(bool persist_cleanup);

    struct target_discovery_snapshot_t {
      std::vector<discovered_target_t> targets;
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::wstring primary_gdi_name;
      std::string primary_device_path;
    };

    std::optional<target_discovery_snapshot_t> discover_targets(
      std::vector<DISPLAYCONFIG_PATH_INFO> paths,
      const std::vector<DISPLAYCONFIG_MODE_INFO> &modes
    );
    std::optional<target_discovery_snapshot_t> discover_targets();
    std::optional<RECT> monitor_watchdog_rect(std::wstring_view device_path);
    target_state_t materialize_target(const target_discovery_snapshot_t &snapshot, std::size_t index);
    std::optional<target_state_t> find_target(
      std::wstring_view required_device_path = {},
      std::wstring_view preferred_device_path = {},
      bool *query_succeeded = nullptr,
      std::vector<target_state_t> *approved_targets = nullptr
    );
  }  // namespace internal
}  // namespace ar_glasses
