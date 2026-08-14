/**
 * @file src/platform/windows/display_config.h
 * @brief Shared Windows display-topology and Advanced Color API adapters.
 */
#pragma once

#include <optional>
#include <vector>

#include <windows.h>

namespace platf::display_config {

  struct query_api_t {
    decltype(&::GetDisplayConfigBufferSizes) get_buffer_sizes =
      &::GetDisplayConfigBufferSizes;
    decltype(&::QueryDisplayConfig) query = &::QueryDisplayConfig;
    decltype(&::Sleep) sleep = &::Sleep;
  };

  struct query_result_t {
    LONG status = ERROR_INVALID_PARAMETER;

    [[nodiscard]] explicit operator bool() const noexcept {
      return status == ERROR_SUCCESS;
    }
  };

  enum class retry_delay_e {
    none,
    exponential_milliseconds,
  };

  struct query_policy_t {
    unsigned int max_attempts = 8;
    retry_delay_e retry_delay = retry_delay_e::exponential_milliseconds;
  };

  /**
   * Query one CCD topology snapshot, retrying the documented size/query race.
   * The injected overload exists so the retry state machine can be tested without mutating CCD.
   */
  query_result_t query_display_config(
    UINT32 flags,
    std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
    std::vector<DISPLAYCONFIG_MODE_INFO> &modes,
    const query_api_t &api,
    query_policy_t policy = {}
  );

  query_result_t query_display_config(
    UINT32 flags,
    std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
    std::vector<DISPLAYCONFIG_MODE_INFO> &modes,
    query_policy_t policy = {}
  );

  enum class advanced_color_api_e {
    modern,
    legacy,
  };

  enum class legacy_fallback_e {
    unsupported_modern_api,
    any_modern_failure,
  };

  struct advanced_color_state_t {
    advanced_color_api_e api = advanced_color_api_e::legacy;
    bool advanced_color_enabled = false;
    bool hdr_supported = false;
    bool hdr_user_enabled = false;
    bool wcg_user_enabled = false;
    bool advanced_color_active = false;
    bool limited_by_policy = false;
    UINT32 bits_per_color_channel = 0;
    DISPLAYCONFIG_ADVANCED_COLOR_MODE active_mode =
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR;
  };

  struct device_info_api_t {
    decltype(&::DisplayConfigGetDeviceInfo) get = &::DisplayConfigGetDeviceInfo;
    decltype(&::DisplayConfigSetDeviceInfo) set = &::DisplayConfigSetDeviceInfo;
  };

  std::optional<advanced_color_state_t> query_advanced_color(
    const LUID &adapter_id,
    UINT32 target_id,
    legacy_fallback_e fallback,
    const device_info_api_t &api
  );

  std::optional<advanced_color_state_t> query_advanced_color(
    const LUID &adapter_id,
    UINT32 target_id,
    legacy_fallback_e fallback = legacy_fallback_e::unsupported_modern_api
  );

  bool set_hdr_state(
    const LUID &adapter_id,
    UINT32 target_id,
    bool enabled,
    const device_info_api_t &api
  );
  bool set_hdr_state(const LUID &adapter_id, UINT32 target_id, bool enabled);

  bool set_wcg_state(
    const LUID &adapter_id,
    UINT32 target_id,
    bool enabled,
    const device_info_api_t &api
  );
  bool set_wcg_state(const LUID &adapter_id, UINT32 target_id, bool enabled);

  bool set_legacy_advanced_color_state(
    const LUID &adapter_id,
    UINT32 target_id,
    bool enabled,
    const device_info_api_t &api
  );
  bool set_hdr_state_with_legacy_fallback(
    const LUID &adapter_id,
    UINT32 target_id,
    bool enabled,
    const device_info_api_t &api
  );
  bool set_hdr_state_with_legacy_fallback(
    const LUID &adapter_id,
    UINT32 target_id,
    bool enabled
  );

}  // namespace platf::display_config
