/**
 * @file src/platform/windows/display_config.h
 * @brief Shared Windows display-topology and Advanced Color API adapters.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

namespace platf::display_config {

  /** Whether a desktop rectangle has a positive width and height, each of which fits in an int.
   * The widths are computed in 64 bits, so a rectangle spanning the whole LONG range is invalid
   * rather than an overflow.
   */
  [[nodiscard]] inline bool valid_rect(const RECT &rect) noexcept {
    const auto width = static_cast<std::int64_t>(rect.right) - rect.left;
    const auto height = static_cast<std::int64_t>(rect.bottom) - rect.top;
    return width > 0 && height > 0 && width <= std::numeric_limits<int>::max() && height <= std::numeric_limits<int>::max();
  }

  /** Whether two desktop rectangles have the same edges. */
  [[nodiscard]] inline bool same_rect(const RECT &left, const RECT &right) noexcept {
    return left.left == right.left && left.top == right.top && left.right == right.right && left.bottom == right.bottom;
  }

  /** Whether two adapter identifiers name the same adapter. */
  [[nodiscard]] inline bool same_luid(const LUID &left, const LUID &right) noexcept {
    return left.HighPart == right.HighPart && left.LowPart == right.LowPart;
  }

  /**
   * Whether two Windows display device names are equal: monitor and adapter device paths
   * (`\\?\DISPLAY#...`) and GDI device names (`\\.\DISPLAYn`). They are case-insensitive, and
   * Windows does not keep their case stable between its APIs (CCD, SetupAPI, the registry),
   * so the comparison ignores it. Two empty names are equal.
   */
  [[nodiscard]] inline bool equal_device_paths(std::wstring_view left, std::wstring_view right) noexcept {
    return left.size() == right.size() &&
           (left.empty() || CompareStringOrdinal(left.data(), static_cast<int>(left.size()), right.data(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL);
  }

  /** Whether two display device names name the same device. An empty name names no device and
   * matches nothing, not even another empty name.
   */
  [[nodiscard]] inline bool same_device_path(std::wstring_view left, std::wstring_view right) noexcept {
    return !left.empty() && equal_device_paths(left, right);
  }

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

  struct advanced_color_state_t {
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

  struct display_target_t {
    LUID adapter_id {};
    UINT32 target_id = 0;
    std::wstring device_path;
    std::wstring display_name;
  };

  /** Resolve one exact active target. A learned device path always takes precedence over
   * the recyclable GDI name; clones and ambiguous identities are rejected.
   */
  std::optional<display_target_t> find_display_target(
    const std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
    std::wstring_view device_path,
    std::wstring_view display_name,
    const device_info_api_t &api
  );
  std::optional<display_target_t> resolve_display_target(
    std::wstring_view device_path,
    std::wstring_view display_name = {}
  );

  /** Query the Windows 11 24H2 Advanced Color state; a failed query reports no state. */
  std::optional<advanced_color_state_t> query_advanced_color(
    const LUID &adapter_id,
    UINT32 target_id,
    const device_info_api_t &api
  );

  std::optional<advanced_color_state_t> query_advanced_color(
    const LUID &adapter_id,
    UINT32 target_id
  );

  // Every HDR and WCG request below advances platf::display_change_epoch() before and after the
  // call, because the change restarts Desktop Duplication.
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

  /** What one look at the display shows after a color request. */
  enum class color_observation_e {
    matched,  ///< Windows reports every requested state.
    pending,  ///< Not yet.
    failed,  ///< The request can no longer settle, for example because the topology moved.
  };

  enum class color_change_result_e {
    rejected,  ///< The request was not accepted.
    interrupted,  ///< The caller's wait was cut short.
    unverified,  ///< Windows did not report the requested state within the bound.
    verified,
  };

  struct color_settle_policy_t {
    std::chrono::milliseconds bound;
    std::chrono::milliseconds poll;
  };

  /**
   * Request a color change and wait until Windows reports it. Windows can accept an Advanced
   * Color request before it reports the new state, so the display is observed at once and then
   * after every poll interval, for at most bound / poll more observations. `wait` sleeps for one
   * interval and returns false when the caller's wait was interrupted.
   */
  color_change_result_e change_color_and_settle(
    const std::function<bool()> &request,
    const std::function<color_observation_e()> &observe,
    color_settle_policy_t policy,
    const std::function<bool(std::chrono::milliseconds)> &wait
  );

}  // namespace platf::display_config
