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

  /** One CCD observation: the paths and modes that a single QueryDisplayConfig reported. */
  struct topology_t {
    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    std::vector<DISPLAYCONFIG_MODE_INFO> modes;
  };

  /** One CCD observation and the monitor device path of each path's target, in path order. */
  struct named_topology_t: topology_t {
    std::vector<std::wstring> device_paths;
  };

  /**
   * Query one CCD topology, retrying the documented size/query race. The flags choose the
   * paths and how their mode indices are laid out, so each caller passes its own. A failed query
   * leaves the topology empty. The api is injected so tests can drive the retry state machine
   * without mutating CCD.
   */
  query_result_t query_topology(
    UINT32 flags,
    topology_t &topology,
    query_policy_t policy = {},
    const query_api_t &api = {}
  );

  struct device_info_api_t {
    decltype(&::DisplayConfigGetDeviceInfo) get = &::DisplayConfigGetDeviceInfo;
    decltype(&::DisplayConfigSetDeviceInfo) set = &::DisplayConfigSetDeviceInfo;
  };

  /** What Windows reports for one target's monitor. */
  struct target_name_t {
    LONG status = ERROR_INVALID_PARAMETER;
    std::wstring device_path;  ///< May be empty when Windows names no monitor interface.
    std::wstring friendly_name;

    [[nodiscard]] explicit operator bool() const noexcept {
      return status == ERROR_SUCCESS;
    }
  };

  /** Look up a target's monitor names. A name without its terminator is a failed lookup. */
  target_name_t query_target_name(const LUID &adapter_id, UINT32 target_id, const device_info_api_t &api = {});

  /** Look up a source's GDI device name (`\\.\DISPLAYn`); nothing when the lookup fails. */
  std::optional<std::wstring> query_source_name(const LUID &adapter_id, UINT32 source_id, const device_info_api_t &api = {});

  enum class unnamed_target_e {
    rejected,  ///< A target without a monitor device path fails the whole query.
    allowed,  ///< Such a target gets an empty device path.
  };

  struct named_query_result_t: query_result_t {
    /** The target that failed the query because it could not be named. */
    std::optional<DISPLAYCONFIG_PATH_TARGET_INFO> unnamed_target;
  };

  /**
   * Query one CCD topology and name every path's target. When a name lookup fails, or finds no
   * monitor device path while unnamed targets are rejected, the query fails with that lookup's
   * status (ERROR_NOT_FOUND for a missing path) and reports the target.
   */
  named_query_result_t query_named_topology(
    UINT32 flags,
    named_topology_t &topology,
    unnamed_target_e unnamed,
    query_policy_t policy = {},
    const query_api_t &api = {},
    const device_info_api_t &device_info = {}
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

  using set_display_config_t = decltype(&::SetDisplayConfig);

  /** Pass a supplied topology to SetDisplayConfig with the caller's flags. */
  LONG set_display_config(topology_t &topology, UINT32 flags, set_display_config_t set = &::SetDisplayConfig);

  enum class supplied_apply_e {
    invalid,  ///< SDC_VALIDATE rejected the supplied topology.
    stale,  ///< The re-query between validation and apply found a changed desktop.
    failed,  ///< SDC_APPLY failed.
    applied,
  };

  struct supplied_apply_result_t {
    supplied_apply_e state = supplied_apply_e::invalid;
    LONG status = ERROR_SUCCESS;  ///< The status of the failed call, or of the apply.
  };

  /**
   * Validate a supplied topology, then apply it. SDC_VALIDATE does not reserve the desktop
   * topology, so a caller that must not overwrite a change made in between passes `requery`. It
   * runs after a successful validation and either replaces the topology with a fresh observation
   * carrying the same edit, or returns false when the desktop is no longer the validated one;
   * nothing is applied then. Both calls use `flags` with SDC_USE_SUPPLIED_DISPLAY_CONFIG, and
   * `apply_flags` are added to the apply only.
   */
  supplied_apply_result_t validate_and_apply(
    topology_t &topology,
    UINT32 flags,
    UINT32 apply_flags = 0,
    const std::function<bool(topology_t &)> &requery = {},
    set_display_config_t set = &::SetDisplayConfig
  );

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
