/**
 * @file src/platform/windows/display_config.cpp
 * @brief Shared Windows display-topology and Advanced Color API adapters.
 */
#include "display_config.h"

#include "src/platform/common.h"

#include <algorithm>

namespace platf::display_config {

  namespace {
    // An Advanced Color change invalidates Desktop Duplication as a mode change does. Record it
    // before and after the request, as VDISPLAY::changeDisplaySettings does for mode sets, so
    // capture failover can attribute the restart without depending on a Windows notification.
    bool set_color_device_info(const device_info_api_t &api, DISPLAYCONFIG_DEVICE_INFO_HEADER &header) {
      platf::note_display_change();
      const bool accepted = api.set(&header) == ERROR_SUCCESS;
      platf::note_display_change();
      return accepted;
    }

    /** The name in a fixed-size CCD field, or nothing when the field holds no terminator. */
    template<std::size_t size>
    std::optional<std::wstring> terminated_name(const wchar_t (&name)[size]) {
      const auto end = std::find(std::begin(name), std::end(name), L'\0');
      if (end == std::end(name)) {
        return std::nullopt;
      }
      return std::wstring(std::begin(name), end);
    }
  }  // namespace

  target_name_t query_target_name(const LUID &adapter_id, const UINT32 target_id, const device_info_api_t &api) {
    if (!api.get) {
      return {};
    }
    DISPLAYCONFIG_TARGET_DEVICE_NAME target {};
    target.header = {DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME, sizeof(target), adapter_id, target_id};
    target_name_t result {api.get(&target.header)};
    if (!result) {
      return result;
    }
    auto device_path = terminated_name(target.monitorDevicePath);
    auto friendly_name = terminated_name(target.monitorFriendlyDeviceName);
    if (!device_path || !friendly_name) {
      return {ERROR_INVALID_DATA};
    }
    result.device_path = std::move(*device_path);
    result.friendly_name = std::move(*friendly_name);
    return result;
  }

  std::optional<std::wstring> query_source_name(const LUID &adapter_id, const UINT32 source_id, const device_info_api_t &api) {
    if (!api.get) {
      return std::nullopt;
    }
    DISPLAYCONFIG_SOURCE_DEVICE_NAME source {};
    source.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof(source), adapter_id, source_id};
    if (api.get(&source.header) != ERROR_SUCCESS) {
      return std::nullopt;
    }
    return terminated_name(source.viewGdiDeviceName);
  }

  std::optional<display_target_t> find_display_target(
    const std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
    std::wstring_view device_path,
    std::wstring_view display_name,
    const device_info_api_t &api
  ) {
    if (!api.get || (device_path.empty() && display_name.empty())) {
      return std::nullopt;
    }
    std::optional<display_target_t> found;
    for (const auto &path : paths) {
      if (!(path.flags & DISPLAYCONFIG_PATH_ACTIVE)) {
        continue;
      }
      // An unresolved active path can make a GDI-name match ambiguous (cloning).
      auto target = query_target_name(path.targetInfo.adapterId, path.targetInfo.id, api);
      if (!target) {
        return std::nullopt;
      }
      auto source = query_source_name(path.sourceInfo.adapterId, path.sourceInfo.id, api);
      if (!source || target.device_path.empty()) {
        return std::nullopt;
      }
      if (!(device_path.empty() ? same_device_path(display_name, *source) :
                                  same_device_path(device_path, target.device_path))) {
        continue;
      }
      if (found) {
        return std::nullopt;
      }
      found = display_target_t {path.targetInfo.adapterId, path.targetInfo.id, std::move(target.device_path), std::move(*source)};
    }
    return found;
  }

  std::optional<display_target_t> resolve_display_target(
    std::wstring_view device_path,
    std::wstring_view display_name
  ) {
    topology_t topology;
    if (!query_topology(QDC_ONLY_ACTIVE_PATHS, topology)) {
      return std::nullopt;
    }
    return find_display_target(topology.paths, device_path, display_name, device_info_api_t {});
  }

  named_query_result_t query_named_topology(
    const UINT32 flags,
    named_topology_t &topology,
    const unnamed_target_e unnamed,
    const query_policy_t policy,
    const query_api_t &api,
    const device_info_api_t &device_info
  ) {
    topology.device_paths.clear();
    named_query_result_t result;
    result.status = query_topology(flags, topology, policy, api).status;
    if (!result) {
      return result;
    }
    topology.device_paths.reserve(topology.paths.size());
    for (const auto &path : topology.paths) {
      auto target = query_target_name(path.targetInfo.adapterId, path.targetInfo.id, device_info);
      if (target && target.device_path.empty() && unnamed == unnamed_target_e::rejected) {
        target.status = ERROR_NOT_FOUND;
      }
      if (!target && unnamed == unnamed_target_e::rejected) {
        result.status = target.status;
        result.unnamed_target = path.targetInfo;
        topology.paths.clear();
        topology.modes.clear();
        topology.device_paths.clear();
        return result;
      }
      topology.device_paths.emplace_back(target ? std::move(target.device_path) : std::wstring {});
    }
    return result;
  }

  query_result_t query_topology(
    const UINT32 flags,
    topology_t &topology,
    const query_policy_t policy,
    const query_api_t &api
  ) {
    auto &paths = topology.paths;
    auto &modes = topology.modes;
    if (!api.get_buffer_sizes || !api.query || !api.sleep || policy.max_attempts == 0) {
      paths.clear();
      modes.clear();
      return {};
    }

    for (unsigned int attempt = 0; attempt < policy.max_attempts; ++attempt) {
      UINT32 path_count = 0;
      UINT32 mode_count = 0;
      const auto size_status =
        api.get_buffer_sizes(flags, &path_count, &mode_count);
      if (size_status == ERROR_SUCCESS) {
        paths.resize(path_count);
        modes.resize(mode_count);
        const auto query_status = api.query(
          flags,
          &path_count,
          paths.data(),
          &mode_count,
          modes.data(),
          nullptr
        );
        if (query_status == ERROR_SUCCESS) {
          paths.resize(path_count);
          modes.resize(mode_count);
          return {ERROR_SUCCESS};
        }
        if (query_status != ERROR_INSUFFICIENT_BUFFER) {
          paths.clear();
          modes.clear();
          return {query_status};
        }
      } else if (size_status != ERROR_INSUFFICIENT_BUFFER) {
        paths.clear();
        modes.clear();
        return {size_status};
      }

      if (attempt + 1 < policy.max_attempts && policy.retry_delay == retry_delay_e::exponential_milliseconds) {
        api.sleep(1u << std::min(attempt, 5u));
      }
    }

    paths.clear();
    modes.clear();
    return {ERROR_INSUFFICIENT_BUFFER};
  }


  std::optional<advanced_color_state_t> query_advanced_color(
    const LUID &adapter_id,
    const UINT32 target_id,
    const device_info_api_t &api
  ) {
    if (!api.get) {
      return std::nullopt;
    }

    DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO_2 info {};
    info.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO_2;
    info.header.size = sizeof(info);
    info.header.adapterId = adapter_id;
    info.header.id = target_id;
    if (api.get(&info.header) != ERROR_SUCCESS) {
      return std::nullopt;
    }
    return advanced_color_state_t {
      info.highDynamicRangeSupported != 0,
      info.highDynamicRangeUserEnabled != 0,
      info.wideColorUserEnabled != 0,
      info.advancedColorActive != 0,
      info.advancedColorLimitedByPolicy != 0,
      info.bitsPerColorChannel,
      info.activeColorMode,
    };
  }

  std::optional<advanced_color_state_t> query_advanced_color(
    const LUID &adapter_id,
    const UINT32 target_id
  ) {
    return query_advanced_color(adapter_id, target_id, device_info_api_t {});
  }

  bool set_hdr_state(
    const LUID &adapter_id,
    const UINT32 target_id,
    const bool enabled,
    const device_info_api_t &api
  ) {
    if (!api.set) {
      return false;
    }
    DISPLAYCONFIG_SET_HDR_STATE state {};
    state.header.type = DISPLAYCONFIG_DEVICE_INFO_SET_HDR_STATE;
    state.header.size = sizeof(state);
    state.header.adapterId = adapter_id;
    state.header.id = target_id;
    state.enableHdr = enabled;
    return set_color_device_info(api, state.header);
  }

  bool set_hdr_state(
    const LUID &adapter_id,
    const UINT32 target_id,
    const bool enabled
  ) {
    return set_hdr_state(adapter_id, target_id, enabled, device_info_api_t {});
  }

  bool set_wcg_state(
    const LUID &adapter_id,
    const UINT32 target_id,
    const bool enabled,
    const device_info_api_t &api
  ) {
    if (!api.set) {
      return false;
    }
    DISPLAYCONFIG_SET_WCG_STATE state {};
    state.header.type = DISPLAYCONFIG_DEVICE_INFO_SET_WCG_STATE;
    state.header.size = sizeof(state);
    state.header.adapterId = adapter_id;
    state.header.id = target_id;
    state.enableWcg = enabled;
    return set_color_device_info(api, state.header);
  }

  bool set_wcg_state(
    const LUID &adapter_id,
    const UINT32 target_id,
    const bool enabled
  ) {
    return set_wcg_state(adapter_id, target_id, enabled, device_info_api_t {});
  }

  color_change_result_e change_color_and_settle(
    const std::function<bool()> &request,
    const std::function<color_observation_e()> &observe,
    const color_settle_policy_t policy,
    const std::function<bool(std::chrono::milliseconds)> &wait
  ) {
    if (!request()) {
      return color_change_result_e::rejected;
    }
    const std::chrono::milliseconds::rep polls = policy.poll.count() > 0 ? policy.bound / policy.poll : 0;
    for (std::chrono::milliseconds::rep poll = 0;; ++poll) {
      switch (observe()) {
        case color_observation_e::matched:
          return color_change_result_e::verified;
        case color_observation_e::failed:
          return color_change_result_e::unverified;
        case color_observation_e::pending:
          break;
      }
      if (poll >= polls) {
        return color_change_result_e::unverified;
      }
      if (!wait(policy.poll)) {
        return color_change_result_e::interrupted;
      }
    }
  }

}  // namespace platf::display_config
