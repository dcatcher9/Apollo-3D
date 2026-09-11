/**
 * @file src/platform/windows/display_config.cpp
 * @brief Shared Windows display-topology and Advanced Color API adapters.
 */
#include "display_config.h"

#include <algorithm>

namespace platf::display_config {

  std::optional<display_target_t> find_display_target(
    const std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
    std::wstring_view device_path,
    std::wstring_view display_name,
    const device_info_api_t &api
  ) {
    if (!api.get || (device_path.empty() && display_name.empty())) {
      return std::nullopt;
    }
    const auto same_name = [](std::wstring_view left, std::wstring_view right) {
      return CompareStringOrdinal(left.data(), static_cast<int>(left.size()), right.data(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL;
    };
    std::optional<display_target_t> found;
    for (const auto &path : paths) {
      if (!(path.flags & DISPLAYCONFIG_PATH_ACTIVE)) {
        continue;
      }
      DISPLAYCONFIG_TARGET_DEVICE_NAME target {};
      target.header = {DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME, sizeof(target), path.targetInfo.adapterId, path.targetInfo.id};
      DISPLAYCONFIG_SOURCE_DEVICE_NAME source {};
      source.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof(source), path.sourceInfo.adapterId, path.sourceInfo.id};
      if (api.get(&target.header) != ERROR_SUCCESS || api.get(&source.header) != ERROR_SUCCESS || !target.monitorDevicePath[0]) {
        // An unresolved active path can make a GDI-name match ambiguous (cloning).
        return std::nullopt;
      }
      if (!(device_path.empty() ? same_name(display_name, source.viewGdiDeviceName) :
                                  same_name(device_path, target.monitorDevicePath))) {
        continue;
      }
      if (found) {
        return std::nullopt;
      }
      found = display_target_t {path.targetInfo.adapterId, path.targetInfo.id, target.monitorDevicePath, source.viewGdiDeviceName};
    }
    return found;
  }

  std::optional<display_target_t> resolve_display_target(
    std::wstring_view device_path,
    std::wstring_view display_name
  ) {
    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    std::vector<DISPLAYCONFIG_MODE_INFO> modes;
    if (!query_display_config(QDC_ONLY_ACTIVE_PATHS, paths, modes)) {
      return std::nullopt;
    }
    return find_display_target(paths, device_path, display_name, device_info_api_t {});
  }

  query_result_t query_display_config(
    const UINT32 flags,
    std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
    std::vector<DISPLAYCONFIG_MODE_INFO> &modes,
    const query_api_t &api,
    const query_policy_t policy
  ) {
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

  query_result_t query_display_config(
    const UINT32 flags,
    std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
    std::vector<DISPLAYCONFIG_MODE_INFO> &modes,
    const query_policy_t policy
  ) {
    return query_display_config(flags, paths, modes, query_api_t {}, policy);
  }

  std::optional<advanced_color_state_t> query_advanced_color(
    const LUID &adapter_id,
    const UINT32 target_id,
    const legacy_fallback_e fallback,
    const device_info_api_t &api
  ) {
    if (!api.get) {
      return std::nullopt;
    }

    DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO_2 modern {};
    modern.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO_2;
    modern.header.size = sizeof(modern);
    modern.header.adapterId = adapter_id;
    modern.header.id = target_id;
    const auto modern_status = api.get(&modern.header);
    if (modern_status == ERROR_SUCCESS) {
      return advanced_color_state_t {
        advanced_color_api_e::modern,
        modern.advancedColorActive != 0,
        modern.highDynamicRangeSupported != 0,
        modern.highDynamicRangeUserEnabled != 0,
        modern.wideColorUserEnabled != 0,
        modern.advancedColorActive != 0,
        modern.advancedColorLimitedByPolicy != 0,
        modern.bitsPerColorChannel,
        modern.activeColorMode,
      };
    }

    const bool modern_api_unsupported =
      modern_status == ERROR_INVALID_PARAMETER || modern_status == ERROR_NOT_SUPPORTED;
    if (fallback == legacy_fallback_e::unsupported_modern_api && !modern_api_unsupported) {
      return std::nullopt;
    }

    DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO legacy {};
    legacy.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO;
    legacy.header.size = sizeof(legacy);
    legacy.header.adapterId = adapter_id;
    legacy.header.id = target_id;
    if (api.get(&legacy.header) != ERROR_SUCCESS) {
      return std::nullopt;
    }

    const bool enabled = legacy.advancedColorEnabled != 0;
    return advanced_color_state_t {
      advanced_color_api_e::legacy,
      enabled,
      legacy.advancedColorSupported != 0,
      enabled,
      false,
      enabled,
      legacy.advancedColorForceDisabled != 0,
      legacy.bitsPerColorChannel,
      enabled ? DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR :
                DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR,
    };
  }

  std::optional<advanced_color_state_t> query_advanced_color(
    const LUID &adapter_id,
    const UINT32 target_id,
    const legacy_fallback_e fallback
  ) {
    return query_advanced_color(adapter_id, target_id, fallback, device_info_api_t {});
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
    return api.set(&state.header) == ERROR_SUCCESS;
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
    return api.set(&state.header) == ERROR_SUCCESS;
  }

  bool set_wcg_state(
    const LUID &adapter_id,
    const UINT32 target_id,
    const bool enabled
  ) {
    return set_wcg_state(adapter_id, target_id, enabled, device_info_api_t {});
  }

  bool set_legacy_advanced_color_state(
    const LUID &adapter_id,
    const UINT32 target_id,
    const bool enabled,
    const device_info_api_t &api
  ) {
    if (!api.set) {
      return false;
    }
    DISPLAYCONFIG_SET_ADVANCED_COLOR_STATE state {};
    state.header.type = DISPLAYCONFIG_DEVICE_INFO_SET_ADVANCED_COLOR_STATE;
    state.header.size = sizeof(state);
    state.header.adapterId = adapter_id;
    state.header.id = target_id;
    state.enableAdvancedColor = enabled;
    return api.set(&state.header) == ERROR_SUCCESS;
  }

  bool set_hdr_state_with_legacy_fallback(
    const LUID &adapter_id,
    const UINT32 target_id,
    const bool enabled,
    const device_info_api_t &api
  ) {
    return set_hdr_state(adapter_id, target_id, enabled, api) ||
           set_legacy_advanced_color_state(adapter_id, target_id, enabled, api);
  }

  bool set_hdr_state_with_legacy_fallback(
    const LUID &adapter_id,
    const UINT32 target_id,
    const bool enabled
  ) {
    return set_hdr_state_with_legacy_fallback(
      adapter_id,
      target_id,
      enabled,
      device_info_api_t {}
    );
  }

}  // namespace platf::display_config
