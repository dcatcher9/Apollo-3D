/**
 * @file src/platform/windows/primary_display.cpp
 * @brief Temporary primary-display transactions with crash recovery.
 */
#include "primary_display.h"

#include "display_config.h"
#include "misc.h"
#include "src/config.h"
#include "src/logging.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>

namespace platf::primary_display {
  namespace {
    constexpr size_t max_displays = 128;
    constexpr size_t max_journal_bytes = 1024 * 1024;
    constexpr size_t max_available_paths = 16384;

    bool same_device(std::wstring_view left, std::wstring_view right) {
      return left.size() == right.size() && !left.empty() &&
             CompareStringOrdinal(left.data(), static_cast<int>(left.size()), right.data(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL;
    }

    bool same_adapter(const LUID &left, const LUID &right) {
      return left.LowPart == right.LowPart && left.HighPart == right.HighPart;
    }

    const detail::position_t *find_position(const detail::layout_t &layout, std::wstring_view identity) {
      const auto found = std::ranges::find_if(layout, [&](const auto &entry) {
        return same_device(entry.device_path, identity);
      });
      return found == layout.end() ? nullptr : &*found;
    }

    bool valid_layout(const detail::layout_t &layout) {
      if (layout.empty() || layout.size() > max_displays) {
        return false;
      }
      size_t origins = 0;
      for (size_t i = 0; i < layout.size(); ++i) {
        const auto &entry = layout[i];
        if (entry.device_path.empty() || entry.device_path.size() > 1024 || entry.device_path.find(L'\0') != std::wstring::npos) {
          return false;
        }
        origins += entry.x == 0 && entry.y == 0;
        for (size_t j = 0; j < i; ++j) {
          if (same_device(entry.device_path, layout[j].device_path)) {
            return false;
          }
        }
      }
      return origins == 1;
    }

    bool same_layout(const detail::layout_t &left, const detail::layout_t &right) {
      return left.size() == right.size() && std::ranges::all_of(left, [&](const auto &entry) {
               const auto *other = find_position(right, entry.device_path);
               return other && entry.x == other->x && entry.y == other->y;
             });
    }

    std::optional<LONG> subtract(LONG value, LONG offset) {
      const auto result = static_cast<int64_t>(value) - offset;
      if (result < std::numeric_limits<LONG>::min() || result > std::numeric_limits<LONG>::max()) {
        return std::nullopt;
      }
      return static_cast<LONG>(result);
    }

    std::optional<detail::layout_t> translated(const detail::layout_t &layout, LONG x, LONG y) {
      auto result = layout;
      for (auto &entry : result) {
        const auto new_x = subtract(entry.x, x);
        const auto new_y = subtract(entry.y, y);
        if (!new_x || !new_y) {
          return std::nullopt;
        }
        entry.x = *new_x;
        entry.y = *new_y;
      }
      return result;
    }

    UINT32 source_index(const DISPLAYCONFIG_PATH_INFO &path) {
      return path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE ? path.sourceInfo.sourceModeInfoIdx : path.sourceInfo.modeInfoIdx;
    }

    UINT32 target_index(const DISPLAYCONFIG_PATH_INFO &path) {
      return path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE ? path.targetInfo.targetModeInfoIdx : path.targetInfo.modeInfoIdx;
    }

    bool valid_journal(const detail::journal_t &journal) {
      if (!valid_layout(journal.original)) {
        return false;
      }
      const auto *original = find_position(journal.original, journal.original_primary);
      if (!original || original->x != 0 || original->y != 0) {
        return false;
      }
      if (journal.exclusive) {
        if (!journal.original_topology || !detail::inspect(*journal.original_topology) || journal.original_topology->colors.size() != journal.original_topology->paths.size() || !std::ranges::all_of(journal.original_topology->colors, [](const auto &color) {
              return color.has_value();
            })) {
          return false;
        }
        const auto saved = detail::inspect(*journal.original_topology);
        for (const auto &entry : *saved) {
          const auto *expected = find_position(journal.original, entry.device_path);
          if (!expected || entry.x != expected->x || entry.y != expected->y) {
            return false;
          }
        }
        if (journal.exclusive_started && (!journal.before_exclusive || !detail::inspect(*journal.before_exclusive) || journal.prepared)) {
          return false;
        }
        if (journal.pending_restore && ((!journal.exclusive_started && !journal.prepared) || !detail::inspect(*journal.pending_restore))) {
          return false;
        }
      } else if (journal.exclusive_started || journal.original_topology || journal.before_exclusive || journal.pending_restore) {
        return false;
      }
      if (journal.prepared) {
        return journal.promoted.empty() && journal.promoted_primary.empty();
      }
      if (!valid_layout(journal.promoted) || (!journal.exclusive && same_device(journal.original_primary, journal.promoted_primary))) {
        return false;
      }
      const auto *target = find_position(journal.original, journal.promoted_primary);
      if (!target) {
        return false;
      }
      const auto expected = translated(journal.original, target->x, target->y);
      return expected && same_layout(*expected, journal.promoted);
    }

    bool same_modes(const detail::snapshot_t &left, const detail::snapshot_t &right) {
      if (left.paths.size() != right.paths.size()) {
        return false;
      }
      for (size_t i = 0; i < left.paths.size(); ++i) {
        const auto match = std::ranges::find_if(right.device_paths, [&](const auto &identity) {
          return same_device(identity, left.device_paths[i]);
        });
        if (match == right.device_paths.end()) {
          return false;
        }
        const auto j = static_cast<size_t>(match - right.device_paths.begin());
        const auto &a = left.paths[i];
        const auto &b = right.paths[j];
        const auto &am = left.modes[source_index(a)].sourceMode;
        const auto &bm = right.modes[source_index(b)].sourceMode;
        const auto &at = a.targetInfo;
        const auto &bt = b.targetInfo;
        if (am.width != bm.width || am.height != bm.height || am.pixelFormat != bm.pixelFormat || at.rotation != bt.rotation || at.scaling != bt.scaling || at.outputTechnology != bt.outputTechnology || at.scanLineOrdering != bt.scanLineOrdering || static_cast<uint64_t>(at.refreshRate.Numerator) * bt.refreshRate.Denominator != static_cast<uint64_t>(bt.refreshRate.Numerator) * at.refreshRate.Denominator) {
          return false;
        }
        const auto &as = left.modes[target_index(a)].targetMode.targetVideoSignalInfo;
        const auto &bs = right.modes[target_index(b)].targetMode.targetVideoSignalInfo;
        if (as.pixelRate != bs.pixelRate || as.videoStandard != bs.videoStandard || as.activeSize.cx != bs.activeSize.cx || as.activeSize.cy != bs.activeSize.cy || as.totalSize.cx != bs.totalSize.cx || as.totalSize.cy != bs.totalSize.cy || as.scanLineOrdering != bs.scanLineOrdering || static_cast<uint64_t>(as.hSyncFreq.Numerator) * bs.hSyncFreq.Denominator != static_cast<uint64_t>(bs.hSyncFreq.Numerator) * as.hSyncFreq.Denominator || static_cast<uint64_t>(as.vSyncFreq.Numerator) * bs.vSyncFreq.Denominator != static_cast<uint64_t>(bs.vSyncFreq.Numerator) * as.vSyncFreq.Denominator) {
          return false;
        }
        const auto desktop_a = a.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE ? a.targetInfo.desktopModeInfoIdx : DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID;
        const auto desktop_b = b.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE ? b.targetInfo.desktopModeInfoIdx : DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID;
        if ((desktop_a == DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID) != (desktop_b == DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID)) {
          return false;
        }
        if (desktop_a != DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID) {
          // DISPLAYCONFIG_DESKTOP_IMAGE_INFO consists of ten 32-bit signed coordinate
          // fields; compare those fields without including unused union bytes/padding.
          std::array<LONG, 10> image_a {}, image_b {};
          std::memcpy(image_a.data(), &left.modes[desktop_a].targetMode, sizeof(image_a));
          std::memcpy(image_b.data(), &right.modes[desktop_b].targetMode, sizeof(image_b));
          if (image_a != image_b) {
            return false;
          }
        }
      }
      return true;
    }

    std::optional<size_t> find_path(const detail::snapshot_t &snapshot, std::wstring_view identity) {
      const auto found = std::ranges::find_if(snapshot.device_paths, [&](const auto &path) {
        return same_device(path, identity);
      });
      return found == snapshot.device_paths.end() ? std::nullopt : std::make_optional(static_cast<size_t>(found - snapshot.device_paths.begin()));
    }

    bool same_topology(const detail::snapshot_t &a, const detail::snapshot_t &b) {
      if (a.paths.empty() || b.paths.empty()) {
        return a.paths.empty() && b.paths.empty();
      }
      const auto al = detail::inspect(a);
      const auto bl = detail::inspect(b);
      return al && bl && same_layout(*al, *bl) && same_modes(a, b);
    }

    struct display_spec_t {
      std::wstring identity;
      DISPLAYCONFIG_PATH_INFO path {};
      DISPLAYCONFIG_SOURCE_MODE source {};
      DISPLAYCONFIG_TARGET_MODE target {};
      std::optional<DISPLAYCONFIG_MODE_INFO> desktop;
      std::optional<display_config::advanced_color_state_t> color;
    };

    display_spec_t display_spec(const detail::snapshot_t &snapshot, size_t index) {
      const auto &path = snapshot.paths[index];
      display_spec_t result {snapshot.device_paths[index], path, snapshot.modes[source_index(path)].sourceMode, snapshot.modes[target_index(path)].targetMode};
      if (path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE) {
        const auto desktop = path.targetInfo.desktopModeInfoIdx;
        if (desktop != DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID && desktop < snapshot.modes.size() && static_cast<int>(snapshot.modes[desktop].infoType) == 3) {
          result.desktop = snapshot.modes[desktop];
        }
      }
      if (index < snapshot.colors.size()) {
        result.color = snapshot.colors[index];
      }
      return result;
    }

    std::optional<detail::snapshot_t> build_topology(const detail::snapshot_t &available, const std::vector<display_spec_t> &wanted) {
      if (wanted.empty() || wanted.size() > max_displays || available.paths.size() > max_available_paths || available.paths.size() != available.device_paths.size()) {
        return std::nullopt;
      }
      using source_key_t = std::tuple<LONG, DWORD, UINT32>;
      std::map<source_key_t, size_t> sources;

      struct route_t {
        size_t path;
        size_t source;
      };

      std::vector<std::vector<route_t>> candidates(wanted.size());
      for (size_t i = 0; i < wanted.size(); ++i) {
        std::optional<std::pair<LUID, UINT32>> target;
        std::set<size_t> seen;
        for (size_t p = 0; p < available.paths.size(); ++p) {
          const auto &path = available.paths[p];
          if (!path.targetInfo.targetAvailable || !same_device(wanted[i].identity, available.device_paths[p])) {
            continue;
          }
          if (target && (!same_adapter(target->first, path.targetInfo.adapterId) || target->second != path.targetInfo.id)) {
            return std::nullopt;
          }
          target = std::pair {path.targetInfo.adapterId, path.targetInfo.id};
          const source_key_t key {path.sourceInfo.adapterId.HighPart, path.sourceInfo.adapterId.LowPart, path.sourceInfo.id};
          const auto entry = sources.emplace(key, sources.size()).first;
          if (seen.insert(entry->second).second) {
            candidates[i].push_back({p, entry->second});
          }
        }
        if (candidates[i].empty()) {
          return std::nullopt;
        }
      }
      // QDC_ALL_PATHS contains alternative routes, not duplicate monitors. An augmenting
      // match keeps sources unique while allowing a later target to displace an earlier
      // greedy choice onto another valid route.
      std::vector<int> owner(sources.size(), -1);
      std::vector<size_t> selected(wanted.size());
      std::function<bool(size_t, std::vector<bool> &)> assign = [&](size_t display, std::vector<bool> &visited) {
        for (const auto &route : candidates[display]) {
          if (visited[route.source]) {
            continue;
          }
          visited[route.source] = true;
          if (owner[route.source] < 0 || assign(static_cast<size_t>(owner[route.source]), visited)) {
            owner[route.source] = static_cast<int>(display);
            selected[display] = route.path;
            return true;
          }
        }
        return false;
      };
      for (size_t i = 0; i < wanted.size(); ++i) {
        std::vector<bool> visited(sources.size());
        if (!assign(i, visited)) {
          return std::nullopt;
        }
      }
      detail::snapshot_t result;
      for (size_t i = 0; i < wanted.size(); ++i) {
        const auto &spec = wanted[i];
        auto path = available.paths[selected[i]];
        path.flags = DISPLAYCONFIG_PATH_ACTIVE | (path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE) | (spec.path.flags & (DISPLAYCONFIG_PATH_PREFERRED_UNSCALED | DISPLAYCONFIG_PATH_BOOST_REFRESH_RATE));
        path.targetInfo.rotation = spec.path.targetInfo.rotation;
        path.targetInfo.scaling = spec.path.targetInfo.scaling;
        path.targetInfo.refreshRate = spec.path.targetInfo.refreshRate;
        path.targetInfo.scanLineOrdering = spec.path.targetInfo.scanLineOrdering;
        DISPLAYCONFIG_MODE_INFO source {};
        source.infoType = DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE;
        source.id = path.sourceInfo.id;
        source.adapterId = path.sourceInfo.adapterId;
        source.sourceMode = spec.source;
        DISPLAYCONFIG_MODE_INFO target {};
        target.infoType = DISPLAYCONFIG_MODE_INFO_TYPE_TARGET;
        target.id = path.targetInfo.id;
        target.adapterId = path.targetInfo.adapterId;
        target.targetMode = spec.target;
        const auto source_idx = static_cast<UINT32>(result.modes.size());
        const auto target_idx = source_idx + 1;
        result.modes.push_back(source);
        result.modes.push_back(target);
        if (path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE) {
          path.sourceInfo.cloneGroupId = DISPLAYCONFIG_PATH_CLONE_GROUP_INVALID;
          path.sourceInfo.sourceModeInfoIdx = source_idx;
          path.targetInfo.targetModeInfoIdx = target_idx;
          path.targetInfo.desktopModeInfoIdx = DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID;
          if (spec.desktop) {
            auto desktop = *spec.desktop;
            desktop.adapterId = path.targetInfo.adapterId;
            desktop.id = path.targetInfo.id;
            path.targetInfo.desktopModeInfoIdx = static_cast<UINT32>(result.modes.size());
            result.modes.push_back(desktop);
          }
        } else {
          path.sourceInfo.modeInfoIdx = source_idx;
          path.targetInfo.modeInfoIdx = target_idx;
        }
        result.paths.push_back(path);
        result.device_paths.push_back(spec.identity);
        result.colors.push_back(spec.color);
      }
      return detail::inspect(result) ? std::make_optional(std::move(result)) : std::nullopt;
    }

    bool owned_subset(const detail::snapshot_t &current, const detail::snapshot_t &reference) {
      if (current.paths.empty()) {
        return true;
      }
      const auto layout = detail::inspect(current);
      if (!layout || !detail::inspect(reference)) {
        return false;
      }
      detail::snapshot_t subset;
      subset.modes = reference.modes;
      std::optional<std::pair<int64_t, int64_t>> translation;
      for (size_t i = 0; i < current.paths.size(); ++i) {
        const auto saved = find_path(reference, current.device_paths[i]);
        if (!saved) {
          return false;
        }
        const auto &point = reference.modes[source_index(reference.paths[*saved])].sourceMode.position;
        const auto delta = std::pair {static_cast<int64_t>(layout->at(i).x) - point.x, static_cast<int64_t>(layout->at(i).y) - point.y};
        if (translation && *translation != delta) {
          return false;
        }
        translation = delta;
        subset.paths.push_back(reference.paths[*saved]);
        subset.device_paths.push_back(reference.device_paths[*saved]);
      }
      return same_modes(current, subset);
    }

    bool color_matches(const display_config::advanced_color_state_t &a, const display_config::advanced_color_state_t &b) {
      if (a.api != b.api) {
        return false;
      }
      if (a.api == display_config::advanced_color_api_e::legacy) {
        return a.advanced_color_enabled == b.advanced_color_enabled;
      }
      return a.hdr_user_enabled == b.hdr_user_enabled && a.wcg_user_enabled == b.wcg_user_enabled && a.advanced_color_active == b.advanced_color_active && a.active_mode == b.active_mode;
    }

    template<class T>
    std::string encode_ccd(const T &value) {
      static_assert(std::is_trivially_copyable_v<T>);
      constexpr char digits[] = "0123456789abcdef";
      std::string result(sizeof(T) * 2, '0');
      const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
      for (size_t i = 0; i < sizeof(T); ++i) {
        result[i * 2] = digits[bytes[i] >> 4];
        result[i * 2 + 1] = digits[bytes[i] & 15];
      }
      return result;
    }

    template<class T>
    T decode_ccd(const nlohmann::json &value) {
      const auto encoded = value.get<std::string>();
      if (encoded.size() != sizeof(T) * 2) {
        throw std::runtime_error("unsupported CCD structure size");
      }
      T result {};
      auto *bytes = reinterpret_cast<unsigned char *>(&result);
      auto nibble = [](char c) -> unsigned char {
        if (c >= '0' && c <= '9') {
          return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
          return c - 'a' + 10;
        }
        throw std::runtime_error("invalid CCD encoding");
      };
      for (size_t i = 0; i < sizeof(T); ++i) {
        bytes[i] = static_cast<unsigned char>((nibble(encoded[i * 2]) << 4) | nibble(encoded[i * 2 + 1]));
      }
      return result;
    }

    nlohmann::json encode_snapshot(const detail::snapshot_t &snapshot) {
      nlohmann::json result {{"encoding", "windows-ccd-v1"}, {"paths", nlohmann::json::array()}, {"modes", nlohmann::json::array()}, {"devices", nlohmann::json::array()}, {"colors", nlohmann::json::array()}};
      // CCD structures have a fixed Win32 ABI. Preserve all timing/scaling data, including
      // desktop-image modes absent from older MinGW headers. IDs/indexes are remapped on
      // recovery; this payload is never sent directly back to SetDisplayConfig.
      for (const auto &path : snapshot.paths) {
        result["paths"].push_back(encode_ccd(path));
      }
      for (const auto &mode : snapshot.modes) {
        result["modes"].push_back(encode_ccd(mode));
      }
      for (const auto &identity : snapshot.device_paths) {
        result["devices"].push_back(platf::to_utf8(identity));
      }
      for (const auto &color : snapshot.colors) {
        if (!color) {
          result["colors"].push_back(nullptr);
        } else {
          result["colors"].push_back({{"legacy", color->api == display_config::advanced_color_api_e::legacy}, {"enabled", color->advanced_color_enabled}, {"hdr_supported", color->hdr_supported}, {"hdr", color->hdr_user_enabled}, {"wcg", color->wcg_user_enabled}, {"active", color->advanced_color_active}, {"limited", color->limited_by_policy}, {"bits", color->bits_per_color_channel}, {"mode", static_cast<int>(color->active_mode)}});
        }
      }
      return result;
    }

    detail::snapshot_t decode_snapshot(const nlohmann::json &value) {
      if (value.at("encoding") != "windows-ccd-v1" || !value.at("paths").is_array() || value.at("paths").size() > max_displays || !value.at("modes").is_array() || value.at("modes").size() > max_displays * 3 || !value.at("devices").is_array() || value.at("devices").size() != value.at("paths").size() || !value.at("colors").is_array() || value.at("colors").size() > value.at("paths").size()) {
        throw std::runtime_error("invalid saved display topology");
      }
      detail::snapshot_t result;
      for (const auto &path : value.at("paths")) {
        result.paths.push_back(decode_ccd<DISPLAYCONFIG_PATH_INFO>(path));
      }
      for (const auto &mode : value.at("modes")) {
        result.modes.push_back(decode_ccd<DISPLAYCONFIG_MODE_INFO>(mode));
      }
      for (const auto &identity : value.at("devices")) {
        result.device_paths.push_back(platf::from_utf8(identity.get<std::string>()));
      }
      for (const auto &entry : value.at("colors")) {
        if (entry.is_null()) {
          result.colors.push_back(std::nullopt);
          continue;
        }
        const int mode = entry.at("mode").get<int>();
        if (mode < DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR || mode > DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR) {
          throw std::runtime_error("invalid saved display color mode");
        }
        result.colors.push_back(display_config::advanced_color_state_t {entry.at("legacy").get<bool>() ? display_config::advanced_color_api_e::legacy : display_config::advanced_color_api_e::modern, entry.at("enabled").get<bool>(), entry.at("hdr_supported").get<bool>(), entry.at("hdr").get<bool>(), entry.at("wcg").get<bool>(), entry.at("active").get<bool>(), entry.at("limited").get<bool>(), entry.at("bits").get<UINT32>(), static_cast<DISPLAYCONFIG_ADVANCED_COLOR_MODE>(mode)});
      }
      if (!detail::inspect(result)) {
        throw std::runtime_error("invalid saved display paths or modes");
      }
      return result;
    }

    std::optional<detail::layout_t> restrict_layout(const detail::layout_t &recorded, const detail::layout_t &current, std::wstring_view virtual_identity) {
      // A crashed host can lose its virtual display before recovery. No other missing or
      // newly attached monitor is silently adopted into an old transaction.
      const bool virtual_missing = !find_position(current, virtual_identity);
      if (recorded.size() != current.size() + (virtual_missing ? 1 : 0)) {
        return std::nullopt;
      }
      detail::layout_t result;
      for (const auto &entry : recorded) {
        if (virtual_missing && same_device(entry.device_path, virtual_identity)) {
          continue;
        }
        if (!find_position(current, entry.device_path)) {
          return std::nullopt;
        }
        result.push_back(entry);
      }
      return result;
    }

    std::filesystem::path journal_path() {
      auto path = std::filesystem::path(platf::from_utf8(config::sunshine.config_file));
      path += L".apollo-primary-display.json";
      return path;
    }

    std::optional<detail::snapshot_t> query_snapshot_flags(UINT32 flags) {
      detail::snapshot_t result;
      const auto status = display_config::query_display_config(flags | QDC_VIRTUAL_MODE_AWARE | QDC_VIRTUAL_REFRESH_RATE_AWARE, result.paths, result.modes);
      if (!status || result.paths.size() > (flags == QDC_ALL_PATHS ? max_available_paths : max_displays)) {
        BOOST_LOG(warning) << "Could not query displays for temporary primary-display change: " << status.status;
        return std::nullopt;
      }
      for (const auto &path : result.paths) {
        DISPLAYCONFIG_TARGET_DEVICE_NAME name {};
        name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
        name.header.size = sizeof(name);
        name.header.adapterId = path.targetInfo.adapterId;
        name.header.id = path.targetInfo.id;
        const bool named = DisplayConfigGetDeviceInfo(&name.header) == ERROR_SUCCESS && name.monitorDevicePath[0];
        if (!named && flags != QDC_ALL_PATHS) {
          return std::nullopt;
        }
        const auto end = std::find(std::begin(name.monitorDevicePath), std::end(name.monitorDevicePath), L'\0');
        if (end == std::end(name.monitorDevicePath)) {
          return std::nullopt;
        }
        result.device_paths.emplace_back(named ? std::wstring(name.monitorDevicePath, end) : std::wstring());
        result.colors.push_back(path.flags & DISPLAYCONFIG_PATH_ACTIVE ? display_config::query_advanced_color(path.targetInfo.adapterId, path.targetInfo.id) : std::nullopt);
      }
      return result;
    }

    std::optional<detail::snapshot_t> query_snapshot() {
      return query_snapshot_flags(QDC_ONLY_ACTIVE_PATHS);
    }

    bool apply_color(const DISPLAYCONFIG_PATH_INFO &path, const display_config::advanced_color_state_t &color) {
      if (color.api == display_config::advanced_color_api_e::legacy) {
        return display_config::set_legacy_advanced_color_state(path.targetInfo.adapterId, path.targetInfo.id, color.advanced_color_enabled, {});
      }
      const auto current = display_config::query_advanced_color(path.targetInfo.adapterId, path.targetInfo.id);
      if (!current || current->api != color.api) {
        return false;
      }
      bool success = true;
      if (current->hdr_user_enabled != color.hdr_user_enabled) {
        success = display_config::set_hdr_state(path.targetInfo.adapterId, path.targetInfo.id, color.hdr_user_enabled) && success;
      }
      if (current->wcg_user_enabled != color.wcg_user_enabled) {
        success = display_config::set_wcg_state(path.targetInfo.adapterId, path.targetInfo.id, color.wcg_user_enabled) && success;
      }
      return success;
    }

    detail::load_result_t load_journal() {
      const auto path = journal_path();
      std::error_code error;
      if (!std::filesystem::exists(path, error)) {
        return {!error, std::nullopt};
      }
      const auto size = std::filesystem::file_size(path, error);
      if (error || size == 0 || size > max_journal_bytes) {
        return {};
      }
      std::ifstream input(path, std::ios::binary);
      std::string contents(static_cast<size_t>(size), '\0');
      if (!input.read(contents.data(), static_cast<std::streamsize>(contents.size())) || input.peek() != std::char_traits<char>::eof()) {
        return {};
      }
      auto journal = detail::deserialize(contents);
      return {journal.has_value(), std::move(journal)};
    }

    bool save_journal(const detail::journal_t &journal) {
      const auto contents = detail::serialize(journal);
      if (contents.empty() || contents.size() > max_journal_bytes) {
        return false;
      }
      const auto path = journal_path();
      auto temporary = path;
      temporary += L".tmp";
      const HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
      if (file == INVALID_HANDLE_VALUE) {
        return false;
      }
      DWORD written = 0;
      const bool durable = WriteFile(file, contents.data(), static_cast<DWORD>(contents.size()), &written, nullptr) &&
                           written == contents.size() && FlushFileBuffers(file);
      const bool closed = CloseHandle(file);
      if (!durable || !closed || !MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        DeleteFileW(temporary.c_str());
        return false;
      }
      return true;
    }

    bool clear_journal() {
      if (DeleteFileW(journal_path().c_str())) {
        return true;
      }
      return GetLastError() == ERROR_FILE_NOT_FOUND;
    }

    bool apply_snapshot(detail::snapshot_t snapshot) {
      // Never save the temporary primary/layout to the Windows topology database and never
      // permit CCD to choose replacement modes. Restore reconstructs inactive paths using
      // their saved modes and currently available source/target identities.
      // This Windows 11 host preserves virtual/physical refresh-rate distinctions in
      // both the query and apply directions, including BOOST_REFRESH_RATE/divider data.
      constexpr UINT32 flags = SDC_USE_SUPPLIED_DISPLAY_CONFIG | SDC_VIRTUAL_MODE_AWARE | SDC_VIRTUAL_REFRESH_RATE_AWARE;
      const auto validate = SetDisplayConfig(static_cast<UINT32>(snapshot.paths.size()), snapshot.paths.data(), static_cast<UINT32>(snapshot.modes.size()), snapshot.modes.data(), flags | SDC_VALIDATE);
      if (validate != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Temporary primary-display configuration is not valid: " << validate;
        return false;
      }
      const auto applied = SetDisplayConfig(static_cast<UINT32>(snapshot.paths.size()), snapshot.paths.data(), static_cast<UINT32>(snapshot.modes.size()), snapshot.modes.data(), flags | SDC_APPLY);
      if (applied != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Temporary primary-display configuration failed: " << applied;
      }
      return applied == ERROR_SUCCESS;
    }

    std::mutex transaction_mutex;
    detail::ownership_t process_ownership;

    bool acquire_ownership() {
      auto path = journal_path();
      path.replace_extension(L".lock");
      if (!process_ownership.acquire(path)) {
        BOOST_LOG(warning) << "Cannot acquire primary-display recovery ownership; another host may be using this configuration.";
        return false;
      }
      return true;
    }

    detail::manager_t manager() {
      return detail::manager_t({query_snapshot, apply_snapshot, load_journal, save_journal, clear_journal, [] {
                                  return query_snapshot_flags(QDC_ALL_PATHS);
                                },
                                apply_color});
    }
  }  // namespace

  namespace detail {
    ownership_t::~ownership_t() {
      if (handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(handle_);
      }
    }

    bool ownership_t::acquire(const std::filesystem::path &path) {
      std::error_code error;
      const auto absolute = std::filesystem::absolute(path, error);
      if (error) {
        return false;
      }
      const auto parent = std::filesystem::weakly_canonical(absolute.parent_path(), error);
      if (error) {
        return false;
      }
      const auto canonical = parent / absolute.filename();
      if (handle_ != INVALID_HANDLE_VALUE) {
        return same_device(path_.wstring(), canonical.wstring());
      }
      // File sharing is enforced across processes and is not tied to the calling thread.
      // Windows closes this handle after a crash; delete-on-close leaves no stale lock file.
      path_ = canonical;
      const auto handle = CreateFileW(canonical.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
      if (handle == INVALID_HANDLE_VALUE) {
        return false;
      }
      handle_ = handle;
      return true;
    }

    std::optional<layout_t> inspect(const snapshot_t &snapshot) {
      if (snapshot.paths.empty() || snapshot.paths.size() > max_displays || snapshot.modes.size() > max_displays * 3 || snapshot.paths.size() != snapshot.device_paths.size()) {
        return std::nullopt;
      }
      layout_t result;
      std::set<UINT32> source_indices;
      for (size_t i = 0; i < snapshot.paths.size(); ++i) {
        const auto &path = snapshot.paths[i];
        const auto index = source_index(path);
        const UINT32 target_index = path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE ? path.targetInfo.targetModeInfoIdx : path.targetInfo.modeInfoIdx;
        if (!(path.flags & DISPLAYCONFIG_PATH_ACTIVE) || !path.targetInfo.targetAvailable || index >= snapshot.modes.size() || target_index >= snapshot.modes.size() || !source_indices.insert(index).second) {
          return std::nullopt;
        }
        const auto &target_mode = snapshot.modes[target_index];
        if (target_mode.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_TARGET || target_mode.id != path.targetInfo.id || !same_adapter(target_mode.adapterId, path.targetInfo.adapterId)) {
          return std::nullopt;
        }
        if (path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE) {
          const auto desktop_index = path.targetInfo.desktopModeInfoIdx;
          if (desktop_index != DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID && (desktop_index >= snapshot.modes.size() || static_cast<int>(snapshot.modes[desktop_index].infoType) != 3 || snapshot.modes[desktop_index].id != path.targetInfo.id || !same_adapter(snapshot.modes[desktop_index].adapterId, path.targetInfo.adapterId))) {
            return std::nullopt;
          }
        }
        const auto &mode = snapshot.modes[index];
        if (mode.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE || mode.id != path.sourceInfo.id || !same_adapter(mode.adapterId, path.sourceInfo.adapterId) || mode.sourceMode.width == 0 || mode.sourceMode.height == 0 || static_cast<int64_t>(mode.sourceMode.position.x) + mode.sourceMode.width > std::numeric_limits<LONG>::max() || static_cast<int64_t>(mode.sourceMode.position.y) + mode.sourceMode.height > std::numeric_limits<LONG>::max()) {
          return std::nullopt;
        }
        for (size_t j = 0; j < i; ++j) {
          const auto &other = snapshot.paths[j];
          if ((same_adapter(path.sourceInfo.adapterId, other.sourceInfo.adapterId) && path.sourceInfo.id == other.sourceInfo.id) || (same_adapter(path.targetInfo.adapterId, other.targetInfo.adapterId) && path.targetInfo.id == other.targetInfo.id)) {
            return std::nullopt;
          }
        }
        result.push_back({snapshot.device_paths[i], mode.sourceMode.position.x, mode.sourceMode.position.y});
      }
      return valid_layout(result) ? std::make_optional(std::move(result)) : std::nullopt;
    }

    std::string serialize(const journal_t &journal) {
      if (!valid_journal(journal)) {
        return {};
      }
      auto encode = [](const layout_t &layout) {
        auto result = nlohmann::json::array();
        for (const auto &entry : layout) {
          result.push_back({{"device_path", platf::to_utf8(entry.device_path)}, {"x", entry.x}, {"y", entry.y}});
        }
        return result;
      };
      nlohmann::json result {{"version", journal.exclusive ? 2 : 1}, {"prepared", journal.prepared}, {"original_primary", platf::to_utf8(journal.original_primary)}, {"promoted_primary", platf::to_utf8(journal.promoted_primary)}, {"original", encode(journal.original)}, {"promoted", encode(journal.promoted)}};
      if (journal.exclusive) {
        result["exclusive"] = true;
        result["exclusive_started"] = journal.exclusive_started;
        result["original_topology"] = encode_snapshot(*journal.original_topology);
        result["before_exclusive"] = journal.before_exclusive ? encode_snapshot(*journal.before_exclusive) : nlohmann::json(nullptr);
        result["pending_restore"] = journal.pending_restore ? encode_snapshot(*journal.pending_restore) : nlohmann::json(nullptr);
      }
      return result.dump();
    }

    std::optional<journal_t> deserialize(std::string_view contents) {
      if (contents.size() > max_journal_bytes) {
        return std::nullopt;
      }
      try {
        const auto value = nlohmann::json::parse(contents);
        const int version = value.at("version").get<int>();
        if (version != 1 && version != 2) {
          return std::nullopt;
        }
        auto decode = [](const nlohmann::json &entries) {
          layout_t result;
          if (!entries.is_array() || entries.size() > max_displays) {
            throw std::runtime_error("invalid primary-display layout");
          }
          for (const auto &entry : entries) {
            auto coordinate = [&](const char *name) {
              const auto &number = entry.at(name);
              if (!number.is_number_integer()) {
                throw std::runtime_error("invalid display coordinate");
              }
              if (number.is_number_unsigned() && number.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<LONG>::max())) {
                throw std::runtime_error("display coordinate out of range");
              }
              const auto parsed = number.get<int64_t>();
              if (parsed < std::numeric_limits<LONG>::min() || parsed > std::numeric_limits<LONG>::max()) {
                throw std::runtime_error("display coordinate out of range");
              }
              return static_cast<LONG>(parsed);
            };
            result.push_back({platf::from_utf8(entry.at("device_path").get<std::string>()), coordinate("x"), coordinate("y")});
          }
          return result;
        };
        journal_t result {platf::from_utf8(value.at("original_primary").get<std::string>()), platf::from_utf8(value.at("promoted_primary").get<std::string>()), decode(value.at("original")), decode(value.at("promoted")), value.at("prepared").get<bool>()};
        if (version == 2) {
          result.exclusive = value.at("exclusive").get<bool>();
          result.exclusive_started = value.at("exclusive_started").get<bool>();
          result.original_topology = decode_snapshot(value.at("original_topology"));
          if (!value.at("before_exclusive").is_null()) {
            result.before_exclusive = decode_snapshot(value.at("before_exclusive"));
          }
          if (!value.at("pending_restore").is_null()) {
            result.pending_restore = decode_snapshot(value.at("pending_restore"));
          }
        }
        return valid_journal(result) ? std::make_optional(std::move(result)) : std::nullopt;
      } catch (const std::exception &) {
        return std::nullopt;
      }
    }

    manager_t::manager_t(io_t io):
        io_(std::move(io)) {}

    bool manager_t::prepare(bool exclusive) {
      const auto loaded = io_.load();
      if (!loaded.success || loaded.journal) {
        return false;
      }
      const auto snapshot = io_.query();
      if (!snapshot) {
        return false;
      }
      if (snapshot->paths.empty() && snapshot->device_paths.empty()) {
        // A headless machine has no original primary display to restore.
        return true;
      }
      const auto layout = inspect(*snapshot);
      if (!layout) {
        return false;
      }
      const auto primary = std::ranges::find_if(*layout, [](const auto &entry) {
        return entry.x == 0 && entry.y == 0;
      });
      journal_t journal {primary->device_path, {}, *layout, {}, true};
      if (exclusive) {
        journal.exclusive = true;
        journal.original_topology = *snapshot;
      }
      return valid_journal(journal) && io_.save(journal);
    }

    bool manager_t::bind_pending(std::wstring_view device_path) {
      const auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
        return false;
      }
      if (!loaded.journal) {
        return true;
      }
      const auto &prepared = *loaded.journal;
      if (!prepared.prepared) {
        return same_device(prepared.promoted_primary, device_path);
      }
      const auto snapshot = io_.query();
      const auto current = snapshot ? inspect(*snapshot) : std::nullopt;
      if (!current) {
        return false;
      }
      const auto *target = find_position(*current, device_path);
      const auto *original_primary = find_position(*current, prepared.original_primary);
      const bool new_target = !find_position(prepared.original, device_path);
      if (prepared.exclusive && target && (!original_primary || current->size() != prepared.original.size() + (new_target ? 1 : 0))) {
        // Windows can restore a previously remembered virtual-only topology inside Add.
        // The exact Add identity is known here; preserve the pre-Add physical snapshot.
        for (const auto &entry : *current) {
          if (!same_device(entry.device_path, device_path) && !find_position(prepared.original, entry.device_path)) {
            return false;
          }
        }
        auto original = prepared.original;
        if (new_target) {
          int64_t right = std::numeric_limits<LONG>::min();
          LONG top = 0;
          for (const auto &path : prepared.original_topology->paths) {
            const auto &source = prepared.original_topology->modes[source_index(path)].sourceMode;
            const int64_t edge = static_cast<int64_t>(source.position.x) + source.width;
            if (edge > right) {
              right = edge;
              top = source.position.y;
            }
          }
          if (right > std::numeric_limits<LONG>::max()) {
            return false;
          }
          original.push_back({std::wstring(device_path), static_cast<LONG>(right), top});
        }
        const auto *baseline_target = find_position(original, device_path);
        const auto promoted = translated(original, baseline_target->x, baseline_target->y);
        if (!promoted) {
          return false;
        }
        auto bound = prepared;
        bound.prepared = false;
        bound.promoted_primary = std::wstring(device_path);
        bound.original = std::move(original);
        bound.promoted = *promoted;
        bound.exclusive_started = true;
        bound.before_exclusive = *snapshot;
        return valid_journal(bound) && io_.save(bound);
      }
      if (!target || !original_primary || current->size() != prepared.original.size() + (new_target ? 1 : 0)) {
        return false;
      }
      const auto original = translated(*current, original_primary->x, original_primary->y);
      if (!original || !std::ranges::all_of(prepared.original, [&](const auto &saved) {
            const auto *observed = find_position(*original, saved.device_path);
            return observed && observed->x == saved.x && observed->y == saved.y;
          })) {
        return false;
      }
      if (same_device(prepared.original_primary, device_path) && !prepared.exclusive) {
        return io_.clear();
      }
      const auto promoted = translated(*current, target->x, target->y);
      if (!promoted) {
        return false;
      }
      auto bound = prepared;
      bound.prepared = false;
      bound.promoted_primary = target->device_path;
      bound.original = *original;
      bound.promoted = *promoted;
      if (bound.exclusive && bound.pending_restore) {
        bound.exclusive_started = true;
        bound.before_exclusive = *snapshot;
      }
      return valid_journal(bound) && io_.save(bound);
    }

    bool manager_t::apply_verified(const snapshot_t &before, const layout_t &desired) {
      const auto baseline = inspect(before);
      auto latest = io_.query();
      const auto current = latest ? inspect(*latest) : std::nullopt;
      if (!baseline || !current || !same_layout(*baseline, *current) || !same_modes(before, *latest)) {
        BOOST_LOG(warning) << "Display configuration changed during primary-display transaction; retaining recovery record.";
        return false;
      }
      for (size_t i = 0; i < latest->paths.size(); ++i) {
        const auto *position = find_position(desired, latest->device_paths[i]);
        if (!position) {
          return false;
        }
        latest->modes[source_index(latest->paths[i])].sourceMode.position = {position->x, position->y};
      }
      if (!inspect(*latest) || !io_.apply(*latest)) {
        return false;
      }
      const auto observed = io_.query();
      const auto layout = observed ? inspect(*observed) : std::nullopt;
      if (!layout || !same_layout(*layout, desired) || !same_modes(*latest, *observed)) {
        BOOST_LOG(warning) << "Could not verify primary-display change; retaining recovery record.";
        return false;
      }
      return true;
    }

    bool manager_t::promote_exclusive(std::wstring_view device_path) {
      auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && (!valid_journal(*loaded.journal) || !loaded.journal->exclusive))) {
        return false;
      }
      auto snapshot = io_.query();
      auto layout = snapshot ? inspect(*snapshot) : std::nullopt;
      if (!layout || !find_position(*layout, device_path)) {
        return false;
      }
      if (!loaded.journal) {
        if (snapshot->paths.size() == 1) {
          return true;
        }
        // Warm reconnect has an active retained virtual display but no outstanding journal.
        if (!prepare(true) || !bind_pending(device_path)) {
          return false;
        }
        loaded = io_.load();
      } else if (loaded.journal->prepared) {
        if (!bind_pending(device_path)) {
          return false;
        }
        loaded = io_.load();
      }
      if (!loaded.success || !loaded.journal || !valid_journal(*loaded.journal)) {
        return false;
      }
      auto journal = *loaded.journal;
      if (!same_device(journal.promoted_primary, device_path)) {
        return false;
      }
      if (journal.exclusive_started && snapshot->paths.size() == 1) {
        return true;
      }
      const bool remembered_exclusive = journal.exclusive_started && !journal.pending_restore && same_topology(*snapshot, *journal.before_exclusive);
      if (journal.exclusive_started && !remembered_exclusive) {
        return false;
      }
      if (!remembered_exclusive && !same_layout(*layout, journal.original) && !same_layout(*layout, journal.promoted)) {
        return false;
      }
      snapshot_t current_physical, saved_physical;
      current_physical.modes = snapshot->modes;
      saved_physical.modes = journal.original_topology->modes;
      for (size_t i = 0; i < journal.original_topology->paths.size(); ++i) {
        if (same_device(journal.original_topology->device_paths[i], device_path)) {
          continue;
        }
        const auto current_index = find_path(*snapshot, journal.original_topology->device_paths[i]);
        if (remembered_exclusive && !current_index) {
          continue;
        }
        if (!current_index || *current_index >= snapshot->colors.size() || !snapshot->colors[*current_index] || (!remembered_exclusive && !color_matches(*snapshot->colors[*current_index], *journal.original_topology->colors[i]))) {
          return false;
        }
        current_physical.paths.push_back(snapshot->paths[*current_index]);
        current_physical.device_paths.push_back(snapshot->device_paths[*current_index]);
        saved_physical.paths.push_back(journal.original_topology->paths[i]);
        saved_physical.device_paths.push_back(journal.original_topology->device_paths[i]);
      }
      if (saved_physical.paths.empty() && !remembered_exclusive) {
        return io_.clear();
      }
      if (!remembered_exclusive && !same_modes(current_physical, saved_physical)) {
        return false;
      }
      auto virtual_spec = display_spec(*snapshot, *find_path(*snapshot, device_path));
      virtual_spec.source.position = {0, 0};
      const auto desired = build_topology(*snapshot, {virtual_spec});
      if (!desired) {
        return false;
      }
      journal.exclusive_started = true;
      journal.before_exclusive = *snapshot;
      if (!io_.save(journal)) {
        return false;
      }
      const auto latest = io_.query();
      if (!latest || !same_topology(*latest, *snapshot)) {
        return false;
      }
      if (!io_.apply(*desired)) {
        return false;
      }
      auto observed = io_.query();
      if (!observed || !same_topology(*observed, *desired)) {
        return false;
      }
      if (virtual_spec.color) {
        if (observed->colors.empty() || !observed->colors[0]) {
          return false;
        }
        if (!color_matches(*observed->colors[0], *virtual_spec.color)) {
          if (!io_.set_color || !io_.set_color(observed->paths[0], *virtual_spec.color)) {
            return false;
          }
          observed = io_.query();
          if (!observed || !same_topology(*observed, *desired) || observed->colors.empty() || !observed->colors[0] || !color_matches(*observed->colors[0], *virtual_spec.color)) {
            return false;
          }
        }
      }
      BOOST_LOG(info) << "Virtual display is temporarily the only active Windows display.";
      return true;
    }

    bool manager_t::restore_exclusive(journal_t journal) {
      if (!io_.query_all || !io_.set_color) {
        return false;
      }
      const auto current = io_.query();
      if (!current || (!current->paths.empty() && !inspect(*current))) {
        return false;
      }
      const auto virtual_index = find_path(*current, journal.promoted_primary);
      const bool only_virtual = current->paths.size() == 1 && virtual_index.has_value();
      const bool owned = current->paths.empty() || only_virtual || same_topology(*current, *journal.before_exclusive) ||
                         (journal.pending_restore && owned_subset(*current, *journal.pending_restore)) ||
                         (!virtual_index && owned_subset(*current, *journal.original_topology));
      const auto available = io_.query_all();
      if (!available || available->paths.size() > max_available_paths || available->paths.size() != available->device_paths.size()) {
        return false;
      }
      auto available_identity = [&](std::wstring_view identity) {
        std::optional<std::pair<LUID, UINT32>> target;
        for (size_t i = 0; i < available->paths.size(); ++i) {
          const auto &path = available->paths[i];
          if (!path.targetInfo.targetAvailable || !same_device(identity, available->device_paths[i])) {
            continue;
          }
          if (target && (!same_adapter(target->first, path.targetInfo.adapterId) || target->second != path.targetInfo.id)) {
            return false;
          }
          target = std::pair {path.targetInfo.adapterId, path.targetInfo.id};
        }
        return target.has_value();
      };
      std::vector<display_spec_t> wanted;
      bool missing = false;
      size_t physical_count = 0;
      for (size_t i = 0; i < journal.original_topology->paths.size(); ++i) {
        const auto &identity = journal.original_topology->device_paths[i];
        if (same_device(identity, journal.promoted_primary)) {
          continue;
        }
        ++physical_count;
        if (!available_identity(identity)) {
          missing = true;
          continue;
        }
        wanted.push_back(display_spec(*journal.original_topology, i));
      }
      if (physical_count == 0) {
        return io_.clear();
      }
      if (wanted.empty()) {
        return false;
      }
      auto primary = std::ranges::find_if(wanted, [&](const auto &spec) {
        return same_device(spec.identity, journal.original_primary);
      });
      if (primary == wanted.end()) {
        primary = wanted.begin();
      }
      const POINTL origin = primary->source.position;
      std::rotate(wanted.begin(), primary, primary + 1);
      if (virtual_index) {
        if (!available_identity(journal.promoted_primary)) {
          return false;
        }
        auto virtual_spec = display_spec(*current, *virtual_index);
        if (journal.pending_restore) {
          const auto pending_index = find_path(*journal.pending_restore, journal.promoted_primary);
          if (pending_index && *pending_index < journal.pending_restore->colors.size() && journal.pending_restore->colors[*pending_index]) {
            // A prior CCD apply may have reset HDR before its color setter failed. Keep
            // that durable color intent across retries while preserving the live VD mode.
            virtual_spec.color = journal.pending_restore->colors[*pending_index];
          }
        }
        const auto *original = find_position(journal.original, journal.promoted_primary);
        virtual_spec.source.position = {original->x, original->y};
        const auto overlaps = [&](const display_spec_t &physical) {
          const auto &v = virtual_spec.source;
          const auto &p = physical.source;
          return static_cast<int64_t>(v.position.x) < static_cast<int64_t>(p.position.x) + p.width &&
                 static_cast<int64_t>(p.position.x) < static_cast<int64_t>(v.position.x) + v.width &&
                 static_cast<int64_t>(v.position.y) < static_cast<int64_t>(p.position.y) + p.height &&
                 static_cast<int64_t>(p.position.y) < static_cast<int64_t>(v.position.y) + v.height;
        };
        if (std::ranges::any_of(wanted, overlaps)) {
          int64_t right = std::numeric_limits<LONG>::min();
          LONG top = 0;
          for (const auto &physical : wanted) {
            const int64_t edge = static_cast<int64_t>(physical.source.position.x) + physical.source.width;
            if (edge > right) {
              right = edge;
              top = physical.source.position.y;
            }
          }
          if (right > std::numeric_limits<LONG>::max()) {
            return false;
          }
          virtual_spec.source.position = {static_cast<LONG>(right), top};
        }
        // The virtual mode may legitimately change during the session. Preserve its current
        // mode and color; it must stay active for orderly retirement and warm reconnect.
        wanted.push_back(std::move(virtual_spec));
      }
      for (auto &spec : wanted) {
        const auto x = subtract(spec.source.position.x, origin.x);
        const auto y = subtract(spec.source.position.y, origin.y);
        if (!x || !y) {
          return false;
        }
        spec.source.position = {*x, *y};
      }
      const auto desired = build_topology(*available, wanted);
      if (!desired) {
        return false;
      }
      // A formerly unavailable saved physical display can return already active. Its
      // identity, original mode and relative arrangement must match the restoration plan;
      // an unrelated active output or a user-edited mode/layout still fails this check.
      if (!owned && !(journal.pending_restore && owned_subset(*current, *desired))) {
        BOOST_LOG(warning) << "Display topology was changed outside the exclusive-display transaction; preserving that arrangement and the recovery journal.";
        return false;
      }
      journal.pending_restore = *desired;
      if (!io_.save(journal)) {
        return false;
      }
      const auto latest = io_.query();
      if (!latest || !same_topology(*current, *latest)) {
        return false;
      }
      if (!same_topology(*current, *desired) && !io_.apply(*desired)) {
        return false;
      }
      auto observed = io_.query();
      if (!observed || !same_topology(*observed, *desired)) {
        return false;
      }
      bool color_ok = true;
      for (const auto &spec : wanted) {
        if (!spec.color) {
          continue;
        }
        const auto index = find_path(*observed, spec.identity);
        if (!index || *index >= observed->colors.size() || !observed->colors[*index]) {
          color_ok = false;
          continue;
        }
        if (!color_matches(*observed->colors[*index], *spec.color)) {
          color_ok = io_.set_color(observed->paths[*index], *spec.color) && color_ok;
        }
      }
      observed = io_.query();
      if (!observed || !same_topology(*observed, *desired)) {
        return false;
      }
      for (const auto &spec : wanted) {
        if (!spec.color) {
          continue;
        }
        const auto index = find_path(*observed, spec.identity);
        color_ok = index && *index < observed->colors.size() && observed->colors[*index] &&
                   color_matches(*observed->colors[*index], *spec.color) && color_ok;
      }
      if (!color_ok || missing) {
        BOOST_LOG(warning) << "Available physical displays are active; retaining exclusive-display recovery until all original displays and color settings are restored.";
        return false;
      }
      BOOST_LOG(info) << "Restored physical displays, modes, color settings, and primary display after exclusive streaming.";
      return io_.clear();
    }

    bool manager_t::recover_prepared_outputs(journal_t journal) {
      if (!io_.query_all || !io_.set_color) {
        return false;
      }
      const auto current = io_.query();
      if (!current || (!current->paths.empty() && !inspect(*current))) {
        return false;
      }
      std::vector<display_spec_t> extras;
      size_t active_originals = 0;
      for (size_t i = 0; i < current->paths.size(); ++i) {
        if (find_path(*journal.original_topology, current->device_paths[i])) {
          ++active_originals;
        } else {
          extras.push_back(display_spec(*current, i));
        }
      }
      const bool prior_recovery = journal.pending_restore && owned_subset(*current, *journal.pending_restore);
      if (active_originals && !prior_recovery && (!extras.empty() || !owned_subset(*current, *journal.original_topology))) {
        return false;
      }
      const auto available = io_.query_all();
      if (!available || available->paths.size() > max_available_paths || available->paths.size() != available->device_paths.size()) {
        return false;
      }
      std::vector<display_spec_t> physical;
      bool missing = false;
      for (size_t i = 0; i < journal.original_topology->paths.size(); ++i) {
        const auto &identity = journal.original_topology->device_paths[i];
        bool found = false;
        for (size_t j = 0; j < available->paths.size(); ++j) {
          found = found || (available->paths[j].targetInfo.targetAvailable && same_device(available->device_paths[j], identity));
        }
        if (found) {
          physical.push_back(display_spec(*journal.original_topology, i));
        } else {
          missing = true;
        }
      }
      if (physical.empty()) {
        return false;
      }
      int64_t offset_x = 0, offset_y = 0;
      if (!extras.empty()) {
        // An unbound output is never identified as ours or moved/disabled. Re-enable only
        // the known physical block alongside it, leaving its primary, modes and positions
        // alone until the driver's orphan cleanup or exact Add identity resolves ownership.
        const auto rightmost = std::ranges::max_element(extras, [](const auto &a, const auto &b) {
          return static_cast<int64_t>(a.source.position.x) + a.source.width < static_cast<int64_t>(b.source.position.x) + b.source.width;
        });
        const auto leftmost = std::ranges::min_element(physical, {}, [](const auto &spec) {
          return spec.source.position.x;
        });
        offset_x = static_cast<int64_t>(rightmost->source.position.x) + rightmost->source.width - leftmost->source.position.x;
        offset_y = static_cast<int64_t>(rightmost->source.position.y) - leftmost->source.position.y;
      } else {
        auto primary = std::ranges::find_if(physical, [&](const auto &spec) {
          return same_device(spec.identity, journal.original_primary);
        });
        if (primary == physical.end()) {
          primary = physical.begin();
        }
        offset_x = -static_cast<int64_t>(primary->source.position.x);
        offset_y = -static_cast<int64_t>(primary->source.position.y);
      }
      for (auto &spec : physical) {
        const int64_t x = spec.source.position.x + offset_x;
        const int64_t y = spec.source.position.y + offset_y;
        if (x < std::numeric_limits<LONG>::min() || x > std::numeric_limits<LONG>::max() || y < std::numeric_limits<LONG>::min() || y > std::numeric_limits<LONG>::max()) {
          return false;
        }
        spec.source.position = {static_cast<LONG>(x), static_cast<LONG>(y)};
      }
      auto wanted = extras;
      wanted.insert(wanted.end(), physical.begin(), physical.end());
      const auto desired = build_topology(*available, wanted);
      if (!desired) {
        return false;
      }
      journal.pending_restore = *desired;
      if (!io_.save(journal)) {
        return false;
      }
      const auto latest = io_.query();
      if (!latest || !same_topology(*latest, *current)) {
        return false;
      }
      if (!same_topology(*current, *desired) && !io_.apply(*desired)) {
        return false;
      }
      auto observed = io_.query();
      if (!observed || !same_topology(*observed, *desired)) {
        return false;
      }
      bool colors_ok = true;
      for (const auto &spec : physical) {
        const auto index = find_path(*observed, spec.identity);
        if (!index || *index >= observed->colors.size() || !observed->colors[*index] || !spec.color) {
          colors_ok = false;
          continue;
        }
        if (!color_matches(*observed->colors[*index], *spec.color)) {
          colors_ok = io_.set_color(observed->paths[*index], *spec.color) && colors_ok;
        }
      }
      observed = io_.query();
      if (!observed || !same_topology(*observed, *desired)) {
        return false;
      }
      for (const auto &spec : physical) {
        const auto index = find_path(*observed, spec.identity);
        colors_ok = index && *index < observed->colors.size() && observed->colors[*index] && spec.color && color_matches(*observed->colors[*index], *spec.color) && colors_ok;
      }
      if (missing || !extras.empty() || !colors_ok) {
        return false;
      }
      return io_.clear();
    }

    bool manager_t::promote(std::wstring_view device_path, bool exclusive) {
      if (exclusive) {
        return promote_exclusive(device_path);
      }
      auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
        return false;
      }
      if (loaded.journal && loaded.journal->exclusive) {
        return false;
      }
      if (loaded.journal && loaded.journal->prepared) {
        if (!bind_pending(device_path)) {
          return false;
        }
        loaded = io_.load();
        if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
          return false;
        }
      }
      auto snapshot = io_.query();
      auto layout = snapshot ? inspect(*snapshot) : std::nullopt;
      if (!layout) {
        return false;
      }
      if (loaded.journal) {
        // Keep the first baseline on repeated promotion (including a resized display).
        if (!same_device(loaded.journal->promoted_primary, device_path)) {
          return false;
        }
        if (same_layout(*layout, loaded.journal->promoted)) {
          return true;
        }
        if (!same_layout(*layout, loaded.journal->original)) {
          return false;
        }
        // A prepared baseline has already been bound. Keep that durable first baseline
        // instead of clearing it and taking a later snapshot after mode changes.
        return apply_verified(*snapshot, loaded.journal->promoted);
      }
      const auto *target = find_position(*layout, device_path);
      if (!target) {
        return false;
      }
      if (target->x == 0 && target->y == 0) {
        return true;
      }
      const auto original_primary = std::ranges::find_if(*layout, [](const auto &entry) {
        return entry.x == 0 && entry.y == 0;
      });
      auto desired = translated(*layout, target->x, target->y);
      if (!desired) {
        return false;
      }
      const journal_t journal {original_primary->device_path, target->device_path, *layout, *desired};
      if (!io_.save(journal)) {
        BOOST_LOG(warning) << "Could not durably save primary-display recovery record; display configuration was not changed.";
        return false;
      }
      if (!apply_verified(*snapshot, *desired)) {
        return false;
      }
      BOOST_LOG(info) << "Virtual display is temporarily the Windows primary display.";
      return true;
    }

    bool manager_t::restore(std::wstring_view expected_device_path) {
      const auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
        BOOST_LOG(warning) << "Could not read primary-display recovery record; leaving displays unchanged.";
        return false;
      }
      if (!loaded.journal) {
        return true;
      }
      const auto &journal = *loaded.journal;
      if (journal.exclusive_started) {
        if (!expected_device_path.empty() && !same_device(expected_device_path, journal.promoted_primary)) {
          return false;
        }
        return restore_exclusive(journal);
      }
      if (journal.prepared) {
        const auto snapshot = io_.query();
        const auto current = snapshot ? inspect(*snapshot) : std::nullopt;
        if (!expected_device_path.empty() && current && find_position(*current, expected_device_path)) {
          return bind_pending(expected_device_path) && restore(expected_device_path);
        }
        if (journal.exclusive) {
          return recover_prepared_outputs(journal);
        }
        const auto *primary = current ? find_position(*current, journal.original_primary) : nullptr;
        if (!primary || current->size() != journal.original.size()) {
          return journal.exclusive && recover_prepared_outputs(journal);
        }
        const auto rebased = translated(*current, primary->x, primary->y);
        if (!rebased || !same_layout(*rebased, journal.original)) {
          return journal.exclusive && recover_prepared_outputs(journal);
        }
        // If a crash occurred before binding the target, wait for the driver's orphan
        // cleanup. Only the exact known physical set may be restored without that identity.
        if (!same_layout(*current, journal.original) && !apply_verified(*snapshot, journal.original)) {
          return false;
        }
        return io_.clear();
      }
      if (!expected_device_path.empty() && !same_device(expected_device_path, journal.promoted_primary)) {
        return false;
      }
      const auto snapshot = io_.query();
      const auto layout = snapshot ? inspect(*snapshot) : std::nullopt;
      if (!layout) {
        return false;
      }
      const auto original = restrict_layout(journal.original, *layout, journal.promoted_primary);
      const auto promoted = restrict_layout(journal.promoted, *layout, journal.promoted_primary);
      if (!original || !promoted) {
        BOOST_LOG(warning) << "Monitors changed since primary-display promotion; retaining recovery record until the original displays are available.";
        return false;
      }
      if (same_layout(*layout, *original)) {
        return io_.clear();
      }
      if (!find_position(*layout, journal.promoted_primary)) {
        const auto *primary = find_position(*layout, journal.original_primary);
        const auto rebased = primary ? translated(*layout, primary->x, primary->y) : std::nullopt;
        if (rebased && same_layout(*rebased, *original)) {
          return apply_verified(*snapshot, *original) && io_.clear();
        }
      }
      if (!same_layout(*layout, *promoted)) {
        BOOST_LOG(warning) << "Monitor positions were changed outside the primary-display transaction; leaving that arrangement unchanged and retaining recovery record.";
        return false;
      }
      if (!apply_verified(*snapshot, *original)) {
        return false;
      }
      if (!io_.clear()) {
        return false;
      }
      BOOST_LOG(info) << "Restored the original Windows primary display and monitor positions.";
      return true;
    }
  }  // namespace detail

  bool promote(std::wstring_view device_path, bool exclusive) {
    std::lock_guard lock(transaction_mutex);
    try {
      return acquire_ownership() && manager().promote(device_path, exclusive);
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Temporary primary-display promotion failed: " << exception.what();
      return false;
    }
  }

  bool prepare(bool exclusive) {
    std::lock_guard lock(transaction_mutex);
    try {
      return acquire_ownership() && manager().prepare(exclusive);
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not prepare primary-display recovery: " << exception.what();
      return false;
    }
  }

  bool bind_pending(std::wstring_view device_path) {
    std::lock_guard lock(transaction_mutex);
    try {
      return acquire_ownership() && manager().bind_pending(device_path);
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not bind primary-display recovery identity: " << exception.what();
      return false;
    }
  }

  bool restore(std::wstring_view expected_device_path) {
    std::lock_guard lock(transaction_mutex);
    try {
      return acquire_ownership() && manager().restore(expected_device_path);
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Temporary primary-display recovery failed: " << exception.what();
      return false;
    }
  }

  bool recover() {
    return restore();
  }
}  // namespace platf::primary_display
