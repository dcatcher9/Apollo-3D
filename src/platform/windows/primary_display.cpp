/**
 * @file src/platform/windows/primary_display.cpp
 * @brief Temporary primary-display transactions with crash recovery.
 */
#include "primary_display.h"

#include "ar_glasses.h"
#include "display_config.h"
#include "misc.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/utility.h"

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
        const bool headless = journal.original_topology && journal.original_topology->paths.empty() &&
                              journal.original_topology->device_paths.empty() && journal.original_topology->modes.empty() &&
                              journal.original_topology->colors.empty() && journal.original.size() == 1 &&
                              same_device(journal.original_primary, journal.promoted_primary) && !journal.prepared && journal.exclusive_started;
        if (!journal.original_topology || (!headless && !detail::inspect(*journal.original_topology)) || journal.original_topology->colors.size() != journal.original_topology->paths.size() || !std::ranges::all_of(journal.original_topology->colors, [](const auto &color) {
              return color.has_value();
            })) {
          return false;
        }
        const auto saved = headless ? std::optional<detail::layout_t> {detail::layout_t {}} : detail::inspect(*journal.original_topology);
        for (const auto &entry : *saved) {
          const auto *expected = find_position(journal.original, entry.device_path);
          if (!expected || entry.x != expected->x || entry.y != expected->y) {
            return false;
          }
        }
        for (size_t i = 0; i < journal.exclusive_preserved.size(); ++i) {
          const auto &identity = journal.exclusive_preserved[i];
          bool duplicate = false;
          for (size_t j = 0; j < i; ++j) {
            duplicate = duplicate || same_device(identity, journal.exclusive_preserved[j]);
          }
          const bool recorded_hotplug = journal.pending_restore &&
                                         std::ranges::any_of(journal.pending_restore->device_paths, [&](const auto &pending) {
                                           return same_device(identity, pending);
                                         });
          if (identity.empty() || (!find_position(*saved, identity) && !recorded_hotplug) || duplicate) {
            return false;
          }
        }
        if (!journal.local_sink.empty() && (!find_position(*saved, journal.local_sink) || journal.exclusive_preserved.size() != 1 || !same_device(journal.exclusive_preserved.front(), journal.local_sink) || same_device(journal.local_sink, journal.promoted_primary))) {
          return false;
        }
        const bool empty_before = headless && journal.before_exclusive && journal.before_exclusive->paths.empty() &&
                                  journal.before_exclusive->device_paths.empty() && journal.before_exclusive->modes.empty() &&
                                  journal.before_exclusive->colors.empty();
        if (journal.exclusive_started && (!journal.before_exclusive || (!empty_before && !detail::inspect(*journal.before_exclusive)) || journal.prepared)) {
          return false;
        }
        if ((!journal.exclusive_started && journal.exclusive_topology) ||
            (journal.exclusive_topology && (!detail::inspect(*journal.exclusive_topology) ||
                                            journal.exclusive_topology->colors.size() != journal.exclusive_topology->paths.size()))) {
          return false;
        }
        if (journal.exclusive_topology) {
          const auto exclusive_layout = detail::inspect(*journal.exclusive_topology);
          const auto *exclusive_primary = find_position(*exclusive_layout, journal.promoted_primary);
          if (!exclusive_primary || exclusive_primary->x != 0 || exclusive_primary->y != 0 ||
              std::ranges::any_of(*exclusive_layout, [&](const auto &entry) {
                if (same_device(entry.device_path, journal.promoted_primary) ||
                    std::ranges::any_of(journal.exclusive_preserved, [&](const auto &preserved) {
                      return same_device(entry.device_path, preserved);
                    })) {
                  return false;
                }
                // A display absent from the launch baseline may join the owned transaction only
                // after restore has durably recorded its approved hotplug identity.
                return !journal.pending_restore || find_position(*saved, entry.device_path) ||
                       std::ranges::none_of(journal.pending_restore->device_paths, [&](const auto &pending) {
                         return same_device(entry.device_path, pending);
                       });
              })) {
            return false;
          }
        }
        if (journal.pending_restore && ((!journal.exclusive_started && !journal.prepared) || !detail::inspect(*journal.pending_restore))) {
          return false;
        }
      } else if (journal.exclusive_started || journal.original_topology || journal.before_exclusive || journal.pending_restore || !journal.exclusive_preserved.empty() || journal.exclusive_topology || !journal.local_sink.empty()) {
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

    // Reactivation owns the recorded VD even if a negotiated mode change follows it. Both
    // promotion and rollback require every identity/position and every physical mode unchanged.
    bool same_layout_and_physical_modes(const detail::snapshot_t &current, const detail::snapshot_t &reference, std::wstring_view virtual_identity) {
      const auto live_layout = detail::inspect(current);
      const auto pending_layout = detail::inspect(reference);
      if (!live_layout || !pending_layout || !same_layout(*live_layout, *pending_layout) || !find_path(current, virtual_identity)) {
        return false;
      }
      auto physical_only = [&](detail::snapshot_t snapshot) {
        const auto index = *find_path(snapshot, virtual_identity);
        snapshot.paths.erase(snapshot.paths.begin() + static_cast<std::ptrdiff_t>(index));
        snapshot.device_paths.erase(snapshot.device_paths.begin() + static_cast<std::ptrdiff_t>(index));
        return snapshot;
      };
      return same_modes(physical_only(current), physical_only(reference));
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

    bool pack_horizontally(
      std::vector<display_spec_t> &displays,
      std::wstring_view anchor_identity,
      LONG anchor_x,
      LONG anchor_y
    ) {
      if (displays.empty()) {
        return false;
      }
      std::ranges::sort(displays, [](const auto &left, const auto &right) {
        if (left.source.position.x != right.source.position.x) {
          return left.source.position.x < right.source.position.x;
        }
        if (left.source.position.y != right.source.position.y) {
          return left.source.position.y < right.source.position.y;
        }
        return CompareStringOrdinal(left.identity.c_str(), -1, right.identity.c_str(), -1, TRUE) == CSTR_LESS_THAN;
      });
      const auto anchor = std::ranges::find_if(displays, [&](const auto &spec) {
        return same_device(spec.identity, anchor_identity);
      });
      if (anchor == displays.end()) {
        return false;
      }

      std::int64_t left_width = 0;
      for (auto it = displays.begin(); it != anchor; ++it) {
        left_width += it->source.width;
        if (left_width > std::numeric_limits<LONG>::max()) {
          return false;
        }
      }
      std::int64_t x = static_cast<std::int64_t>(anchor_x) - left_width;
      for (auto &spec : displays) {
        if (x < std::numeric_limits<LONG>::min() || x > std::numeric_limits<LONG>::max()) {
          return false;
        }
        spec.source.position = {static_cast<LONG>(x), anchor_y};
        x += spec.source.width;
      }
      return x <= std::numeric_limits<LONG>::max();
    }

    // Preserve the active physical arrangement, including user-selected modes/color. Append
    // outputs that Sunshine disabled beside it, instead of overwriting the user's new layout.
    bool preserve_current_desktop(
      const detail::snapshot_t &current,
      std::vector<display_spec_t> &wanted,
      std::wstring_view virtual_identity,
      bool keep_virtual_active
    ) {
      std::vector<display_spec_t> preserved;
      for (size_t i = 0; i < current.paths.size(); ++i) {
        if (!same_device(current.device_paths[i], virtual_identity)) {
          preserved.push_back(display_spec(current, i));
        }
      }
      if (preserved.empty()) {
        return false;
      }
      auto primary = std::ranges::find_if(preserved, [](const auto &spec) {
        return spec.source.position.x == 0 && spec.source.position.y == 0;
      });
      if (primary == preserved.end()) {
        primary = preserved.begin();
      }
      const auto origin = primary->source.position;
      for (auto &spec : preserved) {
        const auto x = subtract(spec.source.position.x, origin.x);
        const auto y = subtract(spec.source.position.y, origin.y);
        if (!x || !y) {
          return false;
        }
        spec.source.position = {*x, *y};
      }
      for (auto spec : wanted) {
        if ((!keep_virtual_active && same_device(spec.identity, virtual_identity)) || std::ranges::any_of(preserved, [&](const auto &entry) {
              return same_device(entry.identity, spec.identity);
            })) {
          continue;
        }
        const auto rightmost = std::ranges::max_element(preserved, {}, [](const auto &entry) {
          return static_cast<std::int64_t>(entry.source.position.x) + entry.source.width;
        });
        const auto right = static_cast<std::int64_t>(rightmost->source.position.x) + rightmost->source.width;
        if (right > std::numeric_limits<LONG>::max() || right + spec.source.width > std::numeric_limits<LONG>::max()) {
          return false;
        }
        spec.source.position = {static_cast<LONG>(right), rightmost->source.position.y};
        preserved.push_back(std::move(spec));
      }
      wanted = std::move(preserved);
      return true;
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

    // A verified user arrangement becomes the existing recovery baseline, not another journal
    // mode. Keep the exact pending apply separately so color failures and new hotplugs can retry
    // without rediscovering the superseded launch positions. Missing outputs start fresh later.
    bool rebase_recovery(detail::journal_t &journal, const detail::snapshot_t &current, const detail::snapshot_t &desired) {
      auto layout = detail::inspect(desired);
      if (!layout) {
        return false;
      }
      auto physical = desired;
      if (const auto index = find_path(physical, journal.promoted_primary)) {
        physical.paths.erase(physical.paths.begin() + static_cast<std::ptrdiff_t>(*index));
        physical.device_paths.erase(physical.device_paths.begin() + static_cast<std::ptrdiff_t>(*index));
        physical.colors.erase(physical.colors.begin() + static_cast<std::ptrdiff_t>(*index));
      }
      if (!detail::inspect(physical)) {
        return false;
      }
      if (!find_position(*layout, journal.promoted_primary)) {
        // A pause detaches the VD, but the existing journal still records its identity and a
        // secondary position for a retry that keeps it active. This does not activate anything.
        std::int64_t right = std::numeric_limits<LONG>::min();
        LONG top = 0;
        for (size_t i = 0; i < physical.paths.size(); ++i) {
          const auto spec = display_spec(physical, i);
          const auto edge = static_cast<std::int64_t>(spec.source.position.x) + spec.source.width;
          if (edge > right) {
            right = edge;
            top = spec.source.position.y;
          }
        }
        if (right > std::numeric_limits<LONG>::max()) {
          return false;
        }
        layout->push_back({journal.promoted_primary, static_cast<LONG>(right), top});
      }
      const auto *virtual_position = find_position(*layout, journal.promoted_primary);
      const auto promoted = translated(*layout, virtual_position->x, virtual_position->y);
      const auto primary = std::ranges::find_if(*layout, [](const auto &position) {
        return position.x == 0 && position.y == 0;
      });
      if (!promoted || primary == layout->end()) {
        return false;
      }
      journal.original_primary = primary->device_path;
      journal.original = std::move(*layout);
      journal.promoted = *promoted;
      journal.original_topology = std::move(physical);
      journal.before_exclusive = current;
      journal.exclusive_topology.reset();
      std::erase_if(journal.exclusive_preserved, [&](const auto &identity) {
        return !find_path(*journal.original_topology, identity);
      });
      if (!journal.local_sink.empty() && !find_path(*journal.original_topology, journal.local_sink)) {
        journal.local_sink.clear();
      }
      return true;
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

    bool owned_exclusive_state(
      const detail::snapshot_t &current,
      std::wstring_view virtual_identity,
      const std::vector<std::wstring> &preserved_identities,
      const detail::snapshot_t &original_topology
    ) {
      const auto current_layout = detail::inspect(current);
      if (!current_layout) {
        return false;
      }
      const auto *current_virtual = find_position(*current_layout, virtual_identity);
      if (!current_virtual || current_virtual->x != 0 || current_virtual->y != 0) {
        return false;
      }

      std::int64_t total_width = 0;
      for (const auto &identity : preserved_identities) {
        const auto saved = find_path(original_topology, identity);
        if (!saved) {
          return false;
        }
        total_width += original_topology.modes[source_index(original_topology.paths[*saved])].sourceMode.width;
        if (total_width > std::numeric_limits<LONG>::max()) {
          return false;
        }
      }
      detail::layout_t canonical {{std::wstring(virtual_identity), 0, 0}};
      LONG x = -static_cast<LONG>(total_width);
      for (const auto &identity : preserved_identities) {
        const auto saved = *find_path(original_topology, identity);
        canonical.push_back({identity, x, 0});
        x += static_cast<LONG>(original_topology.modes[source_index(original_topology.paths[saved])].sourceMode.width);
      }
      return std::ranges::all_of(*current_layout, [&](const auto &observed) {
        const auto *expected = find_position(canonical, observed.device_path);
        return expected && expected->x == observed.x && expected->y == observed.y;
      });
    }

    bool preserved_exclusive_identity(const detail::journal_t &journal, std::wstring_view identity) {
      return std::ranges::any_of(journal.exclusive_preserved, [&](const auto &preserved) {
        return same_device(preserved, identity);
      });
    }

    bool owned_exclusive_layout(
      const detail::snapshot_t &current,
      std::wstring_view virtual_identity,
      const detail::snapshot_t &reference
    ) {
      const auto current_layout = detail::inspect(current);
      const auto reference_layout = detail::inspect(reference);
      if (!current_layout || !reference_layout) {
        return false;
      }
      const auto *current_virtual = find_position(*current_layout, virtual_identity);
      return current_virtual && current_virtual->x == 0 && current_virtual->y == 0 &&
             std::ranges::all_of(*current_layout, [&](const auto &observed) {
               const auto *expected = find_position(*reference_layout, observed.device_path);
               return expected && expected->x == observed.x && expected->y == observed.y;
             });
    }

    bool exclusive_continuation_with_recorded_outputs(
      const detail::snapshot_t &current,
      const detail::journal_t &journal,
      std::wstring_view virtual_identity
    ) {
      if (!journal.exclusive_topology || !journal.original_topology ||
          !find_path(current, virtual_identity)) {
        return false;
      }
      auto owned = current;
      for (size_t i = owned.paths.size(); i-- > 0;) {
        const auto &identity = owned.device_paths[i];
        if (find_path(*journal.exclusive_topology, identity)) {
          continue;
        }
        const bool original = find_path(*journal.original_topology, identity).has_value();
        const bool recorded = journal.pending_restore && find_path(*journal.pending_restore, identity).has_value();
        if (!original && !recorded) {
          return false;
        }
        owned.paths.erase(owned.paths.begin() + static_cast<std::ptrdiff_t>(i));
        owned.device_paths.erase(owned.device_paths.begin() + static_cast<std::ptrdiff_t>(i));
        if (i < owned.colors.size()) {
          owned.colors.erase(owned.colors.begin() + static_cast<std::ptrdiff_t>(i));
        }
      }
      if (!find_path(owned, virtual_identity)) {
        return false;
      }
      if (!journal.local_sink.empty()) {
        // A hardware mode switch can restore Windows' remembered extended layout, including
        // the ordinary monitor as primary and new positions for both local outputs. The full
        // snapshot must be valid; after removing only durably recorded ordinary outputs above,
        // the remaining identities must still be our exact source and selected sink. Promotion
        // reestablishes the source origin instead of requiring it before recovery can begin.
        if (detail::inspect(current) && std::ranges::all_of(owned.device_paths, [&](const auto &identity) {
              return same_device(identity, virtual_identity) || same_device(identity, journal.local_sink);
            })) {
          return true;
        }
      }
      return owned_subset(owned, *journal.exclusive_topology) ||
             owned_exclusive_layout(owned, virtual_identity, *journal.exclusive_topology) ||
             owned_exclusive_state(
               owned,
               virtual_identity,
               journal.exclusive_preserved,
               *journal.original_topology
             );
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

    bool same_colors(const detail::snapshot_t &left, const detail::snapshot_t &right) {
      if (left.paths.size() != right.paths.size() || left.colors.size() != left.paths.size() ||
          right.colors.size() != right.paths.size()) {
        return false;
      }
      for (size_t i = 0; i < left.paths.size(); ++i) {
        const auto match = find_path(right, left.device_paths[i]);
        if (!match || left.colors[i].has_value() != right.colors[*match].has_value()) {
          return false;
        }
        if (left.colors[i] && !color_matches(*left.colors[i], *right.colors[*match])) {
          return false;
        }
      }
      return true;
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
      const bool empty = result.paths.empty() && result.modes.empty() && result.device_paths.empty() && result.colors.empty();
      if (!empty && !detail::inspect(result)) {
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

    detail::cursor_clip_manager_t &exclusive_cursor_clip() {
      static detail::cursor_clip_manager_t cursor_clip({
        []() -> std::optional<detail::cursor_bounds_t> {
          RECT rect {};
          if (!GetClipCursor(&rect)) {
            return std::nullopt;
          }
          return detail::cursor_bounds_t {rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top};
        },
        []() -> std::optional<detail::cursor_bounds_t> {
          detail::cursor_bounds_t desktop {GetSystemMetrics(SM_XVIRTUALSCREEN), GetSystemMetrics(SM_YVIRTUALSCREEN), GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN)};
          return desktop.valid() ? std::optional {desktop} : std::nullopt;
        },
        [](std::optional<detail::cursor_bounds_t> clip) {
          if (!clip) {
            return ClipCursor(nullptr) != FALSE;
          }
          const RECT rect {clip->x, clip->y, clip->x + clip->width, clip->y + clip->height};
          return ClipCursor(&rect) != FALSE;
        },
      });
      return cursor_clip;
    }

    // Access is serialized by transaction_mutex. Session ownership is deliberately separate
    // from both the recovery journal and ClipCursor ownership: a virtual-only session with no
    // active AR output needs hotplug reconciliation, but does not need a cursor clip.
    detail::exclusive_session_state_t exclusive_session_state;
    std::optional<std::wstring> active_cursor_clip_identity;
    std::optional<detail::cursor_bounds_t> active_cursor_clip_bounds;

    bool set_exclusive_cursor_clip(std::wstring_view identity, std::optional<detail::cursor_bounds_t> bounds) {
      auto &cursor_clip = exclusive_cursor_clip();
      const bool existing_identity = active_cursor_clip_identity && same_device(*active_cursor_clip_identity, identity);
      const bool existing_bounds = existing_identity && active_cursor_clip_bounds == bounds;
      if (!bounds) {
        // A retiring older display may not release or cancel maintenance for a newer session.
        if (active_cursor_clip_identity && !identity.empty() && !existing_identity) {
          return false;
        }
        // Stop asynchronous maintenance before attempting release. If release itself fails,
        // restore() will retry it, while display/focus notifications must not trap a disconnected
        // local user by reacquiring the clip.
        active_cursor_clip_identity.reset();
        active_cursor_clip_bounds.reset();
        if (!cursor_clip.owns_clip()) {
          return true;
        }
      }
      if (!syncThreadDesktop()) {
        BOOST_LOG(warning) << "Could not synchronize the desktop for exclusive virtual-display cursor isolation.";
        return false;
      }
      const auto previous_dpi = SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
      auto restore_dpi = util::fail_guard([&]() {
        if (previous_dpi) {
          SetThreadDpiAwarenessContext(previous_dpi);
        }
      });
      if (GetAwarenessFromDpiAwarenessContext(GetThreadDpiAwarenessContext()) != DPI_AWARENESS_PER_MONITOR_AWARE) {
        return false;
      }
      const bool result = bounds ? cursor_clip.confine(identity, *bounds) : cursor_clip.release(identity);
      if (!result) {
        BOOST_LOG(warning) << "Could not " << (bounds ? "apply" : "release") << " exclusive virtual-display cursor isolation; retaining cursor ownership for cleanup.";
      } else if (bounds) {
        active_cursor_clip_identity = identity;
        active_cursor_clip_bounds = bounds;
        if (!existing_bounds) {
          BOOST_LOG(info) << "Windows cursor restricted to the virtual display for exclusive streaming: ["
                          << bounds->x << ',' << bounds->y << ',' << bounds->width << ',' << bounds->height << "].";
        }
      } else {
        BOOST_LOG(info) << "Released exclusive virtual-display cursor isolation.";
      }
      return result;
    }

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
                                apply_color,
                                [](const DISPLAYCONFIG_PATH_INFO &path, std::wstring_view) {
                                  return ar_glasses::preserve_during_remote_virtual_display(
                                    path.targetInfo.adapterId,
                                    path.targetInfo.id
                                  );
                                },
                                set_exclusive_cursor_clip});
    }
  }  // namespace

  struct retained_display_t {
    display_spec_t display;
    std::wstring local_sink;
  };

  namespace detail {
    void exclusive_session_state_t::changed() {
      ++generation_;
    }

    void exclusive_session_state_t::prepared() {
      if (expected_) {
        return;
      }
      expected_ = true;
      changed();
    }

    bool exclusive_session_state_t::can_use_identity(std::wstring_view device_path) const {
      if (device_path.empty()) {
        return false;
      }
      return (!pending_identity_ || same_device(*pending_identity_, device_path)) &&
             (!active_identity_ || same_device(*active_identity_, device_path));
    }

    void exclusive_session_state_t::bound(std::wstring_view device_path) {
      if (!expected_ || active_identity_ || pending_identity_) {
        return;
      }
      pending_identity_ = device_path;
      changed();
    }

    bool exclusive_session_state_t::begin_promotion(std::wstring_view device_path) {
      if (!can_use_identity(device_path)) {
        return false;
      }
      bool state_changed = false;
      if (!expected_) {
        expected_ = true;
        state_changed = true;
      }
      if (!active_identity_ && !pending_identity_) {
        pending_identity_ = device_path;
        state_changed = true;
      }
      if (state_changed) {
        changed();
      }
      return true;
    }

    bool exclusive_session_state_t::promotion_succeeded(std::wstring_view device_path) {
      if (!can_use_identity(device_path)) {
        return false;
      }
      const bool already_active = active_identity_ && same_device(*active_identity_, device_path) && !pending_identity_;
      expected_ = true;
      pending_identity_.reset();
      active_identity_ = device_path;
      if (!already_active) {
        changed();
      }
      return true;
    }

    exclusive_restore_action_e exclusive_session_state_t::restore_action(std::wstring_view expected_device_path) const {
      if (expected_device_path.empty()) {
        return exclusive_restore_action_e::disarm_before;
      }
      if (active_identity_ || pending_identity_) {
        return can_use_identity(expected_device_path) ? exclusive_restore_action_e::disarm_before :
                                                        exclusive_restore_action_e::reject;
      }
      return expected_ ? exclusive_restore_action_e::disarm_after_success :
                         exclusive_restore_action_e::keep;
    }

    void exclusive_session_state_t::restore_finished(exclusive_restore_action_e action, bool restored) {
      if (restored && action == exclusive_restore_action_e::disarm_after_success) {
        disarm();
      }
    }

    void exclusive_session_state_t::disarm() {
      if (!expected_ && !pending_identity_ && !active_identity_) {
        return;
      }
      expected_ = false;
      pending_identity_.reset();
      active_identity_.reset();
      changed();
    }

    exclusive_reconcile_action_e exclusive_session_state_t::reconcile_action() const {
      if (active_identity_) {
        return exclusive_reconcile_action_e::reconcile;
      }
      return expected_ || pending_identity_ ? exclusive_reconcile_action_e::defer :
                                              exclusive_reconcile_action_e::recover;
    }

    bool exclusive_session_state_t::expected() const {
      return expected_;
    }

    const std::optional<std::wstring> &exclusive_session_state_t::pending_identity() const {
      return pending_identity_;
    }

    const std::optional<std::wstring> &exclusive_session_state_t::active_identity() const {
      return active_identity_;
    }

    std::uint64_t exclusive_session_state_t::generation() const {
      return generation_;
    }

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
      nlohmann::json result {{"version", !journal.local_sink.empty() ? 4 : journal.exclusive ? 3 :
                                                                                               1},
                             {"prepared", journal.prepared},
                             {"original_primary", platf::to_utf8(journal.original_primary)},
                             {"promoted_primary", platf::to_utf8(journal.promoted_primary)},
                             {"original", encode(journal.original)},
                             {"promoted", encode(journal.promoted)}};
      if (journal.exclusive) {
        result["exclusive"] = true;
        result["exclusive_started"] = journal.exclusive_started;
        result["original_topology"] = encode_snapshot(*journal.original_topology);
        result["before_exclusive"] = journal.before_exclusive ? encode_snapshot(*journal.before_exclusive) : nlohmann::json(nullptr);
        result["pending_restore"] = journal.pending_restore ? encode_snapshot(*journal.pending_restore) : nlohmann::json(nullptr);
        result["exclusive_preserved"] = nlohmann::json::array();
        for (const auto &identity : journal.exclusive_preserved) {
          result["exclusive_preserved"].push_back(platf::to_utf8(identity));
        }
        result["exclusive_topology"] = journal.exclusive_topology ? encode_snapshot(*journal.exclusive_topology) : nlohmann::json(nullptr);
        if (!journal.local_sink.empty()) {
          result["local_sink"] = platf::to_utf8(journal.local_sink);
        }
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
        if (version != 1 && version != 2 && version != 3 && version != 4) {
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
        if (version >= 2) {
          result.exclusive = value.at("exclusive").get<bool>();
          result.exclusive_started = value.at("exclusive_started").get<bool>();
          result.original_topology = decode_snapshot(value.at("original_topology"));
          if (!value.at("before_exclusive").is_null()) {
            result.before_exclusive = decode_snapshot(value.at("before_exclusive"));
          }
          if (!value.at("pending_restore").is_null()) {
            result.pending_restore = decode_snapshot(value.at("pending_restore"));
          }
          if (version >= 3) {
            const auto &preserved = value.at("exclusive_preserved");
            if (!preserved.is_array() || preserved.size() > max_displays) {
              throw std::runtime_error("invalid preserved-display list");
            }
            for (const auto &identity : preserved) {
              result.exclusive_preserved.push_back(platf::from_utf8(identity.get<std::string>()));
            }
            if (!value.at("exclusive_topology").is_null()) {
              result.exclusive_topology = decode_snapshot(value.at("exclusive_topology"));
            }
          }
          if (version >= 4) {
            result.local_sink = platf::from_utf8(value.at("local_sink").get<std::string>());
            if (result.local_sink.empty()) {
              return std::nullopt;
            }
          }
        }
        return valid_journal(result) ? std::make_optional(std::move(result)) : std::nullopt;
      } catch (const std::exception &) {
        return std::nullopt;
      }
    }

    manager_t::manager_t(io_t io):
        io_(std::move(io)) {}

    bool manager_t::prepare(bool exclusive, std::wstring_view local_sink) {
      if (!local_sink.empty() && !exclusive) {
        return false;
      }
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
        return local_sink.empty();
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
        if (io_.preserve_exclusive) {
          for (size_t i = 0; i < snapshot->paths.size(); ++i) {
            if ((local_sink.empty() || same_device(local_sink, snapshot->device_paths[i])) && io_.preserve_exclusive(snapshot->paths[i], snapshot->device_paths[i])) {
              journal.exclusive_preserved.push_back(snapshot->device_paths[i]);
            }
          }
          std::ranges::sort(journal.exclusive_preserved, [](const auto &left, const auto &right) {
            return CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE) == CSTR_LESS_THAN;
          });
        }
        journal.local_sink = local_sink;
      }
      return valid_journal(journal) && io_.save(journal);
    }

    bool manager_t::is_local_exclusive(std::wstring_view source_device_path) {
      const auto loaded = io_.load();
      return loaded.success && loaded.journal && valid_journal(*loaded.journal) &&
             !loaded.journal->local_sink.empty() &&
             same_device(loaded.journal->promoted_primary, source_device_path);
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
      if (prepared.exclusive && target && (!prepared.local_sink.empty() || !original_primary || current->size() != prepared.original.size() + (new_target ? 1 : 0))) {
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

      const auto available = io_.query_all ? io_.query_all() : std::nullopt;
      if (!available || available->paths.size() > max_available_paths ||
          available->paths.size() != available->device_paths.size()) {
        return false;
      }
      const auto virtual_index = find_path(*snapshot, device_path);
      if (!virtual_index) {
        return false;
      }
      auto virtual_spec = display_spec(*snapshot, *virtual_index);
      if (journal.exclusive_topology && journal.local_sink.empty()) {
        const auto prior_virtual = find_path(*journal.exclusive_topology, device_path);
        if (prior_virtual && *prior_virtual < journal.exclusive_topology->colors.size()) {
          virtual_spec.color = journal.exclusive_topology->colors[*prior_virtual];
        }
      }
      virtual_spec.source.position = {0, 0};
      std::vector<display_spec_t> wanted;
      wanted.push_back(std::move(virtual_spec));

      auto target_available = [&](std::wstring_view identity) {
        for (size_t i = 0; i < available->paths.size(); ++i) {
          if (available->paths[i].targetInfo.targetAvailable && same_device(available->device_paths[i], identity)) {
            return true;
          }
        }
        return false;
      };
      std::vector<display_spec_t> preserved;
      for (const auto &identity : journal.exclusive_preserved) {
        const auto original_index = find_path(*journal.original_topology, identity);
        const auto pending_index = journal.pending_restore ? find_path(*journal.pending_restore, identity) : std::nullopt;
        if (!original_index && !pending_index) {
          return false;
        }
        auto spec = original_index ? display_spec(*journal.original_topology, *original_index) :
                                     display_spec(*journal.pending_restore, *pending_index);
        const auto current_index = find_path(*snapshot, identity);
        if (!journal.local_sink.empty() && same_device(identity, journal.local_sink) && current_index) {
          // The physical button controls the local presentation mode and color.
          spec = display_spec(*snapshot, *current_index);
        }
        const auto prior_index = journal.exclusive_topology ? find_path(*journal.exclusive_topology, identity) : std::nullopt;
        if (journal.local_sink.empty() && prior_index && *prior_index < journal.exclusive_topology->colors.size()) {
          spec.color = journal.exclusive_topology->colors[*prior_index];
        }
        preserved.push_back(std::move(spec));
      }
      std::ranges::sort(preserved, [](const auto &left, const auto &right) {
        return CompareStringOrdinal(left.identity.c_str(), -1, right.identity.c_str(), -1, TRUE) == CSTR_LESS_THAN;
      });
      std::erase_if(preserved, [&](const auto &spec) {
        return !target_available(spec.identity);
      });
      std::int64_t preserved_width = 0;
      for (const auto &spec : preserved) {
        preserved_width += spec.source.width;
        if (preserved_width > std::numeric_limits<LONG>::max()) {
          return false;
        }
      }
      LONG preserved_x = -static_cast<LONG>(preserved_width);
      for (auto &spec : preserved) {
        const auto width = spec.source.width;
        spec.source.position = {preserved_x, 0};
        preserved_x += static_cast<LONG>(width);
        wanted.push_back(std::move(spec));
      }

      const auto desired = build_topology(*available, wanted);
      const auto desired_layout = desired ? inspect(*desired) : std::nullopt;
      if (!desired || !desired_layout) {
        return false;
      }
      if (journal.exclusive_started && same_topology(*snapshot, *desired) && same_colors(*snapshot, *desired)) {
        if (!journal.exclusive_topology || !same_topology(*journal.exclusive_topology, *desired) ||
            !same_colors(*journal.exclusive_topology, *desired)) {
          journal.exclusive_topology = *desired;
          return valid_journal(journal) && io_.save(journal);
        }
        return true;
      }
      const bool remembered_exclusive = journal.exclusive_started && !journal.pending_restore && same_topology(*snapshot, *journal.before_exclusive);
      const bool continuing_exclusive = journal.exclusive_started &&
                                        ((journal.exclusive_topology && owned_subset(*snapshot, *journal.exclusive_topology)) ||
                                         exclusive_continuation_with_recorded_outputs(*snapshot, journal, device_path) ||
                                         owned_exclusive_state(
                                           *snapshot,
                                           device_path,
                                           journal.exclusive_preserved,
                                           *journal.original_topology
                                         ));
      const bool resuming_restore = journal.exclusive_started && journal.pending_restore &&
                                    (owned_subset(*snapshot, *journal.pending_restore) ||
                                     same_layout_and_physical_modes(*snapshot, *journal.pending_restore, device_path));
      if (journal.exclusive_started && !remembered_exclusive && !continuing_exclusive && !resuming_restore) {
        return false;
      }
      if (!remembered_exclusive && !continuing_exclusive && !resuming_restore && !same_layout(*layout, journal.original) && !same_layout(*layout, journal.promoted)) {
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
        if ((remembered_exclusive || continuing_exclusive || resuming_restore) && !current_index) {
          continue;
        }
        if (!current_index || *current_index >= snapshot->colors.size() || !snapshot->colors[*current_index] ||
            (!remembered_exclusive && !continuing_exclusive && !resuming_restore &&
             !color_matches(*snapshot->colors[*current_index], *journal.original_topology->colors[i]))) {
          return false;
        }
        current_physical.paths.push_back(snapshot->paths[*current_index]);
        current_physical.device_paths.push_back(snapshot->device_paths[*current_index]);
        saved_physical.paths.push_back(journal.original_topology->paths[i]);
        saved_physical.device_paths.push_back(journal.original_topology->device_paths[i]);
      }
      if (saved_physical.paths.empty() && !remembered_exclusive && !continuing_exclusive && !resuming_restore) {
        return io_.clear();
      }
      if (!remembered_exclusive && !continuing_exclusive && !resuming_restore &&
          !same_modes(current_physical, saved_physical)) {
        return false;
      }
      journal.exclusive_started = true;
      journal.before_exclusive = *snapshot;
      journal.exclusive_topology = *desired;
      if (!valid_journal(journal) || !io_.save(journal)) {
        return false;
      }
      const auto latest = io_.query();
      if (!latest || !same_topology(*latest, *snapshot)) {
        return false;
      }
      if (!same_topology(*snapshot, *desired) && !io_.apply(*desired)) {
        return false;
      }
      auto observed = io_.query();
      if (!observed || !same_topology(*observed, *desired)) {
        return false;
      }
      bool colors_ok = true;
      for (const auto &spec : wanted) {
        if (!spec.color) {
          continue;
        }
        const auto index = find_path(*observed, spec.identity);
        if (!index || *index >= observed->colors.size() || !observed->colors[*index]) {
          colors_ok = false;
          continue;
        }
        if (!color_matches(*observed->colors[*index], *spec.color)) {
          colors_ok = io_.set_color && io_.set_color(observed->paths[*index], *spec.color) && colors_ok;
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
        colors_ok = index && *index < observed->colors.size() && observed->colors[*index] &&
                    color_matches(*observed->colors[*index], *spec.color) && colors_ok;
      }
      if (!colors_ok) {
        return false;
      }
      if (wanted.size() == 1) {
        BOOST_LOG(info) << "Virtual display is temporarily the only active Windows display.";
      } else {
        BOOST_LOG(info) << "Virtual display is temporarily primary with ordinary monitors disabled; kept "
                        << wanted.size() - 1 << " approved AR display(s) active.";
      }
      return true;
    }

    bool manager_t::pause(std::wstring_view device_path, retained_display_ptr &retained, std::wstring_view local_sink) {
      if (device_path.empty() || (retained && !same_device(retained->display.identity, device_path)) || (!local_sink.empty() && (same_device(local_sink, device_path) || (retained && !same_device(retained->local_sink, local_sink)))) || (io_.cursor_clip && !io_.cursor_clip(device_path, std::nullopt))) {
        return false;
      }
      const auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && (!valid_journal(*loaded.journal) || (!loaded.journal->prepared && !same_device(loaded.journal->promoted_primary, device_path)) || (!local_sink.empty() && !loaded.journal->local_sink.empty() && !same_device(local_sink, loaded.journal->local_sink)) || (retained && !loaded.journal->local_sink.empty() && !same_device(retained->local_sink, loaded.journal->local_sink))))) {
        return false;
      }
      const auto current = io_.query();
      if (!current || (!current->paths.empty() && !inspect(*current))) {
        return false;
      }
      const auto index = find_path(*current, device_path);
      if (!retained) {
        if (!index || *index >= current->colors.size() || !current->colors[*index]) {
          return false;
        }
        retained = std::make_shared<retained_display_t>(retained_display_t {display_spec(*current, *index)});
        retained->display.source.position = {0, 0};
        retained->local_sink = loaded.journal && !loaded.journal->local_sink.empty() ?
                                 loaded.journal->local_sink :
                                 std::wstring(local_sink);
      }
      if (loaded.journal) {
        if (loaded.journal->exclusive_started) {
          return restore_exclusive(*loaded.journal, false);
        }
        // A failed reactivation may have saved its physical baseline before binding. Complete
        // that recovery first, then detach any target which Windows already made active.
        return restore(device_path) && pause(device_path, retained, retained->local_sink);
      }
      if (!index || current->paths.size() == 1) {
        // An already detached target needs no mutation. A truly headless desktop has nowhere
        // to migrate windows, so retain its sole output for the next reconnect.
        return true;
      }
      // Reconnect rollback can reach us after its activation journal was completed. Save the
      // current physical desktop before removing this still-active target from that desktop.
      const auto connected_sink = find_path(*current, retained->local_sink) ? std::wstring_view(retained->local_sink) : std::wstring_view {};
      if (!prepare(true, connected_sink) || !bind_pending(device_path)) {
        return false;
      }
      const auto prepared = io_.load();
      if (!prepared.success || !prepared.journal) {
        return false;
      }
      auto journal = *prepared.journal;
      journal.exclusive_started = true;
      journal.before_exclusive = *current;
      if (!valid_journal(journal) || !io_.save(journal)) {
        return false;
      }
      return restore_exclusive(std::move(journal), false);
    }

    bool manager_t::reactivate(const retained_display_ptr &retained, bool exclusive) {
      if (!retained || !io_.query_all || !io_.set_color) {
        return false;
      }
      const auto &identity = retained->display.identity;
      auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && (!valid_journal(*loaded.journal) || !loaded.journal->exclusive || loaded.journal->local_sink != retained->local_sink || loaded.journal->prepared || !same_device(loaded.journal->promoted_primary, identity)))) {
        return false;
      }
      const auto current = io_.query();
      if (!current || (!current->paths.empty() && !inspect(*current)) || current->colors.size() != current->paths.size() || std::ranges::any_of(current->colors, [](const auto &color) {
            return !color;
          })) {
        return false;
      }
      if (!retained->local_sink.empty() && !find_path(*current, retained->local_sink)) {
        return false;
      }
      auto verify_colors = [&](const snapshot_t &desired) {
        auto observed = io_.query();
        if (!observed || !same_topology(*observed, desired)) {
          return false;
        }
        for (size_t i = 0; i < desired.paths.size(); ++i) {
          const auto index = find_path(*observed, desired.device_paths[i]);
          if (!index || *index >= observed->colors.size() || !observed->colors[*index] || !desired.colors[i]) {
            return false;
          }
          if (!color_matches(*observed->colors[*index], *desired.colors[i]) && !io_.set_color(observed->paths[*index], *desired.colors[i])) {
            return false;
          }
        }
        observed = io_.query();
        return observed && same_topology(*observed, desired) && same_colors(*observed, desired);
      };
      if (find_path(*current, identity)) {
        // Idempotence covers a previous CCD apply whose color verification had to retry, and
        // the sole-output pause which deliberately kept a headless virtual desktop active.
        if (!loaded.journal) {
          const auto index = *find_path(*current, identity);
          const auto wanted = build_topology(*current, {retained->display});
          snapshot_t active_target = *current;
          active_target.paths = {current->paths[index]};
          active_target.device_paths = {current->device_paths[index]};
          if (!wanted || !same_modes(active_target, *wanted)) {
            return false;
          }
          // A local-only PC keeps its sole VD when the glasses are unplugged. Their return
          // needs a fresh local sink journal even though the source never needed activation.
          return !exclusive || retained->local_sink.empty() ||
                 (prepare(true, retained->local_sink) && bind_pending(identity));
        }
        const auto &journal = *loaded.journal;
        if (!journal.pending_restore || !same_topology(*current, *journal.pending_restore) || !verify_colors(*journal.pending_restore)) {
          return false;
        }
        return exclusive || restore(identity);
      }
      if (loaded.journal && (!loaded.journal->pending_restore || !owned_subset(*current, *loaded.journal->pending_restore))) {
        return false;
      }
      const auto available = io_.query_all();
      if (!available) {
        return false;
      }
      std::vector<display_spec_t> wanted;
      std::int64_t right = current->paths.empty() ? 0 : std::numeric_limits<LONG>::min();
      LONG top = 0;
      for (size_t i = 0; i < current->paths.size(); ++i) {
        auto spec = display_spec(*current, i);
        if (loaded.journal && loaded.journal->pending_restore) {
          const auto saved = find_path(*loaded.journal->pending_restore, spec.identity);
          if (saved && *saved < loaded.journal->pending_restore->colors.size()) {
            spec.color = loaded.journal->pending_restore->colors[*saved];
          }
        }
        const auto edge = static_cast<std::int64_t>(spec.source.position.x) + spec.source.width;
        if (edge > right) {
          right = edge;
          top = spec.source.position.y;
        }
        wanted.push_back(std::move(spec));
      }
      if (right > std::numeric_limits<LONG>::max() || right + retained->display.source.width > std::numeric_limits<LONG>::max()) {
        return false;
      }
      auto virtual_spec = retained->display;
      virtual_spec.source.position = {static_cast<LONG>(right), top};
      wanted.push_back(std::move(virtual_spec));
      const auto desired = build_topology(*available, wanted);
      const auto layout = desired ? inspect(*desired) : std::nullopt;
      if (!desired || !layout) {
        return false;
      }
      if (!loaded.journal) {
        // Full color recovery is required even if this reconnect switches to primary-only
        // mode. Its ordinary primary-only journal starts after activation completes below.
        if (current->paths.empty()) {
          // The added VD position and original physical topology are separate journal fields.
          // Record the actual empty baseline when physical outputs vanish during the pause.
          // A local presentation cannot resume without its physical sink.
          if (!retained->local_sink.empty()) {
            return false;
          }
          journal_t headless {identity, identity, *layout, *layout};
          headless.exclusive = true;
          headless.original_topology = *current;
          loaded.journal = std::move(headless);
        } else {
          if (!prepare(true, retained->local_sink)) {
            return false;
          }
          loaded = io_.load();
          if (!loaded.success || !loaded.journal) {
            return false;
          }
        }
      }
      auto journal = *loaded.journal;
      if (journal.prepared) {
        journal.prepared = false;
        journal.promoted_primary = identity;
        journal.original = *layout;
        const auto promoted = translated(*layout, static_cast<LONG>(right), top);
        if (!promoted) {
          return false;
        }
        journal.promoted = *promoted;
      }
      journal.exclusive_started = true;
      journal.before_exclusive = *current;
      journal.pending_restore = *desired;
      if (!valid_journal(journal) || !io_.save(journal)) {
        return false;
      }
      const auto latest = io_.query();
      if (!latest || !same_topology(*latest, *current) || !same_colors(*latest, *current) || !io_.apply(*desired) || !verify_colors(*desired) || !bind_pending(identity)) {
        return false;
      }
      BOOST_LOG(info) << "Reactivated the retained virtual display for session resume.";
      return exclusive || restore(identity);
    }

    bool manager_t::restore_exclusive(journal_t journal, bool keep_virtual_active) {
      if (!io_.query_all || !io_.set_color) {
        return false;
      }
      const auto current = io_.query();
      if (!current || current->colors.size() != current->paths.size() ||
          (!current->paths.empty() && !inspect(*current))) {
        return false;
      }
      const auto virtual_index = find_path(*current, journal.promoted_primary);
      snapshot_t owned_current = *current;
      std::vector<display_spec_t> added_outputs;
      bool newly_recorded_output = false;
      bool preservation_changed = false;
      for (size_t i = current->paths.size(); i-- > 0;) {
        const auto &identity = current->device_paths[i];
        if (same_device(identity, journal.promoted_primary) || find_path(*journal.original_topology, identity)) {
          continue;
        }
        const auto pending_index = journal.pending_restore ? find_path(*journal.pending_restore, identity) : std::nullopt;
        newly_recorded_output = newly_recorded_output || !pending_index;
        added_outputs.push_back(pending_index ? display_spec(*journal.pending_restore, *pending_index) :
                                                display_spec(*current, i));
        if (journal.local_sink.empty() && io_.preserve_exclusive && io_.preserve_exclusive(current->paths[i], identity) && !preserved_exclusive_identity(journal, identity)) {
          journal.exclusive_preserved.push_back(identity);
          preservation_changed = true;
        }
        owned_current.paths.erase(owned_current.paths.begin() + static_cast<std::ptrdiff_t>(i));
        owned_current.device_paths.erase(owned_current.device_paths.begin() + static_cast<std::ptrdiff_t>(i));
        owned_current.colors.erase(owned_current.colors.begin() + static_cast<std::ptrdiff_t>(i));
      }
      std::ranges::sort(journal.exclusive_preserved, [](const auto &left, const auto &right) {
        return CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE) == CSTR_LESS_THAN;
      });
      const auto owned_virtual_index = find_path(owned_current, journal.promoted_primary);
      const bool owned_exclusive = owned_virtual_index &&
                                   ((journal.exclusive_topology &&
                                     owned_exclusive_layout(owned_current, journal.promoted_primary, *journal.exclusive_topology)) ||
                                    owned_exclusive_state(
                                      owned_current,
                                      journal.promoted_primary,
                                      journal.exclusive_preserved,
                                      *journal.original_topology
                                    ));
      // SudoVDA may retire the source when an older host exits before it can recover a
      // hardware mode switch. The remaining exact recorded local outputs still need their
      // saved ordinary layout restored; no replacement or unrecorded output gains ownership.
      const bool retired_local_source = !journal.local_sink.empty() && !virtual_index &&
                                        std::ranges::all_of(current->device_paths, [&](const auto &identity) {
                                          return find_path(*journal.original_topology, identity) ||
                                                 (journal.pending_restore && find_path(*journal.pending_restore, identity));
                                        });
      const bool paused_activation = !keep_virtual_active && virtual_index && journal.pending_restore &&
                                     same_layout_and_physical_modes(*current, *journal.pending_restore, journal.promoted_primary);
      const bool owned = owned_current.paths.empty() || owned_exclusive || retired_local_source || paused_activation || same_topology(owned_current, *journal.before_exclusive) ||
                         exclusive_continuation_with_recorded_outputs(*current, journal, journal.promoted_primary) ||
                         (journal.pending_restore && owned_subset(owned_current, *journal.pending_restore)) ||
                         (!owned_virtual_index && owned_subset(owned_current, *journal.original_topology));
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
      if (journal.pending_restore) {
        for (size_t i = 0; i < journal.pending_restore->paths.size(); ++i) {
          const auto &identity = journal.pending_restore->device_paths[i];
          if (same_device(identity, journal.promoted_primary) || find_path(*journal.original_topology, identity) ||
              !available_identity(identity) || std::ranges::any_of(added_outputs, [&](const auto &entry) {
                return same_device(entry.identity, identity);
              })) {
            continue;
          }
          added_outputs.push_back(display_spec(*journal.pending_restore, i));
        }
      }
      std::ranges::sort(added_outputs, [](const auto &left, const auto &right) {
        return CompareStringOrdinal(left.identity.c_str(), -1, right.identity.c_str(), -1, TRUE) == CSTR_LESS_THAN;
      });
      std::vector<std::wstring> retired_preserved;
      std::erase_if(journal.exclusive_preserved, [&](const auto &identity) {
        const bool retire = !find_path(*journal.original_topology, identity) &&
                            std::ranges::none_of(added_outputs, [&](const auto &added) {
                              return same_device(added.identity, identity);
                            });
        if (retire) {
          retired_preserved.push_back(identity);
        }
        return retire;
      });
      if (journal.exclusive_topology) {
        for (size_t i = journal.exclusive_topology->paths.size(); i-- > 0;) {
          if (std::ranges::none_of(retired_preserved, [&](const auto &identity) {
                return same_device(journal.exclusive_topology->device_paths[i], identity);
              })) {
            continue;
          }
          journal.exclusive_topology->paths.erase(journal.exclusive_topology->paths.begin() + static_cast<std::ptrdiff_t>(i));
          journal.exclusive_topology->device_paths.erase(journal.exclusive_topology->device_paths.begin() + static_cast<std::ptrdiff_t>(i));
          journal.exclusive_topology->colors.erase(journal.exclusive_topology->colors.begin() + static_cast<std::ptrdiff_t>(i));
        }
      }
      if ((newly_recorded_output || preservation_changed) && owned) {
        auto record = added_outputs;
        std::wstring anchor;
        if (virtual_index) {
          auto virtual_spec = display_spec(*current, *virtual_index);
          anchor = virtual_spec.identity;
          record.push_back(std::move(virtual_spec));
        } else if (!record.empty()) {
          anchor = record.front().identity;
        }
        if (anchor.empty() || !pack_horizontally(record, anchor, 0, 0)) {
          return false;
        }
        const auto provisional_restore = build_topology(*available, record);
        if (!provisional_restore) {
          return false;
        }
        journal.before_exclusive = *current;
        journal.pending_restore = *provisional_restore;
        if (!valid_journal(journal) || !io_.save(journal)) {
          return false;
        }
      }
      const bool had_added_outputs = !added_outputs.empty();
      std::vector<display_spec_t> wanted;
      bool missing = false;
      bool missing_local_sink = false;
      bool local_sink_resized = false;
      size_t physical_count = 0;
      for (size_t i = 0; i < journal.original_topology->paths.size(); ++i) {
        const auto &identity = journal.original_topology->device_paths[i];
        if (same_device(identity, journal.promoted_primary)) {
          continue;
        }
        ++physical_count;
        if (!available_identity(identity)) {
          if (!journal.local_sink.empty() && same_device(identity, journal.local_sink)) {
            missing_local_sink = true;
          } else {
            missing = true;
          }
          continue;
        }
        auto spec = display_spec(*journal.original_topology, i);
        const auto sink_index = find_path(*current, identity);
        if (!journal.local_sink.empty() && same_device(identity, journal.local_sink) && sink_index) {
          const auto saved_position = spec.source.position;
          const auto saved_width = spec.source.width;
          const auto saved_height = spec.source.height;
          spec = display_spec(*current, *sink_index);
          local_sink_resized = spec.source.width != saved_width || spec.source.height != saved_height;
          spec.source.position = saved_position;
          // A sink originally attached to the primary's left keeps its right edge when
          // its hardware switches width. Restoring the old left edge would overlap it.
          if (saved_position.x < 0 && static_cast<std::int64_t>(saved_position.x) + saved_width <= 0) {
            const auto rebased_x = static_cast<std::int64_t>(saved_position.x) + saved_width - spec.source.width;
            if (rebased_x < std::numeric_limits<LONG>::min()) {
              return false;
            }
            spec.source.position.x = static_cast<LONG>(rebased_x);
          }
        }
        wanted.push_back(std::move(spec));
      }
      if (physical_count == 0 && added_outputs.empty()) {
        return io_.clear();
      }
      if (missing_local_sink && !missing && physical_count == 1 && wanted.empty() && added_outputs.empty()) {
        // A PC whose only physical output was these glasses is intentionally headless
        // after they are unplugged. There is no remaining physical output to restore.
        return io_.clear();
      }
      if (physical_count != 0 && wanted.empty() && added_outputs.empty()) {
        return false;
      }
      if (wanted.empty()) {
        wanted.push_back(std::move(added_outputs.front()));
        added_outputs.erase(added_outputs.begin());
      }
      auto primary = std::ranges::find_if(wanted, [&](const auto &spec) {
        return same_device(spec.identity, journal.original_primary);
      });
      std::wstring restore_primary;
      POINTL origin {};
      if (primary != wanted.end()) {
        restore_primary = primary->identity;
        origin = primary->source.position;
        std::rotate(wanted.begin(), primary, primary + 1);
      } else if (keep_virtual_active && same_device(journal.original_primary, journal.promoted_primary) && virtual_index && find_path(*journal.original_topology, journal.original_primary)) {
        const auto original_primary_index = find_path(*journal.original_topology, journal.original_primary);
        if (!original_primary_index) {
          return false;
        }
        restore_primary = journal.original_primary;
        origin = journal.original_topology->modes[source_index(journal.original_topology->paths[*original_primary_index])].sourceMode.position;
      } else {
        primary = wanted.begin();
        restore_primary = primary->identity;
        origin = primary->source.position;
        std::rotate(wanted.begin(), primary, primary + 1);
      }
      if (!added_outputs.empty()) {
        std::int64_t rightmost = std::numeric_limits<LONG>::min();
        for (const auto &spec : wanted) {
          const auto edge = static_cast<std::int64_t>(spec.source.position.x) + spec.source.width;
          if (edge > std::numeric_limits<LONG>::max()) {
            return false;
          }
          rightmost = std::max(rightmost, edge);
        }
        for (auto &spec : added_outputs) {
          if (rightmost > std::numeric_limits<LONG>::max() ||
              rightmost + spec.source.width > std::numeric_limits<LONG>::max()) {
            return false;
          }
          spec.source.position = {static_cast<LONG>(rightmost), origin.y};
          rightmost += spec.source.width;
          wanted.push_back(std::move(spec));
        }
      }
      if (virtual_index && keep_virtual_active) {
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
      // User-owned active outputs override the saved plan before its geometry is validated.
      // Removing a VD that bridged two physical monitors can disconnect that arrangement.
      if (!owned) {
        if (!preserve_current_desktop(*current, wanted, journal.promoted_primary, keep_virtual_active)) {
          return false;
        }
        primary = std::ranges::find_if(wanted, [](const auto &spec) {
          return spec.source.position.x == 0 && spec.source.position.y == 0;
        });
        if (primary == wanted.end()) {
          return false;
        }
        restore_primary = primary->identity;
        origin = {};
        BOOST_LOG(info) << "Preserving the user's current desktop while restoring available disabled outputs.";
      }
      bool restore_overlap = false;
      if (!journal.local_sink.empty() || !owned) {
        for (size_t i = 0; i < wanted.size(); ++i) {
          for (size_t j = 0; j < i; ++j) {
            const auto &a = wanted[i].source;
            const auto &b = wanted[j].source;
            restore_overlap = restore_overlap ||
                              (static_cast<std::int64_t>(a.position.x) < static_cast<std::int64_t>(b.position.x) + b.width &&
                               static_cast<std::int64_t>(b.position.x) < static_cast<std::int64_t>(a.position.x) + a.width &&
                               static_cast<std::int64_t>(a.position.y) < static_cast<std::int64_t>(b.position.y) + b.height &&
                               static_cast<std::int64_t>(b.position.y) < static_cast<std::int64_t>(a.position.y) + a.height);
          }
        }
      }
      bool restore_disconnected = false;
      if ((local_sink_resized || !owned) && !wanted.empty()) {
        std::vector<bool> connected(wanted.size(), false);
        connected.front() = true;
        bool progress = true;
        while (progress) {
          progress = false;
          for (size_t i = 0; i < wanted.size(); ++i) {
            if (connected[i]) {
              continue;
            }
            const auto &a = wanted[i].source;
            const auto ar = static_cast<std::int64_t>(a.position.x) + a.width;
            const auto ab = static_cast<std::int64_t>(a.position.y) + a.height;
            for (size_t j = 0; j < wanted.size(); ++j) {
              if (!connected[j]) {
                continue;
              }
              const auto &b = wanted[j].source;
              const auto br = static_cast<std::int64_t>(b.position.x) + b.width;
              const auto bb = static_cast<std::int64_t>(b.position.y) + b.height;
              const bool horizontal = (ar == b.position.x || br == a.position.x) &&
                                      std::max(a.position.y, b.position.y) < std::min(ab, bb);
              const bool vertical = (ab == b.position.y || bb == a.position.y) &&
                                    std::max(a.position.x, b.position.x) < std::min(ar, br);
              if (horizontal || vertical) {
                connected[i] = true;
                progress = true;
                break;
              }
            }
          }
        }
        restore_disconnected = std::ranges::any_of(connected, [](bool reached) {
          return !reached;
        });
      }
      if ((owned && (missing || missing_local_sink)) || restore_overlap || restore_disconnected) {
        // Removing an unavailable monitor from the saved arrangement can leave a hole between
        // survivors. Restore a contiguous usable desktop; absent originals do not retain
        // a blocking recovery transaction after the available outputs are verified.
        if (!pack_horizontally(wanted, restore_primary, 0, 0)) {
          return false;
        }
      } else {
        for (auto &spec : wanted) {
          const auto x = subtract(spec.source.position.x, origin.x);
          const auto y = subtract(spec.source.position.y, origin.y);
          if (!x || !y) {
            return false;
          }
          spec.source.position = {*x, *y};
        }
      }
      const auto desired = build_topology(*available, wanted);
      if (!desired) {
        return false;
      }
      if (!owned && !rebase_recovery(journal, *current, *desired)) {
        return false;
      }
      journal.pending_restore = *desired;
      if (had_added_outputs) {
        // These outputs were not present in the launch baseline. Persist the exact pre-restore
        // ownership point before SetDisplayConfig so a crash remains recoverable.
        journal.before_exclusive = *current;
      }
      if (!valid_journal(journal) || !io_.save(journal)) {
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
      if (!color_ok) {
        BOOST_LOG(warning) << "Available physical displays are active; retaining exclusive-display recovery until their color settings are restored.";
        return false;
      }
      if (missing || missing_local_sink) {
        // All available outputs were restored and verified. An unplugged monitor must not
        // hold future sessions hostage; its return starts from Windows' current layout.
        BOOST_LOG(info) << "Restored available physical displays; absent monitors will use a fresh topology baseline on reconnect.";
      }
      BOOST_LOG(info) << (keep_virtual_active ? "Restored physical displays, modes, color settings, and primary display after exclusive streaming." : "Restored physical displays and detached the retained virtual display from the desktop.");
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
      bool missing_local_sink = false;
      for (size_t i = 0; i < journal.original_topology->paths.size(); ++i) {
        const auto &identity = journal.original_topology->device_paths[i];
        bool found = false;
        for (size_t j = 0; j < available->paths.size(); ++j) {
          found = found || (available->paths[j].targetInfo.targetAvailable && same_device(available->device_paths[j], identity));
        }
        if (found) {
          physical.push_back(display_spec(*journal.original_topology, i));
        } else {
          if (!journal.local_sink.empty() && same_device(identity, journal.local_sink)) {
            missing_local_sink = true;
          } else {
            missing = true;
          }
        }
      }
      if (physical.empty()) {
        return missing_local_sink && !missing && extras.empty() && io_.clear();
      }
      if (missing || missing_local_sink) {
        std::wstring anchor_identity;
        LONG anchor_x = 0;
        LONG anchor_y = 0;
        if (!extras.empty()) {
          // An unbound output is never identified as ours or moved/disabled. Place the compacted
          // known block directly beside it until the driver orphan is gone or exactly bound.
          const auto rightmost = std::ranges::max_element(extras, [](const auto &a, const auto &b) {
            return static_cast<int64_t>(a.source.position.x) + a.source.width < static_cast<int64_t>(b.source.position.x) + b.source.width;
          });
          const auto right_edge = static_cast<int64_t>(rightmost->source.position.x) + rightmost->source.width;
          if (right_edge < std::numeric_limits<LONG>::min() || right_edge > std::numeric_limits<LONG>::max()) {
            return false;
          }
          const auto leftmost = std::ranges::min_element(physical, [](const auto &left, const auto &right) {
            if (left.source.position.x != right.source.position.x) {
              return left.source.position.x < right.source.position.x;
            }
            return left.source.position.y < right.source.position.y;
          });
          anchor_identity = leftmost->identity;
          anchor_x = static_cast<LONG>(right_edge);
          anchor_y = rightmost->source.position.y;
        } else {
          auto primary = std::ranges::find_if(physical, [&](const auto &spec) {
            return same_device(spec.identity, journal.original_primary);
          });
          if (primary == physical.end()) {
            primary = physical.begin();
          }
          anchor_identity = primary->identity;
        }
        if (!pack_horizontally(physical, anchor_identity, anchor_x, anchor_y)) {
          return false;
        }
      } else {
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
      if (!extras.empty() || !colors_ok) {
        return false;
      }
      return io_.clear();
    }

    bool manager_t::clip_cursor_to_display(const snapshot_t &snapshot, std::wstring_view device_path) {
      const auto layout = inspect(snapshot);
      if (!layout) {
        return false;
      }
      const auto index = find_path(snapshot, device_path);
      if (!index) {
        return false;
      }
      const auto &source = snapshot.modes[source_index(snapshot.paths[*index])].sourceMode;
      return io_.cursor_clip(device_path, cursor_bounds_t {source.position.x, source.position.y, static_cast<int>(source.width), static_cast<int>(source.height)});
    }

    bool manager_t::reconcile_active_exclusive(std::wstring_view device_path) {
      auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
        return false;
      }
      if (!loaded.journal) {
        // A session that began truly headless needs no recovery journal. If a display appears
        // later, promotion captures that new baseline before disabling or preserving it.
        return promote_exclusive(device_path) && refresh_exclusive_cursor_clip(device_path);
      }
      auto journal = *loaded.journal;
      if (!journal.exclusive || !journal.exclusive_started ||
          !same_device(journal.promoted_primary, device_path) || !journal.exclusive_topology ||
          !journal.original_topology || !io_.query_all) {
        return false;
      }
      const auto current = io_.query();
      const auto virtual_index = current ? find_path(*current, device_path) : std::nullopt;
      const auto available = io_.query_all();
      if (!current || !virtual_index || !inspect(*current) || !available ||
          available->paths.size() > max_available_paths ||
          available->paths.size() != available->device_paths.size()) {
        return false;
      }

      auto is_available = [&](std::wstring_view identity) {
        for (size_t i = 0; i < available->paths.size(); ++i) {
          if (available->paths[i].targetInfo.targetAvailable && same_device(available->device_paths[i], identity)) {
            return true;
          }
        }
        return false;
      };
      std::vector<display_spec_t> recorded_hotplugs;
      auto upsert_hotplug = [&](display_spec_t spec) {
        const auto existing = std::ranges::find_if(recorded_hotplugs, [&](const auto &entry) {
          return same_device(entry.identity, spec.identity);
        });
        if (existing == recorded_hotplugs.end()) {
          recorded_hotplugs.push_back(std::move(spec));
        } else {
          *existing = std::move(spec);
        }
      };
      if (journal.pending_restore) {
        for (size_t i = 0; i < journal.pending_restore->paths.size(); ++i) {
          const auto &identity = journal.pending_restore->device_paths[i];
          if (!same_device(identity, device_path) && !find_path(*journal.original_topology, identity) &&
              is_available(identity)) {
            upsert_hotplug(display_spec(*journal.pending_restore, i));
          }
        }
      }

      bool discovered_hotplug = false;
      for (size_t i = 0; i < current->paths.size(); ++i) {
        const auto &identity = current->device_paths[i];
        if (same_device(identity, device_path) || find_path(*journal.original_topology, identity)) {
          continue;
        }
        discovered_hotplug = true;
        upsert_hotplug(display_spec(*current, i));
        if (journal.local_sink.empty() && io_.preserve_exclusive && io_.preserve_exclusive(current->paths[i], identity) && !preserved_exclusive_identity(journal, identity)) {
          journal.exclusive_preserved.push_back(identity);
        }
      }
      if (!recorded_hotplugs.empty()) {
        // pending_restore is the durable catalog for outputs absent from the launch baseline.
        // If one of those outputs is no longer available while another appears, retire the stale
        // allowlist entry before replacing the catalog. A later reconnect is classified again.
        std::vector<std::wstring> retired_preserved;
        std::erase_if(journal.exclusive_preserved, [&](const auto &identity) {
          const bool retire = !find_path(*journal.original_topology, identity) &&
                              std::ranges::none_of(recorded_hotplugs, [&](const auto &recorded) {
                                return same_device(recorded.identity, identity);
                              });
          if (retire) {
            retired_preserved.push_back(identity);
          }
          return retire;
        });
        if (journal.exclusive_topology) {
          for (size_t i = journal.exclusive_topology->paths.size(); i-- > 0;) {
            if (std::ranges::none_of(retired_preserved, [&](const auto &identity) {
                  return same_device(journal.exclusive_topology->device_paths[i], identity);
                })) {
              continue;
            }
            journal.exclusive_topology->paths.erase(journal.exclusive_topology->paths.begin() + static_cast<std::ptrdiff_t>(i));
            journal.exclusive_topology->device_paths.erase(journal.exclusive_topology->device_paths.begin() + static_cast<std::ptrdiff_t>(i));
            journal.exclusive_topology->colors.erase(journal.exclusive_topology->colors.begin() + static_cast<std::ptrdiff_t>(i));
          }
        }
      }
      std::ranges::sort(journal.exclusive_preserved, [](const auto &left, const auto &right) {
        return CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE) == CSTR_LESS_THAN;
      });

      if (!recorded_hotplugs.empty()) {
        std::ranges::sort(recorded_hotplugs, [](const auto &left, const auto &right) {
          return CompareStringOrdinal(left.identity.c_str(), -1, right.identity.c_str(), -1, TRUE) == CSTR_LESS_THAN;
        });
        auto virtual_spec = display_spec(*current, *virtual_index);
        virtual_spec.source.position = {0, 0};
        std::vector<display_spec_t> record;
        record.push_back(std::move(virtual_spec));
        std::int64_t x = record.front().source.width;
        for (auto &spec : recorded_hotplugs) {
          if (x > std::numeric_limits<LONG>::max() ||
              x + spec.source.width > std::numeric_limits<LONG>::max()) {
            return false;
          }
          spec.source.position = {static_cast<LONG>(x), 0};
          x += spec.source.width;
          record.push_back(std::move(spec));
        }
        const auto pending = build_topology(*available, record);
        if (!pending) {
          return false;
        }
        journal.pending_restore = *pending;
      }
      if (!exclusive_continuation_with_recorded_outputs(*current, journal, device_path)) {
        return false;
      }
      if (discovered_hotplug) {
        journal.before_exclusive = *current;
      }
      if (!valid_journal(journal) || !io_.save(journal)) {
        return false;
      }
      return promote_exclusive(device_path) && refresh_exclusive_cursor_clip(device_path);
    }

    bool manager_t::recover_inactive_exclusive() {
      const auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
        return false;
      }
      if (!loaded.journal || !loaded.journal->exclusive) {
        return true;
      }
      return restore();
    }

    bool manager_t::refresh_exclusive_cursor_clip(std::wstring_view device_path) {
      if (!io_.cursor_clip) {
        return true;
      }
      const auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
        return false;
      }
      const auto snapshot = io_.query();
      if (!snapshot) {
        return false;
      }
      if (loaded.journal) {
        const auto &journal = *loaded.journal;
        if (!journal.exclusive || !journal.exclusive_started || !same_device(journal.promoted_primary, device_path) ||
            !journal.exclusive_topology || !same_topology(*snapshot, *journal.exclusive_topology)) {
          return false;
        }
      } else if (snapshot->paths.size() != 1) {
        return false;
      }
      // With one active output Windows has nowhere else to move the shared cursor. Avoid taking
      // ownership of the process-global ClipCursor state until a preserved AR output is active.
      if (snapshot->paths.size() == 1) {
        return io_.cursor_clip(device_path, std::nullopt);
      }
      return clip_cursor_to_display(*snapshot, device_path);
    }

    bool manager_t::promote(std::wstring_view device_path, bool exclusive) {
      if (exclusive) {
        const bool promoted = promote_exclusive(device_path) && refresh_exclusive_cursor_clip(device_path);
        if (!promoted && io_.cursor_clip) {
          io_.cursor_clip(device_path, std::nullopt);
        }
        return promoted;
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
      // Cursor ownership is independent of journal readability. Release it even if display
      // recovery must wait; an old monitor identity cannot release a newer session's clip.
      if (io_.cursor_clip && !io_.cursor_clip(expected_device_path, std::nullopt)) {
        return false;
      }
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
      const auto preserve_current_desktop = [&]() {
        if (std::ranges::none_of(*layout, [&](const auto &position) {
              return !same_device(position.device_path, journal.promoted_primary);
            })) {
          // Keep recovery when the virtual source is the only usable desktop.
          return false;
        }
        const auto verified = io_.query();
        if (!verified || !same_topology(*snapshot, *verified)) {
          return false;
        }
        BOOST_LOG(info) << "Preserving the current physical desktop and retiring the superseded primary-display recovery record.";
        return io_.clear();
      };
      if (!original || !promoted) {
        return preserve_current_desktop();
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
        return preserve_current_desktop();
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
      if (exclusive && !exclusive_session_state.begin_promotion(device_path)) {
        BOOST_LOG(warning) << "Refusing to promote a different display while an exclusive display session is pending or active.";
        return false;
      }
      const bool promoted = acquire_ownership() && manager().promote(device_path, exclusive);
      if (exclusive && promoted && !exclusive_session_state.promotion_succeeded(device_path)) {
        return false;
      }
      return promoted;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Temporary primary-display promotion failed: " << exception.what();
      return false;
    }
  }

  bool prepare(bool exclusive) {
    std::lock_guard lock(transaction_mutex);
    try {
      const bool prepared = acquire_ownership() && manager().prepare(exclusive);
      if (exclusive && prepared) {
        exclusive_session_state.prepared();
      }
      return prepared;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not prepare primary-display recovery: " << exception.what();
      return false;
    }
  }

  bool prepare_local_exclusive(std::wstring_view sink_device_path) {
    if (sink_device_path.empty()) {
      return false;
    }
    std::lock_guard lock(transaction_mutex);
    try {
      const bool prepared = acquire_ownership() && manager().prepare(true, sink_device_path);
      if (prepared) {
        exclusive_session_state.prepared();
      }
      return prepared;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not prepare local AR display recovery: " << exception.what();
      return false;
    }
  }

  bool run_local_exclusive_update(std::wstring_view source_device_path, const std::function<bool()> &update) {
    // The callback can refresh the caller's display-name/path strings after Windows
    // renumbers a source. Keep the transaction identity independent of that storage.
    const std::wstring source_identity(source_device_path);
    std::lock_guard lock(transaction_mutex);
    try {
      if (!update || !exclusive_session_state.can_use_identity(source_identity) || !acquire_ownership() || !manager().is_local_exclusive(source_identity) || !exclusive_session_state.begin_promotion(source_identity)) {
        return false;
      }
      const bool updated = update();
      const bool promoted = manager().promote(source_identity, true);
      if (promoted) {
        exclusive_session_state.promotion_succeeded(source_identity);
      }
      return updated && promoted;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Local AR display update failed: " << exception.what();
      return false;
    }
  }

  bool bind_pending(std::wstring_view device_path) {
    std::lock_guard lock(transaction_mutex);
    try {
      if (exclusive_session_state.expected() && !exclusive_session_state.can_use_identity(device_path)) {
        BOOST_LOG(warning) << "Refusing to bind a different display to the pending exclusive display session.";
        return false;
      }
      const bool bound = acquire_ownership() && manager().bind_pending(device_path);
      if (bound) {
        exclusive_session_state.bound(device_path);
      }
      return bound;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not bind primary-display recovery identity: " << exception.what();
      return false;
    }
  }

  bool restore(std::wstring_view expected_device_path) {
    std::lock_guard lock(transaction_mutex);
    try {
      const auto action = exclusive_session_state.restore_action(expected_device_path);
      if (action == detail::exclusive_restore_action_e::reject) {
        return false;
      }
      if (action == detail::exclusive_restore_action_e::disarm_before) {
        exclusive_session_state.disarm();
      }
      const bool restored = acquire_ownership() && manager().restore(expected_device_path);
      exclusive_session_state.restore_finished(action, restored);
      return restored;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Temporary primary-display recovery failed: " << exception.what();
      return false;
    }
  }

  bool recover() {
    return restore();
  }

  bool pause(std::wstring_view device_path, retained_display_ptr &retained, std::wstring_view local_sink) {
    std::lock_guard lock(transaction_mutex);
    try {
      const auto action = exclusive_session_state.restore_action(device_path);
      if (action == detail::exclusive_restore_action_e::reject) {
        return false;
      }
      if (action == detail::exclusive_restore_action_e::disarm_before) {
        exclusive_session_state.disarm();
      }
      const bool restored = acquire_ownership() && manager().pause(device_path, retained, local_sink);
      exclusive_session_state.restore_finished(action, restored);
      return restored;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not pause the retained virtual desktop: " << exception.what();
      return false;
    }
  }

  bool reactivate(const retained_display_ptr &retained, bool exclusive) {
    std::lock_guard lock(transaction_mutex);
    try {
      if (!retained || !exclusive_session_state.can_use_identity(retained->display.identity) || !acquire_ownership()) {
        return false;
      }
      // Reconciliation must defer until the caller completes HDR/mode setup and promotion.
      // Failed activation keeps this identity pending until its exact pause/restore rollback.
      exclusive_session_state.prepared();
      exclusive_session_state.bound(retained->display.identity);
      const bool activated = manager().reactivate(retained, exclusive);
      if (activated && !exclusive) {
        exclusive_session_state.disarm();
      }
      return activated;
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not reactivate the retained virtual desktop: " << exception.what();
      return false;
    }
  }

  bool refresh_exclusive_cursor_clip() {
    std::lock_guard lock(transaction_mutex);
    try {
      if (!active_cursor_clip_identity) {
        return true;
      }
      if (!active_cursor_clip_bounds) {
        return false;
      }
      const auto identity = *active_cursor_clip_identity;
      const auto bounds = *active_cursor_clip_bounds;
      return set_exclusive_cursor_clip(identity, bounds);
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not refresh exclusive virtual-display cursor isolation: " << exception.what();
      return false;
    }
  }

  std::uint64_t exclusive_session_generation() {
    std::lock_guard lock(transaction_mutex);
    return exclusive_session_state.generation();
  }

  namespace {
    exclusive_reconcile_result_t reconcile_exclusive_display_topology_locked(
      std::optional<std::uint64_t> expected_generation
    ) {
      const auto generation = exclusive_session_state.generation();
      if (expected_generation && generation != *expected_generation) {
        return {exclusive_reconcile_result_e::settled, generation};
      }
      try {
        switch (exclusive_session_state.reconcile_action()) {
          case detail::exclusive_reconcile_action_e::defer:
            return {exclusive_reconcile_result_e::settled, generation};
          case detail::exclusive_reconcile_action_e::recover:
            // A crash-recovery journal can remain pending while a saved monitor is unplugged. A
            // later display notification is the earliest safe opportunity to finish that restore.
            return {
              acquire_ownership() && manager().recover_inactive_exclusive() ?
                exclusive_reconcile_result_e::settled :
                exclusive_reconcile_result_e::pending_recovery,
              generation,
            };
          case detail::exclusive_reconcile_action_e::reconcile:
            return {
              manager().reconcile_active_exclusive(*exclusive_session_state.active_identity()) ?
                exclusive_reconcile_result_e::settled :
                exclusive_reconcile_result_e::retry_active,
              generation,
            };
        }
      } catch (const std::exception &exception) {
        BOOST_LOG(error) << "Could not reassert exclusive virtual-display topology: " << exception.what();
        return {
          exclusive_session_state.reconcile_action() == detail::exclusive_reconcile_action_e::reconcile ?
            exclusive_reconcile_result_e::retry_active :
            exclusive_reconcile_result_e::pending_recovery,
          generation,
        };
      }
      return {exclusive_reconcile_result_e::pending_recovery, generation};
    }
  }  // namespace

  exclusive_reconcile_result_t reconcile_exclusive_display_topology() {
    std::lock_guard lock(transaction_mutex);
    return reconcile_exclusive_display_topology_locked(std::nullopt);
  }

  exclusive_reconcile_result_t reconcile_exclusive_display_topology(std::uint64_t expected_generation) {
    std::lock_guard lock(transaction_mutex);
    return reconcile_exclusive_display_topology_locked(expected_generation);
  }
}  // namespace platf::primary_display
