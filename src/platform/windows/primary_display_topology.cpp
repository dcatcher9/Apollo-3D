/**
 * @file src/platform/windows/primary_display_topology.cpp
 * @brief The pure topology solver behind temporary primary-display changes.
 */
#include "primary_display_topology.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace platf::primary_display::detail {
  namespace {
    using display_config::same_device_path;
    using display_config::same_luid;

    UINT32 target_index(const DISPLAYCONFIG_PATH_INFO &path) {
      return path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE ? path.targetInfo.targetModeInfoIdx : path.targetInfo.modeInfoIdx;
    }
  }  // namespace

  const detail::position_t *find_position(const detail::layout_t &layout, std::wstring_view identity) {
    const auto found = std::ranges::find_if(layout, [&](const auto &entry) {
      return same_device_path(entry.device_path, identity);
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
        if (same_device_path(entry.device_path, layout[j].device_path)) {
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

  bool same_modes(const detail::snapshot_t &left, const detail::snapshot_t &right) {
    if (left.paths.size() != right.paths.size()) {
      return false;
    }
    for (size_t i = 0; i < left.paths.size(); ++i) {
      const auto match = std::ranges::find_if(right.device_paths, [&](const auto &identity) {
        return same_device_path(identity, left.device_paths[i]);
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
      return same_device_path(path, identity);
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

  // Windows does not keep the reactivated layout when it re-evaluates the monitors after the
  // restored desktop wakes: it applies its own remembered layout for this monitor set, which
  // can switch a recorded physical output off and move and retime the virtual display into the
  // freed space. That layout is still this session's own. Every active output is a recorded
  // one, at least one recorded output was dropped, and the physical outputs that remain keep
  // their modes and their arrangement, so promoting loses no user arrangement and the recovery
  // record still restores every recorded output. With every recorded output still active, a
  // moved or edited output is a user edit and stays with the stricter rule above.
  bool resume_layout_replaced_by_windows(const detail::snapshot_t &current, const detail::snapshot_t &reference, std::wstring_view virtual_identity) {
    const auto live_layout = detail::inspect(current);
    const auto pending_layout = detail::inspect(reference);
    if (!live_layout || !pending_layout || !find_path(current, virtual_identity) || !find_path(reference, virtual_identity) ||
        current.paths.size() >= reference.paths.size()) {
      return false;
    }
    detail::snapshot_t live_physical, saved_physical;
    live_physical.modes = current.modes;
    saved_physical.modes = reference.modes;
    std::optional<std::pair<int64_t, int64_t>> translation;
    for (size_t i = 0; i < current.paths.size(); ++i) {
      const auto &identity = current.device_paths[i];
      if (same_device_path(identity, virtual_identity)) {
        continue;
      }
      const auto saved = find_path(reference, identity);
      const auto *live_position = find_position(*live_layout, identity);
      const auto *saved_position = find_position(*pending_layout, identity);
      if (!saved || !live_position || !saved_position) {
        return false;
      }
      const auto delta = std::pair {static_cast<int64_t>(live_position->x) - saved_position->x, static_cast<int64_t>(live_position->y) - saved_position->y};
      if (translation && *translation != delta) {
        return false;
      }
      translation = delta;
      live_physical.paths.push_back(current.paths[i]);
      live_physical.device_paths.push_back(identity);
      saved_physical.paths.push_back(reference.paths[*saved]);
      saved_physical.device_paths.push_back(reference.device_paths[*saved]);
    }
    return same_modes(live_physical, saved_physical);
  }

  display_spec_t display_spec(const detail::snapshot_t &snapshot, size_t index) {
    const auto &path = snapshot.paths[index];
    display_spec_t result {snapshot.device_paths[index], path, snapshot.modes[source_index(path)].sourceMode, snapshot.modes[target_index(path)].targetMode};
    if (path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE) {
      const auto desktop = path.targetInfo.desktopModeInfoIdx;
      if (desktop != DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID && desktop < snapshot.modes.size() && snapshot.modes[desktop].infoType == DISPLAYCONFIG_MODE_INFO_TYPE_DESKTOP_IMAGE) {
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
    // Callers may pass the identity of an element in this same vector. Sorting moves its
    // short-string storage, so retain the value before reordering the display owners.
    const std::wstring anchor_device(anchor_identity);
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
      return same_device_path(spec.identity, anchor_device);
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
      if (!same_device_path(current.device_paths[i], virtual_identity)) {
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
      if ((!keep_virtual_active && same_device_path(spec.identity, virtual_identity)) || std::ranges::any_of(preserved, [&](const auto &entry) {
            return same_device_path(entry.identity, spec.identity);
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

  std::optional<detail::snapshot_t> build_topology(const detail::snapshot_t &available, const std::vector<display_spec_t> &wanted, std::string_view *failure_reason) {
    const auto reject = [&](std::string_view reason) -> std::optional<detail::snapshot_t> {
      if (failure_reason) {
        *failure_reason = reason;
      }
      return std::nullopt;
    };
    if (wanted.empty() || wanted.size() > max_displays || available.paths.size() > max_available_paths || available.paths.size() != available.device_paths.size()) {
      return reject("route snapshot has invalid path counts");
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
        if (!path.targetInfo.targetAvailable || !same_device_path(wanted[i].identity, available.device_paths[p])) {
          continue;
        }
        if (target && (!same_luid(target->first, path.targetInfo.adapterId) || target->second != path.targetInfo.id)) {
          return reject("one required monitor identity maps to multiple available targets");
        }
        target = std::pair {path.targetInfo.adapterId, path.targetInfo.id};
        const source_key_t key {path.sourceInfo.adapterId.HighPart, path.sourceInfo.adapterId.LowPart, path.sourceInfo.id};
        const auto entry = sources.emplace(key, sources.size()).first;
        if (seen.insert(entry->second).second) {
          candidates[i].push_back({p, entry->second});
        }
      }
      if (candidates[i].empty()) {
        return reject("a required monitor has no available target route");
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
        return reject("available target routes cannot use distinct display sources");
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
    return detail::inspect(result) ? std::make_optional(std::move(result)) : reject("reconstructed display modes or layout are invalid");
  }

  bool park_beside_desktop(display_spec_t &parked, const std::vector<display_spec_t> &desktop) {
    if (desktop.empty()) {
      return false;
    }
    const auto anchor = std::ranges::max_element(desktop, {}, [](const auto &spec) {
      return static_cast<std::int64_t>(spec.source.position.x) + spec.source.width;
    });
    const auto right = static_cast<std::int64_t>(anchor->source.position.x) + anchor->source.width;
    if (right > std::numeric_limits<LONG>::max() || right + parked.source.width > std::numeric_limits<LONG>::max()) {
      return false;
    }
    parked.source.position = {static_cast<LONG>(right), anchor->source.position.y};
    return true;
  }

  // The physical sink of a monitor session shows only the presenter. Beside the desktop along a
  // whole edge, any pop-up that hangs over that edge lands on the sink above the presenter:
  // a tray menu of another program was measured 28 pixels over it. Windows then composes the
  // presenter instead of showing it directly, its images are no longer paced by the sink's
  // refresh, and local 3D loses its clock. So the sink is parked up and to the right of the
  // desktop, sharing a single pixel of edge at the top-right corner, which keeps the layout
  // connected for Windows and leaves nothing beside it for a window to hang over.
  bool park_sink_at_desktop_corner(display_spec_t &sink, const std::vector<display_spec_t> &desktop) {
    if (!park_beside_desktop(sink, desktop)) {
      return false;
    }
    constexpr LONG shared_edge_pixels = 1;
    const auto y = static_cast<std::int64_t>(sink.source.position.y) - sink.source.height + shared_edge_pixels;
    if (sink.source.height < static_cast<UINT32>(shared_edge_pixels) || y < std::numeric_limits<LONG>::min()) {
      return false;
    }
    sink.source.position.y = static_cast<LONG>(y);
    return true;
  }

  bool monitor_pair_placement_matches(const detail::snapshot_t &current, const detail::snapshot_t &reference,
    std::wstring_view source, std::wstring_view sink) {
    const auto current_source = find_path(current, source);
    const auto current_sink = find_path(current, sink);
    const auto saved_source = find_path(reference, source);
    const auto saved_sink = find_path(reference, sink);
    if (!current_source || !current_sink || !saved_source || !saved_sink) return false;
    const auto a = display_spec(current, *current_source).source.position;
    const auto b = display_spec(current, *current_sink).source.position;
    const auto x = display_spec(reference, *saved_source).source.position;
    const auto y = display_spec(reference, *saved_sink).source.position;
    return static_cast<std::int64_t>(a.x) - b.x == static_cast<std::int64_t>(x.x) - y.x &&
           static_cast<std::int64_t>(a.y) - b.y == static_cast<std::int64_t>(x.y) - y.y;
  }

  bool normalize_monitor_desktop(std::vector<display_spec_t> &wanted, std::wstring_view preferred) {
    auto primary = std::ranges::find_if(wanted, [](const auto &spec) {
      return spec.source.position.x == 0 && spec.source.position.y == 0;
    });
    if (primary == wanted.end()) {
      primary = std::ranges::find_if(wanted, [&](const auto &spec) {
        return same_device_path(spec.identity, preferred);
      });
    }
    if (primary == wanted.end()) {
      primary = wanted.begin();
    }
    if (primary == wanted.end()) {
      return false;
    }
    const auto origin = primary->source.position;
    for (auto &spec : wanted) {
      const auto x = subtract(spec.source.position.x, origin.x);
      const auto y = subtract(spec.source.position.y, origin.y);
      if (!x || !y) {
        return false;
      }
      spec.source.position = {*x, *y};
    }
    return true;
  }

  /** True when two desktop rectangles share any area. Touching edges do not overlap. */
  bool sources_overlap(const display_spec_t &first, const display_spec_t &second) {
    const auto &a = first.source;
    const auto &b = second.source;
    return static_cast<std::int64_t>(a.position.x) < static_cast<std::int64_t>(b.position.x) + b.width &&
           static_cast<std::int64_t>(b.position.x) < static_cast<std::int64_t>(a.position.x) + a.width &&
           static_cast<std::int64_t>(a.position.y) < static_cast<std::int64_t>(b.position.y) + b.height &&
           static_cast<std::int64_t>(b.position.y) < static_cast<std::int64_t>(a.position.y) + a.height;
  }

  bool any_sources_overlap(const std::vector<display_spec_t> &wanted) {
    for (size_t i = 0; i < wanted.size(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        if (sources_overlap(wanted[i], wanted[j])) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * True when every desktop can be reached from the first one through shared edge segments, as
   * Windows needs for one desktop. Corner contact alone does not connect. Empty is not connected.
   */
  bool sources_edge_connected(const std::vector<display_spec_t> &wanted) {
    if (wanted.empty()) {
      return false;
    }
    std::vector<bool> reached(wanted.size(), false);
    reached.front() = true;
    bool progress = true;
    while (progress) {
      progress = false;
      for (size_t i = 0; i < wanted.size(); ++i) {
        if (reached[i]) {
          continue;
        }
        const auto &a = wanted[i].source;
        const auto right_a = static_cast<std::int64_t>(a.position.x) + a.width;
        const auto bottom_a = static_cast<std::int64_t>(a.position.y) + a.height;
        for (size_t j = 0; j < wanted.size(); ++j) {
          if (!reached[j]) {
            continue;
          }
          const auto &b = wanted[j].source;
          const auto right_b = static_cast<std::int64_t>(b.position.x) + b.width;
          const auto bottom_b = static_cast<std::int64_t>(b.position.y) + b.height;
          const bool horizontal = (right_a == b.position.x || right_b == a.position.x) &&
                                  std::max(a.position.y, b.position.y) < std::min(bottom_a, bottom_b);
          const bool vertical = (bottom_a == b.position.y || bottom_b == a.position.y) &&
                                std::max(a.position.x, b.position.x) < std::min(right_a, right_b);
          if (horizontal || vertical) {
            reached[i] = true;
            progress = true;
            break;
          }
        }
      }
    }
    return std::ranges::all_of(reached, [](bool value) {
      return value;
    });
  }

  bool monitor_desktop_connected(const std::vector<display_spec_t> &wanted) {
    return !any_sources_overlap(wanted) && sources_edge_connected(wanted);
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

  bool color_matches(const display_config::advanced_color_state_t &a, const display_config::advanced_color_state_t &b) {
    return a.hdr_user_enabled == b.hdr_user_enabled && a.wcg_user_enabled == b.wcg_user_enabled && a.advanced_color_active == b.advanced_color_active && a.active_mode == b.active_mode;
  }

  bool same_colors(const detail::snapshot_t &left, const detail::snapshot_t &right) {
    if (left.paths.size() != right.paths.size() || left.colors.size() != left.paths.size() || right.colors.size() != right.paths.size()) {
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

  std::optional<detail::layout_t> restrict_layout(const detail::layout_t &recorded, const detail::layout_t &current, std::wstring_view virtual_identity) {
    // A crashed host can lose its virtual display before recovery. No other missing or
    // newly attached monitor is silently adopted into an old transaction.
    const bool virtual_missing = !find_position(current, virtual_identity);
    if (recorded.size() != current.size() + (virtual_missing ? 1 : 0)) {
      return std::nullopt;
    }
    detail::layout_t result;
    for (const auto &entry : recorded) {
      if (virtual_missing && same_device_path(entry.device_path, virtual_identity)) {
        continue;
      }
      if (!find_position(current, entry.device_path)) {
        return std::nullopt;
      }
      result.push_back(entry);
    }
    return result;
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
      if (target_mode.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_TARGET || target_mode.id != path.targetInfo.id || !same_luid(target_mode.adapterId, path.targetInfo.adapterId)) {
        return std::nullopt;
      }
      if (path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE) {
        const auto desktop_index = path.targetInfo.desktopModeInfoIdx;
        if (desktop_index != DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID && (desktop_index >= snapshot.modes.size() || snapshot.modes[desktop_index].infoType != DISPLAYCONFIG_MODE_INFO_TYPE_DESKTOP_IMAGE || snapshot.modes[desktop_index].id != path.targetInfo.id || !same_luid(snapshot.modes[desktop_index].adapterId, path.targetInfo.adapterId))) {
          return std::nullopt;
        }
      }
      const auto &mode = snapshot.modes[index];
      if (mode.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE || mode.id != path.sourceInfo.id || !same_luid(mode.adapterId, path.sourceInfo.adapterId) || mode.sourceMode.width == 0 || mode.sourceMode.height == 0 || static_cast<int64_t>(mode.sourceMode.position.x) + mode.sourceMode.width > std::numeric_limits<LONG>::max() || static_cast<int64_t>(mode.sourceMode.position.y) + mode.sourceMode.height > std::numeric_limits<LONG>::max()) {
        return std::nullopt;
      }
      for (size_t j = 0; j < i; ++j) {
        const auto &other = snapshot.paths[j];
        if ((same_luid(path.sourceInfo.adapterId, other.sourceInfo.adapterId) && path.sourceInfo.id == other.sourceInfo.id) || (same_luid(path.targetInfo.adapterId, other.targetInfo.adapterId) && path.targetInfo.id == other.targetInfo.id)) {
          return std::nullopt;
        }
      }
      result.push_back({snapshot.device_paths[i], mode.sourceMode.position.x, mode.sourceMode.position.y});
    }
    return valid_layout(result) ? std::make_optional(std::move(result)) : std::nullopt;
  }
}  // namespace platf::primary_display::detail
