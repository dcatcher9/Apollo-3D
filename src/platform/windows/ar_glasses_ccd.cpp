/**
 * @file src/platform/windows/ar_glasses_ccd.cpp
 * @brief CCD snapshots, source refresh readback, the planned source/sink row and one output's
 * position restore.
 */
#include "ar_glasses_ccd.h"

#include "ar_glasses.h"
#include "ar_glasses_devices.h"
#include "misc.h"
#include "src/logging.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

namespace ar_glasses {
  using namespace std::literals;
  using namespace internal;

  namespace {
    linear_layout_t compute_linear_layout(
      const RECT &anchor,
      LONG virtual_width,
      LONG virtual_height,
      LONG physical_width,
      LONG physical_height
    ) {
      linear_layout_t layout;
      layout.virtual_rect = {
        anchor.right,
        anchor.top,
        anchor.right + virtual_width,
        anchor.top + virtual_height,
      };
      layout.physical_rect = {
        layout.virtual_rect.right,
        anchor.top,
        layout.virtual_rect.right + physical_width,
        anchor.top + physical_height,
      };
      return layout;
    }

    bool same_target_identity(
      const display_target_identity_t &left,
      const display_target_identity_t &right
    ) {
      return same_luid(left.adapter_id, right.adapter_id) && left.target_id == right.target_id;
    }

    bool same_source_identity(const active_source_t &left, const active_source_t &right) {
      return same_luid(left.source_adapter_id, right.source_adapter_id) &&
             left.source_id == right.source_id;
    }

    std::optional<std::vector<std::string>> topology_snapshot_fingerprint(
      const topology_snapshot_t &snapshot
    ) {
      if (snapshot.sources.size() != snapshot.paths.size()) {
        return std::nullopt;
      }

      std::vector<std::string> entries;
      entries.reserve(snapshot.sources.size());
      for (const auto &source : snapshot.sources) {
        if (source.path_index >= snapshot.paths.size() || source.mode_index >= snapshot.modes.size()) {
          return std::nullopt;
        }
        const auto &path = snapshot.paths[source.path_index];
        const auto &source_mode_info = snapshot.modes[source.mode_index];
        if (source_mode_info.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          return std::nullopt;
        }
        const auto &source_mode = source_mode_info.sourceMode;

        std::ostringstream entry;
        entry << path.sourceInfo.adapterId.HighPart << ':' << path.sourceInfo.adapterId.LowPart
              << ':' << path.sourceInfo.id << ':' << path.sourceInfo.statusFlags
              << '>' << path.targetInfo.adapterId.HighPart << ':' << path.targetInfo.adapterId.LowPart
              << ':' << path.targetInfo.id << ':' << path.targetInfo.statusFlags
              << ':' << (unsigned int) path.targetInfo.outputTechnology
              << ':' << (unsigned int) path.targetInfo.rotation
              << ':' << (unsigned int) path.targetInfo.scaling
              << ':' << path.targetInfo.refreshRate.Numerator
              << '/' << path.targetInfo.refreshRate.Denominator
              << ':' << (unsigned int) path.targetInfo.scanLineOrdering
              << ':' << path.targetInfo.targetAvailable
              << ':' << path.flags
              << "|src=" << source_mode.position.x << ',' << source_mode.position.y
              << ':' << source_mode.width << 'x' << source_mode.height
              << ':' << (unsigned int) source_mode.pixelFormat
              << "|gdi=" << platf::to_utf8(source.gdi_name)
              << "|pnp=" << platf::to_utf8(source.device_path);

        const auto target_mode_index = path.targetInfo.modeInfoIdx;
        if (target_mode_index == DISPLAYCONFIG_PATH_MODE_IDX_INVALID) {
          entry << "|target=none";
        } else {
          if (target_mode_index >= snapshot.modes.size() || snapshot.modes[target_mode_index].infoType != DISPLAYCONFIG_MODE_INFO_TYPE_TARGET) {
            return std::nullopt;
          }
          const auto &signal = snapshot.modes[target_mode_index].targetMode.targetVideoSignalInfo;
          entry << "|target=" << signal.pixelRate
                << ':' << signal.hSyncFreq.Numerator << '/' << signal.hSyncFreq.Denominator
                << ':' << signal.vSyncFreq.Numerator << '/' << signal.vSyncFreq.Denominator
                << ':' << signal.activeSize.cx << 'x' << signal.activeSize.cy
                << ':' << signal.totalSize.cx << 'x' << signal.totalSize.cy
                << ':' << signal.videoStandard
                << ':' << (unsigned int) signal.scanLineOrdering;
        }
        entries.emplace_back(entry.str());
      }
      std::sort(entries.begin(), entries.end());
      return entries;
    }

    bool source_is_cloned(
      const std::vector<active_source_t> &sources,
      const active_source_t &selected
    ) {
      return std::ranges::any_of(sources, [&](const auto &candidate) {
        return !same_target_identity(candidate.target, selected.target) &&
               same_source_identity(candidate, selected);
      });
    }

    struct anchor_candidate_t {
      RECT rect {};
      LUID source_adapter_id {};
      UINT32 source_id = 0;
    };

    bool anchor_is_better(const anchor_candidate_t &candidate, const anchor_candidate_t &current) {
      if (candidate.rect.right != current.rect.right) {
        return candidate.rect.right > current.rect.right;
      }
      if (candidate.rect.top != current.rect.top) {
        return candidate.rect.top < current.rect.top;
      }
      if (candidate.rect.left != current.rect.left) {
        return candidate.rect.left < current.rect.left;
      }
      if (candidate.source_adapter_id.HighPart != current.source_adapter_id.HighPart) {
        return candidate.source_adapter_id.HighPart < current.source_adapter_id.HighPart;
      }
      if (candidate.source_adapter_id.LowPart != current.source_adapter_id.LowPart) {
        return candidate.source_adapter_id.LowPart < current.source_adapter_id.LowPart;
      }
      return candidate.source_id < current.source_id;
    }

    std::optional<anchor_candidate_t> select_anchor(
      const std::vector<active_source_t> &sources,
      const active_source_t &physical,
      const active_source_t &virtual_source
    ) {
      std::optional<anchor_candidate_t> selected;
      for (const auto &source : sources) {
        if (same_source_identity(source, physical) || same_source_identity(source, virtual_source)) {
          continue;
        }
        anchor_candidate_t candidate {source.rect, source.source_adapter_id, source.source_id};
        if (!selected || anchor_is_better(candidate, *selected)) {
          selected = candidate;
        }
      }
      return selected;
    }

    std::optional<anchor_candidate_t> select_pre_add_anchor(
      const std::vector<active_source_t> &sources,
      const active_source_t &physical
    ) {
      std::optional<anchor_candidate_t> selected;
      for (const auto &source : sources) {
        if (same_source_identity(source, physical)) {
          continue;
        }
        anchor_candidate_t candidate {source.rect, source.source_adapter_id, source.source_id};
        if (!selected || anchor_is_better(candidate, *selected)) {
          selected = candidate;
        }
      }
      return selected;
    }
  }  // namespace

  namespace internal {
    bool physical_adapter_contract_valid(const LUID &before, const LUID &after) {
      return same_luid(before, after);
    }

    std::optional<float> query_sdr_white_nits(const LUID &adapter_id, UINT32 target_id) {
      DISPLAYCONFIG_SDR_WHITE_LEVEL white_level {};
      white_level.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
      white_level.header.size = sizeof(white_level);
      white_level.header.adapterId = adapter_id;
      white_level.header.id = target_id;
      if (DisplayConfigGetDeviceInfo(&white_level.header) != ERROR_SUCCESS) {
        return std::nullopt;
      }
      // Windows encodes SDR reference white relative to 80 nits, with 1000 == 80 nits.
      return (float) white_level.SDRWhiteLevel * 80.0f / 1000.0f;
    }

    std::optional<topology_snapshot_t> query_complete_topology_snapshot() {
      topology_snapshot_t snapshot;
      if (!platf::display_config::query_topology(QDC_ONLY_ACTIVE_PATHS, snapshot)) {
        return std::nullopt;
      }

      snapshot.sources.reserve(snapshot.paths.size());
      for (std::size_t path_index = 0; path_index < snapshot.paths.size(); ++path_index) {
        const auto &path = snapshot.paths[path_index];
        if (path.sourceInfo.modeInfoIdx == DISPLAYCONFIG_PATH_MODE_IDX_INVALID || path.sourceInfo.modeInfoIdx >= snapshot.modes.size() || snapshot.modes[path.sourceInfo.modeInfoIdx].infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          return std::nullopt;
        }

        auto gdi_name = platf::display_config::query_source_name(path.sourceInfo.adapterId, path.sourceInfo.id);
        if (!gdi_name) {
          return std::nullopt;
        }
        auto target_name = platf::display_config::query_target_name(path.targetInfo.adapterId, path.targetInfo.id);
        if (!target_name) {
          return std::nullopt;
        }

        const auto &mode = snapshot.modes[path.sourceInfo.modeInfoIdx].sourceMode;
        active_source_t source;
        source.path_index = path_index;
        source.mode_index = path.sourceInfo.modeInfoIdx;
        source.source_adapter_id = path.sourceInfo.adapterId;
        source.source_id = path.sourceInfo.id;
        source.target = {path.targetInfo.adapterId, path.targetInfo.id};
        source.gdi_name = std::move(*gdi_name);
        source.device_path = std::move(target_name.device_path);
        source.friendly_name = std::move(target_name.friendly_name);
        source.rect = {
          mode.position.x,
          mode.position.y,
          mode.position.x + (LONG) mode.width,
          mode.position.y + (LONG) mode.height,
        };
        snapshot.sources.emplace_back(std::move(source));
      }
      return snapshot;
    }

    int source_refresh_millihz(std::uint32_t numerator, std::uint32_t denominator) {
      if (!numerator || !denominator) {
        return 0;
      }
      const auto rounded = (static_cast<std::uint64_t>(numerator) * 1000 + denominator / 2) / denominator;
      return rounded <= std::numeric_limits<int>::max() ? static_cast<int>(rounded) : 0;
    }

    bool source_refresh_matches(int actual_millihz, int requested_millihz) {
      if (actual_millihz <= 0 || requested_millihz <= 0) {
        return false;
      }
      const auto slower = std::min(actual_millihz, requested_millihz);
      const auto difference = static_cast<std::int64_t>(std::max(actual_millihz, requested_millihz)) - slower;
      // Accept at most 0.02% clock variation without rounding either CCD measurement. This
      // includes 120 vs 120.013 Hz while keeping 59.94/60 and 119.88/120 Hz distinct.
      return difference * 5000 <= slower;
    }

    bool same_topology_snapshot(
      const topology_snapshot_t &left,
      const topology_snapshot_t &right
    ) {
      const auto left_fingerprint = topology_snapshot_fingerprint(left);
      const auto right_fingerprint = topology_snapshot_fingerprint(right);
      return left_fingerprint && right_fingerprint && *left_fingerprint == *right_fingerprint;
    }

    std::optional<pre_add_isolation_plan_t> build_pre_add_isolation_plan(
      std::wstring_view target_device_path
    ) {
      const auto snapshot = query_complete_topology_snapshot();
      if (!snapshot) {
        return std::nullopt;
      }

      std::wstring primary_gdi_name;
      for (DWORD index = 0;; ++index) {
        DISPLAY_DEVICEW device {};
        device.cb = sizeof(device);
        if (!EnumDisplayDevicesW(nullptr, index, &device, 0)) {
          break;
        }
        if ((device.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE) != 0) {
          primary_gdi_name = device.DeviceName;
          break;
        }
      }
      if (primary_gdi_name.empty()) {
        return std::nullopt;
      }

      const active_source_t *physical = nullptr;
      const active_source_t *primary = nullptr;
      for (const auto &source : snapshot->sources) {
        if (equal_device_paths(source.device_path, target_device_path)) {
          if (physical) {
            return std::nullopt;
          }
          physical = &source;
        }
        if (equal_device_paths(source.gdi_name, primary_gdi_name)) {
          primary = &source;
        }
      }
      if (!physical || !primary || same_source_identity(*physical, *primary) || source_is_cloned(snapshot->sources, *physical)) {
        return std::nullopt;
      }

      const auto anchor = select_pre_add_anchor(snapshot->sources, *physical);
      if (!anchor) {
        return std::nullopt;
      }
      const auto physical_width = physical->rect.right - physical->rect.left;
      const auto physical_height = physical->rect.bottom - physical->rect.top;
      return pre_add_isolation_plan_t {
        .physical_rect = physical->rect,
        .layout = compute_linear_layout(
          anchor->rect,
          source_width,
          source_height,
          physical_width,
          physical_height
        ),
      };
    }

    std::optional<isolation_plan_t> build_isolation_plan(
      std::wstring_view virtual_gdi_name,
      const std::optional<display_target_identity_t> &virtual_identity,
      std::wstring_view target_device_path
    ) {
      auto snapshot = query_complete_topology_snapshot();
      if (!snapshot) {
        return std::nullopt;
      }

      std::wstring primary_gdi_name;
      for (DWORD index = 0;; ++index) {
        DISPLAY_DEVICEW device {};
        device.cb = sizeof(device);
        if (!EnumDisplayDevicesW(nullptr, index, &device, 0)) {
          break;
        }
        if ((device.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE) != 0) {
          primary_gdi_name = device.DeviceName;
          break;
        }
      }
      if (primary_gdi_name.empty()) {
        return std::nullopt;
      }

      std::optional<std::size_t> physical_index;
      std::optional<std::size_t> virtual_index;
      std::optional<std::size_t> primary_index;
      for (std::size_t index = 0; index < snapshot->sources.size(); ++index) {
        const auto &source = snapshot->sources[index];
        if (equal_device_paths(source.device_path, target_device_path)) {
          if (physical_index) {
            return std::nullopt;
          }
          physical_index = index;
        }
        const bool is_virtual = virtual_identity ?
                                  same_target_identity(source.target, *virtual_identity) :
                                  equal_device_paths(source.gdi_name, virtual_gdi_name);
        if (is_virtual) {
          if (virtual_index) {
            return std::nullopt;
          }
          virtual_index = index;
        }
        if (equal_device_paths(source.gdi_name, primary_gdi_name)) {
          primary_index = index;
        }
      }
      if (!physical_index || !virtual_index || !primary_index || *physical_index == *virtual_index) {
        return std::nullopt;
      }

      const auto &physical = snapshot->sources[*physical_index];
      const auto &virtual_source = snapshot->sources[*virtual_index];
      const auto &primary = snapshot->sources[*primary_index];
      if (same_source_identity(physical, primary)) {
        BOOST_LOG(warning) << "The selected AR display is the Windows primary display; automatic pointer isolation is unavailable."sv;
        return std::nullopt;
      }
      if (source_is_cloned(snapshot->sources, physical)) {
        BOOST_LOG(warning) << "The selected AR display shares a cloned desktop source; refusing to move the clone group for pointer isolation."sv;
        return std::nullopt;
      }
      if (source_is_cloned(snapshot->sources, virtual_source)) {
        BOOST_LOG(warning) << "The private AR virtual desktop unexpectedly shares a cloned source; refusing to move the clone group."sv;
        return std::nullopt;
      }
      const auto anchor = select_anchor(snapshot->sources, physical, virtual_source);
      if (!anchor) {
        return std::nullopt;
      }

      const auto virtual_width = virtual_source.rect.right - virtual_source.rect.left;
      const auto virtual_height = virtual_source.rect.bottom - virtual_source.rect.top;
      const auto physical_width = physical.rect.right - physical.rect.left;
      const auto physical_height = physical.rect.bottom - physical.rect.top;
      isolation_plan_t plan;
      plan.physical_index = *physical_index;
      plan.virtual_index = *virtual_index;
      plan.virtual_identity = virtual_source.target;
      plan.layout = compute_linear_layout(
        anchor->rect,
        virtual_width,
        virtual_height,
        physical_width,
        physical_height
      );
      plan.snapshot = std::move(*snapshot);
      return plan;
    }

    bool layout_is_exact_and_adjacent(const isolation_plan_t &plan) {
      const auto &physical = plan.snapshot.sources[plan.physical_index].rect;
      const auto &virtual_source = plan.snapshot.sources[plan.virtual_index].rect;
      return same_rect(virtual_source, plan.layout.virtual_rect) &&
             same_rect(physical, plan.layout.physical_rect) &&
             virtual_source.right == physical.left &&
             virtual_source.top == physical.top &&
             virtual_source.bottom == physical.bottom;
    }

    std::optional<isolation_plan_t> query_verified_isolation(
      const display_target_identity_t &virtual_identity,
      std::wstring_view target_device_path,
      const linear_layout_t &expected
    ) {
      auto plan = build_isolation_plan({}, virtual_identity, target_device_path);
      if (!plan) {
        return std::nullopt;
      }
      plan->layout = expected;
      return layout_is_exact_and_adjacent(*plan) ? std::move(plan) : std::nullopt;
    }

    bool restore_physical_output_position(
      const RECT &original_rect,
      const std::wstring &target_device_path
    ) {
      auto snapshot = query_complete_topology_snapshot();
      if (!snapshot) {
        return false;
      }

      const auto physical = std::ranges::find_if(snapshot->sources, [&](const auto &source) {
        return equal_device_paths(source.device_path, target_device_path);
      });
      if (physical == snapshot->sources.end() || source_is_cloned(snapshot->sources, *physical)) {
        return false;
      }
      if (same_rect(physical->rect, original_rect)) {
        return true;
      }

      // This snapshot is intentionally acquired after all recovery-file I/O. Validate and apply it
      // immediately so an old full CCD image cannot overwrite an unrelated monitor adjustment.
      const auto validated_baseline = *snapshot;
      auto &source_mode = snapshot->modes[physical->mode_index].sourceMode;
      source_mode.position.x = original_rect.left;
      source_mode.position.y = original_rect.top;

      bool fresh_physical_found = true;
      const auto restore = platf::display_config::validate_and_apply(*snapshot, 0, 0, [&](platf::display_config::topology_t &supplied) {
        auto fresh_snapshot = query_complete_topology_snapshot();
        if (!fresh_snapshot || !same_topology_snapshot(validated_baseline, *fresh_snapshot)) {
          return false;
        }
        const auto fresh_physical = std::ranges::find_if(fresh_snapshot->sources, [&](const auto &source) {
          return equal_device_paths(source.device_path, target_device_path);
        });
        if (fresh_physical == fresh_snapshot->sources.end() || source_is_cloned(fresh_snapshot->sources, *fresh_physical)) {
          fresh_physical_found = false;
          return false;
        }
        auto &fresh_source_mode = fresh_snapshot->modes[fresh_physical->mode_index].sourceMode;
        fresh_source_mode.position.x = original_rect.left;
        fresh_source_mode.position.y = original_rect.top;
        supplied = std::move(*fresh_snapshot);
        return true;
      });
      switch (restore.state) {
        case platf::display_config::supplied_apply_e::invalid:
          BOOST_LOG(warning) << "Validating the AR display-position restore failed with status "sv
                             << restore.status << '.';
          return false;
        case platf::display_config::supplied_apply_e::stale:
          if (fresh_physical_found) {
            BOOST_LOG(info) << "Desktop topology changed after AR restore validation; preserving the newer layout."sv;
          }
          return false;
        case platf::display_config::supplied_apply_e::failed:
          BOOST_LOG(warning) << "Restoring the AR display position failed with status "sv << restore.status << '.';
          return false;
        case platf::display_config::supplied_apply_e::applied:
          break;
      }
      for (int attempt = 0; attempt < 10; ++attempt) {
        if (const auto actual = find_target(target_device_path)) {
          if (same_rect(actual->rect, original_rect)) {
            return true;
          }
        }
        std::this_thread::sleep_for(50ms);
      }
      return false;
    }
  }  // namespace internal

#ifdef SUNSHINE_TESTS
  int detail::local_source_refresh_millihz_for_test(std::uint32_t numerator, std::uint32_t denominator) {
    return source_refresh_millihz(numerator, denominator);
  }

  bool detail::physical_adapter_contract_valid_for_test(
    const LUID &physical_before,
    const LUID &physical_after,
    const LUID &virtual_adapter
  ) {
    (void) virtual_adapter;
    return physical_adapter_contract_valid(physical_before, physical_after);
  }

  detail::linear_layout_t detail::compute_linear_layout_for_test(
    const RECT &anchor,
    LONG virtual_width,
    LONG virtual_height,
    LONG physical_width,
    LONG physical_height
  ) {
    const auto layout = compute_linear_layout(
      anchor,
      virtual_width,
      virtual_height,
      physical_width,
      physical_height
    );
    return {layout.virtual_rect, layout.physical_rect};
  }

  std::optional<detail::anchor_candidate_t> detail::select_anchor_for_test(
    const std::vector<detail::anchor_candidate_t> &candidates
  ) {
    std::optional<::ar_glasses::anchor_candidate_t> selected;
    for (const auto &candidate : candidates) {
      ::ar_glasses::anchor_candidate_t converted {
        candidate.rect,
        candidate.source_adapter_id,
        candidate.source_id,
      };
      if (!selected || anchor_is_better(converted, *selected)) {
        selected = converted;
      }
    }
    if (!selected) {
      return std::nullopt;
    }
    return detail::anchor_candidate_t {
      selected->rect,
      selected->source_adapter_id,
      selected->source_id,
    };
  }

  bool detail::source_is_cloned_for_test(
    const std::vector<detail::topology_path_identity_t> &paths,
    std::size_t selected_index
  ) {
    if (selected_index >= paths.size()) {
      return false;
    }
    std::vector<active_source_t> sources;
    sources.reserve(paths.size());
    for (const auto &path : paths) {
      active_source_t source;
      source.source_adapter_id = path.source_adapter_id;
      source.source_id = path.source_id;
      source.target = {path.target_adapter_id, path.target_id};
      sources.emplace_back(source);
    }
    return source_is_cloned(sources, sources[selected_index]);
  }

  bool detail::isolated_layout_matches_for_test(
    const detail::linear_layout_t &expected,
    const RECT &virtual_rect,
    const RECT &physical_rect
  ) {
    isolation_plan_t plan;
    plan.snapshot.sources.resize(2);
    plan.virtual_index = 0;
    plan.physical_index = 1;
    plan.snapshot.sources[0].rect = virtual_rect;
    plan.snapshot.sources[1].rect = physical_rect;
    plan.layout = {expected.virtual_rect, expected.physical_rect};
    return layout_is_exact_and_adjacent(plan);
  }
#endif
}  // namespace ar_glasses
