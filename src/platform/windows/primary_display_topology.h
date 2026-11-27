/**
 * @file src/platform/windows/primary_display_topology.h
 * @brief The pure topology solver behind temporary primary-display changes: desktop layouts,
 * named CCD snapshots and the topologies built from them. Nothing here queries or changes a
 * display, so every rule can be tested with in-memory snapshots.
 */
#pragma once

#include "display_config.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

namespace platf::primary_display::detail {

  /** The most displays one recorded layout or topology may hold. */
  inline constexpr size_t max_displays = 128;

  /** The most paths one observation of every available route (QDC_ALL_PATHS) may hold. */
  inline constexpr size_t max_available_paths = 16384;

  struct position_t {
    std::wstring device_path;
    LONG x = 0;
    LONG y = 0;
  };

  using layout_t = std::vector<position_t>;

  /** A named CCD observation and each active target's color state, in path order. */
  struct snapshot_t: display_config::named_topology_t {
    std::vector<std::optional<display_config::advanced_color_state_t>> colors;
  };

  /** One output as a topology wants it: its route, source and target modes, desktop image and
   * color. A snapshot path gives one; build_topology maps it onto an available route.
   */
  struct display_spec_t {
    std::wstring identity;
    DISPLAYCONFIG_PATH_INFO path {};
    DISPLAYCONFIG_SOURCE_MODE source {};
    DISPLAYCONFIG_TARGET_MODE target {};
    std::optional<DISPLAYCONFIG_MODE_INFO> desktop;
    std::optional<display_config::advanced_color_state_t> color;
  };

  // Layouts: desktop positions by device path.

  /** The entry for this device path, or null. */
  const position_t *find_position(const layout_t &layout, std::wstring_view identity);
  /** 1 to max_displays distinct, named entries with exactly one at the desktop origin. */
  bool valid_layout(const layout_t &layout);
  /** The same device paths at the same positions, in any order. */
  bool same_layout(const layout_t &left, const layout_t &right);
  /** value - offset, or nothing outside LONG. */
  std::optional<LONG> subtract(LONG value, LONG offset);
  /** Every position moved so that (x, y) becomes the origin, or nothing on overflow. */
  std::optional<layout_t> translated(const layout_t &layout, LONG x, LONG y);
  /** The recorded layout for the current outputs. Only the virtual display may be missing. */
  std::optional<layout_t> restrict_layout(const layout_t &recorded, const layout_t &current, std::wstring_view virtual_identity);

  // Snapshots: comparisons and ownership tests of CCD observations.

  /** The active desktop layout of a consistent snapshot, or nothing. */
  std::optional<layout_t> inspect(const snapshot_t &snapshot);
  /** A path's source-mode index, honouring virtual-mode-aware indexes. */
  UINT32 source_index(const DISPLAYCONFIG_PATH_INFO &path);
  /** The index of the path with this device path. */
  std::optional<size_t> find_path(const snapshot_t &snapshot, std::wstring_view identity);
  /** The same outputs with the same source, target-signal and desktop-image modes. */
  bool same_modes(const snapshot_t &left, const snapshot_t &right);
  /** The same layout and modes. Two empty snapshots match. */
  bool same_topology(const snapshot_t &a, const snapshot_t &b);
  /** The same HDR, WCG, active Advanced Color and active color mode. */
  bool color_matches(const display_config::advanced_color_state_t &a, const display_config::advanced_color_state_t &b);
  /** The same outputs, each with a matching color or none. */
  bool same_colors(const snapshot_t &left, const snapshot_t &right);
  bool same_layout_and_physical_modes(const snapshot_t &current, const snapshot_t &reference, std::wstring_view virtual_identity);
  bool resume_layout_replaced_by_windows(const snapshot_t &current, const snapshot_t &reference, std::wstring_view virtual_identity);
  /** Every current output is a reference one with its modes, all moved by one offset. */
  bool owned_subset(const snapshot_t &current, const snapshot_t &reference);
  /** The virtual display at the origin with the preserved outputs packed to its left. */
  bool owned_exclusive_state(
    const snapshot_t &current,
    std::wstring_view virtual_identity,
    const std::vector<std::wstring> &preserved_identities,
    const snapshot_t &original_topology
  );
  /** The virtual display at the origin and every output at its reference position. */
  bool owned_exclusive_layout(const snapshot_t &current, std::wstring_view virtual_identity, const snapshot_t &reference);
  /** The source and sink keep their recorded offset from each other. */
  bool monitor_pair_placement_matches(const snapshot_t &current, const snapshot_t &reference, std::wstring_view source, std::wstring_view sink);

  // Building a topology from wanted outputs.

  display_spec_t display_spec(const snapshot_t &snapshot, size_t index);
  /** One row ordered by position, placed so the anchor lands at (anchor_x, anchor_y). */
  bool pack_horizontally(std::vector<display_spec_t> &displays, std::wstring_view anchor_identity, LONG anchor_x, LONG anchor_y);
  bool preserve_current_desktop(const snapshot_t &current, std::vector<display_spec_t> &wanted, std::wstring_view virtual_identity, bool keep_virtual_active);
  /** Route every wanted output to a distinct available source and build the CCD topology.
   * A refusal names its reason through failure_reason when one is supplied.
   */
  std::optional<snapshot_t> build_topology(const snapshot_t &available, const std::vector<display_spec_t> &wanted, std::string_view *failure_reason = nullptr);
  /** Right of the rightmost desktop output, aligned with its top edge. */
  bool park_beside_desktop(display_spec_t &parked, const std::vector<display_spec_t> &desktop);
  bool park_sink_at_desktop_corner(display_spec_t &sink, const std::vector<display_spec_t> &desktop);
  /** Move the output at the origin, else the preferred one, else the first, to the origin. */
  bool normalize_monitor_desktop(std::vector<display_spec_t> &wanted, std::wstring_view preferred);
  bool sources_overlap(const display_spec_t &first, const display_spec_t &second);
  bool any_sources_overlap(const std::vector<display_spec_t> &wanted);
  bool sources_edge_connected(const std::vector<display_spec_t> &wanted);
  /** No two outputs overlap and every one shares an edge path with the first. */
  bool monitor_desktop_connected(const std::vector<display_spec_t> &wanted);

}  // namespace platf::primary_display::detail
