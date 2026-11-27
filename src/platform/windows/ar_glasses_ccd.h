/**
 * @file src/platform/windows/ar_glasses_ccd.h
 * @brief CCD topology helpers for local AR: complete active-path snapshots, source refresh
 * readback, the planned virtual-source/physical-sink row, and restoring one output's position.
 */
#pragma once

#include "display_config.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

namespace ar_glasses {
  namespace internal {
    bool physical_adapter_contract_valid(const LUID &before, const LUID &after);
    std::optional<float> query_sdr_white_nits(const LUID &adapter_id, UINT32 target_id);

    struct linear_layout_t {
      RECT virtual_rect {};
      RECT physical_rect {};
    };

    struct display_target_identity_t {
      LUID adapter_id {};
      UINT32 target_id = 0;
    };

    struct active_source_t {
      std::size_t path_index = 0;
      std::size_t mode_index = 0;
      LUID source_adapter_id {};
      UINT32 source_id = 0;
      display_target_identity_t target;
      std::wstring gdi_name;
      std::wstring device_path;
      std::wstring friendly_name;
      RECT rect {};
    };

    struct topology_snapshot_t: platf::display_config::topology_t {
      std::vector<active_source_t> sources;
    };

    std::optional<topology_snapshot_t> query_complete_topology_snapshot();
    int source_refresh_millihz(std::uint32_t numerator, std::uint32_t denominator);
    bool source_refresh_matches(int actual_millihz, int requested_millihz);
    bool same_topology_snapshot(
      const topology_snapshot_t &left,
      const topology_snapshot_t &right
    );

    struct pre_add_isolation_plan_t {
      RECT physical_rect {};
      linear_layout_t layout;
    };

    std::optional<pre_add_isolation_plan_t> build_pre_add_isolation_plan(
      std::wstring_view target_device_path
    );

    struct isolation_plan_t {
      topology_snapshot_t snapshot;
      std::size_t physical_index = 0;
      std::size_t virtual_index = 0;
      display_target_identity_t virtual_identity;
      linear_layout_t layout;
    };

    std::optional<isolation_plan_t> build_isolation_plan(
      std::wstring_view virtual_gdi_name,
      const std::optional<display_target_identity_t> &virtual_identity,
      std::wstring_view target_device_path
    );
    bool layout_is_exact_and_adjacent(const isolation_plan_t &plan);
    std::optional<isolation_plan_t> query_verified_isolation(
      const display_target_identity_t &virtual_identity,
      std::wstring_view target_device_path,
      const linear_layout_t &expected
    );
    bool restore_physical_output_position(
      const RECT &original_rect,
      const std::wstring &target_device_path
    );
  }  // namespace internal
}  // namespace ar_glasses
