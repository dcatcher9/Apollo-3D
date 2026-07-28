/**
 * @file src/offline_sbs_contract.h
 * @brief Shared serialized-contract limits for native offline SBS jobs.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace offline_sbs {
  // The manager performs bounded reads with these exact limits. Keep the worker's
  // pre-publication checks on the same contract so a successful child can always be
  // consumed by its parent.
  inline constexpr std::uintmax_t worker_result_max_bytes =
    16ull * 1024ull * 1024ull;
  inline constexpr std::uintmax_t scene_audit_max_bytes =
    32ull * 1024ull * 1024ull;

  // Reserve one MiB for fixed document fields (source/timeline/cache metadata) and
  // conservatively budget the remaining bytes per finalized scene. A conversion result
  // carries both a scene record and a replay record; an audit carries a scene record and
  // its boundary record. Exact serialized-size checks remain authoritative because one
  // boundary can contain several merged proposals.
  inline constexpr std::uintmax_t serialized_contract_fixed_reserve_bytes =
    1ull * 1024ull * 1024ull;
  inline constexpr std::uintmax_t worker_result_scene_reserve_bytes =
    8ull * 1024ull;
  inline constexpr std::uintmax_t scene_audit_scene_reserve_bytes =
    16ull * 1024ull;

  inline constexpr std::size_t max_serialized_scene_count =
    static_cast<std::size_t>(std::min(
      (worker_result_max_bytes - serialized_contract_fixed_reserve_bytes) /
        worker_result_scene_reserve_bytes,
      (scene_audit_max_bytes - serialized_contract_fixed_reserve_bytes) /
        scene_audit_scene_reserve_bytes
    ));

  // Running snapshots are written at 1, 2, 4, ... scenes. The sum of all snapshot
  // prefixes is less than twice the final scene count, eliminating the previous
  // rewrite-after-every-scene quadratic write amplification. Completion is always
  // published separately, so the final audit still contains every record.
  [[nodiscard]] inline constexpr bool is_scene_audit_checkpoint(
    const std::size_t scene_count
  ) noexcept {
    return scene_count != 0 &&
           (scene_count & (scene_count - 1)) == 0;
  }
}  // namespace offline_sbs
