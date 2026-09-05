/**
 * @file src/offline_sbs_contract.h
 * @brief Shared serialized-contract limits for native offline SBS jobs.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace offline_sbs {
  // Offline conversion is a selected-file frame feed. These serialized values make that
  // boundary machine-checkable: neither the active desktop window nor either live
  // window-region ROI authority may influence analysis or replay.
  inline constexpr std::string_view whole_clip_frame_source =
    "selected-input-only";
  inline constexpr std::string_view whole_clip_analysis_region =
    "full-frame";

  // Increment whenever a cached scene's serialized meaning or authenticated replay
  // dependencies change. Both the producer and consumer must use this single value.
  inline constexpr unsigned scene_cache_contract_schema = 3u;

  // The manager performs bounded reads with these exact limits. Keep the worker's
  // pre-publication checks on the same contract so a successful child can always be
  // consumed by its parent.
  inline constexpr std::uintmax_t worker_result_max_bytes =
    16ull * 1024ull * 1024ull;
  inline constexpr std::uintmax_t scene_audit_max_bytes =
    32ull * 1024ull * 1024ull;

  // Paged diagnostics bound memory independently of movie duration. These are
  // storage limits, not rendering or scene-detection policy.
  inline constexpr std::size_t scene_audit_page_max_scenes = 128;
  inline constexpr std::size_t scene_audit_recent_scenes = 32;
  inline constexpr std::uintmax_t scene_audit_page_max_bytes = 2ull * 1024 * 1024;
  inline constexpr std::uintmax_t scene_audit_storage_max_bytes = 1024ull * 1024 * 1024;
  inline constexpr std::size_t scene_audit_max_pages = 16384;
  inline constexpr std::size_t max_paged_scene_count =
    scene_audit_page_max_scenes * scene_audit_max_pages;

  // Legacy inline worker/audit documents remain readable for retained jobs. Their
  // old per-scene reserves constrain only those schemas, never new paged jobs.
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

  // Running manifest checkpoints are written at 1, 2, 4, ... scenes. Pages are
  // immutable and published independently; the compact index is rewritten only at
  // these checkpoints and completion.
  [[nodiscard]] inline constexpr bool is_scene_audit_checkpoint(
    const std::size_t scene_count
  ) noexcept {
    return scene_count != 0 &&
           (scene_count & (scene_count - 1)) == 0;
  }
}  // namespace offline_sbs
