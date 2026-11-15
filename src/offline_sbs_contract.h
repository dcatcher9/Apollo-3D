/**
 * @file src/offline_sbs_contract.h
 * @brief Shared serialized-contract limits for native offline SBS jobs.
 */
#pragma once

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
  inline constexpr unsigned scene_cache_contract_schema = 5u;

  // The manager performs a bounded read with this exact limit. Keep the worker's
  // pre-publication check on the same contract so a successful child can always be
  // consumed by its parent.
  inline constexpr std::uintmax_t worker_result_max_bytes =
    16ull * 1024ull * 1024ull;
}  // namespace offline_sbs
