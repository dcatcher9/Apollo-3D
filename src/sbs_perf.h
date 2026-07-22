/**
 * @file src/sbs_perf.h
 * @brief Lightweight per-stage performance collector for the host SBS 3D pipeline.
 *
 * The perf benchmark (see docs/sbs-benchmark-plan.md): each pipeline stage pushes timing
 * samples in milliseconds; a rolling p50/p95/max summary is logged every N frames. The
 * offline harness can explicitly snapshot that window to JSON. Collection follows the global
 * `diagnostics` config switch; all entry points are cheap no-ops when disabled.
 *
 * Live samples from the sole active encode pipeline form one recent process-wide system-load
 * window. A small mutex guards the maps and explicit harness snapshots.
 *
 * GPU-stream stages (TensorRT depth/warp/inpaint inference) are measured with CUDA events
 * inside the estimator and the resolved elapsed-ms handed here via add_sample_ms(); this
 * module has no CUDA/D3D dependency of its own.
 */
#pragma once

#include <cstdint>
#include <string>

namespace sbs_perf {

  /// Turn collection on/off (from config at startup / mode switch). Off = every call is a no-op.
  void set_enabled(bool on);
  bool enabled();

  /// Record one timing sample for a named stage. `stage` must be a stable string literal.
  void add_sample_ms(const char *stage, double ms);

  /// Identify the current collection window. An explicit reset advances this value so late
  /// GPU-query results cannot contaminate a new offline benchmark run.
  std::uint64_t generation();

  /// Record only if `expected_generation` still names the active collection window.
  void add_sample_ms_if_current(const char *stage, double ms, std::uint64_t expected_generation);

  /// Call once per SBS convert(): advances the frame counter and, every summary_interval
  /// frames, logs a rolling p50/p95/max line. It never performs filesystem I/O.
  void tick();

  /// Drop all accumulated samples. Live diagnostics intentionally aggregate across active
  /// sessions; the offline benchmark harness resets before starting a measured run.
  void reset();

  /// Write the current per-stage summary to `path` as JSON. Returns true on success.
  /// Used by the offline benchmark harness to snapshot a run.
  bool dump_json(const std::string &path);

}  // namespace sbs_perf
