/**
 * @file src/sbs_perf.h
 * @brief Lightweight per-stage performance collector for the host SBS 3D pipeline.
 *
 * The perf benchmark (see docs/sbs-benchmark-plan.md): each pipeline stage pushes timing
 * samples in milliseconds; a rolling p50/p95/max summary is logged every N frames and
 * mirrored to a JSON file so a change's perf delta is a number, not a vibe. Enabled by the
 * `sbs_3d_perf_stats` config knob; all entry points are cheap no-ops when disabled.
 *
 * Samples are expected from the encode thread (display_vram::convert and the estimator it
 * drives); a small internal mutex still guards the maps so an off-thread dump is safe.
 *
 * GPU-stream stages (TensorRT depth/warp/inpaint inference) are measured with CUDA events
 * inside the estimator and the resolved elapsed-ms handed here via add_sample_ms(); this
 * module has no CUDA/D3D dependency of its own.
 */
#pragma once

#include <string>

namespace sbs_perf {

  /// Turn collection on/off (from config at startup / mode switch). Off = every call is a no-op.
  void set_enabled(bool on);
  bool enabled();

  /// Record one timing sample for a named stage. `stage` must be a stable string literal.
  void add_sample_ms(const char *stage, double ms);

  /// Call once per SBS convert(): advances the frame counter and, every summary_interval
  /// frames, logs a rolling p50/p95/max line and writes the JSON snapshot.
  void tick();

  /// Drop all accumulated samples (e.g. on an SBS mode / depth-model switch, so the window
  /// reflects the new configuration rather than blending across it).
  void reset();

  /// Write the current per-stage summary to `path` as JSON. Returns true on success.
  /// Used by the offline benchmark harness to snapshot a run; the live path also writes to a
  /// default location each summary (see the .cpp).
  bool dump_json(const std::string &path);

}  // namespace sbs_perf
