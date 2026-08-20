/**
 * @file src/sbs_bench_harness.h
 * @brief Headless frame-fed SBS benchmark harness (the `--sbs-bench` subcommand).
 *
 * Runs the REAL depth estimator + the REAL SBS composite shaders over a fixed directory of
 * input frames and writes the resulting per-eye SBS PNGs, deterministically and without a
 * game or a connected client. This is the production-path harness documented in
 * tools/sbsbench/README.md:
 * it closes the offline-sim-vs-headset gap by exercising production code paths, and its
 * output is scored by tools/sbsbench/sbsbench.py.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace sbs_bench {
#ifdef _WIN32
  namespace detail {
    struct resolved_sbs_geometry {
      std::uint32_t eye_width = 0;
      std::uint32_t eye_height = 0;
      std::uint32_t sbs_width = 0;
      std::uint32_t sbs_height = 0;
      float content_scale_x = 0.0f;
      float content_scale_y = 0.0f;
    };

    /** Resolve and authenticate the packed output raster before any per-frame GPU work. */
    std::optional<resolved_sbs_geometry> resolve_sbs_geometry(
      std::uint32_t source_width,
      std::uint32_t source_height,
      int requested_eye_width,
      int requested_eye_height,
      double output_scale,
      int max_output_width,
      std::string &error
    );
  }  // namespace detail
#endif

  /// Entry point for the `--sbs-bench` subcommand. argc/argv are the post-flag args
  /// (see config::sunshine.cmd). Returns a process exit code.
  int run(int argc, char **argv);
}
