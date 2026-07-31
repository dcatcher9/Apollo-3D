/**
 * @file src/sbs_bench_harness.h
 * @brief Headless frame-fed SBS benchmark harness (the `--sbs-bench` subcommand).
 *
 * Runs the REAL depth estimator + the REAL SBS composite shaders over a fixed directory of
 * input frames and writes the resulting per-eye SBS PNGs, deterministically and without a
 * game or a connected client. This is the Tier-1 harness from docs/sbs-benchmark-plan.md:
 * it closes the offline-sim-vs-headset gap by exercising production code paths, and its
 * output is scored by tools/sbsbench/sbsbench.py.
 */
#pragma once

#ifdef SUNSHINE_TESTS
  #include <cstddef>
  #include <filesystem>
  #include <string>
  #include <vector>
#endif

namespace sbs_bench {
  /// Entry point for the `--sbs-bench` subcommand. argc/argv are the post-flag args
  /// (see config::sunshine.cmd). Returns a process exit code.
  int run(int argc, char **argv);

#ifdef SUNSHINE_TESTS
  struct source_time_validation_result {
    std::size_t frame_count = 0;
    double total_elapsed_seconds = 0.0;
    std::string file_sha256;
  };

  bool validate_scene_controller_source_time_for_test(
    const std::filesystem::path &path,
    const std::vector<std::string> &expected_frame_ids,
    source_time_validation_result &result,
    std::string &error
  );
#endif
}
