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

namespace sbs_bench {
  /// Entry point for the `--sbs-bench` subcommand. argc/argv are the post-flag args
  /// (see config::sunshine.cmd). Returns a process exit code.
  int run(int argc, char **argv);
}
