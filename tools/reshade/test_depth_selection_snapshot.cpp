// SPDX-License-Identifier: GPL-3.0-only
#include <Windows.h>
#include "depth_selection_snapshot_test.h"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <utility>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: reshade_depth_selection_snapshot_tests <SunshineSBSTest.addon64>\n");
    return 2;
  }
  // Load the same test add-on as the native fixtures, without AddonInit,
  // ReShade registration, a game process, device creation or GPU work.
  HMODULE addon = LoadLibraryA(argv[1]);
  if (!addon) {
    std::fprintf(stderr, "Failed to load test add-on: %lu\n", GetLastError());
    return 2;
  }
  using test_fn = BOOL (*)(unsigned, unsigned, unsigned, sunshine_depth_snapshot_test_result *);
  const auto test = reinterpret_cast<test_fn>(GetProcAddress(addon, "SunshineDepthTestSnapshot"));
  if (!test) {
    std::fprintf(stderr, "Test-only snapshot entry point is unavailable\n");
    FreeLibrary(addon);
    return 2;
  }
  bool passed = true;
  for (const auto workload : {std::pair{0u, 0u}, std::pair{1u, 0u}, std::pair{16u, 8u},
        std::pair{64u, 32u}, std::pair{256u, 64u}}) {
    sunshine_depth_snapshot_test_result result;
    if (!test(workload.first, workload.second, 128, &result)) {
      std::fprintf(stderr, "FAIL depth selection snapshot: %s\n", result.error);
      passed = false;
      break;
    }
    std::sort(std::begin(result.deep_ns), std::end(result.deep_ns));
    std::sort(std::begin(result.compact_ns), std::end(result.compact_ns));
    std::printf("SNAPSHOT buffers=%u clears_per_buffer=%u iterations=128 rounds=%u "
      "deep_median_ns=%.1f compact_median_ns=%.1f deep_range_ns=%.1f..%.1f compact_range_ns=%.1f..%.1f "
      "deep_value_bytes=%llu compact_value_bytes=%llu clear_records=%llu checksum=%llu\n",
      workload.first, workload.second, result.rounds,
      result.deep_ns[result.rounds / 2], result.compact_ns[result.rounds / 2],
      result.deep_ns[0], result.deep_ns[result.rounds - 1], result.compact_ns[0], result.compact_ns[result.rounds - 1],
      static_cast<unsigned long long>(result.deep_value_bytes), static_cast<unsigned long long>(result.compact_value_bytes),
      static_cast<unsigned long long>(result.clear_records), static_cast<unsigned long long>(result.checksum));
  }
  FreeLibrary(addon);
  if (!passed) return 1;
  std::puts("PASS actual depth-selector snapshot parity, independent ownership, resource lifetime/crop changes and inactive manual evidence retention. "
    "CPU timing includes snapshot construction, a common checksum visit and destruction; value bytes exclude allocator/node/bucket overhead. No GPU or FPS measurement.");
  return 0;
}
