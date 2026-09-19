// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>

// TEST ONLY C ABI. The production add-on does not export this entry point.
struct sunshine_depth_snapshot_test_result {
  static constexpr unsigned rounds = 9;
  std::uint64_t checksum = 0;
  std::uint64_t deep_value_bytes = 0, compact_value_bytes = 0, clear_records = 0;
  double deep_ns[rounds] {}, compact_ns[rounds] {};
  char error[256] {};
};
