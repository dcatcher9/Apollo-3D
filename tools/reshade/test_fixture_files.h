// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// File staging shared by the runtime fixtures that disable their installed
// effect before the native tests (the native depth, native provider and both
// native runtime fixtures).

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <thread>

namespace sunshine_test_files {
  // How long a rename may wait for another process to let go of its source:
  // a freshly written file in the fixture's Temp output folder is briefly
  // held open by whoever scans or indexes it (the 10-27 full checks: four of
  // 27 runs failed "filesystem error: cannot rename ... SunshineGame3D.fx"
  // before any test ran). Two seconds outlast such a scan; a lock held longer
  // is a real failure.
  inline constexpr std::chrono::milliseconds busy_wait {2000};
  inline constexpr std::chrono::milliseconds busy_retry {25};

  // std::filesystem::rename, retried while from still exists and the rename
  // keeps failing, for at most busy_wait. A missing source fails at once, and
  // a rename that still fails after the wait throws the first error, so a
  // real failure is reported as before, only later.
  inline void rename_when_free(const std::filesystem::path &from, const std::filesystem::path &to) {
    std::error_code first;
    const auto until = std::chrono::steady_clock::now() + busy_wait;
    for (;;) {
      std::error_code error;
      std::filesystem::rename(from, to, error);
      if (!error) {
        return;
      }
      if (!first) {
        first = error;
      }
      std::error_code exists_error;
      if (!std::filesystem::exists(from, exists_error) || std::chrono::steady_clock::now() >= until) {
        throw std::filesystem::filesystem_error("cannot rename", from, to, first);
      }
      std::this_thread::sleep_for(busy_retry);
    }
  }
}  // namespace sunshine_test_files
