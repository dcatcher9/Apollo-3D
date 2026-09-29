// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <windows.h>
#include <atomic>
#include <cstdio>
#include <reshade.hpp>
#include "async_log.h"

namespace sunshine_game3d {
  // Names a present-thread step that took long enough to be a visible hitch.
  // Quiet on the normal path: one QueryPerformanceCounter pair per step, and at
  // most one log line per second across all steps.
  class slow_step {
  public:
    explicit slow_step(const char *name, double threshold_ms = 8.0) : name_(name), threshold_ms_(threshold_ms) {
      QueryPerformanceCounter(&start_);
    }
    ~slow_step() {
      LARGE_INTEGER end{}, frequency{};
      QueryPerformanceCounter(&end);
      QueryPerformanceFrequency(&frequency);
      const double ms = double(end.QuadPart - start_.QuadPart) * 1000.0 / double(frequency.QuadPart);
      if (ms < threshold_ms_) return;
      static std::atomic<unsigned long long> next_log{0};
      const auto now = GetTickCount64();
      auto due = next_log.load(std::memory_order_relaxed);
      if (now < due || !next_log.compare_exchange_strong(due, now + 1000)) return;
      char message[160];
      std::snprintf(message, sizeof(message), "Sunshine Game 3D hitch: %s took %.1f ms on the present thread", name_, ms);
      sunshine_log::message(reshade::log::level::warning, message);
    }
    slow_step(const slow_step &) = delete;
    slow_step &operator=(const slow_step &) = delete;

  private:
    const char *name_;
    double threshold_ms_;
    LARGE_INTEGER start_{};
  };
}
