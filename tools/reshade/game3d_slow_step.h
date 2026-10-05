// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <windows.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <reshade.hpp>
#include "async_log.h"

namespace sunshine_game3d {
  namespace slow_step_detail {
    // One log throttle per step name (the name's string literal address), so
    // nested scopes never starve each other: the inner scope destructs first
    // and must not consume the outer scope's slot. Lock-free; a full table
    // shares its last entry.
    struct throttle {
      std::atomic<const char *> name{nullptr};
      std::atomic<unsigned long long> next_log{0};
    };
    inline constexpr unsigned throttle_count = 64;
    inline throttle *table() {
      static throttle values[throttle_count];
      return values;
    }
    inline throttle &find(const char *name) {
      auto *values = table();
      const auto start = static_cast<unsigned>((reinterpret_cast<std::uintptr_t>(name) >> 3) % throttle_count);
      for (unsigned i = 0; i != throttle_count; ++i) {
        auto &entry = values[(start + i) % throttle_count];
        const char *current = entry.name.load(std::memory_order_acquire);
        if (current == name) return entry;
        if (!current) {
          const char *expected = nullptr;
          if (entry.name.compare_exchange_strong(expected, name, std::memory_order_acq_rel) || expected == name)
            return entry;
        }
      }
      return values[throttle_count - 1];
    }
  }  // namespace slow_step_detail

  // Names a present-thread step that took long enough to be a visible hitch.
  // Quiet on the normal path: one QueryPerformanceCounter pair per step, and at
  // most one log line per second for each step name. Every scope over its
  // threshold can log, including an outer scope around an inner one.
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
      auto &throttle = slow_step_detail::find(name_);
      const auto now = GetTickCount64();
      auto due = throttle.next_log.load(std::memory_order_relaxed);
      if (now < due || !throttle.next_log.compare_exchange_strong(due, now + 1000)) return;
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
