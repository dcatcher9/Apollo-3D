// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <windows.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <reshade.hpp>
#include "async_log.h"
#include "diagnostic_log_gate.h"
#include "game3d_vram.h"
#include "game3d_clock.h"

namespace sunshine_game3d {
  namespace slow_step_detail {
    // One log throttle per step name (the name's string literal address), so
    // nested scopes never starve each other: the inner scope destructs first
    // and must not consume the outer scope's slot. Lock-free; a full table
    // shares its last entry.
    struct throttle {
      std::atomic<const char *> name{nullptr};
      std::atomic<std::uint64_t> next_log{0};
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

  // One QueryPerformanceCounter interval, from construction to ms().
  class stopwatch {
  public:
    stopwatch() { QueryPerformanceCounter(&start_); }
    // Milliseconds since construction.
    double ms() const {
      LARGE_INTEGER end{};
      QueryPerformanceCounter(&end);
      return double(end.QuadPart - start_.QuadPart) * 1000.0 / frequency();
    }
    // When it started, in QueryPerformanceCounter milliseconds.
    double began_ms() const { return double(start_.QuadPart) * 1000.0 / frequency(); }

  private:
    static double frequency() {
      LARGE_INTEGER value{};
      QueryPerformanceFrequency(&value);
      return double(value.QuadPart);
    }
    LARGE_INTEGER start_{};
  };

  // Names a present-thread step that took long enough to be a visible hitch.
  // Quiet on the normal path: one QueryPerformanceCounter pair per step, and at
  // most one log line per second for each step name. Every scope over its
  // threshold can log, including an outer scope around an inner one. The line
  // says when the step began (QueryPerformanceCounter in ms), so the steps of
  // one stall nest by their intervals (game3d_log_report.py), never by when
  // the async log writer stamped them.
  class slow_step {
  public:
    explicit slow_step(const char *name, double threshold_ms = 8.0) : name_(name), threshold_ms_(threshold_ms) {}
    ~slow_step() {
      const double ms = watch_.ms();
      if (ms < threshold_ms_) return;
      auto &throttle = slow_step_detail::find(name_);
      if (!sunshine_diagnostics::claim_line(throttle.next_log, sunshine_game3d::clock::game_now(), 1000)) return;
      // The game process's latest video-memory sample (game3d_vram.h) follows.
      const vram::latest_suffix memory;
      char message[224];
      std::snprintf(message, sizeof(message), "Sunshine Game 3D hitch: %s took %.1f ms on the present thread (began %.1f ms)%s",
        name_, ms, watch_.began_ms(), memory.text);
      sunshine_log::message(reshade::log::level::warning, message);
    }
    slow_step(const slow_step &) = delete;
    slow_step &operator=(const slow_step &) = delete;

  private:
    const char *name_;
    double threshold_ms_;
    stopwatch watch_;
  };

  // A release of GPU objects that took ms: on a pool thread (pooled), one of
  // 8 ms or more is logged, at most once a second per kind of release (next,
  // the caller's own throttle), with what it released (what, textures,
  // bytes) and the game process's latest video-memory sample. On the present
  // thread the caller's slow_step names the step instead.
  inline void report_slow_release(std::atomic<std::uint64_t> &next, const char *what, bool pooled, double ms,
      unsigned textures, std::uint64_t bytes) {
    if (!pooled || ms < 8.0 || !sunshine_diagnostics::claim_line(next, clock::game_now(), 1000)) return;
    const vram::latest_suffix memory;
    char text[256];
    std::snprintf(text, sizeof(text), "Sunshine Game 3D: %s took %.1f ms off the present thread (%u textures, %.1f MiB)%s",
      what, ms, textures, double(bytes) / 1048576.0, memory.text);
    sunshine_log::message(reshade::log::level::info, text);
  }
}
