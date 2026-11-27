// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <atomic>
#include <cstdint>

namespace sunshine_diagnostics {
  // The rate limit of one log line that several threads may write: true at
  // most once per interval_ms (the first call included), for the one caller
  // whose compare-exchange claims the slot. next holds when the line is due
  // again (zero: now). Lock-free; it guards no data, so relaxed order is enough.
  inline bool claim_line(std::atomic<std::uint64_t> &next, std::uint64_t now_ms, std::uint64_t interval_ms) noexcept {
    auto due = next.load(std::memory_order_relaxed);
    return now_ms >= due && next.compare_exchange_strong(due, now_ms + interval_ms, std::memory_order_relaxed);
  }

  // Only bounds log writes. Callers still update rendering and UI every frame.
  class log_gate {
  public:
    bool due(std::uint64_t now_ms, bool changed, bool immediate = false, bool periodic = false) noexcept {
      if (!started_ || immediate || now_ms < last_ms_ ||
          (changed && now_ms - last_ms_ >= 1000) ||
          (periodic && now_ms - last_ms_ >= 10000)) {
        started_ = true;
        last_ms_ = now_ms;
        return true;
      }
      return false;
    }

  private:
    std::uint64_t last_ms_{};
    bool started_{};
  };
}
