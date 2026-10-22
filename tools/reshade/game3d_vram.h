// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>

// The game process's video memory against its budget (docs/reshade-sbs.md,
// Diagnostics switch and per-Present cost). DXGI's QueryVideoMemoryInfo is a
// CPU-only kernel query of this process's own usage and the budget the OS
// grants it: the adapter's LOCAL segment group (dedicated VRAM) and its
// NON_LOCAL one (system memory the GPU reads). Above its local budget the OS
// demotes the process's allocations to system memory, which can stall the
// whole GPU. The present thread samples at most every sample_period_ms, never
// per Present; nothing that decides reads it. Readers on any thread take the
// latest sample from atomics for their own log lines (a depth loss, a hitch,
// a gap between Presents), and the 10 s timing window takes the window's
// extremes for its own `Sunshine Game 3D VRAM` line.
namespace sunshine_game3d::vram {
  inline constexpr std::uint64_t sample_period_ms = 250;
  // A sample after a longer interval (the game froze or stopped presenting)
  // counts at most this much of it as time over budget.
  inline constexpr std::uint64_t max_counted_interval_ms = 4 * sample_period_ms;
  inline constexpr std::uint64_t mib = 1024ull * 1024ull;

  struct reading {
    std::uint32_t local_usage_mib = 0, local_budget_mib = 0;
    std::uint32_t nonlocal_usage_mib = 0, nonlocal_budget_mib = 0;
    bool valid = false;
  };

  inline std::uint32_t to_mib(std::uint64_t bytes) noexcept {
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(bytes / mib, UINT32_MAX));
  }

  // One timing window's samples: the largest local usage, the smallest local
  // budget and headroom (budget minus usage, negative over budget), the time
  // attributed to samples over budget, and how many samples saw the local
  // budget change from the sample before (across windows too).
  struct window {
    std::uint32_t samples = 0, max_usage_mib = 0, min_budget_mib = 0, budget_changes = 0;
    std::int64_t min_headroom_mib = 0;
    std::uint64_t over_budget_ms = 0;
    reading last;
  };

  class accumulator {
  public:
    void add(const reading &value, std::uint64_t now_ms) noexcept {
      if (!value.valid) return;
      const auto headroom = std::int64_t(value.local_budget_mib) - std::int64_t(value.local_usage_mib);
      if (!window_.samples) {
        window_.max_usage_mib = value.local_usage_mib;
        window_.min_budget_mib = value.local_budget_mib;
        window_.min_headroom_mib = headroom;
      } else {
        window_.max_usage_mib = std::max(window_.max_usage_mib, value.local_usage_mib);
        window_.min_budget_mib = std::min(window_.min_budget_mib, value.local_budget_mib);
        window_.min_headroom_mib = std::min(window_.min_headroom_mib, headroom);
      }
      if (previous_.valid && previous_.local_budget_mib != value.local_budget_mib) ++window_.budget_changes;
      if (headroom < 0 && previous_.valid && now_ms > previous_ms_)
        window_.over_budget_ms += std::min(now_ms - previous_ms_, max_counted_interval_ms);
      ++window_.samples;
      window_.last = previous_ = value;
      previous_ms_ = now_ms;
    }
    window take() noexcept {
      const auto out = window_;
      window_ = {};
      return out;
    }

  private:
    window window_;
    reading previous_;
    std::uint64_t previous_ms_ = 0;
  };

  // " vram_local_mib=usage/budget nonlocal_mib=usage" for an event line, or
  // "" before the first sample.
  inline void format_suffix(char *out, std::size_t size, const reading &value) noexcept {
    if (!size) return;
    out[0] = '\0';
    if (value.valid)
      std::snprintf(out, size, " vram_local_mib=%u/%u nonlocal_mib=%u", value.local_usage_mib, value.local_budget_mib,
        value.nonlocal_usage_mib);
  }

  // The window's line; false (nothing to log) without a sample in it.
  inline bool format_window_line(char *out, std::size_t size, const window &w, unsigned long long window_ms) noexcept {
    if (!size || !w.samples) return false;
    std::snprintf(out, size,
      "Sunshine Game 3D VRAM: local_mib={usage=%u budget=%u} nonlocal_mib={usage=%u budget=%u} window={samples=%u "
      "max_usage_mib=%u min_budget_mib=%u min_headroom_mib=%lld over_budget_ms=%llu budget_changes=%u} window_ms=%llu",
      w.last.local_usage_mib, w.last.local_budget_mib, w.last.nonlocal_usage_mib, w.last.nonlocal_budget_mib, w.samples,
      w.max_usage_mib, w.min_budget_mib, static_cast<long long>(w.min_headroom_mib),
      static_cast<unsigned long long>(w.over_budget_ms), w.budget_changes, window_ms);
    return true;
  }

  // The process's sampler. due() is one relaxed load; the caller that finds a
  // sample due queries DXGI (outside its own locks) and records it here.
  class sampler {
  public:
    bool due(std::uint64_t now_ms) const noexcept { return now_ms >= next_ms_.load(std::memory_order_relaxed); }
    void record(const reading &value, std::uint64_t now_ms) noexcept {
      std::lock_guard<std::mutex> lock(mutex_);
      next_ms_.store(now_ms + sample_period_ms, std::memory_order_relaxed);
      if (!value.valid) return;
      local_.store(pack(value.local_usage_mib, value.local_budget_mib), std::memory_order_relaxed);
      nonlocal_.store(pack(value.nonlocal_usage_mib, value.nonlocal_budget_mib), std::memory_order_relaxed);
      valid_.store(true, std::memory_order_release);
      accumulator_.add(value, now_ms);
    }
    reading latest() const noexcept {
      reading out;
      if (!valid_.load(std::memory_order_acquire)) return out;
      const auto local = local_.load(std::memory_order_relaxed), nonlocal = nonlocal_.load(std::memory_order_relaxed);
      out.local_usage_mib = std::uint32_t(local >> 32);
      out.local_budget_mib = std::uint32_t(local);
      out.nonlocal_usage_mib = std::uint32_t(nonlocal >> 32);
      out.nonlocal_budget_mib = std::uint32_t(nonlocal);
      out.valid = true;
      return out;
    }
    window take_window() noexcept {
      std::lock_guard<std::mutex> lock(mutex_);
      return accumulator_.take();
    }

  private:
    static std::uint64_t pack(std::uint32_t high, std::uint32_t low) noexcept {
      return (std::uint64_t(high) << 32) | low;
    }
    std::atomic<std::uint64_t> next_ms_{0}, local_{0}, nonlocal_{0};
    std::atomic<bool> valid_{false};
    std::mutex mutex_;
    accumulator accumulator_;
  };

  inline sampler &process() noexcept {
    static sampler value;
    return value;
  }

  // The latest sample as an event line's suffix.
  struct latest_suffix {
    char text[64]{};
    latest_suffix() noexcept { format_suffix(text, sizeof(text), process().latest()); }
  };
}  // namespace sunshine_game3d::vram
