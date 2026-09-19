// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstdint>

namespace sunshine_depth_stats {
  // Describes the depth available to an effects pass, not whether its color
  // originated from a rendered or generated game frame.
  enum class presentation_kind { fresh, reused, unavailable };

  struct presentation_statistics {
    std::uint64_t fresh{}, reused{}, unavailable{}, window_ms{};
    std::uint64_t total() const { return fresh + reused + unavailable; }
  };

  // Five completed one-second buckets. Excluding the current bucket keeps the
  // overlay stable between updates; no rendering-thread allocation is needed.
  // The provider owns resets when its source or viewing interval changes.
  class presentation_window {
  public:
    void reset() { *this = {}; }

    void observe(std::uint64_t present, std::uint64_t now_ms, presentation_kind kind) {
      if (started_ && (now_ms < last_tick_ || present < last_present_)) reset();
      if (!started_) {
        started_ = true;
        first_tick_ = now_ms;
      }
      else if (present == last_present_) {
        // Multiple effect passes in one presentation are one observation.
        last_tick_ = now_ms;
        return;
      }
      last_tick_ = now_ms;
      last_present_ = present;
      const auto epoch = (now_ms - first_tick_) / bucket_ms;
      auto &bucket = buckets_[epoch % buckets_.size()];
      if (!bucket.occupied || bucket.epoch != epoch) bucket = {epoch, {}, true};
      switch (kind) {
        case presentation_kind::fresh: ++bucket.counts.fresh; break;
        case presentation_kind::reused: ++bucket.counts.reused; break;
        case presentation_kind::unavailable: ++bucket.counts.unavailable; break;
      }
    }

    presentation_statistics snapshot(std::uint64_t now_ms) const {
      presentation_statistics out;
      if (!started_ || now_ms < last_tick_) return out;
      const auto completed = (now_ms - first_tick_) / bucket_ms;
      out.window_ms = (completed < window_buckets ? completed : window_buckets) * bucket_ms;
      for (const auto &bucket : buckets_) {
        if (!bucket.occupied || bucket.epoch >= completed || completed - bucket.epoch > window_buckets) continue;
        out.fresh += bucket.counts.fresh;
        out.reused += bucket.counts.reused;
        out.unavailable += bucket.counts.unavailable;
      }
      return out;
    }

  private:
    static constexpr std::uint64_t bucket_ms = 1000, window_buckets = 5;
    struct bucket {
      std::uint64_t epoch{};
      presentation_statistics counts;
      bool occupied{};
    };
    // The sixth slot holds observations still arriving in the current second.
    std::array<bucket, window_buckets + 1> buckets_{};
    std::uint64_t first_tick_{}, last_tick_{}, last_present_{};
    bool started_{};
  };
}
