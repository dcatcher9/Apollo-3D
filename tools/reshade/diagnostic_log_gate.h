// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>

namespace sunshine_diagnostics {
  // A routing family is stable across physical source rotation and missing
  // current capture. Observe native presents, not effect/accessor call counts.
  // This state only decides when to log; it never admits or rejects rendering.
  class family_gap_observer {
  public:
    bool observe(std::uint64_t present, std::uint64_t routing_epoch, bool eligible, bool ready) noexcept {
      if (routing_epoch != epoch_ || !eligible || !routing_epoch || (have_present_ && present < present_)) {
        epoch_ = routing_epoch;
        mask_ = 0;
        count_ = 0;
      }
      if (have_present_ && present == present_) return recurring();
      present_ = present;
      have_present_ = true;
      if (!present || !eligible || !routing_epoch) return false;
      mask_ = static_cast<std::uint16_t>((std::uint32_t(mask_) << 1) | (ready ? 0u : 1u));
      if (count_ < 16) ++count_;
      return recurring();
    }
    unsigned observations() const noexcept { return count_; }
    unsigned misses() const noexcept {
      auto bits = mask_;
      unsigned count = 0;
      while (bits) { bits &= bits - 1; ++count; }
      return count;
    }
    bool recurring() const noexcept { return count_ == 16 && misses() >= 8; }

  private:
    std::uint64_t epoch_{}, present_{};
    std::uint16_t mask_{};
    unsigned count_{};
    bool have_present_{};
  };

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
