// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>

namespace sunshine_diagnostics {
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
