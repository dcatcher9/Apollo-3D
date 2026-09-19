// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <cstdint>

namespace sunshine_raw_scene {
  enum class stereo_basis { raw, projection };
  // Invalid depth cannot be rendered safely. Only the return to valid stereo
  // is gradual; this never blends an old depth/color frame into a new one.
  class reentry_transition {
  public:
    float update(bool ready, std::uint64_t now_ms, stereo_basis basis = stereo_basis::raw) noexcept {
      // Switching between estimated raw controls and validated projection
      // controls must not inherit full strength from the other numeric basis.
      // Physical depth-buffer rotation deliberately is not part of this key.
      if (basis != basis_) { *this = {}; basis_ = basis; }
      const bool backwards = have_time_ && now_ms < previous_ms_;
      const bool long_gap = have_ready_ && (now_ms < last_ready_ms_ || now_ms - last_ready_ms_ > 250);
      if (backwards || long_gap || !have_ready_) elapsed_ms_ = 0;
      // Credit only the interval actually spent displaying a ready frame.
      // A missing current frame still outputs mono, but a brief gap does not
      // erase the preceding ready exposure or credit its missing interval.
      else if (was_ready_)
        elapsed_ms_ = std::min<std::uint64_t>(500, elapsed_ms_ + std::min<std::uint64_t>(250, now_ms - previous_ms_));
      previous_ms_ = now_ms;
      have_time_ = true;
      was_ready_ = ready;
      if (ready) {
        last_ready_ms_ = now_ms;
        have_ready_ = true;
      }
      return ready ? float(elapsed_ms_) / 500.f : 0.f;
    }
  private:
    std::uint64_t previous_ms_{}, last_ready_ms_{}, elapsed_ms_{};
    stereo_basis basis_{stereo_basis::raw};
    bool was_ready_{}, have_ready_{}, have_time_{};
  };
}
