// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <cstdint>

namespace sunshine_raw_scene {
  enum class stereo_basis { raw, projection };
  // Invalid depth cannot be rendered safely. Only the return to valid stereo
  // is gradual; this never blends an old depth/color frame into a new one.
  // The ramp restarts after depth was lost: Presents shown without ready
  // depth for more than 250 ms since the last ready one, a clock rollback or
  // a basis change. A pause in Presents alone loses no depth: when the last
  // Present before it and the first after it both have ready depth on the
  // same basis, strength continues (owner ruling 10-27; until then any
  // Present gap over 250 ms restarted the ramp, so a game hitch showed its
  // first Present after the hitch flat although that frame's depth was ready).
  class reentry_transition {
  public:
    float update(bool ready, std::uint64_t now_ms, stereo_basis basis = stereo_basis::raw) noexcept {
      // Switching between estimated raw controls and validated projection
      // controls must not inherit full strength from the other numeric basis.
      // Physical depth-buffer rotation deliberately is not part of this key.
      if (basis != basis_) { *this = {}; basis_ = basis; }
      const bool backwards = have_time_ && now_ms < previous_ms_;
      // Only a Present shown without ready depth starts a loss; the previous
      // Present being ready means nothing between it and this one was lost.
      const bool long_gap = have_ready_ && !was_ready_ && (now_ms < last_ready_ms_ || now_ms - last_ready_ms_ > 250);
      if (backwards || long_gap || !have_ready_) elapsed_ms_ = 0;
      // Credit only the interval actually spent displaying a ready frame, at
      // most 250 ms of it (a pause in Presents earns no more). A missing
      // current frame still outputs mono, but a brief gap does not erase the
      // preceding ready exposure or credit its missing interval.
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
