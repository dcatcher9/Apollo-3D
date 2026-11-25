// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <cstdint>

namespace sunshine_depth {
  enum class depth_orientation : unsigned { automatic = 0, normal = 1, reversed = 2 };

  // Evidence from real depth clears, not the legacy "any clear != 1" UI hint.
  // Multiple command lists/clears in one frame count as one observation. Mixed
  // endpoints and non-endpoint clears cannot establish a projection convention.
  class depth_orientation_evidence {
  public:
    void observe(std::uint64_t frame, bool zero, bool one, bool other) {
      if (frame <= last_frame_ || (!zero && !one && !other))
        return;
      last_frame_ = frame;
      const auto current = !other && zero != one ?
        (zero ? depth_orientation::reversed : depth_orientation::normal) : depth_orientation::automatic;
      if (current == depth_orientation::automatic) {
        pending_ = current;
        count_ = 0;
      } else if (pending_ != current) {
        pending_ = current;
        count_ = 1;
      } else {
        count_ = std::min(count_ + 1, 3u);
      }
    }
    depth_orientation detected() const { return count_ >= 3 ? pending_ : depth_orientation::automatic; }
    unsigned agreeing_frames() const { return count_; }
    std::uint64_t last_frame() const { return last_frame_; }

  private:
    depth_orientation pending_ = depth_orientation::automatic;
    unsigned count_ = 0;
    std::uint64_t last_frame_ = 0;
  };
}
