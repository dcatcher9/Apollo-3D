// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

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

  struct depth_calibration_key {
    std::uint64_t source_id = 0, layout_epoch = 0;
    std::uint32_t source_width = 0, source_height = 0;
    std::uint32_t x = 0, y = 0, width = 0, height = 0;
    bool operator==(const depth_calibration_key &other) const {
      return source_id == other.source_id && layout_epoch == other.layout_epoch &&
        source_width == other.source_width && source_height == other.source_height &&
        x == other.x && y == other.y && width == other.width && height == other.height;
    }
  };

  struct depth_calibration_result {
    bool calibrated = false;
    float raw_anchor = 0, raw_gain = 0;
    unsigned samples = 0;
    std::uint64_t sampled_frame = 0;
  };

  // Calibrates a dimensionless affine coordinate in RAW hardware depth. This is
  // not camera-space distance, a log histogram, or perspective linearization.
  // Three fresh useful captures establish the median anchor and robust 10–90%
  // span; those values then stay fixed until source/layout invalidation. Stereo
  // strength and screen-plane controls deliberately are not inputs here.
  class native_depth_calibration {
  public:
    void reset(const depth_calibration_key &key) {
      *this = native_depth_calibration {};
      key_ = key;
    }
    bool matches(const depth_calibration_key &key) const { return key_ == key; }
    std::uint64_t last_sample_frame() const { return last_frame_; }

    bool sample(const depth_calibration_key &key, std::uint64_t frame,
                const float *raw, std::size_t count, bool useful = true) {
      if (!(key_ == key))
        reset(key);
      if (key.source_id == 0 || frame <= last_frame_)
        return false;
      last_frame_ = frame;
      // A flat/menu frame cannot move a calibrated scene's zero plane or scale.
      if (count_ == 3)
        return true;
      const auto reject = [this]() { count_ = 0; return false; };
      if (!useful || raw == nullptr || count < 16)
        return reject();
      std::vector<float> supported;
      supported.reserve(count);
      std::size_t finite = 0;
      for (std::size_t i = 0; i != count; ++i) {
        const float value = raw[i];
        if (std::isfinite(value) && value >= 0 && value <= 1) {
          ++finite;
          if (value > 0 && value < 1)
            supported.push_back(value); // Exact clear endpoints do not set scene scale.
        }
      }
      if (finite < count - count / 10 || supported.size() < std::max<std::size_t>(16, count / 8))
        return reject();
      std::sort(supported.begin(), supported.end());
      const auto quantile = [&supported](double q) {
        const double pos = q * static_cast<double>(supported.size() - 1);
        const auto lo = static_cast<std::size_t>(pos);
        const auto hi = std::min(lo + 1, supported.size() - 1);
        return static_cast<double>(supported[lo]) +
          (static_cast<double>(supported[hi]) - supported[lo]) * (pos - lo);
      };
      // Subtract in double: raw FP32's small depth differences near one must
      // survive calibration without a fixed epsilon, FP16, or log conversion.
      const double span = quantile(.9) - quantile(.1);
      if (!(span > 0) || !std::isfinite(1.0 / span) || 1.0 / span > std::numeric_limits<float>::max())
        return reject();
      anchors_[count_] = quantile(.5);
      spans_[count_] = span;
      if (++count_ == 3) {
        std::sort(anchors_.begin(), anchors_.end());
        std::sort(spans_.begin(), spans_.end());
        anchor_ = static_cast<float>(anchors_[1]);
        gain_ = static_cast<float>(1.0 / spans_[1]);
        calibrated_frame_ = frame;
      }
      return count_ == 3;
    }

    depth_calibration_result result(depth_orientation orientation) const {
      depth_calibration_result output;
      output.samples = count_;
      output.sampled_frame = calibrated_frame_;
      output.calibrated = count_ == 3 && orientation != depth_orientation::automatic;
      if (output.calibrated) {
        output.raw_anchor = anchor_;
        output.raw_gain = orientation == depth_orientation::reversed ? gain_ : -gain_;
      }
      return output;
    }

  private:
    depth_calibration_key key_;
    std::uint64_t last_frame_ = 0, calibrated_frame_ = 0;
    unsigned count_ = 0;
    std::array<double, 3> anchors_ {}, spans_ {};
    float anchor_ = 0, gain_ = 0;
  };
}
