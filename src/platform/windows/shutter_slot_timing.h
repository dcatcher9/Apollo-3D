#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace platf::dxgi::detail {
  struct shutter_timing_percentile_t {
    double upper_bound_us = 0;
    // An overflow result is a lower bound: the percentile is at least upper_bound_us.
    bool overflow = false;
  };

  // Diagnostic-only, fixed-size storage. Updating it neither allocates nor writes logs.
  // The histogram bounds percentile precision; exact min/mean/max retain long outliers.
  class shutter_timing_aggregate_t {
  public:
    static constexpr unsigned histogram_bin_us = 100;
    static constexpr unsigned histogram_bin_count = 128;

    void observe(double microseconds) noexcept {
      if (!std::isfinite(microseconds) || microseconds < 0) {
        ++invalid_count_;
        return;
      }
      if (!count_) {
        minimum_us_ = maximum_us_ = microseconds;
      } else {
        minimum_us_ = std::min(minimum_us_, microseconds);
        maximum_us_ = std::max(maximum_us_, microseconds);
      }
      ++count_;
      mean_us_ += (microseconds - mean_us_) / static_cast<double>(count_);
      if (microseconds >= histogram_bin_us * histogram_bin_count) {
        ++overflow_count_;
      } else {
        ++histogram_[static_cast<unsigned>(microseconds / histogram_bin_us)];
      }
    }

    [[nodiscard]] std::uint64_t count() const noexcept {
      return count_;
    }

    [[nodiscard]] std::uint64_t invalid_count() const noexcept {
      return invalid_count_;
    }

    [[nodiscard]] std::uint64_t overflow_count() const noexcept {
      return overflow_count_;
    }

    [[nodiscard]] double minimum_us() const noexcept {
      return minimum_us_;
    }

    [[nodiscard]] double mean_us() const noexcept {
      return mean_us_;
    }

    [[nodiscard]] double maximum_us() const noexcept {
      return maximum_us_;
    }

    [[nodiscard]] shutter_timing_percentile_t percentile(unsigned percent) const noexcept {
      if (!count_) {
        return {};
      }
      percent = std::clamp(percent, 1u, 100u);
      // Nearest-rank percentile without multiplying a potentially large total by percent.
      const auto rank = (count_ / 100) * percent + ((count_ % 100) * percent + 99) / 100;
      std::uint64_t cumulative = 0;
      for (unsigned index = 0; index < histogram_.size(); ++index) {
        cumulative += histogram_[index];
        if (cumulative >= rank) {
          return {static_cast<double>((index + 1) * histogram_bin_us), false};
        }
      }
      return {static_cast<double>(histogram_bin_us * histogram_bin_count), true};
    }

  private:
    std::array<std::uint64_t, histogram_bin_count> histogram_ {};
    std::uint64_t count_ = 0, invalid_count_ = 0, overflow_count_ = 0;
    double minimum_us_ = 0, mean_us_ = 0, maximum_us_ = 0;
  };
}  // namespace platf::dxgi::detail
