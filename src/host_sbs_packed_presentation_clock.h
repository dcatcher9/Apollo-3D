#pragma once

#include <algorithm>
#include <chrono>
#include <optional>

namespace platf::dxgi::detail {
  // Timing for already-rendered packed pixels only. This grants no model, subtitle, or source
  // reuse authority. The pixel/route owner invalidates this clock with its packed image.
  class host_sbs_packed_presentation_clock_t {
  public:
    using clock_t = std::chrono::steady_clock;
    using source_stamp_t = std::optional<clock_t::time_point>;
    enum class output_e { flat, packed, completed, queued };
    enum class timeout_event_e { none, expired, recovered };

    void invalidate() noexcept {
      first_divergence_at_.reset();
    }

    void reset() noexcept {
      invalidate();
      timeout_active_ = false;
    }

    void observe_source(
      const source_stamp_t &packed_source,
      const source_stamp_t &current_source,
      const clock_t::time_point observed_at
    ) noexcept {
      // Missing identity must not create a new presentation allowance, nor erase an interval
      // already observed with exact identities. It retains the old source-age policy below.
      if (!packed_source || !current_source) {
        return;
      }
      if (*packed_source == *current_source) {
        first_divergence_at_.reset();
      } else if (!first_divergence_at_) {
        first_divergence_at_ = observed_at;
      }
    }

    [[nodiscard]] clock_t::duration hold_age(
      const clock_t::duration packed_source_age,
      const source_stamp_t &packed_source,
      const source_stamp_t &current_source,
      const clock_t::time_point now
    ) const noexcept {
      if (!packed_source || !current_source || !first_divergence_at_ ||
          packed_source_age == clock_t::duration::max()) {
        return packed_source_age;
      }
      // Static time before the first changed source is not a failed presentation hold. A newer
      // genuine packed publication can still make progress under the original source-age cap.
      // Neither retries, new changed sources, nor failed admissions restart this interval.
      return std::min(packed_source_age, now - *first_divergence_at_);
    }

    timeout_event_e observe_timeout(const bool age_expired, const output_e output) noexcept {
      const bool showing_expired_flat = age_expired && output == output_e::flat;
      if (showing_expired_flat && !timeout_active_) {
        timeout_active_ = true;
        return timeout_event_e::expired;
      }
      // A packed/queued draw does not prove a completed depth owner recovered.
      if (!showing_expired_flat && timeout_active_ && output == output_e::completed) {
        timeout_active_ = false;
        return timeout_event_e::recovered;
      }
      return timeout_event_e::none;
    }

  private:
    std::optional<clock_t::time_point> first_divergence_at_;
    bool timeout_active_ = false;
  };
}  // namespace platf::dxgi::detail
