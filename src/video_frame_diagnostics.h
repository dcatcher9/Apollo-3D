/** @file src/video_frame_diagnostics.h
 *  @brief Timestamp boundaries for optional host video diagnostics; never scheduling authority.
 */
#pragma once

#include <chrono>
#include <optional>
#include <utility>

namespace video::detail {
  using diagnostic_clock_t = std::chrono::steady_clock;
  using diagnostic_timestamp_t = std::optional<diagnostic_clock_t::time_point>;

  /** An inactive optional must not construct trackers that read clocks or allocate strings. */
  template<class State>
  std::optional<State> make_diagnostic_state(bool enabled) {
    if (enabled) {
      return std::optional<State> {std::in_place};
    }
    return std::nullopt;
  }

  enum class diagnostic_content_e {
    unknown,
    new_content,
    repeated_content,
  };

  /** Retained/older pixels must not be reported as a new-content processing stall. */
  class diagnostic_content_tracker_t {
  public:
    diagnostic_content_e observe(
      const diagnostic_timestamp_t &content_timestamp,
      bool explicitly_repeated = false
    ) noexcept {
      if (!content_timestamp) {
        return diagnostic_content_e::unknown;
      }
      const bool repeated = explicitly_repeated ||
                            (latest_ && *content_timestamp <= *latest_);
      if (!latest_ || *content_timestamp > *latest_) {
        latest_ = content_timestamp;
      }
      return repeated ? diagnostic_content_e::repeated_content :
                        diagnostic_content_e::new_content;
    }

  private:
    diagnostic_timestamp_t latest_;
  };

  template<class Now>
  diagnostic_timestamp_t diagnostic_timestamp(bool enabled, const Now &now) {
    if (!enabled) {
      return std::nullopt;
    }
    return now();
  }

  /** Missing or regressed clocks are unavailable, not a fabricated zero-latency sample. */
  inline std::optional<double> diagnostic_elapsed_ms(
    const diagnostic_timestamp_t &started,
    diagnostic_clock_t::time_point ended
  ) noexcept {
    if (!started || ended < *started) {
      return std::nullopt;
    }
    return std::chrono::duration<double, std::milli>(ended - *started).count();
  }
}  // namespace video::detail
