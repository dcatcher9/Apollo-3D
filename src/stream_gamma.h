/** Session-local gamma selection and encoder-proven state. */
#pragma once

#include <cstdint>
#include <mutex>

namespace video {
  constexpr std::uint32_t stream_gamma_queue_limit = 8;

  enum class stream_gamma_mode_e : std::uint8_t {
    windows_default = 0,
    gamma_2_2 = 1,
    gamma_2_4 = 2,
  };

  [[nodiscard]] constexpr bool valid_stream_gamma_mode(std::uint8_t mode) noexcept {
    return mode <= static_cast<std::uint8_t>(stream_gamma_mode_e::gamma_2_4);
  }

  enum class stream_gamma_status_e : std::uint8_t {
    applied = 0,
    rejected_invalid = 1,
    rejected_unsupported = 2,
    failed = 3,
  };

  struct stream_gamma_request_t {
    stream_gamma_mode_e mode = stream_gamma_mode_e::windows_default;
    std::uint32_t request_id = 0;  // Zero is reserved for initial/rebuilt encoder state.
  };

  struct stream_gamma_snapshot_t {
    stream_gamma_mode_e mode = stream_gamma_mode_e::windows_default;
    std::uint32_t generation = 0;  // Zero means no converted and encoded frame has been proven.
    float white_nits = 203.0f;
  };

  struct stream_gamma_ack_t {
    stream_gamma_status_e status = stream_gamma_status_e::applied;
    stream_gamma_mode_e requested_mode = stream_gamma_mode_e::windows_default;
    std::uint32_t request_id = 0;
    stream_gamma_snapshot_t applied;
  };

  /** Requested mode survives encoder rebuilds independently of the proven applied mode.
   * Only the serialized encode thread changes either state after session construction.
   */
  class stream_gamma_publisher_t {
  public:
    explicit stream_gamma_publisher_t(stream_gamma_mode_e requested_mode = stream_gamma_mode_e::windows_default):
        requested_mode_ {valid_stream_gamma_mode(static_cast<std::uint8_t>(requested_mode)) ?
                           requested_mode :
                           stream_gamma_mode_e::windows_default} {
    }

    [[nodiscard]] stream_gamma_mode_e requested_mode() const {
      std::lock_guard lock(mutex_);
      return requested_mode_;
    }

    bool set_requested_mode(stream_gamma_mode_e mode) {
      if (!valid_stream_gamma_mode(static_cast<std::uint8_t>(mode))) {
        return false;
      }
      std::lock_guard lock(mutex_);
      requested_mode_ = mode;
      return true;
    }

    [[nodiscard]] stream_gamma_snapshot_t current() const {
      std::lock_guard lock(mutex_);
      return state_;
    }

    stream_gamma_snapshot_t publish(stream_gamma_mode_e mode, float white_nits) {
      std::lock_guard lock(mutex_);
      state_.mode = mode;
      state_.white_nits = white_nits;
      if (++state_.generation == 0) {
        ++state_.generation;
      }
      return state_;
    }

  private:
    mutable std::mutex mutex_;
    stream_gamma_mode_e requested_mode_;
    stream_gamma_snapshot_t state_;
  };
}  // namespace video
