/**
 * @file src/platform/windows/audio_pcm_staging.h
 * @brief Bounded PCM staging between WASAPI packets and complete capture frames.
 */
#pragma once

#include "audio_helpers.h"
#include "src/utility.h"

#include <algorithm>
#include <cstddef>
#include <span>

namespace platf::audio::detail {
  class pcm_staging_t {
  public:
    void reset(std::size_t capacity) {
      samples_ = util::buffer_t<float> {capacity};
      read_ = 0;
      written_ = 0;
#ifdef SUNSHINE_TESTS
      relocated_samples_ = 0;
#endif
    }

    std::size_t readable_count() const {
      return written_ - read_;
    }

    // An incomplete frame leaves both its destination and the staged samples untouched.
    bool read_into(std::span<float> output) {
      if (output.size() > readable_count()) {
        return false;
      }
      if (output.empty()) {
        return true;
      }

      std::copy_n(samples_.begin() + read_, output.size(), output.begin());
      read_ += output.size();
      return true;
    }

    // Keep the oldest staged samples and accept only the packet prefix that still fits.
    // Silent WASAPI packets may have a null sample pointer.
    std::size_t append_packet(const float *samples, std::size_t count, bool silent, float gain) {
      if (count == 0) {
        return 0;
      }

      // Reclaim samples left by earlier reads before adding any new packet data.
      // Later packets in the same drain then append without moving earlier packets.
      if (read_ > 0) {
        const auto remaining = readable_count();
        if (remaining > 0) {
#ifdef SUNSHINE_TESTS
          relocated_samples_ += remaining;
#endif
          std::move(samples_.begin() + read_, samples_.begin() + written_, samples_.begin());
        }
        written_ = remaining;
        read_ = 0;
      }

      const auto accepted = std::min(samples_.size() - written_, count);
      if (accepted == 0) {
        return 0;
      }

      auto destination = samples_.begin() + written_;
      if (silent) {
        std::fill_n(destination, accepted, 0.0f);
      } else {
        std::copy_n(samples, accepted, destination);
        apply_endpoint_volume(std::span<float> {destination, accepted}, gain);
      }
      written_ += accepted;
      return accepted;
    }

#ifdef SUNSHINE_TESTS
    std::size_t relocated_samples_for_test() const {
      return relocated_samples_;
    }
#endif

  private:
    util::buffer_t<float> samples_;
    std::size_t read_ = 0;
    std::size_t written_ = 0;
#ifdef SUNSHINE_TESTS
    std::size_t relocated_samples_ = 0;
#endif
  };
}  // namespace platf::audio::detail
