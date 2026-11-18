/**
 * @file src/audio_capture.h
 * @brief Owned PCM frames transferred from audio capture to encoding.
 */
#pragma once

#include "platform/common.h"
#include "thread_safe.h"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

namespace audio::detail {
  using pcm_frame_t = std::unique_ptr<float[]>;
  using pcm_queue_t = safe::queue_t<pcm_frame_t>;

  // sample_count is fixed for the capture stream. Keep its pending allocation through
  // unsuccessful samples, and publish it only when the microphone filled the entire frame.
  inline platf::capture_e capture_pcm_frame(platf::mic_t &mic, pcm_frame_t &pending, std::size_t sample_count, pcm_queue_t &samples) {
    if (!pending) {
      pending = std::make_unique_for_overwrite<float[]>(sample_count);
    }
    const auto status = mic.sample(std::span<float> {pending.get(), sample_count});
    if (status == platf::capture_e::ok) {
      samples.raise(std::move(pending));
    }
    return status;
  }
}  // namespace audio::detail
