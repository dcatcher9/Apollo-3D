/** @file src/video_encode_pacing.h
 *  @brief Encoder-owner image waits, including retained-source depth completion.
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <optional>
#include <type_traits>
#include <utility>

namespace video::detail {
  /** One encoder-owned source, retained until a newer capture replaces it.
   *
   * Receipt and conversion are separate: rejecting an early presentation must not discard the
   * only copy of the final changed image. This state never changes source timestamps or invents
   * content equality. A successful conversion retires pending work but keeps the image for
   * ordinary retained-source completion and same-display encoder rebuilds.
   */
  template<class Image>
  class latest_encode_source_t {
  public:
    void observe(Image image) {
      latest_ = std::move(image);
      pending_ = static_cast<bool>(latest_);
    }

    [[nodiscard]] const Image &latest() const noexcept {
      return latest_;
    }

    [[nodiscard]] bool pending() const noexcept {
      return pending_;
    }

    void converted() noexcept {
      pending_ = false;
    }

    [[nodiscard]] Image release() noexcept {
      pending_ = false;
      return std::exchange(latest_, Image {});
    }

    [[nodiscard]] bool due(
      std::chrono::steady_clock::time_point now,
      std::chrono::steady_clock::time_point encode_target,
      bool force_idr = false
    ) const noexcept {
      return pending_ && (force_idr || now >= encode_target);
    }

    [[nodiscard]] std::optional<std::chrono::nanoseconds> remaining_wait(
      std::chrono::steady_clock::time_point now,
      std::chrono::steady_clock::time_point encode_target
    ) const noexcept {
      if (!pending_) {
        return std::nullopt;
      }
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::max(encode_target - now, std::chrono::steady_clock::duration::zero())
      );
    }

  private:
    Image latest_ {};
    bool pending_ = false;
  };

  /** Wait for a new capture without hiding pending conversion behind the idle heartbeat.
   *
   * The caller still converts/encodes on its normal owner. A pending conversion gets one
   * cadence-sized wait, not a zero-timeout spin or a faster stream of repeated encoded frames.
   * An unconverted capture can shorten that wait further to its presentation deadline; zero is
   * valid only when the caller can immediately retire that pending source on the encode owner.
   * Before the first real capture, there is no retained source to service on a timeout.
   */
  template<class ImageEvent, class Rep, class Period>
  auto wait_for_encode_image(
    ImageEvent &images,
    std::chrono::duration<Rep, Period> idle_interval,
    std::chrono::nanoseconds frame_interval,
    bool has_retained_source,
    bool depth_pipeline_ready,
    bool conversion_poll_pending,
    std::optional<std::chrono::nanoseconds> pending_source_wait = std::nullopt
  ) {
    using namespace std::chrono_literals;
    using wait_duration_t = std::common_type_t<decltype(idle_interval), std::chrono::nanoseconds>;
    wait_duration_t wait = idle_interval;
    if (has_retained_source) {
      if (depth_pipeline_ready) {
        wait = 0ns;
      } else if (conversion_poll_pending) {
        wait = std::min(wait, wait_duration_t {std::max(frame_interval, std::chrono::nanoseconds {1ms})});
      }
      if (pending_source_wait) {
        wait = std::min(wait, wait_duration_t {std::max(*pending_source_wait, 0ns)});
      }
    }
    return images.pop(wait);
  }
}  // namespace video::detail
