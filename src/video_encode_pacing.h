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
    /** Retain the newest capture. One that carries nothing new for the converter (metadata for
     * pixels an external provider owns, with an unchanged cursor) replaces the retained image but
     * neither schedules a conversion nor cancels one still owed for an earlier capture.
     */
    void observe(Image image, bool new_content = true) {
      latest_ = std::move(image);
      pending_ = static_cast<bool>(latest_) && (pending_ || new_content);
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

    /** Return real content to a same-display replacement after either failure or a mode change.
     * A capture already waiting in the mailbox is newer and must win. Display reinitialization
     * owns disposal of old-device images, so never hand one back across that boundary. An image
     * without pixels (metadata for pixels an external provider owned) is never handed back
     * either: the replacement could only keep its black startup input for it.
     */
    template<class ImageEvent>
    void return_for_rebuild(ImageEvent &images, bool shutting_down, bool display_reinit_pending, bool holds_pixels = true) {
      if (latest_ && holds_pixels && !shutting_down && images.running() && !display_reinit_pending) {
        images.try_raise(release());
      }
    }

    [[nodiscard]] bool due(
      std::chrono::steady_clock::time_point now,
      std::chrono::steady_clock::time_point encode_target,
      bool force_idr = false,
      bool independent_conversion_pending = false
    ) const noexcept {
      return (pending_ || (latest_ && independent_conversion_pending)) &&
             (force_idr || now >= encode_target);
    }

    [[nodiscard]] std::optional<std::chrono::nanoseconds> remaining_wait(
      std::chrono::steady_clock::time_point now,
      std::chrono::steady_clock::time_point encode_target,
      bool independent_conversion_pending = false
    ) const noexcept {
      // An external provider can publish new pixels independently of desktop capture. Its
      // retained-source polls share the presentation deadline instead of adding a full frame
      // wait after the preceding conversion/encode. Ordinary depth-completion polls keep their
      // existing cadence-sized wait by leaving this opt-in false.
      if (!pending_ && !(latest_ && independent_conversion_pending)) {
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

  /** Longest wait for capture to deliver the desktop to an encoder that starts without a source
   * because its predecessor's input had no pixels (a live export owned them).
   *
   * Capture re-reads the whole desktop for it no later than when one idle source wait times out,
   * and retries after its idle backoff: 200 ms + 10 ms in platf::dxgi::detail::capture_wait_policy_t.
   */
  inline constexpr std::chrono::milliseconds desktop_after_export_wait {210};

  /** Holds back the black startup input of an encoder whose predecessor's input had no pixels.
   *
   * Such an encoder starts with only its black startup input while capture re-reads the whole
   * desktop. Encoding that input, for the minimum-FPS keepalive or a recovery request, would flash
   * black (as an IDR, for a recovery). Until a conversion replaces it, or desktop_after_export_wait
   * has passed, the encode loop encodes nothing and waits only for capture: no keepalive or poll
   * deadline shortens that wait, so it cannot spin. An IDR requested meanwhile stays latched for
   * the first real frame.
   */
  class startup_input_hold_t {
  public:
    startup_input_hold_t(std::chrono::steady_clock::time_point now, bool hold) noexcept {
      if (hold) {
        deadline_ = now + desktop_after_export_wait;
      }
    }

    /** Whether the startup input is still held at `now`; the hold ends at its deadline. */
    [[nodiscard]] bool active(std::chrono::steady_clock::time_point now) noexcept {
      if (deadline_ && now >= *deadline_) {
        deadline_.reset();
      }
      return deadline_.has_value();
    }

    /** A conversion replaced the startup input. */
    void replaced() noexcept {
      deadline_.reset();
    }

    /** The image wait while held: the idle interval, cut at the deadline. */
    template<class Duration>
    [[nodiscard]] Duration wait(std::chrono::steady_clock::time_point now, Duration idle_interval) const noexcept {
      if (!deadline_) {
        return idle_interval;
      }
      const auto left = std::max(*deadline_ - now, std::chrono::steady_clock::duration::zero());
      return std::min(idle_interval, std::chrono::duration_cast<Duration>(left));
    }

  private:
    std::optional<std::chrono::steady_clock::time_point> deadline_;
  };

  /** Time left until an independent provider's minimum-FPS keepalive is due.
   *
   * Such a provider wakes the encode loop itself and is not polled at stream cadence, and its
   * metadata-only captures do not encode. Measuring from the last encode keeps a steady stream
   * of those captures from postponing the keepalive indefinitely.
   */
  [[nodiscard]] inline std::chrono::nanoseconds provider_keepalive_wait(
    std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::time_point last_encode,
    std::chrono::nanoseconds keepalive_interval
  ) noexcept {
    return std::max(
      std::chrono::duration_cast<std::chrono::nanoseconds>(last_encode + keepalive_interval - now),
      std::chrono::nanoseconds::zero()
    );
  }

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
