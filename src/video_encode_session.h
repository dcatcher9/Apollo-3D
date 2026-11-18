/**
 * @file src/video_encode_session.h
 * @brief Private concrete native encoder session used by the host video loop.
 */
#pragma once

#include "native_encode_owner.h"
#include "nvenc/nvenc_base.h"
#include "video.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace video {
  class nvenc_encode_session_t {
  public:
    explicit nvenc_encode_session_t(platf::native_encode_owner_t encode_device);

    int convert(platf::img_t &img);
    int convert_with_encode_target(platf::img_t &img, const std::chrono::steady_clock::time_point next_encode_target);
    bool needs_conversion_poll() const;
    void set_external_frame_wake(std::function<void()> wake);
    bool conversion_kept_input() const;
    bool external_frame_held() const;
    bool set_stream_gamma(stream_gamma_mode_e mode);
    stream_gamma_mode_e stream_gamma_mode() const;
    float stream_gamma_white_nits() const;
    bool stream_is_hdr() const;
    std::optional<std::chrono::steady_clock::time_point> rendered_content_timestamp() const;
    void request_idr_frame();
    void request_normal_frame();
    void invalidate_ref_frames(int64_t first_frame, int64_t last_frame);
    bool invalidation_needs_drain(int64_t first_frame, int64_t last_frame) const;
    bool reconfigure_bitrate(int bitrate_kbps);
    nvenc::nvenc_encoded_frame encode_frame(uint64_t frame_index, std::vector<std::uint8_t> &&frame_buffer);
    nvenc::nvenc_base::frame_timing_t last_frame_timing() const;
    std::string video_memory_diagnostics() const;
    unsigned pipeline_depth() const;
    bool input_waits_for_retirement() const noexcept;
    void retire_input_reads();
    bool submit_frame(uint64_t frame_index);
    nvenc::nvenc_encoded_frame retrieve_frame();
    std::chrono::nanoseconds last_submit_timing() const;
    hdr_info_t hdr_info() const;
    std::future<void> retire() &&;
    std::vector<std::uint8_t> acquire_frame_buffer();
    void recycle_frame_buffer(std::vector<std::uint8_t> &&frame_buffer);
    const std::shared_ptr<encoded_frame_buffer_pool_t> &frame_pool() const;

  private:
    platf::native_encode_owner_t device;
    nvenc::nvenc_base &encoder;
    bool force_idr = false;
    std::shared_ptr<encoded_frame_buffer_pool_t> frame_buffer_pool =
      std::make_shared<encoded_frame_buffer_pool_t>();
  };

#ifdef SUNSHINE_TESTS
  namespace detail {
    std::unique_ptr<nvenc_encode_session_t> make_native_encode_session_for_test(
      platf::display_t &display,
      const config_t &config,
      bool enforce_launch_hdr_contract = false
    );
    hdr_info_t native_encode_session_hdr_info_for_test(const nvenc_encode_session_t &session);
    std::future<void> retire_native_encode_session_for_test(std::unique_ptr<nvenc_encode_session_t> session);
  }  // namespace detail
#endif
}  // namespace video
