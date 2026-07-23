/**
 * @file src/video.h
 * @brief Declarations for video.
 */
#pragma once

// standard includes
#include <atomic>
#include <chrono>
#include <cstdint>
#include <numeric>

// local includes
#include "config.h"
#include "input.h"
#include "platform/common.h"
#include "thread_safe.h"
#include "video_colorspace.h"

namespace video {
  // The broadcast worker should never be allowed to accumulate hundreds of milliseconds of
  // already encoded video. Three frames absorb ordinary encoder/network scheduling jitter while
  // keeping recovery bounded when the sender cannot keep up.
  constexpr std::uint32_t ENCODED_PACKET_QUEUE_LIMIT = 3;

  /* Host-side SBS 3D mode requested by the client via the 0x3003 control message.
     Must match the SBS_MODE_* wire values in the client's moonlight-common-c Limelight.h. */
  enum sbs_mode_e : int {
    SBS_OFF = 0,  ///< No host depth; encoder emits a plain W x H frame.
    SBS_AI = 1,  ///< Enable the startup-configured AI pipeline; encoder emits 2W x H.
  };

  /* Debug: set true by the 0x3004 "SBS Debug Dump" control message (client button). The next
     SBS convert() in display_vram consumes it (exchange->false) and dumps one frame's source,
     depth and SBS-result images to the configured debug dir. */
  extern std::atomic<bool> sbs_debug_dump_pending;

  /* Encoding configuration requested by remote client */
  struct config_t {
    // DO NOT CHANGE ORDER OR ADD FIELDS IN THE MIDDLE!!!!!
    // ONLY APPEND NEW FIELD AFTERWARDS!!!!!!!!!
    // BIG F WORD to Sunshine!!!!!!!!!
    int width;  // Video width in pixels
    int height;  // Video height in pixels
    int framerate;  // Requested framerate, used in individual frame bitrate budget calculation
    int bitrate;  // Video bitrate in kilobits (1000 bits) for requested framerate
    int slicesPerFrame;  // Number of slices per frame
    int numRefFrames;  // Max number of reference frames

    /* Requested color range and SDR encoding colorspace, HDR encoding colorspace is always BT.2020+ST2084
       Color range (encoderCscMode & 0x1) : 0 - limited, 1 - full
       SDR encoding colorspace (encoderCscMode >> 1) : 0 - BT.601, 1 - BT.709, 2 - BT.2020 */
    int encoderCscMode;

    int videoFormat;  // 0 - H.264, 1 - HEVC, 2 - AV1

    /* Encoding color depth (bit depth): 0 - 8-bit, 1 - 10-bit
       HDR encoding activates when color depth is higher than 8-bit and the display which is being captured is operating in HDR mode */
    int dynamicRange;

    int encodingFramerate;  // Requested display framerate

    // APPEND-ONLY (see warning above). Host-side SBS mode (sbs_mode_e). It is selected during
    // launch/resume and may also be toggled at runtime via the 0x3003 control message.
    // When != SBS_OFF the encoder output width is doubled to carry the side-by-side frame.
    int sbs_mode = SBS_OFF;

    // APPEND-ONLY. Immutable snapshot selected for this encode device. Keeping the complete
    // startup configuration here prevents a config reload from mixing parameters mid-frame.
    config::video_t::sbs_t sbs_config {};

    // APPEND-ONLY. Session-local depth status channel:
    // 0 idle/failure, 1 engine loading/building, 2 ready, 3 device-pipeline initialization.
    safe::mail_raw_t::event_t<int> sbs_depth_status_event;

    // APPEND-ONLY. Optional exact client refresh rate in hundredths of a hertz. This is accepted
    // only when it agrees with the requested stream rate, so a device's unrelated panel refresh
    // cannot silently override the stream cadence.
    int framerateX100 = 0;

    // APPEND-ONLY. Encode-session-local completion signal for the per-stream Host SBS GPU
    // pipeline. The background initializer raises it after its future becomes ready so the
    // encoder can reconvert the last captured image even when the desktop is otherwise static.
    std::shared_ptr<safe::event_t<bool>> sbs_depth_pipeline_ready_event;
  };

  // Preserve standard NTSC rates instead of approximating them as finite decimal fractions.
  struct rational_t {
    int num;
    int den;
  };

  inline rational_t framerate_x100_to_rational(int framerate_x100) {
    if (framerate_x100 % 2997 == 0) {
      return {(framerate_x100 / 2997) * 30000, 1001};
    }

    if (framerate_x100 == 2397 || framerate_x100 == 2398) {
      return {24000, 1001};
    }

    const auto divisor = std::gcd(framerate_x100, 100);
    return {framerate_x100 / divisor, 100 / divisor};
  }

  /** HTTP hdrMode and RTSP dynamicRange must describe the same selected SDR/HDR format. */
  bool hdr_stream_negotiation_is_coherent(bool launch_hdr, int dynamic_range) noexcept;

  struct sbs_output_dimensions_t {
    int width;
    int height;
  };

  /**
   * Keep DDUP as the fast path, but latch WGC after repeated failures before DDUP has
   * demonstrated a stable capture tenure. This state belongs to one capture session.
   */
  class capture_backend_failover_t {
  public:
    [[nodiscard]] platf::capture_backend_e preferred_backend() const noexcept;
    void reset() noexcept;
    void note_backend_opened(platf::capture_backend_e backend) noexcept;
    void note_capture_result(
      platf::capture_backend_e backend,
      platf::capture_e result,
      std::uint64_t captured_frames,
      std::chrono::steady_clock::duration lifetime
    ) noexcept;

  private:
    platf::capture_backend_e preferred_backend_ = platf::capture_backend_e::ddup;
    unsigned early_ddup_failures_ = 0;
  };

  /** Compute the codec-safe packed Host SBS size for a negotiated per-eye frame. */
  sbs_output_dimensions_t host_sbs_output_dimensions(
    int base_width,
    int base_height,
    int video_format,
    int configured_max_width,
    int runtime_max_width = 0
  );

  /* Startup-profile-selected depth model for the host SBS pipeline. The configured name is matched
     against config::depth_model_registry(), else synthesized from the model/url escape hatch. */
  config::depth_model_info active_depth_model();
  config::depth_model_info depth_model_for_profile(const config::video_t::sbs_t &profile);
  using img_event_t = std::shared_ptr<safe::event_t<std::shared_ptr<platf::img_t>>>;

  struct encode_session_t {
    virtual ~encode_session_t() = default;

    virtual int convert(platf::img_t &img) = 0;

    virtual void request_idr_frame() = 0;

    virtual void request_normal_frame() = 0;

    virtual void invalidate_ref_frames(int64_t first_frame, int64_t last_frame) = 0;
  };

  struct packet_raw_t {
    virtual ~packet_raw_t() = default;

    virtual bool is_idr() = 0;

    virtual int64_t frame_index() = 0;

    virtual uint8_t *data() = 0;

    virtual size_t data_size() = 0;

    // Retain only the network-send state needed by the broadcast worker. This deliberately does
    // not own the stream session (which owns the broadcast context), avoiding both dangling raw
    // pointers and a session -> broadcast -> queued packet -> session ownership cycle.
    std::shared_ptr<void> channel_data;
    bool after_ref_frame_invalidation = false;
    std::optional<std::chrono::steady_clock::time_point> frame_timestamp;
    // Timestamp after encoding, used independently from the capture timestamp to bound only
    // host-side encoded-packet backlog.
    std::chrono::steady_clock::time_point encoded_timestamp = std::chrono::steady_clock::now();
  };

  struct packet_raw_generic: packet_raw_t {
    packet_raw_generic(std::vector<uint8_t> &&frame_data, int64_t frame_index, bool idr):
        frame_data {std::move(frame_data)},
        index {frame_index},
        idr {idr} {
    }

    bool is_idr() override {
      return idr;
    }

    int64_t frame_index() override {
      return index;
    }

    uint8_t *data() override {
      return frame_data.data();
    }

    size_t data_size() override {
      return frame_data.size();
    }

    std::vector<uint8_t> frame_data;
    int64_t index;
    bool idr;
  };

  using packet_t = std::unique_ptr<packet_raw_t>;

  struct hdr_info_raw_t {
    explicit hdr_info_raw_t(bool enabled):
        enabled {enabled},
        metadata {} {};
    explicit hdr_info_raw_t(bool enabled, const SS_HDR_METADATA &metadata):
        enabled {enabled},
        metadata {metadata} {};

    bool enabled;
    SS_HDR_METADATA metadata;
  };

  using hdr_info_t = std::unique_ptr<hdr_info_raw_t>;

  struct nvenc_capabilities_t {
    bool hevc;
    bool hevc_hdr;
    bool av1;
    bool av1_hdr;
  };

  /** Return one coherent snapshot of the probed native-NVENC 4:2:0 capabilities. */
  nvenc_capabilities_t nvenc_capabilities_snapshot() noexcept;

  void capture(
    safe::mail_t mail,
    config_t config,
    std::shared_ptr<void> channel_data
  );

  /**
   * @brief Check if we can allow probing for the encoders.
   * @return True if there should be no issues with the probing, false if we should prevent it.
   */
  bool allow_encoder_probing();

  /**
   * @brief Probe encoders and select the preferred encoder.
   * This is called once at startup and each time a stream is launched to
   * ensure the best encoder is selected. Encoder availability can change
   * at runtime due to all sorts of things from driver updates to eGPUs.
   *
   * @warning This is only safe to call when there is no client actively streaming.
   */
  int probe_encoders();
}  // namespace video
