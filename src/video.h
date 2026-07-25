/**
 * @file src/video.h
 * @brief Declarations for video.
 */
#pragma once

// standard includes
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <vector>

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

  /* Live stream geometry/rate change requested by the client via the 0x3007 control message.
     Resolution, frame rate and bitrate change without reconnecting: the encode session is torn
     down and rebuilt in place, exactly as the 0x3003 SBS toggle already does.

     These are post-validation encoder values, not the client's raw request. The control handler
     applies the same clamp and FEC/audio wire-budget deduction that RTSP ANNOUNCE applies, so a
     live change and a fresh launch produce an identical encoder budget for the same request. */
  struct video_mode_change_t {
    int width;  // Base (per-eye) width; SBS doubling is applied on top of this.
    int height;
    int framerate;  // Whole frames per second, used for per-frame bitrate budgeting.
    int framerateX100;  // Exact rate in hundredths of a hertz; preserves 23.976/29.97.
    int encodingFramerate;  // Requested display framerate in millihertz.
    int bitrate;  // Encoder budget in kbps, after the wire-budget transformation.
    // Opaque client-generated correlation token. The host never interprets or validates it and
    // never assumes an ordering from it; it exists only to be echoed back in the acknowledgement.
    int request_id;
  };

  /**
   * What an encode session is actually running, as opposed to what the client asked for.
   *
   * The two can legitimately differ: an oversized width is capped to the codec ceiling with the
   * height scaled to preserve aspect, and the bitrate is the post-budget encoder value rather than
   * the requested wire budget. `width`/`height` are the BASE (per-eye) values, i.e. before any SBS
   * doubling, so they remain directly comparable to what the client requested.
   */
  struct effective_video_mode_t {
    int width;
    int height;
    int framerateX100;
    int bitrate;
  };

  /**
   * Latest effective mode, published by the encode loop and readable by any thread.
   *
   * A client that is told its request was refused still needs to know what is running, so this is
   * seeded with the launch mode and is never empty.
   */
  class effective_video_mode_publisher_t {
  public:
    explicit effective_video_mode_publisher_t(const effective_video_mode_t &initial):
        _mode {initial} {
    }

    void publish(const effective_video_mode_t &mode) {
      std::lock_guard lock {_mutex};
      _mode = mode;
    }

    [[nodiscard]] effective_video_mode_t current() const {
      std::lock_guard lock {_mutex};
      return _mode;
    }

  private:
    mutable std::mutex _mutex;
    effective_video_mode_t _mode;
  };

  /**
   * Outcome of installing a live video mode, reported by the encode loop once an encode session
   * has actually been created. Only this loop knows the geometry that was really installed, so it
   * is the only place that can answer a 0x3007 request truthfully.
   */
  struct video_mode_applied_t {
    // Every request this encode session answers. More than one when a client's requests were
    // collapsed: the newest is honoured and the rest are reported as not applied.
    std::vector<int> request_ids;
    // False when the requests were superseded, or when the mode had to be rolled back because the
    // encoder refused it. `mode` then describes what is running instead.
    bool applied;
    effective_video_mode_t mode;
  };

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

    // APPEND-ONLY. Session-shared view of the mode the encode loop is actually running. The
    // control stream reads it to tell a client what is in effect when its live request is refused.
    std::shared_ptr<effective_video_mode_publisher_t> effective_mode;
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

  /**
   * Clamp a requested encode size to what the codec can actually encode, scaling the height to
   * preserve the aspect ratio. A live 0x3007 mode change is deliberately not width-checked on the
   * control thread, because a rejected-but-creatable mode is worse than a capped one: a failed
   * non-SBS encoder creation tears the whole session down, while capping cannot fail.
   */
  sbs_output_dimensions_t clamp_encode_dimensions(
    int width,
    int height,
    int video_format,
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
