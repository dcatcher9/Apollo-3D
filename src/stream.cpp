/**
 * @file src/stream.cpp
 * @brief Definitions for the streaming protocols.
 */

// standard includes
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

// lib includes
#include <boost/align/aligned_allocator.hpp>
#include <boost/endian/arithmetic.hpp>
#include <rs.h>

extern "C" {
  // clang-format off
#include <moonlight-common-c/src/Limelight-internal.h>
  // clang-format on
}

// local includes
#include "atomic_presentation.h"
#include "client_features.h"
#include "config.h"
#include "crypto.h"
#include "globals.h"
#include "host_sbs_provider.h"
#include "host_sbs_resolution.h"
#include "input.h"
#include "logging.h"
#include "network.h"
#include "platform/common.h"
#include "process.h"
#include "rtsp.h"
#include "session_join_watchdog.h"
#include "session_resume_lifecycle.h"
#include "stream.h"
#ifdef SUNSHINE_TESTS
  #include "atomic_presentation_test.h"
  #include "stream_media_test.h"
#endif
#include "sync.h"
#include "system_tray.h"
#include "thread_safe.h"
#include "utility.h"
#include "video.h"

namespace asio = boost::asio;
namespace sys = boost::system;

using asio::ip::tcp;
using asio::ip::udp;

using namespace std::literals;

namespace stream {
  bool sbs_debug_dump_request_allowed(
    int requested_sbs_mode,
    bool has_session_request_latch
  ) noexcept {
    return has_session_request_latch &&
           (requested_sbs_mode == ::video::SBS_AI || ::video::is_game_mode(requested_sbs_mode));
  }

  namespace control_packet {
    // Gen 7 encrypted control protocol and the extensions used by Artemis.
    constexpr std::uint16_t start = 0x0307;
    constexpr std::uint16_t invalidate_ref_frames = 0x0301;
    constexpr std::uint16_t input = 0x0206;
    constexpr std::uint16_t rumble = 0x010b;
    constexpr std::uint16_t termination = 0x0109;
    constexpr std::uint16_t periodic_ping = 0x0200;
    constexpr std::uint16_t request_idr = 0x0302;
    constexpr std::uint16_t encrypted = 0x0001;
    constexpr std::uint16_t hdr_mode = 0x010e;
    constexpr std::uint16_t set_motion_event = 0x5501;
    // Direction-specific reuse: Moonlight sends frame FEC status on this ID, while Sunshine sends
    // controller RGB feedback on it. Name both directions so an inbound FEC report is never
    // mistaken for an echoed LED command.
    constexpr std::uint16_t frame_fec_status = 0x5502;
    constexpr std::uint16_t set_rgb_led_feedback = 0x5502;
    constexpr std::uint16_t set_adaptive_triggers = 0x5503;
    constexpr std::uint16_t sbs_debug_dump = 0x3004;
    constexpr std::uint16_t depth_status = 0x3006;
    // 0x3005 is retired (Set Depth Model); older clients may still emit it, so it is not reused.
    constexpr std::uint16_t set_video_mode = 0x3007;
    constexpr std::uint16_t live_video_mode_ack = 0x3008;
    constexpr std::uint16_t sbs_telemetry_subscription = 0x3009;
    constexpr std::uint16_t sbs_telemetry_state = 0x300A;
    constexpr std::uint16_t game_source_status = 0x300B;
    constexpr std::uint16_t set_stream_gamma = 0x300C;
    constexpr std::uint16_t stream_gamma_ack = 0x300D;
  }  // namespace control_packet

  enum class socket_e : int {
    video,  ///< Video
    audio  ///< Audio
  };

#pragma pack(push, 1)

  struct video_short_frame_header_t {
    uint8_t *payload() {
      return (uint8_t *) (this + 1);
    }

    std::uint8_t headerType;  // Always 0x01 for short headers

    // Sunshine extension
    // Frame processing latency, in 1/10 ms units
    //     zero when the frame is repeated or there is no backend implementation
    boost::endian::little_uint16_at frame_processing_latency;

    // Currently known values:
    // 1 = Normal P-frame
    // 2 = IDR-frame
    // 4 = P-frame with intra-refresh blocks
    // 5 = P-frame after reference frame invalidation
    std::uint8_t frameType;

    // Length of the final packet payload for codecs that cannot handle
    // zero padding, such as AV1 (Sunshine extension).
    boost::endian::little_uint16_at lastPayloadLen;

    // SOURCE_FRAME_ID_V1: nonzero exact encoder-input token, only after negotiation.
    boost::endian::little_uint16_at source_frame_id;
  };

  static_assert(
    sizeof(video_short_frame_header_t) == 8,
    "Short frame header must be 8 bytes"
  );
  static_assert(offsetof(video_short_frame_header_t, source_frame_id) == 6);

  struct video_packet_raw_t {
    uint8_t *payload() {
      return (uint8_t *) (this + 1);
    }

    RTP_PACKET rtp;
    char reserved[4];

    NV_VIDEO_PACKET packet;
  };

  static_assert(sizeof(video_packet_raw_t) == MAX_RTP_HEADER_SIZE + sizeof(NV_VIDEO_PACKET));

  struct video_packet_enc_prefix_t {
    std::uint8_t iv[12];  // 12-byte IV is ideal for AES-GCM
    std::uint32_t frameNumber;
    std::uint8_t tag[16];
  };

  struct audio_packet_t {
    RTP_PACKET rtp;
  };

  struct control_header_v2 {
    std::uint16_t type;
    std::uint16_t payloadLength;

    uint8_t *payload() {
      return (uint8_t *) (this + 1);
    }
  };

  struct control_terminate_t {
    control_header_v2 header;

    std::uint32_t ec;
  };

  struct control_rumble_t {
    control_header_v2 header;

    std::uint32_t useless;

    std::uint16_t id;
    std::uint16_t lowfreq;
    std::uint16_t highfreq;
  };

  struct control_set_motion_event_t {
    control_header_v2 header;

    std::uint16_t id;
    std::uint16_t reportrate;
    std::uint8_t type;
  };

  struct control_set_rgb_led_t {
    control_header_v2 header;

    std::uint16_t id;
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
  };

  static_assert(
    sizeof(SS_FRAME_FEC_STATUS) == FRAME_FEC_STATUS_PAYLOAD_SIZE,
    "Moonlight frame-FEC status wire size changed"
  );

  struct control_adaptive_triggers_t {
    control_header_v2 header;

    std::uint16_t id;
    /**
     * 0x04 - Right trigger
     * 0x08 - Left trigger
     */
    std::uint8_t event_flags;
    std::uint8_t type_left;
    std::uint8_t type_right;
    std::uint8_t left[DS_EFFECT_PAYLOAD_SIZE];
    std::uint8_t right[DS_EFFECT_PAYLOAD_SIZE];
  };

  struct control_hdr_mode_t {
    control_header_v2 header;

    std::uint8_t enabled;

    // Sunshine protocol extension
    SS_HDR_METADATA metadata;
  };

  // Host->client push of the SBS depth-engine state so the client can show a "loading" indicator
  // while a model builds/loads/warms up (Apollo protocol extension 0x3006).
#pragma pack(push, 1)

  struct control_depth_status_t {
    control_header_v2 header;

    std::uint8_t phase;  // 0 idle/failure, 1 engine load/build, 2 ready, 3 device-pipeline init
  };

  // Host->client acknowledgement of a 0x3007 request (Apollo protocol extension 0x3008). The body
  // is a raw byte array so its exact little-endian layout is owned by
  // encode_atomic_presentation_ack_payload() and cannot drift with struct packing.
  struct control_live_video_mode_ack_v2_t {
    control_header_v2 header;

    std::uint8_t body[ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE];
  };

  static_assert(
    sizeof(control_live_video_mode_ack_v2_t) ==
    sizeof(control_header_v2) + ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE
  );

  struct control_sbs_telemetry_state_t {
    control_header_v2 header;

    std::uint8_t body[SBS_TELEMETRY_STATE_PAYLOAD_SIZE];
  };

  struct control_game_source_status_t {
    control_header_v2 header;
    std::uint8_t body[GAME_SOURCE_STATUS_PAYLOAD_SIZE];
  };

  struct control_stream_gamma_ack_t {
    control_header_v2 header;
    std::uint8_t body[STREAM_GAMMA_ACK_PAYLOAD_SIZE];
  };

#pragma pack(pop)

  typedef struct control_encrypted_t {
    std::uint16_t encryptedHeaderType;  // Always LE 0x0001
    std::uint16_t length;  // sizeof(seq) + 16 byte tag + secondary header and data

    // seq is accepted as an arbitrary value in Artemis
    std::uint32_t seq;  // Monotonically increasing sequence number (used as IV for AES-GCM)

    uint8_t *payload() {
      return (uint8_t *) (this + 1);
    }

    // encrypted control_header_v2 and payload data follow
  } *control_encrypted_p;

  struct audio_fec_packet_t {
    RTP_PACKET rtp;
    AUDIO_FEC_HEADER fecHeader;
  };

#pragma pack(pop)

  stream_gamma_request_decode_e decode_stream_gamma_request_payload(
    std::string_view payload,
    video::stream_gamma_request_t &request
  ) noexcept {
    request = {};
    if (payload.size() != STREAM_GAMMA_REQUEST_PAYLOAD_SIZE) {
      return stream_gamma_request_decode_e::invalid;
    }
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(payload.data());
    if (!video::valid_stream_gamma_mode(bytes[1])) {
      return stream_gamma_request_decode_e::invalid;
    }
    video::stream_gamma_request_t decoded;
    decoded.mode = static_cast<video::stream_gamma_mode_e>(bytes[1]);
    for (unsigned i = 0; i < 4; ++i) {
      decoded.request_id |= static_cast<std::uint32_t>(bytes[4 + i]) << (i * 8);
    }
    if (decoded.request_id == 0) {
      return stream_gamma_request_decode_e::invalid;
    }
    request = decoded;
    if (bytes[0] != STREAM_GAMMA_VERSION) {
      return stream_gamma_request_decode_e::unsupported_version;
    }
    return bytes[2] == 0 && bytes[3] == 0 ?
             stream_gamma_request_decode_e::ok :
             stream_gamma_request_decode_e::invalid;
  }

  bool encode_stream_gamma_ack_payload(
    const video::stream_gamma_ack_t &ack,
    std::uint8_t (&out)[STREAM_GAMMA_ACK_PAYLOAD_SIZE]
  ) noexcept {
    if (static_cast<std::uint8_t>(ack.status) > 3 ||
        !video::valid_stream_gamma_mode(static_cast<std::uint8_t>(ack.requested_mode)) ||
        !video::valid_stream_gamma_mode(static_cast<std::uint8_t>(ack.applied.mode)) ||
        ack.applied.generation == 0 || !std::isfinite(ack.applied.white_nits) ||
        ack.applied.white_nits < 40.0f || ack.applied.white_nits > 1000.0f ||
        (ack.status == video::stream_gamma_status_e::applied && ack.requested_mode != ack.applied.mode)) {
      return false;
    }
    std::uint8_t encoded[STREAM_GAMMA_ACK_PAYLOAD_SIZE] {};
    encoded[0] = STREAM_GAMMA_VERSION;
    encoded[1] = static_cast<std::uint8_t>(ack.status);
    encoded[2] = static_cast<std::uint8_t>(ack.requested_mode);
    encoded[3] = static_cast<std::uint8_t>(ack.applied.mode);
    const auto put_u32 = [&](std::size_t offset, std::uint32_t value) {
      for (unsigned i = 0; i < 4; ++i) {
        encoded[offset + i] = static_cast<std::uint8_t>(value >> (i * 8));
      }
    };
    put_u32(4, ack.request_id);
    put_u32(8, ack.applied.generation);
    put_u32(12, std::bit_cast<std::uint32_t>(ack.applied.white_nits));
    std::copy(std::begin(encoded), std::end(encoded), std::begin(out));
    return true;
  }

  live_video_mode_ack_e live_video_mode_ack_status(proc::live_video_mode_result_e result) {
    switch (result) {
      case proc::live_video_mode_result_e::applied:
      // "Unchanged" means the request repeats the session's requested mode, which the last proven
      // display transaction applied; an application may since have chosen another display mode,
      // and the host leaves that in place. The requested mode is live once the encoder rebuilds.
      // From the client's point of view that is indistinguishable from a real transition, and it
      // must not be reported as a failure.
      case proc::live_video_mode_result_e::unchanged:
        return live_video_mode_ack_e::applied;
      case proc::live_video_mode_result_e::needs_reconnect:
        return live_video_mode_ack_e::rejected_needs_reconnect;
      case proc::live_video_mode_result_e::failed:
        return live_video_mode_ack_e::failed;
    }

    // An unmapped outcome is a host bug, not a client one. Report the retryable status so a
    // client is never told to give up on a mode the host may still be able to deliver.
    return live_video_mode_ack_e::failed;
  }

  live_video_mode_request_decode_e decode_live_video_mode_request_payload(
    std::string_view payload,
    live_video_mode_wire_request_t &request
  ) noexcept {
    request = {};
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(payload.data());
    const auto read_u16 = [&](std::size_t offset) {
      return static_cast<std::uint16_t>(bytes[offset]) |
             static_cast<std::uint16_t>(bytes[offset + 1] << 8);
    };
    const auto read_u32 = [&](std::size_t offset) {
      return static_cast<std::uint32_t>(bytes[offset]) |
             (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
             (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
             (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
    };

    if (payload.size() != ATOMIC_PRESENTATION_V2_REQUEST_PAYLOAD_SIZE) {
      return live_video_mode_request_decode_e::invalid;
    }

    request.protocol_version = bytes[0];
    request.desired_sbs_mode = bytes[1];
    request.flags = read_u16(2);
    request.request_id = read_u32(4);
    request.source_width = read_u16(8);
    request.source_height = read_u16(10);
    request.framerate_x100 = read_u32(12);
    request.bitrate_kbps = read_u32(16);
    return request.protocol_version == ATOMIC_PRESENTATION_VERSION ?
             live_video_mode_request_decode_e::v2 :
             live_video_mode_request_decode_e::unsupported_version;
  }

  live_video_mode_ack_t make_live_video_mode_ack(
    std::uint32_t request_id,
    live_video_mode_ack_e status,
    const video::effective_video_mode_t &mode
  ) noexcept {
    return {
      request_id,
      status,
      mode.framerateX100,
      mode.bitrate,
      mode.sbs_mode,
      mode.source_width,
      mode.source_height,
      mode.encoded_width,
      mode.encoded_height,
      mode.generation,
    };
  }

  bool encode_atomic_presentation_ack_payload(
    const live_video_mode_ack_t &ack,
    std::uint8_t (&out)[ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE]
  ) {
    constexpr std::int64_t u16_max = std::numeric_limits<std::uint16_t>::max();
    constexpr std::int64_t u32_max = std::numeric_limits<std::uint32_t>::max();
    if (ack.status < live_video_mode_ack_e::applied || ack.status > live_video_mode_ack_e::failed || ack.request_id < 0 || ack.request_id > u32_max || (ack.applied_sbs_mode < video::SBS_OFF || ack.applied_sbs_mode > video::SBS_GAME_SBS) || ack.applied_source_width < 0 || ack.applied_source_width > u16_max || ack.applied_source_height < 0 || ack.applied_source_height > u16_max || ack.applied_encoded_width < 0 || ack.applied_encoded_width > u16_max || ack.applied_encoded_height < 0 || ack.applied_encoded_height > u16_max || ack.applied_framerate_x100 < 0 || ack.applied_framerate_x100 > u32_max || ack.applied_bitrate_kbps < 0 || ack.applied_bitrate_kbps > u32_max || ack.applied_generation == 0) {
      return false;
    }

    std::uint8_t encoded[ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE] {};
    const auto write_u16 = [&](std::size_t offset, std::uint16_t value) {
      encoded[offset] = static_cast<std::uint8_t>(value & 0xFF);
      encoded[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
    };
    const auto write_u32 = [&](std::size_t offset, std::uint32_t value) {
      encoded[offset] = static_cast<std::uint8_t>(value & 0xFF);
      encoded[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
      encoded[offset + 2] = static_cast<std::uint8_t>((value >> 16) & 0xFF);
      encoded[offset + 3] = static_cast<std::uint8_t>((value >> 24) & 0xFF);
    };

    encoded[0] = ATOMIC_PRESENTATION_VERSION;
    encoded[1] = static_cast<std::uint8_t>(ack.status);
    encoded[2] = static_cast<std::uint8_t>(ack.applied_sbs_mode);
    encoded[3] = 0;  // response flags, reserved
    write_u32(4, static_cast<std::uint32_t>(ack.request_id));
    write_u32(8, ack.applied_generation);
    write_u16(12, static_cast<std::uint16_t>(ack.applied_source_width));
    write_u16(14, static_cast<std::uint16_t>(ack.applied_source_height));
    write_u16(16, static_cast<std::uint16_t>(ack.applied_encoded_width));
    write_u16(18, static_cast<std::uint16_t>(ack.applied_encoded_height));
    write_u32(20, static_cast<std::uint32_t>(ack.applied_framerate_x100));
    write_u32(24, static_cast<std::uint32_t>(ack.applied_bitrate_kbps));
    std::copy(std::begin(encoded), std::end(encoded), std::begin(out));
    return true;
  }

  bool game_source_status_matches_mode(
    const video::game_source_state_t &state,
    const video::effective_video_mode_t &mode
  ) noexcept {
    return video::is_game_mode(mode.sbs_mode) && mode.generation != 0 &&
           state.presentation_generation == mode.generation &&
           state.source_width == mode.source_width && state.source_height == mode.source_height &&
           ((mode.sbs_mode == video::SBS_GAME_MONO && state.state != video::GAME_SOURCE_READY) ||
            (mode.encoded_width == (mode.sbs_mode == video::SBS_GAME_SBS ? mode.source_width * 2 : mode.source_width) &&
             mode.encoded_height == mode.source_height));
  }

  bool encode_game_source_status_payload(
    const video::game_source_state_t &state,
    std::uint8_t (&out)[GAME_SOURCE_STATUS_PAYLOAD_SIZE]
  ) noexcept {
    if (state.state > video::GAME_SOURCE_UNSUPPORTED || state.provider > video::GAME_PROVIDER_RESHADE || (state.state == video::GAME_SOURCE_READY && state.provider == video::GAME_PROVIDER_NONE) || state.presentation_generation == 0 || state.source_revision == 0 || !is_valid_live_video_mode_dimension(state.source_width) || !is_valid_live_video_mode_dimension(state.source_height) || state.packed_width != state.source_width * 2 || state.packed_height != state.source_height) {
      return false;
    }
    std::uint8_t encoded[GAME_SOURCE_STATUS_PAYLOAD_SIZE] {};
    const auto write_u16 = [&](std::size_t offset, std::uint16_t value) {
      encoded[offset] = static_cast<std::uint8_t>(value);
      encoded[offset + 1] = static_cast<std::uint8_t>(value >> 8);
    };
    const auto write_u32 = [&](std::size_t offset, std::uint32_t value) {
      write_u16(offset, static_cast<std::uint16_t>(value));
      write_u16(offset + 2, static_cast<std::uint16_t>(value >> 16));
    };
    encoded[0] = GAME_SOURCE_STATUS_VERSION;
    encoded[1] = state.state;
    encoded[2] = state.provider;
    write_u32(4, state.presentation_generation);
    write_u32(8, state.source_revision);
    write_u16(12, state.source_width);
    write_u16(14, state.source_height);
    write_u16(16, state.packed_width);
    write_u16(18, state.packed_height);
    std::copy(std::begin(encoded), std::end(encoded), std::begin(out));
    return true;
  }

  sbs_telemetry_subscription_decode_e decode_sbs_telemetry_subscription_payload(
    std::string_view payload,
    sbs_telemetry_subscription_request_t &request
  ) {
    request = {};
    if (payload.size() != SBS_TELEMETRY_SUBSCRIPTION_PAYLOAD_SIZE) {
      return sbs_telemetry_subscription_decode_e::invalid;
    }

    const auto *bytes = reinterpret_cast<const std::uint8_t *>(payload.data());
    request.flags = bytes[1];
    request.request_id =
      static_cast<std::uint16_t>(bytes[2]) |
      static_cast<std::uint16_t>(bytes[3] << 8);
    request.interval_ms =
      static_cast<std::uint16_t>(bytes[4]) |
      static_cast<std::uint16_t>(bytes[5] << 8);
    if (bytes[0] != SBS_TELEMETRY_VERSION) {
      return sbs_telemetry_subscription_decode_e::unsupported_version;
    }

    const std::uint16_t reserved =
      static_cast<std::uint16_t>(bytes[6]) |
      static_cast<std::uint16_t>(bytes[7] << 8);
    constexpr std::uint8_t known_flags =
      SBS_TELEMETRY_SUBSCRIBE_ENABLED | SBS_TELEMETRY_SUBSCRIBE_FOCUSED;
    if (reserved != 0 || (request.flags & ~known_flags) != 0) {
      return sbs_telemetry_subscription_decode_e::invalid;
    }

    return sbs_telemetry_subscription_decode_e::ok;
  }

  std::uint16_t effective_sbs_telemetry_interval_ms(
    const sbs_telemetry_subscription_request_t &request
  ) noexcept {
    if (request.interval_ms == 0) {
      return request.focused() ?
               SBS_TELEMETRY_INTERVAL_FOCUSED_MS :
               SBS_TELEMETRY_INTERVAL_BACKGROUND_MS;
    }
    return std::clamp(
      request.interval_ms,
      SBS_TELEMETRY_INTERVAL_MIN_MS,
      SBS_TELEMETRY_INTERVAL_MAX_MS
    );
  }

  sbs_telemetry_status_e sbs_telemetry_wire_status(
    video::sbs_telemetry_sample_status_e status
  ) noexcept {
    switch (status) {
      case video::sbs_telemetry_sample_status_e::ready:
        return sbs_telemetry_status_e::ok;
      case video::sbs_telemetry_sample_status_e::unavailable:
        return sbs_telemetry_status_e::unavailable;
      case video::sbs_telemetry_sample_status_e::failed:
        return sbs_telemetry_status_e::failed;
    }
    return sbs_telemetry_status_e::failed;
  }

  bool encode_sbs_telemetry_state_payload(
    const sbs_telemetry_state_t &state,
    std::uint8_t (&out)[SBS_TELEMETRY_STATE_PAYLOAD_SIZE]
  ) {
    const auto status = static_cast<std::uint8_t>(state.status);
    if (status > static_cast<std::uint8_t>(sbs_telemetry_status_e::failed)) {
      return false;
    }

    const auto &snapshot = state.snapshot;
    const auto performance = snapshot.performance ? snapshot.performance->snapshot() : host_sbs_telemetry::sample {};
    const std::array<float, 8> float_fields {
      snapshot.pop_floor,
      snapshot.pop_ceiling,
      snapshot.effective_pop,
      snapshot.edge_fraction,
      snapshot.zero_anchor_shift_px,
      snapshot.subject_depth,
      snapshot.valid_depth_fraction,
      snapshot.effective_range_width,
    };
    if (std::any_of(float_fields.begin(), float_fields.end(), [](float value) {
          return !std::isfinite(value);
        })) {
      return false;
    }

    std::array<std::uint8_t, SBS_TELEMETRY_STATE_PAYLOAD_SIZE> encoded {};
    const auto write_u16 = [&](std::size_t offset, std::uint16_t value) {
      encoded[offset] = static_cast<std::uint8_t>(value & 0xffu);
      encoded[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
    };
    const auto write_u32 = [&](std::size_t offset, std::uint32_t value) {
      encoded[offset] = static_cast<std::uint8_t>(value & 0xffu);
      encoded[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
      encoded[offset + 2] = static_cast<std::uint8_t>((value >> 16) & 0xffu);
      encoded[offset + 3] = static_cast<std::uint8_t>((value >> 24) & 0xffu);
    };
    const auto write_f32 = [&](std::size_t offset, float value) {
      write_u32(offset, std::bit_cast<std::uint32_t>(value));
    };
    const auto write_u64 = [&](std::size_t offset, std::uint64_t value) {
      write_u32(offset, static_cast<std::uint32_t>(value));
      write_u32(offset + 4, static_cast<std::uint32_t>(value >> 32));
    };

    encoded[0] = SBS_TELEMETRY_VERSION;
    encoded[1] = status;
    write_u16(2, state.request_id);
    write_u32(4, snapshot.generation);
    write_u32(8, snapshot.sequence);
    write_u32(12, snapshot.valid_fields | performance.valid_fields);
    write_u32(16, snapshot.runtime_flags);
    write_u16(20, snapshot.depth_width);
    write_u16(22, snapshot.depth_height);
    encoded[24] = snapshot.zero_plane_mode;
    // encoded[25..27] remain reserved zero.
    write_f32(28, snapshot.pop_floor);
    write_f32(32, snapshot.pop_ceiling);
    write_f32(36, snapshot.effective_pop);
    write_f32(40, snapshot.edge_fraction);
    // encoded[44..47] stays zero with valid bit 3 clear; protocol 13 clients parse a float there.
    write_f32(48, snapshot.zero_anchor_shift_px);
    write_f32(52, snapshot.subject_depth);
    write_f32(56, snapshot.valid_depth_fraction);
    write_f32(60, snapshot.effective_range_width);
    // encoded[64..75] stay zero with valid bits 8 and 9 clear; protocol 13 clients parse three
    // uint32 words there.
    write_u32(76, snapshot.empty_raw_count);
    write_u32(80, snapshot.collapsed_raw_count);
    write_u32(84, snapshot.sampled_frame_id);
    write_u32(88, performance.outcome_epoch);
    write_u32(92, performance.outcome_sequence);
    write_u32(96, performance.outcome_host_ms);
    write_u64(104, performance.outcomes.infer);
    write_u64(112, performance.outcomes.reuse);
    write_u64(120, performance.outcomes.invalid);
    write_u32(128, performance.output_host_ms);
    write_u32(132, performance.warped);
    write_u32(136, performance.repeated);
    write_u32(140, performance.flat);
    for (std::size_t i = 0; i < performance.stages.size(); ++i) {
      write_f32(144 + i * 12, performance.stages[i].mean_ms);
      write_u32(148 + i * 12, performance.stages[i].count);
      write_u32(152 + i * 12, performance.stages[i].latest_host_ms);
    }
    std::copy(encoded.begin(), encoded.end(), out);
    return true;
  }

  using message_queue_t = std::shared_ptr<safe::queue_t<std::pair<udp::endpoint, std::string>>>;

  struct av_ping_route_t {
    std::uint32_t id;
    std::string payload;
    message_queue_t messages;
  };

  struct av_ping_route_update_t {
    socket_e socket_type;
    std::uint32_t id;
    std::optional<av_ping_route_t> route;
  };

  using av_ping_route_queue_t = std::shared_ptr<safe::queue_t<av_ping_route_update_t>>;

  class control_server_t {
  public:
    struct session_slot_t {
      session_t *session {};
      net::peer_t peer {};
    };

    int bind(net::af_e address_family, std::uint16_t port) {
      _host = net::host_create(address_family, _addr, port);

      return !(bool) _host;
    }

    // Return the sole registered session when this peer matches its established connection or
    // its pending connect-data/address identity. A different peer can never claim the active slot.
    session_t *get_session(const net::peer_t peer, uint32_t connect_data);

    // Circular dependency:
    //   iterate refers to session
    //   session refers to broadcast_ctx_t
    //   broadcast_ctx_t refers to control_server_t
    // Therefore, iterate is implemented further down the source file
    void iterate(std::chrono::milliseconds timeout);

    /**
     * @brief Call the handler for a given control stream message.
     * @param type The message type.
     * @param session The session the message was received on.
     * @param payload The payload of the message.
     * @param reinjected `true` if this message is being reprocessed after decryption.
     */
    void call(std::uint16_t type, session_t *session, const std::string_view &payload, bool reinjected);

    void map(uint16_t type, std::function<void(session_t *, const std::string_view &)> cb) {
      _map_type_cb.emplace(type, std::move(cb));
    }

    void map(uint16_t type, std::size_t minimum_payload_size, std::function<void(session_t *, const std::string_view &)> cb) {
      map(type, [type, minimum_payload_size, cb = std::move(cb)](session_t *session, const std::string_view &payload) {
        if (payload.size() < minimum_payload_size) {
          BOOST_LOG(warning) << "Dropping runt control message "sv << util::hex(type).to_string_view()
                             << ": expected at least "sv << minimum_payload_size
                             << " payload bytes, got "sv << payload.size();
          return;
        }

        cb(session, payload);
      });
    }

    int send(const std::string_view &payload, net::peer_t peer) {
      auto packet = enet_packet_create(payload.data(), payload.size(), ENET_PACKET_FLAG_RELIABLE);
      if (enet_peer_send(peer, 0, packet)) {
        enet_packet_destroy(packet);

        return -1;
      }

      return 0;
    }

    int send_unreliable_sequenced(const std::string_view &payload, net::peer_t peer) {
      // ENet packets without RELIABLE or UNSEQUENCED use unreliable-sequenced delivery. Moonlight
      // reserves channel 1 for urgent IDR/reference-frame traffic; its server-control lane is 8.
      // Use that only when it was actually negotiated, with channel 0 as the legacy fallback.
      constexpr enet_uint8 server_control_channel = 0x08;
      const enet_uint8 channel =
        peer->channelCount > server_control_channel ? server_control_channel : 0;
      auto packet = enet_packet_create(payload.data(), payload.size(), 0);
      if (enet_peer_send(peer, channel, packet)) {
        enet_packet_destroy(packet);
        return -1;
      }
      return 0;
    }

    void flush() {
      enet_host_flush(_host.get());
    }

    // Callbacks
    std::unordered_map<std::uint16_t, std::function<void(session_t *, const std::string_view &)>> _map_type_cb;

    // Apollo admits one remote stream at a time. Keep session and peer in the same synchronized
    // slot so publication, peer claim, and removal are one transaction.
    sync_util::sync_t<session_slot_t> _session;

    ENetAddress _addr;
    net::host_t _host;
  };

  struct broadcast_ctx_t {
    av_ping_route_queue_t av_ping_route_updates;

    std::thread recv_thread;
    std::thread control_thread;
    std::chrono::steady_clock::time_point video_epoch;

    asio::io_context io_context;

    udp::socket video_sock {io_context};
    udp::socket audio_sock {io_context};

    control_server_t control_server;
  };

  // Each sender is bound to its session's send-side state until that sender has joined.
  struct video_channel_t {
    int packet_size;
    int min_required_fec_packets;
    // Published by the encode loop only after a real encoder exists. The packed bitrate/cadence
    // snapshot prevents the packet sender from pacing a rolled-back request or observing a torn
    // pair across a live 0x3007 transition.
    std::shared_ptr<video::effective_video_mode_publisher_t> effective_mode;
    boost::asio::ip::address local_address;
    std::atomic_bool active {true};
    std::atomic_bool pacing_plan_logged {false};
    std::atomic_bool idr_pacing_plan_logged {false};
    std::shared_ptr<std::atomic<bool>> sbs_debug_dump_pending =
      std::make_shared<std::atomic<bool>>(false);
    bool awaiting_recovery_idr {false};
    bool source_frame_id_negotiated {false};
    source_frame_id_t source_frame_id;

    std::string ping_payload;
    std::uint32_t lowseq;
    udp::endpoint peer;
    crypto::cipher::gcm_t cipher;
    std::uint64_t gcm_iv_counter;
    safe::mail_raw_t::event_t<bool> idr_events;
    safe::mail_raw_t::event_t<std::pair<int64_t, int64_t>> invalidate_ref_frames_events;
    // Shared with the encode loop through config_t::encode_wake.
    std::shared_ptr<video::encode_wake_t> encode_wake = std::make_shared<video::encode_wake_t>();
    std::unique_ptr<platf::deinit_t> qos;

    /** Call after raising a request on the session mail that the encode loop services. */
    void wake_encoder() const {
      encode_wake->wake();
    }
  };

  struct audio_channel_t {
    int packet_duration;
    boost::asio::ip::address local_address;
    std::atomic_bool active {true};

    crypto::cipher::cbc_t cipher;
    std::string ping_payload;
    std::uint16_t sequence_number;
    std::uint32_t av_ri_key_id;
    std::uint32_t timestamp;
    udp::endpoint peer;
    util::buffer_t<char> shards;
    util::buffer_t<uint8_t *> shard_ptrs;
    audio_fec_packet_t fec_packet;
    std::unique_ptr<platf::deinit_t> qos;
  };

  class session_media_senders_t {
  public:
    session_media_senders_t() = default;

    ~session_media_senders_t() {
      stop_and_join();
    }

    session_media_senders_t(const session_media_senders_t &) = delete;
    session_media_senders_t &operator=(const session_media_senders_t &) = delete;
    session_media_senders_t(session_media_senders_t &&) = delete;
    session_media_senders_t &operator=(session_media_senders_t &&) = delete;

    const video::packet_queue_t &video_packets() const noexcept {
      return video_packets_;
    }

    const audio::packet_queue_t &audio_packets() const noexcept {
      return audio_packets_;
    }

    void adopt_video(std::thread worker) noexcept {
      video_thread_ = std::move(worker);
    }

    void adopt_audio(std::thread worker) noexcept {
      audio_thread_ = std::move(worker);
    }

    void stop_and_join() {
      if (video_packets_) {
        video_packets_->stop();
      }
      if (audio_packets_) {
        audio_packets_->stop();
      }
      if (video_thread_.joinable()) {
        video_thread_.join();
      }
      if (audio_thread_.joinable()) {
        audio_thread_.join();
      }
      // A stopped queue retains unread packets. Release them even if session_t is retained.
      video_packets_.reset();
      audio_packets_.reset();
    }

  private:
    video::packet_queue_t video_packets_ = std::make_shared<safe::queue_t<video::packet_t>>(video::ENCODED_PACKET_QUEUE_LIMIT);
    audio::packet_queue_t audio_packets_ = std::make_shared<safe::queue_t<audio::packet_t>>(audio::ENCODED_PACKET_QUEUE_LIMIT);
    std::thread video_thread_;
    std::thread audio_thread_;
  };

  // Everything needed to finish or reject one asynchronous live display-mode request. These
  // values deliberately do not retain session_t: the session owns and joins the worker instead.
  using live_video_mode_request_t = detail::atomic_presentation_t::request_t;

  struct live_video_mode_worker_t {
    // Serialize thread publication with stop's barrier; the state owns no thread/session.
    std::mutex thread_mutex;
    std::shared_ptr<detail::atomic_presentation_t> transactions =
      std::make_shared<detail::atomic_presentation_t>();
#ifdef SUNSHINE_TESTS
    std::shared_ptr<const testing::atomic_presentation_process_hooks_t> process_hooks;
#endif
    std::thread thread;

    void stop() {
      // Close admission and drop pending dependencies without either owner lock held. Then
      // wait out any dispatch that passed its stopped check before publishing the thread.
      transactions->stop();
      std::lock_guard barrier(thread_mutex);
    }

    void join() {
      stop();
      if (thread.joinable()) {
        BOOST_LOG(debug) << "Waiting for live video-mode worker to end..."sv;
        thread.join();
      }
      transactions->clear_stopped();
    }

    ~live_video_mode_worker_t() {
      join();
    }
  };

  struct session_t {
    config_t config;
    std::shared_ptr<microphone::session_t> microphone_receiver;

    safe::mail_t mail;

    std::shared_ptr<input::input_t> input;

    std::thread audioThread;
    std::thread videoThread;

    std::chrono::steady_clock::time_point pingTimeout;

    safe::shared_t<broadcast_ctx_t>::ptr_t broadcast_ref;

    std::shared_ptr<video_channel_t> video;
    std::shared_ptr<audio_channel_t> audio;

    struct {
      crypto::cipher::gcm_t cipher;
      crypto::aes_t incoming_iv;
      crypto::aes_t outgoing_iv;

      std::uint32_t connect_data;

      net::peer_t peer;
      std::uint32_t seq;

      platf::feedback_queue_t feedback_queue;
      platf::feedback_queue_t haptics_feedback_queue;
      safe::mail_raw_t::event_t<video::hdr_info_t> hdr_queue;
      // Own these posts across capture startup and encoder handoffs instead of resolving weak
      // mail entries on every control iteration.
      safe::mail_raw_t::event_t<int> depth_status_event;
      // mail_raw_t keeps only a weak reference to each post. Retain this queue for the full
      // session lifetime so worker/encode-thread replies cannot disappear between raise() and
      // the control thread's next drain.
      safe::mail_raw_t::queue_t<live_video_mode_ack_t> live_video_mode_ack_queue;
      // Retained for the same reason: a request committed before the encode loop has subscribed
      // would otherwise be destroyed with the temporary post and never acknowledged.
      safe::mail_raw_t::queue_t<video::video_mode_change_t> video_mode_queue;
      safe::mail_raw_t::queue_t<video::stream_gamma_request_t> stream_gamma_queue;
      safe::mail_raw_t::queue_t<video::stream_gamma_ack_t> stream_gamma_ack_queue;
      std::vector<video::stream_gamma_ack_t> deferred_stream_gamma_acks;
      unsigned outstanding_stream_gamma_requests = 0;
      // A syntactically valid v2 control message can arrive just before the initial encoder has
      // published generation 1. Preserve any immediate validation refusal and correlate it once
      // a complete applied state exists; v2 never emits an unproven generation-zero ACK.
      std::vector<live_video_mode_ack_t> deferred_live_video_mode_acks;
      safe::mail_raw_t::event_t<video::sbs_telemetry_snapshot_t> sbs_telemetry_event;
      safe::mail_raw_t::event_t<video::game_source_state_t> game_source_status_event;
      std::optional<video::sbs_telemetry_snapshot_t> latest_sbs_telemetry;
      std::shared_ptr<video::sbs_telemetry_subscription_t> sbs_telemetry_subscription;
      std::chrono::steady_clock::time_point last_sbs_telemetry_periodic {};
      int last_depth_phase = -1;  // last depth-engine phase pushed to this client (see IDX_DEPTH_STATUS)
    } control;

    std::uint32_t launch_session_id;
    mutable std::mutex device_info_mutex;
    std::string device_name;
    std::string device_uuid;
    std::uint64_t client_policy_generation;
    std::atomic<crypto::PERM> permission;
    // Session-shared Host SBS mode. Atomic-presentation v2 updates it with the immutable mode
    // transaction; Dump 3D is routed to the active AI or Game provider, including Game waiting.
    std::shared_ptr<std::atomic<int>> requested_sbs_mode =
      std::make_shared<std::atomic<int>>(::video::SBS_OFF);

    safe::mail_raw_t::event_t<bool> shutdown_event;
    safe::signal_t controlEnd;

    // The client can send its audio/video UDP ping while platform resume work is still changing
    // the display topology, and before ENet publishes the local source address. Capture must not
    // feed the session media queues until both halves of session startup are complete.
    safe::signal_t startup_ready;
    safe::signal_t control_ready;

    std::mutex stop_mutex;
    std::atomic_bool graceful_stop_requested {false};
    std::atomic<session::state_e> state;

    // Join senders before their channels, mail and shared socket owner can be destroyed.
    session_media_senders_t media_senders;

    // Keep this last: member destruction runs in reverse declaration order, so its defensive
    // destructor joins before any mail/video state captured by the worker can be destroyed.
    live_video_mode_worker_t live_video_mode_worker;
  };

  namespace {
    /**
     * Queue one acknowledgement for the control thread to encrypt and send.
     *
     * This is the only step any off-control thread performs. The control thread owns outgoing
     * control encryption and every ENet call, and it is also the thread that clears
     * session_t::control.peer and the server's session slot, so handing it a plain value keeps
     * the peer and GCM state from ever being touched by a worker. session_t::control retains the
     * queue's strong owner; mail_raw_t itself stores only a weak reference, so a producer-only
     * temporary would otherwise destroy the acknowledgement before the control thread could drain
     * it. If the session has already gone away, the queue dies with it and the client's own
     * timeout covers the missing reply.
     */
    void queue_live_video_mode_ack(const safe::mail_t &mail, const live_video_mode_ack_t &ack) {
      mail->queue<live_video_mode_ack_t>(mail::live_video_mode_ack, LIVE_VIDEO_MODE_ACK_QUEUE_LIMIT)->raise(ack);
    }

    /**
     * Answer a request that never reached the encode loop. Only the encoder knows the geometry it
     * installed, so a refusal reports the last mode it published instead.
     */
    void reject_live_video_mode(
      const safe::mail_t &mail,
      const std::shared_ptr<video::effective_video_mode_publisher_t> &effective_mode,
      std::uint32_t request_id,
      live_video_mode_ack_e status
    ) {
      const auto mode = effective_mode ?
                          effective_mode->current() :
                          video::effective_video_mode_t {};
      queue_live_video_mode_ack(
        mail,
        make_live_video_mode_ack(request_id, status, mode)
      );
    }

    /** Hand one serialized request to the encode loop. */
    void commit_live_video_mode(const live_video_mode_request_t &request) {
      // Do not publish pacing here. The requested encoder may fail, so video.cpp publishes the
      // coherent effective bitrate/cadence pair only after init_encoder() succeeds.
      request.mail->queue<video::video_mode_change_t>(mail::video_mode)->raise(request.change);
      request.video->wake_encoder();
    }

    /**
     * Serialize a live request through display apply, encoder creation and (on failure) display
     * rollback. Only an unstarted pending request may be coalesced.
     */
    void dispatch_live_video_mode(session_t &session, live_video_mode_request_t request) {
      auto &worker_state = session.live_video_mode_worker;
      const auto submitted = worker_state.transactions->submit(std::move(request));
      const auto &superseded = submitted.superseded;
      const auto &rejected = submitted.rejected;

      if (rejected) {
        reject_live_video_mode(
          rejected->mail,
          rejected->effective_mode,
          rejected->change.request_id,
          live_video_mode_ack_e::failed
        );
        return;
      }

      // A queued request that a newer one displaced was never applied, so it is answered as a
      // retryable failure carrying the mode still in effect. Without this a client that changed
      // its mind would be left waiting out a timeout for a reply that never comes.
      if (superseded) {
        reject_live_video_mode(
          superseded->mail,
          superseded->effective_mode,
          superseded->change.request_id,
          live_video_mode_ack_e::failed
        );
      }

      // Stop closes admission before waiting on this publication lock. A disconnect therefore
      // cannot miss a thread whose constructor passed the stopped check while it was racing.
      std::unique_lock worker_lock(worker_state.thread_mutex);
      if (worker_state.transactions->stopping()) {
        return;
      }
      if (worker_state.thread.joinable()) {
        worker_lock.unlock();
        worker_state.transactions->notify_pending();
        return;
      }
      worker_state.thread = std::thread([&session]() {
        auto &state = session.live_video_mode_worker;
        auto &transactions = *state.transactions;
        while (true) {
          auto job = transactions.wait_next();
          if (!job) {
            return;
          }
          const auto transaction_id = job->change.transaction_id;

          // The session can end while a display transition is in flight. Never resize a desktop
          // for a stream that is already gone. The acknowledgement is dropped with it; the client
          // is disconnected at that point and its own timeout covers the missing reply.
          if (job->mail->event<bool>(mail::shutdown)->peek()) {
            (void) transactions.finish(transaction_id);
            continue;
          }

          const auto host_sbs_rejects_geometry = [&]() {
            const int desired_sbs_mode = job->change.requested_sbs_mode;
            return desired_sbs_mode == ::video::SBS_AI &&
                   !models::host_sbs_v2_source_resolution_is_supported(
                     static_cast<std::uint32_t>(job->change.width),
                     static_cast<std::uint32_t>(job->change.height)
                   );
          };
          // Recheck the immutable requested mode's geometry before touching the desktop. This
          // preflight and the encoder's final checks retain their own authority.
          if (host_sbs_rejects_geometry()) {
            BOOST_LOG(warning)
              << "Rejecting live video mode "sv << job->change.width << 'x'
              << job->change.height
              << " because the requested Host SBS V2 geometry is unsupported."sv;
            reject_live_video_mode(
              job->mail,
              job->effective_mode,
              job->change.request_id,
              live_video_mode_ack_e::rejected_invalid
            );
            (void) transactions.finish(transaction_id);
            continue;
          }

          auto result = proc::live_video_mode_result_e::unchanged;
          if (
#ifdef SUNSHINE_TESTS
            state.process_hooks ? state.process_hooks->needs_display_change(job->change.width, job->change.height, job->change.encodingFramerate) :
#endif
                                  proc::proc.live_video_mode_needs_display_change(job->change.width, job->change.height, job->change.encodingFramerate)
          ) {
            result =
#ifdef SUNSHINE_TESTS
              state.process_hooks ? state.process_hooks->apply(
                                      job->change.width,
                                      job->change.height,
                                      job->change.encodingFramerate,
                                      job->launch_session_id
                                    ) :
#endif
                                    proc::proc.apply_live_video_mode(
                                      job->change.width,
                                      job->change.height,
                                      job->change.encodingFramerate,
                                      job->launch_session_id
                                    );
          }
          const bool display_changed =
            result == proc::live_video_mode_result_e::applied;

          if (transactions.stopping()) {
            return;
          }
          switch (result) {
            case proc::live_video_mode_result_e::applied:
            case proc::live_video_mode_result_e::unchanged:
              if (job->mail->event<bool>(mail::shutdown)->peek()) {
                (void) transactions.finish(transaction_id);
                continue;
              }
              // The desktop is ready. Keep the transaction active until the encode loop proves
              // which mode exists and, on failure, the worker restores the prior desktop.
              commit_live_video_mode(*job);
              break;
            case proc::live_video_mode_result_e::needs_reconnect:
              BOOST_LOG(warning) << "type [IDX_SET_VIDEO_MODE]: "sv << job->change.width << 'x'
                                 << job->change.height << " cannot be applied without reconnecting for ["sv
                                 << job->client_name
                                 << "]; the client must reconnect before trusting presentation state."sv;
              reject_live_video_mode(
                job->mail,
                job->effective_mode,
                job->change.request_id,
                live_video_mode_ack_status(result)
              );
              (void) transactions.finish(transaction_id);
              break;
            case proc::live_video_mode_result_e::failed:
              BOOST_LOG(warning) << "type [IDX_SET_VIDEO_MODE]: the desktop could not be resized to "sv
                                 << job->change.width << 'x' << job->change.height << " for ["sv
                                 << job->client_name << "]; the stream keeps its current mode."sv;
              reject_live_video_mode(
                job->mail,
                job->effective_mode,
                job->change.request_id,
                live_video_mode_ack_status(result)
              );
              (void) transactions.finish(transaction_id);
              break;
          }

          if (result != proc::live_video_mode_result_e::applied && result != proc::live_video_mode_result_e::unchanged) {
            continue;
          }

          auto completion = transactions.wait_completion(transaction_id);
          if (!completion) {
            return;
          }

          const auto matching_request = std::find_if(
            completion->requests.begin(),
            completion->requests.end(),
            [&](const video::video_mode_request_ref_t &request) {
              return request.transaction_id == transaction_id;
            }
          );
          if (matching_request == completion->requests.end()) {
            BOOST_LOG(error) << "Live video-mode completion lost its active transaction."sv;
          } else if (matching_request->request_id != job->change.request_id) {
            BOOST_LOG(error) << "Live video-mode completion matched a token but not its client request id."sv;
          }

          bool rollback_restored_desktop = true;
          if (!completion->applied && display_changed) {
            BOOST_LOG(warning) << "Restoring the last encodable virtual desktop mode "
                               << completion->desktop_mode.width << 'x'
                               << completion->desktop_mode.height << " after the encoder rejected "
                               << job->change.width << 'x' << job->change.height << '.';
#ifdef SUNSHINE_TESTS
            const auto rollback_result = state.process_hooks ?
                                           state.process_hooks->apply(
                                             completion->desktop_mode.width,
                                             completion->desktop_mode.height,
                                             completion->desktop_mode.framerate_millihz,
                                             job->launch_session_id
                                           ) :
                                           proc::proc.apply_live_video_mode(
                                             completion->desktop_mode.width,
                                             completion->desktop_mode.height,
                                             completion->desktop_mode.framerate_millihz,
                                             job->launch_session_id
                                           );
#else
            const auto rollback_result = proc::proc.apply_live_video_mode(
              completion->desktop_mode.width,
              completion->desktop_mode.height,
              completion->desktop_mode.framerate_millihz,
              job->launch_session_id
            );
#endif
            if (rollback_result != proc::live_video_mode_result_e::applied && rollback_result != proc::live_video_mode_result_e::unchanged) {
              rollback_restored_desktop = false;
              BOOST_LOG(error) << "The virtual desktop could not be restored after a live encoder "
                                  "rollback; the client will be told to reconnect."sv;
            }
          }

          // Re-arm diagnostics only after the effective pacing pair is known. Resetting at request
          // enqueue lets the old encoder consume the latch before the new encoder is proven.
          job->video->pacing_plan_logged.store(false, std::memory_order_release);
          job->video->idr_pacing_plan_logged.store(false, std::memory_order_release);
          queue_live_video_mode_ack(
            job->mail,
            make_live_video_mode_ack(
              job->change.request_id,
              completion->applied ?
                live_video_mode_ack_e::applied :
                (rollback_restored_desktop ?
                   live_video_mode_ack_e::failed :
                   live_video_mode_ack_e::rejected_needs_reconnect),
              completion->mode
            )
          );

          if (!transactions.finish(transaction_id)) {
            BOOST_LOG(error) << "Live video-mode transaction gate rejected its active completion."sv;
          }
        }
      });
      worker_lock.unlock();
      worker_state.transactions->notify_pending();
    }

    void stop_live_video_mode_worker(session_t &session) {
      session.live_video_mode_worker.stop();
    }

    void join_live_video_mode_worker(session_t &session) {
      session.live_video_mode_worker.join();
    }
  }  // namespace

#ifdef SUNSHINE_TESTS
  namespace testing {
    struct atomic_presentation_fixture_t::impl_t {
      impl_t(video::effective_video_mode_t initial, atomic_presentation_process_hooks_t hooks) {
        if (!hooks.needs_display_change || !hooks.apply) {
          throw std::invalid_argument("The presentation fixture requires both inert process edges");
        }
        session.mail = std::make_shared<safe::mail_raw_t>();
        session.shutdown_event = session.mail->event<bool>(mail::shutdown);
        session.video = std::make_shared<video_channel_t>();
        session.config.monitor.effective_mode = std::make_shared<video::effective_video_mode_publisher_t>(initial);
        session.video->effective_mode = session.config.monitor.effective_mode;
        session.config.monitor.atomic_presentation = session.live_video_mode_worker.transactions;
        session.video->pacing_plan_logged.store(true);
        session.video->idr_pacing_plan_logged.store(true);
        session.control.video_mode_queue = session.mail->queue<video::video_mode_change_t>(mail::video_mode);
        session.control.live_video_mode_ack_queue = session.mail->queue<live_video_mode_ack_t>(mail::live_video_mode_ack, LIVE_VIDEO_MODE_ACK_QUEUE_LIMIT);
        session.launch_session_id = 0x70000001U;
        session.device_name = "atomic presentation fixture";
        session.live_video_mode_worker.process_hooks = std::make_shared<const atomic_presentation_process_hooks_t>(std::move(hooks));
        session.video->encode_wake->install([this] {
          std::unique_lock lock(wake_mutex);
          if (!wake_armed) {
            return;
          }
          wake_armed = false;
          wake_entered = true;
          wake_changed.notify_all();
          if (!wake_changed.wait_for(lock, 10s, [this] {
                return wake_released;
              })) {
            wake_timed_out = true;
          }
        });
      }

      ~impl_t() {
        release_wake();
        stop_live_video_mode_worker(session);
        join_live_video_mode_worker(session);
      }

      void release_wake() {
        std::lock_guard lock(wake_mutex);
        wake_armed = false;
        wake_released = true;
        wake_changed.notify_all();
      }

      mutable std::mutex wake_mutex;
      std::condition_variable wake_changed;
      bool wake_armed = false;
      bool wake_entered = false;
      bool wake_released = true;
      bool wake_timed_out = false;
      // Its real worker is joined before the gate or any captured session state can die.
      session_t session {};
    };

    atomic_presentation_fixture_t::atomic_presentation_fixture_t(video::effective_video_mode_t initial, atomic_presentation_process_hooks_t hooks):
        impl_(std::make_unique<impl_t>(initial, std::move(hooks))) {}

    atomic_presentation_fixture_t::~atomic_presentation_fixture_t() = default;

    void atomic_presentation_fixture_t::submit(video::video_mode_change_t change) {
      auto &session = impl_->session;
      dispatch_live_video_mode(session, {session.mail, session.video, session.config.monitor.effective_mode, change, session.launch_session_id, session.device_name});
    }

    std::optional<video::video_mode_change_t> atomic_presentation_fixture_t::pop_encode_request(std::chrono::milliseconds timeout) {
      return impl_->session.control.video_mode_queue->pop(timeout);
    }

    void atomic_presentation_fixture_t::publish_effective_mode(video::effective_video_mode_t mode) {
      impl_->session.config.monitor.effective_mode->publish(mode);
    }

    void atomic_presentation_fixture_t::deliver_completion(video::video_mode_applied_t completion) {
      (void) impl_->session.config.monitor.atomic_presentation->complete(std::move(completion));
    }

    std::optional<live_video_mode_ack_t> atomic_presentation_fixture_t::pop_ack(std::chrono::milliseconds timeout) {
      return impl_->session.control.live_video_mode_ack_queue->pop(timeout);
    }

    void atomic_presentation_fixture_t::gate_next_encode_wake() {
      std::lock_guard lock(impl_->wake_mutex);
      impl_->wake_armed = true;
      impl_->wake_entered = false;
      impl_->wake_released = false;
    }

    bool atomic_presentation_fixture_t::wait_encode_wake(std::chrono::milliseconds timeout) {
      std::unique_lock lock(impl_->wake_mutex);
      return impl_->wake_changed.wait_for(lock, timeout, [this] {
        return impl_->wake_entered;
      });
    }

    void atomic_presentation_fixture_t::release_encode_wake() {
      impl_->release_wake();
    }

    bool atomic_presentation_fixture_t::gate_timed_out() const {
      std::lock_guard lock(impl_->wake_mutex);
      return impl_->wake_timed_out;
    }

    void atomic_presentation_fixture_t::stop() {
      stop_live_video_mode_worker(impl_->session);
      impl_->session.shutdown_event->raise(true);
    }

    void atomic_presentation_fixture_t::join() {
      join_live_video_mode_worker(impl_->session);
    }

    bool atomic_presentation_fixture_t::stopping() const {
      return impl_->session.live_video_mode_worker.transactions->stopping();
    }

    std::uint32_t atomic_presentation_fixture_t::launch_session_id() const {
      return impl_->session.launch_session_id;
    }

    bool atomic_presentation_fixture_t::pacing_diagnostics_rearmed() const {
      const auto &video = impl_->session.video;
      return !video->pacing_plan_logged.load(std::memory_order_acquire) && !video->idr_pacing_plan_logged.load(std::memory_order_acquire);
    }
  }  // namespace testing
#endif

  /**
   * First part of cipher must be struct of type control_encrypted_t
   *
   * returns empty string_view on failure
   * returns string_view pointing to payload data
   */
  template<std::size_t max_payload_size>
  static inline std::string_view encode_control(session_t *session, const std::string_view &plaintext, std::array<std::uint8_t, max_payload_size> &tagged_cipher) {
    static_assert(
      max_payload_size >= sizeof(control_encrypted_t) + crypto::cipher::tag_size,
      "max_payload_size >= sizeof(control_encrypted_t) + crypto::cipher::tag_size"
    );

    auto seq = session->control.seq++;

    auto &iv = session->control.outgoing_iv;
    // We use the deterministic IV construction algorithm specified in NIST SP 800-38D
    // Section 8.2.1. The sequence number is our "invocation" field and the 'CH' in the
    // high bytes is the "fixed" field. Because each client provides their own unique
    // key, our values in the fixed field need only uniquely identify each independent
    // use of the client's key with AES-GCM in our code.
    //
    // The sequence number is 32 bits long which allows for 2^32 control stream messages
    // to be sent to each client before the IV repeats.
    iv.resize(12);
    std::copy_n((uint8_t *) &seq, sizeof(seq), std::begin(iv));
    iv[10] = 'H';  // Host originated
    iv[11] = 'C';  // Control stream

    auto packet = (control_encrypted_p) tagged_cipher.data();

    if (plaintext.empty() || !session->control.cipher.encrypt(plaintext, packet->payload(), &iv)) {
      BOOST_LOG(error) << "Couldn't encrypt control data"sv;
      return {};
    }

    std::uint16_t packet_length = plaintext.size() + crypto::cipher::tag_size + sizeof(control_encrypted_t::seq);

    packet->encryptedHeaderType = util::endian::little(0x0001);
    packet->length = util::endian::little(packet_length);
    packet->seq = util::endian::little(seq);

    return std::string_view {(char *) tagged_cipher.data(), packet_length + sizeof(control_encrypted_t) - sizeof(control_encrypted_t::seq)};
  }

  enum class control_delivery_e {
    reliable,
    unreliable_sequenced,
  };

  template<typename Plaintext>
  int send_control_packet(
    session_t *session,
    control_server_t &server,
    const Plaintext &plaintext,
    const control_delivery_e delivery = control_delivery_e::reliable
  ) {
    if (!session->control.peer) {
      return -1;
    }
    std::array<
      std::uint8_t,
      sizeof(control_encrypted_t) +
        crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) +
        crypto::cipher::tag_size>
      encrypted_payload;
    const auto payload =
      encode_control(session, util::view(plaintext), encrypted_payload);
    if (payload.empty()) {
      return -1;
    }
    return delivery == control_delivery_e::reliable ?
             server.send(payload, session->control.peer) :
             server.send_unreliable_sequenced(payload, session->control.peer);
  }

  int start_broadcast(broadcast_ctx_t &ctx);
  void end_broadcast(broadcast_ctx_t &ctx);

  static auto broadcast = safe::make_shared<broadcast_ctx_t>(start_broadcast, end_broadcast);

  session_t *control_server_t::get_session(const net::peer_t peer, uint32_t connect_data) {
    auto slot_lock = _session.lock();
    auto *session = _session->session;
    if (!session) {
      return nullptr;
    }

    if (_session->peer) {
      return _session->peer == peer ? session : nullptr;
    }

    // The sole session has not established its ENet peer yet. Modern Artemis echoes the random
    // connect-data value from SETUP, so source address is neither identity nor authorization.
    TUPLE_2D(peer_port, peer_addr, platf::from_sockaddr_ex((sockaddr *) &peer->address.address));
    if (session->control.connect_data != connect_data) {
      return nullptr;
    }
    BOOST_LOG(debug) << "Initialized control stream session by connect-data match"sv;

    // Once the control stream connection is established, RTSP session state can be torn down.
    rtsp_stream::launch_session_clear(session->launch_session_id);

    _session->peer = peer;
    session->control.peer = peer;

    // Use the local address from the control connection as the source address for other
    // communications to the client. This is necessary for correct routing on multi-homed hosts.
    auto local_address = platf::from_sockaddr((sockaddr *) &peer->localAddress.address);
    auto parsed_local_address = boost::asio::ip::make_address(local_address);
    session->video->local_address = parsed_local_address;
    session->audio->local_address = parsed_local_address;

    // Publish source addresses before waking A/V capture. The event mutex provides the release
    // ordering consumed by the capture and broadcast workers.
    session->control_ready.raise(true);

    BOOST_LOG(debug) << "Control local address ["sv << local_address << ']';
    BOOST_LOG(debug) << "Control peer address ["sv << peer_addr << ':' << peer_port << ']';
    return session;
  }

  /**
   * @brief Call the handler for a given control stream message.
   * @param type The message type.
   * @param session The session the message was received on.
   * @param payload The payload of the message.
   * @param reinjected `true` if this message is being reprocessed after decryption.
   */
  void control_server_t::call(std::uint16_t type, session_t *session, const std::string_view &payload, bool reinjected) {
    // The only outer wire message in the modern protocol is the authenticated envelope.
    if (!reinjected && type != control_packet::encrypted) {
      BOOST_LOG(error) << "Dropping unencrypted message on encrypted control stream: "sv << util::hex(type).to_string_view();
      return;
    }

    auto cb = _map_type_cb.find(type);
    if (cb == std::end(_map_type_cb)) {
      BOOST_LOG(debug)
        << "Unknown control message type "sv << util::hex(type).to_string_view()
        << " with "sv << payload.size() << " payload bytes"sv;
    } else {
      cb->second(session, payload);
    }
  }

  void control_server_t::iterate(std::chrono::milliseconds timeout) {
    ENetEvent event;
    auto res = enet_host_service(_host.get(), &event, timeout.count());

    if (res > 0) {
      auto session = get_session(event.peer, event.data);
      if (!session) {
        BOOST_LOG(warning) << "Rejected connection from ["sv << platf::from_sockaddr((sockaddr *) &event.peer->address.address) << "]: it's not properly set up"sv;
        enet_peer_disconnect_now(event.peer, 0);

        return;
      }

      session->pingTimeout = std::chrono::steady_clock::now() + config::stream.ping_timeout;

      switch (event.type) {
        case ENET_EVENT_TYPE_RECEIVE:
          {
            net::packet_t packet {event.packet};

            if (!is_valid_control_packet_size(packet->dataLength)) {
              BOOST_LOG(warning) << "Dropping invalid control packet of "sv << packet->dataLength << " bytes"sv;
              break;
            }

            std::uint16_t type;
            std::memcpy(&type, packet->data, sizeof(type));
            type = util::endian::little(type);
            std::string_view payload {(char *) packet->data + sizeof(type), packet->dataLength - sizeof(type)};

            call(type, session, payload, false);
          }
          break;
        case ENET_EVENT_TYPE_CONNECT:
          BOOST_LOG(info) << "CLIENT CONNECTED"sv;
          break;
        case ENET_EVENT_TYPE_DISCONNECT:
          BOOST_LOG(info) << "CLIENT DISCONNECTED"sv;
          // No more clients to send video data to ^_^
          if (session->state == session::state_e::RUNNING) {
            session::stop(*session);
          }
          break;
        case ENET_EVENT_TYPE_NONE:
          break;
      }
    }
  }

  constexpr std::size_t FEC_PACKET_INDEX_MAX = 1023;
  constexpr std::size_t VIDEO_RS_CONTEXT_CACHE_LIMIT = 8;
  constexpr std::size_t VIDEO_SEND_WORKSPACE_RETAIN_LIMIT = 16 * 1024 * 1024;

  // Keep the data and parity shard bases aligned for the widest Reed–Solomon SIMD backend.
  // Shards retain their negotiated wire stride, so alignment of individual shards depends on it.
  constexpr std::size_t VIDEO_SHARD_ALIGNMENT = 64;
  using video_shard_storage_t = std::vector<
    std::uint8_t,
    boost::alignment::aligned_allocator<std::uint8_t, VIDEO_SHARD_ALIGNMENT>>;

  namespace fec {
    // The video packet header has two bits for the number of blocks minus one.
    constexpr std::size_t MAX_BLOCKS = 4;

    struct block_plan_t {
      std::size_t data_shards;
      std::size_t parity_shards;
      std::size_t percentage;
    };

    struct frame_plan_t {
      std::size_t data_shards;
      std::size_t wire_shards;
      std::size_t block_count;
      std::size_t final_payload_bytes;
      std::array<block_plan_t, MAX_BLOCKS> blocks;
    };

    static std::optional<frame_plan_t> make_frame_plan(
      std::size_t picture_size,
      std::size_t payload_blocksize,
      int fec_percentage,
      std::size_t min_required_fec_packets
    ) {
      const auto tail_size = picture_size % payload_blocksize;
      const auto data_shards = picture_size / payload_blocksize + (tail_size != 0);
      // D data shards at rate F need D + ceil(D * F / 100) slots.
      const auto max_data_per_block = (RS8_DATA_SHARDS_MAX * 100) / (100 + fec_percentage);
      auto block_count = static_cast<std::size_t>(ceil_div_u64(data_shards, max_data_per_block));
      if (block_count > MAX_BLOCKS) {
        BOOST_LOG(warning) << "Skipping FEC for abnormally large encoded frame (needed "sv << block_count << " FEC blocks)"sv;
        fec_percentage = 0;
        block_count = MAX_BLOCKS;
      }

      // Split on shard boundaries. Only the final shard can contain padding.
      const std::size_t data_per_block = ceil_div_u64(data_shards, block_count);
      if (data_per_block > FEC_PACKET_INDEX_MAX) {
        BOOST_LOG(error) << "Dropping encoded frame that exceeds the 10-bit FEC packet index (needed "sv << data_per_block << " packets per block)"sv;
        return std::nullopt;
      }

      frame_plan_t plan {data_shards, data_shards, block_count, tail_size == 0 ? payload_blocksize : tail_size, {}};
      std::size_t remaining = data_shards;
      for (std::size_t index = 0; index < block_count; ++index) {
        const auto count = std::min(data_per_block, remaining);
        auto parity = (count * fec_percentage + 99) / 100;
        auto percentage = static_cast<std::size_t>(fec_percentage);
        if (percentage != 0 && parity < min_required_fec_packets) {
          parity = min_required_fec_packets;
          percentage = (100 * parity) / count;
        }
        plan.blocks[index] = {count, parity, percentage};
        plan.wire_shards += parity;
        remaining -= count;
      }
      return plan;
    }

    using rs_t = util::safe_ptr<reed_solomon, [](reed_solomon *rs) {
      reed_solomon_release(rs);
    }>;

    class rs_cache_t {
    public:
      reed_solomon *acquire(std::size_t data_shards, std::size_t parity_shards) {
        ++use_generation_;
        for (auto &entry : entries_) {
          if (entry.context && entry.data_shards == data_shards && entry.parity_shards == parity_shards) {
            entry.last_use = use_generation_;
            return entry.context.get();
          }
        }

        auto *replacement = &entries_.front();
        for (auto &entry : entries_) {
          if (!entry.context) {
            replacement = &entry;
            break;
          }
          if (entry.last_use < replacement->last_use) {
            replacement = &entry;
          }
        }

        rs_t context {reed_solomon_new(data_shards, parity_shards)};
        if (!context) {
          return nullptr;
        }

        replacement->data_shards = data_shards;
        replacement->parity_shards = parity_shards;
        replacement->last_use = use_generation_;
        replacement->context = std::move(context);
        return replacement->context.get();
      }

      void discard(std::size_t data_shards, std::size_t parity_shards) {
        for (auto &entry : entries_) {
          if (entry.context && entry.data_shards == data_shards && entry.parity_shards == parity_shards) {
            entry.context.reset();
            return;
          }
        }
      }

    private:
      struct entry_t {
        std::size_t data_shards = 0;
        std::size_t parity_shards = 0;
        std::uint64_t last_use = 0;
        rs_t context;
      };

      std::array<entry_t, VIDEO_RS_CONTEXT_CACHE_LIMIT> entries_;
      std::uint64_t use_generation_ = 0;
    };

    struct workspace_t {
      video_shard_storage_t parity;
      std::array<std::uint8_t *, RS8_DATA_SHARDS_MAX> rs_pointers;

      void release_oversized() {
        if (parity.capacity() > VIDEO_SEND_WORKSPACE_RETAIN_LIMIT) {
          video_shard_storage_t().swap(parity);
        }
      }
    };

    struct encoded_block_t {
      platf::udp_payload_block_t payloads;
      std::size_t percentage;
    };

    static encoded_block_t encode(
      std::uint8_t *payload,
      std::size_t blocksize,
      const block_plan_t &plan,
      rs_cache_t &rs_cache,
      workspace_t &workspace
    ) {
      platf::udp_payload_block_t payloads {
        reinterpret_cast<char *>(payload),
        nullptr,
        plan.data_shards,
        0,
        blocksize,
      };

      if (plan.parity_shards == 0) {
        return {payloads, 0};
      }

      // The context validates the total Reed-Solomon shard count before using the fixed table.
      // Data-only blocks can exceed that count and never use the table.
      auto *rs = rs_cache.acquire(plan.data_shards, plan.parity_shards);
      if (!rs) {
        BOOST_LOG(error) << "Failed to allocate video Reed-Solomon context; sending this FEC block without parity"sv;
        return {payloads, 0};
      }

      const auto parity_bytes = plan.parity_shards * blocksize;
      if (workspace.parity.size() < parity_bytes) {
        workspace.parity.resize(parity_bytes);
      }
      payloads.parity = reinterpret_cast<char *>(workspace.parity.data());
      payloads.parity_count = plan.parity_shards;
      for (std::size_t index = 0; index < payloads.size(); ++index) {
        workspace.rs_pointers[index] = reinterpret_cast<std::uint8_t *>(payloads.payload_at(index));
      }

      if (reed_solomon_encode(rs, workspace.rs_pointers.data(), payloads.size(), blocksize) != 0) {
        rs_cache.discard(plan.data_shards, plan.parity_shards);
        BOOST_LOG(error) << "Failed to encode video Reed-Solomon shards; sending this FEC block without parity"sv;
        payloads.parity_count = 0;
        return {payloads, 0};
      }

      return {payloads, plan.percentage};
    }
  }  // namespace fec

  // Pack one block in wire order. FEC and the synchronous sends share these same data shards.
  static std::uint8_t *packetize_video_block(
    video_shard_storage_t &storage,
    std::size_t blocksize,
    std::size_t data_shards,
    std::string_view &payload,
    video_packet_raw_t header,
    std::uint32_t block_lowseq,
    const video_short_frame_header_t *frame_header
  ) {
    const auto size = data_shards * blocksize;
    if (storage.size() < size) {
      storage.resize(size);
    }

    auto *const first_shard = storage.data();
    auto *shard = first_shard;
    auto prefix_bytes = sizeof(header);
    if (frame_header) {
      std::memcpy(shard + prefix_bytes, frame_header, sizeof(*frame_header));
      prefix_bytes += sizeof(*frame_header);
    }
    header.packet.flags = FLAG_CONTAINS_PIC_DATA;
    for (std::size_t index = 0; index < data_shards; ++index, shard += blocksize) {
      std::memcpy(shard, &header, sizeof(header));
      reinterpret_cast<video_packet_raw_t *>(shard)->packet.streamPacketIndex = block_lowseq++ << 8;
      auto *picture = shard + prefix_bytes;
      const auto capacity = blocksize - prefix_bytes;
      const auto bytes = std::min(capacity, payload.size());
      if (bytes != 0) {
        std::memcpy(picture, payload.data(), bytes);
        payload.remove_prefix(bytes);
      }
      if (bytes < capacity) {
        std::memset(picture + bytes, 0, capacity - bytes);
      }
      prefix_bytes = sizeof(header);
    }
    reinterpret_cast<video_packet_raw_t *>(first_shard)->packet.flags |= FLAG_SOF;
    reinterpret_cast<video_packet_raw_t *>(shard - blocksize)->packet.flags |= FLAG_EOF;
    return first_shard;
  }

  /**
   * @brief Pass gamepad feedback data back to the client.
   * @param session The session object.
   * @param msg The message to pass.
   * @return 0 on success.
   */
  int send_feedback_msg(session_t *session, platf::gamepad_feedback_msg_t &msg) {
    if (!session->control.peer) {
      BOOST_LOG(warning) << "Couldn't send gamepad feedback data, still waiting for PING from Artemis"sv;
      // Still waiting for PING from Artemis
      return -1;
    }

    auto &server = session->broadcast_ref->control_server;
    int send_result = -1;
    if (msg.type == platf::gamepad_feedback_e::rumble) {
      control_rumble_t plaintext;
      plaintext.header.type = control_packet::rumble;
      plaintext.header.payloadLength = sizeof(plaintext) - sizeof(control_header_v2);

      auto &data = msg.data.rumble;

      plaintext.useless = 0xC0FFEE;
      plaintext.id = util::endian::little(msg.id);
      plaintext.lowfreq = util::endian::little(data.lowfreq);
      plaintext.highfreq = util::endian::little(data.highfreq);

      send_result = send_control_packet(session, server, plaintext);
    } else if (msg.type == platf::gamepad_feedback_e::set_motion_event_state) {
      control_set_motion_event_t plaintext;
      plaintext.header.type = control_packet::set_motion_event;
      plaintext.header.payloadLength = sizeof(plaintext) - sizeof(control_header_v2);

      auto &data = msg.data.motion_event_state;

      plaintext.id = util::endian::little(msg.id);
      plaintext.reportrate = util::endian::little(data.report_rate);
      plaintext.type = data.motion_type;

      send_result = send_control_packet(session, server, plaintext);
    } else if (msg.type == platf::gamepad_feedback_e::set_rgb_led) {
      control_set_rgb_led_t plaintext;
      plaintext.header.type = control_packet::set_rgb_led_feedback;
      plaintext.header.payloadLength = sizeof(plaintext) - sizeof(control_header_v2);

      auto &data = msg.data.rgb_led;

      plaintext.id = util::endian::little(msg.id);
      plaintext.r = data.r;
      plaintext.g = data.g;
      plaintext.b = data.b;

      send_result = send_control_packet(session, server, plaintext);
    } else if (msg.type == platf::gamepad_feedback_e::set_adaptive_triggers) {
      control_adaptive_triggers_t plaintext;
      plaintext.header.type = control_packet::set_adaptive_triggers;
      plaintext.header.payloadLength = sizeof(plaintext) - sizeof(control_header_v2);

      plaintext.id = util::endian::little(msg.id);
      plaintext.event_flags = msg.data.adaptive_triggers.event_flags;
      plaintext.type_left = msg.data.adaptive_triggers.type_left;
      std::ranges::copy(msg.data.adaptive_triggers.left, plaintext.left);
      plaintext.type_right = msg.data.adaptive_triggers.type_right;
      std::ranges::copy(msg.data.adaptive_triggers.right, plaintext.right);

      send_result = send_control_packet(session, server, plaintext);
    } else if (msg.type == platf::gamepad_feedback_e::ds5_haptics_pcm) {
      if (!session->config.client_supports_authored_pcm) {
        return 0;
      }
      const auto &data = msg.data.ds5_haptics;
      if (data.frame_count > client_features::pcm_max_frames) {
        return -1;
      }
      const auto packet = client_features::encode_pcm({
        msg.id,
        data.flags,
        data.frame_count,
        data.sequence,
        data.presentation_time_us,
        std::span {data.pcm.data(), static_cast<std::size_t>(data.frame_count) * 4},
      });
      if (!packet) {
        return -1;
      }
      std::array<std::uint8_t, client_features::pcm_max_control_size + sizeof(control_encrypted_t) + crypto::cipher::tag_size + 16> encrypted_payload;
      const auto payload = encode_control(session, std::string_view {reinterpret_cast<const char *>(packet->bytes.data()), packet->size}, encrypted_payload);
      if (payload.empty()) {
        return -1;
      }
      // Old waveforms must not wait behind retransmissions. Sequence/PTS allow
      // the client to detect loss and schedule the remaining samples.
      send_result = server.send_unreliable_sequenced(payload, session->control.peer);
    } else {
      BOOST_LOG(error) << "Unknown gamepad feedback message type"sv;
      return -1;
    }

    if (send_result != 0) {
      TUPLE_2D(port, addr, platf::from_sockaddr_ex((sockaddr *) &session->control.peer->address.address));
      BOOST_LOG(warning) << "Couldn't send gamepad feedback to ["sv << addr << ':' << port << ']';

      return -1;
    }

    return 0;
  }

  int send_hdr_mode(session_t *session, video::hdr_info_t hdr_info) {
    if (!session->control.peer) {
      BOOST_LOG(warning) << "Couldn't send HDR mode, still waiting for PING from Artemis"sv;
      // Still waiting for PING from Artemis
      return -1;
    }

    control_hdr_mode_t plaintext {};
    plaintext.header.type = control_packet::hdr_mode;
    plaintext.header.payloadLength = sizeof(control_hdr_mode_t) - sizeof(control_header_v2);

    plaintext.enabled = hdr_info->enabled;
    plaintext.metadata = hdr_info->metadata;

    if (send_control_packet(session, session->broadcast_ref->control_server, plaintext) != 0) {
      TUPLE_2D(port, addr, platf::from_sockaddr_ex((sockaddr *) &session->control.peer->address.address));
      BOOST_LOG(warning) << "Couldn't send HDR mode to ["sv << addr << ':' << port << ']';

      return -1;
    }

    BOOST_LOG(debug) << "Sent HDR mode: " << hdr_info->enabled;
    return 0;
  }

  int send_depth_status(session_t *session, int phase) {
    if (!session->control.peer) {
      // Still waiting for PING from Artemis; the periodic poll will retry on the next change.
      return -1;
    }

    control_depth_status_t plaintext {};
    plaintext.header.type = control_packet::depth_status;
    plaintext.header.payloadLength = sizeof(control_depth_status_t) - sizeof(control_header_v2);
    plaintext.phase = (std::uint8_t) phase;

    if (send_control_packet(session, session->broadcast_ref->control_server, plaintext) != 0) {
      TUPLE_2D(port, addr, platf::from_sockaddr_ex((sockaddr *) &session->control.peer->address.address));
      BOOST_LOG(warning) << "Couldn't send depth status to ["sv << addr << ':' << port << ']';
      return -1;
    }

    BOOST_LOG(debug) << "Sent depth status: phase="sv << phase;
    return 0;
  }

  /**
   * @brief Answer one live video-mode request (0x3008).
   *
   * Runs only on the control thread, which owns outgoing control encryption and every ENet call,
   * and which is also the thread that clears session_t::control.peer. Callers that decided the
   * outcome elsewhere queue a plain value through mail::live_video_mode_ack instead of touching
   * the peer or the cipher themselves.
   */
  int send_live_video_mode_ack(session_t *session, const live_video_mode_ack_t &ack) {
    if (!session->control.peer) {
      // Still waiting for PING from Artemis, or the peer is already gone. There is nothing to
      // answer; the client's own request timeout covers the missing reply.
      return -1;
    }

    control_live_video_mode_ack_v2_t plaintext {};
    plaintext.header.type = control_packet::live_video_mode_ack;
    plaintext.header.payloadLength =
      sizeof(control_live_video_mode_ack_v2_t) - sizeof(control_header_v2);
    if (!encode_atomic_presentation_ack_payload(ack, plaintext.body)) {
      BOOST_LOG(error) << "Refusing to send an atomic-presentation acknowledgement that does not fit the v2 wire format"sv;
      return -1;
    }
    const int send_result = send_control_packet(
      session,
      session->broadcast_ref->control_server,
      plaintext
    );

    if (send_result != 0) {
      TUPLE_2D(port, addr, platf::from_sockaddr_ex((sockaddr *) &session->control.peer->address.address));
      BOOST_LOG(warning) << "Couldn't send live video-mode acknowledgement to ["sv << addr << ':' << port << ']';
      return -1;
    }

    BOOST_LOG(debug) << "Sent atomic-presentation v2 ack: request "sv << ack.request_id
                     << " status="sv << static_cast<int>(ack.status) << ", encoded "sv
                     << ack.applied_encoded_width << 'x' << ack.applied_encoded_height
                     << '@' << (ack.applied_framerate_x100 / 100.0) << "Hz "sv
                     << ack.applied_bitrate_kbps << "kbps"sv;
    return 0;
  }

  int send_stream_gamma_ack(session_t &session, const video::stream_gamma_ack_t &ack) {
    if (!session.control.peer) {
      return -1;
    }
    control_stream_gamma_ack_t plaintext {};
    plaintext.header.type = control_packet::stream_gamma_ack;
    plaintext.header.payloadLength = STREAM_GAMMA_ACK_PAYLOAD_SIZE;
    if (!encode_stream_gamma_ack_payload(ack, plaintext.body)) {
      BOOST_LOG(error) << "Refusing unproven stream gamma acknowledgement"sv;
      session::stop(session);
      return -1;
    }
    return send_control_packet(&session, session.broadcast_ref->control_server, plaintext);
  }

  void send_game_source_status(session_t &session) {
    auto &event = session.control.game_source_status_event;
    if (!event->peek()) {
      return;
    }
    const auto state = event->pop(0ms);
    if (!state || !session.control.peer || !session.config.client_supports_game_provider_v1 || !session.config.client_supports_atomic_presentation_v2 || !session.config.monitor.effective_mode || !game_source_status_matches_mode(*state, session.config.monitor.effective_mode->current())) {
      return;
    }
    control_game_source_status_t plaintext {};
    plaintext.header.type = control_packet::game_source_status;
    plaintext.header.payloadLength = GAME_SOURCE_STATUS_PAYLOAD_SIZE;
    if (encode_game_source_status_payload(*state, plaintext.body)) {
      // Replaceable observations are retried by the provider heartbeat. Never emit an ACK here.
      send_control_packet(&session, session.broadcast_ref->control_server, plaintext);
    }
  }

  video::sbs_telemetry_snapshot_t unavailable_sbs_telemetry_snapshot() noexcept {
    video::sbs_telemetry_snapshot_t snapshot;
    snapshot.status = video::sbs_telemetry_sample_status_e::unavailable;
    return snapshot;
  }

  void drain_sbs_telemetry(session_t &session) {
    auto &event = session.control.sbs_telemetry_event;
    while (event->peek()) {
      if (auto snapshot = event->pop(0ms)) {
        session.control.latest_sbs_telemetry = std::move(*snapshot);
      }
    }
  }

  int send_sbs_telemetry_state(
    session_t *session,
    const video::sbs_telemetry_snapshot_t &snapshot,
    sbs_telemetry_status_e status,
    std::uint16_t request_id,
    bool reliable
  ) {
    if (!session->control.peer) {
      return -1;
    }

    control_sbs_telemetry_state_t plaintext {};
    plaintext.header.type = control_packet::sbs_telemetry_state;
    plaintext.header.payloadLength =
      sizeof(control_sbs_telemetry_state_t) - sizeof(control_header_v2);
    if (!encode_sbs_telemetry_state_payload(
          sbs_telemetry_state_t {status, request_id, snapshot},
          plaintext.body
        )) {
      BOOST_LOG(error) << "Refusing to send non-finite SBS telemetry state"sv;
      return -1;
    }

    const int result = send_control_packet(
      session,
      session->broadcast_ref->control_server,
      plaintext,
      reliable ? control_delivery_e::reliable :
                 control_delivery_e::unreliable_sequenced
    );
    if (result != 0) {
      return -1;
    }
    if (request_id != 0) {
      BOOST_LOG(debug) << "Sent SBS telemetry reply: request="sv << request_id
                       << " status="sv << static_cast<int>(status)
                       << " generation="sv << snapshot.generation
                       << " sequence="sv << snapshot.sequence;
    }
    return 0;
  }

  int answer_sbs_telemetry_request(
    session_t *session,
    std::uint16_t request_id,
    std::optional<sbs_telemetry_status_e> status_override = std::nullopt
  ) {
    drain_sbs_telemetry(*session);
    const auto snapshot = session->control.latest_sbs_telemetry.value_or(
      unavailable_sbs_telemetry_snapshot()
    );
    const auto status = status_override.value_or(
      sbs_telemetry_wire_status(snapshot.status)
    );
    return send_sbs_telemetry_state(
      session,
      snapshot,
      status,
      request_id,
      true
    );
  }

  void send_termination(control_server_t &server, session_t &session) {
    if (!session.control.peer) {
      return;
    }

    control_terminate_t plaintext {};
    plaintext.header.type = control_packet::termination;
    plaintext.header.payloadLength = sizeof(plaintext.ec);
    plaintext.ec = util::endian::big<std::uint32_t>(0x80030023);

    if (send_control_packet(&session, server, plaintext) != 0) {
      TUPLE_2D(port, addr, platf::from_sockaddr_ex((sockaddr *) &session.control.peer->address.address));
      BOOST_LOG(warning) << "Couldn't send termination code to ["sv << addr << ':' << port << ']';
    }
  }

  void controlBroadcastThread(control_server_t *server) {
    server->map(control_packet::periodic_ping, [](session_t *, const std::string_view &) {
    });

    server->map(control_packet::set_stream_gamma, [](session_t *session, const std::string_view &payload) {
      if (!session->config.monitor.stream_gamma_supported) {
        return;
      }
      video::stream_gamma_request_t request;
      const auto decoded = decode_stream_gamma_request_payload(payload, request);
      // No trustworthy correlation is available for a runt, unknown mode or reserved zero ID.
      if (request.request_id == 0) {
        BOOST_LOG(warning) << "Dropping malformed stream gamma request"sv;
        return;
      }
      if (session->control.outstanding_stream_gamma_requests >= STREAM_GAMMA_QUEUE_LIMIT) {
        BOOST_LOG(error) << "Stream gamma request queue exceeded its session bound"sv;
        session::stop(*session);
        return;
      }
      ++session->control.outstanding_stream_gamma_requests;
      if (decoded != stream_gamma_request_decode_e::ok) {
        session->control.stream_gamma_ack_queue->raise(video::stream_gamma_ack_t {
          decoded == stream_gamma_request_decode_e::unsupported_version ?
            video::stream_gamma_status_e::rejected_unsupported :
            video::stream_gamma_status_e::rejected_invalid,
          request.mode,
          request.request_id,
          session->config.monitor.stream_gamma_state->current(),
        });
      } else {
        session->control.stream_gamma_queue->raise(request);
        session->video->wake_encoder();
      }
    });

    // Moonlight sends these per-frame/per-FEC-block reception reports. Apollo does not currently
    // feed them into adaptive FEC, but they are known protocol messages rather than errors.
    // Validate and consume it without a per-frame log burst; malformed reports remain visible.
    server->map(control_packet::frame_fec_status, [](session_t *, const std::string_view &payload) {
      if (!is_valid_frame_fec_status_payload_size(payload.size())) {
        BOOST_LOG(warning) << "Dropping malformed frame FEC status: expected "sv
                           << FRAME_FEC_STATUS_PAYLOAD_SIZE << " payload bytes, got "sv
                           << payload.size();
      }
    });

    server->map(control_packet::start, [](session_t *, const std::string_view &) {
      BOOST_LOG(debug) << "Received control-stream start"sv;
    });

    server->map(control_packet::request_idr, [](session_t *session, const std::string_view &) {
      session->video->idr_events->raise(true);
      session->video->wake_encoder();
    });

    server->map(control_packet::invalidate_ref_frames, 2 * sizeof(std::int64_t), [&](session_t *session, const std::string_view &payload) {
      std::int64_t frames[2];
      std::memcpy(frames, payload.data(), sizeof(frames));
      auto firstFrame = util::endian::little(frames[0]);
      auto lastFrame = util::endian::little(frames[1]);

      session->video->invalidate_ref_frames_events->raise(std::make_pair(firstFrame, lastFrame));
      session->video->wake_encoder();
    });

    server->map(control_packet::sbs_telemetry_subscription, [](session_t *session, const std::string_view &payload) {
      if (!session->config.client_supports_sbs_telemetry) {
        BOOST_LOG(warning) << "Ignoring SBS telemetry subscription from a client that did not "
                              "advertise ML_FF_SBS_TELEMETRY"sv;
        return;
      }
      sbs_telemetry_subscription_request_t request;
      const auto decoded = decode_sbs_telemetry_subscription_payload(payload, request);
      if (decoded == sbs_telemetry_subscription_decode_e::unsupported_version) {
        BOOST_LOG(warning) << "SBS telemetry subscription uses unsupported version "sv
                           << static_cast<unsigned>(
                                static_cast<std::uint8_t>(payload.front())
                              );
        if (request.request_id != 0) {
          answer_sbs_telemetry_request(
            session,
            request.request_id,
            sbs_telemetry_status_e::unsupported_version
          );
        }
        return;
      }
      if (decoded == sbs_telemetry_subscription_decode_e::invalid) {
        BOOST_LOG(warning) << "Dropping malformed SBS telemetry subscription with "sv
                           << payload.size() << " bytes"sv;
        // Exact bodies expose their request id through the decoder. For a wrong-size body, the
        // id is still recoverable when offsets 2..3 exist; otherwise there is no safe correlation
        // token with which to answer status 3.
        std::uint16_t request_id = request.request_id;
        if (payload.size() != SBS_TELEMETRY_SUBSCRIPTION_PAYLOAD_SIZE && payload.size() >= 4) {
          const auto *bytes =
            reinterpret_cast<const std::uint8_t *>(payload.data());
          request_id =
            static_cast<std::uint16_t>(bytes[2]) |
            static_cast<std::uint16_t>(bytes[3] << 8);
        }
        if (request_id != 0) {
          answer_sbs_telemetry_request(
            session,
            request_id,
            sbs_telemetry_status_e::failed
          );
        }
        return;
      }

      const auto interval_ms = effective_sbs_telemetry_interval_ms(request);
      session->control.sbs_telemetry_subscription->update(
        request.enabled(),
        request.focused(),
        interval_ms
      );
      // The release-store in update() disables render-thread copies before this handler returns.
      // Reset periodic scheduling so a new subscription never inherits the old cadence.
      session->control.last_sbs_telemetry_periodic =
        request.enabled() && request.request_id == 0 ?
          std::chrono::steady_clock::time_point {} :
          std::chrono::steady_clock::now();
      BOOST_LOG(debug) << "SBS telemetry subscription: enabled="sv << request.enabled()
                       << " focused="sv << request.focused()
                       << " interval="sv << interval_ms
                       << "ms request="sv << request.request_id;
      if (request.request_id != 0) {
        answer_sbs_telemetry_request(session, request.request_id);
      }
    });

    server->map(control_packet::set_video_mode, [](session_t *session, const std::string_view &payload) {
      // The current host accepts only the exact 20-byte v2 transaction, which binds OFF/AI to
      // source geometry, cadence, and wire bitrate. Clients must advertise the matching feature
      // bit before using this proprietary control.
      live_video_mode_wire_request_t request;
      const auto decoded = decode_live_video_mode_request_payload(payload, request);
      if (decoded == live_video_mode_request_decode_e::invalid) {
        BOOST_LOG(warning) << "Dropping IDX_SET_VIDEO_MODE body with invalid exact size "sv
                           << payload.size() << " (expected 20-byte atomic-presentation v2)"sv;
        return;
      }
      if (decoded == live_video_mode_request_decode_e::unsupported_version) {
        BOOST_LOG(warning) << "Dropping unsupported atomic-presentation version "sv
                           << static_cast<unsigned>(request.protocol_version);
        return;
      }
      if (!session->config.client_supports_atomic_presentation_v2) {
        BOOST_LOG(warning) << "Ignoring atomic-presentation v2 request from a client that did not advertise ML_FF_ATOMIC_PRESENTATION_V2"sv;
        return;
      }

      const int width = request.source_width;
      const int height = request.source_height;
      const auto framerate_x100_u32 = request.framerate_x100;
      const std::int64_t requested_bitrate_kbps = request.bitrate_kbps;

      // A malformed request must never reach the encoder: capture_async cannot recover from a
      // failed non-SBS encoder creation and would tear the whole session down. Every rejection
      // still answers the client, reporting the mode that stays in effect, so it can drop the
      // request and resynchronize instead of waiting out a timeout.
      const auto reject = [&](std::string_view reason) {
        BOOST_LOG(warning) << "type [IDX_SET_VIDEO_MODE]: ignoring "sv << width << 'x' << height
                           << '@' << (framerate_x100_u32 / 100.0) << " from ["sv
                           << session::client_name(*session) << "]: "sv << reason;
        reject_live_video_mode(
          session->mail,
          session->config.monitor.effective_mode,
          request.request_id,
          live_video_mode_ack_e::rejected_invalid
        );
      };
      if (request.flags != ATOMIC_PRESENTATION_V2_REQUEST_FLAGS_KNOWN) {
        reject("v2 request flags contain unsupported bits"sv);
        return;
      }
      if (!is_valid_live_video_sbs_mode(request.desired_sbs_mode, session->config.client_supports_game_provider_v1)) {
        reject("desired mode is invalid or Game provider v1 was not negotiated"sv);
        return;
      }
      if (!is_valid_live_video_mode_dimension(width) || !is_valid_live_video_mode_dimension(height)) {
        reject("resolution must be even and within 2..16384"sv);
        return;
      }
      const auto host_sbs_rejection =
        models::host_sbs_v2_source_resolution_rejection_reason(
          static_cast<std::uint32_t>(width),
          static_cast<std::uint32_t>(height)
        );
      const int desired_sbs_mode = static_cast<int>(request.desired_sbs_mode);
      if (desired_sbs_mode == ::video::SBS_AI && !host_sbs_rejection.empty()) {
        reject(std::format("Host SBS V2 rejects this resolution: {}", host_sbs_rejection));
        return;
      }
      if (desired_sbs_mode == ::video::SBS_GAME_SBS) {
        const auto packed = ::video::host_sbs_output_dimensions(
          width,
          height,
          session->config.monitor.videoFormat,
          config::video.sbs.max_encode_width
        );
        if (!::video::external_sbs_dimensions_match(width, height, packed.width, packed.height)) {
          reject("Game 3D requires an exact full-SBS raster within the codec and configured encoder limits"sv);
          return;
        }
      }
      if (!is_valid_live_video_mode_framerate_x100(framerate_x100_u32)) {
        reject("v2 frame rate must be within 1..1000 Hz"sv);
        return;
      }
      const int framerate_x100 = static_cast<int>(framerate_x100_u32);
      // The encode loop owns the runtime capability snapshot. Ordinary and AI video can fit an
      // oversized request; final ReShade stereo was preflighted above and must remain exact.

      // Reproduce the ANNOUNCE-time budget so a live change and a fresh launch agree for the same
      // request. The Warp multiplier is deliberately not applied: it compensates for a launch-time
      // framerate negotiation, whereas this message carries the client's true rate directly.
      std::int64_t bitrate_kbps = requested_bitrate_kbps;
      if (config::video.max_bitrate > 0 && config::video.max_bitrate < bitrate_kbps) {
        bitrate_kbps = config::video.max_bitrate;
      }
      if (!rtsp_stream::detail::is_safe_encoder_bitrate(bitrate_kbps)) {
        reject("bitrate is outside the safe encoder range"sv);
        return;
      }
      const std::int64_t audio_bitrate_kbps = audio::selected_bitrate_kbps(
        session->config.audio.channels,
        session->config.audio.flags[audio::config_t::HIGH_QUALITY]
      );
      bitrate_kbps = rtsp_stream::detail::calculate_video_bitrate_budget(
        bitrate_kbps,
        config::stream.fec_percentage,
        audio_bitrate_kbps
      );

      video::video_mode_change_t change {};
      change.width = width;
      change.height = height;
      change.framerateX100 = framerate_x100;
      change.framerate = (framerate_x100 + 50) / 100;
      change.encodingFramerate = framerate_x100 * 10;
      change.bitrate = (int) bitrate_kbps;
      change.request_id = request.request_id;
      change.requested_sbs_mode = desired_sbs_mode;

      BOOST_LOG(info) << "type [IDX_SET_VIDEO_MODE]: client requested atomic-presentation v2 "sv
                      << "mode="sv << desired_sbs_mode << ' '
                      << width << 'x' << height
                      << '@' << (framerate_x100 / 100.0) << "Hz "sv << requested_bitrate_kbps
                      << "kbps (encoder budget "sv << change.bitrate << "kbps) for ["sv
                      << session::client_name(*session) << ']';

      live_video_mode_request_t mode_request {
        session->mail,
        session->video,
        session->config.monitor.effective_mode,
        change,
        session->launch_session_id,
        session::client_name(*session),
      };

      // Every request, including a same-geometry bitrate/cadence change, uses the same serial
      // worker. Otherwise a fast-path request can overtake a geometry transition and the older
      // worker can enqueue its encoder rebuild second, reversing client order.
      dispatch_live_video_mode(*session, std::move(mode_request));
    });

    server->map(control_packet::sbs_debug_dump, [](session_t *session, const std::string_view &) {
      if (!sbs_debug_dump_request_allowed(session->requested_sbs_mode->load(std::memory_order_acquire), (bool) session->video->sbs_debug_dump_pending)) {
        BOOST_LOG(warning) << "Ignoring Dump 3D request outside Host AI or Game 3D for ["sv
                           << session::client_name(*session) << ']';
        return;
      }

      // Debug: client tapped the "Dump 3D" button. Flag the next SBS convert() to dump one
      // frame's source/depth/SBS images to the configured debug dir (Apollo protocol extension).
      BOOST_LOG(info) << "type [IDX_SBS_DEBUG_DUMP]: client requested SBS debug frame dump for ["sv
                      << session::client_name(*session) << ']';
      session->video->sbs_debug_dump_pending->store(true, std::memory_order_release);
    });

    server->map(control_packet::encrypted, CONTROL_ENCRYPTED_LENGTH_FIELD_SIZE + CONTROL_ENCRYPTED_SEQUENCE_SIZE, [server](session_t *session, const std::string_view &payload) {
      std::uint16_t length;
      std::uint32_t seq;
      std::memcpy(&length, payload.data(), sizeof(length));
      std::memcpy(&seq, payload.data() + sizeof(length), sizeof(seq));
      length = util::endian::little(length);
      seq = util::endian::little(seq);

      if (!is_valid_encrypted_control_payload(payload.size(), length)) {
        BOOST_LOG(warning) << "Dropping malformed encrypted control message: declared length "sv << length
                           << " does not match "sv << payload.size() << " payload bytes"sv;
        return;
      }

      auto tagged_cipher_length = length - CONTROL_ENCRYPTED_SEQUENCE_SIZE;
      std::string_view tagged_cipher = payload.substr(
        CONTROL_ENCRYPTED_LENGTH_FIELD_SIZE + CONTROL_ENCRYPTED_SEQUENCE_SIZE,
        tagged_cipher_length
      );

      auto &cipher = session->control.cipher;
      auto &iv = session->control.incoming_iv;
      // We use the deterministic IV construction algorithm specified in NIST SP 800-38D
      // Section 8.2.1. The sequence number is our "invocation" field and the 'CC' in the
      // high bytes is the "fixed" field. Because each client provides their own unique
      // key, our values in the fixed field need only uniquely identify each independent
      // use of the client's key with AES-GCM in our code.
      //
      // The sequence number is 32 bits long which allows for 2^32 control stream messages
      // to be received from each client before the IV repeats.
      iv.resize(12);
      std::copy_n((uint8_t *) &seq, sizeof(seq), std::begin(iv));
      iv[10] = 'C';  // Client originated
      iv[11] = 'C';  // Control stream

      std::array<std::uint8_t, CONTROL_HEADER_V2_SIZE> plaintext_header;
      std::vector<uint8_t> plaintext_payload;
      if (cipher.decrypt(tagged_cipher, plaintext_header, plaintext_payload, &iv)) {
        // something went wrong :(

        BOOST_LOG(error) << "Failed to verify tag"sv;

        session::stop(*session);
        return;
      }

      std::uint16_t type;
      std::uint16_t declared_payload_size;
      std::memcpy(&type, plaintext_header.data(), sizeof(type));
      std::memcpy(
        &declared_payload_size,
        plaintext_header.data() + sizeof(type),
        sizeof(declared_payload_size)
      );
      type = util::endian::little(type);
      declared_payload_size = util::endian::little(declared_payload_size);
      if (!is_valid_decrypted_control_payload(CONTROL_HEADER_V2_SIZE + plaintext_payload.size(), declared_payload_size)) {
        BOOST_LOG(warning) << "Dropping encrypted control message with mismatched inner length: declared "sv
                           << declared_payload_size << " bytes in a "sv
                           << (CONTROL_HEADER_V2_SIZE + plaintext_payload.size())
                           << "-byte plaintext"sv;
        return;
      }

      std::string_view next_payload {
        reinterpret_cast<char *>(plaintext_payload.data()),
        plaintext_payload.size()
      };

      if (type == control_packet::encrypted) {
        BOOST_LOG(error) << "Bad packet type [IDX_ENCRYPTED] found"sv;
        session::stop(*session);
        return;
      }

      // Input data is already authenticated by the outer control-v2 envelope.
      if (type == control_packet::input) {
        input::passthrough(
          session->input,
          std::move(plaintext_payload),
          session::permissions(*session)
        );
      } else {
        server->call(type, session, next_payload, true);
      }
    });

    // This thread handles latency-sensitive control messages
    platf::adjust_thread_priority(platf::thread_priority_e::critical);

    // Check for both the full shutdown event and the shutdown event for this
    // broadcast to ensure we can inform connected clients of our graceful
    // termination when we shut down.
    auto shutdown_event = mail::man->event<bool>(mail::shutdown);
    auto broadcast_shutdown_event = mail::man->event<bool>(mail::broadcast_shutdown);
    while (!shutdown_event->peek() && !broadcast_shutdown_event->peek()) {
      auto iteration_timeout = CONTROL_OUTGOING_MAX_WAIT;
      {
        auto slot_lock = server->_session.lock();
        auto *session = server->_session->session;
        if (session && !shutdown_event->peek() && !broadcast_shutdown_event->peek()) {
          if (session->control.sbs_telemetry_subscription->enabled()) {
            iteration_timeout = std::min(
              iteration_timeout,
              session->control.sbs_telemetry_subscription->focused() ? 50ms : 100ms
            );
          }
          if (std::chrono::steady_clock::now() > session->pingTimeout) {
            const auto identity = server->_session->peer ?
                                    platf::from_sockaddr((sockaddr *) &server->_session->peer->address.address) :
                                    session->device_uuid;
            BOOST_LOG(info) << identity << ": Ping Timeout"sv;
            session::stop(*session);
          } else if (proc::proc.stream_process_exited()) {
            BOOST_LOG(info) << "Streamed application exited; stopping its session."sv;
            session::stop(*session);
          }

          if (session->state.load(std::memory_order_acquire) == session::state_e::STOPPING) {
            session->control.sbs_telemetry_subscription->update(
              false,
              false,
              SBS_TELEMETRY_INTERVAL_BACKGROUND_MS
            );
            // Outgoing control encryption and ENet calls remain on this thread. External
            // shutdown callers only publish the stop mode and wake capture workers.
            if (session->graceful_stop_requested.load(std::memory_order_relaxed)) {
              send_termination(*server, *session);
            }
            if (server->_session->peer) {
              enet_peer_disconnect_now(server->_session->peer, 0);
            }
            session->control.deferred_live_video_mode_acks.clear();
            session->control.peer = nullptr;
            server->_session->peer = nullptr;
            server->_session->session = nullptr;
            session->controlEnd.raise(true);
          } else if (!server->_session->peer) {
            // The sole session is still waiting for its ENet peer.
          } else {
            auto &feedback_queue = session->control.feedback_queue;
            while (feedback_queue->peek()) {
              auto feedback_msg = feedback_queue->pop();
              send_feedback_msg(session, *feedback_msg);
            }
            auto &haptics_queue = session->control.haptics_feedback_queue;
            for (unsigned sent = 0; sent < 32 && haptics_queue->peek(); ++sent) {
              auto feedback_msg = haptics_queue->pop();
              send_feedback_msg(session, *feedback_msg);
            }

            auto &hdr_queue = session->control.hdr_queue;
            while (server->_session->peer && hdr_queue->peek()) {
              auto hdr_info = hdr_queue->pop();
              send_hdr_mode(session, std::move(hdr_info));
            }

            // Drain depth-engine state to its latest phase. The event intentionally coalesces a
            // fast loading->ready transition so cached engines do not flash the client UI.
            auto &depth_status_event = session->control.depth_status_event;
            std::optional<int> depth_phase;
            while (depth_status_event->peek()) {
              if (auto phase = depth_status_event->pop(0ms)) {
                depth_phase = *phase;
              }
            }
            if (depth_phase && *depth_phase != session->control.last_depth_phase && send_depth_status(session, *depth_phase) == 0) {
              session->control.last_depth_phase = *depth_phase;
            }

            // Renderer samples arrive through a single-slot event: collapse any burst to the
            // newest completed GPU readback, then periodically send that immutable sample with
            // request_id=0. Request replies are sent synchronously in the 0x3009 handler above
            // and remain reliable; these periodic updates are unreliable-sequenced.
            drain_sbs_telemetry(*session);
            const auto &telemetry_subscription =
              *session->control.sbs_telemetry_subscription;
            if (telemetry_subscription.enabled()) {
              const auto now = std::chrono::steady_clock::now();
              const auto interval =
                std::chrono::milliseconds(telemetry_subscription.interval_ms());
              const bool due =
                session->control.last_sbs_telemetry_periodic.time_since_epoch().count() == 0 ||
                now - session->control.last_sbs_telemetry_periodic >= interval;
              if (due) {
                const auto snapshot =
                  session->control.latest_sbs_telemetry.value_or(
                    unavailable_sbs_telemetry_snapshot()
                  );
                send_sbs_telemetry_state(
                  session,
                  snapshot,
                  sbs_telemetry_wire_status(snapshot.status),
                  0,
                  false
                );
                // A failed ENet enqueue is still an attempt. Rate-limit it to the negotiated
                // cadence rather than retrying every control-loop wake; normal teardown owns a
                // persistently failed peer.
                session->control.last_sbs_telemetry_periodic = now;
              }
            }

            // Answer every live video-mode request (0x3007) exactly once. Unlike the depth phase
            // these must not coalesce, so they arrive on a real queue: a client that sends two
            // requests is owed two replies. The joined presentation worker only queues plain
            // outcomes, leaving the peer, the cipher and ENet to this thread.
            auto &ack_queue = session->control.live_video_mode_ack_queue;
            while (server->_session->peer && ack_queue->peek()) {
              if (auto ack = ack_queue->pop(0ms)) {
                switch (classify_live_video_mode_ack_delivery(*ack)) {
                  case live_video_mode_ack_delivery_e::send:
                    send_live_video_mode_ack(session, *ack);
                    break;
                  case live_video_mode_ack_delivery_e::defer_until_initial_proof:
                    if (session->control.deferred_live_video_mode_acks.size() < LIVE_VIDEO_MODE_ACK_QUEUE_LIMIT) {
                      session->control.deferred_live_video_mode_acks.push_back(*ack);
                    } else {
                      BOOST_LOG(error) << "Atomic-presentation pre-proof ACK queue exceeded its session bound; closing the session"sv;
                      session::stop(*session);
                    }
                    break;
                  case live_video_mode_ack_delivery_e::fail_closed:
                    BOOST_LOG(error) << "An APPLIED atomic-presentation ACK reached the control thread without a proven generation; closing the session"sv;
                    session::stop(*session);
                    break;
                }
              }
            }
            if (server->_session->peer && !session->control.deferred_live_video_mode_acks.empty() && session->config.monitor.effective_mode) {
              const auto proven = session->config.monitor.effective_mode->current();
              if (proven.generation != 0) {
                for (const auto &deferred : session->control.deferred_live_video_mode_acks) {
                  send_live_video_mode_ack(
                    session,
                    make_live_video_mode_ack(
                      static_cast<std::uint32_t>(deferred.request_id),
                      deferred.status,
                      proven
                    )
                  );
                }
                session->control.deferred_live_video_mode_acks.clear();
              }
            }
            // Stream gamma requests (0x300D) follow the same rule: each is answered exactly once.
            // An acknowledgement without encoder proof waits here for that proof.
            auto &gamma_acks = session->control.stream_gamma_ack_queue;
            while (server->_session->peer && gamma_acks->peek()) {
              if (auto ack = gamma_acks->pop(0ms)) {
                if (ack->applied.generation == 0) {
                  if (ack->status == video::stream_gamma_status_e::applied) {
                    BOOST_LOG(error) << "An applied stream gamma acknowledgement has no encoder proof"sv;
                    session::stop(*session);
                    break;
                  }
                  if (session->control.deferred_stream_gamma_acks.size() >= STREAM_GAMMA_QUEUE_LIMIT) {
                    BOOST_LOG(error) << "Stream gamma acknowledgements waiting for encoder proof exceeded their session bound; closing the session"sv;
                    session::stop(*session);
                    break;
                  }
                  session->control.deferred_stream_gamma_acks.push_back(*ack);
                } else {
                  send_stream_gamma_ack(*session, *ack);
                  if (ack->request_id != 0 && session->control.outstanding_stream_gamma_requests > 0) {
                    --session->control.outstanding_stream_gamma_requests;
                  }
                }
              }
            }
            const auto gamma_proven = session->config.monitor.stream_gamma_state->current();
            if (server->_session->peer && gamma_proven.generation != 0) {
              for (auto &ack : session->control.deferred_stream_gamma_acks) {
                ack.applied = gamma_proven;
                send_stream_gamma_ack(*session, ack);
                if (ack.request_id != 0 && session->control.outstanding_stream_gamma_requests > 0) {
                  --session->control.outstanding_stream_gamma_requests;
                }
              }
              session->control.deferred_stream_gamma_acks.clear();
            }
            // Readiness describes the encoder-proven generation, after any queued mode ACK.
            send_game_source_status(*session);
          }
        }
      }

      server->iterate(iteration_timeout);
    }

    // Detach the raw registration before its owning RTSP session can be destroyed.
    auto slot_lock = server->_session.lock();
    if (auto *session = server->_session->session) {
      session->control.sbs_telemetry_subscription->update(
        false,
        false,
        SBS_TELEMETRY_INTERVAL_BACKGROUND_MS
      );
      send_termination(*server, *session);
      session->shutdown_event->raise(true);
      session->video->wake_encoder();
      session->controlEnd.raise(true);
      session->control.deferred_live_video_mode_acks.clear();
      session->control.peer = nullptr;
      server->_session->peer = nullptr;
      server->_session->session = nullptr;
    }

    server->flush();
  }

  void recvThread(broadcast_ctx_t &ctx) {
    std::array<std::optional<av_ping_route_t>, 2> routes;

    auto &video_sock = ctx.video_sock;
    auto &audio_sock = ctx.audio_sock;

    auto &route_updates = ctx.av_ping_route_updates;
    auto broadcast_shutdown_event = mail::man->event<bool>(mail::broadcast_shutdown);

    auto &io = ctx.io_context;

    std::array<udp::endpoint, 2> peers;
    std::array<std::array<char, 2048>, 2> buffers;
    std::array<std::function<void(const boost::system::error_code, size_t)>, 2> recv_func;

    auto route_index = [](socket_e type) {
      return type == socket_e::video ? 0U : 1U;
    };

    auto apply_route_updates = [&]() {
      while (route_updates->peek()) {
        auto update = route_updates->pop();
        if (!update) {
          break;
        }

        auto &route = routes[route_index(update->socket_type)];
        if (update->route) {
          route = std::move(update->route);
        } else if (route && route->id == update->id) {
          // A stopped session may publish removal after its successor has registered. Remove only
          // the generation that issued this update, never the newer route occupying the slot.
          route.reset();
        }
      }
    };

    auto recv_func_init = [&](udp::socket &sock, std::size_t index) {
      recv_func[index] = [&, index](const boost::system::error_code &ec, size_t bytes) {
        auto fg = util::fail_guard([&]() {
          sock.async_receive_from(asio::buffer(buffers[index]), peers[index], 0, recv_func[index]);
        });

        apply_route_updates();

        // No data, yet no error
        if (ec == boost::system::errc::connection_refused || ec == boost::system::errc::connection_reset) {
          return;
        }

        if (ec || !bytes) {
          BOOST_LOG(error) << "Couldn't receive data from udp socket: "sv << ec.message();
          return;
        }

        auto &route = routes[index];
        if (!route) {
          return;
        }

        if (bytes == sizeof(SS_PING)) {
          auto ping = reinterpret_cast<PSS_PING>(buffers[index].data());
          if (route->payload == std::string_view {ping->payload, sizeof(ping->payload)}) {
            route->messages->raise(
              peers[index],
              std::string {buffers[index].data(), bytes}
            );
          }
        }
      };
    };

    recv_func_init(video_sock, 0);
    recv_func_init(audio_sock, 1);

    video_sock.async_receive_from(asio::buffer(buffers[0]), peers[0], 0, recv_func[0]);
    audio_sock.async_receive_from(asio::buffer(buffers[1]), peers[1], 0, recv_func[1]);

    while (!broadcast_shutdown_event->peek()) {
      io.run();
    }
  }

#ifdef SUNSHINE_TESTS
  struct media_sender_test_hooks_t {
    std::function<void()> ready;
    std::function<void()> before_admission;
    std::function<void()> processed;
    std::optional<int> video_fec_percentage;
    std::optional<std::uint64_t> video_fail_encryption_at_iv;
  };
#endif

  void run_video_sender(
    udp::socket &sock,
    video::packet_queue_t packets,
    std::shared_ptr<video_channel_t> channel,
    safe::mail_raw_t::event_t<bool> shutdown_event,
    std::chrono::steady_clock::time_point video_epoch
#ifdef SUNSHINE_TESTS
    ,
    media_sender_test_hooks_t *test_hooks = nullptr
#endif
  ) {
    // Video traffic is sent on this thread
    platf::adjust_thread_priority(platf::thread_priority_e::high);

    logging::min_max_avg_periodic_logger<double> frame_processing_latency_logger(info, "Frame processing latency", "ms");

    logging::time_delta_periodic_logger frame_send_batch_latency_logger(info, "Network: each send_batch() latency");
    logging::time_delta_periodic_logger frame_fec_latency_logger(info, "Network: each FEC block latency");
    logging::time_delta_periodic_logger frame_network_latency_logger(info, "Network: frame's overall network latency");

    struct video_stage_diagnostics_t {
      logging::min_max_avg_periodic_logger<double> encoded_queue {
        info,
        "Video packet: encoded queue residence",
        "ms",
      };
      logging::min_max_avg_periodic_logger<double> new_content_age {
        info,
        "Video output: new content age before packetization",
        "ms",
      };
      logging::min_max_avg_periodic_logger<double> repeated_content_age {
        info,
        "Video output: repeated content age before packetization",
        "ms",
      };
      logging::min_max_avg_periodic_logger<double> packetization {
        info,
        "Video packet: packetization and pacing-plan CPU",
        "ms",
      };
    };

    // Logger construction itself reads the clock and owns strings. Keep even that work outside
    // the disabled path, along with all new samples and packet classification.
    auto stage_diagnostics = video::detail::make_diagnostic_state<video_stage_diagnostics_t>(
      config::sunshine.diagnostics_enabled
    );

    crypto::aes_t iv(12);
    iv[11] = 'V';  // Video stream
    std::array<video_packet_enc_prefix_t, platf::MAX_UDP_BATCH_PACKETS> prefixes {};
    video_shard_storage_t data_shard_storage;
    fec::rs_cache_t video_rs_cache;
    fec::workspace_t video_fec_workspace;

    auto timer = platf::create_high_precision_timer();
    if (!timer || !*timer) {
      BOOST_LOG(error) << "Failed to create timer, aborting video broadcast thread";
      return;
    }

    auto ratecontrol_next_frame_start = std::chrono::steady_clock::now();

#ifdef SUNSHINE_TESTS
    if (test_hooks && test_hooks->ready) {
      test_hooks->ready();
    }
#endif
    while (auto packet = packets->pop()) {
#ifdef SUNSHINE_TESTS
      auto report_processed = util::fail_guard([test_hooks] {
        if (test_hooks && test_hooks->processed) {
          test_hooks->processed();
        }
      });
      if (test_hooks && test_hooks->before_admission) {
        test_hooks->before_admission();
      }
#endif
      auto release_oversized_workspaces = util::fail_guard([&]() {
        if (data_shard_storage.capacity() > VIDEO_SEND_WORKSPACE_RETAIN_LIMIT) {
          video_shard_storage_t().swap(data_shard_storage);
        }
        video_fec_workspace.release_oversized();
      });
      if (shutdown_event->peek()) {
        break;
      }

      frame_network_latency_logger.first_point_now();

      if (!channel || !channel->active.load(std::memory_order_acquire)) {
        continue;
      }
      const auto effective_pacing = channel->effective_mode ?
                                      channel->effective_mode->pacing() :
                                      video::effective_video_pacing_t {0, 0};

      auto request_recovery_idr = [&](std::string_view reason) {
        if (!channel->awaiting_recovery_idr) {
          BOOST_LOG(warning) << "Dropping encoded video until a recovery IDR ("sv << reason << ")."sv;
        }
        channel->awaiting_recovery_idr = true;
        channel->idr_events->try_raise(true);
        channel->wake_encoder();
      };

      const auto packet_dequeued_at = std::chrono::steady_clock::now();
      const auto encoded_packet_age = packet_dequeued_at - packet->encoded_timestamp;
      if (stage_diagnostics) {
        if (const auto elapsed = video::detail::diagnostic_elapsed_ms(packet->encoded_timestamp, packet_dequeued_at)) {
          stage_diagnostics->encoded_queue.collect_and_log(*elapsed);
        }
      }
      const auto max_encoded_packet_age = std::chrono::nanoseconds {
        video_packet_max_queue_age_ns(effective_pacing.framerate_millihz)
      };
      if (encoded_packet_age > max_encoded_packet_age) {
        request_recovery_idr("encoded packet exceeded the host backlog age limit"sv);
        continue;
      }
      if (channel->awaiting_recovery_idr) {
        if (!packet->is_idr()) {
          continue;
        }
        channel->awaiting_recovery_idr = false;
      }

      std::string_view payload {reinterpret_cast<const char *>(packet->data()), packet->data_size()};

      video_short_frame_header_t frame_header = {};
      frame_header.headerType = 0x01;  // Short header type
      frame_header.frameType = packet->is_idr()                     ? 2 :
                               packet->after_ref_frame_invalidation ? 5 :
                                                                      1;
      frame_header.source_frame_id = channel->source_frame_id.next_frame(
        channel->source_frame_id_negotiated,
        static_cast<std::uint32_t>(packet->frame_index()),
        packet->encoder_input_retained,
        packet->is_idr(),
        packet->after_ref_frame_invalidation
      );
      if (packet->processing_started || stage_diagnostics) {
        const auto measured_at = std::chrono::steady_clock::now();
        const auto latency = video::detail::frame_processing_latency_tenths_ms(
          packet->processing_started,
          measured_at
        );
        frame_header.frame_processing_latency = latency;
        if (latency != 0) {
          frame_processing_latency_logger.collect_and_log(latency / 10.);
        }
        if (stage_diagnostics && packet->diagnostic_content != video::detail::diagnostic_content_e::unknown) {
          const auto content_timestamp = video::detail::select_content_timestamp(
            packet->content_timestamp,
            packet->frame_timestamp
          );
          if (const auto elapsed = video::detail::diagnostic_elapsed_ms(content_timestamp, measured_at)) {
            auto &logger = packet->diagnostic_content == video::detail::diagnostic_content_e::new_content ?
                             stage_diagnostics->new_content_age :
                             stage_diagnostics->repeated_content_age;
            logger.collect_and_log(*elapsed);
            if (packet->sbs_telemetry_performance && packet->diagnostic_content == video::detail::diagnostic_content_e::new_content) {
              packet->sbs_telemetry_performance->record(host_sbs_telemetry::stage::new_content_age, *elapsed, measured_at);
            }
          }
        }
      } else {
        frame_header.frame_processing_latency = 0;
      }

      auto fec_percentage = config::stream.fec_percentage;
#ifdef SUNSHINE_TESTS
      if (test_hooks && test_hooks->video_fec_percentage) {
        fec_percentage = *test_hooks->video_fec_percentage;
      }
#endif

      const auto packetization_started = video::detail::diagnostic_timestamp(
        stage_diagnostics.has_value(),
        video::detail::diagnostic_clock_t::now
      );
      if (!is_valid_video_transport_config(channel->packet_size, channel->min_required_fec_packets) || fec_percentage < 0 || fec_percentage > 255 || payload.size() > std::numeric_limits<std::size_t>::max() - sizeof(frame_header)) {
        request_recovery_idr("invalid video packetization parameters"sv);
        continue;
      }

      const std::size_t blocksize = channel->packet_size + MAX_RTP_HEADER_SIZE;
      const auto payload_blocksize = blocksize - sizeof(video_packet_raw_t);
      const auto picture_size = payload.size() + sizeof(frame_header);
      const auto frame_plan = fec::make_frame_plan(
        picture_size,
        payload_blocksize,
        fec_percentage,
        channel->min_required_fec_packets
      );
      if (!frame_plan) {
        request_recovery_idr("encoded frame exceeded the FEC packet-index limit"sv);
        continue;
      }

      frame_header.lastPayloadLen = frame_plan->final_payload_bytes;
      const auto fec_blocks_needed = frame_plan->block_count;

      try {
        const auto wire_packet_bytes = blocksize + sizeof(video_packet_enc_prefix_t);
        const auto pacing_plan = make_video_pacing_plan(
          effective_pacing.bitrate,
          effective_pacing.framerate_millihz,
          frame_plan->data_shards,
          frame_plan->wire_shards,
          payload_blocksize,
          wire_packet_bytes
        );

        // Send less than 64K in a single batch.
        // On Windows, batches above 64K seem to bypass SO_SNDBUF regardless of its size,
        // appear in "Other I/O" and begin waiting for interrupts.
        // This gives inconsistent performance so we'd rather avoid it. Also cap each batch to
        // roughly one pacing quantum so a nominally paced frame is not emitted as a few 64K bursts.
        size_t send_batch_size = std::max<std::size_t>(1, 64 * 1024 / wire_packet_bytes);
        // Also don't exceed 64 packets, which can happen when Artemis requests
        // unusually small packet size.
        // Generic Segmentation Offload on Linux can't do more than 64.
        send_batch_size = std::min({
          platf::MAX_UDP_BATCH_PACKETS,
          send_batch_size,
          pacing_plan.packets_per_quantum,
        });
        auto packetization_time = video::detail::diagnostic_clock_t::duration::zero();
        if (packetization_started) {
          packetization_time = video::detail::diagnostic_clock_t::now() - *packetization_started;
        }

        const auto ceiling_label =
          pacing_plan.target_wire_bps == VIDEO_PACING_MAX_WIRE_BPS ? ", ceiling-limited"sv : ""sv;
        if (packet->is_idr()) {
          if (!channel->idr_pacing_plan_logged.exchange(true, std::memory_order_acq_rel)) {
            BOOST_LOG(debug) << "Video IDR packet pacing (transient): target="sv
                             << (pacing_plan.target_wire_bps / 1'000'000.0)
                             << " Mbps"sv << ceiling_label
                             << "; steady-state pacing will be logged on the first inter frame."sv;
          }
        } else if (!channel->pacing_plan_logged.exchange(true, std::memory_order_acq_rel)) {
          BOOST_LOG(info) << "Video packet pacing (steady state): target="sv
                          << (pacing_plan.target_wire_bps / 1'000'000.0)
                          << " Mbps"sv << ceiling_label
                          << " (negotiated bitrate + actual packet overhead/cadence), batch<="sv
                          << send_batch_size
                          << " packets, frame-span<="sv
                          << (pacing_plan.max_frame_span_ns / 1'000'000.0) << " ms."sv;
        }

        // A duplicate without a capture timestamp uses the same pacing timestamp for every block.
        using rtp_tick = std::chrono::duration<std::uint32_t, std::ratio<1, 90000>>;
        const auto capture_time = packet->frame_timestamp.value_or(ratecontrol_next_frame_start);
        const auto timestamp = std::chrono::round<rtp_tick>(capture_time - video_epoch).count();

        // Leave RTP and fecInfo zero until the parity shards have been generated.
        video_packet_raw_t block_header {};
        block_header.packet.frameIndex = packet->frame_index();
        block_header.packet.multiFecFlags = 0x10;
        auto ratecontrol_frame_start = ratecontrol_next_frame_start;
        auto peer_address = channel->peer.address();

        size_t ratecontrol_frame_packets_sent = 0;
        bool frame_encrypted = true;

        for (std::size_t block_index = 0; block_index < fec_blocks_needed && frame_encrypted; ++block_index) {
          const auto &block = frame_plan->blocks[block_index];
          const auto block_lowseq = channel->lowseq;
          const auto block_packing_started = video::detail::diagnostic_timestamp(
            stage_diagnostics.has_value(),
            video::detail::diagnostic_clock_t::now
          );
          block_header.packet.multiFecBlocks = (block_index << 4) | ((fec_blocks_needed - 1) << 6);
          auto *block_payload = packetize_video_block(
            data_shard_storage,
            blocksize,
            block.data_shards,
            payload,
            block_header,
            block_lowseq,
            block_index == 0 ? &frame_header : nullptr
          );
          if (block_packing_started) {
            packetization_time += video::detail::diagnostic_clock_t::now() - *block_packing_started;
          }
          if (block_index == 0) {
            // Start pacing when the first block is packed and the previous frame's slot has ended.
            ratecontrol_frame_start = std::max(ratecontrol_frame_start, std::chrono::steady_clock::now());
          }

          frame_fec_latency_logger.first_point_now();
          const auto [payloads, percentage] = fec::encode(block_payload, blocksize, block, video_rs_cache, video_fec_workspace);
          frame_fec_latency_logger.second_point_now_and_log();

          // Reserve this block's sequence range before the first send. A recovery IDR starts
          // after that range even when a failed send may have delivered part of the block.
          channel->lowseq += static_cast<std::uint32_t>(payloads.size());

          for (std::size_t first_packet = 0; first_packet < payloads.size(); first_packet += send_batch_size) {
            const auto count = std::min(send_batch_size, payloads.size() - first_packet);

            // Complete encryption of the whole batch before handing any of it to the socket.
            for (std::size_t index = 0; index < count; ++index) {
              const auto packet_index = first_packet + index;
              auto *payload = payloads.payload_at(packet_index);
              auto *inspect = reinterpret_cast<video_packet_raw_t *>(payload);

              inspect->packet.fecInfo =
                (packet_index << 12 |
                 payloads.data_count << 22 |
                 percentage << 4);
              inspect->rtp.header = 0x80 | FLAG_EXTENSION;
              inspect->rtp.sequenceNumber = util::endian::big<uint16_t>(
                static_cast<std::uint16_t>(block_lowseq + static_cast<std::uint32_t>(packet_index))
              );
              inspect->rtp.timestamp = util::endian::big<uint32_t>(timestamp);
              inspect->packet.multiFecBlocks = block_header.packet.multiFecBlocks;
              inspect->packet.frameIndex = block_header.packet.frameIndex;

              // NIST SP 800-38D, Section 8.2.1. The counter identifies each use of the client's unique key.
              const auto counter = channel->gcm_iv_counter++;
              std::memcpy(iv.data(), &counter, sizeof(counter));

              auto &prefix = prefixes[index];
              prefix.frameNumber = block_header.packet.frameIndex;
              std::copy(std::begin(iv), std::end(iv), prefix.iv);
#ifdef SUNSHINE_TESTS
              if (test_hooks && test_hooks->video_fail_encryption_at_iv == counter && channel->cipher.encrypt_ctx) {
                EVP_CIPHER_CTX_reset(channel->cipher.encrypt_ctx.get());
              }
#endif
              if (!channel->cipher.encrypt(std::string_view {payload, blocksize}, prefix.tag, reinterpret_cast<std::uint8_t *>(payload), &iv)) {
                frame_encrypted = false;
                break;
              }
            }
            if (!frame_encrypted) {
              request_recovery_idr("video datagram encryption failed"sv);
              break;
            }

            // Pace every bounded batch. The first batch is due immediately. Later batches
            // are distributed according to the negotiated bitrate, FEC, and cadence.
            const auto due = ratecontrol_frame_start +
                             video_pacing_offset(ratecontrol_frame_packets_sent, pacing_plan.packets_per_second);
            auto now = std::chrono::steady_clock::now();
            const auto schedule_rebase = std::chrono::nanoseconds {
              video_pacing_rebase_ns(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                  due - ratecontrol_frame_start
                )
                  .count(),
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                  now - ratecontrol_frame_start
                )
                  .count()
              )
            };
            ratecontrol_frame_start += schedule_rebase;
            const auto bounded_due = due + schedule_rebase;
            if (now < bounded_due) {
              timer->sleep_for(bounded_due - now);
            }

            const platf::batched_send_info_t batch {
              payloads,
              reinterpret_cast<const char *>(prefixes.data()),
              sizeof(video_packet_enc_prefix_t),
              first_packet,
              count,
              (uintptr_t) sock.native_handle(),
              peer_address,
              channel->peer.port(),
              channel->local_address,
            };
            frame_send_batch_latency_logger.first_point_now();
            if (!platf::send_batch(batch)) {
              for (std::size_t index = 0; index < batch.packet_count; ++index) {
                auto send_info = platf::send_info_t {
                  batch.header_at(index),
                  batch.header_size,
                  batch.payload_at(index),
                  batch.payloads.payload_size,
                  batch.native_socket,
                  peer_address,
                  batch.target_port,
                  channel->local_address,
                };
                platf::send(send_info);
              }
            }
            frame_send_batch_latency_logger.second_point_now_and_log();

            ratecontrol_frame_packets_sent += count;
          }
        }
        if (stage_diagnostics) {
          stage_diagnostics->packetization.collect_and_log(std::chrono::duration<double, std::milli>(packetization_time).count());
        }

        // Remember the final scheduled packet time in case the next frame is already queued.
        ratecontrol_next_frame_start = ratecontrol_frame_start +
                                       video_pacing_offset(
                                         ratecontrol_frame_packets_sent,
                                         pacing_plan.packets_per_second
                                       );

        if (frame_encrypted) {
          frame_network_latency_logger.second_point_now_and_log();
        }
      } catch (const std::exception &e) {
        BOOST_LOG(error) << "Broadcast video failed "sv << e.what();
        request_recovery_idr("video packetization or send failed"sv);
        std::this_thread::sleep_for(100ms);
      }
    }
  }

  void run_audio_sender(
    udp::socket &sock,
    audio::packet_queue_t packets,
    std::shared_ptr<audio_channel_t> channel,
    safe::mail_raw_t::event_t<bool> shutdown_event
#ifdef SUNSHINE_TESTS
    ,
    media_sender_test_hooks_t *test_hooks = nullptr
#endif
  ) {
    audio_packet_t audio_packet;
    fec::rs_t rs {reed_solomon_new(RTPA_DATA_SHARDS, RTPA_FEC_SHARDS)};
    crypto::aes_t iv(16);

    if (!rs) {
      BOOST_LOG(error) << "Failed to allocate audio Reed-Solomon context"sv;
      return;
    }

    // For unknown reasons, the RS parity matrix computed by our RS implementation
    // doesn't match the one Nvidia uses for audio data. I'm not exactly sure why,
    // but we can simply replace it with the matrix generated by OpenFEC which
    // works correctly. This is possible because the data and FEC shard count is
    // constant and known in advance.
    const unsigned char parity[] = {0x77, 0x40, 0x38, 0x0e, 0xc7, 0xa7, 0x0d, 0x6c};
    memcpy(rs.get()->p, parity, sizeof(parity));

    audio_packet.rtp.header = 0x80;
    audio_packet.rtp.packetType = 97;
    audio_packet.rtp.ssrc = 0;

    // Audio traffic is sent on this thread
    platf::adjust_thread_priority(platf::thread_priority_e::high);

#ifdef SUNSHINE_TESTS
    if (test_hooks && test_hooks->ready) {
      test_hooks->ready();
    }
#endif
    while (auto packet = packets->pop()) {
#ifdef SUNSHINE_TESTS
      auto report_processed = util::fail_guard([test_hooks] {
        if (test_hooks && test_hooks->processed) {
          test_hooks->processed();
        }
      });
      if (test_hooks && test_hooks->before_admission) {
        test_hooks->before_admission();
      }
#endif
      if (shutdown_event->peek()) {
        break;
      }

      auto &packet_data = *packet;
      if (!channel || !channel->active.load(std::memory_order_acquire)) {
        continue;
      }

      auto sequenceNumber = channel->sequence_number;
      auto timestamp = channel->timestamp;

      const auto packet_iv = util::endian::big<std::uint32_t>(channel->av_ri_key_id + sequenceNumber);
      std::memcpy(iv.data(), &packet_iv, sizeof(packet_iv));

      auto &shards_p = channel->shard_ptrs;

      const auto bytes = channel->cipher.encrypt(
        std::string_view {
          reinterpret_cast<const char *>(std::begin(packet_data)),
          packet_data.size()
        },
        shards_p[sequenceNumber % RTPA_DATA_SHARDS],
        &iv
      );
      if (bytes < 0) {
        BOOST_LOG(error) << "Couldn't encode audio packet"sv;
        break;
      }

      audio_packet.rtp.sequenceNumber = util::endian::big(sequenceNumber);
      audio_packet.rtp.timestamp = util::endian::big(timestamp);

      channel->sequence_number++;
      channel->timestamp += channel->packet_duration;

      auto peer_address = channel->peer.address();
      try {
        auto send_info = platf::send_info_t {
          (const char *) &audio_packet,
          sizeof(audio_packet),
          (const char *) shards_p[sequenceNumber % RTPA_DATA_SHARDS],
          (size_t) bytes,
          (uintptr_t) sock.native_handle(),
          peer_address,
          channel->peer.port(),
          channel->local_address,
        };
        platf::send(send_info);

        auto &fec_packet = channel->fec_packet;
        // initialize the FEC header at the beginning of the FEC block
        if (sequenceNumber % RTPA_DATA_SHARDS == 0) {
          fec_packet.fecHeader.baseSequenceNumber = util::endian::big(sequenceNumber);
          fec_packet.fecHeader.baseTimestamp = util::endian::big(timestamp);
        }

        // generate parity shards at the end of the FEC block
        if ((sequenceNumber + 1) % RTPA_DATA_SHARDS == 0) {
          if (reed_solomon_encode(rs.get(), shards_p.begin(), RTPA_TOTAL_SHARDS, bytes) != 0) {
            BOOST_LOG(error) << "Failed to encode audio Reed-Solomon shards"sv;
            break;
          }

          for (auto x = 0; x < RTPA_FEC_SHARDS; ++x) {
            fec_packet.rtp.sequenceNumber = util::endian::big<std::uint16_t>(sequenceNumber + x + 1);
            fec_packet.fecHeader.fecShardIndex = x;

            auto send_info = platf::send_info_t {
              (const char *) &fec_packet,
              sizeof(fec_packet),
              (const char *) shards_p[RTPA_DATA_SHARDS + x],
              (size_t) bytes,
              (uintptr_t) sock.native_handle(),
              peer_address,
              channel->peer.port(),
              channel->local_address,
            };
            platf::send(send_info);
          }
        }
      } catch (const std::exception &e) {
        BOOST_LOG(error) << "Broadcast audio failed "sv << e.what();
        std::this_thread::sleep_for(100ms);
      }
    }
  }

#ifdef SUNSHINE_TESTS
  #include "stream_media_test.inc"
#endif

  int start_broadcast(broadcast_ctx_t &ctx) {
    auto address_family = net::af_from_enum_string(config::sunshine.address_family);
    boost::system::error_code ec;
    const auto bind_address_string = net::get_bind_address(address_family);
    if (!bind_address_string) {
      BOOST_LOG(fatal) << "Stream servers refused invalid bind_address ["sv << config::sunshine.bind_address << ']';
      return -1;
    }
    const auto bind_address = boost::asio::ip::make_address(*bind_address_string, ec);
    if (ec) {
      BOOST_LOG(fatal) << "Invalid stream bind address ["sv << *bind_address_string << "]: "sv << ec.message();
      return -1;
    }
    auto protocol = bind_address.is_v4() ? udp::v4() : udp::v6();
    auto control_port = net::map_port(CONTROL_PORT);
    auto video_port = net::map_port(VIDEO_STREAM_PORT);
    auto audio_port = net::map_port(AUDIO_STREAM_PORT);

    if (ctx.control_server.bind(address_family, control_port)) {
      BOOST_LOG(error) << "Couldn't bind Control server to port ["sv << control_port << "], likely another process already bound to the port"sv;

      return -1;
    }

    ctx.video_sock.open(protocol, ec);
    if (ec) {
      BOOST_LOG(fatal) << "Couldn't open socket for Video server: "sv << ec.message();

      return -1;
    }

    // Set video socket send buffer size (SO_SENDBUF) to 1MB
    try {
      ctx.video_sock.set_option(boost::asio::socket_base::send_buffer_size(1024 * 1024));
    } catch (...) {
      BOOST_LOG(error) << "Failed to set video socket send buffer size (SO_SENDBUF)";
    }

    ctx.video_sock.bind(udp::endpoint(bind_address, video_port), ec);
    if (ec) {
      BOOST_LOG(fatal) << "Couldn't bind Video server to port ["sv << video_port << "]: "sv << ec.message();

      return -1;
    }

    ctx.audio_sock.open(protocol, ec);
    if (ec) {
      BOOST_LOG(fatal) << "Couldn't open socket for Audio server: "sv << ec.message();

      return -1;
    }

    ctx.audio_sock.bind(udp::endpoint(bind_address, audio_port), ec);
    if (ec) {
      BOOST_LOG(fatal) << "Couldn't bind Audio server to port ["sv << audio_port << "]: "sv << ec.message();

      return -1;
    }

    ctx.av_ping_route_updates = std::make_shared<av_ping_route_queue_t::element_type>(30);

    // Thread construction can throw after one or more workers have started. Roll back every
    // partially initialized worker before shared_t destroys and retries this broadcast object.
    auto rollback = util::fail_guard([&ctx] {
      end_broadcast(ctx);
    });

    ctx.video_epoch = std::chrono::steady_clock::now();
    ctx.control_thread = std::thread {controlBroadcastThread, &ctx.control_server};

    ctx.recv_thread = std::thread {recvThread, std::ref(ctx)};

    rollback.disable();
    return 0;
  }

  void end_broadcast(broadcast_ctx_t &ctx) {
    auto broadcast_shutdown_event = mail::man->event<bool>(mail::broadcast_shutdown);

    broadcast_shutdown_event->raise(true);

    if (ctx.av_ping_route_updates) {
      ctx.av_ping_route_updates->stop();
    }
    ctx.io_context.stop();

    boost::system::error_code ec;
    ctx.video_sock.close(ec);
    ctx.audio_sock.close(ec);

    BOOST_LOG(debug) << "Waiting for main listening thread to end..."sv;
    if (ctx.recv_thread.joinable()) {
      ctx.recv_thread.join();
    }
    BOOST_LOG(debug) << "Waiting for main control thread to end..."sv;
    if (ctx.control_thread.joinable()) {
      ctx.control_thread.join();
    }
    BOOST_LOG(debug) << "All broadcasting threads ended"sv;

    broadcast_shutdown_event->reset();
  }

  int recv_ping(session_t *session, decltype(broadcast)::ptr_t ref, socket_e type, std::string_view expected_payload, udp::endpoint &peer, std::chrono::milliseconds timeout) {
    if (expected_payload.size() != sizeof(SS_PING::payload)) {
      BOOST_LOG(error) << "Refusing invalid A/V ping identity length: "sv << expected_payload.size();
      return -1;
    }

    auto messages = std::make_shared<message_queue_t::element_type>(30);
    ref->av_ping_route_updates->raise(av_ping_route_update_t {
      type,
      session->launch_session_id,
      av_ping_route_t {
        session->launch_session_id,
        std::string {expected_payload},
        messages,
      },
    });

    auto fg = util::fail_guard([&]() {
      messages->stop();
      ref->av_ping_route_updates->raise(av_ping_route_update_t {
        type,
        session->launch_session_id,
        std::nullopt,
      });
    });

    // Nothing raises this queue when the session stops, so wait in short slices and look at
    // the session's own stop event between them: a session ended before its ping arrived
    // must not hold its teardown for the whole ping timeout.
    constexpr auto stop_poll_interval = 50ms;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      if (session->shutdown_event->peek()) {
        BOOST_LOG(info) << "Session stopped before its initial "sv
                        << (type == socket_e::video ? "video"sv : "audio"sv) << " ping."sv;
        return -1;
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        break;
      }
      const auto slice = std::min(
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now) + 1ms,
        std::chrono::milliseconds {stop_poll_interval}
      );
      auto msg_opt = messages->pop(slice);
      if (!msg_opt) {
        if (!messages->running()) {
          break;
        }
        continue;
      }

      auto &recv_peer = msg_opt->first;
      BOOST_LOG(debug) << "Received initial "sv << (type == socket_e::video ? "video"sv : "audio"sv)
                       << " SS ping from "sv << recv_peer.address() << ':' << recv_peer.port();

      // Update connection details.
      peer = recv_peer;
      return 0;
    }

    BOOST_LOG(error) << "Initial Ping Timeout"sv;
    return -1;
  }

  void videoThread(session_t *session) {
    auto fg = util::fail_guard([&]() {
      session::stop(*session);
    });

    auto ref = broadcast.ref();
    auto error = recv_ping(session, ref, socket_e::video, session->video->ping_payload, session->video->peer, config::stream.ping_timeout);
    if (error < 0) {
      return;
    }

    // Register for the UDP ping immediately so an early client ping is not lost, then wait until
    // platform resume and the control connection have both completed before emitting any frames.
    if (!session->startup_ready.view() || !session->control_ready.view() || session->state.load(std::memory_order_acquire) != session::state_e::RUNNING) {
      return;
    }

    // Enable local prioritization and QoS tagging on video traffic if requested by the client
    auto address = session->video->peer.address();
    session->video->qos = platf::enable_socket_qos(ref->video_sock.native_handle(), address, session->video->peer.port(), platf::qos_data_type_e::video, session->config.videoQosType != 0);

    BOOST_LOG(debug) << "Start capturing Video"sv;
    video::capture(session->mail, session->config.monitor, session->media_senders.video_packets());
  }

  void audioThread(session_t *session) {
    auto fg = util::fail_guard([&]() {
      session::stop(*session);
    });

    auto ref = broadcast.ref();
    auto error = recv_ping(session, ref, socket_e::audio, session->audio->ping_payload, session->audio->peer, config::stream.ping_timeout);
    if (error < 0) {
      return;
    }

    // Audio pings commonly arrive before a resumed virtual display is ready. Holding capture here
    // prevents packets from being sent with an uninitialized local address (WSAEINVAL/10022).
    if (!session->startup_ready.view() || !session->control_ready.view() || session->state.load(std::memory_order_acquire) != session::state_e::RUNNING) {
      return;
    }

    // Enable local prioritization and QoS tagging on audio traffic if requested by the client
    auto address = session->audio->peer.address();
    session->audio->qos = platf::enable_socket_qos(ref->audio_sock.native_handle(), address, session->audio->peer.port(), platf::qos_data_type_e::audio, session->config.audioQosType != 0);

    BOOST_LOG(debug) << "Start capturing Audio"sv;
    audio::capture(session->mail, session->config.audio, session->media_senders.audio_packets());
  }

  namespace session {
    namespace {
      // Construct all four per-session workers behind a shared gate. If creating any std::thread
      // throws, already-created workers are released as no-ops and joined before their thread
      // objects are destroyed. This keeps a resource-exhaustion failure from terminating the host.
      class pending_session_workers_t {
      public:
        ~pending_session_workers_t() {
          cancel();
        }

        pending_session_workers_t(const pending_session_workers_t &) = delete;
        pending_session_workers_t &operator=(const pending_session_workers_t &) = delete;

        pending_session_workers_t() = default;

        bool prepare(
          std::function<void()> audio_entry,
          std::function<void()> video_entry,
          std::function<void()> video_sender_entry,
          std::function<void()> audio_sender_entry,
          std::size_t fail_before_worker_for_test = std::numeric_limits<std::size_t>::max(),
          std::function<void()> failure_handler = {}
        ) noexcept {
          try {
            gate_ = std::make_shared<std::promise<bool>>();
            auto ready = gate_->get_future().share();
            auto start_worker = [ready, failure_handler, fail_before_worker_for_test](std::size_t index, std::string_view name, std::function<void()> entry) {
#ifdef SUNSHINE_TESTS
              if (fail_before_worker_for_test == index) {
                throw std::runtime_error {"injected session-worker start failure"};
              }
#else
              (void) index;
              (void) fail_before_worker_for_test;
#endif
              return std::thread {[ready, failure_handler, name, entry = std::move(entry)]() mutable {
                try {
                  if (!ready.get()) {
                    return;
                  }
                  entry();
                } catch (const std::future_error &) {
                  // A broken startup gate is equivalent to a cancelled session.
                  return;
                } catch (const std::exception &exception) {
                  BOOST_LOG(error) << name << " session worker failed: "sv << exception.what();
                } catch (...) {
                  BOOST_LOG(error) << name << " session worker failed with an unknown error."sv;
                }

                if (failure_handler) {
                  failure_handler();
                }
              }};
            };
            audio_ = start_worker(0, "Audio"sv, std::move(audio_entry));
            video_ = start_worker(1, "Video"sv, std::move(video_entry));
            video_sender_ = start_worker(2, "Video sender"sv, std::move(video_sender_entry));
            audio_sender_ = start_worker(3, "Audio sender"sv, std::move(audio_sender_entry));
            return true;
          } catch (const std::exception &exception) {
            if (fail_before_worker_for_test == std::numeric_limits<std::size_t>::max()) {
              BOOST_LOG(error) << "Failed to create streaming session workers: "sv << exception.what();
            }
          } catch (...) {
            if (fail_before_worker_for_test == std::numeric_limits<std::size_t>::max()) {
              BOOST_LOG(error) << "Failed to create streaming session workers with an unknown error."sv;
            }
          }

          cancel();
          return false;
        }

        void commit(session_t &session) noexcept {
          session.audioThread = std::move(audio_);
          session.videoThread = std::move(video_);
          session.media_senders.adopt_video(std::move(video_sender_));
          session.media_senders.adopt_audio(std::move(audio_sender_));
          resolve(true);
        }

        void cancel() noexcept {
          resolve(false);
          if (audio_sender_.joinable()) {
            audio_sender_.join();
          }
          if (video_sender_.joinable()) {
            video_sender_.join();
          }
          if (video_.joinable()) {
            video_.join();
          }
          if (audio_.joinable()) {
            audio_.join();
          }
        }

        [[nodiscard]] bool has_joinable_worker() const noexcept {
          return audio_.joinable() || video_.joinable() || video_sender_.joinable() || audio_sender_.joinable();
        }

      private:
        void resolve(bool run) noexcept {
          if (!gate_) {
            return;
          }

          try {
            gate_->set_value(run);
          } catch (const std::future_error &) {
            // Resolution is intentionally idempotent for cleanup paths.
          }
          gate_.reset();
        }

        std::shared_ptr<std::promise<bool>> gate_;
        std::thread audio_;
        std::thread video_;
        std::thread video_sender_;
        std::thread audio_sender_;
      };

      template<class Clock>
      void publish_control_registration(
        session_t &session,
        control_server_t::session_slot_t &slot,
        pending_session_workers_t &workers,
        std::chrono::milliseconds ping_timeout,
        Clock &&now
      ) {
        // The caller holds the control slot lock. Platform/input setup and worker construction
        // must not consume the client's connection window before the session is visible.
        session.pingTimeout = now() + ping_timeout;
        session.state.store(state_e::RUNNING, std::memory_order_relaxed);
        session.startup_ready.raise(true);
        slot.session = &session;
        slot.peer = nullptr;
        workers.commit(session);
      }

      // Starting a Windows streaming session reapplies NVIDIA profile settings and can take
      // several seconds. Keep that process-wide platform state warm briefly after disconnect,
      // but also bound an accepted launch that never completes its RTSP handshake. The mutex owns
      // the single-active-session invariant and serializes both expiry paths with startup.
      std::mutex platform_lifecycle_mutex;
      bool remote_session_active {};
      std::uint64_t platform_lifecycle_generation {};
      task_pool_util::TaskPool::task_id_t pending_platform_stop {};
      session_lifecycle::resume_state_t platform_resume_state;
      task_pool_util::TaskPool::task_id_t pending_primary_restore {};
      bool platform_streaming_warm {};
      std::optional<std::uint64_t> warm_process_instance;

      void invalidate_pending_platform_stop_locked() {
        ++platform_lifecycle_generation;
        platform_resume_state.clear();
        if (pending_platform_stop) {
          task_pool.cancel(pending_platform_stop);
          pending_platform_stop = nullptr;
        }
        if (pending_primary_restore) {
          task_pool.cancel(pending_primary_restore);
          pending_primary_restore = nullptr;
        }
      }

      void schedule_primary_restore_locked(detail::primary_display_restore_retry_t retry) {
        pending_primary_restore = task_pool.pushDelayed(
          [retry]() {
            std::lock_guard delayed_lock(platform_lifecycle_mutex);
            // A cancelled callback may already be waiting on the lock. Never clear a new
            // generation's task handle, or restore its pending/active session's display.
            if (retry.generation != platform_lifecycle_generation) {
              return;
            }
            pending_primary_restore = nullptr;
            const auto action = retry.next_action(
              platform_lifecycle_generation,
              proc::proc.get_host_session_id(),
              warm_process_instance,
              remote_session_active,
              !rtsp_stream::launch_session_available(),
              platform_streaming_warm && pending_platform_stop
            );
            if (action == detail::primary_display_restore_retry_t::action_e::stop) {
              return;
            }
            if (action == detail::primary_display_restore_retry_t::action_e::wait || !proc::proc.pause_display_for_resume()) {
              // The existing grace timer remains authoritative; retries never extend it.
              schedule_primary_restore_locked(retry);
            }
          },
          1s
        );
      }

      void stop_warm_platform_locked() {
        invalidate_pending_platform_stop_locked();
        warm_process_instance.reset();
        if (!platform_streaming_warm) {
          return;
        }

        platf::streaming_will_stop();
        platform_streaming_warm = false;
      }

      bool cleanup_exited_idle_session_locked() {
        // running() owns command-exit cleanup, including prep-command undo and display retirement.
        // It is safe only after the media workers have joined, or before admitting an idle launch.
        if (remote_session_active || proc::proc.get_status().app_id == 0 || proc::proc.running() > 0) {
          return false;
        }
        rtsp_stream::clear_pending_launch_session();
        stop_warm_platform_locked();
        return true;
      }

      void schedule_platform_stop_locked(std::chrono::steady_clock::duration delay);
      void schedule_platform_stop_check_locked();

      void check_platform_stop_locked(std::uint64_t generation) {
        if (generation != platform_lifecycle_generation || remote_session_active || !platform_resume_state.deadline()) {
          return;
        }
        pending_platform_stop = nullptr;
        if (cleanup_exited_idle_session_locked()) {
          return;
        }
        if (!platform_resume_state.expired(std::chrono::steady_clock::now())) {
          // The retained driver target can reappear after a verified detach. Reuse this grace
          // poll to maintain the pause invariant, but never touch an accepted reconnect or a
          // replacement process. The absolute expiry and its generation remain unchanged.
          const detail::primary_display_restore_retry_t pause_check {generation, warm_process_instance.value_or(0)};
          if (!pending_primary_restore && pause_check.next_action(
                platform_lifecycle_generation,
                proc::proc.get_host_session_id(),
                warm_process_instance,
                remote_session_active,
                !rtsp_stream::launch_session_available(),
                platform_streaming_warm && platform_resume_state.retained()
              ) == detail::primary_display_restore_retry_t::action_e::restore) {
            (void) proc::proc.pause_display_for_resume();
          }
          // Reuse the grace task to observe command exit while disconnected. Polling preserves
          // both the absolute deadline and the generation that owns primary-display retries.
          schedule_platform_stop_check_locked();
          return;
        }

        // A new /launch can replace the retained process just before publishing its pending RTSP
        // handshake. Give that successor one fresh window without terminating it in the gap.
        const auto current_process_instance = proc::proc.get_host_session_id();
        if (current_process_instance != 0 && current_process_instance != warm_process_instance) {
          warm_process_instance = current_process_instance;
          schedule_platform_stop_locked(config::stream.ping_timeout);
          return;
        }

        invalidate_pending_platform_stop_locked();
        BOOST_LOG(info) << (platform_streaming_warm ? "Streaming session resume grace expired; terminating the retained app." : "Streaming launch handshake expired; terminating the unclaimed app.");
        rtsp_stream::clear_pending_launch_session();
        if (proc::proc.running() > 0) {
          proc::proc.terminate();
        }
        warm_process_instance.reset();
        if (platform_streaming_warm) {
          platf::streaming_will_stop();
          platform_streaming_warm = false;
        }
      }

      void schedule_platform_stop_check_locked() {
        const auto generation = platform_lifecycle_generation;
        const auto remaining = *platform_resume_state.deadline() - std::chrono::steady_clock::now();
        const auto delay = std::clamp<std::chrono::steady_clock::duration>(remaining, 0s, 1s);
        pending_platform_stop = task_pool.pushDelayed([generation]() {
          std::lock_guard delayed_lock(platform_lifecycle_mutex);
          check_platform_stop_locked(generation);
        },
                                                      delay);
      }

      void schedule_platform_stop_locked(std::chrono::steady_clock::duration delay) {
        invalidate_pending_platform_stop_locked();
        platform_resume_state.await_connection(std::chrono::steady_clock::now(), delay);
        schedule_platform_stop_check_locked();
      }

      bool activate_remote_session_locked(std::uint32_t remote_virtual_display_lease) {
        // Every accepted reconnect owns a new launch-session lease. Adopt it before potentially
        // slow platform startup.
        // Failure is terminal for this RTSP session: streaming without the lease would let local
        // AR reclaim and remove the virtual display while capture is running.
        if (!proc::proc.activate_remote_virtual_display_lease(remote_virtual_display_lease)) {
          BOOST_LOG(error) << "Refusing to start a streaming session without its virtual-display ownership lease."sv;
          return false;
        }

        invalidate_pending_platform_stop_locked();
        platform_resume_state.activate();
        if (platform_streaming_warm) {
          BOOST_LOG(info) << "Reusing warm streaming platform state after a short disconnect."sv;
        } else {
          platf::streaming_will_start();
          platform_streaming_warm = true;
        }

        warm_process_instance.reset();
        return true;
      }

      void retain_or_stop_session_locked() {
        if (remote_session_active) {
          return;
        }
        const auto disconnected_at = std::chrono::steady_clock::now();

        // Restore the physical desktop and detach the virtual source immediately. Retain the
        // app and driver device for reconnect, without leaving their desktop region active.
        BOOST_LOG(info) << "Remote streaming session is inactive; restoring the original display topology."sv;
        const bool primary_restored = proc::proc.pause_display_for_resume();
        if (cleanup_exited_idle_session_locked()) {
          return;
        }
        const auto process_status = proc::proc.get_status();
        if (process_status.app_id == 0) {
          // There is no app/session state worth retaining. Match the historical cleanup path.
          rtsp_stream::clear_pending_launch_session();
          stop_warm_platform_locked();
          return;
        }

        if (!platform_streaming_warm) {
          rtsp_stream::clear_pending_launch_session();
          proc::proc.terminate();
          stop_warm_platform_locked();
          return;
        }

        // Keep the app and remote virtual-display ownership active during the grace. Capture,
        // encoding, transport, and input are already stopped with the session; retaining process
        // ownership prevents another presentation path from claiming the display. The next
        // accepted reconnect reactivates that exact device and reapplies its display policy
        // before capture. A fresh launch can replace this retained ownership immediately.
        const auto host_session_id = proc::proc.get_host_session_id();
        warm_process_instance = host_session_id == 0 ? std::nullopt : std::optional<std::uint64_t> {host_session_id};
        // The current launch reservation may still be waiting for its control connection.
        // Preserve the platform for at least that RTSP handshake window even when reconnect
        // grace is off or configured shorter than ping_timeout.
        const bool validated_launch_pending = !rtsp_stream::launch_session_available();
        if (config::stream.session_resume_grace <= 0ms && !validated_launch_pending) {
          BOOST_LOG(info) << "Session resume grace is disabled; terminating the retained app."sv;
          proc::proc.terminate();
          stop_warm_platform_locked();
          return;
        }
        const auto retention = validated_launch_pending ?
                                 std::max(config::stream.session_resume_grace, config::stream.ping_timeout) :
                                 config::stream.session_resume_grace;
        if (!platform_resume_state.retained()) {
          invalidate_pending_platform_stop_locked();
          platform_resume_state.disconnect(disconnected_at, retention);
          schedule_platform_stop_check_locked();
        }
        if (!primary_restored && host_session_id != 0 && !pending_primary_restore) {
          schedule_primary_restore_locked({platform_lifecycle_generation, host_session_id});
        }
      }

      bool prepare_new_session_locked() {
        if (remote_session_active || !rtsp_stream::launch_session_available() || platform_resume_state.phase() == session_lifecycle::phase_e::connecting) {
          return false;
        }
        if (proc::proc.get_status().app_id > 0) {
          if (!platform_resume_state.retained()) {
            return false;
          }
          BOOST_LOG(info) << "A fresh connection is replacing the retained streaming session."sv;
          proc::proc.terminate(false, false);
        }
        stop_warm_platform_locked();
        return true;
      }
    }  // namespace

    struct platform_launch_guard_t::impl_t {
      impl_t():
          lock {platform_lifecycle_mutex},
          idle {!remote_session_active} {
        // A command can exit while disconnected, after the teardown-time check. Reap it before
        // HTTP reads the retained identity, without disturbing a live app's existing grace timer.
        cleanup_exited_idle_session_locked();
      }

      std::unique_lock<std::mutex> lock;
      bool idle;
      bool committed {};
    };

    platform_launch_guard_t::platform_launch_guard_t(std::unique_ptr<impl_t> impl):
        _impl {std::move(impl)} {
    }

    platform_launch_guard_t::platform_launch_guard_t(platform_launch_guard_t &&) noexcept = default;
    platform_launch_guard_t &platform_launch_guard_t::operator=(platform_launch_guard_t &&) noexcept = default;
    platform_launch_guard_t::~platform_launch_guard_t() = default;

    bool platform_launch_guard_t::idle() const {
      return _impl && _impl->idle;
    }

    void platform_launch_guard_t::commit() {
      if (!_impl || _impl->committed || !_impl->lock.owns_lock()) {
        return;
      }

      if (_impl->idle && !remote_session_active) {
        const auto current_process_instance = proc::proc.get_host_session_id();
        if (current_process_instance != 0) {
          warm_process_instance = current_process_instance;
        }
        const auto timeout = platform_streaming_warm ?
                               std::max(config::stream.session_resume_grace, config::stream.ping_timeout) :
                               config::stream.ping_timeout;
        schedule_platform_stop_locked(timeout);
        BOOST_LOG(debug) << "Committed streaming launch reservation with bounded lifetime."sv;
      }
      _impl->committed = true;
      _impl->lock.unlock();
    }

    bool platform_launch_guard_t::restore_primary_display() {
      if (!_impl || !_impl->idle || _impl->committed || !_impl->lock.owns_lock() || remote_session_active) {
        return false;
      }
      const bool launch_pending = !rtsp_stream::launch_session_available();
      if (!launch_pending && proc::proc.pause_display_for_resume()) {
        return true;
      }
      const auto host_session_id = proc::proc.get_host_session_id();
      const detail::primary_display_restore_retry_t retry {platform_lifecycle_generation, host_session_id};
      if (!pending_primary_restore && retry.next_action(platform_lifecycle_generation, host_session_id, warm_process_instance, remote_session_active, launch_pending, platform_streaming_warm && pending_platform_stop) != detail::primary_display_restore_retry_t::action_e::stop) {
        schedule_primary_restore_locked(retry);
      }
      return false;
    }

    bool platform_launch_guard_t::prepare_new_session() {
      return _impl && _impl->idle && !_impl->committed && _impl->lock.owns_lock() &&
             prepare_new_session_locked();
    }

    local_session_admission_e prepare_local_ar_session(bool replace_retained) {
      // Remote launch holds this lock while asking the local controller to stop. The controller
      // must remain able to service that request instead of waiting behind the remote caller.
      std::unique_lock lock(platform_lifecycle_mutex, std::try_to_lock);
      if (!lock.owns_lock()) {
        return local_session_admission_e::retry;
      }
      cleanup_exited_idle_session_locked();
      if (!replace_retained && platform_resume_state.retained() && proc::proc.get_status().app_id > 0) {
        return local_session_admission_e::remote_busy;
      }
      return prepare_new_session_locked() ? local_session_admission_e::ready : local_session_admission_e::remote_busy;
    }

    platform_launch_guard_t guard_platform_launch() {
      return platform_launch_guard_t {std::make_unique<platform_launch_guard_t::impl_t>()};
    }

    state_e state(session_t &session) {
      return session.state.load(std::memory_order_relaxed);
    }

#ifdef SUNSHINE_TESTS
    bool platform_lifecycle_lock_available_for_test() {
      bool available = false;
      std::thread probe([&]() {
        std::unique_lock lock {platform_lifecycle_mutex, std::try_to_lock};
        available = lock.owns_lock();
      });
      probe.join();
      return available;
    }

    void set_state_for_test(session_t &session, state_e state) {
      session.state.store(state, std::memory_order_relaxed);
    }

    bool claim_active_slot_for_test() {
      std::lock_guard lock(platform_lifecycle_mutex);
      if (remote_session_active) {
        return false;
      }
      remote_session_active = true;
      return true;
    }

    void release_active_slot_for_test() {
      std::lock_guard lock(platform_lifecycle_mutex);
      remote_session_active = false;
    }

    void retain_or_stop_session_for_test(bool platform_warm) {
      std::lock_guard lock(platform_lifecycle_mutex);
      // Model a previously started platform without acquiring any Windows scheduling/input state.
      platform_streaming_warm = platform_warm;
      retain_or_stop_session_locked();
    }

    std::uint64_t platform_lifecycle_generation_for_test() {
      std::lock_guard lock(platform_lifecycle_mutex);
      return platform_lifecycle_generation;
    }

    std::optional<std::chrono::steady_clock::time_point> platform_stop_deadline_for_test() {
      std::lock_guard lock(platform_lifecycle_mutex);
      return platform_resume_state.deadline();
    }

    void check_platform_stop_for_test(std::optional<std::uint64_t> dispatched_generation) {
      std::lock_guard lock(platform_lifecycle_mutex);
      const auto generation = dispatched_generation.value_or(platform_lifecycle_generation);
      if (generation == platform_lifecycle_generation && pending_platform_stop) {
        task_pool.cancel(pending_platform_stop);
      }
      check_platform_stop_locked(generation);
    }

    bool worker_start_rollback_for_test(std::size_t fail_before_worker) {
      std::atomic_int executed {};
      pending_session_workers_t workers;
      const auto entry = [&executed]() {
        ++executed;
      };
      const bool prepared = workers.prepare(
        entry,
        entry,
        entry,
        entry,
        fail_before_worker
      );
      return !prepared && executed.load() == 0 && !workers.has_joinable_worker();
    }

    control_registration_test_result_t control_registration_after_preparation_for_test(
      std::chrono::milliseconds preparation,
      std::chrono::milliseconds ping_timeout
    ) {
      control_registration_test_result_t result;
      session_t session;
      sync_util::sync_t<control_server_t::session_slot_t> slot;
      auto now = std::chrono::steady_clock::time_point {1s};
      // Seed the old startup-era deadline, then model platform/input setup taking longer than it.
      session.pingTimeout = now + ping_timeout;
      session.state.store(state_e::STOPPED, std::memory_order_relaxed);
      now += preparation;
      result.expired_before_publication = now > session.pingTimeout;
      std::atomic_int ready_workers {};
      pending_session_workers_t workers;
      const auto observe_registration = [&]() {
        auto lock = slot.lock();
        if (slot->session == &session && !slot->peer && session.state.load(std::memory_order_relaxed) == state_e::RUNNING && session.startup_ready.peek() && session.pingTimeout == now + ping_timeout) {
          ++ready_workers;
        }
      };
      result.workers_prepared = workers.prepare(observe_registration, observe_registration, observe_registration, observe_registration);
      if (!result.workers_prepared) {
        return result;
      }
      {
        auto lock = slot.lock();
        publish_control_registration(session, *slot, workers, ping_timeout, [&]() {
          ++result.clock_samples;
          return now;
        });
        result.remaining_ping_budget = session.pingTimeout - now;
      }
      session.audioThread.join();
      session.videoThread.join();
      session.media_senders.stop_and_join();
      result.ready_workers = ready_workers.load();
      return result;
    }
#endif

    std::string uuid(const session_t &session) {
      return session.device_uuid;
    }

    bool uuid_match(const session_t &session, const std::string_view &uuid) {
      return session.device_uuid == uuid;
    }

    std::string client_name(const session_t &session) {
      std::lock_guard lock(session.device_info_mutex);
      return session.device_name;
    }

    crypto::PERM permissions(const session_t &session) {
      return session.permission.load(std::memory_order_acquire);
    }

    client_policy_result_e update_client_policy(
      session_t &session,
      std::uint64_t generation,
      std::string_view name,
      crypto::PERM new_permissions,
      bool revoked
    ) {
      std::string previous_name;
      bool should_stop;
      {
        // Serialize policy publications per session. Merely making the permission atomic is not
        // enough: two administrator updates can otherwise publish out of generation order.
        std::lock_guard lock(session.device_info_mutex);
        if (generation <= session.client_policy_generation) {
          return client_policy_result_e::ignored;
        }

        session.client_policy_generation = generation;
        if (session.input) {
          input::update_permissions(session.input, new_permissions);
        }
        session.permission.store(new_permissions, std::memory_order_release);
        if (!(new_permissions & crypto::PERM::input_microphone) && session.microphone_receiver) {
          session.microphone_receiver->stop();
        }
        previous_name = session.device_name;
        session.device_name = name;
        should_stop = revoked || !(new_permissions & crypto::PERM::_allow_view);
      }

      if (should_stop) {
        BOOST_LOG(debug) << "Session: Client authorization revoked for [" << previous_name << "], disconnecting...";
        return client_policy_result_e::disconnect;
      }

      BOOST_LOG(debug) << "Session: Permission updated for [" << previous_name << "]";
      if (previous_name != name) {
        BOOST_LOG(debug) << "Session: Device name changed from [" << previous_name << "] to [" << name << "]";
      }
      return client_policy_result_e::updated;
    }

    static bool transition_to_stopping(session_t &session, bool graceful) {
      std::lock_guard lock(session.stop_mutex);
      if (session.state.load(std::memory_order_relaxed) != state_e::RUNNING) {
        return false;
      }

      session.video->active.store(false, std::memory_order_release);
      session.audio->active.store(false, std::memory_order_release);
      if (session.microphone_receiver) {
        session.microphone_receiver->stop();
      }
      if (session.control.sbs_telemetry_subscription) {
        session.control.sbs_telemetry_subscription->update(
          false,
          false,
          SBS_TELEMETRY_INTERVAL_BACKGROUND_MS
        );
      }
      session.startup_ready.stop();
      session.control_ready.stop();
      session.graceful_stop_requested.store(graceful, std::memory_order_relaxed);
      session.state.store(state_e::STOPPING, std::memory_order_release);
      stop_live_video_mode_worker(session);
      return true;
    }

    void stop(session_t &session) {
      if (!transition_to_stopping(session, false)) {
        return;
      }

      session.shutdown_event->raise(true);
      session.video->wake_encoder();
      rtsp_stream::notify_session_stopping();
    }

    void graceful_stop(session_t &session) {
      if (!transition_to_stopping(session, true)) {
        return;
      }

      session.shutdown_event->raise(true);
      session.video->wake_encoder();
      rtsp_stream::notify_session_stopping();
    }

    bool stop_if_client_policy_current(session_t &session, std::uint64_t generation, bool graceful) {
      {
        // Make validation and the RUNNING -> STOPPING claim one transaction with policy updates.
        // A delayed revoke can no longer stop a session after a newer allow policy was applied.
        std::lock_guard lock(session.device_info_mutex);
        if (generation != session.client_policy_generation || !transition_to_stopping(session, graceful)) {
          return false;
        }
      }

      session.shutdown_event->raise(true);
      session.video->wake_encoder();
      rtsp_stream::notify_session_stopping();
      return true;
    }

    void join(session_t &session) {
      if (session.microphone_receiver) {
        session.microphone_receiver->stop();
      }
      // A live desktop resize owns process/display state and can outlive the control thread.
      // Reap it before arming the NVENC hang watchdog and before releasing the active-session
      // slot, so a successor can never overlap stale work from this launch.
      {
        const detail::session_join_watchdog_t mode_worker_watchdog {
          [](std::function<void()> expired) {
            return task_pool.pushDelayed(std::move(expired), 30s);
          },
          [](auto task_id) {
            task_pool.cancel(task_id);
          },
          []() {
            BOOST_LOG(fatal) << "Hang detected! Live video-mode worker failed to terminate in 30 seconds."sv;
            logging::log_flush();
            lifetime::debug_trap();
          },
        };
        join_live_video_mode_worker(session);
      }

      // Current Nvidia drivers have a bug where NVENC can deadlock the encoder thread with hardware-accelerated
      // GPU scheduling enabled. If this happens, we will terminate ourselves and the service can restart.
      // The alternative is that Sunshine can never start another session until it's manually restarted.
      detail::join_workers_before_session_cleanup(
        []() {
          return detail::session_join_watchdog_t {
            [](std::function<void()> expired) {
              return task_pool.pushDelayed(std::move(expired), 10s);
            },
            [](auto task_id) {
              task_pool.cancel(task_id);
            },
            []() {
              BOOST_LOG(fatal) << "Hang detected! Session failed to terminate in 10 seconds."sv;
              logging::log_flush();
              lifetime::debug_trap();
            },
          };
        },
        [&session]() {
          BOOST_LOG(debug) << "Waiting for video to end..."sv;
          session.videoThread.join();
          BOOST_LOG(debug) << "Waiting for audio to end..."sv;
          session.audioThread.join();
          BOOST_LOG(debug) << "Waiting for control to end..."sv;
          session.controlEnd.view();
          // Reset input on session stop to avoid stuck repeated keys.
          BOOST_LOG(debug) << "Resetting Input..."sv;
          input::reset(session.input).wait();
        },
        [&session]() {
          // Worker shutdown is complete and its watchdog is disarmed. App exit, undo commands
          // and display restoration have separate bounds and may legitimately take longer.
          session.media_senders.stop_and_join();
          // Keep the active slot until this lock is held so a successor cannot overlap teardown.
          std::lock_guard lifecycle_lock(platform_lifecycle_mutex);
          if (!remote_session_active) {
            BOOST_LOG(error) << "Streaming session ended without owning the active-session slot."sv;
          } else {
            remote_session_active = false;
            retain_or_stop_session_locked();
          }
        }
      );

      BOOST_LOG(debug) << "Session ended"sv;
    }

    int start(session_t &session) {
      session.broadcast_ref = broadcast.ref();
      if (!session.broadcast_ref) {
        return -1;
      }

      // Claim the sole session under the same lifecycle lock used by launch-time display
      // mutation and encoder probing. A stale RTSP handshake therefore cannot create a second
      // capture/encoder/input stack behind the HTTP admission check.
      {
        std::lock_guard lifecycle_lock(platform_lifecycle_mutex);
        if (remote_session_active) {
          BOOST_LOG(warning) << "Rejecting a second active streaming session."sv;
          return -1;
        }
        remote_session_active = true;
        if (!activate_remote_session_locked(session.launch_session_id)) {
          remote_session_active = false;
          return -1;
        }
        // Capture input ownership only after admission, under the same lock that protects
        // launch/replacement. A stale handshake must not inject its delayed startup mouse nudge.
        session.input = input::alloc(
          session.mail,
          permissions(session),
          session.config.client_supports_authored_pcm
        );
      }

      auto rollback_active_session = []() {
        std::lock_guard lifecycle_lock(platform_lifecycle_mutex);
        if (remote_session_active) {
          remote_session_active = false;
          retain_or_stop_session_locked();
        }
      };

      pending_session_workers_t workers;
      if (!workers.prepare(
            [&session]() {
              audioThread(&session);
            },
            [&session]() {
              videoThread(&session);
            },
            [&session]() {
              run_video_sender(
                session.broadcast_ref->video_sock,
                session.media_senders.video_packets(),
                session.video,
                session.shutdown_event,
                session.broadcast_ref->video_epoch
              );
            },
            [&session]() {
              run_audio_sender(
                session.broadcast_ref->audio_sock,
                session.media_senders.audio_packets(),
                session.audio,
                session.shutdown_event
              );
            },
            std::numeric_limits<std::size_t>::max(),
            [&session]() {
              stop(session);
            }
          )) {
        rollback_active_session();
        return -1;
      }

      // Publish the raw control registration only after platform startup and all worker-thread
      // constructions succeed. Its owner remains the RTSP slot; controlEnd is the lifetime barrier.
      {
        auto slot_lock = session.broadcast_ref->control_server._session.lock();
        if (session.broadcast_ref->control_server._session->session) {
          BOOST_LOG(error) << "Control server still has a registered session while the active slot is empty."sv;
          workers.cancel();
          rollback_active_session();
          return -1;
        }

        // A/V capture may already have received its UDP ping. Release it only after display/audio
        // platform state has finished resuming; control_ready independently protects source routing.
        publish_control_registration(
          session,
          *session.broadcast_ref->control_server._session,
          workers,
          config::stream.ping_timeout,
          std::chrono::steady_clock::now
        );
        if (session.microphone_receiver) {
          if (!!(permissions(session) & crypto::PERM::input_microphone)) {
            session.microphone_receiver->activate();
          } else {
            session.microphone_receiver->stop();
          }
        }
      }

      return 0;
    }

    void flush_platform_state() {
      std::lock_guard lock(platform_lifecycle_mutex);
      stop_warm_platform_locked();
    }

    std::shared_ptr<session_t> alloc(config_t &config, rtsp_stream::launch_session_t &launch_session) {
      auto session = std::make_shared<session_t>();

      auto mail = std::make_shared<safe::mail_raw_t>();

      session->shutdown_event = mail->event<bool>(mail::shutdown);
      session->launch_session_id = launch_session.id;
      session->device_name = launch_session.display_context->device_name;
      session->device_uuid = launch_session.display_context->unique_id;
      session->client_policy_generation = 0;
      session->permission.store(launch_session.perm, std::memory_order_relaxed);

      session->config = config;
      session->microphone_receiver = launch_session.take_microphone();

      // Seed the effective-mode view with the launch mode so a live request refused before the
      // encode loop has published anything still reports a real mode rather than zeros. The video
      // thread receives a copy of session->config.monitor, so both sides share this publisher.
      session->config.monitor.effective_mode = std::make_shared<video::effective_video_mode_publisher_t>(
        video::effective_video_mode_t {
          config.monitor.width,
          config.monitor.height,
          config.monitor.framerateX100 > 0 ? config.monitor.framerateX100 : config.monitor.framerate * 100,
          config.monitor.bitrate,
          config.monitor.width,
          config.monitor.height,
          0,
          0,
          config.monitor.sbs_mode,
          0,
        }
      );
      session->config.monitor.sbs_telemetry_subscription =
        std::make_shared<video::sbs_telemetry_subscription_t>();
      session->config.monitor.stream_gamma_state = std::make_shared<video::stream_gamma_publisher_t>(
        session->config.monitor.stream_gamma
      );

      session->video = std::make_shared<video_channel_t>();
      session->video->packet_size = config.packetsize;
      session->video->min_required_fec_packets = config.minRequiredFecPackets;
      session->video->source_frame_id_negotiated = config.client_supports_source_frame_id_v1;
      session->video->effective_mode = session->config.monitor.effective_mode;
      session->config.monitor.encode_wake = session->video->encode_wake;
      session->config.monitor.atomic_presentation = session->live_video_mode_worker.transactions;
      session->config.monitor.sbs_debug_dump_pending =
        session->video->sbs_debug_dump_pending;
      session->requested_sbs_mode->store(
        session->config.monitor.sbs_mode,
        std::memory_order_relaxed
      );
      session->config.monitor.requested_sbs_mode = session->requested_sbs_mode;

      session->audio = std::make_shared<audio_channel_t>();
      session->audio->packet_duration = config.audio.packetDuration;

      session->control.connect_data = launch_session.control_connect_data;
      session->control.feedback_queue = mail->queue<platf::gamepad_feedback_msg_t>(mail::gamepad_feedback);
      session->control.haptics_feedback_queue =
        mail->queue<platf::gamepad_feedback_msg_t>(mail::ds5_haptics_feedback, 64);
      session->control.hdr_queue = mail->event<video::hdr_info_t>(mail::hdr);
      session->control.depth_status_event = mail->event<int>(mail::sbs_depth_status);
      session->control.live_video_mode_ack_queue = mail->queue<live_video_mode_ack_t>(
        mail::live_video_mode_ack,
        LIVE_VIDEO_MODE_ACK_QUEUE_LIMIT
      );
      session->control.video_mode_queue =
        mail->queue<video::video_mode_change_t>(mail::video_mode);
      session->control.stream_gamma_queue = mail->queue<video::stream_gamma_request_t>(
        mail::stream_gamma, STREAM_GAMMA_QUEUE_LIMIT
      );
      session->control.stream_gamma_ack_queue = mail->queue<video::stream_gamma_ack_t>(
        mail::stream_gamma_ack, 2 * STREAM_GAMMA_QUEUE_LIMIT
      );
      session->control.sbs_telemetry_event =
        mail->event<video::sbs_telemetry_snapshot_t>(mail::sbs_telemetry);
      session->control.game_source_status_event =
        mail->event<video::game_source_state_t>(mail::game_source_status);
      session->config.monitor.game_source_status_event = session->control.game_source_status_event;
      session->control.sbs_telemetry_subscription =
        session->config.monitor.sbs_telemetry_subscription;
      session->control.cipher = crypto::cipher::gcm_t {
        launch_session.gcm_key,
        false
      };

      session->video->idr_events = mail->event<bool>(mail::idr);
      session->video->invalidate_ref_frames_events = mail->event<std::pair<int64_t, int64_t>>(mail::invalidate_ref_frames);
      session->video->lowseq = 0;
      session->video->ping_payload = launch_session.av_ping_payload;
      session->video->cipher = crypto::cipher::gcm_t {
        launch_session.gcm_key,
        false
      };
      session->video->gcm_iv_counter = 0;

      constexpr auto max_block_size = crypto::cipher::round_to_pkcs7_padded(2048);

      util::buffer_t<char> shards {RTPA_TOTAL_SHARDS * max_block_size};
      util::buffer_t<uint8_t *> shards_p {RTPA_TOTAL_SHARDS};

      for (auto x = 0; x < RTPA_TOTAL_SHARDS; ++x) {
        shards_p[x] = (uint8_t *) &shards[x * max_block_size];
      }

      // Audio FEC spans multiple audio packets,
      // therefore its session specific
      session->audio->shards = std::move(shards);
      session->audio->shard_ptrs = std::move(shards_p);

      session->audio->fec_packet.rtp.header = 0x80;
      session->audio->fec_packet.rtp.packetType = 127;
      session->audio->fec_packet.rtp.timestamp = 0;
      session->audio->fec_packet.rtp.ssrc = 0;

      session->audio->fec_packet.fecHeader.payloadType = 97;
      session->audio->fec_packet.fecHeader.ssrc = 0;

      session->audio->cipher = crypto::cipher::cbc_t {
        launch_session.gcm_key,
        true
      };

      session->audio->ping_payload = launch_session.av_ping_payload;
      std::uint32_t av_ri_key_id;
      std::memcpy(&av_ri_key_id, launch_session.iv.data(), sizeof(av_ri_key_id));
      session->audio->av_ri_key_id = util::endian::big(av_ri_key_id);
      session->audio->sequence_number = 0;
      session->audio->timestamp = 0;

      session->control.peer = nullptr;
      session->state.store(state_e::STOPPED, std::memory_order_relaxed);

      session->mail = std::move(mail);

      return session;
    }
  }  // namespace session
}  // namespace stream
