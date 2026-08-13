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
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>

// lib includes
#include <boost/endian/arithmetic.hpp>
#include <openssl/err.h>
#include <rs.h>

extern "C" {
  // clang-format off
#include <moonlight-common-c/src/Limelight-internal.h>
  // clang-format on
}

// local includes
#include "config.h"
#include "crypto.h"
#include "globals.h"
#include "host_sbs_resolution.h"
#include "input.h"
#include "logging.h"
#include "network.h"
#include "platform/common.h"
#include "process.h"
#include "rtsp.h"
#include "stream.h"
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
    bool diagnostics_enabled,
    int requested_sbs_mode,
    bool has_session_request_latch
  ) noexcept {
    return diagnostics_enabled && requested_sbs_mode == ::video::SBS_AI &&
           has_session_request_latch;
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
    constexpr std::uint16_t rumble_triggers = 0x5500;
    constexpr std::uint16_t set_motion_event = 0x5501;
    // Direction-specific reuse: Moonlight sends frame FEC status on this ID, while Sunshine sends
    // controller RGB feedback on it. Name both directions so an inbound FEC report is never
    // mistaken for an echoed LED command.
    constexpr std::uint16_t frame_fec_status = 0x5502;
    constexpr std::uint16_t set_rgb_led_feedback = 0x5502;
    constexpr std::uint16_t set_adaptive_triggers = 0x5503;
    constexpr std::uint16_t set_sbs_mode = 0x3003;
    constexpr std::uint16_t sbs_debug_dump = 0x3004;
    constexpr std::uint16_t depth_status = 0x3006;
    // 0x3005 is retired (Set Depth Model); older clients may still emit it, so it is not reused.
    constexpr std::uint16_t set_video_mode = 0x3007;
    constexpr std::uint16_t live_video_mode_ack = 0x3008;
    constexpr std::uint16_t sbs_telemetry_subscription = 0x3009;
    constexpr std::uint16_t sbs_telemetry_state = 0x300A;
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

    std::uint8_t unknown[2];
  };

  static_assert(
    sizeof(video_short_frame_header_t) == 8,
    "Short frame header must be 8 bytes"
  );

  struct video_packet_raw_t {
    uint8_t *payload() {
      return (uint8_t *) (this + 1);
    }

    RTP_PACKET rtp;
    char reserved[4];

    NV_VIDEO_PACKET packet;
  };

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

  struct control_rumble_triggers_t {
    control_header_v2 header;

    std::uint16_t id;
    std::uint16_t left;
    std::uint16_t right;
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

  // Client->host live stream geometry/rate change (Apollo protocol extension 0x3007). The frame
  // rate is carried in hundredths of a hertz so fractional rates (23.976, 29.97) survive the trip.
  // `bitrate_kbps` is the total wire budget, exactly as RTSP ANNOUNCE reports it; the host applies
  // its own clamp and FEC/audio deduction before the value reaches the encoder.
  struct control_set_video_mode_t {
    std::uint16_t width;
    std::uint16_t height;
    std::uint16_t framerate_x100;
    // Opaque client-generated correlation token, echoed verbatim in the 0x3008 acknowledgement.
    // The host never validates it and never infers an ordering from it.
    std::uint16_t request_id;
    std::uint32_t bitrate_kbps;
  };

  // Host->client acknowledgement of a 0x3007 request (Apollo protocol extension 0x3008). The body
  // is a raw byte array so its exact little-endian layout is owned by
  // encode_live_video_mode_ack_payload() and cannot drift with struct packing.
  struct control_live_video_mode_ack_t {
    control_header_v2 header;

    std::uint8_t body[LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE];
  };

  struct control_sbs_telemetry_state_t {
    control_header_v2 header;

    std::uint8_t body[SBS_TELEMETRY_STATE_PAYLOAD_SIZE];
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

  constexpr std::size_t round_to_pkcs7_padded(std::size_t size) {
    return ((size + 15) / 16) * 16;
  }

  live_video_mode_ack_e live_video_mode_ack_status(proc::live_video_mode_result_e result) {
    switch (result) {
      case proc::live_video_mode_result_e::applied:
      // "Unchanged" means the desktop already presented the requested geometry, so the mode the
      // client asked for is live once the encoder rebuilds. From the client's point of view that
      // is indistinguishable from a real transition, and it must not be reported as a failure.
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

  bool encode_live_video_mode_ack_payload(
    const live_video_mode_ack_t &ack,
    std::uint8_t (&out)[LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE]
  ) {
    constexpr std::int64_t u16_max = std::numeric_limits<std::uint16_t>::max();
    constexpr std::int64_t u32_max = std::numeric_limits<std::uint32_t>::max();
    if (ack.request_id < 0 || ack.request_id > u16_max || ack.applied_width < 0 || ack.applied_width > u16_max || ack.applied_height < 0 || ack.applied_height > u16_max || ack.applied_framerate_x100 < 0 || ack.applied_framerate_x100 > u16_max || ack.applied_bitrate_kbps < 0 || ack.applied_bitrate_kbps > u32_max) {
      return false;
    }

    const std::uint16_t fields[5] = {
      static_cast<std::uint16_t>(ack.request_id),
      static_cast<std::uint16_t>(ack.status),
      static_cast<std::uint16_t>(ack.applied_width),
      static_cast<std::uint16_t>(ack.applied_height),
      static_cast<std::uint16_t>(ack.applied_framerate_x100),
    };
    for (std::size_t i = 0; i < 5; ++i) {
      out[i * 2] = static_cast<std::uint8_t>(fields[i] & 0xFF);
      out[i * 2 + 1] = static_cast<std::uint8_t>((fields[i] >> 8) & 0xFF);
    }

    const auto bitrate = static_cast<std::uint32_t>(ack.applied_bitrate_kbps);
    out[10] = static_cast<std::uint8_t>(bitrate & 0xFF);
    out[11] = static_cast<std::uint8_t>((bitrate >> 8) & 0xFF);
    out[12] = static_cast<std::uint8_t>((bitrate >> 16) & 0xFF);
    out[13] = static_cast<std::uint8_t>((bitrate >> 24) & 0xFF);
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
    const std::array<float, 9> float_fields {
      snapshot.pop_floor,
      snapshot.pop_ceiling,
      snapshot.effective_pop,
      snapshot.edge_fraction,
      snapshot.change_fraction,
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

    encoded[0] = SBS_TELEMETRY_VERSION;
    encoded[1] = status;
    write_u16(2, state.request_id);
    write_u32(4, snapshot.generation);
    write_u32(8, snapshot.sequence);
    write_u32(12, snapshot.valid_fields);
    write_u32(16, snapshot.runtime_flags);
    write_u16(20, snapshot.depth_width);
    write_u16(22, snapshot.depth_height);
    encoded[24] = snapshot.zero_plane_mode;
    // encoded[25..27] remain reserved zero.
    write_f32(28, snapshot.pop_floor);
    write_f32(32, snapshot.pop_ceiling);
    write_f32(36, snapshot.effective_pop);
    write_f32(40, snapshot.edge_fraction);
    write_f32(44, snapshot.change_fraction);
    write_f32(48, snapshot.zero_anchor_shift_px);
    write_f32(52, snapshot.subject_depth);
    write_f32(56, snapshot.valid_depth_fraction);
    write_f32(60, snapshot.effective_range_width);
    write_u32(64, snapshot.scene_age);
    write_u32(68, snapshot.hard_cut_count);
    write_u32(72, snapshot.external_cut_count);
    write_u32(76, snapshot.empty_raw_count);
    write_u32(80, snapshot.collapsed_raw_count);
    write_u32(84, snapshot.sampled_frame_id);
    std::copy(encoded.begin(), encoded.end(), out);
    return true;
  }

  constexpr std::size_t MAX_AUDIO_PACKET_SIZE = 1400;

  using audio_aes_t = std::array<char, round_to_pkcs7_padded(MAX_AUDIO_PACKET_SIZE)>;

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

  // return bytes written on success
  // return -1 on error
  static inline int encode_audio(const audio::buffer_t &plaintext, uint8_t *destination, crypto::aes_t &iv, crypto::cipher::cbc_t &cbc) {
    return cbc.encrypt(std::string_view {(char *) std::begin(plaintext), plaintext.size()}, destination, &iv);
  }

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
    std::thread video_thread;
    std::thread audio_thread;
    std::thread control_thread;

    asio::io_context io_context;

    udp::socket video_sock {io_context};
    udp::socket audio_sock {io_context};

    control_server_t control_server;
  };

  // Encoded packets may outlive their RTSP session while waiting in the global broadcast queues.
  // Keep the exact send-side state alive independently; retaining session_t itself would create a
  // cycle through session_t::broadcast_ref and could run broadcast teardown on a broadcast worker.
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

    std::string ping_payload;
    std::uint32_t lowseq;
    udp::endpoint peer;
    crypto::cipher::gcm_t cipher;
    std::uint64_t gcm_iv_counter;
    safe::mail_raw_t::event_t<bool> idr_events;
    safe::mail_raw_t::event_t<std::pair<int64_t, int64_t>> invalidate_ref_frames_events;
    std::unique_ptr<platf::deinit_t> qos;
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

  // Everything needed to finish or reject one asynchronous live display-mode request. These
  // values deliberately do not retain session_t: the session owns and joins the worker instead.
  struct live_video_mode_request_t {
    safe::mail_t mail;
    std::shared_ptr<video_channel_t> video;
    // What the encode loop is really running. A refused request is answered with this, so the
    // client learns the mode still in effect instead of getting its rejected request back.
    std::shared_ptr<video::effective_video_mode_publisher_t> effective_mode;
    video::video_mode_change_t change;
    std::uint32_t launch_session_id;
    std::string client_name;
  };

  struct live_video_mode_worker_t {
    std::mutex mutex;
    std::condition_variable wake;
    bool stopping = false;
    // Only the most recent request that has not begun matters; a burst collapses to one display
    // transition. Once begun, the serial gate retains exclusive ownership through encoder
    // confirmation and any required desktop rollback.
    std::optional<live_video_mode_request_t> pending;
    std::optional<video::video_mode_applied_t> completion;
    detail::live_video_mode_serial_gate_t serial_gate;
    std::thread thread;

    ~live_video_mode_worker_t() {
      {
        std::lock_guard lock(mutex);
        stopping = true;
        pending.reset();
        serial_gate.cancel_pending();
      }
      wake.notify_all();
      if (thread.joinable()) {
        thread.join();
      }
    }
  };

  struct session_t {
    config_t config;

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
      safe::mail_raw_t::event_t<video::hdr_info_t> hdr_queue;
      // mail_raw_t keeps only a weak reference to each post. Retain this queue for the full
      // session lifetime so worker/encode-thread replies cannot disappear between raise() and
      // the control thread's next drain.
      safe::mail_raw_t::queue_t<live_video_mode_ack_t> live_video_mode_ack_queue;
      safe::mail_raw_t::event_t<video::sbs_telemetry_snapshot_t> sbs_telemetry_event;
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
    // Runtime request latch, updated by 0x3003 before the encode rebuild completes. Dump 3D is
    // accepted only while the requesting session is explicitly in Host SBS AI mode.
    std::shared_ptr<std::atomic<int>> requested_sbs_mode =
      std::make_shared<std::atomic<int>>(::video::SBS_OFF);

    safe::mail_raw_t::event_t<bool> shutdown_event;
    safe::signal_t controlEnd;

    // The client can send its audio/video UDP ping while platform resume work is still changing
    // the display topology, and before ENet publishes the local source address. Capture must not
    // feed the broadcast queues until both halves of session startup are complete.
    safe::signal_t startup_ready;
    safe::signal_t control_ready;

    std::mutex stop_mutex;
    std::atomic_bool graceful_stop_requested {false};
    std::atomic<session::state_e> state;

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
      int request_id,
      live_video_mode_ack_e status
    ) {
      const auto mode = effective_mode ?
                          effective_mode->current() :
                          video::effective_video_mode_t {};
      queue_live_video_mode_ack(
        mail,
        live_video_mode_ack_t {
          request_id,
          status,
          mode.width,
          mode.height,
          mode.framerateX100,
          mode.bitrate,
        }
      );
    }

    /** Hand one serialized request to the encode loop. */
    void commit_live_video_mode(const live_video_mode_request_t &request) {
      // Do not publish pacing here. The requested encoder may fail, so video.cpp publishes the
      // coherent effective bitrate/cadence pair only after init_encoder() succeeds.
      request.mail->queue<video::video_mode_change_t>(mail::video_mode)->raise(request.change);
    }

    /**
     * Deliver an encoder completion to the one matching in-flight transaction.
     *
     * Client request ids cannot be used here because they are opaque and may repeat. The internal
     * token also prevents a delayed completion from releasing a newer transaction.
     */
    void complete_live_video_mode(
      session_t &session,
      video::video_mode_applied_t completion
    ) {
      auto &state = session.live_video_mode_worker;
      bool accepted = false;
      {
        std::lock_guard lock(state.mutex);
        if (!state.stopping && !state.completion) {
          const auto matching_request = std::find_if(
            completion.requests.begin(),
            completion.requests.end(),
            [&](const video::video_mode_request_ref_t &request) {
              return state.serial_gate.accepts_completion(request.transaction_id);
            }
          );
          if (matching_request != completion.requests.end()) {
            state.completion = std::move(completion);
            accepted = true;
          }
        }
      }
      if (accepted) {
        state.wake.notify_one();
      } else {
        BOOST_LOG(warning) << "Ignoring a stale or duplicate live video-mode encoder completion."sv;
      }
    }

    /**
     * Serialize a live request through display apply, encoder creation and (on failure) display
     * rollback. Only an unstarted pending request may be coalesced.
     */
    void dispatch_live_video_mode(session_t &session, live_video_mode_request_t request) {
      auto &worker_state = session.live_video_mode_worker;
      std::optional<live_video_mode_request_t> superseded;
      std::optional<live_video_mode_request_t> rejected;
      bool start_worker = false;
      {
        std::lock_guard lock(worker_state.mutex);
        if (worker_state.stopping) {
          rejected = std::move(request);
        } else {
          const auto submission = worker_state.serial_gate.submit();
          request.change.transaction_id = submission.transaction_id;
          superseded = std::exchange(worker_state.pending, std::move(request));
          if (static_cast<bool>(superseded) !=
              static_cast<bool>(submission.superseded_transaction_id)) {
            BOOST_LOG(error) << "Live video-mode pending queue lost synchronization with its transaction gate."sv;
          }
        }
        if (!worker_state.stopping && !worker_state.thread.joinable()) {
          start_worker = true;
        }
      }

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

      if (!start_worker) {
        worker_state.wake.notify_one();
        return;
      }

      // Publish the joinable thread while holding the same lock used by stop/join. Otherwise a
      // concurrent disconnect could observe "no worker", return from join(), and let session_t
      // die just before this lambda captured it.
      std::lock_guard worker_lock(worker_state.mutex);
      if (worker_state.stopping || worker_state.thread.joinable()) {
        return;
      }
      worker_state.thread = std::thread([&session]() {
        auto &state = session.live_video_mode_worker;
        while (true) {
          std::optional<live_video_mode_request_t> job;
          std::uint64_t transaction_id = 0;
          {
            std::unique_lock lock(state.mutex);
            state.wake.wait(lock, [&]() {
              return state.stopping || state.serial_gate.can_begin();
            });
            if (state.stopping) {
              state.pending.reset();
              state.serial_gate.cancel_pending();
              return;
            }
            const auto next_transaction = state.serial_gate.begin_next();
            job = std::move(state.pending);
            state.pending.reset();
            if (!next_transaction || !job ||
                job->change.transaction_id != *next_transaction) {
              BOOST_LOG(error) << "Live video-mode worker found a transaction/payload mismatch."sv;
              if (next_transaction) {
                (void) state.serial_gate.finish(*next_transaction);
              }
              continue;
            }
            transaction_id = *next_transaction;
          }

          // The session can end while a display transition is in flight. Never resize a desktop
          // for a stream that is already gone. The acknowledgement is dropped with it; the client
          // is disconnected at that point and its own timeout covers the missing reply.
          if (job->mail->event<bool>(mail::shutdown)->peek()) {
            std::lock_guard lock(state.mutex);
            (void) state.serial_gate.finish(transaction_id);
            continue;
          }

          const auto host_sbs_rejects_geometry = [&]() {
            return session.requested_sbs_mode->load(std::memory_order_acquire) ==
                     ::video::SBS_AI &&
                   !models::host_sbs_v2_source_resolution_is_supported(
                static_cast<std::uint32_t>(job->change.width),
                static_cast<std::uint32_t>(job->change.height)
              );
          };
          // The quality request was validated on the control thread, but a Host SBS toggle can
          // race it while the serialized display worker is waiting. Check before touching the
          // desktop, then check again after the potentially slow display operation below.
          if (host_sbs_rejects_geometry()) {
            BOOST_LOG(warning)
              << "Rejecting live video mode "sv << job->change.width << 'x'
              << job->change.height
              << " because Host SBS V2 became active before the transaction committed."sv;
            reject_live_video_mode(
              job->mail,
              job->effective_mode,
              job->change.request_id,
              live_video_mode_ack_e::rejected_invalid
            );
            std::lock_guard lock(state.mutex);
            (void) state.serial_gate.finish(transaction_id);
            continue;
          }

          auto result = proc::live_video_mode_result_e::unchanged;
          if (proc::proc.live_video_mode_needs_display_change(
                job->change.width,
                job->change.height
              )) {
            result = proc::proc.apply_live_video_mode(
              job->change.width,
              job->change.height,
              job->change.encodingFramerate,
              job->launch_session_id
            );
          }
          const bool display_changed =
            result == proc::live_video_mode_result_e::applied;

          {
            std::lock_guard lock(state.mutex);
            if (state.stopping) {
              return;
            }
          }
          if ((result == proc::live_video_mode_result_e::applied ||
               result == proc::live_video_mode_result_e::unchanged) &&
              host_sbs_rejects_geometry()) {
            BOOST_LOG(warning)
              << "Host SBS V2 became active while live video mode "sv
              << job->change.width << 'x' << job->change.height
              << " was being applied; rejecting the transaction."sv;
            if (display_changed && job->effective_mode) {
              const auto previous = job->effective_mode->current();
              const auto rollback = proc::proc.apply_live_video_mode(
                previous.width,
                previous.height,
                previous.framerateX100 * 10,
                job->launch_session_id
              );
              if (rollback != proc::live_video_mode_result_e::applied &&
                  rollback != proc::live_video_mode_result_e::unchanged) {
                BOOST_LOG(error)
                  << "Could not restore the prior desktop after rejecting a raced Host SBS "sv
                     "quality transaction."sv;
              }
            }
            reject_live_video_mode(
              job->mail,
              job->effective_mode,
              job->change.request_id,
              live_video_mode_ack_e::rejected_invalid
            );
            std::lock_guard lock(state.mutex);
            (void) state.serial_gate.finish(transaction_id);
            continue;
          }
          switch (result) {
            case proc::live_video_mode_result_e::applied:
            case proc::live_video_mode_result_e::unchanged:
              if (job->mail->event<bool>(mail::shutdown)->peek()) {
                std::lock_guard lock(state.mutex);
                (void) state.serial_gate.finish(transaction_id);
                continue;
              }
              // The desktop is ready. Keep the transaction active until the encode loop proves
              // which mode exists and, on failure, the worker restores the prior desktop.
              commit_live_video_mode(*job);
              break;
            case proc::live_video_mode_result_e::needs_reconnect:
              BOOST_LOG(warning) << "type [IDX_SET_VIDEO_MODE]: "sv << job->change.width << 'x'
                                 << job->change.height << " cannot be applied without reconnecting for ["sv
                                 << job->client_name << "]; the stream keeps its current mode."sv;
              reject_live_video_mode(
                job->mail,
                job->effective_mode,
                job->change.request_id,
                live_video_mode_ack_status(result)
              );
              {
                std::lock_guard lock(state.mutex);
                (void) state.serial_gate.finish(transaction_id);
              }
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
              {
                std::lock_guard lock(state.mutex);
                (void) state.serial_gate.finish(transaction_id);
              }
              break;
          }

          if (result != proc::live_video_mode_result_e::applied &&
              result != proc::live_video_mode_result_e::unchanged) {
            continue;
          }

          std::optional<video::video_mode_applied_t> completion;
          {
            std::unique_lock lock(state.mutex);
            state.wake.wait(lock, [&]() {
              return state.stopping || state.completion.has_value();
            });
            if (state.stopping) {
              return;
            }
            completion = std::move(state.completion);
            state.completion.reset();
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

          if (!completion->applied && display_changed) {
            BOOST_LOG(warning) << "Restoring the last encodable virtual desktop mode "
                               << completion->desktop_mode.width << 'x'
                               << completion->desktop_mode.height << " after the encoder rejected "
                               << job->change.width << 'x' << job->change.height << '.';
            const auto rollback_result = proc::proc.apply_live_video_mode(
              completion->desktop_mode.width,
              completion->desktop_mode.height,
              completion->desktop_mode.framerate_millihz,
              job->launch_session_id
            );
            if (rollback_result != proc::live_video_mode_result_e::applied &&
                rollback_result != proc::live_video_mode_result_e::unchanged) {
              BOOST_LOG(error) << "The virtual desktop could not be restored after a live encoder "
                                  "rollback; a later request may require reconnecting."sv;
            }
          }

          // Re-arm diagnostics only after the effective pacing pair is known. Resetting at request
          // enqueue lets the old encoder consume the latch before the new encoder is proven.
          job->video->pacing_plan_logged.store(false, std::memory_order_release);
          job->video->idr_pacing_plan_logged.store(false, std::memory_order_release);
          queue_live_video_mode_ack(
            job->mail,
            live_video_mode_ack_t {
              job->change.request_id,
              completion->applied ?
                live_video_mode_ack_e::applied :
                live_video_mode_ack_e::failed,
              completion->mode.width,
              completion->mode.height,
              completion->mode.framerateX100,
              completion->mode.bitrate,
            }
          );

          {
            std::lock_guard lock(state.mutex);
            if (!state.serial_gate.finish(transaction_id)) {
              BOOST_LOG(error) << "Live video-mode transaction gate rejected its active completion."sv;
            }
          }
        }
      });
      worker_state.wake.notify_one();
    }

    void stop_live_video_mode_worker(session_t &session) {
      auto &worker = session.live_video_mode_worker;
      {
        std::lock_guard lock(worker.mutex);
        worker.stopping = true;
        worker.pending.reset();
        worker.serial_gate.cancel_pending();
      }
      worker.wake.notify_all();
    }

    void join_live_video_mode_worker(session_t &session) {
      stop_live_video_mode_worker(session);
      auto &thread = session.live_video_mode_worker.thread;
      if (thread.joinable()) {
        BOOST_LOG(debug) << "Waiting for live video-mode worker to end..."sv;
        thread.join();
      }
    }
  }  // namespace

  /**
   * First part of cipher must be struct of type control_encrypted_t
   *
   * returns empty string_view on failure
   * returns string_view pointing to payload data
   */
  template<std::size_t max_payload_size>
  static inline std::string_view encode_control(session_t *session, const std::string_view &plaintext, std::array<std::uint8_t, max_payload_size> &tagged_cipher) {
    static_assert(
      max_payload_size >= sizeof(control_encrypted_t) + sizeof(crypto::cipher::tag_size),
      "max_payload_size >= sizeof(control_encrypted_t) + sizeof(crypto::cipher::tag_size)"
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

    auto bytes = session->control.cipher.encrypt(plaintext, packet->payload(), &iv);
    if (bytes <= 0) {
      BOOST_LOG(error) << "Couldn't encrypt control data"sv;
      return {};
    }

    std::uint16_t packet_length = bytes + crypto::cipher::tag_size + sizeof(control_encrypted_t::seq);

    packet->encryptedHeaderType = util::endian::little(0x0001);
    packet->length = util::endian::little(packet_length);
    packet->seq = util::endian::little(seq);

    return std::string_view {(char *) tagged_cipher.data(), packet_length + sizeof(control_encrypted_t) - sizeof(control_encrypted_t::seq)};
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

  namespace fec {
    using rs_t = util::safe_ptr<reed_solomon, [](reed_solomon *rs) {
      reed_solomon_release(rs);
    }>;

    struct fec_t {
      size_t data_shards;
      size_t nr_shards;
      size_t percentage;

      size_t blocksize;
      size_t prefixsize;
      util::buffer_t<char> shards;
      util::buffer_t<char> headers;
      util::buffer_t<uint8_t *> shards_p;

      std::vector<platf::buffer_descriptor_t> payload_buffers;

      char *data(size_t el) {
        return (char *) shards_p[el];
      }

      char *prefix(size_t el) {
        return prefixsize ? &headers[el * prefixsize] : nullptr;
      }

      size_t size() const {
        return nr_shards;
      }
    };

    static fec_t encode(const std::string_view &payload, size_t blocksize, size_t fecpercentage, size_t minparityshards, size_t prefixsize) {
      if (payload.empty() || blocksize == 0 || fecpercentage > 255 || minparityshards > MIN_REQUIRED_FEC_PACKETS_MAX) {
        throw std::invalid_argument("Invalid Reed-Solomon FEC parameters");
      }

      auto payload_size = payload.size();

      auto pad = payload_size % blocksize != 0;

      auto aligned_data_shards = payload_size / blocksize;
      auto data_shards = aligned_data_shards + (pad ? 1 : 0);
      auto parity_shards = (data_shards * fecpercentage + 99) / 100;

      // increase the FEC percentage for this frame if the parity shard minimum is not met
      if (parity_shards < minparityshards && fecpercentage != 0) {
        parity_shards = minparityshards;
        fecpercentage = (100 * parity_shards) / data_shards;
      }

      if (fecpercentage != 0 && (data_shards > DATA_SHARDS_MAX || parity_shards > DATA_SHARDS_MAX || data_shards + parity_shards > DATA_SHARDS_MAX)) {
        throw std::invalid_argument("Reed-Solomon shard count exceeds protocol limit");
      }

      auto nr_shards = data_shards + parity_shards;

      // If we need to store a zero-padded data shard, allocate that first to
      // to keep the shards in order and reduce buffer fragmentation
      auto parity_shard_offset = pad ? 1 : 0;
      util::buffer_t<char> shards {(parity_shard_offset + parity_shards) * blocksize};
      util::buffer_t<uint8_t *> shards_p {nr_shards};
      std::vector<platf::buffer_descriptor_t> payload_buffers;
      payload_buffers.reserve(2);

      // Point into the payload buffer for all except the final padded data shard
      auto next = std::begin(payload);
      for (auto x = 0; x < aligned_data_shards; ++x) {
        shards_p[x] = (uint8_t *) next;
        next += blocksize;
      }
      payload_buffers.emplace_back(std::begin(payload), aligned_data_shards * blocksize);

      // If the last data shard needs to be zero-padded, we must use the shards buffer
      if (pad) {
        shards_p[aligned_data_shards] = (uint8_t *) &shards[0];

        // GCC doesn't figure out that std::copy_n() can be replaced with memcpy() here
        // and ends up compiling a horribly slow element-by-element copy loop, so we
        // help it by using memcpy()/memset() directly.
        auto copy_len = std::min<size_t>(blocksize, std::end(payload) - next);
        std::memcpy(shards_p[aligned_data_shards], next, copy_len);
        if (copy_len < blocksize) {
          // Zero any additional space after the end of the payload
          std::memset(shards_p[aligned_data_shards] + copy_len, 0, blocksize - copy_len);
        }
      }

      // Add a payload buffer describing the shard buffer
      payload_buffers.emplace_back(std::begin(shards), shards.size());

      if (fecpercentage != 0) {
        // Point into our allocated buffer for the parity shards
        for (auto x = 0; x < parity_shards; ++x) {
          shards_p[data_shards + x] = (uint8_t *) &shards[(parity_shard_offset + x) * blocksize];
        }

        // packets = parity_shards + data_shards
        rs_t rs {reed_solomon_new(data_shards, parity_shards)};
        if (!rs) {
          BOOST_LOG(error) << "Failed to allocate video Reed-Solomon context; sending this FEC block without parity"sv;
          nr_shards = data_shards;
          fecpercentage = 0;
        } else if (reed_solomon_encode(rs.get(), shards_p.begin(), nr_shards, blocksize) != 0) {
          BOOST_LOG(error) << "Failed to encode video Reed-Solomon shards; sending this FEC block without parity"sv;
          nr_shards = data_shards;
          fecpercentage = 0;
        }
      }

      return {
        data_shards,
        nr_shards,
        fecpercentage,
        blocksize,
        prefixsize,
        std::move(shards),
        util::buffer_t<char> {nr_shards * prefixsize},
        std::move(shards_p),
        std::move(payload_buffers),
      };
    }
  }  // namespace fec

  /**
   * @brief Combines two buffers and inserts new buffers at each slice boundary of the result.
   * @param insert_size The number of bytes to insert.
   * @param slice_size The number of bytes between insertions.
   * @param data1 The first data buffer.
   * @param data2 The second data buffer.
   */
  std::vector<uint8_t> concat_and_insert(uint64_t insert_size, uint64_t slice_size, const std::string_view &data1, const std::string_view &data2) {
    constexpr auto size_max = std::numeric_limits<std::size_t>::max();
    if (slice_size == 0 || slice_size > size_max || insert_size > size_max || data1.size() > size_max - data2.size()) {
      return {};
    }

    const auto slice_bytes = static_cast<std::size_t>(slice_size);
    const auto insert_bytes = static_cast<std::size_t>(insert_size);
    const auto data_size = data1.size() + data2.size();
    const auto elements = data_size / slice_bytes + (data_size % slice_bytes != 0);
    if (elements != 0 && insert_bytes > (size_max - data_size) / elements) {
      return {};
    }

    std::vector<uint8_t> result;
    result.resize(data_size + elements * insert_bytes);

    std::size_t source_offset = 0;
    std::size_t output_offset = 0;
    while (source_offset < data_size) {
      // resize() zero-initializes the inserted prefix.
      output_offset += insert_bytes;
      const auto chunk_size = std::min(slice_bytes, data_size - source_offset);

      std::size_t first_size = 0;
      if (source_offset < data1.size()) {
        first_size = std::min(chunk_size, data1.size() - source_offset);
        std::memcpy(
          result.data() + output_offset,
          data1.data() + source_offset,
          first_size
        );
      }

      const auto second_size = chunk_size - first_size;
      if (second_size != 0) {
        const auto second_offset = source_offset + first_size - data1.size();
        std::memcpy(
          result.data() + output_offset + first_size,
          data2.data() + second_offset,
          second_size
        );
      }

      source_offset += chunk_size;
      output_offset += chunk_size;
    }

    return result;
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

    std::string payload;
    if (msg.type == platf::gamepad_feedback_e::rumble) {
      control_rumble_t plaintext;
      plaintext.header.type = control_packet::rumble;
      plaintext.header.payloadLength = sizeof(plaintext) - sizeof(control_header_v2);

      auto &data = msg.data.rumble;

      plaintext.useless = 0xC0FFEE;
      plaintext.id = util::endian::little(msg.id);
      plaintext.lowfreq = util::endian::little(data.lowfreq);
      plaintext.highfreq = util::endian::little(data.highfreq);

      std::array<std::uint8_t, sizeof(control_encrypted_t) + crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) + crypto::cipher::tag_size>
        encrypted_payload;

      payload = encode_control(session, util::view(plaintext), encrypted_payload);
    } else if (msg.type == platf::gamepad_feedback_e::rumble_triggers) {
      control_rumble_triggers_t plaintext;
      plaintext.header.type = control_packet::rumble_triggers;
      plaintext.header.payloadLength = sizeof(plaintext) - sizeof(control_header_v2);

      auto &data = msg.data.rumble_triggers;

      plaintext.id = util::endian::little(msg.id);
      plaintext.left = util::endian::little(data.left_trigger);
      plaintext.right = util::endian::little(data.right_trigger);

      std::array<std::uint8_t, sizeof(control_encrypted_t) + crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) + crypto::cipher::tag_size>
        encrypted_payload;

      payload = encode_control(session, util::view(plaintext), encrypted_payload);
    } else if (msg.type == platf::gamepad_feedback_e::set_motion_event_state) {
      control_set_motion_event_t plaintext;
      plaintext.header.type = control_packet::set_motion_event;
      plaintext.header.payloadLength = sizeof(plaintext) - sizeof(control_header_v2);

      auto &data = msg.data.motion_event_state;

      plaintext.id = util::endian::little(msg.id);
      plaintext.reportrate = util::endian::little(data.report_rate);
      plaintext.type = data.motion_type;

      std::array<std::uint8_t, sizeof(control_encrypted_t) + crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) + crypto::cipher::tag_size>
        encrypted_payload;

      payload = encode_control(session, util::view(plaintext), encrypted_payload);
    } else if (msg.type == platf::gamepad_feedback_e::set_rgb_led) {
      control_set_rgb_led_t plaintext;
      plaintext.header.type = control_packet::set_rgb_led_feedback;
      plaintext.header.payloadLength = sizeof(plaintext) - sizeof(control_header_v2);

      auto &data = msg.data.rgb_led;

      plaintext.id = util::endian::little(msg.id);
      plaintext.r = data.r;
      plaintext.g = data.g;
      plaintext.b = data.b;

      std::array<std::uint8_t, sizeof(control_encrypted_t) + crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) + crypto::cipher::tag_size>
        encrypted_payload;

      payload = encode_control(session, util::view(plaintext), encrypted_payload);
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

      std::array<std::uint8_t, sizeof(control_encrypted_t) + crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) + crypto::cipher::tag_size>
        encrypted_payload;

      payload = encode_control(session, util::view(plaintext), encrypted_payload);
    } else {
      BOOST_LOG(error) << "Unknown gamepad feedback message type"sv;
      return -1;
    }

    if (session->broadcast_ref->control_server.send(payload, session->control.peer)) {
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

    std::array<std::uint8_t, sizeof(control_encrypted_t) + crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) + crypto::cipher::tag_size>
      encrypted_payload;

    auto payload = encode_control(session, util::view(plaintext), encrypted_payload);
    if (session->broadcast_ref->control_server.send(payload, session->control.peer)) {
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

    std::array<std::uint8_t, sizeof(control_encrypted_t) + crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) + crypto::cipher::tag_size>
      encrypted_payload;

    auto payload = encode_control(session, util::view(plaintext), encrypted_payload);
    if (session->broadcast_ref->control_server.send(payload, session->control.peer)) {
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

    control_live_video_mode_ack_t plaintext {};
    plaintext.header.type = control_packet::live_video_mode_ack;
    plaintext.header.payloadLength = sizeof(control_live_video_mode_ack_t) - sizeof(control_header_v2);
    if (!encode_live_video_mode_ack_payload(ack, plaintext.body)) {
      BOOST_LOG(error) << "Refusing to send a live video-mode acknowledgement that does not fit the wire format"sv;
      return -1;
    }

    std::array<std::uint8_t, sizeof(control_encrypted_t) + crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) + crypto::cipher::tag_size>
      encrypted_payload;

    auto payload = encode_control(session, util::view(plaintext), encrypted_payload);
    if (payload.empty() || session->broadcast_ref->control_server.send(payload, session->control.peer)) {
      TUPLE_2D(port, addr, platf::from_sockaddr_ex((sockaddr *) &session->control.peer->address.address));
      BOOST_LOG(warning) << "Couldn't send live video-mode acknowledgement to ["sv << addr << ':' << port << ']';
      return -1;
    }

    BOOST_LOG(debug) << "Sent live video-mode ack: request "sv << ack.request_id
                     << " status="sv << static_cast<int>(ack.status) << ", in effect "sv
                     << ack.applied_width << 'x' << ack.applied_height
                     << '@' << (ack.applied_framerate_x100 / 100.0) << "Hz "sv
                     << ack.applied_bitrate_kbps << "kbps"sv;
    return 0;
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

    std::array<
      std::uint8_t,
      sizeof(control_encrypted_t) +
        crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) +
        crypto::cipher::tag_size>
      encrypted_payload;
    auto payload = encode_control(session, util::view(plaintext), encrypted_payload);
    if (payload.empty()) {
      return -1;
    }

    const int result = reliable ?
                         session->broadcast_ref->control_server.send(
                           payload,
                           session->control.peer
                         ) :
                         session->broadcast_ref->control_server.send_unreliable_sequenced(
                           payload,
                           session->control.peer
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

    std::array<std::uint8_t, sizeof(control_encrypted_t) + crypto::cipher::round_to_pkcs7_padded(sizeof(plaintext)) + crypto::cipher::tag_size>
      encrypted_payload;
    auto payload = encode_control(&session, util::view(plaintext), encrypted_payload);
    if (payload.empty() || server.send(payload, session.control.peer)) {
      TUPLE_2D(port, addr, platf::from_sockaddr_ex((sockaddr *) &session.control.peer->address.address));
      BOOST_LOG(warning) << "Couldn't send termination code to ["sv << addr << ':' << port << ']';
    }
  }

  void controlBroadcastThread(control_server_t *server) {
    server->map(control_packet::periodic_ping, [](session_t *, const std::string_view &) {
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
    });

    server->map(control_packet::invalidate_ref_frames, 2 * sizeof(std::int64_t), [&](session_t *session, const std::string_view &payload) {
      std::int64_t frames[2];
      std::memcpy(frames, payload.data(), sizeof(frames));
      auto firstFrame = util::endian::little(frames[0]);
      auto lastFrame = util::endian::little(frames[1]);

      session->video->invalidate_ref_frames_events->raise(std::make_pair(firstFrame, lastFrame));
    });

    server->map(control_packet::set_sbs_mode, sizeof(std::uint8_t), [server](session_t *session, const std::string_view &payload) {
      // Host-side SBS mode requested by the client (Apollo protocol extension).
      // Must match SBS_MODE_* in the client's moonlight-common-c Limelight.h.
      auto mode = *(uint8_t *) payload.data();
      if (mode > ::video::SBS_AI) {
        BOOST_LOG(warning) << "type [IDX_SET_SBS_MODE]: unknown mode "sv << (int) mode
                           << " from ["sv << session::client_name(*session) << "]; ignored"sv;
        return;
      }
      if (mode == ::video::SBS_AI) {
        const auto effective_mode = session->config.monitor.effective_mode ?
                                      session->config.monitor.effective_mode->current() :
                                      ::video::effective_video_mode_t {
                                        session->config.monitor.width,
                                        session->config.monitor.height,
                                        session->config.monitor.framerateX100,
                                        session->config.monitor.bitrate,
                                      };
        const auto host_sbs_rejection =
          effective_mode.width > 0 && effective_mode.height > 0 ?
            models::host_sbs_v2_source_resolution_rejection_reason(
              static_cast<std::uint32_t>(effective_mode.width),
              static_cast<std::uint32_t>(effective_mode.height)
            ) :
            std::string_view {"source extent is empty"};
        if (!host_sbs_rejection.empty()) {
          const auto fitted = effective_mode.width > 0 && effective_mode.height > 0 ?
                                models::fit_host_sbs_v2_depth_tensor_shape(
                                  static_cast<std::uint32_t>(effective_mode.width),
                                  static_cast<std::uint32_t>(effective_mode.height)
                                ) :
                                models::depth_tensor_shape_t {};
          BOOST_LOG(warning) << "Rejecting Host SBS V2 toggle at "sv
                             << effective_mode.width << 'x' << effective_mode.height
                             << " for ["sv << session::client_name(*session)
                             << "]: "sv << host_sbs_rejection
                             << "; fitted depth tensor "sv << fitted.width << 'x'
                             << fitted.height << '.';
          session->mail->event<int>(mail::sbs_depth_status)->raise(0);
          return;
        }
      }
      std::string_view mode_name = mode == ::video::SBS_OFF ? "OFF"sv : "AI"sv;
      BOOST_LOG(info) << "type [IDX_SET_SBS_MODE]: client requested host SBS "sv << mode_name
                      << " ("sv << (int) mode << ") for ["sv
                      << session::client_name(*session) << ']';
      session->requested_sbs_mode->store((int) mode, std::memory_order_release);
      if (mode != ::video::SBS_AI && session->video->sbs_debug_dump_pending) {
        session->video->sbs_debug_dump_pending->store(false, std::memory_order_release);
      }

      // Turning SBS off tears down the depth estimator with no replacement, so mark the depth
      // engine idle here (display_vram only ever sets loading/ready). This clears any "loading"
      // indicator on the client if the user switches to Normal mid-spin-up.
      if (mode == ::video::SBS_OFF) {
        session->mail->event<int>(mail::sbs_depth_status)->raise(0);
      }

      // Hand the requested mode to this session's video pipeline. capture_async consumes it and
      // rebuilds the encode device at the new resolution (W x H for OFF, 2W x H for AI). The
      // pinned authenticated live depth model is prepared once during host startup.
      // Log the recalculated pacing plan after the output dimensions change.
      session->video->pacing_plan_logged.store(false, std::memory_order_release);
      session->video->idr_pacing_plan_logged.store(false, std::memory_order_release);
      session->mail->event<int>(mail::sbs_mode)->raise((int) mode);
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
        if (payload.size() != SBS_TELEMETRY_SUBSCRIPTION_PAYLOAD_SIZE &&
            payload.size() >= 4) {
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

    server->map(control_packet::set_video_mode, sizeof(control_set_video_mode_t), [](session_t *session, const std::string_view &payload) {
      // Live resolution/frame-rate/bitrate change requested by the client (Apollo protocol
      // extension). The encode session is rebuilt in place exactly as the 0x3003 SBS toggle
      // does; the capture session, the virtual display and the RTSP session all survive.
      control_set_video_mode_t request {};
      std::memcpy(&request, payload.data(), sizeof(request));
      const int width = util::endian::little(request.width);
      const int height = util::endian::little(request.height);
      const int framerate_x100 = util::endian::little(request.framerate_x100);
      // Opaque correlation token. It is never validated, never ordered on, and never acted upon;
      // it is only carried through to the acknowledgement.
      const int request_id = util::endian::little(request.request_id);
      const std::int64_t requested_bitrate_kbps = util::endian::little(request.bitrate_kbps);

      // A malformed request must never reach the encoder: capture_async cannot recover from a
      // failed non-SBS encoder creation and would tear the whole session down. Every rejection
      // still answers the client, reporting the mode that stays in effect, so it can drop the
      // request and resynchronize instead of waiting out a timeout.
      const auto reject = [&](std::string_view reason) {
        BOOST_LOG(warning) << "type [IDX_SET_VIDEO_MODE]: ignoring "sv << width << 'x' << height
                           << '@' << (framerate_x100 / 100.0) << " from ["sv
                           << session::client_name(*session) << "]: "sv << reason;
        reject_live_video_mode(
          session->mail,
          session->config.monitor.effective_mode,
          request_id,
          live_video_mode_ack_e::rejected_invalid
        );
      };
      if (!is_valid_live_video_mode_dimension(width) || !is_valid_live_video_mode_dimension(height)) {
        reject("resolution must be even and within 2..16384"sv);
        return;
      }
      const auto host_sbs_rejection =
        models::host_sbs_v2_source_resolution_rejection_reason(
          static_cast<std::uint32_t>(width),
          static_cast<std::uint32_t>(height)
        );
      if (session->requested_sbs_mode->load(std::memory_order_acquire) ==
            ::video::SBS_AI &&
          !host_sbs_rejection.empty()) {
        reject(std::format("Host SBS V2 rejects this resolution: {}", host_sbs_rejection));
        return;
      }
      if (!is_valid_live_video_mode_framerate_x100(framerate_x100)) {
        reject("frame rate must be within 1..1000 Hz"sv);
        return;
      }
      // Encoder width limits are deliberately not checked here: the encode loop owns the codec
      // capability snapshot and clamps an oversized request the same way it already caps an SBS
      // packed width. Clamping there cannot fail, whereas rejecting a creatable mode here would.

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
      const std::int64_t audio_bitrate_kbps =
        (session->config.audio.flags[audio::config_t::HIGH_QUALITY] ? 256 : 96) *
        session->config.audio.channels;
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
      change.request_id = request_id;

      BOOST_LOG(info) << "type [IDX_SET_VIDEO_MODE]: client requested "sv << width << 'x' << height
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
      if (!sbs_debug_dump_request_allowed(
            config::sunshine.diagnostics_enabled,
            session->requested_sbs_mode->load(std::memory_order_acquire),
            (bool) session->video->sbs_debug_dump_pending
          )) {
        BOOST_LOG(warning) << "Ignoring Dump 3D request outside Host SBS AI for ["sv
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

      std::vector<uint8_t> plaintext;
      if (cipher.decrypt(tagged_cipher, plaintext, &iv)) {
        // something went wrong :(

        BOOST_LOG(error) << "Failed to verify tag"sv;

        session::stop(*session);
        return;
      }

      if (plaintext.size() < CONTROL_HEADER_V2_SIZE) {
        BOOST_LOG(warning) << "Dropping encrypted control message with a runt plaintext header"sv;
        return;
      }

      std::uint16_t type;
      std::uint16_t declared_payload_size;
      std::memcpy(&type, plaintext.data(), sizeof(type));
      std::memcpy(&declared_payload_size, plaintext.data() + sizeof(type), sizeof(declared_payload_size));
      type = util::endian::little(type);
      declared_payload_size = util::endian::little(declared_payload_size);
      if (!is_valid_decrypted_control_payload(plaintext.size(), declared_payload_size)) {
        BOOST_LOG(warning) << "Dropping encrypted control message with mismatched inner length: declared "sv
                           << declared_payload_size << " bytes in a "sv << plaintext.size() << "-byte plaintext"sv;
        return;
      }

      std::string_view next_payload {
        (char *) plaintext.data() + CONTROL_HEADER_V2_SIZE,
        declared_payload_size
      };

      if (type == control_packet::encrypted) {
        BOOST_LOG(error) << "Bad packet type [IDX_ENCRYPTED] found"sv;
        session::stop(*session);
        return;
      }

      // Input data is already authenticated by the outer control-v2 envelope.
      if (type == control_packet::input) {
        plaintext.erase(std::begin(plaintext), std::begin(plaintext) + CONTROL_HEADER_V2_SIZE);
        input::passthrough(session->input, std::move(plaintext), session::permissions(*session));
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
      auto iteration_timeout = 150ms;
      {
        auto slot_lock = server->_session.lock();
        auto *session = server->_session->session;
        if (session && !shutdown_event->peek() && !broadcast_shutdown_event->peek()) {
          if (session->control.sbs_telemetry_subscription->enabled()) {
            iteration_timeout =
              session->control.sbs_telemetry_subscription->focused() ? 50ms : 100ms;
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

            auto &hdr_queue = session->control.hdr_queue;
            while (server->_session->peer && hdr_queue->peek()) {
              auto hdr_info = hdr_queue->pop();
              send_hdr_mode(session, std::move(hdr_info));
            }

            // Drain depth-engine state to its latest phase. The event intentionally coalesces a
            // fast loading->ready transition so cached engines do not flash the client UI.
            auto depth_status_event = session->mail->event<int>(mail::sbs_depth_status);
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

            // Release the matching serialized live-mode transaction. The worker sends the plain
            // acknowledgement only after any failed-encoder desktop rollback has finished.
            auto video_mode_applied_queue = session->mail->queue<video::video_mode_applied_t>(mail::video_mode_applied);
            while (video_mode_applied_queue->peek()) {
              if (auto applied = video_mode_applied_queue->pop(0ms)) {
                complete_live_video_mode(*session, std::move(*applied));
              }
            }

            // Answer every live video-mode request (0x3007) exactly once. Unlike the depth phase
            // these must not coalesce, so they arrive on a real queue: a client that sends two
            // requests is owed two replies. The detached virtual-display worker and the encode
            // loop only enqueue outcomes, leaving the peer, the cipher and ENet to this thread.
            auto &ack_queue = session->control.live_video_mode_ack_queue;
            while (server->_session->peer && ack_queue->peek()) {
              if (auto ack = ack_queue->pop(0ms)) {
                send_live_video_mode_ack(session, *ack);
              }
            }
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
      session->controlEnd.raise(true);
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

  void videoBroadcastThread(udp::socket &sock) {
    auto shutdown_event = mail::man->event<bool>(mail::broadcast_shutdown);
    auto packets = mail::man->queue<video::packet_t>(
      mail::video_packets,
      video::ENCODED_PACKET_QUEUE_LIMIT
    );
    auto video_epoch = std::chrono::steady_clock::now();

    // Video traffic is sent on this thread
    platf::adjust_thread_priority(platf::thread_priority_e::high);

    logging::min_max_avg_periodic_logger<double> frame_processing_latency_logger(info, "Frame processing latency", "ms");

    logging::time_delta_periodic_logger frame_send_batch_latency_logger(info, "Network: each send_batch() latency");
    logging::time_delta_periodic_logger frame_fec_latency_logger(info, "Network: each FEC block latency");
    logging::time_delta_periodic_logger frame_network_latency_logger(info, "Network: frame's overall network latency");

    crypto::aes_t iv(12);

    auto timer = platf::create_high_precision_timer();
    if (!timer || !*timer) {
      BOOST_LOG(error) << "Failed to create timer, aborting video broadcast thread";
      return;
    }

    auto ratecontrol_next_frame_start = std::chrono::steady_clock::now();

    while (auto packet = packets->pop()) {
      if (shutdown_event->peek()) {
        break;
      }

      frame_network_latency_logger.first_point_now();

      auto channel = std::static_pointer_cast<video_channel_t>(packet->channel_data);
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
      };

      const auto encoded_packet_age = std::chrono::steady_clock::now() - packet->encoded_timestamp;
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

      auto lowseq = channel->lowseq;

      std::string_view payload {(char *) packet->data(), packet->data_size()};

      video_short_frame_header_t frame_header = {};
      frame_header.headerType = 0x01;  // Short header type
      frame_header.frameType = packet->is_idr()                     ? 2 :
                               packet->after_ref_frame_invalidation ? 5 :
                                                                      1;
      frame_header.lastPayloadLen = (payload.size() + sizeof(frame_header)) % (channel->packet_size - sizeof(NV_VIDEO_PACKET));
      if (frame_header.lastPayloadLen == 0) {
        frame_header.lastPayloadLen = channel->packet_size - sizeof(NV_VIDEO_PACKET);
      }

      const auto processing_timestamp = video::detail::select_processing_timestamp(
        packet->content_timestamp,
        packet->frame_timestamp
      );
      if (processing_timestamp) {
        auto duration_to_latency = [](const std::chrono::steady_clock::duration &duration) {
          const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
          return (uint16_t) std::clamp<decltype(duration_us)>((duration_us + 50) / 100, 0, std::numeric_limits<uint16_t>::max());
        };

        uint16_t latency = duration_to_latency(std::chrono::steady_clock::now() - *processing_timestamp);
        frame_header.frame_processing_latency = latency;
        frame_processing_latency_logger.collect_and_log(latency / 10.);
      } else {
        frame_header.frame_processing_latency = 0;
      }

      auto fecPercentage = config::stream.fec_percentage;

      // Insert space for packet headers
      auto blocksize = channel->packet_size + MAX_RTP_HEADER_SIZE;
      auto payload_blocksize = blocksize - sizeof(video_packet_raw_t);
      auto payload_new = concat_and_insert(sizeof(video_packet_raw_t), payload_blocksize, std::string_view {(char *) &frame_header, sizeof(frame_header)}, payload);
      if (payload_new.empty()) {
        request_recovery_idr("video payload packetization failed"sv);
        continue;
      }

      payload = std::string_view {(char *) payload_new.data(), payload_new.size()};

      // There are 2 bits for FEC block count for a maximum of 4 FEC blocks
      constexpr auto MAX_FEC_BLOCKS = 4;

      // The max number of data shards per block is found by solving this system of equations for D:
      // D = 255 - P
      // P = D * F
      // which results in the solution:
      // D = 255 / (1 + F)
      // multiplied by 100 since F is the percentage as an integer:
      // D = (255 * 100) / (100 + F)
      auto max_data_shards_per_fec_block = (DATA_SHARDS_MAX * 100) / (100 + fecPercentage);

      // Compute the number of FEC blocks needed for this frame using the block size and max shards
      auto max_data_per_fec_block = max_data_shards_per_fec_block * blocksize;
      auto fec_blocks_needed = (payload.size() + (max_data_per_fec_block - 1)) / max_data_per_fec_block;

      // If the number of FEC blocks needed exceeds the protocol limit, turn off FEC for this frame.
      // For normal FEC percentages, this should only happen for enormous frames (over 800 packets at 20%).
      if (fec_blocks_needed > MAX_FEC_BLOCKS) {
        BOOST_LOG(warning) << "Skipping FEC for abnormally large encoded frame (needed "sv << fec_blocks_needed << " FEC blocks)"sv;
        fecPercentage = 0;
        fec_blocks_needed = MAX_FEC_BLOCKS;
      }

      std::array<std::string_view, MAX_FEC_BLOCKS> fec_blocks;
      decltype(fec_blocks)::iterator
        fec_blocks_begin = std::begin(fec_blocks),
        fec_blocks_end = std::begin(fec_blocks) + fec_blocks_needed;

      // Align individual FEC blocks to blocksize
      auto unaligned_size = payload.size() / fec_blocks_needed;
      auto aligned_size = ((unaligned_size + (blocksize - 1)) / blocksize) * blocksize;

      // Split the data into aligned FEC blocks
      size_t max_packets_per_block = 0;
      bool fec_block_sizes_valid = true;
      for (int x = 0; x < fec_blocks_needed; ++x) {
        if (x == fec_blocks_needed - 1) {
          // The last block must extend to the end of the payload
          fec_blocks[x] = payload.substr(x * aligned_size);
        } else {
          // Earlier blocks just extend to the next block offset
          fec_blocks[x] = payload.substr(x * aligned_size, aligned_size);
        }

        const auto packets = fec_packet_count(fec_blocks[x].size(), blocksize);
        max_packets_per_block = std::max(max_packets_per_block, packets);
        fec_block_sizes_valid = fec_block_sizes_valid && is_valid_fec_block_size(fec_blocks[x].size(), blocksize);
      }

      // The packet index and data-shard count are 10-bit fields. Inspect each
      // actual split because the final partial block can be one shard larger.
      if (!fec_block_sizes_valid) {
        BOOST_LOG(error) << "Dropping encoded frame that exceeds the 10-bit FEC packet index (needed "sv << max_packets_per_block << " packets per block)"sv;
        request_recovery_idr("encoded frame exceeded the FEC packet-index limit"sv);
        continue;
      }

      try {
        const auto wire_packet_bytes = blocksize + sizeof(video_packet_enc_prefix_t);
        std::size_t estimated_data_packets = 0;
        std::size_t estimated_wire_packets = 0;
        for (auto it = fec_blocks_begin; it != fec_blocks_end; ++it) {
          estimated_data_packets += fec_packet_count(it->size(), blocksize);
          estimated_wire_packets += video_fec_shard_count(
            it->size(),
            blocksize,
            fecPercentage,
            channel->min_required_fec_packets
          );
        }
        const auto pacing_plan = make_video_pacing_plan(
          effective_pacing.bitrate,
          effective_pacing.framerate_millihz,
          estimated_data_packets,
          estimated_wire_packets,
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
          std::size_t {64},
          send_batch_size,
          pacing_plan.packets_per_quantum,
        });

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

        // Don't overlap this frame with the final paced packet of the previous frame.
        auto ratecontrol_frame_start = std::max(ratecontrol_next_frame_start, std::chrono::steady_clock::now());

        size_t ratecontrol_frame_packets_sent = 0;

        auto blockIndex = 0;
        std::for_each(fec_blocks_begin, fec_blocks_end, [&](std::string_view &current_payload) {
          auto packets = (current_payload.size() + (blocksize - 1)) / blocksize;
          const auto block_lowseq = lowseq;

          for (int x = 0; x < packets; ++x) {
            auto *inspect = (video_packet_raw_t *) &current_payload[x * blocksize];

            inspect->packet.frameIndex = packet->frame_index();
            inspect->packet.streamPacketIndex =
              (block_lowseq + static_cast<std::uint32_t>(x)) << 8;

            // Match multiFecFlags with Artemis
            inspect->packet.multiFecFlags = 0x10;
            inspect->packet.multiFecBlocks = (blockIndex << 4) | ((fec_blocks_needed - 1) << 6);

            inspect->packet.flags = FLAG_CONTAINS_PIC_DATA;
            if (x == 0) {
              inspect->packet.flags |= FLAG_SOF;
            }
            if (x == packets - 1) {
              inspect->packet.flags |= FLAG_EOF;
            }
          }

          frame_fec_latency_logger.first_point_now();
          auto shards = fec::encode(current_payload, blocksize, fecPercentage, channel->min_required_fec_packets, sizeof(video_packet_enc_prefix_t));
          frame_fec_latency_logger.second_point_now_and_log();

          // Reserve this whole block's sequence range before the first send. If a send throws
          // after transmitting only part of the block, the recovery IDR must start after the
          // abandoned range rather than reusing sequence numbers the client already observed.
          lowseq += static_cast<std::uint32_t>(shards.size());
          channel->lowseq = lowseq;

          auto peer_address = channel->peer.address();
          auto batch_info = platf::batched_send_info_t {
            shards.headers.begin(),
            shards.prefixsize,
            shards.payload_buffers,
            shards.blocksize,
            0,
            0,
            (uintptr_t) sock.native_handle(),
            peer_address,
            channel->peer.port(),
            channel->local_address,
          };

          size_t next_shard_to_send = 0;

          // RTP video timestamps use a 90 KHz clock and the frame_timestamp from when the frame was captured
          // When a timestamp isn't available (duplicate frames), the timestamp from rate control is used instead.
          if (!packet->frame_timestamp) {
            packet->frame_timestamp = ratecontrol_next_frame_start;
          }
          using rtp_tick = std::chrono::duration<uint32_t, std::ratio<1, 90000>>;
          uint32_t timestamp = std::chrono::round<rtp_tick>(*packet->frame_timestamp - video_epoch).count();

          // set FEC info now that we know for sure what our percentage will be for this frame
          for (auto x = 0; x < shards.size(); ++x) {
            auto *inspect = (video_packet_raw_t *) shards.data(x);

            inspect->packet.fecInfo =
              (x << 12 |
               shards.data_shards << 22 |
               shards.percentage << 4);

            inspect->rtp.header = 0x80 | FLAG_EXTENSION;
            inspect->rtp.sequenceNumber = util::endian::big<uint16_t>(
              static_cast<std::uint16_t>(block_lowseq + static_cast<std::uint32_t>(x))
            );
            inspect->rtp.timestamp = util::endian::big<uint32_t>(timestamp);

            inspect->packet.multiFecBlocks = (blockIndex << 4) | ((fec_blocks_needed - 1) << 6);
            inspect->packet.frameIndex = packet->frame_index();

            // Use the deterministic IV construction algorithm specified in NIST SP 800-38D
            // Section 8.2.1. The counter identifies each use of the client's unique key.
            std::copy_n((uint8_t *) &channel->gcm_iv_counter, sizeof(channel->gcm_iv_counter), std::begin(iv));
            iv[11] = 'V';  // Video stream
            channel->gcm_iv_counter++;

            auto *prefix = (video_packet_enc_prefix_t *) shards.prefix(x);
            prefix->frameNumber = packet->frame_index();
            std::copy(std::begin(iv), std::end(iv), prefix->iv);
            channel->cipher.encrypt(std::string_view {(char *) inspect, (size_t) blocksize}, prefix->tag, (uint8_t *) inspect, &iv);

            if (x - next_shard_to_send + 1 >= send_batch_size || x + 1 == shards.size()) {
              // Pace every bounded batch. The first batch is due immediately; subsequent batches
              // are distributed according to the negotiated bitrate/FEC/cadence plan.
              const auto due = ratecontrol_frame_start +
                               video_pacing_offset(ratecontrol_frame_packets_sent, pacing_plan.packets_per_second);
              auto now = std::chrono::steady_clock::now();
              const auto schedule_rebase = std::chrono::nanoseconds {
                video_pacing_rebase_ns(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                    due - ratecontrol_frame_start
                  ).count(),
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now - ratecontrol_frame_start
                  ).count()
                )
              };
              ratecontrol_frame_start += schedule_rebase;
              const auto bounded_due = due + schedule_rebase;
              if (now < bounded_due) {
                timer->sleep_for(bounded_due - now);
              }

              size_t current_batch_size = x - next_shard_to_send + 1;
              batch_info.block_offset = next_shard_to_send;
              batch_info.block_count = current_batch_size;

              frame_send_batch_latency_logger.first_point_now();
              // Use a batched send if it's supported on this platform
              if (!platf::send_batch(batch_info)) {
                // Batched send is not available, so send each packet individually
                for (auto y = 0; y < current_batch_size; y++) {
                  auto send_info = platf::send_info_t {
                    shards.prefix(next_shard_to_send + y),
                    shards.prefixsize,
                    shards.data(next_shard_to_send + y),
                    shards.blocksize,
                    (uintptr_t) sock.native_handle(),
                    peer_address,
                    channel->peer.port(),
                    channel->local_address,
                  };

                  platf::send(send_info);
                }
              }
              frame_send_batch_latency_logger.second_point_now_and_log();

              ratecontrol_frame_packets_sent += current_batch_size;
              next_shard_to_send = x + 1;
            }
          }

          ++blockIndex;
        });

        // Remember the final scheduled packet time in case the next frame is already queued.
        ratecontrol_next_frame_start = ratecontrol_frame_start +
                                       video_pacing_offset(
                                         ratecontrol_frame_packets_sent,
                                         pacing_plan.packets_per_second
                                       );

        // The start point is recorded once for the encoded frame above. Finish the sample only
        // after every FEC block has been paced and submitted, rather than producing one partial
        // sample per block.
        frame_network_latency_logger.second_point_now_and_log();
      } catch (const std::exception &e) {
        BOOST_LOG(error) << "Broadcast video failed "sv << e.what();
        request_recovery_idr("video packetization or send failed"sv);
        std::this_thread::sleep_for(100ms);
      }
    }

    shutdown_event->raise(true);
  }

  void audioBroadcastThread(udp::socket &sock) {
    auto shutdown_event = mail::man->event<bool>(mail::broadcast_shutdown);
    auto packets = mail::man->queue<audio::packet_t>(mail::audio_packets);

    audio_packet_t audio_packet;
    fec::rs_t rs {reed_solomon_new(RTPA_DATA_SHARDS, RTPA_FEC_SHARDS)};
    crypto::aes_t iv(16);

    if (!rs) {
      BOOST_LOG(error) << "Failed to allocate audio Reed-Solomon context"sv;
      shutdown_event->raise(true);
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

    while (auto packet = packets->pop()) {
      if (shutdown_event->peek()) {
        break;
      }

      TUPLE_2D_REF(channel_data, packet_data, *packet);
      auto channel = std::static_pointer_cast<audio_channel_t>(channel_data);
      if (!channel || !channel->active.load(std::memory_order_acquire)) {
        continue;
      }

      auto sequenceNumber = channel->sequence_number;
      auto timestamp = channel->timestamp;

      const auto packet_iv = util::endian::big<std::uint32_t>(channel->av_ri_key_id + sequenceNumber);
      std::memcpy(iv.data(), &packet_iv, sizeof(packet_iv));

      auto &shards_p = channel->shard_ptrs;

      auto bytes = encode_audio(packet_data, shards_p[sequenceNumber % RTPA_DATA_SHARDS], iv, channel->cipher);
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

    shutdown_event->raise(true);
  }

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

    ctx.video_thread = std::thread {videoBroadcastThread, std::ref(ctx.video_sock)};
    ctx.audio_thread = std::thread {audioBroadcastThread, std::ref(ctx.audio_sock)};
    ctx.control_thread = std::thread {controlBroadcastThread, &ctx.control_server};

    ctx.recv_thread = std::thread {recvThread, std::ref(ctx)};

    rollback.disable();
    return 0;
  }

  void end_broadcast(broadcast_ctx_t &ctx) {
    auto broadcast_shutdown_event = mail::man->event<bool>(mail::broadcast_shutdown);

    broadcast_shutdown_event->raise(true);

    auto video_packets = mail::man->queue<video::packet_t>(
      mail::video_packets,
      video::ENCODED_PACKET_QUEUE_LIMIT
    );
    auto audio_packets = mail::man->queue<audio::packet_t>(mail::audio_packets);

    // Minimize delay stopping video/audio threads
    video_packets->stop();
    audio_packets->stop();

    if (ctx.av_ping_route_updates) {
      ctx.av_ping_route_updates->stop();
    }
    ctx.io_context.stop();

    boost::system::error_code ec;
    ctx.video_sock.close(ec);
    ctx.audio_sock.close(ec);

    video_packets.reset();
    audio_packets.reset();

    BOOST_LOG(debug) << "Waiting for main listening thread to end..."sv;
    if (ctx.recv_thread.joinable()) {
      ctx.recv_thread.join();
    }
    BOOST_LOG(debug) << "Waiting for main video thread to end..."sv;
    if (ctx.video_thread.joinable()) {
      ctx.video_thread.join();
    }
    BOOST_LOG(debug) << "Waiting for main audio thread to end..."sv;
    if (ctx.audio_thread.joinable()) {
      ctx.audio_thread.join();
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

    auto start_time = std::chrono::steady_clock::now();
    auto current_time = start_time;

    while (current_time - start_time < timeout) {
      auto delta_time = current_time - start_time;

      auto msg_opt = messages->pop(timeout - delta_time);
      if (!msg_opt) {
        break;
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
    video::capture(session->mail, session->config.monitor, session->video);
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
    audio::capture(session->mail, session->config.audio, session->audio);
  }

  namespace session {
    namespace {
      // Construct both per-session workers behind a shared gate. If creating either std::thread
      // throws, the already-created worker is released as a no-op and joined before its thread
      // object is destroyed. This keeps a resource-exhaustion failure from terminating the host.
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
          bool fail_before_video_for_test = false,
          std::function<void()> failure_handler = {}
        ) noexcept {
          try {
            gate_ = std::make_shared<std::promise<bool>>();
            auto ready = gate_->get_future().share();
            auto start_worker = [ready, failure_handler](std::string_view name, std::function<void()> entry) {
              return std::thread {[ready, failure_handler, name, entry = std::move(entry)]() mutable {
                try {
                  if (ready.get()) {
                    entry();
                  }
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
            audio_ = start_worker("Audio"sv, std::move(audio_entry));

#ifdef SUNSHINE_TESTS
            if (fail_before_video_for_test) {
              throw std::runtime_error {"injected second session-worker start failure"};
            }
#else
            (void) fail_before_video_for_test;
#endif

            video_ = start_worker("Video"sv, std::move(video_entry));
            return true;
          } catch (const std::exception &exception) {
            if (!fail_before_video_for_test) {
              BOOST_LOG(error) << "Failed to create streaming session workers: "sv << exception.what();
            }
          } catch (...) {
            if (!fail_before_video_for_test) {
              BOOST_LOG(error) << "Failed to create streaming session workers with an unknown error."sv;
            }
          }

          cancel();
          return false;
        }

        void commit(session_t &session) noexcept {
          session.audioThread = std::move(audio_);
          session.videoThread = std::move(video_);
          resolve(true);
        }

        void cancel() noexcept {
          resolve(false);
          if (video_.joinable()) {
            video_.join();
          }
          if (audio_.joinable()) {
            audio_.join();
          }
        }

        [[nodiscard]] bool has_joinable_worker() const noexcept {
          return audio_.joinable() || video_.joinable();
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
      };

      // Starting a Windows streaming session reapplies NVIDIA profile settings and can take
      // several seconds. Keep that process-wide platform state warm briefly after disconnect,
      // but also bound an accepted launch that never completes its RTSP handshake. The mutex owns
      // the single-active-session invariant and serializes both expiry paths with startup.
      std::mutex platform_lifecycle_mutex;
      bool remote_session_active {};
      std::uint64_t platform_lifecycle_generation {};
      task_pool_util::TaskPool::task_id_t pending_platform_stop {};
      bool platform_streaming_warm {};
      std::optional<std::uint64_t> warm_process_instance;

      void invalidate_pending_platform_stop_locked() {
        ++platform_lifecycle_generation;
        if (pending_platform_stop) {
          task_pool.cancel(pending_platform_stop);
          pending_platform_stop = nullptr;
        }
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

      void schedule_platform_stop_locked(std::chrono::steady_clock::duration delay) {
        invalidate_pending_platform_stop_locked();
        const auto generation = platform_lifecycle_generation;
        pending_platform_stop = task_pool.pushDelayed([generation]() {
                                           std::lock_guard delayed_lock(platform_lifecycle_mutex);
                                           if (generation != platform_lifecycle_generation || remote_session_active) {
                                             return;
                                           }

                                           // A new /launch can replace the retained process just before publishing its pending
                                           // RTSP handshake. Do not terminate that new process in the HTTP -> RTSP gap; give it
                                           // one fresh handshake window.
                                            const auto current_process_instance = proc::proc.get_host_session_id();
                                            if (current_process_instance != 0 && current_process_instance != warm_process_instance) {
                                             warm_process_instance = current_process_instance;
                                             schedule_platform_stop_locked(config::stream.ping_timeout);
                                             return;
                                           }

                                           pending_platform_stop = nullptr;
                                           BOOST_LOG(info) << (platform_streaming_warm ? "Streaming session resume grace expired; terminating the retained app." : "Streaming launch handshake expired; terminating the unclaimed app.");
                                           // A reconnect normally reserves the warm state before publishing its RTSP handshake.
                                           // Clear any leftover reservation too so /serverinfo cannot advertise a resumable app
                                           // after this authoritative expiry point.
                                           rtsp_stream::clear_pending_launch_session();
                                            if (proc::proc.running() > 0) {
                                             proc::proc.terminate();
                                           }
                                           warm_process_instance.reset();
                                           if (platform_streaming_warm) {
                                             platf::streaming_will_stop();
                                             platform_streaming_warm = false;
                                           }
                                         },
                                                      delay)
                                  .task_id;
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

        invalidate_pending_platform_stop_locked();
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
        // ownership prevents another presentation path from claiming the display.
        const auto host_session_id = proc::proc.get_host_session_id();
        warm_process_instance = host_session_id == 0 ? std::nullopt : std::optional<std::uint64_t> {host_session_id};
        // The current launch reservation may still be waiting for its control connection.
        // Preserve the platform for at least that RTSP handshake window even when reconnect
        // grace is off or configured shorter than ping_timeout.
        const bool validated_launch_pending = !rtsp_stream::launch_session_available();
        if (config::stream.session_resume_grace <= 0ms && !validated_launch_pending) {
          BOOST_LOG(info) << "Session resume grace is disabled; terminating the retained app."sv;
          proc::proc.terminate();
          warm_process_instance.reset();
          platf::streaming_will_stop();
          platform_streaming_warm = false;
          return;
        }
        const auto retention = validated_launch_pending ?
                                 std::max(config::stream.session_resume_grace, config::stream.ping_timeout) :
                                 config::stream.session_resume_grace;
        schedule_platform_stop_locked(retention);
      }
    }  // namespace

    struct platform_launch_guard_t::impl_t {
      impl_t():
          lock {platform_lifecycle_mutex},
          idle {!remote_session_active} {
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

    platform_launch_guard_t guard_platform_launch() {
      return platform_launch_guard_t {std::make_unique<platform_launch_guard_t::impl_t>()};
    }

    state_e state(session_t &session) {
      return session.state.load(std::memory_order_relaxed);
    }

#ifdef SUNSHINE_TESTS
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

    bool worker_start_rollback_for_test() {
      std::atomic_int executed {};
      pending_session_workers_t workers;
      const bool prepared = workers.prepare(
        [&executed]() {
          ++executed;
        },
        [&executed]() {
          ++executed;
        },
        true
      );
      return !prepared && executed.load() == 0 && !workers.has_joinable_worker();
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
        session.permission.store(new_permissions, std::memory_order_release);
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
    }

    void graceful_stop(session_t &session) {
      if (!transition_to_stopping(session, true)) {
        return;
      }

      session.shutdown_event->raise(true);
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
      return true;
    }

    void join(session_t &session) {
      // A live desktop resize owns process/display state and can outlive the control thread.
      // Reap it before arming the NVENC hang watchdog and before releasing the active-session
      // slot, so a successor can never overlap stale work from this launch.
      auto mode_worker_hang_task = []() {
        BOOST_LOG(fatal) << "Hang detected! Live video-mode worker failed to terminate in 30 seconds."sv;
        logging::log_flush();
        lifetime::debug_trap();
      };
      auto mode_worker_force_kill =
        task_pool.pushDelayed(mode_worker_hang_task, 30s).task_id;
      auto mode_worker_watchdog = util::fail_guard([&mode_worker_force_kill]() {
        task_pool.cancel(mode_worker_force_kill);
      });
      join_live_video_mode_worker(session);
      task_pool.cancel(mode_worker_force_kill);
      mode_worker_watchdog.disable();

      // Current Nvidia drivers have a bug where NVENC can deadlock the encoder thread with hardware-accelerated
      // GPU scheduling enabled. If this happens, we will terminate ourselves and the service can restart.
      // The alternative is that Sunshine can never start another session until it's manually restarted.
      auto task = []() {
        BOOST_LOG(fatal) << "Hang detected! Session failed to terminate in 10 seconds."sv;
        logging::log_flush();
        lifetime::debug_trap();
      };
      auto force_kill = task_pool.pushDelayed(task, 10s).task_id;
      auto fg = util::fail_guard([&force_kill]() {
        // Cancel the kill task if we manage to return from this function
        task_pool.cancel(force_kill);
      });

      BOOST_LOG(debug) << "Waiting for video to end..."sv;
      session.videoThread.join();
      BOOST_LOG(debug) << "Waiting for audio to end..."sv;
      session.audioThread.join();
      BOOST_LOG(debug) << "Waiting for control to end..."sv;
      session.controlEnd.view();
      // Reset input on session stop to avoid stuck repeated keys
      BOOST_LOG(debug) << "Resetting Input..."sv;
      input::reset(session.input).wait();

      // Release the authoritative active slot only after every media/control worker has joined.
      // Validated launch work takes the same lock, so a successor cannot overlap teardown.
      {
        std::lock_guard lifecycle_lock(platform_lifecycle_mutex);
        if (!remote_session_active) {
          BOOST_LOG(error) << "Streaming session ended without owning the active-session slot."sv;
        } else {
          remote_session_active = false;
          retain_or_stop_session_locked();
        }
      }

      BOOST_LOG(debug) << "Session ended"sv;
    }

    int start(session_t &session) {
      session.input = input::alloc(session.mail);

      session.broadcast_ref = broadcast.ref();
      if (!session.broadcast_ref) {
        return -1;
      }

      session.pingTimeout = std::chrono::steady_clock::now() + config::stream.ping_timeout;

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
            false,
            [&session]() {
              stop(session);
            }
          )) {
        rollback_active_session();
        return -1;
      }

      // Publish the raw control registration only after platform startup and both worker-thread
      // constructions succeed. Its owner remains the RTSP slot; controlEnd is the lifetime barrier.
      {
        auto slot_lock = session.broadcast_ref->control_server._session.lock();
        if (session.broadcast_ref->control_server._session->session) {
          BOOST_LOG(error) << "Control server still has a registered session while the active slot is empty."sv;
          workers.cancel();
          rollback_active_session();
          return -1;
        }

        session.state.store(state_e::RUNNING, std::memory_order_relaxed);

        // A/V capture may already have received its UDP ping. Release it only after display/audio
        // platform state has finished resuming; control_ready independently protects source routing.
        session.startup_ready.raise(true);
        session.broadcast_ref->control_server._session->session = &session;
        session.broadcast_ref->control_server._session->peer = nullptr;
        workers.commit(session);
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
      session->device_name = launch_session.device_name;
      session->device_uuid = launch_session.unique_id;
      session->client_policy_generation = 0;
      session->permission.store(launch_session.perm, std::memory_order_relaxed);

      session->config = config;

      // Seed the effective-mode view with the launch mode so a live request refused before the
      // encode loop has published anything still reports a real mode rather than zeros. The video
      // thread receives a copy of session->config.monitor, so both sides share this publisher.
      session->config.monitor.effective_mode = std::make_shared<video::effective_video_mode_publisher_t>(
        video::effective_video_mode_t {
          config.monitor.width,
          config.monitor.height,
          config.monitor.framerateX100 > 0 ? config.monitor.framerateX100 : config.monitor.framerate * 100,
          config.monitor.bitrate,
        }
      );
      session->config.monitor.sbs_telemetry_subscription =
        std::make_shared<video::sbs_telemetry_subscription_t>();

      session->video = std::make_shared<video_channel_t>();
      session->video->packet_size = config.packetsize;
      session->video->min_required_fec_packets = config.minRequiredFecPackets;
      session->video->effective_mode = session->config.monitor.effective_mode;
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
      session->control.hdr_queue = mail->event<video::hdr_info_t>(mail::hdr);
      session->control.live_video_mode_ack_queue = mail->queue<live_video_mode_ack_t>(
        mail::live_video_mode_ack,
        LIVE_VIDEO_MODE_ACK_QUEUE_LIMIT
      );
      session->control.sbs_telemetry_event =
        mail->event<video::sbs_telemetry_snapshot_t>(mail::sbs_telemetry);
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
