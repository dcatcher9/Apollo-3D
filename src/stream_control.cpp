/**
 * @file src/stream_control.cpp
 * @brief Definitions for the encrypted ENet control server and its host-originated messages.
 */

// standard includes
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <format>
#include <optional>
#include <string_view>
#include <vector>

extern "C" {
  // clang-format off
#include <moonlight-common-c/src/Limelight-internal.h>
  // clang-format on
}

// local includes
#include "audio.h"
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
#include "stream.h"
#include "stream_internal.h"
#include "utility.h"
#include "video.h"

using namespace std::literals;

namespace stream {
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

#pragma pack(push, 1)

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

#pragma pack(pop)

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
   *
   * @return `false` only when the acknowledgement does not fit the v2 wire format, which
   * proof_deferred_acks_t treats as a reason to close the session.
   */
  bool send_live_video_mode_ack(session_t *session, const live_video_mode_ack_t &ack) {
    if (!session->control.peer) {
      // Still waiting for PING from Artemis, or the peer is already gone. There is nothing to
      // answer; the client's own request timeout covers the missing reply.
      return true;
    }

    control_live_video_mode_ack_v2_t plaintext {};
    plaintext.header.type = control_packet::live_video_mode_ack;
    plaintext.header.payloadLength =
      sizeof(control_live_video_mode_ack_v2_t) - sizeof(control_header_v2);
    if (!encode_atomic_presentation_ack_payload(ack, plaintext.body)) {
      return false;
    }
    const int send_result = send_control_packet(
      session,
      session->broadcast_ref->control_server,
      plaintext
    );

    if (send_result != 0) {
      TUPLE_2D(port, addr, platf::from_sockaddr_ex((sockaddr *) &session->control.peer->address.address));
      BOOST_LOG(warning) << "Couldn't send live video-mode acknowledgement to ["sv << addr << ':' << port << ']';
      return true;
    }

    BOOST_LOG(debug) << "Sent atomic-presentation v2 ack: request "sv << ack.request_id
                     << " status="sv << static_cast<int>(ack.status) << ", encoded "sv
                     << ack.applied_encoded_width << 'x' << ack.applied_encoded_height
                     << '@' << (ack.applied_framerate_x100 / 100.0) << "Hz "sv
                     << ack.applied_bitrate_kbps << "kbps"sv;
    return true;
  }

  /** Answer one stream gamma request (0x300D); the same return contract as above. */
  bool send_stream_gamma_ack(session_t &session, const video::stream_gamma_ack_t &ack) {
    if (!session.control.peer) {
      return true;
    }
    control_stream_gamma_ack_t plaintext {};
    plaintext.header.type = control_packet::stream_gamma_ack;
    plaintext.header.payloadLength = STREAM_GAMMA_ACK_PAYLOAD_SIZE;
    if (!encode_stream_gamma_ack_payload(ack, plaintext.body)) {
      return false;
    }
    (void) send_control_packet(&session, session.broadcast_ref->control_server, plaintext);
    return true;
  }

  /** The next acknowledgement queued for an ENet peer that is still connected. */
  template<class Ack>
  auto next_queued_ack(control_server_t &server, safe::mail_raw_t::queue_t<Ack> &queue) {
    return [&server, &queue]() -> std::optional<Ack> {
      while (server._session->peer && queue->peek()) {
        if (auto ack = queue->pop(0ms)) {
          return std::move(*ack);
        }
      }
      return std::nullopt;
    };
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
      if (!session->control.deferred_stream_gamma_acks.admit_request()) {
        BOOST_LOG(error) << "Stream gamma request queue exceeded its session bound"sv;
        session::stop(*session);
        return;
      }
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
            session->control.deferred_stream_gamma_acks.clear();
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

            // Answer every live video-mode request (0x3007) and stream gamma request (0x300C)
            // exactly once, under the one rule of proof_deferred_acks_t. Unlike the depth phase
            // these must not coalesce, so they arrive on real queues: a client that sends two
            // requests is owed two replies. The joined presentation worker and the encode loop
            // only queue plain outcomes, leaving the peer, the cipher and ENet to this thread.
            const auto close_session = [&](std::string_view reason) {
              BOOST_LOG(error) << reason;
              session::stop(*session);
            };
            session->control.deferred_live_video_mode_acks.drain(
              next_queued_ack(*server, session->control.live_video_mode_ack_queue),
              [&] {
                // Waiting refusals are released only to a connected client.
                const auto &effective_mode = session->config.monitor.effective_mode;
                return server->_session->peer && effective_mode ? effective_mode->current() : video::effective_video_mode_t {};
              },
              [&](const live_video_mode_ack_t &ack) {
                return send_live_video_mode_ack(session, ack);
              },
              close_session
            );
            session->control.deferred_stream_gamma_acks.drain(
              next_queued_ack(*server, session->control.stream_gamma_ack_queue),
              [&] {
                const auto &gamma_state = session->config.monitor.stream_gamma_state;
                return server->_session->peer && gamma_state ? gamma_state->current() : video::stream_gamma_snapshot_t {};
              },
              [&](const video::stream_gamma_ack_t &ack) {
                return send_stream_gamma_ack(*session, ack);
              },
              close_session
            );
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
      session->control.deferred_stream_gamma_acks.clear();
      session->control.peer = nullptr;
      server->_session->peer = nullptr;
      server->_session->session = nullptr;
    }

    server->flush();
  }
}  // namespace stream
