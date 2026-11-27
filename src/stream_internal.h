/**
 * @file src/stream_internal.h
 * @brief Private session, channel and control-server state shared by the stream sources.
 */
#pragma once

// standard includes
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

// lib includes
#include <boost/asio.hpp>

extern "C" {
  // clang-format off
#include <moonlight-common-c/src/Limelight-internal.h>
  // clang-format on
}

// local includes
#include "atomic_presentation.h"
#include "crypto.h"
#include "input.h"
#include "logging.h"
#include "microphone.h"
#include "network.h"
#include "platform/common.h"
#include "stream.h"
#include "sync.h"
#include "thread_safe.h"
#include "utility.h"
#include "video.h"
#ifdef SUNSHINE_TESTS
  #include "atomic_presentation_test.h"
#endif

namespace stream {
  namespace asio = boost::asio;
  using asio::ip::udp;
  using namespace std::literals;

  enum class socket_e : int {
    video,  ///< Video
    audio  ///< Audio
  };

#pragma pack(push, 1)

  struct audio_fec_packet_t {
    RTP_PACKET rtp;
    AUDIO_FEC_HEADER fecHeader;
  };

#pragma pack(pop)

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
    // Therefore, iterate is implemented in stream_control.cpp
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
      proof_deferred_acks_t<video::stream_gamma_ack_t, STREAM_GAMMA_QUEUE_LIMIT> deferred_stream_gamma_acks;
      unsigned outstanding_stream_gamma_requests = 0;
      // A syntactically valid v2 control message can arrive just before the initial encoder has
      // published generation 1. Preserve any immediate validation refusal and correlate it once
      // a complete applied state exists; v2 never emits an unproven generation-zero ACK.
      proof_deferred_acks_t<live_video_mode_ack_t, LIVE_VIDEO_MODE_ACK_QUEUE_LIMIT> deferred_live_video_mode_acks;
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

  // The serialized 0x3007 worker (stream_live_mode.cpp). The control thread validates and
  // dispatches requests; every acknowledgement is queued back for the control thread to send.
  void reject_live_video_mode(
    const safe::mail_t &mail,
    const std::shared_ptr<video::effective_video_mode_publisher_t> &effective_mode,
    std::uint32_t request_id,
    live_video_mode_ack_e status
  );
  void dispatch_live_video_mode(session_t &session, live_video_mode_request_t request);
  void stop_live_video_mode_worker(session_t &session);
  void join_live_video_mode_worker(session_t &session);

  // The ENet control server's thread body (stream_control.cpp).
  void controlBroadcastThread(control_server_t *server);

  // The per-session encrypted UDP senders (stream_media.cpp).
#ifdef SUNSHINE_TESTS
  struct media_sender_test_hooks_t;
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
  );

  void run_audio_sender(
    udp::socket &sock,
    audio::packet_queue_t packets,
    std::shared_ptr<audio_channel_t> channel,
    safe::mail_raw_t::event_t<bool> shutdown_event
#ifdef SUNSHINE_TESTS
    ,
    media_sender_test_hooks_t *test_hooks = nullptr
#endif
  );

  namespace session {
    // The process-wide platform lifecycle (stream_platform_lifecycle.cpp). Session start and
    // join claim and release the sole active slot; every access holds the mutex.
    extern std::mutex platform_lifecycle_mutex;
    extern bool remote_session_active;

    bool activate_remote_session_locked(std::uint32_t remote_virtual_display_lease);
    void retain_or_stop_session_locked();
  }  // namespace session
}  // namespace stream
