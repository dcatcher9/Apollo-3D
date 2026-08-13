/**
 * @file src/stream.h
 * @brief Declarations for the streaming protocols.
 */
#pragma once

// standard includes
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

// lib includes
#include <boost/asio.hpp>

// local includes
#include "audio.h"
#include "crypto.h"
#include "video.h"

namespace rtsp_stream {
  struct launch_session_t;
}

namespace proc {
  // Opaque declaration so the acknowledgement mapping can be declared here without pulling
  // Boost.Process into every consumer of stream.h. The underlying type must match process.h.
  enum class live_video_mode_result_e : int;
}  // namespace proc

namespace stream {
  constexpr auto VIDEO_STREAM_PORT = 9;
  constexpr auto CONTROL_PORT = 10;
  constexpr auto AUDIO_STREAM_PORT = 11;

  // The upper bound keeps a maximum-size encrypted IPv4 UDP datagram legal:
  // 65507 bytes - 16 bytes maximum RTP header - 32 bytes encryption prefix.
  constexpr int VIDEO_PACKET_SIZE_MIN = 200;
  constexpr int VIDEO_PACKET_SIZE_MAX = 65459;

  // The sender advertises floor(100 * parity / data) in an 8-bit field. With
  // a one-shard data block, the current planner can safely represent at most
  // two minimum parity shards.
  constexpr int MIN_REQUIRED_FEC_PACKETS_MAX = 2;
  constexpr std::size_t FEC_PACKET_INDEX_MAX = 1023;
  // ANNOUNCE validation normally guarantees a positive encoder bitrate. Keep a modest fallback
  // for defensive callers without forcing valid low-bitrate streams up to a large fixed rate.
  constexpr std::uint64_t VIDEO_PACING_FALLBACK_ENCODED_BPS = 10'000'000;
  constexpr std::uint64_t VIDEO_PACING_MAX_WIRE_BPS = 800'000'000;
  constexpr std::int64_t VIDEO_PACING_QUANTUM_NS = 1'000'000;
  constexpr std::int64_t VIDEO_PACING_MAX_CATCHUP_NS = VIDEO_PACING_QUANTUM_NS;
  constexpr std::int64_t VIDEO_BACKLOG_MIN_MAX_AGE_NS = 50'000'000;
  // ENet has no cross-thread wake primitive. Keep its blocking service interval short so
  // host-originated rumble, HDR, depth, telemetry, and acknowledgement queues are observed
  // promptly without moving ENet or cipher ownership off the control thread.
  constexpr std::chrono::milliseconds CONTROL_OUTGOING_MAX_WAIT {10};
  constexpr std::size_t VIDEO_RS_CONTEXT_CACHE_LIMIT = 8;
  constexpr std::size_t VIDEO_SEND_WORKSPACE_RETAIN_LIMIT = 16 * 1024 * 1024;

  constexpr std::size_t CONTROL_HEADER_V2_SIZE = 2 * sizeof(std::uint16_t);
  constexpr std::size_t CONTROL_GCM_TAG_SIZE = 16;
  constexpr std::size_t CONTROL_ENCRYPTED_LENGTH_FIELD_SIZE = sizeof(std::uint16_t);
  constexpr std::size_t CONTROL_ENCRYPTED_SEQUENCE_SIZE = sizeof(std::uint32_t);
  constexpr std::size_t CONTROL_ENCRYPTED_MIN_LENGTH =
    CONTROL_ENCRYPTED_SEQUENCE_SIZE + CONTROL_GCM_TAG_SIZE + CONTROL_HEADER_V2_SIZE;
  constexpr std::size_t CONTROL_PACKET_SIZE_MAX =
    std::numeric_limits<std::uint16_t>::max() + CONTROL_HEADER_V2_SIZE;

  [[nodiscard]] constexpr bool is_valid_video_packet_size(int packet_size) {
    return packet_size >= VIDEO_PACKET_SIZE_MIN && packet_size <= VIDEO_PACKET_SIZE_MAX;
  }

  [[nodiscard]] constexpr bool is_valid_video_transport_config(int packet_size, int min_required_fec_packets) {
    return is_valid_video_packet_size(packet_size) &&
           min_required_fec_packets >= 0 &&
           min_required_fec_packets <= MIN_REQUIRED_FEC_PACKETS_MAX;
  }

  [[nodiscard]] constexpr std::size_t fec_packet_count(std::size_t payload_size, std::size_t block_size) {
    return block_size == 0 ? 0 : payload_size / block_size + (payload_size % block_size != 0);
  }

  [[nodiscard]] constexpr bool is_valid_fec_block_size(std::size_t payload_size, std::size_t block_size) {
    const auto packet_count = fec_packet_count(payload_size, block_size);
    return packet_count > 0 && packet_count <= FEC_PACKET_INDEX_MAX;
  }

  [[nodiscard]] constexpr std::uint64_t ceil_div_u64(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0 ? 0 : numerator / denominator + (numerator % denominator != 0);
  }

  [[nodiscard]] constexpr std::size_t video_fec_shard_count(
    std::size_t payload_size,
    std::size_t block_size,
    int fec_percentage,
    int min_required_fec_packets
  ) {
    const auto data_shards = fec_packet_count(payload_size, block_size);
    if (data_shards == 0 || fec_percentage <= 0) {
      return data_shards;
    }

    const auto parity_shards = std::max<std::size_t>(
      ceil_div_u64(data_shards * static_cast<std::uint64_t>(fec_percentage), 100),
      static_cast<std::size_t>(std::max(0, min_required_fec_packets))
    );
    return data_shards + parity_shards;
  }

  [[nodiscard]] constexpr std::int64_t video_frame_interval_ns(int framerate_millihz) {
    return framerate_millihz > 0 ? static_cast<std::int64_t>(1'000'000'000'000ULL / framerate_millihz) :
                                   0;
  }

  [[nodiscard]] constexpr std::int64_t video_packet_max_queue_age_ns(int framerate_millihz) {
    const auto frame_interval_ns = video_frame_interval_ns(framerate_millihz);
    return frame_interval_ns > 0 ?
             std::max(VIDEO_BACKLOG_MIN_MAX_AGE_NS, frame_interval_ns * 3) :
             VIDEO_BACKLOG_MIN_MAX_AGE_NS;
  }

  struct video_pacing_plan_t {
    std::uint64_t target_wire_bps;
    std::uint64_t packets_per_second;
    std::size_t packets_per_quantum;
    std::int64_t frame_interval_ns;
    std::int64_t max_frame_span_ns;
  };

  /**
   * Build a bounded pacing plan from the encoded-data rate and this frame's expected wire size.
   *
   * The nominal rate accounts for the actual data/parity shard ratio and per-packet transport
   * overhead. A frame-size-derived lower bound keeps every frame within half of one frame
   * interval, reducing decode readiness latency while retaining bitrate-aware anti-burst pacing.
   * The ceiling is a safety limit for
   * pathological frames, not the ordinary pacing rate.
   */
  [[nodiscard]] constexpr video_pacing_plan_t make_video_pacing_plan(
    int encoded_bitrate_kbps,
    int framerate_millihz,
    std::size_t estimated_data_packets,
    std::size_t estimated_wire_packets,
    std::size_t encoded_payload_packet_bytes,
    std::size_t wire_packet_bytes
  ) {
    const auto safe_data_packets = std::max<std::size_t>(1, estimated_data_packets);
    const auto safe_wire_packets = std::max(safe_data_packets, estimated_wire_packets);
    const auto safe_payload_bytes = std::max<std::size_t>(1, encoded_payload_packet_bytes);
    const auto safe_wire_bytes = std::max<std::size_t>(1, wire_packet_bytes);
    const auto encoded_bps = encoded_bitrate_kbps > 0 ?
                               static_cast<std::uint64_t>(encoded_bitrate_kbps) * 1000 :
                               VIDEO_PACING_FALLBACK_ENCODED_BPS;
    const auto nominal_fec_bps = std::min(
      VIDEO_PACING_MAX_WIRE_BPS,
      ceil_div_u64(
        encoded_bps * static_cast<std::uint64_t>(safe_wire_packets),
        static_cast<std::uint64_t>(safe_data_packets)
      )
    );
    const auto nominal_wire_bps = std::min(
      VIDEO_PACING_MAX_WIRE_BPS,
      ceil_div_u64(
        nominal_fec_bps * safe_wire_bytes,
        safe_payload_bytes
      )
    );

    auto frame_interval_ns = video_frame_interval_ns(framerate_millihz);
    if (frame_interval_ns <= 0) {
      frame_interval_ns = video_frame_interval_ns(60'000);
    }
    const auto max_frame_span_ns = std::max<std::int64_t>(1, frame_interval_ns / 2);
    const auto estimated_wire_bits =
      static_cast<std::uint64_t>(safe_wire_packets) *
      safe_wire_bytes * 8;
    const auto cadence_wire_bps = ceil_div_u64(
      estimated_wire_bits * 1'000'000'000ULL,
      static_cast<std::uint64_t>(max_frame_span_ns)
    );
    const auto target_wire_bps = std::min(
      VIDEO_PACING_MAX_WIRE_BPS,
      std::max(nominal_wire_bps, cadence_wire_bps)
    );
    const auto packets_per_second = std::max<std::uint64_t>(
      1,
      ceil_div_u64(target_wire_bps, safe_wire_bytes * 8)
    );
    const auto packets_per_quantum = static_cast<std::size_t>(
      std::max<std::uint64_t>(
        1,
        ceil_div_u64(
          packets_per_second * static_cast<std::uint64_t>(VIDEO_PACING_QUANTUM_NS),
          1'000'000'000ULL
        )
      )
    );

    return {
      target_wire_bps,
      packets_per_second,
      packets_per_quantum,
      frame_interval_ns,
      max_frame_span_ns,
    };
  }

  /**
   * Move a late pacing schedule forward while retaining at most one quantum of catch-up credit.
   * This avoids turning a delayed timer callback into an unbounded run of immediate send batches.
   */
  [[nodiscard]] constexpr std::int64_t video_pacing_rebase_ns(
    std::int64_t scheduled_ns,
    std::int64_t now_ns,
    std::int64_t max_catchup_ns = VIDEO_PACING_MAX_CATCHUP_NS
  ) {
    if (now_ns <= scheduled_ns) {
      return 0;
    }

    const auto overdue_ns = now_ns - scheduled_ns;
    return std::max<std::int64_t>(0, overdue_ns - std::max<std::int64_t>(0, max_catchup_ns));
  }

  [[nodiscard]] constexpr std::chrono::nanoseconds video_pacing_offset(
    std::size_t packets_sent,
    std::uint64_t packets_per_second
  ) {
    return std::chrono::nanoseconds {
      static_cast<std::int64_t>(
        ceil_div_u64(
          static_cast<std::uint64_t>(packets_sent) * 1'000'000'000ULL,
          std::max<std::uint64_t>(1, packets_per_second)
        )
      )
    };
  }

  [[nodiscard]] constexpr bool is_valid_control_packet_size(std::size_t packet_size) {
    return packet_size >= sizeof(std::uint16_t) && packet_size <= CONTROL_PACKET_SIZE_MAX;
  }

  // The ENet payload excludes the two-byte packet type, but still contains the
  // encrypted length field followed by exactly `encrypted_length` bytes.
  [[nodiscard]] constexpr bool is_valid_encrypted_control_payload(
    std::size_t payload_size,
    std::uint16_t encrypted_length
  ) {
    return encrypted_length >= CONTROL_ENCRYPTED_MIN_LENGTH &&
           payload_size >= CONTROL_ENCRYPTED_LENGTH_FIELD_SIZE &&
           encrypted_length == payload_size - CONTROL_ENCRYPTED_LENGTH_FIELD_SIZE;
  }

  [[nodiscard]] constexpr bool is_valid_decrypted_control_payload(
    std::size_t plaintext_size,
    std::uint16_t declared_payload_size
  ) {
    return plaintext_size >= CONTROL_HEADER_V2_SIZE &&
           declared_payload_size == plaintext_size - CONTROL_HEADER_V2_SIZE;
  }

  /**
   * Combine two logical input buffers while reserving and clearing a prefix at every slice.
   * `storage` is a retained high-watermark workspace; only the returned prefix is valid. This
   * avoids a full-size allocation and value-initialization on every encoded frame.
   */
  [[nodiscard]] std::optional<std::size_t> concat_and_insert_into(
    std::vector<std::uint8_t> &storage,
    std::uint64_t insert_size,
    std::uint64_t slice_size,
    std::string_view data1,
    std::string_view data2
  );

  // Sunshine's encrypted Gen-7 protocol uses 0x5502 in both directions for unrelated messages:
  // controller RGB feedback is host->client, while this fixed-size per-frame FEC report is
  // client->host. Keeping the inbound body's size explicit prevents it from falling through as an
  // "unknown control message" and prevents a malformed RGB-sized body from being accepted as FEC.
  constexpr std::size_t FRAME_FEC_STATUS_PAYLOAD_SIZE = 21;

  [[nodiscard]] constexpr bool is_valid_frame_fec_status_payload_size(
    std::size_t payload_size
  ) {
    return payload_size == FRAME_FEC_STATUS_PAYLOAD_SIZE;
  }

  // Bounds for a live 0x3007 stream geometry/rate change. Dimensions must be even because the
  // whole encode path is 4:2:0 subsampled, and the frame rate travels in hundredths of a hertz so
  // fractional rates (23.976, 29.97) survive the wire.
  //
  // The encoder's own maximum width is deliberately NOT part of this check: the encode loop owns
  // the codec capability snapshot and clamps an oversized request the same way it already caps an
  // SBS packed width. Clamping there always succeeds, whereas rejecting a creatable mode here
  // would deny a mode the host could actually deliver.
  constexpr int LIVE_VIDEO_MODE_DIMENSION_MIN = 2;
  constexpr int LIVE_VIDEO_MODE_DIMENSION_MAX = 16384;
  constexpr int LIVE_VIDEO_MODE_FRAMERATE_X100_MIN = 100;
  constexpr int LIVE_VIDEO_MODE_FRAMERATE_X100_MAX = 100'000;

  [[nodiscard]] constexpr bool is_valid_live_video_mode_dimension(int dimension) {
    return dimension >= LIVE_VIDEO_MODE_DIMENSION_MIN &&
           dimension <= LIVE_VIDEO_MODE_DIMENSION_MAX &&
           (dimension & 1) == 0;
  }

  [[nodiscard]] constexpr bool is_valid_live_video_mode_framerate_x100(int framerate_x100) {
    return framerate_x100 >= LIVE_VIDEO_MODE_FRAMERATE_X100_MIN &&
           framerate_x100 <= LIVE_VIDEO_MODE_FRAMERATE_X100_MAX;
  }

  namespace detail {
    /**
     * Transaction gate for live 0x3007 mode requests.
     *
     * The control thread may replace a request that is still pending, but once the worker begins
     * one transaction no later request may begin until the matching encoder completion and any
     * desktop rollback have finished. Tokens are host-generated because client request ids are
     * opaque and may repeat.
     */
    class live_video_mode_serial_gate_t {
    public:
      struct submission_t {
        std::uint64_t transaction_id;
        std::optional<std::uint64_t> superseded_transaction_id;
      };

      [[nodiscard]] submission_t submit() noexcept {
        ++_last_transaction_id;
        if (_last_transaction_id == 0) {
          ++_last_transaction_id;
        }
        const auto superseded = std::exchange(_pending_transaction_id, _last_transaction_id);
        return {_last_transaction_id, superseded};
      }

      [[nodiscard]] bool can_begin() const noexcept {
        return !_active_transaction_id && _pending_transaction_id.has_value();
      }

      [[nodiscard]] std::optional<std::uint64_t> begin_next() noexcept {
        if (!can_begin()) {
          return std::nullopt;
        }
        _active_transaction_id = std::exchange(_pending_transaction_id, std::nullopt);
        return _active_transaction_id;
      }

      [[nodiscard]] bool accepts_completion(std::uint64_t transaction_id) const noexcept {
        return _active_transaction_id == transaction_id;
      }

      [[nodiscard]] bool finish(std::uint64_t transaction_id) noexcept {
        if (!accepts_completion(transaction_id)) {
          return false;
        }
        _active_transaction_id.reset();
        return true;
      }

      void cancel_pending() noexcept {
        _pending_transaction_id.reset();
      }

      [[nodiscard]] std::optional<std::uint64_t> active() const noexcept {
        return _active_transaction_id;
      }

    private:
      std::uint64_t _last_transaction_id = 0;
      std::optional<std::uint64_t> _pending_transaction_id;
      std::optional<std::uint64_t> _active_transaction_id;
    };
  }  // namespace detail

  /**
   * Host->client outcome of a live 0x3007 video-mode request, carried by the 0x3008
   * acknowledgement. Exactly one acknowledgement is produced per accepted control message, so a
   * client never has to distinguish "host refused" from "host is slow" by timing out.
   *
   * WIRE-VISIBLE VALUES. These travel on the control stream and must match the
   * LIVE_VIDEO_MODE_ACK_* constants in the client's moonlight-common-c Limelight.h. Treat the
   * numbering as append-only; never renumber or reuse a value.
   */
  enum class live_video_mode_ack_e : std::uint16_t {
    applied = 0,  ///< The requested mode is live on the stream.
    rejected_invalid = 1,  ///< Failed validation. The client must not retry the same request.
    rejected_needs_reconnect = 2,  ///< Valid, but only a full reconnect can reach this mode.
    failed = 3,  ///< Transient failure; the change was rolled back and the client may retry.
  };

  /** Map a virtual-display transition outcome onto its wire acknowledgement status. */
  [[nodiscard]] live_video_mode_ack_e live_video_mode_ack_status(proc::live_video_mode_result_e result);

  // The acknowledgement body that follows the control header: u16 request_id, u16 status,
  // u16 applied_width, u16 applied_height, u16 applied_framerate_x100, u32 applied_bitrate_kbps,
  // all little-endian. The trailing u32 is deliberately unaligned; the struct is packed and the
  // body is serialized field by field, so no padding is ever inserted.
  constexpr std::size_t LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE = 14;

  // A client cannot have many requests outstanding, and the control thread drains this every
  // iteration. The bound only exists so a pathological client cannot grow the queue without limit.
  constexpr std::uint32_t LIVE_VIDEO_MODE_ACK_QUEUE_LIMIT = 8;

  /**
   * One acknowledgement, queued by whichever component decided the outcome and sent by the control
   * thread.
   *
   * `request_id` is the client's opaque correlation token, echoed verbatim. The `applied_*` fields
   * always describe the mode that is REALLY in effect, never the request:
   *  - on `applied`, that is the running encode session after every host-side transformation, so a
   *    width capped to the codec ceiling reports the capped value and its aspect-scaled height.
   *    With Host SBS active, `applied_width`/`applied_height` are the base per-eye values before
   *    the host's SBS doubling. With Host SBS off they are the effective encoded desktop values
   *    verbatim; this includes a Raw Full client whose desktop is already packed `2W x H`, because
   *    the host deliberately has no Raw-SBS presentation knowledge. `applied_bitrate_kbps` is the
   *    derived encoder budget rather than the requested wire budget.
   *  - on every refusal, that is the mode the session kept, so the client can resynchronize its UI
   *    to reality instead of being handed zeros or its own rejected request back.
   */
  struct live_video_mode_ack_t {
    int request_id;
    live_video_mode_ack_e status;
    int applied_width;
    int applied_height;
    int applied_framerate_x100;
    std::int64_t applied_bitrate_kbps;
  };

  /**
   * Serialize an acknowledgement body in wire order. No control header is written.
   *
   * @return `false` when a field does not fit its wire width, in which case @p out is untouched.
   * Every reachable acknowledgement carries an id decoded from the request's own u16 field and a
   * mode the encoder actually ran, so a rejection here means the caller invented a value.
   */
  [[nodiscard]] bool encode_live_video_mode_ack_payload(
    const live_video_mode_ack_t &ack,
    std::uint8_t (&out)[LIVE_VIDEO_MODE_ACK_PAYLOAD_SIZE]
  );

  // Artemis host-SBS telemetry control extension:
  //   0x3009 client -> host subscription (8-byte body)
  //   0x300A host -> client state        (88-byte body)
  //
  // All values below are wire-visible. Keep their numbering and sizes append-only and in sync
  // with moonlight-common-c's Limelight.h.
  constexpr std::uint8_t SBS_TELEMETRY_VERSION = 1;
  constexpr std::size_t SBS_TELEMETRY_SUBSCRIPTION_PAYLOAD_SIZE = 8;
  constexpr std::size_t SBS_TELEMETRY_STATE_PAYLOAD_SIZE = 88;
  constexpr std::uint16_t SBS_TELEMETRY_INTERVAL_MIN_MS = 100;
  constexpr std::uint16_t SBS_TELEMETRY_INTERVAL_MAX_MS = 2000;
  constexpr std::uint16_t SBS_TELEMETRY_INTERVAL_BACKGROUND_MS = 500;
  constexpr std::uint16_t SBS_TELEMETRY_INTERVAL_FOCUSED_MS = 100;
  constexpr std::uint8_t SBS_TELEMETRY_SUBSCRIBE_ENABLED = 1u << 0;
  constexpr std::uint8_t SBS_TELEMETRY_SUBSCRIBE_FOCUSED = 1u << 1;
  constexpr std::uint32_t CLIENT_FEATURE_SBS_TELEMETRY = 0x04;

  enum class sbs_telemetry_status_e : std::uint8_t {
    ok = 0,
    unavailable = 1,
    unsupported_version = 2,
    failed = 3,
  };

  enum class sbs_telemetry_subscription_decode_e {
    ok,
    unsupported_version,
    invalid,
  };

  struct sbs_telemetry_subscription_request_t {
    std::uint8_t flags = 0;
    std::uint16_t request_id = 0;
    std::uint16_t interval_ms = 0;

    [[nodiscard]] bool enabled() const noexcept {
      return (flags & SBS_TELEMETRY_SUBSCRIBE_ENABLED) != 0;
    }

    [[nodiscard]] bool focused() const noexcept {
      return (flags & SBS_TELEMETRY_SUBSCRIBE_FOCUSED) != 0;
    }
  };

  /**
   * Decode an exact subscription body. An exact-size unsupported-version body still exposes its
   * request id so the caller can return status 2; other malformed inputs return `invalid`.
   */
  [[nodiscard]] sbs_telemetry_subscription_decode_e decode_sbs_telemetry_subscription_payload(
    std::string_view payload,
    sbs_telemetry_subscription_request_t &request
  );

  /** Resolve interval 0 to the focused/background default and clamp every enabled cadence. */
  [[nodiscard]] std::uint16_t effective_sbs_telemetry_interval_ms(
    const sbs_telemetry_subscription_request_t &request
  ) noexcept;

  /** Explicitly bridge the private renderer state enum to the fixed public wire enum. */
  [[nodiscard]] sbs_telemetry_status_e sbs_telemetry_wire_status(
    video::sbs_telemetry_sample_status_e status
  ) noexcept;

  /**
   * Truthful pre-render fallback. Until the renderer publishes an authoritative snapshot, no
   * generation or configuration field is valid.
   */
  [[nodiscard]] video::sbs_telemetry_snapshot_t unavailable_sbs_telemetry_snapshot() noexcept;

  /** Dump 3D is meaningful only for a diagnostics-enabled session currently requesting Host SBS. */
  [[nodiscard]] bool sbs_debug_dump_request_allowed(
    bool diagnostics_enabled,
    int requested_sbs_mode,
    bool has_session_request_latch
  ) noexcept;

  struct sbs_telemetry_state_t {
    sbs_telemetry_status_e status = sbs_telemetry_status_e::unavailable;
    std::uint16_t request_id = 0;
    video::sbs_telemetry_snapshot_t snapshot;
  };

  /** Serialize the exact 88-byte little-endian state body; rejects non-finite float fields. */
  [[nodiscard]] bool encode_sbs_telemetry_state_payload(
    const sbs_telemetry_state_t &state,
    std::uint8_t (&out)[SBS_TELEMETRY_STATE_PAYLOAD_SIZE]
  );

  struct session_t;

  struct config_t {
    audio::config_t audio;
    video::config_t monitor;

    int packetsize;
    int minRequiredFecPackets;
    int audioQosType;
    int videoQosType;
    bool client_supports_sbs_telemetry = false;
  };

  namespace session {
    class platform_launch_guard_t {
    public:
      platform_launch_guard_t(platform_launch_guard_t &&) noexcept;
      platform_launch_guard_t &operator=(platform_launch_guard_t &&) noexcept;
      ~platform_launch_guard_t();

      platform_launch_guard_t(const platform_launch_guard_t &) = delete;
      platform_launch_guard_t &operator=(const platform_launch_guard_t &) = delete;

      /** Whether no streaming session was active when this guarded launch began. */
      [[nodiscard]] bool idle() const;

      /** Publish a validated RTSP launch and renew the retained platform deadline. */
      void commit();

    private:
      struct impl_t;
      explicit platform_launch_guard_t(std::unique_ptr<impl_t> impl);

      std::unique_ptr<impl_t> _impl;

      friend platform_launch_guard_t guard_platform_launch();
    };

    /**
     * Serialize validated launch preparation with sole-session startup and grace expiry.
     * Destroying an uncommitted guard leaves the existing grace deadline unchanged.
     */
    platform_launch_guard_t guard_platform_launch();

    enum class state_e : int {
      STOPPED,  ///< The session is stopped
      STOPPING,  ///< The session is stopping
      RUNNING,  ///< The session is running
    };

    enum class client_policy_result_e {
      ignored,
      updated,
      disconnect,
    };

    std::shared_ptr<session_t> alloc(config_t &config, rtsp_stream::launch_session_t &launch_session);
    std::string uuid(const session_t &session);
    bool uuid_match(const session_t &session, const std::string_view &uuid);
    std::string client_name(const session_t &session);
    crypto::PERM permissions(const session_t &session);
    client_policy_result_e update_client_policy(
      session_t &session,
      std::uint64_t generation,
      std::string_view name,
      crypto::PERM new_permissions,
      bool revoked
    );
    int start(session_t &session);
    void stop(session_t &session);
    void graceful_stop(session_t &session);
    bool stop_if_client_policy_current(session_t &session, std::uint64_t generation, bool graceful);
    void join(session_t &session);

    /**
     * Stop any process-wide streaming state retained for fast session resume.
     * Called during host shutdown before the global task pool is stopped.
     */
    void flush_platform_state();

    state_e state(session_t &session);
#ifdef SUNSHINE_TESTS
    void set_state_for_test(session_t &session, state_e state);
    bool claim_active_slot_for_test();
    void release_active_slot_for_test();
    bool worker_start_rollback_for_test();
#endif
  }  // namespace session
}  // namespace stream
