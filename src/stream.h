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
#include <utility>

// lib includes
#include <boost/asio.hpp>

// local includes
#include "audio.h"
#include "crypto.h"
#include "video.h"

namespace rtsp_stream {
  struct launch_session_t;
}

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
   * overhead. A frame-size-derived lower bound keeps every frame within 75% of one frame
   * interval, leaving capture/encode scheduling slack. The ceiling is a safety limit for
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
    const auto max_frame_span_ns = std::max<std::int64_t>(1, frame_interval_ns * 3 / 4);
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

  struct session_t;

  struct config_t {
    audio::config_t audio;
    video::config_t monitor;

    int packetsize;
    int minRequiredFecPackets;
    int audioQosType;
    int videoQosType;
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
