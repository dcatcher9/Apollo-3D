/**
 * @file src/stream.h
 * @brief Declarations for the streaming protocols.
 */
#pragma once

// standard includes
#include <cstddef>
#include <cstdint>
#include <limits>
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

  constexpr std::size_t CONTROL_HEADER_V1_SIZE = sizeof(std::uint16_t);
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

  [[nodiscard]] constexpr bool is_valid_control_packet_size(std::size_t packet_size) {
    return packet_size >= CONTROL_HEADER_V1_SIZE && packet_size <= CONTROL_PACKET_SIZE_MAX;
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

  [[nodiscard]] constexpr bool is_valid_legacy_input_payload(
    std::size_t payload_size,
    std::uint32_t tagged_cipher_size
  ) {
    return tagged_cipher_size >= CONTROL_GCM_TAG_SIZE &&
           payload_size >= sizeof(tagged_cipher_size) &&
           tagged_cipher_size == payload_size - sizeof(tagged_cipher_size);
  }

  struct session_t;

  struct config_t {
    audio::config_t audio;
    video::config_t monitor;

    int packetsize;
    int minRequiredFecPackets;
    int mlFeatureFlags;
    int controlProtocolType;
    int audioQosType;
    int videoQosType;

    uint32_t encryptionFlagsEnabled;

    std::optional<int> gcmap;
  };

  namespace session {
    enum class state_e : int {
      STOPPED,  ///< The session is stopped
      STOPPING,  ///< The session is stopping
      STARTING,  ///< The session is starting
      RUNNING,  ///< The session is running
    };

    enum class client_policy_result_e {
      ignored,
      updated,
      disconnect,
    };

    std::shared_ptr<session_t> alloc(config_t &config, rtsp_stream::launch_session_t &launch_session);
    std::string uuid(const session_t& session);
    bool uuid_match(const session_t& session, const std::string_view& uuid);
    std::string client_name(const session_t &session);
    crypto::PERM permissions(const session_t &session);
    client_policy_result_e update_client_policy(
      session_t &session,
      std::uint64_t generation,
      std::string_view name,
      crypto::PERM new_permissions,
      bool revoked
    );
    int start(session_t &session, const std::string &addr_string);
    void stop(session_t &session);
    void graceful_stop(session_t& session);
    bool stop_if_client_policy_current(session_t &session, std::uint64_t generation, bool graceful);
    void join(session_t &session);
    state_e state(session_t &session);
#ifdef SUNSHINE_TESTS
    void set_state_for_test(session_t &session, state_e state);
#endif
  }  // namespace session
}  // namespace stream
