/**
 * @file src/stream.h
 * @brief Declarations for the streaming protocols.
 */
#pragma once

// standard includes
#include <cstddef>
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

  [[nodiscard]] constexpr bool is_valid_video_transport_config(int packet_size, int min_required_fec_packets) {
    return packet_size >= VIDEO_PACKET_SIZE_MIN &&
           packet_size <= VIDEO_PACKET_SIZE_MAX &&
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

    std::shared_ptr<session_t> alloc(config_t &config, rtsp_stream::launch_session_t &launch_session);
    std::string uuid(const session_t& session);
    bool uuid_match(const session_t& session, const std::string_view& uuid);
    bool update_device_info(session_t& session, const std::string& name, const crypto::PERM& newPerm);
    int start(session_t &session, const std::string &addr_string);
    void stop(session_t &session);
    void graceful_stop(session_t& session);
    void join(session_t &session);
    state_e state(session_t &session);
    inline bool send(session_t& session, const std::string_view &payload);
  }  // namespace session
}  // namespace stream
