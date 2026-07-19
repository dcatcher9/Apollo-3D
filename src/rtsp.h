/**
 * @file src/rtsp.h
 * @brief Declarations for RTSP streaming.
 */
#pragma once

// standard includes
#include <atomic>
#include <cstdint>
#include <memory>
#include <list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// local includes
#include "crypto.h"
#include "thread_safe.h"

#ifdef _WIN32
  #include <windows.h>
#endif

// Resolve circular dependencies
namespace stream {
  struct session_t;
}

namespace rtsp_stream {
  constexpr auto RTSP_SETUP_PORT = 21;

  namespace detail {
    enum class announce_int_field {
      audio_channels,
      audio_channel_mask,
      audio_packet_duration,
      audio_quality,
      control_protocol,
      feature_flags,
      audio_qos,
      video_qos,
      encryption_flags,
      viewport_dimension,
      max_fps,
      client_refresh_x100,
      bitrate_kbps,
      configured_bitrate_kbps,
      slices_per_frame,
      reference_frames,
      encoder_csc_mode,
      video_format,
      binary_option,
    };

    std::optional<int> parse_announce_int(announce_int_field field, std::string_view value);
    int validated_client_refresh_x100(int announced_fps, int client_refresh_x100);
    int calculate_warp_bitrate_factor(int announced_fps, int session_fps);
    bool is_safe_encoder_bitrate(std::int64_t bitrate_kbps);
    int apply_packet_size_limit(int client_packet_size, int configured_limit);
  }  // namespace detail

  struct launch_session_t {
    uint32_t id;

    crypto::aes_t gcm_key;
    crypto::aes_t iv;

    std::string av_ping_payload;
    uint32_t control_connect_data;

    std::string device_name;
    std::string unique_id;
    crypto::PERM perm;

    bool input_only;
    bool host_audio;
    int width;
    int height;
    int fps;
    int gcmap;
    int surround_info;
    std::string surround_params;
    bool enable_hdr;
    bool enable_sops;
    bool virtual_display;
    uint32_t scale_factor;

    std::optional<crypto::cipher::gcm_t> rtsp_cipher;
    std::string rtsp_url_scheme;
    uint32_t rtsp_iv_counter;

    std::list<crypto::command_entry_t> client_do_cmds;
    std::list<crypto::command_entry_t> client_undo_cmds;

  #ifdef _WIN32
    GUID display_guid{};
  #endif
  };

  void launch_session_raise(std::shared_ptr<launch_session_t> launch_session);

  /**
   * @brief Clear state for the specified launch session.
   * @param launch_session_id The ID of the session to clear.
   */
  void launch_session_clear(uint32_t launch_session_id);

  /**
   * @brief Get the number of active sessions.
   * @return Count of active sessions.
   */
  int session_count();

  std::vector<std::shared_ptr<stream::session_t>>
  find_sessions(std::string_view uuid);

  struct client_policy_stop_t {
    std::shared_ptr<stream::session_t> session;
    std::uint64_t generation;
  };

  struct client_policy_publication_t {
    bool accepted {false};
    std::vector<client_policy_stop_t> stops;
  };

  /**
   * Record the latest authorization policy and update session permissions without shutting down.
   * Callers may invoke this while holding their authorization-state lock, then complete the
   * potentially re-entrant shutdown phase after releasing it.
   */
  client_policy_publication_t stage_client_policy(
    std::string_view uuid,
    std::uint64_t generation,
    std::string name,
    crypto::PERM permissions,
    bool revoked
  );

  void complete_client_policy(client_policy_publication_t publication, bool graceful = true);

  /**
   * Publish the latest authorization policy for a client and apply it to all active sessions.
   * Older generations are ignored, and future session insertion observes the same policy.
   */
  bool publish_client_policy(
    std::string_view uuid,
    std::uint64_t generation,
    std::string name,
    crypto::PERM permissions,
    bool revoked
  );

  std::list<std::string>
  get_all_session_uuids();

#ifdef SUNSHINE_TESTS
  bool insert_session_for_test(const std::shared_ptr<stream::session_t> &session);
  void remove_session_for_test(const std::shared_ptr<stream::session_t> &session);
#endif

  /**
   * @brief Terminates all running streaming sessions.
   */
  void terminate_sessions();

  /**
   * @brief Runs the RTSP server loop.
   */
  void start();
}  // namespace rtsp_stream
