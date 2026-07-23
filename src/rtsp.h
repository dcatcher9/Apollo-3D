/**
 * @file src/rtsp.h
 * @brief Declarations for RTSP streaming.
 */
#pragma once

// standard includes
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

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

  enum class launch_reservation_state_e : std::uint8_t {
    pending,
    claimed,
    revoked,
  };

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
    std::int64_t reserve_video_bitrate_for_fec(std::int64_t total_bitrate_kbps, int fec_percentage);
    std::int64_t calculate_video_bitrate_budget(
      std::int64_t total_bitrate_kbps,
      int fec_percentage,
      std::int64_t audio_bitrate_kbps
    );
    bool is_video_mode_supported(
      int video_format,
      int dynamic_range,
      bool hevc_sdr,
      bool hevc_hdr,
      bool av1_sdr,
      bool av1_hdr
    );
    int apply_packet_size_limit(int client_packet_size, int configured_limit);
    std::optional<std::string_view> parse_setup_stream_type(std::string_view target);
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

    bool host_audio;
    int width;
    int height;
    int fps;
    int surround_info;
    bool enable_hdr;
    bool enable_sops;
    bool virtual_display;
    uint32_t scale_factor;
    int sbs_mode = 0;

    // An accepted socket retains this shared object even after the pending queue is cleared.
    // Keeping revocation here makes teardown visible to every retained copy.
    std::atomic<launch_reservation_state_e> reservation_state {
      launch_reservation_state_e::pending
    };

    [[nodiscard]] bool try_claim_reservation() {
      auto expected = launch_reservation_state_e::pending;
      return reservation_state.compare_exchange_strong(
        expected,
        launch_reservation_state_e::claimed,
        std::memory_order_acq_rel,
        std::memory_order_acquire
      );
    }

    void revoke_reservation() {
      reservation_state.store(launch_reservation_state_e::revoked, std::memory_order_release);
    }

    void revoke_pending_reservation() {
      auto expected = launch_reservation_state_e::pending;
      reservation_state.compare_exchange_strong(
        expected,
        launch_reservation_state_e::revoked,
        std::memory_order_acq_rel,
        std::memory_order_acquire
      );
    }

    [[nodiscard]] launch_reservation_state_e reservation() const {
      return reservation_state.load(std::memory_order_acquire);
    }

    std::optional<crypto::cipher::gcm_t> rtsp_cipher;
    uint32_t rtsp_iv_counter = 0;

#ifdef _WIN32
    GUID display_guid {};
#endif
  };

  [[nodiscard]] bool launch_session_raise(std::shared_ptr<launch_session_t> launch_session);

  /** Whether the single HTTP-to-RTSP launch reservation slot is currently available. */
  [[nodiscard]] bool launch_session_available();

  /**
   * @brief Clear state for the specified launch session.
   * @param launch_session_id The ID of the session to clear.
   */
  void launch_session_clear(uint32_t launch_session_id);

  /** Cancel the pending HTTP-to-RTSP launch reservation, if any. */
  void clear_pending_launch_session();

  /** Return the active stream when it belongs to the specified client. */
  std::shared_ptr<stream::session_t> find_session(std::string_view uuid);

  /** Return the active stream's client UUID, or no value while the host is idle. */
  std::optional<std::string> active_session_uuid();

  struct client_policy_stop_t {
    std::shared_ptr<stream::session_t> session;
    std::uint64_t generation;
  };

  struct client_policy_publication_t {
    bool accepted {false};
    std::optional<client_policy_stop_t> stop;
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
   * Publish the latest authorization policy for a client and apply it to the active session.
   * Older generations are ignored, and future session insertion observes the same policy.
   */
  bool publish_client_policy(
    std::string_view uuid,
    std::uint64_t generation,
    std::string name,
    crypto::PERM permissions,
    bool revoked
  );

#ifdef SUNSHINE_TESTS
  bool insert_session_for_test(const std::shared_ptr<stream::session_t> &session);
  void remove_session_for_test(const std::shared_ptr<stream::session_t> &session);
  bool claim_launch_session_for_test(launch_session_t &launch_session);
  void finish_launch_session_for_test(launch_session_t &launch_session, bool started);
#endif

  /** Terminate the pending or active remote streaming session. */
  void terminate_session();

  /**
   * @brief Runs the RTSP server loop.
   */
  void start();
}  // namespace rtsp_stream
