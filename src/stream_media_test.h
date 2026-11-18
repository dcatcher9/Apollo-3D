/**
 * @file src/stream_media_test.h
 * @brief Private loopback access to the actual media sender loops.
 */
#pragma once

#ifdef SUNSHINE_TESTS
  #include <array>
  #include <chrono>
  #include <cstddef>
  #include <cstdint>
  #include <memory>
  #include <optional>
  #include <string>
  #include <vector>

namespace stream::testing {
  enum class media_lane_e {
    video,
    audio,
  };

  struct media_sender_config_t {
    std::array<std::uint8_t, 16> key {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::chrono::steady_clock::time_point epoch {std::chrono::seconds {1000}};
    int video_packet_size = 1024;
    int video_min_fec_packets = 2;
    std::optional<int> video_fec_percentage = 0;
    std::uint32_t video_sequence = 400;
    std::uint64_t video_iv_counter = 9;
    std::uint16_t audio_sequence = 400;
    std::uint32_t audio_timestamp = 9000;
    std::uint32_t audio_key_id = 0x10203040;
    int audio_packet_duration = 5;
  };

  struct video_packet_t {
    std::vector<std::uint8_t> bytes;
    std::int64_t index = 1;
    bool idr = true;
    bool retained = false;
    bool after_invalidation = false;
    std::chrono::nanoseconds pts = std::chrono::seconds {1};
  };

  struct media_sender_state_t {
    std::uint32_t video_sequence = 0;
    std::uint64_t video_iv_counter = 0;
    std::uint16_t audio_sequence = 0;
    std::uint32_t audio_timestamp = 0;
    bool video_shutdown = false;
    bool audio_shutdown = false;
    std::string video_error;
    std::string audio_error;
  };

  /**
   * Owns private queues, channels, ephemeral IPv4 loopback sockets and actual sender workers.
   * Methods are caller-serialized; the admission gates and ready waits synchronize with workers.
   * stop() joins both workers and releases their queues before state reads.
   * close() also releases the retained channels and originating frame pool.
   * Neither global mail nor global configuration is replaced or modified.
   */
  class media_sender_fixture_t {
  public:
    explicit media_sender_fixture_t(media_sender_config_t config = {});
    ~media_sender_fixture_t();

    media_sender_fixture_t(const media_sender_fixture_t &) = delete;
    media_sender_fixture_t &operator=(const media_sender_fixture_t &) = delete;
    media_sender_fixture_t(media_sender_fixture_t &&) = delete;
    media_sender_fixture_t &operator=(media_sender_fixture_t &&) = delete;

    void start();
    bool wait_ready(media_lane_e lane, std::chrono::milliseconds timeout);
    void gate_next_admission(media_lane_e lane);
    bool wait_admission(media_lane_e lane, std::chrono::milliseconds timeout);
    void release_admission(media_lane_e lane);
    bool wait_processed(media_lane_e lane, std::size_t count, std::chrono::milliseconds timeout);

    bool enqueue_video(video_packet_t packet);
    bool enqueue_audio(std::vector<std::uint8_t> bytes);
    void deactivate(media_lane_e lane);
    void request_shutdown(media_lane_e lane);

    std::vector<std::vector<std::uint8_t>> receive(
      media_lane_e lane,
      std::size_t count,
      std::chrono::milliseconds timeout
    );

    void stop();
    void finish_senders_without_shutdown();
    void close();
    media_sender_state_t state_after_stop() const;
    std::weak_ptr<void> weak_channel(media_lane_e lane) const;
    std::weak_ptr<void> weak_queue(media_lane_e lane) const;
    std::weak_ptr<void> weak_frame_pool() const;

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace stream::testing
#endif
