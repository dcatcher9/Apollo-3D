/**
 * @file src/offline_sbs_worker_ffmpeg.h
 * @brief Worker-private FFmpeg children: raw source decoder, NVENC encoder, and final mux.
 *
 * The decoder and encoder exchange exact raw frames over private bounded pipes; the encoder's
 * filter graph restores the source timeline exactly. The final mux copies the encoded video and
 * the source's supported auxiliary streams, which the inventory checks then verify.
 */
#pragma once

#include "offline_frame_transport.h"
#include "offline_sbs_worker.h"
#include "offline_sbs_worker_io.h"
#include "offline_sbs_worker_probe.h"

#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace offline_sbs::worker_detail {
  class streaming_decoder_t {
  public:
    streaming_decoder_t(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const fs::path &log_path
    );
    [[nodiscard]] const std::string &format() const;
    raw_frame read_next(const frame_timing_t &timing);
    void finish();
    void abort() noexcept;

  private:
    const media_contract_t &media_;
    std::string format_;
    child_process_t process_;
    std::uint64_t next_sequence_ = 1;
  };

  std::string codec_name_for_request(const std::string &codec);

  class whole_clip_encoder_t {
  public:
    whole_clip_encoder_t(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const fs::path &work,
      const std::uint32_t sbs_width,
      const std::uint32_t sbs_height,
      const bool asynchronous
    );
    whole_clip_encoder_t(const whole_clip_encoder_t &) = delete;
    whole_clip_encoder_t &operator=(const whole_clip_encoder_t &) = delete;
    ~whole_clip_encoder_t();
    void publish(
      const std::uint64_t sequence,
      raw_frame frame
    );
    void finish();
    void abort(const std::string &) noexcept;
    [[nodiscard]] const fs::path &encoded_video() const;

  private:
    void write_frame(const raw_frame &frame);
    void consume() noexcept;
    std::uint64_t next_sequence_ = 1;
    child_process_t process_;
    fs::path encoded_video_;
    bool asynchronous_ = false;
    std::thread consumer_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<raw_frame> pending_;
    bool active_ = false;
    bool input_finished_ = false;
    bool stopping_ = false;
    std::string error_;
    bool finished_ = false;
  };

  std::string output_container(const fs::path &path);

  void validate_output_time_base(
    const worker_spec_t &spec,
    const media_contract_t &media
  );

  void validate_stream_preflight(
    const worker_spec_t &spec,
    const stream_inventory_t &inventory
  );

  nlohmann::json validate_stream_inventory(
    const stream_inventory_t &source,
    const stream_inventory_t &output,
    std::uint64_t tolerance_actual_ticks
  );

  std::vector<std::string> build_mux_command(
    const worker_spec_t &spec,
    const media_contract_t &media,
    const stream_inventory_t &source_inventory,
    const fs::path &encoded_video
  );

  void mux_encoded_video(
    const worker_spec_t &spec,
    const media_contract_t &media,
    const stream_inventory_t &source_inventory,
    const fs::path &encoded_video,
    const fs::path &log_path
  );
}  // namespace offline_sbs::worker_detail
