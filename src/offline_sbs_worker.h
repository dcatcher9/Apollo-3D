/**
 * @file src/offline_sbs_worker.h
 * @brief Native causal offline SBS conversion worker.
 *
 * The long-lived Sunshine process launches this entry point as a separate process.  The
 * worker deliberately has no Python dependency: FFprobe establishes an exact source
 * contract, FFmpeg supplies the primary bounded streaming decoder, and the native SBS harness runs
 * the same causal estimator and renderer used online in one unpaced source-order pass.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace offline_sbs {
  enum class media_color_e {
    sdr,
    hdr_pq,
    hdr_hlg,
  };

  struct rational_t {
    std::int64_t numerator = 0;
    std::int64_t denominator = 1;

    [[nodiscard]] double seconds(std::int64_t ticks) const;
  };

  struct frame_timing_t {
    std::uint64_t sequence = 0;
    std::int64_t pts = 0;
    std::int64_t duration = 0;
  };

  struct media_contract_t {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::string codec_name;
    std::string pixel_format;
    std::string color_range;
    std::string color_space;
    std::string color_transfer;
    std::string color_primaries;
    media_color_e color = media_color_e::sdr;
    rational_t time_base;
    std::vector<frame_timing_t> frames;
    nlohmann::json mastering_display;
    nlohmann::json content_light_level;
    std::vector<std::string> static_side_data_types;
    std::vector<std::string> dropped_nonsemantic_side_data_types;

    [[nodiscard]] std::int64_t first_pts() const;
    [[nodiscard]] std::int64_t end_pts_exclusive() const;
    [[nodiscard]] double duration_seconds() const;
    [[nodiscard]] bool variable_frame_rate() const;
  };

  struct worker_spec_t {
    std::string job_id;
    std::string operation;
    std::filesystem::path input_path;
    std::filesystem::path job_directory;
    std::filesystem::path result_directory;
    std::filesystem::path progress_path;
    std::filesystem::path result_path;
    std::optional<std::filesystem::path> staging_output;
    std::filesystem::path sunshine_executable;
    std::filesystem::path sunshine_config;
    std::filesystem::path ffmpeg_executable;
    std::filesystem::path ffprobe_executable;
    std::string codec;
    // Set only by read_authenticated_worker_spec(). It is the digest supplied on the
    // trusted parent command line, not a field trusted from the JSON document.
    std::string authenticated_spec_sha256;
    std::uint64_t transient_raster_hard_cap_bytes = 0;
  };

  /**
   * Parse and validate a job-manager-owned worker specification.
   *
   * Paths must be absolute.  Only the two NVENC codecs exposed by the Web UI are accepted.
   */
  worker_spec_t parse_worker_spec(const nlohmann::json &value);

  /**
   * Bounded-read a worker specification, authenticate its exact bytes with SHA-256,
   * then parse it. Authentication deliberately precedes JSON parsing and path use.
   */
  worker_spec_t read_authenticated_worker_spec(
    const std::filesystem::path &path,
    std::string_view expected_sha256
  );

  /**
   * Conservative residency for the non-overlapped conversion handoff: one decoded source
   * artifact plus one tightly packed SBS CPU snapshot, serializer scratch reservation, and
   * worst-case serialized SBS artifact. The harness uses this before allocating its first SBS
   * output snapshot so a low transient-raster cap fails closed before conversion begins.
   */
  std::uint64_t offline_single_slot_raster_bound(
    std::uint64_t source_raster_bytes,
    std::uint32_t sbs_width,
    std::uint32_t sbs_height,
    bool hdr
  );

  /**
   * Validate the native harness's selected-file/full-frame source-scope attestation.
   *
   * Offline evaluation and conversion must not observe the active window or use either live
   * window-region ROI authority. The worker rejects missing, extended, or contradictory
   * attestations before it can publish a conversion.
   */
  bool offline_full_frame_source_scope_is_valid(
    const nlohmann::json &value
  ) noexcept;

  /**
   * Parse an already-materialized FFprobe stream/frame document.
   *
   * This is intentionally strict.  Rotation, transformed dimensions, dynamic HDR,
   * Dolby Vision, ambiguous high-bit-depth SDR, and missing/inconsistent HDR metadata are
   * rejected before either decoder or TensorRT starts. Production probing applies the same
   * builder incrementally so its full per-frame JSON never becomes a resident DOM.
   */
  media_contract_t parse_ffprobe_contract(
    const nlohmann::json &value,
    bool require_source_packet_durations = true
  );

  /**
   * Verify that a decoded/encoded presentation timeline covers the source without drift.
   * `actual` may use a different time base. Every absolute PTS, duration, and the total
   * presentation duration must remain within one `actual` time-base tick.
   */
  void validate_timeline_equivalence(
    const media_contract_t &source,
    const media_contract_t &actual
  );

  /**
   * Reject timelines whose generated AVExpr would require inexact integer intermediates.
   */
  void validate_avexpr_timeline_exactness(const media_contract_t &media);

  /**
   * Select the lowest defined AV1 level that can represent NVENC's aligned packed
   * SBS coded raster and the fastest exact frame interval in `media`.
   *
   * NVENC's automatic mode can emit reserved 7.x level indices that common AV1
   * decoders reject.  Keeping this calculation native and deterministic both avoids
   * that driver-dependent result and prevents small clips from being over-declared as
   * level 6.x.
   */
  std::string select_av1_level(
    const media_contract_t &media,
    std::uint32_t encoded_width,
    std::uint32_t encoded_height
  );

  /**
   * Build the deterministic encoder-specific FFmpeg arguments used by conversion.
   * Exposed as a pure function so codec/profile/level contracts can be unit tested
   * without starting FFmpeg or the GPU pipeline.
   */
  std::vector<std::string> build_codec_arguments(
    const worker_spec_t &spec,
    const media_contract_t &media,
    bool hdr,
    std::uint32_t encoded_width,
    std::uint32_t encoded_height
  );

  std::vector<std::string> build_ffprobe_command(
    const worker_spec_t &spec,
    const std::filesystem::path &media_path
  );

  std::vector<std::string> build_decoder_command(
    const worker_spec_t &spec,
    const media_contract_t &media
  );

#ifdef SUNSHINE_TESTS
  bool native_stdout_pipe_error_is_eof_for_test(
    std::uint32_t error
  ) noexcept;
  bool adaptive_trace_flags_valid_for_test(
    float cut_flags,
    std::uint32_t analysis_flags
  );
  std::uint64_t retained_timing_frame_limit_for_test();
  bool can_retain_timing_frame_for_test(std::uint64_t retained_frames);
  std::uintmax_t stream_inventory_probe_byte_limit_for_test();
  std::uintmax_t packet_probe_byte_limit_for_test();
  std::uint64_t auxiliary_packet_limit_for_test();
  std::uint64_t retained_packet_payload_limit_for_test();
  std::size_t child_process_log_byte_limit_for_test();
  std::string bound_child_process_log_for_test(std::string_view bytes);
  std::size_t offline_source_pipeline_capacity_for_test();
  std::size_t offline_encoder_pipeline_capacity_for_test();
  bool can_retain_auxiliary_packets_for_test(
    std::uint64_t retained_packets,
    std::uint64_t additional_packets
  );
  bool can_consume_packet_probe_bytes_for_test(
    std::uintmax_t consumed_bytes,
    std::uintmax_t additional_bytes
  );
  std::vector<std::string> build_mux_command_for_test(
    const worker_spec_t &spec,
    const media_contract_t &media,
    const nlohmann::json &source_inventory,
    const std::filesystem::path &encoded_video
  );

  struct streaming_probe_test_result_t {
    media_contract_t media;
    std::size_t peak_retained_frame_descriptors = 0;
    std::uintmax_t probe_bytes = 0;
    std::size_t compact_contract_bytes = 0;
  };

  struct streaming_packet_probe_test_result_t {
    std::uint64_t packet_count = 0;
    std::size_t peak_retained_packet_descriptors = 0;
    std::uintmax_t probe_bytes = 0;
  };

  /**
   * Exercise the production incremental FFprobe frame parser without launching FFprobe.
   *
   * `stream` is the already bounded `-show_streams` descriptor and `frames_path` is a
   * `-show_frames -of json=compact=1` document. The production helper consumes and removes
   * the frame document on both success and failure.
   */
  streaming_probe_test_result_t
  parse_and_remove_ffprobe_frames_for_test(
    const nlohmann::json &stream,
    const std::filesystem::path &frames_path,
    bool require_source_packet_durations = true
  );

  /**
   * Exercise the production bounded SAX parser for `-show_packets` output.
   *
   * No packet JSON DOM or raw probe file survives the call. `max_packets` may be lowered
   * by tests to exercise the same fail-closed count boundary without constructing a
   * two-million-record fixture.
   */
  streaming_packet_probe_test_result_t
  parse_and_remove_ffprobe_packets_for_test(
    const std::filesystem::path &packets_path,
    std::uint64_t max_packets
  );

  void validate_windows_command_line_capacity_for_test(
    const std::vector<std::string> &arguments
  );
#endif

  /**
   * Sunshine command-mode entry point. `argv[0]` is the worker-spec path and
   * `argv[1]` is the manager-computed SHA-256 of its exact bytes.
   */
  int run(int argc, char **argv);
}  // namespace offline_sbs
