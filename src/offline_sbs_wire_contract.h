/**
 * @file src/offline_sbs_wire_contract.h
 * @brief Typed JSON wire contracts shared by the native offline SBS manager and worker.
 */
#pragma once

#include "offline_scene_planner.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace offline_sbs::wire {
  inline constexpr std::uint32_t max_raster_dimension = 16384u;

  class contract_error: public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
  };

  /** Parse one JSON document while rejecting duplicate object keys. */
  nlohmann::json parse_json_without_duplicate_keys(std::string_view bytes);

  /** Return whether a value is exactly one lowercase SHA-256 hexadecimal digest. */
  [[nodiscard]] bool valid_sha256_hex(std::string_view value) noexcept;

  struct worker_spec_contract_t {
    std::string job_id;
    std::string operation;
    std::string input_path;
    std::string job_directory;
    std::string result_directory;
    std::string progress_path;
    std::string result_path;
    std::optional<std::string> staging_output;
    std::string sunshine_executable;
    std::string sunshine_config;
    std::string ffmpeg_path;
    std::string ffmpeg_version;
    std::string ffprobe_path;
    std::string ffprobe_version;
    std::string codec;
    std::uint64_t transient_raster_hard_cap_bytes = 0;
  };

  nlohmann::json to_json(const worker_spec_contract_t &value);
  worker_spec_contract_t parse_worker_spec_contract(const nlohmann::json &value);

  struct observation_timeline_contract_t {
    std::uint32_t schema = 0;
    std::string timestamp_unit;
    std::uint64_t count = 0;
    std::string sha256;
  };

  struct whole_clip_resolved_runtime_t {
    std::string model;
    std::optional<std::string> frame_transport;
    std::string model_url;
    double pop_strength = 1.0;
    int depth_width = 0;
    int depth_height = 0;
    int depth_reuse_interval = 0;
    bool cuda_graph = false;
    bool parallax_v2_shadow = false;
    bool parallax_v2_render = false;
    bool parallax_v2_live = false;
    bool cuda_graph_captured = false;
    bool simulate_hdr = false;
    double hdr_scale = 1.0;
    std::string input_color_space;
    std::string input_frame_format;
    std::string input_texture_format;
    std::string input_transfer;
    std::string input_primaries;
    std::optional<double> input_reference_white_nits;
    std::string packed_texture_format;
    std::string packed_color_space;
    std::string packed_transfer;
    std::string packed_primaries;
    std::optional<double> packed_reference_white_nits;
    std::string output_frame_format;
    std::string output_transfer;
    std::string output_primaries;
    std::optional<double> output_reference_white_nits;
    std::string output_row_order;
    std::optional<std::string> output_ffmpeg_pixel_format;
    double output_scale = 1.0;
    int requested_eye_width = 0;
    int requested_eye_height = 0;
    int configured_max_encode_width = 0;
    int resolved_max_output_width = 0;
    int input_limit = 0;
    std::uint64_t output_every = 0;
    bool follow_mode = false;
    std::optional<std::string> follow_format;
    std::optional<std::uint64_t> follow_count_bound;
    std::optional<std::uint64_t> follow_producer_frame_count;
    std::optional<std::string> follow_frame_pattern;
    std::optional<std::uint64_t> follow_first_sequence;
    std::optional<std::uint32_t> follow_poll_interval_ms;
    std::optional<std::string> follow_done_sentinel;
    std::optional<std::string> follow_failed_sentinel;
    std::optional<std::string> follow_progress_file;
    bool follow_native_input_deletion = false;
    bool follow_atomic_sbs_publication = false;
    std::uint32_t output_eye_width = 0;
    std::uint32_t output_eye_height = 0;
    std::uint32_t output_sbs_width = 0;
    std::uint32_t output_sbs_height = 0;
    double content_scale_x = 0.0;
    double content_scale_y = 0.0;
    bool scene_cache_write = false;
    bool scene_cache_replay = false;
    std::optional<std::uint32_t> scene_cache_contract_schema;
    std::optional<std::uint32_t> scene_plan_schema;
    std::optional<std::string> scene_plan_version;
    std::optional<std::uint64_t> scene_start_sequence;
    std::optional<std::uint64_t> scene_end_sequence_exclusive;
    std::optional<std::string> scene_cache_status_at_replay_start;
    std::optional<std::uint64_t> scene_cache_processed_count_at_replay_start;
  };

  struct whole_clip_adaptive_state_t {
    std::string transport;
    std::optional<std::string> header_file;
    std::optional<std::string> frame_file;
    std::optional<std::string> file;
    bool retained_history = false;
    std::optional<std::uint32_t> schema;
    std::optional<std::string> capture;
    std::uint64_t frame_count = 0;
  };

  struct whole_clip_sbs_t {
    bool enabled = false;
    std::optional<std::string> file_pattern;
    std::uint64_t frame_count = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::string frame_format;
    std::string transfer;
    std::string primaries;
    std::optional<double> reference_white_nits;
    std::string row_order;
    std::optional<std::string> ffmpeg_pixel_format;
    bool atomic_publication = false;
  };

  nlohmann::json to_json(const whole_clip_sbs_t &value);
  whole_clip_sbs_t parse_whole_clip_sbs(const nlohmann::json &value);

  struct whole_clip_contract_t {
    std::optional<observation_timeline_contract_t> observation_timeline;
    std::string artifact_mode;
    std::string inference_mode;
    bool depth_inference_enabled = false;
    std::uint64_t scheduled_depth_update_count = 0;
    std::uint64_t tensorrt_enqueue_count = 0;
    std::string depth_provenance;
    std::string pipeline_state_provenance;
    std::string model;
    std::uint64_t source_frame_count = 0;
    std::uint32_t source_width = 0;
    std::uint32_t source_height = 0;
    std::uint64_t source_first_sequence = 0;
    int depth_reuse_interval = 0;
    whole_clip_resolved_runtime_t resolved_runtime;
    whole_clip_adaptive_state_t adaptive_state;
    whole_clip_sbs_t sbs;
  };

  nlohmann::json to_json(const whole_clip_contract_t &value);
  whole_clip_contract_t parse_whole_clip_contract(const nlohmann::json &value);

  struct replay_result_contract_t {
    std::uint64_t scene_id = 0;
    std::uint64_t start_sequence = 0;
    std::uint64_t end_sequence_exclusive = 0;
    std::string inference_mode;
    bool depth_inference_enabled = false;
    std::uint64_t scheduled_depth_update_count = 0;
    std::uint64_t tensorrt_enqueue_count = 0;
    whole_clip_sbs_t sbs;
  };

  nlohmann::json to_json(const replay_result_contract_t &value);
  replay_result_contract_t parse_replay_result_contract(
    const nlohmann::json &value
  );

  struct worker_result_source_t {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t frame_count = 0;
    double duration_seconds = 0.0;
    bool variable_frame_rate = false;
    std::string color_mode;
    std::string contract_path;
  };

  struct worker_result_cache_t {
    std::uint64_t hard_cap_bytes = 0;
    std::uint64_t peak_bytes = 0;
    std::uint64_t analysis_source_raster_bytes = 0;
    std::uint64_t peak_live_raster_bytes = 0;
    std::uint64_t peak_cache_plus_raster_bytes = 0;
    std::uint64_t remaining_bytes = 0;
  };

  /**
   * Terminal worker result. Timeline and staging identity remain JSON because they are separately
   * owned contracts; shared scene and whole-clip evidence uses typed codecs here.
   */
  struct worker_result_contract_t {
    std::string job_id;
    std::string operation;
    std::string codec;
    std::string worker_spec_sha256;
    worker_result_source_t source;
    std::vector<scene_plan_t> scenes;
    std::string scene_audit_path;
    whole_clip_contract_t analysis_contract;
    std::vector<replay_result_contract_t> replay_contracts;
    worker_result_cache_t cache;
    nlohmann::json timeline_contract;
    std::optional<nlohmann::json> staging_identity;
    std::optional<std::string> output_path;
    // Present for schema 3: scenes contains only the final bounded tail.
    std::optional<std::uint64_t> total_scene_count;
    std::optional<std::string> scene_audit_manifest_sha256;
  };

  nlohmann::json to_json(const worker_result_contract_t &value);
  worker_result_contract_t parse_worker_result_contract(const nlohmann::json &value);

  nlohmann::json worker_failure_json(
    std::string_view job_id,
    std::string_view error
  );

  struct scene_range_t {
    std::uint64_t start_sequence = 0;
    std::uint64_t end_sequence_exclusive = 0;
  };

  struct scene_plan_contract_t {
    std::vector<scene_range_t> scenes;
  };

  nlohmann::json to_json(const scene_plan_contract_t &value);
  scene_plan_contract_t parse_scene_plan_contract(const nlohmann::json &value);

  struct scene_cache_source_t {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::string frame_format;
    std::string texture_format;
    std::string color_space;
  };

  struct scene_cache_render_t {
    std::string model;
    std::string model_url;
    double pop_strength = 1.0;
    bool simulate_hdr = false;
    double hdr_scale = 1.0;
    int depth_reuse_interval = 1;
    int requested_eye_width = 0;
    int requested_eye_height = 0;
    double output_scale = 1.0;
    int resolved_max_output_width = 0;
  };

  struct scene_cache_packed_sbs_t {
    std::uint32_t eye_width = 0;
    std::uint32_t eye_height = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::string texture_format;
    std::string frame_format;
    std::string file_extension;
  };

  struct scene_cache_contract_t {
    std::string status;
    scene_cache_source_t source;
    std::uint32_t depth_width = 0;
    std::uint32_t depth_height = 0;
    scene_cache_render_t render;
    scene_cache_packed_sbs_t packed_sbs;
    std::uint64_t processed_count = 0;
    std::optional<std::uint64_t> frame_count;
  };

  nlohmann::json to_json(const scene_cache_contract_t &value);
  scene_cache_contract_t parse_scene_cache_contract(const nlohmann::json &value);

  nlohmann::json boundary_audit_json(const boundary_audit_t &value);
  boundary_audit_t parse_boundary_audit(const nlohmann::json &value);
  nlohmann::json scene_evidence_json(const scene_evidence_t &value);
  scene_evidence_t parse_scene_evidence(const nlohmann::json &value);
  nlohmann::json scene_record_json(const scene_plan_t &value);
  scene_plan_t parse_scene_record(const nlohmann::json &value);

  struct scene_audit_page_descriptor_t {
    std::uint32_t index = 0;
    std::uint64_t first_scene_id = 0;
    std::uint64_t scene_count = 0;
    std::uint64_t boundary_count = 0;
    std::uint64_t start_sequence = 0;
    std::uint64_t end_sequence_exclusive = 0;
    std::uint64_t bytes = 0;
    std::string sha256;
  };

  struct scene_audit_page_t {
    std::uint32_t index = 0;
    std::vector<scene_plan_t> scenes;
    std::vector<boundary_audit_t> boundary_revisions;
  };
  std::string scene_audit_page_filename(std::uint32_t index);
  nlohmann::json to_json(const scene_audit_page_t &value);
  scene_audit_page_t parse_scene_audit_page(const nlohmann::json &value);
  void validate_scene_audit_page(const scene_audit_page_t &page,
                                const scene_audit_page_descriptor_t &descriptor,
                                bool complete_final_page);

  struct scene_audit_contract_t {
    std::string status;
    std::uint64_t peak_cache_bytes = 0;
    std::uint64_t analysis_source_raster_bytes = 0;
    std::uint64_t peak_live_raster_bytes = 0;
    std::uint64_t peak_cache_plus_raster_bytes = 0;
    std::uint64_t hard_cap_bytes = 0;
    nlohmann::json timeline_contract;
    std::vector<scene_plan_t> scenes;
    std::vector<boundary_audit_t> boundary_revisions;
    bool paged = false;
    std::uint64_t scene_count = 0;
    std::uint64_t boundary_count = 0;
    std::uint64_t covered_end_sequence = 1;
    std::uint64_t total_page_bytes = 0;
    std::vector<scene_audit_page_descriptor_t> pages;
  };

  nlohmann::json to_json(const scene_audit_contract_t &value);
  scene_audit_contract_t parse_scene_audit_contract(const nlohmann::json &value);

  struct job_progress_contract_t {
    std::string phase;
    std::uint64_t processed_frames = 0;
    std::optional<std::uint64_t> total_frames;
    std::optional<double> source_time_seconds;
    std::optional<double> source_duration_seconds;
    std::optional<std::uint64_t> scene_count;
    std::optional<nlohmann::json> current_scene;
    std::vector<nlohmann::json> scene_decisions;
  };

  struct job_snapshot_contract_t {
    std::string id;
    std::string state;
    std::string operation;
    std::string input_path;
    std::optional<std::string> output_path;
    std::optional<std::string> output_location;
    std::string codec;
    std::uint64_t scene_cache_max_bytes = 0;
    std::string cache_budget_policy;
    job_progress_contract_t progress;
    std::int64_t created_at_unix_ms = 0;
    std::optional<std::int64_t> started_at_unix_ms;
    std::optional<std::int64_t> ended_at_unix_ms;
    std::optional<std::string> error;
    std::optional<nlohmann::json> worker_result;
  };

  struct persisted_worker_contract_t {
    std::string spec_path;
    std::string spec_sha256;
    std::string progress_path;
    std::string result_path;
    std::string log_path;
    std::string result_directory;
    std::optional<std::string> staging_output;
    std::string ffmpeg_path;
    std::string ffmpeg_version;
    std::string ffprobe_path;
    std::string ffprobe_version;
  };

  struct persisted_job_contract_t {
    job_snapshot_contract_t snapshot;
    std::optional<std::uint64_t> retention_sequence;
    std::optional<persisted_worker_contract_t> worker;
  };

  nlohmann::json to_json(const job_snapshot_contract_t &value);
  job_snapshot_contract_t parse_job_snapshot_contract(
    const nlohmann::json &value,
    bool persisted_document
  );
  nlohmann::json to_json(const persisted_job_contract_t &value);
  persisted_job_contract_t parse_persisted_job_contract(
    const nlohmann::json &value
  );
}  // namespace offline_sbs::wire
