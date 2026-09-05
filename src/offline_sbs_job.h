/**
 * @file src/offline_sbs_job.h
 * @brief Bounded host-side lifecycle for native offline SBS evaluation/conversion jobs.
 *
 * This service deliberately owns only job admission, durable state, and child-process
 * lifetime. The conversion policy and scene planner live in the native offline worker.
 * Keeping that boundary explicit prevents the Web UI thread, the live-stream task pool,
 * and the offline GPU workload from sharing a scheduler.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace offline_sbs {
  enum class operation_e {
    evaluate,
    convert,
  };

  enum class output_location_e {
    // Compatibility scope for jobs persisted before outputs followed the source.
    legacy_managed_exports,
    input_directory,
  };

  enum class cache_budget_policy_e {
    fail,
    split,
  };

  enum class browse_type_e {
    file,
    directory,
    any,
  };

  enum class job_state_e {
    queued,
    running,
    publishing,
    canceling,
    canceled,
    complete,
    failed,
    interrupted,
  };

  enum class error_code_e {
    none,
    not_initialized,
    invalid_request,
    busy,
    not_found,
    conflict,
    unavailable,
    io_error,
  };

  struct create_request_t {
    std::filesystem::path input_path;
    operation_e operation = operation_e::evaluate;

    // The manager places this basename beside the canonical input file. HTTP clients
    // never select or supply an arbitrary output directory.
    std::string output_name;
    std::string codec = "hevc_nvenc";

    std::uint64_t scene_cache_max_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    cache_budget_policy_e cache_budget_policy = cache_budget_policy_e::fail;
  };

  struct browse_request_t {
    // An absent or empty path requests the bounded local-drive view on Windows.
    std::optional<std::filesystem::path> path;
    browse_type_e type = browse_type_e::any;
  };

  struct service_config_t {
    std::filesystem::path state_root;
    std::filesystem::path worker_root;
    // Retained for schema-1 publication recovery and quota accounting. New jobs
    // publish beside their canonical input and never use this as a destination.
    std::filesystem::path exports_root;
    std::filesystem::path sunshine_executable;
    std::filesystem::path sunshine_config;

    // A trusted host configuration value, never a per-request HTTP field. If empty,
    // discovery is restricted to the Sunshine installation directory.
    std::optional<std::filesystem::path> ffmpeg_executable;
    std::optional<std::filesystem::path> ffprobe_executable;

    // The interactive account selected when the service roots were derived.
    // Production binds every impersonated filesystem action and child launch to
    // this SID/UID so a fast-user switch cannot cross offline workspaces.
    std::optional<std::string> expected_user_id;

    std::size_t max_retained_jobs = 64;
    std::uint64_t max_retained_artifact_bytes =
      4ull * 1024ull * 1024ull * 1024ull;
    std::chrono::milliseconds process_poll_interval {200};
    bool probe_media_tools = true;
#ifdef SUNSHINE_TESTS
    // Focused lifecycle tests can exercise the supported "analysis available,
    // conversion unavailable" state without launching a real NVENC probe.
    std::optional<std::vector<std::string>> test_probed_codecs;
    // Simulates a transient Windows failure after the final hard link is
    // created but before the exact staging link is retired. Recovery tests use
    // this to prove the durable publish identity can safely finish cleanup.
    bool test_force_staging_disposition_failure = false;
#endif
  };

  struct progress_t {
    std::string phase = "queued";
    std::uint64_t processed_frames = 0;
    std::optional<std::uint64_t> total_frames;
    std::optional<double> source_time_seconds;
    std::optional<double> source_duration_seconds;
    std::optional<std::uint64_t> scene_count;
    nlohmann::json current_scene;
    nlohmann::json scene_decisions = nlohmann::json::array();
  };

  struct job_snapshot_t {
    std::string id;
    job_state_e state = job_state_e::queued;
    operation_e operation = operation_e::evaluate;
    std::filesystem::path input_path;
    std::optional<std::filesystem::path> output_path;
    std::optional<output_location_e> output_location;
    std::string codec;
    std::uint64_t scene_cache_max_bytes = 0;
    cache_budget_policy_e cache_budget_policy = cache_budget_policy_e::fail;
    progress_t progress;
    std::int64_t created_at_unix_ms = 0;
    std::optional<std::int64_t> started_at_unix_ms;
    std::optional<std::int64_t> ended_at_unix_ms;
    std::string error;
    nlohmann::json worker_result;

    [[nodiscard]] nlohmann::json json() const;
  };

  struct service_reply_t {
    bool ok = false;
    error_code_e code = error_code_e::none;
    std::string error;
    std::optional<job_snapshot_t> job;

    [[nodiscard]] nlohmann::json json() const;
  };

  struct scene_audit_reply_t {
    bool ok = false;
    error_code_e code = error_code_e::none;
    std::string error;
    nlohmann::json audit;
    // Paged downloads preserve the authenticated file bytes, including whitespace,
    // so external digest checks match the manifest and worker attestations.
    std::optional<std::string> serialized_artifact;

    [[nodiscard]] nlohmann::json json() const;
  };

  struct browse_reply_t {
    bool ok = false;
    error_code_e code = error_code_e::none;
    std::string error;
    std::optional<std::filesystem::path> path;
    std::optional<std::filesystem::path> parent;
    nlohmann::json entries = nlohmann::json::array();
    bool truncated = false;

    [[nodiscard]] nlohmann::json json() const;
  };

  /**
   * Immutable contract passed to one native child worker.
   *
   * The worker must publish progress/result JSON atomically at the supplied paths. A
   * successful conversion writes only staging_output; the service performs the final
   * no-overwrite rename after the child exits and its result contract is validated.
   */
  struct worker_context_t {
    std::string job_id;
    operation_e operation = operation_e::evaluate;
    std::filesystem::path input_path;
    std::optional<std::filesystem::path> final_output;
    std::optional<std::filesystem::path> staging_output;
    std::filesystem::path job_directory;
    std::filesystem::path result_directory;
    std::filesystem::path worker_spec;
    // SHA-256 of the exact worker-spec bytes written by the manager. The digest is
    // carried on the trusted command line so a user-writable spec cannot be swapped
    // between admission and child parsing.
    std::string worker_spec_sha256;
    std::filesystem::path worker_progress;
    std::filesystem::path worker_result;
    std::filesystem::path worker_log;
    std::filesystem::path sunshine_executable;
    std::filesystem::path sunshine_config;
    std::filesystem::path ffmpeg_executable;
    std::string ffmpeg_version;
    std::filesystem::path ffprobe_executable;
    std::string ffprobe_version;
    std::string codec;
    std::uint64_t scene_cache_max_bytes = 0;
    cache_budget_policy_e cache_budget_policy = cache_budget_policy_e::fail;
  };

  struct worker_outcome_t {
    bool completed = false;
    bool canceled = false;
    std::string error;
    nlohmann::json result;
  };

  using progress_callback_t = std::function<void(progress_t)>;
  using worker_runner_t = std::function<worker_outcome_t(
    const worker_context_t &,
    std::stop_token,
    const progress_callback_t &
  )>;

  /**
   * One-active-job manager. Construction does not touch disk; call start() explicitly.
   */
  class job_service_t {
  public:
    explicit job_service_t(service_config_t config, worker_runner_t runner = {});
    ~job_service_t();

    job_service_t(const job_service_t &) = delete;
    job_service_t &operator=(const job_service_t &) = delete;

    bool start(std::string &error);
    void shutdown();

    [[nodiscard]] service_reply_t create(const create_request_t &request);
    [[nodiscard]] service_reply_t cancel(std::string_view id);
    [[nodiscard]] service_reply_t clear(std::string_view id);
    [[nodiscard]] service_reply_t get(std::string_view id) const;
    [[nodiscard]] scene_audit_reply_t scene_audit(std::string_view id, std::optional<std::uint32_t> page = std::nullopt) const;
    [[nodiscard]] browse_reply_t browse(const browse_request_t &request) const;
    [[nodiscard]] std::vector<job_snapshot_t> list() const;
    [[nodiscard]] bool has_active_job() const;
    [[nodiscard]] std::optional<std::string> active_job_id() const;
    [[nodiscard]] nlohmann::json capabilities() const;

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };

  /**
   * Build the production service configuration from Sunshine's executable/config/appdata.
   * Durable state is partitioned under the active user's stable identity, while worker/export
   * files use that account's LocalAppData. FFmpeg discovery remains installation-local unless
   * the caller sets the trusted override.
   */
  service_config_t default_service_config();

  /**
   * Process-global facade used by the HTTPS API and main shutdown path.
   */
  bool initialize(service_config_t config, std::string &error);
  void shutdown();
  service_reply_t create(const create_request_t &request);
  service_reply_t cancel(std::string_view id);
  service_reply_t clear(std::string_view id);
  service_reply_t get(std::string_view id);
  scene_audit_reply_t scene_audit(std::string_view id, std::optional<std::uint32_t> page = std::nullopt);
  browse_reply_t browse(const browse_request_t &request);
  std::vector<job_snapshot_t> list();
  bool has_active_job();
  nlohmann::json capabilities();

#ifdef SUNSHINE_TESTS
  struct bounded_worker_log_test_result_t {
    std::string retained;
    bool limit_exceeded = false;
    bool write_failed = false;
  };

  /**
   * Feed arbitrary read chunk boundaries through the exact production log
   * accumulator. This keeps strict-cap and deterministic-output tests independent
   * from Windows token/process-launch availability.
   */
  bounded_worker_log_test_result_t bound_worker_log_for_test(
    const std::vector<std::string> &chunks
  );
  bounded_worker_log_test_result_t capture_worker_log_pipe_for_test(
    const std::filesystem::path &output,
    const std::vector<std::string> &chunks,
    std::string &error
  );
  bool abort_worker_log_pipe_with_open_writer_for_test(
    const std::filesystem::path &output,
    std::string &error
  );
  std::size_t native_worker_log_max_bytes_for_test();

  std::string build_worker_command_for_test(
    const std::filesystem::path &sunshine_executable,
    const std::filesystem::path &sunshine_config,
    const std::filesystem::path &worker_spec,
    std::string_view worker_spec_sha256
  );
#ifdef _WIN32
  bool windows_publish_identity_attributes_accepted_for_test(
    std::uint32_t attributes,
    std::uint64_t size
  );
#endif
#endif
}  // namespace offline_sbs
