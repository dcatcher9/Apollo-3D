/**
 * @file src/offline_sbs_job_worker.h
 * @brief Job-manager-private side of the isolated worker process and its media tools.
 *
 * The manager discovers and probes the trusted FFmpeg/FFprobe pair, writes the authenticated
 * worker specification, launches and supervises the worker in a hardened process group with a
 * bounded log, relays its progress file, and validates its result before a job can publish.
 */
#pragma once

#include "offline_sbs_job.h"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace offline_sbs::job_detail {
  // The operation's spelling, shared by the worker specification and result and the job record.
  const char *to_string(operation_e value);

  std::optional<std::filesystem::path> discover_media_tool(
    const service_config_t &config,
    const std::optional<std::filesystem::path> &trusted_override,
    std::string_view tool_name,
    std::string &error
  );

  std::optional<std::string> probe_media_tool_version(
    const service_config_t &config,
    const std::filesystem::path &tool,
    std::string_view tool_name,
    std::string &error
  );

  std::optional<std::string> media_tool_version_token(
    std::string_view line,
    std::string_view tool_name
  );

  std::optional<std::vector<std::string>> probe_ffmpeg_codecs(
    const service_config_t &config,
    const std::filesystem::path &ffmpeg,
    std::string &error
  );

  bool probe_ffmpeg_codec_runtime(
    const service_config_t &config,
    const std::filesystem::path &ffmpeg,
    std::string_view codec,
    std::string &error
  );

  nlohmann::json worker_spec_json(const worker_context_t &context);

  nlohmann::json compact_worker_result(
    const nlohmann::json &result,
    const worker_context_t &context
  );

  // Launch, supervise and reap one native worker; its outcome carries the validated result.
  worker_outcome_t run_native_worker(
    const service_config_t &config,
    const worker_context_t &context,
    std::stop_token stop,
    const progress_callback_t &publish_progress
  );
}  // namespace offline_sbs::job_detail
