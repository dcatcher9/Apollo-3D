/**
 * @file src/offline_sbs_wire_contract.cpp
 * @brief Typed JSON wire contracts shared by the native offline SBS manager and worker.
 */

#include "offline_sbs_wire_contract.h"

#include "offline_frame_transport.h"
#include "offline_sbs_contract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <string>
#include <utility>

using namespace std::literals;

namespace offline_sbs::wire {
  bool valid_sha256_hex(const std::string_view value) noexcept {
    return value.size() == 64 &&
           std::ranges::all_of(value, [](const unsigned char character) {
             return (character >= '0' && character <= '9') ||
                    (character >= 'a' && character <= 'f');
           });
  }

  namespace {
    using json = nlohmann::json;

    std::uint64_t checked_raster_bytes(
      const std::uint32_t width,
      const std::uint32_t height,
      const std::uint32_t bytes_per_pixel,
      const std::string_view context
    ) {
      if (width == 0 || height == 0 ||
          width > max_raster_dimension ||
          height > max_raster_dimension) {
        throw contract_error(
          std::string {context} + " dimensions are outside the authenticated bound"
        );
      }
      const auto pixels = static_cast<std::uint64_t>(width) * height;
      if (pixels > std::numeric_limits<std::uint64_t>::max() / bytes_per_pixel) {
        throw contract_error(std::string {context} + " byte count overflows");
      }
      return pixels * bytes_per_pixel;
    }

    template<std::size_t Size>
    void require_exact_keys(
      const json &value,
      const std::array<std::string_view, Size> &expected,
      const std::string_view context
    ) {
      if (!value.is_object() || value.size() != expected.size()) {
        throw contract_error(std::string {context} + " has missing or unknown fields");
      }
      for (const auto key : expected) {
        if (!value.contains(key)) {
          throw contract_error(
            std::string {context} + " is missing field '" + std::string {key} + "'"
          );
        }
      }
    }

    template<std::size_t Size>
    void require_keys_with_optional(
      const json &value,
      const std::array<std::string_view, Size> &expected,
      const std::string_view optional,
      const std::string_view context
    ) {
      if (!value.is_object() || value.size() != expected.size() + value.contains(optional)) {
        throw contract_error(std::string {context} + " has missing or unknown fields");
      }
      for (const auto key : expected) {
        if (!value.contains(key)) {
          throw contract_error(std::string {context} + " is missing field '" + std::string {key} + "'");
        }
      }
    }

    template<std::size_t Size>
    void require_allowed_keys(
      const json &value,
      const std::array<std::string_view, Size> &allowed,
      const std::string_view context
    ) {
      if (!value.is_object()) {
        throw contract_error(std::string {context} + " must be an object");
      }
      for (const auto &[key, field] : value.items()) {
        (void) field;
        if (std::ranges::find(allowed, key) == allowed.end()) {
          throw contract_error(
            std::string {context} + " has unknown field '" + key + "'"
          );
        }
      }
    }

    std::string required_string(
      const json &value,
      const std::string_view key,
      const std::string_view context
    ) {
      const auto &field = value.at(key);
      if (!field.is_string()) {
        throw contract_error(
          std::string {context} + " field '" + std::string {key} + "' must be a string"
        );
      }
      auto result = field.get<std::string>();
      if (result.empty()) {
        throw contract_error(
          std::string {context} + " field '" + std::string {key} + "' must not be empty"
        );
      }
      return result;
    }

    template<class Integer>
    Integer unsigned_integer(
      const json &value,
      const std::string_view key,
      const std::string_view context
    ) {
      const auto &field = value.at(key);
      if (!field.is_number_unsigned() && !field.is_number_integer()) {
        throw contract_error(
          std::string {context} + " field '" + std::string {key} + "' must be an integer"
        );
      }
      std::uint64_t raw = 0;
      if (field.is_number_unsigned()) {
        raw = field.get<std::uint64_t>();
      } else {
        const auto signed_value = field.get<std::int64_t>();
        if (signed_value < 0) {
          throw contract_error(
            std::string {context} + " field '" + std::string {key} + "' must be nonnegative"
          );
        }
        raw = static_cast<std::uint64_t>(signed_value);
      }
      if (raw > static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
        throw contract_error(
          std::string {context} + " field '" + std::string {key} + "' is out of range"
        );
      }
      return static_cast<Integer>(raw);
    }

    template<class Integer = std::int64_t>
    Integer signed_integer(
      const json &value,
      const std::string_view key,
      const std::string_view context
    ) {
      const auto &field = value.at(key);
      if (!field.is_number_integer() && !field.is_number_unsigned()) {
        throw contract_error(
          std::string {context} + " field '" + std::string {key} + "' must be signed integer"
        );
      }
      if (field.is_number_unsigned() &&
          field.get<std::uint64_t>() >
            static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
        throw contract_error(
          std::string {context} + " field '" + std::string {key} + "' is out of range"
        );
      }
      const auto raw = field.is_number_unsigned() ?
                         static_cast<std::int64_t>(field.get<std::uint64_t>()) :
                         field.get<std::int64_t>();
      if (raw < static_cast<std::int64_t>(std::numeric_limits<Integer>::min()) ||
          raw > static_cast<std::int64_t>(std::numeric_limits<Integer>::max())) {
        throw contract_error(
          std::string {context} + " field '" + std::string {key} + "' is out of range"
        );
      }
      return static_cast<Integer>(raw);
    }

    double finite_number(
      const json &value,
      const std::string_view key,
      const std::string_view context
    ) {
      if (!value.at(key).is_number()) {
        throw contract_error(
          std::string {context} + " field '" + std::string {key} + "' must be numeric"
        );
      }
      const auto result = value.at(key).get<double>();
      if (!std::isfinite(result)) {
        throw contract_error(
          std::string {context} + " field '" + std::string {key} + "' must be finite"
        );
      }
      return result;
    }

    bool required_boolean(
      const json &value,
      const std::string_view key,
      const std::string_view context
    ) {
      if (!value.at(key).is_boolean()) {
        throw contract_error(
          std::string {context} + " field '" + std::string {key} + "' must be boolean"
        );
      }
      return value.at(key).get<bool>();
    }

    template<class Value, class Reader>
    std::optional<Value> optional_value(
      const json &value,
      const std::string_view key,
      Reader &&reader
    ) {
      if (value.at(key).is_null()) {
        return std::nullopt;
      }
      return std::forward<Reader>(reader)();
    }

    template<class Value>
    json optional_json(const std::optional<Value> &value) {
      return value ? json(*value) : json(nullptr);
    }

    persisted_worker_contract_t parse_persisted_worker_fields(
      const json &value
    ) {
      require_exact_keys(
        value,
        std::array {
          "spec"sv, "spec_sha256"sv, "progress"sv, "result"sv, "log"sv,
          "result_directory"sv, "staging_output"sv, "ffmpeg_path"sv,
          "ffmpeg_version"sv, "ffprobe_path"sv, "ffprobe_version"sv,
        },
        "persisted job worker"
      );
      persisted_worker_contract_t result {
        .spec_path = required_string(value, "spec", "persisted job worker"),
        .progress_path = required_string(value, "progress", "persisted job worker"),
        .result_path = required_string(value, "result", "persisted job worker"),
        .log_path = required_string(value, "log", "persisted job worker"),
        .result_directory = required_string(
          value, "result_directory", "persisted job worker"
        ),
        .ffmpeg_path = required_string(value, "ffmpeg_path", "persisted job worker"),
        .ffmpeg_version = required_string(
          value, "ffmpeg_version", "persisted job worker"
        ),
        .ffprobe_path = required_string(value, "ffprobe_path", "persisted job worker"),
        .ffprobe_version = required_string(
          value, "ffprobe_version", "persisted job worker"
        ),
      };
      if (!value.at("spec_sha256").is_string()) {
        throw contract_error("persisted job worker spec digest must be a string");
      }
      result.spec_sha256 = value.at("spec_sha256").get<std::string>();
      if (!result.spec_sha256.empty() && !valid_sha256_hex(result.spec_sha256)) {
        throw contract_error("persisted job worker spec digest is invalid");
      }
      if (!value.at("staging_output").is_null()) {
        result.staging_output = required_string(
          value, "staging_output", "persisted job worker"
        );
      }
      return result;
    }
  }  // namespace

  nlohmann::json parse_json_without_duplicate_keys(const std::string_view bytes) {
    std::map<int, std::set<std::string>> object_keys;
    const auto callback = [&object_keys](
                            const int depth,
                            const json::parse_event_t event,
                            json &parsed) {
      if (event == json::parse_event_t::object_start) {
        object_keys[depth].clear();
      } else if (event == json::parse_event_t::key) {
        if (depth <= 0 || !parsed.is_string()) {
          throw contract_error("JSON object key has invalid parser depth or type");
        }
        const auto owner = object_keys.find(depth - 1);
        if (owner == object_keys.end()) {
          throw contract_error("JSON object key has no owning object");
        }
        const auto &key = parsed.get_ref<const std::string &>();
        if (!owner->second.insert(key).second) {
          throw contract_error("JSON object contains duplicate key '" + key + "'");
        }
      } else if (event == json::parse_event_t::object_end) {
        object_keys.erase(depth);
      }
      return true;
    };
    return json::parse(bytes, callback, true, false);
  }

  nlohmann::json to_json(const worker_spec_contract_t &value) {
    json result {
      {"schema", 3},
      {"job_id", value.job_id},
      {"operation", value.operation},
      {"input_path", value.input_path},
      {"job_directory", value.job_directory},
      {"result_directory", value.result_directory},
      {"progress_path", value.progress_path},
      {"result_path", value.result_path},
      {"staging_output", value.staging_output ? json(*value.staging_output) : json(nullptr)},
      {"sunshine", {
        {"executable", value.sunshine_executable},
        {"config", value.sunshine_config},
      }},
      {"ffmpeg", {
        {"path", value.ffmpeg_path},
        {"version", value.ffmpeg_version},
      }},
      {"ffprobe", {
        {"path", value.ffprobe_path},
        {"version", value.ffprobe_version},
      }},
      {"transient_raster", {
        {"hard_cap_bytes", value.transient_raster_hard_cap_bytes},
      }},
      {"codec", value.codec},
      {"pipeline", {
        {"implementation", "native-causal-single-pass"},
        {"lookahead", false},
        {"scene_cache", false},
        {"replay", false},
      }},
      {"python_dependency", false},
    };
    return result;
  }

  worker_spec_contract_t parse_worker_spec_contract(const nlohmann::json &value) {
    static constexpr std::array keys {
      "schema"sv, "job_id"sv, "operation"sv, "input_path"sv,
      "job_directory"sv, "result_directory"sv, "progress_path"sv,
      "result_path"sv, "staging_output"sv, "sunshine"sv, "ffmpeg"sv,
      "ffprobe"sv, "transient_raster"sv, "codec"sv, "pipeline"sv,
      "python_dependency"sv,
    };
    require_exact_keys(value, keys, "worker specification");
    if (value.at("schema") != 3 || value.at("python_dependency") != false) {
      throw contract_error("worker specification is not native schema 3");
    }
    require_exact_keys(
      value.at("sunshine"),
      std::array {"executable"sv, "config"sv},
      "worker specification sunshine"
    );
    require_exact_keys(
      value.at("ffmpeg"),
      std::array {"path"sv, "version"sv},
      "worker specification ffmpeg"
    );
    require_exact_keys(
      value.at("ffprobe"),
      std::array {"path"sv, "version"sv},
      "worker specification ffprobe"
    );
    require_exact_keys(
      value.at("transient_raster"),
      std::array {"hard_cap_bytes"sv},
      "worker specification transient raster"
    );
    require_exact_keys(
      value.at("pipeline"),
      std::array {"implementation"sv, "lookahead"sv, "scene_cache"sv,
                  "replay"sv},
      "worker specification causal pipeline"
    );

    worker_spec_contract_t result;
    result.job_id = required_string(value, "job_id", "worker specification");
    result.operation = required_string(value, "operation", "worker specification");
    result.input_path = required_string(value, "input_path", "worker specification");
    result.job_directory = required_string(value, "job_directory", "worker specification");
    result.result_directory = required_string(value, "result_directory", "worker specification");
    result.progress_path = required_string(value, "progress_path", "worker specification");
    result.result_path = required_string(value, "result_path", "worker specification");
    if (!value.at("staging_output").is_null()) {
      result.staging_output = required_string(value, "staging_output", "worker specification");
    }
    result.sunshine_executable = required_string(
      value.at("sunshine"), "executable", "worker specification sunshine"
    );
    result.sunshine_config = required_string(
      value.at("sunshine"), "config", "worker specification sunshine"
    );
    result.ffmpeg_path = required_string(
      value.at("ffmpeg"), "path", "worker specification ffmpeg"
    );
    result.ffmpeg_version = required_string(
      value.at("ffmpeg"), "version", "worker specification ffmpeg"
    );
    result.ffprobe_path = required_string(
      value.at("ffprobe"), "path", "worker specification ffprobe"
    );
    result.ffprobe_version = required_string(
      value.at("ffprobe"), "version", "worker specification ffprobe"
    );
    result.transient_raster_hard_cap_bytes = unsigned_integer<std::uint64_t>(
      value.at("transient_raster"), "hard_cap_bytes",
      "worker specification transient raster"
    );
    result.codec = required_string(value, "codec", "worker specification");

    if (result.operation != "evaluate" && result.operation != "convert") {
      throw contract_error("worker operation must be evaluate or convert");
    }
    if ((result.operation == "convert") != result.staging_output.has_value()) {
      throw contract_error("conversion requires staging_output and evaluation forbids it");
    }
    if (result.codec != "hevc_nvenc" && result.codec != "av1_nvenc") {
      throw contract_error("worker codec must be hevc_nvenc or av1_nvenc");
    }
    if (result.transient_raster_hard_cap_bytes < 16ull * 1024ull * 1024ull) {
      throw contract_error("transient-raster hard cap is below 16 MiB");
    }
    if (value.at("pipeline") != json {
          {"implementation", "native-causal-single-pass"},
          {"lookahead", false},
          {"scene_cache", false},
          {"replay", false}}) {
      throw contract_error("worker causal pipeline contract is unsupported");
    }
    return result;
  }

  nlohmann::json to_json(const replay_result_contract_t &value) {
    return {
      {"scene_id", value.scene_id},
      {"start_sequence", value.start_sequence},
      {"end_sequence_exclusive", value.end_sequence_exclusive},
      {"inference_mode", value.inference_mode},
      {"depth_inference_enabled", value.depth_inference_enabled},
      {"scheduled_depth_update_count", value.scheduled_depth_update_count},
      {"tensorrt_enqueue_count", value.tensorrt_enqueue_count},
      {"sbs", to_json(value.sbs)},
    };
  }

  replay_result_contract_t parse_replay_result_contract(
    const nlohmann::json &value
  ) {
    require_exact_keys(
      value,
      std::array {
        "scene_id"sv, "start_sequence"sv, "end_sequence_exclusive"sv,
        "inference_mode"sv, "depth_inference_enabled"sv,
        "scheduled_depth_update_count"sv, "tensorrt_enqueue_count"sv,
        "sbs"sv,
      },
      "worker replay result"
    );
    replay_result_contract_t result {
      .scene_id = unsigned_integer<std::uint64_t>(
        value, "scene_id", "worker replay result"
      ),
      .start_sequence = unsigned_integer<std::uint64_t>(
        value, "start_sequence", "worker replay result"
      ),
      .end_sequence_exclusive = unsigned_integer<std::uint64_t>(
        value, "end_sequence_exclusive", "worker replay result"
      ),
      .inference_mode = required_string(
        value, "inference_mode", "worker replay result"
      ),
      .depth_inference_enabled = required_boolean(
        value, "depth_inference_enabled", "worker replay result"
      ),
      .scheduled_depth_update_count = unsigned_integer<std::uint64_t>(
        value, "scheduled_depth_update_count", "worker replay result"
      ),
      .tensorrt_enqueue_count = unsigned_integer<std::uint64_t>(
        value, "tensorrt_enqueue_count", "worker replay result"
      ),
      .sbs = parse_whole_clip_sbs(value.at("sbs")),
    };
    if (result.scene_id == 0 || result.start_sequence == 0 ||
        result.end_sequence_exclusive <= result.start_sequence ||
        result.inference_mode != "scene-cache-replay" ||
        result.depth_inference_enabled ||
        result.scheduled_depth_update_count != 0 ||
        result.tensorrt_enqueue_count != 0 || !result.sbs.enabled) {
      throw contract_error("worker replay result is not zero-inference cache replay");
    }
    return result;
  }

  namespace {
    void validate_worker_result_relations(const worker_result_contract_t &result) {
      const auto &source = result.source;
      const auto &analysis = result.analysis_contract;
      const auto &runtime = analysis.resolved_runtime;
      const bool converting = result.operation == "convert";
      const bool sdr = source.color_mode == "sdr";
      // The manager reads a worker result only from a worker of its own build, which always
      // uses the raw memory transport and its completion evidence.
      if (
        runtime.frame_transport != raw_frame_transport::name ||
        !analysis.observation_timeline ||
        analysis.observation_timeline->count != source.frame_count ||
        analysis.artifact_mode != (converting ? "conversion" : "adaptive") ||
        analysis.inference_mode != "single-pass-tensorrt" ||
        !analysis.depth_inference_enabled ||
        analysis.scheduled_depth_update_count != source.frame_count ||
        analysis.tensorrt_enqueue_count != source.frame_count ||
        analysis.source_frame_count != source.frame_count ||
        analysis.source_width != source.width ||
        analysis.source_height != source.height ||
        analysis.source_first_sequence != 1 ||
        analysis.adaptive_state.transport != completion_channel_t::name ||
        analysis.adaptive_state.retained_history ||
        analysis.adaptive_state.frame_count != source.frame_count ||
        !runtime.parallax_v2_render || !runtime.parallax_v2_live ||
        !runtime.follow_mode ||
        runtime.follow_count_bound.value_or(0) != source.frame_count ||
        runtime.follow_producer_frame_count.value_or(0) != source.frame_count ||
        runtime.follow_first_sequence.value_or(0) != 1 ||
        runtime.follow_poll_interval_ms.value_or(
          std::numeric_limits<std::uint32_t>::max()
        ) != 0 ||
        runtime.follow_native_input_deletion ||
        runtime.follow_format.value_or("") != (sdr ? "bgra8" : "gbrpf32le") ||
        runtime.follow_frame_pattern.has_value() ||
        runtime.input_frame_format != (sdr ? "sRGB-BGRA8-raw" : "linear-scRGB-f32-planar") ||
        runtime.scene_cache_write || runtime.scene_cache_replay ||
        runtime.scene_cache_contract_schema || runtime.scene_plan_schema ||
        runtime.scene_plan_version || runtime.scene_start_sequence ||
        runtime.scene_end_sequence_exclusive ||
        runtime.scene_cache_status_at_replay_start ||
        runtime.scene_cache_processed_count_at_replay_start ||
        !result.replay_contracts.empty() || result.cache.peak_bytes != 0 ||
        result.cache.remaining_bytes != 0 ||
        result.cache.analysis_source_raster_bytes == 0 ||
        result.cache.peak_live_raster_bytes <
          result.cache.analysis_source_raster_bytes ||
        result.cache.peak_cache_plus_raster_bytes !=
          result.cache.peak_live_raster_bytes
      ) {
        throw contract_error(
          "worker causal-pass attestation disagrees with the source or one-pass contract"
        );
      }
      if (converting) {
        const auto &sbs = analysis.sbs;
        if (!sbs.enabled || sbs.frame_count != source.frame_count ||
            sbs.width != runtime.output_sbs_width ||
            sbs.height != runtime.output_sbs_height ||
            sbs.frame_format != (sdr ? "bgra8" : "rgba16f") ||
            sbs.transfer != runtime.output_transfer ||
            sbs.primaries != runtime.output_primaries ||
            sbs.row_order != runtime.output_row_order ||
            runtime.output_frame_format != (sdr ? "sRGB-BGRA8-raw" : "linear-scRGB-f16-raw") ||
            runtime.output_ffmpeg_pixel_format != (sdr ? "bgra" : "gbrpf32le") ||
            sbs.ffmpeg_pixel_format != runtime.output_ffmpeg_pixel_format ||
            !sbs.atomic_publication || !runtime.follow_atomic_sbs_publication) {
          throw contract_error(
            "worker direct SBS attestation disagrees with the causal pass"
          );
        }
      } else if (analysis.sbs.enabled || analysis.sbs.frame_count != 0 ||
                 analysis.sbs.width != 0 || analysis.sbs.height != 0 ||
                 analysis.sbs.atomic_publication ||
                 runtime.follow_atomic_sbs_publication) {
        throw contract_error("analysis-only worker unexpectedly attests SBS output");
      }

      if (source.frame_count == std::numeric_limits<std::uint64_t>::max()) {
        throw contract_error("worker source frame range overflows");
      }

      const auto timeline_mode = required_string(
        result.timeline_contract, "mode", "worker result timeline contract"
      );
      if (!converting) {
        if (result.timeline_contract != json {
              {"mode", "evaluation-only"},
              {"max_output_ticks", nullptr}}) {
          throw contract_error(
            "worker evaluation timeline contract is not evaluation-only"
          );
        }
      } else {
        const auto exact_mp4 =
          timeline_mode == "exact-rational" &&
          result.timeline_contract.value("container", "") == "mp4" &&
          result.timeline_contract.value("max_output_ticks", -1) == 0;
        const auto bounded_matroska =
          timeline_mode == "bounded-output-timebase" &&
          result.timeline_contract.value("container", "") == "matroska" &&
          result.timeline_contract.value("max_output_ticks", -1) == 1 &&
          result.timeline_contract.value("absolute_timestamps", false) &&
          !result.timeline_contract.value("cumulative_drift_allowed", true);
        if ((!exact_mp4 && !bounded_matroska) ||
            !result.timeline_contract.contains("observed_video") ||
            !result.timeline_contract.at("observed_video").is_object() ||
            !result.timeline_contract.contains("observed_auxiliary") ||
            !result.timeline_contract.at("observed_auxiliary").is_object()) {
          throw contract_error(
            "worker conversion timeline contract is incomplete"
          );
        }
      }

    }
  }  // namespace

  nlohmann::json to_json(const worker_result_contract_t &value) {
    json replays = json::array();
    for (const auto &replay : value.replay_contracts) {
      replays.push_back(to_json(replay));
    }
    json result {
      {"schema", 4},
      {"job_id", value.job_id},
      {"status", "complete"},
      {"operation", value.operation},
      {"codec", value.codec},
      {"worker_spec_sha256", value.worker_spec_sha256},
      {"python_dependency", false},
      {"source", {
        {"width", value.source.width},
        {"height", value.source.height},
        {"frame_count", value.source.frame_count},
        {"duration_seconds", value.source.duration_seconds},
        {"variable_frame_rate", value.source.variable_frame_rate},
        {"color_mode", value.source.color_mode},
        {"contract", value.source.contract_path},
      }},
      {"analysis_contract", to_json(value.analysis_contract)},
      {"replay_contracts", std::move(replays)},
      {"cache", {
        {"hard_cap_bytes", value.cache.hard_cap_bytes},
        {"peak_bytes", value.cache.peak_bytes},
        {"analysis_source_raster_bytes", value.cache.analysis_source_raster_bytes},
        {"peak_live_raster_bytes", value.cache.peak_live_raster_bytes},
        {"peak_cache_plus_raster_bytes", value.cache.peak_cache_plus_raster_bytes},
        {"remaining_bytes", value.cache.remaining_bytes},
      }},
      {"timeline_contract", value.timeline_contract},
      {"staging_identity", value.staging_identity ? *value.staging_identity : json(nullptr)},
      {"output", value.output_path ? json(*value.output_path) : json(nullptr)},
    };
    (void) parse_worker_result_contract(result);
    return result;
  }

  worker_result_contract_t parse_worker_result_contract(const nlohmann::json &value) {
    static constexpr std::array keys {
      "schema"sv, "job_id"sv, "status"sv, "operation"sv, "codec"sv,
      "worker_spec_sha256"sv, "python_dependency"sv, "source"sv,
      "analysis_contract"sv, "replay_contracts"sv, "cache"sv,
      "timeline_contract"sv, "staging_identity"sv, "output"sv,
    };
    require_exact_keys(value, keys, "worker result");
    if (value.at("schema") != 4 || value.at("status") != "complete" ||
        value.at("python_dependency") != false) {
      throw contract_error("worker result is not a complete native result");
    }
    require_exact_keys(
      value.at("source"),
      std::array {
        "width"sv, "height"sv, "frame_count"sv, "duration_seconds"sv,
        "variable_frame_rate"sv, "color_mode"sv, "contract"sv,
      },
      "worker result source"
    );
    require_exact_keys(
      value.at("cache"),
      std::array {
        "hard_cap_bytes"sv, "peak_bytes"sv, "analysis_source_raster_bytes"sv,
        "peak_live_raster_bytes"sv, "peak_cache_plus_raster_bytes"sv,
        "remaining_bytes"sv,
      },
      "worker result cache"
    );

    worker_result_contract_t result;
    result.job_id = required_string(value, "job_id", "worker result");
    result.operation = required_string(value, "operation", "worker result");
    result.codec = required_string(value, "codec", "worker result");
    result.worker_spec_sha256 = required_string(
      value, "worker_spec_sha256", "worker result"
    );
    if (!valid_sha256_hex(result.worker_spec_sha256)) {
      throw contract_error("worker result specification digest is not lowercase SHA-256");
    }
    if (result.operation != "evaluate" && result.operation != "convert") {
      throw contract_error("worker result operation is invalid");
    }
    if (result.codec != "hevc_nvenc" && result.codec != "av1_nvenc") {
      throw contract_error("worker result codec is invalid");
    }

    const auto &source = value.at("source");
    result.source.width = unsigned_integer<std::uint32_t>(source, "width", "worker result source");
    result.source.height = unsigned_integer<std::uint32_t>(source, "height", "worker result source");
    result.source.frame_count = unsigned_integer<std::uint64_t>(
      source, "frame_count", "worker result source"
    );
    if (!source.at("duration_seconds").is_number()) {
      throw contract_error("worker result source duration must be numeric");
    }
    result.source.duration_seconds = source.at("duration_seconds").get<double>();
    if (!source.at("variable_frame_rate").is_boolean()) {
      throw contract_error("worker result source VFR flag must be boolean");
    }
    result.source.variable_frame_rate = source.at("variable_frame_rate").get<bool>();
    result.source.color_mode = required_string(source, "color_mode", "worker result source");
    result.source.contract_path = required_string(source, "contract", "worker result source");
    if (result.source.width == 0 || result.source.height == 0 ||
        result.source.frame_count == 0 || !std::isfinite(result.source.duration_seconds) ||
        result.source.duration_seconds <= 0.0 ||
        (result.source.color_mode != "sdr" && result.source.color_mode != "hdr-pq" &&
         result.source.color_mode != "hdr-hlg")) {
      throw contract_error("worker result source summary is invalid");
    }

    if (!value.at("analysis_contract").is_object() ||
        !value.at("replay_contracts").is_array() ||
        !value.at("timeline_contract").is_object()) {
      throw contract_error("worker result nested attestations have invalid types");
    }
    result.analysis_contract = parse_whole_clip_contract(
      value.at("analysis_contract")
    );
    result.timeline_contract = value.at("timeline_contract");
    for (const auto &replay : value.at("replay_contracts")) {
      result.replay_contracts.push_back(parse_replay_result_contract(replay));
    }

    const auto &cache = value.at("cache");
    result.cache.hard_cap_bytes = unsigned_integer<std::uint64_t>(
      cache, "hard_cap_bytes", "worker result cache"
    );
    result.cache.peak_bytes = unsigned_integer<std::uint64_t>(
      cache, "peak_bytes", "worker result cache"
    );
    result.cache.analysis_source_raster_bytes = unsigned_integer<std::uint64_t>(
      cache, "analysis_source_raster_bytes", "worker result cache"
    );
    result.cache.peak_live_raster_bytes = unsigned_integer<std::uint64_t>(
      cache, "peak_live_raster_bytes", "worker result cache"
    );
    result.cache.peak_cache_plus_raster_bytes = unsigned_integer<std::uint64_t>(
      cache, "peak_cache_plus_raster_bytes", "worker result cache"
    );
    result.cache.remaining_bytes = unsigned_integer<std::uint64_t>(
      cache, "remaining_bytes", "worker result cache"
    );
    if (result.cache.hard_cap_bytes == 0 ||
        result.cache.peak_bytes > result.cache.hard_cap_bytes ||
        result.cache.peak_cache_plus_raster_bytes > result.cache.hard_cap_bytes) {
      throw contract_error("worker result cache summary is invalid");
    }

    if (!value.at("staging_identity").is_null()) {
      if (!value.at("staging_identity").is_object()) {
        throw contract_error("worker result staging identity must be an object or null");
      }
      result.staging_identity = value.at("staging_identity");
    }
    if (!value.at("output").is_null()) {
      result.output_path = required_string(value, "output", "worker result");
    }
    if (result.operation == "convert") {
      if (!result.staging_identity || !result.output_path) {
        throw contract_error("conversion result lacks exact output attestations");
      }
    } else if (result.staging_identity || result.output_path ||
               !result.replay_contracts.empty()) {
      throw contract_error("evaluation result unexpectedly reports conversion output");
    }
    validate_worker_result_relations(result);
    return result;
  }

  nlohmann::json worker_failure_json(
    const std::string_view job_id,
    const std::string_view error
  ) {
    return {
      {"schema", 1},
      {"job_id", job_id},
      {"status", "failed"},
      {"error", error.substr(0, 4096)},
      {"python_dependency", false},
    };
  }

  nlohmann::json to_json(const whole_clip_sbs_t &value) {
    return {
      {"enabled", value.enabled},
      {"file_pattern", optional_json(value.file_pattern)},
      {"frame_count", value.frame_count},
      {"width", value.width},
      {"height", value.height},
      {"frame_format", value.frame_format},
      {"transfer", value.transfer},
      {"primaries", value.primaries},
      {"reference_white_nits", optional_json(value.reference_white_nits)},
      {"row_order", value.row_order},
      {"ffmpeg_pixel_format", optional_json(value.ffmpeg_pixel_format)},
      {"atomic_publication", value.atomic_publication},
    };
  }

  whole_clip_sbs_t parse_whole_clip_sbs(const nlohmann::json &value) {
    require_exact_keys(
      value,
      std::array {
        "enabled"sv, "file_pattern"sv, "frame_count"sv, "width"sv,
        "height"sv, "frame_format"sv, "transfer"sv, "primaries"sv,
        "reference_white_nits"sv, "row_order"sv,
        "ffmpeg_pixel_format"sv, "atomic_publication"sv,
      },
      "whole-clip SBS"
    );
    whole_clip_sbs_t result {
      .enabled = required_boolean(value, "enabled", "whole-clip SBS"),
      .frame_count = unsigned_integer<std::uint64_t>(
        value, "frame_count", "whole-clip SBS"
      ),
      .width = unsigned_integer<std::uint32_t>(value, "width", "whole-clip SBS"),
      .height = unsigned_integer<std::uint32_t>(value, "height", "whole-clip SBS"),
      .frame_format = required_string(value, "frame_format", "whole-clip SBS"),
      .transfer = required_string(value, "transfer", "whole-clip SBS"),
      .primaries = required_string(value, "primaries", "whole-clip SBS"),
      .row_order = required_string(value, "row_order", "whole-clip SBS"),
      .atomic_publication = required_boolean(
        value, "atomic_publication", "whole-clip SBS"
      ),
    };
    result.file_pattern = optional_value<std::string>(value, "file_pattern", [&]() {
      return required_string(value, "file_pattern", "whole-clip SBS");
    });
    result.reference_white_nits = optional_value<double>(
      value, "reference_white_nits", [&]() {
        return finite_number(value, "reference_white_nits", "whole-clip SBS");
      }
    );
    result.ffmpeg_pixel_format = optional_value<std::string>(
      value, "ffmpeg_pixel_format", [&]() {
        return required_string(value, "ffmpeg_pixel_format", "whole-clip SBS");
      }
    );
    if ((result.width == 0) != (result.height == 0)) {
      throw contract_error("whole-clip SBS raster dimensions must be both zero or both nonzero");
    }
    if (result.width != 0) {
      (void) checked_raster_bytes(
        result.width, result.height, 1, "whole-clip SBS"
      );
    }
    if (result.enabled ?
          ((!result.file_pattern && result.frame_format != "bgra8" && result.frame_format != "rgba16f") || result.frame_count == 0 ||
           result.width == 0 || result.height == 0) :
          (result.file_pattern || result.frame_count != 0 ||
           result.width != 0 || result.height != 0 ||
           result.atomic_publication)) {
      throw contract_error("whole-clip SBS enablement disagrees with its raster evidence");
    }
    return result;
  }

  nlohmann::json to_json(const whole_clip_contract_t &value) {
    const auto &runtime = value.resolved_runtime;
    json resolved_runtime {
      {"model", runtime.model},
      {"model_url", runtime.model_url},
      {"pop_strength", runtime.pop_strength},
      {"depth_width", runtime.depth_width},
      {"depth_height", runtime.depth_height},
      {"depth_reuse_interval", runtime.depth_reuse_interval},
      {"cuda_graph", runtime.cuda_graph},
      {"parallax_v2_shadow", runtime.parallax_v2_shadow},
      {"parallax_v2_render", runtime.parallax_v2_render},
      {"parallax_v2_live", runtime.parallax_v2_live},
      {"cuda_graph_captured", runtime.cuda_graph_captured},
      {"simulate_hdr", runtime.simulate_hdr},
      {"hdr_scale", runtime.hdr_scale},
      {"input_color_space", runtime.input_color_space},
      {"input_frame_format", runtime.input_frame_format},
      {"input_texture_format", runtime.input_texture_format},
      {"input_transfer", runtime.input_transfer},
      {"input_primaries", runtime.input_primaries},
      {"input_reference_white_nits", optional_json(runtime.input_reference_white_nits)},
      {"packed_texture_format", runtime.packed_texture_format},
      {"packed_color_space", runtime.packed_color_space},
      {"packed_transfer", runtime.packed_transfer},
      {"packed_primaries", runtime.packed_primaries},
      {"packed_reference_white_nits", optional_json(runtime.packed_reference_white_nits)},
      {"output_frame_format", runtime.output_frame_format},
      {"output_transfer", runtime.output_transfer},
      {"output_primaries", runtime.output_primaries},
      {"output_reference_white_nits", optional_json(runtime.output_reference_white_nits)},
      {"output_row_order", runtime.output_row_order},
      {"output_ffmpeg_pixel_format", optional_json(runtime.output_ffmpeg_pixel_format)},
      {"output_scale", runtime.output_scale},
      {"requested_eye_width", runtime.requested_eye_width},
      {"requested_eye_height", runtime.requested_eye_height},
      {"configured_max_encode_width", runtime.configured_max_encode_width},
      {"resolved_max_output_width", runtime.resolved_max_output_width},
      {"input_limit", runtime.input_limit},
      {"output_every", runtime.output_every},
      {"follow_mode", runtime.follow_mode},
      {"follow_format", optional_json(runtime.follow_format)},
      {"follow_count_bound", optional_json(runtime.follow_count_bound)},
      {"follow_producer_frame_count", optional_json(runtime.follow_producer_frame_count)},
      {"follow_frame_pattern", optional_json(runtime.follow_frame_pattern)},
      {"follow_first_sequence", optional_json(runtime.follow_first_sequence)},
      {"follow_poll_interval_ms", optional_json(runtime.follow_poll_interval_ms)},
      {"follow_done_sentinel", optional_json(runtime.follow_done_sentinel)},
      {"follow_failed_sentinel", optional_json(runtime.follow_failed_sentinel)},
      {"follow_progress_file", optional_json(runtime.follow_progress_file)},
      {"follow_native_input_deletion", runtime.follow_native_input_deletion},
      {"follow_atomic_sbs_publication", runtime.follow_atomic_sbs_publication},
      {"output_eye_width", runtime.output_eye_width},
      {"output_eye_height", runtime.output_eye_height},
      {"output_sbs_width", runtime.output_sbs_width},
      {"output_sbs_height", runtime.output_sbs_height},
      {"content_scale_x", runtime.content_scale_x},
      {"content_scale_y", runtime.content_scale_y},
      {"scene_cache_write", runtime.scene_cache_write},
      {"scene_cache_replay", runtime.scene_cache_replay},
      {"scene_cache_contract_schema", optional_json(runtime.scene_cache_contract_schema)},
      {"scene_plan_schema", optional_json(runtime.scene_plan_schema)},
      {"scene_plan_version", optional_json(runtime.scene_plan_version)},
      {"scene_start_sequence", optional_json(runtime.scene_start_sequence)},
      {"scene_end_sequence_exclusive", optional_json(runtime.scene_end_sequence_exclusive)},
      {"scene_cache_status_at_replay_start",
       optional_json(runtime.scene_cache_status_at_replay_start)},
      {"scene_cache_processed_count_at_replay_start",
       optional_json(runtime.scene_cache_processed_count_at_replay_start)},
    };
    if (runtime.frame_transport) {
      resolved_runtime["frame_transport"] = *runtime.frame_transport;
    }
    const auto &adaptive = value.adaptive_state;
    if (adaptive.transport == completion_channel_t::name &&
        (adaptive.header_file || adaptive.frame_file || adaptive.file)) {
      throw contract_error("whole-clip in-memory adaptive state cannot advertise files");
    }
    json adaptive_state {
      {"transport", adaptive.transport},
      {"retained_history", adaptive.retained_history},
      {"frame_count", adaptive.frame_count},
    };
    if (adaptive.transport == "atomic-latest-v1") {
      adaptive_state["header_file"] = optional_json(adaptive.header_file);
      adaptive_state["frame_file"] = optional_json(adaptive.frame_file);
    } else if (adaptive.transport == "jsonl-v1") {
      adaptive_state["file"] = optional_json(adaptive.file);
    }
    if (adaptive.transport != "none") {
      adaptive_state["schema"] = optional_json(adaptive.schema);
      adaptive_state["capture"] = optional_json(adaptive.capture);
    }
    json observation = nullptr;
    if (value.observation_timeline) {
      observation = {
        {"schema", value.observation_timeline->schema},
        {"timestamp_unit", value.observation_timeline->timestamp_unit},
        {"count", value.observation_timeline->count},
        {"sha256", value.observation_timeline->sha256},
      };
    }
    json result {
      {"schema", 1},
      {"source_scope", {
        {"frame_source", whole_clip_frame_source},
        {"analysis_region", whole_clip_analysis_region},
        {"active_window_dependency", false},
        {"window_region_roi", false},
      }},
      {"observation_timeline", std::move(observation)},
      {"artifact_mode", value.artifact_mode},
      {"inference_mode", value.inference_mode},
      {"depth_inference_enabled", value.depth_inference_enabled},
      {"scheduled_depth_update_count", value.scheduled_depth_update_count},
      {"tensorrt_enqueue_count", value.tensorrt_enqueue_count},
      {"depth_provenance", value.depth_provenance},
      {"pipeline_state_provenance", value.pipeline_state_provenance},
      {"model", value.model},
      {"source_frame_count", value.source_frame_count},
      {"source_width", value.source_width},
      {"source_height", value.source_height},
      {"source_first_sequence", value.source_first_sequence},
      {"depth_reuse_interval", value.depth_reuse_interval},
      {"resolved_runtime", std::move(resolved_runtime)},
      {"adaptive_state", std::move(adaptive_state)},
      {"sbs", to_json(value.sbs)},
    };
    (void) parse_whole_clip_contract(result);
    return result;
  }

  whole_clip_contract_t parse_whole_clip_contract(const nlohmann::json &value) {
    require_exact_keys(
      value,
      std::array {
        "schema"sv, "source_scope"sv, "observation_timeline"sv,
        "artifact_mode"sv, "inference_mode"sv, "depth_inference_enabled"sv,
        "scheduled_depth_update_count"sv, "tensorrt_enqueue_count"sv,
        "depth_provenance"sv, "pipeline_state_provenance"sv, "model"sv,
        "source_frame_count"sv, "source_width"sv, "source_height"sv,
        "source_first_sequence"sv, "depth_reuse_interval"sv,
        "resolved_runtime"sv, "adaptive_state"sv, "sbs"sv,
      },
      "whole-clip contract"
    );
    require_exact_keys(
      value.at("source_scope"),
      std::array {"frame_source"sv, "analysis_region"sv,
                  "active_window_dependency"sv, "window_region_roi"sv},
      "whole-clip source scope"
    );
    if (value.at("schema") != 1 ||
        value.at("source_scope") != json {
          {"frame_source", whole_clip_frame_source},
          {"analysis_region", whole_clip_analysis_region},
          {"active_window_dependency", false},
          {"window_region_roi", false}}) {
      throw contract_error("whole-clip contract has unknown source semantics");
    }
    whole_clip_contract_t result;
    if (!value.at("observation_timeline").is_null()) {
      const auto &timeline = value.at("observation_timeline");
      require_exact_keys(
        timeline,
        std::array {"schema"sv, "timestamp_unit"sv, "count"sv, "sha256"sv},
        "whole-clip observation timeline"
      );
      observation_timeline_contract_t parsed {
        .schema = unsigned_integer<std::uint32_t>(
          timeline, "schema", "whole-clip observation timeline"
        ),
        .timestamp_unit = required_string(
          timeline, "timestamp_unit", "whole-clip observation timeline"
        ),
        .count = unsigned_integer<std::uint64_t>(
          timeline, "count", "whole-clip observation timeline"
        ),
        .sha256 = required_string(
          timeline, "sha256", "whole-clip observation timeline"
        ),
      };
      if (!valid_sha256_hex(parsed.sha256) || parsed.count == 0) {
        throw contract_error("whole-clip observation timeline identity is invalid");
      }
      result.observation_timeline = std::move(parsed);
    }
    result.artifact_mode = required_string(value, "artifact_mode", "whole-clip contract");
    result.inference_mode = required_string(value, "inference_mode", "whole-clip contract");
    result.depth_inference_enabled = required_boolean(value, "depth_inference_enabled", "whole-clip contract");
    result.scheduled_depth_update_count = unsigned_integer<std::uint64_t>(value, "scheduled_depth_update_count", "whole-clip contract");
    result.tensorrt_enqueue_count = unsigned_integer<std::uint64_t>(value, "tensorrt_enqueue_count", "whole-clip contract");
    result.depth_provenance = required_string(value, "depth_provenance", "whole-clip contract");
    result.pipeline_state_provenance = required_string(value, "pipeline_state_provenance", "whole-clip contract");
    result.model = required_string(value, "model", "whole-clip contract");
    result.source_frame_count = unsigned_integer<std::uint64_t>(value, "source_frame_count", "whole-clip contract");
    result.source_width = unsigned_integer<std::uint32_t>(value, "source_width", "whole-clip contract");
    result.source_height = unsigned_integer<std::uint32_t>(value, "source_height", "whole-clip contract");
    result.source_first_sequence = unsigned_integer<std::uint64_t>(value, "source_first_sequence", "whole-clip contract");
    result.depth_reuse_interval = signed_integer<int>(value, "depth_reuse_interval", "whole-clip contract");

    const auto &runtime = value.at("resolved_runtime");
    require_keys_with_optional(
      runtime,
      std::array {
        "model"sv, "model_url"sv, "pop_strength"sv, "depth_width"sv,
        "depth_height"sv, "depth_reuse_interval"sv, "cuda_graph"sv,
        "parallax_v2_shadow"sv, "parallax_v2_render"sv,
        "parallax_v2_live"sv, "cuda_graph_captured"sv, "simulate_hdr"sv,
        "hdr_scale"sv, "input_color_space"sv, "input_frame_format"sv,
        "input_texture_format"sv, "input_transfer"sv, "input_primaries"sv,
        "input_reference_white_nits"sv, "packed_texture_format"sv,
        "packed_color_space"sv, "packed_transfer"sv, "packed_primaries"sv,
        "packed_reference_white_nits"sv, "output_frame_format"sv,
        "output_transfer"sv, "output_primaries"sv,
        "output_reference_white_nits"sv, "output_row_order"sv,
        "output_ffmpeg_pixel_format"sv, "output_scale"sv,
        "requested_eye_width"sv, "requested_eye_height"sv,
        "configured_max_encode_width"sv, "resolved_max_output_width"sv,
        "input_limit"sv, "output_every"sv, "follow_mode"sv,
        "follow_format"sv, "follow_count_bound"sv,
        "follow_producer_frame_count"sv, "follow_frame_pattern"sv,
        "follow_first_sequence"sv, "follow_poll_interval_ms"sv,
        "follow_done_sentinel"sv, "follow_failed_sentinel"sv,
        "follow_progress_file"sv, "follow_native_input_deletion"sv,
        "follow_atomic_sbs_publication"sv, "output_eye_width"sv,
        "output_eye_height"sv, "output_sbs_width"sv, "output_sbs_height"sv,
        "content_scale_x"sv, "content_scale_y"sv, "scene_cache_write"sv,
        "scene_cache_replay"sv, "scene_cache_contract_schema"sv,
        "scene_plan_schema"sv, "scene_plan_version"sv,
        "scene_start_sequence"sv, "scene_end_sequence_exclusive"sv,
        "scene_cache_status_at_replay_start"sv,
        "scene_cache_processed_count_at_replay_start"sv,
      },
      "frame_transport",
      "whole-clip resolved runtime"
    );
    auto &typed_runtime = result.resolved_runtime;
    if (runtime.contains("frame_transport")) {
      typed_runtime.frame_transport = required_string(runtime, "frame_transport", "whole-clip resolved runtime");
      if (*typed_runtime.frame_transport != raw_frame_transport::name) {
        throw contract_error("whole-clip frame transport is unsupported");
      }
    }
    typed_runtime.model = required_string(runtime, "model", "whole-clip resolved runtime");
    typed_runtime.model_url = required_string(runtime, "model_url", "whole-clip resolved runtime");
    typed_runtime.pop_strength = finite_number(runtime, "pop_strength", "whole-clip resolved runtime");
    typed_runtime.depth_width = signed_integer<int>(runtime, "depth_width", "whole-clip resolved runtime");
    typed_runtime.depth_height = signed_integer<int>(runtime, "depth_height", "whole-clip resolved runtime");
    typed_runtime.depth_reuse_interval = signed_integer<int>(runtime, "depth_reuse_interval", "whole-clip resolved runtime");
    typed_runtime.cuda_graph = required_boolean(runtime, "cuda_graph", "whole-clip resolved runtime");
    typed_runtime.parallax_v2_shadow = required_boolean(runtime, "parallax_v2_shadow", "whole-clip resolved runtime");
    typed_runtime.parallax_v2_render = required_boolean(runtime, "parallax_v2_render", "whole-clip resolved runtime");
    typed_runtime.parallax_v2_live = required_boolean(runtime, "parallax_v2_live", "whole-clip resolved runtime");
    typed_runtime.cuda_graph_captured = required_boolean(runtime, "cuda_graph_captured", "whole-clip resolved runtime");
    typed_runtime.simulate_hdr = required_boolean(runtime, "simulate_hdr", "whole-clip resolved runtime");
    typed_runtime.hdr_scale = finite_number(runtime, "hdr_scale", "whole-clip resolved runtime");
    typed_runtime.input_color_space = required_string(runtime, "input_color_space", "whole-clip resolved runtime");
    typed_runtime.input_frame_format = required_string(runtime, "input_frame_format", "whole-clip resolved runtime");
    typed_runtime.input_texture_format = required_string(runtime, "input_texture_format", "whole-clip resolved runtime");
    typed_runtime.input_transfer = required_string(runtime, "input_transfer", "whole-clip resolved runtime");
    typed_runtime.input_primaries = required_string(runtime, "input_primaries", "whole-clip resolved runtime");
    typed_runtime.packed_texture_format = required_string(runtime, "packed_texture_format", "whole-clip resolved runtime");
    typed_runtime.packed_color_space = required_string(runtime, "packed_color_space", "whole-clip resolved runtime");
    typed_runtime.packed_transfer = required_string(runtime, "packed_transfer", "whole-clip resolved runtime");
    typed_runtime.packed_primaries = required_string(runtime, "packed_primaries", "whole-clip resolved runtime");
    typed_runtime.output_frame_format = required_string(runtime, "output_frame_format", "whole-clip resolved runtime");
    typed_runtime.output_transfer = required_string(runtime, "output_transfer", "whole-clip resolved runtime");
    typed_runtime.output_primaries = required_string(runtime, "output_primaries", "whole-clip resolved runtime");
    typed_runtime.output_row_order = required_string(runtime, "output_row_order", "whole-clip resolved runtime");
    typed_runtime.output_scale = finite_number(runtime, "output_scale", "whole-clip resolved runtime");
    typed_runtime.requested_eye_width = signed_integer<int>(runtime, "requested_eye_width", "whole-clip resolved runtime");
    typed_runtime.requested_eye_height = signed_integer<int>(runtime, "requested_eye_height", "whole-clip resolved runtime");
    typed_runtime.configured_max_encode_width = signed_integer<int>(runtime, "configured_max_encode_width", "whole-clip resolved runtime");
    typed_runtime.resolved_max_output_width = signed_integer<int>(runtime, "resolved_max_output_width", "whole-clip resolved runtime");
    typed_runtime.input_limit = signed_integer<int>(
      runtime, "input_limit", "whole-clip resolved runtime"
    );
    typed_runtime.output_every = unsigned_integer<std::uint64_t>(runtime, "output_every", "whole-clip resolved runtime");
    typed_runtime.follow_mode = required_boolean(runtime, "follow_mode", "whole-clip resolved runtime");
    typed_runtime.follow_native_input_deletion = required_boolean(runtime, "follow_native_input_deletion", "whole-clip resolved runtime");
    typed_runtime.follow_atomic_sbs_publication = required_boolean(runtime, "follow_atomic_sbs_publication", "whole-clip resolved runtime");
    typed_runtime.output_eye_width = unsigned_integer<std::uint32_t>(runtime, "output_eye_width", "whole-clip resolved runtime");
    typed_runtime.output_eye_height = unsigned_integer<std::uint32_t>(runtime, "output_eye_height", "whole-clip resolved runtime");
    typed_runtime.output_sbs_width = unsigned_integer<std::uint32_t>(runtime, "output_sbs_width", "whole-clip resolved runtime");
    typed_runtime.output_sbs_height = unsigned_integer<std::uint32_t>(runtime, "output_sbs_height", "whole-clip resolved runtime");
    typed_runtime.content_scale_x = finite_number(runtime, "content_scale_x", "whole-clip resolved runtime");
    typed_runtime.content_scale_y = finite_number(runtime, "content_scale_y", "whole-clip resolved runtime");
    typed_runtime.scene_cache_write = required_boolean(runtime, "scene_cache_write", "whole-clip resolved runtime");
    typed_runtime.scene_cache_replay = required_boolean(runtime, "scene_cache_replay", "whole-clip resolved runtime");
    const auto optional_string = [&](const std::string_view key) {
      return optional_value<std::string>(runtime, key, [&]() {
        return required_string(runtime, key, "whole-clip resolved runtime");
      });
    };
    const auto optional_u64 = [&](const std::string_view key) {
      return optional_value<std::uint64_t>(runtime, key, [&]() {
        return unsigned_integer<std::uint64_t>(runtime, key, "whole-clip resolved runtime");
      });
    };
    const auto optional_u32 = [&](const std::string_view key) {
      return optional_value<std::uint32_t>(runtime, key, [&]() {
        return unsigned_integer<std::uint32_t>(runtime, key, "whole-clip resolved runtime");
      });
    };
    const auto optional_number = [&](const std::string_view key) {
      return optional_value<double>(runtime, key, [&]() {
        return finite_number(runtime, key, "whole-clip resolved runtime");
      });
    };
    typed_runtime.input_reference_white_nits = optional_number("input_reference_white_nits");
    typed_runtime.packed_reference_white_nits = optional_number("packed_reference_white_nits");
    typed_runtime.output_reference_white_nits = optional_number("output_reference_white_nits");
    typed_runtime.output_ffmpeg_pixel_format = optional_string("output_ffmpeg_pixel_format");
    typed_runtime.follow_format = optional_string("follow_format");
    typed_runtime.follow_count_bound = optional_u64("follow_count_bound");
    typed_runtime.follow_producer_frame_count = optional_u64("follow_producer_frame_count");
    typed_runtime.follow_frame_pattern = optional_string("follow_frame_pattern");
    typed_runtime.follow_first_sequence = optional_u64("follow_first_sequence");
    typed_runtime.follow_poll_interval_ms = optional_u32("follow_poll_interval_ms");
    typed_runtime.follow_done_sentinel = optional_string("follow_done_sentinel");
    typed_runtime.follow_failed_sentinel = optional_string("follow_failed_sentinel");
    typed_runtime.follow_progress_file = optional_string("follow_progress_file");
    typed_runtime.scene_cache_contract_schema = optional_u32("scene_cache_contract_schema");
    typed_runtime.scene_plan_schema = optional_u32("scene_plan_schema");
    typed_runtime.scene_plan_version = optional_string("scene_plan_version");
    typed_runtime.scene_start_sequence = optional_u64("scene_start_sequence");
    typed_runtime.scene_end_sequence_exclusive = optional_u64("scene_end_sequence_exclusive");
    typed_runtime.scene_cache_status_at_replay_start = optional_string("scene_cache_status_at_replay_start");
    typed_runtime.scene_cache_processed_count_at_replay_start = optional_u64("scene_cache_processed_count_at_replay_start");

    const auto &adaptive = value.at("adaptive_state");
    auto &typed_adaptive = result.adaptive_state;
    if (!adaptive.is_object() || !adaptive.contains("transport") ||
        !adaptive.at("transport").is_string()) {
      throw contract_error("whole-clip adaptive state has no transport");
    }
    typed_adaptive.transport = adaptive.at("transport").get<std::string>();
    if (typed_adaptive.transport == "none") {
      require_exact_keys(adaptive, std::array {"transport"sv, "retained_history"sv, "frame_count"sv}, "whole-clip adaptive state");
    } else if (typed_adaptive.transport == completion_channel_t::name) {
      require_exact_keys(adaptive, std::array {"transport"sv, "retained_history"sv, "schema"sv, "capture"sv, "frame_count"sv}, "whole-clip adaptive state");
    } else if (typed_adaptive.transport == "atomic-latest-v1") {
      require_exact_keys(adaptive, std::array {"transport"sv, "header_file"sv, "frame_file"sv, "retained_history"sv, "schema"sv, "capture"sv, "frame_count"sv}, "whole-clip adaptive state");
      typed_adaptive.header_file = required_string(adaptive, "header_file", "whole-clip adaptive state");
      typed_adaptive.frame_file = required_string(adaptive, "frame_file", "whole-clip adaptive state");
    } else if (typed_adaptive.transport == "jsonl-v1") {
      require_exact_keys(adaptive, std::array {"transport"sv, "file"sv, "retained_history"sv, "schema"sv, "capture"sv, "frame_count"sv}, "whole-clip adaptive state");
      typed_adaptive.file = required_string(adaptive, "file", "whole-clip adaptive state");
    } else {
      throw contract_error("whole-clip adaptive state transport is unsupported");
    }
    typed_adaptive.retained_history = required_boolean(adaptive, "retained_history", "whole-clip adaptive state");
    typed_adaptive.frame_count = unsigned_integer<std::uint64_t>(adaptive, "frame_count", "whole-clip adaptive state");
    if (typed_adaptive.transport != "none") {
      typed_adaptive.schema = unsigned_integer<std::uint32_t>(adaptive, "schema", "whole-clip adaptive state");
      typed_adaptive.capture = required_string(adaptive, "capture", "whole-clip adaptive state");
    }
    if (typed_adaptive.transport == completion_channel_t::name &&
        (typed_runtime.frame_transport != raw_frame_transport::name ||
         !typed_runtime.follow_mode || typed_runtime.follow_frame_pattern ||
         typed_runtime.follow_done_sentinel || typed_runtime.follow_failed_sentinel ||
         typed_runtime.follow_progress_file ||
         typed_runtime.follow_poll_interval_ms != 0 ||
         typed_runtime.follow_native_input_deletion || typed_adaptive.retained_history ||
         typed_adaptive.schema != sbs_adaptive_state::schema_version ||
         typed_adaptive.capture != sbs_adaptive_state::capture ||
         typed_adaptive.frame_count != result.source_frame_count)) {
      throw contract_error("whole-clip in-memory adaptive state disagrees with its native raw transport");
    }
    result.sbs = parse_whole_clip_sbs(value.at("sbs"));
    const bool raw_sbs = result.sbs.frame_format == "bgra8" || result.sbs.frame_format == "rgba16f";
    if (result.sbs.enabled &&
        (typed_runtime.frame_transport.has_value() != raw_sbs ||
         (raw_sbs && (result.sbs.file_pattern || !result.sbs.atomic_publication ||
                      result.sbs.row_order != "top-down")))) {
      throw contract_error("whole-clip SBS publication disagrees with its frame transport");
    }
    if (result.source_frame_count == 0 || result.source_width == 0 ||
        result.source_height == 0 || result.source_first_sequence == 0 ||
        result.depth_reuse_interval <= 0 ||
        typed_runtime.model != result.model ||
        typed_runtime.depth_reuse_interval != result.depth_reuse_interval ||
        typed_runtime.output_every == 0 || typed_runtime.output_scale <= 0.0 ||
        typed_runtime.resolved_max_output_width <= 0 ||
        typed_runtime.output_eye_width == 0 || typed_runtime.output_eye_height == 0 ||
        static_cast<std::uint64_t>(typed_runtime.output_sbs_width) !=
          2ull * typed_runtime.output_eye_width ||
        typed_runtime.output_sbs_height != typed_runtime.output_eye_height) {
      throw contract_error("whole-clip source or resolved runtime is inconsistent");
    }
    (void) checked_raster_bytes(
      result.source_width, result.source_height, 1, "whole-clip source"
    );
    if ((typed_runtime.depth_width == 0) != (typed_runtime.depth_height == 0) ||
        typed_runtime.depth_width < 0 || typed_runtime.depth_height < 0) {
      throw contract_error("whole-clip resolved depth geometry is inconsistent");
    }
    if (typed_runtime.depth_width > 0) {
      (void) checked_raster_bytes(
        static_cast<std::uint32_t>(typed_runtime.depth_width),
        static_cast<std::uint32_t>(typed_runtime.depth_height),
        1,
        "whole-clip resolved depth"
      );
    }
    (void) checked_raster_bytes(
      typed_runtime.output_sbs_width, typed_runtime.output_sbs_height, 1,
      "whole-clip resolved output"
    );
    return result;
  }

  nlohmann::json to_json(const job_snapshot_contract_t &value) {
    return {
      {"schema", 1},
      {"id", value.id},
      {"state", value.state},
      {"operation", value.operation},
      {"input_path", value.input_path},
      {"output_path", value.output_path ? json(*value.output_path) : json(nullptr)},
      {"output_location", value.output_location ? json(*value.output_location) : json(nullptr)},
      {"codec", value.codec},
      {"scene_cache_max_bytes", value.scene_cache_max_bytes},
      // Schema 1 keeps this field. Direct causal conversion has one behavior: fail.
      {"cache_budget_policy", "fail"},
      {"progress", {
        {"phase", value.progress.phase},
        {"processed_frames", value.progress.processed_frames},
        {"total_frames", value.progress.total_frames ?
                           json(*value.progress.total_frames) : json(nullptr)},
        {"source_time_seconds", value.progress.source_time_seconds ?
                                  json(*value.progress.source_time_seconds) : json(nullptr)},
        {"source_duration_seconds", value.progress.source_duration_seconds ?
                                      json(*value.progress.source_duration_seconds) : json(nullptr)},
      }},
      {"created_at_unix_ms", value.created_at_unix_ms},
      {"started_at_unix_ms", value.started_at_unix_ms ?
                               json(*value.started_at_unix_ms) : json(nullptr)},
      {"ended_at_unix_ms", value.ended_at_unix_ms ?
                             json(*value.ended_at_unix_ms) : json(nullptr)},
      {"error", value.error ? json(*value.error) : json(nullptr)},
      {"worker_result", value.worker_result ? *value.worker_result : json(nullptr)},
    };
  }

  job_snapshot_contract_t parse_job_snapshot_contract(
    const nlohmann::json &value,
    const bool persisted_document
  ) {
    static constexpr std::array base_keys {
      "schema"sv, "id"sv, "state"sv, "operation"sv, "input_path"sv,
      "output_path"sv, "output_location"sv, "codec"sv,
      "scene_cache_max_bytes"sv, "cache_budget_policy"sv, "progress"sv,
      "created_at_unix_ms"sv, "started_at_unix_ms"sv, "ended_at_unix_ms"sv,
      "error"sv, "worker_result"sv,
    };
    json normalized = value;
    if (persisted_document) {
      static constexpr std::array persisted_keys {
        "schema"sv, "id"sv, "state"sv, "operation"sv, "input_path"sv,
        "output_path"sv, "output_location"sv, "codec"sv,
        "scene_cache_max_bytes"sv, "cache_budget_policy"sv, "progress"sv,
        "scene_decisions"sv, "created_at_unix_ms"sv, "started_at_unix_ms"sv,
        "ended_at_unix_ms"sv, "error"sv, "worker_result"sv,
        "retention_sequence"sv, "worker"sv,
      };
      require_allowed_keys(value, persisted_keys, "persisted job");
      if (value.contains("worker")) {
        (void) parse_persisted_worker_fields(value.at("worker"));
      }
      if (value.contains("retention_sequence")) {
        if (unsigned_integer<std::uint64_t>(value, "retention_sequence", "persisted job") == 0) {
          throw contract_error("persisted retention sequence must be positive");
        }
      }
      normalized.erase("retention_sequence");
      normalized.erase("worker");
      // These are the only explicitly supported schema-1 durable variants. One predates
      // beside-input publication. The others carry the retired diagnostic scene summary
      // (scene_decisions, progress.scene_count, progress.current_scene), which is accepted
      // and discarded so a retained job stays readable; it is never written.
      if (!normalized.contains("output_location")) {
        normalized["output_location"] = nullptr;
      }
      normalized.erase("scene_decisions");
      if (normalized.contains("progress") && normalized.at("progress").is_object()) {
        normalized["progress"].erase("scene_count");
        normalized["progress"].erase("current_scene");
      }
    }
    require_exact_keys(normalized, base_keys, "job snapshot");
    require_exact_keys(
      normalized.at("progress"),
      std::array {"phase"sv, "processed_frames"sv, "total_frames"sv,
                  "source_time_seconds"sv, "source_duration_seconds"sv},
      "job progress"
    );
    if (normalized.at("schema") != 1) {
      throw contract_error("job snapshot schema is unsupported");
    }
    job_snapshot_contract_t result;
    result.id = required_string(normalized, "id", "job snapshot");
    result.state = required_string(normalized, "state", "job snapshot");
    result.operation = required_string(normalized, "operation", "job snapshot");
    result.input_path = required_string(normalized, "input_path", "job snapshot");
    if (!normalized.at("output_path").is_null()) {
      result.output_path = required_string(normalized, "output_path", "job snapshot");
    }
    if (!normalized.at("output_location").is_null()) {
      result.output_location = required_string(normalized, "output_location", "job snapshot");
    }
    result.codec = required_string(normalized, "codec", "job snapshot");
    result.scene_cache_max_bytes = unsigned_integer<std::uint64_t>(normalized, "scene_cache_max_bytes", "job snapshot");
    // A job retained by a release that still admitted the retired scene split may record
    // "split". It is read as the only remaining behavior and never written.
    const auto cache_budget_policy =
      required_string(normalized, "cache_budget_policy", "job snapshot");
    if (cache_budget_policy != "fail" &&
        !(persisted_document && cache_budget_policy == "split")) {
      throw contract_error("job snapshot cache budget policy is unsupported");
    }
    const auto &progress = normalized.at("progress");
    result.progress.phase = required_string(progress, "phase", "job progress");
    result.progress.processed_frames = unsigned_integer<std::uint64_t>(progress, "processed_frames", "job progress");
    result.progress.total_frames = optional_value<std::uint64_t>(progress, "total_frames", [&]() {
      return unsigned_integer<std::uint64_t>(progress, "total_frames", "job progress");
    });
    result.progress.source_time_seconds = optional_value<double>(progress, "source_time_seconds", [&]() {
      return finite_number(progress, "source_time_seconds", "job progress");
    });
    result.progress.source_duration_seconds = optional_value<double>(progress, "source_duration_seconds", [&]() {
      return finite_number(progress, "source_duration_seconds", "job progress");
    });
    result.created_at_unix_ms = signed_integer(normalized, "created_at_unix_ms", "job snapshot");
    result.started_at_unix_ms = optional_value<std::int64_t>(normalized, "started_at_unix_ms", [&]() {
      return signed_integer(normalized, "started_at_unix_ms", "job snapshot");
    });
    result.ended_at_unix_ms = optional_value<std::int64_t>(normalized, "ended_at_unix_ms", [&]() {
      return signed_integer(normalized, "ended_at_unix_ms", "job snapshot");
    });
    if (!normalized.at("error").is_null()) {
      if (!normalized.at("error").is_string()) {
        throw contract_error("job error must be a string or null");
      }
      result.error = normalized.at("error").get<std::string>();
    }
    if (!normalized.at("worker_result").is_null()) {
      if (!normalized.at("worker_result").is_object()) {
        throw contract_error("job worker result summary must be an object or null");
      }
      result.worker_result = normalized.at("worker_result");
    }
    if (result.progress.phase.size() > 64 ||
        (result.progress.total_frames &&
         (result.progress.processed_frames > *result.progress.total_frames ||
          *result.progress.total_frames == 0)) ||
        (result.progress.source_time_seconds && *result.progress.source_time_seconds < 0.0) ||
        (result.progress.source_duration_seconds &&
         *result.progress.source_duration_seconds <= 0.0)) {
      throw contract_error("job progress has invalid bounded counters or time values");
    }
    return result;
  }

  nlohmann::json to_json(const persisted_job_contract_t &value) {
    auto result = to_json(value.snapshot);
    if (value.retention_sequence) {
      result["retention_sequence"] = *value.retention_sequence;
    }
    if (value.worker) {
      result["worker"] = {
        {"spec", value.worker->spec_path},
        {"spec_sha256", value.worker->spec_sha256},
        {"progress", value.worker->progress_path},
        {"result", value.worker->result_path},
        {"log", value.worker->log_path},
        {"result_directory", value.worker->result_directory},
        {"staging_output", value.worker->staging_output ?
                             json(*value.worker->staging_output) : json(nullptr)},
        {"ffmpeg_path", value.worker->ffmpeg_path},
        {"ffmpeg_version", value.worker->ffmpeg_version},
        {"ffprobe_path", value.worker->ffprobe_path},
        {"ffprobe_version", value.worker->ffprobe_version},
      };
    }
    return result;
  }

  persisted_job_contract_t parse_persisted_job_contract(
    const nlohmann::json &value
  ) {
    persisted_job_contract_t result {
      .snapshot = parse_job_snapshot_contract(value, true),
    };
    if (value.contains("retention_sequence")) {
      result.retention_sequence = unsigned_integer<std::uint64_t>(
        value, "retention_sequence", "persisted job"
      );
      if (*result.retention_sequence == 0) {
        throw contract_error("persisted retention sequence must be positive");
      }
    }
    if (value.contains("worker")) {
      result.worker = parse_persisted_worker_fields(value.at("worker"));
    }
    return result;
  }
}  // namespace offline_sbs::wire
