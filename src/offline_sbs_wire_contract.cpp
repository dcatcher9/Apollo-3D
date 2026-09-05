/**
 * @file src/offline_sbs_wire_contract.cpp
 * @brief Typed JSON wire contracts shared by the native offline SBS manager and worker.
 */

#include "offline_sbs_wire_contract.h"

#include "depth_coordinate_v2.h"
#include "host_sbs_shader_cache.h"
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
      {"schema", 2},
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
        {"cut_state", "authenticated-online-pulse-count"},
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
    if (value.at("schema") != 2 || value.at("python_dependency") != false) {
      throw contract_error("worker specification is not native schema 2");
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
      std::array {"implementation"sv, "cut_state"sv, "lookahead"sv,
                  "scene_cache"sv, "replay"sv},
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
          {"cut_state", "authenticated-online-pulse-count"},
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
      const bool raw_transport = runtime.frame_transport == "bounded-raw-memory-v1";
      const auto expected_follow_format =
        raw_transport ? (source.color_mode == "sdr" ? std::string_view {"bgra8"} : std::string_view {"gbrpf32le"}) :
          (source.color_mode == "sdr" ? std::string_view {"bmp"} : std::string_view {"pfm"});
      const auto expected_input_frame_format =
        raw_transport ?
          (source.color_mode == "sdr" ? std::string_view {"sRGB-BGRA8-raw"} :
                                       std::string_view {"linear-scRGB-f32-planar"}) :
          (source.color_mode == "sdr" ? std::string_view {"sRGB-BMP-WIC"} :
                                       std::string_view {"linear-scRGB-f32-pfm"});
      if (
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
        analysis.adaptive_state.transport != "atomic-latest-v1" ||
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
        runtime.follow_format.value_or("") != expected_follow_format ||
        (raw_transport ? runtime.follow_frame_pattern.has_value() :
          runtime.follow_frame_pattern.value_or("") != "frame_%010d." + std::string {expected_follow_format}) ||
        runtime.input_frame_format != expected_input_frame_format ||
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
            sbs.frame_format != (raw_transport ?
              (source.color_mode == "sdr" ? "bgra8" : "rgba16f") :
              runtime.output_frame_format) ||
            sbs.transfer != runtime.output_transfer ||
            sbs.primaries != runtime.output_primaries ||
            sbs.row_order != runtime.output_row_order ||
            (raw_transport &&
              (runtime.output_frame_format != (source.color_mode == "sdr" ? "sRGB-BGRA8-raw" : "linear-scRGB-f16-raw") ||
               runtime.output_ffmpeg_pixel_format != (source.color_mode == "sdr" ? "bgra" : "gbrpf32le") ||
               sbs.ffmpeg_pixel_format != runtime.output_ffmpeg_pixel_format)) ||
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
      const auto total_scenes = result.total_scene_count.value_or(result.scenes.size());
      const auto first_scene_id = total_scenes - result.scenes.size() + 1;
      std::uint64_t expected_start = first_scene_id == 1 ? 1 : result.scenes.front().start_sequence;
      for (std::size_t index = 0; index < result.scenes.size(); ++index) {
        const auto &scene = result.scenes[index];
        if (
          scene.scene_id != first_scene_id + index ||
          scene.semantic_scene_id != first_scene_id + index ||
          scene.start_sequence != expected_start ||
          scene.evidence.source_frame_count != scene.frame_count ||
          scene.cache_bytes != 0 ||
          scene.cut_state_semantics != "causal-production-exact" ||
          scene.boundary.decision !=
            (index + 1u == result.scenes.size() ?
               boundary_decision_e::end_of_stream :
               boundary_decision_e::confirmed) ||
          scene.boundary.semantic_cut !=
            (index + 1u != result.scenes.size())
        ) {
          throw contract_error(
            "worker scenes are not exact causal production epochs"
          );
        }
        if (index + 1u != result.scenes.size() &&
            scene.boundary.final_sequence != scene.end_sequence_exclusive) {
          throw contract_error(
            "worker causal scene boundary does not start the next epoch"
          );
        }
        expected_start = scene.end_sequence_exclusive;
      }
      if (expected_start != source.frame_count + 1u) {
        throw contract_error("worker scenes do not cover every source frame exactly once");
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
    json scenes = json::array();
    for (const auto &scene : value.scenes) {
      scenes.push_back(scene_record_json(scene));
    }
    json replays = json::array();
    for (const auto &replay : value.replay_contracts) {
      replays.push_back(to_json(replay));
    }
    json result {
      {"schema", value.total_scene_count ? 3 : 2},
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
      {"scene_count", value.total_scene_count.value_or(value.scenes.size())},
      {"scenes", std::move(scenes)},
      {"scene_audit", value.scene_audit_path},
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
    if (value.total_scene_count) {
      result["scene_audit_manifest_sha256"] = optional_json(value.scene_audit_manifest_sha256);
    }
    (void) parse_worker_result_contract(result);
    return result;
  }

  worker_result_contract_t parse_worker_result_contract(const nlohmann::json &value) {
    static constexpr std::array keys {
      "schema"sv, "job_id"sv, "status"sv, "operation"sv, "codec"sv,
      "worker_spec_sha256"sv, "python_dependency"sv, "source"sv,
      "scene_count"sv, "scenes"sv, "scene_audit"sv, "analysis_contract"sv,
      "replay_contracts"sv, "cache"sv, "timeline_contract"sv,
      "staging_identity"sv, "output"sv,
    };
    const bool paged = value.value("schema", 0) == 3;
    if (paged) {
      require_keys_with_optional(value, keys, "scene_audit_manifest_sha256", "worker result");
    } else {
      require_exact_keys(value, keys, "worker result");
    }
    if ((!paged && value.at("schema") != 2) || value.at("status") != "complete" ||
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

    const auto scene_count = unsigned_integer<std::uint64_t>(
      value, "scene_count", "worker result"
    );
    if (!value.at("scenes").is_array() || scene_count == 0 ||
        value.at("scenes").size() != (paged ? std::min<std::uint64_t>(scene_count, scene_audit_recent_scenes) : scene_count) ||
        scene_count > (paged ? max_paged_scene_count : max_serialized_scene_count)) {
      throw contract_error("worker result scene count is invalid");
    }
    if (paged) {
      result.total_scene_count = scene_count;
      result.scene_audit_manifest_sha256 = required_string(value, "scene_audit_manifest_sha256", "worker result");
      if (!valid_sha256_hex(*result.scene_audit_manifest_sha256)) {
        throw contract_error("worker audit manifest digest is not lowercase SHA-256");
      }
    }
    for (const auto &scene : value.at("scenes")) {
      result.scenes.push_back(parse_scene_record(scene));
    }
    result.scene_audit_path = required_string(value, "scene_audit", "worker result");
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

  nlohmann::json to_json(const scene_plan_contract_t &value) {
    json scenes = json::array();
    for (const auto &scene : value.scenes) {
      scenes.push_back({
        {"start_sequence", scene.start_sequence},
        {"end_sequence_exclusive", scene.end_sequence_exclusive},
      });
    }
    return {
      {"schema", 2},
      {"version", "scene-plan-v2"},
      {"cache_contract_schema", scene_cache_contract_schema},
      {"scenes", std::move(scenes)},
    };
  }

  scene_plan_contract_t parse_scene_plan_contract(const nlohmann::json &value) {
    require_exact_keys(
      value,
      std::array {"schema"sv, "version"sv, "cache_contract_schema"sv, "scenes"sv},
      "scene plan"
    );
    if (value.at("schema") != 2 || value.at("version") != "scene-plan-v2" ||
        value.at("cache_contract_schema") != scene_cache_contract_schema ||
        !value.at("scenes").is_array() || value.at("scenes").size() != 1) {
      throw contract_error("scene plan is not the one-scene scene-plan-v2 contract");
    }
    scene_plan_contract_t result;
    for (const auto &scene : value.at("scenes")) {
      require_exact_keys(
        scene,
        std::array {"start_sequence"sv, "end_sequence_exclusive"sv},
        "scene plan range"
      );
      const auto start = unsigned_integer<std::uint64_t>(
        scene, "start_sequence", "scene plan range"
      );
      const auto end = unsigned_integer<std::uint64_t>(
        scene, "end_sequence_exclusive", "scene plan range"
      );
      if (start == 0 || end <= start || end > 10000000001ull) {
        throw contract_error("scene plan has an empty or out-of-range sequence");
      }
      result.scenes.push_back({start, end});
    }
    return result;
  }

  nlohmann::json to_json(const scene_cache_contract_t &value) {
    (void) checked_raster_bytes(
      value.source.width, value.source.height, 1, "scene cache source"
    );
    const auto depth_bytes = checked_raster_bytes(
      value.depth_width, value.depth_height, sizeof(float), "scene cache depth"
    );
    (void) checked_raster_bytes(
      value.packed_sbs.width, value.packed_sbs.height, 1,
      "scene cache packed SBS"
    );
    if (static_cast<std::uint64_t>(value.packed_sbs.width) !=
          2ull * value.packed_sbs.eye_width ||
        value.packed_sbs.height != value.packed_sbs.eye_height) {
      throw contract_error("scene cache packed SBS geometry is inconsistent");
    }
    json result {
      {"schema", scene_cache_contract_schema},
      {"status", value.status},
      {"source", {
        {"width", value.source.width},
        {"height", value.source.height},
        {"frame_format", value.source.frame_format},
        {"texture_format", value.source.texture_format},
        {"color_space", value.source.color_space},
      }},
      {"depth", {
        {"width", value.depth_width},
        {"height", value.depth_height},
        {"dxgi_format", "R32_FLOAT"},
        {"dtype", "float32-le"},
        {"layout", "row-major"},
        {"row_order", "top-down"},
        {"semantics", "depth-coordinate-v2-signed-final-parallax-source-u"},
        {"source_u_limit", static_cast<double>(
          models::depth_coordinate_v2::direct_container_limit)},
        {"file_pattern", "frame_%010d.depth.r32f"},
        {"bytes_per_frame", depth_bytes},
      }},
      {"state", {
        {"schema", 2},
        {"source", "depth_coordinate_v2.ParallaxState[0..2]"},
        {"word_count", models::depth_coordinate_v2::state_words_t {}.size()},
        {"dtype", "uint32-le"},
        {"layout", "raw-word-order"},
        {"contract_schema", models::depth_coordinate_v2::contract_schema},
        {"contract_tag", models::depth_coordinate_v2::contract_tag},
        {"file_pattern", "frame_%010d.state.u32"},
        {"bytes_per_frame", sizeof(models::depth_coordinate_v2::state_words_t)},
      }},
      {"render_config", {
        {"model", value.render.model},
        {"model_url", value.render.model_url},
        {"renderer", "depth-coordinate-v2-live-signed-parallax"},
        {"producer_source_closure_sha256", std::string {
          models::depth_coordinate_v2::shader_source_closure_sha256}},
        {"renderer_source_closure_sha256", std::string {
          models::host_sbs_shader_cache::parallax_v2_live_renderer_source_closure_sha256}},
        {"pop_strength", value.render.pop_strength},
        {"simulate_hdr", value.render.simulate_hdr},
        {"hdr_scale", value.render.hdr_scale},
        {"depth_reuse_interval", value.render.depth_reuse_interval},
        {"requested_eye_width", value.render.requested_eye_width},
        {"requested_eye_height", value.render.requested_eye_height},
        {"output_scale", value.render.output_scale},
        {"resolved_max_output_width", value.render.resolved_max_output_width},
      }},
      {"packed_sbs", {
        {"eye_width", value.packed_sbs.eye_width},
        {"eye_height", value.packed_sbs.eye_height},
        {"width", value.packed_sbs.width},
        {"height", value.packed_sbs.height},
        {"texture_format", value.packed_sbs.texture_format},
        {"frame_format", value.packed_sbs.frame_format},
        {"file_extension", value.packed_sbs.file_extension},
        {"file_pattern", "sbs_%010d." + value.packed_sbs.file_extension},
        {"atomic_replay_publication", true},
      }},
      {"first_sequence", 1},
      {"processed_count", value.processed_count},
      {"last_completed_sequence", value.processed_count > 0 ?
                                      json(value.processed_count) : json(nullptr)},
      {"frame_count", value.frame_count ? json(*value.frame_count) : json(nullptr)},
      {"atomic_frame_publication", true},
      {"native_source_deletion", false},
      {"cache_budget", {
        {"enforced_by", "wrapper"},
        {"native_limit_bytes", nullptr},
        {"on_write_failure", "fail-closed"},
        {"native_eviction", false},
      }},
    };
    (void) parse_scene_cache_contract(result);
    return result;
  }

  scene_cache_contract_t parse_scene_cache_contract(const nlohmann::json &value) {
    require_exact_keys(
      value,
      std::array {
        "schema"sv, "status"sv, "source"sv, "depth"sv, "state"sv,
        "render_config"sv, "packed_sbs"sv, "first_sequence"sv,
        "processed_count"sv, "last_completed_sequence"sv, "frame_count"sv,
        "atomic_frame_publication"sv, "native_source_deletion"sv,
        "cache_budget"sv,
      },
      "scene cache contract"
    );
    if (value.at("schema") != scene_cache_contract_schema) {
      throw contract_error("scene cache contract schema is unsupported");
    }
    require_exact_keys(
      value.at("source"),
      std::array {"width"sv, "height"sv, "frame_format"sv, "texture_format"sv,
                  "color_space"sv},
      "scene cache source"
    );
    require_exact_keys(
      value.at("depth"),
      std::array {
        "width"sv, "height"sv, "dxgi_format"sv, "dtype"sv, "layout"sv,
        "row_order"sv, "semantics"sv, "source_u_limit"sv, "file_pattern"sv,
        "bytes_per_frame"sv,
      },
      "scene cache depth"
    );
    require_exact_keys(
      value.at("state"),
      std::array {
        "schema"sv, "source"sv, "word_count"sv, "dtype"sv, "layout"sv,
        "contract_schema"sv, "contract_tag"sv, "file_pattern"sv,
        "bytes_per_frame"sv,
      },
      "scene cache state"
    );
    require_exact_keys(
      value.at("render_config"),
      std::array {
        "model"sv, "model_url"sv, "renderer"sv,
        "producer_source_closure_sha256"sv, "renderer_source_closure_sha256"sv,
        "pop_strength"sv, "simulate_hdr"sv, "hdr_scale"sv,
        "depth_reuse_interval"sv, "requested_eye_width"sv,
        "requested_eye_height"sv, "output_scale"sv,
        "resolved_max_output_width"sv,
      },
      "scene cache render config"
    );
    require_exact_keys(
      value.at("packed_sbs"),
      std::array {
        "eye_width"sv, "eye_height"sv, "width"sv, "height"sv,
        "texture_format"sv, "frame_format"sv, "file_extension"sv,
        "file_pattern"sv, "atomic_replay_publication"sv,
      },
      "scene cache packed SBS"
    );
    require_exact_keys(
      value.at("cache_budget"),
      std::array {"enforced_by"sv, "native_limit_bytes"sv,
                  "on_write_failure"sv, "native_eviction"sv},
      "scene cache budget"
    );

    scene_cache_contract_t result;
    result.status = required_string(value, "status", "scene cache contract");
    if (result.status != "running" && result.status != "complete") {
      throw contract_error("scene cache status is invalid");
    }
    const auto &source = value.at("source");
    result.source.width = unsigned_integer<std::uint32_t>(source, "width", "scene cache source");
    result.source.height = unsigned_integer<std::uint32_t>(source, "height", "scene cache source");
    result.source.frame_format = required_string(source, "frame_format", "scene cache source");
    result.source.texture_format = required_string(source, "texture_format", "scene cache source");
    result.source.color_space = required_string(source, "color_space", "scene cache source");
    const auto &depth = value.at("depth");
    result.depth_width = unsigned_integer<std::uint32_t>(depth, "width", "scene cache depth");
    result.depth_height = unsigned_integer<std::uint32_t>(depth, "height", "scene cache depth");
    (void) checked_raster_bytes(
      result.source.width, result.source.height, 1, "scene cache source"
    );
    const auto depth_bytes = checked_raster_bytes(
      result.depth_width, result.depth_height, sizeof(float), "scene cache depth"
    );
    const auto &state = value.at("state");
    const auto &render = value.at("render_config");
    result.render.model = required_string(render, "model", "scene cache render config");
    result.render.model_url = required_string(render, "model_url", "scene cache render config");
    result.render.pop_strength = finite_number(render, "pop_strength", "scene cache render config");
    result.render.simulate_hdr = required_boolean(render, "simulate_hdr", "scene cache render config");
    result.render.hdr_scale = finite_number(render, "hdr_scale", "scene cache render config");
    result.render.depth_reuse_interval = signed_integer<int>(
      render, "depth_reuse_interval", "scene cache render config");
    result.render.requested_eye_width = signed_integer<int>(
      render, "requested_eye_width", "scene cache render config");
    result.render.requested_eye_height = signed_integer<int>(
      render, "requested_eye_height", "scene cache render config");
    result.render.output_scale = finite_number(render, "output_scale", "scene cache render config");
    result.render.resolved_max_output_width = signed_integer<int>(
      render, "resolved_max_output_width", "scene cache render config");
    const auto &packed = value.at("packed_sbs");
    result.packed_sbs.eye_width = unsigned_integer<std::uint32_t>(packed, "eye_width", "scene cache packed SBS");
    result.packed_sbs.eye_height = unsigned_integer<std::uint32_t>(packed, "eye_height", "scene cache packed SBS");
    result.packed_sbs.width = unsigned_integer<std::uint32_t>(packed, "width", "scene cache packed SBS");
    result.packed_sbs.height = unsigned_integer<std::uint32_t>(packed, "height", "scene cache packed SBS");
    result.packed_sbs.texture_format = required_string(packed, "texture_format", "scene cache packed SBS");
    result.packed_sbs.frame_format = required_string(packed, "frame_format", "scene cache packed SBS");
    result.packed_sbs.file_extension = required_string(packed, "file_extension", "scene cache packed SBS");
    (void) checked_raster_bytes(
      result.packed_sbs.width, result.packed_sbs.height, 1,
      "scene cache packed SBS"
    );
    result.processed_count = unsigned_integer<std::uint64_t>(value, "processed_count", "scene cache contract");
    if (!value.at("frame_count").is_null()) {
      result.frame_count = unsigned_integer<std::uint64_t>(value, "frame_count", "scene cache contract");
    }
    const auto expected_last = result.processed_count > 0 ?
                                 json(result.processed_count) : json(nullptr);
    if (result.source.width == 0 || result.source.height == 0 ||
        result.depth_width == 0 || result.depth_height == 0 ||
        value.at("first_sequence") != 1 || result.processed_count > 9999999999ull ||
        value.at("last_completed_sequence") != expected_last ||
        (result.status == "complete" ?
           (!result.frame_count || *result.frame_count == 0 ||
            *result.frame_count != result.processed_count) : result.frame_count.has_value()) ||
        value.at("atomic_frame_publication") != true ||
        value.at("native_source_deletion") != false ||
        depth.at("dxgi_format") != "R32_FLOAT" || depth.at("dtype") != "float32-le" ||
        depth.at("layout") != "row-major" || depth.at("row_order") != "top-down" ||
        depth.at("semantics") != "depth-coordinate-v2-signed-final-parallax-source-u" ||
        depth.at("source_u_limit") != static_cast<double>(models::depth_coordinate_v2::direct_container_limit) ||
        depth.at("file_pattern") != "frame_%010d.depth.r32f" ||
        unsigned_integer<std::uint64_t>(depth, "bytes_per_frame", "scene cache depth") != depth_bytes ||
        state.at("schema") != 2 || state.at("source") != "depth_coordinate_v2.ParallaxState[0..2]" ||
        state.at("word_count") != models::depth_coordinate_v2::state_words_t {}.size() ||
        state.at("dtype") != "uint32-le" || state.at("layout") != "raw-word-order" ||
        state.at("contract_schema") != models::depth_coordinate_v2::contract_schema ||
        state.at("contract_tag") != models::depth_coordinate_v2::contract_tag ||
        state.at("file_pattern") != "frame_%010d.state.u32" ||
        state.at("bytes_per_frame") != sizeof(models::depth_coordinate_v2::state_words_t) ||
        render.at("renderer") != "depth-coordinate-v2-live-signed-parallax" ||
        render.at("producer_source_closure_sha256") != models::depth_coordinate_v2::shader_source_closure_sha256 ||
        render.at("renderer_source_closure_sha256") != models::host_sbs_shader_cache::parallax_v2_live_renderer_source_closure_sha256 ||
        result.render.pop_strength < 0.25 || result.render.pop_strength > 2.0 ||
        result.render.depth_reuse_interval != 1 || result.render.output_scale <= 0.0 ||
        result.render.output_scale > 4.0 || result.render.resolved_max_output_width <= 0 ||
        result.packed_sbs.eye_width == 0 || result.packed_sbs.eye_height == 0 ||
        static_cast<std::uint64_t>(result.packed_sbs.width) !=
          2ull * result.packed_sbs.eye_width ||
        result.packed_sbs.height != result.packed_sbs.eye_height ||
        (result.packed_sbs.file_extension != "png" && result.packed_sbs.file_extension != "pfm") ||
        packed.at("file_pattern") != "sbs_%010d." + result.packed_sbs.file_extension ||
        packed.at("atomic_replay_publication") != true ||
        value.at("cache_budget") != json {
          {"enforced_by", "wrapper"}, {"native_limit_bytes", nullptr},
          {"on_write_failure", "fail-closed"}, {"native_eviction", false}}) {
      throw contract_error("scene cache contract has invalid sequence, geometry, or semantics");
    }
    return result;
  }

  nlohmann::json boundary_audit_json(const boundary_audit_t &value) {
    return {
      {"proposal_sequences", value.proposal_sequences},
      {"proposal_frame_ids", value.proposal_frame_ids},
      {"final_sequence", value.final_sequence},
      {"decision", boundary_decision_name(value.decision)},
      {"reason", value.reason},
      {"accepted", value.accepted},
      {"semantic_cut", value.semantic_cut},
      {"truncated", value.truncated},
      {"budget_forced", value.budget_forced},
      {"revision_depth_updates", value.revision_depth_updates},
      {"revision_source_frames", value.revision_source_frames},
      {"candidate_count", value.candidate_count},
      {"selected_evidence_score", value.selected_evidence_score},
      {"evidence_window_first_sequence", value.evidence_window_first_sequence},
      {"evidence_window_last_sequence", value.evidence_window_last_sequence},
      {"selected_sequence", value.selected_sequence},
      {"selected_frame_id", value.selected_frame_id},
      {"selected_depth_change_fraction", value.selected_depth_change_fraction},
      {"selected_raw_rgb_change_fraction", value.selected_raw_rgb_change_fraction},
      {"selected_structural_change_fraction", value.selected_structural_change_fraction},
      {"selected_appearance_qualified", value.selected_appearance_qualified},
      {"selected_geometry_qualified", value.selected_geometry_qualified},
      {"selected_relative_geometry_spike", value.selected_relative_geometry_spike},
    };
  }

  boundary_audit_t parse_boundary_audit(const nlohmann::json &value) {
    require_exact_keys(
      value,
      std::array {
        "proposal_sequences"sv, "proposal_frame_ids"sv, "final_sequence"sv,
        "decision"sv, "reason"sv, "accepted"sv, "semantic_cut"sv,
        "truncated"sv, "budget_forced"sv, "revision_depth_updates"sv,
        "revision_source_frames"sv, "candidate_count"sv,
        "selected_evidence_score"sv, "evidence_window_first_sequence"sv,
        "evidence_window_last_sequence"sv, "selected_sequence"sv,
        "selected_frame_id"sv, "selected_depth_change_fraction"sv,
        "selected_raw_rgb_change_fraction"sv,
        "selected_structural_change_fraction"sv,
        "selected_appearance_qualified"sv, "selected_geometry_qualified"sv,
        "selected_relative_geometry_spike"sv,
      },
      "scene boundary audit"
    );
    boundary_audit_t result;
    if (!value.at("proposal_sequences").is_array() ||
        !value.at("proposal_frame_ids").is_array()) {
      throw contract_error("scene boundary proposals must be arrays");
    }
    for (const auto &sequence : value.at("proposal_sequences")) {
      json sequence_object {{"sequence", sequence}};
      const auto parsed = unsigned_integer<std::uint64_t>(
        sequence_object, "sequence", "scene boundary proposal"
      );
      if (parsed == 0) {
        throw contract_error("scene boundary proposal sequence is invalid");
      }
      result.proposal_sequences.push_back(parsed);
    }
    for (const auto &frame_id : value.at("proposal_frame_ids")) {
      if (!frame_id.is_string() || frame_id.get_ref<const std::string &>().empty() ||
          frame_id.get_ref<const std::string &>().size() > max_scene_frame_id_bytes) {
        throw contract_error("scene boundary proposal frame identity is invalid");
      }
      result.proposal_frame_ids.push_back(frame_id.get<std::string>());
    }
    if (result.proposal_sequences.size() != result.proposal_frame_ids.size()) {
      throw contract_error("scene boundary proposal identities disagree");
    }
    const auto optional_u64 = [&](const std::string_view key) {
      return optional_value<std::uint64_t>(value, key, [&]() {
        const auto parsed = unsigned_integer<std::uint64_t>(value, key, "scene boundary audit");
        if (parsed == 0) {
          throw contract_error("scene boundary optional sequence must be positive");
        }
        return parsed;
      });
    };
    const auto optional_fraction = [&](const std::string_view key) {
      return optional_value<float>(value, key, [&]() {
        const auto parsed = finite_number(value, key, "scene boundary audit");
        if (parsed < 0.0 || parsed > 1.0) {
          throw contract_error("scene boundary fraction must be in [0, 1]");
        }
        return static_cast<float>(parsed);
      });
    };
    result.final_sequence = optional_u64("final_sequence");
    const auto decision = required_string(value, "decision", "scene boundary audit");
    static constexpr std::array decisions {
      std::pair {"confirmed"sv, boundary_decision_e::confirmed},
      std::pair {"moved_to_correlated_evidence"sv,
                 boundary_decision_e::moved_to_correlated_evidence},
      std::pair {"merged_duplicate_proposals"sv,
                 boundary_decision_e::merged_duplicate_proposals},
      std::pair {"confirmed_causal_fallback"sv,
                 boundary_decision_e::confirmed_causal_fallback},
      std::pair {"rejected_supported_flash_return"sv,
                 boundary_decision_e::rejected_supported_flash_return},
      std::pair {"rejected_unsupported_proposal"sv,
                 boundary_decision_e::rejected_unsupported_proposal},
      std::pair {"rejected_minimum_scene_length"sv,
                 boundary_decision_e::rejected_minimum_scene_length},
      std::pair {"administrative_cache_split"sv,
                 boundary_decision_e::administrative_cache_split},
      std::pair {"end_of_stream"sv, boundary_decision_e::end_of_stream},
    };
    const auto found = std::ranges::find_if(decisions, [&](const auto &candidate) {
      return candidate.first == decision;
    });
    if (found == decisions.end()) {
      throw contract_error("scene boundary decision is unknown");
    }
    result.decision = found->second;
    if (!value.at("reason").is_string()) {
      throw contract_error("scene boundary reason must be a string");
    }
    result.reason = value.at("reason").get<std::string>();
    result.accepted = required_boolean(value, "accepted", "scene boundary audit");
    result.semantic_cut = required_boolean(value, "semantic_cut", "scene boundary audit");
    result.truncated = required_boolean(value, "truncated", "scene boundary audit");
    result.budget_forced = required_boolean(value, "budget_forced", "scene boundary audit");
    result.revision_depth_updates = signed_integer(value, "revision_depth_updates", "scene boundary audit");
    result.revision_source_frames = signed_integer(value, "revision_source_frames", "scene boundary audit");
    result.candidate_count = unsigned_integer<std::size_t>(value, "candidate_count", "scene boundary audit");
    result.selected_evidence_score = optional_value<float>(value, "selected_evidence_score", [&]() {
      return static_cast<float>(finite_number(value, "selected_evidence_score", "scene boundary audit"));
    });
    result.evidence_window_first_sequence = optional_u64("evidence_window_first_sequence");
    result.evidence_window_last_sequence = optional_u64("evidence_window_last_sequence");
    result.selected_sequence = optional_u64("selected_sequence");
    result.selected_frame_id = optional_value<std::string>(value, "selected_frame_id", [&]() {
      const auto parsed = required_string(value, "selected_frame_id", "scene boundary audit");
      if (parsed.size() > max_scene_frame_id_bytes) {
        throw contract_error("scene boundary selected frame identity is too long");
      }
      return parsed;
    });
    result.selected_depth_change_fraction = optional_fraction("selected_depth_change_fraction");
    result.selected_raw_rgb_change_fraction = optional_fraction("selected_raw_rgb_change_fraction");
    result.selected_structural_change_fraction = optional_fraction("selected_structural_change_fraction");
    result.selected_appearance_qualified = required_boolean(value, "selected_appearance_qualified", "scene boundary audit");
    result.selected_geometry_qualified = required_boolean(value, "selected_geometry_qualified", "scene boundary audit");
    result.selected_relative_geometry_spike = required_boolean(value, "selected_relative_geometry_spike", "scene boundary audit");
    return result;
  }

  nlohmann::json scene_evidence_json(const scene_evidence_t &value) {
    return {
      {"source_frame_count", value.source_frame_count},
      {"depth_update_count", value.depth_update_count},
      {"appearance_veto_count", value.appearance_veto_count},
      {"depth_change_max", value.depth_change_max},
    };
  }

  scene_evidence_t parse_scene_evidence(const nlohmann::json &value) {
    require_exact_keys(
      value,
      std::array {"source_frame_count"sv, "depth_update_count"sv,
                  "appearance_veto_count"sv, "depth_change_max"sv},
      "scene evidence"
    );
    scene_evidence_t result;
    result.source_frame_count = unsigned_integer<std::size_t>(value, "source_frame_count", "scene evidence");
    result.depth_update_count = unsigned_integer<std::size_t>(value, "depth_update_count", "scene evidence");
    result.appearance_veto_count = unsigned_integer<std::size_t>(value, "appearance_veto_count", "scene evidence");
    result.depth_change_max = optional_value<float>(value, "depth_change_max", [&]() {
      const auto parsed = finite_number(value, "depth_change_max", "scene evidence");
      if (parsed < 0.0 || parsed > 1.0) {
        throw contract_error("scene maximum depth change must be in [0, 1]");
      }
      return static_cast<float>(parsed);
    });
    return result;
  }

  nlohmann::json scene_record_json(const scene_plan_t &value) {
    return {
      {"scene_id", value.scene_id},
      {"semantic_scene_id", value.semantic_scene_id},
      {"start_sequence", value.start_sequence},
      {"end_sequence_exclusive", value.end_sequence_exclusive},
      {"frame_count", value.frame_count},
      {"cache_bytes", value.cache_bytes},
      {"start_pts_seconds", value.start_pts_seconds},
      {"end_pts_seconds_exclusive", value.end_pts_seconds_exclusive},
      {"evidence", scene_evidence_json(value.evidence)},
      {"boundary", boundary_audit_json(value.boundary)},
      {"ground_truth", value.ground_truth},
      {"cut_state_semantics", value.cut_state_semantics},
      {"known_limit", value.known_limit},
    };
  }

  scene_plan_t parse_scene_record(const nlohmann::json &value) {
    require_exact_keys(
      value,
      std::array {
        "scene_id"sv, "semantic_scene_id"sv, "start_sequence"sv,
        "end_sequence_exclusive"sv, "frame_count"sv, "cache_bytes"sv,
        "start_pts_seconds"sv, "end_pts_seconds_exclusive"sv, "evidence"sv,
        "boundary"sv, "ground_truth"sv, "cut_state_semantics"sv,
        "known_limit"sv,
      },
      "scene record"
    );
    scene_plan_t result;
    result.scene_id = unsigned_integer<std::uint64_t>(value, "scene_id", "scene record");
    result.semantic_scene_id = unsigned_integer<std::uint64_t>(value, "semantic_scene_id", "scene record");
    result.start_sequence = unsigned_integer<std::uint64_t>(value, "start_sequence", "scene record");
    result.end_sequence_exclusive = unsigned_integer<std::uint64_t>(value, "end_sequence_exclusive", "scene record");
    result.frame_count = unsigned_integer<std::uint64_t>(value, "frame_count", "scene record");
    result.cache_bytes = unsigned_integer<std::uint64_t>(value, "cache_bytes", "scene record");
    result.start_pts_seconds = optional_value<double>(value, "start_pts_seconds", [&]() {
      return finite_number(value, "start_pts_seconds", "scene record");
    });
    result.end_pts_seconds_exclusive = optional_value<double>(value, "end_pts_seconds_exclusive", [&]() {
      return finite_number(value, "end_pts_seconds_exclusive", "scene record");
    });
    result.evidence = parse_scene_evidence(value.at("evidence"));
    result.boundary = parse_boundary_audit(value.at("boundary"));
    result.ground_truth = required_boolean(value, "ground_truth", "scene record");
    result.cut_state_semantics = required_string(value, "cut_state_semantics", "scene record");
    result.known_limit = required_string(value, "known_limit", "scene record");
    if (result.scene_id == 0 || result.semantic_scene_id == 0 || result.start_sequence == 0 ||
        result.end_sequence_exclusive <= result.start_sequence ||
        result.frame_count != result.end_sequence_exclusive - result.start_sequence ||
        result.ground_truth ||
        (result.start_pts_seconds && result.end_pts_seconds_exclusive &&
         *result.end_pts_seconds_exclusive <= *result.start_pts_seconds)) {
      throw contract_error("scene record identity, range, or claims are invalid");
    }
    return result;
  }

  std::string scene_audit_page_filename(const std::uint32_t index) {
    return "scene-audit-page-" + std::to_string(index) + ".json";
  }

  nlohmann::json to_json(const scene_audit_page_t &value) {
    json scenes = json::array();
    json boundaries = json::array();
    for (const auto &scene : value.scenes) scenes.push_back(scene_record_json(scene));
    for (const auto &boundary : value.boundary_revisions) boundaries.push_back(boundary_audit_json(boundary));
    return {{"schema", 1}, {"version", "whole-clip-scene-audit-page-v1"},
            {"index", value.index}, {"scenes", std::move(scenes)},
            {"boundary_revisions", std::move(boundaries)}};
  }

  scene_audit_page_t parse_scene_audit_page(const nlohmann::json &value) {
    require_exact_keys(value, std::array {"schema"sv, "version"sv, "index"sv,
      "scenes"sv, "boundary_revisions"sv}, "scene audit page");
    if (value.at("schema") != 1 || value.at("version") != "whole-clip-scene-audit-page-v1" ||
        !value.at("scenes").is_array() || value.at("scenes").empty() ||
        value.at("scenes").size() > scene_audit_page_max_scenes ||
        !value.at("boundary_revisions").is_array() ||
        value.at("boundary_revisions").size() > value.at("scenes").size()) {
      throw contract_error("scene audit page semantics or bounds are invalid");
    }
    scene_audit_page_t result;
    result.index = unsigned_integer<std::uint32_t>(value, "index", "scene audit page");
    if (result.index >= scene_audit_max_pages) throw contract_error("scene audit page index exceeds its bound");
    for (const auto &scene : value.at("scenes")) result.scenes.push_back(parse_scene_record(scene));
    for (const auto &boundary : value.at("boundary_revisions")) result.boundary_revisions.push_back(parse_boundary_audit(boundary));
    return result;
  }

  void validate_scene_audit_page(const scene_audit_page_t &page,
                                const scene_audit_page_descriptor_t &descriptor,
                                const bool complete_final_page) {
    if (page.index != descriptor.index || page.scenes.size() != descriptor.scene_count ||
        page.boundary_revisions.size() != descriptor.boundary_count || page.scenes.empty()) {
      throw contract_error("scene audit page disagrees with its descriptor");
    }
    std::uint64_t sequence = descriptor.start_sequence;
    std::size_t boundary_index = 0;
    for (std::size_t i = 0; i < page.scenes.size(); ++i) {
      const auto &scene = page.scenes[i];
      const bool final = complete_final_page && i + 1 == page.scenes.size();
      if (scene.scene_id != descriptor.first_scene_id + i ||
          scene.semantic_scene_id != scene.scene_id || scene.start_sequence != sequence ||
          scene.cache_bytes != 0 || scene.evidence.source_frame_count != scene.frame_count ||
          scene.cut_state_semantics != "causal-production-exact" ||
          scene.boundary.semantic_cut == final ||
          scene.boundary.decision != (final ? boundary_decision_e::end_of_stream : boundary_decision_e::confirmed)) {
        throw contract_error("scene audit page is not contiguous causal production evidence");
      }
      if (!final && (scene.boundary.final_sequence != scene.end_sequence_exclusive ||
          boundary_index >= page.boundary_revisions.size() ||
          boundary_audit_json(scene.boundary) != boundary_audit_json(page.boundary_revisions[boundary_index++]))) {
        throw contract_error("scene audit page boundary evidence disagrees with its scenes");
      }
      sequence = scene.end_sequence_exclusive;
    }
    if (sequence != descriptor.end_sequence_exclusive || boundary_index != page.boundary_revisions.size()) {
      throw contract_error("scene audit page coverage disagrees with its descriptor");
    }
  }

  nlohmann::json to_json(const scene_audit_contract_t &value) {
    json scenes = json::array();
    for (const auto &scene : value.scenes) {
      scenes.push_back(scene_record_json(scene));
    }
    json boundaries = json::array();
    for (const auto &boundary : value.boundary_revisions) {
      boundaries.push_back(boundary_audit_json(boundary));
    }
    json result {
      {"schema", 3},
      {"version", "whole-clip-scene-audit-v3"},
      {"status", value.status},
      {"claims", {{"ground_truth", false}, {"best_parameters", false}}},
      {"policy", {
        {"implementation", "native-causal-scene-tracker"},
        {"version", "causal-scene-epochs-v1"},
        {"lookahead", false},
        {"boundary_only", true},
        {"python_dependency", false},
      }},
      {"cache", {
        {"peak_bytes", value.peak_cache_bytes},
        {"analysis_source_raster_bytes", value.analysis_source_raster_bytes},
        {"peak_live_raster_bytes", value.peak_live_raster_bytes},
        {"peak_cache_plus_raster_bytes", value.peak_cache_plus_raster_bytes},
        {"hard_cap_bytes", value.hard_cap_bytes},
      }},
      {"timeline_contract", value.timeline_contract},
      {"scenes", std::move(scenes)},
      {"boundary_revisions", std::move(boundaries)},
    };
    if (value.paged) {
      result["schema"] = 4;
      result["version"] = "whole-clip-scene-audit-v4";
      result.erase("scenes");
      result.erase("boundary_revisions");
      result["storage"] = {{"format", "immutable-pages-v1"},
        {"scene_count", value.scene_count}, {"boundary_count", value.boundary_count},
        {"covered_end_sequence", value.covered_end_sequence}, {"total_page_bytes", value.total_page_bytes},
        {"page_max_scenes", scene_audit_page_max_scenes}, {"page_max_bytes", scene_audit_page_max_bytes},
        {"total_max_bytes", scene_audit_storage_max_bytes}, {"max_pages", scene_audit_max_pages}};
      result["pages"] = json::array();
      for (const auto &page : value.pages) {
        result["pages"].push_back({{"index", page.index}, {"first_scene_id", page.first_scene_id},
          {"scene_count", page.scene_count}, {"boundary_count", page.boundary_count},
          {"start_sequence", page.start_sequence}, {"end_sequence_exclusive", page.end_sequence_exclusive},
          {"bytes", page.bytes}, {"sha256", page.sha256}});
      }
    }
    return result;
  }

  scene_audit_contract_t parse_scene_audit_contract(const nlohmann::json &value) {
    if (value.value("schema", 0) == 4) {
      require_exact_keys(value, std::array {"schema"sv, "version"sv, "status"sv, "claims"sv,
        "policy"sv, "cache"sv, "timeline_contract"sv, "storage"sv, "pages"sv}, "scene audit manifest");
      if (value.at("version") != "whole-clip-scene-audit-v4" || !value.at("pages").is_array() ||
          value.at("pages").size() > scene_audit_max_pages) throw contract_error("scene audit manifest version or index bound is invalid");
      auto base = value;
      base["schema"] = 3;
      base["version"] = "whole-clip-scene-audit-v3";
      base.erase("storage"); base.erase("pages");
      base["scenes"] = json::array(); base["boundary_revisions"] = json::array();
      auto result = parse_scene_audit_contract(base);
      result.paged = true;
      const auto &storage = value.at("storage");
      require_exact_keys(storage, std::array {"format"sv, "scene_count"sv, "boundary_count"sv,
        "covered_end_sequence"sv, "total_page_bytes"sv, "page_max_scenes"sv, "page_max_bytes"sv,
        "total_max_bytes"sv, "max_pages"sv}, "scene audit storage");
      if (storage.at("format") != "immutable-pages-v1" || storage.at("page_max_scenes") != scene_audit_page_max_scenes ||
          storage.at("page_max_bytes") != scene_audit_page_max_bytes || storage.at("total_max_bytes") != scene_audit_storage_max_bytes ||
          storage.at("max_pages") != scene_audit_max_pages) throw contract_error("scene audit storage limits are invalid");
      result.scene_count = unsigned_integer<std::uint64_t>(storage, "scene_count", "scene audit storage");
      result.boundary_count = unsigned_integer<std::uint64_t>(storage, "boundary_count", "scene audit storage");
      result.covered_end_sequence = unsigned_integer<std::uint64_t>(storage, "covered_end_sequence", "scene audit storage");
      result.total_page_bytes = unsigned_integer<std::uint64_t>(storage, "total_page_bytes", "scene audit storage");
      std::uint64_t scenes = 0, boundaries = 0, bytes = 0, end = 1;
      for (const auto &entry : value.at("pages")) {
        require_exact_keys(entry, std::array {"index"sv, "first_scene_id"sv, "scene_count"sv, "boundary_count"sv,
          "start_sequence"sv, "end_sequence_exclusive"sv, "bytes"sv, "sha256"sv}, "scene audit page descriptor");
        scene_audit_page_descriptor_t page;
        page.index = unsigned_integer<std::uint32_t>(entry, "index", "page descriptor");
        page.first_scene_id = unsigned_integer<std::uint64_t>(entry, "first_scene_id", "page descriptor");
        page.scene_count = unsigned_integer<std::uint64_t>(entry, "scene_count", "page descriptor");
        page.boundary_count = unsigned_integer<std::uint64_t>(entry, "boundary_count", "page descriptor");
        page.start_sequence = unsigned_integer<std::uint64_t>(entry, "start_sequence", "page descriptor");
        page.end_sequence_exclusive = unsigned_integer<std::uint64_t>(entry, "end_sequence_exclusive", "page descriptor");
        page.bytes = unsigned_integer<std::uint64_t>(entry, "bytes", "page descriptor");
        page.sha256 = required_string(entry, "sha256", "page descriptor");
        const bool final = result.status == "complete" && page.index + 1 == value.at("pages").size();
        if (page.index != result.pages.size() || page.first_scene_id != scenes + 1 ||
            page.scene_count == 0 || page.scene_count > scene_audit_page_max_scenes ||
            page.boundary_count != page.scene_count - (final ? 1 : 0) ||
            page.start_sequence != end || page.end_sequence_exclusive <= end ||
            page.bytes == 0 || page.bytes > scene_audit_page_max_bytes || !valid_sha256_hex(page.sha256) ||
            page.bytes > scene_audit_storage_max_bytes - bytes) throw contract_error("scene audit page index, coverage, or storage bound is invalid");
        scenes += page.scene_count; boundaries += page.boundary_count; bytes += page.bytes;
        end = page.end_sequence_exclusive;
        result.pages.push_back(std::move(page));
      }
      if (scenes != result.scene_count || boundaries != result.boundary_count || bytes != result.total_page_bytes ||
          end != result.covered_end_sequence || (result.status == "complete" && scenes == 0)) {
        throw contract_error("scene audit manifest totals disagree with its pages");
      }
      return result;
    }
    require_exact_keys(
      value,
      std::array {"schema"sv, "version"sv, "status"sv, "claims"sv, "policy"sv,
                  "cache"sv, "timeline_contract"sv, "scenes"sv,
                  "boundary_revisions"sv},
      "scene audit"
    );
    require_exact_keys(value.at("claims"), std::array {"ground_truth"sv, "best_parameters"sv}, "scene audit claims");
    require_exact_keys(
      value.at("policy"),
      std::array {"implementation"sv, "version"sv, "lookahead"sv,
                  "boundary_only"sv, "python_dependency"sv},
      "scene audit policy"
    );
    require_exact_keys(
      value.at("cache"),
      std::array {"peak_bytes"sv, "analysis_source_raster_bytes"sv,
                  "peak_live_raster_bytes"sv, "peak_cache_plus_raster_bytes"sv,
                  "hard_cap_bytes"sv},
      "scene audit cache"
    );
    if (value.at("schema") != 3 || value.at("version") != "whole-clip-scene-audit-v3" ||
        value.at("claims") != json {{"ground_truth", false}, {"best_parameters", false}} ||
        value.at("policy") != json {
          {"implementation", "native-causal-scene-tracker"},
          {"version", "causal-scene-epochs-v1"}, {"lookahead", false},
          {"boundary_only", true}, {"python_dependency", false}} ||
        !value.at("timeline_contract").is_object() || !value.at("scenes").is_array() ||
        !value.at("boundary_revisions").is_array()) {
      throw contract_error("scene audit has unknown policy or document semantics");
    }
    scene_audit_contract_t result;
    result.status = required_string(value, "status", "scene audit");
    if (result.status != "running" && result.status != "complete") {
      throw contract_error("scene audit status is invalid");
    }
    const auto &cache = value.at("cache");
    result.peak_cache_bytes = unsigned_integer<std::uint64_t>(cache, "peak_bytes", "scene audit cache");
    result.analysis_source_raster_bytes = unsigned_integer<std::uint64_t>(cache, "analysis_source_raster_bytes", "scene audit cache");
    result.peak_live_raster_bytes = unsigned_integer<std::uint64_t>(cache, "peak_live_raster_bytes", "scene audit cache");
    result.peak_cache_plus_raster_bytes = unsigned_integer<std::uint64_t>(cache, "peak_cache_plus_raster_bytes", "scene audit cache");
    result.hard_cap_bytes = unsigned_integer<std::uint64_t>(cache, "hard_cap_bytes", "scene audit cache");
    if (result.hard_cap_bytes == 0 || result.peak_cache_bytes > result.hard_cap_bytes ||
        result.peak_cache_plus_raster_bytes > result.hard_cap_bytes) {
      throw contract_error("scene audit cache attestation exceeds its hard cap");
    }
    result.timeline_contract = value.at("timeline_contract");
    if (value.at("scenes").size() > max_serialized_scene_count ||
        value.at("boundary_revisions").size() > max_serialized_scene_count) {
      throw contract_error("scene audit exceeds its serialized scene bound");
    }
    for (const auto &scene : value.at("scenes")) {
      result.scenes.push_back(parse_scene_record(scene));
    }
    for (const auto &boundary : value.at("boundary_revisions")) {
      result.boundary_revisions.push_back(parse_boundary_audit(boundary));
    }
    return result;
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
      if (*typed_runtime.frame_transport != "bounded-raw-memory-v1") {
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
    json decisions = json::array();
    for (const auto &decision : value.progress.scene_decisions) {
      decisions.push_back(decision);
    }
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
      {"cache_budget_policy", value.cache_budget_policy},
      {"progress", {
        {"phase", value.progress.phase},
        {"processed_frames", value.progress.processed_frames},
        {"total_frames", value.progress.total_frames ?
                           json(*value.progress.total_frames) : json(nullptr)},
        {"source_time_seconds", value.progress.source_time_seconds ?
                                  json(*value.progress.source_time_seconds) : json(nullptr)},
        {"source_duration_seconds", value.progress.source_duration_seconds ?
                                      json(*value.progress.source_duration_seconds) : json(nullptr)},
        {"scene_count", value.progress.scene_count ?
                          json(*value.progress.scene_count) : json(nullptr)},
        {"current_scene", value.progress.current_scene ?
                            *value.progress.current_scene : json(nullptr)},
      }},
      {"scene_decisions", std::move(decisions)},
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
      "scene_decisions"sv, "created_at_unix_ms"sv, "started_at_unix_ms"sv,
      "ended_at_unix_ms"sv, "error"sv, "worker_result"sv,
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
      // These are the only explicitly supported schema-1 durable variants. They predate the
      // current beside-input publication and bounded in-progress scene summary fields.
      if (!normalized.contains("output_location")) {
        normalized["output_location"] = nullptr;
      }
      if (!normalized.contains("scene_decisions")) {
        normalized["scene_decisions"] = json::array();
      }
      if (normalized.contains("progress") && normalized.at("progress").is_object() &&
          !normalized.at("progress").contains("current_scene")) {
        normalized["progress"]["current_scene"] = nullptr;
      }
    }
    require_exact_keys(normalized, base_keys, "job snapshot");
    require_exact_keys(
      normalized.at("progress"),
      std::array {"phase"sv, "processed_frames"sv, "total_frames"sv,
                  "source_time_seconds"sv, "source_duration_seconds"sv,
                  "scene_count"sv, "current_scene"sv},
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
    result.cache_budget_policy = required_string(normalized, "cache_budget_policy", "job snapshot");
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
    result.progress.scene_count = optional_value<std::uint64_t>(progress, "scene_count", [&]() {
      return unsigned_integer<std::uint64_t>(progress, "scene_count", "job progress");
    });
    if (!progress.at("current_scene").is_null()) {
      if (!progress.at("current_scene").is_object()) {
        throw contract_error("job current scene must be an object or null");
      }
      result.progress.current_scene = progress.at("current_scene");
    }
    if (!normalized.at("scene_decisions").is_array() ||
        normalized.at("scene_decisions").size() > 64) {
      throw contract_error("job scene decisions must be a bounded array");
    }
    for (const auto &decision : normalized.at("scene_decisions")) {
      if (!decision.is_object()) {
        throw contract_error("job scene decision must be an object");
      }
      result.progress.scene_decisions.push_back(decision);
    }
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
         *result.progress.source_duration_seconds <= 0.0) ||
        (result.progress.scene_count &&
         *result.progress.scene_count > max_paged_scene_count)) {
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
