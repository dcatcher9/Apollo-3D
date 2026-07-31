/**
 * @file tests/unit/test_offline_sbs_worker.cpp
 * @brief Pure media-contract/command/timeline tests for the native offline worker.
 */

#include "src/offline_sbs_worker.h"
#include "src/crypto.h"
#include "src/generated/sbs_adaptive_state_contract.h"
#include "src/generated/sbs_scene_controller_contract.h"
#include "src/offline_sbs_contract.h"
#include "src/sbs_scene_cache_contract.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
  namespace fs = std::filesystem;

  std::string sha256_hex(const std::string_view bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    const auto digest = crypto::hash(bytes);
    std::string result(digest.size() * 2, '\0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
      result[index * 2] = digits[digest[index] >> 4u];
      result[index * 2 + 1] = digits[digest[index] & 0x0fu];
    }
    return result;
  }

  bool contains_argument_pair(
    const std::vector<std::string> &arguments,
    const std::string_view first,
    const std::string_view second
  ) {
    for (std::size_t index = 1; index < arguments.size(); ++index) {
      if (arguments[index - 1] == first && arguments[index] == second) {
        return true;
      }
    }
    return false;
  }

  std::string read_source_file(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    return {
      std::istreambuf_iterator<char> {input},
      std::istreambuf_iterator<char> {}
    };
  }

  nlohmann::json worker_spec_json() {
    return {
      {"schema", 1},
      {"job_id", "11111111-2222-3333-4444-555555555555"},
      {"operation", "convert"},
      {"input_path", "C:/media/source.mkv"},
      {"job_directory", "C:/state/jobs/one"},
      {"result_directory", "C:/state/jobs/one/result"},
      {"progress_path", "C:/state/jobs/one/progress.json"},
      {"result_path", "C:/state/jobs/one/result.json"},
      {"staging_output", "C:/exports/video.part.mkv"},
      {"sunshine", {
                     {"executable", "C:/Program Files/Sunshine 3D/sunshine.exe"},
                     {"config", "C:/ProgramData/Sunshine 3D/sunshine.conf"},
                   }},
      {"ffmpeg", {
                   {"path", "C:/Program Files/Sunshine 3D/tools/ffmpeg.exe"},
                   {"version", "ffmpeg 7"},
                 }},
      {"ffprobe", {
                    {"path", "C:/Program Files/Sunshine 3D/tools/ffprobe.exe"},
                    {"version", "ffprobe 7"},
                  }},
      {"scene_cache", {
                        {"hard_cap_bytes", 4ull * 1024ull * 1024ull * 1024ull},
                        {"budget_policy", "fail"},
                      }},
      {"codec", "hevc_nvenc"},
      {"planner", {
                    {"implementation", "native-offline-scene-planner"},
                    {"scene_plan_contract", "scene-plan-v1"},
                  }},
      {"python_dependency", false},
    };
  }

  nlohmann::json native_replay_capabilities() {
    nlohmann::json analysis_flag_bits = nlohmann::json::object();
    for (const auto &flag : sbs_adaptive_state::analysis_flag_bits) {
      analysis_flag_bits[std::string {flag.name}] = flag.bit;
    }
    return {
      {"schema", 1},
      {"native_whole_clip", {
        {"follow_protocol_schema", 1},
        {"follow_global_first_sequence", true},
        {"adaptive_state_schema", sbs_adaptive_state::schema_version},
        {"adaptive_analysis_flag_bits", std::move(analysis_flag_bits)},
        {"scene_controller_trace", {
          {"trace_schema", 1u},
          {"controller_schema", sbs_scene_controller::schema_version},
          {"rule_revision", sbs_scene_controller::rule_revision},
          {"ordered_abi_hash", sbs_scene_controller::ordered_abi_hash},
          {"backends", {"off", "shadow_rules"}},
          {"active_roi_authority", false},
          {"source_time_override", "source-presentation-timeline-v2"},
          {"file", "scene_controller.jsonl"},
          {"transports", {"jsonl-v1", "atomic-latest-v1"}},
          {"atomic_header_file", "scene_controller_header.json"},
          {"atomic_frame_file", "scene_controller_frame.json"},
          {"global_out_word_count",
           sbs_scene_controller::global_out_word_count},
          {"rule_state_word_count",
           sbs_scene_controller::rule_state_word_count},
        }},
        {"scene_cache_contract_schema", sbs_scene_cache::contract_schema},
        {"scene_cache_packed_sbs_contract", true},
        {"scene_cache_depth", {
          {"dtype", "float32-le"},
          {"layout", "row-major"},
          {"dxgi_format", "R32_FLOAT"},
          {"dimensions", "per-frame-metadata"},
          {"bytes_per_frame", nullptr},
        }},
        {"scene_cache_frame_metadata", {
          {"schema", sbs_scene_cache::frame_metadata_schema},
          {"word_count", sbs_scene_cache::frame_metadata_word_count},
          {"roi_transform_word_offset",
           sbs_scene_cache::roi_transform_word_offset},
          {"roi_transform_word_count",
           sbs_scene_cache::roi_transform_word_count},
          {"roi_transform_contract_schema",
           models::frame_roi_transform_contract_version},
        }},
        {"scene_cache_state", {
          {"schema", sbs_scene_cache::cached_state_schema},
          {"subject_word_count",
           sbs_adaptive_state::render_prefix_word_count},
          {"depth_frame_state_word_count",
           sbs_scene_cache::depth_frame_state_word_count},
          {"word_count", sbs_scene_cache::cached_state_word_count},
          {"dtype", "uint32-le"},
        }},
        {"scene_plan", {
          {"schema", 1},
          {"version", "scene-plan-v1"},
          {"one_scene_per_replay", true},
          {"absolute_pop_strength", true},
          {"source_pixel_zero_anchor", true},
        }},
        {"render_cache_follow", true},
        {"render_skips_tensorrt", true},
        {"whole_clip_inference_attestation", {
          {"depth_inference_enabled", true},
          {"scheduled_depth_update_count", true},
          {"tensorrt_enqueue_count", true},
        }},
        {"atomic_sbs_publication", true},
      }},
    };
  }

  nlohmann::json sdr_probe() {
    return {
      {"streams", nlohmann::json::array({
                    {
                      {"codec_name", "h264"},
                      {"width", 1920},
                      {"height", 1080},
                      {"pix_fmt", "yuv420p"},
                      {"color_range", "tv"},
                      {"color_space", "bt709"},
                      {"color_transfer", "bt709"},
                      {"color_primaries", "bt709"},
                      {"time_base", "1/1000"},
                    },
                  })},
      {"frames", nlohmann::json::array({
                   {
                     {"pts", 100},
                     {"duration", 40},
                     {"width", 1920},
                     {"height", 1080},
                     {"pix_fmt", "yuv420p"},
                     {"color_range", "tv"},
                     {"color_space", "bt709"},
                     {"color_transfer", "bt709"},
                     {"color_primaries", "bt709"},
                   },
                   {
                     {"pts", 140},
                     {"duration", 55},
                     {"width", 1920},
                     {"height", 1080},
                   },
                   {
                     {"pts", 195},
                     {"duration", 45},
                     {"width", 1920},
                     {"height", 1080},
                   },
                 })},
      {"format", {
                   {"format_name", "matroska"},
                 }},
    };
  }

  nlohmann::json pq_probe() {
    auto value = sdr_probe();
    auto &stream = value["streams"][0];
    stream["codec_name"] = "hevc";
    stream["pix_fmt"] = "yuv420p10le";
    stream["color_space"] = "bt2020nc";
    stream["color_transfer"] = "smpte2084";
    stream["color_primaries"] = "bt2020";
    stream["side_data_list"] = nlohmann::json::array({
      {
        {"side_data_type", "Mastering display metadata"},
        {"red_x", "34000/50000"},
        {"max_luminance", "10000000/10000"},
      },
      {
        {"side_data_type", "Content light level metadata"},
        {"max_content", 1000},
        {"max_average", 400},
      },
    });
    auto &first = value["frames"][0];
    first["pix_fmt"] = "yuv420p10le";
    first["color_space"] = "bt2020nc";
    first["color_transfer"] = "smpte2084";
    first["color_primaries"] = "bt2020";
    return value;
  }

  nlohmann::json hlg_probe() {
    auto value = pq_probe();
    value["streams"][0]["color_transfer"] = "arib-std-b67";
    value["frames"][0]["color_transfer"] = "arib-std-b67";
    return value;
  }
}  // namespace

TEST(OfflineSbsWorker, SharesBoundedLinearSceneAuditContract) {
  EXPECT_EQ(offline_sbs::worker_result_max_bytes, 16ull * 1024ull * 1024ull);
  EXPECT_EQ(offline_sbs::scene_audit_max_bytes, 32ull * 1024ull * 1024ull);
  EXPECT_EQ(offline_sbs::max_serialized_scene_count, 1920u);

  std::size_t checkpoint_count = 0;
  std::size_t checkpoint_prefix_sum = 0;
  for (
    std::size_t scenes = 1;
    scenes <= offline_sbs::max_serialized_scene_count;
    ++scenes
  ) {
    if (offline_sbs::is_scene_audit_checkpoint(scenes)) {
      ++checkpoint_count;
      checkpoint_prefix_sum += scenes;
    }
  }
  EXPECT_EQ(checkpoint_count, 11u);
  EXPECT_LT(
    checkpoint_prefix_sum,
    offline_sbs::max_serialized_scene_count * 2
  );
  EXPECT_TRUE(offline_sbs::is_scene_audit_checkpoint(1));
  EXPECT_TRUE(offline_sbs::is_scene_audit_checkpoint(1024));
  EXPECT_FALSE(offline_sbs::is_scene_audit_checkpoint(0));
  EXPECT_FALSE(offline_sbs::is_scene_audit_checkpoint(1025));
}

TEST(OfflineSbsWorker, SelectsSceneControllerBackendForAnalysisAndReplay) {
  const auto source = read_source_file(
    fs::path(SUNSHINE_SOURCE_DIR) / "src" / "offline_sbs_worker.cpp"
  );
  ASSERT_FALSE(source.empty());

  const std::string analysis_selection =
    "\"--bounded-adaptive-state\",\n"
    "        \"--scene-controller\",\n"
    "        \"shadow_rules\",";
  const std::string replay_selection =
    "\"--bounded-adaptive-state\",\n"
    "        \"--scene-controller\",\n"
    "        \"off\",\n"
    "        \"--render-cache\",";
  EXPECT_NE(source.find(analysis_selection), std::string::npos);
  EXPECT_NE(source.find(replay_selection), std::string::npos);
  EXPECT_NE(
    source.find("\"--scene-controller-source-time\""),
    std::string::npos
  );
  EXPECT_NE(
    source.find("scene-controller-timeline.jsonl"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("write_scene_controller_source_time_contract("),
    std::string::npos
  );
  EXPECT_NE(
    source.find("{\"scene_controller_source_time\", {"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("{\"disk_reservation_bytes\","),
    std::string::npos
  );

  std::size_t selection_count = 0;
  for (
    auto offset = source.find("\"--scene-controller\"");
    offset != std::string::npos;
    offset = source.find("\"--scene-controller\"", offset + 1)
  ) {
    ++selection_count;
  }
  EXPECT_EQ(selection_count, 2u);
  EXPECT_NE(
    source.find(
      "scene_controller_trace.read_frame(analysis, timing.sequence)"
    ),
    std::string::npos
  ) << "The worker must consume every atomic controller snapshot before "
       "publishing the next source frame";
  EXPECT_NE(
    source.find("\"scene-controller-jsonl-gzip-v1\""),
    std::string::npos
  ) << "Whole-clip controller history must remain durable with bounded memory";

  const auto prepared = source.find("prepared_ = true;");
  const auto renamed = source.find("final_renamed_ = true;", prepared);
  const auto summary = source.find(
    "write_json_atomic(summary_path_, prepared_result_);",
    renamed
  );
  const auto publish = source.find(
    "scene_controller_trace.publish();",
    summary
  );
  const auto result_commit = source.find(
    "write_json_atomic_bounded(\n"
    "        spec.result_path,",
    publish
  );
  const auto committed = source.find(
    "scene_controller_trace.commit();",
    result_commit
  );
  ASSERT_NE(prepared, std::string::npos);
  ASSERT_NE(renamed, std::string::npos);
  ASSERT_NE(summary, std::string::npos);
  ASSERT_NE(publish, std::string::npos);
  ASSERT_NE(result_commit, std::string::npos);
  ASSERT_NE(committed, std::string::npos);
  EXPECT_LT(prepared, renamed);
  EXPECT_LT(renamed, summary);
  EXPECT_LT(summary, publish);
  EXPECT_LT(publish, result_commit);
  EXPECT_LT(result_commit, committed)
    << "The trace pair must remain rollback-owned until result.json commits";
  EXPECT_NE(
    source.find("fs::remove(final_path_, ignored);"),
    std::string::npos
  ) << "A failed summary publication must roll back its renamed gzip";
  EXPECT_NE(
    source.find("fs::remove(summary_path_, ignored);"),
    std::string::npos
  ) << "Any later failed job must also roll back its prepared trace summary";
}

TEST(OfflineSbsWorker, ValidatesBoundedSceneControllerEvidenceAndDisabledReplay) {
  const auto root =
    fs::temp_directory_path() /
    (
      "sunshine3d-scene-controller-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      )
    );
  fs::create_directories(root);
  const auto cleanup = [&]() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  };
  const auto write_json = [&](const fs::path &path, const nlohmann::json &value) {
    std::ofstream output(path, std::ios::binary);
    output << value.dump() << '\n';
  };

  nlohmann::json global_fields = nlohmann::json::array();
  for (const auto name : sbs_scene_controller::global_out_names) {
    global_fields.push_back({{"name", name}, {"type", "float32"}});
  }
  nlohmann::json rule_fields = nlohmann::json::array();
  nlohmann::json rule_state = nlohmann::json::array();
  for (const auto &field : sbs_scene_controller::rule_state_fields) {
    std::string encoding;
    switch (field.gpu_encoding) {
      case sbs_scene_controller::gpu_encoding_e::float_value:
        encoding = "float";
        break;
      case sbs_scene_controller::gpu_encoding_e::uint_bits:
        encoding = "uint_bits";
        break;
      case sbs_scene_controller::gpu_encoding_e::uint_valued_float:
        encoding = "uint_valued_float";
        break;
    }
    rule_fields.push_back({
      {"name", field.name},
      {"type", field.json_type},
      {"gpu_encoding", encoding},
    });
    rule_state.push_back(
      field.gpu_encoding ==
        sbs_scene_controller::gpu_encoding_e::uint_bits ?
        nlohmann::json(0u) :
        nlohmann::json(0.0)
    );
  }
  rule_state[sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::schema_version
  )] = static_cast<double>(sbs_scene_controller::schema_version);
  rule_state[sbs_scene_controller::index(
    sbs_scene_controller::rule_state_word_e::backend_generation
  )] = 7u;

  nlohmann::json contract {
    {"model", "test-model"},
    {"depth_reuse_interval", 2u},
    {"scene_controller", {
      {"enabled", true},
      {"backend", "shadow_rules"},
      {"active_roi_authority", false},
      {"transport", "atomic-latest-v1"},
      {"file", nullptr},
      {"header_file", "scene_controller_header.json"},
      {"frame_file", "scene_controller_frame.json"},
      {"retained_history", false},
      {"trace_schema", 1u},
      {"controller_schema", sbs_scene_controller::schema_version},
      {"rule_revision", sbs_scene_controller::rule_revision},
      {"ordered_abi_hash", sbs_scene_controller::ordered_abi_hash},
      {"frame_count", 4u},
    }},
  };
  const nlohmann::json header {
    {"record", "header"},
    {"trace_schema", 1u},
    {"source", "video_depth_estimator.scene_controller_snapshot"},
    {"capture", "every-source-frame-after-estimator-update"},
    {"backend", "shadow_rules"},
    {"controller_schema", sbs_scene_controller::schema_version},
    {"rule_revision", sbs_scene_controller::rule_revision},
    {"ordered_abi_hash", sbs_scene_controller::ordered_abi_hash},
    {"global_out_fields", std::move(global_fields)},
    {"rule_state_fields", std::move(rule_fields)},
    {"config", {
      {"model", "test-model"},
      {"depth_reuse_interval", 2u},
      {"active_roi_authority", false},
    }},
  };
  nlohmann::json frame {
    {"record", "frame"},
    {"frame_id", "0000000004"},
    {"source_index", 3u},
    {"depth_updated", true},
    {"snapshot_available", true},
    {"controller_frame_id", 3u},
    {"backend_generation", 7u},
    {"shadow", true},
    {"global_out", nlohmann::json::array()},
    {"rule_state", std::move(rule_state)},
  };
  for (
    std::size_t index = 0;
    index < sbs_scene_controller::global_out_word_count;
    ++index
  ) {
    frame["global_out"].push_back(0.0);
  }
  write_json(root / "scene_controller_header.json", header);
  write_json(root / "scene_controller_frame.json", frame);

  EXPECT_NO_THROW(
    offline_sbs::validate_scene_controller_transport_for_test(
      contract,
      root,
      "shadow_rules",
      4u,
      1u
    )
  );
  EXPECT_NO_THROW(
    offline_sbs::validate_scene_controller_frame_for_test(
      frame,
      3u,
      4u,
      4u,
      2u
    )
  );
  auto held_nonterminal = frame;
  held_nonterminal["depth_updated"] = false;
  held_nonterminal["controller_frame_id"] = 2u;
  EXPECT_NO_THROW(
    offline_sbs::validate_scene_controller_frame_for_test(
      held_nonterminal,
      3u,
      4u,
      5u,
      2u
    )
  );
  auto unavailable = frame;
  unavailable["snapshot_available"] = false;
  unavailable["controller_frame_id"] = nullptr;
  unavailable["backend_generation"] = nullptr;
  unavailable["global_out"] = nullptr;
  unavailable["rule_state"] = nullptr;
  EXPECT_THROW(
    offline_sbs::validate_scene_controller_frame_for_test(
      unavailable,
      3u,
      4u,
      4u,
      2u
    ),
    std::runtime_error
  );
  EXPECT_THROW(
    offline_sbs::validate_scene_controller_frame_for_test(
      frame,
      3u,
      4u,
      4u,
      2u,
      2u,
      8u
    ),
    std::runtime_error
  ) << "A held controller identity cannot silently change generation";
  contract["scene_controller"]["active_roi_authority"] = 0;
  EXPECT_THROW(
    offline_sbs::validate_scene_controller_transport_for_test(
      contract,
      root,
      "shadow_rules",
      4u,
      1u
    ),
    std::runtime_error
  );
  contract["scene_controller"]["active_roi_authority"] = false;
  contract["scene_controller"]["trace_schema"] = 1.0;
  EXPECT_THROW(
    offline_sbs::validate_scene_controller_transport_for_test(
      contract,
      root,
      "shadow_rules",
      4u,
      1u
    ),
    std::runtime_error
  );
  contract["scene_controller"]["trace_schema"] = 1u;

  {
    std::ofstream duplicate_header(
      root / "scene_controller_header.json",
      std::ios::binary
    );
    duplicate_header
      << "{\"trace_schema\":1,"
      << header.dump().substr(1)
      << '\n';
  }
  EXPECT_THROW(
    offline_sbs::validate_scene_controller_transport_for_test(
      contract,
      root,
      "shadow_rules",
      4u,
      1u
    ),
    std::runtime_error
  );
  write_json(root / "scene_controller_header.json", header);

  frame["source_index"] = 2u;
  write_json(root / "scene_controller_frame.json", frame);
  EXPECT_THROW(
    offline_sbs::validate_scene_controller_transport_for_test(
      contract,
      root,
      "shadow_rules",
      4u,
      1u
    ),
    std::runtime_error
  );

  std::error_code ignored;
  fs::remove(root / "scene_controller_header.json", ignored);
  fs::remove(root / "scene_controller_frame.json", ignored);
  contract["scene_controller"] = {
    {"enabled", false},
    {"backend", "off"},
    {"active_roi_authority", false},
    {"transport", nullptr},
    {"file", nullptr},
    {"header_file", nullptr},
    {"frame_file", nullptr},
    {"retained_history", false},
    {"trace_schema", nullptr},
    {"controller_schema", sbs_scene_controller::schema_version},
    {"rule_revision", sbs_scene_controller::rule_revision},
    {"ordered_abi_hash", sbs_scene_controller::ordered_abi_hash},
    {"frame_count", 0u},
  };
  EXPECT_NO_THROW(
    offline_sbs::validate_scene_controller_transport_for_test(
      contract,
      root,
      "off",
      4u,
      1u
    )
  );
  cleanup();
}

TEST(OfflineSbsWorker, RejectsUnknownAdaptiveTraceFlagMeanings) {
  EXPECT_TRUE(offline_sbs::adaptive_trace_flags_valid_for_test(0.0f, 0u));
  EXPECT_TRUE(offline_sbs::adaptive_trace_flags_valid_for_test(127.0f, 255u));
  EXPECT_FALSE(offline_sbs::adaptive_trace_flags_valid_for_test(128.0f, 0u));
  EXPECT_FALSE(offline_sbs::adaptive_trace_flags_valid_for_test(0.0f, 256u));
  EXPECT_FALSE(offline_sbs::adaptive_trace_flags_valid_for_test(1.5f, 0u));
  EXPECT_FALSE(offline_sbs::adaptive_trace_flags_valid_for_test(
    std::numeric_limits<float>::quiet_NaN(),
    0u
  ));
}

TEST(OfflineSbsWorker, RejectsEveryNestedReplayCapabilityDrift) {
  const auto valid = native_replay_capabilities();
  ASSERT_TRUE(
    offline_sbs::native_replay_capabilities_valid_for_test(valid)
  );
  ASSERT_TRUE(
    offline_sbs::native_replay_capabilities_json_valid_for_test(
      valid.dump()
    )
  );

  for (const auto &mutation : {
         std::vector<std::string> {
           "scene_cache_depth", "dxgi_format"
         },
         std::vector<std::string> {
           "scene_cache_frame_metadata", "roi_transform_word_offset"
         },
         std::vector<std::string> {
           "scene_cache_frame_metadata", "roi_transform_contract_schema"
         },
         std::vector<std::string> {
           "scene_cache_state", "word_count"
         },
         std::vector<std::string> {
           "scene_controller_trace", "ordered_abi_hash"
         },
         std::vector<std::string> {
           "scene_controller_trace", "source_time_override"
         },
         std::vector<std::string> {
           "scene_plan", "source_pixel_zero_anchor"
         },
         std::vector<std::string> {
           "whole_clip_inference_attestation", "tensorrt_enqueue_count"
         },
       }) {
    auto drifted = valid;
    auto &leaf = drifted["native_whole_clip"][mutation[0]][mutation[1]];
    if (leaf.is_boolean()) {
      leaf = !leaf.get<bool>();
    } else if (leaf.is_number_unsigned() || leaf.is_number_integer()) {
      leaf = leaf.get<std::int64_t>() + 1;
    } else {
      leaf = "drifted";
    }
    EXPECT_FALSE(
      offline_sbs::native_replay_capabilities_valid_for_test(drifted)
    ) << mutation[0] << '.' << mutation[1];
  }

  for (const auto &mutation : {
         std::vector<std::string> {"schema"},
         std::vector<std::string> {
           "native_whole_clip", "follow_protocol_schema"
         },
         std::vector<std::string> {
           "native_whole_clip", "scene_controller_trace", "trace_schema"
         },
       }) {
    auto wrong_type = valid;
    nlohmann::json *leaf = &wrong_type;
    for (const auto &component : mutation) {
      leaf = &leaf->at(component);
    }
    *leaf = 1.0;
    EXPECT_FALSE(
      offline_sbs::native_replay_capabilities_valid_for_test(wrong_type)
    ) << mutation.back() << " must remain a JSON integer";
  }

  auto numeric_boolean = valid;
  numeric_boolean["native_whole_clip"]["scene_controller_trace"]
                 ["active_roi_authority"] = 0;
  EXPECT_FALSE(
    offline_sbs::native_replay_capabilities_valid_for_test(
      numeric_boolean
    )
  );
  numeric_boolean = valid;
  numeric_boolean["native_whole_clip"]
                 ["whole_clip_inference_attestation"]
                 ["tensorrt_enqueue_count"] = 1;
  EXPECT_FALSE(
    offline_sbs::native_replay_capabilities_valid_for_test(
      numeric_boolean
    )
  );

  auto duplicate_root_key = valid.dump();
  duplicate_root_key.insert(1, R"("\u0073chema":1,)");
  EXPECT_FALSE(
    offline_sbs::native_replay_capabilities_json_valid_for_test(
      duplicate_root_key
    )
  ) << "Escaped and literal spellings of the same key must be duplicates";
}

TEST(OfflineSbsWorker, WritesBoundedDurableSceneControllerTimelineJsonl) {
  const auto media = offline_sbs::parse_ffprobe_contract(sdr_probe());
  const auto root =
    fs::temp_directory_path() /
    (
      "sunshine-source-time-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      )
    );
  ASSERT_TRUE(fs::create_directories(root));
  const auto artifact =
    offline_sbs::write_scene_controller_source_time_contract_for_test(
      root,
      media
    );
  const fs::path path =
    artifact.at("path").get<std::string>();
  ASSERT_TRUE(fs::is_regular_file(path));
  const auto bytes = read_source_file(path);
  EXPECT_EQ(artifact.at("file_bytes"), bytes.size());
  EXPECT_EQ(artifact.at("disk_reservation_bytes"), bytes.size());
  EXPECT_EQ(artifact.at("file_sha256"), sha256_hex(bytes));
  EXPECT_EQ(artifact.at("schema"), 2u);
  EXPECT_EQ(
    artifact.at("clock"),
    "source-presentation-timeline-v2"
  );
  EXPECT_EQ(artifact.at("frame_count"), 3u);
  EXPECT_DOUBLE_EQ(
    artifact.at("total_elapsed_seconds").get<double>(),
    0.095
  );

  std::vector<nlohmann::json> records;
  std::istringstream input(bytes);
  for (std::string line; std::getline(input, line);) {
    ASSERT_FALSE(line.empty());
    ASSERT_LE(line.size() + 1u, 256u);
    records.push_back(nlohmann::json::parse(line));
  }
  ASSERT_EQ(records.size(), 4u);
  const auto &header = records.front();
  EXPECT_EQ(
    std::set<std::string>({
      "record",
      "schema",
      "clock",
      "frame_count",
      "time_base",
    }),
    [&] {
      std::set<std::string> keys;
      for (const auto &[key, _value] : header.items()) {
        keys.insert(key);
      }
      return keys;
    }()
  );
  EXPECT_EQ(header.at("record"), "header");
  EXPECT_EQ(header.at("schema"), 2u);
  EXPECT_EQ(
    header.at("clock"),
    "source-presentation-timeline-v2"
  );
  EXPECT_EQ(header.at("frame_count"), 3u);
  EXPECT_EQ(header.at("time_base").at("num"), 1);
  EXPECT_EQ(header.at("time_base").at("den"), 1000);
  EXPECT_EQ(records[1].at("record"), "frame");
  EXPECT_EQ(records[1].at("source_index"), 0u);
  EXPECT_EQ(records[1].at("frame_id"), "0000000001");
  EXPECT_EQ(records[1].at("pts_ticks"), 100);
  EXPECT_EQ(records[1].at("duration_ticks"), 40);
  EXPECT_EQ(records[2].at("pts_ticks"), 140);
  EXPECT_EQ(records[2].at("duration_ticks"), 55);
  EXPECT_EQ(records[3].at("pts_ticks"), 195);
  EXPECT_EQ(records[3].at("duration_ticks"), 45);
  EXPECT_GE(
    offline_sbs::
      scene_controller_source_time_max_artifact_bytes_for_test(),
    (12ull * 60ull * 60ull * 90ull + 1ull) * 256ull
  );
  EXPECT_LT(
    offline_sbs::
      scene_controller_source_time_max_artifact_bytes_for_test(),
    2ull * 1024ull * 1024ull * 1024ull
  );
  std::error_code ec;
  fs::remove_all(root, ec);
  EXPECT_FALSE(ec);
}

TEST(OfflineSbsWorker, RequiresExactDisabledSourceTimeOnReplay) {
  const nlohmann::json disabled {
    {"enabled", false},
    {"schema", nullptr},
    {"clock", nullptr},
    {"file_sha256", nullptr},
    {"frame_count", 0u},
    {"total_elapsed_seconds", 0.0},
  };
  EXPECT_TRUE(
    offline_sbs::scene_controller_source_time_attestation_valid_for_test(
      disabled,
      disabled
    )
  );
  for (const auto &[key, replacement] : std::vector<
         std::pair<std::string, nlohmann::json>
       > {
         {"enabled", 0},
         {"schema", 2u},
         {"clock", "source-presentation-timeline-v2"},
         {"file_sha256", std::string(64u, '0')},
         {"frame_count", 1u},
         {"total_elapsed_seconds", 0},
       }) {
    auto changed = disabled;
    changed[key] = replacement;
    EXPECT_FALSE(
      offline_sbs::scene_controller_source_time_attestation_valid_for_test(
        changed,
        disabled
      )
    ) << key;
  }
  auto added = disabled;
  added["timeline_sha256"] = nullptr;
  EXPECT_FALSE(
    offline_sbs::scene_controller_source_time_attestation_valid_for_test(
      added,
      disabled
    )
  );
}

TEST(OfflineSbsWorker, ParsesNativeSpecAndNeverBuildsPythonCommands) {
  const auto spec = offline_sbs::parse_worker_spec(worker_spec_json());
  EXPECT_EQ(spec.operation, "convert");
  EXPECT_EQ(spec.codec, "hevc_nvenc");
  EXPECT_FALSE(spec.allow_administrative_split);

  const auto media = offline_sbs::parse_ffprobe_contract(sdr_probe());
  const auto probe = offline_sbs::build_ffprobe_command(
    spec,
    "C:/state/jobs/one/probe.json"
  );
  const auto decoder = offline_sbs::build_decoder_command(spec, media);
  const auto joined = [&] {
    std::string value;
    for (const auto &argument : probe) {
      value += argument + ' ';
    }
    for (const auto &argument : decoder) {
      value += argument + ' ';
    }
    return value;
  }();
  EXPECT_EQ(joined.find("python"), std::string::npos);
  EXPECT_NE(joined.find("-show_frames"), std::string::npos);
  EXPECT_NE(joined.find("frame_side_data="), std::string::npos);
  EXPECT_EQ(joined.find("frame_side_data "), std::string::npos);
  EXPECT_NE(joined.find("-noautorotate"), std::string::npos);
  EXPECT_NE(joined.find("rawvideo"), std::string::npos);
  EXPECT_EQ(probe.back(), "C:/state/jobs/one/probe.json");
  EXPECT_TRUE(contains_argument_pair(probe, "-protocol_whitelist", "file"));
  EXPECT_TRUE(contains_argument_pair(decoder, "-protocol_whitelist", "file"));
  EXPECT_FALSE(contains_argument_pair(
    probe,
    "-protocol_whitelist",
    "file,http,tcp"
  ));
  EXPECT_FALSE(contains_argument_pair(
    decoder,
    "-protocol_whitelist",
    "file,http,tcp"
  ));
}

TEST(OfflineSbsWorker, AuthenticatesExactSpecBytesBeforeParsing) {
  const auto path =
    fs::temp_directory_path() /
    (
      "sunshine3d-worker-spec-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      ) +
      ".json"
    );
  const auto cleanup = [&]() {
    std::error_code ignored;
    fs::remove(path, ignored);
  };
  const auto bytes = worker_spec_json().dump(2) + "\n";
  {
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  const auto digest = sha256_hex(bytes);
  const auto authenticated =
    offline_sbs::read_authenticated_worker_spec(path, digest);
  EXPECT_EQ(authenticated.job_id, "11111111-2222-3333-4444-555555555555");
  EXPECT_EQ(authenticated.authenticated_spec_sha256, digest);

  {
    std::ofstream output(path, std::ios::binary | std::ios::app);
    output << ' ';
  }
  EXPECT_THROW(
    offline_sbs::read_authenticated_worker_spec(path, digest),
    std::runtime_error
  );
  EXPECT_THROW(
    offline_sbs::read_authenticated_worker_spec(path, std::string(64, 'A')),
    std::runtime_error
  );
  cleanup();
}

TEST(OfflineSbsWorker, StreamsHighFrameCountProbeIntoCompactTimingAndRemovesRawJson) {
  constexpr std::size_t frame_count = 100000;
  const auto path =
    fs::temp_directory_path() /
    (
      "sunshine3d-frame-probe-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      ) +
      ".json"
    );
  const auto cleanup = [&]() {
    std::error_code ignored;
    fs::remove(path, ignored);
  };
  cleanup();
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output << "{\"frames\":[";
    for (std::size_t index = 0; index < frame_count; ++index) {
      if (index != 0) {
        output << ',';
      }
      output
        << "{\"pts\":" << index
        << ",\"duration\":1,\"width\":1920,\"height\":1080}";
    }
    output << "]}";
    ASSERT_TRUE(output);
  }

  auto stream = sdr_probe()["streams"][0];
  const auto result =
    offline_sbs::parse_and_remove_ffprobe_frames_for_test(
      stream,
      path
    );
  EXPECT_FALSE(fs::exists(path))
    << "Raw per-frame FFprobe JSON must not become a retained job artifact";
  ASSERT_EQ(result.media.frames.size(), frame_count);
  EXPECT_EQ(result.media.frames.front().sequence, 1u);
  EXPECT_EQ(result.media.frames.back().sequence, frame_count);
  EXPECT_EQ(result.media.frames.back().pts, frame_count - 1);
  EXPECT_EQ(result.media.frames.back().duration, 1);
  EXPECT_EQ(result.peak_retained_frame_descriptors, 1u)
    << "The incremental parser must discard each frame object before reading the next";
  EXPECT_GT(result.probe_bytes, 4ull * 1024ull * 1024ull);
  EXPECT_LT(result.compact_contract_bytes, 4096u)
    << "The durable normalized contract must not serialize one record per frame";
  EXPECT_LE(sizeof(offline_sbs::frame_timing_t), 24u);
  EXPECT_LE(
    result.media.frames.capacity(),
    result.media.frames.size() * 2
  ) << "Only the compact timing vector may scale with frame count";
  cleanup();
}

TEST(OfflineSbsWorker, BoundsRetainedTimingBeyondTwelveHoursAtNinetyFps) {
  constexpr std::uint64_t twelve_hours_at_90_fps =
    12ull * 60ull * 60ull * 90ull;
  const auto limit = offline_sbs::retained_timing_frame_limit_for_test();

  EXPECT_GT(limit, twelve_hours_at_90_fps);
  EXPECT_EQ(
    limit,
    (128ull * 1024ull * 1024ull) /
      sizeof(offline_sbs::frame_timing_t)
  );
  ASSERT_GT(limit, 0u);
  EXPECT_TRUE(offline_sbs::can_retain_timing_frame_for_test(limit - 1));
  EXPECT_FALSE(offline_sbs::can_retain_timing_frame_for_test(limit));
  EXPECT_FALSE(
    offline_sbs::can_retain_timing_frame_for_test(
      std::numeric_limits<std::uint64_t>::max()
    )
  );
}

TEST(OfflineSbsWorker, BoundsInventoryDocumentsAndRetainedPackets) {
  EXPECT_EQ(
    offline_sbs::stream_inventory_probe_byte_limit_for_test(),
    8ull * 1024ull * 1024ull
  );
  EXPECT_EQ(
    offline_sbs::packet_probe_byte_limit_for_test(),
    64ull * 1024ull * 1024ull
  );
  EXPECT_EQ(
    offline_sbs::retained_packet_payload_limit_for_test(),
    128ull * 1024ull * 1024ull
  );
  const auto packet_limit =
    offline_sbs::auxiliary_packet_limit_for_test();
  EXPECT_EQ(packet_limit, 2000000u);
  EXPECT_TRUE(
    offline_sbs::can_retain_auxiliary_packets_for_test(
      packet_limit - 1,
      1
    )
  );
  EXPECT_FALSE(
    offline_sbs::can_retain_auxiliary_packets_for_test(
      packet_limit,
      1
    )
  );
  EXPECT_FALSE(
    offline_sbs::can_retain_auxiliary_packets_for_test(
      std::numeric_limits<std::uint64_t>::max(),
      0
    )
  );

  const auto packet_probe_limit =
    offline_sbs::packet_probe_byte_limit_for_test();
  constexpr std::uintmax_t audio_probe_bytes = 20ull * 1024ull * 1024ull;
  constexpr std::uintmax_t subtitle_probe_bytes = 20ull * 1024ull * 1024ull;
  constexpr std::uintmax_t data_probe_bytes = 24ull * 1024ull * 1024ull;
  EXPECT_TRUE(offline_sbs::can_consume_packet_probe_bytes_for_test(
    0,
    audio_probe_bytes
  ));
  EXPECT_TRUE(offline_sbs::can_consume_packet_probe_bytes_for_test(
    audio_probe_bytes,
    subtitle_probe_bytes
  ));
  EXPECT_TRUE(offline_sbs::can_consume_packet_probe_bytes_for_test(
    audio_probe_bytes + subtitle_probe_bytes,
    data_probe_bytes
  ));
  EXPECT_EQ(
    audio_probe_bytes + subtitle_probe_bytes + data_probe_bytes,
    packet_probe_limit
  );
  EXPECT_FALSE(offline_sbs::can_consume_packet_probe_bytes_for_test(
    audio_probe_bytes + subtitle_probe_bytes,
    data_probe_bytes + 1
  ));
  EXPECT_FALSE(offline_sbs::can_consume_packet_probe_bytes_for_test(
    packet_probe_limit + 1,
    0
  ));
}

TEST(OfflineSbsWorker, BoundsChildLogsWhilePreservingPrefixAndTail) {
  const auto limit =
    offline_sbs::child_process_log_byte_limit_for_test();
  ASSERT_EQ(limit, 8ull * 1024ull * 1024ull);

  const std::string short_log = "first diagnostic\nlast diagnostic\n";
  EXPECT_EQ(
    offline_sbs::bound_child_process_log_for_test(short_log),
    short_log
  );

  std::string verbose_log(limit * 2 + 4096, 'm');
  const std::string prefix = "child startup diagnostic\n";
  const std::string suffix = "\nchild terminal diagnostic";
  verbose_log.replace(0, prefix.size(), prefix);
  verbose_log.replace(
    verbose_log.size() - suffix.size(),
    suffix.size(),
    suffix
  );
  const auto bounded =
    offline_sbs::bound_child_process_log_for_test(verbose_log);
  EXPECT_LE(bounded.size(), limit);
  EXPECT_TRUE(bounded.starts_with(prefix));
  EXPECT_TRUE(bounded.ends_with(suffix));
  EXPECT_NE(
    bounded.find("child-log bytes omitted; showing prefix and tail"),
    std::string::npos
  );
}

TEST(OfflineSbsWorker, ReusesOneBoundedReplayLogAcrossMaximumSceneCount) {
  constexpr std::uint64_t max_scene_count = 1920;
  const fs::path work = fs::path("C:/managed-job/native-work");
  std::vector<fs::path> paths;
  paths.reserve(max_scene_count);
  for (std::uint64_t scene_id = 1; scene_id <= max_scene_count; ++scene_id) {
    paths.push_back(
      offline_sbs::replay_scene_log_path_for_test(work, scene_id)
    );
  }
  std::ranges::sort(paths);
  const auto distinct_end = std::ranges::unique(paths).begin();
  EXPECT_EQ(std::distance(paths.begin(), distinct_end), 1)
    << "Serial scene replay must retain O(1), not O(scene_count), child logs";
  EXPECT_EQ(
    paths.front(),
    work / "logs" / "render-current-scene.log"
  );
}

TEST(OfflineSbsWorker, ClearsStaleReplayLogBeforeStartingTheNextScene) {
  const auto root =
    fs::temp_directory_path() /
    (
      "offline-sbs-replay-log-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      )
    );
  const auto work = root / "native-work";
  const auto stale_log =
    offline_sbs::replay_scene_log_path_for_test(work, 41);
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(stale_log.parent_path());
  {
    std::ofstream output(stale_log, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output.is_open());
    output << "previous scene evidence";
  }
  ASSERT_TRUE(fs::exists(stale_log));

  EXPECT_EQ(
    offline_sbs::prepare_replay_scene_log_for_test(work, 42),
    stale_log
  );
  EXPECT_FALSE(fs::exists(stale_log));

  fs::remove_all(root, ec);
}

TEST(OfflineSbsWorker, StreamsHighPacketCountWithoutRetainingJsonDom) {
  constexpr std::size_t packet_count = 100000;
  const auto path =
    fs::temp_directory_path() /
    (
      "sunshine3d-packet-probe-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      ) +
      ".json"
    );
  const auto cleanup = [&]() {
    std::error_code ignored;
    fs::remove(path, ignored);
  };
  cleanup();
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output << "{\"packets\":[";
    for (std::size_t index = 0; index < packet_count; ++index) {
      if (index != 0) {
        output << ',';
      }
      output
        << "{\"stream_index\":1,\"pts\":" << index
        << ",\"dts\":" << index << ",\"duration\":1}";
    }
    output << "]}";
    ASSERT_TRUE(output);
  }

  const auto result =
    offline_sbs::parse_and_remove_ffprobe_packets_for_test(
      path,
      packet_count
    );
  EXPECT_FALSE(fs::exists(path))
    << "Raw packet JSON must never become a retained job artifact";
  EXPECT_EQ(result.packet_count, packet_count);
  EXPECT_EQ(result.peak_retained_packet_descriptors, 1u)
    << "The SAX parser must discard each packet object before reading the next";
  EXPECT_GT(result.probe_bytes, 4ull * 1024ull * 1024ull);
  cleanup();
}

TEST(OfflineSbsWorker, RejectsPacketCountAtIncrementalBoundaryAndCleansProbe) {
  const auto path =
    fs::temp_directory_path() /
    (
      "sunshine3d-packet-bound-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      ) +
      ".json"
    );
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output
      << R"({"packets":[{"stream_index":1,"pts":0,"duration":1},)"
      << R"({"stream_index":1,"pts":1,"duration":1}]})";
  }
  EXPECT_THROW(
    offline_sbs::parse_and_remove_ffprobe_packets_for_test(path, 1),
    std::runtime_error
  );
  EXPECT_FALSE(fs::exists(path));
}

TEST(OfflineSbsWorker, RejectsMalformedIncrementalPacketAndCleansProbe) {
  const auto path =
    fs::temp_directory_path() /
    (
      "sunshine3d-bad-packet-probe-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      ) +
      ".json"
    );
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output << R"({"packets":[{"stream_index":1},42]})";
  }
  EXPECT_THROW(
    offline_sbs::parse_and_remove_ffprobe_packets_for_test(path, 2),
    std::runtime_error
  );
  EXPECT_FALSE(fs::exists(path));
}

TEST(OfflineSbsWorker, RequiresPacketsArrayAtTheTopLevel) {
  const auto path =
    fs::temp_directory_path() /
    (
      "sunshine3d-missing-packet-array-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      ) +
      ".json"
    );
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output << R"({})";
  }
  EXPECT_THROW(
    offline_sbs::parse_and_remove_ffprobe_packets_for_test(path, 1),
    std::runtime_error
  );
  EXPECT_FALSE(fs::exists(path));
}

TEST(OfflineSbsWorker, RemovesMalformedStreamingProbeOnValidationFailure) {
  const auto path =
    fs::temp_directory_path() /
    (
      "sunshine3d-bad-frame-probe-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      ) +
      ".json"
    );
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output << R"({"frames":[{"pts":0,"duration":1},42]})";
  }
  EXPECT_THROW(
    offline_sbs::parse_and_remove_ffprobe_frames_for_test(
      sdr_probe()["streams"][0],
      path
    ),
    std::runtime_error
  );
  EXPECT_FALSE(fs::exists(path));
}

TEST(OfflineSbsWorker, RequiresFramesArrayAtTheTopLevel) {
  const auto path =
    fs::temp_directory_path() /
    (
      "sunshine3d-nested-frame-probe-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      ) +
      ".json"
    );
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output << R"({"wrapper":{"frames":[{"pts":0,"duration":1}]}})";
  }
  EXPECT_THROW(
    offline_sbs::parse_and_remove_ffprobe_frames_for_test(
      sdr_probe()["streams"][0],
      path
    ),
    std::runtime_error
  );
  EXPECT_FALSE(fs::exists(path));
}

TEST(OfflineSbsWorker, RejectsOversizedFrameBeforeRetainingItsDom) {
  const auto path =
    fs::temp_directory_path() /
    (
      "sunshine3d-oversized-frame-probe-" +
      std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
      ) +
      ".json"
    );
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output << R"({"frames":[{"pts":0,"duration":1,"padding":")";
    output << std::string(300ull * 1024ull, 'x');
    output << R"("}]})";
  }
  EXPECT_THROW(
    offline_sbs::parse_and_remove_ffprobe_frames_for_test(
      sdr_probe()["streams"][0],
      path
    ),
    std::runtime_error
  );
  EXPECT_FALSE(fs::exists(path));
}

#ifdef _WIN32
TEST(OfflineSbsWorker, FrameBridgeUsesUnpredictableCapabilityTokens) {
  const auto first = offline_sbs::secure_frame_bridge_token_for_test();
  const auto second = offline_sbs::secure_frame_bridge_token_for_test();
  ASSERT_EQ(first.size(), 64u);
  ASSERT_EQ(second.size(), 64u);
  EXPECT_NE(first, second);
  EXPECT_TRUE(std::all_of(first.begin(), first.end(), [](const char value) {
    return (value >= '0' && value <= '9') ||
           (value >= 'a' && value <= 'f');
  }));
}

TEST(OfflineSbsWorker, RogueLoopbackDisconnectDoesNotAbortFrameBridge) {
  EXPECT_TRUE(
    offline_sbs::frame_bridge_survives_unauthenticated_disconnect_for_test()
  );
}
#endif

TEST(OfflineSbsWorker, BoundsVariableFrameRateToOneOutputTick) {
  const auto media = offline_sbs::parse_ffprobe_contract(sdr_probe());
  ASSERT_EQ(media.frames.size(), 3u);
  EXPECT_EQ(media.frames[0].duration, 40);
  EXPECT_EQ(media.frames[1].duration, 55);
  EXPECT_EQ(media.frames[2].duration, 45);
  EXPECT_TRUE(media.variable_frame_rate());
  EXPECT_DOUBLE_EQ(media.duration_seconds(), 0.140);

  auto equivalent = media;
  equivalent.time_base = {1, 10000};
  for (auto &frame : equivalent.frames) {
    frame.pts *= 10;
    frame.duration *= 10;
  }
  EXPECT_NO_THROW(
    offline_sbs::validate_timeline_equivalence(media, equivalent)
  );
  equivalent.frames[1].pts += 1;
  EXPECT_NO_THROW(
    offline_sbs::validate_timeline_equivalence(media, equivalent)
  );
  equivalent.frames[1].pts += 1;
  EXPECT_THROW(
    offline_sbs::validate_timeline_equivalence(media, equivalent),
    std::runtime_error
  );
}

TEST(OfflineSbsWorker, ComparesCommonVideoTimeBasesAsExactRationals) {
  auto source = offline_sbs::parse_ffprobe_contract(sdr_probe());
  source.time_base = {1, 24000};
  source.frames = {
    {1, 0, 1001},
    {2, 1001, 1001},
    {3, 2002, 1001},
  };
  auto equivalent = source;
  equivalent.time_base = {1, 240000};
  for (auto &frame : equivalent.frames) {
    frame.pts *= 10;
    frame.duration *= 10;
  }
  EXPECT_NO_THROW(
    offline_sbs::validate_timeline_equivalence(source, equivalent)
  );
  equivalent.frames.back().duration += 2;
  EXPECT_THROW(
    offline_sbs::validate_timeline_equivalence(source, equivalent),
    std::runtime_error
  );
}

TEST(OfflineSbsWorker, AcceptsStaticBt2020PqAndBuildsLinearScRgbDecode) {
  const auto spec = offline_sbs::parse_worker_spec(worker_spec_json());
  const auto media = offline_sbs::parse_ffprobe_contract(pq_probe());
  EXPECT_EQ(media.color, offline_sbs::media_color_e::hdr_pq);
  EXPECT_FALSE(media.mastering_display.is_null());
  EXPECT_FALSE(media.content_light_level.is_null());

  const auto command = offline_sbs::build_decoder_command(spec, media);
  std::string joined;
  for (const auto &argument : command) {
    joined += argument + ' ';
  }
  EXPECT_NE(joined.find("transferin=smpte2084"), std::string::npos);
  EXPECT_NE(joined.find("primaries=bt709"), std::string::npos);
  EXPECT_NE(joined.find("transfer=linear"), std::string::npos);
  EXPECT_NE(joined.find("npl=80"), std::string::npos);
  EXPECT_NE(joined.find("image2pipe"), std::string::npos);
}

TEST(OfflineSbsWorker, AcceptsStaticBt2020HlgAndBuildsLinearScRgbDecode) {
  const auto spec = offline_sbs::parse_worker_spec(worker_spec_json());
  const auto media = offline_sbs::parse_ffprobe_contract(hlg_probe());
  EXPECT_EQ(media.color, offline_sbs::media_color_e::hdr_hlg);
  EXPECT_FALSE(media.mastering_display.is_null());
  EXPECT_FALSE(media.content_light_level.is_null());

  const auto command = offline_sbs::build_decoder_command(spec, media);
  std::string joined;
  for (const auto &argument : command) {
    joined += argument + ' ';
  }
  EXPECT_NE(joined.find("transferin=arib-std-b67"), std::string::npos);
  EXPECT_NE(joined.find("primaries=bt709"), std::string::npos);
  EXPECT_NE(joined.find("transfer=linear"), std::string::npos);
  EXPECT_NE(joined.find("npl=80"), std::string::npos);
  EXPECT_NE(joined.find("image2pipe"), std::string::npos);
}

TEST(OfflineSbsWorker, RejectsRotationDynamicHdrAndAmbiguousTenBitSdr) {
  auto rotated = sdr_probe();
  rotated["streams"][0]["tags"] = {{"rotate", "90"}};
  EXPECT_THROW(
    offline_sbs::parse_ffprobe_contract(rotated),
    std::runtime_error
  );

  auto dynamic = pq_probe();
  dynamic["frames"][2]["side_data_list"] = nlohmann::json::array({
    {
      {"side_data_type", "HDR Dynamic Metadata SMPTE2094-40 (HDR10+)"},
    },
  });
  EXPECT_THROW(
    offline_sbs::parse_ffprobe_contract(dynamic),
    std::runtime_error
  );

  auto ambiguous = sdr_probe();
  ambiguous["streams"][0]["pix_fmt"] = "yuv420p10le";
  ambiguous["frames"][0]["pix_fmt"] = "yuv420p10le";
  EXPECT_THROW(
    offline_sbs::parse_ffprobe_contract(ambiguous),
    std::runtime_error
  );

  for (const std::string alpha_format : {"rgba", "yuva420p", "gbrap"}) {
    auto alpha = sdr_probe();
    alpha["streams"][0]["pix_fmt"] = alpha_format;
    alpha["frames"][0]["pix_fmt"] = alpha_format;
    EXPECT_THROW(
      offline_sbs::parse_ffprobe_contract(alpha),
      std::runtime_error
    ) << alpha_format;
  }

  auto dolby_vision = pq_probe();
  dolby_vision["streams"][0]["codec_tag_string"] = "dvh1";
  EXPECT_THROW(
    offline_sbs::parse_ffprobe_contract(dolby_vision),
    std::runtime_error
  );

  auto anamorphic = sdr_probe();
  anamorphic["streams"][0]["sample_aspect_ratio"] = "4:3";
  EXPECT_THROW(
    offline_sbs::parse_ffprobe_contract(anamorphic),
    std::runtime_error
  );

  auto interlaced = sdr_probe();
  interlaced["frames"][1]["interlaced_frame"] = 1;
  EXPECT_THROW(
    offline_sbs::parse_ffprobe_contract(interlaced),
    std::runtime_error
  );
}

TEST(OfflineSbsWorker, AuditsDroppedEncoderSeiButRejectsSemanticSideData) {
  auto encoder_sei = sdr_probe();
  encoder_sei["frames"][0]["side_data_list"] = nlohmann::json::array({
    {
      {"side_data_type", "H.264 User Data Unregistered SEI message"},
    },
  });
  const auto media =
    offline_sbs::parse_ffprobe_contract(encoder_sei);
  ASSERT_EQ(media.dropped_nonsemantic_side_data_types.size(), 1u);
  EXPECT_EQ(
    media.dropped_nonsemantic_side_data_types.front(),
    "H.264 User Data Unregistered SEI message"
  );

  auto captions = sdr_probe();
  captions["frames"][0]["side_data_list"] = nlohmann::json::array({
    {
      {"side_data_type", "A53 Closed Captions"},
    },
  });
  EXPECT_THROW(
    offline_sbs::parse_ffprobe_contract(captions),
    std::runtime_error
  );
}

TEST(OfflineSbsWorker, RejectsTimelineGapsAndMissingFinalDuration) {
  auto gap = sdr_probe();
  gap["frames"][0]["duration"] = 39;
  EXPECT_THROW(
    offline_sbs::parse_ffprobe_contract(gap),
    std::runtime_error
  );

  auto missing = sdr_probe();
  missing["frames"][2].erase("duration");
  EXPECT_THROW(
    offline_sbs::parse_ffprobe_contract(missing),
    std::runtime_error
  );
  missing["streams"][0]["tags"] = {
    {"DURATION", "00:00:00.140000000"},
  };
  const auto encoded = offline_sbs::parse_ffprobe_contract(missing, false);
  ASSERT_EQ(encoded.frames.size(), 3u);
  EXPECT_EQ(encoded.frames.back().duration, 45);
}

TEST(OfflineSbsWorker, ValidatesReportedOutputFrameDurations) {
  const auto source = offline_sbs::parse_ffprobe_contract(sdr_probe());
  auto output_probe = sdr_probe();
  output_probe["frames"][1]["duration"] = 40;

  const auto output =
    offline_sbs::parse_ffprobe_contract(output_probe, false);
  ASSERT_EQ(output.frames.size(), source.frames.size());
  EXPECT_EQ(output.frames[1].duration, 40);
  EXPECT_THROW(
    offline_sbs::validate_timeline_equivalence(source, output),
    std::runtime_error
  );
}

TEST(OfflineSbsWorker, RejectsPresentationSpansLargerThanInt64) {
  auto extreme = sdr_probe();
  constexpr auto first = std::numeric_limits<std::int64_t>::min() + 100;
  constexpr std::int64_t middle = 0;
  constexpr auto last = std::numeric_limits<std::int64_t>::max() - 100;
  extreme["frames"][0]["pts"] = first;
  extreme["frames"][0]["duration"] = -first;
  extreme["frames"][1]["pts"] = middle;
  extreme["frames"][1]["duration"] = last;
  extreme["frames"][2]["pts"] = last;
  extreme["frames"][2]["duration"] = 45;

  EXPECT_THROW(
    offline_sbs::parse_ffprobe_contract(extreme),
    std::runtime_error
  );
}

TEST(OfflineSbsWorker, ReservesExactAnalysisRasterAndNextCachePair) {
  EXPECT_EQ(
    offline_sbs::analysis_open_cache_limit(1000, 200, 100),
    700
  );
  EXPECT_THROW(
    offline_sbs::analysis_open_cache_limit(399, 200, 100),
    std::runtime_error
  );
  EXPECT_THROW(
    offline_sbs::analysis_open_cache_limit(1000, 0, 100),
    std::runtime_error
  );
}

TEST(OfflineSbsWorker, ReservesLargestDynamicSceneCacheTriplet) {
  const auto bounded =
    offline_sbs::analysis_max_cache_triplet_bytes(1920, 1080);
  const auto larger_source =
    offline_sbs::analysis_max_cache_triplet_bytes(3840, 2160);
  const auto smaller_source =
    offline_sbs::analysis_max_cache_triplet_bytes(640, 360);

  EXPECT_EQ(bounded, larger_source);
  EXPECT_LT(smaller_source, bounded);
  EXPECT_GT(bounded, 770ull * 434ull * sizeof(float));
  EXPECT_THROW(
    offline_sbs::analysis_max_cache_triplet_bytes(13, 1080),
    std::runtime_error
  );

  constexpr std::uint64_t live_raster_reservation_bytes =
    8ull * 1024ull * 1024ull;
  const auto hard_cap = live_raster_reservation_bytes + bounded * 3u;
  const auto open_limit = offline_sbs::analysis_open_cache_limit(
    hard_cap,
    live_raster_reservation_bytes,
    bounded
  );
  EXPECT_EQ(open_limit, bounded * 2u);
}

TEST(OfflineSbsWorker, ReservesReplaySbsBeforeLaunchingTheHarness) {
  const auto sdr = offline_sbs::replay_sbs_raster_reservation_bytes(
    offline_sbs::media_color_e::sdr,
    3840u,
    1080u
  );
  const auto hdr = offline_sbs::replay_sbs_raster_reservation_bytes(
    offline_sbs::media_color_e::hdr_pq,
    3840u,
    1080u
  );
  EXPECT_GT(sdr, 3840ull * 1080ull * 4ull);
  EXPECT_EQ(hdr, 3840ull * 1080ull * 12ull + 64ull);

  constexpr std::uint64_t source = 16ull * 1024ull * 1024ull;
  constexpr std::uint64_t triplet = 4ull * 1024ull * 1024ull;
  const auto hard_cap = source + sdr + triplet * 3u;
  EXPECT_EQ(
    offline_sbs::analysis_open_cache_limit(
      hard_cap,
      source + sdr,
      triplet
    ),
    triplet * 2u
  );
  EXPECT_THROW(
    offline_sbs::analysis_open_cache_limit(
      source + sdr + triplet,
      source + sdr,
      triplet
    ),
    std::runtime_error
  );
  EXPECT_THROW(
    offline_sbs::replay_sbs_raster_reservation_bytes(
      offline_sbs::media_color_e::sdr,
      0u,
      1080u
    ),
    std::runtime_error
  );
}

TEST(OfflineSbsWorker, ReservesFirstAnalysisArtifactsBeforeDecoderLaunch) {
  const auto sdr = offline_sbs::analysis_source_raster_reservation_bytes(
    offline_sbs::media_color_e::sdr,
    1920u,
    1080u
  );
  const auto hdr = offline_sbs::analysis_source_raster_reservation_bytes(
    offline_sbs::media_color_e::hdr_pq,
    1920u,
    1080u
  );
  EXPECT_EQ(sdr, 1920ull * 1080ull * 4ull + 54ull);
  EXPECT_EQ(
    hdr,
    1920ull * 1080ull * 12ull + 16ull + 128ull + 64ull
  );
  EXPECT_THROW(
    offline_sbs::analysis_source_raster_reservation_bytes(
      offline_sbs::media_color_e::sdr,
      0u,
      1080u
    ),
    std::runtime_error
  );
  EXPECT_THROW(
    offline_sbs::analysis_source_raster_reservation_bytes(
      offline_sbs::media_color_e::sdr,
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<std::uint32_t>::max()
    ),
    std::runtime_error
  );
}

TEST(OfflineSbsWorker, IdentifiesPreservePreviousDepthFrameState) {
  EXPECT_FALSE(
    offline_sbs::depth_frame_requires_previous_for_test(0.0f, 0.0f)
  );
  EXPECT_FALSE(
    offline_sbs::depth_frame_requires_previous_for_test(1.0f, 1.0f)
  );
  EXPECT_TRUE(
    offline_sbs::depth_frame_requires_previous_for_test(1.0f, 0.0f)
  );
}

TEST(OfflineSbsWorker, RejectsInexactAvexprRunIntermediates) {
  auto media = offline_sbs::parse_ffprobe_contract(sdr_probe());
  constexpr std::int64_t step = 4503599627370496ll;
  media.frames = {
    {1, -9007199254740990ll, step},
    {2, -4503599627370494ll, step},
    {3, 2, step},
  };
  EXPECT_THROW(
    offline_sbs::validate_avexpr_timeline_exactness(media),
    std::runtime_error
  );

  media.frames.pop_back();
  EXPECT_NO_THROW(
    offline_sbs::validate_avexpr_timeline_exactness(media)
  );
}

TEST(OfflineSbsWorker, SelectsLowestCompatibleDefinedAv1Level) {
  auto media = offline_sbs::parse_ffprobe_contract(sdr_probe());

  // Packed 3840x1080 at the fixture's fastest 25 FPS interval needs level 5.0.
  EXPECT_EQ(offline_sbs::select_av1_level(media, 3840, 1080), "5.0");

  // Small clips retain a low decoder requirement instead of being over-declared 6.x.
  media.time_base = {1, 8};
  media.frames = {{1, 0, 1}};
  EXPECT_EQ(offline_sbs::select_av1_level(media, 640, 180), "2.0");

  // NVENC codes 180 visible lines in a 192-line AV1 superblock raster. At
  // 36 FPS that exactly fits level 2.0; one additional frame per second
  // requires 2.1 and is rejected by NVENC if declared as 2.0.
  media.time_base = {1, 36};
  EXPECT_EQ(offline_sbs::select_av1_level(media, 640, 180), "2.0");
  media.time_base = {1, 37};
  EXPECT_EQ(offline_sbs::select_av1_level(media, 640, 180), "2.1");

  // A packed 4K source at 90 FPS exceeds 6.0 display rate but fits defined 6.1.
  media.time_base = {1, 90000};
  media.frames = {{1, 0, 1000}, {2, 1000, 1000}};
  EXPECT_EQ(offline_sbs::select_av1_level(media, 7680, 2160), "6.1");
}

TEST(OfflineSbsWorker, WiresDefinedAv1LevelOnlyIntoAv1Encoder) {
  auto av1_json = worker_spec_json();
  av1_json["codec"] = "av1_nvenc";
  const auto av1 = offline_sbs::parse_worker_spec(av1_json);
  const auto media = offline_sbs::parse_ffprobe_contract(hlg_probe());
  const auto av1_arguments =
    offline_sbs::build_codec_arguments(av1, media, true, 3840, 1080);

  EXPECT_TRUE(contains_argument_pair(av1_arguments, "-c:v", "av1_nvenc"));
  EXPECT_TRUE(contains_argument_pair(av1_arguments, "-level", "5.0"));
  EXPECT_TRUE(contains_argument_pair(av1_arguments, "-extra_sei", "1"));

  const auto hevc = offline_sbs::parse_worker_spec(worker_spec_json());
  const auto hevc_arguments =
    offline_sbs::build_codec_arguments(hevc, media, true, 3840, 1080);
  EXPECT_TRUE(contains_argument_pair(hevc_arguments, "-c:v", "hevc_nvenc"));
  EXPECT_EQ(
    std::find(hevc_arguments.begin(), hevc_arguments.end(), "-level"),
    hevc_arguments.end()
  );
}

TEST(OfflineSbsWorker, RejectsAv1OutputBeyondDefinedLevelSixThree) {
  auto media = offline_sbs::parse_ffprobe_contract(sdr_probe());
  media.time_base = {1, 1000};
  media.frames = {{1, 0, 1}};

  EXPECT_THROW(
    offline_sbs::select_av1_level(media, 16384, 8704),
    std::runtime_error
  );

  media.frames.front().duration = 0;
  EXPECT_THROW(
    offline_sbs::select_av1_level(media, 640, 180),
    std::runtime_error
  );
}

TEST(OfflineSbsWorker, RejectsWindowsCommandLinesBeforeCreateProcess) {
  EXPECT_NO_THROW(
    offline_sbs::validate_windows_command_line_capacity_for_test({
      "ffmpeg.exe",
      "-metadata",
      "title=short",
      "output.mkv",
    })
  );
#ifdef _WIN32
  try {
    offline_sbs::validate_windows_command_line_capacity_for_test({
      "ffmpeg.exe",
      "-metadata",
      "comment=" + std::string(40000, 'x'),
      "output.mkv",
    });
    FAIL() << "Overlong CreateProcess command was accepted";
  } catch (const std::runtime_error &exception) {
    EXPECT_STREQ(
      exception.what(),
      "test command exceeds the Windows command-line limit"
    );
  }
#endif
}
