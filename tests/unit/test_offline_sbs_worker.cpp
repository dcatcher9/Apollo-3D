/**
 * @file tests/unit/test_offline_sbs_worker.cpp
 * @brief Pure media-contract/command/timeline tests for the native offline worker.
 */

#include "src/offline_sbs_worker.h"
#include "src/crypto.h"
#include "src/offline_sbs_contract.h"
#include "src/host_sbs_v2_geometry.h"
#include "src/host_sbs_observation_timeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
#endif

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
                    {"scene_plan_contract", "scene-plan-v2"},
                  }},
      {"python_dependency", false},
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

TEST(OfflineSbsWorker, ObservationTimelineRoundTripsAndRejectsRegression) {
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root = fs::temp_directory_path() /
                    ("sunshine-observation-timeline-" + std::to_string(nonce));
  fs::create_directories(root);
  const auto path = root / "timeline.sbsotl";
  const std::array<std::uint64_t, 4> expected {1u, 16667u, 33334u, 50001u};
  std::string error;
  ASSERT_TRUE(models::host_sbs_observation_timeline::write(path, expected, error)) << error;
  EXPECT_EQ(fs::file_size(path), 24u + expected.size() * sizeof(std::uint64_t));
  std::vector<std::uint64_t> actual;
  ASSERT_TRUE(models::host_sbs_observation_timeline::read(path, actual, error)) << error;
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), actual.begin(), actual.end()));

  const std::array<std::uint64_t, 2> regressed {2u, 1u};
  EXPECT_FALSE(models::host_sbs_observation_timeline::write(path, regressed, error));

  // A forged count must be rejected from the fixed header before the reader sizes its output
  // vector. A failed read also clears caller-owned data instead of leaving stale timestamps that
  // could be mistaken for the rejected sidecar.
  std::array<std::uint8_t, models::host_sbs_observation_timeline::header_bytes> header {};
  std::copy(
    models::host_sbs_observation_timeline::magic.begin(),
    models::host_sbs_observation_timeline::magic.end(),
    header.begin()
  );
  models::host_sbs_observation_timeline::write_u32_le(
    header.data() + 8u,
    models::host_sbs_observation_timeline::schema
  );
  models::host_sbs_observation_timeline::write_u32_le(
    header.data() + 12u,
    models::host_sbs_observation_timeline::header_bytes
  );
  models::host_sbs_observation_timeline::write_u64_le(
    header.data() + 16u,
    models::host_sbs_observation_timeline::max_timestamp_count + 1u
  );
  {
    std::ofstream forged(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(forged);
    forged.write(
      reinterpret_cast<const char *>(header.data()),
      static_cast<std::streamsize>(header.size())
    );
    ASSERT_TRUE(forged.good());
  }
  actual.assign({7u, 8u});
  EXPECT_FALSE(models::host_sbs_observation_timeline::read(path, actual, error));
  EXPECT_TRUE(actual.empty());
  EXPECT_NE(error.find("length is invalid"), std::string::npos) << error;
  std::error_code ec;
  fs::remove_all(root, ec);
}

#ifdef _WIN32
TEST(OfflineSbsWorker, TreatsClosedNativeStdoutPipeAsEofBeforeProcessSignals) {
  EXPECT_TRUE(offline_sbs::native_stdout_pipe_error_is_eof_for_test(
    ERROR_BROKEN_PIPE
  ));
  EXPECT_TRUE(offline_sbs::native_stdout_pipe_error_is_eof_for_test(
    ERROR_NO_DATA
  ));
  EXPECT_FALSE(offline_sbs::native_stdout_pipe_error_is_eof_for_test(
    ERROR_ACCESS_DENIED
  ));
}
#endif

TEST(OfflineSbsWorker, ReplayProgressUsesMonotonicAnalyzedFrontier) {
  std::ifstream stream(
    fs::path(SUNSHINE_SOURCE_DIR) / "src/offline_sbs_worker.cpp",
    std::ios::binary
  );
  ASSERT_TRUE(stream);
  const std::string source {
    std::istreambuf_iterator<char> {stream},
    std::istreambuf_iterator<char> {},
  };
  const auto render_begin = source.find("const auto render_scenes = [&](");
  const auto render_end = source.find("\n      try {", render_begin);
  ASSERT_NE(render_begin, std::string::npos);
  ASSERT_NE(render_end, std::string::npos);
  const auto render = source.substr(render_begin, render_end - render_begin);

  const auto frontier_parameter = render.find("analyzed_through_sequence");
  const auto replay_phase = render.find("\"replay\"");
  const auto replay_frontier = render.find(
    "analyzed_through_sequence",
    replay_phase
  );
  const auto post_replay_frontier = render.find(
    "analyzed_through_sequence",
    replay_frontier + 1
  );
  ASSERT_NE(frontier_parameter, std::string::npos);
  ASSERT_NE(replay_phase, std::string::npos);
  ASSERT_NE(replay_frontier, std::string::npos);
  ASSERT_NE(post_replay_frontier, std::string::npos);
  EXPECT_LT(frontier_parameter, replay_phase);
  EXPECT_LT(replay_phase, replay_frontier);
  EXPECT_LT(replay_frontier, post_replay_frontier);
  EXPECT_EQ(render.find("scene.start_sequence - 1"), std::string::npos);
  EXPECT_EQ(
    render.find("scene.end_sequence_exclusive - 1"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("render_scenes(finalized, timing.sequence)"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("render_scenes(finalized, media.frames.size())"),
    std::string::npos
  );
}

TEST(OfflineSbsWorker, RequiresSelectedInputFullFrameSourceScope) {
  nlohmann::json scope {
    {"frame_source", std::string {offline_sbs::whole_clip_frame_source}},
    {"analysis_region", std::string {offline_sbs::whole_clip_analysis_region}},
    {"active_window_dependency", false},
    {"window_region_roi", false},
  };
  EXPECT_TRUE(offline_sbs::offline_full_frame_source_scope_is_valid(scope));

  auto active_window = scope;
  active_window["active_window_dependency"] = true;
  EXPECT_FALSE(
    offline_sbs::offline_full_frame_source_scope_is_valid(active_window)
  );

  auto roi = scope;
  roi["window_region_roi"] = true;
  EXPECT_FALSE(offline_sbs::offline_full_frame_source_scope_is_valid(roi));

  auto cropped = scope;
  cropped["analysis_region"] = "window-region";
  EXPECT_FALSE(offline_sbs::offline_full_frame_source_scope_is_valid(cropped));

  auto desktop = scope;
  desktop["frame_source"] = "desktop-capture";
  EXPECT_FALSE(offline_sbs::offline_full_frame_source_scope_is_valid(desktop));

  auto extended = scope;
  extended["fallback"] = "active-window";
  EXPECT_FALSE(offline_sbs::offline_full_frame_source_scope_is_valid(extended));
}

TEST(OfflineSbsWorker, HeadlessHarnessHasNoForegroundWindowObserver) {
  std::ifstream stream(
    fs::path(SUNSHINE_SOURCE_DIR) / "src/sbs_bench_harness.cpp",
    std::ios::binary
  );
  ASSERT_TRUE(stream);
  const std::string source {
    std::istreambuf_iterator<char>(stream),
    std::istreambuf_iterator<char>()
  };

  EXPECT_EQ(source.find("GetForegroundWindow("), std::string::npos);
  EXPECT_EQ(source.find("capture_foreground_window_region"), std::string::npos);
  EXPECT_EQ(source.find("foreground_window_region.h"), std::string::npos);
  EXPECT_NE(
    source.find("Bench/replay stays full-frame, so ROI is disabled"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("models::make_host_sbs_v2_full_frame_geometry("),
    std::string::npos
  );
  EXPECT_EQ(source.find("float repro_params[8]"), std::string::npos);
  EXPECT_EQ(source.find("video_roi_active = 1.0f"), std::string::npos);
  EXPECT_NE(source.find("offline_full_frame_request"), std::string::npos);
  EXPECT_NE(
    source.find("depth_analysis_authority_e::full_source"),
    std::string::npos
  );
}

TEST(OfflineSbsWorker, SharedRendererGeometryAbiIsFullFrameOnly) {
  constexpr auto geometry =
    models::make_host_sbs_v2_full_frame_geometry(0.75f, 0.5f);
  static_assert(sizeof(geometry) == 48u);
  EXPECT_FLOAT_EQ(geometry.content_scale_x, 0.75f);
  EXPECT_FLOAT_EQ(geometry.content_scale_y, 0.5f);
  EXPECT_FLOAT_EQ(geometry.video_roi_active, 0.0f);
  EXPECT_FLOAT_EQ(geometry.video_roi_left, 0.0f);
  EXPECT_FLOAT_EQ(geometry.video_roi_top, 0.0f);
  EXPECT_FLOAT_EQ(geometry.video_roi_right, 1.0f);
  EXPECT_FLOAT_EQ(geometry.video_roi_bottom, 1.0f);
  EXPECT_EQ(geometry.tensor_content_left, 0u);
  EXPECT_EQ(geometry.tensor_content_top, 0u);
  EXPECT_EQ(geometry.tensor_content_right, 0u);
  EXPECT_EQ(geometry.tensor_content_bottom, 0u);
}

TEST(OfflineSbsWorker, DeviceConditionalReplayUsesSharedProductionTransactionPolicy) {
  std::ifstream stream(
    fs::path(SUNSHINE_SOURCE_DIR) / "src/sbs_bench_harness.cpp",
    std::ios::binary
  );
  ASSERT_TRUE(stream);
  std::string source {
    std::istreambuf_iterator<char>(stream),
    std::istreambuf_iterator<char>()
  };
  source.erase(std::remove(source.begin(), source.end(), '\r'), source.end());

  EXPECT_NE(source.find("--device-conditional-replay"), std::string::npos);
  EXPECT_NE(
    source.find("--device-conditional-replay-control"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("models::gpu_adaptive_transaction_policy_t device_conditional_policy"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("device_conditional_policy.make_request("),
    std::string::npos
  );
  EXPECT_NE(
    source.find("device_conditional_policy.record_submission("),
    std::string::npos
  );
  EXPECT_NE(
    source.find(".record_known_force_infer_completion(estimator_frame_id, true)"),
    std::string::npos
  );
  EXPECT_EQ(source.find("device_conditional_baseline_is_opaque"), std::string::npos);
  EXPECT_NE(
    source.find(
      "finish_pending_depth_for_evaluation(\n"
      "          input_color,\n"
      "          capture_convex2x_diagnostics\n"
      "        )"
    ),
    std::string::npos
  );
  EXPECT_NE(
    source.find("only unavailable DDup/window/cadence admission is"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("validate_complete_gpu_trace_replay("),
    std::string::npos
  );
  EXPECT_NE(
    source.find("gpu_trace_source_closure_sha256"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("branch-dependent/frozen"),
    std::string::npos
  );
  EXPECT_NE(source.find("final_parallax_"), std::string::npos);
  EXPECT_NE(source.find("shadow_final_parallax"), std::string::npos);
  EXPECT_NE(
    source.find("static_cast<std::uint32_t>(est.field_width)"),
    std::string::npos
  );
  EXPECT_NE(source.find("{\"field_width\", est.field_width}"), std::string::npos);
  EXPECT_NE(source.find(".depth_width = est.field_width"), std::string::npos);
  EXPECT_NE(
    source.find("est.shadow_final_parallax->GetResource"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("final_parallax_publication_policy"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("final_parallax_reuse_policy"),
    std::string::npos
  );
  EXPECT_EQ(source.find("display_parallax_"), std::string::npos);
  EXPECT_EQ(source.find("shadow_target_final_parallax"), std::string::npos);
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

TEST(OfflineSbsWorker, PreservesSubtitleStreamsButClearsTheirDispositions) {
  const auto spec = offline_sbs::parse_worker_spec(worker_spec_json());
  const auto media = offline_sbs::parse_ffprobe_contract(sdr_probe());
  const nlohmann::json inventory {
    {"streams", nlohmann::json::array({
                  {
                    {"index", 0},
                    {"codec_type", "video"},
                    {"codec_name", "h264"},
                    {"time_base", "1/1000"},
                    {"disposition", {{"default", 1}}},
                  },
                  {
                    {"index", 1},
                    {"codec_type", "subtitle"},
                    {"codec_name", "subrip"},
                    {"time_base", "1/1000"},
                    {"tags", {{"language", "eng"}}},
                    {"disposition", {{"default", 1}, {"forced", 1}}},
                  },
                })},
  };

  const auto command = offline_sbs::build_mux_command_for_test(
    spec,
    media,
    inventory,
    "C:/state/jobs/one/encoded-video.mp4"
  );

  EXPECT_TRUE(contains_argument_pair(command, "-map", "1:s?"));
  EXPECT_TRUE(contains_argument_pair(
    command,
    "-metadata:s:s:0",
    "language=eng"
  ));
  EXPECT_TRUE(contains_argument_pair(command, "-disposition:s", "0"));
  EXPECT_TRUE(contains_argument_pair(
    command,
    "-disposition:v:0",
    "default"
  ));
  EXPECT_EQ(
    std::find(command.begin(), command.end(), "-disposition:s:0"),
    command.end()
  );
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
