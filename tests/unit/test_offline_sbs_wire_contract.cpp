#include "src/offline_sbs_wire_contract.h"
#include "src/offline_frame_transport.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <gtest/gtest.h>

namespace {
  // The CLI adaptive trace: a directory clip and its retained JSONL history.
  offline_sbs::wire::whole_clip_contract_t trace_whole_clip() {
    return {
      .observation_timeline = offline_sbs::wire::observation_timeline_contract_t {
        .schema = 1,
        .timestamp_unit = "monotonic-source-us-plus-one",
        .count = 2,
        .sha256 = std::string(64, 'b'),
      },
      .artifact_mode = "adaptive",
      .inference_mode = "single-pass-tensorrt",
      .depth_inference_enabled = true,
      .scheduled_depth_update_count = 2,
      .tensorrt_enqueue_count = 2,
      .depth_provenance = "video_depth_estimator",
      .pipeline_state_provenance = "estimator-state",
      .model = "DepthAnythingV2Small-hf",
      .source_frame_count = 2,
      .source_width = 1920,
      .source_height = 1080,
      .source_first_sequence = 1,
      .depth_reuse_interval = 1,
      .resolved_runtime = {
        .model = "DepthAnythingV2Small-hf",
        .model_url = "https://example.invalid/model.onnx",
        .pop_strength = 1.0,
        .depth_width = 518,
        .depth_height = 294,
        .depth_reuse_interval = 1,
        .cuda_graph = true,
        .cuda_graph_captured = true,
        .input_color_space = "sRGB-SDR",
        .input_frame_format = "sRGB-PNG-WIC",
        .input_texture_format = "B8G8R8A8_UNORM",
        .input_transfer = "sRGB",
        .input_primaries = "sRGB/BT.709",
        .packed_texture_format = "B8G8R8A8_UNORM",
        .packed_color_space = "sRGB-SDR",
        .packed_transfer = "sRGB",
        .packed_primaries = "sRGB/BT.709",
        .output_frame_format = "sRGB-BGRA8-PNG",
        .output_transfer = "sRGB",
        .output_primaries = "sRGB/BT.709",
        .output_row_order = "top-down",
        .configured_max_encode_width = 8192,
        .resolved_max_output_width = 8192,
        .input_limit = 0,
        .output_every = 1,
        .follow_mode = false,
        .follow_native_input_deletion = false,
        .follow_atomic_sbs_publication = false,
        .output_eye_width = 1920,
        .output_eye_height = 1080,
        .output_sbs_width = 3840,
        .output_sbs_height = 1080,
        .content_scale_x = 1.0,
        .content_scale_y = 1.0,
      },
      .adaptive_state = {
        .transport = "jsonl-v1",
        .file = "adaptive_state.jsonl",
        .retained_history = true,
        .schema = 1,
        .capture = "depth-health-state",
        .frame_count = 2,
      },
      .sbs = {
        .enabled = false,
        .frame_format = "sRGB-BGRA8-PNG",
        .transfer = "sRGB",
        .primaries = "sRGB/BT.709",
        .row_order = "top-down",
      },
    };
  }

  offline_sbs::wire::worker_spec_contract_t worker_spec() {
    return {
      .job_id = "11111111-2222-3333-4444-555555555555",
      .operation = "convert",
      .input_path = "C:/media/source.mkv",
      .job_directory = "C:/state/jobs/one",
      .result_directory = "C:/state/jobs/one/result",
      .progress_path = "C:/state/jobs/one/progress.json",
      .result_path = "C:/state/jobs/one/result.json",
      .staging_output = "C:/media/output.part.mkv",
      .sunshine_executable = "C:/Program Files/Sunshine/sunshine.exe",
      .sunshine_config = "C:/ProgramData/Sunshine/sunshine.conf",
      .ffmpeg_path = "C:/Program Files/Sunshine/tools/ffmpeg.exe",
      .ffmpeg_version = "ffmpeg 8",
      .ffprobe_path = "C:/Program Files/Sunshine/tools/ffprobe.exe",
      .ffprobe_version = "ffprobe 8",
      .codec = "hevc_nvenc",
      .transient_raster_hard_cap_bytes = 4ull * 1024ull * 1024ull * 1024ull,
    };
  }

  // The worker's raw-memory transport with its in-memory completion evidence.
  offline_sbs::wire::whole_clip_contract_t memory_whole_clip(const bool conversion) {
    auto contract = trace_whole_clip();
    contract.artifact_mode = conversion ? "conversion" : "adaptive";
    auto &runtime = contract.resolved_runtime;
    runtime.frame_transport = offline_sbs::raw_frame_transport::name;
    runtime.input_frame_format = "sRGB-BGRA8-raw";
    runtime.output_frame_format = "sRGB-BGRA8-raw";
    runtime.output_ffmpeg_pixel_format = "bgra";
    runtime.follow_mode = true;
    runtime.follow_format = "bgra8";
    runtime.follow_count_bound = 2;
    runtime.follow_producer_frame_count = 2;
    runtime.follow_first_sequence = 1;
    runtime.follow_poll_interval_ms = 0;
    runtime.follow_atomic_sbs_publication = conversion;
    auto &adaptive = contract.adaptive_state;
    adaptive.transport = offline_sbs::completion_channel_t::name;
    adaptive.file.reset();
    adaptive.retained_history = false;
    adaptive.schema = sbs_adaptive_state::schema_version;
    adaptive.capture = sbs_adaptive_state::capture;
    auto &sbs = contract.sbs;
    sbs.enabled = conversion;
    sbs.frame_count = conversion ? 2u : 0u;
    sbs.width = conversion ? 3840u : 0u;
    sbs.height = conversion ? 1080u : 0u;
    sbs.frame_format = "bgra8";
    sbs.ffmpeg_pixel_format = "bgra";
    sbs.atomic_publication = conversion;
    return contract;
  }

  offline_sbs::wire::worker_result_contract_t worker_result(const bool conversion = true) {
    offline_sbs::wire::worker_result_contract_t result {
      .job_id = "11111111-2222-3333-4444-555555555555",
      .operation = "convert",
      .codec = "hevc_nvenc",
      .worker_spec_sha256 = std::string(64, 'a'),
      .source = {
        .width = 1920,
        .height = 1080,
        .frame_count = 2,
        .duration_seconds = 2.0 / 30.0,
        .variable_frame_rate = false,
        .color_mode = "sdr",
        .contract_path = "C:/state/jobs/one/result/source-contract.json",
      },
      .analysis_contract = memory_whole_clip(conversion),
      .cache = {
        .hard_cap_bytes = 64 * 1024 * 1024,
        .analysis_source_raster_bytes = 512,
        .peak_live_raster_bytes = 1280,
      },
      .timeline_contract = {
        {"mode", "bounded-output-timebase"},
        {"container", "matroska"},
        {"max_output_ticks", 1},
        {"absolute_timestamps", true},
        {"cumulative_drift_allowed", false},
        {"observed_video", nlohmann::json::object()},
        {"observed_auxiliary", nlohmann::json::object()},
      },
      .staging_identity = nlohmann::json {{"schema", 1}},
      .output_path = "C:/media/output.part.mkv",
    };
    if (!conversion) {
      result.operation = "evaluate";
      result.staging_identity.reset();
      result.output_path.reset();
      result.timeline_contract = {
        {"mode", "evaluation-only"},
        {"max_output_ticks", nullptr},
      };
    }
    return result;
  }

  offline_sbs::wire::job_snapshot_contract_t job_snapshot() {
    return {
      .id = "11111111-2222-3333-4444-555555555555",
      .state = "running",
      .operation = "convert",
      .input_path = "C:/media/source.mkv",
      .output_path = "C:/media/output.mkv",
      .output_location = "input-directory",
      .codec = "hevc_nvenc",
      .transient_raster_hard_cap_bytes = 64 * 1024 * 1024,
      .progress = {
        .phase = "analysis",
        .processed_frames = 1,
        .total_frames = 2,
        .source_time_seconds = 1.0 / 30.0,
        .source_duration_seconds = 2.0 / 30.0,
      },
      .created_at_unix_ms = 1,
      .started_at_unix_ms = 2,
    };
  }

  offline_sbs::wire::persisted_job_contract_t persisted_job() {
    return {
      .snapshot = job_snapshot(),
      .retention_sequence = 1,
      .worker = offline_sbs::wire::persisted_worker_contract_t {
        .spec_path = "C:/state/jobs/one/worker-spec.json",
        .spec_sha256 = std::string(64, 'c'),
        .progress_path = "C:/state/jobs/one/worker-progress.json",
        .result_path = "C:/state/jobs/one/worker-result.json",
        .log_path = "C:/state/jobs/one/worker.log",
        .result_directory = "C:/state/jobs/one/result",
        .staging_output = "C:/media/output.part.mkv",
        .ffmpeg_path = "C:/Program Files/Sunshine/tools/ffmpeg.exe",
        .ffmpeg_version = "ffmpeg 8",
        .ffprobe_path = "C:/Program Files/Sunshine/tools/ffprobe.exe",
        .ffprobe_version = "ffprobe 8",
      },
    };
  }
}  // namespace

TEST(OfflineSbsWireContractTest, WorkerSpecRoundTripsThroughOneExactCodec) {
  const auto original = worker_spec();
  const auto encoded = offline_sbs::wire::to_json(original);
  const auto parsed = offline_sbs::wire::parse_worker_spec_contract(encoded);

  EXPECT_EQ(parsed.job_id, original.job_id);
  EXPECT_EQ(parsed.staging_output, original.staging_output);
  EXPECT_EQ(parsed.ffmpeg_version, original.ffmpeg_version);
  EXPECT_EQ(
    parsed.transient_raster_hard_cap_bytes,
    original.transient_raster_hard_cap_bytes
  );
  EXPECT_EQ(offline_sbs::wire::to_json(parsed), encoded);
}

TEST(OfflineSbsWireContractTest, WorkerSpecRejectsUnknownFields) {
  auto encoded = offline_sbs::wire::to_json(worker_spec());
  encoded["legacy_python_worker"] = false;
  EXPECT_THROW(
    offline_sbs::wire::parse_worker_spec_contract(encoded),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, WorkerSpecRejectsUnknownPipelineField) {
  auto encoded = offline_sbs::wire::to_json(worker_spec());
  EXPECT_EQ(encoded.at("schema"), 3);
  encoded["pipeline"]["unknown_field"] = "unknown";
  EXPECT_THROW(
    offline_sbs::wire::parse_worker_spec_contract(encoded),
    offline_sbs::wire::contract_error
  );
  encoded = offline_sbs::wire::to_json(worker_spec());
  encoded["schema"] = 2;
  EXPECT_THROW(
    offline_sbs::wire::parse_worker_spec_contract(encoded),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, JsonParserRejectsDuplicateKeysBeforeCodecUse) {
  EXPECT_THROW(
    offline_sbs::wire::parse_json_without_duplicate_keys(
      R"({"schema":1,"nested":{"value":1,"value":2}})"
    ),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, Sha256ValidatorRequiresExactLowercaseDigest) {
  EXPECT_TRUE(offline_sbs::wire::valid_sha256_hex(std::string(64u, 'a')));
  EXPECT_TRUE(offline_sbs::wire::valid_sha256_hex(std::string(64u, '0')));
  EXPECT_FALSE(offline_sbs::wire::valid_sha256_hex(std::string(63u, 'a')));
  EXPECT_FALSE(offline_sbs::wire::valid_sha256_hex(std::string(64u, 'A')));
  EXPECT_FALSE(offline_sbs::wire::valid_sha256_hex(
    std::string(63u, 'a') + "g"
  ));
}

TEST(OfflineSbsWireContractTest, WorkerResultRoundTripsThroughOneExactCodec) {
  const auto original = worker_result();
  const auto encoded = offline_sbs::wire::to_json(original);
  const auto parsed = offline_sbs::wire::parse_worker_result_contract(encoded);

  EXPECT_EQ(parsed.worker_spec_sha256, original.worker_spec_sha256);
  EXPECT_EQ(parsed.source.frame_count, original.source.frame_count);
  EXPECT_EQ(parsed.analysis_contract.artifact_mode, "conversion");
  EXPECT_TRUE(parsed.analysis_contract.sbs.enabled);
  EXPECT_EQ(offline_sbs::wire::to_json(parsed), encoded);
}

TEST(OfflineSbsWireContractTest, ConversionAcceptsExactMp4Timeline) {
  auto original = worker_result();
  original.output_path = "C:/media/output.part.mp4";
  original.timeline_contract = {
    {"mode", "exact-rational"},
    {"container", "mp4"},
    {"max_output_ticks", 0},
    {"observed_video", nlohmann::json::object()},
    {"observed_auxiliary", nlohmann::json::object()},
  };

  const auto encoded = offline_sbs::wire::to_json(original);
  const auto parsed =
    offline_sbs::wire::parse_worker_result_contract(encoded);

  EXPECT_EQ(parsed.timeline_contract.at("mode"), "exact-rational");
  EXPECT_EQ(parsed.timeline_contract.at("container"), "mp4");
  EXPECT_EQ(parsed.timeline_contract.at("max_output_ticks"), 0);
}

TEST(OfflineSbsWireContractTest, EvaluationRejectsAnySbsAttestation) {
  const auto encoded_evaluation = offline_sbs::wire::to_json(worker_result(false));

  auto nonzero_geometry = encoded_evaluation;
  nonzero_geometry["analysis_contract"]["sbs"]["width"] = 3840;
  EXPECT_THROW(
    offline_sbs::wire::parse_worker_result_contract(nonzero_geometry),
    offline_sbs::wire::contract_error
  );

  auto atomic_publication = encoded_evaluation;
  atomic_publication["analysis_contract"]["sbs"]["atomic_publication"] = true;
  EXPECT_THROW(
    offline_sbs::wire::parse_worker_result_contract(atomic_publication),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, WorkerResultRejectsNarrowIntegerOverflow) {
  auto encoded = offline_sbs::wire::to_json(worker_result());
  encoded["source"]["width"] =
    static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1u;
  EXPECT_THROW(
    offline_sbs::wire::parse_worker_result_contract(encoded),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, WorkerResultCarriesNoSceneListAndRejectsRetiredShapes) {
  const auto encoded = offline_sbs::wire::to_json(worker_result());
  EXPECT_EQ(encoded.at("schema"), 4);
  for (const auto retired : {
         "scene_count",
         "scenes",
         "scene_audit",
         "scene_audit_manifest_sha256",
       }) {
    EXPECT_FALSE(encoded.contains(retired)) << retired;
    auto with_retired = encoded;
    with_retired[retired] = nullptr;
    EXPECT_THROW(
      offline_sbs::wire::parse_worker_result_contract(with_retired),
      offline_sbs::wire::contract_error
    ) << retired;
  }
  for (const int retired_schema : {2, 3}) {
    auto with_schema = encoded;
    with_schema["schema"] = retired_schema;
    EXPECT_THROW(
      offline_sbs::wire::parse_worker_result_contract(with_schema),
      offline_sbs::wire::contract_error
    ) << retired_schema;
  }
}

TEST(OfflineSbsWireContractTest, WorkerResultRejectsCrossContractMismatches) {
  const auto expect_rejected = [](const auto &mutate) {
    auto encoded = offline_sbs::wire::to_json(worker_result());
    mutate(encoded);
    EXPECT_THROW(
      offline_sbs::wire::parse_worker_result_contract(encoded),
      offline_sbs::wire::contract_error
    );
  };

  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["source_width"] = 1919;
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["artifact_mode"] = "adaptive";
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["sbs"]["frame_count"] = 1;
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["resolved_runtime"]["follow_poll_interval_ms"] = 10;
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["resolved_runtime"]["follow_format"] = "png";
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["resolved_runtime"]["input_frame_format"] =
      "sRGB-PNG-WIC";
  });
  expect_rejected([](auto &encoded) {
    // The CLI adaptive trace is a valid contract but never comes from a worker.
    encoded["analysis_contract"] = offline_sbs::wire::to_json(trace_whole_clip());
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["adaptive_state"] = {
      {"transport", "jsonl-v1"},
      {"file", "adaptive_state.jsonl"},
      {"retained_history", true},
      {"schema", 1},
      {"capture", "depth-health-state"},
      {"frame_count", 2},
    };
  });
  expect_rejected([](auto &encoded) {
    encoded["cache"]["peak_live_raster_bytes"] = 511;
  });
  expect_rejected([](auto &encoded) {
    encoded["cache"]["peak_live_raster_bytes"] = 64 * 1024 * 1024 + 1;
  });
  expect_rejected([](auto &encoded) {
    encoded["timeline_contract"]["max_output_ticks"] = 0;
  });
  expect_rejected([](auto &encoded) {
    encoded["timeline_contract"].erase("observed_video");
  });
  expect_rejected([](auto &encoded) {
    encoded["source"]["frame_count"] = 3;
    encoded["analysis_contract"]["observation_timeline"]["count"] = 3;
    encoded["analysis_contract"]["scheduled_depth_update_count"] = 3;
    encoded["analysis_contract"]["tensorrt_enqueue_count"] = 3;
    encoded["analysis_contract"]["source_frame_count"] = 3;
    encoded["analysis_contract"]["adaptive_state"]["frame_count"] = 3;
  });
}

TEST(OfflineSbsWireContractTest, WholeClipRoundTripsAndRejectsNestedUnknownField) {
  const auto encoded = offline_sbs::wire::to_json(trace_whole_clip());
  const auto parsed = offline_sbs::wire::parse_whole_clip_contract(encoded);
  EXPECT_EQ(parsed.resolved_runtime.model, parsed.model);
  EXPECT_FALSE(parsed.sbs.enabled);
  EXPECT_EQ(parsed.sbs.frame_count, 0u);
  EXPECT_EQ(parsed.sbs.width, 0u);
  EXPECT_EQ(parsed.sbs.height, 0u);
  EXPECT_EQ(offline_sbs::wire::to_json(parsed), encoded);

  auto unknown = encoded;
  unknown["resolved_runtime"]["legacy_roi"] = false;
  EXPECT_THROW(
    offline_sbs::wire::parse_whole_clip_contract(unknown),
    offline_sbs::wire::contract_error
  );

  auto disabled_geometry = encoded;
  disabled_geometry["sbs"]["width"] = 3840;
  EXPECT_THROW(
    offline_sbs::wire::parse_whole_clip_contract(disabled_geometry),
    offline_sbs::wire::contract_error
  );

  auto disabled_atomic_publication = encoded;
  disabled_atomic_publication["sbs"]["atomic_publication"] = true;
  EXPECT_THROW(
    offline_sbs::wire::parse_whole_clip_contract(disabled_atomic_publication),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, PersistedJobRoundTripsCurrentWireShape) {
  const auto original = persisted_job();
  const auto encoded = offline_sbs::wire::to_json(original);
  const auto parsed = offline_sbs::wire::parse_persisted_job_contract(encoded);
  ASSERT_TRUE(parsed.retention_sequence);
  ASSERT_TRUE(parsed.worker);
  EXPECT_EQ(parsed.snapshot.output_location, "input-directory");
  EXPECT_EQ(parsed.worker->spec_sha256, std::string(64, 'c'));
  EXPECT_EQ(offline_sbs::wire::to_json(parsed), encoded);
}

TEST(OfflineSbsWireContractTest, PersistedJobSupportsOnlyExplicitLegacyOmissions) {
  auto legacy = offline_sbs::wire::to_json(persisted_job());
  legacy.erase("retention_sequence");
  legacy.erase("worker");
  const auto parsed = offline_sbs::wire::parse_persisted_job_contract(legacy);
  EXPECT_FALSE(parsed.retention_sequence);
  EXPECT_FALSE(parsed.worker);

  auto without_location = legacy;
  without_location.erase("output_location");
  EXPECT_THROW(
    offline_sbs::wire::parse_persisted_job_contract(without_location),
    offline_sbs::wire::contract_error
  );

  legacy["unsupported_legacy_field"] = true;
  EXPECT_THROW(
    offline_sbs::wire::parse_persisted_job_contract(legacy),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, JobSnapshotRejectsNumericOverflow) {
  auto encoded = offline_sbs::wire::to_json(job_snapshot());
  encoded["progress"]["processed_frames"] = -1;
  EXPECT_THROW(
    offline_sbs::wire::parse_job_snapshot_contract(encoded, false),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, PersistedJobRejectsTheRetiredSceneSummaryAndSplitPolicy) {
  // job.json written before 08-14 carried these; such a record no longer loads.
  const auto current = offline_sbs::wire::to_json(persisted_job());
  EXPECT_EQ(current.at("cache_budget_policy"), "fail");
  const auto rejects = [](const nlohmann::json &retired) {
    EXPECT_THROW(
      offline_sbs::wire::parse_persisted_job_contract(retired),
      offline_sbs::wire::contract_error
    ) << retired.dump();
  };
  auto retired = current;
  retired["scene_decisions"] = nlohmann::json::array();
  rejects(retired);
  retired = current;
  retired["progress"]["scene_count"] = 1;
  rejects(retired);
  retired = current;
  retired["progress"]["current_scene"] = nullptr;
  rejects(retired);
  retired = current;
  retired["cache_budget_policy"] = "split";
  rejects(retired);
}

TEST(OfflineSbsWireContractTest, RawTransportRoundTripsWithoutRasterFilenames) {
  auto value = worker_result();
  auto encoded = offline_sbs::wire::to_json(value);
  EXPECT_EQ(offline_sbs::wire::parse_worker_result_contract(encoded).analysis_contract.sbs.frame_format, "bgra8");
  encoded["analysis_contract"]["resolved_runtime"].erase("frame_transport");
  EXPECT_THROW(offline_sbs::wire::parse_worker_result_contract(encoded), offline_sbs::wire::contract_error);
}

TEST(OfflineSbsWireContractTest, MemoryCompletionConversionRoundTripsWithoutEvidencePaths) {
  const auto encoded = offline_sbs::wire::to_json(worker_result());
  const auto &analysis = encoded.at("analysis_contract");
  EXPECT_EQ(encoded.at("schema"), 4);
  EXPECT_EQ(analysis.at("adaptive_state"), (nlohmann::json {
                                             {"transport", "bounded-completion-memory-v1"},
                                             {"retained_history", false},
                                             {"schema", sbs_adaptive_state::schema_version},
                                             {"capture", sbs_adaptive_state::capture},
                                             {"frame_count", 2},
                                           }));
  const auto &runtime = analysis.at("resolved_runtime");
  EXPECT_EQ(runtime.at("frame_transport"), "bounded-raw-memory-v1");
  EXPECT_EQ(runtime.at("follow_mode"), true);
  EXPECT_EQ(runtime.at("follow_poll_interval_ms"), 0);
  EXPECT_EQ(offline_sbs::wire::to_json(offline_sbs::wire::parse_worker_result_contract(encoded)), encoded);
}

TEST(OfflineSbsWireContractTest, MemoryCompletionEvaluationKeepsTheSameCausalEvidenceWithoutSbs) {
  const auto encoded = offline_sbs::wire::to_json(worker_result(false));
  const auto parsed = offline_sbs::wire::parse_worker_result_contract(encoded);
  EXPECT_EQ(parsed.operation, "evaluate");
  EXPECT_EQ(parsed.analysis_contract.artifact_mode, "adaptive");
  EXPECT_EQ(parsed.analysis_contract.adaptive_state.transport, "bounded-completion-memory-v1");
  EXPECT_FALSE(parsed.analysis_contract.sbs.enabled);
  EXPECT_EQ(parsed.analysis_contract.sbs.frame_count, 0u);
  EXPECT_FALSE(parsed.output_path);
  EXPECT_FALSE(parsed.staging_identity);
  EXPECT_EQ(parsed.timeline_contract, (nlohmann::json {{"mode", "evaluation-only"}, {"max_output_ticks", nullptr}}));
  EXPECT_EQ(offline_sbs::wire::to_json(parsed), encoded);
}

TEST(OfflineSbsWireContractTest, MemoryCompletionRejectsUnknownAndFileKeysEvenWhenNull) {
  const auto valid = offline_sbs::wire::to_json(worker_result().analysis_contract);
  for (const auto *key : {"file", "unknown_evidence_field"}) {
    auto invalid = valid;
    invalid["adaptive_state"][key] = nullptr;
    EXPECT_THROW(offline_sbs::wire::parse_whole_clip_contract(invalid), offline_sbs::wire::contract_error) << key;
  }
  for (const auto *key : {"transport", "retained_history", "schema", "capture", "frame_count"}) {
    auto invalid = valid;
    invalid["adaptive_state"].erase(key);
    EXPECT_THROW(offline_sbs::wire::parse_whole_clip_contract(invalid), offline_sbs::wire::contract_error) << key;
  }
}

TEST(OfflineSbsWireContractTest, MemoryCompletionSerializerRejectsContradictoryTypedFilePaths) {
  auto value = worker_result().analysis_contract;
  value.adaptive_state.file = "frames.jsonl";
  EXPECT_THROW(offline_sbs::wire::to_json(value), offline_sbs::wire::contract_error);
}

TEST(OfflineSbsWireContractTest, MemoryCompletionRequiresNativeRawCausalTransportWithoutFileWakeups) {
  const auto valid = offline_sbs::wire::to_json(worker_result().analysis_contract);
  auto no_raw_transport = valid;
  no_raw_transport["resolved_runtime"].erase("frame_transport");
  EXPECT_THROW(offline_sbs::wire::parse_whole_clip_contract(no_raw_transport), offline_sbs::wire::contract_error);
  const nlohmann::json contradictions {
    {"frame_transport", "future-raw-memory-v2"},
    {"follow_mode", false},
    {"follow_poll_interval_ms", 1},
    {"follow_native_input_deletion", true},
  };
  for (const auto &[key, value] : contradictions.items()) {
    auto invalid = valid;
    invalid["resolved_runtime"][key] = value;
    EXPECT_THROW(offline_sbs::wire::parse_whole_clip_contract(invalid), offline_sbs::wire::contract_error) << key;
  }
  auto no_poll_contract = valid;
  no_poll_contract["resolved_runtime"]["follow_poll_interval_ms"] = nullptr;
  EXPECT_THROW(offline_sbs::wire::parse_whole_clip_contract(no_poll_contract), offline_sbs::wire::contract_error);
}

TEST(OfflineSbsWireContractTest, MemoryCompletionRequiresCurrentGeneratedEvidenceForEverySourceFrame) {
  const auto valid = offline_sbs::wire::to_json(worker_result().analysis_contract);
  const nlohmann::json contradictions {
    {"transport", "unknown-completion-transport"},
    {"retained_history", true},
    {"schema", 1},
    {"capture", "depth-health-state"},
    {"frame_count", 1},
  };
  for (const auto &[key, value] : contradictions.items()) {
    auto invalid = valid;
    invalid["adaptive_state"][key] = value;
    EXPECT_THROW(offline_sbs::wire::parse_whole_clip_contract(invalid), offline_sbs::wire::contract_error) << key;
  }
}

TEST(OfflineSbsWireContractTest, AdaptiveTraceKeepsItsSchemaAndFile) {
  const auto jsonl = offline_sbs::wire::to_json(trace_whole_clip());
  EXPECT_EQ(jsonl.at("adaptive_state").at("schema"), 1);
  EXPECT_EQ(jsonl.at("adaptive_state").at("capture"), "depth-health-state");
  EXPECT_EQ(jsonl.at("adaptive_state").at("file"), "adaptive_state.jsonl");
  EXPECT_EQ(offline_sbs::wire::to_json(offline_sbs::wire::parse_whole_clip_contract(jsonl)), jsonl);
  auto unknown = jsonl;
  unknown["adaptive_state"]["header_file"] = "header.json";
  EXPECT_THROW(offline_sbs::wire::parse_whole_clip_contract(unknown), offline_sbs::wire::contract_error);
}

TEST(OfflineSbsWireContractTest, PersistedWorkerEvidencePreservesLegacyAndNativeBodiesWithoutRetagging) {
  const nlohmann::json old_result {
    {"schema", 3},
    {"analysis_contract", {{"adaptive_state", {{"transport", "atomic-latest-v1"}, {"header_file", "old-header.json"}, {"frame_file", "old-frame.json"}}}}},
    {"retained_extension", {{"opaque", true}}},
  };
  const auto native_result = offline_sbs::wire::to_json(worker_result());
  // A file-transport worker result retained by an earlier release; no current worker emits it.
  auto legacy_file_result = native_result;
  legacy_file_result["analysis_contract"] = offline_sbs::wire::to_json(trace_whole_clip());
  for (const auto &body : {old_result, legacy_file_result, native_result}) {
    auto retained = persisted_job();
    retained.snapshot.worker_result = body;
    const auto encoded = offline_sbs::wire::to_json(retained);
    const auto parsed = offline_sbs::wire::parse_persisted_job_contract(encoded);
    ASSERT_TRUE(parsed.snapshot.worker_result);
    EXPECT_EQ(*parsed.snapshot.worker_result, body);
    EXPECT_EQ(offline_sbs::wire::to_json(parsed).at("worker_result"), body);
  }
}
