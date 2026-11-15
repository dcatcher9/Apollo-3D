#include "src/offline_sbs_wire_contract.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <gtest/gtest.h>

namespace {
  offline_sbs::wire::whole_clip_contract_t whole_clip(
    const bool conversion
  ) {
    return {
      .observation_timeline = offline_sbs::wire::observation_timeline_contract_t {
        .schema = 1,
        .timestamp_unit = "monotonic-source-us-plus-one",
        .count = 2,
        .sha256 = std::string(64, 'b'),
      },
      .artifact_mode = conversion ? "conversion" : "adaptive",
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
        .parallax_v2_shadow = false,
        .parallax_v2_render = true,
        .parallax_v2_live = true,
        .cuda_graph_captured = true,
        .simulate_hdr = false,
        .hdr_scale = 1.0,
        .input_color_space = "sRGB-SDR",
        .input_frame_format = "sRGB-BMP-WIC",
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
        .output_scale = 1.0,
        .requested_eye_width = 0,
        .requested_eye_height = 0,
        .configured_max_encode_width = 8192,
        .resolved_max_output_width = 8192,
        .input_limit = 0,
        .output_every = 1,
        .follow_mode = true,
        .follow_format = "bmp",
        .follow_count_bound = 2,
        .follow_producer_frame_count = 2,
        .follow_frame_pattern = "frame_%010d.bmp",
        .follow_first_sequence = 1,
        .follow_poll_interval_ms = 0,
        .follow_done_sentinel = ".producer-done.json",
        .follow_failed_sentinel = ".producer-failed.json",
        .follow_progress_file = "follow_progress.json",
        .follow_native_input_deletion = false,
        .follow_atomic_sbs_publication = conversion,
        .output_eye_width = 1920,
        .output_eye_height = 1080,
        .output_sbs_width = 3840,
        .output_sbs_height = 1080,
        .content_scale_x = 1.0,
        .content_scale_y = 1.0,
        .scene_cache_write = false,
        .scene_cache_replay = false,
      },
      .adaptive_state = {
        .transport = "atomic-latest-v1",
        .header_file = "adaptive_state_header.json",
        .frame_file = "adaptive_state_frame.json",
        .retained_history = false,
        .schema = 1,
        .capture = "cut-bridge-state",
        .frame_count = 2,
      },
      .sbs = {
        .enabled = conversion,
        .file_pattern = conversion ?
                          std::optional<std::string> {"sbs_<frame-id>.png"} :
                          std::nullopt,
        .frame_count = conversion ? 2u : 0u,
        .width = conversion ? 3840u : 0u,
        .height = conversion ? 1080u : 0u,
        .frame_format = "sRGB-BGRA8-PNG",
        .transfer = "sRGB",
        .primaries = "sRGB/BT.709",
        .row_order = "top-down",
        .atomic_publication = conversion,
      },
    };
  }

  offline_sbs::wire::scene_cache_contract_t scene_cache() {
    return {
      .status = "complete",
      .source = {
        .width = 1920,
        .height = 1080,
        .frame_format = "sRGB-PNG-WIC",
        .texture_format = "B8G8R8A8_UNORM",
        .color_space = "sRGB-SDR",
      },
      .depth_width = 518,
      .depth_height = 294,
      .render = {
        .model = "DepthAnythingV2Small-hf",
        .model_url = "https://example.invalid/model.onnx",
        .pop_strength = 1.0,
        .simulate_hdr = false,
        .hdr_scale = 1.0,
        .depth_reuse_interval = 1,
        .requested_eye_width = 0,
        .requested_eye_height = 0,
        .output_scale = 1.0,
        .resolved_max_output_width = 8192,
      },
      .packed_sbs = {
        .eye_width = 1920,
        .eye_height = 1080,
        .width = 3840,
        .height = 1080,
        .texture_format = "B8G8R8A8_UNORM",
        .frame_format = "sRGB-BGRA8-PNG",
        .file_extension = "png",
      },
      .processed_count = 2,
      .frame_count = 2,
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

  offline_sbs::wire::worker_result_contract_t worker_result() {
    return {
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
      .analysis_contract = whole_clip(true),
      .replay_contracts = {},
      .cache = {
        .hard_cap_bytes = 64 * 1024 * 1024,
        .peak_bytes = 0,
        .analysis_source_raster_bytes = 512,
        .peak_live_raster_bytes = 1280,
        .peak_cache_plus_raster_bytes = 1280,
        .remaining_bytes = 0,
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
      .scene_cache_max_bytes = 64 * 1024 * 1024,
      .cache_budget_policy = "fail",
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

TEST(OfflineSbsWireContractTest, WorkerSpecRejectsRetiredCutStatePipelineField) {
  auto encoded = offline_sbs::wire::to_json(worker_spec());
  EXPECT_EQ(encoded.at("schema"), 3);
  EXPECT_FALSE(encoded.at("pipeline").contains("cut_state"));
  encoded["pipeline"]["cut_state"] = "authenticated-online-pulse-count";
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
  EXPECT_TRUE(parsed.replay_contracts.empty());
  EXPECT_EQ(parsed.analysis_contract.artifact_mode, "conversion");
  EXPECT_TRUE(parsed.analysis_contract.sbs.enabled);
  EXPECT_EQ(offline_sbs::wire::to_json(parsed), encoded);
}

TEST(OfflineSbsWireContractTest, EvaluationUsesTheSameCausalPassWithoutSbs) {
  auto original = worker_result();
  original.operation = "evaluate";
  original.analysis_contract = whole_clip(false);
  original.staging_identity.reset();
  original.output_path.reset();
  original.timeline_contract = {
    {"mode", "evaluation-only"},
    {"max_output_ticks", nullptr},
  };

  const auto encoded = offline_sbs::wire::to_json(original);
  const auto parsed =
    offline_sbs::wire::parse_worker_result_contract(encoded);

  EXPECT_EQ(parsed.operation, "evaluate");
  EXPECT_EQ(parsed.analysis_contract.artifact_mode, "adaptive");
  EXPECT_FALSE(parsed.analysis_contract.sbs.enabled);
  EXPECT_TRUE(parsed.replay_contracts.empty());
  EXPECT_FALSE(parsed.output_path.has_value());
  EXPECT_FALSE(parsed.staging_identity.has_value());
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
  const auto encoded_evaluation = [] {
    auto original = worker_result();
    original.operation = "evaluate";
    original.analysis_contract = whole_clip(false);
    original.staging_identity.reset();
    original.output_path.reset();
    original.timeline_contract = {
      {"mode", "evaluation-only"},
      {"max_output_ticks", nullptr},
    };
    return offline_sbs::wire::to_json(original);
  }();

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
    encoded["analysis_contract"]["resolved_runtime"]["scene_cache_write"] = true;
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["resolved_runtime"]["follow_poll_interval_ms"] = 10;
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["resolved_runtime"]["follow_format"] = "png";
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["resolved_runtime"]["follow_frame_pattern"] =
      "frame_%010d.png";
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["resolved_runtime"]["input_frame_format"] =
      "sRGB-PNG-WIC";
  });
  expect_rejected([](auto &encoded) {
    encoded["analysis_contract"]["resolved_runtime"]["parallax_v2_live"] = false;
  });
  expect_rejected([](auto &encoded) {
    encoded["cache"]["peak_bytes"] = 1;
  });
  expect_rejected([](auto &encoded) {
    encoded["cache"]["peak_cache_plus_raster_bytes"] = 1279;
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

TEST(OfflineSbsWireContractTest, ScenePlanRoundTripsAndRejectsMissingFields) {
  const offline_sbs::wire::scene_plan_contract_t original {
    .scenes = {{1, 3}},
  };
  const auto encoded = offline_sbs::wire::to_json(original);
  EXPECT_EQ(
    offline_sbs::wire::to_json(
      offline_sbs::wire::parse_scene_plan_contract(encoded)
    ),
    encoded
  );
  auto missing = encoded;
  missing.erase("version");
  EXPECT_THROW(
    offline_sbs::wire::parse_scene_plan_contract(missing),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, SceneCacheRoundTripsThroughOneExactCodec) {
  const auto encoded = offline_sbs::wire::to_json(scene_cache());
  const auto parsed = offline_sbs::wire::parse_scene_cache_contract(encoded);
  EXPECT_EQ(parsed.processed_count, 2u);
  EXPECT_EQ(offline_sbs::wire::to_json(parsed), encoded);
}

TEST(OfflineSbsWireContractTest, SceneCacheRejectsUnknownAndOverflowGeometry) {
  auto unknown = offline_sbs::wire::to_json(scene_cache());
  unknown["legacy_preview"] = true;
  EXPECT_THROW(
    offline_sbs::wire::parse_scene_cache_contract(unknown),
    offline_sbs::wire::contract_error
  );

  auto packed_overflow = offline_sbs::wire::to_json(scene_cache());
  packed_overflow["packed_sbs"]["eye_width"] =
    static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
  packed_overflow["packed_sbs"]["width"] =
    static_cast<std::uint32_t>(
      2ull * std::numeric_limits<std::uint32_t>::max()
    );
  EXPECT_THROW(
    offline_sbs::wire::parse_scene_cache_contract(packed_overflow),
    offline_sbs::wire::contract_error
  );

  auto depth_overflow = offline_sbs::wire::to_json(scene_cache());
  depth_overflow["depth"]["width"] =
    std::numeric_limits<std::uint32_t>::max();
  depth_overflow["depth"]["height"] =
    std::numeric_limits<std::uint32_t>::max();
  EXPECT_THROW(
    offline_sbs::wire::parse_scene_cache_contract(depth_overflow),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, SceneCacheRequiresCompleteRetainedGlyphTuple) {
  const auto valid = offline_sbs::wire::to_json(scene_cache());
  EXPECT_EQ(valid.at("schema"), 5u);
  EXPECT_EQ(valid.at("subtitle_tuple").at("schema"), 1u);
  EXPECT_EQ(valid.at("subtitle_tuple").at("renderer"), "ordinary-base-inverse-renderer-v1");
  EXPECT_EQ(valid.at("depth").at("semantics"), "depth-coordinate-v2-signed-base-parallax-source-u");
  for (unsigned mutation = 0u; mutation < 5u; ++mutation) {
    auto invalid = valid;
    if (mutation == 0u) invalid["schema"] = 4u;
    if (mutation == 1u) invalid.erase("subtitle_tuple");
    if (mutation == 2u) invalid["subtitle_tuple"]["payload_suffixes"].erase(3u);
    if (mutation == 3u) invalid["subtitle_tuple"]["renderer"] = "final-field-only";
    if (mutation == 4u) invalid["subtitle_tuple"]["renderer"] = "base-inverse-and-analytic-ui-glyph-selector";
    EXPECT_THROW(offline_sbs::wire::parse_scene_cache_contract(invalid),
                 offline_sbs::wire::contract_error) << mutation;
  }
}

TEST(OfflineSbsWireContractTest, WholeClipRoundTripsAndRejectsNestedUnknownField) {
  const auto encoded = offline_sbs::wire::to_json(whole_clip(false));
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
  legacy.erase("output_location");
  legacy.erase("retention_sequence");
  legacy.erase("worker");
  const auto parsed = offline_sbs::wire::parse_persisted_job_contract(legacy);
  EXPECT_FALSE(parsed.snapshot.output_location);
  EXPECT_FALSE(parsed.retention_sequence);
  EXPECT_FALSE(parsed.worker);

  legacy["unsupported_legacy_field"] = true;
  EXPECT_THROW(
    offline_sbs::wire::parse_persisted_job_contract(legacy),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, PersistedJobAcceptsNullLegacyOutputLocation) {
  auto legacy = offline_sbs::wire::to_json(persisted_job());
  legacy["output_location"] = nullptr;
  const auto parsed = offline_sbs::wire::parse_persisted_job_contract(legacy);
  ASSERT_TRUE(parsed.snapshot.output_path);
  EXPECT_FALSE(parsed.snapshot.output_location);
}

TEST(OfflineSbsWireContractTest, JobSnapshotRejectsNumericOverflow) {
  auto encoded = offline_sbs::wire::to_json(job_snapshot());
  encoded["progress"]["processed_frames"] = -1;
  EXPECT_THROW(
    offline_sbs::wire::parse_job_snapshot_contract(encoded, false),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, PersistedJobAcceptsAndDiscardsRetiredSceneSummary) {
  // The shape job.json had while the diagnostic scene list existed.
  const auto current = offline_sbs::wire::to_json(persisted_job());
  auto retained = current;
  retained["scene_decisions"] = nlohmann::json::array({
    {{"scene_id", 1}, {"boundary", {{"decision", "confirmed"}}}},
  });
  retained["progress"]["scene_count"] = 1;
  retained["progress"]["current_scene"] = {{"scene_id", 2}};
  retained["worker_result"] = {
    {"schema", 3},
    {"scene_count", 2},
    {"scene_audit", "C:/state/jobs/one/result/scene-audit.json"},
    {"scene_audit_manifest_sha256", std::string(64, 'b')},
    {"scene_audit_sha256", std::string(64, 'b')},
    {"scene_decisions_contract", "C:/state/jobs/one/result/scene-audit.json"},
  };

  const auto parsed = offline_sbs::wire::parse_persisted_job_contract(retained);
  EXPECT_EQ(parsed.snapshot.progress.processed_frames, 1u);
  ASSERT_TRUE(parsed.snapshot.worker_result);
  const auto rewritten = offline_sbs::wire::to_json(parsed);
  EXPECT_FALSE(rewritten.contains("scene_decisions"));
  EXPECT_FALSE(rewritten.at("progress").contains("scene_count"));
  EXPECT_FALSE(rewritten.at("progress").contains("current_scene"));

  // The older variants with a null or absent summary stay readable too.
  retained["progress"]["scene_count"] = nullptr;
  retained["progress"]["current_scene"] = nullptr;
  retained.erase("scene_decisions");
  EXPECT_NO_THROW(offline_sbs::wire::parse_persisted_job_contract(retained));

  // The API snapshot is not a durable document and has no retired variant.
  auto snapshot = offline_sbs::wire::to_json(job_snapshot());
  EXPECT_FALSE(snapshot.contains("scene_decisions"));
  EXPECT_FALSE(snapshot.at("progress").contains("scene_count"));
  EXPECT_FALSE(snapshot.at("progress").contains("current_scene"));
  snapshot["progress"]["current_scene"] = nullptr;
  EXPECT_THROW(
    offline_sbs::wire::parse_job_snapshot_contract(snapshot, false),
    offline_sbs::wire::contract_error
  );
}

TEST(OfflineSbsWireContractTest, RawTransportRoundTripsWithoutRasterFilenames) {
  auto value = worker_result();
  auto &runtime = value.analysis_contract.resolved_runtime;
  runtime.frame_transport = "bounded-raw-memory-v1";
  runtime.input_frame_format = "sRGB-BGRA8-raw";
  runtime.output_frame_format = "sRGB-BGRA8-raw";
  runtime.output_ffmpeg_pixel_format = "bgra";
  runtime.follow_format = "bgra8";
  runtime.follow_frame_pattern.reset();
  auto &sbs = value.analysis_contract.sbs;
  sbs.file_pattern.reset();
  sbs.frame_format = "bgra8";
  sbs.ffmpeg_pixel_format = "bgra";
  auto encoded = offline_sbs::wire::to_json(value);
  EXPECT_EQ(offline_sbs::wire::parse_worker_result_contract(encoded).analysis_contract.sbs.frame_format, "bgra8");
  encoded["analysis_contract"]["resolved_runtime"].erase("frame_transport");
  EXPECT_THROW(offline_sbs::wire::parse_worker_result_contract(encoded), offline_sbs::wire::contract_error);
  value.analysis_contract.sbs.file_pattern = "frame_%010d.png";
  EXPECT_THROW(offline_sbs::wire::to_json(value), offline_sbs::wire::contract_error);
}
