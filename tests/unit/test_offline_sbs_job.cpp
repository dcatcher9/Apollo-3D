#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include <gtest/gtest.h>

#include "src/crypto.h"
#include "src/gpu_workload_arbiter.h"
#include "src/offline_sbs_contract.h"
#include "src/offline_sbs_job.h"
#include "src/offline_sbs_wire_contract.h"
#include "src/platform/common.h"

#ifdef _WIN32
  #include <windows.h>
#endif

using namespace std::chrono_literals;

namespace {
  namespace fs = std::filesystem;

  class temporary_tree_t {
  public:
    temporary_tree_t() {
      path = fs::temp_directory_path() /
             ("sunshine3d-offline-job-" +
              std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()
              ));
      fs::create_directories(path);
    }

    ~temporary_tree_t() {
      std::error_code ignored;
      fs::remove_all(path, ignored);
    }

    fs::path path;
  };

  void write_nonempty(const fs::path &path, const std::string_view contents = "x") {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  }

  nlohmann::json valid_scene_audit(const std::string_view status) {
    offline_sbs::scene_plan_t scene;
    scene.scene_id = 1;
    scene.semantic_scene_id = 1;
    scene.start_sequence = 1;
    scene.end_sequence_exclusive = 11;
    scene.frame_count = 10;
    scene.cache_bytes = 0;
    scene.evidence.source_frame_count = 10;
    scene.evidence.depth_update_count = 10;
    scene.boundary.final_sequence = 11;
    scene.boundary.decision = offline_sbs::boundary_decision_e::end_of_stream;
    scene.boundary.reason = "end of stream";
    scene.boundary.accepted = true;
    scene.cut_state_semantics = "causal-production-exact";
    scene.known_limit =
      "diagnostic scene epochs only; rendering is committed causally per source frame";
    return offline_sbs::wire::to_json(
      offline_sbs::wire::scene_audit_contract_t {
        .status = std::string {status},
        .peak_cache_bytes = 0,
        .analysis_source_raster_bytes = 256,
        .peak_live_raster_bytes = 128,
        .peak_cache_plus_raster_bytes = 256,
        .hard_cap_bytes = 2048,
        .timeline_contract = {{"mode", "evaluation-only"}},
        .scenes = {scene},
        .boundary_revisions = {scene.boundary},
      }
    );
  }

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

#ifdef _WIN32
  nlohmann::json windows_file_identity(const fs::path &path) {
    const HANDLE handle = CreateFileW(
      path.c_str(),
      FILE_READ_ATTRIBUTES,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr,
      OPEN_EXISTING,
      FILE_FLAG_OPEN_REPARSE_POINT,
      nullptr
    );
    if (handle == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("cannot open test file identity");
    }
    BY_HANDLE_FILE_INFORMATION information {};
    const auto inspected =
      GetFileInformationByHandle(handle, &information);
    CloseHandle(handle);
    if (!inspected) {
      throw std::runtime_error("cannot inspect test file identity");
    }
    return {
      {"volume_serial", information.dwVolumeSerialNumber},
      {"file_index_high", information.nFileIndexHigh},
      {"file_index_low", information.nFileIndexLow},
      {
        "size_bytes",
        (static_cast<std::uint64_t>(information.nFileSizeHigh) << 32u) |
          information.nFileSizeLow
      },
    };
  }
#endif

  offline_sbs::service_config_t service_config(
    const fs::path &root
  ) {
    const auto executable = root / "sunshine.exe";
    const auto config = root / "sunshine.conf";
    const auto ffmpeg = root / "ffmpeg.exe";
    const auto ffprobe = root / "ffprobe.exe";
    write_nonempty(executable);
    write_nonempty(config);
    write_nonempty(ffmpeg);
    write_nonempty(ffprobe);
    return {
      .state_root = root / "state",
      .worker_root = root / "worker",
      .exports_root = root / "exports",
      .sunshine_executable = executable,
      .sunshine_config = config,
      .ffmpeg_executable = ffmpeg,
      .ffprobe_executable = ffprobe,
      .max_retained_jobs = 8,
      .process_poll_interval = 5ms,
      .probe_media_tools = false,
    };
  }

  offline_sbs::job_snapshot_t wait_for_terminal(
    offline_sbs::job_service_t &service,
    const std::string &id
  ) {
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while (std::chrono::steady_clock::now() < deadline) {
      const auto reply = service.get(id);
      if (
        reply.ok &&
        (
          reply.job->state == offline_sbs::job_state_e::complete ||
          reply.job->state == offline_sbs::job_state_e::canceled ||
          reply.job->state == offline_sbs::job_state_e::failed ||
          reply.job->state == offline_sbs::job_state_e::interrupted
        )
      ) {
        return *reply.job;
      }
      std::this_thread::sleep_for(5ms);
    }
    throw std::runtime_error("timed out waiting for offline job");
  }
}  // namespace

TEST(OfflineSbsJob, CompletesThroughStagingAndPersistsAtomicTerminalState) {
  temporary_tree_t tree;
  const auto input = tree.path / "media" / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &progress) {
      std::ifstream spec_stream(context.worker_spec, std::ios::binary);
      const std::string spec_bytes {
        std::istreambuf_iterator<char> {spec_stream},
        std::istreambuf_iterator<char> {},
      };
      EXPECT_EQ(context.worker_spec_sha256, sha256_hex(spec_bytes));
      progress({
        .phase = "render",
         .processed_frames = 12,
         .total_frames = 12,
         .scene_count = 2,
         .current_scene = {
           {"index", 1},
           {"phase", "render"},
         },
         .scene_decisions = {
           {
             {"index", 0},
             {"start_sequence", 1},
             {"end_sequence_exclusive", 7},
           },
           {
             {"index", 1},
             {"start_sequence", 7},
             {"end_sequence_exclusive", 13},
           },
         },
       });
      write_nonempty(*context.staging_output, "encoded-video");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 2},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "converted.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;

  const auto terminal = wait_for_terminal(service, created.job->id);
  EXPECT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  ASSERT_TRUE(terminal.output_path);
  EXPECT_EQ(*terminal.output_path, input.parent_path() / "converted.mkv");
  EXPECT_EQ(
    terminal.output_location,
    offline_sbs::output_location_e::input_directory
  );
  EXPECT_TRUE(fs::is_regular_file(*terminal.output_path));
  EXPECT_EQ(fs::file_size(*terminal.output_path), 13u);
  EXPECT_EQ(terminal.progress.processed_frames, 12u);
  EXPECT_EQ(terminal.progress.current_scene["index"], 1);
  ASSERT_EQ(terminal.progress.scene_decisions.size(), 2u);
  EXPECT_EQ(terminal.progress.scene_decisions[1]["start_sequence"], 7);
  EXPECT_EQ(terminal.worker_result["scene_count"], 2);

  const auto summaries = service.list();
  ASSERT_EQ(summaries.size(), 1u);
  EXPECT_TRUE(summaries.front().progress.scene_decisions.empty());
  EXPECT_TRUE(summaries.front().worker_result.is_null());
  const auto detail = service.get(terminal.id);
  ASSERT_TRUE(detail.ok);
  ASSERT_TRUE(detail.job);
  EXPECT_EQ(detail.job->progress.scene_decisions.size(), 2u);
  EXPECT_EQ(detail.job->worker_result["scene_count"], 2);

  const auto state_path =
    tree.path / "state" / "jobs" / terminal.id / "job.json";
  ASSERT_TRUE(fs::is_regular_file(state_path));
  EXPECT_FALSE(fs::exists(state_path.wstring() + L".part"));
  const auto persisted = nlohmann::json::parse(std::ifstream(state_path));
  EXPECT_EQ(persisted["state"], "complete");
  EXPECT_EQ(persisted["output_location"], "input-directory");
  EXPECT_EQ(persisted["progress"]["processed_frames"], 12);
  EXPECT_EQ(persisted["progress"]["current_scene"]["index"], 1);
  ASSERT_EQ(persisted["scene_decisions"].size(), 2u);
  EXPECT_EQ(persisted["scene_decisions"][0]["end_sequence_exclusive"], 7);
  const auto capabilities = service.capabilities();
  EXPECT_EQ(capabilities["schema"], 2);
  EXPECT_TRUE(capabilities["pipeline"]["causal_online_logic"].get<bool>());
  EXPECT_TRUE(
    capabilities["pipeline"]["single_estimator_renderer_pass"].get<bool>()
  );
  EXPECT_FALSE(capabilities["pipeline"]["lookahead"].get<bool>());
  EXPECT_FALSE(capabilities["pipeline"]["scene_cache"].get<bool>());
  EXPECT_FALSE(capabilities["pipeline"]["replay"].get<bool>());
  EXPECT_FALSE(capabilities.contains("scene_cache"));
  EXPECT_EQ(capabilities["output_security"]["location"], "input-directory");
  EXPECT_TRUE(
    capabilities["output_security"]["destination_derived_from_input"]
      .get<bool>()
  );
  EXPECT_TRUE(capabilities["output_security"]["legacy_root"].is_string());
  service.shutdown();
}

TEST(OfflineSbsJob, MissingFormerInputDirectoryDoesNotBlockQuotaAccounting) {
  temporary_tree_t tree;
  const auto media_directory = tree.path / "removable-media";
  const auto input = media_directory / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      if (context.operation == offline_sbs::operation_e::convert) {
        write_nonempty(*context.staging_output, "encoded-video");
      }
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto converted = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "converted.mkv",
  });
  ASSERT_TRUE(converted.ok) << converted.error;
  ASSERT_EQ(
    wait_for_terminal(service, converted.job->id).state,
    offline_sbs::job_state_e::complete
  );

  ASSERT_TRUE(fs::remove(media_directory / "source.mkv"));
  ASSERT_TRUE(fs::remove(media_directory / "converted.mkv"));
  ASSERT_TRUE(fs::remove(media_directory));

  const auto next_input = tree.path / "next-source.mkv";
  write_nonempty(next_input, "next");
  const auto admitted = service.create({
    .input_path = next_input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(admitted.ok) << admitted.error;
  EXPECT_EQ(
    wait_for_terminal(service, admitted.job->id).state,
    offline_sbs::job_state_e::complete
  );
  service.shutdown();
}

TEST(OfflineSbsJob, PublishesCausalSceneDuringDirectConversion) {
  temporary_tree_t tree;
  const auto input = tree.path / "media" / "source.mkv";
  write_nonempty(input, "source");
  std::atomic_bool causal_scene_published {false};
  std::atomic_bool release_worker {false};

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [&](const offline_sbs::worker_context_t &context,
        const std::stop_token stop,
        const offline_sbs::progress_callback_t &progress) {
      progress({
        .phase = "convert",
        .processed_frames = 183,
        .total_frames = 185,
        .scene_count = 0,
      });
      progress({
        .phase = "convert",
        .processed_frames = 185,
        .total_frames = 185,
        .scene_count = 1,
        .current_scene = {
          {"scene_id", 1},
          {"start_sequence", 1},
          {"end_sequence_exclusive", 186},
        },
        .scene_decisions = {
          {
            {"scene_id", 1},
            {"start_sequence", 1},
            {"end_sequence_exclusive", 186},
            {"boundary", {{"decision", "end_of_stream"}}},
          },
        },
      });
      causal_scene_published = true;
      while (!release_worker && !stop.stop_requested()) {
        std::this_thread::sleep_for(1ms);
      }
      if (stop.stop_requested()) {
        return offline_sbs::worker_outcome_t {.canceled = true};
      }
      write_nonempty(*context.staging_output, "encoded-video");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "converted.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!causal_scene_published &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_TRUE(causal_scene_published);
  const auto active = service.get(created.job->id);
  ASSERT_TRUE(active.ok);
  ASSERT_TRUE(active.job);
  EXPECT_EQ(active.job->state, offline_sbs::job_state_e::running);
  EXPECT_EQ(active.job->progress.phase, "convert");
  EXPECT_EQ(active.job->progress.processed_frames, 185u);
  ASSERT_TRUE(active.job->progress.scene_count);
  EXPECT_EQ(*active.job->progress.scene_count, 1u);
  ASSERT_EQ(active.job->progress.scene_decisions.size(), 1u);
  EXPECT_EQ(
    active.job->progress.scene_decisions[0]["boundary"]["decision"],
    "end_of_stream"
  );

  release_worker = true;
  const auto terminal = wait_for_terminal(service, created.job->id);
  EXPECT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  EXPECT_EQ(terminal.progress.processed_frames, 185u);
  ASSERT_EQ(terminal.progress.scene_decisions.size(), 1u);
  service.shutdown();
}

TEST(OfflineSbsJob, NativeWorkerForcesFinalProgressDrainBeforeExitStatus) {
  std::ifstream stream(
    fs::path(SUNSHINE_SOURCE_DIR) / "src/offline_sbs_job.cpp",
    std::ios::binary
  );
  ASSERT_TRUE(stream);
  const std::string source {
    std::istreambuf_iterator<char> {stream},
    std::istreambuf_iterator<char> {},
  };
  const auto runner_begin = source.find("worker_outcome_t run_native_worker(");
  const auto runner_end = source.find("\n    void publish_progress(", runner_begin);
  ASSERT_NE(runner_begin, std::string::npos);
  ASSERT_NE(runner_end, std::string::npos);
  const auto runner = source.substr(runner_begin, runner_end - runner_begin);

  const auto polling_loop = runner.find("while (child.running())");
  const auto forced_drain = runner.find(
    "consume_progress_update(true)",
    polling_loop
  );
  const auto exit_status = runner.find(
    "const auto exit_code = child.exit_code()",
    polling_loop
  );
  ASSERT_NE(polling_loop, std::string::npos);
  ASSERT_NE(forced_drain, std::string::npos);
  ASSERT_NE(exit_status, std::string::npos);
  EXPECT_LT(polling_loop, forced_drain);
  EXPECT_LT(forced_drain, exit_status);
  EXPECT_NE(
    runner.find("force_read || write_time != last_progress_write"),
    std::string::npos
  );
}

TEST(OfflineSbsJob, EvaluationDoesNotRequireOrTrustAnEncoderSelection) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
    .codec = "not-an-encoder",
  });
  ASSERT_TRUE(created.ok) << created.error;
  const auto terminal = wait_for_terminal(service, created.job->id);
  EXPECT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  EXPECT_EQ(terminal.codec, "hevc_nvenc");
  EXPECT_FALSE(terminal.output_path);
  service.shutdown();
}

TEST(OfflineSbsJob, StartsForEvaluationWhenNoNvencCodecIsAvailable) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  auto config = service_config(tree.path);
  config.test_probed_codecs = std::vector<std::string> {};

  offline_sbs::job_service_t service {
    std::move(config),
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  EXPECT_TRUE(service.capabilities()["codecs"].empty());

  const auto evaluation = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(evaluation.ok) << evaluation.error;
  EXPECT_EQ(
    wait_for_terminal(service, evaluation.job->id).state,
    offline_sbs::job_state_e::complete
  );

  const auto conversion = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "unavailable.mkv",
    .codec = "hevc_nvenc",
  });
  EXPECT_FALSE(conversion.ok);
  EXPECT_EQ(conversion.code, offline_sbs::error_code_e::unavailable);
  EXPECT_NE(conversion.error.find("encoder"), std::string::npos);
  service.shutdown();
}

TEST(OfflineSbsJob, AcceptsExistingUserWorkspaceDirectories) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  fs::create_directories(config.worker_root / "jobs");
  fs::create_directories(config.exports_root);

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  EXPECT_TRUE(service.start(error)) << error;
  service.shutdown();
}

TEST(OfflineSbsJob, CreatesACompletelyAbsentNestedUserWorkspace) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  const auto user_workspace =
    tree.path / "new-user-root" / "Sunshine 3D" / "offline-sbs";
  config.worker_root = user_workspace / "worker";
  config.exports_root = user_workspace / "exports";

  ASSERT_FALSE(fs::exists(user_workspace));
  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  EXPECT_TRUE(fs::is_directory(config.worker_root / "jobs"));
  EXPECT_TRUE(fs::is_directory(config.exports_root));
  service.shutdown();
}

TEST(OfflineSbsJob, ReportsExactWorkerWorkspaceFileConflict) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  const auto conflict = config.worker_root / "jobs";
  write_nonempty(conflict, "not-a-directory");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  EXPECT_FALSE(service.start(error));
  EXPECT_NE(
    error.find("offline SBS worker root path already exists but is not a real directory"),
    std::string::npos
  ) << error;
  EXPECT_NE(error.find(conflict.generic_string()), std::string::npos) << error;
}

TEST(OfflineSbsJob, ReportsTheExactIntermediateWorkspaceFileConflict) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  const auto conflict = tree.path / "blocked-user-root";
  write_nonempty(conflict, "not-a-directory");
  config.worker_root = conflict / "offline-sbs" / "worker";

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  EXPECT_FALSE(service.start(error));
  EXPECT_NE(
    error.find(
      "offline SBS worker root path already exists but is not a real directory"
    ),
    std::string::npos
  ) << error;
  EXPECT_NE(error.find(conflict.generic_string()), std::string::npos) << error;
}

TEST(OfflineSbsJob, RejectsAnIntermediateProtectedStateFileBeforeMutation) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  const auto conflict = tree.path / "blocked-state-root";
  write_nonempty(conflict, "not-a-directory");
  config.state_root = conflict / "offline-sbs" / "state";

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  EXPECT_FALSE(service.start(error));
  EXPECT_NE(
    error.find(
      "offline SBS state root path already exists but is not a real directory"
    ),
    std::string::npos
  ) << error;
  EXPECT_NE(error.find(conflict.generic_string()), std::string::npos) << error;
}

TEST(OfflineSbsJob, ReportsExactExportsWorkspaceFileConflict) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  const auto conflict = config.exports_root;
  write_nonempty(conflict, "not-a-directory");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  EXPECT_FALSE(service.start(error));
  EXPECT_NE(
    error.find("offline SBS exports root path already exists but is not a real directory"),
    std::string::npos
  ) << error;
  EXPECT_NE(error.find(conflict.generic_string()), std::string::npos) << error;
}

TEST(OfflineSbsJob, ManagerRemovesTransientWorkAfterSuccessAndFailure) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  std::atomic<int> calls {0};

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [&calls](const offline_sbs::worker_context_t &context,
             std::stop_token,
             const offline_sbs::progress_callback_t &) {
      write_nonempty(
        context.job_directory / "native-work" / "cache.bin",
        "transient"
      );
      if (calls.fetch_add(1) == 0) {
        return offline_sbs::worker_outcome_t {
          .completed = true,
          .result = {
            {"schema", 1},
            {"job_id", context.job_id},
            {"status", "complete"},
            {"scene_count", 1},
          },
        };
      }
      return offline_sbs::worker_outcome_t {
        .error = "injected worker failure",
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;

  const auto successful = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(successful.ok) << successful.error;
  EXPECT_EQ(
    wait_for_terminal(service, successful.job->id).state,
    offline_sbs::job_state_e::complete
  );
  EXPECT_FALSE(fs::exists(
    tree.path / "worker" / "jobs" / successful.job->id / "native-work"
  ));

  const auto failed = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(failed.ok) << failed.error;
  EXPECT_EQ(
    wait_for_terminal(service, failed.job->id).state,
    offline_sbs::job_state_e::failed
  );
  EXPECT_FALSE(fs::exists(
    tree.path / "worker" / "jobs" / failed.job->id / "native-work"
  ));
  service.shutdown();
}

TEST(OfflineSbsJob, NeverDeletesAnUnattestedFailedStagingPath) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      // An injected/non-native child can create the selected pathname but cannot
      // prove it owns that file identity. The manager must preserve it on failure.
      write_nonempty(*context.staging_output, "unattested-user-data");
      return offline_sbs::worker_outcome_t {
        .error = "injected worker failed",
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "failed.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  const auto terminal = wait_for_terminal(service, created.job->id);
  ASSERT_EQ(terminal.state, offline_sbs::job_state_e::failed);
  const auto staging =
    input.parent_path() /
    (
      "failed.sunshine3d-" + created.job->id + ".part.mkv"
    );
  ASSERT_TRUE(fs::is_regular_file(staging));
  std::ifstream retained(staging, std::ios::binary);
  const std::string contents {
    std::istreambuf_iterator<char> {retained},
    std::istreambuf_iterator<char> {},
  };
  EXPECT_EQ(contents, "unattested-user-data");
  service.shutdown();
}

#ifdef _WIN32
TEST(OfflineSbsJob, ExactHandlePublicationNeverOverwritesALateDestination) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "encoded-video");
      write_nonempty(*context.final_output, "late-user-file");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "collision.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  const auto terminal = wait_for_terminal(service, created.job->id);
  EXPECT_EQ(terminal.state, offline_sbs::job_state_e::failed);
  EXPECT_NE(terminal.error.find("refusing to overwrite"), std::string::npos);
  std::ifstream final(input.parent_path() / "collision.mkv", std::ios::binary);
  const std::string final_contents {
    std::istreambuf_iterator<char> {final},
    std::istreambuf_iterator<char> {},
  };
  EXPECT_EQ(final_contents, "late-user-file");
  service.shutdown();
}

TEST(OfflineSbsJob, RejectsAStagingLeafSwappedAfterWorkerAttestation) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  fs::path retained_original;

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [&retained_original](const offline_sbs::worker_context_t &context,
                         std::stop_token,
                         const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "attested-video");
      const auto identity = windows_file_identity(*context.staging_output);
      retained_original = context.staging_output->wstring() + L".original";
      fs::rename(*context.staging_output, retained_original);
      write_nonempty(*context.staging_output, "replacement-data");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
          {"staging_identity", identity},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "swapped.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  const auto terminal = wait_for_terminal(service, created.job->id);
  EXPECT_EQ(terminal.state, offline_sbs::job_state_e::failed);
  EXPECT_NE(terminal.error.find("identity changed"), std::string::npos);
  EXPECT_FALSE(fs::exists(input.parent_path() / "swapped.mkv"));
  EXPECT_TRUE(fs::is_regular_file(retained_original));
  const auto current_staging =
    input.parent_path() /
    ("swapped.sunshine3d-" + created.job->id + ".part.mkv");
  EXPECT_TRUE(fs::is_regular_file(current_staging));
  service.shutdown();
}

TEST(OfflineSbsJob, RestartRetiresStagingAfterCrashFollowingFinalHardLink) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "encoded-video");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "crash-window.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  ASSERT_EQ(
    wait_for_terminal(service, created.job->id).state,
    offline_sbs::job_state_e::complete
  );
  service.shutdown();

  const auto final_output = input.parent_path() / "crash-window.mkv";
  const auto staging_output =
    input.parent_path() /
    (
      "crash-window.sunshine3d-" + created.job->id + ".part.mkv"
    );
  ASSERT_TRUE(CreateHardLinkW(
    staging_output.c_str(),
    final_output.c_str(),
    nullptr
  ));
  ASSERT_EQ(
    windows_file_identity(staging_output),
    windows_file_identity(final_output)
  );

  // Rewind only the durable terminal transition to model a process crash after
  // CreateHardLinkW succeeds and before staging disposition/state completion.
  const auto state_path =
    config.state_root / "jobs" / created.job->id / "job.json";
  auto persisted = nlohmann::json::parse(std::ifstream(state_path));
  ASSERT_TRUE(
    persisted["worker_result"]["publish_identity"].is_object()
  );
  persisted["state"] = "publishing";
  persisted["progress"]["phase"] = "publishing";
  persisted["ended_at_unix_ms"] = nullptr;
  write_nonempty(state_path, persisted.dump(2));

  offline_sbs::job_service_t recovered {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  ASSERT_TRUE(recovered.start(error)) << error;
  const auto snapshot = recovered.get(created.job->id);
  ASSERT_TRUE(snapshot.ok) << snapshot.error;
  ASSERT_TRUE(snapshot.job);
  EXPECT_EQ(snapshot.job->state, offline_sbs::job_state_e::complete);
  EXPECT_FALSE(fs::exists(staging_output));
  EXPECT_TRUE(fs::is_regular_file(final_output));
  EXPECT_FALSE(
    snapshot.job->worker_result.value("staging_cleanup_pending", true)
  );
  recovered.shutdown();
}

TEST(OfflineSbsJob, RecoversLegacyUnmarkedPublicationInManagedExports) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  const auto input = tree.path / "media" / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "encoded-video");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "legacy-recovery.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  ASSERT_EQ(
    wait_for_terminal(service, created.job->id).state,
    offline_sbs::job_state_e::complete
  );
  service.shutdown();

  const auto current_final = input.parent_path() / "legacy-recovery.mkv";
  const auto legacy_final = config.exports_root / "legacy-recovery.mkv";
  const auto legacy_staging =
    config.exports_root /
    (
      "legacy-recovery.sunshine3d-" + created.job->id + ".part.mkv"
    );
  fs::rename(current_final, legacy_final);
  ASSERT_TRUE(CreateHardLinkW(
    legacy_staging.c_str(),
    legacy_final.c_str(),
    nullptr
  ));

  const auto state_path =
    config.state_root / "jobs" / created.job->id / "job.json";
  auto persisted = nlohmann::json::parse(std::ifstream(state_path));
  persisted["output_path"] = legacy_final.generic_string();
  persisted.erase("output_location");
  persisted["state"] = "publishing";
  persisted["progress"]["phase"] = "publishing";
  persisted["ended_at_unix_ms"] = nullptr;
  write_nonempty(state_path, persisted.dump(2));

  offline_sbs::job_service_t recovered {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  ASSERT_TRUE(recovered.start(error)) << error;
  const auto snapshot = recovered.get(created.job->id);
  ASSERT_TRUE(snapshot.ok) << snapshot.error;
  ASSERT_TRUE(snapshot.job);
  EXPECT_EQ(snapshot.job->state, offline_sbs::job_state_e::complete);
  ASSERT_TRUE(snapshot.job->output_path);
  EXPECT_EQ(*snapshot.job->output_path, legacy_final);
  EXPECT_EQ(
    snapshot.job->output_location,
    offline_sbs::output_location_e::legacy_managed_exports
  );
  EXPECT_FALSE(fs::exists(legacy_staging));
  EXPECT_TRUE(fs::is_regular_file(legacy_final));
  recovered.shutdown();
}

TEST(OfflineSbsJob, RestartRetriesForcedStagingDispositionFailure) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  config.max_retained_artifact_bytes = 512ull * 1024ull;
  config.test_force_staging_disposition_failure = true;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "encoded-video");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "retry-disposition.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  const auto terminal = wait_for_terminal(service, created.job->id);
  ASSERT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  EXPECT_TRUE(
    terminal.worker_result.value("staging_cleanup_pending", false)
  );
  const auto staging_output =
    input.parent_path() /
    (
      "retry-disposition.sunshine3d-" + created.job->id + ".part.mkv"
    );
  ASSERT_TRUE(fs::is_regular_file(staging_output));

  const auto quota_blocked = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  EXPECT_FALSE(quota_blocked.ok);
  EXPECT_EQ(quota_blocked.code, offline_sbs::error_code_e::unavailable);
  EXPECT_NE(quota_blocked.error.find("quota"), std::string::npos);
  service.shutdown();

  config.test_force_staging_disposition_failure = false;
  offline_sbs::job_service_t recovered {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  ASSERT_TRUE(recovered.start(error)) << error;
  EXPECT_FALSE(fs::exists(staging_output));
  const auto snapshot = recovered.get(created.job->id);
  ASSERT_TRUE(snapshot.ok) << snapshot.error;
  ASSERT_TRUE(snapshot.job);
  EXPECT_EQ(snapshot.job->state, offline_sbs::job_state_e::complete);
  EXPECT_FALSE(
    snapshot.job->worker_result.value("staging_cleanup_pending", true)
  );

  const auto admitted = recovered.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(admitted.ok) << admitted.error;
  EXPECT_EQ(
    wait_for_terminal(recovered, admitted.job->id).state,
    offline_sbs::job_state_e::complete
  );
  recovered.shutdown();
}

TEST(OfflineSbsJob, RecoveryNeverRetiresAReplacementStagingIdentity) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  config.test_force_staging_disposition_failure = true;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "encoded-video");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "replacement.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  ASSERT_EQ(
    wait_for_terminal(service, created.job->id).state,
    offline_sbs::job_state_e::complete
  );
  service.shutdown();

  const auto staging_output =
    input.parent_path() /
    (
      "replacement.sunshine3d-" + created.job->id + ".part.mkv"
    );
  ASSERT_TRUE(fs::remove(staging_output));
  write_nonempty(staging_output, "replacement-user-data");

  config.test_force_staging_disposition_failure = false;
  offline_sbs::job_service_t recovered {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  ASSERT_TRUE(recovered.start(error)) << error;
  const auto snapshot = recovered.get(created.job->id);
  ASSERT_TRUE(snapshot.ok) << snapshot.error;
  ASSERT_TRUE(snapshot.job);
  EXPECT_EQ(snapshot.job->state, offline_sbs::job_state_e::complete);
  EXPECT_TRUE(
    snapshot.job->worker_result.value("staging_cleanup_pending", false)
  );
  ASSERT_TRUE(fs::is_regular_file(staging_output));
  std::ifstream replacement(staging_output, std::ios::binary);
  const std::string contents {
    std::istreambuf_iterator<char> {replacement},
    std::istreambuf_iterator<char> {},
  };
  EXPECT_EQ(contents, "replacement-user-data");
  recovered.shutdown();
}

TEST(OfflineSbsJob, RestartRejectsFinalSymlinkToAttestedStagingIdentity) {
  temporary_tree_t tree;
  const auto config = service_config(tree.path);
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "encoded-video");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "symlink-final.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  ASSERT_EQ(
    wait_for_terminal(service, created.job->id).state,
    offline_sbs::job_state_e::complete
  );
  service.shutdown();

  const auto final_output = input.parent_path() / "symlink-final.mkv";
  const auto staging_output =
    input.parent_path() /
    (
      "symlink-final.sunshine3d-" + created.job->id + ".part.mkv"
    );
  fs::rename(final_output, staging_output);
  constexpr DWORD allow_unprivileged_symlink_create = 0x2;
  if (!CreateSymbolicLinkW(
        final_output.c_str(),
        staging_output.c_str(),
        allow_unprivileged_symlink_create
      )) {
    const auto first_error = GetLastError();
    if (
      first_error != ERROR_INVALID_PARAMETER ||
      !CreateSymbolicLinkW(
        final_output.c_str(),
        staging_output.c_str(),
        0
      )
    ) {
      GTEST_SKIP()
        << "Windows file symlink creation is unavailable (error "
        << GetLastError() << ')';
    }
  }
  const auto final_attributes = GetFileAttributesW(final_output.c_str());
  ASSERT_NE(final_attributes, INVALID_FILE_ATTRIBUTES);
  ASSERT_NE(final_attributes & FILE_ATTRIBUTE_REPARSE_POINT, 0u);
  ASSERT_EQ(
    windows_file_identity(staging_output),
    windows_file_identity(final_output)
  );

  // Model a crash after publication while an untrusted actor replaces the
  // final leaf with a symlink to the same attested file identity.
  const auto state_path =
    config.state_root / "jobs" / created.job->id / "job.json";
  auto persisted = nlohmann::json::parse(std::ifstream(state_path));
  persisted["state"] = "publishing";
  persisted["progress"]["phase"] = "publishing";
  persisted["ended_at_unix_ms"] = nullptr;
  write_nonempty(state_path, persisted.dump(2));

  offline_sbs::job_service_t recovered {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  ASSERT_TRUE(recovered.start(error)) << error;
  const auto snapshot = recovered.get(created.job->id);
  ASSERT_TRUE(snapshot.ok) << snapshot.error;
  ASSERT_TRUE(snapshot.job);
  EXPECT_EQ(snapshot.job->state, offline_sbs::job_state_e::interrupted);
  EXPECT_NE(
    snapshot.job->error.find("Could not reconcile"),
    std::string::npos
  );
  EXPECT_TRUE(fs::is_regular_file(staging_output));
  const auto recovered_final_attributes =
    GetFileAttributesW(final_output.c_str());
  EXPECT_NE(recovered_final_attributes, INVALID_FILE_ATTRIBUTES);
  EXPECT_NE(
    recovered_final_attributes & FILE_ATTRIBUTE_REPARSE_POINT,
    0u
  );
  recovered.shutdown();
}

TEST(OfflineSbsJob, WindowsPublishIdentityRejectsNonFileAttributes) {
  EXPECT_TRUE(
    offline_sbs::windows_publish_identity_attributes_accepted_for_test(
      FILE_ATTRIBUTE_NORMAL,
      1
    )
  );
  EXPECT_TRUE(
    offline_sbs::windows_publish_identity_attributes_accepted_for_test(
      FILE_ATTRIBUTE_ARCHIVE,
      4096
    )
  );
  EXPECT_FALSE(
    offline_sbs::windows_publish_identity_attributes_accepted_for_test(
      FILE_ATTRIBUTE_REPARSE_POINT,
      1
    )
  );
  EXPECT_FALSE(
    offline_sbs::windows_publish_identity_attributes_accepted_for_test(
      FILE_ATTRIBUTE_DIRECTORY,
      1
    )
  );
  EXPECT_FALSE(
    offline_sbs::windows_publish_identity_attributes_accepted_for_test(
      FILE_ATTRIBUTE_NORMAL,
      0
    )
  );
}

TEST(OfflineSbsJob, RejectsPersistedInputDirectoryOutputThatWasRedirected) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  const auto input = tree.path / "media" / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "encoded-video");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "redirected.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  ASSERT_EQ(
    wait_for_terminal(service, created.job->id).state,
    offline_sbs::job_state_e::complete
  );
  service.shutdown();

  const auto outside = config.exports_root / "redirected.mkv";
  write_nonempty(outside, "unrelated-user-file");
  const auto state_path =
    config.state_root / "jobs" / created.job->id / "job.json";
  auto persisted = nlohmann::json::parse(std::ifstream(state_path));
  persisted["output_path"] = outside.generic_string();
  ASSERT_EQ(persisted["output_location"], "input-directory");
  write_nonempty(state_path, persisted.dump(2));

  offline_sbs::job_service_t recovered {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  ASSERT_TRUE(recovered.start(error)) << error;
  EXPECT_FALSE(recovered.get(created.job->id).ok);
  std::ifstream outside_stream(outside, std::ios::binary);
  const std::string outside_contents {
    std::istreambuf_iterator<char> {outside_stream},
    std::istreambuf_iterator<char> {},
  };
  EXPECT_EQ(outside_contents, "unrelated-user-file");
  recovered.shutdown();
}
#endif

TEST(OfflineSbsJob, ServesOnlyTheBoundedManagerOwnedSceneAudit) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  const auto decoy = tree.path / "outside-scene-audit.json";
  write_nonempty(input, "source");
  write_nonempty(
    decoy,
    R"({"schema":3,"version":"whole-clip-scene-audit-v3","scenes":[{"scene_id":"decoy"}],"boundary_revisions":[]})"
  );

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [decoy](const offline_sbs::worker_context_t &context,
            std::stop_token,
            const offline_sbs::progress_callback_t &progress) {
      progress({
        .phase = "analysis",
        .processed_frames = 10,
        .total_frames = 10,
        .scene_count = 1,
        .scene_decisions = {{{"scene_id", "managed"}}},
      });
      write_nonempty(
        context.result_directory / "scene-audit.json",
        valid_scene_audit("complete").dump()
      );
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
          // A child-reported path is never used by the audit API.
          {"scene_audit", decoy.generic_string()},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(created.ok) << created.error;
  ASSERT_EQ(
    wait_for_terminal(service, created.job->id).state,
    offline_sbs::job_state_e::complete
  );

  const auto audit = service.scene_audit(created.job->id);
  ASSERT_TRUE(audit.ok) << audit.error;
  ASSERT_EQ(audit.audit["scenes"].size(), 1u);
  EXPECT_EQ(audit.audit["scenes"][0]["scene_id"], 1);
  EXPECT_EQ(
    service.scene_audit("00000000-0000-0000-0000-000000000000").code,
    offline_sbs::error_code_e::not_found
  );

  const auto managed_audit =
    tree.path / "worker" / "jobs" / created.job->id / "result" /
    "scene-audit.json";
  nlohmann::json excessive_boundaries = nlohmann::json::array();
  for (
    std::size_t index = 0;
    index <= offline_sbs::max_serialized_scene_count;
    ++index
  ) {
    excessive_boundaries.push_back({{"index", index}});
  }
  write_nonempty(
    managed_audit,
    nlohmann::json {
      {"schema", 3},
      {"version", "whole-clip-scene-audit-v3"},
      {"status", "complete"},
      {"scenes", nlohmann::json::array({{{"scene_id", "managed"}}})},
      {"boundary_revisions", std::move(excessive_boundaries)},
    }.dump()
  );
  const auto excessive = service.scene_audit(created.job->id);
  EXPECT_FALSE(excessive.ok);
  EXPECT_EQ(excessive.code, offline_sbs::error_code_e::io_error);
  EXPECT_NE(excessive.error.find("identity changed"), std::string::npos);

  fs::remove(managed_audit);
  const auto missing = service.scene_audit(created.job->id);
  EXPECT_FALSE(missing.ok);
  EXPECT_EQ(missing.code, offline_sbs::error_code_e::io_error);
  EXPECT_NE(missing.error.find("unavailable"), std::string::npos);
  service.shutdown();
}

TEST(OfflineSbsJob, RetainsAnAttestedPartialAuditAfterWorkerFailure) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &progress) {
      progress({
        .phase = "analysis",
        .processed_frames = 4,
        .total_frames = 20,
        .scene_count = 1,
      });
      write_nonempty(
        context.result_directory / "scene-audit.json",
        valid_scene_audit("running").dump()
      );
      return offline_sbs::worker_outcome_t {
        .error = "injected failure after one finalized scene",
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(created.ok) << created.error;
  ASSERT_EQ(
    wait_for_terminal(service, created.job->id).state,
    offline_sbs::job_state_e::failed
  );
  const auto audit = service.scene_audit(created.job->id);
  ASSERT_TRUE(audit.ok) << audit.error;
  EXPECT_EQ(audit.audit["availability"], "partial");
  EXPECT_EQ(audit.audit["job_terminal_state"], "failed");
  ASSERT_EQ(audit.audit["scenes"].size(), 1u);
  EXPECT_EQ(audit.audit["scenes"][0]["scene_id"], 1);
  service.shutdown();
}

TEST(OfflineSbsJob, EnforcesOneActiveJobAndCancelsTheWholeWorkerContract) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  std::atomic_bool entered {false};

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [&entered](const offline_sbs::worker_context_t &,
               const std::stop_token stop,
               const offline_sbs::progress_callback_t &) {
      entered = true;
      while (!stop.stop_requested()) {
        std::this_thread::sleep_for(2ms);
      }
      return offline_sbs::worker_outcome_t {.canceled = true};
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto first = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(first.ok);
  for (int iteration = 0; iteration < 200 && !entered; ++iteration) {
    std::this_thread::sleep_for(2ms);
  }
  ASSERT_TRUE(entered);

  const auto second = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  EXPECT_FALSE(second.ok);
  EXPECT_EQ(second.code, offline_sbs::error_code_e::busy);
  ASSERT_TRUE(second.job);
  EXPECT_EQ(second.job->id, first.job->id);

  const auto cancel = service.cancel(first.job->id);
  ASSERT_TRUE(cancel.ok) << cancel.error;
  EXPECT_EQ(cancel.job->state, offline_sbs::job_state_e::canceling);
  const auto terminal = wait_for_terminal(service, first.job->id);
  EXPECT_EQ(terminal.state, offline_sbs::job_state_e::canceled);
  EXPECT_FALSE(service.has_active_job());
  service.shutdown();
}

TEST(OfflineSbsJob, RestartRecoveryNeverResumesAnUnfinishedGpuJob) {
  temporary_tree_t tree;
  const auto config = service_config(tree.path);
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  constexpr auto id = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE";
  const auto job_directory = config.state_root / "jobs" / id;
  fs::create_directories(job_directory);
  const nlohmann::json running {
    {"schema", 1},
    {"id", id},
    {"state", "running"},
    {"operation", "evaluate"},
    {"input_path", input.generic_string()},
    {"output_path", nullptr},
    {"codec", "hevc_nvenc"},
    {"scene_cache_max_bytes", 4ull * 1024ull * 1024ull * 1024ull},
    {"cache_budget_policy", "fail"},
    {"progress", {
      {"phase", "analysis"},
      {"processed_frames", 19},
      {"total_frames", 100},
      {"source_time_seconds", nullptr},
      {"source_duration_seconds", nullptr},
      {"scene_count", 1},
    }},
    {"created_at_unix_ms", 1},
    {"started_at_unix_ms", 2},
    {"ended_at_unix_ms", nullptr},
    {"error", nullptr},
    {"worker_result", nullptr},
  };
  write_nonempty(job_directory / "job.json", running.dump(2));

  std::atomic_bool runner_called {false};
  offline_sbs::job_service_t service {
    config,
    [&runner_called](const offline_sbs::worker_context_t &,
                     std::stop_token,
                     const offline_sbs::progress_callback_t &) {
      runner_called = true;
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto recovered = service.get(id);
  ASSERT_TRUE(recovered.ok) << recovered.error;
  EXPECT_EQ(recovered.job->state, offline_sbs::job_state_e::interrupted);
  EXPECT_EQ(recovered.job->progress.phase, "interrupted");
  EXPECT_NE(recovered.job->error.find("not resumed"), std::string::npos);
  EXPECT_FALSE(runner_called);
  EXPECT_FALSE(service.has_active_job());
  service.shutdown();
}

TEST(OfflineSbsJob, RestartNeverDeletesAWorkerTreeWithoutValidProtectedState) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  config.max_retained_artifact_bytes = 1024;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  constexpr auto id = "11111111-2222-3333-4444-555555555555";
  write_nonempty(
    config.state_root / "jobs" / id / "job.json",
    R"({"schema":1,"id":"wrong"})"
  );
  const auto preserved =
    config.worker_root / "jobs" / id / "result" / "user-data.bin";
  write_nonempty(preserved, std::string(4096, 'u'));

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  EXPECT_TRUE(fs::is_regular_file(preserved));
  EXPECT_FALSE(fs::exists(config.state_root / "jobs" / id));

  const auto blocked = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  EXPECT_FALSE(blocked.ok);
  EXPECT_EQ(blocked.code, offline_sbs::error_code_e::unavailable);
  EXPECT_NE(blocked.error.find("quota"), std::string::npos);
  EXPECT_TRUE(fs::is_regular_file(preserved));
  service.shutdown();
}

TEST(OfflineSbsJob, RejectsOutputTraversalAndBuildsOnlyNativeWorkerCommand) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto rejected = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "..\\outside.mkv",
  });
  EXPECT_FALSE(rejected.ok);
  EXPECT_EQ(rejected.code, offline_sbs::error_code_e::invalid_request);
  for (const auto output : {"CON.preview.mkv", "NUL.archive.mp4", "CONOUT$.mkv"}) {
    const auto device_name = service.create({
      .input_path = input,
      .operation = offline_sbs::operation_e::convert,
      .output_name = output,
    });
    EXPECT_FALSE(device_name.ok) << output;
    EXPECT_EQ(device_name.code, offline_sbs::error_code_e::invalid_request)
      << output;
  }
  const auto reserved_staging_name = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name =
      "movie.sunshine3d-00000000-0000-0000-0000-000000000000.part.mkv",
  });
  EXPECT_FALSE(reserved_staging_name.ok);
  EXPECT_EQ(
    reserved_staging_name.code,
    offline_sbs::error_code_e::invalid_request
  );
#ifdef _WIN32
  const auto remote = service.create({
    .input_path = R"(\\untrusted-host\videos\source.mkv)",
    .operation = offline_sbs::operation_e::evaluate,
  });
  EXPECT_FALSE(remote.ok);
  EXPECT_EQ(remote.code, offline_sbs::error_code_e::invalid_request);
  EXPECT_NE(remote.error.find("local drive"), std::string::npos);
#endif
  service.shutdown();

  const auto command = offline_sbs::build_worker_command_for_test(
    tree.path / "Sunshine 3D" / "sunshine.exe",
    tree.path / "Config Files" / "sunshine.conf",
    tree.path / "Job Files" / "worker-spec.json",
    std::string(64, 'a')
  );
  EXPECT_NE(command.find("--offline-sbs-worker"), std::string::npos);
  EXPECT_NE(command.find(std::string(64, 'a')), std::string::npos);
  EXPECT_EQ(command.find("python"), std::string::npos);
  EXPECT_EQ(command.front(), '"');
  EXPECT_EQ(command.back(), '"');
}

TEST(OfflineSbsJob, BrowsesBoundUserFilesWithFilteringAndStableOrdering) {
  temporary_tree_t tree;
  const auto media = tree.path / "browse-media";
  fs::create_directories(media / "alpha-dir");
  fs::create_directories(media / "Bravo-dir");
  write_nonempty(media / "alpha-video.mkv", "alpha");
  write_nonempty(media / "Bravo-video.mkv", "bravo");

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;

  const auto files = service.browse({
    .path = media,
    .type = offline_sbs::browse_type_e::file,
  });
  ASSERT_TRUE(files.ok) << files.error;
  ASSERT_TRUE(files.path);
  EXPECT_EQ(*files.path, fs::canonical(media));
  EXPECT_FALSE(files.truncated);
  ASSERT_TRUE(files.entries.is_array());
  ASSERT_EQ(files.entries.size(), 4u);
  EXPECT_EQ(files.entries[0].at("name"), "alpha-dir");
  EXPECT_EQ(files.entries[1].at("name"), "Bravo-dir");
  EXPECT_EQ(files.entries[2].at("name"), "alpha-video.mkv");
  EXPECT_EQ(files.entries[3].at("name"), "Bravo-video.mkv");
  EXPECT_EQ(files.entries[0].at("type"), "directory");
  EXPECT_FALSE(files.entries[0].at("selectable").get<bool>());
  EXPECT_EQ(files.entries[2].at("type"), "file");
  EXPECT_TRUE(files.entries[2].at("selectable").get<bool>());

  const auto directories = service.browse({
    .path = media,
    .type = offline_sbs::browse_type_e::directory,
  });
  ASSERT_TRUE(directories.ok) << directories.error;
  ASSERT_EQ(directories.entries.size(), 2u);
  EXPECT_TRUE(directories.entries[0].at("selectable").get<bool>());
  EXPECT_TRUE(directories.entries[1].at("selectable").get<bool>());

  // A manually entered existing file opens its containing directory instead
  // of producing a dead-end "not a directory" response.
  const auto from_file = service.browse({
    .path = media / "Bravo-video.mkv",
    .type = offline_sbs::browse_type_e::file,
  });
  ASSERT_TRUE(from_file.ok) << from_file.error;
  ASSERT_TRUE(from_file.path);
  EXPECT_EQ(*from_file.path, fs::canonical(media));
  ASSERT_EQ(from_file.entries.size(), 4u);

#ifdef _WIN32
  const auto relative = service.browse({
    .path = fs::path {"relative"},
    .type = offline_sbs::browse_type_e::any,
  });
  EXPECT_FALSE(relative.ok);
  EXPECT_EQ(relative.code, offline_sbs::error_code_e::invalid_request);

  const auto drives = service.browse({
    .type = offline_sbs::browse_type_e::file,
  });
  ASSERT_TRUE(drives.ok) << drives.error;
  EXPECT_FALSE(drives.path);
  ASSERT_FALSE(drives.entries.empty());
  for (const auto &drive : drives.entries) {
    EXPECT_EQ(drive.at("type"), "directory");
    EXPECT_FALSE(drive.at("selectable").get<bool>());
  }
#endif
  service.shutdown();
}

TEST(OfflineSbsJob, TruncatesBrowseResponsesToTheSerializedByteBudget) {
  temporary_tree_t tree;
  const auto media = tree.path / "large-browse";
  fs::create_directories(media);
  for (int index = 0; index < 1800; ++index) {
    const auto name =
      std::to_string(index) + "-" + std::string(120, 'x') + ".mkv";
    write_nonempty(media / name, "video");
  }

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;

  const auto reply = service.browse({
    .path = media,
    .type = offline_sbs::browse_type_e::file,
  });
  ASSERT_TRUE(reply.ok) << reply.error;
  EXPECT_TRUE(reply.truncated);
  EXPECT_LT(reply.entries.size(), 1800u);
  EXPECT_LE(reply.json().dump().size(), 512ull * 1024ull);
  service.shutdown();
}

TEST(OfflineSbsJob, NativeWorkerDiagnosticLogHasAStrictRetainedByteCap) {
  const auto limit = offline_sbs::native_worker_log_max_bytes_for_test();
  ASSERT_EQ(limit, 1ull * 1024ull * 1024ull);

  const std::string prefix = "worker-startup-diagnostic\n";
  const std::string ignored_tail = "\nthis-must-not-be-retained";
  const auto result = offline_sbs::bound_worker_log_for_test({
    prefix,
    std::string(limit * 2, 'x'),
    ignored_tail,
  });

  EXPECT_TRUE(result.limit_exceeded);
  EXPECT_FALSE(result.write_failed);
  ASSERT_EQ(result.retained.size(), limit);
  EXPECT_TRUE(result.retained.starts_with(prefix));
  const auto marker = result.retained.find(
    "\n[Sunshine 3D stopped this offline worker"
  );
  ASSERT_NE(marker, std::string::npos);
  EXPECT_NE(
    result.retained.find(
      "diagnostic output exceeded the bounded log limit",
      marker
    ),
    std::string::npos
  );
  EXPECT_EQ(result.retained.find(ignored_tail), std::string::npos);

  const auto exact_boundary = offline_sbs::bound_worker_log_for_test({
    std::string(marker, 'b'),
  });
  EXPECT_FALSE(exact_boundary.limit_exceeded);
  EXPECT_EQ(exact_boundary.retained.size(), marker);

  const auto one_byte_over = offline_sbs::bound_worker_log_for_test({
    std::string(marker, 'b'),
    "z",
  });
  EXPECT_TRUE(one_byte_over.limit_exceeded);
  EXPECT_EQ(one_byte_over.retained.size(), limit);
}

TEST(OfflineSbsJob, NativeWorkerDiagnosticBoundIsIndependentOfPipeChunking) {
  const auto limit = offline_sbs::native_worker_log_max_bytes_for_test();
  std::string source;
  source.reserve(limit + 4096);
  for (std::size_t index = 0; index < limit + 4096; ++index) {
    source.push_back(static_cast<char>('a' + (index % 23)));
  }

  const auto single =
    offline_sbs::bound_worker_log_for_test({source});
  std::vector<std::string> fragmented;
  for (std::size_t offset = 0; offset < source.size();) {
    const auto count = std::min<std::size_t>(
      1 + ((offset * 17) % 65521),
      source.size() - offset
    );
    fragmented.emplace_back(source.substr(offset, count));
    offset += count;
  }
  const auto many =
    offline_sbs::bound_worker_log_for_test(fragmented);

  EXPECT_TRUE(single.limit_exceeded);
  EXPECT_TRUE(many.limit_exceeded);
  EXPECT_FALSE(single.write_failed);
  EXPECT_FALSE(many.write_failed);
  EXPECT_EQ(single.retained.size(), limit);
  EXPECT_EQ(many.retained.size(), limit);
  EXPECT_EQ(many.retained, single.retained);

  const std::string ordinary {"first\0second\n", 13};
  const auto below_limit = offline_sbs::bound_worker_log_for_test({
    ordinary.substr(0, 3),
    ordinary.substr(3),
  });
  EXPECT_FALSE(below_limit.limit_exceeded);
  EXPECT_FALSE(below_limit.write_failed);
  EXPECT_EQ(below_limit.retained, ordinary);
}

TEST(OfflineSbsJob, NativeWorkerDiagnosticPipeDrainsToTheExactBoundAndEof) {
  temporary_tree_t tree;
  const auto output = tree.path / "worker.log";
  const auto limit = offline_sbs::native_worker_log_max_bytes_for_test();
  std::vector<std::string> chunks;
  chunks.emplace_back("pipe-startup\n");
  std::size_t generated = chunks.front().size();
  while (generated <= limit + 128ull * 1024ull) {
    const auto count = std::min<std::size_t>(
      7919 + ((generated * 13) % 49157),
      limit + 128ull * 1024ull - generated + 1
    );
    chunks.emplace_back(count, static_cast<char>('a' + (chunks.size() % 19)));
    generated += count;
  }

  std::string error;
  const auto started = std::chrono::steady_clock::now();
  const auto result = offline_sbs::capture_worker_log_pipe_for_test(
    output,
    chunks,
    error
  );
  const auto elapsed = std::chrono::steady_clock::now() - started;

  EXPECT_TRUE(error.empty()) << error;
  EXPECT_TRUE(result.limit_exceeded);
  EXPECT_FALSE(result.write_failed);
  EXPECT_EQ(result.retained.size(), limit);
  EXPECT_EQ(fs::file_size(output), limit);
  EXPECT_TRUE(result.retained.starts_with("pipe-startup\n"));
  EXPECT_NE(
    result.retained.find("diagnostic output exceeded the bounded log limit"),
    std::string::npos
  );
  EXPECT_LT(elapsed, 2s);
}

TEST(OfflineSbsJob, NativeWorkerDiagnosticPipeAbortDoesNotWaitForOpenWriter) {
  temporary_tree_t tree;
  std::string error;
  const auto started = std::chrono::steady_clock::now();
  const bool aborted =
    offline_sbs::abort_worker_log_pipe_with_open_writer_for_test(
      tree.path / "worker.log",
      error
    );
  const auto elapsed = std::chrono::steady_clock::now() - started;

  EXPECT_TRUE(aborted) << error;
  EXPECT_TRUE(error.empty()) << error;
  EXPECT_LT(elapsed, 1s);
}

#ifdef _WIN32
TEST(OfflineSbsJob, RefusesAWorkspaceBoundToAnotherInteractiveUser) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  const auto active_user = platf::active_user_id();
  ASSERT_TRUE(active_user);
  config.expected_user_id = *active_user + "-different";
  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  EXPECT_FALSE(service.start(error));
  EXPECT_NE(error.find("active user"), std::string::npos);
}

TEST(OfflineSbsJob, PinsInputAgainstMutationUntilTheJobIsTerminal) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  std::atomic_bool entered {false};

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [&entered](const offline_sbs::worker_context_t &,
               const std::stop_token stop,
               const offline_sbs::progress_callback_t &) {
      entered = true;
      while (!stop.stop_requested()) {
        std::this_thread::sleep_for(2ms);
      }
      return offline_sbs::worker_outcome_t {.canceled = true};
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(created.ok) << created.error;
  for (int iteration = 0; iteration < 200 && !entered; ++iteration) {
    std::this_thread::sleep_for(2ms);
  }
  ASSERT_TRUE(entered);

  const HANDLE while_running = CreateFileW(
    input.c_str(),
    GENERIC_WRITE,
    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
    nullptr,
    OPEN_EXISTING,
    FILE_ATTRIBUTE_NORMAL,
    nullptr
  );
  EXPECT_EQ(while_running, INVALID_HANDLE_VALUE);
  if (while_running != INVALID_HANDLE_VALUE) {
    CloseHandle(while_running);
  }

  ASSERT_TRUE(service.cancel(created.job->id).ok);
  EXPECT_EQ(
    wait_for_terminal(service, created.job->id).state,
    offline_sbs::job_state_e::canceled
  );

  const HANDLE after_terminal = CreateFileW(
    input.c_str(),
    GENERIC_WRITE,
    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
    nullptr,
    OPEN_EXISTING,
    FILE_ATTRIBUTE_NORMAL,
    nullptr
  );
  ASSERT_NE(after_terminal, INVALID_HANDLE_VALUE);
  CloseHandle(after_terminal);
  service.shutdown();
}

TEST(OfflineSbsJob, PinsManagedRootsAndRetainedJobDirectoriesByIdentity) {
  temporary_tree_t tree;
  const auto config = service_config(tree.path);
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  EXPECT_FALSE(MoveFileExW(
    config.worker_root.c_str(),
    (tree.path / "worker-swapped").c_str(),
    MOVEFILE_WRITE_THROUGH
  ));
  EXPECT_FALSE(MoveFileExW(
    config.exports_root.c_str(),
    (tree.path / "exports-swapped").c_str(),
    MOVEFILE_WRITE_THROUGH
  ));

  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(created.ok) << created.error;
  ASSERT_EQ(
    wait_for_terminal(service, created.job->id).state,
    offline_sbs::job_state_e::complete
  );
  const auto job_directory =
    config.worker_root / "jobs" / created.job->id;
  EXPECT_FALSE(MoveFileExW(
    job_directory.c_str(),
    (job_directory.wstring() + L".swapped").c_str(),
    MOVEFILE_WRITE_THROUGH
  ));
  EXPECT_FALSE(MoveFileExW(
    (job_directory / "result").c_str(),
    (job_directory / "result-swapped").c_str(),
    MOVEFILE_WRITE_THROUGH
  ));

  service.shutdown();
  EXPECT_TRUE(MoveFileExW(
    job_directory.c_str(),
    (job_directory.wstring() + L".after-shutdown").c_str(),
    MOVEFILE_WRITE_THROUGH
  ));
}
#endif

TEST(OfflineSbsJob, PrunesProtectedStateAndUserArtifactsAfterEveryTerminalJob) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  config.max_retained_jobs = 1;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;

  const auto first = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(first.ok) << first.error;
  ASSERT_EQ(
    wait_for_terminal(service, first.job->id).state,
    offline_sbs::job_state_e::complete
  );

  const auto second = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(second.ok) << second.error;
  ASSERT_EQ(
    wait_for_terminal(service, second.job->id).state,
    offline_sbs::job_state_e::complete
  );

  EXPECT_EQ(service.get(first.job->id).code, offline_sbs::error_code_e::not_found);
  EXPECT_FALSE(fs::exists(config.state_root / "jobs" / first.job->id));
  EXPECT_FALSE(fs::exists(config.worker_root / "jobs" / first.job->id));
  ASSERT_EQ(service.list().size(), 1u);
  EXPECT_EQ(service.list().front().id, second.job->id);
  service.shutdown();
}

TEST(OfflineSbsJob, PrunesOldestTerminalArtifactsToMeetAggregateByteQuota) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  config.max_retained_jobs = 8;
  config.max_retained_artifact_bytes = 160ull * 1024ull;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  const std::string retained_payload(96ull * 1024ull, 'r');

  offline_sbs::job_service_t service {
    config,
    [retained_payload](const offline_sbs::worker_context_t &context,
                       std::stop_token,
                       const offline_sbs::progress_callback_t &) {
      write_nonempty(
        context.result_directory / "retained-probe.json",
        retained_payload
      );
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;

  const auto first = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(first.ok) << first.error;
  ASSERT_EQ(
    wait_for_terminal(service, first.job->id).state,
    offline_sbs::job_state_e::complete
  );
  ASSERT_TRUE(service.get(first.job->id).ok);
  const auto first_state_path =
    config.state_root / "jobs" / first.job->id / "job.json";
  const auto first_state = nlohmann::json::parse(std::ifstream(first_state_path));
  const auto first_retention_sequence =
    first_state.at("retention_sequence").get<std::uint64_t>();

  const auto second = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(second.ok) << second.error;
  ASSERT_EQ(
    wait_for_terminal(service, second.job->id).state,
    offline_sbs::job_state_e::complete
  );
  const auto second_state = nlohmann::json::parse(std::ifstream(
    config.state_root / "jobs" / second.job->id / "job.json"
  ));
  EXPECT_GT(
    second_state.at("retention_sequence").get<std::uint64_t>(),
    first_retention_sequence
  );

  EXPECT_EQ(
    service.get(first.job->id).code,
    offline_sbs::error_code_e::not_found
  );
  EXPECT_FALSE(fs::exists(config.state_root / "jobs" / first.job->id));
  EXPECT_FALSE(fs::exists(config.worker_root / "jobs" / first.job->id));
  ASSERT_TRUE(service.get(second.job->id).ok);
  ASSERT_EQ(service.list().size(), 1u);
  EXPECT_EQ(service.list().front().id, second.job->id);
  const auto capabilities = service.capabilities();
  EXPECT_EQ(
    capabilities["retention"]["max_artifact_bytes"],
    config.max_retained_artifact_bytes
  );
  service.shutdown();
}

TEST(OfflineSbsJob, RetainedUnattestedStagingBlocksFurtherAdmissionAtQuota) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  config.max_retained_artifact_bytes = 512ull * 1024ull;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, {});
      return offline_sbs::worker_outcome_t {
        .error = "injected worker failed before staging attestation",
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto failed = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "retained.mkv",
  });
  ASSERT_TRUE(failed.ok) << failed.error;
  ASSERT_EQ(
    wait_for_terminal(service, failed.job->id).state,
    offline_sbs::job_state_e::failed
  );

  const auto staging =
    input.parent_path() /
    ("retained.sunshine3d-" + failed.job->id + ".part.mkv");
  ASSERT_TRUE(fs::is_regular_file(staging));
  ASSERT_EQ(fs::file_size(staging), 0u);
  ASSERT_TRUE(service.get(failed.job->id).ok);

  const auto blocked = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  EXPECT_FALSE(blocked.ok);
  EXPECT_EQ(blocked.code, offline_sbs::error_code_e::unavailable);
  EXPECT_NE(blocked.error.find("quota"), std::string::npos);
  EXPECT_NE(blocked.error.find(".part"), std::string::npos);
  EXPECT_TRUE(fs::is_regular_file(staging));
  EXPECT_TRUE(service.get(failed.job->id).ok);
  EXPECT_EQ(
    service.capabilities()["retention"]
                          ["unattested_staging_minimum_charge_bytes"],
    1ull * 1024ull * 1024ull
  );
  service.shutdown();
}

TEST(OfflineSbsJob, RetainedUnattestedStagingBlocksTheJobRecordCapacity) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  config.max_retained_jobs = 1;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, {});
      return offline_sbs::worker_outcome_t {
        .error = "injected worker failed before staging attestation",
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto failed = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "retained.mkv",
  });
  ASSERT_TRUE(failed.ok) << failed.error;
  ASSERT_EQ(
    wait_for_terminal(service, failed.job->id).state,
    offline_sbs::job_state_e::failed
  );

  const auto staging =
    input.parent_path() /
    ("retained.sunshine3d-" + failed.job->id + ".part.mkv");
  ASSERT_TRUE(fs::is_regular_file(staging));
  ASSERT_EQ(fs::file_size(staging), 0u);

  const auto blocked = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  EXPECT_FALSE(blocked.ok);
  EXPECT_EQ(blocked.code, offline_sbs::error_code_e::unavailable);
  EXPECT_NE(blocked.error.find("1-record limit"), std::string::npos);
  EXPECT_NE(blocked.error.find(".part"), std::string::npos);
  EXPECT_TRUE(fs::is_regular_file(staging));
  ASSERT_EQ(service.list().size(), 1u);
  EXPECT_EQ(service.list().front().id, failed.job->id);
  service.shutdown();
}

TEST(OfflineSbsJob, PublishedFilenameThatOnlyResemblesStagingDoesNotUseQuota) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  config.max_retained_artifact_bytes = 64ull * 1024ull;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  write_nonempty(
    config.exports_root / "movie.sunshine3d-note.part.mkv",
    std::string(256ull * 1024ull, 'p')
  );

  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto admitted = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(admitted.ok) << admitted.error;
  EXPECT_EQ(
    wait_for_terminal(service, admitted.job->id).state,
    offline_sbs::job_state_e::complete
  );
  EXPECT_TRUE(fs::is_regular_file(
    config.exports_root / "movie.sunshine3d-note.part.mkv"
  ));
  service.shutdown();
}

TEST(OfflineSbsJob, RejectsOverlappingManagedRoots) {
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  config.exports_root = config.worker_root;
  offline_sbs::job_service_t service {
    config,
    [](const offline_sbs::worker_context_t &,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {.completed = true};
    }
  };
  std::string error;
  EXPECT_FALSE(service.start(error));
  EXPECT_NE(error.find("disjoint"), std::string::npos);
}

#ifdef _WIN32
TEST(OfflineSbsJob, RejectsRemoteOrDeviceManagedRootsBeforeFilesystemAccess) {
  temporary_tree_t tree;
  for (int root_index = 0; root_index < 3; ++root_index) {
    auto config = service_config(tree.path / std::to_string(root_index));
    const fs::path unsupported {
      LR"(\\offline-invalid.example\share\Sunshine 3D\offline-sbs)"
    };
    if (root_index == 0) {
      config.state_root = unsupported / "state";
    } else if (root_index == 1) {
      config.worker_root = unsupported / "worker";
    } else {
      config.exports_root = unsupported / "exports";
    }
    offline_sbs::job_service_t service {
      config,
      [](const offline_sbs::worker_context_t &,
         std::stop_token,
         const offline_sbs::progress_callback_t &) {
        return offline_sbs::worker_outcome_t {.completed = true};
      }
    };
    std::string error;
    EXPECT_FALSE(service.start(error));
    EXPECT_NE(error.find("local drive"), std::string::npos) << error;
  }
}
#endif

TEST(OfflineSbsJob, RefusesAdmissionWhileLiveStreamingOwnsTheGpu) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &context,
       std::stop_token,
       const offline_sbs::progress_callback_t &) {
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"scene_count", 1},
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;

  auto live =
    gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
  ASSERT_TRUE(live);
  const auto blocked = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  EXPECT_FALSE(blocked.ok);
  EXPECT_EQ(blocked.code, offline_sbs::error_code_e::busy);
  EXPECT_NE(blocked.error.find("live streaming"), std::string::npos);

  live.reset();
  const auto admitted = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(admitted.ok) << admitted.error;
  EXPECT_EQ(
    wait_for_terminal(service, admitted.job->id).state,
    offline_sbs::job_state_e::complete
  );
  service.shutdown();
}
