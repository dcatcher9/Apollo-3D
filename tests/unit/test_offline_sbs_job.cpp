#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "src/crypto.h"
#include "src/gpu_workload_arbiter.h"
#include "src/offline_sbs_contract.h"
#include "src/offline_sbs_job.h"
#include "src/offline_sbs_progress_test.h"
#include "src/offline_sbs_wire_contract.h"
#include "src/platform/common.h"
#include "src/utility.h"

#ifdef _WIN32
  #include <windows.h>
#endif

using namespace std::chrono_literals;

namespace {
  namespace fs = std::filesystem;

  class temporary_tree_t {
  public:
    temporary_tree_t() {
      temp_base_ = fs::weakly_canonical(fs::temp_directory_path());
      path = temp_base_ /
             ("sunshine3d-offline-job-" +
              std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()
              ));
      fs::create_directories(path);
    }

    ~temporary_tree_t() {
      std::error_code ignored;
      const auto resolved_base = fs::weakly_canonical(temp_base_, ignored);
      if (ignored) {
        ADD_FAILURE() << "Cannot resolve the owned temporary base: " << ignored.message();
        return;
      }
      const auto resolved_target = fs::weakly_canonical(path, ignored);
      if (ignored || !resolved_target.is_absolute() || resolved_target.parent_path() != resolved_base || !resolved_target.filename().string().starts_with("sunshine3d-offline-job-")) {
        ADD_FAILURE() << "Refusing recursive cleanup outside the owned temporary child: " << path;
        return;
      }
      fs::remove_all(resolved_target, ignored);
    }

    fs::path path;

  private:
    fs::path temp_base_;
  };

  void write_nonempty(const fs::path &path, const std::string_view contents = "x") {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  }

  std::string read_bytes(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("cannot read test artifact: " + path.generic_string());
    }
    return {std::istreambuf_iterator<char> {input}, std::istreambuf_iterator<char> {}};
  }

  nlohmann::json read_json(const fs::path &path) {
    return nlohmann::json::parse(read_bytes(path));
  }

  void expect_persisted_snapshot(
    const fs::path &state_path,
    const offline_sbs::job_snapshot_t &snapshot
  ) {
    auto persisted = read_json(state_path);
    ASSERT_TRUE(persisted.at("retention_sequence").is_number_unsigned());
    EXPECT_GT(persisted.at("retention_sequence").get<std::uint64_t>(), 0u);
    ASSERT_TRUE(persisted.at("worker").is_object());
    // Only the durable record adds retention order and private worker bindings.
    // Every public field, including the compact publication result, must agree.
    persisted.erase("retention_sequence");
    persisted.erase("worker");
    EXPECT_EQ(persisted, snapshot.json());
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

  void expect_owned_job_list_summary(
    offline_sbs::job_service_t &service,
    const offline_sbs::job_snapshot_t &terminal
  ) {
    const auto original_detail = terminal.json();
    {
      auto summaries = service.list();
      ASSERT_EQ(summaries.size(), 1u);
      EXPECT_TRUE(summaries.front().worker_result.is_null());
      auto detail = service.get(terminal.id);
      ASSERT_TRUE(detail.ok);
      ASSERT_TRUE(detail.job);
      EXPECT_EQ(detail.job->json(), original_detail);
      auto summary_metadata = summaries.front().json();
      auto detail_metadata = detail.job->json();
      summary_metadata.erase("worker_result");
      detail_metadata.erase("worker_result");
      EXPECT_EQ(summary_metadata, detail_metadata);

      summaries.front().id.assign(256, 'i');
      summaries.front().input_path = "caller-edited-input";
      summaries.front().output_path = fs::path {"caller-edited-output"};
      summaries.front().codec.assign(256, 'c');
      summaries.front().progress.phase.assign(256, 'p');
      summaries.front().error.assign(256, 'e');
      summaries.front().worker_result = {{"caller", "summary-only"}};
      detail.job->worker_result["source"]["width"] = 1;
      detail.job->worker_result["cache"]["hard_cap_bytes"] = 1;
    }
    // Both returned containers have been mutated and destroyed. Neither operation can alter
    // the canonical metadata or the compact nested result retained for the selected job.
    const auto detail = service.get(terminal.id);
    ASSERT_TRUE(detail.ok);
    ASSERT_TRUE(detail.job);
    EXPECT_EQ(detail.job->json(), original_detail);
    const auto summaries = service.list();
    ASSERT_EQ(summaries.size(), 1u);
    EXPECT_TRUE(summaries.front().worker_result.is_null());
    auto summary_metadata = summaries.front().json();
    auto detail_metadata = detail.job->json();
    summary_metadata.erase("worker_result");
    detail_metadata.erase("worker_result");
    EXPECT_EQ(summary_metadata, detail_metadata);
  }
}  // namespace

TEST(OfflineSbsJob, CompletesThroughStagingAndPersistsAtomicTerminalState) {
  temporary_tree_t tree;
  const auto input = tree.path / "media" / "source.mkv";
  write_nonempty(input, "source");

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &context, std::stop_token, const offline_sbs::progress_callback_t &progress) {
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
        .source_time_seconds = 0.5,
        .source_duration_seconds = 0.5,
      });
      write_nonempty(*context.staging_output, "encoded-video");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"source", {
                       {"width", 1920},
                       {"height", 1080},
                       {"frame_count", 12},
                       {"duration_seconds", 0.5},
                       {"variable_frame_rate", false},
                       {"color_mode", "sdr"},
                       {"contract", "source-contract.json"},
                     }},
          {"cache", {{"hard_cap_bytes", 4096}, {"peak_live_raster_bytes", 1024}}},
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
  EXPECT_EQ(terminal.worker_result["status"], "complete");
  EXPECT_FALSE(terminal.worker_result.contains("scene_count"));
  EXPECT_FALSE(terminal.worker_result.contains("scene_decisions_contract"));
  EXPECT_FALSE(terminal.worker_result.contains("scene_audit_sha256"));
  EXPECT_EQ(terminal.worker_result.at("source").at("width"), 1920);
  EXPECT_EQ(
    terminal.worker_result.at("cache"),
    (nlohmann::json {{"hard_cap_bytes", 4096}})
  );
  expect_owned_job_list_summary(service, terminal);

  const auto state_path =
    tree.path / "state" / "jobs" / terminal.id / "job.json";
  ASSERT_TRUE(fs::is_regular_file(state_path));
  EXPECT_FALSE(fs::exists(state_path.wstring() + L".part"));
  const auto persisted = nlohmann::json::parse(std::ifstream(state_path));
  expect_persisted_snapshot(state_path, terminal);
  const auto publication_path = tree.path / "worker" / "jobs" / terminal.id / "result" / "publication-contract.json";
  EXPECT_EQ(read_json(publication_path), terminal.worker_result);
  EXPECT_EQ(terminal.worker_result.at("output"), terminal.output_path->generic_string());
  EXPECT_EQ(terminal.worker_result.at("published"), true);
  EXPECT_EQ(terminal.worker_result.at("staging_cleanup_pending"), false);
#ifdef _WIN32
  const auto final_identity = windows_file_identity(*terminal.output_path);
  EXPECT_EQ(terminal.worker_result.at("publish_identity"), final_identity);
#endif
  EXPECT_EQ(read_bytes(*terminal.output_path), "encoded-video");
  EXPECT_EQ(persisted["state"], "complete");
  EXPECT_EQ(persisted["output_location"], "input-directory");
  EXPECT_EQ(persisted["progress"]["processed_frames"], 12);
  EXPECT_FALSE(persisted["progress"].contains("scene_count"));
  EXPECT_FALSE(persisted["progress"].contains("current_scene"));
  EXPECT_FALSE(persisted.contains("scene_decisions"));
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

  const auto worker_path = tree.path / "worker" / "jobs" / terminal.id;
  ASSERT_TRUE(fs::is_directory(worker_path));
  const auto cleared = service.clear(terminal.id);
  ASSERT_TRUE(cleared.ok) << cleared.error;
  ASSERT_TRUE(cleared.job);
  EXPECT_EQ(cleared.job->id, terminal.id);
  EXPECT_EQ(
    service.get(terminal.id).code,
    offline_sbs::error_code_e::not_found
  );
  EXPECT_TRUE(service.list().empty());
  EXPECT_FALSE(fs::exists(state_path.parent_path()));
  EXPECT_FALSE(fs::exists(worker_path));
  EXPECT_TRUE(fs::is_regular_file(*terminal.output_path));
  EXPECT_EQ(fs::file_size(*terminal.output_path), 13u);
  EXPECT_EQ(read_bytes(*terminal.output_path), "encoded-video");
#ifdef _WIN32
  EXPECT_EQ(windows_file_identity(*terminal.output_path), final_identity);
#endif
  EXPECT_EQ(
    service.clear(terminal.id).code,
    offline_sbs::error_code_e::not_found
  );
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

TEST(OfflineSbsJob, PublishesProgressDuringDirectConversion) {
  temporary_tree_t tree;
  const auto input = tree.path / "media" / "source.mkv";
  write_nonempty(input, "source");
  std::atomic_bool progress_published {false};
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
      });
      progress({
        .phase = "convert",
        .processed_frames = 185,
        .total_frames = 185,
      });
      progress_published = true;
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
  while (!progress_published &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_TRUE(progress_published);
  const auto active = service.get(created.job->id);
  ASSERT_TRUE(active.ok);
  ASSERT_TRUE(active.job);
  EXPECT_EQ(active.job->state, offline_sbs::job_state_e::running);
  EXPECT_EQ(active.job->progress.phase, "convert");
  EXPECT_EQ(active.job->progress.processed_frames, 185u);
  ASSERT_TRUE(active.job->progress.total_frames);
  EXPECT_EQ(*active.job->progress.total_frames, 185u);

  release_worker = true;
  const auto terminal = wait_for_terminal(service, created.job->id);
  EXPECT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  EXPECT_EQ(terminal.progress.processed_frames, 185u);
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
  const auto transient_read_bound = runner.find(
    "constexpr int transient_progress_read_attempts = 50;"
  );
  const auto progress_read = runner.find(
    "value = read_json_contract(",
    transient_read_bound
  );
  const auto successful_read = runner.find(
    "if (value || !acquisition_failure)",
    progress_read
  );
  const auto retry_guard = runner.find(
    "if (attempt + 1 < transient_progress_read_attempts)",
    successful_read
  );
  const auto retry_wait = runner.find(
    "std::this_thread::sleep_for(std::chrono::milliseconds(2))",
    retry_guard
  );
  ASSERT_NE(transient_read_bound, std::string::npos);
  ASSERT_NE(progress_read, std::string::npos);
  ASSERT_NE(successful_read, std::string::npos);
  ASSERT_NE(retry_guard, std::string::npos);
  ASSERT_NE(retry_wait, std::string::npos);
  EXPECT_NE(
    runner.find("&acquisition_failure", progress_read),
    std::string::npos
  );
  EXPECT_LT(progress_read, successful_read);
  EXPECT_LT(successful_read, retry_guard)
    << "A normal read or immutable parse failure must not take the acquisition retry";
  EXPECT_LT(retry_guard, retry_wait);
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
  expect_persisted_snapshot(tree.path / "state" / "jobs" / terminal.id / "job.json", terminal);
  EXPECT_FALSE(fs::exists(tree.path / "worker" / "jobs" / terminal.id / "result" / "publication-contract.json"));
  EXPECT_TRUE(terminal.worker_result.is_null());
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
  nlohmann::json late_identity;
  nlohmann::json staged_identity;

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [&](const offline_sbs::worker_context_t &context, std::stop_token, const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "encoded-video");
      write_nonempty(*context.final_output, "late-user-file");
      late_identity = windows_file_identity(*context.final_output);
      staged_identity = windows_file_identity(*context.staging_output);
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
          {"source", {
                       {"width", 1920},
                       {"height", 1080},
                       {"frame_count", 12},
                       {"duration_seconds", 0.5},
                       {"variable_frame_rate", false},
                       {"color_mode", "sdr"},
                       {"contract", "source-contract.json"},
                     }},
          {"cache", {{"hard_cap_bytes", 4096}, {"peak_live_raster_bytes", 1024}}},
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
  EXPECT_EQ(terminal.worker_result.at("source").at("width"), 1920);
  EXPECT_EQ(terminal.worker_result.at("cache").at("hard_cap_bytes"), 4096);
  expect_persisted_snapshot(tree.path / "state" / "jobs" / terminal.id / "job.json", terminal);
  EXPECT_FALSE(fs::exists(tree.path / "worker" / "jobs" / terminal.id / "result" / "publication-contract.json"));
  EXPECT_EQ(terminal.worker_result.at("publish_identity"), staged_identity);
  EXPECT_FALSE(terminal.worker_result.contains("published"));
  EXPECT_FALSE(terminal.worker_result.contains("staging_cleanup_pending"));
  EXPECT_EQ(windows_file_identity(input.parent_path() / "collision.mkv"), late_identity);
  const auto staging = input.parent_path() / ("collision.sunshine3d-" + terminal.id + ".part.mkv");
  EXPECT_EQ(windows_file_identity(staging), staged_identity);
  EXPECT_EQ(read_bytes(staging), "encoded-video");
  expect_owned_job_list_summary(service, terminal);
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
  nlohmann::json attested_identity;
  nlohmann::json replacement_identity;

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [&](const offline_sbs::worker_context_t &context, std::stop_token, const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "attested-video");
      const auto identity = windows_file_identity(*context.staging_output);
      attested_identity = identity;
      retained_original = context.staging_output->wstring() + L".original";
      fs::rename(*context.staging_output, retained_original);
      write_nonempty(*context.staging_output, "replacement-data");
      replacement_identity = windows_file_identity(*context.staging_output);
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {
          {"schema", 1},
          {"job_id", context.job_id},
          {"status", "complete"},
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
  expect_persisted_snapshot(tree.path / "state" / "jobs" / terminal.id / "job.json", terminal);
  EXPECT_TRUE(terminal.worker_result.is_null());
  EXPECT_FALSE(fs::exists(tree.path / "worker" / "jobs" / terminal.id / "result" / "publication-contract.json"));
  EXPECT_FALSE(fs::exists(input.parent_path() / "swapped.mkv"));
  EXPECT_TRUE(fs::is_regular_file(retained_original));
  const auto current_staging =
    input.parent_path() /
    ("swapped.sunshine3d-" + created.job->id + ".part.mkv");
  EXPECT_TRUE(fs::is_regular_file(current_staging));
  EXPECT_EQ(read_bytes(retained_original), "attested-video");
  EXPECT_EQ(read_bytes(current_staging), "replacement-data");
  EXPECT_EQ(windows_file_identity(retained_original), attested_identity);
  EXPECT_EQ(windows_file_identity(current_staging), replacement_identity);
  service.shutdown();
}

TEST(OfflineSbsJob, PublicationContractWriteFailureDoesNotUndoPublishedOutput) {
  temporary_tree_t tree;
  const auto config = service_config(tree.path);
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  nlohmann::json staged_identity;

  offline_sbs::job_service_t service {
    config,
    [&](const offline_sbs::worker_context_t &context, std::stop_token, const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "encoded-video");
      staged_identity = windows_file_identity(*context.staging_output);
      // A real nonempty destination directory makes atomic artifact replacement fail.
      write_nonempty(context.result_directory / "publication-contract.json" / "owned-sentinel", "block artifact replacement");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {{"schema", 1}, {"job_id", context.job_id}, {"status", "complete"}},
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "artifact-warning.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  const auto terminal = wait_for_terminal(service, created.job->id);
  ASSERT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  EXPECT_TRUE(terminal.error.empty());
  ASSERT_TRUE(terminal.output_path);
  EXPECT_EQ(terminal.worker_result.at("published"), true);
  EXPECT_EQ(terminal.worker_result.at("staging_cleanup_pending"), false);
  EXPECT_EQ(terminal.worker_result.at("publish_identity"), staged_identity);
  EXPECT_EQ(windows_file_identity(*terminal.output_path), staged_identity);
  EXPECT_EQ(read_bytes(*terminal.output_path), "encoded-video");
  const auto state_path = config.state_root / "jobs" / terminal.id / "job.json";
  expect_persisted_snapshot(state_path, terminal);
  const auto publication_path = config.worker_root / "jobs" / terminal.id / "result" / "publication-contract.json";
  EXPECT_TRUE(fs::is_directory(publication_path));
  EXPECT_EQ(read_bytes(publication_path / "owned-sentinel"), "block artifact replacement");
  EXPECT_FALSE(fs::exists(publication_path.wstring() + L".part"));
  EXPECT_FALSE(fs::exists(input.parent_path() / ("artifact-warning.sunshine3d-" + terminal.id + ".part.mkv")));
  service.shutdown();
  EXPECT_TRUE(fs::remove(publication_path / "owned-sentinel"));
  EXPECT_TRUE(fs::remove(publication_path));
}

TEST(OfflineSbsJob, PrePublishStateWriteFailurePreventsPublication) {
  temporary_tree_t tree;
  const auto config = service_config(tree.path);
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  nlohmann::json staged_identity;
  std::string prior_running_state;

  offline_sbs::job_service_t service {
    config,
    [&](const offline_sbs::worker_context_t &context, std::stop_token, const offline_sbs::progress_callback_t &) {
      write_nonempty(*context.staging_output, "encoded-video");
      staged_identity = windows_file_identity(*context.staging_output);
      const auto state_path = config.state_root / "jobs" / context.job_id / "job.json";
      prior_running_state = read_bytes(state_path);
      // Atomic state writes cannot remove this nonempty temporary directory or open it as a file.
      write_nonempty(fs::path {state_path.wstring() + L".part"} / "owned-sentinel", "block protected state write");
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {{"schema", 1}, {"job_id", context.job_id}, {"status", "complete"}},
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "not-published.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  const auto terminal = wait_for_terminal(service, created.job->id);
  ASSERT_EQ(terminal.state, offline_sbs::job_state_e::failed);
  EXPECT_NE(terminal.error.find("cannot persist pre-publish state"), std::string::npos);
  EXPECT_EQ(terminal.worker_result.at("publish_identity"), staged_identity);
  EXPECT_FALSE(terminal.worker_result.contains("published"));
  EXPECT_FALSE(terminal.worker_result.contains("staging_cleanup_pending"));
  ASSERT_TRUE(terminal.output_path);
  EXPECT_FALSE(fs::exists(*terminal.output_path));
  const auto staging_path = input.parent_path() / ("not-published.sunshine3d-" + terminal.id + ".part.mkv");
  EXPECT_EQ(windows_file_identity(staging_path), staged_identity);
  EXPECT_EQ(read_bytes(staging_path), "encoded-video");
  EXPECT_FALSE(fs::exists(config.worker_root / "jobs" / terminal.id / "result" / "publication-contract.json"));
  const auto state_path = config.state_root / "jobs" / terminal.id / "job.json";
  EXPECT_EQ(read_bytes(state_path), prior_running_state);
  const auto durable = nlohmann::json::parse(prior_running_state);
  EXPECT_EQ(durable.at("state"), "running");
  EXPECT_TRUE(durable.at("worker_result").is_null());
  // The terminal write also fails, leaving the last successfully persisted running state.
  const auto blocker = fs::path {state_path.wstring() + L".part"};
  EXPECT_EQ(read_bytes(blocker / "owned-sentinel"), "block protected state write");
  service.shutdown();
  EXPECT_TRUE(fs::remove(blocker / "owned-sentinel"));
  EXPECT_TRUE(fs::remove(blocker));
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
  const auto final_identity = windows_file_identity(final_output);
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
  // Publishing recovery replaces the cleanup marker without interpreting its old type.
  persisted["worker_result"]["staging_cleanup_pending"] = {{"old", "malformed"}};
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
  expect_persisted_snapshot(state_path, *snapshot.job);
  const auto publication_path = config.worker_root / "jobs" / created.job->id / "result" / "publication-contract.json";
  EXPECT_EQ(read_json(publication_path), snapshot.job->worker_result);
  EXPECT_EQ(snapshot.job->worker_result.at("publish_identity"), final_identity);
  EXPECT_EQ(snapshot.job->worker_result.at("published"), true);
  EXPECT_EQ(snapshot.job->worker_result.at("output"), final_output.generic_string());
  EXPECT_EQ(read_json(state_path).at("retention_sequence"), persisted.at("retention_sequence"));
  EXPECT_EQ(windows_file_identity(final_output), final_identity);
  EXPECT_EQ(read_bytes(final_output), "encoded-video");
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
  const auto final_identity = windows_file_identity(legacy_final);
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
  expect_persisted_snapshot(state_path, *snapshot.job);
  const auto publication_path = config.worker_root / "jobs" / created.job->id / "result" / "publication-contract.json";
  EXPECT_EQ(read_json(publication_path), snapshot.job->worker_result);
  EXPECT_EQ(snapshot.job->worker_result.at("output"), legacy_final.generic_string());
  EXPECT_EQ(snapshot.job->worker_result.at("publish_identity"), final_identity);
  EXPECT_EQ(snapshot.job->worker_result.at("published"), true);
  EXPECT_EQ(snapshot.job->worker_result.at("staging_cleanup_pending"), false);
  EXPECT_EQ(read_json(state_path).at("retention_sequence"), persisted.at("retention_sequence"));
  EXPECT_EQ(windows_file_identity(legacy_final), final_identity);
  EXPECT_EQ(read_bytes(legacy_final), "encoded-video");
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
  ASSERT_TRUE(terminal.output_path);
  const auto final_identity = windows_file_identity(*terminal.output_path);
  EXPECT_EQ(windows_file_identity(staging_output), final_identity);
  EXPECT_EQ(terminal.worker_result.at("publish_identity"), final_identity);
  EXPECT_EQ(read_bytes(staging_output), "encoded-video");
  const auto state_path = config.state_root / "jobs" / terminal.id / "job.json";
  const auto publication_path = config.worker_root / "jobs" / terminal.id / "result" / "publication-contract.json";
  expect_persisted_snapshot(state_path, terminal);
  const auto original_state = read_json(state_path);
  const auto original_publication = read_bytes(publication_path);
  EXPECT_EQ(nlohmann::json::parse(original_publication), terminal.worker_result);

  const auto quota_blocked = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  EXPECT_FALSE(quota_blocked.ok);
  EXPECT_EQ(quota_blocked.code, offline_sbs::error_code_e::unavailable);
  EXPECT_NE(quota_blocked.error.find("quota"), std::string::npos);
  service.shutdown();

  auto state_without_cleanup_marker = original_state;
  state_without_cleanup_marker["worker_result"].erase("staging_cleanup_pending");
  const auto missing_marker_bytes = state_without_cleanup_marker.dump(2);
  write_nonempty(state_path, missing_marker_bytes);
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
  // Missing defaults to false: successful cleanup need not rewrite that absent durable field.
  EXPECT_EQ(read_bytes(state_path), missing_marker_bytes);
  EXPECT_FALSE(read_json(state_path).at("worker_result").contains("staging_cleanup_pending"));
  EXPECT_EQ(windows_file_identity(*terminal.output_path), final_identity);
  EXPECT_EQ(read_bytes(*terminal.output_path), "encoded-video");
  auto expected_result = terminal.worker_result;
  expected_result["staging_cleanup_pending"] = false;
  EXPECT_EQ(snapshot.job->worker_result, expected_result);
  // Completed-job cleanup updates the durable job, not its earlier publication artifact.
  EXPECT_EQ(read_bytes(publication_path), original_publication);

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

TEST(OfflineSbsJob, ClearRetiresAttestedPendingStagingAndKeepsFinalOutput) {
#ifndef _WIN32
  GTEST_SKIP() << "exact staging-link retirement is Windows-specific";
#else
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
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "clear-pending.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  const auto terminal = wait_for_terminal(service, created.job->id);
  ASSERT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  ASSERT_TRUE(terminal.output_path);
  ASSERT_TRUE(
    terminal.worker_result.value("staging_cleanup_pending", false)
  );
  const auto staging_output =
    input.parent_path() /
    (
      "clear-pending.sunshine3d-" + created.job->id + ".part.mkv"
    );
  ASSERT_TRUE(fs::is_regular_file(staging_output));
  ASSERT_TRUE(fs::is_regular_file(*terminal.output_path));
  const auto final_identity = windows_file_identity(*terminal.output_path);
  EXPECT_EQ(windows_file_identity(staging_output), final_identity);
  const auto state_path = config.state_root / "jobs" / terminal.id / "job.json";
  const auto publication_path = config.worker_root / "jobs" / terminal.id / "result" / "publication-contract.json";
  expect_persisted_snapshot(state_path, terminal);
  EXPECT_EQ(read_json(publication_path), terminal.worker_result);

  const auto cleared = service.clear(created.job->id);
  ASSERT_TRUE(cleared.ok) << cleared.error;
  ASSERT_TRUE(cleared.job);
  EXPECT_EQ(cleared.job->worker_result.at("staging_cleanup_pending"), false);
  EXPECT_FALSE(fs::exists(staging_output));
  EXPECT_TRUE(fs::is_regular_file(*terminal.output_path));
  EXPECT_EQ(fs::file_size(*terminal.output_path), 13u);
  EXPECT_EQ(windows_file_identity(*terminal.output_path), final_identity);
  EXPECT_EQ(read_bytes(*terminal.output_path), "encoded-video");
  EXPECT_FALSE(fs::exists(state_path.parent_path()));
  EXPECT_FALSE(fs::exists(publication_path.parent_path()));
  EXPECT_EQ(
    service.get(created.job->id).code,
    offline_sbs::error_code_e::not_found
  );
  service.shutdown();
#endif
}

TEST(OfflineSbsJob, ClearAcceptsStalePendingMarkerWhenBothOutputsAreGone) {
#ifndef _WIN32
  GTEST_SKIP() << "forced staging disposition failure is Windows-specific";
#else
  temporary_tree_t tree;
  auto config = service_config(tree.path);
  config.test_force_staging_disposition_failure = true;
  const auto media_directory = tree.path / "former-media";
  ASSERT_TRUE(fs::create_directories(media_directory));
  const auto input = media_directory / "source.mkv";
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
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "clear-missing.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  const auto terminal = wait_for_terminal(service, created.job->id);
  ASSERT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  ASSERT_TRUE(terminal.output_path);
  ASSERT_TRUE(
    terminal.worker_result.value("staging_cleanup_pending", false)
  );
  const auto staging_output =
    input.parent_path() /
    (
      "clear-missing.sunshine3d-" + created.job->id + ".part.mkv"
  );
  ASSERT_TRUE(fs::remove(staging_output));
  ASSERT_TRUE(fs::remove(*terminal.output_path));
  ASSERT_TRUE(fs::remove(input));
  ASSERT_TRUE(fs::remove(media_directory));

  const auto cleared = service.clear(created.job->id);
  ASSERT_TRUE(cleared.ok) << cleared.error;
  ASSERT_TRUE(cleared.job);
  EXPECT_EQ(cleared.job->worker_result.at("staging_cleanup_pending"), false);
  EXPECT_FALSE(fs::exists(config.state_root / "jobs" / terminal.id));
  EXPECT_FALSE(fs::exists(config.worker_root / "jobs" / terminal.id));
  EXPECT_EQ(
    service.get(created.job->id).code,
    offline_sbs::error_code_e::not_found
  );
  service.shutdown();
#endif
}

TEST(OfflineSbsJob, ClearPreservesPendingStagingWhenFinalOutputIsGone) {
#ifndef _WIN32
  GTEST_SKIP() << "forced staging disposition failure is Windows-specific";
#else
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
        },
      };
    }
  };
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::convert,
    .output_name = "keep-last-copy.mkv",
  });
  ASSERT_TRUE(created.ok) << created.error;
  const auto terminal = wait_for_terminal(service, created.job->id);
  ASSERT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  ASSERT_TRUE(terminal.output_path);
  const auto staging_output =
    input.parent_path() /
    (
      "keep-last-copy.sunshine3d-" + created.job->id + ".part.mkv"
    );
  ASSERT_TRUE(fs::is_regular_file(staging_output));
  const auto staged_identity = windows_file_identity(staging_output);
  const auto state_path = config.state_root / "jobs" / terminal.id / "job.json";
  const auto publication_path = config.worker_root / "jobs" / terminal.id / "result" / "publication-contract.json";
  expect_persisted_snapshot(state_path, terminal);
  const auto original_state = read_bytes(state_path);
  const auto original_publication = read_bytes(publication_path);
  ASSERT_TRUE(fs::remove(*terminal.output_path));

  const auto cleared = service.clear(created.job->id);
  EXPECT_FALSE(cleared.ok);
  EXPECT_EQ(cleared.code, offline_sbs::error_code_e::io_error);
  EXPECT_NE(cleared.error.find("last intact copy"), std::string::npos);
  EXPECT_TRUE(fs::is_regular_file(staging_output));
  EXPECT_TRUE(service.get(created.job->id).ok);
  ASSERT_TRUE(cleared.job);
  EXPECT_EQ(cleared.job->json(), terminal.json());
  EXPECT_EQ(read_bytes(state_path), original_state);
  EXPECT_EQ(read_bytes(publication_path), original_publication);
  EXPECT_EQ(windows_file_identity(staging_output), staged_identity);
  EXPECT_EQ(read_bytes(staging_output), "encoded-video");
  service.shutdown();
#endif
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
  const auto terminal = wait_for_terminal(service, created.job->id);
  ASSERT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  ASSERT_TRUE(terminal.output_path);
  const auto final_identity = windows_file_identity(*terminal.output_path);
  const auto state_path = config.state_root / "jobs" / terminal.id / "job.json";
  const auto publication_path = config.worker_root / "jobs" / terminal.id / "result" / "publication-contract.json";
  expect_persisted_snapshot(state_path, terminal);
  const auto original_state = read_bytes(state_path);
  const auto original_publication = read_bytes(publication_path);
  service.shutdown();

  const auto staging_output =
    input.parent_path() /
    (
      "replacement.sunshine3d-" + created.job->id + ".part.mkv"
    );
  ASSERT_TRUE(fs::remove(staging_output));
  write_nonempty(staging_output, "replacement-user-data");
  const auto replacement_identity = windows_file_identity(staging_output);
  EXPECT_NE(replacement_identity, final_identity);

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
  replacement.close();
  const auto clear = recovered.clear(created.job->id);
  EXPECT_FALSE(clear.ok);
  EXPECT_EQ(clear.code, offline_sbs::error_code_e::io_error);
  EXPECT_NE(clear.error.find("identity changed"), std::string::npos)
    << clear.error;
  EXPECT_TRUE(fs::is_regular_file(staging_output));
  EXPECT_TRUE(recovered.get(created.job->id).ok);
  EXPECT_EQ(snapshot.job->json(), terminal.json());
  EXPECT_EQ(read_bytes(state_path), original_state);
  EXPECT_EQ(read_bytes(publication_path), original_publication);
  EXPECT_EQ(windows_file_identity(staging_output), replacement_identity);
  EXPECT_EQ(windows_file_identity(*terminal.output_path), final_identity);
  EXPECT_EQ(read_bytes(*terminal.output_path), "encoded-video");
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
  const auto final_identity = windows_file_identity(final_output);
  const auto publication_path = config.worker_root / "jobs" / created.job->id / "result" / "publication-contract.json";
  const auto original_publication = read_bytes(publication_path);
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
  expect_persisted_snapshot(state_path, *snapshot.job);
  EXPECT_EQ(read_bytes(publication_path), original_publication);
  EXPECT_EQ(windows_file_identity(staging_output), final_identity);
  EXPECT_EQ(read_bytes(staging_output), "encoded-video");
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
  const auto outside_identity = windows_file_identity(outside);
  const auto final_output = input.parent_path() / "redirected.mkv";
  const auto final_identity = windows_file_identity(final_output);
  const auto publication_path = config.worker_root / "jobs" / created.job->id / "result" / "publication-contract.json";
  const auto original_publication = read_bytes(publication_path);
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
  EXPECT_EQ(windows_file_identity(outside), outside_identity);
  EXPECT_EQ(windows_file_identity(final_output), final_identity);
  EXPECT_EQ(read_bytes(final_output), "encoded-video");
  EXPECT_EQ(read_bytes(publication_path), original_publication);
  EXPECT_FALSE(fs::exists(state_path.parent_path()));
  recovered.shutdown();
}
#endif

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

  const auto clear_active = service.clear(first.job->id);
  EXPECT_FALSE(clear_active.ok);
  EXPECT_EQ(clear_active.code, offline_sbs::error_code_e::conflict);
  EXPECT_NE(clear_active.error.find("must be canceled"), std::string::npos);
  EXPECT_TRUE(service.get(first.job->id).ok);

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
    {"cache_budget_policy", "split"},
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

TEST(OfflineSbsJob, RestartKeepsACompletedJobThatStillNamesItsSceneAudit) {
  temporary_tree_t tree;
  const auto config = service_config(tree.path);
  const auto input = tree.path / "media" / "source.mkv";
  write_nonempty(input, "source");
  const auto runner = [](const offline_sbs::worker_context_t &context,
                         std::stop_token,
                         const offline_sbs::progress_callback_t &) {
    write_nonempty(*context.staging_output, "encoded-video");
    return offline_sbs::worker_outcome_t {
      .completed = true,
      .result = {
        {"schema", 1},
        {"job_id", context.job_id},
        {"status", "complete"},
      },
    };
  };
  std::string error;
  std::string id;
  {
    offline_sbs::job_service_t service {config, runner};
    ASSERT_TRUE(service.start(error)) << error;
    const auto created = service.create({
      .input_path = input,
      .operation = offline_sbs::operation_e::convert,
      .output_name = "converted.mkv",
    });
    ASSERT_TRUE(created.ok) << created.error;
    id = created.job->id;
    ASSERT_EQ(
      wait_for_terminal(service, id).state,
      offline_sbs::job_state_e::complete
    );
    service.shutdown();
  }

  // Rewrite the durable record and result directory into the shape a release with the
  // diagnostic scene audit and the retired split budget policy left behind for a completed
  // conversion.
  const auto result_directory = config.worker_root / "jobs" / id / "result";
  const auto audit_manifest = result_directory / "scene-audit.json";
  const auto audit_page = result_directory / "scene-audit-page-0.json";
  write_nonempty(audit_manifest, R"({"schema":4,"version":"whole-clip-scene-audit-v4"})");
  write_nonempty(audit_page, R"({"schema":1,"version":"whole-clip-scene-audit-page-v1"})");
  const auto state_path = config.state_root / "jobs" / id / "job.json";
  auto persisted = nlohmann::json::parse(std::ifstream(state_path));
  ASSERT_TRUE(persisted.at("worker_result").is_object());
  EXPECT_EQ(persisted.at("cache_budget_policy"), "fail");
  persisted["cache_budget_policy"] = "split";
  persisted["scene_decisions"] = nlohmann::json::array({
    {
      {"scene_id", 2},
      {"start_sequence", 7},
      {"end_sequence_exclusive", 13},
      {"boundary", {{"decision", "end_of_stream"}}},
    },
  });
  persisted["progress"]["scene_count"] = 2;
  persisted["progress"]["current_scene"] = nullptr;
  persisted["worker_result"]["schema"] = 3;
  persisted["worker_result"]["scene_count"] = 2;
  persisted["worker_result"]["scene_audit"] = audit_manifest.generic_string();
  persisted["worker_result"]["scene_audit_manifest_sha256"] = std::string(64, 'a');
  persisted["worker_result"]["scene_audit_sha256"] = std::string(64, 'a');
  persisted["worker_result"]["scene_decisions_contract"] =
    audit_manifest.generic_string();
  write_nonempty(state_path, persisted.dump(2));

  offline_sbs::job_service_t recovered {config, runner};
  ASSERT_TRUE(recovered.start(error)) << error;
  const auto snapshot = recovered.get(id);
  ASSERT_TRUE(snapshot.ok) << snapshot.error;
  ASSERT_TRUE(snapshot.job);
  EXPECT_EQ(snapshot.job->state, offline_sbs::job_state_e::complete);
  EXPECT_EQ(snapshot.job->progress.phase, "complete");
  ASSERT_TRUE(snapshot.job->output_path);
  EXPECT_TRUE(fs::is_regular_file(*snapshot.job->output_path));
  EXPECT_EQ(snapshot.job->worker_result["status"], "complete");
  EXPECT_TRUE(snapshot.job->worker_result.value("published", false));
  for (const auto retired : {
         "scene_count",
         "scene_audit",
         "scene_audit_manifest_sha256",
         "scene_audit_sha256",
         "scene_decisions_contract",
       }) {
    EXPECT_FALSE(snapshot.job->worker_result.contains(retired)) << retired;
  }
  const auto served = snapshot.job->json();
  EXPECT_EQ(served.at("cache_budget_policy"), "fail");
  EXPECT_FALSE(served.contains("scene_decisions"));
  EXPECT_FALSE(served.at("progress").contains("scene_count"));
  EXPECT_FALSE(served.at("progress").contains("current_scene"));
  ASSERT_EQ(recovered.list().size(), 1u);
  // The retained audit files are user data owned by the job: untouched until it is cleared.
  EXPECT_TRUE(fs::is_regular_file(audit_manifest));
  EXPECT_TRUE(fs::is_regular_file(audit_page));
  EXPECT_TRUE(fs::is_directory(config.state_root / "jobs" / id));

  const auto cleared = recovered.clear(id);
  EXPECT_TRUE(cleared.ok) << cleared.error;
  EXPECT_FALSE(recovered.get(id).ok);
  EXPECT_FALSE(fs::exists(audit_manifest));
  EXPECT_TRUE(fs::is_regular_file(input.parent_path() / "converted.mkv"));
  recovered.shutdown();
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
  const auto long_directory = "zULU-" + std::string(64, 'D') + "-dir";
  const auto long_alpha = "common-" + std::string(64, 'q') + "-alpha.mkv";
  const auto long_bravo = "CoMmOn-" + std::string(64, 'Q') + "-Bravo.mkv";

  struct expected_entry_t {
    fs::path leaf;
    std::string name;
    bool directory;
  };

  const std::array expected_order {
    expected_entry_t {"alpha-dir", "alpha-dir", true},
    expected_entry_t {"Bravo-dir", "Bravo-dir", true},
    expected_entry_t {long_directory, long_directory, true},
    expected_entry_t {"alpha-video.mkv", "alpha-video.mkv", false},
    expected_entry_t {"Bravo-video.mkv", "Bravo-video.mkv", false},
    expected_entry_t {long_alpha, long_alpha, false},
    expected_entry_t {long_bravo, long_bravo, false},
    // Native UTF-16 puts the surrogate pair for U+10000 before U+E000. UTF-8 and
    // Unicode scalar ordering put this pair in the opposite order.
    expected_entry_t {fs::path {u8"\U00010000-video.mkv"}, "\xf0\x90\x80\x80-video.mkv", false},
    expected_entry_t {fs::path {u8"\ue000-video.mkv"}, "\xee\x80\x80-video.mkv", false},
  };
  for (auto entry = expected_order.rbegin(); entry != expected_order.rend(); ++entry) {
    if (entry->directory) {
      fs::create_directories(media / entry->leaf);
    } else {
      write_nonempty(media / entry->leaf, "video");
    }
  }
  auto expected_files = nlohmann::json::array();
  auto expected_directories = nlohmann::json::array();
  const auto canonical_media = fs::canonical(media);
  for (const auto &entry : expected_order) {
    const auto path = (canonical_media / entry.leaf).generic_u8string();
    nlohmann::json value {
      {"name", entry.name},
      {"path", std::string {path.begin(), path.end()}},
      {"type", entry.directory ? "directory" : "file"},
      {"selectable", !entry.directory},
    };
    expected_files.push_back(value);
    if (entry.directory) {
      value["selectable"] = true;
      expected_directories.push_back(value);
    }
  }

  offline_sbs::job_service_t service {
    service_config(tree.path),
    [](const offline_sbs::worker_context_t &, std::stop_token, const offline_sbs::progress_callback_t &) {
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
  EXPECT_EQ(*files.path, canonical_media);
  EXPECT_FALSE(files.truncated);
  ASSERT_TRUE(files.entries.is_array());
  EXPECT_EQ(files.entries, expected_files);
  const auto wire = nlohmann::json::parse(files.json().dump());
  EXPECT_TRUE(wire.at("status").get<bool>());
  EXPECT_FALSE(wire.at("truncated").get<bool>());
  EXPECT_EQ(wire.at("entries"), expected_files);

  const auto directories = service.browse({
    .path = media,
    .type = offline_sbs::browse_type_e::directory,
  });
  ASSERT_TRUE(directories.ok) << directories.error;
  EXPECT_EQ(directories.entries, expected_directories);

  // A manually entered existing file opens its containing directory instead
  // of producing a dead-end "not a directory" response.
  const auto from_file = service.browse({
    .path = media / "Bravo-video.mkv",
    .type = offline_sbs::browse_type_e::file,
  });
  ASSERT_TRUE(from_file.ok) << from_file.error;
  ASSERT_TRUE(from_file.path);
  EXPECT_EQ(*from_file.path, canonical_media);
  EXPECT_EQ(from_file.entries, expected_files);

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
  std::string previous_drive;
  for (const auto &drive : drives.entries) {
    EXPECT_EQ(drive.at("type"), "directory");
    EXPECT_FALSE(drive.at("selectable").get<bool>());
    const auto name = drive.at("name").get<std::string>();
    if (!previous_drive.empty()) {
      EXPECT_LT(previous_drive, name);
    }
    previous_drive = name;
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
    [](const offline_sbs::worker_context_t &, std::stop_token, const offline_sbs::progress_callback_t &) {
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
  ASSERT_FALSE(reply.entries.empty());
  EXPECT_LT(reply.entries.size(), 1800u);
  const auto wire = reply.json().dump();
  EXPECT_LE(wire.size(), 512ull * 1024ull);
  EXPECT_EQ(nlohmann::json::parse(wire).at("entries"), reply.entries);
  std::string previous_name;
  for (const auto &entry : reply.entries) {
    const auto name = entry.at("name").get<std::string>();
    SCOPED_TRACE(name);
    const auto index = std::stoi(name);
    EXPECT_GE(index, 0);
    EXPECT_LT(index, 1800);
    EXPECT_EQ(name, std::to_string(index) + "-" + std::string(120, 'x') + ".mkv");
    EXPECT_EQ(entry.at("type"), "file");
    EXPECT_TRUE(entry.at("selectable").get<bool>());
    if (!previous_name.empty()) {
      EXPECT_LT(previous_name, name);
    }
    previous_name = name;
  }
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
  const auto clear = service.clear(failed.job->id);
  EXPECT_FALSE(clear.ok);
  EXPECT_EQ(clear.code, offline_sbs::error_code_e::io_error);
  EXPECT_NE(clear.error.find(".sunshine3d-*.part*"), std::string::npos);
  EXPECT_TRUE(fs::is_regular_file(staging));
  EXPECT_TRUE(service.get(failed.job->id).ok);
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

TEST(OfflineSbsJob, ActualWorkerProgressFilesReachTheServiceWithoutRegressingItsSnapshot) {
  temporary_tree_t tree;
  const auto input = tree.path / "source.mkv";
  write_nonempty(input, "source");
  offline_sbs::media_contract_t media;
  media.time_base = {1, 1000};
  media.frames = {{1, 1000, 40}, {2, 1040, 60}, {3, 1100, 50}};
  auto inconsistent_total = media;
  inconsistent_total.frames.push_back({4, 1150, 50});
  std::promise<void> observed_promise, release_promise;
  auto observed = observed_promise.get_future();
  const auto release = release_promise.get_future().share();
  std::atomic_bool released {false};
  const auto release_worker = [&]() {
    if (!released.exchange(true)) {
      release_promise.set_value();
    }
  };
  offline_sbs::job_service_t service {
    service_config(tree.path),
    [&](const offline_sbs::worker_context_t &context, const std::stop_token stop, const offline_sbs::progress_callback_t &progress) {
      offline_sbs::worker_progress_test_publisher_t publisher {context.worker_progress, context.job_id};
      const auto start = std::chrono::steady_clock::time_point {};
      publisher.publish("evaluate", 0, &media, start);
      progress(offline_sbs::read_worker_progress_for_test(context));
      publisher.publish("evaluate", 2, &media, start + 200ms);
      progress(offline_sbs::read_worker_progress_for_test(context));
      // The actual manager rejects both a regressed count and a changed source total.
      publisher.publish("evaluate", 1, &media, start + 400ms);
      progress(offline_sbs::read_worker_progress_for_test(context));
      publisher.publish("evaluate", 2, &inconsistent_total, start + 600ms);
      progress(offline_sbs::read_worker_progress_for_test(context));
      observed_promise.set_value();
      if (release.wait_for(3s) != std::future_status::ready || stop.stop_requested()) {
        return offline_sbs::worker_outcome_t {.error = "progress fixture was stopped before completion"};
      }
      publisher.publish("complete", 3, &media, start + 800ms);
      progress(offline_sbs::read_worker_progress_for_test(context));
      return offline_sbs::worker_outcome_t {
        .completed = true,
        .result = {{"schema", 1}, {"job_id", context.job_id}, {"status", "complete"}},
      };
    }
  };
  // This guard runs before the service joins its worker, including after a fatal assertion.
  auto release_guard = util::fail_guard(release_worker);
  std::string error;
  ASSERT_TRUE(service.start(error)) << error;
  const auto created = service.create({
    .input_path = input,
    .operation = offline_sbs::operation_e::evaluate,
  });
  ASSERT_TRUE(created.ok) << created.error;
  ASSERT_EQ(observed.wait_for(3s), std::future_status::ready);
  const auto active = service.get(created.job->id);
  ASSERT_TRUE(active.ok);
  ASSERT_TRUE(active.job);
  EXPECT_EQ(active.job->state, offline_sbs::job_state_e::running);
  EXPECT_EQ(active.job->progress.phase, "evaluate");
  EXPECT_EQ(active.job->progress.processed_frames, 2u);
  EXPECT_EQ(active.job->progress.total_frames, 3u);
  EXPECT_EQ(active.job->progress.source_time_seconds, 0.04);
  EXPECT_EQ(active.job->progress.source_duration_seconds, 0.15);
  release_worker();
  const auto terminal = wait_for_terminal(service, created.job->id);
  EXPECT_EQ(terminal.state, offline_sbs::job_state_e::complete);
  EXPECT_EQ(terminal.progress.phase, "complete");
  EXPECT_EQ(terminal.progress.processed_frames, 3u);
  EXPECT_EQ(terminal.progress.total_frames, 3u);
  EXPECT_EQ(terminal.progress.source_time_seconds, 0.1);
  EXPECT_EQ(terminal.progress.source_duration_seconds, 0.15);
  expect_persisted_snapshot(tree.path / "state" / "jobs" / terminal.id / "job.json", terminal);
  service.shutdown();
}
